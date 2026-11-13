// `motionforge keypose`: key-pose drawings (landmarks JSON from
// python/keypose/landmarks.py) -> a timed clip on a rigged GLB.

use motion_core::animate::{round, write_animation};
use motion_core::glb::Document;
use motion_core::helpers;
use motion_core::json::{emit_pretty, Json};
use motion_core::keypose::fit::{fit_sheet, restance, FitParams, KeyFit, RigInfo};
use motion_core::keypose::timing::{build, Key, Role, TimingParams};
use motion_core::keypose::{fuse, lm, order_legs, parse_landmarks, FuseParams, Fused, ViewMap};
use motion_core::math::Vec3;
use motion_core::rig::{Animation, Trs};

use crate::{parse_f64, parse_usize, read_file, Cmd};

type CmdResult = Result<String, (i32, String)>;

fn side_of(v: &str) -> Result<Option<usize>, (i32, String)> {
    match v {
        "L" | "l" => Ok(Some(0)),
        "R" | "r" => Ok(Some(1)),
        "none" => Ok(None),
        o => Err((1, format!("--sword must be R, L or none, got '{o}'"))),
    }
}

fn num(v: f64, places: i32) -> Json {
    Json::num(round(v, places))
}

fn v3(v: Vec3) -> Json {
    Json::Arr(vec![num(v.x, 4), num(v.y, 4), num(v.z, 4)])
}

/// Fitted joints drawn back into a view: `(landmark, rig bone)` pairs.
const OVERLAY: [(usize, &str); 12] = [
    (lm::SH_L, "DEF-upper_arm.L"),
    (lm::SH_R, "DEF-upper_arm.R"),
    (lm::EL_L, "DEF-forearm.L"),
    (lm::EL_R, "DEF-forearm.R"),
    (lm::WR_L, "DEF-hand.L"),
    (lm::WR_R, "DEF-hand.R"),
    (lm::HIP_L, "DEF-thigh.L"),
    (lm::HIP_R, "DEF-thigh.R"),
    (lm::KN_L, "DEF-shin.L"),
    (lm::KN_R, "DEF-shin.R"),
    (lm::AN_L, "DEF-foot.L"),
    (lm::AN_R, "DEF-foot.R"),
];

/// Per view: the fitted joints in the drawing's pixels and the mean
/// distance to MediaPipe's landmarks as a share of the figure's height.
fn overlay(
    info: &RigInfo,
    fit: &KeyFit,
    map: &ViewMap,
    scale: f64,
    img: &[[f64; 3]],
) -> (Json, f64) {
    let w = info.rig.world(&fit.locals);
    let to_px = |p: Vec3| {
        let c = info
            .from_rig(Vec3::new(p.x, p.y - info.ground, p.z))
            .scale(1.0 / scale);
        map.to_px(c)
    };
    let mut pts = Vec::new();
    let mut err = 0.0;
    let mut n = 0.0;
    let ys: Vec<f64> = img.iter().map(|p| p[1]).collect();
    let height = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - ys.iter().cloned().fold(f64::INFINITY, f64::min);
    for (l, bone) in OVERLAY {
        let Some(node) = info.rig.find(bone) else {
            continue;
        };
        let (u, v) = to_px(w[node].t);
        pts.push(Json::Arr(vec![Json::num(l as f64), num(u, 1), num(v, 1)]));
        if img[l][2] > 0.5 {
            err += ((u - img[l][0]).powi(2) + (v - img[l][1]).powi(2)).sqrt();
            n += 1.0;
        }
    }
    let mut extra = Vec::new();
    for s in 0..2 {
        if let Some(sk) = info.socket[s] {
            let p = w[sk].t;
            let tip = p + w[sk]
                .rotation()
                .rotate_vec(Vec3::new(0.0, 1.0, 0.0))
                .scale(0.9 * info.k);
            let (a, b) = to_px(p);
            let (c, d) = to_px(tip);
            extra.push(Json::Arr(vec![num(a, 1), num(b, 1), num(c, 1), num(d, 1)]));
        }
    }
    let share = if n > 0.0 && height > 0.0 {
        err / n / height
    } else {
        0.0
    };
    (
        Json::obj(vec![
            ("joints", Json::Arr(pts)),
            ("blades", Json::Arr(extra)),
            ("mean_err_share", num(share, 4)),
        ]),
        share,
    )
}

fn roles_for(n: usize, spec: Option<&str>) -> Result<Vec<Role>, (i32, String)> {
    if let Some(s) = spec {
        let r: Option<Vec<Role>> = s.split(',').map(|w| Role::parse(w.trim())).collect();
        let r = r.ok_or_else(|| (1, format!("--roles: unknown role in '{s}'")))?;
        if r.len() != n {
            return Err((
                1,
                format!("--roles names {} keys, the drawings have {n}", r.len()),
            ));
        }
        return Ok(r);
    }
    Ok(match n {
        1 => vec![Role::Pose],
        2 => vec![Role::Load, Role::Follow],
        3 => vec![Role::Load, Role::Strike, Role::Follow],
        _ => {
            let mut v = vec![Role::Pose; n];
            v[n - 3] = Role::Load;
            v[n - 2] = Role::Strike;
            v[n - 1] = Role::Follow;
            v
        }
    })
}

pub fn cmd_keypose(cmd: &Cmd) -> CmdResult {
    let rig_path = cmd.required("rig")?;
    let lm_path = cmd.required("landmarks")?;
    let output = cmd.required("output")?;
    let lms = parse_landmarks(&read_file(lm_path)?).map_err(|e| (2, e))?;
    let name = cmd.get("name").unwrap_or(&lms.name).to_string();
    let doc = Document::read(rig_path).map_err(|e| (2, e))?;
    let info = RigInfo::new(&doc).map_err(|e| (2, e))?;

    let fuse_params = FuseParams {
        depth_weight: parse_f64(cmd, "depth-weight", 0.2)?,
        side_turn_deg: match cmd.get("side-turn") {
            Some("auto") => None,
            _ => Some(parse_f64(cmd, "side-turn", 0.0)?),
        },
        ..FuseParams::default()
    };
    let mut fp = FitParams {
        sword_hand: side_of(cmd.get("sword").unwrap_or("R"))?,
        ..FitParams::default()
    };
    fp.two_hands = match cmd.get("hands").unwrap_or("auto") {
        "auto" => None,
        "one" => Some(false),
        "two" => Some(true),
        o => return Err((1, format!("--hands must be one, two or auto, got '{o}'"))),
    };
    let d = TimingParams::default();
    let tp = TimingParams {
        fps: parse_f64(cmd, "fps", d.fps)?,
        windup_frames: parse_usize(cmd, "windup-frames", d.windup_frames)?,
        hold_load: parse_usize(cmd, "hold-load", d.hold_load)?,
        strike_frames: parse_usize(cmd, "strike-frames", d.strike_frames)?,
        follow_frames: parse_usize(cmd, "follow-frames", d.follow_frames)?,
        hold_follow: parse_usize(cmd, "hold-follow", d.hold_follow)?,
        recover_frames: parse_usize(cmd, "recover-frames", d.recover_frames)?,
        overshoot: parse_f64(cmd, "overshoot", d.overshoot)?,
        anticipation: parse_f64(cmd, "anticipation", d.anticipation)?,
        creep: parse_f64(cmd, "creep", d.creep)?,
        ..d
    };

    let mut fused: Fused = fuse(&lms, &fuse_params);
    let roles = roles_for(fused.poses.len(), cmd.get("roles"))?;
    // Feet: `hold` (default) stands every key on the strike's feet (one
    // step in, one step out); `order` gives every key the strike's leg
    // order and keeps each key's own drawn feet otherwise (a step per key
    // where they differ); `drawn` takes the feet as read.
    let feet_mode = cmd.get("feet").unwrap_or("hold");
    let strike_key = roles.iter().position(|r| *r == Role::Strike).unwrap_or(0);
    let reordered = match feet_mode {
        "order" | "hold" => order_legs(&mut fused, strike_key),
        "drawn" => Vec::new(),
        o => return Err((1, format!("--feet must be order, hold or drawn, got '{o}'"))),
    };
    let (fits, scale, two) = fit_sheet(&info, &fused.poses, &fp);

    // The idle the clip starts from and returns to.
    let anims = Animation::load_all(&doc).map_err(|e| (2, e))?;
    let rest_pose = |which: &str| -> Result<Option<Vec<Option<Trs>>>, (i32, String)> {
        let clip = cmd.get(which).unwrap_or("idle");
        if clip == "none" {
            return Ok(None);
        }
        match anims.iter().find(|a| a.name == clip) {
            Some(a) => Ok(Some(a.sample(&info.rig, 0.0))),
            None if cmd.get(which).is_some() => Err((2, format!("no clip '{clip}' in the rig"))),
            None => Ok(None),
        }
    };
    let start = rest_pose("start-clip")?;
    let end = rest_pose("end-clip")?;
    let stance = (feet_mode == "hold").then_some(strike_key);
    // The report shows the fit to the drawings; the clip uses the staged keys.
    let mut staged = fits.clone();
    if let Some(si) = stance {
        let st = staged[si].locals.clone();
        for (k, f) in staged.iter_mut().enumerate() {
            if k != si {
                restance(&info, &mut f.locals, &st);
            }
            f.planted = [true, true];
        }
    }
    let mut keys = Vec::new();
    if let Some(l) = &start {
        keys.push(Key {
            locals: l.clone(),
            role: Role::Rest,
            planted: [true, true],
            drawn: false,
            off_rel: None,
            grip_gap: 0.0,
        });
    }
    for (k, f) in staged.iter().enumerate() {
        keys.push(Key {
            locals: f.locals.clone(),
            role: roles[k],
            planted: f.planted,
            drawn: true,
            off_rel: f.off_rel,
            grip_gap: f.grip_gap,
        });
    }
    if let Some(l) = &end {
        keys.push(Key {
            locals: l.clone(),
            role: Role::Rest,
            planted: [true, true],
            drawn: false,
            off_rel: None,
            grip_gap: 0.0,
        });
    }
    let (mut frames, trep) = build(&info, &keys, &tp, &fp, two);
    for l in frames.iter_mut() {
        helpers::drive(&info.rig, l);
    }

    // Write: every mapped bone and helper, the hips' translation.
    let mut nodes: Vec<usize> = info.map.pairs.iter().map(|(n, _)| *n).collect();
    for (h, _) in helpers::present(&info.rig) {
        if !nodes.contains(&h) {
            nodes.push(h);
        }
    }
    let times: Vec<f64> = (0..frames.len()).map(|f| f as f64 / tp.fps).collect();
    let tracks: Vec<Vec<Trs>> = nodes
        .iter()
        .map(|&n| {
            frames
                .iter()
                .map(|l| l[n].unwrap_or(info.rig.rest[n]))
                .collect()
        })
        .collect();
    let mut out = doc.clone();
    if cmd.flag("only") {
        out.array_mut("animations").clear();
    }
    let existing: Vec<String> = out
        .array("animations")
        .iter()
        .filter_map(|a| a.get("name").and_then(Json::as_str).map(str::to_string))
        .collect();
    if existing.contains(&name) {
        return Err((2, format!("the rig already has a clip named '{name}'")));
    }
    write_animation(&mut out, &name, &times, &nodes, &tracks, Some(info.hips))
        .map_err(|e| (3, e))?;
    if cmd.flag("key-clips") {
        for (k, f) in fits.iter().enumerate() {
            let mut l = f.locals.clone();
            helpers::drive(&info.rig, &mut l);
            let tr: Vec<Vec<Trs>> = nodes
                .iter()
                .map(|&n| vec![l[n].unwrap_or(info.rig.rest[n])])
                .collect();
            write_animation(
                &mut out,
                &format!("{name}_key{k}"),
                &[0.0],
                &nodes,
                &tr,
                Some(info.hips),
            )
            .map_err(|e| (3, e))?;
        }
    }
    out.compact();
    out.write(output).map_err(|e| (3, e))?;

    // Report.
    let mut key_rows = Vec::new();
    let mut text = String::from("=== motionforge keypose ===\n");
    text.push_str(&format!(
        "{}: {} keys, {} frames at {} fps ({:.2} s); scale {:.3} rig m per drawing m; {}\n",
        name,
        fits.len(),
        trep.frames,
        tp.fps,
        trep.frames as f64 / tp.fps,
        scale,
        if two {
            "two hands on the grip"
        } else {
            "one hand"
        }
    ));
    for (k, f) in fits.iter().enumerate() {
        let pose = &fused.poses[k];
        let (fm, sm) = &fused.maps[k];
        let mut views = Vec::new();
        let mut shares = Vec::new();
        for (view, map, sheet) in [("front", fm, &lms.front), ("side", sm, &lms.side)] {
            if let (Some(m), Some(sh)) = (map, sheet) {
                let img = match (view, &fused.side_img[k]) {
                    ("side", Some(i)) => i,
                    _ => &sh.figures[k].img,
                };
                let (j, share) = overlay(&info, f, m, scale, img);
                shares.push(format!("{view} {:.1}%", share * 100.0));
                views.push((view, j));
            }
        }
        let limbs: Vec<f64> = f
            .errors
            .iter()
            .filter(|(n, _)| n.contains("arm") || n.contains("thigh") || n.contains("shin"))
            .map(|(_, e)| *e)
            .collect();
        let limb_mean = limbs.iter().sum::<f64>() / limbs.len().max(1) as f64;
        let limb_max = limbs.iter().cloned().fold(0.0, f64::max);
        text.push_str(&format!(
            "key {k} {:<6} views {} (side turned {:+.1} deg{})  limbs {:.1}/{:.1} deg mean/max  torso {:.1}  blade {} ({})  grip {}  feet {}{}  drawn joints off by {}\n",
            roles[k].name(),
            pose.views,
            fused.side_turn_deg[k],
            if fused.side_swapped[k].is_empty() {
                String::new()
            } else {
                format!(", its {} named the other way round", fused.side_swapped[k].join(" and "))
            },
            limb_mean,
            limb_max,
            f.errors.iter().find(|(n, _)| n == "torso").map(|e| e.1).unwrap_or(0.0),
            f.blade_err_deg.map(|e| format!("{e:.1} deg")).unwrap_or("-".into()),
            pose.blade_how,
            f.grip_err_m.map(|e| format!("{:.1} cm", e * 100.0)).unwrap_or("-".into()),
            if f.planted[0] { "L" } else { "-" },
            if f.planted[1] { "R" } else { "-" },
            shares.join(", ")
        ));
        if !f.capped.is_empty() {
            text.push_str(&format!(
                "      capped: {}\n",
                f.capped
                    .iter()
                    .map(|(n, d)| format!("{n} {d:.0} deg"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        key_rows.push(Json::obj(vec![
            ("role", Json::str(roles[k].name())),
            ("views", Json::num(pose.views as f64)),
            ("side_turn_deg", num(fused.side_turn_deg[k], 1)),
            (
                "side_swapped",
                Json::Arr(fused.side_swapped[k].iter().map(|c| Json::str(c)).collect()),
            ),
            ("blade_how", Json::str(&pose.blade_how)),
            (
                "blade_err_deg",
                f.blade_err_deg.map(|e| num(e, 2)).unwrap_or(Json::Null),
            ),
            (
                "grip_err_m",
                f.grip_err_m.map(|e| num(e, 4)).unwrap_or(Json::Null),
            ),
            (
                "planted",
                Json::Arr(f.planted.iter().map(|p| Json::Bool(*p)).collect()),
            ),
            (
                "errors_deg",
                Json::Obj(
                    f.errors
                        .iter()
                        .map(|(n, e)| (n.clone(), num(*e, 2)))
                        .collect(),
                ),
            ),
            (
                "capped_deg",
                Json::Obj(
                    f.capped
                        .iter()
                        .map(|(n, e)| (n.clone(), num(*e, 2)))
                        .collect(),
                ),
            ),
            (
                "overlay",
                Json::Obj(views.into_iter().map(|(v, j)| (v.to_string(), j)).collect()),
            ),
            (
                "frame",
                Json::num(trep.key_frames[k + usize::from(start.is_some())] as f64),
            ),
        ]));
    }
    if !reordered.is_empty() {
        text.push_str(&format!(
            "feet: keys {:?} given the strike's leg order (front and back foot exchanged)\n",
            reordered
        ));
    }
    text.push_str(&format!(
        "timing: keys at frames {:?}; windup in the drawings: {}; counter added: {}; steps: {}; locked feet drawn up to {:.1} cm apart\n",
        trep.key_frames,
        if trep.windup_found { "yes" } else { "no" },
        if trep.counter_added { "yes" } else { "no" },
        if trep.steps.is_empty() {
            "none".to_string()
        } else {
            trep.steps
                .iter()
                .map(|(f, a, b, d)| format!("{f} {a}->{b} {:.0} cm", d * 100.0))
                .collect::<Vec<_>>()
                .join(", ")
        },
        trep.lock_slide_max * 100.0
    ));
    text.push_str(&format!("output: {output} (clip {name})\n"));
    if let Some(path) = cmd.get("report") {
        let j = Json::obj(vec![
            ("format", Json::str("motionforge-keypose-report")),
            ("clip", Json::str(&name)),
            ("scale", num(scale, 4)),
            ("two_hands", Json::Bool(two)),
            ("fps", Json::num(tp.fps)),
            ("frames", Json::num(trep.frames as f64)),
            (
                "key_frames",
                Json::Arr(
                    trep.key_frames
                        .iter()
                        .map(|f| Json::num(*f as f64))
                        .collect(),
                ),
            ),
            ("windup_found", Json::Bool(trep.windup_found)),
            ("counter_added", Json::Bool(trep.counter_added)),
            (
                "steps",
                Json::Arr(
                    trep.steps
                        .iter()
                        .map(|(f, a, b, d)| {
                            Json::obj(vec![
                                ("foot", Json::str(f)),
                                ("from", Json::num(*a as f64)),
                                ("to", Json::num(*b as f64)),
                                ("m", num(*d, 3)),
                            ])
                        })
                        .collect(),
                ),
            ),
            (
                "offsets",
                Json::Arr(trep.offsets.iter().map(|o| v3(*o)).collect()),
            ),
            ("keys", Json::Arr(key_rows)),
        ]);
        std::fs::write(path, emit_pretty(&j).map_err(|e| (3, e))?)
            .map_err(|e| (3, format!("cannot write {path}: {e}")))?;
    }
    Ok(text)
}
