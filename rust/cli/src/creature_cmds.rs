// Creature motion commands: profile from a rig, random poses and the
// exam motion, synthetic detections, the two-view solve, and scoring.
// Spec: docs/creature.md.

use crate::{parse_f64, parse_usize, read_file, write_file, Cmd};
use motion_core::clip::{emit_clip, parse_clip};
use motion_core::creature::compare::compare;
use motion_core::creature::detect::{project_clip, Detections};
use motion_core::creature::gait;
use motion_core::creature::glbanim::clip_to_glb;
use motion_core::creature::motion::{exam_motion, training_poses};
use motion_core::creature::profile::Creature;
use motion_core::creature::rigcheck::{from_glb, Options};
use motion_core::creature::solve::{solve, SolveParams};
use motion_core::glb::Document;
use motion_core::json::{emit_pretty, Json};
use motion_core::math::Vec3;

fn load_creature(cmd: &Cmd) -> Result<Creature, (i32, String)> {
    let text = read_file(cmd.required("creature")?)?;
    Creature::parse(&text).map_err(|e| (2, format!("creature: {}", e)))
}

fn parse_seed(cmd: &Cmd) -> Result<u64, (i32, String)> {
    match cmd.get("seed") {
        None => Ok(1),
        Some(v) => v.parse::<u64>().map_err(|_| {
            (
                1,
                format!("--seed must be a non-negative integer, got '{}'", v),
            )
        }),
    }
}

fn parse_forward(v: Option<&str>) -> Result<Option<Vec3>, (i32, String)> {
    Ok(match v.unwrap_or("auto") {
        "auto" => None,
        "+x" => Some(Vec3::new(1.0, 0.0, 0.0)),
        "-x" => Some(Vec3::new(-1.0, 0.0, 0.0)),
        "+y" => Some(Vec3::new(0.0, 1.0, 0.0)),
        "-y" => Some(Vec3::new(0.0, -1.0, 0.0)),
        other => {
            return Err((
                1,
                format!(
                    "--forward must be auto|+x|-x|+y|-y (Blender Z-up axes), got '{}'",
                    other
                ),
            ))
        }
    })
}

/// creature-profile --rig IN.glb --output creature.json [--clean-glb OUT.glb]
/// [--forward auto|+x|-x|+y|-y] [--name N] [--view-size 512] [--report r.json]
pub fn cmd_profile(cmd: &Cmd) -> Result<String, (i32, String)> {
    let rig_path = cmd.required("rig")?;
    let output = cmd.required("output")?.to_string();
    let doc = Document::read(rig_path).map_err(|e| (2, format!("rig: {}", e)))?;
    let name = cmd.get("name").map(str::to_string).unwrap_or_else(|| {
        std::path::Path::new(rig_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("creature")
            .to_string()
    });
    let opts = Options {
        forward: parse_forward(cmd.get("forward"))?,
        view_size: parse_usize(cmd, "view-size", 512)?.max(16),
        name,
    };
    let built = from_glb(&doc, &opts).map_err(|e| (3, e))?;
    write_file(&output, &built.creature.emit().map_err(|e| (3, e))?)?;
    if let Some(p) = cmd.get("clean-glb") {
        built.glb.write(p).map_err(|e| (3, e))?;
    }
    let mut r = String::new();
    r.push_str("=== motionforge creature-profile ===\n");
    let c = &built.creature;
    let k = &built.checks;
    r.push_str(&format!(
        "bones: {} (from {} joints; {} fingers folded, {} out-of-body cut)\n",
        c.skeleton.len(),
        k.joints_in,
        k.fingers_folded,
        k.out_of_body_cut
    ));
    r.push_str(&format!(
        "legs on the ground: {} of {} have a bone; feet: {}\n",
        k.legs_with_bone,
        k.legs_found,
        c.feet
            .iter()
            .map(|&f| c.skeleton.bones[f].name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    ));
    r.push_str(&format!(
        "symmetry: {} of {} side joints have a mirror\n",
        k.side_matched, k.side_joints
    ));
    r.push_str(&format!(
        "joints left outside the body: {}\n",
        k.out_of_body_left
    ));
    r.push_str(&format!(
        "forward: ({:.0}, {:.0}, 0) [{}]\n",
        c.forward.x, c.forward.y, k.forward_reason
    ));
    r.push_str(&format!(
        "height: {:.3} m, ground z {:.3}\n",
        c.height, c.ground
    ));
    for v in &c.views {
        r.push_str(&format!(
            "view {}: at ({:.2}, {:.2}, {:.2}) looking at ({:.2}, {:.2}, {:.2}), fov {:.0} deg, {}x{}\n",
            v.name,
            v.position.x,
            v.position.y,
            v.position.z,
            v.look_at.x,
            v.look_at.y,
            v.look_at.z,
            v.fov_y_deg,
            v.width,
            v.height
        ));
    }
    if !built.pruned.is_empty() {
        r.push_str(&format!("pruned: {}\n", built.pruned.join(", ")));
    }
    r.push_str(&format!("score: {:.4}\n", k.score()));
    if let Some(p) = cmd.get("report") {
        let mut j = k.to_json();
        j.set("name", Json::str(&c.name));
        j.set(
            "pruned",
            Json::Arr(built.pruned.iter().map(|s| Json::str(s)).collect()),
        );
        j.set(
            "feet",
            Json::Arr(
                c.feet
                    .iter()
                    .map(|&f| Json::str(&c.skeleton.bones[f].name))
                    .collect(),
            ),
        );
        write_file(p, &emit_pretty(&j).map_err(|e| (3, e))?)?;
        r.push_str(&format!("report: {}\n", p));
    }
    if let Some(p) = cmd.get("clean-glb") {
        r.push_str(&format!("clean glb: {}\n", p));
    }
    r.push_str(&format!("output: {}\n", output));
    Ok(r)
}

/// creature-poses --creature c.json --output clip.json [--count 3000] [--seed 1] [--fast-ratio 0.5]
pub fn cmd_poses(cmd: &Cmd) -> Result<String, (i32, String)> {
    let c = load_creature(cmd)?;
    let output = cmd.required("output")?.to_string();
    let count = parse_usize(cmd, "count", 3000)?;
    if count == 0 {
        return Err((1, "--count must be at least 1".into()));
    }
    let seed = parse_seed(cmd)?;
    let fast = parse_f64(cmd, "fast-ratio", 0.5)?;
    if !(0.0..=1.0).contains(&fast) {
        return Err((1, "--fast-ratio must be in [0, 1]".into()));
    }
    let clip = training_poses(&c, count, seed, fast);
    write_file(&output, &emit_clip(&clip).map_err(|e| (3, e))?)?;
    let mut r = String::new();
    r.push_str("=== motionforge creature-poses ===\n");
    r.push_str(&format!(
        "samples: {} (frames 3k, 3k+1, 3k+2 = before, pose, after; render 3k+1 with a one-frame shutter)\n",
        count
    ));
    r.push_str(&format!("fast ratio: {:.2}, seed: {}\n", fast, seed));
    r.push_str(&format!("output: {}\n", output));
    Ok(r)
}

/// creature-motion --creature c.json --output clip.json [--frames 48] [--fps 24] [--seed 1] [--amplitude 0.8]
pub fn cmd_motion(cmd: &Cmd) -> Result<String, (i32, String)> {
    let c = load_creature(cmd)?;
    let output = cmd.required("output")?.to_string();
    let frames = parse_usize(cmd, "frames", 48)?;
    if frames == 0 {
        return Err((1, "--frames must be at least 1".into()));
    }
    let fps = parse_f64(cmd, "fps", 24.0)?;
    if !(fps > 0.0 && fps <= 1e4) {
        return Err((1, "--fps must be in (0, 1e4]".into()));
    }
    let seed = parse_seed(cmd)?;
    let amplitude = parse_f64(cmd, "amplitude", 0.8)?;
    if !(0.0..=1.0).contains(&amplitude) {
        return Err((1, "--amplitude must be in [0, 1]".into()));
    }
    let clip = exam_motion(&c, frames, fps, seed, amplitude);
    write_file(&output, &emit_clip(&clip).map_err(|e| (3, e))?)?;
    let mut r = String::new();
    r.push_str("=== motionforge creature-motion ===\n");
    r.push_str(&format!(
        "frames: {} at {} fps, seed {}, amplitude {:.2} of the limits\n",
        frames, fps, seed, amplitude
    ));
    r.push_str(&format!("output: {}\n", output));
    Ok(r)
}

/// creature-project --creature c.json --input clip.json --output det.json [--noise-px 0] [--drop 0] [--seed 1]
pub fn cmd_project(cmd: &Cmd) -> Result<String, (i32, String)> {
    let c = load_creature(cmd)?;
    let clip = parse_clip(&read_file(cmd.required("input")?)?)
        .map_err(|e| (2, format!("input: {}", e)))?;
    let output = cmd.required("output")?.to_string();
    let noise = parse_f64(cmd, "noise-px", 0.0)?;
    let drop = parse_f64(cmd, "drop", 0.0)?;
    if noise < 0.0 || !(0.0..=1.0).contains(&drop) {
        return Err((1, "--noise-px must be >= 0 and --drop in [0, 1]".into()));
    }
    let seed = parse_seed(cmd)?;
    if clip.skeleton.len() != c.skeleton.len() {
        return Err((2, "input: clip is not on the creature's skeleton".into()));
    }
    let dets = project_clip(&c, &clip, noise, drop, seed);
    write_file(&output, &dets.emit().map_err(|e| (3, e))?)?;
    let mut r = String::new();
    r.push_str("=== motionforge creature-project ===\n");
    r.push_str(&format!(
        "frames: {}, views: {}, joints: {}, noise {:.2} px, drop {:.2}, seed {}\n",
        dets.frames.len(),
        dets.views.join(", "),
        dets.joints.len(),
        noise,
        drop,
        seed
    ));
    r.push_str(&format!("output: {}\n", output));
    Ok(r)
}

/// creature-solve --creature c.json --detections det.json --output clip.json
/// [--glb-in rig.glb --glb-out anim.glb [--clip-name NAME]] [--no-pin] [--force]
/// [--smooth-sigma 1] [--quat-sigma 0.8] [--max-gap-ratio 0.08] [--max-motion-ratio 0.5]
/// [--min-conf 0.2] [--report r.json]
pub fn cmd_solve(cmd: &Cmd) -> Result<String, (i32, String)> {
    let c = load_creature(cmd)?;
    let dets = Detections::parse(&read_file(cmd.required("detections")?)?)
        .map_err(|e| (2, format!("detections: {}", e)))?;
    let output = cmd.required("output")?.to_string();
    let params = SolveParams {
        smooth_sigma: parse_f64(cmd, "smooth-sigma", 1.0)?.max(0.0),
        quat_sigma: parse_f64(cmd, "quat-sigma", 0.8)?.max(0.0),
        pin: !cmd.flag("no-pin"),
        max_gap_ratio: parse_f64(cmd, "max-gap-ratio", 0.08)?,
        max_motion_ratio: parse_f64(cmd, "max-motion-ratio", 0.5)?,
        min_conf: parse_f64(cmd, "min-conf", 0.2)?,
        force: cmd.flag("force"),
    };
    if params.max_gap_ratio <= 0.0 || params.max_motion_ratio <= 0.0 {
        return Err((
            1,
            "--max-gap-ratio and --max-motion-ratio must be > 0".into(),
        ));
    }
    let (glb_in, glb_out) = (cmd.get("glb-in"), cmd.get("glb-out"));
    if glb_in.is_some() != glb_out.is_some() {
        return Err((1, "--glb-in and --glb-out go together".into()));
    }
    let (clip, report) = solve(&c, &dets, &params).map_err(|e| (3, e))?;
    write_file(&output, &emit_clip(&clip).map_err(|e| (3, e))?)?;
    let mut r = String::new();
    r.push_str("=== motionforge creature-solve ===\n");
    r.push_str(&format!(
        "frames: {} at {} fps; joints seen: {} of {}\n",
        report.frames, dets.fps, report.joints_seen, report.joints_total
    ));
    r.push_str(&format!(
        "views agree: {} (median ray gap {:.2}% of height, motion ratio {:.3})\n",
        if report.consistent {
            "yes"
        } else {
            "NO (forced)"
        },
        report.median_gap_ratio * 100.0,
        report.motion_ratio
    ));
    r.push_str(&format!(
        "fit residual: {:.1} mm rms; limited bone-frames: {}\n",
        report.residual_mm, report.limited
    ));
    r.push_str(&format!(
        "feet: {} plants, max pin shift {:.1} mm\n",
        report.plants, report.max_pin_shift_mm
    ));
    if let (Some(gi), Some(go)) = (glb_in, glb_out) {
        let doc = Document::read(gi).map_err(|e| (2, format!("glb-in: {}", e)))?;
        let name = cmd.get("clip-name").unwrap_or("creature");
        let out = clip_to_glb(&doc, &clip, name).map_err(|e| (3, e))?;
        out.write(go).map_err(|e| (3, e))?;
        r.push_str(&format!("glb: {} (animation \"{}\")\n", go, name));
    }
    if let Some(p) = cmd.get("report") {
        write_file(p, &emit_pretty(&report.to_json()).map_err(|e| (3, e))?)?;
        r.push_str(&format!("report: {}\n", p));
    }
    r.push_str(&format!("output: {}\n", output));
    Ok(r)
}

/// creature-gait --creature c.json [--glb-in rig.glb --glb-out out.glb]
/// [--clips walk,trot,attack] [--frames 24] [--fps 24] [--rom rom.ron]
/// [--clips-json <dir>] (each clip also as <dir>/<name>.clip.json, for the renderer)
///
/// A quadruped's parts read from its profile; a walk and a lunge made on
/// its own skeleton and written into its GLB; its range of motion as a
/// weightforge pose set.
pub fn cmd_gait(cmd: &Cmd) -> Result<String, (i32, String)> {
    let c = load_creature(cmd)?;
    let parts = gait::parts(&c).map_err(|e| (3, e))?;
    let frames = parse_usize(cmd, "frames", 24)?;
    if frames < 2 {
        return Err((1, "--frames must be at least 2".into()));
    }
    let fps = parse_f64(cmd, "fps", 24.0)?;
    if !(fps > 0.0 && fps <= 1e4) {
        return Err((1, "--fps must be in (0, 1e4]".into()));
    }
    let name = |b: usize| c.skeleton.bones[b].name.as_str();
    let list = |bones: &[usize]| {
        bones
            .iter()
            .map(|&b| name(b))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let mut r = String::new();
    r.push_str("=== motionforge creature-gait ===\n");
    for leg in &parts.legs {
        r.push_str(&format!(
            "{} {} leg: {}\n",
            if leg.front { "front" } else { "hind" },
            if leg.left { "left" } else { "right" },
            list(&leg.bones)
        ));
    }
    r.push_str(&format!("spine: {}\n", list(&parts.spine)));
    r.push_str(&format!("neck: {}\n", list(&parts.neck)));
    r.push_str(&format!("tail: {}\n", list(&parts.tail)));
    if let Some(p) = cmd.get("rom") {
        write_file(p, &gait::rom_ron(&c, &parts))?;
        r.push_str(&format!("rom: {}\n", p));
    }
    match (cmd.get("glb-in"), cmd.get("glb-out")) {
        (Some(gi), Some(go)) => {
            let mut doc = Document::read(gi).map_err(|e| (2, format!("glb-in: {}", e)))?;
            let wanted = cmd.get("clips").unwrap_or("walk,trot,attack");
            for clip_name in wanted.split(',').filter(|s| !s.is_empty()) {
                let planted = |g: gait::Gait| {
                    let s = gait::stride(&c, &parts, g, fps);
                    let note = format!(
                        ", feet planted: move the body {:.3} m/s ({:.2} s a cycle, each foot down {:.0}% of it over {:.3} m)",
                        s.speed,
                        s.period,
                        s.duty * 100.0,
                        s.stance
                    );
                    (s.clip, note)
                };
                let (clip, note) = match clip_name {
                    "walk" => planted(gait::Gait::Walk),
                    "trot" => planted(gait::Gait::Trot),
                    "walk_spot" => (gait::walk_spot(&c, &parts, frames, fps), String::new()),
                    "attack" => (gait::attack(&c, &parts, frames, fps), String::new()),
                    other => {
                        return Err((
                            1,
                            format!(
                                "--clips: unknown clip '{}' (walk, trot, walk_spot, attack)",
                                other
                            ),
                        ))
                    }
                };
                if let Some(dir) = cmd.get("clips-json") {
                    let path = format!("{}/{}.clip.json", dir.trim_end_matches('/'), clip_name);
                    let text = motion_core::clip::emit_clip(&clip).map_err(|e| (3, e))?;
                    write_file(&path, &text)?;
                }
                doc = clip_to_glb(&doc, &clip, clip_name).map_err(|e| (3, e))?;
                r.push_str(&format!(
                    "clip {}: {} frames at {} fps{}\n",
                    clip_name,
                    clip.frames.len(),
                    fps,
                    note
                ));
            }
            doc.write(go).map_err(|e| (3, e))?;
            r.push_str(&format!("glb: {}\n", go));
        }
        (None, None) => {}
        _ => return Err((1, "--glb-in and --glb-out go together".into())),
    }
    Ok(r)
}

/// creature-compare --creature c.json --truth t.json --solved s.json [--output scores.json]
pub fn cmd_compare(cmd: &Cmd) -> Result<String, (i32, String)> {
    let c = load_creature(cmd)?;
    let truth = parse_clip(&read_file(cmd.required("truth")?)?)
        .map_err(|e| (2, format!("truth: {}", e)))?;
    let solved = parse_clip(&read_file(cmd.required("solved")?)?)
        .map_err(|e| (2, format!("solved: {}", e)))?;
    let scores = compare(&c, &truth, &solved).map_err(|e| (2, e))?;
    let mut r = String::new();
    r.push_str("=== motionforge creature-compare ===\n");
    r.push_str(&format!("frames compared: {}\n", scores.frames));
    r.push_str(&format!(
        "bone rotation error: mean {:.2} deg ({:.2} deg over bones with children), max {:.2} deg\n",
        scores.mean_deg, scores.mean_deg_inner, scores.max_deg
    ));
    r.push_str(&format!(
        "joint position error: mean {:.1} mm, max {:.1} mm\n",
        scores.mean_mm, scores.max_mm
    ));
    r.push_str(&format!(
        "foot slide: {:.1} mm per still frame ({} still foot-frames)\n",
        scores.foot_slide_mm, scores.still_frames
    ));
    for b in &scores.bones {
        r.push_str(&format!(
            "  {}: {:.2} deg (max {:.2}), {:.1} mm (max {:.1})\n",
            b.name, b.mean_deg, b.max_deg, b.mean_mm, b.max_mm
        ));
    }
    if let Some(p) = cmd.get("output") {
        write_file(p, &emit_pretty(&scores.to_json()).map_err(|e| (3, e))?)?;
        r.push_str(&format!("output: {}\n", p));
    }
    Ok(r)
}
