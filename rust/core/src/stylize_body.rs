//! Whole-body stylizer: key poses chosen for the body together, one clock.
//!
//! The per-bone stylizer (`stylize.rs`) keys each bone on its own
//! extremes, so the arm arrives at frame 9, the spine at 11 and the hips
//! at 7: no frame where the body as a whole holds a pose, and every bone's
//! easing overlaps the others' (attacks read rubbery). Here:
//!
//! 1. **Body speed.** Per frame, the summed speed of the hands, forearms,
//!    feet, shins, chest and head relative to the hips, plus the hips'
//!    own speed (finite differences of world positions, smoothed
//!    [1,2,1]/4).
//! 2. **Shared keys.** Frames where the body as a whole is at an extreme
//!    or turns around are minima of that speed. Kept: both ends, and every
//!    interior minimum whose prominence (how far the speed climbs on both
//!    sides before it gets lower again) is at least `key_prominence` of
//!    the lower of those two peaks (a slow windup's stop counts as much
//!    as a fast strike's), in motion faster than 5% of the clip's top
//!    speed (idle noise is not a key), strongest first, `min_spacing`
//!    apart.
//! 3. **Retime on one clock.** Each segment between two keys keeps its
//!    frame count. A *big* segment (its top speed at least half the
//!    clip's) gets anticipation: the pose eases `anticipation` of the way
//!    *against* the coming move over `anticipation_frames`, holds `hold`
//!    frames, then plays the source motion sped up evenly into what is
//!    left (an eased warp skipped the arm's extremes inside a strike),
//!    unwinding the counter pose as it goes. After a big segment the
//!    next one starts with a moving hold: the pose carries on past the key
//!    along the big segment's rotation (up to `0.1 * overshoot` of it)
//!    and settles back over `hold + 2` frames. Other segments hold `hold`
//!    frames, then ease out. The source is resampled at the warped time,
//!    so arcs and secondary motion survive; every bone uses the same
//!    warp.
//! 4. **Exaggeration from the clip's own neutral.** Every frame's local
//!    rotation is pushed away from the bone's mean rotation over the clip
//!    (not the bind pose): `neutral * (neutral^-1 q)^factor`, ramped to
//!    no change over the first and last 4 frames so the clip still
//!    starts and ends on its own poses (it blends with what is next to
//!    it).
//!
//! Translations follow the warp (no exaggeration, no overshoot), so root
//! travel ends where it ended. The GLB path re-plants the feet afterwards
//! (`stylize_glb`): plants found on the original clip stay where they
//! were, with two-bone leg IK.

use crate::detmath;
use crate::glb::Document;
use crate::math::{Quat, Vec3};
use crate::rig::{Animation, Rig, Trs};
use crate::stylize::StylizeParams;

#[derive(Clone, Debug)]
pub struct BodyParams {
    pub base: StylizeParams,
    /// A key's speed dip, as a share of the lower of the speeds on its
    /// two sides.
    pub key_prominence: f64,
    /// Frames at each end where exaggeration ramps in/out.
    pub end_ramp: usize,
}

impl Default for BodyParams {
    fn default() -> Self {
        BodyParams {
            base: StylizeParams::default(),
            key_prominence: 0.25,
            end_ramp: 4,
        }
    }
}

/// Input: per frame, per channel bone, local rotation and translation;
/// per bone the exaggeration factor; per frame probe positions (hips
/// first) that make the body-speed signal.
pub struct Track {
    pub fps: f64,
    pub rot: Vec<Vec<Quat>>,
    pub loc: Vec<Vec<Vec3>>,
    pub factor: Vec<f64>,
    pub probes: Vec<Vec<Vec3>>,
}

#[derive(Clone, Debug, Default)]
pub struct BodyReport {
    pub keys: Vec<usize>,
    /// Segments `(from key, to key)` that got anticipation and overshoot.
    pub big: Vec<(usize, usize)>,
    pub counters_added: usize,
    pub counters_skipped: usize,
    /// Source frame (fractional) shown at each output frame.
    pub warp: Vec<f64>,
}

fn ease_out(u: f64) -> f64 {
    1.0 - (1.0 - u).powi(3)
}

fn smoothstep(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/// Mean rotation (sign-aligned normalized sum; exact enough for poses
/// within a hemisphere, which a clip's frames of one bone are).
pub fn mean_quat(qs: &[Quat]) -> Quat {
    if qs.is_empty() {
        return Quat::IDENTITY;
    }
    let r = qs[0];
    let (mut w, mut x, mut y, mut z) = (0.0, 0.0, 0.0, 0.0);
    for q in qs {
        let q = if q.dot(r) < 0.0 { q.neg() } else { *q };
        w += q.w;
        x += q.x;
        y += q.y;
        z += q.z;
    }
    let m = Quat::new(w, x, y, z);
    if m.length() < 1e-12 {
        r
    } else {
        m.normalized()
    }
}

/// Body speed per frame, smoothed [1,2,1]/4.
pub fn body_speed(probes: &[Vec<Vec3>], fps: f64) -> Vec<f64> {
    smooth3(&raw_speed(probes, fps))
}

fn smooth3(raw: &[f64]) -> Vec<f64> {
    let n = raw.len();
    (0..n)
        .map(|f| {
            let a = raw[f.saturating_sub(1)];
            let c = raw[(f + 1).min(n.saturating_sub(1))];
            (a + 2.0 * raw[f] + c) / 4.0
        })
        .collect()
}

/// Body speed per frame, unsmoothed (central differences).
pub fn raw_speed(probes: &[Vec<Vec3>], fps: f64) -> Vec<f64> {
    let n = probes.len();
    let raw: Vec<f64> = (0..n)
        .map(|f| {
            if n < 2 {
                return 0.0;
            }
            let (a, b) = (f.saturating_sub(1), (f + 1).min(n - 1));
            let dt = (b - a) as f64 / fps;
            let rel = |g: usize, j: usize| {
                if j == 0 {
                    probes[g][0]
                } else {
                    probes[g][j] - probes[g][0]
                }
            };
            (0..probes[f].len())
                .map(|j| (rel(b, j) - rel(a, j)).length() / dt)
                .sum()
        })
        .collect();
    raw
}

/// Shared key frames from the body speed: minima of the smoothed speed
/// (`raw` smoothed), each moved to the slowest raw frame next to it (the
/// smoothing drifts a minimum toward the slower side of a reversal).
pub fn body_keys(raw: &[f64], prominence: f64, min_spacing: usize) -> Vec<usize> {
    let speed = &smooth3(raw);
    let n = speed.len();
    if n == 0 {
        return Vec::new();
    }
    let top = speed.iter().cloned().fold(0.0, f64::max);
    let mut cands: Vec<(usize, f64)> = Vec::new();
    for f in 1..n.saturating_sub(1) {
        if !(speed[f] <= speed[f - 1] && speed[f] < speed[f + 1]) {
            continue;
        }
        let mut left = speed[f];
        for g in (0..f).rev() {
            if speed[g] < speed[f] {
                break;
            }
            left = left.max(speed[g]);
        }
        let mut right = speed[f];
        for &s in speed.iter().skip(f + 1) {
            if s < speed[f] {
                break;
            }
            right = right.max(s);
        }
        // A real stop or turn: the dip is deep against the lower of its
        // two shoulders, in motion that is not idle noise.
        let shoulder = left.min(right);
        let prom = shoulder - speed[f];
        if shoulder >= 0.05 * top && prom >= prominence * shoulder && prom > 0.0 {
            let mut at = f;
            for g in [f - 1, f + 1] {
                if g > 0 && g < n - 1 && raw[g] < raw[at] {
                    at = g;
                }
            }
            cands.push((at, prom));
        }
    }
    cands.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
    let mut keys = vec![0, n - 1];
    for (f, _) in cands {
        if keys.iter().all(|k| k.abs_diff(f) >= min_spacing) {
            keys.push(f);
        }
    }
    keys.sort();
    keys.dedup();
    keys
}

fn rot_at(track: &Track, b: usize, t: f64) -> Quat {
    let n = track.rot.len();
    let t = t.clamp(0.0, (n - 1) as f64);
    let f0 = t.floor() as usize;
    let f1 = (f0 + 1).min(n - 1);
    let u = t - f0 as f64;
    if u <= 0.0 || f0 == f1 {
        return track.rot[f0][b];
    }
    track.rot[f0][b].slerp(track.rot[f1][b], u)
}

fn loc_at(track: &Track, b: usize, t: f64) -> Vec3 {
    let n = track.loc.len();
    let t = t.clamp(0.0, (n - 1) as f64);
    let f0 = t.floor() as usize;
    let f1 = (f0 + 1).min(n - 1);
    let u = t - f0 as f64;
    track.loc[f0][b].lerp(track.loc[f1][b], u)
}

/// Stylize a whole-body track. Returns new rotations and translations
/// (same frame count) and what was done.
pub fn stylize_body(
    track: &Track,
    params: &BodyParams,
) -> Result<(Vec<Vec<Quat>>, Vec<Vec<Vec3>>, BodyReport), String> {
    params.base.validate()?;
    if !(0.0..=1.0).contains(&params.key_prominence) {
        return Err("key-prominence must be in 0..=1".into());
    }
    let p = &params.base;
    let n = track.rot.len();
    let nb = track.factor.len();
    if n < 2 {
        return Ok((track.rot.clone(), track.loc.clone(), BodyReport::default()));
    }
    let raw = raw_speed(&track.probes, track.fps);
    let speed = smooth3(&raw);
    let keys = body_keys(&raw, params.key_prominence, p.min_spacing);
    let top = speed.iter().cloned().fold(0.0, f64::max);
    let mut report = BodyReport {
        keys: keys.clone(),
        ..Default::default()
    };
    // Per output frame: source time; anticipation weight and its segment;
    // overshoot amount and the segment it overshoots.
    struct Step {
        src: f64,
        anti: f64,
        seg: (usize, usize),
        over: f64,
        over_seg: (usize, usize),
    }
    let mut plan: Vec<Step> = Vec::with_capacity(n);
    let mut prev_big: Option<(usize, usize)> = None;
    for w in keys.windows(2) {
        let (k0, k1) = (w[0], w[1]);
        let len = k1 - k0;
        let peak = speed[k0..=k1].iter().cloned().fold(0.0, f64::max);
        let big = top > 0.0 && peak >= 0.5 * top;
        let a = p.anticipation_frames;
        // Anticipation and its hold may take at most half the segment:
        // squeezing a strike into the rest skips the source's extremes
        // (a 7-frame overhead chop kept 2 frames and lost its top).
        let room = big && p.anticipation > 0.0 && len >= 2 * (a + p.hold);
        if big && p.anticipation > 0.0 {
            if room {
                report.counters_added += 1;
            } else {
                report.counters_skipped += 1;
            }
        }
        if big {
            report.big.push((k0, k1));
        }
        // Moving hold after a big arrival: the pose carries on past the
        // key along that segment's rotation and settles back.
        let settle = p.hold + 2;
        for i in 0..len {
            let mut step = Step {
                src: k0 as f64,
                anti: 0.0,
                seg: (k0, k1),
                over: 0.0,
                over_seg: (k0, k1),
            };
            if let Some(pb) = prev_big {
                if i < settle && p.overshoot > 0.0 {
                    step.over = 0.1
                        * p.overshoot
                        * detmath::sin(std::f64::consts::PI * i as f64 / settle as f64);
                    step.over_seg = pb;
                }
            }
            if room {
                if i < a {
                    step.anti = smoothstep(i as f64 / a as f64);
                } else if i < a + p.hold {
                    step.anti = 1.0;
                } else {
                    // The strike: the source, sped up evenly (an eased
                    // warp would skip the extremes inside it), unwinding
                    // the counter pose as it goes.
                    let u = (i - a - p.hold) as f64 / (len - a - p.hold) as f64;
                    step.src = k0 as f64 + u * len as f64;
                    step.anti = 1.0 - u;
                }
            } else {
                let h = if big {
                    p.hold.min(len / 4)
                } else {
                    p.hold.min(len.saturating_sub(1))
                };
                if i >= h {
                    let u = (i - h) as f64 / (len - h) as f64;
                    let e = if big { u } else { ease_out(u) };
                    step.src = k0 as f64 + e * len as f64;
                }
            }
            plan.push(step);
        }
        prev_big = if big { Some((k0, k1)) } else { None };
    }
    plan.push(Step {
        src: (n - 1) as f64,
        anti: 0.0,
        seg: (n - 1, n - 1),
        over: 0.0,
        over_seg: (n - 1, n - 1),
    });

    let neutral: Vec<Quat> = (0..nb)
        .map(|b| mean_quat(&track.rot.iter().map(|fr| fr[b]).collect::<Vec<_>>()))
        .collect();
    let ramp = params.end_ramp.max(1).min((n / 4).max(1));
    let mut rot = Vec::with_capacity(n);
    let mut loc = Vec::with_capacity(n);
    for (g, st) in plan.iter().enumerate() {
        let src = st.src;
        report.warp.push(src);
        let edge = g.min(n - 1 - g) as f64 / ramp as f64;
        let w_ex = smoothstep(edge);
        let mut fr = Vec::with_capacity(nb);
        for b in 0..nb {
            let delta = |(k0, k1): (usize, usize)| track.rot[k0][b].conj().mul(track.rot[k1][b]);
            let mut q = rot_at(track, b, src);
            if st.anti > 0.0 {
                q = q.mul(Quat::IDENTITY.slerp(delta(st.seg), -p.anticipation * st.anti));
            }
            if st.over > 0.0 {
                q = q.mul(Quat::IDENTITY.slerp(delta(st.over_seg), st.over));
            }
            let f = 1.0 + (track.factor[b] - 1.0) * w_ex;
            if (f - 1.0).abs() > 1e-12 {
                let rel = neutral[b].conj().mul(q);
                q = neutral[b].mul(Quat::IDENTITY.slerp(rel, f));
            }
            fr.push(q.normalized());
        }
        rot.push(fr);
        loc.push((0..nb).map(|b| loc_at(track, b, src)).collect());
    }
    Ok((rot, loc, report))
}

/// Which stylizer runs on a GLB.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mode {
    /// Shared keys, one clock, exaggeration from the clip's neutral.
    Body,
    /// The original per-bone stylizer (keys per bone, exaggeration from
    /// the bind pose), as on clip JSON.
    PerBone,
}

#[derive(Clone, Debug)]
pub struct GlbClipReport {
    pub name: String,
    pub source: String,
    pub frames: usize,
    pub keys: Vec<usize>,
    pub big: Vec<(usize, usize)>,
    pub counters_added: usize,
    pub counters_skipped: usize,
    pub plants: usize,
    pub replant_frames: usize,
    pub replant_max_m: f64,
}

/// Probe joints for the body-speed signal (hips first).
pub fn probe_nodes(map: &crate::humanoid::Mapping) -> Vec<usize> {
    let mut v = Vec::new();
    for n in [
        "DEF-spine",
        "DEF-spine.003",
        "DEF-spine.006",
        "DEF-forearm.L",
        "DEF-hand.L",
        "DEF-forearm.R",
        "DEF-hand.R",
        "DEF-shin.L",
        "DEF-foot.L",
        "DEF-shin.R",
        "DEF-foot.R",
    ] {
        if let Some(i) = map.node_of(n) {
            v.push(i);
        }
    }
    v
}

/// Stylize clips of a rigged GLB. `clips` empty = every clip. With
/// `suffix`, the stylized clip is added next to the original as
/// `<name><suffix>`; without, it replaces it. `replant` pins the feet
/// where the original clip planted them.
pub fn stylize_glb(
    doc: &Document,
    params: &BodyParams,
    mode: Mode,
    clips: &[String],
    suffix: Option<&str>,
    replant: bool,
) -> Result<(Document, Vec<GlbClipReport>), String> {
    let rig = Rig::from_doc(doc)?;
    let map = crate::humanoid::map_rig(&rig);
    let hips = map.node_of("DEF-spine").ok_or("no hips bone found")?;
    let anims = Animation::load_all(doc)?;
    for c in clips {
        if !anims.iter().any(|a| &a.name == c) {
            return Err(format!("no clip named '{}'", c));
        }
    }
    let probes = probe_nodes(&map);
    let mut out = doc.clone();
    let mut reports = Vec::new();
    let mut new_anims: Vec<(usize, crate::json::Json, String)> = Vec::new();
    for (ai, anim) in anims.iter().enumerate() {
        if !clips.is_empty() && !clips.contains(&anim.name) {
            continue;
        }
        let (times, frames) = crate::motioncheck::sample(&rig, anim, None);
        let n = frames.len();
        if n < 2 {
            continue;
        }
        let fps = (n - 1) as f64 / (times[n - 1] - times[0]).max(1e-9);
        let mut nodes = anim.animated_nodes();
        if !nodes.contains(&hips) {
            nodes.push(hips);
            nodes.sort_unstable();
        }
        let trs = |f: usize, node: usize| frames[f][node].unwrap_or(rig.rest[node]);
        let factor: Vec<f64> = nodes
            .iter()
            .map(|&nd| params.base.factor_for(&rig.names[nd]))
            .collect();
        let (rot, loc, keys, big, added, skipped, warp) = match mode {
            Mode::Body => {
                let world: Vec<_> = frames.iter().map(|f| rig.world(f)).collect();
                let track = Track {
                    fps,
                    rot: (0..n)
                        .map(|f| nodes.iter().map(|&nd| trs(f, nd).r).collect())
                        .collect(),
                    loc: (0..n)
                        .map(|f| nodes.iter().map(|&nd| trs(f, nd).t).collect())
                        .collect(),
                    factor,
                    probes: (0..n)
                        .map(|f| probes.iter().map(|&p| world[f][p].t).collect())
                        .collect(),
                };
                let (rot, loc, rep) = stylize_body(&track, params)?;
                (
                    rot,
                    loc,
                    rep.keys,
                    rep.big,
                    rep.counters_added,
                    rep.counters_skipped,
                    rep.warp,
                )
            }
            Mode::PerBone => {
                // The old algorithm exactly: rotations relative to the
                // bind pose, as clip JSON stores them.
                use crate::clip::{Bone, Clip, Frame, Pose, Skeleton};
                let skeleton = Skeleton {
                    bones: nodes
                        .iter()
                        .map(|&nd| Bone {
                            name: rig.names[nd].clone(),
                            parent: None,
                            head: Vec3::ZERO,
                            tail: Vec3::new(0.0, 0.1, 0.0),
                        })
                        .collect(),
                };
                let clip = Clip {
                    fps,
                    skeleton,
                    frames: (0..n)
                        .map(|f| Frame {
                            poses: nodes
                                .iter()
                                .map(|&nd| Pose {
                                    loc: trs(f, nd).t,
                                    quat: rig.rest[nd].r.conj().mul(trs(f, nd).r),
                                })
                                .collect(),
                        })
                        .collect(),
                };
                let (sc, rep) = crate::stylize::stylize(&clip, &params.base)?;
                let rot = sc
                    .frames
                    .iter()
                    .map(|fr| {
                        fr.poses
                            .iter()
                            .zip(&nodes)
                            .map(|(p, &nd)| rig.rest[nd].r.mul(p.quat).normalized())
                            .collect()
                    })
                    .collect();
                let loc = sc
                    .frames
                    .iter()
                    .map(|fr| fr.poses.iter().map(|p| p.loc).collect())
                    .collect();
                (
                    rot,
                    loc,
                    rep.keys_union,
                    Vec::new(),
                    rep.counters_added,
                    rep.counters_skipped,
                    (0..n).map(|f| f as f64).collect(),
                )
            }
        };
        // Back to per-frame local overrides.
        let mut posed: Vec<Vec<Option<Trs>>> = (0..n)
            .map(|f| {
                let mut l = frames[f].clone();
                for (k, &nd) in nodes.iter().enumerate() {
                    let base = trs(f, nd);
                    l[nd] = Some(Trs {
                        t: loc[f][k],
                        r: rot[f][k],
                        s: base.s,
                    });
                }
                l
            })
            .collect();
        let (plants, replant_frames, replant_max) = if replant {
            replant_feet(&rig, &map, &frames, &times, &warp, &mut posed)
        } else {
            (0, 0, 0.0)
        };
        let name = match suffix {
            Some(s) => format!("{}{}", anim.name, s),
            None => anim.name.clone(),
        };
        let tracks: Vec<Vec<Trs>> = nodes
            .iter()
            .map(|&nd| {
                (0..n)
                    .map(|f| posed[f][nd].unwrap_or(rig.rest[nd]))
                    .collect()
            })
            .collect();
        // Translation channels for every node whose translation is not its
        // rest one on some frame (moving, or held away from rest: dropping
        // a held offset put the hips back at rest height).
        let moving: Vec<usize> = nodes
            .iter()
            .enumerate()
            .filter(|(k, &nd)| {
                tracks[*k]
                    .iter()
                    .any(|t| (t.t - rig.rest[nd].t).length() > 1e-9)
            })
            .map(|(_, &nd)| nd)
            .collect();
        let json = encode_animation(&mut out, &name, &times, &nodes, &tracks, &moving);
        new_anims.push((ai, json, name.clone()));
        reports.push(GlbClipReport {
            name,
            source: anim.name.clone(),
            frames: n,
            keys,
            big,
            counters_added: added,
            counters_skipped: skipped,
            plants,
            replant_frames,
            replant_max_m: replant_max,
        });
    }
    let list = out.array_mut("animations");
    match suffix {
        Some(_) => {
            for (_, j, _) in new_anims {
                list.push(j);
            }
        }
        None => {
            for (ai, j, _) in new_anims {
                list[ai] = j;
            }
        }
    }
    out.compact();
    Ok((out, reports))
}

/// Animation JSON (accessors pushed into `out`): rotation for every node,
/// translation for the `moving` ones.
fn encode_animation(
    out: &mut Document,
    name: &str,
    times: &[f64],
    nodes: &[usize],
    tracks: &[Vec<Trs>],
    moving: &[usize],
) -> crate::json::Json {
    use crate::json::Json;
    let t0 = times[0];
    let rows: Vec<Vec<f64>> = times.iter().map(|t| vec![t - t0]).collect();
    let input = out.push_float_accessor(&rows, "SCALAR", true);
    let mut samplers = Vec::new();
    let mut channels = Vec::new();
    let mut add = |out: &mut Document, node: usize, path: &str, rows: Vec<Vec<f64>>| {
        let kind = if path == "rotation" { "VEC4" } else { "VEC3" };
        let output = out.push_float_accessor(&rows, kind, false);
        channels.push(Json::obj(vec![
            ("sampler", Json::num(samplers.len() as f64)),
            (
                "target",
                Json::obj(vec![
                    ("node", Json::num(node as f64)),
                    ("path", Json::str(path)),
                ]),
            ),
        ]));
        samplers.push(Json::obj(vec![
            ("input", Json::num(input as f64)),
            ("interpolation", Json::str("LINEAR")),
            ("output", Json::num(output as f64)),
        ]));
    };
    for (k, &node) in nodes.iter().enumerate() {
        let mut prev: Option<Quat> = None;
        let rows: Vec<Vec<f64>> = tracks[k]
            .iter()
            .map(|trs| {
                let mut q = trs.r.normalized();
                if let Some(p) = prev {
                    if p.dot(q) < 0.0 {
                        q = q.neg();
                    }
                }
                prev = Some(q);
                vec![q.x, q.y, q.z, q.w]
            })
            .collect();
        add(out, node, "rotation", rows);
        if moving.contains(&node) {
            let rows = tracks[k]
                .iter()
                .map(|t| vec![t.t.x, t.t.y, t.t.z])
                .collect();
            add(out, node, "translation", rows);
        }
    }
    Json::obj(vec![
        ("name", Json::str(name)),
        ("channels", Json::Arr(channels)),
        ("samplers", Json::Arr(samplers)),
    ])
}

/// Pin the feet of `posed` where the original `frames` planted them:
/// output frame g shows source time `warp[g]`; inside a source plant the
/// ankle goes back to the original's ankle at that source time and the
/// foot to its world rotation (leg IK), eased over 3 frames at each end.
/// Returns (plants, frames changed, largest ankle move m).
fn replant_feet(
    rig: &Rig,
    map: &crate::humanoid::Mapping,
    frames: &[Vec<Option<Trs>>],
    times: &[f64],
    warp: &[f64],
    posed: &mut [Vec<Option<Trs>>],
) -> (usize, usize, f64) {
    let n = posed.len();
    let orig_world: Vec<_> = frames.iter().map(|f| rig.world(f)).collect();
    let mut plants_total = 0;
    let mut changed = vec![false; n];
    let mut max_move: f64 = 0.0;
    for side in ["L", "R"] {
        let (Some(th), Some(sh), Some(ft)) = (
            map.node_of(&format!("DEF-thigh.{side}")),
            map.node_of(&format!("DEF-shin.{side}")),
            map.node_of(&format!("DEF-foot.{side}")),
        ) else {
            continue;
        };
        let leg = (rig.head(sh) - rig.head(th)).length() + (rig.head(ft) - rig.head(sh)).length();
        let ankle: Vec<Vec3> = orig_world.iter().map(|w| w[ft].t).collect();
        let plants = crate::footpin::plants(&ankle, times, leg);
        plants_total += plants.len();
        if plants.is_empty() {
            continue;
        }
        // Weight per source frame: 1 inside a plant, easing out over 3.
        let ramp = 3usize;
        let mut wsrc = vec![0.0f64; n];
        for &(a, b) in &plants {
            for w in wsrc.iter_mut().take(b).skip(a) {
                *w = 1.0;
            }
            for r in 1..=ramp {
                let w = smoothstep(1.0 - r as f64 / (ramp + 1) as f64);
                if a >= r {
                    wsrc[a - r] = wsrc[a - r].max(w);
                }
                if b - 1 + r < n {
                    wsrc[b - 1 + r] = wsrc[b - 1 + r].max(w);
                }
            }
        }
        for g in 0..n {
            let s = warp[g].clamp(0.0, (n - 1) as f64);
            let f0 = s.floor() as usize;
            let f1 = (f0 + 1).min(n - 1);
            let u = s - f0 as f64;
            let w = wsrc[f0] * (1.0 - u) + wsrc[f1] * u;
            if w <= 1e-6 {
                continue;
            }
            let want_p = ankle[f0].lerp(ankle[f1], u);
            let want_r = orig_world[f0][ft]
                .rotation()
                .slerp(orig_world[f1][ft].rotation(), u);
            let world = rig.world(&posed[g]);
            let cur_p = world[ft].t;
            let cur_r = world[ft].rotation();
            let target = cur_p.lerp(want_p, w);
            if (target - cur_p).length() < 1e-7 && cur_r.angle_to(want_r) < 1e-7 {
                continue;
            }
            crate::contact::solve_arm(rig, &mut posed[g], th, sh, ft, target);
            let world = rig.world(&posed[g]);
            let pr = world[sh].rotation();
            let want = cur_r.slerp(want_r, w);
            let mut l = posed[g][ft].unwrap_or(rig.rest[ft]);
            l.r = crate::glb::canonical_quat(pr.conj().mul(want));
            posed[g][ft] = Some(l);
            changed[g] = true;
            max_move = max_move.max((target - cur_p).length());
        }
    }
    (
        plants_total,
        changed.iter().filter(|c| **c).count(),
        max_move,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn track_from(angles: &[f64]) -> Track {
        // One bone; probe = a hand on a 1 m arm swinging in the XY plane.
        let axis = Vec3::new(0.0, 0.0, 1.0);
        Track {
            fps: 30.0,
            rot: angles
                .iter()
                .map(|a| vec![Quat::from_axis_angle(axis, *a)])
                .collect(),
            loc: angles.iter().map(|_| vec![Vec3::ZERO]).collect(),
            factor: vec![1.0],
            probes: angles
                .iter()
                .map(|a| vec![Vec3::ZERO, Vec3::new(a.cos(), a.sin(), 0.0)])
                .collect(),
        }
    }

    fn plain() -> BodyParams {
        BodyParams {
            base: StylizeParams {
                exaggeration: 1.0,
                chain_factors: vec![],
                angle_threshold: 0.15,
                min_spacing: 3,
                hold: 0,
                anticipation: 0.0,
                anticipation_frames: 3,
                overshoot: 0.0,
            },
            key_prominence: 0.25,
            end_ramp: 4,
        }
    }

    /// A windup to -0.6 rad at frame 10, a fast strike to +1.2 at 18,
    /// back to 0 by 40.
    fn attack_angles() -> Vec<f64> {
        (0..41)
            .map(|f| {
                let f = f as f64;
                if f <= 10.0 {
                    -0.6 * smoothstep(f / 10.0)
                } else if f <= 18.0 {
                    -0.6 + 1.8 * smoothstep((f - 10.0) / 8.0)
                } else {
                    1.2 * (1.0 - smoothstep((f - 18.0) / 22.0))
                }
            })
            .collect()
    }

    #[test]
    fn keys_land_on_the_body_extremes() {
        let t = track_from(&attack_angles());
        let speed = raw_speed(&t.probes, t.fps);
        let keys = body_keys(&speed, 0.25, 3);
        // The extremes are at 10 and 18; sampled at 30 fps the slowest
        // frame of a reversal can sit one frame off it.
        assert_eq!(keys.len(), 4, "{keys:?}");
        assert_eq!((keys[0], keys[3]), (0, 40));
        assert!(
            keys[1].abs_diff(10) <= 1 && keys[2].abs_diff(18) <= 1,
            "{keys:?}"
        );
    }

    #[test]
    fn identity_params_reproduce_the_source_on_keys() {
        let angles = attack_angles();
        let t = track_from(&angles);
        let (rot, _, rep) = stylize_body(&t, &plain()).unwrap();
        for &k in &rep.keys {
            let a = Quat::IDENTITY.angle_to(rot[k][0]);
            assert!((a - angles[k].abs()).abs() < 1e-9, "frame {k}");
        }
    }

    #[test]
    fn every_bone_shares_one_clock() {
        // Two bones with different extremes still hold on the same frames.
        let angles = attack_angles();
        let mut t = track_from(&angles);
        for (f, fr) in t.rot.iter_mut().enumerate() {
            let b = Quat::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), 0.3 * (f as f64 * 0.37).sin());
            fr.push(b);
            t.loc[f].push(Vec3::ZERO);
        }
        t.factor.push(1.0);
        let mut p = plain();
        p.base.hold = 2;
        let (rot, _, rep) = stylize_body(&t, &p).unwrap();
        // Both bones are frozen over the first 2 frames after each key.
        for &k in &rep.keys[..rep.keys.len() - 1] {
            for b in 0..2 {
                assert!(
                    rot[k][b].angle_to(rot[k + 1][b]) < 1e-12,
                    "key {k} bone {b}"
                );
            }
        }
    }

    #[test]
    fn anticipation_and_overshoot_on_the_strike() {
        let angles = attack_angles();
        let t = track_from(&angles);
        let mut p = plain();
        p.base.anticipation = 0.25;
        p.base.anticipation_frames = 2;
        p.base.hold = 1;
        p.base.overshoot = 1.0;
        let (rot, _, rep) = stylize_body(&t, &p).unwrap();
        // The strike (about 10 -> 18) is the only big segment; the
        // windup goes further back than the source's -0.6 rad before it.
        assert_eq!(rep.big.len(), 1, "{:?}", rep.big);
        let (k0, k1) = rep.big[0];
        assert!(
            k0.abs_diff(10) <= 1 && k1.abs_diff(18) <= 1,
            "{:?}",
            rep.big
        );
        let signed = |q: Quat| {
            let a = Quat::IDENTITY.angle_to(q);
            if q.z * q.w < 0.0 {
                -a
            } else {
                a
            }
        };
        let min = (k0..k1).map(|f| signed(rot[f][0])).fold(f64::MAX, f64::min);
        assert!(min < -0.6 - 0.2, "windup {min}");
        // ...arrives exactly on the key, then carries on past it and
        // settles (the moving hold).
        assert!((signed(rot[k1][0]) - angles[k1]).abs() < 1e-9);
        let max = (k1..k1 + 3)
            .map(|f| signed(rot[f][0]))
            .fold(f64::MIN, f64::max);
        assert!(max > angles[k1] + 0.05, "overshoot {max}");
    }

    #[test]
    fn exaggeration_is_from_the_clips_own_neutral() {
        // A bone that sits at 1.0 rad and wiggles +-0.2: exaggeration 2
        // doubles the wiggle about 1.0, it does not double the 1.0.
        let angles: Vec<f64> = (0..40)
            .map(|f| 1.0 + 0.2 * (f as f64 * 0.3).sin())
            .collect();
        let mut t = track_from(&angles);
        t.factor = vec![2.0];
        let mut p = plain();
        p.base.exaggeration = 2.0;
        p.key_prominence = 2.0; // no interior keys
        assert!(stylize_body(&t, &p).is_err());
        p.key_prominence = 1.0;
        let (rot, _, _) = stylize_body(&t, &p).unwrap();
        let mut hi: f64 = 0.0;
        let mut lo: f64 = 9.0;
        for (f, fr) in rot.iter().enumerate().take(36).skip(4) {
            let a = Quat::IDENTITY.angle_to(fr[0]);
            assert!(a < 1.6 && a > 0.4, "frame {f}: {a}");
            hi = hi.max(a);
            lo = lo.min(a);
        }
        // The wiggle (source 0.8..1.2) is wider, about the same middle.
        assert!(hi > 1.3 && lo < 0.7, "{lo}..{hi}");
        // Ends untouched.
        assert!((Quat::IDENTITY.angle_to(rot[0][0]) - angles[0]).abs() < 1e-9);
    }

    #[test]
    fn deterministic() {
        let t = track_from(&attack_angles());
        let mut p = plain();
        p.base = StylizeParams::default();
        let a = stylize_body(&t, &p).unwrap();
        let b = stylize_body(&t, &p).unwrap();
        assert_eq!(format!("{:?}", a.0), format!("{:?}", b.0));
    }
}
