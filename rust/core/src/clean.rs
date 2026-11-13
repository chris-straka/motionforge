//! Clean-up for a rigged GLB's own clips: the defects `motion check`
//! measures, fixed where a fix is unambiguous.
//!
//! 1. **Ground.** When a clip keeps both feet off the floor on nearly
//!    every frame (90%+), or keeps a foot under it on nearly every frame,
//!    it carries a height offset (a take keyed with the hips somewhere
//!    else): the whole clip moves by the 10th-percentile height of its
//!    lowest foot point (its most grounded frames), so those land on the
//!    floor. A jump keeps its air: most of its frames are not off the
//!    floor. Only upright frames count (hips at 60%+ of their standing
//!    height above the feet), and only when at least half the clip is
//!    upright: lying down, the feet are no floor reference. And only a
//!    steady offset (the median within 4 cm of the 10th percentile): a
//!    death whose feet sink while it buckles is fixed frame by frame.
//! 2. **No foot under the floor.** Per frame, when both feet are under
//!    the floor the hips come up by the smaller sink; a foot whose lowest
//!    point (heel side or ball) is still under is lifted by leg IK, its
//!    world rotation kept. Lifts ease in and out over 3 frames.
//! 3. **Plants locked.** The plants `motion check` scores (lowest foot
//!    point under 5 cm, slower than 0.75 m/s, on the clip's treadmill
//!    for an in-place cycle), when they slide more than 2.5 cm: the point
//!    that is down stays where it touched down (heel, then the ball after
//!    a roll; `docs/motion-check.md`), the ankle moved by leg IK, eased in
//!    and out over 3 frames. The Kovar-style footskate lock, per contact
//!    point. Not on in-place cycles (their stance speed is the game's to
//!    match) or plants that would need a foot moved more than 35 cm (a
//!    fall or a drag).
//! 4. **Arms out of the body** (opt-in, `--contact`). The contact pass
//!    (`contact::fix_clip`), as `animate` runs it.
//!
//! Thresholds are `motion check`'s, scaled by the character's height.
//! Twist helpers are re-driven from their drivers afterwards.

use crate::contact::{Body, ClipContact, Options as ContactOptions};
use crate::glb::{canonical_quat, Document};
use crate::humanoid::Mapping;
use crate::json::Json;
use crate::math::{Quat, Vec3};
use crate::motioncheck::{
    self, CONTACT_HEIGHT, PLANT_SPEED, REF_HEIGHT, SINK_DEPTH, SKATE_DISTANCE,
};
use crate::rig::{Animation, Rig, Trs};

/// Share of frames that must float (or sink) for the whole clip to move.
pub const GROUND_SHARE: f64 = 0.9;
/// Frames of ease at each end of a locked plant (and of a lift).
pub const LOCK_RAMP: usize = 3;
/// A whole-clip offset is steady: the median grounded height is within
/// this (m at reference height) of the 10th percentile.
pub const STEADY: f64 = 0.04;
/// Upright: hips at least this share of their standing height above the
/// lowest foot point.
pub const UPRIGHT: f64 = 0.6;
/// Largest foot move a lock may make (m at reference height).
pub const LOCK_MAX: f64 = 0.35;

#[derive(Clone, Debug)]
pub struct CleanOptions {
    pub ground: bool,
    pub lock: bool,
    /// The contact pass, when given.
    pub contact: Option<ContactOptions>,
}

impl Default for CleanOptions {
    fn default() -> Self {
        CleanOptions {
            ground: true,
            lock: true,
            // Off by default: on the game's 31 clips it took through-body
            // frames 889 -> 186 but doubled back-and-forth and added wrist
            // turns past their limits (docs/log.md, 2026-11-13).
            contact: None,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct CleanReport {
    pub name: String,
    pub source: String,
    pub frames: usize,
    /// Whole-clip height change, metres (negative = down).
    pub ground_shift_m: f64,
    /// Largest per-frame hips lift (both feet under the floor), metres.
    pub hips_lift_max_m: f64,
    pub lift_frames: usize,
    pub lift_max_m: f64,
    pub plants_locked: usize,
    pub lock_frames: usize,
    pub lock_max_m: f64,
    pub contact: Option<ClipContact>,
}

impl CleanReport {
    pub fn to_json(&self) -> Json {
        let cm = |m: f64| Json::num((m * 1000.0).round() / 10.0);
        let mut f = vec![
            ("name", Json::str(&self.name)),
            ("source", Json::str(&self.source)),
            ("frames", Json::num(self.frames as f64)),
            ("ground_shift_cm", cm(self.ground_shift_m)),
            ("hips_lift_max_cm", cm(self.hips_lift_max_m)),
            ("lift_frames", Json::num(self.lift_frames as f64)),
            ("lift_max_cm", cm(self.lift_max_m)),
            ("plants_locked", Json::num(self.plants_locked as f64)),
            ("lock_frames", Json::num(self.lock_frames as f64)),
            ("lock_max_cm", cm(self.lock_max_m)),
        ];
        if let Some(c) = &self.contact {
            f.push(("contact", c.to_json()));
        }
        Json::obj(f)
    }
}

/// A leg: thigh, shin, foot nodes and the foot's contact points with
/// their rest heights.
struct Leg {
    th: usize,
    sh: usize,
    ft: usize,
    points: Vec<(usize, f64)>,
}

fn legs(rig: &Rig, map: &Mapping) -> Vec<Leg> {
    let mut out = Vec::new();
    for side in ["L", "R"] {
        let get = |b: &str| map.node_of(&format!("DEF-{b}.{side}"));
        let (Some(th), Some(sh), Some(ft)) = (get("thigh"), get("shin"), get("foot")) else {
            continue;
        };
        let mut points = vec![(ft, rig.head(ft).y)];
        if let Some(t) = get("toe") {
            points.push((t, rig.head(t).y));
        }
        out.push(Leg { th, sh, ft, points });
    }
    out
}

/// Lowest contact point of a leg on a posed frame: (node, height above
/// its rest standing height).
fn lowest(leg: &Leg, world: &[crate::glb::Affine]) -> (usize, f64) {
    leg.points
        .iter()
        .map(|&(n, y)| (n, world[n].t.y - y))
        .fold(
            (leg.ft, f64::INFINITY),
            |a, b| if b.1 < a.1 { b } else { a },
        )
}

/// Move the ankle to `target` by leg IK, keeping the foot's world rotation.
fn place_foot(rig: &Rig, locals: &mut [Option<Trs>], leg: &Leg, target: Vec3, foot_rot: Quat) {
    crate::contact::solve_arm(rig, locals, leg.th, leg.sh, leg.ft, target);
    let world = rig.world(locals);
    let mut l = locals[leg.ft].unwrap_or(rig.rest[leg.ft]);
    l.r = canonical_quat(world[leg.sh].rotation().conj().mul(foot_rot));
    locals[leg.ft] = Some(l);
}

/// Move the hips by `delta` (world) on one frame.
fn shift_hips(rig: &Rig, locals: &mut [Option<Trs>], hips: usize, delta: Vec3) {
    let world = rig.world(locals);
    let parent = rig.parent[hips]
        .map(|p| world[p])
        .unwrap_or(crate::glb::Affine::IDENTITY);
    let Some(inv) = parent.inverse() else {
        return;
    };
    let mut l = locals[hips].unwrap_or(rig.rest[hips]);
    l.t = inv.apply(world[hips].t + delta);
    locals[hips] = Some(l);
}

fn smoothstep(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/// Clean one clip in place (`frames`: local overrides at `times`).
#[allow(clippy::too_many_arguments)]
pub fn clean_clip(
    rig: &Rig,
    map: &Mapping,
    body: Option<&Body>,
    height: f64,
    frames: &mut [Vec<Option<Trs>>],
    times: &[f64],
    name: &str,
    opts: &CleanOptions,
) -> CleanReport {
    let n = frames.len();
    let mut rep = CleanReport {
        name: name.to_string(),
        source: name.to_string(),
        frames: n,
        ..Default::default()
    };
    let Some(hips) = map.node_of("DEF-spine") else {
        return rep;
    };
    let legs = legs(rig, map);
    if n < 2 || legs.is_empty() {
        return rep;
    }
    let k = height / REF_HEIGHT;
    let up = Vec3::new(0.0, 1.0, 0.0);
    let dt = ((times[n - 1] - times[0]) / (n - 1) as f64).max(1e-9);

    // 1. Ground offset, read on upright frames only (lying down, the feet
    // are no floor reference: a death lifted by its lying frames floated).
    if opts.ground {
        let hip_rest = rig.rest_world[hips].t.y
            - legs
                .iter()
                .flat_map(|l| l.points.iter().map(|p| p.1))
                .fold(f64::INFINITY, f64::min);
        let mut low = Vec::new();
        for f in frames.iter() {
            let w = rig.world(f);
            let feet_y = legs
                .iter()
                .flat_map(|l| l.points.iter().map(|&(nd, _)| w[nd].t.y))
                .fold(f64::INFINITY, f64::min);
            if w[hips].t.y - feet_y < UPRIGHT * hip_rest {
                continue;
            }
            low.push(
                legs.iter()
                    .map(|l| lowest(l, &w).1)
                    .fold(f64::INFINITY, f64::min),
            );
        }
        if low.len() * 4 >= n && low.len() >= 5 {
            let m = low.len();
            let floating = low.iter().filter(|h| **h > CONTACT_HEIGHT * k).count();
            let sinking = low.iter().filter(|h| **h < -SINK_DEPTH * k).count();
            let mut sorted = low.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let c = sorted[(m - 1) / 10];
            // An offset is steady: the grounded half of the frames agree.
            let steady = sorted[(m - 1) / 2] - c < STEADY * k;
            let whole = (m as f64 * GROUND_SHARE).ceil() as usize;
            if steady
                && ((floating >= whole && c > CONTACT_HEIGHT * k)
                    || (sinking >= whole && c < -SINK_DEPTH * k))
            {
                for f in frames.iter_mut() {
                    shift_hips(rig, f, hips, up.scale(-c));
                }
                rep.ground_shift_m = -c;
            }
        }
    }

    // 2. No foot under the floor. When both feet are under, the body is
    // too low: the hips come up by the smaller sink; then each foot still
    // under is lifted by leg IK, its world rotation kept. Every lift is
    // spread over LOCK_RAMP frames each side (one that starts and stops on
    // a single frame is a kink).
    if opts.ground {
        let spread = |need: &[f64]| -> Vec<f64> {
            (0..n)
                .map(|f| {
                    let lo = f.saturating_sub(LOCK_RAMP);
                    let hi = (f + LOCK_RAMP).min(n - 1);
                    (lo..=hi)
                        .map(|g| {
                            let d = f.abs_diff(g) as f64 / (LOCK_RAMP + 1) as f64;
                            need[g] * smoothstep(1.0 - d)
                        })
                        .fold(0.0, f64::max)
                })
                .collect()
        };
        let both: Vec<f64> = frames
            .iter()
            .map(|f| {
                let w = rig.world(f);
                legs.iter()
                    .map(|l| (-lowest(l, &w).1).max(0.0))
                    .fold(f64::INFINITY, f64::min)
            })
            .collect();
        let body_lift = spread(&both);
        for (f, fr) in frames.iter_mut().enumerate() {
            if body_lift[f] > 1e-4 * k {
                shift_hips(rig, fr, hips, up.scale(body_lift[f]));
                rep.hips_lift_max_m = rep.hips_lift_max_m.max(body_lift[f]);
            }
        }
        for leg in &legs {
            let need: Vec<f64> = frames
                .iter()
                .map(|f| (-lowest(leg, &rig.world(f)).1).max(0.0))
                .collect();
            let lift = spread(&need);
            for (f, fr) in frames.iter_mut().enumerate() {
                if lift[f] < 1e-4 * k {
                    continue;
                }
                let w = rig.world(fr);
                let rot = w[leg.ft].rotation();
                place_foot(rig, fr, leg, w[leg.ft].t + up.scale(lift[f]), rot);
                rep.lift_frames += 1;
                rep.lift_max_m = rep.lift_max_m.max(lift[f]);
            }
        }
    }

    // 3. Plants locked.
    if opts.lock {
        let world: Vec<Vec<crate::glb::Affine>> = frames.iter().map(|f| rig.world(f)).collect();
        let pos = |f: usize, node: usize| world[f][node].t;
        let belt = motioncheck::treadmill(
            &legs
                .iter()
                .map(|l| {
                    (0..n)
                        .map(|f| {
                            let (nd, h) = lowest(l, &world[f]);
                            (pos(f, nd), h)
                        })
                        .collect()
                })
                .collect::<Vec<_>>(),
            &(0..n).map(|f| pos(f, hips)).collect::<Vec<_>>(),
            dt,
            k,
        );
        // An in-place cycle's stance speed is the game's to match, not a
        // slide to lock (locking it to the median belt made the walk's
        // skate worse).
        let cycle = belt.length() > 0.0;
        let t0 = times[0];
        let fpos = |f: usize, node: usize| pos(f, node) - belt.scale(times[f] - t0);
        let mut changed = vec![false; n];
        // Per leg: the horizontal ankle offset and its weight per frame.
        for leg in legs.iter().filter(|_| !cycle) {
            let low: Vec<(usize, f64)> = (0..n).map(|f| lowest(leg, &world[f])).collect();
            let speed = |f: usize| -> f64 {
                let node = low[f].0;
                let (a, b) = (f.saturating_sub(1), (f + 1).min(n - 1));
                if a == b {
                    return 0.0;
                }
                flat(fpos(b, node) - fpos(a, node)).length() / ((b - a) as f64 * dt)
            };
            let planted: Vec<bool> = (0..n)
                .map(|f| low[f].1 < CONTACT_HEIGHT * k && speed(f) < PLANT_SPEED * k)
                .collect();
            let mut offset = vec![Vec3::ZERO; n];
            let mut weight = vec![0.0f64; n];
            let mut f = 0;
            while f < n {
                if !planted[f] {
                    f += 1;
                    continue;
                }
                let a = f;
                while f < n && planted[f] {
                    f += 1;
                }
                let b = f;
                if b - a < 2 {
                    continue;
                }
                let slide: f64 = (a..b - 1)
                    .map(|g| flat(fpos(g + 1, low[g].0) - fpos(g, low[g].0)).length())
                    .sum();
                if slide <= SKATE_DISTANCE * k {
                    continue;
                }
                // Each contact point stays where it was when it became the
                // point that is down (where the lock had put it, so the
                // offset carries on across a heel-to-ball roll).
                let mut plant_off = vec![Vec3::ZERO; b - a];
                let mut anchor = Vec3::ZERO;
                let mut cur = usize::MAX;
                for g in a..b {
                    let node = low[g].0;
                    if node != cur {
                        // Where the lock had this point on the frame before.
                        anchor = if g > a {
                            flat(fpos(g - 1, node)) + plant_off[g - a - 1]
                        } else {
                            flat(fpos(g, node))
                        };
                        cur = node;
                    }
                    plant_off[g - a] = anchor - flat(fpos(g, node));
                }
                // A plant that would have to move the foot far is a fall or
                // a deliberate drag, not a skate.
                if plant_off.iter().any(|o| o.length() > LOCK_MAX * k) {
                    continue;
                }
                rep.plants_locked += 1;
                for g in a..b {
                    offset[g] = plant_off[g - a];
                    weight[g] = 1.0;
                }
                for r in 1..=LOCK_RAMP {
                    let w = smoothstep(1.0 - r as f64 / (LOCK_RAMP + 1) as f64);
                    if a >= r && w > weight[a - r] {
                        offset[a - r] = offset[a];
                        weight[a - r] = w;
                    }
                    if b - 1 + r < n && w > weight[b - 1 + r] {
                        offset[b - 1 + r] = offset[b - 1];
                        weight[b - 1 + r] = w;
                    }
                }
            }
            for g in 0..n {
                let o = offset[g].scale(weight[g]);
                if o.length() < 1e-7 {
                    continue;
                }
                let w = rig.world(&frames[g]);
                let rot = w[leg.ft].rotation();
                place_foot(rig, &mut frames[g], leg, w[leg.ft].t + o, rot);
                changed[g] = true;
                rep.lock_max_m = rep.lock_max_m.max(o.length());
            }
        }
        rep.lock_frames = changed.iter().filter(|c| **c).count();
    }

    // 4. Arms out of the body.
    if let (Some(b), Some(o)) = (body, &opts.contact) {
        rep.contact = Some(crate::contact::fix_clip(b, rig, frames, times, name, o));
    }
    for f in frames.iter_mut() {
        crate::helpers::drive(rig, f);
    }
    rep
}

fn flat(v: Vec3) -> Vec3 {
    Vec3::new(v.x, 0.0, v.z)
}

/// Nodes a cleaned clip writes: everything it animated, plus the hips,
/// legs, arms and twist helpers the passes may move.
pub fn written_nodes(rig: &Rig, map: &Mapping, anim: &Animation) -> Vec<usize> {
    let mut nodes = anim.animated_nodes();
    let mut add = |n: Option<usize>| {
        if let Some(n) = n {
            if !nodes.contains(&n) {
                nodes.push(n);
            }
        }
    };
    add(map.node_of("DEF-spine"));
    for side in ["L", "R"] {
        for b in [
            "thigh",
            "shin",
            "foot",
            "toe",
            "upper_arm",
            "forearm",
            "hand",
            "shoulder",
        ] {
            add(map.node_of(&format!("DEF-{b}.{side}")));
        }
    }
    for (h, _) in crate::helpers::present(rig) {
        add(Some(h));
    }
    nodes.sort_unstable();
    nodes
}

/// Clean clips of a rigged GLB (every clip when `clips` is empty). With
/// `suffix` the cleaned clip is added as `<name><suffix>`; without, it
/// replaces the original.
pub fn clean_glb(
    doc: &Document,
    opts: &CleanOptions,
    clips: &[String],
    suffix: Option<&str>,
) -> Result<(Document, Vec<CleanReport>), String> {
    let rig = Rig::from_doc(doc)?;
    let map = crate::humanoid::map_rig(&rig);
    map.node_of("DEF-spine").ok_or("no hips bone found")?;
    let anims = Animation::load_all(doc)?;
    for c in clips {
        if !anims.iter().any(|a| &a.name == c) {
            return Err(format!("no clip named '{}'", c));
        }
    }
    let body = match &opts.contact {
        Some(o) => Body::build(doc, &rig, o)?,
        None => None,
    };
    let height = match &body {
        Some(b) => b.height,
        None => Body::build(doc, &rig, &ContactOptions::default())?
            .map(|b| b.height)
            .unwrap_or_else(|| {
                let ys: Vec<f64> = rig.joints().iter().map(|&j| rig.head(j).y).collect();
                let lo = ys.iter().cloned().fold(f64::INFINITY, f64::min);
                let hi = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                (hi - lo) * 1.08
            }),
    };
    let mut out = doc.clone();
    let mut reports = Vec::new();
    let mut new_anims = Vec::new();
    for (ai, anim) in anims.iter().enumerate() {
        if !clips.is_empty() && !clips.contains(&anim.name) {
            continue;
        }
        let (times, mut frames) = motioncheck::sample(&rig, anim, None);
        let name = match suffix {
            Some(s) => format!("{}{}", anim.name, s),
            None => anim.name.clone(),
        };
        let mut rep = clean_clip(
            &rig,
            &map,
            body.as_ref(),
            height,
            &mut frames,
            &times,
            &anim.name,
            opts,
        );
        rep.name = name.clone();
        let nodes = written_nodes(&rig, &map, anim);
        let json = crate::stylize_body::encode_clip(&mut out, &rig, &name, &times, &nodes, &frames);
        new_anims.push((ai, json));
        reports.push(rep);
    }
    let list = out.array_mut("animations");
    for (ai, j) in new_anims {
        match suffix {
            Some(_) => list.push(j),
            None => list[ai] = j,
        }
    }
    out.compact();
    Ok((out, reports))
}
