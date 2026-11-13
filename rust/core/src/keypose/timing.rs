//! Timing between key poses, the way an animator spaces them.
//!
//! Each key has a role. A **load** (the windup's top) is reached with an
//! ease in and out and held for a few frames while it keeps creeping
//! further along the windup (a moving hold). The **strike** is a pass
//! key: the move from the load is *fast in* (accelerating, then at full
//! speed through the contact; 2-4 frames at 30 fps: the smear) and runs
//! through it without stopping; the follow-through leaves at that speed
//! (its frames scale with the blade's travel in it). The
//! **follow-through** is reached easing out, overshoots by a share of
//! its last move and settles back, then holds. A plain **pose** eases in
//! and out with a short hold. The clip starts from and returns to the
//! rig's idle when there is one.
//!
//! Rotations are interpolated per bone in local space with squad
//! (spherical cubic): through a pass key the tangent is continuous, so
//! the sword hand sweeps on an arc instead of turning a corner; a stop
//! key has no tangent (the motion turns round there). The hips'
//! translation follows the same spacing.
//!
//! Feet: between two keys a foot that stays (on the floor in both, moved
//! less than `step_threshold`) is locked where it was; one that moved
//! steps: it lifts after the hold before, arcs `step_height` up and is
//! down at the destination key. The other foot is the anchor that fixes
//! where the drawn key stands (the later key's rear foot when a step is
//! drawn: it pushes, the front foot steps out). Feet
//! are put on their marks by two-bone IK each frame (the hips lowered
//! when a leg cannot reach).
//!
//! Anticipation: when the drawings show no windup (the sword hand's tip
//! does not move against the strike by `windup_share` of the strike
//! before it), the load's hold eases `anticipation` of the strike's
//! rotation back against it instead of creeping.

use super::fit::{
    cap_wrists, grip_with, knees_over_toes, move_root, needed_drop, plant_feet_with, set_world_rot,
    FitParams, RigInfo,
};
use crate::detmath;
use crate::math::{Quat, Vec3};
use crate::rig::Trs;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Role {
    /// The pose the clip starts or ends on (idle): eased, no hold.
    Rest,
    Load,
    Strike,
    Follow,
    Pose,
}

impl Role {
    pub fn parse(s: &str) -> Option<Role> {
        Some(match s {
            "rest" | "idle" => Role::Rest,
            "load" | "windup" => Role::Load,
            "strike" | "contact" => Role::Strike,
            "follow" | "follow-through" => Role::Follow,
            "pose" | "key" => Role::Pose,
            _ => return None,
        })
    }
    pub fn name(self) -> &'static str {
        match self {
            Role::Rest => "rest",
            Role::Load => "load",
            Role::Strike => "strike",
            Role::Follow => "follow",
            Role::Pose => "pose",
        }
    }
}

#[derive(Clone, Debug)]
pub struct TimingParams {
    pub fps: f64,
    /// Frames from the start pose to the first key.
    pub windup_frames: usize,
    pub hold_load: usize,
    /// Load to strike: the fast part (the smear), 2-4 at 30 fps.
    pub strike_frames: usize,
    pub follow_frames: usize,
    pub hold_follow: usize,
    pub recover_frames: usize,
    pub pose_frames: usize,
    pub hold_pose: usize,
    /// Follow-through overshoot, share of the strike-to-follow move's last
    /// tenth carried on (1 = one more tenth).
    pub overshoot: f64,
    pub settle_frames: usize,
    /// Moving hold on a load: it coils back against the strike by this
    /// share of the strike's rotation (the drawings' windup is kept).
    pub creep: f64,
    pub anticipation: f64,
    pub windup_share: f64,
    /// Metres on a 1.75 m body.
    pub step_threshold: f64,
    pub step_height: f64,
    /// A foot that moved less than this (m, 1.75 m body) between keys is
    /// locked; more is a step (drawn keys may differ a little).
    pub lock_slack: f64,
}

impl Default for TimingParams {
    fn default() -> Self {
        TimingParams {
            fps: 30.0,
            windup_frames: 9,
            hold_load: 4,
            strike_frames: 3,
            follow_frames: 4,
            hold_follow: 7,
            recover_frames: 10,
            pose_frames: 6,
            hold_pose: 2,
            overshoot: 0.6,
            settle_frames: 3,
            creep: 0.06,
            anticipation: 0.15,
            windup_share: 0.3,
            step_threshold: 0.10,
            step_height: 0.07,
            lock_slack: 0.03,
        }
    }
}

/// A key pose ready to time.
#[derive(Clone, Debug)]
pub struct Key {
    pub locals: Vec<Option<Trs>>,
    pub role: Role,
    /// Feet on the floor in this pose.
    pub planted: [bool; 2],
    /// From the drawings (both hands on the grip apply), not the idle.
    pub drawn: bool,
    /// Off hand relative to the sword socket, and the grip gap (two hands).
    pub off_rel: Option<Quat>,
    pub grip_gap: f64,
}

#[derive(Clone, Debug, Default)]
pub struct TimingReport {
    pub key_frames: Vec<usize>,
    pub frames: usize,
    pub windup_found: bool,
    pub counter_added: bool,
    /// `(foot, from key, to key, distance m)`.
    pub steps: Vec<(String, usize, usize, f64)>,
    /// Horizontal offset given to each key (where it stands), metres.
    pub offsets: Vec<Vec3>,
    /// Feet locked but drawn this far apart between keys (m), worst.
    pub lock_slide_max: f64,
    /// Frames given to the follow-through (speed carried from the strike).
    pub follow_frames: usize,
}

pub fn ease_in_out(u: f64) -> f64 {
    u * u * (3.0 - 2.0 * u)
}

/// Fast in: accelerating all the way (the strike).
pub fn ease_in(u: f64) -> f64 {
    u * u
}

pub fn ease_out(u: f64) -> f64 {
    1.0 - (1.0 - u) * (1.0 - u)
}

/// The strike: accelerating over its first half, then at full speed
/// through the contact (no braking before the hit).
pub fn strike_in(x: f64) -> f64 {
    let a = 0.5;
    let c = 1.0 / (1.0 - 0.5 * a);
    if x < a {
        c * x * x / (2.0 * a)
    } else {
        c * (0.5 * a + x - a)
    }
}

/// The follow-through: leaving the contact at full speed, braking over
/// its second half.
pub fn follow_out(x: f64) -> f64 {
    1.0 - strike_in(1.0 - x)
}

fn qlog(q: Quat) -> Vec3 {
    let q = if q.w < 0.0 { q.neg() } else { q };
    let v = Vec3::new(q.x, q.y, q.z);
    let s = v.length();
    if s < 1e-12 {
        return Vec3::ZERO;
    }
    let a = detmath::acos(q.w.clamp(-1.0, 1.0));
    v.scale(a / s)
}

fn qexp(v: Vec3) -> Quat {
    let a = v.length();
    if a < 1e-12 {
        return Quat::IDENTITY;
    }
    let (s, c) = detmath::sin_cos(a);
    let n = v.scale(s / a);
    Quat::new(c, n.x, n.y, n.z)
}

fn align(q: Quat, to: Quat) -> Quat {
    if q.dot(to) < 0.0 {
        q.neg()
    } else {
        q
    }
}

/// Squad control point at `q` between `prev` and `next`.
fn squad_ctrl(prev: Quat, q: Quat, next: Quat) -> Quat {
    let prev = align(prev, q);
    let next = align(next, q);
    let inv = q.conj();
    let a = qlog(inv.mul(next));
    let b = qlog(inv.mul(prev));
    q.mul(qexp((a + b).scale(-0.25))).normalized()
}

fn squad(a: Quat, b: Quat, ca: Quat, cb: Quat, u: f64) -> Quat {
    let b = align(b, a);
    let p = a.slerp(b, u);
    let q = ca.slerp(align(cb, ca), u);
    p.slerp(align(q, p), 2.0 * u * (1.0 - u))
}

/// Root translation, hips' local.
fn root_t(info: &RigInfo, l: &[Option<Trs>]) -> Vec3 {
    l[info.hips].unwrap_or(info.rig.rest[info.hips]).t
}

/// Ankle world positions and foot world rotations of a pose.
fn feet(info: &RigInfo, l: &[Option<Trs>]) -> ([Vec3; 2], [Quat; 2]) {
    let w = info.rig.world(l);
    (
        [w[info.foot[0]].t, w[info.foot[1]].t],
        [w[info.foot[0]].rotation(), w[info.foot[1]].rotation()],
    )
}

#[derive(Clone, Copy, Debug)]
enum FootPlan {
    Free,
    Lock(Vec3, Quat),
    Step {
        from: (Vec3, Quat),
        to: (Vec3, Quat),
    },
}

/// One output frame's recipe.
#[derive(Clone, Copy, Debug)]
struct Sample {
    seg: usize,
    /// Spacing parameter along the segment (may pass 1: overshoot, creep).
    u: f64,
}

struct Segment {
    a: usize,
    b: usize,
}

/// Build the clip: per frame, local transforms of every node.
pub fn build(
    info: &RigInfo,
    keys: &[Key],
    params: &TimingParams,
    fit: &FitParams,
    two_hands: bool,
) -> (Vec<Vec<Option<Trs>>>, TimingReport) {
    let rig = &info.rig;
    let n_nodes = rig.names.len();
    let mut keys: Vec<Key> = keys.to_vec();
    let mut rep = TimingReport::default();
    let k = info.k;
    if keys.len() < 2 {
        return (vec![keys[0].locals.clone()], rep);
    }

    // Where each key stands: anchor feet stay put between keys.
    let mut offsets = vec![Vec3::ZERO; keys.len()];
    let mut plans: Vec<[FootPlan; 2]> = Vec::new(); // per transition
    for i in 0..keys.len() {
        let (cur, cur_rot) = feet(info, &keys[i].locals);
        if i == 0 {
            continue;
        }
        // The earlier key already stands where it was placed.
        let (prev_l, prev_rot) = feet(info, &keys[i - 1].locals);
        let prev: Vec<Vec3> = prev_l.to_vec();
        let both = keys[i - 1].planted[0]
            && keys[i - 1].planted[1]
            && keys[i].planted[0]
            && keys[i].planted[1];
        let last_rest = i == keys.len() - 1 && keys[i].role == Role::Rest;
        let flat = |v: Vec3| Vec3::new(v.x, 0.0, v.z);
        let o = if last_rest {
            Vec3::ZERO
        } else if both {
            let m = flat((prev[0] - cur[0]) + (prev[1] - cur[1])).scale(0.5);
            let r = (0..2)
                .map(|s| flat(prev[s] - (cur[s] + m)).length())
                .fold(0.0, f64::max);
            if r < params.step_threshold * k {
                m
            } else {
                // A step: the later key's rear foot anchors (it pushes,
                // the front foot steps out toward the target).
                let a = if cur[0].dot(info.fwd) <= cur[1].dot(info.fwd) {
                    0
                } else {
                    1
                };
                flat(prev[a] - cur[a])
            }
        } else {
            match (0..2).find(|&s| keys[i - 1].planted[s] && keys[i].planted[s]) {
                Some(a) => flat(prev[a] - cur[a]),
                None => flat(prev[0] + prev[1] - cur[0] - cur[1]).scale(0.5),
            }
        };
        offsets[i] = o;
        let mut plan = [FootPlan::Free; 2];
        for s in 0..2 {
            if !(keys[i - 1].planted[s] && keys[i].planted[s]) {
                continue;
            }
            let to = cur[s] + o;
            let d = flat(to - prev[s]).length();
            if d <= params.lock_slack * k || (d < params.step_threshold * k && !last_rest && both) {
                rep.lock_slide_max = rep.lock_slide_max.max(d);
                plan[s] = FootPlan::Lock(prev[s], prev_rot[s]);
            } else {
                rep.steps.push((
                    ["L", "R"][s].to_string(),
                    i - 1,
                    i,
                    (d * 1000.0).round() / 1000.0,
                ));
                plan[s] = FootPlan::Step {
                    from: (prev[s], prev_rot[s]),
                    to: (to, cur_rot[s]),
                };
            }
        }
        plans.push(plan);
        // The locked feet define this key's marks from now on.
        let mut l = keys[i].locals.clone();
        crate::keypose::fit::move_root(info, &mut l, o);
        for s in 0..2 {
            if let FootPlan::Lock(p, r) = plan[s] {
                set_world_rot(rig, &mut l, info.foot[s], r);
                let _ = p;
            }
        }
        let marks: Vec<Vec3> = (0..2)
            .map(|s| match plan[s] {
                FootPlan::Lock(p, _) => p,
                _ => cur[s] + o,
            })
            .collect();
        let pl = [
            !matches!(plan[0], FootPlan::Free) && keys[i].planted[0],
            !matches!(plan[1], FootPlan::Free) && keys[i].planted[1],
        ];
        let knees = knees_over_toes(info, &l);
        plant_feet_with(info, &mut l, &pl, &marks, Some(knees), false);
        keys[i].locals = l;
        // Offsets are now baked into the key's root.
        offsets[i] = o;
    }
    rep.offsets = offsets.clone();

    // Windup in the drawings?
    let sword = fit.sword_hand.unwrap_or(1);
    let tip = |l: &[Option<Trs>]| {
        super::fit::blade_tip(info, l, sword, 0.9 * k)
            .unwrap_or_else(|| info.rig.world(l)[info.hand[sword]].t)
    };
    let mut counter_at: Option<usize> = None;
    for i in 0..keys.len() {
        if keys[i].role != Role::Load || i + 1 >= keys.len() {
            continue;
        }
        let tl = tip(&keys[i].locals);
        let ts = tip(&keys[i + 1].locals);
        let strike = ts - tl;
        let found = if i > 0 {
            let back = tl - tip(&keys[i - 1].locals);
            back.dot(strike.normalized()) <= -params.windup_share * strike.length()
        } else {
            false
        };
        rep.windup_found |= found;
        if !found {
            counter_at = Some(i);
        }
    }

    // Exit pose of each load (where the strike starts): during its hold
    // it coils back against the strike, a little (creep) when the
    // drawings already wind up, more (the counter) when they do not.
    let mut exit: Vec<Vec<Option<Trs>>> = keys.iter().map(|k| k.locals.clone()).collect();
    let mut coil = vec![false; keys.len()];
    for i in 0..keys.len() {
        if keys[i].role != Role::Load || i + 1 >= keys.len() {
            continue;
        }
        let amount = if counter_at == Some(i) {
            rep.counter_added = true;
            params.anticipation
        } else {
            params.creep
        };
        if amount <= 0.0 {
            continue;
        }
        let mut l = keys[i].locals.clone();
        for n in 0..n_nodes {
            if let (Some(a), Some(b)) = (keys[i].locals[n], keys[i + 1].locals[n]) {
                let mut t = a;
                t.r = b.r.slerp(a.r, 1.0 + amount);
                l[n] = Some(t);
            }
        }
        exit[i] = l;
        coil[i] = true;
    }

    // Segments and frame samples.
    let segs: Vec<Segment> = (1..keys.len())
        .map(|i| Segment { a: i - 1, b: i })
        .collect();
    // The follow-through leaves the contact as fast as the strike reached
    // it: its frames scale with how far the blade travels in it.
    let travel = |si: usize| {
        let mut d = 0.0;
        let mut last = tip(&pose_at(info, &keys, &exit, &segs, si, 0.0));
        for i in 1..=8 {
            let p = tip(&pose_at(info, &keys, &exit, &segs, si, i as f64 / 8.0));
            d += (p - last).length();
            last = p;
        }
        d
    };
    let mut follow_frames = vec![params.follow_frames; segs.len()];
    for si in 1..segs.len() {
        let (a, b) = (&segs[si - 1], &segs[si]);
        if keys[a.a].role == Role::Load
            && keys[a.b].role == Role::Strike
            && keys[b.b].role == Role::Follow
        {
            let (d1, d2) = (travel(si - 1), travel(si));
            if d1 > 1e-6 {
                let t = params.strike_frames as f64 * d2 / d1;
                follow_frames[si] = (t.round() as usize).clamp(2, params.follow_frames.max(2) * 2);
            }
        }
    }
    rep.follow_frames = (0..segs.len())
        .find(|&si| keys[segs[si].a].role == Role::Strike && keys[segs[si].b].role == Role::Follow)
        .map(|si| follow_frames[si])
        .unwrap_or(0);
    let mut samples: Vec<Sample> = Vec::new();
    let mut key_frames = vec![0usize];
    let mut holds = vec![0usize; keys.len()];
    samples.push(Sample { seg: 0, u: 0.0 });
    for (si, sg) in segs.iter().enumerate() {
        let (ra, rb) = (keys[sg.a].role, keys[sg.b].role);
        let (frames, curve): (usize, fn(f64) -> f64) = match (ra, rb) {
            (Role::Load, Role::Strike) => (params.strike_frames, strike_in),
            (Role::Strike, Role::Follow) => (follow_frames[si], follow_out),
            (Role::Load, Role::Follow) => {
                (params.strike_frames + params.follow_frames, ease_in_out)
            }
            (_, Role::Load) => (params.windup_frames, ease_in_out),
            (_, Role::Rest) => (params.recover_frames, ease_in_out),
            _ => (params.pose_frames, ease_in_out),
        };
        let frames = frames.max(1);
        let peak = if rb == Role::Follow {
            1.0 + params.overshoot * 0.1
        } else {
            1.0
        };
        for f in 1..=frames {
            let x = f as f64 / frames as f64;
            samples.push(Sample {
                seg: si,
                u: curve(x) * peak,
            });
        }
        key_frames.push(samples.len() - 1);
        // Holds.
        let (hold, from, to): (usize, f64, f64) = match rb {
            Role::Load => (params.hold_load, 1.0, 1.0),
            Role::Follow => (params.hold_follow, peak, 1.0),
            Role::Pose => (params.hold_pose, 1.0, 1.0),
            _ => (0, 1.0, 1.0),
        };
        for f in 1..=hold {
            let x = f as f64 / hold as f64;
            let settle = if rb == Role::Follow {
                (f as f64 / params.settle_frames.max(1) as f64).min(1.0)
            } else {
                x
            };
            samples.push(Sample {
                seg: si,
                u: from + (to - from) * ease_in_out(settle),
            });
        }
        holds[sg.b] = hold;
    }
    rep.key_frames = key_frames.clone();
    rep.frames = samples.len();

    // The counter shows during the load's hold: blend into it.
    let mut frames_out: Vec<Vec<Option<Trs>>> = Vec::with_capacity(samples.len());
    for (f, s) in samples.iter().enumerate() {
        let mut l = pose_at(info, &keys, &exit, &segs, s.seg, s.u);
        for ci in 0..keys.len() {
            if !coil[ci] {
                continue;
            }
            let hold_start = key_frames[ci];
            let hold_end = hold_start + holds[ci];
            if f > hold_start && f <= hold_end {
                let x = (f - hold_start) as f64 / holds[ci].max(1) as f64;
                l = blend(&keys[ci].locals, &exit[ci], ease_in_out(x));
            }
        }
        frames_out.push(l);
    }

    // Feet on their marks and both hands on the grip, frame by frame. A
    // transition's feet move from the end of the earlier key's hold (from
    // its start when the later key is a strike: the lunge begins in the
    // windup's hold) to the later key.
    let last = keys.len() - 1;
    let n_out = frames_out.len();
    let mut marks_all = vec![([Vec3::ZERO; 2], [false; 2]); n_out];
    let mut where_ = vec![(1usize, 0.0f64); n_out];
    for (f, l) in frames_out.iter_mut().enumerate() {
        let i = (1..=last).find(|&i| f <= key_frames[i]).unwrap_or(last);
        let start = key_frames[i - 1]
            + if keys[i].role == Role::Strike {
                0
            } else {
                holds[i - 1]
            };
        let span = key_frames[i].saturating_sub(start).max(1) as f64;
        let x = ((f as f64 - start as f64) / span).clamp(0.0, 1.0);
        where_[f] = (i, x);
        let plan = plans[i - 1];
        let mut marks = [Vec3::ZERO; 2];
        let mut pl = [false; 2];
        for (sd, p) in plan.iter().enumerate() {
            match *p {
                FootPlan::Free => {}
                FootPlan::Lock(pos, rot) => {
                    set_world_rot(rig, l, info.foot[sd], rot);
                    marks[sd] = pos;
                    pl[sd] = true;
                }
                FootPlan::Step { from, to } => {
                    let e = ease_in_out(x);
                    let d = (to.0 - from.0).length();
                    let lift =
                        (params.step_height * k * (4.0 * x * (1.0 - x))).min(0.5 * d + 0.01 * k);
                    marks[sd] = from.0.lerp(to.0, e) + Vec3::new(0.0, lift, 0.0);
                    set_world_rot(rig, l, info.foot[sd], from.1.slerp(to.1, e));
                    pl[sd] = true;
                }
            }
        }
        marks_all[f] = (marks, pl);
    }
    // The hips come down where a leg cannot reach its mark; the drop is
    // spread over the neighbouring frames (widest within 2, then smoothed)
    // so the body sinks into it instead of jolting.
    let raw: Vec<f64> = (0..n_out)
        .map(|f| needed_drop(info, &frames_out[f], &marks_all[f].1, &marks_all[f].0).max(0.0))
        .collect();
    let mut drop: Vec<f64> = (0..n_out)
        .map(|f| {
            let (a, b) = (f.saturating_sub(2), (f + 2).min(n_out - 1));
            raw[a..=b].iter().cloned().fold(0.0, f64::max)
        })
        .collect();
    for _ in 0..2 {
        let d = drop.clone();
        for f in 0..n_out {
            let (a, b) = (f.saturating_sub(1), (f + 1).min(n_out - 1));
            drop[f] = (d[a] + 2.0 * d[f] + d[b]) / 4.0;
        }
    }
    for (f, l) in frames_out.iter_mut().enumerate() {
        if drop[f] > 0.0 {
            move_root(info, l, Vec3::new(0.0, -drop[f], 0.0));
        }
        let (marks, pl) = marks_all[f];
        let knees = knees_over_toes(info, l);
        plant_feet_with(info, l, &pl, &marks, Some(knees), false);
        let (i, x) = where_[f];
        if two_hands {
            if let Some(sw) = fit.sword_hand {
                let (da, db) = (keys[i - 1].drawn, keys[i].drawn);
                let w = match (da, db) {
                    (true, true) => 1.0,
                    (false, true) => ease_in_out(x),
                    (true, false) => 1.0 - ease_in_out(x),
                    (false, false) => 0.0,
                };
                let ra = keys[i - 1].off_rel.or(keys[i].off_rel);
                let rb = keys[i].off_rel.or(keys[i - 1].off_rel);
                if let (true, Some(ra), Some(rb)) = (w > 0.0, ra, rb) {
                    let ga = if da {
                        keys[i - 1].grip_gap
                    } else {
                        keys[i].grip_gap
                    };
                    let gb = if db {
                        keys[i].grip_gap
                    } else {
                        keys[i - 1].grip_gap
                    };
                    let free = l.clone();
                    grip_with(info, l, sw, ga + (gb - ga) * x, ra.slerp(rb, x), fit);
                    cap_wrists(info, l, fit);
                    if w < 1.0 {
                        let o = 1 - sw;
                        for n in [info.upper[o], info.fore[o], info.hand[o]] {
                            if let (Some(a), Some(b)) = (free[n], l[n]) {
                                let mut t = a;
                                t.r = a.r.slerp(b.r, w);
                                l[n] = Some(t);
                            }
                        }
                    }
                }
            }
        }
    }
    for l in frames_out.iter_mut() {
        cap_wrists(info, l, fit);
    }
    (frames_out, rep)
}

fn blend(a: &[Option<Trs>], b: &[Option<Trs>], x: f64) -> Vec<Option<Trs>> {
    a.iter()
        .zip(b)
        .map(|(p, q)| match (p, q) {
            (Some(p), Some(q)) => Some(Trs {
                t: p.t.lerp(q.t, x),
                r: p.r.slerp(q.r, x),
                s: p.s,
            }),
            (p, None) => *p,
            (None, q) => *q,
        })
        .collect()
}

/// The pose at spacing `u` along segment `si` (u past 1 carries on along
/// the arrival direction: overshoot, creep).
fn pose_at(
    info: &RigInfo,
    keys: &[Key],
    exit: &[Vec<Option<Trs>>],
    segs: &[Segment],
    si: usize,
    u: f64,
) -> Vec<Option<Trs>> {
    let rig = &info.rig;
    let sg = &segs[si];
    let (a, b) = (sg.a, sg.b);
    let n = rig.names.len();
    let mut out = vec![None; n];
    let pass = |i: usize| keys[i].role == Role::Strike && i > 0 && i + 1 < keys.len();
    let h = 0.1;
    for node in 0..n {
        let qa_t = exit[a][node].unwrap_or(rig.rest[node]);
        let qb_t = keys[b].locals[node].unwrap_or(rig.rest[node]);
        if exit[a][node].is_none() && keys[b].locals[node].is_none() {
            continue;
        }
        let (qa, qb) = (qa_t.r, align(qb_t.r, qa_t.r));
        let ca = if pass(a) {
            let p = keys[a - 1].locals[node].unwrap_or(rig.rest[node]).r;
            squad_ctrl(p, qa, qb)
        } else {
            qa
        };
        let cb = if pass(b) {
            let nx = exit[b + 1][node].unwrap_or(rig.rest[node]).r;
            squad_ctrl(qa, qb, nx)
        } else {
            qb
        };
        let r = if u <= 1.0 {
            squad(qa, qb, ca, cb, u.max(0.0))
        } else {
            let near = squad(qa, qb, ca, cb, 1.0 - h);
            near.slerp(qb, 1.0 + (u - 1.0) / h)
        };
        let ut = u.clamp(0.0, 1.0);
        // Translation: cubic Hermite, a tangent through pass keys.
        let ma = if pass(a) {
            (qb_t.t - keys[a - 1].locals[node].unwrap_or(rig.rest[node]).t).scale(0.5)
        } else {
            Vec3::ZERO
        };
        let mb = if pass(b) {
            (exit[b + 1][node].unwrap_or(rig.rest[node]).t - qa_t.t).scale(0.5)
        } else {
            Vec3::ZERO
        };
        let (u2, u3) = (ut * ut, ut * ut * ut);
        let tr = qa_t.t.scale(2.0 * u3 - 3.0 * u2 + 1.0)
            + ma.scale(u3 - 2.0 * u2 + ut)
            + qb_t.t.scale(-2.0 * u3 + 3.0 * u2)
            + mb.scale(u3 - u2);
        out[node] = Some(Trs {
            t: tr,
            r: r.normalized(),
            s: qb_t.s,
        });
    }
    let _ = root_t;
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curves_start_end_and_shape() {
        for c in [ease_in, ease_out, ease_in_out, strike_in, follow_out] {
            assert!(c(0.0).abs() < 1e-12 && (c(1.0) - 1.0).abs() < 1e-12);
            let mut last = 0.0;
            for i in 1..=10 {
                let v = c(i as f64 / 10.0);
                assert!(v >= last);
                last = v;
            }
        }
        // Fast in: the strike covers most ground in its last frame.
        let f3: Vec<f64> = (1..=3).map(|i| ease_in(i as f64 / 3.0)).collect();
        assert!(f3[2] - f3[1] > f3[1] - f3[0] && f3[1] - f3[0] > f3[0]);
        // Ease out: the first frame covers most.
        assert!(ease_out(1.0 / 3.0) > 1.0 / 3.0);
    }

    #[test]
    fn squad_hits_keys_and_log_exp_invert() {
        let a = Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 0.3);
        let b = Quat::from_axis_angle(Vec3::new(1.0, 0.2, 0.0).normalized(), 1.1);
        let c = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), -0.8);
        let ca = squad_ctrl(a, b, c);
        assert!(squad(a, b, ca, b, 0.0).approx_eq(a, 1e-9));
        assert!(squad(a, b, ca, b, 1.0).approx_eq(b, 1e-9));
        let q = Quat::from_axis_angle(Vec3::new(0.3, -0.4, 0.5).normalized(), 2.0);
        assert!(qexp(qlog(q)).approx_eq(q, 1e-12));
    }
}
