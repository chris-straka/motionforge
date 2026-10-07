//! Foot pinning for GLB retargets (`animate`).
//!
//! Copying rotations and scaling the hips' travel by the hip-height ratio
//! reproduces the source's foot plants only when the legs have the same
//! proportions; otherwise a planted foot drifts (bench: 2-3% of a leg per
//! plant over the source's own drift, 6% in a jog). This pass finds the
//! plants on the source (ankle near the clip's lowest height and nearly
//! still) and moves the target's ankle, with two-bone leg IK, to where the
//! source's ankle would be on the target: one anchor per plant plus the
//! source's own drift inside it (heel roll), turned and scaled like the
//! hips. So the target's feet do exactly what the source's feet did; a
//! source that slides still slides, a planted foot stays planted.
//! Blends in and out over a few frames; the foot keeps its world rotation.

use crate::animate::{Mapped, Transfer};
use crate::contact::solve_arm;
use crate::math::Vec3;
use crate::rig::{Animation, Trs};

/// Plant: ankle within this share of the source's leg of its lowest
/// height in the clip...
const PLANT_HEIGHT: f64 = 0.06;
/// ...and moving slower than this share of a leg per second.
const PLANT_SPEED: f64 = 0.35;
/// Shortest plant, frames.
const PLANT_MIN_FRAMES: usize = 3;
/// Blend in/out, seconds.
const RAMP_S: f64 = 0.12;

#[derive(Clone, Debug, Default)]
pub struct PinReport {
    pub plants: usize,
    pub frames_changed: usize,
    /// Largest ankle move, metres.
    pub max_shift: f64,
}

fn leg(m: &Mapped, side: &str) -> Option<(usize, usize, usize)> {
    Some((
        m.map.node_of(&format!("DEF-thigh.{}", side))?,
        m.map.node_of(&format!("DEF-shin.{}", side))?,
        m.map.node_of(&format!("DEF-foot.{}", side))?,
    ))
}

fn flat(v: Vec3) -> Vec3 {
    Vec3::new(v.x, 0.0, v.z)
}

/// Frame intervals `[a, b)` where the source foot is planted.
fn plants(ankle: &[Vec3], times: &[f64], leg_len: f64) -> Vec<(usize, usize)> {
    let n = ankle.len();
    if n < PLANT_MIN_FRAMES {
        return Vec::new();
    }
    let low = ankle.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
    let speed = |f: usize| -> f64 {
        let (a, b) = (f.saturating_sub(1), (f + 1).min(n - 1));
        let dt = times[b] - times[a];
        if dt <= 0.0 {
            return 0.0;
        }
        flat(ankle[b] - ankle[a]).length() / dt
    };
    let on: Vec<bool> = (0..n)
        .map(|f| ankle[f].y - low < PLANT_HEIGHT * leg_len && speed(f) < PLANT_SPEED * leg_len)
        .collect();
    let mut out = Vec::new();
    let mut start = None;
    for f in 0..=n {
        let x = f < n && on[f];
        match (x, start) {
            (true, None) => start = Some(f),
            (false, Some(s)) => {
                if f - s >= PLANT_MIN_FRAMES {
                    out.push((s, f));
                }
                start = None;
            }
            _ => {}
        }
    }
    out
}

fn smooth(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/// Pin the target's feet in `posed` (one local-override set per frame at
/// `times`, as `Transfer::frame` produced them).
pub fn pin_feet(
    src: &Mapped,
    tgt: &Mapped,
    transfer: &Transfer,
    anim: &Animation,
    times: &[f64],
    posed: &mut [Vec<Option<Trs>>],
) -> PinReport {
    let mut report = PinReport::default();
    let n = posed.len();
    if n < PLANT_MIN_FRAMES || times.len() != n {
        return report;
    }
    let sworld: Vec<_> = times
        .iter()
        .map(|&t| src.rig.world(&anim.sample(&src.rig, t)))
        .collect();
    let dt = (times[n - 1] - times[0]) / (n - 1) as f64;
    let ramp = if dt > 0.0 {
        ((RAMP_S / dt).round() as usize).max(1)
    } else {
        1
    };
    let mut changed = vec![false; n];
    for side in ["L", "R"] {
        let (Some((sth, ssh, sft)), Some((tth, tsh, tft))) = (leg(src, side), leg(tgt, side))
        else {
            continue;
        };
        let srig = &src.rig;
        let leg_s =
            (srig.head(ssh) - srig.head(sth)).length() + (srig.head(sft) - srig.head(ssh)).length();
        if leg_s < 1e-9 {
            continue;
        }
        let s_ankle: Vec<Vec3> = sworld.iter().map(|w| w[sft].t).collect();
        let t_ankle: Vec<Vec3> = posed.iter().map(|l| tgt.rig.world(l)[tft].t).collect();
        // Per frame: horizontal ankle offset and its blend weight.
        let mut offset = vec![Vec3::ZERO; n];
        let mut weight = vec![0.0f64; n];
        for (a, b) in plants(&s_ankle, times, leg_s) {
            report.plants += 1;
            let k = (b - a) as f64;
            let mean_s = s_ankle[a..b]
                .iter()
                .fold(Vec3::ZERO, |acc, p| acc + flat(*p))
                .scale(1.0 / k);
            let drift: Vec<Vec3> = (a..b)
                .map(|f| {
                    transfer
                        .yaw
                        .rotate_vec(flat(s_ankle[f]) - mean_s)
                        .scale(transfer.scale)
                })
                .collect();
            let anchor = (a..b)
                .fold(Vec3::ZERO, |acc, f| acc + flat(t_ankle[f]) - drift[f - a])
                .scale(1.0 / k);
            for f in a..b {
                offset[f] = anchor + drift[f - a] - flat(t_ankle[f]);
                weight[f] = 1.0;
            }
            for r in 1..=ramp {
                let w = smooth(1.0 - r as f64 / (ramp + 1) as f64);
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
        for f in 0..n {
            let o = offset[f].scale(weight[f]);
            if o.length() < 1e-7 {
                continue;
            }
            solve_arm(&tgt.rig, &mut posed[f], tth, tsh, tft, t_ankle[f] + o);
            changed[f] = true;
            report.max_shift = report.max_shift.max(o.length());
        }
    }
    report.frames_changed = changed.iter().filter(|c| **c).count();
    report
}
