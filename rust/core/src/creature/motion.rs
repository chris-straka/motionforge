//! Random poses for the joint finder's training renders, and random
//! smooth motions for the exam. Every bone bends only within its
//! profile limit; the root turns and shifts a little and stays on the
//! spot.

use super::profile::Creature;
use super::rng::Rng;
use crate::clip::{Clip, Frame, Pose};
use crate::detmath;
use crate::math::{Quat, Vec3};

const TAU: f64 = std::f64::consts::PI * 2.0;

/// A random pose: each bone rotated about a random axis by a random
/// fraction of its limit (biased toward larger bends; `scale` shrinks
/// everything), the root shifted by up to `root_shift` metres.
fn random_pose(c: &Creature, rng: &mut Rng, scale: f64, root_shift: f64) -> Frame {
    let n = c.skeleton.len();
    let mut poses = Vec::with_capacity(n);
    for i in 0..n {
        let axis = rng.unit_vec();
        let u = rng.unit();
        let frac = 1.0 - (1.0 - u) * (1.0 - u); // biased toward the limit
        let angle = c.max_deg[i].to_radians() * frac * scale;
        let quat = Quat::from_axis_angle(axis, angle);
        let loc = if c.skeleton.bones[i].parent.is_none() {
            Vec3::new(
                rng.range(-root_shift, root_shift),
                rng.range(-root_shift, root_shift),
                rng.range(-root_shift, root_shift) * 0.5,
            )
        } else {
            Vec3::ZERO
        };
        poses.push(Pose { loc, quat });
    }
    Frame { poses }
}

/// Training poses. Sample `k` occupies frames 3k, 3k+1, 3k+2 = (A, M, B):
/// M is the pose to learn; for a fast sample A and B are M moved part
/// way toward two other random poses, so a render of frame 3k+1 with a
/// one-frame shutter blurs half way to each. Slow samples have A = M = B.
pub fn training_poses(c: &Creature, count: usize, seed: u64, fast_ratio: f64) -> Clip {
    let mut rng = Rng::new(seed);
    let root_shift = 0.05 * c.height;
    let mut frames = Vec::with_capacity(count * 3);
    for _ in 0..count {
        let m = random_pose(c, &mut rng, 1.0, root_shift);
        let fast = rng.unit() < fast_ratio;
        if fast {
            let a_target = random_pose(c, &mut rng, 1.0, root_shift);
            let b_target = random_pose(c, &mut rng, 1.0, root_shift);
            let blend = 0.35;
            let a = blend_frames(&m, &a_target, blend);
            let b = blend_frames(&m, &b_target, blend);
            frames.push(a);
            frames.push(m);
            frames.push(b);
        } else {
            frames.push(m.clone());
            frames.push(m.clone());
            frames.push(m);
        }
    }
    Clip {
        fps: 24.0,
        skeleton: c.skeleton.clone(),
        frames,
    }
}

fn blend_frames(a: &Frame, b: &Frame, t: f64) -> Frame {
    Frame {
        poses: a
            .poses
            .iter()
            .zip(&b.poses)
            .map(|(p, q)| Pose {
                loc: p.loc.lerp(q.loc, t),
                quat: p.quat.slerp(q.quat, t),
            })
            .collect(),
    }
}

/// The exam motion: per bone two sinusoids about two random axes whose
/// amplitudes sum to at most `amplitude` of the limit (leg bones that
/// carry feet swing less, so feet stay low and plant), the root bobbing
/// and swaying by a few percent of the height and turning a little.
pub fn exam_motion(c: &Creature, frames: usize, fps: f64, seed: u64, amplitude: f64) -> Clip {
    let mut rng = Rng::new(seed);
    let n = c.skeleton.len();
    // Leg bones: ancestors chains of the feet.
    let mut leg = vec![false; n];
    for &f in &c.feet {
        for b in c.leg_chain(f, 4) {
            leg[b] = true;
        }
    }
    struct Osc {
        axis: Vec3,
        amp: f64,
        freq: f64,
        phase: f64,
    }
    let mut bones: Vec<[Osc; 2]> = Vec::with_capacity(n);
    for i in 0..n {
        let limit = c.max_deg[i].to_radians() * amplitude;
        let scale = if leg[i] { 0.45 } else { 1.0 };
        let split = rng.range(0.5, 0.85);
        let mk = |rng: &mut Rng, share: f64| Osc {
            axis: rng.unit_vec(),
            amp: limit * share * scale * rng.range(0.6, 1.0),
            freq: rng.range(0.25, 1.1),
            phase: rng.range(0.0, TAU),
        };
        bones.push([mk(&mut rng, split), mk(&mut rng, 1.0 - split)]);
    }
    let bob = 0.02 * c.height;
    let sway = 0.015 * c.height;
    let root_freq = rng.range(0.4, 0.9);
    let root_phase = rng.range(0.0, TAU);
    let yaw_amp = (c.max_deg[c.root()] * 0.5).to_radians() * amplitude;
    let mut out = Vec::with_capacity(frames);
    for f in 0..frames {
        let t = f as f64 / fps;
        let mut poses = Vec::with_capacity(n);
        for (i, osc) in bones.iter().enumerate() {
            let q0 = Quat::from_axis_angle(
                osc[0].axis,
                osc[0].amp * detmath::sin(TAU * osc[0].freq * t + osc[0].phase),
            );
            let q1 = Quat::from_axis_angle(
                osc[1].axis,
                osc[1].amp * detmath::sin(TAU * osc[1].freq * t + osc[1].phase),
            );
            let mut quat = q0.mul(q1).normalized();
            let mut loc = Vec3::ZERO;
            if c.skeleton.bones[i].parent.is_none() {
                // Root: a yaw about world up, and a bob/sway on the spot.
                quat = Quat::from_axis_angle(
                    Vec3::new(0.0, 0.0, 1.0),
                    yaw_amp * detmath::sin(TAU * root_freq * 0.5 * t + root_phase),
                )
                .mul(quat)
                .normalized();
                loc = Vec3::new(
                    sway * detmath::sin(TAU * root_freq * t + root_phase),
                    sway * detmath::cos(TAU * root_freq * 0.7 * t + root_phase),
                    bob * detmath::sin(TAU * root_freq * 2.0 * t),
                );
            }
            poses.push(Pose { loc, quat });
        }
        out.push(Frame { poses });
    }
    Clip {
        fps,
        skeleton: c.skeleton.clone(),
        frames: out,
    }
}

/// Angle of a local pose quaternion in degrees.
pub fn angle_deg(q: Quat) -> f64 {
    2.0 * detmath::acos(q.normalized().w.abs().min(1.0)).to_degrees()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip::parse_skeleton;
    use crate::creature::profile::{build, skeleton_bounds};

    fn hero() -> Creature {
        let text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/hero_skel.json"
        ))
        .unwrap();
        let sk = parse_skeleton(&text).unwrap();
        let feet = vec![
            sk.index("DEF-foot.L").unwrap(),
            sk.index("DEF-foot.R").unwrap(),
        ];
        let (lo, hi) = skeleton_bounds(&sk);
        build("hero", sk, feet, Vec3::new(0.0, -1.0, 0.0), lo, hi, 256)
    }

    #[test]
    fn poses_stay_within_limits_and_are_deterministic() {
        let c = hero();
        let a = training_poses(&c, 20, 3, 0.5);
        let b = training_poses(&c, 20, 3, 0.5);
        assert_eq!(a.frames.len(), 60);
        for (fa, fb) in a.frames.iter().zip(&b.frames) {
            for (i, (pa, pb)) in fa.poses.iter().zip(&fb.poses).enumerate() {
                assert_eq!(pa.quat, pb.quat);
                assert!(angle_deg(pa.quat) <= c.max_deg[i] + 1e-9);
            }
        }
        let m = exam_motion(&c, 48, 24.0, 9, 0.9);
        assert_eq!(m.frames.len(), 48);
        for f in &m.frames {
            for (i, p) in f.poses.iter().enumerate() {
                let cap = if c.skeleton.bones[i].parent.is_none() {
                    c.max_deg[i] * 1.5 // root adds its yaw
                } else {
                    c.max_deg[i]
                };
                assert!(
                    angle_deg(p.quat) <= cap + 1e-9,
                    "bone {} {}",
                    i,
                    angle_deg(p.quat)
                );
            }
        }
        // Motion actually moves.
        let d = m.frames[0].poses[3]
            .quat
            .angle_to(m.frames[10].poses[3].quat);
        assert!(d > 0.01);
    }
}
