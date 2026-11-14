//! Two-bone IK on plain points: the game's solver (`hlh/bevy/src/ik/solve.rs`
//! `two_bone`, `soft_reach`) mirrored here in f64 with `detmath`, so an
//! authored pose is solved the way the game will solve reaches on top of
//! it. The limb bends in the plane of its root, the target and a pole; the
//! middle joint points toward the pole.
//!
//! [`set_limb`] turns the answer into bone rotations as a true hinge: the
//! upper bone is rolled about itself until its anatomical hinge axis
//! (`contact::hinge_axis`) is square to the bend plane, and the lower bone
//! is its rest pose on the upper one turned about that axis alone. (The
//! game's `limb_turns` lays each bone along the answer the shortest way;
//! baked into a clip that left elbows bent out of their hinge plane.)

use crate::detmath;
use crate::keypose::fit::{set_world_rot, signed_angle};
use crate::math::{shortest_arc, Quat, Vec3};
use crate::rig::{Rig, Trs};

/// How far a hinge may bend, radians (0 = straight): the game's
/// `Hinge::LIMB`.
pub const MIN_BEND: f64 = 0.02;
pub const MAX_BEND: f64 = 2.6;

#[derive(Clone, Copy, Debug)]
pub struct TwoBone {
    pub mid: Vec3,
    pub end: Vec3,
    /// How far the end is from the target.
    pub miss: f64,
}

/// The reach a target at `d` gets under the soft limit: straight through
/// up to `full - soft`, then easing in so it only nears `full` far out.
pub fn soft_reach(d: f64, full: f64, soft: f64) -> f64 {
    if soft <= 0.0 {
        return d.min(full);
    }
    let start = full - soft;
    if d <= start {
        d
    } else {
        start + soft * (1.0 - detmath::exp(-(d - start) / soft))
    }
}

fn any_perp(v: Vec3) -> Vec3 {
    let other = if v.x.abs() < 0.9 {
        Vec3::new(1.0, 0.0, 0.0)
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    };
    v.cross(other).normalized()
}

/// Solve a limb from `a` (its root, fixed) through `b` to `c` for
/// `target`; the middle joint points toward `pole`.
pub fn two_bone(a: Vec3, b: Vec3, c: Vec3, target: Vec3, pole: Vec3, soft: f64) -> TwoBone {
    let l1 = (b - a).length().max(1e-9);
    let l2 = (c - b).length().max(1e-9);
    let to = target - a;
    let dist = to.length();
    let dir = if dist > 1e-9 {
        to.scale(1.0 / dist)
    } else if (c - a).length() > 1e-9 {
        (c - a).normalized()
    } else {
        Vec3::new(0.0, -1.0, 0.0)
    };
    let mut d = soft_reach(dist, l1 + l2, soft).max((l1 - l2).abs() + 1e-6);
    let cos_mid = ((l1 * l1 + l2 * l2 - d * d) / (2.0 * l1 * l2)).clamp(-1.0, 1.0);
    let bend = std::f64::consts::PI - detmath::acos(cos_mid);
    let held = bend.clamp(MIN_BEND, MAX_BEND);
    if (held - bend).abs() > 1e-9 {
        d = (l1 * l1 + l2 * l2 + 2.0 * l1 * l2 * detmath::cos(held))
            .max(0.0)
            .sqrt();
    }
    let cos_root = ((l1 * l1 + d * d - l2 * l2) / (2.0 * l1 * d.max(1e-9))).clamp(-1.0, 1.0);
    let sin_root = (1.0 - cos_root * cos_root).max(0.0).sqrt();
    let side = |p: Vec3| {
        let v = p - a;
        v - dir.scale(v.dot(dir))
    };
    let mut bend_dir = side(pole);
    if bend_dir.length() < 1e-6 {
        bend_dir = side(b);
    }
    let bend_dir = if bend_dir.length() < 1e-6 {
        any_perp(dir)
    } else {
        bend_dir.normalized()
    };
    let mid = a + (dir.scale(cos_root) + bend_dir.scale(sin_root)).scale(l1);
    let end = a + dir.scale(d);
    TwoBone {
        mid,
        end,
        miss: (end - target).length(),
    }
}

/// Write a [`two_bone`] answer onto the limb's bones as a hinge (module
/// doc). `axis_rest` is the limb's rest hinge axis in the world
/// (`contact::hinge_axis`); without one each bone turns the shortest way.
/// The end bone's rotation is the caller's to set.
pub fn set_limb(
    rig: &Rig,
    locals: &mut [Option<Trs>],
    upper: usize,
    lower: usize,
    end: usize,
    sol: &TwoBone,
    axis_rest: Option<Vec3>,
) {
    let world = rig.world(locals);
    let a = world[upper].t;
    let u_cur = (world[lower].t - a).normalized();
    let u_new = (sol.mid - a).normalized();
    let v_new = (sol.end - sol.mid).normalized();
    let mut ru = shortest_arc(u_cur, u_new).mul(world[upper].rotation());
    let ru_rest = rig.rest_world[upper].rotation();
    let rl_rest = rig.rest_world[lower].rotation();
    let normal = u_new.cross(v_new);
    let hinge = match axis_rest {
        Some(ax) if normal.length() > 1e-4 => {
            let n = normal.normalized();
            // Roll the upper bone about itself: its hinge axis onto the
            // bend plane's normal.
            let h = ru.mul(ru_rest.conj()).rotate_vec(ax);
            ru = Quat::from_axis_angle(u_new, signed_angle(h, n, u_new)).mul(ru);
            Some(n)
        }
        _ => None,
    };
    set_world_rot(rig, locals, upper, ru.normalized());
    let rl = match hinge {
        Some(n) => {
            // The lower bone as it rests on the upper one, turned about the
            // hinge alone.
            let base = ru.mul(ru_rest.conj()).mul(rl_rest);
            let v_rest = rig.head(end) - rig.head(lower);
            let v_base = base.mul(rl_rest.conj()).rotate_vec(v_rest).normalized();
            Quat::from_axis_angle(n, signed_angle(v_base, v_new, n)).mul(base)
        }
        None => {
            let w = rig.world(locals);
            let v_cur = (w[end].t - w[lower].t).normalized();
            shortest_arc(v_cur, v_new).mul(w[lower].rotation())
        }
    };
    set_world_rot(rig, locals, lower, rl.normalized());
}

/// Solve and write a limb so its end lands on `target`: [`two_bone`] and
/// [`set_limb`], then aimed again past what the hinge left over (a limb
/// whose bones do not rest exactly in their hinge plane lands a few
/// millimetres off; a planted foot must not). Returns the end's distance
/// from the target.
#[allow(clippy::too_many_arguments)]
pub fn reach(
    rig: &Rig,
    locals: &mut [Option<Trs>],
    upper: usize,
    lower: usize,
    end: usize,
    target: Vec3,
    pole: Vec3,
    axis_rest: Option<Vec3>,
) -> f64 {
    let mut aim = target;
    let mut left = f64::INFINITY;
    for _ in 0..4 {
        let w = rig.world(locals);
        let sol = two_bone(w[upper].t, w[lower].t, w[end].t, aim, pole, 0.0);
        set_limb(rig, locals, upper, lower, end, &sol, axis_rest);
        let got = rig.world(locals)[end].t;
        left = (got - target).length();
        // Out of reach (the solve itself missed): nothing to correct.
        if left < 1e-10 || sol.miss > 1e-9 {
            break;
        }
        aim = aim + (target - got);
    }
    left
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reaches_a_target_inside_its_range_with_the_knee_at_the_pole() {
        let a = Vec3::new(0.0, 1.0, 0.0);
        let b = Vec3::new(0.0, 0.5, 0.0);
        let c = Vec3::new(0.0, 0.0, 0.0);
        let target = Vec3::new(0.0, 0.3, 0.2);
        let pole = Vec3::new(0.0, 0.6, 1.0);
        let s = two_bone(a, b, c, target, pole, 0.0);
        assert!(s.miss < 1e-9, "{s:?}");
        assert!(((s.mid - a).length() - 0.5).abs() < 1e-9);
        assert!(((s.end - s.mid).length() - 0.5).abs() < 1e-9);
        // The knee is on the pole's side of the hip-target line.
        assert!(s.mid.z > 0.1, "{s:?}");
        // Flip the pole: the knee flips.
        let s2 = two_bone(a, b, c, target, Vec3::new(0.0, 0.6, -1.0), 0.0);
        assert!(s2.mid.z < 0.0 && s2.miss < 1e-9, "{s2:?}");
    }

    #[test]
    fn a_target_out_of_reach_is_missed_by_the_overshoot_never_stretched() {
        let a = Vec3::new(0.0, 1.0, 0.0);
        let b = Vec3::new(0.0, 0.5, 0.0);
        let c = Vec3::new(0.0, 0.0, 0.0);
        let s = two_bone(
            a,
            b,
            c,
            Vec3::new(0.0, 1.0, 1.5),
            Vec3::new(0.0, 2.0, 0.5),
            0.0,
        );
        // Never quite straight (the least bend), so a hair under 1.0.
        let reach = (s.end - a).length();
        assert!(reach < 1.0 && reach > 0.999, "{reach}");
        assert!((s.miss - (1.5 - reach)).abs() < 1e-9);
        // The soft limit eases the last of the reach.
        assert!(soft_reach(0.95, 1.0, 0.1) < 0.95);
        assert!((soft_reach(0.5, 1.0, 0.1) - 0.5).abs() < 1e-12);
        assert!(soft_reach(5.0, 1.0, 0.1) <= 1.0 && soft_reach(1.05, 1.0, 0.1) < 1.0);
    }
}
