//! A four-legged creature's parts read off its profile (which chains are
//! legs, the spine between them, the neck, the tail), and what follows
//! from knowing them:
//!
//! - a range-of-motion pose set for weightforge (`weights check --poses`):
//!   legs swing, fold and spread, the spine arches and twists, the neck and
//!   tail bend. Without it a creature is judged on weightforge's fallback
//!   for skeletons it can't read, every bone bent 45 degrees each way and
//!   twisted, which no wolf does and no wolf's weights survive;
//! - a walk and a lunge made on the creature's own skeleton, for bodies
//!   with no clip library to retarget from. They are plain forward
//!   kinematics (a four-beat walk on the spot, a crouch and strike), meant
//!   to show a rig moving, not to be the last word on its animation.
//!
//! Everything is in the profile's frame (Z up). Rotations are given about
//! world axes and carried into each bone's own rest frame, so a parent's
//! turn carries its children as in any skeleton.

use super::profile::Creature;
use crate::clip::{Clip, Frame, Pose};
use crate::detmath;
use crate::math::{zero_roll_basis, Quat, Vec3};

const TAU: f64 = std::f64::consts::PI * 2.0;

/// One leg: its bones from the body down to the foot.
#[derive(Clone, Debug, PartialEq)]
pub struct Leg {
    pub bones: Vec<usize>,
    pub front: bool,
    pub left: bool,
}

/// A quadruped's parts, as bone indices into the profile's skeleton.
#[derive(Clone, Debug, PartialEq)]
pub struct Parts {
    /// Hind left, hind right, front left, front right.
    pub legs: Vec<Leg>,
    /// From the hips forward to the chest; the hierarchy root is left out
    /// (turning it turns the whole body).
    pub spine: Vec<usize>,
    /// From the chest out to the head.
    pub neck: Vec<usize>,
    /// From the hips out to the tip.
    pub tail: Vec<usize>,
}

fn children(c: &Creature, bone: usize) -> Vec<usize> {
    (0..c.skeleton.len())
        .filter(|&i| c.skeleton.bones[i].parent == Some(bone))
        .collect()
}

/// The chain from `start` to a leaf, at each fork taking the child whose
/// subtree reaches farthest along `dir`.
fn chain_toward(c: &Creature, start: usize, dir: Vec3, skip: &[usize]) -> Vec<usize> {
    fn reach(c: &Creature, bone: usize, dir: Vec3) -> f64 {
        let b = &c.skeleton.bones[bone];
        let mut best = b.head.dot(dir).max(b.tail.dot(dir));
        for k in children(c, bone) {
            best = best.max(reach(c, k, dir));
        }
        best
    }
    let mut out = vec![start];
    let mut cur = start;
    loop {
        let kids: Vec<usize> = children(c, cur)
            .into_iter()
            .filter(|k| !skip.contains(k))
            .collect();
        let Some(&next) = kids.iter().max_by(|a, b| {
            reach(c, **a, dir)
                .total_cmp(&reach(c, **b, dir))
                .then(b.cmp(a))
        }) else {
            break;
        };
        out.push(next);
        cur = next;
    }
    out
}

/// The profile's left (Z up): up cross forward.
fn left_of(c: &Creature) -> Vec3 {
    Vec3::new(0.0, 0.0, 1.0).cross(c.forward).normalized()
}

/// Read a quadruped's parts. Errors when the profile doesn't have four
/// legs on the ground, two toward the front and two behind.
pub fn parts(c: &Creature) -> Result<Parts, String> {
    if c.feet.len() != 4 {
        return Err(format!(
            "a quadruped has four feet on the ground; this profile has {}",
            c.feet.len()
        ));
    }
    let left = left_of(c);
    let mut legs: Vec<(Vec<usize>, f64, f64)> = Vec::new();
    for &f in &c.feet {
        let mut chain = c.leg_chain(f, 8);
        chain.reverse();
        let top = &c.skeleton.bones[chain[0]];
        let foot = &c.skeleton.bones[f];
        legs.push((chain, top.head.dot(c.forward), foot.head.dot(left)));
    }
    // Front and hind by where each leg leaves the body, left and right by
    // where its foot stands: the two most forward are the front pair.
    let mut by_along: Vec<usize> = (0..4).collect();
    by_along.sort_by(|a, b| legs[*a].1.total_cmp(&legs[*b].1).then(a.cmp(b)));
    let (hind, front) = (&by_along[..2], &by_along[2..]);
    let pair = |ids: &[usize], is_front: bool| -> Vec<Leg> {
        let (a, b) = (ids[0], ids[1]);
        let (l, r) = if legs[a].2 >= legs[b].2 {
            (a, b)
        } else {
            (b, a)
        };
        vec![
            Leg {
                bones: legs[l].0.clone(),
                front: is_front,
                left: true,
            },
            Leg {
                bones: legs[r].0.clone(),
                front: is_front,
                left: false,
            },
        ]
    };
    let mut out_legs = pair(hind, false);
    out_legs.extend(pair(front, true));
    let attach = |leg: &Leg| c.skeleton.bones[leg.bones[0]].parent;
    let (Some(hips), Some(chest)) = (attach(&out_legs[0]), attach(&out_legs[2])) else {
        return Err("a leg hangs off nothing: the skeleton has no body above it".into());
    };
    if attach(&out_legs[1]) != Some(hips) || attach(&out_legs[3]) != Some(chest) {
        return Err("a pair of legs doesn't share a bone to hang from".into());
    }
    // The spine: the path through the tree from the hips to the chest.
    let up_from = |mut b: usize| {
        let mut v = vec![b];
        while let Some(p) = c.skeleton.bones[b].parent {
            v.push(p);
            b = p;
        }
        v
    };
    let (from_hips, from_chest) = (up_from(hips), up_from(chest));
    let Some(meet) = from_hips.iter().find(|b| from_chest.contains(b)).copied() else {
        return Err("hips and chest are on separate skeletons".into());
    };
    let mut path: Vec<usize> = from_hips
        .iter()
        .take_while(|b| **b != meet)
        .copied()
        .collect();
    path.push(meet);
    let mut down: Vec<usize> = from_chest
        .iter()
        .take_while(|b| **b != meet)
        .copied()
        .collect();
    down.reverse();
    path.extend(down);
    let in_legs: Vec<usize> = out_legs.iter().flat_map(|l| l.bones.clone()).collect();
    let mut taken = in_legs.clone();
    taken.extend(&path);
    // Neck: what leaves the chest reaching forward. Tail: what leaves the
    // hips reaching back. Neither when nothing does.
    let branch = |from: usize, dir: Vec3| -> Vec<usize> {
        let base = c.skeleton.bones[from].head.dot(dir);
        children(c, from)
            .into_iter()
            .filter(|k| !taken.contains(k))
            .map(|k| chain_toward(c, k, dir, &taken))
            .filter(|ch| {
                let tip = &c.skeleton.bones[*ch.last().unwrap()];
                tip.tail.dot(dir) > base + 0.02 * c.height
            })
            .max_by(|a, b| {
                let reach = |ch: &Vec<usize>| c.skeleton.bones[*ch.last().unwrap()].tail.dot(dir);
                reach(a).total_cmp(&reach(b)).then(b[0].cmp(&a[0]))
            })
            .unwrap_or_default()
    };
    let neck = branch(chest, c.forward);
    let tail = branch(hips, c.forward.scale(-1.0));
    let root = c.root();
    Ok(Parts {
        legs: out_legs,
        spine: path.into_iter().filter(|b| *b != root).collect(),
        neck,
        tail,
    })
}

/// The pose delta that turns `bone` by `angle` about a world axis.
fn turn(c: &Creature, bone: usize, axis: Vec3, angle: f64) -> Quat {
    let b = &c.skeleton.bones[bone];
    let local = zero_roll_basis(b.head, b.tail).transpose().mul_vec(axis);
    Quat::from_axis_angle(local.normalized(), angle)
}

struct Poser<'a> {
    c: &'a Creature,
    quats: Vec<Quat>,
    /// Turning about it swings a hanging bone's tip forward, and lifts the
    /// tip of one that points forward.
    pitch: Vec3,
    up: Vec3,
    /// The whole body moved (world), on the root.
    shift: Vec3,
}

impl<'a> Poser<'a> {
    fn new(c: &'a Creature) -> Self {
        let up = Vec3::new(0.0, 0.0, 1.0);
        Poser {
            c,
            quats: vec![Quat::IDENTITY; c.skeleton.len()],
            pitch: up.scale(-1.0).cross(c.forward).normalized(),
            up,
            shift: Vec3::ZERO,
        }
    }
    fn add(&mut self, bone: usize, axis: Vec3, degrees: f64) {
        let q = turn(self.c, bone, axis, degrees.to_radians());
        self.quats[bone] = q.mul(self.quats[bone]).normalized();
    }
    /// Swing forward (a leg bone) or lift (a spine, neck or tail bone).
    fn pitch(&mut self, bone: usize, degrees: f64) {
        self.add(bone, self.pitch, degrees);
    }
    /// Turn to the left.
    fn yaw(&mut self, bone: usize, degrees: f64) {
        self.add(bone, self.up, degrees);
    }
    /// `degrees` in all, shared along a chain.
    fn pitch_chain(&mut self, bones: &[usize], degrees: f64) {
        for &b in bones {
            self.pitch(b, degrees / bones.len() as f64);
        }
    }
    fn yaw_chain(&mut self, bones: &[usize], degrees: f64) {
        for &b in bones {
            self.yaw(b, degrees / bones.len() as f64);
        }
    }
    fn frame(self) -> Frame {
        let root = self.c.root();
        // The root's own frame: its pose offset is in its rest axes.
        let b = &self.c.skeleton.bones[root];
        let local = zero_roll_basis(b.head, b.tail)
            .transpose()
            .mul_vec(self.shift);
        Frame {
            poses: self
                .quats
                .into_iter()
                .enumerate()
                .map(|(i, quat)| Pose {
                    loc: if i == root { local } else { Vec3::ZERO },
                    quat,
                })
                .collect(),
        }
    }
}

/// Fold a leg by `amount` (0 straight, 1 fully tucked for a step): the
/// joints below the top bend against each other, a front leg's first
/// joint backward (an elbow), a hind leg's forward (a stifle).
fn fold(p: &mut Poser, leg: &Leg, amount: f64) {
    let way = if leg.front { -1.0 } else { 1.0 };
    if leg.bones.len() > 2 {
        p.pitch(leg.bones[1], way * 28.0 * amount);
        p.pitch(leg.bones[2], -way * 40.0 * amount);
    }
    if leg.bones.len() > 3 {
        p.pitch(leg.bones[3], way * 18.0 * amount);
    }
}

/// A four-beat walk on the spot, one cycle that loops: hind left, front
/// left, hind right, front right, each leg swinging back on the ground and
/// folding as it comes forward; the spine sways toward the leading hind
/// leg, the head nods twice a cycle, the tail swings behind.
pub fn walk(c: &Creature, parts: &Parts, frames: usize, fps: f64) -> Clip {
    let offsets = [0.0, 0.5, 0.25, 0.75]; // hind L, hind R, front L, front R
    let mut out = Vec::with_capacity(frames);
    for f in 0..frames {
        let t = f as f64 / frames as f64;
        let mut p = Poser::new(c);
        for (leg, off) in parts.legs.iter().zip(offsets) {
            let phase = TAU * (t + off);
            // Forward of vertical at the start of the stance, behind at its end.
            let swing = detmath::cos(phase) * if leg.front { 20.0 } else { 18.0 };
            p.pitch(leg.bones[0], swing);
            // In the air (the second half of the phase), folded most half way.
            let lift = (-detmath::sin(phase)).max(0.0);
            fold(&mut p, leg, lift);
            // The paw keeps to the ground while the leg leans.
            if let Some(&paw) = leg.bones.get(leg.bones.len().saturating_sub(2)) {
                if leg.bones.len() > 3 {
                    p.pitch(paw, -swing * 0.4 * (1.0 - lift));
                }
            }
        }
        let sway = detmath::sin(TAU * t);
        p.yaw_chain(&parts.spine, 5.0 * sway);
        p.pitch_chain(&parts.neck, 4.0 * detmath::sin(2.0 * TAU * t));
        p.yaw_chain(&parts.neck, -3.0 * sway);
        for (i, &b) in parts.tail.iter().enumerate() {
            let lag = i as f64 * 0.35;
            p.yaw(
                b,
                14.0 / parts.tail.len() as f64 * detmath::sin(TAU * t - lag),
            );
        }
        out.push(p.frame());
    }
    Clip {
        fps,
        skeleton: c.skeleton.clone(),
        frames: out,
    }
}

fn smooth(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/// A lunge: sink onto folded legs with the head low and the tail up, then
/// throw the body, the forelegs and the head forward, and come back to
/// standing. It starts and ends at rest.
pub fn attack(c: &Creature, parts: &Parts, frames: usize, fps: f64) -> Clip {
    let mut out = Vec::with_capacity(frames);
    for f in 0..frames {
        let t = if frames > 1 {
            f as f64 / (frames - 1) as f64
        } else {
            0.0
        };
        // Crouch builds to 0.35 and is spent by 0.5; the strike peaks at
        // 0.5 and eases out to the end.
        let crouch = smooth(t / 0.35) * (1.0 - smooth((t - 0.35) / 0.15));
        let strike = smooth((t - 0.35) / 0.15) * (1.0 - smooth((t - 0.55) / 0.45));
        let mut p = Poser::new(c);
        for leg in &parts.legs {
            if leg.front {
                fold(&mut p, leg, 0.9 * crouch);
                p.pitch(leg.bones[0], -8.0 * crouch + 26.0 * strike);
            } else {
                fold(&mut p, leg, 0.6 * crouch + 0.2 * strike);
                p.pitch(leg.bones[0], 6.0 * crouch - 16.0 * strike);
            }
        }
        p.pitch_chain(&parts.spine, -10.0 * crouch + 6.0 * strike);
        // The head low in the crouch, then thrown out level with the back.
        p.pitch_chain(&parts.neck, -22.0 * crouch - 8.0 * strike);
        p.pitch_chain(&parts.tail, 25.0 * crouch + 10.0 * strike);
        // The body sinks, then goes with the strike.
        p.shift = c.forward.scale(c.height * (0.16 * strike - 0.03 * crouch))
            + Vec3::new(0.0, 0.0, -c.height * (0.07 * crouch + 0.03 * strike));
        out.push(p.frame());
    }
    Clip {
        fps,
        skeleton: c.skeleton.clone(),
        frames: out,
    }
}

/// The range-of-motion pose set, in weightforge's RON (`poses/*.ron`
/// there): what a four-legged body does with each part, by exact bone
/// name. weightforge's frame is glTF's (+Y up, +Z forward), and its
/// directions are the character's own, so the set holds whichever way the
/// profile faces.
pub fn rom_ron(c: &Creature, parts: &Parts) -> String {
    let name = |b: usize| c.skeleton.bones[b].name.as_str();
    let mut poses: Vec<(String, Vec<(usize, String)>)> = Vec::new();
    let bend = |dir: &str, deg: f64| format!("Bend({dir}, {deg})");
    let shared = |bones: &[usize], dir: &str, total: f64| -> Vec<(usize, String)> {
        let each = (total / bones.len().max(1) as f64 * 10.0).round() / 10.0;
        bones.iter().map(|&b| (b, bend(dir, each))).collect()
    };
    for leg in &parts.legs {
        let tag = format!(
            "{}_{}",
            if leg.front { "front" } else { "hind" },
            if leg.left { "left" } else { "right" }
        );
        let top = leg.bones[0];
        poses.push((format!("{tag}_forward"), vec![(top, bend("Forward", 35.0))]));
        poses.push((format!("{tag}_back"), vec![(top, bend("Back", 35.0))]));
        poses.push((
            format!("{tag}_out"),
            vec![(top, bend(if leg.left { "Left" } else { "Right" }, 20.0))],
        ));
        if leg.bones.len() > 2 {
            let (a, b) = if leg.front {
                ("Back", "Forward")
            } else {
                ("Forward", "Back")
            };
            poses.push((
                format!("{tag}_fold"),
                vec![(leg.bones[1], bend(a, 45.0)), (leg.bones[2], bend(b, 60.0))],
            ));
        }
        if leg.bones.len() > 3 {
            let paw = leg.bones[leg.bones.len() - 2];
            poses.push((format!("{tag}_paw"), vec![(paw, bend("Forward", 30.0))]));
        }
    }
    let chains: [(&str, &[usize], [(&str, &str, f64); 4]); 3] = [
        (
            "spine",
            &parts.spine,
            [
                ("arch", "Up", 30.0),
                ("hollow", "Down", 20.0),
                ("left", "Left", 30.0),
                ("right", "Right", 30.0),
            ],
        ),
        (
            "neck",
            &parts.neck,
            [
                ("up", "Up", 45.0),
                ("down", "Down", 50.0),
                ("left", "Left", 45.0),
                ("right", "Right", 45.0),
            ],
        ),
        (
            "tail",
            &parts.tail,
            [
                ("up", "Up", 60.0),
                ("down", "Down", 30.0),
                ("left", "Left", 60.0),
                ("right", "Right", 60.0),
            ],
        ),
    ];
    for (part, bones, moves) in chains {
        if bones.is_empty() {
            continue;
        }
        for (what, dir, total) in moves {
            poses.push((format!("{part}_{what}"), shared(bones, dir, total)));
        }
    }
    if !parts.spine.is_empty() {
        let each = (25.0 / parts.spine.len() as f64 * 10.0).round() / 10.0;
        poses.push((
            "spine_twist".into(),
            parts
                .spine
                .iter()
                .map(|&b| (b, format!("Twist({each})")))
                .collect(),
        ));
    }
    if let Some(&head) = parts.neck.last() {
        poses.push(("head_twist".into(), vec![(head, "Twist(30.0)".into())]));
    }
    let mut s = String::new();
    s.push_str("// A four-legged creature's range of motion, written by\n");
    s.push_str("// `motionforge creature-gait --rom` from its profile.\n");
    s.push_str("PoseSet(\n    name: \"quadruped\",\n    poses: [\n");
    for (pose, moves) in &poses {
        s.push_str(&format!(
            "        PoseSpec(name: \"{pose}\", mirror: false, moves: [\n"
        ));
        for (bone, motion) in moves {
            s.push_str(&format!(
                "            Move(bone: \"{}\", motion: {motion}),\n",
                name(*bone)
            ));
        }
        s.push_str("        ]),\n");
    }
    s.push_str("    ],\n)\n");
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip::{fk, Bone, Skeleton};
    use crate::creature::motion::angle_deg;
    use crate::creature::profile::{build, skeleton_bounds};

    /// A plain dog facing -Y (Z up): hips at the back, four spine bones to
    /// the chest, a neck and head, a tail, four legs of four bones.
    fn dog() -> Creature {
        let mut bones: Vec<Bone> = Vec::new();
        let mut add =
            |name: &str, parent: Option<usize>, head: [f64; 3], tail: [f64; 3]| -> usize {
                bones.push(Bone {
                    name: name.into(),
                    parent,
                    head: Vec3::new(head[0], head[1], head[2]),
                    tail: Vec3::new(tail[0], tail[1], tail[2]),
                });
                bones.len() - 1
            };
        let hips = add("hips", None, [0.0, 0.30, 0.50], [0.0, 0.20, 0.50]);
        let mut prev = hips;
        for (i, y) in [0.20, 0.08, -0.04, -0.16].iter().enumerate() {
            prev = add(
                &format!("spine{i}"),
                Some(prev),
                [0.0, *y, 0.50],
                [0.0, y - 0.12, 0.50],
            );
        }
        let chest = prev;
        let mut n = chest;
        for (i, (y, z)) in [(-0.28, 0.52), (-0.36, 0.58), (-0.44, 0.60)]
            .iter()
            .enumerate()
        {
            n = add(
                &format!("neck{i}"),
                Some(n),
                [0.0, *y, *z],
                [0.0, y - 0.08, z + 0.03],
            );
        }
        let mut t = hips;
        for (i, (y, z)) in [(0.32, 0.50), (0.42, 0.44), (0.50, 0.36)]
            .iter()
            .enumerate()
        {
            t = add(
                &format!("tail{i}"),
                Some(t),
                [0.0, *y, *z],
                [0.0, y + 0.09, z - 0.07],
            );
        }
        let mut feet = Vec::new();
        for (tag, parent, y) in [("hind", hips, 0.28), ("front", chest, -0.22)] {
            for (side, x) in [("l", 0.08), ("r", -0.08)] {
                let mut p = parent;
                for (i, (z0, z1)) in [(0.46, 0.30), (0.30, 0.14), (0.14, 0.04), (0.04, 0.0)]
                    .iter()
                    .enumerate()
                {
                    p = add(
                        &format!("{tag}_{side}{i}"),
                        Some(p),
                        [x, y, *z0],
                        [x, y, *z1],
                    );
                }
                feet.push(p);
            }
        }
        let sk = Skeleton { bones };
        let (lo, hi) = skeleton_bounds(&sk);
        build("dog", sk, feet, Vec3::new(0.0, -1.0, 0.0), lo, hi, 256)
    }

    fn names(c: &Creature, bones: &[usize]) -> Vec<String> {
        bones
            .iter()
            .map(|&b| c.skeleton.bones[b].name.clone())
            .collect()
    }

    #[test]
    fn a_dogs_parts_are_read_from_its_skeleton() {
        let c = dog();
        let p = parts(&c).unwrap();
        let tops: Vec<String> = p
            .legs
            .iter()
            .map(|l| c.skeleton.bones[l.bones[0]].name.clone())
            .collect();
        assert_eq!(tops, ["hind_l0", "hind_r0", "front_l0", "front_r0"]);
        assert_eq!(
            p.legs.iter().map(|l| (l.front, l.left)).collect::<Vec<_>>(),
            [(false, true), (false, false), (true, true), (true, false)]
        );
        assert!(p.legs.iter().all(|l| l.bones.len() == 4));
        assert_eq!(
            names(&c, &p.spine),
            ["spine0", "spine1", "spine2", "spine3"]
        );
        assert_eq!(names(&c, &p.neck), ["neck0", "neck1", "neck2"]);
        assert_eq!(names(&c, &p.tail), ["tail0", "tail1", "tail2"]);
    }

    #[test]
    fn a_biped_is_not_a_quadruped() {
        let mut c = dog();
        c.feet.truncate(2);
        assert!(parts(&c).unwrap_err().contains("four feet"));
    }

    #[test]
    fn pitching_a_leg_swings_its_foot_forward_and_a_spine_bone_lifts_its_tip() {
        let c = dog();
        let p = parts(&c).unwrap();
        let leg = &p.legs[2];
        let mut poser = Poser::new(&c);
        poser.pitch(leg.bones[0], 30.0);
        let frame = poser.frame();
        let posed = fk(&c.skeleton, &frame);
        let foot = *leg.bones.last().unwrap();
        let rest = fk(
            &c.skeleton,
            &Frame {
                poses: vec![
                    Pose {
                        loc: Vec3::ZERO,
                        quat: Quat::IDENTITY
                    };
                    c.skeleton.len()
                ],
            },
        );
        let moved = posed[foot].head - rest[foot].head;
        assert!(moved.dot(c.forward) > 0.15, "foot moved {moved:?}");
        let mut poser = Poser::new(&c);
        poser.pitch(p.neck[0], 30.0);
        let posed = fk(&c.skeleton, &poser.frame());
        let head = *p.neck.last().unwrap();
        assert!(posed[head].head.z > rest[head].head.z + 0.03);
    }

    #[test]
    fn the_walk_loops_and_moves_every_leg_within_reason() {
        let c = dog();
        let p = parts(&c).unwrap();
        let w = walk(&c, &p, 24, 24.0);
        assert_eq!(w.frames.len(), 24);
        // One more step of the same cycle lands on the first frame.
        let again = walk(&c, &p, 24, 24.0);
        assert_eq!(w.frames[0].poses[5].quat, again.frames[0].poses[5].quat);
        for leg in &p.legs {
            let top = leg.bones[0];
            let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
            for f in &w.frames {
                let a = angle_deg(f.poses[top].quat);
                lo = lo.min(a);
                hi = hi.max(a);
                for &b in &leg.bones {
                    assert!(angle_deg(f.poses[b].quat) < 60.0);
                }
            }
            assert!(hi > 15.0 && lo < 3.0, "leg swings {lo}..{hi}");
        }
        // Diagonal legs are a quarter-cycle apart, never all four folded at once.
        let folded = |f: &Frame| {
            p.legs
                .iter()
                .filter(|l| angle_deg(f.poses[l.bones[2]].quat) > 20.0)
                .count()
        };
        assert!(w.frames.iter().all(|f| folded(f) <= 2));
    }

    #[test]
    fn the_lunge_starts_and_ends_at_rest_and_reaches_forward() {
        let c = dog();
        let p = parts(&c).unwrap();
        let a = attack(&c, &p, 24, 24.0);
        for f in [&a.frames[0], &a.frames[23]] {
            assert!(f.poses.iter().all(|q| angle_deg(q.quat) < 1e-6));
        }
        let head = *p.neck.last().unwrap();
        let rest = fk(&c.skeleton, &a.frames[0]);
        let far = a
            .frames
            .iter()
            .map(|f| fk(&c.skeleton, f)[head].head.dot(c.forward))
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            far > rest[head].head.dot(c.forward) + 0.03,
            "head reached {far}"
        );
    }

    #[test]
    fn the_range_of_motion_names_every_part() {
        let c = dog();
        let p = parts(&c).unwrap();
        let ron = rom_ron(&c, &p);
        for want in [
            "PoseSet(",
            "name: \"front_left_forward\"",
            "Move(bone: \"front_l0\", motion: Bend(Forward, 35))",
            "name: \"hind_right_fold\"",
            "Move(bone: \"hind_r1\", motion: Bend(Forward, 45))",
            "name: \"spine_twist\"",
            "Move(bone: \"spine2\", motion: Twist(6.3))",
            "name: \"neck_down\"",
            "name: \"tail_left\"",
            "name: \"head_twist\"",
        ] {
            assert!(ron.contains(want), "missing {want}\n{ron}");
        }
        assert_eq!(ron.matches("PoseSpec(").count(), 4 * 5 + 3 * 4 + 2);
    }
}
