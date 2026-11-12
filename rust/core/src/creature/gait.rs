//! A four-legged creature's parts read off its profile (which chains are
//! legs, the spine between them, the neck, the tail), and what follows
//! from knowing them:
//!
//! - a range-of-motion pose set for weightforge (`weights check --poses`):
//!   legs swing, fold and spread, the spine arches and twists, the neck and
//!   tail bend. Without it a creature is judged on weightforge's fallback
//!   for skeletons it can't read, every bone bent 45 degrees each way and
//!   twisted, which no wolf does and no wolf's weights survive;
//! - a walk and a trot whose feet stay where they are put (`stride`): each
//!   foot is given a spot on the ground for as long as it bears weight and
//!   an arc through the air when it steps, and the leg is solved to reach
//!   it. The clip stays on the spot and comes with the speed to move the
//!   body at; moved at that speed, no foot slides;
//! - a lunge, and the older walk on the spot, in plain forward kinematics
//!   (a crouch and strike; legs swung by angle), meant to show a rig
//!   moving, not to be the last word on its animation.
//!
//! Everything is in the profile's frame (Z up). Rotations are given about
//! world axes and carried into each bone's own rest frame, so a parent's
//! turn carries its children as in any skeleton.

use super::profile::Creature;
use crate::clip::{fk, Clip, Frame, Pose};
use crate::detmath;
use crate::math::{zero_roll_basis, Mat3, Quat, Vec3};

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
        make_frame(self.c, &self.quats, self.shift)
    }
}

/// A frame from every bone's pose turn and the whole body's move (world).
fn make_frame(c: &Creature, quats: &[Quat], shift: Vec3) -> Frame {
    let root = c.root();
    // The root's own frame: its pose offset is in its rest axes.
    let b = &c.skeleton.bones[root];
    let local = zero_roll_basis(b.head, b.tail).transpose().mul_vec(shift);
    Frame {
        poses: quats
            .iter()
            .enumerate()
            .map(|(i, quat)| Pose {
                loc: if i == root { local } else { Vec3::ZERO },
                quat: *quat,
            })
            .collect(),
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

/// The older four-beat walk on the spot, by angle alone: each leg swings
/// back and folds as it comes forward, with nothing holding a foot to the
/// ground (its feet slide). Kept to set beside `stride`.
pub fn walk_spot(c: &Creature, parts: &Parts, frames: usize, fps: f64) -> Clip {
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

/// A gait with planted feet.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Gait {
    /// Four beats, each foot on the ground about two thirds of the time.
    Walk,
    /// Two beats, diagonal pairs together, a moment in the air between.
    Trot,
}

impl Gait {
    /// (share of the cycle a foot is down, the legs' places in the cycle in
    /// `Parts` order, Froude number, stride in hip heights, foot lift in
    /// hip heights, body bob in hip heights, spine sway in degrees).
    fn shape(self) -> (f64, [f64; 4], f64, f64, f64, f64, f64) {
        match self {
            // Hind left, front left, hind right, front right.
            Gait::Walk => (0.65, [0.0, 0.5, 0.25, 0.75], 0.20, 1.6, 0.11, 0.012, 4.0),
            Gait::Trot => (0.42, [0.0, 0.5, 0.5, 0.0], 0.90, 2.3, 0.16, 0.030, 2.0),
        }
    }
}

/// One cycle of a planted-feet gait, on the spot, and what the game needs
/// to move the body under it.
pub struct Stride {
    pub clip: Clip,
    /// Metres a second the body must move forward for the feet to hold.
    pub speed: f64,
    /// Seconds a cycle.
    pub period: f64,
    /// Share of the cycle each foot is on the ground.
    pub duty: f64,
    /// Metres the body passes over one planted foot.
    pub stance: f64,
}

/// A leg as the solver sees it: its joints' offsets in their parent
/// bone's rest frame, and the paw that is kept flat.
struct LegRig {
    bones: Vec<usize>,
    /// Bone k's joint to the next joint, in bone k's rest frame.
    segs: Vec<Vec3>,
    /// The paw's rest orientation and its head-to-tail in the world.
    paw_rest: Mat3,
    paw_vec: Vec3,
    /// Where the paw's tail stands at rest.
    contact: Vec3,
    /// All the solved segments end to end.
    reach: f64,
    /// The top joint's height above the ankle at rest.
    drop: f64,
}

fn leg_rig(c: &Creature, leg: &Leg) -> LegRig {
    let sk = &c.skeleton;
    let n = leg.bones.len();
    let paw = leg.bones[n - 1];
    let segs: Vec<Vec3> = (0..n - 1)
        .map(|k| {
            let (a, b) = (leg.bones[k], leg.bones[k + 1]);
            sk.rest_world(a)
                .transpose()
                .mul_vec(sk.bones[b].head - sk.bones[a].head)
        })
        .collect();
    LegRig {
        bones: leg.bones.clone(),
        reach: segs.iter().map(|s| s.length()).sum(),
        segs,
        paw_rest: sk.rest_world(paw),
        paw_vec: sk.bones[paw].tail - sk.bones[paw].head,
        contact: sk.bones[paw].tail,
        drop: sk.bones[leg.bones[0]].head.z - sk.bones[paw].head.z,
    }
}

/// Solve one leg onto `contact` (where its paw's tail must be), given the
/// body already posed: joint positions by reaching back and forth from the
/// carried rest shape (so every joint keeps bending the way it does at
/// rest), then each bone aimed along its segment with no added twist, and
/// the paw kept flat as it stands at rest. A spot out of reach is met as
/// far as a nearly straight leg allows.
fn solve_leg(c: &Creature, rig: &LegRig, quats: &mut [Quat], shift: Vec3, contact: Vec3) {
    let sk = &c.skeleton;
    let m = rig.segs.len();
    if m == 0 {
        return;
    }
    for &b in &rig.bones {
        quats[b] = Quat::IDENTITY;
    }
    let posed = fk(sk, &make_frame(c, quats, shift));
    let mut pts: Vec<Vec3> = rig.bones.iter().map(|&b| posed[b].head).collect();
    let lens: Vec<f64> = rig.segs.iter().map(|s| s.length()).collect();
    let hip = pts[0];
    let mut ankle = contact - rig.paw_vec;
    let far = (ankle - hip).length();
    if far > 0.99 * rig.reach {
        ankle = hip + (ankle - hip).scale(0.99 * rig.reach / far);
    }
    for _ in 0..24 {
        pts[m] = ankle;
        for k in (0..m).rev() {
            pts[k] = pts[k + 1] + (pts[k] - pts[k + 1]).normalized().scale(lens[k]);
        }
        pts[0] = hip;
        for k in 0..m {
            pts[k + 1] = pts[k] + (pts[k + 1] - pts[k]).normalized().scale(lens[k]);
        }
    }
    let top = rig.bones[0];
    let mut parent_rot = match sk.bones[top].parent {
        Some(p) => posed[p].rot,
        None => Mat3::identity(),
    };
    for k in 0..m {
        let b = rig.bones[k];
        // The root's own turn was applied in `posed`; a leg's top bone is
        // never the root (`parts` requires a body above it).
        let carried = parent_rot.mul_mat(sk.rest_parent_rel(b).0);
        let cur = carried.mul_vec(rig.segs[k]).normalized();
        let want = (pts[k + 1] - pts[k]).normalized();
        let axis = cur.cross(want);
        let turn = if axis.length() < 1e-12 {
            Quat::IDENTITY
        } else {
            let angle = detmath::acos(cur.dot(want).clamp(-1.0, 1.0));
            Quat::from_axis_angle(carried.transpose().mul_vec(axis).normalized(), angle)
        };
        quats[b] = turn;
        parent_rot = carried.mul_mat(turn.to_mat3());
    }
    let paw = rig.bones[m];
    let carried = parent_rot.mul_mat(sk.rest_parent_rel(paw).0);
    quats[paw] = Quat::from_mat3(carried.transpose().mul_mat(rig.paw_rest)).normalized();
}

/// One cycle of a walk or a trot with planted feet, sized from the
/// creature's own legs: the cadence from the gait's Froude number and its
/// stride in hip heights, the stride shortened to what the legs can reach
/// with the body a little lowered, and the speed with it.
///
/// On the spot: while a foot bears weight its spot on the ground slides
/// back under the body at exactly `speed`; when it steps it leaves and
/// lands at that same speed and rises in between. The body bobs twice a
/// cycle, the spine sways with the hind legs, the head nods against the
/// bob, the tail swings behind.
pub fn stride(c: &Creature, parts: &Parts, gait: Gait, fps: f64) -> Stride {
    let (duty, offsets, froude, stride_in_hips, lift_in_hips, bob_in_hips, sway_deg) = gait.shape();
    let rigs: Vec<LegRig> = parts.legs.iter().map(|l| leg_rig(c, l)).collect();
    let hip = rigs.iter().map(|r| r.drop).sum::<f64>() / rigs.len() as f64;
    let hip = hip.max(1e-6);
    // The cadence comes from the gait and the leg (Froude number, stride
    // in hip heights); what the legs can't reach is taken off the speed,
    // not made up with quicker steps.
    let full_speed = (froude * 9.81 * hip).sqrt();
    let frames = ((stride_in_hips * hip / full_speed) * fps).round().max(8.0) as usize;
    let period = frames as f64 / fps;
    // The body rides a little low, so a leg has length to spare at both
    // ends of its stance.
    let crouch = 0.07 * hip;
    let bob = bob_in_hips * hip;
    let half_reach = rigs
        .iter()
        .map(|r| {
            let h = (r.drop - crouch - bob).max(0.0);
            (0.97 * r.reach).powi(2) - h * h
        })
        .fold(f64::INFINITY, f64::min)
        .max(0.0)
        .sqrt();
    let stance = (duty * stride_in_hips * hip).min(2.0 * half_reach);
    let speed = stance / (duty * period);
    let half = 0.5 * stance;
    let lift = lift_in_hips * hip;
    let up = Vec3::new(0.0, 0.0, 1.0);
    let swing_time = (1.0 - duty) * period;
    let mut out = Vec::with_capacity(frames);
    for f in 0..frames {
        let t = f as f64 / frames as f64;
        let mut p = Poser::new(c);
        let sway = detmath::sin(TAU * t);
        p.yaw_chain(&parts.spine, sway_deg * sway);
        p.pitch_chain(&parts.neck, 3.0 * detmath::sin(2.0 * TAU * t));
        p.yaw_chain(&parts.neck, -0.6 * sway_deg * sway);
        for (i, &b) in parts.tail.iter().enumerate() {
            let lag = i as f64 * 0.35;
            p.yaw(
                b,
                12.0 / parts.tail.len() as f64 * detmath::sin(TAU * t - lag),
            );
        }
        // Lowest as a pair of feet takes the weight, twice a cycle.
        p.shift = up.scale(-crouch - bob * detmath::cos(2.0 * TAU * t));
        let (mut quats, shift) = (p.quats, p.shift);
        for (rig, off) in rigs.iter().zip(offsets) {
            let phase = (t + off).fract();
            let (along, height) = if phase < duty {
                (half - stance * (phase / duty), 0.0)
            } else {
                // Leaves and lands moving back at the body's speed.
                let u = (phase - duty) / (1.0 - duty);
                let (u2, u3) = (u * u, u * u * u);
                let slope = -speed * swing_time;
                let x = (2.0 * u3 - 3.0 * u2 + 1.0) * -half
                    + (u3 - 2.0 * u2 + u) * slope
                    + (-2.0 * u3 + 3.0 * u2) * half
                    + (u3 - u2) * slope;
                (x, lift * 0.5 * (1.0 - detmath::cos(TAU * u)))
            };
            let contact = rig.contact + c.forward.scale(along) + up.scale(height);
            solve_leg(c, rig, &mut quats, shift, contact);
        }
        out.push(make_frame(c, &quats, shift));
    }
    Stride {
        clip: Clip {
            fps,
            skeleton: c.skeleton.clone(),
            frames: out,
        },
        speed,
        period,
        duty,
        stance,
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
    fn the_walk_on_the_spot_loops_and_moves_every_leg_within_reason() {
        let c = dog();
        let p = parts(&c).unwrap();
        let w = walk_spot(&c, &p, 24, 24.0);
        assert_eq!(w.frames.len(), 24);
        // One more step of the same cycle lands on the first frame.
        let again = walk_spot(&c, &p, 24, 24.0);
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

    /// The dog with its knees and hocks bent, as a real one stands: its
    /// legs have length to spare, so a stride is not cut short.
    fn bent_dog() -> Creature {
        let mut c = dog();
        for b in c.skeleton.bones.iter_mut() {
            let knee = |v: &mut Vec3, back: f64| {
                if (v.z - 0.30).abs() < 1e-9 {
                    v.y += back;
                } else if (v.z - 0.14).abs() < 1e-9 {
                    v.y -= back * 0.8;
                }
            };
            if b.name.starts_with("hind_") {
                knee(&mut b.head, -0.07);
                knee(&mut b.tail, -0.07);
            } else if b.name.starts_with("front_") {
                knee(&mut b.head, 0.05);
                knee(&mut b.tail, 0.05);
            }
        }
        c
    }

    /// Where a leg's paw stands, frame by frame, once the body is moved
    /// forward at the stride's speed.
    fn paw_track(c: &Creature, s: &Stride, leg: &Leg) -> Vec<Vec3> {
        let paw = *leg.bones.last().unwrap();
        s.clip
            .frames
            .iter()
            .enumerate()
            .map(|(f, frame)| {
                let posed = fk(&c.skeleton, frame);
                let len = c.skeleton.bone_length(paw);
                let tail = posed[paw].head + posed[paw].rot.mul_vec(Vec3::new(0.0, len, 0.0));
                tail + c.forward.scale(s.speed * f as f64 / s.clip.fps)
            })
            .collect()
    }

    #[test]
    fn a_planted_foot_stays_where_it_was_put() {
        for gait in [Gait::Walk, Gait::Trot] {
            let c = bent_dog();
            let p = parts(&c).unwrap();
            let s = stride(&c, &p, gait, 30.0);
            let n = s.clip.frames.len();
            assert!((s.period - n as f64 / 30.0).abs() < 1e-12);
            assert!(s.stance > 0.15 * c.height, "{gait:?} stance {}", s.stance);
            let (duty, offsets, ..) = gait.shape();
            for (leg, off) in p.legs.iter().zip(offsets) {
                let track = paw_track(&c, &s, leg);
                let down: Vec<Vec3> = (0..n)
                    .filter(|f| (*f as f64 / n as f64 + off).fract() < duty)
                    .map(|f| {
                        // Laps of the cycle put the spot a stride further on.
                        let lap = ((f as f64 / n as f64 + off).floor()) * s.speed * s.period;
                        track[f] - c.forward.scale(lap)
                    })
                    .collect();
                assert!(down.len() >= 3);
                let mean = down
                    .iter()
                    .fold(Vec3::ZERO, |a, b| a + *b)
                    .scale(1.0 / down.len() as f64);
                let slide = down
                    .iter()
                    .map(|d| (*d - mean).length())
                    .fold(0.0, f64::max);
                assert!(
                    slide < 0.002 * c.height,
                    "{gait:?}: a planted foot slides {slide} m"
                );
                // In the air it clears the ground, and never goes under it.
                let rest_z = c.skeleton.bones[*leg.bones.last().unwrap()].tail.z;
                let top = track.iter().map(|t| t.z).fold(f64::NEG_INFINITY, f64::max);
                let low = track.iter().map(|t| t.z).fold(f64::INFINITY, f64::min);
                assert!(
                    top > rest_z + 0.03 * c.height,
                    "{gait:?}: lifts {}",
                    top - rest_z
                );
                assert!(low > rest_z - 0.002 * c.height);
            }
            // Bones keep their length: nothing is stretched to reach.
            for frame in &s.clip.frames {
                let posed = fk(&c.skeleton, frame);
                for leg in &p.legs {
                    for w in leg.bones.windows(2) {
                        let rest =
                            (c.skeleton.bones[w[1]].head - c.skeleton.bones[w[0]].head).length();
                        let now = (posed[w[1]].head - posed[w[0]].head).length();
                        assert!((rest - now).abs() < 1e-9);
                    }
                }
            }
        }
    }

    #[test]
    fn a_trot_is_faster_than_a_walk_and_pairs_its_diagonals() {
        let c = bent_dog();
        let p = parts(&c).unwrap();
        let (w, t) = (
            stride(&c, &p, Gait::Walk, 30.0),
            stride(&c, &p, Gait::Trot, 30.0),
        );
        assert!(t.speed > 1.5 * w.speed && t.period < w.period);
        // Hind left and front right step together in a trot.
        let (hl, fr) = (paw_track(&c, &t, &p.legs[0]), paw_track(&c, &t, &p.legs[3]));
        for (a, b) in hl.iter().zip(&fr) {
            assert!((a.z - b.z).abs() < 0.01 * c.height);
        }
    }

    #[test]
    fn a_straight_legged_body_still_walks_with_short_steps() {
        // The plain dog stands on straight legs: nothing to spare, so the
        // stride shrinks instead of the legs stretching.
        let c = dog();
        let p = parts(&c).unwrap();
        let s = stride(&c, &p, Gait::Walk, 30.0);
        assert!(s.stance > 0.0 && s.stance < 0.5 * c.height);
        assert!(s.clip.frames.len() >= 8);
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
