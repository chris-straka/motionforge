//! Attack authoring: a sword attack built from the sword's path backward.
//!
//! A readable swing is, in order: a wind-up that changes the silhouette, a
//! fast strike whose tip draws one clear arc, a follow-through and a hold.
//! The game's first attacks were an arm moving over an upright body (the
//! same keys read the same on professionally rigged bodies: the clips, not
//! the model). Professional sword attacks (CC0 Quaternius sets, measured on
//! the hero's body, `docs/attack.md`) turn the hips 60-120 degrees through
//! the strike, twist the chest further, lean 30-55 degrees into the cut,
//! drop the hips 0.15-0.2 of the body's height and stand 0.5-0.9 of it
//! wide.
//!
//! A spec (`motionforge-attack` v1, JSON) is 4-6 keys. Each key says where
//! the sword is (the grip's middle and the blade's tip) and what the body
//! does under it (hips turn, drop and shift; chest twist and lean; where
//! each foot stands; the free hand). Nothing is keyed as bone rotations:
//! every frame is solved from the interpolated key values:
//!
//! 1. hips placed and turned, the spine twisted and leaned bone by bone,
//!    the head kept toward the target;
//! 2. each foot planted where the key stands it (two-bone IK, the game's
//!    solver mirrored in `ik.rs`), the hips lowered if a leg cannot reach;
//! 3. the sword hand turned so the blade (held as the game holds it in the
//!    hand socket) runs from the grip to the tip with the wrist as straight
//!    as that allows, the arm solved to put the grip there;
//! 4. the free hand to its place (out for balance, or on the hilt).
//!
//! So planted feet cannot slide (they are solved to the same point every
//! frame) and the tip's path is the authored one. Between keys the body
//! values ease; the grip follows a Hermite curve through the keys (it
//! stops at a held key, passes through a contact key at speed); the blade
//! turns along the great arc between its directions. A foot whose place
//! changes between two keys steps: it lifts on the way.
//!
//! Lengths in a spec are in body heights (stature: the head joint at
//! 0.8418 of it, as in the game), in the character's frame: x to his left,
//! y up, z forward, from the floor under his standing hips. Angles are
//! degrees; a positive turn is to his left.

pub mod ik;

use crate::contact::hinge_axis;
use crate::detmath;
use crate::glb::Document;
use crate::json::{self, Json};
use crate::keypose::fit::{move_root, set_world_rot, signed_angle, twist_angle, RigInfo};
use crate::math::{shortest_arc, Quat, Vec3};
use crate::readcam::{HEAD_SHARE, HELD_GRIP, HELD_ROT};
use crate::rig::Trs;

pub const FORMAT: &str = "motionforge-attack";
const UP: Vec3 = Vec3 {
    x: 0.0,
    y: 1.0,
    z: 0.0,
};
/// The props kit's sword: tip 0.91 m from the grip's middle on a 2 m hero.
pub const BLADE_HEIGHTS: f64 = 0.91 / 2.0;
/// The hand may turn this far from its forearm, degrees.
pub const WRIST_MAX_DEG: f64 = 50.0;
/// How far the heel comes up in a pivot, degrees.
pub const PIVOT_HEEL_DEG: f64 = 10.0;
/// A forearm rolls this far either way, degrees.
pub const FOREARM_ROLL_DEG: f64 = 80.0;
/// The foot may turn this far from its rest on the shin before the heel
/// lifts, degrees.
pub const ANKLE_MAX_DEG: f64 = 45.0;
/// A leg is never straighter than this share of its length.
pub const LEG_REACH: f64 = 0.985;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Body {
    /// Hips' turn about the vertical, degrees (+ to his left).
    pub turn: f64,
    /// Hips lowered, heights.
    pub drop: f64,
    /// Hips moved to his left / forward, heights (weight shift).
    pub shift_x: f64,
    pub shift_z: f64,
    /// Chest turned further than the hips, degrees.
    pub twist: f64,
    /// Chest leaned forward / to his left, degrees.
    pub lean: f64,
    pub side: f64,
    /// Where the head faces, degrees from the character's forward.
    pub head: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Foot {
    /// Where the ball of the foot is from its standing place: to his
    /// left, forward (heights); the foot turned about its ball (degrees, +
    /// toes to his left).
    pub dx: f64,
    pub dz: f64,
    pub yaw: f64,
    /// Heel raised (the foot pitched about its ball), degrees.
    pub heel: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Off {
    /// Out for balance (a place that follows the chest).
    Free,
    /// On the hilt, under the sword hand.
    Grip,
    /// At a point (character frame, heights).
    At(Vec3),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Ease {
    Linear,
    /// Starts slow, arrives at speed (a strike into contact).
    In,
    /// Starts at speed, settles (a follow-through).
    Out,
    InOut,
}

impl Ease {
    pub fn at(self, u: f64) -> f64 {
        let u = u.clamp(0.0, 1.0);
        match self {
            Ease::Linear => u,
            // Half its speed at the slow end, so a three-frame strike
            // still shows its middle (u^2 put it all in the last frame).
            Ease::In => u * (0.4 + 0.6 * u),
            Ease::Out => 1.0 - (1.0 - u) * (0.4 + 0.6 * (1.0 - u)),
            Ease::InOut => u * u * (3.0 - 2.0 * u),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Key {
    pub role: String,
    /// Seconds to arrive here from the key before, then seconds held.
    pub t: f64,
    pub hold: f64,
    pub ease: Ease,
    pub body: Body,
    pub grip: Vec3,
    pub tip: Vec3,
    pub off: Off,
    /// Left, right.
    pub feet: [Foot; 2],
}

#[derive(Clone, Debug)]
pub struct Spec {
    pub name: String,
    pub fps: f64,
    /// Sword hand: 0 left, 1 right.
    pub side: usize,
    /// Grip's middle to the tip, heights.
    pub blade: f64,
    /// How high a stepping foot lifts, heights.
    pub step: f64,
    pub keys: Vec<Key>,
}

fn num(j: Option<&Json>, what: &str) -> Result<Option<f64>, String> {
    match j {
        None => Ok(None),
        Some(v) => v
            .as_f64()
            .map(Some)
            .ok_or_else(|| format!("{what} must be a number")),
    }
}

fn vec3(j: &Json, what: &str) -> Result<Vec3, String> {
    let a = j
        .as_arr()
        .filter(|a| a.len() == 3)
        .ok_or_else(|| format!("{what} must be [x, y, z]"))?;
    let f = |i: usize| {
        a[i].as_f64()
            .ok_or_else(|| format!("{what} must be numbers"))
    };
    Ok(Vec3::new(f(0)?, f(1)?, f(2)?))
}

/// Parse a spec. A key inherits what it does not say from the key before
/// (the first from standing at rest); its first key must place the sword.
pub fn parse_spec(text: &str) -> Result<Spec, String> {
    let j = json::parse(text)?;
    if j.get("format").and_then(Json::as_str) != Some(FORMAT) {
        return Err(format!("not a {FORMAT} file"));
    }
    let side = match j.get("hand").and_then(Json::as_str).unwrap_or("R") {
        "R" | "r" => 1,
        "L" | "l" => 0,
        o => return Err(format!("hand must be L or R, got '{o}'")),
    };
    let blade = num(j.get("blade"), "blade")?.unwrap_or(BLADE_HEIGHTS);
    let mut keys: Vec<Key> = Vec::new();
    for (i, k) in j
        .get("keys")
        .and_then(Json::as_arr)
        .ok_or("no keys")?
        .iter()
        .enumerate()
    {
        let prev = keys.last().cloned();
        let what = |f: &str| format!("key {i} {f}");
        let mut body = prev.as_ref().map(|p| p.body).unwrap_or_default();
        if let Some(h) = k.get("hips") {
            if let Some(v) = num(h.get("turn"), &what("hips.turn"))? {
                body.turn = v;
            }
            if let Some(v) = num(h.get("drop"), &what("hips.drop"))? {
                body.drop = v;
            }
            if let Some(s) = h.get("shift") {
                let a = s
                    .as_arr()
                    .filter(|a| a.len() == 2)
                    .ok_or_else(|| what("hips.shift must be [left, forward]"))?;
                body.shift_x = a[0].as_f64().ok_or_else(|| what("hips.shift"))?;
                body.shift_z = a[1].as_f64().ok_or_else(|| what("hips.shift"))?;
            }
        }
        if let Some(t) = k.get("torso") {
            if let Some(v) = num(t.get("twist"), &what("torso.twist"))? {
                body.twist = v;
            }
            if let Some(v) = num(t.get("lean"), &what("torso.lean"))? {
                body.lean = v;
            }
            if let Some(v) = num(t.get("side"), &what("torso.side"))? {
                body.side = v;
            }
        }
        if let Some(v) = num(k.get("head"), &what("head"))? {
            body.head = v;
        }
        let grip = match (k.get("grip"), &prev) {
            (Some(g), _) => vec3(g, &what("grip"))?,
            (None, Some(p)) => p.grip,
            (None, None) => return Err("the first key must give grip and tip".into()),
        };
        // The tip, or the blade's direction from the grip (any length).
        let tip = match (k.get("tip"), k.get("blade"), &prev) {
            (Some(g), _, _) => vec3(g, &what("tip"))?,
            (None, Some(d), _) => {
                let d = vec3(d, &what("blade"))?;
                if d.length() < 1e-9 {
                    return Err(what("blade has no direction"));
                }
                grip + d.normalized().scale(blade)
            }
            // The sword moved with the hand: the blade keeps its direction.
            (None, None, Some(p)) => grip + (p.tip - p.grip),
            (None, None, None) => return Err("the first key must give grip and tip".into()),
        };
        if (tip - grip).length() < 1e-6 {
            return Err(what("tip is on the grip"));
        }
        let off = match k.get("off") {
            None => prev.as_ref().map(|p| p.off).unwrap_or(Off::Free),
            Some(o) => match o.as_str() {
                Some("free") => Off::Free,
                Some("grip") => Off::Grip,
                Some(other) => return Err(what(&format!("off '{other}'"))),
                None => Off::At(vec3(o, &what("off"))?),
            },
        };
        let mut feet = prev.as_ref().map(|p| p.feet).unwrap_or_default();
        if let Some(f) = k.get("feet") {
            for (s, name) in ["L", "R"].iter().enumerate() {
                if let Some(v) = f.get(name) {
                    let a = v
                        .as_arr()
                        .filter(|a| a.len() == 3 || a.len() == 4)
                        .ok_or_else(|| what("feet must be [left, forward, turn] or [.., heel]"))?;
                    let g = |i: usize| a[i].as_f64().ok_or_else(|| what("feet must be numbers"));
                    feet[s] = Foot {
                        dx: g(0)?,
                        dz: g(1)?,
                        yaw: g(2)?,
                        heel: if a.len() == 4 { g(3)? } else { 0.0 },
                    };
                }
            }
        }
        let ease = match k.get("ease").and_then(Json::as_str).unwrap_or("inout") {
            "linear" => Ease::Linear,
            "in" => Ease::In,
            "out" => Ease::Out,
            "inout" => Ease::InOut,
            o => return Err(what(&format!("ease '{o}'"))),
        };
        let t = num(k.get("t"), &what("t"))?.unwrap_or(0.0);
        let hold = num(k.get("hold"), &what("hold"))?.unwrap_or(0.0);
        if !(0.0..=10.0).contains(&t) || !(0.0..=10.0).contains(&hold) {
            return Err(what("t and hold must be 0..10 s"));
        }
        if i > 0 && t <= 0.0 {
            return Err(what("needs t > 0"));
        }
        keys.push(Key {
            role: k
                .get("role")
                .and_then(Json::as_str)
                .unwrap_or("key")
                .to_string(),
            t,
            hold,
            ease,
            body,
            grip,
            tip,
            off,
            feet,
        });
    }
    if keys.len() < 2 {
        return Err("an attack needs at least two keys".into());
    }
    let fps = num(j.get("fps"), "fps")?.unwrap_or(30.0);
    if !(1.0..=240.0).contains(&fps) {
        return Err("fps must be 1..240".into());
    }
    Ok(Spec {
        name: j
            .get("name")
            .and_then(Json::as_str)
            .unwrap_or("attack")
            .to_string(),
        fps,
        side,
        blade,
        step: num(j.get("step"), "step")?.unwrap_or(0.07),
        keys,
    })
}

/// One frame's values, between keys.
#[derive(Clone, Copy, Debug)]
pub struct Params {
    pub body: Body,
    pub grip: Vec3,
    /// Blade direction, unit.
    pub dir: Vec3,
    pub feet: [Foot; 2],
    /// Extra height of each foot (a step).
    pub lift: [f64; 2],
    /// The free hand between two keys' ways of holding it.
    pub off: (Off, Off, f64),
}

fn lerp(a: f64, b: f64, e: f64) -> f64 {
    a + (b - a) * e
}

impl Spec {
    fn dir(&self, k: usize) -> Vec3 {
        (self.keys[k].tip - self.keys[k].grip).normalized()
    }

    /// The grip's velocity through key `k` (Hermite tangent): none where
    /// the sword stops (the ends, a held key, a turn-round), else along the
    /// line between its neighbours.
    fn tangent(&self, k: usize) -> Vec3 {
        let n = self.keys.len();
        if k == 0 || k + 1 >= n || self.keys[k].hold > 0.0 {
            return Vec3::ZERO;
        }
        let a = self.keys[k].grip - self.keys[k - 1].grip;
        let b = self.keys[k + 1].grip - self.keys[k].grip;
        if a.dot(b) <= 0.0 {
            return Vec3::ZERO;
        }
        (self.keys[k + 1].grip - self.keys[k - 1].grip).scale(0.5)
    }

    /// Values `e` (0..1, already eased) of the way from key `i` to `i+1`.
    pub fn at(&self, i: usize, e: f64) -> Params {
        let (a, b) = (&self.keys[i], &self.keys[(i + 1).min(self.keys.len() - 1)]);
        let body = Body {
            turn: lerp(a.body.turn, b.body.turn, e),
            drop: lerp(a.body.drop, b.body.drop, e),
            shift_x: lerp(a.body.shift_x, b.body.shift_x, e),
            shift_z: lerp(a.body.shift_z, b.body.shift_z, e),
            twist: lerp(a.body.twist, b.body.twist, e),
            lean: lerp(a.body.lean, b.body.lean, e),
            side: lerp(a.body.side, b.body.side, e),
            head: lerp(a.body.head, b.body.head, e),
        };
        // Hermite through the grips.
        let (m0, m1) = (self.tangent(i), self.tangent(i + 1));
        let (e2, e3) = (e * e, e * e * e);
        let grip = a.grip.scale(2.0 * e3 - 3.0 * e2 + 1.0)
            + m0.scale(e3 - 2.0 * e2 + e)
            + b.grip.scale(-2.0 * e3 + 3.0 * e2)
            + m1.scale(e3 - e2);
        // The blade along the great arc between its two directions.
        let (d0, d1) = (self.dir(i), self.dir((i + 1).min(self.keys.len() - 1)));
        let dir = Quat::IDENTITY
            .slerp(shortest_arc(d0, d1), e)
            .rotate_vec(d0)
            .normalized();
        let mut feet = [Foot::default(); 2];
        let mut lift = [0.0; 2];
        for s in 0..2 {
            feet[s] = Foot {
                dx: lerp(a.feet[s].dx, b.feet[s].dx, e),
                dz: lerp(a.feet[s].dz, b.feet[s].dz, e),
                yaw: lerp(a.feet[s].yaw, b.feet[s].yaw, e),
                heel: lerp(a.feet[s].heel, b.feet[s].heel, e),
            };
            let arc = detmath::sin(std::f64::consts::PI * e);
            if a.feet[s].dx != b.feet[s].dx || a.feet[s].dz != b.feet[s].dz {
                // A foot whose place changes steps.
                lift[s] = self.step * arc;
            } else if a.feet[s].yaw != b.feet[s].yaw {
                // A turn on the spot is a pivot on the ball, the heel
                // just off the floor.
                feet[s].heel += PIVOT_HEEL_DEG * arc;
            }
        }
        Params {
            body,
            grip,
            dir,
            feet,
            lift,
            off: (a.off, b.off, e),
        }
    }

    /// Every frame's values and the frame each key lands on.
    pub fn frames(&self) -> (Vec<Params>, Vec<usize>) {
        let mut out = vec![self.at(0, 0.0)];
        let mut at = vec![0usize];
        let hold = |k: &Key| (k.hold * self.fps).round() as usize;
        for _ in 0..hold(&self.keys[0]) {
            out.push(self.at(0, 0.0));
        }
        for i in 0..self.keys.len() - 1 {
            let b = &self.keys[i + 1];
            let n = ((b.t * self.fps).round() as usize).max(1);
            for j in 1..=n {
                out.push(self.at(i, b.ease.at(j as f64 / n as f64)));
            }
            at.push(out.len() - 1);
            for _ in 0..hold(b) {
                out.push(self.at(i, 1.0));
            }
        }
        (out, at)
    }
}

/// How a hand holds the blade, in the hand's own rotation frame.
#[derive(Clone, Copy, Debug)]
struct Hold {
    /// Blade direction (guard to tip).
    blade: Vec3,
    /// The grip's middle from the wrist joint (GLB units).
    grip: Vec3,
    /// The forearm's direction (elbow to wrist) with the wrist at rest.
    forearm: Vec3,
}

/// What one frame's solve could not do.
#[derive(Clone, Copy, Debug, Default)]
pub struct Miss {
    /// The grip's distance from where it was asked to be, heights.
    pub grip: f64,
    /// The blade's angle from where it was asked to point, degrees.
    pub blade_deg: f64,
    /// Hips lowered so the legs reach, heights.
    pub drop: f64,
    /// A planted foot's distance from its place, heights.
    pub foot: f64,
}

/// A rig ready to be posed from [`Params`].
pub struct Solver<'a> {
    pub info: &'a RigInfo,
    /// Stature, GLB units.
    pub h: f64,
    /// The floor under the standing hips.
    pub origin: Vec3,
    side: usize,
    hold: [Option<Hold>; 2],
    arm_axis: [Option<Vec3>; 2],
    leg_axis: [Option<Vec3>; 2],
}

fn rad(deg: f64) -> f64 {
    deg.to_radians()
}

impl<'a> Solver<'a> {
    pub fn new(info: &'a RigInfo, spec: &Spec) -> Result<Solver<'a>, String> {
        let rig = &info.rig;
        let head = info.head.ok_or("rig has no head joint")?;
        let h = (rig.head(head).y - info.ground) / HEAD_SHARE;
        let hips = rig.head(info.hips);
        let unit = h / 2.0; // the game's hero is 2 m
        let q = Quat::new(HELD_ROT[3], HELD_ROT[0], HELD_ROT[1], HELD_ROT[2]).normalized();
        let mut hold = [None; 2];
        for s in 0..2 {
            let Some(socket) = info.socket[s] else {
                continue;
            };
            let hw = rig.rest_world[info.hand[s]];
            let sw = rig.rest_world[socket];
            let inv = hw.rotation().conj();
            let grip_w = sw.apply(Vec3::new(HELD_GRIP[0], HELD_GRIP[1], HELD_GRIP[2]).scale(unit));
            hold[s] = Some(Hold {
                blade: inv
                    .rotate_vec(sw.apply_linear(q.rotate_vec(UP)))
                    .normalized(),
                grip: inv.rotate_vec(grip_w - hw.t),
                forearm: inv
                    .rotate_vec(hw.t - rig.rest_world[info.fore[s]].t)
                    .normalized(),
            });
        }
        if hold[spec.side].is_none() {
            return Err(format!(
                "rig has no Socket_Hand_{} (run standardize)",
                ["L", "R"][spec.side]
            ));
        }
        let ax = |a: usize, b: usize, c: usize| hinge_axis(rig, a, b, c);
        Ok(Solver {
            info,
            h,
            origin: Vec3::new(hips.x, info.ground, hips.z),
            side: spec.side,
            hold,
            arm_axis: [
                ax(info.upper[0], info.fore[0], info.hand[0]),
                ax(info.upper[1], info.fore[1], info.hand[1]),
            ],
            leg_axis: [
                ax(info.thigh[0], info.shin[0], info.foot[0]),
                ax(info.thigh[1], info.shin[1], info.foot[1]),
            ],
        })
    }

    /// Where a pose holds the sword: the grip's middle and the blade's
    /// direction, in the world.
    pub fn sword(&self, locals: &[Option<Trs>]) -> (Vec3, Vec3) {
        let hold = self.hold[self.side].expect("checked in new");
        let w = self.info.rig.world(locals);
        let hand = w[self.info.hand[self.side]];
        let rot = hand.rotation();
        (
            hand.t + rot.rotate_vec(hold.grip),
            rot.rotate_vec(hold.blade),
        )
    }

    /// Character frame (heights) to the world.
    pub fn point(&self, p: Vec3) -> Vec3 {
        self.origin + self.info.to_rig(p).scale(self.h)
    }

    /// The rotation that turns the standing body: yaw, then lean forward
    /// about his (turned) left, then lean to his left.
    fn body_rot(&self, yaw: f64, lean: f64, side: f64) -> Quat {
        Quat::from_axis_angle(UP, rad(yaw))
            .mul(Quat::from_axis_angle(self.info.left, rad(lean)))
            .mul(Quat::from_axis_angle(self.info.fwd, rad(-side)))
    }

    fn rest_rot(&self, n: usize) -> Quat {
        self.info.rig.rest_world[n].rotation()
    }

    /// The hand's world rotation that runs the blade along `dir` with the
    /// wrist as straight as that allows for a forearm pointing `forearm`.
    fn hand_rot(&self, hold: &Hold, dir: Vec3, forearm: Vec3) -> Quat {
        let r0 = shortest_arc(hold.blade, dir);
        let roll = signed_angle(r0.rotate_vec(hold.forearm), forearm, dir);
        Quat::from_axis_angle(dir, roll).mul(r0)
    }

    /// Put hand `s` where its grip point is at `grip` with the blade along
    /// `dir` (`blend` of the way from the hand's own rest turn on the
    /// forearm). Returns (grip miss, blade angle off), GLB units / degrees.
    fn place_hand(
        &self,
        locals: &mut [Option<Trs>],
        s: usize,
        grip: Vec3,
        dir: Vec3,
        chest_yaw: f64,
        blend: f64,
    ) -> (f64, f64) {
        let info = self.info;
        let rig = &info.rig;
        let (up, fo, ha) = (info.upper[s], info.fore[s], info.hand[s]);
        let Some(hold) = self.hold[s] else {
            return (0.0, 0.0);
        };
        // The elbow points down, out and a little back of the chest.
        let out = if s == 1 {
            info.left.scale(-1.0)
        } else {
            info.left
        };
        let yaw = Quat::from_axis_angle(UP, rad(chest_yaw));
        let pole_off = yaw.rotate_vec(out.scale(0.6) + info.fwd.scale(-0.45)) + UP.scale(-0.8);
        let mut forearm = {
            let w = rig.world(locals);
            (grip - w[up].t).normalized()
        };
        let mut rot = self.hand_rot(&hold, dir, forearm);
        // Solve the arm for the hand's turn, turn the hand for the arm's
        // forearm (and its wrist limit), and again; the last pass solves
        // the arm for the final turn, so the grip is on its point.
        for pass in 0..4 {
            let w = rig.world(locals);
            // A hand on the sword puts its grip point there; a free hand
            // its wrist.
            let wrist = grip - rot.rotate_vec(hold.grip).scale(blend);
            ik::reach(
                rig,
                locals,
                up,
                fo,
                ha,
                wrist,
                w[up].t + pole_off.scale(self.h),
                self.arm_axis[s],
            );
            if pass == 3 {
                break;
            }
            let w = rig.world(locals);
            forearm = (w[ha].t - w[fo].t).normalized();
            rot = self.hand_rot(&hold, dir, forearm);
            // The wrist's limit: turn the hand back toward the forearm.
            let bent = detmath::acos(rot.rotate_vec(hold.forearm).dot(forearm).clamp(-1.0, 1.0));
            if bent > rad(WRIST_MAX_DEG) {
                let straight = shortest_arc(rot.rotate_vec(hold.forearm), forearm).mul(rot);
                rot = rot.slerp(straight, 1.0 - rad(WRIST_MAX_DEG) / bent);
            }
        }
        // The hand's own rest turn on this forearm, for the free hand.
        let w = rig.world(locals);
        let natural = w[fo]
            .rotation()
            .mul(self.rest_rot(fo).conj())
            .mul(self.rest_rot(ha));
        let rot = natural.slerp(rot, blend).normalized();
        // The hand's twist about the forearm is the forearm's, up to what
        // a forearm turns; only the rest is the wrist's.
        let axis_w = (w[ha].t - w[fo].t).normalized();
        let delta = rot.mul(natural.conj());
        let twist = twist_angle(delta, axis_w).clamp(-rad(FOREARM_ROLL_DEG), rad(FOREARM_ROLL_DEG));
        let rf = Quat::from_axis_angle(axis_w, twist).mul(w[fo].rotation());
        set_world_rot(rig, locals, fo, rf.normalized());
        set_world_rot(rig, locals, ha, rot);
        // What the pose actually does, measured.
        let w = rig.world(locals);
        let at = w[ha].t + rot.rotate_vec(hold.grip).scale(blend);
        let got = rot.rotate_vec(hold.blade);
        (
            (at - grip).length(),
            detmath::acos(got.dot(dir).clamp(-1.0, 1.0)).to_degrees(),
        )
    }

    /// Where the free hand's grip point goes for one way of holding it.
    fn off_target(&self, off: Off, p: &Params, grip: Vec3, dir: Vec3) -> Vec3 {
        let s = 1 - self.side;
        match off {
            Off::At(v) => self.point(v),
            // On the hilt, a hand's width under the sword hand.
            Off::Grip => grip - dir.scale(0.045 * self.h),
            Off::Free => {
                // Out from its shoulder at the waist, a little ahead: carried
                // by the chest.
                let out = if s == 0 { 1.0 } else { -1.0 };
                let yaw = Quat::from_axis_angle(UP, rad(p.body.turn + p.body.twist));
                let chest = self.point(Vec3::new(
                    p.body.shift_x,
                    0.80 - p.body.drop,
                    p.body.shift_z,
                ));
                chest
                    + yaw
                        .rotate_vec(self.info.to_rig(Vec3::new(out * 0.17, -0.19, 0.07)))
                        .scale(self.h)
            }
        }
    }

    /// Solve one frame.
    pub fn pose(&self, p: &Params) -> (Vec<Option<Trs>>, Miss) {
        let info = self.info;
        let rig = &info.rig;
        let mut l: Vec<Option<Trs>> = vec![None; rig.names.len()];
        let mut miss = Miss::default();
        let b = &p.body;
        // 1. Hips, spine, head.
        set_world_rot(
            rig,
            &mut l,
            info.hips,
            self.body_rot(b.turn, 0.3 * b.lean, 0.3 * b.side)
                .mul(self.rest_rot(info.hips)),
        );
        let hips_rest = rig.rest_world[info.hips].t;
        let want = hips_rest
            + info
                .to_rig(Vec3::new(b.shift_x, -b.drop, b.shift_z))
                .scale(self.h);
        let now = rig.world(&l)[info.hips].t;
        move_root(info, &mut l, want - now);
        let n = info.spine.len().max(1) as f64;
        for (i, &sp) in info.spine.iter().enumerate() {
            let f = (i + 1) as f64 / n;
            let q = self.body_rot(
                b.turn + b.twist * f,
                b.lean * (0.3 + 0.7 * f),
                b.side * (0.3 + 0.7 * f),
            );
            set_world_rot(rig, &mut l, sp, q.mul(self.rest_rot(sp)));
        }
        let chest_yaw = b.turn + b.twist;
        // The head faces where the key says, never more than 70 degrees
        // off the chest; it stays more upright than the chest.
        let head_yaw = b.head.clamp(chest_yaw - 70.0, chest_yaw + 70.0);
        if let Some(nk) = info.neck {
            let q = self.body_rot(0.5 * (chest_yaw + head_yaw), 0.6 * b.lean, 0.6 * b.side);
            set_world_rot(rig, &mut l, nk, q.mul(self.rest_rot(nk)));
        }
        if let Some(hd) = info.head {
            let q = self.body_rot(head_yaw, 0.25 * b.lean, 0.25 * b.side);
            set_world_rot(rig, &mut l, hd, q.mul(self.rest_rot(hd)));
        }
        // 2. Feet: lower the hips until both legs reach, then plant.
        // The foot's world rotation (turned about the vertical, the heel
        // raised about the foot's own side axis) and its ankle: the foot
        // is placed by its ball, so a turn pivots on the ball.
        let foot_rot = |s: usize, heel: f64| {
            let f = &p.feet[s];
            let yaw = Quat::from_axis_angle(UP, rad(f.yaw));
            let pitch = match info.toe[s] {
                Some(toe) => {
                    let d = rig.head(toe) - rig.head(info.foot[s]);
                    let along = Vec3::new(d.x, 0.0, d.z).normalized();
                    Quat::from_axis_angle(UP.cross(along), rad(heel))
                }
                None => Quat::IDENTITY,
            };
            yaw.mul(pitch)
        };
        let ankle_at = |s: usize, heel: f64| {
            let f = &p.feet[s];
            let ankle = rig.rest_world[info.foot[s]].t;
            let off = info.to_rig(Vec3::new(f.dx, p.lift[s], f.dz)).scale(self.h);
            match info.toe[s] {
                Some(toe) => {
                    let ball = rig.head(toe);
                    ball + off + foot_rot(s, heel).rotate_vec(ankle - ball)
                }
                None => ankle + off,
            }
        };
        let target = |s: usize| ankle_at(s, p.feet[s].heel);
        {
            // The least drop that lets both legs reach (exact: a stepped
            // search made the hips jitter).
            let w = rig.world(&l);
            let mut need: f64 = 0.0;
            for s in 0..2 {
                let len = LEG_REACH
                    * ((rig.head(info.shin[s]) - rig.head(info.thigh[s])).length()
                        + (rig.head(info.foot[s]) - rig.head(info.shin[s])).length());
                let v = w[info.thigh[s]].t - target(s);
                let flat2 = v.x * v.x + v.z * v.z;
                if flat2 < len * len {
                    need = need.max(v.y - (len * len - flat2).sqrt());
                }
            }
            if need > 0.0 {
                move_root(info, &mut l, UP.scale(-need));
                miss.drop = need / self.h;
            }
        }
        for s in 0..2 {
            let f = &p.feet[s];
            // The knee tracks the toes (an ankle barely turns): mostly
            // where the foot points, a little where the hips face.
            let knee =
                Quat::from_axis_angle(UP, rad(0.85 * f.yaw + 0.15 * b.turn)).rotate_vec(info.fwd);
            let rel_rest = self
                .rest_rot(info.shin[s])
                .conj()
                .mul(self.rest_rot(info.foot[s]));
            // The heel as keyed; then raised further while the ankle would
            // fold past its limit (a back foot comes up onto its ball).
            let mut heel = f.heel;
            let mut last = f64::INFINITY;
            let mut off = 0.0;
            for pass in 0..5 {
                let t = ankle_at(s, heel);
                let left = ik::reach(
                    rig,
                    &mut l,
                    info.thigh[s],
                    info.shin[s],
                    info.foot[s],
                    t,
                    t + knee.scale(0.5 * self.h) + UP.scale(0.5 * self.h),
                    self.leg_axis[s],
                );
                off = left;
                set_world_rot(
                    rig,
                    &mut l,
                    info.foot[s],
                    foot_rot(s, heel).mul(self.rest_rot(info.foot[s])),
                );
                let w = rig.world(&l);
                let rel = w[info.shin[s]]
                    .rotation()
                    .conj()
                    .mul(w[info.foot[s]].rotation());
                let bent = rel_rest.angle_to(rel).to_degrees();
                if bent <= ANKLE_MAX_DEG || info.toe[s].is_none() || pass == 4 || heel >= 60.0 {
                    break;
                }
                if bent >= last {
                    // Raising the heel did not help (a leg stretched out
                    // behind points its foot instead): back to the key's
                    // heel, once.
                    heel = f.heel;
                    last = f64::NEG_INFINITY;
                    continue;
                }
                if last == f64::NEG_INFINITY {
                    break;
                }
                last = bent;
                heel = (heel + (bent - ANKLE_MAX_DEG)).min(60.0);
            }
            miss.foot = miss.foot.max(off / self.h);
        }
        // 3. The sword hand.
        let grip = self.point(p.grip);
        let dir = info.to_rig(p.dir).normalized();
        let (gm, bd) = self.place_hand(&mut l, self.side, grip, dir, chest_yaw, 1.0);
        miss.grip = gm / self.h;
        miss.blade_deg = bd;
        // 4. The free hand, between the two keys' ways of holding it.
        let (oa, ob, e) = p.off;
        let ta = self.off_target(oa, p, grip, dir);
        let tb = self.off_target(ob, p, grip, dir);
        let on_hilt = lerp(
            (oa == Off::Grip) as u8 as f64,
            (ob == Off::Grip) as u8 as f64,
            e,
        );
        let other = 1 - self.side;
        if self.hold[other].is_some() {
            self.place_hand(&mut l, other, ta.lerp(tb, e), dir, chest_yaw, on_hilt);
        }
        crate::helpers::drive(rig, &mut l);
        (l, miss)
    }
}

/// What a built attack did.
#[derive(Clone, Debug)]
pub struct Report {
    pub name: String,
    pub frames: usize,
    pub fps: f64,
    /// The frame each key lands on, with its role.
    pub keys: Vec<(usize, String)>,
    /// What each key's own pose missed.
    pub key_miss: Vec<Miss>,
    /// Worst of each miss over the clip, and the frame of the worst grip.
    pub worst: Miss,
    pub worst_grip_frame: usize,
}

impl Report {
    pub fn to_json(&self) -> Json {
        let r = |v: f64, k: f64| Json::num((v * k).round() / k);
        Json::obj(vec![
            ("name", Json::str(&self.name)),
            ("frames", Json::num(self.frames as f64)),
            ("fps", Json::num(self.fps)),
            (
                "keys",
                Json::Arr(
                    self.keys
                        .iter()
                        .zip(&self.key_miss)
                        .map(|((f, role), m)| {
                            Json::obj(vec![
                                ("frame", Json::num(*f as f64)),
                                ("role", Json::str(role)),
                                ("grip_miss_heights", r(m.grip, 1e4)),
                                ("blade_off_deg", r(m.blade_deg, 10.0)),
                                ("hips_lowered_heights", r(m.drop, 1e3)),
                            ])
                        })
                        .collect(),
                ),
            ),
            ("grip_miss_heights", r(self.worst.grip, 1e4)),
            ("grip_miss_frame", Json::num(self.worst_grip_frame as f64)),
            ("blade_off_deg", r(self.worst.blade_deg, 10.0)),
            ("hips_lowered_heights", r(self.worst.drop, 1e3)),
            ("foot_miss_heights", r(self.worst.foot, 1e4)),
        ])
    }
}

/// Solve every frame of `spec` on the rig.
pub fn solve(
    info: &RigInfo,
    spec: &Spec,
) -> Result<(Vec<f64>, Vec<Vec<Option<Trs>>>, Report), String> {
    let solver = Solver::new(info, spec)?;
    let (params, at) = spec.frames();
    let mut frames = Vec::with_capacity(params.len());
    let mut worst = Miss::default();
    let mut worst_grip_frame = 0;
    let mut key_miss = vec![Miss::default(); at.len()];
    for (f, p) in params.iter().enumerate() {
        let (l, m) = solver.pose(p);
        if let Some(k) = at.iter().position(|a| *a == f) {
            key_miss[k] = m;
        }
        if m.grip > worst.grip {
            worst.grip = m.grip;
            worst_grip_frame = f;
        }
        worst.blade_deg = worst.blade_deg.max(m.blade_deg);
        worst.drop = worst.drop.max(m.drop);
        worst.foot = worst.foot.max(m.foot);
        frames.push(l);
    }
    let times: Vec<f64> = (0..frames.len()).map(|f| f as f64 / spec.fps).collect();
    let report = Report {
        name: spec.name.clone(),
        frames: frames.len(),
        fps: spec.fps,
        keys: at
            .iter()
            .zip(&spec.keys)
            .map(|(f, k)| (*f, k.role.clone()))
            .collect(),
        key_miss,
        worst,
        worst_grip_frame,
    };
    Ok((times, frames, report))
}

/// Build the attack as a clip on a rigged GLB (added; a clip of the same
/// name is replaced).
pub fn build(
    doc: &Document,
    spec: &Spec,
    name: Option<&str>,
) -> Result<(Document, Report), String> {
    let info = RigInfo::new(doc)?;
    let (times, frames, mut report) = solve(&info, spec)?;
    let name = name.unwrap_or(&spec.name).to_string();
    report.name = name.clone();
    let mut out = doc.clone();
    let mut nodes = info.rig.joints();
    nodes.sort_unstable();
    nodes.dedup();
    let anim =
        crate::stylize_body::encode_clip(&mut out, &info.rig, &name, &times, &nodes, &frames);
    let list = out.array_mut("animations");
    match list
        .iter()
        .position(|a| a.get("name").and_then(Json::as_str) == Some(name.as_str()))
    {
        Some(i) => list[i] = anim,
        None => list.push(anim),
    }
    out.compact();
    Ok((out, report))
}

/// The attacks that ship with the tool (`presets/attacks/*.json`).
pub fn preset(name: &str) -> Option<&'static str> {
    Some(match name {
        "cleave" => include_str!("../../../../presets/attacks/cleave.json"),
        "sweep" => include_str!("../../../../presets/attacks/sweep.json"),
        "backcut" => include_str!("../../../../presets/attacks/backcut.json"),
        "thrust" => include_str!("../../../../presets/attacks/thrust.json"),
        _ => return None,
    })
}

pub const PRESETS: [&str; 4] = ["cleave", "backcut", "sweep", "thrust"];
