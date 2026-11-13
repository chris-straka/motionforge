//! Readability at the game camera: can a viewer tell the move from his
//! silhouette, seen from where the player sees him?
//!
//! Per frame the skinned body and the blade (a ribbon from the guard to
//! the tip, held at the hand socket as the game holds it) are projected
//! through the game's follow camera and filled into a small mask. From
//! the masks:
//!
//! - **Pose change**: the largest `1 - IoU` between a frame of the strike
//!   (load to settle) and the neutral silhouette (the clip's own first
//!   frame, the stance it starts from; or a given clip's): how far the
//!   strike's key pose reads from where it began.
//! - **Body change**: the same against the neutral body alone (no
//!   blade), from the windup's start to the settle: whether the body
//!   takes part or only a thin blade moves (the owner could not read
//!   HLH's attack_1, where only the arm and blade move). The score uses
//!   this one.
//! - **Sweep**: the screen area the blade covers over the strike (union of
//!   its masks), over the neutral body's own area.
//! - **Arc**: the blade tip's screen path over the strike, in neutral body
//!   heights, times how consistently it turns one way
//!   (`|sum of signed turns| / sum of turns`; 1 for an arc or a straight
//!   cut, near 0 for a wiggle).
//! - **Load hold**: the longest run of frames in the second before the
//!   strike's fastest frame whose silhouette changes less than
//!   `STILL_SHARE` from the frame before: a held anticipation. It counts
//!   in the score only as far as the arc after it reads.
//!
//! The strike's frames are `motion check`'s: the fastest tip frame, the
//! load before it ([`crate::motioncheck::strike_timing`]); the settle is
//! the first frame after the peak with the tip under 15% of its peak
//! speed. Deterministic: f64 arithmetic in a fixed order, `detmath` for
//! the trigonometry.

use crate::contact::{skinned_mesh, SkinVert};
use crate::detmath;
use crate::glb::{Affine, Document};
use crate::json::Json;
use crate::math::{Quat, Vec3};
use crate::motioncheck::{strike_timing, Subject, HOLD_SHARE};
use crate::rig::{Rig, Trs};

/// A frame whose silhouette changed less than this (1 - IoU with the
/// frame before) is held still.
pub const STILL_SHARE: f64 = 0.04;
/// Head joint height over stature: Andras is 2 m with his head joint at
/// 1.6836 m (HLH `character.rs`, `RigRest::stature`).
pub const HEAD_SHARE: f64 = 1.6836 / 2.0;
/// HLH's `tracks::held_pose` (Bevy, in the hand socket's space): the
/// blade's rotation (x, y, z, w) and the grip's middle.
pub const HELD_ROT: [f64; 4] = [0.30965, -0.67445, 0.00552, 0.67023];
pub const HELD_GRIP: [f64; 3] = [0.0, 0.0428, -0.0850];

/// How the blade sits in the hand socket.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BladePose {
    /// As the game holds it (`HELD_ROT`, `HELD_GRIP`).
    Game,
    /// Straight out of the fist along the socket's +Y from its origin.
    Socket,
}

/// The camera and the blade. Defaults are HLH's follow camera in a
/// source level (`bevy/src/fx.rs` `camera_follow`, recorded by the game in
/// `proportions.rs`): focus 0.535 m above the body's centre, itself 1.0 m
/// above the feet; 3.0 m back at pitch -0.32 rad; 75 deg vertical field of
/// view; straight behind him. Masks are 480x270 (the game's 1280x720 at
/// three eighths).
#[derive(Clone, Debug)]
pub struct CamOptions {
    pub focus_m: f64,
    pub distance_m: f64,
    pub pitch_rad: f64,
    pub fov_deg: f64,
    /// Round him from behind, toward his right, degrees.
    pub yaw_deg: f64,
    pub width: usize,
    pub height: usize,
    /// Scale him so his head joint is at `HEAD_SHARE` of this (metres); 0
    /// keeps the GLB's units as metres.
    pub stature_m: f64,
    /// The blade from the grip: guard and tip, metres; its width.
    pub blade_from_m: f64,
    pub blade_to_m: f64,
    pub blade_width_m: f64,
    pub blade_pose: BladePose,
}

impl Default for CamOptions {
    fn default() -> Self {
        CamOptions {
            focus_m: 1.535,
            distance_m: 3.0,
            pitch_rad: -0.32,
            fov_deg: 75.0,
            yaw_deg: 0.0,
            width: 480,
            height: 270,
            stature_m: 2.0,
            // The props kit's sword (grip at its origin, guard 0.11 m and
            // tip 0.91 m down the blade).
            blade_from_m: 0.11,
            blade_to_m: 0.91,
            blade_width_m: 0.05,
            blade_pose: BladePose::Game,
        }
    }
}

/// A pinhole camera: screen x right, y down, in pixels.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub eye: Vec3,
    pub right: Vec3,
    pub up: Vec3,
    /// Points from the scene to the eye (the view looks along -back).
    pub back: Vec3,
    /// Focal length in pixels.
    pub f: f64,
    pub w: usize,
    pub h: usize,
}

impl Camera {
    /// At `eye` looking at `at`, Y up, `fov_deg` vertical.
    pub fn look(eye: Vec3, at: Vec3, fov_deg: f64, w: usize, h: usize) -> Camera {
        let back = (eye - at).normalized();
        let right = Vec3::new(0.0, 1.0, 0.0).cross(back).normalized();
        let up = back.cross(right);
        let (s, c) = detmath::sin_cos(fov_deg.to_radians() * 0.5);
        Camera {
            eye,
            right,
            up,
            back,
            f: h as f64 * 0.5 * c / s,
            w,
            h,
        }
    }

    /// The game's follow camera on a character standing at the origin
    /// facing `forward` (horizontal), in units of `unit` per metre.
    pub fn game(forward: Vec3, opts: &CamOptions, unit: f64) -> Camera {
        let upv = Vec3::new(0.0, 1.0, 0.0);
        let his_right = forward.cross(upv);
        let (sy, cy) = detmath::sin_cos(opts.yaw_deg.to_radians());
        let look = forward.scale(cy) + his_right.scale(sy);
        let focus = upv.scale(opts.focus_m * unit);
        let (sp, cp) = detmath::sin_cos(-opts.pitch_rad);
        let d = opts.distance_m * unit;
        let eye = focus - look.scale(d * cp) + upv.scale(d * sp);
        Camera::look(eye, focus, opts.fov_deg, opts.width, opts.height)
    }

    /// Screen position and depth; `None` behind the eye.
    pub fn project(&self, p: Vec3) -> Option<(f64, f64, f64)> {
        let v = p - self.eye;
        let z = -v.dot(self.back);
        if z < 1e-6 {
            return None;
        }
        Some((
            self.w as f64 * 0.5 + self.f * v.dot(self.right) / z,
            self.h as f64 * 0.5 - self.f * v.dot(self.up) / z,
            z,
        ))
    }
}

/// A coverage mask: a pixel is in when its centre is inside a shape.
#[derive(Clone, Debug, PartialEq)]
pub struct Mask {
    pub w: usize,
    pub h: usize,
    pub px: Vec<bool>,
}

fn edge(a: (f64, f64), b: (f64, f64), p: (f64, f64)) -> f64 {
    (b.0 - a.0) * (p.1 - a.1) - (b.1 - a.1) * (p.0 - a.0)
}

impl Mask {
    pub fn new(w: usize, h: usize) -> Mask {
        Mask {
            w,
            h,
            px: vec![false; w * h],
        }
    }

    pub fn area(&self) -> usize {
        self.px.iter().filter(|&&b| b).count()
    }

    /// Fill a triangle (either winding): every pixel whose centre is
    /// inside or on an edge.
    pub fn fill_tri(&mut self, a: (f64, f64), b: (f64, f64), c: (f64, f64)) {
        let area = edge(a, b, c);
        if area.abs() < 1e-12 {
            return;
        }
        let (lx, hx) = (a.0.min(b.0).min(c.0), a.0.max(b.0).max(c.0));
        let (ly, hy) = (a.1.min(b.1).min(c.1), a.1.max(b.1).max(c.1));
        let x0 = ((lx - 0.5).ceil() as i64).max(0);
        let x1 = ((hx - 0.5).floor() as i64).min(self.w as i64 - 1);
        let y0 = ((ly - 0.5).ceil() as i64).max(0);
        let y1 = ((hy - 0.5).floor() as i64).min(self.h as i64 - 1);
        if x0 > x1 || y0 > y1 {
            return;
        }
        for y in y0..=y1 {
            let py = y as f64 + 0.5;
            for x in x0..=x1 {
                let p = (x as f64 + 0.5, py);
                let (e0, e1, e2) = (edge(a, b, p), edge(b, c, p), edge(c, a, p));
                let inside = if area > 0.0 {
                    e0 >= 0.0 && e1 >= 0.0 && e2 >= 0.0
                } else {
                    e0 <= 0.0 && e1 <= 0.0 && e2 <= 0.0
                };
                if inside {
                    self.px[y as usize * self.w + x as usize] = true;
                }
            }
        }
    }

    /// Fill a thick segment: half width `ra` at `a`, `rb` at `b` (pixels).
    /// A segment seen end on is a square of half width the larger.
    pub fn fill_segment(&mut self, a: (f64, f64), b: (f64, f64), ra: f64, rb: f64) {
        let (dx, dy) = (b.0 - a.0, b.1 - a.1);
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-9 {
            let r = ra.max(rb);
            let (p, q) = ((a.0 - r, a.1 - r), (a.0 + r, a.1 + r));
            self.fill_tri(p, (q.0, p.1), q);
            self.fill_tri(p, q, (p.0, q.1));
            return;
        }
        let (nx, ny) = (-dy / len, dx / len);
        let p0 = (a.0 + nx * ra, a.1 + ny * ra);
        let p1 = (b.0 + nx * rb, b.1 + ny * rb);
        let p2 = (b.0 - nx * rb, b.1 - ny * rb);
        let p3 = (a.0 - nx * ra, a.1 - ny * ra);
        self.fill_tri(p0, p1, p2);
        self.fill_tri(p0, p2, p3);
    }

    pub fn union_with(&mut self, o: &Mask) {
        for (a, b) in self.px.iter_mut().zip(&o.px) {
            *a |= *b;
        }
    }

    /// Intersection over union (1 when both are empty).
    pub fn iou(&self, o: &Mask) -> f64 {
        let (mut i, mut u) = (0usize, 0usize);
        for (a, b) in self.px.iter().zip(&o.px) {
            i += (*a && *b) as usize;
            u += (*a || *b) as usize;
        }
        if u == 0 {
            1.0
        } else {
            i as f64 / u as f64
        }
    }

    /// Rows from the first to the last covered one (0 when empty).
    pub fn rows(&self) -> usize {
        let mut lo = None;
        let mut hi = 0;
        for y in 0..self.h {
            if self.px[y * self.w..(y + 1) * self.w].iter().any(|&b| b) {
                lo.get_or_insert(y);
                hi = y;
            }
        }
        lo.map(|l| hi - l + 1).unwrap_or(0)
    }

    /// Binary PGM: body black, `blade` grey, the rest white.
    pub fn to_pgm(&self, blade: Option<&Mask>) -> Vec<u8> {
        let mut out = format!("P5\n{} {}\n255\n", self.w, self.h).into_bytes();
        for i in 0..self.px.len() {
            out.push(if blade.is_some_and(|b| b.px[i]) {
                128
            } else if self.px[i] {
                0
            } else {
                255
            });
        }
        out
    }
}

/// Screen path of a point: its length (pixels), how consistently it turns
/// one way (`|sum of signed turns| / sum of |turns|`, each turn weighted
/// by the shorter of its two steps; 1 when it turns under 10 deg a step
/// on average), and the total turn (degrees, unweighted). Steps under
/// half a pixel are skipped.
pub fn path_arc(pts: &[(f64, f64)]) -> (f64, f64, f64) {
    let mut kept: Vec<(f64, f64)> = Vec::new();
    let mut len = 0.0;
    for &p in pts {
        match kept.last() {
            None => kept.push(p),
            Some(&q) => {
                let d = ((p.0 - q.0).powi(2) + (p.1 - q.1).powi(2)).sqrt();
                if d >= 0.5 {
                    len += d;
                    kept.push(p);
                }
            }
        }
    }
    let (mut signed, mut total, mut weight, mut turn) = (0.0, 0.0, 0.0, 0.0);
    for w in kept.windows(3) {
        let a = (w[1].0 - w[0].0, w[1].1 - w[0].1);
        let b = (w[2].0 - w[1].0, w[2].1 - w[1].1);
        let (la, lb) = (
            (a.0 * a.0 + a.1 * a.1).sqrt(),
            (b.0 * b.0 + b.1 * b.1).sqrt(),
        );
        let cos = ((a.0 * b.0 + a.1 * b.1) / (la * lb)).clamp(-1.0, 1.0);
        let ang = detmath::acos(cos).to_degrees();
        let cross = a.0 * b.1 - a.1 * b.0;
        // A turn counts by the shorter of its two steps: the tip's
        // jitter where it slows at either end is not a wiggle.
        let wgt = la.min(lb);
        signed += if cross >= 0.0 { ang * wgt } else { -ang * wgt };
        total += ang * wgt;
        weight += wgt;
        turn += ang;
    }
    // Under 10 deg a step on average (weighted): it barely turns.
    let consistency = if total < 10.0 * weight || total <= 0.0 {
        1.0
    } else {
        signed.abs() / total
    };
    (len, consistency, turn)
}

/// The skinned mesh, ready to pose.
pub struct Mesh {
    verts: Vec<SkinVert>,
    tris: Vec<[usize; 3]>,
    rest_inv: Vec<Affine>,
}

impl Mesh {
    pub fn build(doc: &Document, rig: &Rig) -> Result<Mesh, String> {
        let (verts, tris) = skinned_mesh(doc, rig)?;
        let rest_inv = rig
            .rest_world
            .iter()
            .map(|m| m.inverse().unwrap_or(Affine::IDENTITY))
            .collect();
        Ok(Mesh {
            verts,
            tris,
            rest_inv,
        })
    }

    pub fn triangles(&self) -> usize {
        self.tris.len()
    }

    fn posed(&self, world: &[Affine]) -> Vec<Vec3> {
        let skin: Vec<Affine> = world
            .iter()
            .zip(&self.rest_inv)
            .map(|(w, r)| w.mul(r))
            .collect();
        self.verts
            .iter()
            .map(|v| {
                let mut acc = Vec3::ZERO;
                let mut total = 0.0;
                for &(j, w) in &v.inf {
                    acc = acc + skin[j].apply(v.p).scale(w);
                    total += w;
                }
                if total > 0.0 {
                    acc.scale(1.0 / total)
                } else {
                    v.p
                }
            })
            .collect()
    }
}

/// Readability of one clip at the game camera.
#[derive(Clone, Debug, Default)]
pub struct ReadCam {
    pub peak_frame: usize,
    pub load_frame: usize,
    pub settle_frame: usize,
    /// The frame of the strike furthest from neutral, and how far.
    pub key_frame: usize,
    pub pose_change: f64,
    /// The same for the body alone (no blade), from the windup's start to
    /// the settle, and its frame: does the body take part, or only a thin
    /// blade?
    pub body_change: f64,
    pub body_frame: usize,
    pub sweep: f64,
    /// Tip path in neutral body heights, its turn consistency and total
    /// turn (degrees), and their product.
    pub arc_len: f64,
    pub arc_consistency: f64,
    pub arc_turn_deg: f64,
    pub arc: f64,
    pub load_hold_frames: usize,
    /// Neutral body height on screen (pixels) and area.
    pub body_px: usize,
    pub body_area: usize,
    pub blade: bool,
    /// 0..1: the mean of the four parts, each against what a clearly
    /// readable attack reaches (`READ_FULL`).
    pub score: f64,
}

/// The read score from the body's change, the blade's sweep, the arc and
/// the held load: each part against `READ_FULL`, capped at 1, averaged;
/// the held load counts only as far as the strike after it reads as an
/// arc (a still pose is not a held load).
pub fn score_of(body: f64, sweep: f64, arc: f64, hold: usize) -> f64 {
    let p = |v: f64, i: usize| (v / READ_FULL[i]).min(1.0);
    (p(body, 0) + p(sweep, 1) + p(arc, 2) + p(hold as f64, 3) * p(arc, 2)) / 4.0
}

/// What a clearly readable attack reaches in each part (calibrated on
/// HLH's attacks against its idle: `docs/motion-check.md`).
pub const READ_FULL: [f64; 4] = [0.7, 0.8, 2.0, 4.0];

impl ReadCam {
    pub fn to_json(&self) -> Json {
        let r = |v: f64, p: i32| {
            let k = 10f64.powi(p);
            (v * k).round() / k
        };
        Json::obj(vec![
            ("peak_frame", Json::num(self.peak_frame as f64)),
            ("load_frame", Json::num(self.load_frame as f64)),
            ("settle_frame", Json::num(self.settle_frame as f64)),
            ("key_frame", Json::num(self.key_frame as f64)),
            ("pose_change", Json::num(r(self.pose_change, 3))),
            ("body_change", Json::num(r(self.body_change, 3))),
            ("body_frame", Json::num(self.body_frame as f64)),
            ("sweep", Json::num(r(self.sweep, 3))),
            ("arc_len", Json::num(r(self.arc_len, 3))),
            ("arc_consistency", Json::num(r(self.arc_consistency, 3))),
            ("arc_turn_deg", Json::num(r(self.arc_turn_deg, 1))),
            ("arc", Json::num(r(self.arc, 3))),
            ("load_hold_frames", Json::num(self.load_hold_frames as f64)),
            ("body_px", Json::num(self.body_px as f64)),
            ("body_area", Json::num(self.body_area as f64)),
            ("blade", Json::Bool(self.blade)),
            ("score", Json::num(r(self.score, 3))),
        ])
    }

    pub fn line(&self) -> String {
        format!(
            "{:<18} read at game camera {:.2}: body {:.2} (f{}), pose {:.2} (f{}), sweep {:.2}, arc {:.2} ({:.2} heights x {:.2}), load held {} fr; strike f{}-{}-{}\n",
            "",
            self.score,
            self.body_change,
            self.body_frame,
            self.pose_change,
            self.key_frame,
            self.sweep,
            self.arc,
            self.arc_len,
            self.arc_consistency,
            self.load_hold_frames,
            self.load_frame,
            self.peak_frame,
            self.settle_frame
        )
    }
}

/// Metres to the GLB's units: his head joint goes to `HEAD_SHARE` of the
/// stature.
pub fn unit_of(subject: &Subject, opts: &CamOptions) -> f64 {
    if opts.stature_m <= 0.0 {
        return 1.0;
    }
    match subject.map.node_of("DEF-spine.006") {
        Some(h) if subject.rig.head(h).y > 1e-6 => {
            subject.rig.head(h).y / (opts.stature_m * HEAD_SHARE)
        }
        _ => 1.0,
    }
}

/// The blade's guard and tip in the hand socket's space (GLB units).
fn blade_local(opts: &CamOptions, unit: f64) -> (Vec3, Vec3) {
    let (grip, dir) = match opts.blade_pose {
        BladePose::Game => {
            let q = Quat::new(HELD_ROT[3], HELD_ROT[0], HELD_ROT[1], HELD_ROT[2]).normalized();
            (
                Vec3::new(HELD_GRIP[0], HELD_GRIP[1], HELD_GRIP[2]),
                q.rotate_vec(Vec3::new(0.0, 1.0, 0.0)),
            )
        }
        BladePose::Socket => (Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0)),
    };
    (
        (grip + dir.scale(opts.blade_from_m)).scale(unit),
        (grip + dir.scale(opts.blade_to_m)).scale(unit),
    )
}

/// One frame's silhouettes at the camera, and the tip on screen.
pub struct Shot {
    pub body: Mask,
    pub blade: Option<Mask>,
    pub tip: Option<(f64, f64)>,
}

/// Everything needed to film a character.
pub struct Stage {
    pub cam: Camera,
    pub unit: f64,
    mesh: Mesh,
    socket: Option<usize>,
    hand: Option<usize>,
    blade: (Vec3, Vec3),
    width: f64,
}

impl Stage {
    pub fn new(doc: &Document, subject: &Subject, opts: &CamOptions) -> Result<Stage, String> {
        let unit = unit_of(subject, opts);
        let side = subject
            .opts
            .weapon_side
            .clone()
            .unwrap_or_else(|| "R".to_string());
        let hand = subject.map.node_of(&format!("DEF-hand.{side}"));
        let socket = hand.and_then(|h| {
            subject.rig.children[h]
                .iter()
                .copied()
                .find(|&c| subject.rig.names[c] == format!("Socket_Hand_{side}"))
        });
        let socket = if subject.opts.weapon_side.is_some() {
            socket
        } else {
            None
        };
        Ok(Stage {
            cam: Camera::game(subject.forward(), opts, unit),
            unit,
            mesh: Mesh::build(doc, &subject.rig)?,
            socket,
            hand,
            blade: blade_local(opts, unit),
            width: opts.blade_width_m * unit,
        })
    }

    /// The blade's guard and tip in the world, posed.
    pub fn blade_at(&self, world: &[Affine]) -> Option<(Vec3, Vec3)> {
        self.socket
            .map(|s| (world[s].apply(self.blade.0), world[s].apply(self.blade.1)))
    }

    pub fn shoot(&self, world: &[Affine]) -> Shot {
        let cam = &self.cam;
        let mut body = Mask::new(cam.w, cam.h);
        let pts: Vec<Option<(f64, f64)>> = self
            .mesh
            .posed(world)
            .into_iter()
            .map(|p| cam.project(p).map(|(x, y, _)| (x, y)))
            .collect();
        for t in &self.mesh.tris {
            if let (Some(a), Some(b), Some(c)) = (pts[t[0]], pts[t[1]], pts[t[2]]) {
                body.fill_tri(a, b, c);
            }
        }
        let mut tip = None;
        let blade = self.blade_at(world).and_then(|(g, t)| {
            let (pg, pt) = (cam.project(g)?, cam.project(t)?);
            let mut m = Mask::new(cam.w, cam.h);
            let half = |z: f64| cam.f * self.width * 0.5 / z;
            m.fill_segment((pg.0, pg.1), (pt.0, pt.1), half(pg.2), half(pt.2));
            tip = Some((pt.0, pt.1));
            Some(m)
        });
        if tip.is_none() {
            tip = self
                .hand
                .and_then(|h| cam.project(world[h].t))
                .map(|(x, y, _)| (x, y));
        }
        Shot { body, blade, tip }
    }

    /// Readability of a clip (`frames` at `dt`), against `neutral`.
    pub fn read(
        &self,
        rig: &Rig,
        frames: &[Vec<Option<Trs>>],
        dt: f64,
        neutral: &[Option<Trs>],
    ) -> (ReadCam, Vec<Shot>) {
        let n = frames.len();
        let world: Vec<Vec<Affine>> = frames.iter().map(|f| rig.world(f)).collect();
        let shots: Vec<Shot> = world.iter().map(|w| self.shoot(w)).collect();
        let full = |s: &Shot| {
            let mut m = s.body.clone();
            if let Some(b) = &s.blade {
                m.union_with(b);
            }
            m
        };
        let ns = self.shoot(&rig.world(neutral));
        let neutral_full = full(&ns);
        let body_area = ns.body.area().max(1);
        let body_px = ns.body.rows().max(1);
        let mut out = ReadCam {
            body_px,
            body_area,
            blade: self.socket.is_some(),
            ..ReadCam::default()
        };
        if n < 3 {
            return (out, shots);
        }
        let hand_p: Vec<Vec3> = (0..n)
            .map(|f| self.hand.map(|h| world[f][h].t).unwrap_or(Vec3::ZERO))
            .collect();
        let tip_p: Vec<Vec3> = (0..n)
            .map(|f| match self.blade_at(&world[f]) {
                Some((_, t)) => t,
                None => hand_p[f],
            })
            .collect();
        let (peak, load, windup) = strike_timing(&hand_p, &tip_p, dt);
        let speed = |f: usize| {
            let (a, b) = (f.saturating_sub(1), (f + 1).min(n - 1));
            (tip_p[b] - tip_p[a]).length() / ((b - a) as f64 * dt)
        };
        let top = speed(peak);
        let mut settle = n - 1;
        for f in peak + 1..n {
            if speed(f) < HOLD_SHARE * top {
                settle = f;
                break;
            }
        }
        out.peak_frame = peak;
        out.load_frame = load;
        out.settle_frame = settle;
        let fulls: Vec<Mask> = shots.iter().map(full).collect();
        let mut sweep = Mask::new(self.cam.w, self.cam.h);
        let mut path = Vec::new();
        for f in load..=settle {
            let d = 1.0 - fulls[f].iou(&neutral_full);
            if d > out.pose_change {
                out.pose_change = d;
                out.key_frame = f;
            }
            if let Some(b) = &shots[f].blade {
                sweep.union_with(b);
            }
            if let Some(t) = shots[f].tip {
                path.push(t);
            }
        }
        if out.pose_change == 0.0 {
            out.key_frame = load;
        }
        out.body_frame = windup;
        for f in windup..=settle {
            let d = 1.0 - shots[f].body.iou(&ns.body);
            if d > out.body_change {
                out.body_change = d;
                out.body_frame = f;
            }
        }
        out.sweep = sweep.area() as f64 / body_area as f64;
        let (len, cons, turn) = path_arc(&path);
        out.arc_len = len / body_px as f64;
        out.arc_consistency = cons;
        out.arc_turn_deg = turn;
        out.arc = out.arc_len * cons;
        // Held load: the longest run of near-still frames in the second
        // before the peak.
        let win = ((1.0 / dt).round() as usize).max(1);
        let mut run = 0;
        for f in peak.saturating_sub(win).max(1)..peak {
            if 1.0 - fulls[f].iou(&fulls[f - 1]) < STILL_SHARE {
                run += 1;
                out.load_hold_frames = out.load_hold_frames.max(run);
            } else {
                run = 0;
            }
        }
        out.score = score_of(out.body_change, out.sweep, out.arc, out.load_hold_frames);
        (out, shots)
    }
}
