//! Fit a fused 3D key pose onto a standard (`DEF-`) humanoid rig, bone by
//! bone. See `keypose/mod.rs` for the pipeline and `docs/keypose.md` for
//! the numbers.

use super::{lm, Pose3};
use crate::contact::{hinge_axis, solve_arm};
use crate::detmath;
use crate::glb::{canonical_quat, Document};
use crate::humanoid::{self, Mapping};
use crate::math::{shortest_arc, Mat3, Quat, Vec3};
use crate::rig::{Rig, Trs};

const UP: Vec3 = Vec3 {
    x: 0.0,
    y: 1.0,
    z: 0.0,
};
const SIDES: [&str; 2] = ["L", "R"];

/// Limits and choices of the fit.
#[derive(Clone, Debug)]
pub struct FitParams {
    /// Hand that holds the sword (its socket follows the drawn blade):
    /// 0 = L, 1 = R, None = no sword.
    pub sword_hand: Option<usize>,
    /// Both hands on the grip: Some(true/false), or None to decide from
    /// the drawings (wrists closer than `two_hand_gap` in half the keys).
    pub two_hands: Option<bool>,
    /// Wrist distance (m on a 1.75 m body) under which a key is two-handed.
    pub two_hand_gap: f64,
    /// A foot whose drawn ankle is within this (m on a 1.75 m body) of
    /// its standing height is planted: flat, on the floor.
    pub plant_height: f64,
    /// Share of the torso's lean the pelvis takes.
    pub pelvis_follow: f64,
    /// Caps (degrees), a few under `motion check`'s limits.
    pub torso_max_deg: f64,
    pub head_max_deg: f64,
    pub wrist_max_deg: f64,
    pub ankle_max_deg: f64,
    /// Forearm roll about its own axis the sword hand may borrow.
    pub forearm_roll_max_deg: f64,
    /// Elbow and knee bend range (motion check: -10 .. 160).
    pub flex_min_deg: f64,
    pub flex_max_deg: f64,
}

impl Default for FitParams {
    fn default() -> Self {
        FitParams {
            sword_hand: Some(1),
            two_hands: None,
            two_hand_gap: 0.22,
            plant_height: 0.10,
            pelvis_follow: 0.4,
            torso_max_deg: 75.0,
            head_max_deg: 80.0,
            wrist_max_deg: 82.0,
            ankle_max_deg: 60.0,
            forearm_roll_max_deg: 80.0,
            flex_min_deg: -5.0,
            flex_max_deg: 150.0,
        }
    }
}

/// The rig with the bones the fit uses looked up once.
pub struct RigInfo {
    pub rig: Rig,
    pub map: Mapping,
    /// Rig frame: forward, left (= up x forward), up.
    pub fwd: Vec3,
    pub left: Vec3,
    pub hips: usize,
    pub spine: Vec<usize>,
    pub neck: Option<usize>,
    pub head: Option<usize>,
    pub shoulder: [Option<usize>; 2],
    pub upper: [usize; 2],
    pub fore: [usize; 2],
    pub hand: [usize; 2],
    pub thigh: [usize; 2],
    pub shin: [usize; 2],
    pub foot: [usize; 2],
    pub toe: [Option<usize>; 2],
    pub socket: [Option<usize>; 2],
    /// Floor height and the standing ankle height above it.
    pub ground: f64,
    pub ankle_h: [f64; 2],
    /// thigh + shin, averaged over both legs.
    pub leg_len: f64,
    /// Body scale: hip height / 0.95 (1 on a 1.75 m body).
    pub k: f64,
}

impl RigInfo {
    pub fn new(doc: &Document) -> Result<RigInfo, String> {
        let rig = Rig::from_doc(doc)?;
        let map = humanoid::map_rig(&rig);
        if !map.complete() {
            return Err(format!("rig is missing {}", map.missing.join(", ")));
        }
        let node = |n: &str| map.node_of(n);
        let req = |n: &str| node(n).ok_or_else(|| format!("rig has no {n}"));
        let fwd = crate::animate::forward(&rig, &map);
        let left = UP.cross(fwd).normalized();
        let side = |b: &str| -> Result<[usize; 2], String> {
            Ok([req(&format!("DEF-{b}.L"))?, req(&format!("DEF-{b}.R"))?])
        };
        let opt = |b: &str| [node(&format!("DEF-{b}.L")), node(&format!("DEF-{b}.R"))];
        let socket = [
            find_child(&rig, req("DEF-hand.L")?, "Socket_Hand_L"),
            find_child(&rig, req("DEF-hand.R")?, "Socket_Hand_R"),
        ];
        let hips = req("DEF-spine")?;
        let spine: Vec<usize> = ["DEF-spine.001", "DEF-spine.002", "DEF-spine.003"]
            .iter()
            .filter_map(|n| node(n))
            .collect();
        let neck = node("DEF-spine.004").or_else(|| node("DEF-spine.005"));
        let head = node("DEF-spine.006");
        let thigh = side("thigh")?;
        let shin = side("shin")?;
        let foot = side("foot")?;
        let toe = opt("toe");
        let mut ground = f64::INFINITY;
        for s in 0..2 {
            ground = ground.min(rig.head(foot[s]).y);
            if let Some(t) = toe[s] {
                ground = ground.min(rig.head(t).y);
            }
        }
        // A toe bone's head sits inside the toes, a little above the sole.
        let k = (rig.head(hips).y - ground) / 0.95;
        let ground = ground - 0.02 * k;
        let ankle_h = [rig.head(foot[0]).y - ground, rig.head(foot[1]).y - ground];
        let mut leg_len = 0.0;
        for s in 0..2 {
            leg_len += (rig.head(shin[s]) - rig.head(thigh[s])).length()
                + (rig.head(foot[s]) - rig.head(shin[s])).length();
        }
        leg_len *= 0.5;
        Ok(RigInfo {
            fwd,
            left,
            hips,
            spine,
            neck,
            head,
            shoulder: opt("shoulder"),
            upper: side("upper_arm")?,
            fore: side("forearm")?,
            hand: side("hand")?,
            thigh,
            shin,
            foot,
            toe,
            socket,
            ground,
            ankle_h,
            leg_len,
            k,
            rig,
            map,
        })
    }

    /// Character frame (X left, Y up, Z forward) to rig world directions.
    pub fn to_rig(&self, v: Vec3) -> Vec3 {
        self.left.scale(v.x) + UP.scale(v.y) + self.fwd.scale(v.z)
    }

    pub fn from_rig(&self, v: Vec3) -> Vec3 {
        Vec3::new(v.dot(self.left), v.y, v.dot(self.fwd))
    }

    fn rest_rot(&self, n: usize) -> Quat {
        self.rig.rest_world[n].rotation()
    }

    /// A bone's direction to `child` in its own rest frame.
    fn local_dir(&self, n: usize, child: usize) -> Vec3 {
        self.rest_rot(n)
            .conj()
            .rotate_vec(self.rig.head(child) - self.rig.head(n))
            .normalized()
    }

    /// The hand's pointing direction in its own frame (toward the
    /// fingers; else the bone's own Y).
    fn hand_dir(&self, s: usize) -> Vec3 {
        let h = self.hand[s];
        let mut best: Option<(f64, usize)> = None;
        for &c in &self.rig.children[h] {
            if self.rig.names[c].starts_with("Socket") {
                continue;
            }
            let d = (self.rig.head(c) - self.rig.head(h)).length();
            if d > 1e-6 && best.is_none_or(|(bd, _)| d > bd) {
                best = Some((d, c));
            }
        }
        match best {
            Some((_, c)) => self.local_dir(h, c),
            None => UP,
        }
    }
}

fn find_child(rig: &Rig, parent: usize, name: &str) -> Option<usize> {
    rig.children[parent]
        .iter()
        .copied()
        .find(|&c| rig.names[c] == name)
}

/// Rotation taking the frame (primary `a0`, secondary `b0`) onto (`a1`, `b1`).
pub fn frame_rot(a0: Vec3, b0: Vec3, a1: Vec3, b1: Vec3) -> Quat {
    let basis = |a: Vec3, b: Vec3| {
        let x = a.normalized();
        let y = (b - x.scale(b.dot(x))).normalized();
        Mat3::from_cols(x, y, x.cross(y))
    };
    let m0 = basis(a0, b0);
    let m1 = basis(a1, b1);
    Quat::from_mat3(m1.mul_mat(m0.transpose())).normalized()
}

/// `q` to the power `f` (slerp from identity).
fn pow(q: Quat, f: f64) -> Quat {
    Quat::IDENTITY.slerp(q, f)
}

/// Turn of `q` about unit `axis`, radians in (-pi, pi].
pub fn twist_angle(q: Quat, axis: Vec3) -> f64 {
    let v = Vec3::new(q.x, q.y, q.z);
    let (mut w, mut p) = (q.w, v.dot(axis));
    if w < 0.0 {
        w = -w;
        p = -p;
    }
    let len = (w * w + p * p).sqrt();
    if len < 1e-12 {
        return 0.0;
    }
    2.0 * detmath::asin((p / len).clamp(-1.0, 1.0))
}

/// Signed angle from `a` to `b` about `axis`, both projected onto the
/// plane normal to it.
pub fn signed_angle(a: Vec3, b: Vec3, axis: Vec3) -> f64 {
    let ap = a - axis.scale(a.dot(axis));
    let bp = b - axis.scale(b.dot(axis));
    if ap.length() < 1e-9 || bp.length() < 1e-9 {
        return 0.0;
    }
    let c = (ap.dot(bp) / (ap.length() * bp.length())).clamp(-1.0, 1.0);
    let s = if ap.cross(bp).dot(axis) >= 0.0 {
        1.0
    } else {
        -1.0
    };
    s * detmath::acos(c)
}

pub fn angle_deg(a: Vec3, b: Vec3) -> f64 {
    if a.length() < 1e-12 || b.length() < 1e-12 {
        return 0.0;
    }
    detmath::acos((a.dot(b) / (a.length() * b.length())).clamp(-1.0, 1.0)).to_degrees()
}

/// Set `node`'s world rotation by writing its local one.
pub fn set_world_rot(rig: &Rig, locals: &mut [Option<Trs>], node: usize, w: Quat) {
    let world = rig.world(locals);
    let pr = rig.parent[node]
        .map(|p| world[p].rotation())
        .unwrap_or(Quat::IDENTITY);
    let mut l = locals[node].unwrap_or(rig.rest[node]);
    l.r = canonical_quat(pr.conj().mul(w).normalized());
    locals[node] = Some(l);
}

fn world_rot(rig: &Rig, locals: &[Option<Trs>], node: usize) -> Quat {
    rig.world(locals)[node].rotation()
}

/// Keep `node` within `max_deg` of its rest rotation relative to `parent`.
/// Returns the excess that was removed (degrees).
fn cap(
    info: &RigInfo,
    locals: &mut [Option<Trs>],
    node: usize,
    parent: usize,
    max_deg: f64,
) -> f64 {
    let rig = &info.rig;
    let wp = world_rot(rig, locals, parent);
    let wn = world_rot(rig, locals, node);
    let rel = wp.conj().mul(wn);
    let rest = info.rest_rot(parent).conj().mul(info.rest_rot(node));
    let ang = rest.angle_to(rel).to_degrees();
    if ang <= max_deg {
        return 0.0;
    }
    let r2 = rest.slerp(rel, max_deg / ang);
    set_world_rot(rig, locals, node, wp.mul(r2));
    ang - max_deg
}

/// Both wrists within `wrist_max_deg` (between keys the coil and the
/// interpolation can carry a wrist past what each key had).
pub fn cap_wrists(info: &RigInfo, locals: &mut [Option<Trs>], params: &FitParams) {
    for s in 0..2 {
        cap(
            info,
            locals,
            info.hand[s],
            info.fore[s],
            params.wrist_max_deg,
        );
    }
}

/// One key pose fitted onto the rig.
#[derive(Clone, Debug)]
pub struct KeyFit {
    /// Local transforms of every node (None = rest).
    pub locals: Vec<Option<Trs>>,
    /// Feet drawn on the floor.
    pub planted: [bool; 2],
    /// Landmarks in rig world (scaled), hip midpoint over the origin.
    pub targets: Vec<Vec3>,
    /// `(part, degrees)`: angle between the fitted bone and the drawn one.
    pub errors: Vec<(String, f64)>,
    /// Socket blade axis vs the drawn blade, degrees.
    pub blade_err_deg: Option<f64>,
    /// Off hand's socket to its place on the grip, metres.
    pub grip_err_m: Option<f64>,
    /// Caps that bit, `(joint, degrees removed)`.
    pub capped: Vec<(String, f64)>,
    pub two_hands: bool,
    /// Off hand's world rotation relative to the sword socket.
    pub off_rel: Option<Quat>,
    /// Signed grip gap along the blade.
    pub grip_gap: f64,
}

/// Drawing metres -> rig metres: the rig's leg length over the drawn one,
/// over every key and both legs (weighted by confidence).
pub fn sheet_scale(info: &RigInfo, poses: &[Pose3]) -> f64 {
    let (mut drawn, mut w) = (0.0, 0.0);
    for p in poses {
        for (h, kn, an) in [
            (lm::HIP_L, lm::KN_L, lm::AN_L),
            (lm::HIP_R, lm::KN_R, lm::AN_R),
        ] {
            let c = p.conf[h].min(p.conf[kn]).min(p.conf[an]);
            drawn += c * ((p.p[kn] - p.p[h]).length() + (p.p[an] - p.p[kn]).length());
            w += c;
        }
    }
    if w < 1e-9 || drawn < 1e-9 {
        return 1.0;
    }
    info.leg_len / (drawn / w)
}

/// Whether the drawings hold the sword in both hands.
pub fn two_handed(info: &RigInfo, poses: &[Pose3], scale: f64, params: &FitParams) -> bool {
    if let Some(t) = params.two_hands {
        return t;
    }
    let near = poses
        .iter()
        .filter(|p| (p.p[lm::WR_L] - p.p[lm::WR_R]).length() * scale < params.two_hand_gap * info.k)
        .count();
    2 * near >= poses.len()
}

struct Targets<'a> {
    info: &'a RigInfo,
    pose: &'a Pose3,
    scale: f64,
}

impl Targets<'_> {
    fn at(&self, i: usize) -> Vec3 {
        let p = self.pose.p[i];
        let v = self.info.to_rig(p.scale(self.scale));
        Vec3::new(v.x, v.y + self.info.ground, v.z)
    }
    fn mid(&self, a: usize, b: usize) -> Vec3 {
        (self.at(a) + self.at(b)).scale(0.5)
    }
    fn conf(&self, i: usize) -> f64 {
        self.pose.conf[i]
    }
}

/// Fit an upper/lower/end limb as a hinge. Returns `(upper err, lower err)`
/// in degrees and the bend clamp applied.
#[allow(clippy::too_many_arguments)]
fn fit_limb(
    info: &RigInfo,
    locals: &mut [Option<Trs>],
    upper: usize,
    lower: usize,
    end: usize,
    ut: Vec3,
    ft: Vec3,
    params: &FitParams,
) -> (f64, f64, f64) {
    let rig = &info.rig;
    let wu = world_rot(rig, locals, upper);
    let lu = info.local_dir(upper, lower);
    let hinge = hinge_axis(rig, upper, lower, end);
    let au = hinge.map(|n| info.rest_rot(upper).conj().rotate_vec(n));
    let swing = shortest_arc(wu.rotate_vec(lu), ut.normalized()).mul(wu);
    let bend = angle_deg(ut, ft);
    let w_new = match au {
        Some(au) if bend > 8.0 => {
            let nt = ut.cross(ft).normalized();
            let full = frame_rot(lu, au, ut, nt);
            // A nearly straight limb's bend side is noise: keep the swing's
            // own hinge when it disagrees.
            if bend < 25.0 && swing.rotate_vec(au).dot(nt) < 0.0 {
                swing
            } else {
                full
            }
        }
        _ => swing,
    };
    set_world_rot(rig, locals, upper, w_new);
    let wl = world_rot(rig, locals, lower);
    let ll = info.local_dir(lower, end);
    let udir = w_new.rotate_vec(lu);
    let mut clamp = 0.0;
    let wl_new = match au {
        Some(au) => {
            let n = w_new.rotate_vec(au).normalized();
            let c = wl.rotate_vec(ll);
            let turn = signed_angle(c, ft, n);
            let mut wl2 = Quat::from_axis_angle(n, turn).mul(wl);
            let flex = signed_angle(udir, wl2.rotate_vec(ll), n).to_degrees();
            let want = flex.clamp(params.flex_min_deg, params.flex_max_deg);
            if (want - flex).abs() > 1e-9 {
                clamp = want - flex;
                wl2 = Quat::from_axis_angle(n, (want - flex).to_radians()).mul(wl2);
            }
            wl2
        }
        None => shortest_arc(wl.rotate_vec(ll), ft.normalized()).mul(wl),
    };
    set_world_rot(rig, locals, lower, wl_new);
    let world = rig.world(locals);
    let eu = angle_deg(world[lower].t - world[upper].t, ut);
    let el = angle_deg(world[end].t - world[lower].t, ft);
    (eu, el, clamp)
}

/// Turn the hand so its socket's +Y (the blade) follows `blade`, the
/// socket's X edge toward `edge` when given; the forearm lends its roll,
/// the wrist is capped. Returns the remaining blade error (degrees).
fn aim_sword(
    info: &RigInfo,
    locals: &mut [Option<Trs>],
    s: usize,
    blade: Vec3,
    edge: Option<Vec3>,
    params: &FitParams,
    capped: &mut Vec<(String, f64)>,
) -> Option<f64> {
    let rig = &info.rig;
    let sock = info.socket[s]?;
    let (hand, fore) = (info.hand[s], info.fore[s]);
    let lsk = rig.rest[sock].r;
    let wh = world_rot(rig, locals, hand);
    let axis = wh.mul(lsk).rotate_vec(UP);
    let mut wh1 = shortest_arc(axis, blade).mul(wh);
    if let Some(e) = edge {
        let ep = e - blade.scale(e.dot(blade));
        if ep.length() > 1e-6 {
            let x = wh1.mul(lsk).rotate_vec(Vec3::new(1.0, 0.0, 0.0));
            let a = signed_angle(x, ep, blade);
            let b = signed_angle(x, ep.scale(-1.0), blade);
            let t = if a.abs() <= b.abs() { a } else { b };
            wh1 = Quat::from_axis_angle(blade, t).mul(wh1);
        }
    }
    place_hand(info, locals, s, wh1, params, capped);
    let got = world_rot(rig, locals, sock).rotate_vec(UP);
    let _ = fore;
    Some(angle_deg(got, blade))
}

/// Give the hand world rotation `want`: the forearm takes up to
/// `forearm_roll_max_deg` of the twist about its own axis, the wrist is
/// capped at `wrist_max_deg`.
fn place_hand(
    info: &RigInfo,
    locals: &mut [Option<Trs>],
    s: usize,
    want: Quat,
    params: &FitParams,
    capped: &mut Vec<(String, f64)>,
) {
    let rig = &info.rig;
    let (hand, fore) = (info.hand[s], info.fore[s]);
    let rest_rel = info.rest_rot(fore).conj().mul(info.rest_rot(hand));
    let wf = world_rot(rig, locals, fore);
    let af = wf.rotate_vec(info.local_dir(fore, hand)).normalized();
    let delta = want.mul(wf.mul(rest_rel).conj());
    let tw = twist_angle(delta, af);
    let lim = params.forearm_roll_max_deg.to_radians();
    let roll = tw.clamp(-lim, lim);
    set_world_rot(rig, locals, fore, Quat::from_axis_angle(af, roll).mul(wf));
    set_world_rot(rig, locals, hand, want);
    let ex = cap(info, locals, hand, fore, params.wrist_max_deg);
    if ex > 0.0 {
        capped.push((format!("wrist.{}", SIDES[s]), ex));
    }
}

/// Fit one key pose. `edge`: the sword's swing direction at this key
/// (the blade's edge leads), when known.
pub fn fit_key(
    info: &RigInfo,
    pose: &Pose3,
    scale: f64,
    two_hands: bool,
    edge: Option<Vec3>,
    params: &FitParams,
) -> KeyFit {
    let rig = &info.rig;
    let t = Targets { info, pose, scale };
    let mut locals: Vec<Option<Trs>> = vec![None; rig.names.len()];
    let mut errors = Vec::new();
    let mut capped = Vec::new();

    // Hips: the hip line, tilted by part of the torso's lean.
    let l0 = info.rig.head(info.thigh[0]) - info.rig.head(info.thigh[1]);
    let lt = t.at(lm::HIP_L) - t.at(lm::HIP_R);
    let torso = t.mid(lm::SH_L, lm::SH_R) - t.mid(lm::HIP_L, lm::HIP_R);
    let up_p = (UP.scale(1.0 - params.pelvis_follow)
        + torso.normalized().scale(params.pelvis_follow))
    .normalized();
    let qh = frame_rot(l0, UP, lt, up_p);
    set_world_rot(
        rig,
        &mut locals,
        info.hips,
        qh.mul(info.rest_rot(info.hips)),
    );

    // Chest: the shoulder line and the torso; the spine bends evenly.
    let ls0 = info.rig.head(info.upper[0]) - info.rig.head(info.upper[1]);
    let up0 = (info.rig.head(info.upper[0]) + info.rig.head(info.upper[1])).scale(0.5)
        - (info.rig.head(info.thigh[0]) + info.rig.head(info.thigh[1])).scale(0.5);
    let lst = t.at(lm::SH_L) - t.at(lm::SH_R);
    // The torso's lean leads (it is what reads); the shoulder line sets
    // the twist about it.
    let qc = frame_rot(up0, ls0, torso, lst);
    let mut d = qc.mul(qh.conj());
    let nsp = info.spine.len().max(1) as f64;
    let set_spine = |locals: &mut Vec<Option<Trs>>, d: Quat| {
        for (i, &b) in info.spine.iter().enumerate() {
            let f = (i + 1) as f64 / nsp;
            set_world_rot(rig, locals, b, pow(d, f).mul(qh).mul(info.rest_rot(b)));
        }
    };
    let chord = |locals: &[Option<Trs>]| {
        let w = rig.world(locals);
        (w[info.upper[0]].t + w[info.upper[1]].t).scale(0.5)
            - (w[info.thigh[0]].t + w[info.thigh[1]].t).scale(0.5)
    };
    // A bent chain's chord lags its top bone: bend on until the chord
    // (hips to shoulders) points where the drawn torso does.
    for _ in 0..4 {
        set_spine(&mut locals, d);
        let corr = shortest_arc(chord(&locals).normalized(), torso.normalized());
        d = corr.mul(d);
    }
    let dang = Quat::IDENTITY.angle_to(d).to_degrees();
    if dang > params.torso_max_deg {
        capped.push(("torso".into(), dang - params.torso_max_deg));
        d = pow(d, params.torso_max_deg / dang);
    }
    set_spine(&mut locals, d);
    let qc = d.mul(qh);

    // Neck and head: ears across, nose forward (a little below the ears).
    if let Some(head) = info.head {
        let ear_c = t
            .conf(lm::EAR_L)
            .min(t.conf(lm::EAR_R))
            .min(t.conf(lm::NOSE));
        let qhd = if ear_c > 0.3 {
            let (s20, c20) = detmath::sin_cos(20f64.to_radians());
            let fe0 = info.fwd.scale(c20) - UP.scale(s20);
            let let_ = t.at(lm::EAR_L) - t.at(lm::EAR_R);
            let fet = t.at(lm::NOSE) - t.mid(lm::EAR_L, lm::EAR_R);
            frame_rot(info.left, fe0, let_, fet)
        } else {
            qc
        };
        let mut e = qhd.mul(qc.conj());
        let eang = Quat::IDENTITY.angle_to(e).to_degrees();
        if eang > params.head_max_deg {
            capped.push(("head".into(), eang - params.head_max_deg));
            e = pow(e, params.head_max_deg / eang);
        }
        if let Some(neck) = info.neck {
            set_world_rot(
                rig,
                &mut locals,
                neck,
                pow(e, 0.5).mul(qc).mul(info.rest_rot(neck)),
            );
        }
        set_world_rot(rig, &mut locals, head, e.mul(qc).mul(info.rest_rot(head)));
    }
    for s in 0..2 {
        if let Some(sh) = info.shoulder[s] {
            set_world_rot(rig, &mut locals, sh, qc.mul(info.rest_rot(sh)));
        }
    }
    errors.push((
        "hip line".into(),
        angle_deg(
            rig.world(&locals)[info.thigh[0]].t - rig.world(&locals)[info.thigh[1]].t,
            lt,
        ),
    ));
    {
        let w = rig.world(&locals);
        errors.push((
            "shoulder line".into(),
            angle_deg(w[info.upper[0]].t - w[info.upper[1]].t, lst),
        ));
        let neck_head = info
            .neck
            .or(info.spine.last().copied())
            .unwrap_or(info.hips);
        errors.push((
            "torso".into(),
            angle_deg(
                (w[info.upper[0]].t + w[info.upper[1]].t).scale(0.5)
                    - (w[info.thigh[0]].t + w[info.thigh[1]].t).scale(0.5),
                torso,
            ),
        ));
        let _ = neck_head;
    }

    // Arms and legs.
    let arm = [
        (lm::SH_L, lm::EL_L, lm::WR_L),
        (lm::SH_R, lm::EL_R, lm::WR_R),
    ];
    let leg = [
        (lm::HIP_L, lm::KN_L, lm::AN_L),
        (lm::HIP_R, lm::KN_R, lm::AN_R),
    ];
    for s in 0..2 {
        let (a, b, c) = arm[s];
        let (eu, el, cl) = fit_limb(
            info,
            &mut locals,
            info.upper[s],
            info.fore[s],
            info.hand[s],
            t.at(b) - t.at(a),
            t.at(c) - t.at(b),
            params,
        );
        let _ = (eu, el);
        if cl.abs() > 0.0 {
            capped.push((format!("elbow.{}", SIDES[s]), cl.abs()));
        }
        let (a, b, c) = leg[s];
        let (eu, el, cl) = fit_limb(
            info,
            &mut locals,
            info.thigh[s],
            info.shin[s],
            info.foot[s],
            t.at(b) - t.at(a),
            t.at(c) - t.at(b),
            params,
        );
        let _ = (eu, el);
        if cl.abs() > 0.0 {
            capped.push((format!("knee.{}", SIDES[s]), cl.abs()));
        }
    }

    // Hands: toward the drawn knuckles (the sword hand: the blade).
    let knuckles = [
        (lm::WR_L, lm::IX_L, lm::PK_L),
        (lm::WR_R, lm::IX_R, lm::PK_R),
    ];
    for s in 0..2 {
        let (w_, ix, pk) = knuckles[s];
        if Some(s) == params.sword_hand && pose.blade.is_some() && info.socket[s].is_some() {
            continue;
        }
        if t.conf(ix).min(t.conf(pk)) < 0.3 {
            continue;
        }
        let ht = t.mid(ix, pk) - t.at(w_);
        let wh = world_rot(rig, &locals, info.hand[s]);
        let want = shortest_arc(wh.rotate_vec(info.hand_dir(s)), ht.normalized()).mul(wh);
        place_hand(info, &mut locals, s, want, params, &mut capped);
    }
    let mut blade_err = None;
    if let (Some(s), Some(b)) = (params.sword_hand, pose.blade) {
        blade_err = aim_sword(
            info,
            &mut locals,
            s,
            info.to_rig(b),
            edge,
            params,
            &mut capped,
        );
        if two_hands {
            // Both fists where the drawing has them, relative to the
            // shoulders and scaled by arm length (drawn arms are longer
            // than the rig's); the sword arm reaches there by IK, keeping
            // the drawn elbow's plane.
            let arm_len = |a: (usize, usize, usize)| {
                (t.at(a.1) - t.at(a.0)).length() + (t.at(a.2) - t.at(a.1)).length()
            };
            let drawn_arm = 0.5 * (arm_len(arm[0]) + arm_len(arm[1]));
            let w = rig.world(&locals);
            let rig_arm = 0.5
                * ((0..2)
                    .map(|q| {
                        (w[info.fore[q]].t - w[info.upper[q]].t).length()
                            + (w[info.hand[q]].t - w[info.fore[q]].t).length()
                    })
                    .sum::<f64>());
            let m_d = t.mid(lm::SH_L, lm::SH_R);
            let m_r = (w[info.upper[0]].t + w[info.upper[1]].t).scale(0.5);
            let g_d = t.mid(lm::WR_L, lm::WR_R);
            let g = m_r + (g_d - m_d).scale(rig_arm / drawn_arm.max(1e-6));
            if let Some(sk) = info.socket[s] {
                let bl = w[sk].rotation().rotate_vec(UP);
                let drawn = t.at(arm[1 - s].2) - t.at(arm[s].2);
                let gap = drawn.dot(bl).abs().clamp(0.08 * info.k, 0.14 * info.k);
                let sign = if drawn.dot(bl) >= 0.0 { 1.0 } else { -1.0 };
                let place = g - bl.scale(0.5 * gap * sign);
                let off = w[sk].t - w[info.hand[s]].t;
                solve_arm(
                    rig,
                    &mut locals,
                    info.upper[s],
                    info.fore[s],
                    info.hand[s],
                    place - off,
                );
                swivel(
                    info,
                    &mut locals,
                    info.upper[s],
                    info.fore[s],
                    info.hand[s],
                    t.at(arm[s].1) - t.at(arm[s].0),
                );
            }
        }
    }

    // Feet: flat on the floor when drawn there, else along the drawn sole.
    let mut planted = [false; 2];
    let soles = [
        (lm::HEEL_L, lm::FT_L, lm::AN_L),
        (lm::HEEL_R, lm::FT_R, lm::AN_R),
    ];
    for s in 0..2 {
        let (he, ft, an) = soles[s];
        let foot = info.foot[s];
        let rest_sole = match info.toe[s] {
            Some(toe) => {
                let d = info.rig.head(toe) - info.rig.head(foot);
                Vec3::new(d.x, 0.0, d.z).normalized()
            }
            None => info.fwd,
        };
        let sole_t = t.at(ft) - t.at(he);
        // Drawn on the floor: the lower of heel and toe near it (the
        // ankle landmark sits anywhere inside a boot).
        let low = t.at(he).y.min(t.at(ft).y) - info.ground;
        planted[s] = low < params.plant_height * info.k;
        let _ = an;
        if planted[s] {
            let flat = Vec3::new(sole_t.x, 0.0, sole_t.z);
            let yaw = if flat.length() > 1e-6 {
                signed_angle(rest_sole, flat, UP)
            } else {
                0.0
            };
            set_world_rot(
                rig,
                &mut locals,
                foot,
                Quat::from_axis_angle(UP, yaw).mul(info.rest_rot(foot)),
            );
        } else {
            let wf = world_rot(rig, &locals, foot);
            let sl = info.rest_rot(foot).conj().rotate_vec(rest_sole);
            set_world_rot(
                rig,
                &mut locals,
                foot,
                shortest_arc(wf.rotate_vec(sl), sole_t.normalized()).mul(wf),
            );
            let ex = cap(info, &mut locals, foot, info.shin[s], params.ankle_max_deg);
            if ex > 0.0 {
                capped.push((format!("ankle.{}", SIDES[s]), ex));
            }
        }
    }

    // Root: the hip joints over the drawn hips.
    {
        let w = rig.world(&locals);
        let mid = (w[info.thigh[0]].t + w[info.thigh[1]].t).scale(0.5);
        let delta = t.mid(lm::HIP_L, lm::HIP_R) - mid;
        move_root(info, &mut locals, delta);
    }
    // Planted feet: on the floor where drawn; the hips come down when a
    // leg cannot reach.
    // Each foot where the drawing has it relative to its hip, the reach
    // scaled by the rig's leg over this key's drawn leg (the two views
    // can stretch a lunge: one sheet's strike legs read 15% long).
    let ankle_targets: Vec<Vec3> = {
        let w = rig.world(&locals);
        (0..2)
            .map(|s| {
                let (h, kn, an) = leg[s];
                let drawn = (t.at(kn) - t.at(h)).length() + (t.at(an) - t.at(kn)).length();
                let rig_len = (w[info.shin[s]].t - w[info.thigh[s]].t).length()
                    + (w[info.foot[s]].t - w[info.shin[s]].t).length();
                let k = if drawn > 1e-6 {
                    (rig_len / drawn).min(1.0)
                } else {
                    1.0
                };
                let a = t.at(h) + (t.at(an) - t.at(h)).scale(k);
                Vec3::new(a.x, info.ground + info.ankle_h[s], a.z)
            })
            .collect()
    };
    let knees = [
        t.at(lm::KN_L) - t.at(lm::HIP_L),
        t.at(lm::KN_R) - t.at(lm::HIP_R),
    ];
    plant_feet(info, &mut locals, &planted, &ankle_targets, Some(knees));

    let mut grip_err = None;
    if two_hands {
        if let Some(s) = params.sword_hand {
            let drawn = t.at(arm[1 - s].2) - t.at(arm[s].2);
            let elbow = t.at(arm[1 - s].1) - t.at(arm[1 - s].0);
            grip_err = grip_off_hand(
                info,
                &mut locals,
                s,
                drawn,
                Some(elbow),
                params,
                &mut capped,
            );
        }
    }

    // How far each fitted bone points from the drawn one, after
    // everything (planting and the grip move arms and legs).
    {
        let w = rig.world(&locals);
        for s in 0..2 {
            let (a, b, c) = arm[s];
            errors.push((
                format!("upper_arm.{}", SIDES[s]),
                angle_deg(w[info.fore[s]].t - w[info.upper[s]].t, t.at(b) - t.at(a)),
            ));
            errors.push((
                format!("forearm.{}", SIDES[s]),
                angle_deg(w[info.hand[s]].t - w[info.fore[s]].t, t.at(c) - t.at(b)),
            ));
            let (a, b, c) = leg[s];
            errors.push((
                format!("thigh.{}", SIDES[s]),
                angle_deg(w[info.shin[s]].t - w[info.thigh[s]].t, t.at(b) - t.at(a)),
            ));
            errors.push((
                format!("shin.{}", SIDES[s]),
                angle_deg(w[info.foot[s]].t - w[info.shin[s]].t, t.at(c) - t.at(b)),
            ));
        }
    }
    let targets = (0..lm::COUNT).map(|i| t.at(i)).collect();
    let (off_rel, gap) = match params.sword_hand {
        Some(s) if two_hands => (
            off_hand_rel(info, &locals, s),
            grip_gap(info, &locals, s, t.at(arm[1 - s].2) - t.at(arm[s].2)),
        ),
        _ => (None, 0.0),
    };
    KeyFit {
        off_rel,
        grip_gap: gap,
        locals,
        planted,
        targets,
        errors,
        blade_err_deg: blade_err,
        grip_err_m: grip_err,
        capped,
        two_hands,
    }
}

/// Turn a solved limb about its root-to-end line so the middle joint
/// points along `dir` (from the root) as nearly as it can; the end and
/// the end bone's world rotation stay.
pub fn swivel(
    info: &RigInfo,
    locals: &mut [Option<Trs>],
    upper: usize,
    lower: usize,
    end: usize,
    dir: Vec3,
) {
    let rig = &info.rig;
    let w = rig.world(locals);
    let (a, b, c) = (w[upper].t, w[lower].t, w[end].t);
    let axis = c - a;
    if axis.length() < 1e-9 || dir.length() < 1e-9 {
        return;
    }
    let axis = axis.normalized();
    let ang = signed_angle(b - a, dir, axis);
    let keep = w[end].rotation();
    set_world_rot(
        rig,
        locals,
        upper,
        Quat::from_axis_angle(axis, ang).mul(w[upper].rotation()),
    );
    set_world_rot(rig, locals, end, keep);
}

/// Stand a fitted key on another key's feet (the stance): its feet's
/// midpoint over the stance's, the feet where the stance has them (same
/// turn), the legs bent to reach (the hips come down when they cannot).
/// The drawn upper body and the hips' place over the feet stay.
pub fn restance(info: &RigInfo, locals: &mut [Option<Trs>], stance: &[Option<Trs>]) {
    let rig = &info.rig;
    let ws = rig.world(stance);
    let w = rig.world(locals);
    let flat = |v: Vec3| Vec3::new(v.x, 0.0, v.z);
    let mid = |w: &[crate::glb::Affine]| (w[info.foot[0]].t + w[info.foot[1]].t).scale(0.5);
    move_root(info, locals, flat(mid(&ws) - mid(&w)));
    let marks = [ws[info.foot[0]].t, ws[info.foot[1]].t];
    for s in 0..2 {
        set_world_rot(rig, locals, info.foot[s], ws[info.foot[s]].rotation());
    }
    plant_feet(info, locals, &[true, true], &marks, None);
}

/// Move the hips (root) by a world offset.
pub fn move_root(info: &RigInfo, locals: &mut [Option<Trs>], delta: Vec3) {
    let rig = &info.rig;
    let w = rig.world(locals);
    let want = w[info.hips].t + delta;
    let pw = rig.parent[info.hips]
        .map(|p| w[p])
        .unwrap_or(crate::glb::Affine::IDENTITY);
    let inv = pw.inverse().unwrap_or(crate::glb::Affine::IDENTITY);
    let mut l = locals[info.hips].unwrap_or(rig.rest[info.hips]);
    l.t = inv.apply(want);
    locals[info.hips] = Some(l);
}

/// Put each planted ankle on its target (two-bone IK, the foot's world
/// rotation kept), first lowering the hips if a leg cannot reach.
pub fn plant_feet(
    info: &RigInfo,
    locals: &mut [Option<Trs>],
    planted: &[bool; 2],
    targets: &[Vec3],
    knees: Option<[Vec3; 2]>,
) {
    plant_feet_with(info, locals, planted, targets, knees, knees.is_some());
}

/// Knee directions over each foot's toes (a locked foot keeps its turn
/// and the knee follows it, so the ankle does not twist).
pub fn knees_over_toes(info: &RigInfo, locals: &[Option<Trs>]) -> [Vec3; 2] {
    let w = info.rig.world(locals);
    let mut out = [info.fwd; 2];
    for (s, o) in out.iter_mut().enumerate() {
        if let Some(toe) = info.toe[s] {
            let d = w[toe].t - w[info.foot[s]].t;
            let f = Vec3::new(d.x, 0.0, d.z);
            if f.length() > 1e-6 {
                *o = f.normalized();
            }
        }
    }
    out
}

/// [`plant_feet`]; `follow_shin`: the foot's turn follows the fitted shin
/// (when fitting a key) instead of the knee following the foot.
pub fn plant_feet_with(
    info: &RigInfo,
    locals: &mut [Option<Trs>],
    planted: &[bool; 2],
    targets: &[Vec3],
    knees: Option<[Vec3; 2]>,
    follow_shin: bool,
) {
    let drop = needed_drop(info, locals, planted, targets);
    if drop > 0.0 {
        move_root(info, locals, Vec3::new(0.0, -drop, 0.0));
    }
    let rig = &info.rig;
    for s in 0..2 {
        if planted[s] {
            solve_arm(
                rig,
                locals,
                info.thigh[s],
                info.shin[s],
                info.foot[s],
                targets[s],
            );
            if let Some(k) = knees {
                swivel(
                    info,
                    locals,
                    info.thigh[s],
                    info.shin[s],
                    info.foot[s],
                    k[s],
                );
            }
            if follow_shin {
                flat_from_shin(info, locals, s);
            }
            onto_toes(info, locals, s, knees.map(|k| k[s]));
        }
    }
}

/// A planted foot pointing where its shin does: the foot as it sits on
/// the shin at rest, turned flat (sole level). Used when fitting a key
/// (the drawn foot's own direction is the least reliable landmark);
/// between keys a locked foot keeps its key's turn.
fn flat_from_shin(info: &RigInfo, locals: &mut [Option<Trs>], s: usize) {
    let rig = &info.rig;
    let w = rig.world(locals);
    let (foot, shin) = (info.foot[s], info.shin[s]);
    let rest_rel = info.rest_rot(shin).conj().mul(info.rest_rot(foot));
    let on_shin = w[shin].rotation().mul(rest_rel);
    // Rest sole (heel to toe, level) in the foot's frame.
    let sole_rest = match info.toe[s] {
        Some(toe) => {
            let d = rig.head(toe) - rig.head(foot);
            Vec3::new(d.x, 0.0, d.z).normalized()
        }
        None => info.fwd,
    };
    let sole_local = info.rest_rot(foot).conj().rotate_vec(sole_rest);
    let sole = on_shin.rotate_vec(sole_local);
    let flat = Vec3::new(sole.x, 0.0, sole.z);
    if flat.length() < 1e-6 {
        return;
    }
    let yaw = signed_angle(sole_rest, flat, UP);
    set_world_rot(
        rig,
        locals,
        foot,
        Quat::from_axis_angle(UP, yaw).mul(info.rest_rot(foot)),
    );
}

/// Planted ankles stay this far (degrees) within `motion check`'s limit.
pub const PLANT_ANKLE_MAX_DEG: f64 = 60.0;

fn ankle_excess(info: &RigInfo, locals: &[Option<Trs>], s: usize) -> f64 {
    let w = info.rig.world(locals);
    let rel = w[info.shin[s]]
        .rotation()
        .conj()
        .mul(w[info.foot[s]].rotation());
    let rest = info
        .rest_rot(info.shin[s])
        .conj()
        .mul(info.rest_rot(info.foot[s]));
    rest.angle_to(rel).to_degrees() - PLANT_ANKLE_MAX_DEG
}

/// A planted foot whose ankle would pass its limit (the back leg of a
/// lunge) rises onto its toes: it turns about the toe, which stays on the
/// floor, just far enough (bisection), and the leg reaches the raised
/// ankle.
fn onto_toes(info: &RigInfo, locals: &mut [Option<Trs>], s: usize, knee: Option<Vec3>) {
    let Some(toe) = info.toe[s] else {
        return;
    };
    if ankle_excess(info, locals, s) <= 0.0 {
        return;
    }
    let rig = &info.rig;
    let w = rig.world(locals);
    let (toe_p, ank, wf) = (w[toe].t, w[info.foot[s]].t, w[info.foot[s]].rotation());
    let d = toe_p - ank;
    let sole = Vec3::new(d.x, 0.0, d.z);
    if sole.length() < 1e-6 {
        return;
    }
    let axis = UP.cross(sole.normalized());
    let start = locals.to_vec();
    let try_at = |ang: f64| {
        let q = Quat::from_axis_angle(axis, ang);
        let mut l = start.clone();
        set_world_rot(rig, &mut l, info.foot[s], q.mul(wf));
        let a = toe_p + q.rotate_vec(ank - toe_p);
        solve_arm(rig, &mut l, info.thigh[s], info.shin[s], info.foot[s], a);
        if let Some(k) = knee {
            swivel(info, &mut l, info.thigh[s], info.shin[s], info.foot[s], k);
        }
        l
    };
    // The heel goes up: the sign that raises the ankle.
    let up = Quat::from_axis_angle(axis, 0.1).rotate_vec(ank - toe_p).y > (ank - toe_p).y;
    let sign = if up { 1.0 } else { -1.0 };
    let (mut lo, mut hi) = (0.0, 70f64.to_radians());
    if ankle_excess(info, &try_at(sign * hi), s) > 0.0 {
        // Not a heel-up case (the foot is turned on the shin, not bent):
        // leave it.
        return;
    }
    for _ in 0..14 {
        let mid = 0.5 * (lo + hi);
        if ankle_excess(info, &try_at(sign * mid), s) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    locals.copy_from_slice(&try_at(sign * hi));
}

/// How far the hips must come down for every planted ankle to reach its
/// target.
pub fn needed_drop(
    info: &RigInfo,
    locals: &[Option<Trs>],
    planted: &[bool; 2],
    targets: &[Vec3],
) -> f64 {
    let rig = &info.rig;
    let w = rig.world(locals);
    let mut drop: f64 = 0.0;
    for s in 0..2 {
        if !planted[s] {
            continue;
        }
        let th = w[info.thigh[s]].t;
        let a = (w[info.shin[s]].t - th).length();
        let b = (w[info.foot[s]].t - w[info.shin[s]].t).length();
        let reach = 0.995 * (a + b);
        let tg = targets[s];
        let horiz = Vec3::new(tg.x - th.x, 0.0, tg.z - th.z).length();
        if horiz < reach {
            let ymax = tg.y + (reach * reach - horiz * horiz).sqrt();
            drop = drop.max(th.y - ymax);
        }
    }
    drop
}

/// Put the off hand on the grip next to the sword hand: its socket in
/// the sword socket's frame, `drawn`'s share of a hand's width along the
/// blade (the drawn wrist order says above or below). Returns the
/// distance left between the socket and its place.
pub fn grip_off_hand(
    info: &RigInfo,
    locals: &mut [Option<Trs>],
    s: usize,
    drawn: Vec3,
    elbow: Option<Vec3>,
    params: &FitParams,
    capped: &mut Vec<(String, f64)>,
) -> Option<f64> {
    let rig = &info.rig;
    let o = 1 - s;
    let (sk, so) = (info.socket[s]?, info.socket[o]?);
    let w = rig.world(locals);
    let r = w[sk].rotation();
    let b = r.rotate_vec(UP);
    let g = drawn.dot(b);
    let gap = g.abs().clamp(0.08 * info.k, 0.14 * info.k) * if g >= 0.0 { 1.0 } else { -1.0 };
    let place = w[sk].t + b.scale(gap);
    // The off fist wraps the grip like the sword hand (its socket in the
    // sword socket's frame), turned about the grip by whichever quarter
    // turn the wrist reaches best.
    let start = locals.to_vec();
    let mut best: Option<(f64, Vec<Option<Trs>>, Vec<(String, f64)>, f64)> = None;
    for quarter in 0..4 {
        let mut l = start.clone();
        let mut cap_here = Vec::new();
        let turn = Quat::from_axis_angle(b, quarter as f64 * std::f64::consts::FRAC_PI_2);
        let want_hand = turn.mul(r).mul(rig.rest[so].r.conj());
        let err = grip_try(
            info,
            &mut l,
            o,
            place,
            want_hand,
            elbow,
            params,
            &mut cap_here,
        );
        let excess: f64 = cap_here.iter().map(|c| c.1).sum();
        let score = excess + 100.0 * err / info.k;
        if best.as_ref().is_none_or(|bst| score < bst.0) {
            best = Some((score, l, cap_here, err));
        }
    }
    let (_, l, cap_here, err) = best?;
    locals.copy_from_slice(&l);
    capped.extend(cap_here);
    Some(err)
}

/// The off hand to `place` (its socket) with world rotation `want_hand`:
/// two-bone IK, the drawn elbow's swivel, the wrist capped; three rounds.
#[allow(clippy::too_many_arguments)]
fn grip_try(
    info: &RigInfo,
    l: &mut [Option<Trs>],
    o: usize,
    place: Vec3,
    want_hand: Quat,
    elbow: Option<Vec3>,
    params: &FitParams,
    cap_here: &mut Vec<(String, f64)>,
) -> f64 {
    let rig = &info.rig;
    let Some(so) = info.socket[o] else {
        return 0.0;
    };
    place_hand(info, l, o, want_hand, params, cap_here);
    for _ in 0..3 {
        let w = rig.world(l);
        let off = w[so].t - w[info.hand[o]].t;
        solve_arm(
            rig,
            l,
            info.upper[o],
            info.fore[o],
            info.hand[o],
            place - off,
        );
        if let Some(e) = elbow {
            swivel(info, l, info.upper[o], info.fore[o], info.hand[o], e);
        }
        cap_here.clear();
        place_hand(info, l, o, want_hand, params, cap_here);
    }
    let w = rig.world(l);
    (w[so].t - place).length()
}

/// The off hand's rotation relative to the sword socket (world), when
/// both sockets exist.
pub fn off_hand_rel(info: &RigInfo, locals: &[Option<Trs>], s: usize) -> Option<Quat> {
    let sk = info.socket[s]?;
    let w = info.rig.world(locals);
    Some(w[sk].rotation().conj().mul(w[info.hand[1 - s]].rotation()))
}

/// Per frame: the off hand on the grip `gap` along the blade from the
/// sword socket, turned `rel` from the sword socket (interpolated between
/// keys, so the fist does not flip between frames).
pub fn grip_with(
    info: &RigInfo,
    locals: &mut [Option<Trs>],
    s: usize,
    gap: f64,
    rel: Quat,
    params: &FitParams,
) -> Option<f64> {
    let sk = info.socket[s]?;
    let w = info.rig.world(locals);
    let r = w[sk].rotation();
    let place = w[sk].t + r.rotate_vec(UP).scale(gap);
    let mut cap = Vec::new();
    Some(grip_try(
        info,
        locals,
        1 - s,
        place,
        r.mul(rel),
        None,
        params,
        &mut cap,
    ))
}

/// Signed grip gap along the blade (off hand minus sword hand), clamped
/// to a hand's width, from the drawn wrists.
pub fn grip_gap(info: &RigInfo, locals: &[Option<Trs>], s: usize, drawn: Vec3) -> f64 {
    let Some(sk) = info.socket[s] else {
        return 0.0;
    };
    let b = info.rig.world(locals)[sk].rotation().rotate_vec(UP);
    let g = drawn.dot(b);
    g.abs().clamp(0.08 * info.k, 0.14 * info.k) * if g >= 0.0 { 1.0 } else { -1.0 }
}

/// The sword socket's blade tip in world (`len` m along +Y).
pub fn blade_tip(info: &RigInfo, locals: &[Option<Trs>], s: usize, len: f64) -> Option<Vec3> {
    let sk = info.socket[s]?;
    let w = info.rig.world(locals);
    Some(w[sk].t + w[sk].rotation().rotate_vec(UP).scale(len))
}

/// Fit every key: once plain, then again with each key's swing direction
/// (from the blade tips of its neighbours) so the edge leads.
pub fn fit_sheet(info: &RigInfo, poses: &[Pose3], params: &FitParams) -> (Vec<KeyFit>, f64, bool) {
    let scale = sheet_scale(info, poses);
    let two = two_handed(info, poses, scale, params);
    let first: Vec<KeyFit> = poses
        .iter()
        .map(|p| fit_key(info, p, scale, two, None, params))
        .collect();
    let Some(s) = params.sword_hand else {
        return (first, scale, two);
    };
    let tips: Vec<Option<Vec3>> = first
        .iter()
        .map(|f| blade_tip(info, &f.locals, s, 0.9 * info.k))
        .collect();
    let n = poses.len();
    let fits = (0..n)
        .map(|k| {
            let prev = if k > 0 { tips[k - 1] } else { None };
            let next = if k + 1 < n { tips[k + 1] } else { None };
            let edge = match (prev, tips[k], next) {
                (_, Some(c), Some(nx)) if k == 0 => Some(nx - c),
                (Some(pv), _, Some(nx)) => Some(nx - pv),
                (Some(pv), Some(c), None) => Some(c - pv),
                _ => None,
            };
            fit_key(info, &poses[k], scale, two, edge, params)
        })
        .collect();
    (fits, scale, two)
}
