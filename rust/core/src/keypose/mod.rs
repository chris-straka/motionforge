//! Key-pose drawings -> a timed attack clip on a rig (`motionforge keypose`).
//!
//! 1. **Read** (`python/keypose/landmarks.py`, MediaPipe Pose Landmarker):
//!    per drawn figure, 33 image landmarks, MediaPipe's own 3D guess
//!    ("world" landmarks) and the blade as a 2D segment.
//! 2. **Fuse** ([`fuse`]): one 3D point set per key pose. A front view
//!    is exact across (left-right) and up, a side view exact front-back
//!    and up; each view's depth is MediaPipe's guess, trusted
//!    [`FuseParams::depth_weight`] as much. Two views: weighted least
//!    squares per point, the side view's scale fitted to the front's
//!    heights and its turn (a "front" drawn at three-quarters) fitted to
//!    the front's points. One view: its own depth guess.
//! 3. **Fit** (`fit.rs`): bone by bone onto the rig: hips and chest from
//!    the hip and shoulder lines, the spine bent evenly between them,
//!    neck and head from ears and nose, arms and legs as hinges (elbows
//!    and knees bend only in their anatomical plane, within the limits
//!    `motion check` scores), feet flat on the floor when drawn there,
//!    the sword hand turned so the socket's blade axis follows the drawn
//!    blade, the other hand on the grip.
//! 4. **Time** (`timing.rs`): animator spacing between the keys.

pub mod fit;
pub mod timing;

use crate::json::Json;
use crate::math::Vec3;

/// MediaPipe Pose landmark indices.
pub mod lm {
    pub const NOSE: usize = 0;
    pub const EAR_L: usize = 7;
    pub const EAR_R: usize = 8;
    pub const SH_L: usize = 11;
    pub const SH_R: usize = 12;
    pub const EL_L: usize = 13;
    pub const EL_R: usize = 14;
    pub const WR_L: usize = 15;
    pub const WR_R: usize = 16;
    pub const PK_L: usize = 17;
    pub const PK_R: usize = 18;
    pub const IX_L: usize = 19;
    pub const IX_R: usize = 20;
    pub const HIP_L: usize = 23;
    pub const HIP_R: usize = 24;
    pub const KN_L: usize = 25;
    pub const KN_R: usize = 26;
    pub const AN_L: usize = 27;
    pub const AN_R: usize = 28;
    pub const HEEL_L: usize = 29;
    pub const HEEL_R: usize = 30;
    pub const FT_L: usize = 31;
    pub const FT_R: usize = 32;
    pub const COUNT: usize = 33;
}

/// One drawn figure in one view.
#[derive(Clone, Debug)]
pub struct Figure {
    /// `(u, v, visibility)`: pixels, v down.
    pub img: Vec<[f64; 3]>,
    /// MediaPipe's world landmarks (metres, hip-centred, x right, y
    /// down, z away from the camera).
    pub world: Vec<Vec3>,
    /// The blade, `(hilt, tip)` in pixels.
    pub blade: Option<([f64; 2], [f64; 2])>,
}

/// One picture: figures left to right (in time order), the drawn floor.
#[derive(Clone, Debug)]
pub struct ViewSheet {
    pub figures: Vec<Figure>,
    pub floor_v: Option<f64>,
    pub file: String,
}

#[derive(Clone, Debug)]
pub struct Landmarks {
    pub name: String,
    pub front: Option<ViewSheet>,
    pub side: Option<ViewSheet>,
}

impl Landmarks {
    pub fn keys(&self) -> usize {
        self.front
            .as_ref()
            .or(self.side.as_ref())
            .map(|s| s.figures.len())
            .unwrap_or(0)
    }
}

fn num(j: &Json) -> Result<f64, String> {
    j.as_f64().ok_or_else(|| "expected a number".to_string())
}

fn parse_view(j: &Json) -> Result<ViewSheet, String> {
    let mut figures = Vec::new();
    for f in j.get("figures").and_then(Json::as_arr).unwrap_or(&[]) {
        let img_rows = f
            .get("image")
            .and_then(Json::as_arr)
            .ok_or("figure without image landmarks")?;
        let world_rows = f
            .get("world")
            .and_then(Json::as_arr)
            .ok_or("figure without world landmarks")?;
        if img_rows.len() != lm::COUNT || world_rows.len() != lm::COUNT {
            return Err(format!("expected {} landmarks per figure", lm::COUNT));
        }
        let mut img = Vec::new();
        for r in img_rows {
            let a = r.as_arr().ok_or("bad image landmark")?;
            if a.len() < 3 {
                return Err("image landmark needs u, v, visibility".into());
            }
            img.push([num(&a[0])?, num(&a[1])?, num(&a[2])?]);
        }
        let mut world = Vec::new();
        for r in world_rows {
            let a = r.as_arr().ok_or("bad world landmark")?;
            if a.len() < 3 {
                return Err("world landmark needs x, y, z".into());
            }
            world.push(Vec3::new(num(&a[0])?, num(&a[1])?, num(&a[2])?));
        }
        let blade = match f.get("blade") {
            Some(b) if !b.is_null() => {
                let p = |k: &str| -> Result<[f64; 2], String> {
                    let a = b.get(k).and_then(Json::as_arr).ok_or("bad blade")?;
                    if a.len() < 2 {
                        return Err("bad blade".into());
                    }
                    Ok([num(&a[0])?, num(&a[1])?])
                };
                Some((p("hilt")?, p("tip")?))
            }
            _ => None,
        };
        figures.push(Figure { img, world, blade });
    }
    Ok(ViewSheet {
        figures,
        floor_v: j.get("floor_y").and_then(Json::as_f64),
        file: j
            .get("file")
            .and_then(Json::as_str)
            .unwrap_or("")
            .to_string(),
    })
}

/// Parse `landmarks.py` output (`motionforge-keypose-landmarks` v1).
pub fn parse_landmarks(text: &str) -> Result<Landmarks, String> {
    let j = crate::json::parse(text)?;
    if j.get("format").and_then(Json::as_str) != Some("motionforge-keypose-landmarks") {
        return Err("not a motionforge-keypose-landmarks file".into());
    }
    let views = j.get("views").ok_or("no views")?;
    let front = views.get("front").map(parse_view).transpose()?;
    let side = views.get("side").map(parse_view).transpose()?;
    let out = Landmarks {
        name: j
            .get("name")
            .and_then(Json::as_str)
            .unwrap_or("keypose")
            .to_string(),
        front,
        side,
    };
    if let (Some(f), Some(s)) = (&out.front, &out.side) {
        if f.figures.len() != s.figures.len() {
            return Err(format!(
                "front has {} figures, side {}",
                f.figures.len(),
                s.figures.len()
            ));
        }
    }
    if out.keys() == 0 {
        return Err("no figures".into());
    }
    Ok(out)
}

#[derive(Clone, Debug)]
pub struct FuseParams {
    /// How much a view's depth guess counts against its exact axes.
    pub depth_weight: f64,
    /// Largest turn (degrees) fitted between the front and side views.
    pub max_side_turn_deg: f64,
    /// A fixed turn instead of fitting one.
    pub side_turn_deg: Option<f64>,
}

impl Default for FuseParams {
    fn default() -> Self {
        FuseParams {
            depth_weight: 0.2,
            max_side_turn_deg: 40.0,
            // The drawings are made as a front and a strict profile;
            // fitting a turn made the fits worse (docs/keypose.md).
            side_turn_deg: Some(0.0),
        }
    }
}

/// One key pose in 3D, in the character's frame: X to his left, Y up
/// from the drawn floor, Z forward (toward the front view's camera).
/// Units: metres as the drawing's own scale gives them (the fit rescales
/// to the rig).
#[derive(Clone, Debug)]
pub struct Pose3 {
    pub p: Vec<Vec3>,
    /// Confidence per landmark (0..1).
    pub conf: Vec<f64>,
    /// Blade direction (hilt to tip), unit, when read.
    pub blade: Option<Vec3>,
    /// How the blade's direction was found ("two views", "front only", ...).
    pub blade_how: String,
    pub views: usize,
}

/// A view's mapping between the character frame and its pixels.
#[derive(Clone, Copy, Debug)]
pub struct ViewMap {
    /// Image-right axis in the character frame.
    pub h: Vec3,
    /// Toward-the-camera axis in the character frame.
    pub d: Vec3,
    /// Pixels per drawing metre.
    pub s: f64,
    /// Pixel column of the hip midpoint.
    pub u0: f64,
    /// Pixel row of the floor.
    pub v0: f64,
}

impl ViewMap {
    pub fn to_px(&self, p: Vec3) -> (f64, f64) {
        (self.u0 + self.s * p.dot(self.h), self.v0 - self.s * p.y)
    }
}

/// Fused key poses and the per-figure view maps.
#[derive(Clone, Debug)]
pub struct Fused {
    pub poses: Vec<Pose3>,
    /// Per key: (front map, side map).
    pub maps: Vec<(Option<ViewMap>, Option<ViewMap>)>,
    /// Side view's fitted turn about the vertical, degrees, per key.
    pub side_turn_deg: Vec<f64>,
    /// Per key: chains whose left and right the side view had swapped.
    pub side_swapped: Vec<Vec<String>>,
    /// Per key: the side view's image landmarks as used (after swaps).
    pub side_img: Vec<Option<Vec<[f64; 3]>>>,
}

const BODY: std::ops::Range<usize> = 11..33;

fn rot_y(v: Vec3, deg: f64) -> Vec3 {
    let (s, c) = crate::detmath::sin_cos(deg.to_radians());
    Vec3::new(v.x * c + v.z * s, v.y, -v.x * s + v.z * c)
}

/// A view's own 3D guess: exact in the picture's plane, MediaPipe's depth
/// along the camera.
fn mono(fig: &Figure, map: &ViewMap) -> Vec<Vec3> {
    (0..lm::COUNT)
        .map(|i| {
            let [u, v, _] = fig.img[i];
            let across = (u - map.u0) / map.s;
            let up = (map.v0 - v) / map.s;
            // MediaPipe's z grows away from the camera.
            let depth = -fig.world[i].z;
            map.h.scale(across) + Vec3::new(0.0, up, 0.0) + map.d.scale(depth)
        })
        .collect()
}

/// Pixels per metre: least squares of the picture's landmark spread on
/// MediaPipe's world spread (both x right, y down).
fn px_per_m(fig: &Figure) -> f64 {
    let (mut mu, mut mw, mut n) = ([0.0; 2], [0.0; 2], 0.0);
    for i in BODY {
        mu[0] += fig.img[i][0];
        mu[1] += fig.img[i][1];
        mw[0] += fig.world[i].x;
        mw[1] += fig.world[i].y;
        n += 1.0;
    }
    let (mut num, mut den) = (0.0, 0.0);
    for i in BODY {
        let a = [fig.img[i][0] - mu[0] / n, fig.img[i][1] - mu[1] / n];
        let b = [fig.world[i].x - mw[0] / n, fig.world[i].y - mw[1] / n];
        num += a[0] * b[0] + a[1] * b[1];
        den += b[0] * b[0] + b[1] * b[1];
    }
    if den > 1e-12 && num > 0.0 {
        num / den
    } else {
        1.0
    }
}

fn floor_of(sheet: &ViewSheet, fig: &Figure) -> f64 {
    sheet.floor_v.unwrap_or_else(|| {
        [lm::HEEL_L, lm::HEEL_R, lm::FT_L, lm::FT_R]
            .iter()
            .map(|&i| fig.img[i][1])
            .fold(f64::NEG_INFINITY, f64::max)
    })
}

fn hip_u(fig: &Figure) -> f64 {
    0.5 * (fig.img[lm::HIP_L][0] + fig.img[lm::HIP_R][0])
}

fn vis(fig: &Figure, i: usize) -> f64 {
    fig.img[i][2].clamp(0.05, 1.0)
}

/// Solve the symmetric 3x3 system `m x = b` (Cramer); None when singular.
fn solve3(m: [[f64; 3]; 3], b: Vec3) -> Option<Vec3> {
    let det = |a: [[f64; 3]; 3]| {
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    };
    let d = det(m);
    if d.abs() < 1e-14 {
        return None;
    }
    let bb = [b.x, b.y, b.z];
    let mut out = [0.0; 3];
    for (c, o) in out.iter_mut().enumerate() {
        let mut a = m;
        for r in 0..3 {
            a[r][c] = bb[r];
        }
        *o = det(a) / d;
    }
    Some(Vec3::new(out[0], out[1], out[2]))
}

fn add_outer(m: &mut [[f64; 3]; 3], v: Vec3, w: f64) {
    let a = [v.x, v.y, v.z];
    for r in 0..3 {
        for c in 0..3 {
            m[r][c] += w * a[r] * a[c];
        }
    }
}

fn mat_vec(m: &[[f64; 3]; 3], v: Vec3) -> Vec3 {
    Vec3::new(
        m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
        m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
        m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z,
    )
}

/// The blade's 2D direction in a view as a 3D direction in the picture's
/// plane (v down in pixels is -Y).
fn blade_in_plane(fig: &Figure, map: &ViewMap) -> Option<Vec3> {
    let (h, t) = fig.blade?;
    let (du, dv) = (t[0] - h[0], t[1] - h[1]);
    let e = map.h.scale(du) + Vec3::new(0.0, -dv, 0.0);
    (e.length() > 1e-6).then(|| e.normalized())
}

/// Fuse front and side views into one 3D pose per key.
pub fn fuse(lms: &Landmarks, params: &FuseParams) -> Fused {
    let n = lms.keys();
    let mut poses = Vec::new();
    let mut maps = Vec::new();
    let mut turns = Vec::new();
    let mut swaps = Vec::new();
    let mut side_imgs = Vec::new();
    for k in 0..n {
        let ff = lms.front.as_ref().map(|s| (s, &s.figures[k]));
        let mut side_fig: Option<Figure> = lms.side.as_ref().map(|s| s.figures[k].clone());
        let mut fmap = ff.map(|(s, f)| ViewMap {
            h: Vec3::new(1.0, 0.0, 0.0),
            d: Vec3::new(0.0, 0.0, 1.0),
            s: px_per_m(f),
            u0: hip_u(f),
            v0: floor_of(s, f),
        });
        let mut smap = lms
            .side
            .as_ref()
            .zip(side_fig.as_ref())
            .map(|(s, f)| ViewMap {
                h: Vec3::new(0.0, 0.0, -1.0),
                d: Vec3::new(1.0, 0.0, 0.0),
                s: px_per_m(f),
                u0: hip_u(f),
                v0: floor_of(s, f),
            });
        let mut turn = 0.0;
        let mut swapped = Vec::new();
        if let (Some((_, f)), Some(s), Some(fm), Some(sm)) =
            (ff, side_fig.as_mut(), fmap.as_mut(), smap.as_mut())
        {
            // The side's scale: its heights onto the front's.
            let (mut num, mut den) = (0.0, 0.0);
            for i in BODY {
                let w = vis(f, i) * vis(s, i);
                let yf = (fm.v0 - f.img[i][1]) / fm.s;
                let ys = (sm.v0 - s.img[i][1]) / sm.s;
                num += w * yf * ys;
                den += w * ys * ys;
            }
            if den > 1e-12 && num > 0.0 {
                sm.s /= num / den;
            }
            // A profile hides which arm or leg is which: MediaPipe often
            // names the far one as the near one. Per chain, keep the side
            // view's naming or swap it, whichever agrees with the front
            // (heights exactly, horizontal by both views' guesses at half
            // weight).
            let pf = mono(f, fm);
            for (chain, pairs) in [
                (
                    "arms",
                    &[(11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22)][..],
                ),
                (
                    "legs",
                    &[(23, 24), (25, 26), (27, 28), (29, 30), (31, 32)][..],
                ),
            ] {
                let ps = mono(s, sm);
                let cost = |swap: bool| {
                    let mut e = 0.0;
                    for &(a, b) in pairs {
                        for (i, j) in [(a, b), (b, a)] {
                            let js = if swap { j } else { i };
                            let d = pf[i] - ps[js];
                            let w = vis(f, i) * vis(s, js);
                            e += w * (d.y * d.y + 0.5 * (d.x * d.x + d.z * d.z));
                        }
                    }
                    e
                };
                if cost(true) < cost(false) {
                    for &(a, b) in pairs {
                        s.img.swap(a, b);
                        s.world.swap(a, b);
                    }
                    swapped.push(chain.to_string());
                }
            }
            // The side's turn: the angle that best lays its 3D guess on
            // the front's (horizontal plane, hip-centred), searched in
            // half-degree steps.
            let base = *sm;
            let mut best = (f64::INFINITY, 0.0);
            let steps = (params.max_side_turn_deg * 2.0).round() as i64;
            for st in -steps..=steps {
                let deg = st as f64 * 0.5;
                let m = ViewMap {
                    h: rot_y(base.h, deg),
                    d: rot_y(base.d, deg),
                    ..base
                };
                let ps = mono(s, &m);
                let mut e = 0.0;
                for i in BODY {
                    let w = vis(f, i) * vis(s, i);
                    let a = pf[i] - ps[i];
                    e += w * (a.x * a.x + a.z * a.z);
                }
                // A light pull toward the drawn convention (a strict
                // profile) so noise alone does not turn it.
                e += 1e-4 * deg * deg;
                if e < best.0 {
                    best = (e, deg);
                }
            }
            turn = params.side_turn_deg.unwrap_or(best.1);
            sm.h = rot_y(base.h, turn);
            sm.d = rot_y(base.d, turn);
        }
        let views: Vec<(&Figure, ViewMap)> = ff
            .map(|(_, f)| (f, fmap.unwrap()))
            .into_iter()
            .chain(side_fig.as_ref().map(|f| (f, smap.unwrap())))
            .collect();
        let monos: Vec<Vec<Vec3>> = views.iter().map(|(f, m)| mono(f, m)).collect();
        let wd = params.depth_weight * params.depth_weight;
        let mut p = Vec::with_capacity(lm::COUNT);
        let mut conf = Vec::with_capacity(lm::COUNT);
        for i in 0..lm::COUNT {
            let mut m = [[0.0; 3]; 3];
            let mut b = Vec3::ZERO;
            let mut c = 0.0;
            for ((f, map), q) in views.iter().zip(&monos) {
                let w = vis(f, i);
                let mut mv = [[0.0; 3]; 3];
                add_outer(&mut mv, map.h, w);
                add_outer(&mut mv, Vec3::new(0.0, 1.0, 0.0), w);
                add_outer(&mut mv, map.d, w * wd);
                b = b + mat_vec(&mv, q[i]);
                for r in 0..3 {
                    for cc in 0..3 {
                        m[r][cc] += mv[r][cc];
                    }
                }
                c += w;
            }
            p.push(solve3(m, b).unwrap_or(monos[0][i]));
            conf.push((c / views.len() as f64).min(1.0));
        }
        // The blade: each view's line and its camera ray span a plane; two
        // planes meet in the blade's 3D direction.
        let planes: Vec<(Vec3, Vec3)> = views
            .iter()
            .filter_map(|(f, m)| blade_in_plane(f, m).map(|e| (m.d.cross(e), e)))
            .collect();
        let (blade, blade_how) = match planes.as_slice() {
            [(n1, e1), (n2, e2)] => {
                let d = n1.cross(*n2);
                if d.length() > 0.2 * n1.length() * n2.length() {
                    let d = d.normalized();
                    let d = if d.dot(*e1) + d.dot(*e2) < 0.0 {
                        d.scale(-1.0)
                    } else {
                        d
                    };
                    (Some(d), "two views".to_string())
                } else {
                    (Some(*e1), "two views, parallel: front line".to_string())
                }
            }
            [(_, e)] => (Some(*e), "one view, in its plane".to_string()),
            _ => (None, "not read".to_string()),
        };
        poses.push(Pose3 {
            p,
            conf,
            blade,
            blade_how,
            views: views.len(),
        });
        maps.push((fmap, smap));
        turns.push(turn);
        swaps.push(swapped);
        side_imgs.push(side_fig.as_ref().map(|f| f.img.clone()));
    }
    Fused {
        poses,
        maps,
        side_turn_deg: turns,
        side_swapped: swaps,
        side_img: side_imgs,
    }
}

/// Give every key the reference key's leg order (which foot is in
/// front). A profile drawing often has the legs named the other way
/// round, and a drawn sheet whose feet swap front and back between keys
/// would make the feet cross; the side view's front-back reading of each
/// leg pair is exchanged (the front view's across reading stays). Returns
/// the keys changed.
pub fn order_legs(fused: &mut Fused, reference: usize) -> Vec<usize> {
    let order = |p: &Pose3| (p.p[lm::AN_L].z - p.p[lm::AN_R].z) >= 0.0;
    let want = order(&fused.poses[reference]);
    let mut changed = Vec::new();
    for (k, pose) in fused.poses.iter_mut().enumerate() {
        if k == reference || order(pose) == want {
            continue;
        }
        for (a, b) in [(23, 24), (25, 26), (27, 28), (29, 30), (31, 32)] {
            let (za, zb) = (pose.p[a].z, pose.p[b].z);
            pose.p[a].z = zb;
            pose.p[b].z = za;
        }
        changed.push(k);
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve3_inverts() {
        let m = [[4.0, 1.0, 0.0], [1.0, 3.0, 0.5], [0.0, 0.5, 2.0]];
        let x = Vec3::new(0.3, -1.2, 2.0);
        let b = mat_vec(&m, x);
        assert!(solve3(m, b).unwrap().approx_eq(x, 1e-12));
    }
}

#[cfg(test)]
mod known_answer;
