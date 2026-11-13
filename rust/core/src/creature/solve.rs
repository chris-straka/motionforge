//! From two-view detections to a clip on the creature's skeleton:
//! consistency check (both halves show the same move), triangulation,
//! temporal smoothing, a top-down rotation fit that keeps bone lengths
//! and joint limits, quaternion smoothing, and planted feet that don't
//! slide (CCD on the leg chain within limits).

use super::align::{prepare, AlignReport, TimeMode};
use super::camera::{closest_points, gap_ratio, View};
use super::detect::Detections;
use super::profile::Creature;
use crate::clip::{fk, tail_world, Clip, FkBone, Frame, Pose};
use crate::detmath;
use crate::json::Json;
use crate::math::{shortest_arc, Mat3, Quat, Vec3};

#[derive(Clone, Debug)]
pub struct SolveParams {
    /// Gaussian sigma (frames) for the triangulated positions.
    pub smooth_sigma: f64,
    /// Gaussian sigma (frames) for the solved rotations.
    pub quat_sigma: f64,
    /// Plant and pin the feet.
    pub pin: bool,
    /// Reject the clip when the median ray gap exceeds this fraction of
    /// the creature's height.
    pub max_gap_ratio: f64,
    /// Reject when the views' height disagreement moves more than this
    /// fraction of the joints' own height motion (see `motion_ratio`).
    pub max_motion_ratio: f64,
    /// Detections below this confidence are treated as missing.
    pub min_conf: f64,
    /// Solve even when the consistency check fails.
    pub force: bool,
    /// Re-fit each view's zoom and shift on frame `align_frame` (the
    /// rest pose of the start picture) before triangulating.
    pub align_views: bool,
    pub align_frame: usize,
    /// Slide the second view in time against the first.
    pub time_mode: TimeMode,
    /// Largest slide tried, in frames (also capped at a quarter of the clip).
    pub max_lag: usize,
}

impl Default for SolveParams {
    fn default() -> SolveParams {
        SolveParams {
            smooth_sigma: 1.0,
            quat_sigma: 0.8,
            pin: true,
            max_gap_ratio: 0.08,
            max_motion_ratio: 0.5,
            min_conf: 0.2,
            force: false,
            align_views: false,
            align_frame: 0,
            time_mode: TimeMode::Off,
            max_lag: 12,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SolveReport {
    pub frames: usize,
    pub joints_seen: usize,
    pub joints_total: usize,
    /// Median ray gap as a fraction of the height, over every joint with
    /// two detections.
    pub median_gap_ratio: f64,
    /// Variance of the signed height disagreement between the views over
    /// the variance of the joints' height motion, pooled over joints:
    /// near 0 for one move seen twice, around 4 for two different moves.
    pub motion_ratio: f64,
    pub consistent: bool,
    /// RMS distance (mm) between triangulated joints and the solved
    /// skeleton's joints (the fit residual).
    pub residual_mm: f64,
    /// Bone-frames clamped to a limit.
    pub limited: usize,
    pub plants: usize,
    pub max_pin_shift_mm: f64,
    /// What lining up the views did (absent when not asked for).
    pub align: Option<AlignReport>,
}

impl SolveReport {
    pub fn to_json(&self) -> Json {
        let r = |v: f64, p: i32| (v * 10f64.powi(p)).round() / 10f64.powi(p);
        let mut out = Json::obj(vec![
            ("frames", Json::num(self.frames as f64)),
            ("joints_seen", Json::num(self.joints_seen as f64)),
            ("joints_total", Json::num(self.joints_total as f64)),
            ("median_gap_ratio", Json::num(r(self.median_gap_ratio, 4))),
            ("motion_ratio", Json::num(r(self.motion_ratio, 4))),
            ("consistent", Json::Bool(self.consistent)),
            ("residual_mm", Json::num(r(self.residual_mm, 2))),
            ("limited", Json::num(self.limited as f64)),
            ("plants", Json::num(self.plants as f64)),
            ("max_pin_shift_mm", Json::num(r(self.max_pin_shift_mm, 2))),
        ]);
        if let Some(a) = &self.align {
            out.set("align", a.to_json());
        }
        out
    }
}

/// A triangulated joint: position and confidence (0 = missing).
#[derive(Clone, Copy, Debug)]
struct Point {
    p: Vec3,
    conf: f64,
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let m = v.len() / 2;
    if v.len() % 2 == 1 {
        v[m]
    } else {
        0.5 * (v[m - 1] + v[m])
    }
}

/// Triangulate every (frame, joint) from the first two views found in
/// both the detections and the creature. Returns points[frame][bone]
/// and the consistency numbers.
fn triangulate_all(
    c: &Creature,
    d: &Detections,
    params: &SolveParams,
) -> Result<(Vec<Vec<Point>>, f64, f64, usize), String> {
    let mut view_idx: Vec<(usize, &View)> = Vec::new();
    for (vi, name) in d.views.iter().enumerate() {
        if let Some(v) = c.view(name) {
            view_idx.push((vi, v));
        }
    }
    if view_idx.len() < 2 {
        return Err(format!(
            "detections need two of the creature's views ({}); found {:?}",
            c.views
                .iter()
                .map(|v| v.name.as_str())
                .collect::<Vec<_>>()
                .join(", "),
            d.views
        ));
    }
    let (ai, av) = view_idx[0];
    let (bi, bv) = view_idx[1];
    // Creature bone -> detection joint column.
    let cols: Vec<Option<usize>> = c
        .skeleton
        .bones
        .iter()
        .map(|b| d.joints.iter().position(|j| *j == b.name))
        .collect();
    let seen = cols.iter().filter(|c| c.is_some()).count();
    if cols[c.root()].is_none() {
        return Err(format!(
            "detections lack the root bone \"{}\"",
            c.skeleton.bones[c.root()].name
        ));
    }
    let n = c.skeleton.len();
    let mut points = Vec::with_capacity(d.frames.len());
    let mut gaps_all = Vec::new();
    // Per joint: (signed height disagreement, height) per frame.
    let mut tracks: Vec<Vec<(f64, f64)>> = vec![Vec::new(); n];
    for frame in &d.frames {
        let mut row = Vec::with_capacity(n);
        for (bi_, col) in cols.iter().enumerate() {
            let Some(j) = col else {
                row.push(Point {
                    p: Vec3::ZERO,
                    conf: 0.0,
                });
                continue;
            };
            let da = &frame[ai][*j];
            let db = &frame[bi][*j];
            if da.conf < params.min_conf || db.conf < params.min_conf {
                row.push(Point {
                    p: Vec3::ZERO,
                    conf: 0.0,
                });
                continue;
            }
            let (pa, pb) = closest_points(av.ray(da.u, da.v), bv.ray(db.u, db.v));
            let p = (pa + pb).scale(0.5);
            let gap = (pa - pb).length();
            let g = gap_ratio(gap, c.height);
            gaps_all.push(g);
            tracks[bi_].push((pa.z - pb.z, p.z));
            // Confidence shrinks with the gap: a joint the views disagree on
            // counts less in the fit.
            let agree = (1.0 - g / params.max_gap_ratio).clamp(0.05, 1.0);
            row.push(Point {
                p,
                conf: da.conf.min(db.conf) * agree,
            });
        }
        points.push(row);
    }
    let med = median(&mut gaps_all);
    // Pooled variance ratio: disagreement motion over height motion.
    let mut var_gap = 0.0;
    let mut var_z = 0.0;
    for t in &tracks {
        if t.len() < 2 {
            continue;
        }
        let m = t.len() as f64;
        let mean_g = t.iter().map(|x| x.0).sum::<f64>() / m;
        let mean_z = t.iter().map(|x| x.1).sum::<f64>() / m;
        var_gap += t
            .iter()
            .map(|x| (x.0 - mean_g) * (x.0 - mean_g))
            .sum::<f64>();
        var_z += t
            .iter()
            .map(|x| (x.1 - mean_z) * (x.1 - mean_z))
            .sum::<f64>();
    }
    let frames = d.frames.len().max(1) as f64;
    let still = (0.005 * c.height) * (0.005 * c.height) * frames * n as f64;
    let ratio = if var_z <= still { 0.0 } else { var_gap / var_z };
    Ok((points, med, ratio, seen))
}

/// The consistency numbers alone: (median ray gap over the height,
/// height-disagreement motion ratio). What the time and camera
/// alignment (`align`) minimise and what the solve's check reads.
pub fn consistency(
    c: &Creature,
    d: &Detections,
    params: &SolveParams,
) -> Result<(f64, f64), String> {
    let (_, med, ratio, _) = triangulate_all(c, d, params)?;
    Ok((med, ratio))
}

/// Fill missing points from the nearest frame in time, then Gaussian
/// smooth with confidence weights.
fn smooth_points(points: &mut [Vec<Point>], sigma: f64) {
    let frames = points.len();
    if frames == 0 {
        return;
    }
    let n = points[0].len();
    for j in 0..n {
        // Fill.
        let valid: Vec<usize> = (0..frames).filter(|&f| points[f][j].conf > 0.0).collect();
        if valid.is_empty() {
            continue;
        }
        for f in 0..frames {
            if points[f][j].conf > 0.0 {
                continue;
            }
            let nearest = *valid
                .iter()
                .min_by_key(|&&v| ((v as i64 - f as i64).abs(), v))
                .unwrap();
            points[f][j] = Point {
                p: points[nearest][j].p,
                conf: 0.2 * points[nearest][j].conf,
            };
        }
        if sigma <= 0.0 {
            continue;
        }
        // Smooth.
        let radius = (3.0 * sigma).ceil() as i64;
        let orig: Vec<Point> = (0..frames).map(|f| points[f][j]).collect();
        for f in 0..frames {
            let mut acc = Vec3::ZERO;
            let mut wsum = 0.0;
            for k in -radius..=radius {
                let g = f as i64 + k;
                if g < 0 || g >= frames as i64 {
                    continue;
                }
                let o = orig[g as usize];
                let w = o.conf * detmath::exp(-(k * k) as f64 / (2.0 * sigma * sigma));
                acc = acc + o.p.scale(w);
                wsum += w;
            }
            if wsum > 0.0 {
                points[f][j].p = acc.scale(1.0 / wsum);
            }
        }
    }
}

/// Largest-eigenvalue eigenvector of a symmetric 4x4 (cyclic Jacobi),
/// signed to agree with `prefer`.
fn top_eigenvector(mut a: [[f64; 4]; 4], prefer: [f64; 4]) -> [f64; 4] {
    let mut v = [[0.0; 4]; 4];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _sweep in 0..60 {
        let mut off = 0.0;
        for p in 0..4 {
            for q in (p + 1)..4 {
                off += a[p][q] * a[p][q];
            }
        }
        if off < 1e-30 {
            break;
        }
        for p in 0..4 {
            for q in (p + 1)..4 {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let cs = 1.0 / (t * t + 1.0).sqrt();
                let sn = t * cs;
                for k in 0..4 {
                    let akp = a[k][p];
                    let akq = a[k][q];
                    a[k][p] = cs * akp - sn * akq;
                    a[k][q] = sn * akp + cs * akq;
                }
                for k in 0..4 {
                    let apk = a[p][k];
                    let aqk = a[q][k];
                    a[p][k] = cs * apk - sn * aqk;
                    a[q][k] = sn * apk + cs * aqk;
                }
                for row in v.iter_mut() {
                    let vkp = row[p];
                    let vkq = row[q];
                    row[p] = cs * vkp - sn * vkq;
                    row[q] = sn * vkp + cs * vkq;
                }
            }
        }
    }
    let mut best = 0;
    for i in 1..4 {
        if a[i][i] > a[best][best] {
            best = i;
        }
    }
    let mut out = [v[0][best], v[1][best], v[2][best], v[3][best]];
    let dot: f64 = out.iter().zip(&prefer).map(|(a, b)| a * b).sum();
    if dot < 0.0 {
        for x in out.iter_mut() {
            *x = -*x;
        }
    }
    out
}

/// Rotation taking directions `a` to `b` (weighted, Horn 1987) that is
/// closest in sign to `start`.
fn fit_rotation(pairs: &[(Vec3, Vec3, f64)], start: Quat) -> Quat {
    let mut s = [[0.0; 3]; 3];
    for (a, b, w) in pairs {
        let a = [a.x, a.y, a.z];
        let b = [b.x, b.y, b.z];
        for (i, ai) in a.iter().enumerate() {
            for (j, bj) in b.iter().enumerate() {
                s[i][j] += w * ai * bj;
            }
        }
    }
    let (sxx, sxy, sxz) = (s[0][0], s[0][1], s[0][2]);
    let (syx, syy, syz) = (s[1][0], s[1][1], s[1][2]);
    let (szx, szy, szz) = (s[2][0], s[2][1], s[2][2]);
    let n = [
        [sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
        [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
        [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
        [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz],
    ];
    let e = top_eigenvector(n, [start.w, start.x, start.y, start.z]);
    Quat::new(e[0], e[1], e[2], e[3]).normalized()
}

/// Clamp a local pose quaternion to `max_deg`; returns whether it was.
pub fn clamp_to_limit(q: Quat, max_deg: f64) -> (Quat, bool) {
    let q = q.normalized();
    let angle = 2.0 * detmath::acos(q.w.abs().min(1.0));
    let max = max_deg.to_radians();
    if angle <= max + 1e-12 {
        (q, false)
    } else {
        let q = if q.w < 0.0 { q.neg() } else { q };
        (Quat::IDENTITY.slerp(q, max / angle).normalized(), true)
    }
}

/// Top-down fit of one frame. `targets[i]` is the wanted world head
/// position of bone i (confidence 0 = unknown).
fn fit_frame(c: &Creature, targets: &[Point], limited: &mut usize) -> Frame {
    let sk = &c.skeleton;
    let n = sk.len();
    let mut poses: Vec<Pose> = vec![
        Pose {
            loc: Vec3::ZERO,
            quat: Quat::IDENTITY
        };
        n
    ];
    let mut world: Vec<FkBone> = Vec::with_capacity(n);
    let children: Vec<Vec<usize>> = (0..n)
        .map(|i| (0..n).filter(|&k| sk.bones[k].parent == Some(i)).collect())
        .collect();
    for i in 0..n {
        let (rrot, roff) = sk.rest_parent_rel(i);
        let bone = &sk.bones[i];
        // Inherited orientation and head (identity local pose).
        let (parent_rot, parent_head) = match bone.parent {
            Some(p) => (world[p].rot, world[p].head),
            None => (Mat3::identity(), Vec3::ZERO),
        };
        let inherited = parent_rot.mul_mat(rrot);
        let head = if bone.parent.is_none() {
            if targets[i].conf > 0.0 {
                targets[i].p
            } else {
                bone.head
            }
        } else {
            parent_rot.mul_vec(roff) + parent_head
        };
        // Pairs: rest offsets to children -> observed offsets.
        let mut pairs: Vec<(Vec3, Vec3, f64)> = Vec::new();
        for &k in &children[i] {
            if targets[k].conf <= 0.0 {
                continue;
            }
            let rest = sk.bones[k].head - bone.head;
            let obs = targets[k].p - head;
            if rest.length() < 1e-9 || obs.length() < 1e-9 {
                continue;
            }
            pairs.push((
                rest.normalized(),
                obs.normalized(),
                targets[k].conf * rest.length(),
            ));
        }
        let rest_basis = sk.rest_world(i);
        let want_world: Mat3 = if pairs.is_empty() {
            inherited
        } else if pairs.len() == 1 {
            // Twist-free: the smallest turn from the inherited orientation.
            let cur = inherited
                .mul_mat(rest_basis.transpose())
                .mul_vec(pairs[0].0);
            let q = shortest_arc(cur, pairs[0].1);
            q.to_mat3().mul_mat(inherited)
        } else {
            // World delta D with D * rest = observed, started from the
            // inherited delta.
            let start_delta = Quat::from_mat3(inherited.mul_mat(rest_basis.transpose()));
            let d = fit_rotation(&pairs, start_delta);
            d.to_mat3().mul_mat(rest_basis)
        };
        // Local pose quat: inherited^T * want.
        let mut q = Quat::from_mat3(inherited.transpose().mul_mat(want_world));
        if bone.parent.is_some() {
            let (cq, was) = clamp_to_limit(q, c.max_deg[i]);
            if was {
                *limited += 1;
            }
            q = cq;
        } else {
            q = q.normalized();
        }
        let rot = inherited.mul_mat(q.to_mat3());
        let loc = if bone.parent.is_none() {
            rrot.transpose().mul_vec(head - roff)
        } else {
            Vec3::ZERO
        };
        poses[i] = Pose { loc, quat: q };
        world.push(FkBone { rot, head });
    }
    Frame { poses }
}

/// Gaussian smoothing of each bone's quaternion track (sign-aligned),
/// then re-clamped.
fn smooth_quats(c: &Creature, frames: &mut [Frame], sigma: f64) {
    if sigma <= 0.0 || frames.len() < 3 {
        return;
    }
    let radius = (3.0 * sigma).ceil() as i64;
    let n = c.skeleton.len();
    for i in 0..n {
        let orig: Vec<Quat> = frames.iter().map(|f| f.poses[i].quat).collect();
        let locs: Vec<Vec3> = frames.iter().map(|f| f.poses[i].loc).collect();
        for f in 0..frames.len() {
            let centre = orig[f];
            let mut acc = Quat::new(0.0, 0.0, 0.0, 0.0);
            let mut lacc = Vec3::ZERO;
            let mut wsum = 0.0;
            for k in -radius..=radius {
                let g = f as i64 + k;
                if g < 0 || g >= frames.len() as i64 {
                    continue;
                }
                let w = detmath::exp(-(k * k) as f64 / (2.0 * sigma * sigma));
                let mut q = orig[g as usize];
                if q.dot(centre) < 0.0 {
                    q = q.neg();
                }
                acc = Quat::new(
                    acc.w + w * q.w,
                    acc.x + w * q.x,
                    acc.y + w * q.y,
                    acc.z + w * q.z,
                );
                lacc = lacc + locs[g as usize].scale(w);
                wsum += w;
            }
            if wsum > 0.0 && acc.length() > 1e-12 {
                let q = acc.normalized();
                frames[f].poses[i].quat = if c.skeleton.bones[i].parent.is_some() {
                    clamp_to_limit(q, c.max_deg[i]).0
                } else {
                    q
                };
                frames[f].poses[i].loc = lacc.scale(1.0 / wsum);
            }
        }
    }
}

fn smoothstep(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Plant detection on one foot's tail track: (start, end) frame ranges
/// where the foot is low and slow for at least `min_frames`.
fn plants(
    track: &[Vec3],
    ground: f64,
    height_tol: f64,
    speed_tol: f64,
    min_frames: usize,
) -> Vec<(usize, usize)> {
    let n = track.len();
    let mut still = vec![false; n];
    for f in 0..n {
        let low = track[f].z <= ground + height_tol;
        let prev = if f > 0 { f - 1 } else { f };
        let next = if f + 1 < n { f + 1 } else { f };
        let mut d = track[next] - track[prev];
        d.z = 0.0;
        let span = (next - prev).max(1) as f64;
        still[f] = low && d.length() / span <= speed_tol;
    }
    let mut out = Vec::new();
    let mut f = 0;
    while f < n {
        if still[f] {
            let s = f;
            while f < n && still[f] {
                f += 1;
            }
            if f - s >= min_frames {
                out.push((s, f));
            }
        } else {
            f += 1;
        }
    }
    out
}

/// Move the foot's tail toward `target` by CCD on the leg chain, within
/// limits; `strength` in [0, 1] scales the correction.
fn ccd_foot(
    c: &Creature,
    frame: &mut Frame,
    foot: usize,
    chain: &[usize],
    target: Vec3,
    strength: f64,
) {
    let sk = &c.skeleton;
    for _iter in 0..8 {
        let posed = fk(sk, frame);
        let eff = tail_world(sk, &posed, foot);
        let goal = eff.lerp(target, strength);
        if (eff - goal).length() < 1e-5 {
            break;
        }
        for &b in chain.iter() {
            let posed = fk(sk, frame);
            let eff = tail_world(sk, &posed, foot);
            let head = posed[b].head;
            let v1 = eff - head;
            let v2 = goal - head;
            if v1.length() < 1e-9 || v2.length() < 1e-9 {
                continue;
            }
            let turn = shortest_arc(v1.normalized(), v2.normalized());
            // Damp for stability.
            let turn = Quat::IDENTITY.slerp(turn, 0.7);
            let new_world = turn.to_mat3().mul_mat(posed[b].rot);
            let (rrot, _) = sk.rest_parent_rel(b);
            let parent_rot = match sk.bones[b].parent {
                Some(p) => posed[p].rot,
                None => Mat3::identity(),
            };
            let inherited = parent_rot.mul_mat(rrot);
            let q = Quat::from_mat3(inherited.transpose().mul_mat(new_world));
            let (q, _) = clamp_to_limit(q, c.max_deg[b]);
            frame.poses[b].quat = q;
        }
    }
}

fn pin_feet(c: &Creature, frames: &mut [Frame], fps: f64, report: &mut SolveReport) {
    if c.feet.is_empty() || frames.len() < 3 {
        return;
    }
    let sk = &c.skeleton;
    let tracks: Vec<Vec<Vec3>> = c
        .feet
        .iter()
        .map(|&f| {
            frames
                .iter()
                .map(|fr| tail_world(sk, &fk(sk, fr), f))
                .collect()
        })
        .collect();
    let ground = tracks
        .iter()
        .flat_map(|t| t.iter().map(|p| p.z))
        .fold(f64::INFINITY, f64::min);
    let ramp_frames = (0.12 * fps).max(1.0);
    for (k, &foot) in c.feet.iter().enumerate() {
        let chain = c.leg_chain(foot, 3);
        let leg_len: f64 = chain.iter().map(|&b| sk.bone_length(b)).sum();
        let height_tol = 0.05 * c.height;
        let speed_tol = 0.5 * leg_len / fps;
        for (s, e) in plants(&tracks[k], ground, height_tol, speed_tol, 3) {
            report.plants += 1;
            let mut anchor = Vec3::ZERO;
            for p in &tracks[k][s..e] {
                anchor = anchor + *p;
            }
            let anchor = anchor.scale(1.0 / (e - s) as f64);
            let lo = (s as f64 - ramp_frames).floor().max(0.0) as usize;
            let hi = ((e as f64 + ramp_frames).ceil() as usize).min(frames.len());
            for f in lo..hi {
                let strength = if f < s {
                    smoothstep(1.0 - (s - f) as f64 / ramp_frames)
                } else if f >= e {
                    smoothstep(1.0 - (f + 1 - e) as f64 / ramp_frames)
                } else {
                    1.0
                };
                if strength <= 0.0 {
                    continue;
                }
                let before = tail_world(sk, &fk(sk, &frames[f]), foot);
                ccd_foot(c, &mut frames[f], foot, &chain, anchor, strength);
                let after = tail_world(sk, &fk(sk, &frames[f]), foot);
                let shift = (after - before).length() * 1000.0;
                if shift > report.max_pin_shift_mm {
                    report.max_pin_shift_mm = shift;
                }
            }
        }
    }
}

/// The whole solve. Errors when the two views disagree (unless forced).
pub fn solve(
    c: &Creature,
    d: &Detections,
    params: &SolveParams,
) -> Result<(Clip, SolveReport), String> {
    let (aligned, align) = prepare(c, d, params)?;
    let d = &aligned;
    let (mut points, med, ratio, seen) = triangulate_all(c, d, params)?;
    let mut report = SolveReport {
        frames: d.frames.len(),
        align,
        joints_seen: seen,
        joints_total: c.skeleton.len(),
        median_gap_ratio: med,
        motion_ratio: ratio,
        consistent: med <= params.max_gap_ratio && ratio <= params.max_motion_ratio,
        ..Default::default()
    };
    if !report.consistent && !params.force {
        return Err(format!(
            "the two views do not show the same move (median ray gap {:.1}% of the height, limit {:.0}%; height-disagreement motion ratio {:.2}, limit {:.2}); rejected (--force to solve anyway)",
            med * 100.0,
            params.max_gap_ratio * 100.0,
            ratio,
            params.max_motion_ratio
        ));
    }
    smooth_points(&mut points, params.smooth_sigma);
    let mut frames: Vec<Frame> = points
        .iter()
        .map(|row| fit_frame(c, row, &mut report.limited))
        .collect();
    smooth_quats(c, &mut frames, params.quat_sigma);
    if params.pin {
        pin_feet(c, &mut frames, d.fps, &mut report);
    }
    // Residual: triangulated vs solved heads.
    let mut sum = 0.0;
    let mut count = 0usize;
    for (f, frame) in frames.iter().enumerate() {
        let posed = fk(&c.skeleton, frame);
        for (i, pt) in points[f].iter().enumerate() {
            if pt.conf > 0.0 {
                sum += (posed[i].head - pt.p).length_sq();
                count += 1;
            }
        }
    }
    report.residual_mm = if count > 0 {
        (sum / count as f64).sqrt() * 1000.0
    } else {
        0.0
    };
    Ok((
        Clip {
            fps: d.fps,
            skeleton: c.skeleton.clone(),
            frames,
        },
        report,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip::parse_skeleton;
    use crate::creature::compare::compare;
    use crate::creature::detect::project_clip;
    use crate::creature::motion::exam_motion;
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
        build("hero", sk, feet, Vec3::new(0.0, -1.0, 0.0), lo, hi, 512)
    }

    #[test]
    fn fit_rotation_recovers_a_known_turn() {
        let q = Quat::from_axis_angle(Vec3::new(0.3, -0.5, 0.8).normalized(), 1.1);
        let a = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.2, 0.3, 0.9).normalized(),
        ];
        let pairs: Vec<(Vec3, Vec3, f64)> = a.iter().map(|v| (*v, q.rotate_vec(*v), 1.0)).collect();
        let got = fit_rotation(&pairs, Quat::IDENTITY);
        assert!(got.angle_to(q) < 1e-9, "{:?} vs {:?}", got, q);
    }

    #[test]
    fn clamp_caps_the_angle() {
        let q = Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 1.0);
        let (c, was) = clamp_to_limit(q, 30.0);
        assert!(was);
        assert!((2.0 * detmath::acos(c.w) - 30f64.to_radians()).abs() < 1e-9);
        assert!(!clamp_to_limit(q, 90.0).1);
    }

    #[test]
    fn perfect_detections_solve_back_to_the_truth() {
        let c = hero();
        let truth = exam_motion(&c, 36, 24.0, 11, 0.8);
        let dets = project_clip(&c, &truth, 0.0, 0.0, 1);
        let params = SolveParams {
            smooth_sigma: 0.0,
            quat_sigma: 0.0,
            pin: false,
            ..Default::default()
        };
        let (solved, report) = solve(&c, &dets, &params).unwrap();
        assert!(report.consistent);
        assert!(report.median_gap_ratio < 1e-6);
        let s = compare(&c, &truth, &solved).unwrap();
        // Heads are reproduced (leaf twist is unobservable, so compare positions).
        assert!(
            s.mean_mm < 1.0,
            "mean {} mm, max {} mm",
            s.mean_mm,
            s.max_mm
        );
        assert!(report.residual_mm < 1.0, "residual {}", report.residual_mm);
    }

    #[test]
    fn noisy_detections_stay_close_and_are_deterministic() {
        let c = hero();
        let truth = exam_motion(&c, 48, 24.0, 4, 0.8);
        let dets = project_clip(&c, &truth, 2.0, 0.05, 2);
        let (a, ra) = solve(&c, &dets, &SolveParams::default()).unwrap();
        let (b, _) = solve(&c, &dets, &SolveParams::default()).unwrap();
        assert_eq!(
            crate::clip::emit_clip(&a).unwrap(),
            crate::clip::emit_clip(&b).unwrap()
        );
        let s = compare(&c, &truth, &a).unwrap();
        // 2 px noise at 512 px over ~2.6 m of view is ~1 cm per joint.
        assert!(
            s.mean_mm < 40.0,
            "mean {} mm (max {}), report {:?}",
            s.mean_mm,
            s.max_mm,
            ra
        );
    }

    #[test]
    fn different_moves_in_the_two_views_are_rejected() {
        let c = hero();
        let a = exam_motion(&c, 24, 24.0, 1, 0.9);
        let b = exam_motion(&c, 24, 24.0, 2, 0.9);
        let da = project_clip(&c, &a, 0.0, 0.0, 1);
        let db = project_clip(&c, &b, 0.0, 0.0, 1);
        let mut mixed = da.clone();
        for (f, frame) in mixed.frames.iter_mut().enumerate() {
            frame[1] = db.frames[f][1].clone();
        }
        let err = solve(&c, &mixed, &SolveParams::default()).unwrap_err();
        assert!(err.contains("do not show the same move"), "{}", err);
        assert!(solve(
            &c,
            &mixed,
            &SolveParams {
                force: true,
                ..Default::default()
            }
        )
        .is_ok());
    }
}
