//! Lining up the two halves of a generated two-view video before the
//! solve. A video model is handed the start picture (the rest pose seen
//! through both cameras side by side) and returns a clip in which
//!
//! - the frames may carry their own timestamps (`times`): resampled to
//!   a uniform grid at the clip's fps when they are not uniform;
//! - each half may be re-framed (zoomed and shifted against the start
//!   picture, or simply decoded at another size): a zoom and a shift
//!   per view are fitted on the first frame, where the creature still
//!   stands in its rest pose, and every frame is mapped back through
//!   them;
//! - the two halves may move out of step: one view is slid in time
//!   against the other. Height is the only coordinate two orthogonal
//!   views share, so the slide is judged on height: a constant lag is
//!   chosen by the solve's own height-disagreement motion ratio, and a
//!   slowly varying lag (dynamic time warping on per-view joint heights,
//!   one lag step per frame at most) is kept only when it beats the
//!   constant lag clearly.
//!
//! Spec: docs/creature.md ("Lining up the views").

use super::camera::{closest_points, View};
use super::detect::{Det, Detections};
use super::profile::Creature;
use super::solve::{consistency, SolveParams};
use crate::detmath;
use crate::json::Json;
use crate::math::Vec3;

/// How one view is slid in time against the other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimeMode {
    Off,
    /// A constant lag (whole frames, then refined to a fraction).
    Lag,
    /// A slowly varying lag (dynamic time warping on joint heights).
    Warp,
    /// The constant lag when it lowers the motion ratio; the warp only
    /// when it beats the constant lag by `WARP_MARGIN`.
    Auto,
}

impl TimeMode {
    pub fn parse(s: &str) -> Option<TimeMode> {
        match s {
            "off" => Some(TimeMode::Off),
            "lag" => Some(TimeMode::Lag),
            "warp" => Some(TimeMode::Warp),
            "auto" => Some(TimeMode::Auto),
            _ => None,
        }
    }
}

/// The warp has more freedom than a constant lag, so it must lower the
/// motion ratio to below this fraction of the constant lag's to be used.
pub const WARP_MARGIN: f64 = 0.9;
/// Cost of changing the lag by one frame in the warp (in units of a
/// frame's normalised height disagreement: 0 = same heights, about 2 =
/// unrelated).
pub const WARP_STEP_COST: f64 = 0.1;
/// Gaussian sigma (frames) applied to the warp's lag path.
pub const WARP_SMOOTH: f64 = 2.0;

#[derive(Clone, Debug, Default)]
pub struct TimingReport {
    pub frames_in: usize,
    pub frames_out: usize,
    /// Largest distance of a timestamp from the uniform grid, in frames.
    pub max_jitter_frames: f64,
    pub resampled: bool,
}

#[derive(Clone, Debug, Default)]
pub struct ViewFit {
    pub view: String,
    /// Observed pixel offset from the image centre = scale x expected.
    pub scale: f64,
    pub du: f64,
    pub dv: f64,
    pub joints: usize,
    /// RMS distance (px) of the fitted frame's detections from the rest
    /// pose's projection, before and after the fit.
    pub rms_before_px: f64,
    pub rms_after_px: f64,
}

#[derive(Clone, Debug, Default)]
pub struct TimeReport {
    /// "none", "lag" or "warp".
    pub used: String,
    pub reason: String,
    /// The view slid (the second of the pair) and the constant lag found
    /// (frames; positive = that view runs late).
    pub view: String,
    pub lag_frames: f64,
    pub ratio_none: f64,
    pub ratio_lag: f64,
    pub ratio_warp: Option<f64>,
    /// Range of the warp's lag (frames), when it ran.
    pub warp_lag_min: f64,
    pub warp_lag_max: f64,
    pub frames_kept: usize,
    /// First frame of the original clip kept (frames before it lost
    /// their partner in the other view).
    pub first_frame: usize,
}

#[derive(Clone, Debug, Default)]
pub struct AlignReport {
    pub gap_before: f64,
    pub ratio_before: f64,
    pub timing: Option<TimingReport>,
    pub fits: Vec<ViewFit>,
    pub time: Option<TimeReport>,
}

fn r(v: f64, p: i32) -> f64 {
    (v * 10f64.powi(p)).round() / 10f64.powi(p)
}

impl AlignReport {
    pub fn to_json(&self) -> Json {
        let mut pairs: Vec<(&str, Json)> = vec![
            ("median_gap_ratio_before", Json::num(r(self.gap_before, 4))),
            ("motion_ratio_before", Json::num(r(self.ratio_before, 4))),
        ];
        if let Some(t) = &self.timing {
            pairs.push((
                "timing",
                Json::obj(vec![
                    ("frames_in", Json::num(t.frames_in as f64)),
                    ("frames_out", Json::num(t.frames_out as f64)),
                    ("max_jitter_frames", Json::num(r(t.max_jitter_frames, 3))),
                    ("resampled", Json::Bool(t.resampled)),
                ]),
            ));
        }
        if !self.fits.is_empty() {
            pairs.push((
                "views",
                Json::Arr(
                    self.fits
                        .iter()
                        .map(|f| {
                            Json::obj(vec![
                                ("view", Json::str(&f.view)),
                                ("scale", Json::num(r(f.scale, 4))),
                                ("du_px", Json::num(r(f.du, 2))),
                                ("dv_px", Json::num(r(f.dv, 2))),
                                ("joints", Json::num(f.joints as f64)),
                                ("rms_before_px", Json::num(r(f.rms_before_px, 2))),
                                ("rms_after_px", Json::num(r(f.rms_after_px, 2))),
                            ])
                        })
                        .collect(),
                ),
            ));
        }
        if let Some(t) = &self.time {
            let mut tp = vec![
                ("used", Json::str(&t.used)),
                ("reason", Json::str(&t.reason)),
                ("view", Json::str(&t.view)),
                ("lag_frames", Json::num(r(t.lag_frames, 3))),
                ("motion_ratio_none", Json::num(r(t.ratio_none, 4))),
                ("motion_ratio_lag", Json::num(r(t.ratio_lag, 4))),
            ];
            if let Some(w) = t.ratio_warp {
                tp.push(("motion_ratio_warp", Json::num(r(w, 4))));
                tp.push(("warp_lag_min", Json::num(r(t.warp_lag_min, 3))));
                tp.push(("warp_lag_max", Json::num(r(t.warp_lag_max, 3))));
            }
            tp.push(("frames_kept", Json::num(t.frames_kept as f64)));
            tp.push(("first_frame", Json::num(t.first_frame as f64)));
            pairs.push(("time", Json::obj(tp)));
        }
        Json::obj(pairs)
    }
}

/// The first two detection views the creature knows: (column, view).
fn view_pair<'a>(c: &'a Creature, d: &Detections) -> Result<[(usize, &'a View); 2], String> {
    let mut found = Vec::new();
    for (vi, name) in d.views.iter().enumerate() {
        if let Some(v) = c.view(name) {
            found.push((vi, v));
        }
    }
    if found.len() < 2 {
        return Err(format!(
            "detections need two of the creature's views; found {:?}",
            d.views
        ));
    }
    Ok([found[0], found[1]])
}

/// Bone index -> detection column.
fn columns(c: &Creature, d: &Detections) -> Vec<Option<usize>> {
    c.skeleton
        .bones
        .iter()
        .map(|b| d.joints.iter().position(|j| *j == b.name))
        .collect()
}

fn lerp_det(a: &Det, b: &Det, t: f64) -> Det {
    if t <= 1e-9 {
        return a.clone();
    }
    if t >= 1.0 - 1e-9 {
        return b.clone();
    }
    if a.conf > 0.0 && b.conf > 0.0 {
        Det {
            u: a.u + (b.u - a.u) * t,
            v: a.v + (b.v - a.v) * t,
            conf: a.conf.min(b.conf),
        }
    } else if t < 0.5 {
        a.clone()
    } else {
        b.clone()
    }
}

/// One view's detections at fractional frame `t` (clamped to the clip).
fn sample_view(d: &Detections, view: usize, t: f64) -> Vec<Det> {
    let n = d.frames.len();
    let t = t.clamp(0.0, (n - 1) as f64);
    let i = (t.floor() as usize).min(n - 1);
    let frac = t - i as f64;
    if i + 1 >= n {
        return d.frames[i][view].clone();
    }
    d.frames[i][view]
        .iter()
        .zip(&d.frames[i + 1][view])
        .map(|(a, b)| lerp_det(a, b, frac))
        .collect()
}

// ---------------------------------------------------------------- timing

/// Resample to a uniform grid at the clip's fps when the frames carry
/// timestamps more than a quarter frame off it; otherwise the frames are
/// kept as they are. The output never carries timestamps.
pub fn uniform_timing(d: &Detections) -> (Detections, TimingReport) {
    let n = d.frames.len();
    let mut out = d.clone();
    out.times = None;
    let mut rep = TimingReport {
        frames_in: n,
        frames_out: n,
        ..Default::default()
    };
    let Some(times) = &d.times else {
        return (out, rep);
    };
    let t0 = times[0];
    rep.max_jitter_frames = times
        .iter()
        .enumerate()
        .map(|(i, t)| ((t - t0) * d.fps - i as f64).abs())
        .fold(0.0, f64::max);
    if rep.max_jitter_frames <= 0.25 {
        return (out, rep);
    }
    let last = times[n - 1];
    let mut frames = Vec::new();
    let mut seg = 0usize;
    let mut k = 0usize;
    loop {
        let t = t0 + k as f64 / d.fps;
        if t > last + 1e-9 {
            break;
        }
        while seg + 1 < n - 1 && times[seg + 1] < t {
            seg += 1;
        }
        let (a, b) = (times[seg], times[(seg + 1).min(n - 1)]);
        let frac = if b > a {
            ((t - a) / (b - a)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let ft = seg as f64 + frac;
        frames.push(
            (0..d.views.len())
                .map(|v| sample_view(d, v, ft))
                .collect::<Vec<_>>(),
        );
        k += 1;
    }
    rep.frames_out = frames.len();
    rep.resampled = true;
    out.frames = frames;
    (out, rep)
}

// ---------------------------------------------------------------- cameras

/// Fit, per view, a zoom about the image centre and a shift that take
/// the rest pose's projection to the detections of frame `frame`
/// (observed - centre = scale * (expected - centre) + (du, dv)),
/// weighted by confidence, with outliers (a joint the finder put
/// elsewhere) down-weighted over five reweighting rounds.
pub fn fit_views(
    c: &Creature,
    d: &Detections,
    frame: usize,
    min_conf: f64,
) -> Result<Vec<ViewFit>, String> {
    if frame >= d.frames.len() {
        return Err(format!(
            "camera fit frame {} is past the clip ({} frames)",
            frame,
            d.frames.len()
        ));
    }
    let cols = columns(c, d);
    let mut fits = Vec::new();
    for (vi, name) in d.views.iter().enumerate() {
        let Some(view) = c.view(name) else { continue };
        let (cx, cy) = (view.width as f64 * 0.5, view.height as f64 * 0.5);
        // (expected - centre, observed - centre, confidence)
        let mut pts: Vec<([f64; 2], [f64; 2], f64)> = Vec::new();
        for (bi, col) in cols.iter().enumerate() {
            let Some(j) = col else { continue };
            let det = &d.frames[frame][vi][*j];
            if det.conf < min_conf {
                continue;
            }
            let (u, v, depth) = view.project(c.skeleton.bones[bi].head);
            if depth <= 0.0 {
                continue;
            }
            pts.push(([u - cx, v - cy], [det.u - cx, det.v - cy], det.conf));
        }
        if pts.len() < 3 {
            return Err(format!(
                "camera fit: view {} has {} confident joints on frame {} (need 3)",
                name,
                pts.len(),
                frame
            ));
        }
        let rms_before = (pts
            .iter()
            .map(|(x, y, _)| (y[0] - x[0]).powi(2) + (y[1] - x[1]).powi(2))
            .sum::<f64>()
            / pts.len() as f64)
            .sqrt();
        let mut robust = vec![1.0; pts.len()];
        let (mut s, mut du, mut dv) = (1.0, 0.0, 0.0);
        for _round in 0..5 {
            let mut wsum = 0.0;
            let mut mx = [0.0; 2];
            let mut my = [0.0; 2];
            for (k, (x, y, cf)) in pts.iter().enumerate() {
                let w = cf * robust[k];
                wsum += w;
                for a in 0..2 {
                    mx[a] += w * x[a];
                    my[a] += w * y[a];
                }
            }
            if wsum <= 0.0 {
                break;
            }
            for a in 0..2 {
                mx[a] /= wsum;
                my[a] /= wsum;
            }
            let mut num = 0.0;
            let mut den = 0.0;
            for (k, (x, y, cf)) in pts.iter().enumerate() {
                let w = cf * robust[k];
                for a in 0..2 {
                    num += w * (x[a] - mx[a]) * (y[a] - my[a]);
                    den += w * (x[a] - mx[a]) * (x[a] - mx[a]);
                }
            }
            if den <= 1e-12 {
                break;
            }
            s = num / den;
            du = my[0] - s * mx[0];
            dv = my[1] - s * mx[1];
            let res: Vec<f64> = pts
                .iter()
                .map(|(x, y, _)| {
                    ((s * x[0] + du - y[0]).powi(2) + (s * x[1] + dv - y[1]).powi(2)).sqrt()
                })
                .collect();
            let mut sorted = res.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let med = sorted[sorted.len() / 2];
            let cut = 2.5 * med.max(0.5);
            for (k, rk) in res.iter().enumerate() {
                robust[k] = 1.0 / (1.0 + (rk / cut) * (rk / cut));
            }
        }
        let rms_after = (pts
            .iter()
            .map(|(x, y, _)| (s * x[0] + du - y[0]).powi(2) + (s * x[1] + dv - y[1]).powi(2))
            .sum::<f64>()
            / pts.len() as f64)
            .sqrt();
        fits.push(ViewFit {
            view: name.clone(),
            scale: s,
            du,
            dv,
            joints: pts.len(),
            rms_before_px: rms_before,
            rms_after_px: rms_after,
        });
    }
    Ok(fits)
}

/// Map every detection back through its view's fit.
pub fn apply_view_fits(c: &Creature, d: &Detections, fits: &[ViewFit]) -> Detections {
    let mut out = d.clone();
    for (vi, name) in d.views.iter().enumerate() {
        let (Some(view), Some(fit)) = (c.view(name), fits.iter().find(|f| f.view == *name)) else {
            continue;
        };
        if fit.scale.abs() < 1e-9 {
            continue;
        }
        let (cx, cy) = (view.width as f64 * 0.5, view.height as f64 * 0.5);
        for frame in out.frames.iter_mut() {
            for det in frame[vi].iter_mut() {
                det.u = cx + (det.u - cx - fit.du) / fit.scale;
                det.v = cy + (det.v - cy - fit.dv) / fit.scale;
            }
        }
    }
    out
}

// ---------------------------------------------------------------- time

/// The clip with view `bv` read at `tb[f]` (fractional frame of the
/// original clip) against view `av`'s frame `f`; frames whose `tb` is
/// `None` are dropped. Returns the detections and the first kept frame.
fn resample_pair(d: &Detections, bv: usize, tb: &[Option<f64>]) -> (Detections, usize) {
    let mut out = d.clone();
    out.times = None;
    out.frames = Vec::new();
    let mut first = usize::MAX;
    for (f, t) in tb.iter().enumerate() {
        let Some(t) = t else { continue };
        if first == usize::MAX {
            first = f;
        }
        let mut frame = d.frames[f].clone();
        frame[bv] = sample_view(d, bv, *t);
        out.frames.push(frame);
    }
    (out, if first == usize::MAX { 0 } else { first })
}

fn lag_times(n: usize, lag: f64) -> Vec<Option<f64>> {
    (0..n)
        .map(|f| {
            let t = f as f64 + lag;
            if t >= -1e-9 && t <= (n - 1) as f64 + 1e-9 {
                Some(t.clamp(0.0, (n - 1) as f64))
            } else {
                None
            }
        })
        .collect()
}

/// Rays per frame and bone for one view (`None` when missing).
fn view_rays(
    d: &Detections,
    vi: usize,
    view: &View,
    cols: &[Option<usize>],
    min_conf: f64,
) -> Vec<Vec<Option<(Vec3, Vec3)>>> {
    d.frames
        .iter()
        .map(|frame| {
            cols.iter()
                .map(|col| {
                    let det = &frame[vi][(*col)?];
                    (det.conf >= min_conf).then(|| view.ray(det.u, det.v))
                })
                .collect()
        })
        .collect()
}

/// A slowly varying lag by dynamic time warping on height, the one
/// coordinate the two views share: for view A's frame `f` and lag `l`,
/// each joint's signed height disagreement between A's ray at `f` and
/// B's ray at `f + l` (the quantity the motion ratio is built from),
/// less that joint's mean disagreement along the best constant lag
/// `base`, squared and summed, over the joints' height variance. One lag
/// state per frame in [-k, k], changing by at most one frame per frame
/// at `WARP_STEP_COST` a step, solved exactly (Viterbi) over the frames
/// where every lag is in range, held at the ends, then smoothed.
/// Returns the lag per frame, or `None` when the clip is too short or
/// nothing moves in height.
fn warp_lags(
    c: &Creature,
    d: &Detections,
    pair: &[(usize, &View); 2],
    k: usize,
    base: i64,
    min_conf: f64,
) -> Option<Vec<f64>> {
    let n = d.frames.len();
    if k == 0 || n < 2 * k + 8 {
        return None;
    }
    let cols = columns(c, d);
    let ra = view_rays(d, pair[0].0, pair[0].1, &cols, min_conf);
    let rb = view_rays(d, pair[1].0, pair[1].1, &cols, min_conf);
    let nb = cols.len();
    // (disagreement, height) of joint j between A at f and B at g.
    let meet = |f: usize, g: usize, j: usize| -> Option<(f64, f64)> {
        let (a, b) = (ra[f][j]?, rb[g][j]?);
        let (pa, pb) = closest_points(a, b);
        Some((pa.z - pb.z, 0.5 * (pa.z + pb.z)))
    };
    // Per joint along the best constant lag: mean disagreement, and the
    // pooled height variance.
    let mut mean_gap: Vec<Option<f64>> = vec![None; nb];
    let mut var = 0.0;
    let mut cnt = 0usize;
    for (j, slot) in mean_gap.iter_mut().enumerate() {
        let pairs: Vec<(f64, f64)> = (0..n)
            .filter_map(|f| {
                let g = f as i64 + base;
                if g < 0 || g >= n as i64 {
                    return None;
                }
                meet(f, g as usize, j)
            })
            .collect();
        if pairs.len() * 2 < n {
            continue;
        }
        let m = pairs.len() as f64;
        *slot = Some(pairs.iter().map(|p| p.0).sum::<f64>() / m);
        let mz = pairs.iter().map(|p| p.1).sum::<f64>() / m;
        var += pairs.iter().map(|p| (p.1 - mz) * (p.1 - mz)).sum::<f64>();
        cnt += pairs.len();
    }
    if cnt == 0 {
        return None;
    }
    var /= cnt as f64;
    if var <= (0.005 * c.height).powi(2) {
        return None; // nothing moves in height: no lag to read
    }
    let states = 2 * k + 1;
    let cost = |f: usize, li: usize| -> f64 {
        let g = (f as i64 + li as i64 - k as i64) as usize;
        let mut s = 0.0;
        let mut w = 0.0;
        for (j, m) in mean_gap.iter().enumerate() {
            let Some(m) = m else { continue };
            if let Some((e, _)) = meet(f, g, j) {
                s += (e - m) * (e - m);
                w += 1.0;
            }
        }
        if w > 0.0 {
            s / (w * var)
        } else {
            1.0
        }
    };
    let (lo, hi) = (k, n - k);
    let mut acc: Vec<f64> = (0..states).map(|li| cost(lo, li)).collect();
    let mut back: Vec<Vec<usize>> = Vec::with_capacity(hi - lo);
    back.push((0..states).collect());
    for f in (lo + 1)..hi {
        let mut next = vec![0.0; states];
        let mut from = vec![0usize; states];
        for li in 0..states {
            // Stay first, then one step down, then one step up (ties stay).
            let mut best = acc[li];
            let mut arg = li;
            if li > 0 && acc[li - 1] + WARP_STEP_COST < best {
                best = acc[li - 1] + WARP_STEP_COST;
                arg = li - 1;
            }
            if li + 1 < states && acc[li + 1] + WARP_STEP_COST < best {
                best = acc[li + 1] + WARP_STEP_COST;
                arg = li + 1;
            }
            next[li] = best + cost(f, li);
            from[li] = arg;
        }
        acc = next;
        back.push(from);
    }
    // Best end state (ties to the smallest |lag|).
    let mut end = k;
    for li in 0..states {
        let better = acc[li] < acc[end]
            || (acc[li] == acc[end]
                && (li as i64 - k as i64).abs() < (end as i64 - k as i64).abs());
        if better {
            end = li;
        }
    }
    let mut path = vec![0usize; hi - lo];
    let mut cur = end;
    for i in (0..(hi - lo)).rev() {
        path[i] = cur;
        cur = back[i][cur];
    }
    let mut lags = vec![0.0; n];
    for (f, lag) in lags.iter_mut().enumerate() {
        let i = f.clamp(lo, hi - 1) - lo;
        *lag = path[i] as f64 - k as f64;
    }
    // Smooth the path (edges held), keep B's time non-decreasing.
    let radius = (3.0 * WARP_SMOOTH).ceil() as i64;
    let mut smooth = vec![0.0; n];
    for (f, out) in smooth.iter_mut().enumerate() {
        let mut acc = 0.0;
        let mut wsum = 0.0;
        for o in -radius..=radius {
            let g = (f as i64 + o).clamp(0, n as i64 - 1) as usize;
            let w = detmath::exp(-((o * o) as f64) / (2.0 * WARP_SMOOTH * WARP_SMOOTH));
            acc += w * lags[g];
            wsum += w;
        }
        *out = acc / wsum;
    }
    for f in 1..n {
        let prev = (f - 1) as f64 + smooth[f - 1];
        if f as f64 + smooth[f] < prev {
            smooth[f] = prev - f as f64;
        }
    }
    Some(smooth)
}

/// Slide the second view of the pair in time against the first.
pub fn align_time(
    c: &Creature,
    d: &Detections,
    mode: TimeMode,
    max_lag: usize,
    params: &SolveParams,
) -> Result<(Detections, TimeReport), String> {
    let pair = view_pair(c, d)?;
    let bv = pair[1].0;
    let n = d.frames.len();
    let (_, ratio_none) = consistency(c, d, params)?;
    let mut rep = TimeReport {
        used: "none".into(),
        view: d.views[bv].clone(),
        ratio_none,
        ratio_lag: ratio_none,
        frames_kept: n,
        ..Default::default()
    };
    if mode == TimeMode::Off {
        rep.reason = "off".into();
        return Ok((d.clone(), rep));
    }
    let k = max_lag.min(n.saturating_sub(1) / 4);
    // Constant lag: whole frames, nearest zero first on ties.
    let ratio_at = |lag: f64| -> Result<f64, String> {
        let (dd, _) = resample_pair(d, bv, &lag_times(n, lag));
        Ok(consistency(c, &dd, params)?.1)
    };
    let mut ratios = vec![0.0; 2 * k + 1];
    let mut best = 0i64;
    for step in 0..=(2 * k) {
        let lag = if step % 2 == 1 {
            -(step.div_ceil(2) as i64)
        } else {
            (step / 2) as i64
        };
        let rl = if lag == 0 {
            ratio_none
        } else {
            ratio_at(lag as f64)?
        };
        ratios[(lag + k as i64) as usize] = rl;
        if rl < ratios[(best + k as i64) as usize] {
            best = lag;
        }
    }
    let mut lag = best as f64;
    let mut ratio_lag = ratios[(best + k as i64) as usize];
    let bi = (best + k as i64) as usize;
    if bi > 0 && bi + 1 < ratios.len() {
        let (rm, r0, rp) = (ratios[bi - 1], ratios[bi], ratios[bi + 1]);
        let den = rm - 2.0 * r0 + rp;
        if den > 1e-12 {
            let delta = (0.5 * (rm - rp) / den).clamp(-0.5, 0.5);
            if delta.abs() > 1e-3 {
                let rf = ratio_at(lag + delta)?;
                if rf < ratio_lag {
                    lag += delta;
                    ratio_lag = rf;
                }
            }
        }
    }
    rep.lag_frames = lag;
    rep.ratio_lag = ratio_lag;
    // Warp.
    let warp = if matches!(mode, TimeMode::Warp | TimeMode::Auto) {
        match warp_lags(c, d, &pair, k, best, params.min_conf) {
            Some(lags) => {
                let tb: Vec<Option<f64>> = lags
                    .iter()
                    .enumerate()
                    .map(|(f, l)| {
                        let t = f as f64 + l;
                        (t >= -1e-9 && t <= (n - 1) as f64 + 1e-9)
                            .then(|| t.clamp(0.0, (n - 1) as f64))
                    })
                    .collect();
                let (dd, first) = resample_pair(d, bv, &tb);
                if dd.frames.len() < 8 {
                    None
                } else {
                    let rw = consistency(c, &dd, params)?.1;
                    rep.ratio_warp = Some(rw);
                    rep.warp_lag_min = lags.iter().cloned().fold(f64::INFINITY, f64::min);
                    rep.warp_lag_max = lags.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    Some((dd, first, rw))
                }
            }
            None => None,
        }
    } else {
        None
    };
    let use_lag = lag.abs() > 1e-9 && ratio_lag < ratio_none;
    let base = if use_lag { ratio_lag } else { ratio_none };
    let choose = match mode {
        TimeMode::Lag => "lag",
        TimeMode::Warp => {
            if warp.is_some() {
                "warp"
            } else {
                "lag"
            }
        }
        _ => match &warp {
            Some((_, _, rw)) if *rw < WARP_MARGIN * base => "warp",
            _ if use_lag => "lag",
            _ => "none",
        },
    };
    rep.reason = match (mode, choose) {
        (TimeMode::Lag, _) => "lag asked for".into(),
        (TimeMode::Warp, "warp") => "warp asked for".into(),
        (TimeMode::Warp, _) => "warp asked for but the clip is too short or still".into(),
        (_, "warp") => format!(
            "the warp's motion ratio is under {:.0}% of the constant lag's",
            WARP_MARGIN * 100.0
        ),
        (_, "lag") => "a constant lag lowers the motion ratio; the warp does not beat it".into(),
        _ => "no lag lowers the motion ratio".into(),
    };
    rep.used = choose.into();
    match choose {
        "warp" => {
            let (dd, first, _) = warp.expect("warp chosen only when it ran");
            rep.frames_kept = dd.frames.len();
            rep.first_frame = first;
            Ok((dd, rep))
        }
        "lag" => {
            let (dd, first) = resample_pair(d, bv, &lag_times(n, lag));
            rep.frames_kept = dd.frames.len();
            rep.first_frame = first;
            Ok((dd, rep))
        }
        _ => Ok((d.clone(), rep)),
    }
}

/// Everything `solve` does to the detections before triangulating, in
/// order: uniform timing, the camera fit on `params.align_frame`, the
/// time alignment. Returns the detections to solve and the report
/// (`None` when no alignment was asked for: the detections go through
/// untouched).
pub fn prepare(
    c: &Creature,
    d: &Detections,
    params: &SolveParams,
) -> Result<(Detections, Option<AlignReport>), String> {
    let timing_needed = d.times.is_some();
    if !params.align_views && params.time_mode == TimeMode::Off && !timing_needed {
        return Ok((d.clone(), None));
    }
    let (gap_before, ratio_before) = consistency(c, d, params)?;
    let mut rep = AlignReport {
        gap_before,
        ratio_before,
        ..Default::default()
    };
    let (mut cur, timing) = uniform_timing(d);
    if timing_needed {
        rep.timing = Some(timing);
    }
    if params.align_views {
        let fits = fit_views(c, &cur, params.align_frame, params.min_conf)?;
        for f in &fits {
            if !(0.5..=2.0).contains(&f.scale) && !params.force {
                return Err(format!(
                    "camera fit: view {} would be zoomed {:.3}x (outside 0.5..2); is frame {} the rest pose? (--force to go on)",
                    f.view, f.scale, params.align_frame
                ));
            }
        }
        cur = apply_view_fits(c, &cur, &fits);
        rep.fits = fits;
    }
    if params.time_mode != TimeMode::Off {
        let (dd, t) = align_time(c, &cur, params.time_mode, params.max_lag, params)?;
        cur = dd;
        rep.time = Some(t);
    }
    Ok((cur, Some(rep)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lerp_keeps_missing_points_missing() {
        let a = Det {
            u: 0.0,
            v: 0.0,
            conf: 1.0,
        };
        let b = Det {
            u: 10.0,
            v: 20.0,
            conf: 0.5,
        };
        let m = lerp_det(&a, &b, 0.25);
        assert_eq!((m.u, m.v, m.conf), (2.5, 5.0, 0.5));
        let gone = Det {
            u: 0.0,
            v: 0.0,
            conf: 0.0,
        };
        assert_eq!(lerp_det(&a, &gone, 0.4).conf, 1.0);
        assert_eq!(lerp_det(&a, &gone, 0.6).conf, 0.0);
    }

    #[test]
    fn modes_parse() {
        assert_eq!(TimeMode::parse("auto"), Some(TimeMode::Auto));
        assert_eq!(TimeMode::parse("warp"), Some(TimeMode::Warp));
        assert_eq!(TimeMode::parse("x"), None);
    }
}
