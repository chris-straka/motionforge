//! Physics pass: make keyed motion physically believable.
//!
//! No ML, one small IK. Three checks over a clip, each with a fix:
//! - Balance: horizontal center of mass against the support feet on
//!   contact frames. The fix leans the body about the ground-level
//!   support pivot toward the pinned pre-fix support (plus ankle
//!   compensation); the report also gives a suggested nudge vector
//!   for the animator.
//! - Ballistic: root height over airborne phases must ride a parabola;
//!   the fix replaces it (blended at the boundaries).
//! - Momentum: flags root horizontal acceleration and turn-rate spikes
//!   (teleports, unphysical corners); the fix selectively smooths the
//!   root x/y curve around flagged frames.
//!
//! COM is a segment-weighted mean of joint world positions (Dempster
//! weights via Winter, mapped by bone-name keywords, normalized;
//! unmatched bones share a small remainder). Approximate by design —
//! the report quotes numbers, not verdicts, and the Blender side draws
//! the overlay from them.

use crate::clip::{fk, Clip};
use crate::math::{Mat3, Quat, Vec3};

#[derive(Clone, Debug)]
pub struct PhysicsParams {
    pub root: String,
    pub feet: Vec<String>,
    pub contact_margin: f64,
    pub foot_radius: f64,
    pub balance_margin: f64,
    pub min_air_frames: usize,
    pub blend_frames: usize,
    pub accel_limit: f64,
    pub turn_deg: f64,
    pub min_turn_speed: f64,
    pub smooth_sigma: f64,
    pub smooth_pad: usize,
    pub fix_ballistic: bool,
    pub fix_momentum: bool,
    pub fix_balance: bool,
    /// Total lean cap per frame (deg). Big violations get an honest
    /// partial fix: more lean would float or bury the feet.
    pub max_lean_deg: f64,
}

impl Default for PhysicsParams {
    fn default() -> PhysicsParams {
        PhysicsParams {
            root: String::new(),
            feet: Vec::new(),
            contact_margin: 0.03,
            foot_radius: 0.12,
            balance_margin: 0.05,
            min_air_frames: 4,
            blend_frames: 2,
            accel_limit: 25.0,
            turn_deg: 40.0,
            min_turn_speed: 0.5,
            smooth_sigma: 1.0,
            smooth_pad: 2,
            fix_ballistic: true,
            fix_momentum: true,
            fix_balance: true,
            max_lean_deg: 8.0,
        }
    }
}

impl PhysicsParams {
    pub fn validate(&self) -> Result<(), String> {
        if self.root.is_empty() {
            return Err("physics needs --root <bone>".to_string());
        }
        if self.feet.is_empty() {
            return Err("physics needs --feet <a,b,...>".to_string());
        }
        if self.contact_margin < 0.0 || self.foot_radius <= 0.0 || self.balance_margin < 0.0 {
            return Err("contact/foot/balance margins must be >= 0 (foot radius > 0)".to_string());
        }
        if self.min_air_frames < 3 {
            return Err("min-air-frames must be >= 3".to_string());
        }
        if self.blend_frames < 1 {
            return Err("blend-frames must be >= 1".to_string());
        }
        if self.accel_limit <= 0.0 || self.turn_deg <= 0.0 || self.min_turn_speed < 0.0 {
            return Err("accel/turn limits must be positive".to_string());
        }
        if self.smooth_sigma <= 0.0 {
            return Err("smooth-sigma must be positive".to_string());
        }
        if self.max_lean_deg <= 0.0 || self.max_lean_deg > 45.0 {
            return Err("max-lean-deg must be in (0, 45]".to_string());
        }
        Ok(())
    }
}

/// Raw COM weight for a bone name (normalized later). First matching
/// rule wins; order matters ("upleg" before "leg", "forearm" before
/// "arm").
fn raw_weight(name: &str) -> f64 {
    let n = name.to_lowercase();
    if n.contains("upleg") || n.contains("thigh") {
        return 0.10;
    }
    if n.contains("forearm") || n.contains("fore arm") {
        return 0.016;
    }
    if n.contains("shin") || n.contains("shank") || n.contains("leg") {
        return 0.0465;
    }
    if n.contains("foot") || n.contains("toe") {
        return 0.0145;
    }
    if n.contains("upper_arm") || n.contains("upperarm") || n.contains(" arm") || n == "arm" {
        return 0.028;
    }
    // Bare *Arm (Mixamo LeftArm) after forearm/upper checks.
    if n.contains("arm") {
        return 0.022;
    }
    if n.contains("hand") || n.contains("finger") || n.contains("thumb") {
        return 0.007;
    }
    if n.contains("head") || n.contains("neck") {
        return 0.035;
    }
    if n.contains("hips") || n.contains("pelvis") {
        return 0.15;
    }
    if n.contains("spine") || n.contains("torso") || n.contains("chest") {
        return 0.06;
    }
    0.005
}

/// Normalized per-bone COM weights (sum = 1).
pub fn com_weights(clip: &Clip) -> Vec<f64> {
    let raw: Vec<f64> = clip
        .skeleton
        .bones
        .iter()
        .map(|b| raw_weight(&b.name))
        .collect();
    let sum: f64 = raw.iter().sum();
    raw.iter().map(|w| w / sum).collect()
}

pub fn center_of_mass(weights: &[f64], heads: &[Vec3]) -> Vec3 {
    let mut c = Vec3::ZERO;
    for (w, h) in weights.iter().zip(heads.iter()) {
        c = c + h.scale(*w);
    }
    c
}

#[derive(Clone, Debug, Default)]
pub struct BalanceFinding {
    pub frame: usize,
    pub excursion_m: f64,
    /// Suggested COM nudge (x, y) toward the support center, meters.
    pub nudge: (f64, f64),
}

#[derive(Clone, Debug, Default)]
pub struct BallisticPhase {
    pub start: usize,
    pub end: usize,
    pub residual_m: f64,
}

#[derive(Clone, Debug)]
pub struct FramePhysics {
    pub frame: usize,
    pub com: Vec3,
    pub root: Vec3,
    pub feet: Vec<(String, Vec3)>,
    pub supporters: Vec<String>,
    pub support_center: Option<Vec3>,
    pub support_radius: f64,
    pub excursion_m: f64,
    pub balanced: bool,
    pub airborne: bool,
}

/// Single-frame balance snapshot for the 3D overlay: COM, support
/// polygon summary, per-foot positions. Ground is the clip minimum, as
/// in the full check.
pub fn frame_physics(
    clip: &Clip,
    params: &PhysicsParams,
    frame: usize,
) -> Result<FramePhysics, String> {
    params.validate()?;
    if frame >= clip.frames.len() {
        return Err(format!(
            "frame {} out of range ({} frames)",
            frame,
            clip.frames.len()
        ));
    }
    let root_idx = clip
        .skeleton
        .index(&params.root)
        .ok_or_else(|| format!("unknown root bone \"{}\"", params.root))?;
    let mut feet_idx = Vec::new();
    for name in &params.feet {
        feet_idx.push(
            clip.skeleton
                .index(name)
                .ok_or_else(|| format!("unknown foot bone \"{}\"", name))?,
        );
    }
    let tracks = build_tracks(clip, root_idx, &feet_idx);
    let weights = com_weights(clip);
    let ground = tracks
        .feet
        .iter()
        .flat_map(|f| f.iter().map(|p| p.z))
        .fold(f64::INFINITY, f64::min);
    let heads = &tracks.heads[frame];
    let com = center_of_mass(&weights, heads);
    let feet: Vec<(String, Vec3)> = feet_idx
        .iter()
        .map(|i| (clip.skeleton.bones[*i].name.clone(), heads[*i]))
        .collect();
    let supporters: Vec<(String, Vec3)> = feet
        .iter()
        .filter(|(_, p)| p.z <= ground + params.contact_margin)
        .cloned()
        .collect();
    if supporters.is_empty() {
        return Ok(FramePhysics {
            frame,
            com,
            root: tracks.root[frame],
            feet,
            supporters: vec![],
            support_center: None,
            support_radius: 0.0,
            excursion_m: 0.0,
            balanced: true,
            airborne: true,
        });
    }
    let center = supporters
        .iter()
        .fold(Vec3::ZERO, |a, (_, p)| a + *p)
        .scale(1.0 / supporters.len() as f64);
    let radius = supporters
        .iter()
        .map(|(_, p)| ((p.x - center.x).powi(2) + (p.y - center.y).powi(2)).sqrt())
        .fold(0.0, f64::max)
        + params.foot_radius;
    let dist = ((com.x - center.x).powi(2) + (com.y - center.y).powi(2)).sqrt();
    let excursion = dist - radius;
    Ok(FramePhysics {
        frame,
        com,
        root: tracks.root[frame],
        feet,
        supporters: supporters.iter().map(|(n, _)| n.clone()).collect(),
        support_center: Some(center),
        support_radius: radius,
        excursion_m: excursion,
        balanced: excursion <= params.balance_margin,
        airborne: false,
    })
}

/// Emit a `motionforge-frame-physics` JSON document (stdout of
/// `physics-frame`, parsed by the Blender overlay operator).
pub fn emit_frame_physics(snap: &FramePhysics) -> Result<String, String> {
    use crate::json::{push_num, push_str};
    let mut out = String::new();
    out.push_str("{\"format\": \"motionforge-frame-physics\", \"version\": 1,\n");
    out.push_str(&format!(" \"frame\": {},\n", snap.frame));
    let mut vec = |key: &str, v: Vec3| -> Result<(), String> {
        out.push_str(&format!(" \"{}\": [", key));
        push_num(&mut out, v.x)?;
        out.push_str(", ");
        push_num(&mut out, v.y)?;
        out.push_str(", ");
        push_num(&mut out, v.z)?;
        out.push_str("],\n");
        Ok(())
    };
    vec("com", snap.com)?;
    vec("root", snap.root)?;
    out.push_str(" \"feet\": {");
    for (i, (name, pos)) in snap.feet.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        push_str(&mut out, name);
        out.push_str(": [");
        push_num(&mut out, pos.x)?;
        out.push_str(", ");
        push_num(&mut out, pos.y)?;
        out.push_str(", ");
        push_num(&mut out, pos.z)?;
        out.push(']');
    }
    out.push_str("},\n \"supporters\": [");
    for (i, name) in snap.supporters.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        push_str(&mut out, name);
    }
    out.push_str("],\n");
    match snap.support_center {
        Some(c) => {
            out.push_str(" \"support_center\": [");
            push_num(&mut out, c.x)?;
            out.push_str(", ");
            push_num(&mut out, c.y)?;
            out.push_str(", ");
            push_num(&mut out, c.z)?;
            out.push_str("],\n");
        }
        None => out.push_str(" \"support_center\": null,\n"),
    }
    out.push_str(" \"support_radius\": ");
    push_num(&mut out, snap.support_radius)?;
    out.push_str(",\n \"excursion_m\": ");
    push_num(&mut out, snap.excursion_m)?;
    out.push_str(&format!(
        ",\n \"balanced\": {},\n \"airborne\": {}\n}}\n",
        snap.balanced, snap.airborne
    ));
    Ok(out)
}

#[derive(Clone, Debug, Default)]
pub struct PhysicsReport {
    pub contact_frames: usize,
    pub airborne_frames: usize,
    pub balance_violations: Vec<BalanceFinding>,
    pub mean_excursion_m: f64,
    pub ballistic_phases: Vec<BallisticPhase>,
    pub accel_flags: Vec<usize>,
    pub turn_flags: Vec<usize>,
    pub max_accel_m_s2: f64,
    pub max_turn_deg: f64,
    pub fixed_ballistic: usize,
    pub fixed_momentum_frames: usize,
    pub fixed_balance_frames: usize,
    pub balance_worst_before_m: f64,
    pub balance_worst_after_m: f64,
}

struct Tracks {
    heads: Vec<Vec<Vec3>>, // heads[frame][bone]
    root: Vec<Vec3>,
    feet: Vec<Vec<Vec3>>, // feet[foot][frame]
}

fn build_tracks(clip: &Clip, root_idx: usize, feet_idx: &[usize]) -> Tracks {
    let mut heads = Vec::with_capacity(clip.frames.len());
    let mut root = Vec::with_capacity(clip.frames.len());
    let mut feet = vec![Vec::with_capacity(clip.frames.len()); feet_idx.len()];
    for frame in &clip.frames {
        let posed = fk(&clip.skeleton, frame);
        let h: Vec<Vec3> = posed.iter().map(|p| p.head).collect();
        root.push(h[root_idx]);
        for (fi, idx) in feet_idx.iter().enumerate() {
            feet[fi].push(h[*idx]);
        }
        heads.push(h);
    }
    Tracks { heads, root, feet }
}

/// Least-squares parabola `z = a t^2 + b t + c` over samples; returns
/// (a, b, c) or None when singular.
fn fit_parabola(t: &[f64], z: &[f64]) -> Option<(f64, f64, f64)> {
    debug_assert_eq!(t.len(), z.len());
    let n = t.len() as f64;
    let (mut s1, mut s2, mut s3, mut s4) = (0.0, 0.0, 0.0, 0.0);
    let (mut u0, mut u1, mut u2) = (0.0, 0.0, 0.0);
    for (ti, zi) in t.iter().zip(z.iter()) {
        let t2 = ti * ti;
        s1 += ti;
        s2 += t2;
        s3 += t2 * ti;
        s4 += t2 * t2;
        u0 += zi;
        u1 += zi * ti;
        u2 += zi * t2;
    }
    // Normal equations in (c, b, a) order; Gaussian elimination.
    let mut m = [[n, s1, s2, u0], [s1, s2, s3, u1], [s2, s3, s4, u2]];
    for col in 0..3 {
        let mut pivot = col;
        for row in col + 1..3 {
            if m[row][col].abs() > m[pivot][col].abs() {
                pivot = row;
            }
        }
        if m[pivot][col].abs() < 1e-12 {
            return None;
        }
        m.swap(col, pivot);
        for row in col + 1..3 {
            let f = m[row][col] / m[col][col];
            for k in col..4 {
                m[row][k] -= f * m[col][k];
            }
        }
    }
    let mut x = [0.0; 3];
    for i in (0..3).rev() {
        x[i] = (m[i][3] - (i + 1..3).map(|k| m[i][k] * x[k]).sum::<f64>()) / m[i][i];
    }
    Some((x[2], x[1], x[0]))
}

fn gaussian_smooth(values: &[f64], sigma: f64) -> Vec<f64> {
    let radius = (sigma * 3.0).ceil() as isize;
    let mut out = Vec::with_capacity(values.len());
    for i in 0..values.len() {
        let mut acc = 0.0;
        let mut norm = 0.0;
        for k in -radius..=radius {
            let j = (i as isize + k).clamp(0, values.len() as isize - 1) as usize;
            let w = (-0.5 * (k as f64 / sigma).powi(2)).exp();
            acc += values[j] * w;
            norm += w;
        }
        out.push(acc / norm);
    }
    out
}

/// Analyze only (shared by check and fix; fix applies corrections to
/// `clip` first when the corresponding flags are set).
pub fn physics_fix(clip: &Clip, params: &PhysicsParams) -> Result<(Clip, PhysicsReport), String> {
    params.validate()?;
    let root_idx = clip
        .skeleton
        .index(&params.root)
        .ok_or_else(|| format!("unknown root bone \"{}\"", params.root))?;
    let mut feet_idx = Vec::new();
    for name in &params.feet {
        feet_idx.push(
            clip.skeleton
                .index(name)
                .ok_or_else(|| format!("unknown foot bone \"{}\"", name))?,
        );
    }

    let mut out = clip.clone();

    // Ballistic fix (root z over airborne phases).
    let mut fixed_ballistic = 0usize;
    if params.fix_ballistic {
        let tracks = build_tracks(&out, root_idx, &feet_idx);
        let ground = tracks
            .feet
            .iter()
            .flat_map(|f| f.iter().map(|p| p.z))
            .fold(f64::INFINITY, f64::min);
        let contact: Vec<bool> = (0..out.frames.len())
            .map(|f| {
                tracks
                    .feet
                    .iter()
                    .any(|foot| foot[f].z <= ground + params.contact_margin)
            })
            .collect();
        let mut f = 0;
        while f < contact.len() {
            if contact[f] {
                f += 1;
                continue;
            }
            let mut e = f;
            while e + 1 < contact.len() && !contact[e + 1] {
                e += 1;
            }
            if e + 1 - f >= params.min_air_frames {
                let t: Vec<f64> = (f..=e).map(|k| (k - f) as f64 / out.fps).collect();
                let z: Vec<f64> = (f..=e).map(|k| tracks.root[k].z).collect();
                if let Some((a, b, c)) = fit_parabola(&t, &z) {
                    let len = e + 1 - f;
                    for k in f..=e {
                        let kk = k - f;
                        let w = ((kk.min(len - 1 - kk)) as f64 / params.blend_frames as f64)
                            .clamp(0.0, 1.0);
                        let tk = (kk) as f64 / out.fps;
                        let fit = a * tk * tk + b * tk + c;
                        // Root z fix adjusts the loc channel: convert the
                        // world-space delta through the rest basis. The
                        // root's parent frame is identity-ish for the
                        // translation part only when the root rest
                        // rotation is identity, so convert properly via
                        // the parent-relative rest rotation.
                        let dz = (fit - tracks.root[k].z) * w;
                        if dz != 0.0 {
                            // World delta -> local loc delta. Ancestors of
                            // the root are unposed in v1 clips, so the
                            // world-to-local rotation is exactly the
                            // root's rest armature orientation, transposed.
                            let to_local = out.skeleton.rest_world(root_idx).transpose();
                            let corr = to_local.mul_vec(Vec3::new(0.0, 0.0, dz));
                            let loc = &mut out.frames[k].poses[root_idx].loc;
                            *loc = *loc + corr;
                        }
                    }
                    fixed_ballistic += 1;
                }
            }
            f = e + 1;
        }
    }

    // Momentum fix (selective root x/y smoothing around flagged frames).
    let mut fixed_momentum_frames = 0usize;
    if params.fix_momentum && out.frames.len() >= 3 {
        let tracks = build_tracks(&out, root_idx, &feet_idx);
        let (accel_flags, turn_flags) = momentum_flags(&tracks.root, out.fps, params);
        let mut mask = vec![false; out.frames.len()];
        for flag in accel_flags.iter().chain(turn_flags.iter()) {
            let lo = flag.saturating_sub(params.smooth_pad);
            let hi = (*flag + params.smooth_pad).min(mask.len() - 1);
            for k in lo..=hi {
                mask[k] = true;
            }
        }
        if mask.iter().any(|m| *m) {
            let xs: Vec<f64> = tracks.root.iter().map(|p| p.x).collect();
            let ys: Vec<f64> = tracks.root.iter().map(|p| p.y).collect();
            let sx = gaussian_smooth(&xs, params.smooth_sigma);
            let sy = gaussian_smooth(&ys, params.smooth_sigma);
            // Crossfade over the outer pad ring of each masked run.
            for k in 0..mask.len() {
                if !mask[k] {
                    continue;
                }
                // Distance to the nearest unmasked frame (0 = edge).
                let mut edge = params.smooth_pad + 1;
                for d in 0..=params.smooth_pad {
                    let lo = k.saturating_sub(d);
                    let hi = (k + d).min(mask.len() - 1);
                    if !mask[lo] || !mask[hi] {
                        edge = d;
                        break;
                    }
                }
                let w = (edge as f64 / params.smooth_pad.max(1) as f64).clamp(0.0, 1.0);
                let dx = (sx[k] - tracks.root[k].x) * w;
                let dy = (sy[k] - tracks.root[k].y) * w;
                if dx != 0.0 || dy != 0.0 {
                    let to_local = out.skeleton.rest_world(root_idx).transpose();
                    let corr = to_local.mul_vec(Vec3::new(dx, dy, 0.0));
                    let loc = &mut out.frames[k].poses[root_idx].loc;
                    *loc = *loc + corr;
                    fixed_momentum_frames += 1;
                }
            }
        }
    }

    // Balance fix LAST: it is the final placement of the body over the
    // feet, and earlier root-curve fixes must not move COM afterwards.
    let mut fixed_balance_frames = 0usize;
    let mut balance_worst_before = worst_excursion(&out, params, root_idx, &feet_idx);
    if params.fix_balance {
        fixed_balance_frames = balance_fix_pass(&mut out, params, root_idx, &feet_idx);
    }
    let balance_worst_after = worst_excursion(&out, params, root_idx, &feet_idx);
    if !params.fix_balance {
        balance_worst_before = balance_worst_after;
    }

    let mut report = physics_check(&out, params, fixed_ballistic, fixed_momentum_frames)?;
    report.fixed_balance_frames = fixed_balance_frames;
    report.balance_worst_before_m = balance_worst_before;
    report.balance_worst_after_m = balance_worst_after;
    Ok((out, report))
}

/// Worst COM excursion over contact frames (before/after bookkeeping).
fn worst_excursion(
    clip: &Clip,
    params: &PhysicsParams,
    root_idx: usize,
    feet_idx: &[usize],
) -> f64 {
    let tracks = build_tracks(clip, root_idx, feet_idx);
    let weights = com_weights(clip);
    let ground = tracks
        .feet
        .iter()
        .flat_map(|f| f.iter().map(|p| p.z))
        .fold(f64::INFINITY, f64::min);
    let mut worst = f64::NEG_INFINITY;
    for f in 0..clip.frames.len() {
        let supporters: Vec<Vec3> = tracks
            .feet
            .iter()
            .filter(|foot| foot[f].z <= ground + params.contact_margin)
            .map(|foot| foot[f])
            .collect();
        if supporters.is_empty() {
            continue;
        }
        let center = supporters
            .iter()
            .fold(Vec3::ZERO, |a, p| a + *p)
            .scale(1.0 / supporters.len() as f64);
        let radius = supporters
            .iter()
            .map(|p| ((p.x - center.x).powi(2) + (p.y - center.y).powi(2)).sqrt())
            .fold(0.0, f64::max)
            + params.foot_radius;
        let com = center_of_mass(&weights, &tracks.heads[f]);
        let dist = ((com.x - center.x).powi(2) + (com.y - center.y).powi(2)).sqrt();
        worst = worst.max(dist - radius);
    }
    if worst == f64::NEG_INFINITY {
        0.0
    } else {
        worst
    }
}

/// Balance fix (mini-IK): lean the whole body about the ground-level
/// support center to bring the COM back over the feet, then
/// counter-rotate the ankles so the feet stay flat. Each frame leans
/// toward its PINNED pre-fix support: without pinning, the lean lifts
/// one foot, the support set flips mid-pass, and the iterations chase
/// a moving target. The lean targets the margin edge (not the support
/// edge), so the correction ramps to zero at violation boundaries by
/// construction — no extra strength term, no Zeno stall. Total lean
/// per frame is capped twice: by the user budget, and by a support
/// cap that keeps the feet symmetric about the ground within the
/// contact margin (a bigger lean would float one foot past the margin
/// AND drag the ground down with the sunk one, collapsing support —
/// the final honest check would read the "fix" as worse). Big
/// violations get an honest partial fix; the residual stays reported.
/// Returns touched frames.
fn balance_fix_pass(
    clip: &mut Clip,
    params: &PhysicsParams,
    root_idx: usize,
    feet_idx: &[usize],
) -> usize {
    let weights = com_weights(clip);
    let budget = params.max_lean_deg.to_radians();
    let root_head = clip.skeleton.bones[root_idx].head;
    let root_world = clip.skeleton.rest_world(root_idx);
    // Ground and per-frame supports are clip properties: fixing one
    // frame must not un-ground the others, so they are pinned from the
    // pre-fix clip for the whole pass.
    let pre_tracks = build_tracks(clip, root_idx, feet_idx);
    let ground = pre_tracks
        .feet
        .iter()
        .flat_map(|ft| ft.iter().map(|p| p.z))
        .fold(f64::INFINITY, f64::min);
    // Pinned per frame: support (center, radius) plus the pre-fix foot
    // height spread and max pivot distance that bound the support cap.
    let mut pinned: Vec<Option<(Vec3, f64, f64, f64)>> = Vec::with_capacity(clip.frames.len());
    for f in 0..clip.frames.len() {
        let supporters: Vec<Vec3> = pre_tracks
            .feet
            .iter()
            .filter(|foot| foot[f].z <= ground + params.contact_margin)
            .map(|foot| foot[f])
            .collect();
        if supporters.is_empty() {
            pinned.push(None);
            continue;
        }
        let center = supporters
            .iter()
            .fold(Vec3::ZERO, |a, p| a + *p)
            .scale(1.0 / supporters.len() as f64);
        let radius = supporters
            .iter()
            .map(|p| ((p.x - center.x).powi(2) + (p.y - center.y).powi(2)).sqrt())
            .fold(0.0, f64::max)
            + params.foot_radius;
        let lo = supporters.iter().map(|p| p.z).fold(f64::INFINITY, f64::min);
        let hi = supporters
            .iter()
            .map(|p| p.z)
            .fold(f64::NEG_INFINITY, f64::max);
        let d_max = supporters
            .iter()
            .map(|p| ((p.x - center.x).powi(2) + (p.y - center.y).powi(2)).sqrt())
            .fold(0.0, f64::max);
        pinned.push(Some((center, radius, hi - lo, d_max)));
    }
    let to_local = root_world.transpose();
    let mut touched = 0usize;
    for f in 0..clip.frames.len() {
        let Some((center, radius, spread, d_max)) = pinned[f] else {
            continue;
        };
        let pivot = Vec3::new(center.x, center.y, ground);
        // Support cap: the lean leaves feet at +/-(d*sin); both stay
        // supporters while 2*d*sin <= margin - spread (0.8 safety).
        // A foot already at the pivot (single support) barely moves.
        let support_cap = if spread >= params.contact_margin {
            0.0
        } else if d_max < 1e-9 {
            f64::INFINITY
        } else {
            (((params.contact_margin - spread) / 2.0 * 0.8) / d_max)
                .min(1.0)
                .asin()
        };
        let mut applied = false;
        let mut spent = 0.0;
        let mut spent_support = 0.0;
        for _ in 0..3 {
            let tracks = build_tracks(clip, root_idx, feet_idx);
            let com = center_of_mass(&weights, &tracks.heads[f]);
            let dx = center.x - com.x;
            let dy = center.y - com.y;
            let dist = (dx * dx + dy * dy).sqrt();
            let excursion = dist - radius;
            if excursion <= params.balance_margin {
                break;
            }
            let h = com.z - ground;
            if h < 0.1 || dist < 1e-9 {
                break;
            }
            // Lean just enough to reach the margin edge.
            let need = (excursion - params.balance_margin) / h;
            let theta = need.min(budget - spent).min(support_cap - spent_support);
            if theta <= 0.0 {
                break;
            }
            spent += theta;
            spent_support += theta;
            let ux = dx / dist;
            let uy = dy / dist;
            // Positive rotation about up x u moves the COM along +u.
            let axis = Vec3::new(-uy, ux, 0.0);
            let lean = Quat::from_axis_angle(axis, theta).to_mat3();

            let posed = fk(&clip.skeleton, &clip.frames[f]);
            let old_foot_rot: Vec<Mat3> = feet_idx.iter().map(|i| posed[*i].rot).collect();
            let r_new = lean.mul_mat(posed[root_idx].rot);
            let t_new = pivot + lean.mul_vec(posed[root_idx].head - pivot);
            clip.frames[f].poses[root_idx] = crate::clip::Pose {
                loc: to_local.mul_vec(t_new - root_head),
                quat: Quat::from_mat3(to_local.mul_mat(r_new)),
            };
            // Ankle compensation: preserve each foot's world orientation
            // under the new parent chain (toe children follow the foot).
            let posed2 = fk(&clip.skeleton, &clip.frames[f]);
            for (fi, idx) in feet_idx.iter().enumerate() {
                let (rrel, _) = clip.skeleton.rest_parent_rel(*idx);
                let p_new = match clip.skeleton.bones[*idx].parent {
                    None => Mat3::identity(),
                    Some(p) => posed2[p].rot,
                };
                let local = rrel
                    .transpose()
                    .mul_mat(p_new.transpose())
                    .mul_mat(old_foot_rot[fi]);
                clip.frames[f].poses[*idx].quat = Quat::from_mat3(local);
            }
            applied = true;
        }
        if applied {
            // No contact lift: the lean leaves the feet symmetric about
            // the ground (+/-dz), which is already optimal — lifting the
            // sunk foot would double the risen one and break support.
            // Residual penetration (<= spread * sin(cap), mm-scale for
            // realistic fixes) is the documented cost; full leg IK is v2.
            touched += 1;
        }
    }
    touched
}

/// (accel flags, turn flags) for a root track.
fn momentum_flags(root: &[Vec3], fps: f64, params: &PhysicsParams) -> (Vec<usize>, Vec<usize>) {
    let n = root.len();
    if n < 3 {
        return (vec![], vec![]);
    }
    // Horizontal velocities per frame-start + accel per interior frame.
    let mut vel = Vec::with_capacity(n - 1);
    for w in root.windows(2) {
        vel.push(((w[1].x - w[0].x) * fps, (w[1].y - w[0].y) * fps));
    }
    let mut accel_flags = Vec::new();
    for (i, w) in vel.windows(2).enumerate() {
        let ax = (w[1].0 - w[0].0) * fps;
        let ay = (w[1].1 - w[0].1) * fps;
        if (ax * ax + ay * ay).sqrt() > params.accel_limit {
            accel_flags.push(i + 1);
        }
    }
    let mut turn_flags = Vec::new();
    for (i, w) in vel.windows(2).enumerate() {
        let s0 = (w[0].0 * w[0].0 + w[0].1 * w[0].1).sqrt();
        let s1 = (w[1].0 * w[1].0 + w[1].1 * w[1].1).sqrt();
        if s0 < params.min_turn_speed || s1 < params.min_turn_speed {
            continue;
        }
        let dot = (w[0].0 * w[1].0 + w[0].1 * w[1].1) / (s0 * s1);
        let deg = dot.clamp(-1.0, 1.0).acos().to_degrees();
        if deg > params.turn_deg {
            turn_flags.push(i + 1);
        }
    }
    (accel_flags, turn_flags)
}

fn physics_check(
    clip: &Clip,
    params: &PhysicsParams,
    fixed_ballistic: usize,
    fixed_momentum_frames: usize,
) -> Result<PhysicsReport, String> {
    let root_idx = clip.skeleton.index(&params.root).unwrap();
    let feet_idx: Vec<usize> = params
        .feet
        .iter()
        .map(|n| clip.skeleton.index(n).unwrap())
        .collect();
    let tracks = build_tracks(clip, root_idx, &feet_idx);
    let weights = com_weights(clip);
    let ground = tracks
        .feet
        .iter()
        .flat_map(|f| f.iter().map(|p| p.z))
        .fold(f64::INFINITY, f64::min);

    let mut report = PhysicsReport {
        fixed_ballistic,
        fixed_momentum_frames,
        ..Default::default()
    };

    // Balance over contact frames.
    let mut excursions = Vec::new();
    for f in 0..clip.frames.len() {
        let supporters: Vec<Vec3> = tracks
            .feet
            .iter()
            .filter(|foot| foot[f].z <= ground + params.contact_margin)
            .map(|foot| foot[f])
            .collect();
        if supporters.is_empty() {
            report.airborne_frames += 1;
            continue;
        }
        report.contact_frames += 1;
        let center = supporters
            .iter()
            .fold(Vec3::ZERO, |a, p| a + *p)
            .scale(1.0 / supporters.len() as f64);
        let radius = supporters
            .iter()
            .map(|p| ((p.x - center.x).powi(2) + (p.y - center.y).powi(2)).sqrt())
            .fold(0.0, f64::max)
            + params.foot_radius;
        let com = center_of_mass(&weights, &tracks.heads[f]);
        let dist = ((com.x - center.x).powi(2) + (com.y - center.y).powi(2)).sqrt();
        let excursion = dist - radius;
        excursions.push(excursion);
        if excursion > params.balance_margin {
            report.balance_violations.push(BalanceFinding {
                frame: f,
                excursion_m: excursion,
                nudge: (center.x - com.x, center.y - com.y),
            });
        }
    }
    report.mean_excursion_m = if excursions.is_empty() {
        0.0
    } else {
        excursions.iter().sum::<f64>() / excursions.len() as f64
    };

    // Ballistic phases.
    let mut f = 0;
    while f < clip.frames.len() {
        let contact = tracks
            .feet
            .iter()
            .any(|foot| foot[f].z <= ground + params.contact_margin);
        if contact {
            f += 1;
            continue;
        }
        let mut e = f;
        while e + 1 < clip.frames.len()
            && !tracks
                .feet
                .iter()
                .any(|foot| foot[e + 1].z <= ground + params.contact_margin)
        {
            e += 1;
        }
        if e + 1 - f >= params.min_air_frames {
            let t: Vec<f64> = (f..=e).map(|k| (k - f) as f64 / clip.fps).collect();
            let z: Vec<f64> = (f..=e).map(|k| tracks.root[k].z).collect();
            let residual = fit_parabola(&t, &z)
                .map(|(a, b, c)| {
                    t.iter()
                        .zip(z.iter())
                        .map(|(ti, zi)| (a * ti * ti + b * ti + c - zi).abs())
                        .fold(0.0, f64::max)
                })
                .unwrap_or(f64::INFINITY);
            report.ballistic_phases.push(BallisticPhase {
                start: f,
                end: e,
                residual_m: residual,
            });
        }
        f = e + 1;
    }

    // Momentum.
    let (accel_flags, turn_flags) = momentum_flags(&tracks.root, clip.fps, params);
    // Maxima for the report.
    let mut max_accel: f64 = 0.0;
    let mut max_turn: f64 = 0.0;
    let mut vel = Vec::new();
    for w in tracks.root.windows(2) {
        vel.push(((w[1].x - w[0].x) * clip.fps, (w[1].y - w[0].y) * clip.fps));
    }
    for w in vel.windows(2) {
        let ax = (w[1].0 - w[0].0) * clip.fps;
        let ay = (w[1].1 - w[0].1) * clip.fps;
        max_accel = max_accel.max((ax * ax + ay * ay).sqrt());
        let s0 = (w[0].0 * w[0].0 + w[0].1 * w[0].1).sqrt();
        let s1 = (w[1].0 * w[1].0 + w[1].1 * w[1].1).sqrt();
        if s0 >= params.min_turn_speed && s1 >= params.min_turn_speed {
            let dot = (w[0].0 * w[1].0 + w[0].1 * w[1].1) / (s0 * s1);
            max_turn = max_turn.max(dot.clamp(-1.0, 1.0).acos().to_degrees());
        }
    }
    report.accel_flags = accel_flags;
    report.turn_flags = turn_flags;
    report.max_accel_m_s2 = max_accel;
    report.max_turn_deg = max_turn;

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip::{Bone, Clip, Frame, Pose, Skeleton};
    use crate::math::Quat;

    fn stance_skeleton() -> Skeleton {
        // All bones run along +Y (identity rest bases), so local loc
        // channels equal world offsets and the tests stay readable. The
        // local/world conversion itself is covered by clip + retarget.
        Skeleton {
            bones: vec![
                Bone {
                    name: "Hips".to_string(),
                    parent: None,
                    head: Vec3::new(0.0, 0.0, 1.0),
                    tail: Vec3::new(0.0, 1.0, 1.0),
                },
                Bone {
                    name: "Foot.L".to_string(),
                    parent: Some(0),
                    head: Vec3::new(-0.2, 0.0, 0.0),
                    tail: Vec3::new(-0.2, 0.3, 0.0),
                },
                Bone {
                    name: "Foot.R".to_string(),
                    parent: Some(0),
                    head: Vec3::new(0.2, 0.0, 0.0),
                    tail: Vec3::new(0.2, 0.3, 0.0),
                },
            ],
        }
    }

    fn params() -> PhysicsParams {
        PhysicsParams {
            root: "Hips".to_string(),
            feet: vec!["Foot.L".to_string(), "Foot.R".to_string()],
            ..Default::default()
        }
    }

    fn still_clip(frames: usize, root_x: f64) -> Clip {
        let skeleton = stance_skeleton();
        // Root loc +x shifts the whole body (feet included).
        let frame = Frame {
            poses: vec![
                Pose {
                    loc: Vec3::new(root_x, 0.0, 0.0),
                    quat: Quat::IDENTITY,
                },
                Pose {
                    loc: Vec3::ZERO,
                    quat: Quat::IDENTITY,
                },
                Pose {
                    loc: Vec3::ZERO,
                    quat: Quat::IDENTITY,
                },
            ],
        };
        Clip {
            fps: 30.0,
            skeleton,
            frames: vec![frame; frames],
        }
    }

    #[test]
    fn balanced_stance_passes() {
        let clip = still_clip(5, 0.0);
        let mut p = params();
        p.fix_ballistic = false;
        p.fix_momentum = false;
        p.fix_balance = false;
        let (_, report) = physics_fix(&clip, &p).unwrap();
        assert_eq!(report.contact_frames, 5);
        assert!(report.balance_violations.is_empty());
    }

    #[test]
    fn leaning_pose_flags_balance() {
        // Lean the torso (a non-root bone) so COM leaves the support.
        let mut clip = still_clip(5, 0.0);
        // Cant the feet apart asymmetrically is rigid; instead move COM by
        // posing... v1 clips move the whole body with the root, so lean by
        // adding a heavy head bone offset via a 4th bone.
        clip.skeleton.bones.push(Bone {
            name: "Head".to_string(),
            parent: Some(0),
            head: Vec3::new(0.0, 0.0, 1.6),
            tail: Vec3::new(0.0, 1.0, 1.6),
        });
        for fr in clip.frames.iter_mut() {
            fr.poses.push(Pose {
                loc: Vec3::new(4.0, 0.0, 0.0),
                quat: Quat::IDENTITY,
            });
        }
        let mut p = params();
        p.fix_ballistic = false;
        p.fix_momentum = false;
        p.fix_balance = false;
        let (_, report) = physics_fix(&clip, &p).unwrap();
        assert_eq!(report.balance_violations.len(), 5);
        assert!(report.balance_violations[0].excursion_m > 0.05);
        // Suggested nudge points back toward the support (-x).
        assert!(report.balance_violations[0].nudge.0 < 0.0);
    }

    #[test]
    fn frame_snapshot_contact_and_air() {
        let skeleton = stance_skeleton();
        // Frame 0 grounded, frame 1 airborne (both feet +1 z).
        let mk = |lift: f64| Frame {
            poses: vec![
                Pose {
                    loc: Vec3::ZERO,
                    quat: Quat::IDENTITY,
                },
                Pose {
                    loc: Vec3::new(0.0, 0.0, lift),
                    quat: Quat::IDENTITY,
                },
                Pose {
                    loc: Vec3::new(0.0, 0.0, lift),
                    quat: Quat::IDENTITY,
                },
            ],
        };
        let clip = Clip {
            fps: 30.0,
            skeleton,
            frames: vec![mk(0.0), mk(1.0)],
        };
        let p = params();
        let grounded = frame_physics(&clip, &p, 0).unwrap();
        assert!(!grounded.airborne);
        assert_eq!(grounded.supporters.len(), 2);
        assert!(grounded.balanced);
        assert!(grounded.support_center.is_some());
        let air = frame_physics(&clip, &p, 1).unwrap();
        assert!(air.airborne);
        assert!(air.supporters.is_empty());
        assert!(air.support_center.is_none());
        assert!(frame_physics(&clip, &p, 2).is_err());
        // Emission parses as JSON with the documented envelope.
        let text = emit_frame_physics(&grounded).unwrap();
        let doc = crate::json::parse(&text).unwrap();
        assert_eq!(
            doc.get("format").unwrap().as_str().unwrap(),
            "motionforge-frame-physics"
        );
        assert_eq!(doc.get("balanced").unwrap().as_bool().unwrap(), true);
    }

    fn leaning_clip(frames: usize, head_x: f64) -> Clip {
        let mut clip = still_clip(frames, 0.0);
        clip.skeleton.bones.push(Bone {
            name: "Head".to_string(),
            parent: Some(0),
            head: Vec3::new(0.0, 0.0, 1.6),
            tail: Vec3::new(0.0, 1.0, 1.6),
        });
        for fr in clip.frames.iter_mut() {
            fr.poses.push(Pose {
                loc: Vec3::new(head_x, 0.0, 0.0),
                quat: Quat::IDENTITY,
            });
        }
        clip
    }

    #[test]
    fn balance_fix_clears_small_violations() {
        // Excursion ~0.09 (needs ~5 deg): fully fixed, feet stay down.
        let clip = leaning_clip(5, 2.5);
        let mut p = params();
        p.fix_ballistic = false;
        p.fix_momentum = false;
        p.fix_balance = false;
        let (_, before) = physics_fix(&clip, &p).unwrap();
        assert_eq!(before.balance_violations.len(), 5);
        p.fix_balance = true;
        let (fixed, after) = physics_fix(&clip, &p).unwrap();
        assert_eq!(after.fixed_balance_frames, 5);
        assert!(after.balance_violations.is_empty());
        assert!(after.balance_worst_after_m <= p.balance_margin);
        // Feet still planted: both within contact margin, penetration
        // bounded (no lift — the lean leaves them +/-dz by design).
        for fr in &fixed.frames {
            let posed = fk(&fixed.skeleton, fr);
            for fi in [1, 2] {
                assert!(posed[fi].head.z >= -0.025, "foot {}", fi);
                assert!(posed[fi].head.z <= p.contact_margin + 1e-9, "foot {}", fi);
            }
        }
    }

    #[test]
    fn balance_fix_partially_fixes_big_violations() {
        // Excursion ~0.33 needs ~20 deg; the 8 deg total cap allows an
        // honest partial fix, and the residual stays reported.
        let clip = leaning_clip(5, 4.0);
        let mut p = params();
        p.fix_ballistic = false;
        p.fix_momentum = false;
        p.fix_balance = true;
        let (fixed, after) = physics_fix(&clip, &p).unwrap();
        assert_eq!(after.fixed_balance_frames, 5);
        assert!(after.balance_worst_after_m < after.balance_worst_before_m);
        assert!(!after.balance_violations.is_empty());
        for fr in &fixed.frames {
            let lean = Quat::IDENTITY.angle_to(fr.poses[0].quat);
            assert!(lean <= 8.0f64.to_radians() + 1e-9);
        }
    }

    #[test]
    fn balance_fix_is_noop_when_balanced() {
        let clip = still_clip(5, 0.0);
        let mut p = params();
        p.fix_ballistic = false;
        p.fix_momentum = false;
        p.fix_balance = false;
        p.fix_balance = true;
        let (fixed, report) = physics_fix(&clip, &p).unwrap();
        assert_eq!(report.fixed_balance_frames, 0);
        assert_eq!(
            crate::clip::emit_clip(&fixed).unwrap(),
            crate::clip::emit_clip(&clip).unwrap()
        );
    }

    #[test]
    fn balance_fix_preserves_foot_orientation_and_caps_lean() {
        let clip = leaning_clip(3, 4.0);
        let before_rots: Vec<crate::math::Mat3> = clip
            .frames
            .iter()
            .map(|fr| fk(&clip.skeleton, fr)[1].rot)
            .collect();
        let mut p = params();
        p.fix_ballistic = false;
        p.fix_momentum = false;
        p.fix_balance = false;
        p.fix_balance = true;
        let (fixed, _) = physics_fix(&clip, &p).unwrap();
        for (fi, fr) in fixed.frames.iter().enumerate() {
            let after = fk(&fixed.skeleton, fr)[1].rot;
            assert!(after.approx_eq(before_rots[fi], 1e-9), "frame {}", fi);
        }
    }

    #[test]
    fn balance_fix_leaves_clear_frames_bit_exact() {
        // Violation on frames 2-3 only: neighbors must be untouched
        // (need <= 0 exactly at/below the margin).
        let mut clip = leaning_clip(6, 0.0);
        for fr in clip.frames.iter_mut().skip(2).take(2) {
            fr.poses[3].loc = Vec3::new(4.0, 0.0, 0.0);
        }
        let mut p = params();
        p.fix_ballistic = false;
        p.fix_momentum = false;
        p.fix_balance = false;
        p.fix_balance = true;
        let (fixed, report) = physics_fix(&clip, &p).unwrap();
        assert_eq!(report.fixed_balance_frames, 2);
        for fi in [0, 1, 4, 5] {
            for (bi, pose) in fixed.frames[fi].poses.iter().enumerate() {
                let want = &clip.frames[fi].poses[bi];
                assert!(
                    pose.loc.approx_eq(want.loc, 0.0),
                    "frame {} bone {}",
                    fi,
                    bi
                );
                assert!(
                    pose.quat.approx_eq(want.quat, 0.0),
                    "frame {} bone {}",
                    fi,
                    bi
                );
            }
        }
    }

    #[test]
    fn com_weights_sum_to_one() {
        let clip = still_clip(1, 0.0);
        let w = com_weights(&clip);
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        // Hips outweigh a foot.
        assert!(w[0] > w[1]);
    }

    #[test]
    fn ballistic_parabola_passes_and_bump_is_fixed() {
        // Frames 0 and 7 grounded; 1..=6 airborne on a parabola.
        let skeleton = stance_skeleton();
        let mut frames = Vec::new();
        for i in 0..8 {
            let air = [1, 2, 3, 4, 5, 6].contains(&i);
            let t = i as f64 / 30.0;
            // Grounded frames sit at rest height (feet are children of the
            // root, so a lifted root would lift the "grounded" feet too).
            let mut z = if air {
                1.0 + 2.0 * t - 4.9 * t * t
            } else {
                1.0
            };
            if i == 4 {
                z += 0.2; // the bump
            }
            frames.push(Frame {
                poses: vec![
                    Pose {
                        loc: Vec3::new(0.0, 0.0, z - 1.0),
                        quat: Quat::IDENTITY,
                    },
                    Pose {
                        loc: Vec3::new(0.0, 0.0, if air { 1.0 } else { 0.0 }),
                        quat: Quat::IDENTITY,
                    },
                    Pose {
                        loc: Vec3::ZERO,
                        quat: Quat::IDENTITY,
                    },
                ],
            });
        }
        // Right foot mirrors left (airborne too).
        for (i, fr) in frames.iter_mut().enumerate() {
            let air = [1, 2, 3, 4, 5, 6].contains(&i);
            fr.poses[2].loc = Vec3::new(0.0, 0.0, if air { 1.0 } else { 0.0 });
        }
        let clip = Clip {
            fps: 30.0,
            skeleton,
            frames,
        };
        let mut p = params();
        p.fix_ballistic = false;
        p.fix_momentum = false;
        p.fix_balance = false;
        let (_, before) = physics_fix(&clip, &p).unwrap();
        assert_eq!(before.ballistic_phases.len(), 1);
        assert!(before.ballistic_phases[0].residual_m > 0.05);
        p.fix_ballistic = true;
        let (fixed, after) = physics_fix(&clip, &p).unwrap();
        assert_eq!(after.fixed_ballistic, 1);
        // Residual collapses (blended fix keeps boundary frames exact,
        // which sit on the true parabola here).
        assert!(after.ballistic_phases[0].residual_m < before.ballistic_phases[0].residual_m * 0.3);
        assert!(after.ballistic_phases[0].residual_m < 0.02);
        // Boundary frames untouched (blended fix).
        assert!(fixed.frames[1].poses[0]
            .loc
            .approx_eq(clip.frames[1].poses[0].loc, 1e-12));
        assert!(fixed.frames[6].poses[0]
            .loc
            .approx_eq(clip.frames[6].poses[0].loc, 1e-12));
    }

    #[test]
    fn teleport_flagged_and_smoothed() {
        let skeleton = stance_skeleton();
        let mut frames = Vec::new();
        for i in 0..10 {
            let x = if i == 5 { 1.0 } else { i as f64 * 0.01 };
            frames.push(Frame {
                poses: vec![
                    Pose {
                        loc: Vec3::new(x, 0.0, 0.0),
                        quat: Quat::IDENTITY,
                    },
                    Pose {
                        loc: Vec3::ZERO,
                        quat: Quat::IDENTITY,
                    },
                    Pose {
                        loc: Vec3::ZERO,
                        quat: Quat::IDENTITY,
                    },
                ],
            });
        }
        let clip = Clip {
            fps: 30.0,
            skeleton,
            frames,
        };
        let mut p = params();
        p.fix_ballistic = false;
        p.fix_momentum = false;
        p.fix_balance = false;
        let (_, before) = physics_fix(&clip, &p).unwrap();
        assert!(!before.accel_flags.is_empty());
        let max_before = before.max_accel_m_s2;
        p.fix_momentum = true;
        let (_, after) = physics_fix(&clip, &p).unwrap();
        assert!(after.fixed_momentum_frames > 0);
        assert!(after.max_accel_m_s2 < max_before);
    }

    #[test]
    fn sharp_turn_flagged() {
        let skeleton = stance_skeleton();
        let mut frames = Vec::new();
        for i in 0..8 {
            // Fast +x, then instant +y: a 90-degree corner at frame 4.
            let (x, y) = if i < 4 {
                (i as f64 * 0.1, 0.0)
            } else {
                (0.3, (i as f64 - 3.0) * 0.1)
            };
            frames.push(Frame {
                poses: vec![
                    Pose {
                        loc: Vec3::new(x, y, 0.0),
                        quat: Quat::IDENTITY,
                    },
                    Pose {
                        loc: Vec3::ZERO,
                        quat: Quat::IDENTITY,
                    },
                    Pose {
                        loc: Vec3::ZERO,
                        quat: Quat::IDENTITY,
                    },
                ],
            });
        }
        let clip = Clip {
            fps: 30.0,
            skeleton,
            frames,
        };
        let mut p = params();
        p.fix_ballistic = false;
        p.fix_momentum = false;
        p.fix_balance = false;
        let (_, report) = physics_fix(&clip, &p).unwrap();
        assert!(report.turn_flags.contains(&3));
        assert!(report.max_turn_deg > 80.0);
    }
}
