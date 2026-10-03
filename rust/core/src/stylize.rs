//! Motion stylizer: realistic clip in, snappy game motion out.
//!
//! Pure curve math, no training. Per bone, independently:
//! 1. Find extreme poses: local maxima of the joint-angle-from-rest
//!    signal (raw signal; ripples lose to real extremes in the greedy
//!    spacing pass), keeping the strongest peaks at least `min_spacing`
//!    apart plus any endpoint above threshold. This is the key
//!    reduction: dense frames -> sparse keys.
//! 2. Exaggerate: push each extreme rotation away from neutral,
//!    `slerp(identity, q, factor)` with a per-chain factor.
//! 3. Anticipation: before each extreme with room, insert a counter-key
//!    `anticipation_frames` earlier at `slerp(prev, extreme, -amount)`.
//! 4. Retime: resample every frame between keys with holds (first `hold`
//!    frames of each segment freeze) and snappy easing (ease-out cubic,
//!    or ease-out-back scaled by `overshoot` for settle wobble).
//!
//! Steps 2-4 act on rotations only. Locations carry root travel and
//! pass through unchanged on every frame: retiming them stretched a
//! walk's whole travel over one eased segment (the root sprinted ahead,
//! overshot, then walked backwards for a third of the clip).
//!
//! Output is a dense clip (same fps, same frame count) with editable
//! baked keys; the report lists the key frames each bone kept.

use crate::clip::{Clip, Frame, Pose};
use crate::math::Quat;

#[derive(Clone, Debug)]
pub struct StylizeParams {
    /// Push extremes away from neutral (1.0 = none). Range 0..=4.
    pub exaggeration: f64,
    /// (bone-name substring, factor) rules in priority order; the first
    /// substring matching a bone wins, else `exaggeration`.
    pub chain_factors: Vec<(String, f64)>,
    /// Minimum joint angle (rad) from rest to count as an extreme.
    pub angle_threshold: f64,
    /// Minimum frames between kept interior extremes.
    pub min_spacing: usize,
    /// Frames frozen at each segment's start key.
    pub hold: usize,
    /// Anticipation counter-pose strength (0 = off).
    pub anticipation: f64,
    /// Frames before the extreme to place the counter-key.
    pub anticipation_frames: usize,
    /// Ease-out-back strength, 0 = pure ease-out cubic.
    pub overshoot: f64,
}

impl Default for StylizeParams {
    fn default() -> StylizeParams {
        StylizeParams {
            exaggeration: 1.35,
            chain_factors: Vec::new(),
            angle_threshold: 0.15,
            min_spacing: 4,
            hold: 2,
            anticipation: 0.25,
            anticipation_frames: 3,
            overshoot: 0.6,
        }
    }
}

impl StylizeParams {
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..=4.0).contains(&self.exaggeration) {
            return Err("exaggeration must be in 0..=4".to_string());
        }
        for (name, f) in &self.chain_factors {
            if name.is_empty() || !(0.0..=4.0).contains(f) {
                return Err(format!("bad chain factor {:?} (want substr:0..=4)", name));
            }
        }
        if self.angle_threshold < 0.0 {
            return Err("angle-threshold must be >= 0".to_string());
        }
        if self.min_spacing < 1 {
            return Err("min-spacing must be >= 1".to_string());
        }
        if self.anticipation < 0.0 || self.anticipation > 1.0 {
            return Err("anticipation must be in 0..=1".to_string());
        }
        if self.anticipation_frames < 1 {
            return Err("anticipation-frames must be >= 1".to_string());
        }
        if self.overshoot < 0.0 || self.overshoot > 2.0 {
            return Err("overshoot must be in 0..=2".to_string());
        }
        Ok(())
    }

    fn factor_for(&self, bone: &str) -> f64 {
        for (substr, f) in &self.chain_factors {
            if bone.contains(substr) {
                return *f;
            }
        }
        self.exaggeration
    }
}

#[derive(Clone, Debug, Default)]
pub struct StylizeReport {
    /// Union of kept key frames over all bones, ascending.
    pub keys_union: Vec<usize>,
    /// (bone name, key count) for bones with more than the 2 endpoints.
    pub busy_bones: Vec<(String, usize)>,
    /// Per-bone key frames (skeleton order) for the keys sidecar.
    pub bone_keys: Vec<Vec<usize>>,
    pub counters_added: usize,
    pub counters_skipped: usize,
}

/// Emit the `motionforge-keys` sidecar: per-bone key frames for sparse
/// (thinned) import. Deterministic: skeleton bone order, ascending frames.
pub fn emit_keys(skeleton: &crate::clip::Skeleton, report: &StylizeReport) -> String {
    use crate::json::push_str;
    let mut out = String::new();
    out.push_str("{\"format\": \"motionforge-keys\", \"version\": 1,\n \"bones\": {\n");
    for (bi, bone) in skeleton.bones.iter().enumerate() {
        out.push_str("  ");
        push_str(&mut out, &bone.name);
        out.push_str(": [");
        for (ki, k) in report.bone_keys[bi].iter().enumerate() {
            if ki > 0 {
                out.push_str(", ");
            }
            out.push_str(&k.to_string());
        }
        out.push(']');
        if bi + 1 < skeleton.len() {
            out.push(',');
        }
        out.push('\n');
    }
    out.push_str(" }}\n");
    out
}

/// Ease-out cubic (snappy attack, soft arrival).
fn ease_out(u: f64) -> f64 {
    1.0 - (1.0 - u).powi(3)
}

/// Ease-out-back scaled by `strength` (0 = ease-out cubic).
fn ease_back(u: f64, strength: f64) -> f64 {
    if strength <= 0.0 {
        return ease_out(u);
    }
    let c1 = 1.70158 * strength;
    let c3 = c1 + 1.0;
    1.0 + c3 * (u - 1.0).powi(3) + c1 * (u - 1.0).powi(2)
}

/// Interior extreme frames for one bone's angle signal: local maxima
/// above threshold, greedily strongest-first with `min_spacing`.
fn find_extremes(angles: &[f64], params: &StylizeParams) -> Vec<usize> {
    let n = angles.len();
    let mut candidates: Vec<usize> = (1..n.saturating_sub(1))
        .filter(|f| {
            angles[*f] >= params.angle_threshold
                && angles[*f] > angles[*f - 1]
                && angles[*f] >= angles[*f + 1]
        })
        .collect();
    // Strongest first, ties to the earlier frame: deterministic.
    candidates.sort_by(|a, b| angles[*b].partial_cmp(&angles[*a]).unwrap().then(a.cmp(b)));
    let mut kept = Vec::new();
    for c in candidates {
        if kept
            .iter()
            .all(|k: &usize| k.abs_diff(c) >= params.min_spacing)
        {
            kept.push(c);
        }
    }
    // Endpoints join as extremes on amplitude alone.
    if n > 0 && angles[0] >= params.angle_threshold {
        kept.push(0);
    }
    if n > 1 && angles[n - 1] >= params.angle_threshold {
        kept.push(n - 1);
    }
    kept.sort();
    kept.dedup();
    kept
}

pub fn stylize(clip: &Clip, params: &StylizeParams) -> Result<(Clip, StylizeReport), String> {
    params.validate()?;
    let n = clip.frames.len();
    let nb = clip.skeleton.len();

    // Per-bone angle signals.
    let mut extremes: Vec<Vec<usize>> = Vec::with_capacity(nb);
    for b in 0..nb {
        let angles: Vec<f64> = clip
            .frames
            .iter()
            .map(|fr| Quat::IDENTITY.angle_to(fr.poses[b].quat))
            .collect();
        extremes.push(find_extremes(&angles, params));
    }

    // Exaggerated key poses per bone: extremes pushed from neutral.
    let mut key_pose: Vec<Vec<Option<Pose>>> = vec![vec![None; n]; nb];
    for b in 0..nb {
        let factor = params.factor_for(&clip.skeleton.bones[b].name);
        for f in 0..n {
            let pose = &clip.frames[f].poses[b];
            let q = if extremes[b].contains(&f) {
                Quat::IDENTITY.slerp(pose.quat, factor)
            } else {
                pose.quat
            };
            // Endpoints and extremes are keys (poses fixed here);
            // anticipation counters are added below.
            if f == 0 || f == n - 1 || extremes[b].contains(&f) {
                key_pose[b][f] = Some(Pose {
                    loc: pose.loc,
                    quat: q,
                });
            }
        }
    }

    // Anticipation counters, from the exaggerated key poses.
    let mut report = StylizeReport::default();
    if params.anticipation > 0.0 {
        for b in 0..nb {
            let keys: Vec<usize> = (0..n).filter(|f| key_pose[b][*f].is_some()).collect();
            for e in extremes[b].iter().copied() {
                if e == 0 {
                    continue;
                }
                let prev = *keys.iter().filter(|k| **k < e).max().unwrap();
                let at = e as isize - params.anticipation_frames as isize;
                if at <= prev as isize {
                    report.counters_skipped += 1;
                    continue;
                }
                let at = at as usize;
                let prev_pose = key_pose[b][prev].clone().unwrap();
                let ext_pose = key_pose[b][e].clone().unwrap();
                key_pose[b][at] = Some(Pose {
                    loc: clip.frames[at].poses[b].loc,
                    quat: prev_pose.quat.slerp(ext_pose.quat, -params.anticipation),
                });
                report.counters_added += 1;
            }
        }
    }

    // Resample each bone's rotation over its keys; location is the
    // source's, frame for frame.
    let mut out_frames = Vec::with_capacity(n);
    for f in 0..n {
        let mut poses = Vec::with_capacity(nb);
        for b in 0..nb {
            let loc = clip.frames[f].poses[b].loc;
            // Bounding keys (endpoints always exist, so both are found).
            let mut k0 = 0;
            for k in 0..=f {
                if key_pose[b][k].is_some() {
                    k0 = k;
                }
            }
            let mut k1 = n - 1;
            for k in f..n {
                if key_pose[b][k].is_some() {
                    k1 = k;
                    break;
                }
            }
            let p0 = key_pose[b][k0].clone().unwrap();
            if k0 == k1 {
                poses.push(Pose { loc, quat: p0.quat });
                continue;
            }
            let p1 = key_pose[b][k1].clone().unwrap();
            let span = (k1 - k0) as f64;
            let held = params.hold.min(k1 - k0) as f64;
            let e = if (f - k0) as f64 <= held {
                0.0
            } else if span - held <= 0.0 {
                0.0
            } else {
                let u = ((f - k0) as f64 - held) / (span - held);
                ease_back(u.clamp(0.0, 1.0), params.overshoot)
            };
            poses.push(Pose {
                loc,
                quat: p0.quat.slerp(p1.quat, e),
            });
        }
        out_frames.push(Frame { poses });
    }

    let mut union = Vec::new();
    for b in 0..nb {
        let keys: Vec<usize> = (0..n).filter(|f| key_pose[b][*f].is_some()).collect();
        if keys.len() > 2 {
            report
                .busy_bones
                .push((clip.skeleton.bones[b].name.clone(), keys.len()));
        }
        for f in keys.iter() {
            if !union.contains(f) {
                union.push(*f);
            }
        }
        report.bone_keys.push(keys);
    }
    union.sort();
    report.keys_union = union;

    Ok((
        Clip {
            fps: clip.fps,
            skeleton: clip.skeleton.clone(),
            frames: out_frames,
        },
        report,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip::{Bone, Clip, Frame, Pose, Skeleton};
    use crate::math::Vec3;
    use std::f64::consts::PI;

    fn one_bone() -> Skeleton {
        Skeleton {
            bones: vec![Bone {
                name: "arm.L".to_string(),
                parent: None,
                head: Vec3::ZERO,
                tail: Vec3::new(0.0, 0.0, 1.0),
            }],
        }
    }

    fn swing_clip(angles: &[f64]) -> Clip {
        let skeleton = one_bone();
        let frames = angles
            .iter()
            .map(|a| Frame {
                poses: vec![Pose {
                    loc: Vec3::ZERO,
                    quat: Quat::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), *a),
                }],
            })
            .collect();
        Clip {
            fps: 30.0,
            skeleton,
            frames,
        }
    }

    fn angle_of(clip: &Clip, f: usize) -> f64 {
        Quat::IDENTITY.angle_to(clip.frames[f].poses[0].quat)
    }

    fn plain() -> StylizeParams {
        StylizeParams {
            exaggeration: 1.0,
            chain_factors: vec![],
            angle_threshold: 0.15,
            min_spacing: 1,
            hold: 0,
            anticipation: 0.0,
            anticipation_frames: 3,
            overshoot: 0.0,
        }
    }

    #[test]
    fn two_frame_clip_is_exact() {
        let clip = swing_clip(&[0.2, 0.8]);
        let (out, _) = stylize(&clip, &plain()).unwrap();
        assert!((angle_of(&out, 0) - 0.2).abs() < 1e-12);
        assert!((angle_of(&out, 1) - 0.8).abs() < 1e-12);
    }

    #[test]
    fn exaggeration_pushes_extreme_from_neutral() {
        let clip = swing_clip(&[0.0, 0.5, 0.0]);
        let mut p = plain();
        p.exaggeration = 2.0;
        p.angle_threshold = 0.1;
        let (out, report) = stylize(&clip, &p).unwrap();
        assert_eq!(report.keys_union, vec![0, 1, 2]);
        assert!((angle_of(&out, 1) - 1.0).abs() < 1e-9);
        assert!(angle_of(&out, 0) < 1e-12);
    }

    #[test]
    fn chain_factors_override_per_bone() {
        let clip = swing_clip(&[0.0, 0.5, 0.0]);
        let mut p = plain();
        p.exaggeration = 1.0;
        p.angle_threshold = 0.1;
        p.chain_factors = vec![("arm".to_string(), 3.0)];
        let (out, _) = stylize(&clip, &p).unwrap();
        assert!((angle_of(&out, 1) - 1.5).abs() < 1e-9);
    }

    #[test]
    fn sine_wave_reduces_to_sparse_keys() {
        let angles: Vec<f64> = (0..60)
            .map(|i| 0.6 * (2.0 * PI * i as f64 / 30.0).sin())
            .collect();
        let clip = swing_clip(&angles);
        let mut p = plain();
        p.angle_threshold = 0.2;
        p.min_spacing = 4;
        let (_, report) = stylize(&clip, &p).unwrap();
        // Two peaks + endpoints range; far below the 60 dense frames.
        assert!(
            report.keys_union.len() <= 6,
            "keys: {:?}",
            report.keys_union
        );
        assert_eq!(report.keys_union[0], 0);
        assert_eq!(*report.keys_union.last().unwrap(), 59);
    }

    #[test]
    fn hold_freezes_segment_start() {
        // Interior below threshold: keys {0, 10} only.
        let mut angles = vec![0.05; 11];
        angles[0] = 0.0;
        angles[10] = 0.1;
        let clip = swing_clip(&angles);
        let mut p = plain();
        p.angle_threshold = 0.2;
        p.hold = 3;
        let (out, report) = stylize(&clip, &p).unwrap();
        assert_eq!(report.keys_union, vec![0, 10]);
        for f in 1..=3 {
            assert!(angle_of(&out, f) < 1e-12, "frame {}", f);
        }
        assert!(angle_of(&out, 10) > 0.09);
    }

    #[test]
    fn overshoot_exceeds_target_mid_segment() {
        let mut angles = vec![0.05; 5];
        angles[0] = 0.0;
        angles[4] = 0.1;
        let clip = swing_clip(&angles);
        let mut p = plain();
        p.angle_threshold = 0.2;
        p.overshoot = 1.0;
        let (out, _) = stylize(&clip, &p).unwrap();
        // easeOutBack(0.5) = 1.0875 -> 0.10875 > 0.1.
        assert!((angle_of(&out, 2) - 0.10875).abs() < 1e-4);
        p.overshoot = 0.0;
        let (out2, _) = stylize(&clip, &p).unwrap();
        // easeOutCubic(0.5) = 0.875 -> 0.0875, no overshoot.
        assert!((angle_of(&out2, 2) - 0.0875).abs() < 1e-9);
        assert!(angle_of(&out2, 2) < 0.1);
    }

    #[test]
    fn anticipation_inserts_counter_pose() {
        // Peak at frame 10 with room: counter lands at frame 7.
        let mut angles = vec![0.0; 13];
        for (i, a) in angles.iter_mut().enumerate() {
            *a = 0.5 * (1.0 - ((i as f64 - 10.0) / 10.0).abs()).max(0.0);
        }
        let clip = swing_clip(&angles);
        let mut p = plain();
        p.angle_threshold = 0.2;
        p.min_spacing = 2;
        p.anticipation = 0.25;
        p.anticipation_frames = 3;
        let (out, report) = stylize(&clip, &p).unwrap();
        assert!(report.counters_added >= 1);
        assert!(report.keys_union.contains(&7));
        // Counter angle = 0.25 * peak angle, opposite side (resampled
        // exactly since frame 7 is a key).
        assert!((angle_of(&out, 7) - 0.125).abs() < 1e-9);
    }

    #[test]
    fn anticipation_skips_without_room() {
        let clip = swing_clip(&[0.0, 0.0, 0.9, 0.0, 0.0]);
        let mut p = plain();
        p.angle_threshold = 0.2;
        p.anticipation = 0.5;
        p.anticipation_frames = 3;
        let (_, report) = stylize(&clip, &p).unwrap();
        // Extreme at 2, counter would land at -1: skipped.
        assert_eq!(report.counters_added, 0);
        assert_eq!(report.counters_skipped, 1);
    }

    #[test]
    fn params_rejected() {
        let mut p = plain();
        p.exaggeration = 9.0;
        assert!(p.validate().is_err());
        p = plain();
        p.min_spacing = 0;
        assert!(p.validate().is_err());
    }

    #[test]
    fn keys_sidecar_lists_per_bone_keys() {
        let clip = swing_clip(&[0.0, 0.5, 0.0]);
        let mut p = plain();
        p.exaggeration = 2.0;
        p.angle_threshold = 0.1;
        let (_, report) = stylize(&clip, &p).unwrap();
        assert_eq!(report.bone_keys, vec![vec![0, 1, 2]]);
        let text = emit_keys(&clip.skeleton, &report);
        assert!(text.contains("\"motionforge-keys\""), "{}", text);
        assert!(text.contains("\"arm.L\": [0, 1, 2]"), "{}", text);
    }

    #[test]
    fn locations_pass_through_unchanged() {
        // Steady root travel under a swinging rotation: every rotation
        // stage fires (extremes, anticipation, holds, overshoot), but
        // travel must stay frame-for-frame the source's. Before the fix
        // the walk fixture's root froze, sprinted, overshot its end and
        // walked backwards on 15 of 47 frames.
        let angles: Vec<f64> = (0..40).map(|i| 0.6 * (i as f64 * 0.3).sin()).collect();
        let mut clip = swing_clip(&angles);
        for (i, fr) in clip.frames.iter_mut().enumerate() {
            fr.poses[0].loc = Vec3::new(0.0, 0.0, -0.025 * i as f64);
        }
        let (out, report) = stylize(&clip, &StylizeParams::default()).unwrap();
        assert!(report.counters_added > 0 && report.keys_union.len() > 2);
        for (a, b) in clip.frames.iter().zip(out.frames.iter()) {
            assert_eq!(a.poses[0].loc, b.poses[0].loc);
        }
        // Rotations are still stylized.
        assert!((0..40).any(|f| (angle_of(&out, f) - angles[f].abs()).abs() > 1e-3));
    }

    #[test]
    fn stylize_is_deterministic() {
        let angles: Vec<f64> = (0..40).map(|i| 0.6 * (i as f64 * 0.3).sin()).collect();
        let clip = swing_clip(&angles);
        let p = StylizeParams::default();
        let (a, _) = stylize(&clip, &p).unwrap();
        let (b, _) = stylize(&clip, &p).unwrap();
        assert_eq!(
            crate::clip::emit_clip(&a).unwrap(),
            crate::clip::emit_clip(&b).unwrap()
        );
    }
}
