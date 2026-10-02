//! Retarget assist: transfer a source clip onto a target skeleton.
//!
//! Port of rigforge's `tools/retarget_mixamo.py` rotation-transfer math
//! onto clip JSON, with one deliberate improvement (see below). The
//! Blender extension handles FBX import, action export/import, and GLB
//! export; this module does the frame math deterministically.
//!
//! Transfer, per mapped pair per frame: the source bone's local motion
//! `d` (deviation from source rest) is re-expressed in the target
//! bone's local frame by conjugation through armature space,
//! `d_t = C d C^-1` with `C = R_t^-1 P_t^-1 Q P_s R_s` (`R` =
//! parent-relative rest, `P` = posed parent, `Q` = yaw fix). Rest source
//! frames map to rest target frames, and transfer between identical
//! skeletons is exact. (The reference script instead applies the
//! armature-space motion on top of the target's posed parent chain,
//! `A' (P R)`, which is exact only at the root and first-order
//! elsewhere; the `identity_transfer_is_exact` test pins the
//! improvement.) Root translation transfers scaled by the leg-length
//! ratio (stride scale) onto every driven chain root, since the DEF
//! hierarchy does not hang the limbs under the pelvis.
//!
//! Foot-slide measurement and the pin pass (ramp stance drift out
//! through the chain-root location curves) are faithful ports of the
//! reference: stance = bottom 35% of foot height AND below-median
//! horizontal speed.

use crate::clip::{fk, parse_skeleton, Clip, Frame, Pose, Skeleton};
use crate::json::{parse, Json};
use crate::math::{Mat3, Quat, Vec3};
use std::f64::consts::PI;

pub const BONEMAP_FORMAT: &str = "motionforge-bonemap";

#[derive(Clone, Debug, Default)]
pub struct BoneMap {
    pub pairs: Vec<(String, String)>,
    pub root_source: String,
    pub root_target: String,
    pub stride_source: Option<(String, String)>,
    pub stride_target: Option<(String, String)>,
    pub feet_source: Vec<String>,
    pub feet_target: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum YawMode {
    Auto,
    Flip,
    None,
}

#[derive(Clone, Debug)]
pub struct FootStats {
    pub name: String,
    pub stance_frames: usize,
    pub stance_mean_m_s: f64,
    pub stance_max_m_s: f64,
    pub overall_max_m_s: f64,
}

#[derive(Clone, Debug, Default)]
pub struct RetargetReport {
    pub mapped: usize,
    pub skipped: Vec<String>,
    pub chain_roots: Vec<String>,
    pub stride_scale: f64,
    pub stride_defaulted: bool,
    pub yaw_flip: bool,
    pub yaw_note: Option<String>,
    pub slide_before: Vec<FootStats>,
    pub slide_after: Vec<FootStats>,
    pub pin_intervals: usize,
    pub pin_drift_m: f64,
    pub root_travel_m: f64,
    pub max_swing_rad: f64,
}

pub fn parse_bonemap(text: &str) -> Result<BoneMap, String> {
    let root = parse(text).map_err(|e| format!("bonemap json: {}", e))?;
    let ctx = "invalid motionforge-bonemap";
    if root.get("format").and_then(|v| v.as_str()) != Some(BONEMAP_FORMAT) {
        return Err(format!("{}: bad \"format\"", ctx));
    }
    match root.get("version").and_then(|v| v.as_f64()) {
        Some(1.0) => {}
        _ => return Err(format!("{}: unsupported version", ctx)),
    }
    let get_str = |key: &str| -> Result<String, String> {
        match root.get(key).and_then(|v| v.as_str()) {
            Some(s) => Ok(s.to_string()),
            _ => Err(format!("{}: missing string \"{}\"", ctx, key)),
        }
    };
    let pairs_json = root
        .get("pairs")
        .ok_or_else(|| format!("{}: missing \"pairs\"", ctx))?;
    let pairs_arr = pairs_json
        .as_arr()
        .ok_or_else(|| "bonemap: \"pairs\" must be an array".to_string())?;
    let mut pairs = Vec::with_capacity(pairs_arr.len());
    for (i, item) in pairs_arr.iter().enumerate() {
        let s = item.get("source").and_then(|v| v.as_str());
        let t = item.get("target").and_then(|v| v.as_str());
        match (s, t) {
            (Some(s), Some(t)) => pairs.push((s.to_string(), t.to_string())),
            _ => return Err(format!("bonemap: pair {} needs source/target strings", i)),
        }
    }
    if pairs.is_empty() {
        return Err("bonemap: no pairs".to_string());
    }
    let stride_pair = |key: &str| -> Result<Option<(String, String)>, String> {
        match root.get(key) {
            None => Ok(None),
            Some(v) => {
                let arr = v
                    .as_arr()
                    .ok_or_else(|| format!("bonemap: \"{}\" must be [upper, foot]", key))?;
                if arr.len() != 2 {
                    return Err(format!("bonemap: \"{}\" must be [upper, foot]", key));
                }
                let a = arr[0]
                    .as_str()
                    .ok_or_else(|| format!("bonemap: \"{}\" must be strings", key))?;
                let b = arr[1]
                    .as_str()
                    .ok_or_else(|| format!("bonemap: \"{}\" must be strings", key))?;
                Ok(Some((a.to_string(), b.to_string())))
            }
        }
    };
    let feet_list = |key: &str| -> Result<Vec<String>, String> {
        match root.get(key) {
            None => Ok(vec![]),
            Some(v) => {
                let arr = v
                    .as_arr()
                    .ok_or_else(|| format!("bonemap: \"{}\" must be a string array", key))?;
                let mut out = Vec::with_capacity(arr.len());
                for item in arr {
                    out.push(
                        item.as_str()
                            .ok_or_else(|| format!("bonemap: \"{}\" must be strings", key))?
                            .to_string(),
                    );
                }
                Ok(out)
            }
        }
    };
    Ok(BoneMap {
        pairs,
        root_source: get_str("root_source")?,
        root_target: get_str("root_target")?,
        stride_source: stride_pair("stride_source")?,
        stride_target: stride_pair("stride_target")?,
        feet_source: feet_list("feet_source")?,
        feet_target: feet_list("feet_target")?,
    })
}

/// Parse a target skeleton that may be either a `motionforge-skeleton`
/// document or a full clip (whose skeleton is used).
pub fn parse_target_skeleton(text: &str) -> Result<Skeleton, String> {
    if let Ok(skeleton) = parse_skeleton(text) {
        return Ok(skeleton);
    }
    let root = parse(text).map_err(|e| format!("target json: {}", e))?;
    match root.get("format").and_then(|v| v.as_str()) {
        Some(crate::clip::CLIP_FORMAT) => {
            let clip = crate::clip::parse_clip(text)?;
            Ok(clip.skeleton)
        }
        _ => Err("target must be a motionforge-skeleton or motionforge-clip".to_string()),
    }
}

/// Facing sign from the first Y-dominant foot bone's rest direction:
/// +1 faces +Y, -1 faces -Y, None when no foot bone qualifies.
fn facing_sign(skeleton: &Skeleton, feet: &[String]) -> Option<f64> {
    for name in feet {
        let i = skeleton.index(name)?;
        let v = skeleton.bones[i].tail - skeleton.bones[i].head;
        if v.y.abs() >= v.x.abs() && v.y.abs() >= v.z.abs() && v.y.abs() > 1e-9 {
            return Some(if v.y > 0.0 { 1.0 } else { -1.0 });
        }
    }
    None
}

fn leg_length(skeleton: &Skeleton, upper: &str, foot: &str) -> Result<f64, String> {
    let u = skeleton
        .index(upper)
        .ok_or_else(|| format!("unknown stride bone \"{}\"", upper))?;
    let f = skeleton
        .index(foot)
        .ok_or_else(|| format!("unknown stride bone \"{}\"", foot))?;
    Ok((skeleton.bones[u].head - skeleton.bones[f].head).length())
}

/// Stance frames: bottom 35% of foot height AND below-median horizontal
/// speed. Returns (mask, per-frame-start speeds in m/s).
fn stance_mask(pts: &[Vec3], fps: f64) -> (Vec<bool>, Vec<f64>) {
    let mut speeds = Vec::new();
    for w in pts.windows(2) {
        let dx = w[1].x - w[0].x;
        let dy = w[1].y - w[0].y;
        speeds.push((dx * dx + dy * dy).sqrt() * fps);
    }
    let mut sorted = speeds.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = sorted.get(sorted.len() / 2).copied().unwrap_or(0.0);
    let mut per_frame = speeds.clone();
    per_frame.push(*speeds.last().unwrap_or(&0.0));
    let lo = pts.iter().map(|p| p.z).fold(f64::INFINITY, f64::min);
    let hi = pts.iter().map(|p| p.z).fold(f64::NEG_INFINITY, f64::max);
    let mask = pts
        .iter()
        .zip(per_frame.iter())
        .map(|(p, s)| p.z <= lo + 0.35 * (hi - lo) && *s <= med)
        .collect();
    (mask, speeds)
}

fn measure_slide(skeleton: &Skeleton, clip: &Clip, feet: &[usize]) -> Vec<FootStats> {
    let mut out = Vec::new();
    for (fi, name) in feet.iter().map(|i| &skeleton.bones[*i].name).enumerate() {
        let idx = feet[fi];
        let pts: Vec<Vec3> = clip
            .frames
            .iter()
            .map(|f| fk(&skeleton, f)[idx].head)
            .collect();
        let (mask, speeds) = stance_mask(&pts, clip.fps);
        let st: Vec<f64> = speeds
            .iter()
            .enumerate()
            .filter(|(i, _)| mask[*i])
            .map(|(_, s)| *s)
            .collect();
        out.push(FootStats {
            name: name.clone(),
            stance_frames: mask.iter().filter(|m| **m).count(),
            stance_mean_m_s: if st.is_empty() {
                0.0
            } else {
                st.iter().sum::<f64>() / st.len() as f64
            },
            stance_max_m_s: st.iter().copied().fold(0.0, f64::max),
            overall_max_m_s: speeds.iter().copied().fold(0.0, f64::max),
        });
    }
    out
}

pub struct RetargetPlan {
    /// (target bone, source bone) pairs, target depth order.
    drive: Vec<(usize, usize)>,
    chain_roots: Vec<usize>,
    skipped: Vec<String>,
    root_source: usize,
    _root_target: usize,
}

fn build_plan(source: &Clip, target: &Skeleton, map: &BoneMap) -> Result<RetargetPlan, String> {
    let mut drive = Vec::new();
    let mut skipped = Vec::new();
    for (s, t) in &map.pairs {
        match (source.skeleton.index(s), target.index(t)) {
            (Some(si), Some(ti)) => {
                if !drive.iter().any(|(d, _)| *d == ti) {
                    drive.push((ti, si));
                }
            }
            (None, _) => skipped.push(format!("{} (no source bone)", s)),
            (_, None) => skipped.push(format!("{} (no {})", s, t)),
        }
    }
    if drive.is_empty() {
        return Err("correspondence map matched zero bones".to_string());
    }
    let root_source = source
        .skeleton
        .index(&map.root_source)
        .ok_or_else(|| format!("root_source \"{}\" not in source clip", map.root_source))?;
    let root_target = target
        .index(&map.root_target)
        .ok_or_else(|| format!("root_target \"{}\" not in target skeleton", map.root_target))?;
    if !drive.iter().any(|(d, _)| *d == root_target) {
        return Err("root pair did not match; cannot transfer root motion".to_string());
    }
    let driven = |t: usize| drive.iter().any(|(d, _)| *d == t);
    let has_driven_ancestor = |mut t: usize| {
        let mut b = &target.bones[t];
        while let Some(p) = b.parent {
            if driven(p) {
                return true;
            }
            t = p;
            b = &target.bones[t];
        }
        false
    };
    let mut chain_roots: Vec<usize> = drive
        .iter()
        .map(|(d, _)| *d)
        .filter(|t| !has_driven_ancestor(*t))
        .collect();
    chain_roots.sort_by_key(|t| target.depth(*t));
    drive.sort_by_key(|(d, _)| target.depth(*d));
    Ok(RetargetPlan {
        drive,
        chain_roots,
        skipped,
        root_source,
        _root_target: root_target,
    })
}

/// Retarget `source` onto `target`. When `pin` is set, stance drift is
/// ramped out through the chain roots (foot pin pass).
pub fn retarget(
    source: &Clip,
    target: &Skeleton,
    map: &BoneMap,
    yaw: YawMode,
    pin: bool,
) -> Result<(Clip, RetargetReport), String> {
    let plan = build_plan(source, target, map)?;

    let yaw_flip = match yaw {
        YawMode::Flip => true,
        YawMode::None => false,
        YawMode::Auto => match (
            facing_sign(&source.skeleton, &map.feet_source),
            facing_sign(target, &map.feet_target),
        ) {
            (Some(a), Some(b)) => a != b,
            _ => false,
        },
    };
    let yaw_note = match yaw {
        YawMode::Auto
            if facing_sign(&source.skeleton, &map.feet_source).is_none()
                || facing_sign(target, &map.feet_target).is_none() =>
        {
            Some("no feet for facing probe; yaw left unchanged".to_string())
        }
        _ => None,
    };
    let q_fix = if yaw_flip {
        Mat3::rotation_z(PI)
    } else {
        Mat3::identity()
    };

    let (stride_scale, stride_defaulted) = match (&map.stride_source, &map.stride_target) {
        (Some((su, sf)), Some((tu, tf))) => {
            let s = leg_length(&source.skeleton, su, sf)?;
            let t = leg_length(target, tu, tf)?;
            if s < 1e-9 || t < 1e-9 {
                return Err("stride leg has zero length".to_string());
            }
            (t / s, false)
        }
        _ => (1.0, true),
    };

    let src_rest_head = source.skeleton.bones[plan.root_source].head;
    // Target rest snapshots.
    let mut tgt_rel_rot = Vec::with_capacity(target.len());
    let mut tgt_rel_off = Vec::with_capacity(target.len());
    let mut tgt_rest_head = Vec::with_capacity(target.len());
    for i in 0..target.len() {
        let (r, o) = target.rest_parent_rel(i);
        tgt_rel_rot.push(r);
        tgt_rel_off.push(o);
        tgt_rest_head.push(target.bones[i].head);
    }
    let target_parent_rest = |t: usize| -> (Mat3, Vec3) {
        match target.bones[t].parent {
            None => (Mat3::identity(), Vec3::ZERO),
            Some(p) => (target.rest_world(p), target.bones[p].head),
        }
    };
    // Source rest snapshots.
    let mut src_rel_rot = Vec::with_capacity(source.skeleton.len());
    for i in 0..source.skeleton.len() {
        src_rel_rot.push(source.skeleton.rest_parent_rel(i).0);
    }

    let driven_source =
        |t: usize| -> Option<usize> { plan.drive.iter().find(|(d, _)| *d == t).map(|(_, s)| *s) };

    let mut out_frames = Vec::with_capacity(source.frames.len());
    for frame in &source.frames {
        let src_posed = fk(&source.skeleton, frame);
        let src_root_off = q_fix
            .mul_vec(src_posed[plan.root_source].head - src_rest_head)
            .scale(stride_scale);
        // Target bones in depth (= storage) order; `desired` accumulates
        // posed (rot, head) for every bone, driven or not.
        let mut desired_rot = vec![Mat3::identity(); target.len()];
        let mut desired_head = vec![Vec3::ZERO; target.len()];
        let mut poses = vec![
            Pose {
                loc: Vec3::ZERO,
                quat: Quat::IDENTITY
            };
            target.len()
        ];
        for t in 0..target.len() {
            let (p_rot, p_head) = match target.bones[t].parent {
                None => (Mat3::identity(), Vec3::ZERO),
                Some(p) => (desired_rot[p], desired_head[p]),
            };
            let r_rot = tgt_rel_rot[t];
            let r_off = tgt_rel_off[t];
            match driven_source(t) {
                Some(s) => {
                    // C maps source-local vectors to target-local vectors
                    // through armature space (+ yaw fix); d_t = C d C^-1.
                    let (ps_rot, _) = match source.skeleton.bones[s].parent {
                        None => (Mat3::identity(), Vec3::ZERO),
                        Some(p) => (src_posed[p].rot, src_posed[p].head),
                    };
                    let c = r_rot
                        .transpose()
                        .mul_mat(p_rot.transpose())
                        .mul_mat(q_fix)
                        .mul_mat(ps_rot)
                        .mul_mat(src_rel_rot[s]);
                    let d = frame.poses[s].quat.to_mat3();
                    let dt = c.mul_mat(d).mul_mat(c.transpose());
                    let q = Quat::from_mat3(dt);
                    let mut loc = Vec3::ZERO;
                    if plan.chain_roots.contains(&t) {
                        let head_want = tgt_rest_head[t] + src_root_off;
                        let pr = p_rot.mul_vec(r_off) + p_head;
                        loc = p_rot.mul_mat(r_rot).transpose().mul_vec(head_want - pr);
                    }
                    poses[t] = Pose { loc, quat: q };
                }
                None => {
                    // Undriven: identity basis = rigid follow of the posed
                    // parent ( Blender behavior for unkeyed bones).
                    poses[t] = Pose {
                        loc: Vec3::ZERO,
                        quat: Quat::IDENTITY,
                    };
                }
            }
            let b_rot = poses[t].quat.to_mat3();
            desired_rot[t] = p_rot.mul_mat(r_rot).mul_mat(b_rot);
            desired_head[t] = p_rot.mul_vec(r_rot.mul_vec(poses[t].loc) + r_off) + p_head;
        }
        out_frames.push(Frame { poses });
    }

    let mut out = Clip {
        fps: source.fps,
        skeleton: target.clone(),
        frames: out_frames,
    };

    let feet_target: Vec<usize> = map
        .feet_target
        .iter()
        .filter_map(|n| target.index(n))
        .collect();
    let slide_before = measure_slide(target, &out, &feet_target);

    // Pin pass: ramp each stance interval's drift out through the chain
    // roots' location curves.
    let mut pin_intervals = 0usize;
    let mut pin_drift = 0.0f64;
    if pin && !plan.chain_roots.is_empty() && !feet_target.is_empty() && out.frames.len() >= 2 {
        // Chain-root ancestors are all undriven, so rest parent matrices
        // are exact converters from world drift to local loc channels.
        let mut to_local = Vec::new();
        for t in &plan.chain_roots {
            let (pp_rot, _) = target_parent_rest(*t);
            to_local.push(pp_rot.mul_mat(tgt_rel_rot[*t]).transpose());
        }
        for f in &feet_target {
            let pts: Vec<Vec3> = out
                .frames
                .iter()
                .map(|fr| fk(target, fr)[*f].head)
                .collect();
            let (mask, _) = stance_mask(&pts, out.fps);
            let mut i = 0;
            while i < mask.len() {
                if !mask[i] {
                    i += 1;
                    continue;
                }
                let mut j = i;
                while j + 1 < mask.len() && mask[j + 1] {
                    j += 1;
                }
                if j > i {
                    let drift = Vec3::new(pts[j].x - pts[i].x, pts[j].y - pts[i].y, 0.0);
                    pin_drift += drift.length();
                    for k in i..=j {
                        let world = drift.scale((k - i) as f64 / (j - i) as f64);
                        for (ri, t) in plan.chain_roots.iter().enumerate() {
                            let corr = to_local[ri].mul_vec(world);
                            let loc = &mut out.frames[k].poses[*t].loc;
                            *loc = *loc - corr;
                        }
                    }
                    pin_intervals += 1;
                }
                i = j + 1;
            }
        }
    }
    let slide_after = measure_slide(target, &out, &feet_target);

    // Motion proof: root travel (path length) + max joint swing.
    let root_idx = target.index(&map.root_target).unwrap();
    let root_pts: Vec<Vec3> = out
        .frames
        .iter()
        .map(|fr| fk(target, fr)[root_idx].head)
        .collect();
    let mut root_travel = 0.0;
    for w in root_pts.windows(2) {
        root_travel += (w[1] - w[0]).length();
    }
    let mut max_swing: f64 = 0.0;
    for (t, _) in &plan.drive {
        let q0 = out.frames[0].poses[*t].quat;
        for fr in &out.frames {
            max_swing = max_swing.max(q0.angle_to(fr.poses[*t].quat));
        }
    }

    Ok((
        out,
        RetargetReport {
            mapped: plan.drive.len(),
            skipped: plan.skipped,
            chain_roots: plan
                .chain_roots
                .iter()
                .map(|t| target.bones[*t].name.clone())
                .collect(),
            stride_scale,
            stride_defaulted,
            yaw_flip,
            yaw_note,
            slide_before,
            slide_after,
            pin_intervals,
            pin_drift_m: pin_drift,
            root_travel_m: root_travel,
            max_swing_rad: max_swing,
        },
    ))
}

/// Parse a `Json` object that is already known to be a skeleton object
/// (`{"bones": [...]}`), used by the autopose effector format.
pub fn skeleton_from_json(obj: &Json) -> Result<Skeleton, String> {
    crate::clip::parse_skeleton_obj(obj, "skeleton")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip::{emit_clip, parse_clip, Bone, Clip, Frame, Pose, Skeleton};
    use crate::math::{Quat, Vec3};

    fn chain_skeleton() -> Skeleton {
        Skeleton {
            bones: vec![
                Bone {
                    name: "Root".to_string(),
                    parent: None,
                    head: Vec3::new(0.0, 0.0, 1.0),
                    tail: Vec3::new(0.0, 0.0, 1.5),
                },
                Bone {
                    name: "Mid".to_string(),
                    parent: Some(0),
                    head: Vec3::new(0.0, 0.0, 1.5),
                    tail: Vec3::new(0.0, 0.5, 1.5),
                },
                Bone {
                    name: "Tip".to_string(),
                    parent: Some(1),
                    head: Vec3::new(0.0, 0.5, 1.5),
                    tail: Vec3::new(0.0, 1.0, 1.5),
                },
            ],
        }
    }

    fn pose_clip() -> Clip {
        let skeleton = chain_skeleton();
        let q = |w: f64, x: f64, y: f64, z: f64| Quat::new(w, x, y, z).normalized();
        let frames = vec![
            Frame {
                poses: vec![
                    Pose {
                        loc: Vec3::ZERO,
                        quat: Quat::IDENTITY,
                    },
                    Pose {
                        loc: Vec3::ZERO,
                        quat: q(0.9, 0.43589, 0.0, 0.0),
                    },
                    Pose {
                        loc: Vec3::ZERO,
                        quat: q(0.7, 0.0, 0.71414, 0.0),
                    },
                ],
            },
            Frame {
                poses: vec![
                    Pose {
                        loc: Vec3::new(0.1, 0.2, 0.0),
                        quat: q(0.95, 0.0, 0.0, 0.31225),
                    },
                    Pose {
                        loc: Vec3::ZERO,
                        quat: q(0.8, 0.6, 0.0, 0.0),
                    },
                    Pose {
                        loc: Vec3::ZERO,
                        quat: Quat::IDENTITY,
                    },
                ],
            },
        ];
        Clip {
            fps: 30.0,
            skeleton,
            frames,
        }
    }

    fn identity_map() -> BoneMap {
        BoneMap {
            pairs: vec![
                ("Root".to_string(), "Root".to_string()),
                ("Mid".to_string(), "Mid".to_string()),
                ("Tip".to_string(), "Tip".to_string()),
            ],
            root_source: "Root".to_string(),
            root_target: "Root".to_string(),
            stride_source: None,
            stride_target: None,
            feet_source: vec![],
            feet_target: vec![],
        }
    }

    #[test]
    fn identity_transfer_is_exact() {
        // Posed ancestors included: the C-conjugation transfer reproduces
        // the source clip bit-near-exactly on identical skeletons (the
        // reference script's A'(P R) form is only first-order here).
        let clip = pose_clip();
        let (out, report) = retarget(
            &clip,
            &chain_skeleton(),
            &identity_map(),
            YawMode::None,
            false,
        )
        .unwrap();
        assert_eq!(report.mapped, 3);
        for (fi, frame) in out.frames.iter().enumerate() {
            for (bi, pose) in frame.poses.iter().enumerate() {
                let want = &clip.frames[fi].poses[bi];
                assert!(
                    pose.quat.approx_eq(want.quat, 1e-9),
                    "frame {} bone {}",
                    fi,
                    bi
                );
                assert!(
                    pose.loc.approx_eq(want.loc, 1e-9),
                    "frame {} bone {}",
                    fi,
                    bi
                );
            }
        }
        // And it round-trips through the text format.
        let text = emit_clip(&out).unwrap();
        let back = parse_clip(&text).unwrap();
        assert_eq!(emit_clip(&back).unwrap(), text);
    }

    #[test]
    fn rest_maps_to_rest_across_different_rests() {
        let skeleton = chain_skeleton();
        let rest = Clip {
            fps: 24.0,
            skeleton: skeleton.clone(),
            frames: vec![Frame {
                poses: vec![
                    Pose {
                        loc: Vec3::ZERO,
                        quat: Quat::IDENTITY
                    };
                    3
                ],
            }],
        };
        // Target with different proportions.
        let mut target = chain_skeleton();
        target.bones[1].head = Vec3::new(0.0, 0.0, 2.0);
        target.bones[1].tail = Vec3::new(0.0, 1.0, 2.0);
        target.bones[2].head = Vec3::new(0.0, 1.0, 2.0);
        target.bones[2].tail = Vec3::new(0.0, 2.0, 2.0);
        let (out, _) = retarget(&rest, &target, &identity_map(), YawMode::None, false).unwrap();
        for pose in &out.frames[0].poses {
            assert!(pose.quat.approx_eq(Quat::IDENTITY, 1e-12));
            assert!(pose.loc.approx_eq(Vec3::ZERO, 1e-12));
        }
    }

    #[test]
    fn stride_scales_root_motion() {
        let skeleton = chain_skeleton();
        let clip = Clip {
            fps: 30.0,
            skeleton: skeleton.clone(),
            frames: vec![Frame {
                poses: vec![
                    Pose {
                        loc: Vec3::new(1.0, 0.0, 0.0),
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
            }],
        };
        // Target legs half as long: Root->Tip distance 1.0 vs 0.5.
        let mut target = chain_skeleton();
        for b in target.bones.iter_mut() {
            b.head = b.head.scale(0.5);
            b.tail = b.tail.scale(0.5);
        }
        let mut map = identity_map();
        map.stride_source = Some(("Root".to_string(), "Tip".to_string()));
        map.stride_target = Some(("Root".to_string(), "Tip".to_string()));
        let (out, report) = retarget(&clip, &target, &map, YawMode::None, false).unwrap();
        assert!((report.stride_scale - 0.5).abs() < 1e-12);
        // Target root head travels 0.5 in x (rest head + scaled offset).
        let posed = fk(&target, &out.frames[0]);
        assert!((posed[0].head.x - target.bones[0].head.x - 0.5).abs() < 1e-9);
    }

    #[test]
    fn yaw_flip_reverses_travel() {
        // Source faces +Y, target faces -Y (mirrored): auto yaw flips so
        // forward motion stays forward.
        let skeleton = chain_skeleton();
        let mut clip_frames = Vec::new();
        for i in 0..3 {
            clip_frames.push(Frame {
                poses: vec![
                    Pose {
                        loc: Vec3::new(0.0, i as f64 * 0.5, 0.0),
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
        // Feet: source foot bone points +Y, target foot points -Y.
        // The root runs along +Y (identity basis) so root loc == world.
        let mut src_sk = skeleton.clone();
        src_sk.bones[0].head = Vec3::new(0.0, 0.0, 1.0);
        src_sk.bones[0].tail = Vec3::new(0.0, 1.0, 1.0);
        src_sk.bones.push(Bone {
            name: "Foot".to_string(),
            parent: Some(0),
            head: Vec3::new(0.1, 0.0, 0.1),
            tail: Vec3::new(0.1, 0.3, 0.1),
        });
        let mut tgt_sk = skeleton.clone();
        for b in tgt_sk.bones.iter_mut() {
            b.head.y = -b.head.y;
            b.tail.y = -b.tail.y;
        }
        tgt_sk.bones.push(Bone {
            name: "Foot".to_string(),
            parent: Some(0),
            head: Vec3::new(0.1, 0.0, 0.1),
            tail: Vec3::new(0.1, -0.3, 0.1),
        });
        for fr in clip_frames.iter_mut() {
            fr.poses.push(Pose {
                loc: Vec3::ZERO,
                quat: Quat::IDENTITY,
            });
        }
        let clip = Clip {
            fps: 30.0,
            skeleton: src_sk,
            frames: clip_frames,
        };
        let mut map = BoneMap {
            pairs: vec![
                ("Root".to_string(), "Root".to_string()),
                ("Mid".to_string(), "Mid".to_string()),
                ("Tip".to_string(), "Tip".to_string()),
                ("Foot".to_string(), "Foot".to_string()),
            ],
            root_source: "Root".to_string(),
            root_target: "Root".to_string(),
            stride_source: None,
            stride_target: None,
            feet_source: vec!["Foot".to_string()],
            feet_target: vec!["Foot".to_string()],
        };
        let (out, report) = retarget(&clip, &tgt_sk, &map, YawMode::Auto, false).unwrap();
        assert!(report.yaw_flip);
        let heads: Vec<Vec3> = out
            .frames
            .iter()
            .map(|fr| fk(&tgt_sk, fr)[0].head)
            .collect();
        // Target root travels toward -Y (its forward).
        assert!(heads[2].y < heads[0].y - 0.9);
        // Explicit none keeps +Y travel.
        map.feet_source.clear();
        let (out2, report2) = retarget(&clip, &tgt_sk, &map, YawMode::None, false).unwrap();
        assert!(!report2.yaw_flip);
        let heads2: Vec<Vec3> = out2
            .frames
            .iter()
            .map(|fr| fk(&tgt_sk, fr)[0].head)
            .collect();
        assert!(heads2[2].y > heads2[0].y + 0.9);
    }

    #[test]
    fn pin_pass_cancels_stance_drift() {
        // One driven foot chain root drifting +x at constant height: the
        // whole clip is stance, and the pin pass must flatten the track.
        let skeleton = Skeleton {
            bones: vec![
                Bone {
                    name: "Root".to_string(),
                    parent: None,
                    head: Vec3::ZERO,
                    tail: Vec3::new(0.0, 0.0, 1.0),
                },
                Bone {
                    name: "Foot".to_string(),
                    parent: Some(0),
                    head: Vec3::new(0.1, 0.0, 0.1),
                    tail: Vec3::new(0.1, 0.4, 0.1),
                },
            ],
        };
        let mut frames = Vec::new();
        for i in 0..5 {
            frames.push(Frame {
                poses: vec![
                    Pose {
                        loc: Vec3::ZERO,
                        quat: Quat::IDENTITY,
                    },
                    Pose {
                        loc: Vec3::new(i as f64 * 0.01, 0.0, 0.0),
                        quat: Quat::IDENTITY,
                    },
                ],
            });
        }
        let clip = Clip {
            fps: 30.0,
            skeleton: skeleton.clone(),
            frames,
        };
        let map = BoneMap {
            pairs: vec![("Foot".to_string(), "Foot".to_string())],
            root_source: "Foot".to_string(),
            root_target: "Foot".to_string(),
            stride_source: None,
            stride_target: None,
            feet_source: vec!["Foot".to_string()],
            feet_target: vec!["Foot".to_string()],
        };
        let (_, before) = retarget(&clip, &skeleton, &map, YawMode::None, false).unwrap();
        let (_, after) = retarget(&clip, &skeleton, &map, YawMode::None, true).unwrap();
        assert!(before.slide_after[0].stance_max_m_s > 0.2);
        assert_eq!(after.pin_intervals, 1);
        assert!(after.slide_after[0].stance_max_m_s < 1e-9);
    }

    #[test]
    fn skipped_rows_and_missing_root() {
        let clip = pose_clip();
        let mut map = identity_map();
        map.pairs.push(("Nope".to_string(), "Root".to_string()));
        map.pairs.push(("Root".to_string(), "Missing".to_string()));
        let (_, report) = retarget(&clip, &chain_skeleton(), &map, YawMode::None, false).unwrap();
        assert_eq!(report.skipped.len(), 2);
        map.root_source = "Nope".to_string();
        assert!(retarget(&clip, &chain_skeleton(), &map, YawMode::None, false).is_err());
    }

    #[test]
    fn bonemap_parse() {
        let text = r#"{"format": "motionforge-bonemap", "version": 1,
          "pairs": [{"source": "A", "target": "B"}],
          "root_source": "A", "root_target": "B",
          "stride_source": ["U", "F"], "stride_target": ["U2", "F2"],
          "feet_source": ["F"], "feet_target": ["F2"]}"#;
        let map = parse_bonemap(text).unwrap();
        assert_eq!(map.pairs.len(), 1);
        assert!(map.stride_source.is_some());
        assert_eq!(map.feet_target, vec!["F2".to_string()]);
        assert!(parse_bonemap(r#"{"format": "nope", "version": 1}"#).is_err());
    }
}
