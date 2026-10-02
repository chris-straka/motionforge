//! Joint-limit tables (`motionforge-limits` JSON, exported by rigforge).
//!
//! v1 is deliberately small: each entry caps one bone's LOCAL pose
//! rotation from rest (identity), measured as `2*acos(|w|)` in degrees.
//! Bones missing from the table are unconstrained; table entries for
//! bones the clip does not have are reported as unmatched (sorted) so
//! a stale table can never silently pass. Per-axis or swing/twist
//! limits are a v2 format extension, not silent extra keys.

use crate::json::parse;
use crate::math::Quat;

pub const LIMITS_FORMAT: &str = "motionforge-limits";

/// Parsed table: (bone, max degrees), sorted by bone for determinism.
#[derive(Debug)]
pub struct Limits {
    pub rig: String,
    pub max_deg: Vec<(String, f64)>,
}

impl Limits {
    pub fn max_for(&self, bone: &str) -> Option<f64> {
        self.max_deg
            .iter()
            .find(|(b, _)| b == bone)
            .map(|(_, m)| *m)
    }
}

#[derive(Debug)]
pub struct Violation {
    pub frame: usize,
    pub bone: String,
    pub angle_deg: f64,
    pub max_deg: f64,
}

#[derive(Debug)]
pub struct LimitReport {
    /// Frame-major, bone order within a frame.
    pub violations: Vec<Violation>,
    /// Sorted table bones absent from the skeleton.
    pub unmatched: Vec<String>,
}

pub fn parse_limits(text: &str) -> Result<Limits, String> {
    let root = parse(text).map_err(|e| format!("limits json: {}", e))?;
    if root.get("format").and_then(|v| v.as_str()) != Some(LIMITS_FORMAT) {
        return Err("invalid motionforge-limits: bad \"format\"".to_string());
    }
    match root.get("version").and_then(|v| v.as_f64()) {
        Some(1.0) => {}
        _ => return Err("invalid motionforge-limits: unsupported version".to_string()),
    }
    let rig = root
        .get("rig")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let bones_json = root
        .get("bones")
        .ok_or_else(|| "limits: missing \"bones\"".to_string())?;
    let bones_obj = bones_json
        .as_obj()
        .ok_or_else(|| "limits: \"bones\" must be an object".to_string())?;
    let mut max_deg = Vec::with_capacity(bones_obj.len());
    for (name, entry) in bones_obj {
        let max = entry
            .get("max_angle_deg")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| format!("limits: bone {} missing numeric max_angle_deg", name))?;
        if !max.is_finite() || max <= 0.0 || max > 180.0 {
            return Err(format!(
                "limits: bone {} max_angle_deg {} out of (0, 180]",
                name, max
            ));
        }
        max_deg.push((name.clone(), max));
    }
    max_deg.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(Limits { rig, max_deg })
}

/// Check local-pose quats (`frames[f][bone_idx]`) against the table.
/// `bones` names each quat slot. Angles within 1e-9 deg of the cap pass.
pub fn check_limits(
    limits: &Limits,
    bones: &[String],
    frames: &[Vec<Quat>],
) -> Result<LimitReport, String> {
    for (f, quats) in frames.iter().enumerate() {
        if quats.len() != bones.len() {
            return Err(format!(
                "limits: frame {} has {} quats, want {} bones",
                f,
                quats.len(),
                bones.len()
            ));
        }
    }
    let mut violations = Vec::new();
    for (f, quats) in frames.iter().enumerate() {
        for (i, q) in quats.iter().enumerate() {
            if let Some(max) = limits.max_for(&bones[i]) {
                let angle = Quat::IDENTITY.angle_to(*q).to_degrees();
                if angle > max + 1e-9 {
                    violations.push(Violation {
                        frame: f,
                        bone: bones[i].clone(),
                        angle_deg: angle,
                        max_deg: max,
                    });
                }
            }
        }
    }
    let mut unmatched: Vec<String> = limits
        .max_deg
        .iter()
        .filter(|(b, _)| !bones.iter().any(|s| s == b))
        .map(|(b, _)| b.clone())
        .collect();
    unmatched.sort();
    Ok(LimitReport {
        violations,
        unmatched,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const TABLE: &str = r#"{
        "format": "motionforge-limits", "version": 1, "rig": "hll_hero",
        "bones": {
            "DEF-shin.L": {"max_angle_deg": 150.0},
            "DEF-foot.L": {"max_angle_deg": 70.0}
        }
    }"#;

    fn bones() -> Vec<String> {
        ["DEF-shin.L", "DEF-foot.L", "DEF-toe.L"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    #[test]
    fn parses_and_sorts() {
        let lim = parse_limits(TABLE).unwrap();
        assert_eq!(lim.rig, "hll_hero");
        assert_eq!(lim.max_deg.len(), 2);
        assert!(lim.max_deg[0].0 < lim.max_deg[1].0);
        assert_eq!(lim.max_for("DEF-foot.L"), Some(70.0));
        assert_eq!(lim.max_for("DEF-toe.L"), None);
    }

    #[test]
    fn rejects_bad_envelopes() {
        for (bad, why) in [
            (r#"{"format": "nope", "version": 1, "bones": {}}"#, "format"),
            (
                r#"{"format": "motionforge-limits", "version": 2, "bones": {}}"#,
                "version",
            ),
            (
                r#"{"format": "motionforge-limits", "version": 1}"#,
                "missing",
            ),
            (
                r#"{"format": "motionforge-limits", "version": 1, "bones": []}"#,
                "shape",
            ),
            (
                r#"{"format": "motionforge-limits", "version": 1, "bones": {"a": {}}}"#,
                "entry",
            ),
            (
                r#"{"format": "motionforge-limits", "version": 1, "bones": {"a": {"max_angle_deg": 0}}}"#,
                "zero",
            ),
            (
                r#"{"format": "motionforge-limits", "version": 1, "bones": {"a": {"max_angle_deg": 181}}}"#,
                "over",
            ),
            (
                r#"{"format": "motionforge-limits", "version": 1, "bones": {"a": {"max_angle_deg": "x"}}}"#,
                "type",
            ),
        ] {
            assert!(parse_limits(bad).is_err(), "{}", why);
        }
        // Empty table with a valid envelope = explicitly unconstrained.
        let lim =
            parse_limits(r#"{"format": "motionforge-limits", "version": 1, "bones": {}}"#).unwrap();
        assert!(lim.max_deg.is_empty());
    }

    #[test]
    fn checks_frames_and_reports_unmatched() {
        let mut lim = parse_limits(TABLE).unwrap();
        lim.max_deg.push(("DEF-hand.L".to_string(), 90.0));
        let frames = vec![
            vec![Quat::IDENTITY; 3],
            vec![
                Quat::from_axis_angle(crate::math::Vec3::new(1.0, 0.0, 0.0), 80.0f64.to_radians()),
                Quat::from_axis_angle(crate::math::Vec3::new(1.0, 0.0, 0.0), 80.0f64.to_radians()),
                Quat::from_axis_angle(crate::math::Vec3::new(1.0, 0.0, 0.0), 170.0f64.to_radians()),
            ],
        ];
        let rep = check_limits(&lim, &bones(), &frames).unwrap();
        // shin 80 <= 150 ok; foot 80 > 70 violates; toe unconstrained.
        assert_eq!(rep.violations.len(), 1);
        let v = &rep.violations[0];
        assert_eq!((v.frame, v.bone.as_str()), (1, "DEF-foot.L"));
        assert!((v.angle_deg - 80.0).abs() < 1e-9);
        assert_eq!(v.max_deg, 70.0);
        assert_eq!(rep.unmatched, vec!["DEF-hand.L".to_string()]);
    }

    #[test]
    fn rejects_ragged_frames() {
        let lim = parse_limits(TABLE).unwrap();
        let err = check_limits(&lim, &bones(), &[vec![Quat::IDENTITY; 2]]).unwrap_err();
        assert!(err.contains("frame 0"), "{}", err);
    }
}
