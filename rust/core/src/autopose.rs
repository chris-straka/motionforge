//! AutoPose inference: a few moved joints in, full-body pose out.
//!
//! The model is a small MLP (trained in `python/`, see `docs/autopose.md`)
//! exported to the `motionforge-weights` format: row-major layers, ReLU
//! hidden activations, linear output. Input per bone (weights-bone
//! order): constrained `[x, y, z, mask]` (`mask` 1 when the animator
//! moved that joint, else zeros), followed by the `n` bone lengths.
//! Positions are root-relative (minus the root bone's current head,
//! `root_position` in the effectors file) for weights tagged
//! [`INPUT_ROOT_RELATIVE`], so a pose predicts the same wherever the
//! character stands; untagged / [`INPUT_ABSOLUTE`] weights (the original
//! encoding) get armature-space positions. Output: one `(w, x, y, z)` quat per bone, normalized (zero
//! rows become identity). Inference is plain CPU matvecs — microseconds
//! for the shipped sizes, far under the 10 ms gate.
//!
//! Training uses random effector subsets so any 1-6 joint combination
//! works; the CLI accepts 1-6 effectors per request.

use crate::clip::Skeleton;
use crate::json::parse;
use crate::math::{Quat, Vec3};
use crate::retarget::skeleton_from_json;

pub const WEIGHTS_FORMAT: &str = "motionforge-weights";
pub const EFFECTORS_FORMAT: &str = "motionforge-effectors";
pub const MAX_EFFECTORS: usize = 6;
/// Weights `input` tags (must match `python/autopose/dataset.py`).
pub const INPUT_ABSOLUTE: &str = "effector-pos-mask+lengths";
pub const INPUT_ROOT_RELATIVE: &str = "effector-pos-mask+lengths/root-relative";

#[derive(Clone, Debug)]
pub struct Layer {
    pub rows: usize,
    pub cols: usize,
    /// Row-major `rows x cols`.
    pub weights: Vec<f64>,
    pub bias: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct Weights {
    pub bones: Vec<String>,
    pub layers: Vec<Layer>,
    /// Inputs are measured from the root's head (see module docs).
    pub root_relative: bool,
}

impl Weights {
    pub fn n_bones(&self) -> usize {
        self.bones.len()
    }
}

pub fn parse_weights(text: &str) -> Result<Weights, String> {
    let root = parse(text).map_err(|e| format!("weights json: {}", e))?;
    if root.get("format").and_then(|v| v.as_str()) != Some(WEIGHTS_FORMAT) {
        return Err("invalid motionforge-weights: bad \"format\"".to_string());
    }
    match root.get("version").and_then(|v| v.as_f64()) {
        Some(1.0) => {}
        _ => return Err("invalid motionforge-weights: unsupported version".to_string()),
    }
    let root_relative = match root.get("input") {
        None => false,
        Some(v) => match v.as_str() {
            Some(INPUT_ABSOLUTE) => false,
            Some(INPUT_ROOT_RELATIVE) => true,
            _ => return Err(format!("weights: unknown \"input\" encoding {:?}", v)),
        },
    };
    let bones_json = root
        .get("bones")
        .ok_or_else(|| "weights: missing \"bones\"".to_string())?;
    let bones_arr = bones_json
        .as_arr()
        .ok_or_else(|| "weights: \"bones\" must be an array".to_string())?;
    if bones_arr.is_empty() {
        return Err("weights: no bones".to_string());
    }
    let mut bones = Vec::with_capacity(bones_arr.len());
    for b in bones_arr {
        let name = b
            .as_str()
            .ok_or_else(|| "weights: bone names must be strings".to_string())?;
        if name.is_empty() || bones.contains(&name.to_string()) {
            return Err("weights: empty or duplicate bone name".to_string());
        }
        bones.push(name.to_string());
    }
    let layers_json = root
        .get("layers")
        .ok_or_else(|| "weights: missing \"layers\"".to_string())?;
    let layers_arr = layers_json
        .as_arr()
        .ok_or_else(|| "weights: \"layers\" must be an array".to_string())?;
    if layers_arr.is_empty() {
        return Err("weights: no layers".to_string());
    }
    let mut layers = Vec::with_capacity(layers_arr.len());
    for (li, item) in layers_arr.iter().enumerate() {
        let w_json = item
            .get("weights")
            .ok_or_else(|| format!("weights: layer {} missing weights", li))?;
        let w_arr = w_json
            .as_arr()
            .ok_or_else(|| format!("weights: layer {} weights must be rows", li))?;
        if w_arr.is_empty() {
            return Err(format!("weights: layer {} has no rows", li));
        }
        let rows = w_arr.len();
        let cols = w_arr[0]
            .as_arr()
            .ok_or_else(|| format!("weights: layer {} row 0 not an array", li))?
            .len();
        if cols == 0 {
            return Err(format!("weights: layer {} has no cols", li));
        }
        let mut weights = Vec::with_capacity(rows * cols);
        for (ri, row) in w_arr.iter().enumerate() {
            let cells = row
                .as_arr()
                .ok_or_else(|| format!("weights: layer {} row {} not an array", li, ri))?;
            if cells.len() != cols {
                return Err(format!(
                    "weights: layer {} row {} has {} cols, want {}",
                    li,
                    ri,
                    cells.len(),
                    cols
                ));
            }
            for c in cells {
                weights.push(
                    c.as_f64()
                        .ok_or_else(|| format!("weights: layer {} non-numeric cell", li))?,
                );
            }
        }
        let b_json = item
            .get("bias")
            .ok_or_else(|| format!("weights: layer {} missing bias", li))?;
        let b_arr = b_json
            .as_arr()
            .ok_or_else(|| format!("weights: layer {} bias must be an array", li))?;
        if b_arr.len() != rows {
            return Err(format!(
                "weights: layer {} bias len {} != rows {}",
                li,
                b_arr.len(),
                rows
            ));
        }
        let mut bias = Vec::with_capacity(rows);
        for b in b_arr {
            bias.push(
                b.as_f64()
                    .ok_or_else(|| format!("weights: layer {} non-numeric bias", li))?,
            );
        }
        layers.push(Layer {
            rows,
            cols,
            weights,
            bias,
        });
    }
    // Dimension chain: 5n in, 4n out, layers agree.
    let n = bones.len();
    if layers[0].cols != 5 * n {
        return Err(format!(
            "weights: input dim {} != 5 * {} bones",
            layers[0].cols, n
        ));
    }
    for w in layers.windows(2) {
        if w[0].rows != w[1].cols {
            return Err(format!(
                "weights: layer dims {} -> {} disagree",
                w[0].rows, w[1].cols
            ));
        }
    }
    if layers.last().unwrap().rows != 4 * n {
        return Err(format!(
            "weights: output dim {} != 4 * {} bones",
            layers.last().unwrap().rows,
            n
        ));
    }
    Ok(Weights {
        bones,
        layers,
        root_relative,
    })
}

#[derive(Clone, Debug)]
pub struct EffectorInput {
    pub skeleton: Skeleton,
    /// (bone index, armature-space position).
    pub effectors: Vec<(usize, Vec3)>,
    /// The root bone's (bone 0's) current armature-space head; required
    /// by root-relative weights.
    pub root_position: Option<Vec3>,
}

pub fn parse_effectors(text: &str) -> Result<EffectorInput, String> {
    let root = parse(text).map_err(|e| format!("effectors json: {}", e))?;
    if root.get("format").and_then(|v| v.as_str()) != Some(EFFECTORS_FORMAT) {
        return Err("invalid motionforge-effectors: bad \"format\"".to_string());
    }
    match root.get("version").and_then(|v| v.as_f64()) {
        Some(1.0) => {}
        _ => return Err("invalid motionforge-effectors: unsupported version".to_string()),
    }
    let skel_json = root
        .get("skeleton")
        .ok_or_else(|| "effectors: missing \"skeleton\"".to_string())?;
    let skeleton = skeleton_from_json(skel_json)?;
    let root_position = match root.get("root_position") {
        None => None,
        Some(v) => Some(parse_xyz(v, "effectors: root_position")?),
    };
    let eff_json = root
        .get("effectors")
        .ok_or_else(|| "effectors: missing \"effectors\"".to_string())?;
    let eff_arr = eff_json
        .as_arr()
        .ok_or_else(|| "effectors: \"effectors\" must be an array".to_string())?;
    if eff_arr.is_empty() {
        return Err("effectors: need at least 1 effector".to_string());
    }
    if eff_arr.len() > MAX_EFFECTORS {
        return Err(format!("effectors: at most {} effectors", MAX_EFFECTORS));
    }
    let mut effectors = Vec::with_capacity(eff_arr.len());
    for (i, item) in eff_arr.iter().enumerate() {
        let name = item
            .get("bone")
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("effectors: entry {} missing bone", i))?;
        let idx = skeleton
            .index(name)
            .ok_or_else(|| format!("effectors: unknown bone \"{}\"", name))?;
        if effectors.iter().any(|(e, _)| *e == idx) {
            return Err(format!("effectors: duplicate bone \"{}\"", name));
        }
        let pos = item
            .get("position")
            .ok_or_else(|| format!("effectors: entry {} missing position", i))?;
        effectors.push((
            idx,
            parse_xyz(pos, &format!("effectors: entry {} position", i))?,
        ));
    }
    Ok(EffectorInput {
        skeleton,
        effectors,
        root_position,
    })
}

fn parse_xyz(v: &crate::json::Json, ctx: &str) -> Result<Vec3, String> {
    let arr = v
        .as_arr()
        .filter(|a| a.len() == 3)
        .ok_or_else(|| format!("{} must be [x, y, z]", ctx))?;
    let mut xyz = [0.0; 3];
    for (k, item) in arr.iter().enumerate() {
        xyz[k] = item
            .as_f64()
            .ok_or_else(|| format!("{} must be numeric", ctx))?;
    }
    Ok(Vec3::new(xyz[0], xyz[1], xyz[2]))
}

/// Run the MLP: returns one local-delta quat per bone (weights order).
pub fn infer(weights: &Weights, input: &EffectorInput) -> Result<Vec<Quat>, String> {
    let n = weights.n_bones();
    if input.skeleton.len() != n
        || input
            .skeleton
            .bones
            .iter()
            .map(|b| &b.name)
            .ne(weights.bones.iter())
    {
        return Err(
            "weights bones do not match the effector skeleton (same rig, same order)".to_string(),
        );
    }
    let origin = if weights.root_relative {
        input.root_position.ok_or_else(|| {
            "these weights are root-relative: the effectors file needs \"root_position\" \
             (the root bone's current head); re-export from the Blender extension"
                .to_string()
        })?
    } else {
        Vec3::ZERO
    };
    // Input vector: per bone [x, y, z, mask], then bone lengths.
    let mut x = vec![0.0; 5 * n];
    for (idx, pos) in &input.effectors {
        let p = *pos - origin;
        x[4 * idx] = p.x;
        x[4 * idx + 1] = p.y;
        x[4 * idx + 2] = p.z;
        x[4 * idx + 3] = 1.0;
    }
    for i in 0..n {
        x[4 * n + i] = input.skeleton.bone_length(i);
    }
    let last = weights.layers.len() - 1;
    for (li, layer) in weights.layers.iter().enumerate() {
        let mut y = vec![0.0; layer.rows];
        for r in 0..layer.rows {
            let mut acc = layer.bias[r];
            let base = r * layer.cols;
            for c in 0..layer.cols {
                acc += layer.weights[base + c] * x[c];
            }
            y[r] = if li == last { acc } else { acc.max(0.0) };
        }
        x = y;
    }
    let mut quats = Vec::with_capacity(n);
    for i in 0..n {
        quats.push(Quat::new(x[4 * i], x[4 * i + 1], x[4 * i + 2], x[4 * i + 3]).normalized());
    }
    Ok(quats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip::Bone;

    fn one_bone_skeleton() -> Skeleton {
        Skeleton {
            bones: vec![Bone {
                name: "Hips".to_string(),
                parent: None,
                head: Vec3::ZERO,
                tail: Vec3::new(0.0, 1.0, 0.0),
            }],
        }
    }

    fn skeleton_doc() -> &'static str {
        r#"{"bones": [{"name": "Hips", "parent": null, "head": [0,0,0], "tail": [0,1,0]}]}"#
    }

    #[test]
    fn linear_net_known_answer() {
        // 1 bone: 5 inputs -> 4 outputs, out = (1, px, py, pz).
        let weights = parse_weights(
            r#"{"format": "motionforge-weights", "version": 1, "bones": ["Hips"],
            "layers": [{"weights": [
              [0,0,0,0,0], [1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0]],
              "bias": [1, 0, 0, 0]}]}"#,
        )
        .unwrap();
        let input = parse_effectors(&format!(
            r#"{{"format": "motionforge-effectors", "version": 1,
            "skeleton": {}, "effectors": [{{"bone": "Hips", "position": [0.2, 0, 0]}}]}}"#,
            skeleton_doc()
        ))
        .unwrap();
        let quats = infer(&weights, &input).unwrap();
        let want = Quat::new(1.0, 0.2, 0.0, 0.0).normalized();
        assert!(quats[0].approx_eq(want, 1e-12));
    }

    /// 1-bone linear net echoing the input position: out = (1, px, py, pz).
    fn echo_weights(input_tag: Option<&str>) -> Weights {
        let tag = input_tag
            .map(|t| format!(r#""input": "{}","#, t))
            .unwrap_or_default();
        parse_weights(&format!(
            r#"{{"format": "motionforge-weights", "version": 1, "bones": ["Hips"], {}
            "layers": [{{"weights": [
              [0,0,0,0,0], [1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0]],
              "bias": [1, 0, 0, 0]}}]}}"#,
            tag
        ))
        .unwrap()
    }

    fn effectors_at(pos: [f64; 3], root: Option<[f64; 3]>) -> EffectorInput {
        let root = root
            .map(|r| format!(r#""root_position": [{}, {}, {}],"#, r[0], r[1], r[2]))
            .unwrap_or_default();
        parse_effectors(&format!(
            r#"{{"format": "motionforge-effectors", "version": 1, "skeleton": {}, {}
            "effectors": [{{"bone": "Hips", "position": [{}, {}, {}]}}]}}"#,
            skeleton_doc(),
            root,
            pos[0],
            pos[1],
            pos[2]
        ))
        .unwrap()
    }

    #[test]
    fn root_relative_inputs_ignore_where_the_character_stands() {
        let rel = echo_weights(Some(INPUT_ROOT_RELATIVE));
        assert!(rel.root_relative);
        // Input is position minus root: (0.2, 0, 0).
        let here = infer(&rel, &effectors_at([1.2, 3.0, 0.5], Some([1.0, 3.0, 0.5]))).unwrap();
        assert!(here[0].approx_eq(Quat::new(1.0, 0.2, 0.0, 0.0).normalized(), 1e-12));
        // The same pose 7 m away predicts the same rotation.
        let there = infer(&rel, &effectors_at([8.2, 3.0, 0.5], Some([8.0, 3.0, 0.5]))).unwrap();
        assert!(here[0].approx_eq(there[0], 1e-12));
        // Absolute (legacy, tagged or untagged) weights still see the shift.
        for abs in [echo_weights(Some(INPUT_ABSOLUTE)), echo_weights(None)] {
            assert!(!abs.root_relative);
            let a = infer(&abs, &effectors_at([1.2, 3.0, 0.5], Some([1.0, 3.0, 0.5]))).unwrap();
            let b = infer(&abs, &effectors_at([8.2, 3.0, 0.5], Some([8.0, 3.0, 0.5]))).unwrap();
            assert!(!a[0].approx_eq(b[0], 1e-3));
        }
    }

    #[test]
    fn root_relative_needs_root_position() {
        let rel = echo_weights(Some(INPUT_ROOT_RELATIVE));
        let err = infer(&rel, &effectors_at([0.2, 0.0, 0.0], None)).unwrap_err();
        assert!(err.contains("root_position"), "{}", err);
        // Absolute weights don't need it.
        assert!(infer(&echo_weights(None), &effectors_at([0.2, 0.0, 0.0], None)).is_ok());
        // Unknown encodings are refused, not guessed.
        let err = parse_weights(
            r#"{"format": "motionforge-weights", "version": 1, "bones": ["Hips"],
            "input": "something-else",
            "layers": [{"weights": [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]],
              "bias": [1, 0, 0, 0]}]}"#,
        )
        .unwrap_err();
        assert!(err.contains("input"), "{}", err);
    }

    #[test]
    fn relu_clips_negatives() {
        // 5 -> 2 -> 4: hidden unit 0 sees -10 (clipped), unit 1 sees +2.
        let weights = parse_weights(
            r#"{"format": "motionforge-weights", "version": 1, "bones": ["Hips"],
            "layers": [
              {"weights": [[0,0,0,0,0],[0,0,0,1,0]], "bias": [-10, 1]},
              {"weights": [[1,0],[0,1],[0,0],[0,0]], "bias": [0, 0, 0.5, 0]}]}"#,
        )
        .unwrap();
        let input = parse_effectors(&format!(
            r#"{{"format": "motionforge-effectors", "version": 1,
            "skeleton": {}, "effectors": [{{"bone": "Hips", "position": [9, 9, 9]}}]}}"#,
            skeleton_doc()
        ))
        .unwrap();
        // Hidden = (max(0,-10), max(0, 1*1+1)) = (0, 2); out = (0, 2, 0.5, 0).
        let quats = infer(&weights, &input).unwrap();
        let want = Quat::new(0.0, 2.0, 0.5, 0.0).normalized();
        assert!(quats[0].approx_eq(want, 1e-12));
    }

    #[test]
    fn zero_row_becomes_identity() {
        let weights = parse_weights(
            r#"{"format": "motionforge-weights", "version": 1, "bones": ["Hips"],
            "layers": [{"weights": [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]],
              "bias": [0, 0, 0, 0]}]}"#,
        )
        .unwrap();
        let input = parse_effectors(&format!(
            r#"{{"format": "motionforge-effectors", "version": 1,
            "skeleton": {}, "effectors": [{{"bone": "Hips", "position": [1, 2, 3]}}]}}"#,
            skeleton_doc()
        ))
        .unwrap();
        assert!(infer(&weights, &input).unwrap()[0].approx_eq(Quat::IDENTITY, 1e-12));
    }

    #[test]
    fn validation_errors() {
        // Bad dims.
        assert!(parse_weights(
            r#"{"format": "motionforge-weights", "version": 1, "bones": ["Hips"],
            "layers": [{"weights": [[1,2,3]], "bias": [0]}]}"#
        )
        .is_err());
        // Zero effectors.
        assert!(parse_effectors(&format!(
            r#"{{"format": "motionforge-effectors", "version": 1, "skeleton": {}, "effectors": []}}"#,
            skeleton_doc()
        ))
        .is_err());
        // Unknown bone.
        assert!(parse_effectors(&format!(
            r#"{{"format": "motionforge-effectors", "version": 1, "skeleton": {},
            "effectors": [{{"bone": "Nope", "position": [0,0,0]}}]}}"#,
            skeleton_doc()
        ))
        .is_err());
        // Bone mismatch between weights and request.
        let weights = parse_weights(
            r#"{"format": "motionforge-weights", "version": 1, "bones": ["Other"],
            "layers": [{"weights": [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]],
              "bias": [1, 0, 0, 0]}]}"#,
        )
        .unwrap();
        let input = parse_effectors(&format!(
            r#"{{"format": "motionforge-effectors", "version": 1,
            "skeleton": {}, "effectors": [{{"bone": "Hips", "position": [0,0,0]}}]}}"#,
            skeleton_doc()
        ))
        .unwrap();
        assert!(infer(&weights, &input).is_err());
        let _ = one_bone_skeleton();
    }
}
