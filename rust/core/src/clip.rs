//! Clip + skeleton formats: parsing, deterministic emission, FK.
//!
//! FK convention (verified Blender-exact by headless probe on Blender
//! 5.2, 2026-10-01 — see `docs/retarget.md` and the `fk_matches_blender`
//! test below): `M = P @ R @ B`, where `R` is the parent-relative rest
//! matrix (translation + rotation both parent-relative), `B` is the
//! pose delta (`loc` + `quat`, i.e. Blender's `matrix_basis`), and `P`
//! is the parent's posed matrix. Rest orientations are zero-roll bases
//! from head/tail (see `math::zero_roll_basis`).
//!
//! Parse rules: bones must be topologically sorted (parent before
//! child); frames are dense (every bone in every frame); quats are
//! normalized on load.

use crate::json::{parse, push_num, push_str, Json};
use crate::math::{zero_roll_basis, Mat3, Quat, Vec3};

pub const CLIP_FORMAT: &str = "motionforge-clip";
pub const SKELETON_FORMAT: &str = "motionforge-skeleton";
pub const FORMAT_VERSION: i64 = 1;

#[derive(Clone, Debug)]
pub struct Bone {
    pub name: String,
    pub parent: Option<usize>,
    pub head: Vec3,
    pub tail: Vec3,
}

#[derive(Clone, Debug, Default)]
pub struct Skeleton {
    pub bones: Vec<Bone>,
}

#[derive(Clone, Debug)]
pub struct Pose {
    pub loc: Vec3,
    pub quat: Quat,
}

#[derive(Clone, Debug)]
pub struct Frame {
    /// Poses in skeleton bone order.
    pub poses: Vec<Pose>,
}

#[derive(Clone, Debug)]
pub struct Clip {
    pub fps: f64,
    pub skeleton: Skeleton,
    pub frames: Vec<Frame>,
}

impl Skeleton {
    pub fn index(&self, name: &str) -> Option<usize> {
        self.bones.iter().position(|b| b.name == name)
    }

    pub fn len(&self) -> usize {
        self.bones.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bones.is_empty()
    }

    pub fn bone_length(&self, i: usize) -> f64 {
        (self.bones[i].tail - self.bones[i].head).length()
    }

    /// Armature-space rest orientation of bone `i`.
    pub fn rest_world(&self, i: usize) -> Mat3 {
        zero_roll_basis(self.bones[i].head, self.bones[i].tail)
    }

    /// Parent-relative rest rotation and offset: `M = P @ T(off) @ R @ B`.
    pub fn rest_parent_rel(&self, i: usize) -> (Mat3, Vec3) {
        let world = self.rest_world(i);
        match self.bones[i].parent {
            None => (world, self.bones[i].head),
            Some(p) => {
                let pw = self.rest_world(p);
                let rot = pw.transpose().mul_mat(world);
                let off = pw
                    .transpose()
                    .mul_vec(self.bones[i].head - self.bones[p].head);
                (rot, off)
            }
        }
    }

    pub fn depth(&self, i: usize) -> usize {
        let mut n = 0;
        let mut b = &self.bones[i];
        while let Some(p) = b.parent {
            n += 1;
            b = &self.bones[p];
        }
        n
    }
}

/// Posed armature-space transform of one bone: rotation + head position.
#[derive(Clone, Debug)]
pub struct FkBone {
    pub rot: Mat3,
    pub head: Vec3,
}

/// Forward kinematics for one frame. `out[i]` is bone `i`'s posed
/// armature-space rotation and head position.
pub fn fk(skeleton: &Skeleton, frame: &Frame) -> Vec<FkBone> {
    let n = skeleton.len();
    debug_assert_eq!(frame.poses.len(), n);
    let mut out = Vec::with_capacity(n);
    for (i, bone) in skeleton.bones.iter().enumerate() {
        let (rrot, roff) = skeleton.rest_parent_rel(i);
        let pose = &frame.poses[i];
        let brot = pose.quat.to_mat3();
        let local_rot = rrot.mul_mat(brot);
        let local_off = rrot.mul_vec(pose.loc) + roff;
        match bone.parent {
            None => out.push(FkBone {
                rot: local_rot,
                head: local_off,
            }),
            Some(p) => {
                let parent = &out[p];
                out.push(FkBone {
                    rot: parent.rot.mul_mat(local_rot),
                    head: parent.rot.mul_vec(local_off) + parent.head,
                });
            }
        }
    }
    out
}

/// World-space tail position of bone `i` given its FK result.
pub fn tail_world(skeleton: &Skeleton, posed: &[FkBone], i: usize) -> Vec3 {
    let len = skeleton.bone_length(i);
    posed[i].head + posed[i].rot.mul_vec(Vec3::new(0.0, len, 0.0))
}

// ---------------------------------------------------------------------------
// Parsing.
// ---------------------------------------------------------------------------

fn err_here(v: &Json, what: &str) -> String {
    format!("expected {} but found {}", what, v.kind())
}

fn get<'a>(obj: &'a Json, key: &str, ctx: &str) -> Result<&'a Json, String> {
    obj.get(key)
        .ok_or_else(|| format!("{}: missing key \"{}\"", ctx, key))
}

fn as_version(v: &Json, ctx: &str) -> Result<i64, String> {
    match v.as_f64() {
        Some(f) if f.fract() == 0.0 && f >= 1.0 && f <= 9.0 => Ok(f as i64),
        _ => Err(format!("{}: version must be a small integer", ctx)),
    }
}

fn check_envelope(root: &Json, want_format: &str) -> Result<(), String> {
    let ctx = format!("invalid {}", want_format);
    let format = get(root, "format", &ctx)?;
    if format.as_str() != Some(want_format) {
        return Err(format!("{}: bad \"format\"", ctx));
    }
    let version = as_version(get(root, "version", &ctx)?, &ctx)?;
    if version != FORMAT_VERSION {
        return Err(format!("{}: unsupported version {}", ctx, version));
    }
    Ok(())
}

/// Largest accepted coordinate (m). Generous for root motion, small
/// enough that squared distances stay finite downstream.
const MAX_COORD: f64 = 1e6;
/// Largest accepted frame rate.
const MAX_FPS: f64 = 1e4;

fn parse_vec3(v: &Json, ctx: &str) -> Result<Vec3, String> {
    let arr = v
        .as_arr()
        .ok_or_else(|| err_here(v, &format!("{} [x, y, z]", ctx)))?;
    if arr.len() != 3 {
        return Err(format!("{}: expected 3 numbers, found {}", ctx, arr.len()));
    }
    let mut xyz = [0.0; 3];
    for (i, item) in arr.iter().enumerate() {
        xyz[i] = item
            .as_f64()
            .ok_or_else(|| err_here(item, &format!("{} number", ctx)))?;
        if xyz[i].abs() > MAX_COORD {
            return Err(format!(
                "{}: coordinate {} out of range (|x| <= {} m)",
                ctx, xyz[i], MAX_COORD
            ));
        }
    }
    Ok(Vec3::new(xyz[0], xyz[1], xyz[2]))
}

fn parse_quat(v: &Json, ctx: &str) -> Result<Quat, String> {
    let arr = v
        .as_arr()
        .ok_or_else(|| err_here(v, &format!("{} [w, x, y, z]", ctx)))?;
    if arr.len() != 4 {
        return Err(format!("{}: expected 4 numbers, found {}", ctx, arr.len()));
    }
    let mut wxyz = [0.0; 4];
    for (i, item) in arr.iter().enumerate() {
        wxyz[i] = item
            .as_f64()
            .ok_or_else(|| err_here(item, &format!("{} number", ctx)))?;
    }
    let q = Quat::new(wxyz[0], wxyz[1], wxyz[2], wxyz[3]);
    let len = q.length();
    if len < 1e-9 {
        return Err(format!("{}: zero-length quaternion", ctx));
    }
    if !len.is_finite() {
        // Would normalize to all zeros (1/inf) and slip through.
        return Err(format!("{}: quaternion too large to normalize", ctx));
    }
    Ok(q.normalized())
}

fn parse_bones(obj: &Json, ctx: &str) -> Result<Skeleton, String> {
    let arr = obj
        .as_arr()
        .ok_or_else(|| err_here(obj, &format!("{} bone array", ctx)))?;
    if arr.is_empty() {
        return Err(format!("{}: skeleton has no bones", ctx));
    }
    let mut skeleton = Skeleton::default();
    for (bi, item) in arr.iter().enumerate() {
        let bctx = format!("{} bone {}", ctx, bi);
        let name = get(item, "name", &bctx)?;
        let name = name
            .as_str()
            .ok_or_else(|| err_here(name, "bone name string"))?;
        if name.is_empty() {
            return Err(format!("{}: empty bone name", bctx));
        }
        if skeleton.index(name).is_some() {
            return Err(format!("{}: duplicate bone \"{}\"", bctx, name));
        }
        let parent_json = get(item, "parent", &bctx)?;
        let parent = if parent_json.is_null() {
            None
        } else if let Some(pname) = parent_json.as_str() {
            match skeleton.index(pname) {
                Some(p) => Some(p),
                // Parents must precede children (topological order), so
                // any unknown-at-this-point name is either a typo or a
                // forward reference; both are errors.
                None => return Err(format!("{}: unknown parent \"{}\"", bctx, pname)),
            }
        } else {
            return Err(err_here(parent_json, "parent name or null"));
        };
        let head = parse_vec3(get(item, "head", &bctx)?, &format!("{} head", bctx))?;
        let tail = parse_vec3(get(item, "tail", &bctx)?, &format!("{} tail", bctx))?;
        if (tail - head).length() < 1e-9 {
            return Err(format!("{}: bone \"{}\" has zero length", bctx, name));
        }
        skeleton.bones.push(Bone {
            name: name.to_string(),
            parent,
            head,
            tail,
        });
    }
    Ok(skeleton)
}

/// Parse a `motionforge-skeleton` document.
pub fn parse_skeleton(text: &str) -> Result<Skeleton, String> {
    let root = parse(text).map_err(|e| format!("skeleton json: {}", e))?;
    check_envelope(&root, SKELETON_FORMAT)?;
    parse_skeleton_obj(&root, "skeleton")
}

/// Parse a `{"bones": [...]}` skeleton object (shared by skeleton
/// documents and the autopose effector format).
pub fn parse_skeleton_obj(obj: &Json, ctx: &str) -> Result<Skeleton, String> {
    let bones = get(obj, "bones", ctx)?;
    parse_bones(bones, ctx)
}

/// Parse a `motionforge-clip` document.
pub fn parse_clip(text: &str) -> Result<Clip, String> {
    let root = parse(text).map_err(|e| format!("clip json: {}", e))?;
    check_envelope(&root, CLIP_FORMAT)?;
    let fps_json = get(&root, "fps", "clip")?;
    let fps = fps_json
        .as_f64()
        .ok_or_else(|| err_here(fps_json, "fps number"))?;
    if fps <= 0.0 || fps > MAX_FPS {
        return Err(format!("clip: fps must be in (0, {}]", MAX_FPS));
    }
    let skel_json = get(&root, "skeleton", "clip")?;
    let bones_json = get(skel_json, "bones", "clip skeleton")?;
    let skeleton = parse_bones(bones_json, "clip skeleton")?;
    let frames_json = get(&root, "frames", "clip")?;
    let frames_arr = frames_json
        .as_arr()
        .ok_or_else(|| err_here(frames_json, "frames array"))?;
    if frames_arr.is_empty() {
        return Err("clip: no frames".to_string());
    }
    let mut frames = Vec::with_capacity(frames_arr.len());
    for (fi, item) in frames_arr.iter().enumerate() {
        let fctx = format!("clip frame {}", fi);
        let obj = item
            .as_obj()
            .ok_or_else(|| err_here(item, &format!("{} object", fctx)))?;
        let mut poses = Vec::with_capacity(skeleton.len());
        for bone in &skeleton.bones {
            let pose_json = item
                .get(&bone.name)
                .ok_or_else(|| format!("{}: missing bone \"{}\"", fctx, bone.name))?;
            let pctx = format!("{} bone \"{}\"", fctx, bone.name);
            let loc = parse_vec3(get(pose_json, "loc", &pctx)?, &format!("{} loc", pctx))?;
            let quat = parse_quat(get(pose_json, "quat", &pctx)?, &format!("{} quat", pctx))?;
            poses.push(Pose { loc, quat });
        }
        if obj.len() != skeleton.len() {
            return Err(format!("{}: has keys outside the skeleton", fctx));
        }
        frames.push(Frame { poses });
    }
    Ok(Clip {
        fps,
        skeleton,
        frames,
    })
}

// ---------------------------------------------------------------------------
// Emission (deterministic: fixed key order, skeleton bone order).
// ---------------------------------------------------------------------------

fn emit_bones(out: &mut String, skeleton: &Skeleton, indent: usize) -> Result<(), String> {
    let pad = " ".repeat(indent);
    let pad2 = " ".repeat(indent + 2);
    for (i, bone) in skeleton.bones.iter().enumerate() {
        out.push_str(&pad);
        out.push_str("{\"name\": ");
        push_str(out, &bone.name);
        out.push_str(", \"parent\": ");
        match bone.parent {
            None => out.push_str("null"),
            Some(p) => push_str(out, &skeleton.bones[p].name),
        }
        out.push_str(", \"head\": [");
        push_num(out, bone.head.x)?;
        out.push_str(", ");
        push_num(out, bone.head.y)?;
        out.push_str(", ");
        push_num(out, bone.head.z)?;
        out.push_str("], \"tail\": [");
        push_num(out, bone.tail.x)?;
        out.push_str(", ");
        push_num(out, bone.tail.y)?;
        out.push_str(", ");
        push_num(out, bone.tail.z)?;
        out.push(']');
        out.push('}');
        if i + 1 < skeleton.len() {
            out.push(',');
        }
        out.push('\n');
        let _ = pad2;
    }
    Ok(())
}

/// Emit a `motionforge-skeleton` document.
pub fn emit_skeleton(skeleton: &Skeleton) -> Result<String, String> {
    let mut out = String::new();
    out.push_str("{\"format\": \"motionforge-skeleton\", \"version\": 1,\n");
    out.push_str(" \"bones\": [\n");
    emit_bones(&mut out, skeleton, 2)?;
    out.push_str(" ]}\n");
    Ok(out)
}

/// Emit a `motionforge-clip` document.
pub fn emit_clip(clip: &Clip) -> Result<String, String> {
    let mut out = String::new();
    out.push_str("{\"format\": \"motionforge-clip\", \"version\": 1,\n \"fps\": ");
    push_num(&mut out, clip.fps)?;
    out.push_str(",\n \"skeleton\": {\"bones\": [\n");
    emit_bones(&mut out, &clip.skeleton, 3)?;
    out.push_str(" ]},\n \"frames\": [\n");
    for (fi, frame) in clip.frames.iter().enumerate() {
        out.push_str("  {");
        for (bi, bone) in clip.skeleton.bones.iter().enumerate() {
            if bi > 0 {
                out.push_str(", ");
            }
            push_str(&mut out, &bone.name);
            let pose = &frame.poses[bi];
            out.push_str(": {\"loc\": [");
            push_num(&mut out, pose.loc.x)?;
            out.push_str(", ");
            push_num(&mut out, pose.loc.y)?;
            out.push_str(", ");
            push_num(&mut out, pose.loc.z)?;
            out.push_str("], \"quat\": [");
            push_num(&mut out, pose.quat.w)?;
            out.push_str(", ");
            push_num(&mut out, pose.quat.x)?;
            out.push_str(", ");
            push_num(&mut out, pose.quat.y)?;
            out.push_str(", ");
            push_num(&mut out, pose.quat.z)?;
            out.push_str("]}");
        }
        out.push('}');
        if fi + 1 < clip.frames.len() {
            out.push(',');
        }
        out.push('\n');
    }
    out.push_str(" ]}\n");
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    pub fn two_bone_skeleton() -> Skeleton {
        Skeleton {
            bones: vec![
                Bone {
                    name: "Root".to_string(),
                    parent: None,
                    head: Vec3::new(100.0, 0.0, 1.0),
                    tail: Vec3::new(100.0, 0.0, 2.0),
                },
                Bone {
                    name: "Child".to_string(),
                    parent: Some(0),
                    head: Vec3::new(100.5, 0.0, 1.5),
                    tail: Vec3::new(100.5, 1.0, 1.5),
                },
            ],
        }
    }

    fn identity_frame(skeleton: &Skeleton) -> Frame {
        Frame {
            poses: skeleton
                .bones
                .iter()
                .map(|_| Pose {
                    loc: Vec3::ZERO,
                    quat: Quat::IDENTITY,
                })
                .collect(),
        }
    }

    #[test]
    fn fk_identity_returns_rest_heads() {
        let sk = two_bone_skeleton();
        let posed = fk(&sk, &identity_frame(&sk));
        assert!(posed[0].head.approx_eq(Vec3::new(100.0, 0.0, 1.0), 1e-12));
        assert!(posed[1].head.approx_eq(Vec3::new(100.5, 0.0, 1.5), 1e-12));
        assert!(tail_world(&sk, &posed, 1).approx_eq(Vec3::new(100.5, 1.0, 1.5), 1e-12));
    }

    #[test]
    fn fk_matches_blender() {
        // Same scene as the headless Blender probe (2026-10-01): root
        // basis = 90 deg about local Z, child head evaluated by Blender
        // at (99.5, 0, 1.5). Agreement proves the Rust FK convention
        // (M = P @ R @ B, zero-roll rests) reproduces Blender exactly.
        let sk = two_bone_skeleton();
        let q = Quat::new(0.7071067811865476, 0.0, 0.0, 0.7071067811865476);
        let frame = Frame {
            poses: vec![
                Pose {
                    loc: Vec3::ZERO,
                    quat: q,
                },
                Pose {
                    loc: Vec3::ZERO,
                    quat: Quat::IDENTITY,
                },
            ],
        };
        let posed = fk(&sk, &frame);
        assert!(posed[0].head.approx_eq(Vec3::new(100.0, 0.0, 1.0), 1e-9));
        assert!(posed[1].head.approx_eq(Vec3::new(99.5, 0.0, 1.5), 1e-9));
    }

    #[test]
    fn clip_roundtrip() {
        let clip = Clip {
            fps: 30.0,
            skeleton: two_bone_skeleton(),
            frames: vec![],
        };
        let mut clip = clip;
        clip.frames.push(identity_frame(&clip.skeleton));
        let text = emit_clip(&clip).unwrap();
        let back = parse_clip(&text).unwrap();
        assert_eq!(back.frames.len(), 1);
        assert_eq!(back.skeleton.len(), 2);
        assert!((back.fps - 30.0).abs() < 1e-12);
        // Byte-deterministic re-emit.
        assert_eq!(emit_clip(&back).unwrap(), text);
    }

    #[test]
    fn parse_rejects_bad_clips() {
        // Forward parent reference.
        let bad = r#"{"format": "motionforge-clip", "version": 1, "fps": 30,
          "skeleton": {"bones": [
            {"name": "A", "parent": "B", "head": [0,0,0], "tail": [0,0,1]},
            {"name": "B", "parent": null, "head": [0,0,0], "tail": [0,0,1]}]},
          "frames": []}"#;
        assert!(parse_clip(bad).is_err());
        // Missing bone in frame.
        let bad2 = r#"{"format": "motionforge-clip", "version": 1, "fps": 30,
          "skeleton": {"bones": [
            {"name": "A", "parent": null, "head": [0,0,0], "tail": [0,0,1]}]},
          "frames": [{"B": {"loc": [0,0,0], "quat": [1,0,0,0]}}]}"#;
        assert!(parse_clip(bad2).is_err());
        // Zero-length bone.
        let bad3 = r#"{"format": "motionforge-skeleton", "version": 1,
          "bones": [{"name": "A", "parent": null, "head": [1,2,3], "tail": [1,2,3]}]}"#;
        assert!(parse_skeleton(bad3).is_err());
    }

    #[test]
    fn parse_rejects_absurd_magnitudes() {
        let clip = |fps: &str, loc: &str, quat: &str| {
            format!(
                r#"{{"format": "motionforge-clip", "version": 1, "fps": {},
                  "skeleton": {{"bones": [
                    {{"name": "A", "parent": null, "head": [0,0,0], "tail": [0,0,1]}}]}},
                  "frames": [{{"A": {{"loc": {}, "quat": {}}}}}]}}"#,
                fps, loc, quat
            )
        };
        assert!(parse_clip(&clip("30", "[0,0,1e6]", "[1,0,0,0]")).is_ok());
        assert!(parse_clip(&clip("10000", "[0,0,0]", "[2,0,0,0]")).is_ok());
        let e = parse_clip(&clip("30", "[0,1e300,0]", "[1,0,0,0]")).unwrap_err();
        assert!(e.contains("out of range"), "{}", e);
        let e = parse_clip(&clip("1e300", "[0,0,0]", "[1,0,0,0]")).unwrap_err();
        assert!(e.contains("fps"), "{}", e);
        // |q| overflows to inf; normalizing would give all zeros.
        let e = parse_clip(&clip("30", "[0,0,0]", "[1e300,1e300,0,0]")).unwrap_err();
        assert!(e.contains("too large"), "{}", e);
    }
}
