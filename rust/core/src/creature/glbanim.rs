//! Write a clip solved on a creature profile into the creature's rigged
//! GLB as a glTF animation (the file the games load). The clip is in
//! the Blender Z-up convention; glTF is Y-up with arbitrary rest node
//! orientations, so each bone's world rotation delta is carried over
//! and re-expressed as node-local TRS.

use super::rigcheck::{to_yup, to_zup};
use crate::animate::write_animation;
use crate::clip::{fk, Clip};
use crate::glb::Document;
use crate::math::{Mat3, Quat, Vec3};
use crate::rig::{Rig, Trs};

/// Z-up -> Y-up as a rotation (x, y, z) -> (x, z, -y).
fn c_mat() -> Mat3 {
    Mat3::from_rows([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
}

/// Append animation `name` to `doc` from `clip`; every clip bone must be
/// a node of the GLB. Returns the new document.
pub fn clip_to_glb(doc: &Document, clip: &Clip, name: &str) -> Result<Document, String> {
    let rig = Rig::from_doc(doc)?;
    let n = clip.skeleton.len();
    let mut nodes = Vec::with_capacity(n);
    for b in &clip.skeleton.bones {
        let node = rig
            .find(&b.name)
            .ok_or_else(|| format!("bone \"{}\" is not a node of the GLB", b.name))?;
        let (_, _, s) = rig.rest_world[node].decompose();
        if (s.x - 1.0).abs() > 1e-3 || (s.y - 1.0).abs() > 1e-3 || (s.z - 1.0).abs() > 1e-3 {
            return Err(format!(
                "node \"{}\" is scaled ({:?}); scaled joints are unsupported",
                b.name, s
            ));
        }
        // Rest head must match the GLB's rest joint position.
        let rest = to_zup(rig.rest_world[node].t);
        if !(rest - b.head).approx_eq(Vec3::ZERO, 1e-3 * (1.0 + b.head.length())) {
            return Err(format!(
                "bone \"{}\" rest head {:?} differs from the GLB joint {:?}",
                b.name, b.head, rest
            ));
        }
        nodes.push(node);
    }
    let c = c_mat();
    let ct = c.transpose();
    let times: Vec<f64> = (0..clip.frames.len())
        .map(|f| f as f64 / clip.fps)
        .collect();
    let mut tracks: Vec<Vec<Trs>> = vec![Vec::with_capacity(clip.frames.len()); n];
    let root = clip
        .skeleton
        .bones
        .iter()
        .position(|b| b.parent.is_none())
        .unwrap_or(0);
    for frame in &clip.frames {
        let posed = fk(&clip.skeleton, frame);
        // Posed world rotation per node (glTF space) and position.
        let mut world_rot: Vec<Mat3> = Vec::with_capacity(n);
        let mut world_pos: Vec<Vec3> = Vec::with_capacity(n);
        for i in 0..n {
            let rest_basis = clip.skeleton.rest_world(i);
            let delta = posed[i].rot.mul_mat(rest_basis.transpose()); // Z-up world delta
            let delta_g = c.mul_mat(delta).mul_mat(ct);
            let rest_rot = rig.rest_world[nodes[i]].rotation().to_mat3();
            world_rot.push(delta_g.mul_mat(rest_rot));
            world_pos.push(to_yup(posed[i].head));
        }
        for i in 0..n {
            let (prot, ppos) = match clip.skeleton.bones[i].parent {
                Some(p) => (world_rot[p], world_pos[p]),
                None => match rig.parent[nodes[i]] {
                    // A non-joint parent (an armature node) stays at rest.
                    Some(pn) => (
                        rig.rest_world[pn].rotation().to_mat3(),
                        rig.rest_world[pn].t,
                    ),
                    None => (Mat3::identity(), Vec3::ZERO),
                },
            };
            let local_rot = prot.transpose().mul_mat(world_rot[i]);
            let local_t = prot.transpose().mul_vec(world_pos[i] - ppos);
            tracks[i].push(Trs {
                t: if i == root {
                    local_t
                } else {
                    rig.rest[nodes[i]].t
                },
                r: Quat::from_mat3(local_rot),
                s: Vec3::new(1.0, 1.0, 1.0),
            });
        }
    }
    let mut out = doc.clone();
    write_animation(&mut out, name, &times, &nodes, &tracks, Some(nodes[root]))?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::creature::motion::exam_motion;
    use crate::creature::rigcheck::{from_glb, Options};
    use crate::fixture::{humanoid, Naming};
    use crate::rig::Animation;

    #[test]
    fn written_animation_reproduces_clip_joint_positions() {
        let doc = humanoid(Naming::Mixamo, false, false).unwrap();
        let built = from_glb(
            &doc,
            &Options {
                forward: None,
                view_size: 128,
                name: "f".into(),
            },
        )
        .unwrap();
        let c = built.creature;
        let clip = exam_motion(&c, 12, 24.0, 5, 0.8);
        let out = clip_to_glb(&built.glb, &clip, "exam").unwrap();
        let rig = Rig::from_doc(&out).unwrap();
        let anims = Animation::load_all(&out).unwrap();
        assert_eq!(anims.len(), 1);
        for (f, frame) in clip.frames.iter().enumerate() {
            let posed = fk(&c.skeleton, frame);
            let locals = anims[0].sample(&rig, f as f64 / clip.fps);
            let world = rig.world(&locals);
            for (i, b) in c.skeleton.bones.iter().enumerate() {
                let node = rig.find(&b.name).unwrap();
                let got = to_zup(world[node].t);
                assert!(
                    got.approx_eq(posed[i].head, 2e-4),
                    "frame {} bone {} got {:?} want {:?}",
                    f,
                    b.name,
                    got,
                    posed[i].head
                );
            }
        }
    }
}
