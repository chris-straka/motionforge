//! Range-of-motion poses for the owner's pose-test sheet.
//!
//! Each pose becomes a one-key glTF animation on the character, so the
//! sheet renderer (`blender/tools/pose_sheet.py`) and any glTF viewer
//! can show it. Moves are written like weightforge's ROM set: bend a
//! bone toward a character direction, or twist it about itself, by
//! degrees. A move is applied in the bone's own rest frame on top of its
//! parent's posture, so a bent elbow stays bent when the shoulder moves.
//! Optional clip samples (frames of retargeted clips) join the sheet.

use crate::animate::{forward, write_animation};
use crate::glb::{canonical_quat, Document};
use crate::humanoid::{self, Mapping};
use crate::json::Json;
use crate::math::{Quat, Vec3};
use crate::rig::{Animation, Rig, Trs};

#[derive(Clone, Copy, Debug)]
pub enum Dir {
    Up,
    Down,
    Forward,
    Back,
    Out,
    In,
    Left,
}

#[derive(Clone, Copy, Debug)]
pub enum Motion {
    Bend(Dir, f64),
    Twist(f64),
}

/// `(pose name, [(bone without side or with .L, motion)], mirror both sides)`.
pub struct PoseSpec {
    pub name: &'static str,
    pub moves: Vec<(&'static str, Motion)>,
}

/// The humanoid sheet: 12 poses, both sides at once where it applies,
/// so one tile shows a joint pair.
pub fn humanoid_poses() -> Vec<PoseSpec> {
    use Dir::*;
    use Motion::*;
    vec![
        PoseSpec {
            name: "rest",
            moves: vec![],
        },
        PoseSpec {
            name: "arms up 70",
            moves: vec![("upper_arm.*", Bend(Up, 70.0))],
        },
        PoseSpec {
            name: "arms forward 80",
            moves: vec![("upper_arm.*", Bend(Forward, 80.0))],
        },
        PoseSpec {
            name: "elbows 140",
            moves: vec![("forearm.*", Bend(Forward, 140.0))],
        },
        PoseSpec {
            name: "shrug, wrists, fists",
            moves: vec![
                ("shoulder.*", Bend(Up, 20.0)),
                ("hand.*", Bend(Down, 60.0)),
                ("fingers.*", Bend(Down, 60.0)),
            ],
        },
        PoseSpec {
            name: "arms back 40",
            moves: vec![("upper_arm.*", Bend(Back, 40.0))],
        },
        PoseSpec {
            name: "knee up 90/110",
            moves: vec![
                ("thigh.L", Bend(Forward, 90.0)),
                ("shin.L", Bend(Back, 110.0)),
            ],
        },
        PoseSpec {
            name: "lunge",
            moves: vec![
                ("thigh.R", Bend(Forward, 60.0)),
                ("shin.R", Bend(Back, 70.0)),
                ("thigh.L", Bend(Back, 30.0)),
                ("foot.L", Bend(Up, 30.0)),
            ],
        },
        PoseSpec {
            name: "legs out 45",
            moves: vec![("thigh.*", Bend(Out, 45.0))],
        },
        PoseSpec {
            name: "bend forward 45",
            moves: vec![
                ("spine*", Bend(Forward, 45.0)),
                ("neck", Bend(Forward, 20.0)),
                ("head", Bend(Forward, 20.0)),
            ],
        },
        PoseSpec {
            name: "lean left 30",
            moves: vec![("spine*", Bend(Left, 30.0)), ("neck", Bend(Back, 20.0))],
        },
        PoseSpec {
            name: "twist 40, head 35",
            moves: vec![
                ("spine*", Twist(40.0)),
                ("neck", Twist(35.0)),
                ("head", Twist(35.0)),
            ],
        },
    ]
}

fn expand(bone: &str, map: &Mapping) -> Vec<(String, bool)> {
    // Returns (canonical name, is right side) pairs present in the rig.
    let sided = |b: &str| -> Vec<(String, bool)> {
        vec![
            (format!("DEF-{}.L", b), false),
            (format!("DEF-{}.R", b), true),
        ]
    };
    let list: Vec<(String, bool)> = match bone {
        "spine*" => ["DEF-spine.001", "DEF-spine.002", "DEF-spine.003"]
            .iter()
            .map(|s| (s.to_string(), false))
            .collect(),
        "neck" => vec![
            ("DEF-spine.004".into(), false),
            ("DEF-spine.005".into(), false),
        ],
        "head" => vec![("DEF-spine.006".into(), false)],
        "fingers.*" => {
            let mut v = Vec::new();
            for f in ["thumb", "f_index", "f_middle", "f_ring", "f_pinky"] {
                for seg in 1..=3 {
                    v.push((format!("DEF-{}.{:02}.L", f, seg), false));
                    v.push((format!("DEF-{}.{:02}.R", f, seg), true));
                }
            }
            v
        }
        b if b.ends_with(".*") => sided(&b[..b.len() - 2]),
        b if b.ends_with(".L") => vec![(format!("DEF-{}", b), false)],
        b if b.ends_with(".R") => vec![(format!("DEF-{}", b), true)],
        b => vec![(format!("DEF-{}", b), false)],
    };
    list.into_iter()
        .filter(|(n, _)| map.node_of(n).is_some())
        .collect()
}

/// Rest direction of a bone: toward its main child, else its parent's.
fn bone_dir(rig: &Rig, map: &Mapping, name: &str, up: Vec3, fwd: Vec3) -> Vec3 {
    let child = humanoid::main_children(name);
    let node = map.node_of(name).unwrap();
    for c in child {
        if let Some(cn) = map.node_of(&c) {
            let d = rig.head(cn) - rig.head(node);
            if d.length() > 1e-6 {
                return d.normalized();
            }
        }
    }
    if name == "DEF-spine.006" {
        return up;
    }
    if name.contains("foot") || name.contains("toe") {
        return fwd;
    }
    // Parent chain direction (hand tips, toes, finger ends).
    if let Some(p) = humanoid::present_parent(name, &|n| map.node_of(n).is_some()) {
        let pn = map.node_of(&p).unwrap();
        let d = rig.head(node) - rig.head(pn);
        if d.length() > 1e-6 {
            return d.normalized();
        }
    }
    up
}

/// Locals for one pose (`None` = rest).
pub fn pose_locals(rig: &Rig, map: &Mapping, spec: &PoseSpec) -> (Vec<Option<Trs>>, Vec<String>) {
    let up = Vec3::new(0.0, 1.0, 0.0);
    let fwd = forward(rig, map);
    let left = up.cross(fwd).normalized();
    let mut locals: Vec<Option<Trs>> = vec![None; rig.names.len()];
    let mut skipped = Vec::new();
    for (bone, motion) in &spec.moves {
        let targets = expand(bone, map);
        if targets.is_empty() {
            skipped.push(bone.to_string());
            continue;
        }
        let spread = if *bone == "spine*" || *bone == "neck" {
            targets.len() as f64
        } else {
            1.0
        };
        for (name, right) in targets {
            let node = map.node_of(&name).unwrap();
            let dir = bone_dir(rig, map, &name, up, fwd);
            let out = if right { left.scale(-1.0) } else { left };
            let (axis, deg) = match motion {
                Motion::Twist(d) => (dir, if right { -d } else { *d }),
                Motion::Bend(to, d) => {
                    let target = match to {
                        Dir::Up => up,
                        Dir::Down => up.scale(-1.0),
                        Dir::Forward => fwd,
                        Dir::Back => fwd.scale(-1.0),
                        Dir::Out => out,
                        Dir::In => out.scale(-1.0),
                        Dir::Left => left,
                    };
                    let axis = dir.cross(target);
                    if axis.length() < 1e-6 {
                        continue;
                    }
                    (axis.normalized(), *d)
                }
            };
            let w = Quat::from_axis_angle(axis, (deg / spread).to_radians());
            let rw = rig.rest_world[node].rotation();
            let local_delta = rw.conj().mul(w).mul(rw);
            let mut trs = locals[node].unwrap_or(rig.rest[node]);
            trs.r = canonical_quat(trs.r.mul(local_delta));
            locals[node] = Some(trs);
        }
    }
    (locals, skipped)
}

/// Write the pose sheet animations onto a copy of `doc`: every pose,
/// then `samples` (`(label, locals)`), each as a one-key animation with
/// all joints keyed (so switching actions never inherits a pose).
pub fn build(
    doc: &Document,
    samples: &[(String, Vec<Option<Trs>>)],
) -> Result<(Document, Json), String> {
    let rig = Rig::from_doc(doc)?;
    let map = humanoid::map_rig(&rig);
    if !map.complete() {
        return Err(format!(
            "not a standardized humanoid; missing {}",
            map.missing.join(", ")
        ));
    }
    let joints: Vec<usize> = map.pairs.iter().map(|(n, _)| *n).collect();
    let hips = map.node_of("DEF-spine");
    let mut out = doc.clone();
    out.json.remove("animations");
    let mut names = Vec::new();
    let mut skipped_all: Vec<String> = Vec::new();
    let mut emit = |out: &mut Document, name: &str, locals: &[Option<Trs>]| -> Result<(), String> {
        let tracks: Vec<Vec<Trs>> = joints
            .iter()
            .map(|&j| vec![locals[j].unwrap_or(rig.rest[j])])
            .collect();
        write_animation(out, name, &[0.0], &joints, &tracks, hips)?;
        names.push(Json::str(name));
        Ok(())
    };
    for (i, spec) in humanoid_poses().iter().enumerate() {
        let (locals, skipped) = pose_locals(&rig, &map, spec);
        for s in skipped {
            if !skipped_all.contains(&s) {
                skipped_all.push(s);
            }
        }
        emit(&mut out, &format!("{:02} {}", i + 1, spec.name), &locals)?;
    }
    for (label, locals) in samples {
        emit(&mut out, label, locals)?;
    }
    let fwd = forward(&rig, &map);
    let report = Json::obj(vec![
        ("poses", Json::Arr(names)),
        (
            "skipped_moves",
            Json::Arr(skipped_all.iter().map(|s| Json::str(s)).collect()),
        ),
        (
            "forward",
            Json::Arr(vec![Json::num(fwd.x), Json::num(fwd.y), Json::num(fwd.z)]),
        ),
        ("bones", Json::num(joints.len() as f64)),
    ]);
    Ok((out, report))
}

/// Sample frames of animations already on `animated` (same node layout
/// as the character) at `fractions` of each clip, root travel removed
/// horizontally so tiles stay centered.
pub fn clip_samples(
    animated: &Document,
    max_clips: usize,
    fractions: &[f64],
) -> Result<Vec<(String, Vec<Option<Trs>>)>, String> {
    let rig = Rig::from_doc(animated)?;
    let map = humanoid::map_rig(&rig);
    let hips = map.node_of("DEF-spine");
    let mut out = Vec::new();
    for anim in Animation::load_all(animated)?.into_iter().take(max_clips) {
        for f in fractions {
            let mut locals = anim.sample(&rig, anim.duration * f);
            if let Some(h) = hips {
                if let Some(trs) = locals[h].as_mut() {
                    let rest = rig.rest[h].t;
                    trs.t = Vec3::new(rest.x, trs.t.y, rest.z);
                }
            }
            out.push((format!("{} @{:.0}%", anim.name, f * 100.0), locals));
        }
    }
    Ok(out)
}
