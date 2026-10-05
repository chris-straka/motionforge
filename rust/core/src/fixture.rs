//! Procedural skinned humanoid GLBs for tests (synthetic test vectors,
//! not assets). Y-up, facing +Z, 1.7 m tall, one box per bone.
//!
//! Naming styles cover what the standardizer must read: Mixamo
//! (`mixamorig:` prefix, also Tripo's `spec: mixamo`), plain
//! `UpperArm_L` style, and canonical `DEF-*`. `twisted` gives every joint
//! an arbitrary rest rotation (world positions unchanged), the way rigs
//! from other tools arrive. Both styles carry a non-deform `Root` joint
//! above the hips and an end joint that maps to nothing, so weight
//! merging is exercised.

use crate::glb::{set_node_trs, Affine, Document};
use crate::json::Json;
use crate::math::{Quat, Vec3};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Naming {
    Mixamo,
    Plain,
    Def,
}

impl Naming {
    pub fn parse(s: &str) -> Option<Naming> {
        match s {
            "mixamo" => Some(Naming::Mixamo),
            "plain" => Some(Naming::Plain),
            "def" => Some(Naming::Def),
            _ => None,
        }
    }
}

/// `(role, parent role, head position)`; roles are canonical-ish keys.
fn layout() -> Vec<(&'static str, Option<&'static str>, [f64; 3])> {
    let mut v: Vec<(&'static str, Option<&'static str>, [f64; 3])> = vec![
        ("root", None, [0.0, 0.0, 0.0]),
        ("hips", Some("root"), [0.0, 0.95, 0.0]),
        ("spine1", Some("hips"), [0.0, 1.05, 0.0]),
        ("spine2", Some("spine1"), [0.0, 1.2, 0.0]),
        ("chest", Some("spine2"), [0.0, 1.35, 0.0]),
        ("neck", Some("chest"), [0.0, 1.5, 0.0]),
        ("head", Some("neck"), [0.0, 1.6, 0.0]),
    ];
    for (s, x) in [("L", 1.0), ("R", -1.0)] {
        let leak = |a: String| -> &'static str { Box::leak(a.into_boxed_str()) };
        let k = |b: &str| leak(format!("{}.{}", b, s));
        v.push((k("shoulder"), Some("chest"), [0.04 * x, 1.45, 0.0]));
        v.push((k("upper_arm"), Some(k("shoulder")), [0.17 * x, 1.45, 0.0]));
        v.push((k("forearm"), Some(k("upper_arm")), [0.43 * x, 1.45, 0.0]));
        v.push((k("hand"), Some(k("forearm")), [0.68 * x, 1.45, 0.0]));
        v.push((k("middle1"), Some(k("hand")), [0.76 * x, 1.45, 0.0]));
        v.push((k("extra"), Some(k("hand")), [0.80 * x, 1.45, 0.02]));
        v.push((k("thigh"), Some("hips"), [0.09 * x, 0.92, 0.0]));
        v.push((k("shin"), Some(k("thigh")), [0.09 * x, 0.5, 0.0]));
        v.push((k("foot"), Some(k("shin")), [0.09 * x, 0.08, 0.0]));
        v.push((k("toe"), Some(k("foot")), [0.09 * x, 0.02, 0.12]));
    }
    v
}

fn bone_name(role: &str, naming: Naming) -> String {
    let (base, side) = match role.split_once('.') {
        Some((b, s)) => (b, Some(s)),
        None => (role, None),
    };
    let lr = |s: &str| if s == "L" { "Left" } else { "Right" };
    match naming {
        Naming::Mixamo => {
            let n = match (base, side) {
                ("root", _) => return "Root".into(),
                ("hips", _) => "Hips".into(),
                ("spine1", _) => "Spine".into(),
                ("spine2", _) => "Spine1".into(),
                ("chest", _) => "Spine2".into(),
                ("neck", _) => "Neck".into(),
                ("head", _) => "Head".into(),
                (b, Some(s)) => {
                    let part = match b {
                        "shoulder" => "Shoulder",
                        "upper_arm" => "Arm",
                        "forearm" => "ForeArm",
                        "hand" => "Hand",
                        "middle1" => "HandMiddle1",
                        "extra" => "HandPinky4",
                        "thigh" => "UpLeg",
                        "shin" => "Leg",
                        "foot" => "Foot",
                        _ => "ToeBase",
                    };
                    format!("{}{}", lr(s), part)
                }
                _ => unreachable!(),
            };
            format!("mixamorig:{}", n)
        }
        Naming::Plain => match (base, side) {
            ("root", _) => "Root".into(),
            ("hips", _) => "Hips".into(),
            ("spine1", _) => "Spine".into(),
            ("spine2", _) => "Spine1".into(),
            ("chest", _) => "Chest".into(),
            ("neck", _) => "Neck".into(),
            ("head", _) => "Head".into(),
            (b, Some(s)) => {
                let part = match b {
                    "shoulder" => "Clavicle",
                    "upper_arm" => "UpperArm",
                    "forearm" => "Forearm",
                    "hand" => "Hand",
                    "middle1" => "Finger2",
                    "extra" => "Finger1",
                    "thigh" => "Thigh",
                    "shin" => "Shin",
                    "foot" => "Foot",
                    _ => "Toe",
                };
                format!("{}_{}", part, s)
            }
            _ => unreachable!(),
        },
        Naming::Def => match (base, side) {
            ("root", _) => "root".into(),
            ("hips", _) => "DEF-spine".into(),
            ("spine1", _) => "DEF-spine.001".into(),
            ("spine2", _) => "DEF-spine.002".into(),
            ("chest", _) => "DEF-spine.003".into(),
            ("neck", _) => "DEF-spine.004".into(),
            ("head", _) => "DEF-spine.006".into(),
            ("middle1", Some(s)) => format!("DEF-f_middle.01.{}", s),
            ("extra", Some(s)) => format!("DEF-palm.04.{}", s),
            (b, Some(s)) => format!("DEF-{}.{}", b, s),
            _ => unreachable!(),
        },
    }
}

/// Deterministic "arbitrary" rest rotation per joint index.
fn twist_of(i: usize) -> Quat {
    let a = 0.37 * (i as f64 + 1.0);
    Quat::from_axis_angle(
        Vec3::new(1.0 + 0.1 * i as f64, -0.5, 0.25 * (i % 3) as f64 + 0.2),
        a,
    )
}

pub fn humanoid(naming: Naming, twisted: bool, walk: bool) -> Result<Document, String> {
    let lay = layout();
    let idx = |role: &str| lay.iter().position(|(r, _, _)| *r == role).unwrap();
    let n = lay.len();
    let mut doc = Document {
        json: Json::obj(vec![(
            "asset",
            Json::obj(vec![
                ("version", Json::str("2.0")),
                ("generator", Json::str("motionforge fixture")),
            ]),
        )]),
        bin: Vec::new(),
    };
    // Joint nodes 0..n, then the mesh node n, scene roots: root joint + mesh.
    let mut world: Vec<Affine> = vec![Affine::IDENTITY; n];
    let mut nodes: Vec<Json> = Vec::new();
    for (i, (role, parent, head)) in lay.iter().enumerate() {
        let pos = Vec3::new(head[0], head[1], head[2]);
        let rot = if twisted { twist_of(i) } else { Quat::IDENTITY };
        let want = Affine::from_trs(pos, rot, Vec3::new(1.0, 1.0, 1.0));
        let pw = parent.map(|p| world[idx(p)]).unwrap_or(Affine::IDENTITY);
        let local = pw.inverse().unwrap().mul(&want);
        world[i] = want;
        let (t, r, s) = local.decompose();
        let mut node = Json::obj(vec![("name", Json::str(&bone_name(role, naming)))]);
        set_node_trs(&mut node, t, r, s);
        let kids: Vec<Json> = lay
            .iter()
            .enumerate()
            .filter(|(_, (_, p, _))| *p == Some(*role))
            .map(|(k, _)| Json::num(k as f64))
            .collect();
        if !kids.is_empty() {
            node.set("children", Json::Arr(kids));
        }
        nodes.push(node);
    }
    // Mesh: a box per bone from its head toward its first child (or a
    // small cube), near-head ring half weighted to the parent.
    let mut positions: Vec<Vec<f64>> = Vec::new();
    let mut joints: Vec<[u16; 4]> = Vec::new();
    let mut weights: Vec<Vec<f64>> = Vec::new();
    let mut indices: Vec<u16> = Vec::new();
    for (i, (role, parent, head)) in lay.iter().enumerate() {
        if *role == "root" {
            continue;
        }
        let a = Vec3::new(head[0], head[1], head[2]);
        let child = lay
            .iter()
            .find(|(_, p, _)| *p == Some(*role))
            .map(|(_, _, h)| Vec3::new(h[0], h[1], h[2]));
        let b = child.unwrap_or(a + Vec3::new(0.0, 0.06, 0.0));
        let d = (b - a).normalized();
        let helper = if d.y.abs() > 0.9 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };
        let u = d.cross(helper).normalized().scale(0.035);
        let w = d.cross(u).normalized().scale(0.035);
        let base = positions.len() as u16;
        let parent_idx = parent.map(idx).unwrap_or(i) as u16;
        for (end, p) in [(0, a), (1, b)] {
            for (su, sw) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                let v = p + u.scale(su) + w.scale(sw);
                positions.push(vec![v.x, v.y, v.z]);
                if end == 0 && parent.is_some() {
                    joints.push([i as u16, parent_idx, 0, 0]);
                    weights.push(vec![0.5, 0.5, 0.0, 0.0]);
                } else {
                    joints.push([i as u16, 0, 0, 0]);
                    weights.push(vec![1.0, 0.0, 0.0, 0.0]);
                }
            }
        }
        let quads = [
            [0, 1, 2, 3],
            [4, 7, 6, 5],
            [0, 4, 5, 1],
            [1, 5, 6, 2],
            [2, 6, 7, 3],
            [3, 7, 4, 0],
        ];
        for q in quads {
            for t in [[q[0], q[1], q[2]], [q[0], q[2], q[3]]] {
                for k in t {
                    indices.push(base + k as u16);
                }
            }
        }
    }
    let pos_acc = doc.push_float_accessor(&positions, "VEC3", true);
    let j_acc = doc.push_joints_accessor(&joints);
    let w_acc = doc.push_float_accessor(&weights, "VEC4", false);
    let mut ibytes = Vec::new();
    for i in &indices {
        ibytes.extend_from_slice(&i.to_le_bytes());
    }
    // Index buffer view + accessor.
    while doc.bin.len() % 4 != 0 {
        doc.bin.push(0);
    }
    let off = doc.bin.len();
    doc.bin.extend_from_slice(&ibytes);
    let views = doc.array_mut("bufferViews");
    views.push(Json::obj(vec![
        ("buffer", Json::num(0.0)),
        ("byteOffset", Json::num(off as f64)),
        ("byteLength", Json::num(ibytes.len() as f64)),
        ("target", Json::num(34963.0)),
    ]));
    let view = views.len() - 1;
    let accs = doc.array_mut("accessors");
    accs.push(Json::obj(vec![
        ("bufferView", Json::num(view as f64)),
        ("componentType", Json::num(5123.0)),
        ("count", Json::num(indices.len() as f64)),
        ("type", Json::str("SCALAR")),
    ]));
    let idx_acc = accs.len() - 1;
    let ibm_rows: Vec<Vec<f64>> = world
        .iter()
        .map(|w| w.inverse().unwrap().to_gltf_matrix())
        .collect();
    let ibm = doc.push_float_accessor(&ibm_rows, "MAT4", false);
    nodes.push(Json::obj(vec![
        ("name", Json::str("Body")),
        ("mesh", Json::num(0.0)),
        ("skin", Json::num(0.0)),
    ]));
    doc.json.set("nodes", Json::Arr(nodes));
    doc.json.set(
        "meshes",
        Json::Arr(vec![Json::obj(vec![
            ("name", Json::str("Body")),
            (
                "primitives",
                Json::Arr(vec![Json::obj(vec![
                    (
                        "attributes",
                        Json::obj(vec![
                            ("POSITION", Json::num(pos_acc as f64)),
                            ("JOINTS_0", Json::num(j_acc as f64)),
                            ("WEIGHTS_0", Json::num(w_acc as f64)),
                        ]),
                    ),
                    ("indices", Json::num(idx_acc as f64)),
                ])]),
            ),
        ])]),
    );
    doc.json.set(
        "skins",
        Json::Arr(vec![Json::obj(vec![
            (
                "joints",
                Json::Arr((0..n).map(|j| Json::num(j as f64)).collect()),
            ),
            ("inverseBindMatrices", Json::num(ibm as f64)),
        ])]),
    );
    doc.json.set(
        "scenes",
        Json::Arr(vec![Json::obj(vec![(
            "nodes",
            Json::Arr(vec![Json::num(0.0), Json::num(n as f64)]),
        )])]),
    );
    doc.json.set("scene", Json::num(0.0));
    if walk {
        add_walk(
            &mut doc,
            &world,
            &lay.iter().map(|(r, _, _)| *r).collect::<Vec<_>>(),
        )?;
    }
    Ok(doc)
}

/// One-second walk: legs and arms swing about the character's left axis,
/// knees bend on the passing pose, hips travel 1.2 m forward and bob.
fn add_walk(doc: &mut Document, world: &[Affine], roles: &[&str]) -> Result<(), String> {
    use crate::detmath::sin;
    use crate::rig::{Rig, Trs};
    let rig = Rig::from_doc(doc)?;
    let frames = 31;
    let left = Vec3::new(1.0, 0.0, 0.0);
    let swings: Vec<(&str, f64, f64)> = vec![
        ("thigh.L", 30.0, 0.0),
        ("thigh.R", 30.0, 0.5),
        ("shin.L", -25.0, 0.25),
        ("shin.R", -25.0, 0.75),
        ("upper_arm.L", 20.0, 0.5),
        ("upper_arm.R", 20.0, 0.0),
        ("spine2", 4.0, 0.25),
    ];
    let mut nodes = Vec::new();
    let mut tracks: Vec<Vec<Trs>> = Vec::new();
    let hips = roles.iter().position(|r| *r == "hips").unwrap();
    let mut times = Vec::new();
    for f in 0..frames {
        times.push(f as f64 / 30.0);
    }
    for (role, amp, phase) in &swings {
        let node = roles.iter().position(|r| r == role).unwrap();
        let rw = world[node].rotation();
        let track: Vec<Trs> = times
            .iter()
            .map(|t| {
                let ang = amp.to_radians() * sin(2.0 * std::f64::consts::PI * (t + phase));
                let w = Quat::from_axis_angle(left, ang);
                let mut trs = rig.rest[node];
                trs.r = trs.r.mul(rw.conj().mul(w).mul(rw));
                trs
            })
            .collect();
        nodes.push(node);
        tracks.push(track);
    }
    let parent_inv = rig.rest_world[rig.parent[hips].unwrap()].inverse().unwrap();
    let hip_track: Vec<Trs> = times
        .iter()
        .map(|t| {
            let mut trs = rig.rest[hips];
            let pos =
                world[hips].t + Vec3::new(0.0, 0.02 * sin(4.0 * std::f64::consts::PI * t), 1.2 * t);
            trs.t = parent_inv.apply(pos);
            trs
        })
        .collect();
    nodes.push(hips);
    tracks.push(hip_track);
    crate::animate::write_animation(doc, "walk", &times, &nodes, &tracks, Some(hips))
}
