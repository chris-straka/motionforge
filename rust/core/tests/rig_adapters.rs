// GLB rig adapters: standardize, animate (retarget), pose test, on the
// procedural fixtures (`motion_core::fixture`).

use motion_core::animate::{retarget_clips, Mapped};
use motion_core::fixture::{humanoid, Naming};
use motion_core::glb::{Affine, Document};
use motion_core::humanoid::{canonical, map_rig, REQUIRED};
use motion_core::json::Json;
use motion_core::math::Vec3;
use motion_core::posetest;
use motion_core::rig::{Animation, Rig};
use motion_core::standardize::{standardize, Outcome};

fn done(o: Outcome) -> (Document, motion_core::standardize::Report) {
    match o {
        Outcome::Done(d, r) => (d, r),
        Outcome::Refused(_, why) => panic!("refused: {}", why),
    }
}

/// Rest-pose skinned vertex positions (what the mesh looks like).
fn skinned_positions(doc: &Document) -> Vec<Vec3> {
    let rig = Rig::from_doc(doc).unwrap();
    let mut out = Vec::new();
    for node in doc.array("nodes") {
        let (Some(mesh), Some(skin)) = (
            node.get("mesh").and_then(Json::as_usize),
            node.get("skin").and_then(Json::as_usize),
        ) else {
            continue;
        };
        let skin_json = &doc.array("skins")[skin];
        let ibms = doc
            .read_accessor(
                skin_json
                    .get("inverseBindMatrices")
                    .unwrap()
                    .as_usize()
                    .unwrap(),
            )
            .unwrap();
        let joints = &rig.skins[skin];
        for prim in doc.array("meshes")[mesh]
            .get("primitives")
            .unwrap()
            .as_arr()
            .unwrap()
        {
            let attrs = prim.get("attributes").unwrap();
            let pos = doc
                .read_accessor(attrs.get("POSITION").unwrap().as_usize().unwrap())
                .unwrap();
            let j = doc
                .read_accessor(attrs.get("JOINTS_0").unwrap().as_usize().unwrap())
                .unwrap();
            let w = doc
                .read_accessor(attrs.get("WEIGHTS_0").unwrap().as_usize().unwrap())
                .unwrap();
            for v in 0..pos.len() {
                let p = Vec3::new(pos[v][0], pos[v][1], pos[v][2]);
                let mut acc = Vec3::ZERO;
                for c in 0..4 {
                    if w[v][c] == 0.0 {
                        continue;
                    }
                    let ji = j[v][c] as usize;
                    let m = rig.rest_world[joints[ji]].mul(&Affine::from_gltf_matrix(&ibms[ji]));
                    acc = acc + m.apply(p).scale(w[v][c]);
                }
                out.push(acc);
            }
        }
    }
    out
}

fn joint_names(doc: &Document) -> Vec<String> {
    let rig = Rig::from_doc(doc).unwrap();
    rig.joints().iter().map(|&j| rig.names[j].clone()).collect()
}

fn world_heads(doc: &Document) -> Vec<(String, Vec3)> {
    let rig = Rig::from_doc(doc).unwrap();
    rig.joints()
        .iter()
        .map(|&j| (rig.names[j].clone(), rig.head(j)))
        .collect()
}

#[test]
fn standardize_every_naming_style_keeps_the_mesh_in_place() {
    for naming in [Naming::Mixamo, Naming::Plain, Naming::Def] {
        for twisted in [false, true] {
            let src = humanoid(naming, twisted, true).unwrap();
            let before = skinned_positions(&src);
            let (out, report) = done(standardize(&src, "humanoid").unwrap());
            // Round-trip through bytes like the CLI does.
            let out = Document::parse(&out.to_bytes().unwrap()).unwrap();
            let names = joint_names(&out);
            assert!(
                names.iter().all(|n| n.starts_with("DEF-")),
                "{:?}: {:?}",
                naming,
                names
            );
            for r in REQUIRED {
                assert!(names.iter().any(|n| n == r), "{:?} missing {}", naming, r);
            }
            assert_eq!(report.animations_dropped, 1);
            assert!(out.json.get("animations").is_none());
            let after = skinned_positions(&out);
            assert_eq!(before.len(), after.len());
            for (a, b) in before.iter().zip(&after) {
                assert!(
                    a.approx_eq(*b, 1e-5),
                    "{:?} twisted={} mesh moved: {:?} -> {:?}",
                    naming,
                    twisted,
                    a,
                    b
                );
            }
            // Canonical hierarchy: thighs hang from the hips, arms from shoulders.
            let rig = Rig::from_doc(&out).unwrap();
            let parent_of =
                |n: &str| rig.parent[rig.find(n).unwrap()].map(|p| rig.names[p].clone());
            assert_eq!(parent_of("DEF-thigh.L").as_deref(), Some("DEF-spine"));
            assert_eq!(
                parent_of("DEF-upper_arm.R").as_deref(),
                Some("DEF-shoulder.R")
            );
            assert_eq!(
                parent_of("DEF-shoulder.L").as_deref(),
                Some("DEF-spine.003")
            );
            // Joint world rest positions survive reparenting.
            let heads_src = world_heads(&src);
            let map = map_rig(&Rig::from_doc(&src).unwrap());
            for (name, head) in world_heads(&out) {
                // Twist/helper bones sit on their driver's joint.
                let name = motion_core::helpers::driver_of(&name).unwrap_or(name);
                let node = map.node_of(&name).unwrap();
                let src_rig = Rig::from_doc(&src).unwrap();
                assert!(src_rig.head(node).approx_eq(head, 1e-6), "{} moved", name);
                let _ = &heads_src;
            }
            // Weights stay normalized with at most four influences.
            for prim in out.array("meshes")[0]
                .get("primitives")
                .unwrap()
                .as_arr()
                .unwrap()
            {
                let w = out
                    .read_accessor(
                        prim.get("attributes")
                            .unwrap()
                            .get("WEIGHTS_0")
                            .unwrap()
                            .as_usize()
                            .unwrap(),
                    )
                    .unwrap();
                for row in w {
                    let s: f64 = row.iter().sum();
                    assert!((s - 1.0).abs() < 1e-4, "weights sum {}", s);
                }
            }
        }
    }
}

#[test]
fn standardize_maps_mixamo_names_and_merges_extras() {
    let src = humanoid(Naming::Mixamo, false, false).unwrap();
    let (_, report) = done(standardize(&src, "humanoid").unwrap());
    let renamed: Vec<(&str, &str)> = report
        .renamed
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    assert!(renamed.contains(&("mixamorig:LeftArm", "DEF-upper_arm.L")));
    assert!(renamed.contains(&("mixamorig:Spine2", "DEF-spine.003")));
    assert!(renamed.contains(&("mixamorig:LeftHandMiddle1", "DEF-f_middle.01.L")));
    assert!(report
        .merged
        .iter()
        .any(|(a, b)| a == "mixamorig:LeftHandPinky4" && b == "DEF-hand.L"));
    assert!(report.merged.iter().any(|(a, _)| a == "Root"));
    // 24 HLL bones + the four twist/helpers.
    assert_eq!(report.joints_out, 28);
    assert_eq!(
        report.helpers,
        vec![
            "DEF-thigh_twist.L",
            "DEF-upper_arm_twist.L",
            "DEF-thigh_twist.R",
            "DEF-upper_arm_twist.R"
        ]
    );
}

#[test]
fn standardize_is_idempotent_and_deterministic() {
    let src = humanoid(Naming::Plain, true, false).unwrap();
    let (once, _) = done(standardize(&src, "humanoid").unwrap());
    let again = done(standardize(&src, "humanoid").unwrap()).0;
    assert_eq!(once.to_bytes().unwrap(), again.to_bytes().unwrap());
    let (twice, report) = done(standardize(&once, "humanoid").unwrap());
    assert!(report.renamed.is_empty() && report.reparented.is_empty() && report.merged.is_empty());
    assert_eq!(joint_names(&once), joint_names(&twice));
}

#[test]
fn standardize_refuses_rigs_without_the_core() {
    let mut src = humanoid(Naming::Mixamo, false, false).unwrap();
    let nodes = src.array_mut("nodes");
    for n in nodes.iter_mut() {
        if n.get("name").and_then(Json::as_str) == Some("mixamorig:LeftHand") {
            n.set("name", Json::str("mystery"));
        }
    }
    match standardize(&src, "humanoid").unwrap() {
        Outcome::Refused(r, why) => {
            assert!(why.contains("DEF-hand.L"), "{}", why);
            assert_eq!(r.missing, vec!["DEF-hand.L".to_string()]);
        }
        Outcome::Done(..) => panic!("should refuse"),
    }
    let unrigged = Document {
        json: motion_core::json::parse(r#"{"asset":{"version":"2.0"},"nodes":[{"name":"a"}]}"#)
            .unwrap(),
        bin: vec![],
    };
    assert!(matches!(
        standardize(&unrigged, "humanoid").unwrap(),
        Outcome::Refused(..)
    ));
}

#[test]
fn non_humanoids_only_get_the_def_prefix() {
    let src = humanoid(Naming::Plain, false, false).unwrap();
    let (out, report) = done(standardize(&src, "quadruped").unwrap());
    let names = joint_names(&out);
    assert!(names.iter().all(|n| n.starts_with("DEF-")));
    assert!(names.contains(&"DEF-UpperArm_L".to_string()));
    assert_eq!(report.joints_in, report.joints_out);
    assert_eq!(skinned_positions(&src), skinned_positions(&out));
}

#[test]
fn canonical_table_covers_rigforge_mobile_set() {
    let names: Vec<String> = canonical().into_iter().map(|(n, _)| n).collect();
    for n in [
        "DEF-spine.005",
        "DEF-palm.03.R",
        "DEF-f_pinky.03.L",
        "DEF-breast.L",
        "DEF-pelvis.R",
        "DEF-toe.R",
    ] {
        assert!(names.contains(&n.to_string()), "{}", n);
    }
}

/// Retarget a walk onto a standardized copy of the same body: every
/// shared joint must land where the source joint is, frame by frame.
#[test]
fn retarget_onto_same_body_reproduces_joint_positions() {
    let src = humanoid(Naming::Mixamo, true, true).unwrap();
    let body = humanoid(Naming::Plain, false, false).unwrap();
    let (target, _) = done(standardize(&body, "humanoid").unwrap());
    let (animated, reports) =
        retarget_clips(&target, &[("walk.glb".into(), src.clone())], 30.0).unwrap();
    let animated = Document::parse(&animated.to_bytes().unwrap()).unwrap();
    assert_eq!(reports.len(), 1);
    let r = &reports[0];
    assert_eq!(r.frames, 31);
    assert!(r.max_error_deg < 1e-4, "self-check {}", r.max_error_deg);
    assert!(
        (r.root_travel_m - 1.2).abs() < 1e-3,
        "travel {}",
        r.root_travel_m
    );
    let s = Mapped::new(&src).unwrap();
    let t = Mapped::new(&animated).unwrap();
    let sa = &Animation::load_all(&src).unwrap()[0];
    let ta = &Animation::load_all(&animated).unwrap()[0];
    assert_eq!(ta.name, "walk");
    let mut worst: f64 = 0.0;
    for f in 0..31 {
        let time = f as f64 / 30.0;
        let sw = s.rig.world(&sa.sample(&s.rig, time));
        let tw = t.rig.world(&ta.sample(&t.rig, time));
        for (tn, name) in &t.map.pairs {
            if let Some(sn) = s.map.node_of(name) {
                worst = worst.max((sw[sn].t - tw[*tn].t).length());
            }
        }
    }
    // f32 storage of keys bounds this, not the math.
    assert!(worst < 2e-4, "joint drift {} m", worst);
}

#[test]
fn retarget_turns_a_source_facing_backwards() {
    // Source faces -Z: rotate the whole fixture 180 deg about Y.
    let mut src = humanoid(Naming::Mixamo, false, true).unwrap();
    let nodes = src.array_mut("nodes");
    nodes[0].set(
        "rotation",
        Json::Arr(vec![
            Json::num(0.0),
            Json::num(1.0),
            Json::num(0.0),
            Json::num(0.0),
        ]),
    );
    let body = humanoid(Naming::Def, false, false).unwrap();
    let (target, _) = done(standardize(&body, "humanoid").unwrap());
    let (animated, reports) = retarget_clips(&target, &[("back.glb".into(), src)], 30.0).unwrap();
    assert!(reports[0].max_error_deg < 1e-6);
    let t = Mapped::new(&animated).unwrap();
    let ta = &Animation::load_all(&animated).unwrap()[0];
    let hips = t.map.node_of("DEF-spine").unwrap();
    let end = t.rig.world(&ta.sample(&t.rig, 1.0))[hips].t;
    // The target still walks toward its own front (+Z).
    assert!(end.z > 1.1, "walked to {:?}", end);
}

#[test]
fn pose_sheet_animations_bend_what_they_say() {
    let body = humanoid(Naming::Mixamo, true, false).unwrap();
    let (target, _) = done(standardize(&body, "humanoid").unwrap());
    let (posed, report) = posetest::build(&target, &[]).unwrap();
    let posed = Document::parse(&posed.to_bytes().unwrap()).unwrap();
    let anims = Animation::load_all(&posed).unwrap();
    assert_eq!(anims.len(), 12);
    assert_eq!(report.get("poses").unwrap().as_arr().unwrap().len(), 12);
    let m = Mapped::new(&posed).unwrap();
    let dir = |world: &[Affine], a: &str, b: &str| {
        let (a, b) = (m.map.node_of(a).unwrap(), m.map.node_of(b).unwrap());
        (world[b].t - world[a].t).normalized()
    };
    let rest = m.rig.world(&vec![None; m.rig.names.len()]);
    let angle = |anim: &Animation, a: &str, b: &str| {
        let w = m.rig.world(&anim.sample(&m.rig, 0.0));
        dir(&rest, a, b)
            .dot(dir(&w, a, b))
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees()
    };
    let find = |label: &str| anims.iter().find(|a| a.name.contains(label)).unwrap();
    assert!((angle(find("elbows 140"), "DEF-forearm.L", "DEF-hand.L") - 140.0).abs() < 0.01);
    assert!((angle(find("arms up 70"), "DEF-upper_arm.R", "DEF-forearm.R") - 70.0).abs() < 0.01);
    assert!((angle(find("knee up"), "DEF-thigh.L", "DEF-shin.L") - 90.0).abs() < 0.01);
    assert!(angle(find("rest"), "DEF-thigh.L", "DEF-shin.L") < 1e-3);
    // Arms up raises the hands.
    let w = m.rig.world(&find("arms up").sample(&m.rig, 0.0));
    let hand = m.map.node_of("DEF-hand.L").unwrap();
    assert!(w[hand].t.y > rest[hand].t.y + 0.2);
}

#[test]
fn pose_sheet_takes_clip_samples() {
    let src = humanoid(Naming::Plain, false, true).unwrap();
    let body = humanoid(Naming::Mixamo, false, false).unwrap();
    let (target, _) = done(standardize(&body, "humanoid").unwrap());
    let (animated, _) = retarget_clips(&target, &[("walk".into(), src)], 30.0).unwrap();
    let samples = posetest::clip_samples(&animated, 3, &[0.25, 0.6]).unwrap();
    assert_eq!(samples.len(), 2);
    assert_eq!(samples[0].0, "walk @25%");
    let (posed, _) = posetest::build(&target, &samples).unwrap();
    assert_eq!(Animation::load_all(&posed).unwrap().len(), 14);
}

/// A clip made on an A-pose rig must keep its bone directions on a
/// T-pose rig (rest alignment), not lift the arms by the rest difference.
#[test]
fn retarget_aligns_rest_poses() {
    use motion_core::fixture::humanoid_posed;
    let src = humanoid_posed(Naming::Mixamo, false, true, 45.0).unwrap();
    let body = humanoid(Naming::Plain, true, false).unwrap(); // T-pose, twisted joints
    let (target, _) = done(standardize(&body, "humanoid").unwrap());
    let (animated, reports) =
        retarget_clips(&target, &[("apose-walk".into(), src.clone())], 30.0).unwrap();
    assert!(
        reports[0].max_error_deg < 1e-4,
        "{}",
        reports[0].max_error_deg
    );
    let s = Mapped::new(&src).unwrap();
    let t = Mapped::new(&animated).unwrap();
    let sa = &Animation::load_all(&src).unwrap()[0];
    let ta = &Animation::load_all(&animated).unwrap()[0];
    let dir = |m: &Mapped, w: &[Affine], a: &str, b: &str| {
        let (a, b) = (m.map.node_of(a).unwrap(), m.map.node_of(b).unwrap());
        (w[b].t - w[a].t).normalized()
    };
    let mut worst: f64 = 0.0;
    for f in [0, 7, 15, 22, 30] {
        let time = f as f64 / 30.0;
        let sw = s.rig.world(&sa.sample(&s.rig, time));
        let tw = t.rig.world(&ta.sample(&t.rig, time));
        for (a, b) in [
            ("DEF-upper_arm.L", "DEF-forearm.L"),
            ("DEF-forearm.R", "DEF-hand.R"),
            ("DEF-thigh.L", "DEF-shin.L"),
        ] {
            let cos = dir(&s, &sw, a, b).dot(dir(&t, &tw, a, b)).clamp(-1.0, 1.0);
            worst = worst.max(cos.acos().to_degrees());
        }
    }
    assert!(worst < 0.05, "bone directions differ by {} deg", worst);
}

/// Rotation of `node` relative to its parent, as a change from rest.
fn local_delta(rig: &Rig, world: &[Affine], node: usize) -> motion_core::math::Quat {
    let p = rig.parent[node].unwrap();
    let posed = world[p].inverse().unwrap().mul(&world[node]).rotation();
    let rest = rig.rest_world[p]
        .inverse()
        .unwrap()
        .mul(&rig.rest_world[node])
        .rotation();
    posed.mul(rest.conj())
}

#[test]
fn standardize_adds_unweighted_helpers_on_their_drivers() {
    let body = humanoid(Naming::Mixamo, true, false).unwrap();
    let (out, _) = done(standardize(&body, "humanoid").unwrap());
    let out = Document::parse(&out.to_bytes().unwrap()).unwrap();
    let rig = Rig::from_doc(&out).unwrap();
    let skin = &rig.skins[0];
    let ibm = out
        .read_accessor(
            out.array("skins")[0]
                .get("inverseBindMatrices")
                .unwrap()
                .as_usize()
                .unwrap(),
        )
        .unwrap();
    let pairs = motion_core::helpers::present(&rig);
    assert_eq!(pairs.len(), 4);
    for (h, d) in pairs {
        assert_eq!(rig.parent[h], rig.parent[d]);
        assert!(rig.head(h).approx_eq(rig.head(d), 1e-9));
        let (sh, sd) = (
            skin.iter().position(|&j| j == h).unwrap(),
            skin.iter().position(|&j| j == d).unwrap(),
        );
        assert_eq!(ibm[sh], ibm[sd]);
        let extras = out.array("nodes")[h]
            .get("extras")
            .unwrap()
            .get("hll_helper")
            .unwrap();
        assert_eq!(
            extras.get("driver").unwrap().as_str(),
            Some(rig.names[d].as_str())
        );
        assert_eq!(extras.get("share").unwrap().as_f64(), Some(0.5));
        // No vertex weight yet (weightforge's fix assigns it).
        let prim = &out.array("meshes")[0]
            .get("primitives")
            .unwrap()
            .as_arr()
            .unwrap()[0];
        let attrs = prim.get("attributes").unwrap();
        let j = out
            .read_accessor(attrs.get("JOINTS_0").unwrap().as_usize().unwrap())
            .unwrap();
        let w = out
            .read_accessor(attrs.get("WEIGHTS_0").unwrap().as_usize().unwrap())
            .unwrap();
        for (jr, wr) in j.iter().zip(&w) {
            for c in 0..4 {
                assert!(!(jr[c] as usize == sh && wr[c] > 0.0));
            }
        }
    }
}

#[test]
fn clips_and_poses_bake_half_the_driver_into_helpers() {
    let src = humanoid(Naming::Mixamo, true, true).unwrap();
    let body = humanoid(Naming::Plain, false, false).unwrap();
    let (target, _) = done(standardize(&body, "humanoid").unwrap());
    let (animated, _) = retarget_clips(&target, &[("walk".into(), src)], 30.0).unwrap();
    let animated = Document::parse(&animated.to_bytes().unwrap()).unwrap();
    let rig = Rig::from_doc(&animated).unwrap();
    let anim = &Animation::load_all(&animated).unwrap()[0];
    let keyed = anim.animated_nodes();
    let pairs = motion_core::helpers::present(&rig);
    let mut moved: f64 = 0.0;
    for (h, d) in &pairs {
        assert!(keyed.contains(h), "{} not keyed", rig.names[*h]);
        for f in 0..31 {
            let w = rig.world(&anim.sample(&rig, f as f64 / 30.0));
            let (dh, dd) = (local_delta(&rig, &w, *h), local_delta(&rig, &w, *d));
            // Twice the helper's turn is the driver's turn (same axis).
            assert!(
                dh.mul(dh).angle_to(dd) < 1e-4,
                "{} frame {}",
                rig.names[*h],
                f
            );
            moved = moved.max(dd.angle_to(motion_core::math::Quat::IDENTITY));
        }
    }
    assert!(moved > 0.2, "walk barely moves the limbs ({moved} rad)");
    // Pose sheet: arms up 70 turns each upper arm helper 35 degrees.
    let (posed, _) = posetest::build(&target, &[]).unwrap();
    let posed = Document::parse(&posed.to_bytes().unwrap()).unwrap();
    let prig = Rig::from_doc(&posed).unwrap();
    let up = Animation::load_all(&posed)
        .unwrap()
        .into_iter()
        .find(|a| a.name.contains("arms up"))
        .unwrap();
    let w = prig.world(&up.sample(&prig, 0.0));
    let h = prig.find("DEF-upper_arm_twist.L").unwrap();
    let deg = local_delta(&prig, &w, h)
        .angle_to(motion_core::math::Quat::IDENTITY)
        .to_degrees();
    assert!((deg - 35.0).abs() < 0.01, "helper turned {deg}");
}
