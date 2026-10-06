// Contact pass, sockets and clip picks on the procedural fixture.

use motion_core::animate::{parse_pick, retarget_picked};
use motion_core::contact::{fix_clip, solve_arm, Body, Options};
use motion_core::fixture::{humanoid, Naming};
use motion_core::glb::Document;
use motion_core::math::Vec3;
use motion_core::rig::{Rig, Trs};
use motion_core::standardize::{standardize, Outcome};

fn target() -> Document {
    let body = humanoid(Naming::Plain, false, false).unwrap();
    match standardize(&body, "humanoid").unwrap() {
        Outcome::Done(d, _) => Document::parse(&d.to_bytes().unwrap()).unwrap(),
        Outcome::Refused(_, why) => panic!("{why}"),
    }
}

/// A clip whose right hand travels from rest into the middle of the chest
/// and back (11 frames at 30 fps).
fn through_chest(rig: &Rig) -> (Vec<Vec<Option<Trs>>>, Vec<f64>) {
    let find = |n: &str| rig.find(n).unwrap();
    let (u, f, h) = (
        find("DEF-upper_arm.R"),
        find("DEF-forearm.R"),
        find("DEF-hand.R"),
    );
    // Into the chest box, off its faces (the fixture's boxes share their
    // half-width, so a hand exactly on the bone line sits on the surface).
    let chest = rig.head(find("DEF-spine.002")) + Vec3::new(0.0, 0.05, 0.01);
    let rest = vec![None; rig.names.len()];
    let w0 = rig.world(&rest)[h].t;
    let mut frames = Vec::new();
    let mut times = Vec::new();
    for k in 0..11 {
        let s = 1.0 - ((k as f64 - 5.0) / 5.0).abs(); // 0 -> 1 -> 0
        let mut locals = rest.clone();
        let target = w0 + (chest - w0).scale(s);
        solve_arm(rig, &mut locals, u, f, h, target);
        frames.push(locals);
        times.push(k as f64 / 30.0);
    }
    (frames, times)
}

#[test]
fn contact_pass_moves_the_hand_out_of_the_chest_and_eases() {
    let doc = target();
    let rig = Rig::from_doc(&doc).unwrap();
    let opts = Options {
        weapon_side: None,
        ..Options::default()
    };
    let body = Body::build(&doc, &rig, &opts).unwrap().expect("proxies");
    let (mut frames, times) = through_chest(&rig);
    let original = frames.clone();
    let report = fix_clip(&body, &rig, &mut frames, &times, "attack_test", &opts);
    let right = report.arms.iter().find(|a| a.side == "R").unwrap();
    assert!(right.mesh_frames_before > 0, "{right:?}");
    assert_eq!(right.mesh_frames_after, 0, "{right:?}");
    assert!(right.frames_changed > 0);
    // The first frame is far from any contact (more than the ramp away):
    // untouched. The wrist path keeps its start and end.
    let h = rig.find("DEF-hand.R").unwrap();
    let w = |f: &Vec<Option<Trs>>| rig.world(f)[h].t;
    assert!((w(&frames[0]) - w(&original[0])).length() < 1e-9);
    assert!((w(&frames[10]) - w(&original[10])).length() < 1e-9);
    // Eased: the correction grows toward the deepest frame.
    let shift: Vec<f64> = (0..11)
        .map(|k| (w(&frames[k]) - w(&original[k])).length())
        .collect();
    assert!(shift[5] >= shift[3] && shift[3] >= shift[1], "{shift:?}");
    // The hand keeps its world rotation (IK leaves it alone).
    let r0 = rig.world(&original[5])[h].rotation();
    let r1 = rig.world(&frames[5])[h].rotation();
    assert!(r0.angle_to(r1) < 1e-6);
    // Same input, same output.
    let mut again = original.clone();
    fix_clip(&body, &rig, &mut again, &times, "attack_test", &opts);
    assert_eq!(format!("{again:?}"), format!("{frames:?}"));
}

#[test]
fn clean_clips_are_untouched() {
    let doc = target();
    let rig = Rig::from_doc(&doc).unwrap();
    let opts = Options::default();
    let body = Body::build(&doc, &rig, &opts).unwrap().unwrap();
    let mut frames = vec![vec![None; rig.names.len()]; 5];
    let times: Vec<f64> = (0..5).map(|k| k as f64 / 30.0).collect();
    let report = fix_clip(&body, &rig, &mut frames, &times, "idle", &opts);
    assert!(!report.changed());
    assert!(frames.iter().all(|f| f.iter().all(Option::is_none)));
}

#[test]
fn standardize_adds_hand_sockets_once() {
    let doc = target();
    let rig = Rig::from_doc(&doc).unwrap();
    for side in ["L", "R"] {
        let s = rig.find(&format!("Socket_Hand_{side}")).expect("socket");
        let hand = rig.find(&format!("DEF-hand.{side}")).unwrap();
        assert_eq!(rig.parent[s], Some(hand));
        // Not a joint: the skin is unchanged by it.
        assert!(!rig.joints().contains(&s));
        // Blade (+Y) is a unit direction.
        let y = rig.rest_world[s].apply_linear(Vec3::new(0.0, 1.0, 0.0));
        assert!((y.length() - 1.0).abs() < 1e-6);
    }
    let again = match standardize(&doc, "humanoid").unwrap() {
        Outcome::Done(d, r) => {
            assert!(r.sockets.is_empty());
            d
        }
        Outcome::Refused(_, why) => panic!("{why}"),
    };
    let n = |d: &Document| {
        Rig::from_doc(d)
            .unwrap()
            .names
            .iter()
            .filter(|n| n.starts_with("Socket_"))
            .count()
    };
    assert_eq!(n(&again), 2);
}

#[test]
fn pick_selects_renames_and_orders_clips() {
    assert!(parse_pick("").is_err());
    let p = parse_pick("walk=stroll, src.glb:walk=walk2,run").unwrap();
    assert_eq!(p[0], ("walk".into(), "stroll".into()));
    assert_eq!(p[1], ("src.glb:walk".into(), "walk2".into()));
    assert_eq!(p[2], ("run".into(), "run".into()));
    let src = humanoid(Naming::Mixamo, false, true).unwrap();
    let tgt = target();
    let pick = parse_pick("src.glb:walk=b,walk=a").unwrap();
    let (out, reports) = retarget_picked(
        &tgt,
        &[("src.glb".into(), src.clone())],
        30.0,
        Some(&pick),
        None,
    )
    .unwrap();
    let names: Vec<&str> = reports.iter().map(|r| r.name.as_str()).collect();
    assert_eq!(names, ["b", "a"]);
    assert_eq!(out.array("animations").len(), 2);
    let bad = parse_pick("nope").unwrap();
    assert!(retarget_picked(&tgt, &[("src.glb".into(), src)], 30.0, Some(&bad), None).is_err());
}
