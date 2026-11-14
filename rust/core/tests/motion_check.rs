// Motion score on synthetic clips with known answers (procedural fixture).

use motion_core::contact::{solve_arm, Options};
use motion_core::fixture::{humanoid, Naming};
use motion_core::glb::Document;
use motion_core::math::{Quat, Vec3};
use motion_core::motioncheck::{check_doc, score, Subject, ATTACK_WORDS};
use motion_core::rig::{Rig, Trs};
use motion_core::standardize::{standardize, Outcome};

fn target() -> Document {
    let body = humanoid(Naming::Plain, false, false).unwrap();
    match standardize(&body, "humanoid").unwrap() {
        Outcome::Done(d, _) => Document::parse(&d.to_bytes().unwrap()).unwrap(),
        Outcome::Refused(_, why) => panic!("{why}"),
    }
}

fn subject() -> Subject {
    Subject::new(&target(), &Options::default()).unwrap()
}

fn times(n: usize) -> Vec<f64> {
    (0..n).map(|f| f as f64 / 30.0).collect()
}

fn still(rig: &Rig, n: usize) -> Vec<Vec<Option<Trs>>> {
    vec![vec![None; rig.names.len()]; n]
}

#[test]
fn a_still_clip_scores_clean() {
    let s = subject();
    let frames = still(&s.rig, 20);
    let c = score(&s, "idle", &times(20), &frames, false);
    assert_eq!(c.feet.skating_plants, 0);
    assert_eq!(c.feet.plants, 2);
    assert_eq!(c.feet.sink_frames + c.feet.float_frames, 0);
    assert!(c.feet.worst_plant_m < 1e-9);
    assert_eq!(c.through.as_ref().unwrap().mesh_frames, 0);
    assert!(c.jerk.iter().all(|j| j.peak < 1e-6), "{:?}", c.jerk);
    assert!(c.root_jerk.peak < 1e-6);
}

#[test]
fn a_slid_foot_skates_by_the_slide() {
    let s = subject();
    let rig = &s.rig;
    let n = 20;
    let mut frames = still(rig, n);
    let (th, sh, ft) = (
        rig.find("DEF-thigh.L").unwrap(),
        rig.find("DEF-shin.L").unwrap(),
        rig.find("DEF-foot.L").unwrap(),
    );
    let ankle = rig.head(ft);
    let slide = 0.08;
    for (f, fr) in frames.iter_mut().enumerate() {
        let x = slide * f as f64 / (n - 1) as f64;
        solve_arm(rig, fr, th, sh, ft, ankle + Vec3::new(0.0, 0.0, x));
    }
    let c = score(&s, "idle", &times(n), &frames, false);
    assert_eq!(c.feet.plants, 2);
    assert_eq!(c.feet.skating_plants, 1);
    assert_eq!(c.feet.skate_frames, n);
    assert_eq!(c.feet.worst_plant, "L");
    // Reported at reference height: slide / k.
    let got = c.feet.worst_plant_m * s.k;
    assert!((got - slide).abs() < 1e-3, "{got}");
    // Under 2.5 cm (at reference height) is not a skate.
    let mut small = still(rig, n);
    for (f, fr) in small.iter_mut().enumerate() {
        let x = 0.02 * s.k * f as f64 / (n - 1) as f64;
        solve_arm(rig, fr, th, sh, ft, ankle + Vec3::new(0.0, 0.0, x));
    }
    let c = score(&s, "idle", &times(n), &small, false);
    assert_eq!(c.feet.skating_plants, 0);
    // A foot moving fast near the ground is a step, not a plant.
    let m = 10;
    let mut fast = still(rig, m);
    for (f, fr) in fast.iter_mut().enumerate() {
        let x = -0.2 + 0.4 * f as f64 / (m - 1) as f64; // 1.33 m/s
        solve_arm(rig, fr, th, sh, ft, ankle + Vec3::new(0.0, 0.01, x));
    }
    let c = score(&s, "idle", &times(m), &fast, false);
    assert_eq!(c.feet.skating_plants, 0, "{:?}", c.feet);
}

#[test]
fn a_sunk_and_a_lifted_body_are_reported() {
    let s = subject();
    let rig = &s.rig;
    let hips = rig.find("DEF-spine").unwrap();
    let n = 10;
    let at = |dy: f64| {
        let mut frames = still(rig, n);
        for fr in frames.iter_mut() {
            let mut t = rig.rest[hips];
            // Hips' parent is the root at the origin: local = world shift.
            let up = Vec3::new(0.0, dy, 0.0);
            let shift = match rig.parent[hips] {
                Some(p) => rig.rest_world[p].inverse().unwrap().apply_linear(up),
                None => up,
            };
            t.t = t.t + shift;
            fr[hips] = Some(t);
        }
        frames
    };
    let c = score(&s, "idle", &times(n), &at(-0.04), false);
    assert_eq!(c.feet.sink_frames, n);
    assert!((c.feet.sink_m * s.k - 0.04).abs() < 1e-6, "{:?}", c.feet);
    let c = score(&s, "idle", &times(n), &at(0.2), false);
    assert_eq!(c.feet.float_frames, n);
    assert!((c.feet.float_m * s.k - 0.2).abs() < 1e-6);
}

#[test]
fn a_hand_through_the_chest_is_caught() {
    let s = subject();
    let rig = &s.rig;
    let find = |n: &str| rig.find(n).unwrap();
    let (u, f, h) = (
        find("DEF-upper_arm.R"),
        find("DEF-forearm.R"),
        find("DEF-hand.R"),
    );
    let chest = rig.head(find("DEF-spine.002")) + Vec3::new(0.0, 0.05, 0.01);
    let rest = vec![None; rig.names.len()];
    let w0 = rig.world(&rest)[h].t;
    let frames: Vec<Vec<Option<Trs>>> = (0..11)
        .map(|k| {
            let s = 1.0 - ((k as f64 - 5.0) / 5.0).abs();
            let mut l = rest.clone();
            solve_arm(rig, &mut l, u, f, h, w0 + (chest - w0).scale(s));
            l
        })
        .collect();
    let c = score(&s, "idle", &times(11), &frames, false);
    let t = c.through.unwrap();
    assert!(t.mesh_frames >= 1, "{t:?}");
    assert!(t.mesh_depth_m > 0.005, "{t:?}");
    assert_eq!(t.worst_frame, 5);
    assert!(t.contacts.iter().any(|c| c.contains(".R>")), "{t:?}");
}

#[test]
fn jitter_shows_as_jerk_on_its_chain_only() {
    let s = subject();
    let rig = &s.rig;
    let fore = rig.find("DEF-forearm.R").unwrap();
    let hand = rig.find("DEF-hand.R").unwrap();
    let along = rig.rest[hand].t;
    let len = along.length();
    let perp = along.cross(Vec3::new(0.0, 0.0, 1.0));
    let perp = if perp.length() > 1e-6 {
        perp.normalized()
    } else {
        along.cross(Vec3::new(1.0, 0.0, 0.0)).normalized()
    };
    let theta: f64 = 0.02;
    let n = 30;
    let mut frames = still(rig, n);
    for (f, fr) in frames.iter_mut().enumerate() {
        let sign = if f % 2 == 0 { 1.0 } else { -1.0 };
        let mut t = rig.rest[fore];
        t.r = t.r.mul(Quat::from_axis_angle(perp, sign * theta));
        fr[fore] = Some(t);
    }
    let c = score(&s, "idle", &times(n), &frames, false);
    let get = |name: &str| c.jerk.iter().find(|j| j.chain == name).unwrap();
    // Hand alternates +-len*sin(theta): third difference 8x that.
    let dt = 1.0 / 30.0;
    let want = 8.0 * len * theta.sin() / (dt * dt * dt) / s.k;
    let arm = get("arm.R");
    assert!(
        (arm.peak - want).abs() / want < 1e-3,
        "{} vs {want}",
        arm.peak
    );
    // Forearm head (elbow) is still, the hand jitters: mean is half.
    assert!((arm.mean - want / 2.0).abs() / want < 1e-3, "{}", arm.mean);
    assert!(get("arm.L").peak < 1e-6);
    assert!(get("leg.L").peak < 1e-6 && get("spine").peak < 1e-6);
    assert!(c.root_jerk.peak < 1e-6);
}

#[test]
fn a_swing_scores_windup_strike_and_hold() {
    let s = subject();
    let rig = &s.rig;
    let find = |n: &str| rig.find(n).unwrap();
    let (u, f, h) = (
        find("DEF-upper_arm.R"),
        find("DEF-forearm.R"),
        find("DEF-hand.R"),
    );
    let rest = vec![None; rig.names.len()];
    let w0 = rig.world(&rest)[h].t;
    // Hand path along +Z (forward): back 10 cm over frames 0..10, then
    // 40 cm forward over 10..14, then still to frame 24.
    let z = |k: usize| -> f64 {
        let k = k as f64;
        if k <= 10.0 {
            -0.10 * (k / 10.0)
        } else if k <= 14.0 {
            -0.10 + 0.40 * ((k - 10.0) / 4.0)
        } else {
            0.30
        }
    };
    let n = 25;
    let frames: Vec<Vec<Option<Trs>>> = (0..n)
        .map(|k| {
            let mut l = rest.clone();
            // Off the straight arm's line, so the elbow's bend plane is defined.
            solve_arm(rig, &mut l, u, f, h, w0 + Vec3::new(0.15, -0.08, z(k)));
            l
        })
        .collect();
    let c = score(&s, "attack_test", &times(n), &frames, true);
    let a = c.attack.unwrap();
    assert!((11..=14).contains(&a.peak_frame), "{a:?}");
    assert!((9..=11).contains(&a.load_frame), "{a:?}");
    // Windup: the hand moved back from frame 0 to the load.
    assert!(
        a.windup_s > 8.0 / 30.0 && a.windup_s <= 11.0 / 30.0,
        "{a:?}"
    );
    assert!(a.strike_s > 0.0 && a.strike_s <= 5.0 / 30.0, "{a:?}");
    // Still from frame 15 (speed is a central difference) to the end.
    assert!(a.hold_frames >= 9, "{a:?}");
    // 40 cm in 4 frames = 3 m/s.
    assert!((a.hand_peak_mps * s.k - 3.0).abs() < 0.2, "{a:?}");
    assert!(a.root_travel_m < 1e-9);
}

#[test]
fn check_doc_scores_every_clip_deterministically() {
    // A walk retargeted onto the fixture (its stance foot slides at about
    // 0.5 m/s by design, so plants are not asserted here).
    use motion_core::animate::retarget_with;
    let src = humanoid(Naming::Mixamo, false, true).unwrap();
    let (doc, _) = retarget_with(
        &target(),
        &[("src.glb".into(), src)],
        30.0,
        None,
        None,
        true,
    )
    .unwrap();
    let words: Vec<String> = ATTACK_WORDS.iter().map(|w| w.to_string()).collect();
    let (_, a) = check_doc(&doc, &Options::default(), &[], &words, None).unwrap();
    let (_, b) = check_doc(&doc, &Options::default(), &[], &words, None).unwrap();
    assert_eq!(a.len(), 1);
    assert!(a[0].frames > 10);
    let ja: Vec<String> = a.iter().map(|c| format!("{:?}", c.to_json())).collect();
    let jb: Vec<String> = b.iter().map(|c| format!("{:?}", c.to_json())).collect();
    assert_eq!(ja, jb);
}

#[test]
fn stylize_glb_adds_clips_both_ways_and_keeps_feet() {
    use motion_core::animate::retarget_with;
    use motion_core::stylize_body::{stylize_glb, BodyParams, Mode};
    let src = humanoid(Naming::Mixamo, false, true).unwrap();
    let (doc, _) = retarget_with(
        &target(),
        &[("src.glb".into(), src)],
        30.0,
        None,
        None,
        true,
    )
    .unwrap();
    let words: Vec<String> = ATTACK_WORDS.iter().map(|w| w.to_string()).collect();
    for mode in [Mode::Body, Mode::PerBone] {
        let (out, reps) = stylize_glb(
            &doc,
            &BodyParams::default(),
            mode,
            &[],
            Some("_s"),
            mode == Mode::Body,
        )
        .unwrap();
        assert_eq!(reps.len(), 1);
        let out = Document::parse(&out.to_bytes().unwrap()).unwrap();
        let (_, scores) = check_doc(&out, &Options::default(), &[], &words, None).unwrap();
        assert_eq!(scores.len(), 2);
        assert!(scores[1].name.ends_with("_s"));
        assert_eq!(scores[0].frames, scores[1].frames);
        if mode == Mode::Body {
            // Shared keys. (Feet are not asserted: this walk has no plants
            // to keep, and exaggerating its legs lifts both feet off the
            // floor on most frames; body mode is for attacks.)
            assert!(reps[0].keys.len() > 2, "{:?}", reps[0]);
        }
    }
}

/// World rotation of a node at rest.
fn rest_rot(rig: &Rig, node: usize) -> Quat {
    rig.rest_world[node].rotation()
}

/// Turn `node` (at rest) by world rotation `r` about its head.
fn turned(rig: &Rig, node: usize, r: Quat) -> Trs {
    let parent = rig.parent[node].unwrap();
    let mut t = rig.rest[node];
    t.r = rest_rot(rig, parent)
        .conj()
        .mul(r)
        .mul(rest_rot(rig, node))
        .normalized();
    t
}

#[test]
fn back_and_forth_counts_jitter_not_speed() {
    let s = subject();
    let rig = &s.rig;
    let fore = rig.find("DEF-forearm.R").unwrap();
    let hand = rig.find("DEF-hand.R").unwrap();
    let len = rig.rest[hand].t.length();
    let fwd = motion_core::animate::forward(rig, &s.map);
    let u = rig.head(hand) - rig.head(fore);
    let axis = u.cross(fwd).normalized();
    // Jitter +-theta every frame: the hand goes +-L sin(theta) about its
    // middle, so its acceleration flips by 4 L sin(theta) each frame.
    let theta: f64 = 0.02;
    let n = 30;
    let mut frames = still(rig, n);
    for (f, fr) in frames.iter_mut().enumerate() {
        let sign = if f % 2 == 0 { 1.0 } else { -1.0 };
        fr[fore] = Some(turned(rig, fore, Quat::from_axis_angle(axis, sign * theta)));
    }
    let c = score(&s, "idle", &times(n), &frames, false);
    let arm = c.rough.iter().find(|r| r.chain == "arm.R").unwrap();
    let want = 4.0 * len * theta.sin() / s.k * 1000.0;
    assert!(
        (arm.peak_mm - want).abs() / want < 1e-3,
        "{} vs {want}",
        arm.peak_mm
    );
    assert!(
        (arm.mean_mm - want / 2.0).abs() / want < 1e-3,
        "{}",
        arm.mean_mm
    );
    assert!(c
        .rough
        .iter()
        .filter(|r| r.chain != "arm.R")
        .all(|r| r.peak_mm < 1e-9));
    // A fast, even swing (0.25 rad a frame, 7.5 rad/s): jerk all along
    // (the hand's path curves), but no back-and-forth anywhere.
    let mut swing = still(rig, n);
    for (f, fr) in swing.iter_mut().enumerate() {
        fr[fore] = Some(turned(
            rig,
            fore,
            Quat::from_axis_angle(axis, 0.25 * f as f64),
        ));
    }
    let c = score(&s, "idle", &times(n), &swing, false);
    let arm = c.rough.iter().find(|r| r.chain == "arm.R").unwrap();
    assert!(arm.peak_mm < 1e-6, "{arm:?}");
    let jerk = c.jerk.iter().find(|j| j.chain == "arm.R").unwrap();
    assert!(jerk.mean > 50.0, "{jerk:?}");
}

#[test]
fn joint_limits_catch_backward_knees_and_elbows_only() {
    let s = subject();
    let rig = &s.rig;
    let fwd = motion_core::animate::forward(rig, &s.map);
    let n = 4;
    assert_eq!(
        score(&s, "idle", &times(n), &still(rig, n), false)
            .limits
            .frames,
        0
    );
    // Bend a lower bone about its hinge by `deg` toward `toward * fwd`
    // (positive deg) or away (negative).
    let bend = |upper: &str, lower: &str, end: &str, toward: f64, deg: f64| {
        let (a, b, c) = (
            rig.find(upper).unwrap(),
            rig.find(lower).unwrap(),
            rig.find(end).unwrap(),
        );
        let u = rig.head(b) - rig.head(a);
        let axis = u.cross(fwd).normalized();
        let mut q = Quat::from_axis_angle(axis, deg.to_radians());
        let v = q.rotate_vec(rig.head(c) - rig.head(b));
        if (v.dot(fwd) * toward < 0.0) == (deg > 0.0) {
            q = Quat::from_axis_angle(axis, -deg.to_radians());
        }
        let mut frames = still(rig, n);
        for fr in frames.iter_mut() {
            fr[b] = Some(turned(rig, b, q));
        }
        score(&s, "idle", &times(n), &frames, false).limits
    };
    // Natural bends: no violation.
    let l = bend("DEF-upper_arm.R", "DEF-forearm.R", "DEF-hand.R", 1.0, 100.0);
    assert_eq!(l.frames, 0, "{l:?}");
    let l = bend("DEF-thigh.L", "DEF-shin.L", "DEF-foot.L", -1.0, 90.0);
    assert_eq!(l.frames, 0, "{l:?}");
    // Backward by 40 degrees: 30 past the 10 allowed.
    let l = bend("DEF-upper_arm.R", "DEF-forearm.R", "DEF-hand.R", 1.0, -40.0);
    assert_eq!(l.frames, n);
    assert_eq!(l.worst, "elbow.R straighter than straight");
    assert!((l.worst_deg - 30.0).abs() < 0.5, "{l:?}");
    let l = bend("DEF-thigh.L", "DEF-shin.L", "DEF-foot.L", -1.0, -40.0);
    assert_eq!(l.worst, "knee.L straighter than straight", "{l:?}");
    assert!((l.worst_deg - 30.0).abs() < 0.5, "{l:?}");
    // A wrist bent 100 degrees further than at rest (the hand turned in
    // the plane of the forearm and the hand's own direction): 10 past.
    // Twisted 100 degrees about the forearm instead (the forearm's own
    // turn): the bend is as at rest, nothing past.
    let hand = rig.find("DEF-hand.R").unwrap();
    let fore = rig.find("DEF-forearm.R").unwrap();
    let kids: Vec<Vec3> = rig.children[hand].iter().map(|&c| rig.head(c)).collect();
    let tip = kids
        .iter()
        .fold(Vec3::ZERO, |a, k| a + *k)
        .scale(1.0 / kids.len() as f64);
    let along = (rig.head(hand) - rig.head(fore)).normalized();
    let across = along.cross(tip - rig.head(hand)).normalized();
    let mut frames = still(rig, n);
    for fr in frames.iter_mut() {
        fr[hand] = Some(turned(
            rig,
            hand,
            Quat::from_axis_angle(across, 100f64.to_radians()),
        ));
    }
    let l = score(&s, "idle", &times(n), &frames, false).limits;
    assert_eq!(l.worst, "wrist.R turned too far");
    assert!((l.worst_deg - 10.0).abs() < 1e-6, "{l:?}");
    for fr in frames.iter_mut() {
        fr[hand] = Some(turned(
            rig,
            hand,
            Quat::from_axis_angle(along, 100f64.to_radians()),
        ));
    }
    let l = score(&s, "idle", &times(n), &frames, false).limits;
    assert_eq!(l.frames, 0, "{l:?}");
}

#[test]
fn an_in_place_cycle_is_scored_on_its_treadmill() {
    // Hips still, both feet pushed back together at 0.6 m/s for 9 frames
    // (a treadmill walk's stance), then swung forward 15 cm up back to
    // where they started (it loops). On the world floor the stance is a
    // 16 cm skate per foot; on its own treadmill it is no skate at all.
    let s = subject();
    let rig = &s.rig;
    let n = 13;
    let speed = 0.6 * s.k;
    let mut frames = still(rig, n);
    for side in ["L", "R"] {
        let (th, sh, ft) = (
            rig.find(&format!("DEF-thigh.{side}")).unwrap(),
            rig.find(&format!("DEF-shin.{side}")).unwrap(),
            rig.find(&format!("DEF-foot.{side}")).unwrap(),
        );
        let fwd = motion_core::animate::forward(rig, &s.map);
        let ankle = rig.head(ft);
        let stance_end = speed * 8.0 / 30.0;
        for (f, fr) in frames.iter_mut().enumerate() {
            let (back, up) = if f <= 8 {
                (speed * f as f64 / 30.0, 0.03 * s.k)
            } else {
                let u = (f - 8) as f64 / 4.0;
                (
                    stance_end * (1.0 - u),
                    if f < 12 { 0.15 * s.k } else { 0.03 * s.k },
                )
            };
            let target = ankle + fwd.scale(-back) + Vec3::new(0.0, up, 0.0);
            solve_arm(rig, fr, th, sh, ft, target);
        }
    }
    let c = score(&s, "walk", &times(n), &frames, false);
    assert!((c.feet.belt_mps - 0.6).abs() < 0.05, "{:?}", c.feet);
    assert_eq!(c.feet.skating_plants, 0, "{:?}", c.feet);
    assert_eq!(c.feet.plants, 2, "{:?}", c.feet);
}

#[test]
fn readability_of_a_held_t_pose() {
    let s = subject();
    let n = 10;
    let c = score(&s, "idle", &times(n), &still(&s.rig, n), false);
    // Arms out, legs apart below the hips: every limb point is clear.
    assert!((c.read.clear - 1.0).abs() < 1e-12, "{:?}", c.read);
    assert!(c.read.contrast_deg.abs() < 1e-9, "{:?}", c.read);
}

#[test]
fn arm_ik_keeps_the_elbow_a_hinge() {
    // A bent elbow, then its hand pulled sideways out of the arm's plane
    // (the move the contact pass makes to clear the body): the hand
    // reaches the target and the elbow stays in its hinge plane. Solving
    // in the plane through shoulder, old elbow and target (the old IK)
    // bent it sideways.
    let s = subject();
    let rig = &s.rig;
    let fwd = motion_core::animate::forward(rig, &s.map);
    let (up, fore, hand) = (
        rig.find("DEF-upper_arm.R").unwrap(),
        rig.find("DEF-forearm.R").unwrap(),
        rig.find("DEF-hand.R").unwrap(),
    );
    let u = rig.head(fore) - rig.head(up);
    let axis = u.cross(fwd).normalized();
    let mut q = Quat::from_axis_angle(axis, 1.2);
    if q.rotate_vec(rig.head(hand) - rig.head(fore)).dot(fwd) < 0.0 {
        q = Quat::from_axis_angle(axis, -1.2);
    }
    let n = 4;
    let mut frames = still(rig, n);
    for (f, fr) in frames.iter_mut().enumerate() {
        fr[fore] = Some(turned(rig, fore, q));
        let w = rig.world(fr)[hand].t;
        let target = w + axis.scale(0.05 * f as f64) + fwd.scale(-0.04 * f as f64);
        solve_arm(rig, fr, up, fore, hand, target);
        assert!(rig.world(fr)[hand].t.approx_eq(target, 1e-9), "frame {f}");
    }
    let l = score(&s, "idle", &times(n), &frames, false).limits;
    assert!(!l.joints.iter().any(|j| j.starts_with("elbow")), "{l:?}");
    // And the bend itself is exactly in the hinge plane.
    for fr in &frames {
        let w = rig.world(fr);
        let (a, b) = (w[fore].t - w[up].t, w[hand].t - w[fore].t);
        let hinge = w[up]
            .rotation()
            .rotate_vec(rig.rest_world[up].rotation().conj().rotate_vec(axis));
        assert!(b.normalized().dot(hinge).abs() < 1e-9 + a.normalized().dot(hinge).abs());
    }
}

#[test]
fn a_fast_stop_is_not_back_and_forth() {
    // The forearm swings faster and faster for 6 frames, then stops dead:
    // a punch's speed peak. Its acceleration turns back once; a zigzag
    // turns back twice running. The pros' strikes scored 100-950 mm when
    // one turn counted.
    let s = subject();
    let rig = &s.rig;
    let fore = rig.find("DEF-forearm.R").unwrap();
    let hand = rig.find("DEF-hand.R").unwrap();
    let fwd = motion_core::animate::forward(rig, &s.map);
    let axis = (rig.head(hand) - rig.head(fore)).cross(fwd).normalized();
    let n = 20;
    let mut frames = still(rig, n);
    for (f, fr) in frames.iter_mut().enumerate() {
        let g = (f.clamp(5, 11) - 5) as f64;
        fr[fore] = Some(turned(rig, fore, Quat::from_axis_angle(axis, 0.02 * g * g)));
    }
    let c = score(&s, "idle", &times(n), &frames, false);
    let arm = c.rough.iter().find(|r| r.chain == "arm.R").unwrap();
    assert!(arm.peak_mm < 1e-6, "{arm:?}");
    // The same stop overshot and pulled back every frame (+-0.02 rad for
    // 4 frames) is a zigzag again.
    for (f, fr) in frames.iter_mut().enumerate().skip(12).take(4) {
        let sign = if f % 2 == 0 { 1.0 } else { -1.0 };
        fr[fore] = Some(turned(
            rig,
            fore,
            Quat::from_axis_angle(axis, 0.72 + sign * 0.02),
        ));
    }
    let c = score(&s, "idle", &times(n), &frames, false);
    let arm = c.rough.iter().find(|r| r.chain == "arm.R").unwrap();
    assert!(arm.peak_mm > 10.0, "{arm:?}");
}

#[test]
fn a_body_lying_on_the_floor_does_not_float() {
    // Tipped 90 degrees forward about the hips and lowered until the mesh
    // is 1 cm off the floor: the feet are far up, the body is not
    // floating. 20 cm higher it is.
    let s = subject();
    let rig = &s.rig;
    let body = s.body.as_ref().unwrap();
    let hips = rig.find("DEF-spine").unwrap();
    let fwd = motion_core::animate::forward(rig, &s.map);
    let lat = Vec3::new(0.0, 1.0, 0.0).cross(fwd).normalized();
    // Face down (the front turned to the floor), knees bent 90 degrees so
    // the feet point up.
    let mut tip = Quat::from_axis_angle(lat, std::f64::consts::FRAC_PI_2);
    if tip.rotate_vec(fwd).y > 0.0 {
        tip = Quat::from_axis_angle(lat, -std::f64::consts::FRAC_PI_2);
    }
    let mut knee = Quat::from_axis_angle(lat, std::f64::consts::FRAC_PI_2);
    if knee.rotate_vec(Vec3::new(0.0, -1.0, 0.0)).dot(fwd) > 0.0 {
        knee = Quat::from_axis_angle(lat, -std::f64::consts::FRAC_PI_2);
    }
    let lying = |dy: f64| {
        let mut fr = vec![None; rig.names.len()];
        for side in ["L", "R"] {
            let sh = rig.find(&format!("DEF-shin.{side}")).unwrap();
            fr[sh] = Some(turned(rig, sh, knee));
        }
        let mut t = match rig.parent[hips] {
            Some(_) => turned(rig, hips, tip),
            None => {
                let mut t = rig.rest[hips];
                t.r = tip.mul(t.r).normalized();
                t
            }
        };
        fr[hips] = Some(t);
        let low = body.lowest(&rig.world(&fr)) - body.floor;
        let up = Vec3::new(0.0, 0.01 + dy - low, 0.0);
        let shift = match rig.parent[hips] {
            Some(p) => rig.rest_world[p].inverse().unwrap().apply_linear(up),
            None => up,
        };
        t.t = t.t + shift;
        fr[hips] = Some(t);
        vec![fr; 6]
    };
    let c = score(&s, "death", &times(6), &lying(0.0), false);
    assert!(
        c.detail.lowest.iter().all(|h| *h > 0.05 * s.k),
        "feet {:?}",
        c.detail.lowest
    );
    assert_eq!(c.feet.float_frames, 0, "{:?}", c.feet);
    let c = score(&s, "death", &times(6), &lying(0.2), false);
    assert_eq!(c.feet.float_frames, 6);
    assert!((c.feet.float_m * s.k - 0.21).abs() < 1e-3, "{:?}", c.feet);
}

#[test]
fn an_elbow_out_of_its_plane_is_not_a_limit_a_knee_is() {
    // Bent 60 degrees about the facing, out of the hinge plane. An elbow
    // gets there by the upper arm's twist, which joint positions do not
    // show (the pros do it all the time); a knee cannot.
    let s = subject();
    let rig = &s.rig;
    let fwd = motion_core::animate::forward(rig, &s.map);
    let n = 4;
    let side = |lower: &str| {
        let b = rig.find(lower).unwrap();
        let mut frames = still(rig, n);
        for fr in frames.iter_mut() {
            fr[b] = Some(turned(
                rig,
                b,
                Quat::from_axis_angle(fwd, 60f64.to_radians()),
            ));
        }
        score(&s, "idle", &times(n), &frames, false).limits
    };
    let l = side("DEF-forearm.R");
    assert_eq!(l.frames, 0, "{l:?}");
    let l = side("DEF-shin.L");
    assert_eq!(l.worst, "knee.L bent sideways", "{l:?}");
    assert!((l.worst_deg - 35.0).abs() < 0.5, "{l:?}");
}

#[test]
fn a_knee_bends_with_its_foot_whatever_the_thigh_twist() {
    // The thigh rolled 40 degrees about its own length (the hip's twist,
    // which the joints do not show), then the knee bent 90 degrees back
    // across the body's left-right axis, the foot carried along: the
    // joints are exactly a natural bend. A hinge carried by the thigh read
    // that as 40 degrees sideways (15 past); the foot's is right.
    let s = subject();
    let rig = &s.rig;
    let fwd = motion_core::animate::forward(rig, &s.map);
    let lat = Vec3::new(0.0, 1.0, 0.0).cross(fwd).normalized();
    let (th, sh) = (
        rig.find("DEF-thigh.L").unwrap(),
        rig.find("DEF-shin.L").unwrap(),
    );
    let along = (rig.head(sh) - rig.head(th)).normalized();
    let roll = Quat::from_axis_angle(along, 40f64.to_radians());
    let mut bend = Quat::from_axis_angle(lat, std::f64::consts::FRAC_PI_2);
    if bend.rotate_vec(Vec3::new(0.0, -1.0, 0.0)).dot(fwd) > 0.0 {
        bend = Quat::from_axis_angle(lat, -std::f64::consts::FRAC_PI_2);
    }
    let n = 4;
    let mut frames = still(rig, n);
    for fr in frames.iter_mut() {
        fr[th] = Some(turned(rig, th, roll));
        let mut t = rig.rest[sh];
        t.r = roll
            .mul(rest_rot(rig, th))
            .conj()
            .mul(bend)
            .mul(rest_rot(rig, sh))
            .normalized();
        fr[sh] = Some(t);
    }
    let l = score(&s, "idle", &times(n), &frames, false).limits;
    assert!(!l.joints.iter().any(|j| j.starts_with("knee")), "{l:?}");
}
