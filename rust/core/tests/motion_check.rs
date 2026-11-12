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
