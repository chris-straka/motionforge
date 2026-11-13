//! Known-answer tests: a rig posed by hand, drawn as MediaPipe would read
//! it (front and side pictures, exact landmarks), fused and fitted back.

use super::fit::{angle_deg, fit_key, set_world_rot, sheet_scale, FitParams, RigInfo};
use super::{fuse, lm, Figure, FuseParams, Landmarks, ViewSheet};
use crate::contact::hinge_axis;
use crate::fixture::{humanoid_posed, Naming};
use crate::math::{shortest_arc, Quat, Vec3};
use crate::rig::Trs;

const UP: Vec3 = Vec3 {
    x: 0.0,
    y: 1.0,
    z: 0.0,
};

fn info() -> RigInfo {
    // Twisted rest rotations (no bone's frame lines up with the world)
    // and an A-pose, like a game rig.
    let doc = humanoid_posed(Naming::Def, true, false, 45.0).unwrap();
    RigInfo::new(&doc).unwrap()
}

fn rot(info: &RigInfo, l: &[Option<Trs>], n: usize) -> Quat {
    info.rig.world(l)[n].rotation()
}

/// Bend `lower` about its limb's anatomical hinge by `deg`.
fn bend(
    info: &RigInfo,
    l: &mut Vec<Option<Trs>>,
    upper: usize,
    lower: usize,
    end: usize,
    deg: f64,
) {
    let rig = &info.rig;
    let axis_rest = hinge_axis(rig, upper, lower, end).unwrap();
    let a_local = rig.rest_world[upper]
        .rotation()
        .conj()
        .rotate_vec(axis_rest);
    let axis = rot(info, l, upper).rotate_vec(a_local);
    let w = rot(info, l, lower);
    set_world_rot(
        rig,
        l,
        lower,
        Quat::from_axis_angle(axis, deg.to_radians()).mul(w),
    );
}

fn aim(info: &RigInfo, l: &mut Vec<Option<Trs>>, bone: usize, child: usize, dir: Vec3) {
    let w = info.rig.world(l);
    let cur = (w[child].t - w[bone].t).normalized();
    let q = shortest_arc(cur, dir.normalized()).mul(w[bone].rotation());
    set_world_rot(&info.rig, l, bone, q);
}

/// A sword-ish pose: hips and chest turned, the chest leaning, the head
/// turned, the right arm up and forward bent 70 degrees, the left arm
/// forward bent 30, the left knee up (foot off the floor), the right leg
/// standing.
fn known_pose(info: &RigInfo) -> Vec<Option<Trs>> {
    let rig = &info.rig;
    let mut l: Vec<Option<Trs>> = vec![None; rig.names.len()];
    let qh = Quat::from_axis_angle(UP, 15f64.to_radians());
    set_world_rot(
        rig,
        &mut l,
        info.hips,
        qh.mul(rig.rest_world[info.hips].rotation()),
    );
    let lean = Quat::from_axis_angle(info.left, 12f64.to_radians())
        .mul(Quat::from_axis_angle(UP, 12f64.to_radians()));
    let n = info.spine.len() as f64;
    for (i, &b) in info.spine.iter().enumerate() {
        let f = (i + 1) as f64 / n;
        let q = Quat::IDENTITY.slerp(lean, f).mul(qh);
        set_world_rot(rig, &mut l, b, q.mul(rig.rest_world[b].rotation()));
    }
    let qc = lean.mul(qh);
    for s in 0..2 {
        if let Some(sh) = info.shoulder[s] {
            set_world_rot(rig, &mut l, sh, qc.mul(rig.rest_world[sh].rotation()));
        }
    }
    let turn = Quat::from_axis_angle(UP, 20f64.to_radians());
    if let Some(neck) = info.neck {
        let q = Quat::IDENTITY.slerp(turn, 0.5).mul(qc);
        set_world_rot(rig, &mut l, neck, q.mul(rig.rest_world[neck].rotation()));
    }
    if let Some(head) = info.head {
        set_world_rot(
            rig,
            &mut l,
            head,
            turn.mul(qc).mul(rig.rest_world[head].rotation()),
        );
    }
    let (fw, lf) = (info.fwd, info.left);
    aim(
        info,
        &mut l,
        info.upper[1],
        info.fore[1],
        fw.scale(0.6) + UP.scale(0.7) - lf.scale(0.3),
    );
    bend(
        info,
        &mut l,
        info.upper[1],
        info.fore[1],
        info.hand[1],
        70.0,
    );
    aim(
        info,
        &mut l,
        info.upper[0],
        info.fore[0],
        fw.scale(0.8) - UP.scale(0.5) + lf.scale(0.2),
    );
    bend(
        info,
        &mut l,
        info.upper[0],
        info.fore[0],
        info.hand[0],
        30.0,
    );
    aim(
        info,
        &mut l,
        info.thigh[0],
        info.shin[0],
        fw.scale(0.9) - UP.scale(0.45),
    );
    bend(
        info,
        &mut l,
        info.thigh[0],
        info.shin[0],
        info.foot[0],
        70.0,
    );
    // The standing leg a little bent (the fit never locks a knee straight:
    // the fixture's rest leg is dead straight), the hips lowered so its
    // foot stays on the floor.
    let before = rig.world(&l)[info.foot[1]].t.y;
    bend(
        info,
        &mut l,
        info.thigh[1],
        info.shin[1],
        info.foot[1],
        25.0,
    );
    let rise = rig.world(&l)[info.foot[1]].t.y - before;
    super::fit::move_root(info, &mut l, Vec3::new(0.0, -rise, 0.0));
    l
}

/// MediaPipe landmarks of a posed rig: marker points fixed in bones.
fn markers(info: &RigInfo, l: &[Option<Trs>]) -> Vec<Vec3> {
    let rig = &info.rig;
    let w = rig.world(l);
    let rest = |n: usize| rig.rest_world[n];
    // A marker at rest position `p` carried by bone `n`.
    let at = |n: usize, p: Vec3| w[n].apply(rest(n).inverse().unwrap().apply(p));
    let head = info.head.unwrap();
    let hp = rig.head(head);
    let (f, lf) = (info.fwd, info.left);
    let mut m = vec![Vec3::ZERO; lm::COUNT];
    // The nose 20 degrees below the ears' level, as the fit assumes.
    m[lm::NOSE] = at(
        head,
        hp + f.scale(0.1) + UP.scale(0.05 - 0.1 * 20f64.to_radians().tan()),
    );
    for i in 1..=6 {
        let side = if i <= 3 { 1.0 } else { -1.0 };
        m[i] = at(
            head,
            hp + f.scale(0.08) + UP.scale(0.06) + lf.scale(0.03 * side),
        );
    }
    m[lm::EAR_L] = at(head, hp + lf.scale(0.07) + UP.scale(0.05));
    m[lm::EAR_R] = at(head, hp - lf.scale(0.07) + UP.scale(0.05));
    m[9] = at(head, hp + f.scale(0.09) + lf.scale(0.02));
    m[10] = at(head, hp + f.scale(0.09) - lf.scale(0.02));
    for s in 0..2 {
        let o = s; // L landmarks are odd-first: 11 L, 12 R ...
        m[11 + o] = w[info.upper[s]].t;
        m[13 + o] = w[info.fore[s]].t;
        m[15 + o] = w[info.hand[s]].t;
        let hand = info.hand[s];
        let mid = rig.children[hand]
            .iter()
            .map(|&c| rig.head(c))
            .fold(Vec3::ZERO, |a, b| a + b)
            .scale(1.0 / rig.children[hand].len().max(1) as f64);
        m[17 + o] = at(hand, mid - f.scale(0.02));
        m[19 + o] = at(hand, mid + f.scale(0.02));
        m[21 + o] = at(hand, mid + f.scale(0.03));
        m[23 + o] = w[info.thigh[s]].t;
        m[25 + o] = w[info.shin[s]].t;
        m[27 + o] = w[info.foot[s]].t;
        let fp = rig.head(info.foot[s]);
        m[29 + o] = at(
            info.foot[s],
            Vec3::new(fp.x, info.ground + 0.01, fp.z) - f.scale(0.05),
        );
        let tp = rig.head(info.toe[s].unwrap());
        m[31 + o] = at(
            info.foot[s],
            Vec3::new(tp.x, info.ground + 0.01, tp.z) + f.scale(0.03),
        );
    }
    m
}

/// Draw markers as a view: `h` image-right, `d` toward the camera; 400 px
/// per metre, the floor at row 900. `depth_noise` adds a fixed pattern
/// (metres) to MediaPipe's depth guess.
fn draw(info: &RigInfo, m: &[Vec3], h: Vec3, d: Vec3, depth_noise: f64) -> Figure {
    let hip = (m[lm::HIP_L] + m[lm::HIP_R]).scale(0.5);
    let img = m
        .iter()
        .map(|p| {
            let c = info.from_rig(*p);
            [
                500.0 + 400.0 * c.dot(h),
                900.0 - 400.0 * (p.y - info.ground),
                1.0,
            ]
        })
        .collect();
    let world = m
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let c = info.from_rig(*p - hip);
            let noise = depth_noise * [1.0, -0.6, 0.3, -1.0, 0.8][i % 5];
            Vec3::new(c.dot(h), -c.y, -c.dot(d) + noise)
        })
        .collect();
    Figure {
        img,
        world,
        blade: None,
    }
}

fn sheet(fig: Figure) -> ViewSheet {
    ViewSheet {
        figures: vec![fig],
        floor_v: Some(900.0),
        file: String::new(),
    }
}

/// Per part: angle between the fitted and the known pose, degrees.
fn compare(info: &RigInfo, a: &[Option<Trs>], b: &[Option<Trs>]) -> Vec<(String, f64)> {
    let (wa, wb) = (info.rig.world(a), info.rig.world(b));
    let mut out = Vec::new();
    let pairs = |n: &str, x: usize, y: usize| {
        (
            n.to_string(),
            angle_deg(wa[y].t - wa[x].t, wb[y].t - wb[x].t),
        )
    };
    for s in 0..2 {
        let sd = ["L", "R"][s];
        out.push(pairs(
            &format!("upper_arm.{sd}"),
            info.upper[s],
            info.fore[s],
        ));
        out.push(pairs(&format!("forearm.{sd}"), info.fore[s], info.hand[s]));
        out.push(pairs(&format!("thigh.{sd}"), info.thigh[s], info.shin[s]));
        out.push(pairs(&format!("shin.{sd}"), info.shin[s], info.foot[s]));
    }
    out.push(pairs("hip line", info.thigh[1], info.thigh[0]));
    out.push(pairs("shoulder line", info.upper[1], info.upper[0]));
    let chord = |w: &[crate::glb::Affine]| {
        (w[info.upper[0]].t + w[info.upper[1]].t).scale(0.5)
            - (w[info.thigh[0]].t + w[info.thigh[1]].t).scale(0.5)
    };
    out.push(("torso".into(), angle_deg(chord(&wa), chord(&wb))));
    let head = info.head.unwrap();
    out.push((
        "head".into(),
        wa[head]
            .rotation()
            .angle_to(wb[head].rotation())
            .to_degrees(),
    ));
    out
}

fn run(front: bool, side: bool, depth_noise: f64) -> Vec<(String, f64)> {
    let info = info();
    let truth = known_pose(&info);
    let m = markers(&info, &truth);
    let x = Vec3::new(1.0, 0.0, 0.0);
    let z = Vec3::new(0.0, 0.0, 1.0);
    let lms = Landmarks {
        name: "known".into(),
        front: front.then(|| sheet(draw(&info, &m, x, z, depth_noise))),
        side: side.then(|| sheet(draw(&info, &m, z.scale(-1.0), x, depth_noise))),
    };
    let fused = fuse(&lms, &FuseParams::default());
    let params = FitParams {
        sword_hand: None,
        two_hands: Some(false),
        ..FitParams::default()
    };
    let scale = sheet_scale(&info, &fused.poses);
    if depth_noise == 0.0 {
        assert!((scale - 1.0).abs() < 0.02, "scale {scale}");
    }
    let fit = fit_key(&info, &fused.poses[0], scale, false, None, &params);
    let e = compare(&info, &fit.locals, &truth);
    // `cargo test --release -p motion_core known_answer -- --nocapture`
    // prints the table in docs/keypose.md.
    eprintln!(
        "known pose, front {front} side {side}, depth noise {depth_noise} m: limbs worst {:.2} deg, torso/hip/shoulder lines worst {:.2}, head {:.2}",
        worst(&e, &LIMBS),
        worst(&e, &["torso", "hip line", "shoulder line"]),
        worst(&e, &["head"])
    );
    e
}

fn worst(errs: &[(String, f64)], parts: &[&str]) -> f64 {
    errs.iter()
        .filter(|(n, _)| parts.iter().any(|p| n.starts_with(p)))
        .map(|(_, e)| *e)
        .fold(0.0, f64::max)
}

const LIMBS: [&str; 4] = ["upper_arm", "forearm", "thigh", "shin"];

#[test]
fn two_views_fit_back_the_known_pose() {
    let e = run(true, true, 0.0);
    assert!(worst(&e, &LIMBS) < 2.0, "{e:?}");
    assert!(
        worst(&e, &["torso", "hip line", "shoulder line"]) < 2.0,
        "{e:?}"
    );
    assert!(worst(&e, &["head"]) < 8.0, "{e:?}");
}

#[test]
fn two_views_shrug_off_a_bad_depth_guess() {
    // MediaPipe's depth off by up to 15 cm: each view's exact axes carry it.
    let e = run(true, true, 0.15);
    assert!(worst(&e, &LIMBS) < 4.0, "{e:?}");
    assert!(
        worst(&e, &["torso", "hip line", "shoulder line"]) < 4.0,
        "{e:?}"
    );
}

#[test]
fn one_view_leans_on_the_depth_guess() {
    let exact = run(true, false, 0.0);
    assert!(worst(&exact, &LIMBS) < 2.0, "{exact:?}");
    let noisy = run(true, false, 0.05);
    // With one view the depth guess is all there is (5 cm off here): worse.
    assert!(worst(&noisy, &LIMBS) > worst(&exact, &LIMBS));
    assert!(worst(&noisy, &LIMBS) < 45.0, "{noisy:?}");
}

#[test]
fn fit_is_deterministic() {
    let a = run(true, true, 0.05);
    let b = run(true, true, 0.05);
    assert_eq!(format!("{a:?}"), format!("{b:?}"));
}

/// Timing on three hand-made keys between two rest poses: the frame plan,
/// the strike as the fastest stretch, the feet staying put, the windup
/// found in the keys.
#[test]
fn timing_plan_strike_and_planted_feet() {
    use super::timing::{build, Key, Role, TimingParams};
    let info = info();
    let rest: Vec<Option<Trs>> = vec![None; info.rig.names.len()];
    let arm = |dir: Vec3, elbow: f64| {
        let mut l = rest.clone();
        aim(&info, &mut l, info.upper[1], info.fore[1], dir);
        bend(
            &info,
            &mut l,
            info.upper[1],
            info.fore[1],
            info.hand[1],
            elbow,
        );
        l
    };
    let (f, lf) = (info.fwd, info.left);
    let key = |locals: Vec<Option<Trs>>, role: Role, drawn: bool| Key {
        locals,
        role,
        planted: [true, true],
        drawn,
        off_rel: None,
        grip_gap: 0.0,
    };
    let keys = vec![
        key(rest.clone(), Role::Rest, false),
        key(arm(UP.scale(0.9) - lf.scale(0.4), 60.0), Role::Load, true),
        key(arm(f.scale(0.9) + lf.scale(0.4), 10.0), Role::Strike, true),
        key(
            arm(f.scale(0.3) - UP.scale(0.9) + lf.scale(0.5), 20.0),
            Role::Follow,
            true,
        ),
        key(rest.clone(), Role::Rest, false),
    ];
    let tp = TimingParams::default();
    let fp = FitParams {
        sword_hand: Some(1),
        two_hands: Some(false),
        ..FitParams::default()
    };
    let (frames, rep) = build(&info, &keys, &tp, &fp, false);
    // Windup 9 frames, load held 4, strike 3; the follow-through's frames
    // follow from the blade's travel, then its hold (7) and the recovery (10).
    assert_eq!(rep.key_frames[..3], [0, 9, 16]);
    assert_eq!(rep.key_frames[3], 16 + rep.follow_frames);
    assert_eq!(frames.len(), rep.key_frames[4] + 1);
    assert_eq!(rep.key_frames[4], rep.key_frames[3] + 7 + 10);
    assert!(rep.windup_found && !rep.counter_added);
    let hand = |l: &[Option<Trs>]| info.rig.world(l)[info.hand[1]].t;
    let speed: Vec<f64> = (1..frames.len())
        .map(|i| (hand(&frames[i]) - hand(&frames[i - 1])).length())
        .collect();
    let peak = (0..speed.len())
        .max_by(|&a, &b| speed[a].partial_cmp(&speed[b]).unwrap())
        .unwrap()
        + 1;
    assert!((14..=17).contains(&peak), "fastest frame {peak}");
    // Fast in: the strike's last frame covers more than its first.
    assert!(speed[15] > 2.0 * speed[13], "{speed:?}");
    let foot = |l: &[Option<Trs>], s: usize| info.rig.world(l)[info.foot[s]].t;
    for s in 0..2 {
        let moved = frames
            .iter()
            .map(|l| (foot(l, s) - foot(&frames[0], s)).length())
            .fold(0.0, f64::max);
        assert!(moved < 1e-3, "foot {s} moved {moved}");
    }
    // The load's hold coils back against the strike: the hand moves
    // against it there.
    let strike = hand(&frames[16]) - hand(&frames[9]);
    let coil = hand(&frames[13]) - hand(&frames[9]);
    assert!(coil.dot(strike) < 0.0);
}
