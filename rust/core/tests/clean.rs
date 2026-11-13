// The clean-up pass on synthetic clips with known answers (procedural
// fixture), scored by `motion check`.

use motion_core::clean::{clean_clip, CleanOptions};
use motion_core::contact::{solve_arm, Options};
use motion_core::fixture::{humanoid, Naming};
use motion_core::glb::Document;
use motion_core::math::Vec3;
use motion_core::motioncheck::{score, Subject};
use motion_core::rig::{Rig, Trs};
use motion_core::standardize::{standardize, Outcome};

fn subject() -> Subject {
    let body = humanoid(Naming::Plain, false, false).unwrap();
    let doc = match standardize(&body, "humanoid").unwrap() {
        Outcome::Done(d, _) => Document::parse(&d.to_bytes().unwrap()).unwrap(),
        Outcome::Refused(_, why) => panic!("{why}"),
    };
    Subject::new(&doc, &Options::default()).unwrap()
}

fn times(n: usize) -> Vec<f64> {
    (0..n).map(|f| f as f64 / 30.0).collect()
}

fn still(rig: &Rig, n: usize) -> Vec<Vec<Option<Trs>>> {
    vec![vec![None; rig.names.len()]; n]
}

/// Hips moved `dy` (world) on the frames `dy_of` gives.
fn hips_offset(s: &Subject, n: usize, dy_of: impl Fn(usize) -> f64) -> Vec<Vec<Option<Trs>>> {
    let rig = &s.rig;
    let hips = rig.find("DEF-spine").unwrap();
    let mut frames = still(rig, n);
    for (f, fr) in frames.iter_mut().enumerate() {
        let up = Vec3::new(0.0, dy_of(f), 0.0);
        let shift = match rig.parent[hips] {
            Some(p) => rig.rest_world[p].inverse().unwrap().apply_linear(up),
            None => up,
        };
        let mut t = rig.rest[hips];
        t.t = t.t + shift;
        fr[hips] = Some(t);
    }
    frames
}

fn clean(s: &Subject, frames: &mut [Vec<Option<Trs>>]) -> motion_core::clean::CleanReport {
    let opts = CleanOptions {
        contact: None,
        ..CleanOptions::default()
    };
    let n = frames.len();
    clean_clip(
        &s.rig,
        &s.map,
        s.body.as_ref(),
        s.height,
        frames,
        &times(n),
        "idle",
        &opts,
    )
}

#[test]
fn a_clip_held_off_the_floor_comes_down_whole() {
    let s = subject();
    let n = 12;
    let mut frames = hips_offset(&s, n, |_| 0.15 * s.k);
    assert_eq!(
        score(&s, "idle", &times(n), &frames, false)
            .feet
            .float_frames,
        n
    );
    let rep = clean(&s, &mut frames);
    assert!((rep.ground_shift_m + 0.15 * s.k).abs() < 1e-9, "{rep:?}");
    let c = score(&s, "idle", &times(n), &frames, false);
    assert_eq!(c.feet.float_frames + c.feet.sink_frames, 0, "{:?}", c.feet);
}

#[test]
fn a_clip_sunk_into_the_floor_comes_up_whole() {
    let s = subject();
    let n = 12;
    let mut frames = hips_offset(&s, n, |_| -0.04 * s.k);
    let rep = clean(&s, &mut frames);
    assert!((rep.ground_shift_m - 0.04 * s.k).abs() < 1e-9, "{rep:?}");
    let c = score(&s, "idle", &times(n), &frames, false);
    assert_eq!(c.feet.sink_frames, 0, "{:?}", c.feet);
    assert!(c.feet.sink_m < 1e-6);
}

#[test]
fn a_dip_into_the_floor_is_lifted_frame_by_frame() {
    // The body drops 6 cm into the floor on frames 8-11 of 20 only: no
    // whole-clip shift; the hips come up there, easing in and out, and
    // frames far from the dip are untouched.
    let s = subject();
    let n = 20;
    let dip = |f: usize| {
        if (8..12).contains(&f) {
            -0.06 * s.k
        } else {
            0.0
        }
    };
    let before = hips_offset(&s, n, dip);
    let mut frames = before.clone();
    let rep = clean(&s, &mut frames);
    assert_eq!(rep.ground_shift_m, 0.0, "{rep:?}");
    assert!((rep.hips_lift_max_m - 0.06 * s.k).abs() < 1e-6, "{rep:?}");
    let c = score(&s, "idle", &times(n), &frames, false);
    assert!(c.feet.sink_m < 1e-6, "{:?}", c.feet);
    let hips = s.rig.find("DEF-spine").unwrap();
    for f in [0, 1, 2, 3, 16, 17, 18, 19] {
        assert_eq!(
            format!("{:?}", frames[f][hips]),
            format!("{:?}", before[f][hips]),
            "frame {f}"
        );
    }
}

#[test]
fn a_jump_keeps_its_air() {
    // Hips up 20 cm on 6 of 20 frames: floating on purpose.
    let s = subject();
    let n = 20;
    let mut frames = hips_offset(
        &s,
        n,
        |f| if (7..13).contains(&f) { 0.2 * s.k } else { 0.0 },
    );
    let rep = clean(&s, &mut frames);
    assert_eq!(rep.ground_shift_m, 0.0, "{rep:?}");
    assert_eq!(
        score(&s, "idle", &times(n), &frames, false)
            .feet
            .float_frames,
        6
    );
}

#[test]
fn a_sliding_plant_is_locked() {
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
    for (f, fr) in frames.iter_mut().enumerate() {
        let x = 0.08 * s.k * f as f64 / (n - 1) as f64;
        solve_arm(rig, fr, th, sh, ft, ankle + Vec3::new(0.0, 0.0, x));
    }
    assert_eq!(
        score(&s, "idle", &times(n), &frames, false)
            .feet
            .skating_plants,
        1
    );
    let rep = clean(&s, &mut frames);
    assert_eq!(rep.plants_locked, 1, "{rep:?}");
    let c = score(&s, "idle", &times(n), &frames, false);
    assert_eq!(c.feet.skating_plants, 0, "{:?}", c.feet);
    assert!(c.feet.worst_plant_m < 1e-6, "{:?}", c.feet);
}
