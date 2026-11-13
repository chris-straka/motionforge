// Readability at the game camera: the geometry with known answers, and
// the measure on the procedural fixture humanoid.

use motion_core::contact::Options;
use motion_core::fixture::{humanoid, Naming};
use motion_core::glb::Document;
use motion_core::math::{Quat, Vec3};
use motion_core::motioncheck::{strike_timing, Subject};
use motion_core::readcam::{path_arc, score_of, CamOptions, Camera, Mask, Stage, READ_FULL};
use motion_core::rig::Trs;
use motion_core::standardize::{standardize, Outcome};

fn target() -> Document {
    let body = humanoid(Naming::Plain, false, false).unwrap();
    match standardize(&body, "humanoid").unwrap() {
        Outcome::Done(d, _) => Document::parse(&d.to_bytes().unwrap()).unwrap(),
        Outcome::Refused(_, why) => panic!("{why}"),
    }
}

#[test]
fn a_triangle_fills_the_pixel_centres_inside_it() {
    // Centres (i + 0.5, j + 0.5) with i + j <= 9: 55 of them.
    let mut m = Mask::new(20, 20);
    m.fill_tri((0.0, 0.0), (10.0, 0.0), (0.0, 10.0));
    assert_eq!(m.area(), 55);
    let mut r = Mask::new(20, 20);
    r.fill_tri((0.0, 0.0), (0.0, 10.0), (10.0, 0.0));
    assert_eq!(r, m, "winding does not matter");
    // Off screen and degenerate: nothing.
    let mut e = Mask::new(20, 20);
    e.fill_tri((-9.0, -9.0), (-1.0, -9.0), (-1.0, -1.0));
    e.fill_tri((1.0, 1.0), (5.0, 5.0), (9.0, 9.0));
    assert_eq!(e.area(), 0);
}

#[test]
fn a_thick_segment_is_its_rectangle() {
    // (2,5)-(8,5), half width 2: x 2..8, y 3..7: 6 x 4 centres.
    let mut m = Mask::new(20, 20);
    m.fill_segment((2.0, 5.0), (8.0, 5.0), 2.0, 2.0);
    assert_eq!(m.area(), 24);
    assert_eq!(m.rows(), 4);
    // Seen end on: a square of the half width.
    let mut d = Mask::new(20, 20);
    d.fill_segment((10.0, 10.0), (10.0, 10.0), 1.0, 1.0);
    assert_eq!(d.area(), 4);
}

#[test]
fn iou_of_two_offset_squares_is_a_third() {
    let sq = |x: f64| {
        let mut m = Mask::new(20, 20);
        m.fill_segment((x, 6.0), (x + 4.0, 6.0), 2.0, 2.0);
        m
    };
    let (a, b) = (sq(2.0), sq(4.0));
    assert_eq!((a.area(), b.area()), (16, 16));
    assert!((a.iou(&b) - 8.0 / 24.0).abs() < 1e-12);
    assert_eq!(a.iou(&a), 1.0);
    let mut u = a.clone();
    u.union_with(&b);
    assert_eq!(u.area(), 24);
}

#[test]
fn the_camera_projects_by_its_focal_length() {
    // 90 deg vertical field of view, 100 px high: f = 50 px.
    let cam = Camera::look(
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, -2.0),
        90.0,
        200,
        100,
    );
    assert!((cam.f - 50.0).abs() < 1e-9);
    let (x, y, z) = cam.project(Vec3::new(0.0, 0.0, -2.0)).unwrap();
    assert!((x - 100.0).abs() < 1e-9 && (y - 50.0).abs() < 1e-9 && (z - 2.0).abs() < 1e-12);
    // 1 m up and 1 m right at 2 m: 25 px up and right of the centre.
    let (x, y, _) = cam.project(Vec3::new(1.0, 1.0, -2.0)).unwrap();
    assert!(
        (x - 125.0).abs() < 1e-9 && (y - 25.0).abs() < 1e-9,
        "{x} {y}"
    );
    assert!(cam.project(Vec3::new(0.0, 0.0, 1.0)).is_none());
}

#[test]
fn the_game_camera_sits_behind_him_and_looks_at_the_focus() {
    let o = CamOptions::default();
    let cam = Camera::game(Vec3::new(0.0, 0.0, 1.0), &o, 1.0);
    // Pitch -0.32 rad at 3 m: 3 sin 0.32 up, 3 cos 0.32 behind.
    let want = Vec3::new(0.0, 1.535 + 3.0 * 0.32f64.sin(), -3.0 * 0.32f64.cos());
    assert!((cam.eye - want).length() < 1e-9, "{:?}", cam.eye);
    let (x, y, z) = cam.project(Vec3::new(0.0, 1.535, 0.0)).unwrap();
    assert!((x - 240.0).abs() < 1e-9 && (y - 135.0).abs() < 1e-9);
    assert!((z - 3.0).abs() < 1e-9);
    // His feet are below the centre, on screen.
    let (_, yf, _) = cam.project(Vec3::ZERO).unwrap();
    assert!(yf > 135.0 && yf < 270.0, "{yf}");
}

#[test]
fn an_arc_turns_one_way_and_a_wiggle_does_not() {
    // Half a circle of radius 50 in 20 steps: chord 2 r sin(4.5 deg) a
    // step, 19 turns of 9 deg.
    let pts: Vec<(f64, f64)> = (0..=20)
        .map(|i| {
            let a = std::f64::consts::PI * i as f64 / 20.0;
            (100.0 + 50.0 * a.cos(), 100.0 - 50.0 * a.sin())
        })
        .collect();
    let (len, cons, turn) = path_arc(&pts);
    let step = 100.0 * (std::f64::consts::PI / 40.0).sin();
    assert!((len - 20.0 * step).abs() < 1e-9, "{len}");
    assert!((cons - 1.0).abs() < 1e-12);
    assert!((turn - 171.0).abs() < 1e-6, "{turn}");
    // A zigzag with eight equal turns, alternating: no net turn.
    let zig: Vec<(f64, f64)> = (0..10)
        .map(|i| (5.0 * i as f64, if i % 2 == 0 { 0.0 } else { 5.0 }))
        .collect();
    let (_, cons, _) = path_arc(&zig);
    assert!(cons.abs() < 1e-12, "{cons}");
    // A straight cut barely turns: fully consistent.
    let line: Vec<(f64, f64)> = (0..10).map(|i| (3.0 * i as f64, 0.0)).collect();
    assert_eq!(path_arc(&line), (27.0, 1.0, 0.0));
}

#[test]
fn the_score_needs_an_arc_for_a_held_load_to_count() {
    assert_eq!(score_of(0.0, 0.0, 0.0, 30), 0.0);
    let f = READ_FULL;
    assert!((score_of(f[0], f[1], f[2], 4) - 1.0).abs() < 1e-12);
    let half = score_of(f[0] / 2.0, 0.0, f[2] / 2.0, 4);
    assert!((half - (0.5 + 0.5 + 0.5) / 4.0).abs() < 1e-12, "{half}");
}

#[test]
fn strike_timing_finds_the_peak_and_the_load() {
    // Hand back 10 cm over 10 frames, forward 40 cm in 4, then still.
    let mut hand = Vec::new();
    for f in 0..=10 {
        hand.push(Vec3::new(0.0, 1.0, -0.01 * f as f64));
    }
    for f in 1..=4 {
        hand.push(Vec3::new(0.0, 1.0, -0.1 + 0.1 * f as f64));
    }
    for _ in 0..10 {
        hand.push(Vec3::new(0.0, 1.0, 0.3));
    }
    let (peak, load, windup) = strike_timing(&hand, &hand, 1.0 / 30.0);
    assert!((11..=14).contains(&peak), "{peak}");
    assert!((9..=11).contains(&load), "{load}");
    assert!(windup <= 1, "{windup}");
}

#[test]
fn a_raised_arm_reads_and_standing_still_does_not() {
    let doc = target();
    let s = Subject::new(&doc, &Options::default()).unwrap();
    let stage = Stage::new(&doc, &s, &CamOptions::default()).unwrap();
    let rig = &s.rig;
    let n = 24;
    let still: Vec<Vec<Option<Trs>>> = vec![vec![None; rig.names.len()]; n];
    let neutral = still[0].clone();
    let dt = 1.0 / 30.0;
    let (calm, shots) = stage.read(rig, &still, dt, &neutral);
    assert_eq!(shots.len(), n);
    assert!(shots[0].body.area() > 500, "{}", shots[0].body.area());
    assert_eq!(calm.pose_change, 0.0);
    assert_eq!(calm.body_change, 0.0);
    assert_eq!(calm.arc, 0.0);
    // Swing the right arm out and up, sideways, 150 deg over 8 frames
    // from frame 6, and hold it.
    let up = rig.find("DEF-upper_arm.R").unwrap();
    let axis = s.forward();
    let mut raise = still.clone();
    for (f, fr) in raise.iter_mut().enumerate() {
        let k = ((f as f64 - 6.0) / 8.0).clamp(0.0, 1.0);
        let mut t = rig.rest[up];
        let world_rot = Quat::from_axis_angle(axis, -2.618 * k);
        let parent = rig.rest_world[rig.parent[up].unwrap()].rotation();
        t.r = parent.conj().mul(world_rot).mul(parent).mul(t.r);
        fr[up] = Some(t);
    }
    let (read, _) = stage.read(rig, &raise, dt, &neutral);
    assert!(read.body_change > 0.05, "{read:?}");
    assert!(read.arc_len > 0.3, "{read:?}");
    assert!(read.score > calm.score + 0.05, "{read:?} vs {calm:?}");
    // Byte for byte the same twice.
    let (again, _) = stage.read(rig, &raise, dt, &neutral);
    assert_eq!(format!("{read:?}"), format!("{again:?}"));
}
