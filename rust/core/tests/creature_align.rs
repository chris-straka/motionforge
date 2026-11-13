//! Lining up the two views of a generated video (creature::align):
//! known-answer tests on the fixture biped. One view delayed by a known
//! number of frames, one view playing slower, each view zoomed and
//! shifted by known amounts, timestamps off the grid, and two different
//! moves that must stay rejected however they are slid.

use motion_core::clip::{parse_skeleton, Clip, Frame, Pose};
use motion_core::creature::align::{fit_views, uniform_timing, TimeMode};
use motion_core::creature::compare::compare;
use motion_core::creature::detect::{project_clip, Detections};
use motion_core::creature::motion::exam_motion;
use motion_core::creature::profile::{build, skeleton_bounds, Creature};
use motion_core::creature::solve::{consistency, solve, SolveParams};
use motion_core::math::{Quat, Vec3};

fn hero() -> Creature {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/hero_skel.json"
    ))
    .unwrap();
    let sk = parse_skeleton(&text).unwrap();
    let feet = vec![
        sk.index("DEF-foot.L").unwrap(),
        sk.index("DEF-foot.R").unwrap(),
    ];
    let (lo, hi) = skeleton_bounds(&sk);
    build("hero", sk, feet, Vec3::new(0.0, -1.0, 0.0), lo, hi, 512)
}

/// The exam motion eased in from the rest pose over the first 8 frames,
/// as a video starts from its start picture.
fn from_rest(clip: Clip) -> Clip {
    let frames = clip
        .frames
        .iter()
        .enumerate()
        .map(|(f, fr)| {
            let x = (f as f64 / 8.0).min(1.0);
            let w = x * x * (3.0 - 2.0 * x);
            Frame {
                poses: fr
                    .poses
                    .iter()
                    .map(|p| Pose {
                        loc: Vec3::ZERO.lerp(p.loc, w),
                        quat: Quat::IDENTITY.slerp(p.quat, w).normalized(),
                    })
                    .collect(),
            }
        })
        .collect();
    Clip { frames, ..clip }
}

/// View 1 shows at frame f what view 0 showed at frame f - k.
fn delay_view(d: &Detections, k: usize) -> Detections {
    let mut out = d.clone();
    for f in 0..d.frames.len() {
        out.frames[f][1] = d.frames[f.saturating_sub(k)][1].clone();
    }
    out
}

fn reframe(d: &Detections, view: usize, scale: f64, du: f64, dv: f64) -> Detections {
    let mut out = d.clone();
    for frame in out.frames.iter_mut() {
        for det in frame[view].iter_mut() {
            det.u = 256.0 + scale * (det.u - 256.0) + du;
            det.v = 256.0 + scale * (det.v - 256.0) + dv;
        }
    }
    out
}

fn sub_clip(c: &Clip, first: usize, len: usize) -> Clip {
    Clip {
        fps: c.fps,
        skeleton: c.skeleton.clone(),
        frames: c.frames[first..first + len].to_vec(),
    }
}

#[test]
fn a_view_five_frames_late_is_slid_back() {
    let c = hero();
    let truth = from_rest(exam_motion(&c, 72, 24.0, 5, 0.9));
    let clean = project_clip(&c, &truth, 1.0, 0.0, 3);
    let late = delay_view(&clean, 5);
    let plain = SolveParams::default();
    let (_, before) = consistency(&c, &late, &plain).unwrap();
    assert!(before > plain.max_motion_ratio, "before {}", before);
    assert!(solve(&c, &late, &plain).is_err());
    for mode in [TimeMode::Lag, TimeMode::Auto] {
        let params = SolveParams {
            time_mode: mode,
            ..Default::default()
        };
        let (solved, rep) = solve(&c, &late, &params).expect("aligned clip is accepted");
        let t = rep.align.as_ref().unwrap().time.as_ref().unwrap();
        assert!(
            (t.lag_frames - 5.0).abs() <= 1.0,
            "{:?}: lag {} (want 5)",
            mode,
            t.lag_frames
        );
        assert!(rep.motion_ratio < 0.5 * before, "{:?} {:?}", mode, rep);
        assert!(rep.consistent);
        // The solved clip starts at the first kept frame of the truth.
        let s = compare(
            &c,
            &sub_clip(&truth, t.first_frame, solved.frames.len()),
            &solved,
        )
        .unwrap();
        assert!(s.mean_mm < 40.0, "{:?}: mean {} mm", mode, s.mean_mm);
        eprintln!(
            "late view {:?}: ratio {:.3} -> {:.3}, lag {:.2}, used {}, {} frames from {}, mean {:.1} mm",
            mode,
            before,
            rep.motion_ratio,
            t.lag_frames,
            t.used,
            t.frames_kept,
            t.first_frame,
            s.mean_mm
        );
    }
}

#[test]
fn a_view_playing_slower_is_warped() {
    let c = hero();
    // View 1 runs 8% slow: at frame f it shows the truth of f / 1.08
    // (the lag grows from 0 to about 7 frames over the clip).
    let n = 96;
    let a = exam_motion(&c, n, 24.0, 8, 0.9);
    let b = exam_motion(&c, n, 24.0 * 1.08, 8, 0.9);
    let da = project_clip(&c, &a, 1.0, 0.0, 4);
    let db = project_clip(&c, &b, 1.0, 0.0, 5);
    let mut mixed = da.clone();
    for f in 0..n {
        mixed.frames[f][1] = db.frames[f][1].clone();
    }
    let lag = SolveParams {
        time_mode: TimeMode::Lag,
        force: true,
        ..Default::default()
    };
    let auto = SolveParams {
        time_mode: TimeMode::Auto,
        force: true,
        ..Default::default()
    };
    let (_, rl) = solve(&c, &mixed, &lag).unwrap();
    let (solved, ra) = solve(&c, &mixed, &auto).unwrap();
    let t = ra.align.as_ref().unwrap().time.as_ref().unwrap();
    eprintln!(
        "slow view: ratio none {:.3}, lag {:.3} ({:.2} frames), warp {:?} (lag {:.1}..{:.1}), used {}",
        t.ratio_none, t.ratio_lag, t.lag_frames, t.ratio_warp, t.warp_lag_min, t.warp_lag_max, t.used
    );
    assert_eq!(t.used, "warp", "{:?}", t);
    assert!(ra.motion_ratio < rl.motion_ratio);
    assert!(ra.motion_ratio < 0.5, "{}", ra.motion_ratio);
    // The warp's lag climbs with the clip: near 0 early, about 7 late.
    assert!(t.warp_lag_min < 2.5 && t.warp_lag_max > 4.5, "{:?}", t);
    let s = compare(
        &c,
        &sub_clip(&a, t.first_frame, solved.frames.len()),
        &solved,
    )
    .unwrap();
    assert!(s.mean_mm < 45.0, "mean {} mm", s.mean_mm);
}

#[test]
fn reframed_views_are_fitted_back_on_the_first_frame() {
    let c = hero();
    let truth = from_rest(exam_motion(&c, 48, 24.0, 6, 0.9));
    let clean = project_clip(&c, &truth, 1.0, 0.0, 7);
    let moved = reframe(&reframe(&clean, 0, 1.25, 30.0, -12.0), 1, 0.9, -8.0, 15.0);
    let fits = fit_views(&c, &moved, 0, 0.2).unwrap();
    let want = [(1.25, 30.0, -12.0), (0.9, -8.0, 15.0)];
    for (f, (s, du, dv)) in fits.iter().zip(want) {
        eprintln!(
            "view {}: zoom {:.4} (want {}), shift ({:.2}, {:.2}) (want ({}, {})), rms {:.1} -> {:.2} px",
            f.view, f.scale, s, f.du, f.dv, du, dv, f.rms_before_px, f.rms_after_px
        );
        assert!((f.scale / s - 1.0).abs() < 0.005, "{:?}", f);
        assert!(
            (f.du - du).abs() < 1.0 && (f.dv - dv).abs() < 1.0,
            "{:?}",
            f
        );
        assert!(f.rms_after_px < 2.0, "{:?}", f);
    }
    let plain = SolveParams::default();
    let (gap_raw, _) = consistency(&c, &clean, &plain).unwrap();
    let (gap_moved, _) = consistency(&c, &moved, &plain).unwrap();
    let params = SolveParams {
        align_views: true,
        ..Default::default()
    };
    let (solved, rep) = solve(&c, &moved, &params).unwrap();
    eprintln!(
        "reframed: median gap {:.4} (untouched {:.4}) -> {:.4}",
        gap_moved, gap_raw, rep.median_gap_ratio
    );
    assert!(gap_moved > 4.0 * gap_raw);
    assert!(rep.median_gap_ratio < 1.5 * gap_raw + 1e-4, "{:?}", rep);
    let s = compare(&c, &truth, &solved).unwrap();
    assert!(s.mean_mm < 40.0, "mean {} mm", s.mean_mm);
}

#[test]
fn late_and_reframed_together_like_the_first_real_clip() {
    // The Seedance clip: each half decoded at 640 px for 512 px views
    // (zoom 1.25 about the corner, i.e. a shift too) and the views out of step.
    let c = hero();
    let truth = from_rest(exam_motion(&c, 96, 24.0, 2, 0.9));
    let clean = project_clip(&c, &truth, 1.5, 0.02, 9);
    let mut moved = delay_view(&clean, 6);
    for frame in moved.frames.iter_mut() {
        for view in frame.iter_mut() {
            for det in view.iter_mut() {
                det.u *= 1.25;
                det.v *= 1.25;
            }
        }
    }
    let plain = SolveParams::default();
    let err = solve(&c, &moved, &plain).unwrap_err();
    assert!(err.contains("do not show the same move"), "{}", err);
    let params = SolveParams {
        align_views: true,
        time_mode: TimeMode::Auto,
        ..Default::default()
    };
    let (solved, rep) = solve(&c, &moved, &params).unwrap();
    let a = rep.align.as_ref().unwrap();
    let t = a.time.as_ref().unwrap();
    for f in &a.fits {
        assert!((f.scale - 1.25).abs() < 0.01, "{:?}", f);
        // 1.25x about the corner = 1.25x about the centre + 64 px.
        assert!(
            (f.du - 64.0).abs() < 1.5 && (f.dv - 64.0).abs() < 1.5,
            "{:?}",
            f
        );
    }
    assert!((t.lag_frames - 6.0).abs() <= 1.0, "{:?}", t);
    eprintln!(
        "640 px halves + 6 frames late: gap {:.4} ratio {:.3} -> gap {:.4} ratio {:.3}",
        a.gap_before, a.ratio_before, rep.median_gap_ratio, rep.motion_ratio
    );
    assert!(rep.consistent);
    let s = compare(
        &c,
        &sub_clip(&truth, t.first_frame, solved.frames.len()),
        &solved,
    )
    .unwrap();
    assert!(s.mean_mm < 45.0, "mean {} mm", s.mean_mm);
    // Deterministic.
    let (again, _) = solve(&c, &moved, &params).unwrap();
    assert_eq!(
        motion_core::clip::emit_clip(&solved).unwrap(),
        motion_core::clip::emit_clip(&again).unwrap()
    );
}

#[test]
fn different_moves_stay_rejected_however_they_are_slid() {
    let c = hero();
    let a = exam_motion(&c, 72, 24.0, 1, 0.9);
    let b = exam_motion(&c, 72, 24.0, 2, 0.9);
    let da = project_clip(&c, &a, 0.0, 0.0, 1);
    let db = project_clip(&c, &b, 0.0, 0.0, 1);
    let mut mixed = da.clone();
    for (f, frame) in mixed.frames.iter_mut().enumerate() {
        frame[1] = db.frames[f][1].clone();
    }
    for mode in [TimeMode::Lag, TimeMode::Warp, TimeMode::Auto] {
        let params = SolveParams {
            time_mode: mode,
            ..Default::default()
        };
        let err = solve(&c, &mixed, &params).unwrap_err();
        assert!(
            err.contains("do not show the same move"),
            "{:?}: {}",
            mode,
            err
        );
    }
}

#[test]
fn timestamps_off_the_grid_are_resampled() {
    let c = hero();
    let truth = exam_motion(&c, 50, 24.0, 3, 0.9);
    let full = project_clip(&c, &truth, 0.0, 0.0, 1);
    // Every fifth frame dropped (a stutter), timestamps kept.
    let keep: Vec<usize> = (0..50).filter(|f| f % 5 != 2).collect();
    let mut gappy = full.clone();
    gappy.frames = keep.iter().map(|&f| full.frames[f].clone()).collect();
    gappy.times = Some(keep.iter().map(|&f| f as f64 / 24.0).collect());
    let (back, rep) = uniform_timing(&gappy);
    assert!(rep.resampled && rep.max_jitter_frames > 1.0, "{:?}", rep);
    assert_eq!(back.frames.len(), 50);
    let mut worst: f64 = 0.0;
    for (f, frame) in back.frames.iter().enumerate() {
        for (vi, view) in frame.iter().enumerate() {
            for (j, d) in view.iter().enumerate() {
                let t = &full.frames[f][vi][j];
                worst = worst.max(((d.u - t.u).powi(2) + (d.v - t.v).powi(2)).sqrt());
            }
        }
    }
    // Only the dropped frames are interpolated (over two frames of motion).
    assert!(worst < 6.0, "worst {} px", worst);
    // Uniform timestamps are left alone.
    let mut even = full.clone();
    even.times = Some((0..50).map(|f| 1.5 + f as f64 / 24.0).collect());
    let (same, rep) = uniform_timing(&even);
    assert!(!rep.resampled);
    assert_eq!(same.frames, full.frames);
}
