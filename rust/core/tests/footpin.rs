// Foot pinning in `animate` (`footpin.rs`): on a body with longer legs,
// a planted source foot must stay planted on the target; on the same
// body, pinning must change nothing.

use motion_core::animate::{retarget_with, Mapped};
use motion_core::fixture::{humanoid, Naming};
use motion_core::glb::{node_trs, set_node_trs, Document};
use motion_core::math::Vec3;
use motion_core::rig::{Animation, Rig, Trs};
use motion_core::standardize::{standardize, Outcome};

fn standardized(naming: Naming) -> Document {
    match standardize(&humanoid(naming, false, false).unwrap(), "humanoid").unwrap() {
        Outcome::Done(d, _) => d,
        Outcome::Refused(_, why) => panic!("refused: {}", why),
    }
}

/// The fixture walk with real foot plants: each foot is held (two-bone IK)
/// at the clip's lowest ankle height over a stance window, the way mocap
/// plants look. The procedural walk alone slides its feet.
fn planted_walk() -> Document {
    let mut doc = humanoid(Naming::Mixamo, false, true).unwrap();
    let m = Mapped::new(&doc).unwrap();
    let rig: &Rig = &m.rig;
    let anim = Animation::load_all(&doc).unwrap().remove(0);
    let n = 31usize;
    let times: Vec<f64> = (0..n).map(|f| f as f64 / 30.0).collect();
    let mut posed: Vec<Vec<Option<Trs>>> = times.iter().map(|&t| anim.sample(rig, t)).collect();
    for (side, a, b) in [("L", 2usize, 11usize), ("R", 17, 26)] {
        let th = m.map.node_of(&format!("DEF-thigh.{}", side)).unwrap();
        let sh = m.map.node_of(&format!("DEF-shin.{}", side)).unwrap();
        let ft = m.map.node_of(&format!("DEF-foot.{}", side)).unwrap();
        let ankle: Vec<Vec3> = posed.iter().map(|l| rig.world(l)[ft].t).collect();
        let low = ankle.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
        let mid = ankle[(a + b) / 2];
        let pin = Vec3::new(mid.x, low, mid.z);
        for l in posed.iter_mut().take(b).skip(a) {
            motion_core::contact::solve_arm(rig, l, th, sh, ft, pin);
        }
    }
    let nodes: Vec<usize> = (0..rig.names.len())
        .filter(|&i| posed.iter().any(|l| l[i].is_some()))
        .collect();
    let tracks: Vec<Vec<Trs>> = nodes
        .iter()
        .map(|&i| posed.iter().map(|l| l[i].unwrap_or(rig.rest[i])).collect())
        .collect();
    let hips = m.map.node_of("DEF-spine");
    doc.json.remove("animations");
    motion_core::animate::write_animation(&mut doc, "walk", &times, &nodes, &tracks, hips).unwrap();
    Document::parse(&doc.to_bytes().unwrap()).unwrap()
}

/// Lengthen the shins and feet offsets (longer legs, same upper body).
fn long_legs(mut doc: Document, k: f64) -> Document {
    for node in doc.array_mut("nodes").iter_mut() {
        let name = node
            .get("name")
            .and_then(|n| n.as_str())
            .unwrap_or("")
            .to_string();
        if ["DEF-shin.L", "DEF-shin.R", "DEF-foot.L", "DEF-foot.R"].contains(&name.as_str()) {
            let (t, r, s) = node_trs(node);
            set_node_trs(node, t.scale(k), r, s);
        }
    }
    doc
}

/// Per source plant (ankle low and still): how far the target ankle's
/// horizontal path strays from the source's, turned and scaled. Metres,
/// worst over plants.
fn plant_error(src: &Document, out: &Document) -> (f64, usize) {
    let s = Mapped::new(src).unwrap();
    let t = Mapped::new(out).unwrap();
    let sa = &Animation::load_all(src).unwrap()[0];
    let ta = &Animation::load_all(out).unwrap()[0];
    let scale = t.hip_height / s.hip_height;
    let n: usize = 31;
    let mut worst: f64 = 0.0;
    let mut plants = 0;
    for side in ["L", "R"] {
        let sf = s.map.node_of(&format!("DEF-foot.{}", side)).unwrap();
        let tf = t.map.node_of(&format!("DEF-foot.{}", side)).unwrap();
        let sp: Vec<Vec3> = (0..n)
            .map(|f| s.rig.world(&sa.sample(&s.rig, f as f64 / 30.0))[sf].t)
            .collect();
        let tp: Vec<Vec3> = (0..n)
            .map(|f| t.rig.world(&ta.sample(&t.rig, f as f64 / 30.0))[tf].t)
            .collect();
        let low = sp.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
        let leg = s.hip_height;
        let planted: Vec<bool> = (0..n)
            .map(|f| {
                let g = (f + 1).min(n - 1);
                let h = f.saturating_sub(1);
                let v = Vec3::new(sp[g].x - sp[h].x, 0.0, sp[g].z - sp[h].z).length() * 15.0;
                sp[f].y - low < 0.05 * leg && v < 0.25 * leg
            })
            .collect();
        let mut f = 0;
        while f < n {
            if !planted[f] {
                f += 1;
                continue;
            }
            let a = f;
            while f < n && planted[f] {
                f += 1;
            }
            if f - a < 3 {
                continue;
            }
            plants += 1;
            for g in a..f {
                let ds = Vec3::new(sp[g].x - sp[a].x, 0.0, sp[g].z - sp[a].z).scale(scale);
                let dt = Vec3::new(tp[g].x - tp[a].x, 0.0, tp[g].z - tp[a].z);
                worst = worst.max((dt - ds).length());
            }
        }
    }
    (worst, plants)
}

#[test]
fn pinning_keeps_planted_feet_on_longer_legs() {
    let src = planted_walk();
    let target = long_legs(standardized(Naming::Plain), 1.4);
    let run = |pin| {
        let (out, reports) = retarget_with(
            &target,
            &[("walk".into(), src.clone())],
            30.0,
            None,
            None,
            pin,
        )
        .unwrap();
        (Document::parse(&out.to_bytes().unwrap()).unwrap(), reports)
    };
    let (free, _) = run(false);
    let (pinned, reports) = run(true);
    let p = reports[0].pin.as_ref().unwrap();
    assert!(p.plants >= 2, "plants {}", p.plants);
    let (e_free, n_free) = plant_error(&src, &free);
    let (e_pin, n_pin) = plant_error(&src, &pinned);
    eprintln!(
        "plants {}: free {:.4} m, pinned {:.4} m (max shift {:.4} m)",
        n_pin, e_free, e_pin, p.max_shift
    );
    assert!(n_free >= 2 && n_pin == n_free);
    assert!(
        e_pin < 0.3 * e_free,
        "pinned {} m vs free {} m",
        e_pin,
        e_free
    );
    assert!(e_pin < 0.01, "pinned feet stray {} m", e_pin);
}

#[test]
fn pinning_changes_nothing_on_the_same_body() {
    let src = planted_walk();
    let target = standardized(Naming::Plain);
    let (out, reports) = retarget_with(
        &target,
        &[("walk".into(), src.clone())],
        30.0,
        None,
        None,
        true,
    )
    .unwrap();
    let p = reports[0].pin.as_ref().unwrap();
    assert!(
        p.max_shift < 1e-4,
        "ankle moved {} m on an identical body",
        p.max_shift
    );
    let out = Document::parse(&out.to_bytes().unwrap()).unwrap();
    assert!(plant_error(&src, &out).0 < 1e-3);
}
