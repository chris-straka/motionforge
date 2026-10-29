//! The exam's scoring: a solved clip against the truth it was made
//! from, bone by bone (local rotation error) and joint by joint (world
//! head position error), plus foot slide.

use super::profile::Creature;
use crate::clip::{fk, tail_world, Clip};
use crate::json::Json;
use crate::math::Quat;

#[derive(Clone, Debug)]
pub struct BoneScore {
    pub name: String,
    pub mean_deg: f64,
    pub max_deg: f64,
    pub mean_mm: f64,
    pub max_mm: f64,
}

#[derive(Clone, Debug)]
pub struct Scores {
    pub frames: usize,
    pub bones: Vec<BoneScore>,
    pub mean_deg: f64,
    /// Mean rotation error over bones that have children (a leaf's twist
    /// about its own axis is invisible to any joint finder).
    pub mean_deg_inner: f64,
    pub max_deg: f64,
    pub mean_mm: f64,
    pub max_mm: f64,
    /// Mean per-frame horizontal foot movement (mm) over frames where the
    /// truth foot is still.
    pub foot_slide_mm: f64,
    pub still_frames: usize,
}

fn round(v: f64, places: i32) -> f64 {
    let s = 10f64.powi(places);
    (v * s).round() / s
}

impl Scores {
    pub fn to_json(&self) -> Json {
        Json::obj(vec![
            ("frames", Json::num(self.frames as f64)),
            ("mean_deg", Json::num(round(self.mean_deg, 3))),
            ("mean_deg_inner", Json::num(round(self.mean_deg_inner, 3))),
            ("max_deg", Json::num(round(self.max_deg, 3))),
            ("mean_mm", Json::num(round(self.mean_mm, 2))),
            ("max_mm", Json::num(round(self.max_mm, 2))),
            ("foot_slide_mm", Json::num(round(self.foot_slide_mm, 2))),
            ("still_frames", Json::num(self.still_frames as f64)),
            (
                "bones",
                Json::Arr(
                    self.bones
                        .iter()
                        .map(|b| {
                            Json::obj(vec![
                                ("name", Json::str(&b.name)),
                                ("mean_deg", Json::num(round(b.mean_deg, 3))),
                                ("max_deg", Json::num(round(b.max_deg, 3))),
                                ("mean_mm", Json::num(round(b.mean_mm, 2))),
                                ("max_mm", Json::num(round(b.max_mm, 2))),
                            ])
                        })
                        .collect(),
                ),
            ),
        ])
    }
}

/// Compare two clips on the creature's skeleton (same bones, same frame
/// count; a frame-count mismatch compares the common prefix).
pub fn compare(c: &Creature, truth: &Clip, solved: &Clip) -> Result<Scores, String> {
    let n = c.skeleton.len();
    if truth.skeleton.len() != n || solved.skeleton.len() != n {
        return Err("clips must be on the creature's skeleton".into());
    }
    for i in 0..n {
        if truth.skeleton.bones[i].name != c.skeleton.bones[i].name
            || solved.skeleton.bones[i].name != c.skeleton.bones[i].name
        {
            return Err(format!("bone {} differs between the clips", i));
        }
    }
    let frames = truth.frames.len().min(solved.frames.len());
    if frames == 0 {
        return Err("no frames to compare".into());
    }
    let mut sum_deg = vec![0.0; n];
    let mut max_deg = vec![0.0f64; n];
    let mut sum_mm = vec![0.0; n];
    let mut max_mm = vec![0.0f64; n];
    let mut slide_sum = 0.0;
    let mut still = 0usize;
    let mut prev_truth_feet: Option<Vec<crate::math::Vec3>> = None;
    let mut prev_solved_feet: Option<Vec<crate::math::Vec3>> = None;
    let still_speed = 0.02 * c.height; // per frame
    for f in 0..frames {
        let ft = fk(&c.skeleton, &truth.frames[f]);
        let fs = fk(&c.skeleton, &solved.frames[f]);
        for i in 0..n {
            // World rotation error (the clip quats are local; compare world
            // orientations so a parent's error isn't double-counted).
            let qt = Quat::from_mat3(ft[i].rot);
            let qs = Quat::from_mat3(fs[i].rot);
            let d = qt.angle_to(qs).to_degrees();
            sum_deg[i] += d;
            max_deg[i] = max_deg[i].max(d);
            let e = (ft[i].head - fs[i].head).length() * 1000.0;
            sum_mm[i] += e;
            max_mm[i] = max_mm[i].max(e);
        }
        let tf: Vec<_> = c
            .feet
            .iter()
            .map(|&b| tail_world(&c.skeleton, &ft, b))
            .collect();
        let sf: Vec<_> = c
            .feet
            .iter()
            .map(|&b| tail_world(&c.skeleton, &fs, b))
            .collect();
        if let (Some(pt), Some(ps)) = (&prev_truth_feet, &prev_solved_feet) {
            for k in 0..tf.len() {
                let mut dt = tf[k] - pt[k];
                dt.z = 0.0;
                if dt.length() <= still_speed {
                    let mut ds = sf[k] - ps[k];
                    ds.z = 0.0;
                    slide_sum += ds.length() * 1000.0;
                    still += 1;
                }
            }
        }
        prev_truth_feet = Some(tf);
        prev_solved_feet = Some(sf);
    }
    let bones: Vec<BoneScore> = (0..n)
        .map(|i| BoneScore {
            name: c.skeleton.bones[i].name.clone(),
            mean_deg: sum_deg[i] / frames as f64,
            max_deg: max_deg[i],
            mean_mm: sum_mm[i] / frames as f64,
            max_mm: max_mm[i],
        })
        .collect();
    let mean_deg = bones.iter().map(|b| b.mean_deg).sum::<f64>() / n as f64;
    let mean_mm = bones.iter().map(|b| b.mean_mm).sum::<f64>() / n as f64;
    let inner: Vec<usize> = (0..n)
        .filter(|&i| c.skeleton.bones.iter().any(|b| b.parent == Some(i)))
        .collect();
    let mean_deg_inner = if inner.is_empty() {
        mean_deg
    } else {
        inner.iter().map(|&i| bones[i].mean_deg).sum::<f64>() / inner.len() as f64
    };
    Ok(Scores {
        frames,
        mean_deg_inner,
        max_deg: bones.iter().map(|b| b.max_deg).fold(0.0, f64::max),
        max_mm: bones.iter().map(|b| b.max_mm).fold(0.0, f64::max),
        bones,
        mean_deg,
        mean_mm,
        foot_slide_mm: if still > 0 {
            slide_sum / still as f64
        } else {
            0.0
        },
        still_frames: still,
    })
}
