//! Per-frame 2D joint detections (`motionforge-detections` v1): for
//! every frame and view, one `[u, v, confidence]` per joint in the
//! creature's bone order, in the view's own pixel coordinates. Written
//! by the joint finder (python/creature/detect.py) or by
//! `creature-project` (synthetic, for the exam and the tests).

use super::camera::View;
use super::profile::Creature;
use super::rng::Rng;
use crate::clip::{fk, Clip};
use crate::json::{self, Json};

pub const DETECTIONS_FORMAT: &str = "motionforge-detections";

#[derive(Clone, Debug, PartialEq)]
pub struct Det {
    pub u: f64,
    pub v: f64,
    pub conf: f64,
}

#[derive(Clone, Debug)]
pub struct Detections {
    pub fps: f64,
    pub views: Vec<String>,
    pub joints: Vec<String>,
    /// frames[f][view][joint]
    pub frames: Vec<Vec<Vec<Det>>>,
    /// Optional presentation time of each frame in seconds (the video's
    /// own timestamps). Absent: frame `f` is at `f / fps`.
    pub times: Option<Vec<f64>>,
}

impl Detections {
    pub fn emit(&self) -> Result<String, String> {
        let mut out = String::new();
        out.push_str("{\"format\":\"");
        out.push_str(DETECTIONS_FORMAT);
        out.push_str("\",\"version\":1,\"fps\":");
        json::push_num(&mut out, self.fps)?;
        out.push_str(",\"views\":[");
        for (i, v) in self.views.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            json::push_str(&mut out, v);
        }
        out.push_str("],\"joints\":[");
        for (i, j) in self.joints.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            json::push_str(&mut out, j);
        }
        out.push(']');
        if let Some(times) = &self.times {
            out.push_str(",\"times\":[");
            for (i, t) in times.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                json::push_num(&mut out, *t)?;
            }
            out.push(']');
        }
        out.push_str(",\"frames\":[\n");
        for (f, frame) in self.frames.iter().enumerate() {
            if f > 0 {
                out.push_str(",\n");
            }
            out.push('[');
            for (vi, view) in frame.iter().enumerate() {
                if vi > 0 {
                    out.push(',');
                }
                out.push('[');
                for (ji, d) in view.iter().enumerate() {
                    if ji > 0 {
                        out.push(',');
                    }
                    out.push('[');
                    json::push_num(&mut out, d.u)?;
                    out.push(',');
                    json::push_num(&mut out, d.v)?;
                    out.push(',');
                    json::push_num(&mut out, d.conf)?;
                    out.push(']');
                }
                out.push(']');
            }
            out.push(']');
        }
        out.push_str("\n]}\n");
        Ok(out)
    }

    pub fn parse(text: &str) -> Result<Detections, String> {
        let v = json::parse(text)?;
        let ctx = "detections";
        match v.get("format").and_then(Json::as_str) {
            Some(f) if f == DETECTIONS_FORMAT => {}
            other => {
                return Err(format!(
                    "{}: format must be \"{}\", found {:?}",
                    ctx, DETECTIONS_FORMAT, other
                ))
            }
        }
        let fps = v
            .get("fps")
            .and_then(Json::as_f64)
            .filter(|f| *f > 0.0 && *f <= 1e4)
            .ok_or_else(|| format!("{}: fps must be in (0, 1e4]", ctx))?;
        let strings = |key: &str| -> Result<Vec<String>, String> {
            v.get(key)
                .and_then(Json::as_arr)
                .ok_or_else(|| format!("{}: missing \"{}\"", ctx, key))?
                .iter()
                .map(|s| {
                    s.as_str()
                        .map(str::to_string)
                        .ok_or_else(|| format!("{}: {} must be strings", ctx, key))
                })
                .collect()
        };
        let views = strings("views")?;
        let joints = strings("joints")?;
        if views.is_empty() || joints.is_empty() {
            return Err(format!("{}: views and joints must not be empty", ctx));
        }
        let mut frames = Vec::new();
        for (f, frame) in v
            .get("frames")
            .and_then(Json::as_arr)
            .ok_or_else(|| format!("{}: missing \"frames\"", ctx))?
            .iter()
            .enumerate()
        {
            let fv = frame
                .as_arr()
                .filter(|a| a.len() == views.len())
                .ok_or_else(|| format!("{}: frame {} needs one entry per view", ctx, f))?;
            let mut per_view = Vec::with_capacity(views.len());
            for (vi, view) in fv.iter().enumerate() {
                let js = view
                    .as_arr()
                    .filter(|a| a.len() == joints.len())
                    .ok_or_else(|| {
                        format!("{}: frame {} view {} needs one entry per joint", ctx, f, vi)
                    })?;
                let mut dets = Vec::with_capacity(joints.len());
                for (ji, d) in js.iter().enumerate() {
                    let a = d.as_arr().filter(|a| a.len() == 3).ok_or_else(|| {
                        format!(
                            "{}: frame {} view {} joint {} must be [u, v, conf]",
                            ctx, f, vi, ji
                        )
                    })?;
                    let num = |k: usize| {
                        a[k].as_f64()
                            .filter(|x| x.is_finite() && x.abs() < 1e7)
                            .ok_or_else(|| {
                                format!(
                                    "{}: frame {} view {} joint {} has a bad number",
                                    ctx, f, vi, ji
                                )
                            })
                    };
                    let conf = num(2)?;
                    dets.push(Det {
                        u: num(0)?,
                        v: num(1)?,
                        conf: conf.clamp(0.0, 1.0),
                    });
                }
                per_view.push(dets);
            }
            frames.push(per_view);
        }
        if frames.is_empty() {
            return Err(format!("{}: no frames", ctx));
        }
        let times = match v.get("times") {
            None => None,
            Some(t) => {
                let arr = t
                    .as_arr()
                    .filter(|a| a.len() == frames.len())
                    .ok_or_else(|| format!("{}: times needs one entry per frame", ctx))?;
                let mut out = Vec::with_capacity(arr.len());
                for (i, x) in arr.iter().enumerate() {
                    let x = x
                        .as_f64()
                        .filter(|x| x.is_finite())
                        .ok_or_else(|| format!("{}: times[{}] is not a number", ctx, i))?;
                    if let Some(&prev) = out.last() {
                        if x <= prev {
                            return Err(format!("{}: times must increase (at {})", ctx, i));
                        }
                    }
                    out.push(x);
                }
                Some(out)
            }
        };
        Ok(Detections {
            fps,
            views,
            joints,
            frames,
            times,
        })
    }
}

/// Synthetic detections: the clip's joints projected through the
/// creature's views, with Gaussian pixel noise of `noise_px` and a
/// `drop` fraction of detections lost (confidence 0) at random.
pub fn project_clip(c: &Creature, clip: &Clip, noise_px: f64, drop: f64, seed: u64) -> Detections {
    let mut rng = Rng::new(seed);
    let views: Vec<&View> = c.views.iter().collect();
    let mut frames = Vec::with_capacity(clip.frames.len());
    for frame in &clip.frames {
        let posed = fk(&c.skeleton, frame);
        let mut per_view = Vec::with_capacity(views.len());
        for view in &views {
            let mut dets = Vec::with_capacity(posed.len());
            for b in &posed {
                let (u, v, depth) = view.project(b.head);
                let lost = depth <= 0.0 || (drop > 0.0 && rng.unit() < drop);
                let (nu, nv) = if noise_px > 0.0 {
                    (rng.gauss() * noise_px, rng.gauss() * noise_px)
                } else {
                    (0.0, 0.0)
                };
                dets.push(if lost {
                    Det {
                        u: 0.0,
                        v: 0.0,
                        conf: 0.0,
                    }
                } else {
                    Det {
                        u: u + nu,
                        v: v + nv,
                        conf: 1.0,
                    }
                });
            }
            per_view.push(dets);
        }
        frames.push(per_view);
    }
    Detections {
        fps: clip.fps,
        views: views.iter().map(|v| v.name.clone()).collect(),
        joints: c.joint_names(),
        frames,
        times: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let d = Detections {
            fps: 24.0,
            views: vec!["front".into(), "side".into()],
            joints: vec!["a".into(), "b".into()],
            frames: vec![vec![
                vec![
                    Det {
                        u: 1.5,
                        v: 2.0,
                        conf: 1.0,
                    },
                    Det {
                        u: 3.0,
                        v: 4.0,
                        conf: 0.5,
                    },
                ],
                vec![
                    Det {
                        u: 5.0,
                        v: 6.0,
                        conf: 0.0,
                    },
                    Det {
                        u: 7.0,
                        v: 8.0,
                        conf: 1.0,
                    },
                ],
            ]],
            times: None,
        };
        let text = d.emit().unwrap();
        let back = Detections::parse(&text).unwrap();
        assert_eq!(back.frames, d.frames);
        assert_eq!(back.emit().unwrap(), text);
        assert!(
            Detections::parse(&text.replace("\"frames\":[\n[[", "\"frames\":[\n[[[1]")).is_err()
        );
        let mut timed = d.clone();
        timed.times = Some(vec![0.125]);
        let text = timed.emit().unwrap();
        let back = Detections::parse(&text).unwrap();
        assert_eq!(back.times, Some(vec![0.125]));
        assert_eq!(back.emit().unwrap(), text);
        assert!(Detections::parse(&text.replace("[0.125]", "[0.1,0.2]")).is_err());
    }
}
