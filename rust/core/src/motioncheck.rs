//! Motion score: how a clip on a rigged character will look, measured.
//!
//! Per clip, from the posed skeleton (and the skinned mesh, for limbs in
//! the body):
//!
//! - **Foot skate.** Each foot has two contact points, the ankle (heel
//!   side) and the ball of the foot (the toe joint). A point's height is
//!   its height above where it stands in the rest pose, so a flat foot on
//!   the ground is 0 whatever the ankle's own height. Per frame the lower
//!   point is the foot's contact. The foot is *planted* on a frame when
//!   that point is under 5 cm high and moving slower than 0.75 m/s
//!   horizontally (both scaled to the character's height over 1.75 m), so
//!   a foot skimming the ground mid-stride is not a plant. A plant is a
//!   run of 2+ planted frames; its slide is the horizontal path of the
//!   contact point over the run (frame to frame, the point that was lowest
//!   on the first of each pair, so a heel-to-toe roll is not a slide). A
//!   plant *skates* when it slides more than 2.5 cm. Reported: share of
//!   frames inside skating plants, total skate distance, the worst plant.
//! - **Ground.** Frames where a foot point is more than 1 cm below its
//!   standing height (penetration), and frames where both feet are more
//!   than 5 cm up (floating: a jump floats on purpose).
//! - **Limbs through the body.** The contact pass's proxies and its exact
//!   mesh check (`contact::check_frames`): forearm, hand and blade against
//!   torso, head and thighs, beyond what the rest pose already touches.
//! - **Jerk.** Third difference of joint positions relative to the hips,
//!   per chain (spine+head, each arm, each leg), mean and peak; the root
//!   (hips' world position) apart. In m/s³ on a 1.75 m body.
//! - **Attacks** ("weight and snap"): the weapon hand's (or blade tip's)
//!   fastest frame; the *load* pose (the last frame before it where the
//!   hand still moved against the strike, else the slowest hand frame in
//!   the second before it); *windup* (time the hand moved against the strike's
//!   direction leading into the load); *strike* (load to fastest frame);
//!   *hold* (frames after the fastest frame with the hand under 15% of
//!   its peak speed, counted from the first such frame); root travel
//!   (hips' horizontal start-to-end and furthest excursion).
//!
//! Clips are sampled at 30 fps (`--fps`), whatever they were keyed at.
//! Distances are scaled by `k = height / 1.75 m` so one threshold fits a
//! child and a giant. Deterministic: fixed frame order, no maps.

use crate::contact::{check_frames, Body, Options};
use crate::glb::{Affine, Document};
use crate::humanoid::{self, Mapping};
use crate::json::Json;
use crate::math::Vec3;
use crate::rig::{Animation, Rig, Trs};

/// Reference body height the thresholds are written for, metres.
pub const REF_HEIGHT: f64 = 1.75;
/// A foot point under this (m, at reference height) counts as on the ground.
pub const CONTACT_HEIGHT: f64 = 0.05;
/// A plant sliding further than this (m) skates.
pub const SKATE_DISTANCE: f64 = 0.025;
/// A foot moving faster than this (m/s) is not planted, however low.
pub const PLANT_SPEED: f64 = 0.75;
/// A foot point this far (m) below its standing height is in the ground.
pub const SINK_DEPTH: f64 = 0.01;
/// Hold: hand under this share of its peak speed.
pub const HOLD_SHARE: f64 = 0.15;

/// Name fragments of clips scored as attacks.
pub const ATTACK_WORDS: [&str; 9] = [
    "attack", "slash", "cut", "combo", "stab", "swing", "strike", "punch", "kick",
];

pub fn is_attack(name: &str, words: &[String]) -> bool {
    let low = name.to_ascii_lowercase();
    words.iter().any(|w| low.contains(w.as_str()))
}

struct Foot {
    side: String,
    /// (node, rest world height)
    points: Vec<(usize, f64)>,
}

/// A rigged character ready to score clips on.
pub struct Subject {
    pub rig: Rig,
    pub map: Mapping,
    pub body: Option<Body>,
    /// Character height, metres (skinned mesh, else the skeleton).
    pub height: f64,
    /// `height / REF_HEIGHT`.
    pub k: f64,
    pub opts: Options,
    feet: Vec<Foot>,
    hips: usize,
    chains: Vec<(String, Vec<usize>)>,
}

impl Subject {
    pub fn new(doc: &Document, opts: &Options) -> Result<Subject, String> {
        let rig = Rig::from_doc(doc)?;
        let map = humanoid::map_rig(&rig);
        let hips = map.node_of("DEF-spine").ok_or("no hips bone found")?;
        let body = Body::build(doc, &rig, opts)?;
        let height = match &body {
            Some(b) => b.height,
            None => {
                let ys: Vec<f64> = rig.joints().iter().map(|&j| rig.head(j).y).collect();
                let lo = ys.iter().cloned().fold(f64::INFINITY, f64::min);
                let hi = ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                ((hi - lo) * 1.08).max(1e-6)
            }
        };
        let mut feet = Vec::new();
        for side in ["L", "R"] {
            let points: Vec<(usize, f64)> = [format!("DEF-foot.{side}"), format!("DEF-toe.{side}")]
                .iter()
                .filter_map(|n| map.node_of(n))
                .map(|n| (n, rig.head(n).y))
                .collect();
            if !points.is_empty() {
                feet.push(Foot {
                    side: side.to_string(),
                    points,
                });
            }
        }
        let pick = |names: &[String]| -> Vec<usize> {
            names.iter().filter_map(|n| map.node_of(n)).collect()
        };
        let mut chains = Vec::new();
        let spine: Vec<String> = (1..=6).map(|i| format!("DEF-spine.{:03}", i)).collect();
        chains.push(("spine".to_string(), pick(&spine)));
        for side in ["L", "R"] {
            chains.push((
                format!("arm.{side}"),
                pick(&[format!("DEF-forearm.{side}"), format!("DEF-hand.{side}")]),
            ));
        }
        for side in ["L", "R"] {
            chains.push((
                format!("leg.{side}"),
                pick(&[
                    format!("DEF-shin.{side}"),
                    format!("DEF-foot.{side}"),
                    format!("DEF-toe.{side}"),
                ]),
            ));
        }
        chains.retain(|(_, nodes)| !nodes.is_empty());
        Ok(Subject {
            k: height / REF_HEIGHT,
            rig,
            map,
            body,
            height,
            opts: opts.clone(),
            feet,
            hips,
            chains,
        })
    }
}

/// Sample a clip at `fps` (30 when none is given), the rate the renders
/// and the game's own sampling use, whatever rate it was keyed at.
pub fn sample(rig: &Rig, anim: &Animation, fps: Option<f64>) -> (Vec<f64>, Vec<Vec<Option<Trs>>>) {
    let fps = fps.unwrap_or(30.0);
    let n = ((anim.duration * fps).round() as usize).max(1) + 1;
    let times: Vec<f64> = (0..n)
        .map(|f| anim.duration * f as f64 / (n - 1).max(1) as f64)
        .collect();
    let frames = times.iter().map(|&t| anim.sample(rig, t)).collect();
    (times, frames)
}

#[derive(Clone, Debug, Default)]
pub struct FeetScore {
    pub plants: usize,
    pub skating_plants: usize,
    /// Frames inside a skating plant (either foot).
    pub skate_frames: usize,
    /// Total slide of skating plants, metres (at reference height).
    pub skate_m: f64,
    /// Largest slide of one plant, metres, its foot and start frame.
    pub worst_plant_m: f64,
    pub worst_plant: String,
    pub worst_frame: usize,
    pub sink_frames: usize,
    pub sink_m: f64,
    pub float_frames: usize,
    pub float_m: f64,
}

#[derive(Clone, Debug, Default)]
pub struct ClipThrough {
    /// Frames with any arm (or blade) inside the body mesh.
    pub mesh_frames: usize,
    pub mesh_depth_m: f64,
    pub worst_frame: usize,
    pub proxy_frames: usize,
    pub proxy_depth_m: f64,
    pub contacts: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct JerkScore {
    pub chain: String,
    pub mean: f64,
    pub peak: f64,
    pub peak_frame: usize,
}

#[derive(Clone, Debug, Default)]
pub struct AttackScore {
    pub side: String,
    /// Fastest frame of the tip (blade when the rig has one, else hand).
    pub peak_frame: usize,
    pub peak_s: f64,
    pub hand_peak_mps: f64,
    pub tip_peak_mps: f64,
    pub load_frame: usize,
    pub windup_s: f64,
    pub strike_s: f64,
    pub hold_frames: usize,
    pub root_travel_m: f64,
    pub root_reach_m: f64,
}

#[derive(Clone, Debug)]
pub struct ClipScore {
    pub name: String,
    pub frames: usize,
    pub fps: f64,
    pub duration: f64,
    pub feet: FeetScore,
    pub through: Option<ClipThrough>,
    pub jerk: Vec<JerkScore>,
    pub root_jerk: JerkScore,
    pub attack: Option<AttackScore>,
}

fn flat(v: Vec3) -> Vec3 {
    Vec3::new(v.x, 0.0, v.z)
}

fn r(v: f64, places: i32) -> f64 {
    let k = 10f64.powi(places);
    (v * k).round() / k
}

/// Score one clip given as local poses at `times`.
pub fn score(
    subject: &Subject,
    name: &str,
    times: &[f64],
    frames: &[Vec<Option<Trs>>],
    attack: bool,
) -> ClipScore {
    let n = frames.len();
    let rig = &subject.rig;
    let k = subject.k;
    let world: Vec<Vec<Affine>> = frames.iter().map(|f| rig.world(f)).collect();
    let dt = if n > 1 {
        (times[n - 1] - times[0]) / (n - 1) as f64
    } else {
        1.0 / 30.0
    };
    let dt = if dt > 1e-9 { dt } else { 1.0 / 30.0 };
    let pos = |f: usize, node: usize| world[f][node].t;

    // Feet.
    let mut feet = FeetScore::default();
    let mut skating = vec![false; n];
    let mut lowest_any = vec![f64::INFINITY; n];
    for foot in &subject.feet {
        let mut low_node = vec![foot.points[0].0; n];
        let mut low_h = vec![f64::INFINITY; n];
        for f in 0..n {
            for &(node, rest_y) in &foot.points {
                let h = pos(f, node).y - rest_y;
                if h < low_h[f] {
                    low_h[f] = h;
                    low_node[f] = node;
                }
                if h < -SINK_DEPTH * k {
                    feet.sink_m = feet.sink_m.max(-h / k);
                }
            }
            lowest_any[f] = lowest_any[f].min(low_h[f]);
        }
        let speed = |f: usize| -> f64 {
            let node = low_node[f];
            let (a, b) = (f.saturating_sub(1), (f + 1).min(n - 1));
            if b == a {
                return 0.0;
            }
            flat(pos(b, node) - pos(a, node)).length() / ((b - a) as f64 * dt)
        };
        let planted: Vec<bool> = (0..n)
            .map(|f| low_h[f] < CONTACT_HEIGHT * k && speed(f) < PLANT_SPEED * k)
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
            let b = f; // [a, b)
            if b - a < 2 {
                continue;
            }
            feet.plants += 1;
            let slide: f64 = (a..b - 1)
                .map(|g| flat(pos(g + 1, low_node[g]) - pos(g, low_node[g])).length())
                .sum::<f64>()
                / k;
            if slide > feet.worst_plant_m {
                feet.worst_plant_m = slide;
                feet.worst_plant = foot.side.clone();
                feet.worst_frame = a;
            }
            if slide > SKATE_DISTANCE {
                feet.skating_plants += 1;
                feet.skate_m += slide;
                for s in skating.iter_mut().take(b).skip(a) {
                    *s = true;
                }
            }
        }
    }
    feet.skate_frames = skating.iter().filter(|s| **s).count();
    for f in 0..n {
        if !subject.feet.is_empty() {
            let pts: Vec<f64> = subject
                .feet
                .iter()
                .flat_map(|ft| ft.points.iter().map(move |&(node, y)| (node, y)))
                .map(|(node, y)| pos(f, node).y - y)
                .collect();
            if pts.iter().any(|h| *h < -SINK_DEPTH * k) {
                feet.sink_frames += 1;
            }
            if lowest_any[f] > CONTACT_HEIGHT * k {
                feet.float_frames += 1;
                feet.float_m = feet.float_m.max(lowest_any[f] / k);
            }
        }
    }

    // Limbs through the body.
    let through = subject.body.as_ref().map(|body| {
        let weapon = subject.opts.weapon_in(name);
        let arms = check_frames(body, rig, frames, weapon);
        let mut t = ClipThrough::default();
        let mut any = vec![false; n];
        for a in &arms {
            for (f, inside) in a.mesh_inside.iter().enumerate() {
                any[f] |= *inside;
            }
            if a.mesh_depth / k > t.mesh_depth_m {
                t.mesh_depth_m = a.mesh_depth / k;
                t.worst_frame = a.worst_frame;
            }
            t.proxy_frames = t.proxy_frames.max(a.proxy_frames);
            t.proxy_depth_m = t.proxy_depth_m.max(a.proxy_depth / k);
            for c in &a.contacts {
                if !t.contacts.contains(c) {
                    t.contacts.push(c.clone());
                }
            }
        }
        t.contacts.sort();
        t.mesh_frames = any.iter().filter(|x| **x).count();
        t
    });

    // Jerk.
    let hips = subject.hips;
    let jerk_of = |chain: &str, track: &dyn Fn(usize) -> Vec<Vec3>| -> JerkScore {
        let mut sum = 0.0;
        let mut count = 0usize;
        let mut peak = 0.0;
        let mut peak_frame = 0;
        if n >= 4 {
            let p: Vec<Vec<Vec3>> = (0..n).map(track).collect();
            for f in 1..n - 2 {
                for j in 0..p[f].len() {
                    let v = p[f + 2][j] - p[f + 1][j].scale(3.0) + p[f][j].scale(3.0) - p[f - 1][j];
                    let mag = v.length() / (dt * dt * dt) / k;
                    sum += mag;
                    count += 1;
                    if mag > peak {
                        peak = mag;
                        peak_frame = f;
                    }
                }
            }
        }
        JerkScore {
            chain: chain.to_string(),
            mean: if count > 0 { sum / count as f64 } else { 0.0 },
            peak,
            peak_frame,
        }
    };
    let jerk: Vec<JerkScore> = subject
        .chains
        .iter()
        .map(|(cname, nodes)| {
            jerk_of(cname, &|f| {
                nodes.iter().map(|&j| pos(f, j) - pos(f, hips)).collect()
            })
        })
        .collect();
    let root_jerk = jerk_of("root", &|f| vec![pos(f, hips)]);

    // Attack.
    let attack = if attack && n >= 3 {
        let side = subject
            .opts
            .weapon_side
            .clone()
            .unwrap_or_else(|| "R".to_string());
        subject
            .map
            .node_of(&format!("DEF-hand.{side}"))
            .map(|hand| {
                let blade = subject.body.as_ref().and_then(|b| {
                    b.arms
                        .iter()
                        .find(|a| a.side == side)
                        .and_then(|a| a.weapon.clone())
                });
                let tip = |f: usize| match &blade {
                    Some(c) => world[f][c.bone].apply(c.b),
                    None => pos(f, hand),
                };
                let vel = |p: &dyn Fn(usize) -> Vec3, f: usize| -> Vec3 {
                    let (a, b) = (f.saturating_sub(1), (f + 1).min(n - 1));
                    (p(b) - p(a)).scale(1.0 / ((b - a) as f64 * dt))
                };
                let hand_p = |f: usize| pos(f, hand);
                let hand_speed: Vec<f64> = (0..n).map(|f| vel(&hand_p, f).length()).collect();
                let tip_speed: Vec<f64> = (0..n).map(|f| vel(&tip, f).length()).collect();
                let mut peak_frame = 0;
                for f in 0..n {
                    if tip_speed[f] > tip_speed[peak_frame] {
                        peak_frame = f;
                    }
                }
                let dir = vel(&tip, peak_frame).normalized();
                let win = ((1.0 / dt).round() as usize).max(1);
                let lo = peak_frame.saturating_sub(win);
                // Load: the last frame before the peak where the hand still
                // moved against (or across) the strike; else the slowest
                // hand frame in the second before it.
                let mut load = peak_frame;
                for f in (lo..peak_frame).rev() {
                    if vel(&hand_p, f).dot(dir) <= 0.0 {
                        load = f;
                        break;
                    }
                }
                if load == peak_frame {
                    for f in lo..peak_frame {
                        if load == peak_frame || hand_speed[f] < hand_speed[load] {
                            load = f;
                        }
                    }
                }
                // Windup: frames leading into the load where the hand moved
                // against the strike direction.
                let mut w = load;
                while w > 0 && vel(&hand_p, w - 1).dot(dir) < 0.0 {
                    w -= 1;
                }
                let peak_hand = hand_speed.iter().cloned().fold(0.0, f64::max);
                let mut hold = 0;
                let mut f = peak_frame + 1;
                while f < n && hand_speed[f] >= HOLD_SHARE * peak_hand {
                    f += 1;
                }
                while f < n && hand_speed[f] < HOLD_SHARE * peak_hand {
                    hold += 1;
                    f += 1;
                }
                let h0 = flat(pos(0, hips));
                let reach = (0..n)
                    .map(|f| (flat(pos(f, hips)) - h0).length())
                    .fold(0.0, f64::max);
                AttackScore {
                    side: side.clone(),
                    peak_frame,
                    peak_s: times[peak_frame] - times[0],
                    hand_peak_mps: peak_hand / k,
                    tip_peak_mps: tip_speed[peak_frame] / k,
                    load_frame: load,
                    windup_s: (load - w) as f64 * dt,
                    strike_s: (peak_frame - load) as f64 * dt,
                    hold_frames: hold,
                    root_travel_m: (flat(pos(n - 1, hips)) - h0).length() / k,
                    root_reach_m: reach / k,
                }
            })
    } else {
        None
    };

    ClipScore {
        name: name.to_string(),
        frames: n,
        fps: 1.0 / dt,
        duration: if n > 0 { times[n - 1] - times[0] } else { 0.0 },
        feet,
        through,
        jerk,
        root_jerk,
        attack,
    }
}

/// Score every clip of a rigged GLB (or those named in `only`).
pub fn check_doc(
    doc: &Document,
    opts: &Options,
    only: &[String],
    attack_words: &[String],
    fps: Option<f64>,
) -> Result<(Subject, Vec<ClipScore>), String> {
    let subject = Subject::new(doc, opts)?;
    let mut out = Vec::new();
    for anim in Animation::load_all(doc)? {
        if !only.is_empty() && !only.iter().any(|o| *o == anim.name) {
            continue;
        }
        let (times, frames) = sample(&subject.rig, &anim, fps);
        let attack = is_attack(&anim.name, attack_words);
        out.push(score(&subject, &anim.name, &times, &frames, attack));
    }
    Ok((subject, out))
}

impl JerkScore {
    fn to_json(&self) -> Json {
        Json::obj(vec![
            ("chain", Json::str(&self.chain)),
            ("mean", Json::num(r(self.mean, 1))),
            ("peak", Json::num(r(self.peak, 1))),
            ("peak_frame", Json::num(self.peak_frame as f64)),
        ])
    }
}

impl ClipScore {
    /// Largest chain mean jerk (the clip's jerk number) and its chain.
    pub fn jerk_worst(&self) -> (&str, f64, f64) {
        let mut best: Option<&JerkScore> = None;
        for j in &self.jerk {
            if best.is_none_or(|b| j.mean > b.mean) {
                best = Some(j);
            }
        }
        best.map(|j| (j.chain.as_str(), j.mean, j.peak))
            .unwrap_or(("-", 0.0, 0.0))
    }

    pub fn to_json(&self) -> Json {
        let cm = |m: f64| Json::num(r(m * 100.0, 1));
        let n = self.frames.max(1) as f64;
        let f = &self.feet;
        let mut fields = vec![
            ("name", Json::str(&self.name)),
            ("frames", Json::num(self.frames as f64)),
            ("fps", Json::num(r(self.fps, 3))),
            ("duration_s", Json::num(r(self.duration, 4))),
            (
                "feet",
                Json::obj(vec![
                    ("plants", Json::num(f.plants as f64)),
                    ("skating_plants", Json::num(f.skating_plants as f64)),
                    ("skate_frames", Json::num(f.skate_frames as f64)),
                    ("skate_share", Json::num(r(f.skate_frames as f64 / n, 4))),
                    ("skate_cm", cm(f.skate_m)),
                    ("worst_plant_cm", cm(f.worst_plant_m)),
                    ("worst_plant_foot", Json::str(&f.worst_plant)),
                    ("worst_plant_frame", Json::num(f.worst_frame as f64)),
                    ("sink_frames", Json::num(f.sink_frames as f64)),
                    ("sink_cm", cm(f.sink_m)),
                    ("float_frames", Json::num(f.float_frames as f64)),
                    ("float_cm", cm(f.float_m)),
                ]),
            ),
        ];
        if let Some(t) = &self.through {
            fields.push((
                "through_body",
                Json::obj(vec![
                    ("mesh_frames", Json::num(t.mesh_frames as f64)),
                    ("mesh_share", Json::num(r(t.mesh_frames as f64 / n, 4))),
                    ("mesh_depth_cm", cm(t.mesh_depth_m)),
                    ("worst_frame", Json::num(t.worst_frame as f64)),
                    ("proxy_frames", Json::num(t.proxy_frames as f64)),
                    ("proxy_depth_cm", cm(t.proxy_depth_m)),
                    (
                        "contacts",
                        Json::Arr(t.contacts.iter().map(|c| Json::str(c)).collect()),
                    ),
                ]),
            ));
        }
        fields.push((
            "jerk",
            Json::Arr(self.jerk.iter().map(JerkScore::to_json).collect()),
        ));
        fields.push(("root_jerk", self.root_jerk.to_json()));
        if let Some(a) = &self.attack {
            fields.push((
                "attack",
                Json::obj(vec![
                    ("side", Json::str(&a.side)),
                    ("peak_frame", Json::num(a.peak_frame as f64)),
                    ("peak_s", Json::num(r(a.peak_s, 3))),
                    ("hand_peak_mps", Json::num(r(a.hand_peak_mps, 2))),
                    ("tip_peak_mps", Json::num(r(a.tip_peak_mps, 2))),
                    ("load_frame", Json::num(a.load_frame as f64)),
                    ("windup_s", Json::num(r(a.windup_s, 3))),
                    ("strike_s", Json::num(r(a.strike_s, 3))),
                    ("hold_frames", Json::num(a.hold_frames as f64)),
                    ("root_travel_cm", cm(a.root_travel_m)),
                    ("root_reach_cm", cm(a.root_reach_m)),
                ]),
            ));
        }
        Json::obj(fields)
    }

    /// One text line (two for attacks).
    pub fn line(&self) -> String {
        let n = self.frames.max(1) as f64;
        let f = &self.feet;
        let (jc, jm, jp) = self.jerk_worst();
        let through = match &self.through {
            Some(t) => format!(
                "through {:>3} fr {:>4.1} cm",
                t.mesh_frames,
                t.mesh_depth_m * 100.0
            ),
            None => "through   - (no mesh)".into(),
        };
        let mut s = format!(
            "{:<18} {:>4} fr  skate {:>5.1}% {:>5.1} cm (worst {:>4.1})  sink {:>4.1} cm  float {:>3} fr  {}  jerk {:>6.0}/{:>6.0} {:<6} root {:>5.0}\n",
            self.name,
            self.frames,
            100.0 * f.skate_frames as f64 / n,
            f.skate_m * 100.0,
            f.worst_plant_m * 100.0,
            f.sink_m * 100.0,
            f.float_frames,
            through,
            jm,
            jp,
            jc,
            self.root_jerk.mean
        );
        if let Some(a) = &self.attack {
            s.push_str(&format!(
                "{:<18} attack: peak {:.2} s tip {:.1} m/s hand {:.1} m/s, windup {:.2} s, strike {:.2} s, hold {} fr, root {:.0} cm (reach {:.0})\n",
                "",
                a.peak_s,
                a.tip_peak_mps,
                a.hand_peak_mps,
                a.windup_s,
                a.strike_s,
                a.hold_frames,
                a.root_travel_m * 100.0,
                a.root_reach_m * 100.0
            ));
        }
        s
    }
}
