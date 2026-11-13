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
use crate::detmath;
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
/// Joint limits, degrees: a knee or elbow straighter than straight by
/// more than this...
pub const LIMIT_HYPEREXTEND_DEG: f64 = 10.0;
/// ...or bent further than this...
pub const LIMIT_FLEX_DEG: f64 = 160.0;
/// ...or bent sideways (out of its hinge plane) by more than this.
pub const LIMIT_SIDEWAYS_DEG: f64 = 25.0;
/// Ball joints: turned this far from their rest pose (relative to the
/// parent named in `Subject`): chest on hips, head on chest, hand on
/// forearm, foot on shin.
pub const LIMIT_TORSO_DEG: f64 = 80.0;
pub const LIMIT_HEAD_DEG: f64 = 90.0;
pub const LIMIT_WRIST_DEG: f64 = 90.0;
pub const LIMIT_ANKLE_DEG: f64 = 65.0;
/// An in-place cycle: the hips travel less than this (m) over the clip
/// while the planted foot moves at least `BELT_SPEED` (m/s): its skate is
/// measured on the treadmill the game will move it on.
pub const BELT_TRAVEL: f64 = 0.10;
pub const BELT_SPEED: f64 = 0.3;
/// ...and each foot ends within this (m) of where it started, under the
/// hips (a cycle loops).
pub const BELT_LOOP: f64 = 0.15;

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
    hinges: Vec<Hinge>,
    caps: Vec<Cap>,
    /// Character facing (rest, horizontal) and lateral axis (its left).
    forward: Vec3,
    /// Shoulder width at rest, metres.
    shoulder_w: f64,
    /// Body-speed probes (hips first), as the stylizer's keys use them.
    probes: Vec<usize>,
    /// Torso line for readability: hips, chest, head.
    torso: Vec<usize>,
    /// Limb points that should read apart from the torso.
    limbs: Vec<usize>,
}

/// A hinge joint (knee, elbow): bend about one axis fixed in the upper
/// bone, measured from the three joint heads.
struct Hinge {
    name: String,
    a: usize,
    b: usize,
    c: usize,
    /// Hinge axis in `a`'s frame.
    axis_local: Vec3,
    /// +1 or -1 so that the joint's natural bend is positive.
    sign: f64,
}

/// A ball joint capped at an angle from its rest pose relative to `parent`.
struct Cap {
    name: String,
    node: usize,
    parent: usize,
    max_deg: f64,
}

impl Subject {
    /// His facing at rest, horizontal (both feet heel to toe).
    pub fn forward(&self) -> Vec3 {
        self.forward
    }

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
        let forward = crate::animate::forward(&rig, &map);
        let mut hinges = Vec::new();
        for side in ["L", "R"] {
            for (joint, upper, lower, end, toward) in [
                ("knee", "thigh", "shin", "foot", -1.0),
                ("elbow", "upper_arm", "forearm", "hand", 1.0),
            ] {
                let get = |b: &str| map.node_of(&format!("DEF-{b}.{side}"));
                let (Some(a), Some(b), Some(c)) = (get(upper), get(lower), get(end)) else {
                    continue;
                };
                let u = rig.head(b) - rig.head(a);
                let axis = u.cross(forward);
                if u.length() < 1e-9 || axis.length() < 1e-6 * u.length() {
                    continue;
                }
                let axis = axis.normalized();
                // The natural bend swings the lower bone toward `toward`
                // times the facing (knee back, elbow forward).
                let v_bent = u.normalized() + forward.scale(0.2 * toward);
                let sign = if u.cross(v_bent).dot(axis) >= 0.0 {
                    1.0
                } else {
                    -1.0
                };
                hinges.push(Hinge {
                    name: format!("{joint}.{side}"),
                    a,
                    b,
                    c,
                    axis_local: rig.rest_world[a].rotation().conj().rotate_vec(axis),
                    sign,
                });
            }
        }
        let mut caps = Vec::new();
        let mut cap = |name: &str, node: &str, parent: &str, max_deg: f64| {
            if let (Some(n), Some(p)) = (map.node_of(node), map.node_of(parent)) {
                caps.push(Cap {
                    name: name.to_string(),
                    node: n,
                    parent: p,
                    max_deg,
                });
            }
        };
        cap("torso", "DEF-spine.003", "DEF-spine", LIMIT_TORSO_DEG);
        cap("head", "DEF-spine.006", "DEF-spine.003", LIMIT_HEAD_DEG);
        for side in ["L", "R"] {
            cap(
                &format!("wrist.{side}"),
                &format!("DEF-hand.{side}"),
                &format!("DEF-forearm.{side}"),
                LIMIT_WRIST_DEG,
            );
            cap(
                &format!("ankle.{side}"),
                &format!("DEF-foot.{side}"),
                &format!("DEF-shin.{side}"),
                LIMIT_ANKLE_DEG,
            );
        }
        let shoulder_w = match (
            map.node_of("DEF-upper_arm.L"),
            map.node_of("DEF-upper_arm.R"),
        ) {
            (Some(l), Some(r)) => (rig.head(l) - rig.head(r)).length(),
            _ => 0.22 * height,
        };
        let probes = crate::stylize_body::probe_nodes(&map);
        let torso: Vec<usize> = ["DEF-spine", "DEF-spine.003", "DEF-spine.006"]
            .iter()
            .filter_map(|n| map.node_of(n))
            .collect();
        let mut limbs = Vec::new();
        for side in ["L", "R"] {
            for b in ["forearm", "hand", "shin", "foot"] {
                if let Some(n) = map.node_of(&format!("DEF-{b}.{side}")) {
                    limbs.push(n);
                }
            }
        }
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
            hinges,
            caps,
            forward,
            shoulder_w,
            probes,
            torso,
            limbs,
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
    /// Treadmill speed an in-place cycle was measured on (m/s at
    /// reference height), 0 when the clip travels.
    pub belt_mps: f64,
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

/// Back-and-forth: per joint and frame, how far the next frame's
/// acceleration (second difference) points back against this frame's,
/// `max(0, -a_f . a_f+1) / max(|a_f|, |a_f+1|)`, counted only inside a run
/// of such flips (the frame before or after flips too). A fast arc or a
/// sharp corner scores 0 (its accelerations agree or come alone), so does
/// a snap that flips once (a two-frame strike); a joint that jitters +-e
/// every frame scores 4e. Millimetres at 30 fps on a 1.75 m
/// body, relative to the hips.
#[derive(Clone, Debug)]
pub struct RoughScore {
    pub chain: String,
    pub mean_mm: f64,
    pub peak_mm: f64,
    pub peak_frame: usize,
}

#[derive(Clone, Debug, Default)]
pub struct LimitScore {
    /// Frames with any joint past its limit.
    pub frames: usize,
    /// Worst excess past a limit, degrees, which joint and how, and when.
    pub worst_deg: f64,
    pub worst: String,
    pub worst_frame: usize,
    /// Every joint and way that went past, sorted.
    pub joints: Vec<String>,
}

/// How a clip's key poses read (higher is better; for comparing a clip
/// with its own variants).
#[derive(Clone, Debug, Default)]
pub struct ReadScore {
    /// Key poses (the body's speed minima, as the stylizer finds them).
    pub keys: Vec<usize>,
    /// Mean over key poses of the best view's share of limb points
    /// (elbows, hands, knees, ankles) clear of the torso's silhouette,
    /// over four views round the character (orthographic, level).
    pub clear: f64,
    /// Mean change between successive key poses: the angle each limb
    /// segment and the spine turned in the hips' frame, degrees.
    pub contrast_deg: f64,
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
    pub rough: Vec<RoughScore>,
    pub limits: LimitScore,
    pub read: ReadScore,
    pub attack: Option<AttackScore>,
    /// Readability at the game camera (`readcam`), when asked for.
    pub read_cam: Option<crate::readcam::ReadCam>,
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

    // Treadmill for an in-place cycle: the planted feet's median
    // horizontal velocity, when the hips go nowhere.
    let belt = treadmill(
        &subject
            .feet
            .iter()
            .map(|foot| {
                (0..n)
                    .map(|f| {
                        foot.points
                            .iter()
                            .map(|&(nd, y)| (pos(f, nd), pos(f, nd).y - y))
                            .collect()
                    })
                    .collect()
            })
            .collect::<Vec<_>>(),
        &(0..n).map(|f| pos(f, subject.hips)).collect::<Vec<_>>(),
        dt,
        k,
    );
    let t0 = times.first().copied().unwrap_or(0.0);
    // Foot positions on the treadmill (the world's when there is none).
    let fpos = |f: usize, node: usize| pos(f, node) - belt.scale(times[f] - t0);

    // Feet.
    let mut feet = FeetScore {
        belt_mps: belt.length() / k,
        ..FeetScore::default()
    };
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
            flat(fpos(b, node) - fpos(a, node)).length() / ((b - a) as f64 * dt)
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
                .map(|g| flat(fpos(g + 1, low_node[g]) - fpos(g, low_node[g])).length())
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

    // Back-and-forth (jitter that a fast clean move does not trip).
    let rough: Vec<RoughScore> = subject
        .chains
        .iter()
        .map(|(cname, nodes)| {
            let mut sum = 0.0;
            let mut count = 0usize;
            let mut peak = 0.0;
            let mut peak_frame = 0;
            if n >= 4 {
                let p: Vec<Vec<Vec3>> = (0..n)
                    .map(|f| nodes.iter().map(|&j| pos(f, j) - pos(f, hips)).collect())
                    .collect();
                let acc = |f: usize, j: usize| p[f + 1][j] - p[f][j].scale(2.0) + p[f - 1][j];
                // How far frame g+1's acceleration points back against
                // frame g's.
                let flip = |g: usize, j: usize| -> f64 {
                    let (a, b) = (acc(g, j), acc(g + 1, j));
                    let big = a.length().max(b.length());
                    if big > 0.0 {
                        (-a.dot(b)).max(0.0) / big
                    } else {
                        0.0
                    }
                };
                for f in 1..n - 2 {
                    for j in 0..nodes.len() {
                        // A flip counts only inside a run of flips (the
                        // frame before or after flips too): a snap (a strike
                        // that starts and stops within two frames, a wind-back
                        // turned straight into the strike) flips once; jitter
                        // and a zigzag keep flipping.
                        let here = flip(f, j);
                        let before = f >= 2 && flip(f - 1, j) > 0.0;
                        let after = f + 2 < n - 1 && flip(f + 1, j) > 0.0;
                        let rev = if before || after { here } else { 0.0 };
                        let mm = rev / k * 1000.0;
                        sum += mm;
                        count += 1;
                        if mm > peak {
                            peak = mm;
                            peak_frame = f;
                        }
                    }
                }
            }
            RoughScore {
                chain: cname.clone(),
                mean_mm: if count > 0 { sum / count as f64 } else { 0.0 },
                peak_mm: peak,
                peak_frame,
            }
        })
        .collect();

    // Joint limits.
    let mut limits = LimitScore::default();
    {
        let mut past = vec![false; n];
        let mut note = |limits: &mut LimitScore, f: usize, what: String, excess: f64| {
            if excess <= 0.0 {
                return;
            }
            past[f] = true;
            if excess > limits.worst_deg {
                limits.worst_deg = excess;
                limits.worst = what.clone();
                limits.worst_frame = f;
            }
            if !limits.joints.contains(&what) {
                limits.joints.push(what);
            }
        };
        for f in 0..n {
            for h in &subject.hinges {
                let u = pos(f, h.b) - pos(f, h.a);
                let v = pos(f, h.c) - pos(f, h.b);
                if u.length() < 1e-9 || v.length() < 1e-9 {
                    continue;
                }
                let axis = world[f][h.a]
                    .rotation()
                    .rotate_vec(h.axis_local)
                    .normalized();
                // Bend in the hinge plane (both bones projected onto it).
                let up = u - axis.scale(u.dot(axis));
                let vp = v - axis.scale(v.dot(axis));
                let flex = if up.length() < 1e-9 || vp.length() < 1e-9 {
                    0.0
                } else {
                    let c = (up.dot(vp) / (up.length() * vp.length())).clamp(-1.0, 1.0);
                    let s = if up.cross(vp).dot(axis) >= 0.0 {
                        1.0
                    } else {
                        -1.0
                    };
                    h.sign * s * detmath::acos(c).to_degrees()
                };
                let side = (detmath::asin(v.normalized().dot(axis).clamp(-1.0, 1.0))
                    - detmath::asin(u.normalized().dot(axis).clamp(-1.0, 1.0)))
                .to_degrees();
                note(
                    &mut limits,
                    f,
                    format!("{} straighter than straight", h.name),
                    -flex - LIMIT_HYPEREXTEND_DEG,
                );
                note(
                    &mut limits,
                    f,
                    format!("{} bent too far", h.name),
                    flex - LIMIT_FLEX_DEG,
                );
                note(
                    &mut limits,
                    f,
                    format!("{} bent sideways", h.name),
                    side.abs() - LIMIT_SIDEWAYS_DEG,
                );
            }
            for c in &subject.caps {
                let rel = world[f][c.parent]
                    .rotation()
                    .conj()
                    .mul(world[f][c.node].rotation());
                let rest = rig.rest_world[c.parent]
                    .rotation()
                    .conj()
                    .mul(rig.rest_world[c.node].rotation());
                let deg = rest.angle_to(rel).to_degrees();
                note(
                    &mut limits,
                    f,
                    format!("{} turned too far", c.name),
                    deg - c.max_deg,
                );
            }
        }
        limits.frames = past.iter().filter(|p| **p).count();
        limits.joints.sort();
    }

    // Readability of the key poses.
    let read = {
        let probes: Vec<Vec<Vec3>> = (0..n)
            .map(|f| subject.probes.iter().map(|&p| pos(f, p)).collect())
            .collect();
        let keys = if n >= 3 {
            crate::stylize_body::body_keys(
                &crate::stylize_body::raw_speed(&probes, 1.0 / dt),
                0.25,
                3,
            )
        } else {
            (0..n).collect()
        };
        let poses: Vec<usize> = if keys.len() > 2 {
            keys[1..keys.len() - 1].to_vec()
        } else {
            keys.clone()
        };
        let up = Vec3::new(0.0, 1.0, 0.0);
        let hips_turn = |f: usize| {
            world[f][hips]
                .rotation()
                .mul(rig.rest_world[hips].rotation().conj())
        };
        let mut clear_sum = 0.0;
        for &f in &poses {
            let face = flat(hips_turn(f).rotate_vec(subject.forward));
            let face = if face.length() > 1e-9 {
                face.normalized()
            } else {
                subject.forward
            };
            let lat = up.cross(face);
            let torso: Vec<Vec3> = subject.torso.iter().map(|&t| pos(f, t)).collect();
            let mut best: f64 = 0.0;
            for step in 0..4 {
                let a = step as f64 * std::f64::consts::FRAC_PI_4;
                let (sa, ca) = detmath::sin_cos(a);
                let d = face.scale(ca) + lat.scale(sa);
                let right = up.cross(d).normalized();
                let r =
                    subject.shoulder_w * (0.5 * right.dot(lat).abs() + 0.3 * right.dot(face).abs());
                let pr = |p: Vec3| (p.dot(right), p.y);
                let line: Vec<(f64, f64)> = torso.iter().map(|&p| pr(p)).collect();
                let clear = subject
                    .limbs
                    .iter()
                    .filter(|&&l| seg_dist(pr(pos(f, l)), &line) >= r)
                    .count();
                if !subject.limbs.is_empty() {
                    best = best.max(clear as f64 / subject.limbs.len() as f64);
                }
            }
            clear_sum += best;
        }
        let segs: Vec<(usize, usize)> = {
            let mut v = Vec::new();
            let mut add = |a: &str, b: &str| {
                if let (Some(x), Some(y)) = (subject.map.node_of(a), subject.map.node_of(b)) {
                    v.push((x, y));
                }
            };
            add("DEF-spine", "DEF-spine.006");
            for side in ["L", "R"] {
                add(
                    &format!("DEF-upper_arm.{side}"),
                    &format!("DEF-forearm.{side}"),
                );
                add(&format!("DEF-forearm.{side}"), &format!("DEF-hand.{side}"));
                add(&format!("DEF-thigh.{side}"), &format!("DEF-shin.{side}"));
                add(&format!("DEF-shin.{side}"), &format!("DEF-foot.{side}"));
            }
            v
        };
        let dir = |f: usize, (a, b): (usize, usize)| {
            world[f][hips]
                .rotation()
                .conj()
                .rotate_vec(pos(f, b) - pos(f, a))
                .normalized()
        };
        let mut contrast = 0.0;
        let pairs = keys.len().saturating_sub(1);
        for w in keys.windows(2) {
            if segs.is_empty() {
                break;
            }
            let m: f64 = segs
                .iter()
                .map(|&sg| {
                    detmath::acos(dir(w[0], sg).dot(dir(w[1], sg)).clamp(-1.0, 1.0)).to_degrees()
                })
                .sum::<f64>()
                / segs.len() as f64;
            contrast += m;
        }
        ReadScore {
            clear: if poses.is_empty() {
                0.0
            } else {
                clear_sum / poses.len() as f64
            },
            contrast_deg: if pairs > 0 {
                contrast / pairs as f64
            } else {
                0.0
            },
            keys,
        }
    };

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
                let hand_v: Vec<Vec3> = (0..n).map(|f| pos(f, hand)).collect();
                let tip_v: Vec<Vec3> = (0..n).map(tip).collect();
                let hand_speed: Vec<f64> = (0..n)
                    .map(|f| {
                        let (a, b) = (f.saturating_sub(1), (f + 1).min(n - 1));
                        (hand_v[b] - hand_v[a])
                            .scale(1.0 / ((b - a) as f64 * dt))
                            .length()
                    })
                    .collect();
                let tip_speed = |f: usize| {
                    let (a, b) = (f.saturating_sub(1), (f + 1).min(n - 1));
                    (tip_v[b] - tip_v[a])
                        .scale(1.0 / ((b - a) as f64 * dt))
                        .length()
                };
                let (peak_frame, load, w) = strike_timing(&hand_v, &tip_v, dt);
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
                    tip_peak_mps: tip_speed(peak_frame) / k,
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
        rough,
        limits,
        read,
        attack,
        read_cam: None,
    }
}

/// A strike's frames from the weapon hand's and the tip's positions per
/// frame (`dt` apart): the tip's fastest frame (peak); the load, the last
/// frame in the second before the peak where the hand still moved against
/// (or across) the strike's direction, else the slowest hand frame in that
/// second; and where the windup leading into the load began (frames where
/// the hand moved against the strike). Velocities are central
/// differences.
pub fn strike_timing(hand: &[Vec3], tip: &[Vec3], dt: f64) -> (usize, usize, usize) {
    let n = hand.len().min(tip.len());
    if n == 0 {
        return (0, 0, 0);
    }
    let vel = |p: &[Vec3], f: usize| -> Vec3 {
        let (a, b) = (f.saturating_sub(1), (f + 1).min(n - 1));
        (p[b] - p[a]).scale(1.0 / ((b - a) as f64 * dt))
    };
    let hand_speed: Vec<f64> = (0..n).map(|f| vel(hand, f).length()).collect();
    let tip_speed: Vec<f64> = (0..n).map(|f| vel(tip, f).length()).collect();
    let mut peak = 0;
    for f in 0..n {
        if tip_speed[f] > tip_speed[peak] {
            peak = f;
        }
    }
    let dir = vel(tip, peak).normalized();
    let win = ((1.0 / dt).round() as usize).max(1);
    let lo = peak.saturating_sub(win);
    let mut load = peak;
    for f in (lo..peak).rev() {
        if vel(hand, f).dot(dir) <= 0.0 {
            load = f;
            break;
        }
    }
    if load == peak {
        for f in lo..peak {
            if load == peak || hand_speed[f] < hand_speed[load] {
                load = f;
            }
        }
    }
    let mut w = load;
    while w > 0 && vel(hand, w - 1).dot(dir) < 0.0 {
        w -= 1;
    }
    (peak, load, w)
}

/// The treadmill an in-place cycle runs on: per foot, the median
/// horizontal velocity of its low frames (`feet[foot][frame][point]` =
/// each contact point's position and height above standing, the lowest
/// point followed across each frame; `hips` the hips' position per frame); a belt when the hips travel under `BELT_TRAVEL`, each foot
/// ends within `BELT_LOOP` of where it started under the hips, and every
/// foot's median is at least
/// `BELT_SPEED` and agrees in direction (a single foot skimming the floor
/// while the other stands is a step, not a belt). Zero otherwise.
pub fn treadmill(feet: &[Vec<Vec<(Vec3, f64)>>], hips: &[Vec3], dt: f64, k: f64) -> Vec3 {
    let n = hips.len();
    if n < 3
        || feet
            .iter()
            .any(|f| f.len() != n || f.iter().any(|p| p.is_empty()))
    {
        return Vec3::ZERO;
    }
    let hips_travel = flat(hips[n - 1] - hips[0]).length();
    if hips_travel >= BELT_TRAVEL * k {
        return Vec3::ZERO;
    }
    let lowest = |pts: &[(Vec3, f64)]| -> usize {
        (0..pts.len()).fold(0, |a, i| if pts[i].1 < pts[a].1 { i } else { a })
    };
    // A cycle ends where it starts: each foot back in the same place
    // under the hips (a lunge that pulls both feet back is not a belt).
    let loops = feet.iter().all(|foot| {
        let i = lowest(&foot[0]);
        flat((foot[n - 1][i].0 - hips[n - 1]) - (foot[0][i].0 - hips[0])).length() < BELT_LOOP * k
    });
    if !loops {
        return Vec3::ZERO;
    }
    let mut per_foot: Vec<Vec3> = Vec::new();
    for foot in feet {
        let mut vx = Vec::new();
        let mut vz = Vec::new();
        for f in 1..n - 1 {
            // The point that is lowest on this frame, followed across it
            // (a heel-to-ball switch is not a foot-length jump).
            let i = lowest(&foot[f]);
            if foot[f][i].1 < CONTACT_HEIGHT * k {
                let v = (foot[f + 1][i].0 - foot[f - 1][i].0).scale(1.0 / (2.0 * dt));
                vx.push(v.x);
                vz.push(v.z);
            }
        }
        per_foot.push(Vec3::new(median(&mut vx), 0.0, median(&mut vz)));
    }
    let agree = !per_foot.is_empty()
        && per_foot.iter().all(|v| v.length() >= BELT_SPEED * k)
        && per_foot
            .iter()
            .all(|v| v.normalized().dot(per_foot[0].normalized()) > 0.8);
    if agree {
        per_foot
            .iter()
            .fold(Vec3::ZERO, |a, v| a + *v)
            .scale(1.0 / per_foot.len() as f64)
    } else {
        Vec3::ZERO
    }
}

/// Median (0 for none); sorts `v`.
fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = v.len() / 2;
    if v.len() % 2 == 1 {
        v[m]
    } else {
        0.5 * (v[m - 1] + v[m])
    }
}

/// Distance from a 2D point to a polyline.
fn seg_dist(p: (f64, f64), line: &[(f64, f64)]) -> f64 {
    let mut best = f64::INFINITY;
    if line.len() == 1 {
        return ((p.0 - line[0].0).powi(2) + (p.1 - line[0].1).powi(2)).sqrt();
    }
    for w in line.windows(2) {
        let (a, b) = (w[0], w[1]);
        let (dx, dy) = (b.0 - a.0, b.1 - a.1);
        let l2 = dx * dx + dy * dy;
        let t = if l2 > 0.0 {
            (((p.0 - a.0) * dx + (p.1 - a.1) * dy) / l2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let (qx, qy) = (a.0 + t * dx, a.1 + t * dy);
        best = best.min(((p.0 - qx).powi(2) + (p.1 - qy).powi(2)).sqrt());
    }
    best
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
                    ("belt_mps", Json::num(r(f.belt_mps, 3))),
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
        fields.push((
            "rough",
            Json::Arr(
                self.rough
                    .iter()
                    .map(|x| {
                        Json::obj(vec![
                            ("chain", Json::str(&x.chain)),
                            ("mean_mm", Json::num(r(x.mean_mm, 3))),
                            ("peak_mm", Json::num(r(x.peak_mm, 2))),
                            ("peak_frame", Json::num(x.peak_frame as f64)),
                        ])
                    })
                    .collect(),
            ),
        ));
        let l = &self.limits;
        fields.push((
            "limits",
            Json::obj(vec![
                ("frames", Json::num(l.frames as f64)),
                ("share", Json::num(r(l.frames as f64 / n, 4))),
                ("worst_deg", Json::num(r(l.worst_deg, 1))),
                ("worst", Json::str(&l.worst)),
                ("worst_frame", Json::num(l.worst_frame as f64)),
                (
                    "joints",
                    Json::Arr(l.joints.iter().map(|c| Json::str(c)).collect()),
                ),
            ]),
        ));
        fields.push((
            "read",
            Json::obj(vec![
                (
                    "keys",
                    Json::Arr(
                        self.read
                            .keys
                            .iter()
                            .map(|k| Json::num(*k as f64))
                            .collect(),
                    ),
                ),
                ("clear", Json::num(r(self.read.clear, 3))),
                ("contrast_deg", Json::num(r(self.read.contrast_deg, 1))),
            ]),
        ));
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
        if let Some(rc) = &self.read_cam {
            fields.push(("read_cam", rc.to_json()));
        }
        Json::obj(fields)
    }

    /// Largest chain mean back-and-forth (mm) and its chain.
    pub fn rough_worst(&self) -> (&str, f64, f64) {
        let mut best: Option<&RoughScore> = None;
        for j in &self.rough {
            if best.is_none_or(|b| j.mean_mm > b.mean_mm) {
                best = Some(j);
            }
        }
        best.map(|j| (j.chain.as_str(), j.mean_mm, j.peak_mm))
            .unwrap_or(("-", 0.0, 0.0))
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
        let (rc, rm, rp) = self.rough_worst();
        let l = &self.limits;
        s.push_str(&format!(
            "{:<18} rough {:>5.2}/{:>5.1} mm {:<6} limits {:>3} fr (worst {:>4.0} deg {} f{})  read clear {:.2} contrast {:>4.1} deg ({} keys){}\n",
            "",
            rm,
            rp,
            rc,
            l.frames,
            l.worst_deg,
            if l.worst.is_empty() { "-" } else { l.worst.as_str() },
            l.worst_frame,
            self.read.clear,
            self.read.contrast_deg,
            self.read.keys.len(),
            if f.belt_mps > 0.0 {
                format!("  belt {:.2} m/s", f.belt_mps)
            } else {
                String::new()
            }
        ));
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
        if let Some(rc) = &self.read_cam {
            s.push_str(&rc.line());
        }
        s
    }
}
