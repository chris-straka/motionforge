//! Holding the sword: where the handle sits in the hand and how the
//! fingers close round it.
//!
//! The game's hold (`tracks::held_pose`, constants in the hand socket's
//! space) was made for another rig's socket: on a hero with real hands it
//! laid the handle across the back of the wrist with the hand open (the
//! owner: "he's not holding the sword properly"). On a hand with finger
//! bones the hold is read from the hand itself:
//!
//! - the handle runs across the palm under the knuckles, from the little
//!   finger's side out past the index finger (a hammer grip), canted a
//!   little toward the fingers;
//! - its middle is a handle's and a finger's thickness to the palm side of
//!   the knuckle line, a little way down the fingers, so the first finger
//!   bones wrap over it;
//! - each finger then curls (every joint in the proportion of a fist, from
//!   a grips file when one is given) until it touches the handle; the
//!   thumb comes round the other side the same way.
//!
//! The hold is written into the hand socket (origin at the grip's middle,
//! +Y along the blade, +X the leading edge toward the knuckles), so a
//! sword parented to the socket with no further offset sits in the hand.

use crate::glb::Document;
use crate::json::Json;
use crate::keypose::fit::RigInfo;
use crate::math::{Mat3, Quat, Vec3};
use crate::readcam::HEAD_SHARE;
use crate::rig::Trs;

/// Handle and finger half-thickness, metres on a 2 m hero.
pub const HANDLE_R: f64 = 0.016;
pub const FINGER_R: f64 = 0.010;
/// The handle's middle down the fingers from the knuckle line, metres.
pub const DOWN_FINGERS: f64 = 0.020;
/// How far the blade cants toward the fingers (share of the finger
/// direction added to the knuckle line).
pub const CANT: f64 = 0.25;
/// How far a free hand's fingers curl (share of a fist).
pub const RELAXED: f64 = 0.35;

const FINGERS: [&str; 5] = ["f_index", "f_middle", "f_ring", "f_pinky", "thumb"];
/// A fist when no grips file says: radians per joint (root, middle, tip).
const DEFAULT_FIST: [[f64; 3]; 5] = [
    [0.84, 0.99, 0.70],
    [0.92, 1.08, 0.75],
    [0.79, 0.93, 0.64],
    [0.42, 0.50, 0.50],
    [0.26, 0.31, 0.21],
];

/// A hand's hold of the sword, in the hand's own rotation frame (GLB
/// units from the wrist joint).
#[derive(Clone, Copy, Debug)]
pub struct PalmHold {
    /// Blade direction (guard to tip).
    pub blade: Vec3,
    /// The grip's middle from the wrist joint.
    pub grip: Vec3,
    /// The socket's rotation: +Y the blade, +X the leading edge.
    pub socket: Quat,
}

fn chain(info: &RigInfo, finger: &str, s: usize) -> Option<[usize; 3]> {
    let side = ["L", "R"][s];
    let f = |k: usize| info.rig.find(&format!("DEF-{finger}.0{k}.{side}"));
    Some([f(1)?, f(2)?, f(3)?])
}

fn bent(info: &RigInfo, node: usize, angle: f64) -> Trs {
    let mut t = info.rig.rest[node];
    t.r =
        t.r.mul(Quat::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), angle));
    t
}

/// Where the handle sits in hand `s` (module doc); None on a hand with no
/// finger bones. `unit` is GLB units per metre on a 2 m hero.
pub fn palm_hold(info: &RigInfo, s: usize, unit: f64) -> Option<PalmHold> {
    let rig = &info.rig;
    let (index, pinky, middle) = (
        chain(info, "f_index", s)?,
        chain(info, "f_pinky", s)?,
        chain(info, "f_middle", s)?,
    );
    let (ki, kp) = (rig.head(index[0]), rig.head(pinky[0]));
    let along = rig.head(middle[1]) - rig.head(middle[0]);
    if along.length() < 1e-9 || (ki - kp).length() < 1e-9 {
        return None;
    }
    let fd = along.normalized();
    let kn = (ki - kp).normalized();
    // The palm's side: where the middle finger's second joint goes when
    // its first curls.
    let mut l: Vec<Option<Trs>> = vec![None; rig.names.len()];
    l[middle[0]] = Some(bent(info, middle[0], 0.5));
    let moved = rig.world(&l)[middle[1]].t - rig.head(middle[1]);
    let mut palm = fd.cross(kn).normalized();
    if moved.dot(palm) < 0.0 {
        palm = palm.scale(-1.0);
    }
    let centre = (ki + kp).scale(0.5)
        + palm.scale((HANDLE_R + FINGER_R) * unit)
        + fd.scale(DOWN_FINGERS * unit);
    let blade = (kn + fd.scale(CANT)).normalized();
    let edge = (fd - blade.scale(fd.dot(blade))).normalized();
    let world = Quat::from_mat3(Mat3::from_cols(edge, blade, edge.cross(blade)));
    let hw = rig.rest_world[info.hand[s]];
    let inv = hw.rotation().conj();
    Some(PalmHold {
        blade: inv.rotate_vec(blade),
        grip: inv.rotate_vec(centre - hw.t),
        socket: inv.mul(world).normalized(),
    })
}

/// Put each hand's `Socket_Hand_<side>` in its palm (`palm_hold`: the
/// grip's middle, +Y along the blade a fist holds, +X its leading edge),
/// adding the socket when missing. Hands without finger bones are left as
/// they are. Mocap sword clips (Mixamo's) are made for a fist round a
/// handle there: a socket elsewhere points the blade where the clip did
/// not mean it (the game's Quirin clips held the blade along the forearm).
/// Returns the sockets placed.
pub fn place_sockets(doc: &mut Document) -> Result<Vec<String>, String> {
    let info = RigInfo::new(doc)?;
    let rig = &info.rig;
    let head = info.head.ok_or("rig has no head joint")?;
    let h = (rig.head(head).y - info.ground) / HEAD_SHARE;
    let unit = h / 2.0;
    let mut placed = Vec::new();
    for (s, side) in ["L", "R"].iter().enumerate() {
        let Some(ph) = palm_hold(&info, s, unit) else {
            continue;
        };
        let hand = info.hand[s];
        let hw = rig.rest_world[hand];
        let world = hw.t + hw.rotation().rotate_vec(ph.grip);
        let inv = hw.inverse().ok_or("hand transform is singular")?;
        let (_, _, hs) = hw.decompose();
        let inv_s = Vec3::new(1.0 / hs.x, 1.0 / hs.y, 1.0 / hs.z);
        let name = format!("Socket_Hand_{side}");
        let node_index = match info.socket[s] {
            Some(n) => n,
            None => {
                let nodes = doc.array_mut("nodes");
                nodes.push(Json::obj(vec![("name", Json::str(&name))]));
                let sn = nodes.len() - 1;
                let mut kids = nodes[hand]
                    .get("children")
                    .and_then(Json::as_arr)
                    .map(|k| k.to_vec())
                    .unwrap_or_default();
                kids.push(Json::num(sn as f64));
                nodes[hand].set("children", Json::Arr(kids));
                sn
            }
        };
        let node = &mut doc.array_mut("nodes")[node_index];
        crate::glb::set_node_trs(node, inv.apply(world), ph.socket, inv_s);
        placed.push(name);
    }
    Ok(placed)
}

fn line_dist(p: Vec3, on: Vec3, dir: Vec3) -> f64 {
    let v = p - on;
    (v - dir.scale(v.dot(dir))).length()
}

/// Finger joint locals for hand `s`: curled `amount` of a fist, or (None)
/// each finger closed until it touches the handle of `hold`. `fist` gives
/// a joint's bend in a fist by its name (a grips file), else a default.
pub fn fingers(
    info: &RigInfo,
    s: usize,
    hold: Option<&PalmHold>,
    amount: Option<f64>,
    unit: f64,
    fist: &dyn Fn(&str) -> Option<f64>,
) -> Vec<(usize, Trs)> {
    let rig = &info.rig;
    let mut out = Vec::new();
    let hw = rig.rest_world[info.hand[s]];
    let handle = hold.map(|h| {
        (
            hw.t + hw.rotation().rotate_vec(h.grip),
            hw.rotation().rotate_vec(h.blade),
        )
    });
    for (fi, finger) in FINGERS.iter().enumerate() {
        let Some(joints) = chain(info, finger, s) else {
            continue;
        };
        let bends: Vec<f64> = (0..3)
            .map(|k| fist(&rig.names[joints[k]]).unwrap_or(DEFAULT_FIST[fi][k]))
            .collect();
        let pose = |c: f64| -> Vec<Option<Trs>> {
            let mut l: Vec<Option<Trs>> = vec![None; rig.names.len()];
            for k in 0..3 {
                l[joints[k]] = Some(bent(info, joints[k], c * bends[k]));
            }
            l
        };
        let c = match (amount, handle) {
            (Some(a), _) => a,
            (None, None) => RELAXED,
            (None, Some((on, dir))) => {
                // Close until any part of the finger touches the handle.
                let touch = (HANDLE_R + FINGER_R) * unit;
                let most = if *finger == "thumb" { 3.0 } else { 1.6 };
                let mut c = 0.0;
                let mut found = 1.0;
                while c <= most {
                    let w = rig.world(&pose(c));
                    let (p1, p2) = (w[joints[1]].t, w[joints[2]].t);
                    let tip = p2 + (p2 - p1).scale(0.8);
                    let hit = [(p1, p2), (p2, tip)].iter().any(|(a, b)| {
                        (0..=4).any(|k| line_dist(a.lerp(*b, k as f64 / 4.0), on, dir) <= touch)
                    });
                    if hit {
                        found = c;
                        break;
                    }
                    c += 0.02;
                }
                found
            }
        };
        for k in 0..3 {
            out.push((joints[k], bent(info, joints[k], c * bends[k])));
        }
    }
    out
}

/// The `fist` entry of a hand-tools grips file (`grips.ron`): every
/// `"joint name": radians` pair under it, both hands.
pub fn parse_grips(text: &str, entry: &str) -> Result<Vec<(String, f64)>, String> {
    let key = format!("\"{entry}\"");
    let at = text
        .find(&key)
        .ok_or_else(|| format!("no \"{entry}\" entry in the grips file"))?;
    let rest = &text[at + key.len()..];
    let open = rest.find('{').ok_or("grips file: no { after the entry")?;
    // The entry's own block: to its matching brace.
    let mut depth = 0usize;
    let mut end = None;
    for (i, ch) in rest[open..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end = Some(open + i);
                    break;
                }
            }
            _ => {}
        }
    }
    let block = &rest[open..end.ok_or("grips file: unbalanced braces")?];
    let mut out = Vec::new();
    let mut parts = block.split('"');
    parts.next();
    while let (Some(name), Some(after)) = (parts.next(), parts.next()) {
        let after = after.trim_start();
        let Some(val) = after.strip_prefix(':') else {
            continue;
        };
        let num: String = val
            .trim_start()
            .chars()
            .take_while(|c| c.is_ascii_digit() || matches!(c, '.' | '-' | '+' | 'e' | 'E'))
            .collect();
        if let Ok(v) = num.parse::<f64>() {
            if !v.is_finite() || v.abs() > 4.0 {
                return Err(format!("grips file: {name} bends {v} rad"));
            }
            out.push((name.to_string(), v));
        }
    }
    if out.is_empty() {
        return Err(format!("grips file: \"{entry}\" has no joints"));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_grips_file_gives_the_named_entrys_joints_only() {
        let text = r#"( grips: {
            "door": { "R": { "DEF-f_index.01.R": 0.1, }, },
            "fist": {
                "L": { "DEF-f_index.01.L": 0.8116, "DEF-thumb.03.L": 0.2142, },
                "R": { "DEF-f_index.01.R": 0.8378, },
            },
            "sword": { "R": { "DEF-f_index.01.R": 0.2880, }, },
        }, )"#;
        let g = parse_grips(text, "fist").unwrap();
        assert_eq!(
            g,
            vec![
                ("DEF-f_index.01.L".to_string(), 0.8116),
                ("DEF-thumb.03.L".to_string(), 0.2142),
                ("DEF-f_index.01.R".to_string(), 0.8378),
            ]
        );
        assert!(parse_grips(text, "claw").is_err());
        assert!(parse_grips(r#""fist": { "R": { "x": 9.0 } }"#, "fist").is_err());
    }
}
