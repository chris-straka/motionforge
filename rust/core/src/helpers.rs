//! Twist/helper bones of the HLL humanoid skeleton.
//!
//! Linear blend skinning collapses and stretches the armpit and groin
//! when an upper arm or thigh swings 90 degrees or more: a vertex shared
//! by the chest and the upper arm is averaged between two transforms
//! 90 degrees apart. A helper bone sits on the limb's joint, under the
//! limb's parent, and turns by a share (`SHARE`, half) of the limb's
//! rotation, swing and twist alike. Shared vertices then blend over two
//! 45 degree steps instead of one 90 degree step.
//!
//! The pipeline bakes the helper's rotation into every clip (`animate`)
//! and pose (`posetest`), so a game plays them as ordinary keyed joints
//! with no runtime constraint code. Each helper node carries
//! `extras.hll_helper = {"driver": <limb>, "share": 0.5}` so tools that
//! pose the rig (weightforge) can drive it the same way.

use crate::glb::{node_trs, set_node_trs, Document};
use crate::json::Json;
use crate::math::Quat;
use crate::rig::{Rig, Trs};

/// Fraction of the driver's rotation the helper takes.
pub const SHARE: f64 = 0.5;

/// `(helper, driver)`; the helper's parent is the driver's parent.
pub fn table() -> Vec<(String, String)> {
    let mut out = Vec::new();
    for side in ["L", "R"] {
        // Canonical table order (humanoid.rs): thigh before upper arm.
        out.push((
            format!("DEF-thigh_twist.{side}"),
            format!("DEF-thigh.{side}"),
        ));
        out.push((
            format!("DEF-upper_arm_twist.{side}"),
            format!("DEF-upper_arm.{side}"),
        ));
    }
    out
}

pub fn is_helper(name: &str) -> bool {
    table().iter().any(|(h, _)| h == name)
}

pub fn driver_of(name: &str) -> Option<String> {
    table().into_iter().find(|(h, _)| h == name).map(|(_, d)| d)
}

/// `(helper node, driver node)` for every helper the rig has, when the
/// helper hangs from the driver's parent (the only layout the rule
/// knows).
pub fn present(rig: &Rig) -> Vec<(usize, usize)> {
    table()
        .iter()
        .filter_map(|(h, d)| {
            let (hn, dn) = (rig.find(h)?, rig.find(d)?);
            (rig.parent[hn] == rig.parent[dn]).then_some((hn, dn))
        })
        .collect()
}

/// The helper's local transform for a driver local transform: the rest
/// pose turned by `SHARE` of the driver's change from its own rest.
pub fn helper_local(rig: &Rig, helper: usize, driver: usize, driver_local: Trs) -> Trs {
    let delta = driver_local.r.mul(rig.rest[driver].r.conj());
    let part = Quat::IDENTITY.slerp(delta.normalized(), SHARE);
    let mut out = rig.rest[helper];
    out.r = part.mul(rig.rest[helper].r).normalized();
    // A translated driver (rare for limbs) moves its joint: follow it.
    out.t = rig.rest[helper].t + (driver_local.t - rig.rest[driver].t);
    out
}

/// Set every present helper's local from its driver (posed or rest).
pub fn drive(rig: &Rig, locals: &mut [Option<Trs>]) {
    for (h, d) in present(rig) {
        let dl = locals[d].unwrap_or(rig.rest[d]);
        locals[h] = Some(helper_local(rig, h, d, dl));
    }
}

/// What `insert` did.
#[derive(Clone, Debug, Default)]
pub struct Inserted {
    pub bones: Vec<String>,
}

/// Add the missing helpers of a standardized humanoid: a node at its
/// driver's joint with the driver's rest transform and a skin joint (last
/// in the list, so existing JOINTS indices stay valid) with the driver's
/// inverse bind matrix. They start with no weight: weightforge's fix
/// weights them (its helper band) when the deformation check needs it,
/// and SkinTokens skins around them. No-op for helpers already present.
pub fn insert(doc: &mut Document) -> Result<Inserted, String> {
    let mut done = Inserted::default();
    let rig = Rig::from_doc(doc)?;
    let mut added: Vec<(usize, usize)> = Vec::new(); // (helper node, driver node)
    for (helper, driver) in table() {
        if rig.find(&helper).is_some() {
            continue;
        }
        let Some(dn) = rig.find(&driver) else {
            continue;
        };
        let Some(parent) = rig.parent[dn] else {
            continue;
        };
        if !rig.joints().contains(&dn) {
            continue;
        }
        let (t, r, s) = node_trs(&doc.array("nodes")[dn]);
        let mut node = Json::obj(vec![
            ("name", Json::str(&helper)),
            (
                "extras",
                Json::obj(vec![(
                    "hll_helper",
                    Json::obj(vec![
                        ("driver", Json::str(&driver)),
                        ("share", Json::num(SHARE)),
                    ]),
                )]),
            ),
        ]);
        set_node_trs(&mut node, t, r, s);
        let nodes = doc.array_mut("nodes");
        nodes.push(node);
        let hn = nodes.len() - 1;
        let mut kids = nodes[parent]
            .get("children")
            .and_then(Json::as_arr)
            .map(|k| k.to_vec())
            .unwrap_or_default();
        kids.push(Json::num(hn as f64));
        nodes[parent].set("children", Json::Arr(kids));
        added.push((hn, dn));
        done.bones.push(helper);
    }
    if added.is_empty() {
        return Ok(done);
    }
    let rig = Rig::from_doc(doc)?;

    // Skins: append each helper with its driver's inverse bind matrix.
    let skin_count = doc.array("skins").len();
    for si in 0..skin_count {
        let mut joints = rig.skins[si].clone();
        let mut rows = match doc.array("skins")[si]
            .get("inverseBindMatrices")
            .and_then(Json::as_usize)
        {
            Some(a) => Some(doc.read_accessor(a)?),
            None => None,
        };
        for &(hn, dn) in &added {
            let Some(ds) = joints.iter().position(|&j| j == dn) else {
                continue;
            };
            joints.push(hn);
            if let Some(rows) = rows.as_mut() {
                let row = rows
                    .get(ds)
                    .cloned()
                    .ok_or("inverseBindMatrices shorter than joints")?;
                rows.push(row);
            }
        }
        let skins = doc.array_mut("skins");
        skins[si].set(
            "joints",
            Json::Arr(joints.iter().map(|&j| Json::num(j as f64)).collect()),
        );
        if let Some(rows) = rows {
            let acc = doc.push_float_accessor(&rows, "MAT4", false);
            doc.array_mut("skins")[si].set("inverseBindMatrices", Json::num(acc as f64));
        }
    }
    Ok(done)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_names() {
        assert!(is_helper("DEF-upper_arm_twist.L"));
        assert_eq!(
            driver_of("DEF-thigh_twist.R").as_deref(),
            Some("DEF-thigh.R")
        );
        assert!(!is_helper("DEF-upper_arm.L"));
    }
}
