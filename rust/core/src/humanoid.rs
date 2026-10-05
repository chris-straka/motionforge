//! The HLL humanoid skeleton and an auto-mapper from other rigs onto it.
//!
//! Canonical names are rigforge's `hll_hero` deform bones (Rigify `DEF-*`
//! naming), so clips made on the rigforge hero rig and clips retargeted
//! by motionforge address the same bones. The table below is the bone
//! set of rigforge's mobile game export (65 bones) plus four twist/helper
//! bones (`helpers.rs`: upper arms and thighs); its parents are the
//! anatomical chain (thigh under the hips, arm under the shoulder),
//! not Rigify's flat deform parenting, so local rotations mean the same
//! thing on every character. Rigs keep whatever subset they have; the
//! required core is what animation needs.
//!
//! The mapper understands Rigify `DEF-*`, Mixamo (`mixamorig:` prefix
//! or bare, which is also Tripo's `spec: mixamo`), Unreal-style
//! (`upperarm_l`, `calf_r`, `spine_01`) and plain names (`UpperArm_L`,
//! `Thigh.R`, `Clavicle_L`). The spine between hips and neck is mapped
//! by position in the chain, not by name.

use crate::rig::Rig;

/// `(name, canonical parent)`; parents always precede children.
pub fn canonical() -> Vec<(String, Option<String>)> {
    let mut out: Vec<(String, Option<String>)> = Vec::new();
    let mut add = |name: String, parent: Option<String>| out.push((name, parent));
    add("DEF-spine".into(), None);
    for (i, p) in [
        "DEF-spine",
        "DEF-spine.001",
        "DEF-spine.002",
        "DEF-spine.003",
        "DEF-spine.004",
        "DEF-spine.005",
    ]
    .iter()
    .enumerate()
    {
        add(format!("DEF-spine.{:03}", i + 1), Some(p.to_string()));
    }
    for side in ["L", "R"] {
        let s = |b: &str| format!("DEF-{}.{}", b, side);
        add(s("pelvis"), Some("DEF-spine".into()));
        add(s("thigh"), Some("DEF-spine".into()));
        add(s("thigh_twist"), Some("DEF-spine".into()));
        add(s("shin"), Some(s("thigh")));
        add(s("foot"), Some(s("shin")));
        add(s("toe"), Some(s("foot")));
        add(s("breast"), Some("DEF-spine.003".into()));
        add(s("shoulder"), Some("DEF-spine.003".into()));
        add(s("upper_arm"), Some(s("shoulder")));
        add(s("upper_arm_twist"), Some(s("shoulder")));
        add(s("forearm"), Some(s("upper_arm")));
        add(s("hand"), Some(s("forearm")));
        for palm in 1..=4 {
            add(format!("DEF-palm.{:02}.{}", palm, side), Some(s("hand")));
        }
        for finger in ["thumb", "f_index", "f_middle", "f_ring", "f_pinky"] {
            for seg in 1..=3 {
                let parent = if seg == 1 {
                    s("hand")
                } else {
                    format!("DEF-{}.{:02}.{}", finger, seg - 1, side)
                };
                add(format!("DEF-{}.{:02}.{}", finger, seg, side), Some(parent));
            }
        }
    }
    out
}

/// Bones every humanoid needs before it can share animation.
pub const REQUIRED: [&str; 14] = [
    "DEF-spine",
    "DEF-spine.006",
    "DEF-upper_arm.L",
    "DEF-forearm.L",
    "DEF-hand.L",
    "DEF-upper_arm.R",
    "DEF-forearm.R",
    "DEF-hand.R",
    "DEF-thigh.L",
    "DEF-shin.L",
    "DEF-foot.L",
    "DEF-thigh.R",
    "DEF-shin.R",
    "DEF-foot.R",
];

pub fn canonical_parent(name: &str) -> Option<String> {
    canonical()
        .into_iter()
        .find(|(n, _)| n == name)
        .and_then(|(_, p)| p)
}

/// Nearest canonical ancestor of `name` present in `present`.
pub fn present_parent(name: &str, present: &dyn Fn(&str) -> bool) -> Option<String> {
    let mut cur = canonical_parent(name);
    while let Some(p) = cur {
        if present(&p) {
            return Some(p);
        }
        cur = canonical_parent(&p);
    }
    None
}

/// Canonical bones whose head defines `name`'s direction, best first
/// (upper arm -> forearm, hand -> middle finger, spine.001 -> spine.002 ...).
pub fn main_children(name: &str) -> Vec<String> {
    let side = if name.ends_with(".L") { "L" } else { "R" };
    match name
        .trim_start_matches("DEF-")
        .trim_end_matches(".L")
        .trim_end_matches(".R")
    {
        "shoulder" => vec![format!("DEF-upper_arm.{}", side)],
        "upper_arm" => vec![format!("DEF-forearm.{}", side)],
        "forearm" => vec![format!("DEF-hand.{}", side)],
        "hand" => vec![
            format!("DEF-f_middle.01.{}", side),
            format!("DEF-f_index.01.{}", side),
        ],
        "thigh" => vec![format!("DEF-shin.{}", side)],
        "shin" => vec![format!("DEF-foot.{}", side)],
        "foot" => vec![format!("DEF-toe.{}", side)],
        "spine" => (1..=6).map(|j| format!("DEF-spine.{:03}", j)).collect(),
        other => {
            if let Some(rest) = other.strip_prefix("spine.") {
                let k: usize = rest.parse().unwrap_or(6);
                (k + 1..=6).map(|j| format!("DEF-spine.{:03}", j)).collect()
            } else if other.starts_with("thumb") || other.starts_with("f_") {
                let seg: usize = other
                    .rsplit('.')
                    .next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(3);
                let base = other.rsplit_once('.').map(|(b, _)| b).unwrap_or(other);
                vec![format!("DEF-{}.{:02}.{}", base, seg + 1, side)]
            } else {
                vec![]
            }
        }
    }
}

/// Result of mapping a rig's joints onto canonical names.
#[derive(Clone, Debug, Default)]
pub struct Mapping {
    /// `(joint node, canonical name)` in canonical table order.
    pub pairs: Vec<(usize, String)>,
    /// Joints with no canonical counterpart.
    pub unmapped: Vec<usize>,
    /// Required canonical bones that were not found.
    pub missing: Vec<String>,
    /// Joints whose canonical name was already taken by another joint.
    pub duplicates: Vec<usize>,
}

impl Mapping {
    pub fn canonical_of(&self, node: usize) -> Option<&str> {
        self.pairs
            .iter()
            .find(|(n, _)| *n == node)
            .map(|(_, c)| c.as_str())
    }

    pub fn node_of(&self, name: &str) -> Option<usize> {
        self.pairs.iter().find(|(_, c)| c == name).map(|(n, _)| *n)
    }

    pub fn complete(&self) -> bool {
        self.missing.is_empty()
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Side {
    L,
    R,
}

/// Lowercase, strip namespaces/known prefixes, split off the side.
fn split_side(raw: &str) -> (String, Option<Side>) {
    let mut s = raw.rsplit(':').next().unwrap_or(raw).to_ascii_lowercase();
    for prefix in [
        "mixamorig_",
        "mixamorig",
        "bip01_",
        "bip01 ",
        "bip001 ",
        "def-",
        "def_",
        "org-",
        "b_",
        "j_",
    ] {
        if let Some(rest) = s.strip_prefix(prefix) {
            s = rest.to_string();
        }
    }
    let seps = ['_', '.', '-', ' '];
    for (word, side) in [("left", Side::L), ("right", Side::R)] {
        if let Some(rest) = s.strip_prefix(word) {
            return (rest.trim_start_matches(seps).to_string(), Some(side));
        }
        if let Some(rest) = s.strip_suffix(word) {
            return (rest.trim_end_matches(seps).to_string(), Some(side));
        }
    }
    for (letter, side) in [('l', Side::L), ('r', Side::R)] {
        for sep in seps {
            if let Some(rest) = s.strip_suffix(&format!("{}{}", sep, letter)) {
                return (rest.to_string(), Some(side));
            }
            if let Some(rest) = s.strip_prefix(&format!("{}{}", letter, sep)) {
                return (rest.to_string(), Some(side));
            }
        }
        // Rigify finger segments: "f_index.01.l" -> handled by the sep
        // rule above; "thumb01l" style is too ambiguous to guess.
    }
    (s, None)
}

fn squash(s: &str) -> String {
    s.chars().filter(|c| c.is_ascii_alphanumeric()).collect()
}

/// Canonical base for a sided bone, from its squashed name.
fn sided_base(base: &str) -> Option<String> {
    let simple = match base {
        "shoulder" | "clavicle" | "collar" | "collarbone" => Some("shoulder"),
        "arm" | "upperarm" | "uparm" | "humerus" => Some("upper_arm"),
        "forearm" | "lowerarm" | "elbow" | "loarm" => Some("forearm"),
        "hand" | "wrist" => Some("hand"),
        "upleg" | "thigh" | "upperleg" | "hip" | "femur" => Some("thigh"),
        "leg" | "lowerleg" | "shin" | "calf" | "knee" | "loleg" => Some("shin"),
        "foot" | "ankle" => Some("foot"),
        "toebase" | "toe" | "toes" | "ball" | "toe0" | "toe01" => Some("toe"),
        "pelvis" => Some("pelvis"),
        "breast" => Some("breast"),
        _ => None,
    };
    if let Some(s) = simple {
        return Some(s.to_string());
    }
    // Fingers: optional "hand" prefix, finger word, segment digits.
    let word_end = base.find(|c: char| c.is_ascii_digit())?;
    let (word, digits) = base.split_at(word_end);
    let seg: usize = digits.parse().ok()?;
    let word = word.strip_prefix("hand").unwrap_or(word);
    let word = word
        .strip_prefix('f')
        .filter(|w| w.len() > 3)
        .unwrap_or(word);
    let finger = match word {
        "thumb" => "thumb",
        "index" => "f_index",
        "middle" => "f_middle",
        "ring" => "f_ring",
        "pinky" | "little" => "f_pinky",
        "palm" => {
            return (1..=4).contains(&seg).then(|| format!("palm.{:02}", seg));
        }
        _ => return None,
    };
    (1..=3)
        .contains(&seg)
        .then(|| format!("{}.{:02}", finger, seg))
}

fn is_spine_word(base: &str) -> bool {
    base.starts_with("spine")
        || base.starts_with("chest")
        || base.starts_with("upperchest")
        || base == "torso"
        || base == "abdomen"
}

/// Map a rig's skin joints onto the canonical humanoid names.
pub fn map_rig(rig: &Rig) -> Mapping {
    let joints = rig.joints();
    let table = canonical();
    let canon: Vec<&str> = table.iter().map(|(n, _)| n.as_str()).collect();
    let mut assigned: Vec<(usize, String)> = Vec::new();
    let mut duplicates = Vec::new();
    let mut take = |node: usize, name: String, assigned: &mut Vec<(usize, String)>| {
        if assigned.iter().any(|(n, _)| *n == node) {
            return;
        }
        match assigned.iter().position(|(_, c)| *c == name) {
            Some(i) => {
                // Keep the joint nearer the root; the other is a duplicate.
                let other = assigned[i].0;
                if rig.is_ancestor(node, other) {
                    duplicates.push(other);
                    assigned[i].0 = node;
                } else {
                    duplicates.push(node);
                }
            }
            None => assigned.push((node, name)),
        }
    };

    // 1. Exact canonical names (rigforge rigs, already standardized rigs).
    for &j in &joints {
        if canon.contains(&rig.names[j].as_str()) {
            take(j, rig.names[j].clone(), &mut assigned);
        }
    }
    // 2. Aliases for sided bones, hips, neck and head.
    let mut neck_candidates = Vec::new();
    for &j in &joints {
        let (base, side) = split_side(&rig.names[j]);
        let base = squash(&base);
        match side {
            Some(side) => {
                if let Some(b) = sided_base(&base) {
                    let suffix = if side == Side::L { "L" } else { "R" };
                    take(j, format!("DEF-{}.{}", b, suffix), &mut assigned);
                }
            }
            None => match base.as_str() {
                "hips" | "pelvis" | "hip" => take(j, "DEF-spine".into(), &mut assigned),
                "head" => take(j, "DEF-spine.006".into(), &mut assigned),
                b if b.starts_with("neck") => neck_candidates.push(j),
                _ => {}
            },
        }
    }
    // Hips fallback: the joint both thighs hang from.
    let find = |assigned: &Vec<(usize, String)>, name: &str| {
        assigned.iter().find(|(_, c)| c == name).map(|(n, _)| *n)
    };
    if find(&assigned, "DEF-spine").is_none() {
        if let Some(thigh) = find(&assigned, "DEF-thigh.L") {
            if let Some(p) = rig.parent[thigh].filter(|p| joints.contains(p)) {
                take(p, "DEF-spine".into(), &mut assigned);
            }
        }
    }
    // 3. Spine and neck by chain position between hips and head.
    let hips = find(&assigned, "DEF-spine");
    let head = find(&assigned, "DEF-spine.006");
    if let (Some(hips), Some(head)) = (hips, head) {
        if rig.is_ancestor(hips, head) {
            let mut chain = Vec::new();
            let mut cur = rig.parent[head];
            while let Some(c) = cur {
                if c == hips {
                    break;
                }
                chain.push(c);
                cur = rig.parent[c];
            }
            chain.reverse(); // hips -> head order
            let necks: Vec<usize> = chain
                .iter()
                .copied()
                .filter(|c| neck_candidates.contains(c))
                .collect();
            let spines: Vec<usize> = chain
                .iter()
                .copied()
                .filter(|c| !necks.contains(c) && joints.contains(c))
                .filter(|&c| {
                    let (b, side) = split_side(&rig.names[c]);
                    side.is_none()
                        && (is_spine_word(&squash(&b)) || canon.contains(&rig.names[c].as_str()))
                })
                .collect();
            let spine_names: Vec<&str> = match spines.len() {
                0 => vec![],
                1 => vec!["DEF-spine.001"],
                2 => vec!["DEF-spine.001", "DEF-spine.003"],
                _ => vec!["DEF-spine.001", "DEF-spine.002", "DEF-spine.003"],
            };
            let picks: Vec<usize> = match spines.len() {
                0..=3 => spines.clone(),
                n => vec![spines[0], spines[1], spines[n - 1]],
            };
            for (node, name) in picks.into_iter().zip(spine_names) {
                if !canon.contains(&rig.names[node].as_str()) {
                    take(node, name.to_string(), &mut assigned);
                }
            }
            for (node, name) in necks.into_iter().zip(["DEF-spine.004", "DEF-spine.005"]) {
                take(node, name.to_string(), &mut assigned);
            }
        }
    }
    let mut pairs: Vec<(usize, String)> = Vec::new();
    for name in &canon {
        if let Some(&(node, _)) = assigned.iter().find(|(_, c)| c == name) {
            pairs.push((node, name.to_string()));
        }
    }
    let unmapped = joints
        .iter()
        .copied()
        .filter(|j| !pairs.iter().any(|(n, _)| n == j))
        .collect();
    let missing = REQUIRED
        .iter()
        .filter(|r| !pairs.iter().any(|(_, c)| c == *r))
        .map(|r| r.to_string())
        .collect();
    duplicates.sort_unstable();
    duplicates.dedup();
    Mapping {
        pairs,
        unmapped,
        missing,
        duplicates,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_parents_precede_children() {
        let table = canonical();
        assert_eq!(table.len(), 69);
        for (i, (_, parent)) in table.iter().enumerate() {
            if let Some(p) = parent {
                assert!(table[..i].iter().any(|(n, _)| n == p), "{} after child", p);
            }
        }
    }

    #[test]
    fn side_and_alias_parsing() {
        assert_eq!(
            sided_base(&squash(&split_side("mixamorig:LeftArm").0)),
            Some("upper_arm".into())
        );
        assert_eq!(split_side("mixamorig:LeftArm").1, Some(Side::L));
        assert_eq!(split_side("UpperArm_R").1, Some(Side::R));
        assert_eq!(
            sided_base(&squash(&split_side("calf_l").0)),
            Some("shin".into())
        );
        assert_eq!(
            sided_base(&squash(&split_side("LeftHandIndex2").0)),
            Some("f_index.02".into())
        );
        assert_eq!(
            sided_base(&squash(&split_side("index_01_r").0)),
            Some("f_index.01".into())
        );
        assert_eq!(sided_base(&squash(&split_side("RightHandPinky4").0)), None);
        assert_eq!(
            sided_base(&squash(&split_side("Toe_L").0)),
            Some("toe".into())
        );
        assert_eq!(sided_base(&squash(&split_side("Finger1_L").0)), None);
    }

    #[test]
    fn present_parent_skips_missing_bones() {
        let have = |n: &str| ["DEF-spine", "DEF-spine.001", "DEF-upper_arm.L"].contains(&n);
        assert_eq!(
            present_parent("DEF-upper_arm.L", &have),
            Some("DEF-spine.001".into())
        );
        assert_eq!(present_parent("DEF-spine", &have), None);
    }
}
