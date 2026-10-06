//! Standardize a rigged GLB onto the HLL skeleton.
//!
//! Humanoids: joints are renamed to the canonical `DEF-*` names
//! (`humanoid.rs`), reparented into the canonical chain with every
//! joint's world rest transform unchanged (so inverse bind matrices stay
//! valid and the mesh does not move), joints with no canonical
//! counterpart leave the skin and hand their weights to the nearest
//! kept ancestor, and old animations are dropped (they were authored
//! for the old hierarchy; `animate` adds retargeted ones). Last, the
//! twist/helper bones are added at the upper arms and thighs
//! (`helpers.rs`), unweighted until weightforge's fix needs them.
//!
//! Other classes keep their own skeleton; joints only get the `DEF-`
//! prefix the rig contract requires.

use crate::glb::{set_node_trs, Affine, Document};
use crate::humanoid::{self, Mapping};
use crate::json::Json;
use crate::rig::Rig;

#[derive(Clone, Debug, Default)]
pub struct Report {
    pub class: String,
    pub joints_in: usize,
    pub joints_out: usize,
    pub renamed: Vec<(String, String)>,
    pub reparented: Vec<(String, String)>,
    /// `(dropped joint, joint that received its weights)`.
    pub merged: Vec<(String, String)>,
    pub missing: Vec<String>,
    pub animations_dropped: usize,
    pub vertices: usize,
    /// Twist/helper bones added (`helpers.rs`).
    pub helpers: Vec<String>,
    /// Hand sockets added (`contact::insert_sockets`).
    pub sockets: Vec<String>,
}

impl Report {
    pub fn to_json(&self) -> Json {
        let pairs = |v: &[(String, String)], a: &str, b: &str| {
            Json::Arr(
                v.iter()
                    .map(|(x, y)| Json::obj(vec![(a, Json::str(x)), (b, Json::str(y))]))
                    .collect(),
            )
        };
        Json::obj(vec![
            ("class", Json::str(&self.class)),
            ("joints_in", Json::num(self.joints_in as f64)),
            ("joints_out", Json::num(self.joints_out as f64)),
            ("renamed", pairs(&self.renamed, "from", "to")),
            ("reparented", pairs(&self.reparented, "bone", "parent")),
            ("merged", pairs(&self.merged, "from", "into")),
            (
                "missing",
                Json::Arr(self.missing.iter().map(|m| Json::str(m)).collect()),
            ),
            (
                "animations_dropped",
                Json::num(self.animations_dropped as f64),
            ),
            ("skinned_vertices", Json::num(self.vertices as f64)),
            (
                "helpers_added",
                Json::Arr(self.helpers.iter().map(|m| Json::str(m)).collect()),
            ),
            (
                "sockets_added",
                Json::Arr(self.sockets.iter().map(|m| Json::str(m)).collect()),
            ),
        ])
    }
}

#[derive(Debug)]
pub enum Outcome {
    /// Standardized document + report.
    Done(Document, Report),
    /// The rig cannot be standardized (e.g. required bones missing).
    Refused(Report, String),
}

pub fn standardize(doc: &Document, class: &str) -> Result<Outcome, String> {
    let rig = Rig::from_doc(doc)?;
    if rig.skins.is_empty() {
        let report = Report {
            class: class.into(),
            ..Default::default()
        };
        return Ok(Outcome::Refused(
            report,
            "model has no skin (rig it first)".into(),
        ));
    }
    if class == "humanoid" {
        humanoid_standardize(doc, &rig)
    } else {
        prefix_standardize(doc, &rig, class)
    }
}

fn rename_node(out: &mut Document, node: usize, name: &str) {
    if let Some(n) = out.array_mut("nodes").get_mut(node) {
        n.set("name", Json::str(name));
    }
}

/// Rename any non-target node already holding a name we need.
fn clear_collisions(out: &mut Document, rig: &Rig, targets: &[(usize, String)]) {
    for (i, name) in rig.names.iter().enumerate() {
        if targets.iter().any(|(n, c)| c == name && *n != i) {
            rename_node(out, i, &format!("{}-orig", name));
        }
    }
}

fn prefix_standardize(doc: &Document, rig: &Rig, class: &str) -> Result<Outcome, String> {
    let mut out = doc.clone();
    let mut report = Report {
        class: class.into(),
        ..Default::default()
    };
    let joints = rig.joints();
    let mut targets = Vec::new();
    for &j in &joints {
        let name = &rig.names[j];
        if name.starts_with("DEF-") {
            continue;
        }
        let clean: String = name
            .rsplit(':')
            .next()
            .unwrap_or(name)
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '.' || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        targets.push((j, format!("DEF-{}", clean)));
    }
    let mut seen = std::collections::BTreeSet::new();
    for (_, t) in &targets {
        if !seen.insert(t.clone()) || joints.iter().any(|&j| rig.names[j] == *t) {
            return Ok(Outcome::Refused(
                report,
                format!("two joints would both be named {}", t),
            ));
        }
    }
    clear_collisions(&mut out, rig, &targets);
    for (j, t) in &targets {
        report.renamed.push((rig.names[*j].clone(), t.clone()));
        rename_node(&mut out, *j, t);
    }
    report.joints_in = joints.len();
    report.joints_out = joints.len();
    Ok(Outcome::Done(out, report))
}

fn humanoid_standardize(doc: &Document, rig: &Rig) -> Result<Outcome, String> {
    let map: Mapping = humanoid::map_rig(rig);
    let joints = rig.joints();
    let mut report = Report {
        class: "humanoid".into(),
        joints_in: joints.len(),
        missing: map.missing.clone(),
        ..Default::default()
    };
    if !map.complete() {
        return Ok(Outcome::Refused(
            report,
            format!(
                "cannot find the humanoid core bones: {}",
                map.missing.join(", ")
            ),
        ));
    }
    let mut out = doc.clone();
    clear_collisions(&mut out, rig, &map.pairs);
    for (node, name) in &map.pairs {
        if rig.names[*node] != *name {
            report
                .renamed
                .push((rig.names[*node].clone(), name.clone()));
            rename_node(&mut out, *node, name);
        }
    }

    // Reparent kept joints into the canonical chain, keeping world rest.
    let present = |n: &str| map.node_of(n).is_some();
    let mut new_parent: Vec<Option<usize>> = rig.parent.clone();
    for (node, name) in &map.pairs {
        let target = match humanoid::present_parent(name, &present) {
            Some(p) => map.node_of(&p),
            // The hips keep their non-joint ancestors (armature node).
            None => {
                let mut cur = rig.parent[*node];
                while let Some(c) = cur {
                    if map.canonical_of(c).is_none() && !joints.contains(&c) {
                        break;
                    }
                    cur = rig.parent[c];
                }
                cur
            }
        };
        if target != rig.parent[*node] {
            new_parent[*node] = target;
            let parent_name = target.map(|p| {
                map.canonical_of(p)
                    .map(str::to_string)
                    .unwrap_or_else(|| rig.names[p].clone())
            });
            report.reparented.push((
                name.clone(),
                parent_name.unwrap_or_else(|| "(scene root)".into()),
            ));
        }
    }
    // Acyclic check on the new hierarchy.
    for start in 0..new_parent.len() {
        let mut cur = new_parent[start];
        let mut steps = 0;
        while let Some(c) = cur {
            steps += 1;
            if c == start || steps > new_parent.len() {
                return Ok(Outcome::Refused(
                    report,
                    format!("reparenting {} would create a cycle", rig.names[start]),
                ));
            }
            cur = new_parent[c];
        }
    }
    apply_hierarchy(&mut out, rig, &new_parent)?;

    // Weight owners: kept joints keep theirs; dropped joints go to the
    // nearest kept ancestor, else the nearest kept joint by position.
    let kept: Vec<usize> = map.pairs.iter().map(|(n, _)| *n).collect();
    let owner = |j: usize| -> usize {
        if kept.contains(&j) {
            return j;
        }
        let mut cur = rig.parent[j];
        while let Some(c) = cur {
            if kept.contains(&c) {
                return c;
            }
            cur = rig.parent[c];
        }
        let head = rig.head(j);
        let mut best = kept[0];
        let mut best_d = f64::INFINITY;
        for &k in &kept {
            let d = (rig.head(k) - head).length_sq();
            if d < best_d {
                best_d = d;
                best = k;
            }
        }
        best
    };
    for &j in &map.unmapped {
        let into = owner(j);
        report.merged.push((
            rig.names[j].clone(),
            map.canonical_of(into).unwrap_or("?").to_string(),
        ));
    }

    // Rebuild every skin with the kept joints (canonical order) and
    // remap each skinned primitive's weights.
    // Helpers last, as `helpers::insert` adds them, so standardizing twice
    // gives the same skin.
    let mut new_joints: Vec<usize> = map
        .pairs
        .iter()
        .filter(|(_, c)| !crate::helpers::is_helper(c))
        .map(|(n, _)| *n)
        .collect();
    new_joints.extend(
        map.pairs
            .iter()
            .filter(|(_, c)| crate::helpers::is_helper(c))
            .map(|(n, _)| *n),
    );
    let mut skin_ibm_rows: Vec<Vec<f64>> = Vec::new();
    for &j in &new_joints {
        skin_ibm_rows.push(original_ibm(doc, rig, j)?.to_gltf_matrix());
    }
    let ibm = out.push_float_accessor(&skin_ibm_rows, "MAT4", false);
    let hips = map.node_of("DEF-spine").expect("required");
    let joint_json = Json::Arr(new_joints.iter().map(|&j| Json::num(j as f64)).collect());
    let skin_count = rig.skins.len();
    for si in 0..skin_count {
        let skins = out.array_mut("skins");
        let skin = &mut skins[si];
        skin.set("joints", joint_json.clone());
        skin.set("inverseBindMatrices", Json::num(ibm as f64));
        skin.set("skeleton", Json::num(hips as f64));
    }
    let index_of = |node: usize| new_joints.iter().position(|&k| k == node).expect("kept") as u16;
    let mut done: Vec<(usize, usize, usize)> = Vec::new(); // (mesh, prim, skin)
    let node_list: Vec<Json> = doc.array("nodes").to_vec();
    for node in &node_list {
        let (Some(mesh), Some(skin)) = (
            node.get("mesh").and_then(Json::as_usize),
            node.get("skin").and_then(Json::as_usize),
        ) else {
            continue;
        };
        let old_joints = rig
            .skins
            .get(skin)
            .ok_or("node references a missing skin")?
            .clone();
        let prim_count = doc
            .array("meshes")
            .get(mesh)
            .and_then(|m| m.get("primitives"))
            .and_then(Json::as_arr)
            .map(|p| p.len())
            .unwrap_or(0);
        for pi in 0..prim_count {
            if let Some(&(_, _, s)) = done.iter().find(|(m, p, _)| *m == mesh && *p == pi) {
                if rig.skins[s] != old_joints {
                    return Err(format!(
                        "mesh {} is shared by nodes with different skins",
                        mesh
                    ));
                }
                continue;
            }
            done.push((mesh, pi, skin));
            let attrs = doc.array("meshes")[mesh]
                .get("primitives")
                .and_then(Json::as_arr)
                .unwrap()[pi]
                .get("attributes")
                .cloned()
                .unwrap_or(Json::Obj(Vec::new()));
            let mut sets = Vec::new();
            for k in 0..4 {
                let (Some(j), Some(w)) = (
                    attrs.get(&format!("JOINTS_{}", k)).and_then(Json::as_usize),
                    attrs
                        .get(&format!("WEIGHTS_{}", k))
                        .and_then(Json::as_usize),
                ) else {
                    break;
                };
                sets.push((doc.read_accessor(j)?, doc.read_accessor(w)?));
            }
            if sets.is_empty() {
                continue;
            }
            let count = sets[0].0.len();
            let mut joints_rows = Vec::with_capacity(count);
            let mut weight_rows = Vec::with_capacity(count);
            for v in 0..count {
                let mut acc: Vec<(u16, f64)> = Vec::new();
                for (jr, wr) in &sets {
                    let (Some(jrow), Some(wrow)) = (jr.get(v), wr.get(v)) else {
                        return Err("JOINTS/WEIGHTS accessors disagree in length".into());
                    };
                    for c in 0..4 {
                        let w = wrow[c];
                        if w <= 0.0 {
                            continue;
                        }
                        let old = *old_joints
                            .get(jrow[c] as usize)
                            .ok_or("joint index out of range")?;
                        let new = index_of(owner(old));
                        match acc.iter_mut().find(|(j, _)| *j == new) {
                            Some(slot) => slot.1 += w,
                            None => acc.push((new, w)),
                        }
                    }
                }
                acc.sort_by(|a, b| {
                    b.1.partial_cmp(&a.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(a.0.cmp(&b.0))
                });
                acc.truncate(4);
                let total: f64 = acc.iter().map(|(_, w)| w).sum();
                let mut jrow = [0u16; 4];
                let mut wrow = vec![0.0; 4];
                for (c, (j, w)) in acc.iter().enumerate() {
                    jrow[c] = *j;
                    wrow[c] = if total > 0.0 { w / total } else { 0.0 };
                }
                // Renormalize in f32 so the stored sum is 1 within 1e-6.
                let sum32: f32 = wrow.iter().map(|w| *w as f32).sum();
                if total > 0.0 && sum32 != 0.0 {
                    let fix = 1.0 - sum32 as f64;
                    wrow[0] += fix;
                }
                joints_rows.push(jrow);
                weight_rows.push(wrow);
            }
            report.vertices += count;
            let ja = out.push_joints_accessor(&joints_rows);
            let wa = out.push_float_accessor(&weight_rows, "VEC4", false);
            let meshes = out.array_mut("meshes");
            let prim = &mut meshes[mesh]
                .get_mut("primitives")
                .and_then(Json::as_arr_mut)
                .unwrap()[pi];
            let attrs = prim
                .get_mut("attributes")
                .ok_or("primitive without attributes")?;
            for k in 1..4 {
                attrs.remove(&format!("JOINTS_{}", k));
                attrs.remove(&format!("WEIGHTS_{}", k));
            }
            attrs.set("JOINTS_0", Json::num(ja as f64));
            attrs.set("WEIGHTS_0", Json::num(wa as f64));
        }
    }
    report.animations_dropped = doc.array("animations").len();
    out.json.remove("animations");
    // Twist/helper bones at the upper arms and thighs (no weight yet).
    let inserted = crate::helpers::insert(&mut out)?;
    report.joints_out = new_joints.len() + inserted.bones.len();
    report.helpers = inserted.bones;
    // Weapon sockets on the hands (attachment nodes, not joints).
    report.sockets = crate::contact::insert_sockets(&mut out)?;
    // The dropped animations' data leaves the file.
    out.compact();
    Ok(Outcome::Done(out, report))
}

/// The joint's inverse bind matrix from its original skin, or the
/// inverse of its world rest transform when no skin held one.
fn original_ibm(doc: &Document, rig: &Rig, joint: usize) -> Result<Affine, String> {
    for (si, skin) in doc.array("skins").iter().enumerate() {
        let Some(slot) = rig.skins[si].iter().position(|&j| j == joint) else {
            continue;
        };
        if let Some(acc) = skin.get("inverseBindMatrices").and_then(Json::as_usize) {
            let rows = doc.read_accessor(acc)?;
            let m = rows
                .get(slot)
                .ok_or("inverseBindMatrices shorter than joints")?;
            return Ok(Affine::from_gltf_matrix(m));
        }
        return Ok(Affine::IDENTITY);
    }
    rig.rest_world[joint]
        .inverse()
        .ok_or_else(|| format!("joint {} has a singular transform", rig.names[joint]))
}

/// Rewrite `children`/scene roots and the moved nodes' local transforms.
fn apply_hierarchy(
    out: &mut Document,
    rig: &Rig,
    new_parent: &[Option<usize>],
) -> Result<(), String> {
    let n = rig.names.len();
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
    // Keep the original child order, then append moved children.
    for (p, kids) in rig.children.iter().enumerate() {
        for &c in kids {
            if new_parent[c] == Some(p) {
                children[p].push(c);
            }
        }
    }
    for c in 0..n {
        if let Some(p) = new_parent[c] {
            if rig.parent[c] != Some(p) {
                children[p].push(c);
            }
        }
    }
    for i in 0..n {
        if new_parent[i] == rig.parent[i] {
            continue;
        }
        let parent_world = match new_parent[i] {
            Some(p) => rig.rest_world[p],
            None => Affine::IDENTITY,
        };
        let local = parent_world
            .inverse()
            .ok_or_else(|| format!("{} has a singular transform", rig.names[i]))?
            .mul(&rig.rest_world[i]);
        let (t, r, s) = local.decompose();
        let nodes = out.array_mut("nodes");
        set_node_trs(&mut nodes[i], t, r, s);
    }
    let nodes = out.array_mut("nodes");
    for (i, kids) in children.iter().enumerate() {
        if kids.is_empty() {
            nodes[i].remove("children");
        } else {
            nodes[i].set(
                "children",
                Json::Arr(kids.iter().map(|&c| Json::num(c as f64)).collect()),
            );
        }
    }
    // Scenes list roots: drop nodes that gained a parent, add new roots.
    let became_root: Vec<usize> = (0..n)
        .filter(|&i| new_parent[i].is_none() && rig.parent[i].is_some())
        .collect();
    for scene in out.array_mut("scenes").iter_mut() {
        let mut roots: Vec<usize> = scene
            .get("nodes")
            .and_then(Json::as_arr)
            .unwrap_or(&[])
            .iter()
            .filter_map(Json::as_usize)
            .filter(|&r| r < n && new_parent[r].is_none())
            .collect();
        roots.extend(became_root.iter().copied());
        scene.set(
            "nodes",
            Json::Arr(roots.iter().map(|&r| Json::num(r as f64)).collect()),
        );
    }
    Ok(())
}
