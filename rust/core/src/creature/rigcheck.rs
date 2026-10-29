//! From a rigged GLB (skintokens output) to a creature profile, with the
//! checks that pick the best of several rig tries: every leg on the
//! ground has a bone, left and right match, fingers are folded into the
//! hand, and bones that stick out of the body are cut off. Pruned
//! joints hand their skin weights to the nearest kept ancestor and are
//! detached from the node tree.
//!
//! Coordinates: the GLB is Y-up; everything here is converted to the
//! clip convention (Blender Z-up) with (x, y, z) -> (x, -z, y).

use super::profile::{build, retail, Creature};
use crate::clip::{Bone, Skeleton};
use crate::glb::{Affine, Document};
use crate::json::Json;
use crate::math::Vec3;
use crate::rig::Rig;

/// glTF Y-up -> Blender Z-up.
pub fn to_zup(p: Vec3) -> Vec3 {
    Vec3::new(p.x, -p.z, p.y)
}

/// Blender Z-up -> glTF Y-up.
pub fn to_yup(p: Vec3) -> Vec3 {
    Vec3::new(p.x, p.z, -p.y)
}

#[derive(Clone, Debug)]
pub struct RigChecks {
    pub joints_in: usize,
    pub joints_out: usize,
    /// Ground-contact blobs found on the mesh, and how many have a bone.
    pub legs_found: usize,
    pub legs_with_bone: usize,
    /// Off-centre joints and how many have a mirror partner.
    pub side_joints: usize,
    pub side_matched: usize,
    pub fingers_folded: usize,
    pub out_of_body_cut: usize,
    pub out_of_body_left: usize,
    pub forward: Vec3,
    pub forward_reason: String,
}

impl RigChecks {
    /// One number to keep the best try by: legs grounded (weight 1),
    /// symmetry (0.5), minus 0.1 per joint left outside the body.
    pub fn score(&self) -> f64 {
        let legs = if self.legs_found > 0 {
            self.legs_with_bone as f64 / self.legs_found as f64
        } else {
            0.0
        };
        let sym = if self.side_joints > 0 {
            self.side_matched as f64 / self.side_joints as f64
        } else {
            1.0
        };
        legs + 0.5 * sym - 0.1 * self.out_of_body_left as f64
    }

    pub fn to_json(&self) -> Json {
        Json::obj(vec![
            ("score", Json::num(round4(self.score()))),
            ("joints_in", Json::num(self.joints_in as f64)),
            ("joints_out", Json::num(self.joints_out as f64)),
            ("legs_found", Json::num(self.legs_found as f64)),
            ("legs_with_bone", Json::num(self.legs_with_bone as f64)),
            ("side_joints", Json::num(self.side_joints as f64)),
            ("side_matched", Json::num(self.side_matched as f64)),
            ("fingers_folded", Json::num(self.fingers_folded as f64)),
            ("out_of_body_cut", Json::num(self.out_of_body_cut as f64)),
            ("out_of_body_left", Json::num(self.out_of_body_left as f64)),
            ("forward", super::camera::vec3_json(self.forward)),
            ("forward_reason", Json::str(&self.forward_reason)),
        ])
    }
}

impl Default for RigChecks {
    fn default() -> RigChecks {
        RigChecks {
            joints_in: 0,
            joints_out: 0,
            legs_found: 0,
            legs_with_bone: 0,
            side_joints: 0,
            side_matched: 0,
            fingers_folded: 0,
            out_of_body_cut: 0,
            out_of_body_left: 0,
            forward: Vec3::ZERO,
            forward_reason: String::new(),
        }
    }
}

fn round4(v: f64) -> f64 {
    (v * 1e4).round() / 1e4
}

/// Rest-pose mesh in Z-up: vertices and triangles of every skinned
/// primitive (unskinned meshes count too, placed by their node).
pub struct Mesh {
    pub verts: Vec<Vec3>,
    pub tris: Vec<[usize; 3]>,
}

pub fn read_mesh(doc: &Document, rig: &Rig) -> Result<Mesh, String> {
    let mut verts = Vec::new();
    let mut tris = Vec::new();
    for (ni, node) in doc.array("nodes").iter().enumerate() {
        let Some(mesh) = node.get("mesh").and_then(Json::as_usize) else {
            continue;
        };
        let skin = node.get("skin").and_then(Json::as_usize);
        let (joints, ibm): (Vec<usize>, Vec<Affine>) = match skin {
            Some(si) => {
                let joints = rig
                    .skins
                    .get(si)
                    .ok_or("node references a missing skin")?
                    .clone();
                let ibm = match doc.array("skins")[si]
                    .get("inverseBindMatrices")
                    .and_then(Json::as_usize)
                {
                    Some(a) => doc
                        .read_accessor(a)?
                        .iter()
                        .map(|r| Affine::from_gltf_matrix(r))
                        .collect(),
                    None => vec![Affine::IDENTITY; joints.len()],
                };
                (joints, ibm)
            }
            None => (Vec::new(), Vec::new()),
        };
        let mats: Vec<Affine> = joints
            .iter()
            .zip(&ibm)
            .map(|(&j, b)| rig.rest_world[j].mul(b))
            .collect();
        let prims = doc.array("meshes")[mesh]
            .get("primitives")
            .and_then(Json::as_arr)
            .unwrap_or(&[])
            .to_vec();
        for prim in prims {
            let mode = prim.get("mode").and_then(Json::as_usize).unwrap_or(4);
            if mode != 4 {
                continue;
            }
            let Some(attrs) = prim.get("attributes") else {
                continue;
            };
            let Some(pa) = attrs.get("POSITION").and_then(Json::as_usize) else {
                continue;
            };
            let pos = doc.read_accessor(pa)?;
            let skinned = match (
                attrs.get("JOINTS_0").and_then(Json::as_usize),
                attrs.get("WEIGHTS_0").and_then(Json::as_usize),
            ) {
                (Some(ja), Some(wa)) if !joints.is_empty() => {
                    Some((doc.read_accessor(ja)?, doc.read_accessor(wa)?))
                }
                _ => None,
            };
            let base = verts.len();
            for (v, row) in pos.iter().enumerate() {
                let p = Vec3::new(row[0], row[1], row[2]);
                let world = match &skinned {
                    Some((jr, wr)) if v < jr.len() && v < wr.len() => {
                        let mut acc = Vec3::ZERO;
                        let mut total = 0.0;
                        for c in 0..4 {
                            let w = wr[v][c];
                            let k = jr[v][c] as usize;
                            if w <= 0.0 || k >= mats.len() {
                                continue;
                            }
                            acc = acc + mats[k].apply(p).scale(w);
                            total += w;
                        }
                        if total > 0.0 {
                            acc.scale(1.0 / total)
                        } else {
                            rig.rest_world[ni].apply(p)
                        }
                    }
                    _ => rig.rest_world[ni].apply(p),
                };
                verts.push(to_zup(world));
            }
            let idx: Vec<usize> = match prim.get("indices").and_then(Json::as_usize) {
                Some(ia) => doc
                    .read_accessor(ia)?
                    .iter()
                    .map(|r| r[0] as usize)
                    .collect(),
                None => (0..pos.len()).collect(),
            };
            for t in idx.chunks_exact(3) {
                if t.iter().all(|&i| i < pos.len()) {
                    tris.push([base + t[0], base + t[1], base + t[2]]);
                }
            }
        }
    }
    if verts.is_empty() {
        return Err("GLB has no triangle mesh".into());
    }
    Ok(Mesh { verts, tris })
}

impl Mesh {
    pub fn bounds(&self) -> (Vec3, Vec3) {
        let mut lo = Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
        let mut hi = Vec3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        for p in &self.verts {
            lo = Vec3::new(lo.x.min(p.x), lo.y.min(p.y), lo.z.min(p.z));
            hi = Vec3::new(hi.x.max(p.x), hi.y.max(p.y), hi.z.max(p.z));
        }
        (lo, hi)
    }

    /// Parity of ray crossings along `dir` from `p` (Möller–Trumbore).
    fn crossings(&self, p: Vec3, dir: Vec3) -> usize {
        let mut n = 0;
        for t in &self.tris {
            let (a, b, c) = (self.verts[t[0]], self.verts[t[1]], self.verts[t[2]]);
            let e1 = b - a;
            let e2 = c - a;
            let h = dir.cross(e2);
            let det = e1.dot(h);
            if det.abs() < 1e-12 {
                continue;
            }
            let inv = 1.0 / det;
            let s = p - a;
            let u = inv * s.dot(h);
            if !(0.0..=1.0).contains(&u) {
                continue;
            }
            let q = s.cross(e1);
            let v = inv * dir.dot(q);
            if v < 0.0 || u + v > 1.0 {
                continue;
            }
            let dist = inv * e2.dot(q);
            if dist > 1e-9 {
                n += 1;
            }
        }
        n
    }

    /// Inside test by majority of three axis rays (robust to edge hits).
    pub fn contains(&self, p: Vec3) -> bool {
        let dirs = [
            Vec3::new(0.7071067811865476, 0.3, 0.6435),
            Vec3::new(-0.21, 0.9, -0.38),
            Vec3::new(0.33, -0.57, 0.75),
        ];
        let inside = dirs
            .iter()
            .filter(|d| self.crossings(p, d.normalized()) % 2 == 1)
            .count();
        inside >= 2
    }

    /// Distance to the nearest vertex (a cheap stand-in for surface distance).
    pub fn nearest_vertex_distance(&self, p: Vec3) -> f64 {
        let mut best = f64::INFINITY;
        for v in &self.verts {
            let d = (*v - p).length_sq();
            if d < best {
                best = d;
            }
        }
        best.sqrt()
    }
}

/// Ground-contact blobs: clusters of vertices within `band` of the lowest
/// point, joined when within `cell` of each other in the ground plane.
/// Returns cluster centroids (Z-up), largest first, ties by x then y.
pub fn ground_blobs(mesh: &Mesh, band: f64, cell: f64, min_verts: usize) -> Vec<Vec3> {
    let z0 = mesh.verts.iter().map(|v| v.z).fold(f64::INFINITY, f64::min);
    let low: Vec<Vec3> = mesh
        .verts
        .iter()
        .copied()
        .filter(|v| v.z <= z0 + band)
        .collect();
    let n = low.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    // Grid buckets keyed by cell; neighbours in the 3x3 around are joined.
    let key = |v: Vec3| ((v.x / cell).floor() as i64, (v.y / cell).floor() as i64);
    let mut keys: Vec<((i64, i64), usize)> = low.iter().map(|v| key(*v)).zip(0..n).collect();
    keys.sort();
    for i in 0..n {
        let (kx, ky) = key(low[i]);
        for dx in -1..=1 {
            for dy in -1..=1 {
                let k = (kx + dx, ky + dy);
                let start = keys.partition_point(|(kk, _)| *kk < k);
                let mut s = start;
                while s < keys.len() && keys[s].0 == k {
                    let j = keys[s].1;
                    if (low[i] - low[j]).length() <= cell {
                        let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                        if a != b {
                            parent[a.max(b)] = a.min(b);
                        }
                    }
                    s += 1;
                }
            }
        }
    }
    let mut groups: Vec<(usize, Vec3, usize)> = Vec::new(); // (root, sum, count)
    for i in 0..n {
        let r = find(&mut parent, i);
        match groups.iter_mut().find(|g| g.0 == r) {
            Some(g) => {
                g.1 = g.1 + low[i];
                g.2 += 1;
            }
            None => groups.push((r, low[i], 1)),
        }
    }
    let mut out: Vec<(usize, Vec3)> = groups
        .into_iter()
        .filter(|g| g.2 >= min_verts)
        .map(|g| (g.2, g.1.scale(1.0 / g.2 as f64)))
        .collect();
    out.sort_by(|a, b| {
        b.0.cmp(&a.0)
            .then(a.1.x.partial_cmp(&b.1.x).unwrap())
            .then(a.1.y.partial_cmp(&b.1.y).unwrap())
    });
    out.into_iter().map(|(_, c)| c).collect()
}

/// Facing direction from the mesh: the horizontal offset from the body
/// centre to the top of the creature (ears, head, crest). Bipeds carry
/// their head over the centre; then +Z glTF (Blender -Y) is assumed.
pub fn guess_forward(mesh: &Mesh, lo: Vec3, hi: Vec3) -> (Vec3, String) {
    let height = hi.z - lo.z;
    let centre = (lo + hi).scale(0.5);
    let top: Vec<Vec3> = mesh
        .verts
        .iter()
        .copied()
        .filter(|v| v.z >= hi.z - 0.08 * height)
        .collect();
    if top.is_empty() {
        return (
            Vec3::new(0.0, -1.0, 0.0),
            "no top vertices; default -Y".into(),
        );
    }
    let mut m = Vec3::ZERO;
    for v in &top {
        m = m + *v;
    }
    let m = m.scale(1.0 / top.len() as f64);
    let off = Vec3::new(m.x - centre.x, m.y - centre.y, 0.0);
    let horizontal = (hi.x - lo.x).max(hi.y - lo.y);
    if off.length() > 0.15 * horizontal {
        // Snap to the nearest axis so the cameras stay square to the body.
        let f = if off.x.abs() >= off.y.abs() {
            Vec3::new(off.x.signum(), 0.0, 0.0)
        } else {
            Vec3::new(0.0, off.y.signum(), 0.0)
        };
        (
            f,
            format!(
                "top of the mesh sits {:.2} m from the centre toward ({:.0}, {:.0})",
                off.length(),
                f.x,
                f.y
            ),
        )
    } else {
        (
            Vec3::new(0.0, -1.0, 0.0),
            "head over the centre; assumed glTF +Z (Blender -Y)".into(),
        )
    }
}

#[derive(Clone, Debug)]
pub struct Options {
    pub forward: Option<Vec3>,
    pub view_size: usize,
    pub name: String,
}

/// Outcome of `from_glb`: the profile, the checks, the cleaned GLB (same
/// as the input when nothing was pruned) and the pruned joint names.
pub struct Built {
    pub creature: Creature,
    pub checks: RigChecks,
    pub glb: Document,
    pub pruned: Vec<String>,
}

pub fn from_glb(doc: &Document, opts: &Options) -> Result<Built, String> {
    let rig = Rig::from_doc(doc)?;
    let joints = rig.joints();
    if joints.is_empty() {
        return Err("GLB has no skin; rig it first (skintokens rig)".into());
    }
    let mesh = read_mesh(doc, &rig)?;
    let (lo, hi) = mesh.bounds();
    let height = (hi.z - lo.z).max(1e-6);

    // Joint nodes in parent-first order, each with its nearest joint ancestor.
    let ordered: Vec<usize> = rig
        .order
        .iter()
        .copied()
        .filter(|n| joints.contains(n))
        .collect();
    let mut jparent: Vec<Option<usize>> = Vec::new(); // index into `ordered`
    for &n in &ordered {
        let mut p = rig.parent[n];
        let mut found = None;
        while let Some(pn) = p {
            if let Some(k) = ordered.iter().position(|&o| o == pn) {
                found = Some(k);
                break;
            }
            p = rig.parent[pn];
        }
        jparent.push(found);
    }
    let heads: Vec<Vec3> = ordered
        .iter()
        .map(|&n| to_zup(rig.rest_world[n].t))
        .collect();
    let names: Vec<String> = ordered.iter().map(|&n| rig.names[n].clone()).collect();
    let nj = ordered.len();
    let mut keep = vec![true; nj];
    let mut checks = RigChecks {
        joints_in: nj,
        ..Default::default()
    };
    let kids = |keep: &[bool], i: usize| -> Vec<usize> {
        (0..nj)
            .filter(|&c| keep[c] && jparent[c] == Some(i))
            .collect()
    };

    // Fingers: leaves within `tiny` of their parent, and hands (a joint
    // with three or more leaf children all within `hand` of it) fold
    // into the parent.
    let tiny = 0.03 * height;
    let hand = 0.09 * height;
    loop {
        let mut changed = false;
        for i in 0..nj {
            if !keep[i] {
                continue;
            }
            let ks = kids(&keep, i);
            let leaves: Vec<usize> = ks
                .iter()
                .copied()
                .filter(|&c| kids(&keep, c).is_empty())
                .collect();
            let hand_like = leaves.len() >= 3
                && leaves
                    .iter()
                    .all(|&c| (heads[c] - heads[i]).length() <= hand);
            for &c in &leaves {
                if hand_like || (heads[c] - heads[i]).length() <= tiny {
                    keep[c] = false;
                    checks.fingers_folded += 1;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    // Out-of-body joints: a leaf outside the mesh and clear of its
    // surface is cut; an inner joint outside is counted against the rig.
    let clearance = 0.02 * height;
    loop {
        let mut changed = false;
        for i in 0..nj {
            if !keep[i] || jparent[i].is_none() {
                continue;
            }
            let outside =
                !mesh.contains(heads[i]) && mesh.nearest_vertex_distance(heads[i]) > clearance;
            if outside && kids(&keep, i).is_empty() {
                keep[i] = false;
                checks.out_of_body_cut += 1;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    for i in 0..nj {
        if keep[i] && jparent[i].is_some() {
            let outside =
                !mesh.contains(heads[i]) && mesh.nearest_vertex_distance(heads[i]) > clearance;
            if outside {
                checks.out_of_body_left += 1;
            }
        }
    }
    if !keep[0]
        || jparent
            .iter()
            .zip(&keep)
            .filter(|(p, k)| p.is_none() && **k)
            .count()
            != 1
    {
        return Err("rig must have exactly one root joint".into());
    }

    // Kept skeleton (parent-first order preserved).
    let map: Vec<Option<usize>> = {
        let mut m = Vec::with_capacity(nj);
        let mut next = 0;
        for &k in &keep {
            m.push(if k {
                next += 1;
                Some(next - 1)
            } else {
                None
            });
        }
        m
    };
    let mut bones: Vec<Bone> = Vec::new();
    for i in 0..nj {
        if !keep[i] {
            continue;
        }
        let mut p = jparent[i];
        while let Some(pi) = p {
            if keep[pi] {
                break;
            }
            p = jparent[pi];
        }
        bones.push(Bone {
            name: names[i].clone(),
            parent: p.and_then(|pi| map[pi]),
            head: heads[i],
            tail: heads[i],
        });
    }
    retail(&mut bones, 0.01 * height);
    let skeleton = Skeleton { bones };
    checks.joints_out = skeleton.len();

    // Legs on the ground: each contact blob needs a bone end near it.
    let min_verts = (mesh.verts.len() / 2000).clamp(3, 20);
    let blobs = ground_blobs(&mesh, 0.03 * height, 0.05 * height, min_verts);
    checks.legs_found = blobs.len();
    let mut feet: Vec<usize> = Vec::new();
    let reach = 0.12 * height;
    for blob in &blobs {
        let mut best: Option<(f64, usize)> = None;
        for (i, b) in skeleton.bones.iter().enumerate() {
            // Leaf bones' tails are synthetic; use heads for leaves, tails otherwise.
            let is_leaf = !skeleton.bones.iter().any(|c| c.parent == Some(i));
            let end = if is_leaf { b.head } else { b.tail };
            let dxy = Vec3::new(end.x - blob.x, end.y - blob.y, 0.0).length();
            let low = end.z <= lo.z + 0.2 * height;
            if low && dxy <= reach {
                let d = dxy + (end.z - lo.z).abs();
                if best.map(|(bd, _)| d < bd).unwrap_or(true) {
                    best = Some((d, i));
                }
            }
        }
        if let Some((_, i)) = best {
            checks.legs_with_bone += 1;
            // The foot is the lowest bone of that chain: walk down while a
            // single child keeps the chain, then take the deepest.
            let mut foot = i;
            loop {
                let ks: Vec<usize> = (0..skeleton.len())
                    .filter(|&c| skeleton.bones[c].parent == Some(foot))
                    .collect();
                if ks.len() == 1 && skeleton.bones[ks[0]].head.z <= lo.z + 0.2 * height {
                    foot = ks[0];
                } else {
                    break;
                }
            }
            if !feet.contains(&foot) {
                feet.push(foot);
            }
        }
    }
    feet.sort();

    // Facing.
    let (forward, reason) = match opts.forward {
        Some(f) => (Vec3::new(f.x, f.y, 0.0).normalized(), "given".to_string()),
        None => guess_forward(&mesh, lo, hi),
    };
    checks.forward = forward;
    checks.forward_reason = reason;

    // Symmetry about the plane through the root along forward/up.
    let side = forward.cross(Vec3::new(0.0, 0.0, 1.0)).normalized();
    let root_head = skeleton.bones[0].head;
    let off_axis = 0.05 * height;
    let tol = 0.06 * height;
    for b in &skeleton.bones {
        let s = side.dot(b.head - root_head);
        if s.abs() <= off_axis {
            continue;
        }
        checks.side_joints += 1;
        let mirrored = b.head - side.scale(2.0 * s);
        if skeleton
            .bones
            .iter()
            .any(|o| (o.head - mirrored).length() <= tol)
        {
            checks.side_matched += 1;
        }
    }

    // Cameras and height come from the mesh alone, so every rig try of
    // one creature shares the same start frame.
    let creature = build(&opts.name, skeleton, feet, forward, lo, hi, opts.view_size);

    // Cleaned GLB.
    let pruned: Vec<String> = (0..nj)
        .filter(|&i| !keep[i])
        .map(|i| names[i].clone())
        .collect();
    let glb = if pruned.is_empty() {
        doc.clone()
    } else {
        let pruned_nodes: Vec<usize> = (0..nj).filter(|&i| !keep[i]).map(|i| ordered[i]).collect();
        prune_joints(doc, &rig, &pruned_nodes)?
    };
    Ok(Built {
        creature,
        checks,
        glb,
        pruned,
    })
}

/// Drop joints from every skin: their weights go to the nearest kept
/// ancestor, the nodes are detached from their parents.
fn prune_joints(doc: &Document, rig: &Rig, pruned: &[usize]) -> Result<Document, String> {
    let mut out = doc.clone();
    let is_pruned = |n: usize| pruned.contains(&n);
    let kept_ancestor = |mut n: usize| -> Option<usize> {
        while let Some(p) = rig.parent[n] {
            if !is_pruned(p) {
                return Some(p);
            }
            n = p;
        }
        None
    };
    for si in 0..rig.skins.len() {
        let joints = &rig.skins[si];
        let ibm_acc = doc.array("skins")[si]
            .get("inverseBindMatrices")
            .and_then(Json::as_usize);
        let ibm: Option<Vec<Vec<f64>>> = match ibm_acc {
            Some(a) => Some(doc.read_accessor(a)?),
            None => None,
        };
        let new_joints: Vec<usize> = joints.iter().copied().filter(|&j| !is_pruned(j)).collect();
        if new_joints.is_empty() {
            return Err("pruning would empty a skin".into());
        }
        // old skin slot -> new skin slot
        let slot: Vec<Option<usize>> = joints
            .iter()
            .map(|&j| {
                let target = if is_pruned(j) {
                    kept_ancestor(j)
                } else {
                    Some(j)
                };
                target.and_then(|t| new_joints.iter().position(|&k| k == t))
            })
            .collect();
        // Rewrite JOINTS_0/WEIGHTS_0 of every primitive using this skin.
        let node_meshes: Vec<usize> = doc
            .array("nodes")
            .iter()
            .filter(|n| n.get("skin").and_then(Json::as_usize) == Some(si))
            .filter_map(|n| n.get("mesh").and_then(Json::as_usize))
            .collect();
        for mesh in node_meshes {
            let prims = doc.array("meshes")[mesh]
                .get("primitives")
                .and_then(Json::as_arr)
                .unwrap_or(&[])
                .to_vec();
            for (pi, prim) in prims.iter().enumerate() {
                let Some(attrs) = prim.get("attributes") else {
                    continue;
                };
                let (Some(ja), Some(wa)) = (
                    attrs.get("JOINTS_0").and_then(Json::as_usize),
                    attrs.get("WEIGHTS_0").and_then(Json::as_usize),
                ) else {
                    continue;
                };
                let jr = doc.read_accessor(ja)?;
                let wr = doc.read_accessor(wa)?;
                let mut joints_rows = Vec::with_capacity(jr.len());
                let mut weight_rows = Vec::with_capacity(jr.len());
                for v in 0..jr.len().min(wr.len()) {
                    let mut acc: Vec<(u16, f64)> = Vec::new();
                    for c in 0..4 {
                        let w = wr[v][c];
                        let k = jr[v][c] as usize;
                        if w <= 0.0 || k >= slot.len() {
                            continue;
                        }
                        let Some(new) = slot[k] else {
                            continue;
                        };
                        let new = new as u16;
                        match acc.iter_mut().find(|(j, _)| *j == new) {
                            Some(s) => s.1 += w,
                            None => acc.push((new, w)),
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
                    let sum32: f32 = wrow.iter().map(|w| *w as f32).sum();
                    if total > 0.0 && sum32 != 0.0 {
                        wrow[0] += 1.0 - sum32 as f64;
                    }
                    joints_rows.push(jrow);
                    weight_rows.push(wrow);
                }
                let nja = out.push_joints_accessor(&joints_rows);
                let nwa = out.push_float_accessor(&weight_rows, "VEC4", false);
                let meshes = out.array_mut("meshes");
                let prim = &mut meshes[mesh]
                    .get_mut("primitives")
                    .and_then(Json::as_arr_mut)
                    .unwrap()[pi];
                let attrs = prim.get_mut("attributes").unwrap();
                attrs.set("JOINTS_0", Json::num(nja as f64));
                attrs.set("WEIGHTS_0", Json::num(nwa as f64));
            }
        }
        // Skin joints + IBMs.
        let new_ibm: Option<usize> = ibm.map(|rows| {
            let kept: Vec<Vec<f64>> = joints
                .iter()
                .zip(rows.iter())
                .filter(|(j, _)| !is_pruned(**j))
                .map(|(_, r)| r.clone())
                .collect();
            out.push_float_accessor(&kept, "MAT4", false)
        });
        let skins = out.array_mut("skins");
        skins[si].set(
            "joints",
            Json::Arr(new_joints.iter().map(|&j| Json::num(j as f64)).collect()),
        );
        if let Some(a) = new_ibm {
            skins[si].set("inverseBindMatrices", Json::num(a as f64));
        }
        if let Some(sk) = skins[si].get("skeleton").and_then(Json::as_usize) {
            if is_pruned(sk) {
                skins[si].remove("skeleton");
            }
        }
    }
    // Detach pruned nodes from their parents' children lists.
    let nodes = out.array_mut("nodes");
    for node in nodes.iter_mut() {
        if let Some(children) = node.get_mut("children").and_then(Json::as_arr_mut) {
            children.retain(|c| c.as_usize().map(|c| !is_pruned(c)).unwrap_or(true));
            if children.is_empty() {
                node.remove("children");
            }
        }
    }
    // Drop animations that drove pruned nodes (none expected from a fresh rig).
    out.json.remove("animations");
    out.compact();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture::{humanoid, Naming};

    #[test]
    fn humanoid_fixture_becomes_a_biped_profile() {
        let doc = humanoid(Naming::Mixamo, false, false).unwrap();
        let built = from_glb(
            &doc,
            &Options {
                forward: None,
                view_size: 256,
                name: "fixture".into(),
            },
        )
        .unwrap();
        let c = &built.creature;
        assert!(c.height > 1.5 && c.height < 1.9, "height {}", c.height);
        assert_eq!(built.checks.legs_found, 2, "{:?}", built.checks);
        assert_eq!(built.checks.legs_with_bone, 2, "{:?}", built.checks);
        assert_eq!(c.feet.len(), 2, "{:?}", c.feet);
        assert!(
            built.checks.side_matched * 10 >= built.checks.side_joints * 9,
            "{:?}",
            built.checks
        );
        assert_eq!(c.views.len(), 2);
        // Fixture faces +Z glTF = -Y Blender.
        assert!(
            c.forward.approx_eq(Vec3::new(0.0, -1.0, 0.0), 1e-9),
            "{:?}",
            built.checks
        );
        // Round trip through text keeps the skeleton.
        let back = Creature::parse(&c.emit().unwrap()).unwrap();
        assert_eq!(back.skeleton.len(), c.skeleton.len());
    }

    #[test]
    fn zup_conversion_is_an_inverse_pair() {
        let p = Vec3::new(0.1, 0.2, 0.3);
        assert!(to_yup(to_zup(p)).approx_eq(p, 1e-15));
        assert!(to_zup(Vec3::new(0.0, 1.0, 0.0)).approx_eq(Vec3::new(0.0, 0.0, 1.0), 1e-15));
    }

    #[test]
    fn mesh_contains_works_on_a_box() {
        let v = |x: f64, y: f64, z: f64| Vec3::new(x, y, z);
        let verts = vec![
            v(0., 0., 0.),
            v(1., 0., 0.),
            v(1., 1., 0.),
            v(0., 1., 0.),
            v(0., 0., 1.),
            v(1., 0., 1.),
            v(1., 1., 1.),
            v(0., 1., 1.),
        ];
        let tris = vec![
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ];
        let m = Mesh { verts, tris };
        assert!(m.contains(v(0.5, 0.5, 0.5)));
        assert!(!m.contains(v(1.5, 0.5, 0.5)));
        let blobs = ground_blobs(&m, 0.01, 2.0, 1);
        assert_eq!(blobs.len(), 1);
    }
}
