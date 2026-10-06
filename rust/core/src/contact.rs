//! Contact pass: keep the hands, forearms and a held weapon out of the
//! body in every clip.
//!
//! Retargeted or hand-keyed FK clips know nothing about the character's
//! surface, so a swing made on a slim rig cuts through a broad chest
//! (HLL's attack_1/attack_2 placeholders put the sword hand through the
//! torso). This pass gives the character simple collision proxies and
//! moves the arm out of the body per frame with two-bone IK, eased in
//! and out over time so the swing keeps its timing and arc.
//!
//! Proxies (all sized from the skinned rest mesh, never per character
//! tuning):
//! - torso: per spine segment (hips to neck), a "lozenge" of two
//!   capsules side by side fitted to the segment's cross-section (depth
//!   sets the radius, the extra width sets the side offset), so a flat
//!   chest is not modelled as a fat cylinder;
//! - head: a capsule from the head joint to the crown;
//! - thighs: one capsule each;
//! - arm parts that move: forearm and hand capsules per side, and a
//!   weapon capsule (blade) on the socket of the weapon hand.
//!
//! Pairs that already touch in the rest pose (a hand resting on the
//! thigh in an A-pose) keep that depth as slack, so only motion that
//! goes deeper than rest is changed.
//!
//! The fix, per clip and arm: for each frame, find the wrist offset that
//! clears every contact (iterating: push along the deepest contact's
//! normal, re-solve IK, measure again); then smooth the offsets over time
//! (a max-envelope with a smoothstep falloff of `ramp` seconds, so a
//! one-frame contact still eases in and out); then re-solve every frame
//! with the smoothed target. The shoulder stays put, the elbow stays in
//! its bend plane, and the hand keeps its world rotation, so a held
//! weapon keeps its angle and the hand's path keeps its shape, shifted
//! only where it would cut the body. Rounds repeat until clean (max 3).

use crate::glb::{canonical_quat, Affine, Document};
use crate::humanoid::{self, Mapping};
use crate::json::Json;
use crate::math::{shortest_arc, Quat, Vec3};
use crate::rig::{Rig, Trs};

/// Settings for the pass.
#[derive(Clone, Debug)]
pub struct Options {
    /// Hand that holds the weapon ("R", "L") or none.
    pub weapon_side: Option<String>,
    /// Blade length from the socket to the tip, in character heights
    /// (HLL's sword: tip 1.5 units from the grip on the 2.0-unit Andras).
    pub weapon_length: f64,
    /// Clip-name fragments (case-insensitive) that hold the weapon; other
    /// clips get no weapon proxy. Empty = every clip.
    pub weapon_clips: Vec<String>,
    /// Ease-in/out time of a correction, seconds.
    pub ramp: f64,
    /// Clearance kept between surfaces, in character heights.
    pub margin: f64,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            weapon_side: Some("R".into()),
            weapon_length: 0.75,
            weapon_clips: [
                "attack", "sword", "slash", "combo", "block", "parry", "stab", "swing",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
            ramp: 0.15,
            margin: 0.004,
        }
    }
}

impl Options {
    pub fn weapon_in(&self, clip: &str) -> bool {
        if self.weapon_side.is_none() {
            return false;
        }
        let low = clip.to_ascii_lowercase();
        self.weapon_clips.is_empty() || self.weapon_clips.iter().any(|f| low.contains(f.as_str()))
    }
}

/// A capsule fixed to a bone: endpoints in the bone's rest-world frame.
#[derive(Clone, Debug)]
pub struct Capsule {
    pub name: String,
    pub bone: usize,
    pub a: Vec3,
    pub b: Vec3,
    pub r: f64,
}

impl Capsule {
    fn posed(&self, _inv_rest: &[Affine], world: &[Affine]) -> (Vec3, Vec3) {
        let m = &world[self.bone];
        (m.apply(self.a), m.apply(self.b))
    }

    pub fn to_json(&self, rig: &Rig) -> Json {
        let v = |p: Vec3| {
            Json::Arr(vec![
                Json::num(r4(p.x)),
                Json::num(r4(p.y)),
                Json::num(r4(p.z)),
            ])
        };
        Json::obj(vec![
            ("name", Json::str(&self.name)),
            ("bone", Json::str(&rig.names[self.bone])),
            ("a", v(self.a)),
            ("b", v(self.b)),
            ("r", Json::num(r4(self.r))),
        ])
    }
}

fn r4(v: f64) -> f64 {
    (v * 1e4).round() / 1e4
}

/// One arm: its chain and moving parts.
#[derive(Clone, Debug)]
pub struct Arm {
    pub side: String,
    pub upper: usize,
    pub fore: usize,
    pub hand: usize,
    pub forearm: Capsule,
    pub palm: Capsule,
    pub weapon: Option<Capsule>,
}

/// The character's collision proxies.
#[derive(Clone, Debug)]
pub struct Body {
    pub height: f64,
    pub static_parts: Vec<Capsule>,
    pub arms: Vec<Arm>,
    /// Rest-pose depth allowed per (arm index, part index, body index).
    slack: Vec<f64>,
    inv_rest: Vec<Affine>,
    /// Body surface points (torso, head, thighs) in their bone's rest
    /// frame: (bone, position, normal). They catch shallow contacts the
    /// inner capsules miss; the capsules catch deep ones.
    pub surface: Vec<(usize, Vec3, Vec3)>,
    /// Distinct bones of `surface`, and rest slack per (arm, part, bone).
    surf_bones: Vec<usize>,
    surf_slack: Vec<f64>,
    /// The mesh for the exact check: rest positions, influences, body
    /// triangles (torso/head/thighs), and per arm the vertices of its
    /// forearm and hand.
    mesh_rest: Vec<Vec3>,
    mesh_inf: Vec<Vec<(usize, f64)>>,
    body_tris: Vec<[usize; 3]>,
    arm_verts: Vec<Vec<usize>>,
    /// Rest-pose mesh depth per arm (slack for the exact check).
    mesh_slack: Vec<f64>,
}

/// Skinned rest vertex: world position and (joint node, weight).
pub struct SkinVert {
    pub p: Vec3,
    /// Area-weighted vertex normal at rest (zero when no triangle uses it).
    pub n: Vec3,
    pub inf: Vec<(usize, f64)>,
}

/// Rest positions of every skinned vertex (through the inverse bind
/// matrices, so the mesh node's own transform does not matter).
pub fn skinned_rest(doc: &Document, rig: &Rig) -> Result<Vec<SkinVert>, String> {
    Ok(skinned_mesh(doc, rig)?.0)
}

/// [`skinned_rest`] plus the triangles (indices into the vertex list).
pub fn skinned_mesh(doc: &Document, rig: &Rig) -> Result<(Vec<SkinVert>, Vec<[usize; 3]>), String> {
    let mut out = Vec::new();
    let mut tris_out: Vec<[usize; 3]> = Vec::new();
    for node in doc.array("nodes") {
        let (Some(mesh), Some(skin)) = (
            node.get("mesh").and_then(Json::as_usize),
            node.get("skin").and_then(Json::as_usize),
        ) else {
            continue;
        };
        let joints = rig
            .skins
            .get(skin)
            .ok_or("node references a missing skin")?;
        let ibm: Vec<Affine> = match doc.array("skins")[skin]
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
            let Some(attrs) = prim.get("attributes") else {
                continue;
            };
            let (Some(pa), Some(ja), Some(wa)) = (
                attrs.get("POSITION").and_then(Json::as_usize),
                attrs.get("JOINTS_0").and_then(Json::as_usize),
                attrs.get("WEIGHTS_0").and_then(Json::as_usize),
            ) else {
                continue;
            };
            let (pos, jr, wr) = (
                doc.read_accessor(pa)?,
                doc.read_accessor(ja)?,
                doc.read_accessor(wa)?,
            );
            let base = out.len();
            let count = pos.len().min(jr.len()).min(wr.len());
            for v in 0..count {
                let p = Vec3::new(pos[v][0], pos[v][1], pos[v][2]);
                let mut acc = Vec3::ZERO;
                let mut inf = Vec::new();
                let mut total = 0.0;
                for c in 0..4 {
                    let w = wr[v][c];
                    let k = jr[v][c] as usize;
                    if w <= 0.0 || k >= joints.len() {
                        continue;
                    }
                    acc = acc + mats[k].apply(p).scale(w);
                    total += w;
                    inf.push((joints[k], w));
                }
                out.push(SkinVert {
                    p: if total > 0.0 {
                        acc.scale(1.0 / total)
                    } else {
                        p
                    },
                    n: Vec3::ZERO,
                    inf,
                });
            }
            // Normals from the triangles (mode 4 or unset only).
            let mode = prim.get("mode").and_then(Json::as_usize).unwrap_or(4);
            if mode == 4 {
                let idx: Vec<usize> = match prim.get("indices").and_then(Json::as_usize) {
                    Some(ia) => doc
                        .read_accessor(ia)?
                        .iter()
                        .map(|r| r[0] as usize)
                        .collect(),
                    None => (0..count).collect(),
                };
                for tri in idx.chunks(3) {
                    if tri.len() < 3 || tri.iter().any(|&i| i >= count) {
                        continue;
                    }
                    let (a, b, c) = (
                        out[base + tri[0]].p,
                        out[base + tri[1]].p,
                        out[base + tri[2]].p,
                    );
                    let fnrm = (b - a).cross(c - a);
                    for &i in tri {
                        out[base + i].n = out[base + i].n + fnrm;
                    }
                    tris_out.push([base + tri[0], base + tri[1], base + tri[2]]);
                }
                for v in &mut out[base..] {
                    if v.n.length() > 1e-12 {
                        v.n = v.n.normalized();
                    }
                }
            }
        }
    }
    Ok((out, tris_out))
}

fn dominant(v: &SkinVert) -> usize {
    let mut best = (usize::MAX, -1.0);
    for &(j, w) in &v.inf {
        if w > best.1 || (w == best.1 && j < best.0) {
            best = (j, w);
        }
    }
    best.0
}

fn percentile(mut xs: Vec<f64>, q: f64) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let i = ((xs.len() - 1) as f64 * q).round() as usize;
    xs[i.min(xs.len() - 1)]
}

/// Closest points between segments p1-q1 and p2-q2: (s, t, distance).
pub fn seg_seg(p1: Vec3, q1: Vec3, p2: Vec3, q2: Vec3) -> (f64, f64, f64) {
    let d1 = q1 - p1;
    let d2 = q2 - p2;
    let r = p1 - p2;
    let a = d1.dot(d1);
    let e = d2.dot(d2);
    let f = d2.dot(r);
    let (mut s, mut t);
    if a <= 1e-12 && e <= 1e-12 {
        return (0.0, 0.0, r.length());
    }
    if a <= 1e-12 {
        s = 0.0;
        t = (f / e).clamp(0.0, 1.0);
    } else {
        let c = d1.dot(r);
        if e <= 1e-12 {
            t = 0.0;
            s = (-c / a).clamp(0.0, 1.0);
        } else {
            let b = d1.dot(d2);
            let denom = a * e - b * b;
            s = if denom > 1e-12 {
                ((b * f - c * e) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };
            t = (b * s + f) / e;
            if t < 0.0 {
                t = 0.0;
                s = (-c / a).clamp(0.0, 1.0);
            } else if t > 1.0 {
                t = 1.0;
                s = ((b - c) / a).clamp(0.0, 1.0);
            }
        }
    }
    let c1 = p1 + d1.scale(s);
    let c2 = p2 + d2.scale(t);
    (s, t, (c1 - c2).length())
}

impl Body {
    /// Fit the proxies of a humanoid. `None` when the rig lacks the arm
    /// chains or a torso.
    pub fn build(doc: &Document, rig: &Rig, opts: &Options) -> Result<Option<Body>, String> {
        let map = humanoid::map_rig(rig);
        let (verts, tris) = skinned_mesh(doc, rig)?;
        if verts.is_empty() {
            return Ok(None);
        }
        let up = Vec3::new(0.0, 1.0, 0.0);
        let fwd = crate::animate::forward(rig, &map);
        let left = up.cross(fwd).normalized();
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for v in &verts {
            lo = lo.min(v.p.y);
            hi = hi.max(v.p.y);
        }
        let height = (hi - lo).max(1e-6);
        let owner = owners(rig, &map);
        let dom: Vec<usize> = verts
            .iter()
            .map(|v| owner[dominant(v).min(owner.len() - 1)])
            .collect();
        let head_of = |n: &str| map.node_of(n).map(|i| rig.head(i));
        let to_local = |bone: usize, p: Vec3| {
            rig.rest_world[bone]
                .inverse()
                .map(|m| m.apply(p))
                .unwrap_or(p)
        };

        let mut parts: Vec<Capsule> = Vec::new();
        // Torso: spine segments up to the neck.
        let chain: Vec<(String, usize)> = [
            "DEF-spine",
            "DEF-spine.001",
            "DEF-spine.002",
            "DEF-spine.003",
            "DEF-spine.004",
            "DEF-spine.005",
            "DEF-spine.006",
        ]
        .iter()
        .filter_map(|n| map.node_of(n).map(|i| (n.to_string(), i)))
        .collect();
        let neck_idx = chain
            .iter()
            .position(|(n, _)| n == "DEF-spine.004" || n == "DEF-spine.005");
        let head_node = map.node_of("DEF-spine.006");
        let torso_end = neck_idx.unwrap_or(chain.len().saturating_sub(1));
        for k in 0..torso_end {
            let (name, bone) = (&chain[k].0, chain[k].1);
            let (a, b) = (rig.head(bone), rig.head(chain[k + 1].1));
            let mut set: Vec<Vec3> = verts
                .iter()
                .zip(&dom)
                .filter(|(_, &d)| d == bone)
                .map(|(v, _)| v.p)
                .collect();
            if set.len() < 8 {
                set = verts
                    .iter()
                    .map(|v| v.p)
                    .filter(|p| {
                        let u = (b - a).normalized();
                        let t = (*p - a).dot(u);
                        t >= 0.0 && t <= (b - a).length()
                    })
                    .collect();
            }
            lozenge(&mut parts, name, bone, a, b, &set, left, fwd, &to_local);
        }
        // Head: joint to crown.
        if let Some(hn) = head_node {
            let set: Vec<Vec3> = verts
                .iter()
                .zip(&dom)
                .filter(|(_, &d)| d == hn)
                .map(|(v, _)| v.p)
                .collect();
            if set.len() >= 8 {
                let a = rig.head(hn);
                let top = percentile(set.iter().map(|p| (*p - a).dot(up)).collect(), 0.97);
                let mid = a + up.scale(top * 0.5);
                let r = percentile(
                    set.iter()
                        .map(|p| {
                            let q = *p - mid;
                            (q - up.scale(q.dot(up))).length()
                        })
                        .collect(),
                    0.6,
                );
                let a2 = a + up.scale(r.min(top * 0.5));
                let b2 = a + up.scale((top - r).max(r.min(top * 0.5)));
                parts.push(Capsule {
                    name: "head".into(),
                    bone: hn,
                    a: to_local(hn, a2),
                    b: to_local(hn, b2),
                    r: r * 0.95,
                });
            }
        }
        // Thighs.
        for side in ["L", "R"] {
            let (Some(th), Some(sh)) = (
                map.node_of(&format!("DEF-thigh.{side}")),
                head_of(&format!("DEF-shin.{side}")),
            ) else {
                continue;
            };
            let a = rig.head(th);
            let set: Vec<Vec3> = verts
                .iter()
                .zip(&dom)
                .filter(|(_, &d)| d == th)
                .map(|(v, _)| v.p)
                .collect();
            let r = radial(&set, a, sh, 0.5);
            if r > 0.0 {
                parts.push(Capsule {
                    name: format!("thigh.{side}"),
                    bone: th,
                    a: to_local(th, a + (sh - a).scale(0.15)),
                    b: to_local(th, sh),
                    r: r * 0.9,
                });
            }
        }
        if parts.is_empty() {
            return Ok(None);
        }
        // Arms.
        let mut arms = Vec::new();
        for side in ["L", "R"] {
            let (Some(up_n), Some(fo), Some(ha)) = (
                map.node_of(&format!("DEF-upper_arm.{side}")),
                map.node_of(&format!("DEF-forearm.{side}")),
                map.node_of(&format!("DEF-hand.{side}")),
            ) else {
                continue;
            };
            let (e, w) = (rig.head(fo), rig.head(ha));
            let fset: Vec<Vec3> = verts
                .iter()
                .zip(&dom)
                .filter(|(_, &d)| d == fo)
                .map(|(v, _)| v.p)
                .collect();
            let rf = radial(&fset, e, w, 0.5).max(0.01 * height);
            let hset: Vec<Vec3> = verts
                .iter()
                .zip(&dom)
                .filter(|(_, &d)| d == ha)
                .map(|(v, _)| v.p)
                .collect();
            let hdir = hand_dir(rig, &map, side).unwrap_or((w - e).normalized());
            let hlen = percentile(hset.iter().map(|p| (*p - w).dot(hdir)).collect(), 0.95)
                .max(0.04 * height);
            let tip = w + hdir.scale(hlen);
            let rh = radial(&hset, w, tip, 0.5).max(0.01 * height);
            let forearm = Capsule {
                name: format!("forearm.{side}"),
                bone: fo,
                a: to_local(fo, e + (w - e).scale(0.2)),
                b: to_local(fo, w),
                r: rf * 0.85,
            };
            let palm = Capsule {
                name: format!("hand.{side}"),
                bone: ha,
                a: to_local(ha, w + hdir.scale(hlen * 0.2)),
                b: to_local(ha, w + hdir.scale((hlen - rh).max(hlen * 0.3))),
                r: rh * 0.85,
            };
            let weapon = if opts.weapon_side.as_deref() == Some(side) {
                let (origin, dir) = socket_frame(doc, rig, ha, side)
                    .unwrap_or_else(|| (w + hdir.scale(hlen * 0.45), fwd));
                let len = opts.weapon_length * height;
                Some(Capsule {
                    name: format!("weapon.{side}"),
                    bone: ha,
                    a: to_local(ha, origin + dir.scale(0.12 * len)),
                    b: to_local(ha, origin + dir.scale(len)),
                    r: 0.012 * height,
                })
            } else {
                None
            };
            arms.push(Arm {
                side: side.to_string(),
                upper: up_n,
                fore: fo,
                hand: ha,
                forearm,
                palm,
                weapon,
            });
        }
        if arms.is_empty() {
            return Ok(None);
        }
        let inv_rest: Vec<Affine> = rig
            .rest_world
            .iter()
            .map(|m| m.inverse().unwrap_or(Affine::IDENTITY))
            .collect();
        // Surface points: torso, head and thigh vertices that no arm bone
        // moves, kept in their dominant bone's rest frame.
        let mut body_bones: Vec<usize> = parts.iter().map(|c| c.bone).collect();
        body_bones.sort_unstable();
        body_bones.dedup();
        let arm_bone = |j: usize| {
            let o = owner[j.min(owner.len() - 1)];
            arms.iter()
                .any(|a| o == a.upper || o == a.fore || o == a.hand)
                || map
                    .canonical_of(o)
                    .is_some_and(|n| n.starts_with("DEF-shoulder"))
        };
        let mut surface = Vec::new();
        for (v, &d) in verts.iter().zip(&dom) {
            if v.inf.is_empty() || v.n.length() < 0.5 || !body_bones.contains(&d) {
                continue;
            }
            let wd: f64 = v
                .inf
                .iter()
                .filter(|(j, _)| owner[*j] == d)
                .map(|(_, w)| w)
                .sum();
            let wa: f64 = v
                .inf
                .iter()
                .filter(|(j, _)| arm_bone(*j))
                .map(|(_, w)| w)
                .sum();
            if wd < 0.6 || wa > 0.1 {
                continue;
            }
            let inv = rig.rest_world[d].inverse().unwrap_or(Affine::IDENTITY);
            surface.push((d, inv.apply(v.p), inv.apply_linear(v.n)));
        }
        let surf_bones = body_bones.clone();
        // Exact-check mesh: body triangles (all three corners owned by a
        // body bone and not moved by an arm), arm vertices per side.
        let is_body = |v: usize| {
            let vv = &verts[v];
            !vv.inf.is_empty()
                && body_bones.contains(&dom[v])
                && vv
                    .inf
                    .iter()
                    .filter(|(j, _)| arm_bone(*j))
                    .map(|(_, w)| w)
                    .sum::<f64>()
                    < 0.3
        };
        let body_tris: Vec<[usize; 3]> = tris
            .iter()
            .copied()
            .filter(|t| t.iter().all(|&v| is_body(v)))
            .collect();
        let arm_verts: Vec<Vec<usize>> = arms
            .iter()
            .map(|a| {
                (0..verts.len())
                    .filter(|&v| dom[v] == a.fore || dom[v] == a.hand)
                    .collect()
            })
            .collect();
        let mut body = Body {
            height,
            static_parts: parts,
            arms,
            slack: Vec::new(),
            inv_rest,
            surface,
            surf_bones,
            surf_slack: Vec::new(),
            mesh_rest: verts.iter().map(|v| v.p).collect(),
            mesh_inf: verts.iter().map(|v| v.inf.clone()).collect(),
            body_tris,
            arm_verts,
            mesh_slack: Vec::new(),
        };
        // Rest-pose slack.
        let world = rig.world(&vec![None; rig.names.len()]);
        let nb = body.static_parts.len();
        let mut slack = vec![0.0; body.arms.len() * 3 * nb];
        for (ai, arm) in body.arms.iter().enumerate() {
            for (pi, part) in arm.parts(true).iter().enumerate() {
                for (bi, s) in body.static_parts.iter().enumerate() {
                    let d = depth(part, s, &body.inv_rest, &world, 0.0).0;
                    slack[(ai * 3 + pi) * nb + bi] = d.max(0.0);
                }
            }
        }
        body.slack = slack;
        let nsb = body.surf_bones.len();
        let mut ss: Vec<f64> = vec![0.0; body.arms.len() * 3 * nsb];
        let posed = body.posed_surface(&world);
        for (ai, arm) in body.arms.iter().enumerate() {
            for (pi, part) in arm.parts(true).iter().enumerate() {
                for (k, d, _) in surface_hits(part, &posed, &body.inv_rest, &world, 0.0) {
                    let slot = (ai * 3 + pi) * nsb
                        + body
                            .surf_bones
                            .iter()
                            .position(|&b| b == body.surface[k].0)
                            .unwrap();
                    ss[slot] = ss[slot].max(d);
                }
            }
        }
        body.surf_slack = ss;
        body.mesh_slack = (0..body.arms.len())
            .map(|ai| body.mesh_depth(rig, &world, ai, None).0)
            .collect();
        Ok(Some(body))
    }

    pub fn to_json(&self, rig: &Rig) -> Json {
        let mut all: Vec<Json> = self.static_parts.iter().map(|c| c.to_json(rig)).collect();
        for arm in &self.arms {
            for p in arm.parts(true) {
                all.push(p.to_json(rig));
            }
        }
        Json::obj(vec![
            ("height", Json::num(r4(self.height))),
            ("capsules", Json::Arr(all)),
            ("surface_points", Json::num(self.surface.len() as f64)),
            ("body_triangles", Json::num(self.body_tris.len() as f64)),
            (
                "arm_vertices",
                Json::Arr(
                    self.arm_verts
                        .iter()
                        .map(|v| Json::num(v.len() as f64))
                        .collect(),
                ),
            ),
        ])
    }
}

impl Body {
    fn posed_surface(&self, world: &[Affine]) -> Vec<(Vec3, Vec3)> {
        self.surface
            .iter()
            .map(|(b, p, n)| (world[*b].apply(*p), world[*b].apply_linear(*n).normalized()))
            .collect()
    }
}

impl Body {
    /// Exact check for one arm in one pose: arm vertices (forearm, hand)
    /// and blade sample points inside the body mesh (generalized winding
    /// number over the body triangles, so the open cuts at the neck,
    /// shoulders and hips do not matter). Returns (deepest inside point's
    /// distance to the nearest body vertex, count of inside points).
    pub fn mesh_depth(
        &self,
        rig: &Rig,
        world: &[Affine],
        ai: usize,
        weapon: Option<&Capsule>,
    ) -> (f64, usize) {
        let mats: Vec<Affine> = (0..world.len())
            .map(|j| world[j].mul(&self.inv_rest[j]))
            .collect();
        let skin = |v: usize| {
            let mut acc = Vec3::ZERO;
            let mut tot = 0.0;
            for &(j, w) in &self.mesh_inf[v] {
                acc = acc + mats[j].apply(self.mesh_rest[v]).scale(w);
                tot += w;
            }
            if tot > 0.0 {
                acc.scale(1.0 / tot)
            } else {
                self.mesh_rest[v]
            }
        };
        let _ = rig;
        // Posed body triangles and their bounds.
        let mut used = vec![false; self.mesh_rest.len()];
        for t in &self.body_tris {
            for &v in t {
                used[v] = true;
            }
        }
        let mut posed = vec![Vec3::ZERO; self.mesh_rest.len()];
        let (mut lo, mut hi) = (
            Vec3::new(f64::MAX, f64::MAX, f64::MAX),
            Vec3::new(f64::MIN, f64::MIN, f64::MIN),
        );
        for v in 0..posed.len() {
            if used[v] {
                let p = skin(v);
                posed[v] = p;
                lo = Vec3::new(lo.x.min(p.x), lo.y.min(p.y), lo.z.min(p.z));
                hi = Vec3::new(hi.x.max(p.x), hi.y.max(p.y), hi.z.max(p.z));
            }
        }
        let mut pts: Vec<Vec3> = self.arm_verts[ai].iter().map(|&v| skin(v)).collect();
        if let Some(w) = weapon {
            let (a, b) = w.posed(&self.inv_rest, world);
            for k in 0..=12 {
                pts.push(a + (b - a).scale(k as f64 / 12.0));
            }
        }
        let mut worst: f64 = 0.0;
        let mut count = 0;
        for p in pts {
            if p.x < lo.x || p.y < lo.y || p.z < lo.z || p.x > hi.x || p.y > hi.y || p.z > hi.z {
                continue;
            }
            if winding(&posed, &self.body_tris, p).abs() < 0.5 {
                continue;
            }
            count += 1;
            let d = self
                .body_tris
                .iter()
                .flat_map(|t| t.iter())
                .map(|&v| (posed[v] - p).length())
                .fold(f64::MAX, f64::min);
            worst = worst.max(d);
        }
        (worst, count)
    }
}

/// Blade test mesh for [`grip_search`]: every triangle except the holding
/// hand's, posed per frame.
struct GripMesh {
    rest: Vec<Vec3>,
    inf: Vec<Vec<(usize, f64)>>,
    tris: Vec<[usize; 3]>,
}

impl GripMesh {
    fn posed(&self, world: &[Affine], inv_rest: &[Affine]) -> Vec<Vec3> {
        let mats: Vec<Affine> = (0..world.len())
            .map(|j| world[j].mul(&inv_rest[j]))
            .collect();
        self.rest
            .iter()
            .zip(&self.inf)
            .map(|(p, inf)| {
                let mut acc = Vec3::ZERO;
                let mut tot = 0.0;
                for &(j, w) in inf {
                    acc = acc + mats[j].apply(*p).scale(w);
                    tot += w;
                }
                if tot > 0.0 {
                    acc.scale(1.0 / tot)
                } else {
                    *p
                }
            })
            .collect()
    }
}

/// Find the grip (blade direction in the hand) that a set of weapon clips
/// was animated for: the animator kept the prop out of the body, the
/// other arm, its own forearm and the floor, and swords reach away from
/// the body. Each of 98 directions (a Fibonacci sphere in rest world
/// space) is scored over every frame: blade points inside the mesh
/// (holding hand excluded) or below the lowest rest vertex count as hits;
/// fewer hits win, then more clearance (mean distance of the blade to the
/// nearest vertex, capped at 15% of height). Returns (direction, hits,
/// clearance), best first.
pub fn grip_search(
    doc: &Document,
    rig: &Rig,
    hand: usize,
    origin: Vec3,
    length: f64,
    clips: &[Vec<Vec<Option<Trs>>>],
) -> Result<Vec<(Vec3, usize, f64)>, String> {
    let (verts, tris) = skinned_mesh(doc, rig)?;
    let map = humanoid::map_rig(rig);
    let owner = owners(rig, &map);
    let held = |v: usize| {
        verts[v]
            .inf
            .iter()
            .filter(|(j, _)| owner[*j] == hand)
            .map(|(_, w)| w)
            .sum::<f64>()
            > 0.5
    };
    let floor = verts.iter().map(|v| v.p.y).fold(f64::MAX, f64::min);
    let height = verts.iter().map(|v| v.p.y).fold(f64::MIN, f64::max) - floor;
    let mesh = GripMesh {
        rest: verts.iter().map(|v| v.p).collect(),
        inf: verts.iter().map(|v| v.inf.clone()).collect(),
        tris: tris
            .into_iter()
            .filter(|t| !t.iter().any(|&v| held(v)))
            .collect(),
    };
    let inv_rest: Vec<Affine> = rig
        .rest_world
        .iter()
        .map(|m| m.inverse().unwrap_or(Affine::IDENTITY))
        .collect();
    let n = 98;
    let ga = std::f64::consts::PI * (3.0 - 5f64.sqrt());
    let dirs: Vec<Vec3> = (0..n)
        .map(|i| {
            let y = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
            let r = (1.0 - y * y).max(0.0).sqrt();
            let (s, c) = crate::detmath::sin_cos(ga * i as f64);
            Vec3::new(c * r, y, s * r)
        })
        .collect();
    let inv_hand = inv_rest[hand];
    let local: Vec<Vec<Vec3>> = dirs
        .iter()
        .map(|d| {
            (3..=12)
                .map(|k| inv_hand.apply(origin + d.scale(length * k as f64 / 12.0)))
                .collect()
        })
        .collect();
    let mut hits = vec![0usize; n];
    let mut clear = vec![0.0f64; n];
    let mut samples = 0usize;
    let cap = 0.15 * height;
    for clip in clips {
        for f in clip {
            let world = rig.world(f);
            let posed = mesh.posed(&world, &inv_rest);
            let hw = world[hand];
            for (di, pts) in local.iter().enumerate() {
                for lp in pts {
                    let p = hw.apply(*lp);
                    let near = mesh
                        .tris
                        .iter()
                        .flat_map(|t| t.iter())
                        .map(|&v| (posed[v] - p).length())
                        .fold(f64::MAX, f64::min);
                    let inside =
                        p.y < floor || (near < cap && winding(&posed, &mesh.tris, p).abs() >= 0.5);
                    if inside {
                        hits[di] += 1;
                    }
                    clear[di] += near.min(cap);
                }
            }
            samples += 10;
        }
    }
    let mut out: Vec<(Vec3, usize, f64)> = (0..n)
        .map(|i| (dirs[i], hits[i], clear[i] / samples.max(1) as f64))
        .collect();
    out.sort_by(|x, y| {
        x.1.cmp(&y.1)
            .then(y.2.partial_cmp(&x.2).unwrap_or(std::cmp::Ordering::Equal))
    });
    Ok(out)
}

fn atan2(y: f64, x: f64) -> f64 {
    let r = (x * x + y * y).sqrt();
    if r < 1e-300 {
        return 0.0;
    }
    let a = crate::detmath::acos((x / r).clamp(-1.0, 1.0));
    if y < 0.0 {
        -a
    } else {
        a
    }
}

/// Generalized winding number of `p` (Jacobson et al. 2013): the summed
/// signed solid angle of the triangles over 4 pi.
fn winding(pos: &[Vec3], tris: &[[usize; 3]], p: Vec3) -> f64 {
    let mut sum = 0.0;
    for t in tris {
        let (a, b, c) = (pos[t[0]] - p, pos[t[1]] - p, pos[t[2]] - p);
        let (la, lb, lc) = (a.length(), b.length(), c.length());
        let num = a.dot(b.cross(c));
        let den = la * lb * lc + a.dot(b) * lc + a.dot(c) * lb + b.dot(c) * la;
        sum += 2.0 * atan2(num, den);
    }
    sum / (4.0 * std::f64::consts::PI)
}

/// Surface points inside `part` (grown by `grow`): (index, depth, normal).
/// Depth is how far the part's axis must move along the point's normal
/// to clear it by the part's radius (tangent-plane estimate).
fn surface_hits(
    part: &Capsule,
    posed: &[(Vec3, Vec3)],
    inv_rest: &[Affine],
    world: &[Affine],
    grow: f64,
) -> Vec<(usize, f64, Vec3)> {
    let (a, b) = part.posed(inv_rest, world);
    let r = part.r + grow;
    let ab = b - a;
    let l2 = ab.dot(ab).max(1e-12);
    let mut out = Vec::new();
    for (k, (p, n)) in posed.iter().enumerate() {
        let t = ((*p - a).dot(ab) / l2).clamp(0.0, 1.0);
        let c = a + ab.scale(t);
        if (c - *p).length() >= r {
            continue;
        }
        let s = (c - *p).dot(*n);
        out.push((k, r - s, *n));
    }
    out
}

impl Arm {
    fn parts(&self, weapon: bool) -> Vec<&Capsule> {
        let mut v = vec![&self.forearm, &self.palm];
        if weapon {
            if let Some(w) = &self.weapon {
                v.push(w);
            }
        }
        v
    }
}

/// Each joint's proxy owner: twist/helper bones and the pelvis/breast
/// count as their parents' region.
fn owners(rig: &Rig, map: &Mapping) -> Vec<usize> {
    (0..rig.names.len())
        .map(|i| {
            let name = map.canonical_of(i).unwrap_or("");
            if let Some(d) = crate::helpers::driver_of(name) {
                return map.node_of(&d).unwrap_or(i);
            }
            if name.starts_with("DEF-pelvis") {
                return map.node_of("DEF-spine").unwrap_or(i);
            }
            if name.starts_with("DEF-breast") {
                return map
                    .node_of("DEF-spine.003")
                    .or_else(|| map.node_of("DEF-spine.002"))
                    .unwrap_or(i);
            }
            if name.starts_with("DEF-palm")
                || name.starts_with("DEF-f_")
                || name.starts_with("DEF-thumb")
            {
                let side = if name.ends_with(".L") { "L" } else { "R" };
                return map.node_of(&format!("DEF-hand.{side}")).unwrap_or(i);
            }
            // Unmapped joints (finger stubs) ride the nearest mapped ancestor.
            if map.canonical_of(i).is_none() {
                let mut cur = rig.parent[i];
                while let Some(p) = cur {
                    if map.canonical_of(p).is_some() {
                        return p;
                    }
                    cur = rig.parent[p];
                }
            }
            i
        })
        .collect()
}

/// Median-ish distance of `set` to segment a-b (quantile `q`).
fn radial(set: &[Vec3], a: Vec3, b: Vec3, q: f64) -> f64 {
    let ab = b - a;
    let l2 = ab.dot(ab).max(1e-12);
    percentile(
        set.iter()
            .map(|p| {
                let t = ((*p - a).dot(ab) / l2).clamp(0.0, 1.0);
                (*p - (a + ab.scale(t))).length()
            })
            .collect(),
        q,
    )
}

#[allow(clippy::too_many_arguments)]
fn lozenge(
    parts: &mut Vec<Capsule>,
    name: &str,
    bone: usize,
    a: Vec3,
    b: Vec3,
    set: &[Vec3],
    left: Vec3,
    fwd: Vec3,
    to_local: &dyn Fn(usize, Vec3) -> Vec3,
) {
    let u = (b - a).normalized();
    let len = (b - a).length();
    let lp = (left - u.scale(left.dot(u))).normalized();
    let fp = (fwd - u.scale(fwd.dot(u))).normalized();
    let pts: Vec<Vec3> = set
        .iter()
        .copied()
        .filter(|p| {
            let t = (*p - a).dot(u);
            t >= -0.1 * len && t <= 1.1 * len
        })
        .collect();
    if pts.len() < 8 {
        return;
    }
    let xs: Vec<f64> = pts.iter().map(|p| (*p - a).dot(lp)).collect();
    let ys: Vec<f64> = pts.iter().map(|p| (*p - a).dot(fp)).collect();
    let (x0, x1) = (percentile(xs.clone(), 0.08), percentile(xs, 0.92));
    let (y0, y1) = (percentile(ys.clone(), 0.08), percentile(ys, 0.92));
    let c = lp.scale((x0 + x1) * 0.5) + fp.scale((y0 + y1) * 0.5);
    let (hw, hd) = ((x1 - x0) * 0.5, (y1 - y0) * 0.5);
    let r = hd * 0.95;
    let off = (hw * 0.95 - r).max(0.0);
    let sides: Vec<(f64, &str)> = if off > 0.15 * r {
        vec![(-1.0, "R"), (1.0, "L")]
    } else {
        vec![(0.0, "")]
    };
    for (sgn, tag) in sides {
        let shift = c + lp.scale(sgn * off);
        let nm = if tag.is_empty() {
            name.trim_start_matches("DEF-").to_string()
        } else {
            format!("{}.{}", name.trim_start_matches("DEF-"), tag)
        };
        parts.push(Capsule {
            name: nm,
            bone,
            a: to_local(bone, a + shift),
            b: to_local(bone, b + shift),
            r,
        });
    }
}

fn hand_dir(rig: &Rig, map: &Mapping, side: &str) -> Option<Vec3> {
    let h = rig.head(map.node_of(&format!("DEF-hand.{side}"))?);
    for c in [
        "DEF-f_middle.01",
        "DEF-f_index.01",
        "DEF-f_ring.01",
        "DEF-palm.02",
    ] {
        if let Some(n) = map.node_of(&format!("{c}.{side}")) {
            let d = rig.head(n) - h;
            if d.length() > 1e-6 {
                return Some(d.normalized());
            }
        }
    }
    // Unmapped hand children (finger stubs): their mean direction.
    let hn = map.node_of(&format!("DEF-hand.{side}"))?;
    let mut sum = Vec3::ZERO;
    for &c in &rig.children[hn] {
        let d = rig.head(c) - h;
        if d.length() > 1e-6 && rig.names[c] != format!("Socket_Hand_{side}") {
            sum = sum + d.normalized();
        }
    }
    (sum.length() > 1e-6).then(|| sum.normalized())
}

/// Add `Socket_Hand_L/R` under each hand when missing: a plain node (not a
/// joint) where a game attaches a held weapon, the HLL convention from
/// the hand-built Andras rig: 45% along the hand, local +Y (the blade)
/// pointing at the character's front in the rest pose, local +Z down.
/// Returns the sockets added.
pub fn insert_sockets(doc: &mut Document) -> Result<Vec<String>, String> {
    let rig = Rig::from_doc(doc)?;
    let map = humanoid::map_rig(&rig);
    let verts = skinned_rest(doc, &rig)?;
    let fwd = crate::animate::forward(&rig, &map);
    let up = Vec3::new(0.0, 1.0, 0.0);
    let left = up.cross(fwd).normalized();
    let mut added = Vec::new();
    for side in ["L", "R"] {
        let name = format!("Socket_Hand_{side}");
        if rig.find(&name).is_some() {
            continue;
        }
        let (Some(hand), Some(fore)) = (
            map.node_of(&format!("DEF-hand.{side}")),
            map.node_of(&format!("DEF-forearm.{side}")),
        ) else {
            continue;
        };
        let w = rig.head(hand);
        let hdir = hand_dir(&rig, &map, side).unwrap_or((w - rig.head(fore)).normalized());
        let owner = owners(&rig, &map);
        let pts: Vec<f64> = verts
            .iter()
            .filter(|v| !v.inf.is_empty() && owner[dominant(v).min(owner.len() - 1)] == hand)
            .map(|v| (v.p - w).dot(hdir))
            .collect();
        let hlen = if pts.len() >= 4 {
            percentile(pts, 0.95)
        } else {
            (w - rig.head(fore)).length() * 0.35
        };
        let pos = w + hdir.scale(0.45 * hlen.max(1e-4));
        // Blade: across the palm from the little finger's knuckle to the
        // index finger's (the axis a fist closes around) when the hand has
        // fingers; else the character's front (the hand-built Andras rule).
        let knuckles = (
            map.node_of(&format!("DEF-f_index.01.{side}")),
            map.node_of(&format!("DEF-f_pinky.01.{side}")),
        );
        let blade = match knuckles {
            (Some(i), Some(p)) if (rig.head(i) - rig.head(p)).length() > 1e-6 => {
                (rig.head(i) - rig.head(p)).normalized()
            }
            _ => fwd,
        };
        // Socket frame: +Y blade, +X across (blade x hand direction), +Z = X x Y.
        let mut x = blade.cross(hdir);
        if x.length() < 1e-6 {
            x = left;
        }
        let x = x.normalized();
        let m = if blade.dot(fwd) > 0.999 {
            crate::math::Mat3::from_cols(left, fwd, up.scale(-1.0))
        } else {
            crate::math::Mat3::from_cols(x, blade, x.cross(blade))
        };
        let world_rot = Quat::from_mat3(m);
        let hw = rig.rest_world[hand];
        let inv = hw.inverse().ok_or("hand transform is singular")?;
        let local_t = inv.apply(pos);
        let local_r = canonical_quat(hw.rotation().conj().mul(world_rot));
        let mut node = Json::obj(vec![("name", Json::str(&name))]);
        let (_, _, hs) = hw.decompose();
        let inv_s = Vec3::new(1.0 / hs.x, 1.0 / hs.y, 1.0 / hs.z);
        crate::glb::set_node_trs(&mut node, local_t, local_r, inv_s);
        let nodes = doc.array_mut("nodes");
        nodes.push(node);
        let sn = nodes.len() - 1;
        let mut kids = nodes[hand]
            .get("children")
            .and_then(Json::as_arr)
            .map(|k| k.to_vec())
            .unwrap_or_default();
        kids.push(Json::num(sn as f64));
        nodes[hand].set("children", Json::Arr(kids));
        added.push(name);
    }
    Ok(added)
}

/// Socket node under the hand (`Socket_Hand_<side>`): origin and blade
/// direction (its local +Y) at rest.
pub fn socket_frame(doc: &Document, rig: &Rig, hand: usize, side: &str) -> Option<(Vec3, Vec3)> {
    let _ = doc;
    let s = rig.children[hand]
        .iter()
        .copied()
        .find(|&c| rig.names[c] == format!("Socket_Hand_{side}"))?;
    let m = rig.rest_world[s];
    Some((m.t, m.apply_linear(Vec3::new(0.0, 1.0, 0.0)).normalized()))
}

/// Penetration of `part` into `body` beyond `slack`: (depth, normal from
/// body to part, param along part).
fn depth(
    part: &Capsule,
    body: &Capsule,
    inv_rest: &[Affine],
    world: &[Affine],
    slack: f64,
) -> (f64, Vec3, f64) {
    let (p1, q1) = part.posed(inv_rest, world);
    let (p2, q2) = body.posed(inv_rest, world);
    let (s, t, d) = seg_seg(p1, q1, p2, q2);
    let c1 = p1 + (q1 - p1).scale(s);
    let c2 = p2 + (q2 - p2).scale(t);
    let mut n = c1 - c2;
    if n.length() < 1e-9 {
        // Axes cross: push perpendicular to the body axis.
        let ax = (q2 - p2).normalized();
        let side = (q1 - p1).cross(ax);
        n = if side.length() > 1e-9 {
            side
        } else {
            Vec3::new(0.0, 0.0, 1.0)
        };
    }
    (part.r + body.r - d - slack, n.normalized(), s)
}

/// One contact found in a frame.
#[derive(Clone, Copy, Debug)]
struct Hit {
    depth: f64,
    normal: Vec3,
    /// How much of a wrist move reaches the contact point (1 for the hand
    /// and weapon, the param along the forearm for the forearm).
    reach: f64,
    part: usize,
    body: usize,
    /// Contact point on the part (world).
    point: Vec3,
}

fn hits(body: &Body, ai: usize, world: &[Affine], weapon: bool, margin: f64) -> Vec<Hit> {
    let arm = &body.arms[ai];
    let nb = body.static_parts.len();
    let mut out = Vec::new();
    let posed = body.posed_surface(world);
    let nsb = body.surf_bones.len();
    for (pi, part) in arm.parts(weapon).iter().enumerate() {
        let reach_of = |at: f64| {
            if pi == 0 {
                (0.2 + 0.8 * at).max(0.35)
            } else {
                1.0
            }
        };
        let (pa, pb) = part.posed(&body.inv_rest, world);
        // Deepest surface point per body bone.
        let mut best: Vec<Option<(f64, Vec3, usize)>> = vec![None; nsb];
        for (k, d, n) in surface_hits(part, &posed, &body.inv_rest, world, margin) {
            let slot = body
                .surf_bones
                .iter()
                .position(|&b| b == body.surface[k].0)
                .unwrap();
            let sl = body.surf_slack[(ai * 3 + pi) * nsb + slot];
            // Pairs that touch at rest keep their rest depth and get no
            // extra clearance (the margin would push them off every frame).
            let d = if sl > 0.0 { d - margin - sl } else { d };
            if d > 0.0 && best[slot].is_none_or(|x| d > x.0) {
                best[slot] = Some((d, n, k));
            }
        }
        for (slot, b) in best.into_iter().enumerate() {
            if let Some((d, n, k)) = b {
                let p = posed[k].0;
                let ab = pb - pa;
                let at = ((p - pa).dot(ab) / ab.dot(ab).max(1e-12)).clamp(0.0, 1.0);
                out.push(Hit {
                    depth: d,
                    normal: n,
                    reach: reach_of(at),
                    part: pi,
                    body: nb + slot,
                    point: pa + ab.scale(at),
                });
            }
        }
    }
    for (pi, part) in arm.parts(weapon).iter().enumerate() {
        for (bi, s) in body.static_parts.iter().enumerate() {
            let slack = body.slack[(ai * 3 + pi) * nb + bi];
            let (d, n, at) = depth(
                part,
                s,
                &body.inv_rest,
                world,
                if slack > 0.0 { slack } else { -margin },
            );
            if d > 0.0 {
                let reach = if pi == 0 {
                    (0.2 + 0.8 * at).max(0.35)
                } else {
                    1.0
                };
                let (pa, pb) = part.posed(&body.inv_rest, world);
                out.push(Hit {
                    depth: d,
                    normal: n,
                    reach,
                    part: pi,
                    body: bi,
                    point: pa + (pb - pa).scale(at),
                });
            }
        }
    }
    out
}

/// Two-bone IK on one arm: move the wrist to `target`, keep the shoulder,
/// the elbow's bend plane and the hand's world rotation.
pub fn solve_arm(
    rig: &Rig,
    locals: &mut [Option<Trs>],
    upper: usize,
    fore: usize,
    hand: usize,
    target: Vec3,
) {
    solve_arm_turn(rig, locals, upper, fore, hand, target, Quat::IDENTITY);
}

/// [`solve_arm`] with the hand's world rotation turned by `turn` (a held
/// blade swung clear of the body about the wrist).
#[allow(clippy::too_many_arguments)]
pub fn solve_arm_turn(
    rig: &Rig,
    locals: &mut [Option<Trs>],
    upper: usize,
    fore: usize,
    hand: usize,
    target: Vec3,
    turn: Quat,
) {
    let world = rig.world(locals);
    let (s, e, w) = (world[upper].t, world[fore].t, world[hand].t);
    let a = (e - s).length();
    let b = (w - e).length();
    if a < 1e-9 || b < 1e-9 {
        return;
    }
    let mut to = target - s;
    let reach = to
        .length()
        .clamp((a - b).abs() + 1e-6, a + b - 1e-6 * (a + b));
    if to.length() < 1e-9 {
        return;
    }
    to = to.normalized();
    let t = s + to.scale(reach);
    // Bend plane: the current elbow's offset from the shoulder-target line.
    let ce = e - s;
    let mut pole = ce - to.scale(ce.dot(to));
    if pole.length() < 1e-6 * a {
        let cw = (w - e).normalized();
        pole = cw.scale(-1.0) - to.scale(-cw.dot(to));
        if pole.length() < 1e-9 {
            return;
        }
    }
    let pole = pole.normalized();
    let cos_a = ((a * a + reach * reach - b * b) / (2.0 * a * reach)).clamp(-1.0, 1.0);
    let ang = crate::detmath::acos(cos_a);
    let (sa, ca) = crate::detmath::sin_cos(ang);
    let e2 = s + to.scale(a * ca) + pole.scale(a * sa);
    let ru_old = world[upper].rotation();
    let rf_old = world[fore].rotation();
    let rh_old = turn.mul(world[hand].rotation());
    let q1 = shortest_arc((e - s).normalized(), (e2 - s).normalized());
    let ru_new = q1.mul(ru_old);
    let fdir = q1.rotate_vec((w - e).normalized());
    let q2 = shortest_arc(fdir, (t - e2).normalized());
    let rf_new = q2.mul(q1).mul(rf_old);
    let parent_rot = rig.parent[upper]
        .map(|p| world[p].rotation())
        .unwrap_or(Quat::IDENTITY);
    let mut lu = locals[upper].unwrap_or(rig.rest[upper]);
    lu.r = canonical_quat(parent_rot.conj().mul(ru_new));
    locals[upper] = Some(lu);
    // The forearm's parent is the upper arm (anatomical chain); fall back
    // to whatever its parent is.
    let fore_parent = rig.parent[fore];
    let fp_rot = if fore_parent == Some(upper) {
        ru_new
    } else {
        fore_parent
            .map(|p| world[p].rotation())
            .unwrap_or(Quat::IDENTITY)
    };
    let mut lf = locals[fore].unwrap_or(rig.rest[fore]);
    lf.r = canonical_quat(fp_rot.conj().mul(rf_new));
    locals[fore] = Some(lf);
    let hand_parent = rig.parent[hand];
    let hp_rot = if hand_parent == Some(fore) {
        rf_new
    } else {
        hand_parent
            .map(|p| world[p].rotation())
            .unwrap_or(Quat::IDENTITY)
    };
    let mut lh = locals[hand].unwrap_or(rig.rest[hand]);
    lh.r = canonical_quat(hp_rot.conj().mul(rh_old));
    locals[hand] = Some(lh);
}

/// Per clip and arm: what the pass found and changed.
#[derive(Clone, Debug, Default)]
pub struct ArmReport {
    pub side: String,
    pub frames_before: usize,
    pub frames_after: usize,
    pub max_depth_before: f64,
    pub max_depth_after: f64,
    pub frames_changed: usize,
    pub max_shift: f64,
    /// Largest blade turn about the wrist (radians).
    pub max_turn: f64,
    /// Exact mesh check: frames with an arm vertex or blade point inside
    /// the body (beyond rest), and the deepest, before and after.
    pub mesh_frames_before: usize,
    pub mesh_frames_after: usize,
    pub mesh_depth_before: f64,
    pub mesh_depth_after: f64,
    /// Clip time of the deepest contact before the fix (for review).
    pub worst_s: f64,
    /// Proxies that were hit, e.g. "hand.R>spine.002.L".
    pub contacts: Vec<String>,
}

#[derive(Clone, Debug, Default)]
pub struct ClipContact {
    pub weapon: bool,
    pub arms: Vec<ArmReport>,
    pub height: f64,
}

impl ClipContact {
    pub fn to_json(&self) -> Json {
        let cm = |m: f64| Json::num(((m * 1000.0).round()) / 10.0);
        Json::obj(vec![
            ("weapon", Json::Bool(self.weapon)),
            (
                "arms",
                Json::Arr(
                    self.arms
                        .iter()
                        .map(|a| {
                            Json::obj(vec![
                                ("side", Json::str(&a.side)),
                                (
                                    "frames_in_contact_before",
                                    Json::num(a.frames_before as f64),
                                ),
                                ("frames_in_contact_after", Json::num(a.frames_after as f64)),
                                ("max_depth_cm_before", cm(a.max_depth_before)),
                                ("max_depth_cm_after", cm(a.max_depth_after)),
                                (
                                    "mesh_frames_inside_before",
                                    Json::num(a.mesh_frames_before as f64),
                                ),
                                (
                                    "mesh_frames_inside_after",
                                    Json::num(a.mesh_frames_after as f64),
                                ),
                                ("mesh_depth_cm_before", cm(a.mesh_depth_before)),
                                ("mesh_depth_cm_after", cm(a.mesh_depth_after)),
                                ("worst_s", Json::num((a.worst_s * 1000.0).round() / 1000.0)),
                                ("frames_changed", Json::num(a.frames_changed as f64)),
                                ("max_wrist_shift_cm", cm(a.max_shift)),
                                (
                                    "max_blade_turn_deg",
                                    Json::num((a.max_turn.to_degrees() * 10.0).round() / 10.0),
                                ),
                                (
                                    "contacts",
                                    Json::Arr(a.contacts.iter().map(|c| Json::str(c)).collect()),
                                ),
                            ])
                        })
                        .collect(),
                ),
            ),
        ])
    }

    /// Changed anything at all.
    pub fn changed(&self) -> bool {
        self.arms.iter().any(|a| a.frames_changed > 0)
    }
}

fn smoothstep(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

fn measure(
    body: &Body,
    ai: usize,
    rig: &Rig,
    frames: &[Vec<Option<Trs>>],
    weapon: bool,
) -> (usize, f64, Vec<String>) {
    let mut count = 0;
    let mut worst: f64 = 0.0;
    let mut names: Vec<String> = Vec::new();
    let arm = &body.arms[ai];
    for f in frames {
        let world = rig.world(f);
        let hs = hits(body, ai, &world, weapon, 0.0);
        if !hs.is_empty() {
            count += 1;
        }
        for h in hs {
            worst = worst.max(h.depth);
            let target = match body.static_parts.get(h.body) {
                Some(c) => c.name.clone(),
                None => format!(
                    "skin:{}",
                    rig.names[body.surf_bones[h.body - body.static_parts.len()]]
                        .trim_start_matches("DEF-")
                ),
            };
            let n = format!("{}>{}", arm.parts(weapon)[h.part].name, target);
            if !names.contains(&n) {
                names.push(n);
            }
        }
    }
    names.sort();
    (count, worst, names)
}

/// Debug: posed capsules of every frame, as JSON
/// `[[{"name", "a", "b", "r"}...] per frame]` in glTF world space.
pub fn dump_posed(body: &Body, rig: &Rig, frames: &[Vec<Option<Trs>>], weapon: bool) -> Json {
    let v = |p: Vec3| {
        Json::Arr(vec![
            Json::num(r4(p.x)),
            Json::num(r4(p.y)),
            Json::num(r4(p.z)),
        ])
    };
    Json::Arr(
        frames
            .iter()
            .map(|f| {
                let world = rig.world(f);
                let mut caps: Vec<&Capsule> = body.static_parts.iter().collect();
                for a in &body.arms {
                    caps.extend(a.parts(weapon));
                }
                Json::Arr(
                    caps.iter()
                        .map(|c| {
                            let (a, b) = c.posed(&body.inv_rest, &world);
                            Json::obj(vec![
                                ("name", Json::str(&c.name)),
                                ("a", v(a)),
                                ("b", v(b)),
                                ("r", Json::num(r4(c.r))),
                            ])
                        })
                        .collect(),
                )
            })
            .collect(),
    )
}

/// Exact (mesh) contact over a clip for one arm: frames with an arm
/// vertex or blade point inside the body beyond rest, and the deepest.
fn mesh_measure(
    body: &Body,
    ai: usize,
    rig: &Rig,
    frames: &[Vec<Option<Trs>>],
    weapon: bool,
) -> (Vec<bool>, f64, usize) {
    let arm = &body.arms[ai];
    let w = if weapon { arm.weapon.as_ref() } else { None };
    let mut at = 0;
    let slack = body.mesh_slack.get(ai).copied().unwrap_or(0.0);
    let mut inside = Vec::with_capacity(frames.len());
    let mut worst: f64 = 0.0;
    for f in frames {
        let world = rig.world(f);
        let (d, n) = body.mesh_depth(rig, &world, ai, w);
        let deep = n > 0 && d > slack + 0.002 * body.height;
        if deep && d - slack > worst {
            worst = d - slack;
            at = inside.len();
        }
        inside.push(deep);
    }
    (inside, worst, at)
}

/// One fix attempt on one arm with the arm proxies grown by `grow`
/// (radius multiplier). Returns the new frames and wrist offsets.
#[allow(clippy::too_many_arguments)]
fn fix_arm(
    body: &Body,
    rig: &Rig,
    ai: usize,
    original: &[Vec<Option<Trs>>],
    times: &[f64],
    use_weapon: bool,
    margin: f64,
    ramp: f64,
    grow: f64,
) -> (Vec<Vec<Option<Trs>>>, Vec<Vec3>, Vec<Vec3>) {
    let mut grown = body.clone();
    {
        let arm = &mut grown.arms[ai];
        arm.forearm.r *= grow;
        arm.palm.r *= grow;
        if let Some(w) = arm.weapon.as_mut() {
            w.r *= grow;
        }
    }
    let body = &grown;
    let arm = &body.arms[ai];
    let n = original.len();
    let mut frames: Vec<Vec<Option<Trs>>> = original.to_vec();
    let wrist0: Vec<Vec3> = original.iter().map(|f| rig.world(f)[arm.hand].t).collect();
    let mut offset = vec![Vec3::ZERO; n];
    let mut turn_off = vec![Vec3::ZERO; n];
    // Limits that keep the swing's intent: the wrist moves at most 12% of
    // the character's height, a held blade turns at most 40 degrees.
    let max_shift = 0.12 * body.height;
    let max_turn = 40f64.to_radians();
    for _round in 0..3 {
        // 1. Per frame: the extra wrist offset that clears this frame.
        let mut need = vec![Vec3::ZERO; n];
        let mut any = false;
        let mut need_turn = vec![Vec3::ZERO; n];
        for f in 0..n {
            let mut locals = frames[f].clone();
            let mut extra = Vec3::ZERO;
            let mut turn = Vec3::ZERO; // rotation vector (axis * angle)
            let base = wrist0[f] + offset[f];
            for _ in 0..12 {
                let world = rig.world(&locals);
                let hs = hits(body, ai, &world, use_weapon, margin);
                let Some(h) = hs.iter().max_by(|a, b| {
                    (a.depth / a.reach)
                        .partial_cmp(&(b.depth / b.reach))
                        .unwrap_or(std::cmp::Ordering::Equal)
                }) else {
                    break;
                };
                let wrist = world[arm.hand].t;
                let lever = h.point - wrist;
                if h.part == 2 && lever.length() > 1e-6 && turn.length() < max_turn {
                    // Blade: swing it clear about the wrist.
                    let axis = lever.cross(h.normal);
                    if axis.length() > 1e-9 {
                        let ang = ((h.depth + 0.5 * margin) / lever.length()).min(0.35);
                        turn = turn + axis.normalized().scale(ang);
                        if turn.length() > max_turn {
                            turn = turn.scale(max_turn / turn.length());
                        }
                    }
                } else {
                    extra = extra + h.normal.scale((h.depth + 0.5 * margin) / h.reach);
                    if extra.length() > max_shift {
                        extra = extra.scale(max_shift / extra.length());
                    }
                }
                locals = frames[f].clone();
                solve_arm_turn(
                    rig,
                    &mut locals,
                    arm.upper,
                    arm.fore,
                    arm.hand,
                    base + extra,
                    rotvec_quat(turn),
                );
            }
            if extra.length() > 0.0 || turn.length() > 0.0 {
                any = true;
            }
            need[f] = extra;
            need_turn[f] = turn;
        }
        if !any {
            break;
        }
        // 2. Smooth over time: max envelope with a smoothstep falloff.
        let smooth = envelope(&need, times, ramp);
        let smooth_turn = envelope(&need_turn, times, ramp);
        // 3. Apply.
        for f in 0..n {
            if smooth[f].length() <= 0.0 && smooth_turn[f].length() <= 0.0 {
                continue;
            }
            offset[f] = offset[f] + smooth[f];
            if offset[f].length() > max_shift {
                offset[f] = offset[f].scale(max_shift / offset[f].length());
            }
            turn_off[f] = turn_off[f] + smooth_turn[f];
            if turn_off[f].length() > max_turn {
                turn_off[f] = turn_off[f].scale(max_turn / turn_off[f].length());
            }
            let mut locals = original[f].clone();
            solve_arm_turn(
                rig,
                &mut locals,
                arm.upper,
                arm.fore,
                arm.hand,
                wrist0[f] + offset[f],
                rotvec_quat(turn_off[f]),
            );
            frames[f] = locals;
        }
    }
    (frames, offset, turn_off)
}

/// Rotation vector (axis * angle) to quaternion.
fn rotvec_quat(v: Vec3) -> Quat {
    let a = v.length();
    if a < 1e-12 {
        Quat::IDENTITY
    } else {
        Quat::from_axis_angle(v.scale(1.0 / a), a)
    }
}

/// Max envelope with a smoothstep falloff of `ramp` seconds: every frame
/// gets at least the falloff of any nearby frame's need, so a correction
/// eases in before a contact and out after it. Direction is the
/// falloff-weighted mean of the needs.
fn envelope(need: &[Vec3], times: &[f64], ramp: f64) -> Vec<Vec3> {
    let n = need.len();
    let mut out = vec![Vec3::ZERO; n];
    for f in 0..n {
        let mut mag: f64 = 0.0;
        let mut dir = Vec3::ZERO;
        for g in 0..n {
            let len = need[g].length();
            if len <= 0.0 {
                continue;
            }
            let k = smoothstep(1.0 - (times[f] - times[g]).abs() / ramp.max(1e-6));
            if k <= 0.0 {
                continue;
            }
            mag = mag.max(len * k);
            dir = dir + need[g].scale(k);
        }
        if mag > 0.0 && dir.length() > 1e-12 {
            out[f] = dir.normalized().scale(mag);
        }
    }
    out
}

/// One fix attempt: frames, wrist offsets, frames still inside the mesh,
/// deepest residual, blade turns.
type Attempt = (Vec<Vec<Option<Trs>>>, Vec<Vec3>, usize, f64, Vec<Vec3>);

/// Run the pass on one clip's frames (target locals per frame, sampled
/// at `times`). Edits `frames` in place. Proxies drive the fix; the exact
/// mesh check verifies it, and an arm that still cuts the mesh is
/// re-solved from the original with fatter proxies (x1.4, x1.9).
pub fn fix_clip(
    body: &Body,
    rig: &Rig,
    frames: &mut [Vec<Option<Trs>>],
    times: &[f64],
    clip: &str,
    opts: &Options,
) -> ClipContact {
    let weapon = opts.weapon_in(clip);
    let margin = opts.margin * body.height;
    let mut report = ClipContact {
        weapon,
        arms: Vec::new(),
        height: body.height,
    };
    for ai in 0..body.arms.len() {
        let arm = &body.arms[ai];
        let use_weapon = weapon && arm.weapon.is_some();
        let (fb, db, contacts) = measure(body, ai, rig, frames, use_weapon);
        let (mb, mdb, worst_at) = mesh_measure(body, ai, rig, frames, use_weapon);
        let mut rep = ArmReport {
            side: arm.side.clone(),
            frames_before: fb,
            max_depth_before: db,
            mesh_frames_before: mb.iter().filter(|&&x| x).count(),
            mesh_depth_before: mdb,
            worst_s: times.get(worst_at).copied().unwrap_or(0.0),
            contacts,
            ..Default::default()
        };
        if fb == 0 && rep.mesh_frames_before == 0 {
            report.arms.push(rep);
            continue;
        }
        let original: Vec<Vec<Option<Trs>>> = frames.to_vec();
        let mut best: Option<Attempt> = None;
        for grow in [1.0, 1.15, 1.3, 1.5, 1.75, 2.0, 2.4, 3.0] {
            let (fixed, offset, turns) = fix_arm(
                body, rig, ai, &original, times, use_weapon, margin, opts.ramp, grow,
            );
            let (ma, mda, _) = mesh_measure(body, ai, rig, &fixed, use_weapon);
            let left = ma.iter().filter(|&&x| x).count();
            let better = best
                .as_ref()
                .is_none_or(|b| left < b.2 || (left == b.2 && mda < b.3));
            if better {
                best = Some((fixed, offset, left, mda, turns));
            }
            if left == 0 {
                break;
            }
        }
        let (fixed, offset, left, mda, turns) = best.unwrap();
        frames.clone_from_slice(&fixed);
        let (fa, da, _) = measure(body, ai, rig, frames, use_weapon);
        rep.frames_after = fa;
        rep.max_depth_after = da;
        rep.mesh_frames_after = left;
        rep.mesh_depth_after = mda;
        rep.frames_changed = offset
            .iter()
            .zip(&turns)
            .filter(|(o, t)| o.length() > 0.0 || t.length() > 0.0)
            .count();
        rep.max_turn = turns.iter().map(|t| t.length()).fold(0.0, f64::max);
        rep.max_shift = offset.iter().map(|o| o.length()).fold(0.0, f64::max);
        report.arms.push(rep);
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_distance_cases() {
        let (s, t, d) = seg_seg(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.5, 1.0, -1.0),
            Vec3::new(0.5, 1.0, 1.0),
        );
        assert!((s - 0.5).abs() < 1e-12 && (t - 0.5).abs() < 1e-12 && (d - 1.0).abs() < 1e-12);
        // Parallel segments.
        let (_, _, d) = seg_seg(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(1.0, 2.0, 0.0),
        );
        assert!((d - 2.0).abs() < 1e-12);
    }

    #[test]
    fn winding_of_a_cube() {
        let mut pos = Vec::new();
        for z in [-1.0, 1.0] {
            for (x, y) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                pos.push(Vec3::new(x, y, z));
            }
        }
        let quads = [
            [0, 3, 2, 1],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [3, 0, 4, 7],
        ];
        let mut tris = Vec::new();
        for q in quads {
            tris.push([q[0], q[1], q[2]]);
            tris.push([q[0], q[2], q[3]]);
        }
        let inside = winding(&pos, &tris, Vec3::new(0.2, -0.3, 0.1));
        let outside = winding(&pos, &tris, Vec3::new(2.5, 0.0, 0.0));
        assert!((inside.abs() - 1.0).abs() < 1e-9, "{inside}");
        assert!(outside.abs() < 1e-9, "{outside}");
    }

    #[test]
    fn smoothstep_ends() {
        assert_eq!(smoothstep(0.0), 0.0);
        assert_eq!(smoothstep(1.0), 1.0);
        assert!((smoothstep(0.5) - 0.5).abs() < 1e-12);
    }
}
