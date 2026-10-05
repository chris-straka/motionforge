//! Retarget glTF animation clips onto a rigged GLB.
//!
//! Both rigs are mapped onto the canonical humanoid names
//! (`humanoid.rs`), so any rig the mapper understands can be a source.
//! Per frame and per shared bone, the source bone's world-space rotation
//! change from its rest pose is applied to the target bone's world rest
//! rotation (after a yaw fix when the two rigs face different ways) and
//! converted back to a local rotation under the target's posed parent.
//! This is the same transfer as `retarget.rs` (`d_t = C d C^-1`) but
//! works on glTF joints with arbitrary rest orientations (no zero-roll
//! limit) and stays exact when the source has extra in-between bones.
//! Root travel is scaled by the hip-height ratio. Unshared target bones
//! keep their rest pose relative to their parent, except the twist/helper
//! bones, which are keyed with their share of their driver's rotation
//! (`helpers.rs`).
//!
//! Rest alignment: before the motion is applied, each target bone is
//! turned (shortest arc) so its rest direction matches the source's, so
//! "same pose" means "same bone directions", not "same change from each
//! rig's own rest". Without it a clip made on an A-pose rig lifts a
//! T-pose rig's arms by the difference (45 deg) in every frame.

use crate::glb::{canonical_quat, Affine, Document};
use crate::humanoid::{self, Mapping};
use crate::json::Json;
use crate::math::{shortest_arc, Quat, Vec3};
use crate::rig::{Animation, Rig, Trs};

#[derive(Clone, Debug)]
pub struct ClipReport {
    pub name: String,
    pub source: String,
    pub frames: usize,
    pub duration: f64,
    pub bones_driven: usize,
    /// Largest world-rotation mismatch between source and target bone
    /// motion over the clip (a self-check; ~0 unless scales shear).
    pub max_error_deg: f64,
    pub root_travel_m: f64,
}

impl ClipReport {
    pub fn to_json(&self) -> Json {
        Json::obj(vec![
            ("name", Json::str(&self.name)),
            ("source", Json::str(&self.source)),
            ("frames", Json::num(self.frames as f64)),
            ("duration_s", Json::num(round(self.duration, 4))),
            ("bones_driven", Json::num(self.bones_driven as f64)),
            ("max_error_deg", Json::num(round(self.max_error_deg, 4))),
            ("root_travel_m", Json::num(round(self.root_travel_m, 4))),
        ])
    }
}

pub fn round(v: f64, places: i32) -> f64 {
    let k = 10f64.powi(places);
    (v * k).round() / k
}

/// A rig plus its canonical mapping and the facts retargeting needs.
pub struct Mapped {
    pub rig: Rig,
    pub map: Mapping,
    pub forward: Vec3,
    pub hip_height: f64,
}

impl Mapped {
    pub fn new(doc: &Document) -> Result<Mapped, String> {
        let rig = Rig::from_doc(doc)?;
        let map = humanoid::map_rig(&rig);
        if map.node_of("DEF-spine").is_none() {
            return Err("no hips bone found".into());
        }
        let forward = forward(&rig, &map);
        let hips = map.node_of("DEF-spine").unwrap();
        let ground = ["DEF-foot.L", "DEF-foot.R", "DEF-toe.L", "DEF-toe.R"]
            .iter()
            .filter_map(|n| map.node_of(n))
            .map(|n| rig.head(n).y)
            .fold(f64::INFINITY, f64::min);
        let hip_height = if ground.is_finite() {
            rig.head(hips).y - ground
        } else {
            rig.head(hips).y
        };
        Ok(Mapped {
            rig,
            map,
            forward,
            hip_height,
        })
    }
}

/// Horizontal facing from heel to toe (foot -> toe heads), glTF +Y up.
/// Falls back to +Z (glTF's front) when feet give no direction.
pub fn forward(rig: &Rig, map: &Mapping) -> Vec3 {
    let mut sum = Vec3::ZERO;
    for side in ["L", "R"] {
        if let (Some(f), Some(t)) = (
            map.node_of(&format!("DEF-foot.{}", side)),
            map.node_of(&format!("DEF-toe.{}", side)),
        ) {
            let d = rig.head(t) - rig.head(f);
            sum = sum + Vec3::new(d.x, 0.0, d.z);
        }
    }
    if sum.length() < 1e-6 {
        // No toes: use the facing implied by the left/right hips.
        if let (Some(l), Some(r)) = (map.node_of("DEF-thigh.L"), map.node_of("DEF-thigh.R")) {
            let d = rig.head(l) - rig.head(r);
            // Character left = up x forward, so forward = left x up.
            let out = Vec3::new(d.x, 0.0, d.z).cross(Vec3::new(0.0, 1.0, 0.0));
            if out.length() > 1e-6 {
                return out.normalized();
            }
        }
        return Vec3::new(0.0, 0.0, 1.0);
    }
    sum.normalized()
}

/// Yaw rotation (about +Y) taking `from` onto `to` (both horizontal).
pub fn yaw_between(from: Vec3, to: Vec3) -> Quat {
    let a = Vec3::new(from.x, 0.0, from.z).normalized();
    let b = Vec3::new(to.x, 0.0, to.z).normalized();
    if a.dot(b) <= -1.0 + 1e-9 {
        return Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), std::f64::consts::PI);
    }
    shortest_arc(a, b)
}

/// Rotation taking the target's rest direction of `name` onto the
/// source's (after yaw), measured toward the first child both rigs have.
/// Identity when no common child exists.
fn rest_alignment(src: &Mapped, tgt: &Mapped, name: &str, yaw: Quat) -> Quat {
    let (Some(ts), Some(ss)) = (tgt.map.node_of(name), src.map.node_of(name)) else {
        return Quat::IDENTITY;
    };
    for child in humanoid::main_children(name) {
        let (Some(tc), Some(sc)) = (tgt.map.node_of(&child), src.map.node_of(&child)) else {
            continue;
        };
        let dt = tgt.rig.head(tc) - tgt.rig.head(ts);
        let ds = yaw.rotate_vec(src.rig.head(sc) - src.rig.head(ss));
        if dt.length() < 1e-9 || ds.length() < 1e-9 {
            return Quat::IDENTITY;
        }
        return shortest_arc(dt.normalized(), ds.normalized());
    }
    Quat::IDENTITY
}

/// Local transforms of every target node for one source frame.
pub struct Transfer<'a> {
    pub src: &'a Mapped,
    pub tgt: &'a Mapped,
    pub yaw: Quat,
    pub scale: f64,
    /// `(target node, source node)` per shared canonical bone.
    pub shared: Vec<(usize, usize)>,
    /// Per shared bone: world rotation taking the target's rest bone
    /// direction onto the (yawed) source's, so a clip made on an A-pose
    /// rig plays as an A-pose-relative motion on a T-pose rig too.
    pub align: Vec<Quat>,
}

impl<'a> Transfer<'a> {
    pub fn new(src: &'a Mapped, tgt: &'a Mapped) -> Transfer<'a> {
        let mut shared = Vec::new();
        for (tnode, name) in &tgt.map.pairs {
            // Helpers follow their driver by rule (`helpers.rs`), never
            // the source's own helper.
            if crate::helpers::is_helper(name) {
                continue;
            }
            if let Some(snode) = src.map.node_of(name) {
                shared.push((*tnode, snode));
            }
        }
        let scale = if src.hip_height > 1e-6 {
            tgt.hip_height / src.hip_height
        } else {
            1.0
        };
        let yaw = yaw_between(src.forward, tgt.forward);
        let align = shared
            .iter()
            .map(|(tnode, _)| {
                rest_alignment(src, tgt, tgt.map.canonical_of(*tnode).unwrap_or(""), yaw)
            })
            .collect();
        Transfer {
            src,
            tgt,
            yaw,
            scale,
            shared,
            align,
        }
    }

    /// Returns target local overrides and the max world-rotation error.
    pub fn frame(&self, src_locals: &[Option<Trs>]) -> (Vec<Option<Trs>>, f64) {
        let srig = &self.src.rig;
        let trig = &self.tgt.rig;
        let sworld = srig.world(src_locals);
        let n = trig.names.len();
        let mut locals: Vec<Option<Trs>> = vec![None; n];
        let mut world: Vec<Affine> = vec![Affine::IDENTITY; n];
        let hips_t = self.tgt.map.node_of("DEF-spine");
        let hips_s = self.src.map.node_of("DEF-spine");
        let mut max_err: f64 = 0.0;
        for &i in &trig.order {
            let parent_world = trig.parent[i].map(|p| world[p]).unwrap_or(Affine::IDENTITY);
            let mut local = trig.rest[i];
            let mut driven_delta = None;
            if let Some(k) = self.shared.iter().position(|(t, _)| *t == i) {
                let s = self.shared[k].1;
                let rs = srig.rest_world[s].rotation();
                let ps = sworld[s].rotation();
                let d = ps.mul(rs.conj());
                let d = self.yaw.mul(d).mul(self.yaw.conj()).mul(self.align[k]);
                let want = d.mul(trig.rest_world[i].rotation());
                let pr = parent_world.rotation();
                local.r = canonical_quat(pr.conj().mul(want));
                driven_delta = Some(d);
                if Some(i) == hips_t {
                    if let Some(hs) = hips_s {
                        let delta = sworld[hs].t - srig.rest_world[hs].t;
                        let moved = self.yaw.rotate_vec(delta).scale(self.scale);
                        let pos = trig.rest_world[i].t + moved;
                        if let Some(inv) = parent_world.inverse() {
                            local.t = inv.apply(pos);
                        }
                    }
                }
            }
            let is_override = driven_delta.is_some();
            world[i] = parent_world.mul(&local.affine());
            if let Some(d) = driven_delta {
                let got = world[i]
                    .rotation()
                    .mul(trig.rest_world[i].rotation().conj());
                max_err = max_err.max(got.angle_to(d));
            }
            if is_override {
                locals[i] = Some(local);
            }
        }
        (locals, max_err)
    }
}

/// Retarget every animation of every source onto `target`, replacing the
/// target's own animations. `sources` are `(label, document)`.
pub fn retarget_clips(
    target: &Document,
    sources: &[(String, Document)],
    fps: f64,
) -> Result<(Document, Vec<ClipReport>), String> {
    let tgt = Mapped::new(target)?;
    let mut out = target.clone();
    out.json.remove("animations");
    let mut reports = Vec::new();
    let mut used_names: Vec<String> = Vec::new();
    for (label, src_doc) in sources {
        let src = Mapped::new(src_doc).map_err(|e| format!("{}: {}", label, e))?;
        let transfer = Transfer::new(&src, &tgt);
        if transfer.shared.len() < 10 {
            return Err(format!(
                "{}: only {} bones shared with the target",
                label,
                transfer.shared.len()
            ));
        }
        for anim in Animation::load_all(src_doc)? {
            let frames = ((anim.duration * fps).round() as usize).max(1) + 1;
            let mut name = anim.name.clone();
            let mut k = 2;
            while used_names.contains(&name) {
                name = format!("{}-{}", anim.name, k);
                k += 1;
            }
            used_names.push(name.clone());
            let mut times = Vec::with_capacity(frames);
            // Driven nodes: the shared bones, then the twist/helpers baked
            // from their drivers (the game needs no constraint code).
            let helpers = crate::helpers::present(&tgt.rig);
            let mut nodes: Vec<usize> = transfer.shared.iter().map(|(t, _)| *t).collect();
            nodes.extend(helpers.iter().map(|(h, _)| *h));
            let mut tracks: Vec<Vec<Trs>> = vec![Vec::with_capacity(frames); nodes.len()];
            let mut max_err: f64 = 0.0;
            for f in 0..frames {
                let t = if frames > 1 {
                    anim.duration * f as f64 / (frames - 1) as f64
                } else {
                    0.0
                };
                times.push(t);
                let (mut locals, err) = transfer.frame(&anim.sample(&src.rig, t));
                max_err = max_err.max(err);
                crate::helpers::drive(&tgt.rig, &mut locals);
                for (k, node) in nodes.iter().enumerate() {
                    tracks[k].push(locals[*node].unwrap_or(tgt.rig.rest[*node]));
                }
            }
            let hips = tgt.map.node_of("DEF-spine");
            let root_travel = hips
                .and_then(|h| transfer.shared.iter().position(|(t, _)| *t == h))
                .map(|k| (tracks[k].last().unwrap().t - tracks[k][0].t).length())
                .unwrap_or(0.0);
            write_animation(&mut out, &name, &times, &nodes, &tracks, hips)?;
            reports.push(ClipReport {
                name,
                source: label.clone(),
                frames,
                duration: anim.duration,
                bones_driven: transfer.shared.len(),
                max_error_deg: max_err.to_degrees(),
                root_travel_m: root_travel,
            });
        }
    }
    Ok((out, reports))
}

/// Append one animation: rotation tracks for `nodes`, plus translation
/// for `root` (when it is among them).
pub fn write_animation(
    out: &mut Document,
    name: &str,
    times: &[f64],
    nodes: &[usize],
    tracks: &[Vec<Trs>],
    root: Option<usize>,
) -> Result<(), String> {
    let time_rows: Vec<Vec<f64>> = times.iter().map(|t| vec![*t]).collect();
    let input = out.push_float_accessor(&time_rows, "SCALAR", true);
    let interp = if times.len() == 1 { "STEP" } else { "LINEAR" };
    let mut samplers = Vec::new();
    let mut channels = Vec::new();
    for (k, &node) in nodes.iter().enumerate() {
        let mut prev: Option<Quat> = None;
        let rows: Vec<Vec<f64>> = tracks[k]
            .iter()
            .map(|trs| {
                let mut q = trs.r.normalized();
                if let Some(p) = prev {
                    if p.dot(q) < 0.0 {
                        q = q.neg();
                    }
                }
                prev = Some(q);
                vec![q.x, q.y, q.z, q.w]
            })
            .collect();
        let output = out.push_float_accessor(&rows, "VEC4", false);
        channels.push(Json::obj(vec![
            ("sampler", Json::num(samplers.len() as f64)),
            (
                "target",
                Json::obj(vec![
                    ("node", Json::num(node as f64)),
                    ("path", Json::str("rotation")),
                ]),
            ),
        ]));
        samplers.push(Json::obj(vec![
            ("input", Json::num(input as f64)),
            ("interpolation", Json::str(interp)),
            ("output", Json::num(output as f64)),
        ]));
        if Some(node) == root {
            let rows: Vec<Vec<f64>> = tracks[k]
                .iter()
                .map(|trs| vec![trs.t.x, trs.t.y, trs.t.z])
                .collect();
            let output = out.push_float_accessor(&rows, "VEC3", false);
            channels.push(Json::obj(vec![
                ("sampler", Json::num(samplers.len() as f64)),
                (
                    "target",
                    Json::obj(vec![
                        ("node", Json::num(node as f64)),
                        ("path", Json::str("translation")),
                    ]),
                ),
            ]));
            samplers.push(Json::obj(vec![
                ("input", Json::num(input as f64)),
                ("interpolation", Json::str(interp)),
                ("output", Json::num(output as f64)),
            ]));
        }
    }
    out.array_mut("animations").push(Json::obj(vec![
        ("name", Json::str(name)),
        ("channels", Json::Arr(channels)),
        ("samplers", Json::Arr(samplers)),
    ]));
    Ok(())
}
