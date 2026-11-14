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
    /// What the contact pass found and changed (when it ran).
    pub contact: Option<crate::contact::ClipContact>,
    /// What foot pinning changed (when it ran).
    pub pin: Option<crate::footpin::PinReport>,
}

impl ClipReport {
    pub fn to_json(&self) -> Json {
        let mut fields = vec![
            ("name", Json::str(&self.name)),
            ("source", Json::str(&self.source)),
            ("frames", Json::num(self.frames as f64)),
            ("duration_s", Json::num(round(self.duration, 4))),
            ("bones_driven", Json::num(self.bones_driven as f64)),
            ("max_error_deg", Json::num(round(self.max_error_deg, 4))),
            ("root_travel_m", Json::num(round(self.root_travel_m, 4))),
        ];
        if let Some(p) = &self.pin {
            fields.push((
                "pin",
                Json::obj(vec![
                    ("plants", Json::num(p.plants as f64)),
                    ("frames_changed", Json::num(p.frames_changed as f64)),
                    ("max_shift_m", Json::num(round(p.max_shift, 4))),
                ]),
            ));
        }
        if let Some(c) = &self.contact {
            fields.push(("contact", c.to_json()));
        }
        Json::obj(fields)
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
        let base = shortest_arc(dt.normalized(), ds.normalized());
        return match name.strip_prefix("DEF-hand.") {
            Some(side) => hand_roll(src, tgt, side, yaw, base, ds.normalized()).mul(base),
            None => base,
        };
    }
    // A hand whose fingers the two rigs do not share by name (a mitten
    // hand, one finger bone, sockets only): toward the middle of each
    // rig's own hand children, else as its forearm. Left unaligned, the
    // hand kept the source's rest direction on a rig that rests otherwise:
    // every clip made on a T-pose rig bent an A-pose rig's wrists by the
    // difference (70 deg on the pro bench's hero rig).
    if let Some(side) = name.strip_prefix("DEF-hand.") {
        let tip = |m: &Mapped, node: usize| -> Option<Vec3> {
            let kids: Vec<Vec3> = m.rig.children[node]
                .iter()
                .map(|&c| m.rig.head(c) - m.rig.head(node))
                .filter(|d| d.length() > 1e-9)
                .collect();
            if kids.is_empty() {
                return None;
            }
            let mean = kids
                .iter()
                .fold(Vec3::ZERO, |a, d| a + *d)
                .scale(1.0 / kids.len() as f64);
            (mean.length() > 1e-9).then(|| mean.normalized())
        };
        return match (tip(tgt, ts), tip(src, ss)) {
            (Some(dt), Some(ds)) => shortest_arc(dt, yaw.rotate_vec(ds).normalized()),
            _ => rest_alignment(src, tgt, &format!("DEF-forearm.{side}"), yaw),
        };
    }
    Quat::IDENTITY
}

/// The turn about the (aligned) hand direction `axis` that lays the
/// target's knuckle line (index to little finger roots) on the source's,
/// after `base` has aligned the hands' directions. A hand's direction
/// leaves its roll free: the palm of a rig resting with its knuckles
/// otherwise kept that difference in every frame, so a sword held in the
/// palm pointed 24 degrees off the original Mixamo clip's on the hero.
/// Identity unless both rigs share an index and a little or ring finger.
fn hand_roll(src: &Mapped, tgt: &Mapped, side: &str, yaw: Quat, base: Quat, axis: Vec3) -> Quat {
    // The same little finger on both rigs (pinky with pinky, else ring
    // with ring), so the two lines mean the same thing.
    let has = |m: &Mapped, f: &str| m.map.node_of(&format!("DEF-{f}.01.{side}")).is_some();
    let Some(little) = ["f_pinky", "f_ring"]
        .into_iter()
        .find(|f| has(src, f) && has(tgt, f))
    else {
        return Quat::IDENTITY;
    };
    let pick = |m: &Mapped| -> Option<Vec3> {
        let index = m.map.node_of(&format!("DEF-f_index.01.{side}"))?;
        let little = m.map.node_of(&format!("DEF-{little}.01.{side}"))?;
        Some(m.rig.head(index) - m.rig.head(little))
    };
    let (Some(lt), Some(ls)) = (pick(tgt), pick(src)) else {
        return Quat::IDENTITY;
    };
    let lt = base.rotate_vec(lt);
    let ls = yaw.rotate_vec(ls);
    let angle = crate::keypose::fit::signed_angle(lt, ls, axis);
    Quat::from_axis_angle(axis, angle)
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

/// Which clips to take and how to name them: `(source clip name, new
/// name)`, optionally `label:clip` to pick from one source file. Output
/// order follows the pick list.
pub type Pick = Vec<(String, String)>;

/// Parse `"Idle_Loop=idle,ual2.glb:Sword_Regular_A=attack_1"`; a bare name
/// keeps its name.
pub fn parse_pick(spec: &str) -> Result<Pick, String> {
    let mut out = Vec::new();
    for item in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let (from, to) = match item.split_once('=') {
            Some((a, b)) => (a.trim(), b.trim()),
            None => (item, item.rsplit(':').next().unwrap_or(item)),
        };
        if from.is_empty() || to.is_empty() {
            return Err(format!("bad --pick entry '{}'", item));
        }
        out.push((from.to_string(), to.to_string()));
    }
    if out.is_empty() {
        return Err("--pick is empty".into());
    }
    Ok(out)
}

/// Retarget every animation of every source onto `target`, replacing the
/// target's own animations. `sources` are `(label, document)`.
pub fn retarget_clips(
    target: &Document,
    sources: &[(String, Document)],
    fps: f64,
) -> Result<(Document, Vec<ClipReport>), String> {
    retarget_picked(target, sources, fps, None, None)
}

/// [`retarget_clips`] limited to (and renamed by) `pick`, feet pinned.
pub fn retarget_picked(
    target: &Document,
    sources: &[(String, Document)],
    fps: f64,
    pick: Option<&Pick>,
    contact: Option<&crate::contact::Options>,
) -> Result<(Document, Vec<ClipReport>), String> {
    retarget_with(target, sources, fps, pick, contact, true)
}

/// [`retarget_picked`] with foot pinning (`footpin.rs`) on or off.
pub fn retarget_with(
    target: &Document,
    sources: &[(String, Document)],
    fps: f64,
    pick: Option<&Pick>,
    contact: Option<&crate::contact::Options>,
    pin: bool,
) -> Result<(Document, Vec<ClipReport>), String> {
    let tgt = Mapped::new(target)?;
    let mut out = target.clone();
    out.json.remove("animations");
    let mut reports = Vec::new();
    let mut used_names: Vec<String> = Vec::new();
    let mapped: Vec<Mapped> = sources
        .iter()
        .map(|(label, doc)| Mapped::new(doc).map_err(|e| format!("{}: {}", label, e)))
        .collect::<Result<_, _>>()?;
    let anims: Vec<Vec<Animation>> = sources
        .iter()
        .map(|(_, doc)| Animation::load_all(doc))
        .collect::<Result<_, _>>()?;
    // Jobs: (source index, animation index, output base name).
    let mut jobs: Vec<(usize, usize, String)> = Vec::new();
    match pick {
        None => {
            for (si, list) in anims.iter().enumerate() {
                for (ai, a) in list.iter().enumerate() {
                    jobs.push((si, ai, a.name.clone()));
                }
            }
        }
        Some(pick) => {
            for (from, to) in pick {
                let (want_label, want) = match from.rsplit_once(':') {
                    Some((l, n)) => (Some(l), n),
                    None => (None, from.as_str()),
                };
                let found = sources.iter().enumerate().find_map(|(si, (label, _))| {
                    if want_label.is_some_and(|l| l != label) {
                        return None;
                    }
                    anims[si]
                        .iter()
                        .position(|a| a.name == want)
                        .map(|ai| (si, ai))
                });
                let (si, ai) = found.ok_or_else(|| format!("--pick: no clip named '{}'", from))?;
                jobs.push((si, ai, to.clone()));
            }
        }
    }
    let mut transfers: Vec<Option<Transfer>> = (0..sources.len()).map(|_| None).collect();
    for &(si, _, _) in &jobs {
        if transfers[si].is_none() {
            let t = Transfer::new(&mapped[si], &tgt);
            if t.shared.len() < 10 {
                return Err(format!(
                    "{}: only {} bones shared with the target",
                    sources[si].0,
                    t.shared.len()
                ));
            }
            transfers[si] = Some(t);
        }
    }
    // Weapon sockets follow the clips' grip: the first source with a
    // `Socket_Hand_*` sets the target's socket rotation (`sync_sockets`).
    let mut sockets_synced = Vec::new();
    for &(si, _, _) in &jobs {
        let done = sync_sockets(&mut out, &tgt, &mapped[si], transfers[si].as_ref().unwrap());
        if !done.is_empty() {
            sockets_synced = done;
            break;
        }
    }
    let _ = &sockets_synced;
    let body = match contact {
        Some(o) => {
            let rig = Rig::from_doc(&out)?;
            crate::contact::Body::build(&out, &rig, o)?
        }
        None => None,
    };
    for (si, ai, base) in &jobs {
        let label = &sources[*si].0;
        let src = &mapped[*si];
        let transfer = transfers[*si].as_ref().unwrap();
        let anim = &anims[*si][*ai];
        {
            let frames = ((anim.duration * fps).round() as usize).max(1) + 1;
            let mut name = base.clone();
            let mut k = 2;
            while used_names.contains(&name) {
                name = format!("{}-{}", base, k);
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
            let mut posed: Vec<Vec<Option<Trs>>> = Vec::with_capacity(frames);
            for f in 0..frames {
                let t = if frames > 1 {
                    anim.duration * f as f64 / (frames - 1) as f64
                } else {
                    0.0
                };
                times.push(t);
                let (locals, err) = transfer.frame(&anim.sample(&src.rig, t));
                max_err = max_err.max(err);
                posed.push(locals);
            }
            // Planted feet stay planted when the legs differ (`footpin.rs`).
            let pin_report = if pin {
                Some(crate::footpin::pin_feet(
                    src, &tgt, transfer, anim, &times, &mut posed,
                ))
            } else {
                None
            };
            // Contact pass on the character's own body (`contact.rs`).
            let contact_report = match (&body, contact) {
                (Some(b), Some(o)) => Some(crate::contact::fix_clip(
                    b, &tgt.rig, &mut posed, &times, &name, o,
                )),
                _ => None,
            };
            for mut locals in posed {
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
                contact: contact_report,
                pin: pin_report,
            });
        }
    }
    // The target's replaced animations leave the file.
    out.compact();
    Ok((out, reports))
}

/// Turn the target's hand sockets so a weapon sits in the hand the way it
/// sat in the source's (the same rest transfer the hand gets:
/// `S_t = align^-1 * yaw * S_s`). Sockets keep their own position.
/// Returns the sockets changed.
pub fn sync_sockets(
    out: &mut Document,
    tgt: &Mapped,
    src: &Mapped,
    transfer: &Transfer,
) -> Vec<String> {
    let mut done = Vec::new();
    for side in ["L", "R"] {
        let name = format!("Socket_Hand_{side}");
        let hand = format!("DEF-hand.{side}");
        let (Some(th), Some(sh)) = (tgt.map.node_of(&hand), src.map.node_of(&hand)) else {
            continue;
        };
        let find = |rig: &Rig, h: usize| {
            rig.children[h]
                .iter()
                .copied()
                .find(|&c| rig.names[c] == name)
        };
        let (Some(ts), Some(ss)) = (find(&tgt.rig, th), find(&src.rig, sh)) else {
            continue;
        };
        let Some(k) = transfer.shared.iter().position(|(t, _)| *t == th) else {
            continue;
        };
        let sw = src.rig.rest_world[ss].rotation();
        let want = transfer.align[k].conj().mul(transfer.yaw).mul(sw);
        let local_r = canonical_quat(tgt.rig.rest_world[th].rotation().conj().mul(want));
        let node = &mut out.array_mut("nodes")[ts];
        let (t, _, sc) = crate::glb::node_trs(node);
        crate::glb::set_node_trs(node, t, local_r, sc);
        done.push(name);
    }
    done
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
