//! Node hierarchy view of a GLB: parents, rest transforms, world
//! matrices, skins, and animation sampling. Read-only; the edits live in
//! `standardize`, `animate` and `posetest`.

use crate::glb::{node_local, node_trs, Affine, Document};
use crate::json::Json;
use crate::math::{Quat, Vec3};

/// Local TRS of one node (posed or rest).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Trs {
    pub t: Vec3,
    pub r: Quat,
    pub s: Vec3,
}

impl Trs {
    pub fn affine(&self) -> Affine {
        Affine::from_trs(self.t, self.r, self.s)
    }
}

#[derive(Clone, Debug)]
pub struct Rig {
    pub names: Vec<String>,
    pub parent: Vec<Option<usize>>,
    pub children: Vec<Vec<usize>>,
    /// Rest local transforms as stored (matrix nodes decomposed).
    pub rest: Vec<Trs>,
    /// Rest local matrices exactly as stored.
    pub rest_local: Vec<Affine>,
    /// Rest world matrices.
    pub rest_world: Vec<Affine>,
    /// Joint node indices per skin.
    pub skins: Vec<Vec<usize>>,
    /// Nodes in parent-before-child order.
    pub order: Vec<usize>,
}

impl Rig {
    pub fn from_doc(doc: &Document) -> Result<Rig, String> {
        let nodes = doc.array("nodes");
        let n = nodes.len();
        let mut names = Vec::with_capacity(n);
        let mut parent = vec![None; n];
        let mut children = vec![Vec::new(); n];
        for (i, node) in nodes.iter().enumerate() {
            names.push(
                node.get("name")
                    .and_then(Json::as_str)
                    .map(str::to_string)
                    .unwrap_or_else(|| format!("node{}", i)),
            );
            for c in node.get("children").and_then(Json::as_arr).unwrap_or(&[]) {
                let c = c
                    .as_usize()
                    .filter(|&c| c < n)
                    .ok_or_else(|| format!("node {} has a bad child", i))?;
                if parent[c].is_some() || c == i {
                    return Err(format!("node {} has more than one parent", c));
                }
                parent[c] = Some(i);
                children[i].push(c);
            }
        }
        let mut order = Vec::with_capacity(n);
        let mut stack: Vec<usize> = (0..n).rev().filter(|&i| parent[i].is_none()).collect();
        while let Some(i) = stack.pop() {
            order.push(i);
            for &c in children[i].iter().rev() {
                stack.push(c);
            }
        }
        if order.len() != n {
            return Err("node hierarchy has a cycle".into());
        }
        let rest_local: Vec<Affine> = nodes.iter().map(node_local).collect();
        let rest = nodes
            .iter()
            .map(|node| {
                let (t, r, s) = node_trs(node);
                Trs { t, r, s }
            })
            .collect();
        let mut rest_world = vec![Affine::IDENTITY; n];
        for &i in &order {
            rest_world[i] = match parent[i] {
                Some(p) => rest_world[p].mul(&rest_local[i]),
                None => rest_local[i],
            };
        }
        let mut skins = Vec::new();
        for (si, skin) in doc.array("skins").iter().enumerate() {
            let joints: Vec<usize> = skin
                .get("joints")
                .and_then(Json::as_arr)
                .unwrap_or(&[])
                .iter()
                .map(|j| {
                    j.as_usize()
                        .filter(|&j| j < n)
                        .ok_or_else(|| format!("skin {} has a bad joint", si))
                })
                .collect::<Result<_, _>>()?;
            if joints.is_empty() {
                return Err(format!("skin {} has no joints", si));
            }
            skins.push(joints);
        }
        Ok(Rig {
            names,
            parent,
            children,
            rest,
            rest_local,
            rest_world,
            skins,
            order,
        })
    }

    pub fn find(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|n| n == name)
    }

    /// Every joint of every skin, deduplicated, in first-seen order.
    pub fn joints(&self) -> Vec<usize> {
        let mut out = Vec::new();
        for skin in &self.skins {
            for &j in skin {
                if !out.contains(&j) {
                    out.push(j);
                }
            }
        }
        out
    }

    pub fn is_ancestor(&self, ancestor: usize, mut node: usize) -> bool {
        while let Some(p) = self.parent[node] {
            if p == ancestor {
                return true;
            }
            node = p;
        }
        false
    }

    pub fn head(&self, node: usize) -> Vec3 {
        self.rest_world[node].t
    }

    /// World matrices for a pose given as local overrides (`None` keeps
    /// the stored rest matrix, which preserves matrix-only nodes exactly).
    pub fn world(&self, locals: &[Option<Trs>]) -> Vec<Affine> {
        let mut world = vec![Affine::IDENTITY; self.names.len()];
        for &i in &self.order {
            let local = match locals.get(i).copied().flatten() {
                Some(trs) => trs.affine(),
                None => self.rest_local[i],
            };
            world[i] = match self.parent[i] {
                Some(p) => world[p].mul(&local),
                None => local,
            };
        }
        world
    }
}

/// One sampled animation channel.
#[derive(Clone, Debug)]
struct Channel {
    node: usize,
    path: String,
    times: Vec<f64>,
    values: Vec<Vec<f64>>,
    interp: String,
}

/// A glTF animation ready to sample at arbitrary times.
#[derive(Clone, Debug)]
pub struct Animation {
    pub name: String,
    pub duration: f64,
    channels: Vec<Channel>,
}

impl Animation {
    pub fn load_all(doc: &Document) -> Result<Vec<Animation>, String> {
        let mut out = Vec::new();
        for (ai, anim) in doc.array("animations").iter().enumerate() {
            let name = anim
                .get("name")
                .and_then(Json::as_str)
                .map(str::to_string)
                .unwrap_or_else(|| format!("clip{}", ai));
            let samplers = anim.get("samplers").and_then(Json::as_arr).unwrap_or(&[]);
            let mut channels = Vec::new();
            let mut duration: f64 = 0.0;
            for ch in anim.get("channels").and_then(Json::as_arr).unwrap_or(&[]) {
                let Some(target) = ch.get("target") else {
                    continue;
                };
                let Some(node) = target.get("node").and_then(Json::as_usize) else {
                    continue;
                };
                let path = target
                    .get("path")
                    .and_then(Json::as_str)
                    .unwrap_or("")
                    .to_string();
                if !matches!(path.as_str(), "translation" | "rotation" | "scale") {
                    continue;
                }
                let sampler = ch
                    .get("sampler")
                    .and_then(Json::as_usize)
                    .and_then(|s| samplers.get(s))
                    .ok_or_else(|| format!("animation {} channel has a bad sampler", name))?;
                let input = sampler
                    .get("input")
                    .and_then(Json::as_usize)
                    .ok_or("sampler without input")?;
                let output = sampler
                    .get("output")
                    .and_then(Json::as_usize)
                    .ok_or("sampler without output")?;
                let interp = sampler
                    .get("interpolation")
                    .and_then(Json::as_str)
                    .unwrap_or("LINEAR")
                    .to_string();
                let times: Vec<f64> = doc
                    .read_accessor(input)?
                    .into_iter()
                    .map(|r| r[0])
                    .collect();
                let mut values = doc.read_accessor(output)?;
                if interp == "CUBICSPLINE" {
                    // (in-tangent, value, out-tangent) triplets: keep the
                    // values and sample them linearly.
                    values = values
                        .chunks(3)
                        .filter(|c| c.len() == 3)
                        .map(|c| c[1].clone())
                        .collect();
                }
                if times.is_empty() || values.len() != times.len() {
                    return Err(format!(
                        "animation {}: keyframe/value counts disagree",
                        name
                    ));
                }
                duration = duration.max(*times.last().unwrap());
                channels.push(Channel {
                    node,
                    path,
                    times,
                    values,
                    interp,
                });
            }
            out.push(Animation {
                name,
                duration,
                channels,
            });
        }
        Ok(out)
    }

    /// Local transforms of every node at time `t` (rest for unkeyed nodes).
    pub fn sample(&self, rig: &Rig, t: f64) -> Vec<Option<Trs>> {
        let mut locals: Vec<Option<Trs>> = vec![None; rig.names.len()];
        for ch in &self.channels {
            if ch.node >= locals.len() {
                continue;
            }
            let mut trs = locals[ch.node].unwrap_or(rig.rest[ch.node]);
            let v = sample_channel(ch, t);
            match ch.path.as_str() {
                "translation" if v.len() == 3 => trs.t = Vec3::new(v[0], v[1], v[2]),
                "scale" if v.len() == 3 => trs.s = Vec3::new(v[0], v[1], v[2]),
                "rotation" if v.len() == 4 => {
                    trs.r = Quat::new(v[3], v[0], v[1], v[2]).normalized()
                }
                _ => {}
            }
            locals[ch.node] = Some(trs);
        }
        locals
    }

    pub fn animated_nodes(&self) -> Vec<usize> {
        let mut out: Vec<usize> = self.channels.iter().map(|c| c.node).collect();
        out.sort_unstable();
        out.dedup();
        out
    }
}

fn sample_channel(ch: &Channel, t: f64) -> Vec<f64> {
    let times = &ch.times;
    if t <= times[0] {
        return ch.values[0].clone();
    }
    let last = times.len() - 1;
    if t >= times[last] {
        return ch.values[last].clone();
    }
    let k = times.partition_point(|&x| x <= t) - 1;
    let (t0, t1) = (times[k], times[k + 1]);
    let (a, b) = (&ch.values[k], &ch.values[k + 1]);
    if ch.interp == "STEP" || t1 <= t0 {
        return a.clone();
    }
    let u = (t - t0) / (t1 - t0);
    if ch.path == "rotation" && a.len() == 4 {
        let qa = Quat::new(a[3], a[0], a[1], a[2]);
        let qb = Quat::new(b[3], b[0], b[1], b[2]);
        let q = qa.slerp(qb, u);
        return vec![q.x, q.y, q.z, q.w];
    }
    a.iter().zip(b).map(|(x, y)| x + (y - x) * u).collect()
}
