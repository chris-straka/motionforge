//! The creature profile (`motionforge-creature` v1): the skeleton the
//! tool animates (motionforge-skeleton convention: Blender Z-up,
//! head/tail bones named after the rig GLB's joint nodes), a scalar
//! joint-limit cap per bone, the feet, the facing direction, the
//! creature's height, and the two fixed cameras.

use super::camera::{default_views, vec3_from, vec3_json, View};
use crate::clip::{parse_skeleton_obj, Bone, Skeleton};
use crate::json::{self, Json};
use crate::math::Vec3;

pub const CREATURE_FORMAT: &str = "motionforge-creature";
pub const FORMAT_VERSION: f64 = 1.0;

#[derive(Clone, Debug)]
pub struct Creature {
    pub name: String,
    pub skeleton: Skeleton,
    /// Max local rotation angle (degrees) per bone, in bone order.
    pub max_deg: Vec<f64>,
    /// Bone indices whose tails stand on the ground.
    pub feet: Vec<usize>,
    /// Facing direction, horizontal unit vector (Z-up world).
    pub forward: Vec3,
    /// Height of the rest mesh (or skeleton) in metres.
    pub height: f64,
    /// Ground height (rest pose), Z-up.
    pub ground: f64,
    pub views: Vec<View>,
}

impl Creature {
    pub fn root(&self) -> usize {
        self.skeleton
            .bones
            .iter()
            .position(|b| b.parent.is_none())
            .unwrap_or(0)
    }

    /// Bone names in skeleton order (the joint order of detections).
    pub fn joint_names(&self) -> Vec<String> {
        self.skeleton.bones.iter().map(|b| b.name.clone()).collect()
    }

    pub fn view(&self, name: &str) -> Option<&View> {
        self.views.iter().find(|v| v.name == name)
    }

    /// Chain from a foot up to (not including) the bone attached to the
    /// body: the foot bone and its ancestors while each has one child,
    /// at most `max` bones, never the root.
    pub fn leg_chain(&self, foot: usize, max: usize) -> Vec<usize> {
        let mut chain = vec![foot];
        let mut cur = foot;
        while chain.len() < max {
            let Some(p) = self.skeleton.bones[cur].parent else {
                break;
            };
            if self.skeleton.bones[p].parent.is_none() {
                break;
            }
            let kids = self
                .skeleton
                .bones
                .iter()
                .filter(|b| b.parent == Some(p))
                .count();
            if kids != 1 {
                break;
            }
            chain.push(p);
            cur = p;
        }
        chain
    }

    pub fn to_json(&self) -> Json {
        let bones: Vec<Json> = self
            .skeleton
            .bones
            .iter()
            .enumerate()
            .map(|(i, b)| {
                Json::obj(vec![
                    ("name", Json::str(&b.name)),
                    (
                        "parent",
                        match b.parent {
                            Some(p) => Json::str(&self.skeleton.bones[p].name),
                            None => Json::Null,
                        },
                    ),
                    ("head", vec3_json(b.head)),
                    ("tail", vec3_json(b.tail)),
                    ("max_angle_deg", Json::num(self.max_deg[i])),
                ])
            })
            .collect();
        Json::obj(vec![
            ("format", Json::str(CREATURE_FORMAT)),
            ("version", Json::num(FORMAT_VERSION)),
            ("name", Json::str(&self.name)),
            ("height", Json::num(self.height)),
            ("ground", Json::num(self.ground)),
            ("forward", vec3_json(self.forward)),
            (
                "feet",
                Json::Arr(
                    self.feet
                        .iter()
                        .map(|&f| Json::str(&self.skeleton.bones[f].name))
                        .collect(),
                ),
            ),
            ("bones", Json::Arr(bones)),
            (
                "views",
                Json::Arr(self.views.iter().map(View::to_json).collect()),
            ),
        ])
    }

    pub fn emit(&self) -> Result<String, String> {
        json::emit_pretty(&self.to_json())
    }

    pub fn parse(text: &str) -> Result<Creature, String> {
        let v = json::parse(text)?;
        let ctx = "creature";
        match v.get("format").and_then(Json::as_str) {
            Some(f) if f == CREATURE_FORMAT => {}
            other => {
                return Err(format!(
                    "{}: format must be \"{}\", found {:?}",
                    ctx, CREATURE_FORMAT, other
                ))
            }
        }
        match v.get("version").and_then(Json::as_f64) {
            Some(x) if x == FORMAT_VERSION => {}
            other => return Err(format!("{}: unsupported version {:?}", ctx, other)),
        }
        let name = v
            .get("name")
            .and_then(Json::as_str)
            .unwrap_or("creature")
            .to_string();
        let skeleton = parse_skeleton_obj(&v, ctx)?;
        let bones = v.get("bones").and_then(Json::as_arr).unwrap_or(&[]);
        let mut max_deg = Vec::with_capacity(bones.len());
        for (i, b) in bones.iter().enumerate() {
            let m = b
                .get("max_angle_deg")
                .and_then(Json::as_f64)
                .unwrap_or(DEFAULT_MAX_DEG);
            if !(m > 0.0 && m <= 180.0) {
                return Err(format!(
                    "{}: bone {} max_angle_deg must be in (0, 180]",
                    ctx, skeleton.bones[i].name
                ));
            }
            max_deg.push(m);
        }
        let mut feet = Vec::new();
        for f in v.get("feet").and_then(Json::as_arr).unwrap_or(&[]) {
            let n = f
                .as_str()
                .ok_or_else(|| format!("{}: feet must be bone names", ctx))?;
            let i = skeleton
                .index(n)
                .ok_or_else(|| format!("{}: foot \"{}\" is not a bone", ctx, n))?;
            if !feet.contains(&i) {
                feet.push(i);
            }
        }
        let forward = vec3_from(v.get("forward"), &format!("{}: forward", ctx))?;
        let forward = Vec3::new(forward.x, forward.y, 0.0).normalized();
        if forward.length() < 0.5 {
            return Err(format!("{}: forward must be a horizontal direction", ctx));
        }
        let height = v
            .get("height")
            .and_then(Json::as_f64)
            .filter(|h| *h > 0.0 && h.is_finite())
            .ok_or_else(|| format!("{}: height must be > 0", ctx))?;
        let ground = v
            .get("ground")
            .and_then(Json::as_f64)
            .filter(|g| g.is_finite())
            .unwrap_or(0.0);
        let mut views = Vec::new();
        for (i, vj) in v
            .get("views")
            .and_then(Json::as_arr)
            .unwrap_or(&[])
            .iter()
            .enumerate()
        {
            let view = View::from_json(vj, &format!("{}: views[{}]", ctx, i))?;
            if views.iter().any(|w: &View| w.name == view.name) {
                return Err(format!("{}: duplicate view \"{}\"", ctx, view.name));
            }
            views.push(view);
        }
        Ok(Creature {
            name,
            skeleton,
            max_deg,
            feet,
            forward,
            height,
            ground,
            views,
        })
    }
}

pub const DEFAULT_MAX_DEG: f64 = 40.0;
pub const ROOT_MAX_DEG: f64 = 20.0;
pub const HUB_MAX_DEG: f64 = 25.0;
pub const LEAF_MAX_DEG: f64 = 30.0;

/// Default joint-limit caps: the root turns little, hubs (three or more
/// children: a pelvis, a shoulder girdle) bend less than limbs, leaves
/// (paws, ears, a nose) less than the limbs that carry them.
pub fn default_limits(skeleton: &Skeleton) -> Vec<f64> {
    let n = skeleton.len();
    let mut kids = vec![0usize; n];
    for b in &skeleton.bones {
        if let Some(p) = b.parent {
            kids[p] += 1;
        }
    }
    (0..n)
        .map(|i| {
            if skeleton.bones[i].parent.is_none() {
                ROOT_MAX_DEG
            } else if kids[i] >= 3 {
                HUB_MAX_DEG
            } else if kids[i] == 0 {
                LEAF_MAX_DEG
            } else {
                DEFAULT_MAX_DEG
            }
        })
        .collect()
}

/// Build a creature from a skeleton plus geometry facts; the cameras are
/// placed around the rest-pose bounding box.
pub fn build(
    name: &str,
    skeleton: Skeleton,
    feet: Vec<usize>,
    forward: Vec3,
    lo: Vec3,
    hi: Vec3,
    view_size: usize,
) -> Creature {
    let centre = (lo + hi).scale(0.5);
    let height = (hi.z - lo.z).max(1e-6);
    let radius = (hi - lo).length() * 0.5;
    let max_deg = default_limits(&skeleton);
    Creature {
        name: name.to_string(),
        skeleton,
        max_deg,
        feet,
        forward,
        height,
        ground: lo.z,
        views: default_views(centre, radius, forward, view_size),
    }
}

/// Bounding box of a skeleton's heads and tails.
pub fn skeleton_bounds(skeleton: &Skeleton) -> (Vec3, Vec3) {
    let mut lo = Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
    let mut hi = Vec3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for b in &skeleton.bones {
        for p in [b.head, b.tail] {
            lo = Vec3::new(lo.x.min(p.x), lo.y.min(p.y), lo.z.min(p.z));
            hi = Vec3::new(hi.x.max(p.x), hi.y.max(p.y), hi.z.max(p.z));
        }
    }
    (lo, hi)
}

/// Rebuild tails after a skeleton edit: a bone's tail is the mean of its
/// children's heads; a leaf points on along its parent (or up).
pub fn retail(bones: &mut [Bone], min_len: f64) {
    let n = bones.len();
    let heads: Vec<Vec3> = bones.iter().map(|b| b.head).collect();
    let parents: Vec<Option<usize>> = bones.iter().map(|b| b.parent).collect();
    for i in 0..n {
        let kids: Vec<usize> = (0..n).filter(|&c| parents[c] == Some(i)).collect();
        let mut tail = if kids.is_empty() {
            let dir = match parents[i] {
                Some(p) => (heads[i] - heads[p]).normalized(),
                None => Vec3::ZERO,
            };
            let dir = if dir.length() < 0.5 {
                Vec3::new(0.0, 0.0, 1.0)
            } else {
                dir
            };
            let len = match parents[i] {
                Some(p) => ((heads[i] - heads[p]).length() * 0.5).max(min_len),
                None => min_len * 4.0,
            };
            heads[i] + dir.scale(len)
        } else {
            let mut m = Vec3::ZERO;
            for &c in &kids {
                m = m + heads[c];
            }
            m.scale(1.0 / kids.len() as f64)
        };
        if (tail - heads[i]).length() < min_len {
            // Children sit on the head (or very near): point on along the parent.
            let dir = match parents[i] {
                Some(p) => (heads[i] - heads[p]).normalized(),
                None => Vec3::ZERO,
            };
            let dir = if dir.length() < 0.5 {
                Vec3::new(0.0, 0.0, 1.0)
            } else {
                dir
            };
            tail = heads[i] + dir.scale(min_len);
        }
        bones[i].tail = tail;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip::parse_skeleton;

    fn hero() -> Skeleton {
        let text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/hero_skel.json"
        ))
        .unwrap();
        parse_skeleton(&text).unwrap()
    }

    #[test]
    fn round_trip_and_defaults() {
        let sk = hero();
        let feet = vec![
            sk.index("DEF-foot.L").unwrap(),
            sk.index("DEF-foot.R").unwrap(),
        ];
        let (lo, hi) = skeleton_bounds(&sk);
        let c = build("hero", sk, feet, Vec3::new(0.0, -1.0, 0.0), lo, hi, 512);
        let text = c.emit().unwrap();
        let back = Creature::parse(&text).unwrap();
        assert_eq!(back.emit().unwrap(), text);
        assert_eq!(back.feet, c.feet);
        assert_eq!(back.views.len(), 2);
        assert_eq!(back.max_deg[back.root()], ROOT_MAX_DEG);
        assert!(back.leg_chain(back.feet[0], 3).len() >= 2);
    }

    #[test]
    fn rejects_bad_limits_and_feet() {
        let sk = hero();
        let (lo, hi) = skeleton_bounds(&sk);
        let c = build("hero", sk, vec![], Vec3::new(0.0, -1.0, 0.0), lo, hi, 256);
        let text = c.emit().unwrap();
        let bad = text.replacen("\"max_angle_deg\": 20", "\"max_angle_deg\": 0", 1);
        assert!(Creature::parse(&bad).is_err());
        let bad = text.replace("\"feet\": []", "\"feet\": [\"nope\"]");
        assert!(Creature::parse(&bad).is_err());
    }
}
