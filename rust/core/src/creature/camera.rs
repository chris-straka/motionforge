//! Two fixed, known cameras: pinhole model (Blender convention: the
//! camera looks down its local -Z with +Y up; world is Z-up), pixel
//! projection, back-projected rays, two-view triangulation and the
//! "same move in both halves" consistency check.
//!
//! Pixel coordinates: `u` right, `v` down, origin at the top-left of the
//! view's own image (not the side-by-side composite), in pixels.

use crate::detmath;
use crate::json::Json;
use crate::math::{Mat3, Vec3};

#[derive(Clone, Debug, PartialEq)]
pub struct View {
    pub name: String,
    pub position: Vec3,
    pub look_at: Vec3,
    /// Vertical field of view in degrees (perspective).
    pub fov_y_deg: f64,
    pub width: usize,
    pub height: usize,
}

impl View {
    /// Camera basis as a rotation whose columns are the camera's local
    /// X (right), Y (up) and Z (backward) axes in world space.
    pub fn basis(&self) -> Mat3 {
        let fwd = (self.look_at - self.position).normalized();
        let world_up = Vec3::new(0.0, 0.0, 1.0);
        let mut right = fwd.cross(world_up);
        if right.length() < 1e-9 {
            right = Vec3::new(1.0, 0.0, 0.0);
        }
        let right = right.normalized();
        let up = right.cross(fwd).normalized();
        let back = fwd.scale(-1.0);
        Mat3::from_cols(right, up, back)
    }

    fn tan_half_fov_y(&self) -> f64 {
        let half = self.fov_y_deg.to_radians() * 0.5;
        let (s, c) = detmath::sin_cos(half);
        s / c
    }

    fn tan_half_fov_x(&self) -> f64 {
        self.tan_half_fov_y() * self.width as f64 / self.height as f64
    }

    /// Project a world point: `(u, v, depth)`; depth is the distance
    /// along the view direction (negative when behind the camera).
    pub fn project(&self, p: Vec3) -> (f64, f64, f64) {
        let b = self.basis();
        let d = p - self.position;
        let x = b.col(0).dot(d);
        let y = b.col(1).dot(d);
        let depth = -b.col(2).dot(d);
        if depth.abs() < 1e-12 {
            return (f64::NAN, f64::NAN, depth);
        }
        let xn = x / depth / self.tan_half_fov_x();
        let yn = y / depth / self.tan_half_fov_y();
        let u = (xn + 1.0) * 0.5 * self.width as f64;
        let v = (1.0 - yn) * 0.5 * self.height as f64;
        (u, v, depth)
    }

    /// World ray `(origin, unit direction)` through a pixel.
    pub fn ray(&self, u: f64, v: f64) -> (Vec3, Vec3) {
        let b = self.basis();
        let xn = u / self.width as f64 * 2.0 - 1.0;
        let yn = 1.0 - v / self.height as f64 * 2.0;
        let dir = b.col(0).scale(xn * self.tan_half_fov_x())
            + b.col(1).scale(yn * self.tan_half_fov_y())
            + b.col(2).scale(-1.0);
        (self.position, dir.normalized())
    }

    pub fn to_json(&self) -> Json {
        Json::obj(vec![
            ("name", Json::str(&self.name)),
            ("position", vec3_json(self.position)),
            ("look_at", vec3_json(self.look_at)),
            ("fov_y_deg", Json::num(self.fov_y_deg)),
            ("width", Json::num(self.width as f64)),
            ("height", Json::num(self.height as f64)),
        ])
    }

    pub fn from_json(v: &Json, ctx: &str) -> Result<View, String> {
        let name = v
            .get("name")
            .and_then(Json::as_str)
            .ok_or_else(|| format!("{}: view without name", ctx))?
            .to_string();
        let width = v
            .get("width")
            .and_then(Json::as_usize)
            .filter(|&w| w > 0)
            .ok_or_else(|| format!("{}: view {} needs width > 0", ctx, name))?;
        let height = v
            .get("height")
            .and_then(Json::as_usize)
            .filter(|&h| h > 0)
            .ok_or_else(|| format!("{}: view {} needs height > 0", ctx, name))?;
        let fov_y_deg = v
            .get("fov_y_deg")
            .and_then(Json::as_f64)
            .filter(|f| *f > 0.0 && *f < 180.0)
            .ok_or_else(|| format!("{}: view {} needs fov_y_deg in (0, 180)", ctx, name))?;
        let position = vec3_from(
            v.get("position"),
            &format!("{}: view {} position", ctx, name),
        )?;
        let look_at = vec3_from(v.get("look_at"), &format!("{}: view {} look_at", ctx, name))?;
        if (look_at - position).length() < 1e-9 {
            return Err(format!("{}: view {} looks at itself", ctx, name));
        }
        Ok(View {
            name,
            position,
            look_at,
            fov_y_deg,
            width,
            height,
        })
    }
}

pub fn vec3_json(v: Vec3) -> Json {
    Json::Arr(vec![Json::num(v.x), Json::num(v.y), Json::num(v.z)])
}

pub fn vec3_from(v: Option<&Json>, ctx: &str) -> Result<Vec3, String> {
    let arr = v
        .and_then(Json::as_arr)
        .filter(|a| a.len() == 3)
        .ok_or_else(|| format!("{}: expected [x, y, z]", ctx))?;
    let mut out = [0.0; 3];
    for (i, x) in arr.iter().enumerate() {
        out[i] = x
            .as_f64()
            .filter(|f| f.is_finite())
            .ok_or_else(|| format!("{}: component {} is not a finite number", ctx, i))?;
    }
    Ok(Vec3::new(out[0], out[1], out[2]))
}

/// Closest point between two rays: the midpoint of the shortest segment
/// and the segment's length (the "ray gap"; zero when the rays cross).
pub fn triangulate(a: (Vec3, Vec3), b: (Vec3, Vec3)) -> (Vec3, f64) {
    let (pa, pb) = closest_points(a, b);
    ((pa + pb).scale(0.5), (pa - pb).length())
}

/// The closest points of two rays, one on each (`a`'s first).
pub fn closest_points(a: (Vec3, Vec3), b: (Vec3, Vec3)) -> (Vec3, Vec3) {
    let (p, u) = a;
    let (q, v) = b;
    let w = p - q;
    let uu = u.dot(u);
    let uv = u.dot(v);
    let vv = v.dot(v);
    let uw = u.dot(w);
    let vw = v.dot(w);
    let den = uu * vv - uv * uv;
    let (s, t) = if den.abs() < 1e-12 {
        (0.0, if vv > 0.0 { vw / vv } else { 0.0 })
    } else {
        ((uv * vw - vv * uw) / den, (uu * vw - uv * uw) / den)
    };
    (p + u.scale(s), q + v.scale(t))
}

/// Place two cameras around a subject: `front` on the subject's forward
/// side and `side` on its right, both at the subject centre's height,
/// far enough that a sphere of `radius` (with room to move) fills the
/// frame height.
pub fn default_views(centre: Vec3, radius: f64, forward: Vec3, size: usize) -> Vec<View> {
    let fov_y_deg: f64 = 35.0;
    let half = fov_y_deg.to_radians() * 0.5;
    let (s, c) = detmath::sin_cos(half);
    let distance = 1.2 * radius / (s / c);
    let fwd = Vec3::new(forward.x, forward.y, 0.0).normalized();
    let right = fwd.cross(Vec3::new(0.0, 0.0, 1.0)).normalized();
    let mk = |name: &str, dir: Vec3| View {
        name: name.to_string(),
        position: centre + dir.scale(distance),
        look_at: centre,
        fov_y_deg,
        width: size,
        height: size,
    };
    vec![mk("front", fwd), mk("side", right)]
}

/// How well two views agree about one 3D point: the ray gap relative to
/// `scale` (the creature height). A clip whose halves show different
/// moves has gaps of a large fraction of the body.
pub fn gap_ratio(gap: f64, scale: f64) -> f64 {
    if scale > 0.0 {
        gap / scale
    } else {
        gap
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn views() -> Vec<View> {
        default_views(
            Vec3::new(0.0, 0.0, 1.0),
            1.0,
            Vec3::new(0.0, -1.0, 0.0),
            512,
        )
    }

    #[test]
    fn project_then_ray_round_trip() {
        let vs = views();
        let p = Vec3::new(0.3, -0.2, 1.4);
        for v in &vs {
            let (u, w, depth) = v.project(p);
            assert!(depth > 0.0);
            assert!(u > 0.0 && u < 512.0 && w > 0.0 && w < 512.0, "{} {}", u, w);
            let (o, d) = v.ray(u, w);
            let back = o + d.scale((p - o).dot(d));
            assert!(back.approx_eq(p, 1e-9), "{:?} vs {:?}", back, p);
        }
        let (a, b) = (&vs[0], &vs[1]);
        let (pt, gap) = triangulate(
            a.ray(a.project(p).0, a.project(p).1),
            b.ray(b.project(p).0, b.project(p).1),
        );
        assert!(pt.approx_eq(p, 1e-9));
        assert!(gap < 1e-9);
    }

    #[test]
    fn centre_projects_to_image_centre_and_up_is_up() {
        let vs = views();
        for v in &vs {
            let (u, w, _) = v.project(Vec3::new(0.0, 0.0, 1.0));
            assert!((u - 256.0).abs() < 1e-9 && (w - 256.0).abs() < 1e-9);
            let (_, w2, _) = v.project(Vec3::new(0.0, 0.0, 1.5));
            assert!(w2 < 256.0, "higher points are higher in the image");
        }
        // Front camera sits on the forward side; the subject's right
        // appears on the image's left when seen from the front.
        let front = &vs[0];
        assert!(front.position.y < 0.0);
        let right_of_subject = Vec3::new(-0.3, 0.0, 1.0); // forward -Y => right = -X
        let (u, _, _) = front.project(right_of_subject);
        assert!(u < 256.0);
    }

    #[test]
    fn json_round_trip() {
        let v = &views()[1];
        let back = View::from_json(&v.to_json(), "t").unwrap();
        assert_eq!(&back, v);
    }

    #[test]
    fn skew_rays_have_a_gap() {
        let (pt, gap) = triangulate(
            (Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)),
            (Vec3::new(0.0, 1.0, 1.0), Vec3::new(0.0, 0.0, 1.0)),
        );
        assert!((gap - 1.0).abs() < 1e-12);
        assert!(pt.approx_eq(Vec3::new(0.0, 0.5, 0.0), 1e-12));
    }
}
