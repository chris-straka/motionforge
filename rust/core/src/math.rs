//! Minimal 3D math: Vec3, rotation-only Mat3, unit Quat.
//!
//! Conventions (shared with the Blender exporter, which must implement
//! [`zero_roll_basis`] identically):
//! - Quaternions are `[w, x, y, z]`, unit length, rotation of vectors by
//!   `q * v * q^-1`.
//! - Mat3 is row-major `rows[row][col]`, rotation-only; inverses of
//!   rotations are transposes, so no general inverse exists here.
//! - A zero-roll bone basis is the shortest-arc rotation taking +Y onto
//!   the head->tail direction (matching Blender's `vec_roll_to_mat3`
//!   with roll 0, verified empirically on Blender 5.2 — see
//!   `docs/retarget.md`). The degenerate direction -Y maps via 180 deg
//!   about +Z.

use crate::detmath;
use std::ops::{Add, Mul, Sub};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const ZERO: Vec3 = Vec3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    pub fn new(x: f64, y: f64, z: f64) -> Vec3 {
        Vec3 { x, y, z }
    }

    pub fn dot(self, o: Vec3) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    pub fn cross(self, o: Vec3) -> Vec3 {
        Vec3 {
            x: self.y * o.z - self.z * o.y,
            y: self.z * o.x - self.x * o.z,
            z: self.x * o.y - self.y * o.x,
        }
    }

    pub fn length_sq(self) -> f64 {
        self.dot(self)
    }

    pub fn length(self) -> f64 {
        self.length_sq().sqrt()
    }

    pub fn normalized(self) -> Vec3 {
        let l = self.length();
        if l == 0.0 {
            Vec3::ZERO
        } else {
            self * (1.0 / l)
        }
    }

    pub fn scale(self, s: f64) -> Vec3 {
        Vec3 {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }

    pub fn lerp(self, o: Vec3, t: f64) -> Vec3 {
        self + (o - self) * t
    }

    pub fn approx_eq(self, o: Vec3, eps: f64) -> bool {
        (self - o).length() <= eps
    }
}

impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}

impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}

impl Mul<f64> for Vec3 {
    type Output = Vec3;
    fn mul(self, s: f64) -> Vec3 {
        self.scale(s)
    }
}

/// Row-major 3x3 rotation matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat3 {
    pub rows: [[f64; 3]; 3],
}

impl Mat3 {
    pub fn identity() -> Mat3 {
        Mat3 {
            rows: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    pub fn from_rows(rows: [[f64; 3]; 3]) -> Mat3 {
        Mat3 { rows }
    }

    /// Build from basis columns (x, y, z axes in armature space).
    pub fn from_cols(x: Vec3, y: Vec3, z: Vec3) -> Mat3 {
        Mat3 {
            rows: [[x.x, y.x, z.x], [x.y, y.y, z.y], [x.z, y.z, z.z]],
        }
    }

    pub fn col(self, i: usize) -> Vec3 {
        Vec3::new(self.rows[0][i], self.rows[1][i], self.rows[2][i])
    }

    pub fn rotation_z(angle: f64) -> Mat3 {
        let (s, c) = detmath::sin_cos(angle);
        Mat3::from_rows([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    }

    pub fn transpose(self) -> Mat3 {
        let r = self.rows;
        Mat3::from_rows([
            [r[0][0], r[1][0], r[2][0]],
            [r[0][1], r[1][1], r[2][1]],
            [r[0][2], r[1][2], r[2][2]],
        ])
    }

    pub fn mul_mat(self, o: Mat3) -> Mat3 {
        let mut rows = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                rows[i][j] = self.rows[i][0] * o.rows[0][j]
                    + self.rows[i][1] * o.rows[1][j]
                    + self.rows[i][2] * o.rows[2][j];
            }
        }
        Mat3 { rows }
    }

    pub fn mul_vec(self, v: Vec3) -> Vec3 {
        Vec3::new(
            self.rows[0][0] * v.x + self.rows[0][1] * v.y + self.rows[0][2] * v.z,
            self.rows[1][0] * v.x + self.rows[1][1] * v.y + self.rows[1][2] * v.z,
            self.rows[2][0] * v.x + self.rows[2][1] * v.y + self.rows[2][2] * v.z,
        )
    }

    pub fn approx_eq(self, o: Mat3, eps: f64) -> bool {
        for i in 0..3 {
            for j in 0..3 {
                if (self.rows[i][j] - o.rows[i][j]).abs() > eps {
                    return false;
                }
            }
        }
        true
    }
}

/// Unit quaternion `[w, x, y, z]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quat {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Quat {
    pub const IDENTITY: Quat = Quat {
        w: 1.0,
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    pub fn new(w: f64, x: f64, y: f64, z: f64) -> Quat {
        Quat { w, x, y, z }
    }

    pub fn dot(self, o: Quat) -> f64 {
        self.w * o.w + self.x * o.x + self.y * o.y + self.z * o.z
    }

    pub fn length(self) -> f64 {
        self.dot(self).sqrt()
    }

    pub fn normalized(self) -> Quat {
        let l = self.length();
        if l == 0.0 {
            Quat::IDENTITY
        } else {
            self.scale(1.0 / l)
        }
    }

    pub fn scale(self, s: f64) -> Quat {
        Quat::new(self.w * s, self.x * s, self.y * s, self.z * s)
    }

    pub fn conj(self) -> Quat {
        Quat::new(self.w, -self.x, -self.y, -self.z)
    }

    pub fn neg(self) -> Quat {
        Quat::new(-self.w, -self.x, -self.y, -self.z)
    }

    pub fn mul(self, o: Quat) -> Quat {
        Quat::new(
            self.w * o.w - self.x * o.x - self.y * o.y - self.z * o.z,
            self.w * o.x + self.x * o.w + self.y * o.z - self.z * o.y,
            self.w * o.y - self.x * o.z + self.y * o.w + self.z * o.x,
            self.w * o.z + self.x * o.y - self.y * o.x + self.z * o.w,
        )
    }

    pub fn rotate_vec(self, v: Vec3) -> Vec3 {
        let qv = Quat::new(0.0, v.x, v.y, v.z);
        let r = self.mul(qv).mul(self.conj());
        Vec3::new(r.x, r.y, r.z)
    }

    pub fn from_axis_angle(axis: Vec3, angle: f64) -> Quat {
        let a = axis.normalized();
        let (s, c) = detmath::sin_cos(0.5 * angle);
        Quat::new(c, a.x * s, a.y * s, a.z * s)
    }

    pub fn to_mat3(self) -> Mat3 {
        let (w, x, y, z) = (self.w, self.x, self.y, self.z);
        Mat3::from_rows([
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ])
    }

    /// Rotation matrix -> quaternion (Shepperd's method, branch on trace).
    pub fn from_mat3(m: Mat3) -> Quat {
        let r = m.rows;
        let trace = r[0][0] + r[1][1] + r[2][2];
        let q = if trace > 0.0 {
            let s = (trace + 1.0).sqrt() * 2.0;
            Quat::new(
                0.25 * s,
                (r[2][1] - r[1][2]) / s,
                (r[0][2] - r[2][0]) / s,
                (r[1][0] - r[0][1]) / s,
            )
        } else if r[0][0] > r[1][1] && r[0][0] > r[2][2] {
            let s = (1.0 + r[0][0] - r[1][1] - r[2][2]).sqrt() * 2.0;
            Quat::new(
                (r[2][1] - r[1][2]) / s,
                0.25 * s,
                (r[0][1] + r[1][0]) / s,
                (r[0][2] + r[2][0]) / s,
            )
        } else if r[1][1] > r[2][2] {
            let s = (1.0 + r[1][1] - r[0][0] - r[2][2]).sqrt() * 2.0;
            Quat::new(
                (r[0][2] - r[2][0]) / s,
                (r[0][1] + r[1][0]) / s,
                0.25 * s,
                (r[1][2] + r[2][1]) / s,
            )
        } else {
            let s = (1.0 + r[2][2] - r[0][0] - r[1][1]).sqrt() * 2.0;
            Quat::new(
                (r[1][0] - r[0][1]) / s,
                (r[0][2] + r[2][0]) / s,
                (r[1][2] + r[2][1]) / s,
                0.25 * s,
            )
        };
        q.normalized()
    }

    /// Angle between two orientations in radians: `2*acos(|dot|)`.
    pub fn angle_to(self, o: Quat) -> f64 {
        2.0 * detmath::acos(self.dot(o).abs().min(1.0))
    }

    /// Slerp that also extrapolates for `t` outside `[0, 1]` (used by the
    /// stylizer's exaggeration). Takes the shortest arc (negates `b` when
    /// the dot product is negative).
    pub fn slerp(self, b: Quat, t: f64) -> Quat {
        let mut b = b;
        let mut d = self.dot(b);
        if d < 0.0 {
            b = b.neg();
            d = -d;
        }
        if d > 0.999_999_999_9 {
            // Near-identical: normalized lerp (exact in the limit).
            return Quat::new(
                self.w + (b.w - self.w) * t,
                self.x + (b.x - self.x) * t,
                self.y + (b.y - self.y) * t,
                self.z + (b.z - self.z) * t,
            )
            .normalized();
        }
        let omega = detmath::acos(d.min(1.0));
        let s = detmath::sin(omega);
        let a = detmath::sin((1.0 - t) * omega) / s;
        let c = detmath::sin(t * omega) / s;
        Quat::new(
            self.w * a + b.w * c,
            self.x * a + b.x * c,
            self.y * a + b.y * c,
            self.z * a + b.z * c,
        )
        .normalized()
    }

    pub fn approx_eq(self, o: Quat, eps: f64) -> bool {
        // q and -q are the same rotation; compare the closer sign.
        let d = (self.w - o.w).abs()
            + (self.x - o.x).abs()
            + (self.y - o.y).abs()
            + (self.z - o.z).abs();
        let dn = (self.w + o.w).abs()
            + (self.x + o.x).abs()
            + (self.y + o.y).abs()
            + (self.z + o.z).abs();
        d.min(dn) <= eps
    }
}

/// Shortest-arc quaternion taking unit `from` onto unit `to`.
/// The degenerate opposite case rotates 180 deg about +Z (matching
/// Blender's roll-0 basis for -Y bones, verified empirically).
pub fn shortest_arc(from: Vec3, to: Vec3) -> Quat {
    let d = from.dot(to);
    if d >= 1.0 - 1e-12 {
        return Quat::IDENTITY;
    }
    if d <= -1.0 + 1e-12 {
        return Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), std::f64::consts::PI);
    }
    let axis = from.cross(to);
    // axis is non-degenerate here (|axis| = sin(angle) >> 0).
    Quat::from_axis_angle(axis.normalized(), detmath::acos(d.clamp(-1.0, 1.0)))
}

/// Zero-roll rest basis for a bone running head->tail: the shortest-arc
/// rotation taking +Y onto the bone direction, as a matrix.
pub fn zero_roll_basis(head: Vec3, tail: Vec3) -> Mat3 {
    let dir = (tail - head).normalized();
    shortest_arc(Vec3::new(0.0, 1.0, 0.0), dir).to_mat3()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn quat_axis_angle_roundtrip() {
        let q = Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 0.7);
        let v = q.rotate_vec(Vec3::new(1.0, 0.0, 0.0));
        assert!(v.approx_eq(Vec3::new(0.7f64.cos(), 0.7f64.sin(), 0.0), 1e-12));
    }

    #[test]
    fn mat3_quat_roundtrip() {
        let q = Quat::from_axis_angle(Vec3::new(1.0, 2.0, 3.0), 1.1).normalized();
        let back = Quat::from_mat3(q.to_mat3());
        assert!(q.approx_eq(back, 1e-12));
        // All four Shepperd branches: rotations near 180 deg about each axis.
        for axis in [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ] {
            let q = Quat::from_axis_angle(axis, PI - 0.01);
            assert!(q.approx_eq(Quat::from_mat3(q.to_mat3()), 1e-9));
        }
    }

    #[test]
    fn slerp_endpoints_and_midpoint() {
        let a = Quat::IDENTITY;
        let b = Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), PI / 2.0);
        assert!(a.slerp(b, 0.0).approx_eq(a, 1e-12));
        assert!(a.slerp(b, 1.0).approx_eq(b, 1e-12));
        let mid = a.slerp(b, 0.5);
        assert!((mid.angle_to(a) - PI / 4.0).abs() < 1e-12);
    }

    #[test]
    fn slerp_extrapolates_for_exaggeration() {
        let a = Quat::IDENTITY;
        let b = Quat::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), 0.5);
        let ex = a.slerp(b, 2.0);
        assert!((ex.angle_to(a) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn zero_roll_basis_matches_blender() {
        // Empirical Blender 5.2 roll-0 bases (headless probe, 2026-10-01).
        // Each row: direction -> expected columns (X, Y, Z).
        let cases = [
            (
                Vec3::new(1.0, 0.0, 0.0),
                [(0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)],
            ),
            (
                Vec3::new(-1.0, 0.0, 0.0),
                [(0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)],
            ),
            (
                Vec3::new(0.0, 1.0, 0.0),
                [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
            ),
            (
                Vec3::new(0.0, -1.0, 0.0),
                [(-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)],
            ),
            (
                Vec3::new(0.0, 0.0, 1.0),
                [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)],
            ),
            (
                Vec3::new(0.0, 0.0, -1.0),
                [(1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)],
            ),
        ];
        for (dir, cols) in cases {
            let m = zero_roll_basis(Vec3::ZERO, dir);
            for (i, c) in cols.iter().enumerate() {
                assert!(
                    m.col(i).approx_eq(Vec3::new(c.0, c.1, c.2), 1e-9),
                    "dir {:?} col {}",
                    dir,
                    i
                );
            }
            assert!(m.transpose().mul_mat(m).approx_eq(Mat3::identity(), 1e-12));
        }
        // Diagonal: Y along the bone, orthonormal, right-handed.
        let m = zero_roll_basis(Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0));
        assert!(m
            .col(1)
            .approx_eq(Vec3::new(1.0, 1.0, 1.0).normalized(), 1e-12));
        assert!(m.transpose().mul_mat(m).approx_eq(Mat3::identity(), 1e-12));
        assert!(m.col(0).cross(m.col(1)).approx_eq(m.col(2), 1e-12));
    }
}
