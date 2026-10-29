//! Seeded, platform-independent random numbers for the creature tools
//! (random training poses, exam motions, synthetic detection noise).
//! xorshift64*; no floats from the platform, no trig.

#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        // SplitMix64 to spread small seeds; never zero.
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        Rng {
            state: if z == 0 { 0x2545_F491_4F6C_DD1D } else { z },
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform in [lo, hi).
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }

    /// Approximately standard normal (Irwin-Hall of 12 uniforms; no log/trig).
    pub fn gauss(&mut self) -> f64 {
        let mut s = 0.0;
        for _ in 0..12 {
            s += self.unit();
        }
        s - 6.0
    }

    /// Uniform point on the unit sphere (rejection sampling in the cube).
    pub fn unit_vec(&mut self) -> crate::math::Vec3 {
        loop {
            let v = crate::math::Vec3::new(
                self.range(-1.0, 1.0),
                self.range(-1.0, 1.0),
                self.range(-1.0, 1.0),
            );
            let l = v.length();
            if l > 0.05 && l <= 1.0 {
                return v.scale(1.0 / l);
            }
        }
    }

    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() % n as u64) as usize
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_in_range() {
        let mut a = Rng::new(7);
        let mut b = Rng::new(7);
        for _ in 0..1000 {
            let u = a.unit();
            assert_eq!(u, b.unit());
            assert!((0.0..1.0).contains(&u));
        }
        let mut g = Rng::new(1);
        let n = 20000;
        let mean: f64 = (0..n).map(|_| g.gauss()).sum::<f64>() / n as f64;
        assert!(mean.abs() < 0.05, "mean {}", mean);
        let v = g.unit_vec();
        assert!((v.length() - 1.0).abs() < 1e-12);
    }
}
