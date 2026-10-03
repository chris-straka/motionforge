//! Cross-platform deterministic transcendentals: `sin`, `cos`, `sin_cos`,
//! `asin`, `acos`, `exp`.
//!
//! `f64::sin` & co. call the platform libm (Apple libSystem on macOS,
//! glibc on Linux, ...). Those are not correctly rounded and disagree in
//! the last ulp, which broke the byte-determinism contract across
//! machines (the stylize golden drifted on Linux x86_64 vs the macOS
//! aarch64 machine that generated it). These ports of the FreeBSD msun /
//! musl routines (via rust-lang/libm, MIT) use only IEEE-754 `+ - * /`,
//! `sqrt`, `floor` and bit manipulation, all of which are exactly
//! rounded on every target, so the same input gives the same bits
//! everywhere. Accuracy is < 1 ulp, same as the originals.
//!
//! Core code must call these instead of the `f64` methods; `sqrt`,
//! `powi` with small constant exponents, `to_degrees`/`to_radians`,
//! `abs`, `min`/`max`, `floor` are already exact and stay as-is.
//!
//! ====================================================
//! Copyright (C) 1993, 2004 by Sun Microsystems, Inc. All rights reserved.
//!
//! Developed at SunSoft, a Sun Microsystems, Inc. business.
//! Permission to use, copy, modify, and distribute this
//! software is freely granted, provided that this notice
//! is preserved.
//! ====================================================

// ---------------------------------------------------------------- sin/cos

const S1: f64 = -1.66666666666666324348e-01; // 0xBFC55555, 0x55555549
const S2: f64 = 8.33333333332248946124e-03; // 0x3F811111, 0x1110F8A6
const S3: f64 = -1.98412698298579493134e-04; // 0xBF2A01A0, 0x19C161D5
const S4: f64 = 2.75573137070700676789e-06; // 0x3EC71DE3, 0x57B1FE7D
const S5: f64 = -2.50507602534068634195e-08; // 0xBE5AE5E6, 0x8A2B9CEB
const S6: f64 = 1.58969099521155010221e-10; // 0x3DE5D93A, 0x5ACFD57C

/// Kernel sin on ~[-pi/4, pi/4]; `y` is the tail of `x` (ignored when
/// `iy == 0`). Callers handle -0 and tiny `x`.
fn k_sin(x: f64, y: f64, iy: i32) -> f64 {
    let z = x * x;
    let w = z * z;
    let r = S2 + z * (S3 + z * S4) + z * w * (S5 + z * S6);
    let v = z * x;
    if iy == 0 {
        x + v * (S1 + z * r)
    } else {
        x - ((z * (0.5 * y - v * r) - y) - v * S1)
    }
}

const C1: f64 = 4.16666666666666019037e-02; // 0x3FA55555, 0x5555554C
const C2: f64 = -1.38888888888741095749e-03; // 0xBF56C16C, 0x16C15177
const C3: f64 = 2.48015872894767294178e-05; // 0x3EFA01A0, 0x19CB1590
const C4: f64 = -2.75573143513906633035e-07; // 0xBE927E4F, 0x809C52AD
const C5: f64 = 2.08757232129817482790e-09; // 0x3E21EE9E, 0xBDB4B1C4
const C6: f64 = -1.13596475577881948265e-11; // 0xBDA8FAE9, 0xBE8838D4

/// Kernel cos on [-pi/4, pi/4]; `y` is the tail of `x`.
fn k_cos(x: f64, y: f64) -> f64 {
    let z = x * x;
    let w = z * z;
    let r = z * (C1 + z * (C2 + z * C3)) + w * w * (C4 + z * (C5 + z * C6));
    let hz = 0.5 * z;
    let w = 1.0 - hz;
    w + (((1.0 - w) - hz) + (z * r - x * y))
}

const TO_INT: f64 = 1.5 / f64::EPSILON;
const INV_PIO2: f64 = 6.36619772367581382433e-01; // 0x3FE45F30, 0x6DC9C883
const PIO2_1: f64 = 1.57079632673412561417e+00; // 0x3FF921FB, 0x54400000
const PIO2_1T: f64 = 6.07710050650619224932e-11; // 0x3DD0B461, 0x1A626331
const PIO2_2: f64 = 6.07710050630396597660e-11; // 0x3DD0B461, 0x1A600000
const PIO2_2T: f64 = 2.02226624879595063154e-21; // 0x3BA3198A, 0x2E037073
const PIO2_3: f64 = 2.02226624871116645580e-21; // 0x3BA3198A, 0x2E000000
const PIO2_3T: f64 = 8.47842766036889956997e-32; // 0x397B839A, 0x252049C1

/// `x rem pi/2` as `(n, y0, y1)` with `x = n*pi/2 + y0 + y1`.
/// Caller handles `|x| ~<= pi/4` (no reduction needed) and inf/NaN.
fn rem_pio2(x: f64) -> (i32, f64, f64) {
    let sign = (x.to_bits() >> 63) != 0;
    let ix = (x.to_bits() >> 32) as u32 & 0x7fffffff;

    fn medium(x: f64, ix: u32) -> (i32, f64, f64) {
        // rint(x/(pi/2)), round-to-nearest.
        let f_n = (x * INV_PIO2 + TO_INT) - TO_INT;
        let n = f_n as i32;
        let mut r = x - f_n * PIO2_1;
        let mut w = f_n * PIO2_1T; // 1st round, good to 85 bits
        let mut y0 = r - w;
        let ey = (y0.to_bits() >> 52) as i32 & 0x7ff;
        let ex = (ix >> 20) as i32;
        if ex - ey > 16 {
            // 2nd round, good to 118 bits
            let t = r;
            w = f_n * PIO2_2;
            r = t - w;
            w = f_n * PIO2_2T - ((t - r) - w);
            y0 = r - w;
            let ey = (y0.to_bits() >> 52) as i32 & 0x7ff;
            if ex - ey > 49 {
                // 3rd round, good to 151 bits, covers all cases
                let t = r;
                w = f_n * PIO2_3;
                r = t - w;
                w = f_n * PIO2_3T - ((t - r) - w);
                y0 = r - w;
            }
        }
        let y1 = (r - y0) - w;
        (n, y0, y1)
    }

    // Small multiples of pi/2: one subtraction of k*(pi/2) is enough,
    // except right next to a multiple (cancellation -> medium path).
    let small = |k: i32| -> (i32, f64, f64) {
        let kf = k as f64;
        if !sign {
            let z = x - kf * PIO2_1;
            let y0 = z - kf * PIO2_1T;
            (k, y0, (z - y0) - kf * PIO2_1T)
        } else {
            let z = x + kf * PIO2_1;
            let y0 = z + kf * PIO2_1T;
            (-k, y0, (z - y0) + kf * PIO2_1T)
        }
    };

    if ix <= 0x400f6a7a {
        // |x| ~<= 5pi/4
        if (ix & 0xfffff) == 0x921fb {
            return medium(x, ix); // |x| ~= pi/2 or 2pi/2
        }
        return if ix <= 0x4002d97c { small(1) } else { small(2) };
    }
    if ix <= 0x401c463b {
        // |x| ~<= 9pi/4
        if ix <= 0x4015fdbc {
            if ix == 0x4012d97c {
                return medium(x, ix); // |x| ~= 3pi/2
            }
            return small(3);
        }
        if ix == 0x401921fb {
            return medium(x, ix); // |x| ~= 4pi/2
        }
        return small(4);
    }
    if ix < 0x413921fb {
        // |x| ~< 2^20*(pi/2)
        return medium(x, ix);
    }
    // Large arguments: split |x| into three 24-bit chunks and reduce
    // against a long expansion of 2/pi.
    let x1p24 = f64::from_bits(0x4170000000000000);
    let mut ui = x.to_bits();
    ui &= u64::MAX >> 12;
    ui |= (0x3ff + 23) << 52;
    let mut z = f64::from_bits(ui);
    let mut tx = [0.0; 3];
    for t in tx.iter_mut().take(2) {
        *t = z as i32 as f64;
        z = (z - *t) * x1p24;
    }
    tx[2] = z;
    let mut nx = 3;
    while nx > 1 && tx[nx - 1] == 0.0 {
        nx -= 1;
    }
    let (n, y0, y1) = rem_pio2_large(&tx[..nx], ((ix as i32) >> 20) - (0x3ff + 23));
    if sign {
        (-n, -y0, -y1)
    } else {
        (n, y0, y1)
    }
}

/// 2/pi in 24-bit chunks: `IPIO2[i] * 2^(-24(i+1))`. 66 terms cover
/// every f64 exponent at double precision ((e0-3)/24 + jk <= 46).
const IPIO2: [i32; 66] = [
    0xA2F983, 0x6E4E44, 0x1529FC, 0x2757D1, 0xF534DD, 0xC0DB62, 0x95993C, 0x439041, 0xFE5163,
    0xABDEBB, 0xC561B7, 0x246E3A, 0x424DD2, 0xE00649, 0x2EEA09, 0xD1921C, 0xFE1DEB, 0x1CB129,
    0xA73EE8, 0x8235F5, 0x2EBB44, 0x84E99C, 0x7026B4, 0x5F7E41, 0x3991D6, 0x398353, 0x39F49C,
    0x845F8B, 0xBDF928, 0x3B1FF8, 0x97FFDE, 0x05980F, 0xEF2F11, 0x8B5A0A, 0x6D1F6D, 0x367ECF,
    0x27CB09, 0xB74F46, 0x3F669E, 0x5FEA2D, 0x7527BA, 0xC7EBE5, 0xF17B3D, 0x0739F7, 0x8A5292,
    0xEA6BFB, 0x5FB11F, 0x8D5D08, 0x560330, 0x46FC7B, 0x6BABF0, 0xCFBC20, 0x9AF436, 0x1DA9E3,
    0x91615E, 0xE61B08, 0x659985, 0x5F14A0, 0x68408D, 0xFFD880, 0x4D7327, 0x310606, 0x1556CA,
    0x73A8C9, 0x60E27B, 0xC08C6B,
];

/// pi/2 in 24-bit chunks.
const PIO2: [f64; 8] = [
    1.57079625129699707031e+00, // 0x3FF921FB, 0x40000000
    7.54978941586159635335e-08, // 0x3E74442D, 0x00000000
    5.39030252995776476554e-15, // 0x3CF84698, 0x80000000
    3.28200341580791294123e-22, // 0x3B78CC51, 0x60000000
    1.27065575308067607349e-29, // 0x39F01B83, 0x80000000
    1.22933308981111328932e-36, // 0x387A2520, 0x40000000
    2.73370053816464559624e-44, // 0x36E38222, 0x80000000
    2.16741683877804819444e-51, // 0x3569F31D, 0x00000000
];

/// FreeBSD `__kernel_rem_pio2` at double precision (prec = 1, jk = 4):
/// `x` is positive, given as 24-bit integer chunks with `x[0]*2^e0`
/// matching x's top 24 bits. Returns `(n mod 8, y0, y1)`.
fn rem_pio2_large(x: &[f64], e0: i32) -> (i32, f64, f64) {
    let x1p24 = f64::from_bits(0x4170000000000000);
    let x1p_24 = f64::from_bits(0x3e70000000000000);
    const JK: usize = 4;
    const JP: usize = JK;

    let mut f = [0.0f64; 20];
    let mut fq = [0.0f64; 20];
    let mut q = [0.0f64; 20];
    let mut iq = [0i32; 20];

    let jx = x.len() - 1;
    let jv = ((e0 - 3) / 24).max(0);
    let mut q0 = e0 - 24 * (jv + 1);
    let jv = jv as usize;

    // f[0..=jx+jk] = IPIO2[jv-jx ..= jv+jk] (zero below the table).
    let mut j = jv as i32 - jx as i32;
    for fi in f.iter_mut().take(jx + JK + 1) {
        *fi = if j < 0 { 0.0 } else { IPIO2[j as usize] as f64 };
        j += 1;
    }
    for i in 0..=JK {
        let mut fw = 0.0;
        for j in 0..=jx {
            fw += x[j] * f[jx + i - j];
        }
        q[i] = fw;
    }

    let mut jz = JK;
    let mut z;
    let mut n;
    let mut ih;
    loop {
        // Distill q[] into iq[] reversingly.
        z = q[jz];
        let mut i = 0usize;
        for j in (1..=jz).rev() {
            let fw = (x1p_24 * z) as i32 as f64;
            iq[i] = (z - x1p24 * fw) as i32;
            z = q[j - 1] + fw;
            i += 1;
        }

        z = scalbn(z, q0);
        z -= 8.0 * (z * 0.125).floor(); // trim off integer >= 8
        n = z as i32;
        z -= n as f64;
        ih = 0;
        if q0 > 0 {
            // need iq[jz-1] to determine n
            let k = iq[jz - 1] >> (24 - q0);
            n += k;
            iq[jz - 1] -= k << (24 - q0);
            ih = iq[jz - 1] >> (23 - q0);
        } else if q0 == 0 {
            ih = iq[jz - 1] >> 23;
        } else if z >= 0.5 {
            ih = 2;
        }

        if ih > 0 {
            // q > 0.5: compute 1 - q
            n += 1;
            let mut carry = 0;
            for v in iq.iter_mut().take(jz) {
                let j = *v;
                if carry == 0 {
                    if j != 0 {
                        carry = 1;
                        *v = 0x1000000 - j;
                    }
                } else {
                    *v = 0xffffff - j;
                }
            }
            if q0 > 0 {
                // rare case: chance is 1 in 12
                match q0 {
                    1 => iq[jz - 1] &= 0x7fffff,
                    2 => iq[jz - 1] &= 0x3fffff,
                    _ => {}
                }
            }
            if ih == 2 {
                z = 1.0 - z;
                if carry != 0 {
                    z -= scalbn(1.0, q0);
                }
            }
        }

        // Recompute with more terms if the fraction cancelled to zero.
        if z == 0.0 && iq[JK..jz].iter().all(|&v| v == 0) {
            let mut k = 1;
            while iq[JK - k] == 0 {
                k += 1;
            }
            for i in (jz + 1)..=(jz + k) {
                f[jx + i] = IPIO2[jv + i] as f64;
                let mut fw = 0.0;
                for j in 0..=jx {
                    fw += x[j] * f[jx + i - j];
                }
                q[i] = fw;
            }
            jz += k;
            continue;
        }
        break;
    }

    // Chop off zero terms, or break z into 24-bit chunks.
    if z == 0.0 {
        jz -= 1;
        q0 -= 24;
        while iq[jz] == 0 {
            jz -= 1;
            q0 -= 24;
        }
    } else {
        z = scalbn(z, -q0);
        if z >= x1p24 {
            let fw = (x1p_24 * z) as i32 as f64;
            iq[jz] = (z - x1p24 * fw) as i32;
            jz += 1;
            q0 += 24;
            iq[jz] = fw as i32;
        } else {
            iq[jz] = z as i32;
        }
    }

    let mut fw = scalbn(1.0, q0);
    for i in (0..=jz).rev() {
        q[i] = fw * iq[i] as f64;
        fw *= x1p_24;
    }
    // fq = PIO2[0..=jp] * q[jz..=0]
    for i in (0..=jz).rev() {
        let mut fw = 0.0;
        let mut k = 0;
        while k <= JP && k <= jz - i {
            fw += PIO2[k] * q[i + k];
            k += 1;
        }
        fq[jz - i] = fw;
    }
    // Compress fq[] into (y0, y1).
    let mut fw = 0.0;
    for i in (0..=jz).rev() {
        fw += fq[i];
    }
    let y0 = if ih == 0 { fw } else { -fw };
    fw = fq[0] - fw;
    for v in fq.iter().take(jz + 1).skip(1) {
        fw += v;
    }
    let y1 = if ih == 0 { fw } else { -fw };
    (n & 7, y0, y1)
}

/// `x * 2^n`, exact unless it under/overflows (musl `scalbn`).
fn scalbn(x: f64, mut n: i32) -> f64 {
    let x1p1023 = f64::from_bits(0x7fe0000000000000);
    let x1p53 = f64::from_bits(0x4340000000000000);
    let x1p_1022 = f64::from_bits(0x0010000000000000);
    let mut y = x;
    if n > 1023 {
        y *= x1p1023;
        n -= 1023;
        if n > 1023 {
            y *= x1p1023;
            n -= 1023;
            if n > 1023 {
                n = 1023;
            }
        }
    } else if n < -1022 {
        // Scale by 2^-1022 * 2^53 to keep 53 bits through the subnormal
        // range, matching musl.
        y *= x1p_1022 * x1p53;
        n += 1022 - 53;
        if n < -1022 {
            y *= x1p_1022 * x1p53;
            n += 1022 - 53;
            if n < -1022 {
                n = -1022;
            }
        }
    }
    y * f64::from_bits(((0x3ff + n) as u64) << 52)
}

/// Deterministic `sin(x)`, `x` in radians.
pub fn sin(x: f64) -> f64 {
    let ix = (x.to_bits() >> 32) as u32 & 0x7fffffff;
    if ix <= 0x3fe921fb {
        // |x| ~< pi/4
        if ix < 0x3e500000 {
            return x; // |x| < 2^-26 (also preserves -0)
        }
        return k_sin(x, 0.0, 0);
    }
    if ix >= 0x7ff00000 {
        return x - x; // inf or NaN -> NaN
    }
    let (n, y0, y1) = rem_pio2(x);
    match n & 3 {
        0 => k_sin(y0, y1, 1),
        1 => k_cos(y0, y1),
        2 => -k_sin(y0, y1, 1),
        _ => -k_cos(y0, y1),
    }
}

/// Deterministic `cos(x)`, `x` in radians.
pub fn cos(x: f64) -> f64 {
    let ix = (x.to_bits() >> 32) as u32 & 0x7fffffff;
    if ix <= 0x3fe921fb {
        if ix < 0x3e46a09e {
            return 1.0; // |x| < 2^-27 * sqrt(2)
        }
        return k_cos(x, 0.0);
    }
    if ix >= 0x7ff00000 {
        return x - x;
    }
    let (n, y0, y1) = rem_pio2(x);
    match n & 3 {
        0 => k_cos(y0, y1),
        1 => -k_sin(y0, y1, 1),
        2 => -k_cos(y0, y1),
        _ => k_sin(y0, y1, 1),
    }
}

/// Deterministic `(sin(x), cos(x))`; bit-identical to calling [`sin`]
/// and [`cos`] separately.
pub fn sin_cos(x: f64) -> (f64, f64) {
    let ix = (x.to_bits() >> 32) as u32 & 0x7fffffff;
    if ix <= 0x3fe921fb {
        if ix < 0x3e46a09e {
            return (x, 1.0); // |x| < 2^-27 * sqrt(2): same as sin / cos
        }
        return (k_sin(x, 0.0, 0), k_cos(x, 0.0));
    }
    if ix >= 0x7ff00000 {
        let nan = x - x;
        return (nan, nan);
    }
    let (n, y0, y1) = rem_pio2(x);
    let s = k_sin(y0, y1, 1);
    let c = k_cos(y0, y1);
    match n & 3 {
        0 => (s, c),
        1 => (c, -s),
        2 => (-s, -c),
        _ => (-c, s),
    }
}

// -------------------------------------------------------------- asin/acos

const PIO2_HI: f64 = 1.57079632679489655800e+00; // 0x3FF921FB, 0x54442D18
const PIO2_LO: f64 = 6.12323399573676603587e-17; // 0x3C91A626, 0x33145C07
const PS0: f64 = 1.66666666666666657415e-01; // 0x3FC55555, 0x55555555
const PS1: f64 = -3.25565818622400915405e-01; // 0xBFD4D612, 0x03EB6F7D
const PS2: f64 = 2.01212532134862925881e-01; // 0x3FC9C155, 0x0E884455
const PS3: f64 = -4.00555345006794114027e-02; // 0xBFA48228, 0xB5688F3B
const PS4: f64 = 7.91534994289814532176e-04; // 0x3F49EFE0, 0x7501B288
const PS5: f64 = 3.47933107596021167570e-05; // 0x3F023DE1, 0x0DFDF709
const QS1: f64 = -2.40339491173441421878e+00; // 0xC0033A27, 0x1C8A2D4B
const QS2: f64 = 2.02094576023350569471e+00; // 0x40002AE5, 0x9C598AC8
const QS3: f64 = -6.88283971605453293030e-01; // 0xBFE6066C, 0x1B8D0159
const QS4: f64 = 7.70381505559019352791e-02; // 0x3FB3B8C5, 0xB12E9282

/// Rational approximation of `(asin(x) - x) / x^3` in `z = x^2`.
fn asin_r(z: f64) -> f64 {
    let p = z * (PS0 + z * (PS1 + z * (PS2 + z * (PS3 + z * (PS4 + z * PS5)))));
    let q = 1.0 + z * (QS1 + z * (QS2 + z * (QS3 + z * QS4)));
    p / q
}

/// Zero the low 32 bits (the "hi part" of a double).
fn hi_word_only(x: f64) -> f64 {
    f64::from_bits(x.to_bits() & 0xffff_ffff_0000_0000)
}

/// Deterministic `asin(x)`; NaN outside [-1, 1].
pub fn asin(x: f64) -> f64 {
    let hx = (x.to_bits() >> 32) as u32;
    let ix = hx & 0x7fffffff;
    if ix >= 0x3ff00000 {
        // |x| >= 1 or NaN
        let lx = x.to_bits() as u32;
        if ((ix - 0x3ff00000) | lx) == 0 {
            return x * PIO2_HI + f64::from_bits(0x3870000000000000); // +-pi/2
        }
        return 0.0 / (x - x);
    }
    if ix < 0x3fe00000 {
        // |x| < 0.5
        if (0x00100000..0x3e500000).contains(&ix) {
            return x; // 2^-1022 <= |x| < 2^-26
        }
        return x + x * asin_r(x * x);
    }
    // 0.5 <= |x| < 1
    let z = (1.0 - x.abs()) * 0.5;
    let s = z.sqrt();
    let r = asin_r(z);
    let y = if ix >= 0x3fef3333 {
        // |x| > 0.975
        PIO2_HI - (2.0 * (s + s * r) - PIO2_LO)
    } else {
        let f = hi_word_only(s);
        let c = (z - f * f) / (s + f);
        0.5 * PIO2_HI - (2.0 * s * r - (PIO2_LO - 2.0 * c) - (0.5 * PIO2_HI - 2.0 * f))
    };
    if hx >> 31 != 0 {
        -y
    } else {
        y
    }
}

/// Deterministic `acos(x)`; NaN outside [-1, 1].
pub fn acos(x: f64) -> f64 {
    let x1p_120 = f64::from_bits(0x3870000000000000);
    let hx = (x.to_bits() >> 32) as u32;
    let ix = hx & 0x7fffffff;
    if ix >= 0x3ff00000 {
        let lx = x.to_bits() as u32;
        if ((ix - 0x3ff00000) | lx) == 0 {
            // acos(1) = 0, acos(-1) = pi
            return if hx >> 31 != 0 {
                2.0 * PIO2_HI + x1p_120
            } else {
                0.0
            };
        }
        return 0.0 / (x - x);
    }
    if ix < 0x3fe00000 {
        // |x| < 0.5
        if ix <= 0x3c600000 {
            return PIO2_HI + x1p_120; // |x| < 2^-57
        }
        return PIO2_HI - (x - (PIO2_LO - x * asin_r(x * x)));
    }
    if hx >> 31 != 0 {
        // x < -0.5
        let z = (1.0 + x) * 0.5;
        let s = z.sqrt();
        let w = asin_r(z) * s - PIO2_LO;
        return 2.0 * (PIO2_HI - (s + w));
    }
    // x > 0.5
    let z = (1.0 - x) * 0.5;
    let s = z.sqrt();
    let df = hi_word_only(s);
    let c = (z - df * df) / (s + df);
    let w = asin_r(z) * s + c;
    2.0 * (df + w)
}

// -------------------------------------------------------------------- exp

const LN2HI: f64 = 6.93147180369123816490e-01; // 0x3fe62e42, 0xfee00000
const LN2LO: f64 = 1.90821492927058770002e-10; // 0x3dea39ef, 0x35793c76
const INVLN2: f64 = 1.44269504088896338700e+00; // 0x3ff71547, 0x652b82fe
const P1: f64 = 1.66666666666666019037e-01; // 0x3FC55555, 0x5555553E
const P2: f64 = -2.77777777770155933842e-03; // 0xBF66C16C, 0x16BEBD93
const P3: f64 = 6.61375632143793436117e-05; // 0x3F11566A, 0xAF25DE2C
const P4: f64 = -1.65339022054652515390e-06; // 0xBEBBBD41, 0xC5D26BF1
const P5: f64 = 4.13813679705723846039e-08; // 0x3E663769, 0x72BEA4D0

/// Deterministic `e^x`.
pub fn exp(x: f64) -> f64 {
    let hx_full = (x.to_bits() >> 32) as u32;
    let negative = hx_full >> 31 != 0;
    let hx = hx_full & 0x7fffffff;

    if hx >= 0x4086232b {
        // |x| >= 708.39...
        if x.is_nan() {
            return x;
        }
        if x > 709.782712893383973096 {
            return x * f64::from_bits(0x7fe0000000000000); // overflow -> inf
        }
        if x < -745.13321910194110842 {
            return 0.0;
        }
    }

    // Reduce x = k*ln2 + r, |r| <= 0.5*ln2, r held as hi - lo.
    let (k, hi, lo);
    if hx > 0x3fd62e42 {
        // |x| > 0.5 ln2
        k = if hx >= 0x3ff0a2b2 {
            // |x| >= 1.5 ln2
            (INVLN2 * x + if negative { -0.5 } else { 0.5 }) as i32
        } else if negative {
            -1
        } else {
            1
        };
        hi = x - k as f64 * LN2HI; // k*ln2hi is exact here
        lo = k as f64 * LN2LO;
    } else if hx > 0x3e300000 {
        // |x| > 2^-28
        k = 0;
        hi = x;
        lo = 0.0;
    } else {
        return 1.0 + x;
    }
    let r = hi - lo;
    let rr = r * r;
    let c = r - rr * (P1 + rr * (P2 + rr * (P3 + rr * (P4 + rr * P5))));
    let y = 1.0 + (r * c / (2.0 - c) - lo + hi);
    if k == 0 {
        y
    } else {
        scalbn(y, k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Distance in ulps between two finite doubles of the same sign.
    fn ulps(a: f64, b: f64) -> u64 {
        if a == b {
            return 0;
        }
        let key = |v: f64| {
            let b = v.to_bits() as i64;
            if b < 0 {
                i64::MIN - b
            } else {
                b
            }
        };
        key(a).abs_diff(key(b))
    }

    /// Deterministic sweep of doubles: a fixed LCG over bit patterns
    /// mapped into `[lo, hi]`.
    fn sweep(lo: f64, hi: f64, n: usize) -> Vec<f64> {
        let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = (s >> 11) as f64 / (1u64 << 53) as f64;
                lo + (hi - lo) * u
            })
            .collect()
    }

    // The platform libm is within ~1 ulp of the true value and so are
    // these, so they agree to within 2 ulps on any host.
    #[test]
    fn trig_matches_platform_within_2ulp() {
        let mut xs = sweep(-10.0, 10.0, 20_000);
        xs.extend(sweep(-1e6, 1e6, 5_000)); // medium reduction
        xs.extend(sweep(-1e300, 1e300, 2_000)); // large reduction
        xs.extend([0.0, -0.0, 1e-30, -1e-9, 3e-8, std::f64::consts::PI]);
        for x in xs {
            assert!(ulps(sin(x), x.sin()) <= 2, "sin({:e})", x);
            assert!(ulps(cos(x), x.cos()) <= 2, "cos({:e})", x);
            let (s, c) = sin_cos(x);
            assert_eq!(s.to_bits(), sin(x).to_bits(), "sin_cos.0({:e})", x);
            assert_eq!(c.to_bits(), cos(x).to_bits(), "sin_cos.1({:e})", x);
        }
    }

    #[test]
    fn inverse_trig_matches_platform_within_2ulp() {
        let mut xs = sweep(-1.0, 1.0, 20_000);
        xs.extend([-1.0, 1.0, 0.0, -0.0, 0.5, -0.5, 0.975, 1e-20, 1e-60]);
        for x in xs {
            assert!(ulps(asin(x), x.asin()) <= 2, "asin({:e})", x);
            assert!(ulps(acos(x), x.acos()) <= 2, "acos({:e})", x);
        }
        assert!(asin(1.5).is_nan() && acos(-1.5).is_nan() && acos(f64::NAN).is_nan());
    }

    #[test]
    fn exp_matches_platform_within_2ulp() {
        let mut xs = sweep(-745.0, 709.0, 20_000);
        xs.extend(sweep(-2.0, 2.0, 5_000));
        xs.extend([0.0, 1e-30, -1e-30, 0.3, -0.3, 1.0]);
        for x in xs {
            assert!(ulps(exp(x), x.exp()) <= 2, "exp({:e})", x);
        }
        assert_eq!(exp(-800.0), 0.0);
        assert_eq!(exp(800.0), f64::INFINITY);
        assert!(exp(f64::NAN).is_nan());
    }

    #[test]
    fn special_values() {
        assert_eq!(sin(-0.0).to_bits(), (-0.0f64).to_bits());
        assert_eq!(cos(0.0), 1.0);
        assert!(sin(f64::INFINITY).is_nan() && cos(f64::NAN).is_nan());
        assert_eq!(acos(1.0), 0.0);
        assert_eq!(acos(-1.0), std::f64::consts::PI);
        assert_eq!(asin(1.0), std::f64::consts::FRAC_PI_2);
        assert_eq!(exp(0.0), 1.0);
    }

    /// Guard: non-test core code must not call the platform libm.
    #[test]
    fn core_does_not_call_platform_libm() {
        const BANNED: [&str; 22] = [
            ".sin(",
            ".cos(",
            ".tan(",
            ".sin_cos(",
            ".asin(",
            ".acos(",
            ".atan(",
            ".atan2(",
            ".sinh(",
            ".cosh(",
            ".tanh(",
            ".exp(",
            ".exp2(",
            ".exp_m1(",
            ".ln(",
            ".ln_1p(",
            ".log(",
            ".log2(",
            ".log10(",
            ".powf(",
            ".cbrt(",
            ".hypot(",
        ];
        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files: Vec<_> = std::fs::read_dir(&src)
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| p.extension().is_some_and(|e| e == "rs"))
            .filter(|p| !p.ends_with("detmath.rs"))
            .collect();
        files.sort();
        assert!(files.len() > 5, "no sources found in {:?}", src);
        for path in files {
            let text = std::fs::read_to_string(&path).unwrap();
            let code = text.split("#[cfg(test)]").next().unwrap();
            for (n, line) in code.lines().enumerate() {
                let line = line.split("//").next().unwrap();
                for b in BANNED {
                    let bare = format!("f64::{}", &b[1..]);
                    assert!(
                        !line.contains(b) && !line.contains(&bare),
                        "{}:{} calls platform `{}`; use crate::detmath",
                        path.display(),
                        n + 1,
                        &b[1..b.len() - 1]
                    );
                }
            }
        }
    }

    /// Bit patterns pinned from this implementation. They are the same
    /// on every target by construction; a change here means the
    /// algorithm changed and every CLI golden must be regenerated.
    #[test]
    fn pinned_bits() {
        let got = [
            sin(0.7).to_bits(),
            cos(0.7).to_bits(),
            sin(123.456).to_bits(),
            cos(1e22).to_bits(),
            acos(0.3).to_bits(),
            acos(0.99).to_bits(),
            asin(-0.6).to_bits(),
            exp(-1.25).to_bits(),
        ];
        let want: [u64; 8] = [
            0x3fe49d6e694619b8,
            0x3fe87996529f9d93,
            0xbfe9b9dadc41aeb6,
            0x3fe0be2cef01c8f4,
            0x3ff441f5ecbeef59,
            0x3fc21df72882bfd8,
            0xbfe4978fa3269ee1,
            0x3fd25618372a584f,
        ];
        assert_eq!(got, want, "got {:#018x?}", got);
    }
}
