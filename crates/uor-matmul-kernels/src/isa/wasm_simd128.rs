//! WebAssembly SIMD128 (§7.2).
//!
//! `i16x8_extend_low/high_i8x16` widens, `i32x4_dot_i16x8` pairs two `k`-steps
//! into an `i32` lane, and `i32x4_add` accumulates. The same factorization as
//! AVX2 at a quarter of the width, which is the point of having a
//! [`KernelSpec`] rather than a per-ISA driver.

use core::arch::wasm32::*;

use uor_matmul_core::Backend;

use crate::spec::KernelSpec;

/// Rows of `C` per call.
pub const MR: usize = 4;
/// Columns of `C` per call.
pub const NR: usize = 8;
/// `i32x4_dot_i16x8` consumes two `k`-steps at a time.
pub const K_GROUP: usize = 2;

/// The wasm SIMD128 spec.
pub const SPEC: KernelSpec = KernelSpec {
    backend: Backend::WasmSimd128,
    mr: MR,
    nr: NR,
    k_group: K_GROUP,
    lane_cap: i32::MAX as u128,
    mac_tile,
};

/// Can this build run it?
///
/// SIMD128 is a compile-time target feature on wasm; there is nothing to detect
/// at runtime, and `CB-05` asserts that a SIMD128-off build agrees with a
/// SIMD128-on one.
pub fn is_available() -> bool {
    cfg!(target_feature = "simd128")
}

/// Accumulate a `4 x 8` tile.
///
/// # Safety
///
/// `pa` must have `MR * kc` readable elements, `pb` must have `NR * kc`, and
/// `acc` must have `MR * NR` writable lanes.
unsafe fn mac_tile(kc: usize, pa: *const i8, pb: *const i8, acc: *mut i32) {
    // SAFETY: the caller guaranteed the three extents. One conversion here
    // keeps every panel read below safe.
    let (pa, pb, acc) = unsafe {
        (
            core::slice::from_raw_parts(pa, MR * kc),
            core::slice::from_raw_parts(pb, NR * kc),
            core::slice::from_raw_parts_mut(acc, MR * NR),
        )
    };

    let mut lo = [i32x4_splat(0); MR];
    let mut hi = [i32x4_splat(0); MR];

    let pairs = kc / K_GROUP;
    for q in 0..pairs {
        let (p0, p1) = (q * K_GROUP, q * K_GROUP + 1);
        // Interleave the two `k`-steps of B so `dot` sees adjacent pairs.
        let mut b_pairs = [0i16; NR * 2];
        for j in 0..NR {
            b_pairs[j * 2] = pb[p0 * NR + j] as i16;
            b_pairs[j * 2 + 1] = pb[p1 * NR + j] as i16;
        }
        // SAFETY: `b_pairs` holds `NR * 2 = 16` i16 = two v128 loads.
        let (bv0, bv1) = unsafe {
            (
                v128_load(b_pairs.as_ptr().cast()),
                v128_load(b_pairs.as_ptr().add(8).cast()),
            )
        };

        for i in 0..MR {
            let a0 = pa[p0 * MR + i] as i16;
            let a1 = pa[p1 * MR + i] as i16;
            let av = i32x4_splat(((a1 as u16 as u32) << 16 | (a0 as u16 as u32)) as i32);
            lo[i] = i32x4_add(lo[i], i32x4_dot_i16x8(av, bv0));
            hi[i] = i32x4_add(hi[i], i32x4_dot_i16x8(av, bv1));
        }
    }

    // The `k`-tail, one step at a time.
    for p in (pairs * K_GROUP)..kc {
        for i in 0..MR {
            let a = pa[p * MR + i] as i32;
            let mut lane = [0i32; NR];
            for (j, slot) in lane.iter_mut().enumerate() {
                *slot = a.wrapping_mul(pb[p * NR + j] as i32);
            }
            // SAFETY: `lane` holds `NR = 8` i32 = two v128 loads.
            let (l0, l1) = unsafe {
                (
                    v128_load(lane.as_ptr().cast()),
                    v128_load(lane.as_ptr().add(4).cast()),
                )
            };
            lo[i] = i32x4_add(lo[i], l0);
            hi[i] = i32x4_add(hi[i], l1);
        }
    }

    for i in 0..MR {
        // SAFETY: `acc` has `MR * NR` lanes and `i < MR`.
        unsafe {
            v128_store(acc.as_mut_ptr().add(i * NR).cast(), lo[i]);
            v128_store(acc.as_mut_ptr().add(i * NR + 4).cast(), hi[i]);
        }
    }
}
