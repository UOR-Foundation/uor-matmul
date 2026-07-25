//! WebAssembly SIMD128 (§7.2).
//!
//! `i32x4_dot_i16x8` pairs two `k`-steps into an `i32` lane, and `i32x4_add`
//! accumulates. The same factorization as AVX2 at a quarter of the width,
//! which is the point of having a [`KernelSpec`] rather than a per-ISA driver.

use core::arch::wasm32::*;

use uor_matmul_core::Backend;

use crate::spec::{Factorization, KernelSpec};

crate::tile_fits!(4, 8);

/// Is SIMD128 available?
///
/// A compile-time target feature on wasm; there is nothing to detect at
/// runtime, and `CB-05` asserts that a SIMD128-off build agrees with a
/// SIMD128-on one.
pub fn simd128_available() -> bool {
    cfg!(target_feature = "simd128")
}

const MR: usize = 4;
const NR: usize = 8;

/// The wasm SIMD128 `i8` spec.
pub const SIMD128_I8_I32: KernelSpec<i8, i32> = KernelSpec {
    backend: Backend::WasmSimd128,
    factorization: Factorization::Exact,
    mr: MR,
    nr: NR,
    k_group: 2,
    lane_cap: i32::MAX as u128,
    mac_tile: simd128_i8,
};

/// # Safety
///
/// `pa` must have `4 * kc` readable elements, `pb` `8 * kc`, and `acc` 32
/// writable lanes.
unsafe fn simd128_i8(kc: usize, pa: *const i8, pb: *const i8, acc: *mut i32) {
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

    let pairs = kc / 2;
    for q in 0..pairs {
        let (p0, p1) = (q * 2, q * 2 + 1);
        let mut b_pairs = [0i16; NR * 2];
        for j in 0..NR {
            b_pairs[j * 2] = i16::from(pb[p0 * NR + j]);
            b_pairs[j * 2 + 1] = i16::from(pb[p1 * NR + j]);
        }
        // SAFETY: `b_pairs` holds 16 i16 = two v128 loads.
        let (bv0, bv1) = unsafe {
            (
                v128_load(b_pairs.as_ptr().cast()),
                v128_load(b_pairs.as_ptr().add(8).cast()),
            )
        };
        for i in 0..MR {
            let a0 = i16::from(pa[p0 * MR + i]);
            let a1 = i16::from(pa[p1 * MR + i]);
            let av = i32x4_splat(((a1 as u16 as u32) << 16 | (a0 as u16 as u32)) as i32);
            lo[i] = i32x4_add(lo[i], i32x4_dot_i16x8(av, bv0));
            hi[i] = i32x4_add(hi[i], i32x4_dot_i16x8(av, bv1));
        }
    }

    for p in (pairs * 2)..kc {
        for i in 0..MR {
            let a = i32::from(pa[p * MR + i]);
            let mut lane = [0i32; NR];
            for (j, slot) in lane.iter_mut().enumerate() {
                *slot = a.wrapping_mul(i32::from(pb[p * NR + j]));
            }
            // SAFETY: `lane` holds 8 i32 = two v128 loads.
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
        // SAFETY: `i < MR`, so these two stores land inside `MR * NR`.
        unsafe {
            v128_store(acc.as_mut_ptr().add(i * NR).cast(), lo[i]);
            v128_store(acc.as_mut_ptr().add(i * NR + 4).cast(), hi[i]);
        }
    }
}
