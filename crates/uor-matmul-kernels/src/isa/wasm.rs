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
    // `i32x4_dot_i16x8` is `madd`: the pair sum is `2 * bound^2`, and an `i8`
    // alphabet cannot reach the bound where that leaves an `i32`.
    max_bound: 32767,
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

    for q in 0..kc / 2 {
        // The panel is packed in `k`-pairs, so the sixteen bytes of `B` for this
        // pair widen straight into the two `i16x8` vectors `dot` consumes: the
        // low half is columns 0..3 and the high half columns 4..7, each lane
        // holding the pair `(b[p0][j], b[p1][j])`.
        //
        // SAFETY: `pb[q * NR * 2 ..][..16]` is in bounds: one v128 load.
        let raw = unsafe { v128_load(pb.as_ptr().add(q * NR * 2).cast()) };
        let bv0 = i16x8_extend_low_i8x16(raw);
        let bv1 = i16x8_extend_high_i8x16(raw);
        for i in 0..MR {
            // Splatting the pair as a halfword and sign-extending it puts
            // `(a1 << 16) | a0` in every 32-bit lane.
            //
            // SAFETY: `q * MR * 2 + i * 2 + 1 < MR * kc`.
            let av = unsafe {
                i16x8_extend_low_i8x16(v128_load16_splat(
                    pa.as_ptr().add(q * MR * 2 + i * 2).cast(),
                ))
            };
            lo[i] = i32x4_add(lo[i], i32x4_dot_i16x8(av, bv0));
            hi[i] = i32x4_add(hi[i], i32x4_dot_i16x8(av, bv1));
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
