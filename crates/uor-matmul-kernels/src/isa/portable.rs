//! The reference kernels (§7.2, row `Scalar`).
//!
//! The model transcribed: widen, multiply, add. No `unsafe` beyond the raw
//! reads the shared signature requires, runs under Miri, and never optimized
//! (R6).
//!
//! These are not fallbacks. Every other kernel in this crate is a factorization
//! of *these* accumulations into wider instructions, and `CB-01` pins the `i8`
//! one to [`uor_matmul_core::dot_ref`] so the whole chain is anchored to the
//! reference the plan names.

use uor_matmul_core::Backend;

use crate::spec::{Factorization, KernelSpec};

/// Build one reference kernel: `mr x nr` of `E`, accumulating in `L`.
///
/// The body is the same three lines at every instantiation, which is the point:
/// the families differ in their widths and in nothing else.
macro_rules! reference_kernel {
    (
        $(#[$meta:meta])*
        $name:ident, $fnname:ident, $e:ty, $l:ty, $mr:expr, $nr:expr,
        $factorization:expr, $cap:expr, $mul:expr
    ) => {
        $crate::tile_fits!($mr, $nr);

        $(#[$meta])*
        pub const $name: KernelSpec<$e, $l> = KernelSpec {
            backend: Backend::Portable,
            factorization: $factorization,
            mr: $mr,
            nr: $nr,
            // The reference needs no grouping at all, which is why it is the
            // one kernel that never has a tail.
            k_group: 1,
            lane_cap: $cap,
            // The reference multiplies in the lane's own width, so there is
            // no intermediate to outgrow and no alphabet it is inexact on.
            max_bound: u128::MAX,
            mac_tile: $fnname,
        };

        /// # Safety
        ///
        /// `pa` must have `mr * kc` readable elements, `pb` must have
        /// `nr * kc`, and `acc` must have `mr * nr` writable lanes.
        unsafe fn $fnname(kc: usize, pa: *const $e, pb: *const $e, acc: *mut $l) {
            // SAFETY: the caller guaranteed the three extents. Turning them
            // into slices once, here, is what lets the loop below be safe
            // indexing rather than three raw reads per product.
            let (pa, pb, acc) = unsafe {
                (
                    core::slice::from_raw_parts(pa, $mr * kc),
                    core::slice::from_raw_parts(pb, $nr * kc),
                    core::slice::from_raw_parts_mut(acc, $mr * $nr),
                )
            };
            let mut tile = [<$l>::default(); $mr * $nr];
            for p in 0..kc {
                for i in 0..$mr {
                    let a = pa[p * $mr + i];
                    for j in 0..$nr {
                        let m: $l = $mul(a, pb[p * $nr + j]);
                        // The driver only offers a chunk whose depth this
                        // lane admits, so the running sum stays inside it.
                        // Spelled `wrapping_add` because R5 asks the overflow
                        // behaviour to be written rather than inherited from
                        // the build profile --- and because for a modular lane
                        // the wrap is the answer.
                        tile[i * $nr + j] = tile[i * $nr + j].wrapping_add(m);
                    }
                }
            }
            acc.copy_from_slice(&tile);
        }
    };
}

reference_kernel!(
    /// `i8 x i8 -> i32`, exact. The reference `CB-01` pins to `dot_ref`.
    I8_I32, mac_i8_i32, i8, i32, 4, 4, Factorization::Exact, i32::MAX as u128,
    |a: i8, b: i8| i32::from(a) * i32::from(b)
);

reference_kernel!(
    /// `i16 x i16 -> i64`, exact.
    I16_I64, mac_i16_i64, i16, i64, 4, 4, Factorization::Exact, i64::MAX as u128,
    |a: i16, b: i16| i64::from(a) * i64::from(b)
);

reference_kernel!(
    /// `i32 x i32 -> i64`, exact. Two full-range products fill the lane, which
    /// is why a declared narrower bound buys so much here.
    I32_I64, mac_i32_i64, i32, i64, 4, 4, Factorization::Exact, i64::MAX as u128,
    |a: i32, b: i32| i64::from(a) * i64::from(b)
);

reference_kernel!(
    /// `i32 x i32 -> i32` in `Z/2^32`. Exact in the quotient the caller asked
    /// to encode into, and therefore unbounded in depth.
    I32_MOD, mac_i32_mod, i32, i32, 4, 4, Factorization::Modular, 0,
    |a: i32, b: i32| a.wrapping_mul(b)
);

reference_kernel!(
    /// `i64 x i64 -> i64` in `Z/2^64`. The only factorization there is for
    /// `i64`: an exact product needs 128 bits, and the quotient needs the low
    /// 64, which is what `wrapping_mul` gives.
    I64_MOD, mac_i64_mod, i64, i64, 4, 4, Factorization::Modular, 0,
    |a: i64, b: i64| a.wrapping_mul(b)
);

reference_kernel!(
    /// `i64 x i64 -> i128`, exact. The lane is the only width that holds the
    /// product, which is why no SIMD reaches it on any supported target.
    I64_I128, mac_i64_i128, i64, i128, 4, 4, Factorization::Exact, i128::MAX as u128,
    |a: i64, b: i64| i128::from(a) * i128::from(b)
);

reference_kernel!(
    /// `i16 x i16 -> i32` in `Z/2^32`.
    I16_MOD, mac_i16_mod, i16, i32, 4, 4, Factorization::Modular, 0,
    |a: i16, b: i16| i32::from(a).wrapping_mul(i32::from(b))
);
