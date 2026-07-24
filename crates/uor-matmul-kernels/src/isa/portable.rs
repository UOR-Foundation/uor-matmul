//! The reference kernel (§7.2, row `Scalar`).
//!
//! This is the model transcribed: widen, multiply, add, in an `i32` lane. It
//! contains no `unsafe` beyond the raw-pointer reads its shared signature
//! requires, it runs under Miri, and it is never optimized (R6).
//!
//! It is not a fallback. Every other kernel in this crate is a factorization of
//! *this* accumulation into wider instructions, and `CB-01` pins it to
//! [`uor_matmul_core::dot_ref`] so that the whole chain is anchored to the one
//! reference the plan names.

use uor_matmul_core::Backend;

use crate::spec::KernelSpec;

/// Rows of `C` per call.
pub const MR: usize = 4;
/// Columns of `C` per call.
pub const NR: usize = 4;
/// The `k`-multiple the packing must respect. One: this kernel needs no
/// grouping at all, which is why it is the one that never has a tail.
pub const K_GROUP: usize = 1;

/// The reference kernel's spec.
pub const SPEC: KernelSpec = KernelSpec {
    backend: Backend::Portable,
    mr: MR,
    nr: NR,
    k_group: K_GROUP,
    lane_cap: i32::MAX as u128,
    mac_tile,
};

/// Accumulate a `4 x 4` tile.
///
/// # Safety
///
/// `pa` must have `MR * kc` readable elements, `pb` must have `NR * kc`, and
/// `acc` must have `MR * NR` writable lanes. [`KernelSpec::mac_tile`]
/// establishes all three before calling.
unsafe fn mac_tile(kc: usize, pa: *const i8, pb: *const i8, acc: *mut i32) {
    // SAFETY: the caller guaranteed `MR * kc` readable elements at `pa`,
    // `NR * kc` at `pb`, and `MR * NR` writable lanes at `acc`. Turning them
    // into slices once, here, is what lets the whole loop below be safe
    // indexing rather than forty separate raw reads.
    let (pa, pb, acc) = unsafe {
        (
            core::slice::from_raw_parts(pa, MR * kc),
            core::slice::from_raw_parts(pb, NR * kc),
            core::slice::from_raw_parts_mut(acc, MR * NR),
        )
    };

    let mut tile = [0i32; MR * NR];
    for p in 0..kc {
        for i in 0..MR {
            let a = pa[p * MR + i] as i32;
            for j in 0..NR {
                let b = pb[p * NR + j] as i32;
                // Exact: `|a * b| <= 128 * 128`, and the driver only offers a
                // tile whose depth `narrow_cap_for` admitted for this lane, so
                // the running sum stays inside `i32` (§5.1). Written with
                // `wrapping_*` because R5 asks the overflow behaviour to be
                // written rather than inherited from the build profile --- not
                // because a wrap can occur.
                tile[i * NR + j] = tile[i * NR + j].wrapping_add(a.wrapping_mul(b));
            }
        }
    }
    acc.copy_from_slice(&tile);
}
