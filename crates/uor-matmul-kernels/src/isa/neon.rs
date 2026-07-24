//! AArch64 NEON (§7.2).
//!
//! `vmull_s8` gives eight `i8 x i8` products as `i16`. Each is at most
//! `128 * 128 = 16384`, which fits an `i16` --- but *two* of them do not
//! (`3 * 127^2 = 48387 > i16::MAX`), so the products are widened to `i32`
//! immediately rather than accumulated in `i16`. That is not a precaution
//! against a rare case: it is the difference between an exact kernel and one
//! that silently saturates (R3).

use core::arch::aarch64::*;

use uor_matmul_core::Backend;

use crate::spec::KernelSpec;

/// Rows of `C` per call.
pub const MR: usize = 4;
/// Columns of `C` per call.
pub const NR: usize = 8;
/// The `k`-multiple the packing respects.
pub const K_GROUP: usize = 8;

/// The NEON spec.
pub const SPEC: KernelSpec = KernelSpec {
    backend: Backend::Neon,
    mr: MR,
    nr: NR,
    k_group: K_GROUP,
    lane_cap: i32::MAX as u128,
    mac_tile,
};

/// Can this host run it? NEON is mandatory on AArch64, so this is always true;
/// it is a function rather than a constant so that every ISA module has the
/// same shape.
pub fn is_available() -> bool {
    true
}

/// Accumulate a `4 x 8` tile.
///
/// # Safety
///
/// `pa` must have `MR * kc` readable elements, `pb` must have `NR * kc`, and
/// `acc` must have `MR * NR` writable lanes. [`KernelSpec::mac_tile`]
/// establishes all three.
unsafe fn mac_tile(kc: usize, pa: *const i8, pb: *const i8, acc: *mut i32) {
    // SAFETY: the caller forwarded the length guarantees, and NEON is
    // unconditionally present on this target.
    unsafe { neon(kc, pa, pb, acc) }
}

/// # Safety
///
/// As [`mac_tile`].
#[target_feature(enable = "neon")]
unsafe fn neon(kc: usize, pa: *const i8, pb: *const i8, acc: *mut i32) {
    // SAFETY: the caller guaranteed the three extents. One conversion here
    // keeps every panel read below safe.
    let (pa, pb, acc) = unsafe {
        (
            core::slice::from_raw_parts(pa, MR * kc),
            core::slice::from_raw_parts(pb, NR * kc),
            core::slice::from_raw_parts_mut(acc, MR * NR),
        )
    };

    let mut lo = [vdupq_n_s32(0); MR];
    let mut hi = [vdupq_n_s32(0); MR];

    for p in 0..kc {
        // SAFETY: `pb[p * NR .. p * NR + 8]` is in bounds, and `vld1_s8` reads
        // exactly those eight bytes.
        let bv = unsafe { vld1_s8(pb[p * NR..].as_ptr()) };
        for i in 0..MR {
            // Eight exact `i16` products, each at most 16384.
            let prod = vmull_s8(vdup_n_s8(pa[p * MR + i]), bv);
            // Widen to `i32` at once. Nothing is ever summed in `i16`.
            lo[i] = vaddq_s32(lo[i], vmovl_s16(vget_low_s16(prod)));
            hi[i] = vaddq_s32(hi[i], vmovl_s16(vget_high_s16(prod)));
        }
    }

    for i in 0..MR {
        // SAFETY: `acc` has `MR * NR` lanes and `i < MR`, so these two 128-bit
        // stores land inside it.
        unsafe {
            vst1q_s32(acc.as_mut_ptr().add(i * NR), lo[i]);
            vst1q_s32(acc.as_mut_ptr().add(i * NR + 4), hi[i]);
        }
    }
}
