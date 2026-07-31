//! AArch64 NEON, with and without the dot-product extension (§7.2).

use core::arch::aarch64::*;

use uor_matmul_core::Backend;

use crate::spec::{Factorization, KernelSpec, LaneLayout};
use crate::table::TableSpec;

crate::tile_fits!(4, 8);
crate::tile_fits!(8, 12);
crate::tile_fits!(1, 8);

/// Is NEON available? Mandatory on AArch64, so always.
pub fn neon_available() -> bool {
    true
}

/// Is the ARMv8.2-A dot-product extension available?
pub fn dotprod_available() -> bool {
    #[cfg(any(feature = "std", test))]
    {
        std::arch::is_aarch64_feature_detected!("dotprod")
    }
    #[cfg(not(any(feature = "std", test)))]
    {
        cfg!(target_feature = "dotprod")
    }
}

/// `i8 x i8 -> i32` through `vmull_s8`.
///
/// `vmull_s8` gives eight `i8 x i8` products as `i16`. Each is at most
/// `128 * 128 = 16384`, which fits --- but *two* do not (`3 * 127^2 = 48387 >
/// i16::MAX`), so the products are widened to `i32` immediately rather than
/// accumulated in `i16`. That is not a precaution against a rare case: it is
/// the difference between an exact kernel and one that silently saturates (R3).
pub const NEON_I8_I32: KernelSpec<i8, i32> = KernelSpec {
    backend: Backend::Neon,
    factorization: Factorization::Exact,
    mr: 4,
    nr: 8,
    lane_layout: LaneLayout::Interleaved,
    // `vmull_s8` widens across the *columns* of one `k`-step, not across `k`, so
    // this kernel consumes one step at a time and wants the plain `k`-major
    // panel. Its eight is the vector's width in columns, which is `nr`.
    k_group: 1,
    products_per_step: 8,
    lane_cap: i32::MAX as u128,
    // `vmull_s8` widens each product to `i16` and this kernel widens again to
    // `i32` before accumulating, so nothing narrower than the lane is held.
    max_bound: u128::MAX,
    mac_tile: neon_i8,
};

/// `i8 x i8 -> i32` through `sdot`.
///
/// Four `k`-steps per lane in one instruction, accumulating straight into
/// `i32`, so there is no intermediate width to overflow and no compensation
/// term. The widest reach of any sequence in this crate.
pub const NEON_DOTPROD_I8_I32: KernelSpec<i8, i32> = KernelSpec {
    backend: Backend::NeonDotprod,
    factorization: Factorization::Exact,
    mr: 8,
    nr: 12,
    lane_layout: LaneLayout::Interleaved,
    k_group: 4,
    products_per_step: 16,
    lane_cap: i32::MAX as u128,
    max_bound: u128::MAX,
    mac_tile: neon_dotprod_i8,
};

const NEON_MR: usize = 4;
const NEON_NR: usize = 8;

/// # Safety
///
/// `pa` must have `4 * kc` readable elements, `pb` `8 * kc`, `acc` 32 writable
/// lanes.
unsafe fn neon_i8(kc: usize, pa: *const i8, pb: *const i8, acc: *mut i32) {
    // SAFETY: the caller forwarded the length guarantees, and NEON is
    // unconditionally present on this target.
    unsafe { neon_i8_inner(kc, pa, pb, acc) }
}

/// # Safety
///
/// As [`neon_i8`].
#[target_feature(enable = "neon")]
unsafe fn neon_i8_inner(kc: usize, pa: *const i8, pb: *const i8, acc: *mut i32) {
    // SAFETY: the caller guaranteed the three extents.
    let (pa, pb, acc) = unsafe {
        (
            core::slice::from_raw_parts(pa, NEON_MR * kc),
            core::slice::from_raw_parts(pb, NEON_NR * kc),
            core::slice::from_raw_parts_mut(acc, NEON_MR * NEON_NR),
        )
    };
    let mut lo = [vdupq_n_s32(0); NEON_MR];
    let mut hi = [vdupq_n_s32(0); NEON_MR];

    for p in 0..kc {
        // SAFETY: `pb[p * NR ..][..8]` is in bounds.
        let bv = unsafe { vld1_s8(pb[p * NEON_NR..].as_ptr()) };
        for i in 0..NEON_MR {
            let prod = vmull_s8(vdup_n_s8(pa[p * NEON_MR + i]), bv);
            lo[i] = vaddq_s32(lo[i], vmovl_s16(vget_low_s16(prod)));
            hi[i] = vaddq_s32(hi[i], vmovl_s16(vget_high_s16(prod)));
        }
    }
    for i in 0..NEON_MR {
        // SAFETY: `i < MR`, so these two stores land inside `MR * NR`.
        unsafe {
            vst1q_s32(acc.as_mut_ptr().add(i * NEON_NR), lo[i]);
            vst1q_s32(acc.as_mut_ptr().add(i * NEON_NR + 4), hi[i]);
        }
    }
}

const DOT_MR: usize = 8;
const DOT_NR: usize = 12;

/// `sdot Vd.4S, Vn.16B, Vm.16B`.
///
/// Inline assembly because `vdotq_s32` is still unstable in `core::arch`. The
/// instruction is the one the intrinsic emits, and reaching it through `asm!`
/// rather than a nightly feature keeps the crate on the pinned stable
/// toolchain --- which matters more than the spelling, because a backend that
/// cannot be built cannot be validated.
///
/// # Safety
///
/// The host must have the `dotprod` extension.
#[target_feature(enable = "neon,dotprod")]
#[inline]
unsafe fn sdot(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    let mut out = acc;
    // SAFETY: `sdot` reads two vector registers and accumulates into a third.
    // It touches no memory, sets no flags, and its result depends only on its
    // inputs, which is what the options below declare.
    unsafe {
        core::arch::asm!(
            "sdot {out:v}.4s, {a:v}.16b, {b:v}.16b",
            out = inout(vreg) out,
            a = in(vreg) a,
            b = in(vreg) b,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    out
}

/// # Safety
///
/// As [`neon_i8`], with `8 * kc`, `12 * kc`, 96 lanes, and the host must have
/// `dotprod`.
unsafe fn neon_dotprod_i8(kc: usize, pa: *const i8, pb: *const i8, acc: *mut i32) {
    // SAFETY: the caller established the lengths, and `available_i8` established
    // the `dotprod` feature before returning this spec.
    unsafe { neon_dotprod_inner(kc, pa, pb, acc) }
}

/// # Safety
///
/// As [`neon_dotprod_i8`].
#[target_feature(enable = "neon,dotprod")]
unsafe fn neon_dotprod_inner(kc: usize, pa: *const i8, pb: *const i8, acc: *mut i32) {
    const QUADS: usize = DOT_NR / 4;
    const G: usize = 4;
    // SAFETY: the caller guaranteed the three extents.
    let (pa, pb, acc) = unsafe {
        (
            core::slice::from_raw_parts(pa, DOT_MR * kc),
            core::slice::from_raw_parts(pb, DOT_NR * kc),
            core::slice::from_raw_parts_mut(acc, DOT_MR * DOT_NR),
        )
    };
    let mut tile = [[vdupq_n_s32(0); QUADS]; DOT_MR];

    for q in 0..kc / G {
        // The panel is packed in `k`-quads, so lane `4c + g` of a sixteen-byte
        // load already holds `b[4q + g][4 * quad + c]`, which is exactly the
        // layout `sdot` reads --- no transpose, one load per quad of columns.
        let bv: [int8x16_t; QUADS] = core::array::from_fn(|quad| {
            // SAFETY: `q * DOT_NR * G + quad * 16 + 16 <= DOT_NR * kc`.
            unsafe { vld1q_s8(pb.as_ptr().add(q * DOT_NR * G + quad * 16)) }
        });

        for (i, row) in tile.iter_mut().enumerate() {
            // `A`'s four `k`-steps for this row are four contiguous bytes.
            //
            // SAFETY: `q * DOT_MR * G + i * G + 3 < DOT_MR * kc`, and
            // `read_unaligned` waives `i32`'s alignment.
            let quad = unsafe {
                pa.as_ptr()
                    .add(q * DOT_MR * G + i * G)
                    .cast::<i32>()
                    .read_unaligned()
            };
            let av = vreinterpretq_s8_s32(vdupq_n_s32(quad));
            for (quad, lane) in row.iter_mut().enumerate() {
                // SAFETY: `dotprod` is enabled on this function.
                *lane = unsafe { sdot(*lane, av, bv[quad]) };
            }
        }
    }

    for (i, row) in tile.iter().enumerate() {
        for (quad, lane) in row.iter().enumerate() {
            // SAFETY: `i < MR` and `quad * 4 + 4 <= NR`, inside `MR * NR`.
            unsafe { vst1q_s32(acc.as_mut_ptr().add(i * DOT_NR + quad * 4), *lane) };
        }
    }
}

// ---------------------------------------------------------------------------
// i32 x i32 -> i64, exact
// ---------------------------------------------------------------------------

/// `i32 x i32 -> i64` through `vmlal_s32`.
///
/// The product of two `i32` needs 62 bits, so the lane is 64 and the
/// instruction is the signed `32 x 32 -> 64` widening multiply-accumulate ---
/// this family's whole arithmetic in one instruction, two columns at a time.
/// Unlike `_mm256_mul_epi32` the pairs it covers are adjacent, so the
/// accumulators store back in column order with no deinterleave.
pub const NEON_I32_I64: KernelSpec<i32, i64> = KernelSpec {
    backend: Backend::Neon,
    factorization: Factorization::Exact,
    mr: NEON_MR,
    nr: NEON_NR,
    lane_layout: LaneLayout::Interleaved,
    // `vmlal_s32` widens across the *columns* of one `k`-step, not across `k`,
    // so this kernel consumes one step at a time, as `NEON_I8_I32` does.
    k_group: 1,
    products_per_step: 2,
    lane_cap: i64::MAX as u128,
    // Each product is computed at its own full width, so every alphabet.
    max_bound: u128::MAX,
    mac_tile: neon_i32_exact,
};

/// The same sequence at a one-row panel.
///
/// A tile panel taller than the output is zero-padded, and the kernel does that
/// padding's arithmetic: at `m = 1` a four-row panel is four times the work the
/// product needs. So the table offers the heights the shapes fill, and the
/// driver takes the tallest one the rows do. Same instructions, same answer
/// (`CB-06`).
pub const NEON_I32_I64_M1: KernelSpec<i32, i64> = KernelSpec {
    backend: Backend::Neon,
    factorization: Factorization::Exact,
    mr: 1,
    nr: NEON_NR,
    lane_layout: LaneLayout::Interleaved,
    k_group: 1,
    products_per_step: 2,
    lane_cap: i64::MAX as u128,
    // Each product is computed at its own full width, so every alphabet.
    max_bound: u128::MAX,
    mac_tile: neon_i32_exact_one,
};

/// # Safety
///
/// `pa` must have `4 * kc` readable elements, `pb` `8 * kc`, `acc` 32 writable
/// lanes.
unsafe fn neon_i32_exact(kc: usize, pa: *const i32, pb: *const i32, acc: *mut i64) {
    // SAFETY: the caller forwarded the length guarantees, and NEON is
    // unconditionally present on this target.
    unsafe { neon_i32_exact_inner::<NEON_MR>(kc, pa, pb, acc) }
}

/// # Safety
///
/// As [`neon_i32_exact`], with a one-row panel.
unsafe fn neon_i32_exact_one(kc: usize, pa: *const i32, pb: *const i32, acc: *mut i64) {
    // SAFETY: the caller forwarded the length guarantees, and NEON is
    // unconditionally present on this target.
    unsafe { neon_i32_exact_inner::<1>(kc, pa, pb, acc) }
}

/// # Safety
///
/// As [`neon_i32_exact`].
#[target_feature(enable = "neon")]
unsafe fn neon_i32_exact_inner<const MR: usize>(
    kc: usize,
    pa: *const i32,
    pb: *const i32,
    acc: *mut i64,
) {
    const NR: usize = NEON_NR;
    // SAFETY: the caller guaranteed the three extents.
    let (pa, pb, acc) = unsafe {
        (
            core::slice::from_raw_parts(pa, MR * kc),
            core::slice::from_raw_parts(pb, NR * kc),
            core::slice::from_raw_parts_mut(acc, MR * NR),
        )
    };
    let mut tile = [[vdupq_n_s64(0); NR / 2]; MR];

    for p in 0..kc {
        // SAFETY: `pb[p * NR ..][..8]` is in bounds.
        let bv = unsafe {
            [
                vld1_s32(pb[p * NR..].as_ptr()),
                vld1_s32(pb[p * NR + 2..].as_ptr()),
                vld1_s32(pb[p * NR + 4..].as_ptr()),
                vld1_s32(pb[p * NR + 6..].as_ptr()),
            ]
        };
        for i in 0..MR {
            let av = vdup_n_s32(pa[p * MR + i]);
            for (pair, lane) in tile[i].iter_mut().enumerate() {
                *lane = vmlal_s32(*lane, av, bv[pair]);
            }
        }
    }

    for (i, row) in tile.iter().enumerate() {
        for (pair, lane) in row.iter().enumerate() {
            // SAFETY: `i < MR` and `pair * 2 + 2 <= NR`, inside `MR * NR`.
            unsafe { vst1q_s64(acc.as_mut_ptr().add(i * NR + pair * 2), *lane) };
        }
    }
}

// ---------------------------------------------------------------------------
// The reduce factorization: vector lanes on `k` rather than on the output
// ---------------------------------------------------------------------------

const R_MR: usize = 4;

crate::tile_fits!(R_MR, 1);
crate::tile_fits!(1, 1);

/// NEON `i8`, reducing four rows against one column with the lanes on `k`.
///
/// Sixteen `k`-steps per iteration. `vmull_s8` widens eight products to `i16` and
/// they are widened again to `i32` before accumulating, so nothing narrower than
/// the lane is held and the sequence is exact at every `i8`.
pub const NEON_R_I8_I32: KernelSpec<i8, i32> = KernelSpec {
    backend: Backend::Neon,
    factorization: Factorization::Exact,
    mr: R_MR,
    nr: 1,
    lane_layout: LaneLayout::Contiguous,
    k_group: 16,
    products_per_step: 8,
    lane_cap: i32::MAX as u128,
    max_bound: u128::MAX,
    mac_tile: neon_r_i8,
};

/// # Safety
///
/// `pa` must have `4 * kc` readable elements with row `i` at `pa[i * kc ..][..kc]`,
/// `pb` must have `kc`, `acc` 4 writable lanes, and `kc` a multiple of 16.
unsafe fn neon_r_i8(kc: usize, pa: *const i8, pb: *const i8, acc: *mut i32) {
    // SAFETY: the caller forwarded the lengths, and NEON is unconditionally
    // present on this target.
    unsafe { neon_r_i8_inner::<R_MR>(kc, pa, pb, acc) }
}

/// # Safety
///
/// As [`neon_r_i8`].
#[target_feature(enable = "neon")]
unsafe fn neon_r_i8_inner<const MR: usize>(kc: usize, pa: *const i8, pb: *const i8, acc: *mut i32) {
    // SAFETY: the caller guaranteed the three extents.
    let (pa, pb, acc) = unsafe {
        (
            core::slice::from_raw_parts(pa, MR * kc),
            core::slice::from_raw_parts(pb, kc),
            core::slice::from_raw_parts_mut(acc, MR),
        )
    };
    let mut sums = [vdupq_n_s32(0); MR];

    for q in 0..kc / 16 {
        // SAFETY: `pb[q * 16 ..][..16]` is in bounds: two eight-byte loads.
        let (bl, bh) = unsafe {
            let base = pb.as_ptr().add(q * 16);
            (vld1_s8(base), vld1_s8(base.add(8)))
        };
        for (i, sum) in sums.iter_mut().enumerate() {
            // SAFETY: `i * kc + q * 16 + 16 <= MR * kc`.
            let (al, ah) = unsafe {
                let base = pa.as_ptr().add(i * kc + q * 16);
                (vld1_s8(base), vld1_s8(base.add(8)))
            };
            for (av, bv) in [(al, bl), (ah, bh)] {
                let prod = vmull_s8(av, bv);
                *sum = vaddq_s32(*sum, vmovl_s16(vget_low_s16(prod)));
                *sum = vaddq_s32(*sum, vmovl_s16(vget_high_s16(prod)));
            }
        }
    }

    for (i, sum) in sums.iter().enumerate() {
        acc[i] = vaddvq_s32(*sum);
    }
}

/// NEON `i8` through `sdot`, reducing four rows against one column.
///
/// `sdot` accumulates four `i8` products straight into an `i32` lane, so there is
/// no intermediate width at all: sixteen `k`-steps per instruction per row.
pub const NEON_DOTPROD_R_I8_I32: KernelSpec<i8, i32> = KernelSpec {
    backend: Backend::NeonDotprod,
    factorization: Factorization::Exact,
    mr: R_MR,
    nr: 1,
    lane_layout: LaneLayout::Contiguous,
    k_group: 16,
    products_per_step: 16,
    lane_cap: i32::MAX as u128,
    max_bound: u128::MAX,
    mac_tile: neon_dotprod_r_i8,
};

/// # Safety
///
/// As [`neon_r_i8`], and the host must have `dotprod`.
unsafe fn neon_dotprod_r_i8(kc: usize, pa: *const i8, pb: *const i8, acc: *mut i32) {
    // SAFETY: the caller established the lengths, and `available_reduce_i8`
    // established the `dotprod` feature before returning this spec.
    unsafe { neon_dotprod_r_i8_inner::<R_MR>(kc, pa, pb, acc) }
}

/// # Safety
///
/// As [`neon_dotprod_r_i8`].
#[target_feature(enable = "neon,dotprod")]
unsafe fn neon_dotprod_r_i8_inner<const MR: usize>(
    kc: usize,
    pa: *const i8,
    pb: *const i8,
    acc: *mut i32,
) {
    // SAFETY: the caller guaranteed the three extents.
    let (pa, pb, acc) = unsafe {
        (
            core::slice::from_raw_parts(pa, MR * kc),
            core::slice::from_raw_parts(pb, kc),
            core::slice::from_raw_parts_mut(acc, MR),
        )
    };
    let mut sums = [vdupq_n_s32(0); MR];

    for q in 0..kc / 16 {
        // SAFETY: `pb[q * 16 ..][..16]` is in bounds: one sixteen-byte load.
        let bv = unsafe { vld1q_s8(pb.as_ptr().add(q * 16)) };
        for (i, sum) in sums.iter_mut().enumerate() {
            // SAFETY: `i * kc + q * 16 + 16 <= MR * kc`.
            let av = unsafe { vld1q_s8(pa.as_ptr().add(i * kc + q * 16)) };
            // SAFETY: `dotprod` is enabled on this function.
            *sum = unsafe { sdot(*sum, av, bv) };
        }
    }

    for (i, sum) in sums.iter().enumerate() {
        acc[i] = vaddvq_s32(*sum);
    }
}

/// The same sequence at a one-row panel. See [`NEON_R_I8_I32`].
pub const NEON_R_I8_I32_1: KernelSpec<i8, i32> = KernelSpec {
    backend: Backend::Neon,
    factorization: Factorization::Exact,
    mr: 1,
    nr: 1,
    lane_layout: LaneLayout::Contiguous,
    k_group: 16,
    products_per_step: 8,
    lane_cap: i32::MAX as u128,
    max_bound: u128::MAX,
    mac_tile: neon_r_i8_one,
};

/// # Safety
///
/// As [`neon_r_i8`], with a one-row panel.
unsafe fn neon_r_i8_one(kc: usize, pa: *const i8, pb: *const i8, acc: *mut i32) {
    // SAFETY: the caller established the lengths and the target features.
    unsafe { neon_r_i8_inner::<1>(kc, pa, pb, acc) }
}

/// The same sequence at a one-row panel. See [`NEON_DOTPROD_R_I8_I32`].
pub const NEON_DOTPROD_R_I8_I32_1: KernelSpec<i8, i32> = KernelSpec {
    backend: Backend::NeonDotprod,
    factorization: Factorization::Exact,
    mr: 1,
    nr: 1,
    lane_layout: LaneLayout::Contiguous,
    k_group: 16,
    products_per_step: 16,
    lane_cap: i32::MAX as u128,
    max_bound: u128::MAX,
    mac_tile: neon_dotprod_r_i8_one,
};

/// # Safety
///
/// As [`neon_dotprod_r_i8`], with a one-row panel.
unsafe fn neon_dotprod_r_i8_one(kc: usize, pa: *const i8, pb: *const i8, acc: *mut i32) {
    // SAFETY: the caller established the lengths and the target features.
    unsafe { neon_dotprod_r_i8_inner::<1>(kc, pa, pb, acc) }
}

// ---------------------------------------------------------------------------
// The table sequences (§7.3)
// ---------------------------------------------------------------------------

/// Lane words one 128-bit add covers, at a 32-bit lane.
const NEON_TABLE_LANES: usize = 4;

/// The `i8` table sequence at `rows` rows and `group` columns.
///
/// `rows` must be a whole number of 128-bit registers of `i32` *and* a whole
/// number of the eight-byte activation load the build widens, so the two tiles
/// this offers are sixteen and eight. Narrower tiles take the reference, whose
/// row count is a compile-time constant there too.
pub fn neon_table_i8_i32(rows: usize, group: usize) -> Option<TableSpec<i8, i32>> {
    let (build, gather, gather_codes): (
        crate::table::TableBuild<i8, i32>,
        crate::table::TableGather<i32>,
        crate::table::TableGatherCodes<i32>,
    ) = match (rows, group) {
        (16, 1) => (neon_table_build_v4, neon_gather_v4_u1, neon_codes_v4_u1),
        (16, 2) => (neon_table_build_v4, neon_gather_v4_u2, neon_codes_v4_u2),
        (8, 1) => (neon_table_build_v2, neon_gather_v2_u1, neon_codes_v2_u1),
        (8, 2) => (neon_table_build_v2, neon_gather_v2_u2, neon_codes_v2_u2),
        _ => return None,
    };
    Some(TableSpec {
        backend: Backend::Neon,
        rows,
        group,
        // `vmlal_s16` takes one block step at a time, so the activation tile
        // wants the plain `k`-major layout and the sequence has no tail (S8).
        k_group: 1,
        lanes_per_add: NEON_TABLE_LANES,
        // One `vmlal_s16` per four lanes and one block step.
        build_products_per_step: NEON_TABLE_LANES,
        lane_cap: i32::MAX as u128,
        // Every product is widened to `i32` before it is accumulated, so
        // nothing narrower than the lane is held and no alphabet is out of
        // reach --- the same statement `NEON_I8_I32` makes.
        max_bound: u128::MAX,
        build_multiplies: true,
        build,
        gather,
        gather_codes,
    })
}

/// The bound-1 `i8` table sequence at `rows` rows and `group` columns.
///
/// The same shapes and the same gathers as [`neon_table_i8_i32`] --- the
/// gathers are bound-independent, so they are shared, not duplicated. Only the
/// build differs: at bound 1 the widening multiply has nothing left to do, and
/// the slot is a sign mask, an XOR and a subtract (`CB-10`).
pub fn neon_table_i8_i32_bound1(rows: usize, group: usize) -> Option<TableSpec<i8, i32>> {
    let (build, gather, gather_codes): (
        crate::table::TableBuild<i8, i32>,
        crate::table::TableGather<i32>,
        crate::table::TableGatherCodes<i32>,
    ) = match (rows, group) {
        (16, 1) => (
            neon_table_build_bound1_v4,
            neon_gather_v4_u1,
            neon_codes_v4_u1,
        ),
        (16, 2) => (
            neon_table_build_bound1_v4,
            neon_gather_v4_u2,
            neon_codes_v4_u2,
        ),
        (8, 1) => (
            neon_table_build_bound1_v2,
            neon_gather_v2_u1,
            neon_codes_v2_u1,
        ),
        (8, 2) => (
            neon_table_build_bound1_v2,
            neon_gather_v2_u2,
            neon_codes_v2_u2,
        ),
        _ => return None,
    };
    Some(TableSpec {
        backend: Backend::Neon,
        rows,
        group,
        // Nothing is paired: one block step is one sign mask, so an odd block
        // packs as well as an even one and there is no tail.
        k_group: 1,
        lanes_per_add: NEON_TABLE_LANES,
        // One XOR, one subtract and one add per four lanes and one block step.
        build_products_per_step: NEON_TABLE_LANES,
        lane_cap: i32::MAX as u128,
        // Exact exactly when every book word is in `{-1, 0, +1}`: the sign
        // mask is the whole of the arithmetic, and this is the declaration
        // `choose_table` reads.
        max_bound: 1,
        build_multiplies: false,
        build,
        gather,
        gather_codes,
    })
}

/// Lane words one 128-bit add covers, at a 64-bit lane.
const NEON_TABLE_LANES_64: usize = 2;

/// The `i16` table sequence: two `i64` lanes to a register.
///
/// `vmull_s16` widens each product to `i32` and this widens again to `i64`
/// before accumulating, so nothing narrower than the lane is held and no `i16`
/// alphabet is out of reach --- unlike the AVX2 sequence for this family, whose
/// `madd` sums a pair into an `i32` and which therefore declares a bound.
pub fn neon_table_i16_i64(rows: usize, group: usize) -> Option<TableSpec<i16, i64>> {
    let (build, gather, gather_codes): (
        crate::table::TableBuild<i16, i64>,
        crate::table::TableGather<i64>,
        crate::table::TableGatherCodes<i64>,
    ) = match (rows, group) {
        (16, 1) => (neon_build16_v8, neon_gather64_v8_u1, neon_codes64_v8_u1),
        (16, 2) => (neon_build16_v8, neon_gather64_v8_u2, neon_codes64_v8_u2),
        (8, 1) => (neon_build16_v4, neon_gather64_v4_u1, neon_codes64_v4_u1),
        (8, 2) => (neon_build16_v4, neon_gather64_v4_u2, neon_codes64_v4_u2),
        _ => return None,
    };
    Some(TableSpec {
        backend: Backend::Neon,
        rows,
        group,
        k_group: 1,
        lanes_per_add: NEON_TABLE_LANES_64,
        build_products_per_step: NEON_TABLE_LANES_64,
        lane_cap: i64::MAX as u128,
        max_bound: u128::MAX,
        build_multiplies: true,
        build,
        gather,
        gather_codes,
    })
}

/// One slot of the `i16` table, at `V` registers of `i64`.
///
/// # Safety
///
/// [`crate::table::TableBuild`]'s contract, with `rows == V * 2` and `V` a
/// multiple of four.
#[target_feature(enable = "neon")]
unsafe fn neon_build16<const V: usize>(
    rows: usize,
    space: usize,
    block: usize,
    book: *const i16,
    acts: *const i16,
    out: *mut i64,
) {
    debug_assert_eq!(rows, V * NEON_TABLE_LANES_64);
    debug_assert!(V.is_multiple_of(4));
    // SAFETY: the caller established every extent.
    unsafe {
        for c in 0..space {
            let d = book.add(c * block);
            let mut entry = [vdupq_n_s64(0); V];
            for t in 0..block {
                let w = vdup_n_s16(*d.add(t));
                let a = acts.add(t * rows);
                // Four activations per widening multiply, which is two
                // registers of the lane --- so one iteration covers four
                // registers and therefore `4 * NEON_TABLE_LANES_64` rows. The
                // register index and the row index advance at different rates
                // and are written separately for that reason: an `i64` register
                // holds two rows, so the row cursor moves twice as fast as the
                // register cursor, and collapsing them silently re-reads the
                // previous iteration's activations.
                for quad in 0..V / 4 {
                    let reg = quad * 4;
                    let row = reg * NEON_TABLE_LANES_64;
                    let p = vmull_s16(vld1_s16(a.add(row)), w);
                    entry[reg] = vaddq_s64(entry[reg], vmovl_s32(vget_low_s32(p)));
                    entry[reg + 1] = vaddq_s64(entry[reg + 1], vmovl_s32(vget_high_s32(p)));
                    let q = vmull_s16(vld1_s16(a.add(row + 4)), w);
                    entry[reg + 2] = vaddq_s64(entry[reg + 2], vmovl_s32(vget_low_s32(q)));
                    entry[reg + 3] = vaddq_s64(entry[reg + 3], vmovl_s32(vget_high_s32(q)));
                }
            }
            let o = out.add(c * rows);
            for (v, cell) in entry.iter().enumerate() {
                vst1q_s64(o.add(v * NEON_TABLE_LANES_64), *cell);
            }
        }
    }
}

/// One column group in the 64-bit lane.
///
/// # Safety
///
/// [`crate::table::TableGather`]'s contract, with `rows == V * 2`, `group == U`.
#[target_feature(enable = "neon")]
unsafe fn neon_gather64<const V: usize, const U: usize>(
    rows: usize,
    _group: usize,
    depth: usize,
    slab: usize,
    stack: *const i64,
    off: *const u32,
    lane: *mut i64,
) {
    debug_assert_eq!(rows, V * NEON_TABLE_LANES_64);
    debug_assert_eq!(_group, U);
    // `slab - 1` alone bounds the entry's *base* and not the `rows` lanes read
    // from it, and `gather` is a safe public method: an offset that is not a
    // multiple of `rows` would start the read inside the last entry and run past
    // the slab. `rows` is a power of two and `slab` is a multiple of it, so
    // clearing the sub-row bits costs nothing --- both operands are constants and
    // this is the same single `and` --- and makes every read row-aligned, and so
    // in-slab, by construction rather than by the caller's discipline.
    let mask = ((slab - 1) & !(rows - 1)) as u32;
    // SAFETY: the caller established every extent.
    unsafe {
        let mut acc = [[vdupq_n_s64(0); V]; U];
        for (u, cols) in acc.iter_mut().enumerate() {
            for (v, cell) in cols.iter_mut().enumerate() {
                *cell = vld1q_s64(lane.add(u * rows + v * NEON_TABLE_LANES_64));
            }
        }
        let mut words = stack;
        for slot in 0..depth {
            for (u, cols) in acc.iter_mut().enumerate() {
                let entry = words.add((*off.add(slot * U + u) & mask) as usize);
                for (v, cell) in cols.iter_mut().enumerate() {
                    *cell = vaddq_s64(*cell, vld1q_s64(entry.add(v * NEON_TABLE_LANES_64)));
                }
            }
            words = words.add(slab);
        }
        for (u, cols) in acc.iter().enumerate() {
            for (v, cell) in cols.iter().enumerate() {
                vst1q_s64(lane.add(u * rows + v * NEON_TABLE_LANES_64), *cell);
            }
        }
    }
}

/// The same, over the coded operand's own memory.
///
/// # Safety
///
/// [`crate::table::TableGatherCodes`]'s contract, with `rows == V * 2`,
/// `group == U`.
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn neon_codes64<const V: usize, const U: usize>(
    rows: usize,
    _group: usize,
    depth: usize,
    slab: usize,
    shift: u32,
    stack: *const i64,
    codes: *const u16,
    stride: usize,
    lane: *mut i64,
) {
    debug_assert_eq!(rows, V * NEON_TABLE_LANES_64);
    debug_assert_eq!(_group, U);
    let mask = (slab >> shift) - 1;
    // SAFETY: the caller established every extent.
    unsafe {
        let mut acc = [[vdupq_n_s64(0); V]; U];
        for (u, cols) in acc.iter_mut().enumerate() {
            for (v, cell) in cols.iter_mut().enumerate() {
                *cell = vld1q_s64(lane.add(u * rows + v * NEON_TABLE_LANES_64));
            }
        }
        let mut cursor = [codes; U];
        for u in 1..U {
            cursor[u] = cursor[u - 1].add(stride);
        }
        let mut words = stack;
        for _ in 0..depth {
            for (u, cols) in acc.iter_mut().enumerate() {
                let entry = words.add((*cursor[u] as usize & mask) << shift);
                cursor[u] = cursor[u].add(1);
                for (v, cell) in cols.iter_mut().enumerate() {
                    *cell = vaddq_s64(*cell, vld1q_s64(entry.add(v * NEON_TABLE_LANES_64)));
                }
            }
            words = words.add(slab);
        }
        for (u, cols) in acc.iter().enumerate() {
            for (v, cell) in cols.iter().enumerate() {
                vst1q_s64(lane.add(u * rows + v * NEON_TABLE_LANES_64), *cell);
            }
        }
    }
}

/// # Safety
///
/// [`crate::table::TableBuild`]'s contract at `rows == 16`, 64-bit lane.
unsafe fn neon_build16_v8(
    rows: usize,
    space: usize,
    block: usize,
    book: *const i16,
    acts: *const i16,
    out: *mut i64,
) {
    // SAFETY: NEON is mandatory here and the caller forwarded the extents.
    unsafe { neon_build16::<8>(rows, space, block, book, acts, out) }
}

/// # Safety
///
/// [`crate::table::TableBuild`]'s contract at `rows == 8`, 64-bit lane.
unsafe fn neon_build16_v4(
    rows: usize,
    space: usize,
    block: usize,
    book: *const i16,
    acts: *const i16,
    out: *mut i64,
) {
    // SAFETY: NEON is mandatory here and the caller forwarded the extents.
    unsafe { neon_build16::<4>(rows, space, block, book, acts, out) }
}

/// Generate the four `(rows, group)` entry points for the 64-bit lane.
macro_rules! neon_gathers64 {
    ($($g:ident, $c:ident, $v:expr, $u:expr, $rows:expr;)*) => {$(
        #[doc = concat!("# Safety\n\n[`crate::table::TableGather`]'s contract at `rows == ", stringify!($rows), "`, `group == ", stringify!($u), "`.")]
        unsafe fn $g(
            rows: usize,
            group: usize,
            depth: usize,
            slab: usize,
            stack: *const i64,
            off: *const u32,
            lane: *mut i64,
        ) {
            // SAFETY: NEON is mandatory here and the caller forwarded the extents.
            unsafe { neon_gather64::<$v, $u>(rows, group, depth, slab, stack, off, lane) }
        }

        #[doc = concat!("# Safety\n\n[`crate::table::TableGatherCodes`]'s contract at `rows == ", stringify!($rows), "`, `group == ", stringify!($u), "`.")]
        #[allow(clippy::too_many_arguments)]
        unsafe fn $c(
            rows: usize,
            group: usize,
            depth: usize,
            slab: usize,
            shift: u32,
            stack: *const i64,
            codes: *const u16,
            stride: usize,
            lane: *mut i64,
        ) {
            // SAFETY: NEON is mandatory here and the caller forwarded the extents.
            unsafe {
                neon_codes64::<$v, $u>(rows, group, depth, slab, shift, stack, codes, stride, lane)
            }
        }
    )*};
}

neon_gathers64! {
    neon_gather64_v8_u1, neon_codes64_v8_u1, 8, 1, 16;
    neon_gather64_v8_u2, neon_codes64_v8_u2, 8, 2, 16;
    neon_gather64_v4_u1, neon_codes64_v4_u1, 4, 1, 8;
    neon_gather64_v4_u2, neon_codes64_v4_u2, 4, 2, 8;
}

/// One slot of the table, at `V` registers of `i32`.
///
/// `T[c][i] = sum_t acts[t][i] * book[c][t]`, one block step at a time.
/// `vmlal_s16` is a widening multiply-accumulate, so each product reaches the
/// `i32` lane without an `i16` intermediate to overflow --- the same reason
/// `NEON_I8_I32` widens rather than accumulating in `i16`.
///
/// # Safety
///
/// [`crate::table::TableBuild`]'s contract, with `rows == V * 4` and `V` even.
#[target_feature(enable = "neon")]
unsafe fn neon_table_build<const V: usize>(
    rows: usize,
    space: usize,
    block: usize,
    book: *const i8,
    acts: *const i8,
    out: *mut i32,
) {
    debug_assert_eq!(rows, V * NEON_TABLE_LANES);
    debug_assert!(V.is_multiple_of(2));
    // SAFETY: the caller established every extent.
    unsafe {
        for c in 0..space {
            let d = book.add(c * block);
            let mut entry = [vdupq_n_s32(0); V];
            for t in 0..block {
                let w = vdup_n_s16(*d.add(t) as i16);
                let a = acts.add(t * rows);
                // Eight activations per widening load, which is two lanes'
                // worth of `i32` register.
                for pair in 0..V / 2 {
                    let x = vmovl_s8(vld1_s8(a.add(pair * 8)));
                    entry[pair * 2] = vmlal_s16(entry[pair * 2], vget_low_s16(x), w);
                    entry[pair * 2 + 1] = vmlal_s16(entry[pair * 2 + 1], vget_high_s16(x), w);
                }
            }
            let o = out.add(c * rows);
            for (v, cell) in entry.iter().enumerate() {
                vst1q_s32(o.add(v * NEON_TABLE_LANES), *cell);
            }
        }
    }
}

/// One slot of the table at bound 1, at `V` registers: no multiply.
///
/// `T[c][i] = sum_t +-acts[t][i]`: the book word is in `{-1, 0, +1}`, so the
/// product is a masked negation. Two masks per block step, broadcast into the
/// lane's width --- `keep`, all-ones unless the word is zero, and `sign`,
/// all-ones when it is `-1` --- and `((a & keep) ^ sign) - sign` is the
/// product: in two's complement the XOR with all-ones is the one's complement,
/// and subtracting the mask back adds the one that makes it the negation. No
/// instruction in the loop multiplies.
///
/// The masked negation is computed in the `i16` the widening load produces ---
/// `|a| <= 128`, so the negation is exact there --- and the widening add
/// (`vaddw`) folds the move to the `i32` lane into the accumulation. Measured
/// against computing the masks in the `i32` lane directly (two more widening
/// steps and three more vector ops per eight rows), that is the difference
/// between this sequence and the autovectorized reference at parity, and the
/// reason the reference was the faster of the two.
///
/// # Safety
///
/// [`crate::table::TableBuild`]'s contract, with `rows == V * 4`, `V` even, and
/// every element of `book` in `{-1, 0, +1}` --- which the caller's bound-1
/// alphabet has already established at the boundary.
#[target_feature(enable = "neon")]
unsafe fn neon_table_build_bound1<const V: usize>(
    rows: usize,
    space: usize,
    block: usize,
    book: *const i8,
    acts: *const i8,
    out: *mut i32,
) {
    debug_assert_eq!(rows, V * NEON_TABLE_LANES);
    debug_assert!(V.is_multiple_of(2));
    // SAFETY: the caller established every extent.
    unsafe {
        for c in 0..space {
            let d = book.add(c * block);
            let mut entry = [vdupq_n_s32(0); V];
            for t in 0..block {
                let w = *d.add(t);
                // `w >> 7` is the arithmetic shift: all-ones exactly when the
                // word is -1, and `keep` is all-ones exactly when it is not 0.
                let sign = vdupq_n_s16(i16::from(w >> 7));
                let keep = vdupq_n_s16(-i16::from(w != 0));
                let a = acts.add(t * rows);
                // Eight activations per widening load, which is two lanes'
                // worth of `i32` register, as in `neon_table_build`.
                for pair in 0..V / 2 {
                    let x = vmovl_s8(vld1_s8(a.add(pair * 8)));
                    let z = vsubq_s16(veorq_s16(vandq_s16(x, keep), sign), sign);
                    entry[pair * 2] = vaddw_s16(entry[pair * 2], vget_low_s16(z));
                    entry[pair * 2 + 1] = vaddw_high_s16(entry[pair * 2 + 1], z);
                }
            }
            let o = out.add(c * rows);
            for (v, cell) in entry.iter().enumerate() {
                vst1q_s32(o.add(v * NEON_TABLE_LANES), *cell);
            }
        }
    }
}

/// One column group over a run of slots, at `V` registers of `i32` per column.
///
/// # Safety
///
/// [`crate::table::TableGather`]'s contract, with `rows == V * 4`, `group == U`.
#[target_feature(enable = "neon")]
unsafe fn neon_gather<const V: usize, const U: usize>(
    rows: usize,
    _group: usize,
    depth: usize,
    slab: usize,
    stack: *const i32,
    off: *const u32,
    lane: *mut i32,
) {
    debug_assert_eq!(rows, V * NEON_TABLE_LANES);
    debug_assert_eq!(_group, U);
    // `slab - 1` alone bounds the entry's *base* and not the `rows` lanes read
    // from it, and `gather` is a safe public method: an offset that is not a
    // multiple of `rows` would start the read inside the last entry and run past
    // the slab. `rows` is a power of two and `slab` is a multiple of it, so
    // clearing the sub-row bits costs nothing --- both operands are constants and
    // this is the same single `and` --- and makes every read row-aligned, and so
    // in-slab, by construction rather than by the caller's discipline.
    let mask = ((slab - 1) & !(rows - 1)) as u32;
    // SAFETY: the caller established every extent.
    unsafe {
        let mut acc = [[vdupq_n_s32(0); V]; U];
        for (u, cols) in acc.iter_mut().enumerate() {
            for (v, cell) in cols.iter_mut().enumerate() {
                *cell = vld1q_s32(lane.add(u * rows + v * NEON_TABLE_LANES));
            }
        }
        let mut words = stack;
        for slot in 0..depth {
            for (u, cols) in acc.iter_mut().enumerate() {
                // The mask, not a comparison: every offset reads in-slab
                // whatever it holds, so this is one `and` and never a branch.
                let entry = words.add((*off.add(slot * U + u) & mask) as usize);
                for (v, cell) in cols.iter_mut().enumerate() {
                    *cell = vaddq_s32(*cell, vld1q_s32(entry.add(v * NEON_TABLE_LANES)));
                }
            }
            words = words.add(slab);
        }
        for (u, cols) in acc.iter().enumerate() {
            for (v, cell) in cols.iter().enumerate() {
                vst1q_s32(lane.add(u * rows + v * NEON_TABLE_LANES), *cell);
            }
        }
    }
}

/// The same, over the coded operand's own memory.
///
/// # Safety
///
/// [`crate::table::TableGatherCodes`]'s contract, with `rows == V * 4`,
/// `group == U`.
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn neon_codes<const V: usize, const U: usize>(
    rows: usize,
    _group: usize,
    depth: usize,
    slab: usize,
    shift: u32,
    stack: *const i32,
    codes: *const u16,
    stride: usize,
    lane: *mut i32,
) {
    debug_assert_eq!(rows, V * NEON_TABLE_LANES);
    debug_assert_eq!(_group, U);
    let mask = (slab >> shift) - 1;
    // SAFETY: the caller established every extent.
    unsafe {
        let mut acc = [[vdupq_n_s32(0); V]; U];
        for (u, cols) in acc.iter_mut().enumerate() {
            for (v, cell) in cols.iter_mut().enumerate() {
                *cell = vld1q_s32(lane.add(u * rows + v * NEON_TABLE_LANES));
            }
        }
        let mut cursor = [codes; U];
        for u in 1..U {
            cursor[u] = cursor[u - 1].add(stride);
        }
        let mut words = stack;
        for _ in 0..depth {
            for (u, cols) in acc.iter_mut().enumerate() {
                let entry = words.add((*cursor[u] as usize & mask) << shift);
                cursor[u] = cursor[u].add(1);
                for (v, cell) in cols.iter_mut().enumerate() {
                    *cell = vaddq_s32(*cell, vld1q_s32(entry.add(v * NEON_TABLE_LANES)));
                }
            }
            words = words.add(slab);
        }
        for (u, cols) in acc.iter().enumerate() {
            for (v, cell) in cols.iter().enumerate() {
                vst1q_s32(lane.add(u * rows + v * NEON_TABLE_LANES), *cell);
            }
        }
    }
}

/// Build one slot at `rows == 16`.
///
/// # Safety
///
/// [`crate::table::TableBuild`]'s contract at `rows == 16`.
unsafe fn neon_table_build_v4(
    rows: usize,
    space: usize,
    block: usize,
    book: *const i8,
    acts: *const i8,
    out: *mut i32,
) {
    // SAFETY: NEON is mandatory on this target and the caller forwarded the
    // extents.
    unsafe { neon_table_build::<4>(rows, space, block, book, acts, out) }
}

/// Build one slot at `rows == 8`.
///
/// # Safety
///
/// [`crate::table::TableBuild`]'s contract at `rows == 8`.
unsafe fn neon_table_build_v2(
    rows: usize,
    space: usize,
    block: usize,
    book: *const i8,
    acts: *const i8,
    out: *mut i32,
) {
    // SAFETY: as [`neon_table_build_v4`], at a narrower tile.
    unsafe { neon_table_build::<2>(rows, space, block, book, acts, out) }
}

/// Build one slot at bound 1, at `rows == 16`.
///
/// # Safety
///
/// [`crate::table::TableBuild`]'s contract at `rows == 16`, every book word in
/// `{-1, 0, +1}`.
unsafe fn neon_table_build_bound1_v4(
    rows: usize,
    space: usize,
    block: usize,
    book: *const i8,
    acts: *const i8,
    out: *mut i32,
) {
    // SAFETY: NEON is mandatory on this target and the caller forwarded the
    // extents.
    unsafe { neon_table_build_bound1::<4>(rows, space, block, book, acts, out) }
}

/// Build one slot at bound 1, at `rows == 8`.
///
/// # Safety
///
/// [`crate::table::TableBuild`]'s contract at `rows == 8`, every book word in
/// `{-1, 0, +1}`.
unsafe fn neon_table_build_bound1_v2(
    rows: usize,
    space: usize,
    block: usize,
    book: *const i8,
    acts: *const i8,
    out: *mut i32,
) {
    // SAFETY: as [`neon_table_build_bound1_v4`], at a narrower tile.
    unsafe { neon_table_build_bound1::<2>(rows, space, block, book, acts, out) }
}

/// Generate the four `(rows, group)` gather entry points, each a named
/// monomorphization of one sequence.
macro_rules! neon_gathers {
    ($($g:ident, $c:ident, $v:expr, $u:expr, $rows:expr;)*) => {$(
        #[doc = concat!("# Safety\n\n[`crate::table::TableGather`]'s contract at `rows == ", stringify!($rows), "`, `group == ", stringify!($u), "`.")]
        unsafe fn $g(
            rows: usize,
            group: usize,
            depth: usize,
            slab: usize,
            stack: *const i32,
            off: *const u32,
            lane: *mut i32,
        ) {
            // SAFETY: NEON is mandatory here and the caller forwarded the extents.
            unsafe { neon_gather::<$v, $u>(rows, group, depth, slab, stack, off, lane) }
        }

        #[doc = concat!("# Safety\n\n[`crate::table::TableGatherCodes`]'s contract at `rows == ", stringify!($rows), "`, `group == ", stringify!($u), "`.")]
        #[allow(clippy::too_many_arguments)]
        unsafe fn $c(
            rows: usize,
            group: usize,
            depth: usize,
            slab: usize,
            shift: u32,
            stack: *const i32,
            codes: *const u16,
            stride: usize,
            lane: *mut i32,
        ) {
            // SAFETY: NEON is mandatory here and the caller forwarded the extents.
            unsafe {
                neon_codes::<$v, $u>(rows, group, depth, slab, shift, stack, codes, stride, lane)
            }
        }
    )*};
}

neon_gathers! {
    neon_gather_v4_u1, neon_codes_v4_u1, 4, 1, 16;
    neon_gather_v4_u2, neon_codes_v4_u2, 4, 2, 16;
    neon_gather_v2_u1, neon_codes_v2_u1, 2, 1, 8;
    neon_gather_v2_u2, neon_codes_v2_u2, 2, 2, 8;
}
