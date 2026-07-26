//! WebAssembly SIMD128 (§7.2).
//!
//! `i32x4_dot_i16x8` pairs two `k`-steps into an `i32` lane, and `i32x4_add`
//! accumulates. The same factorization as AVX2 at a quarter of the width,
//! which is the point of having a [`KernelSpec`] rather than a per-ISA driver.

use core::arch::wasm32::*;

use uor_matmul_core::Backend;

use crate::spec::{Factorization, KernelSpec, LaneLayout};
use crate::table::TableSpec;

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
    lane_layout: LaneLayout::Interleaved,
    k_group: 2,
    products_per_step: 8,
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

const R_MR: usize = 4;

crate::tile_fits!(R_MR, 1);
crate::tile_fits!(1, 1);

/// wasm SIMD128 `i8`, reducing four rows against one column with the lanes on
/// `k`.
///
/// Sixteen `k`-steps per iteration, through the same `dot` instruction the tile
/// kernel uses --- so the declared alphabet is the same `32767`, which an `i8`
/// alphabet cannot reach.
pub const SIMD128_R_I8_I32: KernelSpec<i8, i32> = KernelSpec {
    backend: Backend::WasmSimd128,
    factorization: Factorization::Exact,
    mr: R_MR,
    nr: 1,
    lane_layout: LaneLayout::Contiguous,
    k_group: 16,
    products_per_step: 8,
    lane_cap: i32::MAX as u128,
    max_bound: 32767,
    mac_tile: simd128_r_i8,
};

/// # Safety
///
/// `pa` must have `4 * kc` readable elements with row `i` at `pa[i * kc ..][..kc]`,
/// `pb` must have `kc`, `acc` 4 writable lanes, and `kc` a multiple of 16.
unsafe fn simd128_r_i8_generic<const MR: usize>(
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
    let mut sums = [i32x4_splat(0); MR];

    for q in 0..kc / 16 {
        // SAFETY: `pb[q * 16 ..][..16]` is in bounds: one v128 load.
        let braw = unsafe { v128_load(pb.as_ptr().add(q * 16).cast()) };
        let (bl, bh) = (i16x8_extend_low_i8x16(braw), i16x8_extend_high_i8x16(braw));
        for (i, sum) in sums.iter_mut().enumerate() {
            // SAFETY: `i * kc + q * 16 + 16 <= MR * kc`.
            let araw = unsafe { v128_load(pa.as_ptr().add(i * kc + q * 16).cast()) };
            *sum = i32x4_add(*sum, i32x4_dot_i16x8(i16x8_extend_low_i8x16(araw), bl));
            *sum = i32x4_add(*sum, i32x4_dot_i16x8(i16x8_extend_high_i8x16(araw), bh));
        }
    }

    for (i, sum) in sums.iter().enumerate() {
        // Every partial sum is bounded by the sum of the lane magnitudes, which
        // `lane_cap` keeps inside an `i32`.
        acc[i] = i32x4_extract_lane::<0>(*sum)
            .wrapping_add(i32x4_extract_lane::<1>(*sum))
            .wrapping_add(i32x4_extract_lane::<2>(*sum))
            .wrapping_add(i32x4_extract_lane::<3>(*sum));
    }
}

/// The same sequence at a one-row panel. See [`SIMD128_R_I8_I32`].
pub const SIMD128_R_I8_I32_1: KernelSpec<i8, i32> = KernelSpec {
    backend: Backend::WasmSimd128,
    factorization: Factorization::Exact,
    mr: 1,
    nr: 1,
    lane_layout: LaneLayout::Contiguous,
    k_group: 16,
    products_per_step: 8,
    lane_cap: i32::MAX as u128,
    max_bound: 32767,
    mac_tile: simd128_r_i8_one,
};

/// # Safety
///
/// As [`simd128_r_i8`], with a one-row panel.
unsafe fn simd128_r_i8_one(kc: usize, pa: *const i8, pb: *const i8, acc: *mut i32) {
    // SAFETY: the caller forwarded the lengths.
    unsafe { simd128_r_i8_generic::<1>(kc, pa, pb, acc) }
}

/// # Safety
///
/// `pa` must have `4 * kc` readable elements with row `i` at `pa[i * kc ..][..kc]`,
/// `pb` must have `kc`, `acc` 4 writable lanes, and `kc` a multiple of 16.
unsafe fn simd128_r_i8(kc: usize, pa: *const i8, pb: *const i8, acc: *mut i32) {
    // SAFETY: the caller forwarded the lengths.
    unsafe { simd128_r_i8_generic::<R_MR>(kc, pa, pb, acc) }
}

// ---------------------------------------------------------------------------
// The table sequences (§7.3)
// ---------------------------------------------------------------------------

/// Lane words one 128-bit add covers, at a 32-bit lane.
const SIMD_TABLE_LANES: usize = 4;

/// The `i8` table sequence at `rows` rows and `group` columns.
///
/// Sixteen and eight rows, which are four and two 128-bit registers of `i32`.
/// Narrower tiles take the reference, whose row count is a compile-time constant
/// there too.
pub fn simd128_table_i8_i32(rows: usize, group: usize) -> Option<TableSpec<i8, i32>> {
    let (build, gather, gather_codes): (
        crate::table::TableBuild<i8, i32>,
        crate::table::TableGather<i32>,
        crate::table::TableGatherCodes<i32>,
    ) = match (rows, group) {
        (16, 1) => (simd_build_v4, simd_gather_v4_u1, simd_codes_v4_u1),
        (16, 2) => (simd_build_v4, simd_gather_v4_u2, simd_codes_v4_u2),
        (8, 1) => (simd_build_v2, simd_gather_v2_u1, simd_codes_v2_u1),
        (8, 2) => (simd_build_v2, simd_gather_v2_u2, simd_codes_v2_u2),
        _ => return None,
    };
    Some(TableSpec {
        backend: Backend::WasmSimd128,
        rows,
        group,
        // `extmul` takes one block step at a time, so the activation tile wants
        // the plain `k`-major layout and the sequence has no tail (S8).
        k_group: 1,
        lanes_per_add: SIMD_TABLE_LANES,
        build_products_per_step: SIMD_TABLE_LANES,
        lane_cap: i32::MAX as u128,
        // Every product is widened to `i32` before it is accumulated, so
        // nothing narrower than the lane is held.
        max_bound: u128::MAX,
        build,
        gather,
        gather_codes,
    })
}

/// The `i16` table sequence. Absent here; the reference carries this family.
pub fn simd128_table_i16_i64(_rows: usize, _group: usize) -> Option<TableSpec<i16, i64>> {
    None
}

/// One slot of the table, at `V` registers of `i32`.
///
/// `i32x4_extmul_*_i16x8` is a widening multiply, so each product reaches the
/// `i32` lane with no `i16` intermediate to overflow.
///
/// # Safety
///
/// [`crate::table::TableBuild`]'s contract, with `rows == V * 4` and
/// `V` in `{2, 4}`.
unsafe fn simd_build<const V: usize>(
    rows: usize,
    space: usize,
    block: usize,
    book: *const i8,
    acts: *const i8,
    out: *mut i32,
) {
    debug_assert_eq!(rows, V * SIMD_TABLE_LANES);
    // SAFETY: the caller established every extent.
    unsafe {
        for c in 0..space {
            let d = book.add(c * block);
            let mut entry = [i32x4_splat(0); V];
            for t in 0..block {
                let w = i16x8_splat(*d.add(t) as i16);
                let a = acts.add(t * rows);
                // Sixteen activations is a whole register; eight is a half-load
                // that zeroes the rest, so a tile of eight never reads past its
                // own row (S13, and the reason there is no `v128_load` here).
                let x = if V == 4 {
                    v128_load(a as *const v128)
                } else {
                    v128_load64_zero(a as *const u64)
                };
                let lo = i16x8_extend_low_i8x16(x);
                entry[0] = i32x4_add(entry[0], i32x4_extmul_low_i16x8(lo, w));
                entry[1] = i32x4_add(entry[1], i32x4_extmul_high_i16x8(lo, w));
                if V == 4 {
                    let hi = i16x8_extend_high_i8x16(x);
                    entry[2] = i32x4_add(entry[2], i32x4_extmul_low_i16x8(hi, w));
                    entry[3] = i32x4_add(entry[3], i32x4_extmul_high_i16x8(hi, w));
                }
            }
            let o = out.add(c * rows);
            for (v, cell) in entry.iter().enumerate() {
                v128_store(o.add(v * SIMD_TABLE_LANES) as *mut v128, *cell);
            }
        }
    }
}

/// One column group over a run of slots, at `V` registers of `i32` per column.
///
/// # Safety
///
/// [`crate::table::TableGather`]'s contract, with `rows == V * 4`, `group == U`.
unsafe fn simd_gather<const V: usize, const U: usize>(
    rows: usize,
    _group: usize,
    depth: usize,
    slab: usize,
    stack: *const i32,
    off: *const u32,
    lane: *mut i32,
) {
    debug_assert_eq!(rows, V * SIMD_TABLE_LANES);
    debug_assert_eq!(_group, U);
    let mask = (slab - 1) as u32;
    // SAFETY: the caller established every extent.
    unsafe {
        let mut acc = [[i32x4_splat(0); V]; U];
        for (u, cols) in acc.iter_mut().enumerate() {
            for (v, cell) in cols.iter_mut().enumerate() {
                *cell = v128_load(lane.add(u * rows + v * SIMD_TABLE_LANES) as *const v128);
            }
        }
        let mut words = stack;
        for slot in 0..depth {
            for (u, cols) in acc.iter_mut().enumerate() {
                // The mask, not a comparison: every offset reads in-slab
                // whatever it holds, so this is one `and` and never a branch.
                let entry = words.add((*off.add(slot * U + u) & mask) as usize);
                for (v, cell) in cols.iter_mut().enumerate() {
                    *cell = i32x4_add(
                        *cell,
                        v128_load(entry.add(v * SIMD_TABLE_LANES) as *const v128),
                    );
                }
            }
            words = words.add(slab);
        }
        for (u, cols) in acc.iter().enumerate() {
            for (v, cell) in cols.iter().enumerate() {
                v128_store(
                    lane.add(u * rows + v * SIMD_TABLE_LANES) as *mut v128,
                    *cell,
                );
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
#[allow(clippy::too_many_arguments)]
unsafe fn simd_codes<const V: usize, const U: usize>(
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
    debug_assert_eq!(rows, V * SIMD_TABLE_LANES);
    debug_assert_eq!(_group, U);
    let mask = (slab >> shift) - 1;
    // SAFETY: the caller established every extent.
    unsafe {
        let mut acc = [[i32x4_splat(0); V]; U];
        for (u, cols) in acc.iter_mut().enumerate() {
            for (v, cell) in cols.iter_mut().enumerate() {
                *cell = v128_load(lane.add(u * rows + v * SIMD_TABLE_LANES) as *const v128);
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
                    *cell = i32x4_add(
                        *cell,
                        v128_load(entry.add(v * SIMD_TABLE_LANES) as *const v128),
                    );
                }
            }
            words = words.add(slab);
        }
        for (u, cols) in acc.iter().enumerate() {
            for (v, cell) in cols.iter().enumerate() {
                v128_store(
                    lane.add(u * rows + v * SIMD_TABLE_LANES) as *mut v128,
                    *cell,
                );
            }
        }
    }
}

/// # Safety
///
/// [`crate::table::TableBuild`]'s contract at `rows == 16`.
unsafe fn simd_build_v4(
    rows: usize,
    space: usize,
    block: usize,
    book: *const i8,
    acts: *const i8,
    out: *mut i32,
) {
    // SAFETY: the caller forwarded the extents.
    unsafe { simd_build::<4>(rows, space, block, book, acts, out) }
}

/// # Safety
///
/// [`crate::table::TableBuild`]'s contract at `rows == 8`.
unsafe fn simd_build_v2(
    rows: usize,
    space: usize,
    block: usize,
    book: *const i8,
    acts: *const i8,
    out: *mut i32,
) {
    // SAFETY: the caller forwarded the extents.
    unsafe { simd_build::<2>(rows, space, block, book, acts, out) }
}

/// Generate the four `(rows, group)` gather entry points, each a named
/// monomorphization of one sequence.
macro_rules! simd_gathers {
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
            // SAFETY: the caller forwarded the extents.
            unsafe { simd_gather::<$v, $u>(rows, group, depth, slab, stack, off, lane) }
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
            // SAFETY: the caller forwarded the extents.
            unsafe {
                simd_codes::<$v, $u>(rows, group, depth, slab, shift, stack, codes, stride, lane)
            }
        }
    )*};
}

simd_gathers! {
    simd_gather_v4_u1, simd_codes_v4_u1, 4, 1, 16;
    simd_gather_v4_u2, simd_codes_v4_u2, 4, 2, 16;
    simd_gather_v2_u1, simd_codes_v2_u1, 2, 1, 8;
    simd_gather_v2_u2, simd_codes_v2_u2, 2, 2, 8;
}
