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
// The (max, +) reduction in a packed i16 lane
// ---------------------------------------------------------------------------

/// Rows of the tropical tile: four `v128` accumulators.
const TROP_MR: usize = 4;
/// Columns: eight `i16`, which is one `v128` exactly.
const TROP_NR: usize = 8;

crate::tile_fits!(TROP_MR, TROP_NR);

/// wasm SIMD128 `(max, +)`: `i16x8_add_sat` is `⊗` and `i16x8_max` is `⊕`.
///
/// The AVX2 sequence at a quarter of the width and the NEON one at the same
/// width, with the same two instructions in the same order --- which is what a
/// semiring with no carry and no growth buys: there is no widening step for
/// three ISAs to disagree about, so the three bodies differ only in how many
/// lanes a register holds.
///
/// `i16x8_add_sat` and not `i16x8_add`: the saturating variant *is* the
/// absorbing law `-inf ⊗ a = -inf`, and [`crate::tropical`] derives why the
/// wrapping one is wrong at exactly the input a random sweep never generates
/// --- two masked operands, where `i16::MIN + i16::MIN` wraps to `0`.
pub const SIMD128_TROP_I16: KernelSpec<i16, i16> = KernelSpec {
    backend: Backend::WasmSimd128,
    factorization: Factorization::Exact,
    mr: TROP_MR,
    nr: TROP_NR,
    lane_layout: LaneLayout::Interleaved,
    // One `k`-step per instruction: the splat covers `A` and the load covers a
    // whole `k`-step of `B`.
    k_group: 1,
    products_per_step: TROP_NR,
    lane_cap: u128::MAX,
    max_bound: crate::tropical::TROP_I16_MAX_BOUND,
    mac_tile: simd128_trop_i16,
};

/// # Safety
///
/// `pa` must have `4 * kc` readable elements, `pb` `8 * kc`, and `acc` 32
/// writable lanes.
unsafe fn simd128_trop_i16(kc: usize, pa: *const i16, pb: *const i16, acc: *mut i16) {
    // SAFETY: the caller guaranteed the three extents. One conversion here
    // keeps every panel read below safe.
    let (pa, pb, acc) = unsafe {
        (
            core::slice::from_raw_parts(pa, TROP_MR * kc),
            core::slice::from_raw_parts(pb, TROP_NR * kc),
            core::slice::from_raw_parts_mut(acc, TROP_MR * TROP_NR),
        )
    };
    // The identity of `max`, which is the semiring zero and not zero: at
    // `kc == 0` this is the whole answer.
    let mut tile = [i16x8_splat(crate::tropical::TROP_ZERO); TROP_MR];

    for p in 0..kc {
        // The panel is `k`-major at `k_group == 1`, so one v128 load is a whole
        // `k`-step of `B`: eight columns, in lane order.
        //
        // SAFETY: `pb[p * TROP_NR ..][..8]` is in bounds: one v128 load.
        let bv = unsafe { v128_load(pb.as_ptr().add(p * TROP_NR).cast()) };
        for (i, cell) in tile.iter_mut().enumerate() {
            let av = i16x8_splat(pa[p * TROP_MR + i]);
            *cell = i16x8_max(*cell, i16x8_add_sat(av, bv));
        }
    }

    for (i, cell) in tile.iter().enumerate() {
        // SAFETY: `i < TROP_MR`, so this store lands inside `TROP_MR * TROP_NR`.
        unsafe { v128_store(acc.as_mut_ptr().add(i * TROP_NR).cast(), *cell) };
    }
}

// ---------------------------------------------------------------------------
// The SWAR broadcast sequence
// ---------------------------------------------------------------------------

const SWAR_MR: usize = 4;
const SWAR_NR: usize = 12;

crate::tile_fits!(SWAR_MR, SWAR_NR);

/// The field spacing, in bits: three fields to a 64-bit lane, five guard bits
/// a field over the widest biased product.
const SWAR_SPACING: u32 = 21;

/// The fields per 64-bit lane.
const SWAR_T: usize = 3;

/// The deepest run of products one field absorbs before extraction. Derived,
/// not chosen: a field is `SWAR_SPACING` bits and a biased product reaches
/// `255 * 255`, so the run is `floor(((1 << SWAR_SPACING) - 1) / (255 * 255))`.
/// The model's `wasm_swar_field_w8a8` row records the same derivation.
const SWAR_CHUNK: usize = 32;

// R1: the field-capacity threshold, pinned. This is the one place in the
// shipped crates the derivation's numeral may appear.
const _: () = assert!(SWAR_CHUNK as u128 == ((1u128 << SWAR_SPACING) - 1) / (255 * 255));

/// The wasm SIMD128 SWAR broadcast spec: three `B` elements packed at 21-bit
/// spacing in each 64-bit lane, multiplied by one broadcast scalar.
///
/// Kronecker substitution over the integers: a plain identity, available to
/// any library, exact by construction rather than by anything this library
/// adds. Both operands are biased to unsigned --- the same `+128` offset
/// identity `AVX512_DPBUSD_I8_I32` uses, applied on both sides here, with the
/// compensation paid at extraction --- so a product reaches `255 * 255`,
/// three fields at [`SWAR_SPACING`]-bit spacing sit in one 64-bit lane with
/// five guard bits a field, and one `i64x2.mul` against a splatted scalar
/// produces six products with no cross terms. The guard bits absorb a chunk
/// of [`SWAR_CHUNK`] products a field before the fields are extracted and
/// corrected into the `i32` lanes; a deeper `k` is more chunks. That *is* the
/// guard-bits-with-periodic-extraction mechanism, and the driver's lane
/// capacity sits on top of it unchanged.
///
/// Why this sequence exists on baseline SIMD128 and nowhere else: the ISA has
/// `i64x2.mul` and no byte-width dot product. relaxed-simd's
/// `i8x16.dot_i8x16_i7x8_s` is specified non-deterministic --- its
/// intermediate precision is implementation-defined --- so this library cannot
/// use it regardless of availability (C3: the schedule cannot change the
/// answer), and the choice on baseline SIMD128 is this or the extending dot.
/// On x86 `vpdpbusd` does the multiplexing in silicon and on thumbv7em
/// `SMLAD` does; neither gets a SWAR sequence.
///
/// Measured under wasmtime (`CG-17`, `just swar-sweep`): 0.32--0.38x of the
/// extending dot sequence it would displace, at every depth and both bounds
/// --- the field packing costs more than the extends it saves, the same
/// reason OpenBLAS declines the trick for throughput rather than numerics. So
/// no family list carries it: a listed sequence is one `Auto` may select, and
/// selecting this one would be a measured regression. The spec stays exported
/// and `CB-12`-pinned --- the bytes, and the absence from every availability
/// list, which is what keeps the decline a decision rather than an accident.
pub const SIMD128_SWAR_I8_I32: KernelSpec<i8, i32> = KernelSpec {
    backend: Backend::WasmSimd128,
    factorization: Factorization::Exact,
    mr: SWAR_MR,
    nr: SWAR_NR,
    lane_layout: LaneLayout::Interleaved,
    // The plain `k`-major layout: one step of the pack is one byte from each
    // column, contiguous in the panel. The chunking is internal, so there is
    // no group for the driver to pad to and no tail (S8).
    k_group: 1,
    products_per_step: 2 * SWAR_T,
    lane_cap: i32::MAX as u128,
    // The biased-field construction is exact at every `i8` alphabet: the
    // biased operands are bytes whatever the declared bound, so no alphabet
    // outgrows a field.
    max_bound: u128::MAX,
    mac_tile: simd128_swar_i8,
};

/// # Safety
///
/// `pa` must have `4 * kc` readable elements, `pb` `12 * kc`, and `acc` 48
/// writable lanes.
unsafe fn simd128_swar_i8(kc: usize, pa: *const i8, pb: *const i8, acc: *mut i32) {
    // SAFETY: the caller guaranteed the three extents. One conversion here
    // keeps every panel read below safe.
    let (pa, pb, acc) = unsafe {
        (
            core::slice::from_raw_parts(pa, SWAR_MR * kc),
            core::slice::from_raw_parts(pb, SWAR_NR * kc),
            core::slice::from_raw_parts_mut(acc, SWAR_MR * SWAR_NR),
        )
    };
    // The field constants, derived from the spacing. A byte moves to a 21-bit
    // field with one mask and one shift per field past the first, and the bias
    // and its mask are one splat each: `(raw + 128) & 0xFF` is the biased
    // element `b + 128` exactly, in two's complement and at any `i8`.
    let byte1 = i64x2_splat(0xFF00);
    let byte2 = i64x2_splat(0xFF_0000);
    let bias = i64x2_splat(128 * ((1i64 << 42) + (1 << 21) + 1));
    let field_mask = i64x2_splat(0xFF * ((1i64 << 42) + (1 << 21) + 1));
    let lane_mask = i64x2_splat((1i64 << SWAR_SPACING) - 1);

    // The tile is written, not accumulated into: what was in `acc` is the
    // caller's business, the same contract the other kernels keep.
    acc.fill(0);

    let mut k = 0;
    while k < kc {
        let d = SWAR_CHUNK.min(kc - k);
        let mut packed = [i64x2_splat(0); SWAR_MR * 2];
        let mut bsum = [i64x2_splat(0); 2];
        let mut asum = [0i64; SWAR_MR];
        for kk in 0..d {
            let row = k + kk;
            // The biased scalar, one per row: `a + 128` as a 64-bit lane, and
            // its sum for the compensation.
            let mut avec = [i64x2_splat(0); SWAR_MR];
            for (i, (av, s)) in avec.iter_mut().zip(asum.iter_mut()).enumerate() {
                let a = i64::from(pa[row * SWAR_MR + i]) + 128;
                *s += a;
                *av = i64x2_splat(a);
            }
            for g in 0..2 {
                // Six columns to a register, lane 0 holding columns `6g..6g+3`
                // and lane 1 `6g+3..6g+6`. Both eight-byte reads stay inside
                // the panel at every `row < kc` --- the second block reads
                // columns 4..12 and uses the last six --- where a six-byte
                // read at the last row of the second block would not.
                let base = row * SWAR_NR + g * 4;
                // SAFETY: `base + 8 <= row * 12 + 12 <= kc * 12`.
                let raw = unsafe { v128_load64_zero(pb.as_ptr().add(base).cast::<u64>()) };
                let y = if g == 0 {
                    i8x16_shuffle::<0, 1, 2, 0, 0, 0, 0, 0, 3, 4, 5, 0, 0, 0, 0, 0>(raw, raw)
                } else {
                    i8x16_shuffle::<2, 3, 4, 0, 0, 0, 0, 0, 5, 6, 7, 0, 0, 0, 0, 0>(raw, raw)
                };
                let spread = v128_or(
                    v128_or(
                        v128_and(y, i64x2_splat(0xFF)),
                        i64x2_shl(v128_and(y, byte1), 13),
                    ),
                    i64x2_shl(v128_and(y, byte2), 26),
                );
                let b = v128_and(i64x2_add(spread, bias), field_mask);
                bsum[g] = i64x2_add(bsum[g], b);
                for (i, av) in avec.iter().enumerate() {
                    let cell = &mut packed[i * 2 + g];
                    *cell = i64x2_add(*cell, i64x2_mul(*av, b));
                }
            }
        }
        // Extraction and the compensation, the two-sided form of the `dpbusd`
        // offset identity:
        //
        // ```text
        // sum(a*b) = sum(a'*b') - 128 * sum(a') - 128 * sum(b') + 16384 * d
        // ```
        //
        // Every term is an exact integer and the chunk bounds keep every
        // intermediate inside its lane: a field holds at most
        // `CHUNK * 255 * 255 < 2^21`, a biased column sum at most
        // `CHUNK * 255`, and the corrected chunk sum at most
        // `CHUNK * 128 * 128 < 2^31`, so the low four bytes of each corrected
        // lane are the `i32` to accumulate, sign included.
        let step_bias = (d as i64) << 14;
        for g in 0..2 {
            let c = [
                i64x2_shl(v128_and(bsum[g], lane_mask), 7),
                i64x2_shl(v128_and(i64x2_shr(bsum[g], SWAR_SPACING), lane_mask), 7),
                i64x2_shl(i64x2_shr(bsum[g], 2 * SWAR_SPACING), 7),
            ];
            for i in 0..SWAR_MR {
                let p = packed[i * 2 + g];
                let f = [
                    v128_and(p, lane_mask),
                    v128_and(i64x2_shr(p, SWAR_SPACING), lane_mask),
                    i64x2_shr(p, 2 * SWAR_SPACING),
                ];
                let rv = i64x2_splat((asum[i] << 7) - step_bias);
                let mut lanes = [0i64; 2];
                for (field, (fv, cv)) in f.iter().zip(c.iter()).enumerate() {
                    let t = i64x2_sub(i64x2_sub(*fv, rv), *cv);
                    // SAFETY: a sixteen-byte store into a sixteen-byte stack
                    // slot.
                    unsafe { v128_store(lanes.as_mut_ptr().cast(), t) };
                    for (half, x) in lanes.iter().enumerate() {
                        let col = g * 6 + half * SWAR_T + field;
                        let cell = &mut acc[i * SWAR_NR + col];
                        *cell = cell.wrapping_add(*x as i32);
                    }
                }
            }
        }
        k += d;
    }
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
    let (build, gather, gather_codes, gather_codes_u8): (
        crate::table::TableBuild<i8, i32>,
        crate::table::TableGather<i32>,
        crate::table::TableGatherCodes<i32>,
        crate::table::TableGatherCodesU8<i32>,
    ) = match (rows, group) {
        (16, 1) => (
            simd_build_v4,
            simd_gather_v4_u1,
            simd_codes_v4_u1,
            simd_codes_v4_u1_u8,
        ),
        (16, 2) => (
            simd_build_v4,
            simd_gather_v4_u2,
            simd_codes_v4_u2,
            simd_codes_v4_u2_u8,
        ),
        (8, 1) => (
            simd_build_v2,
            simd_gather_v2_u1,
            simd_codes_v2_u1,
            simd_codes_v2_u1_u8,
        ),
        (8, 2) => (
            simd_build_v2,
            simd_gather_v2_u2,
            simd_codes_v2_u2,
            simd_codes_v2_u2_u8,
        ),
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
        build_multiplies: true,
        build_adds: crate::table::product_build_adds,
        build,
        gather,
        gather_codes,
        gather_codes_u8,
    })
}

/// Lane words one 128-bit add covers, at a 64-bit lane.
const SIMD_TABLE_LANES_64: usize = 2;

/// The `i16` table sequence: two `i64` lanes to a register.
///
/// `i32x4_extmul_*_i16x8` widens each product to `i32` and `i64x2_extend_*`
/// widens again before accumulating, so nothing narrower than the lane is held
/// and no `i16` alphabet is out of reach.
pub fn simd128_table_i16_i64(rows: usize, group: usize) -> Option<TableSpec<i16, i64>> {
    let (build, gather, gather_codes, gather_codes_u8): (
        crate::table::TableBuild<i16, i64>,
        crate::table::TableGather<i64>,
        crate::table::TableGatherCodes<i64>,
        crate::table::TableGatherCodesU8<i64>,
    ) = match (rows, group) {
        (16, 1) => (
            simd_build16_v8,
            simd_gather64_v8_u1,
            simd_codes64_v8_u1,
            simd_codes64_v8_u1_u8,
        ),
        (16, 2) => (
            simd_build16_v8,
            simd_gather64_v8_u2,
            simd_codes64_v8_u2,
            simd_codes64_v8_u2_u8,
        ),
        (8, 1) => (
            simd_build16_v4,
            simd_gather64_v4_u1,
            simd_codes64_v4_u1,
            simd_codes64_v4_u1_u8,
        ),
        (8, 2) => (
            simd_build16_v4,
            simd_gather64_v4_u2,
            simd_codes64_v4_u2,
            simd_codes64_v4_u2_u8,
        ),
        _ => return None,
    };
    Some(TableSpec {
        backend: Backend::WasmSimd128,
        rows,
        group,
        k_group: 1,
        lanes_per_add: SIMD_TABLE_LANES_64,
        build_products_per_step: SIMD_TABLE_LANES_64,
        lane_cap: i64::MAX as u128,
        max_bound: u128::MAX,
        build_multiplies: true,
        build_adds: crate::table::product_build_adds,
        build,
        gather,
        gather_codes,
        gather_codes_u8,
    })
}

/// One slot of the `i16` table, at `V` registers of `i64`.
///
/// # Safety
///
/// [`crate::table::TableBuild`]'s contract, with `rows == V * 2` and `V` a
/// multiple of four.
unsafe fn simd_build16<const V: usize>(
    rows: usize,
    space: usize,
    block: usize,
    book: *const i16,
    acts: *const i16,
    out: *mut i64,
) {
    debug_assert_eq!(rows, V * SIMD_TABLE_LANES_64);
    debug_assert!(V.is_multiple_of(4));
    // SAFETY: the caller established every extent.
    unsafe {
        for c in 0..space {
            let d = book.add(c * block);
            let mut entry = [i64x2_splat(0); V];
            for t in 0..block {
                let w = i16x8_splat(*d.add(t));
                let a = acts.add(t * rows);
                // Eight activations per register, which is four registers of
                // the lane.
                for oct in 0..V / 4 {
                    let x = v128_load(a.add(oct * 8) as *const v128);
                    let lo = i32x4_extmul_low_i16x8(x, w);
                    let hi = i32x4_extmul_high_i16x8(x, w);
                    entry[oct * 4] = i64x2_add(entry[oct * 4], i64x2_extend_low_i32x4(lo));
                    entry[oct * 4 + 1] = i64x2_add(entry[oct * 4 + 1], i64x2_extend_high_i32x4(lo));
                    entry[oct * 4 + 2] = i64x2_add(entry[oct * 4 + 2], i64x2_extend_low_i32x4(hi));
                    entry[oct * 4 + 3] = i64x2_add(entry[oct * 4 + 3], i64x2_extend_high_i32x4(hi));
                }
            }
            let o = out.add(c * rows);
            for (v, cell) in entry.iter().enumerate() {
                v128_store(o.add(v * SIMD_TABLE_LANES_64) as *mut v128, *cell);
            }
        }
    }
}

/// One column group in the 64-bit lane.
///
/// # Safety
///
/// [`crate::table::TableGather`]'s contract, with `rows == V * 2`, `group == U`.
unsafe fn simd_gather64<const V: usize, const U: usize>(
    rows: usize,
    _group: usize,
    depth: usize,
    slab: usize,
    stack: *const i64,
    off: *const u32,
    lane: *mut i64,
) {
    debug_assert_eq!(rows, V * SIMD_TABLE_LANES_64);
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
        let mut acc = [[i64x2_splat(0); V]; U];
        for (u, cols) in acc.iter_mut().enumerate() {
            for (v, cell) in cols.iter_mut().enumerate() {
                *cell = v128_load(lane.add(u * rows + v * SIMD_TABLE_LANES_64) as *const v128);
            }
        }
        let mut words = stack;
        for slot in 0..depth {
            for (u, cols) in acc.iter_mut().enumerate() {
                let entry = words.add((*off.add(slot * U + u) & mask) as usize);
                for (v, cell) in cols.iter_mut().enumerate() {
                    *cell = i64x2_add(
                        *cell,
                        v128_load(entry.add(v * SIMD_TABLE_LANES_64) as *const v128),
                    );
                }
            }
            words = words.add(slab);
        }
        for (u, cols) in acc.iter().enumerate() {
            for (v, cell) in cols.iter().enumerate() {
                v128_store(
                    lane.add(u * rows + v * SIMD_TABLE_LANES_64) as *mut v128,
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
/// [`crate::table::TableGatherCodes`]'s contract, with `rows == V * 2`,
/// `group == U`.
#[allow(clippy::too_many_arguments)]
unsafe fn simd_codes64<const V: usize, const U: usize, K: Copy + Into<usize>>(
    rows: usize,
    _group: usize,
    depth: usize,
    slab: usize,
    shift: u32,
    stack: *const i64,
    codes: *const K,
    stride: usize,
    lane: *mut i64,
) {
    debug_assert_eq!(rows, V * SIMD_TABLE_LANES_64);
    debug_assert_eq!(_group, U);
    let mask = (slab >> shift) - 1;
    // SAFETY: the caller established every extent.
    unsafe {
        let mut acc = [[i64x2_splat(0); V]; U];
        for (u, cols) in acc.iter_mut().enumerate() {
            for (v, cell) in cols.iter_mut().enumerate() {
                *cell = v128_load(lane.add(u * rows + v * SIMD_TABLE_LANES_64) as *const v128);
            }
        }
        let mut cursor = [codes; U];
        for u in 1..U {
            cursor[u] = cursor[u - 1].add(stride);
        }
        let mut words = stack;
        for _ in 0..depth {
            for (u, cols) in acc.iter_mut().enumerate() {
                let code: usize = (*cursor[u]).into();
                let entry = words.add((code & mask) << shift);
                cursor[u] = cursor[u].add(1);
                for (v, cell) in cols.iter_mut().enumerate() {
                    *cell = i64x2_add(
                        *cell,
                        v128_load(entry.add(v * SIMD_TABLE_LANES_64) as *const v128),
                    );
                }
            }
            words = words.add(slab);
        }
        for (u, cols) in acc.iter().enumerate() {
            for (v, cell) in cols.iter().enumerate() {
                v128_store(
                    lane.add(u * rows + v * SIMD_TABLE_LANES_64) as *mut v128,
                    *cell,
                );
            }
        }
    }
}

/// # Safety
///
/// [`crate::table::TableBuild`]'s contract at `rows == 16`, 64-bit lane.
unsafe fn simd_build16_v8(
    rows: usize,
    space: usize,
    block: usize,
    book: *const i16,
    acts: *const i16,
    out: *mut i64,
) {
    // SAFETY: the caller forwarded the extents.
    unsafe { simd_build16::<8>(rows, space, block, book, acts, out) }
}

/// # Safety
///
/// [`crate::table::TableBuild`]'s contract at `rows == 8`, 64-bit lane.
unsafe fn simd_build16_v4(
    rows: usize,
    space: usize,
    block: usize,
    book: *const i16,
    acts: *const i16,
    out: *mut i64,
) {
    // SAFETY: the caller forwarded the extents.
    unsafe { simd_build16::<4>(rows, space, block, book, acts, out) }
}

/// Generate the four `(rows, group)` entry points for the 64-bit lane.
macro_rules! simd_gathers64 {
    ($($g:ident, $c:ident, $c8:ident, $v:expr, $u:expr, $rows:expr;)*) => {$(
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
            // SAFETY: the caller forwarded the extents.
            unsafe { simd_gather64::<$v, $u>(rows, group, depth, slab, stack, off, lane) }
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
            // SAFETY: the caller forwarded the extents.
            unsafe {
                simd_codes64::<$v, $u, u16>(rows, group, depth, slab, shift, stack, codes, stride, lane)
            }
        }

        #[doc = concat!("# Safety\n\n[`crate::table::TableGatherCodesU8`]'s contract at `rows == ", stringify!($rows), "`, `group == ", stringify!($u), "`.")]
        #[allow(clippy::too_many_arguments)]
        unsafe fn $c8(
            rows: usize,
            group: usize,
            depth: usize,
            slab: usize,
            shift: u32,
            stack: *const i64,
            codes: *const u8,
            stride: usize,
            lane: *mut i64,
        ) {
            // SAFETY: the caller forwarded the extents.
            unsafe {
                simd_codes64::<$v, $u, u8>(rows, group, depth, slab, shift, stack, codes, stride, lane)
            }
        }
    )*};
}

simd_gathers64! {
    simd_gather64_v8_u1, simd_codes64_v8_u1, simd_codes64_v8_u1_u8, 8, 1, 16;
    simd_gather64_v8_u2, simd_codes64_v8_u2, simd_codes64_v8_u2_u8, 8, 2, 16;
    simd_gather64_v4_u1, simd_codes64_v4_u1, simd_codes64_v4_u1_u8, 4, 1, 8;
    simd_gather64_v4_u2, simd_codes64_v4_u2, simd_codes64_v4_u2_u8, 4, 2, 8;
}

/// One slot of the table, at `V` registers of `i32`.
///
/// `i32x4_extmul_*_i16x8` is a widening multiply, so each product reaches the
/// `i32` lane with no `i16` intermediate to overflow.
///
/// # Safety
///
/// [`crate::table::TableBuild`]'s contract, with `rows == V * 4` and `V` even.
///
/// Parametric in `V`, as [`simd_build16`] is. It was hand-unrolled for `V` in
/// `{2, 4}` behind two `if V == 4` branches, with the real precondition stated
/// only in this comment: a third instantiation would have passed the one
/// assertion, written `entry[0..2]`, and left the rest of the table zeroed ---
/// silently, and in release. Sixteen activations is a whole register and four
/// registers of the lane, so the loop below walks them; eight is a half-load that
/// zeroes the rest, which is how a tile of eight never reads past its own row
/// (S13, and the reason `v128_load64_zero` is here).
unsafe fn simd_build<const V: usize>(
    rows: usize,
    space: usize,
    block: usize,
    book: *const i8,
    acts: *const i8,
    out: *mut i32,
) {
    debug_assert_eq!(rows, V * SIMD_TABLE_LANES);
    debug_assert!(V.is_multiple_of(2), "a register pair is the smallest step");
    // SAFETY: the caller established every extent.
    unsafe {
        for c in 0..space {
            let d = book.add(c * block);
            let mut entry = [i32x4_splat(0); V];
            for t in 0..block {
                let w = i16x8_splat(*d.add(t) as i16);
                let a = acts.add(t * rows);
                // The whole registers: sixteen activations, four lanes' worth.
                for quad in 0..V / 4 {
                    let reg = quad * 4;
                    let x = v128_load(a.add(quad * 16) as *const v128);
                    let lo = i16x8_extend_low_i8x16(x);
                    entry[reg] = i32x4_add(entry[reg], i32x4_extmul_low_i16x8(lo, w));
                    entry[reg + 1] = i32x4_add(entry[reg + 1], i32x4_extmul_high_i16x8(lo, w));
                    let hi = i16x8_extend_high_i8x16(x);
                    entry[reg + 2] = i32x4_add(entry[reg + 2], i32x4_extmul_low_i16x8(hi, w));
                    entry[reg + 3] = i32x4_add(entry[reg + 3], i32x4_extmul_high_i16x8(hi, w));
                }
                // An even `V` that is not a multiple of four leaves one pair, and
                // a half-load is what reads it without touching the next row.
                if !V.is_multiple_of(4) {
                    let reg = (V / 4) * 4;
                    let x = v128_load64_zero(a.add((V / 4) * 16) as *const u64);
                    let lo = i16x8_extend_low_i8x16(x);
                    entry[reg] = i32x4_add(entry[reg], i32x4_extmul_low_i16x8(lo, w));
                    entry[reg + 1] = i32x4_add(entry[reg + 1], i32x4_extmul_high_i16x8(lo, w));
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
unsafe fn simd_codes<const V: usize, const U: usize, K: Copy + Into<usize>>(
    rows: usize,
    _group: usize,
    depth: usize,
    slab: usize,
    shift: u32,
    stack: *const i32,
    codes: *const K,
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
                let code: usize = (*cursor[u]).into();
                let entry = words.add((code & mask) << shift);
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
    ($($g:ident, $c:ident, $c8:ident, $v:expr, $u:expr, $rows:expr;)*) => {$(
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
                simd_codes::<$v, $u, u16>(rows, group, depth, slab, shift, stack, codes, stride, lane)
            }
        }

        #[doc = concat!("# Safety\n\n[`crate::table::TableGatherCodesU8`]'s contract at `rows == ", stringify!($rows), "`, `group == ", stringify!($u), "`.")]
        #[allow(clippy::too_many_arguments)]
        unsafe fn $c8(
            rows: usize,
            group: usize,
            depth: usize,
            slab: usize,
            shift: u32,
            stack: *const i32,
            codes: *const u8,
            stride: usize,
            lane: *mut i32,
        ) {
            // SAFETY: the caller forwarded the extents.
            unsafe {
                simd_codes::<$v, $u, u8>(rows, group, depth, slab, shift, stack, codes, stride, lane)
            }
        }
    )*};
}

simd_gathers! {
    simd_gather_v4_u1, simd_codes_v4_u1, simd_codes_v4_u1_u8, 4, 1, 16;
    simd_gather_v4_u2, simd_codes_v4_u2, simd_codes_v4_u2_u8, 4, 2, 16;
    simd_gather_v2_u1, simd_codes_v2_u1, simd_codes_v2_u1_u8, 2, 1, 8;
    simd_gather_v2_u2, simd_codes_v2_u2, simd_codes_v2_u2_u8, 2, 2, 8;
}
