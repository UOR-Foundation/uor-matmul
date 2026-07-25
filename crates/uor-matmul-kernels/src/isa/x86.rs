//! x86-64: AVX2 for every integer family, and AVX-512 VNNI for `i8` (§7.2).

use core::arch::x86_64::*;

use uor_matmul_core::Backend;

use crate::spec::{Factorization, KernelSpec};

crate::tile_fits!(6, 16);
crate::tile_fits!(4, 8);
crate::tile_fits!(8, 16);

/// Is AVX2 available?
pub fn avx2_available() -> bool {
    #[cfg(any(feature = "std", test))]
    {
        std::arch::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(any(feature = "std", test)))]
    {
        // Without `std` there is nothing to detect, so the answer is whatever
        // the target features said at compile time (C1).
        cfg!(target_feature = "avx2")
    }
}

/// Is AVX-512 with VNNI available?
pub fn avx512vnni_available() -> bool {
    #[cfg(any(feature = "std", test))]
    {
        std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vnni")
    }
    #[cfg(not(any(feature = "std", test)))]
    {
        cfg!(target_feature = "avx512vnni")
    }
}

// ---------------------------------------------------------------------------
// i8 x i8 -> i32
// ---------------------------------------------------------------------------

const A2_I8_MR: usize = 6;
const A2_I8_NR: usize = 16;

/// AVX2 `i8`: widen with `cvtepi8_epi16`, then `madd` a pair of `k`-steps into
/// an `i32` lane.
///
/// `madd` on i8-derived i16 peaks at `2 * 128 * 128 = 32768`, far inside `i32`,
/// so the pairing costs nothing in reach: the lane fills at the same depth the
/// plain `i32` tile does.
pub const AVX2_I8_I32: KernelSpec<i8, i32> = KernelSpec {
    backend: Backend::Avx2,
    factorization: Factorization::Exact,
    mr: A2_I8_MR,
    nr: A2_I8_NR,
    k_group: 2,
    lane_cap: i32::MAX as u128,
    mac_tile: avx2_i8,
};

/// # Safety
///
/// `pa` must have `6 * kc` readable elements, `pb` `16 * kc`, `acc` 96 writable
/// lanes, and the host must have `avx2`.
unsafe fn avx2_i8(kc: usize, pa: *const i8, pb: *const i8, acc: *mut i32) {
    // SAFETY: the caller established `avx2` and forwarded the lengths.
    unsafe { avx2_i8_inner(kc, pa, pb, acc) }
}

/// # Safety
///
/// As [`avx2_i8`].
#[target_feature(enable = "avx2")]
unsafe fn avx2_i8_inner(kc: usize, pa: *const i8, pb: *const i8, acc: *mut i32) {
    const MR: usize = A2_I8_MR;
    const NR: usize = A2_I8_NR;
    // SAFETY: the caller guaranteed the three extents. One conversion here
    // keeps every panel read below safe, so the only remaining `unsafe` is the
    // vector loads and stores, which is where it belongs.
    let (pa, pb, acc) = unsafe {
        (
            core::slice::from_raw_parts(pa, MR * kc),
            core::slice::from_raw_parts(pb, NR * kc),
            core::slice::from_raw_parts_mut(acc, MR * NR),
        )
    };
    // Two `__m256i` per row. Their lanes are *permuted*: `madd` consumes the
    // interleave `unpack` produces, and `unpack` works within 128-bit halves,
    // so `lo` holds columns 0..3 and 8..11 and `hi` holds 4..7 and 12..15. The
    // permutation is undone once per tile at the store, rather than being
    // avoided by interleaving `B` a scalar element at a time --- which is what
    // this kernel used to do, and it cost sixteen stores per `k`-step.
    let mut lo = [_mm256_setzero_si256(); MR];
    let mut hi = [_mm256_setzero_si256(); MR];

    let pairs = kc / 2;
    for q in 0..pairs {
        let (p0, p1) = (q * 2, q * 2 + 1);
        // SAFETY: `pb[p * NR ..][..16]` is in bounds; each is one 128-bit load
        // widened to sixteen `i16`.
        let (b0, b1) = unsafe {
            (
                _mm256_cvtepi8_epi16(_mm_loadu_si128(pb.as_ptr().add(p0 * NR).cast::<__m128i>())),
                _mm256_cvtepi8_epi16(_mm_loadu_si128(pb.as_ptr().add(p1 * NR).cast::<__m128i>())),
            )
        };
        // Lane `j` of the result holds the pair `(b[p0][j], b[p1][j])`, which is
        // what `madd` sums.
        let bv_lo = _mm256_unpacklo_epi16(b0, b1);
        let bv_hi = _mm256_unpackhi_epi16(b0, b1);

        for i in 0..MR {
            let a0 = i16::from(pa[p0 * MR + i]);
            let a1 = i16::from(pa[p1 * MR + i]);
            let av = _mm256_set1_epi32(((a1 as u16 as u32) << 16 | (a0 as u16 as u32)) as i32);
            // Exact: `|a0*b0 + a1*b1| <= 2 * 128 * 128`, far inside `i32`.
            lo[i] = _mm256_add_epi32(lo[i], _mm256_madd_epi16(av, bv_lo));
            hi[i] = _mm256_add_epi32(hi[i], _mm256_madd_epi16(av, bv_hi));
        }
    }

    // The `k`-tail, one step at a time, in the same permuted lane order. Zero
    // padding would have been exact too; walking the tail is simply cheaper
    // than materialising a padded panel.
    for p in (pairs * 2)..kc {
        // SAFETY: `pb[p * NR ..][..16]` is in bounds.
        let b0 = unsafe {
            _mm256_cvtepi8_epi16(_mm_loadu_si128(pb.as_ptr().add(p * NR).cast::<__m128i>()))
        };
        let zero = _mm256_setzero_si256();
        let bv_lo = _mm256_unpacklo_epi16(b0, zero);
        let bv_hi = _mm256_unpackhi_epi16(b0, zero);
        for i in 0..MR {
            let av = _mm256_set1_epi32(i32::from(pa[p * MR + i]));
            lo[i] = _mm256_add_epi32(lo[i], _mm256_madd_epi16(av, bv_lo));
            hi[i] = _mm256_add_epi32(hi[i], _mm256_madd_epi16(av, bv_hi));
        }
    }

    for i in 0..MR {
        // Undo the permutation: `lo` holds j 0..3 and 8..11, `hi` holds 4..7
        // and 12..15, so the low halves make j 0..7 and the high halves j 8..15.
        let j0_7 = _mm256_permute2x128_si256(lo[i], hi[i], 0x20);
        let j8_15 = _mm256_permute2x128_si256(lo[i], hi[i], 0x31);
        // SAFETY: `i < MR`, so these two stores land inside `MR * NR`.
        unsafe {
            _mm256_storeu_si256(acc.as_mut_ptr().add(i * NR).cast::<__m256i>(), j0_7);
            _mm256_storeu_si256(acc.as_mut_ptr().add(i * NR + 8).cast::<__m256i>(), j8_15);
        }
    }
}

// ---------------------------------------------------------------------------
// i16 x i16 -> i64
// ---------------------------------------------------------------------------

const A2_I16_MR: usize = 4;
const A2_I16_NR: usize = 8;

/// AVX2 `i16`: `madd` is exactly this family's arithmetic.
///
/// `_mm256_madd_epi16` multiplies signed words and sums adjacent pairs into an
/// `i32`. A pair of full-range `i16` products reaches `2 * 2^30`, which is one
/// bit past `i32` --- so the pairs are widened to `i64` before accumulating,
/// and the lane is exact at every depth the driver offers.
pub const AVX2_I16_I64: KernelSpec<i16, i64> = KernelSpec {
    backend: Backend::Avx2,
    factorization: Factorization::Exact,
    mr: A2_I16_MR,
    nr: A2_I16_NR,
    k_group: 2,
    lane_cap: i64::MAX as u128,
    mac_tile: avx2_i16,
};

/// # Safety
///
/// `pa` must have `4 * kc` readable elements, `pb` `8 * kc`, `acc` 32 writable
/// lanes, and the host must have `avx2`.
unsafe fn avx2_i16(kc: usize, pa: *const i16, pb: *const i16, acc: *mut i64) {
    // SAFETY: the caller established `avx2` and forwarded the lengths.
    unsafe { avx2_i16_inner(kc, pa, pb, acc) }
}

/// # Safety
///
/// As [`avx2_i16`].
#[target_feature(enable = "avx2")]
unsafe fn avx2_i16_inner(kc: usize, pa: *const i16, pb: *const i16, acc: *mut i64) {
    const MR: usize = A2_I16_MR;
    const NR: usize = A2_I16_NR;
    // SAFETY: the caller guaranteed the three extents.
    let (pa, pb, acc) = unsafe {
        (
            core::slice::from_raw_parts(pa, MR * kc),
            core::slice::from_raw_parts(pb, NR * kc),
            core::slice::from_raw_parts_mut(acc, MR * NR),
        )
    };
    // Two `__m256i` of four i64 lanes each cover the eight columns.
    let mut tile = [[_mm256_setzero_si256(); 2]; MR];

    let pairs = kc / 2;
    for q in 0..pairs {
        let (p0, p1) = (q * 2, q * 2 + 1);
        let mut b_pairs = [0i16; NR * 2];
        for j in 0..NR {
            b_pairs[j * 2] = pb[p0 * NR + j];
            b_pairs[j * 2 + 1] = pb[p1 * NR + j];
        }
        // SAFETY: `b_pairs` holds 16 i16, exactly one 256-bit load.
        let bv = unsafe { _mm256_loadu_si256(b_pairs.as_ptr().cast::<__m256i>()) };

        for (i, row) in tile.iter_mut().enumerate() {
            let a0 = pa[p0 * MR + i];
            let a1 = pa[p1 * MR + i];
            let av = _mm256_set1_epi32(((a1 as u16 as u32) << 16 | (a0 as u16 as u32)) as i32);
            // Eight i32 pair-sums, each exact and at most 2^31.
            let m = _mm256_madd_epi16(av, bv);
            // Widen to i64 before accumulating, so no depth can fill the lane.
            row[0] = _mm256_add_epi64(row[0], _mm256_cvtepi32_epi64(_mm256_castsi256_si128(m)));
            row[1] = _mm256_add_epi64(
                row[1],
                _mm256_cvtepi32_epi64(_mm256_extracti128_si256(m, 1)),
            );
        }
    }

    for p in (pairs * 2)..kc {
        for (i, row) in tile.iter_mut().enumerate() {
            let a = i64::from(pa[p * MR + i]);
            let mut lane = [0i64; NR];
            for (j, slot) in lane.iter_mut().enumerate() {
                *slot = a * i64::from(pb[p * NR + j]);
            }
            // SAFETY: `lane` holds 8 i64, exactly two 256-bit loads.
            let (l0, l1) = unsafe {
                (
                    _mm256_loadu_si256(lane.as_ptr().cast::<__m256i>()),
                    _mm256_loadu_si256(lane.as_ptr().add(4).cast::<__m256i>()),
                )
            };
            row[0] = _mm256_add_epi64(row[0], l0);
            row[1] = _mm256_add_epi64(row[1], l1);
        }
    }

    for (i, row) in tile.iter().enumerate() {
        // SAFETY: `i < MR`, so these two stores land inside `MR * NR`.
        unsafe {
            _mm256_storeu_si256(acc.as_mut_ptr().add(i * NR).cast::<__m256i>(), row[0]);
            _mm256_storeu_si256(acc.as_mut_ptr().add(i * NR + 4).cast::<__m256i>(), row[1]);
        }
    }
}

// ---------------------------------------------------------------------------
// i32 x i32 -> i64, exact
// ---------------------------------------------------------------------------

const A2_I32_MR: usize = 4;
const A2_I32_NR: usize = 8;

/// AVX2 `i32`, exact: `_mm256_mul_epi32` is a signed `32x32 -> 64` multiply,
/// which is this family's whole arithmetic in one instruction.
///
/// It multiplies the *even* 32-bit lanes, so the odd ones are reached by
/// shifting both operands down 32 bits --- two multiplies for eight products,
/// which is the arrangement AVX2 offers.
pub const AVX2_I32_I64: KernelSpec<i32, i64> = KernelSpec {
    backend: Backend::Avx2,
    factorization: Factorization::Exact,
    mr: A2_I32_MR,
    nr: A2_I32_NR,
    k_group: 1,
    lane_cap: i64::MAX as u128,
    mac_tile: avx2_i32_exact,
};

/// # Safety
///
/// `pa` must have `4 * kc` readable elements, `pb` `8 * kc`, `acc` 32 writable
/// lanes, and the host must have `avx2`.
unsafe fn avx2_i32_exact(kc: usize, pa: *const i32, pb: *const i32, acc: *mut i64) {
    // SAFETY: the caller established `avx2` and forwarded the lengths.
    unsafe { avx2_i32_exact_inner(kc, pa, pb, acc) }
}

/// # Safety
///
/// As [`avx2_i32_exact`].
#[target_feature(enable = "avx2")]
unsafe fn avx2_i32_exact_inner(kc: usize, pa: *const i32, pb: *const i32, acc: *mut i64) {
    const MR: usize = A2_I32_MR;
    const NR: usize = A2_I32_NR;
    // SAFETY: the caller guaranteed the three extents.
    let (pa, pb, acc) = unsafe {
        (
            core::slice::from_raw_parts(pa, MR * kc),
            core::slice::from_raw_parts(pb, NR * kc),
            core::slice::from_raw_parts_mut(acc, MR * NR),
        )
    };
    // `even[i]` holds columns 0, 2, 4, 6 and `odd[i]` holds 1, 3, 5, 7, which
    // is the lane order `mul_epi32` produces. They are interleaved back on the
    // way out rather than on every step.
    let mut even = [_mm256_setzero_si256(); MR];
    let mut odd = [_mm256_setzero_si256(); MR];

    for p in 0..kc {
        // SAFETY: `pb[p * NR ..][..8]` is in bounds, and this is one 256-bit
        // load of eight i32.
        let bv = unsafe { _mm256_loadu_si256(pb.as_ptr().add(p * NR).cast::<__m256i>()) };
        let bv_odd = _mm256_srli_epi64(bv, 32);
        for i in 0..MR {
            let av = _mm256_set1_epi32(pa[p * MR + i]);
            let av_odd = _mm256_srli_epi64(av, 32);
            // Exact: each product needs at most 62 bits and lands in an i64.
            even[i] = _mm256_add_epi64(even[i], _mm256_mul_epi32(av, bv));
            odd[i] = _mm256_add_epi64(odd[i], _mm256_mul_epi32(av_odd, bv_odd));
        }
    }

    for i in 0..MR {
        let mut e = [0i64; 4];
        let mut o = [0i64; 4];
        // SAFETY: both destinations hold exactly four i64.
        unsafe {
            _mm256_storeu_si256(e.as_mut_ptr().cast::<__m256i>(), even[i]);
            _mm256_storeu_si256(o.as_mut_ptr().cast::<__m256i>(), odd[i]);
        }
        for c in 0..4 {
            acc[i * NR + c * 2] = e[c];
            acc[i * NR + c * 2 + 1] = o[c];
        }
    }
}

// ---------------------------------------------------------------------------
// i32 x i32 -> i32, modular
// ---------------------------------------------------------------------------

/// AVX2 `i32` in `Z/2^32`: `_mm256_mullo_epi32` gives the low half of eight
/// products at once, which is the whole of the arithmetic in the quotient.
///
/// Eight products per instruction against the exact kernel's four, and no
/// widening, because in `Z/2^32` there is nothing to widen *to*. This is what
/// the caller gets for asking to encode by wrapping.
pub const AVX2_I32_MOD: KernelSpec<i32, i32> = KernelSpec {
    backend: Backend::Avx2,
    factorization: Factorization::Modular,
    mr: 6,
    nr: 16,
    k_group: 1,
    lane_cap: 0,
    mac_tile: avx2_i32_mod,
};

/// # Safety
///
/// `pa` must have `6 * kc` readable elements, `pb` `16 * kc`, `acc` 96 writable
/// lanes, and the host must have `avx2`.
unsafe fn avx2_i32_mod(kc: usize, pa: *const i32, pb: *const i32, acc: *mut i32) {
    // SAFETY: the caller established `avx2` and forwarded the lengths.
    unsafe { avx2_i32_mod_inner(kc, pa, pb, acc) }
}

/// # Safety
///
/// As [`avx2_i32_mod`].
#[target_feature(enable = "avx2")]
unsafe fn avx2_i32_mod_inner(kc: usize, pa: *const i32, pb: *const i32, acc: *mut i32) {
    const MR: usize = 6;
    const NR: usize = 16;
    // SAFETY: the caller guaranteed the three extents.
    let (pa, pb, acc) = unsafe {
        (
            core::slice::from_raw_parts(pa, MR * kc),
            core::slice::from_raw_parts(pb, NR * kc),
            core::slice::from_raw_parts_mut(acc, MR * NR),
        )
    };
    let mut tile = [[_mm256_setzero_si256(); 2]; MR];

    for p in 0..kc {
        // SAFETY: `pb[p * NR ..][..16]` is in bounds: two 256-bit loads.
        let (bv0, bv1) = unsafe {
            (
                _mm256_loadu_si256(pb.as_ptr().add(p * NR).cast::<__m256i>()),
                _mm256_loadu_si256(pb.as_ptr().add(p * NR + 8).cast::<__m256i>()),
            )
        };
        for (i, row) in tile.iter_mut().enumerate() {
            let av = _mm256_set1_epi32(pa[p * MR + i]);
            // `mullo` keeps the low 32 bits, and the add wraps. Both are the
            // ring operations of `Z/2^32`, so the lane holds the exact value
            // the caller asked to encode into.
            row[0] = _mm256_add_epi32(row[0], _mm256_mullo_epi32(av, bv0));
            row[1] = _mm256_add_epi32(row[1], _mm256_mullo_epi32(av, bv1));
        }
    }

    for (i, row) in tile.iter().enumerate() {
        // SAFETY: `i < MR`, so these two stores land inside `MR * NR`.
        unsafe {
            _mm256_storeu_si256(acc.as_mut_ptr().add(i * NR).cast::<__m256i>(), row[0]);
            _mm256_storeu_si256(acc.as_mut_ptr().add(i * NR + 8).cast::<__m256i>(), row[1]);
        }
    }
}

// ---------------------------------------------------------------------------
// i16 x i16 -> i32, modular
// ---------------------------------------------------------------------------

/// AVX2 `i16` in `Z/2^32`: `madd` lands in `i32` and stays there.
///
/// Twice the columns of the exact `i16` kernel per instruction, because in the
/// quotient there is nothing to widen to. This is what the caller gets for
/// asking to encode by wrapping.
pub const AVX2_I16_MOD: KernelSpec<i16, i32> = KernelSpec {
    backend: Backend::Avx2,
    factorization: Factorization::Modular,
    mr: 6,
    nr: 16,
    k_group: 2,
    lane_cap: 0,
    mac_tile: avx2_i16_mod,
};

/// # Safety
///
/// `pa` must have `6 * kc` readable elements, `pb` `16 * kc`, `acc` 96 writable
/// lanes, and the host must have `avx2`.
unsafe fn avx2_i16_mod(kc: usize, pa: *const i16, pb: *const i16, acc: *mut i32) {
    // SAFETY: the caller established `avx2` and forwarded the lengths.
    unsafe { avx2_i16_mod_inner(kc, pa, pb, acc) }
}

/// # Safety
///
/// As [`avx2_i16_mod`].
#[target_feature(enable = "avx2")]
unsafe fn avx2_i16_mod_inner(kc: usize, pa: *const i16, pb: *const i16, acc: *mut i32) {
    const MR: usize = 6;
    const NR: usize = 16;
    // SAFETY: the caller guaranteed the three extents.
    let (pa, pb, acc) = unsafe {
        (
            core::slice::from_raw_parts(pa, MR * kc),
            core::slice::from_raw_parts(pb, NR * kc),
            core::slice::from_raw_parts_mut(acc, MR * NR),
        )
    };
    let mut tile = [[_mm256_setzero_si256(); 2]; MR];

    let pairs = kc / 2;
    for q in 0..pairs {
        let (p0, p1) = (q * 2, q * 2 + 1);
        let mut b_pairs = [0i16; NR * 2];
        for j in 0..NR {
            b_pairs[j * 2] = pb[p0 * NR + j];
            b_pairs[j * 2 + 1] = pb[p1 * NR + j];
        }
        // SAFETY: `b_pairs` holds 32 i16 = two 256-bit loads.
        let (bv0, bv1) = unsafe {
            (
                _mm256_loadu_si256(b_pairs.as_ptr().cast::<__m256i>()),
                _mm256_loadu_si256(b_pairs.as_ptr().add(16).cast::<__m256i>()),
            )
        };
        for (i, row) in tile.iter_mut().enumerate() {
            let a0 = pa[p0 * MR + i];
            let a1 = pa[p1 * MR + i];
            let av = _mm256_set1_epi32(((a1 as u16 as u32) << 16 | (a0 as u16 as u32)) as i32);
            // `madd` and the add both wrap, and both are the ring operations of
            // `Z/2^32`, so the lane holds the value the caller asked to encode.
            row[0] = _mm256_add_epi32(row[0], _mm256_madd_epi16(av, bv0));
            row[1] = _mm256_add_epi32(row[1], _mm256_madd_epi16(av, bv1));
        }
    }

    for p in (pairs * 2)..kc {
        for (i, row) in tile.iter_mut().enumerate() {
            let a = i32::from(pa[p * MR + i]);
            let mut lane = [0i32; NR];
            for (j, slot) in lane.iter_mut().enumerate() {
                *slot = a.wrapping_mul(i32::from(pb[p * NR + j]));
            }
            // SAFETY: `lane` holds 16 i32 = two 256-bit loads.
            let (l0, l1) = unsafe {
                (
                    _mm256_loadu_si256(lane.as_ptr().cast::<__m256i>()),
                    _mm256_loadu_si256(lane.as_ptr().add(8).cast::<__m256i>()),
                )
            };
            row[0] = _mm256_add_epi32(row[0], l0);
            row[1] = _mm256_add_epi32(row[1], l1);
        }
    }

    for (i, row) in tile.iter().enumerate() {
        // SAFETY: `i < MR`, so these two stores land inside `MR * NR`.
        unsafe {
            _mm256_storeu_si256(acc.as_mut_ptr().add(i * NR).cast::<__m256i>(), row[0]);
            _mm256_storeu_si256(acc.as_mut_ptr().add(i * NR + 8).cast::<__m256i>(), row[1]);
        }
    }
}

// ---------------------------------------------------------------------------
// AVX-512 VNNI, i8 x i8 -> i32
// ---------------------------------------------------------------------------

const V_MR: usize = 8;
const V_NR: usize = 16;

/// The signed-word VNNI sequence. Reaches further per lane than the offset one
/// and needs no compensation term.
pub const AVX512_DPWSSD_I8_I32: KernelSpec<i8, i32> = KernelSpec {
    backend: Backend::Avx512Vnni,
    factorization: Factorization::Exact,
    mr: V_MR,
    nr: V_NR,
    k_group: 2,
    lane_cap: i32::MAX as u128,
    mac_tile: vnni_dpwssd,
};

/// The offset VNNI sequence: four bytes per lane instead of two.
///
/// `vpdpbusd` multiplies **unsigned** bytes by signed bytes, so reaching it
/// from `i8 x i8` needs the offset identity
///
/// ```text
/// sum(a_i8 * b) = sum((a_i8 + 128) * b) - 128 * sum(b)
/// ```
///
/// with `sum(b)` accumulated per column. Both terms are exact integers, so the
/// result is still the exact accumulation. The *intermediates* are not free:
/// the offset term reaches `255 * 128` per step against `dpwssd`'s `128 * 128`,
/// so it fills its lane sooner --- a threshold on a register rather than on an
/// answer, which is why `lane_cap` says so and the driver simply chunks more.
pub const AVX512_DPBUSD_I8_I32: KernelSpec<i8, i32> = KernelSpec {
    backend: Backend::Avx512Vnni,
    factorization: Factorization::Exact,
    mr: V_MR,
    nr: V_NR,
    k_group: 4,
    lane_cap: (i32::MAX as u128) / 255 * 128,
    mac_tile: vnni_dpbusd,
};

/// # Safety
///
/// `pa` must have `8 * kc` readable elements, `pb` `16 * kc`, `acc` 128
/// writable lanes, and the host must have `avx512f`, `avx512bw`, `avx512vnni`.
unsafe fn vnni_dpwssd(kc: usize, pa: *const i8, pb: *const i8, acc: *mut i32) {
    // SAFETY: the caller established the target features and the lengths.
    unsafe { dpwssd_inner(kc, pa, pb, acc) }
}

/// # Safety
///
/// As [`vnni_dpwssd`].
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn dpwssd_inner(kc: usize, pa: *const i8, pb: *const i8, acc: *mut i32) {
    // SAFETY: the caller guaranteed the three extents.
    let (pa, pb, acc) = unsafe {
        (
            core::slice::from_raw_parts(pa, V_MR * kc),
            core::slice::from_raw_parts(pb, V_NR * kc),
            core::slice::from_raw_parts_mut(acc, V_MR * V_NR),
        )
    };
    let mut tile = [_mm512_setzero_si512(); V_MR];
    let pairs = kc / 2;

    for q in 0..pairs {
        let (p0, p1) = (q * 2, q * 2 + 1);
        let mut b_pairs = [0i16; V_NR * 2];
        for j in 0..V_NR {
            b_pairs[j * 2] = i16::from(pb[p0 * V_NR + j]);
            b_pairs[j * 2 + 1] = i16::from(pb[p1 * V_NR + j]);
        }
        // SAFETY: `b_pairs` holds 32 i16 = 512 bits.
        let bv = unsafe { _mm512_loadu_si512(b_pairs.as_ptr().cast()) };
        for (i, lane) in tile.iter_mut().enumerate() {
            let a0 = i16::from(pa[p0 * V_MR + i]);
            let a1 = i16::from(pa[p1 * V_MR + i]);
            let av = _mm512_set1_epi32(((a1 as u16 as u32) << 16 | (a0 as u16 as u32)) as i32);
            *lane = _mm512_dpwssd_epi32(*lane, av, bv);
        }
    }

    // SAFETY: same features, same lengths, `pairs * 2 <= kc`.
    unsafe { vnni_tail(kc, pairs * 2, pa, pb, &mut tile) };
    // SAFETY: `acc` has `V_MR * V_NR` lanes.
    unsafe { vnni_store(acc, &tile) };
}

/// # Safety
///
/// As [`vnni_dpwssd`].
unsafe fn vnni_dpbusd(kc: usize, pa: *const i8, pb: *const i8, acc: *mut i32) {
    // SAFETY: the caller established the target features and the lengths.
    unsafe { dpbusd_inner(kc, pa, pb, acc) }
}

/// # Safety
///
/// As [`vnni_dpwssd`].
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn dpbusd_inner(kc: usize, pa: *const i8, pb: *const i8, acc: *mut i32) {
    const G: usize = 4;
    // SAFETY: the caller guaranteed the three extents.
    let (pa, pb, acc) = unsafe {
        (
            core::slice::from_raw_parts(pa, V_MR * kc),
            core::slice::from_raw_parts(pb, V_NR * kc),
            core::slice::from_raw_parts_mut(acc, V_MR * V_NR),
        )
    };
    let mut tile = [_mm512_setzero_si512(); V_MR];
    let mut colsum = [0i32; V_NR];

    let groups = kc / G;
    for q in 0..groups {
        let base = q * G;
        let mut b_quads = [0u8; V_NR * G];
        for j in 0..V_NR {
            for g in 0..G {
                let v = pb[(base + g) * V_NR + j];
                b_quads[j * G + g] = v as u8;
                colsum[j] = colsum[j].wrapping_add(i32::from(v));
            }
        }
        // SAFETY: `b_quads` holds 64 bytes = 512 bits.
        let bv = unsafe { _mm512_loadu_si512(b_quads.as_ptr().cast()) };
        for (i, lane) in tile.iter_mut().enumerate() {
            let mut a_quad = [0u8; 4];
            for (g, slot) in a_quad.iter_mut().enumerate() {
                // `a + 128` is exactly `a as u8` with its top bit flipped,
                // which is why the identity costs no arithmetic here.
                *slot = (pa[(base + g) * V_MR + i] as u8) ^ 0x80;
            }
            *lane = _mm512_dpbusd_epi32(*lane, _mm512_set1_epi32(i32::from_le_bytes(a_quad)), bv);
        }
    }

    // The compensation, then the tail in the plain sequence. Both are exact
    // integers, so the total is still the exact accumulation.
    // SAFETY: `colsum` holds 16 i32 = 512 bits.
    let comp = unsafe { _mm512_loadu_si512(colsum.as_ptr().cast()) };
    let scaled = _mm512_mullo_epi32(comp, _mm512_set1_epi32(128));
    for lane in tile.iter_mut() {
        *lane = _mm512_sub_epi32(*lane, scaled);
    }

    // SAFETY: same features, same lengths, `groups * G <= kc`.
    unsafe { vnni_tail(kc, groups * G, pa, pb, &mut tile) };
    // SAFETY: `acc` has `V_MR * V_NR` lanes.
    unsafe { vnni_store(acc, &tile) };
}

/// The `k`-tail, one step at a time, in the plain sequence.
///
/// # Safety
///
/// The host must have `avx512f`, `avx512bw`, `avx512vnni`.
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn vnni_tail(kc: usize, from: usize, pa: &[i8], pb: &[i8], tile: &mut [__m512i; V_MR]) {
    for p in from..kc {
        let mut lane = [0i32; V_NR];
        for (j, slot) in lane.iter_mut().enumerate() {
            *slot = i32::from(pb[p * V_NR + j]);
        }
        // SAFETY: `lane` holds 16 i32 = 512 bits.
        let bv = unsafe { _mm512_loadu_si512(lane.as_ptr().cast()) };
        for (i, t) in tile.iter_mut().enumerate() {
            let a = i32::from(pa[p * V_MR + i]);
            *t = _mm512_add_epi32(*t, _mm512_mullo_epi32(bv, _mm512_set1_epi32(a)));
        }
    }
}

/// # Safety
///
/// `acc` must have `V_MR * V_NR` writable lanes, and the host must have
/// `avx512f`.
#[target_feature(enable = "avx512f")]
unsafe fn vnni_store(acc: &mut [i32], tile: &[__m512i; V_MR]) {
    for (i, lane) in tile.iter().enumerate() {
        // SAFETY: `i < V_MR`, so this 512-bit store lands inside `MR * NR`.
        unsafe { _mm512_storeu_si512(acc.as_mut_ptr().add(i * V_NR).cast(), *lane) };
    }
}
