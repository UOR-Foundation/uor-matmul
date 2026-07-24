//! x86-64 AVX2 (§7.2).
//!
//! Widen with `_mm256_cvtepi8_epi16`, then `_mm256_madd_epi16` a pair of
//! `k`-steps into an `i32` lane and add. `madd` on i8-derived i16 peaks at
//! `2 * 128 * 128 = 32768`, far inside `i32`, so the pairing costs nothing in
//! reach: the lane fills at the same depth the plain `i32` tile does.

use core::arch::x86_64::*;

use uor_matmul_core::Backend;

use crate::spec::KernelSpec;

/// Rows of `C` per call.
pub const MR: usize = 6;
/// Columns of `C` per call.
pub const NR: usize = 16;
/// `madd` consumes two `k`-steps at a time.
pub const K_GROUP: usize = 2;

/// The AVX2 spec.
pub const SPEC: KernelSpec = KernelSpec {
    backend: Backend::Avx2,
    mr: MR,
    nr: NR,
    k_group: K_GROUP,
    lane_cap: i32::MAX as u128,
    mac_tile,
};

/// Can this host run it?
pub fn is_available() -> bool {
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

/// Accumulate a `6 x 16` tile.
///
/// # Safety
///
/// `pa` must have `MR * kc` readable elements, `pb` must have `NR * kc`, `acc`
/// must have `MR * NR` writable lanes, and the host must have `avx2`.
/// [`KernelSpec::mac_tile`] establishes the lengths and
/// [`crate::available`] establishes the feature.
unsafe fn mac_tile(kc: usize, pa: *const i8, pb: *const i8, acc: *mut i32) {
    // SAFETY: the caller established `avx2`, which is this function's only
    // target-feature precondition, and forwarded the length guarantees.
    unsafe { mac_tile_avx2(kc, pa, pb, acc) }
}

/// # Safety
///
/// As [`mac_tile`], and the host must have `avx2`.
#[target_feature(enable = "avx2")]
unsafe fn mac_tile_avx2(kc: usize, pa: *const i8, pb: *const i8, acc: *mut i32) {
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

    // Two `__m256i` of eight i32 lanes each cover the sixteen columns.
    let mut tile = [[_mm256_setzero_si256(); 2]; MR];

    let pairs = kc / K_GROUP;
    for q in 0..pairs {
        let (p0, p1) = (q * K_GROUP, q * K_GROUP + 1);

        // Interleave the two k-steps of B so that `madd` sees adjacent pairs:
        // lane j holds (b[p0][j], b[p1][j]).
        let mut b_lo = [0i16; NR * K_GROUP];
        for j in 0..NR {
            b_lo[j * 2] = pb[p0 * NR + j] as i16;
            b_lo[j * 2 + 1] = pb[p1 * NR + j] as i16;
        }
        // SAFETY: `b_lo` holds `NR * 2 = 32` i16, which is exactly the two
        // 256-bit loads below.
        let (bv0, bv1) = unsafe {
            (
                _mm256_loadu_si256(b_lo.as_ptr().cast::<__m256i>()),
                _mm256_loadu_si256(b_lo.as_ptr().add(16).cast::<__m256i>()),
            )
        };

        for (i, row) in tile.iter_mut().enumerate() {
            let a0 = pa[p0 * MR + i] as i16;
            let a1 = pa[p1 * MR + i] as i16;
            // Broadcast the pair (a0, a1) so it aligns with B's interleaving.
            let av = _mm256_set1_epi32(((a1 as u16 as u32) << 16 | (a0 as u16 as u32)) as i32);
            // madd: (a0*b[p0][j] + a1*b[p1][j]) into one i32 lane, exactly.
            row[0] = _mm256_add_epi32(row[0], _mm256_madd_epi16(av, bv0));
            row[1] = _mm256_add_epi32(row[1], _mm256_madd_epi16(av, bv1));
        }
    }

    // The k-tail, one step at a time. Zero padding would have been exact too;
    // walking the tail is simply cheaper than materialising a padded panel.
    for p in (pairs * K_GROUP)..kc {
        for (i, row) in tile.iter_mut().enumerate() {
            let a = pa[p * MR + i] as i32;
            let mut lane = [0i32; NR];
            for (j, slot) in lane.iter_mut().enumerate() {
                *slot = a.wrapping_mul(pb[p * NR + j] as i32);
            }
            // SAFETY: `lane` holds `NR = 16` i32, exactly two 256-bit loads.
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
        // SAFETY: `acc` has `MR * NR` lanes and `i < MR`, so these two 256-bit
        // stores land inside it.
        unsafe {
            _mm256_storeu_si256(acc.as_mut_ptr().add(i * NR).cast::<__m256i>(), row[0]);
            _mm256_storeu_si256(acc.as_mut_ptr().add(i * NR + 8).cast::<__m256i>(), row[1]);
        }
    }
}
