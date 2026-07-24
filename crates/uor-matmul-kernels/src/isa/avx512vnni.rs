//! x86-64 AVX-512 with VNNI (§7.2, §7.3).
//!
//! Two sequences, three thresholds, one answer.
//!
//! `vpdpwssd` multiplies **signed words** and is reached directly from an i8
//! widen; it consumes two bytes per lane. `vpdpbusd` multiplies **unsigned**
//! bytes by signed bytes and consumes four, which is why it is worth reaching
//! at all --- but reaching it from `i8 x i8` needs the offset identity
//!
//! ```text
//! sum(a_i8 * b) = sum((a_i8 + 128) * b) - 128 * sum(b)
//! ```
//!
//! with `sum(b)` precomputed per column. Both terms are exact integers, so the
//! final result is still the exact accumulation and bit-identity is preserved.
//! The *intermediates* are not free: the offset term reaches `255 * 128` per
//! step against `dpwssd`'s `128 * 128`, so it fills its lane sooner. That is
//! the whole content of §7.3, and it is a threshold on a register rather than
//! on an answer: past it the driver hands the tile to `dpwssd`, and past
//! *that* to the wide accumulator (§5.1, `CU-03`).
//!
//! Neither sequence is a fallback for the other. They compute the same integer,
//! `CB-03` asserts it, and the choice between them is made by `lane_cap` and
//! the depth --- never by which one is trusted more (R13).

use core::arch::x86_64::*;

use uor_matmul_core::Backend;

use crate::spec::KernelSpec;

/// Rows of `C` per call.
pub const MR: usize = 8;
/// Columns of `C` per call.
pub const NR: usize = 16;
/// `dpwssd` consumes two `k`-steps at a time.
pub const K_GROUP: usize = 2;

/// The signed-word sequence. The default: it reaches further per lane, and it
/// needs no compensation term.
pub const SPEC_DPWSSD: KernelSpec = KernelSpec {
    backend: Backend::Avx512Vnni,
    mr: MR,
    nr: NR,
    k_group: K_GROUP,
    lane_cap: i32::MAX as u128,
    mac_tile: mac_tile_dpwssd,
};

/// The offset sequence. Four bytes per lane instead of two, at the cost of a
/// smaller reach and a compensation term.
pub const SPEC_DPBUSD: KernelSpec = KernelSpec {
    backend: Backend::Avx512Vnni,
    mr: MR,
    nr: NR,
    k_group: 4,
    // `|sum((a + 128) * b)| <= 255 * 128` per step, so this lane fills sooner
    // than the plain one. Declared here rather than assumed, so that
    // `narrow_cap_for` can see it.
    lane_cap: (i32::MAX as u128) / 255 * 128,
    mac_tile: mac_tile_dpbusd,
};

/// Can this host run them?
pub fn is_available() -> bool {
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

/// # Safety
///
/// As [`KernelSpec::mac_tile`], and the host must have `avx512f`, `avx512bw`,
/// and `avx512vnni`.
unsafe fn mac_tile_dpwssd(kc: usize, pa: *const i8, pb: *const i8, acc: *mut i32) {
    // SAFETY: the caller established the target features and the lengths.
    unsafe { dpwssd(kc, pa, pb, acc) }
}

/// # Safety
///
/// As [`mac_tile_dpwssd`].
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn dpwssd(kc: usize, pa: *const i8, pb: *const i8, acc: *mut i32) {
    // SAFETY: the caller guaranteed the three extents. One conversion here
    // keeps every panel read below safe.
    let (pa, pb, acc) = unsafe {
        (
            core::slice::from_raw_parts(pa, MR * kc),
            core::slice::from_raw_parts(pb, NR * kc),
            core::slice::from_raw_parts_mut(acc, MR * NR),
        )
    };

    let mut tile = [_mm512_setzero_si512(); MR];
    let pairs = kc / K_GROUP;

    for q in 0..pairs {
        let (p0, p1) = (q * K_GROUP, q * K_GROUP + 1);
        let mut b_pairs = [0i16; NR * 2];
        for j in 0..NR {
            b_pairs[j * 2] = pb[p0 * NR + j] as i16;
            b_pairs[j * 2 + 1] = pb[p1 * NR + j] as i16;
        }
        // SAFETY: `b_pairs` holds `NR * 2 = 32` i16 = 512 bits.
        let bv = unsafe { _mm512_loadu_si512(b_pairs.as_ptr().cast()) };

        for (i, lane) in tile.iter_mut().enumerate() {
            let a0 = pa[p0 * MR + i] as i16;
            let a1 = pa[p1 * MR + i] as i16;
            let av = _mm512_set1_epi32(((a1 as u16 as u32) << 16 | (a0 as u16 as u32)) as i32);
            // Exact: a0*b0 + a1*b1 into one i32 lane.
            *lane = _mm512_dpwssd_epi32(*lane, av, bv);
        }
    }

    tail(kc, pairs * K_GROUP, pa, pb, &mut tile);
    store(acc, &tile);
}

/// # Safety
///
/// As [`mac_tile_dpwssd`].
unsafe fn mac_tile_dpbusd(kc: usize, pa: *const i8, pb: *const i8, acc: *mut i32) {
    // SAFETY: the caller established the target features and the lengths.
    unsafe { dpbusd(kc, pa, pb, acc) }
}

/// # Safety
///
/// As [`mac_tile_dpwssd`].
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn dpbusd(kc: usize, pa: *const i8, pb: *const i8, acc: *mut i32) {
    const G: usize = 4;
    // SAFETY: the caller guaranteed the three extents.
    let (pa, pb, acc) = unsafe {
        (
            core::slice::from_raw_parts(pa, MR * kc),
            core::slice::from_raw_parts(pb, NR * kc),
            core::slice::from_raw_parts_mut(acc, MR * NR),
        )
    };

    let mut tile = [_mm512_setzero_si512(); MR];
    // The compensation term: `128 * sum_p b[p][j]`, accumulated per column and
    // subtracted once at the end. Exact, and independent of the row.
    let mut colsum = [0i32; NR];

    let groups = kc / G;
    for q in 0..groups {
        let base = q * G;
        let mut b_quads = [0u8; NR * G];
        for j in 0..NR {
            for g in 0..G {
                let v = pb[(base + g) * NR + j];
                b_quads[j * G + g] = v as u8;
                colsum[j] = colsum[j].wrapping_add(v as i32);
            }
        }
        // SAFETY: `b_quads` holds `NR * 4 = 64` bytes = 512 bits.
        let bv = unsafe { _mm512_loadu_si512(b_quads.as_ptr().cast()) };

        for (i, lane) in tile.iter_mut().enumerate() {
            let mut a_quad = [0u8; 4];
            for (g, slot) in a_quad.iter_mut().enumerate() {
                // The offset: `a + 128` is exactly `a as u8` with its top bit
                // flipped, which is why the identity costs no arithmetic here.
                *slot = (pa[(base + g) * MR + i] as u8) ^ 0x80;
            }
            let av = _mm512_set1_epi32(i32::from_le_bytes(a_quad));
            // `vpdpbusd`: unsigned A byte times signed B byte, four per lane.
            *lane = _mm512_dpbusd_epi32(*lane, av, bv);
        }
    }

    // The compensation, and then the k-tail in the plain sequence. Both are
    // exact integers, so the total is still the exact accumulation.
    // SAFETY: `colsum` holds `NR = 16` i32 = 512 bits.
    let comp = unsafe { _mm512_loadu_si512(colsum.as_ptr().cast()) };
    let scaled = _mm512_mullo_epi32(comp, _mm512_set1_epi32(128));
    for lane in tile.iter_mut() {
        *lane = _mm512_sub_epi32(*lane, scaled);
    }

    tail(kc, groups * G, pa, pb, &mut tile);
    store(acc, &tile);
}

/// The `k`-tail, one step at a time, in the plain sequence.
///
/// Zero padding would have been exact too; walking the tail is simply cheaper
/// than materialising a padded panel. Either way the shape is not a special
/// case (S8).
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
fn tail(kc: usize, from: usize, pa: &[i8], pb: &[i8], tile: &mut [__m512i; MR]) {
    for p in from..kc {
        let mut lane = [0i32; NR];
        for (j, slot) in lane.iter_mut().enumerate() {
            *slot = pb[p * NR + j] as i32;
        }
        // SAFETY: `lane` holds `NR = 16` i32 = 512 bits.
        let bv = unsafe { _mm512_loadu_si512(lane.as_ptr().cast()) };
        for (i, t) in tile.iter_mut().enumerate() {
            let a = pa[p * MR + i] as i32;
            *t = _mm512_add_epi32(*t, _mm512_mullo_epi32(bv, _mm512_set1_epi32(a)));
        }
    }
}

/// Write the tile out.
#[target_feature(enable = "avx512f")]
fn store(acc: &mut [i32], tile: &[__m512i; MR]) {
    for (i, lane) in tile.iter().enumerate() {
        // SAFETY: `acc` has `MR * NR` lanes and `i < MR`, so this 512-bit store
        // lands inside it.
        unsafe { _mm512_storeu_si512(acc.as_mut_ptr().add(i * NR).cast(), *lane) };
    }
}
