//! AArch64 NEON, with and without the dot-product extension (§7.2).

use core::arch::aarch64::*;

use uor_matmul_core::Backend;

use crate::spec::{Factorization, KernelSpec};

crate::tile_fits!(4, 8);
crate::tile_fits!(8, 12);

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
    k_group: 8,
    lane_cap: i32::MAX as u128,
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
    k_group: 4,
    lane_cap: i32::MAX as u128,
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

    let groups = kc / G;
    for q in 0..groups {
        let base = q * G;
        // B, transposed into `k`-quads per column: lane `4c + g` holds
        // `b[base + g][4 * quad + c]`, which is the layout `sdot` reads.
        let mut bq = [[0i8; 16]; QUADS];
        for (quad, block) in bq.iter_mut().enumerate() {
            for c in 0..4 {
                for g in 0..G {
                    block[c * 4 + g] = pb[(base + g) * DOT_NR + quad * 4 + c];
                }
            }
        }
        let bv: [int8x16_t; QUADS] = core::array::from_fn(|quad| {
            // SAFETY: each block is exactly sixteen bytes.
            unsafe { vld1q_s8(bq[quad].as_ptr()) }
        });

        for (i, row) in tile.iter_mut().enumerate() {
            let mut aq = [0u8; 4];
            for (g, slot) in aq.iter_mut().enumerate() {
                *slot = pa[(base + g) * DOT_MR + i] as u8;
            }
            let av = vreinterpretq_s8_s32(vdupq_n_s32(i32::from_le_bytes(aq)));
            for (quad, lane) in row.iter_mut().enumerate() {
                // SAFETY: `dotprod` is enabled on this function.
                *lane = unsafe { sdot(*lane, av, bv[quad]) };
            }
        }
    }

    // The `k`-tail, one step at a time. Zero padding would have been exact too;
    // walking the tail is simply cheaper than materialising a padded panel.
    for p in (groups * G)..kc {
        for (i, row) in tile.iter_mut().enumerate() {
            let a = i32::from(pa[p * DOT_MR + i]);
            for (quad, lane) in row.iter_mut().enumerate() {
                let mut cols = [0i32; 4];
                for (c, slot) in cols.iter_mut().enumerate() {
                    *slot = a.wrapping_mul(i32::from(pb[p * DOT_NR + quad * 4 + c]));
                }
                // SAFETY: `cols` holds exactly four i32.
                *lane = vaddq_s32(*lane, unsafe { vld1q_s32(cols.as_ptr()) });
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
