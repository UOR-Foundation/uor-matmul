//! AArch64 NEON with the ARMv8.2-A dot-product extension (§7.2).
//!
//! `vdotq_s32` consumes four `k`-steps per lane in one instruction and
//! accumulates straight into `i32`, so there is no intermediate width to
//! overflow and no compensation term. It is the widest reach of any sequence in
//! this crate, and it computes the same integer as every other one.

use core::arch::aarch64::*;

use uor_matmul_core::Backend;

/// `sdot Vd.4S, Vn.16B, Vm.16B`.
///
/// Written as inline assembly because `vdotq_s32` is still unstable in
/// `core::arch`. The instruction is the same one the intrinsic emits, and
/// reaching it through `asm!` rather than through a nightly feature keeps the
/// crate on the pinned stable toolchain --- which matters more here than the
/// spelling, because a backend that cannot be built cannot be validated.
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

use crate::spec::KernelSpec;

/// Rows of `C` per call.
pub const MR: usize = 8;
/// Columns of `C` per call. A multiple of four, because `vdotq_s32` produces
/// four columns at a time.
pub const NR: usize = 12;
/// `vdotq_s32` consumes four `k`-steps at a time.
pub const K_GROUP: usize = 4;

/// The NEON dot-product spec.
pub const SPEC: KernelSpec = KernelSpec {
    backend: Backend::NeonDotprod,
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
        std::arch::is_aarch64_feature_detected!("dotprod")
    }
    #[cfg(not(any(feature = "std", test)))]
    {
        cfg!(target_feature = "dotprod")
    }
}

/// Accumulate an `8 x 12` tile.
///
/// # Safety
///
/// `pa` must have `MR * kc` readable elements, `pb` must have `NR * kc`, `acc`
/// must have `MR * NR` writable lanes, and the host must have `dotprod`.
unsafe fn mac_tile(kc: usize, pa: *const i8, pb: *const i8, acc: *mut i32) {
    // SAFETY: the caller established the lengths, and `available` established
    // the `dotprod` feature before returning this spec.
    unsafe { dotprod(kc, pa, pb, acc) }
}

/// # Safety
///
/// As [`mac_tile`].
#[target_feature(enable = "neon,dotprod")]
unsafe fn dotprod(kc: usize, pa: *const i8, pb: *const i8, acc: *mut i32) {
    const QUADS: usize = NR / 4;
    // SAFETY: the caller guaranteed the three extents.
    let (pa, pb, acc) = unsafe {
        (
            core::slice::from_raw_parts(pa, MR * kc),
            core::slice::from_raw_parts(pb, NR * kc),
            core::slice::from_raw_parts_mut(acc, MR * NR),
        )
    };

    let mut tile = [[vdupq_n_s32(0); QUADS]; MR];
    let groups = kc / K_GROUP;

    for q in 0..groups {
        let base = q * K_GROUP;
        // B, transposed into `k`-quads per column: lane `4c + g` holds
        // `b[base + g][4 * quad + c]`, which is the layout `sdot` reads.
        let mut bq = [[0i8; 16]; QUADS];
        for (quad, block) in bq.iter_mut().enumerate() {
            for c in 0..4 {
                for g in 0..K_GROUP {
                    block[c * 4 + g] = pb[(base + g) * NR + quad * 4 + c];
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
                *slot = pa[(base + g) * MR + i] as u8;
            }
            // The same four `a` values against every group of four columns.
            let av = vreinterpretq_s8_s32(vdupq_n_s32(i32::from_le_bytes(aq)));
            for (quad, lane) in row.iter_mut().enumerate() {
                // SAFETY: `dotprod` is enabled on this function.
                *lane = unsafe { sdot(*lane, av, bv[quad]) };
            }
        }
    }

    // The `k`-tail, one step at a time. Zero padding would have been exact too;
    // walking the tail is simply cheaper than materialising a padded panel.
    for p in (groups * K_GROUP)..kc {
        for (i, row) in tile.iter_mut().enumerate() {
            let a = pa[p * MR + i] as i32;
            for (quad, lane) in row.iter_mut().enumerate() {
                let mut cols = [0i32; 4];
                for (c, slot) in cols.iter_mut().enumerate() {
                    *slot = a.wrapping_mul(pb[p * NR + quad * 4 + c] as i32);
                }
                // SAFETY: `cols` holds exactly four i32.
                *lane = vaddq_s32(*lane, unsafe { vld1q_s32(cols.as_ptr()) });
            }
        }
    }

    for (i, row) in tile.iter().enumerate() {
        for (quad, lane) in row.iter().enumerate() {
            // SAFETY: `acc` has `MR * NR` lanes, `i < MR`, `quad * 4 + 4 <= NR`.
            unsafe { vst1q_s32(acc.as_mut_ptr().add(i * NR + quad * 4), *lane) };
        }
    }
}
