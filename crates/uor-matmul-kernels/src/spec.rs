//! The kernel table (§7.1).
//!
//! One signature per element family, identical across ISAs. A [`KernelSpec`] is
//! a value; adding an ISA adds one and touches no driver code.
//!
//! # The two factorizations
//!
//! Both compute the same thing. Neither is a fallback, and neither is a
//! classical method: they are the one identity, factored two ways.
//!
//! [`Factorization::Exact`] accumulates in a lane wide enough that the partial
//! sum cannot leave it, and the lanes fold into `AccOf<E>` --- which cannot
//! overflow at all --- before the single encode step. The lane width is decided
//! from the declared alphabet bound.
//!
//! [`Factorization::Modular`] applies when the caller asks to encode by
//! wrapping into a `w`-bit output. Reduction modulo `2^w` is a ring
//! homomorphism, so reducing the exact sum once at the end equals reducing at
//! every step: accumulating in `Z/2^w` *is* the exact accumulation, seen in the
//! quotient the caller asked for. That is the same fact §3.4 rests on for the
//! integer oracles, and it is what lets a `w`-bit lane carry an unbounded depth
//! with no wide accumulator behind it.
//!
//! Which one runs is decided by the caller's *declarations* --- the alphabet
//! bound and the encode mode --- never by a heuristic and never by the data.

use uor_matmul_core::Backend;

/// Which factorization of the identity a kernel realizes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Factorization {
    /// The lane holds the partial sum exactly; lanes fold into `AccOf<E>`.
    Exact,
    /// The lane is `Z/2^w` for the `w`-bit output the caller asked to wrap
    /// into. Exact in the quotient, by ring homomorphism.
    Modular,
}

/// One backend's microkernel for one element family, and its blocking shape.
///
/// `E` is the packed element type and `L` is the lane the kernel accumulates
/// in. The driver reads `mr`, `nr`, and `k_group` through this value rather
/// than through a `match` on [`Backend`], which is what keeps the driver free
/// of per-ISA code.
pub struct KernelSpec<E, L> {
    /// Which backend this is.
    pub backend: Backend,
    /// Which factorization it realizes.
    pub factorization: Factorization,
    /// Rows of `C` this kernel produces per call.
    pub mr: usize,
    /// Columns of `C` this kernel produces per call.
    pub nr: usize,
    /// The `k`-multiple the packing must respect.
    ///
    /// Not a restriction on the caller's `k`: the driver pads the tail with the
    /// alphabet's zero, which is exact, so an arbitrary `k` takes this path and
    /// not a different one (S8).
    pub k_group: usize,
    /// The largest magnitude one lane holds.
    ///
    /// Ignored for [`Factorization::Modular`], where the lane wraps by design
    /// and the depth is unbounded.
    pub lane_cap: u128,
    /// Accumulate an `mr x nr` tile of `C` over a `kc`-deep packed panel pair.
    ///
    /// # Safety
    ///
    /// The caller must ensure `pa` has `mr * kc` readable elements, `pb` has
    /// `nr * kc`, `acc` has `mr * nr` writable lanes, and that the target
    /// features this backend names are present on the host. [`Self::mac_tile`]
    /// establishes the first three; the `available_*` functions establish the
    /// fourth.
    pub mac_tile: unsafe fn(kc: usize, pa: *const E, pb: *const E, acc: *mut L),
}

impl<E, L> Clone for KernelSpec<E, L> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<E, L> Copy for KernelSpec<E, L> {}

impl<E, L> core::fmt::Debug for KernelSpec<E, L> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("KernelSpec")
            .field("backend", &self.backend)
            .field("factorization", &self.factorization)
            .field("mr", &self.mr)
            .field("nr", &self.nr)
            .field("k_group", &self.k_group)
            .finish_non_exhaustive()
    }
}

impl<E, L> KernelSpec<E, L> {
    /// The safe entry point: accumulate an `mr x nr` tile.
    ///
    /// Panics only on a length disagreement, which is a programming error in
    /// the *driver* rather than a condition of the data --- no input a caller
    /// can supply reaches it, which is why `gemm` still returns `()`.
    pub fn mac_tile(&self, kc: usize, pa: &[E], pb: &[E], acc: &mut [L]) {
        assert_eq!(pa.len(), self.mr * kc, "packed A panel is mr * kc");
        assert_eq!(pb.len(), self.nr * kc, "packed B panel is nr * kc");
        assert_eq!(acc.len(), self.mr * self.nr, "accumulator tile is mr * nr");
        // SAFETY: the three lengths are exactly what `mac_tile` requires, and
        // this `KernelSpec` came from one of the `available_*` functions, which
        // only ever return a spec whose target features the host has.
        unsafe { (self.mac_tile)(kc, pa.as_ptr(), pb.as_ptr(), acc.as_mut_ptr()) }
    }

    /// The deepest chunk this lane holds for an alphabet bounded by `bound`.
    ///
    /// A question about a register, not a limit on `k`: a deeper accumulation
    /// is split into more chunks, and the chunks combine exactly. For a modular
    /// lane there is nothing to bound --- the wrap *is* the encode.
    pub fn lane_depth(&self, bound: u128) -> usize {
        if matches!(self.factorization, Factorization::Modular) {
            return usize::MAX;
        }
        let per_step = bound.saturating_mul(bound); // R3-ok: a lane-width question, not an accumulation
        if per_step == 0 {
            return usize::MAX;
        }
        usize::try_from(self.lane_cap / per_step)
            .unwrap_or(usize::MAX)
            .max(1)
    }
}

/// The specs a build can run, portable first.
///
/// A chain of options rather than a fixed array, so the number of kernels a
/// family may have is not capped by a constant somebody chose. Adding one adds
/// a line here and nothing else (R8).
macro_rules! collect {
    ($($cond:expr => $spec:expr),* $(,)?) => {{
        core::iter::empty()
        $(
            .chain(core::iter::once_with(|| if $cond { Some($spec) } else { None }))
        )*
        .flatten()
    }};
}

/// The largest `mr * nr` any shipped kernel produces.
///
/// Derived rather than chosen: every kernel below carries a `const` assertion
/// that its own tile fits, so a kernel too large for this fails the *build*
/// rather than overflowing a buffer. That is what keeps it a derivation and not
/// a ceiling --- there is no input that can reach it, and no way to add a
/// kernel that quietly exceeds it.
pub const MAX_TILE_LANES: usize = 8 * 16;

/// Assert at compile time that a kernel's tile fits [`MAX_TILE_LANES`].
#[macro_export]
macro_rules! tile_fits {
    ($mr:expr, $nr:expr) => {
        const _: () = assert!(
            $mr * $nr <= $crate::MAX_TILE_LANES,
            "this kernel's tile exceeds MAX_TILE_LANES; raise it in the same commit"
        );
    };
}

/// Every `i8 x i8 -> i32` kernel this build can run, portable first.
pub fn available_i8() -> impl Iterator<Item = KernelSpec<i8, i32>> {
    collect![
        true => crate::isa::portable::I8_I32,
        crate::isa::x86::avx2_available() => crate::isa::x86::AVX2_I8_I32,
        crate::isa::x86::avx512vnni_available() => crate::isa::x86::AVX512_DPWSSD_I8_I32,
        crate::isa::x86::avx512vnni_available() => crate::isa::x86::AVX512_DPBUSD_I8_I32,
        crate::isa::arm::neon_available() => crate::isa::arm::NEON_I8_I32,
        crate::isa::arm::dotprod_available() => crate::isa::arm::NEON_DOTPROD_I8_I32,
        crate::isa::wasm::simd128_available() => crate::isa::wasm::SIMD128_I8_I32,
    ]
}

/// Every `i16 x i16 -> i64` kernel this build can run.
///
/// `_mm256_madd_epi16` multiplies signed words and sums adjacent pairs into an
/// `i32`, which is exactly this family's arithmetic --- so `i16` reaches the
/// same instruction `i8` reaches after widening, without the widening.
pub fn available_i16() -> impl Iterator<Item = KernelSpec<i16, i64>> {
    collect![
        true => crate::isa::portable::I16_I64,
        crate::isa::x86::avx2_available() => crate::isa::x86::AVX2_I16_I64,
    ]
}

/// Every exact `i32 x i32 -> i64` kernel this build can run.
///
/// The product of two `i32` needs 62 bits, so the lane must be 64.
/// `_mm256_mul_epi32` is a signed `32x32 -> 64` multiply, which is this
/// family's whole arithmetic in one instruction.
pub fn available_i32_exact() -> impl Iterator<Item = KernelSpec<i32, i64>> {
    collect![
        true => crate::isa::portable::I32_I64,
        crate::isa::x86::avx2_available() => crate::isa::x86::AVX2_I32_I64,
    ]
}

/// Every modular `i32 x i32 -> i32` kernel this build can run.
///
/// The lane is `Z/2^32`. Legitimate exactly when the caller asked to encode by
/// wrapping into a 32-bit output, because then the lane's own wrap *is* the
/// encode and nothing is lost that the caller did not ask to lose.
pub fn available_i32_modular() -> impl Iterator<Item = KernelSpec<i32, i32>> {
    collect![
        true => crate::isa::portable::I32_MOD,
        crate::isa::x86::avx2_available() => crate::isa::x86::AVX2_I32_MOD,
    ]
}

/// Every exact `i64 x i64 -> i128` kernel this build can run.
///
/// The product of two `i64` needs 128 bits, so the lane is an `i128`. No SIMD
/// integer multiply reaches that width on any target this crate supports, so
/// the portable kernel is not a placeholder here --- it is the whole of what
/// the hardware offers, and packing still buys it the locality every other
/// family gets.
pub fn available_i64_exact() -> impl Iterator<Item = KernelSpec<i64, i128>> {
    collect![
        true => crate::isa::portable::I64_I128,
    ]
}

/// Every modular `i16 x i16 -> i32` kernel this build can run.
///
/// Twice the lanes of the exact `i16` kernel, because in `Z/2^32` there is
/// nothing to widen to: `madd` already lands in `i32` and the accumulation
/// stays there.
pub fn available_i16_modular() -> impl Iterator<Item = KernelSpec<i16, i32>> {
    collect![
        true => crate::isa::portable::I16_MOD,
        crate::isa::x86::avx2_available() => crate::isa::x86::AVX2_I16_MOD,
    ]
}

/// Every modular `i64 x i64 -> i64` kernel this build can run.
///
/// A single `i64 x i64` product needs 128 bits, so there is no exact 64-bit
/// lane and no SIMD integer multiply that reaches it. In the quotient there is:
/// `Z/2^64` needs only the low half of each product, which is what a plain
/// `wrapping_mul` gives.
pub fn available_i64_modular() -> impl Iterator<Item = KernelSpec<i64, i64>> {
    collect![
        true => crate::isa::portable::I64_MOD,
    ]
}

/// The reference `i8` kernel. Always present, always correct, never a
/// fallback (R6).
pub const fn portable_i8() -> KernelSpec<i8, i32> {
    crate::isa::portable::I8_I32
}

/// Choose from a family: the backend the caller named, or the widest available.
///
/// Selection cannot fail. [`Backend::Auto`] takes the last entry, which is the
/// widest one the host can run; a named backend the host cannot run yields the
/// first, which computes the same value. That is not a fallback --- there is no
/// answer being given up (R13).
pub fn choose<E, L>(
    specs: impl Iterator<Item = KernelSpec<E, L>>,
    requested: Backend,
) -> Option<KernelSpec<E, L>> {
    let mut first = None;
    let mut widest = None;
    let mut named = None;
    for spec in specs {
        if first.is_none() {
            first = Some(spec);
        }
        if spec.backend == requested {
            named = Some(spec);
        }
        widest = Some(spec);
    }
    match requested {
        Backend::Auto => widest.or(first),
        _ => named.or(first),
    }
}
