//! The kernel table (§7.1).
//!
//! One signature per element type, identical across ISAs. A [`KernelSpec`] is a
//! value; adding an ISA adds one and touches no driver code.

use uor_matmul_core::Backend;

/// One backend's microkernel and its blocking shape.
///
/// The driver reads `mr`, `nr`, and `k_group` through this value rather than
/// through a `match` on [`Backend`], which is what keeps the driver free of
/// per-ISA code.
#[derive(Clone, Copy)]
pub struct KernelSpec {
    /// Which backend this is.
    pub backend: Backend,
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
    /// The largest magnitude this kernel's accumulator lane holds.
    ///
    /// Read by [`uor_matmul_core::narrow_cap_for`] to decide whether a tile of
    /// a given depth fits. A tile that does not fit takes a wider lane and
    /// computes the same integer (§5.1).
    pub lane_cap: u128,
    /// Accumulate an `mr x nr` tile of `C` over a `kc`-deep packed panel pair.
    ///
    /// # Safety
    ///
    /// The caller must ensure `pa` has `mr * kc` readable elements, `pb` has
    /// `nr * kc`, `acc` has `mr * nr` writable lanes, and that the target
    /// features this backend names are present on the host. [`Self::mac_tile`]
    /// establishes the first three; [`available`] establishes the fourth.
    pub mac_tile: unsafe fn(kc: usize, pa: *const i8, pb: *const i8, acc: *mut i32),
}

impl core::fmt::Debug for KernelSpec {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("KernelSpec")
            .field("backend", &self.backend)
            .field("mr", &self.mr)
            .field("nr", &self.nr)
            .field("k_group", &self.k_group)
            .finish_non_exhaustive()
    }
}

impl KernelSpec {
    /// The safe entry point: accumulate an `mr x nr` tile.
    ///
    /// Panics only on a length disagreement, which is a programming error in
    /// the *driver* rather than a condition of the data --- no input a caller
    /// can supply reaches it, which is why `gemm` still returns `()`.
    pub fn mac_tile(&self, kc: usize, pa: &[i8], pb: &[i8], acc: &mut [i32]) {
        assert_eq!(pa.len(), self.mr * kc, "packed A panel is mr * kc");
        assert_eq!(pb.len(), self.nr * kc, "packed B panel is nr * kc");
        assert_eq!(acc.len(), self.mr * self.nr, "accumulator tile is mr * nr");
        // SAFETY: the three lengths are exactly what `mac_tile` requires, and
        // this `KernelSpec` was obtained from `select` or `available`, both of
        // which only ever return a spec whose target features the host has.
        unsafe { (self.mac_tile)(kc, pa.as_ptr(), pb.as_ptr(), acc.as_mut_ptr()) }
    }
}

/// The reference. Always present, always correct, never a fallback (R6).
pub const fn portable() -> KernelSpec {
    crate::isa::portable::SPEC
}

/// Every backend this build can run, portable first.
///
/// With `std`, this consults the host at runtime. Without it, the answer is
/// whatever the target features say at compile time --- which is what an
/// embedded target wants, because there is nothing to detect (C1, S11).
pub fn available() -> impl Iterator<Item = KernelSpec> {
    let mut specs = [None::<KernelSpec>; 8];
    let mut n = 0;
    let mut push = |s: KernelSpec| {
        specs[n] = Some(s);
        n += 1;
    };

    push(portable());

    #[cfg(target_arch = "x86_64")]
    {
        if crate::isa::avx2::is_available() {
            push(crate::isa::avx2::SPEC);
        }
        if crate::isa::avx512vnni::is_available() {
            push(crate::isa::avx512vnni::SPEC_DPWSSD);
            push(crate::isa::avx512vnni::SPEC_DPBUSD);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if crate::isa::neon::is_available() {
            push(crate::isa::neon::SPEC);
        }
        if crate::isa::neon_dotprod::is_available() {
            push(crate::isa::neon_dotprod::SPEC);
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        if crate::isa::wasm_simd128::is_available() {
            push(crate::isa::wasm_simd128::SPEC);
        }
    }

    specs.into_iter().flatten()
}

/// The fastest available factorization, or the one the caller named.
///
/// Selection cannot fail. [`Backend::Auto`] takes the last entry of
/// [`available`], which is the widest one the host can run; a named backend the
/// host cannot run yields the portable kernel, which computes the same integer.
/// That is not a fallback --- there is no answer being given up (R13).
pub fn select(requested: Backend) -> KernelSpec {
    match requested {
        Backend::Auto => available().last().unwrap_or_else(portable),
        named => available()
            .find(|s| s.backend == named)
            .unwrap_or_else(portable),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::vec;
    use std::vec::Vec;
    use uor_matmul_core::{as_alphabet_full, dot_ref};

    /// Reference: the exact `mr x nr` tile, computed by the core's own
    /// accumulation. Every kernel is compared against this.
    fn reference_tile(spec: &KernelSpec, kc: usize, pa: &[i8], pb: &[i8]) -> Vec<i32> {
        let mut out = vec![0i32; spec.mr * spec.nr];
        for i in 0..spec.mr {
            // The packed A panel is k-major: `pa[p * mr + i]`.
            let a: Vec<i8> = (0..kc).map(|p| pa[p * spec.mr + i]).collect();
            for j in 0..spec.nr {
                let b: Vec<i8> = (0..kc).map(|p| pb[p * spec.nr + j]).collect();
                let exact = dot_ref(as_alphabet_full(&a), as_alphabet_full(&b));
                out[i * spec.nr + j] = exact as i32;
            }
        }
        out
    }

    fn fill(len: usize, salt: u64) -> Vec<i8> {
        let mut s = 0x243F_6A88_85A3_08D3u64 ^ salt;
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 33) as i8
            })
            .collect()
    }

    /// `CB-01`: the portable kernel equals `dot_ref` on the whole corpus.
    #[test]
    fn portable_equals_dot_ref_cb_01() {
        let spec = portable();
        for kc in [0usize, 1, 2, 3, 4, 7, 8, 16, 31, 64, 127, 256] {
            let pa = fill(spec.mr * kc, kc as u64);
            let pb = fill(spec.nr * kc, kc as u64 ^ 0x5A);
            let mut acc = vec![0i32; spec.mr * spec.nr];
            spec.mac_tile(kc, &pa, &pb, &mut acc);
            assert_eq!(acc, reference_tile(&spec, kc, &pa, &pb), "kc={kc}");
        }
    }

    /// `CB-02` .. `CB-05`, `CD-01`: every backend this host can run equals the
    /// portable reference, byte for byte.
    ///
    /// The test enumerates what the host actually supports and asserts that
    /// each one agrees. A host with no SIMD still exercises the portable
    /// kernel, and the assertion at the end makes "nothing ran" a failure
    /// rather than a pass.
    #[test]
    fn every_backend_equals_portable_cb_02() {
        let mut checked = 0usize;
        for spec in available() {
            for kc in [0usize, 1, 2, 3, 5, 8, 16, 17, 64, 129, 512] {
                let pa = fill(spec.mr * kc, kc as u64 ^ 0xC0);
                let pb = fill(spec.nr * kc, kc as u64 ^ 0x0D);
                let mut acc = vec![0i32; spec.mr * spec.nr];
                spec.mac_tile(kc, &pa, &pb, &mut acc);
                assert_eq!(
                    acc,
                    reference_tile(&spec, kc, &pa, &pb),
                    "{} disagrees with the reference at kc={kc}",
                    spec.backend.as_str()
                );
            }
            checked += 1;
        }
        assert!(checked >= 1, "at least the portable kernel must have run");
        // Report what was actually exercised, so a green run on a host with no
        // SIMD is not mistaken for a green run on one with it.
        std::eprintln!(
            "CB-02: {checked} backend(s) checked: {}",
            available()
                .map(|s| s.backend.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    /// `CU-03`: every instruction sequence agrees at depths straddling its own
    /// threshold.
    ///
    /// The worst case for a signed byte pair is `128 * 128` per step, so these
    /// depths are chosen around where each sequence's lane would fill. The
    /// values are the extremes, not random, because a random fill cancels and
    /// would never reach the threshold at all.
    #[test]
    fn sequences_agree_across_their_thresholds_cu_03() {
        for spec in available() {
            for kc in [1usize, 2, 3, 4, 8, 15, 16, 17, 128, 129, 130] {
                // All-extreme inputs, so the lane fills as fast as it can.
                let pa = vec![i8::MIN; spec.mr * kc];
                let pb = vec![i8::MIN; spec.nr * kc];
                let mut acc = vec![0i32; spec.mr * spec.nr];
                spec.mac_tile(kc, &pa, &pb, &mut acc);
                let expect = (kc as i32) * 128 * 128;
                assert!(
                    acc.iter().all(|&x| x == expect),
                    "{} at kc={kc}: expected {expect}",
                    spec.backend.as_str()
                );
            }
        }
    }

    /// Check one named backend against the reference, and say whether the host
    /// could run it. Shared by `CB-03`, `CB-04`, and `CB-05`.
    fn check_named(backend: Backend) -> bool {
        let Some(spec) = available().find(|s| s.backend == backend) else {
            std::eprintln!(
                "{}: not available on this host; the cross-architecture CI job runs it",
                backend.as_str()
            );
            return false;
        };
        for kc in [0usize, 1, 2, 3, 4, 5, 8, 16, 17, 63, 64, 129, 512] {
            let pa = fill(spec.mr * kc, kc as u64 ^ 0xAB);
            let pb = fill(spec.nr * kc, kc as u64 ^ 0xCD);
            let mut acc = vec![0i32; spec.mr * spec.nr];
            spec.mac_tile(kc, &pa, &pb, &mut acc);
            assert_eq!(
                acc,
                reference_tile(&spec, kc, &pa, &pb),
                "{} disagrees at kc={kc}",
                backend.as_str()
            );
        }
        true
    }

    /// `CB-03`: AVX-512 VNNI equals portable, on all of its sequences.
    ///
    /// `available` returns both the `dpwssd` and the `dpbusd` spec under the
    /// same `Backend`, so `check_named` exercises whichever comes first and the
    /// loop below covers the rest. Two sequences, one answer.
    #[test]
    fn avx512vnni_equals_portable_cb_03() {
        let ran = check_named(Backend::Avx512Vnni);
        if ran {
            let mut sequences = 0usize;
            for spec in available().filter(|s| s.backend == Backend::Avx512Vnni) {
                for kc in [1usize, 4, 5, 64, 129] {
                    let pa = fill(spec.mr * kc, 1);
                    let pb = fill(spec.nr * kc, 2);
                    let mut acc = vec![0i32; spec.mr * spec.nr];
                    spec.mac_tile(kc, &pa, &pb, &mut acc);
                    assert_eq!(acc, reference_tile(&spec, kc, &pa, &pb));
                }
                sequences += 1;
            }
            assert!(sequences >= 2, "both VNNI sequences must be exercised");
        }
    }

    /// `CB-04`: NEON and NEON dotprod equal portable.
    #[test]
    fn neon_equals_portable_cb_04() {
        let _ = check_named(Backend::Neon);
        let _ = check_named(Backend::NeonDotprod);
    }

    /// `CB-05`: wasm SIMD128 equals portable, and a SIMD128-off build agrees
    /// with a SIMD128-on one.
    ///
    /// The second half is what the portable kernel is for: on wasm without
    /// `simd128` the driver runs it, and `CB-01` already pins it to `dot_ref`.
    /// So "SIMD128-off equals SIMD128-on" is the composition of `CB-01` and
    /// this test, and the wasm CI job runs both configurations.
    #[test]
    fn wasm_simd128_equals_portable_cb_05() {
        let _ = check_named(Backend::WasmSimd128);
    }

    /// `CD-01`: the backend a caller names never changes the answer, and
    /// naming one the host cannot run is not an error.
    #[test]
    fn backend_selection_cannot_fail_cd_01() {
        for backend in Backend::ALL {
            let spec = select(backend);
            let kc = 33;
            let pa = fill(spec.mr * kc, 1);
            let pb = fill(spec.nr * kc, 2);
            let mut acc = vec![0i32; spec.mr * spec.nr];
            spec.mac_tile(kc, &pa, &pb, &mut acc);
            assert_eq!(acc, reference_tile(&spec, kc, &pa, &pb));
        }
        // `Auto` is the widest available, never a failure.
        let _ = select(Backend::Auto);
    }
}
