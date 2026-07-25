//! `CB-01` .. `CB-05`, `CD-01`, `CU-02`, `CU-03`: every backend, in every
//! family, equals its reference.
//!
//! The point of a kernel table is that adding an instruction cannot change an
//! answer. These tests are what makes that a fact rather than an intention:
//! each family has a reference, and each backend is compared against it on
//! shapes chosen to hit every tail and every threshold.

use uor_matmul_core::{as_alphabet_full, dot_ref, Backend};
use uor_matmul_kernels::{
    available_i16, available_i32_exact, available_i32_modular, available_i64_modular, available_i8,
    choose, portable_i8, Factorization, KernelSpec,
};

/// Deterministic fill. A recorded generator rather than a crate, so a failure
/// reproduces from the seed alone.
fn fill<T, F: Fn(i64) -> T>(len: usize, salt: u64, map: F) -> Vec<T> {
    let mut s = 0x243F_6A88_85A3_08D3u64 ^ salt;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            map((s >> 33) as i64)
        })
        .collect()
}

/// The exact `mr x nr` tile for an `i8` kernel, from the core's own
/// accumulation. This is the link that anchors the whole chain to `dot_ref`.
fn reference_i8(spec: &KernelSpec<i8, i32>, kc: usize, pa: &[i8], pb: &[i8]) -> Vec<i32> {
    let mut out = vec![0i32; spec.mr * spec.nr];
    for i in 0..spec.mr {
        let a: Vec<i8> = (0..kc).map(|p| pa[p * spec.mr + i]).collect();
        for j in 0..spec.nr {
            let b: Vec<i8> = (0..kc).map(|p| pb[p * spec.nr + j]).collect();
            out[i * spec.nr + j] = dot_ref(as_alphabet_full(&a), as_alphabet_full(&b)) as i32;
        }
    }
    out
}

const DEPTHS: &[usize] = &[0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 63, 64, 65, 129, 512];

/// `CB-01`: the portable kernel equals `dot_ref` on the whole corpus.
#[test]
fn portable_equals_dot_ref_cb_01() {
    let spec = portable_i8();
    for &kc in DEPTHS {
        let pa = fill(spec.mr * kc, kc as u64, |v| v as i8);
        let pb = fill(spec.nr * kc, kc as u64 ^ 0x5A, |v| v as i8);
        let mut acc = vec![0i32; spec.mr * spec.nr];
        spec.mac_tile(kc, &pa, &pb, &mut acc);
        assert_eq!(acc, reference_i8(&spec, kc, &pa, &pb), "kc={kc}");
    }
}

/// `CB-02`: every `i8` backend this host can run equals the portable
/// reference, byte for byte.
#[test]
fn every_i8_backend_equals_portable_cb_02() {
    let mut names = Vec::new();
    for spec in available_i8() {
        for &kc in DEPTHS {
            let pa = fill(spec.mr * kc, kc as u64 ^ 0xC0, |v| v as i8);
            let pb = fill(spec.nr * kc, kc as u64 ^ 0x0D, |v| v as i8);
            let mut acc = vec![0i32; spec.mr * spec.nr];
            spec.mac_tile(kc, &pa, &pb, &mut acc);
            assert_eq!(
                acc,
                reference_i8(&spec, kc, &pa, &pb),
                "{} disagrees at kc={kc}",
                spec.backend.as_str()
            );
        }
        names.push(spec.backend.as_str());
    }
    assert!(
        !names.is_empty(),
        "at least the portable kernel must have run"
    );
    eprintln!("CB-02: {} i8 backend(s): {}", names.len(), names.join(", "));
}

/// Check one named `i8` backend, and say whether the host could run it.
fn check_named(backend: Backend) -> bool {
    let Some(spec) = available_i8().find(|s| s.backend == backend) else {
        eprintln!(
            "{}: not available on this host; the cross-architecture CI job runs it",
            backend.as_str()
        );
        return false;
    };
    for &kc in DEPTHS {
        let pa = fill(spec.mr * kc, kc as u64 ^ 0xAB, |v| v as i8);
        let pb = fill(spec.nr * kc, kc as u64 ^ 0xCD, |v| v as i8);
        let mut acc = vec![0i32; spec.mr * spec.nr];
        spec.mac_tile(kc, &pa, &pb, &mut acc);
        assert_eq!(
            acc,
            reference_i8(&spec, kc, &pa, &pb),
            "{} disagrees at kc={kc}",
            backend.as_str()
        );
    }
    true
}

/// `CB-03`: AVX-512 VNNI equals portable, on both of its sequences.
#[test]
fn avx512vnni_equals_portable_cb_03() {
    if check_named(Backend::Avx512Vnni) {
        let sequences = available_i8()
            .filter(|s| s.backend == Backend::Avx512Vnni)
            .count();
        assert!(sequences >= 2, "both VNNI sequences must be registered");
    }
}

/// `CB-04`: NEON and NEON dotprod equal portable.
#[test]
fn neon_equals_portable_cb_04() {
    let _ = check_named(Backend::Neon);
    let _ = check_named(Backend::NeonDotprod);
}

/// `CB-05`: wasm SIMD128 equals portable. A SIMD128-off build runs the portable
/// kernel, so "off equals on" is the composition of this with `CB-01`.
#[test]
fn wasm_simd128_equals_portable_cb_05() {
    let _ = check_named(Backend::WasmSimd128);
}

/// `CU-03`: every sequence agrees at depths straddling its own threshold, on
/// the extremes where a lane fills fastest.
#[test]
fn sequences_agree_across_their_thresholds_cu_03() {
    for spec in available_i8() {
        for kc in [1usize, 2, 3, 4, 8, 15, 16, 17, 128, 129, 130] {
            let pa = vec![i8::MIN; spec.mr * kc];
            let pb = vec![i8::MIN; spec.nr * kc];
            let mut acc = vec![0i32; spec.mr * spec.nr];
            spec.mac_tile(kc, &pa, &pb, &mut acc);
            let expect = (kc as i32) * 128 * 128;
            assert!(
                acc.iter().all(|&x| x == expect),
                "{} at kc={kc}",
                spec.backend.as_str()
            );
        }
    }
}

/// `CB-02`, the wider families: `i16`, `i32` exact, `i32` modular, and `i64`
/// modular each equal their own reference.
///
/// Without these, those instantiations would have no kernel and the driver
/// would be running an unoptimised path. That is not a smaller version of the
/// same library --- it is a different measurement, and comparing it against an
/// oracle would say nothing.
#[test]
fn the_wider_families_equal_their_references_cb_02() {
    for spec in available_i16() {
        for &kc in DEPTHS {
            let pa = fill(spec.mr * kc, 7, |v| (v * 251) as i16);
            let pb = fill(spec.nr * kc, 8, |v| (v * 251) as i16);
            let mut acc = vec![0i64; spec.mr * spec.nr];
            spec.mac_tile(kc, &pa, &pb, &mut acc);
            for i in 0..spec.mr {
                for j in 0..spec.nr {
                    let want: i64 = (0..kc)
                        .map(|p| i64::from(pa[p * spec.mr + i]) * i64::from(pb[p * spec.nr + j]))
                        .sum();
                    assert_eq!(
                        acc[i * spec.nr + j],
                        want,
                        "{} i16 kc={kc}",
                        spec.backend.as_str()
                    );
                }
            }
        }
    }

    for spec in available_i32_exact() {
        // Values bounded by 2^20, so the *reference* sum fits an `i64` at every
        // depth below. The kernel's own lane is bounded the same way, and the
        // driver only ever offers it a chunk `lane_depth` admits --- testing it
        // past that would be testing something the driver never asks for.
        for &kc in DEPTHS {
            let pa = fill(spec.mr * kc, 9, |v| ((v & 0xFFFF) - 0x8000) as i32);
            let pb = fill(spec.nr * kc, 10, |v| ((v & 0xFFFF) - 0x8000) as i32);
            let mut acc = vec![0i64; spec.mr * spec.nr];
            spec.mac_tile(kc, &pa, &pb, &mut acc);
            for i in 0..spec.mr {
                for j in 0..spec.nr {
                    let want: i64 = (0..kc)
                        .map(|p| i64::from(pa[p * spec.mr + i]) * i64::from(pb[p * spec.nr + j]))
                        .sum();
                    assert_eq!(
                        acc[i * spec.nr + j],
                        want,
                        "{} i32 kc={kc}",
                        spec.backend.as_str()
                    );
                }
            }
        }
    }

    // The modular lanes wrap, and that *is* the answer in the quotient: the
    // references below wrap too, deliberately.
    for spec in available_i32_modular() {
        for &kc in DEPTHS {
            let pa = fill(spec.mr * kc, 11, |v| (v.wrapping_mul(99_991)) as i32);
            let pb = fill(spec.nr * kc, 12, |v| (v.wrapping_mul(65_537)) as i32);
            let mut acc = vec![0i32; spec.mr * spec.nr];
            spec.mac_tile(kc, &pa, &pb, &mut acc);
            for i in 0..spec.mr {
                for j in 0..spec.nr {
                    let want = (0..kc).fold(0i32, |s, p| {
                        s.wrapping_add(pa[p * spec.mr + i].wrapping_mul(pb[p * spec.nr + j]))
                    });
                    assert_eq!(
                        acc[i * spec.nr + j],
                        want,
                        "{} i32 modular kc={kc}",
                        spec.backend.as_str()
                    );
                }
            }
        }
    }

    for spec in available_i64_modular() {
        for &kc in DEPTHS {
            let pa = fill(spec.mr * kc, 13, |v| {
                v.wrapping_mul(0x9E37_79B9_7F4A_7C15u64 as i64)
            });
            let pb = fill(spec.nr * kc, 14, |v| v.wrapping_mul(0x1000_0000_01B3));
            let mut acc = vec![0i64; spec.mr * spec.nr];
            spec.mac_tile(kc, &pa, &pb, &mut acc);
            for i in 0..spec.mr {
                for j in 0..spec.nr {
                    let want = (0..kc).fold(0i64, |s, p| {
                        s.wrapping_add(pa[p * spec.mr + i].wrapping_mul(pb[p * spec.nr + j]))
                    });
                    assert_eq!(
                        acc[i * spec.nr + j],
                        want,
                        "{} i64 modular kc={kc}",
                        spec.backend.as_str()
                    );
                }
            }
        }
    }
}

/// `CD-01`: naming a backend never changes the answer, and naming one the host
/// cannot run is not an error.
#[test]
fn backend_selection_cannot_fail_cd_01() {
    for backend in Backend::ALL {
        let spec = choose(available_i8(), backend).expect("the portable kernel is always there");
        let kc = 33;
        let pa = fill(spec.mr * kc, 1, |v| v as i8);
        let pb = fill(spec.nr * kc, 2, |v| v as i8);
        let mut acc = vec![0i32; spec.mr * spec.nr];
        spec.mac_tile(kc, &pa, &pb, &mut acc);
        assert_eq!(acc, reference_i8(&spec, kc, &pa, &pb));

        // Every family answers for every backend, so no instantiation is left
        // without a kernel.
        assert!(choose(available_i16(), backend).is_some());
        assert!(choose(available_i32_exact(), backend).is_some());
        assert!(choose(available_i32_modular(), backend).is_some());
        assert!(choose(available_i64_modular(), backend).is_some());
    }
    assert!(choose(available_i8(), Backend::Auto).is_some());
}

/// `CU-02`: a modular lane has no depth limit, because the wrap is the encode
/// rather than an overflow --- and an exact lane's limit is a property of the
/// declared bound, not of the library.
#[test]
fn lane_depth_follows_the_declaration_cu_02() {
    let exact = choose(available_i32_exact(), Backend::Auto).unwrap();
    let modular = choose(available_i32_modular(), Backend::Auto).unwrap();

    assert_eq!(exact.factorization, Factorization::Exact);
    assert_eq!(modular.factorization, Factorization::Modular);

    // At the full `i32` bound an exact `i64` lane holds exactly one product:
    // two of magnitude `2^62` would sum past `i64::MAX`. That is not a limit on
    // `k` --- it is the depth at which the driver starts a new chunk.
    assert_eq!(exact.lane_depth(1 << 31), 1);
    // A declared narrower bound buys a deeper chunk, and nothing else.
    assert!(exact.lane_depth(1 << 10) > 1_000_000);
    // The modular lane is unbounded at every declared bound.
    assert_eq!(modular.lane_depth(1 << 31), usize::MAX);
    assert_eq!(modular.lane_depth(1), usize::MAX);
}
