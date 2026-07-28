//! `CB-01` .. `CB-05`, `CD-01`, `CU-02`, `CU-03`: every backend, in every
//! family, equals its reference.
//!
//! The point of a kernel table is that adding an instruction cannot change an
//! answer. These tests are what makes that a fact rather than an intention:
//! each family has a reference, and each backend is compared against it on
//! shapes chosen to hit every tail and every threshold.

use uor_matmul_core::{as_alphabet_full, dot_ref, Backend};
use uor_matmul_kernels::{
    available_i16, available_i16_modular, available_i32_exact, available_i32_modular,
    available_i64_modular, available_i8, available_i8_narrow, choose_for_rows, packed_slot,
    portable_i8, Factorization, KernelSpec,
};

/// Element `p` of lane `l`, read the way the kernel reads it.
///
/// The panel layout is the kernel contract, so the reference goes through the
/// crate's own [`packed_slot`] rather than restating it. A kernel that
/// misreads its panel therefore fails here, which is the whole point.
fn at<T: Copy>(panel: &[T], p: usize, lane: usize, lanes: usize, group: usize) -> T {
    panel[packed_slot(p, lane, lanes, group)]
}

/// The depths a kernel can be handed: a whole number of `k`-groups, which is
/// what the driver always packs (it pads with the alphabet's zero).
fn depths<E, L>(spec: &KernelSpec<E, L>) -> Vec<usize> {
    // Empty, one, a whole group, and one past a group: the four shapes the tail
    // handling has. Under Miri that is the corpus; natively it is the whole list.
    let mut v: Vec<usize> = corpus(DEPTHS, &[0, 1, 8, 65])
        .iter()
        .map(|&kc| kc.div_ceil(spec.k_group) * spec.k_group)
        .collect();
    v.dedup();
    v
}

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
    let g = spec.k_group;
    for i in 0..spec.mr {
        let a: Vec<i8> = (0..kc).map(|p| at(pa, p, i, spec.mr, g)).collect();
        for j in 0..spec.nr {
            let b: Vec<i8> = (0..kc).map(|p| at(pb, p, j, spec.nr, g)).collect();
            out[i * spec.nr + j] = dot_ref(as_alphabet_full(&a), as_alphabet_full(&b)) as i32;
        }
    }
    out
}

/// The corpus, and the smaller one Miri takes.
///
/// Miri checks *soundness* --- provenance, bounds, initialisation --- and one
/// instance of a code path shows that as well as three hundred do, at something
/// like a hundredfold the cost per instance. The native run takes the whole
/// corpus; this is what keeps the Miri job inside a CI budget instead of running
/// past the six-hour ceiling and being cancelled, which is what it did on every
/// push it was ever enabled for.
///
/// It reduces the number of *instances*, not the set of *paths*: every arm of
/// `dispatch_run!` and `dispatch_slab!` is the same macro body at different
/// constants, so a narrow arm and a wide arm together exercise the code, and
/// keeping a code space that is not a power of two keeps the padding claim. The
/// native run adds the rest of the corpus and still runs all of it.
fn corpus<T: Copy>(all: &[T], under_miri: &[T]) -> Vec<T> {
    if cfg!(miri) {
        under_miri.to_vec()
    } else {
        all.to_vec()
    }
}

const DEPTHS: &[usize] = &[0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 63, 64, 65, 129, 512];

/// `CB-01`: the portable kernel equals `dot_ref` on the whole corpus.
#[test]
fn portable_equals_dot_ref_cb_01() {
    let spec = portable_i8();
    for kc in depths(&spec) {
        let pa = fill(spec.mr * kc, kc as u64, |v| v as i8);
        let pb = fill(spec.nr * kc, kc as u64 ^ 0x5A, |v| v as i8);
        let mut acc = vec![0i32; spec.mr * spec.nr];
        spec.mac_tile(kc, &pa, &pb, &mut acc);
        assert_eq!(acc, reference_i8(&spec, kc, &pa, &pb), "kc={kc}");
    }
}

/// Every `i8` tile sequence this build can run, at every panel width.
///
/// The narrow panels live in their own list because the driver only asks for
/// one when the shape is narrower than the tile --- but a differential test has
/// no such condition. A sequence outside the net is a sequence nothing checks.
fn every_i8_tile() -> impl Iterator<Item = uor_matmul_kernels::KernelSpec<i8, i32>> {
    available_i8().chain(available_i8_narrow())
}

/// `CB-02`: every `i8` backend this host can run equals the portable
/// reference, byte for byte.
#[test]
fn every_i8_backend_equals_portable_cb_02() {
    let mut names = Vec::new();
    for spec in every_i8_tile() {
        for kc in depths(&spec) {
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
    eprintln!("CB-02: {} i8 backend(s): {}", names.len(), names.join(", "));
    assert!(
        !names.is_empty(),
        "at least the portable kernel must have run"
    );
    // A test that only ever ran the reference against itself would pass while
    // asserting nothing, which is exactly what this test did before the
    // `std` feature reached it: without runtime detection every `available_*`
    // predicate answered from the *build*'s target features, and on a stock
    // x86-64 build that is none of them. So the vacuous case is named.
    //
    // Not under Miri, which models no vector intrinsics and answers every
    // feature-detection predicate with false. There the portable sequence *is*
    // the whole list, and that is the thing Miri is there to check --- so the
    // absence is expected rather than the misconfiguration this assert catches on
    // a native build.
    if cfg!(any(target_arch = "x86_64", target_arch = "aarch64")) && !cfg!(miri) {
        assert!(
            names.len() > 1,
            "x86-64 and aarch64 both have an i8 kernel past the portable one; \
             seeing only {names:?} means feature detection did not reach this test"
        );
    }
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
    for kc in depths(&spec) {
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
    for spec in every_i8_tile() {
        for kc in [1usize, 2, 3, 4, 8, 15, 16, 17, 128, 129, 130]
            .map(|kc| kc.div_ceil(spec.k_group) * spec.k_group)
        {
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
        for kc in depths(&spec) {
            let pa = fill(spec.mr * kc, 7, |v| (v * 251) as i16);
            let pb = fill(spec.nr * kc, 8, |v| (v * 251) as i16);
            let mut acc = vec![0i64; spec.mr * spec.nr];
            spec.mac_tile(kc, &pa, &pb, &mut acc);
            for i in 0..spec.mr {
                for j in 0..spec.nr {
                    let want: i64 = (0..kc)
                        .map(|p| {
                            i64::from(at(&pa, p, i, spec.mr, spec.k_group))
                                * i64::from(at(&pb, p, j, spec.nr, spec.k_group))
                        })
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
        for kc in depths(&spec) {
            let pa = fill(spec.mr * kc, 9, |v| ((v & 0xFFFF) - 0x8000) as i32);
            let pb = fill(spec.nr * kc, 10, |v| ((v & 0xFFFF) - 0x8000) as i32);
            let mut acc = vec![0i64; spec.mr * spec.nr];
            spec.mac_tile(kc, &pa, &pb, &mut acc);
            for i in 0..spec.mr {
                for j in 0..spec.nr {
                    let want: i64 = (0..kc)
                        .map(|p| {
                            i64::from(at(&pa, p, i, spec.mr, spec.k_group))
                                * i64::from(at(&pb, p, j, spec.nr, spec.k_group))
                        })
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
        for kc in depths(&spec) {
            let pa = fill(spec.mr * kc, 11, |v| (v.wrapping_mul(99_991)) as i32);
            let pb = fill(spec.nr * kc, 12, |v| (v.wrapping_mul(65_537)) as i32);
            let mut acc = vec![0i32; spec.mr * spec.nr];
            spec.mac_tile(kc, &pa, &pb, &mut acc);
            for i in 0..spec.mr {
                for j in 0..spec.nr {
                    let want = (0..kc).fold(0i32, |s, p| {
                        s.wrapping_add(at(&pa, p, i, spec.mr, spec.k_group).wrapping_mul(at(
                            &pb,
                            p,
                            j,
                            spec.nr,
                            spec.k_group,
                        )))
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
        for kc in depths(&spec) {
            let pa = fill(spec.mr * kc, 13, |v| {
                v.wrapping_mul(0x9E37_79B9_7F4A_7C15u64 as i64)
            });
            let pb = fill(spec.nr * kc, 14, |v| v.wrapping_mul(0x1000_0000_01B3));
            let mut acc = vec![0i64; spec.mr * spec.nr];
            spec.mac_tile(kc, &pa, &pb, &mut acc);
            for i in 0..spec.mr {
                for j in 0..spec.nr {
                    let want = (0..kc).fold(0i64, |s, p| {
                        s.wrapping_add(at(&pa, p, i, spec.mr, spec.k_group).wrapping_mul(at(
                            &pb,
                            p,
                            j,
                            spec.nr,
                            spec.k_group,
                        )))
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
        let spec = choose_for_rows(available_i8(), backend, 128, usize::MAX)
            .expect("the portable kernel is always there");
        let kc = 33usize.div_ceil(spec.k_group) * spec.k_group;
        let pa = fill(spec.mr * kc, 1, |v| v as i8);
        let pb = fill(spec.nr * kc, 2, |v| v as i8);
        let mut acc = vec![0i32; spec.mr * spec.nr];
        spec.mac_tile(kc, &pa, &pb, &mut acc);
        assert_eq!(acc, reference_i8(&spec, kc, &pa, &pb));

        // Every family answers for every backend, so no instantiation is left
        // without a kernel.
        assert!(choose_for_rows(available_i16(), backend, 32768, usize::MAX).is_some());
        assert!(choose_for_rows(available_i32_exact(), backend, 1 << 31, usize::MAX).is_some());
        assert!(choose_for_rows(available_i32_modular(), backend, 1 << 31, usize::MAX).is_some());
        assert!(choose_for_rows(available_i64_modular(), backend, 1 << 63, usize::MAX).is_some());
    }
    assert!(choose_for_rows(available_i8(), Backend::Auto, 128, usize::MAX).is_some());

    // The *table* sequences answer for every backend at every tile the driver
    // walks, for the same reason and with no weaker a claim. This half was
    // absent, and `choose_table` filtered the reference out on a named backend
    // rather than falling through to it: measured here before the fix, 246 of
    // these 250 selections were `None`, and every one of them was a panic inside
    // `gemm` -- including `Backend::Avx2` on an AVX2 host at every tile below
    // eight rows, which is every `m` under eight.
    // `block` is swept too, and the odd widths are the point: a sequence that
    // folds `k_group` block steps into one instruction cannot pack a block that
    // is not a whole number of them, so it is inadmissible there. The reference
    // declares `k_group: 1`, which divides every block, so something is always
    // left to choose --- and whatever is chosen must be able to pack what it was
    // chosen for.
    let mut selections = 0usize;
    for backend in Backend::ALL.iter().copied().chain([Backend::Auto]) {
        for rows in [16usize, 8, 4, 2, 1] {
            for group in [16usize, 8, 4, 2, 1] {
                for block in [1usize, 2, 3, 5, 8] {
                    let i8_spec = uor_matmul_kernels::choose_table(
                        uor_matmul_kernels::available_table_i8(rows, group),
                        backend,
                        128,
                        block,
                    )
                    .unwrap_or_else(|| {
                        panic!("i8 table selection failed at {backend:?} {rows}x{group} b={block}")
                    });
                    assert!(
                        block.is_multiple_of(i8_spec.k_group),
                        "i8 {backend:?} {rows}x{group} b={block} chose k_group={} it cannot pack",
                        i8_spec.k_group
                    );
                    let i16_spec = uor_matmul_kernels::choose_table(
                        uor_matmul_kernels::available_table_i16(rows, group),
                        backend,
                        32768,
                        block,
                    )
                    .unwrap_or_else(|| {
                        panic!("i16 table selection failed at {backend:?} {rows}x{group} b={block}")
                    });
                    assert!(
                        block.is_multiple_of(i16_spec.k_group),
                        "i16 {backend:?} {rows}x{group} b={block} chose k_group={} it cannot pack",
                        i16_spec.k_group
                    );
                    selections += 2;
                }
            }
        }
    }
    assert!(selections > 0, "CD-01's table half compared nothing");
}

/// `CU-02`: a modular lane has no depth limit, because the wrap is the encode
/// rather than an overflow --- and an exact lane's limit is a property of the
/// declared bound, not of the library.
#[test]
fn lane_depth_follows_the_declaration_cu_02() {
    let exact = choose_for_rows(available_i32_exact(), Backend::Auto, 1 << 31, usize::MAX).unwrap();
    let modular =
        choose_for_rows(available_i32_modular(), Backend::Auto, 1 << 31, usize::MAX).unwrap();

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

/// `CB-07`, at the extremes of every alphabet a sequence declares.
///
/// A random fill will not find the one input where a paired-product instruction
/// overflows its intermediate: `madd` sums two products into an `i32`, and two
/// full-magnitude `i16` products are `2 * 2^30`, which is one bit past it. That
/// needs both operands to be exactly `i16::MIN` at both `k` of a pair, and a
/// generator reaches it with probability `2^-64`.
///
/// So the extremes are asked for by name --- the extremes of
/// [`KernelSpec::max_bound`], which is the alphabet the sequence claims. This is
/// the input that decides whether a kernel is exact on what it declares or
/// merely exact on likely data.
#[test]
fn the_extremes_of_every_alphabet_are_exact_cb_07() {
    /// The largest and smallest value an alphabet bounded by `bound` holds,
    /// within the element type.
    fn extremes(bound: u128, type_bound: u128) -> (i64, i64) {
        let b = bound.min(type_bound) as i64;
        // A bound of `B` admits magnitudes up to `B`, and the type's own
        // negative extreme is `-B`; the positive one is `B - 1` when the bound is
        // the type's, and `B` when the caller declared it.
        if bound >= type_bound {
            (-b, b - 1)
        } else {
            (-b, b)
        }
    }

    for spec in available_i16() {
        let (lo, hi) = extremes(spec.max_bound, 32768);
        for kc in [2usize, 4, 8, 16, 64].map(|k| k.div_ceil(spec.k_group) * spec.k_group) {
            for (av, bv) in [(lo, lo), (lo, hi), (hi, hi), (hi, lo)] {
                let (av, bv) = (av as i16, bv as i16);
                let pa = vec![av; spec.mr * kc];
                let pb = vec![bv; spec.nr * kc];
                let mut acc = vec![0i64; spec.mr * spec.nr];
                spec.mac_tile(kc, &pa, &pb, &mut acc);
                let want = (kc as i64) * i64::from(av) * i64::from(bv);
                assert!(
                    acc.iter().all(|&x| x == want),
                    "{} i16 (max_bound {}) at ({av}, {bv}) kc={kc}: got {:?}, want {want}",
                    spec.backend.as_str(),
                    spec.max_bound,
                    &acc[..4.min(acc.len())]
                );
            }
        }
    }

    for spec in available_i32_exact() {
        let (lo, hi) = extremes(spec.max_bound, 1 << 31);
        // One product at a time: `lane_depth` at the full bound is 1, and the
        // driver never offers this lane more than that.
        for (av, bv) in [(lo, lo), (lo, hi), (hi, hi)] {
            let (av, bv) = (av as i32, bv as i32);
            let kc = spec.k_group;
            let pa = vec![av; spec.mr * kc];
            let pb = vec![bv; spec.nr * kc];
            let mut acc = vec![0i64; spec.mr * spec.nr];
            spec.mac_tile(kc, &pa, &pb, &mut acc);
            let want = (kc as i64) * i64::from(av) * i64::from(bv);
            assert!(
                acc.iter().all(|&x| x == want),
                "{} i32 at ({av}, {bv}): got {:?}, want {want}",
                spec.backend.as_str(),
                &acc[..4.min(acc.len())]
            );
        }
    }
}

/// `CB-07`: selection respects the alphabet a sequence declares.
///
/// The `i16` family has two AVX2 sequences whose declared alphabets differ, so
/// this is where the rule is falsifiable: at a bound the paired sequence admits,
/// it is chosen; one past that, it is not considered at all. Not because it is
/// riskier there --- because there it computes a different number.
#[test]
fn selection_respects_the_declared_alphabet_cb_07() {
    let all: Vec<_> = available_i16().collect();
    if !all.iter().any(|s| s.backend == Backend::Avx2) {
        eprintln!("no AVX2 i16 sequence on this host; the cross-architecture job covers it");
        return;
    }
    let narrow = choose_for_rows(available_i16(), Backend::Auto, 32767, usize::MAX).unwrap();
    let full = choose_for_rows(available_i16(), Backend::Auto, 32768, usize::MAX).unwrap();
    assert!(narrow.max_bound >= 32767);
    assert!(full.max_bound >= 32768);
    assert_eq!(narrow.backend, Backend::Avx2);
    assert_eq!(full.backend, Backend::Avx2);
    // Two distinct sequences, or the rule is not being exercised.
    assert_ne!(
        narrow.k_group, full.k_group,
        "the two i16 sequences must actually differ"
    );
    // And every sequence a bound admits agrees with every other at that bound,
    // which is what makes the choice free.
    for bound in [1u128, 127, 32767, 32768] {
        let picked = choose_for_rows(available_i16(), Backend::Auto, bound, usize::MAX).unwrap();
        assert!(picked.max_bound >= bound, "bound {bound}");
    }
}

/// `CB-06`: every reduce sequence equals its own reference.
///
/// The reduce factorization puts the vector lanes on `k`, so its panel layout is
/// the contiguous one and its reference reads rows rather than strides. What this
/// asserts is that moving the lanes does not move the answer.
#[test]
fn every_reduce_sequence_equals_its_reference_cb_06() {
    use uor_matmul_kernels::{
        available_reduce_i16, available_reduce_i16_modular, available_reduce_i32_exact,
        available_reduce_i32_modular, available_reduce_i64_exact, available_reduce_i64_modular,
        available_reduce_i8, LaneLayout,
    };

    let mut seen = 0usize;
    for spec in available_reduce_i8() {
        assert_eq!(spec.nr, 1, "a reduce kernel produces one column");
        assert_eq!(spec.lane_layout, LaneLayout::Contiguous);
        for kc in depths(&spec) {
            let pa = fill(spec.mr * kc, 21, |v| v as i8);
            let pb: Vec<i8> = fill(kc, 22, |v| v as i8);
            let mut acc = vec![0i32; spec.mr];
            spec.mac_tile(kc, &pa, &pb, &mut acc);
            for i in 0..spec.mr {
                let row = &pa[i * kc..][..kc];
                let want = dot_ref(as_alphabet_full(row), as_alphabet_full(&pb[..kc])) as i32;
                assert_eq!(acc[i], want, "{} i8 reduce kc={kc}", spec.backend.as_str());
            }
        }
        seen += 1;
    }
    assert!(seen > 0);
    // Native only, as `CB-02`'s counterpart is: Miri models no vector intrinsics,
    // so the reference is the whole list there by design.
    if cfg!(any(target_arch = "x86_64", target_arch = "aarch64")) && !cfg!(miri) {
        assert!(
            seen > 1,
            "x86-64 and aarch64 both have a vector i8 reduce sequence past the reference"
        );
    }

    // The extremes too: the reduce sequences accumulate a whole row into one
    // lane, so their intermediate widths are a different question from the tile
    // sequences' and deserve the same input.
    for spec in available_reduce_i8() {
        for kc in [16usize, 64, 1024].map(|k| k.div_ceil(spec.k_group) * spec.k_group) {
            let pa = vec![i8::MIN; spec.mr * kc];
            let pb = vec![i8::MIN; kc];
            let mut acc = vec![0i32; spec.mr];
            spec.mac_tile(kc, &pa, &pb, &mut acc);
            let want = (kc as i32) * 128 * 128;
            assert!(
                acc.iter().all(|&x| x == want),
                "{} i8 reduce at the extreme, kc={kc}: {acc:?} want {want}",
                spec.backend.as_str()
            );
        }
    }

    for spec in available_reduce_i16() {
        for kc in depths(&spec) {
            let pa = fill(spec.mr * kc, 23, |v| v as i16);
            let pb: Vec<i16> = fill(kc, 24, |v| v as i16);
            let mut acc = vec![0i64; spec.mr];
            spec.mac_tile(kc, &pa, &pb, &mut acc);
            for i in 0..spec.mr {
                let want: i64 = (0..kc)
                    .map(|p| i64::from(pa[i * kc + p]) * i64::from(pb[p]))
                    .sum();
                assert_eq!(acc[i], want, "{} i16 reduce kc={kc}", spec.backend.as_str());
            }
        }
        // At the full alphabet, where a paired-product sequence would be wrong.
        let kc = 16usize.div_ceil(spec.k_group) * spec.k_group;
        for (av, bv) in [(i16::MIN, i16::MIN), (i16::MIN, i16::MAX)] {
            let pa = vec![av; spec.mr * kc];
            let pb = vec![bv; kc];
            let mut acc = vec![0i64; spec.mr];
            spec.mac_tile(kc, &pa, &pb, &mut acc);
            let want = (kc as i64) * i64::from(av) * i64::from(bv);
            assert!(
                acc.iter().all(|&x| x == want),
                "{} i16 reduce at ({av}, {bv}): {acc:?} want {want}",
                spec.backend.as_str()
            );
        }
    }

    for spec in available_reduce_i32_exact() {
        for kc in depths(&spec) {
            // Bounded so the reference sum fits an `i64` at every depth here.
            let pa = fill(spec.mr * kc, 25, |v| ((v & 0xFFFF) - 0x8000) as i32);
            let pb: Vec<i32> = fill(kc, 26, |v| ((v & 0xFFFF) - 0x8000) as i32);
            let mut acc = vec![0i64; spec.mr];
            spec.mac_tile(kc, &pa, &pb, &mut acc);
            for i in 0..spec.mr {
                let want: i64 = (0..kc)
                    .map(|p| i64::from(pa[i * kc + p]) * i64::from(pb[p]))
                    .sum();
                assert_eq!(acc[i], want, "{} i32 reduce kc={kc}", spec.backend.as_str());
            }
        }
    }

    for spec in available_reduce_i32_modular() {
        for kc in depths(&spec) {
            let pa = fill(spec.mr * kc, 27, |v| v.wrapping_mul(99_991) as i32);
            let pb: Vec<i32> = fill(kc, 28, |v| v.wrapping_mul(65_537) as i32);
            let mut acc = vec![0i32; spec.mr];
            spec.mac_tile(kc, &pa, &pb, &mut acc);
            for i in 0..spec.mr {
                let want = (0..kc).fold(0i32, |s, p| {
                    s.wrapping_add(pa[i * kc + p].wrapping_mul(pb[p]))
                });
                assert_eq!(
                    acc[i],
                    want,
                    "{} i32 modular reduce kc={kc}",
                    spec.backend.as_str()
                );
            }
        }
    }

    for spec in available_reduce_i16_modular() {
        for kc in depths(&spec) {
            let pa = fill(spec.mr * kc, 29, |v| v as i16);
            let pb: Vec<i16> = fill(kc, 30, |v| v as i16);
            let mut acc = vec![0i32; spec.mr];
            spec.mac_tile(kc, &pa, &pb, &mut acc);
            for i in 0..spec.mr {
                let want = (0..kc).fold(0i32, |s, p| {
                    s.wrapping_add(i32::from(pa[i * kc + p]).wrapping_mul(i32::from(pb[p])))
                });
                assert_eq!(
                    acc[i],
                    want,
                    "{} i16 modular reduce kc={kc}",
                    spec.backend.as_str()
                );
            }
        }
    }

    for spec in available_reduce_i64_exact() {
        // Bounded by `2^31`, so the *reference* sum fits an `i128` at every depth
        // below. The kernel's lane is bounded the same way: at the full `i64`
        // bound one product needs 126 bits and `lane_depth` is 1, so the driver
        // never offers this lane a deeper chunk, and testing past that would be
        // testing something no caller can reach.
        for kc in depths(&spec) {
            let pa = fill(spec.mr * kc, 31, |v| (v & 0xFFFF_FFFF) - 0x8000_0000);
            let pb: Vec<i64> = fill(kc, 32, |v| (v & 0xFFFF_FFFF) - 0x8000_0000);
            let mut acc = vec![0i128; spec.mr];
            spec.mac_tile(kc, &pa, &pb, &mut acc);
            for i in 0..spec.mr {
                let want: i128 = (0..kc)
                    .map(|p| i128::from(pa[i * kc + p]) * i128::from(pb[p]))
                    .sum();
                assert_eq!(acc[i], want, "{} i64 reduce kc={kc}", spec.backend.as_str());
            }
        }
    }

    for spec in available_reduce_i64_modular() {
        for kc in depths(&spec) {
            let pa = fill(spec.mr * kc, 33, |v| v.wrapping_mul(0x1000_0000_01B3));
            let pb: Vec<i64> = fill(kc, 34, |v| v.wrapping_mul(0x9E37_79B9));
            let mut acc = vec![0i64; spec.mr];
            spec.mac_tile(kc, &pa, &pb, &mut acc);
            for i in 0..spec.mr {
                let want = (0..kc).fold(0i64, |s, p| {
                    s.wrapping_add(pa[i * kc + p].wrapping_mul(pb[p]))
                });
                assert_eq!(
                    acc[i],
                    want,
                    "{} i64 modular reduce kc={kc}",
                    spec.backend.as_str()
                );
            }
        }
    }
}

/// Every declared number about a sequence is consistent with its own shape.
///
/// `products_per_step` is a claim about instructions, so no test can confirm it
/// by running the kernel --- but it can be bounded: a sequence cannot compute
/// more products in one instruction than its whole `k`-group of its whole tile,
/// and it cannot compute fewer than one. A transcription slip in either
/// direction lands outside that band, and a slip is the failure mode a hand-kept
/// table has.
///
/// The same for the rest: a panel has at least one lane, a group is at least one
/// step, and an exact sequence's lane holds at least one full-magnitude product
/// of the alphabet it declares.
#[test]
fn every_declaration_is_consistent_with_the_shape_cb_07() {
    fn check<E, L>(spec: &KernelSpec<E, L>, family: &str) {
        let name = spec.backend.as_str();
        assert!(spec.mr >= 1 && spec.nr >= 1, "{family}/{name}: empty panel");
        assert!(spec.k_group >= 1, "{family}/{name}: empty k-group");
        assert!(
            spec.products_per_step >= 1,
            "{family}/{name}: a sequence computes at least one product per step"
        );
        assert!(
            spec.products_per_step <= spec.mr * spec.nr * spec.k_group,
            "{family}/{name}: {} products per step is more than the whole tile-step, {}",
            spec.products_per_step,
            spec.mr * spec.nr * spec.k_group
        );
        assert!(
            spec.mr * spec.nr <= uor_matmul_kernels::MAX_TILE_LANES,
            "{family}/{name}: tile exceeds the buffer"
        );
        if matches!(spec.factorization, Factorization::Exact) {
            assert!(
                spec.lane_cap > 0,
                "{family}/{name}: an exact lane must hold something"
            );
            assert!(
                spec.max_bound >= 1,
                "{family}/{name}: an alphabet with no values is not an alphabet"
            );
        }
    }
    for s in every_i8_tile() {
        check(&s, "i8");
    }
    for s in uor_matmul_kernels::available_reduce_i8() {
        check(&s, "i8 reduce");
    }
    for s in available_i16() {
        check(&s, "i16");
    }
    for s in uor_matmul_kernels::available_reduce_i16() {
        check(&s, "i16 reduce");
    }
    for s in available_i16_modular() {
        check(&s, "i16 mod");
    }
    for s in uor_matmul_kernels::available_reduce_i16_modular() {
        check(&s, "i16 mod reduce");
    }
    for s in available_i32_exact() {
        check(&s, "i32");
    }
    for s in uor_matmul_kernels::available_reduce_i32_exact() {
        check(&s, "i32 reduce");
    }
    for s in available_i32_modular() {
        check(&s, "i32 mod");
    }
    for s in uor_matmul_kernels::available_reduce_i32_modular() {
        check(&s, "i32 mod reduce");
    }
    for s in uor_matmul_kernels::available_i64_exact() {
        check(&s, "i64");
    }
    for s in uor_matmul_kernels::available_reduce_i64_exact() {
        check(&s, "i64 reduce");
    }
    for s in available_i64_modular() {
        check(&s, "i64 mod");
    }
    for s in uor_matmul_kernels::available_reduce_i64_modular() {
        check(&s, "i64 mod reduce");
    }
}

// ---------------------------------------------------------------------------
// CB-08: the table sequences
// ---------------------------------------------------------------------------

/// Every table sequence equals the reference, lane for lane.
///
/// Both halves: the build, which is the only place the tabulated traversal
/// multiplies, and the gather, which is the only place it adds. A sequence that
/// disagreed with the reference on either would produce a different sum, and
/// this is the test that says so before `CD-13` sees it end to end.
///
/// The code spaces include one that is not a power of two, because the slab is
/// rounded up and the padding must not reach any answer.
#[test]
fn every_table_sequence_equals_the_reference_cb_08() {
    use uor_matmul_kernels::{available_table_i8, packed_slot, TableSpec};

    /// One codebook and one activation tile, deterministic and full-range.
    fn fill(len: usize, salt: u64) -> Vec<i8> {
        let mut s = 0x243F_6A88_85A3_08D3u64 ^ salt;
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (((s >> 33) % 255) as i64 - 127) as i8
            })
            .collect()
    }

    /// The activation tile in the layout `spec` declared.
    fn pack(flat: &[i8], rows: usize, block: usize, spec: &TableSpec<i8, i32>) -> Vec<i8> {
        let mut out = vec![0i8; rows * block];
        for t in 0..block {
            for i in 0..rows {
                out[packed_slot(t, i, rows, spec.k_group)] = flat[t * rows + i];
            }
        }
        out
    }

    let mut compared = 0usize;
    for &space in &corpus(&[16usize, 64, 200, 256], &[200, 256]) {
        for &block in &corpus(&[2usize, 4, 8], &[2, 8]) {
            let book = fill(space * block, 0xb00c ^ space as u64);
            for &rows in &corpus(&[1usize, 2, 4, 8, 16], &[1, 16]) {
                let flat = fill(rows * block, 0xac75 ^ rows as u64);
                for &group in &corpus(&[1usize, 2, 4, 8, 16], &[1, 16]) {
                    let specs: Vec<_> = available_table_i8(rows, group).collect();
                    let reference = specs[0];
                    assert_eq!(
                        reference.backend,
                        uor_matmul_core::Backend::Portable,
                        "the reference is listed first"
                    );

                    // The build, against the model rather than against the
                    // reference's own run. `T[c][i] = sum_t A[i][t] * D[c][t]` is
                    // the whole definition, and below eight rows no ISA offers a
                    // build sequence --- so `specs[1..]` is empty there and reading
                    // `want` off the reference compared it with itself. The gather
                    // half of this test was given a model oracle for exactly this
                    // reason, and this is the same oracle for the build.
                    let model = {
                        let mut out = vec![0i32; space * rows];
                        for c in 0..space {
                            for i in 0..rows {
                                let mut acc = 0i32;
                                for t in 0..block {
                                    acc += i32::from(flat[t * rows + i])
                                        * i32::from(book[c * block + t]);
                                }
                                out[c * rows + i] = acc;
                            }
                        }
                        out
                    };
                    let mut want = vec![0i32; space * rows];
                    reference.build(
                        space,
                        block,
                        &book,
                        &pack(&flat, rows, block, &reference),
                        &mut want,
                    );
                    assert_eq!(
                        want, model,
                        "the reference build disagrees with the model at space {space}, \
                         block {block}, rows {rows}"
                    );
                    compared += 1;
                    for spec in &specs[1..] {
                        let mut got = vec![0i32; space * rows];
                        spec.build(
                            space,
                            block,
                            &book,
                            &pack(&flat, rows, block, spec),
                            &mut got,
                        );
                        assert_eq!(
                            got, want,
                            "{:?} build disagrees at space {space}, block {block}, rows {rows}",
                            spec.backend
                        );
                        compared += 1;
                    }

                    // The gather, over a stack whose slab is the rounded space.
                    let codes = space.next_power_of_two();
                    let slab = codes * rows;
                    let depth = 5usize;
                    let mut stack = vec![0i32; depth * slab];
                    for slot in 0..depth {
                        // The live entries; the padding stays zero, exactly as
                        // `Table::new` leaves it.
                        let at = slot * slab;
                        stack[at..at + space * rows].copy_from_slice(
                            &fill(space * rows, slot as u64)
                                .iter()
                                .map(|&x| x as i32)
                                .collect::<Vec<_>>(),
                        );
                    }
                    let off: Vec<u32> = (0..depth * group)
                        .map(|i| ((i * 37 % space) * rows) as u32)
                        .collect();
                    // The oracle is the contract transcribed, not the reference's
                    // own run. `gather` binds the slab's code count to a
                    // compile-time constant, and at the tile heights no ISA
                    // offers a sequence for --- which is every height below
                    // eight --- the reference is the only party to the
                    // comparison, so reading `want` off it would compare the
                    // dispatch against itself and pass whatever it did.
                    let model = {
                        let mut out = vec![7i32; group * rows];
                        for slot in 0..depth {
                            for u in 0..group {
                                let at = off[slot * group + u] as usize & (slab - 1);
                                for i in 0..rows {
                                    out[u * rows + i] += stack[slot * slab + at + i];
                                }
                            }
                        }
                        out
                    };
                    let mut want = vec![7i32; group * rows];
                    reference.gather(depth, slab as u32, &stack, &off, &mut want);
                    assert_eq!(
                        want, model,
                        "the reference gather disagrees with the model at space {space}, \
                         rows {rows}, group {group}"
                    );
                    compared += 1;
                    for spec in &specs[1..] {
                        let mut got = vec![7i32; group * rows];
                        spec.gather(depth, slab as u32, &stack, &off, &mut got);
                        assert_eq!(
                            got, want,
                            "{:?} gather disagrees at space {space}, rows {rows}, group {group}",
                            spec.backend
                        );
                        compared += 1;
                    }

                    // Offsets that are NOT multiples of the tile height. The
                    // driver pre-scales, but `gather` is a safe public method and
                    // its safety cannot rest on a caller doing so: without the
                    // sub-row bits cleared, an offset near the end of the slab
                    // starts the read inside the last entry and runs `rows - 1`
                    // lanes past it -- a panic in the reference and an
                    // out-of-bounds read in every ISA sequence. Every sequence
                    // must agree on the row-aligned address, and none may fault.
                    if rows > 1 {
                        let ragged: Vec<u32> = (0..depth * group)
                            .map(|i| ((i * 37 + 1) % (codes * rows)) as u32)
                            .collect();
                        let mut model = vec![0i32; group * rows];
                        for slot in 0..depth {
                            for u in 0..group {
                                let at =
                                    ragged[slot * group + u] as usize & (slab - 1) & !(rows - 1);
                                for i in 0..rows {
                                    model[u * rows + i] += stack[slot * slab + at + i];
                                }
                            }
                        }
                        for spec in &specs {
                            let mut got = vec![0i32; group * rows];
                            spec.gather(depth, slab as u32, &stack, &ragged, &mut got);
                            assert_eq!(
                                got, model,
                                "{:?} disagrees on a ragged offset at space {space}, \
                                 rows {rows}, group {group}",
                                spec.backend
                            );
                            compared += 1;
                        }
                    }

                    // The same reduction read from a code stream instead of an
                    // index stream. Only where a codec could claim it: the
                    // enumeration has to be addressed by the code, which needs a
                    // power-of-two space.
                    if codes == space {
                        let stride = depth + 3;
                        let stream: Vec<u16> = (0..(group - 1) * stride + depth)
                            .map(|i| ((i * 37) % space) as u16)
                            .collect();
                        let by_off: Vec<u32> = (0..depth * group)
                            .map(|i| {
                                let (slot, u) = (i / group, i % group);
                                (stream[u * stride + slot] as usize % space * rows) as u32
                            })
                            .collect();
                        // The same, read from the code stream: the shift the
                        // boundary derives is `rows.trailing_zeros()`, so the
                        // entry's address is the masked code scaled by the tile
                        // height and nothing else.
                        let model = {
                            let mut out = vec![-3i32; group * rows];
                            for slot in 0..depth {
                                for u in 0..group {
                                    let at =
                                        (stream[u * stride + slot] as usize & (codes - 1)) * rows;
                                    for i in 0..rows {
                                        out[u * rows + i] += stack[slot * slab + at + i];
                                    }
                                }
                            }
                            out
                        };
                        let mut want = vec![-3i32; group * rows];
                        reference.gather(depth, slab as u32, &stack, &by_off, &mut want);
                        assert_eq!(
                            want, model,
                            "the reference gather disagrees with the model read by code at \
                             space {space}, rows {rows}, group {group}"
                        );
                        compared += 1;
                        for spec in &specs {
                            let mut got = vec![-3i32; group * rows];
                            spec.gather_codes(
                                depth,
                                slab as u32,
                                &stack,
                                &stream,
                                stride,
                                &mut got,
                            );
                            assert_eq!(
                                got, want,
                                "{:?} gather_codes disagrees at space {space}, rows {rows}, \
                                 group {group}",
                                spec.backend
                            );
                            compared += 1;
                        }
                    }
                }
            }
        }
    }
    // `dispatch_slab!`'s wildcard, which is what keeps the enumeration from being
    // a ceiling (R8). Sixteen exponents cover every code space a `u16` can name,
    // and no shipped codec exceeds them --- but `slab` is a `u32` the caller
    // passes, so a slab of `2^17` lane words at one row asks for a code count the
    // list does not have, and that arm binds the constant to zero and reads the
    // slab from its argument instead. Untested, that arm was the one line standing
    // between the enumeration and a limit.
    for shift in [17u32, 18, 20] {
        let codes = 1usize << shift;
        let (rows, group, depth) = (1usize, 1usize, 3usize);
        let slab = codes * rows;
        let stack: Vec<i32> = (0..depth * slab)
            .map(|i| (i % 4096) as i32 - 2048)
            .collect();
        let off: Vec<u32> = (0..depth * group)
            .map(|i| ((i * 37 * 101) % codes) as u32)
            .collect();
        let mut model = vec![0i32; group * rows];
        for slot in 0..depth {
            for u in 0..group {
                let at = off[slot * group + u] as usize & (slab - 1);
                model[u * rows] += stack[slot * slab + at];
            }
        }
        for spec in available_table_i8(rows, group) {
            let mut got = vec![0i32; group * rows];
            spec.gather(depth, slab as u32, &stack, &off, &mut got);
            assert_eq!(
                got, model,
                "{:?} disagrees on a code count past the dispatch list, 2^{shift}",
                spec.backend
            );
            compared += 1;
        }
    }

    assert!(
        compared > 0,
        "CB-08 compared nothing; on a host with no table sequence beyond the \
         reference this gate would pass vacuously"
    );
}

/// The same, for the family whose lane is 64 bits wide.
///
/// A separate test because the element and the lane are both different, and the
/// bound is: `madd` sums its pair into an `i32`, which the full `i16` alphabet
/// leaves, so the vector sequence declares 32767 and the alphabet is held at it
/// here. A sequence offered outside its declared alphabet is `CB-07`'s business
/// and not this one's.
#[test]
fn every_i16_table_sequence_equals_the_reference_cb_08() {
    use uor_matmul_kernels::{available_table_i16, packed_slot, TableSpec};

    fn fill(len: usize, salt: u64) -> Vec<i16> {
        let mut s = 0x243F_6A88_85A3_08D3u64 ^ salt;
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                // The extremes of the declared alphabet, which is where a
                // sequence with a narrow intermediate fails first.
                (((s >> 33) % 65535) as i64 - 32767) as i16
            })
            .collect()
    }

    fn pack(flat: &[i16], rows: usize, block: usize, spec: &TableSpec<i16, i64>) -> Vec<i16> {
        let mut out = vec![0i16; rows * block];
        for t in 0..block {
            for i in 0..rows {
                out[packed_slot(t, i, rows, spec.k_group)] = flat[t * rows + i];
            }
        }
        out
    }

    let mut compared = 0usize;
    for &space in &corpus(&[16usize, 200, 256], &[200, 256]) {
        for &block in &corpus(&[2usize, 8], &[2]) {
            let book = fill(space * block, 0x16b0 ^ space as u64);
            for &rows in &corpus(&[1usize, 8, 16], &[1, 16]) {
                let flat = fill(rows * block, 0x16ac ^ rows as u64);
                for &group in &[1usize, 2] {
                    let specs: Vec<_> = available_table_i16(rows, group).collect();
                    let reference = specs[0];

                    // The model, for the reason the 32-bit lane has one: below
                    // eight rows no ISA offers an `i16` build either.
                    let model = {
                        let mut out = vec![0i64; space * rows];
                        for c in 0..space {
                            for i in 0..rows {
                                let mut acc = 0i64;
                                for t in 0..block {
                                    acc += i64::from(flat[t * rows + i])
                                        * i64::from(book[c * block + t]);
                                }
                                out[c * rows + i] = acc;
                            }
                        }
                        out
                    };
                    let mut want = vec![0i64; space * rows];
                    reference.build(
                        space,
                        block,
                        &book,
                        &pack(&flat, rows, block, &reference),
                        &mut want,
                    );
                    assert_eq!(
                        want, model,
                        "the reference i16 build disagrees with the model at space {space}, \
                         block {block}, rows {rows}"
                    );
                    compared += 1;
                    for spec in &specs[1..] {
                        let mut got = vec![0i64; space * rows];
                        spec.build(
                            space,
                            block,
                            &book,
                            &pack(&flat, rows, block, spec),
                            &mut got,
                        );
                        assert_eq!(
                            got, want,
                            "{:?} i16 build disagrees at space {space}, block {block}, rows {rows}",
                            spec.backend
                        );
                        compared += 1;
                    }

                    let codes = space.next_power_of_two();
                    let slab = codes * rows;
                    let depth = 5usize;
                    let mut stack = vec![0i64; depth * slab];
                    for slot in 0..depth {
                        let at = slot * slab;
                        for (i, cell) in stack[at..at + space * rows].iter_mut().enumerate() {
                            *cell = ((i as i64 + slot as i64 * 7) % 4096) - 2048;
                        }
                    }
                    let off: Vec<u32> = (0..depth * group)
                        .map(|i| ((i * 37 % space) * rows) as u32)
                        .collect();
                    // The contract transcribed, for the same reason it is
                    // transcribed at the 32-bit lane: below eight rows the
                    // reference is the only sequence there is, so it has to be
                    // read against the model rather than against itself.
                    let model = {
                        let mut out = vec![11i64; group * rows];
                        for slot in 0..depth {
                            for u in 0..group {
                                let at = off[slot * group + u] as usize & (slab - 1);
                                for i in 0..rows {
                                    out[u * rows + i] += stack[slot * slab + at + i];
                                }
                            }
                        }
                        out
                    };
                    let mut want = vec![11i64; group * rows];
                    reference.gather(depth, slab as u32, &stack, &off, &mut want);
                    assert_eq!(
                        want, model,
                        "the reference i16 gather disagrees with the model at space {space}, \
                         rows {rows}, group {group}"
                    );
                    compared += 1;
                    for spec in &specs[1..] {
                        let mut got = vec![11i64; group * rows];
                        spec.gather(depth, slab as u32, &stack, &off, &mut got);
                        assert_eq!(
                            got, want,
                            "{:?} i16 gather disagrees at space {space}, rows {rows}",
                            spec.backend
                        );
                        compared += 1;
                    }

                    if codes == space {
                        let stride = depth + 3;
                        let stream: Vec<u16> = (0..(group - 1) * stride + depth)
                            .map(|i| ((i * 37) % space) as u16)
                            .collect();
                        let by_off: Vec<u32> = (0..depth * group)
                            .map(|i| {
                                let (slot, u) = (i / group, i % group);
                                (stream[u * stride + slot] as usize % space * rows) as u32
                            })
                            .collect();
                        let model = {
                            let mut out = vec![-5i64; group * rows];
                            for slot in 0..depth {
                                for u in 0..group {
                                    let at =
                                        (stream[u * stride + slot] as usize & (codes - 1)) * rows;
                                    for i in 0..rows {
                                        out[u * rows + i] += stack[slot * slab + at + i];
                                    }
                                }
                            }
                            out
                        };
                        let mut want = vec![-5i64; group * rows];
                        reference.gather(depth, slab as u32, &stack, &by_off, &mut want);
                        assert_eq!(
                            want, model,
                            "the reference i16 gather disagrees with the model read by code at \
                             space {space}, rows {rows}, group {group}"
                        );
                        compared += 1;
                        for spec in &specs {
                            let mut got = vec![-5i64; group * rows];
                            spec.gather_codes(
                                depth,
                                slab as u32,
                                &stack,
                                &stream,
                                stride,
                                &mut got,
                            );
                            assert_eq!(
                                got, want,
                                "{:?} i16 gather_codes disagrees at space {space}, rows {rows}",
                                spec.backend
                            );
                            compared += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(compared > 0, "CB-08 compared nothing for the 64-bit lane");
}

/// `CB-09`: every modular table sequence equals the portable modular
/// reference, lane for lane.
///
/// The modular lane is `Z/2^w`: the build's products wrap and the gather's
/// adds wrap, and both are the ring's own operations, so the model oracle is
/// written in wrapping arithmetic. As in `CB-08`, the reference is read
/// against the model rather than against itself, because below eight rows no
/// ISA offers a sequence and the reference is the only party to the
/// comparison. For `i64` the reference is the *only* sequence at every height
/// --- the build's multiply is the table's only one, and no SIMD integer
/// multiply reaches an `i64` lane, which is why the dense family is
/// portable-only too.
#[test]
fn every_modular_table_sequence_equals_the_reference_cb_09() {
    use uor_matmul_kernels::{
        available_table_i32_modular, available_table_i64_modular, packed_slot, Mod32, Mod64,
        TableSpec,
    };

    /// Full-range fills: the modular lane declares no bound, so the extremes
    /// of the element type are ordinary inputs here, and a product of two of
    /// them wraps on purpose.
    fn fill32(len: usize, salt: u64) -> Vec<i32> {
        fill(len, salt, |v| v.wrapping_mul(0x9E37_79B9) as i32)
    }

    fn fill64(len: usize, salt: u64) -> Vec<i64> {
        fill(len, salt, |v| {
            v.wrapping_mul(0x9E37_79B9_7F4A_7C15u64 as i64)
        })
    }

    fn pack32(flat: &[i32], rows: usize, block: usize, spec: &TableSpec<i32, Mod32>) -> Vec<i32> {
        let mut out = vec![0i32; rows * block];
        for t in 0..block {
            for i in 0..rows {
                out[packed_slot(t, i, rows, spec.k_group)] = flat[t * rows + i];
            }
        }
        out
    }

    fn pack64(flat: &[i64], rows: usize, block: usize, spec: &TableSpec<i64, Mod64>) -> Vec<i64> {
        let mut out = vec![0i64; rows * block];
        for t in 0..block {
            for i in 0..rows {
                out[packed_slot(t, i, rows, spec.k_group)] = flat[t * rows + i];
            }
        }
        out
    }

    let mut compared = 0usize;
    for &space in &corpus(&[16usize, 64, 200, 256], &[200, 256]) {
        for &block in &corpus(&[2usize, 4, 8], &[2, 8]) {
            let book = fill32(space * block, 0xb32c ^ space as u64);
            for &rows in &corpus(&[1usize, 2, 4, 8, 16], &[1, 16]) {
                let flat = fill32(rows * block, 0xa32c ^ rows as u64);
                for &group in &corpus(&[1usize, 2, 4, 8, 16], &[1, 16]) {
                    let specs: Vec<_> = available_table_i32_modular(rows, group).collect();
                    let reference = specs[0];
                    assert_eq!(
                        reference.backend,
                        uor_matmul_core::Backend::Portable,
                        "the reference is listed first"
                    );

                    // The build, against the model in the ring itself. `T[c][i]
                    // = sum_t A[i][t] * D[c][t]` with every operation taken mod
                    // `2^32` is the whole definition.
                    let model = {
                        let mut out = vec![0i32; space * rows];
                        for c in 0..space {
                            for i in 0..rows {
                                let mut acc = 0i32;
                                for t in 0..block {
                                    acc = acc.wrapping_add(
                                        flat[t * rows + i].wrapping_mul(book[c * block + t]),
                                    );
                                }
                                out[c * rows + i] = acc;
                            }
                        }
                        out
                    };
                    let mut want = vec![Mod32(0); space * rows];
                    reference.build(
                        space,
                        block,
                        &book,
                        &pack32(&flat, rows, block, &reference),
                        &mut want,
                    );
                    assert_eq!(
                        want.iter().map(|m| m.0).collect::<Vec<_>>(),
                        model,
                        "the reference mod32 build disagrees with the model at space {space}, \
                         block {block}, rows {rows}"
                    );
                    compared += 1;
                    for spec in &specs[1..] {
                        let mut got = vec![Mod32(0); space * rows];
                        spec.build(
                            space,
                            block,
                            &book,
                            &pack32(&flat, rows, block, spec),
                            &mut got,
                        );
                        assert_eq!(
                            got, want,
                            "{:?} mod32 build disagrees at space {space}, block {block}, \
                             rows {rows}",
                            spec.backend
                        );
                        compared += 1;
                    }

                    // The gather, over a stack whose slab is the rounded space,
                    // with wrapping adds on both sides.
                    let codes = space.next_power_of_two();
                    let slab = codes * rows;
                    let depth = 5usize;
                    let mut stack = vec![Mod32(0); depth * slab];
                    for slot in 0..depth {
                        let at = slot * slab;
                        for (i, cell) in stack[at..at + space * rows].iter_mut().enumerate() {
                            *cell = Mod32(fill32(1, (slot * space * rows + i) as u64)[0]);
                        }
                    }
                    let off: Vec<u32> = (0..depth * group)
                        .map(|i| ((i * 37 % space) * rows) as u32)
                        .collect();
                    let model = {
                        let mut out = vec![7i32; group * rows];
                        for slot in 0..depth {
                            for u in 0..group {
                                let at = off[slot * group + u] as usize & (slab - 1);
                                for i in 0..rows {
                                    out[u * rows + i] = out[u * rows + i]
                                        .wrapping_add(stack[slot * slab + at + i].0);
                                }
                            }
                        }
                        out
                    };
                    let mut want = vec![Mod32(7); group * rows];
                    reference.gather(depth, slab as u32, &stack, &off, &mut want);
                    assert_eq!(
                        want.iter().map(|m| m.0).collect::<Vec<_>>(),
                        model,
                        "the reference mod32 gather disagrees with the model at space {space}, \
                         rows {rows}, group {group}"
                    );
                    compared += 1;
                    for spec in &specs[1..] {
                        let mut got = vec![Mod32(7); group * rows];
                        spec.gather(depth, slab as u32, &stack, &off, &mut got);
                        assert_eq!(
                            got, want,
                            "{:?} mod32 gather disagrees at space {space}, rows {rows}, \
                             group {group}",
                            spec.backend
                        );
                        compared += 1;
                    }

                    // Ragged offsets, exactly as `CB-08` requires of the exact
                    // lane: the sub-row bits are cleared, every read is
                    // row-aligned, and every sequence agrees on which.
                    if rows > 1 {
                        let ragged: Vec<u32> = (0..depth * group)
                            .map(|i| ((i * 37 + 1) % (codes * rows)) as u32)
                            .collect();
                        let mut model = vec![0i32; group * rows];
                        for slot in 0..depth {
                            for u in 0..group {
                                let at =
                                    ragged[slot * group + u] as usize & (slab - 1) & !(rows - 1);
                                for i in 0..rows {
                                    model[u * rows + i] = model[u * rows + i]
                                        .wrapping_add(stack[slot * slab + at + i].0);
                                }
                            }
                        }
                        for spec in &specs {
                            let mut got = vec![Mod32(0); group * rows];
                            spec.gather(depth, slab as u32, &stack, &ragged, &mut got);
                            assert_eq!(
                                got.iter().map(|m| m.0).collect::<Vec<_>>(),
                                model,
                                "{:?} mod32 disagrees on a ragged offset at space {space}, \
                                 rows {rows}, group {group}",
                                spec.backend
                            );
                            compared += 1;
                        }
                    }

                    // The same reduction read from a code stream.
                    if codes == space {
                        let stride = depth + 3;
                        let stream: Vec<u16> = (0..(group - 1) * stride + depth)
                            .map(|i| ((i * 37) % space) as u16)
                            .collect();
                        let model = {
                            let mut out = vec![-3i32; group * rows];
                            for slot in 0..depth {
                                for u in 0..group {
                                    let at =
                                        (stream[u * stride + slot] as usize & (codes - 1)) * rows;
                                    for i in 0..rows {
                                        out[u * rows + i] = out[u * rows + i]
                                            .wrapping_add(stack[slot * slab + at + i].0);
                                    }
                                }
                            }
                            out
                        };
                        for spec in &specs {
                            let mut got = vec![Mod32(-3); group * rows];
                            spec.gather_codes(
                                depth,
                                slab as u32,
                                &stack,
                                &stream,
                                stride,
                                &mut got,
                            );
                            assert_eq!(
                                got.iter().map(|m| m.0).collect::<Vec<_>>(),
                                model,
                                "{:?} mod32 gather_codes disagrees at space {space}, rows \
                                 {rows}, group {group}",
                                spec.backend
                            );
                            compared += 1;
                        }
                    }
                }
            }
        }
    }

    // The `i64` half: the portable reference is the whole list, so the sweep
    // reads it against the model at every shape --- the comparison `CB-08`
    // gives the families that have ISA sequences, with the ISA half empty.
    for &space in &corpus(&[16usize, 200, 256], &[200, 256]) {
        for &block in &corpus(&[2usize, 8], &[2]) {
            let book = fill64(space * block, 0xb64c ^ space as u64);
            for &rows in &corpus(&[1usize, 8, 16], &[1, 16]) {
                let flat = fill64(rows * block, 0xa64c ^ rows as u64);
                for &group in &[1usize, 2] {
                    let specs: Vec<_> = available_table_i64_modular(rows, group).collect();
                    assert_eq!(
                        specs.len(),
                        1,
                        "the i64 modular table is portable-only: no SIMD integer multiply \
                         reaches the lane"
                    );
                    let reference = specs[0];
                    let model = {
                        let mut out = vec![0i64; space * rows];
                        for c in 0..space {
                            for i in 0..rows {
                                let mut acc = 0i64;
                                for t in 0..block {
                                    acc = acc.wrapping_add(
                                        flat[t * rows + i].wrapping_mul(book[c * block + t]),
                                    );
                                }
                                out[c * rows + i] = acc;
                            }
                        }
                        out
                    };
                    let mut want = vec![Mod64(0); space * rows];
                    reference.build(
                        space,
                        block,
                        &book,
                        &pack64(&flat, rows, block, &reference),
                        &mut want,
                    );
                    assert_eq!(
                        want.iter().map(|m| m.0).collect::<Vec<_>>(),
                        model,
                        "the reference mod64 build disagrees with the model at space {space}, \
                         block {block}, rows {rows}"
                    );
                    compared += 1;

                    let codes = space.next_power_of_two();
                    let slab = codes * rows;
                    let depth = 5usize;
                    let mut stack = vec![Mod64(0); depth * slab];
                    for slot in 0..depth {
                        let at = slot * slab;
                        for (i, cell) in stack[at..at + space * rows].iter_mut().enumerate() {
                            *cell = Mod64(fill64(1, (slot * space * rows + i) as u64 | 0x5A)[0]);
                        }
                    }
                    let off: Vec<u32> = (0..depth * group)
                        .map(|i| ((i * 37 % space) * rows) as u32)
                        .collect();
                    let model = {
                        let mut out = vec![11i64; group * rows];
                        for slot in 0..depth {
                            for u in 0..group {
                                let at = off[slot * group + u] as usize & (slab - 1);
                                for i in 0..rows {
                                    out[u * rows + i] = out[u * rows + i]
                                        .wrapping_add(stack[slot * slab + at + i].0);
                                }
                            }
                        }
                        out
                    };
                    let mut want = vec![Mod64(11); group * rows];
                    reference.gather(depth, slab as u32, &stack, &off, &mut want);
                    assert_eq!(
                        want.iter().map(|m| m.0).collect::<Vec<_>>(),
                        model,
                        "the reference mod64 gather disagrees with the model at space {space}, \
                         rows {rows}, group {group}"
                    );
                    compared += 1;
                }
            }
        }
    }

    assert!(
        compared > 0,
        "CB-09 compared nothing; on a host with no modular table sequence beyond the \
         reference this gate would pass vacuously"
    );
}
