//! `CB-01` .. `CB-05`, `CD-01`, `CU-02`, `CU-03`: every backend, in every
//! family, equals its reference.
//!
//! The point of a kernel table is that adding an instruction cannot change an
//! answer. These tests are what makes that a fact rather than an intention:
//! each family has a reference, and each backend is compared against it on
//! shapes chosen to hit every tail and every threshold.
//!
//! The check bodies live in [`parity`], shared with the Cortex-M executor
//! (`CB-11`): this file is the host harness --- the whole corpus, `Vec`
//! scratch, and the host-only assertions --- and the executor is the same
//! bodies on the reduced corpus with static scratch. One implementation, two
//! harnesses (R13).
//!
//! The reduced corpus is the one Miri walks, and the discipline is Miri's: it
//! reduces the number of *instances*, not the set of *paths* --- every arm of
//! `dispatch_run!` and `dispatch_slab!` is the same macro body at different
//! constants, so a narrow arm and a wide arm together exercise the code, and
//! keeping a code space that is not a power of two keeps the padding claim.
//! That is what keeps the Miri job inside a CI budget instead of running past
//! the six-hour ceiling and being cancelled, which is what it did on every
//! push it was ever enabled for. The native run takes the whole corpus.

use uor_matmul_core::Backend;
use uor_matmul_kernels::{
    available_i16, available_i16_modular, available_i32_exact, available_i32_modular,
    available_i64_exact, available_i64_modular, available_i8, available_i8_narrow,
    available_reduce_i16, available_reduce_i16_modular, available_reduce_i32_exact,
    available_reduce_i32_modular, available_reduce_i64_exact, available_reduce_i64_modular,
    available_reduce_i8, choose_for_rows, parity, portable_i8, Factorization, KernelSpec,
};

/// The depths this harness walks: the whole corpus natively, the reduced one
/// under Miri.
fn depths() -> &'static [usize] {
    if cfg!(miri) {
        parity::DEPTHS_REDUCED
    } else {
        parity::DEPTHS_FULL
    }
}

/// The table corpora, with the same native/Miri split as [`depths`].
fn table_32() -> parity::TableCorpus {
    if cfg!(miri) {
        parity::TABLE_REDUCED
    } else {
        parity::TABLE_32_FULL
    }
}

/// The 64-bit lane's table corpus; the reduced corpus serves every family.
fn table_64() -> parity::TableCorpus {
    if cfg!(miri) {
        parity::TABLE_REDUCED
    } else {
        parity::TABLE_64_FULL
    }
}

/// The bound-1 corpus, natively and under Miri.
fn bound1() -> parity::Bound1Corpus {
    if cfg!(miri) {
        parity::BOUND1_REDUCED
    } else {
        parity::BOUND1_FULL
    }
}

/// The deepest panel any spec in `specs` reaches over `corpus`.
fn kc_max<E, L>(specs: &[KernelSpec<E, L>], corpus: &[usize]) -> usize {
    specs
        .iter()
        .flat_map(|s| parity::depths_of(s.k_group, corpus))
        .max()
        .unwrap_or(0)
}

/// `Vec`-backed tile scratch for the host harness.
struct TileVecs<E, L> {
    pa: Vec<E>,
    pb: Vec<E>,
    lane_a: Vec<E>,
    lane_b: Vec<E>,
    acc: Vec<L>,
    want: Vec<L>,
}

impl<E: Copy + Default, L: Copy + Default> TileVecs<E, L> {
    fn for_specs(specs: &[KernelSpec<E, L>], corpus: &[usize]) -> Self {
        let kc = kc_max(specs, corpus);
        let mr = specs.iter().map(|s| s.mr).max().unwrap_or(0);
        let nr = specs.iter().map(|s| s.nr).max().unwrap_or(0);
        TileVecs {
            pa: vec![E::default(); mr * kc],
            pb: vec![E::default(); nr * kc],
            lane_a: vec![E::default(); kc],
            lane_b: vec![E::default(); kc],
            acc: vec![L::default(); mr * nr],
            want: vec![L::default(); mr * nr],
        }
    }

    fn scratch(&mut self) -> parity::TileScratch<'_, E, L> {
        parity::TileScratch {
            pa: &mut self.pa,
            pb: &mut self.pb,
            lane_a: &mut self.lane_a,
            lane_b: &mut self.lane_b,
            acc: &mut self.acc,
            want: &mut self.want,
        }
    }
}

/// `Vec`-backed reduce scratch for the host harness.
struct ReduceVecs<E, L> {
    pa: Vec<E>,
    pb: Vec<E>,
    acc: Vec<L>,
}

impl<E: Copy + Default, L: Copy + Default> ReduceVecs<E, L> {
    fn for_specs(specs: &[KernelSpec<E, L>], corpus: &[usize]) -> Self {
        let kc = kc_max(specs, corpus);
        let mr = specs.iter().map(|s| s.mr).max().unwrap_or(0);
        ReduceVecs {
            pa: vec![E::default(); mr * kc],
            pb: vec![E::default(); kc],
            acc: vec![L::default(); mr],
        }
    }

    fn scratch(&mut self) -> parity::ReduceScratch<'_, E, L> {
        parity::ReduceScratch {
            pa: &mut self.pa,
            pb: &mut self.pb,
            acc: &mut self.acc,
        }
    }
}

/// `Vec`-backed table scratch for the host harness.
struct TableVecs<E, L> {
    book: Vec<E>,
    flat: Vec<E>,
    packed: Vec<E>,
    model: Vec<L>,
    out: Vec<L>,
    stack: Vec<L>,
    off: Vec<u32>,
    stream: Vec<u16>,
    lane_a: Vec<L>,
    lane_b: Vec<L>,
}

impl<E: Copy + Default, L: Copy> TableVecs<E, L> {
    fn for_corpus(c: &parity::TableCorpus, zero: L) -> Self {
        let max = |xs: &[usize]| xs.iter().copied().max().unwrap_or(0);
        let (space, block, rows, group) =
            (max(c.spaces), max(c.blocks), max(c.rows), max(c.groups));
        let slab = space.next_power_of_two() * rows;
        TableVecs {
            book: vec![E::default(); space * block],
            flat: vec![E::default(); rows * block],
            packed: vec![E::default(); rows * block],
            model: vec![zero; space * rows],
            out: vec![zero; space * rows],
            stack: vec![zero; c.depth * slab],
            off: vec![0; c.depth * group],
            stream: vec![0; (group - 1) * (c.depth + 3) + c.depth],
            lane_a: vec![zero; group * rows],
            lane_b: vec![zero; group * rows],
        }
    }

    fn for_bound1(c: &parity::Bound1Corpus, zero: L) -> Self {
        let max = |xs: &[usize]| xs.iter().copied().max().unwrap_or(0);
        let (space, block) = c
            .pairs
            .iter()
            .copied()
            .fold((0, 0), |(s, b), (s2, b2)| (s.max(s2), b.max(b2)));
        let rows = max(c.rows);
        TableVecs {
            book: vec![E::default(); space * block],
            flat: vec![E::default(); rows * block],
            packed: vec![E::default(); rows * block],
            model: vec![zero; space * rows],
            out: vec![zero; space * rows],
            stack: Vec::new(),
            off: Vec::new(),
            stream: Vec::new(),
            lane_a: Vec::new(),
            lane_b: Vec::new(),
        }
    }

    fn scratch(&mut self) -> parity::TableScratch<'_, E, L> {
        parity::TableScratch {
            book: &mut self.book,
            flat: &mut self.flat,
            packed: &mut self.packed,
            model: &mut self.model,
            out: &mut self.out,
            stack: &mut self.stack,
            off: &mut self.off,
            stream: &mut self.stream,
            lane_a: &mut self.lane_a,
            lane_b: &mut self.lane_b,
        }
    }
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

/// Element `p` of lane `l`, read the way the kernel reads it.
///
/// The panel layout is the kernel contract, so the reference goes through the
/// crate's own [`uor_matmul_kernels::packed_slot`] rather than restating it. A
/// kernel that misreads its panel therefore fails here, which is the whole
/// point.
fn at<T: Copy>(panel: &[T], p: usize, lane: usize, lanes: usize, group: usize) -> T {
    panel[uor_matmul_kernels::packed_slot(p, lane, lanes, group)]
}

/// The exact `mr x nr` tile for an `i8` kernel, from the core's own
/// accumulation. This is the link that anchors the whole chain to `dot_ref`.
fn reference_i8(spec: &KernelSpec<i8, i32>, kc: usize, pa: &[i8], pb: &[i8]) -> Vec<i32> {
    use uor_matmul_core::{as_alphabet_full, dot_ref};
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

/// `CB-01`: the portable kernel equals `dot_ref` on the whole corpus.
#[test]
fn portable_equals_dot_ref_cb_01() {
    let spec = portable_i8();
    let mut vecs = TileVecs::for_specs(&[spec], depths());
    let compared = parity::tile_parity(
        "CB-01",
        core::iter::once(spec),
        depths(),
        (0, 0x5A),
        (|v| v as i8, |v| v as i8),
        parity::dot_i8,
        &mut vecs.scratch(),
    );
    assert!(compared > 0, "CB-01 compared nothing");
}

/// Every `i8` tile sequence this build can run, at every panel width.
///
/// The narrow panels live in their own list because the driver only asks for
/// one when the shape is narrower than the tile --- but a differential test has
/// no such condition. A sequence outside the net is a sequence nothing checks.
fn every_i8_tile() -> impl Iterator<Item = KernelSpec<i8, i32>> {
    available_i8().chain(available_i8_narrow())
}

/// `CB-02`: every `i8` backend this host can run equals the portable
/// reference, byte for byte.
#[test]
fn every_i8_backend_equals_portable_cb_02() {
    let specs: Vec<_> = every_i8_tile().collect();
    let mut vecs = TileVecs::for_specs(&specs, depths());
    let compared = parity::tile_parity(
        "CB-02",
        specs.iter().copied(),
        depths(),
        (0xC0, 0x0D),
        (|v| v as i8, |v| v as i8),
        parity::dot_i8,
        &mut vecs.scratch(),
    );
    let names: Vec<_> = specs.iter().map(|s| s.backend.as_str()).collect();
    eprintln!("CB-02: {} i8 backend(s): {}", names.len(), names.join(", "));
    assert!(compared > 0, "at least the portable kernel must have run");
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
fn check_named(check: &'static str, backend: Backend) -> bool {
    let Some(spec) = available_i8().find(|s| s.backend == backend) else {
        eprintln!(
            "{}: not available on this host; the cross-architecture CI job runs it",
            backend.as_str()
        );
        return false;
    };
    let mut vecs = TileVecs::for_specs(&[spec], depths());
    parity::tile_parity(
        check,
        core::iter::once(spec),
        depths(),
        (0xAB, 0xCD),
        (|v| v as i8, |v| v as i8),
        parity::dot_i8,
        &mut vecs.scratch(),
    );
    true
}

/// `CB-03`: AVX-512 VNNI equals portable, on both of its sequences.
#[test]
fn avx512vnni_equals_portable_cb_03() {
    if check_named("CB-03", Backend::Avx512Vnni) {
        let sequences = available_i8()
            .filter(|s| s.backend == Backend::Avx512Vnni)
            .count();
        assert!(sequences >= 2, "both VNNI sequences must be registered");
    }
}

/// `CB-04`: NEON and NEON dotprod equal portable.
#[test]
fn neon_equals_portable_cb_04() {
    let _ = check_named("CB-04", Backend::Neon);
    let _ = check_named("CB-04", Backend::NeonDotprod);
}

/// `CB-05`: wasm SIMD128 equals portable. A SIMD128-off build runs the portable
/// kernel, so "off equals on" is the composition of this with `CB-01`.
#[test]
fn wasm_simd128_equals_portable_cb_05() {
    let _ = check_named("CB-05", Backend::WasmSimd128);
}

/// `CB-12`: the wasm SIMD128 SWAR broadcast sequence equals the portable
/// reference lane for lane, and selection offers it nowhere.
///
/// The sequence packs three `B` elements at 21-bit spacing in each 64-bit lane
/// and multiplies by one broadcast scalar --- an ordinary integer identity, not
/// an exactness argument this library is needed for. What this test pins is
/// both halves of the measured outcome: the packing and its bias correction
/// still compute *that* integer where the sequence can run, and no family list
/// carries it --- `CG-17` measured it at 0.32--0.38x of the dot sequence it
/// would displace, so an entry that reappears in a list is a shipped
/// regression, and this assert is what says so.
#[test]
fn wasm_swar_broadcast_equals_portable_cb_12() {
    let spec = uor_matmul_kernels::isa::wasm::SIMD128_SWAR_I8_I32;
    let offered = available_i8().any(|s| {
        s.backend == Backend::WasmSimd128
            && s.mr == spec.mr
            && s.nr == spec.nr
            && s.k_group == spec.k_group
    });
    assert!(
        !offered,
        "CG-17 measured the SWAR sequence slower than the dot sequence; selection must not offer it"
    );
    if !cfg!(all(target_arch = "wasm32", target_feature = "simd128")) {
        eprintln!(
            "CB-12: the SWAR sequence runs only on baseline wasm SIMD128; the wasmtime cross-run covers the parity half"
        );
        return;
    }
    // Every packed depth, lane for lane: the spacing, the bias correction, and
    // the chunk boundary all show up here if they are wrong. Depths are a
    // whole number of `k`-groups, which is what the driver always packs.
    let mut ks: Vec<usize> = depths()
        .iter()
        .map(|&kc| kc.div_ceil(spec.k_group) * spec.k_group)
        .collect();
    ks.dedup();
    for kc in ks {
        let pa = fill(spec.mr * kc, kc as u64 ^ 0x5B, |v| v as i8);
        let pb = fill(spec.nr * kc, kc as u64 ^ 0xA7, |v| v as i8);
        let mut acc = vec![0i32; spec.mr * spec.nr];
        spec.mac_tile(kc, &pa, &pb, &mut acc);
        assert_eq!(
            acc,
            reference_i8(&spec, kc, &pa, &pb),
            "SWAR broadcast disagrees at kc={kc}"
        );
    }
    // The alphabet's extremes, shallow and deep: the biased fields must absorb
    // `-128` and `127` alike without a carry crossing a field boundary, at a
    // depth well past the sequence's internal chunk.
    for kc in [32usize, 160] {
        for (av, bv) in [(i8::MIN, i8::MIN), (i8::MIN, i8::MAX), (i8::MAX, i8::MAX)] {
            let pa = vec![av; spec.mr * kc];
            let pb = vec![bv; spec.nr * kc];
            let mut acc = vec![0i32; spec.mr * spec.nr];
            spec.mac_tile(kc, &pa, &pb, &mut acc);
            let want = (kc as i32) * i32::from(av) * i32::from(bv);
            assert!(
                acc.iter().all(|&x| x == want),
                "SWAR broadcast at ({av}, {bv}) kc={kc}: {:?} want {want}",
                &acc[..4.min(acc.len())]
            );
        }
    }
    // The tile contract is overwrite, not accumulate: the driver reuses the
    // tile buffer without zeroing it, so a sequence that adds into `acc`
    // double-counts on the second tile. A garbage-filled tile says which.
    let kc = 96usize;
    let pa = fill(spec.mr * kc, 0x33, |v| v as i8);
    let pb = fill(spec.nr * kc, 0x44, |v| v as i8);
    let mut acc = vec![0x5A5A_5A5Ai32; spec.mr * spec.nr];
    spec.mac_tile(kc, &pa, &pb, &mut acc);
    assert_eq!(
        acc,
        reference_i8(&spec, kc, &pa, &pb),
        "the SWAR tile must be written, not accumulated into"
    );
    // The W4A8 bound, where the biased product is smallest: the sequence does
    // not get a narrower declaration for it, so the fill is the check.
    for kc in [33usize, 96] {
        let pa = fill(spec.mr * kc, kc as u64 ^ 0x11, |v| (v % 15 - 7) as i8);
        let pb = fill(spec.nr * kc, kc as u64 ^ 0x22, |v| (v % 15 - 7) as i8);
        let mut acc = vec![0i32; spec.mr * spec.nr];
        spec.mac_tile(kc, &pa, &pb, &mut acc);
        assert_eq!(
            acc,
            reference_i8(&spec, kc, &pa, &pb),
            "SWAR broadcast at the W4A8 bound disagrees at kc={kc}"
        );
    }
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
    let specs: Vec<_> = available_i16().collect();
    let mut vecs = TileVecs::for_specs(&specs, depths());
    parity::tile_parity(
        "CB-02 i16",
        specs.iter().copied(),
        depths(),
        (7, 8),
        (|v| (v * 251) as i16, |v| (v * 251) as i16),
        parity::dot_i16,
        &mut vecs.scratch(),
    );

    // Values bounded by 2^20, so the *reference* sum fits an `i64` at every
    // depth below. The kernel's own lane is bounded the same way, and the
    // driver only ever offers it a chunk `lane_depth` admits --- testing it
    // past that would be testing something the driver never asks for.
    let specs: Vec<_> = available_i32_exact().collect();
    let mut vecs = TileVecs::for_specs(&specs, depths());
    parity::tile_parity(
        "CB-02 i32",
        specs.iter().copied(),
        depths(),
        (9, 10),
        (
            |v| ((v & 0xFFFF) - 0x8000) as i32,
            |v| ((v & 0xFFFF) - 0x8000) as i32,
        ),
        parity::dot_i32_exact,
        &mut vecs.scratch(),
    );

    // The modular lanes wrap, and that *is* the answer in the quotient: the
    // references below wrap too, deliberately.
    let specs: Vec<_> = available_i32_modular().collect();
    let mut vecs = TileVecs::for_specs(&specs, depths());
    parity::tile_parity(
        "CB-02 i32 modular",
        specs.iter().copied(),
        depths(),
        (11, 12),
        (
            |v| (v.wrapping_mul(99_991)) as i32,
            |v| (v.wrapping_mul(65_537)) as i32,
        ),
        parity::dot_i32_mod,
        &mut vecs.scratch(),
    );

    let specs: Vec<_> = available_i64_modular().collect();
    let mut vecs = TileVecs::for_specs(&specs, depths());
    parity::tile_parity(
        "CB-02 i64 modular",
        specs.iter().copied(),
        depths(),
        (13, 14),
        (
            |v| v.wrapping_mul(0x9E37_79B9_7F4A_7C15u64 as i64),
            |v| v.wrapping_mul(0x1000_0000_01B3),
        ),
        parity::dot_i64_mod,
        &mut vecs.scratch(),
    );
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

/// `CG-13`: the cached availability list *is* the full walk's list, in every
/// family, and a selection from it returns the same sequence at every bound
/// and panel height.
///
/// The driver selects from `uor_matmul_kernels::cached`, which reads each
/// family's feature predicates once per process and reuses the bits. The claim
/// is that this changes *when* the predicates are read and never *what*
/// selection returns --- so this test compares the two walks directly, and the
/// resolution counter is what makes "the cache is consulted" a number rather
/// than an intention (the same shape of instrument as `dot_instrumented`).
#[test]
fn the_cached_walk_is_the_full_walk_cg_13() {
    use uor_matmul_kernels::cached;

    /// The identity of a spec, as a comparable value. `KernelSpec` carries no
    /// `PartialEq` --- nothing in the library compares one --- so the test
    /// names the fields it means, the entry point included.
    fn key<E, L>(
        s: &KernelSpec<E, L>,
    ) -> (
        Backend,
        Factorization,
        usize,
        usize,
        usize,
        usize,
        u128,
        u128,
        usize,
    ) {
        (
            s.backend,
            s.factorization,
            s.mr,
            s.nr,
            s.k_group,
            s.products_per_step,
            s.lane_cap,
            s.max_bound,
            s.mac_tile as usize,
        )
    }

    fn same<E, L, I, J>(family: &str, walk: impl Fn() -> I, cached_walk: impl Fn() -> J)
    where
        I: Iterator<Item = KernelSpec<E, L>>,
        J: Iterator<Item = KernelSpec<E, L>>,
    {
        let full: Vec<_> = walk().collect();
        let hit = cached::resolutions();
        let cached_list: Vec<_> = cached_walk().collect();
        assert_eq!(
            cached::resolutions() - hit,
            1,
            "{family}: the first cached touch resolves its predicates exactly once"
        );
        assert_eq!(
            full.iter().map(key).collect::<Vec<_>>(),
            cached_list.iter().map(key).collect::<Vec<_>>(),
            "{family}: the cached list differs from the full walk"
        );
        // What the driver does with the list: choose, at every declared bound
        // and panel height, for every backend a caller can name.
        for bound in [1, 127, 32767, 32768, 1 << 31, u128::MAX] {
            for rows in [1, 4, 8, 64] {
                for backend in Backend::ALL.into_iter().chain([Backend::Auto]) {
                    let a = choose_for_rows(walk(), backend, bound, rows);
                    let b = choose_for_rows(cached_walk(), backend, bound, rows);
                    assert_eq!(
                        a.as_ref().map(key),
                        b.as_ref().map(key),
                        "{family}: cached selection differs at bound {bound}, rows {rows}, \
                         {backend:?}"
                    );
                }
            }
        }
        // The loop above is hundreds of selections against one family. The
        // count is the assertion that the cache is consulted, read the way
        // `CU-02` reads `narrow_tiles` --- a number, not an intention.
        assert_eq!(
            cached::resolutions() - hit,
            1,
            "{family}: a repeat selection resolves nothing"
        );
    }

    let cold = cached::resolutions();
    let mut families = 0usize;
    macro_rules! family {
        ($name:literal, $walk:expr, $cached:expr) => {{
            same($name, $walk, $cached);
            families += 1;
        }};
    }
    family!("i8 tile", available_i8, cached::available_i8);
    family!(
        "i8 narrow",
        available_i8_narrow,
        cached::available_i8_narrow
    );
    family!("i16 tile", available_i16, cached::available_i16);
    family!(
        "i16 modular",
        available_i16_modular,
        cached::available_i16_modular
    );
    family!(
        "i32 exact",
        available_i32_exact,
        cached::available_i32_exact
    );
    family!(
        "i32 modular",
        available_i32_modular,
        cached::available_i32_modular
    );
    family!(
        "i64 exact",
        available_i64_exact,
        cached::available_i64_exact
    );
    family!(
        "i64 modular",
        available_i64_modular,
        cached::available_i64_modular
    );
    family!(
        "i8 reduce",
        available_reduce_i8,
        cached::available_reduce_i8
    );
    family!(
        "i16 reduce",
        available_reduce_i16,
        cached::available_reduce_i16
    );
    family!(
        "i16 modular reduce",
        available_reduce_i16_modular,
        cached::available_reduce_i16_modular
    );
    family!(
        "i32 exact reduce",
        available_reduce_i32_exact,
        cached::available_reduce_i32_exact
    );
    family!(
        "i32 modular reduce",
        available_reduce_i32_modular,
        cached::available_reduce_i32_modular
    );
    family!(
        "i64 exact reduce",
        available_reduce_i64_exact,
        cached::available_reduce_i64_exact
    );
    family!(
        "i64 modular reduce",
        available_reduce_i64_modular,
        cached::available_reduce_i64_modular
    );
    assert_eq!(
        cached::resolutions() - cold,
        families,
        "every family resolved once and no selection re-resolved"
    );
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
    for spec in available_i16() {
        let (lo, hi) = parity::extremes(spec.max_bound, 32768);
        let values: Vec<(i16, i16)> = [(lo, lo), (lo, hi), (hi, hi), (hi, lo)]
            .iter()
            .map(|&(a, b)| (a as i16, b as i16))
            .collect();
        let mut vecs = TileVecs::for_specs(&[spec], &[64]);
        parity::tile_extremes(
            "CB-07 i16",
            core::iter::once(spec),
            &[2, 4, 8, 16, 64],
            &values,
            |kc, a, b| (kc as i64) * i64::from(a) * i64::from(b),
            &mut vecs.scratch(),
        );
    }

    for spec in available_i32_exact() {
        let (lo, hi) = parity::extremes(spec.max_bound, 1 << 31);
        let values: Vec<(i32, i32)> = [(lo, lo), (lo, hi), (hi, hi)]
            .iter()
            .map(|&(a, b)| (a as i32, b as i32))
            .collect();
        // One product at a time: `lane_depth` at the full bound is 1, and the
        // driver never offers this lane more than that.
        let mut vecs = TileVecs::for_specs(&[spec], &[1]);
        parity::tile_extremes(
            "CB-07 i32",
            core::iter::once(spec),
            &[1],
            &values,
            |kc, a, b| (kc as i64) * i64::from(a) * i64::from(b),
            &mut vecs.scratch(),
        );
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
        available_reduce_i8,
    };

    let specs: Vec<_> = available_reduce_i8().collect();
    let mut vecs = ReduceVecs::for_specs(&specs, depths());
    let seen = parity::reduce_parity(
        "CB-06 i8",
        specs.iter().copied(),
        depths(),
        (21, 22),
        (|v| v as i8, |v| v as i8),
        parity::dot_i8,
        &mut vecs.scratch(),
    );
    assert!(seen > 0);
    // Native only, as `CB-02`'s counterpart is: Miri models no vector intrinsics,
    // so the reference is the whole list there by design.
    if cfg!(any(target_arch = "x86_64", target_arch = "aarch64")) && !cfg!(miri) {
        assert!(
            specs.len() > 2,
            "x86-64 and aarch64 both have a vector i8 reduce sequence past the reference"
        );
    }

    // The extremes too: the reduce sequences accumulate a whole row into one
    // lane, so their intermediate widths are a different question from the tile
    // sequences' and deserve the same input.
    let mut vecs = ReduceVecs::for_specs(&specs, &[1024]);
    parity::reduce_extremes(
        "CB-06 i8",
        specs.iter().copied(),
        &[16, 64, 1024],
        &[(i8::MIN, i8::MIN)],
        |kc, a, b| (kc as i32) * i32::from(a) * i32::from(b),
        &mut vecs.scratch(),
    );

    let specs: Vec<_> = available_reduce_i16().collect();
    let mut vecs = ReduceVecs::for_specs(&specs, depths());
    parity::reduce_parity(
        "CB-06 i16",
        specs.iter().copied(),
        depths(),
        (23, 24),
        (|v| v as i16, |v| v as i16),
        parity::dot_i16,
        &mut vecs.scratch(),
    );
    // At the full alphabet, where a paired-product sequence would be wrong.
    let mut vecs = ReduceVecs::for_specs(&specs, &[16]);
    parity::reduce_extremes(
        "CB-06 i16",
        specs.iter().copied(),
        &[16],
        &[(i16::MIN, i16::MIN), (i16::MIN, i16::MAX)],
        |kc, a, b| (kc as i64) * i64::from(a) * i64::from(b),
        &mut vecs.scratch(),
    );

    let specs: Vec<_> = available_reduce_i32_exact().collect();
    let mut vecs = ReduceVecs::for_specs(&specs, depths());
    // Bounded so the reference sum fits an `i64` at every depth here.
    parity::reduce_parity(
        "CB-06 i32",
        specs.iter().copied(),
        depths(),
        (25, 26),
        (
            |v| ((v & 0xFFFF) - 0x8000) as i32,
            |v| ((v & 0xFFFF) - 0x8000) as i32,
        ),
        parity::dot_i32_exact,
        &mut vecs.scratch(),
    );

    let specs: Vec<_> = available_reduce_i32_modular().collect();
    let mut vecs = ReduceVecs::for_specs(&specs, depths());
    parity::reduce_parity(
        "CB-06 i32 modular",
        specs.iter().copied(),
        depths(),
        (27, 28),
        (
            |v| v.wrapping_mul(99_991) as i32,
            |v| v.wrapping_mul(65_537) as i32,
        ),
        parity::dot_i32_mod,
        &mut vecs.scratch(),
    );

    let specs: Vec<_> = available_reduce_i16_modular().collect();
    let mut vecs = ReduceVecs::for_specs(&specs, depths());
    parity::reduce_parity(
        "CB-06 i16 modular",
        specs.iter().copied(),
        depths(),
        (29, 30),
        (|v| v as i16, |v| v as i16),
        parity::dot_i16_mod,
        &mut vecs.scratch(),
    );

    // Bounded by `2^31`, so the *reference* sum fits an `i128` at every depth
    // below. The kernel's lane is bounded the same way: at the full `i64`
    // bound one product needs 126 bits and `lane_depth` is 1, so the driver
    // never offers this lane a deeper chunk, and testing past that would be
    // testing something no caller can reach.
    let specs: Vec<_> = available_reduce_i64_exact().collect();
    let mut vecs = ReduceVecs::for_specs(&specs, depths());
    parity::reduce_parity(
        "CB-06 i64",
        specs.iter().copied(),
        depths(),
        (31, 32),
        (
            |v| (v & 0xFFFF_FFFF) - 0x8000_0000,
            |v| (v & 0xFFFF_FFFF) - 0x8000_0000,
        ),
        parity::dot_i64_exact,
        &mut vecs.scratch(),
    );

    let specs: Vec<_> = available_reduce_i64_modular().collect();
    let mut vecs = ReduceVecs::for_specs(&specs, depths());
    parity::reduce_parity(
        "CB-06 i64 modular",
        specs.iter().copied(),
        depths(),
        (33, 34),
        (
            |v| v.wrapping_mul(0x1000_0000_01B3),
            |v| v.wrapping_mul(0x9E37_79B9),
        ),
        parity::dot_i64_mod,
        &mut vecs.scratch(),
    );
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

/// `CB-08`: every table sequence equals the reference, lane for lane.
///
/// Both halves: the build, which is the only place the tabulated traversal
/// multiplies, and the gather, which is the only place it adds. A sequence that
/// disagreed with the reference on either would produce a different sum, and
/// this is the test that says so before `CD-13` sees it end to end.
#[test]
fn every_table_sequence_equals_the_reference_cb_08() {
    let corpus = table_32();
    let mut vecs = TableVecs::for_corpus(&corpus, 0);
    let compared = parity::table_parity(
        "CB-08",
        uor_matmul_kernels::available_table_i8,
        &corpus,
        &parity::ops_i8(),
        (0xb00c, 0xac75),
        |v| ((v % 255) - 127) as i8,
        parity::stack_value32,
        (7, -3),
        128,
        true,
        &mut vecs.scratch(),
    );

    // `dispatch_slab!`'s wildcard, which is what keeps the enumeration from being
    // a ceiling (R8). Sixteen exponents cover every code space a `u16` can name,
    // and no shipped codec exceeds them --- but `slab` is a `u32` the caller
    // passes, so a slab of `2^17` lane words at one row asks for a code count the
    // list does not have, and that arm binds the constant to zero and reads the
    // slab from its argument instead. Untested, that arm was the one line standing
    // between the enumeration and a limit.
    //
    // Host-only: `2^20` lane words is 12 MB of stack, past anything the
    // executor's static scratch will hold. The wildcard arm is the same macro
    // body as the enumerated ones, so the reduced corpus covers the code on the
    // device; this sweep covers the *sizes*.
    let mut compared = compared;
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
        for spec in uor_matmul_kernels::available_table_i8(rows, group) {
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
    let corpus = table_64();
    let mut vecs = TableVecs::for_corpus(&corpus, 0);
    let compared = parity::table_parity(
        "CB-08 i16",
        uor_matmul_kernels::available_table_i16,
        &corpus,
        &parity::ops_i16(),
        (0x16b0, 0x16ac),
        |v| ((v % 65535) - 32767) as i16,
        parity::stack_value64,
        (11, -5),
        0,
        false,
        &mut vecs.scratch(),
    );
    assert!(compared > 0, "CB-08 compared nothing for the 64-bit lane");
}

/// `CB-09`: every modular table sequence equals the portable modular
/// reference, lane for lane.
///
/// For `i64` the reference is the *only* sequence at every height --- the
/// build's multiply is the table's only one, and no SIMD integer multiply
/// reaches an `i64` lane, which is why the dense family is portable-only too.
#[test]
fn every_modular_table_sequence_equals_the_reference_cb_09() {
    let corpus = table_32();
    let mut vecs = TableVecs::for_corpus(&corpus, uor_matmul_kernels::Mod32(0));
    let mut compared = parity::table_parity(
        "CB-09 mod32",
        uor_matmul_kernels::available_table_i32_modular,
        &corpus,
        &parity::ops_mod32(),
        (0xb32c, 0xa32c),
        |v| v.wrapping_mul(0x9E37_79B9) as i32,
        parity::stack_value32,
        (7, -3),
        0,
        true,
        &mut vecs.scratch(),
    );

    let corpus = table_64();
    let mut vecs = TableVecs::for_corpus(&corpus, uor_matmul_kernels::Mod64(0));
    compared += parity::table_parity(
        "CB-09 mod64",
        uor_matmul_kernels::available_table_i64_modular,
        &corpus,
        &parity::ops_mod64(),
        (0xb64c, 0xa64c),
        |v| v.wrapping_mul(0x9E37_79B9_7F4A_7C15u64 as i64),
        parity::stack_value64,
        (11, -5),
        0,
        false,
        &mut vecs.scratch(),
    );

    assert!(
        compared > 0,
        "CB-09 compared nothing; on a host with no modular table sequence beyond the \
         reference this gate would pass vacuously"
    );
}

// ---------------------------------------------------------------------------
// CB-10: the bound-1 table build
// ---------------------------------------------------------------------------

/// `CB-10`: at bound 1 the table build issues no multiply, and selection
/// offers the adds-only build exactly when the declared bound admits it.
///
/// At bound 1 every product is `+-a` or `0`, so a build can fill the same slot
/// the reference fills with adds and subtracts alone. The code spaces are the
/// sign codec's --- `Packed<Grid<2>, Bk>` enumerates `2^Bk` words, and 256 is
/// also the ternary spelling's --- and the fill includes zero, so both
/// spellings `CK-13` declared are exercised.
#[test]
fn every_bound1_table_build_equals_the_reference_cb_10() {
    let corpus = bound1();
    let mut vecs = TableVecs::for_bound1(&corpus, 0);
    let compared = parity::table_bound1(
        "CB-10",
        uor_matmul_kernels::available_table_i8,
        &corpus,
        &parity::ops_i8(),
        (0xb10c, 0xb10a),
        |v| ((v % 3) - 1) as i8,
        &mut vecs.scratch(),
    );
    assert!(
        compared > 0,
        "CB-10 compared nothing; the portable bound-1 build is present on every host"
    );

    // The gathers are bound-independent, so the bound-1 spec carries its
    // backend's own --- shared, not duplicated (R13). Under Miri every
    // fn-pointer creation is a fresh allocation, so address equality cannot
    // witness sharing there; the claim is asserted on real targets.
    if !cfg!(miri) {
        for rows in [1usize, 16] {
            for group in [1usize, 16] {
                let specs: Vec<_> = uor_matmul_kernels::available_table_i8(rows, group).collect();
                let reference = specs[0];
                for spec in &specs[1..] {
                    if spec.max_bound > 1 {
                        continue;
                    }
                    let donor = specs[1..]
                        .iter()
                        .find(|s| s.backend == spec.backend && s.max_bound > 1)
                        .unwrap_or(&reference);
                    assert!(
                        core::ptr::fn_addr_eq(spec.gather, donor.gather),
                        "{:?} bound-1 spec duplicates the gather instead of sharing it",
                        spec.backend
                    );
                    assert!(
                        core::ptr::fn_addr_eq(spec.gather_codes, donor.gather_codes),
                        "{:?} bound-1 spec duplicates gather_codes instead of sharing it",
                        spec.backend
                    );
                }
            }
        }
    }
}

/// The Gray-walk sign build against the per-codeword build, slot for slot.
///
/// No new conformance ID: this is a construction change on a path `CB-10` and
/// `CU-06` already gate --- the table's bytes and the zero multiply count.
/// What those gates do not say is *which* construction produced the table,
/// and that is what this test pins: every code, at the alphabet's extremes,
/// on activation tiles chosen so a wrong sign on any one bit moves at least
/// one entry, at every tile height the driver walks. The census charge is a
/// formula over the spec's own declaration, and it is asserted here against
/// the same terms the walk executes.
#[test]
fn the_gray_walk_builds_the_sign_table_slot_for_slot() {
    use uor_matmul_kernels::{gray_sign_table, portable_table_bound1};

    // The sign codebook, built the way the codec decodes it.
    let sign_book = |space: usize, block: usize| -> Vec<i8> {
        (0..space * block)
            .map(|i| {
                let (c, t) = (i / block, i % block);
                if (c >> t) & 1 == 1 {
                    1
                } else {
                    -1
                }
            })
            .collect()
    };

    // (space, block, rows, tiles): `Sign<4>` and `Sign<8>` at every tile
    // height, `Sign<16>` --- the `u16` code type's whole space --- at the two
    // ends of it. The tiles hold the alphabet's extremes: a wrong bit flip or
    // a store at the walk ordinal is visible in at least one entry. Under
    // Miri the largest space is 4096: the claim the big case carries is
    // slot-for-slot equality of the two builds, which is structural in the
    // space, and a 65536-entry table is a corpus Miri cannot finish inside
    // the CI budget (the workspace tests' `HUGE_K` is the same discipline).
    let (s4, s8, s16) = if cfg!(miri) {
        (16, 256, 4096)
    } else {
        (16, 256, 65536)
    };
    let cases: Vec<(usize, usize, Vec<usize>)> = if cfg!(miri) {
        vec![
            (s4, 4, vec![1, 16]),
            (s8, 8, vec![1, 16]),
            (s16, 16, vec![1]),
        ]
    } else {
        vec![
            (s4, 4, vec![1, 2, 4, 8, 16]),
            (s8, 8, vec![1, 2, 4, 8, 16]),
            (s16, 16, vec![1, 16]),
        ]
    };
    for (space, block, row_list) in cases {
        let book = sign_book(space, block);
        for rows in row_list {
            let tiles: Vec<Vec<i8>> = vec![
                vec![i8::MIN; block * rows],
                vec![i8::MAX; block * rows],
                (0..block * rows)
                    .map(|i| if i % 2 == 0 { i8::MIN } else { i8::MAX })
                    .collect(),
                (0..block * rows).map(|i| ((i * 37) % 255) as i8).collect(),
            ];
            let independent = portable_table_bound1(rows, 1);
            let gray = gray_sign_table(rows, 1);
            for acts in &tiles {
                let mut want = vec![0i32; space * rows];
                let mut got = vec![0i32; space * rows];
                independent.build(space, block, &book, acts, &mut want);
                gray.build(space, block, &book, acts, &mut got);
                assert_eq!(
                    got, want,
                    "space {space}, block {block}, rows {rows}: the Gray walk disagrees"
                );
                // And the reference itself is checked against the model, so the
                // comparison above is not two builds sharing a mistake.
                for &c in &[0usize, space / 2, space - 1] {
                    for i in 0..rows {
                        let mut acc = 0i32;
                        for t in 0..block {
                            let a = i32::from(acts[t * rows + i]);
                            acc += if (c >> t) & 1 == 1 { a } else { -a };
                        }
                        assert_eq!(
                            want[c * rows + i],
                            acc,
                            "the model at code {c}, row {i} (space {space}, rows {rows})"
                        );
                    }
                }
            }
        }
    }

    // The census charge, read off the spec's own declaration: `T[0]` and the
    // doubled activations, then one update per code past the first.
    let spec = gray_sign_table(16, 1);
    assert_eq!((spec.build_adds)(256, 8, 16), (2 * 8 + 256 - 1) as u64 * 16);
    assert!(!spec.build_multiplies, "the walk is adds and subtracts");
}
