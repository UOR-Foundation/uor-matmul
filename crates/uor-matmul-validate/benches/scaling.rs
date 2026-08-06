//! The scaling harness (§13, R12, C3).
//!
//! Scaling is a V&V axis, not a benchmark. Every claim here is a **fitted
//! exponent with a confidence interval**, measured for this library and for
//! each oracle over the *same* sweep, and reported together with its residuals.
//! A throughput table with no asymptotic content would say nothing about
//! whether the two implementations have the same shape, which is the only
//! comparison that survives a change of machine.
//!
//! Every `CG-*` claim is `open`: measured and reported, never asserted (R4).
//! Nothing in this file fails a gate on a *performance* number. What it does
//! assert, inside the timed harness, is byte-equality --- because a speed
//! measured on the wrong answer is not a measurement of anything (`CG-06`).

// `criterion_group!` generates a function this crate cannot document, and the
// workspace denies `missing_docs`. The exemption is scoped to this file, which
// is a bench harness rather than a shipped crate.
#![allow(missing_docs)]

use criterion::{criterion_group, criterion_main, Criterion};
use std::mem::size_of;
use uor_matmul::driver::modular_level_needs;
use uor_matmul::prelude::*;
use uor_matmul::{
    suggested_collapse_index, suggested_collapse_rows, suggested_float_panels, suggested_scratch,
    suggested_tabulation, workspace_report, Backend, Collapse, Shape, TabulatedTriple,
};
use uor_matmul_core::{
    as_alphabet_tropical, Alphabet, EncodeMode, Full, PackedCode, Traversal, Trop,
};
use uor_matmul_gemm::{gemm as gemm_dense, gemm_selected, SelectedTriple, Witness};
use uor_matmul_validate::corpus::{Case, Corpus, Mask};
use uor_matmul_validate::scaling::{self, Labelled, Sweep};

/// Emit the fitted exponents, then time a few shapes so `cargo bench` has
/// something to report alongside them.
fn scaling_report(c: &mut Criterion) {
    let sweep = Sweep::standard();

    // `CG-01`: the arithmetic scaling exponent. The oracle columns live in the
    // test harness, which has the `ref-*` features; this emits our side so that
    // `cargo bench` and `cargo test` report the same fit for the same sweep.
    let ours = Labelled {
        id: "uor-matmul",
        name: "uor-matmul i8",
        fit: scaling::fit_ours(&sweep),
    };
    scaling::Report::new(sweep.clone(), ours, Vec::new()).emit();

    // A single criterion benchmark so `cargo bench` has something to time; the
    // asymptotic content is in the report above, which is what R12 asks for.
    let mut group = c.benchmark_group("gemm");
    for &n in &[64usize, 128, 256] {
        group.bench_function(format!("i8_i32_{n}cubed"), |b| {
            let a = vec![1i8; n * n];
            let w = vec![1i8; n * n];
            let mut out = vec![0i32; n * n];
            b.iter(|| scaling::run_case(n, n, n, &a, &w, &mut out));
        });
    }
    group.finish();
}

/// `cargo bench` as the quick answer to "are we faster?": this library and
/// every enabled oracle timed in the one run, at shapes chosen to separate the
/// questions --- one latency-bound, one arithmetic-bound, one non-square. The
/// sweeps (`oracle_sweep`, `just scaling`) remain the measurement of record;
/// this group is the thirty-second version, and its figures are every bit as
/// `open`. Criterion's own report gives the ratio within a shape.
///
/// What is asserted, inside each timed closure, is that the answer is right
/// (`CG-06`). The operands are a constant fill, which every implementation here
/// computes exactly --- and which flatters the float path, whose limb window
/// never flushes when every product shares one exponent. The recorded-random
/// fill lives in `oracle_sweep`, whose header explains what the flattery hid.
///
/// The `handwritten/` row is the comparison against *no* library: the triple
/// loop a caller writes without one. The `i32` half is §3.4's wrapping oracle,
/// which is deliberately the dumbest possible implementation --- one function,
/// not two, or the baseline and the oracle could drift apart.
fn vs_oracles(c: &mut Criterion) {
    const SHAPES: [(usize, usize, usize); 3] = [(16, 16, 16), (128, 128, 128), (64, 512, 1024)];

    let mut i32_group = c.benchmark_group("gemm_i32");
    for &(m, k, n) in &SHAPES {
        let shape = format!("{m}x{k}x{n}");
        let a = vec![1i32; m * k];
        let b = vec![1i32; k * n];

        i32_group.bench_function(format!("uor-matmul/{shape}"), |bench| {
            let mut out = vec![0i32; m * n];
            let mut scratch =
                vec![Alphabet::<i32, Full<i32>>::ZERO; suggested_scratch(Shape { m, k, n })];
            bench.iter(|| {
                let av = MatView::row_major(as_alphabet_full(&a), m, k).expect("A fits");
                let bv = MatView::row_major(as_alphabet_full(&b), k, n).expect("B fits");
                let cv = MatViewMut::row_major(&mut out, m, n).expect("C fits");
                let mut t = Triple::new(av, bv, cv).expect("the product exists");
                // The kernelized driver directly. The documented safe entry
                // reaches the same selection through `gemm_auto`, and the
                // `public_api` group below times the two side by side so the
                // claim "the public route is this route" is measured, not
                // asserted in a comment.
                uor_matmul::gemm_packed(
                    &mut t,
                    &Linear::OVERWRITE,
                    GemmOptions {
                        encode: EncodeMode::Wrapping,
                        ..Default::default()
                    },
                    &mut Scratch::new(&mut scratch),
                );
                assert!(
                    out.iter().all(|&v| v == k as i32),
                    "the timed call must be correct"
                );
            });
        });

        i32_group.bench_function(format!("handwritten/{shape}"), |bench| {
            bench.iter(|| {
                let out = uor_matmul_validate::reference_wrapping_i32(m, k, n, &a, &b);
                assert!(
                    out.iter().all(|&v| v == k as i32),
                    "the timed call must be correct"
                );
            });
        });

        #[cfg(feature = "ref-ndarray")]
        i32_group.bench_function(format!("ndarray/{shape}"), |bench| {
            use uor_matmul_validate::oracle::{NdArray, Oracle};
            bench.iter(|| {
                let out = NdArray::product_i32(m, k, n, &a, &b);
                assert!(
                    out.iter().all(|&v| v == k as i32),
                    "the timed call must be correct"
                );
            });
        });

        #[cfg(feature = "ref-nalgebra")]
        i32_group.bench_function(format!("nalgebra/{shape}"), |bench| {
            use uor_matmul_validate::oracle::{Nalgebra, Oracle};
            bench.iter(|| {
                let out = Nalgebra::product_i32(m, k, n, &a, &b);
                assert!(
                    out.iter().all(|&v| v == k as i32),
                    "the timed call must be correct"
                );
            });
        });
    }
    i32_group.finish();

    let mut f32_group = c.benchmark_group("gemm_f32");
    for &(m, k, n) in &SHAPES {
        let shape = format!("{m}x{k}x{n}");
        let a = vec![1.0f32; m * k];
        let b = vec![1.0f32; k * n];

        f32_group.bench_function(format!("uor-matmul/{shape}"), |bench| {
            let mut out = vec![0.0f32; m * n];
            // The packed float path, offered what `suggested_float_panels`
            // names for the shape, because that is what a caller who follows
            // the suggestion holds: the offer admits the float placement
            // bridge where the shape does, so this times the kernel-table
            // lane a real caller gets, not the decline a hand-sized offer
            // would price. The generic driver would time a factorization no
            // caller gets, and it reads several times slower.
            let (pa_len, pb_len) = suggested_float_panels(Shape { m, k, n });
            let mut pa = vec![PackedCode::default(); pa_len];
            let mut pb = vec![PackedCode::default(); pb_len];
            bench.iter(|| {
                let av = MatView::row_major(&a, m, k).expect("A fits");
                let bv = MatView::row_major(&b, k, n).expect("B fits");
                let cv = MatViewMut::row_major(&mut out, m, n).expect("C fits");
                let mut t = Triple::new(av, bv, cv).expect("the product exists");
                uor_matmul::gemm_float_packed(
                    &mut t,
                    &Linear::OVERWRITE,
                    GemmOptions::default(),
                    &mut pa,
                    &mut pb,
                );
                assert!(
                    out.iter().all(|&v| v == k as f32),
                    "the timed call must be correct"
                );
            });
        });

        f32_group.bench_function(format!("handwritten/{shape}"), |bench| {
            let mut out = vec![0.0f32; m * n];
            bench.iter(|| {
                handwritten_f32(m, k, n, &a, &b, &mut out);
                assert!(
                    out.iter().all(|&v| v == k as f32),
                    "the timed call must be correct"
                );
            });
        });

        #[cfg(feature = "ref-matrixmultiply")]
        f32_group.bench_function(format!("matrixmultiply/{shape}"), |bench| {
            use uor_matmul_validate::oracle::{FloatOracle, MatrixMultiply};
            bench.iter(|| {
                let out = MatrixMultiply::product_f32(m, k, n, &a, &b);
                assert!(
                    out.iter().all(|&v| v == k as f32),
                    "the timed call must be correct"
                );
            });
        });

        #[cfg(feature = "ref-faer")]
        f32_group.bench_function(format!("faer/{shape}"), |bench| {
            use uor_matmul_validate::oracle::{Faer, FloatOracle};
            bench.iter(|| {
                let out = Faer::product_f32(m, k, n, &a, &b);
                assert!(
                    out.iter().all(|&v| v == k as f32),
                    "the timed call must be correct"
                );
            });
        });
    }
    f32_group.finish();
}

/// The `f32` half of "no library": three loops and an `f32` accumulator.
///
/// Iterator-shaped rather than index-shaped so the workspace's clippy denies
/// stay quiet; the arithmetic is the same naive sum, in the same order, with
/// the same order-dependent rounding a classical caller gets.
fn handwritten_f32(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(out.len(), m * n, "the output is the product's shape");
    for (i, row) in out.chunks_exact_mut(n).enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = (0..k).map(|p| a[i * k + p] * b[p * n + j]).sum();
        }
    }
}

/// The public route against the explicit one: `slice::gemm`, which is the
/// entry the README shows, against `gemm_packed` at the same shapes and the
/// same suggested offer. The two must time the same, because the facade's
/// `gemm` *is* `gemm_auto` --- `gemm_packed` with the selection already
/// inside --- and this group is the measurement that keeps that sentence
/// honest (`CD-22` asserts the route; this prices it). Byte-equality is
/// asserted inside each timed closure, as everywhere in this file.
fn public_api(c: &mut Criterion) {
    const SHAPES: [(usize, usize, usize); 3] = [(16, 16, 16), (64, 64, 64), (128, 128, 128)];

    let mut group = c.benchmark_group("public_api");
    for &(m, k, n) in &SHAPES {
        let shape = format!("{m}x{k}x{n}");
        let a = vec![1i8; m * k];
        let b = vec![1i8; k * n];

        group.bench_function(format!("slice::gemm/{shape}"), |bench| {
            let mut out = vec![0i32; m * n];
            let mut scratch = vec![0i8; suggested_scratch(Shape { m, k, n })];
            bench.iter(|| {
                uor_matmul::slice::gemm(m, k, n, &a, &b, &mut out, &mut scratch)
                    .expect("the product exists");
                assert!(
                    out.iter().all(|&v| v == k as i32),
                    "the timed call must be correct"
                );
            });
        });

        group.bench_function(format!("gemm_packed/{shape}"), |bench| {
            let mut out = vec![0i32; m * n];
            let mut scratch =
                vec![Alphabet::<i8, Full<i8>>::ZERO; suggested_scratch(Shape { m, k, n })];
            bench.iter(|| {
                let av = MatView::row_major(as_alphabet_full(&a), m, k).expect("A fits");
                let bv = MatView::row_major(as_alphabet_full(&b), k, n).expect("B fits");
                let cv = MatViewMut::row_major(&mut out, m, n).expect("C fits");
                let mut t = Triple::new(av, bv, cv).expect("the product exists");
                uor_matmul::gemm_packed(
                    &mut t,
                    &Linear::OVERWRITE,
                    GemmOptions::default(),
                    &mut Scratch::new(&mut scratch),
                );
                assert!(
                    out.iter().all(|&v| v == k as i32),
                    "the timed call must be correct"
                );
            });
        });
    }
    group.finish();
}

/// The Phase B workspace matrix: the public two-offer call at each plan the
/// report names --- no offer, the suggested full-depth plan, and the bounded
/// depth-chunked plan --- over deep `k` shapes and one large square. The row
/// label carries the peak caller-provided bytes, because that is the number
/// this phase is about: the bounded plan's label does not grow with `k`, and
/// the suggested one's does. Byte-equality is asserted inside each timed
/// closure, as everywhere in this file.
fn workspace_plans(c: &mut Criterion) {
    const SHAPES: [(usize, usize, usize); 4] = [
        (1, 1_048_576, 1),
        (8, 262_144, 8),
        (16, 400_000, 16),
        (256, 256, 256),
    ];
    let acc_bytes = size_of::<uor_matmul::AccOf<i8>>();

    let mut group = c.benchmark_group("workspace");
    for &(m, k, n) in &SHAPES {
        let shape = Shape { m, k, n };
        let a = vec![1i8; m * k];
        let b = vec![1i8; k * n];
        let report = workspace_report::<i8, Full<i8>>(shape);
        let offers = [
            ("none", 0usize, 0usize),
            (
                "suggested",
                report.suggested.panels,
                report.suggested.accumulators / acc_bytes,
            ),
            (
                "bounded",
                report.bounded.panels,
                report.bounded.accumulators / acc_bytes,
            ),
        ];
        for (plan, panels, accs) in offers {
            let bytes = panels + accs * acc_bytes;
            group.bench_function(format!("{plan}[{bytes}B]/{m}x{k}x{n}"), |bench| {
                let mut out = vec![0i32; m * n];
                let mut panels_buf = vec![0i8; panels];
                let mut acc_buf = vec![0i128; accs];
                bench.iter(|| {
                    uor_matmul::slice::gemm_full(
                        m,
                        k,
                        n,
                        &a,
                        &b,
                        &mut out,
                        &mut panels_buf,
                        &mut acc_buf,
                    )
                    .expect("the product exists");
                    assert!(
                        out.iter().all(|&v| v == k as i32),
                        "the timed call must be correct"
                    );
                });
            });
        }
    }
    group.finish();
}

/// The Gray-walk sign build, measured: the isolated build against the
/// per-codeword build it would displace, and the tabulated gemm end to end at
/// shapes where the build is a visible fraction of the run. The verdict rule
/// is pre-registered in `MEASUREMENT-LOG.md`: the Gray build stays only if
/// the isolated build wins *and* the end-to-end run does not regress; an
/// isolated win with an end-to-end regression is a store-bound decline, and
/// the table stays per-codeword. Compiled here; the run waits for a quiet
/// window, so this group has produced no figure yet.
fn gray_sign(c: &mut Criterion) {
    let mut group = c.benchmark_group("gray_sign");

    // (a) The isolated build at the widest tile, `Sign<8>` and `Sign<16>`.
    for &(space, block) in &[(256usize, 8usize), (65536, 16)] {
        let rows = 16usize;
        let book: Vec<i8> = (0..space * block)
            .map(|i| {
                let (c, t) = (i / block, i % block);
                if (c >> t) & 1 == 1 {
                    1
                } else {
                    -1
                }
            })
            .collect();
        let acts = vec![1i8; block * rows];
        let specs = [
            (
                "independent",
                uor_matmul::kernels::portable_table_bound1(rows, 1),
            ),
            ("gray", uor_matmul::kernels::gray_sign_table(rows, 1)),
        ];
        for (name, spec) in specs {
            group.bench_function(format!("build/{name}/space{space}-blk{block}"), |b| {
                let mut out = vec![0i32; space * rows];
                b.iter(|| {
                    spec.build(space, block, &book, &acts, &mut out);
                    assert!(
                        out[..rows].iter().all(|&v| v == -(block as i32)),
                        "T[0] is the negative row sum"
                    );
                });
            });
        }
    }

    // (b) End to end: the tabulated traversal at a shape where the build is a
    // visible fraction --- small `m`, wide `n`, and a slot count that grows
    // with `k`. `Auto` takes the Gray build at bound 1 with the sign
    // codebook; a named backend is the per-codeword build it would displace.
    let sign =
        uor_matmul::codec::Sign::<i8, uor_matmul::Bnd<1>, 8>::new().expect("bound 1 admits +-1");
    let isa = if cfg!(target_arch = "aarch64") {
        Backend::Neon
    } else if cfg!(target_arch = "x86_64") {
        Backend::Avx2
    } else {
        Backend::Portable
    };
    let backends = [
        ("gray-auto", Backend::Auto),
        ("independent-portable", Backend::Portable),
        ("independent-isa", isa),
    ];
    for &(m, k, n) in &[(8usize, 1024usize, 2100usize), (8, 4096, 2100)] {
        let shape = Shape { m, k, n };
        let (space, block) = (256usize, 8usize);
        let codes: Vec<u16> = (0..n * (k / block))
            .map(|i| (i * 37 % 256) as u16)
            .collect();
        let a: Vec<i8> = (0..m * k).map(|i| ((i % 3) as i8) - 1).collect();
        let w = uor_matmul::codec::CodedMatrix::new(sign, n, k, &codes)
            .expect("the codes describe n x k");

        let run = |backend: Backend, out: &mut Vec<i32>| {
            let mut panel = vec![
                Alphabet::<i8, uor_matmul::Bnd<1>>::ZERO;
                uor_matmul::driver::suggested_tabulation_panel(space, block)
                    .max(n * k + suggested_scratch(shape))
            ];
            let mut accs = vec![
                0i128;
                suggested_tabulation::<i8, uor_matmul::Bnd<1>>(shape, space, block)
                    .max(1)
            ];
            let mut lane_words = vec![
                0i64;
                uor_matmul::driver::suggested_tabulation_lanes::<
                    i8,
                    uor_matmul::Bnd<1>,
                >(shape, space, block,)
                .max(1)
            ];
            let mut ids = vec![0usize; uor_matmul::driver::suggested_tabulation_index(shape)];
            let mut collapse_index = vec![0usize; suggested_collapse_index(m)];
            let mut collapse_rows =
                vec![Alphabet::<i8, uor_matmul::Bnd<1>>::ZERO; suggested_collapse_rows(shape)];
            let av = MatView::row_major(
                as_alphabet::<i8, uor_matmul::Bnd<1>>(&a).expect("the activations fit bound 1"),
                m,
                k,
            )
            .expect("A fits");
            let cv = MatViewMut::row_major(out, m, n).expect("C fits");
            let mut t = TabulatedTriple::new(av, w, cv).expect("the product exists");
            uor_matmul::gemm_tabulated(
                &mut t,
                &Linear::OVERWRITE,
                GemmOptions {
                    traversal: Traversal::Tabulated,
                    backend,
                    ..Default::default()
                },
                &mut Scratch::with_accumulators(&mut panel, &mut accs),
                &mut uor_matmul::driver::Tabulation::with_index(&mut lane_words, &mut ids),
                &mut Collapse::new(&mut collapse_index, &mut collapse_rows),
            );
        };

        let mut want = vec![0i32; m * n];
        run(Backend::Portable, &mut want);
        for (name, backend) in backends {
            group.bench_function(format!("e2e/{name}/{m}x{k}x{n}"), |b| {
                let mut out = vec![0i32; m * n];
                b.iter(|| {
                    run(backend, &mut out);
                    assert_eq!(out, want, "the timed call must be correct");
                });
            });
        }
    }
    group.finish();
}

/// The one-level modular bilinear factorization against the direct packed
/// modular walk it would displace, at the squares the crossover will be read
/// from. `Auto` and the explicit entry run the level where it is admitted
/// (which today is only through the explicit entry: the arm's threshold is
/// `usize::MAX` until the lane is measured); `Packed` is the direct walk.
/// Byte-identity is asserted inside each timed closure, as everywhere in this
/// file, and the verdict rule is pre-registered in `MEASUREMENT-LOG.md`.
/// Compiled here; the run waits for a quiet window.
fn modular_strassen(c: &mut Criterion) {
    let mut group = c.benchmark_group("modular_strassen");
    let wrapping = GemmOptions {
        encode: EncodeMode::Wrapping,
        ..Default::default()
    };
    for &n in &[512usize, 1024, 2048, 4096] {
        let (panels, accs) = modular_level_needs(Shape { m: n, k: n, n });

        // The i8 lane: operands in `{-1, 0, 1}` so the fill stays small and
        // exact at every depth, output wrapped to `i32`.
        let a8: Vec<i8> = (0..n * n).map(|i| ((i * 37) % 3) as i8 - 1).collect();
        let b8: Vec<i8> = (0..n * n).map(|i| ((i * 53) % 3) as i8 - 1).collect();
        let run8 = |strassen: bool, out: &mut Vec<i32>| {
            let mut panel = vec![Alphabet::<i8, Full<i8>>::ZERO; panels];
            let mut acc_buf = vec![0i128; accs];
            let av = MatView::row_major(as_alphabet_full(&a8), n, n).expect("A fits");
            let bv = MatView::row_major(as_alphabet_full(&b8), n, n).expect("B fits");
            let cv = MatViewMut::row_major(out, n, n).expect("C fits");
            let mut t = Triple::new(av, bv, cv).expect("the product exists");
            let mut scratch = Scratch::with_accumulators(&mut panel, &mut acc_buf);
            if strassen {
                uor_matmul::gemm_strassen_modular(
                    &mut t,
                    &Linear::OVERWRITE,
                    wrapping,
                    &mut scratch,
                );
            } else {
                uor_matmul::gemm_packed(&mut t, &Linear::OVERWRITE, wrapping, &mut scratch);
            }
        };
        let mut want8 = vec![0i32; n * n];
        run8(false, &mut want8);
        for (name, strassen) in [("packed", false), ("level", true)] {
            group.bench_function(format!("{name}/i8_{n}cubed"), |bench| {
                let mut out = vec![0i32; n * n];
                bench.iter(|| {
                    run8(strassen, &mut out);
                    assert_eq!(out, want8, "the timed call must be correct");
                });
            });
        }

        // The i32 lane, output wrapped to `i32`.
        let a: Vec<i32> = (0..n * n).map(|i| ((i * 37) % 255) as i32 - 127).collect();
        let b: Vec<i32> = (0..n * n).map(|i| ((i * 53) % 255) as i32 - 127).collect();
        let run = |strassen: bool, out: &mut Vec<i32>| {
            let mut panel = vec![Alphabet::<i32, Full<i32>>::ZERO; panels];
            let mut acc_buf = vec![0i128; accs];
            let av = MatView::row_major(as_alphabet_full(&a), n, n).expect("A fits");
            let bv = MatView::row_major(as_alphabet_full(&b), n, n).expect("B fits");
            let cv = MatViewMut::row_major(out, n, n).expect("C fits");
            let mut t = Triple::new(av, bv, cv).expect("the product exists");
            let mut scratch = Scratch::with_accumulators(&mut panel, &mut acc_buf);
            if strassen {
                uor_matmul::gemm_strassen_modular(
                    &mut t,
                    &Linear::OVERWRITE,
                    wrapping,
                    &mut scratch,
                );
            } else {
                uor_matmul::gemm_packed(&mut t, &Linear::OVERWRITE, wrapping, &mut scratch);
            }
        };
        let mut want = vec![0i32; n * n];
        run(false, &mut want);
        for (name, strassen) in [("packed", false), ("level", true)] {
            group.bench_function(format!("{name}/i32_{n}cubed"), |bench| {
                let mut out = vec![0i32; n * n];
                bench.iter(|| {
                    run(strassen, &mut out);
                    assert_eq!(out, want, "the timed call must be correct");
                });
            });
        }
    }
    group.finish();
}

/// `CG-19`: the selection lane beside the accumulation lane, and the two
/// witness mechanisms beside each other.
///
/// Two questions, and they need different operands, which is why this is one
/// group with two labelled halves rather than one sweep.
///
/// **The lanes.** `lane/tropical` and `lane/ring` are the *same function* ---
/// `uor_matmul_gemm::gemm` --- at two instantiations of `E`, which is what
/// `CD-29` claims and what this half prices. Neither is offered a kernel, so the
/// comparison is between two arithmetics and not between a hand-written
/// sequence and a portable loop. `lane/ring-packed` is the third row for the
/// reason a two-row table would mislead: it is what a ring caller actually gets,
/// and the gap between it and `lane/ring` is the kernel selection, not the
/// semiring. The operands carry no masked lane here, because a masked lane
/// performs no arithmetic at all --- that is the absorbing law, and a mask
/// density silently chosen by the harness would be a throughput knob rather
/// than a measurement.
///
/// **The mechanisms.** `Lexicographic` carries `(value, index)` through one
/// pass; `ComparePass` takes the value in one pass and the least index in a
/// second, compare-only one. The second pass stops at the first index attaining
/// the maximum, so *where the maximum sits* is the whole of its cost --- and
/// timing it on operands whose maximum sits at index zero would report a second
/// pass that never happens. Both ends are therefore timed: `tie-dense` is the
/// corpus draw, where a tie is the common case and the scan stops early, and
/// `max-last` places the unique maximum at the final index, which is the scan's
/// worst case. Reading only one of the two would be reading a knob.
fn tropical(c: &mut Criterion) {
    const SHAPES: [(usize, usize, usize); 3] = [(64, 64, 64), (128, 128, 128), (16, 4096, 16)];
    /// The corpus seed the selection gates walk, so the bench and the gates
    /// price the same operands.
    const SEED: u64 = 20_260_805;

    /// `max_p (a[i,p] + b[p,j])`, computed here so that every timed closure can
    /// assert its own answer (`CG-06`).
    fn max_plus_ref(
        m: usize,
        k: usize,
        n: usize,
        a: &[Option<i64>],
        b: &[Option<i64>],
    ) -> Vec<Trop<i32>> {
        let mut out = Vec::with_capacity(m * n);
        for i in 0..m {
            for j in 0..n {
                let mut best: Option<i64> = None;
                for p in 0..k {
                    if let (Some(x), Some(y)) = (a[i * k + p], b[p * n + j]) {
                        best = best.max(Some(x + y));
                    }
                }
                out.push(best.map_or(Trop::NEG_INF, |v| Trop::finite(v as i32)));
            }
        }
        out
    }

    let mut group = c.benchmark_group("tropical");
    for &(m, k, n) in &SHAPES {
        let shape = format!("{m}x{k}x{n}");
        let case = Case {
            m,
            k,
            n,
            seed: SEED,
        };
        // Unmasked, so the two lanes do the same amount of arithmetic.
        let (da, db) = (
            case.draw_tropical(m * k, 1, Mask::NONE),
            case.draw_tropical(k * n, 2, Mask::NONE),
        );
        let ta = case.fill_tropical_i8(m * k, 1, Mask::NONE);
        let tb = case.fill_tropical_i8(k * n, 2, Mask::NONE);
        let want = max_plus_ref(m, k, n, &da, &db);

        group.bench_function(format!("lane/tropical/{shape}"), |bench| {
            let mut out = vec![Trop::<i32>::NEG_INF; m * n];
            let mut scratch = vec![Alphabet::ZERO; k];
            bench.iter(|| {
                let av = MatView::row_major(as_alphabet_tropical(&ta), m, k).expect("A fits");
                let bv = MatView::row_major(as_alphabet_tropical(&tb), k, n).expect("B fits");
                let cv = MatViewMut::row_major(&mut out, m, n).expect("C fits");
                let mut t = Triple::new(av, bv, cv).expect("the product exists");
                gemm_dense(
                    &mut t,
                    &MaxPlus::OVERWRITE,
                    GemmOptions::default(),
                    &mut Scratch::new(&mut scratch),
                );
                assert_eq!(out, want, "the timed call must be correct");
            });
        });

        // The same driver, the same offer, the same shape; only `E` differs.
        let ra: Vec<i8> = da.iter().map(|v| v.unwrap_or(0) as i8).collect();
        let rb: Vec<i8> = db.iter().map(|v| v.unwrap_or(0) as i8).collect();
        let ring_want = uor_matmul_validate::ours_i8_i32(m, k, n, &ra, &rb, EncodeMode::Wrapping);

        group.bench_function(format!("lane/ring/{shape}"), |bench| {
            let mut out = vec![0i32; m * n];
            let mut scratch = vec![Alphabet::<i8, Full<i8>>::ZERO; k];
            bench.iter(|| {
                let av = MatView::row_major(as_alphabet_full(&ra), m, k).expect("A fits");
                let bv = MatView::row_major(as_alphabet_full(&rb), k, n).expect("B fits");
                let cv = MatViewMut::row_major(&mut out, m, n).expect("C fits");
                let mut t = Triple::new(av, bv, cv).expect("the product exists");
                gemm_dense(
                    &mut t,
                    &Linear::OVERWRITE,
                    GemmOptions {
                        encode: EncodeMode::Wrapping,
                        ..Default::default()
                    },
                    &mut Scratch::new(&mut scratch),
                );
                assert_eq!(out, ring_want, "the timed call must be correct");
            });
        });

        group.bench_function(format!("lane/ring-packed/{shape}"), |bench| {
            let mut out = vec![0i32; m * n];
            let mut scratch =
                vec![Alphabet::<i8, Full<i8>>::ZERO; suggested_scratch(Shape { m, k, n })];
            bench.iter(|| {
                let av = MatView::row_major(as_alphabet_full(&ra), m, k).expect("A fits");
                let bv = MatView::row_major(as_alphabet_full(&rb), k, n).expect("B fits");
                let cv = MatViewMut::row_major(&mut out, m, n).expect("C fits");
                let mut t = Triple::new(av, bv, cv).expect("the product exists");
                uor_matmul::gemm_packed(
                    &mut t,
                    &Linear::OVERWRITE,
                    GemmOptions {
                        encode: EncodeMode::Wrapping,
                        ..Default::default()
                    },
                    &mut Scratch::new(&mut scratch),
                );
                assert_eq!(out, ring_want, "the timed call must be correct");
            });
        });

        // The mechanism half. `tie-dense` is the corpus's own masked draw ---
        // the operands the gates run on --- and `max-last` puts the unique
        // maximum at the last index of every reduction, which is where the
        // compare pass has to walk the whole depth.
        let last_a: Vec<Trop<i8>> = (0..m * k)
            .map(|i| Trop::finite(if i % k == k - 1 { 1 } else { 0 }))
            .collect();
        let last_b: Vec<Trop<i8>> = (0..k * n)
            .map(|i| Trop::finite(if i / n == k - 1 { 1 } else { 0 }))
            .collect();
        let masked_a = case.fill_tropical_i8(m * k, 1, Mask::LEFT);
        let masked_b = case.fill_tropical_i8(k * n, 2, Mask::RIGHT);
        let tie_want = max_plus_ref(
            m,
            k,
            n,
            &case.draw_tropical(m * k, 1, Mask::LEFT),
            &case.draw_tropical(k * n, 2, Mask::RIGHT),
        );
        let last_want = vec![Trop::<i32>::finite(2); m * n];

        for (fill, a, b, cells) in [
            ("tie-dense", &masked_a, &masked_b, &tie_want),
            ("max-last", &last_a, &last_b, &last_want),
        ] {
            for mechanism in [Witness::Lexicographic, Witness::ComparePass] {
                let name = match mechanism {
                    Witness::Lexicographic => "lexicographic",
                    _ => "compare-pass",
                };
                group.bench_function(format!("witness/{name}/{fill}/{shape}"), |bench| {
                    let mut out = vec![Trop::<i32>::NEG_INF; m * n];
                    let mut witness = vec![usize::MAX; m * n];
                    let mut scratch = vec![Alphabet::ZERO; k];
                    bench.iter(|| {
                        let av = MatView::row_major(as_alphabet_tropical(a), m, k).expect("A fits");
                        let bv = MatView::row_major(as_alphabet_tropical(b), k, n).expect("B fits");
                        let cv = MatViewMut::row_major(&mut out, m, n).expect("C fits");
                        let wv = MatViewMut::row_major(&mut witness, m, n).expect("W fits");
                        let t = Triple::new(av, bv, cv).expect("the product exists");
                        let mut s = SelectedTriple::new(t, wv).expect("the witness conforms");
                        gemm_selected(
                            &mut s,
                            &MaxPlus::OVERWRITE,
                            GemmOptions::default(),
                            mechanism,
                            &mut Scratch::new(&mut scratch),
                        );
                        assert_eq!(&out, cells, "the timed call must be correct");
                    });
                });
            }
        }
    }
    group.finish();

    // The corpus the gates walk, so that a reader of this group can see the
    // shapes it was validated on beside the shapes it was timed on.
    eprintln!(
        "tropical: timed at {} shapes; gated at {} corpus cases",
        SHAPES.len(),
        Corpus::tropical(SEED).cases.len()
    );
}

criterion_group!(
    benches,
    scaling_report,
    vs_oracles,
    public_api,
    workspace_plans,
    gray_sign,
    modular_strassen,
    tropical
);
criterion_main!(benches);
