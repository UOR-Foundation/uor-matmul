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
use uor_matmul::prelude::*;
use uor_matmul::{suggested_float_panels, suggested_scratch, Shape};
use uor_matmul_core::{Alphabet, EncodeMode, Full, PackedCode};
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
                // The packed driver, which is what the library runs. Timing the
                // generic traversal would time a factorization no caller gets.
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

criterion_group!(benches, scaling_report, vs_oracles);
criterion_main!(benches);
