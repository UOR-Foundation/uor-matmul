//! Head-to-head throughput against every oracle, for development.
//!
//! Not a claim: `CG-*` and `just scaling` are where the measured figures live,
//! with confidence intervals and a credibility judgement. This reports the
//! *best* of several runs, which is the robust estimator for throughput on a
//! shared machine --- the minimum is the run least interfered with, and the
//! mean on a noisy box measures the neighbours.

use std::time::Instant;

use uor_matmul::prelude::*;
use uor_matmul_core::EncodeMode;

/// Best of `reps` runs, in seconds.
fn best(reps: usize, mut run: impl FnMut() -> f64) -> f64 {
    (0..reps).map(|_| run()).fold(f64::INFINITY, f64::min)
}

fn gmacs(n: usize, secs: f64) -> f64 {
    (n as f64).powi(3) / 1e9 / secs
}

/// Compare every path against every oracle that speaks the same types.
fn main() {
    const REPS: usize = 5;
    println!(
        "{:<10} {:>22} {:>22} {:>22}",
        "shape", "uor-matmul", "oracle", "ratio"
    );

    // --- integer: i32 against ndarray and nalgebra -------------------------
    #[cfg(feature = "ref-ndarray")]
    for n in [64usize, 128, 256, 384] {
        use uor_matmul_validate::oracle::{NdArray, Oracle};
        let a = vec![1i32; n * n];
        let b = vec![1i32; n * n];

        let ours = best(REPS, || {
            let mut c = vec![0i32; n * n];
            let t = Instant::now();
            {
                let av = MatView::row_major(as_alphabet_full(&a), n, n).unwrap();
                let bv = MatView::row_major(as_alphabet_full(&b), n, n).unwrap();
                let cv = MatViewMut::row_major(&mut c, n, n).unwrap();
                let mut tr = Triple::new(av, bv, cv).unwrap();
                gemm(
                    &mut tr,
                    &Linear::OVERWRITE,
                    GemmOptions {
                        encode: EncodeMode::Wrapping,
                        ..Default::default()
                    },
                    &mut Scratch::none(),
                );
            }
            assert!(c.iter().all(|&v| v == n as i32));
            t.elapsed().as_secs_f64()
        });
        let theirs = best(REPS, || {
            let t = Instant::now();
            let c = NdArray::product_i32(n, n, n, &a, &b);
            let e = t.elapsed().as_secs_f64();
            assert!(c.iter().all(|&v| v == n as i32));
            e
        });
        println!(
            "i32 {n:<6} {:>15.2} Gmac/s {:>15.2} Gmac/s {:>18}",
            gmacs(n, ours),
            gmacs(n, theirs),
            format!("{:.2}x ndarray", ours / theirs)
        );
    }

    #[cfg(feature = "ref-nalgebra")]
    for n in [64usize, 128, 256, 384] {
        use uor_matmul_validate::oracle::{Nalgebra, Oracle};
        let a = vec![1i32; n * n];
        let b = vec![1i32; n * n];
        let ours = best(REPS, || {
            let mut c = vec![0i32; n * n];
            let t = Instant::now();
            {
                let av = MatView::row_major(as_alphabet_full(&a), n, n).unwrap();
                let bv = MatView::row_major(as_alphabet_full(&b), n, n).unwrap();
                let cv = MatViewMut::row_major(&mut c, n, n).unwrap();
                let mut tr = Triple::new(av, bv, cv).unwrap();
                gemm(
                    &mut tr,
                    &Linear::OVERWRITE,
                    GemmOptions {
                        encode: EncodeMode::Wrapping,
                        ..Default::default()
                    },
                    &mut Scratch::none(),
                );
            }
            t.elapsed().as_secs_f64()
        });
        let theirs = best(REPS, || {
            let t = Instant::now();
            let c = Nalgebra::product_i32(n, n, n, &a, &b);
            let e = t.elapsed().as_secs_f64();
            assert!(c.iter().all(|&v| v == n as i32));
            e
        });
        println!(
            "i32 {n:<6} {:>15.2} Gmac/s {:>15.2} Gmac/s {:>18}",
            gmacs(n, ours),
            gmacs(n, theirs),
            format!("{:.2}x nalgebra", ours / theirs)
        );
    }

    // --- W8A8, the instantiation the instructions exist for -----------------
    for n in [64usize, 128, 256, 384] {
        let a = vec![1i8; n * n];
        let b = vec![1i8; n * n];
        let mut scratch = vec![Alphabet::<i8, Full<i8>>::ZERO; 1 << 20];
        let ours = best(REPS, || {
            let mut c = vec![0i32; n * n];
            let t = Instant::now();
            {
                let av = MatView::row_major(as_alphabet_full(&a), n, n).unwrap();
                let bv = MatView::row_major(as_alphabet_full(&b), n, n).unwrap();
                let cv = MatViewMut::row_major(&mut c, n, n).unwrap();
                let mut tr = Triple::new(av, bv, cv).unwrap();
                uor_matmul::gemm_w8a8(
                    &mut tr,
                    &Linear::OVERWRITE,
                    GemmOptions {
                        encode: EncodeMode::Wrapping,
                        ..Default::default()
                    },
                    &mut Scratch::new(&mut scratch),
                );
            }
            assert!(c.iter().all(|&v| v == n as i32));
            t.elapsed().as_secs_f64()
        });
        println!(
            "W8A8 {n:<5} {:>15.2} Gmac/s {:>22} {:>18}",
            gmacs(n, ours),
            "--",
            "no i8 oracle"
        );
    }

    // --- float: our exact sum against the classical approximations ----------
    #[cfg(feature = "ref-matrixmultiply")]
    for n in [64usize, 128, 256] {
        use uor_matmul_validate::oracle::{FloatOracle, MatrixMultiply};
        let a = vec![1.0f32; n * n];
        let b = vec![1.0f32; n * n];
        let ours = best(REPS, || {
            let mut c = vec![0.0f32; n * n];
            let t = Instant::now();
            {
                let av = MatView::row_major(&a, n, n).unwrap();
                let bv = MatView::row_major(&b, n, n).unwrap();
                let cv = MatViewMut::row_major(&mut c, n, n).unwrap();
                let mut tr = Triple::new(av, bv, cv).unwrap();
                uor_matmul::gemm_float(&mut tr, &Linear::OVERWRITE, GemmOptions::default());
            }
            assert!(c.iter().all(|&v| v == n as f32));
            t.elapsed().as_secs_f64()
        });
        let theirs = best(REPS, || {
            let t = Instant::now();
            let c = MatrixMultiply::product_f32(n, n, n, &a, &b);
            let e = t.elapsed().as_secs_f64();
            assert!(c.iter().all(|&v| v == n as f32));
            e
        });
        println!(
            "f32 {n:<6} {:>15.2} Gmac/s {:>15.2} Gmac/s {:>18}",
            gmacs(n, ours),
            gmacs(n, theirs),
            format!("{:.1}x matrixmul", ours / theirs)
        );
    }

    #[cfg(feature = "ref-faer")]
    for n in [64usize, 128, 256] {
        use uor_matmul_validate::oracle::{Faer, FloatOracle};
        let a = vec![1.0f32; n * n];
        let b = vec![1.0f32; n * n];
        let ours = best(REPS, || {
            let mut c = vec![0.0f32; n * n];
            let t = Instant::now();
            {
                let av = MatView::row_major(&a, n, n).unwrap();
                let bv = MatView::row_major(&b, n, n).unwrap();
                let cv = MatViewMut::row_major(&mut c, n, n).unwrap();
                let mut tr = Triple::new(av, bv, cv).unwrap();
                uor_matmul::gemm_float(&mut tr, &Linear::OVERWRITE, GemmOptions::default());
            }
            t.elapsed().as_secs_f64()
        });
        let theirs = best(REPS, || {
            let t = Instant::now();
            let c = Faer::product_f32(n, n, n, &a, &b);
            let e = t.elapsed().as_secs_f64();
            assert!(c.iter().all(|&v| v == n as f32));
            e
        });
        println!(
            "f32 {n:<6} {:>15.2} Gmac/s {:>15.2} Gmac/s {:>18}",
            gmacs(n, ours),
            gmacs(n, theirs),
            format!("{:.1}x faer", ours / theirs)
        );
    }
}
