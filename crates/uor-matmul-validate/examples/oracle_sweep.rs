//! Throughput, latency, and scaling, against every oracle, from miniscule to
//! astronomical.
//!
//! Three questions, because they have different answers:
//!
//! - **Latency** at sizes where a call is dominated by whatever happens before
//!   the arithmetic. This is where a library with a thread pool or a blocking
//!   decision loses to one with neither.
//! - **Throughput** at sizes where the arithmetic dominates.
//! - **Scaling** across the whole range, which is the only comparison that
//!   survives a change of machine.
//!
//! Every figure is `open`: measured and reported, never asserted. What *is*
//! asserted, inside the timed region, is that the answer is right --- a speed
//! measured on the wrong bytes is not a measurement.

use std::time::Instant;

use uor_matmul::prelude::*;
use uor_matmul_core::{EncodeMode, Full, PackedCode, Shape};

/// Best of `reps`, in seconds per call.
///
/// The minimum is the run least interfered with. On a shared machine the mean
/// measures the neighbours.
fn best(reps: usize, mut run: impl FnMut() -> f64) -> f64 {
    (0..reps).map(|_| run()).fold(f64::INFINITY, f64::min)
}

/// Enough repetitions that the clock is not what is being measured.
fn reps_for(macs: f64) -> usize {
    if macs < 1e4 {
        10_000
    } else if macs < 1e6 {
        100
    } else if macs < 1e8 {
        5
    } else {
        2
    }
}

/// `i8 x i8 -> i32`, the instantiation the SIMD instructions name.
fn ours_i8(n: usize, a: &[i8], b: &[i8], scratch: &mut [Alphabet<i8, Full<i8>>]) -> f64 {
    let mut c = vec![0i32; n * n];
    let reps = reps_for((n as f64).powi(3));
    let t = best(reps, || {
        let s = Instant::now();
        {
            let av = MatView::row_major(as_alphabet_full(a), n, n).unwrap();
            let bv = MatView::row_major(as_alphabet_full(b), n, n).unwrap();
            let cv = MatViewMut::row_major(&mut c, n, n).unwrap();
            let mut tr = Triple::new(av, bv, cv).unwrap();
            uor_matmul::gemm_packed(
                &mut tr,
                &Linear::OVERWRITE,
                GemmOptions {
                    encode: EncodeMode::Wrapping,
                    ..Default::default()
                },
                &mut Scratch::new(scratch),
            );
        }
        s.elapsed().as_secs_f64()
    });
    assert!(
        c.iter().all(|&v| v == n as i32),
        "the timed call must be correct"
    );
    t
}

/// `i32 x i32 -> i32`.
fn ours_i32(n: usize, a: &[i32], b: &[i32], scratch: &mut [Alphabet<i32, Full<i32>>]) -> f64 {
    let mut c = vec![0i32; n * n];
    let reps = reps_for((n as f64).powi(3));
    let t = best(reps, || {
        let s = Instant::now();
        {
            let av = MatView::row_major(as_alphabet_full(a), n, n).unwrap();
            let bv = MatView::row_major(as_alphabet_full(b), n, n).unwrap();
            let cv = MatViewMut::row_major(&mut c, n, n).unwrap();
            let mut tr = Triple::new(av, bv, cv).unwrap();
            uor_matmul::gemm_packed(
                &mut tr,
                &Linear::OVERWRITE,
                GemmOptions {
                    encode: EncodeMode::Wrapping,
                    ..Default::default()
                },
                &mut Scratch::new(scratch),
            );
        }
        s.elapsed().as_secs_f64()
    });
    assert!(c.iter().all(|&v| v == n as i32));
    t
}

/// `f32`, computed exactly.
fn ours_f32(n: usize, a: &[f32], b: &[f32], pa: &mut [PackedCode], pb: &mut [PackedCode]) -> f64 {
    let mut c = vec![0.0f32; n * n];
    let reps = reps_for((n as f64).powi(3));
    let t = best(reps, || {
        let s = Instant::now();
        {
            let av = MatView::row_major(a, n, n).unwrap();
            let bv = MatView::row_major(b, n, n).unwrap();
            let cv = MatViewMut::row_major(&mut c, n, n).unwrap();
            let mut tr = Triple::new(av, bv, cv).unwrap();
            uor_matmul::gemm_float_packed(
                &mut tr,
                &Linear::OVERWRITE,
                GemmOptions::default(),
                pa,
                pb,
            );
        }
        s.elapsed().as_secs_f64()
    });
    assert!(c.iter().all(|&v| v == n as f32));
    t
}

fn gmacs(n: usize, secs: f64) -> f64 {
    (n as f64).powi(3) / 1e9 / secs
}

/// Least squares in log-log space; returns the exponent against MAC count.
fn exponent(points: &[(f64, f64)]) -> f64 {
    let usable: Vec<(f64, f64)> = points
        .iter()
        .filter(|(x, y)| *x > 0.0 && *y > 0.0)
        .map(|(x, y)| (x.log10(), y.log10()))
        .collect();
    if usable.len() < 3 {
        return f64::NAN;
    }
    let n = usable.len() as f64;
    let mx = usable.iter().map(|p| p.0).sum::<f64>() / n;
    let my = usable.iter().map(|p| p.1).sum::<f64>() / n;
    let sxx: f64 = usable.iter().map(|p| (p.0 - mx).powi(2)).sum();
    let sxy: f64 = usable.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
    sxy / sxx
}

/// From one element to as many as the machine will hold in a reasonable time.
///
/// Geometric, and spanning nine orders of magnitude in MAC count, because a
/// sweep that does not reach both ends measures neither latency nor asymptote.
const SIZES: &[usize] = &[1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1024];

/// Run the whole sweep.
fn main() {
    // Say which kernels are actually running. A sweep that silently measured
    // the portable kernel while the reader assumed SIMD would be worse than no
    // sweep at all.
    println!("# Throughput, latency, and scaling --- every figure is `open`");
    println!(
        "# i8 kernels available: {}",
        uor_matmul::kernels::available_i8()
            .map(|s| s.backend.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!(
        "# i32 modular kernels available: {}",
        uor_matmul::kernels::available_i32_modular()
            .map(|s| s.backend.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("# best-of-N per point; the answer is asserted inside the timed region");
    println!();
    println!(
        "{:>6} {:>12} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11}",
        "n", "macs", "uor i8", "uor i32", "ndarray", "nalgebra", "uor f32", "matrixmul"
    );

    let mut fit_i8 = Vec::new();
    let mut fit_i32 = Vec::new();
    let mut fit_nd = Vec::new();
    let mut fit_na = Vec::new();
    let mut fit_f32 = Vec::new();
    let mut fit_mm = Vec::new();

    for &n in SIZES {
        let macs = (n as f64).powi(3);
        let a8 = vec![1i8; n * n];
        let b8 = vec![1i8; n * n];
        let a32 = vec![1i32; n * n];
        let b32 = vec![1i32; n * n];
        let af = vec![1.0f32; n * n];
        let bf = vec![1.0f32; n * n];

        // The panels the library needs. Sized from `suggested_scratch`, which
        // is a query with a closed form and no allocation of its own.
        let want = uor_matmul::suggested_scratch(Shape { m: n, k: n, n }).max(64 * n);
        let mut s8 = vec![Alphabet::<i8, Full<i8>>::ZERO; want];
        let mut s32 = vec![Alphabet::<i32, Full<i32>>::ZERO; want];
        let mut pa = vec![PackedCode::default(); n];
        let mut pb = vec![PackedCode::default(); n * n];

        let t8 = ours_i8(n, &a8, &b8, &mut s8);
        let t32 = ours_i32(n, &a32, &b32, &mut s32);
        let tf = ours_f32(n, &af, &bf, &mut pa, &mut pb);

        #[cfg(feature = "ref-ndarray")]
        let tnd = {
            use uor_matmul_validate::oracle::{NdArray, Oracle};
            best(reps_for(macs), || {
                let s = Instant::now();
                let c = NdArray::product_i32(n, n, n, &a32, &b32);
                let e = s.elapsed().as_secs_f64();
                assert!(c.iter().all(|&v| v == n as i32));
                e
            })
        };
        #[cfg(not(feature = "ref-ndarray"))]
        let tnd = f64::NAN;

        #[cfg(feature = "ref-nalgebra")]
        let tna = {
            use uor_matmul_validate::oracle::{Nalgebra, Oracle};
            best(reps_for(macs), || {
                let s = Instant::now();
                let c = Nalgebra::product_i32(n, n, n, &a32, &b32);
                let e = s.elapsed().as_secs_f64();
                assert!(c.iter().all(|&v| v == n as i32));
                e
            })
        };
        #[cfg(not(feature = "ref-nalgebra"))]
        let tna = f64::NAN;

        #[cfg(feature = "ref-matrixmultiply")]
        let tmm = {
            use uor_matmul_validate::oracle::{FloatOracle, MatrixMultiply};
            best(reps_for(macs), || {
                let s = Instant::now();
                let c = MatrixMultiply::product_f32(n, n, n, &af, &bf);
                let e = s.elapsed().as_secs_f64();
                assert!(c.iter().all(|&v| v == n as f32));
                e
            })
        };
        #[cfg(not(feature = "ref-matrixmultiply"))]
        let tmm = f64::NAN;

        println!(
            "{n:>6} {:>12.3e} {:>11.3} {:>11.3} {:>11.3} {:>11.3} {:>11.3} {:>11.3}",
            macs,
            gmacs(n, t8),
            gmacs(n, t32),
            gmacs(n, tnd),
            gmacs(n, tna),
            gmacs(n, tf),
            gmacs(n, tmm),
        );

        fit_i8.push((macs, t8));
        fit_i32.push((macs, t32));
        fit_nd.push((macs, tnd));
        fit_na.push((macs, tna));
        fit_f32.push((macs, tf));
        fit_mm.push((macs, tmm));
    }

    println!();
    println!("# latency at n = 1, in nanoseconds per call --- the small-shape question");
    let n = 1usize;
    let a8 = vec![1i8; 1];
    let b8 = vec![1i8; 1];
    let mut s8 = vec![Alphabet::<i8, Full<i8>>::ZERO; 64];
    println!(
        "uor i8      {:>10.1} ns",
        ours_i8(n, &a8, &b8, &mut s8) * 1e9
    );
    #[cfg(feature = "ref-ndarray")]
    {
        use uor_matmul_validate::oracle::{NdArray, Oracle};
        let a = vec![1i32; 1];
        let b = vec![1i32; 1];
        let t = best(10_000, || {
            let s = Instant::now();
            let _ = NdArray::product_i32(1, 1, 1, &a, &b);
            s.elapsed().as_secs_f64()
        });
        println!("ndarray     {:>10.1} ns", t * 1e9);
    }
    #[cfg(feature = "ref-matrixmultiply")]
    {
        use uor_matmul_validate::oracle::{FloatOracle, MatrixMultiply};
        let a = vec![1.0f32; 1];
        let b = vec![1.0f32; 1];
        let t = best(10_000, || {
            let s = Instant::now();
            let _ = MatrixMultiply::product_f32(1, 1, 1, &a, &b);
            s.elapsed().as_secs_f64()
        });
        println!("matrixmul   {:>10.1} ns", t * 1e9);
    }

    println!();
    println!("# fitted exponent against MAC count, over the whole sweep");
    println!("uor i8      {:>8.4}", exponent(&fit_i8));
    println!("uor i32     {:>8.4}", exponent(&fit_i32));
    println!("ndarray     {:>8.4}", exponent(&fit_nd));
    println!("nalgebra    {:>8.4}", exponent(&fit_na));
    println!("uor f32     {:>8.4}", exponent(&fit_f32));
    println!("matrixmul   {:>8.4}", exponent(&fit_mm));
}
