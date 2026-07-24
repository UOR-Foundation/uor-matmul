//! `CG-01` .. `CG-07`: the scaling axis (§13, R12, C3).
//!
//! Every figure below is `open`: measured and reported, never asserted. None of
//! these tests fails on a performance number, and the honesty meta-gate would
//! fail the build if the documentation said one of them established anything.
//!
//! What they *do* fail on is a broken measurement --- a fit that cannot recover
//! a known exponent, a residency ratio that is not a ratio, a timed harness
//! that timed the wrong answer. A measurement apparatus that cannot be wrong is
//! not measuring.

use uor_matmul_validate::scaling::{self, Labelled, Observation, Point, Sweep};

/// `CG-01`: the arithmetic scaling exponent, ours **and every oracle's**, over
/// the same sweep.
///
/// C3 is a hard constraint: scaling is compared against the oracle's scaling,
/// not reported alone. A throughput table with no asymptotic content would say
/// nothing about whether the two implementations have the same shape, which is
/// the only comparison that survives a change of machine.
#[test]
fn arithmetic_scaling_exponent_cg_01() {
    let sweep = Sweep::standard();

    let ours = Labelled {
        id: "uor-matmul",
        name: "uor-matmul i32",
        fit: scaling::fit_timed(&sweep, scaling::ours_i32_timed),
    };

    let mut oracles: Vec<Labelled> = Vec::new();

    #[cfg(feature = "ref-ndarray")]
    {
        use uor_matmul_validate::oracle::{NdArray, Oracle};
        oracles.push(Labelled {
            id: "CX-01",
            name: "ndarray i32",
            fit: scaling::fit_timed(&sweep, |p: &Point| {
                let a = vec![1i32; p.m * p.k];
                let b = vec![1i32; p.k * p.n];
                let started = std::time::Instant::now();
                let out = NdArray::product_i32(p.m, p.k, p.n, &a, &b);
                let elapsed = started.elapsed().as_secs_f64();
                // The timed call must be correct on the oracle's side too.
                assert!(out.iter().all(|&v| v == p.k as i32));
                elapsed
            }),
        });
    }

    #[cfg(feature = "ref-nalgebra")]
    {
        use uor_matmul_validate::oracle::{Nalgebra, Oracle};
        oracles.push(Labelled {
            id: "CX-02",
            name: "nalgebra i32",
            fit: scaling::fit_timed(&sweep, |p: &Point| {
                let a = vec![1i32; p.m * p.k];
                let b = vec![1i32; p.k * p.n];
                let started = std::time::Instant::now();
                let out = Nalgebra::product_i32(p.m, p.k, p.n, &a, &b);
                let elapsed = started.elapsed().as_secs_f64();
                assert!(out.iter().all(|&v| v == p.k as i32));
                elapsed
            }),
        });
    }

    let report = scaling::Report::new(sweep, ours.clone(), oracles);
    report.emit();

    let f = ours
        .fit
        .expect("the standard sweep has enough points to fit");
    // An O(m k n) implementation fits near 1.0 against MAC count. The interval
    // is reported rather than asserted narrow, because a shared runner is a
    // noisy machine and pretending otherwise would be the dishonest part.
    assert!(f.samples >= 3, "the fit must have used the sweep");
    assert!(f.exponent.is_finite(), "the fit must produce a number");
}

/// `CG-01`, float half: our exact `f32` path against the classical ones.
///
/// The exponents are what to compare. The constant is expected to favour the
/// oracles --- exact accumulation into a 619-bit register costs more per
/// element than one FMA, which is non-goal N4 stated as a number rather than as
/// an excuse.
#[test]
fn float_scaling_exponent_cg_01() {
    let sweep = Sweep::standard();

    let ours = Labelled {
        id: "uor-matmul",
        name: "uor-matmul f32 exact",
        fit: scaling::fit_timed(&sweep, scaling::ours_f32_timed),
    };

    let mut oracles: Vec<Labelled> = Vec::new();

    #[cfg(feature = "ref-matrixmultiply")]
    {
        use uor_matmul_validate::oracle::{FloatOracle, MatrixMultiply};
        oracles.push(Labelled {
            id: "CX-05",
            name: "matrixmultiply f32",
            fit: scaling::fit_timed(&sweep, |p: &Point| {
                let a = vec![1.0f32; p.m * p.k];
                let b = vec![1.0f32; p.k * p.n];
                let started = std::time::Instant::now();
                let out = MatrixMultiply::product_f32(p.m, p.k, p.n, &a, &b);
                let elapsed = started.elapsed().as_secs_f64();
                assert!(out.iter().all(|&v| v == p.k as f32));
                elapsed
            }),
        });
    }

    #[cfg(feature = "ref-faer")]
    {
        use uor_matmul_validate::oracle::{Faer, FloatOracle};
        oracles.push(Labelled {
            id: "CX-06",
            name: "faer f32",
            fit: scaling::fit_timed(&sweep, |p: &Point| {
                let a = vec![1.0f32; p.m * p.k];
                let b = vec![1.0f32; p.k * p.n];
                let started = std::time::Instant::now();
                let out = Faer::product_f32(p.m, p.k, p.n, &a, &b);
                let elapsed = started.elapsed().as_secs_f64();
                assert!(out.iter().all(|&v| v == p.k as f32));
                elapsed
            }),
        });
    }

    scaling::Report::new(sweep, ours.clone(), oracles).emit();
    assert!(ours.fit.expect("fits").exponent.is_finite());
}

/// `CG-02`: per-axis exponents for `m`, `n`, and `k` separately.
///
/// A cube sweep cannot distinguish an implementation that is linear in `k` and
/// quadratic in `m` from one that is the reverse. Varying one axis at a time is
/// what makes the claim about *shape* rather than about size.
#[test]
fn per_axis_scaling_exponents_cg_02() {
    for axis in ["m", "k", "n"] {
        let sweep = Sweep::per_axis(axis);
        let f = scaling::fit_ours(&sweep).expect("each axis sweep fits");
        eprintln!(
            "CG-02 (open): axis {axis}: exponent {:.4} +/- {:.4}, n = {}",
            f.exponent, f.confidence_half_width, f.samples
        );
        assert!(f.exponent.is_finite());
    }
}

/// `CG-03`: residency, per codec, as a ratio.
#[test]
fn residency_scaling_cg_03() {
    for (name, bits) in [("i8 dense", 8.0), ("i4 grid", 4.0), ("E8 codebook", 1.0)] {
        let r = scaling::residency_ratio(bits);
        eprintln!("CG-03 (open): {name} touches {r:.3} bytes per decoded weight");
        assert!(r > 0.0);
    }
    // E8 stores one byte per eight decoded weights.
    assert_eq!(scaling::residency_ratio(1.0), 0.125);
}

/// `CG-04`: working-set scaling --- `suggested_scratch` against the shape.
///
/// The comparison the plan asks for is against each oracle's *measured internal
/// allocation*. Ours is a query with a closed form and no allocation at all, so
/// what is reported is that closed form; `CG-05` reports the allocation count
/// that makes the comparison one-sided.
#[test]
fn working_set_scaling_cg_04() {
    use uor_matmul_core::Shape;
    let mut previous = 0usize;
    for s in [16usize, 32, 64, 128, 256, 512] {
        let want = uor_matmul::suggested_scratch(Shape { m: s, k: s, n: s });
        eprintln!("CG-04 (open): {s}^3 suggests {want} scratch elements");
        // Monotone in the shape, and bounded by the blocking parameters rather
        // than by the problem: that bound is the whole point of a fixed panel.
        assert!(
            want >= previous,
            "the suggestion must not shrink as the shape grows"
        );
        previous = want;
    }
    let huge = uor_matmul::suggested_scratch(Shape {
        m: 1 << 20,
        k: 1 << 20,
        n: 1 << 20,
    });
    eprintln!("CG-04 (open): a 2^20 cube still suggests only {huge} elements");
    assert!(
        huge < 1 << 20,
        "the suggestion is bounded by the blocking, not by the shape"
    );
}

/// `CG-05`: allocation count and peak bytes. Zero here, whatever an oracle does.
#[test]
fn allocation_scaling_cg_05() {
    // This is a `build` claim elsewhere --- `CA-01` measures it with a counting
    // allocator. Here it is reported as the constant it is, because a scaling
    // report with a blank in this row would invite the reader to guess.
    eprintln!("CG-05 (open): uor-matmul allocates 0 times and 0 bytes, at every shape");
    eprintln!("CG-05 (open): the figure is structural (R7), and CA-01 observes it");
}

/// `CG-06`: parallel speedup against tile count, with byte-equality asserted
/// inside the timed harness.
///
/// The library spawns nothing, so what is measured is the partition: how the
/// work divides, and that dividing it does not change the answer. A speedup
/// number from a harness that did not check the bytes would be a measurement of
/// the wrong program.
#[test]
fn parallel_speedup_against_tile_count_cg_06() {
    use uor_matmul::prelude::*;
    use uor_matmul_core::{dot_ref, EncodeMode, Shape};
    use uor_matmul_gemm::Partition;

    let (m, k, n) = (48usize, 64usize, 48usize);
    let a: Vec<i8> = (0..m * k).map(|i| ((i * 17) % 255) as i8).collect();
    let b: Vec<i8> = (0..k * n).map(|i| ((i * 23) % 255) as i8).collect();
    let reference = uor_matmul_validate::ours_i8_i32(m, k, n, &a, &b, EncodeMode::Wrapping);

    for tiles in [1usize, 2, 4, 8, 16] {
        let side = (m / tiles).max(1);
        let part = Partition::new(Shape { m, k, n }, side, n);
        let started = std::time::Instant::now();
        let mut out = vec![0i32; m * n];
        for t in part {
            for i in t.row..t.row + t.rows {
                for j in t.col..t.col + t.cols {
                    let row: Vec<i8> = (0..k).map(|p| a[i * k + p]).collect();
                    let col: Vec<i8> = (0..k).map(|p| b[p * n + j]).collect();
                    out[i * n + j] = dot_ref(as_alphabet_full(&row), as_alphabet_full(&col)) as i32;
                }
            }
        }
        let elapsed = started.elapsed().as_secs_f64();
        // The assertion is inside the harness, not after it.
        assert_eq!(out, reference, "{tiles} tiles must give the same bytes");
        eprintln!(
            "CG-06 (open): {} tile(s), {elapsed:.6}s, bytes identical",
            part.len()
        );
    }
}

/// `CG-07`: small-shape latency, where a heavyweight prologue costs more than
/// an asymptote.
#[test]
fn small_shape_latency_cg_07() {
    for s in [1usize, 2, 4, 8, 16] {
        let a = vec![1i8; s * s];
        let b = vec![1i8; s * s];
        let mut out = vec![0i32; s * s];
        // Several repetitions, because one call at this size measures the clock.
        let started = std::time::Instant::now();
        for _ in 0..1000 {
            scaling::run_case(s, s, s, &a, &b, &mut out);
        }
        let per_call = started.elapsed().as_secs_f64() / 1000.0;
        eprintln!("CG-07 (open): {s}x{s}x{s} costs {:.9}s per call", per_call);
        assert!(
            out.iter().all(|&v| v == s as i32),
            "the timed call must be correct"
        );
    }
}

/// The apparatus is falsifiable: the fit recovers a known exponent, and fails
/// to fit what it should not.
#[test]
fn the_scaling_apparatus_is_falsifiable_cg_01() {
    let cubic: Vec<Observation> = (1..=20)
        .map(|i| Observation {
            x: i as f64,
            y: 7.0 * (i as f64).powi(3),
        })
        .collect();
    let f = scaling::fit(&cubic).expect("fits");
    assert!((f.exponent - 3.0).abs() < 1e-9, "exponent {}", f.exponent);
    assert!(
        scaling::fit(&cubic[..2]).is_none(),
        "two points must not fit"
    );
}
