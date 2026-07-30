//! `CG-15` (open): achieved MACs per second of the float placement bridge,
//! through the default driver's auto-selection and through the explicit
//! entry's richer offers.
//!
//! The bridge reifies the float driver's scaled panels as an `i32` alphabet
//! and hands the reduction to the integer kernel table, placing the table's
//! exact sum at the one known scale (`CD-19` pins the bytes). The default
//! driver takes the same lane when its panel offer re-reads as the room the
//! bridge needs, so the two columns are: `default`, the packed entry at the
//! panel offer every caller has (`k` and `k * n` codes), and `explicit`,
//! [`gemm_float_bridged`] at its full named offers, kernel scratch and
//! accumulators included. Which factorization each column runs is the
//! fill's business and the header's: at spans the alphabet admits with a
//! depth the lane holds, both are the table; past the lane's depth the
//! default declines to the scalar scaled lanes --- the chunked traversal
//! keeps its partial sums in the accumulator room only the explicit offer
//! spells --- and at spans past the alphabet both are the scalar lanes,
//! which is the wide fill's row.
//!
//! What is left to ask is the economics, and either answer is the finding:
//! the prediction recorded in ANALYSIS.md before this harness ran is
//! single-digit Gmac/s on the AVX2 runner --- four `i64` lanes against eight
//! `f32` FMA lanes --- and a lesser, auto-vectorization-dependent figure on
//! this host, where the `i32`-exact family has no hand-written NEON
//! sequence.
//!
//! Every figure is `open`: printed, never asserted. What *is* asserted,
//! inside each timed run, is byte-identity between the two columns --- a
//! speed measured on the wrong bytes is not a measurement.
//!
//! Ignored by default, like the other minute-long sweeps: `just
//! bridge-sweep` runs it, in release, where a throughput figure means
//! something.

use std::time::Instant;

use uor_matmul::{
    gemm_float_bridged, gemm_float_packed, suggested_accumulators, suggested_bridge_scaled,
    suggested_scratch, GemmOptions, Linear, Scratch,
};
use uor_matmul_core::{Alphabet, Full, MatView, MatViewMut, PackedCode, Shape, Triple};
use uor_matmul_validate::oracle::{FloatOracle, MatrixMultiply};

/// Deterministic fill, the same recorded generator the other harnesses use.
fn fill<T, F: Fn(u64) -> T>(len: usize, salt: u64, map: F) -> Vec<T> {
    let mut s = 0x243F_6A88_85A3_08D3u64 ^ salt;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            map(s >> 33)
        })
        .collect()
}

/// Best of as many repetitions as fit a fixed budget, in seconds per call.
///
/// The minimum is the run least interfered with; the budget is wall-clock so a
/// fast point and a slow one both get enough samples for the minimum to mean
/// something. The same discipline `oracle_sweep` uses.
fn best(mut run: impl FnMut()) -> f64 {
    const BUDGET: f64 = 0.35;
    let mut best = f64::INFINITY;
    let mut spent = 0.0;
    loop {
        let s = Instant::now();
        run();
        let t = s.elapsed().as_secs_f64();
        best = best.min(t);
        spent += t;
        if spent >= BUDGET {
            return best;
        }
    }
}

/// A fill whose decoded exponent span is exactly `span` binades: significands
/// drawn from `[2^23, 2^24)`, so the decode does not add a bit of its own.
fn spanned(len: usize, salt: u64, span: i32) -> Vec<f32> {
    fill(len, salt, |v| {
        let s = if span == 0 {
            0
        } else {
            (v as i64 % span as i64) as i32
        };
        let m = 8_388_608 + v % 8_388_607;
        let x = m as f32 * 2.0f32.powi(s - span / 2);
        if v & 1 == 0 {
            -x
        } else {
            x
        }
    })
}

/// The oracle sweep's fill: eighteen and twenty-two binades of span, which no
/// scaled lane admits. The bridge declines it; the row is here so the report
/// shows the boundary rather than silently measuring only the admitted case.
fn wide(len: usize, salt: u64) -> Vec<f32> {
    fill(len, salt, |v| {
        (v % 2048) as f32 * 2.0f32.powi((v % 19) as i32 - 9)
    })
}

/// The shapes ANALYSIS tables for the float placement: the two cubes and the
/// one awkward product, plus the small end where latency dominates.
const SHAPES: &[(usize, usize, usize)] = &[
    (32, 32, 32),
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (509, 1021, 257),
];

/// One timed path on one shape, in calls per second's reciprocal.
struct Row {
    default: f64,
    explicit: f64,
    oracle: f64,
}

fn measure(m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Row {
    let shape = Shape { m, k, n };
    let mut c_default = vec![0.0f32; m * n];
    let mut c_explicit = vec![0.0f32; m * n];

    let mut pa = vec![PackedCode::default(); k.max(1)];
    let mut pb = vec![PackedCode::default(); k * n];
    let mut scaled = vec![0i32; suggested_bridge_scaled(shape)];
    let mut kernel_buf = vec![Alphabet::<i32, Full<i32>>::of(0); suggested_scratch(shape)];
    let mut acc_buf = vec![0i128; suggested_accumulators(shape)];

    // The default driver at the panel offer every caller has. Where the offer
    // and the spans admit, this *is* the table lane --- that is the point of
    // the auto-selection --- and where they do not, it is the scalar lanes.
    let default = best(|| {
        let av = MatView::row_major(a, m, k).unwrap();
        let bv = MatView::row_major(b, k, n).unwrap();
        let cv = MatViewMut::row_major(&mut c_default, m, n).unwrap();
        let mut t = Triple::new(av, bv, cv).unwrap();
        gemm_float_packed(
            &mut t,
            &Linear::OVERWRITE,
            GemmOptions::default(),
            &mut pa,
            &mut pb,
        );
    });

    let explicit = best(|| {
        let av = MatView::row_major(a, m, k).unwrap();
        let bv = MatView::row_major(b, k, n).unwrap();
        let cv = MatViewMut::row_major(&mut c_explicit, m, n).unwrap();
        let mut t = Triple::new(av, bv, cv).unwrap();
        gemm_float_bridged(
            &mut t,
            &Linear::OVERWRITE,
            GemmOptions::default(),
            &mut pa,
            &mut pb,
            &mut scaled,
            &mut Scratch::with_accumulators(&mut kernel_buf, &mut acc_buf),
        );
        // Byte-identity between the two entries, inside the timed region
        // (`CD-19`): a speed measured on the wrong bytes is not a measurement.
        assert_eq!(
            c_explicit.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            c_default.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "the two entries must give the same bytes at {m}x{k}x{n}"
        );
    });

    let oracle = best(|| {
        let c = MatrixMultiply::product_f32(m, k, n, a, b);
        std::hint::black_box(&c);
    });

    Row {
        default,
        explicit,
        oracle,
    }
}

#[test]
#[ignore = "a minutes-long release-mode sweep: `just bridge-sweep`"]
fn the_bridge_against_the_scalar_lanes_cg_15() {
    let gmacs = |m: usize, k: usize, n: usize, secs: f64| (m * k * n) as f64 / secs / 1e9;
    println!();
    println!("# CG-15 (open): the float placement bridge, default auto-selection against the explicit entry, Gmac/s");
    println!(
        "# host: {}-{}; best of a 0.35 s budget per point;",
        std::env::consts::ARCH,
        std::env::consts::OS
    );
    println!("# byte-identity between the two entries is asserted inside every timed run;");
    println!("# at admitted spans with a depth the lane holds, the default column is the table;");
    println!("# past the lane's depth it declines to the scalar scaled lanes;");
    println!("# matrixmultiply is the inexact oracle, reported for scale (`CX-05` records its deviation)");
    for (label, span_a, span_b) in [("one exponent", 0, 0), ("a few binades", 3, 4)] {
        println!();
        println!(
            "| fill: {label} (spans {span_a}/{span_b}) | default | explicit | matrixmultiply |"
        );
        println!("| --- | --- | --- | --- |");
        for &(m, k, n) in SHAPES {
            let a = spanned(m * k, 5, span_a);
            let b = spanned(k * n, 6, span_b);
            let r = measure(m, k, n, &a, &b);
            println!(
                "| {m}x{k}x{n} | {:.3} | {:.3} | {:.3} |",
                gmacs(m, k, n, r.default),
                gmacs(m, k, n, r.explicit),
                gmacs(m, k, n, r.oracle),
            );
        }
    }
    println!();
    println!("| fill: wide spans (18/22 binades, both columns decline to the scalar lanes) | default | explicit | matrixmultiply |");
    println!("| --- | --- | --- | --- |");
    for &(m, k, n) in SHAPES {
        let a = wide(m * k, 5);
        let b = wide(k * n, 6);
        let r = measure(m, k, n, &a, &b);
        println!(
            "| {m}x{k}x{n} | {:.3} | {:.3} | {:.3} |",
            gmacs(m, k, n, r.default),
            gmacs(m, k, n, r.explicit),
            gmacs(m, k, n, r.oracle),
        );
    }
}
