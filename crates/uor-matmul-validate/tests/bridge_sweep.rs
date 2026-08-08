//! Historical `CG-15` instrument, retained to reproduce the workspace-spelling
//! comparison after removal of the float placement bridge.
//!
//! Both columns now delegate to the same pure-UOR Atlas-octet body. `default`
//! offers only decoded-code caches; `compatibility` calls
//! [`slice::gemm_float_full`] with every historical workspace parameter. The
//! latter's scaled, integer-panel, and accumulator buffers are untouched
//! compatibility channels: no operand is reified and no alternate arithmetic
//! can be selected (`CD-19`, `CA-05`).
//!
//! The exponent fills remain useful because they exercise different numbers of
//! finite Atlas pages and gauge-coherent groups. They no longer express an
//! admission boundary. Current float throughput, traffic, latency, confidence
//! intervals, and controls belong to `CG-21` (`just uor-float-sweep`); numbers
//! emitted here are a reproducibility record for the superseded `CG-15`
//! protocol.
//!
//! Every figure is `open`: printed, never asserted. What *is* asserted,
//! inside each timed run, is byte-identity between the two spellings --- a
//! speed measured on the wrong bytes is not a measurement.
//!
//! Ignored by default, like the other minute-long sweeps: `just
//! bridge-sweep` runs it, in release, where a throughput figure means
//! something.

use std::time::Instant;

use uor_matmul::{
    gemm_float_packed, suggested_accumulators, suggested_bridge_scaled, suggested_scratch,
    GemmOptions, Linear,
};
use uor_matmul_core::{MatView, MatViewMut, PackedCode, Shape, Triple};
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

/// The oracle sweep's wide fill. It exercises many Atlas grade pages rather
/// than an admission boundary: every finite span reaches the same total body.
fn wide(len: usize, salt: u64) -> Vec<f32> {
    fill(len, salt, |v| {
        (v % 2048) as f32 * 2.0f32.powi((v % 19) as i32 - 9)
    })
}

/// The original CG-15 shapes, retained so its historical records reproduce.
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
    let mut kernel_buf = vec![0i32; suggested_scratch(shape)];
    let mut acc_buf = vec![0i128; suggested_accumulators(shape)];

    // The default public spelling, with decoded-code caches only.
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
        uor_matmul::slice::gemm_float_full(
            m,
            k,
            n,
            a,
            b,
            &mut c_explicit,
            &mut pa,
            &mut pb,
            &mut scaled,
            &mut kernel_buf,
            &mut acc_buf,
        )
        .expect("the full float product exists");
        // Byte-identity between the two spellings, inside the timed region
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
#[ignore = "a historical release-mode workspace sweep: `just bridge-sweep`"]
fn the_historical_workspace_spellings_share_atlas_cg_15() {
    let gmacs = |m: usize, k: usize, n: usize, secs: f64| (m * k * n) as f64 / secs / 1e9;
    println!();
    println!(
        "# CG-15 (historical, superseded by CG-21): pure-Atlas workspace spellings, Gproduct/s"
    );
    println!(
        "# host: {}-{}; best of a 0.35 s budget per point;",
        std::env::consts::ARCH,
        std::env::consts::OS
    );
    println!("# byte-identity between the two spellings is asserted inside every timed run;");
    println!("# default offers decoded-code caches; compatibility also supplies the inert historical buffers;");
    println!("# every exponent span uses the same Atlas-octet operation family;");
    println!("# matrixmultiply is the inexact oracle, reported for scale (`CX-05` records its deviation)");
    for (label, span_a, span_b) in [("one exponent", 0, 0), ("a few binades", 3, 4)] {
        println!();
        println!(
            "| fill: {label} (spans {span_a}/{span_b}) | default | compatibility | matrixmultiply |"
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
    println!("| fill: wide spans (18/22 binades) | default | compatibility | matrixmultiply |");
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
