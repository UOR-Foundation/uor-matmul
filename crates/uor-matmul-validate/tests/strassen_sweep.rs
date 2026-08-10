//! `CG-12` (open): achieved MACs per second of the sub-cubic recursion against
//! the cubic packed walk on the `i32`-exact lane, swept through the crossover.
//!
//! The question the queue puts to this harness is whether the library's win
//! repertoire is bounded by quantized or structured data. The operands here are
//! arbitrary random dense `i32` at a declared 24-bit bound, and the recursion
//! does the same products, regrouped: `(7/8)^L` of them at `L` levels. If the
//! nominal rate (`m*k*n` per second) crosses the fastest sustained product
//! rate this library reaches on this host, the silicon performed fewer than
//! `m*k*n` products, which no `Theta(m*k*n)` implementation can do --- that
//! line, not anything involving bandwidth, is the bar.
//!
//! Two baselines, because the library has two cubic walks a caller can mean.
//! The *default* is the modular lane, which a wrapping `i32 -> i32` call
//! selects; the recursion does not serve that call. The *exact lane* is what
//! the recursion factorizes: it is what a saturating call or a wider output
//! runs. Both are drawn, and the recursion's columns are measured under
//! `EncodeMode::Saturating` so the comparison is within one lane.
//!
//! Every figure is `open`: printed, never asserted. What *is* asserted,
//! inside each timed run, is byte-identity with the cubic walk at the same
//! encode --- a speed measured on the wrong bytes is not a measurement.
//!
//! Ignored by default, like the other minute-long sweeps: `just
//! strassen-sweep` runs it, in release, where a throughput figure means
//! something.

use std::time::Instant;

use uor_matmul::prelude::*;
use uor_matmul::{driver::Scratch, gemm_strassen, strassen_levels, strassen_scratch, Shape};
use uor_matmul_validate::scaling::{fit, Observation};

/// The recorded seed; the operands are generated at runtime from it.
const SEED: u64 = 20_260_730;

/// The declared bound: 24-bit random dense data.
const BOUND: u128 = 1 << 24;

/// Deterministic dense fill, uniform in `[-2^23, 2^23)`, the same recorded
/// generator the other harnesses use.
fn fill(len: usize, salt: u64) -> Vec<i32> {
    let mut s = SEED ^ salt.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (((s >> 33) as i32) & 0x007F_FFFF) - 0x0040_0000
        })
        .collect()
}

/// Best of as many repetitions as fit a fixed budget, in seconds per call ---
/// the same discipline the other retained scaling sweeps use.
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

/// One product, `levels` requested recursion levels under `encode`, with the
/// offer the plan asks for. The call the timed closures make.
#[allow(clippy::too_many_arguments)]
fn call(
    n: usize,
    a: &[i32],
    b: &[i32],
    levels: usize,
    encode: EncodeMode,
    panel: &mut [Alphabet<i32, Bnd<{ 1 << 24 }>>],
    accs: &mut [i128],
    c: &mut [i32],
) {
    let av = MatView::row_major(
        as_alphabet::<i32, Bnd<{ 1 << 24 }>>(a).expect("the fill is 24-bit"),
        n,
        n,
    )
    .expect("A fits");
    let bv = MatView::row_major(
        as_alphabet::<i32, Bnd<{ 1 << 24 }>>(b).expect("the fill is 24-bit"),
        n,
        n,
    )
    .expect("B fits");
    let cv = MatViewMut::row_major(c, n, n).expect("C fits");
    let mut t = Triple::new(av, bv, cv).expect("the product exists");
    gemm_strassen(
        &mut t,
        &Linear::OVERWRITE,
        GemmOptions {
            encode,
            ..Default::default()
        },
        &mut Scratch::with_accumulators(panel, accs),
        levels,
    );
}

/// One timed column, byte-asserted against `reference` on every repetition.
#[allow(clippy::too_many_arguments)]
fn timed(
    n: usize,
    a: &[i32],
    b: &[i32],
    levels: usize,
    encode: EncodeMode,
    reference: &[i32],
) -> f64 {
    let (panels, accs) = strassen_scratch(Shape { m: n, k: n, n }, levels);
    let mut panel = vec![Alphabet::<i32, Bnd<{ 1 << 24 }>>::ZERO; panels];
    let mut acc_buf = vec![0i128; accs];
    let mut c = vec![0i32; n * n];
    best(|| {
        call(n, a, b, levels, encode, &mut panel, &mut acc_buf, &mut c);
        assert_eq!(c, reference, "the timed call must give the cubic bytes");
    })
}

/// The fastest sustained product rate this library reaches on this host: the
/// `i8` lane, whose kernels are the table's densest, at a shape past every
/// cache. Stated as a method: measured, in this harness, not derived from a
/// data sheet.
fn peak_macs() -> f64 {
    let n = 4096usize;
    let a = fill(n * n, 77)
        .into_iter()
        .map(|x| x as i8)
        .collect::<Vec<_>>();
    let b = fill(n * n, 78)
        .into_iter()
        .map(|x| x as i8)
        .collect::<Vec<_>>();
    let mut panel = vec![Alphabet::<i8, Full<i8>>::ZERO; 1 << 22];
    let mut c = vec![0i32; n * n];
    let t = best(|| {
        let av = MatView::row_major(as_alphabet_full(&a), n, n).expect("A fits");
        let bv = MatView::row_major(as_alphabet_full(&b), n, n).expect("B fits");
        let cv = MatViewMut::row_major(&mut c, n, n).expect("C fits");
        let mut t = Triple::new(av, bv, cv).expect("the product exists");
        uor_matmul::gemm_packed(
            &mut t,
            &Linear::OVERWRITE,
            GemmOptions {
                encode: EncodeMode::Wrapping,
                ..Default::default()
            },
            &mut Scratch::new(&mut panel),
        );
    });
    (n * n * n) as f64 / t
}

/// `CG-12`: the sweep, the fit, and the peak-rate line.
#[test]
#[ignore = "minutes; just strassen-sweep runs it in release"]
fn the_recursions_exponent_against_the_cubic_walk_cg_12() {
    // Geometric spacing, every extent divisible by eight so three levels are
    // even-divisible at every point.
    let sizes = [256usize, 384, 512, 768, 1024, 1536, 2048, 3072, 4096];
    let peak = peak_macs();
    println!("# CG-12: every figure is `open`: measured, never asserted");
    println!("# seed {SEED}, declared bound 2^24 random dense i32");
    println!(
        "# fastest sustained product rate on this host (i8 lane, measured here): {:.3} Gmac/s",
        peak / 1e9
    );
    println!();
    println!(
        "{:>6} {:>14} {:>12} {:>12} {:>12} {:>12} {:>10}",
        "n", "default (mod)", "exact cubic", "L=1", "L=2", "L=3", "levels"
    );
    let mut cubic_obs = Vec::new();
    let mut rec_obs = Vec::new();
    let mut cubic_obs_n = Vec::new();
    let mut rec_obs_n = Vec::new();
    for &n in &sizes {
        let a = fill(n * n, 11);
        let b = fill(n * n, 12);
        // The references, computed once per encode mode. Every timed run is
        // byte-asserted against one of the two inside its timed region.
        let ref_sat = {
            let (panels, accs) = strassen_scratch(Shape { m: n, k: n, n }, 0);
            let mut panel = vec![Alphabet::<i32, Bnd<{ 1 << 24 }>>::ZERO; panels];
            let mut acc_buf = vec![0i128; accs];
            let mut c = vec![0i32; n * n];
            call(
                n,
                &a,
                &b,
                0,
                EncodeMode::Saturating,
                &mut panel,
                &mut acc_buf,
                &mut c,
            );
            c
        };
        let ref_wrap = {
            let mut panel = vec![Alphabet::<i32, Bnd<{ 1 << 24 }>>::ZERO; 1 << 22];
            let mut acc_buf = vec![];
            let mut c = vec![0i32; n * n];
            call(
                n,
                &a,
                &b,
                0,
                EncodeMode::Wrapping,
                &mut panel,
                &mut acc_buf,
                &mut c,
            );
            c
        };
        let shape = Shape { m: n, k: n, n };
        let t_default = timed(n, &a, &b, 0, EncodeMode::Wrapping, &ref_wrap);
        let t0 = timed(n, &a, &b, 0, EncodeMode::Saturating, &ref_sat);
        let t1 = timed(n, &a, &b, 1, EncodeMode::Saturating, &ref_sat);
        let t2 = timed(n, &a, &b, 2, EncodeMode::Saturating, &ref_sat);
        let t3 = timed(n, &a, &b, 3, EncodeMode::Saturating, &ref_sat);
        let (p, q) = strassen_scratch(shape, 3);
        let taken = strassen_levels::<i32>(shape, BOUND, p, q, usize::MAX);
        let gmacs = (n as f64).powi(3) / 1e9;
        println!(
            "{:>6} {:>14.3} {:>12.3} {:>12.3} {:>12.3} {:>12.3} {:>10}",
            n,
            gmacs / t_default,
            gmacs / t0,
            gmacs / t1,
            gmacs / t2,
            gmacs / t3,
            taken
        );
        let macs = (n * n * n) as f64;
        cubic_obs.push(Observation { x: macs, y: t0 });
        cubic_obs_n.push(Observation { x: n as f64, y: t0 });
        // The recursive fit is over the levels the plan takes at the full
        // offer --- the auto-selected path, not a fixed depth.
        let t_auto = match taken {
            1 => t1,
            2 => t2,
            3 => t3,
            _ => t0,
        };
        rec_obs.push(Observation { x: macs, y: t_auto });
        rec_obs_n.push(Observation {
            x: n as f64,
            y: t_auto,
        });
    }
    println!();
    for (name, obs) in [
        ("exact cubic, time vs MACs", &cubic_obs),
        ("recursion (auto levels), time vs MACs", &rec_obs),
        ("exact cubic, time vs n", &cubic_obs_n),
        ("recursion (auto levels), time vs n", &rec_obs_n),
    ] {
        if let Some(f) = fit(obs) {
            println!(
                "{name}: exponent {:.4} +/- {:.4} (95%), {} samples, residual {:.4}",
                f.exponent, f.confidence_half_width, f.samples, f.rms_residual
            );
        }
    }
}
