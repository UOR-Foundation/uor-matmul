//! `CG-21` (open): pure-UOR float throughput, caller-visible traffic, and
//! latency against the retained exact reference and two external float
//! oracles, with integer and tropical controls in the same run.
//!
//! The measurement grid is deliberately small enough that the unoptimized
//! exact reference computes every expected output byte.  This bounds the V&V
//! run, never the implementation: no dimension below is visible to a shipped
//! crate. Every calibration and sample batch starts from an output code derived
//! from, and asserted distinct from, that cell's expected code, then compares
//! the complete result after the batch. Views, conformant triples,
//! offer wrappers, and external-adapter copies are also prepared outside the
//! timer: the interval contains only repeated calls to the production operation
//! being compared.  Thus a missing write, an elided call, a wrong answer, or
//! benchmark setup accidentally priced as throughput is a hard failure.
//!
//! The intervals are two-sided 95% Student intervals over nine independently
//! timed batches.  Batching is calibrated per route to a target duration so a
//! nanosecond-scale oracle and a slow exact reference both receive useful clock
//! resolution.  The reported traffic is explicitly the logical lower bound
//! `A read + B read + C write`; guard traffic and implementation-private panel
//! traffic are named separately rather than passed off as hardware counters.
//! Every aggregate is accompanied by a machine-readable raw batch duration.

#[cfg(all(feature = "ref-matrixmultiply", feature = "ref-faer"))]
use core::mem::size_of;
#[cfg(all(feature = "ref-matrixmultiply", feature = "ref-faer"))]
use std::time::{Duration, Instant};

#[cfg(all(feature = "ref-matrixmultiply", feature = "ref-faer"))]
use uor_matmul::prelude::*;
#[cfg(all(feature = "ref-matrixmultiply", feature = "ref-faer"))]
use uor_matmul::{suggested_float_panels, suggested_scratch, Shape};
#[cfg(all(feature = "ref-matrixmultiply", feature = "ref-faer"))]
use uor_matmul_core::{
    as_alphabet_full, as_alphabet_tropical, as_alphabet_whole, AccOf, Alphabet, EncodeFrom,
    EncodeMode, Full, PackedCode, Trop,
};
#[cfg(all(feature = "ref-matrixmultiply", feature = "ref-faer"))]
use uor_matmul_gemm::epilogue::{AbsorbPrior, ScaleExact};
#[cfg(all(feature = "ref-matrixmultiply", feature = "ref-faer"))]
use uor_matmul_gemm::{PlaceAt, SignedPlace};
#[cfg(all(feature = "ref-matrixmultiply", feature = "ref-faer"))]
use uor_matmul_validate::float_corpus::{
    exact_product, operands, CorpusFloat, FloatCase, PERFORMANCE_CASES,
};

#[cfg(all(feature = "ref-matrixmultiply", feature = "ref-faer"))]
const SAMPLE_COUNT: usize = 9;
#[cfg(all(feature = "ref-matrixmultiply", feature = "ref-faer"))]
const SAMPLE_TARGET: Duration = Duration::from_millis(4);
// Student's t at 95%, two-sided, for SAMPLE_COUNT - 1 = 8 degrees of freedom.
#[cfg(all(feature = "ref-matrixmultiply", feature = "ref-faer"))]
const T95_DF8: f64 = 2.306_004_135_204_166;

fn require_measurement_build() {
    if cfg!(debug_assertions) {
        panic!("CG-21 is a performance measurement: run `just uor-float-sweep`");
    }
    #[cfg(not(all(feature = "ref-matrixmultiply", feature = "ref-faer")))]
    panic!("CG-21 requires both independent float oracles; run `just uor-float-sweep`");
}

#[cfg(all(feature = "ref-matrixmultiply", feature = "ref-faer"))]
#[derive(Clone, Copy)]
struct Estimate {
    mean: f64,
    half_width: f64,
}

#[cfg(all(feature = "ref-matrixmultiply", feature = "ref-faer"))]
impl Estimate {
    fn from_values(values: impl IntoIterator<Item = f64>) -> Self {
        let values: Vec<f64> = values.into_iter().collect();
        assert_eq!(values.len(), SAMPLE_COUNT);
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let sum_squares = values
            .iter()
            .map(|value| {
                let residual = value - mean;
                residual * residual
            })
            .sum::<f64>();
        let sample_variance = sum_squares / (n - 1.0);
        Self {
            mean,
            half_width: T95_DF8 * (sample_variance / n).sqrt(),
        }
    }
}

#[cfg(all(feature = "ref-matrixmultiply", feature = "ref-faer"))]
struct Metrics {
    latency_us: Estimate,
    throughput_gops: Estimate,
    caller_gbytes: Estimate,
    batch: usize,
    elapsed_ns: [u128; SAMPLE_COUNT],
}

/// Measure seconds per invocation in calibrated batches.  The calibration
/// derives a repetition count from time, so it cannot become a shape or data
/// admission limit. `measured_batch` owns the clock boundary because it is the
/// only place that can construct a route's borrowed triple before that boundary
/// and release it before checking the caller's output afterward.
#[cfg(all(feature = "ref-matrixmultiply", feature = "ref-faer"))]
fn samples(mut measured_batch: impl FnMut(usize) -> Duration) -> ([Duration; SAMPLE_COUNT], usize) {
    measured_batch(1);
    let pilot = measured_batch(1);
    let batch = SAMPLE_TARGET
        .as_nanos()
        .checked_div(pilot.as_nanos().max(1))
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(1)
        .max(1);

    let mut elapsed = [Duration::ZERO; SAMPLE_COUNT];
    for sample in &mut elapsed {
        *sample = measured_batch(batch);
    }
    (elapsed, batch)
}

#[cfg(all(feature = "ref-matrixmultiply", feature = "ref-faer"))]
fn metrics(
    macs: usize,
    caller_bytes: usize,
    measured_batch: impl FnMut(usize) -> Duration,
) -> Metrics {
    let (elapsed, batch) = samples(measured_batch);
    let seconds = elapsed.map(|elapsed| elapsed.as_secs_f64() / batch as f64);
    Metrics {
        latency_us: Estimate::from_values(seconds.iter().map(|seconds| seconds * 1e6)),
        throughput_gops: Estimate::from_values(
            seconds.iter().map(|seconds| macs as f64 / seconds / 1e9),
        ),
        caller_gbytes: Estimate::from_values(
            seconds
                .iter()
                .map(|seconds| caller_bytes as f64 / seconds / 1e9),
        ),
        batch,
        elapsed_ns: elapsed.map(|elapsed| elapsed.as_nanos()),
    }
}

#[cfg(all(feature = "ref-matrixmultiply", feature = "ref-faer"))]
#[allow(clippy::too_many_arguments)]
fn print_metrics(
    width: &str,
    case: &str,
    shape: Shape,
    fill: &str,
    route_id: &str,
    route: &str,
    measured: &Metrics,
    deviation: &str,
) {
    for (round, elapsed_ns) in measured.elapsed_ns.iter().enumerate() {
        eprintln!(
            "CG21_SAMPLE phase=public width={width} case={case} m={} k={} n={} fill={fill} route={route_id} round={round} batch={} elapsed_ns={elapsed_ns}",
            shape.m,
            shape.k,
            shape.n,
            measured.batch,
        );
    }
    eprintln!(
        "| {route} | {:.3} +/- {:.3} | {:.4} +/- {:.4} | {:.4} +/- {:.4} | {} | {deviation} |",
        measured.latency_us.mean,
        measured.latency_us.half_width,
        measured.throughput_gops.mean,
        measured.throughput_gops.half_width,
        measured.caller_gbytes.mean,
        measured.caller_gbytes.half_width,
        measured.batch,
    );
}

#[cfg(all(feature = "ref-matrixmultiply", feature = "ref-faer"))]
fn assert_bits<E: CorpusFloat>(got: &[E], want: &[E]) {
    assert_eq!(got.len(), want.len());
    for (at, (&got, &want)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            got.corpus_bits(),
            want.corpus_bits(),
            "timed output differs at {at}"
        );
    }
}

#[cfg(all(feature = "ref-matrixmultiply", feature = "ref-faer"))]
fn poison_from_expected<E: CorpusFloat>(expected: E) -> E {
    let poisoned = E::from_corpus_bits(expected.corpus_bits() ^ 1);
    assert_ne!(
        poisoned.corpus_bits(),
        expected.corpus_bits(),
        "CG-21 poison must differ from the expected output code"
    );
    poisoned
}

#[cfg(all(feature = "ref-matrixmultiply", feature = "ref-faer"))]
fn poison_output<E: CorpusFloat>(out: &mut [E], expected: &[E]) {
    assert_eq!(out.len(), expected.len(), "CG-21 poison covers every cell");
    for (out, &expected) in out.iter_mut().zip(expected) {
        *out = poison_from_expected(expected);
    }
}

#[cfg(all(feature = "ref-matrixmultiply", feature = "ref-faer"))]
fn ordered_float_code<E: CorpusFloat>(value: E) -> Option<u128> {
    let exponent_mask = (1u64 << E::EXPONENT_BITS) - 1;
    let bits = value.corpus_bits();
    let exponent = (bits >> E::FRACTION_BITS) & exponent_mask;
    let fraction_mask = (1u64 << E::FRACTION_BITS) - 1;
    if exponent == exponent_mask && bits & fraction_mask != 0 {
        return None;
    }
    let sign = 1u64 << (E::FRACTION_BITS + E::EXPONENT_BITS);
    let value_mask = sign | (sign - 1);
    Some(if bits & sign == 0 {
        u128::from(sign) + u128::from(bits)
    } else {
        u128::from((!bits) & value_mask)
    })
}

#[cfg(all(feature = "ref-matrixmultiply", feature = "ref-faer"))]
fn deviation<E: CorpusFloat>(exact: &[E], approximate: &[E]) -> String {
    let mut different = 0usize;
    let mut incomparable = 0usize;
    let mut max_ulps = 0u128;
    for (&exact, &approximate) in exact.iter().zip(approximate) {
        if exact.corpus_bits() == approximate.corpus_bits() {
            continue;
        }
        different += 1;
        match (ordered_float_code(exact), ordered_float_code(approximate)) {
            (Some(exact), Some(approximate)) => {
                max_ulps = max_ulps.max(exact.abs_diff(approximate));
            }
            _ => incomparable += 1,
        }
    }
    if incomparable == 0 {
        format!(
            "{different}/{} result codes differ; max {max_ulps} ulp",
            exact.len()
        )
    } else {
        format!(
            "{different}/{} result codes differ; {incomparable} non-ordered; finite max {max_ulps} ulp",
            exact.len()
        )
    }
}

#[cfg(all(feature = "ref-matrixmultiply", feature = "ref-faer"))]
fn logical_float_bytes<E>(case: FloatCase) -> usize {
    (case.m * case.k + case.k * case.n + case.m * case.n) * size_of::<E>()
}

#[cfg(all(feature = "ref-matrixmultiply", feature = "ref-faer"))]
fn guard_float_bytes<E>(case: FloatCase) -> usize {
    // One poison write and one byte-identity read around each timed batch.
    2 * case.m * case.n * size_of::<E>()
}

#[cfg(all(feature = "ref-matrixmultiply", feature = "ref-faer"))]
trait MeasuredFloat: CorpusFloat {
    type FaerState;

    const LABEL: &'static str;

    fn matrixmultiply(case: FloatCase, a: &[Self], b: &[Self], out: &mut [Self]);
    fn prepare_faer(case: FloatCase, a: &[Self], b: &[Self]) -> Self::FaerState;
    fn poison_faer_output(state: &mut Self::FaerState, case: FloatCase, expected: &[Self]);
    fn faer_compute(state: &mut Self::FaerState);
    fn copy_faer_output(state: &Self::FaerState, case: FloatCase, out: &mut [Self]);
}

#[cfg(all(feature = "ref-matrixmultiply", feature = "ref-faer"))]
struct FaerF32 {
    a: faer::Mat<f32>,
    b: faer::Mat<f32>,
    c: faer::Mat<f32>,
}

#[cfg(all(feature = "ref-matrixmultiply", feature = "ref-faer"))]
struct FaerF64 {
    a: faer::Mat<f64>,
    b: faer::Mat<f64>,
    c: faer::Mat<f64>,
}

#[cfg(all(feature = "ref-matrixmultiply", feature = "ref-faer"))]
impl MeasuredFloat for f32 {
    type FaerState = FaerF32;

    const LABEL: &'static str = "f32";

    fn matrixmultiply(case: FloatCase, a: &[Self], b: &[Self], out: &mut [Self]) {
        // SAFETY: the buffers have the row-major extents and strides supplied
        // here, and this validation-only adapter owns the output exclusively.
        unsafe {
            matrixmultiply::sgemm(
                case.m,
                case.k,
                case.n,
                1.0,
                a.as_ptr(),
                case.k as isize,
                1,
                b.as_ptr(),
                case.n as isize,
                1,
                0.0,
                out.as_mut_ptr(),
                case.n as isize,
                1,
            );
        }
    }

    fn prepare_faer(case: FloatCase, a: &[Self], b: &[Self]) -> Self::FaerState {
        FaerF32 {
            a: faer::Mat::from_fn(case.m, case.k, |i, p| a[i * case.k + p]),
            b: faer::Mat::from_fn(case.k, case.n, |p, j| b[p * case.n + j]),
            c: faer::Mat::zeros(case.m, case.n),
        }
    }

    fn poison_faer_output(state: &mut Self::FaerState, case: FloatCase, expected: &[Self]) {
        assert_eq!(expected.len(), case.m * case.n);
        for i in 0..case.m {
            for j in 0..case.n {
                state.c[(i, j)] = poison_from_expected(expected[i * case.n + j]);
            }
        }
    }

    fn faer_compute(state: &mut Self::FaerState) {
        faer::linalg::matmul::matmul(
            state.c.as_mut(),
            faer::Accum::Replace,
            state.a.as_ref(),
            state.b.as_ref(),
            1.0,
            faer::Par::Seq,
        );
    }

    fn copy_faer_output(state: &Self::FaerState, case: FloatCase, out: &mut [Self]) {
        for i in 0..case.m {
            for j in 0..case.n {
                out[i * case.n + j] = state.c[(i, j)];
            }
        }
    }
}

#[cfg(all(feature = "ref-matrixmultiply", feature = "ref-faer"))]
impl MeasuredFloat for f64 {
    type FaerState = FaerF64;

    const LABEL: &'static str = "f64";

    fn matrixmultiply(case: FloatCase, a: &[Self], b: &[Self], out: &mut [Self]) {
        // SAFETY: as in the f32 adapter above.
        unsafe {
            matrixmultiply::dgemm(
                case.m,
                case.k,
                case.n,
                1.0,
                a.as_ptr(),
                case.k as isize,
                1,
                b.as_ptr(),
                case.n as isize,
                1,
                0.0,
                out.as_mut_ptr(),
                case.n as isize,
                1,
            );
        }
    }

    fn prepare_faer(case: FloatCase, a: &[Self], b: &[Self]) -> Self::FaerState {
        FaerF64 {
            a: faer::Mat::from_fn(case.m, case.k, |i, p| a[i * case.k + p]),
            b: faer::Mat::from_fn(case.k, case.n, |p, j| b[p * case.n + j]),
            c: faer::Mat::zeros(case.m, case.n),
        }
    }

    fn poison_faer_output(state: &mut Self::FaerState, case: FloatCase, expected: &[Self]) {
        assert_eq!(expected.len(), case.m * case.n);
        for i in 0..case.m {
            for j in 0..case.n {
                state.c[(i, j)] = poison_from_expected(expected[i * case.n + j]);
            }
        }
    }

    fn faer_compute(state: &mut Self::FaerState) {
        faer::linalg::matmul::matmul(
            state.c.as_mut(),
            faer::Accum::Replace,
            state.a.as_ref(),
            state.b.as_ref(),
            1.0,
            faer::Par::Seq,
        );
    }

    fn copy_faer_output(state: &Self::FaerState, case: FloatCase, out: &mut [Self]) {
        for i in 0..case.m {
            for j in 0..case.n {
                out[i * case.n + j] = state.c[(i, j)];
            }
        }
    }
}

/// One offered production batch. All public boundary objects are live before
/// the clock starts; dropping the triple at the block boundary releases `out`
/// for the complete check that follows it.
#[cfg(all(feature = "ref-matrixmultiply", feature = "ref-faer"))]
#[allow(clippy::too_many_arguments)]
fn timed_float_packed_batch<E>(
    case: FloatCase,
    a: &[E],
    b: &[E],
    out: &mut [E],
    expected: &[E],
    pa: &mut [PackedCode],
    pb: &mut [PackedCode],
    repetitions: usize,
) -> Duration
where
    E: MeasuredFloat + EncodeFrom<AccOf<E>> + EncodeFrom<i128>,
    AccOf<E>: SignedPlace + PlaceAt + ScaleExact + AbsorbPrior<E>,
{
    poison_output(out, expected);
    let elapsed = {
        let av = MatView::row_major(a, case.m, case.k).expect("A fits");
        let bv = MatView::row_major(b, case.k, case.n).expect("B fits");
        let cv = MatViewMut::row_major(out, case.m, case.n).expect("C fits");
        let mut triple = Triple::new(av, bv, cv).expect("the product exists");
        let options = GemmOptions::default();
        let start = Instant::now();
        for _ in 0..repetitions {
            uor_matmul::gemm_float_packed(&mut triple, &Linear::OVERWRITE, options, pa, pb);
        }
        start.elapsed()
    };
    assert_bits(out, expected);
    elapsed
}

#[cfg(all(feature = "ref-matrixmultiply", feature = "ref-faer"))]
fn timed_float_no_offer_batch<E>(
    case: FloatCase,
    a: &[E],
    b: &[E],
    out: &mut [E],
    expected: &[E],
    repetitions: usize,
) -> Duration
where
    E: MeasuredFloat + EncodeFrom<AccOf<E>> + EncodeFrom<i128>,
    AccOf<E>: SignedPlace + PlaceAt + ScaleExact + AbsorbPrior<E>,
{
    poison_output(out, expected);
    let elapsed = {
        let av = MatView::row_major(a, case.m, case.k).expect("A fits");
        let bv = MatView::row_major(b, case.k, case.n).expect("B fits");
        let cv = MatViewMut::row_major(out, case.m, case.n).expect("C fits");
        let mut triple = Triple::new(av, bv, cv).expect("the product exists");
        let options = GemmOptions::default();
        let start = Instant::now();
        for _ in 0..repetitions {
            uor_matmul::gemm_float(&mut triple, &Linear::OVERWRITE, options);
        }
        start.elapsed()
    };
    assert_bits(out, expected);
    elapsed
}

#[cfg(all(feature = "ref-matrixmultiply", feature = "ref-faer"))]
fn timed_incumbent_batch<E>(
    case: FloatCase,
    a: &[E],
    b: &[E],
    out: &mut [E],
    expected: &[E],
    repetitions: usize,
) -> Duration
where
    E: MeasuredFloat + EncodeFrom<AccOf<E>> + EncodeFrom<i128>,
    AccOf<E>: SignedPlace + PlaceAt + ScaleExact + AbsorbPrior<E>,
{
    poison_output(out, expected);
    let elapsed = {
        let av = MatView::row_major(as_alphabet_whole(a), case.m, case.k).expect("A fits");
        let bv = MatView::row_major(as_alphabet_whole(b), case.k, case.n).expect("B fits");
        let cv = MatViewMut::row_major(out, case.m, case.n).expect("C fits");
        let mut triple = Triple::new(av, bv, cv).expect("the product exists");
        let mut scratch = Scratch::none();
        let options = GemmOptions::default();
        let start = Instant::now();
        for _ in 0..repetitions {
            uor_matmul_gemm::gemm(&mut triple, &Linear::OVERWRITE, options, &mut scratch);
        }
        start.elapsed()
    };
    assert_bits(out, expected);
    elapsed
}

#[cfg(all(feature = "ref-matrixmultiply", feature = "ref-faer"))]
fn timed_matrixmultiply_batch<E: MeasuredFloat>(
    case: FloatCase,
    a: &[E],
    b: &[E],
    out: &mut [E],
    expected: &[E],
    repetitions: usize,
) -> Duration {
    poison_output(out, expected);
    let start = Instant::now();
    for _ in 0..repetitions {
        E::matrixmultiply(case, a, b, out);
    }
    let elapsed = start.elapsed();
    assert_bits(out, expected);
    elapsed
}

#[cfg(all(feature = "ref-matrixmultiply", feature = "ref-faer"))]
fn timed_faer_batch<E: MeasuredFloat>(
    case: FloatCase,
    state: &mut E::FaerState,
    out: &mut [E],
    expected: &[E],
    repetitions: usize,
) -> Duration {
    poison_output(out, expected);
    E::poison_faer_output(state, case, expected);
    let start = Instant::now();
    for _ in 0..repetitions {
        E::faer_compute(state);
    }
    let elapsed = start.elapsed();
    E::copy_faer_output(state, case, out);
    assert_bits(out, expected);
    elapsed
}

#[cfg(all(feature = "ref-matrixmultiply", feature = "ref-faer"))]
#[allow(clippy::too_many_arguments)]
fn timed_integer_batch(
    case: FloatCase,
    a: &[i8],
    b: &[i8],
    panels: &mut [Alphabet<i8, Full<i8>>],
    out: &mut [i32],
    expected: &[i32],
    repetitions: usize,
) -> Duration {
    assert_eq!(out.len(), expected.len());
    for (out, &expected) in out.iter_mut().zip(expected) {
        *out = expected ^ 1;
        assert_ne!(*out, expected);
    }
    let elapsed = {
        let av = MatView::row_major(as_alphabet_full(a), case.m, case.k).expect("A fits");
        let bv = MatView::row_major(as_alphabet_full(b), case.k, case.n).expect("B fits");
        let cv = MatViewMut::row_major(out, case.m, case.n).expect("C fits");
        let mut triple = Triple::new(av, bv, cv).expect("the product exists");
        let mut scratch = Scratch::new(panels);
        let options = GemmOptions {
            encode: EncodeMode::Wrapping,
            ..Default::default()
        };
        let start = Instant::now();
        for _ in 0..repetitions {
            uor_matmul::gemm_packed(&mut triple, &Linear::OVERWRITE, options, &mut scratch);
        }
        start.elapsed()
    };
    assert_eq!(out, expected, "timed integer control differs");
    elapsed
}

#[cfg(all(feature = "ref-matrixmultiply", feature = "ref-faer"))]
fn timed_tropical_batch(
    shape: Shape,
    a: &[Trop<i8>],
    b: &[Trop<i8>],
    out: &mut [Trop<i32>],
    expected: &[Trop<i32>],
    repetitions: usize,
) -> Duration {
    assert_eq!(out.len(), expected.len());
    for (out, &expected) in out.iter_mut().zip(expected) {
        *out = match expected.get() {
            Some(value) => Trop::finite(value.wrapping_add(1)),
            None => Trop::finite(0),
        };
        assert_ne!(*out, expected);
    }
    let elapsed = {
        let av = MatView::row_major(as_alphabet_tropical(a), shape.m, shape.k).expect("A fits");
        let bv = MatView::row_major(as_alphabet_tropical(b), shape.k, shape.n).expect("B fits");
        let cv = MatViewMut::row_major(out, shape.m, shape.n).expect("C fits");
        let mut triple = Triple::new(av, bv, cv).expect("the product exists");
        let mut scratch = Scratch::none();
        let options = GemmOptions::default();
        let start = Instant::now();
        for _ in 0..repetitions {
            uor_matmul_gemm::gemm(&mut triple, &MaxPlus::OVERWRITE, options, &mut scratch);
        }
        start.elapsed()
    };
    assert_eq!(out, expected, "timed tropical control differs");
    elapsed
}

#[cfg(all(feature = "ref-matrixmultiply", feature = "ref-faer"))]
fn measure_float<E>(case: FloatCase)
where
    E: MeasuredFloat + EncodeFrom<AccOf<E>> + EncodeFrom<i128>,
    AccOf<E>: SignedPlace + PlaceAt + ScaleExact + AbsorbPrior<E>,
{
    let (a, b) = operands::<E>(case);
    let exact = exact_product(case, &a, &b, EncodeMode::Nearest);
    let shape = Shape {
        m: case.m,
        k: case.k,
        n: case.n,
    };
    let macs = case.m * case.k * case.n;
    let caller_bytes = logical_float_bytes::<E>(case);
    let guard_bytes = guard_float_bytes::<E>(case);
    let case_id = case.seed.to_string();
    let fill = format!("{:?}", case.fill);

    eprintln!();
    eprintln!(
        "## {} {:?} {}x{}x{} seed {}; logical bytes/call {caller_bytes}; guard bytes/batch outside timing {guard_bytes}",
        E::LABEL,
        case.fill,
        case.m,
        case.k,
        case.n,
        case.seed,
    );
    eprintln!(
        "| route | latency us (95% CI) | Gproduct/s (95% CI) | logical GB/s (95% CI) | batch | exact comparison |"
    );
    eprintln!("| --- | ---: | ---: | ---: | ---: | --- |");

    let (pa_len, pb_len) = suggested_float_panels(shape);
    let mut pa = vec![PackedCode::default(); pa_len];
    let mut pb = vec![PackedCode::default(); pb_len];
    let mut out = vec![E::ZERO; case.m * case.n];
    let offered = metrics(macs, caller_bytes, |batch| {
        timed_float_packed_batch(case, &a, &b, &mut out, &exact, &mut pa, &mut pb, batch)
    });
    print_metrics(
        E::LABEL,
        &case_id,
        shape,
        &fill,
        "pure-uor-offered",
        "pure-UOR offered",
        &offered,
        "identical",
    );

    let mut out = vec![E::ZERO; case.m * case.n];
    let no_offer = metrics(macs, caller_bytes, |batch| {
        timed_float_no_offer_batch(case, &a, &b, &mut out, &exact, batch)
    });
    print_metrics(
        E::LABEL,
        &case_id,
        shape,
        &fill,
        "pure-uor-no-offer",
        "pure-UOR no offer",
        &no_offer,
        "identical",
    );

    let mut out = vec![E::ZERO; case.m * case.n];
    let incumbent = metrics(macs, caller_bytes, |batch| {
        timed_incumbent_batch(case, &a, &b, &mut out, &exact, batch)
    });
    print_metrics(
        E::LABEL,
        &case_id,
        shape,
        &fill,
        "incumbent-exact-reference",
        "incumbent exact reference",
        &incumbent,
        "identical",
    );

    let mut matrix_want = vec![E::ZERO; case.m * case.n];
    E::matrixmultiply(case, &a, &b, &mut matrix_want);
    let matrix_deviation = deviation(&exact, &matrix_want);
    let mut out = vec![E::ZERO; case.m * case.n];
    let matrix = metrics(macs, caller_bytes, |batch| {
        timed_matrixmultiply_batch(case, &a, &b, &mut out, &matrix_want, batch)
    });
    print_metrics(
        E::LABEL,
        &case_id,
        shape,
        &fill,
        "matrixmultiply-oracle",
        "matrixmultiply oracle",
        &matrix,
        &matrix_deviation,
    );

    let mut faer_state = E::prepare_faer(case, &a, &b);
    let mut faer_want = vec![E::ZERO; case.m * case.n];
    E::faer_compute(&mut faer_state);
    E::copy_faer_output(&faer_state, case, &mut faer_want);
    let faer_deviation = deviation(&exact, &faer_want);
    let mut out = vec![E::ZERO; case.m * case.n];
    let faer = metrics(macs, caller_bytes, |batch| {
        timed_faer_batch(case, &mut faer_state, &mut out, &faer_want, batch)
    });
    print_metrics(
        E::LABEL,
        &case_id,
        shape,
        &fill,
        "faer-oracle",
        "faer oracle",
        &faer,
        &faer_deviation,
    );
}

#[cfg(all(feature = "ref-matrixmultiply", feature = "ref-faer"))]
fn integer_control() {
    let case = FloatCase {
        m: 32,
        k: 256,
        n: 32,
        fill: uor_matmul_validate::float_corpus::FloatFill::InverseGauge,
        seed: 0xC621_1001,
    };
    let fill = |len: usize, salt: u64| {
        let mut state = case.seed ^ salt;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as i8
            })
            .collect::<Vec<_>>()
    };
    let a = fill(case.m * case.k, 0xA5);
    let b = fill(case.k * case.n, 0x5A);
    let shape = Shape {
        m: case.m,
        k: case.k,
        n: case.n,
    };
    let want =
        uor_matmul_validate::ours_i8_i32(case.m, case.k, case.n, &a, &b, EncodeMode::Wrapping);
    let mut out = vec![0i32; case.m * case.n];
    let mut panels = vec![Alphabet::<i8, Full<i8>>::ZERO; suggested_scratch(shape)];
    let caller_bytes =
        a.len() * size_of::<i8>() + b.len() * size_of::<i8>() + out.len() * size_of::<i32>();
    let measured = metrics(case.m * case.k * case.n, caller_bytes, |batch| {
        timed_integer_batch(case, &a, &b, &mut panels, &mut out, &want, batch)
    });

    eprintln!();
    eprintln!("## integer control i8 32x256x32 seed {}", case.seed);
    eprintln!(
        "| route | latency us (95% CI) | Gproduct/s (95% CI) | logical GB/s (95% CI) | batch | exact comparison |"
    );
    eprintln!("| --- | ---: | ---: | ---: | ---: | --- |");
    print_metrics(
        "i8-i32",
        "integer-control",
        shape,
        "deterministic-xorshift",
        "packed-integer",
        "packed integer",
        &measured,
        "identical",
    );
}

#[cfg(all(feature = "ref-matrixmultiply", feature = "ref-faer"))]
fn tropical_control() {
    let (m, k, n) = (32usize, 256usize, 32usize);
    let shape = Shape { m, k, n };
    let a: Vec<Trop<i8>> = (0..m * k)
        .map(|at| Trop::finite(((at.wrapping_mul(37).wrapping_add(11)) as i8) >> 1))
        .collect();
    let b: Vec<Trop<i8>> = (0..k * n)
        .map(|at| Trop::finite(((at.wrapping_mul(53).wrapping_add(7)) as i8) >> 1))
        .collect();
    let mut want = vec![Trop::<i32>::NEG_INF; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut best = i32::MIN;
            for p in 0..k {
                let left = match a[i * k + p].get() {
                    Some(value) => i32::from(value),
                    None => continue,
                };
                let right = match b[p * n + j].get() {
                    Some(value) => i32::from(value),
                    None => continue,
                };
                best = best.max(left + right);
            }
            want[i * n + j] = Trop::finite(best);
        }
    }
    let mut out = vec![Trop::<i32>::NEG_INF; m * n];
    let caller_bytes = a.len() * size_of::<Trop<i8>>()
        + b.len() * size_of::<Trop<i8>>()
        + out.len() * size_of::<Trop<i32>>();
    let measured = metrics(m * k * n, caller_bytes, |batch| {
        timed_tropical_batch(shape, &a, &b, &mut out, &want, batch)
    });

    eprintln!();
    eprintln!("## tropical control Trop<i8> 32x256x32 deterministic affine fill");
    eprintln!(
        "| route | latency us (95% CI) | Goperation/s (95% CI) | logical GB/s (95% CI) | batch | exact comparison |"
    );
    eprintln!("| --- | ---: | ---: | ---: | ---: | --- |");
    print_metrics(
        "trop-i8-i32",
        "tropical-control",
        shape,
        "deterministic-affine",
        "portable-max-plus",
        "portable max-plus",
        &measured,
        "identical",
    );
}

/// `CG-21`: all rates are open observations; wrong bytes are hard failures.
#[test]
#[ignore = "release-mode measurement: `just uor-float-sweep`"]
fn pure_uor_float_measurement_with_oracles_and_controls_cg_21() {
    require_measurement_build();

    #[cfg(all(feature = "ref-matrixmultiply", feature = "ref-faer"))]
    {
        eprintln!("CG-21 (open): pure-UOR float performance");
        eprintln!(
            "host: {}-{}; cpu: {}; samples: {}; target batch: {:?}",
            std::env::consts::ARCH,
            std::env::consts::OS,
            cpu_name(),
            SAMPLE_COUNT,
            SAMPLE_TARGET,
        );
        eprintln!(
            "intervals: two-sided 95% Student t (df=8); output poison and full byte validation bracket each batch outside timing"
        );
        eprintln!(
            "traffic: logical A-read + B-read + C-write lower bound; no hardware-counter claim"
        );
        for &case in PERFORMANCE_CASES {
            measure_float::<f32>(case);
        }
        for &case in PERFORMANCE_CASES {
            measure_float::<f64>(case);
        }
        integer_control();
        tropical_control();
    }
}

#[cfg(all(feature = "ref-matrixmultiply", feature = "ref-faer"))]
fn cpu_name() -> String {
    if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
        if let Some(value) = cpuinfo.lines().find_map(|line| {
            line.strip_prefix("model name")
                .and_then(|line| line.split_once(':'))
                .map(|(_, value)| value.trim())
        }) {
            return value.to_owned();
        }
    }
    std::env::var("PROCESSOR_IDENTIFIER").unwrap_or_else(|_| "unreported".to_owned())
}
