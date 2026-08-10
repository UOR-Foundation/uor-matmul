//! Measurement-only control for the native radix-address refactor.
//!
//! The `raw/*` rows deliberately retain the superseded shift/mask address
//! spelling. They exist only in this Criterion target, are never linked into a
//! shipped crate, and provide the pre-change clock beside the resolved
//! production declarations. Correctness is compared before either clock runs.
//!
//! CG-23 records the linked inspection made with the current compiler on the
//! recorded host: the MR1 reduction hot loop is normalized-static-equivalent
//! to its retained control, as are the unchanged NR16 tiles, while the one
//! production alphabet is 64-byte aligned and reached by a direct RIP-relative
//! address rather than a weaker payload access. Those host-specific clocks
//! remain open observations; the inspection does not promote them to build
//! truth. Only bodies structurally changed by the refactor retain the
//! preregistered demonstrated-superiority decision.

use criterion::{black_box, measurement::WallTime, BenchmarkGroup, Criterion};
use std::arch::x86_64::*;
use std::time::{Duration, Instant};
use uor_matmul::kernels::{KernelSpec, TableSpec};
use uor_matmul::Backend;

const OCTET_SPACE: usize = 256;
const NIBBLE_SPACE: usize = 16;
const PROJECTOR_ROW_BYTES: usize = 64;
const PRODUCT_ENTRIES: usize = OCTET_SPACE * OCTET_SPACE;
const PROJECTOR_ENTRIES: usize = OCTET_SPACE * PROJECTOR_ROW_BYTES;

static RAW_PRODUCTS: [i32; PRODUCT_ENTRIES] = build_raw_products();
static RAW_PROJECTORS: [u8; PROJECTOR_ENTRIES] = build_raw_projectors();

/// Calls per paired epoch. Clock reads are outside each batch and therefore
/// contribute at most one thirty-second of their cost to a reported call.
const PAIRED_BATCH: u64 = 32;

/// The retained decision gate uses enough paired observations to estimate the
/// route ratio directly rather than comparing two independently fitted clocks.
const ACCEPTANCE_SAMPLES: usize = 256;
const ACCEPTANCE_TARGET: Duration = Duration::from_millis(50);
const ACCEPTANCE_CHUNK: usize = 256;
const ACCEPTANCE_SENTINEL: &str = "__native_lookup_acceptance_only__";
// Conservatively rounded above Student's t at 95%, two-sided, for
// ACCEPTANCE_SAMPLES - 1 = 255 degrees of freedom. The upper endpoint is the
// preregistered demonstrated-superiority decision for changed cases only.
const T95_DF255: f64 = 1.97;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AcceptanceCase {
    Tile { rows: usize, columns: usize },
    Reduction { rows: usize },
    Table,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AcceptanceClass {
    Changed,
    StaticEquivalent,
}

const ACCEPTANCE_CASES: [(AcceptanceCase, AcceptanceClass); 7] = [
    (
        AcceptanceCase::Tile {
            rows: 1,
            columns: 8,
        },
        AcceptanceClass::Changed,
    ),
    (
        AcceptanceCase::Tile {
            rows: 6,
            columns: 8,
        },
        AcceptanceClass::Changed,
    ),
    (
        AcceptanceCase::Reduction { rows: 4 },
        AcceptanceClass::Changed,
    ),
    (AcceptanceCase::Table, AcceptanceClass::Changed),
    (
        AcceptanceCase::Reduction { rows: 1 },
        AcceptanceClass::StaticEquivalent,
    ),
    (
        AcceptanceCase::Tile {
            rows: 1,
            columns: 16,
        },
        AcceptanceClass::StaticEquivalent,
    ),
    (
        AcceptanceCase::Tile {
            rows: 6,
            columns: 16,
        },
        AcceptanceClass::StaticEquivalent,
    ),
];

impl AcceptanceClass {
    fn label(self) -> &'static str {
        match self {
            AcceptanceClass::Changed => "changed",
            AcceptanceClass::StaticEquivalent => "open/static-control",
        }
    }
}

fn acceptance_class(case: AcceptanceCase) -> AcceptanceClass {
    ACCEPTANCE_CASES
        .iter()
        .find_map(|&(candidate, class)| (candidate == case).then_some(class))
        .unwrap_or_else(|| panic!("unclassified native lookup acceptance case: {case:?}"))
}

fn hard_acceptance_requested() -> bool {
    std::env::args_os().any(|argument| argument == ACCEPTANCE_SENTINEL)
}

fn poison_output(output: &mut [i32], expected: &[i32]) {
    assert_eq!(output.len(), expected.len());
    for (cell, &want) in output.iter_mut().zip(expected) {
        // Every cell differs from its expected value, including at the wrap,
        // so a route that leaves even one output untouched makes the guard
        // falsifiable independently of the fixture's values.
        *cell = want.wrapping_add(1);
    }
}

/// Poison, run, and completely verify one fixed batch. Only the repeated route
/// calls lie between the clock reads; both guards are deliberately outside.
fn timed_checked_batch<Run>(
    route: &str,
    repetitions: usize,
    output: &mut [i32],
    expected: &[i32],
    run: &mut Run,
) -> Duration
where
    Run: FnMut(&mut [i32]),
{
    poison_output(output, expected);
    let start = Instant::now();
    for _ in 0..repetitions {
        run(output);
    }
    let elapsed = start.elapsed();
    black_box(&*output);
    assert_eq!(output, expected, "{route} changed an output byte");
    elapsed
}

/// Measure one guarded paired sample in short alternating chunks.
///
/// This Criterion target has `harness = false`, so there is no unit-test seam
/// for its clock boundary. The retained runtime structure is the falsifiable
/// witness: both outputs are poisoned before the first clock read, each route
/// receives the same fixed-size chunks in alternating order, and both complete
/// outputs are checked only after the final read. At 256 calls per chunk the
/// two symmetric clock pairs contribute less than 0.1% of either interval.
fn timed_interleaved_pair<Raw, Resolved>(
    raw_first: bool,
    repetitions: usize,
    raw_output: &mut [i32],
    resolved_output: &mut [i32],
    expected: &[i32],
    raw_run: &mut Raw,
    resolved_run: &mut Resolved,
) -> (Duration, Duration)
where
    Raw: FnMut(&mut [i32]),
    Resolved: FnMut(&mut [i32]),
{
    assert_eq!(repetitions % ACCEPTANCE_CHUNK, 0);
    poison_output(raw_output, expected);
    poison_output(resolved_output, expected);

    let mut raw_elapsed = Duration::ZERO;
    let mut resolved_elapsed = Duration::ZERO;
    let mut remaining = repetitions;
    let mut raw_goes_first = raw_first;
    while remaining != 0 {
        if raw_goes_first {
            let start = Instant::now();
            for _ in 0..ACCEPTANCE_CHUNK {
                raw_run(raw_output);
            }
            raw_elapsed += start.elapsed();

            let start = Instant::now();
            for _ in 0..ACCEPTANCE_CHUNK {
                resolved_run(resolved_output);
            }
            resolved_elapsed += start.elapsed();
        } else {
            let start = Instant::now();
            for _ in 0..ACCEPTANCE_CHUNK {
                resolved_run(resolved_output);
            }
            resolved_elapsed += start.elapsed();

            let start = Instant::now();
            for _ in 0..ACCEPTANCE_CHUNK {
                raw_run(raw_output);
            }
            raw_elapsed += start.elapsed();
        }
        raw_goes_first = !raw_goes_first;
        remaining -= ACCEPTANCE_CHUNK;
    }

    black_box(&*raw_output);
    black_box(&*resolved_output);
    assert_eq!(raw_output, expected, "raw sample changed an output byte");
    assert_eq!(
        resolved_output, expected,
        "resolved sample changed an output byte"
    );
    (raw_elapsed, resolved_elapsed)
}

/// Calibrate one common batch and alternate the route order. Every case emits
/// its complete paired evidence; only CG-23's structurally changed class keeps
/// the preregistered demonstrated-superiority decision.
fn measure_paired_acceptance<Raw, Resolved>(
    case: AcceptanceCase,
    case_label: &str,
    raw_output: &mut [i32],
    resolved_output: &mut [i32],
    expected: &[i32],
    mut raw_run: Raw,
    mut resolved_run: Resolved,
) where
    Raw: FnMut(&mut [i32]),
    Resolved: FnMut(&mut [i32]),
{
    assert!(
        hard_acceptance_requested(),
        "paired native lookup acceptance is sentinel-only"
    );
    let class = acceptance_class(case);
    let class_label = class.label();
    // Equal guarded warm-ups precede calibration. Doubling derives a common
    // batch whose faster route reaches the target without imposing a data or
    // dimension limit on either implementation.
    timed_checked_batch("raw warm-up", 1, raw_output, expected, &mut raw_run);
    timed_checked_batch(
        "resolved warm-up",
        1,
        resolved_output,
        expected,
        &mut resolved_run,
    );
    let mut repetitions = ACCEPTANCE_CHUNK;
    loop {
        let raw = timed_checked_batch(
            "raw calibration",
            repetitions,
            raw_output,
            expected,
            &mut raw_run,
        );
        let resolved = timed_checked_batch(
            "resolved calibration",
            repetitions,
            resolved_output,
            expected,
            &mut resolved_run,
        );
        if raw.min(resolved) >= ACCEPTANCE_TARGET {
            break;
        }
        repetitions = repetitions
            .checked_add(repetitions)
            .expect("native lookup calibration repetition count is representable");
    }

    let mut paired_log_ratios = [0.0f64; ACCEPTANCE_SAMPLES];
    for (round, log_ratio) in paired_log_ratios.iter_mut().enumerate() {
        let (raw, resolved) = timed_interleaved_pair(
            round % 2 == 0,
            repetitions,
            raw_output,
            resolved_output,
            expected,
            &mut raw_run,
            &mut resolved_run,
        );
        *log_ratio = (resolved.as_secs_f64() / raw.as_secs_f64()).ln();
    }

    let count = ACCEPTANCE_SAMPLES as f64;
    let mean_log = paired_log_ratios.iter().sum::<f64>() / count;
    let sum_squares = paired_log_ratios
        .iter()
        .map(|value| {
            let residual = value - mean_log;
            residual * residual
        })
        .sum::<f64>();
    let variance = sum_squares / (count - 1.0);
    let upper_log = mean_log + T95_DF255 * (variance / count).sqrt();
    let geometric_mean = mean_log.exp();
    let upper_95 = upper_log.exp();
    eprintln!(
        "native_lookup acceptance class={class_label} {case_label}: samples={ACCEPTANCE_SAMPLES} batch={repetitions} resolved/raw={geometric_mean:.6} upper95={upper_95:.6} paired_log_ratios={paired_log_ratios:?}"
    );
    if class == AcceptanceClass::Changed {
        assert!(
            upper_95 <= 1.0,
            "native lookup changed-case demonstrated-superiority failure for {case_label}: resolved/raw upper95 {upper_95:.6} exceeds 1.0 (geometric mean {geometric_mean:.6})"
        );
    }
}

/// Measure one side while executing its counterpart between equally sized
/// batches. Rotating which side goes first makes cache and frequency state a
/// property of the pair rather than of registration order.
fn paired_duration<Measured, Counterpart>(
    iterations: u64,
    measured_first: &mut bool,
    mut measured: Measured,
    mut counterpart: Counterpart,
) -> Duration
where
    Measured: FnMut(),
    Counterpart: FnMut(),
{
    let mut elapsed = Duration::ZERO;
    let mut remaining = iterations;
    while remaining != 0 {
        let calls = remaining.min(PAIRED_BATCH);
        if *measured_first {
            let start = Instant::now();
            for _ in 0..calls {
                measured();
            }
            elapsed += start.elapsed();
            for _ in 0..calls {
                counterpart();
            }
        } else {
            for _ in 0..calls {
                counterpart();
            }
            let start = Instant::now();
            for _ in 0..calls {
                measured();
            }
            elapsed += start.elapsed();
        }
        *measured_first = !*measured_first;
        remaining -= calls;
    }
    elapsed
}

const fn build_raw_products() -> [i32; PRODUCT_ENTRIES] {
    let mut table = [0i32; PRODUCT_ENTRIES];
    let mut left = 0usize;
    let mut at = 0usize;
    while left < OCTET_SPACE {
        let mut right = 0usize;
        while right < OCTET_SPACE {
            table[at] = (left as u8 as i8 as i32) * (right as u8 as i8 as i32);
            at += 1;
            right += 1;
        }
        left += 1;
    }
    table
}

const fn build_raw_projectors() -> [u8; PROJECTOR_ENTRIES] {
    let mut table = [0u8; PROJECTOR_ENTRIES];
    let mut left_code = 0usize;
    while left_code < OCTET_SPACE {
        let left = left_code as u8 as i8;
        let row = left_code * PROJECTOR_ROW_BYTES;
        let mut digit = 0usize;
        while digit < NIBBLE_SPACE {
            let signed_high = if digit < NIBBLE_SPACE / 2 {
                digit as i8
            } else {
                digit as i8 - NIBBLE_SPACE as i8
            };
            let low = (left as i16) * (digit as i16);
            let high = (left as i16) * ((signed_high << 4) as i16);
            let low_bytes = low.to_le_bytes();
            let high_bytes = high.to_le_bytes();
            table[row + digit] = low_bytes[0];
            table[row + NIBBLE_SPACE + digit] = low_bytes[1];
            table[row + 2 * NIBBLE_SPACE + digit] = high_bytes[0];
            table[row + 3 * NIBBLE_SPACE + digit] = high_bytes[1];
            digit += 1;
        }
        left_code += 1;
    }
    table
}

/// The superseded AVX2 nibble address spelling, retained only as a clock.
#[target_feature(enable = "avx2")]
unsafe fn raw_nibble_products(a: i8, indices: __m128i) -> [__m128i; 2] {
    let row = (a as u8 as usize) * PROJECTOR_ROW_BYTES;
    // SAFETY: one row contains four complete sixteen-byte projector tables.
    let tables = unsafe {
        [
            _mm_loadu_si128(RAW_PROJECTORS.as_ptr().add(row).cast()),
            _mm_loadu_si128(RAW_PROJECTORS.as_ptr().add(row + NIBBLE_SPACE).cast()),
            _mm_loadu_si128(RAW_PROJECTORS.as_ptr().add(row + 2 * NIBBLE_SPACE).cast()),
            _mm_loadu_si128(RAW_PROJECTORS.as_ptr().add(row + 3 * NIBBLE_SPACE).cast()),
        ]
    };
    let mask = _mm_set1_epi8((NIBBLE_SPACE - 1) as i8);
    let low_index = _mm_and_si128(indices, mask);
    let high_index = _mm_and_si128(_mm_srli_epi16::<4>(indices), mask);
    let low_bytes = _mm_shuffle_epi8(tables[0], low_index);
    let low_sign = _mm_shuffle_epi8(tables[1], low_index);
    let high_bytes = _mm_shuffle_epi8(tables[2], high_index);
    let high_sign = _mm_shuffle_epi8(tables[3], high_index);
    [
        _mm_add_epi16(
            _mm_unpacklo_epi8(low_bytes, low_sign),
            _mm_unpacklo_epi8(high_bytes, high_sign),
        ),
        _mm_add_epi16(
            _mm_unpackhi_epi8(low_bytes, low_sign),
            _mm_unpackhi_epi8(high_bytes, high_sign),
        ),
    ]
}

/// The exact pre-change AVX2 tile, specialized to each resolved production
/// geometry by the dispatcher below.
#[target_feature(enable = "avx2")]
unsafe fn raw_tile<const MR: usize, const NR: usize>(
    kc: usize,
    pa: *const i8,
    pb: *const i8,
    acc: *mut i32,
) {
    // SAFETY: the measurement KernelSpec applies the same exact extent checks
    // as the resolved declaration before entering this retained body.
    let (pa, pb, acc) = unsafe {
        (
            core::slice::from_raw_parts(pa, MR * kc),
            core::slice::from_raw_parts(pb, NR * kc),
            core::slice::from_raw_parts_mut(acc, MR * NR),
        )
    };
    let mut tile = [[_mm256_setzero_si256(); 2]; MR];
    let nibble_mask = _mm_set1_epi8((NIBBLE_SPACE - 1) as i8);
    for p in 0..kc {
        // SAFETY: the measurement KernelSpec checks the exact panel extents.
        let b = unsafe {
            if NR == 16 {
                _mm_loadu_si128(pb.as_ptr().add(p * NR).cast())
            } else {
                _mm_loadl_epi64(pb.as_ptr().add(p * NR).cast())
            }
        };
        let low_index = _mm_and_si128(b, nibble_mask);
        let high_index = _mm_and_si128(_mm_srli_epi16::<4>(b), nibble_mask);
        for i in 0..MR {
            let row = (pa[p * MR + i] as u8 as usize) * PROJECTOR_ROW_BYTES;
            // SAFETY: the selected projector row contains four full tables.
            let tables = unsafe {
                [
                    _mm_loadu_si128(RAW_PROJECTORS.as_ptr().add(row).cast()),
                    _mm_loadu_si128(RAW_PROJECTORS.as_ptr().add(row + NIBBLE_SPACE).cast()),
                    _mm_loadu_si128(RAW_PROJECTORS.as_ptr().add(row + 2 * NIBBLE_SPACE).cast()),
                    _mm_loadu_si128(RAW_PROJECTORS.as_ptr().add(row + 3 * NIBBLE_SPACE).cast()),
                ]
            };
            let low_bytes = _mm_shuffle_epi8(tables[0], low_index);
            let low_sign = _mm_shuffle_epi8(tables[1], low_index);
            let high_bytes = _mm_shuffle_epi8(tables[2], high_index);
            let high_sign = _mm_shuffle_epi8(tables[3], high_index);
            let products0 = _mm_add_epi16(
                _mm_unpacklo_epi8(low_bytes, low_sign),
                _mm_unpacklo_epi8(high_bytes, high_sign),
            );
            let products1 = _mm_add_epi16(
                _mm_unpackhi_epi8(low_bytes, low_sign),
                _mm_unpackhi_epi8(high_bytes, high_sign),
            );
            tile[i][0] = _mm256_add_epi32(tile[i][0], _mm256_cvtepi16_epi32(products0));
            if NR == 16 {
                tile[i][1] = _mm256_add_epi32(tile[i][1], _mm256_cvtepi16_epi32(products1));
            }
        }
    }
    for (i, row) in tile.iter().enumerate() {
        // SAFETY: the measurement KernelSpec supplies the exact output extent.
        unsafe {
            _mm256_storeu_si256(acc.as_mut_ptr().add(i * NR).cast(), row[0]);
            if NR == 16 {
                _mm256_storeu_si256(acc.as_mut_ptr().add(i * NR + 8).cast(), row[1]);
            }
        }
    }
}

fn raw_tile_spec(spec: KernelSpec<i8, i32>) -> KernelSpec<i8, i32> {
    let mac_tile: unsafe fn(usize, *const i8, *const i8, *mut i32) = match (spec.mr, spec.nr) {
        (1, 8) => raw_tile::<1, 8>,
        (1, 16) => raw_tile::<1, 16>,
        (6, 8) => raw_tile::<6, 8>,
        (6, 16) => raw_tile::<6, 16>,
        shape => panic!("unmeasured AVX2 lookup tile geometry {shape:?}"),
    };
    KernelSpec { mac_tile, ..spec }
}

/// The pre-change AVX2 reduction address construction.
#[target_feature(enable = "avx2")]
unsafe fn raw_reduce<const MR: usize>(kc: usize, pa: *const i8, pb: *const i8, acc: *mut i32) {
    // SAFETY: the measurement KernelSpec applies the same exact extent checks
    // as the resolved declaration before entering this retained body.
    let (pa, pb, acc) = unsafe {
        (
            core::slice::from_raw_parts(pa, MR * kc),
            core::slice::from_raw_parts(pb, kc),
            core::slice::from_raw_parts_mut(acc, MR),
        )
    };
    if MR == 1 {
        let mut sum = _mm256_setzero_si256();
        let vector_end = kc - kc % 8;
        for p in (0..vector_end).step_by(8) {
            // SAFETY: both panels have at least eight values from `p`.
            let (a_octets, b_octets) = unsafe {
                (
                    _mm_loadl_epi64(pa.as_ptr().add(p).cast()),
                    _mm_loadl_epi64(pb.as_ptr().add(p).cast()),
                )
            };
            let a = _mm256_cvtepu8_epi32(a_octets);
            let b = _mm256_cvtepu8_epi32(b_octets);
            let indices = _mm256_or_si256(_mm256_slli_epi32::<8>(a), b);
            // SAFETY: every lane is one complete unsigned-octet pair address.
            let products = unsafe { _mm256_i32gather_epi32(RAW_PRODUCTS.as_ptr(), indices, 4) };
            sum = _mm256_add_epi32(sum, products);
        }
        let mut lanes = [0i32; 8];
        // SAFETY: the local has exactly eight writable lanes.
        unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast(), sum) };
        let mut total = lanes.into_iter().fold(0i32, i32::wrapping_add);
        for p in vector_end..kc {
            total = total.wrapping_add(i32::from(pa[p]) * i32::from(pb[p]));
        }
        acc[0] = total;
        return;
    }

    let mut sum = _mm_setzero_si128();
    for p in 0..kc {
        let indices = _mm_setr_epi8(
            pa[p],
            pa[kc + p],
            pa[2 * kc + p],
            pa[3 * kc + p],
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        );
        // SAFETY: the right value selects one complete projector row.
        let products = unsafe { raw_nibble_products(pb[p], indices) };
        sum = _mm_add_epi32(sum, _mm_cvtepi16_epi32(products[0]));
    }
    // SAFETY: the four-row arm supplies four writable lanes.
    unsafe { _mm_storeu_si128(acc.as_mut_ptr().cast(), sum) };
}

fn raw_reduce_spec(spec: KernelSpec<i8, i32>) -> KernelSpec<i8, i32> {
    let mac_tile: unsafe fn(usize, *const i8, *const i8, *mut i32) = match spec.mr {
        1 => raw_reduce::<1>,
        4 => raw_reduce::<4>,
        rows => panic!("unmeasured AVX2 lookup reduction height {rows}"),
    };
    KernelSpec { mac_tile, ..spec }
}

/// One shared safe wrapper for both retained and resolved kernel clocks.
#[inline(never)]
fn clock_kernel(spec: &KernelSpec<i8, i32>, kc: usize, pa: &[i8], pb: &[i8], acc: &mut [i32]) {
    spec.mac_tile(kc, pa, pb, acc);
}

/// The pre-change AVX2 full-alphabet table builder at its resolved 16 rows.
#[target_feature(enable = "avx2")]
unsafe fn raw_table_build(
    _rows: usize,
    space: usize,
    block: usize,
    book: *const i8,
    acts: *const i8,
    out: *mut i32,
) {
    const ROWS: usize = 16;
    // SAFETY: the measurement TableSpec applies the same exact extent checks
    // as the resolved declaration before entering this retained body.
    let (book, acts, out) = unsafe {
        (
            core::slice::from_raw_parts(book, space * block),
            core::slice::from_raw_parts(acts, ROWS * block),
            core::slice::from_raw_parts_mut(out, space * ROWS),
        )
    };
    for c in 0..space {
        let mut entry = [_mm256_setzero_si256(); 2];
        for t in 0..block {
            let weight = book[c * block + t] as u8 as i32;
            for (vector, cell) in entry.iter_mut().enumerate() {
                let mut indices = [0i32; 8];
                for (lane, index) in indices.iter_mut().enumerate() {
                    let row = vector * 8 + lane;
                    let activation =
                        acts[uor_matmul::kernels::packed_slot(t, row, 16, 2)] as u8 as i32;
                    *index = (activation << 8) | weight;
                }
                // SAFETY: every lane is one complete unsigned-octet address.
                let products = unsafe {
                    _mm256_i32gather_epi32(
                        RAW_PRODUCTS.as_ptr(),
                        _mm256_loadu_si256(indices.as_ptr().cast()),
                        4,
                    )
                };
                *cell = _mm256_add_epi32(*cell, products);
            }
        }
        // SAFETY: each slot owns sixteen contiguous output lanes.
        unsafe {
            _mm256_storeu_si256(out.as_mut_ptr().add(c * 16).cast(), entry[0]);
            _mm256_storeu_si256(out.as_mut_ptr().add(c * 16 + 8).cast(), entry[1]);
        }
    }
}

fn raw_table_spec(spec: TableSpec<i8, i32>) -> TableSpec<i8, i32> {
    TableSpec {
        build: raw_table_build,
        ..spec
    }
}

/// One shared safe wrapper for both retained and resolved table clocks.
#[inline(never)]
fn clock_table(
    spec: &TableSpec<i8, i32>,
    space: usize,
    block: usize,
    book: &[i8],
    acts: &[i8],
    out: &mut [i32],
) {
    spec.build(space, block, book, acts, out);
}

fn label(prefix: &str, kind: &str, backend: Backend, rows: usize, columns: usize) -> String {
    format!(
        "{prefix}/{kind}/{}-mr{rows}-nr{columns}-kg1",
        backend.as_str()
    )
}

fn bench_tile(
    group: &mut BenchmarkGroup<'_, WallTime>,
    spec: KernelSpec<i8, i32>,
    run_acceptance: bool,
) {
    let kc = 256usize;
    let raw_spec = raw_tile_spec(spec);
    let pa: Vec<_> = (0..spec.mr * kc)
        .map(|i| ((i.wrapping_mul(29) % 255) as i16 - 127) as i8)
        .collect();
    let pb: Vec<_> = (0..spec.nr * kc)
        .map(|i| ((i.wrapping_mul(43) % 255) as i16 - 127) as i8)
        .collect();
    let mut raw = vec![0i32; spec.mr * spec.nr];
    let mut resolved = vec![0i32; spec.mr * spec.nr];
    clock_kernel(&raw_spec, kc, &pa, &pb, &mut raw);
    clock_kernel(&spec, kc, &pa, &pb, &mut resolved);
    assert_eq!(resolved, raw, "resolved tile must match its raw clock");
    if run_acceptance {
        let expected = raw.clone();
        measure_paired_acceptance(
            AcceptanceCase::Tile {
                rows: spec.mr,
                columns: spec.nr,
            },
            &label("acceptance", "tile", spec.backend, spec.mr, spec.nr),
            &mut raw,
            &mut resolved,
            &expected,
            |output| clock_kernel(&raw_spec, kc, &pa, &pb, output),
            |output| clock_kernel(&spec, kc, &pa, &pb, output),
        );
    }

    group.bench_function(label("raw", "tile", spec.backend, spec.mr, spec.nr), |b| {
        b.iter(|| {
            clock_kernel(&raw_spec, kc, &pa, &pb, &mut raw);
            black_box(&raw);
        });
    });
    group.bench_function(
        label("resolved", "tile", spec.backend, spec.mr, spec.nr),
        |b| {
            b.iter(|| {
                clock_kernel(&spec, kc, &pa, &pb, &mut resolved);
                black_box(&resolved);
            });
        },
    );

    let mut raw_first = true;
    group.bench_function(
        label("paired/raw", "tile", spec.backend, spec.mr, spec.nr),
        |b| {
            b.iter_custom(|iterations| {
                paired_duration(
                    iterations,
                    &mut raw_first,
                    || {
                        clock_kernel(&raw_spec, kc, &pa, &pb, &mut raw);
                        black_box(&raw);
                    },
                    || {
                        clock_kernel(&spec, kc, &pa, &pb, &mut resolved);
                        black_box(&resolved);
                    },
                )
            });
        },
    );
    let mut resolved_first = true;
    group.bench_function(
        label("paired/resolved", "tile", spec.backend, spec.mr, spec.nr),
        |b| {
            b.iter_custom(|iterations| {
                paired_duration(
                    iterations,
                    &mut resolved_first,
                    || {
                        clock_kernel(&spec, kc, &pa, &pb, &mut resolved);
                        black_box(&resolved);
                    },
                    || {
                        clock_kernel(&raw_spec, kc, &pa, &pb, &mut raw);
                        black_box(&raw);
                    },
                )
            });
        },
    );
}

/// Register the raw-address control and every resolved AVX2 lookup declaration.
pub(super) fn bench(c: &mut Criterion) {
    if !std::arch::is_x86_feature_detected!("avx2") {
        return;
    }

    let run_acceptance = hard_acceptance_requested();
    let mut group = c.benchmark_group("native_lookup");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));

    let mut tile_specs: Vec<_> = uor_matmul::kernels::available_i8()
        .chain(uor_matmul::kernels::available_i8_narrow())
        .filter(|spec| spec.backend == Backend::Avx2 && spec.k_group == 1)
        .collect();
    tile_specs.sort_unstable_by_key(|spec| (spec.nr, spec.mr));
    for &spec in tile_specs.iter().filter(|spec| spec.nr == 8) {
        bench_tile(&mut group, spec, run_acceptance);
    }

    let reduce_specs: Vec<_> = uor_matmul::kernels::available_reduce_i8()
        .filter(|spec| spec.backend == Backend::Avx2 && spec.k_group == 1)
        .collect();
    for spec in reduce_specs {
        let kc = 4096usize;
        let raw_spec = raw_reduce_spec(spec);
        let pa: Vec<_> = (0..spec.mr * kc)
            .map(|i| ((i.wrapping_mul(17) % 255) as i16 - 127) as i8)
            .collect();
        let pb: Vec<_> = (0..kc)
            .map(|i| ((i.wrapping_mul(31) % 255) as i16 - 127) as i8)
            .collect();
        let mut raw = vec![0i32; spec.mr];
        let mut resolved = vec![0i32; spec.mr];
        clock_kernel(&raw_spec, kc, &pa, &pb, &mut raw);
        clock_kernel(&spec, kc, &pa, &pb, &mut resolved);
        assert_eq!(resolved, raw, "resolved reduction must match its raw clock");
        if run_acceptance {
            let expected = raw.clone();
            measure_paired_acceptance(
                AcceptanceCase::Reduction { rows: spec.mr },
                &label("acceptance", "reduce", spec.backend, spec.mr, spec.nr),
                &mut raw,
                &mut resolved,
                &expected,
                |output| clock_kernel(&raw_spec, kc, &pa, &pb, output),
                |output| clock_kernel(&spec, kc, &pa, &pb, output),
            );
        }

        group.bench_function(
            label("raw", "reduce", spec.backend, spec.mr, spec.nr),
            |b| {
                b.iter(|| {
                    clock_kernel(&raw_spec, kc, &pa, &pb, &mut raw);
                    black_box(&raw);
                });
            },
        );
        group.bench_function(
            label("resolved", "reduce", spec.backend, spec.mr, spec.nr),
            |b| {
                b.iter(|| {
                    clock_kernel(&spec, kc, &pa, &pb, &mut resolved);
                    black_box(&resolved);
                });
            },
        );

        let mut raw_first = true;
        group.bench_function(
            label("paired/raw", "reduce", spec.backend, spec.mr, spec.nr),
            |b| {
                b.iter_custom(|iterations| {
                    paired_duration(
                        iterations,
                        &mut raw_first,
                        || {
                            clock_kernel(&raw_spec, kc, &pa, &pb, &mut raw);
                            black_box(&raw);
                        },
                        || {
                            clock_kernel(&spec, kc, &pa, &pb, &mut resolved);
                            black_box(&resolved);
                        },
                    )
                });
            },
        );
        let mut resolved_first = true;
        group.bench_function(
            label("paired/resolved", "reduce", spec.backend, spec.mr, spec.nr),
            |b| {
                b.iter_custom(|iterations| {
                    paired_duration(
                        iterations,
                        &mut resolved_first,
                        || {
                            clock_kernel(&spec, kc, &pa, &pb, &mut resolved);
                            black_box(&resolved);
                        },
                        || {
                            clock_kernel(&raw_spec, kc, &pa, &pb, &mut raw);
                            black_box(&raw);
                        },
                    )
                });
            },
        );
    }

    let rows = 16usize;
    let block = 16usize;
    let space = 4096usize;
    let spec: TableSpec<i8, i32> = uor_matmul::kernels::available_table_i8(rows, 1)
        .find(|spec| spec.backend == Backend::Avx2)
        .expect("an AVX2 host resolves its lookup table builder");
    let raw_spec = raw_table_spec(spec);
    let book: Vec<_> = (0..space * block)
        .map(|i| ((i.wrapping_mul(13) % 255) as i16 - 127) as i8)
        .collect();
    let logical_acts: Vec<_> = (0..rows * block)
        .map(|i| ((i.wrapping_mul(7) % 255) as i16 - 127) as i8)
        .collect();
    let mut acts = vec![0i8; rows * block];
    for t in 0..block {
        for row in 0..rows {
            acts[uor_matmul::kernels::packed_slot(t, row, rows, spec.k_group)] =
                logical_acts[t * rows + row];
        }
    }
    let mut raw = vec![0i32; space * rows];
    let mut resolved = vec![0i32; space * rows];
    clock_table(&raw_spec, space, block, &book, &acts, &mut raw);
    clock_table(&spec, space, block, &book, &acts, &mut resolved);
    assert_eq!(
        resolved, raw,
        "resolved table build must match its raw clock"
    );
    let table_label = format!(
        "{}-rows{}-group1-kg{}",
        spec.backend.as_str(),
        rows,
        spec.k_group
    );
    if run_acceptance {
        let expected = raw.clone();
        measure_paired_acceptance(
            AcceptanceCase::Table,
            &format!("acceptance/table/{table_label}"),
            &mut raw,
            &mut resolved,
            &expected,
            |output| clock_table(&raw_spec, space, block, &book, &acts, output),
            |output| clock_table(&spec, space, block, &book, &acts, output),
        );
    }
    group.bench_function(format!("raw/table/{table_label}"), |b| {
        b.iter(|| {
            clock_table(&raw_spec, space, block, &book, &acts, &mut raw);
            black_box(&raw);
        });
    });
    group.bench_function(format!("resolved/table/{table_label}"), |b| {
        b.iter(|| {
            clock_table(&spec, space, block, &book, &acts, &mut resolved);
            black_box(&resolved);
        });
    });
    let mut raw_first = true;
    group.bench_function(format!("paired/raw/table/{table_label}"), |b| {
        b.iter_custom(|iterations| {
            paired_duration(
                iterations,
                &mut raw_first,
                || {
                    clock_table(&raw_spec, space, block, &book, &acts, &mut raw);
                    black_box(&raw);
                },
                || {
                    clock_table(&spec, space, block, &book, &acts, &mut resolved);
                    black_box(&resolved);
                },
            )
        });
    });
    let mut resolved_first = true;
    group.bench_function(format!("paired/resolved/table/{table_label}"), |b| {
        b.iter_custom(|iterations| {
            paired_duration(
                iterations,
                &mut resolved_first,
                || {
                    clock_table(&spec, space, block, &book, &acts, &mut resolved);
                    black_box(&resolved);
                },
                || {
                    clock_table(&raw_spec, space, block, &book, &acts, &mut raw);
                    black_box(&raw);
                },
            )
        });
    });

    // The unchanged full-width declarations are controls for the same radix
    // projector. Run them last so their near-unity interval cannot prevent the
    // changed narrow path, reductions, and table builder from being observed.
    for &spec in tile_specs.iter().filter(|spec| spec.nr == 16) {
        bench_tile(&mut group, spec, run_acceptance);
    }
    group.finish();
}
