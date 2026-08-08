//! The block-16 pricing harness (phase F).
//!
//! Four configurations of one codebook: `Book<256, 8>` and `Book<256, 16>`,
//! each in the `u16` and the `u8` spelling. The 16-wide book is built so both
//! blocks price *the same product*: `book16[c] = book8[c] ++ book8[(c + 128)
//! % 256]`, and the code streams are chosen so `decode16(codes16)` and
//! `decode8(codes8)` are the same matrix. Byte-identity between the two
//! blocks' outputs is asserted inside every timed run.
//!
//! Per shape and configuration this prints the terms the phase prices:
//! persistent bytes per decoded weight, codebook bytes, the resolved plan
//! (rows, columns, depth, and which term binds the depth), slab and stack
//! bytes at that plan, the break-even recomputed from the host's own
//! declarations (never a recorded one), build / gather / end-to-end times,
//! and the census. `BLOCK_SWEEP_ITERS` sizes the timing loop (default 30);
//! `BLOCK_SWEEP_CHECK=1` is the correctness dry run --- one iteration, with
//! the census ratios the work order predicts *asserted*: table reads halve,
//! build products stay constant, codebook decodes double, the resolved plans
//! are identical, and the `u8` spelling's census is the `u16` one's, field
//! for field.
//!
//! Every figure this prints is `open` --- measured and reported, never asserted.
//! `MEASUREMENT-LOG.md` records the quiet-window run: block 16 won at every
//! shape where the table pays, while every timed closure asserted byte identity.

use std::mem::size_of;
use std::time::Instant;

use uor_matmul::codec::{e8_table, Book, CodedMatrix, Enumerable};
use uor_matmul::core_types::generated::blocking;
use uor_matmul::driver::tabulated::{column_group, slab_codes, Plan, ROW_TILES};
use uor_matmul::driver::{Census, Tabulated, Tabulation};
use uor_matmul::{
    as_alphabet_full, gemm_tabulated_counted, suggested_collapse_index, suggested_collapse_rows,
    suggested_scratch, suggested_tabulation, tabulation_fits, tabulation_pays, tabulation_rows,
    Alphabet, Backend, Collapse, Full, GemmOptions, Linear, MatView, MatViewMut, Scratch, Shape,
    TabulatedTriple, Traversal,
};

type A8 = Alphabet<i8, Full<i8>>;

/// The bound every configuration tabulates at: the full `i8` alphabet, which
/// is E8's own.
const BOUND: u128 = <Full<i8> as uor_matmul::Bound>::VALUE;

/// The four shapes from the tabulation sweeps, plus the note the grid needs:
/// `k` must be a whole number of blocks, so `2048x8x2048` runs block 8 only.
const SHAPES: [(usize, usize, usize); 4] = [
    (64, 1024, 4096),
    (64, 4096, 4096),
    (8, 262_144, 8),
    (2048, 8, 2048),
];

/// The code value column `j` names at 16-wide slot `q`, in `0..256`.
fn code_value(j: usize, q: usize) -> usize {
    (j * 37 + q * 101) % 256
}

/// The operand's code stream for one block width, built so the two blocks
/// decode to the same matrix: at block 16, `W[j][16q + t]` is `book8[c][t]`
/// for `t < 8` and `book8[(c + 128) % 256][t]` past it, and the block-8
/// stream names the same entries one codeword at a time.
fn stream<C, F>(n: usize, k: usize, block: usize, widen: F) -> Vec<C::Code>
where
    C: Enumerable<i8, Full<i8>>,
    F: Fn(usize) -> C::Code,
{
    let cpr = k / block;
    (0..n * cpr)
        .map(|i| {
            let (j, p) = (i / cpr, i % cpr);
            let v = if block == 16 {
                code_value(j, p)
            } else {
                let base = code_value(j, p / 2);
                if p % 2 == 0 {
                    base
                } else {
                    (base + 128) % 256
                }
            };
            widen(v)
        })
        .collect()
}

/// The plan the driver resolves for this shape and block, recomputed from the
/// same offers the harness hands the traversal --- CG-18's own recompute, not
/// a copy of the derivation.
fn plan_for(space: usize, shape: Shape, block: usize) -> Plan {
    let lane = size_of::<<i8 as Tabulated>::Lane>();
    Plan::choose(
        space,
        shape,
        lane,
        suggested_tabulation::<i8, Full<i8>>(shape, space, block).max(1),
        uor_matmul::driver::suggested_tabulation_lanes::<i8, Full<i8>>(shape, space, block).max(1)
            * 8
            / lane,
        block,
        <i8 as Tabulated>::probe_capacity::<<i8 as Tabulated>::Lane>(BOUND),
    )
    .expect("the suggested offers admit a plan")
}

/// The break-even for one block width, recomputed from the host's own
/// sequence declarations, as CG-18 does: the first `n` where the table wins.
fn break_even(space: usize, block: usize, m: usize) -> Option<usize> {
    let lane = size_of::<<i8 as Tabulated>::Lane>();
    let rows = ROW_TILES
        .into_iter()
        .find(|&r| r <= tabulation_rows(space, blocking::L1_BYTES, lane).min(m))?;
    let spec =
        <i8 as Tabulated>::table_spec(Backend::Auto, BOUND, false, rows, column_group(rows), block);
    let steps =
        <i8 as Tabulated>::dense_steps(Backend::Auto, BOUND, rows, block * spec.lanes_per_add);
    (1..).find(|&cols| tabulation_pays(space, block, cols, rows, steps, blocking::L1_BYTES, lane))
}

/// One end-to-end tabulated product, with the census.
#[allow(clippy::too_many_arguments)]
fn e2e<C: Enumerable<i8, Full<i8>> + Copy>(
    codec: C,
    codes: &[C::Code],
    a: &[i8],
    m: usize,
    k: usize,
    n: usize,
    block: usize,
) -> (Vec<i32>, Census) {
    let shape = Shape { m, k, n };
    let space = C::CODE_SPACE;
    let w = CodedMatrix::new(codec, n, k, codes).expect("the codes describe n x k");
    let mut out = vec![0i32; m * n];
    let mut census = Census::default();
    {
        let mut panel = vec![
            Alphabet::<i8, Full<i8>>::ZERO;
            uor_matmul::driver::suggested_tabulation_panel(space, block)
                .max(n * k + suggested_scratch(shape))
        ];
        let mut accs =
            vec![0i128; suggested_tabulation::<i8, Full<i8>>(shape, space, block).max(1)];
        let mut lane_words =
            vec![
                0i64;
                uor_matmul::driver::suggested_tabulation_lanes::<i8, Full<i8>>(shape, space, block)
                    .max(1)
            ];
        let mut ids = vec![0usize; uor_matmul::driver::suggested_tabulation_index(shape)];
        let mut collapse_index = vec![0usize; suggested_collapse_index(m)];
        let mut collapse_rows =
            vec![Alphabet::<i8, Full<i8>>::ZERO; suggested_collapse_rows(shape)];
        let av = MatView::row_major(as_alphabet_full(a), m, k).expect("A fits");
        let cv = MatViewMut::row_major(&mut out, m, n).expect("C fits");
        let mut t = TabulatedTriple::new(av, w, cv).expect("the product exists");
        gemm_tabulated_counted(
            &mut t,
            &Linear::OVERWRITE,
            GemmOptions {
                traversal: Traversal::Tabulated,
                ..Default::default()
            },
            &mut Scratch::with_accumulators(&mut panel, &mut accs),
            &mut Tabulation::with_index(&mut lane_words, &mut ids),
            &mut Collapse::new(&mut collapse_index, &mut collapse_rows),
            &mut census,
        );
    }
    (out, census)
}

/// What one configuration's run leaves behind for the cross-checks.
#[derive(Clone, Copy)]
struct Row {
    census: Census,
    plan: Plan,
}

/// Run one configuration at one shape: print the terms, return the row.
#[allow(clippy::too_many_arguments)]
fn run_config<C, F>(
    label: &str,
    codec: C,
    widen: F,
    shape: Shape,
    block: usize,
    book_flat: &[i8],
    a: &[i8],
    want: Option<&[i32]>,
    iters: usize,
) -> (Vec<i32>, Row)
where
    C: Enumerable<i8, Full<i8>> + Copy,
    F: Fn(usize) -> C::Code,
{
    let (m, k, n) = (shape.m, shape.k, shape.n);
    let space = C::CODE_SPACE;
    let lane = size_of::<<i8 as Tabulated>::Lane>();
    let plan = plan_for(space, shape, block);
    let codes = stream::<C, F>(n, k, block, widen);

    // The residency terms, printed with the plan they come from. The slab is
    // `slab_codes * rows` lane words --- it does NOT scale with block, which
    // is the fact this phase exists to record: what doubles is the build
    // products per slab and the codebook, what halves is the slots.
    let slab_bytes = slab_codes(space) * plan.rows * lane;
    let fits = tabulation_fits(space, plan.rows, blocking::L1_BYTES, lane);
    let cache_depth = uor_matmul::driver::tabulation_depth(
        space,
        plan.rows,
        block,
        None,
        blocking::L2_BYTES,
        lane,
    );
    let lane_depth = uor_matmul::driver::tabulation_depth(
        space,
        plan.rows,
        block,
        <i8 as Tabulated>::probe_capacity::<<i8 as Tabulated>::Lane>(BOUND),
        blocking::L2_BYTES,
        lane,
    );
    eprintln!(
        "{label} {m}x{k}x{n}: {:.4} B/weight stored, codebook {} B, plan {:?}, slab {} B/slot, \
         stack {} B, L1 fit {fits}, depth cache {cache_depth} vs lane+{lane_depth}",
        (codes.len() * size_of::<C::Code>()) as f64 / (n * k) as f64,
        space * block,
        plan,
        slab_bytes,
        slab_bytes * plan.depth,
    );

    // The resolved spec for this tile, and the per-phase timings.
    let rows = plan.rows;
    let group = column_group(rows);
    let spec = <i8 as Tabulated>::table_spec(Backend::Auto, BOUND, false, rows, group, block);
    let slab_words = slab_codes(space) * rows;
    let acts: Vec<i8> = (0..rows * block)
        .map(|i| ((i * 13) % 5) as i8 - 2)
        .collect();
    let slots = k / block;

    // Build: every slot of the reduction, once per iteration.
    let mut slab = vec![0i32; slab_words];
    let started = Instant::now();
    for _ in 0..iters {
        for _ in 0..slots {
            spec.build(space, block, book_flat, &acts, &mut slab);
        }
    }
    let build_t = started.elapsed();
    // The timed loop ran: the last slot's content is the walk's, recomputed.
    let mut again = vec![0i32; slab_words];
    spec.build(space, block, book_flat, &acts, &mut again);
    assert_eq!(slab, again, "{label}: the build is deterministic");

    // Gather: one full stack's worth of slots over one column block, from the
    // operand's own memory --- the borrowed stream is the point of the width
    // spelling. Bounded to `plan.depth` slabs, so the phase is a rate, not a
    // residency exercise (the e2e below is the whole reduction).
    let depth = plan.depth.min(slots);
    let mut stack = vec![0i32; depth * slab_words];
    for s in 0..depth {
        spec.build(
            space,
            block,
            book_flat,
            &acts,
            &mut stack[s * slab_words..][..slab_words],
        );
    }
    let cpr = slots;
    let mut lane_words = vec![0i32; group * rows];
    let stream = C::as_index_stream(&codes).expect("a 256-entry book borrows its stream");
    let started = Instant::now();
    for _ in 0..iters {
        let mut j = 0usize;
        while j < plan.cols.min(n) {
            let base = j * cpr;
            match stream {
                uor_matmul::codec::IndexStream::U16(s) => spec.gather_codes(
                    depth,
                    slab_words as u32,
                    &stack,
                    &s[base..base + (group - 1) * cpr + depth],
                    cpr,
                    &mut lane_words,
                ),
                uor_matmul::codec::IndexStream::U8(s) => spec.gather_codes_u8(
                    depth,
                    slab_words as u32,
                    &stack,
                    &s[base..base + (group - 1) * cpr + depth],
                    cpr,
                    &mut lane_words,
                ),
            }
            j += group;
        }
    }
    let gather_t = started.elapsed();

    // End to end, with byte-identity against the block-8 product inside the
    // timed loop.
    let started = Instant::now();
    let (mut out, mut census) = e2e(codec, &codes, a, m, k, n, block);
    for _ in 1..iters {
        let (again_out, again_census) = e2e(codec, &codes, a, m, k, n, block);
        if let Some(want) = want {
            assert_eq!(
                again_out, want,
                "{label} {m}x{k}x{n}: the two blocks must agree"
            );
        }
        out = again_out;
        census = again_census;
    }
    let e2e_t = started.elapsed();
    if let Some(want) = want {
        assert_eq!(out, want, "{label} {m}x{k}x{n}: the two blocks must agree");
    }

    eprintln!(
        "{label} {m}x{k}x{n}: build {build_t:?} ({slots} slots/iter), gather {gather_t:?} \
         (one stack, {depth} slots), e2e {e2e_t:?} over {iters} iters; \
         census {census:?}; break-even block {block} at n = {:?} (m = {m})",
        break_even(space, block, m.max(rows))
    );
    (out, Row { census, plan })
}

/// The harness. Ignored: it is a measurement, and measurements here wait for
/// a quiet window. `just block-sweep` runs it; `BLOCK_SWEEP_CHECK=1` is the
/// correctness dry run.
#[test]
#[ignore = "a measurement harness: `just block-sweep`"]
fn block_sweep_prices_the_longer_codeword() {
    let iters = if std::env::var("BLOCK_SWEEP_CHECK").is_ok() {
        1
    } else {
        std::env::var("BLOCK_SWEEP_ITERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(30)
    };
    let check = std::env::var("BLOCK_SWEEP_CHECK").is_ok();

    // The decoded codebooks, flattened for the build phase. The 16-wide
    // book's back half is the block-8 book rotated by half the space, so the
    // two blocks can spell the same decoded matrix.
    let book8 = e8_table::<Full<i8>>().expect("i8 holds E8");
    let mut book16 = [[A8::ZERO; 16]; 256];
    for c in 0..256 {
        book16[c][..8].copy_from_slice(&book8[c]);
        book16[c][8..].copy_from_slice(&book8[(c + 128) % 256]);
    }
    let book8_flat: Vec<i8> = book8.iter().flatten().map(|a| a.get()).collect();
    let book16_flat: Vec<i8> = book16.iter().flatten().map(|a| a.get()).collect();

    let codec8_16 = Book::<i8, Full<i8>, 256, 8>::new(&book8);
    let codec8_8 = Book::<i8, Full<i8>, 256, 8, u8>::new(&book8);
    let codec16_16 = Book::<i8, Full<i8>, 256, 16>::new(&book16);
    let codec16_8 = Book::<i8, Full<i8>, 256, 16, u8>::new(&book16);

    for &(m, k, n) in &SHAPES {
        let shape = Shape { m, k, n };
        let a: Vec<i8> = (0..m * k).map(|i| ((i * 7) % 5) as i8 - 2).collect();

        // Block 8, both spellings. The `u16` run's output is the witness every
        // other configuration is asserted against.
        let (want, row8_16) = run_config(
            "Book<256,8,u16>",
            codec8_16,
            |v| v as u16,
            shape,
            8,
            &book8_flat,
            &a,
            None,
            iters,
        );
        let (_, row8_8) = run_config(
            "Book<256,8,u8>",
            codec8_8,
            |v| v as u8,
            shape,
            8,
            &book8_flat,
            &a,
            Some(&want),
            iters,
        );

        if k % 16 != 0 {
            eprintln!("{m}x{k}x{n}: k is not a whole number of 16-wide blocks; block 16 skipped");
        } else {
            let (_, row16_16) = run_config(
                "Book<256,16,u16>",
                codec16_16,
                |v| v as u16,
                shape,
                16,
                &book16_flat,
                &a,
                Some(&want),
                iters,
            );
            let (_, row16_8) = run_config(
                "Book<256,16,u8>",
                codec16_8,
                |v| v as u8,
                shape,
                16,
                &book16_flat,
                &a,
                Some(&want),
                iters,
            );

            // The census ratios the work order predicts, asserted once in the
            // dry run: reads halve, build products constant, codebook decodes
            // double, the plans identical, and the width spellings' censuses
            // equal field for field.
            if check {
                assert_eq!(
                    row16_16.plan, row8_16.plan,
                    "{m}x{k}x{n}: the slab does not scale with block, so the plan does not move"
                );
                assert_eq!(
                    row16_16.census.table_reads * 2,
                    row8_16.census.table_reads,
                    "{m}x{k}x{n}: reads per decoded weight halve at block 16"
                );
                assert_eq!(
                    row16_16.census.multiplies, row8_16.census.multiplies,
                    "{m}x{k}x{n}: build products are constant per column block"
                );
                assert_eq!(
                    row16_16.census.decodes,
                    row8_16.census.decodes * 2,
                    "{m}x{k}x{n}: the codebook doubles at block 16"
                );
                for (label, wide, narrow) in [
                    ("block 8", row8_16.census, row8_8.census),
                    ("block 16", row16_16.census, row16_8.census),
                ] {
                    assert_eq!(
                        wide, narrow,
                        "{m}x{k}x{n}: {label} census is width-independent"
                    );
                }
            }
        }

        if check {
            assert_eq!(
                row8_16.census, row8_8.census,
                "{m}x{k}x{n}: the block-8 census is width-independent"
            );
        }
    }
}
