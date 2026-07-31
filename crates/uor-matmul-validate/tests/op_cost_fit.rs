//! `CG-16` (open): does a linear per-op-kind cost model fit the measured
//! times of the forced symbol table, the placement bridge, and the dense
//! float driver --- and if it does, does a selection boundary with a safety
//! margin separate every measured win from every measured loss?
//!
//! This is the feasibility machinery and nothing else: the predicate, the
//! model, and the drivers are untouched, and the decision the numbers argue
//! for or against is made after they exist. The design matrix is the op-kind
//! counts --- the table's read off its census (`tabulated.rs`), the dense
//! side's derived from the drivers' own arithmetic --- the targets are
//! measured seconds, and the unknowns are nanoseconds per op kind. A kind the
//! grid cannot identify is merged or reported, never priced.
//!
//! What is asserted, inside every timed run, is byte-identity with the dense
//! float driver (`CD-19`, `CD-20`); every figure printed is `open`. Three
//! driver predicates are re-stated here rather than imported --- the span
//! walk (`Span::see`), the bridge's admission (`admits_bridge`), and the
//! offer question (`bridge_room`) are `pub(crate)` in `float.rs`, and
//! widening the shipped API for a measurement was the larger change. Each
//! re-statement is cited where it stands; a classification that drifted from
//! the driver's shows up in the affected path's residuals, which are printed.
//!
//! Ignored by default, like the other minute-long sweeps: `just op-cost-fit`
//! runs it, in release, where a throughput figure means something.

mod common;

use common::{best, fill, spanned, wide, SHAPES};

use uor_matmul_codec::{Arena, CodedMatrix};
use uor_matmul_core::{
    as_alphabet_whole, Alphabet, FloatElement, Full, MatView, MatViewMut, PackedCode, Shape,
    Traversal, Triple, Whole,
};
use uor_matmul_gemm::tabulated::Plan;
use uor_matmul_gemm::{
    gemm_float_bridged, gemm_float_packed, gemm_tabulated_counted, suggested_accumulators,
    suggested_bridge_scaled, suggested_scratch, suggested_tabulation, suggested_tabulation_index,
    suggested_tabulation_lanes, suggested_tabulation_panel, Census, Collapse, GemmOptions,
    Kernelized, Linear, Scratch, TabulatedTriple, Tabulation,
};

/// The op kinds the model prices, in count-vector order. The gather's read
/// and its add are one kind: the census charges them together at every gather
/// step, so no grid could price them apart.
const KINDS: [&str; 6] = [
    "table build product",
    "gather read+add",
    "decode",
    "dense lane product",
    "bridge reify word",
    "chunk extraction",
];
const BUILD: usize = 0;
const GATHER: usize = 1;
const DECODE: usize = 2;
const PRODUCT: usize = 3;
const REIFY: usize = 4;
const EXTRACT: usize = 5;

/// One measured path on one point: its best-of seconds and its op-kind
/// counts, census or analytic.
#[derive(Clone, Copy)]
struct Timed {
    secs: f64,
    counts: [f64; 6],
}

/// One grid point: a (fill, shape) pair and what each path did there. A path
/// is `None` where it declined or was never asked, and the point is excluded
/// from that path's fit rows --- reported, not silently dropped.
struct Point {
    fill: &'static str,
    table: Option<Timed>,
    bridge: Option<Timed>,
    bridge_note: &'static str,
    dense: Timed,
    dense_class: &'static str,
}

/// One operand's packed-exponent width and finiteness: the walk `Span::see`
/// is (`float.rs`), re-stated because `Span` is `pub(crate)`. The width is
/// the max minus the min packed exponent over the nonzero codes.
fn span_of(data: &[f32]) -> (u32, bool) {
    let mut finite = true;
    let mut any = false;
    let mut lo = i32::MAX;
    let mut hi = i32::MIN;
    for &v in data {
        let code = v.pack();
        finite &= code.is_finite();
        if code.mantissa != 0 {
            lo = lo.min(code.exp);
            hi = hi.max(code.exp);
            any = true;
        }
    }
    (if any { hi.wrapping_sub(lo) as u32 } else { 0 }, finite)
}

/// `admits_bridge` re-stated: the scaled significand must be an element of
/// the `i32` alphabet, whose magnitude reaches 31 bits. Returns the declared
/// bound the lane depth is then read at.
fn bridge_admits(wa: u32, wb: u32) -> Option<u128> {
    let p = <f32 as FloatElement>::SIGNIFICAND_BITS;
    const ALPHABET: u32 = i32::BITS - 1;
    if p.checked_add(wa)? > ALPHABET || p.checked_add(wb)? > ALPHABET {
        return None;
    }
    Some(1u128 << (p + wa.max(wb)))
}

/// The scaled lane's run in products, as the walk derives it
/// (`tabulated.rs`'s `lane_scale` and `lane_run`): one lane word is an
/// `i64`, and the worst one-product magnitude is `2^(2p + wa + wb)`.
fn lane_run(wa: u32, wb: u32) -> usize {
    let p = <f32 as FloatElement>::SIGNIFICAND_BITS;
    let per_step = 1u128.checked_shl(2 * p + wa + wb).unwrap_or(u128::MAX);
    usize::try_from((i64::MAX as u128) / per_step)
        .unwrap_or(usize::MAX)
        .max(1)
}

/// Placements of the carried lane words into the exact accumulator over one
/// reduction of `k / block` blocks. The census does not count `place`
/// (`tabulated.rs`), so the count is derived where the chunking happens: the
/// stack depth is the walk-shrunk plan depth, and a placement is issued each
/// time another chunk would not fit the lane run, plus the final one. This is
/// the row tile's own loop, arithmetic only.
fn placements(plan: Plan, k: usize, block: usize, run: usize) -> u64 {
    let blocks = k / block;
    let run_blocks = (run / block).max(1);
    let depth = (run / block).min(plan.depth).max(1);
    let mut p0 = 0usize;
    let mut in_run = 0usize;
    let mut placed = 0u64;
    while p0 < blocks {
        let d = depth.min(blocks - p0);
        in_run += d;
        p0 += d;
        if in_run + depth > run_blocks {
            placed += 1;
            in_run = 0;
        }
    }
    placed + 1
}

/// The three timed paths on one grid point, with their count vectors.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_lines)]
fn measure(
    fill_name: &'static str,
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    w: CodedMatrix<'_, f32, Whole<f32>, Arena<'_, f32, 256, u8>>,
    b: &[f32],
    want: &[f32],
    wb_book: u32,
) -> Point {
    let shape = Shape { m, k, n };
    let (wa, finite_a) = span_of(a);
    let (wb_dense, finite_b) = span_of(b);

    // The forced table. The offers are the suggestions, so the plan the
    // traversal resolves to is the one `Plan::choose` computes over the same
    // numbers, and the census of one run says what ran.
    let mut c_table = vec![0.0f32; m * n];
    let mut panel = vec![Alphabet::<f32, Whole<f32>>::ZERO; suggested_tabulation_panel(256, 1)];
    let mut accumulators = vec![
        <uor_matmul_core::AccOf<f32> as uor_matmul_core::Accumulator>::ZERO;
        suggested_tabulation::<f32, Whole<f32>>(shape, 256, 1).max(1)
    ];
    let mut lane_words =
        vec![0i64; suggested_tabulation_lanes::<f32, Whole<f32>>(shape, 256, 1).max(1)];
    let mut ids = vec![0usize; suggested_tabulation_index(shape)];
    let mut census = Census::default();
    let mut run_table = |c_table: &mut [f32], census: &mut Census| {
        let av = MatView::row_major(as_alphabet_whole(a), m, k).unwrap();
        let cv = MatViewMut::row_major(c_table, m, n).unwrap();
        let mut tr = TabulatedTriple::new(av, w, cv).unwrap();
        gemm_tabulated_counted(
            &mut tr,
            &Linear::OVERWRITE,
            GemmOptions {
                traversal: Traversal::Tabulated,
                ..Default::default()
            },
            &mut Scratch::with_accumulators(&mut panel, &mut accumulators),
            &mut Tabulation::with_index(&mut lane_words, &mut ids),
            &mut Collapse::none(),
            census,
        );
    };
    run_table(&mut c_table, &mut census);
    // The census of one run, read before the timed reps accumulate into it.
    let one_run = census;
    let table_secs = best(|| {
        run_table(&mut c_table, &mut census);
        // Byte-identity with the dense float driver, inside the timed region
        // (`CD-20`).
        assert_eq!(
            c_table.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "the symbol table must give the dense driver's bytes at {m}x{k}x{n}"
        );
    });
    let table = if one_run.table_reads > 0 {
        // The census counts what the traversal issued; what it does not count
        // is `place`, so the chunk extractions are derived from the plan the
        // same offers resolve to and the walk's lane run.
        let plan = Plan::choose(
            256,
            shape,
            size_of::<i64>(),
            accumulators.len(),
            lane_words.len(),
            1,
            None,
        );
        let run = lane_run(wa, wb_book);
        let mut counts = [0.0; 6];
        counts[BUILD] = one_run.multiplies as f64;
        counts[GATHER] = one_run.table_reads as f64;
        counts[DECODE] = one_run.decodes as f64;
        counts[EXTRACT] = plan.map_or(0.0, |p| (m * n) as f64 * placements(p, k, 1, run) as f64);
        // The bound-1 build is the only one issued as adds (`CB-10`), and
        // this lane's build is not it, so every add is the gather's and the
        // two gather counts are one kind by construction.
        assert_eq!(
            one_run.adds, one_run.table_reads,
            "{m}x{k}x{n}: the gather's adds and reads parted ways ({one_run:?})"
        );
        Some(Timed {
            secs: table_secs,
            counts,
        })
    } else {
        None
    };

    // The bridge, on the points its own offer question and admission walk
    // would let it run: elsewhere `gemm_float_bridged` is the scalar lanes
    // wearing a second name, and timing it would price the wrong path.
    let worth_asking = m * n > m + n;
    let bridge_admitted =
        worth_asking && finite_a && finite_b && bridge_admits(wa, wb_dense).is_some();
    let mut pa = vec![PackedCode::default(); k.max(1)];
    let mut pb = vec![PackedCode::default(); k * n];
    let (bridge, bridge_note) = if bridge_admitted {
        let mut c_bridged = vec![0.0f32; m * n];
        let mut scaled = vec![0i32; suggested_bridge_scaled(shape)];
        let mut kernel_buf = vec![Alphabet::<i32, Full<i32>>::of(0); suggested_scratch(shape)];
        let mut acc_buf = vec![0i128; suggested_accumulators(shape)];
        let bridged = best(|| {
            let av = MatView::row_major(a, m, k).unwrap();
            let bv = MatView::row_major(b, k, n).unwrap();
            let cv = MatViewMut::row_major(&mut c_bridged, m, n).unwrap();
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
            assert_eq!(
                c_bridged.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                want.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                "the bridge must give the dense driver's bytes at {m}x{k}x{n}"
            );
        });
        // The walk's decodes, one reified word per panel element (each its
        // own decode, folded into the reify kind), and the integer table's
        // products.
        let mut counts = [0.0; 6];
        counts[DECODE] = (k * (m + n)) as f64;
        counts[REIFY] = (k * (m + n)) as f64;
        counts[PRODUCT] = (m * n * k) as f64;
        (
            Some(Timed {
                secs: bridged,
                counts,
            }),
            "ran",
        )
    } else {
        (
            None,
            if worth_asking {
                "declined: the spans do not fit the i32 alphabet"
            } else {
                "not walked: m*n <= m+n"
            },
        )
    };

    // The dense float driver, which is what selection runs today. Its count
    // vector follows the route `gemm_float_packed` resolves to, re-derived
    // from the same declarations the driver reads: the walk is paid on every
    // two-dimensional shape (for the bridge, or for prescaling when the
    // bridge declines), the bridge runs when its whole admission chain
    // holds, and a matrix-vector product is never walked.
    let mut c_dense = vec![0.0f32; m * n];
    let dense = best(|| {
        let av = MatView::row_major(a, m, k).unwrap();
        let bv = MatView::row_major(b, k, n).unwrap();
        let cv = MatViewMut::row_major(&mut c_dense, m, n).unwrap();
        let mut t = Triple::new(av, bv, cv).unwrap();
        gemm_float_packed(
            &mut t,
            &Linear::OVERWRITE,
            GemmOptions::default(),
            &mut pa,
            &mut pb,
        );
    });
    assert_eq!(
        c_dense.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        want.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        "the dense run must give its own first bytes at {m}x{k}x{n}"
    );
    let dense_bridged = bridge_admitted && {
        // The packed entry's own chain, past the walk: the lane must hold the
        // whole depth and the panels must hold the reification plus a
        // full-depth tile pair (`gemm_float_packed`, `float.rs`).
        let bound = bridge_admits(wa, wb_dense).expect("admission just held");
        let spec = <i32 as Kernelized>::exact_spec(GemmOptions::default().backend, bound, m);
        k <= spec.lane_depth(bound)
            && 4 * k * n >= suggested_bridge_scaled(shape) + k * (spec.mr + spec.nr)
    };
    let (dense_class, dense_counts) = if !worth_asking {
        let mut counts = [0.0; 6];
        counts[DECODE] = (m * k + k * n) as f64;
        counts[PRODUCT] = (m * n * k) as f64;
        ("scalar lanes, no walk", counts)
    } else if dense_bridged {
        let mut counts = [0.0; 6];
        counts[DECODE] = (k * (m + n)) as f64;
        counts[REIFY] = (k * (m + n)) as f64;
        counts[PRODUCT] = (m * n * k) as f64;
        ("packed bridge", counts)
    } else {
        let mut counts = [0.0; 6];
        counts[DECODE] = (k * (m + n) + m * k + k * n) as f64;
        counts[PRODUCT] = (m * n * k) as f64;
        ("scalar lanes, walk paid", counts)
    };

    Point {
        fill: fill_name,
        table,
        bridge,
        bridge_note,
        dense: Timed {
            secs: dense,
            counts: dense_counts,
        },
        dense_class,
    }
}

/// One fit row: a count vector, a measured time, and which path produced it.
struct Row {
    counts: [f64; 6],
    secs: f64,
    path: usize,
}

/// Solve `A x = b` by Gaussian elimination with partial pivoting, hand-rolled
/// over `f64`: six unknowns do not buy a dependency. `None` on a singular
/// pivot, which the caller reads as "this unknown is not identifiable" and
/// drops.
fn solve(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    let mut m: Vec<Vec<f64>> = a.to_vec();
    let mut x = b.to_vec();
    let max_diag = (0..n).map(|i| m[i][i].abs()).fold(0.0, f64::max);
    for col in 0..n {
        let piv = (col..n).max_by(|&i, &j| m[i][col].abs().total_cmp(&m[j][col].abs()))?;
        if m[piv][col].abs() < 1e-12 * max_diag {
            return None;
        }
        m.swap(col, piv);
        x.swap(col, piv);
        for row in col + 1..n {
            let f = m[row][col] / m[col][col];
            let (head, tail) = m.split_at_mut(row);
            for (rc, &cc) in tail[0][col..].iter_mut().zip(&head[col][col..]) {
                *rc -= f * cc;
            }
            x[row] -= f * x[col];
        }
    }
    let mut out = vec![0.0; n];
    for col in (0..n).rev() {
        let mut s = x[col];
        for c in col + 1..n {
            s -= m[col][c] * out[c];
        }
        out[col] = s / m[col][col];
    }
    Some(out)
}

/// The fit, and what the grid could not say.
struct Fit {
    /// Nanoseconds per op kind; zero where the kind was dropped.
    beta: [f64; 6],
    /// Kinds with no nonzero count on any row.
    unpriced: Vec<usize>,
    /// `(dropped, kept, ratio)`: the dropped kind's counts are `ratio` times
    /// the kept kind's on every row, so the kept constant prices both.
    merged: Vec<(usize, usize, f64)>,
    /// Kinds dropped when the normal equations stayed singular after merging.
    singular: Vec<usize>,
}

/// Least squares by the normal equations, on columns scaled to unit maximum
/// first --- the counts span nine orders of magnitude and the normal
/// equations square the conditioning. A column the grid cannot identify
/// (zero everywhere, a scalar multiple of another everywhere, or singular
/// past that) is dropped and reported, never priced.
fn fit(rows: &[Row]) -> Fit {
    let kinds = KINDS.len();
    let mut unpriced = Vec::new();
    let mut keep: Vec<usize> = Vec::new();
    for j in 0..kinds {
        if rows.iter().any(|r| r.counts[j] != 0.0) {
            keep.push(j);
        } else {
            unpriced.push(j);
        }
    }
    let mut scale = [1.0f64; 6];
    for &j in &keep {
        scale[j] = rows.iter().map(|r| r.counts[j].abs()).fold(0.0, f64::max);
    }
    // Exact collinearity, merged pairwise: kind `j` whose scaled column is a
    // scalar multiple of an earlier kept column on every row adds its
    // constant to that kind's and leaves the fit.
    let mut merged: Vec<(usize, usize, f64)> = Vec::new();
    let mut j = 0;
    while j < keep.len() {
        let mut parted = false;
        for i in 0..j {
            let (ci, cj) = (keep[i], keep[j]);
            let mut num = 0.0;
            let mut den = 0.0;
            for r in rows {
                let xi = r.counts[ci] / scale[ci];
                let xj = r.counts[cj] / scale[cj];
                num += xi * xj;
                den += xi * xi;
            }
            let alpha = num / den;
            let resid = rows
                .iter()
                .map(|r| (r.counts[cj] / scale[cj] - alpha * r.counts[ci] / scale[ci]).abs())
                .fold(0.0, f64::max);
            if resid < 1e-9 {
                merged.push((cj, ci, alpha * scale[cj] / scale[ci]));
                keep.remove(j);
                parted = true;
                break;
            }
        }
        if !parted {
            j += 1;
        }
    }
    // The normal equations over what is left, dropping a column each time the
    // solve finds the system singular.
    let mut singular = Vec::new();
    let gamma = loop {
        let kk = keep.len();
        let mut a = vec![vec![0.0; kk]; kk];
        let mut rhs = vec![0.0; kk];
        for r in rows {
            for (i, &ci) in keep.iter().enumerate() {
                let xi = r.counts[ci] / scale[ci];
                rhs[i] += xi * r.secs;
                for (jj, &cj) in keep.iter().enumerate().take(i + 1) {
                    a[i][jj] += xi * r.counts[cj] / scale[cj];
                }
            }
        }
        for i in 1..kk {
            let (upper, lower) = a.split_at_mut(i);
            for (jj, row) in upper.iter_mut().enumerate() {
                row[i] = lower[0][jj];
            }
        }
        match solve(&a, &rhs) {
            Some(g) => break g,
            None => {
                singular.push(keep.pop().expect("a kind to drop"));
                if keep.is_empty() {
                    break Vec::new();
                }
            }
        }
    };
    let mut beta = [0.0; 6];
    for (i, &ci) in keep.iter().enumerate() {
        // `gamma` fits seconds against the scaled counts, so this is seconds
        // per op; the printed and predicted unit is nanoseconds.
        beta[ci] = gamma[i] / scale[ci] * 1e9;
    }
    Fit {
        beta,
        unpriced,
        merged,
        singular,
    }
}

/// Predicted seconds of a count vector under the fitted constants.
fn predicted(beta: &[f64; 6], counts: &[f64; 6]) -> f64 {
    counts
        .iter()
        .zip(beta.iter())
        .map(|(c, b)| c * b)
        .sum::<f64>()
        * 1e-9
}

/// The decisive output. Over the points where both sides measured, for a
/// sweep of candidate safety margins: does "select the table iff
/// predicted_table * margin < predicted_other" keep every measured win and
/// exclude every measured loss? Prints the largest margin that keeps the
/// wins, the smallest that excludes the losses, and whether the intervals
/// overlap --- when they do not, that is the finding, stated plainly.
fn separation(name: &str, pairs: &[(f64, f64, [f64; 6], [f64; 6])], beta: &[f64; 6]) {
    const MARGINS: [f64; 9] = [1.0, 1.05, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0, 3.0];
    println!();
    println!("# selection boundary, table against {name}: margin sweep");
    println!("| margin | wins kept | losses excluded | separates |");
    println!("| --- | --- | --- | --- |");
    let mut ratios: Vec<(bool, f64)> = Vec::new();
    let mut dropped = 0usize;
    for &(t_secs, o_secs, ref t_counts, ref o_counts) in pairs {
        let pt = predicted(beta, t_counts);
        let po = predicted(beta, o_counts);
        if pt > 0.0 && po > 0.0 {
            ratios.push((t_secs < o_secs, po / pt));
        } else {
            // A non-positive prediction prices a side at nothing; a boundary
            // read from a subset that excludes it would be a fiction, so the
            // drop is counted and printed, never silent.
            dropped += 1;
        }
    }
    if dropped > 0 {
        println!("# {dropped} of {} points dropped: the fitted model predicts a non-positive time for them", pairs.len());
    }
    let wins = ratios.iter().filter(|r| r.0).count();
    let losses = ratios.len() - wins;
    for &mu in &MARGINS {
        let kept = ratios.iter().filter(|r| r.0 && mu < r.1).count();
        let excluded = ratios.iter().filter(|r| !r.0 && mu >= r.1).count();
        let separates = kept == wins && excluded == losses;
        println!(
            "| {mu:.2} | {kept}/{wins} | {excluded}/{losses} | {} |",
            if separates { "yes" } else { "no" }
        );
    }
    let mu_hi = ratios
        .iter()
        .filter(|r| r.0)
        .map(|r| r.1)
        .fold(f64::INFINITY, f64::min);
    let mu_lo = ratios
        .iter()
        .filter(|r| !r.0)
        .map(|r| r.1)
        .fold(0.0, f64::max);
    println!("# table against {name}:");
    if wins == 0 {
        println!("#   the grid holds no measured win, so every margin keeps them all");
    } else {
        println!("#   largest margin keeping every measured win:   {mu_hi:.3}");
    }
    if losses == 0 {
        println!("#   the grid holds no measured loss, so every margin excludes them all");
    } else {
        println!("#   smallest margin excluding every measured loss: {mu_lo:.3}");
    }
    if wins > 0 && losses > 0 {
        if mu_lo < mu_hi {
            println!(
                "#   the intervals overlap: any margin in [{mu_lo:.3}, {mu_hi:.3}) separates \
                 every measured win from every measured loss"
            );
        } else {
            println!(
                "#   THE INTERVALS DO NOT OVERLAP ({mu_lo:.3} >= {mu_hi:.3}): no safety margin \
                 separates the measured wins from the measured losses under this cost model --- \
                 that is the \"nothing\" outcome, reported"
            );
        }
    }
}

#[test]
#[ignore = "a minutes-long release-mode sweep: `just op-cost-fit`"]
fn a_per_op_kind_cost_model_against_the_measured_grid_cg_16() {
    // The committed corpus's codebook: 256 distinct f32 bit patterns spanning
    // seven binades --- exactly the widest span the lane's alphabet admits at
    // `f32` (`24 + 7 <= 31`), so the design point stands on the admission
    // boundary. Its span is the table lane's `wb`: the lane's walk is over
    // the codebook, not over the decoded operand.
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("two below the root")
        .join("oracles/symbols");
    let codebook: Vec<f32> = std::fs::read(dir.join("codebook.f32.bin"))
        .expect("the committed codebook")
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("four bytes")))
        .collect();
    let table_book: &[Alphabet<f32, Whole<f32>>; 256] =
        as_alphabet_whole(&codebook).try_into().unwrap();
    let (wb_book, finite_book) = span_of(&codebook);
    assert!(finite_book, "the committed codebook must be finite");

    println!();
    println!("# CG-16 op-cost fit (open): a linear per-op-kind cost model against the clock");
    println!(
        "# host: {}-{}; best of a 0.35 s budget per point; byte-identity with the dense",
        std::env::consts::ARCH,
        std::env::consts::OS
    );
    println!("#   float driver asserted inside every timed run; every figure printed is open");
    println!("# codebook span: {wb_book} binades of packed-exponent width");

    let mut points: Vec<Point> = Vec::new();
    let mut rows: Vec<Row> = Vec::new();
    for (label, span_a, admitted) in [
        ("one exponent", 0, true),
        ("a few binades", 3, true),
        ("wide spans (~18, the lane declines)", 18, false),
    ] {
        for &(m, k, n) in SHAPES {
            let a: Vec<f32> = if admitted {
                spanned(m * k, 5, span_a)
            } else {
                wide(m * k, 5)
            };
            let codes: Vec<u8> = fill(n * k, 6, |x| x as u8);
            let b: Vec<f32> = (0..k * n)
                .map(|at| codebook[codes[(at % n) * k + at / n] as usize])
                .collect();
            let w = CodedMatrix::new(Arena::new(table_book), n, k, &codes)
                .expect("the codes describe n x k");
            // The reference bytes, computed once, untimed, by the dense float
            // driver over the dense spelling.
            let mut want = vec![0.0f32; m * n];
            {
                let av = MatView::row_major(&a, m, k).unwrap();
                let bv = MatView::row_major(&b, k, n).unwrap();
                let cv = MatViewMut::row_major(&mut want, m, n).unwrap();
                let mut t = Triple::new(av, bv, cv).unwrap();
                let mut pa = vec![PackedCode::default(); k.max(1)];
                let mut pb = vec![PackedCode::default(); k * n];
                gemm_float_packed(
                    &mut t,
                    &Linear::OVERWRITE,
                    GemmOptions::default(),
                    &mut pa,
                    &mut pb,
                );
            }
            let point = measure(label, m, k, n, &a, w, &b, &want, wb_book);
            println!();
            println!("| {} | {m}x{k}x{n} |", point.fill);
            match &point.table {
                Some(t) => {
                    println!(
                        "#   table   {:>12.3} us   build {:.3e}  gather {:.3e}  decode {:.3e}  extract {:.3e}",
                        t.secs * 1e6,
                        t.counts[BUILD],
                        t.counts[GATHER],
                        t.counts[DECODE],
                        t.counts[EXTRACT],
                    );
                    rows.push(Row {
                        counts: t.counts,
                        secs: t.secs,
                        path: 0,
                    });
                }
                None => println!("#   table   declined: the fill is past the lane's alphabet"),
            }
            match &point.bridge {
                Some(t) => {
                    println!(
                        "#   bridge  {:>12.3} us   decode {:.3e}  reify {:.3e}  product {:.3e}",
                        t.secs * 1e6,
                        t.counts[DECODE],
                        t.counts[REIFY],
                        t.counts[PRODUCT],
                    );
                    rows.push(Row {
                        counts: t.counts,
                        secs: t.secs,
                        path: 1,
                    });
                }
                None => println!("#   bridge  excluded ({})", point.bridge_note),
            }
            println!(
                "#   dense   {:>12.3} us   ({})   decode {:.3e}  reify {:.3e}  product {:.3e}",
                point.dense.secs * 1e6,
                point.dense_class,
                point.dense.counts[DECODE],
                point.dense.counts[REIFY],
                point.dense.counts[PRODUCT],
            );
            rows.push(Row {
                counts: point.dense.counts,
                secs: point.dense.secs,
                path: 2,
            });
            points.push(point);
        }
    }

    let fitted = fit(&rows);
    println!();
    println!("# fitted per-op-kind constants, over {} points", rows.len());
    println!("| op kind | ns per op | rows priced |");
    println!("| --- | --- | --- |");
    for (j, kind) in KINDS.iter().enumerate() {
        let priced = rows.iter().filter(|r| r.counts[j] != 0.0).count();
        if fitted.unpriced.contains(&j) {
            println!("| {kind} | (no row prices it) | 0 |");
        } else if fitted.singular.contains(&j) {
            println!("| {kind} | (not identifiable: singular) | {priced} |");
        } else {
            println!("| {kind} | {:.4} | {priced} |", fitted.beta[j]);
        }
    }
    for &(dropped, kept, ratio) in &fitted.merged {
        println!(
            "# merged: `{}`'s counts are {ratio:.4}x `{}`'s on every row; the kept constant prices both",
            KINDS[dropped], KINDS[kept]
        );
    }

    for (path, name) in [(0usize, "table"), (1, "bridge"), (2, "dense")] {
        let rel: Vec<f64> = rows
            .iter()
            .filter(|r| r.path == path)
            .map(|r| (predicted(&fitted.beta, &r.counts) - r.secs).abs() / r.secs)
            .collect();
        let max = rel.iter().copied().fold(0.0, f64::max);
        let mean = rel.iter().sum::<f64>() / rel.len() as f64;
        println!(
            "# {name}: {} points, max relative residual {max:.4}, mean {mean:.4}",
            rel.len()
        );
    }

    // The decisive comparison: measured win is the table faster than the
    // measured other side, on the same point. The dense side is what
    // selection runs today; the bridge is the factorization the table would
    // displace where the bridge is admissible.
    let dense_pairs: Vec<(f64, f64, [f64; 6], [f64; 6])> = points
        .iter()
        .filter_map(|p| {
            p.table
                .map(|t| (t.secs, p.dense.secs, t.counts, p.dense.counts))
        })
        .collect();
    separation("the dense float driver", &dense_pairs, &fitted.beta);
    let bridge_pairs: Vec<(f64, f64, [f64; 6], [f64; 6])> = points
        .iter()
        .filter_map(|p| {
            p.table
                .zip(p.bridge)
                .map(|(t, b)| (t.secs, b.secs, t.counts, b.counts))
        })
        .collect();
    separation("the placement bridge", &bridge_pairs, &fitted.beta);
}
