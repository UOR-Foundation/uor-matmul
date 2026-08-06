//! `CG-19` (open): the selection lane against the accumulation lane at matched
//! shapes, and the two witness mechanisms against each other.
//!
//! Two measurements, and they answer different questions.
//!
//! The first is the price of the census's second product. `uor_matmul_gemm::gemm`
//! is one function; instantiating `E` at `Trop<i8>` rather than at `i8` is the
//! whole of the change, so the two lanes below are the same traversal, the same
//! offer and the same shape, and what separates their rates is the arithmetic
//! and nothing else. A third column prices the packed ring lane, because a
//! two-column table would invite the reading that the selection lane is slow
//! against *this library's ring path* --- it is being compared against the
//! portable instantiation of one driver, and the packed column is what says how
//! far both of them sit from what a ring caller gets.
//!
//! The second is the price of a witness, and it depends on the operands in a way
//! that has to be shown rather than averaged away. `ComparePass` takes a second
//! pass that stops at the first index attaining the maximum, so its cost is
//! decided by where the maximum sits: on the tie-dense corpus draw the scan
//! stops almost immediately, and on `max-last` --- a unique maximum at the final
//! index of every reduction --- it walks the whole depth. Both are reported.
//! Reporting either alone would be reporting a knob.
//!
//! Every figure here is `open`: measured on one host at one moment, printed,
//! never asserted. What *is* asserted, inside every timed run, is that the
//! answer is right --- a rate measured on the wrong bytes is not a measurement
//! of anything.
//!
//! Ignored by default, like the other release-mode sweeps: `just tropical-sweep`
//! runs it, where a throughput figure means something.

use std::time::Instant;

use uor_matmul::prelude::*;
use uor_matmul::{suggested_scratch, Shape};
use uor_matmul_core::{as_alphabet_tropical, Alphabet, EncodeMode, Full, Trop};
use uor_matmul_gemm::{gemm as gemm_dense, gemm_selected, SelectedTriple, Witness};
use uor_matmul_validate::corpus::{Case, Mask};

/// The corpus seed the selection gates walk, so the sweep prices the operands
/// the gates validated.
const SEED: u64 = 20_260_805;

/// Matched shapes: one cache-resident square, one that is not, one deep and
/// narrow where the reduction dominates, and one gemv.
const SHAPES: &[(usize, usize, usize)] = &[
    (64, 64, 64),
    (256, 256, 256),
    (16, 4096, 16),
    (1, 262_144, 1),
    (512, 8, 512),
];

/// Best of as many repetitions as fit a fixed budget, in seconds per call ---
/// the same discipline every other sweep in this crate uses.
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

/// `max_p (a[i,p] + b[p,j])`, computed from the draw so every timed run can
/// assert its own answer.
fn max_plus_ref(
    m: usize,
    k: usize,
    n: usize,
    a: &[Option<i64>],
    b: &[Option<i64>],
) -> Vec<Trop<i32>> {
    let mut out = Vec::with_capacity(m * n);
    for i in 0..m {
        for j in 0..n {
            let mut best: Option<i64> = None;
            for p in 0..k {
                if let (Some(x), Some(y)) = (a[i * k + p], b[p * n + j]) {
                    best = best.max(Some(x + y));
                }
            }
            out.push(best.map_or(Trop::NEG_INF, |v| Trop::finite(v as i32)));
        }
    }
    out
}

/// The tropical product through the dense driver, timed.
fn time_tropical(
    m: usize,
    k: usize,
    n: usize,
    a: &[Trop<i8>],
    b: &[Trop<i8>],
    want: &[Trop<i32>],
) -> f64 {
    let mut out = vec![Trop::<i32>::NEG_INF; m * n];
    let mut scratch = vec![Alphabet::ZERO; k];
    best(|| {
        let av = MatView::row_major(as_alphabet_tropical(a), m, k).expect("A fits");
        let bv = MatView::row_major(as_alphabet_tropical(b), k, n).expect("B fits");
        let cv = MatViewMut::row_major(&mut out, m, n).expect("C fits");
        let mut t = Triple::new(av, bv, cv).expect("the product exists");
        gemm_dense(
            &mut t,
            &MaxPlus::OVERWRITE,
            GemmOptions::default(),
            &mut Scratch::new(&mut scratch),
        );
        assert_eq!(out, want, "the timed call must be correct");
    })
}

/// The ring product through the *same* driver, timed.
fn time_ring(m: usize, k: usize, n: usize, a: &[i8], b: &[i8], want: &[i32]) -> f64 {
    let mut out = vec![0i32; m * n];
    let mut scratch = vec![Alphabet::<i8, Full<i8>>::ZERO; k];
    best(|| {
        let av = MatView::row_major(as_alphabet_full(a), m, k).expect("A fits");
        let bv = MatView::row_major(as_alphabet_full(b), k, n).expect("B fits");
        let cv = MatViewMut::row_major(&mut out, m, n).expect("C fits");
        let mut t = Triple::new(av, bv, cv).expect("the product exists");
        gemm_dense(
            &mut t,
            &Linear::OVERWRITE,
            GemmOptions {
                encode: EncodeMode::Wrapping,
                ..Default::default()
            },
            &mut Scratch::new(&mut scratch),
        );
        assert_eq!(out, want, "the timed call must be correct");
    })
}

/// The ring product through the packed kernels, timed --- what a ring caller
/// actually gets.
fn time_ring_packed(m: usize, k: usize, n: usize, a: &[i8], b: &[i8], want: &[i32]) -> f64 {
    let mut out = vec![0i32; m * n];
    let mut scratch = vec![Alphabet::<i8, Full<i8>>::ZERO; suggested_scratch(Shape { m, k, n })];
    best(|| {
        let av = MatView::row_major(as_alphabet_full(a), m, k).expect("A fits");
        let bv = MatView::row_major(as_alphabet_full(b), k, n).expect("B fits");
        let cv = MatViewMut::row_major(&mut out, m, n).expect("C fits");
        let mut t = Triple::new(av, bv, cv).expect("the product exists");
        uor_matmul::gemm_packed(
            &mut t,
            &Linear::OVERWRITE,
            GemmOptions {
                encode: EncodeMode::Wrapping,
                ..Default::default()
            },
            &mut Scratch::new(&mut scratch),
        );
        assert_eq!(out, want, "the timed call must be correct");
    })
}

/// The product with its witness, at one mechanism, timed.
fn time_witness(
    m: usize,
    k: usize,
    n: usize,
    a: &[Trop<i8>],
    b: &[Trop<i8>],
    want: &[Trop<i32>],
    mechanism: Witness,
) -> f64 {
    let mut out = vec![Trop::<i32>::NEG_INF; m * n];
    let mut witness = vec![usize::MAX; m * n];
    let mut scratch = vec![Alphabet::ZERO; k];
    best(|| {
        let av = MatView::row_major(as_alphabet_tropical(a), m, k).expect("A fits");
        let bv = MatView::row_major(as_alphabet_tropical(b), k, n).expect("B fits");
        let cv = MatViewMut::row_major(&mut out, m, n).expect("C fits");
        let wv = MatViewMut::row_major(&mut witness, m, n).expect("W fits");
        let t = Triple::new(av, bv, cv).expect("the product exists");
        let mut s = SelectedTriple::new(t, wv).expect("the witness conforms");
        gemm_selected(
            &mut s,
            &MaxPlus::OVERWRITE,
            GemmOptions::default(),
            mechanism,
            &mut Scratch::new(&mut scratch),
        );
        assert_eq!(out, want, "the timed call must be correct");
    })
}

/// `CG-19`: the two tables.
#[test]
#[ignore = "a release-mode sweep: `just tropical-sweep`"]
fn the_selection_lane_against_the_ring_lane_cg_19() {
    let gmacs = |m: usize, k: usize, n: usize, secs: f64| (m * k * n) as f64 / secs / 1e9;
    println!();
    println!("# CG-19 (open): the selection lane against the accumulation lane, Gmac/s");
    println!(
        "# host: {}-{}; best of a 0.35 s budget per point; seed {SEED};",
        std::env::consts::ARCH,
        std::env::consts::OS
    );
    println!("# `tropical` and `ring` are one function --- uor_matmul_gemm::gemm --- at two");
    println!("#   instantiations of `E`, same offer, same traversal, same shape;");
    println!("# `ring packed` is the kernel-selected lane a ring caller gets, for scale;");
    println!("# operands carry no masked lane in this table, because a masked lane performs no");
    println!("#   arithmetic at all and a mask density would be a knob rather than a measurement;");
    println!("# byte-identity with a reference computed off the timed path is asserted inside");
    println!("#   every timed run");
    println!();
    println!("| shape | tropical | ring (same driver) | ring packed |");
    println!("| --- | --- | --- | --- |");
    for &(m, k, n) in SHAPES {
        let case = Case {
            m,
            k,
            n,
            seed: SEED,
        };
        let (da, db) = (
            case.draw_tropical(m * k, 1, Mask::NONE),
            case.draw_tropical(k * n, 2, Mask::NONE),
        );
        let ta = case.fill_tropical_i8(m * k, 1, Mask::NONE);
        let tb = case.fill_tropical_i8(k * n, 2, Mask::NONE);
        let want = max_plus_ref(m, k, n, &da, &db);

        let ra: Vec<i8> = da.iter().map(|v| v.unwrap_or(0) as i8).collect();
        let rb: Vec<i8> = db.iter().map(|v| v.unwrap_or(0) as i8).collect();
        let ring_want = uor_matmul_validate::ours_i8_i32(m, k, n, &ra, &rb, EncodeMode::Wrapping);

        println!(
            "| {m}x{k}x{n} | {:.3} | {:.3} | {:.3} |",
            gmacs(m, k, n, time_tropical(m, k, n, &ta, &tb, &want)),
            gmacs(m, k, n, time_ring(m, k, n, &ra, &rb, &ring_want)),
            gmacs(m, k, n, time_ring_packed(m, k, n, &ra, &rb, &ring_want)),
        );
    }

    println!();
    println!("# CG-19 (open): the witness mechanisms, Gmac/s");
    println!("# `tie-dense` is the corpus draw the gates walk: three values wide and masked, so a");
    println!("#   tie is the common case and the compare pass stops early;");
    println!("# `max-last` places the unique maximum at the final index of every reduction, which");
    println!("#   is where the compare pass walks the whole depth;");
    println!("# the witness bytes of the two mechanisms are identical at every point (`CD-25`),");
    println!("#   which is what makes this a comparison of two factorizations of one answer");
    println!();
    println!("| shape | lexicographic, tie-dense | compare-pass, tie-dense | lexicographic, max-last | compare-pass, max-last |");
    println!("| --- | --- | --- | --- | --- |");
    for &(m, k, n) in SHAPES {
        let case = Case {
            m,
            k,
            n,
            seed: SEED,
        };
        let tie_a = case.fill_tropical_i8(m * k, 1, Mask::LEFT);
        let tie_b = case.fill_tropical_i8(k * n, 2, Mask::RIGHT);
        let tie_want = max_plus_ref(
            m,
            k,
            n,
            &case.draw_tropical(m * k, 1, Mask::LEFT),
            &case.draw_tropical(k * n, 2, Mask::RIGHT),
        );
        let last_a: Vec<Trop<i8>> = (0..m * k)
            .map(|i| Trop::finite(i8::from(i % k == k - 1)))
            .collect();
        let last_b: Vec<Trop<i8>> = (0..k * n)
            .map(|i| Trop::finite(i8::from(i / n == k - 1)))
            .collect();
        let last_want = vec![Trop::<i32>::finite(2); m * n];

        println!(
            "| {m}x{k}x{n} | {:.3} | {:.3} | {:.3} | {:.3} |",
            gmacs(
                m,
                k,
                n,
                time_witness(m, k, n, &tie_a, &tie_b, &tie_want, Witness::Lexicographic)
            ),
            gmacs(
                m,
                k,
                n,
                time_witness(m, k, n, &tie_a, &tie_b, &tie_want, Witness::ComparePass)
            ),
            gmacs(
                m,
                k,
                n,
                time_witness(
                    m,
                    k,
                    n,
                    &last_a,
                    &last_b,
                    &last_want,
                    Witness::Lexicographic
                )
            ),
            gmacs(
                m,
                k,
                n,
                time_witness(m, k, n, &last_a, &last_b, &last_want, Witness::ComparePass)
            ),
        );
    }
    println!();
    println!(
        "# every figure above is `open`: measured on this host at this moment, never asserted"
    );
}
