//! `CD-24`, `CD-25`: the selection witness over the forced-tie corpus.
//!
//! `uor-matmul-gemm` already gates both claims against a small fixture built
//! inside that crate. This file is the differential half, and it differs in the
//! two ways that make it worth having.
//!
//! It walks [`Corpus::tropical`], whose draw is three values wide **because** a
//! witness is only observable where two terms tie: measured, that corpus ties
//! 4586 of its 17656 cells and the ring corpus's wide draw through the same
//! masks ties 52. A gate over the wide draw would be green and would decide
//! almost nothing.
//!
//! And it compares against a reference written here, from the corpus's own
//! draw, rather than against the other mechanism. Two mechanisms agreeing is
//! `CD-25` and is worth asserting; it is not evidence that either is right,
//! because they share a module, a tie-break and an author. The reference below
//! shares none of that: it is a leftmost-argmax over a list, which is what D-6
//! says in the words D-6 says it in (R6).

use uor_matmul_core::{
    as_alphabet_tropical, Alphabet, MatView, MatViewMut, Traversal, Triple, Trop,
};
use uor_matmul_gemm::{gemm_selected, GemmOptions, MaxPlus, Scratch, SelectedTriple, Witness};
use uor_matmul_validate::corpus::{Corpus, Mask};
use uor_matmul_validate::Case;

/// The seed every selection gate walks the corpus at, and the one the committed
/// `CX-11` artifact was drawn at.
const SEED: u64 = 20_260_805;

/// Every partition of a reduction the offer can buy, including none.
///
/// `0` is the streaming traversal's panel and `1` is the degenerate partition
/// where every term is its own panel --- the two ends `CD-24` has to hold at.
const PANELS: [usize; 8] = [0, 1, 2, 3, 5, 64, 4096, 5000];

/// The reference: `(the maximum, the least index attaining it)`, per cell.
///
/// Written against the drawn values and nothing else. `None` is the semiring
/// zero, and a reduction of every-term-`None` has maximum `None`, which index 0
/// attains --- so the two degeneracies fall out of the general statement rather
/// than being handled.
fn reference(
    m: usize,
    k: usize,
    n: usize,
    a: &[Option<i64>],
    b: &[Option<i64>],
) -> Vec<(Option<i64>, usize)> {
    let term = |i: usize, j: usize, p: usize| match (a[i * k + p], b[p * n + j]) {
        (Some(x), Some(y)) => Some(x + y),
        _ => None,
    };
    let mut out = Vec::with_capacity(m * n);
    for i in 0..m {
        for j in 0..n {
            // The maximum, then the first index attaining it. Two passes,
            // deliberately: "the least index attaining the maximum" is the
            // definition, and writing it as one pass would be writing the
            // mechanism under test.
            let mut best = None;
            for p in 0..k {
                if best.is_none() || term(i, j, p) > best {
                    best = best.max(term(i, j, p));
                }
            }
            let at = (0..k).find(|&p| term(i, j, p) == best).unwrap_or(0);
            out.push((best, at));
        }
    }
    out
}

/// One product with its witness, at a mechanism, a traversal and an offer.
fn run(
    case: &Case,
    mechanism: Witness,
    traversal: Traversal,
    panel: usize,
) -> (Vec<Trop<i32>>, Vec<usize>) {
    let Case { m, k, n, .. } = *case;
    let a = case.fill_tropical_i8(m * k, 1, Mask::LEFT);
    let b = case.fill_tropical_i8(k * n, 2, Mask::RIGHT);
    // Filled with a *finite* value, not the semiring zero, so that an epilogue
    // that read `C` when it declared it would not read the answer it was about
    // to write (`CS-11`).
    let mut c = vec![Trop::<i32>::finite(0); m * n];
    let mut w = vec![usize::MAX; m * n];
    let mut buf = vec![Alphabet::ZERO; panel];
    {
        let av = MatView::row_major(as_alphabet_tropical(&a), m, k).expect("A fits");
        let bv = MatView::row_major(as_alphabet_tropical(&b), k, n).expect("B fits");
        let cv = MatViewMut::row_major(&mut c, m, n).expect("C fits");
        let wv = MatViewMut::row_major(&mut w, m, n).expect("W fits");
        let t = Triple::new(av, bv, cv).expect("the product exists");
        let mut s = SelectedTriple::new(t, wv).expect("the witness conforms");
        gemm_selected(
            &mut s,
            &MaxPlus::OVERWRITE,
            GemmOptions {
                traversal,
                ..Default::default()
            },
            mechanism,
            &mut Scratch::new(&mut buf),
        );
    }
    (c, w)
}

/// `CD-24`: the witness is the least index attaining the maximum, at every
/// shape in the corpus and under every partition the offer can buy.
#[test]
fn the_witness_is_the_least_index_at_every_shape_cd_24() {
    let mut cells = 0usize;
    for case in Corpus::tropical(SEED).cases {
        let Case { m, k, n, .. } = case;
        let want = reference(
            m,
            k,
            n,
            &case.draw_tropical(m * k, 1, Mask::LEFT),
            &case.draw_tropical(k * n, 2, Mask::RIGHT),
        );
        for panel in PANELS {
            for mechanism in [Witness::Lexicographic, Witness::ComparePass] {
                let (c, w) = run(&case, mechanism, Traversal::Blocked, panel);
                for (cell, &(value, at)) in want.iter().enumerate() {
                    assert_eq!(
                        c[cell],
                        value.map_or(Trop::NEG_INF, |v| Trop::finite(v as i32)),
                        "{m}x{k}x{n} cell {cell}, {mechanism:?}, panel {panel}: the value"
                    );
                    assert_eq!(
                        w[cell], at,
                        "{m}x{k}x{n} cell {cell}, {mechanism:?}, panel {panel}: the witness"
                    );
                }
            }
        }
        cells += want.len();
    }
    assert!(cells > 10_000, "the corpus must have cells: {cells}");
    eprintln!("CD-24: {cells} cells, {} partitions each", PANELS.len() * 2);
}

/// `CD-25`: both mechanisms give identical bytes, at every shape, degeneracy
/// and offer including none --- and the streaming traversal, which buys no
/// panel at all, gives the same bytes again.
#[test]
fn both_mechanisms_agree_at_every_offer_cd_25() {
    for case in Corpus::tropical(SEED).cases {
        let Case { m, k, n, .. } = case;
        let stream = run(&case, Witness::Lexicographic, Traversal::OutputMajor, 0);
        for panel in PANELS {
            let lex = run(&case, Witness::Lexicographic, Traversal::Blocked, panel);
            let cmp = run(&case, Witness::ComparePass, Traversal::Blocked, panel);
            assert_eq!(lex, cmp, "{m}x{k}x{n}, panel {panel}: the two mechanisms");
            assert_eq!(lex, stream, "{m}x{k}x{n}, panel {panel}: against streaming");
        }
    }
}

/// The reference must be able to disagree, or the two tests above compare a
/// witness against a witness.
///
/// The tie-break is what is checked, in both directions: a reduction whose
/// maximum is attained twice witnesses the *first* index, and reversing the
/// operands moves the answer to the other one.
#[test]
fn the_selection_reference_is_falsifiable() {
    let tie = [Some(1i64), Some(9)];
    let rev = [Some(9i64), Some(1)];
    assert_eq!(reference(1, 2, 1, &tie, &rev), [(Some(10), 0)]);
    // A single un-tied maximum, at the far end, so a reference that always
    // answered zero would fail here.
    assert_eq!(
        reference(1, 2, 1, &tie, &[Some(0i64), Some(9)]),
        [(Some(18), 1)]
    );
    // The two degeneracies are answers, not exceptions.
    assert_eq!(reference(1, 0, 1, &[], &[]), [(None, 0)]);
    assert_eq!(reference(1, 2, 1, &[None, None], &tie), [(None, 0)]);
    // And a masked lane is absorbed rather than treated as a zero, which is
    // the difference a `0` fill would hide.
    assert_eq!(
        reference(1, 2, 1, &[None, Some(0)], &[Some(9i64), Some(1)]),
        [(Some(1), 1)]
    );
}
