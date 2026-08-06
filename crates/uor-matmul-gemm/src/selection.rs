//! Selection: the product, and *which* term achieved it (D-6).
//!
//! A `(max, +)` product answers a question a ring product cannot: not only what
//! the reduction is worth, but which of its terms it came from. That witness is
//! the census's `argmax`, and it needs a tie-break, because several terms may
//! achieve the maximum.
//!
//! The tie-break is **the smallest index**. That is not chosen here --- it is
//! reproduced from the authority (`AUTH-TF1-D6`), which declares it and proves
//! it total. What is constructed here is the realization, and `CD-24` and
//! `CD-25` are what validate it.
//!
//! # Two mechanisms, one answer
//!
//! [`Witness::Lexicographic`] carries `(value, index)` through the reduction and
//! reduces with an order that is total on the pair. Schedule independence is
//! then a property of the *order* and not of the loop: any partition, any
//! completion count, any order of combination gives the same witness, because a
//! total order's maximum does not depend on how the set was enumerated (U-ii).
//!
//! [`Witness::ComparePass`] carries the value only, and takes a second,
//! compare-only pass over the reduction for the least index achieving it.
//!
//! Neither is a fallback and neither is a fast path. They are two factorizations
//! of one answer, and `CD-25` asserts the bytes at every shape, every degeneracy
//! and every offer including none (R13, C5).
//!
//! # Nothing here allocates
//!
//! Both mechanisms run at `Scratch::none()`, output element at a time, in
//! registers. The witness is written into the caller's own view (R7, `CA-01`).

use uor_matmul_core::{
    AccOf, Accumulator, Alphabet, Bound, Element, EncodeFrom, MatViewMut, NotAProduct, Shape,
    Triple, TropicalValue,
};

use crate::driver::GemmOptions;
use crate::epilogue::Epilogue;

/// Which mechanism produces the witness.
///
/// Both produce identical bytes (`CD-25`). The choice is about how many passes
/// the reduction takes, never about the answer.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[non_exhaustive]
pub enum Witness {
    /// One pass, carrying `(value, index)` under an order that is total on the
    /// pair. The default, because schedule independence is then a property of
    /// the order rather than of the traversal.
    #[default]
    Lexicographic,
    /// Two passes: the value, then a compare-only scan for the least index
    /// achieving it.
    ComparePass,
}

/// The conformant triple, with a witness matrix beside the product.
///
/// The third fallible constructor in the library, and it fails with exactly the
/// two conditions [`Triple::new`] fails with and no others: a shape that is not
/// the product's, and strides that map two coordinates onto one cell. Both are
/// decided here, before any arithmetic is named, which is what `CT-05` requires
/// of every reportable condition in the library (R14, C6).
#[derive(Debug)]
pub struct SelectedTriple<'a, 'b, 'c, 'w, E, O> {
    triple: Triple<'a, 'b, 'c, E, O>,
    witness: MatViewMut<'w, usize>,
}

impl<'a, 'b, 'c, 'w, E, O> SelectedTriple<'a, 'b, 'c, 'w, E, O> {
    /// Validate the product and its witness together.
    ///
    /// The witness is `m x n`, exactly as `C` is. Nothing about the *values* in
    /// any of the four buffers can reach this decision.
    pub fn new(
        triple: Triple<'a, 'b, 'c, E, O>,
        witness: MatViewMut<'w, usize>,
    ) -> Result<Self, NotAProduct> {
        let shape = triple.shape();
        if witness.rows() != shape.m || witness.cols() != shape.n {
            return Err(NotAProduct::Nonconformant {
                a: (shape.m, shape.k),
                b: (witness.rows(), witness.cols()),
            });
        }
        let s = witness.strides();
        if s.self_aliases(witness.rows(), witness.cols()) {
            return Err(NotAProduct::OutputAliasesItself {
                m: witness.rows(),
                n: witness.cols(),
                rs: s.rs,
                cs: s.cs,
            });
        }
        Ok(Self { triple, witness })
    }

    /// The shape of the product, which exists because this value does.
    pub fn shape(&self) -> Shape {
        self.triple.shape()
    }

    /// The product, mutably.
    pub fn triple(&mut self) -> &mut Triple<'a, 'b, 'c, E, O> {
        &mut self.triple
    }

    /// The witness, mutably.
    pub fn witness_mut(&mut self) -> &mut MatViewMut<'w, usize> {
        &mut self.witness
    }
}

/// `(value, index)`, ordered lexicographically with the index *descending*.
///
/// The order is total, which is the whole content of `CD-24`: a maximum under a
/// total order does not depend on how the set was enumerated, so the witness is
/// invariant under partition, count and order **by construction** rather than by
/// a loop that happens to run forwards.
///
/// `at` is `usize::MAX` in the identity, so the identity loses to every real
/// offer and an empty reduction is distinguishable from a masked one.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Lex {
    /// `i128::MIN` names the semiring zero, exactly as `TropAcc` does.
    v: i128,
    /// The least index known to achieve `v`.
    at: usize,
}

impl Lex {
    const EMPTY: Self = Self {
        v: i128::MIN,
        at: usize::MAX,
    };

    /// `⊕` on the pair: greater value wins; on a tie, the smaller index does.
    ///
    /// Associative and commutative --- which is why it may be applied in any
    /// order --- and total, which is why the answer is the same one every time.
    fn combine(self, other: Self) -> Self {
        if other.v > self.v {
            other
        } else if other.v < self.v {
            self
        } else if other.at < self.at {
            other
        } else {
            self
        }
    }
}

/// The exact `⊗` of two elements, read out of the element type's own primitive.
///
/// [`Element::mac`] is *the one arithmetic primitive*, so the product is taken
/// from it rather than restated here. There is no second definition of `⊗` in
/// the library and this module does not become one (R13).
fn product<E>(a: E, w: E) -> Option<i128>
where
    E: Element,
    AccOf<E>: TropicalValue,
{
    let mut acc = <AccOf<E> as Accumulator>::ZERO;
    E::mac(&mut acc, a, w);
    acc.value()
}

/// `C := epilogue(A ⊗ B, C)`, and `W := argmax`.
///
/// Returns `()`, for the same reason [`crate::gemm`] does: the requested
/// product exists, because a [`SelectedTriple`] exists (R14, C6).
///
/// The witness of cell `(i, j)` is the **smallest** `p` at which `a[i,p] ⊗
/// b[p,j]` attains the reduction's maximum (`AUTH-TF1-D6`). Two degeneracies
/// are worth stating because they are answers and not exceptions:
///
/// - Every term at the semiring zero: they all attain the maximum, so the
///   witness is `0`.
/// - A reduction of depth zero: there is no term, and the witness is `0`, which
///   is `k` --- the past-the-end index --- at that depth.
///
/// # Examples
///
/// ```
/// # use uor_matmul_core::{as_alphabet_tropical, MatView, MatViewMut, Trop, Triple};
/// # use uor_matmul_gemm::{gemm_selected, GemmOptions, MaxPlus, Scratch, SelectedTriple, Witness};
/// let f = Trop::<i8>::finite;
/// let (av, bv) = ([f(1), f(9)], [f(9), f(1)]);
/// let mut c = [Trop::<i32>::NEG_INF; 1];
/// let mut w = [0usize; 1];
///
/// let a = MatView::row_major(as_alphabet_tropical(&av), 1, 2).unwrap();
/// let b = MatView::row_major(as_alphabet_tropical(&bv), 2, 1).unwrap();
/// let cv = MatViewMut::row_major(&mut c, 1, 1).unwrap();
/// let wv = MatViewMut::row_major(&mut w, 1, 1).unwrap();
/// let t = Triple::new(a, b, cv).unwrap();
/// let mut s = SelectedTriple::new(t, wv).unwrap();
///
/// gemm_selected(
///     &mut s,
///     &MaxPlus::OVERWRITE,
///     GemmOptions::default(),
///     Witness::Lexicographic,
///     &mut Scratch::none(),
/// );
/// // 1+9 and 9+1 both give 10; D-6 takes the smaller index.
/// assert_eq!(c[0], Trop::finite(10));
/// assert_eq!(w[0], 0);
/// ```
pub fn gemm_selected<E, Bd, O, Ep>(
    selected: &mut SelectedTriple<'_, '_, '_, '_, Alphabet<E, Bd>, O>,
    epilogue: &Ep,
    options: GemmOptions,
    mechanism: Witness,
    scratch: &mut crate::scratch::Scratch<'_, E, Bd>,
) where
    E: Element,
    Bd: Bound,
    AccOf<E>: TropicalValue,
    O: Element + EncodeFrom<AccOf<E>>,
    Ep: Epilogue<E, O>,
{
    // The offer buys a panel length here exactly as it does in the dense
    // driver, and a panel length is invisible in the answer for the same reason
    // --- `combine` is associative on both the value and the pair (`CD-24`).
    let shape = selected.shape();
    let panel = match options.traversal {
        uor_matmul_core::Traversal::OutputMajor => 0,
        _ => scratch.panel(shape.k),
    };
    if shape.m == 0 || shape.n == 0 {
        return;
    }

    let reads_c = epilogue.reads_c();
    let SelectedTriple { triple, witness } = selected;
    let (a, b, c) = triple.parts();

    for i in 0..shape.m {
        for j in 0..shape.n {
            let (acc, at) = match mechanism {
                Witness::Lexicographic => {
                    let mut lex = Lex::EMPTY;
                    if panel == 0 {
                        for p in 0..shape.k {
                            lex = lex.combine(Lex {
                                v: product::<E>(a.at(i, p).get(), b.at(p, j).get())
                                    .unwrap_or(i128::MIN),
                                at: p,
                            });
                        }
                    } else {
                        // The same reduction, in panels, each reduced on its own
                        // and combined. The order is total, so this is the same
                        // pair whatever the panel length is.
                        let mut p = 0;
                        while p < shape.k {
                            let end = shape.k.min(p + panel);
                            let mut part = Lex::EMPTY;
                            for q in p..end {
                                part = part.combine(Lex {
                                    v: product::<E>(a.at(i, q).get(), b.at(q, j).get())
                                        .unwrap_or(i128::MIN),
                                    at: q,
                                });
                            }
                            lex = lex.combine(part);
                            p = end;
                        }
                    }
                    let value = if lex.v == i128::MIN {
                        None
                    } else {
                        Some(lex.v)
                    };
                    // `usize::MAX` is the identity's index and means the
                    // reduction was empty; at depth zero the past-the-end index
                    // *is* zero, which is what `min` says here.
                    (AccOf::<E>::of_value(value), lex.at.min(shape.k))
                }
                Witness::ComparePass => {
                    // Pass one: the value, by the ordinary accumulation. This is
                    // the *same* reduction the dense driver runs, in panels or
                    // streaming exactly as it does.
                    let mut acc = <AccOf<E> as Accumulator>::ZERO;
                    let mut p = 0;
                    while p < shape.k {
                        let end = if panel == 0 {
                            shape.k
                        } else {
                            shape.k.min(p + panel)
                        };
                        let mut part = <AccOf<E> as Accumulator>::ZERO;
                        for q in p..end {
                            E::mac(&mut part, a.at(i, q).get(), b.at(q, j).get());
                        }
                        acc = acc.combine(part);
                        p = end;
                    }
                    // Pass two: compare only. The first index that attains the
                    // maximum is the least one that does, so the ascending scan
                    // *is* D-6's tie-break and needs no second comparison.
                    let target = acc.value();
                    let mut at = shape.k;
                    for q in 0..shape.k {
                        if product::<E>(a.at(i, q).get(), b.at(q, j).get()) == target {
                            at = q;
                            break;
                        }
                    }
                    // A reduction whose every term is the semiring zero has a
                    // maximum of `None`, which the scan above matches at `q =
                    // 0`; a reduction of depth zero leaves `at` at `k`, which is
                    // zero there.
                    (acc, at)
                }
            };

            let prior = if reads_c { Some(*c.at(i, j)) } else { None };
            *c.at_mut(i, j) = epilogue.finish(acc, prior, options.encode);
            *witness.at_mut(i, j) = at;
        }
    }
}

#[cfg(test)]
// R7 governs the library, not its tests: these build operands on the heap so
// that awkward shapes can be generated.
#[allow(clippy::disallowed_types, clippy::too_many_arguments)]
mod tests {
    use super::*;
    use crate::epilogue::MaxPlus;
    use crate::scratch::Scratch;
    use std::vec;
    use std::vec::Vec;
    use uor_matmul_core::{as_alphabet_tropical, MatView, Traversal, Trop};

    /// A fixture whose value range is deliberately tiny, so that ties are
    /// *forced*. A randomized sweep over a wide range produces none, and a
    /// tie-break gate over such a sweep would read as evidence while testing
    /// nothing (D-6's own check does the same).
    fn tied_operands(m: usize, k: usize, n: usize) -> (Vec<Trop<i8>>, Vec<Trop<i8>>) {
        let a = (0..m * k)
            .map(|i| {
                if i % 7 == 3 {
                    Trop::NEG_INF
                } else {
                    Trop::finite((i % 3) as i8)
                }
            })
            .collect();
        let b = (0..k * n)
            .map(|i| {
                if i % 11 == 5 {
                    Trop::NEG_INF
                } else {
                    Trop::finite((i % 3) as i8)
                }
            })
            .collect();
        (a, b)
    }

    fn run(
        a: &[Trop<i8>],
        b: &[Trop<i8>],
        m: usize,
        k: usize,
        n: usize,
        mechanism: Witness,
        traversal: Traversal,
        panel: usize,
    ) -> (Vec<Trop<i32>>, Vec<usize>) {
        let mut c = vec![Trop::<i32>::NEG_INF; m * n];
        let mut w = vec![0usize; m * n];
        let mut buf = vec![Alphabet::ZERO; panel];
        {
            let av = MatView::row_major(as_alphabet_tropical(a), m, k).unwrap();
            let bv = MatView::row_major(as_alphabet_tropical(b), k, n).unwrap();
            let cv = MatViewMut::row_major(&mut c, m, n).unwrap();
            let wv = MatViewMut::row_major(&mut w, m, n).unwrap();
            let t = Triple::new(av, bv, cv).unwrap();
            let mut s = SelectedTriple::new(t, wv).unwrap();
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

    /// `CD-25`: the two mechanisms give identical witness bytes at every shape,
    /// degeneracy and offer including none.
    #[test]
    fn both_mechanisms_give_the_same_witness_cd_25() {
        for (m, k, n) in [
            (1, 1, 1),
            (1, 0, 1),
            (3, 1, 2),
            (2, 5, 3),
            (5, 97, 7),
            (4, 13, 1),
            (0, 5, 5),
            (5, 5, 0),
        ] {
            let (a, b) = tied_operands(m, k, n);
            for panel in [0usize, 1, 3, 97, 1000] {
                let lex = run(
                    &a,
                    &b,
                    m,
                    k,
                    n,
                    Witness::Lexicographic,
                    Traversal::Blocked,
                    panel,
                );
                let cmp = run(
                    &a,
                    &b,
                    m,
                    k,
                    n,
                    Witness::ComparePass,
                    Traversal::Blocked,
                    panel,
                );
                assert_eq!(lex, cmp, "shape {m}x{k}x{n}, panel {panel}");
                // And against the streaming traversal at no offer.
                let stream = run(
                    &a,
                    &b,
                    m,
                    k,
                    n,
                    Witness::Lexicographic,
                    Traversal::OutputMajor,
                    0,
                );
                assert_eq!(lex, stream, "shape {m}x{k}x{n}: streaming");
            }
        }
    }

    /// `CD-24`: the witness is invariant under partition, count and order
    /// **precisely because** the order on `(value, index)` is total --- and a
    /// non-total order is exhibited under which it is not.
    #[test]
    fn the_witness_is_invariant_because_the_order_is_total_cd_24() {
        let (m, k, n) = (4usize, 40usize, 3usize);
        let (a, b) = tied_operands(m, k, n);
        let reference = run(
            &a,
            &b,
            m,
            k,
            n,
            Witness::Lexicographic,
            Traversal::OutputMajor,
            0,
        );
        // Every partition of the reduction the offer can buy.
        for panel in [1usize, 2, 3, 7, 39, 40, 41, 4096] {
            assert_eq!(
                run(
                    &a,
                    &b,
                    m,
                    k,
                    n,
                    Witness::Lexicographic,
                    Traversal::Blocked,
                    panel
                ),
                reference,
                "panel {panel}"
            );
        }
        // The order has two halves and this row pins both. The *totality* is
        // what the partition sweep above exercises --- and it would still hold
        // if the index ran the other way, because "largest index" is a total
        // order too. So the *convention* is pinned separately, on a forced tie:
        // `AUTH-TF1-D6` declares the smallest index, and nothing else in this
        // module states which direction that is.
        let tie = [Trop::<i8>::finite(1), Trop::finite(9)];
        let rev = [Trop::<i8>::finite(9), Trop::finite(1)];
        for mechanism in [Witness::Lexicographic, Witness::ComparePass] {
            let (c, w) = run(&tie, &rev, 1, 2, 1, mechanism, Traversal::Blocked, 0);
            assert_eq!(c[0], Trop::finite(10), "{mechanism:?}: both terms give 10");
            assert_eq!(
                w[0], 0,
                "{mechanism:?}: D-6 declares the smallest index, not the largest"
            );
        }

        // The fixture must actually produce ties, or the row is vacuous.
        let ties = (0..k)
            .filter(|&p| product::<Trop<i8>>(a[p], b[p * n]) == product::<Trop<i8>>(a[0], b[0]))
            .count();
        assert!(ties > 1, "the fixture must force ties, found {ties}");

        // The other side. Combining under an order that is *not* total on the
        // pair --- value only, keeping the later offer on a tie --- makes the
        // answer depend on the enumeration. Same terms, two orders, two
        // witnesses.
        let terms: Vec<(i128, usize)> = (0..k)
            .map(|p| (product::<Trop<i8>>(a[p], b[p * n]).unwrap_or(i128::MIN), p))
            .collect();
        let value_only = |xs: &[(i128, usize)]| {
            let mut best = (i128::MIN, usize::MAX);
            for &(v, at) in xs {
                // `>=` rather than `>`: a tie takes the later offer, which is
                // exactly the missing totality.
                if v >= best.0 {
                    best = (v, at);
                }
            }
            best.1
        };
        let mut reversed = terms.clone();
        reversed.reverse();
        assert_ne!(
            value_only(&terms),
            value_only(&reversed),
            "a non-total order must be order-dependent, or the contrast is vacuous"
        );
        // While the total one is not.
        let total = |xs: &[(i128, usize)]| {
            xs.iter()
                .fold(Lex::EMPTY, |acc, &(v, at)| acc.combine(Lex { v, at }))
                .at
        };
        assert_eq!(total(&terms), total(&reversed));
    }

    /// The two degeneracies are answers, not exceptions: an all-masked
    /// reduction witnesses index zero, and a depth-zero one witnesses `k`.
    #[test]
    fn the_degenerate_selections_are_answers_ct_04() {
        for mechanism in [Witness::Lexicographic, Witness::ComparePass] {
            // Every term at the semiring zero.
            let masked = [Trop::<i8>::NEG_INF; 4];
            let (c, w) = run(&masked, &masked, 1, 4, 1, mechanism, Traversal::Blocked, 0);
            assert_eq!(c[0], Trop::NEG_INF);
            assert_eq!(w[0], 0, "{mechanism:?}: every term attains the maximum");

            // Depth zero: no term at all.
            let (c, w) = run(&[], &[], 1, 0, 1, mechanism, Traversal::Blocked, 0);
            assert_eq!(c[0], Trop::NEG_INF);
            assert_eq!(w[0], 0, "{mechanism:?}: the past-the-end index at k = 0");
        }
    }

    /// The witness view is validated where every other non-existence is:
    /// once, at construction, before any arithmetic is named (`CT-05`).
    #[test]
    fn a_nonconformant_witness_is_reported_at_construction_ct_05() {
        let f = Trop::<i8>::finite;
        let (av, bv) = ([f(1), f(2)], [f(3), f(4)]);
        let mut c = [Trop::<i32>::NEG_INF; 1];
        let mut w = [0usize; 4];
        let a = MatView::row_major(as_alphabet_tropical(&av), 1, 2).unwrap();
        let b = MatView::row_major(as_alphabet_tropical(&bv), 2, 1).unwrap();
        let cv = MatViewMut::row_major(&mut c, 1, 1).unwrap();
        // A 2x2 witness for a 1x1 product.
        let wv = MatViewMut::row_major(&mut w, 2, 2).unwrap();
        let t = Triple::new(a, b, cv).unwrap();
        assert!(matches!(
            SelectedTriple::new(t, wv),
            Err(NotAProduct::Nonconformant { .. })
        ));
    }
}
