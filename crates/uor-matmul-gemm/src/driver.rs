//! The dense driver (§5.5, S7--S13).

use uor_matmul_core::{
    AccOf, Accumulator, Alphabet, Backend, Bound, Element, EncodeFrom, EncodeMode, Traversal,
    Triple,
};

use crate::epilogue::Epilogue;
use crate::scratch::Scratch;

/// What a caller may choose. None of it changes the value computed.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct GemmOptions {
    /// Which order to walk the output in.
    pub traversal: Traversal,
    /// Which factorization of the identity to run.
    pub backend: Backend,
    /// How the single encode step presents the exact accumulator.
    pub encode: EncodeMode,
}

/// `C := epilogue(A * B, C)`.
///
/// Returns `()`. Nothing about the data, the size, or the host can make this
/// fail: the requested product exists, because a [`Triple`] exists (R14, C6).
///
/// # Totality
///
/// - Any `m`, `k`, `n`, including 0, 1, primes, and shapes far past any block
///   size. Zero padding is exact, so an unaligned shape takes this path and not
///   a different one (S8).
/// - Any strides, including negative and zero on the inputs. Transposition is a
///   stride (S7).
/// - Any magnitude, at any depth. The accumulator cannot overflow (§3.2).
/// - Any scratch amount, including none (S13, `CD-04`).
///
/// # Parametric in the algebra
///
/// The bound is `E: Element`, not `E: IntegerElement`, and the body names
/// exactly two operations: [`Element::mac`] and
/// [`uor_matmul_core::Accumulator::combine`]. Both are the element type's own,
/// so this one traversal computes a ring product over `Alphabet<i8, _>` and a
/// `(max, +)` product over `Alphabet<Trop<i8>, _>` with no branch, no second
/// driver, and no line that asks which algebra it is in. The census's two
/// products are one traversal at two instantiations, which is what R13 says
/// about every other axis of this library and is now true of this one too.
///
/// [`Trop`]: uor_matmul_core::Trop
///
/// # Examples
///
/// ```
/// # use uor_matmul_core::{as_alphabet_full, MatView, MatViewMut, Triple};
/// # use uor_matmul_gemm::{gemm, GemmOptions, Linear, Scratch};
/// let a = [1i8, 2, 3, 4];
/// let b = [5i8, 6, 7, 8];
/// let mut c = [0i32; 4];
///
/// let av = MatView::row_major(as_alphabet_full(&a), 2, 2).unwrap();
/// let bv = MatView::row_major(as_alphabet_full(&b), 2, 2).unwrap();
/// let cv = MatViewMut::row_major(&mut c, 2, 2).unwrap();
/// let mut t = Triple::new(av, bv, cv).unwrap();
///
/// gemm(&mut t, &Linear::OVERWRITE, GemmOptions::default(), &mut Scratch::none());
/// assert_eq!(c, [19, 22, 43, 50]);
/// ```
pub fn gemm<E, Bd, O, Ep>(
    triple: &mut Triple<'_, '_, '_, Alphabet<E, Bd>, O>,
    epilogue: &Ep,
    options: GemmOptions,
    scratch: &mut Scratch<'_, E, Bd>,
) where
    E: Element,
    Bd: Bound,
    O: Element + EncodeFrom<AccOf<E>>,
    Ep: Epilogue<E, O>,
{
    // The backend is not consulted here. Every backend computes the same
    // integer, so selecting one is a question about instructions and never
    // about which function is being computed (R13). The portable factorization
    // below is the reference every other one is validated against (R6).
    let _ = options.backend;

    let shape = triple.shape();
    if shape.m == 0 || shape.n == 0 {
        // Nothing to write. Not a special case: the loops below would do the
        // same thing, and saying so costs one comparison.
        return;
    }

    let panel = match options.traversal {
        // Streaming: one output element at a time, no packed panel at all.
        Traversal::OutputMajor => 0,
        // Blocked: as long a k-panel as the offer supports.
        //
        // `Tabulated` is the same length here. A dense operand is
        // `Identity`-coded, whose code space is the alphabet itself, so there is
        // no table to index --- which the type system says at
        // [`crate::tabulated::gemm_tabulated`]'s boundary rather than here. This
        // is not a fallback: both walk the same products into the same exact sum,
        // and `CD-13` asserts the bytes (R13, C5).
        Traversal::Blocked | Traversal::Tabulated => scratch.panel(shape.k),
    };

    let (a, b, c) = triple.parts();
    let reads_c = epilogue.reads_c();

    for i in 0..shape.m {
        for j in 0..shape.n {
            let mut acc = <AccOf<E> as Accumulator>::ZERO;

            if panel == 0 {
                for p in 0..shape.k {
                    E::mac(&mut acc, a.at(i, p).get(), b.at(p, j).get());
                }
            } else {
                // The same accumulation, in panels. `combine` is associative on
                // every value that can arise, so the panel length is invisible
                // in the result --- which is what `CD-01` and `CD-04` assert.
                let mut p = 0;
                while p < shape.k {
                    let end = shape.k.min(p + panel);
                    let mut part = <AccOf<E> as Accumulator>::ZERO;
                    for q in p..end {
                        E::mac(&mut part, a.at(i, q).get(), b.at(q, j).get());
                    }
                    acc = acc.combine(part);
                    p = end;
                }
            }

            // The single encode step, exactly once per output element. `prior`
            // is `None` when the epilogue does not read `C`, so an
            // uninitialised output buffer is admissible (`CS-04`).
            let prior = if reads_c { Some(*c.at(i, j)) } else { None };
            *c.at_mut(i, j) = epilogue.finish(acc, prior, options.encode);
        }
    }
}

#[cfg(test)]
// R7 governs the library, not its tests: these build operands on the heap so
// that awkward shapes can be generated. `CA-01` witnesses the library's own
// zero-allocation claim with a counting allocator instead.
#[allow(clippy::disallowed_types)]
mod tests {
    use super::*;
    use crate::epilogue::Linear;
    use crate::scratch::Scratch;
    use std::vec;
    use std::vec::Vec;
    use uor_matmul_core::{as_alphabet_full, MatView, MatViewMut, Strides};

    fn product(m: usize, k: usize, n: usize, traversal: Traversal, panel: usize) -> Vec<i32> {
        let a: Vec<i8> = (0..m * k).map(|i| ((i * 37) % 255) as i8).collect();
        let b: Vec<i8> = (0..k * n).map(|i| ((i * 53) % 255) as i8).collect();
        let mut c = vec![0i32; m * n];
        let mut buf = vec![Alphabet::ZERO; panel];

        let av = MatView::row_major(as_alphabet_full(&a), m, k).unwrap();
        let bv = MatView::row_major(as_alphabet_full(&b), k, n).unwrap();
        let cv = MatViewMut::row_major(&mut c, m, n).unwrap();
        let mut t = Triple::new(av, bv, cv).unwrap();
        gemm(
            &mut t,
            &Linear::OVERWRITE,
            GemmOptions {
                traversal,
                ..Default::default()
            },
            &mut Scratch::new(&mut buf),
        );
        c
    }

    /// CD-04, CD-05: every traversal and every scratch amount, including none,
    /// produce the same bytes. Only the traversal differs.
    #[test]
    fn traversal_and_scratch_are_invisible_cd_04() {
        // A prime depth, so no panel length divides it evenly.
        let reference = product(5, 97, 7, Traversal::OutputMajor, 0);
        for panel in [0usize, 1, 2, 3, 13, 96, 97, 98, 1000] {
            assert_eq!(
                product(5, 97, 7, Traversal::Blocked, panel),
                reference,
                "panel {panel}"
            );
        }
        assert_eq!(product(5, 97, 7, Traversal::OutputMajor, 1000), reference);
    }

    /// `CD-10`: `Scratch::none`, one element, `suggested_scratch - 1`,
    /// `suggested_scratch`, and ten times it all give the same bytes.
    ///
    /// Scratch is an offer. The library's whole response to the amount is a
    /// panel length, and a panel length is invisible in an exact sum.
    #[test]
    fn every_scratch_amount_gives_the_same_bytes_cd_10() {
        let (m, k, n) = (5usize, 97usize, 7usize);
        let suggested = crate::suggested_scratch(uor_matmul_core::Shape { m, k, n });
        let reference = product(m, k, n, Traversal::Blocked, 0);
        for amount in [0, 1, suggested.saturating_sub(1), suggested, suggested * 10] {
            assert_eq!(
                product(m, k, n, Traversal::Blocked, amount),
                reference,
                "scratch = {amount} (suggested = {suggested})"
            );
        }
        // And the query is a query: offering less is not an error.
        assert!(
            suggested > 0,
            "a non-degenerate shape suggests some scratch"
        );
    }

    /// CS-04: `beta = 0` overwrites `C` without reading it, so an output buffer
    /// holding a signalling pattern is admissible.
    #[test]
    fn beta_zero_overwrites_without_reading_cs_04() {
        let a = [1i8, 2, 3, 4];
        let b = [5i8, 6, 7, 8];
        let mut c = [i32::MIN; 4];
        let av = MatView::row_major(as_alphabet_full(&a), 2, 2).unwrap();
        let bv = MatView::row_major(as_alphabet_full(&b), 2, 2).unwrap();
        let cv = MatViewMut::row_major(&mut c, 2, 2).unwrap();
        let mut t = Triple::new(av, bv, cv).unwrap();
        assert!(!Epilogue::<i8, i32>::reads_c(&Linear::OVERWRITE));
        gemm(
            &mut t,
            &Linear::OVERWRITE,
            GemmOptions::default(),
            &mut Scratch::none(),
        );
        assert_eq!(c, [19, 22, 43, 50]);
    }

    /// CS-05: alpha and beta are applied exactly, in the accumulator's width,
    /// once per output element.
    #[test]
    fn alpha_and_beta_are_exact_cs_05() {
        let a = [1i8, 2, 3, 4];
        let b = [5i8, 6, 7, 8];
        let mut c = [100i32, 200, 300, 400];
        let av = MatView::row_major(as_alphabet_full(&a), 2, 2).unwrap();
        let bv = MatView::row_major(as_alphabet_full(&b), 2, 2).unwrap();
        let cv = MatViewMut::row_major(&mut c, 2, 2).unwrap();
        let mut t = Triple::new(av, bv, cv).unwrap();
        gemm(
            &mut t,
            &Linear { alpha: 3, beta: -2 },
            GemmOptions::default(),
            &mut Scratch::none(),
        );
        assert_eq!(c, [3 * 19 - 200, 3 * 22 - 400, 3 * 43 - 600, 3 * 50 - 800]);
    }

    /// CT-04: 0, 1, and prime dimensions take this path and not a different one.
    #[test]
    fn degenerate_and_prime_shapes_take_the_same_path_ct_04() {
        assert!(product(0, 5, 5, Traversal::Blocked, 4).is_empty());
        assert!(product(5, 0, 5, Traversal::Blocked, 4)
            .iter()
            .all(|&x| x == 0));
        assert_eq!(product(1, 1, 1, Traversal::Blocked, 4).len(), 1);
        assert_eq!(product(13, 17, 19, Traversal::Blocked, 5).len(), 13 * 19);
    }

    /// CS-02, CS-06: a transposed operand is a stride, and produces the
    /// transpose of the row-major product.
    #[test]
    fn transposed_operands_are_strides_cs_06() {
        let a = [1i8, 2, 3, 4, 5, 6]; // 2x3 row-major
        let b = [1i8, 2, 3, 4, 5, 6]; // 3x2 row-major
        let mut c = [0i32; 4];
        let av = MatView::row_major(as_alphabet_full(&a), 2, 3).unwrap();
        let bv = MatView::row_major(as_alphabet_full(&b), 3, 2).unwrap();
        let cv = MatViewMut::row_major(&mut c, 2, 2).unwrap();
        let mut t = Triple::new(av, bv, cv).unwrap();
        gemm(
            &mut t,
            &Linear::OVERWRITE,
            GemmOptions::default(),
            &mut Scratch::none(),
        );
        assert_eq!(c, [22, 28, 49, 64]);

        // The same B, read as its own transpose by swapping the strides.
        let mut d = [0i32; 9];
        let av = MatView::new(as_alphabet_full(&a), 3, 2, Strides { rs: 1, cs: 3 }).unwrap();
        let bv = MatView::row_major(as_alphabet_full(&b), 2, 3).unwrap();
        let cv = MatViewMut::row_major(&mut d, 3, 3).unwrap();
        let mut t = Triple::new(av, bv, cv).unwrap();
        gemm(
            &mut t,
            &Linear::OVERWRITE,
            GemmOptions::default(),
            &mut Scratch::none(),
        );
        // A^T (3x2) * B (2x3), computed by hand:
        //   [[1,4],[2,5],[3,6]] * [[1,2,3],[4,5,6]]
        assert_eq!(d, [17, 22, 27, 22, 29, 36, 27, 36, 45]);
    }

    /// `CD-29`: the *same* driver computes the `(max, +)` product.
    ///
    /// Not a second traversal and not a branch: the loops above name `E::mac`
    /// and `combine`, so instantiating `E` at `Trop<i8>` is the whole of the
    /// change. The census's two products are one traversal.
    #[test]
    fn the_same_driver_computes_the_tropical_product_cd_29() {
        use uor_matmul_core::{as_alphabet_tropical, Trop};
        let f = Trop::<i8>::finite;
        let g = Trop::<i32>::finite;
        // A = [[1,2],[3,4]], B = [[5,6],[7,8]] over (max, +):
        //   c00 = max(1+5, 2+7) = 9    c01 = max(1+6, 2+8) = 10
        //   c10 = max(3+5, 4+7) = 11   c11 = max(3+6, 4+8) = 12
        let (av, bv) = ([f(1), f(2), f(3), f(4)], [f(5), f(6), f(7), f(8)]);
        let mut c = [Trop::<i32>::NEG_INF; 4];
        let a = MatView::row_major(as_alphabet_tropical(&av), 2, 2).unwrap();
        let b = MatView::row_major(as_alphabet_tropical(&bv), 2, 2).unwrap();
        let cv = MatViewMut::row_major(&mut c, 2, 2).unwrap();
        let mut t = Triple::new(a, b, cv).unwrap();
        gemm(
            &mut t,
            &crate::epilogue::MaxPlus::OVERWRITE,
            GemmOptions::default(),
            &mut Scratch::none(),
        );
        assert_eq!(
            c,
            [g(9), g(10), g(11), g(12)],
            "the (max, +) product of the same two operands"
        );

        // The value half above is necessary and nowhere near sufficient. This
        // row's `refuted_by` names *a branch on the semiring inside the
        // traversal*, and four correct numbers cannot see one: a driver that
        // dispatched on which algebra it was in would compute exactly these
        // numbers by a second route and read as discharged. So the structural
        // half is read from the source, the way `CU-06` reads its claim from the
        // emitted instructions rather than inferring it.
        //
        // A planted `if <AccOf<E> as Accumulator>::BITS <= 66 { ...second
        // traversal... }` --- the ring accumulator at `i8` is 79 bits and the
        // tropical one 10, so the guard is exactly a semiring test --- passed
        // every gate in the workspace before this half existed.
        let source = include_str!("driver.rs");
        let start = source
            .find("pub fn gemm<E, Bd, O, Ep>")
            .expect("the driver is in this file");
        let body = &source[start
            ..start
                + source[start..]
                    .find("\n#[cfg(test)]")
                    .expect("the tests follow the driver")];
        for token in [
            // The width of an accumulator, which is what tells the two families
            // apart without naming either.
            "BITS",
            // Either family, named.
            "Trop",
            "IntegerElement",
            "FloatElement",
            // Any run-time question about which type this is.
            "TypeId",
            "type_name",
        ] {
            assert!(
                !body.contains(token),
                "`gemm` names `{token}`, so it can tell the two semirings apart --- and a \
                 traversal that can tell them apart is two traversals sharing a name"
            );
        }
        // And it does name the two operations it is written against, so the
        // absence above is parametricity rather than an empty function.
        assert!(
            body.contains("E::mac(&mut acc"),
            "the one arithmetic primitive"
        );
        assert!(body.contains(".combine(part)"), "and the one reduction");
    }

    /// `CS-11`: the tropical sibling of `CS-04`. `beta` at the semiring zero
    /// overwrites `C` without reading it, so a masked or garbage-filled output
    /// buffer is admissible.
    #[test]
    fn the_semiring_zero_beta_overwrites_without_reading_cs_11() {
        use crate::epilogue::MaxPlus;
        use uor_matmul_core::{as_alphabet_tropical, Trop};
        let f = Trop::<i8>::finite;
        let g = Trop::<i32>::finite;
        let (av, bv) = ([f(1), f(2), f(3), f(4)], [f(5), f(6), f(7), f(8)]);
        // A pattern no reduction can produce, so a read would show.
        let mut c = [g(i32::MAX); 4];
        let a = MatView::row_major(as_alphabet_tropical(&av), 2, 2).unwrap();
        let b = MatView::row_major(as_alphabet_tropical(&bv), 2, 2).unwrap();
        let cv = MatViewMut::row_major(&mut c, 2, 2).unwrap();
        let mut t = Triple::new(a, b, cv).unwrap();
        assert!(!Epilogue::<Trop<i8>, Trop<i32>>::reads_c(
            &MaxPlus::OVERWRITE
        ));
        assert!(Epilogue::<Trop<i8>, Trop<i32>>::reads_c(
            &MaxPlus::ACCUMULATE
        ));
        gemm(
            &mut t,
            &MaxPlus::OVERWRITE,
            GemmOptions::default(),
            &mut Scratch::none(),
        );
        assert_eq!(c, [g(9), g(10), g(11), g(12)]);
    }

    /// `CS-12`: the tropical epilogue's `alpha` and `beta` are applied exactly,
    /// in the accumulator's width, once per output element --- the sibling of
    /// `CS-05`, and the row that makes `MaxPlus`'s `⊗` arithmetic live.
    ///
    /// Both scalars had no gate at all until this one. Every call site in the
    /// workspace passed `MaxPlus::OVERWRITE`, whose `alpha` is the
    /// multiplicative identity and whose `beta` is the semiring zero, so
    /// `ShiftExact::shift_exact` was only ever evaluated at an identity and
    /// `AbsorbPrior::of_prior` was never evaluated at all. Two planted defects
    /// --- `⊗` discarding its scalar, and `-inf` in `C` absorbed to a finite
    /// zero --- both survived `cargo test --workspace`. This row is what those
    /// plants now fail.
    #[test]
    fn the_tropical_alpha_and_beta_are_exact_cs_12() {
        use crate::epilogue::MaxPlus;
        use uor_matmul_core::{as_alphabet_tropical, Trop};

        let f = Trop::<i8>::finite;
        let g = Trop::<i32>::finite;
        // A ⊗ B over (max, +): c00 = max(1+5, 2+7) = 9, c01 = 10, c10 = 11, c11 = 12.
        let (av, bv) = ([f(1), f(2), f(3), f(4)], [f(5), f(6), f(7), f(8)]);
        let product = [9i32, 10, 11, 12];

        let run = |ep: &MaxPlus, prior: [Trop<i32>; 4]| {
            let mut c = prior;
            let a = MatView::row_major(as_alphabet_tropical(&av), 2, 2).unwrap();
            let b = MatView::row_major(as_alphabet_tropical(&bv), 2, 2).unwrap();
            let cv = MatViewMut::row_major(&mut c, 2, 2).unwrap();
            let mut t = Triple::new(a, b, cv).unwrap();
            gemm(&mut t, ep, GemmOptions::default(), &mut Scratch::none());
            c
        };

        // `alpha` shifts the product, exactly. A scalar the epilogue discarded
        // would leave the product unmoved, which is the first plant.
        for alpha in [-40i64, -1, 0, 1, 37] {
            let ep = MaxPlus {
                alpha: Trop::finite(alpha),
                beta: Trop::NEG_INF,
            };
            let want: [Trop<i32>; 4] = core::array::from_fn(|i| g(product[i] + alpha as i32));
            assert_eq!(run(&ep, [Trop::NEG_INF; 4]), want, "alpha = {alpha}");
        }

        // `beta` shifts what `C` already held, and `⊕` takes the larger. Three
        // regimes, so neither side can be the answer by accident: `C` far below
        // the product, `C` far above it, and `C` straddling it.
        for (beta, prior, want) in [
            (0i64, -100i32, product),
            (0, 100, [100, 100, 100, 100]),
            (0, 10, [10, 10, 11, 12]),
            (5, 0, [9, 10, 11, 12]),
            (-100, 100, product),
        ] {
            let ep = MaxPlus {
                alpha: Trop::finite(0),
                beta: Trop::finite(beta),
            };
            let got = run(&ep, [g(prior); 4]);
            let want: [Trop<i32>; 4] = core::array::from_fn(|i| g(want[i]));
            assert_eq!(got, want, "beta = {beta}, C = {prior}");
        }

        // The case the second plant broke: `C` at the semiring zero, under a
        // *finite* `beta`. `-inf ⊗ b` is `-inf`, which contributes nothing, so
        // the answer is the product --- and if `of_prior` mapped `-inf` to a
        // finite zero instead, every cell would gain a floor of zero and the
        // two negative cells below would be wrong.
        let (nv, mv) = ([f(-5), f(-6), f(-7), f(-8)], [f(-5), f(-6), f(-7), f(-8)]);
        let negative = {
            let mut c = [g(0); 4];
            let a = MatView::row_major(as_alphabet_tropical(&nv), 2, 2).unwrap();
            let b = MatView::row_major(as_alphabet_tropical(&mv), 2, 2).unwrap();
            let cv = MatViewMut::row_major(&mut c, 2, 2).unwrap();
            let mut t = Triple::new(a, b, cv).unwrap();
            gemm(
                &mut t,
                &MaxPlus {
                    alpha: Trop::finite(0),
                    beta: Trop::NEG_INF,
                },
                GemmOptions::default(),
                &mut Scratch::none(),
            );
            c
        };
        assert!(
            negative.iter().all(|x| x.get().is_some_and(|v| v < 0)),
            "the fixture must produce negative cells, or the floor is invisible"
        );
        let mut c = [Trop::<i32>::NEG_INF; 4];
        let a = MatView::row_major(as_alphabet_tropical(&nv), 2, 2).unwrap();
        let b = MatView::row_major(as_alphabet_tropical(&mv), 2, 2).unwrap();
        let cv = MatViewMut::row_major(&mut c, 2, 2).unwrap();
        let mut t = Triple::new(a, b, cv).unwrap();
        gemm(
            &mut t,
            &MaxPlus::ACCUMULATE,
            GemmOptions::default(),
            &mut Scratch::none(),
        );
        assert_eq!(c, negative, "`-inf` in C is absorbed, not floored at zero");

        // `alpha` at the semiring zero absorbs the whole product, whatever the
        // operands were --- the other half of `⊗`'s absorbing law, which no
        // other row reaches.
        let absorbed = run(
            &MaxPlus {
                alpha: Trop::NEG_INF,
                beta: Trop::NEG_INF,
            },
            [g(0); 4],
        );
        assert_eq!(absorbed, [Trop::NEG_INF; 4]);
    }

    /// `CD-04` and `CD-10` at the tropical instance: traversal and scratch
    /// amount are invisible there for the same reason they are invisible in the
    /// ring --- `⊕` is associative, so the panel length cannot be seen.
    #[test]
    fn tropical_traversal_and_scratch_are_invisible_cd_04() {
        use crate::epilogue::MaxPlus;
        use uor_matmul_core::{as_alphabet_tropical, Trop};

        // A prime depth, so no panel length divides it evenly, and a mask in
        // the middle of every row so the semiring zero is walked too.
        const K: usize = 97;
        // Row 0 of A is masked *entirely*, so cell (0, j) is the semiring zero
        // for every j and A-6's mask reaches the answer rather than only the
        // operand. Scattered masks alone cannot do that: at K = 97 with a mask
        // every eleventh element, some term of every cell is always live, and
        // an assertion about masked output would have had nothing to assert.
        let a: Vec<Trop<i8>> = (0..5 * K)
            .map(|i| {
                if i < K || i % 11 == 0 {
                    Trop::NEG_INF
                } else {
                    Trop::finite(((i * 37) % 255) as i8)
                }
            })
            .collect();
        let b: Vec<Trop<i8>> = (0..K * 7)
            .map(|i| {
                if i % 13 == 0 {
                    Trop::NEG_INF
                } else {
                    Trop::finite(((i * 53) % 255) as i8)
                }
            })
            .collect();

        let run = |traversal: Traversal, panel: usize| {
            let mut c = vec![Trop::<i32>::NEG_INF; 5 * 7];
            let mut buf = vec![Alphabet::ZERO; panel];
            let av = MatView::row_major(as_alphabet_tropical(&a), 5, K).unwrap();
            let bv = MatView::row_major(as_alphabet_tropical(&b), K, 7).unwrap();
            let cv = MatViewMut::row_major(&mut c, 5, 7).unwrap();
            let mut t = Triple::new(av, bv, cv).unwrap();
            gemm(
                &mut t,
                &MaxPlus::OVERWRITE,
                GemmOptions {
                    traversal,
                    ..Default::default()
                },
                &mut Scratch::new(&mut buf),
            );
            c
        };

        let reference = run(Traversal::OutputMajor, 0);
        for panel in [0usize, 1, 2, 3, 13, 96, 97, 98, 1000] {
            assert_eq!(run(Traversal::Blocked, panel), reference, "panel {panel}");
        }
        assert_eq!(run(Traversal::OutputMajor, 1000), reference);
        // The masks reached the *answer*, not merely the operand: the whole of
        // row 0 is the semiring zero, and the rest is finite. Both halves are
        // asserted, because either alone is satisfied by a fixture that does
        // not exercise A-6 at all.
        assert!(
            reference[..7].iter().all(|x| !x.is_finite()),
            "a wholly masked row reduces to the semiring zero, not to a number"
        );
        assert!(
            reference[7..].iter().all(|x| x.is_finite()),
            "and every other cell has a live term, or the fixture says nothing"
        );
    }

    /// CT-01: a depth past every narrow-register threshold is exact, because
    /// there is no threshold on the answer.
    #[test]
    fn depth_past_every_threshold_is_exact_ct_01() {
        const K: usize = 150_000;
        let a = vec![i8::MIN; K];
        let b = vec![i8::MIN; K];
        let mut c = [0i64; 1];
        let av = MatView::row_major(as_alphabet_full(&a), 1, K).unwrap();
        let bv = MatView::row_major(as_alphabet_full(&b), K, 1).unwrap();
        let cv = MatViewMut::row_major(&mut c, 1, 1).unwrap();
        let mut t = Triple::new(av, bv, cv).unwrap();
        gemm(
            &mut t,
            &Linear::OVERWRITE,
            GemmOptions::default(),
            &mut Scratch::none(),
        );
        assert_eq!(c[0], (K as i64) * 128 * 128);
    }
}
