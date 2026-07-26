//! GEMM against a coded operand, streaming (§6.3).
//!
//! Not a second driver. The weights arrive as codes instead of as alphabet
//! elements, they are decoded, and from there this is the same accumulation the
//! dense driver runs --- which is the whole content of `CL-MM01` and what
//! `CK-05` measures.
//!
//! # Decoding is not the only thing a code is good for
//!
//! Decode-then-multiply issues the same `m * k * n` products the dense driver
//! does and adds a decode on top, so the codec buys residency and pays for it in
//! throughput. [`crate::tabulated`] is the other direction: it indexes a table
//! *by* the code, so the column loop is one read and one add per code covering
//! `MAX_BLOCK` weights and contains no multiply at all.
//!
//! Which one applies is a question about the operand's layout, not about quality.
//! Here `B` is `k x n` and a code names `MAX_BLOCK` consecutive elements of one
//! *row* --- `MAX_BLOCK` different output columns --- so there is nothing for a
//! partial sum to be a partial sum of. Tabulation needs the code block to run
//! along the reduction, which is the `n x k` orientation, and that is the shape
//! [`crate::TabulatedTriple`] takes.
//!
//! This traversal keeps the property that one has to give up for a table: it
//! needs no scratch at all, not even one decoded row, so it runs on a target
//! whose RAM cannot hold either (S13).

use uor_matmul_codec::{Codec, CodedMatrix};
use uor_matmul_core::{
    AccOf, Accumulator, Alphabet, Bound, Element, EncodeFrom, IntegerElement, MatView, MatViewMut,
    NotAProduct, Shape,
};

use crate::driver::GemmOptions;
use crate::epilogue::Epilogue;

/// The conformant triple with a coded right operand.
///
/// The library's second and last fallible constructor, with the same signature
/// and the same two failures as [`uor_matmul_core::Triple::new`] (§5.5).
pub struct CodedTriple<'a, 'b, 'c, E: IntegerElement, Bd: Bound, C: Codec<E, Bd>, O> {
    a: MatView<'a, Alphabet<E, Bd>>,
    b: CodedMatrix<'b, E, Bd, C>,
    c: MatViewMut<'c, O>,
}

impl<'a, 'b, 'c, E: IntegerElement, Bd: Bound, C: Codec<E, Bd>, O>
    CodedTriple<'a, 'b, 'c, E, Bd, C, O>
{
    /// Report non-existence once, before any arithmetic is named.
    pub fn new(
        a: MatView<'a, Alphabet<E, Bd>>,
        b: CodedMatrix<'b, E, Bd, C>,
        c: MatViewMut<'c, O>,
    ) -> Result<Self, NotAProduct> {
        if a.cols() != b.rows() || c.rows() != a.rows() || c.cols() != b.cols() {
            return Err(NotAProduct::Nonconformant {
                a: (a.rows(), a.cols()),
                b: (b.rows(), b.cols()),
            });
        }
        let s = c.strides();
        if s.self_aliases(c.rows(), c.cols()) {
            return Err(NotAProduct::OutputAliasesItself {
                m: c.rows(),
                n: c.cols(),
                rs: s.rs,
                cs: s.cs,
            });
        }
        Ok(Self { a, b, c })
    }

    /// The shape of the product, which exists because this value does.
    pub fn shape(&self) -> Shape {
        Shape {
            m: self.a.rows(),
            k: self.a.cols(),
            n: self.b.cols(),
        }
    }
}

/// `C := epilogue(A * decode(B), C)`.
///
/// Returns `()`, for the same reason [`crate::gemm`] does.
///
/// The decode is streamed one element at a time, so this needs no scratch at
/// all --- not even one decoded row, which is what lets it run where neither
/// fits (S13).
///
/// A caller with memory to spare should take the other path: decode once through
/// [`CodedMatrix::decode_row_into`] and run [`crate::gemm_packed`], which reaches
/// the packed kernels, where this traversal decodes every weight once per row of
/// `A`. That composition gives the same bytes, and `CD-04` asserts it here --- on
/// a fixed-width tier and on a run codec, at no offer and at the suggested one.
/// It is the only fast path a run codec has, since a variable-length code has no
/// `p`-th block for a table to index.
pub fn coded_gemm<E, Bd, C, O, Ep>(
    triple: &mut CodedTriple<'_, '_, '_, E, Bd, C, O>,
    epilogue: &Ep,
    options: GemmOptions,
) where
    E: IntegerElement,
    Bd: Bound,
    C: Codec<E, Bd>,
    O: Element + EncodeFrom<AccOf<E>>,
    Ep: Epilogue<E, O>,
{
    let shape = triple.shape();
    if shape.m == 0 || shape.n == 0 {
        return;
    }
    let reads_c = epilogue.reads_c();

    for i in 0..shape.m {
        for j in 0..shape.n {
            let mut acc = <AccOf<E> as Accumulator>::ZERO;
            // Walked, not indexed. `at(p, j)` is O(1) only for a fixed-width
            // tier; for a variable-length one it walks row `p` *and*
            // `row_code_range` walks rows `0..p`, so this loop cost O(k^2) per
            // output element and the whole traversal O(m n k^2) --- which is the
            // hazard `row_code_range`'s own note names, on the one driver that
            // reads a coded operand an element at a time. The walk carries its
            // cursor, so a column is one pass over the codes whatever the tier,
            // and the identity is O(m n k) again.
            for (p, w) in triple.b.column_walk(j).enumerate() {
                if p >= shape.k {
                    break;
                }
                // Decode, then accumulate exactly. The codec is not an argument
                // of the arithmetic below it.
                E::mac(&mut acc, triple.a.at(i, p).get(), w.get());
            }
            let prior = if reads_c {
                Some(*triple.c.at(i, j))
            } else {
                None
            };
            *triple.c.at_mut(i, j) = epilogue.finish(acc, prior, options.encode);
        }
    }
}

#[cfg(test)]
extern crate std;

#[cfg(test)]
// R7 governs the library, not its tests: these build operands on the heap so
// that a decoded matrix can be held whole and compared against the streamed one.
#[allow(clippy::disallowed_types)]
mod tests {
    use super::*;
    use crate::epilogue::Linear;
    use crate::{gemm, gemm_packed, Scratch};
    use std::vec;
    use std::vec::Vec;
    use uor_matmul_codec::{Grid, Runs};
    use uor_matmul_core::{as_alphabet_full, Full, Triple};

    type A8 = Alphabet<i8, Full<i8>>;

    /// `CD-04`: decoding first and running the dense driver gives the bytes the
    /// streaming coded traversal gives.
    ///
    /// [`coded_gemm`] needs no scratch, not even one decoded row, which is what
    /// lets it run where neither fits (S13). The claim beside it --- that a caller
    /// with memory to spare gets the same bytes by decoding through
    /// `CodedMatrix::decode_row_into` --- is the *other* path, and it is the one a
    /// caller with a cache should take: it reaches the packed kernels, where the
    /// streaming traversal decodes every weight once per row of `A`.
    ///
    /// That claim cited `CD-04`, and `CD-04` asserted the streaming decode equals
    /// the row decode --- true, and not this. The composition is asserted here, on
    /// a fixed-width tier and on a variable-length one, so the sentence and the
    /// suite say the same thing.
    #[test]
    fn decoding_first_gives_the_streamed_bytes_cd_04() {
        let vals: [A8; 16] = core::array::from_fn(|i| Alphabet::of((i as i8) - 8));
        let grid = Grid::<i8, Full<i8>, 16>::new(&vals);

        for &(m, k, n) in &[(1usize, 4usize, 1usize), (3, 8, 5), (8, 12, 16), (5, 24, 7)] {
            let a: Vec<i8> = (0..m * k)
                .map(|i| ((i * 13 % 255) as i64 - 127) as i8)
                .collect();
            let codes: Vec<u16> = (0..k * n).map(|i| (i % 16) as u16).collect();
            let w = CodedMatrix::new(grid, k, n, &codes).expect("k x n");

            // The streaming traversal, with nothing offered.
            let mut streamed = vec![0i32; m * n];
            {
                let av = MatView::row_major(as_alphabet_full(&a), m, k).unwrap();
                let cv = MatViewMut::row_major(&mut streamed, m, n).unwrap();
                let mut t = CodedTriple::new(av, w, cv).expect("a product");
                coded_gemm(&mut t, &Linear::OVERWRITE, GemmOptions::default());
            }

            // Decode the whole operand once, then the dense driver --- both with
            // no offer and with one, since an offer must not change a byte.
            let mut decoded = vec![A8::ZERO; k * n];
            for r in 0..k {
                let wrote = w.decode_row_into(r, &mut decoded[r * n..(r + 1) * n]);
                assert_eq!(wrote, n, "a row decodes to exactly `cols`");
            }
            for offer in [0usize, crate::suggested_scratch(Shape { m, k, n })] {
                let mut dense = vec![0i32; m * n];
                {
                    let av = MatView::row_major(as_alphabet_full(&a), m, k).unwrap();
                    let bv = MatView::row_major(&decoded, k, n).unwrap();
                    let cv = MatViewMut::row_major(&mut dense, m, n).unwrap();
                    let mut t = Triple::new(av, bv, cv).expect("a product");
                    let mut buf = vec![A8::ZERO; offer];
                    if offer == 0 {
                        gemm(
                            &mut t,
                            &Linear::OVERWRITE,
                            GemmOptions::default(),
                            &mut Scratch::none(),
                        );
                    } else {
                        gemm_packed(
                            &mut t,
                            &Linear::OVERWRITE,
                            GemmOptions::default(),
                            &mut Scratch::new(&mut buf),
                        );
                    }
                }
                assert_eq!(dense, streamed, "{m}x{k}x{n} at an offer of {offer}");
            }
        }
    }

    /// The same, on the one tier whose codes are not a fixed width.
    ///
    /// A run codec has no `p`-th block to index, so it reaches neither the table
    /// nor a fixed-width shortcut --- which makes decode-then-dense the only
    /// fast path it has, and therefore the one worth asserting.
    #[test]
    fn a_run_codec_decodes_to_the_same_bytes_cd_04() {
        let vals: [A8; 16] = core::array::from_fn(|i| Alphabet::of((i as i8) - 8));
        let grid = Grid::<i8, Full<i8>, 16>::new(&vals);
        // Four runs of three, so each row is twelve elements.
        let runs: Vec<(u32, u16)> = (0..4 * 8).map(|i| (3u32, (i % 16) as u16)).collect();
        let sparse: Runs<'_, i8, Full<i8>, _, 8> = Runs::new(grid, &runs).expect("runs");
        let (m, k, n) = (4usize, 8usize, 12usize);
        let indices: Vec<u32> = (0..(4 * k) as u32).collect();
        let w = CodedMatrix::new(sparse, k, n, &indices).expect("4 runs of 3 per row");
        let a: Vec<i8> = (0..m * k)
            .map(|i| ((i * 13 % 255) as i64 - 127) as i8)
            .collect();

        let mut streamed = vec![0i32; m * n];
        {
            let av = MatView::row_major(as_alphabet_full(&a), m, k).unwrap();
            let cv = MatViewMut::row_major(&mut streamed, m, n).unwrap();
            let mut t = CodedTriple::new(av, w, cv).expect("a product");
            coded_gemm(&mut t, &Linear::OVERWRITE, GemmOptions::default());
        }

        let mut decoded = vec![A8::ZERO; k * n];
        for r in 0..k {
            assert_eq!(w.decode_row_into(r, &mut decoded[r * n..(r + 1) * n]), n);
        }
        let mut dense = vec![0i32; m * n];
        {
            let av = MatView::row_major(as_alphabet_full(&a), m, k).unwrap();
            let bv = MatView::row_major(&decoded, k, n).unwrap();
            let cv = MatViewMut::row_major(&mut dense, m, n).unwrap();
            let mut t = Triple::new(av, bv, cv).expect("a product");
            let mut buf = vec![A8::ZERO; crate::suggested_scratch(Shape { m, k, n })];
            gemm_packed(
                &mut t,
                &Linear::OVERWRITE,
                GemmOptions::default(),
                &mut Scratch::new(&mut buf),
            );
        }
        assert_eq!(dense, streamed, "a run codec decodes to the streamed bytes");
    }
}
