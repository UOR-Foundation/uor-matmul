//! Experiment scaffolding: a block codec over `f32` at a tiny code space.
//!
//! This is a measurement instrument, not a tier. The question it exists to
//! answer is whether the tabulated traversal pays for *floats* once the codec
//! has a block wider than one: the only shipped float codec, the arena tier, is
//! `MAX_BLOCK = 1`, which `tabulation_pays` refuses outright, so the float
//! table route has never run outside tests. A codebook of `S` codewords of
//! `BLK` symbols each makes the slab `S * rows * size_of::<AccOf<f32>>()` ---
//! a few kilobytes at `S` of 4 or 16, inside L1 --- and turns the column loop
//! into one table read and one exact accumulator combine per `BLK` products.
//!
//! It lives here and not in `uor-matmul-codec` because nothing about it is a
//! claim yet: the measurement decides whether a real tier is worth building
//! (R9), and until then there is no ID, no scenario, and no shipped code.
//! Mechanically it is [`uor_matmul_codec::Book`] transcribed from
//! `IntegerElement` to `f32` --- the shipped `Book` is generic over integer
//! alphabets only, and a float's bound is [`Whole`].

use uor_matmul_codec::{Codec, Enumerable, TierId};
use uor_matmul_core::{Alphabet, Whole};

/// A codebook of `S` codewords of `BLK` `f32` symbols each, `Code = u16`.
///
/// `S` must be a power of two: `index_of` is then a mask, the stored `u16`
/// *is* the index, and [`Enumerable::as_index_stream`] borrows the operand's
/// own memory --- the same rule the shipped codecs follow
/// ([`uor_matmul_codec::Book`], [`uor_matmul_codec::Arena`]).
#[derive(Clone, Copy, Debug)]
pub struct FloatBook<'a, const S: usize, const BLK: usize> {
    table: &'a [[Alphabet<f32, Whole<f32>>; BLK]; S],
}

impl<'a, const S: usize, const BLK: usize> FloatBook<'a, S, BLK> {
    /// Borrow a codebook. Panics unless `S` is a power of two in `2..=65536`
    /// and `BLK` is at least one --- the invariants the enumeration below is
    /// written against, checked at construction because a dev instrument
    /// fails loud rather than declining.
    pub fn new(table: &'a [[Alphabet<f32, Whole<f32>>; BLK]; S]) -> Self {
        assert!(
            S.is_power_of_two() && S >= 2 && S <= 65536,
            "the code space must be a power of two the u16 code can address"
        );
        assert!(BLK >= 1, "a codeword names at least one symbol");
        Self { table }
    }

    /// The codebook.
    pub const fn table(&self) -> &'a [[Alphabet<f32, Whole<f32>>; BLK]; S] {
        self.table
    }
}

impl<const S: usize, const BLK: usize> Codec<f32, Whole<f32>> for FloatBook<'_, S, BLK> {
    type Code = u16;
    const MAX_BLOCK: usize = BLK;
    // A codebook of any entry count and any block size is what `Book` names;
    // the label is never a dispatch key, so the experiment reports as the
    // shape it has.
    const TIER: TierId = TierId::Book;

    fn decode_element(&self, code: Self::Code, i: usize) -> Alphabet<f32, Whole<f32>> {
        // Total for every `u16`, as the shipped codecs are: the mask is the
        // same reduction `index_of` performs, so equal indices and equal
        // decodes are the same relation (C6).
        self.table[(code as usize) & (S - 1)][i % BLK]
    }

    fn decode_into(&self, code: Self::Code, out: &mut [Alphabet<f32, Whole<f32>>]) -> usize {
        out[..BLK].copy_from_slice(&self.table[(code as usize) & (S - 1)]);
        BLK
    }
}

/// Build a codebook from raw symbols: `table[c][t]` is `symbols[c * BLK + t]`.
///
/// Free of any canonicalization discipline on purpose --- the arena tier owns
/// that claim (`CK-10`), and this instrument needs only a table whose entries
/// are distinct enough to make a wrong index visible.
pub fn codebook<const S: usize, const BLK: usize>(
    symbols: &[f32],
) -> [[Alphabet<f32, Whole<f32>>; BLK]; S] {
    assert_eq!(
        symbols.len(),
        S * BLK,
        "a codebook is S codewords of BLK symbols"
    );
    std::array::from_fn(|c| {
        std::array::from_fn(|t| Alphabet::<f32, Whole<f32>>::symbol(symbols[c * BLK + t]))
    })
}

impl<const S: usize, const BLK: usize> Enumerable<f32, Whole<f32>> for FloatBook<'_, S, BLK> {
    const CODE_SPACE: usize = S;

    fn code_at(index: usize) -> Self::Code {
        // `index < CODE_SPACE <= 65536`, so the cast is the identity on the
        // domain the law quantifies over.
        (index % S.max(1)) as u16
    }

    fn index_of(code: Self::Code) -> usize {
        // The mask, total for every `u16`: the constructor has already
        // established that `S` is a power of two.
        (code as usize) & (S - 1)
    }

    fn as_index_stream(codes: &[u16]) -> Option<&[u16]> {
        // `index_of` *is* the mask, so the stored stream already addresses
        // the enumeration and there is nothing to build.
        Some(codes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uor_matmul_codec::CodedMatrix;
    use uor_matmul_core::{
        as_alphabet_whole, AccOf, Accumulator, FloatElement, MatView, MatViewMut, Shape, Traversal,
        Triple,
    };
    use uor_matmul_gemm::{
        gemm_float, gemm_tabulated_counted, suggested_tabulation, suggested_tabulation_index,
        suggested_tabulation_panel, Census, Collapse, GemmOptions, Linear, Scratch,
        TabulatedTriple, Tabulation,
    };

    /// The xorshift fill every sweep in this workspace uses, so two runs of
    /// the experiment see the same operands.
    pub(crate) fn fill(len: usize, salt: u64) -> Vec<u64> {
        let mut s = 0x243F_6A88_85A3_08D3u64 ^ salt;
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                s >> 33
            })
            .collect()
    }

    /// The symbols one codeword can name, deterministic and small: a handful
    /// of magnitudes across a few exponents, so the exact accumulator's
    /// placement step is exercised rather than dodged.
    pub(crate) fn symbols(len: usize, salt: u64) -> Vec<f32> {
        fill(len, salt)
            .into_iter()
            .map(|x| ((x % 2000) as f32 - 1000.0) / 500.0)
            .collect()
    }

    #[test]
    fn float_book_obeys_the_enumeration_laws() {
        type C<'a> = FloatBook<'a, 4, 4>;
        let table = codebook::<4, 4>(&symbols(16, 0xb00c));
        let codec = C::new(&table);
        for (i, word) in table.iter().enumerate() {
            assert_eq!(
                <C as Enumerable<f32, Whole<f32>>>::index_of(
                    <C as Enumerable<f32, Whole<f32>>>::code_at(i)
                ),
                i
            );
            for (t, &symbol) in word.iter().enumerate() {
                assert_eq!(
                    codec.decode_element(<C as Enumerable<f32, Whole<f32>>>::code_at(i), t),
                    symbol
                );
            }
        }
        // Totality: every `u16`, including ones no encoder produced, lands
        // below the code space.
        for code in [0u16, 1, 3, 4, 5, 255, 4096, u16::MAX] {
            assert!(<C as Enumerable<f32, Whole<f32>>>::index_of(code) < 4);
        }
    }

    /// The sanity the whole experiment stands on: a forced tabulated float
    /// run is byte-identical to the dense float driver over the decoded
    /// weights, and the census proves the table really ran.
    #[test]
    fn forced_float_tabulation_matches_the_dense_driver() {
        const S: usize = 4;
        const BLK: usize = 4;
        let (m, k, n) = (3usize, 8usize, 5usize);
        let shape = Shape { m, k, n };

        let table = codebook::<S, BLK>(&symbols(S * BLK, 0xb00c));
        let codec = FloatBook::<'_, S, BLK>::new(&table);
        let codes: Vec<u16> = fill(n * (k / BLK), 0xc0de)
            .into_iter()
            .map(|x| x as u16)
            .collect();
        let w = CodedMatrix::new(codec, n, k, &codes).expect("the codes describe n x k");
        let a = symbols(m * k, 0xa11);

        // The dense float driver over the decoded weights: the bytes every
        // traversal of this product must reproduce.
        let mut b = vec![0.0f32; k * n];
        for p in 0..k {
            for j in 0..n {
                b[p * n + j] = w.at(j, p).get();
            }
        }
        let mut want = vec![0.0f32; m * n];
        {
            let av = MatView::row_major(&a, m, k).unwrap();
            let bv = MatView::row_major(&b, k, n).unwrap();
            let cv = MatViewMut::row_major(&mut want, m, n).unwrap();
            let mut tr = Triple::new(av, bv, cv).unwrap();
            gemm_float(&mut tr, &Linear::OVERWRITE, GemmOptions::default());
        }

        let mut accumulators = vec![
            <AccOf<f32> as Accumulator>::ZERO;
            suggested_tabulation::<f32, Whole<f32>>(shape, S, BLK,)
        ];
        let mut ids = vec![0usize; suggested_tabulation_index(shape)];
        let mut panel = vec![Alphabet::<f32, Whole<f32>>::ZERO; suggested_tabulation_panel(S, BLK)];
        let mut got = vec![0.0f32; m * n];
        let mut census = Census::default();
        {
            let av = MatView::row_major(as_alphabet_whole(&a), m, k).unwrap();
            let cv = MatViewMut::row_major(&mut got, m, n).unwrap();
            let mut tr = TabulatedTriple::new(av, w, cv).unwrap();
            gemm_tabulated_counted(
                &mut tr,
                &Linear::OVERWRITE,
                GemmOptions {
                    traversal: Traversal::Tabulated,
                    ..Default::default()
                },
                &mut Scratch::with_accumulators(&mut panel, &mut accumulators),
                &mut Tabulation::with_index(&mut [], &mut ids),
                &mut Collapse::none(),
                &mut census,
            );
        }
        let want_bits: Vec<u64> = want.iter().map(|v| v.symbol_bits()).collect();
        let got_bits: Vec<u64> = got.iter().map(|v| v.symbol_bits()).collect();
        assert_eq!(
            got_bits, want_bits,
            "the forced table must give the dense float driver's bytes ({census:?})"
        );
        assert!(
            census.table_reads > 0,
            "the offer was sized for a table and none was read ({census:?})"
        );
    }
}
