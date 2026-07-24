//! Coded matrices (§6.3).

use core::ops::Range;

use uor_matmul_core::{Alphabet, Bound, IntegerElement};

use crate::tier::Codec;

/// A borrowed matrix of codes, together with the codec that decodes them.
///
/// The codes are borrowed and the codec's table is borrowed, so a
/// `CodedMatrix` is a handful of pointers and three integers. Nothing here is
/// owned, nothing is copied, and nothing is allocated (R7, C1).
#[derive(Clone, Copy, Debug)]
pub struct CodedMatrix<'a, E: IntegerElement, Bd: Bound, C: Codec<E, Bd>> {
    codec: C,
    rows: usize,
    cols: usize,
    codes: &'a [C::Code],
    _marker: core::marker::PhantomData<fn() -> (E, Bd)>,
}

impl<'a, E: IntegerElement, Bd: Bound, C: Codec<E, Bd>> CodedMatrix<'a, E, Bd, C> {
    /// Borrow `codes` as an `rows x cols` coded matrix.
    ///
    /// `None` only when the codes do not describe the declared shape, which
    /// means no such matrix exists. There is nothing else to validate: the
    /// codec's table is already `Alphabet<E, Bd>`, so its image is in the
    /// alphabet by construction (§6.3).
    ///
    /// `CK-06`: the codec's own decoded lengths must sum to the declared row
    /// width, on every row. It is the *codec* that says how long a code is, so
    /// a variable-length tier needs no special case here and no separate matrix
    /// type --- which is what makes run coding a tier rather than a second
    /// algorithm (S5b).
    pub fn new(codec: C, rows: usize, cols: usize, codes: &'a [C::Code]) -> Option<Self> {
        if C::MAX_BLOCK == 0 {
            return None;
        }
        let mut at = 0usize;
        for _ in 0..rows {
            let mut width = 0usize;
            while width < cols {
                let code = *codes.get(at)?;
                let n = codec.decode_len(code);
                if n == 0 {
                    // A code that produces nothing would make the walk
                    // non-terminating; no such matrix exists.
                    return None;
                }
                width = width.checked_add(n)?;
                at = at.checked_add(1)?;
            }
            if width != cols {
                // The last code of the row overshot it: the codes describe a
                // different shape from the declared one.
                return None;
            }
        }
        if at != codes.len() {
            return None;
        }
        Some(Self {
            codec,
            rows,
            cols,
            codes,
            _marker: core::marker::PhantomData,
        })
    }

    /// Rows.
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Decoded elements per row.
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// The codec.
    pub const fn codec(&self) -> &C {
        &self.codec
    }

    /// Codes per row, for a fixed-width tier.
    ///
    /// A variable-length tier has no such constant; use
    /// [`CodedMatrix::row_code_range`], which walks the codec's own lengths.
    pub const fn codes_per_row(&self) -> usize {
        self.cols / C::MAX_BLOCK
    }

    /// The half-open range of codes belonging to row `r`.
    ///
    /// Walks [`Codec::decode_len`], so it is correct for a fixed-width tier and
    /// for a variable-length one without branching on which it has.
    pub fn row_code_range(&self, r: usize) -> core::ops::Range<usize> {
        let mut at = 0usize;
        for _ in 0..r {
            let mut width = 0usize;
            while width < self.cols {
                width += self.codec.decode_len(self.codes[at]);
                at += 1;
            }
        }
        let start = at;
        let mut width = 0usize;
        while width < self.cols {
            width += self.codec.decode_len(self.codes[at]);
            at += 1;
        }
        start..at
    }

    /// The raw code slice.
    pub const fn codes(&self) -> &'a [C::Code] {
        self.codes
    }

    /// Decode row `r`. `out.len() >= cols`. The caller owns the buffer.
    ///
    /// Returns how many elements were written, which `CodedMatrix::new`
    /// established is exactly `cols` (`CK-06`).
    pub fn decode_row_into(&self, r: usize, out: &mut [Alphabet<E, Bd>]) -> usize {
        let range = self.row_code_range(r);
        self.codec.decode_seq(&self.codes[range], out)
    }

    /// Streaming decode, for a caller whose buffer is smaller than one row.
    ///
    /// This is what makes the library usable on a microcontroller whose RAM
    /// cannot hold a decoded row, and it is what makes the zero-scratch
    /// traversal possible (S13).
    pub fn decode_range_into(&self, r: usize, cols: Range<usize>, out: &mut [Alphabet<E, Bd>]) {
        for (slot, col) in out.iter_mut().zip(cols) {
            *slot = self.at(r, col);
        }
    }

    /// The element at `(r, c)` of the decoded matrix.
    ///
    /// Walks the codec's own lengths from the start of the row, so it is
    /// correct for a variable-length tier. A caller decoding a whole row should
    /// use [`CodedMatrix::decode_row_into`], which walks it once.
    pub fn at(&self, r: usize, c: usize) -> Alphabet<E, Bd> {
        let range = self.row_code_range(r);
        let mut width = 0usize;
        for &code in &self.codes[range] {
            let n = self.codec.decode_len(code);
            if c < width + n {
                return self.codec.decode_element(code, c - width);
            }
            width += n;
        }
        Alphabet::ZERO
    }
}
