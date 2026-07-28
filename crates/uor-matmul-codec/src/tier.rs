//! The one trait every tier instantiates (§6.1).

use uor_matmul_core::{Alphabet, Bound, Element};

/// A decode from a stored code to alphabet elements.
///
/// `Bd` being a parameter of the trait is what makes the alphabet bound a
/// type-level fact the kernels can rely on without rechecking: a codec's table
/// is `&[Alphabet<E, Bd>]`, so its image is in the alphabet by construction and
/// there is nothing for a constructor to validate or reject (§6.2).
///
/// `E` is any element type. An integer codec's alphabet is a magnitude bound;
/// a float codec's is the codebook itself, declared as `Whole<E>` --- the
/// arena tier is the one instantiation of that, and no tier branches on the
/// difference.
pub trait Codec<E: Element, Bd: Bound>: Send + Sync {
    /// The stored code type.
    ///
    /// Any `Copy` type: `i8` for identity, `u8` for a nibble pair or an E8
    /// index, `u16` for a 65536-entry codebook. The library carries no
    /// hardcoded code width.
    type Code: Copy + Send + Sync + 'static;

    /// The most alphabet elements one code can produce.
    ///
    /// 1 for scalar, 2 for nibble-packed, 8 for E8, arbitrary in general. This
    /// is the const that makes a scalar codec the block codec of its
    /// singletons, so that one identity and one kernel cover both (`CL-MM02`).
    ///
    /// It is a *maximum*, not a fixed width: a variable-length codec ---
    /// a run codec, for one --- produces fewer than `MAX_BLOCK` elements for
    /// some codes, and [`Codec::decode_into`] returns how many it produced. A
    /// fixed width would have made run coding a second algorithm rather than a
    /// tier (S4, S5b).
    const MAX_BLOCK: usize;

    /// Which tier this is, for reports and for the kappa manifest.
    const TIER: TierId;

    /// Decode element `i` of the block `code` names. Total for every `i`.
    ///
    /// This is the trait's one required method rather than the block decode,
    /// so that a composing tier --- [`crate::Offset`], [`crate::Runs`],
    /// [`crate::Transcode`] --- can defer to its inner codec without owning a
    /// scratch buffer the size of a block. No allocation is possible anywhere
    /// in this crate, so a required block decode would have forced either a
    /// hardcoded maximum block size or an `alloc` dependency, and both are
    /// arbitrary limitations (R7, R8).
    ///
    /// There is no code value of `Self::Code` and no `i` for which this fails:
    /// a codec whose table did not cover its code space could not have been
    /// constructed, and `i >= BLOCK` is answered by the block's own padding.
    fn decode_element(&self, code: Self::Code, i: usize) -> Alphabet<E, Bd>;

    /// Is `decode_len` always `MAX_BLOCK`?
    ///
    /// True for every fixed-width tier, which is all of them except a run
    /// codec. It is what lets [`crate::CodedMatrix`] find the codes of row `r`
    /// by arithmetic rather than by walking, and the difference is not a
    /// constant factor: a walk makes random access O(row length), and a driver
    /// that reads one element at a time then runs in O(k^2 n) instead of
    /// O(m k n). No driver here does: `decode_row_into` and
    /// [`crate::CodedMatrix::column_walk`] each walk once and carry the cursor,
    /// and the coded traversal uses the second --- measured at 215x on a 512-row
    /// run matrix, a factor that grows with the row count.
    const IS_FIXED_WIDTH: bool = true;

    /// How many elements `code` actually produces.
    ///
    /// `MAX_BLOCK` for a fixed-width tier, and possibly fewer for a
    /// variable-length one. `CK-06` asserts that these counts sum to the
    /// declared row width on every row, which is the invariant that lets a run
    /// codec live inside this trait rather than beside it.
    fn decode_len(&self, _code: Self::Code) -> usize {
        Self::MAX_BLOCK
    }

    /// Decode one code, returning how many elements were written.
    ///
    /// `out.len() >= decode_len(code)`. Total, side-effect free, and
    /// allocation-free. The default loops [`Codec::decode_element`]; a tier
    /// overrides it only when it can produce the same bytes faster, never
    /// differently.
    fn decode_into(&self, code: Self::Code, out: &mut [Alphabet<E, Bd>]) -> usize {
        let n = self.decode_len(code).min(out.len());
        for (i, slot) in out.iter_mut().enumerate().take(n) {
            *slot = self.decode_element(code, i);
        }
        n
    }

    /// Bulk path, returning how many elements were written in total.
    fn decode_seq(&self, codes: &[Self::Code], out: &mut [Alphabet<E, Bd>]) -> usize {
        let mut at = 0usize;
        for &code in codes {
            at += self.decode_into(code, &mut out[at..]);
        }
        at
    }
}

/// A codec whose code space can be enumerated.
///
/// [`Codec`] decodes *from* a code. That is the whole of what a
/// decode-then-multiply driver needs, and it is why a table indexed by code
/// cannot be written against [`Codec`] alone: such a table requires the *set* of
/// codes, and nothing in [`Codec`] hands it over. This trait is that set.
///
/// # Why this is separate from [`Codec`]
///
/// Not every codec should be tabulated. [`crate::Identity`] over `i32` has a
/// code space of `2^32`, and a trait that admitted it would invite a table
/// nobody can hold. Requiring this trait at the tabulated traversal's boundary
/// makes "this codec cannot be tabulated" a compile-time fact rather than a
/// runtime refusal.
///
/// # Laws
///
/// 1. `index_of(code_at(i)) == i` for every `i < CODE_SPACE`.
/// 2. `index_of` is total: every value of [`Codec::Code`] the codec can hold
///    lands strictly below `CODE_SPACE`. This is what makes the tabulated
///    traversal total --- no code can miss the table (`CT-07`).
///
/// Both are asserted by `CK-09`. Note what is *not* a law: `code_at` need not be
/// injective *on decodes*. Two indices may decode alike, in which case the table
/// carries a dead entry. That is not an error, it is a cost, and `CG-10` reports
/// the ratio.
///
/// # The enumeration is stateless
///
/// Both methods are associated functions rather than methods on `&self`, because
/// the enumeration is a property of the code *space* and not of any particular
/// table. A composing tier defers to its inner codec's enumeration without
/// holding an instance of it.
pub trait Enumerable<E: Element, Bd: Bound>: Codec<E, Bd> {
    /// The number of distinct codes.
    ///
    /// `N` for an `N`-entry [`crate::Grid`] or [`crate::Book`], and the product
    /// of the sub-spaces for a [`crate::Packed`] byte. Never larger than the
    /// code type can address: an enumeration wider than its own code type would
    /// name codes that cannot be stored.
    const CODE_SPACE: usize;

    /// The `index`-th code. Total for `index < CODE_SPACE`.
    fn code_at(index: usize) -> Self::Code;

    /// Where `code` sits in the enumeration.
    ///
    /// Total, and total *into the enumeration*: for every value of
    /// [`Codec::Code`], including values no encoder would produce, the answer is
    /// below [`Self::CODE_SPACE`]. Both halves are law, and `CK-09` asserts
    /// them.
    ///
    /// That is not politeness. The tabulated traversal reads its table with no
    /// bounds check and no branch, and this is what makes the read correct: the
    /// mask that makes it *safe* holds unconditionally, and this is what makes
    /// the entry it lands on the right one.
    fn index_of(code: Self::Code) -> usize;

    /// The stored code stream, read as the index stream it already is.
    ///
    /// `Some` when the code *addresses* the enumeration: when
    /// `index_of(c) == (c as usize) & (CODE_SPACE - 1)` for every `c`, which
    /// needs `CODE_SPACE` to be a power of two and the enumeration to be the
    /// code type's own order. `None` otherwise --- a [`crate::Packed`] byte, for
    /// one, whose index is a mixed-radix decomposition of it and not the byte.
    ///
    /// This is the same rule [`uor_matmul_core::MatView::row_block`] follows on
    /// the dense side: *borrow when the layout already holds what is wanted,
    /// copy otherwise*. A tabulated traversal addresses its table from an index
    /// stream, and when the operand's own memory is one there is nothing to
    /// build. Measured at a one-row tile, where the index a traversal would
    /// materialize is as wide as the table entry it addresses, that pass was
    /// two thirds of the work.
    ///
    /// The default is `None`, so a codec says nothing by saying nothing and the
    /// traversal builds the stream. `CK-09` asserts the claim of any codec that
    /// does answer `Some`.
    fn as_index_stream(codes: &[Self::Code]) -> Option<&[u16]> {
        let _ = codes;
        None
    }
}

/// Which tier a codec is.
///
/// A label, never a dispatch key: nothing in the library branches on it, and
/// two codecs with different `TierId`s and equal decodes produce byte-identical
/// output (`CK-05`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[non_exhaustive]
pub enum TierId {
    /// Decoding is a validated copy.
    Identity,
    /// A lookup table of any code width.
    Grid,
    /// Sub-codes unpacked from one stored byte.
    Packed,
    /// A codebook of any entry count and any block size.
    Book,
    /// `d(c) - z`: asymmetric quantization as a codec composition.
    Offset,
    /// Sparse storage as a codec.
    Runs,
    /// The composite of two codecs.
    Transcode,
    /// A codebook of one artifact's distinct bit patterns, canonicalized.
    Arena,
}

impl TierId {
    /// The token used in the kappa manifest and in reports.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Identity => "Identity",
            Self::Grid => "Grid",
            Self::Packed => "Packed",
            Self::Book => "Book",
            Self::Offset => "Offset",
            Self::Runs => "Runs",
            Self::Transcode => "Transcode",
            Self::Arena => "Arena",
        }
    }
}
