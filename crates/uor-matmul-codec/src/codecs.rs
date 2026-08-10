//! The tiers (§6.2).
//!
//! Instantiations, not special cases. Each is roughly twenty lines, and none of
//! them contains any arithmetic that the others do not: a tier names a decode,
//! and the accumulation downstream of it is the same accumulation for every
//! tier in this file.

use core::marker::PhantomData;

use bytemuck::TransparentWrapper as _;
use uor_matmul_core::{Alphabet, Bound, FloatElement, IntegerElement, Whole};

use crate::tier::{Codec, Enumerable, IndexStream, TierId};

/// How many codes a `u16`-indexed table can actually be reached with.
///
/// A property of the code type, not a limit on the table: a `Grid<70000>` is a
/// well-formed decode whose entries past this are unreachable, and saying so is
/// how the enumeration stays honest about its own size.
const U16_CODES: usize = u16::MAX as usize + 1;

/// The three parameters `Offset` carries without storing: the input bound, the
/// output bound, and the element type. Invariant in none of them, so the marker
/// is a function pointer rather than a `PhantomData<T>`.
type OffsetMarker<BdIn, BdOut, E> = PhantomData<fn() -> (BdIn, BdOut, E)>;

/// The same marker for the width-parameterized constant-table tiers: the
/// element, the bound, and the code width, none of them stored.
type TierMarker<E, Bd, K> = PhantomData<fn() -> (E, Bd, K)>;

// ---------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------

/// Decoding is a validated copy.
///
/// Zero-sized. The validation is the alphabet wrap itself, performed once at
/// the boundary by `as_alphabet_full` (which cannot fail) or `as_alphabet`
/// (which reports the observed bound rather than failing), so the code type is
/// [`Alphabet`] and there is nothing left for the decode to check (§5.2, §6.2).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Identity;

impl<E: IntegerElement, Bd: Bound> Codec<E, Bd> for Identity {
    type Code = Alphabet<E, Bd>;
    const MAX_BLOCK: usize = 1;
    const TIER: TierId = TierId::Identity;

    fn decode_element(&self, code: Self::Code, _i: usize) -> Alphabet<E, Bd> {
        code
    }

    fn decode_seq(&self, codes: &[Self::Code], out: &mut [Alphabet<E, Bd>]) -> usize {
        out[..codes.len()].copy_from_slice(codes);
        codes.len()
    }
}

// `Identity` does not implement `Enumerable`, deliberately. Its code space is
// the alphabet itself --- `2^32` values over `i32` --- and its block is one, so
// a table indexed by it would be both unholdable and pointless: `tabulation_pays`
// refuses `MAX_BLOCK == 1` on op count anyway. The absence is enforced by the
// type system rather than by a runtime refusal, which is why the tabulated
// traversal needs no branch for it (§4).

// ---------------------------------------------------------------------------
// Grid
// ---------------------------------------------------------------------------

/// Any lookup codec, of any code width.
///
/// The 16-entry i4 grid is `Grid<16>`; a 65536-entry one is `Grid<65536>`. The
/// library carries no hardcoded table size, and no size is privileged.
///
/// The code space of an `N`-entry grid is `Z/N`, so an arbitrary `u16` indexes
/// it modulo `N`. That is not a clamp and not an error path: it is what
/// "index into a table of `N` entries" means, and it is why this decode is
/// total on all `2^16` codes (C6).
#[derive(Clone, Copy, Debug)]
pub struct Grid<'a, E: IntegerElement, Bd: Bound, const N: usize> {
    table: &'a [Alphabet<E, Bd>; N],
}

impl<'a, E: IntegerElement, Bd: Bound, const N: usize> Grid<'a, E, Bd, N> {
    /// Borrow a decode table. The table is `Alphabet<E, Bd>`, so its image is
    /// in the alphabet by construction and there is nothing to validate.
    pub const fn new(table: &'a [Alphabet<E, Bd>; N]) -> Self {
        Self { table }
    }

    /// The table.
    pub const fn table(&self) -> &'a [Alphabet<E, Bd>; N] {
        self.table
    }
}

impl<E: IntegerElement, Bd: Bound, const N: usize> Codec<E, Bd> for Grid<'_, E, Bd, N> {
    type Code = u16;
    const MAX_BLOCK: usize = 1;
    const TIER: TierId = TierId::Grid;

    fn decode_element(&self, code: Self::Code, _i: usize) -> Alphabet<E, Bd> {
        self.table[(code as usize) % N]
    }
}

impl<E: IntegerElement, Bd: Bound, const N: usize> Enumerable<E, Bd> for Grid<'_, E, Bd, N> {
    // The reachable code space, which is the table size unless the table is
    // wider than the code type can address.
    const CODE_SPACE: usize = if N < U16_CODES { N } else { U16_CODES };

    fn code_at(index: usize) -> Self::Code {
        // `index < CODE_SPACE <= U16_CODES`, so the cast is the identity on the
        // enumeration's own domain. The `%` is what makes it total off it, and
        // the `max(1)` keeps it total for a table with no entries at all --- a
        // codec `tabulation_pays` refuses, so no traversal reaches it.
        (index % Self::CODE_SPACE.max(1)) as u16
    }

    fn index_of(code: Self::Code) -> usize {
        // The same reduction `decode_element` performs, so equal indices and
        // equal decodes are the same relation.
        (code as usize) % Self::CODE_SPACE.max(1)
    }

    fn as_index_stream(codes: &[u16]) -> Option<IndexStream<'_>> {
        // `index_of` is `% CODE_SPACE`, which is `& (CODE_SPACE - 1)` exactly
        // when the space is a power of two --- and then the stored `u16` is the
        // index, masked. At any other entry count the two differ and the
        // traversal builds the stream, at the same bytes.
        Self::CODE_SPACE
            .is_power_of_two()
            .then_some(IndexStream::U16(codes))
    }
}

// ---------------------------------------------------------------------------
// Packed
// ---------------------------------------------------------------------------

/// Unpacks `P` sub-codes from one stored byte, then defers to `C`.
///
/// The i4 tier is `Packed<Grid<16>, 2>`. **Low sub-code first** --- normative,
/// pinned by case `nibble-order` and asserted by `CK-03`.
///
/// `P` is arbitrary: two nibbles, four two-bit codes, eight one-bit codes. The
/// sub-code width is `8 / P`, so `P` must divide 8, which [`Packed::new`]
/// checks once at construction.
#[derive(Clone, Copy, Debug)]
pub struct Packed<C, const P: usize> {
    inner: C,
}

impl<C, const P: usize> Packed<C, P> {
    /// Bits per sub-code.
    pub const SUB_BITS: u32 = (8 / P) as u32;

    /// Wrap `inner`, unpacking `P` sub-codes per byte.
    ///
    /// `None` when `P` does not divide 8, which means no such packing exists.
    pub fn new(inner: C) -> Option<Self> {
        if P == 0 || 8 % P != 0 {
            return None;
        }
        Some(Self { inner })
    }
}

impl<E, Bd, C, const P: usize> Codec<E, Bd> for Packed<C, P>
where
    E: IntegerElement,
    Bd: Bound,
    C: Codec<E, Bd>,
    C::Code: From<u8>,
{
    type Code = u8;
    const MAX_BLOCK: usize = P * C::MAX_BLOCK;
    const TIER: TierId = TierId::Packed;
    const IS_FIXED_WIDTH: bool = C::IS_FIXED_WIDTH;

    fn decode_element(&self, code: Self::Code, i: usize) -> Alphabet<E, Bd> {
        let bits = Self::SUB_BITS;
        let mask: u8 = if bits >= 8 {
            u8::MAX
        } else {
            (1u8 << bits) - 1
        };
        // Low sub-code first. This is normative: a reader that took the high
        // sub-code first would decode a different matrix from the same bytes,
        // so the order is pinned rather than left to convention (CK-03).
        let p = i / C::MAX_BLOCK;
        let sub = (code >> (bits * p as u32)) & mask;
        self.inner
            .decode_element(C::Code::from(sub), i % C::MAX_BLOCK)
    }
}

/// How many distinct sub-codes one field of a packed byte can store.
///
/// The inner codec's own space, unless the field is narrower than that space ---
/// a `Grid<256>` packed two to a byte can only ever be handed a nibble, so its
/// reachable sub-space is sixteen and not two hundred and fifty-six. Getting
/// this wrong in either direction breaks a law: too large and `code_at` names
/// codes the byte cannot hold, too small and `index_of` collides.
const fn sub_space(inner: usize, sub_bits: u32) -> usize {
    let per_field = 1usize << sub_bits;
    if inner < per_field {
        inner
    } else {
        per_field
    }
}

impl<E, Bd, C, const P: usize> Enumerable<E, Bd> for Packed<C, P>
where
    E: IntegerElement,
    Bd: Bound,
    C: Enumerable<E, Bd>,
    C::Code: From<u8>,
{
    // Mixed radix in the sub-space, which is at most `2^(SUB_BITS * P) = 256`,
    // so this cannot overflow whatever `P` and `C` are.
    const CODE_SPACE: usize = sub_space(C::CODE_SPACE, Self::SUB_BITS).pow(P as u32);

    fn code_at(index: usize) -> Self::Code {
        let bits = Self::SUB_BITS;
        let radix = sub_space(C::CODE_SPACE, bits).max(1);
        let mut rest = index % Self::CODE_SPACE.max(1);
        let mut code = 0u8;
        for field in 0..P {
            let digit = rest % radix;
            rest /= radix;
            // Low sub-code first, the same order `decode_element` reads them in.
            code |= (digit as u8) << (bits * field as u32);
        }
        code
    }

    fn index_of(code: Self::Code) -> usize {
        let bits = Self::SUB_BITS;
        let mask: u8 = if bits >= 8 {
            u8::MAX
        } else {
            (1u8 << bits) - 1
        };
        let radix = sub_space(C::CODE_SPACE, bits).max(1);
        let mut index = 0usize;
        let mut place = 1usize;
        for field in 0..P {
            let sub = (code >> (bits * field as u32)) & mask;
            let digit = C::index_of(C::Code::from(sub)) % radix;
            index += digit * place;
            place *= radix;
        }
        index
    }
}

// ---------------------------------------------------------------------------
// Book
// ---------------------------------------------------------------------------

/// Any codebook: `N` entries of `BLK` alphabet elements each.
///
/// E8 is `Book<256, 8>`. Nothing privileges that shape; `BLK` and `N` are
/// parameters, and no quality claim attaches to any table (N3).
#[derive(Clone, Copy, Debug)]
pub struct Book<
    'a,
    E: IntegerElement,
    Bd: Bound,
    const N: usize,
    const BLK: usize,
    K: SymbolCode = u16,
> {
    table: &'a [[Alphabet<E, Bd>; BLK]; N],
    // The width is a parameter of the decode, not of the table: the struct
    // stores nothing of it.
    _code: PhantomData<fn() -> K>,
}

impl<'a, E: IntegerElement, Bd: Bound, const N: usize, const BLK: usize, K: SymbolCode>
    Book<'a, E, Bd, N, BLK, K>
{
    /// Borrow a codebook. On an embedded target this is a pointer into flash.
    pub const fn new(table: &'a [[Alphabet<E, Bd>; BLK]; N]) -> Self {
        Self {
            table,
            _code: PhantomData,
        }
    }

    /// The codebook.
    pub const fn table(&self) -> &'a [[Alphabet<E, Bd>; BLK]; N] {
        self.table
    }
}

impl<E: IntegerElement, Bd: Bound, const N: usize, const BLK: usize, K: SymbolCode> Codec<E, Bd>
    for Book<'_, E, Bd, N, BLK, K>
{
    type Code = K;
    const MAX_BLOCK: usize = BLK;
    const TIER: TierId = TierId::Book;

    fn decode_element(&self, code: Self::Code, i: usize) -> Alphabet<E, Bd> {
        self.table[K::index(code) % N][i % BLK]
    }

    fn decode_into(&self, code: Self::Code, out: &mut [Alphabet<E, Bd>]) -> usize {
        out[..BLK].copy_from_slice(&self.table[K::index(code) % N]);
        BLK
    }
}

impl<E: IntegerElement, Bd: Bound, const N: usize, const BLK: usize, K: SymbolCode>
    Enumerable<E, Bd> for Book<'_, E, Bd, N, BLK, K>
{
    const CODE_SPACE: usize = if N < K::CODES { N } else { K::CODES };

    fn code_at(index: usize) -> Self::Code {
        K::at(index % Self::CODE_SPACE.max(1))
    }

    fn index_of(code: Self::Code) -> usize {
        K::index(code) % Self::CODE_SPACE.max(1)
    }

    fn as_index_stream(codes: &[K]) -> Option<IndexStream<'_>> {
        // `index_of` is `% CODE_SPACE`, which is `& (CODE_SPACE - 1)` exactly
        // when the space is a power of two --- and then the stored code, at
        // either width, is the index, masked. At any other entry count the two
        // differ and the traversal builds the stream, at the same bytes.
        K::index_stream(codes, Self::CODE_SPACE)
    }
}

// ---------------------------------------------------------------------------
// Sign
// ---------------------------------------------------------------------------

/// Weights in `{-1, +1}`, one bit per element, with no table at all.
///
/// The `Packed<Grid<2>,8>` spelling of the same decode (`CK-13`) tabulates but
/// can never answer [`Enumerable::as_index_stream`]: a packed byte's index is
/// a mixed-radix decomposition of it, so the tabulated traversal builds the
/// index stream it gathers from. Measured, that build is a fifth to a third of
/// the work at a one-row tile (ANALYSIS.md, "What the missing index stream
/// costs the sign composition"). This tier spells the same decode with the
/// code *being* the index --- `Code = u16`, one bit per element of the block
/// --- so the operand's own memory is the stream and there is nothing to
/// build. `CK-11` pins the two spellings byte for byte.
///
/// The codebook is the constant `{-1, +1}`, so there is nothing to borrow and
/// no lifetime: the decode is a bit test. **Bit `t` of the code is element
/// `t`; set is `+1`, clear is `-1`, low bit first** --- the order `Packed`
/// reads its sub-codes in (`CK-03`), so a `Sign` code spells the block the
/// composition spells from the same byte value. The convention is normative:
/// the encoder side is out of tree, and this decode is the whole contract
/// with it.
#[derive(Clone, Copy, Debug)]
pub struct Sign<E: IntegerElement, Bd: Bound, const BLK: usize, K: SymbolCode = u16> {
    _marker: TierMarker<E, Bd, K>,
}

impl<E: IntegerElement, Bd: Bound, const BLK: usize, K: SymbolCode> Sign<E, Bd, BLK, K> {
    /// The one non-existence check: an alphabet that does not admit `+-1` has
    /// no sign codec. Decided at construction, before any arithmetic, like
    /// every other non-existence in this library (C6).
    pub const fn new() -> Option<Self> {
        if Bd::VALUE < 1 || E::FULL < 1 {
            return None;
        }
        Some(Self {
            _marker: PhantomData,
        })
    }
}

impl<E: IntegerElement, Bd: Bound, const BLK: usize, K: SymbolCode> Codec<E, Bd>
    for Sign<E, Bd, BLK, K>
{
    type Code = K;
    const MAX_BLOCK: usize = BLK;
    const TIER: TierId = TierId::Sign;

    fn decode_element(&self, code: Self::Code, i: usize) -> Alphabet<E, Bd> {
        let bit = i % BLK;
        // The guard keeps the shift total past the code type's own width: at
        // `BLK > K::BITS` a code holds no bit for those positions, and they
        // decode to `-1` like any clear bit --- the same rule at either width.
        let v = if bit < K::BITS as usize && (K::index(code) >> bit) & 1 == 1 {
            E::ONE
        } else {
            // `0 - 1`: exact for every integer element, never a wrap.
            E::ZERO.sub(E::ONE)
        };
        // `new` established that the declared alphabet admits `+-1`, so the
        // wrap cannot put a value outside its bound: it is that O(1) check
        // being cashed in, exactly as `Offset`'s decode cashes in its own.
        Alphabet::wrap(v)
    }
}

impl<E: IntegerElement, Bd: Bound, const BLK: usize, K: SymbolCode> Enumerable<E, Bd>
    for Sign<E, Bd, BLK, K>
{
    // Every bit pattern of the block, capped at what the code type can
    // address --- the same honesty about the code type's width as `Grid`'s.
    const CODE_SPACE: usize = if BLK < K::BITS as usize {
        1 << BLK
    } else {
        K::CODES
    };

    // The whole point of the tier: the code *is* the index, and the decode is
    // the bit test. The Gray-walk build reads this flag and nothing else.
    const SIGN_BIT_BOOK: bool = true;

    fn code_at(index: usize) -> Self::Code {
        K::at(index % Self::CODE_SPACE.max(1))
    }

    fn index_of(code: Self::Code) -> usize {
        // A mask, not a remainder: the space is a power of two at every block
        // width, and the stored code *is* the index, which is the whole point
        // of the tier.
        K::index(code) & (Self::CODE_SPACE - 1)
    }

    fn as_index_stream(codes: &[K]) -> Option<IndexStream<'_>> {
        // Always --- not at the power-of-two entry counts, as `Grid` and
        // `Book` answer, but at every block width, because the code addresses
        // the enumeration unconditionally: the space is a power of two at
        // every `BLK`. This is the answer `Packed` can never give, and the gap
        // this tier exists to close.
        K::index_stream(codes, Self::CODE_SPACE)
    }
}

// ---------------------------------------------------------------------------
// Ternary
// ---------------------------------------------------------------------------

/// Weights in `{-1, 0, +1}`, two bits per element, with no table at all.
///
/// The `Packed<Grid<4>,4>` spelling of the same decode (`CK-10`) tabulates but
/// can never answer [`Enumerable::as_index_stream`]: a packed byte's index is
/// a mixed-radix decomposition of it, so the tabulated traversal builds the
/// index stream it gathers from. This tier spells the same decode with the
/// code *being* the index --- `Code = u16`, one base-4 digit per element of
/// the block, and a base-4 digit stream is a power-of-two code space (4^BLK =
/// 2^(2BLK)) --- so the operand's own memory is the stream and there is
/// nothing to build. `CK-12` pins the two spellings byte for byte.
///
/// The codebook is the constant `{-1, 0, +1}`, so there is nothing to borrow
/// and no lifetime: the decode is a two-bit field test. **Digit `t` of the
/// code --- bits `2t` and `2t+1` --- is element `t`; 0 is `-1`, 1 is `0`, 2 is
/// `+1`, and 3, the dead digit no ternary encoder emits, decodes to `0`, low
/// digit first** --- the order `Packed` reads its sub-codes in (`CK-03`) and
/// the decode the spelling's `[-1, 0, +1, dead]` table gives every digit
/// (`CK-10`), so a `Ternary` code spells the block the composition spells from
/// the same byte value. The convention is normative: the encoder side is out
/// of tree, and this decode is the whole contract with it.
#[derive(Clone, Copy, Debug)]
pub struct Ternary<E: IntegerElement, Bd: Bound, const BLK: usize, K: SymbolCode = u16> {
    _marker: TierMarker<E, Bd, K>,
}

impl<E: IntegerElement, Bd: Bound, const BLK: usize, K: SymbolCode> Ternary<E, Bd, BLK, K> {
    /// The one non-existence check: an alphabet that does not admit `+-1` has
    /// no ternary codec --- `0` is in every alphabet. Decided at construction,
    /// before any arithmetic, like every other non-existence in this library
    /// (C6).
    pub const fn new() -> Option<Self> {
        if Bd::VALUE < 1 || E::FULL < 1 {
            return None;
        }
        Some(Self {
            _marker: PhantomData,
        })
    }
}

impl<E: IntegerElement, Bd: Bound, const BLK: usize, K: SymbolCode> Codec<E, Bd>
    for Ternary<E, Bd, BLK, K>
{
    type Code = K;
    const MAX_BLOCK: usize = BLK;
    const TIER: TierId = TierId::Ternary;

    fn decode_element(&self, code: Self::Code, i: usize) -> Alphabet<E, Bd> {
        let shift = 2 * (i % BLK);
        // The guard keeps the shift total past the code type's own width: at
        // `BLK > K::BITS / 2` a code holds no digit for those positions, and
        // they decode to `-1` like any zero field --- the same rule at either
        // width.
        let digit = if shift < K::BITS as usize {
            (K::index(code) >> shift) & 3
        } else {
            0
        };
        let v = match digit {
            // `0 - 1`: exact for every integer element, never a wrap.
            0 => E::ZERO.sub(E::ONE),
            2 => E::ONE,
            // 1 is the zero digit, and 3 the dead one --- which also decodes
            // to 0, a priced duplicate of it rather than an error (`CK-10`).
            _ => E::ZERO,
        };
        // `new` established that the declared alphabet admits `+-1`, so the
        // wrap cannot put a value outside its bound: it is that O(1) check
        // being cashed in, exactly as `Sign`'s decode cashes in its own.
        Alphabet::wrap(v)
    }
}

impl<E: IntegerElement, Bd: Bound, const BLK: usize, K: SymbolCode> Enumerable<E, Bd>
    for Ternary<E, Bd, BLK, K>
{
    // 4^BLK, capped at what the code type can address --- a power of two at
    // every block width, the same honesty about the code type's width as
    // `Grid`'s, and what makes the index stream unconditional below.
    const CODE_SPACE: usize = if 2 * BLK < K::BITS as usize {
        1 << (2 * BLK)
    } else {
        K::CODES
    };

    fn code_at(index: usize) -> Self::Code {
        K::at(index % Self::CODE_SPACE.max(1))
    }

    fn index_of(code: Self::Code) -> usize {
        // A mask, not a remainder: the space is a power of two at every block
        // width, and the stored code *is* the index, which is the whole point
        // of the tier.
        K::index(code) & (Self::CODE_SPACE - 1)
    }

    fn as_index_stream(codes: &[K]) -> Option<IndexStream<'_>> {
        // Always, as `Sign` answers and for one reason more: 4^BLK is 2^(2BLK),
        // so the code addresses the enumeration unconditionally. This is the
        // answer `Packed` can never give, and the gap this tier exists to close.
        K::index_stream(codes, Self::CODE_SPACE)
    }
}

// ---------------------------------------------------------------------------
// Offset
// ---------------------------------------------------------------------------

/// `d(c) - z`: asymmetric quantization as a codec composition, not a feature.
///
/// A zero point is a codec, which is why the library needs no separate
/// "asymmetric" mode and no branch for one (§1.3).
///
/// Decoding widens the alphabet: an inner value bounded by `BdIn` and a zero
/// point of magnitude `|z|` produce a value bounded by `BdIn + |z|`. That is
/// why the output bound is a separate parameter, and why [`Offset::new`]
/// checks, once and in O(1), that the declared output bound covers the image
/// and that no value overflows the element type. A `None` means no such codec
/// exists --- the same category as a non-conformant shape, decided before any
/// arithmetic (C6).
#[derive(Clone, Copy, Debug)]
pub struct Offset<E: IntegerElement, BdIn: Bound, BdOut: Bound, C: Codec<E, BdIn>> {
    inner: C,
    zero: E,
    _marker: OffsetMarker<BdIn, BdOut, E>,
}

impl<E: IntegerElement, BdIn: Bound, BdOut: Bound, C: Codec<E, BdIn>> Offset<E, BdIn, BdOut, C> {
    /// Compose `inner` with the zero point `zero`.
    ///
    /// `None` when `BdIn + |zero|` exceeds the declared output bound or the
    /// element type's own range, in which case the decode this describes does
    /// not land in the alphabet it claims to.
    pub fn new(inner: C, zero: E) -> Option<Self> {
        let image = BdIn::VALUE.checked_add(zero.magnitude())?;
        if image > BdOut::VALUE || image > E::FULL {
            return None;
        }
        Some(Self {
            inner,
            zero,
            _marker: PhantomData,
        })
    }

    /// The zero point.
    pub const fn zero(&self) -> E {
        self.zero
    }
}

impl<E, BdIn, BdOut, C> Codec<E, BdOut> for Offset<E, BdIn, BdOut, C>
where
    E: IntegerElement,
    BdIn: Bound,
    BdOut: Bound,
    C: Codec<E, BdIn>,
{
    type Code = C::Code;
    const MAX_BLOCK: usize = C::MAX_BLOCK;
    const TIER: TierId = TierId::Offset;
    const IS_FIXED_WIDTH: bool = C::IS_FIXED_WIDTH;

    fn decode_element(&self, code: Self::Code, i: usize) -> Alphabet<E, BdOut> {
        let inner = self.inner.decode_element(code, i).get();
        // Exact: `new` established that `BdIn + |zero| <= min(BdOut, E::FULL)`,
        // so this subtraction neither overflows the element type nor leaves the
        // declared output alphabet. The wrap is therefore not an unchecked
        // assertion; it is the O(1) check in `new` being cashed in.
        Alphabet::<E, BdOut>::wrap(inner.sub(self.zero))
    }
}

impl<E, BdIn, BdOut, C> Enumerable<E, BdOut> for Offset<E, BdIn, BdOut, C>
where
    E: IntegerElement,
    BdIn: Bound,
    BdOut: Bound,
    C: Enumerable<E, BdIn>,
{
    // A zero point relabels the *image*, never the code space: the same codes
    // name the same blocks, shifted. So the enumeration is the inner one,
    // unchanged, and a tabulated asymmetric quantization needs nothing new.
    const CODE_SPACE: usize = C::CODE_SPACE;

    fn code_at(index: usize) -> Self::Code {
        C::code_at(index)
    }

    fn index_of(code: Self::Code) -> usize {
        C::index_of(code)
    }

    fn as_index_stream(codes: &[Self::Code]) -> Option<IndexStream<'_>> {
        // A zero point relabels the image, never the code space, so whether the
        // code addresses the enumeration is the inner codec's answer unchanged.
        C::as_index_stream(codes)
    }
}

// ---------------------------------------------------------------------------
// Runs
// ---------------------------------------------------------------------------

/// Sparse storage as a codec, not as a separate algorithm and not as a separate
/// crate (D-16).
///
/// A run is `(length, code)`: `length` consecutive decoded elements, all of
/// them `d(code)`. A gap is a run whose code decodes to the alphabet's zero, so
/// the zeros are *explicit* and the arithmetic downstream stays dense. No
/// sparse-specific speedup is claimed; the benefit is residency, and `CG-03`
/// measures it like any other codec's.
///
/// A code is a *run index*, and the run's length is what [`Codec::decode_len`]
/// reports. `MAX_RUN` is the longest run the caller declares, so this tier is
/// variable-length within a compile-time bound --- which is exactly what the
/// `MAX_BLOCK` spelling is for (S4, S5b).
#[derive(Clone, Copy, Debug)]
pub struct Runs<'a, E: IntegerElement, Bd: Bound, C: Codec<E, Bd>, const MAX_RUN: usize> {
    runs: &'a [(u32, C::Code)],
    inner: C,
    _marker: PhantomData<fn() -> (E, Bd)>,
}

impl<'a, E: IntegerElement, Bd: Bound, C: Codec<E, Bd>, const MAX_RUN: usize>
    Runs<'a, E, Bd, C, MAX_RUN>
{
    /// Borrow a run list.
    ///
    /// `None` when a run is empty or longer than `MAX_RUN`, both of which mean
    /// the list describes no stream. Decided at construction, before any
    /// arithmetic, like every other non-existence in this library (C6).
    pub fn new(inner: C, runs: &'a [(u32, C::Code)]) -> Option<Self> {
        if runs
            .iter()
            .any(|&(len, _)| len == 0 || len as usize > MAX_RUN)
        {
            return None;
        }
        Some(Self {
            runs,
            inner,
            _marker: PhantomData,
        })
    }

    /// The number of stored runs, which is the residency this tier buys.
    pub const fn run_count(&self) -> usize {
        self.runs.len()
    }

    /// The decoded length of the whole run list.
    ///
    /// `CK-06` asserts this equals the declared row width, which is the
    /// invariant that lets a variable-length tier live inside `Codec`.
    pub fn decoded_len(&self) -> usize {
        self.runs.iter().map(|&(len, _)| len as usize).sum()
    }
}

impl<E: IntegerElement, Bd: Bound, C: Codec<E, Bd>, const MAX_RUN: usize> Codec<E, Bd>
    for Runs<'_, E, Bd, C, MAX_RUN>
{
    type Code = u32;
    const MAX_BLOCK: usize = MAX_RUN;
    const TIER: TierId = TierId::Runs;
    // The one variable-length tier: a run is as long as it is.
    const IS_FIXED_WIDTH: bool = false;

    fn decode_len(&self, run: Self::Code) -> usize {
        self.runs
            .get(run as usize)
            .map_or(0, |&(len, _)| len as usize)
    }

    fn decode_element(&self, run: Self::Code, i: usize) -> Alphabet<E, Bd> {
        match self.runs.get(run as usize) {
            // Every element of a run decodes the run's code. For an inner tier
            // that is itself a block codec, the position within the block is
            // `i % C::MAX_BLOCK`, so a run of an E8 codeword repeats the whole
            // codeword rather than its first element.
            Some(&(_, code)) => self.inner.decode_element(code, i % C::MAX_BLOCK),
            // Past the end of the run list: the alphabet's zero, explicitly.
            None => Alphabet::ZERO,
        }
    }
}

// `Runs` does not implement `Enumerable`. A run code carries a *length*, so the
// table would have to be indexed by `(value, length)` rather than by code, and
// the length is artifact-local data rather than the codec's fixed enumerable
// space. Such a pair is not the `code -> fixed block` declaration tabulation
// requires. The absence is structural at the traversal's type boundary, not a
// performance capability withheld behind a runtime refusal (§4).

// ---------------------------------------------------------------------------
// Transcode
// ---------------------------------------------------------------------------

/// A total map from one code space to another.
///
/// This is the first half of the upstream `transcode_decode` composite: the
/// part that is a relabelling rather than a decode.
pub trait CodeMap<In: Copy, Out: Copy>: Send + Sync {
    /// Relabel. Total on the whole input code space.
    fn map(&self, code: In) -> Out;
}

impl<In: Copy, Out: Copy, F> CodeMap<In, Out> for F
where
    F: Fn(In) -> Out + Send + Sync,
{
    fn map(&self, code: In) -> Out {
        self(code)
    }
}

/// The upstream `transcode_decode` composite: relabel, then decode.
///
/// Closed under composition, which is what makes a tier change a *change of
/// artifact* rather than a change of arithmetic (`CK-04`, `CL-MM03`).
///
/// The plan spells this `Transcode<C1, C2>`; the first parameter is a
/// [`CodeMap`] rather than a second [`Codec`], because a decode into the
/// alphabet cannot be the input of another decode --- only a relabelling can.
#[derive(Clone, Copy, Debug)]
pub struct Transcode<M, C, In> {
    map: M,
    inner: C,
    _marker: PhantomData<fn() -> In>,
}

impl<M, C, In> Transcode<M, C, In> {
    /// Compose a relabelling with a decode.
    pub const fn new(map: M, inner: C) -> Self {
        Self {
            map,
            inner,
            _marker: PhantomData,
        }
    }
}

impl<E, Bd, M, C, In> Codec<E, Bd> for Transcode<M, C, In>
where
    E: IntegerElement,
    Bd: Bound,
    C: Codec<E, Bd>,
    M: CodeMap<In, C::Code>,
    In: Copy + Send + Sync + 'static,
{
    type Code = In;
    const MAX_BLOCK: usize = C::MAX_BLOCK;
    const TIER: TierId = TierId::Transcode;
    const IS_FIXED_WIDTH: bool = C::IS_FIXED_WIDTH;

    fn decode_element(&self, code: Self::Code, i: usize) -> Alphabet<E, Bd> {
        self.inner.decode_element(self.map.map(code), i)
    }
}

// `Transcode` does not implement `Enumerable`, and the reason is the shape of
// this repository's spelling rather than anything about tabulation. The plan
// spells the composite `Transcode<C1, C2>` with two codecs; here the first
// parameter is a [`CodeMap`] --- a *function* --- because a decode into the
// alphabet cannot be the input of another decode, only a relabelling can. A
// function's domain cannot be enumerated and `code_at` would need its inverse.
//
// Nothing is lost. `decode(c) = inner.decode(map(c))`, so the table over the
// inner code space *is* the table for the composite: a caller tabulates `C` and
// reaches the entry with `C::index_of(map(code))`. The relabelling is a property
// of the code stream, and tabulation has already left the code stream behind by
// the time it reads the table.

// ---------------------------------------------------------------------------
// Arena
// ---------------------------------------------------------------------------

/// A code type whose own order is an enumeration's order.
///
/// `u16` and `u8`, and nothing else: the width is a fact about the artifact's
/// storage, and the two widths are the two residencies a codec has a use for
/// --- two bytes a symbol, and one where the code space fits a byte.
///
/// The trait exists so that [`Arena`], [`Book`], [`Sign`] and [`Ternary`] are
/// one tier at two widths rather than two tiers with one decode: the table
/// read, the bit fields, the enumeration, and the canonicalization are written
/// once, against these methods.
pub trait SymbolCode: Copy + Send + Sync + 'static {
    /// How many distinct codes the type holds.
    const CODES: usize;

    /// The type's own width in bits, for the shift guard a bit-field decode
    /// totals against: a position past it reads as a clear field, at either
    /// width.
    const BITS: u32;

    /// The code at `index` in the type's own order. Total below `CODES`.
    fn at(index: usize) -> Self;

    /// Where `code` sits in the type's own order.
    fn index(code: Self) -> usize;

    /// The stored code stream read as the index stream it already is, when it
    /// is one.
    ///
    /// `index_of` is `% space` at every tier here, which is `& (space - 1)`
    /// exactly when the space is a power of two --- and then the stored code,
    /// masked, *is* the index, at either width, and the traversal borrows it.
    /// At any other entry count the two differ and the traversal builds the
    /// stream, at the same bytes. The variant names the width: a byte stream
    /// answers [`IndexStream::U8`], and the gather dispatches on it once.
    fn index_stream(codes: &[Self], space: usize) -> Option<IndexStream<'_>>;
}

impl SymbolCode for u16 {
    const CODES: usize = U16_CODES;
    const BITS: u32 = u16::BITS;

    fn at(index: usize) -> Self {
        // `index < CODE_SPACE <= CODES` at every call site, so the cast is the
        // identity on the enumeration's own domain.
        index as u16
    }

    fn index(code: Self) -> usize {
        code as usize
    }

    fn index_stream(codes: &[Self], space: usize) -> Option<IndexStream<'_>> {
        space.is_power_of_two().then_some(IndexStream::U16(codes))
    }
}

impl SymbolCode for u8 {
    const CODES: usize = u8::MAX as usize + 1;
    const BITS: u32 = u8::BITS;

    fn at(index: usize) -> Self {
        // As `u16`'s: total on the enumeration's own domain.
        index as u8
    }

    fn index(code: Self) -> usize {
        code as usize
    }

    fn index_stream(codes: &[Self], space: usize) -> Option<IndexStream<'_>> {
        space.is_power_of_two().then_some(IndexStream::U8(codes))
    }
}

/// A codebook of one artifact's distinct bit patterns.
///
/// Mechanically the arena is a one-element block grid, and its decode is the
/// same table read. What makes it a tier rather than a [`Grid`] is the
/// construction discipline the token carries into the kappa manifest: the
/// table is the source stream's distinct symbols in canonical order, built by
/// [`canonicalize`], so two artifacts holding the same values share a
/// codebook --- and an address --- whatever order their streams stored them
/// in (§6.4, `CK-10`).
///
/// The element type is a float: the symbols are bit patterns, and the bound is
/// [`Whole`] because membership is discharged by the table's construction and
/// no magnitude question applies to a float (§5.2b).
///
/// The code width is a parameter, `u16` by default and `u8` where the
/// artifact's distinct patterns fit a byte (`CK-14`): one type at two
/// residencies, not two tiers. At a power-of-two code space the stored code
/// masked *is* the index at either width, so both spellings answer their own
/// stream to the tabulated gather --- `IndexStream::U16` and
/// `IndexStream::U8` --- and the traversal builds nothing.
#[derive(Clone, Copy, Debug)]
pub struct Arena<'a, E: FloatElement, const N: usize, K: SymbolCode = u16> {
    table: &'a [Alphabet<E, Whole<E>>; N],
    // The width is a parameter of the decode, not of the table: the struct
    // stores nothing of it.
    _code: PhantomData<fn() -> K>,
}

impl<'a, E: FloatElement, const N: usize, K: SymbolCode> Arena<'a, E, N, K> {
    /// Borrow a canonical codebook. The canonicalization is [`canonicalize`]'s
    /// discipline; like [`Grid`], the borrow itself validates nothing (§6.2).
    pub const fn new(table: &'a [Alphabet<E, Whole<E>>; N]) -> Self {
        Self {
            table,
            _code: PhantomData,
        }
    }

    /// The codebook.
    pub const fn table(&self) -> &'a [Alphabet<E, Whole<E>>; N] {
        self.table
    }
}

impl<E: FloatElement, const N: usize, K: SymbolCode> Codec<E, Whole<E>> for Arena<'_, E, N, K> {
    type Code = K;
    const MAX_BLOCK: usize = 1;
    const TIER: TierId = TierId::Arena;

    fn decode_element(&self, code: Self::Code, _i: usize) -> Alphabet<E, Whole<E>> {
        // The same total reduction `Grid` performs: an arbitrary code indexes
        // a table of `N` entries modulo `N` (C6).
        self.table[K::index(code) % N]
    }
}

impl<E: FloatElement, const N: usize, K: SymbolCode> Enumerable<E, Whole<E>>
    for Arena<'_, E, N, K>
{
    // The reachable code space, as for `Grid`: the table size unless the table
    // is wider than the code type can address.
    const CODE_SPACE: usize = if N < K::CODES { N } else { K::CODES };

    fn code_at(index: usize) -> Self::Code {
        K::at(index % Self::CODE_SPACE.max(1))
    }

    fn index_of(code: Self::Code) -> usize {
        // The same reduction `decode_element` performs, so equal indices and
        // equal decodes are the same relation.
        K::index(code) % Self::CODE_SPACE.max(1)
    }

    fn as_index_stream(codes: &[K]) -> Option<IndexStream<'_>> {
        K::index_stream(codes, Self::CODE_SPACE)
    }
}

/// Sort `values` into canonical arena order and move the distinct symbols to
/// the front, returning how many there are. The codebook is `&values[..n]`.
///
/// Canonical order is unsigned order on bit patterns, and equality is pattern
/// equality: `-0.0` and `+0.0` are distinct symbols, two NaNs with different
/// payloads are distinct symbols, and deciding either takes no float
/// comparison (`CK-10`, `CU-01`). The sort is a heapsort: in place,
/// allocation-free, and `O(n log n)` whatever the stream, so an adversarial
/// order meets the same bound as a friendly one.
pub fn canonicalize<E: FloatElement>(values: &mut [E]) -> usize {
    heapsort_by_pattern(values);
    // Equal patterns are adjacent after the sort; keep the first of each run.
    let mut n = 0usize;
    for i in 0..values.len() {
        if n == 0 || values[i].symbol_bits() != values[n - 1].symbol_bits() {
            values[n] = values[i];
            n += 1;
        }
    }
    n
}

/// Heapsort by bit pattern: a total order with no float comparisons.
fn heapsort_by_pattern<E: FloatElement>(values: &mut [E]) {
    fn sift_down<E: FloatElement>(values: &mut [E], mut root: usize, end: usize) {
        loop {
            let child = 2 * root + 1;
            if child >= end {
                return;
            }
            let mut swap = root;
            if values[swap].symbol_bits() < values[child].symbol_bits() {
                swap = child;
            }
            if child + 1 < end && values[swap].symbol_bits() < values[child + 1].symbol_bits() {
                swap = child + 1;
            }
            if swap == root {
                return;
            }
            values.swap(root, swap);
            root = swap;
        }
    }

    let len = values.len();
    for start in (0..len / 2).rev() {
        sift_down(values, start, len);
    }
    for end in (1..len).rev() {
        values.swap(0, end);
        sift_down(values, 0, end);
    }
}
