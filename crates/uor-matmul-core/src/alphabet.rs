//! The parametric working alphabet (§5.2).
//!
//! The upstream formalization's `InAlphabet` hypothesis is [`Alphabet`]'s
//! invariant and [`Bound::VALUE`] is its `B`. Every kernel entry point takes
//! `&[Alphabet<E, Bd>]`, so the alphabet hypothesis is discharged by the type
//! system at the boundary, once, rather than per element per kernel.

use core::marker::PhantomData;

use bytemuck::TransparentWrapper;

use crate::acc::{Accumulator, EncodeFrom};

/// A machine value usable as an alphabet element and as a coded weight.
///
/// Adding an element type is an impl, not a branch anywhere else. Nothing in
/// the library matches on which element type it was handed.
pub trait Element: Copy + PartialEq + Send + Sync + core::fmt::Debug + 'static {
    /// The accumulator for this element type.
    ///
    /// Not a parameter and not a choice: exactly one type, wide enough that the
    /// worst case the machine can express does not reach its range (§3.2).
    type Acc: Accumulator;

    /// Width of one component, in bits.
    const BITS: u32;

    /// Element-products summed by one [`Element::mac`].
    ///
    /// One for a scalar type. Two for a complex type, whose real part is
    /// `a.re * w.re - a.im * w.im`. The accumulator derivation reads this so
    /// that a complex alphabet needs no separate width table and no branch.
    const PRODUCT_TERMS: u32 = 1;

    /// The additive identity, which is what a padded position decodes to.
    ///
    /// Zero padding is exact, which is why an unaligned or prime shape takes
    /// the same path as any other and not a different one (S8).
    const ZERO: Self;

    /// The one arithmetic primitive: accumulate the exact product `a * w`.
    ///
    /// Exact, total, and free of any rounding or saturating step. Everything
    /// else in the library is a traversal over this (R3, R13).
    ///
    /// In place, by `&mut`, rather than by value. For an `i128` that is a wash;
    /// for a complete accumulator it is the difference between a float GEMM
    /// that works and one that spends most of its time copying an 88- or
    /// 544-byte register in and out of this call, twice per product.
    fn mac(acc: &mut Self::Acc, a: Self, w: Self);

    /// Does this element type have a narrow register at all?
    ///
    /// `false` means every tile takes the wide accumulator. That changes
    /// nothing about the answer --- only about how many instructions produce
    /// it --- which is the whole content of the narrow/wide distinction (R13).
    const HAS_NARROW: bool = false;

    /// Accumulate into the narrow register.
    ///
    /// Only called for a tile whose depth [`crate::bounds::fits_narrow`] has
    /// admitted against [`NARROW_CAP`], so this is exact wherever it is
    /// reached. The default is unreachable and is never called, because a type
    /// with `HAS_NARROW = false` is never offered a narrow tile.
    fn mac_narrow(acc: i64, _a: Self, _w: Self) -> i64 {
        acc
    }

    /// Fold a completed narrow partial into the wide accumulator.
    fn combine_narrow(acc: Self::Acc, narrow: i64) -> Self::Acc;

    /// Does this element type have a *32-bit* narrow register?
    ///
    /// The second entry of [`crate::NARROW_CAPS`], as a product primitive. It
    /// exists for the same reason the 64-bit one does and buys something the
    /// 64-bit one cannot: a 32-bit multiply-add is eight lanes to a 256-bit
    /// register where a 64-bit one is four and, on x86, is not a single
    /// instruction at all. Reaching it is worth an order of magnitude in any loop
    /// that is bound by products rather than by memory.
    ///
    /// `false` means every run takes [`Element::mac_narrow`] or the wide
    /// accumulator. That changes nothing about the answer (R13).
    const HAS_NARROW32: bool = false;

    /// Accumulate into the 32-bit narrow register.
    ///
    /// Only called for a run [`crate::fits_narrow`] has admitted against
    /// `i32::MAX`, so this is exact wherever it is reached. The default is
    /// unreachable and is never called, because a type with `HAS_NARROW32 = false`
    /// is never offered a 32-bit run.
    fn mac_narrow32(acc: i32, _a: Self, _w: Self) -> i32 {
        acc
    }
}

/// Elements with a magnitude, and therefore an alphabet.
///
/// [`Alphabet`] and [`Bound`] are defined only over these. A float has an
/// exponent range rather than a magnitude bound, which is why it implements
/// [`FloatElement`] instead and carries no alphabet (§5.2b).
pub trait IntegerElement: Element {
    /// The largest magnitude this type can hold.
    ///
    /// [`Full<Self>`] is the bound that admits every value of the type, so no
    /// value is ever outside its own alphabet and the default path cannot
    /// reject anything. `i8::MIN` is an element of `Alphabet<i8, Full<i8>>`,
    /// whose bound is 128; refusing a representable value would be an arbitrary
    /// limitation dressed as rigour (R8).
    const FULL: u128;

    /// The multiplicative identity.
    ///
    /// [`Element::ZERO`] exists because padding decodes to it; `ONE` is here
    /// because a constant-table tier --- `Sign`, whose decode is `+-1` and
    /// nothing else --- has to name it without a branch on the element type.
    /// One fact per impl, the same discipline as every other constant here.
    const ONE: Self;

    /// The magnitude of this value, for [`observe_bound`] and for the alphabet
    /// declaration check. Never used by an accumulation.
    fn magnitude(self) -> u128;

    /// Exact difference, for the one codec that needs one.
    ///
    /// Used only by `Offset`, whose constructor establishes in O(1) that
    /// `BdIn + |zero|` fits both the declared output bound and the element
    /// type, so the subtraction cannot leave the type's range. It is spelled
    /// `wrapping_sub` because R5 requires the overflow behaviour to be written
    /// rather than inherited from the build profile --- not because a wrap can
    /// occur.
    fn sub(self, other: Self) -> Self;

    /// Exact sum, for the sub-cubic recursion's block sums.
    ///
    /// The recursion's level plan admits a level only when the grown declared
    /// bound keeps every block sum inside the element type, so the addition
    /// cannot leave the type's range wherever it is reached. Spelled
    /// `wrapping_add` for the same reason [`IntegerElement::sub`] is spelled
    /// `wrapping_sub`: R5 asks that the overflow behaviour be written, not
    /// because a wrap can occur.
    fn add(self, other: Self) -> Self;
}

/// What an IEEE code names.
///
/// The decode is total on every bit pattern: there is no `f32` for which it
/// fails, which is why non-finite inputs are codes rather than an error
/// condition (`CT-03`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Decoded {
    /// An exact dyadic rational: `(-1)^sign * mantissa * 2^exp`.
    ///
    /// Exact, with no rounding anywhere in the decode. Zero is this with a
    /// mantissa of zero.
    Finite {
        /// True for a negative value, including negative zero.
        sign: bool,
        /// The integer significand, subnormals included.
        mantissa: u64,
        /// The binary exponent of bit 0 of `mantissa`.
        exp: i32,
    },
    /// A signed infinity.
    Infinite {
        /// True for negative infinity.
        sign: bool,
    },
    /// A NaN. The payload is not preserved: IEEE 754 does not require it to be,
    /// and preserving it would make the result depend on the accumulation order
    /// of the NaNs, which is the one thing this library refuses to let happen.
    NotANumber,
}

/// A decoded float code, flattened for the inner loop.
///
/// Named for what it is rather than `Packed`, which in this workspace is a
/// codec tier --- a different thing that would otherwise share the name.
///
/// Sixteen bytes: a significand word, its exact exponent, and the layout word
/// that makes the representation lossless on the whole public [`Decoded`]
/// domain. Ordinary IEEE finite values keep the signed-`i64` inline form. A
/// finite value that cannot use that form stores its raw `u64` magnitude bits
/// and the reserved sign tag `-1` or `+1` in the trailing word. Non-finite
/// kinds retain the zero-tag sentinel-exponent form (`CK-21`).
///
/// These finite fields seed a lazy canonical Laurent word: valuation becomes
/// its grade, sign becomes modality, and the remaining odd coefficient is
/// refined into as many Atlas pages as its precision requires. The struct stays
/// plain data (`Pod`), so a panel remains a zero-copy cache of decoded source
/// codes rather than a reified arithmetic operand (R7, `CA-05`).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub struct PackedCode {
    /// The signed inline significand, raw escaped `u64` bits, or non-finite sign.
    pub mantissa: i64,
    /// The exact binary exponent of bit 0 of the finite magnitude, or a sentinel.
    pub exp: i32,
    /// Zero for inline and non-finite codes; exactly `-1` or `+1` for an
    /// escaped finite code, with the tag's sign carrying the value's sign.
    ///
    /// This field remains public because changing its visibility would break
    /// existing struct literals and plain-data tooling. Its original published
    /// invariant required callers to write zero. Consequently, reading the two
    /// formerly-unused reserved values extends the representation without
    /// changing the meaning of any value that satisfied that invariant; every
    /// other noncanonical padding value retains the old exponent-based kind.
    #[doc(hidden)]
    pub _pad: i32,
}

/// The exponent that marks an infinity when the finite escape tag is zero.
/// Below every real IEEE exponent, and distinct from [`PackedCode::NAN_EXP`].
const INF_EXP: i32 = i32::MIN + 1;
/// The exponent that marks a NaN when the finite escape tag is zero.
const NAN_EXP: i32 = i32::MIN;

impl PackedCode {
    /// The sentinel exponent of an infinity.
    pub const INF_EXP: i32 = INF_EXP;
    /// The sentinel exponent of a NaN.
    pub const NAN_EXP: i32 = NAN_EXP;

    /// Whether the trailing layout word carries the canonical finite escape.
    #[inline(always)]
    const fn has_finite_escape(&self) -> bool {
        self._pad == -1 || self._pad == 1
    }

    /// Pack a decoded code.
    pub const fn of(d: Decoded) -> Self {
        match d {
            Decoded::Finite {
                sign,
                mantissa,
                exp,
            } => {
                // The inline coefficient is symmetric around zero so its sign
                // and magnitude recover uniquely. Negative zero, the high half
                // of `u64`, and a finite sentinel exponent use the layout word
                // as a sign tag and leave every source bit untouched.
                let inline =
                    exp > INF_EXP && mantissa <= i64::MAX as u64 && !(sign && mantissa == 0);
                if inline {
                    Self {
                        mantissa: if sign {
                            -(mantissa as i64)
                        } else {
                            mantissa as i64
                        },
                        exp,
                        _pad: 0,
                    }
                } else {
                    Self {
                        mantissa: mantissa as i64,
                        exp,
                        _pad: if sign { -1 } else { 1 },
                    }
                }
            }
            // The sign of an infinity rides in the mantissa, as `-1` or `1`,
            // so the product's sign is the product of the two mantissas and
            // needs no separate rule.
            Decoded::Infinite { sign } => Self {
                mantissa: if sign { -1 } else { 1 },
                exp: INF_EXP,
                _pad: 0,
            },
            Decoded::NotANumber => Self {
                mantissa: 0,
                exp: NAN_EXP,
                _pad: 0,
            },
        }
    }

    /// Is this an ordinary finite value?
    ///
    /// The overwhelmingly common IEEE case remains the first comparison,
    /// because both sentinels are below every real IEEE exponent. The second
    /// term admits the lossless finite escape at those exponents (`CK-21`).
    pub const fn is_finite(&self) -> bool {
        self.exp > INF_EXP || self.has_finite_escape()
    }

    /// Is this a NaN?
    pub const fn is_nan(&self) -> bool {
        !self.has_finite_escape() && self.exp == NAN_EXP
    }

    /// Is this an infinity?
    pub const fn is_infinite(&self) -> bool {
        !self.has_finite_escape() && self.exp == INF_EXP
    }
}

/// Elements that are finite-width dyadic codes.
///
/// A float is a code, and this trait is its codec: the decode from a bit
/// pattern to the exact dyadic rational it names. It is total, it never rounds,
/// and it is the same shape as any other codec in `uor-matmul-codec` (§3.3).
///
/// # Validity law
///
/// Every Laurent grade occupied by the exact product of any two finite values
/// returned by [`FloatElement::decode`] must lie in the inclusive interval
/// [`FloatElement::MIN_PRODUCT_EXP`] ..= [`FloatElement::MAX_PRODUCT_EXP`], and
/// `Self::Acc` must span that interval plus its reduction guard. This is the
/// float analogue of [`Element`]'s requirement that its one accumulator be
/// wide enough for every product the element can name. The range constants use
/// `i32`, so implementing this trait asserts that the format's complete product
/// support is representable by the public placement channel; a codec that can
/// emit a product outside its own declaration is not a valid implementation.
pub trait FloatElement: Element {
    /// Bits in a significand, implicit bit included: 24 for `f32`, 53 for `f64`.
    ///
    /// This declares the finite support available to the canonical Laurent
    /// refinement. It is a property of the format, never a data-dependent
    /// route selector.
    const SIGNIFICAND_BITS: u32;

    /// Inclusive lower bound on every Laurent grade a finite product can
    /// occupy. For an IEEE interchange format this is twice its minimum
    /// subnormal exponent.
    const MIN_PRODUCT_EXP: i32;
    /// Inclusive upper bound on every Laurent grade a finite product can
    /// occupy. For an IEEE interchange format, twice its exclusive maximum
    /// finite exponent is a convenient exact bound.
    const MAX_PRODUCT_EXP: i32;
    /// The codec. Total on all bit patterns, non-finite ones included.
    fn decode(self) -> Decoded;

    /// The bit pattern, widened to `u64`.
    ///
    /// The symbol's *identity*: two codes are the same symbol exactly when
    /// their patterns are equal, so `-0.0` and `+0.0` are distinct symbols with
    /// equal dyadic evaluations, and two NaNs with different payloads are
    /// distinct symbols. Canonical order on symbols is unsigned order on patterns
    /// (`CK-10`). No float arithmetic is involved in producing it.
    fn symbol_bits(self) -> u64;

    /// The codec's output in packed form, for a driver that decodes once and
    /// multiplies many times.
    fn pack(self) -> PackedCode {
        PackedCode::of(self.decode())
    }
}

/// The largest magnitude the portable reference's narrow register holds.
///
/// This is a property of the register, not of the library: on a backend the
/// same predicate is evaluated against that backend's lane width instead
/// (§7.2). Nothing anywhere reads it as a limit.
pub const NARROW_CAP: u128 = i64::MAX as u128;

/// The alphabet bound, carried as a sealed trait with an associated const
/// rather than a `const B: u128` parameter (D-7).
///
/// This encoding compiles on stable, which the const-generic form does not,
/// because [`Full`] needs an associated const in what would otherwise be a
/// const-generic argument.
pub trait Bound: Copy + Send + Sync + core::fmt::Debug + 'static {
    /// `B`: every element of the alphabet has magnitude at most this.
    const VALUE: u128;
}

/// The bound that admits every value of `E`. Zero-sized.
///
/// This is the default, and it is why the library has no alphabet-violation
/// error: under `Full<E>` there is nothing to violate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Full<E: IntegerElement>(PhantomData<E>);

impl<E: IntegerElement> Bound for Full<E> {
    const VALUE: u128 = E::FULL;
}

/// The bound a float element carries: none, because a float has an exponent
/// range rather than a magnitude (§5.2b). Zero-sized.
///
/// Used by the arena tier, where the codebook *is* the alphabet: membership is
/// discharged by the table's construction, so there is nothing for a bound
/// check to ask. `VALUE` is `u128::MAX` so that any reader which asks a
/// magnitude question anyway gets the conservative answer --- `fits_narrow`'s
/// `b * b` overflows to `None`, which selects the wide register, the only
/// register a float accumulation has.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Whole<E: FloatElement>(PhantomData<E>);

impl<E: FloatElement> Bound for Whole<E> {
    const VALUE: u128 = u128::MAX;
}

/// A declared narrower bound, e.g. `Bnd<127>` for W8A8.
///
/// A narrower bound lets more tiles take the narrow-register path (§5.1), and
/// nothing else changes. It is a declaration about the data, not a restriction
/// on it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Bnd<const V: u128>;

impl<const V: u128> Bound for Bnd<V> {
    const VALUE: u128 = V;
}

/// An element of the symmetric alphabet `{-B..B}`.
///
/// The bound is `E: Element`, not `E: IntegerElement`: a float carries no
/// magnitude, so it cannot be *admitted* by a bound check --- but an arena's
/// codebook is an alphabet of symbols, and the wrapper is what the codec's
/// decode returns for any element type. The checking constructor
/// [`Alphabet::new`] stays integer-only, because `magnitude` does.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default, TransparentWrapper)]
#[repr(transparent)]
#[transparent(E)]
pub struct Alphabet<E: Element, Bd: Bound>(E, PhantomData<Bd>);

impl<E: Element, Bd: Bound> Alphabet<E, Bd> {
    /// `B`, the bound this alphabet declares.
    pub const BOUND: u128 = Bd::VALUE;

    /// The alphabet's zero, which is what a padded position decodes to.
    pub const ZERO: Self = Self(E::ZERO, PhantomData);

    /// The underlying value.
    pub const fn get(self) -> E {
        self.0
    }
}

impl<E: IntegerElement, Bd: Bound> Alphabet<E, Bd> {
    /// Admit a value whose magnitude is within the declared bound.
    ///
    /// Returns the observed magnitude rather than a unit error, so a caller who
    /// guessed wrong learns what the data actually is.
    pub fn new(value: E) -> Result<Self, ObservedBound> {
        let observed = value.magnitude();
        if observed <= Bd::VALUE {
            Ok(Self(value, PhantomData))
        } else {
            Err(ObservedBound {
                declared: Bd::VALUE,
                observed,
            })
        }
    }
}

impl<E: IntegerElement> Alphabet<E, Full<E>> {
    /// Admit any value of `E`. Infallible, because `Full<E>` admits everything.
    pub const fn of(value: E) -> Self {
        Self(value, PhantomData)
    }
}

/// What the data actually is, returned when a bound declaration is wrong.
///
/// This is not the operation failing. It is a *declaration* being checked at
/// the boundary, before any product exists. Being wrong about your own data is
/// not a reason for a matmul to fail (C6), so the recovery is to use
/// [`as_alphabet_full`] --- which cannot fail --- or to widen the declaration
/// to the observed value.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ObservedBound {
    /// The bound the caller declared.
    pub declared: u128,
    /// The tightest bound the data actually satisfies at or beyond `declared`.
    pub observed: u128,
}

impl core::fmt::Display for ObservedBound {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "declared alphabet bound {} but the data reaches {}; \
             use as_alphabet_full, which admits every value of the element type",
            self.declared, self.observed
        )
    }
}

/// The total entry point, and the one every convenience API uses.
///
/// Infallible for every slice of every element type, including `i8::MIN`.
/// Zero-copy, allocation-free, and with no validating pass: `Full<E>` admits
/// every value of `E`, so there is nothing to check.
pub fn as_alphabet_full<E: IntegerElement>(xs: &[E]) -> &[Alphabet<E, Full<E>>] {
    Alphabet::wrap_slice(xs)
}

/// The mutable twin of [`as_alphabet_full`].
pub fn as_alphabet_full_mut<E: IntegerElement>(xs: &mut [E]) -> &mut [Alphabet<E, Full<E>>] {
    Alphabet::wrap_slice_mut(xs)
}

impl<E: FloatElement> Alphabet<E, Whole<E>> {
    /// Admit any code of `E`. Infallible: `Whole<E>` declares no magnitude, and
    /// the arena's codebook is what makes the symbol a member.
    ///
    /// Named `symbol` rather than `of`: a float in an alphabet is a code's
    /// name, and the distinct name keeps `Alphabet::of` unambiguous for the
    /// integer `Full<E>` impl.
    pub const fn symbol(value: E) -> Self {
        Self(value, PhantomData)
    }
}

/// The float twin of [`as_alphabet_full`]: every bit pattern is a symbol, so
/// the wrap is infallible, zero-copy, and with no validating pass.
pub fn as_alphabet_whole<E: FloatElement>(xs: &[E]) -> &[Alphabet<E, Whole<E>>] {
    Alphabet::wrap_slice(xs)
}

/// The declared-bound entry point, for a caller who knows their data is
/// narrower and wants more tiles on the narrow-register path.
///
/// Returns the observed bound rather than an error when the declaration is
/// wrong, because being wrong about your own data is not a reason for a matmul
/// to fail (C6).
pub fn as_alphabet<E: IntegerElement, Bd: Bound>(
    xs: &[E],
) -> Result<&[Alphabet<E, Bd>], ObservedBound> {
    let mut worst = 0u128;
    for x in xs {
        let m = x.magnitude();
        if m > worst {
            worst = m;
        }
    }
    if worst <= Bd::VALUE {
        Ok(Alphabet::wrap_slice(xs))
    } else {
        Err(ObservedBound {
            declared: Bd::VALUE,
            observed: worst,
        })
    }
}

/// One pass, returning the tightest bound the data actually satisfies.
///
/// Callers who want the narrow-register path without guessing use this.
pub fn observe_bound<E: IntegerElement>(xs: &[E]) -> u128 {
    let mut worst = 0u128;
    for x in xs {
        let m = x.magnitude();
        if m > worst {
            worst = m;
        }
    }
    worst
}

/// Implement [`Element`] for a signed machine integer.
///
/// Every impl is the same three facts --- the width, the largest magnitude, and
/// the exact product --- which is why adding an element type is an impl and not
/// a branch anywhere else.
macro_rules! impl_element_for_signed {
    ($($t:ty),* $(,)?) => { $(
        impl Element for $t {
            type Acc = i128;
            const BITS: u32 = <$t>::BITS;
            const ZERO: Self = 0;

            fn mac(acc: &mut i128, a: Self, w: Self) {
                // Exact by construction: the product of two values of this type
                // needs at most `2 * BITS` bits, and `acc_bits::<Self>()` leaves
                // room for every `k` the machine can address (§3.2). No
                // wrapping, no saturation, no rounding (R3).
                *acc += (a as i128) * (w as i128);
            }

            // The product of two values of this type needs at most 64 bits, so
            // an i64 register can hold a run of them --- for exactly as long as
            // `fits_narrow` says, and not one step longer.
            const HAS_NARROW: bool = 2 * <$t>::BITS <= 63;

            fn mac_narrow(acc: i64, a: Self, w: Self) -> i64 {
                acc + (a as i64) * (w as i64)
            }

            fn combine_narrow(acc: i128, narrow: i64) -> i128 {
                acc + narrow as i128
            }

            // The product of two values of this type needs at most 32 bits, so an
            // i32 register can hold a run of them --- again for exactly as long
            // as `fits_narrow` says.
            const HAS_NARROW32: bool = 2 * <$t>::BITS <= 31;

            fn mac_narrow32(acc: i32, a: Self, w: Self) -> i32 {
                acc + (a as i32) * (w as i32)
            }
        }

        impl IntegerElement for $t {
            const FULL: u128 = 1u128 << (<$t>::BITS - 1);
            const ONE: Self = 1;

            fn magnitude(self) -> u128 {
                self.unsigned_abs() as u128
            }

            fn sub(self, other: Self) -> Self {
                self.wrapping_sub(other)
            }

            fn add(self, other: Self) -> Self {
                self.wrapping_add(other)
            }
        }
    )* };
}

impl_element_for_signed!(i8, i16, i32);

impl Element for i64 {
    type Acc = crate::acc::Limbs<3>;
    const BITS: u32 = i64::BITS;
    const ZERO: Self = 0;

    fn mac(acc: &mut Self::Acc, a: Self, w: Self) {
        // The product of two i64 needs at most 128 bits and is therefore exact
        // in an i128; the 192-bit accumulator absorbs every `k` on top of it.
        acc.add_i128_in_place((a as i128) * (w as i128));
    }

    // A single i64 x i64 product does not fit in an i64, so there is no narrow
    // register for this element type and every tile takes the wide one. The
    // answer is the same either way, which is the point.
    const HAS_NARROW: bool = false;

    fn combine_narrow(acc: Self::Acc, narrow: i64) -> Self::Acc {
        acc.add_i128(narrow as i128)
    }
}

impl IntegerElement for i64 {
    const FULL: u128 = 1u128 << 63;
    const ONE: Self = 1;

    fn magnitude(self) -> u128 {
        self.unsigned_abs() as u128
    }

    fn sub(self, other: Self) -> Self {
        self.wrapping_sub(other)
    }

    fn add(self, other: Self) -> Self {
        self.wrapping_add(other)
    }
}

/// A complex element. Its real and imaginary parts are drawn from the same
/// alphabet, and its bound bounds both.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
#[repr(C)]
pub struct Complex<T> {
    /// The real part.
    pub re: T,
    /// The imaginary part.
    pub im: T,
}

impl<T> Complex<T> {
    /// Construct from parts.
    pub const fn new(re: T, im: T) -> Self {
        Self { re, im }
    }
}

/// A complex accumulator: two independent exact accumulators, one per part.
///
/// Not a special case. `combine` and `mac` are the scalar operations applied
/// componentwise, so the same identity holds and the same tests apply.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ComplexAcc<A> {
    /// The real part's accumulator.
    pub re: A,
    /// The imaginary part's accumulator.
    pub im: A,
}

impl<A: Accumulator> Accumulator for ComplexAcc<A> {
    const ZERO: Self = Self {
        re: A::ZERO,
        im: A::ZERO,
    };
    const BITS: u32 = A::BITS;

    fn combine(self, other: Self) -> Self {
        Self {
            re: self.re.combine(other.re),
            im: self.im.combine(other.im),
        }
    }
}

/// Implement [`Element`] for `Complex<T>` over a signed machine integer.
macro_rules! impl_element_for_complex {
    ($($t:ty => $acc:ty),* $(,)?) => { $(
        impl Element for Complex<$t> {
            type Acc = ComplexAcc<$acc>;
            const BITS: u32 = <$t>::BITS;
            // (a.re * w.re - a.im * w.im) sums two element-products, so the
            // accumulator derivation gets one more bit than the scalar case.
            const PRODUCT_TERMS: u32 = 2;
            const ZERO: Self = Complex { re: 0, im: 0 };

            fn mac(acc: &mut Self::Acc, a: Self, w: Self) {
                <$t as Element>::mac(&mut acc.re, a.re, w.re);
                <$t as Element>::mac(&mut acc.re, a.im, -w.im);
                <$t as Element>::mac(&mut acc.im, a.re, w.im);
                <$t as Element>::mac(&mut acc.im, a.im, w.re);
            }

            // Two independent accumulators; a single i64 register would have to
            // carry both, so there is no narrow path here either.
            const HAS_NARROW: bool = false;

            fn combine_narrow(acc: Self::Acc, narrow: i64) -> Self::Acc {
                ComplexAcc {
                    re: <$t as Element>::combine_narrow(acc.re, narrow),
                    im: acc.im,
                }
            }

        }

        impl IntegerElement for Complex<$t> {
            const FULL: u128 = 1u128 << (<$t>::BITS - 1);
            const ONE: Self = Complex { re: 1, im: 0 };

            fn magnitude(self) -> u128 {
                // The bound bounds both parts, so the magnitude of the pair is
                // the larger of the two. This is not an approximation of the
                // complex modulus and is not used as one: it is the quantity
                // the accumulator width is derived against (§5.2b).
                let re = (self.re.unsigned_abs()) as u128;
                let im = (self.im.unsigned_abs()) as u128;
                if re > im { re } else { im }
            }

            fn sub(self, other: Self) -> Self {
                Complex {
                    re: <$t as IntegerElement>::sub(self.re, other.re),
                    im: <$t as IntegerElement>::sub(self.im, other.im),
                }
            }

            fn add(self, other: Self) -> Self {
                Complex {
                    re: <$t as IntegerElement>::add(self.re, other.re),
                    im: <$t as IntegerElement>::add(self.im, other.im),
                }
            }
        }
    )* };
}

impl_element_for_complex!(i32 => i128, i64 => crate::acc::Limbs<3>);

impl<A, O> EncodeFrom<ComplexAcc<A>> for Complex<O>
where
    A: Accumulator,
    O: EncodeFrom<A>,
{
    fn encode_from(acc: ComplexAcc<A>, mode: crate::policy::EncodeMode) -> Self {
        Complex {
            re: O::encode_from(acc.re, mode),
            im: O::encode_from(acc.im, mode),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// §5.2: `i8::MIN` is not rejected. It is an element of its own alphabet,
    /// whose bound is 128, and refusing a representable value would be an
    /// arbitrary limitation dressed as rigour (R8, C6).
    #[test]
    fn i8_min_is_in_its_own_alphabet_ct_01() {
        let data = [i8::MIN, -1, 0, 1, i8::MAX];
        let a = as_alphabet_full(&data);
        assert_eq!(a.len(), 5);
        assert_eq!(a[0].get(), i8::MIN);
        assert_eq!(Alphabet::<i8, Full<i8>>::BOUND, 128);
    }

    /// §5.2: a wrong declaration reports what the data is, not merely that the
    /// caller was wrong, and the recovery never involves failing the operation.
    #[test]
    fn a_wrong_declaration_reports_the_observed_bound_cs_03() {
        let data = [1i8, -100, 7];
        let err = as_alphabet::<i8, Bnd<63>>(&data).unwrap_err();
        assert_eq!(
            err,
            ObservedBound {
                declared: 63,
                observed: 100
            }
        );
        assert_eq!(observe_bound(&data), 100);
        // The recovery: the total entry point, which cannot fail.
        assert_eq!(as_alphabet_full(&data).len(), 3);
    }

    /// The wrap is zero-copy: the alphabet is a transparent wrapper, so no
    /// buffer is produced and nothing is allocated (R7, CA-01).
    #[test]
    fn wrapping_is_zero_copy_ca_01() {
        let data = [1i32, 2, 3, 4];
        let a = as_alphabet_full(&data);
        assert_eq!(a.as_ptr() as usize, data.as_ptr() as usize);
        assert_eq!(core::mem::size_of::<Alphabet<i32, Full<i32>>>(), 4);
    }

    /// The complex `mac` is the scalar one applied componentwise; it is not a
    /// second method (R13).
    #[test]
    fn complex_mac_is_componentwise_ck_01() {
        let a = Complex::new(3i32, 4);
        let w = Complex::new(5i32, -2);
        let mut acc = ComplexAcc::ZERO;
        <Complex<i32> as Element>::mac(&mut acc, a, w);
        // (3 + 4i)(5 - 2i) = 15 - 6i + 20i + 8 = 23 + 14i
        assert_eq!(acc.re, 23);
        assert_eq!(acc.im, 14);
    }
}

// ---------------------------------------------------------------------------
// The float family: a float is a code (§3.3, §5.2b)
// ---------------------------------------------------------------------------
//
// R2 forbids float *arithmetic*, not the float *types*. `f32` and `f64` appear
// here as element types being decoded, and in the encode step, and nowhere
// else: `to_bits()` is the symbol boundary and the decode below takes exact
// radix quotients and remainders of that code. The accumulation it feeds is
// the same integer accumulation every other element family uses. No float is
// ever added to a float (`CU-01`).

/// Implement the float family for one IEEE interchange format.
macro_rules! impl_float_element {
    ($t:ty, $bits:ty, $limbs:expr, $min_prod:expr, $max_prod:expr) => {
        impl Element for $t {
            type Acc = crate::acc::Complete<{ $limbs }, { $min_prod }>;
            // The format's whole width: stored mantissa, exponent, and sign.
            const BITS: u32 = <$bits>::BITS;
            const ZERO: Self = 0.0;

            fn mac(acc: &mut Self::Acc, a: Self, w: Self) {
                // Decode the symbols, normalize their finite values into the
                // canonical Atlas section, contract signed coordinates by
                // lookup and addition, and encode only after the reduction.
                // Every step is exact and none is float arithmetic.
                match (a.decode(), w.decode()) {
                    (Decoded::NotANumber, _) | (_, Decoded::NotANumber) => acc.set_nan(),
                    (Decoded::Infinite { sign: sa }, other)
                    | (other, Decoded::Infinite { sign: sa }) => match other {
                        // Infinity times zero is NaN, by IEEE 754 clause 7.2.
                        Decoded::Finite { mantissa: 0, .. } => acc.set_nan(),
                        Decoded::Finite { sign: sb, .. } => acc.set_infinity(sa != sb),
                        Decoded::Infinite { sign: sb } => acc.set_infinity(sa != sb),
                        Decoded::NotANumber => acc.set_nan(),
                    },
                    (
                        Decoded::Finite {
                            sign: sa,
                            mantissa: ma,
                            exp: ea,
                        },
                        Decoded::Finite {
                            sign: sb,
                            mantissa: mb,
                            exp: eb,
                        },
                    ) => {
                        crate::float_atlas::finite_product(acc, (sa, ma, ea), (sb, mb, eb));
                    }
                }
            }

            fn combine_narrow(acc: Self::Acc, _narrow: i64) -> Self::Acc {
                // A complete accumulator has no narrow register: the whole
                // point of it is that every product lands at full width.
                acc
            }
        }

        impl FloatElement for $t {
            const MIN_PRODUCT_EXP: i32 = $min_prod;
            const MAX_PRODUCT_EXP: i32 = $max_prod;
            // The stored fraction plus the implicit leading bit.
            const SIGNIFICAND_BITS: u32 = <$t>::MANTISSA_DIGITS;

            fn symbol_bits(self) -> u64 {
                self.to_bits() as u64
            }

            fn decode(self) -> Decoded {
                // Total on all bit patterns. No branch here can fail, and none
                // rounds: an IEEE value *names* a dyadic rational exactly, and
                // this reads the name (`CT-03`).
                const MANT_BITS: u32 = <$t>::MANTISSA_DIGITS - 1;
                // Whatever the format has left: its width, less the stored
                // mantissa and the sign. Asked rather than told, so `f16` or
                // `bf16` would need no numeral here either.
                const EXP_BITS: u32 = <$bits>::BITS - MANT_BITS - 1;
                const MANT_RADIX: $bits = (2 as $bits).pow(MANT_BITS);
                const EXP_RADIX: $bits = (2 as $bits).pow(EXP_BITS);
                const LAST_EXPONENT: $bits = EXP_RADIX - 1;
                const BIAS: i32 = (EXP_RADIX / 2 - 1) as i32;

                let bits = self.to_bits();
                let fraction = bits % MANT_RADIX;
                let upper = bits / MANT_RADIX;
                let biased = upper % EXP_RADIX;
                let sign = upper / EXP_RADIX != 0;

                if biased == LAST_EXPONENT {
                    return if fraction == 0 {
                        Decoded::Infinite { sign }
                    } else {
                        Decoded::NotANumber
                    };
                }
                if biased == 0 {
                    // Subnormal, or zero. The exponent is that of the smallest
                    // normal, and there is no implicit leading one.
                    return Decoded::Finite {
                        sign,
                        mantissa: fraction as u64,
                        exp: 1 - BIAS - MANT_BITS as i32,
                    };
                }
                Decoded::Finite {
                    sign,
                    mantissa: (fraction + MANT_RADIX) as u64,
                    exp: biased as i32 - BIAS - MANT_BITS as i32,
                }
            }
        }
    };
}

// The three widths come from the model, not from a literal restated beside it
// (R1). `model/widths.toml` owns them, `generated::complete_width` is the single
// place they are written, and `CM-01` pins all three for both formats --- it
// pinned only the minimum product exponent, so the limb count and the maximum
// could have drifted from the model without any gate noticing.
//
// `23`/`8` and `52`/`11` are gone for a related reason: they are properties of
// the IEEE format, so the format can be asked instead of told.
impl_float_element!(
    f32,
    u32,
    crate::generated::complete_width::F32_LIMBS,
    crate::generated::complete_width::F32_MIN_PRODUCT_EXP,
    crate::generated::complete_width::F32_MAX_PRODUCT_EXP
);
impl_float_element!(
    f64,
    u64,
    crate::generated::complete_width::F64_LIMBS,
    crate::generated::complete_width::F64_MIN_PRODUCT_EXP,
    crate::generated::complete_width::F64_MAX_PRODUCT_EXP
);

#[cfg(test)]
mod float_tests {
    use super::*;

    /// A deliberately conventional, test-only field oracle. The production
    /// codec is a radix section; retaining a distinct factorization here makes
    /// an extraction defect observable instead of comparing the codec with
    /// itself (R6).
    fn reference_ieee_decode(bits: u64, total_bits: u32, mant_bits: u32) -> Decoded {
        let exp_bits = total_bits - mant_bits - 1;
        let fraction_mask = (1u64 << mant_bits) - 1;
        let exponent_mask = (1u64 << exp_bits) - 1;
        let sign = bits >> (mant_bits + exp_bits) == 1;
        let biased = (bits >> mant_bits) & exponent_mask;
        let fraction = bits & fraction_mask;
        let bias = ((1u64 << (exp_bits - 1)) - 1) as i32;

        if biased == exponent_mask {
            return if fraction == 0 {
                Decoded::Infinite { sign }
            } else {
                Decoded::NotANumber
            };
        }
        if biased == 0 {
            return Decoded::Finite {
                sign,
                mantissa: fraction,
                exp: 1 - bias - mant_bits as i32,
            };
        }
        Decoded::Finite {
            sign,
            mantissa: fraction | (1u64 << mant_bits),
            exp: biased as i32 - bias - mant_bits as i32,
        }
    }

    fn assert_f32_decode(bits: u32) {
        assert_eq!(
            f32::from_bits(bits).decode(),
            reference_ieee_decode(
                bits as u64,
                <f32 as Element>::BITS,
                f32::MANTISSA_DIGITS - 1,
            ),
            "f32 symbol {bits:#010x}"
        );
    }

    fn assert_f64_decode(bits: u64) {
        assert_eq!(
            f64::from_bits(bits).decode(),
            reference_ieee_decode(bits, <f64 as Element>::BITS, f64::MANTISSA_DIGITS - 1),
            "f64 symbol {bits:#018x}"
        );
    }

    /// `CU-11`: the production IEEE section is pinned independently at every
    /// exponent code for both formats, on both signs and on every fraction
    /// transition that changes its interpretation. The f32 half-word sweeps
    /// additionally exhaust every field prefix and suffix.
    #[test]
    fn ieee_decode_matches_the_independent_field_oracle_cu_11() {
        let f32_mant_bits = f32::MANTISSA_DIGITS - 1;
        let f32_exp_bits = <f32 as Element>::BITS - f32_mant_bits - 1;
        let f32_fraction_radix = 1u32 << f32_mant_bits;
        let f32_fraction_edges = [
            0,
            1,
            f32_fraction_radix / 2 - 1,
            f32_fraction_radix / 2,
            f32_fraction_radix - 2,
            f32_fraction_radix - 1,
        ];
        for sign in 0u32..=1 {
            for biased in 0u32..(1u32 << f32_exp_bits) {
                for fraction in f32_fraction_edges {
                    let bits = (sign << (f32_exp_bits + f32_mant_bits))
                        | (biased << f32_mant_bits)
                        | fraction;
                    assert_f32_decode(bits);
                }
            }
        }
        for half in 0u32..=u16::MAX as u32 {
            assert_f32_decode(half << 16);
            for prefix in [0, 0x007f_0000, 0x3f80_0000, 0x7f7f_0000, 0x7f80_0000] {
                assert_f32_decode(prefix | half);
                assert_f32_decode((prefix | half) | 0x8000_0000);
            }
        }

        let f64_mant_bits = f64::MANTISSA_DIGITS - 1;
        let f64_exp_bits = <f64 as Element>::BITS - f64_mant_bits - 1;
        let f64_fraction_radix = 1u64 << f64_mant_bits;
        let f64_fraction_edges = [
            0,
            1,
            f64_fraction_radix / 2 - 1,
            f64_fraction_radix / 2,
            f64_fraction_radix - 2,
            f64_fraction_radix - 1,
        ];
        for sign in 0u64..=1 {
            for biased in 0u64..(1u64 << f64_exp_bits) {
                for fraction in f64_fraction_edges {
                    let bits = (sign << (f64_exp_bits + f64_mant_bits))
                        | (biased << f64_mant_bits)
                        | fraction;
                    assert_f64_decode(bits);
                }
            }
        }
    }

    /// The decode is exact: reading the name of a dyadic rational and
    /// reconstructing it must round-trip, for every kind of finite value.
    #[test]
    fn the_float_decode_is_exact_ct_03() {
        for x in [
            1.0f32,
            -1.0,
            0.5,
            3.0,
            1e30,
            -1e-30,
            f32::MIN_POSITIVE,
            0.0,
            -0.0,
        ] {
            let Decoded::Finite {
                sign,
                mantissa,
                exp,
            } = x.decode()
            else {
                panic!("{x} is finite");
            };
            // Reconstruct exactly, and the two words matter. `powi` is an
            // *approximation* of a power --- Miri models that by perturbing it,
            // which is how this read `1.0000000000000004` against `1.0` the first
            // time the job ever ran. The scale here is a power of two, so it is
            // built from its own bits, and an `f32` mantissa times it is exact in
            // an `f64`: at most 24 significant bits against 53, and `exp` for
            // every finite `f32` lies in `[-149, 104]`, well inside `f64`'s
            // normal range.
            let reconstructed = if mantissa == 0 {
                0.0
            } else {
                (mantissa as f64) * f64::from_bits(((exp + 1023) as u64) << 52)
            };
            let expected = x as f64;
            // Two assertions, not one. `if sign { -expected } else { expected }
            // .abs()` takes the magnitude of the whole expression, so the decoded
            // sign never entered it and no negative decode was ever checked.
            assert_eq!(sign, x.is_sign_negative(), "sign of {x}");
            assert_eq!(reconstructed, expected.abs(), "magnitude of {x}");
        }
        // The smallest subnormal is the unit in the last place of the format.
        let tiny = f32::from_bits(1);
        assert_eq!(
            tiny.decode(),
            Decoded::Finite {
                sign: false,
                mantissa: 1,
                exp: -149
            }
        );
    }

    /// CT-03: non-finite codes are codes. The decode is total on all of them
    /// and never errors.
    #[test]
    fn non_finite_codes_decode_and_never_error_ct_03() {
        assert_eq!(f32::INFINITY.decode(), Decoded::Infinite { sign: false });
        assert_eq!(f32::NEG_INFINITY.decode(), Decoded::Infinite { sign: true });
        assert_eq!(f32::NAN.decode(), Decoded::NotANumber);
        assert_eq!(f64::NAN.decode(), Decoded::NotANumber);
        // A signalling NaN pattern is a NaN like any other; nothing traps.
        assert_eq!(f32::from_bits(0x7F80_0001).decode(), Decoded::NotANumber);
        // Every one of the 2^16 patterns in the exponent-boundary region
        // decodes without panicking, which is what "total" means here.
        for hi in 0u32..=0xFFFF {
            let _ = f32::from_bits(hi << 16).decode();
        }
    }

    fn interpret_packed(code: PackedCode) -> Decoded {
        if code._pad == -1 || code._pad == 1 {
            return Decoded::Finite {
                sign: code._pad < 0,
                mantissa: code.mantissa as u64,
                exp: code.exp,
            };
        }
        if code.exp == PackedCode::NAN_EXP {
            return Decoded::NotANumber;
        }
        if code.exp == PackedCode::INF_EXP {
            return Decoded::Infinite {
                sign: code.mantissa < 0,
            };
        }
        Decoded::Finite {
            sign: code.mantissa < 0,
            mantissa: code.mantissa.unsigned_abs(),
            exp: code.exp,
        }
    }

    /// `CK-21`: the fourth layout word is a finite escape, so the public
    /// `Decoded` domain has a left inverse without widening the packed cache.
    #[test]
    fn every_decoded_code_has_a_lossless_packed_interpretation_ck_21() {
        let mantissas = [0, 1, i64::MAX as u64, (i64::MAX as u64) + 1, u64::MAX];
        let exponents = [
            PackedCode::NAN_EXP,
            PackedCode::INF_EXP,
            PackedCode::INF_EXP + 1,
            -1074,
            -149,
            0,
            i32::MAX,
        ];
        for sign in [false, true] {
            for mantissa in mantissas {
                for exp in exponents {
                    let decoded = Decoded::Finite {
                        sign,
                        mantissa,
                        exp,
                    };
                    let packed = PackedCode::of(decoded);
                    assert_eq!(interpret_packed(packed), decoded, "{decoded:?}");
                }
            }
        }

        for decoded in [
            Decoded::Infinite { sign: false },
            Decoded::Infinite { sign: true },
            Decoded::NotANumber,
        ] {
            assert_eq!(interpret_packed(PackedCode::of(decoded)), decoded);
        }
    }

    /// `CK-21`: kind predicates are a total partition even on noncanonical
    /// `Pod` words, the two reserved tags are the only escapes, and ordinary
    /// IEEE values retain the signed inline form.
    #[test]
    fn packed_kind_layout_and_inline_ieee_codes_are_stable_ck_21() {
        for pad in [-1, 1] {
            for exp in [PackedCode::NAN_EXP, PackedCode::INF_EXP, 0, i32::MAX] {
                let code = PackedCode {
                    mantissa: -1,
                    exp,
                    _pad: pad,
                };
                assert!(code.is_finite(), "reserved tag {pad} at exponent {exp}");
                assert!(!code.is_nan());
                assert!(!code.is_infinite());
            }
        }

        // Padding values outside the published zero invariant were ignored by
        // the original representation. Keep their exponent-based meaning so
        // the extension consumes only the two tags `PackedCode::of` writes.
        for pad in [i32::MIN, -2, 2, i32::MAX] {
            let nan = PackedCode {
                mantissa: -1,
                exp: PackedCode::NAN_EXP,
                _pad: pad,
            };
            let infinity = PackedCode {
                mantissa: -1,
                exp: PackedCode::INF_EXP,
                _pad: pad,
            };
            assert!(nan.is_nan(), "noncanonical padding {pad} stays ignored");
            assert!(
                infinity.is_infinite(),
                "noncanonical padding {pad} stays ignored"
            );
        }

        for (decoded, negative) in [
            (0.0f64.decode(), false),
            (1.5f64.decode(), false),
            ((-1.5f64).decode(), true),
            (f64::MAX.decode(), false),
            (f64::MIN.decode(), true),
        ] {
            let Decoded::Finite {
                sign,
                mantissa,
                exp,
            } = decoded
            else {
                panic!("the inline corpus is finite")
            };
            let packed = PackedCode::of(decoded);
            assert_eq!(packed._pad, 0);
            assert_eq!(packed.exp, exp);
            assert_eq!(packed.mantissa < 0, negative && mantissa != 0);
            assert_eq!(packed.mantissa.unsigned_abs(), mantissa);
            assert_eq!(sign, negative);
        }

        let negative_zero = PackedCode::of((-0.0f64).decode());
        assert_ne!(negative_zero, PackedCode::of(0.0f64.decode()));
        assert!(negative_zero.is_finite());
        assert_eq!(interpret_packed(negative_zero), (-0.0f64).decode());

        let nan = PackedCode::of(Decoded::NotANumber);
        let positive_infinity = PackedCode::of(Decoded::Infinite { sign: false });
        let negative_infinity = PackedCode::of(Decoded::Infinite { sign: true });
        assert!(nan.is_nan());
        assert!(positive_infinity.is_infinite());
        assert!(negative_infinity.is_infinite());
        assert_eq!(nan._pad, 0);
        assert_eq!(positive_infinity._pad, 0);
        assert_eq!(negative_infinity._pad, 0);

        assert_eq!(core::mem::size_of::<PackedCode>(), 16);
        assert_eq!(core::mem::align_of::<PackedCode>(), 8);
        assert_eq!(core::mem::offset_of!(PackedCode, mantissa), 0);
        assert_eq!(core::mem::offset_of!(PackedCode, exp), 8);
        assert_eq!(core::mem::offset_of!(PackedCode, _pad), 12);
        assert_eq!(
            interpret_packed(PackedCode::default()),
            Decoded::Finite {
                sign: false,
                mantissa: 0,
                exp: 0,
            }
        );
    }

    /// §3.3: the product exponent range in `model/widths.toml` is the range
    /// this decode actually produces.
    #[test]
    fn the_product_exponent_range_matches_the_model_cm_01() {
        let smallest = f32::from_bits(1).decode();
        let Decoded::Finite { exp, .. } = smallest else {
            panic!()
        };
        assert_eq!(exp * 2, <f32 as FloatElement>::MIN_PRODUCT_EXP);
        assert_eq!(
            <f32 as FloatElement>::MIN_PRODUCT_EXP,
            crate::generated::complete_width::F32_MIN_PRODUCT_EXP
        );
        assert_eq!(
            <f64 as FloatElement>::MIN_PRODUCT_EXP,
            crate::generated::complete_width::F64_MIN_PRODUCT_EXP
        );
        // The other two widths, which this test did not pin. The limb count and
        // the maximum product exponent were literals written beside the model's
        // own values, so either could have drifted from `model/widths.toml`
        // without a gate noticing (R1). They are now read from the model, and
        // read back here.
        assert_eq!(
            <f32 as FloatElement>::MAX_PRODUCT_EXP,
            crate::generated::complete_width::F32_MAX_PRODUCT_EXP
        );
        assert_eq!(
            <f64 as FloatElement>::MAX_PRODUCT_EXP,
            crate::generated::complete_width::F64_MAX_PRODUCT_EXP
        );
        // The limb count, against the derivation rather than against its own
        // numeral: the accumulator spans the product exponent range, plus
        // `MAX_K_BITS` of guard so that summing the deepest reduction the machine
        // can address cannot leave it, plus a sign. `size_of` will not serve here
        // --- `Complete` carries three sticky non-finite flags beyond its limbs ---
        // and the const generic is not readable from outside, so the derivation is
        // the handle. This is the same arithmetic `model/widths.toml` shows in its
        // `span_bits`/`guard_bits`/`sign_bits` columns, recomputed here through the
        // library's own `limbs_for`.
        for (limbs, min, max) in [
            (
                crate::generated::complete_width::F32_LIMBS,
                <f32 as FloatElement>::MIN_PRODUCT_EXP,
                <f32 as FloatElement>::MAX_PRODUCT_EXP,
            ),
            (
                crate::generated::complete_width::F64_LIMBS,
                <f64 as FloatElement>::MIN_PRODUCT_EXP,
                <f64 as FloatElement>::MAX_PRODUCT_EXP,
            ),
        ] {
            assert!(max > min, "the product exponent range is non-empty");
            let span = max.abs_diff(min);
            let total = span + crate::generated::MAX_K_BITS + 1;
            assert_eq!(
                limbs,
                crate::bounds::limbs_for(total),
                "the model's limb count must be the width the span, the guard and \
                 the sign demand"
            );
        }
    }
}
