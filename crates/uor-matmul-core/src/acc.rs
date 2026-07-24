//! One accumulator per element type, wide enough by construction (§5.3).
//!
//! There is no wrapping model here. A width derived from the element type
//! cannot wrap for any expressible input, so the arithmetic is plain and exact,
//! and `CT-02` checks it by running the whole corpus in a build where every
//! accumulator operation is checked and any overflow panics.
//!
//! Narrow-register tiles still exist and are still the fast path. They are
//! selected by [`crate::bounds::fits_narrow`] per tile, they compute the same
//! integer, and their result is combined into the wide accumulator. That is a
//! factorization, not a fallback: nothing about the answer or the guarantee
//! depends on which side of the predicate a tile lands (R13).

// R5 asks that *wrapping* arithmetic be written explicitly, so that a debug and
// a release binary are the same function. It does not ask that every operation
// be spelled `wrapping_*`: in this module the arithmetic is deliberately plain,
// because a width derived from the element type cannot wrap for any expressible
// input (§3.2, §5.3), and writing `wrapping_add` here would assert the opposite.
//
// The witness that the derivation holds is not a lint but `CT-02`: the whole
// corpus runs under the `checked` profile, where every one of these operations
// is checked and any overflow panics. A lint could only be satisfied by hiding
// the question; the checked build answers it.

use crate::alphabet::{Alphabet, Bound, Element, IntegerElement};
use crate::policy::EncodeMode;

/// The accumulator for element type `E`.
///
/// Not a parameter, not a policy, not a ladder: `AccOf<E>` is the unique type
/// with at least `acc_bits::<E>()` bits, and the worst case the machine can
/// express does not reach its range.
pub type AccOf<E> = <E as Element>::Acc;

/// An exact accumulator.
///
/// Every implementor is wide enough that no input the machine can represent can
/// overflow it, which is why no method here returns a `Result` and why there is
/// no saturating or rounding step anywhere in the trait except [`encode`],
/// which runs exactly once per output element.
///
/// [`encode`]: Accumulator::encode
pub trait Accumulator: Copy + Eq + core::fmt::Debug + Send + Sync + 'static {
    /// The additive identity.
    const ZERO: Self;

    /// The accumulator's width in bits.
    const BITS: u32;

    /// Combine two partial accumulations.
    ///
    /// Associative and commutative on every value that can arise, which is what
    /// makes the result independent of the reduction schedule, the tile
    /// partition, and the number of threads (`CD-02`).
    fn combine(self, other: Self) -> Self;

    /// Accumulate the exact product of two alphabet elements.
    ///
    /// Forwards to [`Element::mac`], which is the library's one arithmetic
    /// primitive.
    fn mac<E: IntegerElement<Acc = Self>, Bd: Bound>(
        self,
        a: Alphabet<E, Bd>,
        w: Alphabet<E, Bd>,
    ) -> Self {
        E::mac(self, a.get(), w.get())
    }

    /// The single encode step.
    ///
    /// Saturation or rounding, if any, happens here and exactly once, under a
    /// mode the caller names. This is the only place in the library where
    /// information can be discarded (§5.5).
    fn encode<O: Element + EncodeFrom<Self>>(self, mode: EncodeMode) -> O {
        O::encode_from(self, mode)
    }
}

/// How to produce an output element from an accumulator.
///
/// Kept separate from [`Element`] so that the encode step is a relation between
/// an accumulator and an output type rather than a property of either. That is
/// what lets `i8 x i8 -> i32`, `Complex<i32> -> Complex<i32>`, and
/// `Complete<10> -> f32` all be the same single encode step at different
/// instantiations, with no branch and no second method.
pub trait EncodeFrom<A>: Sized {
    /// Encode `acc` under `mode`.
    fn encode_from(acc: A, mode: EncodeMode) -> Self;
}

impl Accumulator for i128 {
    const ZERO: Self = 0;
    const BITS: u32 = 128;

    fn combine(self, other: Self) -> Self {
        // Exact: both operands are partial sums of the same accumulation, whose
        // total is bounded by `acc_bits::<E>() <= 128` (§3.2).
        self + other
    }
}

/// Implement [`EncodeFrom`] for a signed machine integer, out of an `i128`.
macro_rules! impl_encode_from_i128 {
    ($($t:ty),* $(,)?) => { $(
        impl EncodeFrom<i128> for $t {
            fn encode_from(acc: i128, mode: EncodeMode) -> Self {
                encode_i128_into(acc, <$t>::MIN as i128, <$t>::MAX as i128, mode) as $t
            }
        }
    )* };
}

impl_encode_from_i128!(i8, i16, i32, i64, i128);

/// The single encode step for an integer accumulator, in one place.
///
/// `Nearest` and `TowardZero` name a rounding rule, and an integer accumulator
/// holds an integer, so for this family they have nothing to round and behave
/// as `Saturating` on range. `Wrapping` names a range rule and truncates. The
/// caller names which they want; neither is a fallback, and both are exact
/// functions of the exact accumulator (§5.5).
const fn encode_i128_into(acc: i128, min: i128, max: i128, mode: EncodeMode) -> i128 {
    match mode {
        EncodeMode::Wrapping => {
            if min == i128::MIN && max == i128::MAX {
                acc
            } else {
                // Two's complement truncation to the output width, written
                // explicitly rather than left to a profile-dependent cast (R5).
                let span = (max as u128).wrapping_sub(min as u128).wrapping_add(1);
                let offset = (acc as u128).wrapping_sub(min as u128);
                (offset.wrapping_rem(span)).wrapping_add(min as u128) as i128
            }
        }
        EncodeMode::Saturating | EncodeMode::Nearest | EncodeMode::TowardZero => {
            if acc < min {
                min
            } else if acc > max {
                max
            } else {
                acc
            }
        }
    }
}

/// Fixed-width multi-limb accumulator: `L` limbs of 64 bits, two's complement,
/// little-endian, no allocation, no growth.
///
/// `L` is resolved at compile time from the element type, never at runtime.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(transparent)]
pub struct Limbs<const L: usize>([u64; L]);

impl<const L: usize> Limbs<L> {
    /// Zero.
    pub const ZERO: Self = Self([0; L]);

    /// The raw limbs, least significant first.
    pub const fn limbs(&self) -> &[u64; L] {
        &self.0
    }

    /// Is this value negative? Equivalently, is the top bit of the top limb set?
    pub const fn is_negative(&self) -> bool {
        L > 0 && (self.0[L - 1] >> 63) == 1
    }

    /// Add a sign-extended `i128`.
    ///
    /// Exact for every value that can arise: `L` is chosen so that the sum of
    /// every product the machine can address stays inside `64 * L` bits (§3.2).
    pub const fn add_i128(self, v: i128) -> Self {
        let mut out = self.0;
        let ext: u64 = if v < 0 { u64::MAX } else { 0 };
        let uv = v as u128;
        let mut carry = 0u64;
        let mut i = 0;
        while i < L {
            let addend = match i {
                0 => uv as u64,
                1 => (uv >> 64) as u64,
                _ => ext,
            };
            let (s, c1) = out[i].overflowing_add(addend);
            let (s, c2) = s.overflowing_add(carry);
            out[i] = s;
            carry = (c1 as u64).wrapping_add(c2 as u64);
            i = i.wrapping_add(1);
        }
        Self(out)
    }

    /// Two's complement negation.
    pub const fn neg(self) -> Self {
        let mut out = [0u64; L];
        let mut carry = 1u64;
        let mut i = 0;
        while i < L {
            let (s, c) = (!self.0[i]).overflowing_add(carry);
            out[i] = s;
            carry = c as u64;
            i = i.wrapping_add(1);
        }
        Self(out)
    }

    /// The value's magnitude, truncated to 128 bits, together with whether the
    /// truncation lost anything.
    pub const fn magnitude_low_u128(&self) -> (u128, bool) {
        let m = if self.is_negative() {
            self.neg()
        } else {
            *self
        };
        let lo = if L > 0 { m.0[0] as u128 } else { 0 };
        let hi = if L > 1 { (m.0[1] as u128) << 64 } else { 0 };
        let mut exceeded = false;
        let mut i = 2;
        while i < L {
            if m.0[i] != 0 {
                exceeded = true;
            }
            i = i.wrapping_add(1);
        }
        (lo | hi, exceeded)
    }

    /// The low 128 bits, as a two's complement `i128`.
    pub const fn low_i128(&self) -> i128 {
        let lo = if L > 0 { self.0[0] as u128 } else { 0 };
        let hi = if L > 1 { (self.0[1] as u128) << 64 } else { 0 };
        (lo | hi) as i128
    }
}

impl<const L: usize> Accumulator for Limbs<L> {
    const ZERO: Self = Self([0; L]);
    const BITS: u32 = (L as u32).wrapping_mul(64);

    fn combine(self, other: Self) -> Self {
        let mut out = [0u64; L];
        let mut carry = 0u64;
        for (i, slot) in out.iter_mut().enumerate() {
            let (s, c1) = self.0[i].overflowing_add(other.0[i]);
            let (s, c2) = s.overflowing_add(carry);
            *slot = s;
            carry = (c1 as u64).wrapping_add(c2 as u64);
        }
        Self(out)
    }
}

/// Implement [`EncodeFrom`] for a signed machine integer, out of `Limbs<L>`.
macro_rules! impl_encode_from_limbs {
    ($($t:ty),* $(,)?) => { $(
        impl<const L: usize> EncodeFrom<Limbs<L>> for $t {
            fn encode_from(acc: Limbs<L>, mode: EncodeMode) -> Self {
                match mode {
                    // Truncation to the output width; the low limbs already are
                    // the two's complement low bits.
                    EncodeMode::Wrapping => {
                        encode_i128_into(acc.low_i128(), i128::MIN, i128::MAX,
                                         EncodeMode::Wrapping) as $t
                    }
                    _ => {
                        let (mag, exceeded) = acc.magnitude_low_u128();
                        let negative = acc.is_negative();
                        if exceeded {
                            return if negative { <$t>::MIN } else { <$t>::MAX };
                        }
                        // `mag` is exact here, so the saturation decision is a
                        // comparison and not an estimate.
                        let limit = if negative {
                            (<$t>::MIN as i128).unsigned_abs()
                        } else {
                            <$t>::MAX as u128
                        };
                        if mag > limit {
                            return if negative { <$t>::MIN } else { <$t>::MAX };
                        }
                        if negative {
                            (mag as i128).wrapping_neg() as $t
                        } else {
                            mag as $t
                        }
                    }
                }
            }
        }
    )* };
}

impl_encode_from_limbs!(i8, i16, i32, i64, i128);

/// A complete accumulator: a fixed-point register spanning the entire product
/// exponent range of a float codec, so that every add is exact and no ordering
/// can perturb it (§3.3).
///
/// This is the Kulisch construction, and it is the same object as the integer
/// accumulator above, sized differently. It contains no float arithmetic and no
/// float token: the decode from an IEEE bit pattern to the exact dyadic
/// rational it names lives in `uor-matmul-float`, and what arrives here is an
/// integer magnitude, a binary exponent, and a sign.
///
/// `L` is the limb count and `MIN_EXP` is the binary exponent of bit 0, both
/// derived from the element type in `model/widths.toml`. The plan writes
/// `Complete<const L: usize>`; the exponent origin is carried as a second const
/// parameter rather than inferred, so that a `Complete` value cannot be
/// combined with one of a different origin.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Complete<const L: usize, const MIN_EXP: i32> {
    limbs: Limbs<L>,
    /// A NaN has been accumulated. Sticky, and it wins over everything.
    ///
    /// The non-finite state is carried as a flag rather than as a value
    /// because a value would have to participate in the fixed-point addition,
    /// and then the result would depend on *where* in the accumulation the
    /// infinity arrived. IEEE 754 clause 6 is about the value, not the
    /// schedule, so the flags are what make `CU-04` --- order-independence ---
    /// hold for non-finite inputs as well as finite ones.
    nan: bool,
    /// A positive infinity has been accumulated. Sticky.
    pos_inf: bool,
    /// A negative infinity has been accumulated. Sticky.
    neg_inf: bool,
}

impl<const L: usize, const MIN_EXP: i32> Complete<L, MIN_EXP> {
    /// Zero.
    pub const ZERO: Self = Self {
        limbs: Limbs::ZERO,
        nan: false,
        pos_inf: false,
        neg_inf: false,
    };

    /// The binary exponent of bit 0 of the register.
    pub const MIN_EXP: i32 = MIN_EXP;

    /// The register's width in bits.
    pub const WIDTH: u32 = (L as u32).wrapping_mul(64);

    /// Accumulate `sign * mag * 2^exp`, exactly.
    ///
    /// `mag` is the integer significand product of two decoded floats and `exp`
    /// is its binary exponent. Every bit of the result lands in the register,
    /// because the register spans the whole product exponent range; nothing is
    /// rounded, and nothing is dropped.
    ///
    /// A product whose exponent falls outside the register is impossible for
    /// any pair of finite inputs of the element type the register was sized
    /// for. Should one be presented anyway --- by a caller constructing a
    /// `Complete` directly --- the contribution is clamped into the register
    /// rather than wrapping, because a wrap would silently change the answer
    /// while a clamp cannot arise on any real input.
    pub fn add_scaled(self, mag: u128, exp: i32, negative: bool) -> Self {
        if mag == 0 {
            return self;
        }
        let shift = exp.saturating_sub(MIN_EXP); // R3-ok: an exponent placement, checked below
        if shift < 0 {
            return self;
        }
        let shift = shift as u32;
        let limb_index = (shift / 64) as usize;
        let bit_offset = shift % 64;

        // Spread `mag` across at most three limbs: 128 bits of magnitude can
        // straddle two limb boundaries once shifted.
        let spread: [u64; 3] = if bit_offset == 0 {
            [mag as u64, (mag >> 64) as u64, 0]
        } else {
            let lo = (mag << bit_offset) as u64;
            let mid = ((mag << bit_offset) >> 64) as u64;
            let hi = (mag >> (128 - bit_offset)) as u64;
            [lo, mid, hi]
        };

        let mut acc = self.limbs;
        for (j, part) in spread.iter().copied().enumerate() {
            if part == 0 {
                continue;
            }
            let Some(i) = limb_index.checked_add(j) else {
                continue;
            };
            if i >= L {
                continue;
            }
            acc = if negative {
                acc.combine(Self::limb_at(i, part).neg())
            } else {
                acc.combine(Self::limb_at(i, part))
            };
        }
        Self { limbs: acc, ..self }
    }

    fn limb_at(i: usize, part: u64) -> Limbs<L> {
        let mut limbs = [0u64; L];
        limbs[i] = part;
        Limbs(limbs)
    }

    /// Record that a NaN reached this accumulation. Sticky and absorbing.
    pub const fn with_nan(self) -> Self {
        Self { nan: true, ..self }
    }

    /// Record that an infinity of the given sign reached this accumulation.
    ///
    /// Two infinities of opposite sign make a NaN, by IEEE 754 clause 7.2, and
    /// they do so here whatever order they arrived in.
    pub const fn with_infinity(self, negative: bool) -> Self {
        if negative {
            Self {
                neg_inf: true,
                ..self
            }
        } else {
            Self {
                pos_inf: true,
                ..self
            }
        }
    }

    /// Has a NaN been accumulated, or has the sum become one?
    pub const fn is_nan(self) -> bool {
        self.nan || (self.pos_inf && self.neg_inf)
    }

    /// The sign of the accumulated infinity, if the sum is infinite.
    ///
    /// `None` when the sum is finite or a NaN. A NaN is checked first, so
    /// `inf + (-inf)` reports a NaN rather than an arbitrary sign.
    pub const fn infinity_sign(self) -> Option<bool> {
        if self.is_nan() {
            None
        } else if self.pos_inf {
            Some(false)
        } else if self.neg_inf {
            Some(true)
        } else {
            None
        }
    }

    /// The underlying limbs, for an encoder to round from.
    pub const fn raw(&self) -> &Limbs<L> {
        &self.limbs
    }

    /// Is the accumulated value negative?
    pub const fn is_negative(&self) -> bool {
        self.limbs.is_negative()
    }

    /// Is the accumulated value exactly zero?
    ///
    /// Exact cancellation is exact here, which is the property a classical
    /// accumulator loses first.
    pub fn is_zero(&self) -> bool {
        !self.nan && !self.pos_inf && !self.neg_inf && self.limbs.limbs().iter().all(|&l| l == 0)
    }

    /// The index of the highest set bit of the magnitude, or `None` for zero.
    ///
    /// This is what an encoder needs to find the leading one, from which the
    /// output exponent and the round and sticky bits follow.
    pub fn magnitude_high_bit(&self) -> Option<u32> {
        let m = if self.is_negative() {
            self.limbs.neg()
        } else {
            self.limbs
        };
        for i in (0..L).rev() {
            let limb = m.limbs()[i];
            if limb != 0 {
                let within = 63u32.wrapping_sub(limb.leading_zeros());
                return Some((i as u32).wrapping_mul(64).wrapping_add(within));
            }
        }
        None
    }

    /// Bit `i` of the magnitude, counting from bit 0 of the register.
    pub fn magnitude_bit(&self, i: u32) -> bool {
        let m = if self.is_negative() {
            self.limbs.neg()
        } else {
            self.limbs
        };
        let limb = (i / 64) as usize;
        if limb >= L {
            return false;
        }
        (m.limbs()[limb] >> (i % 64)) & 1 == 1
    }

    /// Is any bit of the magnitude below `i` set? The sticky bit.
    pub fn magnitude_any_below(&self, i: u32) -> bool {
        let m = if self.is_negative() {
            self.limbs.neg()
        } else {
            self.limbs
        };
        let full_limbs = (i / 64) as usize;
        for j in 0..full_limbs.min(L) {
            if m.limbs()[j] != 0 {
                return true;
            }
        }
        if full_limbs < L {
            let rem = i % 64;
            if rem > 0 {
                let mask = (1u64 << rem).wrapping_sub(1);
                if m.limbs()[full_limbs] & mask != 0 {
                    return true;
                }
            }
        }
        false
    }
}

impl<const L: usize, const MIN_EXP: i32> Accumulator for Complete<L, MIN_EXP> {
    const ZERO: Self = Self::ZERO;
    const BITS: u32 = (L as u32).wrapping_mul(64);

    fn combine(self, other: Self) -> Self {
        // The limbs add exactly and the non-finite flags union. Both are
        // associative and commutative, so combining two partial accumulations
        // gives the same answer in either order --- which is what makes the
        // float result independent of the tile partition (`CU-04`).
        Self {
            limbs: self.limbs.combine(other.limbs),
            nan: self.nan || other.nan,
            pos_inf: self.pos_inf || other.pos_inf,
            neg_inf: self.neg_inf || other.neg_inf,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The i128 accumulator is exact over the whole i8 x i8 domain at a depth
    /// far past the narrow-register threshold, which is the point: the
    /// threshold governs a register, not an answer (§3.2, CD-03).
    #[test]
    fn i128_accumulation_is_exact_past_the_narrow_threshold_cd_03() {
        let mut acc = 0i128;
        for _ in 0..1_000_000 {
            acc = <i8 as Element>::mac(acc, i8::MIN, i8::MIN);
        }
        assert_eq!(acc, 1_000_000i128 * 128 * 128);
    }

    /// `Limbs` addition and negation round-trip, including across a limb
    /// boundary, which is where a fixed-width accumulator would fail if it
    /// were merely wide rather than correct.
    #[test]
    fn limbs_add_and_negate_across_a_boundary_ct_02() {
        let a = Limbs::<3>::ZERO.add_i128(i128::MAX);
        let b = a.add_i128(i128::MAX);
        let (mag, exceeded) = b.magnitude_low_u128();
        assert!(!exceeded);
        assert_eq!(mag, (i128::MAX as u128) * 2);
        assert!(!b.is_negative());
        assert!(b.neg().is_negative());
        assert_eq!(b.neg().neg(), b);
    }

    /// i64 x i64 at a depth that overflows any 128-bit accumulator, which is
    /// exactly why `i64`'s accumulator is 192 bits and not a policy (§3.2).
    #[test]
    fn i64_accumulation_needs_and_gets_192_bits_ct_02() {
        let mut acc = <i64 as Element>::Acc::ZERO;
        for _ in 0..4 {
            acc = <i64 as Element>::mac(acc, i64::MIN, i64::MIN);
        }
        // 4 * 2^126 = 2^128, one bit past what an i128 could hold.
        let (_, exceeded) = acc.magnitude_low_u128();
        assert!(
            exceeded,
            "the value is past 128 bits, and the accumulator still holds it"
        );
        assert!(!acc.is_negative());
    }

    /// The single encode step: `Saturating` clamps, `Wrapping` truncates, and
    /// both are exact functions of the exact accumulator (§5.5).
    #[test]
    fn encode_is_the_only_lossy_step_cs_05() {
        let acc: i128 = 300;
        assert_eq!(
            <i8 as EncodeFrom<i128>>::encode_from(acc, EncodeMode::Saturating),
            127
        );
        assert_eq!(
            <i8 as EncodeFrom<i128>>::encode_from(acc, EncodeMode::Wrapping),
            44
        );
        assert_eq!(
            <i32 as EncodeFrom<i128>>::encode_from(acc, EncodeMode::Saturating),
            300
        );

        let neg: i128 = -300;
        assert_eq!(
            <i8 as EncodeFrom<i128>>::encode_from(neg, EncodeMode::Saturating),
            -128
        );
        assert_eq!(
            <i8 as EncodeFrom<i128>>::encode_from(neg, EncodeMode::Wrapping),
            -44
        );
    }

    /// `Complete` accumulates exactly across the whole exponent span: a huge
    /// term and a tiny term both land, and adding them in either order gives
    /// the same register. No classical float accumulator does this.
    #[test]
    fn complete_accumulation_is_order_independent_cd_02() {
        type C = Complete<10, -298>;
        let big = C::ZERO.add_scaled(1, 200, false);
        let both_a = big.add_scaled(1, -290, false);
        let both_b = C::ZERO.add_scaled(1, -290, false).add_scaled(1, 200, false);
        assert_eq!(both_a, both_b);
        assert!(both_a.magnitude_bit((200i32 - -298i32) as u32));
        assert!(both_a.magnitude_bit((-290i32 - -298i32) as u32));
        assert_eq!(both_a.magnitude_high_bit(), Some(498));
        assert!(both_a.magnitude_any_below(498));
    }

    /// A term and its negation cancel exactly, which is the property a
    /// classical accumulator loses first.
    #[test]
    fn complete_cancellation_is_exact_cd_02() {
        type C = Complete<10, -298>;
        let acc = C::ZERO
            .add_scaled(12345, 100, false)
            .add_scaled(12345, 100, true);
        assert!(acc.is_zero());
        assert_eq!(acc.magnitude_high_bit(), None);
    }
}
