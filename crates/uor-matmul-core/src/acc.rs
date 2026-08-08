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
use crate::generated::complete_state::{
    BASE as COMPLETE_NONFINITE_STATE_BASE, COUNT as COMPLETE_NONFINITE_STATE_COUNT,
    NAN as COMPLETE_NAN_STATE, NAN_MASK as COMPLETE_NAN_MASK, NEG_INF as COMPLETE_NEG_INF_STATE,
    NEG_INF_MASK as COMPLETE_NEG_INF_MASK, POS_INF as COMPLETE_POS_INF_STATE,
    POS_INF_MASK as COMPLETE_POS_INF_MASK,
};
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
        &mut self,
        a: Alphabet<E, Bd>,
        w: Alphabet<E, Bd>,
    ) {
        E::mac(self, a.get(), w.get());
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
pub(crate) const fn encode_i128_into(acc: i128, min: i128, max: i128, mode: EncodeMode) -> i128 {
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

/// Add `parts` into `limbs` starting at limb `at`, propagating the carry only
/// as far as it reaches.
///
/// The terminal Atlas placement touches only the limbs reached by one resolved
/// Laurent coordinate. The coordinate contraction has already happened; this
/// routine merely embeds its exact coefficient in the complete register.
fn add_at<const L: usize>(limbs: &mut [u64; L], high: &mut i64, at: usize, parts: [u64; 3]) {
    let mut carry = 0u64;
    let mut i = at;
    let mut part_at = 0usize;
    while part_at < parts.len() || carry != 0 {
        let part = parts.get(part_at).copied().unwrap_or(0);
        if i < L {
            let (sum, c1) = limbs[i].overflowing_add(part);
            let (sum, c2) = sum.overflowing_add(carry);
            limbs[i] = sum;
            carry = u64::from(c1) + u64::from(c2);
        } else if i == L {
            *high = (*high as u64).wrapping_add(part).wrapping_add(carry) as i64;
            // A carry beyond the extension word is outside every model-sized
            // float accumulation and terminal expression. `Complete` remains
            // total for a hand-built, undersized register by discarding that
            // out-of-representation part, as the former low-only carrier did.
            return;
        } else {
            return;
        }
        i += 1;
        part_at += 1;
    }
}

/// Subtract `parts` from `limbs` starting at limb `at`, propagating the borrow
/// only as far as it reaches.
fn sub_at<const L: usize>(limbs: &mut [u64; L], high: &mut i64, at: usize, parts: [u64; 3]) {
    let mut borrow = 0u64;
    let mut i = at;
    let mut part_at = 0usize;
    while part_at < parts.len() || borrow != 0 {
        let part = parts.get(part_at).copied().unwrap_or(0);
        if i < L {
            let (diff, b1) = limbs[i].overflowing_sub(part);
            let (diff, b2) = diff.overflowing_sub(borrow);
            limbs[i] = diff;
            borrow = u64::from(b1) + u64::from(b2);
        } else if i == L {
            *high = (*high as u64).wrapping_sub(part).wrapping_sub(borrow) as i64;
            return;
        } else {
            return;
        }
        i += 1;
        part_at += 1;
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

    /// Bit `i`, counting from bit 0 of the register.
    pub fn bit(&self, i: u32) -> bool {
        let limb = (i / 64) as usize;
        limb < L && (self.0[limb] >> (i % 64)) & 1 == 1
    }

    /// The `n` bits starting at bit `at`, with `n <= 64`.
    ///
    /// One shift-pair rather than `n` bit tests. That matters more than it
    /// looks: reading a significand a bit at a time is `n` passes over a
    /// register that is 67 limbs wide for `f64`, and the encode step is once per
    /// output element.
    pub fn window(&self, at: u32, n: u32) -> u64 {
        debug_assert!(n <= 64);
        let limb = (at / 64) as usize;
        let sh = at % 64;
        let lo = if limb < L { self.0[limb] >> sh } else { 0 };
        // `sh == 0` would shift a `u64` by 64, which is not a shift.
        let hi = if sh != 0 && limb + 1 < L {
            self.0[limb + 1] << (64 - sh)
        } else {
            0
        };
        let w = lo | hi;
        if n < 64 {
            w & ((1u64 << n) - 1)
        } else {
            w
        }
    }

    /// The index of the highest set bit, or `None` when every limb is zero.
    pub fn high_bit(&self) -> Option<u32> {
        for i in (0..L).rev() {
            let limb = self.0[i];
            if limb != 0 {
                let within = 63u32.wrapping_sub(limb.leading_zeros());
                return Some((i as u32).wrapping_mul(64).wrapping_add(within));
            }
        }
        None
    }

    /// Is any bit strictly below `i` set? The sticky bit.
    pub fn any_below(&self, i: u32) -> bool {
        let full = (i / 64) as usize;
        for j in 0..full.min(L) {
            if self.0[j] != 0 {
                return true;
            }
        }
        if full < L {
            let rem = i % 64;
            if rem > 0 && self.0[full] & ((1u64 << rem).wrapping_sub(1)) != 0 {
                return true;
            }
        }
        false
    }

    /// Add a sign-extended `i128` in place, propagating the carry only as far
    /// as it reaches.
    ///
    /// The hot path for `i64` elements: a full-width traversal per product
    /// would make a 192-bit accumulator cost three times an `i128` one for no
    /// reason, since the carry almost never leaves the low two limbs.
    pub fn add_i128_in_place(&mut self, v: i128) {
        let ext: u64 = if v < 0 { u64::MAX } else { 0 };
        let uv = v as u128;
        let mut carry = 0u64;
        for i in 0..L {
            let addend = match i {
                0 => uv as u64,
                1 => (uv >> 64) as u64,
                _ => ext,
            };
            let (sum, c1) = self.0[i].overflowing_add(addend);
            let (sum, c2) = sum.overflowing_add(carry);
            self.0[i] = sum;
            carry = u64::from(c1) + u64::from(c2);
            // A non-negative addend past limb 1 contributes nothing but the
            // carry, and once that is gone the remaining limbs are unchanged.
            if ext == 0 && carry == 0 && i >= 1 {
                break;
            }
        }
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

// The former three boolean flags occupied one alignment-rounded word after the
// low limbs. Seven extreme values of that same word encode every nonempty
// union of those flags; every other value is the signed extension limb. The
// model proves the terminal expression stays many bits away from these
// extrema, so no finite value the float APIs can form aliases a sentinel
// (CS-13).
/// A complete accumulator: a fixed-point register spanning the entire product
/// exponent range of a float codec and the terminal integer-scaled expression,
/// so that every add is exact and no ordering can perturb it (§3.3).
///
/// This is the terminal fixed-point object of the Atlas contraction. It
/// contains no float arithmetic: the decode from an IEEE bit pattern to the
/// exact dyadic rational it names is [`FloatElement::decode`](crate::FloatElement::decode),
/// the pure-UOR body resolves signed Laurent coordinates through lookup and
/// addition, and only their exact coefficients and grades arrive here.
///
/// `L` is the low-limb count and `MIN_EXP` is the binary exponent of bit 0,
/// both derived from the element type in `model/widths.toml`. The exponent
/// origin is carried as a second const parameter rather than inferred, so that
/// a `Complete` value cannot be combined with one of a different origin. The
/// existing aligned tail word supplies the derived scalar headroom without
/// changing the concrete type or its size.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Complete<const L: usize, const MIN_EXP: i32> {
    limbs: Limbs<L>,
    /// A signed extension limb, or one of the seven nonempty unions of the
    /// former three flags. Reusing the old flag word preserves the concrete
    /// layout without collapsing any observable flag state.
    state: i64,
}

const COMPLETE_LIMB_RADIX: u128 = u64::MAX as u128 + 1;

const fn radix_has_flag(mask: u8, place: u8) -> bool {
    (mask / place) % 2 == 1
}

const fn radix_nonfinite_mask(nan: bool, pos_inf: bool, neg_inf: bool) -> u8 {
    let mut mask = 0;
    if nan {
        mask += COMPLETE_NAN_MASK;
    }
    if pos_inf {
        mask += COMPLETE_POS_INF_MASK;
    }
    if neg_inf {
        mask += COMPLETE_NEG_INF_MASK;
    }
    mask
}

const fn radix_union_nonfinite_masks(left: u8, right: u8) -> u8 {
    radix_nonfinite_mask(
        radix_has_flag(left, COMPLETE_NAN_MASK) || radix_has_flag(right, COMPLETE_NAN_MASK),
        radix_has_flag(left, COMPLETE_POS_INF_MASK) || radix_has_flag(right, COMPLETE_POS_INF_MASK),
        radix_has_flag(left, COMPLETE_NEG_INF_MASK) || radix_has_flag(right, COMPLETE_NEG_INF_MASK),
    )
}

fn radix_neg_limbs<const L: usize>(value: Limbs<L>) -> Limbs<L> {
    let mut out = [0u64; L];
    let mut carry = 1u64;
    for (at, word) in out.iter_mut().enumerate() {
        let complement = u64::MAX - value.0[at];
        let (sum, next) = complement.overflowing_add(carry);
        *word = sum;
        carry = u64::from(next);
    }
    Limbs(out)
}

const fn radix_limbs_is_negative<const L: usize>(value: Limbs<L>) -> bool {
    L != 0 && value.0[L - 1] / radix_power_u64(u64::BITS - 1) == 1
}

fn radix_binary_width(mut word: u64) -> u32 {
    let mut width = 0;
    while word != 0 {
        word /= 2;
        width += 1;
    }
    width
}

fn radix_spread_u128(magnitude: u128, coordinate: u32) -> [u64; 3] {
    let mut words = [
        (magnitude % COMPLETE_LIMB_RADIX) as u64,
        (magnitude / COMPLETE_LIMB_RADIX) as u64,
        0,
    ];
    let mut remaining = coordinate;
    while remaining != 0 {
        let mut carry = 0u128;
        for word in &mut words {
            let doubled = u128::from(*word) + u128::from(*word) + carry;
            *word = (doubled % COMPLETE_LIMB_RADIX) as u64;
            carry = doubled / COMPLETE_LIMB_RADIX;
        }
        debug_assert_eq!(carry, 0);
        remaining -= 1;
    }
    words
}

const fn radix_power_u64(mut coordinate: u32) -> u64 {
    let mut place = 1u64;
    while coordinate != 0 {
        place += place;
        coordinate -= 1;
    }
    place
}

const fn radix_scale_u64(mut value: u64, mut coordinate: u32) -> u64 {
    while coordinate != 0 {
        value += value;
        coordinate -= 1;
    }
    value
}

const fn compose_ieee_bits(
    negative: bool,
    exponent: u64,
    fraction: u64,
    mantissa_width: u32,
    exponent_width: u32,
) -> u64 {
    let sign = if negative {
        radix_power_u64(mantissa_width + exponent_width)
    } else {
        0
    };
    sign + radix_scale_u64(exponent, mantissa_width) + fraction
}

// `Debug` and `Hash` were derived over `limbs, nan, pos_inf, neg_inf` before
// the aligned flag tail became a signed extension. Keep those public
// observations in the former field order. The extension is deliberately not
// hashed: values equal under the new representation still hash equally, while
// every value representable before this change produces the same hash input it
// did before. Distinct finite extension values may collide, which `Hash`
// explicitly permits.
impl<const L: usize, const MIN_EXP: i32> core::fmt::Debug for Complete<L, MIN_EXP> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mask = self.nonfinite_mask().unwrap_or(0);
        formatter
            .debug_struct("Complete")
            .field("limbs", &self.limbs)
            .field("nan", &radix_has_flag(mask, COMPLETE_NAN_MASK))
            .field("pos_inf", &radix_has_flag(mask, COMPLETE_POS_INF_MASK))
            .field("neg_inf", &radix_has_flag(mask, COMPLETE_NEG_INF_MASK))
            .finish()
    }
}

impl<const L: usize, const MIN_EXP: i32> core::hash::Hash for Complete<L, MIN_EXP> {
    fn hash<H: core::hash::Hasher>(&self, hasher: &mut H) {
        let mask = self.nonfinite_mask().unwrap_or(0);
        core::hash::Hash::hash(&self.limbs, hasher);
        core::hash::Hash::hash(&radix_has_flag(mask, COMPLETE_NAN_MASK), hasher);
        core::hash::Hash::hash(&radix_has_flag(mask, COMPLETE_POS_INF_MASK), hasher);
        core::hash::Hash::hash(&radix_has_flag(mask, COMPLETE_NEG_INF_MASK), hasher);
    }
}

#[allow(dead_code)]
struct CompleteFlagTailLayout<const L: usize> {
    limbs: Limbs<L>,
    nan: bool,
    pos_inf: bool,
    neg_inf: bool,
}

// Evaluated by every library build, including no-std 32-bit targets: the
// extension occupies exactly the storage and alignment of the former flags.
// Keeping this outside the test harness makes the layout claim cross-checkable
// on targets that cannot link `std` or `libtest`.
const _: () = assert!(
    core::mem::size_of::<
        Complete<
            { crate::generated::complete_width::F32_LIMBS },
            { crate::generated::complete_width::F32_MIN_PRODUCT_EXP },
        >,
    >() == core::mem::size_of::<
        CompleteFlagTailLayout<{ crate::generated::complete_width::F32_LIMBS }>,
    >()
);
const _: () = assert!(
    core::mem::align_of::<
        Complete<
            { crate::generated::complete_width::F32_LIMBS },
            { crate::generated::complete_width::F32_MIN_PRODUCT_EXP },
        >,
    >() == core::mem::align_of::<
        CompleteFlagTailLayout<{ crate::generated::complete_width::F32_LIMBS }>,
    >()
);
const _: () = assert!(
    core::mem::size_of::<
        Complete<
            { crate::generated::complete_width::F64_LIMBS },
            { crate::generated::complete_width::F64_MIN_PRODUCT_EXP },
        >,
    >() == core::mem::size_of::<
        CompleteFlagTailLayout<{ crate::generated::complete_width::F64_LIMBS }>,
    >()
);
const _: () = assert!(
    core::mem::align_of::<
        Complete<
            { crate::generated::complete_width::F64_LIMBS },
            { crate::generated::complete_width::F64_MIN_PRODUCT_EXP },
        >,
    >() == core::mem::align_of::<
        CompleteFlagTailLayout<{ crate::generated::complete_width::F64_LIMBS }>,
    >()
);

#[derive(Clone, Copy)]
struct CompleteMagnitude<const L: usize> {
    low: Limbs<L>,
    high: u64,
}

impl<const L: usize> CompleteMagnitude<L> {
    fn limb(&self, at: usize) -> u64 {
        if at < L {
            self.low.0[at]
        } else if at == L {
            self.high
        } else {
            0
        }
    }

    fn high_bit(&self) -> Option<u32> {
        if self.high != 0 {
            return Some((L as u32).wrapping_mul(u64::BITS) + radix_binary_width(self.high) - 1);
        }
        let mut at = L;
        while at != 0 {
            at -= 1;
            let word = self.low.0[at];
            if word != 0 {
                return Some((at as u32).wrapping_mul(u64::BITS) + radix_binary_width(word) - 1);
            }
        }
        None
    }

    fn radix_bit(&self, i: u32) -> bool {
        let limb = (i / u64::BITS) as usize;
        let mut word = self.limb(limb);
        let mut coordinate = i % u64::BITS;
        while coordinate != 0 {
            word /= 2;
            coordinate -= 1;
        }
        word % 2 == 1
    }

    fn radix_window(&self, at: u32, n: u32) -> u64 {
        debug_assert!(n <= u64::BITS);
        let mut limb = (at / u64::BITS) as usize;
        let mut within = at % u64::BITS;
        let mut word = self.limb(limb);
        let mut discarded = within;
        while discarded != 0 {
            word /= 2;
            discarded -= 1;
        }

        let mut remaining = n;
        let mut out = 0u64;
        let mut place = 1u64;
        while remaining != 0 {
            if word % 2 == 1 {
                out += place;
            }
            word /= 2;
            within += 1;
            remaining -= 1;
            if remaining != 0 {
                place += place;
            }
            if within == u64::BITS {
                limb += 1;
                within = 0;
                word = self.limb(limb);
            }
        }
        out
    }

    fn radix_any_below(&self, i: u32) -> bool {
        let full = (i / u64::BITS) as usize;
        for at in 0..full.min(L + 1) {
            if self.limb(at) != 0 {
                return true;
            }
        }
        if full <= L {
            let mut remaining = i % u64::BITS;
            let mut word = self.limb(full);
            while remaining != 0 {
                if word % 2 == 1 {
                    return true;
                }
                word /= 2;
                remaining -= 1;
            }
        }
        false
    }
}

impl<const L: usize, const MIN_EXP: i32> Complete<L, MIN_EXP> {
    /// Zero.
    pub const ZERO: Self = Self {
        limbs: Limbs::ZERO,
        state: 0,
    };

    /// The binary exponent of bit 0 of the register.
    pub const MIN_EXP: i32 = MIN_EXP;

    /// The register's established low-carrier width in bits.
    ///
    /// The aligned tail word is terminal headroom, not another public limb:
    /// retaining this value preserves the API contract while the model records
    /// the physical width separately.
    pub const WIDTH: u32 = (L as u32).wrapping_mul(u64::BITS);

    const fn nonfinite_mask(&self) -> Option<u8> {
        let first_finite =
            COMPLETE_NONFINITE_STATE_BASE.wrapping_add(COMPLETE_NONFINITE_STATE_COUNT as i64);
        if self.state < first_finite {
            Some(
                self.state
                    .wrapping_sub(COMPLETE_NONFINITE_STATE_BASE)
                    .wrapping_add(1) as u8,
            )
        } else {
            None
        }
    }

    const fn nonfinite_state(mask: u8) -> i64 {
        debug_assert!(mask != 0 && mask as u32 <= COMPLETE_NONFINITE_STATE_COUNT);
        match mask {
            COMPLETE_NAN_MASK => COMPLETE_NAN_STATE,
            COMPLETE_POS_INF_MASK => COMPLETE_POS_INF_STATE,
            COMPLETE_NEG_INF_MASK => COMPLETE_NEG_INF_STATE,
            _ => COMPLETE_NONFINITE_STATE_BASE
                .wrapping_add(mask as i64)
                .wrapping_sub(1),
        }
    }

    const fn is_finite(&self) -> bool {
        self.nonfinite_mask().is_none()
    }

    /// Keep arithmetic on a finite object from manufacturing a non-finite tag.
    ///
    /// The associated `f32` and `f64` widths prove this branch unreachable for
    /// every matrix product and terminal scalar expression. It remains
    /// load-bearing for a caller deliberately instantiating an undersized
    /// public `Complete`: exhausting that chosen register must not silently
    /// turn an integer into NaN or infinity.
    fn retain_finite_state(&mut self) {
        let first_finite =
            COMPLETE_NONFINITE_STATE_BASE.wrapping_add(COMPLETE_NONFINITE_STATE_COUNT as i64);
        if self.state < first_finite {
            self.state = first_finite;
        }
    }

    const fn nan_value() -> Self {
        Self {
            limbs: Limbs::ZERO,
            state: Self::nonfinite_state(COMPLETE_NAN_MASK),
        }
    }

    fn combine_nonfinite(self, other: Self) -> Self {
        let mask = radix_union_nonfinite_masks(
            self.nonfinite_mask().unwrap_or(0),
            other.nonfinite_mask().unwrap_or(0),
        );
        debug_assert!(
            mask != 0,
            "the caller found at least one non-finite operand"
        );
        Self {
            // The former flag representation always combined its low limbs as
            // well as unioning flags. The high extension has no numeric
            // meaning once a flag is present, but preserving this low combine
            // keeps raw/magnitude/equality observations byte-compatible.
            limbs: self.limbs.combine(other.limbs),
            state: Self::nonfinite_state(mask),
        }
    }

    fn scale_nonfinite(self, factor: i64) -> Self {
        let flip = factor < 0;
        let mut magnitude = factor.unsigned_abs();
        let mut low = Limbs::ZERO;
        let mut addend = if flip {
            radix_neg_limbs(self.limbs)
        } else {
            self.limbs
        };
        while magnitude > 0 {
            if magnitude % 2 == 1 {
                low = low.combine(addend);
            }
            magnitude /= 2;
            if magnitude != 0 {
                addend = addend.combine(addend);
            }
        }

        let mask = self
            .nonfinite_mask()
            .expect("the non-finite scaling helper has a flag union");
        let mask = if flip {
            radix_nonfinite_mask(
                radix_has_flag(mask, COMPLETE_NAN_MASK),
                radix_has_flag(mask, COMPLETE_NEG_INF_MASK),
                radix_has_flag(mask, COMPLETE_POS_INF_MASK),
            )
        } else {
            mask
        };
        Self {
            limbs: low,
            state: Self::nonfinite_state(mask),
        }
    }

    fn finite_neg(self) -> Self {
        let low_nonzero = self.limbs.0.iter().any(|&limb| limb != 0);
        // -(h*B + lo) = (-h - [lo != 0])*B + (-lo mod B).
        // The model-derived terminal bound leaves enough signed high bits that
        // this conversion cannot reach a non-finite sentinel.
        let high = -i128::from(self.state) - i128::from(u8::from(low_nonzero));
        let mut out = Self {
            limbs: radix_neg_limbs(self.limbs),
            state: high as i64,
        };
        out.retain_finite_state();
        out
    }

    fn full_magnitude(&self) -> CompleteMagnitude<L> {
        if !self.is_finite() {
            let low = if radix_limbs_is_negative(self.limbs) {
                radix_neg_limbs(self.limbs)
            } else {
                self.limbs
            };
            return CompleteMagnitude { low, high: 0 };
        }
        if !self.is_negative() {
            return CompleteMagnitude {
                low: self.limbs,
                high: self.state as u64,
            };
        }

        // Two's-complement negation crosses into the high word exactly when
        // every low limb is zero.
        let low_is_zero = self.limbs.0.iter().all(|&limb| limb == 0);
        CompleteMagnitude {
            low: radix_neg_limbs(self.limbs),
            high: (u64::MAX - self.state as u64).wrapping_add(u64::from(low_is_zero)),
        }
    }

    /// Accumulate `sign * mag * 2^exp`, exactly.
    ///
    /// `mag` is an already-resolved Atlas coefficient and `exp` is its Laurent
    /// grade. Every bit of the result lands in the register, because the
    /// register spans the whole product exponent range; nothing is rounded,
    /// and nothing is dropped.
    ///
    /// A product whose exponent falls outside the register is impossible for
    /// any pair of finite inputs of the element type the register was sized
    /// for. A hand-built, undersized `Complete` remains total by retaining only
    /// its representable coordinates; no associated float accumulator can take
    /// that branch.
    pub fn add_scaled(&mut self, mag: u128, exp: i32, negative: bool) {
        self.add_scaled_i64(mag, i64::from(exp), negative);
    }

    /// The private terminal placement while an Atlas address is still the
    /// signed sum of two public exponent coordinates. Narrowing that sum before
    /// the register compares it with its own extent would introduce an
    /// artificial overflow at the `i32` boundary.
    pub(crate) fn add_scaled_i64(&mut self, mag: u128, exp: i64, negative: bool) {
        if mag == 0 {
            return;
        }
        let Some(shift) = exp.checked_sub(i64::from(MIN_EXP)) else {
            return;
        };
        if shift < 0 {
            return;
        }
        let shift = shift as u64;
        let Ok(at) = usize::try_from(shift / u64::from(u64::BITS)) else {
            // The coordinate is beyond any addressable limb of this register.
            return;
        };
        // `L` is the width the public instantiation requested. The aligned
        // tail word absorbs carries and terminal scalar growth; it is not an
        // independently addressable coordinate limb.
        if at >= L {
            return;
        }
        let bit = (shift % u64::from(u64::BITS)) as u32;

        // Regrading is the Atlas action of `X`: repeated doubling with a radix
        // carry. It retains all three words of a straddling coefficient
        // without interpreting the carrier as a packed bit field.
        let spread = radix_spread_u128(mag, bit);

        // Add or subtract in place, at the resolved address, propagating only
        // as far as the carry reaches. Materializing a full-width coordinate
        // would cost O(L) per Atlas placement and would violate the zero-copy
        // carrier discipline; this touches three limbs plus the live carry.
        let finite = self.is_finite();
        let mut discarded_extension = 0;
        let extension = if finite {
            &mut self.state
        } else {
            // The former flag representation retained low-limb arithmetic
            // after a flag arrived and discarded its carry at the established
            // low width. Preserve that public raw/equality observation while
            // keeping the tail sentinel unchanged.
            &mut discarded_extension
        };
        if negative {
            sub_at(&mut self.limbs.0, extension, at, spread);
        } else {
            add_at(&mut self.limbs.0, extension, at, spread);
        }
        if finite {
            self.retain_finite_state();
        }
    }

    /// The accumulator holding exactly the value `x` names.
    ///
    /// The exact embedding the epilogue needs: `beta * C` requires the value
    /// already in `C` to enter the accumulation, and it enters by the same
    /// total code decode every operand uses.
    pub fn of<E>(x: E) -> Self
    where
        E: crate::alphabet::FloatElement,
    {
        use crate::alphabet::Decoded;
        let mut out = Self::ZERO;
        match x.decode() {
            Decoded::NotANumber => out.set_nan(),
            Decoded::Infinite { sign } => out.set_infinity(sign),
            Decoded::Finite {
                sign,
                mantissa,
                exp,
            } => out.add_scaled(mantissa as u128, exp, sign),
        }
        out
    }

    /// Multiply by an exact integer scalar.
    ///
    /// Double-and-add over the register's own `combine`, so there is no second
    /// multiplication routine and no rounding: an integer scaling of an exact
    /// fixed-point value is exact.
    pub fn scale(self, factor: i64) -> Self {
        if self.is_nan() {
            return self;
        }
        if factor == 1 {
            // `alpha == 1` is the overwhelmingly common epilogue, and it is the
            // identity. Without this it walked the double-and-add loop once
            // anyway: one `combine` of `L` limbs into the accumulator plus one
            // dead `combine` doubling the addend, and two full-width register
            // materializations, per output element. At 67 low limbs plus the
            // extension word that is 544 bytes moved four times to compute
            // `x * 1`.
            return self;
        }
        if factor == 0 {
            return if self.infinity_sign().is_some() {
                Self::nan_value()
            } else {
                Self::ZERO
            };
        }
        if self.infinity_sign().is_some() {
            return self.scale_nonfinite(factor);
        }
        let flip = factor < 0;
        let mut magnitude = factor.unsigned_abs();
        let mut acc = Self::ZERO;
        let mut addend = if flip { self.finite_neg() } else { self };
        while magnitude > 0 {
            if magnitude % 2 == 1 {
                acc = acc.combine(addend);
            }
            magnitude /= 2;
            // Avoid forming a final, unused double. At `i64::MIN` that value
            // has one more bit than the scalar product and is not part of the
            // terminal expression's model-derived bound.
            if magnitude != 0 {
                addend = addend.combine(addend);
            }
        }
        acc
    }

    /// Accumulate one already-resolved signed coefficient at Laurent grade
    /// `exp`. This is a terminal embedding primitive, not a product route: the
    /// pure-UOR traversal has performed normalization, projection, lookup and
    /// coordinate contraction before calling it.
    pub fn add_signed(&mut self, mantissa: i64, exp: i32) {
        if mantissa == 0 {
            return;
        }
        let Some(shift) = i64::from(exp).checked_sub(i64::from(MIN_EXP)) else {
            return;
        };
        if shift < 0 {
            return;
        }
        let shift = shift as u64;
        let Ok(at) = usize::try_from(shift / u64::from(u64::BITS)) else {
            return;
        };
        if at >= L {
            return;
        }
        let bit = (shift % u64::from(u64::BITS)) as u32;
        let mag = mantissa.unsigned_abs();

        let spread = radix_spread_u128(u128::from(mag), bit);
        let finite = self.is_finite();
        let mut discarded_extension = 0;
        let extension = if finite {
            &mut self.state
        } else {
            &mut discarded_extension
        };
        if mantissa < 0 {
            sub_at(&mut self.limbs.0, extension, at, spread);
        } else {
            add_at(&mut self.limbs.0, extension, at, spread);
        }
        if finite {
            self.retain_finite_state();
        }
    }

    /// Record that a NaN reached this accumulation. Sticky and absorbing.
    pub fn set_nan(&mut self) {
        let mask =
            radix_union_nonfinite_masks(self.nonfinite_mask().unwrap_or(0), COMPLETE_NAN_MASK);
        self.state = Self::nonfinite_state(mask);
    }

    /// Record that an infinity of the given sign reached this accumulation.
    ///
    /// Two infinities of opposite sign make a NaN, by IEEE 754 clause 7.2, and
    /// they do so here whatever order they arrived in.
    pub fn set_infinity(&mut self, negative: bool) {
        let bit = if negative {
            COMPLETE_NEG_INF_MASK
        } else {
            COMPLETE_POS_INF_MASK
        };
        let mask = radix_union_nonfinite_masks(self.nonfinite_mask().unwrap_or(0), bit);
        self.state = Self::nonfinite_state(mask);
    }

    /// Has a NaN been accumulated, or has the sum become one?
    pub const fn is_nan(self) -> bool {
        let Some(mask) = self.nonfinite_mask() else {
            return false;
        };
        radix_has_flag(mask, COMPLETE_NAN_MASK)
            || (radix_has_flag(mask, COMPLETE_POS_INF_MASK)
                && radix_has_flag(mask, COMPLETE_NEG_INF_MASK))
    }

    /// The sign of the accumulated infinity, if the sum is infinite.
    ///
    /// `None` when the sum is finite or a NaN. A NaN is checked first, so
    /// `inf + (-inf)` reports a NaN rather than an arbitrary sign.
    pub const fn infinity_sign(self) -> Option<bool> {
        if self.is_nan() {
            None
        } else {
            match self.nonfinite_mask() {
                Some(COMPLETE_POS_INF_MASK) => Some(false),
                Some(COMPLETE_NEG_INF_MASK) => Some(true),
                _ => None,
            }
        }
    }

    /// The underlying limbs, for an encoder to round from.
    pub const fn raw(&self) -> &Limbs<L> {
        &self.limbs
    }

    /// Is the accumulated value negative?
    pub const fn is_negative(&self) -> bool {
        if self.is_finite() {
            self.state < 0
        } else {
            radix_limbs_is_negative(self.limbs)
        }
    }

    /// Is the accumulated value exactly zero?
    ///
    /// Exact cancellation is exact here, which is the property a classical
    /// accumulator loses first.
    pub fn is_zero(&self) -> bool {
        self.state == 0 && self.limbs.limbs().iter().all(|&limb| limb == 0)
    }

    /// The low limbs of the magnitude of this register.
    ///
    /// An encoder wants the leading one, then `P` significand bits, then a round
    /// bit and a sticky bit --- and every one of those is a question about the
    /// *magnitude*. Forming it once and asking [`Limbs`] the questions is the
    /// difference between one negation and one per bit read, which for `f64`'s
    /// sixty-seven limbs is fifty-five passes over the register per output
    /// element. Terminal scalar headroom lives in the extension word; use the
    /// bit-query methods below when the complete magnitude, rather than its
    /// historical low-limb view, is required.
    pub fn magnitude(&self) -> Limbs<L> {
        self.full_magnitude().low
    }

    /// The index of the highest set bit of the magnitude, or `None` for zero.
    ///
    /// This is what an encoder needs to find the leading one, from which the
    /// output exponent and the round and sticky bits follow.
    pub fn magnitude_high_bit(&self) -> Option<u32> {
        self.full_magnitude().high_bit()
    }

    /// Bit `i` of the magnitude, counting from bit 0 of the register.
    pub fn magnitude_bit(&self, i: u32) -> bool {
        self.full_magnitude().radix_bit(i)
    }

    /// Is any bit of the magnitude below `i` set? The sticky bit.
    pub fn magnitude_any_below(&self, i: u32) -> bool {
        self.full_magnitude().radix_any_below(i)
    }
}

impl<const L: usize, const MIN_EXP: i32> Accumulator for Complete<L, MIN_EXP> {
    const ZERO: Self = Self::ZERO;
    const BITS: u32 = (L as u32).wrapping_mul(u64::BITS);

    fn combine(self, other: Self) -> Self {
        // Every former flag union remains distinct, and the low limbs keep the
        // same associative combine they had beside those flags.
        if !self.is_finite() || !other.is_finite() {
            return self.combine_nonfinite(other);
        }

        let mut limbs = [0u64; L];
        let mut carry = 0u64;
        for (at, out) in limbs.iter_mut().enumerate() {
            let (sum, c1) = self.limbs.0[at].overflowing_add(other.limbs.0[at]);
            let (sum, c2) = sum.overflowing_add(carry);
            *out = sum;
            carry = u64::from(c1) + u64::from(c2);
        }
        // The model includes one terminal-expression bit above both scaled
        // terms, so this signed addition is exact for every public float call.
        let high = i128::from(self.state) + i128::from(other.state) + i128::from(carry);
        let mut out = Self {
            limbs: Limbs(limbs),
            state: high as i64,
        };
        out.retain_finite_state();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate std;

    /// The i128 accumulator is exact over the whole i8 x i8 domain at a depth
    /// far past the narrow-register threshold, which is the point: the
    /// threshold governs a register, not an answer (§3.2, CD-03).
    #[test]
    fn i128_accumulation_is_exact_past_the_narrow_threshold_cd_03() {
        // A million natively; a thousand under Miri, which is checking that the
        // limb arithmetic is *sound* --- provenance, bounds, initialisation ---
        // and sees that at a thousand as well as at a million, at something like
        // a hundredfold the cost per step. The depth claim is the native run's.
        // Measured: this loop alone was minutes of the Miri job.
        let depth: i128 = if cfg!(miri) { 1_000 } else { 1_000_000 };
        let mut acc = 0i128;
        for _ in 0..depth {
            <i8 as Element>::mac(&mut acc, i8::MIN, i8::MIN);
        }
        assert_eq!(acc, depth * 128 * 128);
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
            <i64 as Element>::mac(&mut acc, i64::MIN, i64::MIN);
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
        let mut both_a = C::ZERO;
        both_a.add_scaled(1, 200, false);
        both_a.add_scaled(1, -290, false);
        let mut both_b = C::ZERO;
        both_b.add_scaled(1, -290, false);
        both_b.add_scaled(1, 200, false);
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
        let mut acc = C::ZERO;
        acc.add_scaled(12345, 100, false);
        acc.add_scaled(12345, 100, true);
        assert!(acc.is_zero());
        assert_eq!(acc.magnitude_high_bit(), None);
    }

    /// The old non-finite flags already occupied an aligned word. Reusing that
    /// word as signed terminal headroom therefore changes neither associated
    /// accumulator type nor concrete layout, while its bit queries cross the
    /// low-carrier boundary in both signs.
    #[test]
    fn complete_tail_word_is_exact_headroom_without_layout_growth_cs_13() {
        type C32 = Complete<10, -298>;
        type C64 = Complete<67, -2148>;
        assert_eq!(C32::WIDTH, 10 * u64::BITS);
        assert_eq!(C64::WIDTH, 67 * u64::BITS);
        assert_eq!(<C32 as Accumulator>::BITS, C32::WIDTH);
        assert_eq!(<C64 as Accumulator>::BITS, C64::WIDTH);
        assert_eq!(
            core::mem::size_of::<C32>(),
            core::mem::size_of::<Limbs<10>>() + core::mem::size_of::<i64>()
        );
        assert_eq!(
            core::mem::size_of::<C64>(),
            core::mem::size_of::<Limbs<67>>() + core::mem::size_of::<i64>()
        );

        let mut value = C32::ZERO;
        value.add_scaled(1, 319, false);
        let scaled = value.scale(i64::MAX);
        assert!(!scaled.is_negative());
        assert!(scaled.magnitude_high_bit().is_some_and(|bit| bit >= 640));
        assert!(scaled.magnitude_bit(640));
        assert!(scaled.magnitude_any_below(640));

        let negated = scaled.scale(-1);
        assert!(negated.is_negative());
        assert_eq!(scaled.combine(negated), C32::ZERO);
    }

    /// `CS-13`: reusing the former flag tail must not change observations of
    /// the public `Complete` value. Every reachable union of the three former
    /// flags remains distinct, low limbs survive flag operations and combines,
    /// and the derived `Debug`/`Hash` views retain their former field order.
    #[test]
    fn complete_tail_preserves_former_flag_observations_cs_13() {
        use core::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        use std::format;

        #[derive(Hash)]
        struct Former<const N: usize> {
            limbs: Limbs<N>,
            nan: bool,
            pos_inf: bool,
            neg_inf: bool,
        }

        fn digest(value: &impl Hash) -> u64 {
            let mut hasher = DefaultHasher::new();
            value.hash(&mut hasher);
            hasher.finish()
        }

        type C = Complete<2, 0>;
        let mut finite = C::ZERO;
        finite.add_signed(5, 0);
        let low = *finite.raw();
        let mut states = [C::ZERO; COMPLETE_NONFINITE_STATE_COUNT as usize];

        for mask in 1..=COMPLETE_NONFINITE_STATE_COUNT as u8 {
            let mut value = finite;
            if mask & COMPLETE_NAN_MASK != 0 {
                value.set_nan();
            }
            if mask & COMPLETE_POS_INF_MASK != 0 {
                value.set_infinity(false);
            }
            if mask & COMPLETE_NEG_INF_MASK != 0 {
                value.set_infinity(true);
            }
            states[usize::from(mask - 1)] = value;

            assert_eq!(value.nonfinite_mask(), Some(mask));
            assert_eq!(*value.raw(), low, "mask {mask} discarded low limbs");
            assert_eq!(
                value.is_nan(),
                mask & COMPLETE_NAN_MASK != 0
                    || mask & (COMPLETE_POS_INF_MASK | COMPLETE_NEG_INF_MASK)
                        == (COMPLETE_POS_INF_MASK | COMPLETE_NEG_INF_MASK)
            );
            let expected_sign = match mask {
                COMPLETE_POS_INF_MASK => Some(false),
                COMPLETE_NEG_INF_MASK => Some(true),
                _ => None,
            };
            assert_eq!(value.infinity_sign(), expected_sign);

            let nan = mask & COMPLETE_NAN_MASK != 0;
            let pos_inf = mask & COMPLETE_POS_INF_MASK != 0;
            let neg_inf = mask & COMPLETE_NEG_INF_MASK != 0;
            assert_eq!(
                format!("{value:?}"),
                format!(
                    "Complete {{ limbs: {low:?}, nan: {nan}, pos_inf: {pos_inf}, neg_inf: {neg_inf} }}"
                )
            );
            assert_eq!(
                digest(&value),
                digest(&Former {
                    limbs: low,
                    nan,
                    pos_inf,
                    neg_inf,
                }),
                "mask {mask} changed the former derived Hash field stream"
            );
        }
        for left in 0..states.len() {
            for right in 0..states.len() {
                assert_eq!(states[left] == states[right], left == right);
            }
        }

        let mut negative = C::ZERO;
        negative.add_signed(-5, 0);
        let negative_low = *negative.raw();
        negative.set_infinity(false);
        assert!(negative.is_negative());
        assert_eq!(negative.magnitude(), negative_low.neg());

        let scaled = negative.scale(-3);
        let mut expected_low = Limbs::ZERO;
        expected_low = expected_low.combine(negative_low.neg());
        expected_low = expected_low.combine(negative_low.neg());
        expected_low = expected_low.combine(negative_low.neg());
        assert_eq!(*scaled.raw(), expected_low);
        assert_eq!(scaled.infinity_sign(), Some(true));

        let mut left = finite;
        left.set_infinity(false);
        let mut right = finite;
        right.set_nan();
        let combined = left.combine(right);
        assert_eq!(*combined.raw(), low.combine(low));
        assert_eq!(
            combined.nonfinite_mask(),
            Some(COMPLETE_NAN_MASK | COMPLETE_POS_INF_MASK)
        );

        let mut opposite = finite;
        opposite.set_infinity(false);
        opposite.set_infinity(true);
        assert!(opposite.is_nan());
        assert_ne!(opposite, states[usize::from(COMPLETE_NAN_MASK - 1)]);

        let zero_times_infinity = left.scale(0);
        assert_eq!(*zero_times_infinity.raw(), Limbs::ZERO);
        assert_eq!(
            zero_times_infinity.nonfinite_mask(),
            Some(COMPLETE_NAN_MASK)
        );
    }

    /// `CS-13`: the tail sentinel replaces storage, not the former public
    /// behaviour. Low-limb placement continues after every flag union, combine
    /// still adds those low limbs while unioning flags, and scalar multiplication
    /// retains HEAD's absorbing-NaN and signed-infinity rules.
    #[test]
    fn complete_public_operations_preserve_all_former_nonfinite_states_cs_13() {
        use std::format;

        type C = Complete<2, 0>;

        fn flagged(mask: u8, low: i128) -> C {
            let mut value = C::ZERO;
            value.add_signed(low as i64, 0);
            if mask & COMPLETE_NAN_MASK != 0 {
                value.set_nan();
            }
            if mask & COMPLETE_POS_INF_MASK != 0 {
                value.set_infinity(false);
            }
            if mask & COMPLETE_NEG_INF_MASK != 0 {
                value.set_infinity(true);
            }
            value
        }

        for mask in 1..=COMPLETE_NONFINITE_STATE_COUNT as u8 {
            let mut placed = flagged(mask, 5);
            placed.add_scaled(3, 1, false);
            placed.add_signed(-2, 0);
            assert_eq!(placed.nonfinite_mask(), Some(mask));
            assert_eq!(*placed.raw(), Limbs([9, 0]));

            let nan = mask & COMPLETE_NAN_MASK != 0;
            let pos_inf = mask & COMPLETE_POS_INF_MASK != 0;
            let neg_inf = mask & COMPLETE_NEG_INF_MASK != 0;
            assert_eq!(
                format!("{placed:?}"),
                format!(
                    "Complete {{ limbs: {:?}, nan: {nan}, pos_inf: {pos_inf}, neg_inf: {neg_inf} }}",
                    Limbs([9, 0])
                )
            );

            // A carry out of the former low carrier was discarded after flags
            // arrived. The sentinel must not accidentally turn that carry into
            // a different flag union or a finite extension value.
            let mut edge = flagged(mask, 0);
            edge.add_scaled(1, 127, false);
            assert_eq!(*edge.raw(), Limbs([0, radix_power_u64(63)]));
            edge.add_scaled(1, 127, false);
            assert_eq!(*edge.raw(), Limbs::ZERO);
            assert_eq!(edge.nonfinite_mask(), Some(mask));

            for factor in [-3, 0, 1, 2, i64::MIN] {
                let scaled = flagged(mask, 5).scale(factor);
                let absorbing_nan = nan || (pos_inf && neg_inf);
                if absorbing_nan {
                    assert_eq!(scaled, flagged(mask, 5));
                } else if factor == 0 {
                    assert_eq!(*scaled.raw(), Limbs::ZERO);
                    assert_eq!(scaled.nonfinite_mask(), Some(COMPLETE_NAN_MASK));
                } else {
                    let expected_low = Limbs::ZERO.add_i128(i128::from(factor) * 5);
                    let expected_mask = if factor < 0 {
                        if pos_inf {
                            COMPLETE_NEG_INF_MASK
                        } else {
                            COMPLETE_POS_INF_MASK
                        }
                    } else {
                        mask
                    };
                    assert_eq!(*scaled.raw(), expected_low);
                    assert_eq!(scaled.nonfinite_mask(), Some(expected_mask));
                }
            }
        }

        for left_mask in 1..=COMPLETE_NONFINITE_STATE_COUNT as u8 {
            for right_mask in 1..=COMPLETE_NONFINITE_STATE_COUNT as u8 {
                let combined = flagged(left_mask, 5).combine(flagged(right_mask, 7));
                assert_eq!(*combined.raw(), Limbs([12, 0]));
                assert_eq!(
                    combined.nonfinite_mask(),
                    Some(left_mask | right_mask),
                    "left={left_mask}, right={right_mask}"
                );
            }
        }
    }

    /// `CS-13`: an undersized public const-generic register can exhaust the
    /// finite encoding, but a finite arithmetic operation can never alias one
    /// of the seven nonempty non-finite flag unions.
    #[test]
    fn undersized_complete_arithmetic_never_aliases_nonfinite_state_cs_13() {
        type Tiny = Complete<0, 0>;
        let mut outside = Tiny::ZERO;
        outside.add_scaled(1, 0, false);
        let scaled = outside.scale(i64::MIN);
        assert!(scaled.is_zero());
        assert!(!scaled.is_nan());
        assert_eq!(scaled.infinity_sign(), None);

        type One = Complete<1, 0>;
        let edge = One {
            limbs: Limbs::ZERO,
            state: i64::MIN / 2,
        };
        assert!(edge.is_finite());
        let doubled = edge.combine(edge);
        assert!(doubled.is_finite());
        assert!(!doubled.is_nan());
        assert_eq!(doubled.infinity_sign(), None);
    }

    /// `CS-13`: the public signed-coordinate specialization is byte-identical
    /// to the general terminal placement, including a contribution that
    /// crosses from the last low limb into the extension word.
    #[test]
    fn signed_coordinate_specialization_matches_general_placement_cs_13() {
        type C = Complete<2, 0>;
        for mantissa in [i64::MIN, i64::MIN + 1, -257, -1, 1, 257, i64::MAX] {
            for exp in [0, 1, 63, 64, 65, 127] {
                let mut signed = C::ZERO;
                signed.add_signed(mantissa, exp);
                let mut general = C::ZERO;
                general.add_scaled(u128::from(mantissa.unsigned_abs()), exp, mantissa < 0);
                assert_eq!(signed, general, "mantissa={mantissa}, exp={exp}");
            }
        }
    }

    #[test]
    fn complete_radix_primitives_match_independent_bit_oracles_cu_11() {
        let magnitudes = [
            0,
            1,
            u128::from(u64::MAX),
            1u128 << 64,
            (1u128 << 64) + 1,
            u128::MAX,
        ];
        for magnitude in magnitudes {
            for coordinate in 0..u64::BITS {
                let expected = if coordinate == 0 {
                    [magnitude as u64, (magnitude >> 64) as u64, 0]
                } else {
                    [
                        (magnitude << coordinate) as u64,
                        ((magnitude << coordinate) >> 64) as u64,
                        (magnitude >> (128 - coordinate)) as u64,
                    ]
                };
                assert_eq!(radix_spread_u128(magnitude, coordinate), expected);
            }
        }

        let magnitude = CompleteMagnitude {
            low: Limbs([0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210]),
            high: 0x8000_0000_0000_0001,
        };
        let words = [magnitude.low.0[0], magnitude.low.0[1], magnitude.high];
        for coordinate in 0..(words.len() as u32 * u64::BITS) {
            let word = words[(coordinate / u64::BITS) as usize];
            let expected = (word >> (coordinate % u64::BITS)) & 1 == 1;
            assert_eq!(magnitude.radix_bit(coordinate), expected);
            let expected_sticky = (0..coordinate).any(|lower| {
                let word = words[(lower / u64::BITS) as usize];
                (word >> (lower % u64::BITS)) & 1 == 1
            });
            assert_eq!(magnitude.radix_any_below(coordinate), expected_sticky);
            for width in 0..=u64::BITS.min(words.len() as u32 * u64::BITS - coordinate) {
                let expected_window = (0..width).fold(0u64, |window, offset| {
                    let bit_at = coordinate + offset;
                    let word = words[(bit_at / u64::BITS) as usize];
                    window | (((word >> (bit_at % u64::BITS)) & 1) << offset)
                });
                assert_eq!(magnitude.radix_window(coordinate, width), expected_window);
            }
        }

        for mask in 0..=COMPLETE_NONFINITE_STATE_COUNT as u8 {
            for other in 0..=COMPLETE_NONFINITE_STATE_COUNT as u8 {
                assert_eq!(radix_union_nonfinite_masks(mask, other), mask | other);
            }
        }
        for (negative, exponent, fraction, mantissa, exponent_width) in [
            (false, 0, 0, 23, 8),
            (true, 0xff, 0, 23, 8),
            (false, 0x7fe, (1u64 << 52) - 1, 52, 11),
            (true, 0x3ff, 1, 52, 11),
        ] {
            let expected = ((u64::from(negative)) << (mantissa + exponent_width))
                | (exponent << mantissa)
                | fraction;
            assert_eq!(
                compose_ieee_bits(negative, exponent, fraction, mantissa, exponent_width),
                expected
            );
        }
    }
}

/// The correctly-rounded encode out of a complete accumulator (§3.3).
///
/// This is the single encode step for the float family, and it is the only
/// place in the float path where information is discarded. Everything upstream
/// of it --- code decode, canonical Laurent normalization, Atlas projection,
/// lookup contraction and fixed-point embedding --- is exact, so what this
/// rounds is the *exact sum*, once.
///
/// That is what makes the result schedule-independent: a classical GEMM rounds
/// after every add, so its answer depends on the order; this one rounds after
/// the last add, so it cannot.
macro_rules! impl_encode_into_float {
    ($t:ty, $bits:ty, $mant:expr, $expo:expr) => {
        impl<const L: usize, const MIN_EXP: i32> EncodeFrom<Complete<L, MIN_EXP>> for $t {
            fn encode_from(acc: Complete<L, MIN_EXP>, mode: EncodeMode) -> Self {
                const MANT: u32 = $mant;
                const EXPO: u32 = $expo;
                /// Significand bits including the implicit one.
                const P: u32 = MANT + 1;
                const BIAS: i32 = radix_power_u64(EXPO - 1) as i32 - 1;
                /// The exponent of the least significant bit of a subnormal.
                const SUB_LSB_EXP: i32 = 1 - BIAS - MANT as i32;

                // The non-finite cases first, so that a NaN never reaches the
                // rounding logic and an infinity never competes with it.
                if acc.is_nan() {
                    return <$t>::NAN;
                }
                if let Some(negative) = acc.infinity_sign() {
                    let bits = compose_ieee_bits(negative, radix_power_u64(EXPO) - 1, 0, MANT, EXPO)
                        as $bits;
                    return <$t>::from_bits(bits);
                }

                let negative = acc.is_negative();
                let mag = acc.full_magnitude();
                let Some(high) = mag.high_bit() else {
                    // An exactly zero sum. IEEE 754 gives `+0` for a sum that
                    // cancels under round-to-nearest, and the sign of a zero
                    // that arose from cancellation carries no information, so
                    // the positive zero is returned rather than invented.
                    return <$t>::from_bits(0);
                };

                // The value is `magnitude * 2^MIN_EXP`, with its leading one at
                // register bit `high`, so its binary exponent is this.
                let exp = MIN_EXP + high as i32;

                // Where the output's least significant bit falls, in register
                // bits. Clamped at the subnormal floor, which is what makes the
                // gradual-underflow region come out right rather than being a
                // separate branch.
                let lsb_exp = if exp - (P as i32 - 1) < SUB_LSB_EXP {
                    SUB_LSB_EXP
                } else {
                    exp - (P as i32 - 1)
                };
                let lsb = lsb_exp - MIN_EXP;
                if lsb < 0 {
                    // Unreachable for an accumulator sized by the model: the
                    // register's floor is the minimum *product* exponent, which
                    // is below the minimum subnormal. Returning zero rather
                    // than indexing negatively keeps the function total for a
                    // hand-built `Complete` of the wrong size.
                    return <$t>::from_bits(0);
                }
                let lsb = lsb as u32;

                // The P significand bits, read out of the register in one
                // window rather than one at a time.
                let sig: u64 = mag.radix_window(lsb, P);

                // Round, once, under the caller's mode. `Nearest` is
                // round-half-to-even, which is IEEE 754's default and what
                // makes this *the* correctly-rounded value rather than *a*
                // rounding of it.
                let (round_bit, sticky) = if lsb == 0 {
                    (false, false)
                } else {
                    (mag.radix_bit(lsb - 1), mag.radix_any_below(lsb - 1))
                };
                let increment = match mode {
                    EncodeMode::Nearest => round_bit && (sticky || sig % 2 == 1),
                    // Truncation toward zero: the magnitude is already
                    // truncated, so there is nothing to add.
                    EncodeMode::TowardZero | EncodeMode::Saturating | EncodeMode::Wrapping => false,
                };
                let mut sig = sig + u64::from(increment);
                let mut lsb_exp = lsb_exp;
                if sig / radix_power_u64(P) == 1 {
                    // The increment carried out of the significand: halve it
                    // and step the exponent, which is exact.
                    sig /= 2;
                    lsb_exp += 1;
                }

                if sig == 0 {
                    return <$t>::from_bits(compose_ieee_bits(negative, 0, 0, MANT, EXPO) as $bits);
                }

                // Assemble. A significand with its top bit set is normal; one
                // without is subnormal, and its biased exponent is zero.
                let top = radix_binary_width(sig) - 1;
                let value_exp = lsb_exp + top as i32;
                let bits = if sig / radix_power_u64(P - 1) == 1 {
                    let biased = value_exp + BIAS;
                    if biased >= radix_power_u64(EXPO) as i32 - 1 {
                        // Overflow. Round-to-nearest carries an overflowing
                        // magnitude to infinity; the directed modes clamp to
                        // the largest finite value, which is what `Saturating`
                        // means for this family.
                        return match mode {
                            EncodeMode::Nearest => {
                                if negative {
                                    <$t>::NEG_INFINITY
                                } else {
                                    <$t>::INFINITY
                                }
                            }
                            _ => {
                                if negative {
                                    <$t>::MIN
                                } else {
                                    <$t>::MAX
                                }
                            }
                        };
                    }
                    compose_ieee_bits(
                        negative,
                        biased as u64,
                        sig % radix_power_u64(MANT),
                        MANT,
                        EXPO,
                    )
                } else {
                    compose_ieee_bits(negative, 0, sig, MANT, EXPO)
                } as $bits;
                <$t>::from_bits(bits)
            }
        }
    };
}

impl_encode_into_float!(f32, u32, 23, 8);
impl_encode_into_float!(f64, u64, 52, 11);

/// An exact integer sum encoding into a float.
///
/// An `i128` accumulator is a dyadic rational at exponent zero. This impl is
/// the scale-zero embedding required by generic epilogues and compatibility
/// bounds; the pure-UOR float traversal itself contracts Atlas octets directly
/// into `Complete`. It enters through [`Complete::add_scaled`], so nothing is
/// rounded on the way in; the register spans the format's whole product
/// exponent range and then some, so no `i128` is too wide for it.
macro_rules! impl_encode_from_i128_into_float {
    ($($t:ty),* $(,)?) => { $(
        impl EncodeFrom<i128> for $t {
            fn encode_from(acc: i128, mode: EncodeMode) -> Self {
                let mut placed = <$t as Element>::Acc::ZERO;
                placed.add_scaled(acc.unsigned_abs(), 0, acc < 0);
                <$t as EncodeFrom<<$t as Element>::Acc>>::encode_from(placed, mode)
            }
        }
    )* };
}

impl_encode_from_i128_into_float!(f32, f64);
