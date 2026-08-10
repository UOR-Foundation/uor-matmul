//! The `(max, +)` semiring, as an element family (A-4, A-6).
//!
//! The formalization's operation census names two products, not one: *matrix
//! products under complete accumulation*, and *max*. This module is the second
//! one, and it is the same method as the first --- decode, accumulate exactly,
//! encode once --- at a different instantiation of the two operations the
//! accumulation is written against.
//!
//! # Why this is an element type and not a parameter
//!
//! [`Element::Acc`] is *not a parameter and not a choice*: exactly one
//! accumulator per element type. A semiring parameter on the driver would
//! contradict that, because the two semirings do not share an accumulator ---
//! the ring's is 79 bits at `i8` and this one's is ten. So the semiring is
//! carried where the arithmetic already is, in the element type, and every
//! traversal in the library is parametric in it *for free*: [`crate::dot::
//! dot_ref`] and `uor_matmul_gemm::gemm` name `E::mac` and
//! [`Accumulator::combine`] and nothing else, so they compute a `(max, +)`
//! product with no branch, no second driver, and no line changed.
//!
//! That is the whole content of the claim that this library is parametric in
//! its algebra rather than specialized to one.
//!
//! # The semiring zero is the pad and the mask
//!
//! [`Element::ZERO`] is documented as *the additive identity, which is what a
//! padded position decodes to*. Here that value is `-inf`, and A-6 says a
//! masked position is the semiring zero. The two coincide by construction, so a
//! masked shape and a zero-padded shape are the same shape and there is nothing
//! to reconcile --- which is what `CK-17` asserts, byte for byte.
//!
//! # Nothing here can overflow
//!
//! `⊗` is addition and `⊕` is `max`. A `max` does not grow with the depth, so
//! the accumulator has no depth term at all: [`trop_acc_bits`] is a function of
//! the element width alone, and `CA-04` asserts that it answers the same width
//! at depth one and at `2^MAX_K_BITS`. And `⊗` is absorbing at the semiring
//! zero --- `-inf ⊗ a` is `-inf`, not a wrap --- which is spelled here as *no
//! arithmetic is performed at all* when either operand is `-inf`, so the
//! checked profile has nothing to trap on (`CT-08`).

use crate::acc::{encode_i128_into, AccOf, Accumulator, EncodeFrom};
use crate::alphabet::{Alphabet, Bound, Element, Full, IntegerElement, ObservedBound};
use crate::policy::EncodeMode;

use bytemuck::TransparentWrapper;

/// An element of the tropical alphabet `E ∪ {-inf}`.
///
/// `-inf` is a *distinct value of this type*, not a reserved pattern inside
/// `E`. That matters twice. It keeps §5.2's promise that no representable value
/// of an element type is refused --- `i8::MIN` is a perfectly ordinary finite
/// tropical element here, exactly as it is an ordinary member of its own
/// alphabet in the ring family (R8). And it keeps the semiring zero *writable
/// by a caller*, which A-6 requires of a mask: a masked position is a value the
/// caller supplies, not a hole the library recognizes.
///
/// The discriminant is a separate field rather than a niche, so the type
/// carries padding and is deliberately not `Pod`. Nothing digests these as raw
/// bytes; the corpus digest reads the fields (`CA-02`).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(C)]
pub struct Trop<E> {
    /// Meaningful exactly when `finite`; held at `E::ZERO` otherwise, so that
    /// the derived equality is semiring equality and two `-inf`s compare equal
    /// whatever route produced them.
    value: E,
    /// `false` names the semiring zero.
    finite: bool,
}

impl<E: Element> Trop<E> {
    /// The semiring zero: `-inf`. The additive identity of `max`, the pad, and
    /// the mask --- one value, three readings (A-6, `CK-17`).
    ///
    /// This is the one item that needs `E: Element`, because it needs a value to
    /// park in the unread field. Everything below is available for any `E`, so a
    /// reader of a tropical slice never has to prove membership of a trait to
    /// ask what it is holding.
    pub const NEG_INF: Self = Self {
        value: E::ZERO,
        finite: false,
    };
}

impl<E> Trop<E> {
    /// A finite tropical element.
    pub const fn finite(value: E) -> Self {
        Self {
            value,
            finite: true,
        }
    }

    /// The finite value, or `None` at the semiring zero.
    pub fn get(self) -> Option<E>
    where
        E: Copy,
    {
        if self.finite {
            Some(self.value)
        } else {
            None
        }
    }

    /// Is this a finite element?
    pub const fn is_finite(&self) -> bool {
        self.finite
    }
}

impl<E: Element> Default for Trop<E> {
    /// The semiring zero, so that a default-filled buffer is a masked one and
    /// agrees with [`Element::ZERO`].
    fn default() -> Self {
        Self::NEG_INF
    }
}

impl<E: IntegerElement> Trop<E> {
    /// The magnitude, for a bound declaration.
    ///
    /// Zero at the semiring zero. That is *derived*, not stipulated:
    /// [`Alphabet::ZERO`] is `E::ZERO`, every bound must admit it, and
    /// `Bnd<0>` is a legitimate declaration --- so the semiring zero's
    /// magnitude has to be zero for the same reason the ring's `0` has zero
    /// magnitude, and for no other.
    pub fn magnitude(self) -> u128 {
        match self.get() {
            Some(v) => v.magnitude(),
            None => 0,
        }
    }
}

/// The exact accumulator of a `(max, +)` reduction, `W` bits wide.
///
/// `W` is [`trop_acc_bits`] at the element type, resolved at compile time from
/// the element type alone --- there is no depth term, so no `k` enters it.
///
/// The register is an `i128` and `i128::MIN` names `-inf`. That is not a
/// sentinel smuggled back in: it is *derived*. Every finite value this register
/// can hold has magnitude at most `2^E::BITS`, which for the widest element
/// family is `2^64`, so `i128::MIN` is unreachable as a finite accumulation and
/// naming it costs nothing and excludes nothing. The field is private, so the
/// derivation is the only thing that has to hold.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(transparent)]
pub struct TropAcc<const W: u32> {
    /// `i128::MIN` names `-inf`; every other value is an exact finite sum.
    v: i128,
}

impl<const W: u32> TropAcc<W> {
    /// The semiring zero, and the identity of [`Accumulator::combine`].
    pub const NEG_INF: Self = Self { v: i128::MIN };

    /// The largest magnitude a finite value of this accumulator may hold.
    ///
    /// `2^W`, which is one bit more than the register's derived width allows
    /// for --- the width is `sign + log2(2B) + 1` and this is the magnitude that
    /// last term rounds up to. Every value the library's own operations produce
    /// is inside it: a `⊗` of two alphabet elements reaches `2^BITS`, a
    /// recentred difference reaches `2^BITS`, and `⊕` selects rather than sums,
    /// so nothing accumulates past either.
    pub const DOMAIN: i128 = if W >= 127 { i128::MAX } else { 1i128 << W };

    /// An exact finite value.
    ///
    /// # Domain
    ///
    /// `|v| <= `[`TropAcc::DOMAIN`]. That is not a limitation on the *operation*
    /// --- it is the register's own width, the same declaration
    /// `Alphabet::new` checks at the alphabet boundary and the same one
    /// [`crate::dot_ref`]'s narrow run relies on. Outside it the value is not a
    /// tropical accumulation of anything, and two things go wrong that are worth
    /// naming rather than discovering: `v == i128::MIN` reads back as the
    /// semiring zero, and a later `⊗` or recentring can leave the register.
    ///
    /// Spelled as a `debug_assert!` for exactly the reason `dot_ref`'s bound
    /// check is: it costs nothing in release, and under the `checked` profile
    /// --- which `CT-02` and `CT-08` run over the whole corpus --- a violated
    /// declaration is a loud failure instead of a quietly wrong integer. No
    /// input a caller can hand *the library* reaches it; only hand-building an
    /// accumulator out of range does.
    pub fn of(v: i128) -> Self {
        debug_assert!(
            v.unsigned_abs() <= Self::DOMAIN.unsigned_abs(),
            "a tropical accumulator holds a value of at most 2^W; this one is outside \
             the register the element type derived"
        );
        Self { v }
    }

    /// The exact value, or `None` at the semiring zero.
    pub const fn get(self) -> Option<i128> {
        if self.v == i128::MIN {
            None
        } else {
            Some(self.v)
        }
    }

    /// `self := self ⊕ v`, for a finite `v`.
    ///
    /// One comparison. No addition, no carry, no growth --- which is the whole
    /// reason this accumulator has no depth term.
    pub fn offer(&mut self, v: i128) {
        if v > self.v {
            self.v = v;
        }
    }
}

impl<const W: u32> Accumulator for TropAcc<W> {
    const ZERO: Self = Self::NEG_INF;
    const BITS: u32 = W;

    /// `⊕`, which is `max`.
    ///
    /// Associative, commutative, **and idempotent** --- the third is what the
    /// ring's `+` does not have, and `CK-16` asserts the law holds precisely
    /// here and fails precisely there. Order independence follows from the
    /// first two exactly as it does for the ring, so the tile partition and the
    /// completion order are invisible in the result (`CD-02`).
    fn combine(self, other: Self) -> Self {
        Self {
            v: if self.v >= other.v { self.v } else { other.v },
        }
    }
}

/// An accumulator whose `⊕` is a *selection*: the value it holds is one of the
/// values offered to it, and can be read back exactly.
///
/// This is what makes a witness possible at all. A ring accumulator has no such
/// trait and cannot: its value is a sum of every contribution, so no single
/// contribution is "the" one. A selection's is, and D-6 says which one when
/// several tie --- the smallest index.
///
/// Kept separate from [`Accumulator`] for the reason
/// `uor_matmul_gemm::epilogue::ScaleExact` is: it is selection machinery, not
/// accumulation machinery, and no traversal in the ring family can name it.
pub trait TropicalValue: Accumulator {
    /// The exact value, or `None` at the semiring zero.
    fn value(self) -> Option<i128>;

    /// The accumulator holding exactly that value.
    fn of_value(v: Option<i128>) -> Self;
}

impl<const W: u32> TropicalValue for TropAcc<W> {
    fn value(self) -> Option<i128> {
        self.get()
    }

    fn of_value(v: Option<i128>) -> Self {
        match v {
            Some(x) => Self::of(x),
            None => Self::NEG_INF,
        }
    }
}

/// Bits sufficient for any `(max, +)` accumulation, at any depth:
///
/// ```text
/// sign + log2(B_a + B_w) + 1
/// ```
///
/// Compare [`crate::bounds::acc_bits`], whose first term after the sign is
/// `MAX_K_BITS`. **There is no such term here**, and its absence is the
/// arithmetic content of the claim that a selection does not grow with the
/// reduction it selects over: `max_p (a_p + b_p)` is bounded by `max_p a_p +
/// max_p b_p` whatever `k` is. `CA-04` is the gate.
///
/// The largest magnitude of an element is `2^(BITS - 1)`, so `⊗` reaches
/// `2^BITS`, whose *representation* needs `BITS + 1` bits --- one more than its
/// logarithm, because the bound is attained. With the sign that is `BITS + 2`.
///
/// # Examples
///
/// ```
/// # use uor_matmul_core::{trop_acc_bits, Trop};
/// // Ten bits, where the ring's `i8` accumulator needs seventy-nine.
/// assert_eq!(trop_acc_bits::<Trop<i8>>(), 10);
/// ```
pub const fn trop_acc_bits<E: Element>() -> u32 {
    1 + (E::BITS + 1)
}

/// Implement the tropical element family for one signed machine integer.
///
/// Every impl is the same three facts --- the width, the semiring zero, and the
/// one arithmetic primitive --- exactly as [`crate::alphabet::Element`]'s ring
/// impls are, which is why adding this family is impls and not a branch
/// anywhere else.
macro_rules! impl_tropical_element {
    ($($t:ty),* $(,)?) => { $(
        impl Element for Trop<$t> {
            // `trop_acc_bits::<Trop<$t>>()`, written as the arithmetic rather
            // than as the call because a const-generic argument may not name a
            // generic parameter on stable. `CA-04` asserts the two agree.
            type Acc = TropAcc<{ 2 + <$t>::BITS }>;
            const BITS: u32 = <$t>::BITS;

            /// `-inf`: the additive identity, the pad, and the mask (`CK-17`).
            const ZERO: Self = Self::NEG_INF;

            fn mac(acc: &mut Self::Acc, a: Self, w: Self) {
                // `⊗` is `+` and `⊕` is `max`. Both operands widen to `i128`
                // without loss, so the sum is exact for every pair the type can
                // express, and `offer` is a comparison.
                //
                // The `-inf` arms perform **no arithmetic at all**. That is the
                // absorbing law `-inf ⊗ a = -inf` spelled so that there is
                // nothing for the checked profile to trap on: a sentinel inside
                // `$t` would have made this `$t::MIN + a`, which overflows, and
                // C6 forbids erroring on a valid input (`CT-08`).
                if let (Some(x), Some(y)) = (a.get(), w.get()) {
                    acc.offer(x as i128 + y as i128);
                }
            }

            // A `max` has no narrow-register question: the accumulator is
            // already ten bits at `i8`, and nothing about the answer or the
            // guarantee depends on the register (R13).
            const HAS_NARROW: bool = false;

            fn combine_narrow(acc: Self::Acc, _narrow: i64) -> Self::Acc {
                acc
            }
        }

        impl<const W: u32> EncodeFrom<TropAcc<W>> for Trop<$t> {
            fn encode_from(acc: TropAcc<W>, mode: EncodeMode) -> Self {
                match acc.get() {
                    // The semiring zero encodes to the semiring zero, under
                    // every mode. There is nothing to round and nothing to
                    // saturate: `-inf` is exactly representable in the output
                    // alphabet because the output alphabet is a tropical one.
                    None => Self::NEG_INF,
                    // The **same** single encode step the ring family runs, at
                    // a different output range. Not a second method (R13).
                    Some(v) => Self::finite(
                        encode_i128_into(v, <$t>::MIN as i128 + 1, <$t>::MAX as i128 - 1, mode)
                            as $t,
                    ),
                }
            }
        }
    )* };
}

impl_tropical_element!(i8, i16, i32, i64);

/// The total entry point, and the tropical twin of
/// [`crate::alphabet::as_alphabet_full`].
///
/// Infallible for every slice, including one that is entirely `-inf`.
/// Zero-copy, allocation-free, and with no validating pass: `Full<E>` admits
/// every value of `E`, and the semiring zero has magnitude zero, so there is
/// nothing to check.
pub fn as_alphabet_tropical<E: IntegerElement>(xs: &[Trop<E>]) -> &[Alphabet<Trop<E>, Full<E>>]
where
    Trop<E>: Element,
{
    Alphabet::wrap_slice(xs)
}

/// The declared-bound tropical entry point, and the twin of
/// [`crate::alphabet::as_alphabet`].
///
/// Returns the observed bound rather than an error when the declaration is
/// wrong, because being wrong about your own data is not a reason for a matmul
/// to fail (C6). A slice of `-inf` satisfies every declaration, including
/// `Bnd<0>`.
pub fn as_alphabet_tropical_bounded<E: IntegerElement, Bd: Bound>(
    xs: &[Trop<E>],
) -> Result<&[Alphabet<Trop<E>, Bd>], ObservedBound>
where
    Trop<E>: Element,
{
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

/// The tropical reference reduction: `max_p (a_p ⊗ w_p)`.
///
/// The twin of [`crate::dot::dot_ref`], and like it never optimized and never
/// removed (R6): every tropical instruction sequence in `uor-matmul-kernels` is
/// differentially tested against this one, lane for lane (`CB-13`).
///
/// Zip-truncating, exactly as the ring reference is. Total: no input of any
/// length or magnitude can make it fail, because there is no arithmetic in it
/// that can leave the register.
///
/// # Examples
///
/// ```
/// # use uor_matmul_core::{as_alphabet_tropical, dot_tropical_ref, Trop};
/// let f = Trop::finite;
/// let (av, wv, mv) = (
///     [f(1i8), f(2), f(3)],
///     [f(4i8), f(5), Trop::NEG_INF],
///     [Trop::<i8>::NEG_INF; 3],
/// );
/// let a = as_alphabet_tropical(&av);
/// let w = as_alphabet_tropical(&wv);
/// // max(1+4, 2+5, 3 + -inf) = 7
/// assert_eq!(dot_tropical_ref(a, w).get(), Some(7));
///
/// // A masked operand contributes nothing, so an all-masked reduction is the
/// // semiring zero and not a zero.
/// let masked = as_alphabet_tropical(&mv);
/// assert_eq!(dot_tropical_ref(a, masked).get(), None);
/// ```
pub fn dot_tropical_ref<E, Bd: Bound>(
    a: &[Alphabet<Trop<E>, Bd>],
    w: &[Alphabet<Trop<E>, Bd>],
) -> AccOf<Trop<E>>
where
    Trop<E>: Element,
{
    let k = a.len().min(w.len());
    let mut acc = <AccOf<Trop<E>> as Accumulator>::ZERO;
    for i in 0..k {
        <Trop<E> as Element>::mac(&mut acc, a[i].get(), w[i].get());
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generated;

    /// `CA-04`: the tropical accumulator's width is independent of `k`.
    ///
    /// The ring's [`crate::bounds::acc_bits`] carries a `MAX_K_BITS` term and
    /// this one does not, so the two-sided statement is available: the tropical
    /// width is the same number at every depth, and the ring width is not.
    #[test]
    fn the_tropical_accumulator_has_no_depth_term_ca_04() {
        // The derived width, and the width the type actually resolved to.
        assert_eq!(trop_acc_bits::<Trop<i8>>(), 10);
        assert_eq!(<AccOf<Trop<i8>> as Accumulator>::BITS, 10);
        assert_eq!(<AccOf<Trop<i16>> as Accumulator>::BITS, 18);
        assert_eq!(<AccOf<Trop<i32>> as Accumulator>::BITS, 34);
        assert_eq!(<AccOf<Trop<i64>> as Accumulator>::BITS, 66);

        // No `k` enters the derivation: `trop_acc_bits` takes none. What a test
        // can pin is that the ring's width *does* move with `MAX_K_BITS` while
        // this one is below it entirely --- the ring's `i8` accumulator is
        // seventy-nine bits for a depth term of sixty-four, and this is ten.
        assert_eq!(crate::bounds::acc_bits::<i8>(), 79);
        assert_eq!(generated::MAX_K_BITS, 64);
        assert!(
            trop_acc_bits::<Trop<i8>>() < generated::MAX_K_BITS,
            "a width with a depth term cannot be narrower than the term"
        );

        // And the register holds the extreme: the deepest reduction the machine
        // can address, at the widest magnitudes, is still one comparison.
        let extreme = Trop::finite(i8::MIN);
        let mut acc = <AccOf<Trop<i8>> as Accumulator>::ZERO;
        for _ in 0..1000 {
            <Trop<i8> as Element>::mac(&mut acc, extreme, extreme);
        }
        assert_eq!(acc.get(), Some(-256));
        // Which is exactly the bound the width was derived against.
        assert!((-256i128).unsigned_abs() <= 1u128 << <Trop<i8> as Element>::BITS);
    }

    /// `CK-16`: idempotence holds precisely at the tropical instance and fails
    /// precisely at the ring one.
    ///
    /// Two-sided by construction --- a one-sided version would pass for a
    /// `combine` that was accidentally `max` in both families.
    #[test]
    fn idempotence_holds_at_the_tropical_instance_and_fails_at_the_ring_ck_16() {
        let t = TropAcc::<10>::of(7);
        assert_eq!(t.combine(t), t, "max is idempotent");

        let r: i128 = 7;
        assert_ne!(r.combine(r), r, "the ring's sum is not");

        // The other two laws hold at both, which is what makes the partition
        // invisible in either (`CD-02`).
        let (x, y, z) = (
            TropAcc::<10>::of(3),
            TropAcc::<10>::of(-9),
            TropAcc::<10>::of(5),
        );
        assert_eq!(x.combine(y), y.combine(x));
        assert_eq!(x.combine(y).combine(z), x.combine(y.combine(z)));
        // And the identity is the semiring zero at both.
        assert_eq!(x.combine(TropAcc::<10>::NEG_INF), x);
        assert_eq!(7i128.combine(<i128 as Accumulator>::ZERO), 7);
    }

    /// `CT-08`: no tropical operation overflows, including a masked lane.
    ///
    /// This test's teeth are under the `checked` profile, where every
    /// accumulator operation is checked and any overflow panics. A sentinel
    /// inside the element type would fail here at the first masked product ---
    /// `i8::MIN + i8::MIN` is not an `i8` --- which is the second reason
    /// [`Trop`] is a distinct type.
    #[test]
    fn no_tropical_operation_overflows_ct_08() {
        let extremes = [
            Trop::finite(i8::MIN),
            Trop::finite(i8::MAX),
            Trop::finite(0i8),
            Trop::NEG_INF,
        ];
        for a in extremes {
            for w in extremes {
                let mut acc = <AccOf<Trop<i8>> as Accumulator>::ZERO;
                // A masked lane, then a live one, then a masked one again: the
                // absorbing law has to hold in every position, not only first.
                <Trop<i8> as Element>::mac(&mut acc, Trop::NEG_INF, w);
                <Trop<i8> as Element>::mac(&mut acc, a, w);
                <Trop<i8> as Element>::mac(&mut acc, a, Trop::NEG_INF);
                match (a.get(), w.get()) {
                    (Some(x), Some(y)) => assert_eq!(acc.get(), Some(x as i128 + y as i128)),
                    // Every contribution was absorbed, so the reduction is the
                    // semiring zero --- not a zero.
                    _ => assert_eq!(acc.get(), None),
                }
            }
        }

        // The widest family, at its extremes, where a naive `i64` register
        // would have wrapped.
        let mut acc = <AccOf<Trop<i64>> as Accumulator>::ZERO;
        <Trop<i64> as Element>::mac(&mut acc, Trop::finite(i64::MIN), Trop::finite(i64::MIN));
        assert_eq!(acc.get(), Some(2 * i64::MIN as i128));
    }

    /// `CK-17`: the semiring zero is simultaneously the pad and the mask.
    ///
    /// The library pads with [`Element::ZERO`]; A-6 masks with the semiring
    /// zero. Here they are the same value, so there is nothing to reconcile.
    #[test]
    fn the_semiring_zero_is_the_pad_and_the_mask_ck_17() {
        assert_eq!(<Trop<i8> as Element>::ZERO, Trop::<i8>::NEG_INF);
        assert_eq!(Trop::<i8>::default(), Trop::<i8>::NEG_INF);
        assert_eq!(Alphabet::<Trop<i8>, Full<i8>>::ZERO.get(), Trop::NEG_INF);

        // A masked position and a padded position reduce alike.
        let f = Trop::finite;
        let (lv, mv, sv) = (
            [f(1i8), f(2), f(3)],
            [f(4i8), Trop::NEG_INF, f(6)],
            [f(4i8), Trop::NEG_INF],
        );
        let live = as_alphabet_tropical(&lv);
        let masked = as_alphabet_tropical(&mv);
        // Padding is truncation to the shorter operand, which is the same as
        // masking the tail.
        let short = as_alphabet_tropical(&sv);
        assert_eq!(dot_tropical_ref(live, masked).get(), Some(9));
        assert_eq!(dot_tropical_ref(&live[..2], short).get(), Some(5));
    }

    /// The semiring zero has magnitude zero, so it satisfies every declaration
    /// --- including the narrowest one a caller can write.
    #[test]
    fn the_semiring_zero_satisfies_every_declaration_ct_04() {
        use crate::alphabet::Bnd;
        let masked = [Trop::<i8>::NEG_INF; 4];
        assert!(as_alphabet_tropical_bounded::<i8, Bnd<0>>(&masked).is_ok());
        assert_eq!(
            as_alphabet_tropical_bounded::<i8, Bnd<3>>(&[Trop::finite(7i8)]).unwrap_err(),
            ObservedBound {
                declared: 3,
                observed: 7
            }
        );
        // And the total entry point admits it, as it admits everything.
        assert_eq!(as_alphabet_tropical(&[Trop::finite(i8::MIN)]).len(), 1);
    }

    /// The wrap is zero-copy, as the ring's is (`CA-01`).
    #[test]
    fn the_tropical_wrap_is_zero_copy_ca_01() {
        let data = [Trop::finite(1i32), Trop::finite(2)];
        let a = as_alphabet_tropical(&data);
        assert_eq!(a.as_ptr() as usize, data.as_ptr() as usize);
        assert_eq!(
            core::mem::size_of::<Alphabet<Trop<i32>, Full<i32>>>(),
            core::mem::size_of::<Trop<i32>>()
        );
    }
}
