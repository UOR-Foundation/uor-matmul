//! The two gauge sections the census names (G-2, G-5).
//!
//! A *gauge* is a group acting on values that the downstream answer does not
//! see. A *section* picks one representative per orbit, canonically. There are
//! two in the formalization, and both are already this library's kind of
//! operation --- exact, total, and free of any rounding step:
//!
//! - **G-2, the shift gauge.** `(max, +)` is invariant under adding a constant
//!   to every element of a reduction, so the orbit of a vector under `⊗ c` is
//!   what the answer cannot distinguish. [`recenter`] is the canonical section:
//!   subtract the maximum, so that the representative's maximum is exactly
//!   zero. Classically this is the max-subtraction a softmax performs for
//!   numerical stability; here nothing is unstable and the section is a
//!   statement about *identity*, not about error.
//!
//! - **G-5, the dyadic gauge.** Scaling by a power of two is the orbit;
//!   [`dyadic`] is the section. It is a *placement*, not a division: the
//!   complete accumulator's [`Complete::add_scaled`] takes a **signed**
//!   exponent, so placing a value at `2^-k` is exactly `x / 2^k` in the
//!   fixed-point register, with no division instruction and nothing rounded.
//!   D-8's census has no division in it, and this is why it does not need one.
//!
//! # Neither section can round
//!
//! [`recenter`]'s output is a difference of two alphabet elements, which is
//! exactly the magnitude [`crate::trop_acc_bits`] was derived against, so it
//! lands in the accumulator with a bit to spare. [`dyadic`]'s output is exact
//! wherever the register spans the exponent, which is a compile-time property
//! of the register and is stated by [`dyadic_exact`] rather than assumed.

use crate::acc::Complete;
use crate::alphabet::Element;
use crate::tropical::{Trop, TropicalValue};

/// Lift tropical alphabet elements into the accumulator's width.
///
/// The width is where a section can be taken exactly, and lifting is the
/// zero-cost half of that: the accumulator holding `x` is
/// `TropicalValue::of_value`, which is no arithmetic at all.
///
/// Writes `min(xs.len(), out.len())` elements and returns how many, in the
/// zip-truncating style [`crate::dot_ref`] uses. Allocates nothing; the buffer
/// is the caller's (R7).
pub fn lift_tropical<E>(xs: &[Trop<E>], out: &mut [<Trop<E> as Element>::Acc]) -> usize
where
    Trop<E>: Element,
    <Trop<E> as Element>::Acc: TropicalValue,
    E: Into<i128> + Copy,
{
    let n = xs.len().min(out.len());
    for i in 0..n {
        out[i] = <Trop<E> as Element>::Acc::of_value(xs[i].get().map(Into::into));
    }
    n
}

/// The canonical section of the shift gauge (G-2): `x ↦ x ⊗ (⊕ x)^{-1}`.
///
/// Subtracts the maximum from every element in place, and returns the shift
/// that was removed, so the original is recoverable and nothing is lost. After
/// it, `⊕ xs` is exactly the multiplicative identity --- zero --- which is the
/// canonical representative of the orbit.
///
/// # Totality
///
/// Every element is at the semiring zero: the orbit is a single point, there is
/// no shift to remove, and the slice is returned unchanged with the semiring
/// zero as the shift. An empty slice is the same answer. Neither is a special
/// case; both fall out of `⊕` over an empty or all-identity set.
///
/// # Exactness, and the one declaration it inherits
///
/// Every value the library produces for this register has magnitude at most
/// `2^BITS`: a `⊗` of two alphabet elements reaches exactly that, and `⊕`
/// selects rather than sums, so a reduction of any depth reaches no further. A
/// difference of two such values is therefore at most `2^(BITS+1)`, and the
/// register holds `2^(BITS+2)` --- so the section is exact for every operand the
/// library can produce, at every bound, with nothing to declare and nothing to
/// check (R8), and the result is itself inside
/// [`crate::TropAcc::DOMAIN`] with a bit to spare.
///
/// What it does *not* do is invent a domain of its own. Handed an accumulator
/// built by hand outside `DOMAIN` --- which only [`crate::TropAcc::of`] can do,
/// and which its own `debug_assert!` reports --- the difference can leave the
/// register, exactly as a value past a declared alphabet bound makes
/// [`crate::dot_ref`]'s narrow run wrong. The declaration is checked where it is
/// made, once, and everything downstream inherits it rather than re-checking it.
///
/// Applying it again subtracts zero, so it is idempotent and `CD-26` says so.
///
/// # Examples
///
/// ```
/// # use uor_matmul_core::{recenter, TropAcc, TropicalValue};
/// let mut xs = [TropAcc::<10>::of(5), TropAcc::of(-3), TropAcc::NEG_INF];
/// let shift = recenter(&mut xs);
/// assert_eq!(shift.value(), Some(5));
/// assert_eq!(xs[0].value(), Some(0));
/// assert_eq!(xs[1].value(), Some(-8));
/// // The semiring zero is fixed by the whole gauge: `-inf ⊗ c` is `-inf`.
/// assert_eq!(xs[2].value(), None);
/// ```
pub fn recenter<A: TropicalValue>(xs: &mut [A]) -> A {
    // `⊕` over the slice, by the accumulator's own operation. There is no
    // second maximum in the library and this is not one.
    let mut top = A::ZERO;
    for x in xs.iter() {
        top = top.combine(*x);
    }
    let Some(shift) = top.value() else {
        // Every element is the semiring zero. The orbit is a point.
        return top;
    };
    for x in xs.iter_mut() {
        // `-inf ⊗ c = -inf`: the semiring zero is fixed by every shift, which is
        // what makes the section well defined on a masked vector (A-6).
        if let Some(v) = x.value() {
            *x = A::of_value(Some(v - shift));
        }
    }
    top
}

/// The canonical section of the dyadic gauge (G-5): `v ↦ v · 2^-k`, exactly.
///
/// A *placement*, not a division. [`Complete::add_scaled`] takes a signed
/// exponent and the register is fixed-point, so writing `v` at exponent `-k`
/// **is** `v / 2^k`, with every bit of `v` preserved wherever the register
/// spans the exponent. No division opcode is emitted and none is needed, which
/// is why D-8's census has no division in it (`CU-10`).
///
/// [`dyadic_exact`] is the compile-time question of whether this register spans
/// the exponent; `CD-27` asserts the value.
pub fn dyadic<const L: usize, const MIN_EXP: i32>(v: i128, k: u32) -> Complete<L, MIN_EXP> {
    let mut out = Complete::<L, MIN_EXP>::ZERO;
    // The negation is the whole of the "division". It is taken in `i64` and
    // then narrowed, because `-(k as i32)` is not a total function on `u32`: at
    // `k >= 2^31` the cast is `i32::MIN` and negating it overflows, which the
    // checked profile traps and C6 forbids. Narrowing to `i32::MIN` instead is
    // not a clamp that hides anything --- an exponent that far below the
    // register's floor places no bit at all, which is what `add_scaled` does
    // there and what [`dyadic_exact`] reports as `false`.
    let exponent = i32::try_from(-(k as i64)).unwrap_or(i32::MIN);
    // The magnitude and the sign are handed over separately because the
    // register's primitive takes them that way; nothing here is an
    // approximation of a quotient.
    out.add_scaled(v.unsigned_abs(), exponent, v < 0);
    out
}

/// Does a `Complete<L, MIN_EXP>` hold `v · 2^-k` exactly?
///
/// Two questions, both about the *register* and neither about the data: the
/// exponent must not fall below the register's floor, and the value must not
/// reach its ceiling. A `false` is not a failure --- it says the section needs a
/// wider register, and the caller picks one, exactly as the accumulator width
/// is picked from the element type.
pub const fn dyadic_exact<const L: usize, const MIN_EXP: i32>(bits: u32, k: u32) -> bool {
    // The lowest exponent the placement reaches, against the register's floor.
    let lowest = -(k as i64);
    if lowest < MIN_EXP as i64 {
        return false;
    }
    // The highest bit it reaches, against the register's width.
    let highest = lowest + bits as i64;
    (highest - MIN_EXP as i64) < (L as i64) * 64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acc::Accumulator;
    use crate::tropical::TropAcc;

    /// `CD-26`: recentring is the canonical section --- gauge-invariant,
    /// idempotent, and its representative's maximum is exactly zero.
    #[test]
    fn recentring_is_the_canonical_section_cd_26() {
        let base = [
            TropAcc::<10>::of(7),
            TropAcc::of(-3),
            TropAcc::of(0),
            TropAcc::NEG_INF,
            TropAcc::of(7),
        ];

        // The representative: maximum exactly zero.
        let mut xs = base;
        let shift = recenter(&mut xs);
        assert_eq!(shift.value(), Some(7));
        let top = xs.iter().fold(TropAcc::<10>::ZERO, |a, &b| a.combine(b));
        assert_eq!(
            top.value(),
            Some(0),
            "the section's maximum is the identity"
        );

        // Idempotent: applying it again removes nothing.
        let once = xs;
        let shift2 = recenter(&mut xs);
        assert_eq!(shift2.value(), Some(0));
        assert_eq!(xs, once, "the section is a section");

        // Gauge invariant: `recenter(s ⊗ c) == recenter(s)` for every shift the
        // register holds, including one that moves the whole vector negative.
        for c in [-128i128, -1, 0, 1, 5, 120] {
            let mut shifted = base;
            for x in shifted.iter_mut() {
                if let Some(v) = x.value() {
                    *x = TropAcc::of(v + c);
                }
            }
            recenter(&mut shifted);
            assert_eq!(shifted, once, "the gauge orbit at c = {c}");
        }

        // Degeneracies, and both are answers: an all-identity vector has a
        // single-point orbit, and an empty one has nothing to shift.
        let mut masked = [TropAcc::<10>::NEG_INF; 3];
        assert_eq!(recenter(&mut masked).value(), None);
        assert_eq!(masked, [TropAcc::<10>::NEG_INF; 3]);
        let mut empty: [TropAcc<10>; 0] = [];
        assert_eq!(recenter(&mut empty).value(), None);
    }

    /// `CD-26`, the lift half: an alphabet element and its lifted accumulator
    /// name the same value, so the section may be taken in the width where it
    /// is exact.
    #[test]
    fn the_lift_preserves_every_element_cd_26() {
        let xs = [
            Trop::finite(i8::MIN),
            Trop::finite(0i8),
            Trop::finite(i8::MAX),
            Trop::NEG_INF,
        ];
        let mut out = [<Trop<i8> as Element>::Acc::ZERO; 4];
        assert_eq!(lift_tropical(&xs, &mut out), 4);
        assert_eq!(out[0].value(), Some(-128));
        assert_eq!(out[1].value(), Some(0));
        assert_eq!(out[2].value(), Some(127));
        assert_eq!(out[3].value(), None);

        // And the widest difference the section can produce lands inside the
        // derived width: `127 - (-128)` is `255`, under `2^8`.
        let mut widest = out;
        recenter(&mut widest);
        assert_eq!(widest[0].value(), Some(-255));
        // A `const` block, because this is the width derivation rather than a
        // property of the values above: if the section could ever produce a
        // difference the accumulator does not hold, that should stop the
        // compile and not wait for a test.
        const { assert!(255u128 <= 1u128 << <Trop<i8> as Element>::BITS) };

        // Zip-truncating, exactly as `dot_ref` is.
        let mut short = [<Trop<i8> as Element>::Acc::ZERO; 2];
        assert_eq!(lift_tropical(&xs, &mut short), 2);
    }

    /// `CD-27`: the dyadic section is exact and equals the arithmetic shift at
    /// every `k` where the shift is exact.
    ///
    /// Two-sided: where `2^k` divides `v` the section equals `v >> k` exactly,
    /// and where it does not the section still holds the *rational* --- which is
    /// the case a shift cannot express and is why the section is a placement.
    #[test]
    fn the_dyadic_section_is_exact_cd_27() {
        // `f32`'s register, whose floor is far below any `k` here.
        type Reg = Complete<10, -298>;

        for v in [1i128, 3, 255, -255, 1 << 40, -(1 << 40), 12345] {
            for k in 0u32..24 {
                let placed: Reg = dyadic(v, k);
                assert!(dyadic_exact::<10, -298>(64, k), "the register spans 2^-{k}");

                // The bit that `v`'s bit 0 landed on is `-k` relative to the
                // register's floor, so the value is `v * 2^-k` exactly. Reading
                // it back as the integer `v` shifted up by `k` from the floor
                // is the same statement in the other direction.
                let expect: Reg = {
                    let mut r = Reg::ZERO;
                    r.add_scaled(v.unsigned_abs(), -(k as i32), v < 0);
                    r
                };
                assert_eq!(placed, expect);

                // Against the shift, where the shift is exact.
                if v % (1i128 << k) == 0 {
                    let shifted: Reg = dyadic(v >> k, 0);
                    assert_eq!(
                        placed, shifted,
                        "v = {v}, k = {k}: the section is the shift where the shift exists"
                    );
                } else {
                    // And where it is not, the two differ --- the section holds
                    // the fraction the shift discarded. Without this the row
                    // would pass for a section that truncated.
                    let truncated: Reg = dyadic(v >> k, 0);
                    assert_ne!(
                        placed, truncated,
                        "v = {v}, k = {k}: a truncating section must be visible"
                    );
                }
            }
        }

        // Scaling composes, which is what makes it a group action: placing at
        // `-a` then again at `-b` is placing at `-(a+b)`.
        let a: Reg = dyadic(3, 5);
        let b: Reg = dyadic(3 << 5, 10);
        assert_eq!(a, b, "2^-5 and 2^5 * 2^-10 name one point of the orbit");

        // The register question is about the register, and it answers `false`
        // below its own floor rather than rounding there.
        assert!(!dyadic_exact::<10, -298>(64, 299));
        assert!(dyadic_exact::<10, -298>(64, 298));
    }
}
