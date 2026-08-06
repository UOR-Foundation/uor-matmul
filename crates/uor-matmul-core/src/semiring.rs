//! The algebra an element family accumulates in, as a declaration (A-4, S-3).
//!
//! This module states no new capability. It exists so that a law can be written
//! **once and quantified over every instance**, instead of once per family ---
//! which is the difference between a gate that holds at an instance and a gate
//! that holds at the parameter (P-i).
//!
//! There is no branch anywhere in the library on which semiring an element type
//! is in, and this trait does not create one: nothing in a traversal reads it,
//! no public signature mentions it, and removing it would change no output byte.
//! What it changes is what a *test* can say. `CK-16` is
//!
//! > the semiring laws hold at every instance, and idempotence holds precisely
//! > at the tropical instance and fails precisely at the ring
//!
//! and the second half is only expressible if the two instances are values of
//! one parameter. Written as two separate tests it would be two claims that
//! happen to disagree, and a `combine` that was accidentally `max` in both
//! families would satisfy both.
//!
//! # Why the laws are declared and not probed
//!
//! [`Semiring::IDEMPOTENT`] is a `const`, not something [`laws_of`] discovers.
//! That is the point: the instance *declares* which algebra it is, the gate
//! *measures* which laws hold, and `CK-16` compares the two. A probe would
//! agree with itself no matter what the code did.

use core::marker::PhantomData;

use crate::acc::{AccOf, Accumulator};
use crate::alphabet::{Element, IntegerElement};
use crate::tropical::Trop;

/// An element family, together with the algebra it accumulates in.
///
/// Implemented by two zero-sized markers, [`Ring`] and [`Tropical`], rather
/// than by the element types themselves --- an element type belongs to exactly
/// one algebra, so a marker per algebra keeps the instance visible in the type
/// a test names.
pub trait Semiring {
    /// The element type this instance accumulates.
    type Elem: Element;

    /// What to call this algebra in a failure message.
    const NAME: &'static str;

    /// Is `⊕` idempotent?
    ///
    /// **Declared, not probed.** A selection's `⊕` is idempotent and a sum's is
    /// not, and that single law is the whole difference between the two halves
    /// of D-8's census. [`laws_of`] measures whether idempotence actually holds
    /// and `CK-16` compares the measurement to this declaration, in both
    /// directions.
    const IDEMPOTENT: bool;

    /// Three accumulator values, pairwise distinct and none of them the
    /// identity, at which the `⊕` laws are exercised.
    ///
    /// Supplied by the instance because a value that is interesting in one
    /// algebra is not in the other: the ring wants magnitudes that do not
    /// cancel, and the selection wants values that do not all tie.
    fn witnesses() -> [AccOf<Self::Elem>; 3];

    /// Three element values at which distributivity is exercised.
    ///
    /// Chosen by the instance so that [`Semiring::plus`] on them stays inside
    /// the element type. The law is about the algebra, not about a register,
    /// and asking the instance for its own witnesses is how the two are kept
    /// apart.
    fn elements() -> [Self::Elem; 3];

    /// `⊕` on elements.
    fn plus(a: Self::Elem, b: Self::Elem) -> Self::Elem;

    /// `⊗` on elements, via [`Element::mac`] --- *the one arithmetic
    /// primitive*. There is no second definition of `⊗` in the library and this
    /// is not one (R13).
    fn times(a: Self::Elem, w: Self::Elem) -> AccOf<Self::Elem> {
        let mut acc = <AccOf<Self::Elem> as Accumulator>::ZERO;
        Self::Elem::mac(&mut acc, a, w);
        acc
    }
}

/// Which laws were found to hold at an instance.
///
/// Plain `bool`s rather than assertions, so the caller decides what a `false`
/// means. `CK-16` wants three of them true at every instance and the fourth
/// true at exactly one, which an assertion inside this function could not say.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Laws {
    /// `(a ⊕ b) ⊕ c == a ⊕ (b ⊕ c)`. What makes the reduction schedule
    /// invisible (`CD-02`).
    pub associative: bool,
    /// `a ⊕ b == b ⊕ a`. What makes the tile partition invisible.
    pub commutative: bool,
    /// `a ⊕ 0 == a`, where `0` is [`Accumulator::ZERO`]. What makes a padded
    /// position exact, and --- in the selection half --- a masked one (`CK-17`).
    pub identity: bool,
    /// `a ⊕ a == a`. True of a selection, false of a sum.
    pub idempotent: bool,
    /// `(a ⊕ b) ⊗ c == (a ⊗ c) ⊕ (b ⊗ c)`. What makes the operation a matrix
    /// product at all: without it, a reduction is not a bilinear form and
    /// blocking it would not be a factorization.
    pub distributive: bool,
}

impl Laws {
    /// The three laws every semiring has, whichever one it is.
    pub const fn universal(&self) -> bool {
        self.associative && self.commutative && self.identity && self.distributive
    }
}

/// Measure which laws hold at an instance.
///
/// One body, every instance. Allocation-free and `const`-free of any host
/// question, so it runs identically on every target.
pub fn laws_of<S: Semiring>() -> Laws {
    let [a, b, c] = S::witnesses();
    let zero = <AccOf<S::Elem> as Accumulator>::ZERO;
    let [x, y, z] = S::elements();

    Laws {
        associative: a.combine(b).combine(c) == a.combine(b.combine(c)),
        commutative: a.combine(b) == b.combine(a),
        identity: a.combine(zero) == a && zero.combine(b) == b,
        idempotent: a.combine(a) == a && b.combine(b) == b && c.combine(c) == c,
        distributive: S::times(S::plus(x, y), z) == S::times(x, z).combine(S::times(y, z)),
    }
}

/// The ring instance: `⊕` is addition, `⊗` is multiplication.
///
/// Zero-sized. `E` is carried for the type-level fact and nothing is stored.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Ring<E>(PhantomData<E>);

/// The `(max, +)` instance: `⊕` is `max`, `⊗` is addition (A-4).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Tropical<E>(PhantomData<E>);

/// Declare both instances for one machine integer type.
///
/// The witnesses are small on purpose. A ring witness must not cancel to the
/// identity, and a selection witness must not tie --- and the values that
/// satisfy both readings are the ones a reader can check by eye.
macro_rules! impl_semirings_for {
    ($($t:ty),* $(,)?) => { $(
        impl Semiring for Ring<$t> {
            type Elem = $t;
            const NAME: &'static str = concat!("Ring<", stringify!($t), ">");
            // A sum is not idempotent: `7 + 7` is not `7`.
            const IDEMPOTENT: bool = false;

            fn witnesses() -> [AccOf<$t>; 3] {
                // `3`, `-9`, `5`, built from the family's own `⊗` rather than
                // written as accumulator literals --- `i64`'s accumulator is a
                // `Limbs<3>` and has no integer literal, and asking the
                // primitive is how the same three values are named at every
                // width without a per-width table.
                [
                    Self::times(1, 3),
                    Self::times(-1, 9),
                    Self::times(1, 5),
                ]
            }

            fn elements() -> [$t; 3] {
                // Small enough that `x + y` cannot leave the narrowest element
                // type this macro is instantiated at.
                [3, 5, 7]
            }

            fn plus(a: $t, b: $t) -> $t {
                <$t as IntegerElement>::add(a, b)
            }
        }

        impl Semiring for Tropical<$t> {
            type Elem = Trop<$t>;
            const NAME: &'static str = concat!("Tropical<", stringify!($t), ">");
            // A maximum is: `max(7, 7)` is `7`. This is the one law that
            // separates the two halves of the census.
            const IDEMPOTENT: bool = true;

            fn witnesses() -> [AccOf<Trop<$t>>; 3] {
                // The same three values, `3`, `-9` and `5`, reached through
                // this family's `⊗` --- which is addition.
                [
                    Self::times(Trop::finite(1), Trop::finite(2)),
                    Self::times(Trop::finite(-4), Trop::finite(-5)),
                    Self::times(Trop::finite(2), Trop::finite(3)),
                ]
            }

            fn elements() -> [Trop<$t>; 3] {
                [Trop::finite(3), Trop::finite(5), Trop::finite(7)]
            }

            fn plus(a: Trop<$t>, b: Trop<$t>) -> Trop<$t> {
                // `⊕` on elements is the same `max` the accumulator's own
                // `combine` is, read at the element width.
                match (a.get(), b.get()) {
                    (Some(x), Some(y)) => Trop::finite(if x >= y { x } else { y }),
                    (Some(x), None) => Trop::finite(x),
                    (None, Some(y)) => Trop::finite(y),
                    (None, None) => Trop::NEG_INF,
                }
            }
        }
    )* };
}

impl_semirings_for!(i8, i16, i32, i64);

#[cfg(test)]
mod tests {
    use super::*;

    /// `CK-16`: the semiring laws hold at **every** instance, and idempotence
    /// holds precisely where the instance declares it and nowhere else.
    ///
    /// One body, eight instances. The second half is two-sided by
    /// construction: `laws.idempotent == S::IDEMPOTENT` fails if a selection
    /// stops being idempotent *and* if a sum starts being one, so a `combine`
    /// that was accidentally `max` in both families cannot satisfy it.
    #[test]
    fn the_semiring_laws_hold_at_every_instance_ck_16() {
        fn check<S: Semiring>() {
            let laws = laws_of::<S>();
            assert!(
                laws.associative,
                "{}: `⊕` must be associative, or the reduction schedule is visible",
                S::NAME
            );
            assert!(
                laws.commutative,
                "{}: `⊕` must be commutative, or the tile partition is visible",
                S::NAME
            );
            assert!(
                laws.identity,
                "{}: the identity must be `Accumulator::ZERO`, or padding is not exact",
                S::NAME
            );
            assert!(
                laws.distributive,
                "{}: `⊗` must distribute over `⊕`, or blocking is not a factorization",
                S::NAME
            );
            assert!(
                laws.universal(),
                "{}: every instance has these four",
                S::NAME
            );
            assert_eq!(
                laws.idempotent,
                S::IDEMPOTENT,
                "{}: the measured idempotence must be the declared one, in both \
                 directions --- a selection that stopped being idempotent and a sum \
                 that started being one are the same failure",
                S::NAME
            );
        }

        check::<Ring<i8>>();
        check::<Ring<i16>>();
        check::<Ring<i32>>();
        check::<Ring<i64>>();
        check::<Tropical<i8>>();
        check::<Tropical<i16>>();
        check::<Tropical<i32>>();
        check::<Tropical<i64>>();

        // And the declaration itself separates the two halves of the census:
        // exactly one of the two instances at a given width is idempotent.
        // A `const` block rather than a runtime assertion, because these are
        // declarations --- if they ever agreed, the contradiction should stop
        // the compile rather than wait for a test to run.
        const { assert!(!<Ring<i8> as Semiring>::IDEMPOTENT) };
        const { assert!(<Tropical<i8> as Semiring>::IDEMPOTENT) };
    }
}
