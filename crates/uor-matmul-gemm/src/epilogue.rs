//! The single encode step, and what a caller may fold into it (§5.5, S9).
//!
//! An epilogue is the *only* place in the library where information can be
//! discarded, and it runs exactly once per output element. Everything upstream
//! of it is exact.
//!
//! Requantization is **encoding**: the inverse of the codec's decode, into a
//! coarser alphabet. It is exact integer throughout, and it never consults a
//! platform rounding mode.

use uor_matmul_core::{
    AccOf, Accumulator, Complete, Element, EncodeFrom, EncodeMode, FloatElement, IntegerElement,
    Limbs, Trop, TropAcc,
};

use core::marker::PhantomData;

/// What to do with a finished accumulator and the value already in `C`.
///
/// The trait takes the *exact* accumulator, not a narrowed one, so a user
/// epilogue sees the same value the library does and cannot be handed a
/// pre-rounded input.
pub trait Epilogue<E: Element, O> {
    /// Produce the output element.
    ///
    /// `prior` is the value already in `C`, and is `None` exactly when the
    /// driver did not read `C` --- which is the case whenever the epilogue
    /// declares [`Epilogue::READS_C`] false, so that an uninitialised output
    /// buffer is admissible (`CS-04`).
    fn finish(&self, acc: AccOf<E>, prior: Option<O>, mode: EncodeMode) -> O;

    /// Does this epilogue read the existing contents of `C`?
    ///
    /// `false` lets the driver skip the read entirely, which is what makes
    /// `beta = 0` mean *overwrite* rather than *multiply by zero* --- the
    /// difference matters when `C` holds uninitialised memory or a signalling
    /// pattern. It is a method rather than a const because for [`Linear`] the
    /// answer is a property of `beta`, which is a value.
    fn reads_c(&self) -> bool {
        true
    }
}

/// `C := alpha * A*B + beta * C`, the BLAS-shaped epilogue.
///
/// `alpha` and `beta` are exact integer scalars. They are applied inside the
/// single encode step, in the accumulator's width, so no intermediate is ever
/// rounded and the whole expression is evaluated exactly whenever its value is
/// representable in `O`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Linear {
    /// Scales the product.
    pub alpha: i64,
    /// Scales the value already in `C`.
    pub beta: i64,
}

impl Linear {
    /// `C := A*B`, overwriting. The common case, and the one that never reads
    /// the output buffer.
    pub const OVERWRITE: Self = Self { alpha: 1, beta: 0 };

    /// `C := A*B + C`.
    pub const ACCUMULATE: Self = Self { alpha: 1, beta: 1 };
}

impl<E, O> Epilogue<E, O> for Linear
where
    E: Element,
    O: EncodeFrom<AccOf<E>>,
    AccOf<E>: ScaleExact + AbsorbPrior<O>,
{
    fn reads_c(&self) -> bool {
        // `beta == 0` overwrites `C` without reading it, which is what makes an
        // uninitialised output buffer admissible (`CS-04`).
        self.beta != 0
    }

    fn finish(&self, acc: AccOf<E>, prior: Option<O>, mode: EncodeMode) -> O {
        let scaled = acc.scale_exact(self.alpha);
        let total = match prior {
            // `beta == 0` contributes nothing and the driver did not read `C`.
            None => scaled,
            Some(c) => {
                if self.beta == 0 {
                    scaled
                } else {
                    scaled.combine(AccOf::<E>::of_prior(c).scale_exact(self.beta))
                }
            }
        };
        O::encode_from(total, mode)
    }
}

/// `C := A*B + bias[j]`, the epilogue every quantized inference stack wants.
///
/// The bias is added in the accumulator's width, before the single encode step,
/// so a bias large enough to matter is not rounded twice.
#[derive(Clone, Copy, Debug)]
pub struct Bias<'a> {
    /// One entry per output column.
    pub values: &'a [i64],
}

impl<E, O> Epilogue<E, O> for (Bias<'_>, usize)
where
    E: Element,
    O: EncodeFrom<AccOf<E>>,
    AccOf<E>: ScaleExact,
{
    fn reads_c(&self) -> bool {
        false
    }

    fn finish(&self, acc: AccOf<E>, _prior: Option<O>, mode: EncodeMode) -> O {
        let (bias, column) = *self;
        let b = bias.values.get(column).copied().unwrap_or(0);
        O::encode_from(acc.combine(AccOf::<E>::from_i128(b as i128)), mode)
    }
}

/// `C := (alpha ⊗ A⊗B) ⊕ (beta ⊗ C)`, the `(max, +)` epilogue.
///
/// The tropical reading of [`Linear`], line for line: `⊗` is addition, `⊕` is
/// `max`, and the two scalars are tropical elements rather than integers. It is
/// the *same* epilogue at a different semiring --- the body below differs from
/// [`Linear`]'s only in which trait supplies `⊗`, which is what makes this an
/// instantiation and not a second method (R13).
///
/// `beta` at the semiring zero is the tropical overwrite: `-inf ⊗ C` is `-inf`,
/// which contributes nothing to the `max`, so the driver need not read `C` at
/// all --- and an uninitialised output buffer is admissible here for exactly
/// the reason `beta == 0` makes it admissible in the ring (`CS-11`, the
/// tropical sibling of `CS-04`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MaxPlus {
    /// Shifts the product. The multiplicative identity is `finite(0)`.
    pub alpha: Trop<i64>,
    /// Shifts the value already in `C`. The semiring zero overwrites.
    pub beta: Trop<i64>,
}

impl MaxPlus {
    /// `C := A⊗B`, overwriting. The common case, and the one that never reads
    /// the output buffer.
    pub const OVERWRITE: Self = Self {
        alpha: Trop::finite(0),
        beta: Trop::NEG_INF,
    };

    /// `C := (A⊗B) ⊕ C`.
    pub const ACCUMULATE: Self = Self {
        alpha: Trop::finite(0),
        beta: Trop::finite(0),
    };
}

impl<E, O> Epilogue<E, O> for MaxPlus
where
    E: Element,
    O: EncodeFrom<AccOf<E>>,
    AccOf<E>: ShiftExact + AbsorbPrior<O>,
{
    fn reads_c(&self) -> bool {
        // `beta` at the semiring zero absorbs `C` away, which is what makes an
        // uninitialised output buffer admissible (`CS-11`).
        self.beta.is_finite()
    }

    fn finish(&self, acc: AccOf<E>, prior: Option<O>, mode: EncodeMode) -> O {
        let shifted = acc.shift_exact(self.alpha);
        let total = match prior {
            // `beta` is the semiring zero and the driver did not read `C`.
            None => shifted,
            Some(c) => {
                if self.beta.is_finite() {
                    shifted.combine(AccOf::<E>::of_prior(c).shift_exact(self.beta))
                } else {
                    shifted
                }
            }
        };
        O::encode_from(total, mode)
    }
}

/// Exact `⊗` by a tropical scalar.
///
/// The tropical twin of [`ScaleExact`], and a separate trait for the reason
/// that keeps the two families apart everywhere else: `⊗` is *addition* here,
/// so an accumulator that implements one must not implement the other. The
/// separation is what excludes [`Linear`] from a tropical accumulation and
/// [`MaxPlus`] from a ring one, by construction rather than by a check.
pub trait ShiftExact: Accumulator {
    /// `self ⊗ by`, exactly. Absorbing at the semiring zero.
    fn shift_exact(self, by: Trop<i64>) -> Self;
}

impl<const W: u32> ShiftExact for TropAcc<W> {
    fn shift_exact(self, by: Trop<i64>) -> Self {
        match (self.get(), by.get()) {
            // Exact, and it cannot leave the register: the accumulated value
            // has magnitude at most `2^BITS` --- `⊕` selects rather than sums,
            // so a reduction of any depth reaches no further --- and the scalar
            // at most `2^63`, so the sum is inside `TropAcc::DOMAIN` at every
            // element width. A derivation, so there is no saturating step here
            // (R3), and the domain it rests on is the one `TropAcc::of`
            // declares and checks.
            (Some(v), Some(s)) => Self::of(v + s as i128),
            // `-inf ⊗ a = -inf`, in either argument. No arithmetic is performed
            // at all, which is why the checked profile has nothing to trap on
            // (`CT-08`).
            _ => Self::NEG_INF,
        }
    }
}

impl<const W: u32, T> AbsorbPrior<Trop<T>> for TropAcc<W>
where
    T: IntegerElement + Into<i128>,
{
    fn of_prior(prior: Trop<T>) -> Self {
        match prior.get() {
            Some(v) => Self::of(v.into()),
            None => Self::NEG_INF,
        }
    }
}

/// How the value already in `C` enters the accumulation.
///
/// `beta * C` requires the prior output element to become an accumulator value,
/// and it must do so *exactly* --- otherwise the epilogue would round twice.
/// One trait rather than a bound on `O`, so that the integer families and the
/// float families reach [`Linear`] through the same impl and there is no second
/// epilogue (R13).
pub trait AbsorbPrior<O>: Accumulator {
    /// The accumulator holding exactly the value `prior` names.
    fn of_prior(prior: O) -> Self;
}

impl<O: Into<i128>> AbsorbPrior<O> for i128 {
    fn of_prior(prior: O) -> Self {
        prior.into()
    }
}

impl<O: Into<i128>, const L: usize> AbsorbPrior<O> for Limbs<L> {
    fn of_prior(prior: O) -> Self {
        Self::ZERO.add_i128(prior.into())
    }
}

impl<O: FloatElement, const L: usize, const MIN_EXP: i32> AbsorbPrior<O> for Complete<L, MIN_EXP> {
    fn of_prior(prior: O) -> Self {
        // The same decode every operand goes through. A float already in `C`
        // is a code like any other.
        Self::of(prior)
    }
}

/// Exact scaling and widening for an accumulator.
///
/// Separated from [`Accumulator`] because it is epilogue machinery, not
/// accumulation machinery: nothing in the accumulation path scales anything.
pub trait ScaleExact: Accumulator {
    /// Multiply by an exact integer scalar.
    ///
    /// The implementation must be observationally exact at the terminal
    /// expression's output alphabet. A complete float accumulator retains
    /// every bit through the combine: its requested rounding depends on the
    /// full sign and magnitude, not only on low congruence bits (`CS-13`).
    fn scale_exact(self, factor: i64) -> Self;

    /// Widen a machine integer into this accumulator.
    fn from_i128(v: i128) -> Self;
}

impl ScaleExact for i128 {
    fn scale_exact(self, factor: i64) -> Self {
        self.saturating_mul(factor as i128) // R3-ok: the single encode step, as this trait's doc states
    }

    fn from_i128(v: i128) -> Self {
        v
    }
}

impl<const L: usize, const MIN_EXP: i32> ScaleExact for uor_matmul_core::Complete<L, MIN_EXP> {
    fn scale_exact(self, factor: i64) -> Self {
        // Exact: the model derives an extension word from arbitrary i64 growth
        // plus the two terms of `alpha * sum + beta * C`.
        self.scale(factor)
    }

    fn from_i128(v: i128) -> Self {
        // An integer is a dyadic rational with exponent zero. `v` is at most
        // 128 bits, so it lands in the register without rounding.
        let negative = v < 0;
        let mut out = Self::ZERO;
        out.add_scaled(v.unsigned_abs(), 0, negative);
        out
    }
}

impl<const L: usize> ScaleExact for uor_matmul_core::Limbs<L> {
    fn scale_exact(self, factor: i64) -> Self {
        // Exact whenever the product fits the register, which is the only case
        // an output type can distinguish. Built from the register's own
        // `add_i128`, so there is no second multiplication routine.
        let negative_factor = factor < 0;
        let mut magnitude = factor.unsigned_abs();
        let mut acc = Self::ZERO;
        let mut addend = if negative_factor { self.neg() } else { self };
        while magnitude > 0 {
            if magnitude & 1 == 1 {
                acc = acc.combine(addend);
            }
            addend = addend.combine(addend);
            magnitude >>= 1;
        }
        acc
    }

    fn from_i128(v: i128) -> Self {
        Self::ZERO.add_i128(v)
    }
}

/// An accumulator that can take an exact integer at a dyadic scale.
///
/// [`ScaleExact::from_i128`] is this at exponent zero. The general form is the
/// terminal embedding shared by Atlas factorizations: a resolved exact
/// coefficient arrives at its Laurent grade and [`Complete::add_scaled`]
/// places it without rounding (`CD-19`).
pub trait PlaceAt: Accumulator {
    /// Accumulate exactly `v * 2^exponent`.
    fn place_at(&mut self, v: i128, exponent: i32);
}

impl<const L: usize, const MIN_EXP: i32> PlaceAt for Complete<L, MIN_EXP> {
    fn place_at(&mut self, v: i128, exponent: i32) {
        self.add_scaled(v.unsigned_abs(), exponent, v < 0);
    }
}

/// A compatibility scale channel between an exact integer sum and float output.
///
/// The public type predates the pure-Atlas float body and therefore retains its
/// spelling. Its operation is nevertheless the same terminal Atlas embedding:
/// place an exact `i128` coefficient at the declared Laurent grade, then run
/// the inner epilogue. Dense float GEMM no longer reifies operands to reach
/// this type; symbol tabulation contracts its compact coefficients through
/// signed-octet lookup before placement (`CD-19`, `CD-20`).
#[derive(Clone, Copy, Debug)]
pub struct Scaled<'e, F: FloatElement, Ep> {
    /// The epilogue that runs on the placed accumulator.
    inner: &'e Ep,
    /// The exponent of bit 0 of every integer sum: `base_a + base_b`.
    base: i32,
    float: PhantomData<F>,
}

impl<'e, F: FloatElement, Ep> Scaled<'e, F, Ep> {
    /// Wrap `inner`, placing each exact integer sum at `2^base` first.
    pub fn new(inner: &'e Ep, base: i32) -> Self {
        Self {
            inner,
            base,
            float: PhantomData,
        }
    }
}

impl<F, O, Ep> Epilogue<i32, O> for Scaled<'_, F, Ep>
where
    F: FloatElement,
    Ep: Epilogue<F, O>,
    AccOf<F>: PlaceAt,
{
    fn reads_c(&self) -> bool {
        // A property of the inner epilogue, unchanged by the placement.
        self.inner.reads_c()
    }

    fn finish(&self, acc: i128, prior: Option<O>, mode: EncodeMode) -> O {
        let mut placed = <AccOf<F> as Accumulator>::ZERO;
        placed.place_at(acc, self.base);
        self.inner.finish(placed, prior, mode)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deepest_reachable_sum<F: FloatElement>(max_finite: F) -> AccOf<F> {
        let mut sum = AccOf::<F>::ZERO;
        F::mac(&mut sum, max_finite, max_finite);
        // One term doubled MAX_K_BITS - 1 times represents 2^(MAX_K_BITS - 1)
        // equal terms, a depth a 64-bit address space can present. Building the
        // value by association is the executable witness; the test need not
        // allocate the address space it represents.
        for _ in 1..uor_matmul_core::generated::MAX_K_BITS {
            sum = sum.combine(sum);
        }
        sum
    }

    fn assert_float_linear_terminal_expression<F>(
        half: F,
        one: F,
        positive_infinity: F,
        negative_infinity: F,
        max_finite: F,
        min_finite: F,
    ) where
        F: FloatElement + EncodeFrom<AccOf<F>> + PartialEq + core::fmt::Debug,
        AccOf<F>: ScaleExact + AbsorbPrior<F>,
    {
        let half_min = (i64::MIN.unsigned_abs() >> 1) as i64;
        for mode in [
            EncodeMode::Nearest,
            EncodeMode::TowardZero,
            EncodeMode::Saturating,
            EncodeMode::Wrapping,
        ] {
            let canceled = <Linear as Epilogue<F, F>>::finish(
                &Linear {
                    alpha: i64::MIN,
                    beta: half_min,
                },
                AccOf::<F>::of_prior(half),
                Some(one),
                mode,
            );
            assert_eq!(
                canceled,
                F::ZERO,
                "large alpha/beta cancellation changed under {mode:?}"
            );
        }

        for (mode, expected_positive, expected_negative) in [
            (EncodeMode::Nearest, positive_infinity, negative_infinity),
            (EncodeMode::TowardZero, max_finite, min_finite),
            (EncodeMode::Saturating, max_finite, min_finite),
            (EncodeMode::Wrapping, max_finite, min_finite),
        ] {
            let positive = <Linear as Epilogue<F, F>>::finish(
                &Linear {
                    alpha: i64::MAX,
                    beta: 0,
                },
                deepest_reachable_sum(max_finite),
                None,
                mode,
            );
            assert_eq!(
                positive, expected_positive,
                "a reachable positive terminal overflow lost its sign under {mode:?}"
            );

            let negative = <Linear as Epilogue<F, F>>::finish(
                &Linear {
                    alpha: i64::MIN,
                    beta: 0,
                },
                deepest_reachable_sum(max_finite),
                None,
                mode,
            );
            assert_eq!(
                negative, expected_negative,
                "a reachable negative terminal overflow lost its sign under {mode:?}"
            );
        }

        let infinity = AccOf::<F>::of_prior(positive_infinity);
        let flipped = <Linear as Epilogue<F, F>>::finish(
            &Linear { alpha: -1, beta: 0 },
            infinity,
            None,
            EncodeMode::Nearest,
        );
        assert_eq!(flipped, negative_infinity);
        let zero_times_infinity = <Linear as Epilogue<F, F>>::finish(
            &Linear { alpha: 0, beta: 0 },
            infinity,
            None,
            EncodeMode::Nearest,
        );
        assert!(matches!(
            zero_times_infinity.decode(),
            uor_matmul_core::Decoded::NotANumber
        ));
    }

    #[test]
    fn float_linear_scalars_preserve_the_terminal_expression_cs_13() {
        assert_float_linear_terminal_expression::<f32>(
            0.5,
            1.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::MAX,
            f32::MIN,
        );
        assert_float_linear_terminal_expression::<f64>(
            0.5,
            1.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::MAX,
            f64::MIN,
        );
    }
}
