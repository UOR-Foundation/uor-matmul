//! The packed `(max, +)` lane: a tropical element in an `i16`, the semiring
//! zero at `i16::MIN` (§7.2, A-4, A-6).
//!
//! The dense sequences in [`crate::isa`] are one instruction wide and one
//! instruction deep: a saturating add is `⊗` and a signed max is `⊕`. What
//! makes that legitimate is not the instructions --- every ISA has them --- but
//! the **representation**, and the representation is the whole of this module.
//!
//! # The semiring zero has to live inside the lane
//!
//! [`Trop<E>`] carries `-inf` as a discriminant beside the value, which is what
//! keeps every representable `E` an ordinary finite element (R8). A vector lane
//! has no room for a discriminant: sixteen `i16` in a register are sixteen
//! `i16`, and a `max` over them cannot consult a seventeenth bit. So the packed
//! form widens the element and spends the widening on the semiring zero ---
//! `Trop<i8>` becomes an `i16` whose finite values are the `i8`s themselves and
//! whose `-inf` is [`TROP_ZERO`], `i16::MIN`.
//!
//! That is a *choice of representation*, and it has to be paid for. Below is
//! the payment, in full.
//!
//! # Why the add must saturate
//!
//! Write `B` for the alphabet bound the packed operands declare: every finite
//! element has magnitude at most `B`. Three cases, and the third is the one the
//! whole family turns on.
//!
//! 1. **Two finite operands.** `|a| <= B` and `|b| <= B`, so `a + b` lies in
//!    `[-2B, 2B]`. [`TROP_I16_MAX_BOUND`] is derived below so that `2B` is
//!    inside `i16::MAX`; no saturation occurs, and the packed sum is the exact
//!    integer sum. The arithmetic in this range is therefore *the semiring's
//!    own arithmetic*, not an approximation of it.
//!
//! 2. **At least one operand at the semiring zero.** One addend is `i16::MIN`
//!    and the other is at most `B` --- or is also `i16::MIN`. The true sum is
//!    at most `i16::MIN + B` and can be as low as `2 * i16::MIN`, which is not
//!    an `i16`; the saturating add clamps it to `i16::MIN`. So the packed
//!    result lies in `[i16::MIN, i16::MIN + B]`, and that clamping is exactly
//!    the absorbing law `-inf ⊗ a = -inf` --- a *wrapping* add would send
//!    `i16::MIN + i16::MIN` to `0`, which is not merely wrong but is the
//!    largest wrong answer the lane can hold, and under the `checked` profile
//!    it is a trap rather than an answer at all (`CT-08`).
//!
//! 3. **The two ranges are disjoint, and in the right order.** Case 2's
//!    largest value is `i16::MIN + B` and case 1's smallest is `-2B`, so the
//!    ranges separate exactly when `i16::MIN + B < -2B`, i.e. when `3B <
//!    2^15`. That inequality *is* [`TROP_I16_MAX_BOUND`]. Under it, every
//!    absorbed lane is strictly below every finite lane, so `max` selects a
//!    finite contribution whenever one exists --- which is the semiring law
//!    `-inf ⊕ x = x` --- and [`unpack`] can read the answer back by comparing
//!    against `-2B` alone.
//!
//! Every step is a derivation from `B`, and each of the three is asserted at
//! its own extreme rather than on likely data:
//! `the_saturation_argument_holds_at_its_own_extremes_cb_13`, in this crate's
//! `tests/parity.rs`, is where.
//!
//! # The family issues no multiply, and its symbols say so
//!
//! `⊗` is an addition. There is no multiply anywhere in this family --- not a
//! widening one, not a paired one, not an address computation that LLVM turns
//! into one --- and that is a claim about *emitted instructions*, so it is read
//! off the disassembly rather than argued (`CU-10`).
//!
//! A disassembly scan needs to know which symbols to read, so every entry point
//! in this family carries `trop_i16` in its name: `mac_trop_i16`,
//! `avx2_trop_i16`, `avx2_trop_i16_inner`, `neon_trop_i16`,
//! `simd128_trop_i16`, and the reference dot `parity::dot_trop_i16`. That is
//! the same shape of convention `gather_reference` is for `CU-06` and
//! `f64_exact` is for `CU-01`: the name is the scope, so adding a sequence
//! enrolls it and forgetting to name one is visible in the source.
//!
//! # Why this is not a sentinel smuggled back in
//!
//! [`Trop`]'s own doc refuses a reserved pattern inside `E`, because that would
//! refuse a representable value of the element type. Nothing is refused here:
//! the packed lane is *wider* than the element, so `i16::MIN` names `-inf`
//! without any `i8` losing its meaning, and `i8::MIN` packs to `-128` like
//! every other finite element. The cost is paid in width, which is the only
//! currency that buys this without taking a value away (R8).

use uor_matmul_core::Trop;

/// The packed semiring zero: `-inf`, the identity of `⊕`, the pad, and the
/// mask --- one value, four readings (A-6, `CK-17`).
///
/// `i16::MIN` and not `i16::MIN + 1` or any other spare pattern: the argument
/// in this module's header needs the absorbed range to be as far below the
/// finite range as the type allows, and it needs the saturating add's own
/// clamp to land *on* the semiring zero rather than near it.
pub const TROP_ZERO: i16 = i16::MIN;

/// The widest alphabet bound at which the packed `i16` lane is exact.
///
/// Derived, not chosen. The separation the header's case 3 needs is
/// `i16::MIN + B < -2B`, which rearranges to `3B < 2^15`, so the widest
/// admissible `B` is `floor(i16::MAX / 3)`. Two const assertions below check
/// that rearrangement at the value itself, and a third checks that one past it
/// the ranges meet --- a bound that is derived and never exercised at its own
/// edge is a bound nobody has read.
///
/// This is a property of the *representation*, so every sequence in the family
/// declares it, the portable reference included. That is the one place this
/// family differs in kind from the ring families, where the reference declares
/// `u128::MAX` and only a paired-product sequence declares less: there the
/// bound belongs to an instruction, and here it belongs to the lane.
pub const TROP_I16_MAX_BOUND: u128 = (i16::MAX as u128) / 3;

// Case 1 of the header: two finite operands never saturate.
const _: () = assert!(
    2 * TROP_I16_MAX_BOUND <= i16::MAX as u128,
    "a finite pair must sum inside the lane, or the packed add is not the semiring's add"
);

// Case 3 of the header: the absorbed range is strictly below the finite one.
const _: () = assert!(
    (TROP_ZERO as i128) + (TROP_I16_MAX_BOUND as i128) < -2 * (TROP_I16_MAX_BOUND as i128),
    "an absorbed lane must lose to every finite lane, or `max` is not `⊕`"
);

// And the bound is the widest one that does: one past it, the two ranges meet.
const _: () = assert!(
    (TROP_ZERO as i128) + (TROP_I16_MAX_BOUND as i128 + 1) >= -2 * (TROP_I16_MAX_BOUND as i128 + 1),
    "a wider bound would still separate, so this declaration gives reach away"
);

/// The magnitude bound of the `Trop<i8>` alphabet: `|i8::MIN|`.
///
/// The alphabet [`pack_i8`] packs from. Stated as the type's own extreme rather
/// than as `128` because `Full<i8>` admits every `i8`, `i8::MIN` included --- a
/// bound of `127` would refuse a representable value, which is exactly what
/// `Trop` exists not to do.
pub const TROP_I8_BOUND: u128 = i8::MIN.unsigned_abs() as u128;

// The `Trop<i8>` alphabet is inside the lane's reach, so the whole family
// serves it: a packed `i8` reduction never approaches the separation.
const _: () = assert!(
    TROP_I8_BOUND <= TROP_I16_MAX_BOUND,
    "the i8 tropical alphabet must fit the packed lane it is packed into"
);

/// A tropical `i8` element in its packed lane.
///
/// Total: every `Trop<i8>` has a packed form, and the map is injective because
/// no finite `i8` is `i16::MIN`.
pub fn pack_i8(x: Trop<i8>) -> i16 {
    match x.get() {
        Some(v) => i16::from(v),
        None => TROP_ZERO,
    }
}

/// The finite value a packed lane holds, or `None` at the semiring zero.
///
/// `bound` is the alphabet the packed operands declared, and the threshold is
/// `-2 * bound`: by the header's case 1 every finite reduction is at or above
/// it, and by case 3 every absorbed one is strictly below. So this comparison
/// is not a heuristic about how negative a value looks --- it is the two
/// ranges, read at the point they separate.
///
/// Panics when `bound` exceeds [`TROP_I16_MAX_BOUND`], where the ranges no
/// longer separate and there is no reading to give. That is a disagreement
/// between a driver and a declaration it made, in the same way
/// [`crate::KernelSpec::mac_tile`]'s length assertions are, and no input a
/// caller can supply reaches it (C6).
pub fn unpack(lane: i16, bound: u128) -> Option<i16> {
    assert!(
        bound <= TROP_I16_MAX_BOUND,
        "the packed tropical lane separates only up to TROP_I16_MAX_BOUND"
    );
    // `bound` is at most `i16::MAX / 3`, so `2 * bound` is an `i16` and the
    // negation below cannot leave it.
    let floor = -2 * (bound as i16);
    if lane < floor {
        None
    } else {
        Some(lane)
    }
}
