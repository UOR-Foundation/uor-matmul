//! Derived widths and the one internal predicate (§5.1).
//!
//! There are no public constants such as `B_I8` or `K_MAX_I32`, and no public
//! `k_max`. A library user who instantiates `(i16, 4095)` never sees `127` or
//! `133144` anywhere, and neither does a user who instantiates `(i8, 127)` with
//! `k = 10^9` (R1, R8).

use crate::alphabet::{Bound, IntegerElement};

/// Bits sufficient for any accumulation *any addressable machine* can express:
///
/// ```text
/// sign + MAX_K_BITS + log2(B_a) + log2(B_w) + log2(products per mac)
/// ```
///
/// [`MAX_K_BITS`] is declared in `model/constants.toml`, not probed from the
/// host, so the accumulator type is the same on a 64-bit host, a 32-bit host,
/// and wasm32. A 32-bit host cannot reach that depth, which makes the width
/// conservative there and never wrong anywhere --- and, more usefully, makes
/// `CD-06` and `CA-02` comparisons of one function rather than of two that
/// happen to agree.
///
/// The last term is `0` for a scalar element type and `1` for a complex one,
/// whose real part sums two element-products; it is read from
/// [`Element::PRODUCT_TERMS`] so that a complex alphabet needs no separate
/// width table and no branch.
///
/// The public accumulator width is this and nothing else, so no input can
/// overflow it. There is no ladder, no policy, no promotion, and no `k_max` in
/// the public API (§3.2).
///
/// Over [`IntegerElement`], and only that. The formula is `2 * (BITS - 1)` for
/// the two magnitudes, which is a statement about *magnitudes* --- a float has an
/// exponent range instead, and its accumulator is a `Complete` spanning that
/// range, whose width is `generated::complete_width`. Bounded on `Element`, this
/// answered `127` for `f32`, whose accumulator is in fact 704 bits, while its own
/// doc said the width is "this and nothing else": a public function giving a
/// wrong answer about the one thing the library promises cannot overflow.
///
/// [`MAX_K_BITS`]: crate::generated::MAX_K_BITS
///
/// # Examples
///
/// ```
/// # use uor_matmul_core::acc_bits;
/// // The worst case for i8 needs 79 bits, so the accumulator is an i128 and
/// // there is nothing left for a caller to choose.
/// assert_eq!(acc_bits::<i8>(), 79);
/// assert!(acc_bits::<i8>() <= 128);
/// ```
pub const fn acc_bits<E: IntegerElement>() -> u32 {
    1 + crate::generated::MAX_K_BITS + 2 * (E::BITS - 1) + E::PRODUCT_TERMS.ilog2()
}

/// 64-bit limbs sufficient for `bits` bits.
pub const fn limbs_for(bits: u32) -> usize {
    (bits as usize).div_ceil(64)
}

/// May this tile be accumulated in a narrower register without changing the
/// answer?
///
/// A `false` selects the wide register. It never selects a different method and
/// never reaches the caller: both sides compute the same integer, so the choice
/// is invisible, has no failure mode, and is never surfaced. That is what
/// separates an optimization from a fallback (R13) --- a fallback changes the
/// answer or the guarantee, and this changes neither.
///
/// `cap` is the largest magnitude the narrow register holds; `b` is the
/// alphabet bound; `k` is the depth of the tile.
/// `#[doc(hidden)] pub`, because `uor-matmul-kernels` needs it across a crate
/// boundary and Rust has no workspace-internal visibility. It is outside the
/// semver contract and absent from the rendered docs, so it is internal in
/// every sense that matters.
#[doc(hidden)]
pub const fn fits_narrow(b: u128, cap: u128, k: usize) -> bool {
    // `b * b` is the worst-case magnitude of one product. A zero bound means an
    // alphabet with only zero in it, for which every k fits.
    //
    // Checked, because `b` is the caller's declaration and `Bnd<{1 << 64}>` is a
    // legitimate one --- every `i8` is below it, so `as_alphabet` admits it. Its
    // square is not a `u128`, and unchecked this panicked with "attempt to
    // multiply with overflow" under the checked profile and wrapped to zero in
    // release, where zero means "every depth fits a 32-bit lane" --- the answer
    // that is not merely wrong but maximally so. A bound whose square is not
    // representable does not fit any narrow register, which is what `None` here
    // says, and the wide accumulator carries it as it carries everything else.
    let Some(per_step) = b.checked_mul(b) else {
        return false;
    };
    if per_step == 0 {
        return true;
    }
    (k as u128) <= cap / per_step
}

/// The narrow candidates, widest first.
///
/// A kernel takes the first that fits; if none does, the tile uses `AccOf<E>`
/// directly. Every entry computes the same integer, so this is an ordering by
/// speed and never by quality (R13).
#[doc(hidden)]
pub const NARROW_CAPS: [u128; 2] = [i64::MAX as u128, i32::MAX as u128];

/// The **narrowest** register that can hold a run of `k` products bounded by
/// `b`, or `None` when the tile must use the wide accumulator.
///
/// [`NARROW_CAPS`] is declared widest-first, and this scans it from the narrow
/// end, because a wider cap is *easier* to satisfy: scanning from the wide end
/// would always return the widest and the list would carry no information.
/// What a kernel wants is the narrowest lane that suffices, since every lane
/// computes the same integer and the narrow ones are faster --- which is what
/// "ordered by speed" means (§5.1).
///
/// A `None` is not a failure. It selects `AccOf<E>`, which the width derivation
/// already guaranteed cannot overflow for any expressible `k`.
#[doc(hidden)]
pub const fn narrow_cap_for(b: u128, k: usize) -> Option<u128> {
    let mut i = NARROW_CAPS.len();
    while i > 0 {
        i -= 1;
        if fits_narrow(b, NARROW_CAPS[i], k) {
            return Some(NARROW_CAPS[i]);
        }
    }
    None
}

/// Witness that an f64 arithmetic lane computes the same integer the exact
/// accumulator does, for a declared alphabet bound and a declared depth.
///
/// IEEE 754 binary64 holds every integer up to `2^53` exactly, and an add,
/// subtract, multiply, or FMA whose operands and result are integers inside
/// that range *is* the integer operation --- exact, and independent of
/// schedule, because an exact result has no schedule. So a float opcode is not
/// forbidden by what it is; it is forbidden where it can round, and permitted
/// where it cannot. Where it cannot is a fact about two numbers: the alphabet
/// bound, and the deepest run of products the lane accumulates before it is
/// folded into the exact accumulator. A product of two alphabet elements has
/// magnitude at most `Bd::VALUE^2`, and a run of at most `DEPTH` of them at
/// most `DEPTH * Bd::VALUE^2`, so the lane is exact exactly while that total is
/// at most `2^53`.
///
/// That fact is checked *here*, at compile time, by the type system and not by
/// a reviewer: [`F64Exact::new`] is the only constructor, and it does not
/// compile outside the bound. This is the shape `CU-01` takes after its
/// restatement --- the property is exactness and schedule independence, not the
/// absence of an opcode --- and it is what the gates key on: the disassembly
/// gate permits a float opcode only in a function named for this witness, and
/// the source gate requires every such function to mention the witness, so a
/// float opcode in shipped code implies a proof the compiler has already
/// checked. An fp64 FMA on integer operands below `2^26` is exact; a gate that
/// forbids it is forbidding a factorization, not a defect.
#[derive(Clone, Copy, Debug)]
pub struct F64Exact<Bd: Bound, const DEPTH: u64> {
    // Private, so the only way to hold one is `new`, and `new` is the proof.
    // `Bd` is carried for the type-level fact, not for anything to store.
    proof: core::marker::PhantomData<Bd>,
}

impl<Bd: Bound, const DEPTH: u64> F64Exact<Bd, DEPTH> {
    /// The precondition, as a const. Referencing it is what forces the
    /// compiler to evaluate it; an associated const that nothing references is
    /// never evaluated, which is why [`F64Exact::new`] names it.
    const EXACT: () = {
        // Checked, for the same reason `fits_narrow` is: `Bnd<{1 << 64}>` is a
        // legitimate bound, its square is not a `u128`, and the honest answer
        // there is that no f64 lane is exact --- which `panic!` says here by
        // refusing to compile the witness at all.
        let Some(per_step) = Bd::VALUE.checked_mul(Bd::VALUE) else {
            panic!("an f64 lane is exact only while every product is an integer below 2^53")
        };
        let Some(total) = per_step.checked_mul(DEPTH as u128) else {
            panic!("an f64 lane is exact only while every partial sum is an integer below 2^53")
        };
        // `MANTISSA_DIGITS` is the language's own spelling of the binary64
        // significand width --- 53, counting the implicit leading bit. Every
        // integer up to `2^53` is exactly representable, and arithmetic on
        // integers inside that range is the integer arithmetic (IEEE-754-2019,
        // §3.3 and §5: the format names exact values and the operations are
        // correctly rounded, so an exactly representable result is the exact
        // result).
        assert!(
            total <= 1u128 << f64::MANTISSA_DIGITS,
            "an f64 lane computes the same integer only while DEPTH * Bd::VALUE^2 <= 2^53"
        );
    };

    /// Construct the witness. Compiles exactly when the declared bound and
    /// depth keep every product and every partial sum below `2^53`; outside
    /// that, this function does not exist for the instantiation, which is the
    /// property being proved.
    #[must_use]
    pub const fn new() -> Self {
        let () = Self::EXACT;
        Self {
            proof: core::marker::PhantomData,
        }
    }
}

impl<Bd: Bound, const DEPTH: u64> Default for F64Exact<Bd, DEPTH> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generated;

    /// CM-01: the `const fn` agrees with the model, for every element type.
    #[test]
    fn acc_bits_agrees_with_the_model_cm_01() {
        assert_eq!(acc_bits::<i8>(), generated::acc_width::I8_BITS);
        assert_eq!(acc_bits::<i16>(), generated::acc_width::I16_BITS);
        assert_eq!(acc_bits::<i32>(), generated::acc_width::I32_BITS);
        assert_eq!(acc_bits::<i64>(), generated::acc_width::I64_BITS);
    }

    /// `CS-07`: the generated `133144` pin equals `k_max(127, i32::MAX)`
    /// recomputed at test time.
    ///
    /// The numeral is a check on the derivation, not a definition. And it is a
    /// threshold on a *register*, not on an answer: one past it the value is
    /// still computed, in a wider one.
    #[test]
    fn the_w8a8_threshold_is_recomputed_cs_07() {
        let cap = generated::narrow::CAP_I32;
        let recomputed = cap / (127 * 127);
        assert_eq!(recomputed, generated::narrow::I32_TILE_W8A8_K_MAX);
        assert!(fits_narrow(127, cap, recomputed as usize));
        assert!(!fits_narrow(127, cap, recomputed as usize + 1));
        // One past it, a wider narrow register still holds the tile.
        assert_eq!(
            narrow_cap_for(127, recomputed as usize + 1),
            Some(i64::MAX as u128)
        );
        // A bound of zero is an alphabet containing only zero. Every depth fits.
        assert!(fits_narrow(0, cap, usize::MAX));
    }

    /// `CU-02`: the narrow candidates are ordered by speed, not by quality.
    ///
    /// Every entry computes the same integer, and `None` is not an error: it
    /// selects `AccOf<E>`, which the width derivation already guaranteed cannot
    /// overflow. That is the difference between a factorization and a fallback.
    #[test]
    fn narrow_candidates_are_ordered_by_speed_cu_02() {
        assert_eq!(narrow_cap_for(127, 1_000), Some(i32::MAX as u128));
        assert_eq!(narrow_cap_for(127, 200_000), Some(i64::MAX as u128));
        assert_eq!(NARROW_CAPS, [i64::MAX as u128, i32::MAX as u128]);
        // No narrow register holds this run, and it says so on every target.
        // `usize::MAX` will not serve here: it is `2^32 - 1` on `wasm32` and on
        // `thumbv7em`, and at `b = 127` an `i64` lane holds *every* depth a
        // 32-bit `usize` can express --- so the widest candidate fits and the
        // honest answer on those targets is `Some`, not `None`. Reaching the
        // `None` by way of the bound instead of the depth asks the same question
        // of a 32-bit and a 64-bit host.
        // A declaration whose *square* is not representable. `Bnd<{1 << 64}>` is
        // a legitimate bound for any integer element --- every `i8` is below it,
        // so `as_alphabet` admits it --- and `b * b` is then not a `u128`.
        // Unchecked, this panicked under the checked profile and wrapped to zero
        // in release, where zero per step reads as "a 32-bit lane holds every
        // depth": not merely the wrong answer but the most wrong one available.
        // No narrow register holds it, the wide accumulator does, and the sum is
        // the same integer the full alphabet gives.
        assert_eq!(narrow_cap_for(1u128 << 64, 1), None);
        assert_eq!(narrow_cap_for(u128::MAX, 1), None);
        assert!(!fits_narrow(1u128 << 64, i64::MAX as u128, 1));

        let wide = 1u128 << 31;
        assert!(
            wide * wide * 4 > i64::MAX as u128,
            "the case has to be out of reach of the widest candidate to test it"
        );
        assert_eq!(narrow_cap_for(wide, 4), None);
    }

    /// `CU-01`: the f64-lane witness compiles exactly inside its precondition.
    ///
    /// The failing direction cannot be a runtime test, because it is a
    /// compile-time fact: `F64Exact::<Bnd<128>, 512>` has no constructor. What
    /// a test can pin is the boundary the assertion computes --- the largest
    /// admissible depth at a bound, and one past it, evaluated by the same
    /// arithmetic the const uses.
    #[test]
    fn the_f64_lane_witness_holds_at_its_boundary_cu_01() {
        use crate::alphabet::Bnd;
        // W8A8 at a panel depth: 512 * 127^2 = 8258048 << 2^53.
        let _w = F64Exact::<Bnd<127>, 512>::new();
        // The boundary itself: bound 2^26, whose square is 2^52, admits a depth
        // of two and not three. The `new` calls below are the admissible side;
        // the inadmissible side is witnessed by `EXACT` evaluating to the same
        // comparison this test recomputes.
        let _w = F64Exact::<Bnd<{ 1 << 26 }>, 2>::new();
        let cap = 1u128 << f64::MANTISSA_DIGITS;
        assert!((1u128 << 26) * (1u128 << 26) * 2 <= cap);
        assert!((1u128 << 26) * (1u128 << 26) * 3 > cap);
        // A bound of zero is an alphabet containing only zero; every depth is
        // exact.
        let _w = F64Exact::<Bnd<0>, { u64::MAX }>::new();
    }
}
