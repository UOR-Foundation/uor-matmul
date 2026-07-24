//! Derived widths and the one internal predicate (§5.1).
//!
//! There are no public constants such as `B_I8` or `K_MAX_I32`, and no public
//! `k_max`. A library user who instantiates `(i16, 4095)` never sees `127` or
//! `133144` anywhere, and neither does a user who instantiates `(i8, 127)` with
//! `k = 10^9` (R1, R8).

use crate::alphabet::Element;

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
pub const fn acc_bits<E: Element>() -> u32 {
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
    let per_step = b * b;
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

/// The widest narrow register that can hold a run of `k` products bounded by
/// `b`, or `None` when the tile must use the wide accumulator.
///
/// The whole of the narrow/wide decision, in one place, so that the reference
/// and every backend factor the accumulation the same way and `CB-*` compares
/// like with like.
#[doc(hidden)]
pub const fn narrow_cap_for(b: u128, k: usize) -> Option<u128> {
    let mut i = 0;
    while i < NARROW_CAPS.len() {
        if fits_narrow(b, NARROW_CAPS[i], k) {
            return Some(NARROW_CAPS[i]);
        }
        i += 1;
    }
    None
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

    /// §5.1: the predicate is a question about register width, not a limit.
    /// The W8A8 threshold is where the plain i32 tile stops being usable, and
    /// one past it the answer is still computed --- in a wider register.
    #[test]
    fn fits_narrow_is_a_register_question_not_a_limit() {
        let cap = generated::narrow::CAP_I32;
        assert!(fits_narrow(127, cap, 133_144));
        assert!(!fits_narrow(127, cap, 133_145));
        // A bound of zero is an alphabet containing only zero. Every depth fits.
        assert!(fits_narrow(0, cap, usize::MAX));
    }
}
