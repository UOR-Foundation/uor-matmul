//! The derivations. Every numeral in `model/*.toml` that could be computed is
//! computed here, and [`crate::Model::check`] fails if the model disagrees.
//!
//! These are the host-side twins of the `const fn`s in `uor-matmul-core`. The
//! two must agree, which is what `CM-01` checks; keeping them as separate
//! implementations is deliberate, because a generated constant checked against
//! the generator that produced it checks nothing.

/// Bits sufficient for any accumulation this machine can express:
///
/// ```text
/// sign + log2(max k) + log2(B_a) + log2(B_w) + log2(products per mac)
/// ```
///
/// with `max k` bounded by `usize::MAX / size_of::<E>()`, because `a` and `w`
/// must both exist in memory.
///
/// This is a function of the element type alone. There is no ladder, no policy,
/// no promotion, and no `k_max` in the public API (§3.2).
pub const fn acc_bits(max_k_bits: u32, element_bits: u32, product_terms: u32) -> u32 {
    1 + max_k_bits + 2 * (element_bits - 1) + product_terms.ilog2()
}

/// The unique accumulator type with at least `bits` bits.
///
/// Not a parameter, not a policy, not a ladder: exactly one type per width.
pub fn accumulator_for(bits: u32) -> String {
    if bits <= 128 {
        "i128".to_string()
    } else {
        format!("Limbs<{}>", limbs_for(bits))
    }
}

/// 64-bit limbs sufficient for `bits` bits.
pub const fn limbs_for(bits: u32) -> usize {
    (bits as usize).div_ceil(64)
}

/// May this tile be accumulated in a narrower register without changing the
/// answer? (§5.1)
///
/// A `false` selects the wide register. It never selects a different method and
/// never reaches the caller: both sides compute the same integer, so the choice
/// is invisible and has no failure mode. That is what separates an optimization
/// from a fallback (R13).
pub const fn fits_narrow(b: u128, cap: u128, k: u128) -> bool {
    k <= cap / (b * b)
}

/// The largest `k` for which a sequence contributing `per_step` per step stays
/// inside `cap`.
pub const fn threshold(cap: u128, per_step: u128) -> u128 {
    cap / per_step
}

/// The first `n` at which tabulating a `code_space`-wide enumeration of `block`
/// elements issues fewer operations than the blocked traversal.
///
/// `m * (k / block)` tables, each costing `code_space * block` products to build
/// and read `n` times at one read and one add per code, against `m * k * n`
/// products. So tabulation is cheaper when
/// `code_space + n / block < n`, that is `n * (block - 1) > code_space * block`.
///
/// `None` when `block == 1`: one code names one element, so the table removes
/// every multiply and no add, and no `n` makes the *op count* cross. That case is
/// not worthless --- a read is cheaper than a widening multiply --- but the
/// justification is residency rather than arithmetic, so this derivation declines
/// to claim it.
pub const fn tabulation_break_even(code_space: usize, block: usize) -> Option<usize> {
    if block <= 1 {
        return None;
    }
    // The predicate is strict, so the first satisfying `n` is one past the
    // quotient whether or not the division is exact.
    Some(code_space * block / (block - 1) + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The W8A8 threshold, as an explicit pin. R1 permits `133144` to appear
    /// exactly here and nowhere else in the workspace.
    #[test]
    fn w8a8_i32_threshold_is_pinned() {
        const CAP_I32: u128 = 2147483647;
        assert_eq!(threshold(CAP_I32, 127 * 127), 133144);
        assert!(fits_narrow(127, CAP_I32, 133144));
        assert!(!fits_narrow(127, CAP_I32, 133145));
    }

    /// §3.2's resolved table, on a 64-bit host.
    #[test]
    fn accumulator_widths_resolve_as_documented() {
        assert_eq!(acc_bits(64, 8, 1), 79);
        assert_eq!(acc_bits(64, 16, 1), 95);
        assert_eq!(acc_bits(64, 32, 1), 127);
        assert_eq!(acc_bits(64, 64, 1), 191);
        // A complex element sums two products per mac, so it gets one more bit.
        assert_eq!(acc_bits(64, 32, 2), 128);
        assert_eq!(acc_bits(64, 64, 2), 192);

        assert_eq!(accumulator_for(acc_bits(64, 8, 1)), "i128");
        assert_eq!(accumulator_for(acc_bits(64, 16, 1)), "i128");
        assert_eq!(accumulator_for(acc_bits(64, 32, 1)), "i128");
        assert_eq!(accumulator_for(acc_bits(64, 64, 1)), "Limbs<3>");
    }

    /// §3.3's complete accumulator widths.
    #[test]
    fn complete_accumulator_widths_resolve_as_documented() {
        assert_eq!(limbs_for(619), 10);
        assert_eq!(limbs_for(4261), 67);
    }
}
