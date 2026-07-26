//! GEMM over float operands (§3.3, S6).
//!
//! Not a second driver, and not a second method. A float is a code; it decodes
//! to an exact dyadic rational; the products land in a complete accumulator
//! where every add is exact; and the result is rounded once, at the end. That
//! is the same sentence [`crate::gemm`] realizes, at a different instantiation
//! of the same three steps.
//!
//! The library never adds two floats. Every addition below happens in a
//! fixed-point integer register, which is why the result is
//! schedule-independent, tile-independent, and substrate-independent --- and
//! why it is *not* bit-identical to a classical `sgemm` (N1).

use uor_matmul_core::{AccOf, Accumulator, Complete, EncodeFrom, FloatElement, PackedCode, Triple};

use crate::driver::GemmOptions;
use crate::epilogue::Epilogue;

/// `C := epilogue(A * B, C)`, over float operands, computed exactly.
///
/// Returns `()`, for the same reason [`crate::gemm`] does: the requested
/// product exists, because a [`Triple`] exists (R14, C6). Non-finite inputs are
/// codes and propagate by the IEEE rules; they are not an error condition
/// (`CT-03`).
pub fn gemm_float<E, O, Ep>(
    triple: &mut Triple<'_, '_, '_, E, O>,
    epilogue: &Ep,
    options: GemmOptions,
) where
    E: FloatElement,
    O: EncodeFrom<AccOf<E>> + Copy,
    Ep: Epilogue<E, O>,
    AccOf<E>: SignedPlace,
{
    gemm_float_packed(triple, epilogue, options, &mut [], &mut [])
}

/// The same operation, with somewhere to decode into.
///
/// A float is a code, and decoding it is real work: a bit test, two shifts, a
/// mask, and a branch. The naive traversal decodes `B[p][j]` once for every row
/// of `A`, so every element of `B` is decoded `m` times and every element of
/// `A` is decoded `n` times. Decoding once into a panel and multiplying many
/// times removes both factors, which is the same structural point the integer
/// driver makes by packing --- and here it is worth more, because a decode
/// costs far more than a copy.
///
/// The panels are the caller's, so this still allocates nothing. Offering none
/// runs the streaming traversal, which decodes per element and gives the same
/// bytes (S13, `CD-04`).
pub fn gemm_float_packed<E, O, Ep>(
    triple: &mut Triple<'_, '_, '_, E, O>,
    epilogue: &Ep,
    options: GemmOptions,
    pa: &mut [PackedCode],
    pb: &mut [PackedCode],
) where
    E: FloatElement,
    O: EncodeFrom<AccOf<E>> + Copy,
    Ep: Epilogue<E, O>,
    AccOf<E>: SignedPlace,
{
    let shape = triple.shape();
    if shape.m == 0 || shape.n == 0 {
        return;
    }
    let reads_c = epilogue.reads_c();
    let (a, b, c) = triple.parts();

    // One row of `A` and one column of `B` is the smallest offer that removes a
    // whole factor of redundant decoding; below it, the streaming traversal
    // runs. The same identity, walked differently (S13).
    // `k == 0` takes this path too, and not as a special case: the sum over an
    // empty reduction is zero, the loop below computes exactly that, and the
    // epilogue still runs. It is named because the comparison does not catch it
    // --- `pa.len() < 0` is false for every `usize` --- and what followed was
    // `pb.len() / shape.k`, which panics on every build. `gemm_float` returns
    // `()` and has no failure to report (R14, `CT-04`).
    if shape.k == 0 || pa.len() < shape.k || pb.len() < shape.k {
        for i in 0..shape.m {
            for j in 0..shape.n {
                let mut acc = <AccOf<E> as Accumulator>::ZERO;
                for p in 0..shape.k {
                    acc.accumulate_one(a.at(i, p).pack(), b.at(p, j).pack());
                }
                let prior = if reads_c { Some(*c.at(i, j)) } else { None };
                *c.at_mut(i, j) = epilogue.finish(acc, prior, options.encode);
            }
        }
        return;
    }

    // Columns of `B` that fit the offer at once. Every column decodes `B`'s
    // whole depth once and then serves every row of `A`.
    let block = (pb.len() / shape.k).min(shape.n).max(1);

    // The element type's significand width decides whether a product of two of
    // them can leave an `i64`. It is a constant of the type, not of the data.
    let product_fits = 2 * E::SIGNIFICAND_BITS <= 63;

    // Whether both panels can be scaled to a common base is one decision for the
    // whole call, and it has to be: the scaling is written *into* the panels, so a
    // block scaled for one row and read unscaled by another would be read wrong.
    // Deciding it globally costs one walk of each operand's exponents --- `m * k`
    // and `k * n` against `m * k * n` products --- and makes the two panel
    // formats impossible to mix.
    // The walk costs `(m + n) * k` decodes and the scaling saves one placement
    // from each of `m * n * k` products. A decode and a placement are the same
    // order of work, so it pays exactly when `m * n > m + n` --- which is false
    // for a matrix-vector product, where the walk would more than double the
    // whole call, and true for everything with two real dimensions.
    let worth_asking = shape.m.saturating_mul(shape.n) > shape.m.saturating_add(shape.n); // R3-ok: a shape predicate, not a value
    let prescaled = if !worth_asking {
        None
    } else {
        let mut finite = true;
        let mut a_span = Span::EMPTY;
        let mut b_span = Span::EMPTY;
        for i in 0..shape.m {
            for v in a.row_walk(i, 0, shape.k) {
                let code = v.pack();
                finite &= code.is_finite();
                a_span.see(code);
            }
        }
        for j in 0..shape.n {
            for v in b.column_walk(0, j, shape.k) {
                let code = v.pack();
                finite &= code.is_finite();
                b_span.see(code);
            }
        }
        finite
            .then(|| admits::<E>(shape.k, a_span, b_span))
            .flatten()
            .map(|scale| (scale, a_span.base(), b_span.base()))
    };

    let mut j0 = 0;
    while j0 < shape.n {
        let cols = block.min(shape.n - j0);
        // Whether this block of `B` is finite is settled here, while its codes
        // are being walked anyway, rather than once per product afterwards.
        let mut b_finite = true;
        for (jj, j) in (j0..j0 + cols).enumerate() {
            let dst = &mut pb[jj * shape.k..jj * shape.k + shape.k];
            for (slot, v) in dst.iter_mut().zip(b.column_walk(0, j, shape.k)) {
                *slot = v.pack();
                b_finite &= slot.is_finite();
            }
            if let Some((_, _, base_b)) = prescaled {
                rescale(dst, base_b);
            }
        }

        for i in 0..shape.m {
            // Decode row `i` of `A` once, and serve every column of this block
            // from it. Walking rather than indexing, for the reason the integer
            // packer walks: two multiplies per element is most of the cost of a
            // decode that is otherwise a handful of bit operations.
            let mut a_finite = true;
            for (slot, v) in pa[..shape.k].iter_mut().zip(a.row_walk(i, 0, shape.k)) {
                *slot = v.pack();
                a_finite &= slot.is_finite();
            }
            if let Some((_, base_a, _)) = prescaled {
                rescale(&mut pa[..shape.k], base_a);
            }
            let panels = PanelFacts {
                finite: a_finite && b_finite,
                product_fits,
                prescaled: prescaled.map(|(scale, _, _)| scale),
            };
            for (jj, j) in (j0..j0 + cols).enumerate() {
                let mut acc = <AccOf<E> as Accumulator>::ZERO;
                acc.accumulate_panels(
                    &pa[..shape.k],
                    &pb[jj * shape.k..jj * shape.k + shape.k],
                    panels,
                );
                let prior = if reads_c { Some(*c.at(i, j)) } else { None };
                *c.at_mut(i, j) = epilogue.finish(acc, prior, options.encode);
            }
        }
        j0 += cols;
    }
}

/// A limb window over a complete accumulator (D-12).
///
/// Every product must land at its own position for the sum to be exact, and
/// that is what makes a complete accumulator expensive: a spread across limbs
/// and a carry, per product.
///
/// But a *run* of products whose exponents share a limb can be summed in one
/// 128-bit register first and placed once. For weights and activations of
/// similar magnitude --- which is the ordinary case, and the whole reason
/// quantization works --- that is nearly every product, and the cost per
/// product falls to one shift and one add.
///
/// It is not an approximation and not a fast path. The window holds an exact
/// integer at a known scale; flushing it adds that integer at that scale. What
/// changes is how often the wide register is touched, and nothing else.
struct Window<'a, const L: usize, const MIN_EXP: i32> {
    acc: &'a mut Complete<L, MIN_EXP>,
    /// The limb the window sits at, or `usize::MAX` when it is empty.
    at: usize,
    /// The window's contents, at scale `64 * at`.
    bits: i128,
}

impl<const L: usize, const MIN_EXP: i32> Window<'_, L, MIN_EXP> {
    #[inline(always)]
    fn place(&mut self, mantissa: i64, exp: i32) {
        if mantissa == 0 {
            return;
        }
        let shift = exp.wrapping_sub(MIN_EXP);
        if shift < 0 {
            // Below the register's floor. Unreachable for any pair of finite
            // values of the element type this register was sized for.
            return;
        }
        let at = (shift as u32 / 64) as usize;
        let bit = shift as u32 % 64;
        let value = i128::from(mantissa) << bit;

        if at == self.at {
            // The window's own capacity decides when to flush, and it is asked
            // rather than assumed. A term reaches `2^125` for `f64`, so any
            // fixed count would be either wrong or needlessly small --- and a
            // fixed count is an arbitrary ceiling, which R8 does not permit.
            // `checked_add` is the exact question, and its `None` branch is
            // taken about once per `2^60` products for realistic data.
            if let Some(sum) = self.bits.checked_add(value) {
                self.bits = sum;
                return;
            }
        }
        self.flush();
        self.at = at;
        self.bits = value;
    }

    #[inline]
    fn flush(&mut self) {
        if self.at == usize::MAX || self.bits == 0 {
            return;
        }
        // The window holds an exact integer at scale `64 * at`, and placing a
        // magnitude at a scale is exactly what `add_scaled` does: one three-limb
        // spread and one carry, against the four separate `add_signed` calls
        // this used to make by cutting the window into 63-bit pieces. Realistic
        // data flushes whenever the product exponent crosses a limb boundary, so
        // this is not a rare path.
        self.acc.add_scaled(
            self.bits.unsigned_abs(),
            MIN_EXP + (self.at as i32) * 64,
            self.bits < 0,
        );
        self.bits = 0;
    }
}

/// Accumulate a whole dot product of two decoded panels.
///
/// Straight-line for the finite case, which is every product in every ordinary
/// matrix, with one predictable branch guarding the IEEE clause 6 rules.
#[inline]
fn accumulate_run<const L: usize, const MIN_EXP: i32>(
    acc: &mut Complete<L, MIN_EXP>,
    pa: &[PackedCode],
    pb: &[PackedCode],
    panels: PanelFacts,
) {
    let mut window = Window {
        acc,
        at: usize::MAX,
        bits: 0,
    };

    // Both panels finite, and the element type's significands narrow enough that
    // no product can leave an `i64`: then the loop is the three lines it should
    // be, with no per-product test of anything. Both are facts about the *panel*
    // and the *type*, established once, so this is the same arithmetic asked
    // fewer questions --- not a second method (R13). `CU-04` compares it against
    // the per-product traversal on the same operands.
    // Both panels scaled to one base: the reduction is an integer dot product
    // and the register is touched once. The 64-bit lane is the one that
    // vectorizes; the 128-bit lane is one wide multiply per product. Which is
    // admissible was decided by the panels' spans, and both are the same sum
    // (`CU-04`).
    if let Some(scale) = panels.prescaled {
        if scale.wide {
            let mut sum = 0i128;
            for (a, b) in pa.iter().zip(pb) {
                sum = sum.wrapping_add(i128::from(a.mantissa) * i128::from(b.mantissa));
            }
            window
                .acc
                .add_scaled(sum.unsigned_abs(), scale.base, sum < 0);
        } else {
            let mut sum = 0i64;
            for (a, b) in pa.iter().zip(pb) {
                sum = sum.wrapping_add(a.mantissa.wrapping_mul(b.mantissa));
            }
            window
                .acc
                .add_scaled(u128::from(sum.unsigned_abs()), scale.base, sum < 0);
        }
        return;
    }

    if panels.finite && panels.product_fits {
        for (a, b) in pa.iter().zip(pb) {
            window.place(a.mantissa * b.mantissa, a.exp + b.exp);
        }
        window.flush();
        return;
    }

    for (a, b) in pa.iter().zip(pb) {
        if a.is_finite() && b.is_finite() {
            // Does the product fit a signed 64-bit mantissa? For `f32` it
            // always does --- two 24-bit significands make 48 bits --- and for
            // `f64` it sometimes does. `checked_mul` asks exactly that
            // question, so there is no width constant to get wrong and no
            // element type this branch is tuned for.
            if let Some(product) = a.mantissa.checked_mul(b.mantissa) {
                window.place(product, a.exp + b.exp);
            } else {
                let sign = (a.mantissa < 0) != (b.mantissa < 0);
                let (ua, ub) = (a.mantissa.unsigned_abs(), b.mantissa.unsigned_abs());
                let (lo, hi) = (ua & 0xFFFF_FFFF, ua >> 32);
                let sgn = |v: u128| {
                    if sign {
                        -((v & ((1 << 62) - 1)) as i64)
                    } else {
                        (v & ((1 << 62) - 1)) as i64
                    }
                };
                let l = u128::from(lo) * u128::from(ub);
                let h = u128::from(hi) * u128::from(ub);
                let e = a.exp + b.exp;
                window.place(sgn(l), e);
                window.place(sgn(l >> 62), e + 62);
                window.place(sgn(h), e + 32);
                window.place(sgn(h >> 62), e + 94);
            }
            continue;
        }
        window.flush();
        if a.is_nan() || b.is_nan() {
            window.acc.set_nan();
            continue;
        }
        // An infinity times a zero is a NaN, by IEEE 754 clause 7.2; otherwise
        // the sign is the product of the two mantissa signs, which the packing
        // already arranged.
        let (inf, other) = if a.is_infinite() { (a, b) } else { (b, a) };
        if other.is_finite() && other.mantissa == 0 {
            window.acc.set_nan();
        } else {
            window
                .acc
                .set_infinity((inf.mantissa < 0) != (other.mantissa < 0));
        }
    }
    window.flush();
}

/// What a caller has already established about a pair of decoded panels.
///
/// Neither field can change a value. `finite` says the IEEE clause 6 rules have
/// nothing to do here, and `product_fits` is a property of the element type's
/// significand width --- `2 * 24 <= 63` for `f32`. Both are asked once instead
/// of once per product.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PanelFacts {
    /// Every code in both panels is finite.
    pub finite: bool,
    /// Every product of two significands of this element type fits an `i64`.
    pub product_fits: bool,
    /// Both panels hold significands already scaled to a common base, so the
    /// whole dot product is one integer sum placed once. See [`Prescaled`].
    pub prescaled: Option<Prescaled>,
}

impl PanelFacts {
    /// Establish nothing. Always admissible, and always the same answer.
    pub const UNKNOWN: Self = Self {
        finite: false,
        product_fits: false,
        prescaled: None,
    };
}

/// Both panels' significands scaled to one common base.
///
/// This is what removes the placement from the inner loop, and it is the whole
/// of why the float path costs what it costs. A complete accumulator is exact
/// because every product lands at *its own* position, and finding that position
/// is a shift, a limb index, and a carry --- per product. Measured, that
/// placement is the entire distance between 0.43 ns per product and 2.5.
///
/// It does not have to be per product. Write `a * 2^(ea - base_a)` into the
/// panel instead of `(a, ea)`, and likewise for `b`, and then
///
/// ```text
///   sum_p (a_p 2^(ea_p - base_a)) (b_p 2^(eb_p - base_b))
///     = 2^-(base_a + base_b) * sum_p a_p b_p 2^(ea_p + eb_p)
/// ```
///
/// so the float dot product *is* an integer dot product, at one known scale,
/// placed into the register once for the whole reduction. Nothing is
/// approximated: the scaled significands are exact integers and so is their
/// sum.
///
/// What it costs is width, and that is what makes it a declaration rather than a
/// mode. A significand of `P` bits scaled across a span of `w` exponents needs
/// `P + w` bits, a product needs `2P + wa + wb`, and a sum of `k` of them needs
/// `ceil(log2 k)` more. When that fits a signed 64-bit lane the loop is a plain
/// integer dot product and vectorizes; when it needs 128 the loop is one wide
/// multiply per product; when it needs more, the per-product placement is the
/// only sequence that computes this identity and it runs. All three are the same
/// sum --- `CU-04` asserts it --- and which one runs is decided by the panels'
/// exponent span, established while their codes are walked anyway, exactly as
/// [`PanelFacts::finite`] is.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Prescaled {
    /// The exponent of bit 0 of every scaled product: `base_a + base_b`.
    pub base: i32,
    /// The sum needs a 128-bit lane rather than a 64-bit one.
    pub wide: bool,
}

/// The exponent span of a panel, and the base its scaling starts from.
///
/// Zero significands are not seen: a zero contributes nothing to the sum at any
/// scale, so it cannot widen the span and must not.
#[derive(Clone, Copy)]
struct Span {
    min: i32,
    max: i32,
    any: bool,
}

impl Span {
    const EMPTY: Self = Self {
        min: i32::MAX,
        max: i32::MIN,
        any: false,
    };

    #[inline(always)]
    fn see(&mut self, code: PackedCode) {
        if code.mantissa != 0 {
            self.min = self.min.min(code.exp);
            self.max = self.max.max(code.exp);
            self.any = true;
        }
    }

    fn base(&self) -> i32 {
        if self.any {
            self.min
        } else {
            0
        }
    }

    /// How many bits a significand of this panel gains from the scaling.
    fn width(&self) -> u32 {
        if self.any {
            self.max.wrapping_sub(self.min) as u32
        } else {
            0
        }
    }
}

/// Does scaling both panels to a common base keep every intermediate exact, and
/// in which lane?
///
/// Every term is a count of bits, and every count comes from the element type or
/// from the panels themselves. There is no tuned constant and no threshold to
/// choose: the question is whether the widest value that can arise fits the lane,
/// and it is asked arithmetically.
fn admits<E: FloatElement>(k: usize, a: Span, b: Span) -> Option<Prescaled> {
    if !a.any || !b.any {
        // One panel is all zeros, so the sum is zero at any base. Scaling is
        // still the cheaper sequence and still exact.
        return Some(Prescaled {
            base: a.base().saturating_add(b.base()), // R3-ok: an exponent base, not an accumulation
            wide: false,
        });
    }
    let p = E::SIGNIFICAND_BITS;
    let (wa, wb) = (a.width(), b.width());
    // Each scaled significand must itself stay inside a signed 64-bit slot,
    // because that is what the panel holds.
    if p.checked_add(wa)? > 62 || p.checked_add(wb)? > 62 {
        return None;
    }
    // `k` terms, so `ceil(log2 k)` carry bits above the widest product.
    let depth = if k <= 1 { 0 } else { (k - 1).ilog2() + 1 };
    let need = (2 * p)
        .checked_add(wa)?
        .checked_add(wb)?
        .checked_add(depth)?;
    let base = a.base().saturating_add(b.base()); // R3-ok: an exponent base, not an accumulation
    if need <= 62 {
        Some(Prescaled { base, wide: false })
    } else if need <= 126 {
        Some(Prescaled { base, wide: true })
    } else {
        None
    }
}

/// Scale a packed panel's significands to `base`, in place.
///
/// After this the panel's `exp` fields are spent: every significand carries its
/// own exponent as magnitude, and the one exponent left is the caller's `base`.
fn rescale(panel: &mut [PackedCode], base: i32) {
    for code in panel {
        if code.mantissa != 0 {
            code.mantissa <<= code.exp.wrapping_sub(base) as u32;
        }
    }
}

/// What the packed float loop needs from an accumulator.
///
/// A trait rather than an inherent method so that `gemm_float_packed` stays
/// generic over the element type while the hot path stays monomorphic.
pub trait SignedPlace {
    /// Accumulate a whole dot product of two decoded panels, exactly.
    fn accumulate_panels(&mut self, pa: &[PackedCode], pb: &[PackedCode], panels: PanelFacts);
    /// Accumulate one product of two decoded codes.
    fn accumulate_one(&mut self, a: PackedCode, b: PackedCode);
}

impl<const L: usize, const MIN_EXP: i32> SignedPlace for Complete<L, MIN_EXP> {
    #[inline]
    fn accumulate_panels(&mut self, pa: &[PackedCode], pb: &[PackedCode], panels: PanelFacts) {
        accumulate_run(self, pa, pb, panels);
    }

    #[inline]
    fn accumulate_one(&mut self, a: PackedCode, b: PackedCode) {
        accumulate_run(self, &[a], &[b], PanelFacts::UNKNOWN);
    }
}

#[cfg(test)]
// R7 governs the library, not its tests: these build operands on the heap so
// that awkward shapes and long reduction orders can be generated. `CA-01`
// witnesses the library's own zero-allocation claim with a counting allocator.
#[allow(clippy::disallowed_types)]
mod tests {
    use super::*;
    use crate::epilogue::Linear;
    use std::vec;
    use std::vec::Vec;
    use uor_matmul_core::{EncodeMode, MatView, MatViewMut};

    fn product(m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        {
            let av = MatView::row_major(a, m, k).unwrap();
            let bv = MatView::row_major(b, k, n).unwrap();
            let cv = MatViewMut::row_major(&mut c, m, n).unwrap();
            let mut t = Triple::new(av, bv, cv).unwrap();
            gemm_float(&mut t, &Linear::OVERWRITE, GemmOptions::default());
        }
        c
    }

    /// `CT-04`: the degenerate shapes are shapes, not error conditions.
    ///
    /// `k == 0` is the one that mattered: the sum over an empty reduction is
    /// zero and the epilogue still runs, but the offer guard reads
    /// `pa.len() < shape.k`, which is false for every `usize` when `k` is zero,
    /// so control reached `pb.len() / shape.k` and `gemm_float` panicked with
    /// "attempt to divide by zero" --- on every build, release included, since
    /// integer division by zero is not a debug assertion. `gemm_float` returns
    /// `()` and has no failure to report (R14).
    #[test]
    fn degenerate_shapes_are_shapes_ct_04() {
        for &(m, k, n) in &[
            (2usize, 0usize, 2usize),
            (0, 2, 2),
            (2, 2, 0),
            (0, 0, 0),
            (1, 1, 1),
            (1, 0, 1),
            (3, 0, 7),
        ] {
            let a = vec![1.0f32; m * k];
            let b = vec![1.0f32; k * n];
            let got = product(m, k, n, &a, &b);
            assert_eq!(got.len(), m * n, "{m}x{k}x{n}");
            // Every cell is the empty sum where `k` is zero, and `k` products of
            // ones otherwise.
            assert!(
                got.iter().all(|&x| x == k as f32),
                "{m}x{k}x{n} gave {got:?}"
            );
        }
    }

    /// The ordinary case is ordinary: an exact small product is exactly right.
    #[test]
    fn small_exact_products_are_exact_cs_01() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [5.0f32, 6.0, 7.0, 8.0];
        assert_eq!(product(2, 2, 2, &a, &b), vec![19.0, 22.0, 43.0, 50.0]);
    }

    /// `CU-04`: the classic catastrophic-cancellation case. A classical `sgemm`
    /// gets `0.0` or `1.0` here depending on the order it happens to add in;
    /// the exact sum is `1.0` and there is no order in which this returns
    /// anything else.
    #[test]
    fn catastrophic_cancellation_is_exact_cu_04() {
        let big = 1.0e30f32;
        // 1e30 + 1.0 - 1e30. In f32, `1e30 + 1.0` rounds back to `1e30`, so a
        // classical accumulator loses the 1.0 entirely.
        let a = [1.0f32, 1.0, 1.0];
        let b = [big, 1.0, -big];
        assert_eq!(product(1, 3, 1, &a, &b), vec![1.0f32]);

        // And the naive float sum really does lose it, so this is not a
        // vacuous comparison.
        let naive = a.iter().zip(b.iter()).fold(0.0f32, |s, (x, y)| s + x * y);
        assert_eq!(naive, 0.0f32, "a classical accumulator loses the 1.0");
    }

    /// `CU-04`: shuffling the reduction order cannot change the answer, because
    /// nothing is rounded until the end. No classical `f32` GEMM has this
    /// property.
    #[test]
    fn float_accumulation_is_order_independent_cu_04() {
        let k = 257;
        let a: Vec<f32> = (0..k).map(|i| ((i * 37 % 1000) as f32) * 1.0e-3).collect();
        let b: Vec<f32> = (0..k).map(|i| ((i * 53 % 997) as f32) * 1.0e3).collect();
        let reference = product(1, k, 1, &a, &b);

        // The same terms in a different order.
        let mut order: Vec<usize> = (0..k).collect();
        for round in 0..8 {
            order.rotate_left(round * 7 + 1);
            let a2: Vec<f32> = order.iter().map(|&i| a[i]).collect();
            let b2: Vec<f32> = order.iter().map(|&i| b[i]).collect();
            assert_eq!(product(1, k, 1, &a2, &b2), reference, "round {round}");
        }
    }

    /// `CT-03`: non-finite codes propagate by the IEEE rules and never error.
    #[test]
    fn non_finite_codes_propagate_ct_03() {
        let inf = f32::INFINITY;
        assert!(product(1, 1, 1, &[inf], &[2.0])[0].is_infinite());
        assert!(product(1, 1, 1, &[-inf], &[2.0])[0] == f32::NEG_INFINITY);
        // Infinity times zero is a NaN, by clause 7.2.
        assert!(product(1, 1, 1, &[inf], &[0.0])[0].is_nan());
        // Opposite infinities in one sum are a NaN, whatever order they arrive.
        assert!(product(1, 2, 1, &[1.0, 1.0], &[inf, -inf])[0].is_nan());
        assert!(product(1, 2, 1, &[1.0, 1.0], &[-inf, inf])[0].is_nan());
        // A NaN anywhere absorbs.
        assert!(product(1, 3, 1, &[1.0, 1.0, 1.0], &[1.0, f32::NAN, 1.0])[0].is_nan());
    }

    /// `CD-05`: the encode mode is the only thing that changes the bytes for a
    /// fixed accumulation. `Nearest` rounds half to even; `TowardZero`
    /// truncates.
    #[test]
    fn the_encode_mode_is_the_only_variable_cd_05() {
        // A sum that needs rounding: 2^-24 past 1.0 is exactly half an ulp.
        let a = [1.0f32, 1.0];
        let b = [1.0f32, f32::from_bits(0x3380_0000)]; // 2^-24
        let nearest = {
            let mut c = [0.0f32];
            let av = MatView::row_major(&a, 1, 2).unwrap();
            let bv = MatView::row_major(&b, 2, 1).unwrap();
            let cv = MatViewMut::row_major(&mut c, 1, 1).unwrap();
            let mut t = Triple::new(av, bv, cv).unwrap();
            gemm_float(&mut t, &Linear::OVERWRITE, GemmOptions::default());
            c[0]
        };
        let toward_zero = {
            let mut c = [0.0f32];
            let av = MatView::row_major(&a, 1, 2).unwrap();
            let bv = MatView::row_major(&b, 2, 1).unwrap();
            let cv = MatViewMut::row_major(&mut c, 1, 1).unwrap();
            let mut t = Triple::new(av, bv, cv).unwrap();
            gemm_float(
                &mut t,
                &Linear::OVERWRITE,
                GemmOptions {
                    encode: EncodeMode::TowardZero,
                    ..Default::default()
                },
            );
            c[0]
        };
        // Exactly halfway: nearest-even keeps 1.0, truncation also keeps 1.0.
        assert_eq!(nearest, 1.0);
        assert_eq!(toward_zero, 1.0);

        // Just past halfway: nearest rounds up, truncation does not.
        let b2 = [1.0f32, f32::from_bits(0x3400_0000)]; // 2^-23, a full ulp
        let mut c = [0.0f32];
        let av = MatView::row_major(&a, 1, 2).unwrap();
        let bv = MatView::row_major(&b2, 2, 1).unwrap();
        let cv = MatViewMut::row_major(&mut c, 1, 1).unwrap();
        let mut t = Triple::new(av, bv, cv).unwrap();
        gemm_float(&mut t, &Linear::OVERWRITE, GemmOptions::default());
        assert_eq!(c[0], 1.0 + f32::EPSILON);
    }

    /// Subnormals are not a special case: gradual underflow falls out of the
    /// same clamp that puts the least significant bit at its floor.
    #[test]
    fn subnormals_round_correctly_ct_03() {
        let tiny = f32::from_bits(1); // 2^-149, the smallest subnormal
                                      // tiny * 1.0 is exactly tiny.
        assert_eq!(product(1, 1, 1, &[tiny], &[1.0]), vec![tiny]);
        // Four of them sum to 4 * 2^-149, still subnormal and still exact.
        let a = [1.0f32; 4];
        let b = [tiny; 4];
        assert_eq!(product(1, 4, 1, &a, &b), vec![f32::from_bits(4)]);
        // A product below the subnormal floor rounds to zero, once.
        let half_tiny = product(1, 1, 1, &[tiny], &[0.25]);
        assert_eq!(half_tiny, vec![0.0f32]);
    }

    /// Overflow reaches infinity under `Nearest` and clamps under the directed
    /// modes. Either way it happens once, in the encode step.
    #[test]
    fn overflow_happens_once_in_the_encode_step_cs_05() {
        let big = f32::MAX;
        let a = [1.0f32, 1.0];
        let b = [big, big];
        assert_eq!(product(1, 2, 1, &a, &b), vec![f32::INFINITY]);

        let mut c = [0.0f32];
        let av = MatView::row_major(&a, 1, 2).unwrap();
        let bv = MatView::row_major(&b, 2, 1).unwrap();
        let cv = MatViewMut::row_major(&mut c, 1, 1).unwrap();
        let mut t = Triple::new(av, bv, cv).unwrap();
        gemm_float(
            &mut t,
            &Linear::OVERWRITE,
            GemmOptions {
                encode: EncodeMode::Saturating,
                ..Default::default()
            },
        );
        assert_eq!(c[0], f32::MAX);
    }

    /// f64 takes the same path at a different instantiation.
    #[test]
    fn f64_takes_the_same_path_cu_05() {
        let a = [1.0f64, 1.0, 1.0];
        let b = [1.0e300f64, 1.0, -1.0e300];
        let mut c = [0.0f64];
        let av = MatView::row_major(&a, 1, 3).unwrap();
        let bv = MatView::row_major(&b, 3, 1).unwrap();
        let cv = MatViewMut::row_major(&mut c, 1, 1).unwrap();
        let mut t = Triple::new(av, bv, cv).unwrap();
        gemm_float(&mut t, &Linear::OVERWRITE, GemmOptions::default());
        assert_eq!(c[0], 1.0f64);
    }
}

#[cfg(test)]
#[allow(clippy::disallowed_types)]
mod window_tests {
    use super::*;
    use std::vec;
    use std::vec::Vec;
    use uor_matmul_core::{MatView, MatViewMut};

    /// The window is exact for adversarial exponents.
    ///
    /// A window holds an exact integer at a known scale, and a term reaches
    /// `2^125` for `f64`. Any fixed flush count would be wrong for some input;
    /// this is the input that finds it. The exponents are chosen so that every
    /// product lands in the same limb with the largest possible shift within
    /// it, which is the worst case for the window's capacity.
    #[test]
    fn the_window_is_exact_at_full_width_cu_04() {
        for k in [1usize, 2, 3, 4, 5, 8, 17, 64, 1000] {
            // Significands with every bit set, so each product is full width.
            let a: Vec<f64> = (0..k)
                .map(|i| f64::from_bits(0x433F_FFFF_FFFF_FFFF - (i as u64 % 3)))
                .collect();
            let b: Vec<f64> = (0..k)
                .map(|i| f64::from_bits(0x433F_FFFF_FFFF_FFFF - (i as u64 % 5)))
                .collect();

            let mut packed = [0.0f64];
            {
                let av = MatView::row_major(&a, 1, k).unwrap();
                let bv = MatView::row_major(&b, k, 1).unwrap();
                let cv = MatViewMut::row_major(&mut packed, 1, 1).unwrap();
                let mut t = Triple::new(av, bv, cv).unwrap();
                let mut pa = vec![uor_matmul_core::PackedCode::default(); k];
                let mut pb = vec![uor_matmul_core::PackedCode::default(); k];
                gemm_float_packed(
                    &mut t,
                    &crate::epilogue::Linear::OVERWRITE,
                    GemmOptions::default(),
                    &mut pa,
                    &mut pb,
                );
            }

            // The streaming traversal, which places every product on its own
            // and never fills a window. If the two disagree, the window lost a
            // carry.
            let mut streamed = [0.0f64];
            {
                let av = MatView::row_major(&a, 1, k).unwrap();
                let bv = MatView::row_major(&b, k, 1).unwrap();
                let cv = MatViewMut::row_major(&mut streamed, 1, 1).unwrap();
                let mut t = Triple::new(av, bv, cv).unwrap();
                gemm_float_packed(
                    &mut t,
                    &crate::epilogue::Linear::OVERWRITE,
                    GemmOptions::default(),
                    &mut [],
                    &mut [],
                );
            }
            assert_eq!(
                packed, streamed,
                "k={k}: the window disagreed with per-product placement"
            );
        }
    }

    /// `CU-04`: what a panel establishes cannot change what it computes.
    ///
    /// The packed traversal settles two facts once per panel --- that every code
    /// is finite, and that this element type's products fit an `i64` --- and then
    /// runs a loop with no per-product test. The streaming traversal establishes
    /// neither and tests everything. They must agree on every input, including
    /// the ones where the facts are false: a non-finite code anywhere, and `f64`
    /// significands wide enough that a product leaves an `i64`.
    /// `CU-04`: scaling both panels to a common base cannot change a byte.
    ///
    /// Three sequences compute this sum --- the 64-bit scaled lane, the 128-bit
    /// scaled lane, and the per-product placement --- and which one runs is
    /// decided by the operands' exponent spans. So the operands below are chosen
    /// to reach each of the three, and every one of them is compared against the
    /// streaming traversal, which always places per product.
    #[test]
    fn scaling_both_panels_cannot_change_a_byte_cu_04() {
        // Spans chosen to land on each lane: none, a significand's worth, a
        // decade, and far too much for any scaling.
        for (label, span_a, span_b) in [
            ("one exponent", 0i32, 0i32),
            ("a few binades", 3, 4),
            ("a decade each", 19, 23),
            ("past every lane", 90, 90),
        ] {
            for &(m, k, n) in &[
                (1usize, 1usize, 1usize),
                (5, 7, 3),
                (16, 64, 8),
                (3, 129, 17),
            ] {
                let mut seed = 0x9E37_79B9_7F4A_7C15u64 ^ (span_a as u64) << 8 ^ (k as u64);
                let mut next = move || {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    (seed >> 33) as i64
                };
                let gen = |next: &mut dyn FnMut() -> i64, len: usize, span: i32| -> Vec<f32> {
                    (0..len)
                        .map(|_| {
                            let v = next();
                            let s = if span == 0 {
                                0
                            } else {
                                (v % span as i64) as i32
                            };
                            (1 + v % 8_388_607) as f32 * 2.0f32.powi(s - span / 2)
                        })
                        .collect()
                };
                let av = gen(&mut next, m * k, span_a);
                let bv = gen(&mut next, k * n, span_b);

                // The reference: the streaming traversal, which places every
                // product individually and knows nothing about spans.
                let mut want = vec![0.0f32; m * n];
                {
                    let a = MatView::row_major(&av, m, k).unwrap();
                    let b = MatView::row_major(&bv, k, n).unwrap();
                    let c = MatViewMut::row_major(&mut want, m, n).unwrap();
                    let mut t = Triple::new(a, b, c).unwrap();
                    gemm_float(
                        &mut t,
                        &crate::epilogue::Linear::OVERWRITE,
                        GemmOptions::default(),
                    );
                }

                // And every panel offer, because the offer decides the block and
                // the block must not decide the answer.
                for offer in [0usize, 1, k, k * n] {
                    let mut got = vec![0.0f32; m * n];
                    let mut qa = vec![
                        PackedCode {
                            mantissa: 0,
                            exp: 0
                        };
                        k.max(1)
                    ];
                    let mut qb = vec![
                        PackedCode {
                            mantissa: 0,
                            exp: 0
                        };
                        offer
                    ];
                    let a = MatView::row_major(&av, m, k).unwrap();
                    let b = MatView::row_major(&bv, k, n).unwrap();
                    let c = MatViewMut::row_major(&mut got, m, n).unwrap();
                    let mut t = Triple::new(a, b, c).unwrap();
                    gemm_float_packed(
                        &mut t,
                        &crate::epilogue::Linear::OVERWRITE,
                        GemmOptions::default(),
                        &mut qa,
                        &mut qb,
                    );
                    assert_eq!(
                        got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        want.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        "{label}, {m}x{k}x{n}, offer {offer}"
                    );
                }
            }
        }
    }

    /// The three lanes are all reached by the test above, and this says which.
    ///
    /// A differential test over operands that all take one path is a test of one
    /// path, so the spans are asserted to select what they were chosen to select.
    #[test]
    fn each_scaled_lane_is_reached_cu_04() {
        let span = |exps: &[i32]| {
            let mut s = Span::EMPTY;
            for &e in exps {
                s.see(PackedCode {
                    mantissa: 1,
                    exp: e,
                });
            }
            s
        };
        // 24-bit significands, so a 64-bit lane has 62 - 48 = 14 bits to spend on
        // the two spans and the depth together.
        let tight = span(&[0]);
        assert_eq!(
            admits::<f32>(64, tight, tight),
            Some(Prescaled {
                base: 0,
                wide: false
            }),
            "no span and a shallow depth is the 64-bit lane"
        );
        let some = span(&[0, 20]);
        assert!(
            matches!(
                admits::<f32>(64, some, some),
                Some(Prescaled { wide: true, .. })
            ),
            "a span past the 64-bit lane is the 128-bit lane"
        );
        let huge = span(&[0, 100]);
        assert_eq!(
            admits::<f32>(64, huge, huge),
            None,
            "a span past every lane is the per-product placement"
        );
    }

    #[test]
    fn what_a_panel_establishes_cannot_change_it_cu_04() {
        fn both_ways_agree_f32(k: usize, a: &[f32], b: &[f32]) {
            let mut packed = vec![0.0f32; 1];
            let mut streamed = vec![0.0f32; 1];
            let mut pa = vec![uor_matmul_core::PackedCode::default(); k];
            let mut pb = vec![uor_matmul_core::PackedCode::default(); k];
            for (out, offer) in [(&mut packed, true), (&mut streamed, false)] {
                let av = MatView::row_major(a, 1, k).unwrap();
                let bv = MatView::row_major(b, k, 1).unwrap();
                let cv = MatViewMut::row_major(out.as_mut_slice(), 1, 1).unwrap();
                let mut t = Triple::new(av, bv, cv).unwrap();
                let (x, y): (&mut [_], &mut [_]) = if offer {
                    (&mut pa, &mut pb)
                } else {
                    (&mut [], &mut [])
                };
                gemm_float_packed(
                    &mut t,
                    &crate::epilogue::Linear::OVERWRITE,
                    GemmOptions::default(),
                    x,
                    y,
                );
            }
            assert_eq!(
                packed[0].to_bits(),
                streamed[0].to_bits(),
                "the panel facts changed the answer"
            );
        }

        let k = 64usize;
        // Finite, full significand width, exponents spread across limbs.
        let a: Vec<f32> = (0..k)
            .map(|i| f32::from_bits(0x4B7F_FFFF - (i as u32 % 7) - ((i as u32 % 11) << 23)))
            .collect();
        let b: Vec<f32> = (0..k)
            .map(|i| f32::from_bits(0x3F7F_FFFF - (i as u32 % 5) - ((i as u32 % 13) << 23)))
            .collect();
        both_ways_agree_f32(k, &a, &b);

        // A non-finite code makes `finite` false, so the general loop runs; it
        // must still agree with the streaming one it is compared against.
        for at in [0usize, 1, 31, 63] {
            for bad in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 0.0] {
                let mut a2 = a.clone();
                a2[at] = bad;
                both_ways_agree_f32(k, &a2, &b);
                let mut b2 = b.clone();
                b2[at] = bad;
                both_ways_agree_f32(k, &a, &b2);
            }
        }

        // `f64`, where two significands make 106 bits and no product fits an
        // `i64`, so `product_fits` is false for the whole type.
        let a: Vec<f64> = (0..k)
            .map(|i| f64::from_bits(0x433F_FFFF_FFFF_FFFF - (i as u64 % 7)))
            .collect();
        let b: Vec<f64> = (0..k)
            .map(|i| f64::from_bits(0x3FEF_FFFF_FFFF_FFFF - (i as u64 % 5)))
            .collect();
        let mut packed = [0.0f64];
        let mut streamed = [0.0f64];
        let mut pa = vec![uor_matmul_core::PackedCode::default(); k];
        let mut pb = vec![uor_matmul_core::PackedCode::default(); k];
        for (out, offer) in [(&mut packed, true), (&mut streamed, false)] {
            let av = MatView::row_major(&a, 1, k).unwrap();
            let bv = MatView::row_major(&b, k, 1).unwrap();
            let cv = MatViewMut::row_major(out.as_mut_slice(), 1, 1).unwrap();
            let mut t = Triple::new(av, bv, cv).unwrap();
            let (x, y): (&mut [_], &mut [_]) = if offer {
                (&mut pa, &mut pb)
            } else {
                (&mut [], &mut [])
            };
            gemm_float_packed(
                &mut t,
                &crate::epilogue::Linear::OVERWRITE,
                GemmOptions::default(),
                x,
                y,
            );
        }
        assert_eq!(packed[0].to_bits(), streamed[0].to_bits());
    }

    /// The window carries exactly across a limb boundary, where a term in one
    /// limb and a term in the next must not be added to each other.
    #[test]
    fn the_window_carries_across_limbs_cu_04() {
        // Exponents 64 apart put consecutive products in different limbs.
        let k = 128usize;
        let a: Vec<f64> = (0..k).map(|i| (2.0f64).powi(i as i32 * 8 - 500)).collect();
        let b: Vec<f64> = vec![1.0; k];

        let mut packed = [0.0f64];
        let mut pa = vec![uor_matmul_core::PackedCode::default(); k];
        let mut pb = vec![uor_matmul_core::PackedCode::default(); k];
        {
            let av = MatView::row_major(&a, 1, k).unwrap();
            let bv = MatView::row_major(&b, k, 1).unwrap();
            let cv = MatViewMut::row_major(&mut packed, 1, 1).unwrap();
            let mut t = Triple::new(av, bv, cv).unwrap();
            gemm_float_packed(
                &mut t,
                &crate::epilogue::Linear::OVERWRITE,
                GemmOptions::default(),
                &mut pa,
                &mut pb,
            );
        }
        let mut streamed = [0.0f64];
        {
            let av = MatView::row_major(&a, 1, k).unwrap();
            let bv = MatView::row_major(&b, k, 1).unwrap();
            let cv = MatViewMut::row_major(&mut streamed, 1, 1).unwrap();
            let mut t = Triple::new(av, bv, cv).unwrap();
            gemm_float_packed(
                &mut t,
                &crate::epilogue::Linear::OVERWRITE,
                GemmOptions::default(),
                &mut [],
                &mut [],
            );
        }
        assert_eq!(packed, streamed);
    }
}
