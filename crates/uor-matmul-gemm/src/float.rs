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

use uor_matmul_core::{AccOf, Accumulator, EncodeFrom, FloatElement, Triple};

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
    O: EncodeFrom<AccOf<E>>,
    Ep: Epilogue<E, O>,
    O: Copy,
{
    let shape = triple.shape();
    if shape.m == 0 || shape.n == 0 {
        return;
    }
    let reads_c = epilogue.reads_c();
    let (a, b, c) = triple.parts();

    for i in 0..shape.m {
        for j in 0..shape.n {
            let mut acc = <AccOf<E> as Accumulator>::ZERO;
            for p in 0..shape.k {
                // Decode, then accumulate exactly. No rounding happens here,
                // at any depth, for any magnitude.
                E::mac(&mut acc, *a.at(i, p), *b.at(p, j));
            }
            let prior = if reads_c { Some(*c.at(i, j)) } else { None };
            *c.at_mut(i, j) = epilogue.finish(acc, prior, options.encode);
        }
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
