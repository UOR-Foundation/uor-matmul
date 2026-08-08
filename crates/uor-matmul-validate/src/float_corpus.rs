//! Symmetric `f32`/`f64` operands for the pure-UOR float gates.
//!
//! The corpus constructs IEEE codes directly. That keeps the two formats on
//! the same structural axes without asking host float arithmetic to manufacture
//! the values whose exact interpretation is under test. The finite patterns
//! cover coherent grades, inverse panel gauges, the whole exponent field, and
//! sparse and dense significands; boundary and non-finite patterns exercise the
//! two public codec boundaries.

use core::fmt::Debug;

use bytemuck::Pod;
use uor_matmul_core::{AccOf, Accumulator, Element, EncodeFrom, EncodeMode, FloatElement};

/// A structural distribution of IEEE codes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FloatFill {
    /// Every finite atom has one grade.
    OneGrade,
    /// A small asymmetric neighborhood of grades.
    FewGrades,
    /// `A` rises as `B` falls, so products stay coherent despite wide panels.
    InverseGauge,
    /// Finite codes span the format's entire exponent field.
    FullFiniteRange,
    /// Normal/subnormal/zero and maximal-finite boundary codes.
    Boundaries,
    /// Zeros, infinities, and NaNs beside ordinary finite codes.
    NonFinite,
    /// Significands with one stored bit, the sparse signed-digit end.
    SparseSignificand,
    /// Significands with nearly every stored bit set.
    DenseSignificand,
}

/// One shape, seed, and structural fill shared by both float formats.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct FloatCase {
    /// Output rows.
    pub m: usize,
    /// Reduction depth.
    pub k: usize,
    /// Output columns.
    pub n: usize,
    /// Deterministic operand distribution.
    pub fill: FloatFill,
    /// Reproduction seed.
    pub seed: u64,
}

/// Fast cases run by the byte differential on every ordinary test invocation.
pub const CORRECTNESS_CASES: &[FloatCase] = &[
    FloatCase {
        m: 0,
        k: 3,
        n: 2,
        fill: FloatFill::OneGrade,
        seed: 1,
    },
    FloatCase {
        m: 2,
        k: 0,
        n: 3,
        fill: FloatFill::Boundaries,
        seed: 2,
    },
    FloatCase {
        m: 1,
        k: 17,
        n: 1,
        fill: FloatFill::FullFiniteRange,
        seed: 3,
    },
    FloatCase {
        m: 3,
        k: 5,
        n: 7,
        fill: FloatFill::FewGrades,
        seed: 4,
    },
    FloatCase {
        m: 5,
        k: 33,
        n: 4,
        fill: FloatFill::InverseGauge,
        seed: 5,
    },
    FloatCase {
        m: 4,
        k: 19,
        n: 3,
        fill: FloatFill::Boundaries,
        seed: 6,
    },
    FloatCase {
        m: 3,
        k: 23,
        n: 4,
        fill: FloatFill::NonFinite,
        seed: 7,
    },
    FloatCase {
        m: 7,
        k: 31,
        n: 5,
        fill: FloatFill::SparseSignificand,
        seed: 8,
    },
    FloatCase {
        m: 7,
        k: 31,
        n: 5,
        fill: FloatFill::DenseSignificand,
        seed: 9,
    },
];

/// The release measurement grid. It is evidence, never an implementation
/// envelope.
///
/// Each structural axis is represented while the largest point stays small
/// enough for the deliberately unoptimized exact reference to produce every
/// expected output byte.  That is a property of the V&V workload, not an
/// admission predicate: none of these dimensions is read by shipped code.
pub const PERFORMANCE_CASES: &[FloatCase] = &[
    FloatCase {
        m: 1,
        k: 1,
        n: 1,
        fill: FloatFill::OneGrade,
        seed: 101,
    },
    FloatCase {
        m: 32,
        k: 32,
        n: 32,
        fill: FloatFill::FewGrades,
        seed: 102,
    },
    FloatCase {
        m: 16,
        k: 128,
        n: 16,
        fill: FloatFill::InverseGauge,
        seed: 103,
    },
    FloatCase {
        m: 7,
        k: 31,
        n: 5,
        fill: FloatFill::FullFiniteRange,
        seed: 104,
    },
    FloatCase {
        m: 1,
        k: 65_536,
        n: 1,
        fill: FloatFill::DenseSignificand,
        seed: 105,
    },
    FloatCase {
        m: 128,
        k: 8,
        n: 128,
        fill: FloatFill::SparseSignificand,
        seed: 106,
    },
];

/// The bit-level operations needed to construct the same corpus at both IEEE
/// interchange widths.
pub trait CorpusFloat: FloatElement + Pod + Copy + Debug + PartialEq {
    /// Stored fraction width.
    const FRACTION_BITS: u32;
    /// Stored exponent width.
    const EXPONENT_BITS: u32;
    /// Build the float code from the low bits of `bits`.
    fn from_corpus_bits(bits: u64) -> Self;
    /// Read the code as an unsigned word.
    fn corpus_bits(self) -> u64;
}

impl CorpusFloat for f32 {
    const FRACTION_BITS: u32 = 23;
    const EXPONENT_BITS: u32 = 8;

    fn from_corpus_bits(bits: u64) -> Self {
        Self::from_bits(bits as u32)
    }

    fn corpus_bits(self) -> u64 {
        u64::from(self.to_bits())
    }
}

impl CorpusFloat for f64 {
    const FRACTION_BITS: u32 = 52;
    const EXPONENT_BITS: u32 = 11;

    fn from_corpus_bits(bits: u64) -> Self {
        Self::from_bits(bits)
    }

    fn corpus_bits(self) -> u64 {
        self.to_bits()
    }
}

/// Construct row-major operands for `case`.
pub fn operands<E: CorpusFloat>(case: FloatCase) -> (Vec<E>, Vec<E>) {
    let mut a = Vec::with_capacity(case.m.saturating_mul(case.k));
    for i in 0..case.m {
        for p in 0..case.k {
            a.push(code::<E>(case, Side::A, i, p));
        }
    }
    let mut b = Vec::with_capacity(case.k.saturating_mul(case.n));
    for p in 0..case.k {
        for j in 0..case.n {
            b.push(code::<E>(case, Side::B, j, p));
        }
    }
    (a, b)
}

/// The deliberately unoptimized exact product for this corpus.
///
/// This calls the element primitive one term at a time, so an optimized
/// panel/group/lookup traversal cannot become its own oracle. For floats that
/// primitive is the canonical finite-NAF Atlas reference in `uor-matmul-core`;
/// the only information-losing step is the requested encode below.
pub fn exact_product<E>(case: FloatCase, a: &[E], b: &[E], mode: EncodeMode) -> Vec<E>
where
    E: CorpusFloat + EncodeFrom<AccOf<E>>,
{
    let mut out = Vec::with_capacity(case.m.saturating_mul(case.n));
    for i in 0..case.m {
        for j in 0..case.n {
            let mut acc = <AccOf<E> as Accumulator>::ZERO;
            for p in 0..case.k {
                <E as Element>::mac(&mut acc, a[i * case.k + p], b[p * case.n + j]);
            }
            out.push(<E as EncodeFrom<AccOf<E>>>::encode_from(acc, mode));
        }
    }
    out
}

#[derive(Clone, Copy)]
enum Side {
    A,
    B,
}

fn code<E: CorpusFloat>(case: FloatCase, side: Side, outer: usize, p: usize) -> E {
    let side_salt = match side {
        Side::A => 0xA24B_AED4_963E_E407,
        Side::B => 0x9FB2_1C65_1E98_DF25,
    };
    let random = mix(case.seed
        ^ side_salt
        ^ (outer as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93)
        ^ (p as u64).wrapping_mul(0xA5A3_564E_27F8_862D));
    let frac_mask = low_mask(E::FRACTION_BITS);
    let exponent_max = low_mask(E::EXPONENT_BITS);
    let bias = exponent_max >> 1;
    let sign = (random >> 63) << (E::FRACTION_BITS + E::EXPONENT_BITS);

    let (exponent, fraction) = match case.fill {
        FloatFill::OneGrade => (bias, random & frac_mask),
        FloatFill::FewGrades => {
            let delta = (random % 7) as i64 - 3;
            ((bias as i64 + delta) as u64, random & frac_mask)
        }
        FloatFill::InverseGauge => {
            let radius = (exponent_max / 8).max(1);
            let delta = (p as u64 % (2 * radius + 1)) as i64 - radius as i64;
            let signed = match side {
                Side::A => delta,
                Side::B => -delta,
            };
            ((bias as i64 + signed) as u64, random & frac_mask)
        }
        FloatFill::FullFiniteRange => {
            let finite_exponents = exponent_max - 1;
            (1 + random % finite_exponents, random & frac_mask)
        }
        FloatFill::Boundaries => match random % 6 {
            0 => (0, 0),
            1 => (0, 1),
            2 => (0, frac_mask),
            3 => (1, 0),
            4 => (bias, 0),
            _ => (exponent_max - 1, frac_mask),
        },
        FloatFill::NonFinite => match random % 6 {
            0 => (0, 0),
            1 => (exponent_max, 0),
            2 => (exponent_max, 1),
            3 => (exponent_max, frac_mask),
            _ => (bias, random & frac_mask),
        },
        FloatFill::SparseSignificand => {
            let bit = random as u32 % E::FRACTION_BITS;
            (bias, 1u64 << bit)
        }
        FloatFill::DenseSignificand => {
            (bias, frac_mask ^ (1 << (random % E::FRACTION_BITS as u64)))
        }
    };
    E::from_corpus_bits(sign | (exponent << E::FRACTION_BITS) | fraction)
}

const fn low_mask(bits: u32) -> u64 {
    u64::MAX >> (u64::BITS - bits)
}

fn mix(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_formats_walk_the_same_cases() {
        for case in CORRECTNESS_CASES {
            let (a32, b32) = operands::<f32>(*case);
            let (a64, b64) = operands::<f64>(*case);
            assert_eq!(a32.len(), case.m * case.k);
            assert_eq!(a64.len(), a32.len());
            assert_eq!(b32.len(), case.k * case.n);
            assert_eq!(b64.len(), b32.len());
        }
    }

    #[test]
    fn inverse_gauge_really_is_inverse() {
        let case = FloatCase {
            m: 1,
            k: 41,
            n: 1,
            fill: FloatFill::InverseGauge,
            seed: 19,
        };
        let (a, b) = operands::<f32>(case);
        for (a, b) in a.into_iter().zip(b) {
            let ae = (a.to_bits() >> 23) & 0xff;
            let be = (b.to_bits() >> 23) & 0xff;
            assert_eq!(ae + be, 2 * 127);
        }
    }
}
