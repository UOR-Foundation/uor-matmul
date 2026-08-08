//! The finite, pre-symbolic Atlas reference algebra.
//!
//! This module is deliberately private.  It fixes the objects consumed by the
//! float traversals independently of optimization: the canonical
//! Laurent representative of a finite dyadic value, its Atlas address, the
//! exact four-block carrier decomposition, and the gauge intervals on which a
//! product may be factored.  Keeping the reference here makes every native
//! traversal an optimization of this object rather than a second arithmetic.

#![cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "CK-19, CK-20 and CD-31 are executable theorem witnesses; the portable float reference consumes finite_product while the carrier and gauge certificates remain private"
    )
)]

use crate::acc::Complete;
use crate::alphabet::Decoded;
use crate::generated::atlas;

// These casts are lossless on every supported target: the sources are `u32`,
// `i64`/`i128` are wider, and this workspace's address type is at least `u32`.
const ATLAS_MODALITY: usize = atlas::MODALITY as usize;
const ATLAS_CONTEXT: usize = atlas::CONTEXT as usize;
const ATLAS_CARRIER_DIM: usize = atlas::CARRIER_DIM as usize;
const ATLAS_GRADE_PERIOD: i128 = atlas::PAGE_SITES as i128;
const ATLAS_INPUT_SCALE: i128 = atlas::PROJECTOR_INPUT_SCALE as i128;
const ATLAS_DYADIC_DENOMINATOR: i128 = atlas::PROJECTOR_DENOMINATOR as i128;
const ATLAS_PROJECTOR_DENOMINATOR: i128 =
    atlas::PROJECTOR_INPUT_SCALE as i128 * atlas::PROJECTOR_DENOMINATOR as i128;

/// A coefficient in the canonical non-adjacent form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AtlasDigit {
    Negative,
    Zero,
    Positive,
}

impl AtlasDigit {
    const fn coefficient(self) -> i8 {
        match self {
            Self::Negative => -1,
            Self::Zero => 0,
            Self::Positive => 1,
        }
    }

    /// The Atlas involution `mu(d) = 2 - d`: negation before symbols exist.
    const fn mu(self) -> Self {
        match self {
            Self::Negative => Self::Positive,
            Self::Zero => Self::Zero,
            Self::Positive => Self::Negative,
        }
    }

    /// The modality coordinate used by the canonical carrier embedding.
    const fn modality(self) -> usize {
        match self {
            Self::Negative => 0,
            Self::Zero => 1,
            Self::Positive => 2,
        }
    }

    /// Multiplication in `{-1, 0, +1}`, written as its complete table.
    const fn product(self, other: Self) -> Self {
        match (self, other) {
            (Self::Zero, _) | (_, Self::Zero) => Self::Zero,
            (Self::Negative, Self::Negative) | (Self::Positive, Self::Positive) => Self::Positive,
            (Self::Negative, Self::Positive) | (Self::Positive, Self::Negative) => Self::Negative,
        }
    }
}

/// The canonical finite section of `Z[X, X^-1] / (X - 2)`.
///
/// A nonzero value is `(-1)^negative * unit * 2^grade`, where `unit` is odd.
/// Its digits are produced lazily, so neither significand width nor Laurent
/// grade introduces a storage ceiling.  Zero has the unique section
/// `(false, 0, 0)`; the sign of an IEEE zero belongs to symbol identity, not to
/// the value in this quotient.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FiniteNaf {
    negative: bool,
    unit: u64,
    grade: i64,
}

impl FiniteNaf {
    /// Take the canonical section of one finite IEEE decode.
    pub(crate) fn from_decoded(value: Decoded) -> Option<Self> {
        match value {
            Decoded::Finite {
                sign,
                mantissa,
                exp,
            } => Some(Self::from_scaled(sign, mantissa, i64::from(exp))),
            Decoded::Infinite { .. } | Decoded::NotANumber => None,
        }
    }

    /// Normalize a signed integer coefficient at one Laurent grade.
    #[allow(
        clippy::manual_is_multiple_of,
        reason = "the explicit remainder is the canonical radix-2 valuation recurrence audited by CU-11"
    )]
    fn from_scaled(negative: bool, magnitude: u64, grade: i64) -> Self {
        if magnitude == 0 {
            return Self {
                negative: false,
                unit: 0,
                grade: 0,
            };
        }
        let mut unit = magnitude;
        let mut valuation = 0i64;
        while unit % 2 == 0 {
            unit /= 2;
            valuation += 1;
        }
        Self {
            negative,
            unit,
            grade: grade + valuation,
        }
    }

    /// Every NAF coordinate, including the zeros separating nonzero atoms.
    pub(crate) fn digits(self) -> NafDigits {
        NafDigits {
            rest: u128::from(self.unit),
            grade: self.grade,
            negative: self.negative,
        }
    }

    /// Only the nonzero Laurent atoms.
    pub(crate) fn atoms(self) -> NafAtoms {
        NafAtoms {
            digits: self.digits(),
        }
    }

    /// The inclusive support interval of the canonical polynomial.
    pub(crate) fn support(self) -> Option<GradeInterval> {
        let mut atoms = self.atoms();
        let first = atoms.next()?;
        let mut high = first.grade;
        for atom in atoms {
            high = atom.grade;
        }
        Some(GradeInterval {
            low: first.grade,
            high,
        })
    }

    /// Negation is exactly the Atlas `mu` involution on every digit.
    pub(crate) fn negated(self) -> Self {
        if self.unit == 0 {
            self
        } else {
            Self {
                negative: !self.negative,
                ..self
            }
        }
    }

    /// The coefficient at a zero-based NAF position, padding past the word by
    /// the canonical zero digit.
    pub(crate) fn digit_at(self, position: u32) -> AtlasDigit {
        self.digits()
            .nth(position as usize)
            .map_or(AtlasDigit::Zero, |digit| digit.digit)
    }
}

/// Accumulate one finite product through the canonical Atlas atoms.
///
/// Both values are normalized and decomposed independently.  Their atom
/// product is the complete ternary table, and each resulting Laurent monomial
/// is placed in the complete accumulator.  There is no significand multiply
/// hidden behind the reference: this deliberately unoptimized body is the
/// identity against which blocked Atlas traversals are compared (R6).
pub(crate) fn finite_product<const L: usize, const MIN_EXP: i32>(
    acc: &mut Complete<L, MIN_EXP>,
    a: (bool, u64, i32),
    b: (bool, u64, i32),
) {
    let a = FiniteNaf::from_scaled(a.0, a.1, i64::from(a.2));
    let b = FiniteNaf::from_scaled(b.0, b.1, i64::from(b.2));
    for left in a.atoms() {
        for right in b.atoms() {
            let digit = left.digit.product(right.digit);
            let grade = left.grade + right.grade;
            acc.add_scaled_i64(1, grade, digit == AtlasDigit::Negative);
        }
    }
}

/// One coordinate of a canonical Laurent word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LaurentDigit {
    pub(crate) digit: AtlasDigit,
    pub(crate) grade: i64,
}

impl LaurentDigit {
    /// Its lossless Atlas coordinate, before any packed symbol is chosen.
    pub(crate) fn address(self) -> AtlasAddress {
        let grade = GradeAddress::of(self.grade);
        AtlasAddress {
            revolution: grade.revolution,
            scope: grade.scope,
            modality: u32::try_from(self.digit.modality())
                .expect("the finite NAF modality is a model coordinate"),
            context: grade.context,
        }
    }
}

/// Lazy canonical NAF digits.  `u128` is the carry workspace, not a value
/// bound: it is the derived extra bit needed when `u64::MAX` emits `-1` first.
#[derive(Clone, Debug)]
pub(crate) struct NafDigits {
    rest: u128,
    grade: i64,
    negative: bool,
}

impl Iterator for NafDigits {
    type Item = LaurentDigit;

    #[allow(
        clippy::manual_is_multiple_of,
        clippy::manual_div_ceil,
        reason = "the explicit quotient/remainder equations are the NAF recurrence audited by CU-11"
    )]
    fn next(&mut self) -> Option<Self::Item> {
        if self.rest == 0 {
            return None;
        }
        let digit = if self.rest % 2 == 0 {
            AtlasDigit::Zero
        } else if self.rest % 4 == 1 {
            AtlasDigit::Positive
        } else {
            AtlasDigit::Negative
        };
        self.rest = match digit {
            AtlasDigit::Negative => (self.rest + 1) / 2,
            AtlasDigit::Zero => self.rest / 2,
            AtlasDigit::Positive => (self.rest - 1) / 2,
        };
        let out = LaurentDigit {
            digit: if self.negative { digit.mu() } else { digit },
            grade: self.grade,
        };
        self.grade += 1;
        Some(out)
    }
}

/// The nonzero subsequence of [`NafDigits`].
#[derive(Clone, Debug)]
pub(crate) struct NafAtoms {
    digits: NafDigits,
}

impl Iterator for NafAtoms {
    type Item = LaurentDigit;

    fn next(&mut self) -> Option<Self::Item> {
        self.digits
            .by_ref()
            .find(|digit| digit.digit != AtlasDigit::Zero)
    }
}

/// The quotient and remainder of an arbitrary Laurent grade.
///
/// Euclidean division, rather than truncating division, makes the same identity
/// hold for negative grades: `g = 32r + 8h + l`, `0 <= h < 4`, `0 <= l < 8`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GradeAddress {
    pub(crate) revolution: i128,
    pub(crate) scope: u32,
    pub(crate) context: u32,
}

impl GradeAddress {
    pub(crate) fn of(grade: i64) -> Self {
        let grade = i128::from(grade);
        let revolution = grade.div_euclid(ATLAS_GRADE_PERIOD);
        let within = grade.rem_euclid(ATLAS_GRADE_PERIOD);
        Self {
            revolution,
            scope: u32::try_from(within / ATLAS_CONTEXT as i128)
                .expect("an address scope is bounded by the model's u32 source"),
            context: u32::try_from(within % ATLAS_CONTEXT as i128)
                .expect("an address context is bounded by the model's u32 source"),
        }
    }

    /// Reconstruct in a wider integer so even the extreme `i64` grades are
    /// checked without an intermediate overflow.
    pub(crate) fn grade(self) -> i128 {
        self.revolution * ATLAS_GRADE_PERIOD
            + i128::from(self.scope) * ATLAS_CONTEXT as i128
            + i128::from(self.context)
    }
}

/// One pre-symbolic Atlas coordinate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AtlasAddress {
    pub(crate) revolution: i128,
    pub(crate) scope: u32,
    pub(crate) modality: u32,
    pub(crate) context: u32,
}

/// The caller-owned source of one vector in the canonical lattice.
///
/// A word is kept in its canonical eight-site spelling and a general lattice
/// is kept in its caller-provided spelling.  The distinction is a view choice,
/// not a second carrier representation: both are read through
/// [`AtlasCarrier::coordinate`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AtlasCarrierSource<'a> {
    Lattice(&'a [i64; ATLAS_CARRIER_DIM]),
    Word(&'a [AtlasDigit; ATLAS_CONTEXT]),
}

/// A borrowed, lazy view of one vector in the canonical lattice
/// `3 Z^(3 x 8)`.
///
/// The view owns only a reference and its source tag.  Scaling into the
/// ambient dyadic module and the sparse word embedding happen when a
/// coordinate is observed, so neither construction nor projection copies a
/// carrier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AtlasCarrier<'a> {
    source: AtlasCarrierSource<'a>,
}

impl<'a> AtlasCarrier<'a> {
    /// Embed lattice coordinates.  The factor three is structural: it cancels
    /// the modality denominator, leaving every spectral block in the ambient
    /// dyadic module.
    pub(crate) const fn from_lattice(values: &'a [i64; ATLAS_CARRIER_DIM]) -> Self {
        Self {
            source: AtlasCarrierSource::Lattice(values),
        }
    }

    /// The canonical eight-site embedding
    /// `E(s) = sum_l 3(l+1) e_(s_l+1) tensor e_l`.
    pub(crate) const fn embed_word(word: &'a [AtlasDigit; ATLAS_CONTEXT]) -> Self {
        Self {
            source: AtlasCarrierSource::Word(word),
        }
    }

    /// Observe one ambient-module coordinate without materializing the sparse
    /// word embedding or a scaled lattice.
    pub(crate) fn coordinate(self, index: usize) -> i128 {
        match self.source {
            AtlasCarrierSource::Lattice(values) => i128::from(values[index]) * ATLAS_INPUT_SCALE,
            AtlasCarrierSource::Word(word) => {
                let modality = index / ATLAS_CONTEXT;
                let context = index % ATLAS_CONTEXT;
                if word[context].modality() == modality {
                    (context as i128 + 1) * ATLAS_INPUT_SCALE
                } else {
                    0
                }
            }
        }
    }

    /// Apply the four exact spectral projectors.  All returned numerators have
    /// denominator eight; equivalently their common Laurent exponent is `-3`.
    pub(crate) const fn project(self) -> AtlasBlocks<'a> {
        AtlasBlocks { carrier: self }
    }
}

/// A lazy view of the four exact block numerators.
///
/// The common dyadic denominator is a property of this type, not data repeated
/// beside each coordinate.  Each observation is derived directly from the
/// borrowed carrier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AtlasBlocks<'a> {
    carrier: AtlasCarrier<'a>,
}

impl AtlasBlocks<'_> {
    /// Observe one block numerator at the common implicit denominator.
    pub(crate) fn value(self, block: AtlasBlock, index: usize) -> i128 {
        let out_modality = index / ATLAS_CONTEXT;
        let out_context = index % ATLAS_CONTEXT;
        let total = (0..ATLAS_CARRIER_DIM)
            .map(|coordinate| self.carrier.coordinate(coordinate))
            .sum::<i128>();
        // Every carrier coordinate is in 3Z, so these divisions are exact.
        let third_total = total / ATLAS_INPUT_SCALE;
        let row = (0..ATLAS_CONTEXT)
            .map(|context| {
                self.carrier
                    .coordinate(out_modality * ATLAS_CONTEXT + context)
            })
            .sum::<i128>();
        let third_column = (0..ATLAS_MODALITY)
            .map(|modality| {
                self.carrier
                    .coordinate(modality * ATLAS_CONTEXT + out_context)
            })
            .sum::<i128>()
            / ATLAS_INPUT_SCALE;
        let global = third_total;
        let modality = row - third_total;
        let context = ATLAS_DYADIC_DENOMINATOR * third_column - third_total;
        match block {
            AtlasBlock::Global => global,
            AtlasBlock::Modality => modality,
            AtlasBlock::Context => context,
            AtlasBlock::Interaction => {
                ATLAS_DYADIC_DENOMINATOR * self.carrier.coordinate(index)
                    - global
                    - modality
                    - context
            }
        }
    }

    /// Reconstruct one carrier coordinate, still at the implicit denominator.
    pub(crate) fn reconstruction_at(self, index: usize) -> i128 {
        [
            AtlasBlock::Global,
            AtlasBlock::Modality,
            AtlasBlock::Context,
            AtlasBlock::Interaction,
        ]
        .into_iter()
        .map(|block| self.value(block, index))
        .sum()
    }
}

/// One of the four canonical Atlas blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AtlasBlock {
    Global,
    Modality,
    Context,
    Interaction,
}

/// The numerator of a projector entry at the common denominator 24.
const fn projector_numerator(block: AtlasBlock, out: usize, input: usize) -> i128 {
    let out_modality = out / ATLAS_CONTEXT;
    let out_context = out % ATLAS_CONTEXT;
    let in_modality = input / ATLAS_CONTEXT;
    let in_context = input % ATLAS_CONTEXT;
    let modality = if out_modality == in_modality {
        ATLAS_MODALITY as i128 - 1
    } else {
        -1
    };
    let context = if out_context == in_context {
        ATLAS_CONTEXT as i128 - 1
    } else {
        -1
    };
    match block {
        AtlasBlock::Global => 1,
        AtlasBlock::Modality => modality,
        AtlasBlock::Context => context,
        AtlasBlock::Interaction => modality * context,
    }
}

/// A nonempty inclusive interval of Laurent grades.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GradeInterval {
    pub(crate) low: i64,
    pub(crate) high: i64,
}

impl GradeInterval {
    pub(crate) const fn new(low: i64, high: i64) -> Option<Self> {
        if low <= high {
            Some(Self { low, high })
        } else {
            None
        }
    }

    pub(crate) const fn contains(self, grade: i64) -> bool {
        self.low <= grade && grade <= self.high
    }

    /// `J_p = [A_H + B_H, A_0 + B_0]`, the exact Minkowski support interval.
    pub(crate) fn product(self, other: Self) -> Self {
        // Finite decodes have i32 exponents and at most 65 NAF positions, so
        // both sums are strictly inside i64 by construction.
        Self {
            low: self.low + other.low,
            high: self.high + other.high,
        }
    }

    /// The admissible translation interval at product gauge `q`:
    /// `[max(-A_0, B_H-Q), min(-A_H, B_0-Q)]`.
    pub(crate) fn translations(self, other: Self, q: i64) -> Option<Self> {
        Self::new(
            (-self.high).max(other.low - q),
            (-self.low).min(other.high - q),
        )
    }

    /// The greatest gauge carried by this product interval.  For an
    /// intersecting group the same operation on the intersection is the
    /// greatest common gauge and therefore the group with greatest headroom.
    pub(crate) const fn greatest(self) -> i64 {
        self.high
    }
}

/// The minimum right-endpoint gauges for an arbitrary family of intervals.
///
/// The iterator performs conceptual sorting by repeated scans, so it neither
/// mutates the input nor needs storage proportional to its length.  After a
/// point `p` is chosen, every still-unhit interval starts to the right of `p`;
/// choosing the least right endpoint among them is the classical exchange
/// argument for a minimum interval stabbing set.
#[derive(Clone, Debug)]
pub(crate) struct MinimumGauges<'a> {
    intervals: &'a [GradeInterval],
    after: Option<i64>,
}

impl<'a> MinimumGauges<'a> {
    pub(crate) const fn new(intervals: &'a [GradeInterval]) -> Self {
        Self {
            intervals,
            after: None,
        }
    }
}

impl Iterator for MinimumGauges<'_> {
    type Item = i64;

    fn next(&mut self) -> Option<Self::Item> {
        let next = self
            .intervals
            .iter()
            .filter(|interval| self.after.is_none_or(|point| interval.low > point))
            .map(|interval| interval.high)
            .min()?;
        self.after = Some(next);
        Some(next)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphabet::{Element, FloatElement};

    fn evaluated(section: FiniteNaf) -> i128 {
        let Some(support) = section.support() else {
            return 0;
        };
        section.atoms().fold(0i128, |sum, atom| {
            let shift = (atom.grade - support.low) as u32;
            sum + (i128::from(atom.digit.coefficient()) << shift)
        })
    }

    /// `CK-19`: finite dyadics have one evaluation-preserving, idempotent NAF
    /// section, and `mu` is its exact negation.
    #[test]
    fn finite_naf_is_the_canonical_laurent_section_ck_19() {
        let cases = [
            (false, 0, -1_074),
            (true, 0, 971),
            (false, 1, -1_074),
            (true, 2, -149),
            (false, 79, 0),
            (false, 135, 0),
            (true, u64::MAX, i32::MAX),
        ];
        for (negative, mantissa, exponent) in cases {
            let section = FiniteNaf::from_decoded(Decoded::Finite {
                sign: negative,
                mantissa,
                exp: exponent,
            })
            .unwrap();
            if mantissa == 0 {
                assert_eq!(section, FiniteNaf::from_scaled(false, 0, 0));
                assert_eq!(evaluated(section), 0);
                continue;
            }

            let value = evaluated(section);
            let expected = if negative {
                -i128::from(section.unit)
            } else {
                i128::from(section.unit)
            };
            assert_eq!(value, expected, "the Laurent evaluation is unchanged");

            let second =
                FiniteNaf::from_scaled(value < 0, value.unsigned_abs() as u64, section.grade);
            assert_eq!(second, section, "taking the section is idempotent");

            let mut prior_nonzero = false;
            for digit in section.digits() {
                let nonzero = digit.digit != AtlasDigit::Zero;
                assert!(!(prior_nonzero && nonzero), "NAF has adjacent atoms");
                assert!(matches!(digit.digit.coefficient(), -1..=1));
                prior_nonzero = nonzero;
            }

            let atoms = section.atoms();
            let negated = section.negated().atoms();
            for (a, b) in atoms.zip(negated) {
                assert_eq!(a.grade, b.grade);
                assert_eq!(a.digit.mu(), b.digit);
                assert_eq!(2 - a.digit.modality(), b.digit.modality());
            }
        }

        // Exercise the real decoders as the boundary, including a subnormal.
        for bits in [1u32, 0x3f80_0000, 0x8000_0001, 0x7f7f_ffff] {
            let decoded = <f32 as FloatElement>::decode(f32::from_bits(bits));
            assert!(FiniteNaf::from_decoded(decoded).is_some());
        }
        assert!(FiniteNaf::from_decoded(Decoded::NotANumber).is_none());

        // The production reference consumes these same atoms.  Compare its
        // complete placement with an independent whole-product oracle at both
        // IEEE significand boundaries, including the carry digit above bit 63.
        for (a, b) in [
            ((false, 0, -1_074), (true, u64::MAX >> 11, -1_074)),
            (
                (false, (1u64 << 24) - 1, -149),
                (true, (1u64 << 24) - 1, 104),
            ),
            ((false, u64::MAX, 0), (true, u64::MAX, 0)),
            (
                (true, (1u64 << 53) - 1, -1_074),
                (true, (1u64 << 53) - 1, 971),
            ),
            (
                (false, (1u64 << 53) - 1, 971),
                (true, (1u64 << 53) - 1, 971),
            ),
        ] {
            let mut got = <f64 as Element>::Acc::ZERO;
            finite_product(&mut got, a, b);
            let mut want = <f64 as Element>::Acc::ZERO;
            want.add_scaled(u128::from(a.1) * u128::from(b.1), a.2 + b.2, a.0 != b.0);
            assert_eq!(got, want);
        }

        // Exhaust all short signed non-adjacent words.  No two distinct words
        // evaluate to the same Laurent coefficient, which is the uniqueness
        // half of the section rather than merely a round-trip through itself.
        let mut seen = [false; 2_049];
        let word_count = (ATLAS_MODALITY as u32).pow(10);
        for encoded in 0..word_count {
            let mut code = encoded;
            let mut value = 0i32;
            let mut place = 1i32;
            let mut prior_nonzero = false;
            let mut valid = true;
            for _ in 0..10 {
                let digit = (code % ATLAS_MODALITY as u32) as i32 - 1;
                code /= ATLAS_MODALITY as u32;
                if digit != 0 && prior_nonzero {
                    valid = false;
                    break;
                }
                prior_nonzero = digit != 0;
                value += digit * place;
                place *= 2;
            }
            if valid {
                let slot = (value + 1_024) as usize;
                assert!(!seen[slot], "two canonical words evaluated to {value}");
                seen[slot] = true;
            }
        }
    }

    /// `CK-19`: Laurent grades have an exact Atlas address at every signed
    /// revolution; no finite address-depth or wraparound convention exists.
    #[test]
    fn every_laurent_grade_has_an_unbounded_atlas_address_ck_19() {
        let page = i64::try_from(ATLAS_GRADE_PERIOD)
            .expect("the canonical address page fits one Laurent revolution");
        let context = atlas::CONTEXT as i64;
        for grade in [
            i64::MIN,
            i64::MIN + page - 1,
            -1_000_003,
            -page - 1,
            -page,
            -page + 1,
            -1,
            0,
            context - 1,
            context,
            page - 1,
            page,
            1_000_003,
            i64::MAX - page + 1,
            i64::MAX,
        ] {
            let address = GradeAddress::of(grade);
            assert!(address.scope < atlas::SCOPE);
            assert!(address.context < atlas::CONTEXT);
            assert_eq!(address.grade(), i128::from(grade));
        }

        let section = FiniteNaf::from_scaled(false, u64::MAX, i64::from(i32::MAX));
        for atom in section.atoms() {
            let address = atom.address();
            assert_eq!(address.modality, atom.digit.modality() as u32);
            assert_eq!(address.context, GradeAddress::of(atom.grade).context);
        }
    }

    fn block_values(blocks: AtlasBlocks<'_>, block: AtlasBlock) -> [i128; ATLAS_CARRIER_DIM] {
        core::array::from_fn(|index| blocks.value(block, index))
    }

    /// `CA-05`: carrier and projector construction preserves the exact
    /// caller-owned backing object.  The size checks make this witness fail if
    /// either view regresses to an inline coordinate or block materialization.
    #[test]
    fn carrier_and_projector_preserve_the_caller_backing_address_ca_05() {
        let lattice = core::array::from_fn(|index| index as i64 - 11);
        let carrier = AtlasCarrier::from_lattice(&lattice);
        let AtlasCarrierSource::Lattice(backing) = carrier.source else {
            panic!("a lattice view changed source kind");
        };
        assert!(core::ptr::eq(backing, &lattice));

        let projected = carrier.project();
        let AtlasCarrierSource::Lattice(projected_backing) = projected.carrier.source else {
            panic!("projection changed a lattice view's source kind");
        };
        assert!(core::ptr::eq(projected_backing, &lattice));

        let word = core::array::from_fn(|index| match index % ATLAS_MODALITY {
            0 => AtlasDigit::Negative,
            1 => AtlasDigit::Zero,
            _ => AtlasDigit::Positive,
        });
        let embedded = AtlasCarrier::embed_word(&word);
        let AtlasCarrierSource::Word(word_backing) = embedded.source else {
            panic!("a word view changed source kind");
        };
        assert!(core::ptr::eq(word_backing, &word));
        let AtlasCarrierSource::Word(projected_word_backing) = embedded.project().carrier.source
        else {
            panic!("projection changed a word view's source kind");
        };
        assert!(core::ptr::eq(projected_word_backing, &word));

        assert!(
            core::mem::size_of::<AtlasCarrier<'_>>() <= 2 * core::mem::size_of::<usize>(),
            "a carrier view contains more than a source reference and its tag"
        );
        assert_eq!(
            core::mem::size_of::<AtlasBlocks<'_>>(),
            core::mem::size_of::<AtlasCarrier<'_>>(),
            "a projector view added materialized block storage"
        );
    }

    /// `CK-20`: the four exact projectors resolve the carrier, and the 79/135
    /// witness is carried precisely by `P3` after its equal marginals vanish.
    #[test]
    fn the_exact_atlas_carrier_resolves_and_p3_separates_ck_20() {
        assert!(
            atlas::PROJECTOR_DENOMINATOR.is_power_of_two(),
            "the projected carrier lies in the ambient dyadic module"
        );
        let blocks = [
            AtlasBlock::Global,
            AtlasBlock::Modality,
            AtlasBlock::Context,
            AtlasBlock::Interaction,
        ];

        // Exact projector algebra at common denominator 24: Pk^2 = Pk,
        // PkPl = 0, and sum Pk = I.
        for out in 0..ATLAS_CARRIER_DIM {
            for input in 0..ATLAS_CARRIER_DIM {
                let sum: i128 = blocks
                    .iter()
                    .map(|&block| projector_numerator(block, out, input))
                    .sum();
                assert_eq!(
                    sum,
                    if out == input {
                        ATLAS_PROJECTOR_DENOMINATOR
                    } else {
                        0
                    }
                );
                for &left in &blocks {
                    for &right in &blocks {
                        let composed: i128 = (0..ATLAS_CARRIER_DIM)
                            .map(|middle| {
                                projector_numerator(left, out, middle)
                                    * projector_numerator(right, middle, input)
                            })
                            .sum();
                        let expected = if left == right {
                            ATLAS_PROJECTOR_DENOMINATOR * projector_numerator(left, out, input)
                        } else {
                            0
                        };
                        assert_eq!(composed, expected);
                    }
                }
            }
        }

        let a = FiniteNaf::from_scaled(false, 79, 0);
        let b = FiniteNaf::from_scaled(false, 135, 0);
        let word_a = core::array::from_fn(|i| a.digit_at(i as u32));
        let word_b = core::array::from_fn(|i| b.digit_at(i as u32));
        assert_eq!(
            word_a.map(AtlasDigit::coefficient),
            [-1, 0, 0, 0, 1, 0, 1, 0]
        );
        assert_eq!(
            word_b.map(AtlasDigit::coefficient),
            [-1, 0, 0, 1, 0, 0, 0, 1]
        );
        let eval_word = |word: [AtlasDigit; ATLAS_CONTEXT]| {
            word.into_iter().enumerate().fold(0i128, |sum, (i, d)| {
                sum + (i128::from(d.coefficient()) << i)
            })
        };
        assert_eq!(eval_word(word_a), 79);
        assert_eq!(eval_word(word_b), 135);

        let carrier_a = AtlasCarrier::embed_word(&word_a);
        let carrier_b = AtlasCarrier::embed_word(&word_b);
        let projected_a = carrier_a.project();
        let projected_b = carrier_b.project();
        assert_eq!(
            block_values(projected_a, AtlasBlock::Global),
            block_values(projected_b, AtlasBlock::Global)
        );
        assert_eq!(
            block_values(projected_a, AtlasBlock::Modality),
            block_values(projected_b, AtlasBlock::Modality)
        );
        assert_eq!(
            block_values(projected_a, AtlasBlock::Context),
            block_values(projected_b, AtlasBlock::Context)
        );
        assert_ne!(
            block_values(projected_a, AtlasBlock::Interaction),
            block_values(projected_b, AtlasBlock::Interaction)
        );
        for index in 0..ATLAS_CARRIER_DIM {
            assert_eq!(
                projected_a.reconstruction_at(index),
                carrier_a.coordinate(index) * ATLAS_DYADIC_DENOMINATOR
            );
            assert_eq!(
                projected_b.reconstruction_at(index),
                carrier_b.coordinate(index) * ATLAS_DYADIC_DENOMINATOR
            );
        }

        // The difference is a load-bearing P3 witness, not merely two vectors
        // that happen to differ somewhere in the decomposition.
        let difference_lattice = core::array::from_fn(|index| {
            i64::try_from(
                (carrier_b.coordinate(index) - carrier_a.coordinate(index)) / ATLAS_INPUT_SCALE,
            )
            .unwrap()
        });
        let difference = AtlasCarrier::from_lattice(&difference_lattice);
        let projected_difference = difference.project();
        assert_eq!(
            block_values(projected_difference, AtlasBlock::Global),
            [0; ATLAS_CARRIER_DIM]
        );
        assert_eq!(
            block_values(projected_difference, AtlasBlock::Modality),
            [0; ATLAS_CARRIER_DIM]
        );
        assert_eq!(
            block_values(projected_difference, AtlasBlock::Context),
            [0; ATLAS_CARRIER_DIM]
        );
        assert_ne!(
            block_values(projected_difference, AtlasBlock::Interaction),
            [0; ATLAS_CARRIER_DIM]
        );
    }

    fn brute_minimum(intervals: &[GradeInterval]) -> usize {
        for count in 0..=5usize {
            for chosen in 0..32u32 {
                if chosen.count_ones() as usize != count {
                    continue;
                }
                if intervals.iter().all(|interval| {
                    (0..5).any(|point| chosen & (1 << point) != 0 && interval.contains(point))
                }) {
                    return count;
                }
            }
        }
        intervals.len()
    }

    /// `CD-31`: product supports, translation supports, and the unbounded
    /// right-endpoint grouping are exact and minimal.
    #[test]
    fn gauge_intervals_have_the_minimum_exact_grouping_cd_31() {
        let a = GradeInterval::new(-7, 4).unwrap();
        let b = GradeInterval::new(2, 11).unwrap();
        let product = a.product(b);
        assert_eq!(product, GradeInterval::new(-5, 15).unwrap());
        assert_eq!(product.greatest(), 15, "one interval takes A0+B0");

        for q in product.low..=product.high {
            let translations = a.translations(b, q).unwrap();
            assert_eq!(
                translations,
                GradeInterval::new((-a.high).max(b.low - q), (-a.low).min(b.high - q)).unwrap()
            );
            let t = translations.greatest();
            // The right endpoint maximizes A's high-side headroom; moving left
            // spends exactly the same amount on A that it returns to B.
            let a_headroom = a.high + t;
            let b_headroom = b.high - q - t;
            for candidate in translations.low..=translations.high {
                assert!(a.high + candidate <= a_headroom);
                assert_eq!(
                    (a.high + candidate) + (b.high - q - candidate),
                    a_headroom + b_headroom
                );
            }
            assert!(a.low + t <= 0 && 0 <= a.high + t);
            assert!(b.low - q - t <= 0 && 0 <= b.high - q - t);
        }
        assert!(a.translations(b, product.low - 1).is_none());
        assert!(a.translations(b, product.high + 1).is_none());

        let greatest_headroom = product.greatest() - product.low;
        for candidate in product.low..=product.greatest() {
            assert!(candidate - product.low <= greatest_headroom);
        }

        // Independent direct admission over every small support interval.  A
        // translation is admitted exactly when both shifted supports contain
        // their respective origins; this catches every sign or endpoint swap
        // in the closed-form interval above.
        for a_low in -3..=3 {
            for a_high in a_low..=3 {
                let a = GradeInterval::new(a_low, a_high).unwrap();
                for b_low in -3..=3 {
                    for b_high in b_low..=3 {
                        let b = GradeInterval::new(b_low, b_high).unwrap();
                        let product = a.product(b);
                        for q in product.low - 1..=product.high + 1 {
                            let derived = a.translations(b, q);
                            assert_eq!(derived.is_some(), product.contains(q));
                            let scan_low = (-a.high).min(b.low - q) - 1;
                            let scan_high = (-a.low).max(b.high - q) + 1;
                            for t in scan_low..=scan_high {
                                let direct = a.low + t <= 0
                                    && 0 <= a.high + t
                                    && b.low - q - t <= 0
                                    && 0 <= b.high - q - t;
                                assert_eq!(
                                    derived.is_some_and(|interval| interval.contains(t)),
                                    direct,
                                    "A={a:?}, B={b:?}, Q={q}, t={t}"
                                );
                            }
                        }
                    }
                }
            }
        }

        // Exhaust every family drawn from the integer intervals on [0,4].
        // Input order is deliberately reversed; the greedy iterator is a
        // conceptual sort and must not rely on pre-grouped operands.
        let all = core::array::from_fn::<_, 15, _>(|index| {
            let mut seen = 0;
            for high in 0..5 {
                for low in (0..=high).rev() {
                    if seen == index {
                        return GradeInterval::new(low, high).unwrap();
                    }
                    seen += 1;
                }
            }
            unreachable!()
        });
        for family in 0..(1u32 << all.len()) {
            let mut selected = [GradeInterval::new(0, 0).unwrap(); 15];
            let mut len = 0;
            for (i, interval) in all.iter().enumerate().rev() {
                if family & (1 << i) != 0 {
                    selected[len] = *interval;
                    len += 1;
                }
            }
            let intervals = &selected[..len];
            let count = MinimumGauges::new(intervals).count();
            assert_eq!(count, brute_minimum(intervals), "family {family:#x}");
            assert!(intervals.iter().all(|interval| {
                MinimumGauges::new(intervals).any(|gauge| interval.contains(gauge))
            }));
        }

        // Reconstruction is independent of the chosen gauge and translation:
        // Q + (a+t) + (b-Q-t) is the original Laurent product grade.  Digit
        // multiplication itself is the complete ternary table.
        let left = FiniteNaf::from_scaled(false, 79, -3);
        let right = FiniteNaf::from_scaled(true, 135, 5);
        let joint = left.support().unwrap().product(right.support().unwrap());
        let q = joint.greatest();
        let t = left
            .support()
            .unwrap()
            .translations(right.support().unwrap(), q)
            .unwrap()
            .greatest();
        let mut reconstructed = 0i128;
        for x in left.atoms() {
            for y in right.atoms() {
                let local_a = x.grade + t;
                let local_b = y.grade - q - t;
                assert_eq!(q + local_a + local_b, x.grade + y.grade);
                let shift = (q + local_a + local_b - joint.low) as u32;
                reconstructed += i128::from(x.digit.product(y.digit).coefficient()) << shift;
            }
        }
        assert_eq!(reconstructed, evaluated(left) * evaluated(right));

        // For a group, the greedy right endpoint is the greatest point in the
        // whole intersection, not a threshold chosen independently of it.
        let group = [
            GradeInterval::new(-5, 9).unwrap(),
            GradeInterval::new(1, 7).unwrap(),
            GradeInterval::new(3, 12).unwrap(),
        ];
        let q = MinimumGauges::new(&group).next().unwrap();
        let intersection = GradeInterval::new(
            group.iter().map(|interval| interval.low).max().unwrap(),
            group.iter().map(|interval| interval.high).min().unwrap(),
        )
        .unwrap();
        assert_eq!(q, intersection.greatest());
        assert!(group.iter().all(|interval| interval.contains(q)));
        for candidate in intersection.low..=intersection.high {
            assert!(candidate - intersection.low <= q - intersection.low);
        }
    }
}
