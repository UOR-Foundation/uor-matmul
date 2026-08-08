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
use uor_matmul_core::{
    AccOf, Accumulator, Complete, Element, EncodeFrom, FloatElement, Full, MatView, MatViewMut,
    PackedCode, Shape, Triple,
};
use uor_matmul_kernels::{KernelSpec, MAX_TILE_LANES};

use crate::driver::GemmOptions;
use crate::epilogue::{Epilogue, PlaceAt};
use crate::scratch::Scratch;

/// One normalized coefficient of the Laurent representative.
///
/// Removing the dyadic valuation before any address is chosen is the canonical
/// section of the `X - 2` quotient.  `unit` is therefore odd, `grade` carries
/// every removed factor of two, and the sign is the Atlas modality mirror.
#[derive(Clone, Copy)]
struct AtlasAtom {
    unit: u64,
    grade: i64,
    negative: bool,
}

/// Finite source data not already carried by the six-state boundary quotient.
#[derive(Clone, Copy)]
struct AtlasFiniteSite {
    unit: u64,
    grade: i64,
}

impl AtlasFiniteSite {
    const ZERO: Self = Self { unit: 0, grade: 0 };
}

/// Signed local carrier for one lookup-resolved source product.
///
/// Two `u64` source coefficients have a sub-`2^128` final magnitude. During
/// centered-octet convolution, a signed `i32` diagonal can temporarily extend
/// one lane-width beyond that final range before the next diagonal cancels it;
/// the derived three-limb carrier retains that headroom exactly.
const ATLAS_PRODUCT_LIMBS: usize = (u128::BITS + i32::BITS).div_ceil(u64::BITS) as usize;

#[derive(Clone, Copy)]
struct AtlasProduct {
    limbs: [u64; ATLAS_PRODUCT_LIMBS],
}

impl AtlasProduct {
    const ZERO: Self = Self {
        limbs: [0; ATLAS_PRODUCT_LIMBS],
    };

    /// Add one signed kernel diagonal at its radix-octet address.
    #[inline(always)]
    fn add_diagonal(&mut self, value: i32, octet: usize) {
        if value == 0 {
            return;
        }
        let limb_octets = core::mem::size_of::<u64>();
        let at = octet / limb_octets;
        let within = octet % limb_octets;
        let mut spread_octets = [[0u8; core::mem::size_of::<u64>()]; 2];
        for (offset, byte) in value.unsigned_abs().to_le_bytes().into_iter().enumerate() {
            let address = within + offset;
            spread_octets[address / limb_octets][address % limb_octets] = byte;
        }
        let spread = spread_octets.map(u64::from_le_bytes);

        let mut carry = 0u64;
        for (offset, part) in spread.into_iter().enumerate() {
            let lane = at + offset;
            if lane >= self.limbs.len() {
                debug_assert_eq!(part, 0, "the derived carrier retains every diagonal bit");
                return;
            }
            if value < 0 {
                let (difference, b1) = self.limbs[lane].overflowing_sub(part);
                let (difference, b2) = difference.overflowing_sub(carry);
                self.limbs[lane] = difference;
                carry = u64::from(b1) + u64::from(b2);
            } else {
                let (sum, c1) = self.limbs[lane].overflowing_add(part);
                let (sum, c2) = sum.overflowing_add(carry);
                self.limbs[lane] = sum;
                carry = u64::from(c1) + u64::from(c2);
            }
        }
        let mut lane = at + spread.len();
        while carry != 0 && lane < self.limbs.len() {
            if value < 0 {
                let (difference, borrow) = self.limbs[lane].overflowing_sub(carry);
                self.limbs[lane] = difference;
                carry = u64::from(borrow);
            } else {
                let (sum, overflow) = self.limbs[lane].overflowing_add(carry);
                self.limbs[lane] = sum;
                carry = u64::from(overflow);
            }
            lane += 1;
        }
        // A carry or borrow out of the sign-extension limb is the repeated
        // sign bit, not a coefficient bit. The derived headroom makes signed
        // overflow inside the retained carrier unreachable.
    }

    /// Resolve the final sub-`2^128` coefficient into modality and magnitude.
    #[inline(always)]
    fn signed_magnitude(self) -> (bool, u128) {
        let negative = (self.limbs[self.limbs.len() - 1] as i64) < 0;
        let mut low_octets = [0u8; core::mem::size_of::<u128>()];
        let limb_octets = core::mem::size_of::<u64>();
        low_octets[..limb_octets].copy_from_slice(&self.limbs[0].to_le_bytes());
        low_octets[limb_octets..].copy_from_slice(&self.limbs[1].to_le_bytes());
        let low = u128::from_le_bytes(low_octets);
        debug_assert!(if negative {
            self.limbs[self.limbs.len() - 1] == u64::MAX
        } else {
            self.limbs[self.limbs.len() - 1] == 0
        });
        (negative, if negative { low.wrapping_neg() } else { low })
    }
}

const ZERO_CODE: PackedCode = PackedCode {
    mantissa: 0,
    exp: 0,
    _pad: 0,
};

/// Recover the exact finite fields from either packed layout.
///
/// The trailing word is a signed modality tag only at the canonical `-1` and
/// `+1` escape tags; historical noncanonical padding retains the ordinary
/// exponent-based interpretation. Keeping this one interpretation beside the
/// Atlas projection prevents any factorization from assigning a second meaning
/// to the public cache representation (`CK-21`).
#[inline(always)]
fn finite_parts(code: PackedCode) -> Option<(bool, u64, i32)> {
    if code._pad == -1 || code._pad == 1 {
        Some((code._pad < 0, code.mantissa as u64, code.exp))
    } else if code.exp > PackedCode::INF_EXP {
        Some((code.mantissa < 0, code.mantissa.unsigned_abs(), code.exp))
    } else {
        None
    }
}

/// Bits carried by one lookup coordinate.
///
/// The Atlas context addresses Laurent grades; it is not the radix of a
/// particular kernel alphabet.  This traversal's coordinate radix comes from
/// the signed-i8 lookup alphabet itself.
const ATLAS_DIGIT_BITS: u32 = i8::BITS;

/// Local coordinate words sufficient for every signed `u64` source atom.
///
/// The source width fixes eight radix-256 words and centered recoding can emit
/// one final carry word. Laurent grade is carried independently at placement,
/// so address distance never consumes local storage.
const MAX_ATLAS_WORDS: usize = (u64::BITS.div_ceil(ATLAS_DIGIT_BITS) + 1) as usize;

/// The direct factorization holds one finite source word per physical lane.
const ATLAS_TILE_COORDINATES: usize = MAX_TILE_LANES * MAX_ATLAS_WORDS;

/// Fixed state live beside the exact output cells during one contraction.
///
/// Keeping every fixed array in one object makes its `size_of` the real cache
/// charge used by cell residency, including layout padding. Source state uses
/// the derived sum bound; product state remains one per physical output because
/// the selected kernel overwrites its complete tile on every diagonal.
struct AtlasTileWorkspace {
    source_kinds: [u8; MAX_ATLAS_SOURCE_SITES],
    source_finite: [AtlasFiniteSite; MAX_ATLAS_SOURCE_SITES],
    source_words: [[i8; MAX_ATLAS_WORDS]; MAX_ATLAS_SOURCE_SITES],
    source_extents: [u8; MAX_ATLAS_SOURCE_SITES],
    a_active: [bool; MAX_ATLAS_WORDS],
    b_active: [bool; MAX_ATLAS_WORDS],
    left: [i8; ATLAS_TILE_COORDINATES],
    right: [i8; ATLAS_TILE_COORDINATES],
    lanes: [i32; MAX_TILE_LANES],
    products: [AtlasProduct; MAX_TILE_LANES],
}

impl AtlasTileWorkspace {
    const ZERO: Self = Self {
        source_kinds: [AtlasProjectedKind::FiniteZero as u8; MAX_ATLAS_SOURCE_SITES],
        source_finite: [AtlasFiniteSite::ZERO; MAX_ATLAS_SOURCE_SITES],
        source_words: [[0; MAX_ATLAS_WORDS]; MAX_ATLAS_SOURCE_SITES],
        source_extents: [0; MAX_ATLAS_SOURCE_SITES],
        a_active: [false; MAX_ATLAS_WORDS],
        b_active: [false; MAX_ATLAS_WORDS],
        left: [0; ATLAS_TILE_COORDINATES],
        right: [0; ATLAS_TILE_COORDINATES],
        lanes: [0; MAX_TILE_LANES],
        products: [AtlasProduct::ZERO; MAX_TILE_LANES],
    };
}

const ATLAS_TILE_WORK_BYTES: usize = core::mem::size_of::<AtlasTileWorkspace>();
const ATLAS_SIGNED_PLACE_RADIX: u128 = i128::MIN.unsigned_abs();
const ATLAS_LIMB_RADIX: u128 = i64::MIN.unsigned_abs() as u128 + i64::MIN.unsigned_abs() as u128;

#[inline(always)]
const fn atlas_split_signed_place(magnitude: u128) -> (u128, u128) {
    (
        magnitude % ATLAS_SIGNED_PLACE_RADIX,
        magnitude / ATLAS_SIGNED_PLACE_RADIX,
    )
}

/// Remove the exact dyadic valuation through the defining radix-two quotient.
///
/// The quotient/add witness keeps the canonical section independent of binary
/// field inspection. A wider-radix prepass was measured and rejected because
/// its divisibility test regressed ordinary finite f32 panels.
#[inline(always)]
fn atlas_odd_section(mut magnitude: u64) -> (u64, u32) {
    if magnitude == 0 {
        return (0, 0);
    }
    let mut valuation = 0u32;
    loop {
        let quotient = magnitude / 2;
        if quotient.wrapping_add(quotient) != magnitude {
            return (magnitude, valuation);
        }
        magnitude = quotient;
        valuation += 1;
    }
}

/// Decode a finite nonzero coefficient into the canonical odd section.
#[inline(always)]
fn atlas_atom(code: PackedCode, prescaled: bool) -> Option<AtlasAtom> {
    let (kind, finite) = atlas_source_state(code, prescaled);
    if !atlas_kind_is_productive(kind) {
        return None;
    }
    Some(AtlasAtom {
        unit: finite.unit,
        grade: finite.grade,
        negative: kind == AtlasProjectedKind::FiniteNegative as u8,
    })
}

/// The canonical signed radix-`2^O` coordinates of one Atlas atom.
///
/// This is not a second value representation.  It is the ordered word view of
/// the same NAF/Laurent representative: quotient normalization happens before
/// this projection, and evaluating the coordinates at `X = 2` reconstructs the
/// coefficient exactly.
#[inline(always)]
fn atlas_word(atom: AtlasAtom, coordinates: &mut [i8]) -> usize {
    let radix = i128::from(u8::MAX) + 1;
    let mut context = 0usize;
    let magnitude = u128::from(atom.unit);
    let mut value = if atom.negative {
        -(magnitude as i128)
    } else {
        magnitude as i128
    };
    while value != 0 && context < coordinates.len() {
        let residue = value.rem_euclid(radix);
        let digit = if residue > i128::from(i8::MAX) {
            residue - radix
        } else {
            residue
        };
        coordinates[context] = digit as i8;
        value = (value - digit) / radix;
        context += 1;
    }
    debug_assert_eq!(
        value, 0,
        "the source-derived Atlas width contains every octet"
    );
    context
}

/// Replace one reused coordinate word without pre-clearing its live prefix.
#[inline(always)]
fn replace_atlas_word(atom: AtlasAtom, coordinates: &mut [i8]) -> usize {
    let extent = atlas_word(atom, coordinates);
    coordinates[extent..].fill(0);
    extent
}

/// The caller's sixteen-byte panel word viewed as a ready Atlas projection.
///
/// Nine signed octets are the complete centered word of a `u64` coefficient.
/// The original exponent and removed valuation recover the full Laurent grade;
/// the final byte distinguishes zero, modality, and the three boundary kinds.
/// This has exactly the public cache word's size, so an offered `PackedCode`
/// slot changes interpretation in place without another buffer or copy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct AtlasProjectedCode {
    exponent: i32,
    coordinates: [i8; MAX_ATLAS_WORDS],
    valuation: u8,
    extent: u8,
    kind: u8,
}

/// The complete semantic partition of a projected cache word.
///
/// The ordinal is private storage protocol, derived from this exhaustive list;
/// it is not an Atlas parameter or a model constant. `FiniteZero` is first so
/// a zeroed workspace is the finite additive identity.
#[derive(Clone, Copy)]
#[repr(u8)]
enum AtlasProjectedKind {
    FiniteZero,
    FinitePositive,
    FiniteNegative,
    PositiveInfinity,
    NegativeInfinity,
    NotANumber,
}

#[inline(always)]
fn atlas_source_state(code: PackedCode, prescaled: bool) -> (u8, AtlasFiniteSite) {
    if let Some((negative, magnitude, exponent)) = finite_parts(code) {
        if magnitude == 0 {
            return (AtlasProjectedKind::FiniteZero as u8, AtlasFiniteSite::ZERO);
        }
        let (unit, valuation) = atlas_odd_section(magnitude);
        (
            if negative {
                AtlasProjectedKind::FiniteNegative as u8
            } else {
                AtlasProjectedKind::FinitePositive as u8
            },
            AtlasFiniteSite {
                unit,
                grade: i64::from(if prescaled { 0 } else { exponent }) + i64::from(valuation),
            },
        )
    } else if code.is_nan() {
        (AtlasProjectedKind::NotANumber as u8, AtlasFiniteSite::ZERO)
    } else if code.mantissa < 0 {
        (
            AtlasProjectedKind::NegativeInfinity as u8,
            AtlasFiniteSite::ZERO,
        )
    } else {
        (
            AtlasProjectedKind::PositiveInfinity as u8,
            AtlasFiniteSite::ZERO,
        )
    }
}

#[inline(always)]
fn atlas_kind_is_boundary(kind: u8) -> bool {
    kind == AtlasProjectedKind::PositiveInfinity as u8
        || kind == AtlasProjectedKind::NegativeInfinity as u8
        || kind == AtlasProjectedKind::NotANumber as u8
}

#[inline(always)]
fn atlas_kind_is_productive(kind: u8) -> bool {
    kind == AtlasProjectedKind::FinitePositive as u8
        || kind == AtlasProjectedKind::FiniteNegative as u8
}

#[inline(always)]
fn atlas_boundary_code(kind: u8) -> PackedCode {
    if kind == AtlasProjectedKind::FiniteZero as u8 {
        ZERO_CODE
    } else if kind == AtlasProjectedKind::FinitePositive as u8 {
        UNIT_CODE
    } else if kind == AtlasProjectedKind::FiniteNegative as u8 {
        PackedCode {
            mantissa: -1,
            exp: 0,
            _pad: 0,
        }
    } else if kind == AtlasProjectedKind::PositiveInfinity as u8 {
        PackedCode::of(uor_matmul_core::Decoded::Infinite { sign: false })
    } else if kind == AtlasProjectedKind::NegativeInfinity as u8 {
        PackedCode::of(uor_matmul_core::Decoded::Infinite { sign: true })
    } else if kind == AtlasProjectedKind::NotANumber as u8 {
        PackedCode::of(uor_matmul_core::Decoded::NotANumber)
    } else {
        unreachable!("every Atlas boundary kind is constructed in this module")
    }
}

const _: () = assert!(
    core::mem::size_of::<AtlasProjectedCode>() == core::mem::size_of::<PackedCode>(),
    "the Atlas projection must occupy exactly one public cache word"
);

impl AtlasProjectedCode {
    #[inline(always)]
    fn project(code: PackedCode) -> (Self, bool) {
        let Some((negative, magnitude, exponent)) = finite_parts(code) else {
            return (
                Self {
                    exponent: 0,
                    coordinates: [0; MAX_ATLAS_WORDS],
                    valuation: 0,
                    extent: 0,
                    kind: if code.is_nan() {
                        AtlasProjectedKind::NotANumber as u8
                    } else if code.mantissa < 0 {
                        AtlasProjectedKind::NegativeInfinity as u8
                    } else {
                        AtlasProjectedKind::PositiveInfinity as u8
                    },
                },
                false,
            );
        };
        if magnitude == 0 {
            return (
                Self {
                    exponent,
                    coordinates: [0; MAX_ATLAS_WORDS],
                    valuation: 0,
                    extent: 0,
                    kind: AtlasProjectedKind::FiniteZero as u8,
                },
                false,
            );
        }
        let (unit, valuation) = atlas_odd_section(magnitude);
        let atom = AtlasAtom {
            unit,
            grade: i64::from(exponent) + i64::from(valuation),
            negative,
        };
        let mut coordinates = [0; MAX_ATLAS_WORDS];
        let extent = atlas_word(atom, &mut coordinates);
        (
            Self {
                exponent,
                coordinates,
                valuation: valuation as u8,
                extent: extent as u8,
                kind: if negative {
                    AtlasProjectedKind::FiniteNegative as u8
                } else {
                    AtlasProjectedKind::FinitePositive as u8
                },
            },
            true,
        )
    }

    #[inline(always)]
    fn into_packed(self) -> PackedCode {
        bytemuck::cast(self)
    }

    #[inline(always)]
    fn from_packed(code: PackedCode) -> Self {
        bytemuck::cast(code)
    }

    #[cfg(test)]
    #[inline(always)]
    fn atom(self) -> Option<AtlasAtom> {
        if self.kind != AtlasProjectedKind::FinitePositive as u8
            && self.kind != AtlasProjectedKind::FiniteNegative as u8
        {
            return None;
        }
        Some(AtlasAtom {
            // Projection has already consumed the coefficient. Only presence,
            // modality, and grade are read after this boundary.
            unit: 1,
            grade: i64::from(self.exponent) + i64::from(self.valuation),
            negative: self.kind == AtlasProjectedKind::FiniteNegative as u8,
        })
    }

    #[inline(always)]
    fn finite_site(self) -> AtlasFiniteSite {
        debug_assert!(atlas_kind_is_productive(self.kind));
        AtlasFiniteSite {
            unit: 1,
            grade: i64::from(self.exponent) + i64::from(self.valuation),
        }
    }

    /// A representative with the same IEEE boundary action.
    ///
    /// Finite magnitude matters to a boundary product only through zero versus
    /// nonzero and its modality. The exact finite coefficient continues through
    /// `coordinates`; this representative is used solely when its partner is a
    /// NaN or infinity.
    #[cfg(test)]
    #[inline(always)]
    fn boundary_code(self) -> PackedCode {
        atlas_boundary_code(self.kind)
    }
}

#[derive(Clone, Copy)]
enum AtlasSource {
    Raw(PackedCode),
    Projected(AtlasProjectedCode),
}

#[inline(always)]
fn cache_atlas_source<Lg: AtlasLedger>(code: PackedCode, ledger: &mut Lg) -> PackedCode {
    let (projected, occupied) = AtlasProjectedCode::project(code);
    if occupied {
        ledger.projected();
    }
    projected.into_packed()
}

/// Select a lookup/add reduce kernel from the existing `i8` family.
///
/// A one-coordinate `k_group` is the kernel declaration that the input is
/// already an Atlas octet stream.  The portable references and every ISA
/// lookup sequence declare it; multiply instructions declare the native group
/// they consume instead.  Filtering on the declaration keeps integer dispatch
/// unchanged while making the float call graph lookup-only.
#[inline]
fn resolve_atlas_dot_spec(backend: uor_matmul_core::Backend) -> KernelSpec<i8, i32> {
    uor_matmul_kernels::choose_for_rows(
        uor_matmul_kernels::cached::available_reduce_i8().filter(|spec| spec.k_group == 1),
        backend,
        <i8 as uor_matmul_core::IntegerElement>::FULL,
        1,
    )
    .expect("the portable Atlas lookup reduction is always present")
}

#[cfg(feature = "std")]
static ATLAS_DOT_AUTO_SPEC: std::sync::OnceLock<KernelSpec<i8, i32>> = std::sync::OnceLock::new();

#[cfg(feature = "std")]
static ATLAS_DOT_NAMED_SPECS: [std::sync::OnceLock<KernelSpec<i8, i32>>;
    uor_matmul_core::Backend::ALL.len()] =
    [const { std::sync::OnceLock::new() }; uor_matmul_core::Backend::ALL.len()];

#[cfg(all(feature = "std", test))]
static ATLAS_DOT_AUTO_RESOLUTIONS: core::sync::atomic::AtomicUsize =
    core::sync::atomic::AtomicUsize::new(0);

#[cfg(all(feature = "std", test))]
static ATLAS_DOT_NAMED_RESOLUTIONS: [core::sync::atomic::AtomicUsize;
    uor_matmul_core::Backend::ALL.len()] =
    [const { core::sync::atomic::AtomicUsize::new(0) }; uor_matmul_core::Backend::ALL.len()];

#[cfg(feature = "std")]
fn atlas_dot_backend_index(backend: uor_matmul_core::Backend) -> usize {
    uor_matmul_core::Backend::ALL
        .iter()
        .position(|&candidate| candidate == backend)
        .expect("Backend::ALL contains every named backend")
}

#[cfg(all(feature = "std", test))]
#[inline(always)]
fn record_atlas_dot_resolution(backend: uor_matmul_core::Backend) {
    use core::sync::atomic::Ordering;

    if backend == uor_matmul_core::Backend::Auto {
        ATLAS_DOT_AUTO_RESOLUTIONS.fetch_add(1, Ordering::Relaxed);
    } else {
        ATLAS_DOT_NAMED_RESOLUTIONS[atlas_dot_backend_index(backend)]
            .fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg(all(feature = "std", not(test)))]
#[inline(always)]
fn record_atlas_dot_resolution(_: uor_matmul_core::Backend) {}

#[inline]
pub(crate) fn atlas_dot_spec(backend: uor_matmul_core::Backend) -> KernelSpec<i8, i32> {
    #[cfg(feature = "std")]
    {
        let slot = if backend == uor_matmul_core::Backend::Auto {
            &ATLAS_DOT_AUTO_SPEC
        } else {
            &ATLAS_DOT_NAMED_SPECS[atlas_dot_backend_index(backend)]
        };
        *slot.get_or_init(|| {
            record_atlas_dot_resolution(backend);
            resolve_atlas_dot_spec(backend)
        })
    }
    #[cfg(not(feature = "std"))]
    {
        // Every availability predicate is a build-time constant without
        // `std`, so inlining folds this declaration walk into the call.
        resolve_atlas_dot_spec(backend)
    }
}

#[cfg(all(feature = "std", test))]
pub(crate) fn atlas_dot_resolutions(backend: uor_matmul_core::Backend) -> usize {
    use core::sync::atomic::Ordering;

    if backend == uor_matmul_core::Backend::Auto {
        ATLAS_DOT_AUTO_RESOLUTIONS.load(Ordering::Relaxed)
    } else {
        ATLAS_DOT_NAMED_RESOLUTIONS[atlas_dot_backend_index(backend)].load(Ordering::Relaxed)
    }
}

/// Select the lookup/add declaration whose work is least for this shape.
///
/// The three lists are storage organizations of the same coordinate
/// contraction: full and narrow output-column tiles, and reduction-oriented
/// rows.  Selection walks every compatible declaration.  Choosing one family
/// first would hide candidates from the comparison and turn list order into an
/// undeclared threshold.
#[inline]
fn atlas_tile_spec<A>(
    backend: uor_matmul_core::Backend,
    shape: Shape,
    pa_codes: usize,
    pb_codes: usize,
) -> KernelSpec<i8, i32> {
    let mut automatic = None;
    let mut named = None;
    let mut portable = None;
    for spec in uor_matmul_kernels::cached::available_i8()
        .chain(uor_matmul_kernels::cached::available_i8_narrow())
        .chain(uor_matmul_kernels::cached::available_reduce_i8())
        .filter(|spec| {
            spec.k_group == 1
                && matches!(spec.factorization, uor_matmul_kernels::Factorization::Exact)
                && spec.max_bound >= <i8 as uor_matmul_core::IntegerElement>::FULL
        })
    {
        let take = |current: Option<KernelSpec<i8, i32>>| {
            current.is_none_or(|incumbent| {
                atlas_executed_work::<A>(spec, shape, pa_codes, pb_codes)
                    <= atlas_executed_work::<A>(incumbent, shape, pa_codes, pb_codes)
            })
        };
        if take(automatic) {
            automatic = Some(spec);
        }
        if spec.backend == backend && take(named) {
            named = Some(spec);
        }
        if spec.backend == uor_matmul_core::Backend::Portable && take(portable) {
            portable = Some(spec);
        }
    }

    if backend == uor_matmul_core::Backend::Auto {
        automatic
    } else {
        named.or(portable)
    }
    .expect("the portable Atlas lookup/add declarations are always present")
}

/// Exact count in a radix whose word is the widest source coefficient.
///
/// Four words are derived from the four address-sized factors any candidate
/// can issue: `m`, `k`, `n`, and at most one physical-tile factor.  Arithmetic
/// uses quotient and remainder rather than a host bit decomposition, and the
/// most-significant-first array gives ordinary lexicographic numerical order.
#[derive(Clone, Copy)]
enum AtlasCountFactor {
    Rows,
    Depth,
    Columns,
    PhysicalTile,
}

const ATLAS_COUNT_WORDS: usize = [
    AtlasCountFactor::Rows,
    AtlasCountFactor::Depth,
    AtlasCountFactor::Columns,
    AtlasCountFactor::PhysicalTile,
]
.len();
const ATLAS_COUNT_RADIX: u128 = u64::MAX as u128 + (u64::MAX != u64::MIN) as u128;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct AtlasCount([u64; ATLAS_COUNT_WORDS]);

impl AtlasCount {
    const ZERO: Self = Self([0; ATLAS_COUNT_WORDS]);

    #[inline(always)]
    fn from_u128(mut value: u128) -> Self {
        let mut words = [0u64; ATLAS_COUNT_WORDS];
        for word in words.iter_mut().rev() {
            *word = (value % ATLAS_COUNT_RADIX) as u64;
            value /= ATLAS_COUNT_RADIX;
        }
        debug_assert_eq!(value, 0, "one u128 occupies at most two radix words");
        Self(words)
    }

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        let mut words = [0u64; ATLAS_COUNT_WORDS];
        let mut carry = 0u128;
        for index in (0..ATLAS_COUNT_WORDS).rev() {
            let sum = u128::from(self.0[index]) + u128::from(other.0[index]) + carry;
            words[index] = (sum % ATLAS_COUNT_RADIX) as u64;
            carry = sum / ATLAS_COUNT_RADIX;
        }
        debug_assert_eq!(carry, 0, "four source-derived words retain the work census");
        Self(words)
    }

    #[inline(always)]
    fn multiply(self, factor: usize) -> Self {
        let mut words = [0u64; ATLAS_COUNT_WORDS];
        let mut carry = 0u128;
        for index in (0..ATLAS_COUNT_WORDS).rev() {
            let product = u128::from(self.0[index]) * factor as u128 + carry;
            words[index] = (product % ATLAS_COUNT_RADIX) as u64;
            carry = product / ATLAS_COUNT_RADIX;
        }
        debug_assert_eq!(carry, 0, "four source-derived words retain the work census");
        Self(words)
    }

    #[cfg(test)]
    fn coordinates(self) -> [u128; ATLAS_COUNT_WORDS] {
        self.0.map(u128::from)
    }
}

/// The complete candidate census used by the live selector.
///
/// Field order is deliberately visible: the measured bottleneck, source
/// projection, is compared first; decode reuse and lookup issue work break its
/// ties, followed by live carrier initialization and peak live storage. CG-21
/// measures every eligible declaration against this order. A different winner
/// must change the measured model record rather than smuggle a scalar weight
/// into this code.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct AtlasWork {
    projections: AtlasCount,
    decodes: AtlasCount,
    issued: AtlasCount,
    product_initializations: AtlasCount,
    live_bytes: AtlasCount,
}

impl AtlasWork {
    const ZERO: Self = Self {
        projections: AtlasCount::ZERO,
        decodes: AtlasCount::ZERO,
        issued: AtlasCount::ZERO,
        product_initializations: AtlasCount::ZERO,
        live_bytes: AtlasCount::ZERO,
    };

    #[cfg(test)]
    fn coordinates(self) -> [[u128; ATLAS_COUNT_WORDS]; 5] {
        [
            self.projections.coordinates(),
            self.decodes.coordinates(),
            self.issued.coordinates(),
            self.product_initializations.coordinates(),
            self.live_bytes.coordinates(),
        ]
    }
}

#[inline(always)]
fn atlas_product_count(left: usize, right: usize) -> AtlasCount {
    AtlasCount::from_u128(
        (left as u128)
            .checked_mul(right as u128)
            .expect("two address-sized extents fit u128 exactly"),
    )
}

/// Exact structural work executed by one declaration and caller panel offer.
///
/// Counts include the reduction depth rather than cancelling it as a common
/// factor. `pb_codes` determines the real block boundaries, including a final
/// partial offer; `pa_codes` prices the rows decoded once per such block. The
/// projection and issue terms count the actual edge tiles those boundaries
/// create. Peak storage is the fixed direct workspace plus the largest live
/// output frame, never the padded physical tile.
#[inline(always)]
fn atlas_executed_work<A>(
    spec: KernelSpec<i8, i32>,
    shape: Shape,
    pa_codes: usize,
    pb_codes: usize,
) -> AtlasWork {
    if shape.m == 0 || shape.k == 0 || shape.n == 0 {
        return AtlasWork::ZERO;
    }

    let a_offer_rows = pa_codes.checked_div(shape.k).unwrap_or(0);
    let b_offer_cols = pb_codes.checked_div(shape.k).unwrap_or(0).min(shape.n);
    let streamed_cols = spec.nr.min(shape.n).max(1);
    let block_width = if b_offer_cols == 0 {
        streamed_cols
    } else {
        b_offer_cols
    };
    let full_blocks = shape.n / block_width;
    let tail_cols = shape.n % block_width;
    let block_count = full_blocks + usize::from(tail_cols != 0);
    let full_block_tiles = block_width.div_ceil(spec.nr);
    let column_tiles = full_blocks
        .checked_mul(full_block_tiles)
        .and_then(|tiles| {
            tiles.checked_add(if tail_cols == 0 {
                0
            } else {
                tail_cols.div_ceil(spec.nr)
            })
        })
        .expect("column tiles cannot outnumber addressable columns");
    let row_tiles = shape.m.div_ceil(spec.mr);

    let full_row_tiles = shape.m / spec.mr;
    let tail_rows = shape.m % spec.mr;
    let cached_full_rows = a_offer_rows.min(spec.mr);
    let cached_rows = full_row_tiles
        .checked_mul(cached_full_rows)
        .and_then(|rows| rows.checked_add(a_offer_rows.min(tail_rows)))
        .expect("cached rows are a subset of addressable rows");
    let decoded_a = atlas_product_count(block_count, cached_rows)
        .add(atlas_product_count(column_tiles, shape.m - cached_rows));
    let decoded_b = if b_offer_cols == 0 {
        atlas_product_count(shape.n, row_tiles)
    } else {
        AtlasCount::from_u128(shape.n as u128)
    };
    let decodes = decoded_a.add(decoded_b).multiply(shape.k);
    // Every decoded nonzero source is projected at that same boundary. An
    // offered source then reuses the in-place coordinate word, so the declared
    // all-finite census has exactly the decode geometry. Runtime zero and
    // boundary codes can only remove projections; they cannot select a route.
    let projection_sites = decodes;

    let physical_outputs = spec
        .mr
        .checked_mul(spec.nr)
        .expect("every shipped physical tile fits MAX_TILE_LANES");
    let steps = physical_outputs.div_ceil(spec.products_per_step);
    let issued = atlas_product_count(row_tiles, column_tiles)
        .multiply(steps)
        .multiply(shape.k);
    let product_initializations = atlas_product_count(shape.m, shape.n).multiply(shape.k);

    let live_rows = spec.mr.min(shape.m);
    let live_cols = spec.nr.min(block_width).min(shape.n);
    let live_cells = live_rows
        .checked_mul(live_cols)
        .expect("live cells fit their physical kernel tile");
    let live_bytes = AtlasCount::from_u128(
        (ATLAS_TILE_WORK_BYTES as u128)
            .checked_add(
                (live_cells as u128)
                    .checked_mul(core::mem::size_of::<A>() as u128)
                    .expect("one tile's live exact cells fit u128"),
            )
            .expect("one bounded Atlas frame fits u128"),
    );

    AtlasWork {
        projections: projection_sites,
        decodes,
        issued,
        product_initializations,
        live_bytes,
    }
}

/// Zero-cost operation ledger for the Atlas traversal.
///
/// `()` is the shipped instantiation: every method is empty and monomorphizes
/// away. Tests instantiate the same body with counters, so route and operation
/// claims are observations of executed calls rather than a second walk that
/// predicts what ought to have happened.
trait AtlasLedger {
    fn selected(&mut self, spec: KernelSpec<i8, i32>, shape: Shape);
    fn panel(&mut self);
    fn decoded_a(&mut self);
    fn decoded_b(&mut self);
    fn projected(&mut self);
    fn product_initialized(&mut self, live_products: usize);
    fn kernel_call(&mut self, coordinate_products: usize);
    fn placed(&mut self);
    fn boundary_joined(&mut self);
    fn encoded(&mut self);
}

impl AtlasLedger for () {
    #[inline(always)]
    fn selected(&mut self, _: KernelSpec<i8, i32>, _: Shape) {}
    #[inline(always)]
    fn panel(&mut self) {}
    #[inline(always)]
    fn decoded_a(&mut self) {}
    #[inline(always)]
    fn decoded_b(&mut self) {}
    #[inline(always)]
    fn projected(&mut self) {}
    #[inline(always)]
    fn product_initialized(&mut self, _: usize) {}
    #[inline(always)]
    fn kernel_call(&mut self, _: usize) {}
    #[inline(always)]
    fn placed(&mut self) {}
    #[inline(always)]
    fn boundary_joined(&mut self) {}
    #[inline(always)]
    fn encoded(&mut self) {}
}

/// Exact output cells retained by one bounded Atlas contraction.
trait AtlasCells<A> {
    fn for_each_live<F: FnMut(usize, &mut A)>(&mut self, visit: &mut F);
}

#[cfg(test)]
struct SliceAtlasCells<'a, A> {
    rows: usize,
    cols: usize,
    physical_cols: usize,
    accumulators: &'a mut [A],
}

#[cfg(test)]
impl<A> AtlasCells<A> for SliceAtlasCells<'_, A> {
    #[inline(always)]
    fn for_each_live<F: FnMut(usize, &mut A)>(&mut self, visit: &mut F) {
        for row in 0..self.rows {
            for col in 0..self.cols {
                let physical_lane = row * self.physical_cols + col;
                visit(physical_lane, &mut self.accumulators[physical_lane]);
            }
        }
    }
}

/// A contiguous group of live output cells viewed at their physical kernel
/// lanes. The exact accumulators are dense; only the lane addresses retain a
/// kernel tile's padding.
struct WindowAtlasCells<'a, A> {
    first_logical: usize,
    live_cols: usize,
    physical_cols: usize,
    accumulators: &'a mut [A],
}

impl<A> AtlasCells<A> for WindowAtlasCells<'_, A> {
    #[inline(always)]
    fn for_each_live<F: FnMut(usize, &mut A)>(&mut self, visit: &mut F) {
        for (offset, accumulator) in self.accumulators.iter_mut().enumerate() {
            let logical = self.first_logical + offset;
            let row = logical / self.live_cols;
            let col = logical % self.live_cols;
            visit(row * self.physical_cols + col, accumulator);
        }
    }
}

// The runtime-to-const bridge is generated from the model-owned maximum tile
// geometry. Keeping the exhaustive arms out of this source prevents a second
// representation limit from drifting away from the kernel family.
include!("generated_atlas_dispatch.rs");

#[inline(always)]
fn atlas_panel_slot(
    layout: uor_matmul_kernels::LaneLayout,
    p: usize,
    lane: usize,
    lanes: usize,
    depth: usize,
) -> usize {
    match layout {
        uor_matmul_kernels::LaneLayout::Interleaved => p * lanes + lane,
        uor_matmul_kernels::LaneLayout::Contiguous => lane * depth + p,
    }
}

/// Contract canonical source words one Laurent diagonal at a time.
///
/// This is the self-similar factorization: increasing source precision adds
/// coordinate words governed by the identical projection and diagonal step.
/// No format arm, precision cutoff, or operand-sized representation appears.
#[inline(never)]
#[allow(clippy::needless_range_loop, clippy::too_many_arguments)]
fn accumulate_direct_atlas_tile<A, P, Lg, FA, FB, C>(
    workspace: &mut AtlasTileWorkspace,
    cells: &mut C,
    rows: usize,
    cols: usize,
    depth: usize,
    spec: KernelSpec<i8, i32>,
    mut source_a: FA,
    mut source_b: FB,
    place: P,
    ledger: &mut Lg,
) where
    A: SignedPlace,
    P: Fn(&mut A, i128, i64) + Copy,
    Lg: AtlasLedger,
    FA: FnMut(usize, usize, &mut Lg) -> AtlasSource,
    FB: FnMut(usize, usize, &mut Lg) -> AtlasSource,
    C: AtlasCells<A>,
{
    let (a_kinds, remainder) = workspace.source_kinds.split_at_mut(spec.mr);
    let b_kinds = &mut remainder[..spec.nr];
    let (a_finite, remainder) = workspace.source_finite.split_at_mut(spec.mr);
    let b_finite = &mut remainder[..spec.nr];
    let (a_words, remainder) = workspace.source_words.split_at_mut(spec.mr);
    let b_words = &mut remainder[..spec.nr];
    let (a_extents, remainder) = workspace.source_extents.split_at_mut(spec.mr);
    let b_extents = &mut remainder[..spec.nr];

    for p in 0..depth {
        workspace.a_active.fill(false);
        workspace.b_active.fill(false);
        let mut a_extent = 0usize;
        let mut b_extent = 0usize;
        let mut has_boundary = false;
        let mut productive_a = false;
        let mut productive_b = false;
        for i in 0..rows {
            match source_a(p, i, ledger) {
                AtlasSource::Raw(code) => {
                    (a_kinds[i], a_finite[i]) = atlas_source_state(code, false);
                    a_extents[i] = 0;
                }
                AtlasSource::Projected(projected) => {
                    a_kinds[i] = projected.kind;
                    if atlas_kind_is_productive(projected.kind) {
                        a_finite[i] = projected.finite_site();
                    }
                    a_words[i] = projected.coordinates;
                    a_extents[i] = projected.extent;
                }
            }
            has_boundary = has_boundary || atlas_kind_is_boundary(a_kinds[i]);
            productive_a = productive_a || atlas_kind_is_productive(a_kinds[i]);
        }
        for j in 0..cols {
            match source_b(p, j, ledger) {
                AtlasSource::Raw(code) => {
                    (b_kinds[j], b_finite[j]) = atlas_source_state(code, false);
                    b_extents[j] = 0;
                }
                AtlasSource::Projected(projected) => {
                    b_kinds[j] = projected.kind;
                    if atlas_kind_is_productive(projected.kind) {
                        b_finite[j] = projected.finite_site();
                    }
                    b_words[j] = projected.coordinates;
                    b_extents[j] = projected.extent;
                }
            }
            has_boundary = has_boundary || atlas_kind_is_boundary(b_kinds[j]);
            productive_b = productive_b || atlas_kind_is_productive(b_kinds[j]);
        }

        if has_boundary {
            cells.for_each_live(&mut |physical_lane, accumulator| {
                let i = physical_lane / spec.nr;
                let j = physical_lane % spec.nr;
                let a = atlas_boundary_code(a_kinds[i]);
                let b = atlas_boundary_code(b_kinds[j]);
                if !a.is_finite() || !b.is_finite() {
                    accumulator.accumulate_one(a, b);
                    ledger.boundary_joined();
                }
            });
        }

        if !productive_a || !productive_b {
            continue;
        }
        for i in 0..rows {
            let first = i * spec.nr;
            workspace.products[first..first + cols].fill(AtlasProduct::ZERO);
        }
        ledger.product_initialized(rows * cols);

        for i in 0..rows {
            if atlas_kind_is_productive(a_kinds[i]) {
                let atom = AtlasAtom {
                    unit: a_finite[i].unit,
                    grade: a_finite[i].grade,
                    negative: a_kinds[i] == AtlasProjectedKind::FiniteNegative as u8,
                };
                let extent = if a_extents[i] != 0 {
                    usize::from(a_extents[i])
                } else {
                    let extent = replace_atlas_word(atom, &mut a_words[i]);
                    ledger.projected();
                    extent
                };
                a_extent = a_extent.max(extent);
                for word in 0..extent {
                    workspace.a_active[word] = workspace.a_active[word] || a_words[i][word] != 0;
                }
            }
        }
        for j in 0..cols {
            if atlas_kind_is_productive(b_kinds[j]) {
                let atom = AtlasAtom {
                    unit: b_finite[j].unit,
                    grade: b_finite[j].grade,
                    negative: b_kinds[j] == AtlasProjectedKind::FiniteNegative as u8,
                };
                let extent = if b_extents[j] != 0 {
                    usize::from(b_extents[j])
                } else {
                    let extent = replace_atlas_word(atom, &mut b_words[j]);
                    ledger.projected();
                    extent
                };
                b_extent = b_extent.max(extent);
                for word in 0..extent {
                    workspace.b_active[word] = workspace.b_active[word] || b_words[j][word] != 0;
                }
            }
        }

        for diagonal in 0..a_extent + b_extent - 1 {
            let first_a = diagonal.saturating_sub(b_extent - 1); // R3-ok: a coordinate interval bound, not an accumulation
            let last_a = diagonal.min(a_extent - 1);
            let mut pair_count = 0usize;
            for ca in first_a..=last_a {
                let cb = diagonal - ca;
                if workspace.a_active[ca] && workspace.b_active[cb] {
                    pair_count += 1;
                }
            }
            if pair_count == 0 {
                continue;
            }

            let mut pair = 0usize;
            for ca in first_a..=last_a {
                let cb = diagonal - ca;
                if !workspace.a_active[ca] || !workspace.b_active[cb] {
                    continue;
                }
                for i in 0..spec.mr {
                    let at = atlas_panel_slot(spec.lane_layout, pair, i, spec.mr, pair_count);
                    workspace.left[at] = if i < rows {
                        if atlas_kind_is_productive(a_kinds[i]) {
                            a_words[i][ca]
                        } else {
                            0
                        }
                    } else {
                        0
                    };
                }
                for j in 0..spec.nr {
                    let at = atlas_panel_slot(spec.lane_layout, pair, j, spec.nr, pair_count);
                    workspace.right[at] = if j < cols {
                        if atlas_kind_is_productive(b_kinds[j]) {
                            b_words[j][cb]
                        } else {
                            0
                        }
                    } else {
                        0
                    };
                }
                pair += 1;
            }

            ledger.kernel_call(spec.mr * spec.nr * pair_count);
            spec.mac_tile(
                pair_count,
                &workspace.left[..spec.mr * pair_count],
                &workspace.right[..spec.nr * pair_count],
                &mut workspace.lanes[..spec.mr * spec.nr],
            );
            for i in 0..rows {
                if !atlas_kind_is_productive(a_kinds[i]) {
                    continue;
                }
                for j in 0..cols {
                    if !atlas_kind_is_productive(b_kinds[j]) {
                        continue;
                    }
                    let physical_lane = i * spec.nr + j;
                    let lane = workspace.lanes[physical_lane];
                    if lane != 0 {
                        workspace.products[physical_lane].add_diagonal(lane, diagonal);
                    }
                }
            }
        }

        cells.for_each_live(&mut |physical_lane, accumulator| {
            let i = physical_lane / spec.nr;
            let j = physical_lane % spec.nr;
            if !atlas_kind_is_productive(a_kinds[i]) || !atlas_kind_is_productive(b_kinds[j]) {
                return;
            }
            let (negative, magnitude) = workspace.products[physical_lane].signed_magnitude();
            let (low, high) = atlas_split_signed_place(magnitude);
            if low != 0 {
                ledger.placed();
                let low = i128::try_from(low).expect("the low coefficient has 127 bits");
                place(
                    accumulator,
                    if negative { -low } else { low },
                    a_finite[i].grade + b_finite[j].grade,
                );
            }
            if high != 0 {
                debug_assert_eq!(high, 1, "a u64 product has one bit above i128::MAX");
                ledger.placed();
                place(
                    accumulator,
                    if negative { -1 } else { 1 },
                    a_finite[i].grade + b_finite[j].grade + i64::from(i128::BITS - 1),
                );
            }
        });
    }
}

/// Execute one bounded panel through the self-similar Atlas contraction.
///
/// Projection itself discovers the finite precision needed by this reduction
/// position.  There is consequently no data-route census, tuned threshold, or
/// second projection pass: another occupied source octet simply
/// applies the same diagonal rule once more.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
fn accumulate_atlas_tile<A, P, Lg>(
    accumulators: &mut [A],
    a_codes: &[PackedCode],
    b_codes: &[PackedCode],
    rows: usize,
    cols: usize,
    depth: usize,
    spec: KernelSpec<i8, i32>,
    place: P,
    ledger: &mut Lg,
) where
    A: SignedPlace,
    P: Fn(&mut A, i128, i64) + Copy,
    Lg: AtlasLedger,
{
    ledger.panel();
    let mut workspace = AtlasTileWorkspace::ZERO;
    let mut cells = SliceAtlasCells {
        rows,
        cols,
        physical_cols: spec.nr,
        accumulators,
    };
    accumulate_direct_atlas_tile(
        &mut workspace,
        &mut cells,
        rows,
        cols,
        depth,
        spec,
        |p, i, _| {
            AtlasSource::Raw(a_codes[atlas_panel_slot(spec.lane_layout, p, i, spec.mr, depth)])
        },
        |p, j, _| {
            AtlasSource::Raw(b_codes[atlas_panel_slot(spec.lane_layout, p, j, spec.nr, depth)])
        },
        place,
        ledger,
    );
}

const UNIT_CODE: PackedCode = PackedCode {
    mantissa: 1,
    exp: 0,
    _pad: 0,
};

/// One physical output tile's borrowed execution state.
///
/// The capacity match has one arm for every representation-derived array
/// extent. Bundling the invariant state before that match keeps those arms to
/// one static call each instead of cloning the same argument setup into every
/// capacity, while the context itself continues to borrow every caller-owned
/// object.
struct AtlasOutputTileContext<'call, 'a, 'b, 'c, E, O, Ep, P, Lg> {
    a: &'call MatView<'a, E>,
    b: &'call MatView<'b, E>,
    c: &'call mut MatViewMut<'c, O>,
    epilogue: &'call Ep,
    options: GemmOptions,
    pa: &'call [PackedCode],
    pb: &'call [PackedCode],
    shape: Shape,
    spec: KernelSpec<i8, i32>,
    i0: usize,
    j0: usize,
    block_start: usize,
    rows: usize,
    cols: usize,
    cached_a_rows: usize,
    b_offer_cols: usize,
    place: P,
    workspace: &'call mut AtlasTileWorkspace,
    ledger: &'call mut Lg,
}

/// Own exactly `CELL_CAP` exact outputs in an isolated stack frame.
///
/// The non-inlined frame prevents the caller's exhaustive capacity dispatch
/// from coalescing every possible array extent into one maximum-sized frame.
#[inline(never)]
fn execute_atlas_output_tile<const CELL_CAP: usize, E, O, Ep, P, Lg>(
    context: &mut AtlasOutputTileContext<'_, '_, '_, '_, E, O, Ep, P, Lg>,
) where
    E: FloatElement,
    O: EncodeFrom<AccOf<E>> + Copy,
    Ep: Epilogue<E, O>,
    AccOf<E>: SignedPlace,
    P: Fn(&mut AccOf<E>, i128, i64) + Copy,
    Lg: AtlasLedger,
{
    debug_assert!(CELL_CAP > 0);
    let mut accumulators = [<AccOf<E> as Accumulator>::ZERO; CELL_CAP];
    execute_atlas_output_tile_body(&mut accumulators, context);
}

/// Execute one physical kernel tile through the caller's exact-capacity frame.
///
/// Keeping the shared traversal outside the const wrapper gives every capacity
/// a distinct stack extent without cloning the contraction for every
/// admissible count.
#[inline(never)]
fn execute_atlas_output_tile_body<E, O, Ep, P, Lg>(
    accumulators: &mut [AccOf<E>],
    context: &mut AtlasOutputTileContext<'_, '_, '_, '_, E, O, Ep, P, Lg>,
) where
    E: FloatElement,
    O: EncodeFrom<AccOf<E>> + Copy,
    Ep: Epilogue<E, O>,
    AccOf<E>: SignedPlace,
    P: Fn(&mut AccOf<E>, i128, i64) + Copy,
    Lg: AtlasLedger,
{
    debug_assert!(!accumulators.is_empty());
    let a = context.a;
    let b = context.b;
    let c = &mut *context.c;
    let epilogue = context.epilogue;
    let options = context.options;
    let pa = context.pa;
    let pb = context.pb;
    let shape = context.shape;
    let spec = context.spec;
    let i0 = context.i0;
    let j0 = context.j0;
    let block_start = context.block_start;
    let rows = context.rows;
    let cols = context.cols;
    let cached_a_rows = context.cached_a_rows;
    let b_offer_cols = context.b_offer_cols;
    let place = context.place;
    let workspace = &mut *context.workspace;
    let ledger = &mut *context.ledger;
    let reads_c = epilogue.reads_c();
    let tile_outputs = rows * cols; // R3-ok: output-cell storage, not an accumulation
    debug_assert_eq!(
        accumulators.len(),
        tile_outputs,
        "the dispatch frame owns every live output exactly once"
    );
    let mut cells = WindowAtlasCells {
        first_logical: 0,
        live_cols: cols,
        physical_cols: spec.nr,
        accumulators,
    };
    ledger.panel();
    accumulate_direct_atlas_tile(
        workspace,
        &mut cells,
        rows,
        cols,
        shape.k,
        spec,
        |p, ii, ledger| {
            if ii < cached_a_rows {
                AtlasSource::Projected(AtlasProjectedCode::from_packed(pa[ii * shape.k + p]))
            } else {
                let code = a.at(i0 + ii, p).pack();
                ledger.decoded_a();
                AtlasSource::Raw(code)
            }
        },
        |p, jj, ledger| {
            if b_offer_cols == 0 {
                let code = b.at(p, j0 + jj).pack();
                ledger.decoded_b();
                AtlasSource::Raw(code)
            } else {
                AtlasSource::Projected(AtlasProjectedCode::from_packed(
                    pb[(j0 + jj - block_start) * shape.k + p],
                ))
            }
        },
        place,
        ledger,
    );

    for (logical_cell, &accumulator) in accumulators.iter().enumerate() {
        let ii = logical_cell / cols;
        let jj = logical_cell % cols;
        let prior = if reads_c {
            Some(*c.at(i0 + ii, j0 + jj))
        } else {
            None
        };
        *c.at_mut(i0 + ii, j0 + jj) = epilogue.finish(accumulator, prior, options.encode);
        ledger.encoded();
    }
}

/// The one tiled float traversal. Caller panels cache the decoded-and-projected
/// Atlas source word in place; execution pulls one reduction position directly
/// into fixed source-word state, so no depth-sized intermediate carrier exists.
#[allow(clippy::too_many_arguments)]
fn gemm_float_tiles_with_selector<E, O, Ep, P, Lg, Select>(
    triple: &mut Triple<'_, '_, '_, E, O>,
    epilogue: &Ep,
    options: GemmOptions,
    pa: &mut [PackedCode],
    pb: &mut [PackedCode],
    place: P,
    ledger: &mut Lg,
    select: Select,
) where
    E: FloatElement,
    O: EncodeFrom<AccOf<E>> + Copy,
    Ep: Epilogue<E, O>,
    AccOf<E>: SignedPlace,
    P: Fn(&mut AccOf<E>, i128, i64) + Copy,
    Lg: AtlasLedger,
    Select: FnOnce(uor_matmul_core::Backend, Shape, usize, usize) -> KernelSpec<i8, i32>,
{
    let shape = triple.shape();
    if shape.m == 0 || shape.n == 0 {
        return;
    }
    if shape.k == 0 {
        let reads_c = epilogue.reads_c();
        let (_, _, c) = triple.parts();
        for i in 0..shape.m {
            for j in 0..shape.n {
                let prior = if reads_c { Some(*c.at(i, j)) } else { None };
                *c.at_mut(i, j) =
                    epilogue.finish(<AccOf<E> as Accumulator>::ZERO, prior, options.encode);
                ledger.encoded();
            }
        }
        return;
    }

    let spec = select(options.backend, shape, pa.len(), pb.len());
    ledger.selected(spec, shape);
    debug_assert_eq!(spec.k_group, 1);
    let (a, b, c) = triple.parts();
    let mut workspace = AtlasTileWorkspace::ZERO;

    let b_offer_cols = pb.len().checked_div(shape.k).unwrap_or(0).min(shape.n);
    let streamed_cols = spec.nr.min(shape.n).max(1);
    let mut block_start = 0usize;
    while block_start < shape.n {
        let block_cols = if b_offer_cols == 0 {
            streamed_cols.min(shape.n - block_start)
        } else {
            b_offer_cols.min(shape.n - block_start)
        };

        if b_offer_cols != 0 {
            for (jj, j) in (block_start..block_start + block_cols).enumerate() {
                let dst = &mut pb[jj * shape.k..(jj + 1) * shape.k];
                for (slot, value) in dst.iter_mut().zip(b.column_walk(0, j, shape.k)) {
                    *slot = cache_atlas_source(value.pack(), ledger);
                    ledger.decoded_b();
                }
            }
        }

        let mut i0 = 0usize;
        while i0 < shape.m {
            let rows = spec.mr.min(shape.m - i0);
            let cached_a_rows = pa.len().checked_div(shape.k).unwrap_or(0).min(rows);
            for ii in 0..cached_a_rows {
                let dst = &mut pa[ii * shape.k..(ii + 1) * shape.k];
                for (slot, value) in dst.iter_mut().zip(a.row_walk(i0 + ii, 0, shape.k)) {
                    *slot = cache_atlas_source(value.pack(), ledger);
                    ledger.decoded_a();
                }
            }

            let mut j0 = block_start;
            while j0 < block_start + block_cols {
                let cols = spec.nr.min(block_start + block_cols - j0);
                let cell_capacity = rows * cols; // R3-ok: output-cell storage, not an accumulation
                let mut context = AtlasOutputTileContext {
                    a,
                    b,
                    c,
                    epilogue,
                    options,
                    pa,
                    pb,
                    shape,
                    spec,
                    i0,
                    j0,
                    block_start,
                    rows,
                    cols,
                    cached_a_rows,
                    b_offer_cols,
                    place,
                    workspace: &mut workspace,
                    ledger,
                };
                macro_rules! execute {
                    ($cell_cap:expr) => {
                        execute_atlas_output_tile::<$cell_cap, E, O, Ep, P, Lg>(&mut context)
                    };
                }
                dispatch_atlas_cell_capacity!(cell_capacity, execute);
                j0 += cols;
            }
            i0 += rows;
        }
        block_start += block_cols;
    }
}

/// Production spelling of the tiled body.
///
/// The selector is a zero-sized closure, so the generic body sees one direct
/// model call. Tests instantiate that same body with a declared candidate to
/// measure every family member without adding a runtime branch or public hook.
fn gemm_float_tiles<E, O, Ep, P, Lg>(
    triple: &mut Triple<'_, '_, '_, E, O>,
    epilogue: &Ep,
    options: GemmOptions,
    pa: &mut [PackedCode],
    pb: &mut [PackedCode],
    place: P,
    ledger: &mut Lg,
) where
    E: FloatElement,
    O: EncodeFrom<AccOf<E>> + Copy,
    Ep: Epilogue<E, O>,
    AccOf<E>: SignedPlace,
    P: Fn(&mut AccOf<E>, i128, i64) + Copy,
    Lg: AtlasLedger,
{
    gemm_float_tiles_with_selector(
        triple,
        epilogue,
        options,
        pa,
        pb,
        place,
        ledger,
        |backend, shape, pa_codes, pb_codes| {
            atlas_tile_spec::<AccOf<E>>(backend, shape, pa_codes, pb_codes)
        },
    );
}

/// Reduce decoded panels through the same source-word diagonal contraction as
/// tiled GEMM.
///
/// This public trait entry cannot borrow a matrix tile or trust a caller's
/// panel facts.  It therefore refines each paired source code in place and
/// immediately places its finite diagonal.  Work is linear in panel depth and
/// in the finite source precision; Laurent grade spread is not a term.
pub(crate) trait AtlasPlaceWide {
    fn place_at_wide(&mut self, value: i128, exponent: i64);
}

impl<const L: usize, const MIN_EXP: i32> AtlasPlaceWide for Complete<L, MIN_EXP> {
    #[inline(always)]
    fn place_at_wide(&mut self, value: i128, exponent: i64) {
        let shift = exponent - i64::from(MIN_EXP); // R3-ok: a coordinate address, not an accumulation
        let Ok(shift) = usize::try_from(shift) else {
            return;
        };
        let low_bits = self.raw().limbs().len().saturating_mul(u64::BITS as usize); // R3-ok: a storage extent, not an accumulation
        if shift >= low_bits {
            return;
        }
        let Ok(exponent) = i32::try_from(exponent) else {
            return;
        };
        self.add_scaled(value.unsigned_abs(), exponent, value < 0);
    }
}

#[allow(clippy::needless_range_loop)]
fn accumulate_atlas<A, FA, FB>(
    acc: &mut A,
    depth: usize,
    panels: PanelFacts,
    spec: KernelSpec<i8, i32>,
    mut source_a: FA,
    mut source_b: FB,
) where
    A: SignedPlace + AtlasPlaceWide,
    FA: FnMut(usize) -> PackedCode,
    FB: FnMut(usize) -> PackedCode,
{
    debug_assert_eq!(spec.mr, 1);
    debug_assert_eq!(spec.nr, 1);
    let prescaled = panels.prescaled.is_some();
    let exponent_offset = panels.prescaled.map_or(0, |scale| i64::from(scale.base));
    let mut a_coordinates = [0i8; MAX_ATLAS_WORDS];
    let mut b_coordinates = [0i8; MAX_ATLAS_WORDS];
    let mut left = [0i8; MAX_ATLAS_WORDS];
    let mut right = [0i8; MAX_ATLAS_WORDS];
    let mut lane = [0i32; 1];

    for p in 0..depth {
        let a_code = source_a(p);
        let b_code = source_b(p);
        if !a_code.is_finite() || !b_code.is_finite() {
            acc.accumulate_one(a_code, b_code);
            continue;
        }
        let (Some(a), Some(b)) = (atlas_atom(a_code, prescaled), atlas_atom(b_code, prescaled))
        else {
            continue;
        };
        let a_extent = atlas_word(a, &mut a_coordinates);
        let b_extent = atlas_word(b, &mut b_coordinates);
        let mut product = AtlasProduct::ZERO;

        for diagonal in 0..a_extent + b_extent - 1 {
            let first_a = diagonal.saturating_sub(b_extent - 1); // R3-ok: a coordinate interval bound, not an accumulation
            let last_a = diagonal.min(a_extent - 1);
            let mut pair_count = 0usize;
            for ca in first_a..=last_a {
                let cb = diagonal - ca;
                if a_coordinates[ca] != 0 && b_coordinates[cb] != 0 {
                    left[pair_count] = a_coordinates[ca];
                    right[pair_count] = b_coordinates[cb];
                    pair_count += 1;
                }
            }
            if pair_count == 0 {
                continue;
            }
            spec.mac_tile(
                pair_count,
                &left[..pair_count],
                &right[..pair_count],
                &mut lane,
            );
            if lane[0] != 0 {
                product.add_diagonal(lane[0], diagonal);
            }
        }
        let (negative, magnitude) = product.signed_magnitude();
        let exponent = exponent_offset + a.grade + b.grade;
        let (low, high) = atlas_split_signed_place(magnitude);
        if low != 0 {
            let low = i128::try_from(low).expect("the low coefficient has 127 bits");
            acc.place_at_wide(if negative { -low } else { low }, exponent);
        }
        if high != 0 {
            acc.place_at_wide(
                if negative { -1 } else { 1 },
                exponent + i64::from(i128::BITS - 1),
            );
        }
    }
}

/// Contract one decoded dot product through the bounded balanced-octet Atlas
/// engine.
///
/// This is the one-output spelling used when another traversal already owns
/// its matrix/page walk. It selects one group-one reduction declaration and
/// retains only bounded nine-word coordinate/diagonal buffers and one
/// three-limb local product; it never instantiates the tiled engine's source or
/// lane arrays.
pub(crate) fn accumulate_atlas_dot<A, FA, FB>(
    acc: &mut A,
    depth: usize,
    panels: PanelFacts,
    backend: uor_matmul_core::Backend,
    source_a: FA,
    source_b: FB,
) where
    A: SignedPlace + AtlasPlaceWide,
    FA: FnMut(usize) -> PackedCode,
    FB: FnMut(usize) -> PackedCode,
{
    accumulate_atlas(
        acc,
        depth,
        panels,
        atlas_dot_spec(backend),
        source_a,
        source_b,
    );
}

#[inline]
fn accumulate_atlas_panels<A>(acc: &mut A, pa: &[PackedCode], pb: &[PackedCode], panels: PanelFacts)
where
    A: SignedPlace + AtlasPlaceWide,
{
    accumulate_atlas_dot(
        acc,
        pa.len().max(pb.len()),
        panels,
        uor_matmul_core::Backend::Auto,
        |p| pa.get(p).copied().unwrap_or(ZERO_CODE),
        |p| pb.get(p).copied().unwrap_or(ZERO_CODE),
    );
}

/// `C := epilogue(A * B, C)`, over float operands, computed exactly.
///
/// Returns `()`, for the same reason [`crate::gemm`] does: the requested
/// product exists, because a [`Triple`] exists (R14, C6). Non-finite inputs are
/// codes and propagate by the IEEE rules; they are not an error condition
/// (`CT-03`).
///
/// With no caller cache to offer, decoded codes are streamed into bounded
/// execution panels. The same model-derived lookup route, coordinate
/// contraction, exact placement, and terminal encode run in either case
/// (`CD-30`).
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
    gemm_float_tiles(
        triple,
        epilogue,
        options,
        &mut [],
        &mut [],
        |acc, lane, exponent| {
            let negative = lane < 0;
            let magnitude = lane.unsigned_abs();
            for (word, offset) in [
                ((magnitude % ATLAS_LIMB_RADIX) as u64, 0),
                ((magnitude / ATLAS_LIMB_RADIX) as u64, u64::BITS),
            ] {
                if word == 0 {
                    continue;
                }
                let exponent = exponent + i64::from(offset);
                assert!(
                    (i64::from(E::MIN_PRODUCT_EXP)..=i64::from(E::MAX_PRODUCT_EXP))
                        .contains(&exponent),
                    "a FloatElement must declare its complete finite product exponent range"
                );
                acc.accumulate_one(
                    PackedCode::of(uor_matmul_core::Decoded::Finite {
                        sign: negative,
                        mantissa: word,
                        exp: i32::try_from(exponent).expect(
                            "a declared FloatElement product exponent is representable as i32",
                        ),
                    }),
                    UNIT_CODE,
                );
            }
        },
        &mut (),
    );
}

/// The same operation with caller-owned Atlas-word caches.
///
/// A float is a code, and projection is real work. Each offered sixteen-byte
/// slot is filled in place with the code's exact nine-octet/grade view, so every
/// spatial tile reuses both decode and projection without allocating or
/// changing the source value.
///
/// The panels are the caller's, so this still allocates nothing. Offering none
/// streams bounded panels through the identical body and gives the same bytes
/// (S13, `CD-30`).
///
/// # Which factorization runs
///
/// The selected lookup kernel contracts each source word along its Laurent
/// diagonals. The offer affects source-word reuse only; it cannot select a
/// different answer or a non-Atlas operation (`CG-22`, R13).
pub fn gemm_float_packed<E, O, Ep>(
    triple: &mut Triple<'_, '_, '_, E, O>,
    epilogue: &Ep,
    options: GemmOptions,
    pa: &mut [PackedCode],
    pb: &mut [PackedCode],
) where
    E: FloatElement,
    O: Element + EncodeFrom<AccOf<E>> + EncodeFrom<i128> + Copy,
    Ep: Epilogue<E, O>,
    AccOf<E>: SignedPlace + PlaceAt,
{
    gemm_float_tiles(
        triple,
        epilogue,
        options,
        pa,
        pb,
        |acc, lane, exponent| {
            assert!(
                (i64::from(E::MIN_PRODUCT_EXP)..=i64::from(E::MAX_PRODUCT_EXP)).contains(&exponent),
                "a FloatElement must declare its complete finite product exponent range"
            );
            acc.place_at(
                lane,
                i32::try_from(exponent)
                    .expect("a declared FloatElement product exponent is representable as i32"),
            );
        },
        &mut (),
    );
}

/// Compatibility size query for the two former scaled operand regions.
///
/// The pure Atlas engine reifies neither operand, but this public query keeps
/// its established value so existing caller allocations and layout arithmetic
/// remain unchanged.
pub fn suggested_bridge_scaled(shape: Shape) -> usize {
    shape.k.saturating_mul(shape.m.saturating_add(shape.n)) // R3-ok: a compatibility scratch size query
}

const WORDS_PER_CODE: usize = core::mem::size_of::<PackedCode>() / core::mem::size_of::<i32>();

/// The established full-reuse panel offer `(pa, pb)`, in [`PackedCode`]-sized
/// elements.
///
/// A *query*, like [`crate::suggested_scratch`] --- offering less is not an
/// error and cannot select a lesser computation (`CD-30`). Its value remains
/// byte-for-byte compatible with the public query predating the Atlas engine;
/// the live traversal treats both slices solely as optional in-place Atlas
/// projection caches.
pub fn suggested_float_panels(shape: Shape) -> (usize, usize) {
    let scalar = shape.k.saturating_mul(shape.n); // R3-ok: a scratch size query
    let bridged = suggested_bridge_scaled(shape)
        .saturating_add(crate::suggested_scratch(shape)) // R3-ok: a scratch size query
        .div_ceil(WORDS_PER_CODE);
    (shape.k, scalar.max(bridged))
}

/// Compatibility spelling for the same pure-Atlas float operation.
///
/// The former whole-operand integer reification has been removed. This entry
/// delegates directly to [`gemm_float_packed`], so every public spelling uses
/// the same lazy signed-coordinate projection, lookup/add contraction, exact
/// accumulator, and single terminal encode (`CD-30`, R13).
///
/// # Offers
///
/// `pa` and `pb` remain ordinary same-size source-cache offers. `scaled` and `scratch`
/// are retained only because the public signature is locked; the pure Atlas
/// engine neither reads nor writes them, at any length.
#[allow(clippy::too_many_arguments)]
pub fn gemm_float_bridged<E, O, Ep>(
    triple: &mut Triple<'_, '_, '_, E, O>,
    epilogue: &Ep,
    options: GemmOptions,
    pa: &mut [PackedCode],
    pb: &mut [PackedCode],
    scaled: &mut [i32],
    scratch: &mut Scratch<'_, i32, Full<i32>>,
) where
    E: FloatElement,
    O: Element + EncodeFrom<AccOf<E>> + EncodeFrom<i128> + Copy,
    Ep: Epilogue<E, O>,
    AccOf<E>: SignedPlace + PlaceAt,
{
    // The historic spelling is retained exactly, but its whole-operand `i32`
    // reification no longer exists.  All caller offers are optional; the code
    // panels are consumed directly by the Atlas engine and the established
    // scaled and integer-kernel buffers remain untouched compatibility offers.
    let _ = (scaled, scratch);
    gemm_float_packed(triple, epilogue, options, pa, pb);
}

/// What a caller has already established about a pair of decoded panels.
///
/// No fact can change a value. `prescaled` says the supplied codes already
/// share one Laurent base. The retained `finite` and `product_fits` fields
/// described removed trusted paths and have no effect on the pure lookup/add
/// contraction; public facts are always revalidated from the codes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PanelFacts {
    /// Compatibility declaration from the former trusted panel path; ignored.
    pub finite: bool,
    /// Compatibility declaration from the former integer lane; ignored.
    pub product_fits: bool,
    /// Both panels already use a common Laurent base. See [`Prescaled`].
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

/// A common Laurent base already applied to both panels.
///
/// The Atlas projector treats the stored coefficients as grade-zero words and
/// restores `base` only when a nonzero lookup sum is placed. This is the same
/// quotient representative with a shared gauge, not another multiplication
/// lane. The `wide` field is retained for API compatibility and is ignored.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Prescaled {
    /// The exponent of bit 0 of every scaled product: `base_a + base_b`.
    pub base: i32,
    /// Compatibility declaration from the removed integer lane; ignored.
    pub wide: bool,
}

/// The exponent span of a panel, and the base its scaling starts from.
///
/// Zero significands are not seen: a zero contributes nothing to the sum at any
/// scale, so it cannot widen the span and must not.
///
/// `pub(crate)` for the symbol lane's span walk (`CD-20`): the tabulated
/// traversal measures the same spans by the same walk, and admission is the
/// same declaration.
#[derive(Clone, Copy)]
pub(crate) struct Span {
    min: i32,
    max: i32,
    any: bool,
}

impl Span {
    pub(crate) const EMPTY: Self = Self {
        min: i32::MAX,
        max: i32::MIN,
        any: false,
    };

    #[inline(always)]
    pub(crate) fn see(&mut self, code: PackedCode) {
        if let Some((_, magnitude, exponent)) = finite_parts(code) {
            if magnitude != 0 {
                self.min = self.min.min(exponent);
                self.max = self.max.max(exponent);
                self.any = true;
            }
        }
    }

    pub(crate) fn base(&self) -> i32 {
        if self.any {
            self.min
        } else {
            0
        }
    }

    /// How many bits a significand of this panel gains from the scaling.
    pub(crate) fn width(&self) -> u32 {
        if self.any {
            self.max.wrapping_sub(self.min) as u32
        } else {
            0
        }
    }
}

/// What the packed float loop needs from an accumulator.
///
/// A trait rather than an inherent method so that `gemm_float_packed` stays
/// generic over the element type while the hot path stays monomorphic.
pub trait SignedPlace {
    /// Accumulate a whole dot product of two decoded panels, exactly.
    ///
    /// Unequal panels are zero-extended to the longer length. This is the
    /// algebraic totalization of a missing coordinate, including IEEE boundary
    /// products such as infinity times that implicit zero.
    ///
    /// Every entry must be a valid packed code for the accumulator's element
    /// format, including that format's declared finite product-exponent range.
    /// [`gemm_float_packed`] establishes this contract by decoding `E`; callers
    /// that invoke this compatibility trait directly establish the same fact.
    /// A [`Prescaled`] declaration establishes that restoring its base keeps
    /// each nonzero product inside that same range. Coordinates outside a
    /// hand-built `Complete` register name no stored bit and are ignored, which
    /// is the total semantics of `Complete::add_scaled`; shipped typed products
    /// cannot reach that case.
    fn accumulate_panels(&mut self, pa: &[PackedCode], pb: &[PackedCode], panels: PanelFacts);
    /// Accumulate one product of two decoded codes.
    fn accumulate_one(&mut self, a: PackedCode, b: PackedCode);
}

impl<const L: usize, const MIN_EXP: i32> SignedPlace for Complete<L, MIN_EXP> {
    #[inline]
    fn accumulate_panels(&mut self, pa: &[PackedCode], pb: &[PackedCode], panels: PanelFacts) {
        accumulate_atlas_panels(self, pa, pb, panels);
    }

    #[inline]
    fn accumulate_one(&mut self, a: PackedCode, b: PackedCode) {
        if a.is_finite() && b.is_finite() {
            // Tile lanes are already complete Atlas coordinate sums.  The
            // multiplicative unit is their typed placement channel, so this
            // is one group placement rather than another product reduction.
            if b == UNIT_CODE {
                let (negative, magnitude, exponent) =
                    finite_parts(a).expect("the finite branch has finite fields");
                self.add_scaled(u128::from(magnitude), exponent, negative);
                return;
            }
            if a == UNIT_CODE {
                let (negative, magnitude, exponent) =
                    finite_parts(b).expect("the finite branch has finite fields");
                self.add_scaled(u128::from(magnitude), exponent, negative);
                return;
            }
            accumulate_atlas_panels(
                self,
                core::slice::from_ref(&a),
                core::slice::from_ref(&b),
                PanelFacts::UNKNOWN,
            );
            return;
        }
        if a.is_nan() || b.is_nan() {
            self.set_nan();
            return;
        }
        let (inf, other) = if a.is_infinite() { (a, b) } else { (b, a) };
        let finite_other = finite_parts(other);
        if finite_other.is_some_and(|(_, magnitude, _)| magnitude == 0) {
            self.set_nan();
        } else {
            let other_negative =
                finite_other.map_or(other.mantissa < 0, |(negative, _, _)| negative);
            self.set_infinity((inf.mantissa < 0) != other_negative);
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
    use crate::epilogue::{AbsorbPrior, Linear, ScaleExact};
    use std::eprintln;
    use std::vec;
    use std::vec::Vec;
    use uor_matmul_core::{Backend, Decoded, EncodeMode, MatView, MatViewMut};

    macro_rules! test_float_element {
        ($name:ident, $min_product_exp:expr, $max_product_exp:expr) => {
            #[derive(Clone, Copy, Debug, PartialEq)]
            struct $name(f64);

            impl Element for $name {
                type Acc = <f64 as Element>::Acc;
                const BITS: u32 = <f64 as Element>::BITS;
                const ZERO: Self = Self(0.0);

                fn mac(acc: &mut Self::Acc, a: Self, b: Self) {
                    <f64 as Element>::mac(acc, a.0, b.0);
                }

                fn combine_narrow(acc: Self::Acc, _: i64) -> Self::Acc {
                    acc
                }
            }

            impl FloatElement for $name {
                const SIGNIFICAND_BITS: u32 = f64::MANTISSA_DIGITS;
                const MIN_PRODUCT_EXP: i32 = $min_product_exp;
                const MAX_PRODUCT_EXP: i32 = $max_product_exp;

                fn decode(self) -> Decoded {
                    self.0.decode()
                }

                fn symbol_bits(self) -> u64 {
                    self.0.to_bits()
                }
            }
        };
    }

    test_float_element!(
        BoundaryFloat,
        <f64 as FloatElement>::MIN_PRODUCT_EXP,
        <f64 as FloatElement>::MAX_PRODUCT_EXP
    );
    test_float_element!(DishonestFloat, 0, 0);

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct SentinelGradeFloat {
        exponent: i32,
        magnitude: u64,
    }

    impl Element for SentinelGradeFloat {
        type Acc = Complete<2, { PackedCode::NAN_EXP }>;
        const BITS: u32 = u64::BITS;
        const ZERO: Self = Self {
            exponent: 0,
            magnitude: 0,
        };

        fn mac(acc: &mut Self::Acc, a: Self, b: Self) {
            SignedPlace::accumulate_one(acc, a.pack(), b.pack());
        }

        fn combine_narrow(acc: Self::Acc, _: i64) -> Self::Acc {
            acc
        }
    }

    impl FloatElement for SentinelGradeFloat {
        const SIGNIFICAND_BITS: u32 = 1;
        const MIN_PRODUCT_EXP: i32 = PackedCode::NAN_EXP;
        const MAX_PRODUCT_EXP: i32 = PackedCode::INF_EXP;

        fn decode(self) -> Decoded {
            Decoded::Finite {
                sign: false,
                mantissa: self.magnitude,
                exp: self.exponent,
            }
        }

        fn symbol_bits(self) -> u64 {
            u64::from(self.exponent as u32) << 32 | self.magnitude
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct SentinelOutput(u8);

    impl Element for SentinelOutput {
        type Acc = i128;
        const BITS: u32 = u8::BITS;
        const ZERO: Self = Self(0);

        fn mac(acc: &mut Self::Acc, a: Self, b: Self) {
            *acc += i128::from(a.0) * i128::from(b.0);
        }

        fn combine_narrow(acc: Self::Acc, narrow: i64) -> Self::Acc {
            acc + i128::from(narrow)
        }
    }

    impl EncodeFrom<i128> for SentinelOutput {
        fn encode_from(acc: i128, _: EncodeMode) -> Self {
            Self(acc as u8)
        }
    }

    impl EncodeFrom<Complete<2, { PackedCode::NAN_EXP }>> for SentinelOutput {
        fn encode_from(acc: Complete<2, { PackedCode::NAN_EXP }>, _: EncodeMode) -> Self {
            let mut nan_grade = Complete::ZERO;
            nan_grade.add_scaled(1, PackedCode::NAN_EXP, false);
            let mut infinity_grade = Complete::ZERO;
            infinity_grade.add_scaled(1, PackedCode::INF_EXP, false);
            if acc == nan_grade {
                Self(1)
            } else if acc == infinity_grade {
                Self(2)
            } else {
                Self(0)
            }
        }
    }

    struct SentinelOverwrite;

    impl Epilogue<SentinelGradeFloat, SentinelOutput> for SentinelOverwrite {
        fn finish(
            &self,
            acc: AccOf<SentinelGradeFloat>,
            _: Option<SentinelOutput>,
            mode: EncodeMode,
        ) -> SentinelOutput {
            SentinelOutput::encode_from(acc, mode)
        }

        fn reads_c(&self) -> bool {
            false
        }
    }

    /// `CK-21`: escaped finite codes keep their full modality and coefficient
    /// through both the direct Atlas contraction and the non-finite boundary.
    #[test]
    fn escaped_finite_codes_have_one_atlas_interpretation_ck_21() {
        type Acc = <f64 as Element>::Acc;

        let alternate_unit = PackedCode::of(Decoded::Finite {
            sign: false,
            mantissa: 2,
            exp: -1,
        });
        for (negative, nearest) in [(false, 1.0f64), (true, -1.0f64)] {
            let code = PackedCode::of(Decoded::Finite {
                sign: negative,
                mantissa: u64::MAX,
                exp: -64,
            });
            assert_ne!(code._pad, 0);
            let mut acc = <Acc as Accumulator>::ZERO;
            acc.accumulate_one(code, alternate_unit);
            assert_eq!(f64::encode_from(acc, EncodeMode::Nearest), nearest);
        }

        let positive_at_sentinel = PackedCode::of(Decoded::Finite {
            sign: false,
            mantissa: u64::MAX,
            exp: PackedCode::INF_EXP,
        });
        let negative_at_sentinel = PackedCode::of(Decoded::Finite {
            sign: true,
            mantissa: u64::MAX,
            exp: PackedCode::NAN_EXP,
        });
        for (finite, expected) in [
            (positive_at_sentinel, f64::INFINITY),
            (negative_at_sentinel, f64::NEG_INFINITY),
        ] {
            let mut acc = <Acc as Accumulator>::ZERO;
            acc.accumulate_one(PackedCode::of(f64::INFINITY.decode()), finite);
            assert_eq!(f64::encode_from(acc, EncodeMode::Nearest), expected);
        }

        let escaped_negative_zero = PackedCode::of(Decoded::Finite {
            sign: true,
            mantissa: 0,
            exp: PackedCode::INF_EXP,
        });
        let mut acc = <Acc as Accumulator>::ZERO;
        acc.accumulate_one(
            PackedCode::of(f64::INFINITY.decode()),
            escaped_negative_zero,
        );
        assert!(f64::encode_from(acc, EncodeMode::Nearest).is_nan());

        let historical_padding = PackedCode {
            mantissa: -5,
            exp: PackedCode::INF_EXP + 1,
            _pad: 7,
        };
        assert_eq!(
            finite_parts(historical_padding),
            Some((true, 5, PackedCode::INF_EXP + 1))
        );
        assert_eq!(
            finite_parts(PackedCode {
                exp: PackedCode::INF_EXP,
                ..historical_padding
            }),
            None
        );
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct AtlasRoute {
        backend: Backend,
        mr: usize,
        nr: usize,
        k_group: usize,
        products_per_step: usize,
        lane_layout: uor_matmul_kernels::LaneLayout,
    }

    impl From<KernelSpec<i8, i32>> for AtlasRoute {
        fn from(spec: KernelSpec<i8, i32>) -> Self {
            Self {
                backend: spec.backend,
                mr: spec.mr,
                nr: spec.nr,
                k_group: spec.k_group,
                products_per_step: spec.products_per_step,
                lane_layout: spec.lane_layout,
            }
        }
    }

    /// `CD-30`: a parametric float declaration spanning the real source
    /// format's complete product range reaches both endpoint products without
    /// a route-specific narrowing or rejection.
    #[test]
    fn float_wrapper_accepts_declared_product_boundaries_cd_30() {
        for (input, expected) in [(f64::MAX, f64::INFINITY), (f64::from_bits(1), 0.0)] {
            let a = [BoundaryFloat(input)];
            let b = [BoundaryFloat(input)];
            let mut c = [f64::NAN];
            let av = MatView::row_major(&a, 1, 1).unwrap();
            let bv = MatView::row_major(&b, 1, 1).unwrap();
            let cv = MatViewMut::row_major(&mut c, 1, 1).unwrap();
            let mut triple = Triple::new(av, bv, cv).unwrap();
            gemm_float(
                &mut triple,
                &Linear::OVERWRITE,
                GemmOptions {
                    backend: Backend::Portable,
                    ..GemmOptions::default()
                },
            );
            assert_eq!(c[0].to_bits(), expected.to_bits());
        }
    }

    /// `CG-22`: an element declaration that excludes a coordinate its own
    /// decoder emits is rejected at the generic public boundary, rather than
    /// saturating that Laurent address into a different product.
    #[test]
    fn float_wrapper_rejects_a_dishonest_declared_range_cd_30() {
        let result = std::panic::catch_unwind(|| {
            let a = [DishonestFloat(2.0)];
            let b = [DishonestFloat(2.0)];
            let mut c = [0.0f64];
            let av = MatView::row_major(&a, 1, 1).unwrap();
            let bv = MatView::row_major(&b, 1, 1).unwrap();
            let cv = MatViewMut::row_major(&mut c, 1, 1).unwrap();
            let mut triple = Triple::new(av, bv, cv).unwrap();
            gemm_float(
                &mut triple,
                &Linear::OVERWRITE,
                GemmOptions {
                    backend: Backend::Portable,
                    ..GemmOptions::default()
                },
            );
        });
        assert!(
            result.is_err(),
            "a dishonest FloatElement declaration must not alter an address"
        );
    }

    /// `CD-30`, `CK-21`: finite product grades equal to the packed non-finite
    /// sentinels retain their finite escape tag in both workspace spellings.
    #[test]
    fn sentinel_product_grades_are_finite_in_both_workspace_spellings_cd_30() {
        for (grade, expected) in [
            (PackedCode::NAN_EXP, SentinelOutput(1)),
            (PackedCode::INF_EXP, SentinelOutput(2)),
        ] {
            let left_exponent = grade.div_euclid(2);
            let right_exponent = grade - left_exponent;
            let a = [SentinelGradeFloat {
                exponent: left_exponent,
                magnitude: 1,
            }];
            let b = [SentinelGradeFloat {
                exponent: right_exponent,
                magnitude: 1,
            }];
            let run = |packed: bool| {
                let mut output = [SentinelOutput::ZERO];
                let av = MatView::row_major(&a, 1, 1).unwrap();
                let bv = MatView::row_major(&b, 1, 1).unwrap();
                let cv = MatViewMut::row_major(&mut output, 1, 1).unwrap();
                let mut triple = Triple::new(av, bv, cv).unwrap();
                let options = GemmOptions {
                    backend: Backend::Portable,
                    ..GemmOptions::default()
                };
                if packed {
                    let mut pa = [PackedCode::default(); 1];
                    let mut pb = [PackedCode::default(); 1];
                    gemm_float_packed(&mut triple, &SentinelOverwrite, options, &mut pa, &mut pb);
                } else {
                    gemm_float(&mut triple, &SentinelOverwrite, options);
                }
                output[0]
            };
            let streamed = run(false);
            let cached = run(true);
            assert_eq!(streamed, cached, "grade {grade}");
            assert_eq!(streamed, expected, "grade {grade}");
        }
    }

    #[derive(Default, Debug)]
    struct AtlasCensus {
        route: Option<AtlasRoute>,
        panels: usize,
        decoded_a: usize,
        decoded_b: usize,
        issued_steps: usize,
        coordinate_products: usize,
        coordinate_additions: usize,
        kernel_calls: usize,
        projections: usize,
        product_initializations: usize,
        placements: usize,
        boundary_joins: usize,
        encodes: usize,
    }

    impl AtlasLedger for AtlasCensus {
        fn selected(&mut self, spec: KernelSpec<i8, i32>, _: Shape) {
            assert!(self.route.replace(spec.into()).is_none());
        }

        fn panel(&mut self) {
            self.panels += 1;
        }

        fn decoded_a(&mut self) {
            self.decoded_a += 1;
        }

        fn decoded_b(&mut self) {
            self.decoded_b += 1;
        }

        fn projected(&mut self) {
            self.projections += 1;
        }

        fn product_initialized(&mut self, live_products: usize) {
            self.product_initializations += live_products;
        }

        fn kernel_call(&mut self, coordinate_products: usize) {
            self.kernel_calls += 1;
            self.issued_steps += coordinate_products.div_ceil(
                self.route
                    .expect("the Atlas route is selected before its first call")
                    .products_per_step,
            );
            self.coordinate_products += coordinate_products;
            // Every addressed coordinate is combined once into its exact
            // reduction lane. Vector instructions factor these semantic adds;
            // they do not change their census.
            self.coordinate_additions += coordinate_products;
        }

        fn placed(&mut self) {
            self.placements += 1;
        }

        fn boundary_joined(&mut self) {
            self.boundary_joins += 1;
        }

        fn encoded(&mut self) {
            self.encodes += 1;
        }
    }

    fn counted_uniform_with_panels<E>(
        shape: Shape,
        backend: Backend,
        unit: E,
        pa_len: usize,
        pb_len: usize,
    ) -> (Vec<E>, AtlasCensus)
    where
        E: FloatElement + EncodeFrom<AccOf<E>> + Copy,
        AccOf<E>: SignedPlace + PlaceAt + ScaleExact + AbsorbPrior<E>,
    {
        let a = vec![unit; shape.m * shape.k];
        let b = vec![unit; shape.k * shape.n];
        let mut c = vec![E::ZERO; shape.m * shape.n];
        let av = MatView::row_major(&a, shape.m, shape.k).unwrap();
        let bv = MatView::row_major(&b, shape.k, shape.n).unwrap();
        let cv = MatViewMut::row_major(&mut c, shape.m, shape.n).unwrap();
        let mut triple = Triple::new(av, bv, cv).unwrap();
        let mut census = AtlasCensus::default();
        let mut pa = vec![PackedCode::default(); pa_len];
        let mut pb = vec![PackedCode::default(); pb_len];
        gemm_float_tiles(
            &mut triple,
            &Linear::OVERWRITE,
            GemmOptions {
                backend,
                ..GemmOptions::default()
            },
            &mut pa,
            &mut pb,
            |acc, lane, exponent| acc.place_at(lane, i32::try_from(exponent).unwrap()),
            &mut census,
        );
        (c, census)
    }

    fn counted_uniform<E>(shape: Shape, backend: Backend, unit: E) -> (Vec<E>, AtlasCensus)
    where
        E: FloatElement + EncodeFrom<AccOf<E>> + Copy,
        AccOf<E>: SignedPlace + PlaceAt + ScaleExact + AbsorbPrior<E>,
    {
        counted_uniform_with_panels(shape, backend, unit, 0, 0)
    }

    fn candidate_reference<E>(shape: Shape, a: &[E], b: &[E]) -> Vec<E>
    where
        E: FloatElement + EncodeFrom<AccOf<E>> + Copy,
    {
        let mut output = vec![E::ZERO; shape.m * shape.n];
        for i in 0..shape.m {
            for j in 0..shape.n {
                let mut accumulator = <AccOf<E> as Accumulator>::ZERO;
                for p in 0..shape.k {
                    E::mac(&mut accumulator, a[i * shape.k + p], b[p * shape.n + j]);
                }
                output[i * shape.n + j] = E::encode_from(accumulator, EncodeMode::Nearest);
            }
        }
        output
    }

    #[derive(Clone, Copy, Debug)]
    enum CandidateFill {
        OneGrade,
        FewGrades,
        InverseGauge,
        FullFiniteRange,
        DenseSignificand,
        SparseSignificand,
    }

    #[derive(Clone, Copy, Debug)]
    struct CandidateCase {
        shape: Shape,
        fill: CandidateFill,
        seed: u64,
    }

    const CANDIDATE_CASES: [CandidateCase; 6] = [
        CandidateCase {
            shape: Shape { m: 1, k: 1, n: 1 },
            fill: CandidateFill::OneGrade,
            seed: 101,
        },
        CandidateCase {
            shape: Shape {
                m: 32,
                k: 32,
                n: 32,
            },
            fill: CandidateFill::FewGrades,
            seed: 102,
        },
        CandidateCase {
            shape: Shape {
                m: 16,
                k: 128,
                n: 16,
            },
            fill: CandidateFill::InverseGauge,
            seed: 103,
        },
        CandidateCase {
            shape: Shape { m: 7, k: 31, n: 5 },
            fill: CandidateFill::FullFiniteRange,
            seed: 104,
        },
        CandidateCase {
            shape: Shape {
                m: 1,
                k: 65_536,
                n: 1,
            },
            fill: CandidateFill::DenseSignificand,
            seed: 105,
        },
        CandidateCase {
            shape: Shape {
                m: 128,
                k: 8,
                n: 128,
            },
            fill: CandidateFill::SparseSignificand,
            seed: 106,
        },
    ];

    trait CandidateCorpusFloat: FloatElement + Copy {
        const LABEL: &'static str;
        const FRACTION_BITS: u32;
        const EXPONENT_BITS: u32;
        fn from_candidate_bits(bits: u64) -> Self;
    }

    impl CandidateCorpusFloat for f32 {
        const LABEL: &'static str = "f32";
        const FRACTION_BITS: u32 = 23;
        const EXPONENT_BITS: u32 = 8;

        fn from_candidate_bits(bits: u64) -> Self {
            Self::from_bits(bits as u32)
        }
    }

    impl CandidateCorpusFloat for f64 {
        const LABEL: &'static str = "f64";
        const FRACTION_BITS: u32 = 52;
        const EXPONENT_BITS: u32 = 11;

        fn from_candidate_bits(bits: u64) -> Self {
            Self::from_bits(bits)
        }
    }

    #[derive(Clone, Copy)]
    enum CandidateSide {
        A,
        B,
    }

    fn candidate_mix(mut value: u64) -> u64 {
        value ^= value >> 30;
        value = value.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value ^= value >> 27;
        value = value.wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn candidate_low_mask(bits: u32) -> u64 {
        u64::MAX >> (u64::BITS - bits)
    }

    fn candidate_code<E: CandidateCorpusFloat>(
        case: CandidateCase,
        side: CandidateSide,
        outer: usize,
        p: usize,
    ) -> E {
        let side_salt = match side {
            CandidateSide::A => 0xA24B_AED4_963E_E407,
            CandidateSide::B => 0x9FB2_1C65_1E98_DF25,
        };
        let random = candidate_mix(
            case.seed
                ^ side_salt
                ^ (outer as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93)
                ^ (p as u64).wrapping_mul(0xA5A3_564E_27F8_862D),
        );
        let fraction_mask = candidate_low_mask(E::FRACTION_BITS);
        let exponent_max = candidate_low_mask(E::EXPONENT_BITS);
        let bias = exponent_max >> 1;
        let sign = (random >> 63) << (E::FRACTION_BITS + E::EXPONENT_BITS);
        let (exponent, fraction) = match case.fill {
            CandidateFill::OneGrade => (bias, random & fraction_mask),
            CandidateFill::FewGrades => {
                let delta = (random % 7) as i64 - 3;
                ((bias as i64 + delta) as u64, random & fraction_mask)
            }
            CandidateFill::InverseGauge => {
                let radius = (exponent_max / 8).max(1);
                let delta = (p as u64 % (2 * radius + 1)) as i64 - radius as i64;
                let signed = match side {
                    CandidateSide::A => delta,
                    CandidateSide::B => -delta,
                };
                ((bias as i64 + signed) as u64, random & fraction_mask)
            }
            CandidateFill::FullFiniteRange => {
                let finite_exponents = exponent_max - 1;
                (1 + random % finite_exponents, random & fraction_mask)
            }
            CandidateFill::SparseSignificand => {
                let bit = random as u32 % E::FRACTION_BITS;
                (bias, 1u64 << bit)
            }
            CandidateFill::DenseSignificand => (
                bias,
                fraction_mask ^ (1 << (random % E::FRACTION_BITS as u64)),
            ),
        };
        E::from_candidate_bits(sign | (exponent << E::FRACTION_BITS) | fraction)
    }

    fn candidate_operands<E: CandidateCorpusFloat>(case: CandidateCase) -> (Vec<E>, Vec<E>) {
        let shape = case.shape;
        let mut a = Vec::with_capacity(shape.m * shape.k);
        for i in 0..shape.m {
            for p in 0..shape.k {
                a.push(candidate_code(case, CandidateSide::A, i, p));
            }
        }
        let mut b = Vec::with_capacity(shape.k * shape.n);
        for p in 0..shape.k {
            for j in 0..shape.n {
                b.push(candidate_code(case, CandidateSide::B, j, p));
            }
        }
        (a, b)
    }

    fn same_candidate_route(left: KernelSpec<i8, i32>, right: KernelSpec<i8, i32>) -> bool {
        left.backend == right.backend
            && left.factorization == right.factorization
            && left.mr == right.mr
            && left.nr == right.nr
            && left.lane_layout == right.lane_layout
            && left.k_group == right.k_group
            && left.products_per_step == right.products_per_step
            && left.lane_cap == right.lane_cap
            && left.max_bound == right.max_bound
            && core::ptr::fn_addr_eq(left.mac_tile, right.mac_tile)
    }

    fn assert_candidate_bytes<E: CandidateCorpusFloat>(actual: &[E], expected: &[E]) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "candidate output has the complete cell extent"
        );
        for (at, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(
                actual.symbol_bits(),
                expected.symbol_bits(),
                "candidate changed an output byte at {at}"
            );
        }
    }

    const CANDIDATE_SAMPLES: usize = 9;
    const CANDIDATE_T95_DF8: f64 = 2.306_004_135_204_166;

    struct CandidateMeasurement<E> {
        spec: KernelSpec<i8, i32>,
        output: Vec<E>,
        pa: Vec<PackedCode>,
        pb: Vec<PackedCode>,
        batch: usize,
        seconds: [f64; CANDIDATE_SAMPLES],
        elapsed_ns: [u128; CANDIDATE_SAMPLES],
    }

    impl<E: FloatElement + Copy> CandidateMeasurement<E> {
        fn new(spec: KernelSpec<i8, i32>, shape: Shape, pa_codes: usize, pb_codes: usize) -> Self {
            Self {
                spec,
                output: vec![E::ZERO; shape.m * shape.n],
                pa: vec![PackedCode::default(); pa_codes],
                pb: vec![PackedCode::default(); pb_codes],
                batch: 1,
                seconds: [0.0; CANDIDATE_SAMPLES],
                elapsed_ns: [0; CANDIDATE_SAMPLES],
            }
        }
    }

    fn poison_candidate_output<E: CandidateCorpusFloat>(output: &mut [E], expected: &[E]) {
        assert_eq!(
            output.len(),
            expected.len(),
            "candidate poison covers every output cell"
        );
        for (output, &expected) in output.iter_mut().zip(expected) {
            let poisoned = E::from_candidate_bits(expected.symbol_bits() ^ 1);
            assert_ne!(
                poisoned.symbol_bits(),
                expected.symbol_bits(),
                "candidate poison must differ from the expected output code"
            );
            *output = poisoned;
        }
    }

    fn candidate_place<A: PlaceAt>(accumulator: &mut A, lane: i128, exponent: i64) {
        accumulator.place_at(lane, i32::try_from(exponent).unwrap());
    }

    /// One forced-candidate production batch. Poisoning, view/triple creation,
    /// selector closure construction, and byte validation are all outside the
    /// interval; only the engine under comparison repeats inside it.
    fn candidate_timed_batch<E>(
        shape: Shape,
        a: &[E],
        b: &[E],
        expected: &[E],
        measured: &mut CandidateMeasurement<E>,
        repetitions: usize,
    ) -> core::time::Duration
    where
        E: CandidateCorpusFloat + EncodeFrom<AccOf<E>> + Copy,
        AccOf<E>: SignedPlace + PlaceAt + ScaleExact + AbsorbPrior<E>,
    {
        poison_candidate_output(&mut measured.output, expected);
        let elapsed = {
            let av = MatView::row_major(a, shape.m, shape.k).unwrap();
            let bv = MatView::row_major(b, shape.k, shape.n).unwrap();
            let cv = MatViewMut::row_major(&mut measured.output, shape.m, shape.n).unwrap();
            let mut triple = Triple::new(av, bv, cv).unwrap();
            let options = GemmOptions::default();
            let spec = measured.spec;
            let select = |_, _, _, _| spec;
            let mut ledger = ();
            let start = std::time::Instant::now();
            for _ in 0..repetitions {
                gemm_float_tiles_with_selector(
                    &mut triple,
                    &Linear::OVERWRITE,
                    options,
                    &mut measured.pa,
                    &mut measured.pb,
                    candidate_place::<AccOf<E>>,
                    &mut ledger,
                    select,
                );
            }
            start.elapsed()
        };
        assert_candidate_bytes(&measured.output, expected);
        elapsed
    }

    fn candidate_estimate(values: &[f64; CANDIDATE_SAMPLES]) -> (f64, f64) {
        let n = CANDIDATE_SAMPLES as f64;
        let mean = values.iter().sum::<f64>() / n;
        let sum_squares = values
            .iter()
            .map(|value| {
                let residual = value - mean;
                residual * residual
            })
            .sum::<f64>();
        let variance = sum_squares / (n - 1.0);
        (mean, CANDIDATE_T95_DF8 * (variance / n).sqrt())
    }

    fn candidate_release_sweep<E>()
    where
        E: CandidateCorpusFloat + EncodeFrom<AccOf<E>> + Copy,
        AccOf<E>: SignedPlace + PlaceAt + ScaleExact + AbsorbPrior<E>,
    {
        let mut candidates = Vec::new();
        for spec in uor_matmul_kernels::cached::available_i8()
            .chain(uor_matmul_kernels::cached::available_i8_narrow())
            .chain(uor_matmul_kernels::cached::available_reduce_i8())
            .filter(|spec| {
                spec.k_group == 1
                    && matches!(spec.factorization, uor_matmul_kernels::Factorization::Exact)
                    && spec.max_bound >= <i8 as uor_matmul_core::IntegerElement>::FULL
            })
        {
            if !candidates
                .iter()
                .copied()
                .any(|candidate| same_candidate_route(candidate, spec))
            {
                candidates.push(spec);
            }
        }
        for case in CANDIDATE_CASES {
            let shape = case.shape;
            let (a, b) = candidate_operands::<E>(case);
            let expected = candidate_reference(shape, &a, &b);
            let suggested = suggested_float_panels(shape);
            for (offer, pa_codes, pb_codes) in [
                ("stream", 0, 0),
                ("partial", shape.k, shape.k),
                ("suggested", suggested.0, suggested.1),
            ] {
                let mut measurements: Vec<_> = candidates
                    .iter()
                    .copied()
                    .map(|spec| CandidateMeasurement::new(spec, shape, pa_codes, pb_codes))
                    .collect();
                for measured in &mut measurements {
                    candidate_timed_batch(shape, &a, &b, &expected, measured, 1);
                    let pilot = candidate_timed_batch(shape, &a, &b, &expected, measured, 1);
                    measured.batch = core::time::Duration::from_millis(4)
                        .as_nanos()
                        .checked_div(pilot.as_nanos().max(1))
                        .and_then(|value| usize::try_from(value).ok())
                        .unwrap_or(1)
                        .max(1);
                }
                for round in 0..CANDIDATE_SAMPLES {
                    for offset in 0..measurements.len() {
                        let at = (round + offset) % measurements.len();
                        let measured = &mut measurements[at];
                        let elapsed = candidate_timed_batch(
                            shape,
                            &a,
                            &b,
                            &expected,
                            measured,
                            measured.batch,
                        );
                        measured.seconds[round] = elapsed.as_secs_f64() / measured.batch as f64;
                        measured.elapsed_ns[round] = elapsed.as_nanos();
                    }
                }
                let selected =
                    atlas_tile_spec::<AccOf<E>>(Backend::Auto, shape, pa_codes, pb_codes);
                let selected_at = measurements
                    .iter()
                    .position(|measured| same_candidate_route(measured.spec, selected))
                    .expect("the production selector chooses an enumerated candidate");
                let selected_seconds = measurements[selected_at].seconds;
                for (route, measured) in measurements.iter().enumerate() {
                    for (round, elapsed_ns) in measured.elapsed_ns.iter().enumerate() {
                        eprintln!(
                            "CG21_SAMPLE phase=candidate width={} case={} m={} k={} n={} fill={:?} offer={} route={} backend={:?} factorization={:?} mr={} nr={} layout={:?} k_group={} products_per_step={} lane_cap={} max_bound={} round={} batch={} elapsed_ns={}",
                            E::LABEL,
                            case.seed,
                            shape.m,
                            shape.k,
                            shape.n,
                            case.fill,
                            offer,
                            route,
                            measured.spec.backend,
                            measured.spec.factorization,
                            measured.spec.mr,
                            measured.spec.nr,
                            measured.spec.lane_layout,
                            measured.spec.k_group,
                            measured.spec.products_per_step,
                            measured.spec.lane_cap,
                            measured.spec.max_bound,
                            round,
                            measured.batch,
                            elapsed_ns,
                        );
                    }
                    let (mean, half_width) = candidate_estimate(&measured.seconds);
                    let ratios =
                        std::array::from_fn(|at| measured.seconds[at] / selected_seconds[at]);
                    let (ratio, ratio_half_width) = candidate_estimate(&ratios);
                    eprintln!(
                        "candidate {} route {} {:?} {}x{} {} {:?}/{:?}: {:.1} +/- {:.1} ns (95% CI), paired/selected {:.4} +/- {:.4}, raw ratios {:?} ({} interleaved calibrated batches, batch {}); work {:?}",
                        E::LABEL,
                        route,
                        measured.spec.backend,
                        measured.spec.mr,
                        measured.spec.nr,
                        offer,
                        shape,
                        case.fill,
                        mean * 1e9,
                        half_width * 1e9,
                        ratio,
                        ratio_half_width,
                        ratios,
                        CANDIDATE_SAMPLES,
                        measured.batch,
                        atlas_executed_work::<AccOf<E>>(
                            measured.spec,
                            shape,
                            pa_codes,
                            pb_codes,
                        )
                    );
                }
            }
        }
    }

    /// `CG-21`: release-only forced-candidate measurement over the exact
    /// structural corpus. Complete output bytes are checked after every warmup,
    /// calibration, and timed batch, outside the measured interval. Calibrated
    /// samples are interleaved across candidates and reported as width-labelled
    /// raw batch durations, Student 95% intervals, and paired raw ratios to the
    /// public production selector, without exposing a selector override in that
    /// API.
    #[test]
    #[ignore = "release-mode candidate measurement for `just uor-float-sweep`"]
    #[allow(clippy::assertions_on_constants)]
    fn every_atlas_candidate_is_measurable_with_byte_checks_cg_21() {
        assert!(
            !cfg!(debug_assertions),
            "the candidate sweep is release-only"
        );
        candidate_release_sweep::<f32>();
        candidate_release_sweep::<f64>();
    }

    fn expected_lookup_specs(
        backend: Backend,
        shape: Shape,
    ) -> (KernelSpec<i8, i32>, KernelSpec<i8, i32>) {
        let tile = uor_matmul_kernels::choose_for_rows(
            uor_matmul_kernels::available_i8().filter(|spec| spec.k_group == 1),
            backend,
            <i8 as uor_matmul_core::IntegerElement>::FULL,
            shape.m,
        )
        .unwrap();
        let reduce = uor_matmul_kernels::choose_for_rows(
            uor_matmul_kernels::available_reduce_i8().filter(|spec| spec.k_group == 1),
            backend,
            <i8 as uor_matmul_core::IntegerElement>::FULL,
            shape.m,
        )
        .unwrap();
        (tile, reduce)
    }

    fn assert_global_selector_minimum<A>(
        backend: Backend,
        shape: Shape,
        eligible: &[KernelSpec<i8, i32>],
        pa_codes: usize,
        pb_codes: usize,
    ) -> KernelSpec<i8, i32> {
        let minimum = eligible
            .iter()
            .map(|&spec| atlas_executed_work::<A>(spec, shape, pa_codes, pb_codes))
            .min()
            .unwrap();
        let actual = atlas_tile_spec::<A>(backend, shape, pa_codes, pb_codes);
        assert_eq!(
            atlas_executed_work::<A>(actual, shape, pa_codes, pb_codes),
            minimum,
            "selector hid a lower executed-work candidate at {backend:?} {shape:?}"
        );
        assert!(eligible
            .iter()
            .any(|&spec| AtlasRoute::from(spec) == AtlasRoute::from(actual)));
        actual
    }

    fn assert_model_work<A>(
        spec: KernelSpec<i8, i32>,
        shape: Shape,
        pa_codes: usize,
        pb_codes: usize,
    ) {
        let bytes = core::mem::size_of::<A>();
        assert_eq!(
            atlas_executed_work::<A>(spec, shape, pa_codes, pb_codes).coordinates(),
            uor_matmul_model::derive::atlas_executed_work(
                shape.m,
                shape.k,
                shape.n,
                spec.mr,
                spec.nr,
                spec.products_per_step,
                bytes,
                ATLAS_TILE_WORK_BYTES,
                pa_codes,
                pb_codes,
                MAX_TILE_LANES,
            ),
            "shipped and model executed-work derivations diverged for {spec:?}"
        );
    }

    fn assert_uniform_count<E>(
        shape: Shape,
        spec: KernelSpec<i8, i32>,
        expected: [usize; 5],
        kind: &str,
        unit: E,
        expected_value: E,
    ) where
        E: FloatElement + EncodeFrom<AccOf<E>> + Copy,
        AccOf<E>: SignedPlace + PlaceAt + ScaleExact + AbsorbPrior<E>,
    {
        let (output, census) = counted_uniform::<E>(shape, Backend::Portable, unit);
        assert_eq!(
            census.route,
            Some(spec.into()),
            "{kind} selected a route other than the model derivation"
        );
        assert_eq!(census.coordinate_products, expected[0], "{kind} lookups");
        assert_eq!(census.coordinate_additions, expected[0], "{kind} adds");
        assert_eq!(census.kernel_calls, expected[1], "{kind} kernel calls");
        assert_eq!(census.projections, expected[2], "{kind} projections");
        assert_eq!(census.placements, expected[3], "{kind} placements");
        assert_eq!(census.encodes, expected[4], "{kind} encodes");
        assert_eq!(census.boundary_joins, 0, "{kind} finite-unit boundary");
        assert!(census.panels > 0, "{kind} must execute bounded panels");
        let expected_value = expected_value.pack();
        assert!(
            output.iter().all(|&value| value.pack() == expected_value),
            "{kind} uniform product has the wrong exact value"
        );
    }

    /// `CG-22`: selection is the model's operation derivation and the counters
    /// observe the same body that production instantiates with `()`.
    #[test]
    fn atlas_route_and_operation_census_follow_the_model_cg_22() {
        let narrow: Vec<_> = uor_matmul_kernels::available_i8_narrow()
            .filter(|spec| spec.k_group == 1)
            .map(AtlasRoute::from)
            .collect();
        let mut saw_auto_tile = false;
        let mut saw_auto_reduce = false;
        let mut saw_auto_narrow = false;
        for backend in [Backend::Portable, Backend::Auto] {
            for m in 1..=9 {
                for n in 1..=19 {
                    let shape = Shape { m, k: 7, n };
                    let all: Vec<_> = uor_matmul_kernels::available_i8()
                        .chain(uor_matmul_kernels::available_i8_narrow())
                        .chain(uor_matmul_kernels::available_reduce_i8())
                        .filter(|spec| {
                            spec.k_group == 1
                                && matches!(
                                    spec.factorization,
                                    uor_matmul_kernels::Factorization::Exact
                                )
                                && spec.max_bound >= <i8 as uor_matmul_core::IntegerElement>::FULL
                        })
                        .collect();
                    let named_available =
                        backend != Backend::Auto && all.iter().any(|spec| spec.backend == backend);
                    let eligible: Vec<_> = all
                        .into_iter()
                        .filter(|spec| {
                            backend == Backend::Auto
                                || (named_available && spec.backend == backend)
                                || (!named_available && spec.backend == Backend::Portable)
                        })
                        .collect();
                    assert!(!eligible.is_empty());
                    for spec in eligible.iter().copied() {
                        assert_model_work::<AccOf<f32>>(spec, shape, 0, 0);
                        assert_model_work::<AccOf<f64>>(spec, shape, 0, 0);
                    }
                    let actual_f32 = assert_global_selector_minimum::<AccOf<f32>>(
                        backend, shape, &eligible, 0, 0,
                    );
                    let actual_f64 = assert_global_selector_minimum::<AccOf<f64>>(
                        backend, shape, &eligible, 0, 0,
                    );
                    if backend == Backend::Auto {
                        saw_auto_reduce |= actual_f32.nr == 1 || actual_f64.nr == 1;
                        saw_auto_tile |= actual_f32.nr > 1 || actual_f64.nr > 1;
                        saw_auto_narrow |= narrow.contains(&AtlasRoute::from(actual_f32))
                            || narrow.contains(&AtlasRoute::from(actual_f64));
                    }
                }
            }
        }

        let repeated = [1.0f32];
        let huge_a = MatView::new(
            &repeated,
            usize::MAX,
            1,
            uor_matmul_core::Strides { rs: 0, cs: 0 },
        )
        .unwrap();
        let huge_b = MatView::new(
            &repeated,
            1,
            usize::MAX,
            uor_matmul_core::Strides { rs: 0, cs: 0 },
        )
        .unwrap();
        let huge = Shape {
            m: huge_a.rows(),
            k: 1,
            n: huge_b.cols(),
        };
        let huge_eligible: Vec<_> = uor_matmul_kernels::available_i8()
            .chain(uor_matmul_kernels::available_i8_narrow())
            .chain(uor_matmul_kernels::available_reduce_i8())
            .filter(|spec| {
                spec.k_group == 1
                    && matches!(spec.factorization, uor_matmul_kernels::Factorization::Exact)
                    && spec.max_bound >= <i8 as uor_matmul_core::IntegerElement>::FULL
            })
            .collect();
        let huge_work: Vec<_> = huge_eligible
            .iter()
            .map(|&spec| atlas_executed_work::<AccOf<f64>>(spec, huge, 0, 0))
            .collect();
        assert!(huge_work.iter().any(|work| {
            work.projections
                .coordinates()
                .into_iter()
                .take(ATLAS_COUNT_WORDS - 1)
                .any(|word| word != 0)
        }));
        assert!(huge_work.windows(2).any(|pair| pair[0] != pair[1]));
        for spec in huge_eligible.iter().copied() {
            assert_model_work::<AccOf<f32>>(spec, huge, 0, 0);
            assert_model_work::<AccOf<f64>>(spec, huge, 0, 0);
        }
        assert_global_selector_minimum::<AccOf<f32>>(Backend::Auto, huge, &huge_eligible, 0, 0);
        assert_global_selector_minimum::<AccOf<f64>>(Backend::Auto, huge, &huge_eligible, 0, 0);
        assert!(
            saw_auto_reduce,
            "the boundary sweep must cover reduction tiles"
        );
        let has_native_tile = uor_matmul_kernels::available_i8()
            .chain(uor_matmul_kernels::available_i8_narrow())
            .any(|spec| spec.k_group == 1 && spec.backend != Backend::Portable);
        assert_eq!(
            saw_auto_tile, has_native_tile,
            "a native lookup tile must participate in the global minimum"
        );
        assert_eq!(
            saw_auto_narrow,
            !narrow.is_empty(),
            "a declared narrow lookup tile must participate in the global minimum"
        );

        if !narrow.is_empty() {
            let shape = Shape { m: 6, k: 7, n: 16 };
            let full = uor_matmul_kernels::available_i8()
                .filter(|spec| spec.k_group == 1 && spec.backend == Backend::Avx2)
                .max_by_key(|spec| spec.mr * spec.nr)
                .unwrap();
            let selected = atlas_tile_spec::<AccOf<f64>>(Backend::Auto, shape, 0, 0);
            assert!(
                atlas_executed_work::<AccOf<f64>>(full, shape, 0, 0)
                    >= atlas_executed_work::<AccOf<f64>>(selected, shape, 0, 0),
                "the selector must price every projection, decode, issue, and live byte"
            );
            assert!(
                selected.mr * selected.nr <= MAX_TILE_LANES,
                "the selected one-pass frame is derived from family geometry"
            );
        }

        for shape in [Shape { m: 5, k: 7, n: 1 }, Shape { m: 5, k: 7, n: 4 }] {
            let spec = atlas_tile_spec::<AccOf<f32>>(Backend::Portable, shape, 0, 0);
            let expected = uor_matmul_model::derive::atlas_uniform_census(
                shape.m, shape.k, shape.n, spec.mr, spec.nr, 0,
            );
            assert_uniform_count::<f32>(shape, spec, expected, "f32", 1.0, shape.k as f32);
            assert_uniform_count::<f64>(shape, spec, expected, "f64", 1.0, shape.k as f64);
        }

        // Partial caller offers create real block boundaries. The selector's
        // vector prices the exact decode/projection reuse those boundaries
        // execute, and the production ledger witnesses the same calls.
        let offered_shape = Shape { m: 7, k: 5, n: 11 };
        for (pa_codes, pb_codes) in [
            (0, 0),
            (2 * offered_shape.k, 3 * offered_shape.k),
            (
                offered_shape.m * offered_shape.k,
                offered_shape.n * offered_shape.k,
            ),
        ] {
            let (_, census) = counted_uniform_with_panels::<f64>(
                offered_shape,
                Backend::Auto,
                1.0,
                pa_codes,
                pb_codes,
            );
            let spec =
                atlas_tile_spec::<AccOf<f64>>(Backend::Auto, offered_shape, pa_codes, pb_codes);
            assert_eq!(census.route, Some(spec.into()));
            let work = atlas_executed_work::<AccOf<f64>>(spec, offered_shape, pa_codes, pb_codes);
            let exact_usize = |count: AtlasCount| {
                let words = count.coordinates();
                assert!(words[..ATLAS_COUNT_WORDS - 1].iter().all(|&word| word == 0));
                usize::try_from(words[ATLAS_COUNT_WORDS - 1]).unwrap()
            };
            assert_eq!(census.projections, exact_usize(work.projections));
            assert_eq!(
                census.decoded_a + census.decoded_b,
                exact_usize(work.decodes)
            );
            assert_eq!(census.issued_steps, exact_usize(work.issued));
            assert_eq!(
                census.product_initializations,
                exact_usize(work.product_initializations),
                "only live product carriers are initialized"
            );
            assert_model_work::<AccOf<f64>>(spec, offered_shape, pa_codes, pb_codes);
        }

        for shape in [Shape { m: 7, k: 5, n: 33 }, Shape { m: 17, k: 3, n: 5 }] {
            let (pa_codes, pb_codes) = suggested_float_panels(shape);
            let (_, census) =
                counted_uniform_with_panels::<f64>(shape, Backend::Auto, 1.0, pa_codes, pb_codes);
            let source_lower_bound = (shape.m + shape.n) * shape.k;
            assert_eq!(
                census.projections, source_lower_bound,
                "the established full offer must project every source exactly once"
            );
            assert_eq!(
                census.decoded_a + census.decoded_b,
                source_lower_bound,
                "the established full offer must decode every source exactly once"
            );
        }

        let empty_depth = Shape { m: 3, k: 0, n: 5 };
        let (output, census) =
            counted_uniform_with_panels::<f64>(empty_depth, Backend::Auto, 1.0, 7, 11);
        assert_eq!(census.route, None);
        assert_eq!(census.panels, 0);
        assert_eq!(census.decoded_a + census.decoded_b, 0);
        assert_eq!(census.projections, 0);
        assert_eq!(
            census.kernel_calls + census.placements + census.boundary_joins,
            0
        );
        assert_eq!(census.encodes, empty_depth.m * empty_depth.n);
        assert!(output.iter().all(|&value| value == 0.0));
        let spec = uor_matmul_kernels::available_reduce_i8()
            .find(|spec| spec.k_group == 1)
            .expect("the portable Atlas declaration is total");
        assert_eq!(
            atlas_executed_work::<AccOf<f64>>(spec, empty_depth, 7, 11),
            AtlasWork::ZERO
        );
        assert_model_work::<AccOf<f64>>(spec, empty_depth, 7, 11);

        let shape = Shape { m: 1, k: 17, n: 1 };
        let a: Vec<f64> = (0..shape.k)
            .map(|p| f64::from_bits(((p as u64 * 113 + 1) << 52) | 1))
            .collect();
        let b: Vec<f64> = (0..shape.k)
            .map(|p| f64::from_bits((((2046 - p as u64 * 109).max(1)) << 52) | 3))
            .collect();
        let mut output = vec![0.0f64; 1];
        let av = MatView::row_major(&a, shape.m, shape.k).unwrap();
        let bv = MatView::row_major(&b, shape.k, shape.n).unwrap();
        let cv = MatViewMut::row_major(&mut output, shape.m, shape.n).unwrap();
        let mut triple = Triple::new(av, bv, cv).unwrap();
        let mut census = AtlasCensus::default();
        gemm_float_tiles(
            &mut triple,
            &Linear::OVERWRITE,
            GemmOptions {
                backend: Backend::Portable,
                ..GemmOptions::default()
            },
            &mut [],
            &mut [],
            |acc, lane, exponent| acc.place_at(lane, i32::try_from(exponent).unwrap()),
            &mut census,
        );
        assert!(census.panels > 0, "wide grades must refine recursively");
        assert_eq!(
            census.placements, shape.k,
            "each finite source product resolves before one terminal placement"
        );
        assert!(
            census.kernel_calls > census.placements,
            "the corpus must exercise multiple lookup diagonals per product"
        );

        let mut positive = [0i8; MAX_ATLAS_WORDS];
        let mut negative = [0i8; MAX_ATLAS_WORDS];
        atlas_word(
            atlas_atom(
                PackedCode {
                    mantissa: 32_769,
                    exp: 0,
                    _pad: 0,
                },
                false,
            )
            .unwrap(),
            &mut positive,
        );
        atlas_word(
            atlas_atom(
                PackedCode {
                    mantissa: -32_769,
                    exp: 0,
                    _pad: 0,
                },
                false,
            )
            .unwrap(),
            &mut negative,
        );
        assert_eq!(&positive[..3], &[1, -128, 1]);
        assert_eq!(&negative[..2], &[-1, -128]);

        for spec in uor_matmul_kernels::available_i8()
            .chain(uor_matmul_kernels::available_reduce_i8())
            .filter(|spec| spec.k_group == 1)
        {
            assert_eq!(
                spec.mr * spec.nr % spec.products_per_step,
                0,
                "lookup sequence issue density must divide its physical tile"
            );
            let depth = 3usize;
            let pa = vec![3i8; spec.mr * depth];
            let pb = vec![-5i8; spec.nr * depth];
            let mut clean = vec![0i32; spec.mr * spec.nr];
            let mut poisoned = vec![i32::MIN; spec.mr * spec.nr];
            spec.mac_tile(depth, &pa, &pb, &mut clean);
            spec.mac_tile(depth, &pa, &pb, &mut poisoned);
            assert_eq!(
                poisoned, clean,
                "lookup kernel {:?} must overwrite every output lane",
                spec.backend
            );
        }
    }

    /// `CD-30`: compatibility queries keep their established values while the
    /// offered buffers remain optional in-place projection caches for the same
    /// operation.
    #[test]
    fn suggested_float_panels_retain_the_public_contract_cd_30() {
        for shape in [
            Shape { m: 0, k: 7, n: 3 },
            Shape { m: 3, k: 7, n: 0 },
            Shape { m: 1, k: 1, n: 1 },
            Shape { m: 3, k: 5, n: 7 },
            Shape {
                m: 16,
                k: 128,
                n: 16,
            },
            Shape {
                m: 32,
                k: 32,
                n: 32,
            },
        ] {
            let scaled = shape.k.saturating_mul(shape.m.saturating_add(shape.n));
            let scalar = shape.k.saturating_mul(shape.n);
            let bridged = scaled
                .saturating_add(crate::suggested_scratch(shape))
                .div_ceil(WORDS_PER_CODE);
            assert_eq!(suggested_bridge_scaled(shape), scaled);
            assert_eq!(
                suggested_float_panels(shape),
                (shape.k, scalar.max(bridged))
            );

            let suggested = suggested_float_panels(shape);
            let (output, census) = counted_uniform_with_panels::<f64>(
                shape,
                Backend::Auto,
                1.0,
                suggested.0,
                suggested.1,
            );
            assert_eq!(census.encodes, shape.m.saturating_mul(shape.n));
            assert!(output
                .iter()
                .all(|&value| value.to_bits() == (shape.k as f64).to_bits()));
        }
    }

    /// `CG-22`: projection reports exactly the precision reached by the same
    /// centered-octet refinement that execution consumes.
    #[test]
    fn atlas_projection_extent_is_precision_directed_cg_22() {
        for (mantissa, expected, digits) in [
            (1, 1, &[1i8][..]),
            (257, 2, &[1, 1][..]),
            (32_769, 3, &[1, -128, 1][..]),
            (-32_769, 2, &[-1, -128][..]),
        ] {
            let atom = atlas_atom(
                PackedCode {
                    mantissa,
                    exp: 0,
                    _pad: 0,
                },
                false,
            )
            .unwrap();
            let mut coordinates = [0i8; MAX_ATLAS_WORDS];
            let extent = atlas_word(atom, &mut coordinates);
            assert_eq!(extent, expected);
            assert_eq!(&coordinates[..extent], digits);
        }

        let mut reused = [i8::MAX; MAX_ATLAS_WORDS];
        let extent = replace_atlas_word(
            AtlasAtom {
                unit: 1,
                grade: 0,
                negative: false,
            },
            &mut reused,
        );
        assert_eq!(extent, 1);
        assert_eq!(reused, [1, 0, 0, 0, 0, 0, 0, 0, 0]);

        assert_eq!(
            MAX_ATLAS_WORDS,
            (u64::BITS.div_ceil(ATLAS_DIGIT_BITS) + 1) as usize
        );
        for negative in [false, true] {
            let mut coordinates = [0i8; MAX_ATLAS_WORDS];
            let extent = atlas_word(
                AtlasAtom {
                    unit: u64::MAX,
                    grade: 0,
                    negative,
                },
                &mut coordinates,
            );
            assert_eq!(extent, MAX_ATLAS_WORDS);
        }
    }

    /// `CU-11`: the quotient recurrence is byte-identical to an independent
    /// test-only binary oracle across complete low words and every source-width
    /// boundary.
    #[test]
    fn atlas_valuation_matches_the_independent_binary_oracle_cu_11() {
        for magnitude in 0..=u64::from(u16::MAX) {
            let expected = if magnitude == 0 {
                (0, 0)
            } else {
                let valuation = magnitude.trailing_zeros();
                (magnitude >> valuation, valuation)
            };
            assert_eq!(atlas_odd_section(magnitude), expected, "{magnitude}");
        }
        for valuation in 0..u64::BITS {
            for unit in [1u64, 3, 127, 255, u64::MAX >> valuation] {
                let magnitude = unit << valuation;
                if magnitude == 0 {
                    continue;
                }
                let expected_valuation = magnitude.trailing_zeros();
                assert_eq!(
                    atlas_odd_section(magnitude),
                    (magnitude >> expected_valuation, expected_valuation),
                );
            }
        }
    }

    /// `CD-30`: relabelling an offered cache word preserves the complete Atlas
    /// projection and the IEEE boundary action, including the escaped high half
    /// of the public finite-code domain.
    #[test]
    fn in_place_atlas_cache_preserves_projection_and_boundary_cd_30() {
        let finite_codes = [
            ZERO_CODE,
            PackedCode::of(Decoded::Finite {
                sign: true,
                mantissa: 0,
                exp: -17,
            }),
            PackedCode::of(Decoded::Finite {
                sign: false,
                mantissa: 32_769,
                exp: -101,
            }),
            PackedCode::of(Decoded::Finite {
                sign: true,
                mantissa: u64::MAX,
                exp: i32::MAX,
            }),
        ];
        for code in finite_codes {
            let (projected, occupied) = AtlasProjectedCode::project(code);
            let relabelled = AtlasProjectedCode::from_packed(projected.into_packed());
            assert_eq!(relabelled, projected);
            let atom = atlas_atom(code, false);
            assert_eq!(occupied, atom.is_some());
            match atom {
                None => assert_eq!(projected.extent, 0),
                Some(atom) => {
                    let mut expected = [0; MAX_ATLAS_WORDS];
                    let extent = atlas_word(atom, &mut expected);
                    assert_eq!(usize::from(projected.extent), extent);
                    assert_eq!(projected.coordinates, expected);
                    let cached_atom = projected.atom().unwrap();
                    assert_eq!(cached_atom.grade, atom.grade);
                    assert_eq!(cached_atom.negative, atom.negative);
                }
            }
        }

        let boundary_codes = [
            PackedCode::of((-0.0f64).decode()),
            PackedCode::of((-3.0f64).decode()),
            PackedCode::of(5.0f64.decode()),
            PackedCode::of(f64::INFINITY.decode()),
            PackedCode::of(f64::NEG_INFINITY.decode()),
            PackedCode::of(f64::NAN.decode()),
        ];
        for left in boundary_codes {
            for right in boundary_codes {
                if left.is_finite() && right.is_finite() {
                    continue;
                }
                let mut direct = <AccOf<f64> as Accumulator>::ZERO;
                direct.accumulate_one(left, right);
                let mut cached = <AccOf<f64> as Accumulator>::ZERO;
                cached.accumulate_one(
                    AtlasProjectedCode::project(left).0.boundary_code(),
                    AtlasProjectedCode::project(right).0.boundary_code(),
                );
                assert_eq!(cached, direct, "{left:?} x {right:?}");
                let mut quotient = <AccOf<f64> as Accumulator>::ZERO;
                quotient.accumulate_one(
                    atlas_boundary_code(atlas_source_state(left, false).0),
                    atlas_boundary_code(atlas_source_state(right, false).0),
                );
                assert_eq!(quotient, direct, "boundary quotient {left:?} x {right:?}");
            }
        }
    }

    /// `CK-20`: evaluating the optimized centered-octet word commutes with the
    /// quotient evaluation of the packed coefficient. The oracle is an
    /// independent high-to-low shift/add fold; it does not call the production
    /// quotient step or duplicate its remainder rule.
    #[test]
    fn centered_octet_projection_commutes_with_laurent_evaluation_ck_20() {
        fn evaluate(coordinates: &[i8]) -> i128 {
            coordinates.iter().rev().fold(0i128, |value, &digit| {
                (value << ATLAS_DIGIT_BITS) + i128::from(digit)
            })
        }

        for coefficient in i16::MIN..=i16::MAX {
            if coefficient == 0 {
                continue;
            }
            let atom = atlas_atom(
                PackedCode {
                    mantissa: i64::from(coefficient),
                    exp: 0,
                    _pad: 0,
                },
                false,
            )
            .unwrap();
            let mut coordinates = [i8::MAX; MAX_ATLAS_WORDS];
            let extent = atlas_word(atom, &mut coordinates);
            let reconstructed = evaluate(&coordinates[..extent]) << atom.grade;
            assert_eq!(reconstructed, i128::from(coefficient), "{coefficient}");
        }

        for magnitude in [
            1,
            127,
            128,
            129,
            255,
            256,
            257,
            32_767,
            32_768,
            32_769,
            (1u64 << 63) - 1,
            1u64 << 63,
            u64::MAX,
        ] {
            for negative in [false, true] {
                let atom = atlas_atom(
                    PackedCode::of(Decoded::Finite {
                        sign: negative,
                        mantissa: magnitude,
                        exp: 0,
                    }),
                    false,
                )
                .unwrap();
                let mut coordinates = [i8::MIN; MAX_ATLAS_WORDS];
                let extent = atlas_word(atom, &mut coordinates);
                let reconstructed = evaluate(&coordinates[..extent]) << atom.grade;
                let expected = if negative {
                    -i128::from(magnitude)
                } else {
                    i128::from(magnitude)
                };
                assert_eq!(reconstructed, expected, "{negative}/{magnitude}");
                assert!(
                    coordinates[extent - 1] != 0,
                    "the reported extent must retain its signed carry word"
                );
            }
        }
    }

    /// `CK-20`: the bounded product carrier evaluates the signed diagonal
    /// contraction exactly, including the carry word above bit 127. The test
    /// oracle uses whole-value multiplication only outside shipped code.
    #[test]
    fn atlas_product_carrier_commutes_with_coordinate_contraction_ck_20() {
        fn contract(left: AtlasAtom, right: AtlasAtom) -> (bool, u128) {
            let mut a = [0i8; MAX_ATLAS_WORDS];
            let mut b = [0i8; MAX_ATLAS_WORDS];
            let a_extent = atlas_word(left, &mut a);
            let b_extent = atlas_word(right, &mut b);
            let mut product = AtlasProduct::ZERO;
            for diagonal in 0..a_extent + b_extent - 1 {
                let first_a = diagonal.saturating_sub(b_extent - 1);
                let last_a = diagonal.min(a_extent - 1);
                let mut lane = 0i32;
                for ca in first_a..=last_a {
                    lane += i32::from(a[ca]) * i32::from(b[diagonal - ca]);
                }
                product.add_diagonal(lane, diagonal);
            }
            product.signed_magnitude()
        }

        for left in 1u64..=u64::from(u8::MAX) {
            for right in 1u64..=u64::from(u8::MAX) {
                for negative in [false, true] {
                    assert_eq!(
                        contract(
                            AtlasAtom {
                                unit: left,
                                grade: 0,
                                negative,
                            },
                            AtlasAtom {
                                unit: right,
                                grade: 0,
                                negative: false,
                            },
                        ),
                        (negative, u128::from(left) * u128::from(right))
                    );
                }
            }
        }

        for left in [1, 127, 128, 255, 256, 32_769, 1u64 << 63, u64::MAX] {
            for right in [1, 129, 257, 32_767, (1u64 << 63) - 1, u64::MAX] {
                for left_negative in [false, true] {
                    for right_negative in [false, true] {
                        assert_eq!(
                            contract(
                                AtlasAtom {
                                    unit: left,
                                    grade: 0,
                                    negative: left_negative,
                                },
                                AtlasAtom {
                                    unit: right,
                                    grade: 0,
                                    negative: right_negative,
                                },
                            ),
                            (
                                left_negative != right_negative,
                                u128::from(left) * u128::from(right),
                            ),
                            "{left_negative}/{left} x {right_negative}/{right}"
                        );
                    }
                }
            }
        }
    }

    /// `CG-22`: one exact frame owns every live tile output, with no depth-sized
    /// state and no representation ceiling beside the kernel family's geometry.
    #[test]
    fn atlas_live_frame_is_family_bounded_and_one_pass_cg_22() {
        let shape = Shape {
            m: MAX_TILE_LANES,
            k: 3,
            n: MAX_TILE_LANES,
        };
        let specs: Vec<_> = uor_matmul_kernels::available_i8()
            .chain(uor_matmul_kernels::available_i8_narrow())
            .chain(uor_matmul_kernels::available_reduce_i8())
            .filter(|spec| spec.k_group == 1)
            .collect();
        assert!(
            specs
                .iter()
                .all(|spec| spec.mr + spec.nr <= MAX_ATLAS_SOURCE_SITES),
            "every host-resolved source panel fits the exact declared-family geometry"
        );
        for spec in specs {
            let live_cells = spec.mr.min(shape.m) * spec.nr.min(shape.n);
            assert!(live_cells <= MAX_TILE_LANES);
            for bytes in [
                core::mem::size_of::<AccOf<f32>>(),
                core::mem::size_of::<AccOf<f64>>(),
            ] {
                let expected = ATLAS_TILE_WORK_BYTES as u128 + live_cells as u128 * bytes as u128;
                let modeled = uor_matmul_model::derive::atlas_executed_work(
                    shape.m,
                    shape.k,
                    shape.n,
                    spec.mr,
                    spec.nr,
                    spec.products_per_step,
                    bytes,
                    ATLAS_TILE_WORK_BYTES,
                    0,
                    0,
                    MAX_TILE_LANES,
                );
                assert_eq!(modeled[4], AtlasCount::from_u128(expected).coordinates());
            }
        }
        assert_eq!(
            core::mem::size_of::<AtlasProjectedCode>(),
            core::mem::size_of::<PackedCode>()
        );
    }

    /// `CG-22`: when one physical lane refines fewer octets than its previous
    /// source, coordinates beyond the new extent are the Atlas zero even while
    /// another lane keeps the panel's wider diagonal active.
    #[test]
    fn retired_coordinate_tail_is_the_atlas_zero_cg_22() {
        let shape = Shape { m: 2, k: 2, n: 2 };
        let (tile, reduce) = expected_lookup_specs(Backend::Portable, shape);
        for spec in [tile, reduce] {
            let rows = shape.m.min(spec.mr);
            let cols = shape.n.min(spec.nr);
            if rows < 2 && cols < 2 {
                continue;
            }
            let mut a_codes = vec![ZERO_CODE; spec.mr * shape.k];
            let mut b_codes = vec![ZERO_CODE; spec.nr * shape.k];
            let wide = PackedCode {
                mantissa: 32_769,
                exp: 0,
                _pad: 0,
            };
            for p in 0..shape.k {
                for row in 0..rows {
                    let code = if rows >= 2 && row == p {
                        wide
                    } else {
                        UNIT_CODE
                    };
                    a_codes[atlas_panel_slot(spec.lane_layout, p, row, spec.mr, shape.k)] = code;
                }
                for col in 0..cols {
                    let code = if rows < 2 && col == p {
                        wide
                    } else {
                        UNIT_CODE
                    };
                    b_codes[atlas_panel_slot(spec.lane_layout, p, col, spec.nr, shape.k)] = code;
                }
            }

            let mut accumulators = [<AccOf<f64> as Accumulator>::ZERO; MAX_TILE_LANES];
            let mut census = AtlasCensus::default();
            census.selected(spec, shape);
            accumulate_atlas_tile(
                &mut accumulators[..spec.mr * spec.nr],
                &a_codes,
                &b_codes,
                rows,
                cols,
                shape.k,
                spec,
                |acc, lane, exponent| acc.place_at(lane, i32::try_from(exponent).unwrap()),
                &mut census,
            );
            for row in 0..rows {
                for col in 0..cols {
                    assert_eq!(
                        f64::encode_from(
                            accumulators[row * spec.nr + col],
                            uor_matmul_core::EncodeMode::Nearest,
                        ),
                        32_770.0,
                        "retired tail in {:?}",
                        spec.lane_layout
                    );
                }
            }
        }
    }

    /// `CG-22`: the Atlas absorbing element neither projects a coordinate nor
    /// calls a lookup kernel.
    #[test]
    fn atlas_absorbing_zero_has_no_kernel_work_cg_22() {
        let shape = Shape { m: 1, k: 3, n: 1 };
        let spec = atlas_tile_spec::<AccOf<f64>>(Backend::Portable, shape, 0, 0);
        let a_codes = vec![ZERO_CODE; spec.mr * shape.k];
        let mut b_codes = vec![ZERO_CODE; spec.nr * shape.k];
        for p in 0..shape.k {
            b_codes[atlas_panel_slot(spec.lane_layout, p, 0, spec.nr, shape.k)] = PackedCode {
                mantissa: 1,
                exp: if p == 0 { i32::MIN } else { i32::MAX },
                _pad: 1,
            };
        }
        let mut accumulators = [<AccOf<f64> as Accumulator>::ZERO; MAX_TILE_LANES];
        let mut census = AtlasCensus::default();
        census.selected(spec, shape);
        accumulate_atlas_tile(
            &mut accumulators[..spec.mr * spec.nr],
            &a_codes,
            &b_codes,
            1,
            1,
            shape.k,
            spec,
            |acc, lane, exponent| acc.place_at(lane, i32::try_from(exponent).unwrap()),
            &mut census,
        );
        assert_eq!(census.panels, 1);
        assert_eq!(census.projections, 0);
        assert_eq!(census.kernel_calls, 0);
        assert_eq!(census.coordinate_products, 0);
        assert_eq!(census.placements, 0);
        assert_eq!(census.boundary_joins, 0);
    }

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

    /// The same product with panels offered, which is the only way the prescaling
    /// path is reached: [`gemm_float`] offers none and therefore always streams.
    fn product_packed(m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        let mut pa = vec![PackedCode::default(); k];
        let mut pb = vec![PackedCode::default(); k * n];
        {
            let av = MatView::row_major(a, m, k).unwrap();
            let bv = MatView::row_major(b, k, n).unwrap();
            let cv = MatViewMut::row_major(&mut c, m, n).unwrap();
            let mut t = Triple::new(av, bv, cv).unwrap();
            gemm_float_packed(
                &mut t,
                &Linear::OVERWRITE,
                GemmOptions::default(),
                &mut pa,
                &mut pb,
            );
        }
        c
    }

    /// The same product through the compatibility spelling. Every offer is
    /// named separately so the sweep can starve each inert buffer and vary the
    /// caller-owned in-place projection caches independently.
    #[allow(clippy::too_many_arguments)]
    fn product_bridged(
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
        scaled: usize,
        panels: usize,
        kernel: usize,
        accs: usize,
    ) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        let mut pa = vec![PackedCode::default(); k.max(1)];
        let mut pb = vec![PackedCode::default(); panels];
        let mut scaled_buf = vec![0i32; scaled];
        let mut kernel_buf = vec![uor_matmul_core::Alphabet::of(0i32); kernel];
        let mut acc_buf = vec![0i128; accs];
        {
            let av = MatView::row_major(a, m, k).unwrap();
            let bv = MatView::row_major(b, k, n).unwrap();
            let cv = MatViewMut::row_major(&mut c, m, n).unwrap();
            let mut t = Triple::new(av, bv, cv).unwrap();
            gemm_float_bridged(
                &mut t,
                &Linear::OVERWRITE,
                GemmOptions::default(),
                &mut pa,
                &mut pb,
                &mut scaled_buf,
                &mut crate::scratch::Scratch::with_accumulators(&mut kernel_buf, &mut acc_buf),
            );
        }
        c
    }

    /// `CD-19`: every historical workspace spelling gives the Atlas bytes.
    ///
    /// `scaled` and integer scratch remain in the explicit signature for API
    /// compatibility, but the implementation neither reads nor writes them.
    /// Code panels only change decode reuse; every offer reaches the same
    /// octet/gauge reduction and the streaming spelling is its byte oracle.
    ///
    /// The significands are drawn from `[2^23, 2^24)`, so the decoded exponent
    /// is exactly the one the generator names: a 24-bit significand is the
    /// element type's own, and the span is the generator's and nothing else.
    #[test]
    fn every_float_workspace_spelling_is_the_atlas_reduction_cd_19() {
        for (label, span_a, span_b) in [
            ("one exponent", 0i32, 0i32),
            ("a few binades", 3, 4),
            ("asymmetric, at the alphabet's edge", 7, 0),
            ("too wide for the i32 alphabet", 8, 8),
            ("past every scaled lane", 90, 90),
        ] {
            for &(m, k, n) in &[
                (1usize, 1usize, 1usize),
                (2, 3, 5),
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
                            let m = 8_388_608 + v.unsigned_abs() % 8_388_607;
                            let x = m as f32 * 2.0f32.powi(s - span / 2);
                            if v & 1 == 0 {
                                -x
                            } else {
                                x
                            }
                        })
                        .collect()
                };
                let av = gen(&mut next, m * k, span_a);
                let bv = gen(&mut next, k * n, span_b);

                let want = product(m, k, n, &av, &bv);
                let suggested = suggested_bridge_scaled(uor_matmul_core::Shape { m, k, n });
                let kernel_full = crate::suggested_scratch(uor_matmul_core::Shape { m, k, n });
                // Every combination of compatibility buffers and decode-panel
                // offers must produce the same Atlas bytes.
                for scaled in [0, suggested] {
                    for kernel in [0, kernel_full] {
                        for (panels, accs) in [(0, 0), (k * n, m * n)] {
                            let got =
                                product_bridged(m, k, n, &av, &bv, scaled, panels, kernel, accs);
                            assert_eq!(
                                got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                                want.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                                "{label}, {m}x{k}x{n}, scaled {scaled}, kernel {kernel}, \
                                 panels {panels}"
                            );
                        }
                    }
                }
            }
        }

        // A deep reduction crosses many bounded octet panels; depth changes
        // only the number of chunks and never the exact accumulator bytes.
        let (m, k, n) = (4usize, 4096usize, 6usize);
        let av: Vec<f32> = (0..m * k)
            .map(|i| (8_388_608 + (i as u64 * 37) % 8_388_607) as f32)
            .collect();
        let bv: Vec<f32> = (0..k * n)
            .map(|i| {
                let x = (8_388_608 + (i as u64 * 53) % 8_388_607) as f32;
                if i % 3 == 0 {
                    -x
                } else {
                    x
                }
            })
            .collect();
        // A span of two binades on one side only.
        let av: Vec<f32> = av
            .iter()
            .enumerate()
            .map(|(i, &x)| x * 2.0f32.powi((i % 3) as i32 - 1))
            .collect();
        let want = product(m, k, n, &av, &bv);
        let suggested = suggested_bridge_scaled(uor_matmul_core::Shape { m, k, n });
        let kernel_full = crate::suggested_scratch(uor_matmul_core::Shape { m, k, n });
        for (kernel, accs) in [(0, 0), (kernel_full, 0), (kernel_full, m * n)] {
            let got = product_bridged(m, k, n, &av, &bv, suggested, 0, kernel, accs);
            assert_eq!(
                got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                want.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                "deep and chunked, kernel {kernel}, accs {accs}"
            );
        }

        // Non-finite codes reduce in the boundary coordinates beside the same
        // finite carrier, independent of every workspace offer.
        let av = [1.0f32, f32::INFINITY, 2.0, 3.0, f32::NAN, 4.0];
        let bv = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let want = product(2, 3, 2, &av, &bv);
        let got = product_bridged(2, 3, 2, &av, &bv, 3 * (2 + 2), 3 * 2, 1024, 0);
        assert_eq!(
            got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "non-finite boundary coordinates keep the driver's bytes"
        );
    }
    /// `CS-05`, R14: an all-zero panel absorbs every Laurent grade without
    /// projecting the other side into a shared-width carrier.
    #[test]
    fn a_zero_panel_against_a_wide_span_is_exact_cs_05() {
        let wide = [1e-30f32, 1e30, 1.0, 1e-20, 1e20, 2.0, 1e-10, 1e10, 3.0];
        let zeros = [0.0f32; 9];
        assert_eq!(product_packed(3, 3, 3, &zeros, &wide), vec![0.0; 9]);
        assert_eq!(product_packed(3, 3, 3, &wide, &zeros), vec![0.0; 9]);
        // A wide span on both sides applies the identical recursive coordinate
        // rule; grade distance changes placement addresses, not local width.
        assert_eq!(
            product_packed(3, 3, 3, &wide, &wide),
            product(3, 3, 3, &wide, &wide),
            "the packed and streaming traversals must agree on a wide span"
        );
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

    /// `CT-03`: the public compatibility reducer totalizes unequal panels by
    /// the Atlas zero. Both operand orderings are exercised because truncating
    /// `zip` used to make the result depend on which slice was shorter, and a
    /// debug-only assertion made debug and release disagree.
    #[test]
    fn unequal_packed_panels_zero_extend_totality_ct_03() {
        type Acc = AccOf<f64>;
        let code = |value: f64| PackedCode::of(value.decode());

        for (left, right) in [
            (vec![code(2.0), code(3.0)], vec![code(4.0)]),
            (vec![code(4.0)], vec![code(2.0), code(3.0)]),
        ] {
            let mut acc = <Acc as Accumulator>::ZERO;
            acc.accumulate_panels(&left, &right, PanelFacts::UNKNOWN);
            assert_eq!(f64::encode_from(acc, EncodeMode::Nearest), 8.0);
        }

        for (left, right) in [
            (vec![code(1.0), code(f64::INFINITY)], vec![code(1.0)]),
            (vec![code(1.0)], vec![code(1.0), code(f64::INFINITY)]),
        ] {
            let mut acc = <Acc as Accumulator>::ZERO;
            acc.accumulate_panels(&left, &right, PanelFacts::UNKNOWN);
            assert!(
                f64::encode_from(acc, EncodeMode::Nearest).is_nan(),
                "an omitted coordinate is zero, so infinity times it is NaN"
            );
        }
    }

    /// `CD-19`: restoring a public prescaled base remains a signed-coordinate
    /// operation at both `i32` edges. A hand-built panel outside the typed
    /// float range names no bit in this `Complete` register; it cannot panic or
    /// acquire profile-dependent wrapping semantics.
    #[test]
    fn prescaled_extreme_bases_are_total_cd_19() {
        type Acc = AccOf<f64>;
        let doubled = PackedCode::of(Decoded::Finite {
            sign: false,
            mantissa: 2,
            exp: 0,
        });
        for base in [i32::MIN, i32::MAX] {
            let mut acc = <Acc as Accumulator>::ZERO;
            acc.accumulate_panels(
                &[doubled],
                &[doubled],
                PanelFacts {
                    finite: true,
                    product_fits: true,
                    prescaled: Some(Prescaled { base, wide: true }),
                },
            );
            assert_eq!(
                f64::encode_from(acc, EncodeMode::Nearest).to_bits(),
                0.0f64.to_bits(),
                "base {base}"
            );
        }
    }

    /// `CD-05`: the encode mode is the only thing that changes the bytes for a
    /// fixed accumulation. `Nearest` rounds half to even; `TowardZero` truncates.
    ///
    /// The discriminating case has to be a sum that is *not* a tie and *not*
    /// representable, and this test had neither: it compared the two modes on a
    /// sum exactly half an ulp past 1.0, where nearest-even and truncation both
    /// keep 1.0 --- its own comment says so --- and then ran a second case that
    /// was a whole ulp, hence exact, hence unrounded, and ran it under `Nearest`
    /// only. Nothing here could fail on the encode mode, which is the one thing
    /// the ID names.
    ///
    /// Three quarters of an ulp is the case that separates them.
    #[test]
    fn the_encode_mode_is_the_only_variable_cd_05() {
        fn at_mode(a: &[f32], b: &[f32], encode: EncodeMode) -> f32 {
            let k = b.len();
            let mut c = [0.0f32];
            let av = MatView::row_major(a, 1, k).unwrap();
            let bv = MatView::row_major(b, k, 1).unwrap();
            let cv = MatViewMut::row_major(&mut c, 1, 1).unwrap();
            let mut t = Triple::new(av, bv, cv).unwrap();
            gemm_float(
                &mut t,
                &Linear::OVERWRITE,
                GemmOptions {
                    encode,
                    ..Default::default()
                },
            );
            c[0]
        }

        const HALF_ULP: f32 = f32::from_bits(0x3380_0000); // 2^-24
        const QUARTER_ULP: f32 = f32::from_bits(0x3300_0000); // 2^-25

        // `1 + 3/4 ulp`: above the midpoint, below the next value, and not a tie.
        // Nearest goes up, truncation stays. This is the assertion the ID is about.
        let ones = [1.0f32, 1.0, 1.0];
        let three_quarters = [1.0f32, HALF_ULP, QUARTER_ULP];
        let nearest = at_mode(&ones, &three_quarters, EncodeMode::Nearest);
        let toward_zero = at_mode(&ones, &three_quarters, EncodeMode::TowardZero);
        assert_eq!(
            nearest,
            1.0 + f32::EPSILON,
            "nearest rounds up past halfway"
        );
        assert_eq!(toward_zero, 1.0, "truncation keeps the smaller neighbour");
        assert_ne!(
            nearest, toward_zero,
            "the encode mode has to be able to change the bytes, or this ID \
             asserts nothing"
        );

        // And the tie, where they agree: half to even keeps the even mantissa and
        // truncation keeps the same value for its own reason. Recorded because it
        // is the case that *cannot* discriminate, which is what made this test
        // vacuous when it was the only case.
        let pair = [1.0f32, 1.0];
        let tie = [1.0f32, HALF_ULP];
        assert_eq!(at_mode(&pair, &tie, EncodeMode::Nearest), 1.0);
        assert_eq!(at_mode(&pair, &tie, EncodeMode::TowardZero), 1.0);

        // A sum that is exactly representable is unrounded, so both modes give it
        // back unchanged --- the third case, and the reason a full ulp could not
        // discriminate either.
        let exact = [1.0f32, f32::EPSILON];
        assert_eq!(
            at_mode(&pair, &exact, EncodeMode::Nearest),
            1.0 + f32::EPSILON
        );
        assert_eq!(
            at_mode(&pair, &exact, EncodeMode::TowardZero),
            1.0 + f32::EPSILON
        );
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
