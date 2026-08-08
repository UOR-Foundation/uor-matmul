//! The table sequences (§7.3).
//!
//! When the operand is a code, the product is a table read. `T[c][i]` is one
//! row `i` of `A` accumulated against the block of weights the code `c` decodes
//! to, so the column loop below it is one read and one add per code covering a
//! whole codeword, and contains no multiply at all.
//!
//! Two sequences, and every backend gives both:
//!
//! - [`TableSpec::build`] fills one slot: `T[c][i] = sum_t A[i][t] * D[c][t]`.
//! - [`TableSpec::gather`] reduces one group of output columns over a run of
//!   slots: `lane[u][i] += sum_slot T[slot][code[u][slot]][i]`.
//!
//! # Why they live here and not in the driver
//!
//! Every vector sequence in this workspace is behind `#[target_feature]` in
//! this crate, because that is the only place `unsafe` is permitted. A sequence
//! written anywhere else compiles at the target's baseline, and measured on an
//! AVX2 host that is the whole difference between 17.6 and 86.7 Gmac/s on the
//! column loop and between 2.1 and 25.1 Gprod/s on the build. The tabulated
//! traversal is where this library's arithmetic advantage lives, so it is the
//! last place that can afford to issue baseline code.
//!
//! # The arithmetic density
//!
//! One 256-bit add covers `8 * block` products at a 32-bit lane: eight output
//! rows, each carrying a whole codeword. That number has no dense counterpart
//! that is even the same shape --- `vpdpbusd`, the densest integer instruction
//! x86 has, covers 32 products and cannot be told to cover more --- because the
//! table's density is a property of the *codec*, and a codec may name a longer
//! block. At `Book<256, 8>` it is 64 products per add.
//!
//! # `no_std` and no allocation
//!
//! Every buffer is the caller's. These functions take pointers and lengths and
//! own nothing.

use uor_matmul_core::{Accumulator, Backend, Element};

use crate::MAX_TILE_LANES;

/// The sequences a build can run, reference first.
///
/// A chain of options rather than a fixed array, so the number a family may
/// have is not capped by a constant somebody chose (R8). An ISA entry answers
/// `None` for a `(rows, group)` it has no sequence for, which is the reference
/// carrying that tile and not a degraded path.
macro_rules! collect_table {
    ($($cond:expr => $spec:expr),* $(,)?) => {{
        core::iter::empty()
        $(
            .chain(core::iter::once_with(move || if $cond { $crate::table::IntoSpec::into_option($spec) } else { None }))
        )*
        .flatten()
    }};
}

/// The `(rows, group)` pairs the driver walks, each a compile-time pair.
///
/// The driver's column group is `COLUMN_LANES / rows`, so the accumulation is
/// sixteen lane words at every tile height and fits registers at every one of
/// them; a group of one is the tail a shape that does not divide leaves over.
/// A pair this list does not name takes the same sequence written for a runtime
/// tile, which computes the same integer and issues the address arithmetic the
/// compile-time forms do not need. Not a second method: one identity at a shape
/// the list has no constant for (R13).
macro_rules! dispatch_run {
    ($rows:expr, $group:expr, $any:expr, |$r:ident, $g:ident| $body:expr) => {
        match ($rows, $group) {
            (16, 1) => {
                const $r: usize = 16;
                const $g: usize = 1;
                $body
            }
            (8, 2) => {
                const $r: usize = 8;
                const $g: usize = 2;
                $body
            }
            (8, 1) => {
                const $r: usize = 8;
                const $g: usize = 1;
                $body
            }
            (4, 4) => {
                const $r: usize = 4;
                const $g: usize = 4;
                $body
            }
            (4, 1) => {
                const $r: usize = 4;
                const $g: usize = 1;
                $body
            }
            (2, 8) => {
                const $r: usize = 2;
                const $g: usize = 8;
                $body
            }
            (2, 1) => {
                const $r: usize = 2;
                const $g: usize = 1;
                $body
            }
            (1, 16) => {
                const $r: usize = 1;
                const $g: usize = 16;
                $body
            }
            (1, 1) => {
                const $r: usize = 1;
                const $g: usize = 1;
                $body
            }
            _ => $any,
        }
    };
}

/// The slab's code count, bound at whichever point the caller knows it.
///
/// A slab is `slab_codes(code_space) * rows` lane words and the boundary has
/// already asserted it is a power of two, so the only thing the codec
/// contributes to the column step is *one exponent* --- not a type, not a
/// traversal, and not a second sequence. Sixteen is the whole of it, because a
/// code is a `u16` and `2^16` codes is every code there can be.
///
/// The wildcard arm binds zero, which the runs read as "the caller did not know
/// it" and take from their argument. That is the same body at a different
/// binding, not a fallback (R13), and it is what keeps the enumeration from
/// being a ceiling: a code space this list does not name is computed by the
/// identical loop over a register instead of a literal (R8).
///
/// Measured on `1x1024x4096`, a one-row tile: with the count a literal the
/// column step runs at 9.4--14.9 Gmac/s and with it a register at 3.8--6.9, and
/// the shipped traversal sat at 5.35 inside the second band. The difference is
/// not the mask --- it is that a literal makes every slot's base a constant
/// displacement, so the slot loop unrolls and the slab cursor disappears.
macro_rules! dispatch_slab {
    ($codes:expr, |$c:ident| $body:expr) => {
        match $codes {
            1 => {
                const $c: usize = 1;
                $body
            }
            2 => {
                const $c: usize = 2;
                $body
            }
            4 => {
                const $c: usize = 4;
                $body
            }
            8 => {
                const $c: usize = 8;
                $body
            }
            16 => {
                const $c: usize = 16;
                $body
            }
            32 => {
                const $c: usize = 32;
                $body
            }
            64 => {
                const $c: usize = 64;
                $body
            }
            128 => {
                const $c: usize = 128;
                $body
            }
            256 => {
                const $c: usize = 256;
                $body
            }
            512 => {
                const $c: usize = 512;
                $body
            }
            1024 => {
                const $c: usize = 1024;
                $body
            }
            2048 => {
                const $c: usize = 2048;
                $body
            }
            4096 => {
                const $c: usize = 4096;
                $body
            }
            8192 => {
                const $c: usize = 8192;
                $body
            }
            16384 => {
                const $c: usize = 16384;
                $body
            }
            32768 => {
                const $c: usize = 32768;
                $body
            }
            65536 => {
                const $c: usize = 65536;
                $body
            }
            _ => {
                const $c: usize = 0;
                $body
            }
        }
    };
}

/// The codes one slab addresses, or zero when this slab is not `rows` copies of
/// them.
///
/// Zero is [`dispatch_slab`]'s wildcard, so a slab that does not factor is
/// computed by the same run over a runtime value rather than refused.
#[inline(always)]
const fn code_words(slab: usize, rows: usize) -> usize {
    if rows != 0 && slab.is_multiple_of(rows) {
        slab / rows
    } else {
        0
    }
}

/// A lane word: something a table entry's row can be added into.
///
/// The narrow/wide factorization at the place the table needs it. A table entry
/// is a sum of `block` products of two alphabet elements, and a chunk of the
/// reduction is a sum of those, so a narrow register holds it exactly for a
/// depth [`Lane::capacity`] states.
pub trait LaneWord: Copy + Send + Sync + 'static {
    /// The additive identity.
    const ZERO: Self;

    /// Exact within the lane's declared capacity, which is the only place it is
    /// reached.
    fn add(self, other: Self) -> Self;
}

/// A lane for a particular element type: how to fill it, and how to place it.
pub trait Lane<E: Element>: LaneWord {
    /// The most products this lane holds exactly, for an alphabet bounded by
    /// `b`.
    ///
    /// `None` is unbounded --- the exact accumulator, which the width
    /// derivation already sized against every depth any machine can address.
    ///
    /// This is not a limit on `k`. A deeper reduction is cut into runs of this
    /// many products and each run is placed into the exact accumulator once,
    /// which is the same chunking [`uor_matmul_core::fits_narrow`] already
    /// licenses for the tile kernels.
    fn capacity(b: u128) -> Option<usize>;

    /// Accumulate one exact product. The arguments are the element-sized panel
    /// cells declared by the table family: ordinarily the elements themselves;
    /// for a contextual projection such as `Scaled64`, the paired producer's
    /// in-place spelling. The latter is valid only as that producer/consumer
    /// composition, whose placement must equal the element product exactly.
    ///
    /// This is the only product contraction the generic build issues --- and
    /// at bound 1 not even this one: [`portable_table_bound1`] fills the same
    /// slot with adds and subtracts, because there every product is `+-a` or
    /// `0` (`CB-10`).
    fn mac(self, a: E, w: E) -> Self;

    /// Place a completed run into the exact accumulator.
    fn place(self, acc: E::Acc) -> E::Acc;

    /// Place a completed run into the exact accumulator, at `2^exponent`.
    ///
    /// The scale channel for a lane whose products were built from pre-scaled
    /// elements: the run holds an exact integer at one declared scale, and
    /// placing it is the accumulator's own `add_scaled` --- the decode's own
    /// primitive, so nothing is rounded on the way in. Every lane whose
    /// products are the elements' own is placed at `2^0`, which is what the
    /// default says; a nonzero exponent reaching it is a driver bug, asserted
    /// rather than dropped.
    fn place_scaled(self, acc: E::Acc, exponent: i32) -> E::Acc {
        debug_assert_eq!(exponent, 0, "a lane with no scale channel is placed at 2^0");
        self.place(acc)
    }
}

impl LaneWord for i32 {
    const ZERO: Self = 0;

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        // Exact within `capacity`, which is the only place it is reached. R5
        // asks the overflow behaviour to be written rather than inherited from
        // the build profile; the checked profile witnesses that it is not
        // reached (`CT-02`).
        self.wrapping_add(other)
    }
}

impl<E: Element> Lane<E> for i32 {
    #[inline]
    fn capacity(b: u128) -> Option<usize> {
        capacity_of(i32::MAX as u128, b)
    }

    #[inline(always)]
    fn mac(self, a: E, w: E) -> Self {
        E::mac_narrow32(self, a, w)
    }

    #[inline]
    fn place(self, acc: E::Acc) -> E::Acc {
        E::combine_narrow(acc, self as i64)
    }
}

impl LaneWord for i64 {
    const ZERO: Self = 0;

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        self.wrapping_add(other)
    }
}

impl<E: Element> Lane<E> for i64 {
    #[inline]
    fn capacity(b: u128) -> Option<usize> {
        capacity_of(i64::MAX as u128, b)
    }

    #[inline(always)]
    fn mac(self, a: E, w: E) -> Self {
        E::mac_narrow(self, a, w)
    }

    #[inline]
    fn place(self, acc: E::Acc) -> E::Acc {
        E::combine_narrow(acc, self)
    }
}

/// The depth a lane of `cap` holds for an alphabet bounded by `b`.
///
/// A bound of zero is an alphabet containing only zero, for which every depth
/// fits.
#[inline]
fn capacity_of(cap: u128, b: u128) -> Option<usize> {
    let per_step = b.saturating_mul(b); // R3-ok: a lane-width question, not an accumulation
    if per_step == 0 {
        return None;
    }
    Some(usize::try_from(cap / per_step).unwrap_or(usize::MAX).max(1))
}

/// The exact accumulator, as a lane.
///
/// The wrapper exists so that "the accumulator used as a lane" and "a narrow
/// register used as a lane" are two types rather than two code paths. It is
/// `repr(transparent)`, so a caller's accumulator offer *is* a buffer of these
/// and no copy stands between them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(transparent)]
pub struct Wide<A>(pub A);

impl<A> Wide<A> {
    /// A buffer of accumulators, read as a buffer of lanes.
    ///
    /// `repr(transparent)`, so this is a relabelling and never a copy: a
    /// caller's accumulator offer *is* a lane buffer.
    pub fn wrap_slice_mut(words: &mut [A]) -> &mut [Wide<A>] {
        let (ptr, len) = (words.as_mut_ptr(), words.len());
        // SAFETY: `Wide<A>` is `#[repr(transparent)]` over `A`, so the two have
        // the same size, alignment and validity, and the borrow is moved rather
        // than duplicated.
        unsafe { core::slice::from_raw_parts_mut(ptr.cast::<Wide<A>>(), len) }
    }
}

impl<A: Accumulator> LaneWord for Wide<A> {
    const ZERO: Self = Wide(A::ZERO);

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        Wide(self.0.combine(other.0))
    }
}

impl<E: Element> Lane<E> for Wide<E::Acc> {
    #[inline]
    fn capacity(_: u128) -> Option<usize> {
        // The width derivation already covers every depth any addressable
        // machine can present, so there is nothing left to bound (§3.2).
        None
    }

    #[inline(always)]
    fn mac(mut self, a: E, w: E) -> Self {
        E::mac(&mut self.0, a, w);
        self
    }

    #[inline]
    fn place(self, acc: E::Acc) -> E::Acc {
        acc.combine(self.0)
    }
}

/// A lane word in `Z/2^32`: the wrap is the encode, not an overflow.
///
/// The newtype exists because the blanket `impl<E: Element> Lane<E> for i32`
/// above already speaks for `i32`-as-lane with *exact* semantics --- one
/// product of the full `i32` alphabet and no more. The same bits read as a
/// quotient are a different lane: wrapping arithmetic, unbounded depth, and a
/// placement that is congruent mod `2^32` rather than equal. Admissibility is
/// the driver's question (`CU-08`); this type only states what the ring does.
///
/// Legitimate exactly when the caller asked to encode by wrapping into an
/// output no wider than 32 bits, because then the lane's own wrap *is* the
/// encode and nothing is lost that the caller did not ask to lose --- the same
/// argument [`crate::available_i32_modular`] makes for the dense tile.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(transparent)]
pub struct Mod32(pub i32);

impl Mod32 {
    /// A buffer of `i128` accumulators, read as a buffer of modular lanes.
    ///
    /// Four lanes per accumulator, so an offer sized for the exact lane holds
    /// the same table four times over. This is a relabelling, never a copy.
    pub fn wrap_i128s_mut(words: &mut [i128]) -> &mut [Mod32] {
        let (ptr, len) = (words.as_mut_ptr(), words.len());
        // SAFETY: `i128` is sixteen bytes aligned to sixteen and `Mod32` is
        // `#[repr(transparent)]` over `i32`, so the buffer is a valid buffer of
        // four times as many `Mod32`, and the borrow is moved rather than
        // duplicated.
        unsafe { core::slice::from_raw_parts_mut(ptr.cast::<Mod32>(), len * 4) }
    }
}

impl LaneWord for Mod32 {
    const ZERO: Self = Mod32(0);

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        // The ring's own addition: wrapping here is the quotient's arithmetic,
        // reached at every depth rather than only within a capacity.
        Mod32(self.0.wrapping_add(other.0))
    }
}

impl Lane<i32> for Mod32 {
    #[inline]
    fn capacity(_: u128) -> Option<usize> {
        // Unbounded at every bound: reduction mod `2^32` commutes with `+` and
        // `*`, so no depth can make the lane disagree with the encode. The
        // table-side form of `Factorization::Modular` (`KernelSpec::lane_depth`
        // returns `usize::MAX` for the same reason).
        None
    }

    #[inline(always)]
    fn mac(self, a: i32, w: i32) -> Self {
        // The only multiply the table issues, in the ring: `mullo` semantics.
        Mod32(self.0.wrapping_add(a.wrapping_mul(w)))
    }

    #[inline]
    fn place(self, acc: i128) -> i128 {
        // Congruent mod `2^32` to the exact sum, and the `Wrapping` encode
        // into a <= 32-bit output reads only the low limb --- the argument
        // `Kernelized::modular_as_acc` records on the dense side. With the
        // capacity unbounded a run is the whole column block, so this is
        // reached once per output element and `acc` cannot outgrow the exact
        // sum the width derivation sized it for.
        acc + i128::from(self.0)
    }
}

/// A lane word in `Z/2^64`, one width up from [`Mod32`] and portable-only.
///
/// The build's multiply is the table's only one, and no SIMD integer multiply
/// reaches an `i64` lane --- the same reason [`crate::available_i64_modular`]
/// lists the reference alone. The lane is still worth naming: the gather is
/// the column loop's whole arithmetic, and in the quotient it needs no exact
/// accumulator until placement.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(transparent)]
pub struct Mod64(pub i64);

impl Mod64 {
    /// A buffer of three-limb accumulators, read as a buffer of modular lanes.
    ///
    /// Three lanes per accumulator, for the same reason and in the same way as
    /// [`Mod32::wrap_i128s_mut`]: an offer sized for the exact lane relabelled,
    /// never copied.
    pub fn wrap_limbs_mut(words: &mut [uor_matmul_core::acc::Limbs<3>]) -> &mut [Mod64] {
        let (ptr, len) = (words.as_mut_ptr(), words.len());
        // SAFETY: `Limbs<3>` is twenty-four bytes aligned to eight --- one
        // `[u64; 3]` --- and `Mod64` is `#[repr(transparent)]` over `i64`, so
        // the buffer is a valid buffer of three times as many `Mod64`, and the
        // borrow is moved rather than duplicated.
        unsafe { core::slice::from_raw_parts_mut(ptr.cast::<Mod64>(), len * 3) }
    }
}

impl LaneWord for Mod64 {
    const ZERO: Self = Mod64(0);

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        Mod64(self.0.wrapping_add(other.0))
    }
}

impl Lane<i64> for Mod64 {
    #[inline]
    fn capacity(_: u128) -> Option<usize> {
        // Unbounded at every bound, as `Mod32`'s: the wrap is the encode.
        None
    }

    #[inline(always)]
    fn mac(self, a: i64, w: i64) -> Self {
        Mod64(self.0.wrapping_add(a.wrapping_mul(w)))
    }

    #[inline]
    fn place(self, acc: uor_matmul_core::acc::Limbs<3>) -> uor_matmul_core::acc::Limbs<3> {
        // The low limb *is* the value in `Z/2^64`, which is what the `Wrapping`
        // encode reads; the placement is congruent mod `2^64` to the exact sum
        // for the same reason `Mod32`'s is mod `2^32`.
        acc.add_i128(i128::from(self.0))
    }
}

/// The total contextual Atlas lane for symbol-tabulated `f32`.
///
/// Each in-place panel cell retains the source sign and fraction beside a
/// relative q grade. [`Lane::mac`] contracts the two significands through the
/// complete signed-octet lookup alphabet. A product that fits the established
/// common-grade interval remains the incumbent compact coefficient; otherwise
/// the top-positive model-derived interval carries its unshifted magnitude,
/// relative grade, and sign. The same interval also carries every Complete
/// non-finite union and one `SPLIT` marker used to scalar-fracture an unsafe
/// aggregate codec word. [`Lane::place_scaled`] resolves either spelling once
/// at `base_a + base_b` (`CD-32`).
///
/// The public transparent word and contextual trait signatures are unchanged.
/// Their nominal `f32` inputs are q cells produced by the paired tabulation
/// projection, not standalone IEEE operands; only that producer/consumer
/// composition has meaning. The locked trait surface cannot express that
/// distinction in its argument types, so this is deliberately not a claim
/// that an arbitrary standalone `f32` is a q cell. No additional carrier,
/// allocation, or copy exists.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(transparent)]
pub struct Scaled64(pub i64);

impl Scaled64 {
    /// A buffer of `i64` words, read as a buffer of scaled lanes.
    ///
    /// One lane per word, so the narrow offer the traversal accepts *is* a
    /// lane buffer. This is a relabelling, never a copy --- the same move
    /// [`Mod32::wrap_i128s_mut`] makes on the accumulator offer.
    pub fn wrap_i64s_mut(words: &mut [i64]) -> &mut [Scaled64] {
        let (ptr, len) = (words.as_mut_ptr(), words.len());
        // SAFETY: `Scaled64` is `#[repr(transparent)]` over `i64`, so the two
        // have the same size, alignment and validity, and the borrow is moved
        // rather than duplicated.
        unsafe { core::slice::from_raw_parts_mut(ptr.cast::<Scaled64>(), len) }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum F32QFactor {
    Finite {
        magnitude: u32,
        grade: u32,
        negative: bool,
    },
    Infinite {
        negative: bool,
    },
    NotANumber,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum F32QToken {
    Compact(i64),
    Finite {
        magnitude: u64,
        grade: u32,
        negative: bool,
    },
    Nonfinite(u8),
    Split,
}

#[inline(always)]
fn atlas_power_of_two_u32(bits: u32) -> u32 {
    let mut value = 1u32;
    for _ in 0..bits {
        value = value.wrapping_add(value);
    }
    value
}

#[inline(always)]
fn atlas_double_u64(mut value: u64, times: u32) -> u64 {
    for _ in 0..times {
        value = value.wrapping_add(value);
    }
    value
}

/// Decode the contextual binary32 cell by its two radix boundaries.
///
/// The conversion to `u32` is the symbol boundary. Everything after it is
/// quotient, remainder, and addition in the Atlas address; no mask, shift, or
/// floating-point operation participates in the contraction.
#[inline(always)]
fn decode_f32_q_factor(value: f32) -> F32QFactor {
    use crate::generated_capacity::f32_q;

    let raw: u32 = bytemuck::cast::<f32, u32>(value);
    let sign_place = atlas_power_of_two_u32(u32::BITS - 1);
    let negative = raw >= sign_place;
    let unsigned = if negative { raw - sign_place } else { raw };
    let fraction_place = atlas_power_of_two_u32(f32_q::SIGNIFICAND_BITS - 1);
    let q = unsigned / fraction_place;
    let fraction = unsigned % fraction_place;
    let maximum_relative =
        u32::try_from(f32_q::MAX_FACTOR_EXP - f32_q::MIN_FACTOR_EXP).unwrap_or(u32::MAX);
    let special_q = maximum_relative.wrapping_add(2);

    if q == special_q {
        if fraction == 0 {
            F32QFactor::Infinite { negative }
        } else {
            F32QFactor::NotANumber
        }
    } else if q == 0 {
        F32QFactor::Finite {
            magnitude: fraction,
            grade: 0,
            negative,
        }
    } else {
        F32QFactor::Finite {
            magnitude: fraction_place.wrapping_add(fraction),
            grade: q - 1,
            negative,
        }
    }
}

#[inline(always)]
fn f32_q_nonfinite_states() -> u32 {
    use crate::generated_capacity::f32_q;
    f32_q::STATE_COUNT - f32_q::SIGNED_FINITE_STATES
}

#[inline(always)]
fn f32_q_nonfinite_flag_count() -> u32 {
    let states = f32_q_nonfinite_states();
    let mut cardinality = 1u32;
    let mut flags = 0u32;
    while cardinality - 1 < states {
        cardinality = cardinality.wrapping_add(cardinality);
        flags = flags.wrapping_add(1);
    }
    flags
}

#[inline(always)]
fn f32_q_flag_place(ordinal: u32) -> u8 {
    let mut place = 1u8;
    for _ in 0..ordinal {
        place = place.wrapping_add(place);
    }
    place
}

#[inline(always)]
fn f32_q_has_flag(flags: u8, ordinal: u32) -> bool {
    let place = f32_q_flag_place(ordinal);
    let binary_radix = 1u8.wrapping_add(1);
    !(flags / place).is_multiple_of(binary_radix)
}

#[inline(always)]
fn f32_q_union_flags(left: u8, right: u8) -> u8 {
    let mut union = 0u8;
    for ordinal in 0..f32_q_nonfinite_flag_count() {
        if f32_q_has_flag(left, ordinal) || f32_q_has_flag(right, ordinal) {
            union = union.wrapping_add(f32_q_flag_place(ordinal));
        }
    }
    union
}

#[inline(always)]
fn f32_q_nan_flags() -> u8 {
    f32_q_flag_place(0)
}

#[inline(always)]
fn f32_q_infinity_flags(negative: bool) -> u8 {
    f32_q_flag_place(if negative { 2 } else { 1 })
}

#[inline(always)]
fn f32_q_finite_state(grade: u32, negative: bool) -> u32 {
    use crate::generated_capacity::f32_q;

    let signs = f32_q::SIGNED_FINITE_STATES / f32_q::RELATIVE_GRADE_COUNT;
    let mut state = 0u32;
    for _ in 0..signs {
        state = state.wrapping_add(grade);
    }
    state.wrapping_add(u32::from(negative))
}

#[inline(always)]
fn f32_q_tag(state: u32, magnitude: u64) -> Scaled64 {
    use crate::generated_capacity::f32_q;

    debug_assert_eq!(
        f32_q::PRODUCT_MAGNITUDE_BITS.wrapping_add(f32_q::STATE_BITS),
        f32_q::TAG_PAYLOAD_BITS
    );
    debug_assert!(magnitude < f32_q::MAGNITUDE_RADIX);
    let state_place = atlas_double_u64(state.into(), f32_q::PRODUCT_MAGNITUDE_BITS);
    let raw = f32_q::TAG_BASE
        .wrapping_add(state_place)
        .wrapping_add(magnitude);
    debug_assert!(raw < f32_q::TAG_BASE.wrapping_add(f32_q::TAG_INTERVAL));
    Scaled64(raw as i64)
}

#[inline(always)]
fn f32_q_split() -> Scaled64 {
    use crate::generated_capacity::f32_q;
    f32_q_tag(f32_q::SPLIT_STATE, 0)
}

#[inline(always)]
fn f32_q_nonfinite(flags: u8) -> Scaled64 {
    use crate::generated_capacity::f32_q;

    debug_assert!(flags != 0);
    debug_assert!(u32::from(flags) <= f32_q_nonfinite_states());
    let state = f32_q::SIGNED_FINITE_STATES
        .wrapping_add(u32::from(flags))
        .wrapping_sub(1);
    f32_q_tag(state, 0)
}

#[inline(always)]
fn decode_f32_q_token(word: Scaled64) -> F32QToken {
    use crate::generated_capacity::f32_q;

    if word.0 < 0 {
        return F32QToken::Compact(word.0);
    }
    let raw = word.0 as u64;
    if raw < f32_q::TAG_BASE {
        return F32QToken::Compact(word.0);
    }
    let payload = raw - f32_q::TAG_BASE;
    if payload >= f32_q::TAG_INTERVAL {
        return F32QToken::Nonfinite(f32_q_nan_flags());
    }
    let state = u32::try_from(payload / f32_q::MAGNITUDE_RADIX).unwrap_or(u32::MAX);
    let magnitude = payload % f32_q::MAGNITUDE_RADIX;
    if state < f32_q::SIGNED_FINITE_STATES {
        if magnitude == 0 {
            return F32QToken::Compact(0);
        }
        let signs = f32_q::SIGNED_FINITE_STATES / f32_q::RELATIVE_GRADE_COUNT;
        F32QToken::Finite {
            magnitude,
            grade: state / signs,
            negative: state % signs != 0,
        }
    } else if state < f32_q::STATE_COUNT {
        let flags = state
            .wrapping_sub(f32_q::SIGNED_FINITE_STATES)
            .wrapping_add(1);
        F32QToken::Nonfinite(u8::try_from(flags).unwrap_or_else(|_| f32_q_nan_flags()))
    } else if state == f32_q::SPLIT_STATE {
        F32QToken::Split
    } else {
        // The two remaining ten-bit ordinals have no semantic source. Reading
        // either as NaN keeps the contextual lane total without aliasing it to
        // a finite coefficient or allowing it into a compact run.
        F32QToken::Nonfinite(f32_q_nan_flags())
    }
}

#[inline(always)]
fn encode_f32_q_token(token: F32QToken) -> Scaled64 {
    match token {
        F32QToken::Compact(value) => Scaled64(value),
        F32QToken::Finite {
            magnitude,
            grade,
            negative,
        } => f32_q_tag(f32_q_finite_state(grade, negative), magnitude),
        F32QToken::Nonfinite(flags) => f32_q_nonfinite(flags),
        F32QToken::Split => f32_q_split(),
    }
}

const F32_Q_COORDINATE_CAPACITY: usize = core::mem::size_of::<f32>();
const F32_Q_GRADE_CAPACITY: usize = F32_Q_COORDINATE_CAPACITY + F32_Q_COORDINATE_CAPACITY - 1;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct F32QCoordinates {
    coordinates: [i8; F32_Q_COORDINATE_CAPACITY],
    extent: usize,
}

/// Project only as many balanced octets as this significand occupies.
///
/// Binary32 contributes at most 24 coefficient bits. Three unsigned octets
/// hold those bits and the balanced carry can occupy the existing fourth cell,
/// so the element-sized array is exact rather than a precision ceiling.
#[inline(always)]
fn atlas_balanced_u32_octets(magnitude: u32) -> F32QCoordinates {
    let radix = i64::from(u8::MAX).wrapping_add(1);
    let mut rest = i64::from(magnitude);
    let mut coordinates = [0i8; F32_Q_COORDINATE_CAPACITY];
    let mut extent = 0usize;
    while rest != 0 {
        // The model-derived 24-bit significand needs at most three unsigned
        // octets plus one balanced carry. Indexing keeps that derivation
        // load-bearing in release builds instead of silently truncating if it
        // ever drifts.
        let coordinate = &mut coordinates[extent];
        let residue = rest % radix;
        let digit = if residue > i64::from(i8::MAX) {
            residue - radix
        } else {
            residue
        };
        *coordinate = digit as i8;
        rest = (rest - digit) / radix;
        extent += 1;
    }
    F32QCoordinates {
        coordinates,
        extent,
    }
}

/// Contract the self-similar q words while exposing the actual work boundary.
///
/// The two observers make the V&V census nonvacuous: a test counts the lookup
/// and Horner callbacks that this exact loop issues. Production supplies the
/// canonical lookup and a zero-sized grade observer; inlining erases the
/// observation seam rather than adding a counter to the hot path.
#[inline(always)]
fn atlas_f32_q_magnitude_product_observed<L, G>(
    left: u32,
    right: u32,
    mut lookup: L,
    mut grade_observed: G,
) -> u64
where
    L: FnMut(i8, i8) -> i32,
    G: FnMut(),
{
    let left = atlas_balanced_u32_octets(left);
    let right = atlas_balanced_u32_octets(right);
    if left.extent == 0 || right.extent == 0 {
        return 0;
    }

    let mut grades = [0i64; F32_Q_GRADE_CAPACITY];
    for (left_grade, &left_coordinate) in left.coordinates[..left.extent].iter().enumerate() {
        for (right_grade, &right_coordinate) in right.coordinates[..right.extent].iter().enumerate()
        {
            let grade = left_grade + right_grade;
            grades[grade] =
                grades[grade].wrapping_add(i64::from(lookup(left_coordinate, right_coordinate)));
        }
    }

    let grade_extent = left.extent + right.extent - 1;
    let mut coordinates = grades[..grade_extent].iter().rev();
    let mut product = *coordinates
        .next()
        .expect("two nonempty factors have a nonempty product word");
    grade_observed();
    for &coordinate in coordinates {
        grade_observed();
        for _ in 0..i8::BITS {
            product = product.wrapping_add(product);
        }
        product = product.wrapping_add(coordinate);
    }
    debug_assert!(product >= 0);
    product as u64
}

#[inline(always)]
fn atlas_f32_q_magnitude_product(left: u32, right: u32) -> u64 {
    atlas_f32_q_magnitude_product_observed(left, right, crate::lookup::i8_product, || {})
}

#[inline(always)]
fn f32_q_finite_product(magnitude: u64, grade: u32, negative: bool) -> Scaled64 {
    use crate::generated_capacity::f32_q;

    debug_assert_eq!(
        f32_q::COMPACT_CEILING / f32_q::PRODUCT_BOUND,
        f32_q::ZERO_SPAN_CAPACITY
    );
    if magnitude == 0 {
        return Scaled64(0);
    }
    debug_assert!(magnitude <= f32_q::PRODUCT_BOUND);
    debug_assert!(grade < f32_q::RELATIVE_GRADE_COUNT);
    let mut compact = magnitude;
    for _ in 0..grade {
        if compact > f32_q::COMPACT_CEILING - compact {
            return encode_f32_q_token(F32QToken::Finite {
                magnitude,
                grade,
                negative,
            });
        }
        compact = compact.wrapping_add(compact);
    }
    let compact = compact as i64;
    if negative {
        Scaled64(0i64.wrapping_sub(compact))
    } else {
        Scaled64(compact)
    }
}

#[inline(always)]
fn f32_q_product(left: F32QFactor, right: F32QFactor) -> Scaled64 {
    match (left, right) {
        (F32QFactor::NotANumber, _) | (_, F32QFactor::NotANumber) => {
            f32_q_nonfinite(f32_q_nan_flags())
        }
        (F32QFactor::Infinite { .. }, F32QFactor::Finite { magnitude: 0, .. })
        | (F32QFactor::Finite { magnitude: 0, .. }, F32QFactor::Infinite { .. }) => {
            f32_q_nonfinite(f32_q_nan_flags())
        }
        (
            F32QFactor::Infinite {
                negative: left_negative,
            },
            F32QFactor::Infinite {
                negative: right_negative,
            }
            | F32QFactor::Finite {
                magnitude: 1..,
                negative: right_negative,
                ..
            },
        )
        | (
            F32QFactor::Finite {
                magnitude: 1..,
                negative: left_negative,
                ..
            },
            F32QFactor::Infinite {
                negative: right_negative,
            },
        ) => f32_q_nonfinite(f32_q_infinity_flags(left_negative != right_negative)),
        (
            F32QFactor::Finite {
                magnitude: left_magnitude,
                grade: left_grade,
                negative: left_negative,
            },
            F32QFactor::Finite {
                magnitude: right_magnitude,
                grade: right_grade,
                negative: right_negative,
            },
        ) => f32_q_finite_product(
            atlas_f32_q_magnitude_product(left_magnitude, right_magnitude),
            left_grade.wrapping_add(right_grade),
            left_negative != right_negative,
        ),
    }
}

#[inline(always)]
fn f32_q_add_compact(left: i64, right: i64) -> Scaled64 {
    use crate::generated_capacity::f32_q;

    if left >= 0 && right >= 0 {
        let left = left as u64;
        let right = right as u64;
        if left > f32_q::COMPACT_CEILING
            || right > f32_q::COMPACT_CEILING
            || left > f32_q::COMPACT_CEILING - right
        {
            return f32_q_split();
        }
        Scaled64(left.wrapping_add(right) as i64)
    } else if left <= 0 && right <= 0 {
        let left = left.unsigned_abs();
        let right = right.unsigned_abs();
        if left > f32_q::COMPACT_CEILING
            || right > f32_q::COMPACT_CEILING
            || left > f32_q::COMPACT_CEILING - right
        {
            return f32_q_split();
        }
        Scaled64(0i64.wrapping_sub(left.wrapping_add(right) as i64))
    } else {
        // Opposite signs subtract magnitudes, so their sum lies between the
        // operands and cannot cross either compact boundary.
        Scaled64(left.wrapping_add(right))
    }
}

#[inline(always)]
fn f32_q_add_words(left: Scaled64, right: Scaled64) -> Scaled64 {
    use F32QToken::{Compact, Finite, Nonfinite, Split};

    // `LaneWord::ZERO` is an identity for every bit pattern the public
    // transparent word can carry, including an out-of-protocol word. Do this
    // before semantic decoding so identity cannot normalize or fracture it.
    if left.0 == 0 {
        return right;
    }
    if right.0 == 0 {
        return left;
    }
    match (decode_f32_q_token(left), decode_f32_q_token(right)) {
        (Split, _) | (_, Split) => f32_q_split(),
        (Compact(left), Compact(right)) => f32_q_add_compact(left, right),
        (Nonfinite(left), Nonfinite(right)) => f32_q_nonfinite(f32_q_union_flags(left, right)),
        // A semantic tag is a one-product atom. Combining a special with a
        // finite contribution would discard that finite residue just as
        // combining two finite grades would discard a grade. The scheduler
        // isolates specials; an aggregate build asks it to scalar-fracture.
        (Nonfinite(_), Compact(_))
        | (Compact(_), Nonfinite(_))
        | (Nonfinite(_), Finite { .. })
        | (Finite { .. }, Nonfinite(_))
        | (Finite { .. }, Compact(_))
        | (Compact(_), Finite { .. })
        | (Finite { .. }, Finite { .. }) => f32_q_split(),
    }
}

impl LaneWord for Scaled64 {
    const ZERO: Self = Scaled64(0);

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        f32_q_add_words(self, other)
    }
}

/// Recover the one contextual carry of a four-octet coefficient carrier.
///
/// Four signed radix-256 coordinates spell exactly `2^32` consecutive values,
/// but that interval begins slightly below `i32::MIN`. The high positive tail
/// of the admitted coefficient alphabet consequently has the unique spelling
/// `carrier + 256^4`. It is recognizable without reifying the coefficient:
/// the highest coordinate is `-128` and the lower three-coordinate prefix is
/// negative. The highest nonzero coordinate decides that prefix's sign because
/// one radix place is wider than every lower place combined.
#[inline(always)]
#[cfg(test)]
fn atlas_octets_with_context(carrier: [i8; 4]) -> [i8; 5] {
    let mut coordinates = [0i8; 5];
    coordinates[..4].copy_from_slice(&carrier);
    if carrier[3] == i8::MIN {
        for &coordinate in carrier[..3].iter().rev() {
            if coordinate != 0 {
                coordinates[4] = if coordinate < 0 { 1 } else { 0 };
                break;
            }
        }
    }
    coordinates
}

/// Evaluate two four-octet common-grade carriers through the Atlas product.
///
/// The cells reaching this function have already been projected once, in
/// place, by the symbol traversal. Each coordinate product is therefore one
/// canonical signed-`i8` lookup. Horner placement is the same radix recurrence
/// used by the projection: eight doublings and one add per occupied grade. No
/// significand decode, runtime value multiply, shift, or second carrier exists
/// in the table build.
#[inline]
#[cfg(test)]
fn atlas_octet_product(left: [i8; 4], right: [i8; 4]) -> i64 {
    let left = atlas_octets_with_context(left);
    let right = atlas_octets_with_context(right);
    let mut grades = [0i64; 9];
    for (left_grade, &left_coordinate) in left.iter().enumerate() {
        for (right_grade, &right_coordinate) in right.iter().enumerate() {
            let grade = left_grade + right_grade;
            grades[grade] = grades[grade].wrapping_add(i64::from(crate::lookup::i8_product(
                left_coordinate,
                right_coordinate,
            )));
        }
    }

    let mut product = 0i64;
    for &coordinate in grades.iter().rev() {
        for _ in 0..i8::BITS {
            product = product.wrapping_add(product);
        }
        product = product.wrapping_add(coordinate);
    }
    product
}

#[cfg(test)]
// These representation laws intentionally sit beside the private carrier
// helpers they exhaust; keeping them here makes a later table-sequence item
// follow the test module without changing the shipped item order.
#[allow(clippy::items_after_test_module)]
mod scaled64_tests {
    use super::{
        atlas_balanced_u32_octets, atlas_f32_q_magnitude_product,
        atlas_f32_q_magnitude_product_observed, atlas_octet_product, gray_sign_build_adds,
        product_build_adds, Lane, LaneWord, Scaled64,
    };
    use core::cell::Cell;
    use uor_matmul_core::{Accumulator, Element};

    type F32Acc = <f32 as Element>::Acc;

    fn checked_model() -> uor_matmul_model::Model {
        let model = uor_matmul_model::Model::load_from_repo_root().expect("the model loads");
        model.check().expect("the model-derived q carrier checks");
        model
    }

    fn power_of_two(bits: u32) -> u128 {
        uor_matmul_model::derive::power_of_two(bits).expect("the binary32 field fits u128")
    }

    /// Test-side spelling of the contextual carrier. The oracle deliberately
    /// uses field arithmetic, independently of the lookup/add consumer.
    fn q_cell(negative: bool, q: u32, fraction: u32) -> f32 {
        let fraction_bits = f32::MANTISSA_DIGITS - 1;
        let exponent_bits = u32::BITS - f32::MANTISSA_DIGITS;
        let fraction_radix = u32::try_from(power_of_two(fraction_bits)).unwrap();
        let exponent_radix = u32::try_from(power_of_two(exponent_bits)).unwrap();
        assert!(q < exponent_radix);
        assert!(fraction < fraction_radix);
        let sign_place = u32::try_from(power_of_two(u32::BITS - 1)).unwrap();
        let bits = u32::from(negative)
            .checked_mul(sign_place)
            .and_then(|sign| sign.checked_add(q.checked_mul(fraction_radix)?))
            .and_then(|head| head.checked_add(fraction))
            .expect("the disjoint binary32 fields fit one word");
        f32::from_bits(bits)
    }

    fn q_mac(left: f32, right: f32) -> Scaled64 {
        <Scaled64 as Lane<f32>>::mac(Scaled64(0), left, right)
    }

    fn place_into(token: Scaled64, acc: F32Acc, exponent: i32) -> F32Acc {
        <Scaled64 as Lane<f32>>::place_scaled(token, acc, exponent)
    }

    fn zero_acc() -> F32Acc {
        <F32Acc as Accumulator>::ZERO
    }

    /// Independent raw tag spelling. Production must recover this state by
    /// lookup/add logic; constructing the expected word here does not exercise
    /// or duplicate that decoder.
    fn tagged(q: &uor_matmul_model::registry::F32QCarrier, state: u32, magnitude: u64) -> Scaled64 {
        let radix = power_of_two(q.product_magnitude_bits);
        assert!(u128::from(magnitude) < radix);
        let raw = u128::from(q.tag_base)
            .checked_add(u128::from(state).checked_mul(radix).unwrap())
            .and_then(|head| head.checked_add(u128::from(magnitude)))
            .expect("a model-admitted tag fits its interval");
        assert!(raw < u128::from(q.tag_base) + u128::from(q.tag_interval));
        Scaled64(i64::try_from(raw).expect("the tag interval is positive i64"))
    }

    fn complete_for_mask(mask: u8) -> F32Acc {
        let mut acc = zero_acc();
        if mask & (1 << 0) != 0 {
            acc.set_nan();
        }
        if mask & (1 << 1) != 0 {
            acc.set_infinity(false);
        }
        if mask & (1 << 2) != 0 {
            acc.set_infinity(true);
        }
        acc
    }

    fn balanced_octets(value: i32) -> [i8; 4] {
        let mut rest = i64::from(value);
        let mut coordinates = [0i8; 4];
        for coordinate in &mut coordinates {
            *coordinate = (rest as u8) as i8;
            rest = (rest - i64::from(*coordinate)) / 256;
        }
        assert!(rest == 0 || rest == 1);
        coordinates
    }

    /// `CD-20`: the balanced-octet contraction is the signed product for the
    /// whole lookup alphabet and at every boundary of the admitted coefficient
    /// width. The oracle is ordinary widened arithmetic in test code, separate
    /// from the lookup/add body under test.
    #[test]
    fn balanced_octets_match_the_independent_wide_product_cd_20() {
        for left in i8::MIN..=i8::MAX {
            for right in i8::MIN..=i8::MAX {
                assert_eq!(
                    atlas_octet_product([left, 0, 0, 0], [right, 0, 0, 0],),
                    i64::from(left) * i64::from(right),
                    "signed-octet pair ({left}, {right})"
                );
            }
        }

        let boundaries = [
            i64::from(i32::MIN),
            i64::from(i32::MIN) + 1,
            -256,
            -129,
            -1,
            0,
            1,
            127,
            255,
            256,
            i64::from(i32::MAX) - 1,
            i64::from(i32::MAX),
        ];
        for &left in &boundaries {
            for &right in &boundaries {
                let expected = i64::try_from(i128::from(left) * i128::from(right))
                    .expect("two signed 32-bit coefficients have a signed 64-bit product");
                assert_eq!(
                    atlas_octet_product(
                        balanced_octets(i32::try_from(left).unwrap()),
                        balanced_octets(i32::try_from(right).unwrap()),
                    ),
                    expected,
                    "coefficient-boundary pair ({left}, {right})"
                );
            }
        }
    }

    /// `CD-32`: the public lane word's declared zero is an identity before
    /// contextual interpretation. This includes every semantic token class,
    /// both compact extremes, and unused raw tag ordinals; identity may not
    /// normalize or scalar-fracture an opaque word.
    #[test]
    fn scaled64_zero_is_raw_identity_for_every_token_class_cd_32() {
        let model = checked_model();
        let q = &model.widths.f32_q_carrier;
        let zero = <Scaled64 as LaneWord>::ZERO;
        let words = [
            Scaled64(i64::MIN),
            Scaled64(-1),
            zero,
            Scaled64(1),
            Scaled64(i64::try_from(q.compact_ceiling).unwrap()),
            tagged(q, 0, q.product_bound),
            tagged(q, q.signed_finite_states, 0),
            tagged(q, q.state_count, 0),
            Scaled64(i64::MAX),
        ];
        for word in words {
            assert_eq!(
                <Scaled64 as LaneWord>::add(zero, word),
                word,
                "left identity for {word:?}"
            );
            assert_eq!(
                <Scaled64 as LaneWord>::add(word, zero),
                word,
                "right identity for {word:?}"
            );
        }
    }

    /// `CD-32`: a special token is singleton execution state, not permission
    /// to erase a finite contribution in the same table entry. Every mixed
    /// aggregate therefore requests scalar fracture, while two special states
    /// retain the exact union required by Complete.
    #[test]
    fn mixed_nonfinite_and_finite_words_scalar_fracture_cd_32() {
        let model = checked_model();
        let q = &model.widths.f32_q_carrier;
        let complete = &model.widths.complete_state;
        let split = tagged(q, q.state_count, 0);
        let finite = [Scaled64(1), Scaled64(-1), tagged(q, 0, q.product_bound)];
        for left_mask in 1..=complete.nonfinite_states {
            let special = tagged(q, q.signed_finite_states + left_mask - 1, 0);
            for finite in finite {
                assert_eq!(
                    <Scaled64 as LaneWord>::add(special, finite),
                    split,
                    "special {left_mask:#05b} before {finite:?}"
                );
                assert_eq!(
                    <Scaled64 as LaneWord>::add(finite, special),
                    split,
                    "{finite:?} before special {left_mask:#05b}"
                );
            }
            for right_mask in 1..=complete.nonfinite_states {
                let right = tagged(q, q.signed_finite_states + right_mask - 1, 0);
                let union = left_mask | right_mask;
                assert_eq!(
                    <Scaled64 as LaneWord>::add(special, right),
                    tagged(q, q.signed_finite_states + union - 1, 0),
                    "special union {left_mask:#05b} | {right_mask:#05b}"
                );
            }
        }
    }

    /// `CD-32`: q contraction follows source precision rather than a fixed
    /// five-octet frame. The low-octet residue alphabet is exhaustive; the
    /// complete binary32 coefficient boundaries are differential against a
    /// widened test oracle; and observers count the exact lookup rectangle and
    /// occupied Horner grades issued by the production recurrence.
    #[test]
    fn q_precision_fractal_matches_wide_product_and_exact_work_cd_32() {
        for left in 0u32..=u32::from(u8::MAX) {
            for right in 0u32..=u32::from(u8::MAX) {
                assert_eq!(
                    atlas_f32_q_magnitude_product(left, right),
                    u64::from(left) * u64::from(right),
                    "exhaustive one-octet pair ({left}, {right})"
                );
            }
        }

        let model = checked_model();
        let q = &model.widths.f32_q_carrier;
        let hidden = u32::try_from(power_of_two(q.significand_bits - 1)).unwrap();
        let maximum = u32::try_from(power_of_two(q.significand_bits)).unwrap() - 1;
        let boundaries = [
            0,
            1,
            u32::from(i8::MAX as u8),
            u32::from(u8::MAX),
            u32::from(u8::MAX) + 1,
            u32::from(u16::MAX),
            u32::from(u16::MAX) + 1,
            hidden - 1,
            hidden,
            maximum,
        ];
        for &left in &boundaries {
            for &right in &boundaries {
                let lookups = Cell::new(0usize);
                let grades = Cell::new(0usize);
                let got = atlas_f32_q_magnitude_product_observed(
                    left,
                    right,
                    |a, b| {
                        lookups.set(lookups.get() + 1);
                        crate::lookup::i8_product(a, b)
                    },
                    || grades.set(grades.get() + 1),
                );
                let left_extent = atlas_balanced_u32_octets(left).extent;
                let right_extent = atlas_balanced_u32_octets(right).extent;
                let expected_lookups = left_extent * right_extent;
                let expected_grades = if expected_lookups == 0 {
                    0
                } else {
                    left_extent + right_extent - 1
                };
                assert_eq!(got, u64::from(left) * u64::from(right));
                assert_eq!(
                    lookups.get(),
                    expected_lookups,
                    "lookup pair ({left}, {right})"
                );
                assert_eq!(
                    grades.get(),
                    expected_grades,
                    "grade pair ({left}, {right})"
                );
            }
        }

        assert_eq!(atlas_balanced_u32_octets(1).extent, 1);
        assert_eq!(atlas_balanced_u32_octets(maximum).extent, 4);
        let one_lookups = Cell::new(0usize);
        let one_grades = Cell::new(0usize);
        assert_eq!(
            atlas_f32_q_magnitude_product_observed(
                1,
                1,
                |a, b| {
                    one_lookups.set(one_lookups.get() + 1);
                    crate::lookup::i8_product(a, b)
                },
                || one_grades.set(one_grades.get() + 1),
            ),
            1
        );
        assert_eq!((one_lookups.get(), one_grades.get()), (1, 1));

        let max_lookups = Cell::new(0usize);
        let max_grades = Cell::new(0usize);
        assert_eq!(
            atlas_f32_q_magnitude_product_observed(
                maximum,
                maximum,
                |a, b| {
                    max_lookups.set(max_lookups.get() + 1);
                    crate::lookup::i8_product(a, b)
                },
                || max_grades.set(max_grades.get() + 1),
            ),
            u64::from(maximum) * u64::from(maximum)
        );
        assert_eq!((max_lookups.get(), max_grades.get()), (16, 7));
    }

    /// `CD-32`: every field boundary of the contextual q carrier reaches the
    /// same complete dyadic as its source code. In particular, q zero keeps
    /// zero and subnormal coefficients distinct, q one introduces the normal
    /// hidden coefficient, the last finite q reaches the complete factor
    /// exponent span, sign is orthogonal, and the all-ones q keeps infinity
    /// distinct from a NaN payload.
    #[test]
    fn q_carrier_round_trips_ieee_boundaries_cd_32() {
        let model = checked_model();
        let q = &model.widths.f32_q_carrier;
        assert_eq!(q.significand_bits, f32::MANTISSA_DIGITS);

        let fraction_bits = q.significand_bits - 1;
        let fraction_radix = u32::try_from(power_of_two(fraction_bits)).unwrap();
        let fraction_max = fraction_radix - 1;
        let exponent_bits = u32::BITS - q.significand_bits;
        let special_q = u32::try_from(power_of_two(exponent_bits)).unwrap() - 1;
        let maximum_relative = u32::try_from(q.max_factor_exp - q.min_factor_exp).unwrap();
        let maximum_finite_q = maximum_relative + 1;
        assert!(maximum_finite_q < special_q);

        // `1.0` is the normal coefficient `2^(p-1)` at base `-(p-1)`.
        let unit = q_cell(false, 1, 0);
        let unit_base = -i32::try_from(fraction_bits).unwrap();
        let product_base = q.min_factor_exp.checked_add(unit_base).unwrap();
        let finite = [
            ("positive zero", false, 0, 0),
            ("negative zero", true, 0, 0),
            ("minimum subnormal", false, 0, 1),
            ("negative minimum subnormal", true, 0, 1),
            ("maximum subnormal", false, 0, fraction_max),
            ("minimum normal", false, 1, 0),
            ("negative minimum normal", true, 1, 0),
            (
                "maximum finite grade",
                false,
                maximum_finite_q,
                fraction_max,
            ),
            (
                "negative maximum finite grade",
                true,
                maximum_finite_q,
                fraction_max,
            ),
        ];
        for (label, negative, q_field, fraction) in finite {
            let relative = if q_field == 0 { 0 } else { q_field - 1 };
            let coefficient = if q_field == 0 {
                u128::from(fraction)
            } else {
                u128::from(fraction_radix) + u128::from(fraction)
            };
            let got = place_into(
                q_mac(q_cell(negative, q_field, fraction), unit),
                zero_acc(),
                product_base,
            );
            let mut want = zero_acc();
            want.add_scaled(
                coefficient,
                q.min_factor_exp
                    .checked_add(i32::try_from(relative).unwrap())
                    .unwrap(),
                negative,
            );
            assert_eq!(got, want, "{label}");
        }

        for (label, negative) in [("positive infinity", false), ("negative infinity", true)] {
            let got = place_into(
                q_mac(q_cell(negative, special_q, 0), unit),
                zero_acc(),
                product_base,
            );
            let mut want = zero_acc();
            want.set_infinity(negative);
            assert_eq!(got, want, "{label}");
        }
        for negative in [false, true] {
            let got = place_into(
                q_mac(q_cell(negative, special_q, 1), unit),
                zero_acc(),
                product_base,
            );
            let mut want = zero_acc();
            want.set_nan();
            assert_eq!(got, want, "NaN payload with sign={negative}");
        }
    }

    /// `CD-32`: the raw top-positive interval is a lossless semantic round
    /// trip for every one of the 1,021 model-derived states. The immediately
    /// following unused state is the aggregate `SPLIT` marker, so compact
    /// overflow cannot alias a finite grade or a Complete non-finite union.
    #[test]
    fn tag_interval_round_trips_every_state_and_split_cd_32() {
        let model = checked_model();
        let q = &model.widths.f32_q_carrier;
        let complete = &model.widths.complete_state;
        let sign_count = q.signed_finite_states / q.relative_grade_count;
        assert_eq!(
            q.signed_finite_states + complete.nonfinite_states,
            q.state_count
        );
        assert_eq!(sign_count * q.relative_grade_count, q.signed_finite_states);

        let product_base = q.min_factor_exp.checked_mul(2).unwrap();
        for state in 0..q.signed_finite_states {
            let grade = state / sign_count;
            let negative = state % sign_count != 0;
            let token = tagged(q, state, q.product_bound);
            let got = place_into(token, zero_acc(), product_base);
            let mut want = zero_acc();
            want.add_scaled(
                u128::from(q.product_bound),
                product_base
                    .checked_add(i32::try_from(grade).unwrap())
                    .unwrap(),
                negative,
            );
            assert_eq!(got, want, "finite tag state {state}");
        }
        for mask in 1..=complete.nonfinite_states {
            let state = q.signed_finite_states + mask - 1;
            let got = place_into(tagged(q, state, 0), zero_acc(), product_base);
            assert_eq!(
                got,
                complete_for_mask(u8::try_from(mask).unwrap()),
                "non-finite tag state {state}"
            );
        }

        let radix = u64::try_from(power_of_two(q.product_magnitude_bits)).unwrap();
        let last_valid = tagged(q, q.state_count - 1, radix - 1);
        let split = tagged(q, q.state_count, 0);
        assert_eq!(split.0, last_valid.0.checked_add(1).unwrap());
        assert_eq!(tagged(q, 0, 0).0, i64::try_from(q.tag_base).unwrap());
        assert!(
            u64::try_from(split.0).unwrap() < q.tag_base.checked_add(q.tag_interval).unwrap(),
            "the derived ten-bit state field leaves the SPLIT ordinal in the same interval"
        );
    }

    /// `CD-32`: all seven nonempty unions of `{NaN,+Inf,-Inf}` have distinct
    /// tag spellings, and direct union placement agrees with placing the same
    /// singleton observations one at a time into `Complete`.
    #[test]
    fn all_complete_nonfinite_unions_survive_tag_placement_cd_32() {
        let model = checked_model();
        let q = &model.widths.f32_q_carrier;
        let complete = &model.widths.complete_state;
        let base = q.min_factor_exp.checked_mul(2).unwrap();
        for mask in 1..=u8::try_from(complete.nonfinite_states).unwrap() {
            let union_state = q.signed_finite_states + u32::from(mask) - 1;
            let direct = place_into(tagged(q, union_state, 0), zero_acc(), base);
            let mut joined = zero_acc();
            for singleton in [1u8 << 0, 1u8 << 1, 1u8 << 2] {
                if mask & singleton != 0 {
                    let state = q.signed_finite_states + u32::from(singleton) - 1;
                    joined = place_into(tagged(q, state, 0), joined, base);
                }
            }
            let want = complete_for_mask(mask);
            assert_eq!(direct, want, "direct union mask {mask:#05b}");
            assert_eq!(joined, want, "joined singleton mask {mask:#05b}");

            let mut residue = zero_acc();
            residue.add_scaled(u128::from(q.product_bound), base, true);
            let got = place_into(tagged(q, union_state, 0), residue, base);
            let mut want = residue;
            if mask & (1 << 0) != 0 {
                want.set_nan();
            }
            if mask & (1 << 1) != 0 {
                want.set_infinity(false);
            }
            if mask & (1 << 2) != 0 {
                want.set_infinity(true);
            }
            assert_eq!(
                got, want,
                "immediate union placement preserves the finite residue for mask {mask:#05b}"
            );
        }
    }

    /// `CD-32`: infinity times either signed zero is NaN before any compact
    /// coefficient can be formed. Operand signs cannot turn the invalid IEEE
    /// product into a signed infinity or a finite zero.
    #[test]
    fn infinity_times_zero_is_nan_in_q_carrier_cd_32() {
        let model = checked_model();
        let q = &model.widths.f32_q_carrier;
        let exponent_bits = u32::BITS - q.significand_bits;
        let special_q = u32::try_from(power_of_two(exponent_bits)).unwrap() - 1;
        for infinity_negative in [false, true] {
            for zero_negative in [false, true] {
                let token = q_mac(
                    q_cell(infinity_negative, special_q, 0),
                    q_cell(zero_negative, 0, 0),
                );
                let got = place_into(token, zero_acc(), q.min_factor_exp.checked_mul(2).unwrap());
                let mut want = zero_acc();
                want.set_nan();
                assert_eq!(
                    got, want,
                    "infinity sign={infinity_negative}, zero sign={zero_negative}"
                );
            }
        }
    }

    /// `CD-32`: the formerly admitted common-grade region remains byte-for-
    /// byte the incumbent compact coefficient. The carrier extension is paid
    /// only when a grade cannot inhabit that region; it cannot perturb its hot
    /// lookup/add composition.
    #[test]
    fn former_common_grade_compact_composition_is_unchanged_cd_32() {
        let model = checked_model();
        let q = &model.widths.f32_q_carrier;
        let fraction_bits = q.significand_bits - 1;
        let fraction_radix = u32::try_from(power_of_two(fraction_bits)).unwrap();
        let fraction_max = fraction_radix - 1;
        let incumbent_span = (i32::BITS - 1).checked_sub(q.significand_bits).unwrap();
        let grades = [0, incumbent_span / 2, incumbent_span];

        for left_grade in grades {
            for right_grade in grades {
                for left_fraction in [0, fraction_max] {
                    for right_fraction in [0, fraction_max] {
                        for left_negative in [false, true] {
                            for right_negative in [false, true] {
                                let left = q_cell(left_negative, left_grade + 1, left_fraction);
                                let right = q_cell(right_negative, right_grade + 1, right_fraction);
                                let left_coefficient =
                                    u128::from(fraction_radix) + u128::from(left_fraction);
                                let right_coefficient =
                                    u128::from(fraction_radix) + u128::from(right_fraction);
                                let grade_scale = power_of_two(left_grade + right_grade);
                                let magnitude = left_coefficient
                                    .checked_mul(right_coefficient)
                                    .and_then(|product| product.checked_mul(grade_scale))
                                    .unwrap();
                                assert!(magnitude <= u128::from(q.compact_ceiling));
                                let magnitude = i64::try_from(magnitude).unwrap();
                                let expected = if left_negative != right_negative {
                                    magnitude.checked_neg().unwrap()
                                } else {
                                    magnitude
                                };
                                assert_eq!(
                                    q_mac(left, right),
                                    Scaled64(expected),
                                    "grades=({left_grade},{right_grade}), fractions=({left_fraction},{right_fraction}), signs=({left_negative},{right_negative})"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// `CD-32`: exactly the model-derived zero-span capacity remains compact;
    /// the next product becomes the otherwise-unused `SPLIT` state instead of
    /// entering the reserved interval with an accidental semantic tag.
    #[test]
    fn compact_ceiling_accepts_capacity_and_marks_cap_plus_one_split_cd_32() {
        let model = checked_model();
        let q = &model.widths.f32_q_carrier;
        assert_eq!(
            <Scaled64 as Lane<f32>>::capacity(u128::from(q.product_bound)),
            Some(usize::try_from(q.zero_span_capacity).unwrap()),
            "the contextual Lane declaration consumes an already-product bound exactly once"
        );
        assert_eq!(
            <Scaled64 as Lane<f32>>::capacity(u128::from(q.compact_ceiling) + 1),
            Some(1)
        );
        assert_eq!(<Scaled64 as Lane<f32>>::capacity(0), None);
        let product = Scaled64(i64::try_from(q.product_bound).unwrap());
        let mut at_capacity = Scaled64(0);
        for _ in 0..q.zero_span_capacity {
            at_capacity = <Scaled64 as LaneWord>::add(at_capacity, product);
        }
        let expected = u128::from(q.product_bound)
            .checked_mul(u128::from(q.zero_span_capacity))
            .unwrap();
        assert_eq!(at_capacity.0, i64::try_from(expected).unwrap());
        assert!(u64::try_from(at_capacity.0).unwrap() <= q.compact_ceiling);

        let cap_plus_one = <Scaled64 as LaneWord>::add(at_capacity, product);
        assert_eq!(
            cap_plus_one,
            tagged(q, q.state_count, 0),
            "compact overflow must be the SPLIT marker, not a valid semantic tag"
        );
    }

    /// `CU-11`: diagnostic charges are total at the address boundary and
    /// saturate only the bounded `u64` census, never an intermediate `usize`.
    #[test]
    fn table_build_charges_are_total_census_queries_cu_11() {
        assert_eq!(product_build_adds(16, 4, 8), 512);
        assert_eq!(gray_sign_build_adds(256, 8, 16), 4336);
        assert_eq!(
            product_build_adds(usize::MAX, usize::MAX, usize::MAX),
            u64::MAX
        );
        assert_eq!(
            gray_sign_build_adds(usize::MAX, usize::MAX, usize::MAX),
            u64::MAX
        );
        assert_eq!(gray_sign_build_adds(0, 0, usize::MAX), 0);
    }

    /// `CU-11`: the production quotient/remainder addresses are byte-for-byte
    /// the former power-of-two bit projection for every boundary class the
    /// safe surface admits. The bit spelling lives only in this independent
    /// test oracle; the shipped q-reachable helpers contain no bit operation.
    #[test]
    fn portable_radix_addresses_match_retained_bit_oracle_cu_11() {
        for rows in [1usize, 2, 4, 8, 16] {
            let code_space = 16usize;
            let slab = code_space * rows;
            for offset in [
                0,
                1,
                rows - 1,
                rows,
                slab - 1,
                slab,
                slab + rows + 1,
                u32::MAX as usize,
                usize::MAX,
            ] {
                assert_eq!(
                    super::table_entry_address(offset, slab, rows),
                    offset & (slab - rows),
                    "row-scaled address offset={offset}, slab={slab}, rows={rows}"
                );
            }
            for code in [
                0,
                1,
                code_space - 1,
                code_space,
                code_space + 1,
                u8::MAX as usize,
                u16::MAX as usize,
                usize::MAX,
            ] {
                assert_eq!(
                    super::table_code_address(code, code_space, rows),
                    (code & (code_space - 1)) * rows,
                    "code address code={code}, space={code_space}, rows={rows}"
                );
            }
            assert_eq!(super::table_row_grade(rows), rows.trailing_zeros());
        }
    }
}

impl Lane<f32> for Scaled64 {
    #[inline]
    fn capacity(per_step: u128) -> Option<usize> {
        // Unlike an ordinary element bound, this contextual declaration is
        // already the exact one-product q coefficient bound. Squaring it here
        // would charge the product twice and collapse the zero-span capacity.
        // The data-free query supplies `u128::MAX` and therefore gets the total
        // one-product answer; the f32 plan itself remains cache-derived.
        if per_step == 0 {
            return None;
        }
        Some(
            usize::try_from(
                u128::from(crate::generated_capacity::f32_q::COMPACT_CEILING) / per_step,
            )
            .unwrap_or(usize::MAX)
            .max(1),
        )
    }

    #[inline(always)]
    fn mac(self, a: f32, w: f32) -> Self {
        // The two four-byte cells are q carriers produced in place by the
        // paired tabulation projection. Their significands meet only through
        // the signed-octet Atlas lookup; relative grade and sign are address
        // coordinates retained in the compact-or-tagged lane word.
        let product = f32_q_product(decode_f32_q_factor(a), decode_f32_q_factor(w));
        f32_q_add_words(self, product)
    }

    #[inline]
    fn place(self, acc: <f32 as Element>::Acc) -> <f32 as Element>::Acc {
        // The trait's exponent-free spelling is precisely placement at grade
        // zero. The common-grade table walk supplies its measured grade through
        // `place_scaled`.
        self.place_scaled(acc, 0)
    }

    #[inline]
    fn place_scaled(self, mut acc: <f32 as Element>::Acc, exponent: i32) -> <f32 as Element>::Acc {
        match decode_f32_q_token(self) {
            F32QToken::Compact(value) => {
                acc.add_scaled(value.unsigned_abs() as u128, exponent, value < 0);
            }
            F32QToken::Finite {
                magnitude,
                grade,
                negative,
            } => {
                let grade = i32::try_from(grade).unwrap_or(i32::MAX);
                acc.add_scaled(
                    u128::from(magnitude),
                    exponent.saturating_add(grade), // R3-ok: an exponent address, not an accumulation
                    negative,
                );
            }
            F32QToken::Nonfinite(flags) => {
                if f32_q_has_flag(flags, 0) {
                    acc.set_nan();
                }
                if f32_q_has_flag(flags, 1) {
                    acc.set_infinity(false);
                }
                if f32_q_has_flag(flags, 2) {
                    acc.set_infinity(true);
                }
            }
            // A split token is control for the scalar-fracture scheduler and
            // therefore has no finite Laurent value. Treating an externally
            // constructed or otherwise misplaced marker as NaN keeps this
            // contextual trait operation total without aliasing a coefficient.
            F32QToken::Split => acc.set_nan(),
        }
        acc
    }
}

/// Fill one slot of the table.
///
/// `T[c][i] = sum_{t < block} acts[t][i] * book[c][t]`, for `c < space` and
/// `i < rows`, written to `out[c * rows + i]`.
///
/// # Safety
///
/// - `book` has `space * block` readable elements, code-major.
/// - `acts` has `block * rows` readable elements, in the layout
///   [`crate::packed_slot`] gives for this spec's [`TableSpec::k_group`].
/// - `out` has `space * rows` writable lanes.
/// - `rows` is this spec's, and the host has the features its backend names.
pub type TableBuild<E, L> =
    unsafe fn(rows: usize, space: usize, block: usize, book: *const E, acts: *const E, out: *mut L);

/// Reduce one group of output columns over a run of slots.
///
/// `lane[u * rows + i] += sum_{slot < depth} stack[slot * slab +
/// radix_entry(off[slot * group + u]) + i]`.
///
/// Read-modify-write on `lane`, so a caller carries one accumulation across as
/// many calls as the lane's capacity admits and the exact accumulator never
/// appears in the reduction at all.
///
/// # `off` holds offsets, not indices
///
/// The caller multiplies the code's index by `rows` when it writes the stream,
/// because it is walking that stream anyway. That is the same discipline the
/// packed panel follows on the dense side --- the layout carries the address, so
/// the inner loop walks and never indexes. The portable spelling normalizes the
/// address by Euclidean remainder; target declarations may spell the same
/// finite address projection through their native lookup alphabet.
///
/// # Safety
///
/// - `stack` has `depth * slab` readable lanes and `slab` is nonzero.
/// - `off` has `depth * group` readable words.
/// - `lane` has `group * rows` readable and writable lanes.
/// - Euclidean address projection discharges the bound: every read is in-slab
///   whatever the offset holds. Correctness of the *value* is
///   [`uor_matmul_codec::Enumerable`]'s law and `CK-09` asserts it; safety of
///   the *read* is this projection and holds unconditionally.
/// - `rows` and `group` are this spec's, and the host has the features its
///   backend names.
pub type TableGather<L> = unsafe fn(
    rows: usize,
    group: usize,
    depth: usize,
    slab: usize,
    stack: *const L,
    off: *const u32,
    lane: *mut L,
);

/// The same reduction, reading the coded operand's own memory.
///
/// `lane[u * rows + i] += sum_{slot < depth} stack[slot * slab +
/// radix_code(codes[u * stride + slot]) * rows + i]`.
///
/// # Why there are two of these and not one
///
/// [`TableGather`] takes an index stream the driver builds. Building one costs
/// `4 / (rows * block)` bytes per product, which at the widest tile is a
/// thirty-second of the entry traffic and invisible, and at a one-row tile is
/// exactly as wide as the entry it addresses --- measured, two thirds of the
/// work. When the codec answers
/// [`uor_matmul_codec::Enumerable::as_index_stream`], the operand's own memory
/// *is* that stream and there is nothing to build.
///
/// That is the same rule [`uor_matmul_core::MatView::row_block`] follows on the
/// dense side --- borrow when the layout already holds what is wanted, copy
/// otherwise --- and the two produce the same lane words, which is half of what
/// `CB-08` asserts.
///
/// The retained grade argument belongs to the API-locked native function
/// protocol. The portable spelling derives its code radix directly from
/// `slab / rows`; no packed field extraction participates in the address.
///
/// # Safety
///
/// - `stack` has `depth * slab` readable lanes and `slab / rows` is the code
///   radix.
/// - `codes` has `(group - 1) * stride + depth` readable words.
/// - `lane` has `group * rows` readable and writable lanes.
/// - Euclidean code projection discharges the bound: every read is in-slab
///   whatever the code holds. That the entry is the *right* one is
///   `Enumerable::as_index_stream`'s claim, which `CK-09` asserts.
/// - `rows`, its retained radix grade, and `group` are this spec's, and the host
///   has the features its backend names.
pub type TableGatherCodes<L> = unsafe fn(
    rows: usize,
    group: usize,
    depth: usize,
    slab: usize,
    shift: u32,
    stack: *const L,
    codes: *const u16,
    stride: usize,
    lane: *mut L,
);

/// [`TableGatherCodes`] at a byte-wide code stream.
///
/// The same contract, the same Euclidean radix projection, the same slab
/// reads: the code is widened to the index on load, and the two widths are two monomorphic
/// sequences the driver dispatches between once per arm, never per code. A
/// codec whose code space fits a byte stores half the codes the `u16` spelling
/// does; nothing about the gather's arithmetic moves.
///
/// # Safety
///
/// As [`TableGatherCodes`].
pub type TableGatherCodesU8<L> = unsafe fn(
    rows: usize,
    group: usize,
    depth: usize,
    slab: usize,
    shift: u32,
    stack: *const L,
    codes: *const u8,
    stride: usize,
    lane: *mut L,
);

/// One backend's table sequences for one element family, and the shape they
/// want their operands in.
///
/// A value, exactly like [`crate::KernelSpec`]: adding an ISA adds one and
/// touches no driver code.
pub struct TableSpec<E, L> {
    /// Which backend this is.
    pub backend: Backend,
    /// Rows of `C` one table entry serves.
    pub rows: usize,
    /// Output columns one [`Self::gather`] call reduces at once.
    pub group: usize,
    /// How the activation tile arranges the block's depth, as
    /// [`crate::packed_slot`]'s `group`.
    ///
    /// One for the reference. Two for a sequence that folds a pair of block
    /// steps into each lane with one instruction, which is what `madd` does.
    pub k_group: usize,
    /// Lane words one issued add covers in [`Self::gather`].
    ///
    /// Eight for a 256-bit register of `i32`. One for the reference. The
    /// products one add covers is this times the codec's block, which is why a
    /// longer codeword is a denser instruction and not merely a smaller
    /// operand.
    pub lanes_per_add: usize,
    /// Products one issued instruction covers in [`Self::build`].
    pub build_products_per_step: usize,
    /// The largest magnitude one lane holds. `u128::MAX` for the exact lane.
    pub lane_cap: u128,
    /// The widest alphabet bound at which this sequence is exact.
    ///
    /// The same declaration [`crate::KernelSpec::max_bound`] carries and for
    /// the same reason: a sequence with an intermediate narrower than its lane
    /// is wrong past a magnitude however shallow the chunk.
    pub max_bound: u128,
    /// Whether [`Self::build`] issues multiplies.
    ///
    /// Product builds normally do. Bound-1 and finite-alphabet lookup builds
    /// declare `false`: the former issues adds and subtracts, while the latter
    /// reads a product and adds it. The driver's census charges the operation
    /// actually issued (`CB-10`) rather than the mathematical product it
    /// computes.
    pub build_multiplies: bool,
    /// The add/contraction charge [`Self::build`] exposes per slot, recorded
    /// when [`Self::build_multiplies`] is false.
    ///
    /// Transparent fixed bodies count their actual adds: the independent
    /// bound-one build reports every signed add, the Gray walk reports its own
    /// recurrence, and compact Atlas carriers report their fixed expansion.
    /// A generic `Element::mac` body is an opaque algebra boundary and reports
    /// one contraction presentation per product; a shape-only callback cannot
    /// truthfully invent its data-dependent internal additions.
    pub build_adds: fn(space: usize, block: usize, rows: usize) -> u64,
    /// Fill one slot.
    pub build: TableBuild<E, L>,
    /// Reduce one column group from an index stream the driver built.
    pub gather: TableGather<L>,
    /// Reduce one column group from the operand's own `u16` code stream.
    pub gather_codes: TableGatherCodes<L>,
    /// [`Self::gather_codes`] at a byte-wide code stream: the same lane words,
    /// the operand stored at half the width.
    pub gather_codes_u8: TableGatherCodesU8<L>,
}

impl<E, L> Clone for TableSpec<E, L> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<E, L> Copy for TableSpec<E, L> {}

impl<E, L> core::fmt::Debug for TableSpec<E, L> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TableSpec")
            .field("backend", &self.backend)
            .field("rows", &self.rows)
            .field("group", &self.group)
            .field("k_group", &self.k_group)
            .field("lanes_per_add", &self.lanes_per_add)
            .finish_non_exhaustive()
    }
}

/// The reference sequences: the model transcribed.
///
/// Not a fallback. Every other sequence in this module is a factorization of
/// *these* accumulations into wider instructions, and the parity tests pin each
/// one to this (R6).
///
/// Generic over the element and the lane, so there is one reference for every
/// family rather than one per family --- including the families with no narrow
/// register at all, where this is the complete sequence the hardware offers.
pub const fn portable_table<E: Element, L: Lane<E>>(rows: usize, group: usize) -> TableSpec<E, L> {
    // The staged gather holds one column group of lane words in a compile-time
    // array so the run's adds stay in registers. That pays exactly when the
    // words fit a register file: sixteen of them at sixteen bytes is every
    // 128-bit vector register the baseline targets have. A wider word --- the
    // exact accumulator of a float family is 88 or 544 bytes --- cannot be
    // staged, and trying is a frame the size of the group per dispatched tile
    // of pure copy, which an unoptimized build sums over every arm of the
    // dispatcher. It accumulates where it lies instead: the same reads, the
    // same adds, the same lane words (`CB-08`).
    let (gather, gather_codes, gather_codes_u8): (
        TableGather<L>,
        TableGatherCodes<L>,
        TableGatherCodesU8<L>,
    ) = if core::mem::size_of::<L>() * 16 <= 256 {
        (
            portable_gather::<L>,
            portable_gather_codes::<L, u16>,
            portable_gather_codes::<L, u8>,
        )
    } else {
        (
            portable_gather_wide::<L>,
            portable_gather_codes_wide::<L, u16>,
            portable_gather_codes_wide::<L, u8>,
        )
    };
    TableSpec {
        backend: Backend::Portable,
        rows,
        group,
        // The reference needs no grouping, which is why it is the one sequence
        // that never has a tail.
        k_group: 1,
        lanes_per_add: 1,
        build_products_per_step: 1,
        lane_cap: u128::MAX,
        // The reference multiplies in the lane's own width, so there is no
        // intermediate to outgrow and no alphabet it is inexact on.
        max_bound: u128::MAX,
        build_multiplies: true,
        build_adds: product_build_adds,
        build: portable_build::<E, L>,
        gather,
        gather_codes,
        gather_codes_u8,
    }
}

/// The full-alphabet i8 table build using the static product lookup.
///
/// The gather is the same portable gather as [`portable_table`]; only the
/// build's product operation is factored into a read from [`I8_PRODUCTS`].
pub const fn portable_table_i8_lookup(rows: usize, group: usize) -> TableSpec<i8, i32> {
    let mut spec = portable_table::<i8, i32>(rows, group);
    spec.build_multiplies = false;
    spec.build = portable_build_lookup_i8;
    spec
}

/// # Safety
///
/// [`TableBuild`]'s contract.
pub(crate) unsafe fn portable_build_lookup_i8(
    rows: usize,
    space: usize,
    block: usize,
    book: *const i8,
    acts: *const i8,
    out: *mut i32,
) {
    // SAFETY: the caller guaranteed the three extents.
    let (book, acts, out) = unsafe {
        (
            core::slice::from_raw_parts(book, space * block),
            core::slice::from_raw_parts(acts, block * rows),
            core::slice::from_raw_parts_mut(out, space * rows),
        )
    };
    match rows {
        1 => build_run_lookup_i8::<1>(block, book, acts, out),
        2 => build_run_lookup_i8::<2>(block, book, acts, out),
        4 => build_run_lookup_i8::<4>(block, book, acts, out),
        8 => build_run_lookup_i8::<8>(block, book, acts, out),
        16 => build_run_lookup_i8::<16>(block, book, acts, out),
        _ => build_any_lookup_i8(rows, block, book, acts, out),
    }
}

/// Build an i8 table from an activation panel with a fixed k-group.
#[cfg(target_arch = "wasm32")]
unsafe fn packed_build_lookup_i8<const GROUP: usize>(
    rows: usize,
    space: usize,
    block: usize,
    book: *const i8,
    acts: *const i8,
    out: *mut i32,
) {
    // SAFETY: the caller supplied the extents in the TableBuild contract.
    let (book, acts, out) = unsafe {
        (
            core::slice::from_raw_parts(book, space * block),
            core::slice::from_raw_parts(acts, rows * block),
            core::slice::from_raw_parts_mut(out, space * rows),
        )
    };
    for (entry, word) in out.chunks_exact_mut(rows).zip(book.chunks_exact(block)) {
        for (row, cell) in entry.iter_mut().enumerate() {
            let mut sum = 0i32;
            for (t, &weight) in word.iter().enumerate() {
                let activation = acts[crate::spec::packed_slot(t, row, rows, GROUP)];
                sum = sum.wrapping_add(crate::lookup::i8_product(activation, weight));
            }
            *cell = sum;
        }
    }
}

/// Lookup build for a plain k-major activation panel.
#[cfg(target_arch = "wasm32")]
pub(crate) unsafe fn packed_build_lookup_i8_group1(
    rows: usize,
    space: usize,
    block: usize,
    book: *const i8,
    acts: *const i8,
    out: *mut i32,
) {
    // SAFETY: forwarded from the TableBuild caller.
    unsafe { packed_build_lookup_i8::<1>(rows, space, block, book, acts, out) }
}

#[inline(always)]
fn build_run_lookup_i8<const R: usize>(block: usize, book: &[i8], acts: &[i8], out: &mut [i32]) {
    for (entry, word) in out.chunks_exact_mut(R).zip(book.chunks_exact(block)) {
        let mut acc = [0i32; R];
        for (&w, col) in word.iter().zip(acts.chunks_exact(R)) {
            for (cell, &a) in acc.iter_mut().zip(&col[..R]) {
                *cell = cell.wrapping_add(crate::lookup::i8_product(a, w));
            }
        }
        entry.copy_from_slice(&acc);
    }
}

fn build_any_lookup_i8(rows: usize, block: usize, book: &[i8], acts: &[i8], out: &mut [i32]) {
    for (entry, word) in out.chunks_exact_mut(rows).zip(book.chunks_exact(block)) {
        entry.fill(0);
        for (&w, col) in word.iter().zip(acts.chunks_exact(rows)) {
            for (cell, &a) in entry.iter_mut().zip(col) {
                *cell = cell.wrapping_add(crate::lookup::i8_product(a, w));
            }
        }
    }
}

const fn census_factor(value: usize) -> u64 {
    if value as u128 > u64::MAX as u128 {
        u64::MAX
    } else {
        value as u64
    }
}

const fn census_product3(a: usize, b: usize, c: usize) -> u64 {
    census_factor(a)
        .saturating_mul(census_factor(b)) // R3-ok: a bounded diagnostic counter
        .saturating_mul(census_factor(c)) // R3-ok: a bounded diagnostic counter
}

/// The per-codeword build's observable census charge: every product, either
/// issued transparently as an add/subtract (`CB-10`) or presented once to an
/// opaque `Element::mac` contraction boundary.
pub const fn product_build_adds(space: usize, block: usize, rows: usize) -> u64 {
    census_product3(space, block, rows) // R3-ok: a bounded diagnostic counter
}

/// The Gray walk's census charge: `T[0]` and the doubled activations (two
/// passes of `block` adds), then one update per code past the first ---
/// `2 * block + space - 1` adds per row against the independent build's
/// `space * block`.
pub const fn gray_sign_build_adds(space: usize, block: usize, rows: usize) -> u64 {
    census_factor(block)
        .saturating_mul(2) // R3-ok: a bounded diagnostic counter
        .saturating_add(census_factor(space)) // R3-ok: a bounded diagnostic counter
        .saturating_sub(1) // R3-ok: a bounded diagnostic counter
        .saturating_mul(census_factor(rows)) // R3-ok: a bounded diagnostic counter
}

/// The bound-1 build: the table's only multiply, absent.
///
/// At bound 1 every book word is in `{-1, 0, +1}`, so `T[c][i] = sum_t
/// +-A[i][t]` is adds and subtracts and the multiply [`portable_build`] issues
/// has nothing left to do. Listed after every full-alphabet sequence, so
/// [`choose_table`] hands it exactly the alphabet it declares and no other ---
/// selection by declaration, the same rule the `madd` pair follows on the
/// dense side (R13).
///
/// Only the build differs; the gathers are bound-independent and are the
/// reference's own, shared rather than duplicated.
///
/// Concrete over `i8` rather than generic like [`portable_table`]: the
/// bound-1 spelling is the sign tier's, which `CK-13` declared over `i8`, and
/// a generic negation is not an operation [`Element`] names.
pub const fn portable_table_bound1(rows: usize, group: usize) -> TableSpec<i8, i32> {
    TableSpec {
        backend: Backend::Portable,
        rows,
        group,
        k_group: 1,
        lanes_per_add: 1,
        build_products_per_step: 1,
        lane_cap: u128::MAX,
        // Exact exactly when every book word is in `{-1, 0, +1}`: the whole of
        // what the sequence assumes, stated as the declaration `choose_table`
        // reads.
        max_bound: 1,
        build_multiplies: false,
        build_adds: product_build_adds,
        build: portable_build_bound1,
        gather: portable_gather::<i32>,
        gather_codes: portable_gather_codes::<i32, u16>,
        gather_codes_u8: portable_gather_codes::<i32, u8>,
    }
}

/// # Safety
///
/// [`TableBuild`]'s contract, with every element of `book` in `{-1, 0, +1}` ---
/// which the caller's bound-1 alphabet has already established at the boundary.
unsafe fn portable_build_bound1(
    rows: usize,
    space: usize,
    block: usize,
    book: *const i8,
    acts: *const i8,
    out: *mut i32,
) {
    // SAFETY: the caller guaranteed the three extents, as in `portable_build`.
    let (book, acts, out) = unsafe {
        (
            core::slice::from_raw_parts(book, space * block),
            core::slice::from_raw_parts(acts, block * rows),
            core::slice::from_raw_parts_mut(out, space * rows),
        )
    };
    // The same compile-time tile heights as the reference build, for the same
    // reason: one entry's words live in registers for its whole codeword.
    match rows {
        1 => build_run_bound1::<1>(block, book, acts, out),
        2 => build_run_bound1::<2>(block, book, acts, out),
        4 => build_run_bound1::<4>(block, book, acts, out),
        8 => build_run_bound1::<8>(block, book, acts, out),
        16 => build_run_bound1::<16>(block, book, acts, out),
        _ => build_any_bound1(rows, block, book, acts, out),
    }
}

/// The bound-1 build at a compile-time tile height: [`build_run`]'s loop nest,
/// with the product read as an add, a subtract, or nothing.
#[inline(always)]
fn build_run_bound1<const R: usize>(block: usize, book: &[i8], acts: &[i8], out: &mut [i32]) {
    for (entry, word) in out.chunks_exact_mut(R).zip(book.chunks_exact(block)) {
        let mut acc = [0i32; R];
        for (&w, col) in word.iter().zip(acts.chunks_exact(R)) {
            for (cell, &a) in acc.iter_mut().zip(&col[..R]) {
                // `w` is in `{-1, 0, +1}`: the product is the activation, its
                // negation, or zero, and there is nothing to multiply.
                let a = i32::from(a);
                *cell += if w == 1 {
                    a
                } else if w == -1 {
                    -a
                } else {
                    0
                };
            }
        }
        entry.copy_from_slice(&acc);
    }
}

/// The same, at a row count no shipped tile uses.
fn build_any_bound1(rows: usize, block: usize, book: &[i8], acts: &[i8], out: &mut [i32]) {
    for (entry, word) in out.chunks_exact_mut(rows).zip(book.chunks_exact(block)) {
        entry.fill(0);
        for (&w, col) in word.iter().zip(acts.chunks_exact(rows)) {
            for (cell, &a) in entry.iter_mut().zip(col) {
                let a = i32::from(a);
                *cell += if w == 1 {
                    a
                } else if w == -1 {
                    -a
                } else {
                    0
                };
            }
        }
    }
}

/// The Gray-walk sign build: the same bound-1 table, built by walking the
/// code space in reflected Gray-code order.
///
/// For the sign codebook `T[c][i] = sum_t (2 * bit(c, t) - 1) * A[i][t]`, and
/// consecutive Gray codes differ in exactly one bit `q`, so `T[next]` is
/// `T[cur] +- 2 * A[q]` --- one add per row per code, against the independent
/// build's `block` adds per row per code. At `Sign<8>` that is
/// `2 * 8 + 255 = 271` adds a row against 2048. The doubled activation is
/// written as `a + a`, so the multiply count stays zero, and every table
/// entry is stored exactly once, at the binary code index rather than the
/// walk ordinal: the 256 stores are the floor both builds share, and the win
/// is the arithmetic between them.
///
/// The precondition is the codebook: `space` a power of two and
/// `book[c * block + t] == 2 * bit(c, t) - 1`. The walk derives the signs
/// from the code index and reads no book at all, so an ordinary bound-1 book
/// --- `Ternary`'s, say --- would come out wrong: the driver reaches this
/// build only where the codec declares [`uor_matmul_codec::Enumerable::
/// SIGN_BIT_BOOK`], which is what makes ignoring the book a factorization of
/// the same table rather than a different table.
///
/// There are no ISA variants, deliberately: the walk is a serial dependency
/// chain (`T[next]` needs `T[cur]`), and a vector sequence would widen the
/// one part both builds already do at full width --- the stores --- while
/// serializing nothing away. The win is arithmetic per store, and a chain
/// does not vectorize.
///
/// # Safety
///
/// [`TableBuild`]'s contract, with the sign-codebook precondition above.
unsafe fn gray_sign_build(
    rows: usize,
    space: usize,
    block: usize,
    book: *const i8,
    acts: *const i8,
    out: *mut i32,
) {
    let _ = book;
    // SAFETY: the caller guaranteed the three extents, as in `portable_build`.
    let (acts, out) = unsafe {
        (
            core::slice::from_raw_parts(acts, block * rows),
            core::slice::from_raw_parts_mut(out, space * rows),
        )
    };
    match rows {
        1 => gray_sign_run::<1>(space, block, acts, out),
        2 => gray_sign_run::<2>(space, block, acts, out),
        4 => gray_sign_run::<4>(space, block, acts, out),
        8 => gray_sign_run::<8>(space, block, acts, out),
        16 => gray_sign_run::<16>(space, block, acts, out),
        _ => gray_sign_any(rows, space, block, acts, out),
    }
}

/// The Gray build at a compile-time tile height: the current entry lives in
/// registers for the whole walk.
#[inline(always)]
fn gray_sign_run<const R: usize>(space: usize, block: usize, acts: &[i8], out: &mut [i32]) {
    let walk = space.trailing_zeros() as usize;
    debug_assert!(
        space.is_power_of_two() && walk <= block && walk <= 16,
        "the sign codebook is a power-of-two space read bit by bit"
    );
    // `T[0] = -sum_t A[t]` and the doubled activations, once: every update
    // below is then one add per row. Terms past `walk` are a constant `-1`
    // contribution and are in `T[0]` already.
    let mut cur = [0i32; R];
    let mut doubled = [[0i32; R]; 16];
    for t in 0..block {
        for (cell, &a) in cur.iter_mut().zip(&acts[t * R..][..R]) {
            *cell -= i32::from(a);
        }
        if t < walk {
            for (d, &a) in doubled[t].iter_mut().zip(&acts[t * R..][..R]) {
                let a = i32::from(a);
                // Written as the add it is: the multiply count is zero here
                // exactly as it is in the per-codeword build (`CB-10`).
                *d = a + a;
            }
        }
    }
    out[..R].copy_from_slice(&cur);
    // The reflected walk: the bit that flips at `step` is
    // `step.trailing_zeros()`, read off `gray ^ prev` because that is the fact
    // being used, and `cur` is stored at the binary index `gray`.
    let mut prev = 0usize;
    for step in 1..space {
        let gray = step ^ (step >> 1);
        let q = (gray ^ prev).trailing_zeros() as usize;
        if (gray >> q) & 1 == 1 {
            for (cell, &d) in cur.iter_mut().zip(&doubled[q]) {
                *cell += d;
            }
        } else {
            for (cell, &d) in cur.iter_mut().zip(&doubled[q]) {
                *cell -= d;
            }
        }
        out[gray * R..][..R].copy_from_slice(&cur);
        prev = gray;
    }
}

/// The same, at a row count no shipped tile uses: `cur` capped at the tile
/// lane bound every kernel in this crate already asserts against.
fn gray_sign_any(rows: usize, space: usize, block: usize, acts: &[i8], out: &mut [i32]) {
    let walk = space.trailing_zeros() as usize;
    assert!(
        space.is_power_of_two() && walk <= block && walk <= 16 && rows <= MAX_TILE_LANES,
        "the Gray build's register window is the tile lane bound"
    );
    let mut cur = [0i32; MAX_TILE_LANES];
    let mut doubled = [0i32; 16 * MAX_TILE_LANES];
    for t in 0..block {
        for (cell, &a) in cur[..rows].iter_mut().zip(&acts[t * rows..][..rows]) {
            *cell -= i32::from(a);
        }
        if t < walk {
            for (d, &a) in doubled[t * rows..][..rows]
                .iter_mut()
                .zip(&acts[t * rows..][..rows])
            {
                let a = i32::from(a);
                *d = a + a;
            }
        }
    }
    out[..rows].copy_from_slice(&cur[..rows]);
    let mut prev = 0usize;
    for step in 1..space {
        let gray = step ^ (step >> 1);
        let q = (gray ^ prev).trailing_zeros() as usize;
        let d = &doubled[q * rows..][..rows];
        if (gray >> q) & 1 == 1 {
            for (cell, &d) in cur[..rows].iter_mut().zip(d) {
                *cell += d;
            }
        } else {
            for (cell, &d) in cur[..rows].iter_mut().zip(d) {
                *cell -= d;
            }
        }
        out[gray * rows..][..rows].copy_from_slice(&cur[..rows]);
        prev = gray;
    }
}

/// The Gray-walk sign build, as a spec: [`gray_sign_build`] behind the
/// bound-1 declaration, so selection by declaration reaches it exactly where
/// the codec also declares the sign codebook.
///
/// The spec is the bound-1 spec this host would otherwise select, with only
/// the build swapped: the walk replaces the per-codeword sums, and the
/// gathers are the incumbent's own. That is what keeps the comparison honest
/// --- the build and the gather are separate concerns, and timing the walk
/// against a portable gather would be timing the gather, not the build. The
/// walk itself is a serial dependency chain, so it has no ISA variant: the
/// one part both builds already do at full width is the stores.
///
/// **Selection offers this nowhere on its own**: it is not in
/// [`available_table_i8`], because the bound-1 declaration alone does not
/// imply the sign codebook --- `Ternary` at bound 1 declares the same bound
/// and its book is not the bit decomposition. The driver takes this spec only
/// where [`uor_matmul_codec::Enumerable::SIGN_BIT_BOOK`] says the book *is*
/// the bit decomposition; everywhere else the per-codeword adds build runs,
/// at the same bytes. The pre-registered measurement verdict was to ship: the
/// isolated build won at both measured widths and the end-to-end run did not
/// regress; `MEASUREMENT-LOG.md` records the figures and byte checks.
pub fn gray_sign_table(rows: usize, group: usize) -> TableSpec<i8, i32> {
    // The incumbent: the spec `Auto` hands a bound-1 alphabet. `block = 1`
    // because only the build's `k_group` divides one, and the bound-1 builds
    // are exactly the `k_group: 1` sequences.
    let incumbent = choose_table(available_table_i8(rows, group), Backend::Auto, 1, 1)
        .expect("the bound-1 build is always present");
    TableSpec {
        build: gray_sign_build,
        build_adds: gray_sign_build_adds,
        ..incumbent
    }
}

/// # Safety
///
/// [`TableBuild`]'s contract.
unsafe fn portable_build<E: Element, L: Lane<E>>(
    rows: usize,
    space: usize,
    block: usize,
    book: *const E,
    acts: *const E,
    out: *mut L,
) {
    // SAFETY: the caller guaranteed the three extents. Turning them into slices
    // once, here, is what lets the loop below be safe indexing rather than
    // three raw reads per product.
    let (book, acts, out) = unsafe {
        (
            core::slice::from_raw_parts(book, space * block),
            core::slice::from_raw_parts(acts, block * rows),
            core::slice::from_raw_parts_mut(out, space * rows),
        )
    };
    // The tile heights the traversal walks, each a compile-time constant, for
    // the same reason the gather's are: with `rows` runtime the entry is a slice
    // of unknown length and its accumulation is a chunked iterator around what
    // should be `rows` registers. Measured at a one-row tile that framing was
    // half the traversal.
    match rows {
        1 => build_run::<1, E, L>(block, book, acts, out),
        2 => build_run::<2, E, L>(block, book, acts, out),
        4 => build_run::<4, E, L>(block, book, acts, out),
        8 => build_run::<8, E, L>(block, book, acts, out),
        16 => build_run::<16, E, L>(block, book, acts, out),
        _ => build_any(rows, block, book, acts, out),
    }
}

/// The reference build at a compile-time tile height.
///
/// The code space outer and the block inner, so one entry's `R` words are live
/// in registers for its whole codeword and are written **once**. The other order
/// reads and writes every entry once per element of the block, which is `block`
/// times the traffic and measured half the traversal.
#[inline(always)]
fn build_run<const R: usize, E: Element, L: Lane<E>>(
    block: usize,
    book: &[E],
    acts: &[E],
    out: &mut [L],
) {
    for (entry, word) in out.chunks_exact_mut(R).zip(book.chunks_exact(block)) {
        let mut acc = [L::ZERO; R];
        for (&w, col) in word.iter().zip(acts.chunks_exact(R)) {
            for (cell, &a) in acc.iter_mut().zip(&col[..R]) {
                *cell = cell.mac(a, w);
            }
        }
        entry.copy_from_slice(&acc);
    }
}

/// The same, at a row count no shipped tile uses.
fn build_any<E: Element, L: Lane<E>>(
    rows: usize,
    block: usize,
    book: &[E],
    acts: &[E],
    out: &mut [L],
) {
    for (entry, word) in out.chunks_exact_mut(rows).zip(book.chunks_exact(block)) {
        entry.fill(L::ZERO);
        for (&w, col) in word.iter().zip(acts.chunks_exact(rows)) {
            for (cell, &a) in entry.iter_mut().zip(col) {
                *cell = cell.mac(a, w);
            }
        }
    }
}

/// # Safety
///
/// [`TableGather`]'s contract.
unsafe fn portable_gather<L: LaneWord>(
    rows: usize,
    group: usize,
    depth: usize,
    slab: usize,
    stack: *const L,
    off: *const u32,
    lane: *mut L,
) {
    // SAFETY: the caller guaranteed the three extents.
    let (stack, off, lane) = unsafe {
        (
            core::slice::from_raw_parts(stack, depth * slab),
            core::slice::from_raw_parts(off, depth * group),
            core::slice::from_raw_parts_mut(lane, group * rows),
        )
    };
    // The tile heights the traversal walks, each a compile-time constant so the
    // step's addressing is a walk. The last arm keeps the sequence total for a
    // height no caller reaches, which is what stops this being a ceiling (R8).
    dispatch_run!(
        rows,
        group,
        gather_any(rows, group, slab, stack, off, lane),
        |R, G| dispatch_slab!(code_words(slab, R), |C| gather_run::<C, R, G, L>(
            slab, stack, off, lane
        ))
    )
}

/// # Safety
///
/// [`TableGatherCodes`]'s contract at `K = u16`, [`TableGatherCodesU8`]'s at
/// `K = u8`. One body at two monomorphizations: the code widens to the index
/// on load, and nothing else about the walk knows the width.
#[allow(clippy::too_many_arguments)]
unsafe fn portable_gather_codes<L: LaneWord, K: Copy + Into<usize>>(
    rows: usize,
    group: usize,
    depth: usize,
    slab: usize,
    shift: u32,
    stack: *const L,
    codes: *const K,
    stride: usize,
    lane: *mut L,
) {
    // SAFETY: the caller guaranteed the three extents.
    let (stack, codes, lane) = unsafe {
        (
            core::slice::from_raw_parts(stack, depth * slab),
            core::slice::from_raw_parts(codes, (group - 1) * stride + depth),
            core::slice::from_raw_parts_mut(lane, group * rows),
        )
    };
    dispatch_run!(
        rows,
        group,
        codes_any(rows, depth, slab, shift, stack, codes, stride, lane),
        |R, G| dispatch_slab!(code_words(slab, R), |C| codes_run::<C, R, G, L, K>(
            depth, slab, shift, stack, codes, stride, lane
        ))
    )
}

/// # Safety
///
/// [`TableGather`]'s contract.
unsafe fn portable_gather_wide<L: LaneWord>(
    rows: usize,
    group: usize,
    depth: usize,
    slab: usize,
    stack: *const L,
    off: *const u32,
    lane: *mut L,
) {
    // SAFETY: the caller guaranteed the three extents.
    let (stack, off, lane) = unsafe {
        (
            core::slice::from_raw_parts(stack, depth * slab),
            core::slice::from_raw_parts(off, depth * group),
            core::slice::from_raw_parts_mut(lane, group * rows),
        )
    };
    // No dispatch and no staging: the tile heights exist to name register
    // counts, and a lane this wide has none.
    gather_any(rows, group, slab, stack, off, lane)
}

/// # Safety
///
/// [`TableGatherCodes`]'s contract at `K = u16`, [`TableGatherCodesU8`]'s at
/// `K = u8`.
#[allow(clippy::too_many_arguments)]
unsafe fn portable_gather_codes_wide<L: LaneWord, K: Copy + Into<usize>>(
    rows: usize,
    group: usize,
    depth: usize,
    slab: usize,
    shift: u32,
    stack: *const L,
    codes: *const K,
    stride: usize,
    lane: *mut L,
) {
    // SAFETY: the caller guaranteed the three extents.
    let (stack, codes, lane) = unsafe {
        (
            core::slice::from_raw_parts(stack, depth * slab),
            core::slice::from_raw_parts(codes, (group - 1) * stride + depth),
            core::slice::from_raw_parts_mut(lane, group * rows),
        )
    };
    // As [`portable_gather_wide`]: the same accumulation, walked, with nothing
    // staged.
    codes_any(rows, depth, slab, shift, stack, codes, stride, lane)
}

/// The same, at a row count no shipped tile uses.
#[allow(clippy::too_many_arguments)]
fn codes_any<L: LaneWord, K: Copy + Into<usize>>(
    rows: usize,
    depth: usize,
    slab: usize,
    _shift: u32,
    stack: &[L],
    codes: &[K],
    stride: usize,
    lane: &mut [L],
) {
    let code_space = slab / rows;
    let mut rest = stack;
    for slot in 0..depth {
        let (words, tail) = rest.split_at(slab);
        rest = tail;
        let mut at = slot;
        for cols in lane.chunks_exact_mut(rows) {
            let entry = &words[table_code_address(codes[at].into(), code_space, rows)..];
            for (cell, &e) in cols.iter_mut().zip(&entry[..rows]) {
                *cell = cell.add(e);
            }
            at += stride;
        }
    }
}

/// Normalize an arbitrary row-scaled offset into one complete table entry.
/// The safe gather surface remains total for every offset, while Euclidean
/// radix projection replaces the former packed mask.
#[inline(always)]
fn table_entry_address(offset: usize, slab: usize, rows: usize) -> usize {
    let within = offset % slab;
    within - within % rows
}

/// Project one canonical code into the row-major table alphabet. The product
/// here is structural address scaling, not an element-value product.
#[inline(always)]
fn table_code_address(code: usize, code_space: usize, rows: usize) -> usize {
    (code % code_space) * rows
}

/// Radix-two grade of a power-of-two tile height, expressed by quotient
/// refinement so the public function-pointer protocol does not force bit-field
/// extraction onto the portable q path.
#[inline(always)]
fn table_row_grade(mut rows: usize) -> u32 {
    let mut grade = 0u32;
    while rows > 1 {
        rows /= 2;
        grade += 1;
    }
    grade
}

/// The reference column step: the whole of what `CU-06` reads.
///
/// Everything is *walked*, and that is the claim. The slot's base advances by an
/// add, the entry's address is a Euclidean projection of an offset the caller
/// already scaled, and the accumulation is a compile-time array.
///
/// `R` and `G` are both compile-time, and `G` is the reason this is not simply a
/// slice. With the column group a runtime value the accumulation is a chunked
/// iterator over the caller's buffer, so every lane word is loaded and stored
/// once per slot; as `[[L; R]; G]` it is `R * G` registers loaded once and stored
/// once for the whole run. Measured at a one-row tile and a group of sixteen,
/// that is 32 memory operations per slot that do not happen, and it is the
/// difference between 5.9 and 9.6 Gmac/s on `1x1024x4096`.
///
/// `C` is the slab's code count when the caller knew it and zero when it did
/// not, and it is the only place the codec reaches this loop. At a nonzero `C`
/// the slab is `C * R` and every slot's base is a constant displacement, so the
/// cursor below folds away entirely; at zero it is the argument. One body, one
/// identity, the value bound wherever it is known --- not two sequences (R13).
#[inline(always)]
fn gather_run<const C: usize, const R: usize, const G: usize, L: LaneWord>(
    slab_arg: usize,
    stack: &[L],
    off: &[u32],
    lane: &mut [L],
) {
    let slab = if C == 0 { slab_arg } else { C * R };
    let mut acc = [[L::ZERO; R]; G];
    for (cols, held) in acc.iter_mut().zip(lane.chunks_exact(R)) {
        cols.copy_from_slice(held);
    }
    // A cursor, not `chunks_exact`. The chunk length is a runtime value, and
    // measured on the emitted assembly the iterator re-derives each slot's base
    // as `slot * slab` --- a multiply per code, which is precisely what this
    // traversal exists to not issue. `split_at` advances by an add.
    let mut rest = stack;
    for run in off.chunks_exact(G) {
        let (words, tail) = rest.split_at(slab);
        rest = tail;
        for (cols, &at) in acc.iter_mut().zip(run) {
            let entry = &words[table_entry_address(at as usize, slab, R)..];
            for (cell, &e) in cols.iter_mut().zip(&entry[..R]) {
                *cell = cell.add(e);
            }
        }
    }
    for (cols, held) in acc.iter().zip(lane.chunks_exact_mut(R)) {
        held.copy_from_slice(cols);
    }
}

/// The same, at a `(rows, group)` pair no shipped tile uses.
///
/// Present so the sequence is total for every tile and not only for the ones the
/// driver walks (R8). It computes the same integer and issues the address
/// arithmetic [`gather_run`] does not need.
fn gather_any<L: LaneWord>(
    rows: usize,
    group: usize,
    slab: usize,
    stack: &[L],
    off: &[u32],
    lane: &mut [L],
) {
    let mut rest = stack;
    for run in off.chunks_exact(group) {
        let (words, tail) = rest.split_at(slab);
        rest = tail;
        for (cols, &at) in lane.chunks_exact_mut(rows).zip(run) {
            let entry = &words[table_entry_address(at as usize, slab, rows)..];
            for (cell, &e) in cols.iter_mut().zip(&entry[..rows]) {
                *cell = cell.add(e);
            }
        }
    }
}

/// The same, over the coded operand's own memory.
///
/// There is no index stream to write and none to read back.
///
/// `C` binds as it does in [`gather_run`]. The retained grade argument belongs
/// to the API-locked native protocol; this portable spelling derives the entry
/// address directly as the canonical code remainder scaled by the structural
/// row extent.
#[inline(always)]
fn codes_run<const C: usize, const R: usize, const G: usize, L: LaneWord, K: Copy + Into<usize>>(
    depth: usize,
    slab_arg: usize,
    _shift_arg: u32,
    stack: &[L],
    codes: &[K],
    stride: usize,
    lane: &mut [L],
) {
    let (slab, code_space) = if C == 0 {
        (slab_arg, slab_arg / R)
    } else {
        (C * R, C)
    };
    let mut acc = [[L::ZERO; R]; G];
    for (cols, held) in acc.iter_mut().zip(lane.chunks_exact(R)) {
        cols.copy_from_slice(held);
    }
    let mut rest = stack;
    for slot in 0..depth {
        let (words, tail) = rest.split_at(slab);
        rest = tail;
        let mut at = slot;
        for cols in acc.iter_mut() {
            let entry = &words[table_code_address(codes[at].into(), code_space, R)..];
            for (cell, &e) in cols.iter_mut().zip(&entry[..R]) {
                *cell = cell.add(e);
            }
            at += stride;
        }
    }
    for (cols, held) in acc.iter().zip(lane.chunks_exact_mut(R)) {
        held.copy_from_slice(cols);
    }
}

/// The reference column step at the widest tile and the narrowest lane, named so
/// that `CU-06`'s disassembly gate has a symbol to read.
///
/// Not a second path and not a test hook: it is [`gather_run`], which every
/// reference gather is, named once at an instantiation the traversal reaches ---
/// a generic function emits no code until something instantiates it, and a gate
/// cannot read instructions that were never emitted.
///
/// Instantiated at a *runtime* slab, which is the binding the gate has to read:
/// with the code count a literal every address is a constant displacement and
/// the absence of a value multiply is trivial, so the claim would pass on the
/// easy case while the hard one shipped unread. At `C = 0` the slot's base is a
/// cursor and the entry is projected into the runtime slab.
#[inline(never)]
pub fn gather_reference_i32(slab: usize, stack: &[i32], off: &[u32], lane: &mut [i32]) {
    gather_run::<0, 16, 1, i32>(slab, stack, off, lane);
}

/// The same, in the exact lane, for the element families that have no narrow
/// register at all.
#[inline(never)]
pub fn gather_reference_wide(
    slab: usize,
    stack: &[Wide<i128>],
    off: &[u32],
    lane: &mut [Wide<i128>],
) {
    gather_run::<0, 16, 1, Wide<i128>>(slab, stack, off, lane);
}

impl<E, L> TableSpec<E, L> {
    /// The safe entry point for [`Self::build`].
    ///
    /// Panics only on a length disagreement, which is a programming error in
    /// the *driver* rather than a condition of the data --- no input a caller
    /// can supply reaches it, which is why `gemm` still returns `()`.
    pub fn build(&self, space: usize, block: usize, book: &[E], acts: &[E], out: &mut [L]) {
        assert_eq!(book.len(), space * block, "the book is space * block");
        assert_eq!(acts.len(), block * self.rows, "the tile is block * rows");
        assert_eq!(out.len(), space * self.rows, "the slot is space * rows");
        assert!(
            block.is_multiple_of(self.k_group),
            "the block is a whole number of k-groups"
        );
        // SAFETY: the three lengths are exactly what `TableBuild` requires, and
        // this spec came from a `*_table` selector, which only ever returns one
        // whose target features the host has.
        unsafe {
            (self.build)(
                self.rows,
                space,
                block,
                book.as_ptr(),
                acts.as_ptr(),
                out.as_mut_ptr(),
            )
        }
    }

    /// The safe entry point for [`Self::gather`].
    ///
    /// `slab` is the lane words one slot occupies: the slab's code count, which
    /// is a power of two at or above the codec's own code space, times
    /// [`Self::rows`], which is also one. Every offset is projected into it, so
    /// this checks lengths and nothing per code.
    pub fn gather(&self, depth: usize, slab: u32, stack: &[L], off: &[u32], lane: &mut [L]) {
        assert!(slab.is_power_of_two(), "one slot is 2^j lane words");
        // Every sequence projects an offset onto a complete row. The native
        // declarations use the retained radix grade; the portable declaration
        // uses quotient/remainder directly.
        assert!(self.rows.is_power_of_two(), "the tile height is 2^j");
        let slab = slab as usize;
        // A slab holds a whole number of entries and an entry is `rows` lane
        // words, so a slab below `rows` holds none --- and every sequence then
        // reads `rows` words from a base the radix projection has pinned to zero. Measured at
        // `slab = 1, rows = 16`: a slice-bounds panic in the reference and a read
        // past the stack in each ISA sequence, from a safe method.
        assert!(self.rows <= slab, "one slot holds at least one entry");
        assert_eq!(stack.len(), depth * slab, "the stack is depth * slab");
        assert_eq!(off.len(), depth * self.group, "the run is depth * group");
        assert_eq!(
            lane.len(),
            self.group * self.rows,
            "the lane is group * rows"
        );
        // SAFETY: the lengths are what `TableGather` requires; the sequence's
        // finite address projection keeps every read in-slab, and this spec came
        // from a `*_table` selector whose target features the host has.
        unsafe {
            (self.gather)(
                self.rows,
                self.group,
                depth,
                slab,
                stack.as_ptr(),
                off.as_ptr(),
                lane.as_mut_ptr(),
            )
        }
    }

    /// The safe entry point for [`Self::gather_codes`].
    ///
    /// `codes` is the coded operand's own memory, starting at the first code of
    /// the first column of the group, with `stride` codes between columns. The
    /// codec has claimed this stream addresses its enumeration
    /// ([`uor_matmul_codec::Enumerable::as_index_stream`]); every code is
    /// projected by the slab's Euclidean radix, so this checks lengths and
    /// nothing per code.
    pub fn gather_codes(
        &self,
        depth: usize,
        slab: u32,
        stack: &[L],
        codes: &[u16],
        stride: usize,
        lane: &mut [L],
    ) {
        assert!(slab.is_power_of_two(), "one slot is 2^j lane words");
        assert!(self.rows.is_power_of_two(), "the tile height is 2^j");
        let slab = slab as usize;
        // A slab holds a whole number of entries and an entry is `rows` lane
        // words, so a slab below `rows` holds none --- and every sequence then
        // reads `rows` words from a base the radix projection has pinned to zero. Measured at
        // `slab = 1, rows = 16`: a slice-bounds panic in the reference and a read
        // past the stack in each ISA sequence, from a safe method.
        assert!(self.rows <= slab, "one slot holds at least one entry");
        assert_eq!(stack.len(), depth * slab, "the stack is depth * slab");
        assert_eq!(
            codes.len(),
            (self.group - 1) * stride + depth,
            "the run spans the group's columns"
        );
        assert_eq!(
            lane.len(),
            self.group * self.rows,
            "the lane is group * rows"
        );
        // SAFETY: the lengths are what `TableGatherCodes` requires. The
        // Euclidean code projection makes every read in-slab, and this spec
        // came from a `*_table` selector.
        unsafe {
            (self.gather_codes)(
                self.rows,
                self.group,
                depth,
                slab,
                table_row_grade(self.rows),
                stack.as_ptr(),
                codes.as_ptr(),
                stride,
                lane.as_mut_ptr(),
            )
        }
    }

    /// [`Self::gather_codes`] at a byte-wide code stream.
    ///
    /// The same claim, the same checks, the same lane words: the codec has
    /// claimed this `u8` stream addresses its enumeration
    /// ([`uor_matmul_codec::Enumerable::as_index_stream`]), every code is
    /// widened and projected into the slab, and nothing is read per code.
    pub fn gather_codes_u8(
        &self,
        depth: usize,
        slab: u32,
        stack: &[L],
        codes: &[u8],
        stride: usize,
        lane: &mut [L],
    ) {
        assert!(slab.is_power_of_two(), "one slot is 2^j lane words");
        assert!(self.rows.is_power_of_two(), "the tile height is 2^j");
        let slab = slab as usize;
        assert!(self.rows <= slab, "one slot holds at least one entry");
        assert_eq!(stack.len(), depth * slab, "the stack is depth * slab");
        assert_eq!(
            codes.len(),
            (self.group - 1) * stride + depth,
            "the run spans the group's columns"
        );
        assert_eq!(
            lane.len(),
            self.group * self.rows,
            "the lane is group * rows"
        );
        // SAFETY: as `gather_codes`.
        unsafe {
            (self.gather_codes_u8)(
                self.rows,
                self.group,
                depth,
                slab,
                table_row_grade(self.rows),
                stack.as_ptr(),
                codes.as_ptr(),
                stride,
                lane.as_mut_ptr(),
            )
        }
    }

    /// The deepest run this lane holds for an alphabet bounded by `bound`, in
    /// *products*.
    ///
    /// A question about a register, not a limit on `k`: a deeper reduction is
    /// cut into more runs, and the runs combine exactly.
    pub fn lane_depth(&self, bound: u128) -> usize {
        let per_step = bound.saturating_mul(bound); // R3-ok: a lane-width question, not an accumulation
        if per_step == 0 || self.lane_cap == u128::MAX {
            return usize::MAX;
        }
        usize::try_from(self.lane_cap / per_step)
            .unwrap_or(usize::MAX)
            .max(1)
    }
}

/// The `i8` table sequences this build can run, reference first.
///
/// The lane is `i32`: one entry of a `Book<_, 8>` over a full `i8` alphabet
/// peaks at `8 * 127 * 127`, and the lane carries 133144 products of it --- past
/// every depth a weight row reaches, so the exact accumulator is touched once
/// per output element rather than once per chunk.
#[inline]
pub fn available_table_i8(rows: usize, group: usize) -> impl Iterator<Item = TableSpec<i8, i32>> {
    collect_table![
        true => portable_table_i8_lookup(rows, group),
        crate::isa::x86::avx2_available() => crate::isa::x86::avx2_table_i8_i32(rows, group),
        crate::isa::x86::avx512_available() => crate::isa::x86::avx512_table_i8_i32(rows, group),
        crate::isa::arm::neon_available() => crate::isa::arm::neon_table_i8_i32(rows, group),
        crate::isa::wasm::simd128_available() => crate::isa::wasm::simd128_table_i8_i32(rows, group),
        // The bound-1 builds, last so `Auto` hands them exactly the alphabet
        // they declare: a full-alphabet sequence is never shadowed, and at
        // bound 1 the adds-only build is the last admissible entry (CB-10).
        true => portable_table_bound1(rows, group),
        crate::isa::x86::avx2_available() => crate::isa::x86::avx2_table_i8_i32_bound1(rows, group),
        crate::isa::arm::neon_available() => crate::isa::arm::neon_table_i8_i32_bound1(rows, group),
    ]
}

/// The `i16` table sequences this build can run, reference first.
///
/// The lane is `i64`: two full `i16` products already need 31 bits, so no
/// 32-bit lane holds an entry of any block longer than one.
#[inline]
pub fn available_table_i16(rows: usize, group: usize) -> impl Iterator<Item = TableSpec<i16, i64>> {
    collect_table![
        true => portable_table::<i16, i64>(rows, group),
        crate::isa::x86::avx2_available() => crate::isa::x86::avx2_table_i16_i64(rows, group),
        crate::isa::x86::avx512_available() => crate::isa::x86::avx512_table_i16_i64(rows, group),
        crate::isa::arm::neon_available() => crate::isa::arm::neon_table_i16_i64(rows, group),
        crate::isa::wasm::simd128_available() => crate::isa::wasm::simd128_table_i16_i64(rows, group),
    ]
}

/// The `i32` table sequences in `Z/2^32` this build can run, reference first.
///
/// The lane is [`Mod32`]. Legitimate exactly when the caller asked to encode by
/// wrapping into an output no wider than the lane, because then the lane's own
/// wrap *is* the encode --- the same declaration [`crate::available_i32_modular`]
/// cashes in on the dense side. *Whether* it may run is the driver's question
/// (`CU-08`); this list only answers what the host can run.
#[inline]
pub fn available_table_i32_modular(
    rows: usize,
    group: usize,
) -> impl Iterator<Item = TableSpec<i32, Mod32>> {
    collect_table![
        true => portable_table::<i32, Mod32>(rows, group),
        crate::isa::x86::avx2_available() => crate::isa::x86::avx2_table_i32_mod32(rows, group),
    ]
}

/// The `i64` table sequences in `Z/2^64` this build can run: the reference
/// alone.
///
/// The build's multiply is the table's only one, and no SIMD integer multiply
/// reaches an `i64` lane --- the same reason [`crate::available_i64_modular`]
/// is portable-only. The reference is the complete sequence the hardware
/// offers.
#[inline]
pub fn available_table_i64_modular(
    rows: usize,
    group: usize,
) -> impl Iterator<Item = TableSpec<i64, Mod64>> {
    collect_table![
        true => portable_table::<i64, Mod64>(rows, group),
    ]
}

/// Lets `collect_table!` read a bare spec and an `Option` alike.
pub trait IntoSpec<E, L> {
    /// This spec, if it exists.
    fn into_option(self) -> Option<TableSpec<E, L>>;
}

impl<E, L> IntoSpec<E, L> for TableSpec<E, L> {
    fn into_option(self) -> Option<TableSpec<E, L>> {
        Some(self)
    }
}

impl<E, L> IntoSpec<E, L> for Option<TableSpec<E, L>> {
    fn into_option(self) -> Option<TableSpec<E, L>> {
        self
    }
}

/// The last admissible sequence for `backend` at this alphabet.
///
/// Last, not first, because the list is written reference-first and every entry
/// after it issues fewer instructions for the same integer. `Backend::Portable`
/// pins a portable sequence --- the reference, or at bound 1 the portable
/// bound-1 build listed after it, which computes the same integer by the
/// declaration it carries (`CB-10`). The parity tests compare against the
/// list's first entry either way.
///
/// **Selection cannot fail**, exactly as [`crate::choose`] cannot, and for the
/// same reason: a named backend this host has no sequence for yields the first
/// admissible entry --- the reference --- which computes the same integer. That
/// is not a fallback, because no answer is given up (R13).
///
/// `block` is the codec's codeword width, and a sequence whose `k_group` does
/// not divide it is inadmissible for the same reason a too-narrow `max_bound`
/// is. Without that term an odd `MAX_BLOCK` --- `Book<_, _, N, 3>` is ordinary
/// public API --- selected a `k_group: 2` sequence and panicked in the driver's
/// packer, for every consumer with runtime detection on.
///
/// This filtered on the backend and returned `None` instead, and the caller's
/// `.expect("the reference sequence is always present")` then panicked. The
/// reference *is* always present; it was the filter that removed it. Measured
/// over `Backend::ALL` at every `(rows, group)` the driver walks, 246 of 250
/// selections came back `None` --- including `Backend::Avx2` on an AVX2 host at
/// every tile below eight rows, which is every `m` under eight. `gemm` returns
/// `()` and has no failure to report (R14), so this could only ever have been a
/// crash.
pub fn choose_table<E, L>(
    specs: impl Iterator<Item = TableSpec<E, L>>,
    backend: Backend,
    bound: u128,
    block: usize,
) -> Option<TableSpec<E, L>> {
    let mut first = None;
    let mut widest = None;
    let mut named = None;
    // A sequence that folds `k_group` block steps into one instruction has no
    // way to express a block that is not a whole number of them --- it is not
    // slower on such a block, it cannot pack it at all, which is what
    // [`TableSpec::build`] asserts. So it is not a factorization of *this*
    // identity and is not considered, exactly as an inadmissible `max_bound` is
    // not. The reference declares `k_group: 1`, which divides every block, so
    // there is always something left to choose (R13).
    for spec in specs.filter(|s| bound <= s.max_bound && block.is_multiple_of(s.k_group)) {
        if first.is_none() {
            first = Some(spec);
        }
        if spec.backend == backend {
            named = Some(spec);
        }
        widest = Some(spec);
    }
    match backend {
        Backend::Auto => widest.or(first),
        _ => named.or(first),
    }
}
