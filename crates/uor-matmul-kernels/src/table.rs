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

use uor_matmul_core::{Accumulator, Backend, Element, FloatElement};

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

    /// Accumulate one exact product. The only multiply the table issues ---
    /// and at bound 1 not even this one: [`portable_table_bound1`] fills the
    /// same slot with adds and subtracts, because there every product is
    /// `+-a` or `0` (`CB-10`).
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

/// The float symbol lane: an `i64` word over pre-scaled `f32` significands.
///
/// The newtype exists for the reason [`Mod32`]'s does: the blanket
/// `impl<E: Element> Lane<E> for i64` above already speaks for `i64`-as-lane
/// with *exact full-alphabet* semantics, and for a float that reading is a
/// stub --- a float has no magnitude for its bound to measure. This lane is a
/// different contract on the same register width. The driver pre-scales the
/// codebook and the activation tile to the panels' measured base exponents
/// (the placement bridge's span walk, the same declaration discipline), so
/// what `mac` receives are honest `f32` values that are exact integers below
/// `2^31`; the product of two is an exact integer below `2^62`; and a run of
/// them is exact for the depth the walk's per-side bounds declare. Placement
/// is the accumulator's own `add_scaled` at `base_a + base_b`, reached through
/// [`Lane::place_scaled`] --- the identity the bridge cashes, with the table
/// doing the reduction: scaled significands are exact integers, tabulation is
/// regrouping, and the sum is placed once per run (`CD-20`).
///
/// Admission is the driver's, asked once per call and never by this type: a
/// non-finite code, a span past seven binades, or an `f64` significand is not
/// this lane's alphabet, and those calls tabulate in no lane at all --- the
/// dense float factorization answers for them, at the same bytes.
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

impl LaneWord for Scaled64 {
    const ZERO: Self = Scaled64(0);

    #[inline(always)]
    fn add(self, other: Self) -> Self {
        // Exact within the run the driver derived from the measured spans,
        // which is the only place the gather runs: the same discipline the
        // integer lanes state above, with the bound coming from the panels
        // rather than the alphabet's declaration.
        Scaled64(self.0.wrapping_add(other.0))
    }
}

impl Lane<f32> for Scaled64 {
    #[inline]
    fn capacity(b: u128) -> Option<usize> {
        // Reached by the driver only with the span walk's per-side bound, and
        // by the sizing queries with the alphabet's `u128::MAX` --- where the
        // honest data-free answer is one product, which is why the queries
        // plan against the caches instead (`Tabulated::probe_capacity`).
        capacity_of(i64::MAX as u128, b)
    }

    #[inline(always)]
    fn mac(self, a: f32, w: f32) -> Self {
        let (pa, pw) = (a.pack(), w.pack());
        if pa.mantissa == 0 || pw.mantissa == 0 {
            return self;
        }
        // Each operand is a pre-scaled significand: an exact integer below
        // `2^31`, spelled `mantissa * 2^exp`. The exponents may be negative
        // --- a significand with fewer than 24 significant bits is stored
        // normalized, carrying trailing zeros where the value's low bits are
        // --- so the right shift below drops only zero bits, which is the
        // integer the product is. The left shift cannot lose a bit either:
        // admission bounds each panel's span to seven binades, so the sum of
        // the two exponents is at most fourteen and the product below `2^48`
        // lands below `2^62`.
        let product = pa.mantissa * pw.mantissa;
        let shift = pa.exp.wrapping_add(pw.exp); // R3-ok: an exponent sum, bounded above
        let placed = if shift >= 0 {
            product << shift
        } else {
            product >> shift.wrapping_neg()
        };
        Scaled64(self.0.wrapping_add(placed))
    }

    #[inline]
    fn place(self, acc: <f32 as Element>::Acc) -> <f32 as Element>::Acc {
        // The trait's exponent-free form is placement at `2^0`, and the
        // driver never reaches it for this lane: the traversal places through
        // `place_scaled`, and the streaming decline accumulates raw elements
        // in the family's wide lane. Written out so the contract has no hole.
        self.place_scaled(acc, 0)
    }

    #[inline]
    fn place_scaled(self, mut acc: <f32 as Element>::Acc, exponent: i32) -> <f32 as Element>::Acc {
        // The run is an exact integer at the declared scale; placing it is
        // the decode's own primitive, so nothing is rounded on the way in ---
        // the same sentence `PlaceAt` tells for the bridge's `i128` sum.
        acc.add_scaled(self.0.unsigned_abs() as u128, exponent, self.0 < 0);
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
/// `lane[u * rows + i] += sum_{slot < depth} stack[slot * slab + (off[slot *
/// group + u] & mask) + i]`.
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
/// the inner loop walks and never indexes --- and it is what leaves this loop
/// with one `and` and one fused load-add per code and no multiply of any kind.
/// `CU-06` reads exactly that off the disassembly.
///
/// # Safety
///
/// - `stack` has `depth * slab` readable lanes, and `slab == mask + 1` is a
///   power of two.
/// - `off` has `depth * group` readable words.
/// - `lane` has `group * rows` readable and writable lanes.
/// - Masking is what discharges the bound: every read is in-slab whatever the
///   offset holds, with no branch. Correctness of the *value* is
///   [`uor_matmul_codec::Enumerable`]'s law and `CK-09` asserts it; safety of
///   the *read* is this mask and holds unconditionally.
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
/// `lane[u * rows + i] += sum_{slot < depth} stack[slot * slab + ((codes[u *
/// stride + slot] & mask) << shift) + i]`.
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
/// `shift` rather than a multiply: the tile heights are powers of two, so the
/// index scales into an offset by shifting. `CU-06` reads a column loop with no
/// multiply in it, and this is why it can.
///
/// # Safety
///
/// - `stack` has `depth * slab` readable lanes, and `slab == (mask + 1) << shift`
///   is a power of two.
/// - `codes` has `(group - 1) * stride + depth` readable words.
/// - `lane` has `group * rows` readable and writable lanes.
/// - Masking discharges the bound: every read is in-slab whatever the code
///   holds, with no branch. That the entry is the *right* one is
///   `Enumerable::as_index_stream`'s claim, which `CK-09` asserts.
/// - `rows == 1 << shift` and `group` are this spec's, and the host has the
///   features its backend names.
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
    /// Every build but the bound-1 one does. At bound 1 every product is
    /// `+-a` or `0`, so the build that declares it is adds and subtracts, and
    /// the driver's census charges it as such (`CB-10`) rather than as the
    /// products it computes.
    pub build_multiplies: bool,
    /// Fill one slot.
    pub build: TableBuild<E, L>,
    /// Reduce one column group from an index stream the driver built.
    pub gather: TableGather<L>,
    /// Reduce one column group from the operand's own code stream.
    pub gather_codes: TableGatherCodes<L>,
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
/// register at all, where this is not a placeholder but the whole of what the
/// hardware offers.
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
    let (gather, gather_codes): (TableGather<L>, TableGatherCodes<L>) =
        if core::mem::size_of::<L>() * 16 <= 256 {
            (portable_gather::<L>, portable_gather_codes::<L>)
        } else {
            (portable_gather_wide::<L>, portable_gather_codes_wide::<L>)
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
        build: portable_build::<E, L>,
        gather,
        gather_codes,
    }
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
        build: portable_build_bound1,
        gather: portable_gather::<i32>,
        gather_codes: portable_gather_codes::<i32>,
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
/// [`TableGatherCodes`]'s contract.
#[allow(clippy::too_many_arguments)]
unsafe fn portable_gather_codes<L: LaneWord>(
    rows: usize,
    group: usize,
    depth: usize,
    slab: usize,
    shift: u32,
    stack: *const L,
    codes: *const u16,
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
        |R, G| dispatch_slab!(code_words(slab, R), |C| codes_run::<C, R, G, L>(
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
/// [`TableGatherCodes`]'s contract.
#[allow(clippy::too_many_arguments)]
unsafe fn portable_gather_codes_wide<L: LaneWord>(
    rows: usize,
    group: usize,
    depth: usize,
    slab: usize,
    shift: u32,
    stack: *const L,
    codes: *const u16,
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
fn codes_any<L: LaneWord>(
    rows: usize,
    depth: usize,
    slab: usize,
    shift: u32,
    stack: &[L],
    codes: &[u16],
    stride: usize,
    lane: &mut [L],
) {
    let mask = (slab >> shift) - 1;
    let mut rest = stack;
    for slot in 0..depth {
        let (words, tail) = rest.split_at(slab);
        rest = tail;
        let mut at = slot;
        for cols in lane.chunks_exact_mut(rows) {
            let entry = &words[(codes[at] as usize & mask) << shift..];
            for (cell, &e) in cols.iter_mut().zip(&entry[..rows]) {
                *cell = cell.add(e);
            }
            at += stride;
        }
    }
}

/// The reference column step: the whole of what `CU-06` reads.
///
/// Everything is *walked*, and that is the claim. The slot's base advances by an
/// add, the entry's address is one mask on an offset the caller already scaled,
/// and the accumulation is a compile-time array --- so the loop is one `and` and
/// one add per lane word, and no multiply of any kind.
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
    // Derived here, not passed. `words.len()` is `slab` because `split_at` says
    // so, and the mask below is at most `slab - R`, so the compiler discharges
    // both slice bounds instead of checking them.
    //
    // `slab - 1` alone would bound the entry's *base* and not the `R` lanes read
    // from it, and [`TableSpec::gather`] is a safe public method: an offset that
    // is not a multiple of `R` would start the read inside the last entry and run
    // past the slab --- a panic here and, in the ISA sequences, a read out of
    // bounds. `R` is a power of two and `slab` is a multiple of it, so clearing
    // the sub-row bits is the same single `and` on a different constant, and it
    // makes every read row-aligned by construction rather than by the caller's
    // discipline.
    let mask = (slab - 1) & !(R - 1);
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
            // The mask, not a comparison: every offset reads in-slab whatever it
            // holds, so there is no branch here and none is needed.
            let entry = &words[at as usize & mask..];
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
    let mask = (slab - 1) & !(rows - 1);
    let mut rest = stack;
    for run in off.chunks_exact(group) {
        let (words, tail) = rest.split_at(slab);
        rest = tail;
        for (cols, &at) in lane.chunks_exact_mut(rows).zip(run) {
            let entry = &words[at as usize & mask..];
            for (cell, &e) in cols.iter_mut().zip(&entry[..rows]) {
                *cell = cell.add(e);
            }
        }
    }
}

/// The same, over the coded operand's own memory.
///
/// One shift more than [`gather_run`] and one memory round trip fewer: there is
/// no index stream to write and none to read back.
///
/// `C` binds as it does in [`gather_run`]. The shift is not a second unknown:
/// [`TableSpec::gather_codes`] derives it as `rows.trailing_zeros()`, so at a
/// compile-time `R` it is `R`'s and the entry's address is `(code & (C - 1)) *
/// R` with both factors literal.
#[inline(always)]
fn codes_run<const C: usize, const R: usize, const G: usize, L: LaneWord>(
    depth: usize,
    slab_arg: usize,
    shift_arg: u32,
    stack: &[L],
    codes: &[u16],
    stride: usize,
    lane: &mut [L],
) {
    let (slab, shift) = if C == 0 {
        (slab_arg, shift_arg)
    } else {
        (C * R, R.trailing_zeros())
    };
    let mask = (slab >> shift) - 1;
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
            let entry = &words[(codes[at] as usize & mask) << shift..];
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
/// the absence of a multiply is trivial, so the claim would pass on the easy
/// case while the hard one shipped unread. At `C = 0` the slot's base is a
/// cursor and the entry's is a mask, and that it is still an add is the whole
/// content of `CU-06`.
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
    /// [`Self::rows`], which is also one. Every offset is masked into it, so
    /// this checks lengths and nothing per code.
    pub fn gather(&self, depth: usize, slab: u32, stack: &[L], off: &[u32], lane: &mut [L]) {
        assert!(slab.is_power_of_two(), "one slot is 2^j lane words");
        // The sequences clear the sub-row bits of every offset, which is what
        // keeps a read of `rows` lanes inside the slab whatever the offset holds.
        // That argument needs the tile height to be a power of two, so it is
        // asserted here rather than assumed --- as `gather_codes` already does.
        assert!(self.rows.is_power_of_two(), "the tile height is 2^j");
        let slab = slab as usize;
        // A slab holds a whole number of entries and an entry is `rows` lane
        // words, so a slab below `rows` holds none --- and every sequence then
        // reads `rows` words from a base the mask has pinned to zero. Measured at
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
        // SAFETY: the lengths are what `TableGather` requires, `slab` is a power
        // of two so `slab - 1` is the mask that makes every read in-slab, and
        // this spec came from a `*_table` selector, which only ever returns one
        // whose target features the host has.
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
    /// ([`uor_matmul_codec::Enumerable::as_index_stream`]); every code is masked
    /// into the slab, so this checks lengths and nothing per code.
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
        // reads `rows` words from a base the mask has pinned to zero. Measured at
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
        // SAFETY: the lengths are what `TableGatherCodes` requires, `slab` is a
        // power of two so the mask below it makes every read in-slab, and this
        // spec came from a `*_table` selector.
        unsafe {
            (self.gather_codes)(
                self.rows,
                self.group,
                depth,
                slab,
                self.rows.trailing_zeros(),
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
        true => portable_table::<i8, i32>(rows, group),
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
/// is portable-only. The reference is not a placeholder here but the whole of
/// what the hardware offers.
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
