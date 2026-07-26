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

    /// Accumulate one exact product. The only multiply the table issues.
    fn mac(self, a: E, w: E) -> Self;

    /// Place a completed run into the exact accumulator.
    fn place(self, acc: E::Acc) -> E::Acc;
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
        build: portable_build::<E, L>,
        gather: portable_gather::<L>,
        gather_codes: portable_gather_codes::<L>,
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
        |R, G| gather_run::<R, G, L>(slab, stack, off, lane)
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
        |R, G| codes_run::<R, G, L>(depth, slab, shift, stack, codes, stride, lane)
    )
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
#[inline(always)]
fn gather_run<const R: usize, const G: usize, L: LaneWord>(
    slab: usize,
    stack: &[L],
    off: &[u32],
    lane: &mut [L],
) {
    // Derived here, not passed. `words.len()` is `slab` because `split_at` says
    // so, and `at & (slab - 1) <= slab - 1`, so the compiler discharges both
    // slice bounds below instead of checking them.
    let mask = slab - 1;
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
    let mask = slab - 1;
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
#[inline(always)]
fn codes_run<const R: usize, const G: usize, L: LaneWord>(
    depth: usize,
    slab: usize,
    shift: u32,
    stack: &[L],
    codes: &[u16],
    stride: usize,
    lane: &mut [L],
) {
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
#[inline(never)]
pub fn gather_reference_i32(slab: usize, stack: &[i32], off: &[u32], lane: &mut [i32]) {
    gather_run::<16, 1, i32>(slab, stack, off, lane);
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
    gather_run::<16, 1, Wide<i128>>(slab, stack, off, lane);
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
        let slab = slab as usize;
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
/// pins the reference, which is what the parity tests compare against.
pub fn choose_table<E, L>(
    specs: impl Iterator<Item = TableSpec<E, L>>,
    backend: Backend,
    bound: u128,
) -> Option<TableSpec<E, L>> {
    specs
        .filter(|s| bound <= s.max_bound)
        .filter(|s| backend == Backend::Auto || s.backend == backend)
        .last()
}
