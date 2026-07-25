//! When the operand is a code, the product is a table read.
//!
//! [`crate::coded`] decodes a code and then multiplies, which issues the same
//! `m * k * n` products the dense driver does and adds a decode on top. The codec
//! buys residency and pays for it in throughput. That is the wrong direction: the
//! codec is the thing that should make the *arithmetic* cheaper.
//!
//! This module is the other direction. For one row of `A` and one block of the
//! reduction, the partial sum against every code in the codec's space is computed
//! once:
//!
//! ```text
//! T[i][p][c]  =  sum over t < Bk  of  A[i, p*Bk + t] * decode(c, t)
//! C[i][j]     =  sum over p       of  T[i][p][ index_of(w[p][j]) ]
//! ```
//!
//! The column loop is one table read and one add per code, covering `Bk`
//! weights. It contains no multiply. `CU-06` asserts that by counting.
//!
//! # Why this is the same value and not an approximation
//!
//! `T[i][p][c]` is a partial sum of the same products, and the total is the sum
//! of those partial sums. An exact sum is a function of the multiset of its
//! products, so regrouping them changes nothing. That is the licence the library
//! already uses for tiling and for [`crate::collapse`], applied to sharing a
//! *product* rather than a row.
//!
//! A classical `sgemm` cannot do this at all. Its `T[c]` would carry its own
//! rounding error, and reusing it across `n` columns would propagate that error
//! `n` times --- so it would have to argue about the *order* of its additions,
//! which is exactly what it cannot do. There is no error here to propagate, so
//! there is nothing to argue. Tabulation is available only to an exact library,
//! and that is the sense in which it is not a generic GEMM trick borrowed from
//! elsewhere.
//!
//! # Which operand is coded, and along which axis
//!
//! The reduction has to run *along* the code block, or there is nothing for the
//! table to sum: a code whose `Bk` elements land in `Bk` different output columns
//! contributes one product to each and no partial sum to any. So the coded
//! operand here is `k`-major --- an `n x k` [`CodedMatrix`], one row per output
//! column --- and the product is `C := A * W^T`. That is the storage a quantized
//! linear layer already uses, and the amortization is over `n`, which is why it
//! holds for a single decoded token as much as for a prefill.
//!
//! [`crate::coded`]'s `k x n` orientation blocks along `n` instead. Nothing is
//! wrong with it and it is not a lesser path: it is the zero-scratch traversal,
//! and it is the orientation in which a code names `Bk` consecutive *outputs*.
//! Tabulation simply has nothing to sum there.

use bytemuck::TransparentWrapper;
use uor_matmul_codec::{CodedMatrix, Enumerable};
use uor_matmul_core::generated::blocking;
use uor_matmul_core::{
    dot_ref, AccOf, Accumulator, Alphabet, Bound, Element, EncodeFrom, IntegerElement, MatView,
    MatViewMut, NotAProduct, Shape, Traversal,
};

use crate::coded::self_aliases;
use crate::driver::GemmOptions;
use crate::epilogue::Epilogue;
use crate::scratch::Scratch;

// ---------------------------------------------------------------------------
// The census
// ---------------------------------------------------------------------------

/// What the traversal issued, counted rather than derived.
///
/// A wall-clock comparison measures the machine as much as the library, and a
/// fitted scaling exponent measures the traversal but not the arithmetic. A
/// census measures the claim directly: it turns "tabulation is faster" into "the
/// tabulated column loop issues zero multiplies and `m*k*n/Bk` adds", which is
/// machine-independent, reproducible, and assertable rather than reportable.
///
/// The authority for the shape is `uor-r4-core`'s `OpKernel`, which declares its
/// arithmetic interface as a census with no multiplication field. r4's census
/// also counts shifts and candidate scans; those are absent here because this
/// traversal issues neither, and a field that can only ever be zero would be a
/// claim about a mechanism that is not present.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Census {
    /// Widening multiply-accumulates. Zero in the column loop, by construction.
    pub multiplies: u64,
    /// Accumulator combines: exact adds at the accumulator's full width.
    pub adds: u64,
    /// Reads of a tabulated partial sum.
    pub table_reads: u64,
    /// Calls into the codec's decode.
    pub decodes: u64,
}

impl Census {
    /// Multiplies per operation, which is the ratio the whole construction is
    /// about. `None` when nothing was issued at all.
    pub fn multiply_share(&self) -> Option<(u64, u64)> {
        let total = self
            .multiplies
            .saturating_add(self.adds)
            .saturating_add(self.table_reads);
        (total > 0).then_some((self.multiplies, total))
    }
}

/// Somewhere to put the census, or nowhere.
///
/// One traversal, two instantiations. `()` implements this with empty bodies, so
/// the shipped call site counts nothing and the optimizer deletes the calls; a
/// `Census` counts. There is no second loop nest and no `cfg`, which is what
/// keeps the counted run and the shipped run the same function (R13).
pub trait Ledger {
    /// Record `n` widening multiply-accumulates.
    fn multiplied(&mut self, n: u64);
    /// Record `n` exact accumulator combines.
    fn added(&mut self, n: u64);
    /// Record `n` reads of a tabulated partial sum.
    fn read(&mut self, n: u64);
    /// Record `n` codec decodes.
    fn decoded(&mut self, n: u64);
}

impl Ledger for () {
    fn multiplied(&mut self, _: u64) {}
    fn added(&mut self, _: u64) {}
    fn read(&mut self, _: u64) {}
    fn decoded(&mut self, _: u64) {}
}

impl Ledger for Census {
    fn multiplied(&mut self, n: u64) {
        self.multiplies = self.multiplies.saturating_add(n); // R3-ok: a counter
    }
    fn added(&mut self, n: u64) {
        self.adds = self.adds.saturating_add(n); // R3-ok: a counter
    }
    fn read(&mut self, n: u64) {
        self.table_reads = self.table_reads.saturating_add(n); // R3-ok: a counter
    }
    fn decoded(&mut self, n: u64) {
        self.decodes = self.decodes.saturating_add(n); // R3-ok: a counter
    }
}

// ---------------------------------------------------------------------------
// The lane
// ---------------------------------------------------------------------------

/// A word the table and the column accumulation are held in.
///
/// The exact accumulator is 128 bits wide or more, and the column loop touches
/// one per output cell per block of the reduction. At `m*n*k/Bk` touches that is
/// not arithmetic, it is *traffic*: a 64x1024x4096 product moves a gigabyte of
/// accumulator through the cache to compute a quarter of a billion products.
///
/// A lane is where that stops. One table entry is a sum of `MAX_BLOCK` products
/// of two alphabet elements, and a chunk of the reduction is a sum of those, so a
/// narrow register holds it exactly for a depth this trait states. The exact
/// accumulator is then touched once per *chunk* instead of once per block, and
/// the words the inner loop moves halve or better.
///
/// This is the same narrow/wide factorization the tile kernels already run under
/// [`uor_matmul_core::fits_narrow`], at a different place in the traversal. Both
/// lanes compute the same integer, and `CD-13` asserts the bytes.
pub trait LaneWord: Copy + Send + Sync + 'static {
    /// The additive identity.
    const ZERO: Self;
    /// Exact within the lane's declared capacity, which is the only place it is
    /// reached.
    fn add(self, other: Self) -> Self;
}

/// A lane for a particular element type: how to fill it, and how to place it.
pub trait Lane<E: Element>: LaneWord {
    /// The most products this lane holds exactly, for an alphabet bounded by `b`.
    ///
    /// `None` is unbounded --- the exact accumulator, which the width derivation
    /// already sized against every depth any machine can address.
    fn capacity(b: u128) -> Option<usize>;

    /// Accumulate one exact product. The only multiply the traversal issues.
    fn mac(self, a: E, w: E) -> Self;

    /// Place a completed chunk into the exact accumulator.
    fn place(self, acc: E::Acc) -> E::Acc;
}

impl LaneWord for i64 {
    const ZERO: Self = 0;

    fn add(self, other: Self) -> Self {
        // Exact: both operands are partial sums of a chunk whose length
        // `capacity` bounded, so this cannot overflow where it is reached. A
        // derivation error would panic under the checked profile, which is what
        // that profile is for (`CT-02`).
        self + other
    }
}

impl<E: Element> Lane<E> for i64 {
    fn capacity(b: u128) -> Option<usize> {
        // The worst-case magnitude of one product is `b * b`; a bound of zero is
        // an alphabet containing only zero, for which every depth fits.
        let per_step = b.saturating_mul(b);
        if per_step == 0 {
            return None;
        }
        let room = (i64::MAX as u128) / per_step;
        Some(if room > usize::MAX as u128 {
            usize::MAX
        } else {
            room as usize
        })
    }

    fn mac(self, a: E, w: E) -> Self {
        E::mac_narrow(self, a, w)
    }

    fn place(self, acc: E::Acc) -> E::Acc {
        E::combine_narrow(acc, self)
    }
}

impl LaneWord for i32 {
    const ZERO: Self = 0;

    fn add(self, other: Self) -> Self {
        // Exact within the capacity below, which is the only place it is reached.
        self + other
    }
}

impl<E: Element> Lane<E> for i32 {
    fn capacity(b: u128) -> Option<usize> {
        let per_step = b.saturating_mul(b);
        if per_step == 0 {
            return None;
        }
        let room = (i32::MAX as u128) / per_step;
        Some(if room > usize::MAX as u128 {
            usize::MAX
        } else {
            room as usize
        })
    }

    fn mac(self, a: E, w: E) -> Self {
        E::mac_narrow32(self, a, w)
    }

    fn place(self, acc: E::Acc) -> E::Acc {
        E::combine_narrow(acc, self as i64)
    }
}

/// The exact accumulator, as a lane.
///
/// The wrapper exists so that "the accumulator used as a lane" and "the narrow
/// register used as a lane" are two types rather than two code paths. It is
/// `repr(transparent)`, so a caller's accumulator offer *is* a buffer of these
/// and no copy stands between them.
#[derive(Clone, Copy, PartialEq, Eq, Debug, TransparentWrapper)]
#[repr(transparent)]
pub struct Wide<A>(pub A);

impl<A: Accumulator> LaneWord for Wide<A> {
    const ZERO: Self = Wide(A::ZERO);

    fn add(self, other: Self) -> Self {
        Wide(self.0.combine(other.0))
    }
}

impl<E: Element> Lane<E> for Wide<E::Acc> {
    fn capacity(_: u128) -> Option<usize> {
        // The width derivation already covers every depth any addressable machine
        // can present, so there is nothing left to bound (§3.2).
        None
    }

    fn mac(mut self, a: E, w: E) -> Self {
        E::mac(&mut self.0, a, w);
        self
    }

    fn place(self, acc: E::Acc) -> E::Acc {
        acc.combine(self.0)
    }
}

/// One code's contribution to a tile of `R` rows: one table read and one add
/// each, and no multiply.
///
/// Two arrays of a *compile-time* length, so the whole thing is `R` registers and
/// there is no index arithmetic, no bounds check and no loop framing in it at all.
/// That is not tidiness: it is what makes `CU-06`'s disassembly half a statement
/// about the accumulation rather than about addressing, and measured it was worth
/// more than everything else in this module put together.
#[inline(always)]
fn add_entry<const R: usize, L: LaneWord>(entry: &[L; R], acc: &mut [L; R]) {
    for i in 0..R {
        acc[i] = acc[i].add(entry[i]);
    }
}

/// The accumulation of [`add_entry`] in the narrowest lane at the widest tile,
/// named so that `CU-06`'s disassembly gate has a symbol to read.
///
/// Not a second path and not a test hook: it is the same function, named once at
/// an instantiation the shipped traversal reaches, because a generic function
/// emits no code until something instantiates it and a gate cannot read
/// instructions that were never emitted.
#[inline(never)]
pub fn add_entry_narrow(entry: &[i32; 8], acc: &mut [i32; 8]) {
    add_entry(entry, acc);
}

/// The same, in the exact lane, for the element families that have no narrow
/// register at all.
#[inline(never)]
pub fn add_entry_wide(entry: &[Wide<i128>; 8], acc: &mut [Wide<i128>; 8]) {
    add_entry(entry, acc);
}

// ---------------------------------------------------------------------------
// The table
// ---------------------------------------------------------------------------

/// The tabulation buffer for one row tile and a chunk of the reduction.
///
/// Borrowed, never owned: it lives in the caller's offer like every other working
/// buffer in this library (R7, S13). `depth * code_space * rows` lane words,
/// indexed block-major then code-major, so one code's entry is a contiguous run
/// of `rows` words and [`add_entry`] walks it against the column accumulation
/// without a stride.
///
/// The *depth* is why the exact accumulator stops being traffic. A stack of
/// `depth` tables lets the column loop reduce `depth` blocks into a lane held in
/// registers and place the result once, so the accumulator is touched
/// `m*n*(k/Bk)/depth` times instead of `m*n*(k/Bk)`.
#[derive(Debug)]
pub struct Table<'s, L> {
    words: &'s mut [L],
    code_space: usize,
    rows: usize,
    depth: usize,
}

impl<'s, L: LaneWord> Table<'s, L> {
    /// How many lane words a stack of `depth` tables over `rows` rows of a
    /// `code_space`-wide enumeration occupies.
    ///
    /// A query, so an embedded caller can size a static and know the answer
    /// before it calls anything.
    pub const fn words(code_space: usize, rows: usize, depth: usize) -> usize {
        code_space.saturating_mul(rows).saturating_mul(depth)
    }

    /// Borrow `words` as a stack of `depth` tables.
    ///
    /// `None` when the borrow is shorter than the table it is asked to be, which
    /// means no such table exists in that offer. Decided here, before any
    /// arithmetic, and answered by the caller taking the streaming traversal
    /// instead --- not by an error reaching anyone (C6).
    pub fn new(words: &'s mut [L], code_space: usize, rows: usize, depth: usize) -> Option<Self> {
        if code_space == 0
            || rows == 0
            || depth == 0
            || words.len() < Self::words(code_space, rows, depth)
        {
            return None;
        }
        Some(Self {
            words,
            code_space,
            rows,
            depth,
        })
    }

    /// Distinct codes each table in the stack is indexed by.
    pub const fn code_space(&self) -> usize {
        self.code_space
    }

    /// Rows of `A` the stack covers.
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Blocks of the reduction the stack holds at once.
    pub const fn depth(&self) -> usize {
        self.depth
    }

    /// Every entry of one block of the reduction, into stack slot `slot`.
    ///
    /// `code_space * block * rows` products, and the only multiplies the tabulated
    /// traversal issues at all.
    ///
    /// The reduction's step outer, the code space inner. That order is not free to
    /// choose, and it is the whole cost of the build:
    ///
    /// - The `R` activations of one step are read **once** and stay in registers
    ///   across the entire code space. Codes-outer would re-read them
    ///   `code_space` times, and measured that re-read was half the traversal.
    /// - Each codeword's `block` elements are still decoded once per tile rather
    ///   than once per row, which is `code_space * block` decode calls and not
    ///   `code_space * block * R`.
    ///
    /// `R` is the tile's height and a compile-time one, because the last tile of a
    /// shape that does not divide is shorter and the caller walks down
    /// [`ROW_TILES`] rather than passing a runtime length. The stride stays the
    /// table's, so a ragged tile changes what is written and never where.
    pub fn build<const R: usize, E, Bd, C, Lg>(
        &mut self,
        book: &[Alphabet<E, Bd>],
        acts: &[Alphabet<E, Bd>],
        slot: usize,
        ledger: &mut Lg,
    ) where
        E: IntegerElement,
        Bd: Bound,
        C: Enumerable<E, Bd>,
        L: Lane<E>,
        Lg: Ledger,
    {
        let block = C::MAX_BLOCK;
        let space = self.code_space;
        let slab = space * R;
        let start = slot * slab;
        let words = &mut self.words[start..start + slab];
        // The code space outer, the block inner. That order is the whole cost of
        // the build: one entry's `R` words stay in registers for its whole
        // codeword and are written **once**. Block-outer reads and writes every
        // entry once per element of the block, which is `MAX_BLOCK` times the
        // traffic --- measured, that was half the traversal.
        for (entry, word) in words.chunks_exact_mut(R).zip(book.chunks_exact(block)) {
            let mut acc = [L::ZERO; R];
            for (&w, col) in word.iter().zip(acts.chunks_exact(R)) {
                let w = w.get();
                for i in 0..R {
                    acc[i] = acc[i].mac(col[i].get(), w);
                }
            }
            if let Some(entry) = entry.first_chunk_mut::<R>() {
                *entry = acc;
            }
        }
        ledger.multiplied((space * block * R) as u64);
    }

    /// The stack itself, one slab of `code_space * rows` words per block held.
    ///
    /// The column loop walks it with [`slice::chunks_exact`], so the slot's base
    /// advances by an add rather than a multiply and the slab's bounds are checked
    /// once for the whole walk instead of once per code.
    pub fn stack(&self) -> &[L] {
        self.words
    }

    /// Words per block held: one entry per code, one word per row.
    pub const fn slab(&self) -> usize {
        self.code_space * self.rows
    }

    /// One code's entry in one slot of the stack: the partial sum for each of the
    /// tile's `R` rows, as an array so the column loop keeps it in registers.
    ///
    /// No bounds check is needed in principle: [`Enumerable::index_of`] is total
    /// below `CODE_SPACE` and the buffer is `CODE_SPACE` entries wide per slot by
    /// construction, which is what `CT-07` asserts.
    #[inline(always)]
    pub fn entry<const R: usize>(&self, slot: usize, code_index: usize) -> Option<&[L; R]> {
        let at = (slot * self.code_space + code_index) * self.rows;
        self.words.get(at..at + R)?.first_chunk::<R>()
    }
}

// ---------------------------------------------------------------------------
// The conformant triple
// ---------------------------------------------------------------------------

/// The conformant triple for `C := A * W^T`, with `W` coded `k`-major.
///
/// `A` is `m x k`, `W` is `n x k` --- one coded row per output column --- and `C`
/// is `m x n`. Reports the same two non-existences at the same moment as
/// [`uor_matmul_core::Triple::new`] and [`crate::CodedTriple::new`], and nothing
/// else, ever (§5.5, C6).
pub struct TabulatedTriple<'a, 'w, 'c, E: IntegerElement, Bd: Bound, C: Enumerable<E, Bd>, O> {
    a: MatView<'a, Alphabet<E, Bd>>,
    w: CodedMatrix<'w, E, Bd, C>,
    c: MatViewMut<'c, O>,
}

impl<'a, 'w, 'c, E: IntegerElement, Bd: Bound, C: Enumerable<E, Bd>, O>
    TabulatedTriple<'a, 'w, 'c, E, Bd, C, O>
{
    /// Report non-existence once, before any arithmetic is named.
    pub fn new(
        a: MatView<'a, Alphabet<E, Bd>>,
        w: CodedMatrix<'w, E, Bd, C>,
        c: MatViewMut<'c, O>,
    ) -> Result<Self, NotAProduct> {
        // `W` is transposed by declaration, so conformance is `a.cols == w.cols`
        // and `c.cols == w.rows`. The reported pair is the shape of the product
        // the caller asked for, which is `W^T`.
        if a.cols() != w.cols() || c.rows() != a.rows() || c.cols() != w.rows() {
            return Err(NotAProduct::Nonconformant {
                a: (a.rows(), a.cols()),
                b: (w.cols(), w.rows()),
            });
        }
        let s = c.strides();
        if self_aliases(c.rows(), c.cols(), s.rs, s.cs) {
            return Err(NotAProduct::OutputAliasesItself {
                m: c.rows(),
                n: c.cols(),
                rs: s.rs,
                cs: s.cs,
            });
        }
        Ok(Self { a, w, c })
    }

    /// The shape of the product, which exists because this value does.
    pub fn shape(&self) -> Shape {
        Shape {
            m: self.a.rows(),
            k: self.a.cols(),
            n: self.w.rows(),
        }
    }
}

// ---------------------------------------------------------------------------
// The offer
// ---------------------------------------------------------------------------

/// The narrow lane the tabulated traversal accumulates chunks of the reduction
/// in, offered by the caller.
///
/// A separate offer for the same reason [`crate::Collapse`] is one: it is a
/// different buffer with a different element type, and the library owns neither.
/// Offering none is well formed --- an element family with a narrow register then
/// tabulates in the exact accumulator out of [`Scratch`], and one without was
/// always going to.
#[derive(Debug)]
pub struct Tabulation<'s> {
    lanes: &'s mut [i64],
}

impl<'s> Tabulation<'s> {
    /// Offer a narrow lane buffer.
    pub fn new(lanes: &'s mut [i64]) -> Self {
        Self { lanes }
    }

    /// Offer none.
    ///
    /// Not a degraded mode and not a fallback: the same identity, accumulated in
    /// a wider register (R13).
    pub fn none() -> Tabulation<'static> {
        Tabulation { lanes: &mut [] }
    }

    /// How much was offered.
    pub fn len(&self) -> usize {
        self.lanes.len()
    }

    /// Was nothing offered?
    pub fn is_empty(&self) -> bool {
        self.lanes.is_empty()
    }
}

/// How many narrow lane words would let the tabulated traversal run at its
/// intended stack depth for this shape and this codec.
///
/// A *query*, like [`crate::suggested_scratch`]. Zero when the element type has no
/// narrow register, in which case the traversal accumulates in the exact one and
/// wants that much more of [`suggested_tabulation`] instead.
///
/// It does not grow with `n`, and it grows with `k` only until the stack reaches
/// the depth the cache holds.
pub fn suggested_tabulation_lanes<E: IntegerElement, Bd: Bound>(
    shape: Shape,
    code_space: usize,
    block: usize,
) -> usize {
    let lane = LaneChoice::resolve::<E, Bd>(block);
    if lane.is_exact(core::mem::size_of::<AccOf<E>>()) {
        return 0;
    }
    let Some(plan) = Plan::choose(code_space, shape, lane, usize::MAX, usize::MAX, block) else {
        return 0;
    };
    // Reported in `i64` words, which is what the offer is made of.
    plan.lane_words(code_space)
        .saturating_mul(lane.bytes)
        .div_ceil(core::mem::size_of::<i64>())
}

/// How many exact accumulators would let the tabulated traversal run at the whole
/// output width for this shape and this codec.
///
/// A *query*, like [`crate::suggested_scratch`]. Offering less narrows the column
/// block; offering none gives the same bytes from the streaming traversal
/// (`CD-13`). It does not grow with `k`.
///
/// When the element type has no narrow register this covers the table stack too,
/// because there is then nowhere else for it to live.
pub fn suggested_tabulation<E: IntegerElement, Bd: Bound>(
    shape: Shape,
    code_space: usize,
    block: usize,
) -> usize {
    let lane = LaneChoice::resolve::<E, Bd>(block);
    let exact = lane.is_exact(core::mem::size_of::<AccOf<E>>());
    let Some(plan) = Plan::choose(code_space, shape, lane, usize::MAX, usize::MAX, block) else {
        return 0;
    };
    let tile = plan.rows.saturating_mul(plan.cols);
    if exact {
        tile.saturating_add(plan.lane_words(code_space))
    } else {
        tile
    }
}

// ---------------------------------------------------------------------------
// Selection
// ---------------------------------------------------------------------------

/// Does a table for `rows` rows of a `code_space`-wide enumeration fit L1?
///
/// The factor of two leaves half of L1 for the code stream and the column
/// accumulation. A table that does not sit in L1 turns every output column into a
/// cache miss, and the traversal stops paying long before the op count says it
/// should --- which is why residency is a term of the predicate and not an
/// afterthought.
pub const fn tabulation_fits(
    code_space: usize,
    rows: usize,
    l1_bytes: usize,
    lane_bytes: usize,
) -> bool {
    code_space > 0
        && rows > 0
        && code_space
            .saturating_mul(rows)
            .saturating_mul(lane_bytes)
            .saturating_mul(2)
            <= l1_bytes
}

/// Does tabulation issue fewer operations than blocking, and does its table fit?
///
/// `cols` is the width of the column block the caller's offer supports, which is
/// `n` when the offer holds the whole output width. The build is repeated once per
/// column block, so it is the block and not the shape that the op count turns on.
///
/// Instructions, not operations. One tabulated lane operation covers
/// `block * rows` products --- `rows` outputs at once, each against a whole
/// codeword --- and one instruction of a dense tile kernel covers
/// [`blocking::KERNEL_PRODUCTS_PER_STEP`] of them. Counting both as one apiece
/// over-sells the table, and measured it did: at `1000x512x512` the operation
/// count said the table won and the kernels were four times faster per product.
///
/// ```text
/// tabulated = m*k*S/rows + m*n*(k/block)/rows      dense = m*k*n/kernel_step
/// ```
///
/// so the table is cheaper exactly when
/// `cols * (block*rows - kernel_step) > code_space * kernel_step * block`.
///
/// Two cases have no such `cols` and are refused outright: `block == 1`, where one
/// code names one element; and `block * rows <= kernel_step`, where one lane
/// operation does not even cover what one dense instruction does, so nothing
/// repays the build.
pub const fn tabulation_pays(
    code_space: usize,
    block: usize,
    cols: usize,
    rows: usize,
    l1_bytes: usize,
    lane_bytes: usize,
) -> bool {
    // A dense tile issues its products per instruction only when it has
    // `KERNEL_ROWS` rows to fill; with fewer it pays for the lanes that are not
    // there. That term is what makes this right at `m = 1`, where a tile kernel
    // has one useful row in six and a table has none to waste --- measured, the
    // table was three times faster there and the predicate without it said no.
    let present = if rows < blocking::KERNEL_ROWS {
        rows
    } else {
        blocking::KERNEL_ROWS
    };
    let effective =
        blocking::KERNEL_PRODUCTS_PER_STEP.saturating_mul(present) / blocking::KERNEL_ROWS;
    let per_lane = block.saturating_mul(rows);
    block > 1
        && effective > 0
        && per_lane > effective
        && cols.saturating_mul(per_lane - effective)
            > code_space.saturating_mul(effective).saturating_mul(block)
        && tabulation_fits(code_space, rows, l1_bytes, lane_bytes)
}

/// The most rows of `A` one table can cover and still sit in L1.
///
/// Derived from the cache budget and the code space, and capped by the same `MC`
/// the blocked traversal uses --- not by a number chosen for this traversal (R8).
/// Zero means no table fits at all, which selects the streaming traversal.
pub const fn tabulation_rows(code_space: usize, l1_bytes: usize, lane_bytes: usize) -> usize {
    if code_space == 0 || lane_bytes == 0 {
        return 0;
    }
    let room = l1_bytes / (2 * code_space * lane_bytes);
    if room < blocking::MC {
        room
    } else {
        blocking::MC
    }
}

/// How many blocks of the reduction one stack of tables may hold at once.
///
/// Two bounds, and the smaller wins. The stack is read once per output column, so
/// it must sit in the last level of cache the column loop can afford: that is the
/// `l2_bytes` term. And a chunk of the reduction is accumulated in one lane word,
/// so it may be no longer than the lane holds exactly: that is `lane_capacity`,
/// which is `None` for the exact accumulator because the width derivation already
/// covers every depth a machine can address.
///
/// Depth is the whole reason the exact accumulator stops being traffic. At depth
/// `d` it is touched `m*n*(k/Bk)/d` times instead of `m*n*(k/Bk)`.
pub const fn tabulation_depth(
    code_space: usize,
    rows: usize,
    block: usize,
    lane_capacity: Option<usize>,
    l2_bytes: usize,
    lane_bytes: usize,
) -> usize {
    if code_space == 0 || rows == 0 || lane_bytes == 0 || block == 0 {
        return 0;
    }
    let per_slot = code_space * rows * lane_bytes;
    // Half of L2. The other half is the code stream and the exact accumulator
    // tile, which pass through the same cache. Measured, a quarter was better
    // while the column loop had one load in flight and half is better now that it
    // has `COLUMN_GROUP` of them: the stack stops needing to be resident once the
    // latency is overlapped, and what is left to minimise is placement traffic,
    // which falls as `1/depth`.
    let mut depth = l2_bytes / (2 * per_slot);
    if depth == 0 {
        depth = 1;
    }
    match lane_capacity {
        // The lane holds `cap` products; a block contributes `block` of them.
        Some(cap) => {
            let by_lane = cap / block;
            if by_lane < depth {
                by_lane
            } else {
                depth
            }
        }
        None => depth,
    }
}

/// A lane's two facts, as the planner needs them: how wide one word is, and how
/// many products one word holds exactly.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct LaneChoice {
    bytes: usize,
    capacity: Option<usize>,
}

impl LaneChoice {
    /// The lane this instantiation resolves to: the **narrowest** that can hold a
    /// block of the reduction.
    ///
    /// The same scan, and the same reason, as
    /// [`uor_matmul_core::narrow_cap_for`]: every lane computes the same integer,
    /// and a narrower word halves the table, doubles the rows that sit in L1 and
    /// doubles the depth that sits in L2. Traffic is what the column loop is bound
    /// by, so the narrowest lane is the fastest one.
    ///
    /// `None` for [`Element::HAS_NARROW`] false means the exact accumulator, which
    /// the width derivation already sized against every depth (§3.2).
    fn resolve<E: IntegerElement, Bd: Bound>(block: usize) -> Self {
        let holds = |c: Option<usize>| c.is_some_and(|c| c >= block);
        if E::HAS_NARROW32 {
            let narrow32 = <i32 as Lane<E>>::capacity(Bd::VALUE);
            if holds(narrow32) {
                return Self {
                    bytes: core::mem::size_of::<i32>(),
                    capacity: narrow32,
                };
            }
        }
        if E::HAS_NARROW {
            let narrow64 = <i64 as Lane<E>>::capacity(Bd::VALUE);
            if holds(narrow64) {
                return Self {
                    bytes: core::mem::size_of::<i64>(),
                    capacity: narrow64,
                };
            }
        }
        Self {
            bytes: core::mem::size_of::<AccOf<E>>(),
            capacity: None,
        }
    }

    /// Is this the exact accumulator rather than a narrow register?
    const fn is_exact(&self, acc_bytes: usize) -> bool {
        self.capacity.is_none() && self.bytes == acc_bytes
    }
}

/// The row tile, column block and stack depth one call resolves to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Plan {
    rows: usize,
    cols: usize,
    depth: usize,
}

impl Plan {
    /// Lane words the plan needs: the stack, and nothing else.
    ///
    /// The column accumulation itself is `R` registers and never a buffer, which
    /// is what [`row_tile`]'s compile-time row count buys.
    const fn lane_words(&self, code_space: usize) -> usize {
        code_space
            .saturating_mul(self.rows)
            .saturating_mul(self.depth)
    }

    /// The largest plan the two offers support.
    ///
    /// Rows first, from the cache budget: a wider row tile shares each decode
    /// across more outputs and puts more lane words under one vector instruction.
    /// Then the column block, as wide as the exact offer allows, because the build
    /// repeats once per column block. Then the depth, as deep as the lane offer
    /// and the lane's own capacity allow, because depth is what keeps the exact
    /// accumulator out of the inner loop.
    fn choose(
        code_space: usize,
        shape: Shape,
        lane: LaneChoice,
        exact_offer: usize,
        lane_offer: usize,
        block: usize,
    ) -> Option<Self> {
        if code_space == 0 || block == 0 || shape.m == 0 || shape.n == 0 {
            return None;
        }
        let LaneChoice {
            bytes: lane_bytes,
            capacity: cap,
        } = lane;
        let row_cap = tabulation_rows(code_space, blocking::L1_BYTES, lane_bytes)
            .min(shape.m)
            .min(exact_offer);
        // The row tile is one of the counts the column loop is compiled for, so
        // that its accumulation is registers and not a buffer. Widest first.
        let rows = ROW_TILES
            .into_iter()
            .find(|&r| r <= row_cap && code_space.saturating_mul(r) <= lane_offer)?;
        let cols = shape.n.min(exact_offer / rows);
        if cols == 0 {
            return None;
        }
        let blocks = shape.k / block;
        let by_cache =
            tabulation_depth(code_space, rows, block, cap, blocking::L2_BYTES, lane_bytes);
        let by_offer = lane_offer / (code_space * rows);
        let depth = by_cache.min(by_offer).min(blocks.max(1)).max(1);
        Some(Self { rows, cols, depth })
    }
}

// ---------------------------------------------------------------------------
// The traversal
// ---------------------------------------------------------------------------

/// `C := epilogue(A * W^T, C)`, with the table when the offers admit one.
///
/// Returns `()`, for the same reason [`crate::gemm`] does.
pub fn gemm_tabulated<E, Bd, C, O, Ep>(
    triple: &mut TabulatedTriple<'_, '_, '_, E, Bd, C, O>,
    epilogue: &Ep,
    options: GemmOptions,
    scratch: &mut Scratch<'_, E, Bd>,
    lanes: &mut Tabulation<'_>,
) where
    E: IntegerElement,
    Bd: Bound,
    C: Enumerable<E, Bd>,
    O: Element + EncodeFrom<AccOf<E>>,
    Ep: Epilogue<E, O>,
{
    run(triple, epilogue, options, scratch, lanes, &mut ());
}

/// The same traversal, with the operation census written out.
///
/// Not a second path: [`gemm_tabulated`] is this function at `Lg = ()`, where
/// every ledger call has an empty body and disappears. `CU-06` reads the census
/// this returns and `CD-13` asserts the two give the same bytes.
pub fn gemm_tabulated_counted<E, Bd, C, O, Ep>(
    triple: &mut TabulatedTriple<'_, '_, '_, E, Bd, C, O>,
    epilogue: &Ep,
    options: GemmOptions,
    scratch: &mut Scratch<'_, E, Bd>,
    lanes: &mut Tabulation<'_>,
    census: &mut Census,
) where
    E: IntegerElement,
    Bd: Bound,
    C: Enumerable<E, Bd>,
    O: Element + EncodeFrom<AccOf<E>>,
    Ep: Epilogue<E, O>,
{
    run(triple, epilogue, options, scratch, lanes, census);
}

fn run<E, Bd, C, O, Ep, Lg>(
    triple: &mut TabulatedTriple<'_, '_, '_, E, Bd, C, O>,
    epilogue: &Ep,
    options: GemmOptions,
    scratch: &mut Scratch<'_, E, Bd>,
    lanes: &mut Tabulation<'_>,
    ledger: &mut Lg,
) where
    E: IntegerElement,
    Bd: Bound,
    C: Enumerable<E, Bd>,
    O: Element + EncodeFrom<AccOf<E>>,
    Ep: Epilogue<E, O>,
    Lg: Ledger,
{
    let shape = triple.shape();
    if shape.m == 0 || shape.n == 0 {
        // Nothing to write. Not a special case: the loops below would do the same
        // thing, and saying so costs one comparison.
        return;
    }

    let space = C::CODE_SPACE;
    let block = <C as uor_matmul_codec::Codec<E, Bd>>::MAX_BLOCK;
    // A code stream whose blocks are not a fixed width has no `p`-th block to
    // index, so there is nothing for a table to be built per block of. The one
    // such tier does not implement `Enumerable`, so this is unreachable through
    // the shipped codecs; it is here because the trait does not forbid it.
    let addressable =
        <C as uor_matmul_codec::Codec<E, Bd>>::IS_FIXED_WIDTH && block >= 1 && space > 0;
    if !addressable || options.traversal == Traversal::OutputMajor {
        stream(triple, epilogue, options, scratch.take(shape.k), ledger);
        return;
    }

    // The narrowest lane that holds a block of the reduction, the same scan the
    // tile kernels run. Every lane computes the same integer; this is a question
    // about register width and traffic, and `CD-13` asserts the bytes across it.
    let lane = LaneChoice::resolve::<E, Bd>(block);
    let acc_bytes = core::mem::size_of::<AccOf<E>>();
    let exact_offer = scratch.accumulators();

    if lane.is_exact(acc_bytes) {
        // The exact accumulator as the lane, out of the same offer the tile comes
        // from. Wider words and a shallower stack; the same identity, and for
        // `i64` and complex elements the only lane there is.
        let Some(plan) = Plan::choose(space, shape, lane, exact_offer, exact_offer, block) else {
            stream(triple, epilogue, options, scratch.take(shape.k), ledger);
            return;
        };
        let tile = plan.rows * plan.cols;
        let want = tile + plan.lane_words(space);
        if want > exact_offer || !admits(options.traversal, space, block, plan, lane.bytes) {
            stream(triple, epilogue, options, scratch.take(shape.k), ledger);
            return;
        }
        let (panel, accumulators) = scratch.split(suggested_tabulation_panel(space, block), want);
        if !decode_book(triple.w.codec(), panel, ledger) {
            stream(triple, epilogue, options, scratch.take(shape.k), ledger);
            return;
        }
        let (exact, stack) = accumulators.split_at_mut(tile);
        let words: &mut [Wide<AccOf<E>>] = Wide::wrap_slice_mut(stack);
        tabulate::<E, Bd, C, O, Ep, Wide<AccOf<E>>, Lg>(
            triple, epilogue, options, exact, words, panel, plan, ledger,
        );
        return;
    }

    // A narrow lane, out of the caller's lane offer. The offer is `i64`-shaped
    // because that is the widest narrow word; a 32-bit lane reads twice as many
    // of them out of the same bytes.
    let offered = core::mem::size_of_val(lanes.lanes) / lane.bytes;
    let Some(plan) = Plan::choose(space, shape, lane, exact_offer, offered, block) else {
        stream(triple, epilogue, options, scratch.take(shape.k), ledger);
        return;
    };
    if !admits(options.traversal, space, block, plan, lane.bytes) {
        stream(triple, epilogue, options, scratch.take(shape.k), ledger);
        return;
    }
    let (panel, exact) = scratch.split(
        suggested_tabulation_panel(space, block),
        plan.rows * plan.cols,
    );
    if !decode_book(triple.w.codec(), panel, ledger) {
        stream(triple, epilogue, options, scratch.take(shape.k), ledger);
        return;
    }
    let want = plan.lane_words(space);
    if lane.bytes == core::mem::size_of::<i32>() {
        let words: &mut [i32] = bytemuck::cast_slice_mut(lanes.lanes);
        tabulate::<E, Bd, C, O, Ep, i32, Lg>(
            triple,
            epilogue,
            options,
            exact,
            &mut words[..want],
            panel,
            plan,
            ledger,
        );
    } else {
        tabulate::<E, Bd, C, O, Ep, i64, Lg>(
            triple,
            epilogue,
            options,
            exact,
            &mut lanes.lanes[..want],
            panel,
            plan,
            ledger,
        );
    }
}

/// Does the caller's named traversal admit the table this plan describes?
///
/// [`Traversal::Blocked`] is the default and takes the table when it is the
/// cheaper factorization. [`Traversal::Tabulated`] takes it wherever one fits,
/// whether or not the op count says it wins: `CD-13` needs that to compare bytes
/// on both sides of the predicate, and a caller measuring its own shapes needs it
/// for the same reason.
fn admits(
    traversal: Traversal,
    code_space: usize,
    block: usize,
    plan: Plan,
    lane_bytes: usize,
) -> bool {
    match traversal {
        Traversal::OutputMajor => false,
        Traversal::Blocked => tabulation_pays(
            code_space,
            block,
            plan.cols,
            plan.rows,
            blocking::L1_BYTES,
            lane_bytes,
        ),
        Traversal::Tabulated => {
            tabulation_fits(code_space, plan.rows, blocking::L1_BYTES, lane_bytes)
        }
    }
}

/// Row counts the column loop is compiled for, widest first.
///
/// Sixteen 32-bit lane words are two 256-bit registers, and the entry they read is
/// one cache line. Which entry a call reaches is decided by how much of L1 the
/// table leaves, not by the length of this list.
///
/// Sixteen was measured *worse* than eight while the build read every entry once
/// per element of its codeword, and better once the build kept the entry in
/// registers: the wider tile halves the column steps, and it only costs stack
/// depth, which stopped mattering when the column loop got enough loads in flight
/// to hide the latency. Every figure is in ANALYSIS.md.
///
/// Every entry computes the same integer. A shape that does not divide walks down
/// the list; it does not take a different path, and `CD-13` asserts the bytes at
/// every `m`.
const ROW_TILES: [usize; 5] = [16, 8, 4, 2, 1];

/// Decode the whole codebook once, transposed, into the caller's panel offer.
///
/// `book[index * block + t]` is element `t` of the `index`-th codeword, so one
/// codeword is a contiguous run and the build walks it against the activation
/// tile without a stride. The codec is consulted `code_space * block` times for
/// the *whole call* rather than once per row tile and per block of the reduction
/// --- measured, re-deriving it per tile was half the build.
///
/// This is the only reason the tabulated traversal wants a panel offer at all.
/// Without one there is nowhere to put the decoded book, and the traversal
/// streams --- the same rule every other offer in this library follows.
fn decode_book<E, Bd, C, Lg>(codec: &C, panel: &mut [Alphabet<E, Bd>], ledger: &mut Lg) -> bool
where
    E: IntegerElement,
    Bd: Bound,
    C: Enumerable<E, Bd>,
    Lg: Ledger,
{
    let space = C::CODE_SPACE;
    let block = C::MAX_BLOCK;
    if panel.len() < space * block {
        return false;
    }
    for index in 0..space {
        for t in 0..block {
            panel[index * block + t] = codec.decode_element(C::code_at(index), t);
        }
    }
    ledger.decoded((space * block) as u64);
    true
}

/// How much panel offer the decoded codebook occupies.
///
/// A *query*, like [`crate::suggested_scratch`]. Offering less selects the
/// streaming traversal, at the same bytes (`CD-13`).
pub const fn suggested_tabulation_panel(code_space: usize, block: usize) -> usize {
    // The decoded codebook, plus the widest activation tile a row tile can pack.
    code_space
        .saturating_add(ROW_TILES[0])
        .saturating_mul(block)
}

/// Output columns reduced together in one pass over the stack.
///
/// The table does not fit L1 once the stack is deep enough to keep the exact
/// accumulator out of the loop, so every entry read is an L2 hit and the only
/// question is whether it is *overlapped*. One column at a time gives the machine
/// one load to work on; `U` columns give it `U`, and they are independent because
/// their codes are independent. The lane state is `U * R` words --- four 256-bit
/// registers at the shipped tile --- so nothing spills to make room for it.
const COLUMN_GROUP: usize = 2;

/// `U` output columns, one chunk of the reduction, at a compile-time row count.
///
/// Everything is walked rather than indexed: the stack is one slab per block, the
/// codes of each column are a contiguous run, and the entry's offset is a shift.
/// That leaves the step with one load, one mask, one add, and `R` adds --- no
/// multiply, no bounds check, and no loop framing.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn columns<const R: usize, const U: usize, E, Bd, C, L>(
    stack: &[L],
    slab: usize,
    codes: &[C::Code],
    codes_per_row: usize,
    col: usize,
    p0: usize,
    depth: usize,
    acc: &mut [AccOf<E>],
) where
    E: IntegerElement,
    Bd: Bound,
    C: Enumerable<E, Bd>,
    L: Lane<E>,
{
    let mut base = [0usize; U];
    for (u, at) in base.iter_mut().enumerate() {
        *at = (col + u) * codes_per_row + p0;
    }
    let mut lane = [[L::ZERO; R]; U];
    for (slot, words) in stack.chunks_exact(slab).take(depth).enumerate() {
        for u in 0..U {
            let at = C::index_of(codes[base[u] + slot]) * R;
            if let Some(entry) = words.get(at..).and_then(<[L]>::first_chunk::<R>) {
                add_entry(entry, &mut lane[u]);
            }
        }
    }
    // The placement, once for the whole chunk and all `U` columns.
    for (cell, word) in acc.iter_mut().zip(lane.iter().flatten()) {
        *cell = word.place(*cell);
    }
}

/// One row tile, at a compile-time row count.
///
/// `R` is what makes the column loop a *register* loop. A `rows`-long slice of
/// unknown length compiles to a bounds check, a scalar prologue and a vector
/// epilogue around eight adds; measured, that framing cost more than the adds by
/// an order of magnitude. With `R` known, the whole chunk's accumulation is `R`
/// registers and the table read is one contiguous `R`-word load.
#[allow(clippy::too_many_arguments)]
fn row_tile<const R: usize, E, Bd, C, O, Ep, L, Lg>(
    triple: &mut TabulatedTriple<'_, '_, '_, E, Bd, C, O>,
    epilogue: &Ep,
    options: GemmOptions,
    exact: &mut [AccOf<E>],
    lanes: &mut [L],
    panel: &mut [Alphabet<E, Bd>],
    plan: Plan,
    row0: usize,
    ledger: &mut Lg,
) where
    E: IntegerElement,
    Bd: Bound,
    C: Enumerable<E, Bd>,
    O: Element + EncodeFrom<AccOf<E>>,
    Ep: Epilogue<E, O>,
    L: Lane<E>,
    Lg: Ledger,
{
    let shape = triple.shape();
    let block = <C as uor_matmul_codec::Codec<E, Bd>>::MAX_BLOCK;
    let blocks = shape.k / block;
    let reads_c = epilogue.reads_c();
    let codes_per_row = triple.w.codes_per_row();
    let codes = triple.w.codes();

    // The stack is shaped to *this* tile, so its stride is `R` and its slab is
    // `CODE_SPACE * R` --- both compile-time. That is what turns the step's
    // address arithmetic into a shift and lets the bounds of the entry be proved
    // rather than checked: `index < CODE_SPACE`, so `index * R + R <= slab`.
    let space = C::CODE_SPACE;
    let slab = space * R;
    let Some(mut table) = Table::new(lanes, space, R, plan.depth) else {
        // `Plan::choose` sized the lane offer for the widest tile it admits and
        // `R` is never wider, so this cannot be reached. It is written as the
        // streaming traversal rather than as an assertion because an unreachable
        // branch that could produce no output at all is worse than one that
        // produces the right output slowly (C6, R14).
        stream(triple, epilogue, options, panel, ledger);
        return;
    };
    let (book, acts) = panel.split_at_mut(space * block);

    let mut col0 = 0usize;
    while col0 < shape.n {
        let cols = plan.cols.min(shape.n - col0);
        let acc = &mut exact[..R * cols];
        acc.fill(<AccOf<E> as Accumulator>::ZERO);

        let mut p0 = 0usize;
        while p0 < blocks {
            let depth = plan.depth.min(blocks - p0);
            for slot in 0..depth {
                // The activation tile for this block, packed once: `R` rows by
                // `MAX_BLOCK` steps, read from `A`'s own strides here and walked
                // contiguously inside the build.
                let base = (p0 + slot) * block;
                for t in 0..block {
                    for i in 0..R {
                        acts[t * R + i] = *triple.a.at(row0 + i, base + t);
                    }
                }
                table.build::<R, E, Bd, C, Lg>(book, &acts[..block * R], slot, ledger);
            }

            // No multiply below this line, and no exact accumulator either:
            // `depth` blocks reduce into `R` lane words held in registers, and
            // the placement happens once for all of them.
            //
            // Everything the step needs is walked rather than indexed: the codes
            // of one column are a contiguous run, the stack is one slab per
            // block, and the output tile is one chunk per column. That leaves the
            // step with one load, one mask, one add of the entry's base, and `R`
            // adds --- no multiply, no bounds check, and no loop framing.
            let stack = table.stack();
            let mut j = 0usize;
            while j + COLUMN_GROUP <= cols {
                columns::<R, COLUMN_GROUP, E, Bd, C, L>(
                    stack,
                    slab,
                    codes,
                    codes_per_row,
                    col0 + j,
                    p0,
                    depth,
                    &mut acc[j * R..],
                );
                j += COLUMN_GROUP;
            }
            while j < cols {
                columns::<R, 1, E, Bd, C, L>(
                    stack,
                    slab,
                    codes,
                    codes_per_row,
                    col0 + j,
                    p0,
                    depth,
                    &mut acc[j * R..],
                );
                j += 1;
            }
            ledger.read((cols * depth * R) as u64);
            ledger.added((cols * depth * R) as u64);
            p0 += depth;
        }

        // The single encode step, exactly once per output element.
        for i in 0..R {
            for j in 0..cols {
                let (r, c) = (row0 + i, col0 + j);
                let prior = if reads_c {
                    Some(*triple.c.at(r, c))
                } else {
                    None
                };
                *triple.c.at_mut(r, c) = epilogue.finish(acc[j * R + i], prior, options.encode);
            }
        }
        col0 += cols;
    }
}

#[allow(clippy::too_many_arguments)]
fn tabulate<E, Bd, C, O, Ep, L, Lg>(
    triple: &mut TabulatedTriple<'_, '_, '_, E, Bd, C, O>,
    epilogue: &Ep,
    options: GemmOptions,
    exact: &mut [AccOf<E>],
    lanes: &mut [L],
    panel: &mut [Alphabet<E, Bd>],
    plan: Plan,
    ledger: &mut Lg,
) where
    E: IntegerElement,
    Bd: Bound,
    C: Enumerable<E, Bd>,
    O: Element + EncodeFrom<AccOf<E>>,
    Ep: Epilogue<E, O>,
    L: Lane<E>,
    Lg: Ledger,
{
    let shape = triple.shape();

    let mut row0 = 0usize;
    while row0 < shape.m {
        // The widest compiled tile the plan and the remaining rows admit. A shape
        // that does not divide walks down the list; it does not take a different
        // path, and `CD-13` asserts the bytes at every `m`.
        let rows = ROW_TILES
            .into_iter()
            .find(|&r| r <= plan.rows && r <= shape.m - row0)
            .unwrap_or(1);
        match rows {
            16 => row_tile::<16, E, Bd, C, O, Ep, L, Lg>(
                triple, epilogue, options, exact, lanes, panel, plan, row0, ledger,
            ),
            8 => row_tile::<8, E, Bd, C, O, Ep, L, Lg>(
                triple, epilogue, options, exact, lanes, panel, plan, row0, ledger,
            ),
            4 => row_tile::<4, E, Bd, C, O, Ep, L, Lg>(
                triple, epilogue, options, exact, lanes, panel, plan, row0, ledger,
            ),
            2 => row_tile::<2, E, Bd, C, O, Ep, L, Lg>(
                triple, epilogue, options, exact, lanes, panel, plan, row0, ledger,
            ),
            _ => row_tile::<1, E, Bd, C, O, Ep, L, Lg>(
                triple, epilogue, options, exact, lanes, panel, plan, row0, ledger,
            ),
        }
        row0 += rows;
    }
}

/// The same identity with no table: decode, accumulate exactly, encode once.
///
/// Not a fallback. It is [`Traversal::OutputMajor`] for this operand orientation,
/// it needs no offer at all, and `CD-13` asserts it produces the same bytes as the
/// table does. A caller on a target whose RAM cannot hold a table gets this and
/// loses nothing but time.
fn stream<E, Bd, C, O, Ep, Lg>(
    triple: &mut TabulatedTriple<'_, '_, '_, E, Bd, C, O>,
    epilogue: &Ep,
    options: GemmOptions,
    panel: &mut [Alphabet<E, Bd>],
    ledger: &mut Lg,
) where
    E: IntegerElement,
    Bd: Bound,
    C: Enumerable<E, Bd>,
    O: Element + EncodeFrom<AccOf<E>>,
    Ep: Epilogue<E, O>,
    Lg: Ledger,
{
    let shape = triple.shape();
    let reads_c = epilogue.reads_c();
    // One decoded weight row, if the offer holds one. `W` is `k`-major, so a
    // decoded row is a contiguous run of the reduction and every row of `A` reads
    // it --- which turns the inner loop into [`dot_ref`]'s narrow-lane dot product
    // and decodes each weight once instead of once per row of `A`.
    let borrowed = panel.len() >= shape.k;
    for j in 0..shape.n {
        if borrowed {
            triple.w.decode_row_into(j, panel);
            ledger.decoded(shape.k as u64);
        }
        for i in 0..shape.m {
            let acc = if borrowed {
                match triple.a.row_block(i, 0, 1, shape.k) {
                    // Both operands are runs: the same accumulation, in the
                    // narrowest lane its depth admits, with nothing to walk.
                    Some(row) => dot_ref(row, &panel[..shape.k]),
                    None => {
                        let mut acc = <AccOf<E> as Accumulator>::ZERO;
                        for (p, w) in panel[..shape.k].iter().enumerate() {
                            E::mac(&mut acc, triple.a.at(i, p).get(), w.get());
                        }
                        acc
                    }
                }
            } else {
                // No offer at all: decode one element at a time, which is what
                // makes this runnable on a target whose RAM cannot hold a row.
                let mut acc = <AccOf<E> as Accumulator>::ZERO;
                for p in 0..shape.k {
                    E::mac(&mut acc, triple.a.at(i, p).get(), triple.w.at(j, p).get());
                }
                ledger.decoded(shape.k as u64);
                acc
            };
            ledger.multiplied(shape.k as u64);
            let prior = if reads_c {
                Some(*triple.c.at(i, j))
            } else {
                None
            };
            *triple.c.at_mut(i, j) = epilogue.finish(acc, prior, options.encode);
        }
    }
}

#[cfg(test)]
// R7 governs the library, not its tests: these build operands on the heap so
// that awkward shapes and whole code spaces can be generated. `CA-01` witnesses
// the library's own zero-allocation claim with a counting allocator instead.
#[allow(clippy::disallowed_types)]
mod tests {
    use super::*;
    use crate::driver::gemm;
    use crate::epilogue::Linear;
    use std::vec;
    use std::vec::Vec;
    use uor_matmul_codec::{e8_codec, e8_table, Grid, Packed};
    use uor_matmul_core::{as_alphabet_full, EncodeMode, Full, Triple};

    type A8 = Alphabet<i8, Full<i8>>;

    /// A recorded generator, so any failure reproduces from the seed alone.
    fn fill<T, F: Fn(u64) -> T>(len: usize, salt: u64, map: F) -> Vec<T> {
        let mut s = 0x243F_6A88_85A3_08D3u64 ^ salt;
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                map(s >> 33)
            })
            .collect()
    }

    fn options(traversal: Traversal) -> GemmOptions {
        GemmOptions {
            traversal,
            encode: EncodeMode::Wrapping,
            ..Default::default()
        }
    }

    /// The dense product of the same operands, by the driver whose bytes every
    /// other traversal is measured against.
    ///
    /// `W` is decoded into a `k x n` dense matrix first, so this is the identity
    /// tabulation claims to compute and not a restatement of it.
    fn reference<C: Enumerable<i8, Full<i8>> + Copy>(
        w: &CodedMatrix<'_, i8, Full<i8>, C>,
        a: &[i8],
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<i32> {
        let mut b = vec![0i8; k * n];
        for p in 0..k {
            for j in 0..n {
                b[p * n + j] = w.at(j, p).get();
            }
        }
        let mut c = vec![0i32; m * n];
        let av = MatView::row_major(as_alphabet_full(a), m, k).unwrap();
        let bv = MatView::row_major(as_alphabet_full(&b), k, n).unwrap();
        let cv = MatViewMut::row_major(&mut c, m, n).unwrap();
        let mut t = Triple::new(av, bv, cv).unwrap();
        gemm(
            &mut t,
            &Linear::OVERWRITE,
            options(Traversal::Blocked),
            &mut Scratch::none(),
        );
        c
    }

    /// One tabulated product at one traversal and one pair of offers.
    ///
    /// The offers scale together, because a caller who halves one halves the
    /// other; `CD-13` sweeps the fraction rather than the two independently.
    fn tabulated<C: Enumerable<i8, Full<i8>> + Copy>(
        w: &CodedMatrix<'_, i8, Full<i8>, C>,
        a: &[i8],
        m: usize,
        n: usize,
        traversal: Traversal,
        offer: usize,
    ) -> (Vec<i32>, Census) {
        let k = w.cols();
        let shape = Shape { m, k, n };
        let block = <C as uor_matmul_codec::Codec<i8, Full<i8>>>::MAX_BLOCK;
        let want_acc = suggested_tabulation::<i8, Full<i8>>(shape, C::CODE_SPACE, block).max(1);
        let want_lanes =
            suggested_tabulation_lanes::<i8, Full<i8>>(shape, C::CODE_SPACE, block).max(1);
        // `offer` is a numerator over the suggested amount, so one knob sweeps
        // both buffers and the extremes -- nothing, one word, exactly enough --
        // are all reachable.
        let scale = |want: usize| -> usize {
            if offer >= OFFER_STEPS {
                want.saturating_mul(offer - OFFER_STEPS + 1)
            } else {
                want * offer / OFFER_STEPS
            }
        };
        let mut accumulators = vec![<AccOf<i8> as Accumulator>::ZERO; scale(want_acc)];
        let mut lane_words = vec![0i64; scale(want_lanes)];
        let mut panel = vec![A8::ZERO; scale(suggested_tabulation_panel(C::CODE_SPACE, block))];
        let mut c = vec![0i32; m * n];
        let mut census = Census::default();
        {
            let av = MatView::row_major(as_alphabet_full(a), m, k).unwrap();
            let cv = MatViewMut::row_major(&mut c, m, n).unwrap();
            let mut tr = TabulatedTriple::new(av, *w, cv).unwrap();
            gemm_tabulated_counted(
                &mut tr,
                &Linear::OVERWRITE,
                options(traversal),
                &mut Scratch::with_accumulators(&mut panel, &mut accumulators),
                &mut Tabulation::new(&mut lane_words),
                &mut census,
            );
        }
        (c, census)
    }

    /// Fractions of the suggested offer that `CD-13` sweeps. `OFFER_STEPS` is
    /// "exactly the suggested amount"; below it is a fraction, above it a
    /// multiple.
    const OFFER_STEPS: usize = 8;

    /// Every traversal at every offer, against the dense driver's bytes.
    fn every_traversal_agrees<C: Enumerable<i8, Full<i8>> + Copy>(
        label: &str,
        codec: C,
        stream: &[C::Code],
        m: usize,
        k: usize,
        n: usize,
    ) {
        let w = CodedMatrix::new(codec, n, k, stream).expect("the codes describe n x k");
        let a: Vec<i8> = fill(m * k, 0xa11, |x| ((x % 255) as i64 - 127) as i8);
        let want = reference(&w, &a, m, k, n);

        // Nothing, a sliver, most of it, exactly it, and three times it.
        let offers = [0, 1, 2, OFFER_STEPS - 1, OFFER_STEPS, OFFER_STEPS + 2];
        for traversal in [
            Traversal::Tabulated,
            Traversal::Blocked,
            Traversal::OutputMajor,
        ] {
            for offer in offers {
                let (got, census) = tabulated(&w, &a, m, n, traversal, offer);
                assert_eq!(
                    got, want,
                    "{label} {m}x{k}x{n}: {traversal:?} at an offer of {offer} \
                     must give the dense driver's bytes ({census:?})"
                );
            }
        }

        // And the comparison is not vacuous: the table really was reached at the
        // offer sized for it, and really was not at an offer of nothing.
        let (_, with) = tabulated(&w, &a, m, n, Traversal::Tabulated, OFFER_STEPS);
        let (_, without) = tabulated(&w, &a, m, n, Traversal::Tabulated, 0);
        assert!(
            with.table_reads > 0,
            "{label} {m}x{k}x{n}: the offer was sized for a table and none was read"
        );
        assert_eq!(
            without.table_reads, 0,
            "{label} {m}x{k}x{n}: an offer of nothing cannot read a table"
        );
    }

    /// `CD-13`: `Tabulated`, `Blocked` and `OutputMajor` produce byte-identical
    /// output at every shape, on both sides of `tabulation_pays`, at every offer.
    ///
    /// The reference is the *dense* driver over the decoded weights, not another
    /// tabulated run: an agreement between two tabulations would say nothing
    /// about whether either computes the product.
    #[test]
    fn every_traversal_gives_the_same_bytes_cd_13() {
        let table = e8_table::<Full<i8>>().expect("i8 holds E8");
        let book = e8_codec(&table);
        // Codes past the codebook's 256 entries on purpose: the enumeration is
        // total on `u16`, so these are live codes and not invalid ones.
        for &(m, k, n) in &[
            (1usize, 8usize, 1usize),
            (4, 16, 8),
            (5, 24, 320),
            (8, 32, 512),
            (3, 8, 293),
            (3, 8, 292),
            (7, 40, 7),
        ] {
            let stream: Vec<u16> = fill(n * (k / 8), 0xb00c, |x| (x % 400) as u16);
            every_traversal_agrees("Book<256,8>", book, &stream, m, k, n);
        }

        let i4: [A8; 16] = core::array::from_fn(|i| Alphabet::of((i as i8) - 8));
        let grid = Grid::<i8, Full<i8>, 16>::new(&i4);
        let packed = Packed::<_, 2>::new(grid).expect("2 divides 8");
        for &(m, k, n) in &[(1usize, 2usize, 1usize), (4, 6, 600), (6, 10, 13)] {
            let stream: Vec<u8> = fill(n * (k / 2), 0xd0e, |x| x as u8);
            every_traversal_agrees("Packed<Grid<16>,2>", packed, &stream, m, k, n);
        }
    }

    /// `CT-07`: tabulation is total. Every value the code type can hold indexes a
    /// live table entry, so no code stream can produce a miss or a panic.
    ///
    /// The stream below is *every* `u16`, in order, including the 65280 values
    /// past the codebook's entry count. A codec whose enumeration did not cover
    /// its code type would read off the end of the table here.
    #[test]
    fn every_code_indexes_a_live_entry_ct_07() {
        let table = e8_table::<Full<i8>>().expect("i8 holds E8");
        let book = e8_codec(&table);
        let (k, n) = (8usize, 1usize << 16);
        // One coded row per output column, one code per row: the whole code type.
        let stream: Vec<u16> = (0..=u16::MAX).collect();
        let w = CodedMatrix::new(book, n, k, &stream).expect("65536 rows of one code");
        let a: Vec<i8> = fill(k, 0xfeed, |x| ((x % 255) as i64 - 127) as i8);

        let (got, census) = tabulated(&w, &a, 1, n, Traversal::Tabulated, OFFER_STEPS);
        assert!(census.table_reads > 0, "the table must have been read");
        assert_eq!(got, reference(&w, &a, 1, k, n));

        // The two halves of the code space decode alike, because the enumeration
        // is `code % 256` --- so the answer repeats with period 256 and a table
        // that had indexed the raw code would have gone out of bounds at 256.
        for j in 0..256usize {
            assert_eq!(
                got[j],
                got[j + 256],
                "code {j} and code {} decode alike and must accumulate alike",
                j + 256
            );
        }
    }

    /// `CU-06`: the tabulated column loop issues no multiply, counted.
    ///
    /// Three numbers say it, and each is a closed form rather than a fitted one:
    ///
    /// - `adds == table_reads == m * n * (k / Bk)`. That is the *whole* content of
    ///   the column loop: one read and one add per code, covering `Bk` weights.
    /// - `decodes == code_space * Bk`, for the whole call. The codebook is
    ///   decoded once, not once per row tile and per block of the reduction, so
    ///   the codec's cost does not scale with the shape at all.
    /// - `multiplies == m * k * code_space`, independent of `n`. The build is the
    ///   only arithmetic that scales with the code space, and it does not scale
    ///   with the output width at all.
    #[test]
    fn the_tabulated_column_loop_has_no_multiply_cu_06() {
        let table = e8_table::<Full<i8>>().expect("i8 holds E8");
        let book = e8_codec(&table);
        let space = 256usize;
        let block = 8usize;
        let (m, k, n) = (8usize, 32usize, 4096usize);
        let blocks = k / block;

        // The narrow lane, which is what `i8` resolves to: an `i64` word holds a
        // chunk of this reduction exactly, so the table is `i64` and twice as many
        // rows fit L1 as would in the exact accumulator.
        let rows = tabulation_rows(space, blocking::L1_BYTES, core::mem::size_of::<i64>()).min(m);
        assert!(
            rows >= 1 && m % rows == 0,
            "an exact tiling, so the closed forms are exact"
        );

        let stream: Vec<u16> = fill(n * blocks, 0xc0de, |x| (x % 256) as u16);
        let w = CodedMatrix::new(book, n, k, &stream).expect("the codes describe n x k");
        let a: Vec<i8> = fill(m * k, activation_salt(), |x| ((x % 255) as i64 - 127) as i8);

        let (got, census) = tabulated(&w, &a, m, n, Traversal::Blocked, OFFER_STEPS);
        assert_eq!(
            got,
            reference(&w, &a, m, k, n),
            "and it is still the product"
        );

        assert_eq!(
            census.adds,
            (m * n * blocks) as u64,
            "one add per code per row: {census:?}"
        );
        assert_eq!(
            census.table_reads, census.adds,
            "every table read is exactly one add and nothing else: {census:?}"
        );
        assert_eq!(
            census.decodes,
            (space * block) as u64,
            "the codebook is decoded once for the whole call, not once per tile: {census:?}"
        );
        assert_eq!(
            census.multiplies,
            (m * k * space) as u64,
            "the build is `m * k * code_space` and does not scale with n: {census:?}"
        );

        // The claim the whole construction exists for, in the spec's own
        // accounting: a read and its add are one operation per code.
        let dense = (m * k * n) as u64;
        let tabulated_ops = census.multiplies + census.adds;
        assert!(
            tabulated_ops * 4 <= dense,
            "at {m}x{k}x{n} tabulation must issue at least four times fewer operations \
             than the dense traversal: {tabulated_ops} against {dense}"
        );
        assert!(
            census.multiplies * 8 <= dense,
            "and at least eight times fewer multiplies: {} against {dense}",
            census.multiplies
        );

        // The streaming traversal is the same identity and issues the products
        // themselves, which is what makes the comparison above mean anything.
        let (streamed, plain) = tabulated(&w, &a, m, n, Traversal::OutputMajor, OFFER_STEPS);
        assert_eq!(streamed, got);
        assert_eq!(plain.multiplies, dense);
        assert_eq!(plain.table_reads, 0);
    }

    /// The salt for the activation generator, named so the seed is recorded next
    /// to the test rather than buried in an argument list.
    fn activation_salt() -> u64 {
        0x5a17
    }

    /// The selection predicate is a derivation, and this is it recomputed.
    ///
    /// Part of `CM-04`'s claim lives here rather than in the model crate: that
    /// the shipped `const fn` and the model's recorded break-even are the same
    /// function of the same two numbers.
    #[test]
    fn the_predicate_is_the_derivation_cm_04() {
        // The lane `i8` resolves to, which is what the shipped row tile is sized
        // against.
        let lane = core::mem::size_of::<i32>();
        let l1 = blocking::L1_BYTES;
        let rows = ROW_TILES
            .into_iter()
            .find(|&r| r <= tabulation_rows(256, l1, lane))
            .expect("a 256-entry table fits L1 at some tile");
        let step = blocking::KERNEL_PRODUCTS_PER_STEP;

        // E8 at the shipped tile: the first `n` that pays is 683, and 682 does
        // not. The model records the same numeral and `CM-04`'s model half
        // recomputes it from the same three inputs.
        assert!(tabulation_pays(256, 8, 683, rows, l1, lane));
        assert!(!tabulation_pays(256, 8, 682, rows, l1, lane));
        // A nibble pair covers `2 * rows` products per lane operation, which is
        // exactly what one dense instruction covers, so nothing repays the build
        // and no `n` makes it pay.
        assert_eq!(2 * rows, step);
        assert!(!tabulation_pays(256, 2, usize::MAX, rows, l1, lane));
        // One element per code: likewise, and for the same reason at `block = 1`.
        assert!(!tabulation_pays(16, 1, usize::MAX, rows, l1, lane));
        // A table nobody can hold is refused whatever the instruction count says.
        assert!(!tabulation_pays(1 << 16, 8, usize::MAX, 1, l1, lane));
        assert_eq!(tabulation_rows(1 << 16, l1, lane), 0);
        // And an enumeration of nothing has no table.
        assert!(!tabulation_fits(0, 1, l1, lane));
    }
}
