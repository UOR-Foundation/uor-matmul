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

use uor_matmul_codec::{Addressing, Codec, CodedMatrix, Enumerable, TierId};
use uor_matmul_core::generated::blocking;
use uor_matmul_core::{
    AccOf, Accumulator, Alphabet, Bound, Element, EncodeFrom, EncodeMode, FloatElement, MatView,
    MatViewMut, NotAProduct, Shape, Traversal,
};

use uor_matmul_core::{Backend, Strides, Triple};
use uor_matmul_kernels::{
    available_table_i16, available_table_i32_modular, available_table_i64_modular,
    available_table_i8, choose_table, packed_slot, portable_table, Mod32, Mod64, Scaled64,
};

use crate::collapse::{compact, distinct_rows, expand, Collapse};
use crate::driver::GemmOptions;
use crate::epilogue::Epilogue;
use crate::float::{accumulate_atlas_dot, f32_q, gemm_float, PanelFacts, Span};
use crate::kernel::{gemm_packed, Kernelized};
use crate::scratch::Scratch;

// ---------------------------------------------------------------------------
// The census
// ---------------------------------------------------------------------------

/// What the traversal issued, counted rather than derived.
///
/// A wall-clock comparison measures the machine as much as the library, and a
/// fitted scaling exponent measures the traversal but not the arithmetic. A
/// census measures the transparent traversal bodies directly: it turns
/// "tabulation is faster" into "the tabulated column loop issues zero
/// multiplies and `m*k*n/Bk` adds", which is machine-independent, reproducible,
/// and assertable rather than reportable. A family-owned `dense_gemm` is an
/// opaque public extension point; this ledger records each presentation as a
/// [`Census::kernel_calls`] event and does not invent an internal operation count
/// from the matrix shape.
///
/// The authority for the shape is `uor-r4-core`'s `OpKernel`, which declares its
/// arithmetic interface as a census with no multiplication field. r4's census
/// also counts shifts and candidate scans; those are absent here because this
/// traversal issues neither, and a field that can only ever be zero would be a
/// claim about a mechanism that is not present.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Census {
    /// Widening multiply-accumulates issued by the transparent table build or
    /// `StreamLane` body. Zero in the column loop, by construction. Operations
    /// inside a family-owned dense call are represented by `kernel_calls`, not
    /// by a synthetic `m*k*n` charge.
    pub multiplies: u64,
    /// Transparent accumulator combines and table-build contraction charges.
    /// A spec whose body exposes a fixed expansion reports its actual adds (the
    /// bound-one/Gray builds). A call through the generic `Element::mac`
    /// boundary, including the variable occupied-extent q contraction, reports
    /// one contraction presentation per product; data-dependent additions
    /// inside that opaque algebra are verified by the element family's own
    /// observer/purity/differential gates and are not guessed from the shape.
    pub adds: u64,
    /// Reads of a tabulated partial sum.
    pub table_reads: u64,
    /// Calls into the codec's decode or into a contextual panel projection.
    ///
    /// Ordinary element panels are copied as their own values and contribute
    /// only the codec call. A contextual producer/consumer lane, such as the
    /// compact `f32` Atlas lane, first observes the source symbol to derive the
    /// call's finite grade gauge and exact envelope, then projects it into the
    /// in-place q cell after that gauge is known. Non-finite observations become
    /// singleton tags rather than a refusal. Both presentations are real,
    /// counted work; an offer-resident projection cache makes the second occur
    /// once per cached activation rather than once per column block.
    pub decodes: u64,
    /// Calls into a non-table kernel factorization: one for each decoded-operand
    /// driver call and one for each nonempty source page presented to the
    /// persistent dense stream. A declining presentation is still a call and
    /// is counted; an empty reduction stays in the public `StreamLane`, issues
    /// no product, and therefore calls no kernel.
    ///
    /// The three factorizations are told apart by this and by `table_reads`, so
    /// "which one ran" is something a harness reads rather than something it
    /// recomputes from the predicate and hopes agrees.
    pub kernel_calls: u64,
}

/// Somewhere to put the census, or nowhere.
///
/// One traversal, two instantiations. `()` implements this with empty bodies, so
/// the shipped call site counts nothing and the optimizer deletes the calls; a
/// `Census` counts. There is no second loop nest and no `cfg`, which is what
/// keeps the counted run and the shipped run the same function (R13).
pub trait Ledger {
    /// Record `n` observable widening multiply-accumulates.
    fn multiplied(&mut self, n: u64);
    /// Record `n` transparent combines or declared contraction presentations.
    fn added(&mut self, n: u64);
    /// Record `n` reads of a tabulated partial sum.
    fn read(&mut self, n: u64);
    /// Record `n` codec decodes.
    fn decoded(&mut self, n: u64);
    /// Record a call into a non-table kernel factorization.
    fn kernelled(&mut self);
}

impl Ledger for () {
    fn multiplied(&mut self, _: u64) {}
    fn added(&mut self, _: u64) {}
    fn read(&mut self, _: u64) {}
    fn decoded(&mut self, _: u64) {}
    fn kernelled(&mut self) {}
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
    fn kernelled(&mut self) {
        self.kernel_calls = self.kernel_calls.saturating_add(1); // R3-ok: a counter
    }
}

/// A census is bounded storage even when the operation it observes has more
/// than `u64::MAX` events. Convert each address-sized factor before combining
/// it so the query saturates instead of overflowing in `usize` first.
const fn count_factor(n: usize) -> u64 {
    if n as u128 > u64::MAX as u128 {
        u64::MAX
    } else {
        n as u64
    }
}

const fn count_product2(a: usize, b: usize) -> u64 {
    count_factor(a).saturating_mul(count_factor(b)) // R3-ok: a bounded diagnostic counter
}

const fn count_product3(a: usize, b: usize, c: usize) -> u64 {
    count_product2(a, b).saturating_mul(count_factor(c)) // R3-ok: a bounded diagnostic counter
}

const fn count_sum2(a: u64, b: u64) -> u64 {
    a.saturating_add(b) // R3-ok: a bounded diagnostic counter
}

// ---------------------------------------------------------------------------
// The lane
// ---------------------------------------------------------------------------

/// A word the table and the column accumulation are held in.
///
/// The exact accumulator is 128 bits wide or more, and a column loop that
/// touched one per output cell per block of the reduction would move a gigabyte
/// of accumulator through the cache to compute a quarter of a billion products.
///
/// A lane is where that stops, and it stops completely: one table entry is a sum
/// of `MAX_BLOCK` products of two alphabet elements, a whole reduction is a sum
/// of those, and [`Lane::capacity`] says how many of them one narrow word holds
/// exactly. At `(i8, 128)` that is 133144 products --- past every depth a weight
/// row reaches --- so the exact accumulator is touched **once per output
/// element**, which is what "encode once" already said and what the traversal
/// did not do. Measured, taking it out of the reduction was worth 1.94x.
///
/// The definitions live in [`uor_matmul_kernels`], with the sequences that read
/// them. That is not tidiness: a lane's whole purpose is to be added to in one
/// instruction, and the only crate in this workspace permitted `#[target_feature]`
/// is that one. Written here, the column loop compiled at the target's baseline
/// and issued no vector instruction at all --- 17.6 Gmac/s where the same
/// traversal in vectors measures 86.7.
pub use uor_matmul_kernels::{
    gather_reference_i32, gather_reference_wide, Lane, LaneWord, TableSpec, Wide,
};

/// One dot product, in the narrowest lane its depth admits.
///
/// The same value [`dot_ref`] computes and, for a quantized alphabet, a very
/// different number of instructions: `dot_ref` accumulates in the 64-bit lane, and
/// x86 has no vector multiply that wide, so its inner loop is scalar. The 32-bit
/// lane holds `capacity` products of the declared alphabet exactly --- 131072 of
/// them at `(i8, 128)`, which is past every depth a weight row reaches --- and
/// eight of them fit one register.
///
/// `run` is that capacity. The reduction is cut into runs of it and each run is
/// placed into the exact accumulator once, which is the same chunking
/// [`uor_matmul_core::fits_narrow`] already licenses for the tile kernels. A
/// downstream lane may honestly answer `Some(0)` when even one product exceeds
/// it; that declaration routes the same products directly into the exact
/// accumulator and the lane is never invoked.
#[inline]
fn dot_lane<E, Bd, L>(a: &[Alphabet<E, Bd>], w: &[Alphabet<E, Bd>], run: usize) -> AccOf<E>
where
    E: Element,
    Bd: Bound,
    L: Lane<E>,
{
    let mut acc = <AccOf<E> as Accumulator>::ZERO;
    if run == 0 {
        for (&x, &y) in a.iter().zip(w) {
            E::mac(&mut acc, x.get(), y.get());
        }
        return acc;
    }
    for (ra, rw) in a.chunks(run).zip(w.chunks(run)) {
        let mut lane = L::ZERO;
        for (&x, &y) in ra.iter().zip(rw) {
            lane = lane.mac(x.get(), y.get());
        }
        acc = lane.place(acc);
    }
    acc
}

/// One dot over views whose cells cannot be borrowed as two contiguous runs.
///
/// `StreamLane` is still the arithmetic whenever it admits a product: only the
/// source of each pair changes. Cutting at the lane's declared capacity is
/// identical to [`dot_lane`]. At the public declaration `Some(0)`, the exact
/// accumulator is the only carrier that exists, as in `dot_lane`; no zero-sized
/// run or lane call is constructed.
#[inline]
fn dot_walk<E, L, F>(depth: usize, run: usize, mut pair: F) -> AccOf<E>
where
    E: Element,
    L: Lane<E>,
    F: FnMut(usize) -> (E, E),
{
    let mut acc = <AccOf<E> as Accumulator>::ZERO;
    if run == 0 {
        for p in 0..depth {
            let (a, w) = pair(p);
            E::mac(&mut acc, a, w);
        }
        return acc;
    }
    let mut start = 0usize;
    while start < depth {
        let end = start + run.min(depth - start);
        let mut lane = L::ZERO;
        for p in start..end {
            let (a, w) = pair(p);
            lane = lane.mac(a, w);
        }
        acc = lane.place(acc);
        start = end;
    }
    acc
}

// ---------------------------------------------------------------------------
// The table
// ---------------------------------------------------------------------------

/// The tabulation buffer for one row tile and a chunk of the reduction.
///
/// Borrowed, never owned: it lives in the caller's offer like every other working
/// buffer in this library (R7, S13). `depth * codes * rows` lane words, indexed
/// block-major then code-major, so one code's entry is a contiguous run of `rows`
/// words that one vector load reaches without a stride.
///
/// # The slab is a power of two
///
/// `codes` is [`Enumerable::CODE_SPACE`] rounded **up** to a power of two, and
/// the column loop reads `stack[slot * slab + (index & (codes - 1)) * rows]`.
///
/// The mask is what discharges the read's bound. `index_of` is total below
/// `CODE_SPACE` --- that is the trait's law and `CK-09` asserts it --- so masking
/// changes no value the traversal can reach; but it makes every read in-slab
/// *whatever* the index holds, so there is no comparison and no branch in the
/// step. The entries between `CODE_SPACE` and `codes` are zeroed once and never
/// written, so an enumeration whose space is not a power of two costs padding and
/// never a special case.
///
/// # The depth is not what keeps the accumulator out of the loop
///
/// It was, and that was the error. A narrow lane holds the *whole* reduction ---
/// 133144 products at `(i8, 128)` against a `k` of 1024 --- so the column
/// accumulation is carried across every chunk and the exact accumulator is
/// touched once per output element. The depth is now only what keeps the stack
/// cache-resident, which is the question it should always have been answering.
#[derive(Debug)]
pub struct Table<'s, L> {
    words: &'s mut [L],
    code_space: usize,
    codes: usize,
    rows: usize,
    depth: usize,
}

/// Codes one slab addresses: [`Enumerable::CODE_SPACE`] rounded up to a power of
/// two, so the index needs a mask and never a comparison. Zero means no
/// address-sized power of two contains the requested space (or the space itself
/// is empty), and therefore no masked slab exists.
///
/// A free function, not an associated one, because it is a fact about the
/// enumeration and not about the lane it is tabulated in --- naming a lane to
/// ask it would be naming something the answer does not depend on.
pub const fn slab_codes(code_space: usize) -> usize {
    if code_space == 0 {
        return 0;
    }
    // Zero is the total answer when no address-sized power of two can contain
    // the enumeration. Such a slab does not exist, and `Table::new` declines
    // it before either sizing arithmetic or a masked read can be reached.
    match code_space.checked_next_power_of_two() {
        Some(codes) => codes,
        None => 0,
    }
}

/// How many lane words a stack of `depth` tables over `rows` rows of a
/// `code_space`-wide enumeration occupies.
///
/// A query, so an embedded caller can size a static and know the answer before
/// it calls anything.
pub const fn table_words(code_space: usize, rows: usize, depth: usize) -> usize {
    let codes = slab_codes(code_space);
    if code_space != 0 && rows != 0 && depth != 0 && codes == 0 {
        return usize::MAX;
    }
    codes
        .saturating_mul(rows) // R3-ok: a size query, not an accumulation
        .saturating_mul(depth) // R3-ok: a size query, not an accumulation
}

impl<'s, L: LaneWord> Table<'s, L> {
    /// Borrow `words` as a stack of `depth` tables, with the padding zeroed.
    ///
    /// `None` when the borrow is shorter than the table it is asked to be, which
    /// means no such table exists in that offer. Decided here, before any
    /// arithmetic, and answered by the caller taking the streaming traversal
    /// instead --- not by an error reaching anyone (C6).
    pub fn new(words: &'s mut [L], code_space: usize, rows: usize, depth: usize) -> Option<Self> {
        let table = Self::borrow(words, code_space, rows, depth)?;
        let slab = table.slab();
        let live = code_space.checked_mul(rows)?;
        // Public construction promises readable zero padding, including to a
        // caller inspecting `stack()` or using a masked safe gather offset.
        for slot in 0..depth {
            table.words[slot * slab + live..slot * slab + slab].fill(L::ZERO);
        }
        Some(table)
    }

    /// Reborrow a stack whose padding was zeroed by [`Self::new`] at this exact
    /// geometry and whose intervening builds wrote only its live entries.
    ///
    /// Private because the proof is the enclosing row walk: consecutive full
    /// row tiles reuse the same `(code_space, rows, depth)`. A narrower edge
    /// tile has different slab boundaries and therefore returns to `new`.
    fn reuse_zeroed(
        words: &'s mut [L],
        code_space: usize,
        rows: usize,
        depth: usize,
    ) -> Option<Self> {
        Self::borrow(words, code_space, rows, depth)
    }

    fn borrow(words: &'s mut [L], code_space: usize, rows: usize, depth: usize) -> Option<Self> {
        if code_space == 0 || rows == 0 || depth == 0 {
            return None;
        }
        let codes = slab_codes(code_space);
        if codes == 0 {
            return None;
        }
        let slab = codes.checked_mul(rows)?;
        let want = slab.checked_mul(depth)?;
        if words.len() < want {
            return None;
        }
        let table = Self {
            words: &mut words[..want],
            code_space,
            codes,
            rows,
            depth,
        };
        Some(table)
    }

    /// Distinct codes the enumeration has.
    pub const fn code_space(&self) -> usize {
        self.code_space
    }

    /// Codes one slab addresses, which is [`Self::code_space`] rounded up.
    pub const fn slab_codes(&self) -> usize {
        self.codes
    }

    /// Rows of `A` the stack covers.
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Blocks of the reduction the stack holds at once.
    pub const fn depth(&self) -> usize {
        self.depth
    }

    /// Words per block held: one entry per addressable code, one word per row.
    pub const fn slab(&self) -> usize {
        self.codes * self.rows
    }

    /// Every entry of one block of the reduction, into stack slot `slot`.
    ///
    /// `code_space * block * rows` products, and the only multiplies the
    /// tabulated traversal issues at all --- unless the spec's build declares
    /// it does not multiply: at bound 1 every product is `+-a` or `0`, the
    /// build is adds and subtracts, and the census is charged what was issued
    /// (`CB-10`).
    ///
    /// The sequence is the spec's. The reduction's step is the inner loop and
    /// the code space the outer, so one entry's `rows` words stay in registers
    /// for its whole codeword and are written once; the other order reads and
    /// writes every entry once per element of the block, which is `MAX_BLOCK`
    /// times the traffic and measured half the traversal.
    pub fn build<E, Lg>(
        &mut self,
        spec: &TableSpec<E, L>,
        block: usize,
        book: &[E],
        acts: &[E],
        slot: usize,
        ledger: &mut Lg,
    ) where
        E: Element,
        L: Lane<E>,
        Lg: Ledger,
    {
        let slab = self.slab();
        let live = self.code_space * self.rows;
        let start = slot * slab;
        spec.build(
            self.code_space,
            block,
            book,
            acts,
            &mut self.words[start..start + live],
        );
        Self::charge_build(spec, self.code_space, block, self.rows, ledger);
    }

    /// Build one addressed entry directly in its resident slab cell.
    ///
    /// `TableBuild` is pointwise by contract: a one-entry book and one-entry
    /// output describe the same `T[c]` after the book pointer is advanced to
    /// `c`. The sign Gray walk has an additional whole-enumeration precondition
    /// and is excluded by its codec declaration at the call site.
    #[allow(clippy::too_many_arguments)] // one pointwise presentation of the public TableBuild protocol
    fn build_entry<E, Lg>(
        &mut self,
        spec: &TableSpec<E, L>,
        block: usize,
        book: &[E],
        acts: &[E],
        slot: usize,
        index: usize,
        ledger: &mut Lg,
    ) where
        E: Element,
        L: Lane<E>,
        Lg: Ledger,
    {
        debug_assert!(index < self.code_space);
        let slab = self.slab();
        let book_start = index * block;
        let entry_start = slot * slab + index * self.rows;
        spec.build(
            1,
            block,
            &book[book_start..book_start + block],
            acts,
            &mut self.words[entry_start..entry_start + self.rows],
        );
        Self::charge_build(spec, 1, block, self.rows, ledger);
    }

    /// Build one scalar coordinate of one addressed codec entry.
    ///
    /// The resident slab keeps the original enumeration coordinate, while the
    /// scalar spec sees a one-cell book and a block of one. This is the private
    /// fracture of the same pointwise `TableBuild` identity, not a second table
    /// representation.
    #[allow(clippy::too_many_arguments)] // one scalar presentation of the locked TableBuild protocol
    fn build_cell<E, Lg>(
        &mut self,
        spec: &TableSpec<E, L>,
        source_block: usize,
        coordinate: usize,
        book: &[E],
        acts: &[E],
        index: usize,
        ledger: &mut Lg,
    ) where
        E: Element,
        L: Lane<E>,
        Lg: Ledger,
    {
        debug_assert!(index < self.code_space);
        let source = index * source_block + coordinate;
        let entry = index * self.rows;
        spec.build(
            1,
            1,
            &book[source..source + 1],
            acts,
            &mut self.words[entry..entry + self.rows],
        );
        Self::charge_build(spec, 1, 1, self.rows, ledger);
    }

    fn charge_build<E, Lg>(
        spec: &TableSpec<E, L>,
        space: usize,
        block: usize,
        rows: usize,
        ledger: &mut Lg,
    ) where
        E: Element,
        L: Lane<E>,
        Lg: Ledger,
    {
        let products = count_product3(space, block, rows);
        if spec.build_multiplies {
            ledger.multiplied(products);
        } else {
            // The spec owns the observable-boundary charge: transparent fixed
            // expansions name every add, while a generic `Element::mac` body
            // names one opaque contraction presentation per product.
            ledger.added((spec.build_adds)(space, block, rows));
        }
    }

    /// The stack itself, one slab per block held.
    pub fn stack(&self) -> &[L] {
        self.words
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
pub struct TabulatedTriple<'a, 'w, 'c, E: Element, Bd: Bound, C: Enumerable<E, Bd>, O> {
    a: MatView<'a, Alphabet<E, Bd>>,
    w: CodedMatrix<'w, E, Bd, C>,
    c: MatViewMut<'c, O>,
}

impl<'a, 'w, 'c, E: Element, Bd: Bound, C: Enumerable<E, Bd>, O>
    TabulatedTriple<'a, 'w, 'c, E, Bd, C, O>
{
    /// Report non-existence once, before any arithmetic is named.
    ///
    /// This is where the coded operand's *orientation* is decided, and it is
    /// decided from `w.rows()` and `w.cols()` --- the two the canonical manifest
    /// records as `rows` and `cols`. A caller holding a manifest and no operand
    /// gets the same answer in advance from
    /// `uor_matmul_codec::Manifest::reduces_along_the_block`, and `CS-10`
    /// asserts that the two agree at every shape.
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
        if s.self_aliases(c.rows(), c.cols()) {
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
    index: &'s mut [usize],
}

impl<'s> Tabulation<'s> {
    /// Offer a narrow lane buffer.
    pub fn new(lanes: &'s mut [i64]) -> Self {
        Self {
            lanes,
            index: &mut [],
        }
    }

    /// Offer a lane buffer and room to record which output columns repeat.
    ///
    /// With it the traversal charges per *distinct* column of the coded operand
    /// instead of per column --- the same move [`crate::collapse`] makes on the
    /// rows of `A` and the same move tabulation itself makes on the codes. A cost
    /// that tracks meanings rather than expressions. The collapse applies at any
    /// column-block width: a class spread across blocks is charged once per block
    /// it appears in, which is the most a narrow offer can extract (`CD-16`).
    ///
    /// See [`suggested_tabulation_index`] for the size. An operand that repeats no
    /// column pays one pass over its code stream for the question, and `CG-10`
    /// reports what that costs alongside what it buys.
    pub fn with_index(lanes: &'s mut [i64], index: &'s mut [usize]) -> Self {
        Self { lanes, index }
    }

    /// Offer none.
    ///
    /// Not a degraded mode and not a fallback: the same identity, accumulated in
    /// a wider register (R13).
    pub fn none() -> Tabulation<'static> {
        Tabulation {
            lanes: &mut [],
            index: &mut [],
        }
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

/// How much index room the column collapse wants for this shape.
///
/// One entry per output column, plus an open-addressed dictionary at least twice
/// as wide so a probe stays short --- the same sizing
/// [`crate::suggested_collapse_index`] uses on the row side. Once the classes are
/// known the dictionary is dead, and the block-local first-occurrence map is
/// derived into it, so this size serves both passes.
///
/// A *query*. Offering none gives the same bytes from the uncollapsed traversal,
/// which is the rule every other offer in this library follows (`CD-13`).
pub fn suggested_tabulation_index(shape: Shape) -> usize {
    let dictionary = shape
        .n
        .saturating_mul(2) // R3-ok: a size query, not an accumulation
        .checked_next_power_of_two()
        .unwrap_or(usize::MAX);
    let doubled = dictionary.saturating_mul(2); // R3-ok: a size query, not an accumulation
    shape.n.saturating_add(doubled) // R3-ok: a size query, not an accumulation
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
pub fn suggested_tabulation_lanes<E: Tabulated, Bd: Bound>(
    shape: Shape,
    code_space: usize,
    block: usize,
) -> usize {
    if E::LANE_IS_EXACT {
        return 0;
    }
    let Some(plan) = Plan::choose(
        code_space,
        shape,
        E::LANE_BYTES,
        usize::MAX,
        usize::MAX,
        block,
        E::probe_capacity::<E::Lane>(Bd::VALUE),
    ) else {
        return 0;
    };
    // Reported in `i64` words, which is what the offer is made of.
    plan.lane_words(code_space)
        .saturating_mul(E::LANE_BYTES) // R3-ok: a size query, not an accumulation
        .div_ceil(core::mem::size_of::<i64>())
}

/// How many exact accumulators would let the tabulated traversal run at the whole
/// output width for this shape and this codec.
///
/// A *query*, like [`crate::suggested_scratch`]. Offering less narrows the column
/// block --- the column collapse still applies, at one charge per distinct column
/// per block (`CD-16`) --- and offering none gives the same bytes from the
/// streaming traversal (`CD-13`). It does not grow with `k`.
///
/// When the element type has no narrow register this covers the table stack and
/// the column accumulation too, because there is then nowhere else for them to
/// live.
pub fn suggested_tabulation<E: Tabulated, Bd: Bound>(
    shape: Shape,
    code_space: usize,
    block: usize,
) -> usize {
    let lane_capacity = E::probe_capacity::<E::Lane>(Bd::VALUE);
    let lanes_per_exact = core::mem::size_of::<AccOf<E>>()
        .checked_div(E::LANE_BYTES)
        .unwrap_or(0);
    let plan = if E::LANE_IS_EXACT {
        Plan::choose_shared_exact(
            code_space,
            shape,
            E::LANE_BYTES,
            usize::MAX,
            lanes_per_exact,
            block,
            lane_capacity,
        )
    } else {
        Plan::choose(
            code_space,
            shape,
            E::LANE_BYTES,
            usize::MAX,
            usize::MAX,
            block,
            lane_capacity,
        )
    };
    let Some(plan) = plan else {
        return 0;
    };
    let tile = plan.rows.saturating_mul(plan.cols); // R3-ok: a size query, not an accumulation
    if E::LANE_IS_EXACT {
        plan.shared_exact_charge(code_space, lanes_per_exact)
            .unwrap_or(usize::MAX)
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
    let codes = slab_codes(code_space);
    if codes == 0 || rows == 0 || lane_bytes == 0 {
        return false;
    }
    let Some(words) = codes.checked_mul(rows) else {
        return false;
    };
    let Some(bytes) = words.checked_mul(lane_bytes) else {
        return false;
    };
    let Some(bytes) = bytes.checked_mul(2) else {
        return false;
    };
    bytes <= l1_bytes
}

/// What one issued instruction of each traversal covers.
///
/// Both are *declarations a sequence makes about itself*, read from the
/// [`TableSpec`] and the [`uor_matmul_kernels::KernelSpec`] that will actually
/// run. That is the whole point of the type: the predicate used to price the
/// table at `MAX_BLOCK * rows` products per instruction and the dense tile at a
/// model constant, and neither described any sequence that has ever shipped.
///
/// A table's step is `MAX_BLOCK * lanes_per_add` --- one register of lanes, each
/// carrying a whole codeword --- and *not* `MAX_BLOCK * rows`, which is a whole
/// tile and therefore `rows / lanes_per_add` instructions. The two agree only
/// when a tile is one register.
///
/// Measured against the shipped pair on an AVX2 host the old form gave the right
/// answer, because it over-stated the table by the register count and the model
/// constant over-stated the dense tile by the same factor --- `vpdpbusd`'s
/// density on a host that has no `vpdpbusd`. Two errors that cancel are not a
/// derivation. On a host that does have VNNI they do not cancel: the dense tile
/// is four times denser per instruction and the table is not, and the old form
/// would take the table from `n = 683` where no `n` pays at all.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Steps {
    /// Products one issued table instruction covers: `MAX_BLOCK * lanes_per_add`.
    pub table: usize,
    /// Products one issued dense tile instruction covers.
    pub dense: usize,
    /// Rows of the output one dense tile call produces.
    ///
    /// The dense traversal issues [`Self::dense`] products per instruction only
    /// when it has this many rows to amortize against. With fewer it pays twice:
    /// for lanes the tile does not fill, and for a packed panel of `n * k`
    /// elements copied to compute `m * n * k` products --- which at `m = 1` is
    /// the same order as the arithmetic. That second cost is the larger one and
    /// is why this is the *blocking* row count and not the chosen tile's `mr`.
    pub dense_rows: usize,
}

/// An exact product of three address-sized cost factors.
///
/// Each factor occupies at most one 64-bit radix coordinate on every target
/// this workspace supports, so three coordinates are the derived complete
/// width. Quotient/remainder carry keeps the comparison total without a
/// saturating estimate changing which traversal is selected at `usize::MAX`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct CostProduct([u64; 3]);

impl CostProduct {
    const RADIX: u128 = u64::MAX as u128 + 1;

    const fn of(a: usize, b: usize, c: usize) -> Self {
        let factors = [a as u64, b as u64, c as u64];
        let mut limbs = [1u64, 0, 0];
        let mut factor = 0usize;
        while factor < factors.len() {
            let mut carry = 0u128;
            let mut limb = 0usize;
            while limb < limbs.len() {
                let wide = limbs[limb] as u128 * factors[factor] as u128 + carry;
                limbs[limb] = (wide % Self::RADIX) as u64;
                carry = wide / Self::RADIX;
                limb += 1;
            }
            // Three address-sized factors are strictly below radix^3, so the
            // third multiplication cannot carry beyond the derived width.
            debug_assert!(carry == 0);
            factor += 1;
        }
        Self(limbs)
    }

    const fn greater_than(self, other: Self) -> bool {
        if self.0[2] != other.0[2] {
            self.0[2] > other.0[2]
        } else if self.0[1] != other.0[1] {
            self.0[1] > other.0[1]
        } else {
            self.0[0] > other.0[0]
        }
    }
}

/// Dense products per issued step after charging an unfilled row tile.
const fn effective_dense_step(rows: usize, steps: Steps) -> usize {
    if steps.dense_rows == 0 {
        return 0;
    }
    let present = if rows < steps.dense_rows {
        rows
    } else {
        steps.dense_rows
    };
    // Two address-sized factors fit `u128`; division by at least the second
    // factor leaves a result no larger than `steps.dense`, hence `usize`.
    (steps.dense as u128 * present as u128 / steps.dense_rows as u128) as usize
}

/// Does tabulation issue fewer instructions than the dense tile, and does its
/// table fit?
///
/// `cols` is the width of the column block the caller's offer supports, which is
/// `n` when the offer holds the whole output width. The build is repeated once
/// per column block, so it is the block and not the shape that the count turns
/// on.
///
/// In this reference case `q = lanes_per_add = steps.table / block`, so the
/// build denominator is `q` and clearing it contributes the explicit `block`:
///
/// ```text
/// tabulated = m*k*S*block/steps.table + m*n*k/steps.table
/// dense     = m*k*n/steps.dense
/// ```
///
/// so the table is cheaper exactly when
/// `cols * (steps.table - effective) > code_space * effective * block`, where
/// `effective` is the dense step scaled by the rows actually present.
///
/// This locked public query has no [`TableSpec`] argument, so its build term is
/// the reference case `build_products_per_step == lanes_per_add`. A block-one
/// Atlas lookup replaces a much larger contraction and needs measured
/// build-kind evidence this signature cannot observe; the private driver
/// predicate below applies that evidence to the exact built-in declaration.
/// `block == 1` therefore remains false here rather than being overgeneralized
/// to arbitrary downstream builders.
pub const fn tabulation_pays(
    code_space: usize,
    block: usize,
    cols: usize,
    rows: usize,
    steps: Steps,
    l1_bytes: usize,
    lane_bytes: usize,
) -> bool {
    if block <= 1 || steps.table == 0 {
        return false;
    }
    let effective = effective_dense_step(rows, steps);
    effective > 0
        && steps.table > effective
        && CostProduct::of(cols, steps.table - effective, 1)
            .greater_than(CostProduct::of(code_space, effective, block))
        && tabulation_fits(code_space, rows, l1_bytes, lane_bytes)
}

/// The actual automatic selector, with the build declaration the public
/// operation-count query cannot observe.
///
/// For a block longer than one, `q = build_products_per_step`,
/// `t = block * lanes_per_add`, and `e` is the row-adjusted dense step. The
/// exact costs are `r*k*S/q + r*c*k/t` and `r*c*k/e`; clearing denominators
/// gives `c*q*(t-e) > S*e*t`. A block-one contextual Atlas contraction is not
/// this product-build comparison: CG-16 found no geometry-invariant scalar
/// crossover, so it cannot be assigned one by this shape-only predicate.
/// Forced [`Traversal::Tabulated`] never consults the cost predicate.
#[allow(clippy::too_many_arguments)] // the locked public cost query plus the private build declaration
fn tabulation_pays_for_spec<E, L>(
    code_space: usize,
    block: usize,
    cols: usize,
    rows: usize,
    steps: Steps,
    l1_bytes: usize,
    lane_bytes: usize,
    table: &TableSpec<E, L>,
) -> bool {
    if block == 0 || !tabulation_fits(code_space, rows, l1_bytes, lane_bytes) {
        return false;
    }
    if block == 1 {
        return false;
    }

    let Some(table_step) = block.checked_mul(table.lanes_per_add) else {
        return false;
    };
    let build_step = table.build_products_per_step;
    let effective = effective_dense_step(rows, steps);
    build_step > 0
        && effective > 0
        && steps.table == table_step
        && table_step > effective
        && CostProduct::of(cols, build_step, table_step - effective)
            .greater_than(CostProduct::of(code_space, effective, table_step))
}

/// The most rows of `A` one table can cover and still sit in L1.
///
/// Derived from the cache budget and the code space, and capped by the same `MC`
/// the blocked traversal uses --- not by a number chosen for this traversal (R8).
/// Zero means no table fits at all, which selects the streaming traversal.
pub const fn tabulation_rows(code_space: usize, l1_bytes: usize, lane_bytes: usize) -> usize {
    let codes = slab_codes(code_space);
    if codes == 0 || lane_bytes == 0 {
        return 0;
    }
    let Some(words) = codes.checked_mul(lane_bytes) else {
        return 0;
    };
    let Some(bytes) = words.checked_mul(2) else {
        return 0;
    };
    let room = l1_bytes / bytes;
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
    let codes = slab_codes(code_space);
    if codes == 0 || rows == 0 || lane_bytes == 0 || block == 0 {
        return 0;
    }
    let Some(words) = codes.checked_mul(rows) else {
        return 0;
    };
    let Some(per_slot) = words.checked_mul(lane_bytes) else {
        return 0;
    };
    // Half of L2. The other half is the code stream and the exact accumulator
    // tile, which pass through the same cache. Measured, a quarter was better
    // while the column loop had one load in flight and half is better now that it
    // has `COLUMN_GROUP` of them: the stack stops needing to be resident once the
    // latency is overlapped, and what is left to minimise is placement traffic,
    // which falls as `1/depth`.
    let Some(cache_charge) = per_slot.checked_mul(2) else {
        return 0;
    };
    let mut depth = l2_bytes / cache_charge;
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

/// What one call's panels declare for the lane: the grade gauge used to label
/// its contextual cells, and the nonnegative product envelope from which its
/// local source schedule is derived.
///
/// A fact of the call, not the family --- but for an integer family it is
/// the family's own declaration: bases of zero and the square of the declared
/// bound, with no walk to answer it, because an integer element carries no
/// exponent. The float symbol lane observes the complete addressed extent once
/// per selected table call and remains total for every answer (`CD-32`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct LaneScale {
    /// The lowest finite grade observed in `A`, used as its contextual q gauge.
    pub base_a: i32,
    /// The codebook's lowest finite grade, likewise.
    pub base_b: i32,
    /// The exact nonnegative certificate for one source atom, or the global
    /// envelope whose source-local refinement is required before contraction.
    /// `Q+1` identifies a singleton boundary/tag atom; no value is declined.
    pub per_step: u128,
}

impl LaneScale {
    /// The integral scale: nothing is pre-scaled, and the depth is the
    /// lane's own answer at the declared bound.
    fn integral(bound: u128) -> Self {
        Self {
            base_a: 0,
            base_b: 0,
            per_step: bound.saturating_mul(bound), // R3-ok: a bound, not an accumulation
        }
    }

    /// The exponent a completed run is placed at: `base_a + base_b`.
    fn exponent(&self) -> i32 {
        self.base_a.saturating_add(self.base_b) // R3-ok: an exponent base, not an accumulation
    }
}

/// The lane a family tabulates in, and the sequence that reads it.
///
/// One associated type, not a scan. Which register holds a run of products is a
/// property of the *element type* --- the same kind of fact as `AccOf<E>` --- and
/// writing it as a per-shape search was a mechanism with one answer.
///
/// Not a quality ordering, and not a fallback chain. Every lane computes the same
/// integer; a narrower one moves fewer bytes per product, and `CD-13` asserts the
/// bytes across all of them. An element family with no narrow register at all
/// tabulates in the exact accumulator, which is the complete sequence the
/// hardware offers --- the same status
/// [`uor_matmul_kernels::available_i64_exact`] has for the dense tile.
///
/// The supertrait is [`Element`], not [`Kernelized`]: the tabulated traversal
/// needs a lane, a table sequence, and a dense factorization to decline to, and
/// a float has all three without having a single integer kernel. What it takes
/// from the dense side it takes through [`Tabulated::dense_steps`] and
/// [`Tabulated::dense_gemm`], which the integer families answer with the tile
/// kernels and the float families with the exact float traversal.
pub trait Tabulated: Element {
    /// The lane one table entry's row is held in.
    type Lane: Lane<Self>;

    /// The lane the same table runs in when the caller asked to wrap into an
    /// output no wider than it: `Z/2^w`, where the lane's own wrap *is* the
    /// encode. Which lane runs is a function of two declarations --- the
    /// encode mode and the output type --- decided once at the traversal
    /// boundary, exactly as the dense side decides it in
    /// [`uor_matmul_kernels::KernelSpec::lane_depth`]'s `Factorization::Modular`
    /// arm (`CU-08`).
    ///
    /// `Self::Lane` for a family that offers no quotient (`i8`, `i16`): their
    /// exact lane already holds every depth a weight row reaches, so a
    /// quotient read would buy nothing, and [`Self::modular_table_admitted`]
    /// is `false` there.
    type ModLane: Lane<Self>;

    /// The lane the streaming decline accumulates in.
    ///
    /// The family's own lane for an integer --- a plain dot product is a
    /// table of one entry that nobody shares, so the stream and the table
    /// hold their runs in the same register. A family whose table lane holds
    /// contextual products cannot stream in it: the stream walks raw elements
    /// with no q extent observation ahead of them, so its lane is the one that
    /// holds a raw product exactly --- the complete accumulator, which was the
    /// float families' table lane too before the q lane existed. Its concrete
    /// identity remains part of this public extension point. The optimized
    /// coded-float decline reaches the same complete word through the family's
    /// empty-rest dense capability; it does not replace this associated type
    /// with a private carrier.
    type StreamLane: Lane<Self>;

    /// Bytes one lane word occupies. The column loop's traffic is this divided
    /// by the codec's block, per product, which is the whole of why it is here.
    const LANE_BYTES: usize = core::mem::size_of::<Self::Lane>();

    /// Does this family's lane come out of the accumulator offer rather than
    /// the narrow one?
    ///
    /// True exactly when the lane *is* the exact accumulator, which is where a
    /// word that wide already lives. The family's modular lane is read out of
    /// the same offer, relabelled --- an offer sized for the exact lane holds
    /// the same table several times over in a narrower quotient.
    const LANE_IS_EXACT: bool;

    /// Is the modular lane admissible for an output of `out_bits` bits?
    ///
    /// The width half of the dense side's rule ([`Kernelized::modular_spec`]):
    /// reduction modulo `2^w` is a ring homomorphism, so `Z/2^w` refines
    /// `Z/2^out_bits` exactly when `out_bits <= w`, and nothing is lost that
    /// the caller did not ask to lose. The other half --- the encode mode ---
    /// is asked at the traversal boundary, where `options` lives.
    fn modular_table_admitted(out_bits: u32) -> bool;

    /// The sequence for this backend, at this tile height and column group.
    ///
    /// Always present: the reference is exact on every alphabet and every tile,
    /// so narrowing the choice can never empty it.
    ///
    /// `sign_book` is the codec's [`uor_matmul_codec::Enumerable::SIGN_BIT_BOOK`]
    /// declaration, and a family reads it at bound 1 and nowhere else: it is
    /// what admits the Gray-walk build, whose correctness is the codec's
    /// declaration and not the alphabet's --- `Ternary` at bound 1 declares
    /// the same bound, and its book is not the bit decomposition.
    fn table_spec(
        backend: Backend,
        bound: u128,
        sign_book: bool,
        rows: usize,
        group: usize,
        block: usize,
    ) -> TableSpec<Self, Self::Lane>;

    /// The same, in the modular lane. Always present for the same reason, and
    /// reached only where [`Self::modular_table_admitted`] has said the output
    /// width admits the lane.
    fn table_spec_modular(
        backend: Backend,
        bound: u128,
        rows: usize,
        group: usize,
        block: usize,
    ) -> TableSpec<Self, Self::ModLane>;

    /// Borrow `want` lane words out of whichever offer this family's lane lives
    /// in.
    ///
    /// The two offers have concrete types --- `i64` for the narrow one, the
    /// exact accumulator for the other --- and the lane is an associated type,
    /// so the relabelling has to happen where both are known. Here, in three
    /// lines per family, and nowhere in the traversal.
    ///
    /// `None` when the offer is shorter than `want`, which is the caller getting
    /// a different traversal at the same bytes (S13, `CD-13`).
    fn lanes<'s>(
        narrow: &'s mut [i64],
        exact: &'s mut [AccOf<Self>],
        want: usize,
    ) -> Option<&'s mut [Self::Lane]>;

    /// The dense factorization's declarations for [`tabulation_pays`]: one
    /// issued instruction covers `dense` products when the traversal has
    /// `dense_rows` rows of the output to amortize against. `table` is the
    /// table's own number, which the caller has from the [`TableSpec`] it
    /// already holds.
    ///
    /// Read from the sequence the dense path would run, never from a constant
    /// standing in for it.
    fn dense_steps(backend: Backend, bound: u128, rows: usize, table: usize) -> Steps;

    /// The dense factorization, over the decoded operand: `a` against `b`,
    /// which is `W^T` read through swapped strides, into `c`, with `rest` as
    /// panel room. `false` when the dense triple or its required panel cannot
    /// be built, which the caller answers by streaming --- the same bytes by
    /// another walk (S13). A `false` answer writes no output.
    ///
    /// Acceptance with an empty `rest` is value-independent: the dense
    /// factorization can contract a borrowed dot without caller panel storage.
    /// The coded decline presents its first real bounded source chunk to that
    /// factorization and retains the exact partial when it accepts; there is no
    /// separate capability call and no recomputed product. Shipped float
    /// families accept; packed integer families require
    /// `rest.len() >= b.rows()` and decline before writing. An implementation
    /// cannot answer from operand values or change its answer between conformant
    /// subviews (`CD-20`).
    ///
    /// One method per family, because the dense driver is one per family: the
    /// tile kernels for an integer, the exact float traversal for a float.
    /// There is no float tile kernel --- no float instruction is exact
    /// (`CU-01`) --- and the two compute the same sum over the same decoded
    /// elements, which is what `CD-13` and `CD-14` assert byte for byte.
    fn dense_gemm<Bd, O, Ep>(
        a: MatView<'_, Alphabet<Self, Bd>>,
        b: MatView<'_, Alphabet<Self, Bd>>,
        c: MatViewMut<'_, O>,
        epilogue: &Ep,
        options: GemmOptions,
        rest: &mut [Alphabet<Self, Bd>],
    ) -> bool
    where
        Bd: Bound,
        O: Element + EncodeFrom<AccOf<Self>>,
        Ep: Epilogue<Self, O>;

    /// The same borrow, of the modular lane. Reached only from the boundary's
    /// modular arm, and `None` under the same rule.
    fn lanes_modular<'s>(
        narrow: &'s mut [i64],
        exact: &'s mut [AccOf<Self>],
        want: usize,
    ) -> Option<&'s mut [Self::ModLane]>;

    /// The lane capacity to plan against before the call's panels are
    /// walked.
    ///
    /// The lane's own answer at the declared bound, for every family whose
    /// capacity is a function of the alphabet alone. The q lane's capacity is
    /// a fact of its observed source extents, so it plans against the caches
    /// and derives source-ordered local boundaries after observation. Planning
    /// at a data-free one-product bound would pin every call to one block even
    /// though scalar fracture is total.
    fn probe_capacity<L: Lane<Self>>(declared: u128) -> Option<usize> {
        L::capacity(declared)
    }

    /// The contextual gauge and product envelope one call's panels declare,
    /// from an observation of the activations and codebook.
    ///
    /// The default answers the integral scale without walking: an integer
    /// element carries no exponent and there is nothing to measure. The float
    /// symbol lane's answer is the total q extent observation (`CD-32`), over
    /// `A` and the codebook. Ordinarily that costs `m * k + code_space *
    /// block` decodes rather than `(m + n) * k`; a pointwise block-one call
    /// whose offered dictionary can name its addressed symbols walks each
    /// addressed index once, because unused symbols cannot constrain its
    /// scale. Without that storage the ordinary raw-stream or enumeration
    /// factorization remains total. Non-finite symbols and an envelope above Q
    /// are retained for singleton/local fracture rather than ending the walk.
    ///
    /// Asked only after the traversal has selected the table, so a call the
    /// predicate declines never pays for it; the decodes it issues are
    /// charged to the ledger, so a harness reads their price off the census.
    fn lane_scale<Bd, C, Lg>(
        a: &MatView<'_, Alphabet<Self, Bd>>,
        w: &CodedMatrix<'_, Self, Bd, C>,
        ledger: &mut Lg,
    ) -> Option<LaneScale>
    where
        Bd: Bound,
        C: Enumerable<Self, Bd>,
        Lg: Ledger,
    {
        let _ = (a, w, ledger);
        Some(LaneScale::integral(Bd::VALUE))
    }

    /// The most products one lane word holds for these panels, once the walk
    /// has answered.
    ///
    /// The lane's own answer at the declared bound, for every family the
    /// default `lane_scale` serves. The float symbol lane's is derived from
    /// the walk's per-side bounds (`LaneScale::per_step`).
    fn lane_run<L: Lane<Self>>(declared: u128, scale: &LaneScale) -> Option<usize> {
        let _ = scale;
        L::capacity(declared)
    }

    /// Spell one element at `2^-base`, exactly, in the element-sized cell the
    /// table lane consumes.
    ///
    /// Identity for every family the default `lane_scale` serves: their
    /// elements carry no exponent and the base is zero. The f32 symbol lane
    /// uses the same four-byte cell contextually as a q carrier;
    /// [`Scaled64::mac`] is its paired consumer. That intermediate is panel
    /// storage, not a standalone IEEE result. Their composition and final
    /// placement are the exact power-of-two rescaling this protocol declares
    /// (`CD-20`).
    fn prescale(x: Self, base: i32) -> Self {
        let _ = base;
        x
    }

    /// Number the rows of `a` by symbolic identity, writing each row's
    /// representative into `index` and returning how many are distinct ---
    /// the row half of the collapse (`CD-15`, and `CD-17` for the float
    /// families). `None` when the offer cannot hold the answer.
    ///
    /// The default is always `None`: a family whose elements have no declared
    /// symbolic identity never row-collapses. That is the uncollapsed
    /// traversal, at the same bytes, which is the rule every declined offer
    /// in this library follows.
    fn distinct_a_rows<Bd>(
        a: &MatView<'_, Alphabet<Self, Bd>>,
        index: &mut [usize],
    ) -> Option<usize>
    where
        Bd: Bound,
    {
        let _ = (a, index);
        None
    }
}

/// The integer families' [`Tabulated::dense_steps`]: the tile kernel's own
/// declarations, read from the sequence the dense path would run.
///
/// `dense_rows` is the *blocking* row count, not the chosen tile's `mr`.
/// Reading `mr` says a one-row kernel wastes no lanes, which is true and is not
/// why the dense path is weak at small `m`: it packs `n * k` elements of the
/// operand to compute `m * n * k` products, so at `m = 1` the copy is the same
/// order as the arithmetic. Measured, `mr` here declined the table at
/// `1x1024x4096` where it is 5.4x the dense path.
fn tile_steps<E: Kernelized>(backend: Backend, bound: u128, rows: usize, table: usize) -> Steps {
    let dense = E::exact_spec(backend, bound, rows);
    Steps {
        table,
        dense: dense.products_per_step,
        dense_rows: blocking::KERNEL_ROWS,
    }
}

/// The integer families' [`Tabulated::dense_gemm`]: the tile kernels, with what
/// the offer left after the decoded operand as their panel room.
fn dense_tile<E, Bd, O, Ep>(
    a: MatView<'_, Alphabet<E, Bd>>,
    b: MatView<'_, Alphabet<E, Bd>>,
    c: MatViewMut<'_, O>,
    epilogue: &Ep,
    options: GemmOptions,
    rest: &mut [Alphabet<E, Bd>],
) -> bool
where
    E: Kernelized,
    Bd: Bound,
    O: Element + EncodeFrom<AccOf<E>>,
    Ep: Epilogue<E, O>,
{
    // The packed factorization exists only with one reduction panel.  The
    // outer dense route establishes this before decoding; repeating the fact
    // here makes a first real empty-rest decline occur before any output.
    if rest.len() < b.rows() {
        return false;
    }
    let Ok(mut dense) = Triple::new(a, b, c) else {
        // The shapes conformed when the `TabulatedTriple` was built and the output
        // was checked for aliasing there, so neither failure can arise a second
        // time. Streaming gives the same bytes, which is why this needs no report.
        return false;
    };
    gemm_packed(&mut dense, epilogue, options, &mut Scratch::new(rest));
    true
}

/// `i8`: an `i32` lane holds 133144 products of the full alphabet, which is past
/// every depth a weight row reaches, so the whole reduction is one run.
impl Tabulated for i8 {
    type Lane = i32;
    type ModLane = i32;
    type StreamLane = i32;
    const LANE_IS_EXACT: bool = false;

    fn modular_table_admitted(_: u32) -> bool {
        // The exact lane already holds every depth a weight row reaches, so a
        // quotient read buys no depth and no instructions, and none is offered.
        false
    }

    fn table_spec(
        backend: Backend,
        bound: u128,
        sign_book: bool,
        rows: usize,
        group: usize,
        block: usize,
    ) -> TableSpec<i8, i32> {
        // The Gray walk is the `Auto` answer exactly when the codec declares
        // the sign codebook: the walk derives the signs from the code index,
        // which is the declaration the flag makes. A named backend is the
        // caller asking for that backend's own build, and gets it.
        if sign_book && bound == 1 && matches!(backend, Backend::Auto) {
            return uor_matmul_kernels::gray_sign_table(rows, group);
        }
        choose_table(available_table_i8(rows, group), backend, bound, block)
            .expect("the reference sequence is always present")
    }

    fn table_spec_modular(
        backend: Backend,
        bound: u128,
        rows: usize,
        group: usize,
        block: usize,
    ) -> TableSpec<i8, i32> {
        // Never reached: `modular_table_admitted` is `false`, so the boundary
        // never selects this lane. Written as the exact sequence rather than a
        // panic because the exact `i32` lane is congruent mod `2^32` anyway ---
        // if it ever ran, the bytes would still be the caller's.
        Self::table_spec(backend, bound, false, rows, group, block)
    }

    fn lanes<'s>(
        narrow: &'s mut [i64],
        _: &'s mut [AccOf<i8>],
        want: usize,
    ) -> Option<&'s mut [i32]> {
        bytemuck::cast_slice_mut::<i64, i32>(narrow).get_mut(..want)
    }

    fn dense_steps(backend: Backend, bound: u128, rows: usize, table: usize) -> Steps {
        tile_steps::<Self>(backend, bound, rows, table)
    }

    fn dense_gemm<Bd, O, Ep>(
        a: MatView<'_, Alphabet<Self, Bd>>,
        b: MatView<'_, Alphabet<Self, Bd>>,
        c: MatViewMut<'_, O>,
        epilogue: &Ep,
        options: GemmOptions,
        rest: &mut [Alphabet<Self, Bd>],
    ) -> bool
    where
        Bd: Bound,
        O: Element + EncodeFrom<AccOf<Self>>,
        Ep: Epilogue<Self, O>,
    {
        dense_tile(a, b, c, epilogue, options, rest)
    }

    fn lanes_modular<'s>(
        narrow: &'s mut [i64],
        exact: &'s mut [AccOf<i8>],
        want: usize,
    ) -> Option<&'s mut [i32]> {
        // Never reached, as `table_spec_modular`.
        Self::lanes(narrow, exact, want)
    }

    fn distinct_a_rows<Bd: Bound>(
        a: &MatView<'_, Alphabet<Self, Bd>>,
        index: &mut [usize],
    ) -> Option<usize> {
        distinct_rows(a, index)
    }
}

/// `i16`: two full `i16` products already need 31 bits, so no 32-bit lane holds
/// an entry of any block longer than one.
impl Tabulated for i16 {
    type Lane = i64;
    type ModLane = i64;
    type StreamLane = i64;
    const LANE_IS_EXACT: bool = false;

    fn modular_table_admitted(_: u32) -> bool {
        // As `i8`: the exact lane already reaches every depth there is.
        false
    }

    fn table_spec(
        backend: Backend,
        bound: u128,
        _sign_book: bool,
        rows: usize,
        group: usize,
        block: usize,
    ) -> TableSpec<i16, i64> {
        choose_table(available_table_i16(rows, group), backend, bound, block)
            .expect("the reference sequence is always present")
    }

    fn table_spec_modular(
        backend: Backend,
        bound: u128,
        rows: usize,
        group: usize,
        block: usize,
    ) -> TableSpec<i16, i64> {
        // Never reached, as `i8`'s.
        Self::table_spec(backend, bound, false, rows, group, block)
    }

    fn lanes<'s>(
        narrow: &'s mut [i64],
        _: &'s mut [AccOf<i16>],
        want: usize,
    ) -> Option<&'s mut [i64]> {
        narrow.get_mut(..want)
    }

    fn dense_steps(backend: Backend, bound: u128, rows: usize, table: usize) -> Steps {
        tile_steps::<Self>(backend, bound, rows, table)
    }

    fn dense_gemm<Bd, O, Ep>(
        a: MatView<'_, Alphabet<Self, Bd>>,
        b: MatView<'_, Alphabet<Self, Bd>>,
        c: MatViewMut<'_, O>,
        epilogue: &Ep,
        options: GemmOptions,
        rest: &mut [Alphabet<Self, Bd>],
    ) -> bool
    where
        Bd: Bound,
        O: Element + EncodeFrom<AccOf<Self>>,
        Ep: Epilogue<Self, O>,
    {
        dense_tile(a, b, c, epilogue, options, rest)
    }

    fn lanes_modular<'s>(
        narrow: &'s mut [i64],
        exact: &'s mut [AccOf<i16>],
        want: usize,
    ) -> Option<&'s mut [i64]> {
        // Never reached, as `i8`'s.
        Self::lanes(narrow, exact, want)
    }

    fn distinct_a_rows<Bd: Bound>(
        a: &MatView<'_, Alphabet<Self, Bd>>,
        index: &mut [usize],
    ) -> Option<usize> {
        distinct_rows(a, index)
    }
}

/// `i32`: the product of two `i32` needs 62 bits and a run of them needs more, so
/// the lane is the exact accumulator. Nothing narrower is an *exact* lane here
/// --- an `i64` would hold two products --- and that is a fact about the width,
/// not a gap in the sequence table.
///
/// In the quotient there is a narrower lane: `Z/2^32` needs only the low half
/// of each product, so a `Mod32` word serves at any depth, admissible exactly
/// when the caller encodes by wrapping into an output no wider than it
/// (`CU-08`). It is read out of the accumulator offer, four words to the word,
/// so an offer sized for the exact lane already holds it.
impl Tabulated for i32 {
    type Lane = Wide<AccOf<i32>>;
    type ModLane = Mod32;
    type StreamLane = Wide<AccOf<i32>>;
    const LANE_IS_EXACT: bool = true;

    fn modular_table_admitted(out_bits: u32) -> bool {
        // `Z/2^32` refines `Z/2^out_bits` exactly here.
        out_bits <= 32
    }

    fn table_spec(
        backend: Backend,
        bound: u128,
        _sign_book: bool,
        rows: usize,
        group: usize,
        block: usize,
    ) -> TableSpec<i32, Wide<AccOf<i32>>> {
        // The reference is the only sequence for this family, and its `k_group`
        // is one, which divides every block.
        let _ = (backend, bound, block);
        portable_table::<i32, Wide<AccOf<i32>>>(rows, group)
    }

    fn table_spec_modular(
        backend: Backend,
        bound: u128,
        rows: usize,
        group: usize,
        block: usize,
    ) -> TableSpec<i32, Mod32> {
        choose_table(
            available_table_i32_modular(rows, group),
            backend,
            bound,
            block,
        )
        .expect("the reference sequence is always present")
    }

    fn lanes<'s>(
        _: &'s mut [i64],
        exact: &'s mut [AccOf<i32>],
        want: usize,
    ) -> Option<&'s mut [Wide<AccOf<i32>>]> {
        Some(Wide::wrap_slice_mut(exact.get_mut(..want)?))
    }

    fn dense_steps(backend: Backend, bound: u128, rows: usize, table: usize) -> Steps {
        tile_steps::<Self>(backend, bound, rows, table)
    }

    fn dense_gemm<Bd, O, Ep>(
        a: MatView<'_, Alphabet<Self, Bd>>,
        b: MatView<'_, Alphabet<Self, Bd>>,
        c: MatViewMut<'_, O>,
        epilogue: &Ep,
        options: GemmOptions,
        rest: &mut [Alphabet<Self, Bd>],
    ) -> bool
    where
        Bd: Bound,
        O: Element + EncodeFrom<AccOf<Self>>,
        Ep: Epilogue<Self, O>,
    {
        dense_tile(a, b, c, epilogue, options, rest)
    }

    fn lanes_modular<'s>(
        _: &'s mut [i64],
        exact: &'s mut [AccOf<i32>],
        want: usize,
    ) -> Option<&'s mut [Mod32]> {
        // The family's lanes live in the accumulator offer, as the exact one
        // does; the relabelling is four modular words to the accumulator.
        Mod32::wrap_i128s_mut(exact).get_mut(..want)
    }

    fn distinct_a_rows<Bd: Bound>(
        a: &MatView<'_, Alphabet<Self, Bd>>,
        index: &mut [usize],
    ) -> Option<usize> {
        distinct_rows(a, index)
    }
}

/// `i64`: as `i32`, one width up. The build's multiply has no SIMD instruction
/// at this width, so the modular lane is the portable sequence alone --- the
/// same reason the dense `i64` modular family is portable-only.
impl Tabulated for i64 {
    type Lane = Wide<AccOf<i64>>;
    type ModLane = Mod64;
    type StreamLane = Wide<AccOf<i64>>;
    const LANE_IS_EXACT: bool = true;

    fn modular_table_admitted(out_bits: u32) -> bool {
        out_bits <= 64
    }

    fn table_spec(
        backend: Backend,
        bound: u128,
        _sign_book: bool,
        rows: usize,
        group: usize,
        block: usize,
    ) -> TableSpec<i64, Wide<AccOf<i64>>> {
        // The reference is the only sequence for this family, and its `k_group`
        // is one, which divides every block.
        let _ = (backend, bound, block);
        portable_table::<i64, Wide<AccOf<i64>>>(rows, group)
    }

    fn table_spec_modular(
        backend: Backend,
        bound: u128,
        rows: usize,
        group: usize,
        block: usize,
    ) -> TableSpec<i64, Mod64> {
        choose_table(
            available_table_i64_modular(rows, group),
            backend,
            bound,
            block,
        )
        .expect("the reference sequence is always present")
    }

    fn lanes<'s>(
        _: &'s mut [i64],
        exact: &'s mut [AccOf<i64>],
        want: usize,
    ) -> Option<&'s mut [Wide<AccOf<i64>>]> {
        Some(Wide::wrap_slice_mut(exact.get_mut(..want)?))
    }

    fn dense_steps(backend: Backend, bound: u128, rows: usize, table: usize) -> Steps {
        tile_steps::<Self>(backend, bound, rows, table)
    }

    fn dense_gemm<Bd, O, Ep>(
        a: MatView<'_, Alphabet<Self, Bd>>,
        b: MatView<'_, Alphabet<Self, Bd>>,
        c: MatViewMut<'_, O>,
        epilogue: &Ep,
        options: GemmOptions,
        rest: &mut [Alphabet<Self, Bd>],
    ) -> bool
    where
        Bd: Bound,
        O: Element + EncodeFrom<AccOf<Self>>,
        Ep: Epilogue<Self, O>,
    {
        dense_tile(a, b, c, epilogue, options, rest)
    }

    fn lanes_modular<'s>(
        _: &'s mut [i64],
        exact: &'s mut [AccOf<i64>],
        want: usize,
    ) -> Option<&'s mut [Mod64]> {
        // Three modular words to the three-limb accumulator, as `i32`'s four.
        Mod64::wrap_limbs_mut(exact).get_mut(..want)
    }

    fn distinct_a_rows<Bd: Bound>(
        a: &MatView<'_, Alphabet<Self, Bd>>,
        index: &mut [usize],
    ) -> Option<usize> {
        distinct_rows(a, index)
    }
}

/// `f32`: the table panel stores one contextual q carrier in each existing
/// four-byte cell. Its fraction is the exact IEEE significand residue and its
/// q field is the finite factor's grade relative to the call's common base;
/// q=255 retains all non-finite boundary symbols. [`Scaled64`] contracts those
/// cells through the signed-octet Atlas lookup and holds either the unchanged
/// compact coefficient or a self-describing finite/boundary tag. The result is
/// placed once at `base_a + base_b`.
///
/// The compact interval is not a data limit. When a whole codec block cannot
/// fit, the driver derives least per-scalar envelopes from the already chosen
/// factorization and fractures only the unsafe aggregate at its source-ordered
/// boundary. Every IEEE value therefore remains table-executable (`CD-32`).
const fn f32_q_build_presentations(space: usize, block: usize, rows: usize) -> u64 {
    // The q contraction has data-dependent Atlas work. The generic TableBuild
    // boundary can observe exactly one contraction presentation per product,
    // while CU/CD purity gates inspect the private recurrence itself.
    count_product3(space, block, rows)
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
fn atlas_double_u32(mut value: u32, times: u32) -> u32 {
    for _ in 0..times {
        value = value.wrapping_add(value);
    }
    value
}

/// Relabel one IEEE symbol as its contextual q cell without floating-point
/// arithmetic. Sign and fraction stay in their source positions; only the
/// exponent field becomes the exact grade relative to `base`.
#[inline(always)]
fn project_f32_q(x: f32, base: i32) -> f32 {
    let raw: u32 = bytemuck::cast::<f32, u32>(x);
    let sign_place = atlas_power_of_two_u32(u32::BITS - 1);
    let fraction_place = atlas_power_of_two_u32(f32_q::SIGNIFICAND_BITS - 1);
    let negative = raw >= sign_place;
    let unsigned = if negative { raw - sign_place } else { raw };
    let source_q = unsigned / fraction_place;
    let fraction = unsigned % fraction_place;
    let maximum_relative = u32::try_from(f32_q::MAX_FACTOR_EXP - f32_q::MIN_FACTOR_EXP)
        .expect("the model-derived f32 grade range is nonnegative");
    let special_q = maximum_relative + 2;
    let q: u32 = if source_q == 0 || source_q == special_q {
        source_q
    } else {
        let relative = x
            .pack()
            .exp
            .checked_sub(base)
            .expect("the observed common base is no greater than a finite factor grade");
        u32::try_from(relative)
            .expect("the f32 factor-grade interval fits the model-derived q field")
            + 1
    };
    let sign = if negative { sign_place } else { 0 };
    let q_field = atlas_double_u32(q, f32_q::SIGNIFICAND_BITS - 1);
    bytemuck::cast::<u32, f32>(sign + q_field + fraction)
}

/// `P * 2^width`, clipped only after it has crossed the compact ceiling. The
/// sentinel `Q+1` says that this source atom must be placed by itself.
#[inline]
fn f32_q_step_bound(width: u32) -> u128 {
    let ceiling = u128::from(f32_q::COMPACT_CEILING);
    let singleton = ceiling + 1;
    let mut bound = u128::from(f32_q::PRODUCT_BOUND);
    for _ in 0..width {
        if bound > ceiling - bound.min(ceiling) {
            return singleton;
        }
        bound += bound;
    }
    bound
}

impl Tabulated for f32 {
    type Lane = Scaled64;
    type ModLane = Scaled64;
    type StreamLane = Wide<AccOf<f32>>;
    const LANE_IS_EXACT: bool = false;

    fn modular_table_admitted(_: u32) -> bool {
        // There is no quotient a float wraps into, as before --- and the q lane
        // is not one: it is exact at its contextual gauge, not congruent modulo
        // a width.
        false
    }

    fn table_spec(
        backend: Backend,
        bound: u128,
        _sign_book: bool,
        rows: usize,
        group: usize,
        block: usize,
    ) -> TableSpec<f32, Scaled64> {
        // The reference is the only sequence for this family, exact on every
        // alphabet by its own declaration, and its `k_group` is one, which
        // divides every block. The walk's bound is not a selection input:
        // there is no narrower-`max_bound` sequence to admit or decline.
        let _ = (backend, bound, block);
        let mut spec = portable_table::<f32, Scaled64>(rows, group);
        // `Scaled64::mac` contracts two q cells through their occupied
        // centered-octet extents and adds the resulting Laurent coefficient.
        // The mathematical product therefore issues no widening multiply.
        spec.build_multiplies = false;
        spec.build_adds = f32_q_build_presentations;
        spec.lane_cap = u128::from(f32_q::COMPACT_CEILING);
        spec
    }

    fn table_spec_modular(
        backend: Backend,
        bound: u128,
        rows: usize,
        group: usize,
        block: usize,
    ) -> TableSpec<f32, Scaled64> {
        // Never reached, as `i8`'s.
        Self::table_spec(backend, bound, false, rows, group, block)
    }

    fn lanes<'s>(
        narrow: &'s mut [i64],
        _: &'s mut [AccOf<f32>],
        want: usize,
    ) -> Option<&'s mut [Scaled64]> {
        // The lane word *is* an `i64`, so the narrow offer is the lane
        // buffer, relabelled one word to one lane.
        Some(Scaled64::wrap_i64s_mut(narrow.get_mut(..want)?))
    }

    fn lanes_modular<'s>(
        narrow: &'s mut [i64],
        exact: &'s mut [AccOf<f32>],
        want: usize,
    ) -> Option<&'s mut [Scaled64]> {
        // Never reached, as `i8`'s.
        Self::lanes(narrow, exact, want)
    }

    fn probe_capacity<L: Lane<Self>>(_: u128) -> Option<usize> {
        // The lane's capacity is a fact of the observed q extents, which the
        // plan precedes: plan against the caches, then let the source-local
        // scheduler partition the envelope without refusing a value.
        None
    }

    fn lane_scale<Bd, C, Lg>(
        a: &MatView<'_, Alphabet<Self, Bd>>,
        w: &CodedMatrix<'_, Self, Bd, C>,
        ledger: &mut Lg,
    ) -> Option<LaneScale>
    where
        Bd: Bound,
        C: Enumerable<Self, Bd>,
        Lg: Ledger,
    {
        // The q lane's walk and declaration (`CD-32`), over every addressed
        // activation and book presentation. Non-finite symbols are retained as
        // boundary tags and do not end the walk: later finite symbols still
        // determine the gauge used by the paired q projection.
        let (m, k) = (a.rows(), a.cols());
        let peeled = a.peeled();
        let mut visits = 0u64;
        let mut nonfinite = false;
        let sign_place = atlas_power_of_two_u32(u32::BITS - 1);
        let mut max_a = (0u32, 0.0f32);
        let mut a_span = Span::EMPTY;
        for i in 0..m {
            for v in peeled.row_walk(i, 0, k) {
                let value = *v;
                let code = value.pack();
                visits = count_sum2(visits, 1);
                if code.is_finite() {
                    a_span.see(code);
                    let magnitude = bytemuck::cast::<f32, u32>(value) % sign_place;
                    if magnitude > max_a.0 {
                        max_a = (magnitude, value);
                    }
                } else {
                    nonfinite = true;
                }
            }
        }
        let mut b_span = Span::EMPTY;
        let codec = w.codec();
        let block = <C as uor_matmul_codec::Codec<Self, Bd>>::MAX_BLOCK;
        let mut max_b = (0u32, 0.0f32);
        let sparse_book = block == 1 && !C::SIGN_BIT_BOOK && w.codes().len() < C::CODE_SPACE;
        if sparse_book {
            // The private addressed-codec presentation makes this one visit per
            // distinct address when the caller offered the dictionary. Without
            // it, raw visits are still fewer than a complete enumeration and
            // require no hidden allocation or arbitrary cap.
            for &stored in w.codes() {
                let index = C::index_of(stored);
                let value = codec.decode_element(C::code_at(index), 0).get();
                let code = value.pack();
                visits = count_sum2(visits, 1);
                if code.is_finite() {
                    b_span.see(code);
                    let magnitude = bytemuck::cast::<f32, u32>(value) % sign_place;
                    if magnitude > max_b.0 {
                        max_b = (magnitude, value);
                    }
                } else {
                    nonfinite = true;
                }
            }
        } else {
            for index in 0..C::CODE_SPACE {
                for t in 0..block {
                    let value = codec.decode_element(C::code_at(index), t).get();
                    let code = value.pack();
                    visits = count_sum2(visits, 1);
                    if code.is_finite() {
                        b_span.see(code);
                        let magnitude = bytemuck::cast::<f32, u32>(value) % sign_place;
                        if magnitude > max_b.0 {
                            max_b = (magnitude, value);
                        }
                    } else {
                        nonfinite = true;
                    }
                }
            }
        }
        ledger.decoded(visits);
        let (wa, wb) = (a_span.width(), b_span.width());
        // A boundary tag is a singleton. Otherwise P is the largest exact
        // significand product and the two relative spans regrade it. Crossing
        // Q is represented by Q+1; the local scheduler refines that global
        // envelope before building rather than declining the table.
        let per_step = if nonfinite {
            u128::from(f32_q::COMPACT_CEILING) + 1
        } else if k == 1 && block == 1 {
            // On a scalar source presentation the largest magnitude on each
            // side is the exact L-infinity certificate for every output lane.
            // Binary32's unsigned symbol order is magnitude order on finite
            // values, so selecting the extrema is address arithmetic. Their
            // one product is then contracted by the same q/Atlas operation the
            // table build uses; a tag is the singleton certificate Q+1.
            let product = <Scaled64 as Lane<f32>>::mac(
                Scaled64(0),
                project_f32_q(max_a.1, a_span.base()),
                project_f32_q(max_b.1, b_span.base()),
            );
            ledger.added(1);
            if product.0 >= i64::try_from(f32_q::TAG_BASE).expect("the q tag base fits i64") {
                u128::from(f32_q::COMPACT_CEILING) + 1
            } else {
                u128::from(product.0.unsigned_abs())
            }
        } else {
            f32_q_step_bound(wa + wb)
        };
        Some(LaneScale {
            base_a: a_span.base(),
            base_b: b_span.base(),
            per_step,
        })
    }

    fn lane_run<L: Lane<Self>>(_: u128, scale: &LaneScale) -> Option<usize> {
        // The paired lane owns Q. Passing the exact observed product envelope
        // through its declaration keeps the query and the executed carrier in
        // one model-derived capacity law.
        L::capacity(scale.per_step)
    }

    fn prescale(x: Self, base: i32) -> Self {
        // This contextual protocol cell goes directly to `Scaled64::mac`.
        // Only its address field is relabelled; no semantic float arithmetic or
        // parallel carrier buffer exists.
        project_f32_q(x, base)
    }

    fn dense_steps(_backend: Backend, _bound: u128, _rows: usize, table: usize) -> Steps {
        // The exact float traversal is the dense factorization here, and it is
        // scalar: one product per step, one row of the output at a time. There
        // is no tile kernel whose declarations could be read instead, because
        // no float instruction is exact.
        Steps {
            table,
            dense: 1,
            dense_rows: 1,
        }
    }

    fn dense_gemm<Bd, O, Ep>(
        a: MatView<'_, Alphabet<Self, Bd>>,
        b: MatView<'_, Alphabet<Self, Bd>>,
        c: MatViewMut<'_, O>,
        epilogue: &Ep,
        options: GemmOptions,
        rest: &mut [Alphabet<Self, Bd>],
    ) -> bool
    where
        Bd: Bound,
        O: Element + EncodeFrom<AccOf<Self>>,
        Ep: Epilogue<Self, O>,
    {
        // The float driver reads the codes themselves, so the alphabet wrapper
        // comes off --- a relabelling, not a copy --- and it keeps no panels of
        // its own, so the leftover offer goes unused.
        let _ = rest;
        let Ok(mut dense) = Triple::new(a.peeled(), b.peeled(), c) else {
            return false;
        };
        if dense.shape().m == 1 && dense.shape().n == 1 {
            let mut acc = <AccOf<Self> as Accumulator>::ZERO;
            accumulate_atlas_dot(
                &mut acc,
                dense.shape().k,
                PanelFacts::UNKNOWN,
                options.backend,
                |p| dense.a().at(0, p).pack(),
                |p| dense.b().at(p, 0).pack(),
            );
            let prior = if epilogue.reads_c() {
                Some(*dense.c_mut().at(0, 0))
            } else {
                None
            };
            *dense.c_mut().at_mut(0, 0) = epilogue.finish(acc, prior, options.encode);
            return true;
        }
        gemm_float(&mut dense, epilogue, options);
        true
    }

    fn distinct_a_rows<Bd: Bound>(
        a: &MatView<'_, Alphabet<Self, Bd>>,
        index: &mut [usize],
    ) -> Option<usize> {
        // Bit-pattern identity (`CD-17`): two rows share a sum exactly when
        // they are the same symbols, and the symbol is the bits (`CK-10`).
        distinct_rows(a, index)
    }
}

/// `f64`: the lane is the complete accumulator. Every product and table-entry
/// combine remains the Atlas lookup/add contraction; only its residency is
/// wider than `f32`'s compact q lane. The measured one-element Arena
/// codec does not repay that traffic in the automatic selector, while a
/// downstream enumerable codec with a longer block is priced from its own
/// declaration and may amortize the same pure table build. There is no
/// categorical codec assumption in the family (`CD-20`).
impl Tabulated for f64 {
    type Lane = Wide<AccOf<f64>>;
    type ModLane = Wide<AccOf<f64>>;
    type StreamLane = Wide<AccOf<f64>>;
    const LANE_IS_EXACT: bool = true;

    fn modular_table_admitted(_: u32) -> bool {
        // As `f32`: no quotient to wrap into, no modular lane.
        false
    }

    fn table_spec(
        backend: Backend,
        bound: u128,
        _sign_book: bool,
        rows: usize,
        group: usize,
        block: usize,
    ) -> TableSpec<f64, Wide<AccOf<f64>>> {
        // The reference is the only sequence for this family, and its `k_group`
        // is one, which divides every block.
        let _ = (backend, bound, block);
        let mut spec = portable_table::<f64, Wide<AccOf<f64>>>(rows, group);
        // `Wide<f64>` delegates each product to the exact Atlas accumulator;
        // its lookup/add body issues no widening multiply instruction. The
        // internal NAF atom count depends on the values and is opaque at this
        // shape-only TableSpec boundary, so the inherited one-per-product add
        // charge is explicitly the contraction presentation, not a fabricated
        // claim about the number of internal Atlas additions.
        spec.build_multiplies = false;
        spec
    }

    fn table_spec_modular(
        backend: Backend,
        bound: u128,
        rows: usize,
        group: usize,
        block: usize,
    ) -> TableSpec<f64, Wide<AccOf<f64>>> {
        // Never reached, as `f32`'s.
        Self::table_spec(backend, bound, false, rows, group, block)
    }

    fn lanes<'s>(
        _: &'s mut [i64],
        exact: &'s mut [AccOf<f64>],
        want: usize,
    ) -> Option<&'s mut [Wide<AccOf<f64>>]> {
        Some(Wide::wrap_slice_mut(exact.get_mut(..want)?))
    }

    fn lanes_modular<'s>(
        narrow: &'s mut [i64],
        exact: &'s mut [AccOf<f64>],
        want: usize,
    ) -> Option<&'s mut [Wide<AccOf<f64>>]> {
        // Never reached, as `f32`'s.
        Self::lanes(narrow, exact, want)
    }

    fn dense_steps(_backend: Backend, _bound: u128, _rows: usize, table: usize) -> Steps {
        // As `f32`: the scalar exact traversal, one product per step over one
        // row of the output.
        Steps {
            table,
            dense: 1,
            dense_rows: 1,
        }
    }

    fn dense_gemm<Bd, O, Ep>(
        a: MatView<'_, Alphabet<Self, Bd>>,
        b: MatView<'_, Alphabet<Self, Bd>>,
        c: MatViewMut<'_, O>,
        epilogue: &Ep,
        options: GemmOptions,
        rest: &mut [Alphabet<Self, Bd>],
    ) -> bool
    where
        Bd: Bound,
        O: Element + EncodeFrom<AccOf<Self>>,
        Ep: Epilogue<Self, O>,
    {
        // As `f32`: peel the wrapper, hand the decoded operand to the float
        // driver, and leave the leftover offer unused.
        let _ = rest;
        let Ok(mut dense) = Triple::new(a.peeled(), b.peeled(), c) else {
            return false;
        };
        if dense.shape().m == 1 && dense.shape().n == 1 {
            let mut acc = <AccOf<Self> as Accumulator>::ZERO;
            accumulate_atlas_dot(
                &mut acc,
                dense.shape().k,
                PanelFacts::UNKNOWN,
                options.backend,
                |p| dense.a().at(0, p).pack(),
                |p| dense.b().at(p, 0).pack(),
            );
            let prior = if epilogue.reads_c() {
                Some(*dense.c_mut().at(0, 0))
            } else {
                None
            };
            *dense.c_mut().at_mut(0, 0) = epilogue.finish(acc, prior, options.encode);
            return true;
        }
        gemm_float(&mut dense, epilogue, options);
        true
    }

    fn distinct_a_rows<Bd: Bound>(
        a: &MatView<'_, Alphabet<Self, Bd>>,
        index: &mut [usize],
    ) -> Option<usize> {
        // As `f32`: the bit pattern is the symbol (`CD-17`, `CK-10`).
        distinct_rows(a, index)
    }
}

/// The row tile, column block and stack depth one call resolves to.
///
/// `pub` so a measurement harness asks the planner the same question the
/// traversal asks, once, instead of carrying a copy of the derivation that
/// would drift from it: the chunk-extraction count the census does not see is
/// a function of this plan and the walk's lane run.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Plan {
    /// Rows of `A` one tile reduces together.
    pub rows: usize,
    /// Output columns one block of the traversal covers before `exact` is
    /// reused.
    pub cols: usize,
    /// Blocks of the reduction the table stack holds at once.
    pub depth: usize,
}

impl Plan {
    /// Lane words the plan needs: the column accumulation, then the stack.
    ///
    /// The column accumulation is a buffer now, and that is what bought the
    /// biggest single step in this module. It used to be `R` registers held
    /// across one chunk and folded into the exact accumulator at the end of it,
    /// which meant the exact accumulator was read and written `m*n*(k/Bk)/depth`
    /// times. Held in the lane instead, it is `rows * cols` narrow words --- a
    /// quarter of the bytes at `(i8, i32)` --- and the exact accumulator is
    /// touched once per output element.
    const fn lane_words(&self, code_space: usize) -> usize {
        self.rows
            .saturating_mul(self.cols) // R3-ok: a size query, not an accumulation
            .saturating_add(table_words(code_space, self.rows, self.depth)) // R3-ok: a size query, not an accumulation
    }

    /// Exact-accumulator words jointly occupied by the output tile and by a
    /// lane/table plan relabelled into that same offer.
    fn shared_exact_charge(&self, code_space: usize, lanes_per_exact: usize) -> Option<usize> {
        if lanes_per_exact == 0 {
            return None;
        }
        let tile = self.rows.checked_mul(self.cols)?;
        let lanes = tile.checked_add(table_words(code_space, self.rows, self.depth))?;
        tile.checked_add(lanes.div_ceil(lanes_per_exact))
    }

    /// Widest plan when exact cells and table lanes inhabit one caller offer.
    ///
    /// The ordinary planner receives two independent capacities. An exact lane
    /// has one: charging the tile against it and then independently spending
    /// the same cells on the lane stack produced an over-wide plan which the
    /// driver declined after planning. This search walks the existing derived
    /// row ladder, then solves the monotone column boundary exactly and spends
    /// the remainder on depth. No candidate width or iteration cap is chosen.
    fn choose_shared_exact(
        code_space: usize,
        shape: Shape,
        lane_bytes: usize,
        exact_offer: usize,
        lanes_per_exact: usize,
        block: usize,
        lane_capacity: Option<usize>,
    ) -> Option<Self> {
        if code_space == 0
            || block == 0
            || shape.m == 0
            || shape.n == 0
            || lanes_per_exact == 0
            || lane_capacity.is_some_and(|capacity| capacity < block)
        {
            return None;
        }
        let slab = slab_codes(code_space);
        if slab == 0 {
            return None;
        }
        let row_cap = tabulation_rows(code_space, blocking::L1_BYTES, lane_bytes).min(shape.m);
        for rows in ROW_TILES {
            if rows > row_cap {
                continue;
            }
            let one = Self {
                rows,
                cols: 1,
                depth: 1,
            };
            if one
                .shared_exact_charge(code_space, lanes_per_exact)
                .is_none_or(|charge| charge > exact_offer)
            {
                continue;
            }

            // Fit is monotone in the column count. Binary search reaches the
            // exact widest block in logarithmic address-width steps.
            let mut low = 1usize;
            let mut high = shape.n;
            while low < high {
                let mid = low + (high - low).div_ceil(2);
                let candidate = Self {
                    rows,
                    cols: mid,
                    depth: 1,
                };
                if candidate
                    .shared_exact_charge(code_space, lanes_per_exact)
                    .is_some_and(|charge| charge <= exact_offer)
                {
                    low = mid;
                } else {
                    high = mid - 1;
                }
            }
            let cols = low;
            let tile = rows.checked_mul(cols)?;
            let column_lanes = tile;
            let available_exact = exact_offer.checked_sub(tile)?;
            let available_lanes = available_exact.saturating_mul(lanes_per_exact); // R3-ok: an offer-size capacity
            let for_stack = available_lanes.checked_sub(column_lanes)?;
            let slab_rows = slab.checked_mul(rows)?;
            let by_offer = for_stack / slab_rows;
            let by_cache = tabulation_depth(
                code_space,
                rows,
                block,
                lane_capacity,
                blocking::L2_BYTES,
                lane_bytes,
            );
            let blocks = shape.k / block;
            let depth = by_cache.min(by_offer).min(blocks.max(1));
            let depth = if depth == 0 { 1 } else { depth };
            let plan = Self { rows, cols, depth };
            if plan
                .shared_exact_charge(code_space, lanes_per_exact)
                .is_some_and(|charge| charge <= exact_offer)
            {
                return Some(plan);
            }
        }
        None
    }

    /// The largest plan the two offers support.
    ///
    /// Rows first, from the cache budget: a wider row tile shares each decode
    /// across more outputs and puts more lane words under one vector instruction.
    /// Then the column block, as wide as the exact offer allows, because the
    /// build repeats once per column block. Then the depth, as deep as the lane
    /// offer and the cache allow.
    pub fn choose(
        code_space: usize,
        shape: Shape,
        lane_bytes: usize,
        exact_offer: usize,
        lane_offer: usize,
        block: usize,
        lane_capacity: Option<usize>,
    ) -> Option<Self> {
        if code_space == 0 || block == 0 || shape.m == 0 || shape.n == 0 {
            return None;
        }
        // A lane that holds fewer than one codec block has no table entry it
        // can build exactly. Its public declaration selects the exact stream;
        // rounding the capacity up would invoke arithmetic the lane explicitly
        // said it cannot hold.
        if lane_capacity.is_some_and(|capacity| capacity < block) {
            return None;
        }
        let row_cap = tabulation_rows(code_space, blocking::L1_BYTES, lane_bytes)
            .min(shape.m)
            .min(exact_offer);
        let slab = slab_codes(code_space);
        if slab == 0 {
            return None;
        }
        let rows = ROW_TILES
            .into_iter()
            .find(|&r| r <= row_cap && slab.saturating_mul(r) < lane_offer)?; // R3-ok: a cache or tile question, not an accumulation
        let cols = shape.n.min(exact_offer / rows);
        if cols == 0 {
            return None;
        }
        let blocks = shape.k / block;
        // The stack shares the lane offer with the column accumulation, which is
        // `rows * cols` words and is claimed first because the reduction cannot
        // proceed without it.
        let for_stack = lane_offer.saturating_sub(rows * cols); // R3-ok: a cache or tile question, not an accumulation
                                                                // The lane's own capacity is a term here, not an afterthought handled
                                                                // downstream. `row_tile` places a run when *another* chunk would not fit,
                                                                // so a `depth` already past the lane has overflowed it before the first
                                                                // placement can happen --- the guard cannot rescue the first chunk. For
                                                                // every family this library ships the ratio is enormous (16643 blocks at
                                                                // `(i8, i32)` and `Bk = 8`, against a depth the caches and the offer allow),
                                                                // so this binds on nothing shipped and costs nothing; it is here because
                                                                // a bound that only holds by arithmetic nobody checked is not a bound.
        let by_cache = tabulation_depth(
            code_space,
            rows,
            block,
            lane_capacity,
            blocking::L2_BYTES,
            lane_bytes,
        );
        let by_offer = for_stack / (slab * rows);
        // `GATHER_SLOTS` is *not* a term here, and its own doc comment says why:
        // it is the offset run's frame size, and a chunk deeper than it is walked
        // in windows of it (see `sweep`). Capping the derived depth with it made
        // the doc false and the frame size a limit on the traversal --- exactly the
        // shape R8 forbids. Every term left is a cache, an offer, or the shape.
        let depth = by_cache.min(by_offer).min(blocks.max(1));
        // A stack of no slots is not a stack. Every term above is a cache or an
        // offer, so this is the floor of a derivation and not a chosen minimum.
        let depth = if depth == 0 { 1 } else { depth };
        if for_stack < slab * rows {
            return None;
        }
        Some(Self { rows, cols, depth })
    }
}

// ---------------------------------------------------------------------------
// The traversal
// ---------------------------------------------------------------------------

/// `C := epilogue(A * W^T, C)`, with the table when the offers admit one.
///
/// Returns `()`, for the same reason [`crate::gemm`] does.
///
/// `collapse` is the row side of the same offer [`Tabulation::with_index`] is
/// the column side of: room to number the rows of `A` and hold the distinct
/// ones, so the table is built once per *distinct* row rather than once per
/// row (`CD-15`). It is the [`crate::Collapse`] buffer, unchanged --- `A` is
/// dense here, so the pass, the compaction, and the expansion are literally
/// [`crate::collapse`]'s; for the float families the pass numbers rows by bit
/// pattern, the arena tier's canonical-symbol semantics (`CD-17`, `CK-10`).
/// Offering none, or too little for what the pass
/// finds, gives the same bytes from the uncollapsed traversal; and an
/// epilogue that reads `C` declines it outright, because two rows with equal
/// rows of `A` still have different outputs when the `C` they read differs.
pub fn gemm_tabulated<E, Bd, C, O, Ep>(
    triple: &mut TabulatedTriple<'_, '_, '_, E, Bd, C, O>,
    epilogue: &Ep,
    options: GemmOptions,
    scratch: &mut Scratch<'_, E, Bd>,
    lanes: &mut Tabulation<'_>,
    collapse: &mut Collapse<'_, E, Bd>,
) where
    E: Tabulated,
    Bd: Bound,
    C: Enumerable<E, Bd> + Copy,
    O: Element + EncodeFrom<AccOf<E>>,
    Ep: Epilogue<E, O>,
{
    run(triple, epilogue, options, scratch, lanes, collapse, &mut ());
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
    collapse: &mut Collapse<'_, E, Bd>,
    census: &mut Census,
) where
    E: Tabulated,
    Bd: Bound,
    C: Enumerable<E, Bd> + Copy,
    O: Element + EncodeFrom<AccOf<E>>,
    Ep: Epilogue<E, O>,
{
    run(triple, epilogue, options, scratch, lanes, collapse, census);
}

#[allow(clippy::too_many_arguments)]
fn run<E, Bd, C, O, Ep, Lg>(
    triple: &mut TabulatedTriple<'_, '_, '_, E, Bd, C, O>,
    epilogue: &Ep,
    options: GemmOptions,
    scratch: &mut Scratch<'_, E, Bd>,
    lanes: &mut Tabulation<'_>,
    collapse: &mut Collapse<'_, E, Bd>,
    ledger: &mut Lg,
) where
    E: Tabulated,
    Bd: Bound,
    C: Enumerable<E, Bd> + Copy,
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
    if shape.k == 0 {
        // There is no real partial with which an empty-rest dense family can
        // establish acceptance, and no table entry can be read without a
        // product. Every named traversal is therefore the public StreamLane's
        // exact zero, followed by the caller's one epilogue per output.
        stream(triple, epilogue, options, scratch.take(0), ledger);
        return;
    }

    // The row collapse, before any traversal choice: two equal rows of `A`
    // name the same sum against every column of `W`, so the product is
    // computed once per *distinct* row and then expanded --- the same pass,
    // compaction, and expansion [`crate::collapse`] runs, on an `A` that is
    // always dense here (`CD-15`, mirroring `CD-12`). Every decline of the
    // offer falls through to the ordinary path, at the same bytes.
    //
    // An epilogue that reads `C` gives two rows different answers from the
    // same row of `A`, so there is no shared meaning to find. A declaration,
    // read once.
    if !epilogue.reads_c() {
        let found = E::distinct_a_rows(&triple.a, collapse.index)
            .filter(|&d| d < shape.m)
            .filter(|&d| compact(&triple.a, collapse.index, collapse.rows, d));
        if let Some(d) = found {
            // The compacted product is a product in its own right: `d x k`
            // against the same `W`, written to the first `d` rows of the same
            // `C`. The recursion plans at `m = d` --- offer checks and the
            // column collapse included --- and is offered no row collapse: the
            // compacted rows are pairwise distinct, so a second pass could
            // find nothing.
            let compacted = {
                let a = MatView::row_major(&collapse.rows[..d * shape.k], d, shape.k);
                match (a, triple.c.top_rows(d)) {
                    (Some(a), Some(c)) => TabulatedTriple::new(a, triple.w, c).ok(),
                    _ => None,
                }
            };
            if let Some(mut compacted) = compacted {
                run(
                    &mut compacted,
                    epilogue,
                    options,
                    scratch,
                    lanes,
                    &mut Collapse::none(),
                    ledger,
                );
                expand(&mut triple.c, &collapse.index[..shape.m], shape.m, shape.n);
                return;
            }
        }
    }

    let space = C::CODE_SPACE;
    let block = <C as uor_matmul_codec::Codec<E, Bd>>::MAX_BLOCK;
    // What the operand's *declaration* says about addressing it, read from the
    // three facts `Manifest::of` mints this artifact's address from --- the
    // tier, the block and the bound --- and from nothing the operand holds. The
    // manifest's other value-bearing fields are its two digests, and no term
    // here reads one, so no run of this traversal can have probed a code to
    // decide which factorization to take (`CS-10`). The orientation half of the
    // same claim was settled at `TabulatedTriple::new`, which is where `W` was
    // declared `n x k`.
    let addressing = Addressing::of(
        <C as uor_matmul_codec::Codec<E, Bd>>::TIER,
        block,
        Bd::VALUE,
    );
    // A code stream whose blocks are not a fixed width has no `p`-th block to
    // index, so there is nothing for a table to be built per block of. Asked of
    // the *type* and not of the tier, because a composing tier reports its own
    // token while inheriting its inner codec's width. The one variable-length
    // tier does not implement `Enumerable`, so this is unreachable through the
    // shipped codecs; it is here because the trait does not forbid it.
    let addressable = <C as uor_matmul_codec::Codec<E, Bd>>::IS_FIXED_WIDTH
        && addressing.addresses_an_element()
        && space > 0;
    if !addressable || options.traversal == Traversal::OutputMajor {
        decline(triple, epilogue, options, scratch, ledger);
        return;
    }

    let (words, index) = (&mut *lanes.lanes, &mut *lanes.index);

    // The modular lane is admissible exactly when the caller asked to wrap into
    // an output no wider than it: a question about two declarations --- the
    // encode mode and the output type --- asked once, here, at the boundary,
    // mirroring the dense side (`kernel.rs`). The row-collapse recursion above
    // re-runs this body with the same `options` and the same `O`, so both
    // levels of it decide identically (`CU-08`). One branch, and none in the
    // loops: the chosen lane is a type from here down.
    if matches!(options.encode, EncodeMode::Wrapping) && E::modular_table_admitted(O::BITS) {
        run_lane::<E, Bd, C, O, Ep, Lg, E::ModLane>(
            triple,
            epilogue,
            options,
            scratch,
            words,
            index,
            |backend, bound, _sign_book, rows, group, block| {
                E::table_spec_modular(backend, bound, rows, group, block)
            },
            E::lanes_modular,
            ledger,
        );
    } else {
        run_lane::<E, Bd, C, O, Ep, Lg, E::Lane>(
            triple,
            epilogue,
            options,
            scratch,
            words,
            index,
            E::table_spec,
            E::lanes,
            ledger,
        );
    }
}

/// The sequence lookup for one lane, as a function pointer: the family's own
/// [`Tabulated::table_spec`] or [`Tabulated::table_spec_modular`], named at the
/// boundary because the two lanes are two types.
type SpecOf<E, L> = fn(Backend, u128, bool, usize, usize, usize) -> TableSpec<E, L>;

/// The offer-relabelling half of a lane, likewise: [`Tabulated::lanes`] or
/// [`Tabulated::lanes_modular`].
type LanesOf<E, L> = for<'s> fn(&'s mut [i64], &'s mut [AccOf<E>], usize) -> Option<&'s mut [L]>;

/// The table traversal in the lane the boundary chose.
///
/// Written once, generic over the lane: the exact accumulator and the modular
/// word are two types, and the two `fn` pointers are how the boundary's
/// declaration reaches the two places a type cannot be threaded --- the
/// sequence lookup and the offer relabelling, both of which are the family's
/// own associated functions.
#[allow(clippy::too_many_arguments)]
fn run_lane<E, Bd, C, O, Ep, Lg, L>(
    triple: &mut TabulatedTriple<'_, '_, '_, E, Bd, C, O>,
    epilogue: &Ep,
    options: GemmOptions,
    scratch: &mut Scratch<'_, E, Bd>,
    words: &mut [i64],
    index: &mut [usize],
    spec_of: SpecOf<E, L>,
    lanes_of: LanesOf<E, L>,
    ledger: &mut Lg,
) where
    E: Tabulated,
    Bd: Bound,
    C: Enumerable<E, Bd> + Copy,
    O: Element + EncodeFrom<AccOf<E>>,
    Ep: Epilogue<E, O>,
    Lg: Ledger,
    L: Lane<E>,
{
    let shape = triple.shape();
    let space = C::CODE_SPACE;
    let block = <C as uor_matmul_codec::Codec<E, Bd>>::MAX_BLOCK;
    let lane_bytes = core::mem::size_of::<L>();
    let exact_offer = scratch.accumulators();
    if lane_bytes == 0 {
        decline(triple, epilogue, options, scratch, ledger);
        return;
    }
    // Where the lane lives. A family whose lane *is* the exact accumulator
    // reads that offer, because that is where a word so wide already lives ---
    // and its modular lane reads the same words, relabelled, so a narrower
    // lane simply fits more of them into the offer. Every other family's lane
    // offer is `i64`-shaped, read as however many lane words fit in the same
    // bytes.
    let offered = if E::LANE_IS_EXACT {
        let per_word = core::mem::size_of::<AccOf<E>>() / lane_bytes;
        exact_offer.saturating_mul(per_word) // R3-ok: a size query, not an accumulation
    } else {
        core::mem::size_of_val(&*words) / lane_bytes
    };
    let lane_capacity = E::probe_capacity::<L>(Bd::VALUE);
    let plan = if E::LANE_IS_EXACT {
        let lanes_per_exact = core::mem::size_of::<AccOf<E>>() / lane_bytes;
        Plan::choose_shared_exact(
            space,
            shape,
            lane_bytes,
            exact_offer,
            lanes_per_exact,
            block,
            lane_capacity,
        )
    } else {
        Plan::choose(
            space,
            shape,
            lane_bytes,
            exact_offer,
            offered,
            block,
            lane_capacity,
        )
    };
    let Some(plan) = plan else {
        decline(triple, epilogue, options, scratch, ledger);
        return;
    };
    // Both declarations come from the sequences that will run, at the tile the
    // plan resolved to --- never from a constant standing in for them.
    let table = spec_of(
        options.backend,
        Bd::VALUE,
        C::SIGN_BIT_BOOK,
        plan.rows,
        column_group(plan.rows),
        block,
    );
    let steps = E::dense_steps(
        options.backend,
        Bd::VALUE,
        plan.rows,
        block.saturating_mul(table.lanes_per_add), // R3-ok: a size or cost query, not an accumulation
    );
    if !admits(
        options.traversal,
        space,
        block,
        plan,
        steps,
        lane_bytes,
        &table,
    ) {
        decline(triple, epilogue, options, scratch, ledger);
        return;
    }

    // The column question and its dictionary are table work, so they begin
    // only after admission. Once the global classes are known, the same dead
    // dictionary is the exact addressed-index set for the scale, decoded book,
    // and pointwise builds; no parallel bitmap or code-space-sized allocation
    // exists.
    let distinct =
        distinct_columns::<E, Bd, C>(triple.w.codes(), triple.w.codes_per_row(), shape.n, index);
    let repeated = matches!(distinct, Some(d) if d < shape.n);
    // A one-entry enumeration already has the smallest possible scale/book and
    // build presentation. Its dictionary can still collapse equal columns, but
    // an addressed-entry set cannot remove another operation. Wider pointwise
    // books reuse the dead dictionary to name exactly the entries they need.
    let need_entries = block == 1 && !C::SIGN_BIT_BOOK && space > 1;
    let (collapse, mut entries) = if distinct.is_some() {
        column_workspace(index, shape.n, plan.cols, repeated, need_entries)
    } else {
        (None, None)
    };
    let addressed_count = if need_entries {
        entries
            .as_mut()
            .and_then(|set| set.collect::<E, Bd, C>(triple.w.codes()))
    } else {
        None
    };

    // The panels' own declaration, asked only now that the table is selected:
    // a call the predicate declines never pays the walk. `None` declines the
    // table --- the panels are not the lane's alphabet --- and the dense
    // route answers at the same bytes, with the census saying which ran.
    let addressed =
        addressed_count.and_then(|count| entries.as_ref().map(|set| set.collected(count)));
    let Some(scale) = addressed_lane_scale(&triple.a, &triple.w, addressed, ledger) else {
        decline(triple, epilogue, options, scratch, ledger);
        return;
    };
    // The walk's capacity answer can only shrink the plan the probe made:
    // rows and columns are capacity-independent, so it is applied here rather
    // than re-planned. For a family whose capacity is a function of the
    // alphabet this is the value `Plan::choose` already used, and the minimum
    // is the identity.
    let mut plan = plan;
    let observed_run = E::lane_run::<L>(Bd::VALUE, &scale);
    let data_dependent_lane = lane_capacity.is_none() && observed_run.is_some();
    // A global certificate is already optimal only when it carries the entire
    // reduction in one placement. Otherwise the q lane derives source-local
    // envelopes: a call-wide span is a totality bound, not an optimal schedule.
    let local_envelopes = data_dependent_lane && observed_run.is_some_and(|run| run < shape.k);
    if let Some(run) = observed_run {
        if run < block && !data_dependent_lane {
            decline(triple, epilogue, options, scratch, ledger);
            return;
        }
        if run >= block && !local_envelopes {
            plan.depth = (run / block).min(plan.depth); // R3-ok: a lane-width question, not an accumulation
        }
    }
    // `probe=None` followed by a real post-walk capacity is the public nominal
    // declaration of a contextual panel protocol. It is parametric (no type or
    // callback-value identity) and counts every in-place projection that the
    // paired table builder consumes. An exact lane such as f64 answers None at
    // both points and retains the ordinary element panel.
    let projection_decodes = data_dependent_lane;

    let want = plan.lane_words(space);
    let tile = plan.rows * plan.cols;
    // The accumulator words the lane words occupy: one each when the lane *is*
    // the accumulator, fewer for a modular lane reading the same offer.
    let take = if E::LANE_IS_EXACT {
        tile + want.div_ceil(core::mem::size_of::<AccOf<E>>() / lane_bytes)
    } else {
        tile
    };
    if take > exact_offer {
        decline(triple, epilogue, options, scratch, ledger);
        return;
    }
    // Keep the whole panel offer. The fixed head remains the decoded book and
    // activation tile promised by `suggested_tabulation_panel`; complete rows
    // of projected `A` occupy only the caller-offered tail. No new storage is
    // required, and an exact-sized offer continues to have an empty cache.
    let panel_offer = scratch.len();
    let (panel, accumulators) = scratch.split(panel_offer, take);
    let addressed =
        addressed_count.and_then(|count| entries.as_ref().map(|set| set.collected(count)));
    if !decode_book(
        &triple.w,
        panel,
        scale.base_b,
        addressed,
        projection_decodes,
        ledger,
    ) {
        decline(triple, epilogue, options, scratch, ledger);
        return;
    }
    if let Some(set) = entries.as_mut() {
        set.release_collected();
    }
    let (exact, rest) = accumulators.split_at_mut(tile);
    let Some(lanes) = lanes_of(words, rest, want) else {
        decline(triple, epilogue, options, scratch, ledger);
        return;
    };
    tabulate::<E, Bd, C, O, Ep, Lg, L>(
        triple,
        epilogue,
        options,
        exact,
        lanes,
        panel,
        collapse,
        entries,
        plan,
        scale,
        spec_of,
        projection_decodes,
        local_envelopes,
        ledger,
    );
}

/// Does the caller's named traversal admit the table this plan describes?
///
/// [`Traversal::Blocked`] is the default and takes the table when it is the
/// cheaper factorization. [`Traversal::Tabulated`] takes it wherever one fits,
/// whether or not the op count says it wins: `CD-13` needs that to compare bytes
/// on both sides of the predicate, and a caller measuring its own shapes needs it
/// for the same reason.
fn admits<E, L>(
    traversal: Traversal,
    code_space: usize,
    block: usize,
    plan: Plan,
    steps: Steps,
    lane_bytes: usize,
    table: &TableSpec<E, L>,
) -> bool {
    match traversal {
        Traversal::OutputMajor => false,
        Traversal::Blocked => tabulation_pays_for_spec(
            code_space,
            block,
            plan.cols,
            plan.rows,
            steps,
            blocking::L1_BYTES,
            lane_bytes,
            table,
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
///
/// `pub` so a measurement harness walks the same ladder the traversal does
/// rather than restating it.
pub const ROW_TILES: [usize; 5] = [16, 8, 4, 2, 1];

/// A private one-element view of the indices the caller's coded operand
/// actually addresses.
///
/// This does not copy or reinterpret a code. `usize` is the enumeration's
/// existing coordinate, and the original codec remains the only decoder. It
/// lets the locked `Tabulated::lane_scale` protocol inspect an exact distinct
/// sub-book through the same `CodedMatrix` abstraction, without a type test or
/// a second float-specific scale operation.
#[derive(Clone, Copy)]
struct AddressedCodec<C>(C);

impl<E, Bd, C> Codec<E, Bd> for AddressedCodec<C>
where
    E: Element,
    Bd: Bound,
    C: Enumerable<E, Bd> + Copy,
{
    type Code = usize;

    const MAX_BLOCK: usize = 1;
    const TIER: TierId = C::TIER;

    fn decode_element(&self, code: usize, i: usize) -> Alphabet<E, Bd> {
        self.0.decode_element(C::code_at(code), i)
    }
}

impl<E, Bd, C> Enumerable<E, Bd> for AddressedCodec<C>
where
    E: Element,
    Bd: Bound,
    C: Enumerable<E, Bd> + Copy,
{
    const CODE_SPACE: usize = C::CODE_SPACE;

    fn code_at(index: usize) -> usize {
        index
    }

    fn index_of(code: usize) -> usize {
        if C::CODE_SPACE == 0 {
            0
        } else {
            code % C::CODE_SPACE
        }
    }
}

/// One scalar coordinate of an existing fixed-width codec block.
///
/// The stored code and its canonical index are unchanged; only the decoded
/// coordinate is selected. This lets a data-dependent lane ask its locked
/// scale protocol about one source atom and build that atom through the same
/// `TableBuild` abstraction, without copying a code stream or adding a public
/// scalar-codec type.
struct ScalarCodec<'a, C> {
    codec: &'a C,
    coordinate: usize,
}

impl<C> Copy for ScalarCodec<'_, C> {}

impl<C> Clone for ScalarCodec<'_, C> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<E, Bd, C> Codec<E, Bd> for ScalarCodec<'_, C>
where
    E: Element,
    Bd: Bound,
    C: Enumerable<E, Bd>,
{
    type Code = C::Code;

    const MAX_BLOCK: usize = 1;
    const TIER: TierId = C::TIER;

    fn decode_element(&self, code: Self::Code, _: usize) -> Alphabet<E, Bd> {
        self.codec.decode_element(code, self.coordinate)
    }
}

impl<E, Bd, C> Enumerable<E, Bd> for ScalarCodec<'_, C>
where
    E: Element,
    Bd: Bound,
    C: Enumerable<E, Bd>,
{
    const CODE_SPACE: usize = C::CODE_SPACE;

    fn code_at(index: usize) -> Self::Code {
        C::code_at(index)
    }

    fn index_of(code: Self::Code) -> usize {
        C::index_of(code)
    }
}

fn addressed_lane_scale<E, Bd, C, Lg>(
    a: &MatView<'_, Alphabet<E, Bd>>,
    w: &CodedMatrix<'_, E, Bd, C>,
    addressed: Option<&[usize]>,
    ledger: &mut Lg,
) -> Option<LaneScale>
where
    E: Tabulated,
    Bd: Bound,
    C: Enumerable<E, Bd> + Copy,
    Lg: Ledger,
{
    let Some(addressed) = addressed else {
        return E::lane_scale(a, w, ledger);
    };
    let codec = AddressedCodec(*w.codec());
    let distinct = CodedMatrix::new(codec, addressed.len(), 1, addressed)
        .expect("a nonempty block-one index set is a fixed-width coded matrix");
    E::lane_scale(a, &distinct, ledger)
}

/// Decode the codebook cells this call can address into the caller's panel
/// offer.
///
/// `book[index * block + t]` is element `t` of the `index`-th codeword, so one
/// codeword is a contiguous run and the build walks it against the activation
/// tile without a stride. The codec is consulted `code_space * block` times for
/// the *whole call* rather than once per row tile and per block of the reduction
/// --- measured, re-deriving it per tile was half the build.
/// A pointwise block-one table uses the reclaimed column dictionary to fill each
/// distinct addressed canonical slot once. If that generic set cannot name the
/// call's whole support, the total raw-short/full-book factorization remains;
/// neither case allocates a bitmap or side carrier. The Gray sign walk keeps the
/// full book because its declaration is a whole-enumeration recurrence rather
/// than independent entries.
///
/// Every integral entry is unchanged because its contextual base is zero. The
/// `f32` entry is not numerically pre-scaled: the same four bytes are relabelled
/// in place with the grade relative to the codebook's observed base, then the
/// paired [`Scaled64`] consumer contracts that q address by lookup and addition.
/// No standalone float value, integer reification, or multiply is introduced
/// between the projection and that consumer (`CD-20`, `CD-32`).
///
/// This is the only reason the tabulated traversal wants a panel offer at all.
/// Without one there is nowhere to put the decoded book, and the traversal
/// streams --- the same rule every other offer in this library follows.
fn decode_book<E, Bd, C, Lg>(
    w: &CodedMatrix<'_, E, Bd, C>,
    panel: &mut [Alphabet<E, Bd>],
    base: i32,
    addressed: Option<&[usize]>,
    projection_decodes: bool,
    ledger: &mut Lg,
) -> bool
where
    E: Tabulated,
    Bd: Bound,
    C: Enumerable<E, Bd>,
    Lg: Ledger,
{
    let space = C::CODE_SPACE;
    let block = C::MAX_BLOCK;
    let codec = w.codec();
    // The book *and* the widest activation tile a row tile can pack, because the
    // build walks both and a panel that held only the book would leave the tile
    // with nowhere to go.
    if panel.len() < suggested_tabulation_panel(space, block) {
        return false;
    }
    let sparse_book = block == 1 && !C::SIGN_BIT_BOOK && w.codes().len() < space;
    if let Some(addressed) = addressed {
        for &index in addressed {
            let entry = codec.decode_element(C::code_at(index), 0);
            panel[index] = bytemuck::TransparentWrapper::wrap(E::prescale(entry.get(), base));
        }
        let decoded = count_factor(addressed.len());
        ledger.decoded(decoded);
        if projection_decodes {
            ledger.decoded(decoded);
        }
    } else if sparse_book {
        for &stored in w.codes() {
            let index = C::index_of(stored);
            let entry = codec.decode_element(C::code_at(index), 0);
            panel[index] = bytemuck::TransparentWrapper::wrap(E::prescale(entry.get(), base));
        }
        let decoded = count_factor(w.codes().len());
        ledger.decoded(decoded);
        if projection_decodes {
            ledger.decoded(decoded);
        }
    } else {
        for index in 0..space {
            for t in 0..block {
                let entry = codec.decode_element(C::code_at(index), t);
                panel[index * block + t] =
                    bytemuck::TransparentWrapper::wrap(E::prescale(entry.get(), base));
            }
        }
        let decoded = count_product2(space, block);
        ledger.decoded(decoded);
        if projection_decodes {
            ledger.decoded(decoded);
        }
    }
    true
}

/// How much panel offer the decoded codebook occupies.
///
/// A *query*, like [`crate::suggested_scratch`]. Offering less selects the
/// streaming traversal, at the same bytes (`CD-13`).
pub const fn suggested_tabulation_panel(code_space: usize, block: usize) -> usize {
    // The decoded codebook, plus the widest activation tile a row tile can pack.
    code_space
        .saturating_add(ROW_TILES[0]) // R3-ok: a size query, not an accumulation
        .saturating_mul(block) // R3-ok: a size query, not an accumulation
}

/// Output columns reduced together in one pass over the stack.
/// Slots one index run covers, when there is an index run to build.
///
/// A frame size, and only that: a chunk deeper than this is walked in windows of
/// it, so the traversal's own depth --- which [`tabulation_depth`] derives from
/// L2 --- is not constrained by it and neither is any caller's `k` (R8). A codec
/// whose stream already addresses the enumeration builds no run at all and never
/// reaches this.
const GATHER_SLOTS: usize = 32;

/// Lane words one gather call keeps in flight.
///
/// Every entry read is a random access into a table that does not fit L1 once
/// the stack is deep, so what the column loop is bound by is how many of those
/// reads are *overlapped*. A tile of `rows` gives one call `rows` of them; below
/// the widest tile it takes more than one column to reach the same number, and
/// measured that is exactly the shape of the loss: at a one-row tile, a group of
/// one runs at 3.1 Gmac/s where the same traversal in groups runs at three times
/// that.
///
/// So the group is derived, not chosen: `COLUMN_LANES / rows`, which is one at
/// the widest tile and widens as the tile narrows. The lane state stays the same
/// size whatever the shape, which is what keeps it in registers.
const COLUMN_LANES: usize = 16;

/// The widest column group any tile asks for, and therefore the offset run one
/// call materializes.
///
/// A frame size, not a limit on anything a caller supplies: `COLUMN_LANES` over
/// the narrowest tile, which is one.
const MAX_GROUP: usize = COLUMN_LANES;

/// Output columns one gather call reduces at once, at a tile of `rows`.
///
/// `pub` so a measurement harness groups the way the sweep does.
pub const fn column_group(rows: usize) -> usize {
    if rows == 0 {
        0
    } else if rows >= COLUMN_LANES {
        1
    } else {
        COLUMN_LANES / rows
    }
}

/// Which output columns repeat, and where each one's first occurrence is.
///
/// Two columns whose *index* streams agree read the same table entries in the
/// same order, so their accumulations are equal --- not nearly, identically, and
/// for the same reason [`crate::collapse`]'s rows are: an exact sum is a function
/// of the multiset of its products, so equal operands give equal sums by
/// definition. A classical `sgemm` sharing a column would additionally have to
/// argue that its *order* of additions was the same. This one has nothing to
/// argue.
///
/// Indices, not raw codes. That is the coarser relation and therefore the better
/// one --- two codes at the same index decode alike, so a codec whose enumeration
/// carries duplicates has them collapsed here too --- and it asks nothing of the
/// code type, which [`uor_matmul_codec::Codec`] does not constrain and should not.
///
/// `None` when the offer is too short to answer, which is the caller getting the
/// uncollapsed traversal at the same bytes.
fn distinct_columns<E, Bd, C>(
    codes: &[C::Code],
    codes_per_row: usize,
    n: usize,
    index: &mut [usize],
) -> Option<usize>
where
    E: Element,
    Bd: Bound,
    C: Enumerable<E, Bd>,
{
    if n == 0 || codes_per_row == 0 || codes.len() < n.checked_mul(codes_per_row)? {
        return None;
    }
    // Half full at worst, so an empty slot always exists and a probe terminates.
    let table = n.checked_mul(2)?.checked_next_power_of_two()?;
    if index.len() < n.checked_add(table.checked_mul(2)?)? {
        return None;
    }
    let (position, rest) = index.split_at_mut(n);
    let (slot, key) = rest.split_at_mut(table);
    // Zero is "empty"; a slot holds one more than the column it names.
    slot[..table].fill(0);
    let mut distinct = 0usize;
    for (j, position) in position.iter_mut().enumerate() {
        let run = &codes[j * codes_per_row..(j + 1) * codes_per_row];
        let hash = column_hash::<E, Bd, C>(run, table);
        let mut probe = hash;
        loop {
            match slot[probe] {
                0 => {
                    slot[probe] = j + 1;
                    key[probe] = hash;
                    *position = j;
                    distinct += 1;
                    break;
                }
                seen => {
                    let seen = seen - 1;
                    let other = &codes[seen * codes_per_row..(seen + 1) * codes_per_row];
                    if key[probe] == hash && columns_equal::<E, Bd, C>(run, other) {
                        *position = seen;
                        break;
                    }
                    probe += 1;
                    if probe == table {
                        probe = 0;
                    }
                }
            }
        }
    }
    Some(distinct)
}

/// Do two columns read the same table entries in the same order?
fn columns_equal<E, Bd, C>(a: &[C::Code], b: &[C::Code]) -> bool
where
    E: Element,
    Bd: Bound,
    C: Enumerable<E, Bd>,
{
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(&x, &y)| C::index_of(x) == C::index_of(y))
}

/// Radix recurrence over one column's canonical index stream.
///
/// The hash is only a filter: [`columns_equal`] remains the authority on every
/// hit. Its odd radix is the model-derived Atlas modality: double the prior
/// coordinate, add it once, then add the source coordinate. Its modulus is the
/// already-derived dictionary extent. The model-owned measured prefix bounds
/// filter work only; coordinates beyond it remain governed by exact stream
/// equality. Keeping the complete length and sampled recurrence unreduced and
/// taking one final remainder removes every intermediate modular branch. This
/// preserves the exact dictionary address because reducing the initial
/// coordinate before a radix polynomial cannot change its final residue. The
/// model proves the widest 16-coordinate value still needs only 90 of the
/// carrier's 128 bits. No seed, multiply, rotate, shift, or packed mask
/// participates in float traversal selection. The length is the initial
/// coordinate, so unequal stream extents do not acquire the same empty-word
/// spelling by construction.
fn column_hash<E, Bd, C>(run: &[C::Code], modulus: usize) -> usize
where
    E: Element,
    Bd: Bound,
    C: Enumerable<E, Bd>,
{
    debug_assert!(modulus != 0);
    let mut hash = run.len() as u128;
    let measured = run.len().min(crate::float::COLUMN_HASH_PREFIX);
    for &code in &run[..measured] {
        let doubled = hash + hash;
        hash = doubled + hash + C::index_of(code) as u128;
    }
    (hash % modulus as u128) as usize
}

/// The column collapse, made block-local.
///
/// `first[j]` is the first occurrence of column `j`'s equivalence class
/// *inside `j`'s column block*, relative to the block's start. It has to be
/// block-local: a first occurrence in an earlier block no longer has an
/// accumulator to copy from --- that block was encoded and the `exact` buffer
/// reused before this one began. A class spread across blocks is therefore
/// charged once per block it appears in, which is the most a narrow offer can
/// extract.
///
/// `identity[b]` is nonzero when block `b` repeats nothing, and such a block
/// takes the consecutive loop: an indexed load per column just to learn there
/// is nothing to skip halved every shape it was measured on (ANALYSIS.md).
#[derive(Clone, Copy)]
struct ColumnMap<'a> {
    first: &'a [usize],
    identity: &'a [usize],
}

/// The dead column dictionary, reused as a set of addressed table indices.
///
/// `seen` is an open-addressed table containing `index + 1`; an admitted slab
/// cannot contain `usize::MAX` codes, so the addition is exact. `occupied`
/// records probe slots, which clears only the entries one reduction position
/// inserted rather than the whole dictionary. No storage scales with the code
/// space: the table is the caller's existing `with_index` offer and is at least
/// twice the output width.
struct EntrySet<'a> {
    seen: &'a mut [usize],
    occupied: &'a mut [usize],
    used: usize,
}

enum EntryInsert {
    New,
    Present,
    Full,
}

impl EntrySet<'_> {
    fn insert(&mut self, index: usize) -> EntryInsert {
        let Some(key) = index.checked_add(1) else {
            return EntryInsert::Full;
        };
        if self.seen.is_empty() {
            return EntryInsert::Full;
        }
        let extent = self.seen.len();
        let mut probe = if index < extent {
            index
        } else {
            index % extent
        };
        for _ in 0..self.seen.len() {
            match self.seen[probe] {
                0 if self.used < self.occupied.len() => {
                    self.seen[probe] = key;
                    self.occupied[self.used] = probe;
                    self.used += 1;
                    return EntryInsert::New;
                }
                0 => return EntryInsert::Full,
                present if present == key => return EntryInsert::Present,
                _ => {
                    probe += 1;
                    if probe == extent {
                        probe = 0;
                    }
                }
            }
        }
        EntryInsert::Full
    }

    fn len(&self) -> usize {
        self.used
    }

    fn index(&self, at: usize) -> usize {
        self.seen[self.occupied[at]] - 1
    }

    fn clear(&mut self) {
        for &probe in &self.occupied[..self.used] {
            self.seen[probe] = 0;
        }
        self.used = 0;
    }

    /// Collect a call-wide addressed book when this offer can name every
    /// distinct index. The contiguous result temporarily reuses `occupied`;
    /// each probe slot is cleared as its index is transferred. A larger set
    /// returns `None` before any decode or build is issued, so the ordinary
    /// complete-book presentation remains truthful.
    fn collect<E, Bd, C>(&mut self, codes: &[C::Code]) -> Option<usize>
    where
        E: Element,
        Bd: Bound,
        C: Enumerable<E, Bd>,
    {
        debug_assert_eq!(self.used, 0);
        for &code in codes {
            if matches!(self.insert(C::index_of(code)), EntryInsert::Full) {
                self.clear();
                return None;
            }
        }
        let count = self.used;
        for at in 0..count {
            let slot = self.occupied[at];
            self.occupied[at] = self.seen[slot] - 1;
            self.seen[slot] = 0;
        }
        Some(count)
    }

    fn collected(&self, count: usize) -> &[usize] {
        &self.occupied[..count]
    }

    fn release_collected(&mut self) {
        self.used = 0;
    }
}

/// Rewrite the global column classes block-relative and partition their dead
/// dictionary as an addressed-entry set.
///
/// `distinct_columns` laid the offer out as
/// `position[n] | slot[table] | key[table]`, with `table >= 2n`. When columns
/// repeat, `position` becomes the block-local first map and `key[..blocks]` the
/// identity flags. Only a pointwise book with more than one coordinate needs an
/// addressed-entry set; then `slot` and the rest of `key` are disjoint storage,
/// or the whole dictionary is reclaimed when no columns repeat. Other books do
/// not pay its clear. A short/absent offer has no place to remember an unbounded
/// set of generic codec indices, so it keeps the same total table factorization
/// and truthfully issues duplicate builds.
fn column_workspace(
    index: &mut [usize],
    n: usize,
    cols: usize,
    repeated: bool,
    need_entries: bool,
) -> (Option<ColumnMap<'_>>, Option<EntrySet<'_>>) {
    let Some(table) = n.checked_mul(2).and_then(usize::checked_next_power_of_two) else {
        return (None, None);
    };
    let Some(dictionary_words) = table.checked_mul(2) else {
        return (None, None);
    };
    let Some(want) = n.checked_add(dictionary_words) else {
        return (None, None);
    };
    if index.len() < want || table == 0 {
        return (None, None);
    }

    if !repeated {
        if !need_entries {
            return (None, None);
        }
        let (seen, occupied) = index.split_at_mut(table);
        seen.fill(0);
        return (
            None,
            Some(EntrySet {
                seen,
                occupied,
                used: 0,
            }),
        );
    }

    let blocks = n.div_ceil(cols);
    let (first, rest) = index.split_at_mut(n);
    let (seen, keys) = rest.split_at_mut(table);
    let (identity, occupied) = keys.split_at_mut(blocks);
    identity.fill(1);
    for start in (0..n).step_by(cols) {
        let end = (start + cols).min(n);
        // The dictionary still contains hash probes. Zero exactly the direct
        // representative cells this block will consult before reading them.
        for &representative in &first[start..end] {
            seen[representative] = 0;
        }
        for j in start..end {
            let representative = first[j];
            let prior = seen[representative];
            first[j] = if prior > start {
                prior - 1 - start
            } else {
                seen[representative] = j + 1;
                j - start
            };
            if first[j] != j - start {
                identity[j / cols] = 0;
            }
        }
    }
    let collapse = Some(ColumnMap { first, identity });
    if !need_entries {
        return (collapse, None);
    }
    seen.fill(0);
    (
        collapse,
        Some(EntrySet {
            seen,
            occupied,
            used: 0,
        }),
    )
}

/// Regrade one subset's product certificate to the call's already-projected
/// bases. Values beyond `cap` share the single `cap + 1` singleton marker; no
/// wider arithmetic value is needed by the scheduler.
fn regrade_envelope(local: LaneScale, call: LaneScale, cap: u128) -> u128 {
    let singleton = cap + 1;
    if local.per_step > cap {
        return singleton;
    }
    let Some(a) = local.base_a.checked_sub(call.base_a) else {
        return singleton;
    };
    let Some(b) = local.base_b.checked_sub(call.base_b) else {
        return singleton;
    };
    let Ok(a) = u32::try_from(a) else {
        return singleton;
    };
    let Ok(b) = u32::try_from(b) else {
        return singleton;
    };
    let Some(distance) = a.checked_add(b) else {
        return singleton;
    };
    let mut bound = local.per_step;
    for _ in 0..distance {
        if bound > cap || bound > cap - bound {
            return singleton;
        }
        bound += bound;
    }
    bound
}

/// Least declaration available through the locked lane protocol for one scalar
/// source coordinate, common to every row/column lane that will share its
/// placement boundary.
///
/// `A` is presented as the current row tile's one source column. Each distinct
/// output column presents its one stored code through [`ScalarCodec`]; taking
/// the maximum makes the resulting nonnegative envelope safe for every lane in
/// the carried tile. The returned certificate is expressed at the call's bases
/// so certificates add directly in the source-ordered recurrence.
#[allow(clippy::too_many_arguments)]
fn scalar_envelope<E, Bd, C, O, Lg>(
    triple: &TabulatedTriple<'_, '_, '_, E, Bd, C, O>,
    row0: usize,
    rows: usize,
    col0: usize,
    cols: usize,
    p: usize,
    coordinate: usize,
    collapsed: Option<&[usize]>,
    call_scale: LaneScale,
    cap: u128,
    ledger: &mut Lg,
) -> u128
where
    E: Tabulated,
    Bd: Bound,
    C: Enumerable<E, Bd>,
    Lg: Ledger,
{
    let block = C::MAX_BLOCK;
    let source = p * block + coordinate;
    let a = triple
        .a
        .subview(row0, source, rows, 1)
        .expect("the scalar coordinate lies in the conformant reduction");
    let codes = triple.w.codes();
    let codes_per_row = triple.w.codes_per_row();
    let scalar = ScalarCodec {
        codec: triple.w.codec(),
        coordinate,
    };
    let mut envelope = 0u128;
    for j in 0..cols {
        if collapsed.is_some_and(|first| first[j] != j) {
            continue;
        }
        let code = &codes[(col0 + j) * codes_per_row + p];
        let one = CodedMatrix::new(scalar, 1, 1, core::slice::from_ref(code))
            .expect("one scalar code is a conformant fixed-width matrix");
        let local = E::lane_scale(&a, &one, ledger)
            .expect("an admitted lane scale remains admitted on a source subset");
        envelope = envelope.max(regrade_envelope(local, call_scale, cap));
    }
    envelope
}

#[allow(clippy::too_many_arguments)]
fn build_source_block<E, Bd, C, L, Lg>(
    table: &mut Table<'_, L>,
    spec: &TableSpec<E, L>,
    block: usize,
    book: &[E],
    acts: &[E],
    slot: usize,
    p: usize,
    col0: usize,
    cols: usize,
    codes_per_row: usize,
    codes: &[C::Code],
    collapsed: Option<&[usize]>,
    addressed: usize,
    mut entries: Option<&mut EntrySet<'_>>,
    ledger: &mut Lg,
) where
    E: Element,
    Bd: Bound,
    C: Enumerable<E, Bd>,
    L: Lane<E>,
    Lg: Ledger,
{
    if block != 1 || C::SIGN_BIT_BOOK {
        table.build(spec, block, book, acts, slot, ledger);
        return;
    }
    if let Some(set) = entries.as_mut() {
        let mut complete = true;
        for j in 0..cols {
            if collapsed.is_some_and(|first| first[j] != j) {
                continue;
            }
            let code = codes[(col0 + j) * codes_per_row + p];
            if matches!(set.insert(C::index_of(code)), EntryInsert::Full) {
                complete = false;
                break;
            }
        }
        if complete && set.len() < table.code_space() {
            for at in 0..set.len() {
                table.build_entry(spec, block, book, acts, slot, set.index(at), ledger);
            }
            set.clear();
        } else {
            set.clear();
            table.build(spec, block, book, acts, slot, ledger);
        }
    } else if addressed < table.code_space() {
        for j in 0..cols {
            if collapsed.is_some_and(|first| first[j] != j) {
                continue;
            }
            let code = codes[(col0 + j) * codes_per_row + p];
            table.build_entry(spec, block, book, acts, slot, C::index_of(code), ledger);
        }
    } else {
        table.build(spec, block, book, acts, slot, ledger);
    }
}

#[allow(clippy::too_many_arguments)]
fn build_source_scalar<E, Bd, C, L, Lg>(
    table: &mut Table<'_, L>,
    spec: &TableSpec<E, L>,
    source_block: usize,
    coordinate: usize,
    book: &[E],
    acts: &[E],
    p: usize,
    col0: usize,
    cols: usize,
    codes_per_row: usize,
    codes: &[C::Code],
    collapsed: Option<&[usize]>,
    addressed: usize,
    mut entries: Option<&mut EntrySet<'_>>,
    ledger: &mut Lg,
) where
    E: Element,
    Bd: Bound,
    C: Enumerable<E, Bd>,
    L: Lane<E>,
    Lg: Ledger,
{
    if let Some(set) = entries.as_mut() {
        let mut complete = true;
        for j in 0..cols {
            if collapsed.is_some_and(|first| first[j] != j) {
                continue;
            }
            let code = codes[(col0 + j) * codes_per_row + p];
            if matches!(set.insert(C::index_of(code)), EntryInsert::Full) {
                complete = false;
                break;
            }
        }
        if complete && set.len() < table.code_space() {
            for at in 0..set.len() {
                table.build_cell(
                    spec,
                    source_block,
                    coordinate,
                    book,
                    acts,
                    set.index(at),
                    ledger,
                );
            }
            set.clear();
            return;
        }
        set.clear();
    } else if addressed < table.code_space() {
        for j in 0..cols {
            if collapsed.is_some_and(|first| first[j] != j) {
                continue;
            }
            let code = codes[(col0 + j) * codes_per_row + p];
            table.build_cell(
                spec,
                source_block,
                coordinate,
                book,
                acts,
                C::index_of(code),
                ledger,
            );
        }
        return;
    }
    for index in 0..table.code_space() {
        table.build_cell(spec, source_block, coordinate, book, acts, index, ledger);
    }
}

/// One row tile: build the stack, reduce every distinct column into a lane, and
/// encode once.
///
/// `rows` is a runtime value now, and that is a simplification rather than a
/// concession. It was a const generic so the column accumulation would sit in
/// registers instead of behind a bounds check --- which mattered while the loop
/// was written here, in a crate compiled at the target's baseline. The loop is a
/// [`TableSpec`] sequence now, so the register count is the sequence's own
/// compile-time constant and the tile height is just a number.
///
/// # Where the exact accumulator is
///
/// Once, at the end. `lane` carries the whole reduction for every column of the
/// block, and it is placed into `exact` only when the run reaches the lane's
/// capacity --- 133144 products at `(i8, 128)`, so for every `k` under that,
/// exactly once per output element. That is "encode once" applied inside the
/// traversal, and measured it was worth 1.94x.
#[allow(clippy::too_many_arguments)]
fn row_tile<E, Bd, C, O, Ep, L, Lg>(
    triple: &mut TabulatedTriple<'_, '_, '_, E, Bd, C, O>,
    epilogue: &Ep,
    options: GemmOptions,
    exact: &mut [AccOf<E>],
    lane: &mut [L],
    table: &mut Table<'_, L>,
    spec: &TableSpec<E, L>,
    spec_of: SpecOf<E, L>,
    book: &[E],
    acts: &mut [E],
    projected: &[E],
    collapse: Option<ColumnMap<'_>>,
    mut entries: Option<&mut EntrySet<'_>>,
    plan: Plan,
    scale: LaneScale,
    local_envelopes: bool,
    rows: usize,
    group: usize,
    row0: usize,
    ledger: &mut Lg,
) where
    E: Tabulated,
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
    let slab = table.slab();
    let projected_rows = projected.len() / shape.k;
    // Blocks one lane word holds before it must be placed. A question about a
    // register, never a limit on `k`: a deeper reduction takes more runs, and the
    // runs combine exactly (§5.1). The walk's answer for a family whose capacity
    // is a fact of the data, the lane's own at the declared bound otherwise.
    let run_blocks = E::lane_run::<L>(Bd::VALUE, &scale)
        .map(|c| (c / block).max(1))
        .unwrap_or(usize::MAX);
    // The two widths this tile reduces at, resolved once. `wide` is the group
    // the tile height asks for; `single` is the one column an operand's repeats
    // or a shape that does not divide leaves over.
    let mut arg: Sweep<'_, E, Bd, C, L> = Sweep {
        wide: *spec,
        // The same sequence when the group is one, which is every widest tile.
        // Looking it up twice would be asking a question whose answer is in the
        // argument.
        single: if group == 1 {
            *spec
        } else {
            spec_of(options.backend, Bd::VALUE, C::SIGN_BIT_BOOK, rows, 1, block)
        },
        codes,
        stream: C::as_index_stream(codes),
        codes_per_row,
        // Set per column block, from `collapsed` --- the block's own slice of
        // the map --- rather than from the map itself. The sweep *skips* a
        // repeated column and the expansion below *fills* it, and the two have
        // to be reading the same decision: a sweep that skips on the whole map
        // while the expansion fills on `collapsed` leaves every repeat of a
        // narrowed block holding whatever was in the accumulator.
        index: None,
        rows,
        slab,
    };

    let mut col0 = 0usize;
    while col0 < shape.n {
        let cols = plan.cols.min(shape.n - col0);
        let acc = &mut exact[..rows * cols];
        let carried = &mut lane[..rows * cols];
        carried.fill(L::ZERO);
        // `acc` is not zeroed. The first placement *sets* it and every later one
        // combines, so the pass that zeroed it was a whole write of the output
        // tile --- 1 MiB at the shipped tile and column block --- for a value
        // that is overwritten before it is read. Every cell is written: a
        // computed column by `place`, a repeated one by the expansion below.
        let mut placed = false;

        // A repeat is collapsed only against a first occurrence *inside this
        // block*, which is why the map is block-local: a representative in an
        // earlier block no longer has an accumulator --- that block was encoded
        // and `exact` reused before this one began. A block that repeats
        // nothing takes the consecutive loop and pays no indexed load for the
        // answer.
        let collapsed = match collapse {
            Some(map) if map.identity[col0 / plan.cols] == 0 => Some(&map.first[col0..col0 + cols]),
            _ => None,
        };
        // One decision, read in both places. `sweep` consults this to skip a
        // repeat; the expansion below consults it to fill one.
        arg.index = collapsed;
        let addressed = collapsed.map_or(cols, |first| {
            first
                .iter()
                .enumerate()
                .filter(|&(j, &representative)| representative == j)
                .count()
        });

        let tail_pending = if local_envelopes {
            let cap = spec.lane_cap;
            let scalar_wide = spec_of(options.backend, Bd::VALUE, false, rows, group, 1);
            let scalar_single = if group == 1 {
                scalar_wide
            } else {
                spec_of(options.backend, Bd::VALUE, false, rows, 1, 1)
            };
            debug_assert_eq!(scalar_wide.lane_cap, cap);
            let scalar_arg: Sweep<'_, E, Bd, C, L> = Sweep {
                wide: scalar_wide,
                single: scalar_single,
                codes,
                stream: C::as_index_stream(codes),
                codes_per_row,
                index: collapsed,
                rows,
                slab,
            };
            let mut height = 0u128;
            let mut pending = false;

            for p in 0..blocks {
                // First ask whether the codec's original aggregate is a safe
                // source atom. No bound array is retained: if it is unsafe the
                // scalar certificates are replayed, which is the storage-free
                // cost of an arbitrary codec block width.
                let mut block_bound = 0u128;
                for t in 0..block {
                    let bound = scalar_envelope(
                        triple, row0, rows, col0, cols, p, t, collapsed, scale, cap, ledger,
                    );
                    if bound > cap || block_bound > cap - bound {
                        block_bound = cap + 1;
                        break;
                    }
                    block_bound += bound;
                }

                if block_bound <= cap {
                    if pending && block_bound > cap - height {
                        place(carried, acc, placed, scale.exponent());
                        placed = true;
                        height = 0;
                    }
                    let base = p * block;
                    for t in 0..block {
                        for i in 0..rows {
                            acts[packed_slot(t, i, rows, spec.k_group)] =
                                if row0 + i < projected_rows {
                                    projected[(row0 + i) * shape.k + base + t]
                                } else {
                                    E::prescale(triple.a.at(row0 + i, base + t).get(), scale.base_a)
                                };
                        }
                    }
                    build_source_block::<E, Bd, C, L, Lg>(
                        table,
                        spec,
                        block,
                        book,
                        &acts[..block * rows],
                        0,
                        p,
                        col0,
                        cols,
                        codes_per_row,
                        codes,
                        collapsed,
                        addressed,
                        entries.as_deref_mut(),
                        ledger,
                    );
                    let computed = sweep_group(
                        group,
                        &arg,
                        &table.stack()[..slab],
                        carried,
                        col0,
                        cols,
                        p,
                        1,
                    );
                    let gathered = count_product2(computed, rows);
                    ledger.read(gathered);
                    ledger.added(gathered);
                    height += block_bound;
                    pending = true;
                    continue;
                }

                for t in 0..block {
                    let bound = scalar_envelope(
                        triple, row0, rows, col0, cols, p, t, collapsed, scale, cap, ledger,
                    );
                    let singleton = bound > cap;
                    if pending && (singleton || bound > cap - height) {
                        place(carried, acc, placed, scale.exponent());
                        placed = true;
                        height = 0;
                    }

                    let source = p * block + t;
                    for i in 0..rows {
                        acts[packed_slot(0, i, rows, scalar_wide.k_group)] =
                            if row0 + i < projected_rows {
                                projected[(row0 + i) * shape.k + source]
                            } else {
                                E::prescale(triple.a.at(row0 + i, source).get(), scale.base_a)
                            };
                    }
                    let scalar_cells = rows * scalar_wide.k_group;
                    build_source_scalar::<E, Bd, C, L, Lg>(
                        table,
                        &scalar_wide,
                        block,
                        t,
                        book,
                        &acts[..scalar_cells],
                        p,
                        col0,
                        cols,
                        codes_per_row,
                        codes,
                        collapsed,
                        addressed,
                        entries.as_deref_mut(),
                        ledger,
                    );
                    let computed = sweep_group(
                        group,
                        &scalar_arg,
                        &table.stack()[..slab],
                        carried,
                        col0,
                        cols,
                        p,
                        1,
                    );
                    let gathered = count_product2(computed, rows);
                    ledger.read(gathered);
                    ledger.added(gathered);

                    if singleton {
                        // Boundary/tag words are semantic singletons. Combining
                        // one with any finite residue would let the sticky tag
                        // erase that residue inside the compact lane, so place
                        // it immediately on both sides of the source boundary.
                        place(carried, acc, placed, scale.exponent());
                        placed = true;
                        pending = false;
                        height = 0;
                    } else {
                        height += bound;
                        pending = true;
                    }
                }
            }
            pending
        } else {
            let mut p0 = 0usize;
            let mut in_run = 0usize;
            while p0 < blocks {
                let depth = plan.depth.min(blocks - p0);
                // Ask about the chunk that will actually run. Pricing a full
                // `plan.depth` here split a final short chunk out of a run it fit,
                // and placing after the final chunk then placed a cleared zero word
                // a second time. The subtraction form is total at `usize::MAX`.
                if run_requires_place(in_run, depth, run_blocks) {
                    place(carried, acc, placed, scale.exponent());
                    placed = true;
                    in_run = 0;
                }
                for slot in 0..depth {
                    // The activation tile for this block, packed once into the layout
                    // the sequence declared: `k`-major in groups of `k_group`,
                    // lane-major within a group. That is [`packed_slot`], the same
                    // function the dense packer uses, so a sequence that folds a pair
                    // of block steps into one instruction finds the pair adjacent and
                    // needs no shuffle and no tail.
                    let base = (p0 + slot) * block;
                    for t in 0..block {
                        for i in 0..rows {
                            // The integer families retain the source value at
                            // their zero base. The float symbol family relabels
                            // this same cell with its contextual q grade; the
                            // paired lane contracts it by lookup and addition.
                            acts[packed_slot(t, i, rows, spec.k_group)] =
                                if row0 + i < projected_rows {
                                    projected[(row0 + i) * shape.k + base + t]
                                } else {
                                    E::prescale(triple.a.at(row0 + i, base + t).get(), scale.base_a)
                                };
                        }
                    }
                    build_source_block::<E, Bd, C, L, Lg>(
                        table,
                        spec,
                        block,
                        book,
                        &acts[..block * rows],
                        slot,
                        p0 + slot,
                        col0,
                        cols,
                        codes_per_row,
                        codes,
                        collapsed,
                        addressed,
                        entries.as_deref_mut(),
                        ledger,
                    );
                }

                // No multiply below this line, and no exact accumulator either: the
                // whole chunk reduces into the lane words the previous chunk left,
                // and the placement happens once for the whole run.
                let stack = &table.stack()[..depth * slab];
                let computed = sweep_group(group, &arg, stack, carried, col0, cols, p0, depth);
                let gathered = count_product3(computed, depth, rows);
                ledger.read(gathered);
                ledger.added(gathered);

                p0 += depth;
                in_run += depth;
            }
            true
        };
        if tail_pending {
            place(carried, acc, placed, scale.exponent());
        }

        // The expansion: every repeated column takes the accumulation of the one
        // it repeats. Ascending, and a repeat always names an earlier column, so
        // the source is final by the time it is read.
        if let Some(of) = collapsed {
            for (j, &src) in of[..cols].iter().enumerate() {
                if src != j {
                    acc.copy_within(src * rows..src * rows + rows, j * rows);
                }
            }
        }

        // The single encode step, exactly once per output element.
        for i in 0..rows {
            for j in 0..cols {
                let (r, c) = (row0 + i, col0 + j);
                let prior = if reads_c {
                    Some(*triple.c.at(r, c))
                } else {
                    None
                };
                *triple.c.at_mut(r, c) = epilogue.finish(acc[j * rows + i], prior, options.encode);
            }
        }
        col0 += cols;
    }
}

/// What one column sweep needs and does not change while it runs.
struct Sweep<'c, E: Element, Bd: Bound, C: Enumerable<E, Bd>, L> {
    wide: TableSpec<E, L>,
    single: TableSpec<E, L>,
    codes: &'c [C::Code],
    /// The operand's own memory, when the codec says it already addresses the
    /// enumeration. Then there is no index stream to build --- the same rule
    /// [`uor_matmul_core::MatView::row_block`] follows on the dense side. The
    /// variant names the code word's width: the gather is monomorphic, and the
    /// dispatch on this enum is the only one the width costs.
    stream: Option<uor_matmul_codec::IndexStream<'c>>,
    codes_per_row: usize,
    /// The block's first-occurrence map, block-relative like the lane words it
    /// indexes into, or `None` when the block repeats nothing.
    index: Option<&'c [usize]>,
    rows: usize,
    slab: usize,
}

/// One chunk of the reduction, over every distinct column of the block.
///
/// `G` is the column group and it is a *compile-time* constant, which is the
/// whole reason this is a function. With `G` runtime, the offset run's stride and
/// the code stream's column step are indexed loads rather than folded constants,
/// and measured that cost 2.4x at the widest tile --- on shapes where `G` is one
/// and the grouping does nothing at all. A knob has to disappear when it is not
/// being used.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn sweep<const G: usize, E, Bd, C, L>(
    arg: &Sweep<'_, E, Bd, C, L>,
    stack: &[L],
    carried: &mut [L],
    col0: usize,
    cols: usize,
    p0: usize,
    depth: usize,
) -> usize
where
    E: Element,
    Bd: Bound,
    C: Enumerable<E, Bd>,
    L: Lane<E>,
{
    let (rows, cpr) = (arg.rows, arg.codes_per_row);
    let mut off = [0u32; GATHER_SLOTS * MAX_GROUP];
    let mut computed = 0usize;
    let mut j = 0usize;
    while j < cols {
        // A repeat is not computed at all; it is copied once, after the whole
        // reduction, from the column it repeats. A group's lanes have to be
        // adjacent in `carried`, so a group is a run of consecutive columns and
        // a skipped one ends the run.
        let present = |u: usize| match arg.index {
            Some(of) => of[j + u] == j + u,
            None => true,
        };
        if !present(0) {
            j += 1;
            continue;
        }
        let whole = G > 1 && j + G <= cols && (1..G).all(present);
        let (run, one) = if whole {
            (G, &arg.wide)
        } else {
            (1, &arg.single)
        };
        let base = (col0 + j) * cpr + p0;
        let lane = &mut carried[j * rows..(j + run) * rows];
        // One branch per group of columns, covering `depth * run * rows` adds,
        // and it is the operand's layout that decides it --- not the data, and
        // not a heuristic. The code width is the same kind of branch: one
        // dispatch per group, monomorphic gathers below it. Both arms compute
        // the same lane words, which is what `CB-08` asserts.
        match arg.stream {
            Some(uor_matmul_codec::IndexStream::U16(stream)) => one.gather_codes(
                depth,
                arg.slab as u32,
                stack,
                &stream[base..base + (run - 1) * cpr + depth],
                cpr,
                lane,
            ),
            Some(uor_matmul_codec::IndexStream::U8(stream)) => one.gather_codes_u8(
                depth,
                arg.slab as u32,
                stack,
                &stream[base..base + (run - 1) * cpr + depth],
                cpr,
                lane,
            ),
            None => {
                // In windows of the run buffer, so its size bounds a stack
                // frame and never a reduction. The gather reads and writes the
                // lane, so a window is the same accumulation continued.
                let mut w = 0usize;
                while w < depth {
                    let take = GATHER_SLOTS.min(depth - w);
                    for slot in 0..take {
                        for u in 0..run {
                            off[slot * run + u] =
                                (C::index_of(arg.codes[base + u * cpr + w + slot]) * rows) as u32;
                        }
                    }
                    one.gather(
                        take,
                        arg.slab as u32,
                        &stack[w * arg.slab..(w + take) * arg.slab],
                        &off[..take * run],
                        lane,
                    );
                    w += take;
                }
            }
        }
        computed += run;
        j += run;
    }
    computed
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn sweep_group<E, Bd, C, L>(
    group: usize,
    arg: &Sweep<'_, E, Bd, C, L>,
    stack: &[L],
    carried: &mut [L],
    col0: usize,
    cols: usize,
    p0: usize,
    depth: usize,
) -> usize
where
    E: Element,
    Bd: Bound,
    C: Enumerable<E, Bd>,
    L: Lane<E>,
{
    match group {
        16 => sweep::<16, E, Bd, C, L>(arg, stack, carried, col0, cols, p0, depth),
        8 => sweep::<8, E, Bd, C, L>(arg, stack, carried, col0, cols, p0, depth),
        4 => sweep::<4, E, Bd, C, L>(arg, stack, carried, col0, cols, p0, depth),
        2 => sweep::<2, E, Bd, C, L>(arg, stack, carried, col0, cols, p0, depth),
        _ => sweep::<1, E, Bd, C, L>(arg, stack, carried, col0, cols, p0, depth),
    }
}

/// Place a completed run of lane words into the exact accumulator and clear it.
///
/// The only place the exact accumulator appears in the reduction, and for a
/// narrow lane that holds the whole depth it runs once per output element.
/// `exponent` is the scale the run was built at: zero for every lane whose
/// products are the elements' own, and the walk's `base_a + base_b` for the
/// float symbol lane, whose placement is the accumulator's `add_scaled`.
fn place<E: Element, L: Lane<E>>(lane: &mut [L], acc: &mut [E::Acc], onto: bool, exponent: i32) {
    for (cell, word) in acc.iter_mut().zip(lane.iter_mut()) {
        // The first run sets, the rest combine. Placing onto a zero the caller
        // wrote is the same value and one more pass over the output tile.
        let prior = if onto {
            *cell
        } else {
            <E::Acc as Accumulator>::ZERO
        };
        *cell = word.place_scaled(prior, exponent);
        *word = L::ZERO;
    }
}

/// Must the carried run be placed before the next real chunk is added?
///
/// The next chunk is the remaining-depth-clamped chunk, not the plan's nominal
/// depth. Written with subtraction so the capacity question is total at the
/// address-space boundary.
const fn run_requires_place(in_run: usize, next_depth: usize, capacity: usize) -> bool {
    in_run != 0 && (in_run > capacity || next_depth > capacity - in_run)
}

#[allow(clippy::too_many_arguments)]
fn tabulate<E, Bd, C, O, Ep, Lg, L>(
    triple: &mut TabulatedTriple<'_, '_, '_, E, Bd, C, O>,
    epilogue: &Ep,
    options: GemmOptions,
    exact: &mut [AccOf<E>],
    lanes: &mut [L],
    panel: &mut [Alphabet<E, Bd>],
    collapse: Option<ColumnMap<'_>>,
    mut entries: Option<EntrySet<'_>>,
    plan: Plan,
    scale: LaneScale,
    spec_of: SpecOf<E, L>,
    projection_decodes: bool,
    local_envelopes: bool,
    ledger: &mut Lg,
) where
    E: Tabulated,
    Bd: Bound,
    C: Enumerable<E, Bd>,
    O: Element + EncodeFrom<AccOf<E>>,
    Ep: Epilogue<E, O>,
    Lg: Ledger,
    L: Lane<E>,
{
    let shape = triple.shape();
    let space = C::CODE_SPACE;
    let block = <C as uor_matmul_codec::Codec<E, Bd>>::MAX_BLOCK;
    let (columns, stack) = lanes.split_at_mut(plan.rows * plan.cols);

    // The fixed panel is exactly the public query's decoded book plus widest
    // activation tile. Any caller-offered tail is a zero-copy cache of complete
    // projected activation rows. Partial rows are not admitted: deriving their
    // shape would require a side index, while a complete row is addressed by
    // the matrix's own `(row, p)` coordinates.
    let book_cells = space * block;
    let acts_cells = ROW_TILES[0] * block;
    let (book_panel, rest) = panel.split_at_mut(book_cells);
    let (acts_panel, cache_offer) = rest.split_at_mut(acts_cells);
    let cache_rows = if shape.n > plan.cols {
        (cache_offer.len() / shape.k).min(shape.m)
    } else {
        0
    };
    // `cache_rows <= cache_offer.len() / shape.k`, so this product is both
    // addressable and contained in the offered tail.
    let cache_cells = cache_rows * shape.k;
    let projected_panel = &mut cache_offer[..cache_cells];
    let projected: &mut [E] = bytemuck::TransparentWrapper::peel_slice_mut(projected_panel);
    for i in 0..cache_rows {
        for p in 0..shape.k {
            projected[i * shape.k + p] = E::prescale(triple.a.at(i, p).get(), scale.base_a);
        }
    }

    if projection_decodes {
        // Cached rows are projected once for the call. Every uncached row is
        // projected once per column block because its fixed activation tile is
        // overwritten when that block completes. This is the exact protocol
        // presentation count, including an exact-sized offer's empty cache.
        let column_blocks = shape.n.div_ceil(plan.cols);
        ledger.decoded(count_sum2(
            count_product2(cache_rows, shape.k),
            count_product3(shape.m - cache_rows, shape.k, column_blocks),
        ));
    }

    let book: &[E] = bytemuck::TransparentWrapper::peel_slice(book_panel);
    let acts: &mut [E] = bytemuck::TransparentWrapper::peel_slice_mut(acts_panel);

    let mut row0 = 0usize;
    let mut zeroed_rows = None;
    while row0 < shape.m {
        // The widest tile the plan and the remaining rows admit. A shape that
        // does not divide walks down the list; it does not take a different path,
        // and `CD-13` asserts the bytes at every `m`.
        let rows = ROW_TILES
            .into_iter()
            .find(|&r| r <= plan.rows && r <= shape.m - row0)
            .unwrap_or(1);
        // The sequence for *this* tile height. A narrower tile is a narrower
        // register file, not a different function, and the reference carries
        // every height no vector sequence has (R13). The codec's sign-book
        // declaration goes with the call: at bound 1 it is what admits the
        // Gray-walk build, and nothing else does (`Ternary` declares the
        // bound and not the book).
        let group = column_group(rows);
        let spec = spec_of(
            options.backend,
            Bd::VALUE,
            C::SIGN_BIT_BOOK,
            rows,
            group,
            block,
        );
        let table = if zeroed_rows == Some(rows) {
            Table::reuse_zeroed(stack, space, rows, plan.depth)
        } else {
            let table = Table::new(stack, space, rows, plan.depth);
            if table.is_some() {
                zeroed_rows = Some(rows);
            }
            table
        };
        let Some(mut table) = table else {
            // `Plan::choose` sized the offer for the widest tile it admits and
            // `rows` is never wider, so this cannot be reached. It is written as
            // the streaming traversal rather than as an assertion because an
            // unreachable branch that could produce no output at all is worse
            // than one that produces the right output slowly (C6, R14).
            stream(triple, epilogue, options, panel, ledger);
            return;
        };
        row_tile::<E, Bd, C, O, Ep, L, Lg>(
            triple,
            epilogue,
            options,
            exact,
            columns,
            &mut table,
            &spec,
            spec_of,
            book,
            acts,
            projected,
            collapse,
            entries.as_mut(),
            plan,
            scale,
            local_envelopes,
            rows,
            group,
            row0,
            ledger,
        );
        row0 += rows;
    }
}

/// What to do when the table is not the answer.
///
/// The decoded dense factorization runs when the offer holds it. Otherwise, if
/// a family's first real empty-rest dense partial accepts, that partial starts
/// its persistent stream; a declining family uses its ordinary stream. These
/// are factorizations of one identity, ordered by capability and measured cost
/// --- not by quality, because they produce the same bytes and `CD-13` says so.
fn decline<E, Bd, C, O, Ep, Lg>(
    triple: &mut TabulatedTriple<'_, '_, '_, E, Bd, C, O>,
    epilogue: &Ep,
    options: GemmOptions,
    scratch: &mut Scratch<'_, E, Bd>,
    ledger: &mut Lg,
) where
    E: Tabulated,
    Bd: Bound,
    C: Enumerable<E, Bd>,
    O: Element + EncodeFrom<AccOf<E>>,
    Ep: Epilogue<E, O>,
    Lg: Ledger,
{
    if options.traversal != Traversal::OutputMajor
        && packed_route(triple, epilogue, options, scratch, ledger)
    {
        return;
    }
    let k = triple.shape().k;
    let panel = scratch.take(k);
    if atlas_stream_route(triple, epilogue, options, panel, ledger) {
        return;
    }
    stream(triple, epilogue, options, panel, ledger);
}

/// Capture one exact dense partial without encoding it.
///
/// `dense_gemm` is generic in its output epilogue, so the coded adapter can
/// borrow that factorization without inventing a second float operation or a
/// private replacement for [`Tabulated::StreamLane`]. The sink value is never
/// observed; the complete accumulator is the object this epilogue transfers.
struct DenseCapture<'a, A>(&'a core::cell::Cell<A>);

impl<E: Element, O: Element> Epilogue<E, O> for DenseCapture<'_, AccOf<E>> {
    fn finish(&self, acc: AccOf<E>, _: Option<O>, _: EncodeMode) -> O {
        self.0.set(acc);
        O::ZERO
    }

    fn reads_c(&self) -> bool {
        false
    }
}

/// One decoded `1 x k` by `k x 1` partial through the family's dense engine.
///
/// `None` is the family's value-independent empty-rest decline. The caller
/// asks with the first real partial before touching caller `C`; after one
/// acceptance, a later refusal of a conformant subview violates
/// [`Tabulated::dense_gemm`]'s law and is asserted by the caller rather than
/// silently changing the arithmetic.
fn dense_stream_dot<E, Bd, O, Lg>(
    a: MatView<'_, Alphabet<E, Bd>>,
    b: &[Alphabet<E, Bd>],
    options: GemmOptions,
    ledger: &mut Lg,
) -> Option<AccOf<E>>
where
    E: Tabulated,
    Bd: Bound,
    O: Element + EncodeFrom<AccOf<E>>,
    Lg: Ledger,
{
    if b.is_empty() {
        return Some(<AccOf<E> as Accumulator>::ZERO);
    }
    let right = MatView::row_major(b, b.len(), 1).expect("one column has the declared extent");
    let mut sink = [O::ZERO];
    let output = MatViewMut::row_major(&mut sink, 1, 1).expect("one output cell exists");
    let captured = core::cell::Cell::new(<AccOf<E> as Accumulator>::ZERO);
    ledger.kernelled();
    let ran = E::dense_gemm(a, right, output, &DenseCapture(&captured), options, &mut []);
    ran.then(|| captured.into_inner())
}

/// Stream a coded float dot through bounded decoded source pages.
///
/// A nonempty caller offer is the page, at every size; only no offer uses the
/// model's measured `blocking::KC` frame. Neither is an admission limit: an
/// arbitrary reduction repeats the page and combines its complete partials.
/// The first partial transfers directly, so every dot of at most one page
/// retains one Complete accumulator with no join. A caller offer holding a
/// whole coded row removes even the page boundary and shares that decode across
/// all rows of `A`.
fn dense_stream<E, Bd, C, O, Ep, Lg>(
    triple: &mut TabulatedTriple<'_, '_, '_, E, Bd, C, O>,
    epilogue: &Ep,
    options: GemmOptions,
    panel: &mut [Alphabet<E, Bd>],
    ledger: &mut Lg,
) -> bool
where
    E: Tabulated,
    Bd: Bound,
    C: Enumerable<E, Bd>,
    O: Element + EncodeFrom<AccOf<E>>,
    Ep: Epilogue<E, O>,
    Lg: Ledger,
{
    if !panel.is_empty() {
        return dense_stream_with_page(triple, epilogue, options, panel, ledger);
    }

    // An empty offer is still bounded execution. This model-derived frame is
    // reused for every page; it is a storage factorization, never a limit on
    // the reduction's depth.
    let mut page = [Alphabet::<E, Bd>::ZERO; blocking::KC];
    dense_stream_with_page(triple, epilogue, options, &mut page, ledger)
}

/// Run the persistent dense stream in the page storage selected at the offer
/// boundary.
///
/// Every nonempty caller offer is used directly, including a partial row. The
/// private bounded frame above exists only for an empty offer, so a useful
/// caller byte is never ignored or copied into another page first.
fn dense_stream_with_page<E, Bd, C, O, Ep, Lg>(
    triple: &mut TabulatedTriple<'_, '_, '_, E, Bd, C, O>,
    epilogue: &Ep,
    options: GemmOptions,
    page: &mut [Alphabet<E, Bd>],
    ledger: &mut Lg,
) -> bool
where
    E: Tabulated,
    Bd: Bound,
    C: Enumerable<E, Bd>,
    O: Element + EncodeFrom<AccOf<E>>,
    Ep: Epilogue<E, O>,
    Lg: Ledger,
{
    let shape = triple.shape();
    let reads_c = epilogue.reads_c();
    debug_assert!(shape.k == 0 || !page.is_empty());
    let borrowed = page.len() >= shape.k;
    let mut accepted = false;

    for j in 0..shape.n {
        if borrowed {
            triple.w.decode_row_into(j, &mut page[..shape.k]);
            ledger.decoded(count_factor(shape.k));
            let right = MatView::row_major(&page[..shape.k], shape.k, 1)
                .expect("one decoded column has the declared extent");
            let mut output = [O::ZERO; ROW_TILES[0]];
            let mut row0 = 0usize;
            while row0 < shape.m {
                let rows = (shape.m - row0).min(ROW_TILES[0]);
                let left = triple
                    .a
                    .subview(row0, 0, rows, shape.k)
                    .expect("the row tile is inside the conformant activation view");
                if reads_c {
                    for (i, cell) in output[..rows].iter_mut().enumerate() {
                        *cell = *triple.c.at(row0 + i, j);
                    }
                }
                let sink = MatViewMut::row_major(&mut output[..rows], rows, 1)
                    .expect("the bounded row tile is one output column");
                ledger.kernelled();
                let ran = E::dense_gemm(left, right, sink, epilogue, options, &mut []);
                match ran {
                    true => accepted = true,
                    false if accepted => {
                        panic!("Tabulated::dense_gemm changed its empty-rest acceptance")
                    }
                    false => return false,
                }
                for (i, &cell) in output[..rows].iter().enumerate() {
                    *triple.c.at_mut(row0 + i, j) = cell;
                }
                row0 += rows;
            }
            continue;
        }

        for i in 0..shape.m {
            let acc = {
                let mut acc = <AccOf<E> as Accumulator>::ZERO;
                let mut first = true;
                let mut start = 0usize;
                while start < shape.k {
                    let depth = (shape.k - start).min(page.len());
                    for (p, cell) in page[..depth].iter_mut().enumerate() {
                        *cell = triple.w.at(j, start + p);
                    }
                    ledger.decoded(count_factor(depth));
                    let left = triple
                        .a
                        .subview(i, start, 1, depth)
                        .expect("the page is inside the conformant activation view");
                    let partial = match dense_stream_dot::<E, Bd, O, Lg>(
                        left,
                        &page[..depth],
                        options,
                        ledger,
                    ) {
                        Some(partial) => {
                            accepted = true;
                            partial
                        }
                        None if accepted => {
                            panic!("Tabulated::dense_gemm changed its empty-rest acceptance")
                        }
                        None => return false,
                    };
                    if first {
                        acc = partial;
                        first = false;
                    } else {
                        ledger.added(1);
                        acc = acc.combine(partial);
                    }
                    start += depth;
                }
                acc
            };
            let prior = if reads_c {
                Some(*triple.c.at(i, j))
            } else {
                None
            };
            *triple.c.at_mut(i, j) = epilogue.finish(acc, prior, options.encode);
        }
    }
    true
}

/// Enter the bounded dense stream when the family's first real partial accepts.
///
/// The first decoded source chunk is useful work, not a capability object: its
/// complete partial is retained and becomes the beginning of output `(0, 0)`.
/// Caller `C` remains untouched until that acceptance. A zero-depth product
/// has no real partial with which to establish acceptance, so it remains in
/// the public stream lane and issues no product. In both cases the public
/// stream-lane identity remains unchanged and the census records no scalar
/// multiply.
fn atlas_stream_route<E, Bd, C, O, Ep, Lg>(
    triple: &mut TabulatedTriple<'_, '_, '_, E, Bd, C, O>,
    epilogue: &Ep,
    options: GemmOptions,
    panel: &mut [Alphabet<E, Bd>],
    ledger: &mut Lg,
) -> bool
where
    E: Tabulated,
    Bd: Bound,
    C: Enumerable<E, Bd>,
    O: Element + EncodeFrom<AccOf<E>>,
    Ep: Epilogue<E, O>,
    Lg: Ledger,
{
    // `Whole<E>` is the float alphabet's declaration that there is no finite
    // magnitude bound. Every shipped integer alphabet is finite, so its
    // monomorphization returns before decoding or calling the dense engine. An
    // exotic integer bound spelling `MAX` still presents the first real
    // partial and can decline before caller output; this is an alphabet fact,
    // never a branch on an operand or an element identity.
    if Bd::VALUE != u128::MAX || triple.shape().k == 0 {
        return false;
    }

    dense_stream(triple, epilogue, options, panel, ledger)
}

/// The third factorization: decode the whole operand once and hand it to the tile
/// kernels.
///
/// `W` is `n x k` and the kernels want `k x n`, so the decoded buffer is read
/// through swapped strides --- transposition is a stride, and this is the same
/// `(A * B)^T` move [`crate::collapse`] uses on the other operand.
///
/// Available only when the caller's panel offer holds the whole decoded operand,
/// which is the caller declaring it can afford the dense weights. That is the
/// trade the codec exists to avoid, so it is never taken behind the caller's back:
/// no offer, no route. And it is not a fallback --- the tile kernels compute the
/// same exact sum, and `CD-13` asserts the bytes against the table and the stream
/// alike.
///
/// `false` when the offer is short, in which case the caller streams.
fn packed_route<E, Bd, C, O, Ep, Lg>(
    triple: &mut TabulatedTriple<'_, '_, '_, E, Bd, C, O>,
    epilogue: &Ep,
    options: GemmOptions,
    scratch: &mut Scratch<'_, E, Bd>,
    ledger: &mut Lg,
) -> bool
where
    E: Tabulated,
    Bd: Bound,
    C: Enumerable<E, Bd>,
    O: Element + EncodeFrom<AccOf<E>>,
    Ep: Epilogue<E, O>,
    Lg: Ledger,
{
    let shape = triple.shape();
    let Some(want) = shape.n.checked_mul(shape.k) else {
        return false;
    };
    let (panel, rest) = scratch.split_panel(want);
    // The decoded operand, and enough left over for the kernels' own panels. A
    // caller who offers only the first gets the streaming traversal instead: a
    // packed traversal with nowhere to pack is not the fast factorization this
    // route exists to reach, and measured it was twenty times slower than the
    // kernels it was standing in for.
    if panel.len() < want || rest.len() < shape.k {
        return false;
    }
    for j in 0..shape.n {
        triple
            .w
            .decode_row_into(j, &mut panel[j * shape.k..(j + 1) * shape.k]);
    }
    ledger.decoded(count_factor(want));

    // `panel` holds `W` row-major, which is `W^T` read with the strides swapped.
    let Some(b) = MatView::new(
        panel,
        shape.k,
        shape.n,
        Strides {
            rs: 1,
            cs: shape.k as isize,
        },
    ) else {
        return false;
    };
    // The family's own dense driver over the decoded operand: the tile kernels
    // for an integer, the exact float traversal for a float.
    ledger.kernelled();
    if !E::dense_gemm(triple.a, b, triple.c.reborrow(), epilogue, options, rest) {
        return false;
    }
    true
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
    E: Tabulated,
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
    // it --- which decodes each weight once instead of once per row of `A`, and
    // leaves two contiguous runs for the lane dot product below.
    let borrowed = panel.len() >= shape.k;
    // The lane that holds a *raw* product, at a block of one. This is the
    // ordinary decline for a family whose first empty-rest dense partial
    // answered false. Shipped integers use the same narrow/exact lane their
    // alphabet declares; shipped floats have already entered `dense_stream`
    // above.
    let run = <E::StreamLane as Lane<E>>::capacity(Bd::VALUE).unwrap_or(usize::MAX);
    for j in 0..shape.n {
        if borrowed {
            triple.w.decode_row_into(j, panel);
            ledger.decoded(count_factor(shape.k));
        }
        for i in 0..shape.m {
            let row = if borrowed {
                triple.a.row_block(i, 0, 1, shape.k)
            } else {
                None
            };
            let acc = match row {
                // Both operands are runs: the whole reduction is a loop over two
                // contiguous slices in the lane that holds raw products.
                Some(a) => dot_lane::<E, Bd, E::StreamLane>(a, &panel[..shape.k], run),
                // `A`'s row is not a run, or nothing was offered to decode into.
                // The same lane accumulation, walked: this is what makes the
                // traversal runnable on a target whose RAM cannot hold one row
                // (S13), without changing its product operation.
                None => {
                    if borrowed {
                        dot_walk::<E, E::StreamLane, _>(shape.k, run, |p| {
                            (triple.a.at(i, p).get(), panel[p].get())
                        })
                    } else {
                        let acc = dot_walk::<E, E::StreamLane, _>(shape.k, run, |p| {
                            (triple.a.at(i, p).get(), triple.w.at(j, p).get())
                        });
                        ledger.decoded(count_factor(shape.k));
                        acc
                    }
                }
            };
            ledger.multiplied(count_factor(shape.k));
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
    use crate::coded::CodedTriple;
    use crate::collapse::suggested_collapse_index;
    use crate::driver::gemm;
    use crate::epilogue::{Linear, MaxPlus};
    use crate::partition::Partition;
    use std::format;
    use std::string::String;
    use std::vec;
    use std::vec::Vec;
    use uor_matmul_codec::{
        canonicalize, e8_codec, e8_codec_u8, e8_table, Arena, Book, Codec, Grid, Manifest, Packed,
        Sign, SymbolCode, Ternary, TierId,
    };
    use uor_matmul_core::{
        as_alphabet, as_alphabet_full, as_alphabet_tropical, as_alphabet_whole, Bnd, EncodeMode,
        FloatElement, Full, IntegerElement, Triple, Trop, Whole,
    };

    use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    type A8 = Alphabet<i8, Full<i8>>;

    /// A downstream family whose empty-rest dense operation can accept or
    /// decline. It witnesses that the Atlas route retains an accepted first
    /// partial and that a decline touches no caller output before the ordinary
    /// stream.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    struct DenseDecline(i8);

    /// A truthful downstream lane that cannot hold even one product at its
    /// declared alphabet. Its arithmetic methods panic so the totality test
    /// proves the traversal never invokes a zero-capacity lane by accident.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    struct ZeroLane;

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    struct MaxBound;

    /// A downstream enumerable f64 codec whose code names two reduction
    /// elements. It prevents the in-tree Arena's block of one from becoming a
    /// categorical assumption in the f64 family.
    #[derive(Clone, Copy, Debug)]
    struct PairF64;

    /// A binary32 codec block wider than the q-carrier's one-product boundary.
    /// Its two positions independently span seven grades, so the exact CD-32
    /// capacity is one and executing it requires scalar fracture rather than a
    /// whole-code decline.
    #[derive(Clone, Copy, Debug)]
    struct FractureF32;

    /// The same scalar-fracture law at independently chosen, non-power block
    /// and code-space extents. Const parameters keep the test on the generic
    /// codec surface instead of adding another hand-specialized fixture.
    #[derive(Clone, Copy, Debug)]
    struct ParametricFractureF32<const B: usize, const D: usize>;

    /// A downstream enumeration whose canonical coordinate is deliberately not
    /// its stored code. It witnesses that the private addressed view remains a
    /// relabeling through `code_at/index_of`, not an identity-code assumption.
    #[derive(Clone, Copy, Debug)]
    struct PermutedF32<'a>(&'a [Alphabet<f32, Whole<f32>>; 4]);

    impl Codec<f32, Whole<f32>> for PermutedF32<'_> {
        type Code = u8;
        const MAX_BLOCK: usize = 1;
        const TIER: TierId = TierId::Book;

        fn decode_element(&self, code: Self::Code, _: usize) -> Alphabet<f32, Whole<f32>> {
            self.0[usize::from(code) % 4]
        }
    }

    impl Enumerable<f32, Whole<f32>> for PermutedF32<'_> {
        const CODE_SPACE: usize = 4;

        fn code_at(index: usize) -> Self::Code {
            [2, 0, 3, 1][index % 4]
        }

        fn index_of(code: Self::Code) -> usize {
            [1, 3, 0, 2][usize::from(code) % 4]
        }
    }

    impl Codec<f64, Whole<f64>> for PairF64 {
        type Code = u8;
        const MAX_BLOCK: usize = 2;
        const TIER: TierId = TierId::Book;

        fn decode_element(&self, code: Self::Code, i: usize) -> Alphabet<f64, Whole<f64>> {
            let pair = if code % 2 == 0 {
                [0.5, -1.0]
            } else {
                [1.5, 0.25]
            };
            bytemuck::TransparentWrapper::wrap(pair[i % 2])
        }
    }

    impl Enumerable<f64, Whole<f64>> for PairF64 {
        const CODE_SPACE: usize = 2;

        fn code_at(index: usize) -> Self::Code {
            (index % 2) as u8
        }

        fn index_of(code: Self::Code) -> usize {
            usize::from(code % 2)
        }
    }

    impl Codec<f32, Whole<f32>> for FractureF32 {
        type Code = u8;
        const MAX_BLOCK: usize = 2;
        const TIER: TierId = TierId::Book;

        fn decode_element(&self, code: Self::Code, i: usize) -> Alphabet<f32, Whole<f32>> {
            let low = f32::from_bits(0x3fff_ffff);
            let high = f32::from_bits(0x437f_ffff);
            let pair = if code % 2 == 0 {
                [high, high]
            } else {
                [low, high]
            };
            bytemuck::TransparentWrapper::wrap(pair[i % Self::MAX_BLOCK])
        }
    }

    impl Enumerable<f32, Whole<f32>> for FractureF32 {
        const CODE_SPACE: usize = 2;

        fn code_at(index: usize) -> Self::Code {
            (index % Self::CODE_SPACE) as u8
        }

        fn index_of(code: Self::Code) -> usize {
            usize::from(code % Self::CODE_SPACE as u8)
        }
    }

    impl<const B: usize, const D: usize> Codec<f32, Whole<f32>> for ParametricFractureF32<B, D> {
        type Code = u8;
        const MAX_BLOCK: usize = B;
        const TIER: TierId = TierId::Book;

        fn decode_element(&self, code: Self::Code, i: usize) -> Alphabet<f32, Whole<f32>> {
            let low = f32::from_bits(0x3fff_ffff);
            let high = f32::from_bits(0x437f_ffff);
            let value = if (usize::from(code) + i).is_multiple_of(2) {
                low
            } else {
                high
            };
            bytemuck::TransparentWrapper::wrap(value)
        }
    }

    impl<const B: usize, const D: usize> Enumerable<f32, Whole<f32>> for ParametricFractureF32<B, D> {
        const CODE_SPACE: usize = D;

        fn code_at(index: usize) -> Self::Code {
            u8::try_from(index % D).expect("the test code spaces fit u8")
        }

        fn index_of(code: Self::Code) -> usize {
            usize::from(code) % D
        }
    }

    impl Bound for MaxBound {
        const VALUE: u128 = u128::MAX;
    }

    static DENSE_CALLS: AtomicUsize = AtomicUsize::new(0);
    static DENSE_ACCEPTS: AtomicBool = AtomicBool::new(false);
    static DENSE_DECLINE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    impl Element for DenseDecline {
        type Acc = i128;
        const BITS: u32 = i8::BITS;
        const ZERO: Self = Self(0);
        const HAS_NARROW: bool = true;
        const HAS_NARROW32: bool = true;

        fn mac(acc: &mut Self::Acc, a: Self, w: Self) {
            *acc += i128::from(a.0) * i128::from(w.0);
        }

        fn mac_narrow(acc: i64, a: Self, w: Self) -> i64 {
            acc + i64::from(a.0) * i64::from(w.0)
        }

        fn mac_narrow32(acc: i32, a: Self, w: Self) -> i32 {
            acc + i32::from(a.0) * i32::from(w.0)
        }

        fn combine_narrow(acc: Self::Acc, narrow: i64) -> Self::Acc {
            acc + i128::from(narrow)
        }
    }

    impl IntegerElement for DenseDecline {
        const FULL: u128 = 1u128 << (i8::BITS - 1);
        const ONE: Self = Self(1);

        fn magnitude(self) -> u128 {
            u128::from(self.0.unsigned_abs())
        }

        fn sub(self, other: Self) -> Self {
            Self(self.0.wrapping_sub(other.0))
        }

        fn add(self, other: Self) -> Self {
            Self(self.0.wrapping_add(other.0))
        }
    }

    impl LaneWord for ZeroLane {
        const ZERO: Self = Self;

        fn add(self, _: Self) -> Self {
            panic!("a zero-capacity lane has no valid addition")
        }
    }

    impl Lane<DenseDecline> for ZeroLane {
        fn capacity(_: u128) -> Option<usize> {
            Some(0)
        }

        fn mac(self, _: DenseDecline, _: DenseDecline) -> Self {
            panic!("a zero-capacity lane has no valid product")
        }

        fn place(self, _: AccOf<DenseDecline>) -> AccOf<DenseDecline> {
            panic!("a zero-capacity lane has no valid placement")
        }
    }

    impl Tabulated for DenseDecline {
        type Lane = i32;
        type ModLane = i32;
        type StreamLane = ZeroLane;
        const LANE_IS_EXACT: bool = false;

        fn modular_table_admitted(_: u32) -> bool {
            false
        }

        fn table_spec(
            _: Backend,
            _: u128,
            _: bool,
            rows: usize,
            group: usize,
            _: usize,
        ) -> TableSpec<Self, Self::Lane> {
            portable_table::<Self, i32>(rows, group)
        }

        fn table_spec_modular(
            backend: Backend,
            bound: u128,
            rows: usize,
            group: usize,
            block: usize,
        ) -> TableSpec<Self, Self::ModLane> {
            Self::table_spec(backend, bound, false, rows, group, block)
        }

        fn lanes<'s>(
            narrow: &'s mut [i64],
            _: &'s mut [AccOf<Self>],
            want: usize,
        ) -> Option<&'s mut [Self::Lane]> {
            bytemuck::cast_slice_mut::<i64, i32>(narrow).get_mut(..want)
        }

        fn lanes_modular<'s>(
            narrow: &'s mut [i64],
            exact: &'s mut [AccOf<Self>],
            want: usize,
        ) -> Option<&'s mut [Self::ModLane]> {
            Self::lanes(narrow, exact, want)
        }

        fn dense_steps(_: Backend, _: u128, _: usize, table: usize) -> Steps {
            Steps {
                table,
                dense: 1,
                dense_rows: 1,
            }
        }

        fn dense_gemm<Bd, O, Ep>(
            a: MatView<'_, Alphabet<Self, Bd>>,
            b: MatView<'_, Alphabet<Self, Bd>>,
            c: MatViewMut<'_, O>,
            epilogue: &Ep,
            options: GemmOptions,
            _: &mut [Alphabet<Self, Bd>],
        ) -> bool
        where
            Bd: Bound,
            O: Element + EncodeFrom<AccOf<Self>>,
            Ep: Epilogue<Self, O>,
        {
            DENSE_CALLS.fetch_add(1, Ordering::Relaxed);
            if !DENSE_ACCEPTS.load(Ordering::Relaxed) {
                return false;
            }
            let Ok(mut dense) = Triple::new(a, b, c) else {
                return false;
            };
            let shape = dense.shape();
            let reads_c = epilogue.reads_c();
            for i in 0..shape.m {
                for j in 0..shape.n {
                    let mut acc = <AccOf<Self> as Accumulator>::ZERO;
                    for p in 0..shape.k {
                        Self::mac(&mut acc, dense.a().at(i, p).get(), dense.b().at(p, j).get());
                    }
                    let prior = if reads_c {
                        Some(*dense.c_mut().at(i, j))
                    } else {
                        None
                    };
                    *dense.c_mut().at_mut(i, j) = epilogue.finish(acc, prior, options.encode);
                }
            }
            true
        }
    }

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

    /// Retained test-only clock/control for the removed FNV/prefix spelling.
    /// Production purity audits mask this module; its only role is to keep the
    /// radix replacement accountable to the exact pre-refactor work.
    fn legacy_column_hash<E, Bd, C>(run: &[C::Code]) -> usize
    where
        E: Element,
        Bd: Bound,
        C: Enumerable<E, Bd>,
    {
        // Retained byte-for-byte as the immutable pre-refactor clock arm. It
        // is a measurement oracle only; production purity masks this module.
        const HASH_PREFIX: usize = 16;
        const SEED: u64 = 0xcbf2_9ce4_8422_2325;
        const PRIME: u64 = 0x0000_0100_0000_01b3;
        let mut h = [
            SEED,
            SEED ^ 1,
            SEED ^ 2,
            SEED ^ 3,
            SEED ^ 4,
            SEED ^ 5,
            SEED ^ 6,
            SEED ^ 7,
        ];
        let seeded = (SEED ^ run.len() as u64).wrapping_mul(PRIME);
        h[0] ^= seeded;
        let run = &run[..run.len().min(HASH_PREFIX)];
        let mut chunks = run.chunks_exact(8);
        for chunk in &mut chunks {
            for (lane, &code) in h.iter_mut().zip(chunk) {
                *lane = (*lane ^ C::index_of(code) as u64).wrapping_mul(PRIME);
            }
        }
        for (i, &code) in chunks.remainder().iter().enumerate() {
            h[i] = (h[i] ^ C::index_of(code) as u64).wrapping_mul(PRIME);
        }
        let mut x = 0u64;
        for (i, lane) in h.into_iter().enumerate() {
            x ^= lane.rotate_left((i * 8) as u32);
        }
        x ^= x >> 30;
        x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x ^= x >> 27;
        (x ^ (x >> 31)) as usize
    }

    fn legacy_distinct_columns<E, Bd, C>(
        codes: &[C::Code],
        codes_per_row: usize,
        n: usize,
        index: &mut [usize],
    ) -> Option<usize>
    where
        E: Element,
        Bd: Bound,
        C: Enumerable<E, Bd>,
    {
        if n == 0 || codes_per_row == 0 || codes.len() < n.checked_mul(codes_per_row)? {
            return None;
        }
        let table = n.checked_mul(2)?.checked_next_power_of_two()?;
        if index.len() < n.checked_add(table.checked_mul(2)?)? {
            return None;
        }
        let (position, rest) = index.split_at_mut(n);
        let (slot, key) = rest.split_at_mut(table);
        slot.fill(0);
        let mask = table - 1;
        let mut distinct = 0usize;
        for (j, position) in position.iter_mut().enumerate() {
            let run = &codes[j * codes_per_row..(j + 1) * codes_per_row];
            let hash = legacy_column_hash::<E, Bd, C>(run);
            let mut probe = hash & mask;
            loop {
                match slot[probe] {
                    0 => {
                        slot[probe] = j + 1;
                        key[probe] = hash;
                        *position = j;
                        distinct += 1;
                        break;
                    }
                    seen => {
                        let seen = seen - 1;
                        let other = &codes[seen * codes_per_row..(seen + 1) * codes_per_row];
                        if key[probe] == hash && columns_equal::<E, Bd, C>(run, other) {
                            *position = seen;
                            break;
                        }
                        probe = (probe + 1) & mask;
                    }
                }
            }
        }
        Some(distinct)
    }

    /// `CT-01`: rounding an enumeration up to a mask slab is total at the
    /// address-space boundary. The largest representable power of two remains
    /// itself; every larger code space has no representable mask slab and
    /// declines before sizing, construction, or planning can overflow.
    #[test]
    fn slab_rounding_is_total_for_every_representable_code_space_ct_01() {
        let largest_power = usize::MAX / 2 + 1;
        assert_eq!(slab_codes(0), 0);
        assert_eq!(slab_codes(largest_power), largest_power);
        assert_eq!(slab_codes(largest_power + 1), 0);
        assert_eq!(slab_codes(usize::MAX), 0);
        assert_eq!(table_words(usize::MAX, 1, 1), usize::MAX);
        assert_eq!(
            suggested_tabulation_index(Shape {
                m: 1,
                k: 1,
                n: usize::MAX,
            }),
            usize::MAX
        );

        let mut no_words: [i32; 0] = [];
        assert!(Table::new(&mut no_words, usize::MAX, 1, 1).is_none());
        assert_eq!(
            tabulation_rows(usize::MAX, blocking::L1_BYTES, core::mem::size_of::<i32>()),
            0
        );
        assert!(Plan::choose(
            usize::MAX,
            Shape { m: 1, k: 1, n: 1 },
            core::mem::size_of::<i32>(),
            usize::MAX,
            usize::MAX,
            1,
            None,
        )
        .is_none());

        // Residency is charged for the padded mask slab, not merely its live
        // entries. Three codes occupy four addressable entries: six bytes is
        // enough for the old unpadded calculation and not for the real table.
        assert!(!tabulation_fits(3, 1, 6, 1));
        assert!(tabulation_fits(3, 1, 8, 1));
        assert_eq!(tabulation_rows(3, 24, 1), 3);
        assert_eq!(tabulation_depth(3, 1, 1, None, 24, 1), 3);
        assert!(!tabulation_fits(largest_power, 2, usize::MAX, 2));

        // A query at the empty tile is total and asks for no gather group.
        assert_eq!(column_group(0), 0);
    }

    /// `CD-13`: public construction retains its zero-padding contract, while
    /// the private same-geometry row-tile reborrow leaves that established
    /// padding resident instead of clearing it again. Live cells deliberately
    /// remain nonzero across the reborrow, so a whole-stack clear cannot pass.
    #[test]
    fn table_padding_is_zeroed_once_and_reused_at_the_same_geometry_cd_13() {
        let (space, rows, depth) = (3usize, 2usize, 2usize);
        let slab = slab_codes(space) * rows;
        let live = space * rows;
        let mut words = vec![-1i32; slab * depth];
        {
            let table = Table::new(&mut words, space, rows, depth).expect("the stack fits");
            for slot in 0..depth {
                assert!(table.stack()[slot * slab..slot * slab + live]
                    .iter()
                    .all(|&word| word == -1));
                assert!(table.stack()[slot * slab + live..(slot + 1) * slab]
                    .iter()
                    .all(|&word| word == 0));
            }
        }
        for slot in 0..depth {
            for (at, word) in words[slot * slab..slot * slab + live]
                .iter_mut()
                .enumerate()
            {
                *word = (slot * live + at + 1) as i32;
            }
        }
        let table = Table::reuse_zeroed(&mut words, space, rows, depth)
            .expect("the same resident geometry reborrows");
        for slot in 0..depth {
            assert!(table.stack()[slot * slab..slot * slab + live]
                .iter()
                .all(|&word| word != 0));
            assert!(table.stack()[slot * slab + live..(slot + 1) * slab]
                .iter()
                .all(|&word| word == 0));
        }
    }

    /// `CD-20`: capacity is tested against the next chunk that really exists.
    /// A three-block chunk followed by a two-block tail is one five-block run;
    /// a second full three-block chunk is not. The boundary arithmetic remains
    /// total at the largest address-sized capacity.
    #[test]
    fn a_short_final_chunk_is_not_split_or_placed_twice_cd_20() {
        assert!(!run_requires_place(0, 3, 5));
        assert!(!run_requires_place(3, 2, 5));
        assert!(run_requires_place(3, 3, 5));
        assert!(!run_requires_place(usize::MAX - 1, 1, usize::MAX));
        assert!(run_requires_place(usize::MAX - 1, 2, usize::MAX));
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
    fn reference<Bd: Bound, C: Enumerable<i8, Bd> + Copy>(
        w: &CodedMatrix<'_, i8, Bd, C>,
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
        // A validating wrap rather than the transmute, so the one helper serves
        // any declared bound: at `Full` the check cannot fail, and at a named
        // bound it re-checks the caller's premise at the boundary.
        let av = MatView::row_major(
            as_alphabet::<i8, Bd>(a).expect("the activations fit the declared bound"),
            m,
            k,
        )
        .unwrap();
        let bv = MatView::row_major(
            as_alphabet::<i8, Bd>(&b).expect("decoded weights are in the alphabet by construction"),
            k,
            n,
        )
        .unwrap();
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
    /// The two offers move *independently*, and they have to: nothing obliges a
    /// caller who halves the accumulators to halve the lane and index buffers
    /// too, and the documented contract on [`suggested_tabulation`] invites
    /// exactly that ("offering less narrows the column block"). A sweep that
    /// scaled them together left the disagreeing case unreachable, and a
    /// narrowed column block that skipped a repeated column without filling it
    /// lived there --- silently, at 896 wrong cells out of 1024.
    #[allow(clippy::too_many_arguments)]
    fn tabulated<Bd: Bound, C: Enumerable<i8, Bd> + Copy>(
        w: &CodedMatrix<'_, i8, Bd, C>,
        a: &[i8],
        m: usize,
        n: usize,
        traversal: Traversal,
        acc_offer: usize,
        aux_offer: usize,
        collapse_offer: usize,
    ) -> (Vec<i32>, Census) {
        let k = w.cols();
        let shape = Shape { m, k, n };
        let block = <C as uor_matmul_codec::Codec<i8, Bd>>::MAX_BLOCK;
        let want_acc = suggested_tabulation::<i8, Bd>(shape, C::CODE_SPACE, block).max(1);
        let want_lanes = suggested_tabulation_lanes::<i8, Bd>(shape, C::CODE_SPACE, block).max(1);
        // `offer` is a numerator over the suggested amount, so one knob sweeps
        // both buffers and the extremes -- nothing, one word, exactly enough --
        // are all reachable.
        let scale = |want: usize, offer: usize| -> usize {
            if offer >= OFFER_STEPS {
                want.saturating_mul(offer - OFFER_STEPS + 1) // R3-ok: a size or cost query, not an accumulation
            } else {
                want * offer / OFFER_STEPS
            }
        };
        let mut accumulators = vec![<AccOf<i8> as Accumulator>::ZERO; scale(want_acc, acc_offer)];
        let mut lane_words = vec![0i64; scale(want_lanes, aux_offer)];
        let mut ids = vec![0usize; scale(suggested_tabulation_index(shape), aux_offer)];
        // At the top of the sweep the panel holds the whole decoded operand, so
        // the tile-kernel route is exercised too; below it, only the table and the
        // stream can be reached. All three are asserted against the same bytes.
        let want_panel = suggested_tabulation_panel(C::CODE_SPACE, block)
            .max(n * k + crate::suggested_scratch(shape));
        let mut panel = vec![Alphabet::<i8, Bd>::ZERO; scale(want_panel, aux_offer)];
        // The row-collapse offer is a third knob, and not a fraction like the
        // other two: `0` offers nothing and anything else is the exact number
        // of `Alphabet` elements the distinct rows are allowed to occupy, with
        // the index sized for the pass. A short rows offer is how the decline
        // is exercised (`CD-15`).
        let mut collapse_index = vec![
            0usize;
            if collapse_offer == 0 {
                0
            } else {
                suggested_collapse_index(m)
            }
        ];
        let mut collapse_rows = vec![Alphabet::<i8, Bd>::ZERO; collapse_offer];
        let mut c = vec![0i32; m * n];
        let mut census = Census::default();
        {
            let av = MatView::row_major(
                as_alphabet::<i8, Bd>(a).expect("the activations fit the declared bound"),
                m,
                k,
            )
            .unwrap();
            let cv = MatViewMut::row_major(&mut c, m, n).unwrap();
            let mut tr = TabulatedTriple::new(av, *w, cv).unwrap();
            let mut collapse = if collapse_offer == 0 {
                Collapse::none()
            } else {
                Collapse::new(&mut collapse_index, &mut collapse_rows)
            };
            gemm_tabulated_counted(
                &mut tr,
                &Linear::OVERWRITE,
                options(traversal),
                &mut Scratch::with_accumulators(&mut panel, &mut accumulators),
                &mut Tabulation::with_index(&mut lane_words, &mut ids),
                &mut collapse,
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
                let (got, census) = tabulated(&w, &a, m, n, traversal, offer, offer, 0);
                assert_eq!(
                    got, want,
                    "{label} {m}x{k}x{n}: {traversal:?} at an offer of {offer} \
                     must give the dense driver's bytes ({census:?})"
                );
            }
        }

        // The two offers, disagreeing. Every combination has to give the dense
        // driver's bytes, because each one is a legitimate call: `CD-13`'s claim
        // is that an offer changes the *traversal* and never the answer.
        //
        // This is the pair the single knob could not reach. An accumulator offer
        // below the suggested one narrows the column block, and a narrowed block
        // cannot collapse repeated columns --- a repeat's first occurrence has to
        // be inside the block to be copied from. The sweep read that decision
        // from the index it was handed and the expansion read it from the block
        // width, so with the index buffer full and the accumulators short, every
        // repeat was skipped and never filled.
        for acc_offer in 1..=OFFER_STEPS {
            for aux_offer in 1..=OFFER_STEPS {
                let (got, census) =
                    tabulated(&w, &a, m, n, Traversal::Tabulated, acc_offer, aux_offer, 0);
                assert_eq!(
                    got, want,
                    "{label} {m}x{k}x{n}: accumulators at {acc_offer}/{OFFER_STEPS} and \
                     lanes at {aux_offer}/{OFFER_STEPS} must give the dense driver's \
                     bytes ({census:?})"
                );
            }
        }

        // The collapse is reached and is not vacuous: an operand whose columns
        // repeat is charged for the ones it has.
        if C::CODE_SPACE > 0 {
            let (_, full) = tabulated(
                &w,
                &a,
                m,
                n,
                Traversal::Tabulated,
                OFFER_STEPS,
                OFFER_STEPS,
                0,
            );
            assert!(full.table_reads > 0 || full.kernel_calls > 0);
        }

        // And the comparison is not vacuous: each of the three factorizations is
        // reached by some offer, and the census says which ran rather than the
        // predicate being asked to say it again.
        let (_, with) = tabulated(
            &w,
            &a,
            m,
            n,
            Traversal::Tabulated,
            OFFER_STEPS,
            OFFER_STEPS,
            0,
        );
        let (_, without) = tabulated(&w, &a, m, n, Traversal::Tabulated, 0, 0, 0);
        assert!(
            with.table_reads > 0,
            "{label} {m}x{k}x{n}: the offer was sized for a table and none was read"
        );
        assert_eq!(
            without.table_reads, 0,
            "{label} {m}x{k}x{n}: an offer of nothing cannot read a table"
        );
        // `OutputMajor` names the streaming traversal, and the whole panel offer
        // does not change that: a caller who asks to stream streams.
        let (_, streamed) = tabulated(
            &w,
            &a,
            m,
            n,
            Traversal::OutputMajor,
            OFFER_STEPS,
            OFFER_STEPS,
            0,
        );
        assert_eq!(streamed.table_reads, 0);
        assert_eq!(streamed.multiplies, (m * k * n) as u64);
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

            // The same shape with `d` distinct columns, so the collapse has
            // something to find. Column `j` repeats column `j % d`, which is the
            // shape a weight matrix has when its outputs share a codeword run.
            for d in [1usize, 3, n] {
                let d = d.min(n);
                let base: Vec<u16> = fill(d * (k / 8), 0xd0b, |x| (x % 400) as u16);
                let repeated: Vec<u16> = (0..n * (k / 8))
                    .map(|x| base[(x / (k / 8) % d) * (k / 8) + x % (k / 8)])
                    .collect();
                every_traversal_agrees("Book<256,8> repeated", book, &repeated, m, k, n);
            }
        }

        // An *odd* codeword width. `Book<_, _, N, 3>` is ordinary public API, and
        // no vector table sequence can pack it: they fold `k_group == 2` block
        // steps into one instruction, and three is not a whole number of pairs.
        // Selection has to decline them and leave the reference, which declares
        // `k_group: 1`. It did not, and the driver's packer indexed past the
        // activation tile and panicked --- for every consumer with runtime CPU
        // detection on, which is every consumer of `uor-matmul` with default
        // features. This crate's own tests could not see it until they linked the
        // kernels with `std`.
        let odd: [[A8; 3]; 64] = core::array::from_fn(|c| {
            core::array::from_fn(|t| Alphabet::of(((c * 7 + t * 13) % 200) as i64 as i8))
        });
        let odd_book = Book::<i8, Full<i8>, 64, 3>::new(&odd);
        for &(m, k, n) in &[(1usize, 3usize, 1usize), (16, 24, 32), (5, 9, 37)] {
            let stream: Vec<u16> = fill(n * (k / 3), 0x0dd, |x| (x % 64) as u16);
            every_traversal_agrees("Book<64,3>", odd_book, &stream, m, k, n);
        }

        let i4: [A8; 16] = core::array::from_fn(|i| Alphabet::of((i as i8) - 8));
        let grid = Grid::<i8, Full<i8>, 16>::new(&i4);
        let packed = Packed::<_, 2>::new(grid).expect("2 divides 8");
        for &(m, k, n) in &[(1usize, 2usize, 1usize), (4, 6, 600), (6, 10, 13)] {
            let stream: Vec<u8> = fill(n * (k / 2), 0xd0e, |x| x as u8);
            every_traversal_agrees("Packed<Grid<16>,2>", packed, &stream, m, k, n);
        }
    }

    /// One arena-coded float product at one traversal and one offer.
    ///
    /// The same shape as [`tabulated`], with one knob scaling the panel,
    /// accumulator and index offers together so the extremes --- nothing, one
    /// word, exactly the suggested amount, and a multiple --- are all reachable.
    /// The lane offer moves with the same knob: `f32`'s table lane is the
    /// compact Atlas word (`CD-20`), which lives in the narrow offer; `f64`
    /// has no executable table and its API-locked lane needs no narrow words.
    ///
    /// Generic over the code width (`CK-14`): the `u8` and `u16` spellings of
    /// one codebook are the same tier at two residencies, and both are asserted
    /// against the same dense bytes.
    #[allow(clippy::too_many_arguments)]
    fn arena_tabulated<E, const D: usize, K: SymbolCode, Ep>(
        table: &[Alphabet<E, Whole<E>>; D],
        codes: &[K],
        a: &[E],
        m: usize,
        k: usize,
        n: usize,
        traversal: Traversal,
        offer: usize,
        collapse_offer: usize,
        epilogue: &Ep,
        c0: &[E],
    ) -> (Vec<E>, Census)
    where
        E: FloatElement + EncodeFrom<AccOf<E>> + Tabulated,
        AccOf<E>: crate::SignedPlace,
        Ep: Epilogue<E, E>,
    {
        let shape = Shape { m, k, n };
        let space = <Arena<'_, E, D, K> as Enumerable<E, Whole<E>>>::CODE_SPACE;
        let block = <Arena<'_, E, D, K> as uor_matmul_codec::Codec<E, Whole<E>>>::MAX_BLOCK;
        let scale = |want: usize, offer: usize| -> usize {
            if offer >= OFFER_STEPS {
                want.saturating_mul(offer - OFFER_STEPS + 1) // R3-ok: a size or cost query, not an accumulation
            } else {
                want * offer / OFFER_STEPS
            }
        };
        let mut accumulators = vec![
            <AccOf<E> as Accumulator>::ZERO;
            scale(
                suggested_tabulation::<E, Whole<E>>(shape, space, block).max(1),
                offer
            )
        ];
        let mut lane_words = vec![
            0i64;
            scale(
                suggested_tabulation_lanes::<E, Whole<E>>(shape, space, block),
                offer
            )
        ];
        let mut ids = vec![0usize; scale(suggested_tabulation_index(shape), offer)];
        // At the top of the sweep the panel holds the whole decoded operand, so
        // the dense decline route is exercised too; below it, only the table and
        // the stream can be reached. All three are asserted against the same bytes.
        let want_panel =
            suggested_tabulation_panel(space, block).max(n * k + crate::suggested_scratch(shape));
        let mut panel = vec![Alphabet::<E, Whole<E>>::ZERO; scale(want_panel, offer)];
        // The row-collapse offer, as the integer helper's: `0` offers nothing
        // and anything else is the exact number of `Alphabet` elements the
        // distinct rows are allowed to occupy, with the index sized for the
        // pass (`CD-17`).
        let mut collapse_index = vec![
            0usize;
            if collapse_offer == 0 {
                0
            } else {
                suggested_collapse_index(m)
            }
        ];
        let mut collapse_rows = vec![Alphabet::<E, Whole<E>>::ZERO; collapse_offer];
        let mut c = c0.to_vec();
        let mut census = Census::default();
        {
            let av = MatView::row_major(as_alphabet_whole(a), m, k).unwrap();
            let cv = MatViewMut::row_major(&mut c, m, n).unwrap();
            let w =
                CodedMatrix::new(Arena::new(table), n, k, codes).expect("the codes describe n x k");
            let mut tr = TabulatedTriple::new(av, w, cv).unwrap();
            let mut collapse = if collapse_offer == 0 {
                Collapse::none()
            } else {
                Collapse::new(&mut collapse_index, &mut collapse_rows)
            };
            gemm_tabulated_counted(
                &mut tr,
                epilogue,
                GemmOptions {
                    traversal,
                    ..Default::default()
                },
                &mut Scratch::with_accumulators(&mut panel, &mut accumulators),
                &mut Tabulation::with_index(&mut lane_words, &mut ids),
                &mut collapse,
                &mut census,
            );
        }
        (c, census)
    }

    /// The dense float driver's product of the same operands, which is the
    /// identity the arena claims to code. `W` is decoded into a dense `k x n`
    /// matrix first, so this is the product itself and not a restatement of it.
    ///
    /// The decode is written out here --- a direct table read, codes reduced
    /// modulo `D` --- rather than delegated to the codec under test: a
    /// reference that decoded through the tier would share a wrong decode with
    /// the traversal and the comparison would be the tier against itself. This
    /// is the vacuity R13's harness exists to refuse, and it was planted: an
    /// off-by-one in the tier's decode passed every arena test until this
    /// reference read the table itself.
    #[allow(clippy::too_many_arguments)]
    fn arena_reference<E, const D: usize, K: SymbolCode + Into<usize>, Ep>(
        table: &[Alphabet<E, Whole<E>>; D],
        codes: &[K],
        a: &[E],
        m: usize,
        k: usize,
        n: usize,
        epilogue: &Ep,
        c0: &[E],
    ) -> Vec<E>
    where
        E: FloatElement + EncodeFrom<AccOf<E>>,
        AccOf<E>: crate::SignedPlace,
        Ep: Epilogue<E, E>,
    {
        let mut b = vec![E::ZERO; k * n];
        for p in 0..k {
            for j in 0..n {
                // The intended operand, decoded without the tier: `W` is
                // `n x k`, so `B[p][j]` is the symbol codes[j][p] names.
                let code: usize = codes[j * k + p].into();
                b[p * n + j] = table[code % D].get();
            }
        }
        let mut c = c0.to_vec();
        let av = MatView::row_major(a, m, k).unwrap();
        let bv = MatView::row_major(&b, k, n).unwrap();
        let cv = MatViewMut::row_major(&mut c, m, n).unwrap();
        let mut t = Triple::new(av, bv, cv).unwrap();
        crate::float::gemm_float(&mut t, epilogue, GemmOptions::default());
        c
    }

    /// Independent route expectation for the compact f32 lane.
    ///
    /// The full index offer presents each distinct addressed coordinate once
    /// when its reclaimed dictionary can hold the set, and otherwise retains
    /// the raw-short/full-book factorization. This independent oracle spells
    /// that storage law directly. Symbol values do not participate: the total
    /// q carrier makes both finite and non-finite addressed coordinates table-
    /// executable once the resident table is demanded.
    fn f32_demand_book_indices<const D: usize, K: SymbolCode + Into<usize>>(
        codes: &[K],
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<usize> {
        let fallback = || {
            if codes.len() < D {
                codes.iter().map(|&code| code.into() % D).collect()
            } else {
                (0..D).collect()
            }
        };
        if D <= 1 || n == 0 || k == 0 || codes.len() < n * k {
            return fallback();
        }
        let shape = Shape { m, k, n };
        let exact = suggested_tabulation::<f32, Whole<f32>>(shape, D, 1);
        let lanes = suggested_tabulation_lanes::<f32, Whole<f32>>(shape, D, 1);
        let lane_words = core::mem::size_of::<i64>() * lanes / <f32 as Tabulated>::LANE_BYTES;
        let Some(plan) = Plan::choose(
            D,
            shape,
            <f32 as Tabulated>::LANE_BYTES,
            exact,
            lane_words,
            1,
            <f32 as Tabulated>::probe_capacity::<<f32 as Tabulated>::Lane>(
                <Whole<f32> as Bound>::VALUE,
            ),
        ) else {
            return fallback();
        };
        let dictionary = (2 * n).next_power_of_two();
        let repeated_columns = (0..n).any(|column| {
            let run = &codes[column * k..(column + 1) * k];
            (0..column).any(|prior| {
                run.iter()
                    .zip(&codes[prior * k..(prior + 1) * k])
                    .all(|(&left, &right)| left.into() % D == right.into() % D)
            })
        });
        let occupied = if repeated_columns {
            dictionary - n.div_ceil(plan.cols)
        } else {
            dictionary
        };
        let mut distinct = Vec::new();
        for &code in codes {
            let index = code.into() % D;
            if !distinct.contains(&index) {
                distinct.push(index);
            }
        }
        if distinct.len() <= occupied {
            distinct
        } else {
            fallback()
        }
    }

    fn f32_demand_table_expected<const D: usize, K: SymbolCode + Into<usize>>(
        table: &[Alphabet<f32, Whole<f32>>; D],
        codes: &[K],
        a: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> bool {
        // CD-32's q carrier is total over every binary32 symbol and exponent
        // span. This oracle remains at the call sites to state the forced-route
        // expectation independently of the production scale walk: once the
        // caller supplies the resident table offer, values cannot decline it.
        let _ = (table, codes, a, m, k, n);
        true
    }

    /// Every traversal at every offer, against the dense float driver's bytes.
    ///
    /// `table_expected` is what the census must show for the forced traversal
    /// at a full offer: every binary32 support is executable through the total
    /// q carrier, and `f64` uses its executable complete lane. A false value is
    /// therefore a family/geometry expectation, witnessed by `table_reads == 0`
    /// plus `kernel_calls > 0`, rather than a value-based f32 refusal.
    #[allow(clippy::too_many_arguments)]
    fn every_arena_traversal_agrees<E, const D: usize, K: SymbolCode + Into<usize>>(
        label: &str,
        table: &[Alphabet<E, Whole<E>>; D],
        codes: &[K],
        a: &[E],
        m: usize,
        k: usize,
        n: usize,
        epilogue: &Linear,
        c0: &[E],
        table_expected: bool,
    ) where
        E: FloatElement + EncodeFrom<AccOf<E>> + Tabulated,
        AccOf<E>: crate::SignedPlace,
        Linear: Epilogue<E, E>,
    {
        // Bit patterns, not values: a NaN in the codebook is a NaN in the
        // output, and NaN is not `==` itself.
        let want: Vec<u64> = arena_reference(table, codes, a, m, k, n, epilogue, c0)
            .iter()
            .map(|v| v.symbol_bits())
            .collect();

        // Nothing, a sliver, most of it, exactly it, and three times it.
        let offers = [0, 1, 2, OFFER_STEPS - 1, OFFER_STEPS, OFFER_STEPS + 2];
        for traversal in [
            Traversal::Tabulated,
            Traversal::Blocked,
            Traversal::OutputMajor,
        ] {
            for offer in offers {
                let (got, census) =
                    arena_tabulated(table, codes, a, m, k, n, traversal, offer, 0, epilogue, c0);
                let got: Vec<u64> = got.iter().map(|v| v.symbol_bits()).collect();
                assert_eq!(
                    got, want,
                    "{label} {m}x{k}x{n}: {traversal:?} at an offer of {offer} must give \
                     the dense float driver's bytes ({census:?})"
                );
            }
        }

        // And the comparison is not vacuous: the forced traversal read a table
        // exactly when one was expected to run, the costed one really declined
        // the contextual block-one case that CG-16 found has no universal
        // scalar boundary, and an offer of nothing reached the persistent Atlas
        // StreamLane. The census says which ran rather than the predicate being
        // asked to say it again.
        let (_, tabled) = arena_tabulated(
            table,
            codes,
            a,
            m,
            k,
            n,
            Traversal::Tabulated,
            OFFER_STEPS,
            0,
            epilogue,
            c0,
        );
        if table_expected {
            assert!(
                tabled.table_reads > 0,
                "{label} {m}x{k}x{n}: the offer was sized for a table and none was read"
            );
        } else {
            assert_eq!(
                tabled.table_reads, 0,
                "{label} {m}x{k}x{n}: no table is executable for these panels \
                 ({tabled:?})"
            );
            assert!(
                tabled.kernel_calls > 0,
                "{label} {m}x{k}x{n}: a full offer reaches the dense Atlas engine \
                 ({tabled:?})"
            );
        }
        let (_, declined) = arena_tabulated(
            table,
            codes,
            a,
            m,
            k,
            n,
            Traversal::Blocked,
            OFFER_STEPS,
            0,
            epilogue,
            c0,
        );
        assert!(
            declined.kernel_calls > 0,
            "{label} {m}x{k}x{n}: `Blocked` must decline below the measured \
             block-one boundary ({declined:?})"
        );
        let (_, without) = arena_tabulated(
            table,
            codes,
            a,
            m,
            k,
            n,
            Traversal::Tabulated,
            0,
            0,
            epilogue,
            c0,
        );
        assert_eq!(
            without.table_reads, 0,
            "{label} {m}x{k}x{n}: an offer of nothing cannot read a table"
        );
        assert!(
            without.kernel_calls > 0,
            "{label} {m}x{k}x{n}: an empty offer must declare the Atlas StreamLane \
             ({without:?})"
        );
        assert_eq!(
            without.multiplies, 0,
            "{label} {m}x{k}x{n}: the empty-offer float route must issue no Element::mac \
             ({without:?})"
        );
    }

    /// `CD-14`: an arena-coded float weight matrix gives the dense float
    /// driver's bytes at every shape, through every named traversal and with
    /// every offer including none.
    ///
    /// The reference is `gemm_float` over the decoded weights, not another
    /// tabulated run: an agreement between two tabulations would say nothing
    /// about whether either computes the product.
    #[test]
    fn an_arena_coded_float_matrix_matches_the_dense_driver_cd_14() {
        // The distinct symbols one artifact's weights can hold. `-0.0` and
        // `+0.0` are distinct symbols with equal dyadic values, and an infinity and a
        // NaN are codes like any other (`CT-03`): the exact accumulator carries
        // them as flags and the encode step writes them once. `canonicalize`
        // orders by unsigned bit pattern, so the small tables hold the low
        // patterns and the zeros, the infinity, and the NaN enter at `d = 6, 4,
        // 5` respectively.
        let mut pool32 = [0.5f32, 1.0, f32::INFINITY, f32::NAN, -0.0, -1.5, -2.5, 0.0];
        assert_eq!(canonicalize(&mut pool32), 8, "eight distinct bit patterns");
        let mut pool64 = [0.5f64, 1.0, f64::INFINITY, f64::NAN, -0.0, -1.5, -2.5, 0.0];
        assert_eq!(canonicalize(&mut pool64), 8, "eight distinct bit patterns");

        macro_rules! sweep {
            ($d:literal) => {{
                let t32: &[Alphabet<f32, Whole<f32>>; $d] =
                    as_alphabet_whole(&pool32[..$d]).try_into().unwrap();
                let t64: &[Alphabet<f64, Whole<f64>>; $d] =
                    as_alphabet_whole(&pool64[..$d]).try_into().unwrap();
                for &(m, k, n) in &[
                    (1usize, 1usize, 1usize),
                    (2, 3, 5),
                    (5, 17, 7),
                    (13, 11, 3),
                    (7, 40, 9),
                ] {
                    // Codes past the table on purpose: the enumeration reduces
                    // them modulo `D`, and the reference decodes them the same way.
                    let codes: Vec<u16> = fill(n * k, 0xa4ea, |x| (x % (2 * $d as u64 + 1)) as u16);
                    // A zero among the activations, so `inf * 0` is a product
                    // someone computes and clause 7.2 has to answer for.
                    let a32: Vec<f32> = fill(m * k, 0xac7, |x| (x % 7) as f32 * 0.5 - 1.5);
                    let a64: Vec<f64> = fill(m * k, 0xac7, |x| (x % 7) as f64 * 0.5 - 1.5);
                    every_arena_traversal_agrees(
                        concat!("Arena<", $d, "> f32"),
                        t32,
                        &codes,
                        &a32,
                        m,
                        k,
                        n,
                        &Linear::OVERWRITE,
                        &vec![0.0f32; m * n],
                        f32_demand_table_expected(t32, &codes, &a32, m, k, n),
                    );
                    every_arena_traversal_agrees(
                        concat!("Arena<", $d, "> f64"),
                        t64,
                        &codes,
                        &a64,
                        m,
                        k,
                        n,
                        &Linear::OVERWRITE,
                        &vec![0.0f64; m * n],
                        // The API-locked complete lane is resident and
                        // executable under the forced traversal.
                        true,
                    );
                }
            }};
        }
        // Two to eight distinct symbols: below two there is no codebook to speak
        // of, and eight is where every non-finite symbol above is in play. The
        // odd counts take the offset-run gather, the powers of two the borrowed
        // index stream (`CB-08`).
        sweep!(2);
        sweep!(3);
        sweep!(4);
        sweep!(5);
        sweep!(6);
        sweep!(7);
        sweep!(8);
    }

    /// `CD-20`/`CD-32`: demand decoding remains a property of the symbols this
    /// call addresses, while neither an addressed nor an unused non-finite
    /// symbol can decline the total q table. The two calls share the same live
    /// route and differ only in the boundary token the table contracts.
    #[test]
    fn unused_nonfinite_symbols_do_not_widen_a_demand_table_cd_20() {
        let symbols = [0.5f32, f32::INFINITY, -0.75, 1.25];
        let table: &[Alphabet<f32, Whole<f32>>; 4] =
            as_alphabet_whole(&symbols).try_into().unwrap();
        let activation = [0.875f32];
        let zeros = [0.0f32];

        for code in [0u8, 1] {
            assert!(
                f32_demand_table_expected(table, &[code], &activation, 1, 1, 1),
                "the total q route oracle"
            );
            let (_, census) = arena_tabulated(
                table,
                &[code],
                &activation,
                1,
                1,
                1,
                Traversal::Tabulated,
                OFFER_STEPS,
                0,
                &Linear::OVERWRITE,
                &zeros,
            );
            assert!(census.table_reads > 0, "{census:?}");
            assert_eq!(census.kernel_calls, 0, "{census:?}");
        }
    }

    /// `CD-32`: lane capacity uses the exact compact ceiling and exact maximum
    /// coefficient product. A non-finite witness and an exponent span that
    /// exhausts binary32 are executable one-product lanes rather than failed
    /// scale queries. This is the production-side differential for the model
    /// pins: a model-only arithmetic assertion cannot catch a driver that still
    /// divides `i64::MAX` by the conservative power-of-two bound.
    #[test]
    fn total_f32_lane_scale_uses_the_exact_q_capacity_cd_32() {
        fn observe<const D: usize>(
            symbols: &[f32; D],
            codes: &[u8],
            activations: &[f32],
        ) -> (Option<LaneScale>, Census) {
            let table: &[Alphabet<f32, Whole<f32>>; D] =
                as_alphabet_whole(symbols).try_into().unwrap();
            let k = activations.len();
            let a = MatView::row_major(as_alphabet_whole(activations), 1, k).unwrap();
            let w = CodedMatrix::new(Arena::new(table), 1, k, codes)
                .expect("the codes describe the one-column reduction");
            let mut census = Census::default();
            let scale = <f32 as Tabulated>::lane_scale(&a, &w, &mut census);
            (scale, census)
        }

        let edge = f32::from_bits(0x3fff_ffff);
        let (scale, census) = observe(&[edge], &[0], &[edge]);
        let scale = scale.expect("a finite zero-span panel is executable");
        assert_eq!(
            scale.per_step,
            u128::from(f32_q::PRODUCT_BOUND),
            "the one-product bound is exact rather than the next power of two"
        );
        assert_eq!(
            <f32 as Tabulated>::lane_run::<Scaled64>(0, &scale),
            Some(usize::try_from(f32_q::ZERO_SPAN_CAPACITY).unwrap())
        );
        assert_eq!(census.decodes, 2, "one activation and one book symbol");

        let (scale, census) = observe(&[1.0], &[0], &[f32::NAN]);
        let scale = scale.expect("a non-finite witness is a total one-product token");
        assert_eq!(<f32 as Tabulated>::lane_run::<Scaled64>(0, &scale), Some(1));
        assert_eq!(
            census.decodes, 2,
            "the non-finite witness fixes run one without hiding the book base"
        );

        let extremes = [f32::from_bits(1), f32::MAX];
        let (scale, census) = observe(&extremes, &[0, 1], &extremes);
        let scale = scale.expect("the complete binary32 exponent span is executable");
        assert_eq!(<f32 as Tabulated>::lane_run::<Scaled64>(0, &scale), Some(1));
        assert_eq!(census.decodes, 4, "both complete finite spans are observed");
    }

    /// `CD-32`: the resident forced table is total over the complete binary32
    /// exponent range and all seven sticky non-finite unions. The exponent
    /// sweep alternates extremes, so a global compact span is only a totality
    /// bound; it cannot be used as a reason to enter the dense route.
    #[test]
    fn total_f32_q_carrier_executes_every_ieee_boundary_cd_32() {
        #[allow(clippy::too_many_arguments)]
        fn forced<const D: usize>(
            label: &str,
            table: &[Alphabet<f32, Whole<f32>>; D],
            codes: &[u8],
            activations: &[f32],
            m: usize,
            k: usize,
            n: usize,
        ) {
            let zeros = vec![0.0f32; m * n];
            let want = arena_reference(
                table,
                codes,
                activations,
                m,
                k,
                n,
                &Linear::OVERWRITE,
                &zeros,
            );
            let (got, census) = arena_tabulated(
                table,
                codes,
                activations,
                m,
                k,
                n,
                Traversal::Tabulated,
                OFFER_STEPS,
                0,
                &Linear::OVERWRITE,
                &zeros,
            );
            assert_eq!(
                got.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                want.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                "{label}: complete reference bytes ({census:?})"
            );
            assert!(
                census.table_reads > 0,
                "{label}: the resident forced table executed ({census:?})"
            );
            assert_eq!(
                census.kernel_calls, 0,
                "{label}: no dense decline ({census:?})"
            );
            assert_eq!(
                census.multiplies, 0,
                "{label}: lookup/add UOR only ({census:?})"
            );
        }

        // One representative of every exponent field. The stored fraction is
        // nonzero and signs alternate, so the sweep covers both directions of
        // the full finite range before reaching the infinity field.
        let exponent_symbols: [f32; 256] = core::array::from_fn(|biased| match biased {
            0 => f32::from_bits(1),
            255 => f32::from_bits(0x7f80_0001),
            _ => {
                let sign = if biased % 2 == 0 { 0 } else { 1u32 << 31 };
                f32::from_bits(sign | ((biased as u32) << 23) | biased as u32)
            }
        });
        let exponent_table: &[Alphabet<f32, Whole<f32>>; 256] =
            as_alphabet_whole(&exponent_symbols).try_into().unwrap();
        let exponent_codes: Vec<u8> = (0u8..=u8::MAX).collect();
        forced(
            "every binary32 exponent field",
            exponent_table,
            &exponent_codes,
            &[1.0],
            1,
            1,
            exponent_codes.len(),
        );

        // Columns are the seven nonempty subsets of {+Inf,-Inf,NaN}. The three
        // source positions independently set the sticky Complete states; the
        // zero is the absent member of each subset.
        let union_symbols = [f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 0.0];
        let union_table: &[Alphabet<f32, Whole<f32>>; 4] =
            as_alphabet_whole(&union_symbols).try_into().unwrap();
        let union_codes = [
            0u8, 3, 3, // +Inf
            3, 1, 3, // -Inf
            3, 3, 2, // NaN
            0, 1, 3, // +Inf | -Inf
            0, 3, 2, // +Inf | NaN
            3, 1, 2, // -Inf | NaN
            0, 1, 2, // +Inf | -Inf | NaN
        ];
        forced(
            "all seven Complete non-finite unions",
            union_table,
            &union_codes,
            &[1.0, 1.0, 1.0],
            1,
            3,
            7,
        );
    }

    /// `CD-32`: zero depth is the additive identity before any q scale, book,
    /// table, fracture, or kernel work exists. The exact all-zero Census makes
    /// the boundary independent of an inferred route label.
    #[test]
    fn empty_f32_q_reduction_has_zero_work_cd_32() {
        let shape = Shape { m: 3, k: 0, n: 5 };
        let symbols = [0.5f32];
        let table: &[Alphabet<f32, Whole<f32>>; 1] =
            as_alphabet_whole(&symbols).try_into().unwrap();
        let (got, census) = arena_tabulated(
            table,
            &[] as &[u8],
            &[],
            shape.m,
            shape.k,
            shape.n,
            Traversal::Tabulated,
            OFFER_STEPS,
            0,
            &Linear::OVERWRITE,
            &[7.0; 15],
        );
        assert_eq!(census, Census::default());
        assert_eq!(got, vec![0.0; 15]);
    }

    /// `CD-32`: scalar fracture is parametric in both codec dimensions. Odd
    /// block/code-space extents, a row-strided activation, row/column tails,
    /// repeated codes, and absent/short/full offers all retain the dense bytes;
    /// the full offer remains a multiply-free resident q table.
    #[test]
    fn parametric_nonpower_q_blocks_preserve_bytes_strides_offers_and_census_cd_32() {
        fn exercise<const B: usize, const D: usize>() {
            let codec = ParametricFractureF32::<B, D>;
            let shape = Shape {
                m: 3,
                k: 2 * B,
                n: 7,
            };
            let row_stride = shape.k + 1;
            let mut activation_storage = vec![0.0f32; shape.m * row_stride];
            for i in 0..shape.m {
                for p in 0..shape.k {
                    activation_storage[i * row_stride + p] = if (i + p).is_multiple_of(2) {
                        f32::from_bits(0x3fff_ffff)
                    } else {
                        f32::from_bits(0x437f_ffff)
                    };
                }
            }
            let activation_view = MatView::new(
                as_alphabet_whole(&activation_storage),
                shape.m,
                shape.k,
                Strides {
                    rs: row_stride as isize,
                    cs: 1,
                },
            )
            .unwrap();
            let activation_dense = MatView::new(
                &activation_storage,
                shape.m,
                shape.k,
                Strides {
                    rs: row_stride as isize,
                    cs: 1,
                },
            )
            .unwrap();
            let blocks = shape.k / B;
            let codes: Vec<u8> = (0..shape.n)
                .flat_map(|j| {
                    (0..blocks).map(move |p| {
                        u8::try_from((j + p) % D).expect("the test code space fits u8")
                    })
                })
                .collect();
            let mut decoded = vec![0.0f32; shape.k * shape.n];
            for p in 0..shape.k {
                for j in 0..shape.n {
                    decoded[p * shape.n + j] =
                        codec.decode_element(codes[j * blocks + p / B], p % B).get();
                }
            }
            let mut want = vec![0.0f32; shape.m * shape.n];
            {
                let b = MatView::row_major(&decoded, shape.k, shape.n).unwrap();
                let c = MatViewMut::row_major(&mut want, shape.m, shape.n).unwrap();
                let mut dense = Triple::new(activation_dense, b, c).unwrap();
                gemm_float(&mut dense, &Linear::OVERWRITE, GemmOptions::default());
            }

            for offer in [0, 1, usize::MAX] {
                let offered = |want: usize| want.min(offer);
                let mut exact = vec![
                    <AccOf<f32> as Accumulator>::ZERO;
                    offered(suggested_tabulation::<f32, Whole<f32>>(shape, D, B))
                ];
                let mut lanes =
                    vec![0i64; offered(suggested_tabulation_lanes::<f32, Whole<f32>>(shape, D, B))];
                let mut panel = vec![
                    Alphabet::<f32, Whole<f32>>::ZERO;
                    offered(suggested_tabulation_panel(D, B))
                ];
                let mut index = vec![0usize; offered(suggested_tabulation_index(shape))];
                let mut got = vec![0.0f32; shape.m * shape.n];
                let mut census = Census::default();
                {
                    let c = MatViewMut::row_major(&mut got, shape.m, shape.n).unwrap();
                    let w = CodedMatrix::new(codec, shape.n, shape.k, &codes).unwrap();
                    let mut triple = TabulatedTriple::new(activation_view, w, c).unwrap();
                    gemm_tabulated_counted(
                        &mut triple,
                        &Linear::OVERWRITE,
                        GemmOptions {
                            traversal: Traversal::Tabulated,
                            ..Default::default()
                        },
                        &mut Scratch::with_accumulators(&mut panel, &mut exact),
                        &mut Tabulation::with_index(&mut lanes, &mut index),
                        &mut Collapse::none(),
                        &mut census,
                    );
                }
                assert_eq!(
                    got.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                    want.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                    "B={B}, D={D}, offer={offer}: exact dense bytes ({census:?})"
                );
                assert_eq!(census.multiplies, 0, "B={B}, D={D}, offer={offer}");
                if offer == usize::MAX {
                    assert!(census.table_reads > 0, "resident q table ({census:?})");
                    assert_eq!(census.kernel_calls, 0, "no dense fallback ({census:?})");
                }
            }
        }

        exercise::<3, 3>();
        exercise::<5, 5>();
    }

    /// `CD-32`: special q atoms are placed at their source boundaries, not
    /// combined with a compact finite residue. The final IEEE symbol alone
    /// cannot expose a lost low residue once infinity is sticky, so this test
    /// captures the complete accumulator and compares every limb/state byte to
    /// the independent element contraction in source order.
    #[test]
    fn f32_q_special_atoms_are_immediate_source_order_singletons_cd_32() {
        struct Capture<'a>(&'a core::cell::Cell<AccOf<f32>>);

        impl Epilogue<f32, f32> for Capture<'_> {
            fn finish(&self, acc: AccOf<f32>, _: Option<f32>, _: EncodeMode) -> f32 {
                self.0.set(acc);
                0.0
            }

            fn reads_c(&self) -> bool {
                false
            }
        }

        let symbols = [0.5f32, f32::INFINITY, -0.25, f32::NEG_INFINITY];
        let table: &[Alphabet<f32, Whole<f32>>; 4] =
            as_alphabet_whole(&symbols).try_into().unwrap();
        // Finite residue precedes and follows +Inf, then -Inf, then another
        // finite atom. The global run is one and k is five, so this necessarily
        // executes the local source-ordered scheduler.
        let codes = [0u8, 1, 2, 3, 0];
        let activations = [1.0f32; 5];
        let mut expected = <AccOf<f32> as Accumulator>::ZERO;
        for (&activation, &code) in activations.iter().zip(&codes) {
            f32::mac(&mut expected, activation, symbols[usize::from(code)]);
        }

        let captured = core::cell::Cell::new(<AccOf<f32> as Accumulator>::ZERO);
        let (_, census) = arena_tabulated(
            table,
            &codes,
            &activations,
            1,
            5,
            1,
            Traversal::Tabulated,
            OFFER_STEPS,
            0,
            &Capture(&captured),
            &[0.0],
        );
        assert_eq!(
            captured.get(),
            expected,
            "singleton placement preserves finite residue and every sticky boundary state"
        );
        assert_eq!(census.table_reads, 5, "one gather per source atom");
        assert_eq!(census.kernel_calls, 0, "the total q table never declined");
        assert_eq!(
            census.multiplies, 0,
            "every contraction is Atlas lookup/add"
        );
    }

    /// `CD-32`: a capacity shorter than a codec word is still executable. The
    /// two seven-grade panels have capacity one, while each stored code names
    /// two source positions. Their unsafe whole-block aggregate crosses the
    /// compact ceiling, so exact bytes plus a live table route are a semantic
    /// witness for scalar fracture with the codec's original stride.
    #[test]
    fn f32_q_lane_scalar_fractures_a_wider_codec_block_cd_32() {
        let (m, k, n) = (2usize, 2usize, 5usize);
        let shape = Shape { m, k, n };
        let low = f32::from_bits(0x3fff_ffff);
        let high = f32::from_bits(0x437f_ffff);
        let activations = [low, high, high, high];
        let codes = [0u8, 1, 0, 1, 0];

        let compact_ceiling = u128::from(f32_q::COMPACT_CEILING);
        let product_bound = u128::from(f32_q::PRODUCT_BOUND);
        assert_eq!(
            uor_matmul_model::derive::f32_q_lane_capacity(
                compact_ceiling,
                product_bound,
                7,
                7,
                false,
            ),
            1
        );
        const { assert!(1 < FractureF32::MAX_BLOCK) };
        assert!(
            product_bound
                .checked_mul(1u128 << 14)
                .and_then(|one| one.checked_add(one))
                .is_some_and(|whole| whole > compact_ceiling),
            "a whole two-position worst-case code cannot be one compact token"
        );

        let mut decoded = vec![0.0f32; k * n];
        for p in 0..k {
            for j in 0..n {
                decoded[p * n + j] = FractureF32
                    .decode_element(codes[j], p % FractureF32::MAX_BLOCK)
                    .get();
            }
        }
        let mut want = vec![0.0f32; m * n];
        {
            let a = MatView::row_major(&activations, m, k).unwrap();
            let b = MatView::row_major(&decoded, k, n).unwrap();
            let c = MatViewMut::row_major(&mut want, m, n).unwrap();
            let mut dense = Triple::new(a, b, c).unwrap();
            gemm_float(&mut dense, &Linear::OVERWRITE, GemmOptions::default());
        }

        fn run(
            shape: Shape,
            activations: &[f32],
            codes: &[u8],
            traversal: Traversal,
        ) -> (Vec<f32>, Census) {
            let exact_words = suggested_tabulation::<f32, Whole<f32>>(
                shape,
                FractureF32::CODE_SPACE,
                FractureF32::MAX_BLOCK,
            );
            let lane_words = suggested_tabulation_lanes::<f32, Whole<f32>>(
                shape,
                FractureF32::CODE_SPACE,
                FractureF32::MAX_BLOCK,
            );
            let mut exact = vec![<AccOf<f32> as Accumulator>::ZERO; exact_words];
            let mut lanes = vec![0i64; lane_words];
            let mut panel =
                vec![
                    Alphabet::<f32, Whole<f32>>::ZERO;
                    suggested_tabulation_panel(FractureF32::CODE_SPACE, FractureF32::MAX_BLOCK,)
                ];
            let mut index = vec![0usize; suggested_tabulation_index(shape)];
            let mut got = vec![0.0f32; shape.m * shape.n];
            let mut census = Census::default();
            {
                let a =
                    MatView::row_major(as_alphabet_whole(activations), shape.m, shape.k).unwrap();
                let w = CodedMatrix::new(FractureF32, shape.n, shape.k, codes).unwrap();
                let c = MatViewMut::row_major(&mut got, shape.m, shape.n).unwrap();
                let mut triple = TabulatedTriple::new(a, w, c).unwrap();
                gemm_tabulated_counted(
                    &mut triple,
                    &Linear::OVERWRITE,
                    GemmOptions {
                        traversal,
                        ..Default::default()
                    },
                    &mut Scratch::with_accumulators(&mut panel, &mut exact),
                    &mut Tabulation::with_index(&mut lanes, &mut index),
                    &mut Collapse::none(),
                    &mut census,
                );
            }
            (got, census)
        }

        let (got, forced) = run(shape, &activations, &codes, Traversal::Tabulated);
        assert_eq!(
            got.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|value| value.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            forced.table_reads,
            (m * FractureF32::CODE_SPACE * k) as u64,
            "every fractured scalar of each distinct coded column is gathered; repeated columns are expanded after reduction ({forced:?})"
        );
        assert_eq!(
            forced.adds, 24,
            "eight scalar q contractions, eight gather combines, and eight exact scalar-envelope contractions ({forced:?})"
        );
        assert_eq!(
            forced.decodes, 44,
            "call span, contextual projections, and replayed least scalar certificates are all charged ({forced:?})"
        );
        assert_eq!(forced.kernel_calls, 0, "no dense decline ({forced:?})");
        assert_eq!(forced.multiplies, 0, "lookup/add UOR only ({forced:?})");

        let (_, finite_route) = run(shape, &activations, &codes, Traversal::Blocked);
        let mut nonfinite = activations;
        nonfinite[0] = f32::NAN;
        let (_, special_route) = run(shape, &nonfinite, &codes, Traversal::Blocked);
        let finite_signature = (finite_route.table_reads > 0, finite_route.kernel_calls);
        let special_signature = (special_route.table_reads > 0, special_route.kernel_calls);
        assert_eq!(
            finite_signature, special_signature,
            "selection is value-blind"
        );
        assert!(
            finite_signature.0,
            "the shared automatic route is nonvacuously tabulated"
        );
        assert_eq!(finite_signature.1, 0);
    }

    /// `CK-14`, the traversal half: the `u8` and `u16` spellings of one
    /// codebook give byte-identical output through every traversal --- the
    /// packed decline, the table, and the stream alike --- because they decode
    /// the same stream and the codec is not an argument of the arithmetic
    /// below it.
    #[test]
    fn the_u8_and_u16_spellings_give_the_same_bytes_ck_14() {
        let mut pool32 = [0.5f32, 1.0, f32::INFINITY, f32::NAN, -0.0, -1.5, -2.5, 0.0];
        assert_eq!(canonicalize(&mut pool32), 8, "eight distinct bit patterns");
        let t32: &[Alphabet<f32, Whole<f32>>; 8] = as_alphabet_whole(&pool32).try_into().unwrap();

        for &(m, k, n) in &[(1usize, 1usize, 1usize), (5, 17, 7), (7, 40, 9)] {
            // One artifact at two widths: codes below 256, so the `u8` stream
            // is the `u16` stream narrowed value for value --- including codes
            // past the table, which both spellings reduce modulo it (C6).
            let wide: Vec<u16> = fill(n * k, 0xa4ea, |x| (x % 17) as u16);
            let narrow: Vec<u8> = wide.iter().map(|&c| c as u8).collect();
            let a: Vec<f32> = fill(m * k, 0xac7, |x| (x % 7) as f32 * 0.5 - 1.5);
            let zeros = vec![0.0f32; m * n];
            for traversal in [
                Traversal::Tabulated,
                Traversal::Blocked,
                Traversal::OutputMajor,
            ] {
                for offer in [0, OFFER_STEPS] {
                    let (by_word, _) = arena_tabulated(
                        t32,
                        &wide,
                        &a,
                        m,
                        k,
                        n,
                        traversal,
                        offer,
                        0,
                        &Linear::OVERWRITE,
                        &zeros,
                    );
                    let (by_byte, _) = arena_tabulated(
                        t32,
                        &narrow,
                        &a,
                        m,
                        k,
                        n,
                        traversal,
                        offer,
                        0,
                        &Linear::OVERWRITE,
                        &zeros,
                    );
                    let by_word: Vec<u64> = by_word.iter().map(|v| v.symbol_bits()).collect();
                    let by_byte: Vec<u64> = by_byte.iter().map(|v| v.symbol_bits()).collect();
                    assert_eq!(
                        by_byte, by_word,
                        "{m}x{k}x{n}: {traversal:?} at an offer of {offer}: the two widths of one \
                         codebook must give the same bytes"
                    );
                }
            }
        }
    }

    /// `CD-18`: a `u8`-symbol-coded float weight matrix gives the dense float
    /// driver's bytes at every shape, with the tabulated traversal forced and
    /// declined alike, at every offer including none, and under an epilogue
    /// that reads `C`.
    ///
    /// The reference is `gemm_float` over the decoded weights, as `CD-14`'s:
    /// an agreement between two coded traversals would say nothing about
    /// whether either computes the product.
    #[test]
    fn a_u8_symbol_coded_float_matrix_matches_the_dense_driver_cd_18() {
        // The same pool as `CD-14`: an infinity and a NaN are codes like any
        // other, and the two zeros are distinct symbols with equal dyadic values.
        let mut pool32 = [0.5f32, 1.0, f32::INFINITY, f32::NAN, -0.0, -1.5, -2.5, 0.0];
        assert_eq!(canonicalize(&mut pool32), 8, "eight distinct bit patterns");
        let mut pool64 = [0.5f64, 1.0, f64::INFINITY, f64::NAN, -0.0, -1.5, -2.5, 0.0];
        assert_eq!(canonicalize(&mut pool64), 8, "eight distinct bit patterns");

        macro_rules! sweep {
            ($d:literal) => {{
                let t32: &[Alphabet<f32, Whole<f32>>; $d] =
                    as_alphabet_whole(&pool32[..$d]).try_into().unwrap();
                let t64: &[Alphabet<f64, Whole<f64>>; $d] =
                    as_alphabet_whole(&pool64[..$d]).try_into().unwrap();
                for &(m, k, n) in &[
                    (1usize, 1usize, 1usize),
                    (2, 3, 5),
                    (5, 17, 7),
                    (13, 11, 3),
                    (7, 40, 9),
                ] {
                    // Codes past the table on purpose, as `CD-14`'s: the
                    // enumeration reduces them modulo `D`, and the reference
                    // decodes them the same way.
                    let codes: Vec<u8> = fill(n * k, 0xa4ea, |x| (x % (2 * $d as u64 + 1)) as u8);
                    let a32: Vec<f32> = fill(m * k, 0xac7, |x| (x % 7) as f32 * 0.5 - 1.5);
                    let a64: Vec<f64> = fill(m * k, 0xac7, |x| (x % 7) as f64 * 0.5 - 1.5);
                    every_arena_traversal_agrees(
                        concat!("Arena<", $d, ", u8> f32"),
                        t32,
                        &codes,
                        &a32,
                        m,
                        k,
                        n,
                        &Linear::OVERWRITE,
                        &vec![0.0f32; m * n],
                        f32_demand_table_expected(t32, &codes, &a32, m, k, n),
                    );
                    every_arena_traversal_agrees(
                        concat!("Arena<", $d, ", u8> f64"),
                        t64,
                        &codes,
                        &a64,
                        m,
                        k,
                        n,
                        &Linear::OVERWRITE,
                        &vec![0.0f64; m * n],
                        // The complete lane is resident and executable under
                        // the forced traversal at either code width.
                        true,
                    );
                }
            }};
        }
        sweep!(2);
        sweep!(3);
        sweep!(4);
        sweep!(5);
        sweep!(6);
        sweep!(7);
        sweep!(8);

        // An epilogue that reads `C`: two rows with equal rows of `A` have
        // different priors, so the row collapse is declined outright, and the
        // bytes are still the dense driver's. The offer sweep is the helper's
        // own; two tables stand for the range.
        let t32: &[Alphabet<f32, Whole<f32>>; 8] = as_alphabet_whole(&pool32).try_into().unwrap();
        for &(m, k, n) in &[(2usize, 3usize, 5usize), (7, 40, 9)] {
            let codes: Vec<u8> = fill(n * k, 0xa4ea, |x| (x % 17) as u8);
            let a: Vec<f32> = fill(m * k, 0xac7, |x| (x % 7) as f32 * 0.5 - 1.5);
            // A prior worth reading: not zeros, so a decline that skipped the
            // read would write different bytes.
            let c0: Vec<f32> = fill(m * n, 0xc01, |x| (x % 5) as f32 * 0.25 - 0.5);
            every_arena_traversal_agrees(
                "Arena<8, u8> f32 accumulate",
                t32,
                &codes,
                &a,
                m,
                k,
                n,
                &Linear::ACCUMULATE,
                &c0,
                f32_demand_table_expected(t32, &codes, &a, m, k, n),
            );
        }

        // The tier's design point: a codebook that fills the byte. The
        // codebook's symbols are an exact dequantization grid, so every
        // pattern is distinct by construction --- and `canonicalize` is asked
        // rather than told. Its seven-binade extent exercises nontrivial q
        // grades without assigning an admission limit to their range.
        let mut pool256: Vec<f32> = (0..256).map(|q| (q as f32 - 127.5) * 0.015_625).collect();
        assert_eq!(canonicalize(&mut pool256), 256, "256 distinct bit patterns");
        let t256: &[Alphabet<f32, Whole<f32>>; 256] =
            as_alphabet_whole(&pool256).try_into().unwrap();

        // Over the 88-byte complete accumulator a 256-entry table holds no L1
        // slab at any tile --- `tabulation_fits` says so at a single row ---
        // and that was once the end of the story. The compact lane (`CD-20`)
        // resolves the lane question: at eight bytes a word the slab is 2 KiB
        // a row and the table fits, so the forced traversal *tabulates*, and it is
        // again
        // the census that says so, not the predicate trusted twice.
        let wide = core::mem::size_of::<AccOf<f32>>();
        let narrow = core::mem::size_of::<i64>();
        assert!(
            !tabulation_fits(256, 1, blocking::L1_BYTES, wide),
            "a 256-entry table over a {wide}-byte lane fits no L1 tile"
        );
        assert!(
            tabulation_fits(256, 1, blocking::L1_BYTES, narrow),
            "a 256-entry table over an {narrow}-byte lane holds at one row"
        );

        for &(m, k, n) in &[(1usize, 1usize, 1usize), (5, 17, 7), (7, 40, 9)] {
            // Every byte value is a live code: the enumeration is total on
            // `u8`, so the fill needs no reduction (`CT-07`).
            let codes: Vec<u8> = fill(n * k, 0xa4ea, |x| x as u8);
            let a: Vec<f32> = fill(m * k, 0xac7, |x| (x % 7) as f32 * 0.5 - 1.5);
            let zeros = vec![0.0f32; m * n];
            let want: Vec<u64> =
                arena_reference(t256, &codes, &a, m, k, n, &Linear::OVERWRITE, &zeros)
                    .iter()
                    .map(|v| v.symbol_bits())
                    .collect();
            for traversal in [
                Traversal::Tabulated,
                Traversal::Blocked,
                Traversal::OutputMajor,
            ] {
                for offer in [0, 1, OFFER_STEPS - 1, OFFER_STEPS, OFFER_STEPS + 2] {
                    let (got, census) = arena_tabulated(
                        t256,
                        &codes,
                        &a,
                        m,
                        k,
                        n,
                        traversal,
                        offer,
                        0,
                        &Linear::OVERWRITE,
                        &zeros,
                    );
                    let got: Vec<u64> = got.iter().map(|v| v.symbol_bits()).collect();
                    assert_eq!(
                        got, want,
                        "Arena<256, u8> f32 {m}x{k}x{n}: {traversal:?} at an offer of {offer} \
                         must give the dense float driver's bytes ({census:?})"
                    );
                }
            }
            // The forced traversal tabulates because the narrow q slab fits.
            // The decodes are the extent observation (`m * k` of `A`, plus
            // the addressed book), projection, and codec presentations,
            // counted so their price is read off the census and not re-derived.
            let (_, tabled) = arena_tabulated(
                t256,
                &codes,
                &a,
                m,
                k,
                n,
                Traversal::Tabulated,
                OFFER_STEPS,
                0,
                &Linear::OVERWRITE,
                &zeros,
            );
            assert!(
                tabled.table_reads > 0,
                "the narrow lane fits, so the forced traversal reads a table ({tabled:?})"
            );
            let addressed_book = f32_demand_book_indices::<256, _>(&codes, m, k, n).len();
            assert_eq!(
                tabled.decodes,
                (2 * m * k + 3 * addressed_book) as u64,
                "the activation extent/projection plus the addressed-symbol extent, codec decode, \
                 and contextual projection are the decodes ({tabled:?})"
            );
            // CG-16 found no geometry-invariant scalar crossover for this
            // contextual block-one body, so the shape-only cost predicate does
            // not promote one from timing. At no offer the decoded walk feeds
            // the same persistent Atlas StreamLane.
            let (_, declined) = arena_tabulated(
                t256,
                &codes,
                &a,
                m,
                k,
                n,
                Traversal::Blocked,
                OFFER_STEPS,
                0,
                &Linear::OVERWRITE,
                &zeros,
            );
            assert_eq!(
                declined.table_reads, 0,
                "no universal block-one clock boundary may select this shape"
            );
            assert!(
                declined.kernel_calls > 0,
                "a full offer declines to the dense route ({declined:?})"
            );
            let (_, streamed) = arena_tabulated(
                t256,
                &codes,
                &a,
                m,
                k,
                n,
                Traversal::Tabulated,
                0,
                0,
                &Linear::OVERWRITE,
                &zeros,
            );
            assert_eq!(streamed.table_reads, 0);
            assert!(streamed.kernel_calls > 0);
            assert_eq!(
                streamed.multiplies, 0,
                "an offer of nothing remains lookup/add Atlas ({streamed:?})"
            );
        }
    }

    /// `CD-20`: the compact f32 protocol is contextual but compositional.
    /// `prescale` writes one q cell into the existing panel word and
    /// `Scaled64::mac` consumes precisely that spelling; placing the lane at
    /// the two gauges' sum is the same complete Laurent product as the element
    /// reference. The q contraction's variable occupied work is verified by
    /// its dedicated observer, while this generic boundary reports one
    /// contraction presentation for every table product.
    #[test]
    fn f32_q_projection_and_scaled64_compose_as_one_atlas_protocol_cd_20() {
        assert_eq!(
            f32_q_build_presentations(1, 1, 1),
            1,
            "one table product declares one opaque q contraction presentation"
        );
        assert_eq!(
            f32_q_build_presentations(3, 5, 7),
            105,
            "the declaration is parametric in every table-product axis"
        );
        let values = [
            0.0f32,
            -0.0,
            1.0,
            -1.0,
            f32::from_bits(0x3fff_ffff),
            f32::from_bits(0xbfff_ffff),
        ];

        for &a in &values {
            for &w in &values {
                let a_code = a.pack();
                let w_code = w.pack();
                let base_a = if a_code.mantissa == 0 {
                    0
                } else {
                    a_code.exp - 7
                };
                let base_b = if w_code.mantissa == 0 {
                    0
                } else {
                    w_code.exp - 7
                };
                let projected_a = <f32 as Tabulated>::prescale(a, base_a);
                let projected_w = <f32 as Tabulated>::prescale(w, base_b);
                let lane = <Scaled64 as Lane<f32>>::mac(Scaled64(0), projected_a, projected_w);
                let got = <Scaled64 as Lane<f32>>::place_scaled(
                    lane,
                    <AccOf<f32> as Accumulator>::ZERO,
                    base_a + base_b,
                );
                let mut want = <AccOf<f32> as Accumulator>::ZERO;
                <f32 as Element>::mac(&mut want, a, w);
                assert_eq!(got, want, "contextual carrier pair ({a:?}, {w:?})");
            }
        }

        let symbols = [-1.0f32, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0];
        let table: &[Alphabet<f32, Whole<f32>>; 8] =
            as_alphabet_whole(&symbols).try_into().unwrap();
        let (m, k, n) = (2usize, 3usize, 5usize);
        let codes: Vec<u8> = (0..n * k).map(|at| (at % symbols.len()) as u8).collect();
        let activations = [0.5f32, -1.25, 2.0, -0.75, 1.5, 0.25];
        let (_, census) = arena_tabulated(
            table,
            &codes,
            &activations,
            m,
            k,
            n,
            Traversal::Tabulated,
            OFFER_STEPS,
            0,
            &Linear::OVERWRITE,
            &vec![0.0; m * n],
        );
        let build_products = n * m * k;
        let gathers = m * k * n;
        assert_eq!(census.multiplies, 0);
        assert_eq!(census.table_reads, gathers as u64);
        assert_eq!(
            census.decodes,
            (2 * m * k + 3 * symbols.len()) as u64,
            "the span observation and contextual projection are each counted; the stored stream \
             is longer than the enumeration, so each book walk remains full"
        );
        assert_eq!(
            census.adds,
            (build_products * f32_q_build_presentations(1, 1, 1) as usize + gathers) as u64,
            "the block-one build charges only the entries this column block addresses, plus \
             the gather"
        );
    }

    /// `CG-16`: sparse block-one work is proportional to symbols the call
    /// addresses, not to an unused enumeration. The one stored code drives the
    /// scale walk, panel projection, in-place entry build, and gather exactly
    /// once; no bitmap, copied carrier, or full 256-entry pass remains.
    #[test]
    fn block_one_builds_only_the_addressed_atlas_entries_cg_16() {
        let symbols: [f32; 256] = core::array::from_fn(|index| 0.5 + index as f32 / 1024.0);
        let table: &[Alphabet<f32, Whole<f32>>; 256] =
            as_alphabet_whole(&symbols).try_into().unwrap();
        let codes = [231u8];
        let activations = [0.75f32];
        let zeros = [0.0f32];
        let (got, census) = arena_tabulated(
            table,
            &codes,
            &activations,
            1,
            1,
            1,
            Traversal::Tabulated,
            OFFER_STEPS,
            0,
            &Linear::OVERWRITE,
            &zeros,
        );
        let want = arena_reference(
            table,
            &codes,
            &activations,
            1,
            1,
            1,
            &Linear::OVERWRITE,
            &zeros,
        );
        assert_eq!(got[0].symbol_bits(), want[0].symbol_bits());
        assert_eq!(census.kernel_calls, 0);
        assert_eq!(census.multiplies, 0);
        assert_eq!(census.table_reads, 1);
        assert_eq!(census.decodes, 5, "one span observation and one contextual projection for each activation and addressed symbol, plus the symbol's codec decode");
        assert_eq!(
            census.adds,
            1 + f32_q_build_presentations(1, 1, 1) + 1,
            "one call-wide scale certificate, one fixed Atlas entry contraction, and one \
             gather combine"
        );
    }

    /// `CG-16`: an offered index dictionary makes demand a function of
    /// distinct addressed indices, never of the raw stream length. The same
    /// one symbol is stored just below, at, and above the eight-entry book
    /// extent, and every shape spans several two-column blocks. Scale, decoded
    /// book, table construction, and gathers therefore remain one presentation
    /// per semantic use when crossing the old `codes.len() < CODE_SPACE`
    /// boundary.
    #[test]
    fn repeated_block_one_symbols_are_built_once_per_addressed_index_cg_16() {
        const D: usize = 8;
        let symbols: [f32; D] = core::array::from_fn(|index| 0.5 + index as f32 / 32.0);
        let table: &[Alphabet<f32, Whole<f32>>; D] =
            as_alphabet_whole(&symbols).try_into().unwrap();
        let activations = [0.75f32];

        for n in [D - 1, D, D + 1] {
            let shape = Shape { m: 1, k: 1, n };
            let codes = vec![3u8; n];
            let mut output = vec![0.0f32; n];
            let mut exact = vec![<AccOf<f32> as Accumulator>::ZERO; 2];
            let mut lanes = vec![0i64; suggested_tabulation_lanes::<f32, Whole<f32>>(shape, D, 1)];
            let mut index = vec![0usize; suggested_tabulation_index(shape)];
            let mut panel =
                vec![Alphabet::<f32, Whole<f32>>::ZERO; suggested_tabulation_panel(D, 1)];
            let mut census = Census::default();
            let a = MatView::row_major(as_alphabet_whole(&activations), 1, 1).unwrap();
            let c = MatViewMut::row_major(&mut output, 1, n).unwrap();
            let w = CodedMatrix::new(Arena::new(table), n, 1, &codes).unwrap();
            let mut triple = TabulatedTriple::new(a, w, c).unwrap();
            gemm_tabulated_counted(
                &mut triple,
                &Linear::OVERWRITE,
                GemmOptions {
                    traversal: Traversal::Tabulated,
                    ..Default::default()
                },
                &mut Scratch::with_accumulators(&mut panel, &mut exact),
                &mut Tabulation::with_index(&mut lanes, &mut index),
                &mut Collapse::none(),
                &mut census,
            );

            let want = arena_reference(
                table,
                &codes,
                &activations,
                1,
                1,
                n,
                &Linear::OVERWRITE,
                &vec![0.0; n],
            );
            assert_eq!(
                output
                    .iter()
                    .map(|value| value.symbol_bits())
                    .collect::<Vec<_>>(),
                want.iter()
                    .map(|value| value.symbol_bits())
                    .collect::<Vec<_>>()
            );
            let column_blocks = n.div_ceil(2);
            let contractions = column_blocks as u64;
            assert_eq!(census.kernel_calls, 0);
            assert_eq!(census.multiplies, 0);
            assert_eq!(census.table_reads, contractions);
            assert_eq!(
                census.adds,
                1 + contractions * (f32_q_build_presentations(1, 1, 1) + 1),
                "n={n}: one call-wide scale certificate, then one build and gather per column \
                 block"
            );
            assert_eq!(
                census.decodes,
                column_blocks as u64 + 4,
                "n={n}: one A/span observation, one addressed-book span, one codec decode, one \
                 book projection, and one A projection per column block"
            );
        }
    }

    /// `CG-16`: entry deduplication is independent of whole-column collapse.
    /// Every column below is globally distinct (`[3, j]`), while reduction
    /// position zero addresses index three five times. The table builds that
    /// slot once, builds the five distinct indices at position one, and still
    /// gathers both positions for all five columns.
    #[test]
    fn shared_slot_indices_are_deduplicated_without_collapsing_columns_cg_16() {
        const D: usize = 8;
        let symbols: [f32; D] = core::array::from_fn(|index| 0.5 + index as f32 / 32.0);
        let table: &[Alphabet<f32, Whole<f32>>; D] =
            as_alphabet_whole(&symbols).try_into().unwrap();
        let (m, k, n) = (1usize, 2usize, 5usize);
        let codes: Vec<u8> = (0..n).flat_map(|column| [3u8, column as u8]).collect();
        let activations = [0.75f32, -0.5];
        let shape = Shape { m, k, n };
        let mut output = vec![0.0f32; n];
        let mut exact = vec![<AccOf<f32> as Accumulator>::ZERO; n];
        let mut lanes = vec![0i64; suggested_tabulation_lanes::<f32, Whole<f32>>(shape, D, 1)];
        let mut index = vec![0usize; suggested_tabulation_index(shape)];
        let mut panel = vec![Alphabet::<f32, Whole<f32>>::ZERO; suggested_tabulation_panel(D, 1)];
        let mut census = Census::default();
        let a = MatView::row_major(as_alphabet_whole(&activations), m, k).unwrap();
        let c = MatViewMut::row_major(&mut output, m, n).unwrap();
        let w = CodedMatrix::new(Arena::new(table), n, k, &codes).unwrap();
        let mut triple = TabulatedTriple::new(a, w, c).unwrap();
        gemm_tabulated_counted(
            &mut triple,
            &Linear::OVERWRITE,
            GemmOptions {
                traversal: Traversal::Tabulated,
                ..Default::default()
            },
            &mut Scratch::with_accumulators(&mut panel, &mut exact),
            &mut Tabulation::with_index(&mut lanes, &mut index),
            &mut Collapse::none(),
            &mut census,
        );

        let want = arena_reference(
            table,
            &codes,
            &activations,
            m,
            k,
            n,
            &Linear::OVERWRITE,
            &vec![0.0; n],
        );
        assert_eq!(
            output
                .iter()
                .map(|value| value.symbol_bits())
                .collect::<Vec<_>>(),
            want.iter()
                .map(|value| value.symbol_bits())
                .collect::<Vec<_>>()
        );
        let builds = 1 + n as u64;
        let gathers = (k * n) as u64;
        assert_eq!(census.kernel_calls, 0);
        assert_eq!(census.multiplies, 0);
        assert_eq!(census.table_reads, gathers);
        assert_eq!(
            census.adds,
            builds * f32_q_build_presentations(1, 1, 1) + gathers
        );
        assert_eq!(
            census.decodes, 19,
            "two A and five distinct-book span observations, five codec decodes, five book \
             projections, and two activation projections"
        );
    }

    /// `CG-16`: the reclaimed dictionary is cleared for an addressed-entry set
    /// only when that set can remove work. A block wider than one, a sign-book,
    /// and the one-coordinate enumeration all pass `need_entries = false`; the
    /// column map may still be derived, but untouched dictionary cells must not
    /// pay a table-wide clear.
    #[test]
    fn non_pointwise_books_do_not_construct_an_unused_entry_set_cg_16() {
        let (n, cols) = (3usize, 2usize);
        let table = (2 * n).next_power_of_two();
        let words = n + 2 * table;
        let marker = usize::MAX - 7;

        let mut distinct = vec![marker; words];
        let snapshot = distinct.clone();
        let (map, entries) = column_workspace(&mut distinct, n, cols, false, false);
        assert!(map.is_none() && entries.is_none());
        assert_eq!(
            distinct, snapshot,
            "no-repeat/non-entry work touches no dictionary word"
        );

        let mut repeated = vec![marker; words];
        repeated[..n].copy_from_slice(&[0, 0, 2]);
        let untouched_probe = n + table - 1;
        let (map, entries) = column_workspace(&mut repeated, n, cols, true, false);
        assert!(map.is_some() && entries.is_none());
        assert_eq!(
            repeated[untouched_probe], marker,
            "column-map derivation must not end with the unused EntrySet clear"
        );
    }

    /// `CU-11`: the odd Atlas-modality radix is invertible modulo every
    /// power-of-two dictionary extent. An earliest-site unit difference
    /// therefore survives arbitrarily many following zero coordinates, unlike
    /// the discarded even recurrence. Equality remains the authority on a
    /// deliberately colliding canonical pair and on repeated columns.
    #[test]
    fn ternary_radix_hash_retains_early_sites_and_collisions_are_exact_cu_11() {
        type A32 = Arena<'static, f32, 32, u8>;
        let formerly_reduced = |run: &[u8], modulus: usize| {
            let mut hash = (run.len() % modulus) as u128;
            for &code in &run[..run.len().min(crate::float::COLUMN_HASH_PREFIX)] {
                let doubled = hash + hash;
                hash =
                    doubled + hash + <A32 as Enumerable<f32, Whole<f32>>>::index_of(code) as u128;
            }
            (hash % modulus as u128) as usize
        };
        for length in [0usize, 1, 15, 16, 17, 64, 256] {
            let run = (0..length)
                .map(|at| u8::try_from((at * 17 + length * 29) % 32).unwrap())
                .collect::<Vec<_>>();
            for table in [2usize, 4, 8, 16, 1024] {
                assert_eq!(
                    column_hash::<f32, Whole<f32>, A32>(&run, table),
                    formerly_reduced(&run, table),
                    "removing the initial remainder changed length {length} modulo {table}"
                );
            }
        }
        for table in [2usize, 4, 8, 16] {
            let left = [0u8; 17];
            let mut right = left;
            right[0] = 1;
            assert_ne!(
                column_hash::<f32, Whole<f32>, A32>(&left, table),
                column_hash::<f32, Whole<f32>, A32>(&right, table),
                "the earliest unit coordinate survives modulo {table}"
            );
        }

        // Indices 0 and 8 have the same one-coordinate residue modulo the
        // eight-slot dictionary, so only the independent equality authority
        // can keep them distinct while collapsing their repeats.
        let codes = [0u8, 0, 8, 8];
        let n = codes.len();
        let table = (2 * n).next_power_of_two();
        let mut index = vec![0usize; n + 2 * table];
        assert_eq!(
            distinct_columns::<f32, Whole<f32>, A32>(&codes, 1, n, &mut index),
            Some(2)
        );
        assert_eq!(&index[..n], &[0, 0, 2, 2]);

        // The measured prefix deliberately does not inspect coordinate 16.
        // That is a work boundary, never an equality boundary: a tail-only
        // difference collides in the filter and is still separated exactly,
        // while repeats of both columns collapse to their own first source.
        let head = [0u8; 17];
        let mut tail = head;
        tail[16] = 1;
        assert_eq!(
            column_hash::<f32, Whole<f32>, A32>(&head, 8),
            column_hash::<f32, Whole<f32>, A32>(&tail, 8)
        );
        let mut codes = Vec::new();
        for column in [&head, &head, &tail, &tail] {
            codes.extend_from_slice(column);
        }
        let n = 4usize;
        let table = (2 * n).next_power_of_two();
        let mut index = vec![0usize; n + 2 * table];
        assert_eq!(
            distinct_columns::<f32, Whole<f32>, A32>(&codes, head.len(), n, &mut index),
            Some(2)
        );
        assert_eq!(&index[..n], &[0, 0, 2, 2]);
    }

    /// Release-only retained clock for the hash and the complete collapse pass.
    /// It compares the exact same corpus and verifies the first-occurrence map
    /// before looking at time. This is intentionally not CG-16 calibration: it
    /// is a regression control for removing legacy bitwise address work.
    #[test]
    #[ignore = "release-only radix/legacy collapse clock"]
    fn ternary_radix_column_collapse_does_not_regress_the_retained_legacy_clock_cu_11() {
        type A251 = Arena<'static, f32, 251, u8>;
        let n = 257usize;
        for depth in [1usize, 16, 64, 256] {
            let mut codes = vec![0u8; n * depth];
            for j in 0..n {
                let representative = if j % 5 == 1 { j - 1 } else { j };
                for p in 0..depth {
                    codes[j * depth + p] =
                        u8::try_from((representative * 29 + p * 17) % 251).unwrap();
                }
            }
            let table = (2 * n).next_power_of_two();
            let mut radix_index = vec![0usize; n + 2 * table];
            let mut legacy_index = vec![0usize; n + 2 * table];
            let radix =
                distinct_columns::<f32, Whole<f32>, A251>(&codes, depth, n, &mut radix_index);
            let legacy = legacy_distinct_columns::<f32, Whole<f32>, A251>(
                &codes,
                depth,
                n,
                &mut legacy_index,
            );
            assert_eq!(radix, legacy);
            assert_eq!(&radix_index[..n], &legacy_index[..n]);

            // Each pair sees the same immutable corpus and one common batch.
            // Short alternating chunks keep frequency and interrupt drift common
            // to both arms even when equality work makes a full batch long. The
            // batch still doubles until even its faster arm occupies at least
            // twenty milliseconds, keeping timer quantization below the work
            // being compared. Poisoning and complete map validation surround
            // every paired sample rather than entering either timer. The guard
            // is the upper endpoint of a conservative paired 95% interval, not
            // two aggregate clocks.
            const PAIRED_SAMPLES: usize = 64;
            const PAIRED_CHUNK_CALLS: usize = 32;
            const MIN_BATCH_TIME: std::time::Duration = std::time::Duration::from_millis(20);
            let poison = usize::MAX - 13;
            let expected_distinct = radix;
            let expected_map = radix_index[..n].to_vec();
            let mut radix_samples = Vec::with_capacity(PAIRED_SAMPLES);
            let mut legacy_samples = Vec::with_capacity(PAIRED_SAMPLES);
            let measure_radix = |index: &mut [usize], batch: usize| {
                index.fill(poison);
                let mut observed = expected_distinct;
                let start = std::time::Instant::now();
                for _ in 0..batch {
                    observed = std::hint::black_box(distinct_columns::<f32, Whole<f32>, A251>(
                        std::hint::black_box(&codes),
                        depth,
                        n,
                        std::hint::black_box(&mut *index),
                    ));
                }
                let elapsed = start.elapsed();
                assert_eq!(observed, expected_distinct);
                assert_eq!(&index[..n], expected_map.as_slice());
                elapsed
            };
            let measure_legacy = |index: &mut [usize], batch: usize| {
                index.fill(poison);
                let mut observed = expected_distinct;
                let start = std::time::Instant::now();
                for _ in 0..batch {
                    observed =
                        std::hint::black_box(legacy_distinct_columns::<f32, Whole<f32>, A251>(
                            std::hint::black_box(&codes),
                            depth,
                            n,
                            std::hint::black_box(&mut *index),
                        ));
                }
                let elapsed = start.elapsed();
                assert_eq!(observed, expected_distinct);
                assert_eq!(&index[..n], expected_map.as_slice());
                elapsed
            };
            let run_radix = |index: &mut [usize], calls: usize| {
                let mut observed = expected_distinct;
                let start = std::time::Instant::now();
                for _ in 0..calls {
                    observed = std::hint::black_box(distinct_columns::<f32, Whole<f32>, A251>(
                        std::hint::black_box(&codes),
                        depth,
                        n,
                        std::hint::black_box(&mut *index),
                    ));
                }
                (observed, start.elapsed())
            };
            let run_legacy = |index: &mut [usize], calls: usize| {
                let mut observed = expected_distinct;
                let start = std::time::Instant::now();
                for _ in 0..calls {
                    observed =
                        std::hint::black_box(legacy_distinct_columns::<f32, Whole<f32>, A251>(
                            std::hint::black_box(&codes),
                            depth,
                            n,
                            std::hint::black_box(&mut *index),
                        ));
                }
                (observed, start.elapsed())
            };

            let mut batch = 1usize;
            loop {
                let radix_batch = measure_radix(&mut radix_index, batch);
                let legacy_batch = measure_legacy(&mut legacy_index, batch);
                if radix_batch.min(legacy_batch) >= MIN_BATCH_TIME {
                    break;
                }
                batch = batch
                    .checked_mul(2)
                    .expect("an address-sized batch reaches a twenty-millisecond clock interval");
            }
            for sample in 0..PAIRED_SAMPLES {
                radix_index.fill(poison);
                legacy_index.fill(poison);
                let mut radix_observed = expected_distinct;
                let mut legacy_observed = expected_distinct;
                let mut radix_elapsed = std::time::Duration::default();
                let mut legacy_elapsed = std::time::Duration::default();
                let mut completed = 0usize;
                let mut chunk = 0usize;
                while completed < batch {
                    let calls = (batch - completed).min(PAIRED_CHUNK_CALLS);
                    let radix_first = (sample + chunk).is_multiple_of(2);
                    if radix_first {
                        let (observed, elapsed) = run_radix(&mut radix_index, calls);
                        radix_observed = observed;
                        radix_elapsed += elapsed;
                        let (observed, elapsed) = run_legacy(&mut legacy_index, calls);
                        legacy_observed = observed;
                        legacy_elapsed += elapsed;
                    } else {
                        let (observed, elapsed) = run_legacy(&mut legacy_index, calls);
                        legacy_observed = observed;
                        legacy_elapsed += elapsed;
                        let (observed, elapsed) = run_radix(&mut radix_index, calls);
                        radix_observed = observed;
                        radix_elapsed += elapsed;
                    }
                    completed += calls;
                    chunk += 1;
                }
                assert_eq!(radix_observed, expected_distinct);
                assert_eq!(legacy_observed, expected_distinct);
                assert_eq!(&radix_index[..n], expected_map.as_slice());
                assert_eq!(&legacy_index[..n], expected_map.as_slice());
                radix_samples.push(radix_elapsed);
                legacy_samples.push(legacy_elapsed);
            }

            let log_ratios = radix_samples
                .iter()
                .zip(&legacy_samples)
                .map(|(radix, legacy)| {
                    let radix = radix.as_nanos().max(1) as f64;
                    let legacy = legacy.as_nanos().max(1) as f64;
                    (radix / legacy).ln()
                })
                .collect::<Vec<_>>();
            let count = log_ratios.len() as f64;
            let mean_log = log_ratios.iter().sum::<f64>() / count;
            let variance = log_ratios
                .iter()
                .map(|ratio| {
                    let distance = ratio - mean_log;
                    distance * distance
                })
                .sum::<f64>()
                / (count - 1.0);
            // With 63 degrees of freedom, two standard errors is a
            // conservative 95% Student interval (t_0.975,63 < 2).
            let margin = 2.0 * (variance / count).sqrt();
            let ratio = mean_log.exp();
            let lower_95 = (mean_log - margin).exp();
            let upper_95 = (mean_log + margin).exp();
            std::eprintln!(
                "column-collapse depth={depth}: paired_ratio={ratio:.4}, 95% CI=[{lower_95:.4}, {upper_95:.4}], samples={PAIRED_SAMPLES}, batch={batch}"
            );
            assert!(
                upper_95 <= 1.0,
                "pure radix collapse regressed at depth {depth}: paired ratio={ratio:.4}, 95% CI=[{lower_95:.4}, {upper_95:.4}]"
            );
        }
    }

    /// `CG-16`: a failed call-wide collection clears every occupied probe and
    /// leaves the same offered words reusable by the per-slot set. Canonical
    /// indices that share low address bits must probe onward, be recognized on
    /// a second insertion, and all clear again without a bitmap or allocation.
    #[test]
    fn addressed_entry_set_overflow_collision_and_reuse_are_exact_cg_16() {
        let mut seen = [0usize; 4];
        let mut occupied = [0usize; 2];
        let mut set = EntrySet {
            seen: &mut seen,
            occupied: &mut occupied,
            used: 0,
        };
        assert!(
            set.collect::<f32, Whole<f32>, Arena<'static, f32, 8, u8>>(&[0, 1, 2])
                .is_none(),
            "the third distinct coordinate exceeds the offered occupied run"
        );
        assert!(set.seen.iter().all(|&entry| entry == 0));
        let count = set
            .collect::<f32, Whole<f32>, Arena<'static, f32, 8, u8>>(&[3, 3])
            .expect("the cleared set is immediately reusable");
        assert_eq!(set.collected(count), &[3]);
        set.release_collected();

        let mut colliding_seen = [0usize; 8];
        let mut colliding_occupied = [0usize; 4];
        let mut colliding = EntrySet {
            seen: &mut colliding_seen,
            occupied: &mut colliding_occupied,
            used: 0,
        };
        assert!(matches!(colliding.insert(1), EntryInsert::New));
        assert!(matches!(colliding.insert(9), EntryInsert::New));
        assert!(matches!(colliding.insert(17), EntryInsert::New));
        assert!(matches!(colliding.insert(9), EntryInsert::Present));
        assert_eq!(
            (0..colliding.len())
                .map(|at| colliding.index(at))
                .collect::<Vec<_>>(),
            [1, 9, 17]
        );
        colliding.clear();
        assert!(colliding.seen.iter().all(|&entry| entry == 0));
    }

    /// `CG-16`: without enough index storage, generic duplicate coordinates
    /// remain issued presentations rather than becoming a guessed type-specific
    /// set. The absent and one-word offers therefore build/read four entries;
    /// the complete offer names the one semantic column/address once. All three
    /// are byte-identical to the dense product.
    #[test]
    fn short_index_offers_keep_duplicate_entry_work_truthful_cg_16() {
        const D: usize = 8;
        let symbols: [f32; D] = core::array::from_fn(|index| 0.5 + index as f32 / 32.0);
        let table: &[Alphabet<f32, Whole<f32>>; D] =
            as_alphabet_whole(&symbols).try_into().unwrap();
        let (m, k, n) = (1usize, 1usize, 4usize);
        let shape = Shape { m, k, n };
        let codes = [3u8; 4];
        let activations = [0.75f32];

        let run = |index_offer: Option<usize>| {
            let mut output = vec![0.0f32; n];
            let mut exact = vec![
                <AccOf<f32> as Accumulator>::ZERO;
                suggested_tabulation::<f32, Whole<f32>>(shape, D, 1)
            ];
            let mut lanes = vec![0i64; suggested_tabulation_lanes::<f32, Whole<f32>>(shape, D, 1)];
            let mut index = vec![0usize; index_offer.unwrap_or(0)];
            let mut panel =
                vec![Alphabet::<f32, Whole<f32>>::ZERO; suggested_tabulation_panel(D, 1)];
            let mut census = Census::default();
            let a = MatView::row_major(as_alphabet_whole(&activations), m, k).unwrap();
            let c = MatViewMut::row_major(&mut output, m, n).unwrap();
            let w = CodedMatrix::new(Arena::new(table), n, k, &codes).unwrap();
            let mut triple = TabulatedTriple::new(a, w, c).unwrap();
            let mut tabulation = match index_offer {
                Some(_) => Tabulation::with_index(&mut lanes, &mut index),
                None => Tabulation::new(&mut lanes),
            };
            gemm_tabulated_counted(
                &mut triple,
                &Linear::OVERWRITE,
                GemmOptions {
                    traversal: Traversal::Tabulated,
                    ..Default::default()
                },
                &mut Scratch::with_accumulators(&mut panel, &mut exact),
                &mut tabulation,
                &mut Collapse::none(),
                &mut census,
            );
            (output, census)
        };

        let (absent, absent_census) = run(None);
        let (short, short_census) = run(Some(1));
        let (complete, complete_census) = run(Some(suggested_tabulation_index(shape)));
        let want = arena_reference(
            table,
            &codes,
            &activations,
            m,
            k,
            n,
            &Linear::OVERWRITE,
            &[0.0; 4],
        );
        let bits = |values: &[f32]| {
            values
                .iter()
                .map(|value| value.symbol_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(&absent), bits(&want));
        assert_eq!(bits(&short), bits(&want));
        assert_eq!(bits(&complete), bits(&want));
        for census in [absent_census, short_census] {
            assert_eq!(census.table_reads, 4);
            assert_eq!(
                census.adds,
                1 + 4 * (f32_q_build_presentations(1, 1, 1) + 1)
            );
            assert_eq!(census.decodes, 14);
        }
        assert_eq!(complete_census.table_reads, 1);
        assert_eq!(
            complete_census.adds,
            1 + f32_q_build_presentations(1, 1, 1) + 1
        );
        assert_eq!(complete_census.decodes, 5);
    }

    /// `CG-16`: addressed coordinates are interpreted through the downstream
    /// enumeration's `code_at/index_of` pair. The canonical coordinates 0 and 1
    /// below decode stored codes 2 and 0, an admitted one-binade book; treating
    /// them as identity codes would instead include stored code 1 and falsely
    /// reject its sixteen-binade span.
    #[test]
    fn addressed_codec_preserves_a_nonidentity_enumeration_cg_16() {
        let symbols = [1.0f32, 65536.0, 2.0, 4.0];
        let table: &[Alphabet<f32, Whole<f32>>; 4] =
            as_alphabet_whole(&symbols).try_into().unwrap();
        let codes = [2u8, 0u8];
        let activations = [1.0f32, 1.0];
        let a = MatView::row_major(as_alphabet_whole(&activations), 1, 2).unwrap();
        let w = CodedMatrix::new(PermutedF32(table), 1, 2, &codes).unwrap();
        let mut census = Census::default();
        let scale = addressed_lane_scale(&a, &w, Some(&[0, 1]), &mut census)
            .expect("the canonical addressed book spans one binade");
        assert_eq!(
            scale.base_b,
            symbols[0].pack().exp.min(symbols[2].pack().exp)
        );
        assert_eq!(
            census.decodes, 4,
            "two activations and two canonical book coordinates"
        );
    }

    /// `CG-16`: caller-offered panel tail is resident projected-activation
    /// storage, not discarded capacity. A one-cell exact offer deliberately
    /// makes four column blocks; the fixed-sized panel therefore projects each
    /// activation four times, while a tail of exactly two complete rows projects
    /// it once. Both are the same forced Atlas table and produce the same bytes.
    #[test]
    fn offered_projected_a_rows_are_reused_across_column_blocks_cg_16() {
        const D: usize = 16;
        let symbols: [f32; D] = core::array::from_fn(|index| 0.5 + index as f32 / 32.0);
        let table: &[Alphabet<f32, Whole<f32>>; D] =
            as_alphabet_whole(&symbols).try_into().unwrap();
        let (m, k, n) = (2usize, 3usize, 4usize);
        let codes: Vec<u8> = (0..n * k).map(|at| at as u8).collect();
        let activations = [0.75f32, -0.5, 1.25, -1.0, 0.625, 1.5];
        let fixed_panel = suggested_tabulation_panel(D, 1);

        let run = |cache_cells: usize| {
            let mut output = vec![0.0f32; m * n];
            // One exact cell derives a one-row, one-column plan. The lane offer
            // holds that column plus every one-row slab of the reduction.
            let mut exact = [<AccOf<f32> as Accumulator>::ZERO; 1];
            let mut lanes = vec![0i64; 1 + table_words(D, 1, k)];
            let mut panel = vec![Alphabet::<f32, Whole<f32>>::ZERO; fixed_panel + cache_cells];
            let mut census = Census::default();
            let a = MatView::row_major(as_alphabet_whole(&activations), m, k).unwrap();
            let c = MatViewMut::row_major(&mut output, m, n).unwrap();
            let w = CodedMatrix::new(Arena::new(table), n, k, &codes).unwrap();
            let mut triple = TabulatedTriple::new(a, w, c).unwrap();
            gemm_tabulated_counted(
                &mut triple,
                &Linear::OVERWRITE,
                GemmOptions {
                    traversal: Traversal::Tabulated,
                    ..Default::default()
                },
                &mut Scratch::with_accumulators(&mut panel, &mut exact),
                &mut Tabulation::new(&mut lanes),
                &mut Collapse::none(),
                &mut census,
            );
            (output, census)
        };

        let (uncached, uncached_census) = run(0);
        let (cached, cached_census) = run(m * k);
        let reference = arena_reference(
            table,
            &codes,
            &activations,
            m,
            k,
            n,
            &Linear::OVERWRITE,
            &vec![0.0; m * n],
        );
        assert_eq!(
            cached
                .iter()
                .map(|value| value.symbol_bits())
                .collect::<Vec<_>>(),
            reference
                .iter()
                .map(|value| value.symbol_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            uncached
                .iter()
                .map(|value| value.symbol_bits())
                .collect::<Vec<_>>(),
            cached
                .iter()
                .map(|value| value.symbol_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(uncached_census.table_reads, (m * k * n) as u64);
        assert_eq!(cached_census.table_reads, uncached_census.table_reads);
        assert_eq!(
            uncached_census.decodes - cached_census.decodes,
            (m * k * (n - 1)) as u64,
            "the cache removes exactly the three repeated projections of every activation"
        );
        assert_eq!(cached_census.decodes, (2 * m * k + 3 * n * k) as u64);
    }

    /// `CD-20`: a `u8`-symbol-coded float weight matrix tabulated in the
    /// compact Atlas lane gives the dense Atlas driver's bytes at every shape
    /// and offer. Every finite extent and non-finite union remains table-
    /// executable; a source-ordered local schedule fractures an aggregate that
    /// exceeds Q, places singleton tags immediately, and contracts every
    /// resulting occupied q extent exactly.
    ///
    /// The codebook and activation tile receive their paired contextual bases.
    /// `Scaled64` contracts their q cells through the occupied centered-octet
    /// extents. Each maximal source-ordered group is gathered exactly, resolved,
    /// and placed at `2^(base_a + base_b)` before the next group. The reference is
    /// `gemm_float` over decoded weights, as in `CD-18`; agreement between two
    /// coded traversals alone would not establish the product.
    #[test]
    fn the_scaled_lane_tabulation_matches_the_dense_driver_cd_20() {
        // A codebook of 256 distinct finite symbols whose values span exactly
        // `span` binades: significands in `[2^22, 2^23)`, so every value's
        // exponent is `22 + e` and no two collapse, and `e` cycles through
        // `span + 1` values. The sweep exercises progressively wider q extents;
        // none is an alphabet or admission boundary.
        let codebook = |span: u64| {
            let mut pool: Vec<f32> = (0..256u64)
                .map(|q| {
                    let m = (0x40_0000 + q * 0x1000) as f32;
                    let e = if span == 0 {
                        0
                    } else {
                        (q % (span + 1)) as i32 - (span / 2) as i32
                    };
                    m * 2.0f32.powi(e)
                })
                .collect();
            assert_eq!(canonicalize(&mut pool), 256, "{span} binades, 256 patterns");
            pool
        };

        let shapes: &[(usize, usize, usize)] = &[
            (1, 1, 1),
            (2, 3, 5),
            (5, 17, 7),
            (13, 11, 3),
            (7, 40, 9),
            // Deep enough to require multiple source-ordered local groups at
            // the widest sampled extent. A lane that dropped a carry between
            // group placements would write different bytes here.
            (3, 300, 5),
        ];
        for span in [0u64, 3, 7] {
            let pool = codebook(span);
            let table: &[Alphabet<f32, Whole<f32>>; 256] =
                as_alphabet_whole(&pool).try_into().unwrap();
            for &(m, k, n) in shapes {
                // Every byte value is a live code (`CT-07`); `A` spans one
                // binade, and every combined extent remains table-executable.
                let codes: Vec<u8> = fill(n * k, 0xa4ea, |x| x as u8);
                let a: Vec<f32> = fill(m * k, 0xac7, |x| (x % 7) as f32 * 0.5 - 1.5);
                every_arena_traversal_agrees(
                    &format!("Arena<256, u8> f32 span {span}"),
                    table,
                    &codes,
                    &a,
                    m,
                    k,
                    n,
                    &Linear::OVERWRITE,
                    &vec![0.0f32; m * n],
                    f32_demand_table_expected(table, &codes, &a, m, k, n),
                );
            }
        }

        // A nine-binade extent and non-finite symbols exercise the cases the
        // former compact carrier refused. The q carrier retains the former as
        // locally scheduled finite grades and the latter as singleton tags;
        // all remain table-executable at the dense driver's bytes.
        let wide = codebook(9);
        let wide_t: &[Alphabet<f32, Whole<f32>>; 256] =
            as_alphabet_whole(&wide).try_into().unwrap();
        let mut nonfinite = codebook(3);
        nonfinite.truncate(255);
        nonfinite.push(f32::INFINITY);
        assert_eq!(
            canonicalize(&mut nonfinite),
            256,
            "255 grid points and an infinity"
        );
        let nonfinite_t: &[Alphabet<f32, Whole<f32>>; 256] =
            as_alphabet_whole(&nonfinite).try_into().unwrap();
        let grid3 = codebook(3);
        let grid3_t: &[Alphabet<f32, Whole<f32>>; 256] =
            as_alphabet_whole(&grid3).try_into().unwrap();
        for &(m, k, n) in &[(2usize, 3usize, 5usize), (13, 11, 3), (7, 40, 9)] {
            let codes: Vec<u8> = fill(n * k, 0xa4ea, |x| x as u8);
            let a: Vec<f32> = fill(m * k, 0xac7, |x| (x % 7) as f32 * 0.5 - 1.5);
            let zeros = vec![0.0f32; m * n];
            every_arena_traversal_agrees(
                "Arena<256, u8> f32 span 9",
                wide_t,
                &codes,
                &a,
                m,
                k,
                n,
                &Linear::OVERWRITE,
                &zeros,
                f32_demand_table_expected(wide_t, &codes, &a, m, k, n),
            );
            every_arena_traversal_agrees(
                "Arena<256, u8> f32 non-finite book",
                nonfinite_t,
                &codes,
                &a,
                m,
                k,
                n,
                &Linear::OVERWRITE,
                &zeros,
                f32_demand_table_expected(nonfinite_t, &codes, &a, m, k, n),
            );
            // One NaN in `A` becomes a singleton tag without declining the lane.
            let mut a_nan = a.clone();
            a_nan[0] = f32::NAN;
            every_arena_traversal_agrees(
                "Arena<256, u8> f32 non-finite A",
                grid3_t,
                &codes,
                &a_nan,
                m,
                k,
                n,
                &Linear::OVERWRITE,
                &zeros,
                f32_demand_table_expected(grid3_t, &codes, &a_nan, m, k, n),
            );
        }

        // The complete f64 spelling is executable when explicitly offered.
        // These small automatic shapes remain below its derived cost boundary;
        // the forced and downstream block-two witnesses below exercise the
        // resident table itself.
        let mut pool64: Vec<f64> = (0..256).map(|q| (q as f64 - 127.5) * 0.015_625).collect();
        assert_eq!(canonicalize(&mut pool64), 256, "256 distinct bit patterns");
        let t64: &[Alphabet<f64, Whole<f64>>; 256] = as_alphabet_whole(&pool64).try_into().unwrap();
        for &(m, k, n) in &[(2usize, 3usize, 5usize), (7, 40, 9)] {
            let codes: Vec<u8> = fill(n * k, 0xa4ea, |x| x as u8);
            let a: Vec<f64> = fill(m * k, 0xac7, |x| (x % 7) as f64 * 0.5 - 1.5);
            every_arena_traversal_agrees(
                "Arena<256, u8> f64",
                t64,
                &codes,
                &a,
                m,
                k,
                n,
                &Linear::OVERWRITE,
                &vec![0.0f64; m * n],
                false,
            );
        }

        // The row collapse composes with the lane: repeated rows of `A` are
        // numbered, the compacted product is walked and tabulated, and the
        // expansion copies the sums. The same total q table executes on the
        // compacted rows, and the bytes are the dense driver's.
        let pool = codebook(3);
        let table: &[Alphabet<f32, Whole<f32>>; 256] = as_alphabet_whole(&pool).try_into().unwrap();
        let (m, k, n) = (7usize, 40usize, 9usize);
        let codes: Vec<u8> = fill(n * k, 0xa4ea, |x| x as u8);
        // Rows in threes: two repeats of every row, so the collapse has work
        // to do and the recursion's own q extent observation is exercised.
        let a: Vec<f32> = (0..m * k)
            .map(|at| {
                let row = at / k % 3;
                ((at % 7) as f32 * 0.5 - 1.5) * (row as f32 + 1.0)
            })
            .collect();
        let zeros = vec![0.0f32; m * n];
        let want: Vec<u64> =
            arena_reference(table, &codes, &a, m, k, n, &Linear::OVERWRITE, &zeros)
                .iter()
                .map(|v| v.symbol_bits())
                .collect();
        let (got, census) = arena_tabulated(
            table,
            &codes,
            &a,
            m,
            k,
            n,
            Traversal::Tabulated,
            OFFER_STEPS,
            m * k,
            &Linear::OVERWRITE,
            &zeros,
        );
        let got: Vec<u64> = got.iter().map(|v| v.symbol_bits()).collect();
        assert_eq!(
            got, want,
            "a collapsed and tabulated run must give the dense float driver's bytes ({census:?})"
        );
        assert!(
            census.table_reads > 0,
            "the compacted product tabulated ({census:?})"
        );

        // The capacity boundary, driven: every code names the codebook's
        // widest symbol and every activation is the widest finite
        // significand, all one sign, so a run's sum approaches `2^63` rather
        // than cancelling away from it. Both panels then span zero binades
        // and sit *at* the declared per-step bound: each scaled significand
        // is `2^24 - 1`, each product `2^48 - 2^25 + 1`, and the honest run
        // is 32767 of them --- `k = 40000` is one full run and a tail, where
        // a single run sums past `2^63`. A capacity declaration one run too
        // long wraps the lane and writes different bytes --- which is the
        // plant the falsifiability table records, and it only fires because
        // this fill reaches the bound.
        let mut edge: Vec<f32> = (0..255u64)
            .map(|q| 1.0 + q as f32 * 2.0f32.powi(-12))
            .collect();
        edge.push(f32::from_bits(0x3FFF_FFFF));
        assert_eq!(canonicalize(&mut edge), 256, "256 distinct patterns");
        let edge_t: &[Alphabet<f32, Whole<f32>>; 256] =
            as_alphabet_whole(&edge).try_into().unwrap();
        let (m, k, n) = (2usize, 40000usize, 3usize);
        let codes = vec![255u8; n * k];
        let a = vec![f32::from_bits(0x3FFF_FFFF); m * k];
        let zeros = vec![0.0f32; m * n];
        every_arena_traversal_agrees(
            "Arena<256, u8> f32 worst case",
            edge_t,
            &codes,
            &a,
            m,
            k,
            n,
            &Linear::OVERWRITE,
            &zeros,
            f32_demand_table_expected(edge_t, &codes, &a, m, k, n),
        );
    }

    /// `CD-20`: the API-locked complete f64 lane is an executable pure Atlas
    /// table when the caller explicitly offers its residency. The shape-only
    /// automatic predicate does not assign contextual block one a scalar
    /// timing rule, but that cannot become a categorical family refusal:
    /// longer downstream codec blocks use this same declaration.
    #[test]
    fn forced_f64_symbol_traversal_uses_the_complete_atlas_table_cd_20() {
        let symbols = [0.5f64, 0.75, 1.0, 1.25, -0.5, -0.75, -1.0, -1.25];
        let table: &[Alphabet<f64, Whole<f64>>; 8] =
            as_alphabet_whole(&symbols).try_into().unwrap();
        let (m, k, n) = (3usize, 19usize, 11usize);
        let shape = Shape { m, k, n };
        let suggested = suggested_tabulation::<f64, Whole<f64>>(shape, 8, 1);
        assert!(
            suggested > 0,
            "an executable complete-word table must advertise its exact residency"
        );
        assert_eq!(
            suggested_tabulation_lanes::<f64, Whole<f64>>(shape, 8, 1),
            0,
            "the same declaration needs no narrow lane offer"
        );
        let codes: Vec<u8> = fill(n * k, 0xf64a, |x| (x % 8) as u8);
        let activations: Vec<f64> = fill(m * k, 0xf64b, |x| ((x % 17) as f64 - 8.0) * 0.125);
        let zeros = vec![0.0f64; m * n];
        let want: Vec<u64> = arena_reference(
            table,
            &codes,
            &activations,
            m,
            k,
            n,
            &Linear::OVERWRITE,
            &zeros,
        )
        .iter()
        .map(|value| value.to_bits())
        .collect();

        let (got, census) = arena_tabulated(
            table,
            &codes,
            &activations,
            m,
            k,
            n,
            Traversal::Tabulated,
            OFFER_STEPS,
            0,
            &Linear::OVERWRITE,
            &zeros,
        );
        assert_eq!(
            got.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            want,
            "the named traversal must preserve the direct codebook product's bytes ({census:?})"
        );
        assert_eq!(
            census.kernel_calls, 0,
            "the forced table must not present a dense operand ({census:?})"
        );
        assert!(
            census.table_reads > 0,
            "the complete-word f64 table must execute ({census:?})"
        );
        assert_eq!(
            census.multiplies, 0,
            "the complete table build and gather are Atlas lookup/add only ({census:?})"
        );
        let demanded_entries = (0..k)
            .map(|p| {
                let mut seen = [false; 8];
                for column in codes.chunks_exact(k) {
                    seen[usize::from(column[p]) % symbols.len()] = true;
                }
                seen.into_iter().filter(|present| *present).count()
            })
            .sum::<usize>();
        assert_eq!(
            census.adds,
            (demanded_entries * m + m * n * k) as u64,
            "the complete-Atlas build reports one presentation per distinct addressed entry, \
             plus one transparent gather combine ({census:?})"
        );

        // The exact lane and the output tile share one accumulator offer. One
        // word below the full-width suggestion must replan a narrower
        // column/depth block, not construct the full plan and decline it after
        // double-booking those cells.
        for exact_offer in [suggested - 1, suggested] {
            let mut exact = vec![<AccOf<f64> as Accumulator>::ZERO; exact_offer];
            let mut panel =
                vec![Alphabet::<f64, Whole<f64>>::ZERO; suggested_tabulation_panel(8, 1)];
            let mut lanes: [i64; 0] = [];
            let mut index = vec![0usize; suggested_tabulation_index(shape)];
            let mut output = vec![0.0f64; m * n];
            let mut boundary_census = Census::default();
            {
                let a = MatView::row_major(as_alphabet_whole(&activations), m, k).unwrap();
                let w = CodedMatrix::new(Arena::new(table), n, k, &codes).unwrap();
                let c = MatViewMut::row_major(&mut output, m, n).unwrap();
                let mut triple = TabulatedTriple::new(a, w, c).unwrap();
                gemm_tabulated_counted(
                    &mut triple,
                    &Linear::OVERWRITE,
                    GemmOptions {
                        traversal: Traversal::Tabulated,
                        ..Default::default()
                    },
                    &mut Scratch::with_accumulators(&mut panel, &mut exact),
                    &mut Tabulation::with_index(&mut lanes, &mut index),
                    &mut Collapse::none(),
                    &mut boundary_census,
                );
            }
            assert_eq!(
                output
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                want,
                "shared exact offer {exact_offer}"
            );
            assert!(
                boundary_census.table_reads > 0 && boundary_census.kernel_calls == 0,
                "offer {exact_offer} must execute its replanned table: {boundary_census:?}"
            );
        }
    }

    /// `CD-20`: f64 planning is parametric in the codec declaration. Arena's
    /// block of one cannot disable a downstream two-element code whose table
    /// fits and amortizes its build; the automatic traversal executes the same
    /// complete Atlas lane and the public workspace query supplies it.
    #[test]
    fn downstream_block_two_f64_codec_is_not_categorically_declined_cd_20() {
        let (m, k, n) = (1usize, 6usize, 8usize);
        let shape = Shape { m, k, n };
        let blocks = k / PairF64::MAX_BLOCK;
        let codes: Vec<u8> = (0..n * blocks)
            .map(|at| {
                let (column, block) = (at / blocks, at % blocks);
                ((column / (0..block).fold(1usize, |scale, _| scale + scale)) % 2) as u8
            })
            .collect();
        let activations = [0.5f64, -1.25, 2.0, 0.75, -0.5, 1.5];

        let mut decoded = vec![0.0f64; k * n];
        for p in 0..k {
            for j in 0..n {
                decoded[p * n + j] = PairF64
                    .decode_element(
                        codes[j * blocks + p / PairF64::MAX_BLOCK],
                        p % PairF64::MAX_BLOCK,
                    )
                    .get();
            }
        }
        let mut want = vec![0.0f64; m * n];
        {
            let a = MatView::row_major(&activations, m, k).unwrap();
            let b = MatView::row_major(&decoded, k, n).unwrap();
            let c = MatViewMut::row_major(&mut want, m, n).unwrap();
            let mut dense = Triple::new(a, b, c).unwrap();
            gemm_float(&mut dense, &Linear::OVERWRITE, GemmOptions::default());
        }

        let exact_words = suggested_tabulation::<f64, Whole<f64>>(shape, 2, 2);
        assert!(
            exact_words > 0,
            "the complete lane advertises its residency"
        );
        let mut exact = vec![<AccOf<f64> as Accumulator>::ZERO; exact_words];
        let mut panel = vec![Alphabet::<f64, Whole<f64>>::ZERO; suggested_tabulation_panel(2, 2)];
        let mut lanes: [i64; 0] = [];
        let mut index = vec![0usize; suggested_tabulation_index(shape)];
        let mut got = vec![0.0f64; m * n];
        let mut census = Census::default();
        {
            let a = MatView::row_major(as_alphabet_whole(&activations), m, k).unwrap();
            let w = CodedMatrix::new(PairF64, n, k, &codes).unwrap();
            let c = MatViewMut::row_major(&mut got, m, n).unwrap();
            let mut triple = TabulatedTriple::new(a, w, c).unwrap();
            gemm_tabulated_counted(
                &mut triple,
                &Linear::OVERWRITE,
                GemmOptions {
                    traversal: Traversal::Blocked,
                    ..Default::default()
                },
                &mut Scratch::with_accumulators(&mut panel, &mut exact),
                &mut Tabulation::with_index(&mut lanes, &mut index),
                &mut Collapse::none(),
                &mut census,
            );
        }
        assert_eq!(
            got.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|value| value.to_bits()).collect::<Vec<_>>()
        );
        assert!(
            census.table_reads > 0,
            "the paying block-two table ran: {census:?}"
        );
        assert_eq!(census.kernel_calls, 0, "no dense decline: {census:?}");
        assert_eq!(
            census.multiplies, 0,
            "the complete build is pure Atlas: {census:?}"
        );
    }

    /// Exercise the exact panel lengths at the boundary the scaled-table tests
    /// intentionally abstract into fractions.  In particular, zero and one
    /// element are real offers rather than aliases for a generic stream, and a
    /// reversed or broadcast activation view cannot make the Atlas capability
    /// disappear.
    fn every_float_panel_offer_is_atlas<E, const D: usize>(
        table: &[Alphabet<E, Whole<E>>; D],
        codes: &[u8],
        a: MatView<'_, Alphabet<E, Whole<E>>>,
        m: usize,
        k: usize,
        n: usize,
        c0: &[E],
    ) where
        E: FloatElement + EncodeFrom<AccOf<E>> + Tabulated,
        AccOf<E>: crate::SignedPlace,
        Linear: Epilogue<E, E>,
    {
        let mut decoded = vec![E::ZERO; k * n];
        for p in 0..k {
            for j in 0..n {
                decoded[p * n + j] = table[codes[j * k + p] as usize % D].get();
            }
        }
        let mut want = c0.to_vec();
        {
            let b = MatView::row_major(&decoded, k, n).unwrap();
            let c = MatViewMut::row_major(&mut want, m, n).unwrap();
            let mut dense = Triple::new(a.peeled(), b, c).unwrap();
            gemm_float(&mut dense, &Linear::ACCUMULATE, GemmOptions::default());
        }
        let want: Vec<u64> = want.iter().map(|value| value.symbol_bits()).collect();

        for traversal in [
            Traversal::Tabulated,
            Traversal::Blocked,
            Traversal::OutputMajor,
        ] {
            // `k` is one decoded row; `n*k + k` is the whole transposed
            // operand plus the dense driver's own panel. The values between
            // cross every source-residency boundary while retaining the same
            // StreamLane contraction.
            for offer in [0usize, 1, k - 1, k, n * k + k] {
                let mut panel = vec![Alphabet::<E, Whole<E>>::ZERO; offer];
                let mut got = c0.to_vec();
                let mut census = Census::default();
                {
                    let c = MatViewMut::row_major(&mut got, m, n).unwrap();
                    let w = CodedMatrix::new(Arena::new(table), n, k, codes)
                        .expect("the codes describe n x k");
                    let mut triple = TabulatedTriple::new(a, w, c).unwrap();
                    gemm_tabulated_counted(
                        &mut triple,
                        &Linear::ACCUMULATE,
                        GemmOptions {
                            traversal,
                            ..Default::default()
                        },
                        &mut Scratch::new(&mut panel),
                        &mut Tabulation::none(),
                        &mut Collapse::none(),
                        &mut census,
                    );
                }
                let got: Vec<u64> = got.iter().map(|value| value.symbol_bits()).collect();
                assert_eq!(
                    got, want,
                    "{m}x{k}x{n} {traversal:?}, panel {offer}: the persistent lane must finish \
                     before the caller epilogue ({census:?})"
                );
                assert_eq!(
                    census.table_reads, 0,
                    "no table offer exists at panel {offer} ({census:?})"
                );
                let packed = traversal != Traversal::OutputMajor && offer == n * k + k;
                let source_page = if offer == 0 { blocking::KC } else { offer };
                let expected_calls = if packed {
                    1
                } else if source_page >= k {
                    n * m.div_ceil(ROW_TILES[0])
                } else {
                    m * n * k.div_ceil(source_page)
                };
                assert_eq!(
                    census.kernel_calls, expected_calls as u64,
                    "panel {offer} must count each page actually presented to the dense engine \
                     ({census:?})"
                );
                let expected_decodes = if packed || source_page >= k {
                    n * k
                } else {
                    m * n * k
                };
                assert_eq!(
                    census.decodes, expected_decodes as u64,
                    "panel {offer} must be the source page itself, with whole rows shared \
                     and partial rows repeated per dot ({census:?})"
                );
                if offer <= k {
                    assert_eq!(
                        census.multiplies, 0,
                        "empty and short offers must remain in the dense Atlas stream and issue no Element::mac \
                         ({census:?})"
                    );
                }
            }
        }
    }

    /// `CD-20`: every float panel length, including none and one cell, enters
    /// the dense Atlas factorization.  Negative and zero input strides are
    /// views of the same operation and do not recover the generic stream.
    #[test]
    fn every_float_offer_and_stride_reaches_atlas_cd_20() {
        macro_rules! family {
            ($t:ty) => {{
                let symbols: [$t; 8] = [0.5, -0.75, 1.0, -1.25, 1.5, -1.75, 2.0, -2.25];
                let table: &[Alphabet<$t, Whole<$t>>; 8] =
                    as_alphabet_whole(&symbols).try_into().unwrap();
                // One row past the bounded dense-stream tile proves a full
                // decoded column is presented in exactly two family calls,
                // rather than once per output cell.
                let (m, k, n) = (ROW_TILES[0] + 1, 7usize, 3usize);
                let codes: Vec<u8> = fill(n * k, 0xcd20, |x| (x % 8) as u8);
                let activations: Vec<$t> = (0..m * k)
                    .map(|at| (at as i32 % 11 - 5) as $t * 0.125)
                    .collect();
                let c0: Vec<$t> = (0..m * n).map(|at| (at as $t + 1.0) * 0.25).collect();
                let alphabet = as_alphabet_whole(&activations);

                let ordinary = MatView::row_major(alphabet, m, k).unwrap();
                every_float_panel_offer_is_atlas(table, &codes, ordinary, m, k, n, &c0);

                // Each row is read right-to-left. `MatView::new` places the
                // origin at the high end of the same borrowed buffer.
                let reversed = MatView::new(
                    alphabet,
                    m,
                    k,
                    Strides {
                        rs: k as isize,
                        cs: -1,
                    },
                )
                .unwrap();
                every_float_panel_offer_is_atlas(table, &codes, reversed, m, k, n, &c0);

                // Both rows broadcast one reduction run.  Zero is a stride,
                // not a special arithmetic mode.
                let broadcast =
                    MatView::new(&alphabet[..k], m, k, Strides { rs: 0, cs: 1 }).unwrap();
                every_float_panel_offer_is_atlas(table, &codes, broadcast, m, k, n, &c0);
            }};
        }
        family!(f32);
        family!(f64);
    }

    /// `CD-20`: a downstream family that declines its first real empty-rest
    /// partial reaches its unchanged multiplying stream with caller `C`
    /// intact. The attempted partial uses a private capture cell, so the
    /// caller's accumulating epilogue still runs exactly once.
    #[test]
    fn a_declined_first_partial_preserves_the_ordinary_stream_cd_20() {
        let _guard = DENSE_DECLINE_LOCK.lock().unwrap();
        DENSE_ACCEPTS.store(false, Ordering::Relaxed);
        DENSE_CALLS.store(0, Ordering::Relaxed);
        let alphabet = [
            Alphabet::<DenseDecline, MaxBound>::new(DenseDecline(-3)).unwrap(),
            Alphabet::<DenseDecline, MaxBound>::new(DenseDecline(2)).unwrap(),
            Alphabet::<DenseDecline, MaxBound>::new(DenseDecline(5)).unwrap(),
            Alphabet::<DenseDecline, MaxBound>::new(DenseDecline(-7)).unwrap(),
        ];
        let codec = Grid::<DenseDecline, MaxBound, 4>::new(&alphabet);
        let (m, k, n) = (1usize, 3usize, 2usize);
        let codes = [0u16, 1, 2, 3, 0, 1];
        let weights = CodedMatrix::new(codec, n, k, &codes).unwrap();
        let activations = [
            Alphabet::<DenseDecline, MaxBound>::new(DenseDecline(4)).unwrap(),
            Alphabet::<DenseDecline, MaxBound>::new(DenseDecline(-2)).unwrap(),
            Alphabet::<DenseDecline, MaxBound>::new(DenseDecline(3)).unwrap(),
        ];
        let mut c = [11i32, -13];
        let mut one = [Alphabet::<DenseDecline, MaxBound>::ZERO; 1];
        let mut census = Census::default();
        {
            let a = MatView::row_major(&activations, m, k).unwrap();
            let out = MatViewMut::row_major(&mut c, m, n).unwrap();
            let mut triple = TabulatedTriple::new(a, weights, out).unwrap();
            gemm_tabulated_counted(
                &mut triple,
                &Linear::ACCUMULATE,
                GemmOptions {
                    traversal: Traversal::Tabulated,
                    ..Default::default()
                },
                &mut Scratch::new(&mut one),
                &mut Tabulation::none(),
                &mut Collapse::none(),
                &mut census,
            );
        }
        assert_eq!(
            c,
            [11 + 4 * -3 + -2 * 2 + 3 * 5, -13 + 4 * -7 + -2 * -3 + 3 * 2]
        );
        assert_eq!(
            DENSE_CALLS.load(Ordering::Relaxed),
            1,
            "the first real partial is the only dense call before the decline"
        );
        assert_eq!(
            census.kernel_calls, 1,
            "the census counts the real dense call that declared the decline"
        );
        assert_eq!(census.multiplies, (m * k * n) as u64);
    }

    /// `CD-20`: acceptance retains the first real partial. At one cell beyond
    /// the model's page depth, each output needs exactly two dense calls; a
    /// discarded acceptance call or a recomputed first page makes the counter
    /// larger while producing the same bytes, so the count is the witness.
    #[test]
    fn an_accepted_first_partial_is_not_recomputed_cd_20() {
        let _guard = DENSE_DECLINE_LOCK.lock().unwrap();
        DENSE_ACCEPTS.store(true, Ordering::Relaxed);
        DENSE_CALLS.store(0, Ordering::Relaxed);

        let alphabet = [
            Alphabet::<DenseDecline, MaxBound>::new(DenseDecline(-3)).unwrap(),
            Alphabet::<DenseDecline, MaxBound>::new(DenseDecline(2)).unwrap(),
            Alphabet::<DenseDecline, MaxBound>::new(DenseDecline(5)).unwrap(),
            Alphabet::<DenseDecline, MaxBound>::new(DenseDecline(-7)).unwrap(),
        ];
        let codec = Grid::<DenseDecline, MaxBound, 4>::new(&alphabet);
        let (m, k, n) = (1usize, blocking::KC + 1, 2usize);
        let codes: Vec<u16> = (0..n * k).map(|at| (at % alphabet.len()) as u16).collect();
        let weights = CodedMatrix::new(codec, n, k, &codes).unwrap();
        let activations: Vec<Alphabet<DenseDecline, MaxBound>> = (0..k)
            .map(|at| {
                Alphabet::new(DenseDecline(match at % 3 {
                    0 => 4,
                    1 => -2,
                    _ => 3,
                }))
                .unwrap()
            })
            .collect();
        let mut c = [0i32; 2];
        let mut census = Census::default();
        {
            let a = MatView::row_major(&activations, m, k).unwrap();
            let out = MatViewMut::row_major(&mut c, m, n).unwrap();
            let mut triple = TabulatedTriple::new(a, weights, out).unwrap();
            gemm_tabulated_counted(
                &mut triple,
                &Linear::OVERWRITE,
                GemmOptions {
                    traversal: Traversal::OutputMajor,
                    ..Default::default()
                },
                &mut Scratch::none(),
                &mut Tabulation::none(),
                &mut Collapse::none(),
                &mut census,
            );
        }

        let mut want = [0i128; 2];
        for j in 0..n {
            for p in 0..k {
                want[j] += i128::from(activations[p].get().0)
                    * i128::from(alphabet[codes[j * k + p] as usize].get().0);
            }
        }
        assert_eq!(c, [want[0] as i32, want[1] as i32]);
        assert_eq!(
            DENSE_CALLS.load(Ordering::Relaxed),
            m * n * k.div_ceil(blocking::KC),
            "every real page runs once and no separate acceptance call exists"
        );
        assert_eq!(
            census.kernel_calls,
            (m * n * k.div_ceil(blocking::KC)) as u64
        );
        assert_eq!(
            census.adds,
            (m * n * (k.div_ceil(blocking::KC) - 1)) as u64,
            "every complete page after the first is transparently combined once"
        );
        assert_eq!(census.multiplies, 0);
        assert_eq!(census.decodes, (m * n * k) as u64);
        DENSE_ACCEPTS.store(false, Ordering::Relaxed);
    }

    /// `CD-20`: a finite integer alphabet returns before decoding or calling
    /// the empty-rest dense engine. Its existing stream census and product are
    /// exactly unchanged; only a bound with no finite magnitude presents a
    /// real partial.
    #[test]
    fn a_finite_integer_bound_does_not_pay_dense_acceptance_cd_20() {
        let _guard = DENSE_DECLINE_LOCK.lock().unwrap();
        DENSE_ACCEPTS.store(false, Ordering::Relaxed);
        DENSE_CALLS.store(0, Ordering::Relaxed);
        let alphabet = [
            Alphabet::of(DenseDecline(-3)),
            Alphabet::of(DenseDecline(2)),
            Alphabet::of(DenseDecline(5)),
            Alphabet::of(DenseDecline(-7)),
        ];
        let weights = CodedMatrix::new(
            Grid::<DenseDecline, Full<DenseDecline>, 4>::new(&alphabet),
            1,
            3,
            &[0u16, 1, 2],
        )
        .unwrap();
        let activations = [DenseDecline(4), DenseDecline(-2), DenseDecline(3)];
        let mut c = [11i32];
        let mut census = Census::default();
        {
            let a = MatView::row_major(as_alphabet_full(&activations), 1, 3).unwrap();
            let out = MatViewMut::row_major(&mut c, 1, 1).unwrap();
            let mut triple = TabulatedTriple::new(a, weights, out).unwrap();
            gemm_tabulated_counted(
                &mut triple,
                &Linear::ACCUMULATE,
                GemmOptions {
                    traversal: Traversal::OutputMajor,
                    ..Default::default()
                },
                &mut Scratch::none(),
                &mut Tabulation::none(),
                &mut Collapse::none(),
                &mut census,
            );
        }

        assert_eq!(c, [11 + 4 * -3 + -2 * 2 + 3 * 5]);
        assert_eq!(DENSE_CALLS.load(Ordering::Relaxed), 0);
        assert_eq!(census.kernel_calls, 0);
        assert_eq!(census.multiplies, 3);
    }

    /// `CT-01`: `Lane::capacity` is a public declaration and may truthfully
    /// answer `Some(0)`. Such a lane cannot be rounded up to one product: the
    /// ordinary stream contracts directly in the exact accumulator, remains
    /// total at a nonempty reduction, and never invokes the lane's deliberately
    /// failing arithmetic methods.
    #[test]
    fn a_zero_capacity_public_stream_lane_is_total_ct_01() {
        assert_eq!(<ZeroLane as Lane<DenseDecline>>::capacity(1), Some(0));

        let alphabet = [
            Alphabet::of(DenseDecline(-3)),
            Alphabet::of(DenseDecline(2)),
            Alphabet::of(DenseDecline(5)),
            Alphabet::of(DenseDecline(-7)),
        ];
        let weights = CodedMatrix::new(
            Grid::<DenseDecline, Full<DenseDecline>, 4>::new(&alphabet),
            1,
            3,
            &[0u16, 1, 2],
        )
        .unwrap();
        let activations = [DenseDecline(4), DenseDecline(-2), DenseDecline(3)];
        let mut c = [0i32];
        let mut census = Census::default();
        {
            let a = MatView::row_major(as_alphabet_full(&activations), 1, 3).unwrap();
            let output = MatViewMut::row_major(&mut c, 1, 1).unwrap();
            let mut triple = TabulatedTriple::new(a, weights, output).unwrap();
            gemm_tabulated_counted(
                &mut triple,
                &Linear::OVERWRITE,
                GemmOptions {
                    traversal: Traversal::OutputMajor,
                    ..Default::default()
                },
                &mut Scratch::none(),
                &mut Tabulation::none(),
                &mut Collapse::none(),
                &mut census,
            );
        }

        assert_eq!(c, [4 * -3 + -2 * 2 + 3 * 5]);
        assert_eq!(census.multiplies, 3);
        assert_eq!(census.kernel_calls, 0);

        let empty_weights = CodedMatrix::new(
            Grid::<DenseDecline, Full<DenseDecline>, 4>::new(&alphabet),
            1,
            0,
            &[] as &[u16],
        )
        .unwrap();
        let mut empty = [17i32];
        let mut empty_census = Census::default();
        {
            let a = MatView::row_major(&[] as &[Alphabet<DenseDecline, Full<DenseDecline>>], 1, 0)
                .unwrap();
            let output = MatViewMut::row_major(&mut empty, 1, 1).unwrap();
            let mut triple = TabulatedTriple::new(a, empty_weights, output).unwrap();
            gemm_tabulated_counted(
                &mut triple,
                &Linear::ACCUMULATE,
                GemmOptions {
                    traversal: Traversal::OutputMajor,
                    ..Default::default()
                },
                &mut Scratch::none(),
                &mut Tabulation::none(),
                &mut Collapse::none(),
                &mut empty_census,
            );
        }
        assert_eq!(empty, [17]);
        assert_eq!(empty_census.multiplies, 0);
        assert_eq!(empty_census.kernel_calls, 0);
    }

    /// `CD-20`: empty-rest acceptance is exercised by a real one-product dot.
    /// The float Atlas writes that product; an integer family requiring a
    /// reduction panel declines before touching its output.
    #[test]
    fn empty_rest_acceptance_uses_real_work_and_decline_writes_nothing_cd_20() {
        let a32_value = [2.0f32];
        let b32_value = [3.0f32];
        let a32 = as_alphabet_whole(&a32_value);
        let b32 = as_alphabet_whole(&b32_value);
        let mut c32 = [-7.0f32];
        assert!(<f32 as Tabulated>::dense_gemm(
            MatView::row_major(a32, 1, 1).unwrap(),
            MatView::row_major(b32, 1, 1).unwrap(),
            MatViewMut::row_major(&mut c32, 1, 1).unwrap(),
            &Linear::OVERWRITE,
            GemmOptions::default(),
            &mut [],
        ));
        assert_eq!(c32, [6.0]);

        let a8 = [Alphabet::of(2i8)];
        let b8 = [Alphabet::of(3i8)];
        let mut c8 = [41i32];
        assert!(!<i8 as Tabulated>::dense_gemm(
            MatView::row_major(&a8, 1, 1).unwrap(),
            MatView::row_major(&b8, 1, 1).unwrap(),
            MatViewMut::row_major(&mut c8, 1, 1).unwrap(),
            &Linear::OVERWRITE,
            GemmOptions::default(),
            &mut [],
        ));
        assert_eq!(c8, [41]);
    }

    /// `CD-20`: the public stream-lane identity remains the transparent Wide
    /// complete accumulator, while an empty coded offer contracts through the
    /// dense Atlas on both sides of the model's page depth. With runtime
    /// backend discovery enabled, every real page remains one census call but
    /// the Atlas kernel family is resolved only once for that backend.
    #[test]
    fn atlas_stream_retains_wide_and_crosses_pages_exactly_cd_20() {
        fn is_public_wide<E>()
        where
            E: Tabulated<StreamLane = Wide<AccOf<E>>>,
        {
        }
        is_public_wide::<f32>();
        is_public_wide::<f64>();

        macro_rules! family {
            ($element:ty) => {{
                let symbols: [$element; 2] = [0.5, -0.75];
                let table: &[Alphabet<$element, Whole<$element>>; 2] =
                    as_alphabet_whole(&symbols).try_into().unwrap();
                let k = blocking::KC + 1;
                let codes: Vec<u8> = (0..k).map(|p| (p & 1) as u8).collect();
                let a: Vec<$element> = (0..k)
                    .map(|p| if p & 1 == 0 { 1.25 } else { -1.5 })
                    .collect();
                let (got, census) = arena_tabulated(
                    table,
                    &codes,
                    &a,
                    1,
                    k,
                    1,
                    Traversal::OutputMajor,
                    0,
                    0,
                    &Linear::OVERWRITE,
                    &[0.0 as $element],
                );

                let mut want = <AccOf<$element> as Accumulator>::ZERO;
                for p in 0..k {
                    <$element as Element>::mac(&mut want, a[p], symbols[p & 1]);
                }
                let want = <$element as EncodeFrom<AccOf<$element>>>::encode_from(
                    want,
                    EncodeMode::Nearest,
                );
                assert_eq!(got[0].symbol_bits(), want.symbol_bits());
                assert_eq!(census.multiplies, 0);
                assert_eq!(census.kernel_calls, k.div_ceil(blocking::KC) as u64);
                assert_eq!(census.adds, (k.div_ceil(blocking::KC) - 1) as u64);
                assert_eq!(census.decodes, k as u64);
            }};
        }

        family!(f32);
        family!(f64);

        #[cfg(feature = "std")]
        assert_eq!(
            crate::float::atlas_dot_resolutions(GemmOptions::default().backend),
            1,
            "all f32/f64 pages share one cached Atlas family resolution"
        );
    }

    /// `CD-20`: an empty reduction has no real partial with which to establish
    /// empty-rest dense acceptance. It remains in the public `StreamLane`,
    /// applies the caller epilogue once per output, and records no kernel,
    /// multiply, or decode.
    #[test]
    fn an_empty_float_reduction_uses_the_public_stream_zero_cd_20() {
        let symbols = [0.5f32];
        let table: &[Alphabet<f32, Whole<f32>>; 1] =
            as_alphabet_whole(&symbols).try_into().unwrap();
        let (m, k, n) = (2usize, 0usize, 3usize);
        let c0 = [1.0f32, -2.0, 3.0, -4.0, 5.0, -6.0];
        for traversal in [
            Traversal::Tabulated,
            Traversal::Blocked,
            Traversal::OutputMajor,
        ] {
            for offer in [0, OFFER_STEPS] {
                let (got, census) = arena_tabulated(
                    table,
                    &[] as &[u8],
                    &[],
                    m,
                    k,
                    n,
                    traversal,
                    offer,
                    0,
                    &Linear::ACCUMULATE,
                    &c0,
                );

                assert_eq!(
                    got.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                    c0.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                    "{traversal:?} at offer {offer} must apply the exact stream zero"
                );
                assert_eq!(census.table_reads, 0);
                assert_eq!(census.decodes, 0);
                assert_eq!(census.multiplies, 0);
                assert_eq!(census.kernel_calls, 0);
            }
        }
    }

    /// `CD-17`: collapsing bit-identical rows of `A` in the float tabulated
    /// traversal cannot change a byte, at every degeneracy and every offer ---
    /// and rows that differ only in the sign of a zero or in a NaN payload are
    /// distinct rows, charged as such.
    ///
    /// The census is what keeps this from passing with the collapse silently
    /// declined, as it does for `CD-15`: the build charges per *distinct* row
    /// when the collapse ran and per row when anything declined it, and the
    /// closed form `CU-06` pins says which happened. The bit cases are the
    /// arena tier's canonical-codebook semantics (`CK-10`) read onto the
    /// collapse: the bit pattern is the symbol, so a sign of zero or a NaN
    /// payload makes two rows where numeric `==` would see one --- and a NaN
    /// row is a repeat of itself, which numeric `==` cannot see at all.
    #[test]
    fn collapsing_bit_identical_float_rows_cannot_change_a_byte_cd_17() {
        // The whole pool in the codebook, as `CD-14`'s: the weights carry an
        // infinity, a NaN, and both zeros.
        let mut pool32 = [0.5f32, 1.0, f32::INFINITY, f32::NAN, -0.0, -1.5, -2.5, 0.0];
        assert_eq!(canonicalize(&mut pool32), 8, "eight distinct bit patterns");
        let mut pool64 = [0.5f64, 1.0, f64::INFINITY, f64::NAN, -0.0, -1.5, -2.5, 0.0];
        assert_eq!(canonicalize(&mut pool64), 8, "eight distinct bit patterns");

        macro_rules! case {
            ($t:ty, $label:expr, $pool:expr, $nan1:expr, $nan2:expr, $table:expr) => {{
                let label = $label;
                let table: &[Alphabet<$t, Whole<$t>>; 8] =
                    as_alphabet_whole(&$pool).try_into().unwrap();
                let (k, n) = (4usize, 9usize);
                // Codes past the table on purpose: the enumeration reduces them
                // modulo `D`, and the reference decodes them the same way.
                let codes: Vec<u16> = fill(n * k, 0xa4ea, |x| (x % 17) as u16);
                let demanded_entries = (0..k)
                    .map(|p| {
                        let mut seen = [false; 8];
                        for column in codes.chunks_exact(k) {
                            seen[usize::from(column[p]) % table.len()] = true;
                        }
                        seen.into_iter().filter(|present| *present).count()
                    })
                    .sum::<usize>();
                // Distinct bit patterns, so row `r` of `symbols[r..r + k]`
                // differs from every other row in its first element. Both zeros
                // and two NaN payloads are among them.
                let symbols: [$t; 11] = [
                    0.5,
                    1.0,
                    -0.0,
                    0.0,
                    $nan1,
                    $nan2,
                    -1.5,
                    2.5,
                    <$t>::INFINITY,
                    -2.5,
                    3.75,
                ];
                // One row past the dense stream's derived batch width makes
                // the opaque family-call census distinguish a successful
                // collapse from an uncollapsed product without inventing the
                // dense engine's internal arithmetic count.
                let m = ROW_TILES[0] + 1;
                for d in [1usize, 2, 4, 8] {
                    // `d` distinct rows, numbered by first occurrence: row `i`
                    // is row `i % d`, and the first `d` are pairwise distinct.
                    let a: Vec<$t> = (0..m * k).map(|x| symbols[(x / k % d) + x % k]).collect();
                    let want: Vec<u64> = arena_reference(
                        table,
                        &codes,
                        &a,
                        m,
                        k,
                        n,
                        &Linear::OVERWRITE,
                        &vec![<$t>::default(); m * n],
                    )
                    .iter()
                    .map(|v| v.symbol_bits())
                    .collect();
                    // No offer, a rows offer too short for the distinct rows,
                    // exactly enough, and the worst case.
                    for collapse_offer in [0usize, k / 2, d * k, m * k] {
                        // Keep the f32 panel below the whole decoded operand so
                        // its bounded Atlas stream census, rather than a single
                        // packed presentation, witnesses the row collapse. f64
                        // needs the complete table offer whose build is its
                        // witness.
                        let route_offer = if $table { OFFER_STEPS } else { 1 };
                        let (got, census) = arena_tabulated(
                            table,
                            &codes,
                            &a,
                            m,
                            k,
                            n,
                            Traversal::Tabulated,
                            route_offer,
                            collapse_offer,
                            &Linear::OVERWRITE,
                            &vec![<$t>::default(); m * n],
                        );
                        let got: Vec<u64> = got.iter().map(|v| v.symbol_bits()).collect();
                        assert_eq!(
                            got, want,
                            "{label} {m}x{k}x{n} d={d} rows offer {collapse_offer}: the collapse \
                             must give the dense float driver's bytes ({census:?})"
                        );
                        // The work charges per *distinct* row when the collapse
                        // ran, and per row when it did not. f32's non-finite
                        // panel reaches the dense presentation count; f64's
                        // complete table reports its build presentations plus
                        // gather combines. Both move with `d`, which is the
                        // collapse's evidence.
                        let charged = if collapse_offer >= d * k && d < m {
                            d
                        } else {
                            m
                        };
                        let (observed, expected) = if $table {
                            (census.adds, (charged * (demanded_entries + n * k)) as u64)
                        } else {
                            (
                                census.kernel_calls,
                                (n * charged.div_ceil(ROW_TILES[0])) as u64,
                            )
                        };
                        assert_eq!(
                            observed, expected,
                            "{label} {m}x{k}x{n} d={d} rows offer {collapse_offer}: the selected \
                             factorization must charge per distinct row ({census:?})"
                        );
                    }

                    // The collapse runs before any traversal choice, so the
                    // recursion's own declines --- the costed traversal
                    // refusing the table, the streaming one refusing it by
                    // name --- walk the compacted product and the expansion
                    // too. The bytes are the witness here; the census of a
                    // decline has no closed form in `d`.
                    for traversal in [Traversal::Blocked, Traversal::OutputMajor] {
                        let (got, census) = arena_tabulated(
                            table,
                            &codes,
                            &a,
                            m,
                            k,
                            n,
                            traversal,
                            OFFER_STEPS,
                            d * k,
                            &Linear::OVERWRITE,
                            &vec![<$t>::default(); m * n],
                        );
                        let got: Vec<u64> = got.iter().map(|v| v.symbol_bits()).collect();
                        assert_eq!(
                            got, want,
                            "{label} {m}x{k}x{n} d={d}: {traversal:?} under a full rows offer \
                             must give the dense float driver's bytes ({census:?})"
                        );
                    }
                }

                // The witness that bit identity and not numeric equality
                // decides: seventeen rows over four symbols. Rows 0 and 1 differ only in
                // the sign of a zero; rows 2 and 3 only in a NaN payload; rows
                // after those repeat them bit for bit. Numeric `==` would
                // merge the zero pair and refuse every NaN repeat, while bit
                // identity sees exactly four rows.
                let m = ROW_TILES[0] + 1;
                let patterns = [
                    [1.0, -0.0, 0.5, -1.5],
                    [1.0, 0.0, 0.5, -1.5],
                    [$nan1, 1.0, 0.5, -1.5],
                    [$nan2, 1.0, 0.5, -1.5],
                ];
                let a: Vec<$t> = (0..m)
                    .flat_map(|row| patterns[row % patterns.len()])
                    .collect();
                let d = 4usize;
                let want: Vec<u64> = arena_reference(
                    table,
                    &codes,
                    &a,
                    m,
                    k,
                    n,
                    &Linear::OVERWRITE,
                    &vec![<$t>::default(); m * n],
                )
                .iter()
                .map(|v| v.symbol_bits())
                .collect();
                for collapse_offer in [0usize, d * k] {
                    let route_offer = if $table { OFFER_STEPS } else { 1 };
                    let (got, census) = arena_tabulated(
                        table,
                        &codes,
                        &a,
                        m,
                        k,
                        n,
                        Traversal::Tabulated,
                        route_offer,
                        collapse_offer,
                        &Linear::OVERWRITE,
                        &vec![<$t>::default(); m * n],
                    );
                    let got: Vec<u64> = got.iter().map(|v| v.symbol_bits()).collect();
                    assert_eq!(
                        got, want,
                        "{label} bit cases, rows offer {collapse_offer}: the collapse must give \
                         the dense float driver's bytes ({census:?})"
                    );
                    let charged = if collapse_offer >= d * k { d } else { m };
                    let (observed, expected) = if $table {
                        (census.adds, (charged * (demanded_entries + n * k)) as u64)
                    } else {
                        (
                            census.kernel_calls,
                            (n * charged.div_ceil(ROW_TILES[0])) as u64,
                        )
                    };
                    assert_eq!(
                        observed, expected,
                        "{label} bit cases, rows offer {collapse_offer}: `-0.0` beside `+0.0` \
                         and one NaN payload beside another are distinct rows ({census:?})"
                    );
                }
            }};
        }
        case!(
            f32,
            "Arena<8> f32",
            pool32,
            f32::from_bits(0x7fc0_0001),
            f32::from_bits(0x7fc0_0002),
            // The q lane retains the NaN rows and non-finite codebook as
            // singleton tags. Its stream engine is opaque here; kernel
            // presentations, rather than a fabricated multiply count, witness
            // the collapse.
            false
        );
        case!(
            f64,
            "Arena<8> f64",
            pool64,
            f64::from_bits(0x7ff8_0000_0000_0001),
            f64::from_bits(0x7ff8_0000_0000_0002),
            // The complete table builds each distinct addressed entry once
            // per reduction position and gathers every stored code.
            true
        );
    }

    /// `CK-13`: the sign and ternary tiers are spellings of codec compositions
    /// that already exist, not new arithmetic. Weights in `{-1, +1}` stored as
    /// one-bit codes (`Packed<Grid<2>, 8>`, table `[-1, +1]`) give the dense
    /// spelling's bytes through the packed route and through the table, and the
    /// ternary spelling (`Packed<Grid<4>, 4>`, table `[-1, 0, +1, dead]`)
    /// likewise --- at shapes on both sides of each tier's recorded break-even.
    ///
    /// Because the composition predates this ID, the byte assertions alone would
    /// pass against a table that was silently declined. What makes them a gate
    /// is what they are paired with: the census (`every_traversal_agrees` fails
    /// unless a full offer reads the table and an empty one cannot), and the
    /// `[[tabulation]]` rows in `model/tiers.toml`, whose break-evens `CM-04`
    /// recomputes from the codec's own consts. Note what the byte assertions do
    /// *not* watch: the decode itself, which every route here shares, so a wrong
    /// table would move all of them together --- decode order is `CK-03`'s and
    /// `CK-09`'s ground. What they watch is the traversal: a planted drop of one
    /// block word in the portable table build was seen to fail this test before
    /// it was accepted.
    #[test]
    fn sign_and_ternary_spellings_match_the_dense_spelling_ck_13() {
        // The sign tier: one-bit codes, eight to a byte. `code_space = 256` and
        // `block = 8` are E8's numbers, so the break-even is 683 too --- the
        // shapes straddle it.
        let sign_table: [A8; 2] = [Alphabet::of(-1), Alphabet::of(1)];
        let sign =
            Packed::<_, 8>::new(Grid::<i8, Full<i8>, 2>::new(&sign_table)).expect("8 divides 8");
        for &(m, k, n) in &[
            (1usize, 8usize, 1usize),
            (5, 24, 683),
            (3, 8, 684),
            (4, 16, 700),
        ] {
            let stream: Vec<u8> = fill(n * (k / 8), 0x516, |x| x as u8);
            every_traversal_agrees("Packed<Grid<2>,8>", sign, &stream, m, k, n);
        }

        // The ternary tier: two-bit codes over `[-1, 0, +1, dead]`. The fourth
        // entry is one no ternary encoder emits; it decodes to 0 here, which is
        // what "dead" means --- a priced duplicate, not an error (the ratio is
        // `CG-10`'s to report). The full-range streams hit it on purpose; the
        // restricted streams spell what an encoder would actually write.
        let ternary_table: [A8; 4] = [
            Alphabet::of(-1),
            Alphabet::of(0),
            Alphabet::of(1),
            Alphabet::of(0),
        ];
        let ternary =
            Packed::<_, 4>::new(Grid::<i8, Full<i8>, 4>::new(&ternary_table)).expect("4 divides 8");
        for &(m, k, n) in &[(1usize, 4usize, 1usize), (5, 12, 1025), (4, 16, 1100)] {
            let stream: Vec<u8> = fill(n * (k / 4), 0x7e7, |x| x as u8);
            every_traversal_agrees(
                "Packed<Grid<4>,4> with the dead entry",
                ternary,
                &stream,
                m,
                k,
                n,
            );
            let live: Vec<u8> = fill(n * (k / 4), 0x11e, |x| {
                (0..4).fold(0u8, |b, s| b | (((x >> (2 * s)) % 3) as u8) << (2 * s))
            });
            every_traversal_agrees("Packed<Grid<4>,4> ternary", ternary, &live, m, k, n);
        }

        // The bound the claim names, exercised literally: at `Bnd<1>` the
        // weights *and* the activations are signs, so the products are `+-1`
        // and the table's entries are small sums of them. One route decodes
        // the operand and runs the tile kernels, the other reads the table;
        // the census says the second really did.
        fn check_bound_one<C: Enumerable<i8, Bnd<1>> + Copy>(
            label: &str,
            w: &CodedMatrix<'_, i8, Bnd<1>, C>,
            m: usize,
            n: usize,
        ) {
            let k = w.cols();
            let a: Vec<i8> = fill(m * k, 0xac7, |x| if x & 1 == 0 { -1 } else { 1 });
            let want = reference(w, &a, m, k, n);
            let (packed, _) =
                tabulated(w, &a, m, n, Traversal::Blocked, OFFER_STEPS, OFFER_STEPS, 0);
            let (tabled, census) = tabulated(
                w,
                &a,
                m,
                n,
                Traversal::Tabulated,
                OFFER_STEPS,
                OFFER_STEPS,
                0,
            );
            assert_eq!(
                packed, want,
                "{label} {m}x{k}x{n}: the packed route must give the dense driver's bytes"
            );
            assert_eq!(
                tabled, want,
                "{label} {m}x{k}x{n}: the tabulated route must give the dense driver's bytes"
            );
            assert!(
                census.table_reads > 0,
                "{label} {m}x{k}x{n}: the offer was sized for a table and none was read"
            );
        }

        let sign_one: [Alphabet<i8, Bnd<1>>; 2] = [
            Alphabet::new(-1).expect("|-1| <= 1"),
            Alphabet::new(1).expect("|1| <= 1"),
        ];
        let sign_b1 =
            Packed::<_, 8>::new(Grid::<i8, Bnd<1>, 2>::new(&sign_one)).expect("8 divides 8");
        for &(m, k, n) in &[(4usize, 16usize, 700), (3, 8, 96)] {
            let stream: Vec<u8> = fill(n * (k / 8), 0xb1d, |x| x as u8);
            let w = CodedMatrix::new(sign_b1, n, k, &stream).expect("the codes describe n x k");
            check_bound_one("Packed<Grid<2>,8> at Bnd<1>", &w, m, n);
        }

        let ternary_one: [Alphabet<i8, Bnd<1>>; 4] = [
            Alphabet::new(-1).expect("|-1| <= 1"),
            Alphabet::new(0).expect("|0| <= 1"),
            Alphabet::new(1).expect("|1| <= 1"),
            Alphabet::new(0).expect("|0| <= 1"),
        ];
        let ternary_b1 =
            Packed::<_, 4>::new(Grid::<i8, Bnd<1>, 4>::new(&ternary_one)).expect("4 divides 8");
        for &(m, k, n) in &[(4usize, 16usize, 1100), (3, 8, 96)] {
            let stream: Vec<u8> = fill(n * (k / 4), 0x7e9, |x| x as u8);
            let w = CodedMatrix::new(ternary_b1, n, k, &stream).expect("the codes describe n x k");
            check_bound_one("Packed<Grid<4>,4> at Bnd<1>", &w, m, n);
        }
    }

    /// `CK-11`: the `Sign` tier and the `Packed<Grid<2>,8>` spelling are two
    /// manifests for one decode. The gemm output is byte-identical through the
    /// packed route and through the table, at the shapes `CK-13` straddles the
    /// shared break-even with --- and the tier is run through the same
    /// every-offer gate the composition is.
    ///
    /// What the tier adds over the composition is not a different answer but
    /// the index stream: a `Sign` code addresses its enumeration directly, so
    /// the tabulated run gathers from the operand's own memory instead of a
    /// stream it built. That is asserted where the claim lives, at the codec
    /// (`as_index_stream` answers the same slice it was handed); what this
    /// test watches is that the answer does not move with the spelling.
    #[test]
    fn the_sign_tier_matches_the_composition_byte_for_byte_ck_11() {
        let sign_table: [A8; 2] = [Alphabet::of(-1), Alphabet::of(1)];
        let composition =
            Packed::<_, 8>::new(Grid::<i8, Full<i8>, 2>::new(&sign_table)).expect("8 divides 8");
        let tier = Sign::<i8, Full<i8>, 8>::new().expect("the full alphabet admits +-1");

        for &(m, k, n) in &[
            (1usize, 8usize, 1usize),
            (5, 24, 683),
            (3, 8, 684),
            (4, 16, 700),
        ] {
            // One stream, two spellings: the composition stores the byte, the
            // tier the same value zero-extended to a `u16` code.
            let bytes: Vec<u8> = fill(n * (k / 8), 0x516, |x| x as u8);
            let codes: Vec<u16> = bytes.iter().map(|&b| u16::from(b)).collect();

            // The tier against the dense driver's bytes at every offer --- the
            // gate `CK-13` runs the composition through, unchanged.
            every_traversal_agrees("Sign<8>", tier, &codes, m, k, n);

            // And the two spellings against each other directly: equal decodes
            // under different kappa labels, so identical bytes (`CK-05`
            // restated for this pair), through the packed route and through
            // the table.
            let a: Vec<i8> = fill(m * k, 0xa11, |x| ((x % 255) as i64 - 127) as i8);
            let w_packed =
                CodedMatrix::new(composition, n, k, &bytes).expect("the codes describe n x k");
            let w_tier = CodedMatrix::new(tier, n, k, &codes).expect("the codes describe n x k");
            for traversal in [Traversal::Tabulated, Traversal::Blocked] {
                let (from_composition, _) =
                    tabulated(&w_packed, &a, m, n, traversal, OFFER_STEPS, OFFER_STEPS, 0);
                let (from_tier, census) =
                    tabulated(&w_tier, &a, m, n, traversal, OFFER_STEPS, OFFER_STEPS, 0);
                assert_eq!(
                    from_composition, from_tier,
                    "Sign<8> {m}x{k}x{n} at {traversal:?}: the two spellings of one \
                     decode must give the same bytes"
                );
                if traversal == Traversal::Tabulated {
                    assert!(
                        census.table_reads > 0,
                        "Sign<8> {m}x{k}x{n}: the offer was sized for a table and none \
                         was read ({census:?})"
                    );
                }
            }
        }
    }

    /// `CK-12`: the `Ternary` tier and the `Packed<Grid<4>,4>` spelling are two
    /// manifests for one decode. The gemm output is byte-identical through the
    /// packed route and through the table, at the shapes `CK-10` straddles the
    /// shared break-even with --- and the tier is run through the same
    /// every-offer gate the composition is.
    ///
    /// What the tier adds over the composition is not a different answer but
    /// the index stream: a `Ternary` code addresses its enumeration directly,
    /// so the tabulated run gathers from the operand's own memory instead of a
    /// stream it built. That is asserted where the claim lives, at the codec
    /// (`as_index_stream` answers the same slice it was handed); what this
    /// test watches is that the answer does not move with the spelling.
    #[test]
    fn the_ternary_tier_matches_the_composition_byte_for_byte_ck_12() {
        // The table `CK-10` spells the composition with, dead entry included:
        // digit 3 is one no ternary encoder emits, and it decodes to 0.
        let ternary_table: [A8; 4] = [
            Alphabet::of(-1),
            Alphabet::of(0),
            Alphabet::of(1),
            Alphabet::of(0),
        ];
        let composition =
            Packed::<_, 4>::new(Grid::<i8, Full<i8>, 4>::new(&ternary_table)).expect("4 divides 8");
        let tier = Ternary::<i8, Full<i8>, 4>::new().expect("the full alphabet admits 0 and +-1");

        for &(m, k, n) in &[
            (1usize, 4usize, 1usize),
            (3, 8, 1024),
            (5, 12, 1025),
            (4, 16, 1100),
        ] {
            // One stream, two spellings: the composition stores the byte, the
            // tier the same value zero-extended to a `u16` code.
            let bytes: Vec<u8> = fill(n * (k / 4), 0x7e7, |x| x as u8);
            let codes: Vec<u16> = bytes.iter().map(|&b| u16::from(b)).collect();

            // The tier against the dense driver's bytes at every offer --- the
            // gate `CK-10` runs the composition through, unchanged.
            every_traversal_agrees("Ternary<4>", tier, &codes, m, k, n);

            // And the two spellings against each other directly: equal decodes
            // under different kappa labels, so identical bytes (`CK-05`
            // restated for this pair), through the packed route and through
            // the table.
            let a: Vec<i8> = fill(m * k, 0xa11, |x| ((x % 255) as i64 - 127) as i8);
            let w_packed =
                CodedMatrix::new(composition, n, k, &bytes).expect("the codes describe n x k");
            let w_tier = CodedMatrix::new(tier, n, k, &codes).expect("the codes describe n x k");
            for traversal in [Traversal::Tabulated, Traversal::Blocked] {
                let (from_composition, _) =
                    tabulated(&w_packed, &a, m, n, traversal, OFFER_STEPS, OFFER_STEPS, 0);
                let (from_tier, census) =
                    tabulated(&w_tier, &a, m, n, traversal, OFFER_STEPS, OFFER_STEPS, 0);
                assert_eq!(
                    from_composition, from_tier,
                    "Ternary<4> {m}x{k}x{n} at {traversal:?}: the two spellings of one \
                     decode must give the same bytes"
                );
                if traversal == Traversal::Tabulated {
                    assert!(
                        census.table_reads > 0,
                        "Ternary<4> {m}x{k}x{n}: the offer was sized for a table and none \
                         was read ({census:?})"
                    );
                }
            }
        }
    }

    /// One tier at two widths, run through the whole offer gate at the narrow
    /// one and against its own wide spelling through the table and the dense
    /// route. `CK-15`'s per-tier body.
    fn both_widths_agree<C16, C8>(label: &str, tier16: C16, tier8: C8, m: usize, k: usize, n: usize)
    where
        C16: Enumerable<i8, Full<i8>, Code = u16> + Copy,
        C8: Enumerable<i8, Full<i8>, Code = u8> + Copy,
    {
        let block = <C8 as uor_matmul_codec::Codec<i8, Full<i8>>>::MAX_BLOCK;
        let bytes8: Vec<u8> = fill(n * (k / block), 0xc15, |x| x as u8);
        let codes16: Vec<u16> = bytes8.iter().map(|&b| u16::from(b)).collect();

        // The residency claim: the same stream, half the bytes.
        assert_eq!(
            core::mem::size_of_val(&*bytes8) * 2,
            core::mem::size_of_val(&*codes16),
            "{label}: the u8 stream is half the u16 stream's bytes"
        );

        // The byte stream is the index stream: borrowed, as `U8`, not built.
        let Some(uor_matmul_codec::IndexStream::U8(borrowed)) = C8::as_index_stream(&bytes8) else {
            panic!("{label}: a 256-entry space borrows its byte stream")
        };
        assert!(
            core::ptr::eq(borrowed.as_ptr(), bytes8.as_ptr()),
            "{label}: the stream must be borrowed, not built"
        );

        // The narrow spelling through the same every-offer gate the wide one
        // passes (`CK-13`'s ladder, unchanged).
        every_traversal_agrees(label, tier8, &bytes8, m, k, n);

        // And the two widths against each other directly, through the table
        // and through the dense route.
        let a: Vec<i8> = fill(m * k, 0xa11, |x| ((x % 255) as i64 - 127) as i8);
        let w8 = CodedMatrix::new(tier8, n, k, &bytes8).expect("the codes describe n x k");
        let w16 = CodedMatrix::new(tier16, n, k, &codes16).expect("the codes describe n x k");
        for traversal in [Traversal::Tabulated, Traversal::Blocked] {
            let (from8, _) = tabulated(&w8, &a, m, n, traversal, OFFER_STEPS, OFFER_STEPS, 0);
            let (from16, census) =
                tabulated(&w16, &a, m, n, traversal, OFFER_STEPS, OFFER_STEPS, 0);
            assert_eq!(
                from8, from16,
                "{label} {m}x{k}x{n} at {traversal:?}: the two widths of one tier must \
                 give the same bytes"
            );
            if traversal == Traversal::Tabulated {
                assert!(
                    census.table_reads > 0,
                    "{label} {m}x{k}x{n}: the offer was sized for a table and none was read \
                     ({census:?})"
                );
            }
        }
    }

    /// `CK-15`: the `u8` spelling of a 256-entry tier is the same codec at
    /// half the stream width. The decode is exhaustive over the shared code
    /// space, the residency is the literal `size_of` of the two streams, and
    /// the gather is pinned byte for byte through the tabulated driver at the
    /// shapes `CK-13` straddles the break-even with --- the narrow stream
    /// gathered from the operand's own memory (`CB-08` covers the sequences
    /// lane for lane), never re-spelled.
    #[test]
    fn the_u8_spelling_halves_the_stream_and_matches_byte_for_byte_ck_15() {
        let table = e8_table::<Full<i8>>().expect("i8 holds E8");
        let book16: Book<'_, i8, Full<i8>, 256, 8> = Book::new(&table);
        let book8: Book<'_, i8, Full<i8>, 256, 8, u8> = Book::new(&table);
        let sign16 = Sign::<i8, Full<i8>, 8>::new().expect("the full alphabet admits +-1");
        let sign8 = Sign::<i8, Full<i8>, 8, u8>::new().expect("the full alphabet admits +-1");
        let tern16 = Ternary::<i8, Full<i8>, 4>::new().expect("the full alphabet admits 0 and +-1");
        let tern8 =
            Ternary::<i8, Full<i8>, 4, u8>::new().expect("the full alphabet admits 0 and +-1");

        // Exhaustive over the shared code space: every byte decodes the block
        // its `u16` widening decodes.
        let (mut want, mut got) = ([A8::ZERO; 8], [A8::ZERO; 8]);
        for c in 0..=u8::MAX {
            assert_eq!(
                uor_matmul_codec::Codec::decode_into(&book16, u16::from(c), &mut want),
                8
            );
            assert_eq!(uor_matmul_codec::Codec::decode_into(&book8, c, &mut got), 8);
            assert_eq!(
                want, got,
                "Book<256,8> code {c:#04x} decodes differently at the two widths"
            );
            for t in 0..8 {
                assert_eq!(
                    uor_matmul_codec::Codec::decode_element(&sign16, u16::from(c), t),
                    uor_matmul_codec::Codec::decode_element(&sign8, c, t),
                    "Sign<8> code {c:#04x} element {t} decodes differently at the two widths"
                );
            }
            for t in 0..4 {
                assert_eq!(
                    uor_matmul_codec::Codec::decode_element(&tern16, u16::from(c), t),
                    uor_matmul_codec::Codec::decode_element(&tern8, c, t),
                    "Ternary<4> code {c:#04x} element {t} decodes differently at the two widths"
                );
            }
        }

        // The enumeration is the same enumeration at the narrow width: the
        // space is unchanged, and the byte is its own index.
        assert_eq!(
            <Book<'_, i8, Full<i8>, 256, 8, u8> as Enumerable<i8, Full<i8>>>::CODE_SPACE,
            256
        );
        for c in 0..=u8::MAX {
            assert_eq!(
                <Book<'_, i8, Full<i8>, 256, 8, u8> as Enumerable<i8, Full<i8>>>::index_of(c),
                c as usize
            );
            assert_eq!(
                <Sign<i8, Full<i8>, 8, u8> as Enumerable<i8, Full<i8>>>::index_of(c),
                c as usize
            );
            assert_eq!(
                <Ternary<i8, Full<i8>, 4, u8> as Enumerable<i8, Full<i8>>>::index_of(c),
                c as usize
            );
        }

        // End to end, at the shapes the table's break-even straddles: `k` is a
        // whole number of both tiers' blocks at each.
        for &(m, k, n) in &[
            (1usize, 8usize, 1usize),
            (5, 24, 683),
            (3, 8, 684),
            (4, 16, 700),
        ] {
            both_widths_agree("Book<256,8>", book16, book8, m, k, n);
            both_widths_agree("Sign<8>", sign16, sign8, m, k, n);
            both_widths_agree("Ternary<4>", tern16, tern8, m, k, n);
        }
    }

    /// `CK-12` at the bound the values live in: ternary weights are a subset of
    /// `{-1, 0, +1}`, so at `Bnd<1>` the tier runs the adds-only table build
    /// end to end --- the census's multiply count is zero, the same assertion
    /// `CB-10` makes of the sign composition.
    #[test]
    fn the_ternary_tier_at_bound_one_builds_with_adds_only_ck_12() {
        let tier = Ternary::<i8, Bnd<1>, 4>::new().expect("bound 1 admits 0 and +-1");

        // One column of one block: every closed form is exact, and the dead
        // digit is in the stream on purpose.
        let (m, k, n) = (1usize, 4usize, 1usize);
        let space = 256usize;
        let codes: Vec<u16> = fill(n * (k / 4), 0xc12, |x| x as u16);
        let w = CodedMatrix::new(tier, n, k, &codes).expect("the codes describe n x k");
        let a: Vec<i8> = fill(m * k, 0xa12, |x| if x & 1 == 0 { -1 } else { 1 });
        let (got, census) = tabulated(
            &w,
            &a,
            m,
            n,
            Traversal::Tabulated,
            OFFER_STEPS,
            OFFER_STEPS,
            0,
        );
        assert_eq!(
            got,
            reference(&w, &a, m, k, n),
            "and it is still the product"
        );
        assert_eq!(
            census.multiplies, 0,
            "at bound 1 the build is adds and subtracts: {census:?}"
        );
        assert_eq!(
            census.table_reads, 1,
            "one read per code per row: {census:?}"
        );
        assert_eq!(
            census.adds,
            1 + (m * k * space) as u64,
            "one add per read, and the build's `m * k * code_space` products charged as the \
             signed adds they are: {census:?}"
        );

        // A tile tall enough for the widest sequence, so the build that runs
        // is the ISA's where the host has one --- the census asks the same
        // question of it.
        let (m, k, n) = (16usize, 32usize, 3usize);
        let codes: Vec<u16> = fill(n * (k / 4), 0xc13, |x| x as u16);
        let w = CodedMatrix::new(tier, n, k, &codes).expect("the codes describe n x k");
        let a: Vec<i8> = fill(m * k, 0xa13, |x| if x & 1 == 0 { -1 } else { 1 });
        let (got, census) = tabulated(
            &w,
            &a,
            m,
            n,
            Traversal::Tabulated,
            OFFER_STEPS,
            OFFER_STEPS,
            0,
        );
        assert_eq!(
            got,
            reference(&w, &a, m, k, n),
            "and it is still the product"
        );
        assert_eq!(
            census.multiplies, 0,
            "at bound 1 the widest build is adds and subtracts too: {census:?}"
        );
        assert!(
            census.table_reads > 0,
            "the offer was sized for a table and none was read: {census:?}"
        );
    }

    /// `CB-10`, counted: a bound-1 tabulated run issues no multiply at all.
    ///
    /// `CU-06` counts the tabulated traversal at the full alphabet, where the
    /// build is the table's only multiply; this is the same census at bound 1,
    /// where the build is adds and subtracts and the census's multiply count
    /// is zero. The shape is one column of one block, so the collapse has
    /// nothing to do and the closed forms are exact. That the census *moves*
    /// is also the selection witness: only the adds-only build charges zero
    /// multiplies, so a zero here is `Backend::Auto` having selected it ---
    /// and this test failed with `multiplies: 2048` while the build still
    /// multiplied.
    #[test]
    fn the_bound1_table_build_issues_no_multiply_cb_10() {
        let sign_one: [Alphabet<i8, Bnd<1>>; 2] = [
            Alphabet::new(-1).expect("|-1| <= 1"),
            Alphabet::new(1).expect("|1| <= 1"),
        ];
        let sign = Packed::<_, 8>::new(Grid::<i8, Bnd<1>, 2>::new(&sign_one)).expect("8 divides 8");

        // One column of one block: every closed form is exact.
        let (m, k, n) = (1usize, 8usize, 1usize);
        let space = 256usize;
        let stream: Vec<u8> = fill(n * (k / 8), 0xc10, |x| x as u8);
        let w = CodedMatrix::new(sign, n, k, &stream).expect("the codes describe n x k");
        let a: Vec<i8> = fill(m * k, 0xa10, |x| if x & 1 == 0 { -1 } else { 1 });
        let (got, census) = tabulated(
            &w,
            &a,
            m,
            n,
            Traversal::Tabulated,
            OFFER_STEPS,
            OFFER_STEPS,
            0,
        );
        assert_eq!(
            got,
            reference(&w, &a, m, k, n),
            "and it is still the product"
        );
        assert_eq!(
            census.multiplies, 0,
            "at bound 1 the build is adds and subtracts: {census:?}"
        );
        assert_eq!(
            census.table_reads, 1,
            "one read per code per row: {census:?}"
        );
        assert_eq!(
            census.adds,
            1 + (m * k * space) as u64,
            "one add per read, and the build's `m * k * code_space` products charged as the \
             signed adds they are: {census:?}"
        );

        // A tile tall enough for the widest sequence, so the build that runs
        // is the ISA's where the host has one --- the census asks the same
        // question of it.
        let (m, k, n) = (16usize, 64usize, 3usize);
        let stream: Vec<u8> = fill(n * (k / 8), 0xc11, |x| x as u8);
        let w = CodedMatrix::new(sign, n, k, &stream).expect("the codes describe n x k");
        let a: Vec<i8> = fill(m * k, 0xa11, |x| if x & 1 == 0 { -1 } else { 1 });
        let (got, census) = tabulated(
            &w,
            &a,
            m,
            n,
            Traversal::Tabulated,
            OFFER_STEPS,
            OFFER_STEPS,
            0,
        );
        assert_eq!(
            got,
            reference(&w, &a, m, k, n),
            "and it is still the product"
        );
        assert_eq!(
            census.multiplies, 0,
            "at bound 1 the widest build is adds and subtracts too: {census:?}"
        );
        assert!(
            census.table_reads > 0,
            "the offer was sized for a table and none was read: {census:?}"
        );
    }

    /// `CD-16`: the column collapse applies at every column-block width, not
    /// only when the block is the whole output width.
    ///
    /// The accumulator offer is halved (and quartered) so `Plan::choose`
    /// resolves a column block narrower than `n`, and for `d` in `{1, 3}` the
    /// class representatives all sit in the *first* block --- so a later block
    /// can collapse only if its first occurrences are block-local. Two
    /// assertions:
    ///
    /// - the bytes are the dense driver's, at every width; and
    /// - the census is strictly below `m * n * (k / Bk)`, the count the
    ///   uncollapsed traversal issues and `CU-06` pins in closed form ---
    ///   because a repeated column is never charged twice within its block.
    ///
    /// Without the second assertion this test passed with the collapse
    /// silently disabled, which is exactly what it did before this ID existed.
    /// For `d = n` there is nothing to collapse and the count must be the
    /// closed form itself: nothing may be skipped either.
    #[test]
    fn collapsing_equal_columns_at_any_block_width_cannot_change_a_byte_cd_16() {
        let table = e8_table::<Full<i8>>().expect("i8 holds E8");
        let book = e8_codec(&table);
        let block = 8usize;
        for &(m, k, n) in &[(16usize, 64usize, 512usize), (8, 32, 512), (16, 64, 293)] {
            let a: Vec<i8> = fill(m * k, activation_salt(), |x| ((x % 255) as i64 - 127) as i8);
            let uncollapsed = (m * n * (k / block)) as u64;
            for d in [1usize, 3, n] {
                let d = d.min(n);
                let base: Vec<u16> = fill(d * (k / 8), 0xd0b, |x| (x % 400) as u16);
                let repeated: Vec<u16> = (0..n * (k / 8))
                    .map(|x| base[(x / (k / 8) % d) * (k / 8) + x % (k / 8)])
                    .collect();
                let w = CodedMatrix::new(book, n, k, &repeated).expect("the codes describe n x k");
                let want = reference(&w, &a, m, k, n);
                for acc_offer in [OFFER_STEPS / 2, OFFER_STEPS / 4] {
                    let (got, census) = tabulated(
                        &w,
                        &a,
                        m,
                        n,
                        Traversal::Tabulated,
                        acc_offer,
                        OFFER_STEPS,
                        0,
                    );
                    assert_eq!(
                        got, want,
                        "{m}x{k}x{n} d={d}: a column block narrower than the output must give \
                         the dense driver's bytes ({census:?})"
                    );
                    assert!(
                        census.table_reads > 0,
                        "{m}x{k}x{n} d={d}: the offer was sized for a table and none was read"
                    );
                    if d < n {
                        assert!(
                            census.table_reads < uncollapsed,
                            "{m}x{k}x{n} d={d} at an accumulator offer of \
                             {acc_offer}/{OFFER_STEPS}: a repeated column must not be charged \
                             twice within its block ({census:?} against {uncollapsed})"
                        );
                    } else {
                        assert_eq!(
                            census.table_reads, uncollapsed,
                            "{m}x{k}x{n} d={d}: nothing repeats, so nothing may be skipped"
                        );
                    }
                }
            }
        }
    }

    /// `CD-15`: collapsing equal rows of `A` in the tabulated traversal cannot
    /// change a byte, at every degeneracy and every offer.
    ///
    /// Row `i` of `A` is row `i % d`, so the operand has exactly `d` distinct
    /// rows and the table build may be charged `d * k * code_space` instead of
    /// `m * k * code_space` --- the closed form `CU-06` pins at `d = m`, which
    /// is also the charge when the collapse is offered nothing or finds
    /// nothing. Two assertions per case: the bytes are the dense driver's, and
    /// the census is the charge the offer admits. Without the second this test
    /// passed with the collapse silently inert, which is what it was before
    /// this ID existed.
    #[test]
    fn collapsing_equal_rows_of_a_cannot_change_a_byte_cd_15() {
        let table = e8_table::<Full<i8>>().expect("i8 holds E8");
        let book = e8_codec(&table);
        let space = 256usize;
        let block = 8usize;
        for &(m, k, n) in &[(8usize, 32usize, 64usize), (16, 64, 512), (7, 24, 40)] {
            let stream: Vec<u16> = fill(n * (k / block), 0xb00c, |x| (x % 400) as u16);
            let w = CodedMatrix::new(book, n, k, &stream).expect("the codes describe n x k");
            for d in [1usize, 2, m / 2, m] {
                let d = d.max(1).min(m);
                // `d` distinct rows, numbered by first occurrence: row `i` is
                // row `i % d`, and the first `d` are pairwise distinct.
                let a: Vec<i8> = (0..m * k)
                    .map(|x| {
                        let (row, col) = (x / k, x % k);
                        (((row % d) * 31 + col * 17) % 251) as i8
                    })
                    .collect();
                let want = reference(&w, &a, m, k, n);
                // No offer, a rows offer too short for the distinct rows,
                // exactly enough, and the worst case.
                for rows_offer in [0usize, k / 2, d * k, m * k] {
                    let (got, census) = tabulated(
                        &w,
                        &a,
                        m,
                        n,
                        Traversal::Tabulated,
                        OFFER_STEPS,
                        OFFER_STEPS,
                        rows_offer,
                    );
                    assert_eq!(
                        got, want,
                        "{m}x{k}x{n} d={d} rows offer {rows_offer}: the collapse must give \
                         the dense driver's bytes ({census:?})"
                    );
                    assert!(
                        census.table_reads > 0,
                        "{m}x{k}x{n} d={d}: the offer was sized for a table and none was read"
                    );
                    // The lookup build charges each product as an add. The
                    // collapse still changes only the number of distinct rows
                    // that are built, never the bytes or the table identity.
                    let charged = if rows_offer >= d * k && d < m { d } else { m };
                    assert_eq!(
                        census.multiplies, 0,
                        "{m}x{k}x{n} d={d} rows offer {rows_offer}: the i8 lookup build \
                         must issue no multiplies ({census:?})"
                    );
                    assert_eq!(
                        census.adds - census.table_reads,
                        (charged * k * space) as u64,
                        "{m}x{k}x{n} d={d} rows offer {rows_offer}: the lookup build must \
                         charge each product as an add ({census:?})"
                    );
                }
            }
        }

        // Both collapse axes at once: two distinct rows of `A` against a `W`
        // with four distinct columns. The two expansions read their own
        // indices, and the census says so from both sides at once.
        let (m, k, n) = (16usize, 64usize, 512usize);
        let (rows_d, cols_d) = (2usize, 4usize);
        let a: Vec<i8> = (0..m * k)
            .map(|x| {
                let (row, col) = (x / k, x % k);
                (((row % rows_d) * 31 + col * 17) % 251) as i8
            })
            .collect();
        let base: Vec<u16> = fill(cols_d * (k / block), 0xd0b, |x| (x % 400) as u16);
        let stream: Vec<u16> = (0..n * (k / block))
            .map(|x| base[(x / (k / block) % cols_d) * (k / block) + x % (k / block)])
            .collect();
        let w = CodedMatrix::new(book, n, k, &stream).expect("the codes describe n x k");
        let want = reference(&w, &a, m, k, n);
        let (got, census) = tabulated(
            &w,
            &a,
            m,
            n,
            Traversal::Tabulated,
            OFFER_STEPS,
            OFFER_STEPS,
            rows_d * k,
        );
        assert_eq!(
            got, want,
            "both axes: the dense driver's bytes ({census:?})"
        );
        assert_eq!(
            census.multiplies, 0,
            "both axes: the i8 lookup build issues no multiplies ({census:?})"
        );
        assert_eq!(
            census.adds - census.table_reads,
            (rows_d * k * space) as u64,
            "both axes: the lookup build charges each product as an add ({census:?})"
        );
        assert_eq!(
            census.table_reads,
            (rows_d * cols_d * (k / block)) as u64,
            "both axes: the column loop charges per distinct column ({census:?})"
        );
    }

    /// One `i32` tabulated product at one encode mode and one accumulator
    /// offer, with the panel holding the decoded codebook and nothing more.
    ///
    /// The `i32` lanes --- the exact accumulator and the modular word alike ---
    /// live in the accumulator offer, so there is one knob and the census says
    /// which lane it admitted.
    fn tabulated_i32<C: Enumerable<i32, Full<i32>> + Copy>(
        w: &CodedMatrix<'_, i32, Full<i32>, C>,
        a: &[i32],
        m: usize,
        n: usize,
        encode: EncodeMode,
        acc_offer: usize,
    ) -> (Vec<i32>, Census) {
        let k = w.cols();
        let block = <C as uor_matmul_codec::Codec<i32, Full<i32>>>::MAX_BLOCK;
        let mut panel = vec![
            Alphabet::<i32, Full<i32>>::ZERO;
            suggested_tabulation_panel(C::CODE_SPACE, block)
        ];
        let mut accumulators = vec![<AccOf<i32> as Accumulator>::ZERO; acc_offer];
        let mut c = vec![0i32; m * n];
        let mut census = Census::default();
        {
            let av = MatView::row_major(as_alphabet_full(a), m, k).unwrap();
            let cv = MatViewMut::row_major(&mut c, m, n).unwrap();
            let mut tr = TabulatedTriple::new(av, *w, cv).unwrap();
            gemm_tabulated_counted(
                &mut tr,
                &Linear::OVERWRITE,
                GemmOptions {
                    traversal: Traversal::Tabulated,
                    encode,
                    ..Default::default()
                },
                &mut Scratch::with_accumulators(&mut panel, &mut accumulators),
                &mut Tabulation::none(),
                &mut Collapse::none(),
                &mut census,
            );
        }
        (c, census)
    }

    /// The dense product of the same `i32` operands at the same encode mode,
    /// by the driver whose bytes every other traversal is measured against.
    fn reference_i32<C: Enumerable<i32, Full<i32>> + Copy>(
        w: &CodedMatrix<'_, i32, Full<i32>, C>,
        a: &[i32],
        m: usize,
        k: usize,
        n: usize,
        encode: EncodeMode,
    ) -> Vec<i32> {
        let mut b = vec![0i32; k * n];
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
            GemmOptions {
                traversal: Traversal::Blocked,
                encode,
                ..Default::default()
            },
            &mut Scratch::none(),
        );
        c
    }

    /// The shape, codec, operands, and offer the two modular-lane tests share.
    ///
    /// The offer is the point of the fixture. The exact `i32` lane *is* the
    /// accumulator, so a table in it costs one accumulator per lane word ---
    /// `slab * rows * depth` of them past the `rows * n` tile. The modular
    /// lane is four lane words per accumulator, so its smallest one-row,
    /// one-column, one-slot plan needs one exact output cell plus
    /// `ceil((one carried column + slab) / 4)` accumulator words, while the
    /// exact lane needs one output plus that carried column and slab one for
    /// one. The offer below covers the former and not the latter even after the
    /// shared-offer planner searches narrower plans. The census then says which
    /// lane ran.
    #[allow(clippy::type_complexity)]
    fn modular_i32_fixture() -> (
        [[Alphabet<i32, Full<i32>>; 4]; 16],
        Vec<u16>,
        Vec<i32>,
        usize,
    ) {
        let (m, k, n) = (16usize, 8usize, 64usize);
        let (space, block) = (16usize, 4usize);
        let rows = 16usize;
        assert!(
            tabulation_rows(space, blocking::L1_BYTES, core::mem::size_of::<Mod32>()) >= rows
                && tabulation_rows(space, blocking::L1_BYTES, core::mem::size_of::<i128>()) >= rows,
            "the fixture's row tile fits L1 in either lane"
        );
        let slab = slab_codes(space);
        let offer = 1 + (1 + slab).div_ceil(4);
        assert!(
            offer < 1 + 1 + slab,
            "the exact lane's smallest plan cannot fit"
        );
        let flat: Vec<i32> = fill(space * block, 0xb32c, |x| {
            (x as i32).wrapping_mul(0x9E37_79B9u32 as i32)
        });
        let cells: [[Alphabet<i32, Full<i32>>; 4]; 16] =
            core::array::from_fn(|c| core::array::from_fn(|t| Alphabet::of(flat[c * block + t])));
        // Full-range activations: the modular lane declares no bound, so the
        // extremes of the alphabet are ordinary inputs and a product of two of
        // them wraps on purpose.
        let a: Vec<i32> = fill(m * k, 0xa32c, |x| {
            (x as i32).wrapping_mul(0x85EB_CA6Bu32 as i32)
        });
        let stream: Vec<u16> = fill(n * (k / block), 0xc32c, |x| (x % 16) as u16);
        (cells, stream, a, offer)
    }

    /// `CU-08`: the modular table lane runs exactly when the encode is
    /// `Wrapping` and the output is no wider than the lane, and its depth is
    /// unbounded at every bound.
    ///
    /// The depth half is the table-side form of what `CU-02` pins for the
    /// dense lane: the wrap *is* the encode, so there is nothing to chunk.
    #[test]
    fn the_modular_table_lane_runs_exactly_when_the_encode_admits_it_cu_08() {
        for bound in [0u128, 1, 1 << 10, 1 << 31, u128::MAX] {
            assert_eq!(<Mod32 as Lane<i32>>::capacity(bound), None);
        }
        for bound in [0u128, 1, 1 << 10, 1 << 63, u128::MAX] {
            assert_eq!(<Mod64 as Lane<i64>>::capacity(bound), None);
        }
        // The specs say the same, at every tile the driver walks: the lane's
        // depth is `usize::MAX` at every bound, exactly as
        // `Factorization::Modular` declares on the dense side.
        for &(rows, group) in &[(16usize, 1usize), (8, 2), (1, 1)] {
            let spec32 = choose_table(
                available_table_i32_modular(rows, group),
                Backend::Auto,
                1 << 31,
                4,
            )
            .expect("the reference sequence is always present");
            assert_eq!(spec32.lane_depth(1 << 31), usize::MAX);
            assert_eq!(spec32.lane_depth(1), usize::MAX);
            let spec64 = choose_table(
                available_table_i64_modular(rows, group),
                Backend::Auto,
                1 << 63,
                4,
            )
            .expect("the reference sequence is always present");
            assert_eq!(spec64.lane_depth(1 << 63), usize::MAX);
            assert_eq!(spec64.lane_depth(1), usize::MAX);
        }

        // The width half of admissibility, per family: an output no wider than
        // the lane. `i8` and `i16` offer no modular table lane at all --- their
        // exact lane already holds every depth a weight row reaches, so a
        // quotient read would buy nothing.
        assert!(<i32 as Tabulated>::modular_table_admitted(8));
        assert!(<i32 as Tabulated>::modular_table_admitted(16));
        assert!(<i32 as Tabulated>::modular_table_admitted(32));
        assert!(!<i32 as Tabulated>::modular_table_admitted(64));
        assert!(<i64 as Tabulated>::modular_table_admitted(32));
        assert!(<i64 as Tabulated>::modular_table_admitted(64));
        assert!(!<i64 as Tabulated>::modular_table_admitted(128));
        assert!(!<i8 as Tabulated>::modular_table_admitted(32));
        assert!(!<i16 as Tabulated>::modular_table_admitted(32));

        // The encode-mode half, behaviourally. At the fixture's offer the
        // modular lane holds a table and the exact lane cannot (see
        // `modular_i32_fixture`), so which traversal ran is the census's to
        // say: under `Wrapping` the table runs, under `Saturating` the same
        // call streams --- and both give the dense driver's bytes at their own
        // encode mode.
        let (m, k, n) = (16usize, 8usize, 64usize);
        let (cells, stream, a, offer) = modular_i32_fixture();
        let book = Book::new(&cells);
        let w = CodedMatrix::new(book, n, k, &stream).expect("the codes describe n x k");
        let (wrapped, wrap_census) = tabulated_i32(&w, &a, m, n, EncodeMode::Wrapping, offer);
        assert_eq!(
            wrapped,
            reference_i32(&w, &a, m, k, n, EncodeMode::Wrapping),
            "the modular table must give the dense driver's bytes"
        );
        assert_eq!(
            wrap_census.table_reads,
            (m * n * (k / 4)) as u64,
            "under `Wrapping` the modular table ran, one read per code per row: {wrap_census:?}"
        );
        for mode in [EncodeMode::Nearest, EncodeMode::Saturating] {
            let (got, census) = tabulated_i32(&w, &a, m, n, mode, offer);
            assert_eq!(
                got,
                reference_i32(&w, &a, m, k, n, mode),
                "under {mode:?} the stream must give the dense driver's bytes"
            );
            assert_eq!(
                census.table_reads, 0,
                "under {mode:?} no modular lane is admitted and the exact one cannot fit \
                 this offer, so the call streams: {census:?}"
            );
        }
    }

    /// `CB-09`, end to end: through the driver, the modular table lane gives
    /// the dense modular traversal's bytes.
    ///
    /// The parity half in `uor-matmul-kernels` reads every modular sequence
    /// against the model lane for lane; this half reads the *dispatch*. At
    /// every offer --- nothing, a fraction, the fixture's sized offer, and
    /// beyond it --- the bytes are the dense driver's, and at the sized offer
    /// the census says the table, not the stream, produced them.
    #[test]
    fn the_modular_table_lane_gives_the_dense_drivers_bytes_cb_09() {
        let (m, k, n) = (16usize, 8usize, 64usize);
        let (cells, stream, a, offer) = modular_i32_fixture();
        let book = Book::new(&cells);
        let w = CodedMatrix::new(book, n, k, &stream).expect("the codes describe n x k");
        let want = reference_i32(&w, &a, m, k, n, EncodeMode::Wrapping);
        for offer in [0usize, offer / 2, offer, offer * 2] {
            let (got, census) = tabulated_i32(&w, &a, m, n, EncodeMode::Wrapping, offer);
            assert_eq!(
                got, want,
                "an accumulator offer of {offer} must give the dense driver's bytes ({census:?})"
            );
        }
        let (_, census) = tabulated_i32(&w, &a, m, n, EncodeMode::Wrapping, offer);
        assert_eq!(
            census.table_reads,
            (m * n * (k / 4)) as u64,
            "the offer was sized for the modular table and none was read: {census:?}"
        );
    }

    /// The `i64` half of `CB-09`'s dispatch read: the portable-only lane,
    /// through the same boundary.
    ///
    /// The lane is three `Mod64` words to the three-limb accumulator, so the
    /// sizing argument is `modular_i32_fixture`'s with a third in place of a
    /// quarter. Without this half `Mod64::place` is code no test can fail.
    #[test]
    fn the_modular_i64_table_lane_gives_the_dense_drivers_bytes_cb_09() {
        let (m, k, n) = (16usize, 8usize, 64usize);
        let (space, block) = (16usize, 4usize);
        let rows = 16usize;
        assert!(
            tabulation_rows(space, blocking::L1_BYTES, core::mem::size_of::<Mod64>()) >= rows
                && tabulation_rows(
                    space,
                    blocking::L1_BYTES,
                    core::mem::size_of::<AccOf<i64>>()
                ) >= rows,
            "the fixture's row tile fits L1 in either lane"
        );
        let slab = slab_codes(space);
        let tile = rows * n;
        let offer = tile + (tile + slab * rows * (k / block)).div_ceil(3);
        let flat: Vec<i64> = fill(space * block, 0xb64c, |x| {
            (x as i64).wrapping_mul(0x9E37_79B9_7F4A_7C15u64 as i64)
        });
        let cells: [[Alphabet<i64, Full<i64>>; 4]; 16] =
            core::array::from_fn(|c| core::array::from_fn(|t| Alphabet::of(flat[c * block + t])));
        let a: Vec<i64> = fill(m * k, 0xa64c, |x| {
            (x as i64).wrapping_mul(0x2545_F491_4F6C_DD1Du64 as i64)
        });
        let stream: Vec<u16> = fill(n * (k / block), 0xc64c, |x| (x % 16) as u16);
        let book = Book::new(&cells);
        let w = CodedMatrix::new(book, n, k, &stream).expect("the codes describe n x k");

        // The dense product at the same encode mode, by the same driver the
        // `i32` half is read against.
        let want = {
            let mut b = vec![0i64; k * n];
            for p in 0..k {
                for j in 0..n {
                    b[p * n + j] = w.at(j, p).get();
                }
            }
            let mut c = vec![0i64; m * n];
            let av = MatView::row_major(as_alphabet_full(&a), m, k).unwrap();
            let bv = MatView::row_major(as_alphabet_full(&b), k, n).unwrap();
            let cv = MatViewMut::row_major(&mut c, m, n).unwrap();
            let mut t = Triple::new(av, bv, cv).unwrap();
            gemm(
                &mut t,
                &Linear::OVERWRITE,
                GemmOptions {
                    traversal: Traversal::Blocked,
                    encode: EncodeMode::Wrapping,
                    ..Default::default()
                },
                &mut Scratch::none(),
            );
            c
        };

        let mut panel =
            vec![Alphabet::<i64, Full<i64>>::ZERO; suggested_tabulation_panel(space, block)];
        let mut accumulators = vec![<AccOf<i64> as Accumulator>::ZERO; offer];
        let mut c = vec![0i64; m * n];
        let mut census = Census::default();
        {
            let av = MatView::row_major(as_alphabet_full(&a), m, k).unwrap();
            let cv = MatViewMut::row_major(&mut c, m, n).unwrap();
            let mut tr = TabulatedTriple::new(av, w, cv).unwrap();
            gemm_tabulated_counted(
                &mut tr,
                &Linear::OVERWRITE,
                GemmOptions {
                    traversal: Traversal::Tabulated,
                    encode: EncodeMode::Wrapping,
                    ..Default::default()
                },
                &mut Scratch::with_accumulators(&mut panel, &mut accumulators),
                &mut Tabulation::none(),
                &mut Collapse::none(),
                &mut census,
            );
        }
        assert_eq!(
            c, want,
            "the modular i64 table must give the dense driver's bytes ({census:?})"
        );
        assert_eq!(
            census.table_reads,
            (m * n * (k / block)) as u64,
            "the offer was sized for the modular table and none was read: {census:?}"
        );
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

        let (got, census) = tabulated(
            &w,
            &a,
            1,
            n,
            Traversal::Tabulated,
            OFFER_STEPS,
            OFFER_STEPS,
            0,
        );
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
    /// - `adds == table_reads + m * k * code_space`. The first term is the
    ///   column loop's one read and one add per code; the second is the lookup
    ///   build's one add per product.
    /// - `decodes == code_space * Bk`, for the whole call. The codebook is
    ///   decoded once, not once per row tile and per block of the reduction, so
    ///   the codec's cost does not scale with the shape at all.
    /// - `multiplies == 0`: both the lookup build and the column loop are
    ///   multiply-free, independent of `n`.
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

        // `Tabulated`, not `Blocked`. These are closed forms *of the tabulated
        // column loop*, so the loop has to run for them to be counted --- and
        // whether `Blocked` selects it is a question about the host, not about the
        // loop. On a VNNI runner the dense tile is four times denser per
        // instruction while the table is not, so `tabulation_pays` correctly
        // declines at this shape and the census came back
        // `adds: 0, table_reads: 0, kernel_calls: 1`. That is the predicate being
        // right, and it made this test fail on whichever CI runner happened to
        // have VNNI while passing on the ones that did not. The predicate's own
        // claim is `CM-04`'s, and it asserts that VNNI case directly.
        let (got, census) = tabulated(
            &w,
            &a,
            m,
            n,
            Traversal::Tabulated,
            OFFER_STEPS,
            OFFER_STEPS,
            0,
        );
        assert_eq!(
            got,
            reference(&w, &a, m, k, n),
            "and it is still the product"
        );

        assert_eq!(
            census.adds,
            (m * n * blocks + m * k * space) as u64,
            "one gather add per code plus one lookup-build add per product: {census:?}"
        );
        assert_eq!(
            census.adds - census.table_reads,
            (m * k * space) as u64,
            "the lookup build contributes one add per product: {census:?}"
        );
        assert_eq!(
            census.decodes,
            (space * block) as u64,
            "the codebook is decoded once for the whole call, not once per tile: {census:?}"
        );
        assert_eq!(
            census.multiplies, 0,
            "the lookup build is multiply-free and does not scale with n: {census:?}"
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
        let (streamed, plain) = tabulated(
            &w,
            &a,
            m,
            n,
            Traversal::OutputMajor,
            OFFER_STEPS,
            OFFER_STEPS,
            0,
        );
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
        // The declarations the recorded break-evens are written for: a 256-bit
        // register of `i32` on the table's side, and what `AVX2_I8_I32` declares
        // on the dense side. `CM-04`'s model half recomputes the same numbers
        // from the same two, so a change to either fails on both sides at once.
        let steps = |block: usize| Steps {
            table: block * (256 / 8 / core::mem::size_of::<i32>()),
            dense: blocking::KERNEL_PRODUCTS_PER_STEP,
            dense_rows: blocking::KERNEL_ROWS,
        };

        // E8 at the shipped tile: the first `n` that pays is 683, and 682 does
        // not.
        assert!(tabulation_pays(256, 8, 683, rows, steps(8), l1, lane));
        assert!(!tabulation_pays(256, 8, 682, rows, steps(8), l1, lane));
        // A nibble pair covers `2 * lanes_per_add` products per issued
        // instruction, which is exactly what one dense instruction covers, so
        // nothing repays the build and no `n` makes it pay.
        assert_eq!(steps(2).table, blocking::KERNEL_PRODUCTS_PER_STEP);
        assert!(!tabulation_pays(
            256,
            2,
            usize::MAX,
            rows,
            steps(2),
            l1,
            lane
        ));
        // One element per code: likewise, and for the same reason at `block = 1`.
        assert!(!tabulation_pays(
            16,
            1,
            usize::MAX,
            rows,
            steps(1),
            l1,
            lane
        ));
        // A table nobody can hold is refused whatever the instruction count says.
        assert!(!tabulation_pays(
            1 << 16,
            8,
            usize::MAX,
            1,
            steps(8),
            l1,
            lane
        ));
        assert_eq!(tabulation_rows(1 << 16, l1, lane), 0);
        // And an enumeration of nothing has no table.
        assert!(!tabulation_fits(0, 1, l1, lane));

        // The register width is not decoration. A sequence with no vector add
        // covers one lane per instruction, and then no `n` repays the build ---
        // which is the honest statement that a table is worth building only
        // where the adds are vectors.
        let reference = Steps {
            table: 8,
            dense: blocking::KERNEL_PRODUCTS_PER_STEP,
            dense_rows: blocking::KERNEL_ROWS,
        };
        assert!(!tabulation_pays(
            256,
            8,
            usize::MAX,
            rows,
            reference,
            l1,
            lane
        ));
        // And a dense tile four times denser per instruction --- VNNI --- is not
        // beaten by an AVX2 table at any `n`. The old form, which priced the
        // table at `block * rows` and the tile at a constant, took it from 683.
        let vnni = Steps {
            table: 8 * 8,
            dense: 4 * blocking::KERNEL_PRODUCTS_PER_STEP,
            dense_rows: blocking::KERNEL_ROWS,
        };
        assert!(!tabulation_pays(256, 8, usize::MAX, rows, vnni, l1, lane));
    }

    /// `CM-04`: the shipped selector prices the build sequence that will
    /// actually run, including a build density different from gather density,
    /// and its three-factor comparison stays exact at the address boundary.
    #[test]
    fn the_private_planner_prices_build_density_without_saturation_cm_04() {
        for a in [0usize, 1, 2, 7, 16] {
            for b in [0usize, 1, 2, 7, 16] {
                for c in [0usize, 1, 2, 7, 16] {
                    for x in [0usize, 1, 2, 7, 16] {
                        for y in [0usize, 1, 2, 7, 16] {
                            for z in [0usize, 1, 2, 7, 16] {
                                assert_eq!(
                                    CostProduct::of(a, b, c).greater_than(CostProduct::of(x, y, z)),
                                    (a as u128) * (b as u128) * (c as u128)
                                        > (x as u128) * (y as u128) * (z as u128)
                                );
                            }
                        }
                    }
                }
            }
        }
        assert!(
            CostProduct::of(usize::MAX, usize::MAX, usize::MAX).greater_than(CostProduct::of(
                usize::MAX,
                usize::MAX,
                usize::MAX - 1
            ))
        );
        assert!(!CostProduct::of(usize::MAX, usize::MAX - 1, usize::MAX)
            .greater_than(CostProduct::of(usize::MAX, usize::MAX, usize::MAX)));

        let mut spec = portable_table::<i8, i32>(1, 1);
        let steps = Steps {
            table: 4,
            dense: 1,
            dense_rows: 1,
        };
        spec.build_products_per_step = 1;
        assert!(
            !tabulation_pays_for_spec(
                16,
                4,
                11,
                1,
                steps,
                usize::MAX,
                core::mem::size_of::<i32>(),
                &spec,
            ),
            "3*11 does not repay a 16-code build at q=1"
        );
        spec.build_products_per_step = 2;
        assert!(
            tabulation_pays_for_spec(
                16,
                4,
                11,
                1,
                steps,
                usize::MAX,
                core::mem::size_of::<i32>(),
                &spec,
            ),
            "6*11 repays the same build at q=2"
        );
        spec.build_products_per_step = 0;
        assert!(!tabulation_pays_for_spec(
            16,
            4,
            usize::MAX,
            1,
            steps,
            usize::MAX,
            core::mem::size_of::<i32>(),
            &spec,
        ));
    }

    /// `CG-18`: the performance gate, counted rather than timed.
    ///
    /// A wall-clock gate on shared CI would measure the machine as much as the
    /// library, which is why the CG-* figures are `open`. What is not a
    /// measurement is the operation census: which factorization selection ran,
    /// and what it issued. This test is the regression gate on both, and it is
    /// the same claim twice ---
    ///
    /// - Selection is the derivation. The break-even is recomputed at test
    ///   time from the declarations the host's own sequences make (the table
    ///   spec's independent `build_products_per_step` and `lanes_per_add`
    ///   declarations against the dense tile's numbers, at the model's cache
    ///   and blocking constants), never from a recorded one. The ISA rows of
    ///   `model/tiers.toml` differ because the sequence declarations do. Below
    ///   the first column width the predicate pays at, the census must show the
    ///   dense route; at and above it, the table.
    /// - The gather never multiplies. When the table runs, the census's
    ///   multiplies are exactly the build's charge --- `code_space * block *
    ///   rows` per slot of the stack --- and one more is the asymptotic
    ///   regression this gate exists to catch.
    ///
    /// The boundary the other way is the integer Grid's product build: one
    /// element per code saves no operation, so the reference cost predicate
    /// declines it at every `n`. The separately measured contextual Scaled64
    /// declaration is intentionally not generalized to this builder.
    #[test]
    fn selection_is_the_derivation_and_the_gather_never_multiplies_cg_18() {
        let table = e8_table::<Full<i8>>().expect("i8 holds E8");
        let book = e8_codec(&table);
        let book8 = e8_codec_u8(&table);
        // E8's own shape, as the codec declares it and `model/tiers.toml`
        // records it.
        let space = 256usize;
        let block = 8usize;
        // Two code blocks keep every column distinct through the largest
        // shipped break-even while avoiding work that does not strengthen the
        // boundary assertion.
        let (m, k) = (16usize, 16usize);
        let blocks = k / block;
        let lane = core::mem::size_of::<<i8 as Tabulated>::Lane>();
        let l1 = blocking::L1_BYTES;

        // The derivation, at the host's own pair of declarations: the table
        // sequence the family resolves to against the dense tile it would
        // otherwise run. This is `run_lane`'s own arithmetic, including the
        // independent build density; a boundary computed from a recorded
        // number would pass on one ISA and be wrong on every other.
        let bound = <Full<i8> as Bound>::VALUE;
        let backend = GemmOptions::default().backend;
        let rows = ROW_TILES
            .into_iter()
            .find(|&r| r <= tabulation_rows(space, l1, lane).min(m))
            .expect("a 256-entry table fits L1 at some tile");
        let spec =
            <i8 as Tabulated>::table_spec(backend, bound, false, rows, column_group(rows), block);
        let steps =
            <i8 as Tabulated>::dense_steps(backend, bound, rows, block * spec.lanes_per_add);
        let break_even = (1..)
            .find(|&cols| {
                tabulation_pays_for_spec(space, block, cols, rows, steps, l1, lane, &spec)
            })
            .expect("E8 pays at some column width on every pair this workspace ships");
        assert!(
            !tabulation_pays_for_spec(space, block, break_even - 1, rows, steps, l1, lane, &spec,),
            "the first paying width is a boundary, not a plateau"
        );

        // Sixteen rows of activations; the values are immaterial to a count,
        // and the product is asserted against the dense driver's bytes.
        let a: Vec<i8> = (0..m * k).map(|i| ((i % 255) as i64 - 127) as i8).collect();

        for n in [break_even - 1, break_even, break_even + 1] {
            // Column `j` is `j` in base `space`, so no two columns repeat and
            // the gather's closed form is exact rather than a census of a
            // hash's luck.
            let stream: Vec<u16> = (0..n * blocks)
                .map(|i| {
                    let (j, p) = (i / blocks, i % blocks);
                    // The `p`-th base-`space` digit of `j`, by repeated
                    // division: `space.pow(p)` overflows a 32-bit `usize`
                    // (wasm32) at `p = 4`, and the wrap reads as a divide by
                    // zero.
                    let shifted = (0..p).fold(j, |acc, _| acc / space);
                    (shifted % space) as u16
                })
                .collect();
            let w = CodedMatrix::new(book, n, k, &stream).expect("the codes describe n x k");
            let (got, census) = tabulated(
                &w,
                &a,
                m,
                n,
                Traversal::Blocked,
                OFFER_STEPS,
                OFFER_STEPS,
                0,
            );
            assert_eq!(
                got,
                reference(&w, &a, m, k, n),
                "the product, whichever factorization ran ({census:?})"
            );

            // The plan the traversal resolved to, recomputed from the same
            // offers the helper passes: the accumulator offer is the suggested
            // one, the lane offer the suggested `i64` words re-read as lanes
            // (`run_lane`'s own conversion).
            let shape = Shape { m, k, n };
            let plan = Plan::choose(
                space,
                shape,
                lane,
                suggested_tabulation::<i8, Full<i8>>(shape, space, block).max(1),
                suggested_tabulation_lanes::<i8, Full<i8>>(shape, space, block).max(1) * 8 / lane,
                block,
                <i8 as Tabulated>::probe_capacity::<<i8 as Tabulated>::Lane>(bound),
            )
            .expect("the suggested offers admit a plan");
            assert_eq!(
                plan.cols, n,
                "the offer must be wide enough that the amortization axis is `n` itself"
            );
            assert_eq!(plan.rows, rows, "the tile the derivation priced");

            if n < break_even {
                assert_eq!(
                    census.table_reads, 0,
                    "below the break-even ({break_even}) the table must not run at n = {n}: {census:?}"
                );
                assert!(
                    census.kernel_calls > 0,
                    "and the dense route is what ran at n = {n}: {census:?}"
                );
            } else {
                assert!(
                    census.table_reads > 0,
                    "at and above the break-even ({break_even}) the table runs at n = {n}: {census:?}"
                );
                assert_eq!(
                    census.kernel_calls, 0,
                    "no dense route at n = {n}: {census:?}"
                );
                assert_eq!(
                    census.table_reads,
                    (m * n * blocks) as u64,
                    "one read per code at n = {n}: {census:?}"
                );
                assert_eq!(
                    census.adds - census.table_reads,
                    (m * k * space) as u64,
                    "the lookup build contributes one add per product at n = {n}: {census:?}"
                );
                assert_eq!(
                    census.multiplies, 0,
                    "the lookup build and gather issue no multiplies at n = {n}: {census:?}"
                );
            }

            // The same closed forms at the byte width (`CK-15`): a gather read
            // is a gather read whatever the code word's width, so the census
            // is the same census, and the bytes are the same bytes.
            let stream8: Vec<u8> = stream.iter().map(|&c| c as u8).collect();
            let w8 = CodedMatrix::new(book8, n, k, &stream8).expect("the codes describe n x k");
            let (got8, census8) = tabulated(
                &w8,
                &a,
                m,
                n,
                Traversal::Blocked,
                OFFER_STEPS,
                OFFER_STEPS,
                0,
            );
            assert_eq!(got8, got, "the u8 spelling's bytes at n = {n}");
            assert_eq!(
                census8, census,
                "the census is width-independent at n = {n}: {census8:?} against {census:?}"
            );
        }

        // The boundary the other way: this integer Grid's block-one product
        // build has no measured contextual contraction to replace, so the
        // derivation declines it at every `n`.
        let i4: [A8; 16] = core::array::from_fn(|i| Alphabet::of((i as i8) - 8));
        let grid = Grid::<i8, Full<i8>, 16>::new(&i4);
        for n in [1usize, break_even - 1, break_even, break_even + 1] {
            let stream: Vec<u16> = fill(n * k, 0x61d, |x| (x % 16) as u16);
            let w = CodedMatrix::new(grid, n, k, &stream).expect("the codes describe n x k");
            let (got, census) = tabulated(
                &w,
                &a,
                m,
                n,
                Traversal::Blocked,
                OFFER_STEPS,
                OFFER_STEPS,
                0,
            );
            assert_eq!(got, reference(&w, &a, m, k, n));
            assert_eq!(
                census.table_reads, 0,
                "the integer Grid product build is declined at n={n}: {census:?}"
            );
            assert!(
                census.kernel_calls > 0,
                "and the dense route is what ran at n = {n}: {census:?}"
            );
        }
    }

    /// Which factorization ran, read from the census.
    ///
    /// The census's *counts* move with the operand's degeneracy --- that is what
    /// `CD-15` and `CD-16` are about, and it is a different claim --- so what is
    /// read here is which of the three routes issued anything at all.
    fn ran(census: &Census) -> Traversal {
        if census.table_reads > 0 {
            Traversal::Tabulated
        } else if census.kernel_calls > 0 {
            Traversal::Blocked
        } else {
            Traversal::OutputMajor
        }
    }

    /// A recorded 64-bit hash of a code stream, rendered in the manifest's
    /// digest shape.
    ///
    /// Not SHA-256, and not offered as one. What `CS-10` needs of a digest field
    /// is that it *moves with the bytes*, and a cryptographic digest would need
    /// a dependency this workspace does not carry.
    /// [`Manifest::write_canonical_json`] checks the shape and nothing else,
    /// which is what makes the stand-in usable and what makes the manifests
    /// below genuinely different rather than declared different.
    fn digest_of(codes: &[u16]) -> String {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for &c in codes {
            for b in c.to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        // Sixteen hex digits, four times: the field is 64 lowercase hex
        // characters wide and this fills it from the one number it has.
        let word = format!("{h:016x}");
        format!("sha256:{word}{word}{word}{word}")
    }

    /// `CS-10`: the traversal is selected from the coded operand's declaration,
    /// and that declaration is what the artifact's address is minted from.
    ///
    /// Both directions, because a one-sided version passes for a selector that
    /// reads neither the declaration nor the operand:
    ///
    /// - Hold the declaration and move the *values*. Five operands of one shape,
    ///   at the extremes of the code space and of the alphabet and at a recorded
    ///   fill between them, on both sides of the break-even. Each mints a
    ///   different artifact --- the digest is the manifest's one field that moves
    ///   with the bytes --- and each selects the same traversal.
    /// - Hold the values and move the *declaration*. The same code bytes
    ///   declared `n x k` and `k x n` admit exactly one triple each, opposite
    ///   ways round. And one decoded operand under two declarations differing in
    ///   the block alone takes two factorizations to one answer.
    #[test]
    fn traversal_selection_reads_the_declaration_and_never_the_operand_cs_10() {
        const SPEC: &str = "uor-matmul/1";
        const NO_TABLE: &str =
            "sha256:0000000000000000000000000000000000000000000000000000000000000000";

        let table = e8_table::<Full<i8>>().expect("i8 holds E8");
        let book = e8_codec(&table);
        let space = 256usize;
        let block = 8usize;
        let (m, k) = (16usize, 64usize);
        let blocks = k / block;
        let lane = core::mem::size_of::<<i8 as Tabulated>::Lane>();
        let l1 = blocking::L1_BYTES;
        let bound = <Full<i8> as Bound>::VALUE;
        let backend = GemmOptions::default().backend;

        // The break-even, recomputed from the host's own declarations exactly as
        // `CG-18` recomputes it: a recorded number would make this test about
        // one ISA rather than about the selection.
        let rows = ROW_TILES
            .into_iter()
            .find(|&r| r <= tabulation_rows(space, l1, lane).min(m))
            .expect("a 256-entry table fits L1 at some tile");
        let spec =
            <i8 as Tabulated>::table_spec(backend, bound, false, rows, column_group(rows), block);
        let steps =
            <i8 as Tabulated>::dense_steps(backend, bound, rows, block * spec.lanes_per_add);
        let break_even = (1..)
            .find(|&cols| tabulation_pays(space, block, cols, rows, steps, l1, lane))
            .expect("E8 pays at some column width on every pair this workspace ships");

        // One side of the predicate and the other: the selection has to be
        // value-independent where the table runs and where it declines, or the
        // invariance is an accident of always answering the same thing.
        for n in [8usize, break_even] {
            let shape = Shape { m, k, n };
            let codes = n * blocks;
            let operands: Vec<(&str, Vec<u16>, Vec<i8>)> = vec![
                (
                    "one code and one activation, repeated",
                    vec![0u16; codes],
                    vec![0i8; m * k],
                ),
                (
                    "the top of the code space against the bottom of the alphabet",
                    vec![(space - 1) as u16; codes],
                    vec![-128i8; m * k],
                ),
                (
                    "the two ends of the code space, alternating, against the top of the alphabet",
                    (0..codes)
                        .map(|i| if i % 2 == 0 { 0 } else { (space - 1) as u16 })
                        .collect(),
                    vec![127i8; m * k],
                ),
                (
                    "no two columns alike",
                    (0..codes)
                        .map(|i| {
                            let (j, p) = (i / blocks, i % blocks);
                            ((0..p).fold(j, |acc, _| acc / space) % space) as u16
                        })
                        .collect(),
                    (0..m * k).map(|i| ((i % 255) as i64 - 127) as i8).collect(),
                ),
                (
                    "a recorded pseudorandom fill",
                    fill(codes, 0xc510, |x| (x % space as u64) as u16),
                    fill(m * k, 0xc511, |x| ((x % 255) as i64 - 127) as i8),
                ),
            ];

            let mut selected: Option<(&str, Traversal)> = None;
            let mut minted: Vec<(&str, String)> = Vec::new();
            for (label, stream, a) in &operands {
                let w = CodedMatrix::new(book, n, k, stream).expect("the codes describe n x k");
                // Every field but the digest comes from the operand itself, so
                // this is that artifact's manifest and not a stand-in written
                // beside it.
                let declared = Manifest::of(&w, NO_TABLE, NO_TABLE, SPEC);
                let digest = digest_of(stream);
                let artifact = Manifest {
                    codes_sha256: &digest,
                    ..declared
                };
                let mut buf = vec![0u8; 512];
                let len = artifact
                    .write_canonical_json(&mut buf)
                    .expect("a canonical manifest");
                minted.push((
                    label,
                    String::from_utf8(buf[..len].to_vec()).expect("the manifest is ascii"),
                ));

                assert!(
                    artifact.reduces_along_the_block(shape),
                    "the operand is declared n x k at {label}"
                );
                assert_eq!(
                    artifact.addressing(),
                    Addressing::of(TierId::Book, block, bound),
                    "the addressing is the declaration's and nothing else's at {label}"
                );

                let (got, census) =
                    tabulated(&w, a, m, n, Traversal::Blocked, OFFER_STEPS, OFFER_STEPS, 0);
                assert_eq!(
                    got,
                    reference(&w, a, m, k, n),
                    "the product at n = {n}, {label} ({census:?})"
                );
                match selected {
                    None => selected = Some((label, ran(&census))),
                    Some((first, want)) => assert_eq!(
                        ran(&census),
                        want,
                        "at n = {n} the selection moved from {first} to {label}: {census:?}"
                    ),
                }
            }

            // And the artifacts really are different artifacts: the field that
            // moved with the values is the field the derivation does not read.
            for (i, (x, xs)) in minted.iter().enumerate() {
                for (y, ys) in &minted[i + 1..] {
                    assert_ne!(xs, ys, "at n = {n}, {x} and {y} minted one manifest");
                }
            }
        }

        // ---- the declaration moves: orientation ----
        //
        // The same 64 code words declared `8 x 64` and then `64 x 8`. Not one
        // byte of the operand differs between the two, and each declaration
        // admits exactly one of the two triples --- opposite ways round.
        let n = 8usize;
        let shape = Shape { m, k, n };
        let stream: Vec<u16> = fill(n * blocks, 0xc512, |x| (x % space as u64) as u16);
        let a: Vec<i8> = fill(m * k, 0xc513, |x| ((x % 255) as i64 - 127) as i8);
        let along = CodedMatrix::new(book, n, k, &stream).expect("the codes describe n x k");
        let across = CodedMatrix::new(book, k, n, &stream).expect("the same codes describe k x n");
        let ma = Manifest::of(&along, NO_TABLE, NO_TABLE, SPEC);
        let mx = Manifest::of(&across, NO_TABLE, NO_TABLE, SPEC);
        assert_eq!(
            ma.addressing(),
            mx.addressing(),
            "only the orientation moved, so the code declaration did not"
        );
        assert!(ma.reduces_along_the_block(shape));
        assert!(!ma.reduces_across_the_block(shape));
        assert!(mx.reduces_across_the_block(shape));
        assert!(!mx.reduces_along_the_block(shape));

        // The constructors answer the same question the declaration answered,
        // and they answer it the same way. This is the whole orientation half of
        // `CS-10`: the traversal that exists at all is decided here, from
        // `rows` and `cols`, before a code has been looked at.
        let av = MatView::row_major(as_alphabet_full(&a), m, k).unwrap();
        let mut c = vec![0i32; m * n];
        {
            let cv = MatViewMut::row_major(&mut c, m, n).unwrap();
            assert!(
                TabulatedTriple::new(av, along, cv).is_ok(),
                "the n x k declaration is the tabulated orientation"
            );
        }
        {
            let cv = MatViewMut::row_major(&mut c, m, n).unwrap();
            assert!(
                matches!(
                    TabulatedTriple::new(av, across, cv),
                    Err(NotAProduct::Nonconformant { .. })
                ),
                "and the k x n declaration is not"
            );
        }
        {
            let cv = MatViewMut::row_major(&mut c, m, n).unwrap();
            assert!(
                CodedTriple::new(av, across, cv).is_ok(),
                "the k x n declaration is the streaming orientation"
            );
        }
        {
            let cv = MatViewMut::row_major(&mut c, m, n).unwrap();
            assert!(
                matches!(
                    CodedTriple::new(av, along, cv),
                    Err(NotAProduct::Nonconformant { .. })
                ),
                "and the n x k declaration is not"
            );
        }

        // ---- the declaration moves: the block ----
        //
        // One decoded operand under two declarations differing in the block
        // alone: a 16-entry grid at one element per code, and a 16-codeword
        // book whose `c`-th codeword is eight copies of the grid's `c`-th
        // element. The code space is 16 either way, so the block is what moved.
        let vals: [A8; 16] = core::array::from_fn(|i| Alphabet::of((i as i8) - 8));
        let grid = Grid::<i8, Full<i8>, 16>::new(&vals);
        let words: [[A8; 8]; 16] = core::array::from_fn(|c| [vals[c]; 8]);
        let runs = Book::<i8, Full<i8>, 16, 8, u16>::new(&words);
        let small = 16usize;

        let rows16 = ROW_TILES
            .into_iter()
            .find(|&r| r <= tabulation_rows(small, l1, lane).min(m))
            .expect("a 16-entry table fits L1 at some tile");
        let spec16 =
            <i8 as Tabulated>::table_spec(backend, bound, false, rows16, column_group(rows16), 8);
        let steps16 =
            <i8 as Tabulated>::dense_steps(backend, bound, rows16, 8 * spec16.lanes_per_add);
        let pays_at = (1..)
            .find(|&cols| {
                tabulation_pays_for_spec(small, 8, cols, rows16, steps16, l1, lane, &spec16)
            })
            .expect("a 16-entry codebook of eight-element codewords pays at some column width");
        // This block-one integer product build is refused at every width by the
        // public reference predicate. A contextual Atlas block-one body is not
        // allowed to leak a clock-derived rule into this integer product build.
        let spec1 =
            <i8 as Tabulated>::table_spec(backend, bound, false, rows16, column_group(rows16), 1);
        let steps1 = <i8 as Tabulated>::dense_steps(backend, bound, rows16, spec1.lanes_per_add);
        assert!(
            !tabulation_pays(small, 1, usize::MAX, rows16, steps1, l1, lane),
            "one element per code repays no build at any width"
        );
        assert!(Addressing::of(TierId::Book, 8, bound).addresses_a_run());
        assert!(!Addressing::of(TierId::Grid, 1, bound).addresses_a_run());
        assert!(Addressing::of(TierId::Grid, 1, bound).addresses_an_element());

        let n = pays_at;
        let shape = Shape { m, k, n };
        let a: Vec<i8> = fill(m * k, 0xc514, |x| ((x % 255) as i64 - 127) as i8);
        let coarse: Vec<u16> = fill(n * blocks, 0xc515, |x| (x % 16) as u16);
        let fine: Vec<u16> = (0..n * k).map(|i| coarse[i / block]).collect();
        let wb = CodedMatrix::new(runs, n, k, &coarse).expect("n x k at a block of eight");
        let wg = CodedMatrix::new(grid, n, k, &fine).expect("n x k at a block of one");
        for j in 0..n {
            for t in 0..k {
                assert_eq!(
                    wb.at(j, t),
                    wg.at(j, t),
                    "one operand, two declarations, at ({j}, {t})"
                );
            }
        }
        let mb = Manifest::of(&wb, NO_TABLE, NO_TABLE, SPEC);
        let mg = Manifest::of(&wg, NO_TABLE, NO_TABLE, SPEC);
        assert!(mb.reduces_along_the_block(shape));
        assert!(mg.reduces_along_the_block(shape));
        assert!(mb.addressing().addresses_a_run());
        assert!(!mg.addressing().addresses_a_run());

        // The plan the two calls resolve to, recomputed from the offers the
        // helper passes, so a decline for want of room reads as the failure it
        // would be rather than as the claim.
        let plan = Plan::choose(
            small,
            shape,
            lane,
            suggested_tabulation::<i8, Full<i8>>(shape, small, 8).max(1),
            suggested_tabulation_lanes::<i8, Full<i8>>(shape, small, 8).max(1) * 8 / lane,
            8,
            <i8 as Tabulated>::probe_capacity::<<i8 as Tabulated>::Lane>(bound),
        )
        .expect("the suggested offers admit a plan");
        assert_eq!(plan.cols, n, "the amortization axis is `n` itself");
        assert_eq!(plan.rows, rows16, "the tile the derivation priced");

        let (got_b, cen_b) = tabulated(
            &wb,
            &a,
            m,
            n,
            Traversal::Blocked,
            OFFER_STEPS,
            OFFER_STEPS,
            0,
        );
        let (got_g, cen_g) = tabulated(
            &wg,
            &a,
            m,
            n,
            Traversal::Blocked,
            OFFER_STEPS,
            OFFER_STEPS,
            0,
        );
        assert_eq!(got_b, reference(&wb, &a, m, k, n));
        assert_eq!(
            got_b, got_g,
            "the declaration moved the factorization and not the answer"
        );
        assert_eq!(
            ran(&cen_b),
            Traversal::Tabulated,
            "a block of eight is tabulated at n = {n}: {cen_b:?}"
        );
        assert_eq!(
            ran(&cen_g),
            Traversal::Blocked,
            "and a block of one is not, on the same decoded operand: {cen_g:?}"
        );
    }

    /// Every knob but the encode mode, at one element type and one pair of
    /// operands: the traversal, the backend, the panel offer, and the tile
    /// partition the caller writes its output through.
    ///
    /// Asserts byte identity across the whole sweep and returns those bytes, so
    /// a caller can hold them against another mode's.
    fn one_answer<E, Bd, O, Ep>(
        a: &[Alphabet<E, Bd>],
        b: &[Alphabet<E, Bd>],
        shape: Shape,
        epilogue: &Ep,
        encode: EncodeMode,
    ) -> Vec<O>
    where
        E: Element,
        Bd: Bound,
        O: Element + EncodeFrom<AccOf<E>>,
        Ep: Epilogue<E, O>,
    {
        let Shape { m, k, n } = shape;
        let mut answer: Option<Vec<O>> = None;
        for traversal in [
            Traversal::OutputMajor,
            Traversal::Blocked,
            Traversal::Tabulated,
        ] {
            for backend in core::iter::once(Backend::Auto).chain(Backend::ALL) {
                for panel in [0usize, 1, k, crate::suggested_scratch(shape)] {
                    for (tr, tc) in [(m, n), (1, 1), (2, 3), (m, 1), (1, n)] {
                        let mut c = vec![O::ZERO; m * n];
                        let mut buf = vec![Alphabet::<E, Bd>::ZERO; panel];
                        let mut scratch = Scratch::new(&mut buf);
                        // One tile at a time, into its own region of `C`
                        // through its own strides: the partition is a knob a
                        // caller turns, and the sum a tile computes cannot
                        // depend on which tiles ran beside it (`CD-08`).
                        for tile in Partition::new(shape, tr, tc) {
                            let av = MatView::new(
                                &a[tile.row * k..],
                                tile.rows,
                                k,
                                Strides::row_major(k),
                            )
                            .expect("a row block of A");
                            let bv = MatView::new(
                                &b[tile.col..],
                                k,
                                tile.cols,
                                Strides {
                                    rs: n as isize,
                                    cs: 1,
                                },
                            )
                            .expect("a column block of B");
                            let cv = MatViewMut::new(
                                &mut c[tile.row * n + tile.col..],
                                tile.rows,
                                tile.cols,
                                Strides {
                                    rs: n as isize,
                                    cs: 1,
                                },
                            )
                            .expect("the tile's own region of C");
                            let mut t = Triple::new(av, bv, cv).expect("a product");
                            gemm(
                                &mut t,
                                epilogue,
                                GemmOptions {
                                    traversal,
                                    backend,
                                    encode,
                                },
                                &mut scratch,
                            );
                        }
                        match &answer {
                            None => answer = Some(c),
                            Some(want) => assert_eq!(
                                &c, want,
                                "{traversal:?} on {backend:?}, a panel offer of {panel}, \
                                 tiled {tr}x{tc}, at {encode:?}"
                            ),
                        }
                    }
                }
            }
        }
        answer.expect("the sweep names at least one setting of every knob")
    }

    /// `CD-28`: within one element type, the encode mode is the only knob that
    /// moves the output bytes.
    ///
    /// `CD-05` says the encode mode is the only thing that changes the output
    /// bytes, and with a second semiring in the workspace that reads as
    /// falsified: the ring product and the `(max, +)` product of one pair of
    /// operands write different bytes, and neither changed mode. It is not
    /// falsified. The quantifier is *per element type* --- the element type is
    /// what carries the semiring, so two element types are two functions and
    /// were never obliged to agree --- and this is that quantifier asserted at
    /// both families, so the row is about the quantifier and not about one
    /// algebra.
    ///
    /// Two-sided at each family: every other knob held against byte identity,
    /// then the mode moved and the bytes with it.
    #[test]
    fn within_one_element_type_the_encode_mode_is_the_only_mover_cd_28() {
        let (m, k, n) = (5usize, 7usize, 6usize);
        let shape = Shape { m, k, n };
        let modes = [
            EncodeMode::Nearest,
            EncodeMode::TowardZero,
            EncodeMode::Saturating,
            EncodeMode::Wrapping,
        ];

        // The output alphabet is as narrow as the input's, so the accumulation
        // leaves its range and the mode has something to decide. The extreme is
        // placed rather than hoped for: the first row of `A` and the first
        // column of `B` sit at the top of the alphabet, so cell `(0, 0)` is past
        // `i8` by construction.
        let mut ring_a: Vec<i8> = fill(m * k, 0xcd28, |x| ((x % 255) as i64 - 127) as i8);
        let mut ring_b: Vec<i8> = fill(k * n, 0xcd29, |x| ((x % 255) as i64 - 127) as i8);
        for x in ring_a.iter_mut().take(k) {
            *x = 127;
        }
        for p in 0..k {
            ring_b[p * n] = 127;
        }
        let ring: Vec<Vec<i8>> = modes
            .iter()
            .map(|&mode| {
                one_answer::<i8, Full<i8>, i8, _>(
                    as_alphabet_full(&ring_a),
                    as_alphabet_full(&ring_b),
                    shape,
                    &Linear::OVERWRITE,
                    mode,
                )
            })
            .collect();

        // The same shape at the tropical instance. `⊗` is addition, so two
        // elements at the top of the alphabet sum past it and the mode decides
        // the same question it decides in the ring. Lanes at the semiring zero
        // are swept too, because a masked operand is an operand.
        let mut trop_a: Vec<Trop<i8>> = ring_a
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                if i % 11 == 3 {
                    Trop::NEG_INF
                } else {
                    Trop::finite(x)
                }
            })
            .collect();
        let mut trop_b: Vec<Trop<i8>> = ring_b
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                if i % 13 == 5 {
                    Trop::NEG_INF
                } else {
                    Trop::finite(x)
                }
            })
            .collect();
        trop_a[0] = Trop::finite(127);
        trop_b[0] = Trop::finite(127);
        let trop: Vec<Vec<Trop<i8>>> = modes
            .iter()
            .map(|&mode| {
                one_answer::<Trop<i8>, Full<i8>, Trop<i8>, _>(
                    as_alphabet_tropical(&trop_a),
                    as_alphabet_tropical(&trop_b),
                    shape,
                    &MaxPlus::OVERWRITE,
                    mode,
                )
            })
            .collect();

        // Three of the four modes are one map at an integer output: there is
        // nothing to round, so `Nearest`, `TowardZero` and `Saturating` all
        // clamp. `Wrapping` is a different map, and it is the one knob this row
        // says the bytes turn on --- at both families.
        assert_eq!(ring[0], ring[1], "the ring clamps under TowardZero");
        assert_eq!(ring[0], ring[2], "the ring clamps under Saturating");
        assert_ne!(
            ring[0], ring[3],
            "and Wrapping moves the ring's bytes, so the mode is not inert"
        );
        assert_eq!(trop[0], trop[1], "the tropical instance clamps too");
        assert_eq!(trop[0], trop[2], "and under Saturating as well");
        assert_ne!(
            trop[0], trop[3],
            "and Wrapping moves the tropical bytes, so the row is not about one family"
        );

        // The observation the quantifier exists for: at one mode and one pair of
        // operands the two element types write different numbers. Nothing is
        // violated --- they compute two products --- and that is exactly why
        // `CD-05`'s "only" is read inside an element type.
        let ring_values: Vec<Option<i8>> = ring[0].iter().map(|&x| Some(x)).collect();
        let trop_values: Vec<Option<i8>> = trop[0].iter().map(|&x| x.get()).collect();
        assert_ne!(
            ring_values, trop_values,
            "two element types are two functions, which is what the quantifier says"
        );
    }
}
