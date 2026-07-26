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

use uor_matmul_codec::{CodedMatrix, Enumerable};
use uor_matmul_core::generated::blocking;
use uor_matmul_core::{
    AccOf, Accumulator, Alphabet, Bound, Element, EncodeFrom, IntegerElement, MatView, MatViewMut,
    NotAProduct, Shape, Traversal,
};

use uor_matmul_core::{Backend, Strides, Triple};
use uor_matmul_kernels::{
    available_table_i16, available_table_i8, choose_table, packed_slot, portable_table,
};

use crate::coded::self_aliases;
use crate::driver::GemmOptions;
use crate::epilogue::Epilogue;
use crate::kernel::{gemm_packed, Kernelized};
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
    /// Calls into the dense tile kernels, over a decoded operand.
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
    /// Record `n` widening multiply-accumulates.
    fn multiplied(&mut self, n: u64);
    /// Record `n` exact accumulator combines.
    fn added(&mut self, n: u64);
    /// Record `n` reads of a tabulated partial sum.
    fn read(&mut self, n: u64);
    /// Record `n` codec decodes.
    fn decoded(&mut self, n: u64);
    /// Record a call into the dense tile kernels.
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
/// [`uor_matmul_core::fits_narrow`] already licenses for the tile kernels.
#[inline]
fn dot_lane<E, Bd, L>(a: &[Alphabet<E, Bd>], w: &[Alphabet<E, Bd>], run: usize) -> AccOf<E>
where
    E: IntegerElement,
    Bd: Bound,
    L: Lane<E>,
{
    let mut acc = <AccOf<E> as Accumulator>::ZERO;
    for (ra, rw) in a.chunks(run).zip(w.chunks(run)) {
        let mut lane = L::ZERO;
        for (&x, &y) in ra.iter().zip(rw) {
            lane = lane.mac(x.get(), y.get());
        }
        acc = lane.place(acc);
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
/// two, so the index needs a mask and never a comparison.
///
/// A free function, not an associated one, because it is a fact about the
/// enumeration and not about the lane it is tabulated in --- naming a lane to
/// ask it would be naming something the answer does not depend on.
pub const fn slab_codes(code_space: usize) -> usize {
    code_space.next_power_of_two()
}

/// How many lane words a stack of `depth` tables over `rows` rows of a
/// `code_space`-wide enumeration occupies.
///
/// A query, so an embedded caller can size a static and know the answer before
/// it calls anything.
pub const fn table_words(code_space: usize, rows: usize, depth: usize) -> usize {
    slab_codes(code_space)
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
        let want = table_words(code_space, rows, depth);
        if code_space == 0 || rows == 0 || depth == 0 || words.len() < want {
            return None;
        }
        let codes = slab_codes(code_space);
        let table = Self {
            words: &mut words[..want],
            code_space,
            codes,
            rows,
            depth,
        };
        // The padding, once. The build writes `code_space * rows` of every slab
        // and never the rest, so zeroing here is zeroing for the whole call.
        let slab = codes * rows;
        let live = code_space * rows;
        for slot in 0..depth {
            table.words[slot * slab + live..slot * slab + slab].fill(L::ZERO);
        }
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
    /// tabulated traversal issues at all.
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
        ledger.multiplied((self.code_space * block * self.rows) as u64);
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
    /// that tracks meanings rather than expressions.
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
/// [`crate::suggested_collapse_index`] uses on the row side.
///
/// A *query*. Offering none gives the same bytes from the uncollapsed traversal,
/// which is the rule every other offer in this library follows (`CD-13`).
pub fn suggested_tabulation_index(shape: Shape) -> usize {
    let doubled = shape
        .n
        .saturating_mul(2) // R3-ok: a size query, not an accumulation
        .next_power_of_two()
        .saturating_mul(2); // R3-ok: a size query, not an accumulation
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
        <E::Lane as Lane<E>>::capacity(Bd::VALUE),
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
/// block; offering none gives the same bytes from the streaming traversal
/// (`CD-13`). It does not grow with `k`.
///
/// When the element type has no narrow register this covers the table stack and
/// the column accumulation too, because there is then nowhere else for them to
/// live.
pub fn suggested_tabulation<E: Tabulated, Bd: Bound>(
    shape: Shape,
    code_space: usize,
    block: usize,
) -> usize {
    let Some(plan) = Plan::choose(
        code_space,
        shape,
        E::LANE_BYTES,
        usize::MAX,
        usize::MAX,
        block,
        <E::Lane as Lane<E>>::capacity(Bd::VALUE),
    ) else {
        return 0;
    };
    let tile = plan.rows.saturating_mul(plan.cols); // R3-ok: a size query, not an accumulation
    if E::LANE_IS_EXACT {
        tile.saturating_add(plan.lane_words(code_space)) // R3-ok: a size query, not an accumulation
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
            .saturating_mul(rows) // R3-ok: a cache or tile question, not an accumulation
            .saturating_mul(lane_bytes) // R3-ok: a cache or tile question, not an accumulation
            .saturating_mul(2) // R3-ok: a cache or tile question, not an accumulation
            <= l1_bytes
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

/// Does tabulation issue fewer instructions than the dense tile, and does its
/// table fit?
///
/// `cols` is the width of the column block the caller's offer supports, which is
/// `n` when the offer holds the whole output width. The build is repeated once
/// per column block, so it is the block and not the shape that the count turns
/// on.
///
/// ```text
/// tabulated = m*k*S/steps.table + m*n*k/steps.table
/// dense     = m*k*n/steps.dense
/// ```
///
/// so the table is cheaper exactly when
/// `cols * (steps.table - effective) > code_space * effective * block`, where
/// `effective` is the dense step scaled by the rows actually present.
///
/// Two cases have no such `cols` and are refused outright: `block == 1`, where
/// one code names one element; and `steps.table <= effective`, where one table
/// instruction does not even cover what one dense instruction does, so nothing
/// repays the build.
pub const fn tabulation_pays(
    code_space: usize,
    block: usize,
    cols: usize,
    rows: usize,
    steps: Steps,
    l1_bytes: usize,
    lane_bytes: usize,
) -> bool {
    if steps.dense_rows == 0 {
        return false;
    }
    let present = if rows < steps.dense_rows {
        rows
    } else {
        steps.dense_rows
    };
    let effective = steps.dense.saturating_mul(present) / steps.dense_rows; // R3-ok: a cost estimate, not an accumulation
    block > 1
        && effective > 0
        && steps.table > effective
        && cols.saturating_mul(steps.table - effective) // R3-ok: a cost estimate, not an accumulation
            > code_space.saturating_mul(effective).saturating_mul(block) // R3-ok: a cost estimate, not an accumulation
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

/// The lane a family tabulates in, and the sequence that reads it.
///
/// One associated type, not a scan. Which register holds a run of products is a
/// property of the *element type* --- the same kind of fact as `AccOf<E>` --- and
/// writing it as a per-shape search was a mechanism with one answer.
///
/// Not a quality ordering, and not a fallback chain. Every lane computes the same
/// integer; a narrower one moves fewer bytes per product, and `CD-13` asserts the
/// bytes across all of them. An element family with no narrow register at all
/// tabulates in the exact accumulator, which is not a placeholder there but the
/// whole of what the hardware offers --- the same status
/// [`uor_matmul_kernels::available_i64_exact`] has for the dense tile.
pub trait Tabulated: Kernelized {
    /// The lane one table entry's row is held in.
    type Lane: Lane<Self>;

    /// Bytes one lane word occupies. The column loop's traffic is this divided
    /// by the codec's block, per product, which is the whole of why it is here.
    const LANE_BYTES: usize = core::mem::size_of::<Self::Lane>();

    /// Does this family's lane come out of the accumulator offer rather than
    /// the narrow one?
    ///
    /// True exactly when the lane *is* the exact accumulator, which is where a
    /// word that wide already lives.
    const LANE_IS_EXACT: bool;

    /// The sequence for this backend, at this tile height and column group.
    ///
    /// Always present: the reference is exact on every alphabet and every tile,
    /// so narrowing the choice can never empty it.
    fn table_spec(
        backend: Backend,
        bound: u128,
        rows: usize,
        group: usize,
        block: usize,
    ) -> TableSpec<Self, Self::Lane>;

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
}

/// `i8`: an `i32` lane holds 133144 products of the full alphabet, which is past
/// every depth a weight row reaches, so the whole reduction is one run.
impl Tabulated for i8 {
    type Lane = i32;
    const LANE_IS_EXACT: bool = false;

    fn table_spec(
        backend: Backend,
        bound: u128,
        rows: usize,
        group: usize,
        block: usize,
    ) -> TableSpec<i8, i32> {
        choose_table(available_table_i8(rows, group), backend, bound, block)
            .expect("the reference sequence is always present")
    }

    fn lanes<'s>(
        narrow: &'s mut [i64],
        _: &'s mut [AccOf<i8>],
        want: usize,
    ) -> Option<&'s mut [i32]> {
        bytemuck::cast_slice_mut::<i64, i32>(narrow).get_mut(..want)
    }
}

/// `i16`: two full `i16` products already need 31 bits, so no 32-bit lane holds
/// an entry of any block longer than one.
impl Tabulated for i16 {
    type Lane = i64;
    const LANE_IS_EXACT: bool = false;

    fn table_spec(
        backend: Backend,
        bound: u128,
        rows: usize,
        group: usize,
        block: usize,
    ) -> TableSpec<i16, i64> {
        choose_table(available_table_i16(rows, group), backend, bound, block)
            .expect("the reference sequence is always present")
    }

    fn lanes<'s>(
        narrow: &'s mut [i64],
        _: &'s mut [AccOf<i16>],
        want: usize,
    ) -> Option<&'s mut [i64]> {
        narrow.get_mut(..want)
    }
}

/// `i32`: the product of two `i32` needs 62 bits and a run of them needs more, so
/// the lane is the exact accumulator. Nothing narrower is a lane here --- an
/// `i64` would hold two products --- and that is a fact about the width, not a
/// gap in the sequence table.
impl Tabulated for i32 {
    type Lane = Wide<AccOf<i32>>;
    const LANE_IS_EXACT: bool = true;

    fn table_spec(
        backend: Backend,
        bound: u128,
        rows: usize,
        group: usize,
        block: usize,
    ) -> TableSpec<i32, Wide<AccOf<i32>>> {
        // The reference is the only sequence for this family, and its `k_group`
        // is one, which divides every block.
        let _ = (backend, bound, block);
        portable_table::<i32, Wide<AccOf<i32>>>(rows, group)
    }

    fn lanes<'s>(
        _: &'s mut [i64],
        exact: &'s mut [AccOf<i32>],
        want: usize,
    ) -> Option<&'s mut [Wide<AccOf<i32>>]> {
        Some(Wide::wrap_slice_mut(exact.get_mut(..want)?))
    }
}

/// `i64`: as `i32`, one width up.
impl Tabulated for i64 {
    type Lane = Wide<AccOf<i64>>;
    const LANE_IS_EXACT: bool = true;

    fn table_spec(
        backend: Backend,
        bound: u128,
        rows: usize,
        group: usize,
        block: usize,
    ) -> TableSpec<i64, Wide<AccOf<i64>>> {
        // The reference is the only sequence for this family, and its `k_group`
        // is one, which divides every block.
        let _ = (backend, bound, block);
        portable_table::<i64, Wide<AccOf<i64>>>(rows, group)
    }

    fn lanes<'s>(
        _: &'s mut [i64],
        exact: &'s mut [AccOf<i64>],
        want: usize,
    ) -> Option<&'s mut [Wide<AccOf<i64>>]> {
        Some(Wide::wrap_slice_mut(exact.get_mut(..want)?))
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

    /// The largest plan the two offers support.
    ///
    /// Rows first, from the cache budget: a wider row tile shares each decode
    /// across more outputs and puts more lane words under one vector instruction.
    /// Then the column block, as wide as the exact offer allows, because the
    /// build repeats once per column block. Then the depth, as deep as the lane
    /// offer and the cache allow.
    fn choose(
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
        let row_cap = tabulation_rows(code_space, blocking::L1_BYTES, lane_bytes)
            .min(shape.m)
            .min(exact_offer);
        let slab = slab_codes(code_space);
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
                                                                // `(i8, i32)` and `Bk = 8`, against a depth of at most `GATHER_SLOTS`),
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
        let depth = by_cache.min(by_offer).min(blocks.max(1)).min(GATHER_SLOTS);
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
pub fn gemm_tabulated<E, Bd, C, O, Ep>(
    triple: &mut TabulatedTriple<'_, '_, '_, E, Bd, C, O>,
    epilogue: &Ep,
    options: GemmOptions,
    scratch: &mut Scratch<'_, E, Bd>,
    lanes: &mut Tabulation<'_>,
) where
    E: Tabulated,
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
    E: Tabulated,
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
    E: Tabulated,
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
        decline(triple, epilogue, options, scratch, ledger);
        return;
    }

    // Which output columns carry a distinct meaning. One pass over the code
    // stream, before any arithmetic, and the answer is used by every row tile ---
    // so a weight matrix with repeated columns is charged for the ones it has
    // rather than the ones it is written with. `CG-10` reports both the
    // degeneracy and what the question cost when there was none.
    let (words, index) = (&mut *lanes.lanes, &mut *lanes.index);
    let distinct =
        distinct_columns::<E, Bd, C>(triple.w.codes(), triple.w.codes_per_row(), shape.n, index);
    // Only when it saves work. An operand with no repeated column has paid one
    // pass over its code stream for the question and must not also pay for a
    // traversal shaped around an answer of "none".
    let index = distinct.filter(|&d| d < shape.n).map(|_| &index[..shape.n]);

    let lane_bytes = E::LANE_BYTES;
    let exact_offer = scratch.accumulators();
    // The lane offer is `i64`-shaped, read as however many lane words fit in the
    // same bytes; a family whose lane *is* the exact accumulator reads that
    // offer instead, because that is where a word so wide already lives.
    let offered = if E::LANE_IS_EXACT {
        exact_offer
    } else {
        core::mem::size_of_val(&*words) / lane_bytes
    };
    let Some(plan) = Plan::choose(
        space,
        shape,
        lane_bytes,
        exact_offer,
        offered,
        block,
        <E::Lane as Lane<E>>::capacity(Bd::VALUE),
    ) else {
        decline(triple, epilogue, options, scratch, ledger);
        return;
    };
    // Both declarations come from the sequences that will run, at the tile the
    // plan resolved to --- never from a constant standing in for them.
    let table = E::table_spec(
        options.backend,
        Bd::VALUE,
        plan.rows,
        column_group(plan.rows),
        block,
    );
    let dense = E::exact_spec(options.backend, Bd::VALUE, plan.rows);
    let steps = Steps {
        table: block.saturating_mul(table.lanes_per_add), // R3-ok: a size or cost query, not an accumulation
        dense: dense.products_per_step,
        // The *blocking* row count, not the chosen tile's `mr`. Reading `mr`
        // says a one-row kernel wastes no lanes, which is true and is not why
        // the dense path is weak at small `m`: it packs `n * k` elements of the
        // operand to compute `m * n * k` products, so at `m = 1` the copy is the
        // same order as the arithmetic. Measured, `mr` here declined the table
        // at `1x1024x4096` where it is 5.4x the dense path.
        dense_rows: blocking::KERNEL_ROWS,
    };
    if !admits(options.traversal, space, block, plan, steps, lane_bytes) {
        decline(triple, epilogue, options, scratch, ledger);
        return;
    }

    let want = plan.lane_words(space);
    let tile = plan.rows * plan.cols;
    let take = if E::LANE_IS_EXACT { tile + want } else { tile };
    if take > exact_offer {
        decline(triple, epilogue, options, scratch, ledger);
        return;
    }
    let (panel, accumulators) = scratch.split(suggested_tabulation_panel(space, block), take);
    if !decode_book(triple.w.codec(), panel, ledger) {
        decline(triple, epilogue, options, scratch, ledger);
        return;
    }
    let (exact, rest) = accumulators.split_at_mut(tile);
    let Some(lanes) = E::lanes(words, rest, want) else {
        decline(triple, epilogue, options, scratch, ledger);
        return;
    };
    tabulate::<E, Bd, C, O, Ep, Lg>(
        triple, epilogue, options, exact, lanes, panel, index, plan, ledger,
    );
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
    steps: Steps,
    lane_bytes: usize,
) -> bool {
    match traversal {
        Traversal::OutputMajor => false,
        Traversal::Blocked => tabulation_pays(
            code_space,
            block,
            plan.cols,
            plan.rows,
            steps,
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
    // The book *and* the widest activation tile a row tile can pack, because the
    // build walks both and a panel that held only the book would leave the tile
    // with nowhere to go.
    if panel.len() < suggested_tabulation_panel(space, block) {
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
const fn column_group(rows: usize) -> usize {
    if rows >= COLUMN_LANES {
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
    E: IntegerElement,
    Bd: Bound,
    C: Enumerable<E, Bd>,
{
    if n == 0 || codes_per_row == 0 || codes.len() < n.checked_mul(codes_per_row)? {
        return None;
    }
    // Half full at worst, so an empty slot always exists and a probe terminates.
    let table = n.checked_mul(2)?.next_power_of_two();
    if index.len() < n.checked_add(table.checked_mul(2)?)? {
        return None;
    }
    let (position, rest) = index.split_at_mut(n);
    let (slot, key) = rest.split_at_mut(table);
    // Zero is "empty"; a slot holds one more than the column it names.
    slot[..table].fill(0);
    let mask = table - 1;

    let mut distinct = 0usize;
    for (j, position) in position.iter_mut().enumerate() {
        let run = &codes[j * codes_per_row..(j + 1) * codes_per_row];
        let hash = column_hash::<E, Bd, C>(run);
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

/// Do two columns read the same table entries in the same order?
fn columns_equal<E, Bd, C>(a: &[C::Code], b: &[C::Code]) -> bool
where
    E: IntegerElement,
    Bd: Bound,
    C: Enumerable<E, Bd>,
{
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(&x, &y)| C::index_of(x) == C::index_of(y))
}

/// How much of a column the hash reads.
///
/// The hash is a *filter*: [`columns_equal`] verifies in full on every hit, so a
/// short read costs only a false positive and never a wrong answer. Reading the
/// whole column made the pass half the cost of a traversal pass, which at `m = 1`
/// --- where there is only one such pass --- was a third of the throughput. Two
/// hundred and fifty-six bits of a column separate any two of them that differ in
/// their first sixteen codes, and the ones that do not are separated by the
/// comparison.
const HASH_PREFIX: usize = 16;

/// FNV over the head of one column's index stream, mixed in eight lanes.
///
/// Eight because the mix is a five-cycle multiply and the multiplier retires one
/// a cycle, so eight chains is where the latency stops being what bounds it ---
/// the same reason and the same shape as [`crate::collapse`]'s row hash.
fn column_hash<E, Bd, C>(run: &[C::Code]) -> usize
where
    E: IntegerElement,
    Bd: Bound,
    C: Enumerable<E, Bd>,
{
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
    // The length goes in first: two columns of different lengths cannot be equal,
    // and this is the only place the tail beyond the prefix is represented.
    let mut seeded = SEED ^ run.len() as u64;
    seeded = seeded.wrapping_mul(PRIME);
    h[0] ^= seeded;
    let run = &run[..run.len().min(HASH_PREFIX)];
    let mut chunks = run.chunks_exact(8);
    for c in &mut chunks {
        for (lane, &code) in h.iter_mut().zip(c) {
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
    // splitmix's finisher: the FNV lanes carry their entropy high and an
    // open-addressed probe reads the low bits.
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    (x ^ (x >> 31)) as usize
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
    book: &[E],
    acts: &mut [E],
    index: Option<&[usize]>,
    plan: Plan,
    rows: usize,
    group: usize,
    row0: usize,
    ledger: &mut Lg,
) where
    E: Tabulated<Lane = L>,
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
    // Blocks one lane word holds before it must be placed. A question about a
    // register, never a limit on `k`: a deeper reduction takes more runs, and the
    // runs combine exactly (§5.1).
    let run_blocks = <L as Lane<E>>::capacity(Bd::VALUE)
        .map(|c| (c / block).max(1))
        .unwrap_or(usize::MAX);
    // The two widths this tile reduces at, resolved once. `wide` is the group
    // the tile height asks for; `single` is the one column an operand's repeats
    // or a shape that does not divide leaves over.
    let mut arg = Sweep {
        wide: *spec,
        // The same sequence when the group is one, which is every widest tile.
        // Looking it up twice would be asking a question whose answer is in the
        // argument.
        single: if group == 1 {
            *spec
        } else {
            E::table_spec(options.backend, Bd::VALUE, rows, 1, block)
        },
        codes,
        stream: C::as_index_stream(codes),
        codes_per_row,
        // Set per column block, from `collapsed` rather than from `index`. The
        // sweep *skips* a repeated column and the expansion below *fills* it,
        // and the two have to be reading the same decision: a sweep that skips
        // on `index` while the expansion fills on `collapsed` leaves every
        // repeat of a narrowed block holding whatever was in the accumulator.
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

        // The collapse applies when this block is the whole output width, because
        // a repeat's first occurrence has to be inside the block to be copied
        // from. A caller whose offer forces narrower blocks gets the uncollapsed
        // traversal, at the same bytes.
        let collapsed = if col0 == 0 && cols == shape.n {
            index
        } else {
            None
        };
        // One decision, read in both places. `sweep` consults this to skip a
        // repeat; the expansion below consults it to fill one.
        arg.index = collapsed;

        let mut p0 = 0usize;
        let mut in_run = 0usize;
        while p0 < blocks {
            let depth = plan.depth.min(blocks - p0);
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
                        acts[packed_slot(t, i, rows, spec.k_group)] =
                            triple.a.at(row0 + i, base + t).get();
                    }
                }
                table.build(spec, block, book, &acts[..block * rows], slot, ledger);
            }

            // No multiply below this line, and no exact accumulator either: the
            // whole chunk reduces into the lane words the previous chunk left,
            // and the placement happens once for the whole run.
            let stack = &table.stack()[..depth * slab];
            let computed = match group {
                16 => sweep::<16, E, Bd, C, L>(&arg, stack, carried, col0, cols, p0, depth),
                8 => sweep::<8, E, Bd, C, L>(&arg, stack, carried, col0, cols, p0, depth),
                4 => sweep::<4, E, Bd, C, L>(&arg, stack, carried, col0, cols, p0, depth),
                2 => sweep::<2, E, Bd, C, L>(&arg, stack, carried, col0, cols, p0, depth),
                _ => sweep::<1, E, Bd, C, L>(&arg, stack, carried, col0, cols, p0, depth),
            };
            ledger.read((computed * depth * rows) as u64);
            ledger.added((computed * depth * rows) as u64);

            p0 += depth;
            in_run += depth;
            // The run is full: place it and start the next one. Unreachable for
            // every `k` a narrow lane covers whole, which is every `k` under
            // 133144 at `(i8, 128)`.
            if in_run + plan.depth > run_blocks {
                place(carried, acc, placed);
                placed = true;
                in_run = 0;
            }
        }
        place(carried, acc, placed);

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
struct Sweep<'c, E: IntegerElement, Bd: Bound, C: Enumerable<E, Bd>, L> {
    wide: TableSpec<E, L>,
    single: TableSpec<E, L>,
    codes: &'c [C::Code],
    /// The operand's own memory, when the codec says it already addresses the
    /// enumeration. Then there is no index stream to build --- the same rule
    /// [`uor_matmul_core::MatView::row_block`] follows on the dense side.
    stream: Option<&'c [u16]>,
    codes_per_row: usize,
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
    E: IntegerElement,
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
            Some(of) => of[col0 + j + u] == col0 + j + u,
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
        // not a heuristic. Both arms compute the same lane words, which is what
        // `CB-08` asserts.
        match arg.stream {
            Some(stream) => one.gather_codes(
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

/// Place a completed run of lane words into the exact accumulator and clear it.
///
/// The only place the exact accumulator appears in the reduction, and for a
/// narrow lane that holds the whole depth it runs once per output element.
fn place<E: Element, L: Lane<E>>(lane: &mut [L], acc: &mut [E::Acc], onto: bool) {
    for (cell, word) in acc.iter_mut().zip(lane.iter_mut()) {
        // The first run sets, the rest combine. Placing onto a zero the caller
        // wrote is the same value and one more pass over the output tile.
        let prior = if onto {
            *cell
        } else {
            <E::Acc as Accumulator>::ZERO
        };
        *cell = word.place(prior);
        *word = L::ZERO;
    }
}

#[allow(clippy::too_many_arguments)]
fn tabulate<E, Bd, C, O, Ep, Lg>(
    triple: &mut TabulatedTriple<'_, '_, '_, E, Bd, C, O>,
    epilogue: &Ep,
    options: GemmOptions,
    exact: &mut [AccOf<E>],
    lanes: &mut [E::Lane],
    panel: &mut [Alphabet<E, Bd>],
    index: Option<&[usize]>,
    plan: Plan,
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
    let space = C::CODE_SPACE;
    let block = <C as uor_matmul_codec::Codec<E, Bd>>::MAX_BLOCK;
    let (columns, stack) = lanes.split_at_mut(plan.rows * plan.cols);

    let mut row0 = 0usize;
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
        // every height no vector sequence has (R13).
        let group = column_group(rows);
        let spec = E::table_spec(options.backend, Bd::VALUE, rows, group, block);
        let Some(mut table) = Table::new(stack, space, rows, plan.depth) else {
            // `Plan::choose` sized the offer for the widest tile it admits and
            // `rows` is never wider, so this cannot be reached. It is written as
            // the streaming traversal rather than as an assertion because an
            // unreachable branch that could produce no output at all is worse
            // than one that produces the right output slowly (C6, R14).
            stream(triple, epilogue, options, panel, ledger);
            return;
        };
        // The panel is `Alphabet<E, Bd>`, a transparent wrapper, so the
        // sequences read the caller's elements themselves and no copy stands
        // between them.
        let (book, acts) = panel.split_at_mut(space * block);
        let book: &[E] = bytemuck::TransparentWrapper::peel_slice(book);
        let acts: &mut [E] = bytemuck::TransparentWrapper::peel_slice_mut(acts);
        row_tile::<E, Bd, C, O, Ep, E::Lane, Lg>(
            triple, epilogue, options, exact, columns, &mut table, &spec, book, acts, index, plan,
            rows, group, row0, ledger,
        );
        row0 += rows;
    }
}

/// What to do when the table is not the answer.
///
/// The tile kernels if the caller's offer holds the decoded operand, and the
/// streaming traversal otherwise. Three factorizations of one identity, ordered by
/// what the offer admits and by which is fastest at the shape --- not by quality,
/// because all three produce the same bytes and `CD-13` says so.
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
    stream(triple, epilogue, options, scratch.take(k), ledger);
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
    ledger.decoded(want as u64);
    ledger.multiplied((shape.m * shape.k * shape.n) as u64);

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
    let Ok(mut dense) = Triple::new(triple.a, b, triple.c.reborrow()) else {
        // The shapes conformed when the `TabulatedTriple` was built and the output
        // was checked for aliasing there, so neither failure can arise a second
        // time. Streaming gives the same bytes, which is why this needs no report.
        return false;
    };
    ledger.kernelled();
    gemm_packed(&mut dense, epilogue, options, &mut Scratch::new(rest));
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
    // The same lane the table uses, at a block of one: a plain dot product is a
    // table of one entry that nobody shares.
    let run = <E::Lane as Lane<E>>::capacity(Bd::VALUE).unwrap_or(usize::MAX);
    for j in 0..shape.n {
        if borrowed {
            triple.w.decode_row_into(j, panel);
            ledger.decoded(shape.k as u64);
        }
        for i in 0..shape.m {
            let row = if borrowed {
                triple.a.row_block(i, 0, 1, shape.k)
            } else {
                None
            };
            let acc = match row {
                // Both operands are runs: the whole reduction is a loop over two
                // contiguous slices in the family's own lane.
                Some(a) => dot_lane::<E, Bd, E::Lane>(a, &panel[..shape.k], run),
                // `A`'s row is not a run, or nothing was offered to decode into.
                // The same accumulation, walked: this is what makes the traversal
                // runnable on a target whose RAM cannot hold one row (S13).
                None => {
                    let mut acc = <AccOf<E> as Accumulator>::ZERO;
                    if borrowed {
                        for (p, w) in panel[..shape.k].iter().enumerate() {
                            E::mac(&mut acc, triple.a.at(i, p).get(), w.get());
                        }
                    } else {
                        for p in 0..shape.k {
                            E::mac(&mut acc, triple.a.at(i, p).get(), triple.w.at(j, p).get());
                        }
                        ledger.decoded(shape.k as u64);
                    }
                    acc
                }
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
    use uor_matmul_codec::{e8_codec, e8_table, Book, Grid, Packed};
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
    /// The two offers move *independently*, and they have to: nothing obliges a
    /// caller who halves the accumulators to halve the lane and index buffers
    /// too, and the documented contract on [`suggested_tabulation`] invites
    /// exactly that ("offering less narrows the column block"). A sweep that
    /// scaled them together left the disagreeing case unreachable, and a
    /// narrowed column block that skipped a repeated column without filling it
    /// lived there --- silently, at 896 wrong cells out of 1024.
    fn tabulated<C: Enumerable<i8, Full<i8>> + Copy>(
        w: &CodedMatrix<'_, i8, Full<i8>, C>,
        a: &[i8],
        m: usize,
        n: usize,
        traversal: Traversal,
        acc_offer: usize,
        aux_offer: usize,
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
        let mut panel = vec![A8::ZERO; scale(want_panel, aux_offer)];
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
                &mut Tabulation::with_index(&mut lane_words, &mut ids),
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
                let (got, census) = tabulated(&w, &a, m, n, traversal, offer, offer);
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
                    tabulated(&w, &a, m, n, Traversal::Tabulated, acc_offer, aux_offer);
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
            let (_, full) = tabulated(&w, &a, m, n, Traversal::Tabulated, OFFER_STEPS, OFFER_STEPS);
            assert!(full.table_reads > 0 || full.kernel_calls > 0);
        }

        // And the comparison is not vacuous: each of the three factorizations is
        // reached by some offer, and the census says which ran rather than the
        // predicate being asked to say it again.
        let (_, with) = tabulated(&w, &a, m, n, Traversal::Tabulated, OFFER_STEPS, OFFER_STEPS);
        let (_, without) = tabulated(&w, &a, m, n, Traversal::Tabulated, 0, 0);
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

        let (got, census) = tabulated(&w, &a, 1, n, Traversal::Tabulated, OFFER_STEPS, OFFER_STEPS);
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

        let (got, census) = tabulated(&w, &a, m, n, Traversal::Blocked, OFFER_STEPS, OFFER_STEPS);
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
        let (streamed, plain) = tabulated(
            &w,
            &a,
            m,
            n,
            Traversal::OutputMajor,
            OFFER_STEPS,
            OFFER_STEPS,
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
}
