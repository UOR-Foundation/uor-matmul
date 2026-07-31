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
use crate::float::{admits_bridge, gemm_float, scaled_f32, Span};
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
    /// Accumulator combines: exact adds at the accumulator's full width, and
    /// the bound-1 build's products, which are issued as signed adds (`CB-10`).
    pub adds: u64,
    /// Reads of a tabulated partial sum.
    pub table_reads: u64,
    /// Calls into the codec's decode.
    pub decodes: u64,
    /// Calls into the dense factorization over a decoded operand --- the tile
    /// kernels for an integer family, the exact float traversal for a float.
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
    E: Element,
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
        let products = (self.code_space * block * self.rows) as u64;
        if spec.build_multiplies {
            ledger.multiplied(products);
        } else {
            // The census counts what was issued, and the bound-1 build issues
            // the products as adds and subtracts (CB-10).
            ledger.added(products);
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

/// What one call's panels declare for the lane: the scale its inputs are
/// pre-scaled by, and the worst one-product magnitude its depth is derived
/// from.
///
/// A fact of the call, not the family --- but for an integer family it is
/// the family's own declaration: bases of zero and the square of the declared
/// bound, with no walk to answer it, because an integer element carries no
/// exponent. The float symbol lane's is the span walk's answer, asked once
/// per call and only when the table was selected (`CD-20`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct LaneScale {
    /// `A`'s elements are pre-scaled by `2^-base_a`: the panel's lowest
    /// decoded exponent.
    pub base_a: i32,
    /// The codebook's, likewise.
    pub base_b: i32,
    /// The worst one-product magnitude the lane must hold: the scaled
    /// panels' per-side bounds multiplied, `2^(2p + wa + wb)` --- per side,
    /// because the lane holds a product of one element of each panel and the
    /// panels' spans differ. The kernel table's single declared bound,
    /// squared, is the same answer at a symmetric span and the conservative
    /// one at an asymmetric one; the lane is not the kernel table and is not
    /// bound by its one-alphabet interface.
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
/// tabulates in the exact accumulator, which is not a placeholder there but the
/// whole of what the hardware offers --- the same status
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
    /// *scaled* products cannot stream in it: the stream walks raw elements
    /// with no span walk ahead of them, so its lane is the one that holds a
    /// raw product exactly --- the complete accumulator, which was the float
    /// families' table lane too before the scaled lane existed.
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
    fn table_spec(
        backend: Backend,
        bound: u128,
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
    /// panel room. `false` when the dense triple cannot be built, which the
    /// caller answers by streaming --- the same bytes by another walk (S13).
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
    /// capacity is a function of the alphabet alone. A family whose real
    /// capacity is a fact of the *data* --- the float symbol lane's is the
    /// span walk's --- plans against the caches instead, and the walk's
    /// answer shrinks the depth once it is in: planning at the data-free
    /// answer, one product at an unbounded alphabet, would pin the stack at
    /// one block for every call, admitted or not.
    fn probe_capacity<L: Lane<Self>>(declared: u128) -> Option<usize> {
        L::capacity(declared)
    }

    /// The scale one call's panels declare for the lane, from a walk of the
    /// activations and the codebook --- or `None`, declining the table, when
    /// the panels are not the lane's alphabet.
    ///
    /// The default answers the integral scale without walking: an integer
    /// element carries no exponent and there is nothing to measure. The float
    /// symbol lane's answer is the placement bridge's span walk (`CD-19`),
    /// over `A` and over the codebook --- not the coded stream: the alphabet
    /// of `W` is the codebook, already materialized, so the walk costs
    /// `m * k + code_space * block` decodes rather than `(m + n) * k`.
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

    /// Scale one element by `2^-base`, exactly, on its way into the table's
    /// inputs.
    ///
    /// Identity for every family the default `lane_scale` serves: their
    /// elements carry no exponent and the base is zero. The float symbol
    /// lane's is a power-of-two rescaling of the code --- bit surgery, not
    /// float arithmetic --- exact by the same arithmetic the bridge's
    /// `rescale` states.
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
        rows: usize,
        group: usize,
        block: usize,
    ) -> TableSpec<i8, i32> {
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
        Self::table_spec(backend, bound, rows, group, block)
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
        Self::table_spec(backend, bound, rows, group, block)
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

/// `f32`: the lane is the scaled integer word, [`Scaled64`]. A float has no
/// magnitude for a bound to measure, so the lane's alphabet is not the
/// element type's but the *call's*: the panels' significands pre-scaled to
/// their measured base exponents, which are exact integers below `2^31` ---
/// the placement bridge's alphabet, one level up (`CD-19`). Over it the table
/// is integer tabulation: the build decodes each codebook entry and each
/// activation into that alphabet once, the gather accumulates eight-byte lane
/// words, and the exact sum is placed at `2^(base_a + base_b)`. At a 256-entry
/// codebook and a four-row tile the slab is 8 KiB where the complete
/// accumulator's was 80, and the table fits L1 --- which is the whole of why
/// this lane exists.
///
/// What the lane cannot hold, the walk declines by declaration: a non-finite
/// code, or a span past the seven binades `24 + w <= 31` admits. Those calls
/// take the dense float factorization at the same bytes, and the census says
/// which ran (`CD-20`).
impl Tabulated for f32 {
    type Lane = Scaled64;
    type ModLane = Scaled64;
    type StreamLane = Wide<AccOf<f32>>;
    const LANE_IS_EXACT: bool = false;

    fn modular_table_admitted(_: u32) -> bool {
        // There is no quotient a float wraps into, as before --- and the
        // scaled lane is not one: it is exact at the declared scale, not
        // congruent modulo a width.
        false
    }

    fn table_spec(
        backend: Backend,
        bound: u128,
        rows: usize,
        group: usize,
        block: usize,
    ) -> TableSpec<f32, Scaled64> {
        // The reference is the only sequence for this family, exact on every
        // alphabet by its own declaration, and its `k_group` is one, which
        // divides every block. The walk's bound is not a selection input:
        // there is no narrower-`max_bound` sequence to admit or decline.
        let _ = (backend, bound, block);
        portable_table::<f32, Scaled64>(rows, group)
    }

    fn table_spec_modular(
        backend: Backend,
        bound: u128,
        rows: usize,
        group: usize,
        block: usize,
    ) -> TableSpec<f32, Scaled64> {
        // Never reached, as `i8`'s.
        Self::table_spec(backend, bound, rows, group, block)
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
        // The lane's capacity is a fact of the panels' spans, which the plan
        // is made before: plan against the caches and let `lane_run` shrink
        // the depth once the walk has answered.
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
        // The bridge's walk and the bridge's declaration (`CD-19`), over `A`
        // and over the codebook. The alphabet of `W` is the codebook, already
        // materialized, so the coded stream itself is never read here.
        let (m, k) = (a.rows(), a.cols());
        let peeled = a.peeled();
        let mut finite = true;
        let mut a_span = Span::EMPTY;
        for i in 0..m {
            for v in peeled.row_walk(i, 0, k) {
                let code = v.pack();
                finite &= code.is_finite();
                a_span.see(code);
            }
        }
        let mut b_span = Span::EMPTY;
        let codec = w.codec();
        let block = <C as uor_matmul_codec::Codec<Self, Bd>>::MAX_BLOCK;
        for index in 0..C::CODE_SPACE {
            for t in 0..block {
                let code = codec.decode_element(C::code_at(index), t).get().pack();
                finite &= code.is_finite();
                b_span.see(code);
            }
        }
        ledger.decoded((m * k + C::CODE_SPACE * block) as u64); // R3-ok: a size or cost query, not an accumulation
        let (wa, wb) = (a_span.width(), b_span.width());
        let bridge = finite
            .then(|| admits_bridge::<Self>(a_span, b_span))
            .flatten()?;
        // The worst one-product magnitude, per side: `2^(p + wa)` of an
        // activation against `2^(p + wb)` of a codebook entry.
        let per_step = 1u128 << (2 * Self::SIGNIFICAND_BITS + wa + wb);
        Some(LaneScale {
            base_a: bridge.base_a,
            base_b: bridge.base_b,
            per_step,
        })
    }

    fn lane_run<L: Lane<Self>>(_: u128, scale: &LaneScale) -> Option<usize> {
        // The lane word is an `i64` and the family declares no other scaled
        // lane, so the cap is `i64`'s own; the depth is the walk's per-side
        // bound, exactly as `capacity` derives it from a single one.
        Some(
            usize::try_from((i64::MAX as u128) / scale.per_step) // R3-ok: a lane-width question, not an accumulation
                .unwrap_or(usize::MAX)
                .max(1),
        )
    }

    fn prescale(x: Self, base: i32) -> Self {
        scaled_f32(x.pack(), base)
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

/// `f64`: no scaled lane. A 53-bit significand is not an `i32` at any span ---
/// the bridge's admission arithmetic answers for the type, not a branch on it
/// --- so the lane is the complete accumulator, with the same status `i32`'s
/// has, and a 256-entry table over 536-byte words fits no L1 tile at any
/// tile height. The dense float factorization is what runs, at the same
/// bytes (`CD-20`).
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
        rows: usize,
        group: usize,
        block: usize,
    ) -> TableSpec<f64, Wide<AccOf<f64>>> {
        // The reference is the only sequence for this family, and its `k_group`
        // is one, which divides every block.
        let _ = (backend, bound, block);
        portable_table::<f64, Wide<AccOf<f64>>>(rows, group)
    }

    fn table_spec_modular(
        backend: Backend,
        bound: u128,
        rows: usize,
        group: usize,
        block: usize,
    ) -> TableSpec<f64, Wide<AccOf<f64>>> {
        // Never reached, as `f32`'s.
        Self::table_spec(backend, bound, rows, group, block)
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
    let repeated = matches!(distinct, Some(d) if d < shape.n);

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
            repeated,
            E::table_spec_modular,
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
            repeated,
            E::table_spec,
            E::lanes,
            ledger,
        );
    }
}

/// The sequence lookup for one lane, as a function pointer: the family's own
/// [`Tabulated::table_spec`] or [`Tabulated::table_spec_modular`], named at the
/// boundary because the two lanes are two types.
type SpecOf<E, L> = fn(Backend, u128, usize, usize, usize) -> TableSpec<E, L>;

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
    repeated: bool,
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
    let Some(plan) = Plan::choose(
        space,
        shape,
        lane_bytes,
        exact_offer,
        offered,
        block,
        E::probe_capacity::<L>(Bd::VALUE),
    ) else {
        decline(triple, epilogue, options, scratch, ledger);
        return;
    };
    // The classes `distinct_columns` found are global, and a global first
    // occurrence is no use to a column block narrower than the output: the
    // class's representative may sit in an earlier block whose accumulator was
    // encoded and `exact` reused before this block began. The map is rewritten
    // block-relative here, once per call, where the block width is known ---
    // and into the dictionary region `distinct_columns` is finished with, so
    // the offer's size and layout are unchanged.
    let collapse = if repeated {
        ColumnMap::derive(index, shape.n, plan.cols)
    } else {
        None
    };
    // Both declarations come from the sequences that will run, at the tile the
    // plan resolved to --- never from a constant standing in for them.
    let table = spec_of(
        options.backend,
        Bd::VALUE,
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
    if !admits(options.traversal, space, block, plan, steps, lane_bytes) {
        decline(triple, epilogue, options, scratch, ledger);
        return;
    }

    // The panels' own declaration, asked only now that the table is selected:
    // a call the predicate declines never pays the walk. `None` declines the
    // table --- the panels are not the lane's alphabet --- and the dense
    // route answers at the same bytes, with the census saying which ran.
    let Some(scale) = E::lane_scale(&triple.a, &triple.w, ledger) else {
        decline(triple, epilogue, options, scratch, ledger);
        return;
    };
    // The walk's capacity answer can only shrink the plan the probe made:
    // rows and columns are capacity-independent, so it is applied here rather
    // than re-planned. For a family whose capacity is a function of the
    // alphabet this is the value `Plan::choose` already used, and the minimum
    // is the identity.
    let mut plan = plan;
    if let Some(run) = E::lane_run::<L>(Bd::VALUE, &scale) {
        plan.depth = (run / block).min(plan.depth).max(1); // R3-ok: a lane-width question, not an accumulation
    }

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
    let (panel, accumulators) = scratch.split(suggested_tabulation_panel(space, block), take);
    if !decode_book(triple.w.codec(), panel, scale.base_b, ledger) {
        decline(triple, epilogue, options, scratch, ledger);
        return;
    }
    let (exact, rest) = accumulators.split_at_mut(tile);
    let Some(lanes) = lanes_of(words, rest, want) else {
        decline(triple, epilogue, options, scratch, ledger);
        return;
    };
    tabulate::<E, Bd, C, O, Ep, Lg, L>(
        triple, epilogue, options, exact, lanes, panel, collapse, plan, scale, spec_of, ledger,
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
/// Each entry is pre-scaled by `2^-base` on the way in: the zero scale leaves
/// every element itself, and the float symbol lane's scale shifts each exponent
/// to the codebook's measured base, so the build's multiply reads integers
/// either way (`CD-20`).
///
/// This is the only reason the tabulated traversal wants a panel offer at all.
/// Without one there is nowhere to put the decoded book, and the traversal
/// streams --- the same rule every other offer in this library follows.
fn decode_book<E, Bd, C, Lg>(
    codec: &C,
    panel: &mut [Alphabet<E, Bd>],
    base: i32,
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
    // The book *and* the widest activation tile a row tile can pack, because the
    // build walks both and a panel that held only the book would leave the tile
    // with nowhere to go.
    if panel.len() < suggested_tabulation_panel(space, block) {
        return false;
    }
    for index in 0..space {
        for t in 0..block {
            let entry = codec.decode_element(C::code_at(index), t);
            panel[index * block + t] =
                bytemuck::TransparentWrapper::wrap(E::prescale(entry.get(), base));
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
    E: Element,
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
    E: Element,
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
    E: Element,
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

impl<'a> ColumnMap<'a> {
    /// Rewrite the global classes in `index[..n]` block-relative, one pass,
    /// into the dictionary region `distinct_columns` is finished with.
    ///
    /// The region-reuse invariant: `distinct_columns` laid the offer out as
    /// `position[n] | slot[table] | key[table]` with
    /// `table = next_power_of_two(2n) >= 2n`, so past `position` there are at
    /// least `4n` dead words and this derivation needs `3n + ceil(n / cols)`.
    /// The offer's size and layout are unchanged. `mark` is zeroed rather than
    /// trusted: the region's prior contents are dictionary probes, which read
    /// as plausible first occurrences.
    fn derive(index: &'a mut [usize], n: usize, cols: usize) -> Option<Self> {
        let blocks = n.div_ceil(cols);
        // Unreachable from `run` --- `distinct_columns` returned only with the
        // region behind it --- but a short offer means the uncollapsed
        // traversal at the same bytes, not a report (C6).
        if index.len() < n.checked_mul(3)?.checked_add(blocks)? {
            return None;
        }
        let (position, rest) = index.split_at_mut(n);
        let (first, rest) = rest.split_at_mut(n);
        let (mark, rest) = rest.split_at_mut(n);
        let (identity, _) = rest.split_at_mut(blocks);
        mark.fill(0);
        identity.fill(1);
        for (j, &rep) in position.iter().enumerate() {
            let start = j / cols * cols;
            let seen = mark[rep];
            // `seen` is a global first occurrence plus one, and `j` rises, so
            // it names this block exactly when it does not predate the block.
            first[j] = if seen > start {
                seen - 1 - start
            } else {
                mark[rep] = j + 1;
                j - start
            };
            if first[j] != j - start {
                identity[j / cols] = 0;
            }
        }
        Some(Self { first, identity })
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
    collapse: Option<ColumnMap<'_>>,
    plan: Plan,
    scale: LaneScale,
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
    let mut arg = Sweep {
        wide: *spec,
        // The same sequence when the group is one, which is every widest tile.
        // Looking it up twice would be asking a question whose answer is in the
        // argument.
        single: if group == 1 {
            *spec
        } else {
            spec_of(options.backend, Bd::VALUE, rows, 1, block)
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
                        // Pre-scaled by the walk's base, as the book is: the
                        // zero scale is the identity, and the float symbol
                        // lane's shifts each exponent to the panel's base, so
                        // the build's multiply reads integers either way.
                        acts[packed_slot(t, i, rows, spec.k_group)] =
                            E::prescale(triple.a.at(row0 + i, base + t).get(), scale.base_a);
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
                place(carried, acc, placed, scale.exponent());
                placed = true;
                in_run = 0;
            }
        }
        place(carried, acc, placed, scale.exponent());

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
    /// [`uor_matmul_core::MatView::row_block`] follows on the dense side.
    stream: Option<&'c [u16]>,
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

#[allow(clippy::too_many_arguments)]
fn tabulate<E, Bd, C, O, Ep, Lg, L>(
    triple: &mut TabulatedTriple<'_, '_, '_, E, Bd, C, O>,
    epilogue: &Ep,
    options: GemmOptions,
    exact: &mut [AccOf<E>],
    lanes: &mut [L],
    panel: &mut [Alphabet<E, Bd>],
    collapse: Option<ColumnMap<'_>>,
    plan: Plan,
    scale: LaneScale,
    spec_of: SpecOf<E, L>,
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
        let spec = spec_of(options.backend, Bd::VALUE, rows, group, block);
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
        row_tile::<E, Bd, C, O, Ep, L, Lg>(
            triple, epilogue, options, exact, columns, &mut table, &spec, spec_of, book, acts,
            collapse, plan, scale, rows, group, row0, ledger,
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
    // The family's own dense driver over the decoded operand: the tile kernels
    // for an integer, the exact float traversal for a float.
    if !E::dense_gemm(triple.a, b, triple.c.reborrow(), epilogue, options, rest) {
        return false;
    }
    ledger.kernelled();
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
    // The lane that holds a *raw* product, at a block of one: the family's own
    // for an integer --- a plain dot product is a table of one entry that
    // nobody shares --- and the complete accumulator for a family whose table
    // lane holds scaled products, because the stream walks raw elements.
    let run = <E::StreamLane as Lane<E>>::capacity(Bd::VALUE).unwrap_or(usize::MAX);
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
                // contiguous slices in the lane that holds raw products.
                Some(a) => dot_lane::<E, Bd, E::StreamLane>(a, &panel[..shape.k], run),
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
    use crate::collapse::suggested_collapse_index;
    use crate::driver::gemm;
    use crate::epilogue::Linear;
    use std::format;
    use std::vec;
    use std::vec::Vec;
    use uor_matmul_codec::{
        canonicalize, e8_codec, e8_table, Arena, Book, Grid, Packed, Sign, SymbolCode, Ternary,
    };
    use uor_matmul_core::{
        as_alphabet, as_alphabet_full, as_alphabet_whole, Bnd, EncodeMode, FloatElement, Full,
        Triple, Whole,
    };

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
    /// scaled integer word (`CD-20`), which lives in the narrow offer; `f64`
    /// has no narrow lane and its suggested lane count is zero, so the knob
    /// scales an empty buffer there exactly as it did before the lane existed.
    ///
    /// Generic over the code width (`CK-14`): the `u8` and `u16` spellings of
    /// one codebook are the same tier at two residencies, and both are asserted
    /// against the same dense bytes.
    #[allow(clippy::too_many_arguments)]
    fn arena_tabulated<E, const D: usize, K: SymbolCode>(
        table: &[Alphabet<E, Whole<E>>; D],
        codes: &[K],
        a: &[E],
        m: usize,
        k: usize,
        n: usize,
        traversal: Traversal,
        offer: usize,
        collapse_offer: usize,
        epilogue: &Linear,
        c0: &[E],
    ) -> (Vec<E>, Census)
    where
        E: FloatElement + EncodeFrom<AccOf<E>> + Tabulated,
        AccOf<E>: crate::SignedPlace,
        Linear: Epilogue<E, E>,
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
    fn arena_reference<E, const D: usize, K: SymbolCode + Into<usize>>(
        table: &[Alphabet<E, Whole<E>>; D],
        codes: &[K],
        a: &[E],
        m: usize,
        k: usize,
        n: usize,
        epilogue: &Linear,
        c0: &[E],
    ) -> Vec<E>
    where
        E: FloatElement + EncodeFrom<AccOf<E>>,
        AccOf<E>: crate::SignedPlace,
        Linear: Epilogue<E, E>,
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

    /// Every traversal at every offer, against the dense float driver's bytes.
    ///
    /// `table_expected` is what the census must show for the forced traversal
    /// at a full offer: `true` when a table both fits and is admitted ---
    /// `f64`, whose lane is the exact accumulator, at a code space whose slab
    /// holds in L1; `f32` at a finite codebook whose exponent span the scaled
    /// lane's alphabet holds. A codebook past either boundary is not a defect:
    /// the lane declines by the same declaration the placement bridge makes
    /// (`CD-19`), the dense route computes the same bytes, and the census's
    /// `table_reads == 0` is the decline witnessed rather than the predicate
    /// trusted twice.
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
        // to the dense route --- a one-element block can never pay --- and an
        // offer of nothing reached neither. The census says which ran rather
        // than the predicate being asked to say it again.
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
                "{label} {m}x{k}x{n}: the lane does not hold these panels, so no table ran \
                 ({tabled:?})"
            );
            assert!(
                tabled.kernel_calls > 0,
                "{label} {m}x{k}x{n}: a declined table at a full offer takes the dense route \
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
            "{label} {m}x{k}x{n}: `Blocked` must decline a one-element block to the \
             dense route ({declined:?})"
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
    }

    /// `CD-14`: an arena-coded float weight matrix gives the dense float
    /// driver's bytes at every shape, with the table forced and declined alike,
    /// and with no offer at all.
    ///
    /// The reference is `gemm_float` over the decoded weights, not another
    /// tabulated run: an agreement between two tabulations would say nothing
    /// about whether either computes the product.
    #[test]
    fn an_arena_coded_float_matrix_matches_the_dense_driver_cd_14() {
        // The distinct symbols one artifact's weights can hold. `-0.0` and
        // `+0.0` are distinct symbols with equal decodes, and an infinity and a
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
                        // The scaled lane's alphabet holds this codebook
                        // exactly while every symbol is finite and its span
                        // fits `24 + w <= 31`: the pool's canonical order puts
                        // the infinity at `d = 4` and the NaN at `d = 5`, and a
                        // non-finite symbol is not an integer, scaled or not.
                        $d <= 3,
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
                        // `f64` declares no scaled lane --- a 53-bit
                        // significand is not an `i32` at any span --- and its
                        // table lane is the exact accumulator, which holds
                        // every one of these code spaces.
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
        // other, and the two zeros are distinct symbols with equal decodes.
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
                        // As `CD-14`: finite and in-span through `d = 3`; the
                        // infinity and the NaN are not the scaled lane's
                        // alphabet and the dense route answers for them.
                        $d <= 3,
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
                        // `f64` declares no scaled lane; its table lane is the
                        // exact accumulator, which holds these code spaces.
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
                // The pool of eight holds the infinity and the NaN, which the
                // scaled lane declines; the dense route reads `C` and answers.
                false,
            );
        }

        // The tier's design point: a codebook that fills the byte. The
        // codebook's symbols are an exact dequantization grid, so every
        // pattern is distinct by construction --- and `canonicalize` is asked
        // rather than told. The grid spans seven binades, which is exactly the
        // widest span the scaled lane's alphabet holds at `f32` (`24 + 7 <=
        // 31`): the design point is the admission boundary, not an interior
        // point of it.
        let mut pool256: Vec<f32> = (0..256).map(|q| (q as f32 - 127.5) * 0.015_625).collect();
        assert_eq!(canonicalize(&mut pool256), 256, "256 distinct bit patterns");
        let t256: &[Alphabet<f32, Whole<f32>>; 256] =
            as_alphabet_whole(&pool256).try_into().unwrap();

        // Over the 80-byte complete accumulator a 256-entry table holds no L1
        // slab at any tile --- `tabulation_fits` says so at a single row ---
        // and that was once the end of the story: the forced traversal
        // declined and the tier's claim was identity and residency. The scaled
        // lane (`CD-20`) is the answer to the lane question that paragraph
        // deferred: at eight bytes a lane word the slab is 2 KiB a row and the
        // table fits, so the forced traversal *tabulates* now, and it is again
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
            // The forced traversal now tabulates: the narrow lane fits the
            // slab and the grid's span is exactly the alphabet's edge. The
            // decodes are the span walk (`m * k` of `A`, plus the codebook)
            // and the one book decode, counted so the walk's price is read
            // off the census and not re-derived.
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
            assert_eq!(
                tabled.decodes,
                (m * k + 2 * 256) as u64,
                "the span walk and the book decode are the decodes ({tabled:?})"
            );
            // Under the costed traversal the one-element block still declines:
            // the op-count predicate prices a table read against one dense
            // product and finds no win, which `CG-16` measures against the
            // clock. At no offer it is the stream.
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
            assert_eq!(declined.table_reads, 0, "a one-element block never pays");
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
            assert_eq!(streamed.kernel_calls, 0);
            assert_eq!(
                streamed.multiplies,
                (m * k * n) as u64,
                "an offer of nothing streams ({streamed:?})"
            );
        }
    }

    /// `CD-20`: a `u8`-symbol-coded float weight matrix tabulated in the
    /// scaled integer lane gives the dense float driver's bytes at every shape
    /// and every offer, including none, with the span walk admitted and
    /// declined alike, and a reduction deeper than the lane's run chunked
    /// exactly.
    ///
    /// The lane is the placement bridge's identity one level up (`CD-19`):
    /// the codebook and the activation tile are pre-scaled to the panels'
    /// measured base exponents, the table is built over the scaled integer
    /// alphabet the bridge defines, the gather accumulates `i64` lane words,
    /// and the exact sum is placed at `2^(base_a + base_b)`. The reference is
    /// `gemm_float` over the decoded weights, as `CD-18`'s: an agreement
    /// between two coded traversals would say nothing about whether either
    /// computes the product.
    #[test]
    fn the_scaled_lane_tabulation_matches_the_dense_driver_cd_20() {
        // A codebook of 256 distinct finite symbols whose values span exactly
        // `span` binades: significands in `[2^22, 2^23)`, so every value's
        // exponent is `22 + e` and no two collapse, and `e` cycling through
        // `span + 1` values. Seven binades is the widest span the lane's
        // alphabet holds (`24 + 7 <= 31`), so the sweep stands on the
        // admission boundary rather than inside it.
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
            // Deeper than the lane's run at the widest admitted span --- 127
            // products at seven binades against the one-binade `A` below ---
            // so the reduction is cut into runs and placed more than once: a
            // lane that dropped a carry between runs would write different
            // bytes here.
            (3, 300, 5),
        ];
        for span in [0u64, 3, 7] {
            let pool = codebook(span);
            let table: &[Alphabet<f32, Whole<f32>>; 256] =
                as_alphabet_whole(&pool).try_into().unwrap();
            for &(m, k, n) in shapes {
                // Every byte value is a live code (`CT-07`); `A` spans one
                // binade, so the walk admits at every `span` above.
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
                    // Finite and in span: the table runs.
                    true,
                );
            }
        }

        // Nine binades is past the alphabet (`24 + 9 > 31`): the lane declines
        // by the bridge's own declaration and the dense route answers, at the
        // same bytes. A non-finite symbol is not an integer at any scale, and
        // a non-finite activation is not either: likewise.
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
                false,
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
                false,
            );
            // One NaN in `A` declines the lane however admissible the book is.
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
                false,
            );
        }

        // `f64` declines by declaration: a 53-bit significand is not an `i32`
        // at any span, so the family has no scaled lane --- its table lane is
        // the exact accumulator, over which a 256-entry table fits no L1 tile.
        // The dense route answers, at the same bytes.
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
        // expansion copies the sums. A span that fits admits the same table on
        // the compacted rows, and the bytes are the dense driver's.
        let pool = codebook(3);
        let table: &[Alphabet<f32, Whole<f32>>; 256] = as_alphabet_whole(&pool).try_into().unwrap();
        let (m, k, n) = (7usize, 40usize, 9usize);
        let codes: Vec<u8> = fill(n * k, 0xa4ea, |x| x as u8);
        // Rows in threes: two repeats of every row, so the collapse has work
        // to do and the recursion's own span walk is exercised.
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
            true,
        );
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
            ($t:ty, $label:expr, $pool:expr, $nan1:expr, $nan2:expr, $build:expr) => {{
                let label = $label;
                let table: &[Alphabet<$t, Whole<$t>>; 8] =
                    as_alphabet_whole(&$pool).try_into().unwrap();
                let (k, n) = (4usize, 9usize);
                // Codes past the table on purpose: the enumeration reduces them
                // modulo `D`, and the reference decodes them the same way.
                let codes: Vec<u16> = fill(n * k, 0xa4ea, |x| (x % 17) as u16);
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
                let m = 8usize;
                for d in [1usize, 2, m / 2, m] {
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
                        let (got, census) = arena_tabulated(
                            table,
                            &codes,
                            &a,
                            m,
                            k,
                            n,
                            Traversal::Tabulated,
                            OFFER_STEPS,
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
                        // The build charges per *distinct* row when the collapse
                        // ran, and per row when anything declined it. What one
                        // charged row is charged *for* depends on the route
                        // below the collapse: the table builds `space` entries
                        // per element of the reduction, the dense route computes
                        // `n` products. The scaled lane declines a non-finite
                        // panel by declaration (`CD-20`) --- and `symbols`
                        // carries NaNs and an infinity --- so `f32`'s route is
                        // the dense one and the witness is `n`; `f64`'s lane is
                        // the complete accumulator, which holds them, and its
                        // witness is the table's `space`. Either way the count
                        // moves with `d`, which is the collapse's evidence.
                        let charged = if collapse_offer >= d * k && d < m {
                            d
                        } else {
                            m
                        };
                        assert_eq!(
                            census.multiplies,
                            (charged * k * $build) as u64,
                            "{label} {m}x{k}x{n} d={d} rows offer {collapse_offer}: the build \
                             must charge per distinct row ({census:?})"
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
                // decides: six rows, four symbols. Rows 0 and 1 differ only in
                // the sign of a zero; rows 2 and 3 only in a NaN payload; rows
                // 4 and 5 repeat rows 0 and 2 bit for bit. Numeric `==` would
                // merge the first pair and refuse the repeats --- five rows it
                // can call distinct, where bit identity sees four.
                let m = 6usize;
                let a: Vec<$t> = vec![
                    1.0, -0.0, 0.5, -1.5, // row 0
                    1.0, 0.0, 0.5, -1.5, // row 1: the sign of a zero, and nothing else
                    $nan1, 1.0, 0.5, -1.5, // row 2
                    $nan2, 1.0, 0.5, -1.5, // row 3: a NaN payload, and nothing else
                    1.0, -0.0, 0.5, -1.5, // row 4: row 0's bits again
                    $nan1, 1.0, 0.5, -1.5, // row 5: row 2's bits again
                ];
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
                    let (got, census) = arena_tabulated(
                        table,
                        &codes,
                        &a,
                        m,
                        k,
                        n,
                        Traversal::Tabulated,
                        OFFER_STEPS,
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
                    assert_eq!(
                        census.multiplies,
                        (charged * k * $build) as u64,
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
            // The scaled lane declines the NaN rows and the non-finite
            // codebook, so the route below the collapse is the dense one and
            // a charged row is `n` products --- nine, the `n` bound inside.
            9
        );
        case!(
            f64,
            "Arena<8> f64",
            pool64,
            f64::from_bits(0x7ff8_0000_0000_0001),
            f64::from_bits(0x7ff8_0000_0000_0002),
            // The complete accumulator holds every code, so the table runs
            // and a charged row is the build's `space` entries per element.
            8
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
                    // The build charges per *distinct* row when the collapse
                    // ran, and per row when anything declined it.
                    let charged = if rows_offer >= d * k && d < m { d } else { m };
                    assert_eq!(
                        census.multiplies,
                        (charged * k * space) as u64,
                        "{m}x{k}x{n} d={d} rows offer {rows_offer}: the build must charge \
                         per distinct row ({census:?})"
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
            census.multiplies,
            (rows_d * k * space) as u64,
            "both axes: the build charges per distinct row ({census:?})"
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
    /// lane is four lane words per accumulator, so the same stack costs a
    /// quarter as many, and the offer below covers the modular stack and not
    /// the exact one. The census then says which lane ran, rather than the
    /// predicate being asked to say it again.
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
        let tile = rows * n;
        // `tile + ceil((tile + slab * rows * blocks) / 4)`: the modular take,
        // derived in the doc comment above. The exact take at this offer is
        // `tile + tile + slab * rows` --- one slot deep, which is all the
        // remainder buys at one accumulator per word --- and it exceeds the
        // offer, so the exact lane must decline.
        let offer = tile + (tile + slab * rows * (k / block)).div_ceil(4);
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
}
