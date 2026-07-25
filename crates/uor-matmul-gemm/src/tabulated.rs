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
// The table
// ---------------------------------------------------------------------------

/// The tabulation buffer for one row tile and one block of the reduction.
///
/// Borrowed, never owned: it lives in the caller's accumulator offer like every
/// other working buffer in this library (R7, S13). `code_space * rows`
/// accumulator words, row-major in the *code* index, so the column loop reads
/// consecutive words for the `rows` outputs of one code and the prefetcher sees a
/// stride it can follow.
#[derive(Debug)]
pub struct Table<'s, A: Accumulator> {
    words: &'s mut [A],
    code_space: usize,
    rows: usize,
}

impl<'s, A: Accumulator> Table<'s, A> {
    /// How many accumulator words a table for `rows` rows of a `code_space`-wide
    /// enumeration occupies.
    ///
    /// A query, so an embedded caller can size a static and know the answer
    /// before it calls anything.
    pub const fn words(code_space: usize, rows: usize) -> usize {
        code_space.saturating_mul(rows)
    }

    /// Borrow `words` as the table for `rows` rows.
    ///
    /// `None` when the borrow is shorter than the table it is asked to be, which
    /// means no such table exists in that offer. Decided here, before any
    /// arithmetic, and answered by the caller taking the streaming traversal
    /// instead --- not by an error reaching anyone (C6).
    pub fn new(words: &'s mut [A], code_space: usize, rows: usize) -> Option<Self> {
        if code_space == 0 || rows == 0 || words.len() < Self::words(code_space, rows) {
            return None;
        }
        Some(Self {
            words,
            code_space,
            rows,
        })
    }

    /// Distinct codes this table is indexed by.
    pub const fn code_space(&self) -> usize {
        self.code_space
    }

    /// Rows of `A` this table covers.
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Every entry for one block of the reduction.
    ///
    /// `code_space * block * rows` products, and the only multiplies the tabulated
    /// traversal issues at all.
    ///
    /// Codes outer, rows inner. That order is not free to choose: it decodes each
    /// codeword's `block` elements once per *tile* rather than once per row, which
    /// is `code_space * block` decode calls instead of `code_space * block * rows`
    /// --- and for a book codec whose table is in L2, the decode is the expensive
    /// half.
    pub fn build<E, Bd, C, L>(
        &mut self,
        codec: &C,
        a: &MatView<'_, Alphabet<E, Bd>>,
        row0: usize,
        rows: usize,
        block_index: usize,
        ledger: &mut L,
    ) where
        E: IntegerElement + Element<Acc = A>,
        Bd: Bound,
        C: Enumerable<E, Bd>,
        L: Ledger,
    {
        let block = C::MAX_BLOCK;
        let base = block_index * block;
        // A ragged last tile builds fewer entries and keeps the table's stride, so
        // the addressing is the same whether the shape divides or not (`CS-06`).
        let rows = rows.min(self.rows);
        let stride = self.rows;
        for index in 0..self.code_space {
            let code = C::code_at(index);
            let entry = &mut self.words[index * stride..index * stride + rows];
            entry.fill(A::ZERO);
            for t in 0..block {
                // One decode, then `rows` products against it. This is the only
                // place the codec is consulted: below the build there are no
                // codes left, only indices.
                let w = codec.decode_element(code, t).get();
                ledger.decoded(1);
                for (i, slot) in entry.iter_mut().enumerate() {
                    E::mac(slot, a.at(row0 + i, base + t).get(), w);
                }
                ledger.multiplied(rows as u64);
            }
        }
    }

    /// One code's entry: the partial sum for each of the tile's first `rows` rows.
    ///
    /// A contiguous run, which is why the code index is the *outer* index of the
    /// layout. The column loop hands this straight to [`add_entry`] and the two
    /// slices walk together.
    ///
    /// `rows` is the tile's height rather than the table's, because the last tile
    /// of a shape that does not divide is shorter. The stride stays the table's,
    /// so a ragged tile changes what is read and never where it is read from.
    ///
    /// No bounds check is needed in principle: [`Enumerable::index_of`] is total
    /// below `CODE_SPACE` and the buffer is `CODE_SPACE` entries wide by
    /// construction, which is what `CT-07` asserts.
    #[inline(always)]
    pub fn entry(&self, code_index: usize, rows: usize) -> &[A] {
        let at = code_index * self.rows;
        &self.words[at..at + rows.min(self.rows)]
    }

    /// One tabulated partial sum.
    #[inline(always)]
    pub fn get(&self, code_index: usize, row: usize) -> A {
        self.words[code_index * self.rows + row]
    }
}

/// One code's contribution to a tile of output columns: one table read and one
/// exact add per row, and no multiply.
///
/// Two slices of the same length, walked together. There is no index arithmetic
/// in here at all, which is not tidiness --- it is what makes `CU-06`'s
/// disassembly half a statement about the *accumulation* rather than about
/// addressing.
#[inline(always)]
fn add_entry<A: Accumulator>(entry: &[A], acc: &mut [A]) {
    for (slot, &partial) in acc.iter_mut().zip(entry) {
        *slot = slot.combine(partial);
    }
}

/// The accumulation of [`add_entry`] at one concrete instantiation, so that
/// `CU-06`'s disassembly gate has a symbol to read.
///
/// Not a second path and not a test hook: it is the same function, named once at
/// a width the shipped integer families resolve to, because a generic function
/// emits no code until something instantiates it and a gate cannot read
/// instructions that were never emitted.
#[inline(never)]
pub fn add_entry_wide(entry: &[i128], acc: &mut [i128]) {
    add_entry(entry, acc);
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
// Selection
// ---------------------------------------------------------------------------

/// Does a table for `rows` rows of a `code_space`-wide enumeration fit the
/// declared cache budget?
///
/// The factor of two leaves half of L1 for the code stream and the output
/// accumulators. A table that does not sit in L1 turns every output column into a
/// cache miss, and the traversal stops paying long before the op count says it
/// should --- which is why residency is a term of the predicate and not an
/// afterthought.
pub const fn tabulation_fits(
    code_space: usize,
    rows: usize,
    l1_bytes: usize,
    acc_bytes: usize,
) -> bool {
    code_space > 0
        && rows > 0
        && code_space
            .saturating_mul(rows)
            .saturating_mul(acc_bytes)
            .saturating_mul(2)
            <= l1_bytes
}

/// Does tabulation issue fewer operations than blocking, and does its table fit?
///
/// `cols` is the width of the column block the caller's offer supports, which is
/// `n` when the offer holds the whole output width. The build is repeated once per
/// column block, so it is the block and not the shape that the op count turns on.
///
/// `code_space * block` products to build a table, plus `cols` reads and adds to
/// use it, against `cols * block` products to do it densely. So tabulation is
/// cheaper exactly when `cols * (block - 1) > code_space * block`.
///
/// `block == 1` is refused. One code naming one element removes every multiply and
/// no add, so no `cols` makes the op count cross; the return there is that a read
/// is cheaper than a widening multiply, which is a claim about instructions and
/// belongs to a measurement rather than to this predicate.
pub const fn tabulation_pays(
    code_space: usize,
    block: usize,
    cols: usize,
    rows: usize,
    l1_bytes: usize,
    acc_bytes: usize,
) -> bool {
    block > 1
        && cols.saturating_mul(block - 1) > code_space.saturating_mul(block)
        && tabulation_fits(code_space, rows, l1_bytes, acc_bytes)
}

/// The most rows of `A` one table can cover and still sit in L1.
///
/// Derived from the cache budget and the code space, and capped by the same
/// `MC` the blocked traversal uses --- not by a number chosen for this traversal
/// (R8). Zero means no table fits at all, which selects the streaming traversal.
pub const fn tabulation_rows(code_space: usize, l1_bytes: usize, acc_bytes: usize) -> usize {
    if code_space == 0 || acc_bytes == 0 {
        return 0;
    }
    let room = l1_bytes / (2 * code_space * acc_bytes);
    if room < blocking::MC {
        room
    } else {
        blocking::MC
    }
}

/// How many exact accumulators would let the tabulated traversal run at the whole
/// output width for this shape and this codec.
///
/// A *query*, like [`crate::suggested_scratch`]. Offering less narrows the column
/// block; offering none gives the same bytes from the streaming traversal
/// (`CD-13`). It does not grow with `k`.
pub fn suggested_tabulation<E: IntegerElement>(shape: Shape, code_space: usize) -> usize {
    let acc_bytes = core::mem::size_of::<AccOf<E>>();
    let rows = tabulation_rows(code_space, blocking::L1_BYTES, acc_bytes)
        .min(shape.m)
        .max(1);
    // The table, plus one exact partial sum per output cell of the row tile.
    Table::<AccOf<E>>::words(code_space, rows).saturating_add(rows.saturating_mul(shape.n))
}

// ---------------------------------------------------------------------------
// The traversal
// ---------------------------------------------------------------------------

/// `C := epilogue(A * W^T, C)`, with the table when the offer admits one.
///
/// Returns `()`, for the same reason [`crate::gemm`] does.
pub fn gemm_tabulated<E, Bd, C, O, Ep>(
    triple: &mut TabulatedTriple<'_, '_, '_, E, Bd, C, O>,
    epilogue: &Ep,
    options: GemmOptions,
    scratch: &mut Scratch<'_, E, Bd>,
) where
    E: IntegerElement,
    Bd: Bound,
    C: Enumerable<E, Bd>,
    O: Element + EncodeFrom<AccOf<E>>,
    Ep: Epilogue<E, O>,
{
    run(triple, epilogue, options, scratch, &mut ());
}

/// The same traversal, with the operation census written out.
///
/// Not a second path: [`gemm_tabulated`] is this function at `L = ()`, where every
/// ledger call has an empty body and disappears. `CU-06` reads the census this
/// returns and `CD-13` asserts the two give the same bytes.
pub fn gemm_tabulated_counted<E, Bd, C, O, Ep>(
    triple: &mut TabulatedTriple<'_, '_, '_, E, Bd, C, O>,
    epilogue: &Ep,
    options: GemmOptions,
    scratch: &mut Scratch<'_, E, Bd>,
    census: &mut Census,
) where
    E: IntegerElement,
    Bd: Bound,
    C: Enumerable<E, Bd>,
    O: Element + EncodeFrom<AccOf<E>>,
    Ep: Epilogue<E, O>,
{
    run(triple, epilogue, options, scratch, census);
}

/// The row tile and column block this offer supports, or `None` for neither.
///
/// Widest column block first, then as many rows as what is left allows. That
/// order is derived, not preferred: the build is repeated once per column block
/// and shared across the rows of a tile, so a narrow block multiplies the
/// products while a short tile only multiplies the decodes.
fn partition(offered: usize, cols: usize, code_space: usize, row_cap: usize) -> (usize, usize) {
    if code_space == 0 || cols == 0 {
        return (0, 0);
    }
    let per_row = cols.saturating_add(code_space);
    let rows = (offered / per_row.max(1)).min(row_cap);
    if rows >= 1 {
        return (rows, cols);
    }
    // Not enough for one row at the whole width: one row, and as wide a block as
    // the remainder buys.
    (1, offered.saturating_sub(code_space).min(cols))
}

fn run<E, Bd, C, O, Ep, L>(
    triple: &mut TabulatedTriple<'_, '_, '_, E, Bd, C, O>,
    epilogue: &Ep,
    options: GemmOptions,
    scratch: &mut Scratch<'_, E, Bd>,
    ledger: &mut L,
) where
    E: IntegerElement,
    Bd: Bound,
    C: Enumerable<E, Bd>,
    O: Element + EncodeFrom<AccOf<E>>,
    Ep: Epilogue<E, O>,
    L: Ledger,
{
    let shape = triple.shape();
    if shape.m == 0 || shape.n == 0 {
        // Nothing to write. Not a special case: the loops below would do the same
        // thing, and saying so costs one comparison.
        return;
    }

    let space = C::CODE_SPACE;
    let block = <C as uor_matmul_codec::Codec<E, Bd>>::MAX_BLOCK;
    let acc_bytes = core::mem::size_of::<AccOf<E>>();
    let row_cap = tabulation_rows(space, blocking::L1_BYTES, acc_bytes)
        .min(shape.m)
        .max(1);
    let (rows, cols) = partition(scratch.accumulators(), shape.n, space, row_cap);

    // A code stream whose blocks are not a fixed width has no `p`-th block to
    // index, so there is nothing for a table to be built per block of. The one
    // such tier does not implement `Enumerable`, so this is unreachable through
    // the shipped codecs; it is here because the trait does not forbid it.
    let addressable = <C as uor_matmul_codec::Codec<E, Bd>>::IS_FIXED_WIDTH && block >= 1;

    let tabulate = addressable
        && rows >= 1
        && cols >= 1
        && match options.traversal {
            // Streaming was asked for by name.
            Traversal::OutputMajor => false,
            // The default: take the table when it is the cheaper factorization.
            Traversal::Blocked => {
                tabulation_pays(space, block, cols, rows, blocking::L1_BYTES, acc_bytes)
            }
            // Named by the caller: take the table wherever one fits, whether or
            // not the op count says it wins. `CD-13` needs this to compare bytes
            // on both sides of the predicate, and a caller measuring its own
            // shapes needs it for the same reason.
            Traversal::Tabulated => tabulation_fits(space, rows, blocking::L1_BYTES, acc_bytes),
        };

    if !tabulate {
        stream(triple, epilogue, options, ledger);
        return;
    }

    let blocks = shape.k / block;
    let reads_c = epilogue.reads_c();
    let codes_per_row = triple.w.codes_per_row();
    let want = Table::<AccOf<E>>::words(space, rows) + rows * cols;
    let (_, accumulators) = scratch.split(0, want);
    let (tile, words) = accumulators.split_at_mut(rows * cols);
    let mut table = match Table::new(words, space, rows) {
        Some(t) => t,
        // `partition` sized the offer for exactly this, so a `None` here means
        // the offer shrank under us, which it cannot. Streaming gives the same
        // bytes either way, which is why this needs no report (C6).
        None => {
            stream(triple, epilogue, options, ledger);
            return;
        }
    };

    let mut row0 = 0usize;
    while row0 < shape.m {
        let tile_rows = rows.min(shape.m - row0);
        let mut col0 = 0usize;
        while col0 < shape.n {
            let tile_cols = cols.min(shape.n - col0);
            let acc = &mut tile[..tile_rows * tile_cols];
            acc.fill(<AccOf<E> as Accumulator>::ZERO);

            for p in 0..blocks {
                table.build::<E, Bd, C, L>(triple.w.codec(), &triple.a, row0, tile_rows, p, ledger);

                // No multiply below this line. `acc` is column-major within the
                // tile and the table is code-major, so both sides of the add are
                // contiguous runs of `tile_rows` words.
                for j in 0..tile_cols {
                    let code = triple.w.codes()[(col0 + j) * codes_per_row + p];
                    let index = C::index_of(code);
                    add_entry(
                        table.entry(index, tile_rows),
                        &mut acc[j * tile_rows..j * tile_rows + tile_rows],
                    );
                    ledger.read(tile_rows as u64);
                    ledger.added(tile_rows as u64);
                }
            }

            // The single encode step, exactly once per output element.
            for i in 0..tile_rows {
                for j in 0..tile_cols {
                    let (r, c) = (row0 + i, col0 + j);
                    let prior = if reads_c {
                        Some(*triple.c.at(r, c))
                    } else {
                        None
                    };
                    *triple.c.at_mut(r, c) =
                        epilogue.finish(acc[j * tile_rows + i], prior, options.encode);
                }
            }
            col0 += tile_cols;
        }
        row0 += tile_rows;
    }
}

/// The same identity with no table: decode, accumulate exactly, encode once.
///
/// Not a fallback. It is [`Traversal::OutputMajor`] for this operand orientation,
/// it needs no offer at all, and `CD-13` asserts it produces the same bytes as the
/// table does. A caller on a target whose RAM cannot hold a table gets this and
/// loses nothing but time.
fn stream<E, Bd, C, O, Ep, L>(
    triple: &mut TabulatedTriple<'_, '_, '_, E, Bd, C, O>,
    epilogue: &Ep,
    options: GemmOptions,
    ledger: &mut L,
) where
    E: IntegerElement,
    Bd: Bound,
    C: Enumerable<E, Bd>,
    O: Element + EncodeFrom<AccOf<E>>,
    Ep: Epilogue<E, O>,
    L: Ledger,
{
    let shape = triple.shape();
    let reads_c = epilogue.reads_c();
    for i in 0..shape.m {
        for j in 0..shape.n {
            let mut acc = <AccOf<E> as Accumulator>::ZERO;
            for p in 0..shape.k {
                // `W` is transposed by declaration, so the weight of `(p, j)` is
                // element `p` of coded row `j`.
                E::mac(&mut acc, triple.a.at(i, p).get(), triple.w.at(j, p).get());
            }
            ledger.multiplied(shape.k as u64);
            ledger.decoded(shape.k as u64);
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

    /// One tabulated product at one traversal and one offer.
    fn tabulated<C: Enumerable<i8, Full<i8>> + Copy>(
        w: &CodedMatrix<'_, i8, Full<i8>, C>,
        a: &[i8],
        m: usize,
        n: usize,
        traversal: Traversal,
        offer: usize,
    ) -> (Vec<i32>, Census) {
        let k = w.cols();
        let mut accumulators = vec![<AccOf<i8> as Accumulator>::ZERO; offer];
        let mut panel: Vec<A8> = Vec::new();
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
                &mut census,
            );
        }
        (c, census)
    }

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

        let full = suggested_tabulation::<i8>(Shape { m, k, n }, C::CODE_SPACE);
        let offers = [
            0,
            1,
            C::CODE_SPACE,
            C::CODE_SPACE + 5,
            full.saturating_sub(1),
            full,
            full.saturating_mul(3),
        ];
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
        let (_, with) = tabulated(&w, &a, m, n, Traversal::Tabulated, full);
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

        let full = suggested_tabulation::<i8>(Shape { m: 1, k, n }, 256);
        let (got, census) = tabulated(&w, &a, 1, n, Traversal::Tabulated, full);
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
    /// - `multiplies == decodes * rows`. Every multiply the traversal issued is
    ///   attached to a decode, and decodes happen only in the table build. A
    ///   multiply anywhere else breaks this equality.
    /// - `multiplies == m * k * code_space`, independent of `n`. The build is the
    ///   only arithmetic that scales with the code space, and it does not scale
    ///   with the output width at all.
    #[test]
    fn the_tabulated_column_loop_has_no_multiply_cu_06() {
        let table = e8_table::<Full<i8>>().expect("i8 holds E8");
        let book = e8_codec(&table);
        let space = 256usize;
        let block = 8usize;
        let (m, k, n) = (8usize, 32usize, 2048usize);
        let blocks = k / block;

        let rows =
            tabulation_rows(space, blocking::L1_BYTES, core::mem::size_of::<AccOf<i8>>()).min(m);
        assert!(
            rows >= 1 && m % rows == 0,
            "an exact tiling, so the closed forms are exact"
        );

        let stream: Vec<u16> = fill(n * blocks, 0xc0de, |x| (x % 256) as u16);
        let w = CodedMatrix::new(book, n, k, &stream).expect("the codes describe n x k");
        let a: Vec<i8> = fill(m * k, activation_salt(), |x| ((x % 255) as i64 - 127) as i8);

        let full = suggested_tabulation::<i8>(Shape { m, k, n }, space);
        let (got, census) = tabulated(&w, &a, m, n, Traversal::Blocked, full);
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
            census.multiplies,
            census.decodes * rows as u64,
            "every multiply is attached to a decode, so every multiply is in the build: {census:?}"
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
        let (streamed, plain) = tabulated(&w, &a, m, n, Traversal::OutputMajor, full);
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
        let acc = core::mem::size_of::<AccOf<i8>>();
        let l1 = blocking::L1_BYTES;
        let rows = tabulation_rows(256, l1, acc);
        assert!(
            rows >= 1,
            "a 256-entry table must fit L1 for at least one row"
        );

        // E8: the first `n` that pays is 293, and 292 does not.
        assert!(tabulation_pays(256, 8, 293, rows, l1, acc));
        assert!(!tabulation_pays(256, 8, 292, rows, l1, acc));
        // A nibble pair: 513, and not 512.
        assert!(tabulation_pays(256, 2, 513, rows, l1, acc));
        assert!(!tabulation_pays(256, 2, 512, rows, l1, acc));
        // One element per code: no `n` at all.
        assert!(!tabulation_pays(16, 1, usize::MAX, rows, l1, acc));
        // A table nobody can hold is refused whatever the op count says.
        assert!(!tabulation_pays(1 << 16, 8, usize::MAX, 1, l1, acc));
        assert_eq!(tabulation_rows(1 << 16, l1, acc), 0);
        // And an enumeration of nothing has no table.
        assert!(!tabulation_fits(0, 1, l1, acc));
    }
}
