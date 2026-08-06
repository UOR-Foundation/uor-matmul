//! The parity checks, shared by the host test and the Cortex-M executor.
//!
//! `tests/parity.rs` runs these bodies on the host against the whole corpus
//! with a heap; `uor-matmul-executor` runs them on a Cortex-M target under
//! qemu-system against the reduced corpus with caller-static scratch, and
//! `CB-11` is that run. Two harnesses, one implementation --- a second copy of
//! the check bodies would be a second way of asserting the same claim (R13).
//!
//! Everything here is `no_std` and allocation-free: every buffer is the
//! caller's, sized for the corpus the caller walks. A disagreement panics with
//! the check, the backend, and the depth in the message --- a parity check is
//! an assertion, not a fallible operation, so there is no error channel for
//! R8's audit to read, and the executor's panic handler turns the panic into
//! its `FAIL` marker.

use uor_matmul_core::{as_alphabet_full, dot_ref, Backend};

use crate::table::{choose_table, Mod32, Mod64, TableSpec};
use crate::tropical::{TROP_I16_MAX_BOUND, TROP_I8_BOUND, TROP_ZERO};
use crate::{packed_slot, KernelSpec, LaneLayout};

/// The whole depth corpus: every tail and every threshold the sequences have.
pub const DEPTHS_FULL: &[usize] = &[0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 63, 64, 65, 129, 512];

/// The reduced corpus: few instances, every path.
///
/// The Miri corpus's discipline, and the executor's --- every arm of
/// `dispatch_run!` and `dispatch_slab!` is the same macro body at different
/// constants, so a narrow arm and a wide arm together exercise the code, and
/// one instance of a path shows parity as well as three hundred do. The
/// executor's static scratch is sized for this corpus and the reduced table
/// corpora below; the host runs the full ones.
pub const DEPTHS_REDUCED: &[usize] = &[0, 1, 8, 65];

/// The shapes a table sweep walks.
#[derive(Clone, Copy)]
pub struct TableCorpus {
    /// The code spaces, including one that is not a power of two: the slab is
    /// rounded up, and the padding must not reach any answer.
    pub spaces: &'static [usize],
    /// The block lengths.
    pub blocks: &'static [usize],
    /// The tile heights.
    pub rows: &'static [usize],
    /// The column groups.
    pub groups: &'static [usize],
    /// The gather depth.
    pub depth: usize,
}

/// The table corpus for the 32-bit lane families, in full.
pub const TABLE_32_FULL: TableCorpus = TableCorpus {
    spaces: &[16, 64, 200, 256],
    blocks: &[2, 4, 8],
    rows: &[1, 2, 4, 8, 16],
    groups: &[1, 2, 4, 8, 16],
    depth: 5,
};

/// The same, reduced. The executor's corpus for every family.
pub const TABLE_REDUCED: TableCorpus = TableCorpus {
    spaces: &[200, 256],
    blocks: &[2, 8],
    rows: &[1, 16],
    groups: &[1, 16],
    depth: 5,
};

/// The table corpus for the 64-bit lane families, in full.
pub const TABLE_64_FULL: TableCorpus = TableCorpus {
    spaces: &[16, 200, 256],
    blocks: &[2, 8],
    rows: &[1, 8, 16],
    groups: &[1, 2],
    depth: 5,
};

/// The shapes a bound-1 sweep walks.
#[derive(Clone, Copy)]
pub struct Bound1Corpus {
    /// `(space, block)` pairs: the sign codec's spaces, and 256, which is also
    /// the ternary spelling's.
    pub pairs: &'static [(usize, usize)],
    /// The tile heights.
    pub rows: &'static [usize],
    /// The column groups.
    pub groups: &'static [usize],
    /// The tiles the selection half walks.
    pub sel_tiles: &'static [(usize, usize)],
    /// The blocks the selection half walks.
    pub sel_blocks: &'static [usize],
}

/// Every tile height and group the driver walks.
const ROWS_1_TO_16: &[usize] = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

/// The bound-1 corpus, in full.
pub const BOUND1_FULL: Bound1Corpus = Bound1Corpus {
    pairs: &[(4, 2), (16, 4), (256, 8)],
    rows: ROWS_1_TO_16,
    groups: ROWS_1_TO_16,
    sel_tiles: &[(16, 1), (16, 2), (8, 1), (8, 2), (1, 16), (1, 1)],
    sel_blocks: &[2, 4, 8],
};

/// The same, reduced.
pub const BOUND1_REDUCED: Bound1Corpus = Bound1Corpus {
    pairs: &[(4, 2), (256, 8)],
    rows: &[1, 16],
    groups: &[1, 16],
    sel_tiles: &[(16, 1), (16, 2), (8, 1), (8, 2), (1, 16), (1, 1)],
    sel_blocks: &[2, 4, 8],
};

/// Deterministic fill, into the caller's buffer. A recorded generator rather
/// than a crate, so a failure reproduces from the salt alone.
pub fn fill_into<T>(buf: &mut [T], salt: u64, map: impl Fn(i64) -> T) {
    let mut s = 0x243F_6A88_85A3_08D3u64 ^ salt;
    for x in buf {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *x = map((s >> 33) as i64);
    }
}

/// The depths a kernel can be handed: the corpus entries rounded up to whole
/// `k`-groups, which is what the driver always packs. A rounding collision
/// reruns a depth --- the host used to dedup into a `Vec`, which a heap-less
/// harness has nothing to dedup into, and a rerun costs time and nothing else.
pub fn depths_of(k_group: usize, corpus: &[usize]) -> impl Iterator<Item = usize> + '_ {
    corpus.iter().map(move |&kc| kc.div_ceil(k_group) * k_group)
}

/// The two fills of a parity sweep: one map for A's panel, one for B's.
pub type FillMaps<E> = (fn(i64) -> E, fn(i64) -> E);

/// The reference dot for the exact `i8` family: `dot_ref` itself, the anchor
/// `CB-01` pins the whole chain to.
pub fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    dot_ref(as_alphabet_full(a), as_alphabet_full(b)) as i32
}

/// The reference dot for the exact `i16` family: widen, multiply, add.
pub fn dot_i16(a: &[i16], b: &[i16]) -> i64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| i64::from(x) * i64::from(y))
        .sum()
}

/// The reference dot for the exact `i32` family: widen, multiply, add.
pub fn dot_i32_exact(a: &[i32], b: &[i32]) -> i64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| i64::from(x) * i64::from(y))
        .sum()
}

/// The reference dot for the exact `i64` family: widen, multiply, add.
pub fn dot_i64_exact(a: &[i64], b: &[i64]) -> i128 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| i128::from(x) * i128::from(y))
        .sum()
}

/// The reference dot for the modular `i16` family. The lane wraps, and that
/// *is* the answer in the quotient --- the reference wraps on purpose.
pub fn dot_i16_mod(a: &[i16], b: &[i16]) -> i32 {
    a.iter().zip(b).fold(0i32, |s, (&x, &y)| {
        s.wrapping_add(i32::from(x).wrapping_mul(i32::from(y)))
    })
}

/// The reference dot for the modular `i32` family; the wrap is the encode.
pub fn dot_i32_mod(a: &[i32], b: &[i32]) -> i32 {
    a.iter()
        .zip(b)
        .fold(0i32, |s, (&x, &y)| s.wrapping_add(x.wrapping_mul(y)))
}

/// The reference dot for the modular `i64` family; the wrap is the encode.
pub fn dot_i64_mod(a: &[i64], b: &[i64]) -> i64 {
    a.iter()
        .zip(b)
        .fold(0i64, |s, (&x, &y)| s.wrapping_add(x.wrapping_mul(y)))
}

/// The reference dot for the packed tropical family: `max_p (a_p ⊗ b_p)`, with
/// `⊗` the saturating add and `⊕` a signed max.
///
/// The fold starts at [`TROP_ZERO`] and not at `i16::default()`, which is the
/// one place a `(max, +)` reference cannot borrow the ring's shape: the identity
/// of `max` is the semiring zero, so an empty reduction is `-inf` and a fold
/// from `0` would report it as the largest finite answer there is.
///
/// Zip-truncating, exactly as [`uor_matmul_core::dot_tropical_ref`] is, and
/// written out here rather than delegating to it because it is the *packed*
/// arithmetic that the sequences have to agree with. The link between the two
/// --- pack, run, unpack, and get `dot_tropical_ref`'s answer --- is the anchor
/// half of `CB-13`, and it is a separate assertion for the same reason `CB-01`
/// is separate from `CB-02`.
pub fn dot_trop_i16(a: &[i16], b: &[i16]) -> i16 {
    let mut best = TROP_ZERO;
    for (&x, &y) in a.iter().zip(b) {
        let product = x.saturating_add(y); // R3-ok: the absorbing law, not an accumulation
        best = best.max(product);
    }
    best
}

/// A packed tropical fill at a declared bound: finite values spread across
/// `[-bound, bound]`, with one position in eight at the semiring zero.
///
/// The masked positions are in the ordinary sweep rather than in a special
/// case, because absorption is where a `(max, +)` sequence goes wrong: a fill
/// of finite values alone never issues the saturating add the representation
/// turns on, and would pass against a wrapping one.
fn trop_fill(v: i64, bound: i64) -> i16 {
    if v % 8 == 0 {
        TROP_ZERO
    } else {
        (v.rem_euclid(2 * bound + 1) - bound) as i16
    }
}

/// The tropical fill at the narrowest alphabet past the empty one.
pub fn trop_fill_bound1(v: i64) -> i16 {
    trop_fill(v, 1)
}

/// The tropical fill at the `Trop<i8>` alphabet, which is what the packing is
/// for.
pub fn trop_fill_i8(v: i64) -> i16 {
    trop_fill(v, TROP_I8_BOUND as i64)
}

/// The tropical fill at the widest alphabet the packed lane declares, where the
/// finite range and the absorbed range are a single value apart.
pub fn trop_fill_max(v: i64) -> i16 {
    trop_fill(v, TROP_I16_MAX_BOUND as i64)
}

/// One leg of the tropical sweep: the two panel salts, and the fill that
/// realizes one declared bound.
pub type TropSweep = ((u64, u64), fn(i64) -> i16);

/// The declared bounds `CB-13` sweeps, each with its salts and its fill.
///
/// One list, read by both harnesses, so the host sweep and the Cortex-M sweep
/// cannot come to walk different alphabets while reporting the same ID --- the
/// same reason the check bodies themselves live here (R13).
///
/// Three bounds, and each asks a different question. At `1` the finite range is
/// three values wide and the absorbed one is the rest of the lane. At
/// [`crate::tropical::TROP_I8_BOUND`] it is the alphabet the packing exists
/// for. At [`TROP_I16_MAX_BOUND`] the two ranges are a single value apart,
/// which is where a wrapping add, or an unpacking that compared against the
/// wrong threshold, stops agreeing.
pub const TROP_SWEEPS: [TropSweep; 3] = [
    ((35, 36), trop_fill_bound1),
    ((37, 38), trop_fill_i8),
    ((39, 40), trop_fill_max),
];

/// The tropical extremes, by name.
///
/// A random fill reaches "both operands at the semiring zero" with probability
/// `2^-6` per position and "both at exactly `-B`" with probability `2^-28`. The
/// first is the input that separates a saturating add from a wrapping one, and
/// the second is the input that separates the declared bound from one past it,
/// so neither is left to a generator.
pub const TROP_EXTREMES: [(i16, i16); 9] = {
    let b = TROP_I16_MAX_BOUND as i16;
    [
        (-b, -b),
        (-b, b),
        (b, b),
        (b, -b),
        (TROP_ZERO, b),
        (b, TROP_ZERO),
        (TROP_ZERO, -b),
        (-b, TROP_ZERO),
        (TROP_ZERO, TROP_ZERO),
    ]
};

/// The packed answer for a constant tropical panel: `max` over `kc` copies of
/// one product, which is that product --- and the semiring zero when there is
/// no product at all, because the identity of `max` is `-inf` and not zero.
pub fn trop_expect(kc: usize, a: i16, b: i16) -> i16 {
    if kc == 0 {
        TROP_ZERO
    } else {
        a.saturating_add(b) // R3-ok: the absorbing law, not an accumulation
    }
}

/// The buffers a tile parity check runs on. Sizes are named for the largest
/// walk the caller's corpus reaches.
pub struct TileScratch<'a, E, L> {
    /// The packed A panel: at least `mr * kc`.
    pub pa: &'a mut [E],
    /// The packed B panel: at least `nr * kc`.
    pub pb: &'a mut [E],
    /// One lane of A's depth, for the reference's reads: at least `kc`.
    pub lane_a: &'a mut [E],
    /// The same, for B.
    pub lane_b: &'a mut [E],
    /// The tile under test: at least `mr * nr` lanes.
    pub acc: &'a mut [L],
    /// The reference tile.
    pub want: &'a mut [L],
}

/// Every tile sequence in `specs` equals `dot`, at every depth in `corpus`.
///
/// The panel layout is the kernel contract, so the reference reads the panels
/// through [`packed_slot`] rather than restating the layout: a kernel that
/// misreads its panel fails here, which is the whole point. Returns the number
/// of (sequence, depth) comparisons, so the harness can assert the sweep was
/// not vacuous.
pub fn tile_parity<E: Copy, L: Copy + Default + PartialEq + core::fmt::Debug>(
    check: &'static str,
    specs: impl Iterator<Item = KernelSpec<E, L>>,
    corpus: &[usize],
    salts: (u64, u64),
    maps: FillMaps<E>,
    dot: fn(&[E], &[E]) -> L,
    scratch: &mut TileScratch<'_, E, L>,
) -> usize {
    let TileScratch {
        pa,
        pb,
        lane_a,
        lane_b,
        acc,
        want,
    } = scratch;
    let mut compared = 0;
    for spec in specs {
        assert_eq!(
            spec.lane_layout,
            LaneLayout::Interleaved,
            "{check}: {:?} is not a tile sequence",
            spec.backend
        );
        for kc in depths_of(spec.k_group, corpus) {
            assert!(
                spec.mr * kc <= pa.len()
                    && spec.nr * kc <= pb.len()
                    && kc <= lane_a.len()
                    && kc <= lane_b.len()
                    && spec.mr * spec.nr <= acc.len(),
                "{check}: the harness undersized the scratch for {:?} at kc={kc}",
                spec.backend
            );
            fill_into(&mut pa[..spec.mr * kc], kc as u64 ^ salts.0, maps.0);
            fill_into(&mut pb[..spec.nr * kc], kc as u64 ^ salts.1, maps.1);
            let acc = &mut acc[..spec.mr * spec.nr];
            acc.fill(L::default());
            spec.mac_tile(kc, &pa[..spec.mr * kc], &pb[..spec.nr * kc], acc);
            let want = &mut want[..spec.mr * spec.nr];
            for i in 0..spec.mr {
                for p in 0..kc {
                    lane_a[p] = pa[packed_slot(p, i, spec.mr, spec.k_group)];
                }
                for j in 0..spec.nr {
                    for p in 0..kc {
                        lane_b[p] = pb[packed_slot(p, j, spec.nr, spec.k_group)];
                    }
                    want[i * spec.nr + j] = dot(&lane_a[..kc], &lane_b[..kc]);
                }
            }
            assert_eq!(
                acc, want,
                "{check}: {:?} disagrees at kc={kc}",
                spec.backend
            );
            compared += 1;
        }
    }
    compared
}

/// The buffers a reduce parity check runs on.
pub struct ReduceScratch<'a, E, L> {
    /// The panel: at least `mr * kc`, with row `i` contiguous at `i * kc`.
    pub pa: &'a mut [E],
    /// The single column: at least `kc`.
    pub pb: &'a mut [E],
    /// The output rows: at least `mr`.
    pub acc: &'a mut [L],
}

/// Every reduce sequence in `specs` equals `dot` read across the rows, at
/// every depth in `corpus`.
///
/// The reduce factorization puts the lanes on `k`, so its panel layout is the
/// contiguous one and its reference reads rows rather than strides. What this
/// asserts is that moving the lanes does not move the answer.
pub fn reduce_parity<E: Copy, L: Copy + Default + PartialEq + core::fmt::Debug>(
    check: &'static str,
    specs: impl Iterator<Item = KernelSpec<E, L>>,
    corpus: &[usize],
    salts: (u64, u64),
    maps: FillMaps<E>,
    dot: fn(&[E], &[E]) -> L,
    scratch: &mut ReduceScratch<'_, E, L>,
) -> usize {
    let ReduceScratch { pa, pb, acc } = scratch;
    let mut compared = 0;
    for spec in specs {
        assert_eq!(spec.nr, 1, "{check}: a reduce kernel produces one column");
        assert_eq!(
            spec.lane_layout,
            LaneLayout::Contiguous,
            "{check}: {:?} is not a reduce sequence",
            spec.backend
        );
        for kc in depths_of(spec.k_group, corpus) {
            assert!(
                spec.mr * kc <= pa.len() && kc <= pb.len() && spec.mr <= acc.len(),
                "{check}: the harness undersized the scratch for {:?} at kc={kc}",
                spec.backend
            );
            fill_into(&mut pa[..spec.mr * kc], kc as u64 ^ salts.0, maps.0);
            fill_into(&mut pb[..kc], kc as u64 ^ salts.1, maps.1);
            let acc = &mut acc[..spec.mr];
            acc.fill(L::default());
            spec.mac_tile(kc, &pa[..spec.mr * kc], &pb[..kc], acc);
            for i in 0..spec.mr {
                let want = dot(&pa[i * kc..][..kc], &pb[..kc]);
                assert_eq!(
                    acc[i], want,
                    "{check}: {:?} disagrees at kc={kc} row {i}",
                    spec.backend
                );
            }
            compared += 1;
        }
    }
    compared
}

/// The largest and smallest value an alphabet bounded by `bound` holds, within
/// the element type.
///
/// A bound of `B` admits magnitudes up to `B`, and the type's own negative
/// extreme is `-B`; the positive one is `B - 1` when the bound is the type's,
/// and `B` when the caller declared it.
pub fn extremes(bound: u128, type_bound: u128) -> (i64, i64) {
    let b = bound.min(type_bound) as i64;
    if bound >= type_bound {
        (-b, b - 1)
    } else {
        (-b, b)
    }
}

/// Every tile sequence is exact on constant panels at the named values.
///
/// A random fill will not find the one input where a paired-product
/// instruction overflows its intermediate --- that needs both operands at an
/// extreme at both `k` of a pair, which a generator reaches with probability
/// `2^-64`. So the extremes are asked for by name. `expect` computes the
/// constant tile's value from the depth and the two fills.
pub fn tile_extremes<E: Copy, L: Copy + Default + PartialEq + core::fmt::Debug>(
    check: &'static str,
    specs: impl Iterator<Item = KernelSpec<E, L>>,
    corpus: &[usize],
    values: &[(E, E)],
    expect: fn(usize, E, E) -> L,
    scratch: &mut TileScratch<'_, E, L>,
) -> usize {
    let TileScratch {
        pa,
        pb,
        lane_a: _,
        lane_b: _,
        acc,
        want: _,
    } = scratch;
    let mut compared = 0;
    for spec in specs {
        for kc in depths_of(spec.k_group, corpus) {
            assert!(
                spec.mr * kc <= pa.len()
                    && spec.nr * kc <= pb.len()
                    && spec.mr * spec.nr <= acc.len(),
                "{check}: the harness undersized the scratch for {:?} at kc={kc}",
                spec.backend
            );
            for &(a, b) in values {
                pa[..spec.mr * kc].fill(a);
                pb[..spec.nr * kc].fill(b);
                let acc = &mut acc[..spec.mr * spec.nr];
                acc.fill(L::default());
                spec.mac_tile(kc, &pa[..spec.mr * kc], &pb[..spec.nr * kc], acc);
                let want = expect(kc, a, b);
                assert!(
                    acc.iter().all(|&x| x == want),
                    "{check}: {:?} disagrees on a constant panel at kc={kc}",
                    spec.backend
                );
                compared += 1;
            }
        }
    }
    compared
}

/// The same for the reduce sequences: they accumulate a whole row into one
/// lane, so their intermediate widths are a different question from the tile
/// sequences' and deserve the same input.
pub fn reduce_extremes<E: Copy, L: Copy + Default + PartialEq + core::fmt::Debug>(
    check: &'static str,
    specs: impl Iterator<Item = KernelSpec<E, L>>,
    corpus: &[usize],
    values: &[(E, E)],
    expect: fn(usize, E, E) -> L,
    scratch: &mut ReduceScratch<'_, E, L>,
) -> usize {
    let ReduceScratch { pa, pb, acc } = scratch;
    let mut compared = 0;
    for spec in specs {
        for kc in depths_of(spec.k_group, corpus) {
            assert!(
                spec.mr * kc <= pa.len() && kc <= pb.len() && spec.mr <= acc.len(),
                "{check}: the harness undersized the scratch for {:?} at kc={kc}",
                spec.backend
            );
            for &(a, b) in values {
                pa[..spec.mr * kc].fill(a);
                pb[..kc].fill(b);
                let acc = &mut acc[..spec.mr];
                acc.fill(L::default());
                spec.mac_tile(kc, &pa[..spec.mr * kc], &pb[..kc], acc);
                let want = expect(kc, a, b);
                assert!(
                    acc.iter().all(|&x| x == want),
                    "{check}: {:?} disagrees on a constant panel at kc={kc}",
                    spec.backend
                );
                compared += 1;
            }
        }
    }
    compared
}

/// How one family's model arithmetic reads and writes its lane.
///
/// The table checks compare against a transcribed model oracle ---
/// `T[c][i] = sum_t A[i][t] * D[c][t]` is the whole definition --- rather
/// than against the reference sequence's own run, because at the tile heights
/// no ISA serves, the reference is the only party to the comparison and
/// reading it against itself would pass whatever it did. That keeps the checks
/// meaningful on a portable-only target, which is exactly what the executor
/// is.
pub struct ModelOps<E, L, R> {
    /// One build step of the model's arithmetic.
    pub mac: fn(R, E, E) -> R,
    /// One gather step of the model's arithmetic.
    pub add: fn(R, R) -> R,
    /// The additive identity.
    pub zero: R,
    /// Read a lane as the model's arithmetic type.
    pub raw: fn(L) -> R,
    /// Write the model's arithmetic type into a lane.
    pub into: fn(R) -> L,
}

/// The model arithmetic for the exact `i8` table family.
pub fn ops_i8() -> ModelOps<i8, i32, i32> {
    ModelOps {
        mac: |s, a, b| s + i32::from(a) * i32::from(b),
        add: |a, b| a + b,
        zero: 0,
        raw: |l| l,
        into: |r| r,
    }
}

/// The model arithmetic for the exact `i16` table family.
pub fn ops_i16() -> ModelOps<i16, i64, i64> {
    ModelOps {
        mac: |s, a, b| s + i64::from(a) * i64::from(b),
        add: |a, b| a + b,
        zero: 0,
        raw: |l| l,
        into: |r| r,
    }
}

/// The model arithmetic for the modular `i32` table family. The modular lane
/// is `Z/2^32`: the build's products wrap and the gather's adds wrap, and both
/// are the ring's own operations, so the model oracle is written in wrapping
/// arithmetic.
pub fn ops_mod32() -> ModelOps<i32, Mod32, i32> {
    ModelOps {
        mac: |s, a, b| s.wrapping_add(a.wrapping_mul(b)),
        add: i32::wrapping_add,
        zero: 0,
        raw: |m| m.0,
        into: Mod32,
    }
}

/// The model arithmetic for the modular `i64` table family.
pub fn ops_mod64() -> ModelOps<i64, Mod64, i64> {
    ModelOps {
        mac: |s, a, b| s.wrapping_add(a.wrapping_mul(b)),
        add: i64::wrapping_add,
        zero: 0,
        raw: |m| m.0,
        into: Mod64,
    }
}

/// A bounded stack value for the gather half, 32-bit lane. Bounded so the
/// exact lane's model sum cannot leave its width at any depth a corpus walks.
pub fn stack_value32(v: i64) -> i32 {
    ((v % 4096) - 2048) as i32
}

/// The same, 64-bit lane.
pub fn stack_value64(v: i64) -> i64 {
    (v % 4096) - 2048
}

/// The buffers a table parity check runs on.
pub struct TableScratch<'a, E, L> {
    /// The codebook: `space * block`.
    pub book: &'a mut [E],
    /// The activation tile, flat: `rows * block`.
    pub flat: &'a mut [E],
    /// The same tile, packed the way a spec declares it: `rows * block`.
    pub packed: &'a mut [E],
    /// The model slot: `space * rows` lanes.
    pub model: &'a mut [L],
    /// A sequence's slot: `space * rows` lanes.
    pub out: &'a mut [L],
    /// The gather stack: `depth * slab` lanes.
    pub stack: &'a mut [L],
    /// The offset stream: `depth * group`.
    pub off: &'a mut [u32],
    /// The code stream: `(group - 1) * (depth + 3) + depth`.
    pub stream: &'a mut [u16],
    /// A gather lane: `group * rows`.
    pub lane_a: &'a mut [L],
    /// The reference gather lane.
    pub lane_b: &'a mut [L],
}

/// Every table sequence equals the transcribed model, lane for lane: the
/// build, the gather, the gather over ragged offsets, and the gather read from
/// a code stream.
///
/// `skip_below` drops the build sequences whose declared alphabet the fill
/// leaves: a sequence that declares a narrower alphabet --- the bound-1 build
/// --- is exact only within its declaration, and reading it against data
/// outside that is not a comparison but a contract breach; [`table_bound1`]
/// holds it to the bound it declares. `ragged` asks for the ragged-offset
/// half: offsets that are not multiples of the tile height must have their
/// sub-row bits cleared, because a read that starts inside the last entry runs
/// past the slab --- a panic in the reference and an out-of-bounds read in
/// every ISA sequence.
#[allow(clippy::too_many_arguments)]
pub fn table_parity<E, L, R, I>(
    check: &'static str,
    available: impl Fn(usize, usize) -> I,
    corpus: &TableCorpus,
    ops: &ModelOps<E, L, R>,
    salts: (u64, u64),
    map: fn(i64) -> E,
    stack_map: fn(i64) -> R,
    sentinels: (R, R),
    skip_below: u128,
    ragged: bool,
    scratch: &mut TableScratch<'_, E, L>,
) -> usize
where
    E: Copy,
    L: Copy + PartialEq + core::fmt::Debug,
    R: Copy,
    I: Iterator<Item = TableSpec<E, L>>,
{
    let TableScratch {
        book,
        flat,
        packed,
        model,
        out,
        stack,
        off,
        stream,
        lane_a,
        lane_b,
    } = scratch;
    let depth = corpus.depth;
    let mut compared = 0;
    for &space in corpus.spaces {
        for &block in corpus.blocks {
            fill_into(&mut book[..space * block], salts.0 ^ space as u64, map);
            for &rows in corpus.rows {
                fill_into(&mut flat[..rows * block], salts.1 ^ rows as u64, map);
                for &group in corpus.groups {
                    let mut specs = available(rows, group);
                    let reference = specs
                        .next()
                        .unwrap_or_else(|| panic!("{check}: no sequence at {rows}x{group}"));
                    assert_eq!(
                        reference.backend,
                        Backend::Portable,
                        "{check}: the reference is listed first"
                    );

                    // The build, against the model.
                    for c in 0..space {
                        for i in 0..rows {
                            let mut acc = ops.zero;
                            for t in 0..block {
                                acc = (ops.mac)(acc, flat[t * rows + i], book[c * block + t]);
                            }
                            model[c * rows + i] = (ops.into)(acc);
                        }
                    }
                    pack_into(
                        &flat[..rows * block],
                        rows,
                        block,
                        reference.k_group,
                        packed,
                    );
                    let want = &mut out[..space * rows];
                    reference.build(
                        space,
                        block,
                        &book[..space * block],
                        &packed[..rows * block],
                        want,
                    );
                    assert_eq!(
                        want,
                        &model[..space * rows],
                        "{check}: the reference build disagrees with the model at space {space}, \
                         block {block}, rows {rows}"
                    );
                    compared += 1;
                    for spec in available(rows, group).skip(1) {
                        if spec.max_bound < skip_below {
                            continue;
                        }
                        pack_into(&flat[..rows * block], rows, block, spec.k_group, packed);
                        // `model` is spent once the reference was read against
                        // it; every later sequence reads against the reference.
                        let got = &mut model[..space * rows];
                        spec.build(
                            space,
                            block,
                            &book[..space * block],
                            &packed[..rows * block],
                            got,
                        );
                        assert_eq!(
                            got, want,
                            "{check}: {:?} build disagrees at space {space}, block {block}, \
                             rows {rows}",
                            spec.backend
                        );
                        compared += 1;
                    }

                    // The gather, over a stack whose slab is the rounded space.
                    let codes = space.next_power_of_two();
                    let slab = codes * rows;
                    assert!(
                        depth * slab <= stack.len(),
                        "{check}: the harness undersized the stack"
                    );
                    for cell in stack[..depth * slab].iter_mut() {
                        *cell = (ops.into)(ops.zero);
                    }
                    let mut s = 0x243F_6A88_85A3_08D3u64;
                    for slot in 0..depth {
                        for i in 0..space * rows {
                            s ^= s << 13;
                            s ^= s >> 7;
                            s ^= s << 17;
                            stack[slot * slab + i] = (ops.into)(stack_map((s >> 33) as i64));
                        }
                    }
                    for i in 0..depth * group {
                        off[i] = ((i * 37 % space) * rows) as u32;
                    }
                    // The model per lane position: addition is commutative in
                    // both the exact and the modular arithmetic, so summing per
                    // position equals the original's per-slot accumulation.
                    for u in 0..group {
                        for i in 0..rows {
                            let mut acc = sentinels.0;
                            for slot in 0..depth {
                                let at = off[slot * group + u] as usize & (slab - 1);
                                acc = (ops.add)(acc, (ops.raw)(stack[slot * slab + at + i]));
                            }
                            lane_a[u * rows + i] = (ops.into)(acc);
                        }
                    }
                    let want = &mut lane_b[..group * rows];
                    want.fill((ops.into)(sentinels.0));
                    reference.gather(
                        depth,
                        slab as u32,
                        &stack[..depth * slab],
                        &off[..depth * group],
                        want,
                    );
                    assert_eq!(
                        want,
                        &lane_a[..group * rows],
                        "{check}: the reference gather disagrees with the model at space {space}, \
                         rows {rows}, group {group}"
                    );
                    compared += 1;
                    for spec in available(rows, group).skip(1) {
                        let got = &mut lane_a[..group * rows];
                        got.fill((ops.into)(sentinels.0));
                        spec.gather(
                            depth,
                            slab as u32,
                            &stack[..depth * slab],
                            &off[..depth * group],
                            got,
                        );
                        assert_eq!(
                            got, want,
                            "{check}: {:?} gather disagrees at space {space}, rows {rows}, \
                             group {group}",
                            spec.backend
                        );
                        compared += 1;
                    }

                    // Offsets that are NOT multiples of the tile height.
                    if ragged && rows > 1 {
                        for i in 0..depth * group {
                            off[i] = ((i * 37 + 1) % (codes * rows)) as u32;
                        }
                        for u in 0..group {
                            for i in 0..rows {
                                let mut acc = ops.zero;
                                for slot in 0..depth {
                                    let at =
                                        off[slot * group + u] as usize & (slab - 1) & !(rows - 1);
                                    acc = (ops.add)(acc, (ops.raw)(stack[slot * slab + at + i]));
                                }
                                lane_a[u * rows + i] = (ops.into)(acc);
                            }
                        }
                        for spec in available(rows, group) {
                            let got = &mut lane_b[..group * rows];
                            got.fill((ops.into)(ops.zero));
                            spec.gather(
                                depth,
                                slab as u32,
                                &stack[..depth * slab],
                                &off[..depth * group],
                                got,
                            );
                            assert_eq!(
                                got,
                                &lane_a[..group * rows],
                                "{check}: {:?} disagrees on a ragged offset at space {space}, \
                                 rows {rows}, group {group}",
                                spec.backend
                            );
                            compared += 1;
                        }
                    }

                    // The same reduction read from a code stream instead of an
                    // index stream. Only where a codec could claim it: the
                    // enumeration has to be addressed by the code, which needs a
                    // power-of-two space.
                    if codes == space {
                        let stride = depth + 3;
                        for i in 0..(group - 1) * stride + depth {
                            stream[i] = ((i * 37) % space) as u16;
                        }
                        for i in 0..depth * group {
                            let (slot, u) = (i / group, i % group);
                            off[i] = (stream[u * stride + slot] as usize % space * rows) as u32;
                        }
                        for u in 0..group {
                            for i in 0..rows {
                                let mut acc = sentinels.1;
                                for slot in 0..depth {
                                    let at =
                                        (stream[u * stride + slot] as usize & (codes - 1)) * rows;
                                    acc = (ops.add)(acc, (ops.raw)(stack[slot * slab + at + i]));
                                }
                                lane_a[u * rows + i] = (ops.into)(acc);
                            }
                        }
                        let want = &mut lane_b[..group * rows];
                        want.fill((ops.into)(sentinels.1));
                        reference.gather(
                            depth,
                            slab as u32,
                            &stack[..depth * slab],
                            &off[..depth * group],
                            want,
                        );
                        assert_eq!(
                            want,
                            &lane_a[..group * rows],
                            "{check}: the reference gather disagrees with the model read by code \
                             at space {space}, rows {rows}, group {group}"
                        );
                        compared += 1;
                        for spec in available(rows, group) {
                            let got = &mut lane_a[..group * rows];
                            got.fill((ops.into)(sentinels.1));
                            spec.gather_codes(
                                depth,
                                slab as u32,
                                &stack[..depth * slab],
                                &stream[..(group - 1) * stride + depth],
                                stride,
                                got,
                            );
                            assert_eq!(
                                got, want,
                                "{check}: {:?} gather_codes disagrees at space {space}, rows \
                                 {rows}, group {group}",
                                spec.backend
                            );
                            compared += 1;
                        }

                        // The same reduction read from a byte-wide code
                        // stream: the same values, so `want` still stands, at
                        // half the stream's storage. Only where a byte holds
                        // the code, which is what the u8 spellings store.
                        if space <= 256 {
                            let len = (group - 1) * stride + depth;
                            // SAFETY: the u16 leg is done with `stream`, and
                            // the byte view covers `len` bytes of its
                            // `len` u16s, inside the same allocation.
                            let stream8 = unsafe {
                                core::slice::from_raw_parts_mut(
                                    stream.as_mut_ptr().cast::<u8>(),
                                    len,
                                )
                            };
                            for (i, cell) in stream8.iter_mut().enumerate() {
                                *cell = ((i * 37) % space) as u8;
                            }
                            for i in 0..depth * group {
                                let (slot, u) = (i / group, i % group);
                                off[i] =
                                    (stream8[u * stride + slot] as usize % space * rows) as u32;
                            }
                            for spec in available(rows, group) {
                                let got = &mut lane_a[..group * rows];
                                got.fill((ops.into)(sentinels.1));
                                spec.gather_codes_u8(
                                    depth,
                                    slab as u32,
                                    &stack[..depth * slab],
                                    &stream8[..len],
                                    stride,
                                    got,
                                );
                                assert_eq!(
                                    got, want,
                                    "{check}: {:?} gather_codes_u8 disagrees at space {space}, \
                                     rows {rows}, group {group}",
                                    spec.backend
                                );
                                compared += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    compared
}

/// `CB-10`: every bound-1 build equals the model, and selection offers the
/// adds-only build exactly when the declared bound admits it; a wider finite
/// alphabet lookup build may also issue no multiplies.
///
/// At bound 1 every product is `+-a` or `0`, so a build can fill the same slot
/// the reference fills with adds and subtracts alone. The selection half is
/// `CB-07`'s rule at the bound-1 declaration: the bound-1 builds are listed
/// after every full-alphabet sequence, so `Auto` takes one exactly at bound 1
/// and never one past it.
#[allow(clippy::too_many_arguments)]
pub fn table_bound1<E, L, R, I>(
    check: &'static str,
    available: impl Fn(usize, usize) -> I,
    corpus: &Bound1Corpus,
    ops: &ModelOps<E, L, R>,
    salts: (u64, u64),
    map: fn(i64) -> E,
    scratch: &mut TableScratch<'_, E, L>,
) -> usize
where
    E: Copy,
    L: Copy + PartialEq + core::fmt::Debug,
    R: Copy,
    I: Iterator<Item = TableSpec<E, L>>,
{
    let TableScratch {
        book,
        flat,
        packed,
        model,
        out,
        stack: _,
        off: _,
        stream: _,
        lane_a: _,
        lane_b: _,
    } = scratch;
    let mut compared = 0;
    for &(space, block) in corpus.pairs {
        fill_into(&mut book[..space * block], salts.0 ^ space as u64, map);
        for &rows in corpus.rows {
            fill_into(&mut flat[..rows * block], salts.1 ^ rows as u64, map);
            for &group in corpus.groups {
                let mut specs = available(rows, group);
                let reference = specs
                    .next()
                    .unwrap_or_else(|| panic!("{check}: no sequence at {rows}x{group}"));

                for c in 0..space {
                    for i in 0..rows {
                        let mut acc = ops.zero;
                        for t in 0..block {
                            acc = (ops.mac)(acc, flat[t * rows + i], book[c * block + t]);
                        }
                        model[c * rows + i] = (ops.into)(acc);
                    }
                }
                pack_into(
                    &flat[..rows * block],
                    rows,
                    block,
                    reference.k_group,
                    packed,
                );
                let want = &mut out[..space * rows];
                reference.build(
                    space,
                    block,
                    &book[..space * block],
                    &packed[..rows * block],
                    want,
                );
                assert_eq!(
                    want,
                    &model[..space * rows],
                    "{check}: the reference build disagrees with the model at bound 1, space \
                     {space}, block {block}, rows {rows}"
                );
                compared += 1;

                let mut bound1 = 0usize;
                for spec in available(rows, group).skip(1) {
                    // A full-alphabet sequence at bound-1 data is the ordinary
                    // sweep's ground; this one reads the sequences that declare
                    // the bound.
                    if spec.max_bound > 1 {
                        continue;
                    }
                    bound1 += 1;
                    assert!(
                        !spec.build_multiplies,
                        "{check}: {:?} declares bound 1 but still multiplies",
                        spec.backend
                    );
                    pack_into(&flat[..rows * block], rows, block, spec.k_group, packed);
                    let got = &mut model[..space * rows];
                    spec.build(
                        space,
                        block,
                        &book[..space * block],
                        &packed[..rows * block],
                        got,
                    );
                    assert_eq!(
                        got, want,
                        "{check}: {:?} bound-1 build disagrees at space {space}, block {block}, \
                         rows {rows}",
                        spec.backend
                    );
                    compared += 1;
                }
                assert!(
                    bound1 >= 1,
                    "{check}: no bound-1 build offered at rows {rows}, group {group}; the \
                     portable one is unconditional"
                );
            }
        }
    }

    // The selection half. At bound 1 the adds-only build is what `Auto` runs;
    // one past its declaration a bound-1 build is not considered. A finite
    // alphabet lookup build remains admissible beyond bound 1 and also issues
    // no multiplies, so the assertion distinguishes its wider declaration.
    for &(rows, group) in corpus.sel_tiles {
        for &block in corpus.sel_blocks {
            let picked = choose_table(available(rows, group), Backend::Auto, 1, block)
                .expect("the reference sequence is always present");
            assert_eq!(
                picked.max_bound, 1,
                "{check}: bound 1 must select a bound-1 build at {rows}x{group} b={block}"
            );
            assert!(
                !picked.build_multiplies,
                "{check}: the build bound 1 selects must be the adds-only one at {rows}x{group} \
                 b={block}"
            );
            let past = choose_table(available(rows, group), Backend::Auto, 2, block)
                .expect("the reference sequence is always present");
            assert!(
                past.max_bound > 1,
                "{check}: bound 2 must not select a bound-1 build at {rows}x{group} \
                 b={block}"
            );
        }
    }
    compared
}

/// The activation tile in the layout `spec` declared: `k`-major in groups of
/// `k_group`, lane-major within a group --- [`packed_slot`] directly, because
/// the layout is the kernel contract and a check must not restate it.
fn pack_into<E: Copy>(flat: &[E], rows: usize, block: usize, k_group: usize, packed: &mut [E]) {
    for t in 0..block {
        for i in 0..rows {
            packed[packed_slot(t, i, rows, k_group)] = flat[t * rows + i];
        }
    }
}
