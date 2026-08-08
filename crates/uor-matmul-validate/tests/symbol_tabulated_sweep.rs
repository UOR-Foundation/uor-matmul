//! `CG-16` (open): the measured, value-blind block-one selector boundary for
//! the total binary32 q/Atlas table.
//!
//! The clock compares two public production calls made from identical caller
//! offers: forced [`Traversal::Tabulated`] and forced
//! [`Traversal::OutputMajor`]. Every timed batch is poisoned before its clock
//! and completely byte-checked after it. A separate counted call reconciles the
//! table's live q projection, local scheduler, demand dictionary, and placement
//! events against [`Census`].
//!
//! Selection may not read those post-execution events (`CS-10`). The fitted
//! model therefore receives only [`StructuralWork`], computed from declarations,
//! shape, plan, and caller-offer extents before either operand is inspected.
//! Four calibration keys deliberately have unlike value twins. Before fitting,
//! each key is reduced to the slowest table upper interval and the fastest dense
//! lower interval. A candidate can select the table only when the former remains
//! below the latter. All coefficients and timings are `open`: printed, never
//! asserted as mathematical facts.

use std::collections::{BTreeMap, BTreeSet};
use std::hint::black_box;
use std::time::{Duration, Instant};

use num_bigint::BigInt;
use num_traits::Zero;
use uor_matmul_codec::{Arena, Codec, CodedMatrix, Enumerable, TierId};
use uor_matmul_core::{
    as_alphabet_whole, Accumulator, Alphabet, Backend, Bound, MatView, MatViewMut, Shape,
    Traversal, Whole,
};
use uor_matmul_gemm::tabulated::{slab_codes, Plan, ROW_TILES};
use uor_matmul_gemm::{
    gemm_tabulated, gemm_tabulated_counted, suggested_tabulation, suggested_tabulation_index,
    suggested_tabulation_lanes, suggested_tabulation_panel, Census, Collapse, GemmOptions,
    LaneScale, Linear, Scratch, Tabulated, TabulatedTriple, Tabulation,
};

const SAMPLE_COUNT: usize = 9;
const SAMPLE_TARGET: Duration = Duration::from_millis(3);
const T95_DF8: f64 = 2.306_004_135_204_166;
const POISON: f32 = f32::from_bits(u32::MAX);
const TABLE_BASIS: usize = 14;
const DENSE_BASIS: usize = 6;

/// A normal binary32 significand whose balanced radix-256 section is
/// `[1, 1, -56, 1]`. Both factors therefore exercise the complete 4-by-4 q
/// lookup rectangle and all seven occupied product grades.
const FULL_Q_FRACTION: u32 = 0x0048_0101;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    Calibration,
    Holdout,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PanelOffer {
    Fixed,
    ActivationCache,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IndexOffer {
    Full,
    None,
    Short,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CodePattern {
    Mixed,
    Repeated,
    SharedFirst,
    AddressCollision,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ValueProfile {
    /// One grade, complete 4-by-4 q coordinate rectangle, no local scheduler.
    CompactFull,
    /// Seven grades per panel: local finite runs fill near Q and flush.
    WideLocal,
    /// A low base and distant finite grades force tagged singleton atoms.
    FiniteTag,
    /// Finite residues alternate with every boundary kind.
    SpecialAlternating,
    /// Every product is a non-finite singleton; this is the placement envelope.
    SpecialSingleton,
}

#[derive(Clone, Copy, Debug)]
struct Case {
    id: &'static str,
    role: Role,
    d: usize,
    m: usize,
    k: usize,
    n: usize,
    cols: Option<usize>,
    panel: PanelOffer,
    index: IndexOffer,
    codes: CodePattern,
    values: ValueProfile,
}

macro_rules! case {
    ($id:literal, $role:ident, $d:literal, $m:literal, $k:literal, $n:literal,
     $cols:expr, $panel:ident, $index:ident, $codes:ident, $values:ident) => {
        Case {
            id: $id,
            role: Role::$role,
            d: $d,
            m: $m,
            k: $k,
            n: $n,
            cols: $cols,
            panel: PanelOffer::$panel,
            index: IndexOffer::$index,
            codes: CodePattern::$codes,
            values: ValueProfile::$values,
        }
    };
}

/// Twenty-eight distinct structural calibration keys plus four unlike value
/// twins. The split is source-fixed before any clock is read.
const CALIBRATION: &[Case] = &[
    case!(
        "C01",
        Calibration,
        1,
        1,
        1,
        1,
        None,
        Fixed,
        Full,
        Repeated,
        CompactFull
    ),
    case!(
        "C02",
        Calibration,
        3,
        1,
        2,
        3,
        None,
        Fixed,
        Full,
        Mixed,
        FiniteTag
    ),
    case!(
        "C03",
        Calibration,
        5,
        2,
        15,
        5,
        None,
        Fixed,
        Full,
        Mixed,
        CompactFull
    ),
    case!(
        "C04",
        Calibration,
        16,
        15,
        16,
        7,
        None,
        Fixed,
        Full,
        Mixed,
        SpecialAlternating
    ),
    case!(
        "C05",
        Calibration,
        256,
        1,
        257,
        11,
        None,
        Fixed,
        Full,
        Mixed,
        WideLocal
    ),
    case!(
        "C06",
        Calibration,
        3,
        16,
        17,
        17,
        None,
        Fixed,
        Full,
        SharedFirst,
        CompactFull
    ),
    case!(
        "C07",
        Calibration,
        5,
        17,
        43,
        11,
        Some(3),
        Fixed,
        Full,
        Mixed,
        CompactFull
    ),
    case!(
        "C08",
        Calibration,
        5,
        17,
        43,
        11,
        Some(3),
        ActivationCache,
        Full,
        Mixed,
        SpecialAlternating
    ),
    case!(
        "C09",
        Calibration,
        16,
        17,
        1,
        17,
        Some(2),
        Fixed,
        Full,
        Repeated,
        CompactFull
    ),
    case!(
        "C10",
        Calibration,
        16,
        17,
        1,
        17,
        Some(2),
        ActivationCache,
        Full,
        Repeated,
        CompactFull
    ),
    case!(
        "C11",
        Calibration,
        256,
        2,
        64,
        65,
        Some(8),
        Fixed,
        Full,
        Mixed,
        CompactFull
    ),
    case!(
        "C12",
        Calibration,
        256,
        2,
        64,
        65,
        Some(8),
        ActivationCache,
        Full,
        Mixed,
        FiniteTag
    ),
    case!(
        "C13",
        Calibration,
        3,
        31,
        19,
        33,
        Some(4),
        Fixed,
        Full,
        Mixed,
        WideLocal
    ),
    case!(
        "C14",
        Calibration,
        3,
        31,
        19,
        33,
        Some(4),
        Fixed,
        None,
        Mixed,
        WideLocal
    ),
    case!(
        "C15",
        Calibration,
        5,
        8,
        33,
        64,
        Some(16),
        Fixed,
        Full,
        Mixed,
        CompactFull
    ),
    case!(
        "C16",
        Calibration,
        5,
        8,
        33,
        64,
        Some(16),
        Fixed,
        None,
        Mixed,
        CompactFull
    ),
    case!(
        "C17",
        Calibration,
        16,
        3,
        127,
        9,
        None,
        Fixed,
        Full,
        Mixed,
        WideLocal
    ),
    case!(
        "C18",
        Calibration,
        16,
        3,
        127,
        9,
        None,
        ActivationCache,
        Full,
        Mixed,
        SpecialAlternating
    ),
    case!(
        "C19",
        Calibration,
        256,
        1,
        4096,
        4,
        None,
        Fixed,
        Full,
        Mixed,
        CompactFull
    ),
    case!(
        "C20",
        Calibration,
        256,
        1,
        4096,
        4,
        Some(2),
        Fixed,
        Full,
        Mixed,
        WideLocal
    ),
    case!(
        "C21",
        Calibration,
        3,
        16,
        256,
        1,
        None,
        Fixed,
        Full,
        Mixed,
        FiniteTag
    ),
    case!(
        "C22",
        Calibration,
        5,
        33,
        2,
        7,
        Some(3),
        ActivationCache,
        Full,
        Repeated,
        CompactFull
    ),
    case!(
        "C23",
        Calibration,
        16,
        2,
        31,
        129,
        Some(64),
        Fixed,
        Full,
        SharedFirst,
        CompactFull
    ),
    case!(
        "C24",
        Calibration,
        16,
        2,
        31,
        129,
        Some(64),
        Fixed,
        None,
        SharedFirst,
        CompactFull
    ),
    case!(
        "C25",
        Calibration,
        256,
        9,
        8,
        257,
        Some(64),
        Fixed,
        Full,
        Mixed,
        CompactFull
    ),
    case!(
        "C26",
        Calibration,
        3,
        4,
        64,
        257,
        Some(7),
        Fixed,
        Full,
        Repeated,
        CompactFull
    ),
    case!(
        "C27",
        Calibration,
        5,
        17,
        21,
        2,
        Some(1),
        Fixed,
        Full,
        Mixed,
        SpecialAlternating
    ),
    case!(
        "C28",
        Calibration,
        16,
        16,
        65,
        3,
        Some(1),
        Fixed,
        Full,
        Mixed,
        WideLocal
    ),
    // Same structural keys, deliberately unlike q work.
    case!(
        "C29",
        Calibration,
        5,
        17,
        43,
        11,
        Some(3),
        Fixed,
        Full,
        Mixed,
        SpecialSingleton
    ),
    case!(
        "C30",
        Calibration,
        256,
        2,
        64,
        65,
        Some(8),
        Fixed,
        Full,
        Mixed,
        FiniteTag
    ),
    case!(
        "C31",
        Calibration,
        16,
        3,
        127,
        9,
        None,
        Fixed,
        Full,
        Mixed,
        SpecialSingleton
    ),
    case!(
        "C32",
        Calibration,
        256,
        1,
        4096,
        4,
        None,
        Fixed,
        Full,
        Mixed,
        SpecialAlternating
    ),
];

/// Twelve immutable unseen B=1 cases, including one same-key value twin. Two
/// independently declared non-power block-width controls complete the fourteen
/// holdout identities but remain outside the B=1 fit.
const HOLDOUT: &[Case] = &[
    case!(
        "H01",
        Holdout,
        3,
        2,
        43,
        13,
        Some(4),
        Fixed,
        Full,
        Mixed,
        CompactFull
    ),
    case!(
        "H02",
        Holdout,
        3,
        2,
        43,
        13,
        Some(4),
        Fixed,
        Full,
        Mixed,
        SpecialSingleton
    ),
    case!(
        "H03",
        Holdout,
        5,
        17,
        64,
        19,
        Some(5),
        Fixed,
        Full,
        Repeated,
        CompactFull
    ),
    case!(
        "H04",
        Holdout,
        5,
        17,
        64,
        19,
        Some(5),
        ActivationCache,
        Full,
        Repeated,
        CompactFull
    ),
    case!(
        "H05",
        Holdout,
        16,
        1,
        15,
        33,
        Some(8),
        Fixed,
        Full,
        SharedFirst,
        FiniteTag
    ),
    case!(
        "H06",
        Holdout,
        16,
        1,
        15,
        34,
        Some(8),
        Fixed,
        Short,
        AddressCollision,
        FiniteTag
    ),
    case!(
        "H07",
        Holdout,
        256,
        1,
        1024,
        16,
        None,
        Fixed,
        Full,
        Mixed,
        CompactFull
    ),
    case!(
        "H08",
        Holdout,
        256,
        1,
        1024,
        1,
        None,
        Fixed,
        Full,
        Mixed,
        SpecialSingleton
    ),
    case!(
        "H09",
        Holdout,
        3,
        33,
        31,
        65,
        Some(7),
        ActivationCache,
        Full,
        Mixed,
        WideLocal
    ),
    case!(
        "H10",
        Holdout,
        5,
        15,
        22,
        7,
        None,
        Fixed,
        None,
        Repeated,
        CompactFull
    ),
    case!(
        "H11",
        Holdout,
        16,
        18,
        257,
        5,
        Some(2),
        Fixed,
        Full,
        Mixed,
        SpecialAlternating
    ),
    case!(
        "H12",
        Holdout,
        256,
        9,
        273,
        129,
        Some(32),
        Fixed,
        Full,
        SharedFirst,
        WideLocal
    ),
];

#[derive(Clone, Copy, Debug)]
struct ModelFacts {
    hash_prefix: usize,
    compact_ceiling: u128,
    tag_base: u64,
}

impl ModelFacts {
    fn load() -> Self {
        let model = uor_matmul_model::Model::load_from_repo_root().expect("the model loads");
        model
            .check()
            .expect("the generated q and hash constants check");
        Self {
            hash_prefix: model.constants.column_hash.prefix,
            compact_ceiling: u128::from(model.widths.f32_q_carrier.compact_ceiling),
            tag_base: model.widths.f32_q_carrier.tag_base,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Offer {
    panel: usize,
    exact: usize,
    lanes: usize,
    index: usize,
    plan: Plan,
    cache_rows: usize,
}

struct Workspace {
    panel: Vec<Alphabet<f32, Whole<f32>>>,
    exact: Vec<<f32 as uor_matmul_core::Element>::Acc>,
    lanes: Vec<i64>,
    index: Vec<usize>,
}

impl Workspace {
    fn new(offer: Offer) -> Self {
        Self {
            panel: vec![Alphabet::<f32, Whole<f32>>::ZERO; offer.panel],
            exact: vec![<<f32 as uor_matmul_core::Element>::Acc as Accumulator>::ZERO; offer.exact],
            lanes: vec![0; offer.lanes],
            index: vec![0; offer.index],
        }
    }
}

struct Fixture<const D: usize> {
    shape: Shape,
    table: [f32; D],
    codes: Vec<u8>,
    a: Vec<f32>,
}

impl<const D: usize> Fixture<D> {
    fn new(case: Case) -> Self {
        assert_eq!(case.d, D);
        assert!(D > 0 && D <= usize::from(u8::MAX) + 1);
        let shape = Shape {
            m: case.m,
            k: case.k,
            n: case.n,
        };
        let table = std::array::from_fn(|at| profile_value(case.values, false, at, D));
        let a = (0..shape.m * shape.k)
            .map(|at| profile_value(case.values, true, at, shape.m * shape.k))
            .collect();
        let mut codes = match case.codes {
            CodePattern::Mixed => (0..shape.n * shape.k)
                .map(|at| ((at * 3 + at / shape.k + 1) % D) as u8)
                .collect(),
            CodePattern::Repeated => vec![0; shape.n * shape.k],
            CodePattern::SharedFirst => {
                assert!(shape.k >= 2);
                (0..shape.n * shape.k)
                    .map(|at| {
                        let (column, p) = (at / shape.k, at % shape.k);
                        if p == 0 {
                            (3 % D) as u8
                        } else if p == 1 {
                            (column % D) as u8
                        } else {
                            ((at * 3 + column + 1) % D) as u8
                        }
                    })
                    .collect()
            }
            CodePattern::AddressCollision => (0..shape.n * shape.k)
                .map(|at| {
                    let base = (at * 5 + at / shape.k) % D;
                    let separated = base + D / 2;
                    (if at.is_multiple_of(2) {
                        base
                    } else {
                        separated % D
                    }) as u8
                })
                .collect(),
        };
        if D > 1 && !codes.is_empty() && case.codes != CodePattern::Repeated {
            codes[0] = 0;
            let last = codes.len() - 1;
            codes[last] = (D - 1) as u8;
        }
        Self {
            shape,
            table,
            codes,
            a,
        }
    }

    fn table_alphabet(&self) -> &[Alphabet<f32, Whole<f32>>; D] {
        as_alphabet_whole(&self.table)
            .try_into()
            .expect("the array retains its code-space extent")
    }

    fn weights(&self) -> CodedMatrix<'_, f32, Whole<f32>, Arena<'_, f32, D, u8>> {
        CodedMatrix::new(
            Arena::<f32, D, u8>::new(self.table_alphabet()),
            self.shape.n,
            self.shape.k,
            &self.codes,
        )
        .expect("the fixed-width stream describes n by k")
    }
}

fn sign_place() -> u32 {
    let mut value = 1u32;
    for _ in 0..u32::BITS - 1 {
        value = value.wrapping_add(value);
    }
    value
}

fn exponent_place() -> u32 {
    let mut value = 1u32;
    for _ in 0..23 {
        value = value.wrapping_add(value);
    }
    value
}

fn full_q(biased_exponent: u32, negative: bool) -> f32 {
    let sign = if negative { sign_place() } else { 0 };
    f32::from_bits(sign + biased_exponent * exponent_place() + FULL_Q_FRACTION)
}

fn profile_value(profile: ValueProfile, activation: bool, at: usize, len: usize) -> f32 {
    assert_ne!(len, 0, "a profile is defined on a nonempty measured extent");
    let last = len - 1;
    match profile {
        ValueProfile::CompactFull => full_q(119, (at + usize::from(activation)).is_multiple_of(3)),
        ValueProfile::WideLocal => {
            let high = at != 0 && (at == last || !at.is_multiple_of(5));
            full_q(
                if high { 126 } else { 119 },
                (at + usize::from(activation)).is_multiple_of(2),
            )
        }
        ValueProfile::FiniteTag => {
            let high = at != 0 && (at == last || !at.is_multiple_of(7));
            full_q(
                if high { 148 } else { 119 },
                (at + usize::from(activation)).is_multiple_of(2),
            )
        }
        ValueProfile::SpecialAlternating => match (at + usize::from(activation)) % 6 {
            0 => full_q(119, false),
            1 => f32::INFINITY,
            2 => full_q(126, true),
            3 => f32::NEG_INFINITY,
            4 => f32::NAN,
            _ => f32::from_bits(1),
        },
        ValueProfile::SpecialSingleton => f32::NAN,
    }
}

fn resolve_offer<const D: usize>(case: Case, shape: Shape) -> Offer {
    let fixed_panel = suggested_tabulation_panel(D, 1);
    let panel = match case.panel {
        PanelOffer::Fixed => fixed_panel,
        PanelOffer::ActivationCache => fixed_panel
            .checked_add(shape.m.checked_mul(shape.k).expect("A is addressable"))
            .expect("the cache offer is addressable"),
    };
    let full_exact = suggested_tabulation::<f32, Whole<f32>>(shape, D, 1);
    let lanes = suggested_tabulation_lanes::<f32, Whole<f32>>(shape, D, 1);
    assert!(panel > 0 && full_exact > 0 && lanes > 0);
    let lane_words = std::mem::size_of::<i64>() * lanes / <f32 as Tabulated>::LANE_BYTES;
    let full = Plan::choose(
        D,
        shape,
        <f32 as Tabulated>::LANE_BYTES,
        full_exact,
        lane_words,
        1,
        <f32 as Tabulated>::probe_capacity::<<f32 as Tabulated>::Lane>(
            <Whole<f32> as Bound>::VALUE,
        ),
    )
    .expect("the full public offers admit a q table");
    let exact = case.cols.map_or(full_exact, |cols| {
        assert_ne!(cols, 0, "an explicit measured column offer is nonempty");
        full.rows
            .checked_mul(cols)
            .expect("the partial exact offer is addressable")
            .min(full_exact)
    });
    let plan = Plan::choose(
        D,
        shape,
        <f32 as Tabulated>::LANE_BYTES,
        exact,
        lane_words,
        1,
        <f32 as Tabulated>::probe_capacity::<<f32 as Tabulated>::Lane>(
            <Whole<f32> as Bound>::VALUE,
        ),
    )
    .expect("the chosen public offers admit a q table");
    if let Some(cols) = case.cols {
        assert_eq!(plan.cols, shape.n.min(cols));
    }
    let full_index = suggested_tabulation_index(shape);
    let index = match case.index {
        IndexOffer::Full => full_index,
        IndexOffer::None => 0,
        IndexOffer::Short => full_index.saturating_sub(1),
    };
    let cache_rows = if shape.n > plan.cols {
        ((panel - fixed_panel) / shape.k).min(shape.m)
    } else {
        0
    };
    Offer {
        panel,
        exact,
        lanes,
        index,
        plan,
        cache_rows,
    }
}

fn row_tiles(m: usize, cap: usize) -> Vec<usize> {
    let mut remaining = m;
    let mut tiles = Vec::new();
    while remaining != 0 {
        let rows = ROW_TILES
            .into_iter()
            .find(|&rows| rows <= cap && rows <= remaining)
            .expect("a nonempty row tile exists");
        tiles.push(rows);
        remaining -= rows;
    }
    tiles
}

fn column_widths(n: usize, cols: usize) -> Vec<usize> {
    (0..n).step_by(cols).map(|at| cols.min(n - at)).collect()
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct StructuralWork {
    d: usize,
    shape: ShapeKey,
    plan: PlanKey,
    panel_offer: usize,
    exact_offer: usize,
    lane_offer: usize,
    index_offer: usize,
    table: [u128; TABLE_BASIS],
    dense: [u128; DENSE_BASIS],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ShapeKey {
    m: usize,
    k: usize,
    n: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct PlanKey {
    rows: usize,
    cols: usize,
    depth: usize,
}

impl StructuralWork {
    fn of<const D: usize>(fixture: &Fixture<D>, offer: Offer, facts: ModelFacts) -> Self {
        let Shape { m, k, n } = fixture.shape;
        let Plan { rows, cols, depth } = offer.plan;
        let tiles = row_tiles(m, rows);
        let widths = column_widths(n, cols);
        let column_blocks = widths.len();
        let slab = slab_codes(D);
        let dictionary = n
            .checked_mul(2)
            .and_then(usize::checked_next_power_of_two)
            .expect("the measured dictionary is addressable");
        let dictionary_want = n
            .checked_add(dictionary.checked_mul(2).expect("dictionary words fit"))
            .expect("the index offer is addressable");
        let dictionary = if offer.index >= dictionary_want {
            dictionary
        } else {
            0
        };
        // The first clear belongs to the column dictionary. A pointwise book
        // wider than one reuses it as EntrySet and clears the same word kind a
        // second time. They are inseparable homogeneous writes, so one basis
        // column is the exact quotient rather than two collinear columns.
        let workspace_clear_words = dictionary
            .checked_mul(1 + usize::from(D > 1))
            .expect("workspace clears fit");
        let stored_codes = n.checked_mul(k).expect("the code stream is addressable");
        let entry_attempt_ceiling = if dictionary != 0 && D > 1 {
            stored_codes
                .checked_add(
                    tiles
                        .len()
                        .checked_mul(stored_codes)
                        .expect("slot attempts fit"),
                )
                .expect("entry attempts fit")
        } else {
            0
        };
        let activation_observations = m.checked_mul(k).expect("A is addressable");
        let book_sites = D.min(stored_codes);
        let activation_projections = offer
            .cache_rows
            .checked_mul(k)
            .and_then(|cached| {
                (m - offer.cache_rows)
                    .checked_mul(k)
                    .and_then(|uncached| uncached.checked_mul(column_blocks))
                    .and_then(|uncached| cached.checked_add(uncached))
            })
            .expect("q projections are addressable");
        let row_source_sites = m
            .checked_mul(n)
            .and_then(|cells| cells.checked_mul(k))
            .expect("row-source sites fit");
        let tile_source_sites = tiles
            .len()
            .checked_mul(n)
            .and_then(|cells| cells.checked_mul(k))
            .expect("tile-source sites fit");
        let demanded_per_source: usize = widths.iter().map(|&width| D.min(width)).sum();
        let demand_build_ceiling = m
            .checked_mul(k)
            .and_then(|cells| cells.checked_mul(demanded_per_source))
            .expect("demand builds fit");
        let resident_lane_words = rows
            .checked_mul(cols)
            .and_then(|columns| {
                slab.checked_mul(rows)
                    .and_then(|one| one.checked_mul(depth))
                    .and_then(|stack| columns.checked_add(stack))
            })
            .expect("the lane plan is addressable");
        let mut geometries = BTreeSet::new();
        let padding_words: usize = tiles
            .iter()
            .copied()
            .filter(|rows| geometries.insert(*rows))
            .map(|rows| (slab - D) * rows * depth)
            .sum();
        let tile_block_presentations = tiles
            .len()
            .checked_mul(column_blocks)
            .expect("tile-block presentations fit");

        // Canonical pre-admission quotient basis. The post-run decomposition is
        // intentionally not recoverable from this value-blind object.
        let table = [
            1,
            n as u128,
            n.checked_mul(k.min(facts.hash_prefix))
                .expect("hash visits fit") as u128,
            workspace_clear_words as u128,
            entry_attempt_ceiling as u128,
            activation_observations as u128,
            book_sites as u128,
            activation_projections as u128,
            row_source_sites as u128,
            tile_source_sites as u128,
            demand_build_ceiling as u128,
            resident_lane_words as u128,
            padding_words as u128,
            tile_block_presentations as u128,
        ];

        let page = offer.panel.min(k);
        assert!(page > 0);
        let (dense_decodes, dense_calls, dense_joins) = if page >= k {
            (n * k, n * m.div_ceil(ROW_TILES[0]), 0)
        } else {
            let pages = k.div_ceil(page);
            (m * n * k, m * n * pages, m * n * (pages - 1))
        };
        let dense = [
            1,
            (m * n) as u128,
            dense_decodes as u128,
            row_source_sites as u128,
            dense_calls as u128,
            dense_joins as u128,
        ];
        Self {
            d: D,
            shape: ShapeKey { m, k, n },
            plan: PlanKey { rows, cols, depth },
            panel_offer: offer.panel,
            exact_offer: offer.exact,
            lane_offer: offer.lanes,
            index_offer: offer.index,
            table,
            dense,
        }
    }

    fn table_basis(&self) -> [u128; TABLE_BASIS] {
        self.table
    }

    fn dense_basis(&self) -> [u128; DENSE_BASIS] {
        self.dense
    }
}

/// Fraction-free elimination. A full rank modulo a machine word could prove
/// nonzero rank but could reject a valid design whose determinant happens to
/// share that modulus. Bareiss over `BigInt` has neither outcome nor a numeric
/// tolerance.
fn exact_rank<const C: usize>(matrix: &[[u128; C]]) -> usize {
    if matrix.is_empty() {
        return 0;
    }
    let mut a: Vec<Vec<BigInt>> = matrix
        .iter()
        .map(|row| row.iter().copied().map(BigInt::from).collect())
        .collect();
    let mut rank = 0usize;
    let mut denominator = BigInt::from(1u8);
    for column in 0..C {
        let Some(pivot_row) = (rank..a.len()).find(|&row| !a[row][column].is_zero()) else {
            continue;
        };
        a.swap(rank, pivot_row);
        let pivot = a[rank][column].clone();
        for row in rank + 1..a.len() {
            for col in column + 1..C {
                let numerator = &a[row][col] * &pivot - &a[row][column] * &a[rank][col];
                a[row][col] = numerator / &denominator;
            }
            a[row][column] = BigInt::zero();
        }
        denominator = pivot;
        rank += 1;
        if rank == a.len().min(C) {
            break;
        }
    }
    rank
}

fn assert_design(calibration: &[(Case, StructuralWork)], holdout: &[(Case, StructuralWork)]) {
    let unique: BTreeMap<_, _> = calibration
        .iter()
        .map(|(_, work)| (work.clone(), work.clone()))
        .collect();
    assert_eq!(
        unique.len(),
        28,
        "four and only four CAL rows are value twins"
    );
    let table: Vec<_> = unique.values().map(StructuralWork::table_basis).collect();
    let dense: Vec<_> = unique.values().map(StructuralWork::dense_basis).collect();
    assert_eq!(
        exact_rank(&table),
        TABLE_BASIS,
        "the table design is full rank"
    );
    assert_eq!(
        exact_rank(&dense),
        DENSE_BASIS,
        "the dense design is full rank"
    );

    // Every named basis coordinate is load-bearing. Dropping any one must lower
    // the exact rank, not merely leave another arbitrary spanning subset.
    for dropped in 0..TABLE_BASIS {
        let planted: Vec<[u128; TABLE_BASIS]> = table
            .iter()
            .map(|row| {
                let mut row = *row;
                row[dropped] = 0;
                row
            })
            .collect();
        assert_eq!(
            exact_rank(&planted),
            TABLE_BASIS - 1,
            "table basis column {dropped} is essential"
        );
    }
    for dropped in 0..DENSE_BASIS {
        let planted: Vec<[u128; DENSE_BASIS]> = dense
            .iter()
            .map(|row| {
                let mut row = *row;
                row[dropped] = 0;
                row
            })
            .collect();
        assert_eq!(
            exact_rank(&planted),
            DENSE_BASIS - 1,
            "dense basis column {dropped} is essential"
        );
    }
    let calibration_keys: BTreeSet<_> = unique.keys().cloned().collect();
    let holdout_keys: BTreeSet<_> = holdout.iter().map(|(_, work)| work.clone()).collect();
    assert_eq!(
        holdout_keys.len(),
        11,
        "H01/H02 are the one value-blind holdout twin"
    );
    assert!(
        calibration_keys.is_disjoint(&holdout_keys),
        "no holdout structural key may enter calibration"
    );

    // Mutation plant: aliasing one coordinate to another must turn the gate red.
    let mut aliased = table.clone();
    for row in &mut aliased {
        row[TABLE_BASIS - 1] = row[TABLE_BASIS - 2];
    }
    assert!(exact_rank(&aliased) < TABLE_BASIS);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Route {
    Table,
    Dense,
}

impl Route {
    const fn traversal(self) -> Traversal {
        match self {
            Self::Table => Traversal::Tabulated,
            Self::Dense => Traversal::OutputMajor,
        }
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Table => "table",
            Self::Dense => "dense",
        }
    }
}

fn verify_bits(output: &[f32], expected: &[u32]) {
    assert_eq!(output.len(), expected.len());
    for (at, (&got, &want)) in output.iter().zip(expected).enumerate() {
        assert_eq!(got.to_bits(), want, "complete output differs at cell {at}");
    }
}

fn counted_once<const D: usize>(
    fixture: &Fixture<D>,
    traversal: Traversal,
    workspace: &mut Workspace,
    output: &mut [f32],
) -> Census {
    output.fill(POISON);
    let a = MatView::row_major(
        as_alphabet_whole(&fixture.a),
        fixture.shape.m,
        fixture.shape.k,
    )
    .expect("A has the declared shape");
    let c = MatViewMut::row_major(output, fixture.shape.m, fixture.shape.n)
        .expect("C has the declared shape");
    let mut triple = TabulatedTriple::new(a, fixture.weights(), c).expect("the product exists");
    let mut scratch = Scratch::with_accumulators(&mut workspace.panel, &mut workspace.exact);
    let mut tabulation = Tabulation::with_index(&mut workspace.lanes, &mut workspace.index);
    let mut collapse = Collapse::none();
    let mut census = Census::default();
    gemm_tabulated_counted(
        &mut triple,
        &Linear::OVERWRITE,
        GemmOptions {
            traversal,
            backend: Backend::Auto,
            ..Default::default()
        },
        &mut scratch,
        &mut tabulation,
        &mut collapse,
        &mut census,
    );
    black_box(&*output);
    census
}

/// One poison/run/check batch. Operand views, offers and storage are all
/// constructed before the first clock read; only repeated public production
/// calls occupy the measured interval.
fn timed_batch<const D: usize>(
    fixture: &Fixture<D>,
    route: Route,
    repetitions: usize,
    workspace: &mut Workspace,
    output: &mut [f32],
    expected: &[u32],
) -> Duration {
    output.fill(POISON);
    let elapsed = {
        let a = MatView::row_major(
            as_alphabet_whole(&fixture.a),
            fixture.shape.m,
            fixture.shape.k,
        )
        .expect("A has the declared shape");
        let c = MatViewMut::row_major(output, fixture.shape.m, fixture.shape.n)
            .expect("C has the declared shape");
        let mut triple = TabulatedTriple::new(a, fixture.weights(), c).expect("the product exists");
        let mut scratch = Scratch::with_accumulators(&mut workspace.panel, &mut workspace.exact);
        let mut tabulation = Tabulation::with_index(&mut workspace.lanes, &mut workspace.index);
        let mut collapse = Collapse::none();
        let options = GemmOptions {
            traversal: route.traversal(),
            backend: Backend::Auto,
            ..Default::default()
        };

        let start = Instant::now();
        for _ in 0..repetitions {
            gemm_tabulated(
                &mut triple,
                &Linear::OVERWRITE,
                options,
                &mut scratch,
                &mut tabulation,
                &mut collapse,
            );
        }
        start.elapsed()
    };
    verify_bits(output, expected); // CG16_FULL_BYTE_GUARD
    elapsed
}

#[derive(Clone, Copy, Debug)]
struct Estimate {
    mean: f64,
    half_width: f64,
}

impl Estimate {
    fn of(values: &[f64; SAMPLE_COUNT]) -> Self {
        let n = SAMPLE_COUNT as f64;
        let mean = values.iter().sum::<f64>() / n;
        let variance = values
            .iter()
            .map(|value| {
                let residual = value - mean;
                residual * residual
            })
            .sum::<f64>()
            / (n - 1.0);
        Self {
            mean,
            half_width: T95_DF8 * (variance / n).sqrt(),
        }
    }

    fn lower(self) -> f64 {
        (self.mean - self.half_width).max(0.0)
    }

    fn upper(self) -> f64 {
        self.mean + self.half_width
    }
}

#[derive(Clone, Copy, Debug)]
struct PairTiming {
    table: Estimate,
    dense: Estimate,
    ratio: Estimate,
    table_samples: [f64; SAMPLE_COUNT],
    dense_samples: [f64; SAMPLE_COUNT],
    table_batch: usize,
    dense_batch: usize,
}

fn batch_repetitions(pilot: Duration) -> usize {
    let count = SAMPLE_TARGET.as_nanos().div_ceil(pilot.as_nanos().max(1));
    usize::try_from(count).expect("the three-millisecond target fits every supported usize")
}

fn print_pair_samples(split: &str, id: &str, key: &str, timing: &PairTiming) {
    for round in 0..SAMPLE_COUNT {
        println!(
            "SAMPLE,{split},{id},{key},{round},{},{:.12e}",
            Route::Table.as_str(),
            timing.table_samples[round],
        );
        println!(
            "SAMPLE,{split},{id},{key},{round},{},{:.12e}",
            Route::Dense.as_str(),
            timing.dense_samples[round],
        );
    }
}

fn calibrated_batch<const D: usize>(
    fixture: &Fixture<D>,
    route: Route,
    workspace: &mut Workspace,
    output: &mut [f32],
    expected: &[u32],
) -> usize {
    let pilot = timed_batch(fixture, route, 1, workspace, output, expected);
    batch_repetitions(pilot)
}

#[allow(clippy::too_many_arguments)]
fn measure_pair<const D: usize>(
    fixture: &Fixture<D>,
    table_workspace: &mut Workspace,
    dense_workspace: &mut Workspace,
    table_output: &mut [f32],
    dense_output: &mut [f32],
    expected: &[u32],
) -> PairTiming {
    timed_batch(
        fixture,
        Route::Table,
        1,
        table_workspace,
        table_output,
        expected,
    );
    timed_batch(
        fixture,
        Route::Dense,
        1,
        dense_workspace,
        dense_output,
        expected,
    );
    let table_batch = calibrated_batch(
        fixture,
        Route::Table,
        table_workspace,
        table_output,
        expected,
    );
    let dense_batch = calibrated_batch(
        fixture,
        Route::Dense,
        dense_workspace,
        dense_output,
        expected,
    );
    let mut table = [0.0; SAMPLE_COUNT];
    let mut dense = [0.0; SAMPLE_COUNT];
    for round in 0..SAMPLE_COUNT {
        let table_first = round.is_multiple_of(2);
        let measure_table = |workspace: &mut Workspace, output: &mut [f32]| {
            timed_batch(
                fixture,
                Route::Table,
                table_batch,
                workspace,
                output,
                expected,
            )
            .as_secs_f64()
                / table_batch as f64
        };
        let measure_dense = |workspace: &mut Workspace, output: &mut [f32]| {
            timed_batch(
                fixture,
                Route::Dense,
                dense_batch,
                workspace,
                output,
                expected,
            )
            .as_secs_f64()
                / dense_batch as f64
        };
        if table_first {
            table[round] = measure_table(table_workspace, table_output);
            dense[round] = measure_dense(dense_workspace, dense_output);
        } else {
            dense[round] = measure_dense(dense_workspace, dense_output);
            table[round] = measure_table(table_workspace, table_output);
        }
    }
    let ratio = std::array::from_fn(|at| table[at] / dense[at]);
    PairTiming {
        table: Estimate::of(&table),
        dense: Estimate::of(&dense),
        ratio: Estimate::of(&ratio),
        table_samples: table,
        dense_samples: dense,
        table_batch,
        dense_batch,
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ColumnEvents {
    hash_coordinates: u64,
    hash_adds: u64,
    probe_visits: u64,
    equality_coordinates: u64,
    dictionary_clear_words: u64,
    representative_clear_words: u64,
    distinct_columns: u64,
    repeated_columns: u64,
    collapsed_blocks: u64,
    expanded_cells: u64,
}

#[derive(Clone, Debug)]
struct ColumnReplay {
    first: Vec<usize>,
    identity: Vec<usize>,
    dictionary_extent: usize,
    entry_occupied: usize,
    events: ColumnEvents,
}

impl ColumnReplay {
    fn collapsed(&self, col0: usize, cols: usize, plan_cols: usize) -> Option<&[usize]> {
        if self.dictionary_extent == 0 || self.identity[col0 / plan_cols] != 0 {
            None
        } else {
            Some(&self.first[col0..col0 + cols])
        }
    }
}

fn measured_column_hash(run: &[u8], modulus: usize, prefix: usize) -> (usize, usize) {
    assert_ne!(modulus, 0);
    let mut hash = run.len() as u128;
    let measured = run.len().min(prefix);
    for &code in &run[..measured] {
        let doubled = hash + hash;
        hash = doubled + hash + usize::from(code) as u128;
    }
    ((hash % modulus as u128) as usize, measured)
}

fn replay_columns<const D: usize>(
    fixture: &Fixture<D>,
    offer: Offer,
    facts: ModelFacts,
) -> ColumnReplay {
    let (n, k) = (fixture.shape.n, fixture.shape.k);
    let blocks = n.div_ceil(offer.plan.cols);
    let Some(table) = n.checked_mul(2).and_then(usize::checked_next_power_of_two) else {
        unreachable!("the measured shapes have an addressable column dictionary")
    };
    let want = n + table * 2;
    if offer.index < want {
        return ColumnReplay {
            first: Vec::new(),
            identity: vec![1; blocks],
            dictionary_extent: 0,
            entry_occupied: 0,
            events: ColumnEvents::default(),
        };
    }

    let mut position = vec![0usize; n];
    let mut slot = vec![0usize; table];
    let mut key = vec![0usize; table];
    let mut events = ColumnEvents {
        dictionary_clear_words: table as u64,
        ..ColumnEvents::default()
    };
    let mut distinct = 0usize;
    for (j, position) in position.iter_mut().enumerate() {
        let run = &fixture.codes[j * k..(j + 1) * k];
        let (hash, measured) = measured_column_hash(run, table, facts.hash_prefix);
        events.hash_coordinates += measured as u64;
        events.hash_adds += (measured * 2) as u64;
        let mut probe = hash;
        loop {
            events.probe_visits += 1;
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
                    if key[probe] == hash {
                        let other = &fixture.codes[seen * k..(seen + 1) * k];
                        let mut equal = true;
                        for (&left, &right) in run.iter().zip(other) {
                            events.equality_coordinates += 1;
                            if usize::from(left) % D != usize::from(right) % D {
                                equal = false;
                                break;
                            }
                        }
                        if equal {
                            *position = seen;
                            break;
                        }
                    }
                    probe += 1;
                    if probe == table {
                        probe = 0;
                    }
                }
            }
        }
    }
    events.distinct_columns = distinct as u64;
    events.repeated_columns = (n - distinct) as u64;

    if distinct == n {
        return ColumnReplay {
            first: position,
            identity: vec![1; blocks],
            dictionary_extent: table,
            // `column_workspace` starts at index[0] when there is no map.
            entry_occupied: n + table,
            events,
        };
    }

    let mut identity = vec![1usize; blocks];
    for start in (0..n).step_by(offer.plan.cols) {
        let end = (start + offer.plan.cols).min(n);
        for &representative in &position[start..end] {
            slot[representative] = 0;
            events.representative_clear_words += 1;
        }
        for j in start..end {
            let representative = position[j];
            let prior = slot[representative];
            position[j] = if prior > start {
                prior - 1 - start
            } else {
                slot[representative] = j + 1;
                j - start
            };
            if position[j] != j - start {
                identity[j / offer.plan.cols] = 0;
                events.expanded_cells += fixture.shape.m as u64;
            }
        }
    }
    events.collapsed_blocks = identity.iter().filter(|&&word| word == 0).count() as u64;
    ColumnReplay {
        first: position,
        identity,
        dictionary_extent: table,
        // `keys[..blocks]` holds identity; the tail is EntrySet::occupied.
        entry_occupied: table - blocks,
        events,
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct EntryEvents {
    attempts: u64,
    probe_visits: u64,
    inserted: u64,
    present: u64,
    full: u64,
    cleared_slots: u64,
    transferred_slots: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReplayInsert {
    New,
    Present,
    Full,
}

struct EntryReplay {
    seen: Vec<usize>,
    occupied: Vec<usize>,
    used: usize,
    events: EntryEvents,
}

impl EntryReplay {
    fn new(extent: usize, occupied: usize) -> Self {
        Self {
            seen: vec![0; extent],
            occupied: vec![0; occupied],
            used: 0,
            events: EntryEvents::default(),
        }
    }

    fn insert(&mut self, index: usize) -> ReplayInsert {
        self.events.attempts += 1;
        let Some(key) = index.checked_add(1) else {
            self.events.full += 1;
            return ReplayInsert::Full;
        };
        if self.seen.is_empty() {
            self.events.full += 1;
            return ReplayInsert::Full;
        }
        let extent = self.seen.len();
        let mut probe = if index < extent {
            index
        } else {
            index % extent
        };
        for _ in 0..extent {
            self.events.probe_visits += 1;
            match self.seen[probe] {
                0 if self.used < self.occupied.len() => {
                    self.seen[probe] = key;
                    self.occupied[self.used] = probe;
                    self.used += 1;
                    self.events.inserted += 1;
                    return ReplayInsert::New;
                }
                0 => {
                    self.events.full += 1;
                    return ReplayInsert::Full;
                }
                present if present == key => {
                    self.events.present += 1;
                    return ReplayInsert::Present;
                }
                _ => {
                    probe += 1;
                    if probe == extent {
                        probe = 0;
                    }
                }
            }
        }
        self.events.full += 1;
        ReplayInsert::Full
    }

    fn indices(&self) -> Vec<usize> {
        self.occupied[..self.used]
            .iter()
            .map(|&probe| self.seen[probe] - 1)
            .collect()
    }

    fn clear(&mut self) {
        for &probe in &self.occupied[..self.used] {
            self.seen[probe] = 0;
            self.events.cleared_slots += 1;
        }
        self.used = 0;
    }

    fn collect(&mut self, codes: &[u8], d: usize) -> Option<Vec<usize>> {
        for &code in codes {
            if self.insert(usize::from(code) % d) == ReplayInsert::Full {
                self.clear();
                return None;
            }
        }
        let indices = self.indices();
        for &probe in &self.occupied[..self.used] {
            self.seen[probe] = 0;
            self.events.transferred_slots += 1;
        }
        // Production retains `used` until `release_collected`, but no operation
        // observes the set between decode and release. This is that release.
        self.used = 0;
        Some(indices)
    }
}

fn atlas_coordinates(value: f32) -> Vec<i8> {
    if !value.is_finite() || value == 0.0 {
        return Vec::new();
    }
    let sign = sign_place();
    let exponent = exponent_place();
    let unsigned = value.to_bits() % sign;
    let q = unsigned / exponent;
    let fraction = unsigned % exponent;
    let mut magnitude = u64::from(if q == 0 {
        fraction
    } else {
        exponent + fraction
    });
    while magnitude != 0 {
        let quotient = magnitude / 2;
        if quotient + quotient != magnitude {
            break;
        }
        magnitude = quotient;
    }
    let mut value = i128::from(magnitude);
    let mut coordinates = Vec::new();
    let radix = i128::from(u8::MAX) + 1;
    while value != 0 {
        let residue = value.rem_euclid(radix);
        let digit = if residue > i128::from(i8::MAX) {
            residue - radix
        } else {
            residue
        };
        coordinates.push(digit as i8);
        value = (value - digit) / radix;
    }
    coordinates
}

fn q_lookup_pairs(left: f32, right: f32) -> (usize, bool) {
    let left = atlas_coordinates(left);
    let right = atlas_coordinates(right);
    let pairs = left.iter().filter(|&&coordinate| coordinate != 0).count()
        * right.iter().filter(|&&coordinate| coordinate != 0).count();
    (pairs, left.len() == 4 && right.len() == 4 && pairs == 16)
}

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

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct DiagnosticWork {
    column: ColumnEvents,
    call_entries: EntryEvents,
    build_entries: EntryEvents,
    call_scale: Option<LaneScale>,
    call_scale_observations: u64,
    call_scale_q_macs: u64,
    book_codec_decodes: u64,
    book_q_projections: u64,
    activation_q_projections: u64,
    scalar_envelope_calls: u64,
    scalar_envelope_replays: u64,
    scalar_certificate_observations: u64,
    scalar_certificate_q_macs: u64,
    safe_source_aggregates: u64,
    fractured_source_aggregates: u64,
    finite_tag_singletons: u64,
    special_tag_singletons: u64,
    capacity_flushes: u64,
    singleton_placements: u64,
    tail_placements: u64,
    placed_cells: u64,
    sparse_build_entries: u64,
    full_build_entries: u64,
    duplicate_build_entries: u64,
    build_contractions: u64,
    q_lookup_pairs: u64,
    full_q_rectangles: u64,
    gather_adds: u64,
    padding_words: u64,
    dense_decodes: u64,
    dense_calls: u64,
    dense_joins: u64,
}

fn call_lane_scale<const D: usize>(
    fixture: &Fixture<D>,
    addressed: Option<&[usize]>,
) -> (LaneScale, Census) {
    let addressed_codes;
    let (n, k, codes) = if let Some(addressed) = addressed {
        addressed_codes = addressed
            .iter()
            .map(|&index| u8::try_from(index).expect("Arena coordinates fit u8"))
            .collect::<Vec<_>>();
        (addressed_codes.len(), 1, addressed_codes.as_slice())
    } else {
        (fixture.shape.n, fixture.shape.k, fixture.codes.as_slice())
    };
    let weights = CodedMatrix::new(
        Arena::<f32, D, u8>::new(fixture.table_alphabet()),
        n,
        k,
        codes,
    )
    .expect("the scale presentation is conformant");
    let a = MatView::row_major(
        as_alphabet_whole(&fixture.a),
        fixture.shape.m,
        fixture.shape.k,
    )
    .expect("A has its declared shape");
    let mut census = Census::default();
    let scale = <f32 as Tabulated>::lane_scale(&a, &weights, &mut census)
        .expect("the total q lane has a scale for every binary32 panel");
    (scale, census)
}

#[derive(Clone, Copy, Debug)]
struct ScalarCertificate {
    bound: u128,
    calls: u64,
    observations: u64,
    q_macs: u64,
    special: bool,
}

#[allow(clippy::too_many_arguments)]
fn scalar_certificate<const D: usize>(
    fixture: &Fixture<D>,
    row0: usize,
    rows: usize,
    col0: usize,
    p: usize,
    representatives: &[usize],
    call_scale: LaneScale,
    cap: u128,
) -> ScalarCertificate {
    let full_a = MatView::row_major(
        as_alphabet_whole(&fixture.a),
        fixture.shape.m,
        fixture.shape.k,
    )
    .expect("A has its declared shape");
    let a = full_a
        .subview(row0, p, rows, 1)
        .expect("the scalar activation column is in A");
    let mut bound = 0u128;
    let mut calls = 0u64;
    let mut observations = 0u64;
    let mut q_macs = 0u64;
    let mut special = (0..rows).any(|i| !fixture.a[(row0 + i) * fixture.shape.k + p].is_finite());
    for &j in representatives {
        let code_at = (col0 + j) * fixture.shape.k + p;
        let one_code = &fixture.codes[code_at..code_at + 1];
        let index = usize::from(one_code[0]) % D;
        special |= !fixture.table[index].is_finite();
        let one = CodedMatrix::new(
            Arena::<f32, D, u8>::new(fixture.table_alphabet()),
            1,
            1,
            one_code,
        )
        .expect("one stored scalar is conformant");
        let mut census = Census::default();
        let local = <f32 as Tabulated>::lane_scale(&a, &one, &mut census)
            .expect("an admitted q lane remains admitted on a source subset");
        calls += 1;
        observations += census.decodes;
        q_macs += census.adds;
        bound = bound.max(regrade_envelope(local, call_scale, cap));
    }
    ScalarCertificate {
        bound,
        calls,
        observations,
        q_macs,
        special,
    }
}

fn representative_columns(collapsed: Option<&[usize]>, cols: usize) -> Vec<usize> {
    match collapsed {
        Some(first) => first[..cols]
            .iter()
            .enumerate()
            .filter_map(|(j, &representative)| (representative == j).then_some(j))
            .collect(),
        None => (0..cols).collect(),
    }
}

fn demanded_entries<const D: usize>(
    fixture: &Fixture<D>,
    p: usize,
    col0: usize,
    representatives: &[usize],
    entries: Option<&mut EntryReplay>,
) -> (Vec<usize>, bool, usize) {
    if let Some(entries) = entries {
        let mut complete = true;
        for &j in representatives {
            let code = fixture.codes[(col0 + j) * fixture.shape.k + p];
            if entries.insert(usize::from(code) % D) == ReplayInsert::Full {
                complete = false;
                break;
            }
        }
        if complete && entries.used < D {
            let indices = entries.indices();
            entries.clear();
            return (indices, false, 0);
        }
        entries.clear();
        return ((0..D).collect(), true, 0);
    }
    if representatives.len() < D {
        let indices: Vec<_> = representatives
            .iter()
            .map(|&j| usize::from(fixture.codes[(col0 + j) * fixture.shape.k + p]) % D)
            .collect();
        let distinct = indices.iter().copied().collect::<BTreeSet<_>>().len();
        let duplicates = indices.len() - distinct;
        (indices, false, duplicates)
    } else {
        ((0..D).collect(), true, 0)
    }
}

// Each field mirrors one independently censused production coordinate.
#[allow(clippy::too_many_arguments)]
fn record_build<const D: usize>(
    fixture: &Fixture<D>,
    p: usize,
    row0: usize,
    rows: usize,
    indices: &[usize],
    full: bool,
    duplicates: usize,
    diagnostic: &mut DiagnosticWork,
) {
    if full {
        diagnostic.full_build_entries += indices.len() as u64;
    } else {
        diagnostic.sparse_build_entries += indices.len() as u64;
    }
    diagnostic.duplicate_build_entries += duplicates as u64;
    diagnostic.build_contractions += (indices.len() * rows) as u64;
    for &index in indices {
        for i in 0..rows {
            let activation = fixture.a[(row0 + i) * fixture.shape.k + p];
            let book = fixture.table[index];
            let (pairs, full_rectangle) = q_lookup_pairs(activation, book);
            diagnostic.q_lookup_pairs += pairs as u64;
            diagnostic.full_q_rectangles += u64::from(full_rectangle);
        }
    }
}

fn record_place(rows: usize, cols: usize, diagnostic: &mut DiagnosticWork) {
    diagnostic.placed_cells += (rows * cols) as u64;
}

fn replay_diagnostic<const D: usize>(
    fixture: &Fixture<D>,
    offer: Offer,
    facts: ModelFacts,
) -> DiagnosticWork {
    let Shape { m, k, n } = fixture.shape;
    let mut columns = replay_columns(fixture, offer, facts);
    let mut entries = (D > 1 && columns.dictionary_extent != 0)
        .then(|| EntryReplay::new(columns.dictionary_extent, columns.entry_occupied));
    if entries.is_some() {
        // The dead column slot table is fully cleared before becoming EntrySet.
        columns.events.dictionary_clear_words += columns.dictionary_extent as u64;
    }
    let addressed = entries
        .as_mut()
        .and_then(|set| set.collect(&fixture.codes, D));
    let call_entries = entries.as_mut().map_or_else(EntryEvents::default, |set| {
        let events = set.events;
        set.events = EntryEvents::default();
        events
    });
    let (scale, scale_census) = call_lane_scale(fixture, addressed.as_deref());
    let book_presentations = addressed.as_ref().map_or_else(
        || {
            if fixture.codes.len() < D {
                fixture.codes.len()
            } else {
                D
            }
        },
        Vec::len,
    );
    let observed_run = <f32 as Tabulated>::lane_run::<<f32 as Tabulated>::Lane>(
        <Whole<f32> as Bound>::VALUE,
        &scale,
    )
    .expect("the q lane reports its finite register capacity");
    let local_envelopes = observed_run < k;
    let cache_rows = offer.cache_rows;
    let column_blocks = n.div_ceil(offer.plan.cols);
    let activation_q_projections = cache_rows * k + (m - cache_rows) * k * column_blocks;
    let tiles = row_tiles(m, offer.plan.rows);
    let slab = slab_codes(D);
    let mut geometries = BTreeSet::new();
    let padding_words = tiles
        .iter()
        .copied()
        .filter(|rows| geometries.insert(*rows))
        .map(|rows| (slab - D) * rows * offer.plan.depth)
        .sum::<usize>();
    let page = offer.panel.min(k);
    let (dense_decodes, dense_calls, dense_joins) = if offer.panel >= k {
        (n * k, n * m.div_ceil(ROW_TILES[0]), 0)
    } else {
        let pages = k.div_ceil(page);
        (m * n * k, m * n * pages, m * n * (pages - 1))
    };
    let mut diagnostic = DiagnosticWork {
        column: columns.events,
        call_entries,
        call_scale: Some(scale),
        call_scale_observations: scale_census.decodes,
        call_scale_q_macs: scale_census.adds,
        book_codec_decodes: book_presentations as u64,
        book_q_projections: book_presentations as u64,
        activation_q_projections: activation_q_projections as u64,
        padding_words: padding_words as u64,
        dense_decodes: dense_decodes as u64,
        dense_calls: dense_calls as u64,
        dense_joins: dense_joins as u64,
        ..DiagnosticWork::default()
    };

    let mut row0 = 0usize;
    for rows in tiles {
        for col0 in (0..n).step_by(offer.plan.cols) {
            let cols = offer.plan.cols.min(n - col0);
            let collapsed = columns.collapsed(col0, cols, offer.plan.cols);
            let representatives = representative_columns(collapsed, cols);
            if local_envelopes {
                let mut height = 0u128;
                let mut pending = false;
                for p in 0..k {
                    let first = scalar_certificate(
                        fixture,
                        row0,
                        rows,
                        col0,
                        p,
                        &representatives,
                        scale,
                        facts.compact_ceiling,
                    );
                    diagnostic.scalar_envelope_calls += first.calls;
                    diagnostic.scalar_certificate_observations += first.observations;
                    diagnostic.scalar_certificate_q_macs += first.q_macs;
                    if first.bound <= facts.compact_ceiling {
                        diagnostic.safe_source_aggregates += 1;
                        if pending && first.bound > facts.compact_ceiling - height {
                            diagnostic.capacity_flushes += 1;
                            record_place(rows, cols, &mut diagnostic);
                            height = 0;
                        }
                        let (indices, full, duplicates) =
                            demanded_entries(fixture, p, col0, &representatives, entries.as_mut());
                        record_build(
                            fixture,
                            p,
                            row0,
                            rows,
                            &indices,
                            full,
                            duplicates,
                            &mut diagnostic,
                        );
                        diagnostic.gather_adds += (representatives.len() * rows) as u64;
                        height += first.bound;
                        pending = true;
                        continue;
                    }

                    diagnostic.fractured_source_aggregates += 1;
                    let replay = scalar_certificate(
                        fixture,
                        row0,
                        rows,
                        col0,
                        p,
                        &representatives,
                        scale,
                        facts.compact_ceiling,
                    );
                    diagnostic.scalar_envelope_calls += replay.calls;
                    diagnostic.scalar_envelope_replays += replay.calls;
                    diagnostic.scalar_certificate_observations += replay.observations;
                    diagnostic.scalar_certificate_q_macs += replay.q_macs;
                    let singleton = replay.bound > facts.compact_ceiling;
                    if pending && (singleton || replay.bound > facts.compact_ceiling - height) {
                        diagnostic.capacity_flushes += 1;
                        record_place(rows, cols, &mut diagnostic);
                        height = 0;
                    }
                    let (indices, full, duplicates) =
                        demanded_entries(fixture, p, col0, &representatives, entries.as_mut());
                    record_build(
                        fixture,
                        p,
                        row0,
                        rows,
                        &indices,
                        full,
                        duplicates,
                        &mut diagnostic,
                    );
                    diagnostic.gather_adds += (representatives.len() * rows) as u64;
                    if singleton {
                        diagnostic.singleton_placements += 1;
                        if replay.special {
                            diagnostic.special_tag_singletons += 1;
                        } else {
                            diagnostic.finite_tag_singletons += 1;
                        }
                        record_place(rows, cols, &mut diagnostic);
                        pending = false;
                        height = 0;
                    } else {
                        height += replay.bound;
                        pending = true;
                    }
                }
                if pending {
                    diagnostic.tail_placements += 1;
                    record_place(rows, cols, &mut diagnostic);
                }
            } else {
                for p in 0..k {
                    let (indices, full, duplicates) =
                        demanded_entries(fixture, p, col0, &representatives, entries.as_mut());
                    record_build(
                        fixture,
                        p,
                        row0,
                        rows,
                        &indices,
                        full,
                        duplicates,
                        &mut diagnostic,
                    );
                    diagnostic.gather_adds += (representatives.len() * rows) as u64;
                }
                diagnostic.tail_placements += 1;
                record_place(rows, cols, &mut diagnostic);
            }
        }
        row0 += rows;
    }
    diagnostic.build_entries = entries.map_or_else(EntryEvents::default, |set| set.events);
    diagnostic
}

fn expected_table_census(diagnostic: &DiagnosticWork) -> Census {
    Census {
        multiplies: 0,
        adds: diagnostic.call_scale_q_macs
            + diagnostic.scalar_certificate_q_macs
            + diagnostic.build_contractions
            + diagnostic.gather_adds,
        table_reads: diagnostic.gather_adds,
        decodes: diagnostic.call_scale_observations
            + diagnostic.book_codec_decodes
            + diagnostic.book_q_projections
            + diagnostic.activation_q_projections
            + diagnostic.scalar_certificate_observations,
        kernel_calls: 0,
    }
}

fn expected_dense_census(diagnostic: &DiagnosticWork) -> Census {
    Census {
        multiplies: 0,
        adds: diagnostic.dense_joins,
        table_reads: 0,
        decodes: diagnostic.dense_decodes,
        kernel_calls: diagnostic.dense_calls,
    }
}

fn assert_profile(case: Case, diagnostic: &DiagnosticWork) {
    match case.values {
        ValueProfile::CompactFull => {
            assert_ne!(
                diagnostic.full_q_rectangles, 0,
                "{} realizes q[4] x q[4]",
                case.id
            );
            assert_eq!(
                diagnostic.full_q_rectangles, diagnostic.build_contractions,
                "{} keeps every finite contraction at the full 4-by-4 extent",
                case.id
            );
        }
        ValueProfile::WideLocal => {
            assert_ne!(
                diagnostic.scalar_envelope_calls, 0,
                "{} enters the source-local scheduler",
                case.id
            );
            assert_ne!(
                diagnostic.capacity_flushes, 0,
                "{} realizes a finite capacity flush",
                case.id
            );
        }
        ValueProfile::FiniteTag => assert_ne!(
            diagnostic.finite_tag_singletons, 0,
            "{} realizes a finite tagged singleton",
            case.id
        ),
        ValueProfile::SpecialAlternating | ValueProfile::SpecialSingleton => assert_ne!(
            diagnostic.special_tag_singletons, 0,
            "{} realizes a non-finite source singleton",
            case.id
        ),
    }
}

#[derive(Clone, Debug)]
struct PreparedCase {
    case: Case,
    structural: StructuralWork,
}

fn prepare_case_with<const D: usize>(case: Case, facts: ModelFacts) -> PreparedCase {
    let fixture = Fixture::<D>::new(case);
    let offer = resolve_offer::<D>(case, fixture.shape);
    PreparedCase {
        case,
        structural: StructuralWork::of(&fixture, offer, facts),
    }
}

fn prepare_case(case: Case, facts: ModelFacts) -> PreparedCase {
    match case.d {
        1 => prepare_case_with::<1>(case, facts),
        3 => prepare_case_with::<3>(case, facts),
        5 => prepare_case_with::<5>(case, facts),
        16 => prepare_case_with::<16>(case, facts),
        256 => prepare_case_with::<256>(case, facts),
        _ => panic!("{} has an uninstantiated Arena extent", case.id),
    }
}

#[derive(Debug)]
struct Observation {
    case: Case,
    structural: StructuralWork,
    diagnostic: DiagnosticWork,
    forced_table: Census,
    forced_dense: Census,
    timing: PairTiming,
}

fn measure_case_with<const D: usize>(prepared: &PreparedCase, facts: ModelFacts) -> Observation {
    let case = prepared.case;
    let fixture = Fixture::<D>::new(case);
    let offer = resolve_offer::<D>(case, fixture.shape);
    let structural = StructuralWork::of(&fixture, offer, facts);
    assert_eq!(
        structural, prepared.structural,
        "{} retained its pre-clock StructuralWork",
        case.id
    );
    let diagnostic = replay_diagnostic(&fixture, offer, facts);
    assert_profile(case, &diagnostic);

    let output_len = fixture.shape.m * fixture.shape.n;
    let mut dense_workspace = Workspace::new(offer);
    let mut dense_output = vec![POISON; output_len];
    let forced_dense = counted_once(
        &fixture,
        Traversal::OutputMajor,
        &mut dense_workspace,
        &mut dense_output,
    );
    assert_eq!(
        forced_dense,
        expected_dense_census(&diagnostic),
        "{} dense diagnostic reconciles Census",
        case.id
    );
    let expected: Vec<u32> = dense_output.iter().map(|value| value.to_bits()).collect();

    let mut table_workspace = Workspace::new(offer);
    let mut table_output = vec![POISON; output_len];
    let forced_table = counted_once(
        &fixture,
        Traversal::Tabulated,
        &mut table_workspace,
        &mut table_output,
    );
    verify_bits(&table_output, &expected);
    assert_eq!(
        forced_table,
        expected_table_census(&diagnostic),
        "{} table diagnostic reconciles Census",
        case.id
    );

    let timing = measure_pair(
        &fixture,
        &mut table_workspace,
        &mut dense_workspace,
        &mut table_output,
        &mut dense_output,
        &expected,
    );
    Observation {
        case,
        structural,
        diagnostic,
        forced_table,
        forced_dense,
        timing,
    }
}

fn measure_case(prepared: &PreparedCase, facts: ModelFacts) -> Observation {
    match prepared.case.d {
        1 => measure_case_with::<1>(prepared, facts),
        3 => measure_case_with::<3>(prepared, facts),
        5 => measure_case_with::<5>(prepared, facts),
        16 => measure_case_with::<16>(prepared, facts),
        256 => measure_case_with::<256>(prepared, facts),
        _ => panic!("{} has an uninstantiated Arena extent", prepared.case.id),
    }
}

fn reconcile_case_with<const D: usize>(case: Case, facts: ModelFacts) {
    let fixture = Fixture::<D>::new(case);
    let offer = resolve_offer::<D>(case, fixture.shape);
    let diagnostic = replay_diagnostic(&fixture, offer, facts);
    assert_profile(case, &diagnostic);

    let output_len = fixture.shape.m * fixture.shape.n;
    let mut dense_workspace = Workspace::new(offer);
    let mut dense_output = vec![POISON; output_len];
    let forced_dense = counted_once(
        &fixture,
        Traversal::OutputMajor,
        &mut dense_workspace,
        &mut dense_output,
    );
    assert_eq!(
        forced_dense,
        expected_dense_census(&diagnostic),
        "CG16_LIVE_B1_RECONCILIATION: {} dense Census",
        case.id,
    );
    let expected: Vec<u32> = dense_output.iter().map(|value| value.to_bits()).collect();

    let mut table_workspace = Workspace::new(offer);
    let mut table_output = vec![POISON; output_len];
    let forced_table = counted_once(
        &fixture,
        Traversal::Tabulated,
        &mut table_workspace,
        &mut table_output,
    );
    verify_bits(&table_output, &expected);
    assert_eq!(
        forced_table,
        expected_table_census(&diagnostic),
        "CG16_LIVE_B1_RECONCILIATION: {} table Census",
        case.id,
    );
}

#[derive(Clone, Debug)]
struct CalibrationEnvelope {
    structural: StructuralWork,
    ids: Vec<&'static str>,
    table_target: f64,
    dense_target: f64,
    table_rounds: [f64; SAMPLE_COUNT],
    dense_rounds: [f64; SAMPLE_COUNT],
}

fn calibration_envelopes(observations: &[Observation]) -> Vec<CalibrationEnvelope> {
    let mut grouped: BTreeMap<StructuralWork, Vec<&Observation>> = BTreeMap::new();
    for observation in observations {
        assert_eq!(observation.case.role, Role::Calibration);
        grouped
            .entry(observation.structural.clone())
            .or_default()
            .push(observation);
    }
    let envelopes: Vec<_> = grouped
        .into_iter()
        .map(|(structural, twins)| CalibrationEnvelope {
            structural,
            ids: twins
                .iter()
                .map(|observation| observation.case.id)
                .collect(),
            // The selector must be non-regressing for every value that shares
            // this key, hence the adversarial directions are intentional.
            table_target: twins
                .iter()
                .map(|observation| observation.timing.table.upper())
                .fold(0.0, f64::max),
            dense_target: twins
                .iter()
                .map(|observation| observation.timing.dense.lower())
                .fold(f64::INFINITY, f64::min),
            table_rounds: std::array::from_fn(|round| {
                twins
                    .iter()
                    .map(|observation| observation.timing.table_samples[round])
                    .fold(0.0, f64::max)
            }),
            dense_rounds: std::array::from_fn(|round| {
                twins
                    .iter()
                    .map(|observation| observation.timing.dense_samples[round])
                    .fold(f64::INFINITY, f64::min)
            }),
        })
        .collect();
    assert_eq!(envelopes.len(), 28);
    assert_eq!(
        envelopes
            .iter()
            .filter(|envelope| envelope.ids.len() == 2)
            .count(),
        4,
        "exactly four calibration keys carry value twins"
    );
    envelopes
}

#[derive(Clone, Copy, Debug)]
struct NonnegativeFit<const C: usize> {
    coefficients: [f64; C],
    residual_sum_squares: f64,
    active: usize,
}

// Indexed row operations keep the pivot row and active column visibly paired.
#[allow(clippy::needless_range_loop)]
fn solve_linear(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    if n == 0 {
        return Some(Vec::new());
    }
    let matrix_norm = a
        .iter()
        .flat_map(|row| row.iter())
        .map(|value| value.abs())
        .fold(0.0, f64::max);
    let pivot_floor = f64::EPSILON * matrix_norm.max(1.0) * n as f64;
    for column in 0..n {
        let pivot = (column..n)
            .max_by(|&left, &right| a[left][column].abs().total_cmp(&a[right][column].abs()))?;
        if a[pivot][column].abs() <= pivot_floor {
            return None;
        }
        a.swap(column, pivot);
        b.swap(column, pivot);
        let divisor = a[column][column];
        for at in column..n {
            a[column][at] /= divisor;
        }
        b[column] /= divisor;
        for row in 0..n {
            if row == column {
                continue;
            }
            let factor = a[row][column];
            for at in column..n {
                a[row][at] -= factor * a[column][at];
            }
            b[row] -= factor * b[column];
        }
    }
    Some(b)
}

/// Exact active-set enumeration with only a machine-roundoff gate. The design
/// has fourteen coordinates, so all `2^14` faces are finite and no iteration
/// count, regularizer, or guessed stopping coefficient participates.
fn nonnegative_fit<const C: usize>(rows: &[[u128; C]], target: &[f64]) -> NonnegativeFit<C> {
    assert_eq!(rows.len(), target.len());
    assert!(!rows.is_empty() && C < usize::BITS as usize);
    let column_scale: [f64; C] = std::array::from_fn(|column| {
        rows.iter()
            .map(|row| row[column] as f64)
            .fold(0.0, f64::max)
            .max(1.0)
    });
    let target_scale = target
        .iter()
        .copied()
        .fold(0.0, f64::max)
        .max(f64::MIN_POSITIVE);
    let x: Vec<[f64; C]> = rows
        .iter()
        .map(|row| std::array::from_fn(|column| row[column] as f64 / column_scale[column]))
        .collect();
    let y: Vec<f64> = target.iter().map(|value| value / target_scale).collect();
    // Normal equations square the condition number. The square root of machine
    // epsilon is therefore the mechanically implied sign/KKT uncertainty; it
    // is not a fitted tolerance or a domain coefficient.
    let feasibility = f64::EPSILON.sqrt();
    let mut best: Option<NonnegativeFit<C>> = None;

    for mask in 0usize..(1usize << C) {
        let active: Vec<_> = (0..C)
            .filter(|&column| mask & (1usize << column) != 0)
            .collect();
        let gamma = if active.is_empty() {
            Vec::new()
        } else {
            let mut gram = vec![vec![0.0; active.len()]; active.len()];
            let mut rhs = vec![0.0; active.len()];
            for (left_at, &left) in active.iter().enumerate() {
                rhs[left_at] = x
                    .iter()
                    .zip(&y)
                    .map(|(row, &value)| row[left] * value)
                    .sum();
                for (right_at, &right) in active.iter().enumerate() {
                    gram[left_at][right_at] = x.iter().map(|row| row[left] * row[right]).sum();
                }
            }
            let Some(gamma) = solve_linear(gram, rhs) else {
                continue;
            };
            gamma
        };
        if gamma.iter().any(|&value| value < -feasibility) {
            continue;
        }
        let mut normalized = [0.0; C];
        for (&column, &value) in active.iter().zip(&gamma) {
            normalized[column] = value.max(0.0);
        }
        let residuals: Vec<_> = x
            .iter()
            .zip(&y)
            .map(|(row, &value)| {
                row.iter()
                    .zip(normalized)
                    .map(|(&coordinate, coefficient)| coordinate * coefficient)
                    .sum::<f64>()
                    - value
            })
            .collect();
        let gradients: [f64; C] = std::array::from_fn(|column| {
            x.iter()
                .zip(&residuals)
                .map(|(row, &residual)| row[column] * residual)
                .sum()
        });
        if (0..C).any(|column| mask & (1usize << column) == 0 && gradients[column] < -feasibility) {
            continue;
        }
        let rss_normalized = residuals
            .iter()
            .map(|residual| residual * residual)
            .sum::<f64>();
        let candidate = NonnegativeFit {
            coefficients: std::array::from_fn(|column| {
                normalized[column] * target_scale / column_scale[column]
            }),
            residual_sum_squares: rss_normalized * target_scale * target_scale,
            active: normalized.iter().filter(|&&value| value > 0.0).count(),
        };
        if best.is_none_or(|prior| candidate.residual_sum_squares < prior.residual_sum_squares) {
            best = Some(candidate);
        }
    }
    best.expect("the nonnegative least-squares cone has a finite optimum")
}

#[derive(Clone, Copy, Debug)]
struct Coefficient {
    central: f64,
    estimate: Estimate,
}

impl Coefficient {
    fn conservative_upper(self) -> f64 {
        self.central.max(self.estimate.upper())
    }

    fn conservative_lower(self) -> f64 {
        self.central.min(self.estimate.lower()).max(0.0)
    }
}

#[derive(Clone, Debug)]
struct SelectorModel {
    table: [Coefficient; TABLE_BASIS],
    dense: [Coefficient; DENSE_BASIS],
    table_fit: NonnegativeFit<TABLE_BASIS>,
    dense_fit: NonnegativeFit<DENSE_BASIS>,
    table_round_fits: [NonnegativeFit<TABLE_BASIS>; SAMPLE_COUNT],
    dense_round_fits: [NonnegativeFit<DENSE_BASIS>; SAMPLE_COUNT],
}

fn fit_selector(envelopes: &[CalibrationEnvelope]) -> SelectorModel {
    let table_rows: Vec<_> = envelopes
        .iter()
        .map(|envelope| envelope.structural.table_basis())
        .collect();
    let dense_rows: Vec<_> = envelopes
        .iter()
        .map(|envelope| envelope.structural.dense_basis())
        .collect();
    let table_targets: Vec<_> = envelopes
        .iter()
        .map(|envelope| envelope.table_target)
        .collect();
    let dense_targets: Vec<_> = envelopes
        .iter()
        .map(|envelope| envelope.dense_target)
        .collect();
    let table_fit = nonnegative_fit(&table_rows, &table_targets);
    let dense_fit = nonnegative_fit(&dense_rows, &dense_targets);
    let table_round_fits: [_; SAMPLE_COUNT] = std::array::from_fn(|round| {
        let targets: Vec<_> = envelopes
            .iter()
            .map(|envelope| envelope.table_rounds[round])
            .collect();
        nonnegative_fit(&table_rows, &targets)
    });
    let dense_round_fits: [_; SAMPLE_COUNT] = std::array::from_fn(|round| {
        let targets: Vec<_> = envelopes
            .iter()
            .map(|envelope| envelope.dense_rounds[round])
            .collect();
        nonnegative_fit(&dense_rows, &targets)
    });
    SelectorModel {
        table: std::array::from_fn(|column| Coefficient {
            central: table_fit.coefficients[column],
            estimate: Estimate::of(&std::array::from_fn(|round| {
                table_round_fits[round].coefficients[column]
            })),
        }),
        dense: std::array::from_fn(|column| Coefficient {
            central: dense_fit.coefficients[column],
            estimate: Estimate::of(&std::array::from_fn(|round| {
                dense_round_fits[round].coefficients[column]
            })),
        }),
        table_fit,
        dense_fit,
        table_round_fits,
        dense_round_fits,
    }
}

#[derive(Clone, Copy, Debug)]
struct Prediction {
    table_upper: f64,
    dense_lower: f64,
    route: Route,
}

fn candidate_route(structural: &StructuralWork, model: &SelectorModel) -> Prediction {
    let table_central = structural
        .table_basis()
        .iter()
        .zip(model.table_fit.coefficients)
        .map(|(&coordinate, coefficient)| coordinate as f64 * coefficient)
        .sum::<f64>();
    let dense_central = structural
        .dense_basis()
        .iter()
        .zip(model.dense_fit.coefficients)
        .map(|(&coordinate, coefficient)| coordinate as f64 * coefficient)
        .sum::<f64>();
    let table_samples = std::array::from_fn(|round| {
        structural
            .table_basis()
            .iter()
            .zip(model.table_round_fits[round].coefficients)
            .map(|(&coordinate, coefficient)| coordinate as f64 * coefficient)
            .sum::<f64>()
    });
    let dense_samples = std::array::from_fn(|round| {
        structural
            .dense_basis()
            .iter()
            .zip(model.dense_round_fits[round].coefficients)
            .map(|(&coordinate, coefficient)| coordinate as f64 * coefficient)
            .sum::<f64>()
    });
    let table_upper = table_central.max(Estimate::of(&table_samples).upper());
    let dense_lower = dense_central.min(Estimate::of(&dense_samples).lower());
    Prediction {
        table_upper,
        dense_lower,
        route: if table_upper < dense_lower {
            Route::Table
        } else {
            Route::Dense
        },
    }
}

#[derive(Clone, Copy, Debug)]
struct ParametricFracture<const B: usize, const D: usize>;

fn fracture_value(high: bool) -> f32 {
    if high {
        f32::from_bits(0x437f_ffff)
    } else {
        f32::from_bits(0x3fff_ffff)
    }
}

impl<const B: usize, const D: usize> Codec<f32, Whole<f32>> for ParametricFracture<B, D> {
    type Code = u8;
    const MAX_BLOCK: usize = B;
    const TIER: TierId = TierId::Book;

    fn decode_element(&self, code: Self::Code, i: usize) -> Alphabet<f32, Whole<f32>> {
        bytemuck::TransparentWrapper::wrap(fracture_value(
            !(usize::from(code) + i).is_multiple_of(2),
        ))
    }
}

impl<const B: usize, const D: usize> Enumerable<f32, Whole<f32>> for ParametricFracture<B, D> {
    const CODE_SPACE: usize = D;

    fn code_at(index: usize) -> Self::Code {
        u8::try_from(index % D).expect("the control code spaces fit u8")
    }

    fn index_of(code: Self::Code) -> usize {
        usize::from(code) % D
    }
}

#[derive(Clone, Copy, Debug)]
struct ControlScalar<const B: usize, const D: usize> {
    coordinate: usize,
}

impl<const B: usize, const D: usize> Codec<f32, Whole<f32>> for ControlScalar<B, D> {
    type Code = u8;
    const MAX_BLOCK: usize = 1;
    const TIER: TierId = TierId::Book;

    fn decode_element(&self, code: Self::Code, _: usize) -> Alphabet<f32, Whole<f32>> {
        <ParametricFracture<B, D> as Codec<f32, Whole<f32>>>::decode_element(
            &ParametricFracture,
            code,
            self.coordinate,
        )
    }
}

impl<const B: usize, const D: usize> Enumerable<f32, Whole<f32>> for ControlScalar<B, D> {
    const CODE_SPACE: usize = D;

    fn code_at(index: usize) -> Self::Code {
        <ParametricFracture<B, D> as Enumerable<f32, Whole<f32>>>::code_at(index)
    }

    fn index_of(code: Self::Code) -> usize {
        <ParametricFracture<B, D> as Enumerable<f32, Whole<f32>>>::index_of(code)
    }
}

#[derive(Clone, Copy, Debug)]
struct ControlCase {
    id: &'static str,
    block: usize,
    d: usize,
    m: usize,
    blocks: usize,
    n: usize,
}

const BLOCK_HOLDOUTS: &[ControlCase] = &[
    ControlCase {
        id: "H13-B3-D3",
        block: 3,
        d: 3,
        m: 17,
        blocks: 11,
        n: 7,
    },
    ControlCase {
        id: "H14-B5-D5",
        block: 5,
        d: 5,
        m: 9,
        blocks: 7,
        n: 11,
    },
];

struct ControlFixture<const B: usize, const D: usize> {
    case: ControlCase,
    shape: Shape,
    codes: Vec<u8>,
    a: Vec<f32>,
}

impl<const B: usize, const D: usize> ControlFixture<B, D> {
    fn new(case: ControlCase) -> Self {
        assert_eq!((case.block, case.d), (B, D));
        let shape = Shape {
            m: case.m,
            k: case.blocks * B,
            n: case.n,
        };
        let mut codes: Vec<_> = (0..shape.n * case.blocks)
            .map(|at| ((at * 3 + at / case.blocks + 1) % D) as u8)
            .collect();
        codes[0] = 0;
        let last = codes.len() - 1;
        codes[last] = (D - 1) as u8;
        let a = (0..shape.m * shape.k)
            .map(|at| fracture_value(!at.is_multiple_of(5)))
            .collect();
        Self {
            case,
            shape,
            codes,
            a,
        }
    }

    fn weights(&self) -> CodedMatrix<'_, f32, Whole<f32>, ParametricFracture<B, D>> {
        CodedMatrix::new(ParametricFracture, self.shape.n, self.shape.k, &self.codes)
            .expect("the fixed-width control stream is conformant")
    }
}

fn control_offer<const B: usize, const D: usize>(shape: Shape) -> Offer {
    let panel = suggested_tabulation_panel(D, B);
    let exact = suggested_tabulation::<f32, Whole<f32>>(shape, D, B);
    let lanes = suggested_tabulation_lanes::<f32, Whole<f32>>(shape, D, B);
    let lane_words = std::mem::size_of::<i64>() * lanes / <f32 as Tabulated>::LANE_BYTES;
    let plan = Plan::choose(
        D,
        shape,
        <f32 as Tabulated>::LANE_BYTES,
        exact,
        lane_words,
        B,
        <f32 as Tabulated>::probe_capacity::<<f32 as Tabulated>::Lane>(
            <Whole<f32> as Bound>::VALUE,
        ),
    )
    .expect("the full control offers admit a q table");
    Offer {
        panel,
        exact,
        lanes,
        // These controls isolate scalar fracture; the B=1 matrix owns collapse
        // and EntrySet coverage.
        index: 0,
        plan,
        cache_rows: 0,
    }
}

// These arguments are the exact scalar scheduler certificate, not a convenience API.
#[allow(clippy::too_many_arguments)]
fn control_scalar_certificate<const B: usize, const D: usize>(
    fixture: &ControlFixture<B, D>,
    row0: usize,
    rows: usize,
    col0: usize,
    cols: usize,
    p: usize,
    coordinate: usize,
    call: LaneScale,
    cap: u128,
) -> ScalarCertificate {
    let full_a = MatView::row_major(
        as_alphabet_whole(&fixture.a),
        fixture.shape.m,
        fixture.shape.k,
    )
    .expect("control A has its declared shape");
    let source = p * B + coordinate;
    let a = full_a
        .subview(row0, source, rows, 1)
        .expect("the scalar coordinate is in control A");
    let mut bound = 0u128;
    let mut observations = 0u64;
    let mut q_macs = 0u64;
    for j in 0..cols {
        let at = (col0 + j) * fixture.case.blocks + p;
        let one_code = &fixture.codes[at..at + 1];
        let one = CodedMatrix::new(ControlScalar::<B, D> { coordinate }, 1, 1, one_code)
            .expect("one control scalar is conformant");
        let mut census = Census::default();
        let local = <f32 as Tabulated>::lane_scale(&a, &one, &mut census)
            .expect("control scalar retains the q lane");
        bound = bound.max(regrade_envelope(local, call, cap));
        observations += census.decodes;
        q_macs += census.adds;
    }
    ScalarCertificate {
        bound,
        calls: cols as u64,
        observations,
        q_macs,
        special: false,
    }
}

#[derive(Clone, Debug)]
struct ControlDiagnostic {
    id: &'static str,
    scalar_calls: u64,
    scalar_replays: u64,
    fractured_aggregates: u64,
    safe_scalar_atoms: u64,
    build_contractions: u64,
    gather_adds: u64,
    capacity_flushes: u64,
    singleton_placements: u64,
    tail_placements: u64,
    table_census: Census,
    dense_census: Census,
}

fn replay_control<const B: usize, const D: usize>(
    fixture: &ControlFixture<B, D>,
    offer: Offer,
    facts: ModelFacts,
) -> ControlDiagnostic {
    let a = MatView::row_major(
        as_alphabet_whole(&fixture.a),
        fixture.shape.m,
        fixture.shape.k,
    )
    .expect("control A has its declared shape");
    let mut scale_census = Census::default();
    let scale = <f32 as Tabulated>::lane_scale(&a, &fixture.weights(), &mut scale_census)
        .expect("the control has a total q scale");
    let observed_run = <f32 as Tabulated>::lane_run::<<f32 as Tabulated>::Lane>(
        <Whole<f32> as Bound>::VALUE,
        &scale,
    )
    .expect("the control q lane reports capacity");
    assert!(
        observed_run < fixture.shape.k,
        "{} enters source-local scheduling",
        fixture.case.id
    );
    let tiles = row_tiles(fixture.shape.m, offer.plan.rows);
    let mut scalar_calls = 0u64;
    let mut scalar_replays = 0u64;
    let mut scalar_observations = 0u64;
    let mut scalar_q_macs = 0u64;
    let mut fractured_aggregates = 0u64;
    let mut safe_scalar_atoms = 0u64;
    let mut build_contractions = 0u64;
    let mut gather_adds = 0u64;
    let mut capacity_flushes = 0u64;
    let mut singleton_placements = 0u64;
    let mut tail_placements = 0u64;
    let mut row0 = 0usize;
    for rows in tiles {
        for col0 in (0..fixture.shape.n).step_by(offer.plan.cols) {
            let cols = offer.plan.cols.min(fixture.shape.n - col0);
            let mut height = 0u128;
            let mut pending = false;
            for p in 0..fixture.case.blocks {
                let mut aggregate = 0u128;
                for coordinate in 0..B {
                    let certificate = control_scalar_certificate(
                        fixture,
                        row0,
                        rows,
                        col0,
                        cols,
                        p,
                        coordinate,
                        scale,
                        facts.compact_ceiling,
                    );
                    scalar_calls += certificate.calls;
                    scalar_observations += certificate.observations;
                    scalar_q_macs += certificate.q_macs;
                    if certificate.bound > facts.compact_ceiling
                        || aggregate > facts.compact_ceiling - certificate.bound
                    {
                        aggregate = facts.compact_ceiling + 1;
                        break;
                    }
                    aggregate += certificate.bound;
                }
                if aggregate <= facts.compact_ceiling {
                    if pending && aggregate > facts.compact_ceiling - height {
                        capacity_flushes += 1;
                        height = 0;
                    }
                    build_contractions += (D * B * rows) as u64;
                    gather_adds += (cols * rows) as u64;
                    height += aggregate;
                    pending = true;
                    continue;
                }

                fractured_aggregates += 1;
                for coordinate in 0..B {
                    let certificate = control_scalar_certificate(
                        fixture,
                        row0,
                        rows,
                        col0,
                        cols,
                        p,
                        coordinate,
                        scale,
                        facts.compact_ceiling,
                    );
                    scalar_calls += certificate.calls;
                    scalar_replays += certificate.calls;
                    scalar_observations += certificate.observations;
                    scalar_q_macs += certificate.q_macs;
                    let singleton = certificate.bound > facts.compact_ceiling;
                    if pending && (singleton || certificate.bound > facts.compact_ceiling - height)
                    {
                        capacity_flushes += 1;
                        height = 0;
                    }
                    let entries = if cols < D { cols } else { D };
                    build_contractions += (entries * rows) as u64;
                    gather_adds += (cols * rows) as u64;
                    if singleton {
                        singleton_placements += 1;
                        pending = false;
                        height = 0;
                    } else {
                        safe_scalar_atoms += 1;
                        height += certificate.bound;
                        pending = true;
                    }
                }
            }
            if pending {
                tail_placements += 1;
            }
        }
        row0 += rows;
    }
    let blocks = fixture.shape.n.div_ceil(offer.plan.cols);
    let activation_projections = fixture.shape.m * fixture.shape.k * blocks;
    let book = D * B;
    let table_census = Census {
        multiplies: 0,
        adds: scale_census.adds + scalar_q_macs + build_contractions + gather_adds,
        table_reads: gather_adds,
        decodes: scale_census.decodes
            + (book * 2 + activation_projections) as u64
            + scalar_observations,
        kernel_calls: 0,
    };
    let page = offer.panel.min(fixture.shape.k);
    let (dense_decodes, dense_calls, dense_joins) = if offer.panel >= fixture.shape.k {
        (
            fixture.shape.n * fixture.shape.k,
            fixture.shape.n * fixture.shape.m.div_ceil(ROW_TILES[0]),
            0,
        )
    } else {
        let pages = fixture.shape.k.div_ceil(page);
        (
            fixture.shape.m * fixture.shape.n * fixture.shape.k,
            fixture.shape.m * fixture.shape.n * pages,
            fixture.shape.m * fixture.shape.n * (pages - 1),
        )
    };
    let dense_census = Census {
        multiplies: 0,
        adds: dense_joins as u64,
        table_reads: 0,
        decodes: dense_decodes as u64,
        kernel_calls: dense_calls as u64,
    };
    ControlDiagnostic {
        id: fixture.case.id,
        scalar_calls,
        scalar_replays,
        fractured_aggregates,
        safe_scalar_atoms,
        build_contractions,
        gather_adds,
        capacity_flushes,
        singleton_placements,
        tail_placements,
        table_census,
        dense_census,
    }
}

fn control_counted_once<const B: usize, const D: usize>(
    fixture: &ControlFixture<B, D>,
    traversal: Traversal,
    workspace: &mut Workspace,
    output: &mut [f32],
) -> Census {
    output.fill(POISON);
    let a = MatView::row_major(
        as_alphabet_whole(&fixture.a),
        fixture.shape.m,
        fixture.shape.k,
    )
    .expect("control A has its declared shape");
    let c = MatViewMut::row_major(output, fixture.shape.m, fixture.shape.n)
        .expect("control C has its declared shape");
    let mut triple =
        TabulatedTriple::new(a, fixture.weights(), c).expect("the control product exists");
    let mut scratch = Scratch::with_accumulators(&mut workspace.panel, &mut workspace.exact);
    let mut tabulation = Tabulation::with_index(&mut workspace.lanes, &mut workspace.index);
    let mut collapse = Collapse::none();
    let mut census = Census::default();
    gemm_tabulated_counted(
        &mut triple,
        &Linear::OVERWRITE,
        GemmOptions {
            traversal,
            backend: Backend::Auto,
            ..Default::default()
        },
        &mut scratch,
        &mut tabulation,
        &mut collapse,
        &mut census,
    );
    black_box(&*output);
    census
}

fn control_timed_batch<const B: usize, const D: usize>(
    fixture: &ControlFixture<B, D>,
    route: Route,
    repetitions: usize,
    workspace: &mut Workspace,
    output: &mut [f32],
    expected: &[u32],
) -> Duration {
    output.fill(POISON);
    let elapsed = {
        let a = MatView::row_major(
            as_alphabet_whole(&fixture.a),
            fixture.shape.m,
            fixture.shape.k,
        )
        .expect("control A has its declared shape");
        let c = MatViewMut::row_major(output, fixture.shape.m, fixture.shape.n)
            .expect("control C has its declared shape");
        let mut triple =
            TabulatedTriple::new(a, fixture.weights(), c).expect("the control product exists");
        let mut scratch = Scratch::with_accumulators(&mut workspace.panel, &mut workspace.exact);
        let mut tabulation = Tabulation::with_index(&mut workspace.lanes, &mut workspace.index);
        let mut collapse = Collapse::none();
        let options = GemmOptions {
            traversal: route.traversal(),
            backend: Backend::Auto,
            ..Default::default()
        };
        let start = Instant::now();
        for _ in 0..repetitions {
            gemm_tabulated(
                &mut triple,
                &Linear::OVERWRITE,
                options,
                &mut scratch,
                &mut tabulation,
                &mut collapse,
            );
        }
        start.elapsed()
    };
    verify_bits(output, expected); // CG16_CONTROL_FULL_BYTE_GUARD
    elapsed
}

#[allow(clippy::too_many_arguments)]
fn measure_control_pair<const B: usize, const D: usize>(
    fixture: &ControlFixture<B, D>,
    table_workspace: &mut Workspace,
    dense_workspace: &mut Workspace,
    table_output: &mut [f32],
    dense_output: &mut [f32],
    expected: &[u32],
) -> PairTiming {
    control_timed_batch(
        fixture,
        Route::Table,
        1,
        table_workspace,
        table_output,
        expected,
    );
    control_timed_batch(
        fixture,
        Route::Dense,
        1,
        dense_workspace,
        dense_output,
        expected,
    );
    let table_pilot = control_timed_batch(
        fixture,
        Route::Table,
        1,
        table_workspace,
        table_output,
        expected,
    );
    let dense_pilot = control_timed_batch(
        fixture,
        Route::Dense,
        1,
        dense_workspace,
        dense_output,
        expected,
    );
    let (table_batch, dense_batch) = (
        batch_repetitions(table_pilot),
        batch_repetitions(dense_pilot),
    );
    let mut table = [0.0; SAMPLE_COUNT];
    let mut dense = [0.0; SAMPLE_COUNT];
    for round in 0..SAMPLE_COUNT {
        let run_table = |workspace: &mut Workspace, output: &mut [f32]| {
            control_timed_batch(
                fixture,
                Route::Table,
                table_batch,
                workspace,
                output,
                expected,
            )
            .as_secs_f64()
                / table_batch as f64
        };
        let run_dense = |workspace: &mut Workspace, output: &mut [f32]| {
            control_timed_batch(
                fixture,
                Route::Dense,
                dense_batch,
                workspace,
                output,
                expected,
            )
            .as_secs_f64()
                / dense_batch as f64
        };
        if round.is_multiple_of(2) {
            table[round] = run_table(table_workspace, table_output);
            dense[round] = run_dense(dense_workspace, dense_output);
        } else {
            dense[round] = run_dense(dense_workspace, dense_output);
            table[round] = run_table(table_workspace, table_output);
        }
    }
    let ratio = std::array::from_fn(|round| table[round] / dense[round]);
    PairTiming {
        table: Estimate::of(&table),
        dense: Estimate::of(&dense),
        ratio: Estimate::of(&ratio),
        table_samples: table,
        dense_samples: dense,
        table_batch,
        dense_batch,
    }
}

#[derive(Debug)]
struct ControlObservation {
    diagnostic: ControlDiagnostic,
    timing: PairTiming,
}

fn measure_control_with<const B: usize, const D: usize>(
    case: ControlCase,
    facts: ModelFacts,
) -> ControlObservation {
    let fixture = ControlFixture::<B, D>::new(case);
    let offer = control_offer::<B, D>(fixture.shape);
    let diagnostic = replay_control(&fixture, offer, facts);
    assert_ne!(
        diagnostic.fractured_aggregates, 0,
        "{} fractures unsafe codec aggregates",
        case.id
    );
    assert_ne!(
        diagnostic.safe_scalar_atoms, 0,
        "{} recovers safe scalar source atoms",
        case.id
    );
    assert_eq!(
        diagnostic.singleton_placements, 0,
        "{} needs fracture, not a finite singleton fallback",
        case.id
    );
    let output_len = fixture.shape.m * fixture.shape.n;
    let mut dense_workspace = Workspace::new(offer);
    let mut dense_output = vec![POISON; output_len];
    let dense = control_counted_once(
        &fixture,
        Traversal::OutputMajor,
        &mut dense_workspace,
        &mut dense_output,
    );
    assert_eq!(dense, diagnostic.dense_census, "{} dense Census", case.id);
    let expected: Vec<u32> = dense_output.iter().map(|value| value.to_bits()).collect();
    let mut table_workspace = Workspace::new(offer);
    let mut table_output = vec![POISON; output_len];
    let table = control_counted_once(
        &fixture,
        Traversal::Tabulated,
        &mut table_workspace,
        &mut table_output,
    );
    verify_bits(&table_output, &expected);
    assert_eq!(table, diagnostic.table_census, "{} table Census", case.id);
    let timing = measure_control_pair(
        &fixture,
        &mut table_workspace,
        &mut dense_workspace,
        &mut table_output,
        &mut dense_output,
        &expected,
    );
    ControlObservation { diagnostic, timing }
}

fn measure_control(case: ControlCase, facts: ModelFacts) -> ControlObservation {
    match (case.block, case.d) {
        (3, 3) => measure_control_with::<3, 3>(case, facts),
        (5, 5) => measure_control_with::<5, 5>(case, facts),
        _ => panic!("{} has an uninstantiated block control", case.id),
    }
}

fn reconcile_control_with<const B: usize, const D: usize>(case: ControlCase, facts: ModelFacts) {
    let fixture = ControlFixture::<B, D>::new(case);
    let offer = control_offer::<B, D>(fixture.shape);
    let diagnostic = replay_control(&fixture, offer, facts);
    assert_ne!(
        diagnostic.fractured_aggregates, 0,
        "{} exercises scalar fracture",
        case.id,
    );
    assert_ne!(
        diagnostic.safe_scalar_atoms, 0,
        "{} recovers safe scalar atoms",
        case.id,
    );

    let output_len = fixture.shape.m * fixture.shape.n;
    let mut dense_workspace = Workspace::new(offer);
    let mut dense_output = vec![POISON; output_len];
    let dense = control_counted_once(
        &fixture,
        Traversal::OutputMajor,
        &mut dense_workspace,
        &mut dense_output,
    );
    assert_eq!(
        dense, diagnostic.dense_census,
        "CG16_LIVE_BLOCK_RECONCILIATION: {} dense Census",
        case.id,
    );
    let expected: Vec<u32> = dense_output.iter().map(|value| value.to_bits()).collect();

    let mut table_workspace = Workspace::new(offer);
    let mut table_output = vec![POISON; output_len];
    let table = control_counted_once(
        &fixture,
        Traversal::Tabulated,
        &mut table_workspace,
        &mut table_output,
    );
    verify_bits(&table_output, &expected);
    assert_eq!(
        table, diagnostic.table_census,
        "CG16_LIVE_BLOCK_RECONCILIATION: {} table Census",
        case.id,
    );
}

const LIVE_TABULATED_SOURCE: &str = include_str!("../../uor-matmul-gemm/src/tabulated.rs");
const HARNESS_SOURCE: &str = include_str!("symbol_tabulated_sweep.rs");

fn function_body<'a>(source: &'a str, name: &str) -> Result<&'a str, String> {
    let marker = format!("fn {name}");
    let start = source
        .find(&marker)
        .ok_or_else(|| format!("missing function {name}"))?;
    let open = source[start..]
        .find('{')
        .map(|offset| start + offset)
        .ok_or_else(|| format!("missing body for {name}"))?;
    let mut depth = 0usize;
    for (offset, byte) in source.as_bytes()[open..].iter().copied().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Ok(&source[open..open + offset + 1]);
                }
            }
            _ => {}
        }
    }
    Err(format!("unterminated body for {name}"))
}

fn require_all(haystack: &str, needles: &[&str], context: &str) -> Result<(), String> {
    for &needle in needles {
        if !haystack.contains(needle) {
            return Err(format!("{context} lost `{needle}`"));
        }
    }
    Ok(())
}

fn without_ascii_whitespace(source: &str) -> String {
    source
        .chars()
        .filter(|character| !character.is_ascii_whitespace())
        .collect()
}

fn audit_live_tabulated_source(source: &str) -> Result<(), String> {
    let hash = function_body(source, "column_hash")?;
    require_all(
        hash,
        &[
            "let mut hash = run.len() as u128",
            "run.len().min(crate::float::COLUMN_HASH_PREFIX)",
            "let doubled = hash + hash",
            "hash = doubled + hash + C::index_of(code) as u128",
            "(hash % modulus as u128) as usize",
        ],
        "column hash",
    )?;
    let compact_hash = without_ascii_whitespace(hash);
    if compact_hash.matches("%modulus").count() != 1 || compact_hash.contains("run.len()%modulus") {
        return Err("column hash must take its only remainder at the terminal address".into());
    }
    if hash.contains("wrapping_mul") || hash.contains("rotate_") {
        return Err("column hash regressed to a legacy word mixer".into());
    }

    let insert = function_body(source, "insert")?;
    require_all(
        insert,
        &[
            "let mut probe = if index < extent",
            "index % extent",
            "probe += 1",
            "if probe == extent",
        ],
        "EntrySet probe",
    )?;

    let regrade = function_body(source, "regrade_envelope")?;
    require_all(
        regrade,
        &[
            "local.base_a.checked_sub(call.base_a)",
            "local.base_b.checked_sub(call.base_b)",
            "bound > cap - bound",
            "bound += bound",
        ],
        "q regrade",
    )?;
    let scalar = function_body(source, "scalar_envelope")?;
    require_all(
        scalar,
        &[
            "E::lane_scale(&a, &one, ledger)",
            "regrade_envelope(local, call_scale, cap)",
        ],
        "scalar certificate",
    )?;
    let row_tile = function_body(source, "row_tile")?;
    require_all(
        row_tile,
        &[
            "if block_bound <= cap",
            "build_source_block",
            "build_source_scalar",
            "if singleton",
            "if tail_pending",
        ],
        "q scheduler",
    )?;
    if row_tile.matches("scalar_envelope(").count() < 2 {
        return Err("q scheduler lost its unsafe aggregate replay".into());
    }
    let build_presentations = function_body(source, "f32_q_build_presentations")?;
    require_all(
        build_presentations,
        &["count_product3(space, block, rows)"],
        "q build census",
    )?;
    let f32_impl = source
        .find("impl Tabulated for f32")
        .ok_or_else(|| "missing f32 Tabulated implementation".to_string())?;
    let lane_scale = function_body(&source[f32_impl..], "lane_scale")?;
    require_all(
        lane_scale,
        &["ledger.decoded(visits)", "ledger.added(1)"],
        "q scale census",
    )?;
    Ok(())
}

fn audit_selector_source(source: &str) -> Result<(), String> {
    let selector = function_body(source, "candidate_route")?;
    let compact_selector = without_ascii_whitespace(selector);
    require_all(
        &compact_selector,
        &[
            "structural.table_basis()",
            "structural.dense_basis()",
            "table_upper<dense_lower",
        ],
        "candidate selector",
    )?;
    for forbidden in ["diagnostic", "fixture", ".values", "Census", ".timing"] {
        if compact_selector.contains(forbidden) {
            return Err(format!(
                "candidate selector leaked post field `{forbidden}`"
            ));
        }
    }
    let envelopes = function_body(source, "calibration_envelopes")?;
    require_all(
        envelopes,
        &[
            "observation.timing.table.upper()",
            "observation.timing.dense.lower()",
            ".fold(0.0, f64::max)",
            ".fold(f64::INFINITY, f64::min)",
        ],
        "value-twin envelope",
    )?;
    if envelopes.matches(".fold(0.0, f64::max)").count() != 2
        || envelopes.matches(".fold(f64::INFINITY, f64::min)").count() != 2
    {
        return Err("value-twin envelope lost an adversarial central/round direction".into());
    }
    let fit = function_body(source, "fit_selector")?;
    let compact_fit = without_ascii_whitespace(fit);
    require_all(
        &compact_fit,
        &[
            "envelope.structural.table_basis()",
            "envelope.structural.dense_basis()",
            "nonnegative_fit",
        ],
        "selector fit",
    )?;
    if compact_fit.contains("diagnostic") || compact_fit.contains("case.values") {
        return Err("selector fit reads a post-execution value field".into());
    }
    Ok(())
}

fn audit_timer_source(source: &str) -> Result<(), String> {
    let body = function_body(source, "timed_batch")?;
    let poison = body
        .find("output.fill(POISON)")
        .ok_or_else(|| "timed batch lost poison".to_string())?;
    let start = body
        .find("let start = Instant::now()")
        .ok_or_else(|| "timed batch lost start".to_string())?;
    let call = body
        .find("gemm_tabulated(")
        .ok_or_else(|| "timed batch lost public call".to_string())?;
    let elapsed = body
        .find("start.elapsed()")
        .ok_or_else(|| "timed batch lost stop".to_string())?;
    let guard = body
        .find("CG16_FULL_BYTE_GUARD")
        .ok_or_else(|| "timed batch lost the complete byte guard".to_string())?;
    if !(poison < start && start < call && call < elapsed && elapsed < guard) {
        return Err("timed batch boundaries include setup or omit the guard".into());
    }
    if body[start..elapsed].matches("gemm_tabulated(").count() != 1 {
        return Err("the timed loop has anything but its one public call site".into());
    }
    if body[start..elapsed].contains("verify_bits") {
        return Err("the timed loop contains output verification".into());
    }

    let control = function_body(source, "control_timed_batch")?;
    let control_poison = control
        .find("output.fill(POISON)")
        .ok_or_else(|| "control timed batch lost poison".to_string())?;
    let control_start = control
        .find("let start = Instant::now()")
        .ok_or_else(|| "control timed batch lost start".to_string())?;
    let control_call = control
        .find("gemm_tabulated(")
        .ok_or_else(|| "control timed batch lost public call".to_string())?;
    let control_elapsed = control
        .find("start.elapsed()")
        .ok_or_else(|| "control timed batch lost stop".to_string())?;
    let control_guard = control
        .find("CG16_CONTROL_FULL_BYTE_GUARD")
        .ok_or_else(|| "control timed batch lost the complete byte guard".to_string())?;
    if !(control_poison < control_start
        && control_start < control_call
        && control_call < control_elapsed
        && control_elapsed < control_guard)
    {
        return Err("control timed batch boundaries include setup or omit the guard".into());
    }
    if control[control_start..control_elapsed]
        .matches("gemm_tabulated(")
        .count()
        != 1
        || control[control_start..control_elapsed].contains("verify_bits")
    {
        return Err("control timed loop is not exactly one unguarded production call site".into());
    }
    Ok(())
}

fn audit_raw_sample_source(source: &str) -> Result<(), String> {
    let samples = function_body(source, "print_pair_samples")?;
    require_all(
        samples,
        &[
            "for round in 0..SAMPLE_COUNT",
            "Route::Table.as_str()",
            "timing.table_samples[round]",
            "Route::Dense.as_str()",
            "timing.dense_samples[round]",
        ],
        "raw paired sample rows",
    )?;
    if samples.matches("SAMPLE,{split},{id},{key},{round}").count() != 2 {
        return Err("raw output lost one route row per immutable paired round".into());
    }

    let observation = function_body(source, "print_observation")?;
    require_all(
        observation,
        &["print_pair_samples(split, observation.case.id, &key, &observation.timing)"],
        "calibration/holdout raw samples",
    )?;
    let release = function_body(source, "symbol_tabulated_value_blind_boundary_cg_16")?;
    require_all(
        release,
        &["print_pair_samples(\"CONTROL\", case.id, &key, &observation.timing)"],
        "block-control raw samples",
    )?;
    Ok(())
}

fn audit_reconciliation_source(source: &str) -> Result<(), String> {
    let b1 = function_body(source, "reconcile_case_with")?;
    require_all(
        b1,
        &[
            "expected_dense_census(&diagnostic)",
            "expected_table_census(&diagnostic)",
            "verify_bits(&table_output, &expected)",
            "CG16_LIVE_B1_RECONCILIATION",
        ],
        "live B=1 replay/Census reconciliation",
    )?;
    let block = function_body(source, "reconcile_control_with")?;
    require_all(
        block,
        &[
            "diagnostic.dense_census",
            "diagnostic.table_census",
            "verify_bits(&table_output, &expected)",
            "CG16_LIVE_BLOCK_RECONCILIATION",
        ],
        "live block replay/Census reconciliation",
    )?;
    let gate = function_body(
        source,
        "symbol_tabulated_replay_reconciles_live_census_cg_16",
    )?;
    require_all(
        gate,
        &[
            "reconcile_case_with::<5>(CALIBRATION[6], facts)",
            "reconcile_control_with::<3, 3>(BLOCK_HOLDOUTS[0], facts)",
        ],
        "focused live reconciliation gate",
    )?;
    Ok(())
}

fn assert_mutation_plants() {
    let old_hash = LIVE_TABULATED_SOURCE.replacen(
        "hash = doubled + hash + C::index_of(code) as u128;",
        "hash = hash.wrapping_mul(1_099_511_628_211) + C::index_of(code) as u128;",
        1,
    );
    assert!(audit_live_tabulated_source(&old_hash).is_err());
    let reduced_seed = LIVE_TABULATED_SOURCE.replacen(
        "let mut hash = run.len() as u128;",
        "let mut hash = (run.len() % modulus) as u128;",
        1,
    );
    assert!(audit_live_tabulated_source(&reduced_seed).is_err());
    let masked_entry = LIVE_TABULATED_SOURCE.replacen("index % extent", "index & (extent - 1)", 1);
    assert!(audit_live_tabulated_source(&masked_entry).is_err());
    let unsplit = LIVE_TABULATED_SOURCE.replacen(
        "build_source_scalar::<E, Bd, C, L, Lg>(",
        "build_source_block::<E, Bd, C, L, Lg>(",
        1,
    );
    assert!(audit_live_tabulated_source(&unsplit).is_err());

    let leaked_selector = HARNESS_SOURCE.replacen(
        "structural\n        .table_basis()",
        "diagnostic\n        .table_basis()",
        1,
    );
    assert!(audit_selector_source(&leaked_selector).is_err());
    let selector = function_body(HARNESS_SOURCE, "candidate_route")
        .expect("the candidate selector exists for its mutation plant");
    assert_eq!(selector.matches("table_upper < dense_lower").count(), 1);
    let nonstrict_selector =
        selector.replacen("table_upper < dense_lower", "table_upper <= dense_lower", 1);
    let nonstrict_boundary = HARNESS_SOURCE.replacen(selector, &nonstrict_selector, 1);
    assert!(audit_selector_source(&nonstrict_boundary).is_err());
    let reversed_envelope =
        HARNESS_SOURCE.replacen(".fold(0.0, f64::max)", ".fold(0.0, f64::min)", 1);
    assert!(audit_selector_source(&reversed_envelope).is_err());
    let unguarded = HARNESS_SOURCE.replacen("CG16_FULL_BYTE_GUARD", "REMOVED_GUARD", 1);
    assert!(audit_timer_source(&unguarded).is_err());
    let unguarded_control =
        HARNESS_SOURCE.replacen("CG16_CONTROL_FULL_BYTE_GUARD", "REMOVED_CONTROL_GUARD", 1);
    assert!(audit_timer_source(&unguarded_control).is_err());
    let contaminated = HARNESS_SOURCE.replacen(
        "let start = Instant::now();",
        "let start = Instant::now(); verify_bits(output, expected);",
        1,
    );
    assert!(audit_timer_source(&contaminated).is_err());
    let observation = function_body(HARNESS_SOURCE, "print_observation")
        .expect("the observation printer exists for its mutation plant");
    let raw_call = "print_pair_samples(split, observation.case.id, &key, &observation.timing);";
    assert_eq!(observation.matches(raw_call).count(), 1);
    let omitted_observation = observation.replacen(raw_call, "let _ = (split, &key);", 1);
    let omitted_samples = HARNESS_SOURCE.replacen(observation, &omitted_observation, 1);
    assert!(audit_raw_sample_source(&omitted_samples).is_err());
    let reconciliation_call = ["reconcile_case_with::<5>(CALIBRATION[6], facts)", ";"].concat();
    let bypassed_reconciliation =
        HARNESS_SOURCE.replacen(&reconciliation_call, "let _ = (CALIBRATION[6], facts);", 1);
    assert!(audit_reconciliation_source(&bypassed_reconciliation).is_err());
}

fn assert_case_identities() {
    const CAL_IDS: [&str; 32] = [
        "C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09", "C10", "C11", "C12", "C13",
        "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21", "C22", "C23", "C24", "C25", "C26",
        "C27", "C28", "C29", "C30", "C31", "C32",
    ];
    const HOLDOUT_IDS: [&str; 14] = [
        "H01",
        "H02",
        "H03",
        "H04",
        "H05",
        "H06",
        "H07",
        "H08",
        "H09",
        "H10",
        "H11",
        "H12",
        "H13-B3-D3",
        "H14-B5-D5",
    ];
    assert_eq!(
        CALIBRATION.iter().map(|case| case.id).collect::<Vec<_>>(),
        CAL_IDS
    );
    assert!(CALIBRATION
        .iter()
        .all(|case| case.role == Role::Calibration));
    let mut holdout_ids: Vec<_> = HOLDOUT.iter().map(|case| case.id).collect();
    holdout_ids.extend(BLOCK_HOLDOUTS.iter().map(|case| case.id));
    assert_eq!(holdout_ids, HOLDOUT_IDS);
    assert!(HOLDOUT.iter().all(|case| case.role == Role::Holdout));
    assert_eq!(BLOCK_HOLDOUTS.len(), 2);
    assert!(BLOCK_HOLDOUTS
        .iter()
        .all(|case| !case.block.is_power_of_two() && !case.d.is_power_of_two()));
    let profiles: BTreeSet<_> = CALIBRATION
        .iter()
        .chain(HOLDOUT)
        .map(|case| case.values as u8)
        .collect();
    assert_eq!(profiles.len(), 5, "every q/tag profile is source-pinned");
    assert_eq!(
        atlas_coordinates(full_q(119, false)),
        [1, 1, -56, 1],
        "the compact witness realizes all four nonzero q coordinates"
    );
}

fn prepare_design(facts: ModelFacts) -> (Vec<PreparedCase>, Vec<PreparedCase>) {
    let calibration: Vec<_> = CALIBRATION
        .iter()
        .copied()
        .map(|case| prepare_case(case, facts))
        .collect();
    let holdout: Vec<_> = HOLDOUT
        .iter()
        .copied()
        .map(|case| prepare_case(case, facts))
        .collect();
    let calibration_design: Vec<_> = calibration
        .iter()
        .map(|prepared| (prepared.case, prepared.structural.clone()))
        .collect();
    let holdout_design: Vec<_> = holdout
        .iter()
        .map(|prepared| (prepared.case, prepared.structural.clone()))
        .collect();
    assert_design(&calibration_design, &holdout_design);
    assert_eq!(holdout[0].structural, holdout[1].structural);
    for &case in BLOCK_HOLDOUTS {
        match (case.block, case.d) {
            (3, 3) => {
                let fixture = ControlFixture::<3, 3>::new(case);
                let _ = control_offer::<3, 3>(fixture.shape);
            }
            (5, 5) => {
                let fixture = ControlFixture::<5, 5>::new(case);
                let _ = control_offer::<5, 5>(fixture.shape);
            }
            _ => unreachable!("the source-pinned controls are exhaustively instantiated"),
        }
    }
    (calibration, holdout)
}

fn central_prediction<const C: usize>(basis: [u128; C], fit: NonnegativeFit<C>) -> f64 {
    basis
        .iter()
        .zip(fit.coefficients)
        .map(|(&coordinate, coefficient)| coordinate as f64 * coefficient)
        .sum()
}

fn print_coefficients(model: &SelectorModel) {
    const TABLE_NAMES: [&str; TABLE_BASIS] = [
        "intercept",
        "columns",
        "hash_coordinates",
        "workspace_clear_words",
        "entry_attempt_ceiling",
        "activation_observations",
        "book_site_ceiling",
        "activation_projections",
        "row_source_sites",
        "tile_source_sites",
        "demand_build_ceiling",
        "resident_lane_words",
        "padding_words",
        "tile_block_presentations",
    ];
    const DENSE_NAMES: [&str; DENSE_BASIS] = [
        "intercept",
        "output_cells",
        "dense_decodes",
        "row_source_sites",
        "dense_calls",
        "dense_joins",
    ];
    println!("model,coordinate,central_s,lower95_s,upper95_s");
    for (name, coefficient) in TABLE_NAMES.into_iter().zip(model.table) {
        println!(
            "table,{name},{:.12e},{:.12e},{:.12e}",
            coefficient.central,
            coefficient.conservative_lower(),
            coefficient.conservative_upper()
        );
    }
    for (name, coefficient) in DENSE_NAMES.into_iter().zip(model.dense) {
        println!(
            "dense,{name},{:.12e},{:.12e},{:.12e}",
            coefficient.central,
            coefficient.conservative_lower(),
            coefficient.conservative_upper()
        );
    }
    println!(
        "fit,table,active={},rss={:.12e}",
        model.table_fit.active, model.table_fit.residual_sum_squares
    );
    println!(
        "fit,dense,active={},rss={:.12e}",
        model.dense_fit.active, model.dense_fit.residual_sum_squares
    );
}

fn structural_sample_key(structural: &StructuralWork) -> String {
    format!(
        "D{}:M{}:K{}:N{}:R{}:C{}:P{}:PANEL{}:EXACT{}:LANE{}:INDEX{}",
        structural.d,
        structural.shape.m,
        structural.shape.k,
        structural.shape.n,
        structural.plan.rows,
        structural.plan.cols,
        structural.plan.depth,
        structural.panel_offer,
        structural.exact_offer,
        structural.lane_offer,
        structural.index_offer,
    )
}

fn print_observation(observation: &Observation, prediction: Prediction, split: &str) {
    let diagnostic = &observation.diagnostic;
    let scale = diagnostic
        .call_scale
        .expect("a nonempty measured table has a q scale");
    println!(
        "{split},{},{},{},{},{},{},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
        observation.case.id,
        observation.case.d,
        observation.case.m,
        observation.case.k,
        observation.case.n,
        prediction.route.as_str(),
        observation.timing.table.mean,
        observation.timing.table.half_width,
        observation.timing.dense.mean,
        observation.timing.dense.half_width,
        observation.timing.ratio.mean,
        observation.timing.ratio.half_width,
        prediction.table_upper,
        prediction.dense_lower,
        diagnostic.column.distinct_columns,
        diagnostic.column.repeated_columns,
        diagnostic.column.hash_coordinates,
        diagnostic.call_entries.attempts,
        diagnostic.call_entries.full,
        diagnostic.build_entries.attempts,
        diagnostic.build_entries.full,
        diagnostic.scalar_envelope_calls,
        diagnostic.scalar_envelope_replays,
        diagnostic.safe_source_aggregates,
        diagnostic.fractured_source_aggregates,
        diagnostic.finite_tag_singletons,
        diagnostic.special_tag_singletons,
        diagnostic.capacity_flushes,
        diagnostic.singleton_placements,
        diagnostic.tail_placements,
        diagnostic.build_contractions,
        diagnostic.gather_adds,
        diagnostic.full_q_rectangles,
        observation.forced_table.adds,
        observation.forced_dense.kernel_calls,
        observation.timing.table_batch,
        observation.timing.dense_batch,
        scale.per_step,
    );
    let key = structural_sample_key(&observation.structural);
    print_pair_samples(split, observation.case.id, &key, &observation.timing);
}

#[test]
fn symbol_tabulated_selector_structure_and_plants_cg_16() {
    assert_case_identities();
    audit_live_tabulated_source(LIVE_TABULATED_SOURCE).expect("live q/table source audit");
    audit_selector_source(HARNESS_SOURCE).expect("selector value-blindness audit");
    audit_timer_source(HARNESS_SOURCE).expect("paired timer audit");
    audit_raw_sample_source(HARNESS_SOURCE).expect("raw paired sample audit");
    audit_reconciliation_source(HARNESS_SOURCE).expect("live reconciliation source audit");
    assert_mutation_plants();
    let facts = ModelFacts::load();
    let _ = prepare_design(facts);
}

#[test]
fn symbol_tabulated_replay_reconciles_live_census_cg_16() {
    audit_reconciliation_source(HARNESS_SOURCE).expect("live reconciliation source audit");
    let facts = ModelFacts::load();
    reconcile_case_with::<5>(CALIBRATION[6], facts);
    reconcile_control_with::<3, 3>(BLOCK_HOLDOUTS[0], facts);
}

#[test]
#[ignore = "paired release calibration and immutable CG-16 holdouts"]
#[allow(clippy::assertions_on_constants)]
fn symbol_tabulated_value_blind_boundary_cg_16() {
    assert!(!cfg!(debug_assertions), "CG-16 timing requires --release");
    assert_case_identities();
    audit_live_tabulated_source(LIVE_TABULATED_SOURCE).expect("live q/table source audit");
    audit_selector_source(HARNESS_SOURCE).expect("selector value-blindness audit");
    audit_timer_source(HARNESS_SOURCE).expect("paired timer audit");
    audit_raw_sample_source(HARNESS_SOURCE).expect("raw paired sample audit");
    audit_reconciliation_source(HARNESS_SOURCE).expect("live reconciliation source audit");
    assert_mutation_plants();
    let facts = ModelFacts::load();
    println!(
        "CG-16 hash_prefix={} q_ceiling={} q_tag_base={}",
        facts.hash_prefix, facts.compact_ceiling, facts.tag_base
    );

    // Every identity, structural key, split, and exact rank is resolved before
    // the first Instant is read. The observations below cannot edit this set.
    let (calibration, holdout) = prepare_design(facts);
    let calibration_observations: Vec<_> = calibration
        .iter()
        .map(|prepared| measure_case(prepared, facts))
        .collect();
    let envelopes = calibration_envelopes(&calibration_observations);
    let model = fit_selector(&envelopes);
    print_coefficients(&model);
    println!("calibration_key,ids,table_target_s,table_fit_s,dense_target_s,dense_fit_s");
    for envelope in &envelopes {
        let key = structural_sample_key(&envelope.structural);
        println!(
            "{},{},{:.12e},{:.12e},{:.12e},{:.12e}",
            key,
            envelope.ids.join("+"),
            envelope.table_target,
            central_prediction(envelope.structural.table_basis(), model.table_fit),
            envelope.dense_target,
            central_prediction(envelope.structural.dense_basis(), model.dense_fit),
        );
    }

    let holdout_observations: Vec<_> = holdout
        .iter()
        .map(|prepared| measure_case(prepared, facts))
        .collect();
    println!("sample_kind,split,id,key,round,route,seconds");
    println!("split,id,D,m,k,n,predicted,table_mean_s,table_ci_s,dense_mean_s,dense_ci_s,ratio,ratio_ci_s,table_upper_prediction_s,dense_lower_prediction_s,distinct_columns,repeated_columns,hash_coordinates,call_entry_attempts,call_entry_full,build_entry_attempts,build_entry_full,scalar_calls,scalar_replays,safe_aggregates,fractured_aggregates,finite_tags,special_tags,capacity_flushes,singletons,tail_placements,build_contractions,gather_adds,full_q_rectangles,table_adds,dense_calls,table_batch,dense_batch,call_per_step");
    for observation in &calibration_observations {
        print_observation(
            observation,
            candidate_route(&observation.structural, &model),
            "CAL",
        );
    }
    let mut table_side = 0usize;
    let mut dense_side = 0usize;
    for observation in &holdout_observations {
        let prediction = candidate_route(&observation.structural, &model);
        match prediction.route {
            Route::Table => table_side += 1,
            Route::Dense => dense_side += 1,
        }
        print_observation(observation, prediction, "HOLDOUT");
    }
    println!("HOLDOUT_ROUTE_COUNTS,table={table_side},dense={dense_side}");
    assert_eq!(
        candidate_route(&holdout_observations[0].structural, &model).route,
        candidate_route(&holdout_observations[1].structural, &model).route,
        "H01/H02 cannot diverge through their values"
    );

    for &case in BLOCK_HOLDOUTS {
        let observation = measure_control(case, facts);
        let key = format!(
            "B{}:D{}:M{}:BLOCKS{}:N{}",
            case.block, case.d, case.m, case.blocks, case.n,
        );
        print_pair_samples("CONTROL", case.id, &key, &observation.timing);
        println!(
            "CONTROL,{},{},{},{},{},{},{},{},{},{},{},{},{:.9e},{:.9e},{:.9e},{:.9e}",
            observation.diagnostic.id,
            case.block,
            case.d,
            observation.diagnostic.scalar_calls,
            observation.diagnostic.scalar_replays,
            observation.diagnostic.fractured_aggregates,
            observation.diagnostic.safe_scalar_atoms,
            observation.diagnostic.capacity_flushes,
            observation.diagnostic.singleton_placements,
            observation.diagnostic.tail_placements,
            observation.diagnostic.build_contractions,
            observation.diagnostic.gather_adds,
            observation.timing.table.mean,
            observation.timing.dense.mean,
            observation.timing.ratio.mean,
            observation.timing.ratio.half_width,
        );
    }
}
