//! The declaration, layout, and borrow-only half of the public float API lock.
//!
//! The general API baseline records paths and declaration kinds.  This focused
//! witness holds the narrower compatibility facts that a name-only source scan
//! cannot see.  It deliberately says nothing about standalone behaviour of the
//! contextual q producer or consumer: only their public types and their paired
//! caller-storage representation are under test here.

use core::mem::{align_of, size_of};
use std::path::PathBuf;

use uor_matmul::codec::{Arena, CodedMatrix};
use uor_matmul::core_types::Whole;
use uor_matmul::driver::tabulated::Steps;
use uor_matmul::driver::{Census, LaneScale, Tabulated, Wide};
use uor_matmul::kernels::{Lane, LaneWord, Scaled64, TableSpec};
use uor_matmul::{
    slice, AccOf, Alphabet, Backend, GemmOptions, Linear, MatView, MatViewMut, NotAProduct,
    PackedCode, Shape,
};

type F32Acc = AccOf<f32>;
type F32Bound = Whole<f32>;
type F32Cell = Alphabet<f32, F32Bound>;
type F32Book = Arena<'static, f32, 1, u8>;

trait Same<T> {}

impl<T> Same<T> for T {}

fn require_same<T, U>()
where
    T: Same<U>,
{
}

fn compact_code(source: &str) -> String {
    source
        .lines()
        .map(|line| line.split_once("//").map_or(line, |(code, _)| code))
        .flat_map(str::split_whitespace)
        .collect()
}

fn has_scaled64_declaration(source: &str) -> bool {
    compact_code(source).contains("#[repr(transparent)]pubstructScaled64(pubi64);")
}

type LanesFn = for<'s> fn(&'s mut [i64], &'s mut [F32Acc], usize) -> Option<&'s mut [Scaled64]>;

type LaneScaleFn = for<'data, 'a, 'codes, 'w, 'ledger> fn(
    &'a MatView<'data, F32Cell>,
    &'w CodedMatrix<'codes, f32, F32Bound, F32Book>,
    &'ledger mut Census,
) -> Option<LaneScale>;

type DenseFn = for<'a, 'b, 'c, 'epilogue, 'rest> fn(
    MatView<'a, F32Cell>,
    MatView<'b, F32Cell>,
    MatViewMut<'c, f32>,
    &'epilogue Linear,
    GemmOptions,
    &'rest mut [F32Cell],
) -> bool;

type DistinctRowsFn = for<'data, 'view, 'index> fn(
    &'view MatView<'data, F32Cell>,
    &'index mut [usize],
) -> Option<usize>;

type FloatFullFn =
    for<'a, 'b, 'c, 'pa, 'pb, 'scaled, 'panels, 'accumulators> fn(
        usize,
        usize,
        usize,
        &'a [f32],
        &'b [f32],
        &'c mut [f32],
        &'pa mut [PackedCode],
        &'pb mut [PackedCode],
        &'scaled mut [i32],
        &'panels mut [i32],
        &'accumulators mut [i128],
    ) -> Result<(), NotAProduct>;

type FloatExFullFn =
    for<'a, 'b, 'c, 'pa, 'pb, 'scaled, 'panels, 'accumulators> fn(
        usize,
        usize,
        usize,
        i64,
        &'a [f32],
        usize,
        &'b [f32],
        usize,
        i64,
        &'c mut [f32],
        usize,
        &'pa mut [PackedCode],
        &'pb mut [PackedCode],
        &'scaled mut [i32],
        &'panels mut [i32],
        &'accumulators mut [i128],
    ) -> Result<(), NotAProduct>;

/// `CA-05`: the established f32 table hooks retain their declarations and
/// relabel the caller's lane words without changing address, extent, or bytes.
#[test]
fn the_public_f32_carrier_is_layout_and_borrow_compatible_ca_05() {
    // The source predicate has independent teeth for the representation and
    // the public tuple field before it is applied to the live declaration.
    let fixture = "#[repr(transparent)] pub struct Scaled64(pub i64);";
    assert!(has_scaled64_declaration(fixture));
    assert!(!has_scaled64_declaration("pub struct Scaled64(pub i64);"));
    assert!(!has_scaled64_declaration(
        "#[repr(transparent)] pub struct Scaled64(i64);"
    ));
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("the conformance crate is two levels below the repository root")
        .to_path_buf();
    let table_source = std::fs::read_to_string(root.join("crates/uor-matmul-kernels/src/table.rs"))
        .expect("the shipped lane source reads");
    assert!(
        has_scaled64_declaration(&table_source),
        "Scaled64 must remain exactly #[repr(transparent)] over one public i64 field"
    );

    assert_eq!(size_of::<Scaled64>(), size_of::<i64>());
    assert_eq!(align_of::<Scaled64>(), align_of::<i64>());
    assert_eq!(<f32 as Tabulated>::LANE_BYTES, size_of::<i64>());
    const { assert!(!<f32 as Tabulated>::LANE_IS_EXACT) };

    require_same::<<f32 as Tabulated>::Lane, Scaled64>();
    require_same::<<f32 as Tabulated>::ModLane, Scaled64>();
    require_same::<<f32 as Tabulated>::StreamLane, Wide<F32Acc>>();

    let _: for<'a> fn(&'a mut [i64]) -> &'a mut [Scaled64] = Scaled64::wrap_i64s_mut;
    let _: Scaled64 = <Scaled64 as LaneWord>::ZERO;
    let _: fn(Scaled64, Scaled64) -> Scaled64 = <Scaled64 as LaneWord>::add;
    let _: fn(u128) -> Option<usize> = <Scaled64 as Lane<f32>>::capacity;
    let _: fn(Scaled64, f32, f32) -> Scaled64 = <Scaled64 as Lane<f32>>::mac;
    let _: fn(Scaled64, F32Acc) -> F32Acc = <Scaled64 as Lane<f32>>::place;
    let _: fn(Scaled64, F32Acc, i32) -> F32Acc = <Scaled64 as Lane<f32>>::place_scaled;

    let _: fn(u32) -> bool = <f32 as Tabulated>::modular_table_admitted;
    let _: fn(Backend, u128, bool, usize, usize, usize) -> TableSpec<f32, Scaled64> =
        <f32 as Tabulated>::table_spec;
    let _: fn(Backend, u128, usize, usize, usize) -> TableSpec<f32, Scaled64> =
        <f32 as Tabulated>::table_spec_modular;
    let _: LanesFn = <f32 as Tabulated>::lanes;
    let _: fn(Backend, u128, usize, usize) -> Steps = <f32 as Tabulated>::dense_steps;
    let _: DenseFn = <f32 as Tabulated>::dense_gemm::<F32Bound, f32, Linear>;
    let _: LanesFn = <f32 as Tabulated>::lanes_modular;
    let _: fn(u128) -> Option<usize> = <f32 as Tabulated>::probe_capacity::<Scaled64>;
    let _: LaneScaleFn = <f32 as Tabulated>::lane_scale::<F32Bound, F32Book, Census>;
    let _: fn(u128, &LaneScale) -> Option<usize> = <f32 as Tabulated>::lane_run::<Scaled64>;
    let _: fn(f32, i32) -> f32 = <f32 as Tabulated>::prescale;
    let _: DistinctRowsFn = <f32 as Tabulated>::distinct_a_rows::<F32Bound>;

    // The historical complete-workspace spelling remains callable with the
    // established caller buffers.  These are type checks only: workspace
    // residue and contextual q cells intentionally have no standalone value.
    let _: FloatFullFn = slice::gemm_float_full::<f32, f32>;
    let _: FloatExFullFn = slice::gemm_float_ex_full::<f32, f32>;
    let _: fn(Shape) -> (usize, usize) = uor_matmul::suggested_float_panels;
    let _: fn(Shape) -> usize = uor_matmul::suggested_bridge_scaled;
    let _: fn(Shape) -> usize = uor_matmul::suggested_scratch;
    let _: fn(Shape) -> usize = uor_matmul::suggested_accumulators;

    let mut words = [i64::MIN, -7, 0, 11, i64::MAX];
    let word_ptr = words.as_mut_ptr();
    {
        let lanes = Scaled64::wrap_i64s_mut(&mut words);
        assert_eq!(lanes.len(), 5);
        assert_eq!(lanes.as_mut_ptr().cast::<i64>(), word_ptr);
        assert_eq!(lanes[1].0, -7);
        lanes[1].0 = 0x1357_2468_1357_2468;
    }
    assert_eq!(words[1], 0x1357_2468_1357_2468);

    let word_ptr = words.as_mut_ptr();
    let mut exact: [F32Acc; 0] = [];
    {
        let lanes = <f32 as Tabulated>::lanes(&mut words, &mut exact, 3)
            .expect("three caller words hold three transparent lanes");
        assert_eq!(lanes.len(), 3);
        assert_eq!(lanes.as_mut_ptr().cast::<i64>(), word_ptr);
        lanes[2].0 = -0x2468_1357_2468_1357;
    }
    assert_eq!(words[2], -0x2468_1357_2468_1357);
}
