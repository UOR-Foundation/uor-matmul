//! Pure-UOR float differentials and structural controls.
//!
//! Both IEEE widths walk the same cases. The unoptimized exact traversal is
//! the byte oracle; panel offers and the complete caller-owned workspace are
//! factorizations of that value and must agree under every encode mode.

use uor_matmul::prelude::*;
use uor_matmul::{
    suggested_accumulators, suggested_bridge_scaled, suggested_float_panels, suggested_scratch,
};
use uor_matmul_core::{
    as_alphabet_full, AccOf, Alphabet, EncodeFrom, EncodeMode, Full, PackedCode, Shape,
};
use uor_matmul_gemm::epilogue::{AbsorbPrior, ScaleExact};
use uor_matmul_gemm::{PlaceAt, SignedPlace};
use uor_matmul_validate::bytes_equal;
use uor_matmul_validate::float_corpus::{
    exact_product, operands, CorpusFloat, FloatCase, CORRECTNESS_CASES,
};

fn reference<E>(case: FloatCase, a: &[E], b: &[E], mode: EncodeMode) -> Vec<E>
where
    E: CorpusFloat + EncodeFrom<AccOf<E>>,
{
    exact_product(case, a, b, mode)
}

#[allow(clippy::too_many_arguments)]
fn packed<E>(
    case: FloatCase,
    a: &[E],
    b: &[E],
    mode: EncodeMode,
    pa_len: usize,
    pb_len: usize,
) -> Vec<E>
where
    E: CorpusFloat + EncodeFrom<AccOf<E>> + EncodeFrom<i128>,
    AccOf<E>: SignedPlace + PlaceAt + ScaleExact + AbsorbPrior<E>,
{
    let mut c = vec![E::ZERO; case.m * case.n];
    let mut pa = vec![PackedCode::default(); pa_len];
    let mut pb = vec![PackedCode::default(); pb_len];
    let av = MatView::row_major(a, case.m, case.k).expect("A fits");
    let bv = MatView::row_major(b, case.k, case.n).expect("B fits");
    let cv = MatViewMut::row_major(&mut c, case.m, case.n).expect("C fits");
    let mut triple = Triple::new(av, bv, cv).expect("the product exists");
    uor_matmul::gemm_float_packed(
        &mut triple,
        &Linear::OVERWRITE,
        GemmOptions {
            encode: mode,
            ..Default::default()
        },
        &mut pa,
        &mut pb,
    );
    c
}

fn full<E>(case: FloatCase, a: &[E], b: &[E], mode: EncodeMode) -> Vec<E>
where
    E: CorpusFloat + EncodeFrom<AccOf<E>> + EncodeFrom<i128>,
    AccOf<E>: SignedPlace + PlaceAt + ScaleExact + AbsorbPrior<E>,
{
    let shape = Shape {
        m: case.m,
        k: case.k,
        n: case.n,
    };
    let (pa_len, pb_len) = suggested_float_panels(shape);
    let mut pa = vec![PackedCode::default(); pa_len];
    let mut pb = vec![PackedCode::default(); pb_len];
    let mut scaled = vec![0i32; suggested_bridge_scaled(shape)];
    let mut panels = vec![Alphabet::<i32, Full<i32>>::ZERO; suggested_scratch(shape)];
    let mut accumulators = vec![0i128; suggested_accumulators(shape)];
    let mut c = vec![E::ZERO; case.m * case.n];
    let av = MatView::row_major(a, case.m, case.k).expect("A fits");
    let bv = MatView::row_major(b, case.k, case.n).expect("B fits");
    let cv = MatViewMut::row_major(&mut c, case.m, case.n).expect("C fits");
    let mut triple = Triple::new(av, bv, cv).expect("the product exists");
    uor_matmul::gemm_float_bridged(
        &mut triple,
        &Linear::OVERWRITE,
        GemmOptions {
            encode: mode,
            ..Default::default()
        },
        &mut pa,
        &mut pb,
        &mut scaled,
        &mut Scratch::with_accumulators(&mut panels, &mut accumulators),
    );
    c
}

fn exercise<E>()
where
    E: CorpusFloat + EncodeFrom<AccOf<E>> + EncodeFrom<i128>,
    AccOf<E>: SignedPlace + PlaceAt + ScaleExact + AbsorbPrior<E>,
{
    for &case in CORRECTNESS_CASES {
        let (a, b) = operands::<E>(case);
        let shape = Shape {
            m: case.m,
            k: case.k,
            n: case.n,
        };
        let suggested = suggested_float_panels(shape);
        let offers = [
            (0, 0),
            (case.k.saturating_sub(1), case.k.saturating_sub(1)),
            (case.k, case.k),
            suggested,
        ];
        for mode in [
            EncodeMode::Nearest,
            EncodeMode::TowardZero,
            EncodeMode::Saturating,
            EncodeMode::Wrapping,
        ] {
            let want = reference(case, &a, &b, mode);
            for (pa, pb) in offers {
                let got = packed(case, &a, &b, mode, pa, pb);
                bytes_equal(&got, &want).unwrap_or_else(|error| {
                    panic!(
                        "{:?} {:?} {}x{}x{}, offer {pa}/{pb}: {error}",
                        core::any::type_name::<E>(),
                        case.fill,
                        case.m,
                        case.k,
                        case.n,
                    )
                });
            }
            let got = full(case, &a, &b, mode);
            bytes_equal(&got, &want).unwrap_or_else(|error| {
                panic!(
                    "{:?} {:?} {}x{}x{}, full offer: {error}",
                    core::any::type_name::<E>(),
                    case.fill,
                    case.m,
                    case.k,
                    case.n,
                )
            });
        }
    }
}

/// `CD-30`: both IEEE widths, every structural fill, offer, and encode mode
/// produce the exact reference's bytes.
#[test]
fn pure_uor_float_routes_equal_the_exact_reference_cd_30() {
    exercise::<f32>();
    exercise::<f64>();
}

/// `CG-22`: the float work must not perturb the canonical integer route.
///
/// The route is read from the execution ledger, not inferred from elapsed time
/// or recomputed from the selection predicate.
#[test]
fn the_float_refactor_leaves_the_integer_route_census_unchanged_cg_22() {
    let (m, k, n) = (8usize, 33usize, 17usize);
    let a = vec![1i8; m * k];
    let b = vec![1i8; k * n];
    let mut c = vec![0i32; m * n];
    let mut panels = vec![Alphabet::<i8, Full<i8>>::ZERO; suggested_scratch(Shape { m, k, n })];
    let mut census = uor_matmul::RouteCensus::default();
    let av = MatView::row_major(as_alphabet_full(&a), m, k).expect("A fits");
    let bv = MatView::row_major(as_alphabet_full(&b), k, n).expect("B fits");
    let cv = MatViewMut::row_major(&mut c, m, n).expect("C fits");
    let mut triple = Triple::new(av, bv, cv).expect("the product exists");
    uor_matmul::gemm_auto_counted(
        &mut triple,
        &Linear::OVERWRITE,
        GemmOptions::default(),
        &mut Scratch::new(&mut panels),
        &mut census,
    );
    assert!(
        matches!(census.route, Some(uor_matmul::Route::Kernel { .. })),
        "the suggested offer must still reach the integer kernel table: {census:?}"
    );
    assert!(c.iter().all(|&value| value == k as i32));
}
