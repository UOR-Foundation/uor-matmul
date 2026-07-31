//! The workspace queries and the two-offer slice face.
//!
//! No new conformance ID: the claims here are the driver's existing offer
//! arithmetic made visible. That the bytes do not move at any offer is
//! `CD-04`/`CD-10` --- the ladders below exercise the same claim through the
//! new spelling, not a new one --- and that the accumulator block is
//! `k`-independent is a property of `suggested_accumulators`, which Phase B
//! does not change, only reports. A query that *lies* about the driver's
//! arithmetic is caught by comparing the report against the existing queries
//! and against a run through the route census, which is what the tests below
//! do rather than assert a new behavioral claim.

use core::mem::size_of;

use uor_matmul::{
    as_alphabet_full, gemm_auto_counted, minimum_workspace, slice, suggested_accumulators,
    suggested_scratch, workspace_for_budget, workspace_report, AccOf, Backend, Chunking, Full,
    GemmOptions, IntegerElement, Kernelized, Linear, MatView, MatViewMut, Route, RouteCensus,
    Scratch, Shape, Triple,
};

/// Deterministic operands inside the full `i8` alphabet's bound.
fn operands(m: usize, k: usize, n: usize) -> (Vec<i8>, Vec<i8>) {
    let a: Vec<i8> = (0..m * k).map(|i| ((i * 37) % 127) as i8).collect();
    let b: Vec<i8> = (0..k * n).map(|i| ((i * 53) % 127) as i8).collect();
    (a, b)
}

/// The reference traversal's answer, from the entry that stays directly
/// callable under its own name (R6).
fn reference(m: usize, k: usize, n: usize, a: &[i8], b: &[i8]) -> Vec<i32> {
    let mut c = vec![0i32; m * n];
    let av = MatView::row_major(as_alphabet_full(a), m, k).expect("A fits");
    let bv = MatView::row_major(as_alphabet_full(b), k, n).expect("B fits");
    let cv = MatViewMut::row_major(&mut c, m, n).expect("C fits");
    let mut t = Triple::new(av, bv, cv).expect("the product exists");
    uor_matmul::driver::gemm(
        &mut t,
        &Linear::OVERWRITE,
        GemmOptions::default(),
        &mut Scratch::none(),
    );
    c
}

/// The bounded plan and the accumulator query do not grow with `k`.
#[test]
fn the_bounded_plan_does_not_grow_with_k() {
    let at = |k| Shape { m: 16, k, n: 16 };
    let shallow = workspace_report::<i8, Full<i8>>(at(4096));
    let deep = workspace_report::<i8, Full<i8>>(at(1_048_576));

    assert_eq!(
        suggested_accumulators(at(4096)),
        suggested_accumulators(at(1_048_576)),
        "the accumulator block is k-independent"
    );
    assert_eq!(
        shallow.bounded, deep.bounded,
        "the bounded plan is k-independent, in every term"
    );
    // The report's accumulator term is the existing query's count, in bytes.
    assert_eq!(
        deep.bounded.accumulators,
        suggested_accumulators(at(1_048_576)) * size_of::<AccOf<i8>>(),
        "the report is the same terms the driver reads, not restated numerals"
    );
    assert!(
        matches!(deep.bounded.chunking, Chunking::Chunked { .. }),
        "a k past KC chunks under the bounded plan, got {:?}",
        deep.bounded.chunking
    );

    // The teeth: the suggested full-depth plan *does* grow with k, so the
    // equality above is not two large numbers both reading "big".
    assert!(
        deep.suggested.panels > shallow.suggested.panels,
        "the full-depth plan grows with k"
    );
    // And where k fits the lane and the chunk, the two plans coincide.
    let fits = workspace_report::<i8, Full<i8>>(Shape {
        m: 64,
        k: 64,
        n: 64,
    });
    assert_eq!(
        fits.suggested, fits.bounded,
        "at k <= KC the bounded plan is the suggested one"
    );
    assert_eq!(fits.suggested.chunking, Chunking::FullDepth);
}

/// The report's terms are the driver's own declarations, and the budget
/// ladder lands the named plans.
#[test]
fn the_report_is_built_from_the_drivers_own_terms() {
    let shape = Shape {
        m: 64,
        k: 4096,
        n: 64,
    };
    let report = workspace_report::<i8, Full<i8>>(shape);
    assert_eq!(report.suggested.panels, suggested_scratch(shape));
    assert_eq!(
        report.suggested.accumulators,
        suggested_accumulators(shape) * size_of::<AccOf<i8>>()
    );
    assert_eq!(
        report.suggested.total,
        report.suggested.panels + report.suggested.accumulators
    );

    // The minimum is one packed group of the tile the host resolves, read off
    // the same spec the query resolves.
    let spec = <i8 as Kernelized>::exact_spec(Backend::Auto, <i8 as IntegerElement>::FULL, shape.m);
    let minimum = minimum_workspace::<i8, Full<i8>>(shape);
    assert_eq!(
        minimum.panels,
        (spec.mr + spec.nr) * spec.k_group.max(1),
        "one packed group, in `i8` bytes"
    );
    assert_eq!(minimum.accumulators, 0);

    // The budget ladder: nothing, one group, the bounded plan, the suggested
    // plan. A budget between two rungs buys the lower one.
    let deep = Shape {
        m: 16,
        k: 262_144,
        n: 16,
    };
    let report = workspace_report::<i8, Full<i8>>(deep);
    let minimum = minimum_workspace::<i8, Full<i8>>(deep);
    assert_eq!(
        workspace_for_budget::<i8, Full<i8>>(deep, 0).chunking,
        Chunking::Streaming
    );
    assert_eq!(
        workspace_for_budget::<i8, Full<i8>>(deep, minimum.total),
        minimum
    );
    assert_eq!(
        workspace_for_budget::<i8, Full<i8>>(deep, report.bounded.total),
        report.bounded
    );
    assert_eq!(
        workspace_for_budget::<i8, Full<i8>>(deep, report.suggested.total),
        report.suggested
    );
    assert!(
        matches!(report.bounded.chunking, Chunking::Chunked { .. }),
        "the bounded rung at huge k is the chunked traversal"
    );
}

/// The byte-identity ladder through the two-offer slice face, at a k past the
/// lane: none, panel-only, panel plus one short of the accumulator block,
/// panel plus the block, and the bounded plan exactly.
#[test]
fn every_offer_ladder_rung_gives_the_same_bytes_at_huge_k() {
    let (m, k, n) = (16usize, 262_144usize, 16usize);
    let shape = Shape { m, k, n };
    let (a, b) = operands(m, k, n);
    let want = reference(m, k, n, &a, &b);

    let acc = suggested_accumulators(shape);
    let report = workspace_report::<i8, Full<i8>>(shape);
    assert!(acc > 0, "a deep shape suggests an accumulator block");

    let rungs: Vec<(usize, usize)> = vec![
        (0, 0),                              // no offer
        (suggested_scratch(shape), 0),       // panels only
        (suggested_scratch(shape), acc - 1), // one short of the block
        (suggested_scratch(shape), acc),     // the suggested pair
        (
            report.bounded.panels,
            report.bounded.accumulators / size_of::<AccOf<i8>>(),
        ), // the bounded plan
    ];
    for (panels, accs) in rungs {
        let mut panels_buf = vec![0i8; panels];
        let mut acc_buf = vec![AccOf::<i8>::default(); accs];
        let mut c = vec![0i32; m * n];
        slice::gemm_full(m, k, n, &a, &b, &mut c, &mut panels_buf, &mut acc_buf)
            .expect("the product exists");
        assert_eq!(c, want, "panels {panels}, accumulators {accs}");
    }
}

/// The chunked path, witnessed: at a k past the lane the bounded offer runs a
/// table kernel --- the route census says so --- and the report says the plan
/// chunks; the bytes are the reference's.
#[test]
fn a_bounded_offer_at_huge_k_chunks_and_matches_the_reference() {
    let (m, k, n) = (16usize, 262_144usize, 16usize);
    let shape = Shape { m, k, n };
    let (a, b) = operands(m, k, n);
    let report = workspace_report::<i8, Full<i8>>(shape);
    assert!(
        matches!(report.bounded.chunking, Chunking::Chunked { .. }),
        "the bounded plan chunks at a k past the lane, got {:?}",
        report.bounded.chunking
    );
    assert!(
        matches!(report.suggested.chunking, Chunking::Chunked { .. }),
        "at a k past the lane even the suggested offer chunks, got {:?}",
        report.suggested.chunking
    );

    let mut panels = vec![0i8; report.bounded.panels];
    let mut accs =
        vec![AccOf::<i8>::default(); report.bounded.accumulators / size_of::<AccOf<i8>>()];
    let mut c = vec![0i32; m * n];
    let mut census = RouteCensus::default();
    {
        let av = MatView::row_major(as_alphabet_full(&a), m, k).expect("A fits");
        let bv = MatView::row_major(as_alphabet_full(&b), k, n).expect("B fits");
        let cv = MatViewMut::row_major(&mut c, m, n).expect("C fits");
        let mut t = Triple::new(av, bv, cv).expect("the product exists");
        let mut scratch = Scratch::with_accumulators(
            uor_matmul::core_types::as_alphabet_full_mut(&mut panels),
            &mut accs,
        );
        gemm_auto_counted(
            &mut t,
            &Linear::OVERWRITE,
            GemmOptions::default(),
            &mut scratch,
            &mut census,
        );
    }
    assert!(
        matches!(census.route, Some(Route::Kernel { .. })),
        "the bounded offer runs a table kernel, got {:?}",
        census.route
    );
    assert_eq!(c, reference(m, k, n, &a, &b));
}
