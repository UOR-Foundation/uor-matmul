//! `CD-22`: the documented default integer entry point reaches the kernel
//! table at the offer a caller who follows `suggested_scratch` makes, and
//! every route from it --- table kernel, recursion, streaming reference ---
//! returns the reference traversal's bytes, at every offer including none.
//!
//! The route is read off the route census, never inferred from a clock: the
//! counted twin of the same function the facade calls records which
//! factorization ran, and selection reads declarations only (R13), so the
//! route the census records for a shape and offer is the route the public call
//! took for them. The one thing the census cannot see is that the facade calls
//! that function at all, which is what the sentinel-offer half of the first
//! test pins: the packed traversal packs its panels into the offer and the
//! reference traversal never touches it, so a written offer is the public
//! call's own fingerprint.

use uor_matmul::kernels::cached;
use uor_matmul::{
    as_alphabet_full, gemm_auto_counted, slice, suggested_scratch, Backend, GemmOptions,
    IntegerElement, Linear, MatView, MatViewMut, Route, RouteCensus, Scratch, Shape, Triple,
};

/// The square shape the kernel-table assertions run at: large enough that the
/// cost model prefers packing, small enough to run everywhere.
const N: usize = 64;

/// Deterministic operands inside the full `i8` alphabet's bound.
fn operands() -> (Vec<i8>, Vec<i8>) {
    let a: Vec<i8> = (0..N * N).map(|i| ((i * 37) % 127) as i8).collect();
    let b: Vec<i8> = (0..N * N).map(|i| ((i * 53) % 127) as i8).collect();
    (a, b)
}

/// The reference traversal's answer, from the entry that stays directly
/// callable under its own name (R6).
fn reference(a: &[i8], b: &[i8]) -> Vec<i32> {
    let mut c = vec![0i32; N * N];
    let av = MatView::row_major(as_alphabet_full(a), N, N).expect("A fits");
    let bv = MatView::row_major(as_alphabet_full(b), N, N).expect("B fits");
    let cv = MatViewMut::row_major(&mut c, N, N).expect("C fits");
    let mut t = Triple::new(av, bv, cv).expect("the product exists");
    uor_matmul::driver::gemm(
        &mut t,
        &Linear::OVERWRITE,
        GemmOptions::default(),
        &mut Scratch::none(),
    );
    c
}

/// The auto-selecting entry the facade calls, in its counted form, at `N`
/// cubed with `offer` as the panel offer.
fn counted_auto(offer: &mut [i8], a: &[i8], b: &[i8]) -> (Vec<i32>, RouteCensus) {
    let mut c = vec![0i32; N * N];
    let mut census = RouteCensus::default();
    {
        let av = MatView::row_major(as_alphabet_full(a), N, N).expect("A fits");
        let bv = MatView::row_major(as_alphabet_full(b), N, N).expect("B fits");
        let cv = MatViewMut::row_major(&mut c, N, N).expect("C fits");
        let mut t = Triple::new(av, bv, cv).expect("the product exists");
        let mut scratch = Scratch::new(uor_matmul::core_types::as_alphabet_full_mut(offer));
        gemm_auto_counted(
            &mut t,
            &Linear::OVERWRITE,
            GemmOptions::default(),
            &mut scratch,
            &mut census,
        );
    }
    (c, census)
}

/// CD-22, the selection half: at the suggested offer the public call runs a
/// kernel from the table --- a non-portable one where the host declares one ---
/// and its bytes are the reference's bytes.
#[test]
fn the_default_entry_runs_a_host_kernel_and_returns_the_reference_bytes_cd_22() {
    // The non-portable sequences this host resolves for the full `i8`
    // alphabet, read off the same cached list the driver selects from. On a
    // portable-only host there is nothing to select and the test skips.
    let full = <i8 as IntegerElement>::FULL;
    let host: Vec<Backend> = cached::available_i8()
        .filter(|s| s.backend != Backend::Portable && s.max_bound >= full)
        .map(|s| s.backend)
        .collect();
    if host.is_empty() {
        eprintln!("CD-22: a portable-only host declares nothing to select; skipping");
        return;
    }

    let (a, b) = operands();
    let suggested = suggested_scratch(Shape { m: N, k: N, n: N });

    // The documented call, exactly as the README spells it, over an offer
    // filled with a sentinel: the packed traversal packs its panels into the
    // offer, the reference traversal never writes it, so a changed byte is the
    // packed traversal's fingerprint on the public call itself.
    const SENTINEL: i8 = 0x5A;
    let mut scratch = vec![SENTINEL; suggested];
    let mut c = vec![0i32; N * N];
    slice::gemm(N, N, N, &a, &b, &mut c, &mut scratch).expect("the product exists");
    assert!(
        scratch.iter().any(|&x| x != SENTINEL),
        "the suggested offer was never written: the public call did not run the packed traversal"
    );

    // The route the same shape and offer take through the counted twin.
    let mut offer = vec![0i8; suggested];
    let (counted, census) = counted_auto(&mut offer, &a, &b);
    match census.route {
        Some(Route::Kernel { backend, .. }) => assert!(
            host.contains(&backend),
            "the route ran {backend:?}, not one of the host's non-portable sequences {host:?}"
        ),
        route => panic!("the suggested offer at {N}^3 must run a table kernel, got {route:?}"),
    }

    // Every route returns the reference's bytes.
    assert_eq!(counted, reference(&a, &b));
    assert_eq!(c, counted);
}

/// CD-22, the decline half: no offer runs the reference traversal --- through
/// the same selecting entry, not around it --- at the same bytes.
#[test]
fn zero_scratch_runs_the_reference_at_the_same_bytes() {
    let (a, b) = operands();

    let (streamed, census) = counted_auto(&mut [], &a, &b);
    assert!(
        matches!(
            census.route,
            Some(Route::ReferenceByOffer | Route::ReferenceByCost)
        ),
        "no offer must run the reference traversal, got {:?}",
        census.route
    );
    assert_eq!(streamed, reference(&a, &b));

    // And the public call agrees at the same offer.
    let mut c = vec![0i32; N * N];
    slice::gemm(N, N, N, &a, &b, &mut c, &mut []).expect("the product exists");
    assert_eq!(c, streamed);
}

/// CD-22 at every rung of the ladder: none, one element, one short of the
/// suggestion, the suggestion. The offer decides which factorization runs and
/// never what the bytes are.
#[test]
fn every_offer_from_none_to_suggested_gives_the_same_bytes() {
    let (a, b) = operands();
    let suggested = suggested_scratch(Shape { m: N, k: N, n: N });
    assert!(
        suggested > 0,
        "a non-degenerate shape suggests some scratch"
    );
    let want = reference(&a, &b);
    for offer in [0usize, 1, suggested - 1, suggested] {
        let mut scratch = vec![0i8; offer];
        let mut c = vec![0i32; N * N];
        slice::gemm(N, N, N, &a, &b, &mut c, &mut scratch).expect("the product exists");
        assert_eq!(c, want, "offer {offer} (suggested {suggested})");
    }
}
