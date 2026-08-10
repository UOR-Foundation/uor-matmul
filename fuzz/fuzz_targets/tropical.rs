//! `CT-08`, `CD-25`: the selection half, over unstructured input.
//!
//! Two claims at once, because the same call answers both.
//!
//! Totality first: no tropical operation overflows or fails, at any shape, any
//! stride and any depth, including lanes at the semiring zero. `Trop` is a
//! distinct type rather than a reserved pattern inside its element precisely so
//! that a masked lane performs *no arithmetic at all*, and this target is what
//! puts that in front of a fuzzer instead of in front of a reader.
//!
//! Then the witness: both mechanisms run on the same operands and their bytes
//! are compared. `CD-25` is a claim about every shape, degeneracy and offer, and
//! a corpus is a list of shapes someone thought of --- a fuzzer is the part that
//! did not.
//!
//! `Scratch::none()` throughout: the selection path holds its state in
//! registers, so an offer would only vary the partition and would put an
//! allocation between the fuzzer and the claim.

#![no_main]

use libfuzzer_sys::fuzz_target;
use uor_matmul_core::{
    as_alphabet_tropical, MatView, MatViewMut, Strides, Traversal, Trop, Triple,
};
use uor_matmul_gemm::{
    driver::GemmOptions, gemm, gemm_selected, scratch::Scratch, MaxPlus, SelectedTriple, Witness,
};

/// A byte as a tropical element: one value in five is the semiring zero, so a
/// masked lane is common rather than a corner the fuzzer has to find.
fn trop(b: u8) -> Trop<i8> {
    if b % 5 == 0 {
        Trop::NEG_INF
    } else {
        Trop::finite(b as i8)
    }
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }
    // The same shape space `totality` draws from: zero dimensions, single rows,
    // and non-multiples of every block size.
    let m = (data[0] % 17) as usize;
    let k = (data[1] % 37) as usize;
    let n = (data[2] % 17) as usize;
    // Strides drawn from the fuzzer too, including negative and zero on the
    // inputs, which is what S7 promises and what a checked view must survive.
    let rsa = (data[3] as i8) as isize;
    let csa = (data[4] as i8) as isize;
    let traversal = if data[5] % 2 == 0 {
        Traversal::Blocked
    } else {
        Traversal::OutputMajor
    };
    // `beta` at the semiring zero overwrites `C` without reading it; a finite
    // `beta` reads it. Both are drawn, because the second is the one that can
    // read an output buffer the fuzzer filled with anything.
    let epilogue = if data[6] % 2 == 0 {
        MaxPlus::OVERWRITE
    } else {
        MaxPlus::ACCUMULATE
    };

    let body = &data[7..];
    let a: Vec<Trop<i8>> = body.iter().map(|&b| trop(b)).cycle().take((m * k).max(1)).collect();
    let b: Vec<Trop<i8>> = body.iter().rev().map(|&b| trop(b)).cycle().take((k * n).max(1)).collect();
    let mut c: Vec<Trop<i32>> = body
        .iter()
        .map(|&x| if x % 3 == 0 { Trop::NEG_INF } else { Trop::finite(x as i32) })
        .cycle()
        .take((m * n).max(1))
        .collect();

    // View construction is where non-existence is reported. A `None` here is
    // the library working, not failing.
    let sa = if rsa == 0 && csa == 0 {
        Strides::row_major(k)
    } else {
        Strides { rs: rsa, cs: csa }
    };
    let options = GemmOptions {
        traversal,
        ..Default::default()
    };
    {
        let Some(av) = MatView::new(as_alphabet_tropical(&a), m, k, sa) else {
            return;
        };
        let Some(bv) = MatView::row_major(as_alphabet_tropical(&b), k, n) else {
            return;
        };
        let Some(cv) = MatViewMut::row_major(&mut c, m, n) else {
            return;
        };
        let Ok(mut t) = Triple::new(av, bv, cv) else {
            return;
        };

        // Past this point nothing may fail. That is the whole claim (`CT-08`).
        gemm(&mut t, &epilogue, options, &mut Scratch::none());
    }

    // The witness, both ways, on row-major views so that the two runs differ in
    // the mechanism and in nothing else.
    let mut out = [
        (vec![Trop::<i32>::NEG_INF; (m * n).max(1)], vec![usize::MAX; (m * n).max(1)]),
        (vec![Trop::<i32>::NEG_INF; (m * n).max(1)], vec![usize::MAX; (m * n).max(1)]),
    ];
    for (slot, mechanism) in out
        .iter_mut()
        .zip([Witness::Lexicographic, Witness::ComparePass])
    {
        let (value, witness) = slot;
        let Some(av) = MatView::row_major(as_alphabet_tropical(&a), m, k) else {
            return;
        };
        let Some(bv) = MatView::row_major(as_alphabet_tropical(&b), k, n) else {
            return;
        };
        let Some(cv) = MatViewMut::row_major(value, m, n) else {
            return;
        };
        let Some(wv) = MatViewMut::row_major(witness, m, n) else {
            return;
        };
        let Ok(t) = Triple::new(av, bv, cv) else {
            return;
        };
        let Ok(mut s) = SelectedTriple::new(t, wv) else {
            return;
        };
        gemm_selected(&mut s, &MaxPlus::OVERWRITE, options, mechanism, &mut Scratch::none());
    }

    // `CD-25`: two factorizations of one answer, so the bytes are equal.
    assert_eq!(out[0], out[1], "the two witness mechanisms disagreed");
    // And every witness addresses a term of its own reduction --- or, at depth
    // zero, the past-the-end index, which is zero there.
    for &at in &out[0].1[..m * n] {
        assert!(at <= k, "witness {at} is past the depth {k}");
        assert!(k == 0 || at < k, "witness {at} at depth {k}");
    }
});
