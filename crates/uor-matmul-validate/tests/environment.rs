//! `CA-01`, `CD-06`, `CD-07`, `CD-08`, `CD-11`, `CP-01`, `CS-05`, `CU-05`.
//!
//! The claims that are about the library's *behaviour in an environment*
//! rather than about its arithmetic: that it allocates nothing, that its bytes
//! do not depend on the host's endianness, that splitting and reordering the
//! work changes nothing, and that the raw-pointer face agrees with the safe
//! one.

use uor_matmul::prelude::*;
use uor_matmul_core::{dot_ref, dot_wide, EncodeMode, PackedCode, Shape};
use uor_matmul_gemm::Partition;
use uor_matmul_validate::{counting, Case, Corpus};

/// `CA-01`: the counting allocator observes zero allocations during a call.
///
/// R7 makes this true by construction; this makes it observable. Every buffer
/// the call needs is allocated *before* the measurement, so what is counted is
/// the library and nothing else.
#[global_allocator]
static ALLOC: counting::Counting = counting::Counting;

#[test]
fn a_gemm_call_allocates_nothing_ca_01() {
    let (m, k, n) = (13usize, 97usize, 17usize);
    let a: Vec<i8> = (0..m * k).map(|i| (i % 251) as i8).collect();
    let b: Vec<i8> = (0..k * n).map(|i| (i % 241) as i8).collect();
    let mut c = vec![0i32; m * n];
    let mut scratch = vec![Alphabet::<i8, Full<i8>>::ZERO; 8192];

    // Everything the call touches now exists. Warm the counter once so that any
    // lazily initialised machinery is paid for outside the measurement.
    {
        let av = MatView::row_major(as_alphabet_full(&a), m, k).unwrap();
        let bv = MatView::row_major(as_alphabet_full(&b), k, n).unwrap();
        let cv = MatViewMut::row_major(&mut c, m, n).unwrap();
        let mut t = Triple::new(av, bv, cv).unwrap();
        gemm(
            &mut t,
            &Linear::OVERWRITE,
            GemmOptions::default(),
            &mut Scratch::none(),
        );
    }

    let ((), reading) = counting::measure(|| {
        let av = MatView::row_major(as_alphabet_full(&a), m, k).unwrap();
        let bv = MatView::row_major(as_alphabet_full(&b), k, n).unwrap();
        let cv = MatViewMut::row_major(&mut c, m, n).unwrap();
        let mut t = Triple::new(av, bv, cv).unwrap();
        gemm(
            &mut t,
            &Linear::OVERWRITE,
            GemmOptions::default(),
            &mut Scratch::new(&mut scratch),
        );
    });
    assert_eq!(reading.allocations, 0, "a GEMM call must allocate nothing");
    assert_eq!(reading.bytes, 0);

    // The kernel-driven traversal consults `is_x86_feature_detected!` on its
    // first call, and std's feature detection allocates once per process. That
    // is std's initialization, not the operation: it happens once, before any
    // arithmetic, and on a `no_std` target it does not exist at all --- which
    // is why `CA-03` (no allocator symbol is linked) is the stronger claim.
    // Warming it here measures the operation rather than the runtime.
    {
        let av = MatView::row_major(as_alphabet_full(&a), m, k).unwrap();
        let bv = MatView::row_major(as_alphabet_full(&b), k, n).unwrap();
        let cv = MatViewMut::row_major(&mut c, m, n).unwrap();
        let mut t = Triple::new(av, bv, cv).unwrap();
        uor_matmul::gemm_packed(
            &mut t,
            &Linear::OVERWRITE,
            GemmOptions::default(),
            &mut Scratch::new(&mut scratch),
        );
    }

    // The kernel-driven traversal packs panels, and they live in the caller's
    // scratch, so it allocates nothing either.
    let ((), reading) = counting::measure(|| {
        let av = MatView::row_major(as_alphabet_full(&a), m, k).unwrap();
        let bv = MatView::row_major(as_alphabet_full(&b), k, n).unwrap();
        let cv = MatViewMut::row_major(&mut c, m, n).unwrap();
        let mut t = Triple::new(av, bv, cv).unwrap();
        uor_matmul::gemm_packed(
            &mut t,
            &Linear::OVERWRITE,
            GemmOptions::default(),
            &mut Scratch::new(&mut scratch),
        );
    });
    assert_eq!(
        reading.allocations, 0,
        "the packed traversal must allocate nothing"
    );

    // The counter itself must work, or the two zeros above mean nothing.
    let (v, reading) = counting::measure(|| vec![0u8; 4096]);
    assert!(
        reading.allocations >= 1,
        "the counting allocator must actually count"
    );
    drop(v);
}

/// `CD-06`: the serialized output does not depend on the host's endianness.
///
/// The library computes integers; how a host lays them out in memory is the
/// host's business. What must agree across a big-endian and a little-endian
/// build is the *serialized* form, so the corpus is compared in explicit
/// little-endian bytes rather than in native ones.
#[test]
fn serialized_output_is_endianness_independent_cd_06() {
    for case in Corpus::standard(20_260_724).cases {
        let Case { m, k, n, .. } = case;
        let a = case.fill_i8(m * k, 1);
        let b = case.fill_i8(k * n, 2);
        let out = uor_matmul_validate::ours_i8_i32(m, k, n, &a, &b, EncodeMode::Wrapping);

        // The canonical serialization: little-endian, explicitly.
        let canonical: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
        // Reading it back reproduces the same integers on any host.
        let round: Vec<i32> = canonical
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(round, out, "{m}x{k}x{n}");
        assert_eq!(canonical.len(), out.len() * 4);
    }
}

/// `CD-07`: splitting the operand stream at any index sums the parts.
#[test]
fn splitting_the_stream_sums_the_parts_cd_07() {
    let k = 257usize;
    let a: Vec<i8> = (0..k).map(|i| ((i * 37) % 255) as i8).collect();
    let b: Vec<i8> = (0..k).map(|i| ((i * 53) % 255) as i8).collect();
    let whole = dot_ref(as_alphabet_full(&a), as_alphabet_full(&b));
    for split in [0usize, 1, 2, 128, 256, 257] {
        let (a0, a1) = a.split_at(split);
        let (b0, b1) = b.split_at(split);
        let parts = dot_ref(as_alphabet_full(a0), as_alphabet_full(b0))
            + dot_ref(as_alphabet_full(a1), as_alphabet_full(b1));
        assert_eq!(parts, whole, "split at {split}");
    }
}

/// `CD-08`: two tiles reduce in either order, and a partition covers the output
/// exactly once so no two tiles contend for one sum.
#[test]
fn tiles_reduce_in_either_order_cd_08() {
    let (m, k, n) = (12usize, 33usize, 20usize);
    let a: Vec<i8> = (0..m * k).map(|i| ((i * 29) % 255) as i8).collect();
    let b: Vec<i8> = (0..k * n).map(|i| ((i * 41) % 255) as i8).collect();
    let reference = uor_matmul_validate::ours_i8_i32(m, k, n, &a, &b, EncodeMode::Wrapping);

    for (tr, tc) in [(1usize, 1usize), (2, 3), (5, 8), (m, n), (1000, 1000)] {
        let part = Partition::new(Shape { m, k, n }, tr, tc);
        // Claim the tiles in a shuffled order, as a work-stealing executor
        // would, and write each into its own region.
        let mut order: Vec<usize> = (0..part.len()).collect();
        order.reverse();
        let mut out = vec![0i32; m * n];
        for &idx in &order {
            let tile = part.tile(idx).unwrap();
            for i in tile.row..tile.row + tile.rows {
                for j in tile.col..tile.col + tile.cols {
                    let row: Vec<i8> = (0..k).map(|p| a[i * k + p]).collect();
                    let col: Vec<i8> = (0..k).map(|p| b[p * n + j]).collect();
                    let acc = dot_ref(as_alphabet_full(&row), as_alphabet_full(&col));
                    out[i * n + j] = acc as i32;
                }
            }
        }
        assert_eq!(out, reference, "tiled {tr}x{tc}, claimed in reverse");
    }
}

/// `CD-11`: forcing a wider accumulator than necessary changes nothing but the
/// room. `dot_wide` skips the narrow-register factorization entirely.
#[test]
fn a_wider_accumulator_changes_nothing_cd_11() {
    for case in Corpus::standard(11).cases {
        let k = case.k.max(1);
        let a = case.fill_i8(k, 5);
        let b = case.fill_i8(k, 6);
        assert_eq!(
            dot_ref(as_alphabet_full(&a), as_alphabet_full(&b)),
            dot_wide(as_alphabet_full(&a), as_alphabet_full(&b)),
            "k={k}"
        );
    }
}

/// `CS-05`: the raw-pointer face produces the same bytes as the safe API on the
/// same inputs.
#[test]
fn the_raw_pointer_face_agrees_with_the_safe_api_cs_05() {
    for (m, k, n) in [
        (1usize, 1usize, 1usize),
        (3, 7, 5),
        (8, 64, 8),
        (13, 17, 19),
    ] {
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.5 - 3.0).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * -0.25 + 1.5).collect();

        let mut safe = vec![0.0f32; m * n];
        {
            let av = MatView::row_major(&a, m, k).unwrap();
            let bv = MatView::row_major(&b, k, n).unwrap();
            let cv = MatViewMut::row_major(&mut safe, m, n).unwrap();
            let mut t = Triple::new(av, bv, cv).unwrap();
            uor_matmul::gemm_float(&mut t, &Linear::OVERWRITE, GemmOptions::default());
        }

        let mut raw = vec![0.0f32; m * n];
        // SAFETY: all three buffers are row-major and exactly `rows * cols`
        // long, `raw` is disjoint from `a` and `b`, and the output strides do
        // not alias.
        unsafe {
            uor_matmul::raw::sgemm(
                m,
                k,
                n,
                1,
                a.as_ptr(),
                k as isize,
                1,
                b.as_ptr(),
                n as isize,
                1,
                0,
                raw.as_mut_ptr(),
                n as isize,
                1,
            );
        }
        assert_eq!(raw, safe, "{m}x{k}x{n}");
    }
}

/// `CS-08`: the slice face gives the same bytes as the view API on the same
/// operands, at every leading dimension, `alpha`, `beta`, and offer.
///
/// The point of the face is that it is *not* a second path, so the comparison is
/// against the views rather than against a recomputed expectation: a slice call
/// that agreed with arithmetic but disagreed with `gemm` would be a second
/// method (R13).
#[test]
fn the_slice_face_agrees_with_the_view_api_cs_08() {
    use uor_matmul::slice;
    let mut compared = 0usize;
    for &(m, k, n) in &[
        (1usize, 1usize, 1usize),
        (2, 2, 2),
        (3, 7, 5),
        (8, 64, 8),
        (13, 17, 19),
        (1, 32, 40),
        (5, 1, 3),
    ] {
        // Padding past the row, so a leading dimension that is not the column
        // count is exercised and not just the contiguous case.
        for pad in [0usize, 1, 4] {
            let (lda, ldb, ldc) = (k + pad, n + pad, n + pad);
            let a: Vec<i8> = (0..m * lda)
                .map(|i| ((i * 13 % 255) as i64 - 127) as i8)
                .collect();
            let b: Vec<i8> = (0..k * ldb)
                .map(|i| ((i * 7 % 255) as i64 - 127) as i8)
                .collect();
            for &(alpha, beta) in &[(1i64, 0i64), (1, 1), (3, -2), (0, 1)] {
                // The view API, on views built with exactly these strides.
                let mut want = vec![5i32; m * ldc];
                {
                    let av = MatView::new(
                        as_alphabet_full(&a),
                        m,
                        k,
                        Strides {
                            rs: lda as isize,
                            cs: 1,
                        },
                    )
                    .expect("a fits");
                    let bv = MatView::new(
                        as_alphabet_full(&b),
                        k,
                        n,
                        Strides {
                            rs: ldb as isize,
                            cs: 1,
                        },
                    )
                    .expect("b fits");
                    let cv = MatViewMut::new(
                        &mut want,
                        m,
                        n,
                        Strides {
                            rs: ldc as isize,
                            cs: 1,
                        },
                    )
                    .expect("c fits");
                    let mut t = Triple::new(av, bv, cv).expect("a product");
                    let mut buf =
                        vec![Default::default(); uor_matmul::suggested_scratch(Shape { m, k, n })];
                    uor_matmul::gemm(
                        &mut t,
                        &Linear { alpha, beta },
                        GemmOptions::default(),
                        &mut Scratch::new(&mut buf),
                    );
                }

                // The slice face, at every offer: none, and the suggested amount.
                for offer in [0usize, uor_matmul::suggested_scratch(Shape { m, k, n })] {
                    let mut got = vec![5i32; m * ldc];
                    let mut scratch = vec![0i8; offer];
                    slice::gemm_ex(
                        m,
                        k,
                        n,
                        alpha,
                        &a,
                        lda,
                        &b,
                        ldb,
                        beta,
                        &mut got,
                        ldc,
                        &mut scratch,
                    )
                    .expect("the operands name a product");
                    assert_eq!(
                        got, want,
                        "{m}x{k}x{n} pad={pad} alpha={alpha} beta={beta} offer={offer}"
                    );
                    compared += 1;
                }
            }
        }
        // The contiguous convenience form is the same call at the implied
        // dimensions, so it must agree with itself spelled the long way.
        let a: Vec<i8> = (0..m * k)
            .map(|i| ((i * 13 % 255) as i64 - 127) as i8)
            .collect();
        let b: Vec<i8> = (0..k * n)
            .map(|i| ((i * 7 % 255) as i64 - 127) as i8)
            .collect();
        let mut short = vec![0i32; m * n];
        let mut long = vec![0i32; m * n];
        slice::gemm(m, k, n, &a, &b, &mut short, &mut []).expect("a product");
        slice::gemm_ex(m, k, n, 1, &a, k, &b, n, 0, &mut long, n, &mut []).expect("a product");
        assert_eq!(
            short, long,
            "{m}x{k}x{n} the convenience form is the long one"
        );
        compared += 1;

        // Floats take the same shape, and agree with `gemm_float` on views.
        let fa: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.5 - 3.0).collect();
        let fb: Vec<f32> = (0..k * n).map(|i| (i as f32) * -0.25 + 1.5).collect();
        let mut fwant = vec![0.0f32; m * n];
        {
            let av = MatView::row_major(&fa, m, k).expect("a fits");
            let bv = MatView::row_major(&fb, k, n).expect("b fits");
            let cv = MatViewMut::row_major(&mut fwant, m, n).expect("c fits");
            let mut t = Triple::new(av, bv, cv).expect("a product");
            uor_matmul::gemm_float(&mut t, &Linear::OVERWRITE, GemmOptions::default());
        }
        for (pa_len, pb_len) in [(0usize, 0usize), (k, k * n)] {
            let mut fgot = vec![0.0f32; m * n];
            let mut pa = vec![PackedCode::default(); pa_len];
            let mut pb = vec![PackedCode::default(); pb_len];
            slice::gemm_float(m, k, n, &fa, &fb, &mut fgot, &mut pa, &mut pb).expect("a product");
            assert_eq!(fgot, fwant, "{m}x{k}x{n} float, panels {pa_len}/{pb_len}");
            compared += 1;
        }
    }

    // The error surface is the view API's and no wider: an output whose rows land
    // on each other is a product that does not exist, and it is *reported*.
    let aliased = slice::gemm_ex::<i8, i32>(
        2,
        2,
        2,
        1,
        &[1, 2, 3, 4],
        2,
        &[1, 2, 3, 4],
        2,
        0,
        &mut [0; 4],
        0,
        &mut [],
    );
    assert!(
        matches!(
            aliased,
            Err(uor_matmul::NotAProduct::OutputAliasesItself { .. })
        ),
        "an output whose rows land on each other is reported: {aliased:?}"
    );
    // A non-conformant shape cannot be spelled through this face --- `k` is one
    // argument used for both operands --- so that condition is reachable only
    // through the views, where `CT-05` asserts it.

    assert!(compared > 0, "CS-08 compared nothing");
    eprintln!("CS-08: {compared} slice calls agreed with the view API");
}

/// `CS-08`: a slice that does not hold the matrix its caller described is a
/// programming error at the boundary, not a condition of the data.
///
/// The library's reportable conditions are the two in `NotAProduct` (`CT-05`). A
/// buffer shorter than the dimensions passed beside it is a disagreement between
/// two of the caller's own arguments, which is the same class as handing a kernel
/// a panel of the wrong length --- and that has always been a panic at the
/// boundary here, never an error return.
#[test]
#[should_panic(expected = "leading dimension")]
fn a_slice_that_does_not_hold_its_matrix_panics_cs_08() {
    // `a` is declared 2x2 and has three elements.
    let _ = uor_matmul::slice::gemm::<i8, i32>(
        2,
        2,
        2,
        &[1i8, 2, 3],
        &[1i8, 2, 3, 4],
        &mut [0i32; 4],
        &mut [],
    );
}

/// `CP-01`: the randomized differential suite, at the sample size derived in
/// `ANALYSIS.md`, with the seed recorded.
///
/// The sample size is not a round number chosen for comfort. `ANALYSIS.md`
/// derives it from the probability that a defect confined to one residue class
/// of the packing survives: with `MR * NR <= 256` tile positions and a required
/// 1e-6 escape probability, `n >= 256 * ln(1e6) ≈ 3538` independent cases per
/// shape family, and the suite runs more.
#[test]
fn the_randomized_differential_suite_cp_01() {
    const SEED: u64 = 20_260_724;
    const CASES: usize = 4_096;

    let mut s = SEED;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };

    let mut checked = 0usize;
    for _ in 0..CASES {
        let m = (next() % 9) as usize + 1;
        let k = (next() % 130) as usize;
        let n = (next() % 9) as usize + 1;
        let a: Vec<i8> = (0..m * k).map(|_| (next() >> 33) as i8).collect();
        let b: Vec<i8> = (0..k * n).map(|_| (next() >> 33) as i8).collect();

        let ours = uor_matmul_validate::ours_i8_i32(m, k, n, &a, &b, EncodeMode::Wrapping);
        let widened_a: Vec<i32> = a.iter().map(|&x| x as i32).collect();
        let widened_b: Vec<i32> = b.iter().map(|&x| x as i32).collect();
        let reference =
            uor_matmul_validate::reference_wrapping_i32(m, k, n, &widened_a, &widened_b);
        assert_eq!(ours, reference, "seed {SEED}, case {checked}: {m}x{k}x{n}");
        checked += 1;
    }
    assert_eq!(checked, CASES);
    eprintln!("CP-01: {CASES} cases at seed {SEED}, all byte-identical");
}

/// `CU-05`: there is exactly one accumulation path per element family.
///
/// Every entry point below is the same three steps at a different
/// instantiation, so they must agree wherever their domains overlap. If one of
/// them were a second method, this is where it would show.
#[test]
fn one_accumulation_path_per_family_cu_05() {
    let (m, k, n) = (7usize, 40usize, 5usize);
    let a: Vec<i8> = (0..m * k).map(|i| ((i * 19) % 255) as i8).collect();
    let b: Vec<i8> = (0..k * n).map(|i| ((i * 23) % 255) as i8).collect();

    // The generic driver.
    let generic = uor_matmul_validate::ours_i8_i32(m, k, n, &a, &b, EncodeMode::Wrapping);

    // The kernel-driven traversal.
    let mut packed = vec![0i32; m * n];
    let mut scratch = vec![Alphabet::<i8, Full<i8>>::ZERO; 4096];
    {
        let av = MatView::row_major(as_alphabet_full(&a), m, k).unwrap();
        let bv = MatView::row_major(as_alphabet_full(&b), k, n).unwrap();
        let cv = MatViewMut::row_major(&mut packed, m, n).unwrap();
        let mut t = Triple::new(av, bv, cv).unwrap();
        uor_matmul::gemm_packed(
            &mut t,
            &Linear::OVERWRITE,
            GemmOptions {
                encode: EncodeMode::Wrapping,
                ..Default::default()
            },
            &mut Scratch::new(&mut scratch),
        );
    }
    assert_eq!(
        packed, generic,
        "the packed traversal is the generic one, factored"
    );

    // The coded driver, with the identity codec: the same weights, named as
    // codes rather than as values.
    let mut coded_out = vec![0i32; m * n];
    {
        let coded = CodedMatrix::new(Identity, k, n, as_alphabet_full(&b)).unwrap();
        let av = MatView::row_major(as_alphabet_full(&a), m, k).unwrap();
        let cv = MatViewMut::row_major(&mut coded_out, m, n).unwrap();
        let mut t = uor_matmul::CodedTriple::new(av, coded, cv).unwrap();
        uor_matmul::coded_gemm(
            &mut t,
            &Linear::OVERWRITE,
            GemmOptions {
                encode: EncodeMode::Wrapping,
                ..Default::default()
            },
        );
    }
    assert_eq!(
        coded_out, generic,
        "the coded driver is the generic one, decoded"
    );
}
