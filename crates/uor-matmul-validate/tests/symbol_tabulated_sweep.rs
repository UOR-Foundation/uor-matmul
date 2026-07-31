//! `CG-16` (open): achieved MACs per second of the symbol tabulated traversal
//! in the scaled integer lane, against the float placement bridge, the dense
//! float driver, and an `f32` oracle.
//!
//! The lane is the bridge's identity with the table doing the reduction
//! (`CD-19` pins the bridge's bytes, `CD-20` the lane's): the codebook and the
//! activation tile are pre-scaled to the panels' measured bases, the table is
//! built over the scaled integer alphabet, and the gather is one read and one
//! add per code. `CG-14` priced this lane as the only lever that moves the
//! symbol path; what is left is whether the clock agrees, and either answer is
//! the finding. At `n = 1` it cannot: the build is `code_space` products per
//! element of the reduction amortized over `n` output columns, so the matrix-
//! vector rows are the amortization's absence, reported rather than omitted.
//!
//! Every figure is `open`: printed, never asserted. What *is* asserted,
//! inside each timed run, is byte-identity with the dense float driver --- a
//! speed measured on the wrong bytes is not a measurement --- and the census,
//! which says whether the table really ran rather than asking the predicate
//! twice.
//!
//! Ignored by default, like the other minute-long sweeps: `just
//! symbol-tabulated` runs it, in release, where a throughput figure means
//! something.

mod common;

use common::{best, fill, spanned, wide, SHAPES};

use uor_matmul_codec::{Arena, CodedMatrix};
use uor_matmul_core::{
    as_alphabet_whole, Alphabet, Full, MatView, MatViewMut, PackedCode, Shape, Traversal, Triple,
    Whole,
};
use uor_matmul_gemm::{
    gemm_float_bridged, gemm_float_packed, gemm_tabulated_counted, suggested_accumulators,
    suggested_bridge_scaled, suggested_scratch, suggested_tabulation, suggested_tabulation_index,
    suggested_tabulation_lanes, suggested_tabulation_panel, Census, Collapse, GemmOptions, Linear,
    Scratch, TabulatedTriple, Tabulation,
};
use uor_matmul_validate::oracle::{FloatOracle, MatrixMultiply};

/// The host's STREAM number, measured as `CG-14`'s: a triad over arrays far
/// past the last cache, 12 bytes counted per element.
fn stream_triad() -> f64 {
    let len = 1usize << 25;
    let mut a = vec![0.0f32; len];
    let b = fill(len, 0x571, |x| (x % 4096) as f32 * 0.25);
    let c = fill(len, 0xea4, |x| (x % 4096) as f32 * 0.25);
    let t = best(|| {
        for i in 0..len {
            a[i] = b[i] + 3.0 * c[i];
        }
    });
    std::hint::black_box(&a);
    (len as f64) * 12.0 / t
}

/// One measured path on one shape.
struct Row {
    table: f64,
    bridged: f64,
    dense: f64,
    oracle: f64,
}

#[allow(clippy::too_many_lines)]
#[allow(clippy::too_many_arguments)]
fn measure(
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    w: CodedMatrix<'_, f32, Whole<f32>, Arena<'_, f32, 256, u8>>,
    b: &[f32],
    want: &[f32],
    admitted: bool,
) -> Row {
    let shape = Shape { m, k, n };

    // The symbol path with the table forced: the full offers --- panel,
    // accumulators, lane words, collapse index --- so the plan is the wide
    // one, and the census printed per shape says what ran.
    let mut c_table = vec![0.0f32; m * n];
    let mut panel = vec![Alphabet::<f32, Whole<f32>>::ZERO; suggested_tabulation_panel(256, 1)];
    let mut accumulators = vec![
        <uor_matmul_core::AccOf<f32> as uor_matmul_core::Accumulator>::ZERO;
        suggested_tabulation::<f32, Whole<f32>>(shape, 256, 1).max(1)
    ];
    let mut lane_words =
        vec![0i64; suggested_tabulation_lanes::<f32, Whole<f32>>(shape, 256, 1).max(1)];
    let mut ids = vec![0usize; suggested_tabulation_index(shape)];
    let mut census = Census::default();
    let mut run_table = |c_table: &mut [f32], census: &mut Census| {
        let av = MatView::row_major(as_alphabet_whole(a), m, k).unwrap();
        let cv = MatViewMut::row_major(c_table, m, n).unwrap();
        let mut tr = TabulatedTriple::new(av, w, cv).unwrap();
        gemm_tabulated_counted(
            &mut tr,
            &Linear::OVERWRITE,
            GemmOptions {
                traversal: Traversal::Tabulated,
                ..Default::default()
            },
            &mut Scratch::with_accumulators(&mut panel, &mut accumulators),
            &mut Tabulation::with_index(&mut lane_words, &mut ids),
            &mut Collapse::none(),
            census,
        );
    };
    run_table(&mut c_table, &mut census);
    // The census of one run, read before the timed reps accumulate into it.
    let one_run = census;
    let table = best(|| {
        run_table(&mut c_table, &mut census);
        // Byte-identity with the dense float driver, inside the timed region
        // (`CD-20`).
        assert_eq!(
            c_table.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "the symbol table must give the dense driver's bytes at {m}x{k}x{n}"
        );
    });
    if admitted {
        assert!(
            one_run.table_reads > 0,
            "{m}x{k}x{n}: the fill admits the lane, so the table really ran ({one_run:?})"
        );
    } else {
        assert_eq!(
            one_run.table_reads, 0,
            "{m}x{k}x{n}: the fill is past the alphabet, so the lane declined ({one_run:?})"
        );
    }
    eprintln!(
        "# sym table {m}x{k}x{n} census: table_reads {}, decodes {} (m*k = {}), multiplies {}",
        one_run.table_reads,
        one_run.decodes,
        m * k,
        one_run.multiplies
    );

    // The bridge over the dense spelling, on the shapes its own sweep tables.
    let mut c_bridged = vec![0.0f32; m * n];
    let mut pa = vec![PackedCode::default(); k.max(1)];
    let mut pb = vec![PackedCode::default(); k * n];
    let mut scaled = vec![0i32; suggested_bridge_scaled(shape)];
    let mut kernel_buf = vec![Alphabet::<i32, Full<i32>>::of(0); suggested_scratch(shape)];
    let mut acc_buf = vec![0i128; suggested_accumulators(shape)];
    let bridged = best(|| {
        let av = MatView::row_major(a, m, k).unwrap();
        let bv = MatView::row_major(b, k, n).unwrap();
        let cv = MatViewMut::row_major(&mut c_bridged, m, n).unwrap();
        let mut t = Triple::new(av, bv, cv).unwrap();
        gemm_float_bridged(
            &mut t,
            &Linear::OVERWRITE,
            GemmOptions::default(),
            &mut pa,
            &mut pb,
            &mut scaled,
            &mut Scratch::with_accumulators(&mut kernel_buf, &mut acc_buf),
        );
        assert_eq!(
            c_bridged.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "the bridge must give the dense driver's bytes at {m}x{k}x{n}"
        );
    });

    // The dense float driver: the same exact accumulation, four bytes a
    // weight.
    let mut c_dense = vec![0.0f32; m * n];
    let dense = best(|| {
        let av = MatView::row_major(a, m, k).unwrap();
        let bv = MatView::row_major(b, k, n).unwrap();
        let cv = MatViewMut::row_major(&mut c_dense, m, n).unwrap();
        let mut t = Triple::new(av, bv, cv).unwrap();
        gemm_float_packed(
            &mut t,
            &Linear::OVERWRITE,
            GemmOptions::default(),
            &mut pa,
            &mut pb,
        );
    });
    assert_eq!(
        c_dense.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        want.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        "the dense run must give its own first bytes at {m}x{k}x{n}"
    );

    // The oracle, inexact by its own admission (`CX-05`), reported for scale.
    let oracle = best(|| {
        let c = MatrixMultiply::product_f32(m, k, n, a, b);
        std::hint::black_box(&c);
    });

    Row {
        table,
        bridged,
        dense,
        oracle,
    }
}

#[test]
#[ignore = "a minutes-long release-mode sweep: `just symbol-tabulated`"]
fn the_symbol_table_against_the_bridge_and_the_bus_cg_16() {
    // The committed corpus's codebook: 256 distinct f32 bit patterns spanning
    // seven binades --- exactly the widest span the lane's alphabet admits at
    // `f32` (`24 + 7 <= 31`), so the design point stands on the admission
    // boundary.
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("two below the root")
        .join("oracles/symbols");
    let codebook: Vec<f32> = std::fs::read(dir.join("codebook.f32.bin"))
        .expect("the committed codebook")
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("four bytes")))
        .collect();
    let table: &[Alphabet<f32, Whole<f32>>; 256] = as_alphabet_whole(&codebook).try_into().unwrap();

    let stream = stream_triad();
    let gmacs = |m: usize, k: usize, n: usize, secs: f64| (m * k * n) as f64 / secs / 1e9;
    println!();
    println!("# CG-16 (open): the symbol tabulated traversal in the scaled integer lane, Gmac/s");
    println!(
        "# host: {}-{}; best of a 0.35 s budget per point;",
        std::env::consts::ARCH,
        std::env::consts::OS
    );
    println!(
        "# STREAM triad a[i]=b[i]+3*c[i], 3 x 2^25 f32 (384 MiB), 12 bytes counted per element: {:.2} GB/s",
        stream / 1e9
    );
    println!("# byte-identity with the dense float driver is asserted inside every timed run;");
    println!("#   the census printed per shape says which factorization ran");
    for (label, span_a, admitted) in [
        ("one exponent", 0, true),
        ("a few binades", 3, true),
        ("wide spans (~18, the lane declines)", 18, false),
    ] {
        println!();
        println!("| fill: {label} | m x k x n | sym table | bridge | uor f32 | matrixmultiply |");
        println!("| --- | --- | --- | --- | --- | --- |");
        for &(m, k, n) in SHAPES {
            let a: Vec<f32> = if admitted {
                spanned(m * k, 5, span_a)
            } else {
                wide(m * k, 5)
            };
            let codes: Vec<u8> = fill(n * k, 6, |x| x as u8);
            let b: Vec<f32> = (0..k * n)
                .map(|at| codebook[codes[(at % n) * k + at / n] as usize])
                .collect();
            let w = CodedMatrix::new(Arena::new(table), n, k, &codes)
                .expect("the codes describe n x k");
            // The reference bytes, computed once, untimed, by the dense float
            // driver over the dense spelling.
            let mut want = vec![0.0f32; m * n];
            {
                let av = MatView::row_major(&a, m, k).unwrap();
                let bv = MatView::row_major(&b, k, n).unwrap();
                let cv = MatViewMut::row_major(&mut want, m, n).unwrap();
                let mut t = Triple::new(av, bv, cv).unwrap();
                let mut pa = vec![PackedCode::default(); k.max(1)];
                let mut pb = vec![PackedCode::default(); k * n];
                gemm_float_packed(
                    &mut t,
                    &Linear::OVERWRITE,
                    GemmOptions::default(),
                    &mut pa,
                    &mut pb,
                );
            }
            let r = measure(m, k, n, &a, w, &b, &want, admitted);
            println!(
                "| | {m}x{k}x{n} | {:.3} | {:.3} | {:.3} | {:.3} |",
                gmacs(m, k, n, r.table),
                gmacs(m, k, n, r.bridged),
                gmacs(m, k, n, r.dense),
                gmacs(m, k, n, r.oracle),
            );
        }
    }
}
