//! What a table costs against what the kernels cost.
//!
//! `CG-10` reports the census, which is machine-independent, and the wall clock
//! against the streaming traversal. This sweep asks the harder question: how does
//! the tabulated traversal compare against this library's *fast* dense path ---
//! the packed AVX2 tile kernels --- on the same product.
//!
//! It runs `Traversal::Blocked`, which is the default and therefore what a caller
//! actually gets: the library's own choice between the table and the streaming
//! traversal at each shape, not the table forced. The `picked` column says which
//! way the predicate went, and the point of the sweep is to check that judgement
//! against the clock.
//!
//! Every figure is `open`. The answer is checked to be the same bytes the dense
//! driver produces, inside the timed region, at every shape.

use std::time::Instant;

use uor_matmul_codec::{e8_codec, e8_table, Codec, CodedMatrix, Enumerable};
use uor_matmul_core::{
    as_alphabet_full, AccOf, Accumulator, Alphabet, EncodeMode, Full, MatView, MatViewMut, Shape,
    Traversal, Triple,
};
use uor_matmul_gemm::{
    gemm_packed, gemm_tabulated, gemm_tabulated_counted, suggested_scratch, suggested_tabulation,
    suggested_tabulation_lanes, suggested_tabulation_panel, Census, GemmOptions, Linear, Scratch,
    TabulatedTriple, Tabulation,
};

type Book<'a> = uor_matmul_codec::Book<'a, i8, Full<i8>, 256, 8>;

fn best(mut run: impl FnMut() -> f64) -> f64 {
    const BUDGET: f64 = 0.40;
    let mut best = f64::INFINITY;
    let mut spent = 0.0;
    loop {
        let t = run();
        best = best.min(t);
        spent += t;
        if spent >= BUDGET {
            return best;
        }
    }
}

fn fill(len: usize, salt: u64) -> Vec<u64> {
    let mut s = 0x243F_6A88_85A3_08D3u64 ^ salt;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s >> 33
        })
        .collect()
}

fn main() {
    let table = e8_table::<Full<i8>>().expect("i8 holds E8");
    let book = e8_codec(&table);
    let space = <Book<'_> as Enumerable<i8, Full<i8>>>::CODE_SPACE;
    let block = <Book<'_> as Codec<i8, Full<i8>>>::MAX_BLOCK;

    println!("# Tabulation against the packed kernels");
    println!();
    println!("Gmac/s against the nominal `m * k * n`. `packed` is the dense AVX2");
    println!("tile path over the *decoded* weights; `tabulated` never decodes more");
    println!("than `{space} * {block}` codewords per row tile.");
    println!();
    println!("| `m x k x n` | default | packed | forced stream | vs packed | picked |");
    println!("| --- | --- | --- | --- | --- | --- |");

    for &(m, k, n) in &[
        (1usize, 1024usize, 4096usize),
        (8, 1024, 4096),
        (64, 1024, 4096),
        (64, 4096, 4096),
        (256, 1024, 4096),
        (64, 1024, 16384),
        // Shapes that divide nothing: a ragged row tile, a prime column count,
        // and a depth that is one block past a round number. A traversal that had
        // a cliff at an awkward size would show it here.
        (3, 1024, 4093),
        (17, 1032, 1021),
        (1, 8192, 1),
        (1000, 512, 512),
    ] {
        let macs = (m * k * n) as f64;
        let shape = Shape { m, k, n };
        let blocks = k / block;

        let a: Vec<i8> = fill(m * k, 0xa11)
            .into_iter()
            .map(|x| ((x % 255) as i64 - 127) as i8)
            .collect();
        let stream: Vec<u16> = fill(n * blocks, 0xb00c)
            .into_iter()
            .map(|x| (x % 256) as u16)
            .collect();
        let w = CodedMatrix::new(book, n, k, &stream).expect("the codes describe n x k");

        // The same weights, decoded, for the dense path.
        let mut b = vec![0i8; k * n];
        for p in 0..k {
            for j in 0..n {
                b[p * n + j] = w.at(j, p).get();
            }
        }

        let options = |traversal: Traversal| GemmOptions {
            traversal,
            encode: EncodeMode::Wrapping,
            ..Default::default()
        };

        let offer = suggested_tabulation::<i8, Full<i8>>(shape, space, block);
        let mut accumulators = vec![<AccOf<i8> as Accumulator>::ZERO; offer];
        let mut words = vec![0i64; suggested_tabulation_lanes::<i8, Full<i8>>(shape, space, block)];
        let mut lanes =
            vec![Alphabet::<i8, Full<i8>>::ZERO; suggested_tabulation_panel(space, block)];
        let mut c = vec![0i32; m * n];

        let t_tab = {
            let (a, w, c, lanes, accumulators, words) =
                (&a, &w, &mut c, &mut lanes, &mut accumulators, &mut words);
            best(|| {
                let s = Instant::now();
                {
                    let av = MatView::row_major(as_alphabet_full(a), m, k).unwrap();
                    let cv = MatViewMut::row_major(c, m, n).unwrap();
                    let mut tr = TabulatedTriple::new(av, *w, cv).unwrap();
                    gemm_tabulated(
                        &mut tr,
                        &Linear::OVERWRITE,
                        options(Traversal::Blocked),
                        &mut Scratch::with_accumulators(lanes, accumulators),
                        &mut Tabulation::new(words),
                    );
                }
                s.elapsed().as_secs_f64()
            })
        };
        let tabulated_bytes = c.clone();

        let t_stream = {
            let (a, w, c) = (&a, &w, &mut c);
            let s = Instant::now();
            {
                let av = MatView::row_major(as_alphabet_full(a), m, k).unwrap();
                let cv = MatViewMut::row_major(c, m, n).unwrap();
                let mut tr = TabulatedTriple::new(av, *w, cv).unwrap();
                gemm_tabulated(
                    &mut tr,
                    &Linear::OVERWRITE,
                    options(Traversal::OutputMajor),
                    &mut Scratch::none(),
                    &mut Tabulation::none(),
                );
            }
            s.elapsed().as_secs_f64()
        };
        assert_eq!(
            tabulated_bytes, c,
            "the traversals must agree byte for byte"
        );

        let mut scratch = vec![Alphabet::<i8, Full<i8>>::ZERO; suggested_scratch(shape)];
        // Which way the library went, read from the census rather than recomputed
        // from the predicate: a table that was never read is a table that was not
        // chosen, and that is a fact rather than a restatement.
        let picked = {
            let mut census = Census::default();
            let mut out = vec![0i32; m * n];
            let av = MatView::row_major(as_alphabet_full(&a), m, k).unwrap();
            let cv = MatViewMut::row_major(&mut out, m, n).unwrap();
            let mut tr = TabulatedTriple::new(av, w, cv).unwrap();
            gemm_tabulated_counted(
                &mut tr,
                &Linear::OVERWRITE,
                options(Traversal::Blocked),
                &mut Scratch::with_accumulators(&mut lanes, &mut accumulators),
                &mut Tabulation::new(&mut words),
                &mut census,
            );
            census.table_reads > 0
        };
        let t_packed = {
            let (a, b, c, scratch) = (&a, &b, &mut c, &mut scratch);
            best(|| {
                let s = Instant::now();
                {
                    let av = MatView::row_major(as_alphabet_full(a), m, k).unwrap();
                    let bv = MatView::row_major(as_alphabet_full(b), k, n).unwrap();
                    let cv = MatViewMut::row_major(c, m, n).unwrap();
                    let mut tr = Triple::new(av, bv, cv).unwrap();
                    gemm_packed(
                        &mut tr,
                        &Linear::OVERWRITE,
                        options(Traversal::Blocked),
                        &mut Scratch::new(scratch),
                    );
                }
                s.elapsed().as_secs_f64()
            })
        };
        assert_eq!(tabulated_bytes, c, "the table must give the packed bytes");

        let g = |t: f64| macs / t / 1e9;
        println!(
            "| `{m}x{k}x{n}` | {:.2} | {:.2} | {:.3} | {:.2}x | {} |",
            g(t_tab),
            g(t_packed),
            g(t_stream),
            t_packed / t_tab,
            if picked { "table" } else { "stream" },
        );
    }
}
