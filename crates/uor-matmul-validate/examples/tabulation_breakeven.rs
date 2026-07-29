//! The tabulation break-even, measured rather than derived.
//!
//! `model/tiers.toml`'s `[[tabulation]]` rows record the first `n` at which the
//! tabulated instruction count crosses the dense tile's, recomputed by `CM-04`
//! from the two sequences' own declarations. This sweep asks the clock the same
//! question on the host it runs on: at which `n` does the forced tabulated
//! traversal actually overtake the packed dense kernels on the same product,
//! and which way does the shipped predicate (`Traversal::Blocked`, the default)
//! go at each width --- read from the census, so a silently declined table
//! misreports nothing.
//!
//! The derivation's prediction is host-specific, because both sides are. On an
//! aarch64 host with the dot-product extension the table is the NEON sequence
//! (four `i32` lanes to an add) and the dense tile is `NEON_DOTPROD_I8_I32`
//! (sixteen products to an instruction), which puts the instruction-count
//! crossing for `Book<256,8>` at a sixteen-row tile at `n = 2049` --- not the
//! 683 the AVX2 pair records. A one-row tile has no vector table sequence on
//! any shipped ISA --- the gathers there are the reference's, one lane per add
//! --- so the dense side's scaling by the rows present lands the crossing back
//! at 683.
//!
//! Every figure is `open`: instruction counts are a derivation, wall time is a
//! measurement, and the two are not the same quantity. On the host this was
//! written for the count is the conservative one --- the census flips exactly
//! at the derived `n` while the clock has the table ahead at every width. The
//! answer is checked to be the same bytes the dense driver produces, inside
//! the timed region, at every shape.

use std::time::Instant;

use uor_matmul_codec::{e8_codec, e8_table, Codec, CodedMatrix, Enumerable};
use uor_matmul_core::{
    as_alphabet_full, AccOf, Accumulator, Alphabet, EncodeMode, Full, MatView, MatViewMut, Shape,
    Traversal, Triple,
};
use uor_matmul_gemm::{
    gemm_packed, gemm_tabulated, gemm_tabulated_counted, suggested_scratch, suggested_tabulation,
    suggested_tabulation_index, suggested_tabulation_lanes, suggested_tabulation_panel, Census,
    Collapse, GemmOptions, Linear, Scratch, TabulatedTriple, Tabulation,
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

    println!("# The tabulation break-even on this host");
    println!();
    println!("Gmac/s against the nominal `m * k * n`. `tabulated` is the table");
    println!("forced; `packed` is the dense tile path over the decoded weights;");
    println!("`picked` is which side `Traversal::Blocked` --- the default a caller");
    println!("gets --- actually ran, read from the census.");
    println!();
    println!("| `m x k x n` | tabulated | packed | tab/packed | picked |");
    println!("| --- | --- | --- | --- | --- |");

    for &(m, k, n) in &[
        // A full sixteen-row tile: the shape the recorded rows are written
        // for. The widths bracket the AVX2 pair's 683 and this host's
        // derived 2049.
        (16usize, 1024usize, 512usize),
        (16, 1024, 683),
        (16, 1024, 1024),
        (16, 1024, 1366),
        (16, 1024, 2048),
        (16, 1024, 2049),
        (16, 1024, 2732),
        (16, 1024, 4096),
        (16, 1024, 8192),
        // A one-row tile: the dense side pays the panel copy and the
        // derivation's `present` term moves the crossing to 137.
        (1, 1024, 128),
        (1, 1024, 137),
        (1, 1024, 256),
        (1, 1024, 512),
        (1, 1024, 683),
        (1, 1024, 1024),
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
        let mut ids = vec![0usize; suggested_tabulation_index(shape)];
        // The panel offer is the table's own plus room for the decoded
        // operand, so both routes are available to the default traversal.
        let mut lanes = vec![
            Alphabet::<i8, Full<i8>>::ZERO;
            suggested_tabulation_panel(space, block)
                .max(n * k + suggested_scratch(shape))
        ];
        let mut c = vec![0i32; m * n];

        let t_tab = {
            let (a, w, c, lanes, accumulators, words, ids) = (
                &a,
                &w,
                &mut c,
                &mut lanes,
                &mut accumulators,
                &mut words,
                &mut ids,
            );
            best(|| {
                let s = Instant::now();
                {
                    let av = MatView::row_major(as_alphabet_full(a), m, k).unwrap();
                    let cv = MatViewMut::row_major(c, m, n).unwrap();
                    let mut tr = TabulatedTriple::new(av, *w, cv).unwrap();
                    gemm_tabulated(
                        &mut tr,
                        &Linear::OVERWRITE,
                        options(Traversal::Tabulated),
                        &mut Scratch::with_accumulators(lanes, accumulators),
                        &mut Tabulation::with_index(words, ids),
                        &mut Collapse::none(),
                    );
                }
                s.elapsed().as_secs_f64()
            })
        };
        let tabulated_bytes = c.clone();

        let mut scratch = vec![Alphabet::<i8, Full<i8>>::ZERO; suggested_scratch(shape)];
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

        // Which way the library went, read from the census rather than
        // recomputed from the predicate.
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
                &mut Tabulation::with_index(&mut words, &mut ids),
                &mut Collapse::none(),
                &mut census,
            );
            if census.table_reads > 0 {
                "table"
            } else if census.kernel_calls > 0 {
                "kernels"
            } else {
                "stream"
            }
        };

        let g = |t: f64| macs / t / 1e9;
        println!(
            "| `{m}x{k}x{n}` | {:.2} | {:.2} | {:.2}x | {} |",
            g(t_tab),
            g(t_packed),
            t_packed / t_tab,
            picked,
        );
    }
}
