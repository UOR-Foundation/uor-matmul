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

use uor_matmul_codec::{e8_codec, e8_table, Codec, CodedMatrix, Enumerable, Grid, Packed, Sign};
use uor_matmul_core::{
    as_alphabet, as_alphabet_full, AccOf, Accumulator, Alphabet, Bnd, Bound, EncodeMode, Full,
    MatView, MatViewMut, Shape, Traversal, Triple,
};
use uor_matmul_gemm::{
    gemm, gemm_packed, gemm_tabulated, gemm_tabulated_counted, suggested_scratch,
    suggested_tabulation, suggested_tabulation_index, suggested_tabulation_lanes,
    suggested_tabulation_panel, Census, Collapse, GemmOptions, Linear, Scratch, TabulatedTriple,
    Tabulation,
};

type Book<'a> = uor_matmul_codec::Book<'a, i8, Full<i8>, 256, 8>;

/// The sign tier's composition spelling (`CK-10`): one-bit codes, eight to a
/// byte, over a two-entry table. Its code space and block are `Book<256,8>`'s
/// numbers --- and the dedicated `Sign` tier's (`CK-11`).
type PackedSign<'a, Bd> = Packed<Grid<'a, i8, Bd, 2>, 8>;

/// One timed product, with every buffer the traversal may take.
type Run<'a> = dyn FnMut(
        bool,
        &mut Vec<i32>,
        &mut Vec<Alphabet<i8, Full<i8>>>,
        &mut Vec<AccOf<i8>>,
        &mut Vec<i64>,
        &mut Vec<usize>,
    ) + 'a;

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

    degeneracy(&book, space, block);
    sign_stream(book);

    for &(m, k, n) in &[
        (1usize, 1024usize, 4096usize),
        // The same `m` and `k` at twice the width. The build is `k/block * S *
        // block * rows` and does not move with `n`, so the pair separates it
        // from the column loop, which does.
        (1, 1024, 8192),
        (8, 1024, 4096),
        (64, 1024, 4096),
        (64, 4096, 4096),
        (256, 1024, 4096),
        (64, 1024, 16384),
        // Shapes that divide nothing: a ragged row tile, a prime column count,
        // and a depth that is one block past a round number. A traversal that had
        // a cliff at an awkward size would show it here.
        (1000, 512, 512),
        (1, 8192, 1),
        (3, 1024, 4093),
        (17, 1032, 1021),
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
        // Enough for the decoded operand, so the tile-kernel route is available
        // where the table declines. A caller who cannot afford this gets the
        // streaming traversal instead, and the `forced stream` column is what that
        // costs.
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
                        options(Traversal::Blocked),
                        &mut Scratch::with_accumulators(lanes, accumulators),
                        &mut Tabulation::with_index(words, ids),
                        &mut Collapse::none(),
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
                    &mut Collapse::none(),
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
            picked,
        );
    }
}

/// Throughput against the number of *distinct* columns the coded operand has.
///
/// The table charges per distinct code; this charges per distinct column. Column
/// `j` repeats column `j % d`, which is the shape a weight matrix has when its
/// outputs share a codeword run. `d = n` is the case the collapse exists for and
/// does not get, and it says what the question cost.
///
/// Every figure is `open`, and the answer is checked to be the same bytes the
/// uncollapsed traversal gives at every degeneracy.
fn degeneracy(book: &Book<'_>, space: usize, block: usize) {
    let (m, k, n) = (16usize, 1024usize, 4096usize);
    let macs = (m * k * n) as f64;
    let shape = Shape { m, k, n };
    let blocks = k / block;
    let a: Vec<i8> = fill(m * k, 0xa11)
        .into_iter()
        .map(|x| ((x % 255) as i64 - 127) as i8)
        .collect();

    println!();
    println!("## Distinct columns");
    println!();
    println!("| `d` | degeneracy | collapsed | uncollapsed | narrow block | speedup |");
    println!("| --- | --- | --- | --- | --- | --- |");

    let mut d = 1usize;
    loop {
        let base: Vec<u16> = fill(d * blocks, 0xc01)
            .into_iter()
            .map(|x| (x % 256) as u16)
            .collect();
        let stream: Vec<u16> = (0..n * blocks)
            .map(|x| base[(x / blocks % d) * blocks + x % blocks])
            .collect();
        let w = CodedMatrix::new(*book, n, k, &stream).expect("the codes describe n x k");

        let offer = suggested_tabulation::<i8, Full<i8>>(shape, space, block);
        let mut accumulators = vec![<AccOf<i8> as Accumulator>::ZERO; offer];
        // Half the suggestion narrows the column block below the output width,
        // which is where `CD-14`'s block-local collapse is the only collapse.
        let mut accumulators_half = vec![<AccOf<i8> as Accumulator>::ZERO; offer / 2];
        let mut words = vec![0i64; suggested_tabulation_lanes::<i8, Full<i8>>(shape, space, block)];
        let mut ids = vec![0usize; suggested_tabulation_index(shape)];
        let mut lanes =
            vec![Alphabet::<i8, Full<i8>>::ZERO; suggested_tabulation_panel(space, block)];
        let mut c = vec![0i32; m * n];

        let mut run = |collapse: bool,
                       c: &mut Vec<i32>,
                       lanes: &mut Vec<Alphabet<i8, Full<i8>>>,
                       accumulators: &mut Vec<AccOf<i8>>,
                       words: &mut Vec<i64>,
                       ids: &mut Vec<usize>| {
            let av = MatView::row_major(as_alphabet_full(&a), m, k).unwrap();
            let cv = MatViewMut::row_major(c, m, n).unwrap();
            let mut tr = TabulatedTriple::new(av, w, cv).unwrap();
            let mut none: Vec<usize> = Vec::new();
            gemm_tabulated(
                &mut tr,
                &Linear::OVERWRITE,
                GemmOptions {
                    traversal: Traversal::Tabulated,
                    encode: EncodeMode::Wrapping,
                    ..Default::default()
                },
                &mut Scratch::with_accumulators(lanes, accumulators),
                &mut if collapse {
                    Tabulation::with_index(words, ids)
                } else {
                    Tabulation::with_index(words, &mut none)
                },
                &mut Collapse::none(),
            );
        };

        let time = |collapse: bool,
                    c: &mut Vec<i32>,
                    lanes: &mut Vec<Alphabet<i8, Full<i8>>>,
                    accumulators: &mut Vec<AccOf<i8>>,
                    words: &mut Vec<i64>,
                    ids: &mut Vec<usize>,
                    run: &mut Run<'_>|
         -> f64 {
            run(collapse, c, lanes, accumulators, words, ids);
            let s = Instant::now();
            for _ in 0..3 {
                run(collapse, c, lanes, accumulators, words, ids);
            }
            s.elapsed().as_secs_f64() / 3.0
        };

        let t_on = time(
            true,
            &mut c,
            &mut lanes,
            &mut accumulators,
            &mut words,
            &mut ids,
            &mut run,
        );
        let collapsed = c.clone();
        let t_off = time(
            false,
            &mut c,
            &mut lanes,
            &mut accumulators,
            &mut words,
            &mut ids,
            &mut run,
        );
        assert_eq!(
            collapsed, c,
            "the collapse must not change a byte at d = {d}"
        );

        let t_narrow = time(
            true,
            &mut c,
            &mut lanes,
            &mut accumulators_half,
            &mut words,
            &mut ids,
            &mut run,
        );
        assert_eq!(
            collapsed, c,
            "the collapse must not change a byte at a narrowed column block, d = {d}"
        );

        let g = |t: f64| macs / t / 1e9;
        println!(
            "| {d} | {:.0}x | {:.2} | {:.2} | {:.2} | {:.2}x |",
            n as f64 / d as f64,
            g(t_on),
            g(t_off),
            g(t_narrow),
            t_off / t_on,
        );
        if d >= n {
            break;
        }
        d = (d * 8).min(n);
    }
}

/// The sign composition against the index stream it cannot spell --- and the
/// dedicated tier that can.
///
/// `Packed<Grid<2>,8>` is the sign tier's composition spelling (`CK-10`): a
/// code space of 256 and a block of 8, `Book<256,8>`'s numbers exactly. The
/// one gather-path difference is `Enumerable::as_index_stream`, which `Book`
/// answers and `Packed` cannot --- a packed byte's index is a mixed-radix
/// decomposition, not the byte --- so the composition pays one `index_of`
/// pass over its codes where the book reads the operand's own memory. The
/// dedicated `Sign` tier (`CK-11`) spells the same decode with the `u16` code
/// *being* the index, so it borrows exactly as the book does. This sweep
/// prices both differences, at the one-row and small tiles where the index
/// stream was documented to matter.
///
/// Four columns per shape. The book is the index-stream path. The sign
/// composition at `Full<i8>` keeps the general build, so its ratio against the
/// book isolates the gather. The `Sign<8>` column is the same decode through
/// the dedicated tier: its ratio against the book prices everything *but* the
/// borrowed stream (a wider code word, a table of sign flips against the E8
/// lattice's), and its ratio against the composition is what the missing
/// stream was worth. The sign composition at `Bnd<1>` --- activations and
/// weights both in `{-1,+1}`, so the bound-1 build is the admissible one ---
/// is the tier as it stands, and its ratio against the `Full` column is what
/// the adds-only build is worth on the clock. The census column is the build's
/// multiplies, which is the same fact counted rather than timed.
///
/// Every figure is `open`. Each side is asserted against its own dense
/// reference once per shape, and the census is asserted to have read a table,
/// so a silently declined table fails the sweep rather than misreporting it.
fn sign_stream(book: Book<'_>) {
    let sign_full_table: [Alphabet<i8, Full<i8>>; 2] = [Alphabet::of(-1), Alphabet::of(1)];
    let sign_full: PackedSign<'_, Full<i8>> =
        Packed::<_, 8>::new(Grid::new(&sign_full_table)).expect("8 divides 8");
    let sign_one_table: [Alphabet<i8, Bnd<1>>; 2] = [
        Alphabet::new(-1).expect("|-1| <= 1"),
        Alphabet::new(1).expect("|1| <= 1"),
    ];
    let sign_b1: PackedSign<'_, Bnd<1>> =
        Packed::<_, 8>::new(Grid::new(&sign_one_table)).expect("8 divides 8");
    let tier = Sign::<i8, Full<i8>, 8>::new().expect("the full alphabet admits +-1");

    println!();
    println!("## The sign composition against the index stream");
    println!();
    println!("Gmac/s, `Traversal::Tabulated` at the full offer. `sign` is");
    println!("`Packed<Grid<2>,8>`: at `Full<i8>` with the general build, and at");
    println!("`Bnd<1>` with the adds-only build. `Sign<8>` is the dedicated tier");
    println!("(`CK-11`): the same decode, the code being the index. `build mul` is");
    println!("the census's multiply count for the `Full` column against the");
    println!("`Bnd<1>` one.");
    println!();
    println!("| `m x k x n` | `Book<256,8>` | sign, `Full` | `Sign<8>` | sign, `Bnd<1>` | sign/book | tier/book | b1/book | b1/full | build mul |");
    println!("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |");

    for &(m, k, n) in &[
        (1usize, 1024usize, 1024usize),
        (1, 1024, 4096),
        (1, 4096, 1024),
        (1, 4096, 4096),
        (4, 1024, 1024),
        (4, 1024, 4096),
        (4, 4096, 1024),
        (4, 4096, 4096),
        (16, 1024, 1024),
        (16, 1024, 4096),
        (16, 4096, 1024),
        (16, 4096, 4096),
    ] {
        let macs = (m * k * n) as f64;
        let blocks = k / 8;
        let a: Vec<i8> = fill(m * k, 0xa11)
            .into_iter()
            .map(|x| ((x % 255) as i64 - 127) as i8)
            .collect();
        // At `Bnd<1>` the activations are signs too: the adds-only build is
        // admissible only when both operands are.
        let a1: Vec<i8> = fill(m * k, 0xac7)
            .into_iter()
            .map(|x| if x & 1 == 0 { -1 } else { 1 })
            .collect();
        let book_codes: Vec<u16> = fill(n * blocks, 0xb00c)
            .into_iter()
            .map(|x| (x % 256) as u16)
            .collect();
        let sign_codes: Vec<u8> = fill(n * blocks, 0x516)
            .into_iter()
            .map(|x| x as u8)
            .collect();
        // The tier's stream is the composition's, zero-extended: the same
        // bits, spelling the same decoded operand.
        let tier_codes: Vec<u16> = sign_codes.iter().map(|&b| u16::from(b)).collect();

        let (t_book, _) = side(book, &book_codes, &a, m, k, n);
        let (t_sign, sign_census) = side(sign_full, &sign_codes, &a, m, k, n);
        let (t_tier, _) = side(tier, &tier_codes, &a, m, k, n);
        let (t_b1, b1_census) = side(sign_b1, &sign_codes, &a1, m, k, n);

        let g = |t: f64| macs / t / 1e9;
        println!(
            "| `{m}x{k}x{n}` | {:.2} | {:.2} | {:.2} | {:.2} | {:.2}x | {:.2}x | {:.2}x | {:.2}x | {} -> {} |",
            g(t_book),
            g(t_sign),
            g(t_tier),
            g(t_b1),
            t_book / t_sign,
            t_book / t_tier,
            t_book / t_b1,
            t_sign / t_b1,
            sign_census.multiplies,
            b1_census.multiplies,
        );
    }
}

/// One side of the sign sweep: a forced tabulated run, best of a 0.40 s
/// budget, asserted against the dense driver's bytes on the same operands.
fn side<Bd: Bound, C: Enumerable<i8, Bd> + Copy>(
    codec: C,
    codes: &[C::Code],
    a: &[i8],
    m: usize,
    k: usize,
    n: usize,
) -> (f64, Census) {
    let shape = Shape { m, k, n };
    let space = <C as Enumerable<i8, Bd>>::CODE_SPACE;
    let block = <C as Codec<i8, Bd>>::MAX_BLOCK;
    let w = CodedMatrix::new(codec, n, k, codes).expect("the codes describe n x k");

    // The dense reference, once: the same weights decoded, through the driver
    // whose bytes every traversal in this library is measured against.
    let mut b = vec![0i8; k * n];
    for p in 0..k {
        for j in 0..n {
            b[p * n + j] = w.at(j, p).get();
        }
    }
    let want = {
        let mut c = vec![0i32; m * n];
        let av = MatView::row_major(
            as_alphabet::<i8, Bd>(a).expect("the activations fit the declared bound"),
            m,
            k,
        )
        .unwrap();
        let bv = MatView::row_major(
            as_alphabet::<i8, Bd>(&b).expect("decoded weights are in the alphabet by construction"),
            k,
            n,
        )
        .unwrap();
        let cv = MatViewMut::row_major(&mut c, m, n).unwrap();
        let mut tr = Triple::new(av, bv, cv).unwrap();
        gemm(
            &mut tr,
            &Linear::OVERWRITE,
            GemmOptions {
                traversal: Traversal::Blocked,
                encode: EncodeMode::Wrapping,
                ..Default::default()
            },
            &mut Scratch::none(),
        );
        c
    };

    let mut accumulators =
        vec![<AccOf<i8> as Accumulator>::ZERO; suggested_tabulation::<i8, Bd>(shape, space, block)];
    let mut words = vec![0i64; suggested_tabulation_lanes::<i8, Bd>(shape, space, block)];
    let mut ids = vec![0usize; suggested_tabulation_index(shape)];
    // The panel offer is the table's own, so the tile-kernel route cannot
    // answer for the table and the `Tabulated` request is what runs.
    let mut lanes = vec![Alphabet::<i8, Bd>::ZERO; suggested_tabulation_panel(space, block)];
    let mut c = vec![0i32; m * n];

    let t = {
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
                let av = MatView::row_major(
                    as_alphabet::<i8, Bd>(a).expect("the activations fit the declared bound"),
                    m,
                    k,
                )
                .unwrap();
                let cv = MatViewMut::row_major(c, m, n).unwrap();
                let mut tr = TabulatedTriple::new(av, *w, cv).unwrap();
                gemm_tabulated(
                    &mut tr,
                    &Linear::OVERWRITE,
                    GemmOptions {
                        traversal: Traversal::Tabulated,
                        encode: EncodeMode::Wrapping,
                        ..Default::default()
                    },
                    &mut Scratch::with_accumulators(lanes, accumulators),
                    &mut Tabulation::with_index(words, ids),
                    &mut Collapse::none(),
                );
            }
            s.elapsed().as_secs_f64()
        })
    };
    assert_eq!(c, want, "the table must give the dense driver's bytes");

    let mut census = Census::default();
    let mut out = vec![0i32; m * n];
    {
        let av = MatView::row_major(
            as_alphabet::<i8, Bd>(a).expect("the activations fit the declared bound"),
            m,
            k,
        )
        .unwrap();
        let cv = MatViewMut::row_major(&mut out, m, n).unwrap();
        let mut tr = TabulatedTriple::new(av, w, cv).unwrap();
        gemm_tabulated_counted(
            &mut tr,
            &Linear::OVERWRITE,
            GemmOptions {
                traversal: Traversal::Tabulated,
                encode: EncodeMode::Wrapping,
                ..Default::default()
            },
            &mut Scratch::with_accumulators(&mut lanes, &mut accumulators),
            &mut Tabulation::with_index(&mut words, &mut ids),
            &mut Collapse::none(),
            &mut census,
        );
    }
    assert!(
        census.table_reads > 0,
        "the offer was sized for a table and none was read"
    );
    (t, census)
}
