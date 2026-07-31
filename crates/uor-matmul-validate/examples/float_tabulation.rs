//! Does tabulation pay for *floats* at a tiny code space?
//!
//! The shipped float codec --- the arena tier --- is `MAX_BLOCK = 1`, which
//! `tabulation_pays` refuses, so the float table route has never carried a
//! measurement. This sweep gives a float codec a block: `FloatBook<S, BLK>`
//! from [`uor_matmul_validate::float_tab`], dev-only scaffolding with
//! `S in {4, 16}` codewords of `BLK in {4, 8}` `f32` symbols each. At those
//! sizes the slab (`S * rows * size_of::<AccOf<f32>>()`) is a few kilobytes
//! and sits in L1, and the column loop trades one exact float mac per product
//! for one table read and one exact accumulator combine per `BLK` products.
//! Whether that trade wins is the measurement.
//!
//! Three timed routes per shape. `stream` is `gemm_float` over the decoded
//! weights with no panel offer --- exactly what the tabulated driver's dense
//! decline route runs for `f32` ([`Tabulated::dense_gemm`] calls it with the
//! leftover offer unused). `packed` is the same driver with panels, where the
//! placement bridge lives: both panels prescaled to a common base turns the
//! inner loop into an integer dot product. `table` is `Traversal::Tabulated`
//! forced with the full offer; the census asserts the table really ran, and
//! the answer is asserted byte-identical to the streaming driver's at every
//! shape. `picked` is what `Traversal::Blocked`, the default, chose at the
//! same offer, read from the census rather than recomputed from the
//! predicate.
//!
//! Every figure is `open`, measured on the host in the section header.

use std::hint::black_box;
use std::time::Instant;

use uor_matmul_codec::CodedMatrix;
use uor_matmul_core::{
    as_alphabet_whole, AccOf, Accumulator, Alphabet, Element, FloatElement, MatView, MatViewMut,
    PackedCode, Shape, Traversal, Triple, Whole,
};
use uor_matmul_gemm::{
    gemm_float, gemm_float_packed, gemm_tabulated, gemm_tabulated_counted, suggested_tabulation,
    suggested_tabulation_index, suggested_tabulation_lanes, suggested_tabulation_panel, Census,
    Collapse, GemmOptions, Linear, Scratch, TabulatedTriple, Tabulation,
};
use uor_matmul_validate::float_tab::{codebook, FloatBook};

/// Best of a fixed wall-clock budget, the discipline `tabulation_sweep` uses:
/// the minimum is the figure with the least noise in it, and the budget keeps
/// the whole sweep inside a minute.
fn best(mut run: impl FnMut() -> f64) -> f64 {
    const BUDGET: f64 = 0.30;
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

/// Deterministic `f32`s across a few exponents: the exact float mac's cost is
/// the placement into the fixed-point accumulator, and a single-exponent fill
/// would dodge it. The magnitudes stay inside one five-binade band, because
/// the scaled lane admits a panel only where `24 + span <= 31`: a wider fill
/// is declined by design, and this instrument exists to make the table run,
/// not to probe the decline.
fn symbols(len: usize, salt: u64) -> Vec<f32> {
    fill(len, salt)
        .into_iter()
        .map(|x| {
            // [2^-4, 4): exponents -4 through 1, a span of five.
            let mag = 0.0625 + (x % 1000) as f32 / 1000.0 * 3.9375;
            if x & 1 == 0 {
                mag
            } else {
                -mag
            }
        })
        .collect()
}

/// One (S, BLK) configuration across the shape grid.
fn sweep<const S: usize, const BLK: usize>() {
    println!("## `FloatBook<{S}, {BLK}>`");
    println!();
    println!(
        "Gmac/s against the nominal `m * k * n`. Slab at the widest tile: {} bytes.",
        S * 16 * size_of::<AccOf<f32>>()
    );
    println!();
    println!(
        "| `m x k x n` | stream | packed | table | vs stream | vs packed | picked | census adds |"
    );
    println!("| --- | --- | --- | --- | --- | --- | --- | --- |");

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
        let shape = Shape { m, k, n };
        let blocks = k / BLK;

        let table = codebook::<S, BLK>(&symbols(S * BLK, 0xb00c));
        let codec = FloatBook::<'_, S, BLK>::new(&table);
        let codes: Vec<u16> = fill(n * blocks, 0xc0de)
            .into_iter()
            .map(|x| x as u16)
            .collect();
        let w = CodedMatrix::new(codec, n, k, &codes).expect("the codes describe n x k");
        let a = symbols(m * k, 0xa11);

        // The dense reference, once: the same weights decoded, through the
        // driver whose bytes the table must reproduce.
        let mut b = vec![0.0f32; k * n];
        for p in 0..k {
            for j in 0..n {
                b[p * n + j] = w.at(j, p).get();
            }
        }
        let mut dense_bytes = vec![0.0f32; m * n];
        let t_dense = {
            let (a, b, c) = (&a, &b, &mut dense_bytes);
            best(|| {
                let s = Instant::now();
                {
                    let av = MatView::row_major(a, m, k).unwrap();
                    let bv = MatView::row_major(b, k, n).unwrap();
                    let cv = MatViewMut::row_major(c, m, n).unwrap();
                    let mut tr = Triple::new(av, bv, cv).unwrap();
                    gemm_float(&mut tr, &Linear::OVERWRITE, GemmOptions::default());
                }
                s.elapsed().as_secs_f64()
            })
        };
        let want: Vec<u64> = dense_bytes.iter().map(|v| v.symbol_bits()).collect();

        // The dense route a serious caller takes: the same driver with panel
        // offers, which is where the placement bridge (the prescaling of both
        // panels to a common base) lives. `pb` holds a block of columns, so
        // the whole sweep's offer stays in cache-scale memory.
        let mut pa = vec![PackedCode::default(); m * k];
        let mut pb = vec![PackedCode::default(); k * n.min(512)];
        let mut packed_bytes = vec![0.0f32; m * n];
        let t_packed = {
            let (a, b, c, pa, pb) = (&a, &b, &mut packed_bytes, &mut pa, &mut pb);
            best(|| {
                let s = Instant::now();
                {
                    let av = MatView::row_major(a, m, k).unwrap();
                    let bv = MatView::row_major(b, k, n).unwrap();
                    let cv = MatViewMut::row_major(c, m, n).unwrap();
                    let mut tr = Triple::new(av, bv, cv).unwrap();
                    gemm_float_packed(&mut tr, &Linear::OVERWRITE, GemmOptions::default(), pa, pb);
                }
                s.elapsed().as_secs_f64()
            })
        };
        let packed_bits: Vec<u64> = packed_bytes.iter().map(|v| v.symbol_bits()).collect();
        assert_eq!(
            packed_bits, want,
            "`FloatBook<{S}, {BLK}>` at {m}x{k}x{n}: the packed dense route must give \
             the streaming driver's bytes"
        );

        // The forced table, at the offer sized for it and nothing more, so the
        // dense decline route cannot answer for it.
        let mut accumulators = vec![
            <AccOf<f32> as Accumulator>::ZERO;
            suggested_tabulation::<f32, Whole<f32>>(shape, S, BLK,)
        ];
        let mut ids = vec![0usize; suggested_tabulation_index(shape)];
        // The scaled lane's words are `i64`-shaped and do not live in the
        // exact offer: an empty lanes slice declines the table by design.
        let mut lanes = vec![0i64; suggested_tabulation_lanes::<f32, Whole<f32>>(shape, S, BLK)];
        let mut panel = vec![Alphabet::<f32, Whole<f32>>::ZERO; suggested_tabulation_panel(S, BLK)];
        let mut c = vec![0.0f32; m * n];

        let options = |traversal: Traversal| GemmOptions {
            traversal,
            ..Default::default()
        };

        let t_tab = {
            let (a, w, c, panel, accumulators, lanes, ids) = (
                &a,
                &w,
                &mut c,
                &mut panel,
                &mut accumulators,
                &mut lanes,
                &mut ids,
            );
            best(|| {
                let s = Instant::now();
                {
                    let av = MatView::row_major(as_alphabet_whole(a), m, k).unwrap();
                    let cv = MatViewMut::row_major(c, m, n).unwrap();
                    let mut tr = TabulatedTriple::new(av, *w, cv).unwrap();
                    gemm_tabulated(
                        &mut tr,
                        &Linear::OVERWRITE,
                        options(Traversal::Tabulated),
                        &mut Scratch::with_accumulators(panel, accumulators),
                        &mut Tabulation::with_index(lanes, ids),
                        &mut Collapse::none(),
                    );
                }
                s.elapsed().as_secs_f64()
            })
        };
        let got: Vec<u64> = c.iter().map(|v| v.symbol_bits()).collect();
        assert_eq!(
            got, want,
            "`FloatBook<{S}, {BLK}>` at {m}x{k}x{n}: the table must give the dense \
             float driver's bytes"
        );

        // Which factorization ran, counted rather than re-derived --- and what
        // the default traversal would have picked at the same offer.
        let counted = |traversal: Traversal,
                       c: &mut Vec<f32>,
                       panel: &mut Vec<Alphabet<f32, Whole<f32>>>,
                       accumulators: &mut Vec<AccOf<f32>>,
                       lanes: &mut Vec<i64>,
                       ids: &mut Vec<usize>|
         -> Census {
            let mut census = Census::default();
            let av = MatView::row_major(as_alphabet_whole(&a), m, k).unwrap();
            let cv = MatViewMut::row_major(c, m, n).unwrap();
            let mut tr = TabulatedTriple::new(av, w, cv).unwrap();
            gemm_tabulated_counted(
                &mut tr,
                &Linear::OVERWRITE,
                options(traversal),
                &mut Scratch::with_accumulators(panel, accumulators),
                &mut Tabulation::with_index(lanes, ids),
                &mut Collapse::none(),
                &mut census,
            );
            census
        };
        let forced = counted(
            Traversal::Tabulated,
            &mut c,
            &mut panel,
            &mut accumulators,
            &mut lanes,
            &mut ids,
        );
        assert!(
            forced.table_reads > 0,
            "`FloatBook<{S}, {BLK}>` at {m}x{k}x{n}: the offer was sized for a table \
             and none was read ({forced:?})"
        );
        // The default's choice needs the dense route reachable too, so the
        // panel holds the decoded operand for this one run.
        let mut big_panel = vec![
            Alphabet::<f32, Whole<f32>>::ZERO;
            suggested_tabulation_panel(S, BLK).max(n * k + k)
        ];
        let mut out = vec![0.0f32; m * n];
        let declined = counted(
            Traversal::Blocked,
            &mut out,
            &mut big_panel,
            &mut accumulators,
            &mut lanes,
            &mut ids,
        );
        let picked = if declined.table_reads > 0 {
            "table"
        } else if declined.kernel_calls > 0 {
            "dense"
        } else {
            "stream"
        };

        let g = |t: f64| macs / t / 1e9;
        println!(
            "| `{m}x{k}x{n}` | {:.3} | {:.3} | {:.3} | {:.2}x | {:.2}x | {} | {} |",
            g(t_dense),
            g(t_packed),
            g(t_tab),
            t_dense / t_tab,
            t_packed / t_tab,
            picked,
            forced.adds,
        );
    }
    println!();
}

/// The raw op costs the economics turn on, at one representative reduction
/// length. A microbenchmark, documented as such: each side is one primitive
/// in a loop over `k`, not a traversal.
///
/// - `mac` is [`Element::mac`] for `f32`: decode-free here, but the full exact
///   product --- encode the operands to fixed point and place the product into
///   the complete accumulator --- which is what the dense float driver issues
///   once per product.
/// - `gather+add` is the tabulated column loop's step: read one
///   `AccOf<f32>` table entry by code and `combine` it into a running
///   accumulator, covering `BLK` products per step.
fn op_costs() {
    const K: usize = 4096;
    println!("## The op costs");
    println!();
    println!(
        "One primitive per iteration over `k = {K}`, best of the same budget. \
         `AccOf<f32>` is {} bytes on this host.",
        size_of::<AccOf<f32>>()
    );
    println!();
    println!("| op | per step | per product covered |");
    println!("| --- | --- | --- |");

    let a = symbols(K, 0x0a);
    let w = symbols(K, 0x0b);

    let t_mac = {
        let (a, w) = (&a, &w);
        best(|| {
            let s = Instant::now();
            let mut acc = <AccOf<f32> as Accumulator>::ZERO;
            for p in 0..K {
                <f32 as Element>::mac(&mut acc, *black_box(&a[p]), *black_box(&w[p]));
            }
            black_box(acc);
            s.elapsed().as_secs_f64()
        })
    };

    // A table of `S` entries worth gathering: accumulators with real content,
    // so the combine does the same limb work the column loop's does.
    for &(s_log, blk) in &[(4usize, 4usize), (4, 8), (16, 4), (16, 8)] {
        let entries: Vec<AccOf<f32>> = (0..s_log)
            .map(|e| {
                let mut acc = <AccOf<f32> as Accumulator>::ZERO;
                for t in 0..blk {
                    <f32 as Element>::mac(&mut acc, a[(e + t) % K], w[(e * blk + t) % K]);
                }
                acc
            })
            .collect();
        let codes: Vec<u16> = fill(K / blk, 0x9a7).into_iter().map(|x| x as u16).collect();
        let t_gather = {
            let (entries, codes) = (&entries, &codes);
            best(|| {
                let s = Instant::now();
                let mut acc = <AccOf<f32> as Accumulator>::ZERO;
                for &code in codes.iter() {
                    let entry = black_box(entries[(code as usize) & (s_log - 1)]);
                    acc = black_box(acc.combine(entry));
                }
                black_box(acc);
                s.elapsed().as_secs_f64()
            })
        };
        println!(
            "| gather+add, `S = {s_log}`, `BLK = {blk}` | {:.2} ns | {:.3} ns |",
            t_gather / (K / blk) as f64 * 1e9,
            t_gather / K as f64 * 1e9,
        );
    }
    println!(
        "| exact float mac | {:.2} ns | {:.2} ns |",
        t_mac / K as f64 * 1e9,
        t_mac / K as f64 * 1e9,
    );
    println!();
}

fn main() {
    println!("# Float tabulation at a tiny code space");
    println!();
    println!(
        "Host: Apple M4 Max (aarch64), release build. `AccOf<f32>` = {} bytes, \
         so the float lane *is* the complete accumulator.",
        size_of::<AccOf<f32>>()
    );
    println!();

    op_costs();
    sweep::<4, 4>();
    sweep::<4, 8>();
    sweep::<16, 4>();
    sweep::<16, 8>();
}
