//! `CG-14` (open): achieved bytes per second for the `u8`-symbol-coded gemv
//! and skinny GEMM, against the host's STREAM number measured in this same
//! harness.
//!
//! The bandwidth floor binds where arithmetic intensity is O(1) ---
//! matrix-vector and skinny GEMM --- so that is the whole sweep. The symbol
//! path reads one byte per weight where the dense driver reads four; whether
//! the machine pays that back as throughput is the measurement, and either
//! answer is the finding: near the bus limit means residency was the
//! bottleneck, far below it means decode latency is.
//!
//! Every figure is `open`: printed, never asserted. What *is* asserted,
//! inside each timed run, is byte-identity with the dense float driver --- a
//! speed measured on the wrong bytes is not a measurement --- and the census,
//! which says which factorization ran rather than asking the predicate twice.
//!
//! Ignored by default, like the other minute-long sweeps: `just
//! symbol-bandwidth` runs it, in release, where a throughput figure means
//! something.

use std::time::Instant;

use uor_matmul_codec::{Arena, CodedMatrix};
use uor_matmul_core::{
    as_alphabet_whole, Alphabet, MatView, MatViewMut, PackedCode, Triple, Whole,
};
use uor_matmul_gemm::{
    gemm_float_packed, gemm_tabulated_counted, Census, Collapse, GemmOptions, Linear, Scratch,
    TabulatedTriple, Tabulation,
};

/// Deterministic fill, the same recorded generator the other harnesses use.
fn fill<T, F: Fn(u64) -> T>(len: usize, salt: u64, map: F) -> Vec<T> {
    let mut s = 0x243F_6A88_85A3_08D3u64 ^ salt;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            map(s >> 33)
        })
        .collect()
}

/// Best of as many repetitions as fit a fixed budget, in seconds per call.
///
/// The minimum is the run least interfered with; the budget is wall-clock so a
/// fast point and a slow one both get enough samples for the minimum to mean
/// something. The same discipline `oracle_sweep` uses.
fn best(mut run: impl FnMut()) -> f64 {
    const BUDGET: f64 = 0.35;
    let mut best = f64::INFINITY;
    let mut spent = 0.0;
    loop {
        let s = Instant::now();
        run();
        let t = s.elapsed().as_secs_f64();
        best = best.min(t);
        spent += t;
        if spent >= BUDGET {
            return best;
        }
    }
}

/// The host's STREAM number: a triad over arrays far past the last cache.
///
/// `a[i] = b[i] + 3 * c[i]` over three arrays of 2^25 `f32` --- 384 MiB in
/// all, past any cache this host has --- best of ten. Bytes are counted as 12
/// per element: two reads and the write. The write-allocate read is the
/// machine's own traffic, stated here rather than counted, so the figure is
/// the demand bandwidth the operand streams below are compared against.
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
struct Path {
    secs: f64,
    bytes: f64,
}

/// The shapes the floor binds on: matrix-vector both ways around, a deep thin
/// product, and two skinny GEMMs.
const SHAPES: &[(usize, usize, usize)] = &[
    (1024, 1024, 1),
    (1, 1024, 1024),
    (1, 1_048_576, 1),
    (2048, 8, 2048),
    (8, 262_144, 8),
];

#[test]
#[ignore = "a minutes-long release-mode sweep: `just symbol-bandwidth`"]
fn symbol_coded_gemv_sits_against_the_bus_cg_14() {
    // The committed corpus's codebook: 256 distinct f32 bit patterns in
    // canonical arena order, the realistic case this width exists for. Its
    // digest is verified by the symbol-corpus harness; here it is simply the
    // table, because a measurement is not the place to re-litigate a digest.
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
    println!();
    println!("# CG-14 (open): achieved bytes/second, symbol-coded f32 against the bus");
    println!(
        "# host: {}-{}; STREAM triad a[i]=b[i]+3*c[i], 3 x 2^25 f32 (384 MiB), best of 10,",
        std::env::consts::ARCH,
        std::env::consts::OS
    );
    println!(
        "#   12 bytes counted per element (two reads, one write): {:.2} GB/s",
        stream / 1e9
    );
    println!(
        "# bytes a path is charged: A (m*k*4) + stored W (n*k codes, or k*n*4 dense) + C (m*n*4),"
    );
    println!("#   plus the 1 KiB codebook on the symbol path; the decode panel is cache traffic, not bus");
    println!("# byte-identity with the dense float driver is asserted inside every timed run;");
    println!("#   the census printed per shape says which factorization ran");
    println!();
    println!(
        "{:>15} {:>10} {:>13} {:>13} {:>13} {:>13}",
        "m x k x n", "W stored", "sym walk", "sym panel", "uor f32", "matrixmultiply"
    );
    println!(
        "{:>15} {:>10} {:>13} {:>13} {:>13} {:>13}",
        "", "bytes", "GB/s (%bus)", "GB/s (%bus)", "GB/s (%bus)", "GB/s (%bus)"
    );

    for &(m, k, n) in SHAPES {
        // The operands. `A` spans a decade of exponents, so the complete
        // accumulator's limb window has to move; the codes are the whole byte,
        // so every symbol is live. `W` is `n x k`, one coded row per output
        // column --- the orientation the tabulated traversal takes.
        let a: Vec<f32> = fill(m * k, 5, |x| {
            (x % 2048) as f32 * 2.0f32.powi((x % 19) as i32 - 9)
        });
        let codes: Vec<u8> = fill(n * k, 6, |x| x as u8);
        // The dense spelling, `k x n`, for the dense driver and the oracle.
        let b: Vec<f32> = (0..k * n)
            .map(|at| codebook[codes[(at % n) * k + at / n] as usize])
            .collect();
        let w =
            CodedMatrix::new(Arena::new(table), n, k, &codes).expect("the codes describe n x k");

        // The reference bytes, computed once, untimed, by the dense float
        // driver over the dense spelling. What the timed assertions check
        // against; that the value is the exact sum is `CU-04`'s business.
        let mut pa = vec![PackedCode::default(); k];
        let mut pb = vec![PackedCode::default(); k * n];
        let mut want = vec![0.0f32; m * n];
        {
            let av = MatView::row_major(&a, m, k).unwrap();
            let bv = MatView::row_major(&b, k, n).unwrap();
            let cv = MatViewMut::row_major(&mut want, m, n).unwrap();
            let mut t = Triple::new(av, bv, cv).unwrap();
            gemm_float_packed(
                &mut t,
                &Linear::OVERWRITE,
                GemmOptions::default(),
                &mut pa,
                &mut pb,
            );
        }

        let mut paths: Vec<Path> = Vec::new();

        // The symbol path, at the two offers a bandwidth-seeking caller has:
        // nothing, so the traversal walks the codes and decodes per element;
        // and one decoded row, so each weight is decoded once and read from
        // cache by every row of `A`. Both decline the table --- a one-element
        // block can never pay --- and use the non-table dense stream. The
        // latter's family-owned kernel calls are counted as opaque work rather
        // than guessed at as synthetic multiplies.
        for (label, panel_offer) in [("sym walk", 0usize), ("sym panel", k)] {
            let mut c = vec![0.0f32; m * n];
            let mut panel = vec![Alphabet::<f32, Whole<f32>>::ZERO; panel_offer];
            let mut census = Census::default();
            let run =
                |c: &mut Vec<f32>, panel: &mut [Alphabet<f32, Whole<f32>>], census: &mut Census| {
                    let av = MatView::row_major(as_alphabet_whole(&a), m, k).unwrap();
                    let cv = MatViewMut::row_major(c, m, n).unwrap();
                    let mut tr = TabulatedTriple::new(av, w, cv).unwrap();
                    gemm_tabulated_counted(
                        &mut tr,
                        &Linear::OVERWRITE,
                        GemmOptions::default(),
                        &mut Scratch::new(panel),
                        &mut Tabulation::none(),
                        &mut Collapse::none(),
                        census,
                    );
                };
            run(&mut c, &mut panel, &mut census);
            // The census of one run, read before the timed reps accumulate
            // into it: the streamed symbol route, and nothing else.
            assert!(
                want.iter()
                    .zip(c.iter())
                    .all(|(w, g)| w.to_bits() == g.to_bits()),
                "{label} {m}x{k}x{n}: the symbol run must give the dense driver's bytes"
            );
            assert_eq!(census.table_reads, 0, "{label}: no table fits, so none ran");
            assert!(
                census.kernel_calls > 0,
                "{label}: the non-table dense stream must issue a kernel call"
            );
            assert_eq!(
                census.multiplies,
                0,
                "{label}: opaque dense kernels do not invent a multiply count ({census:?})"
            );
            eprintln!(
                "# {label} {m}x{k}x{n} census: decodes {} (n*k = {}), multiplies {}",
                census.decodes,
                n * k,
                census.multiplies
            );
            let t = best(|| run(&mut c, &mut panel, &mut census));
            assert!(
                want.iter()
                    .zip(c.iter())
                    .all(|(w, g)| w.to_bits() == g.to_bits()),
                "{label} {m}x{k}x{n}: the timed symbol run must give the dense driver's bytes"
            );
            paths.push(Path {
                secs: t,
                bytes: (m * k * 4 + n * k + m * n * 4 + 1024) as f64,
            });
        }

        // The dense float driver over the dense spelling: the same exact
        // accumulation reading four bytes a weight.
        let mut c = vec![0.0f32; m * n];
        let run = |c: &mut Vec<f32>, pa: &mut [PackedCode], pb: &mut [PackedCode]| {
            let av = MatView::row_major(&a, m, k).unwrap();
            let bv = MatView::row_major(&b, k, n).unwrap();
            let cv = MatViewMut::row_major(c, m, n).unwrap();
            let mut t = Triple::new(av, bv, cv).unwrap();
            gemm_float_packed(&mut t, &Linear::OVERWRITE, GemmOptions::default(), pa, pb);
        };
        run(&mut c, &mut pa, &mut pb);
        let t = best(|| run(&mut c, &mut pa, &mut pb));
        assert!(
            want.iter()
                .zip(c.iter())
                .all(|(w, g)| w.to_bits() == g.to_bits()),
            "uor f32 {m}x{k}x{n}: the timed dense run must give its own first bytes"
        );
        paths.push(Path {
            secs: t,
            bytes: (m * k * 4 + k * n * 4 + m * n * 4) as f64,
        });

        // The oracle, on the same dense spelling. Inexact by its own
        // admission (`CX-05`), so its bytes are not asserted --- only its
        // length, which is what a timed region needs to prove it ran.
        #[cfg(feature = "ref-matrixmultiply")]
        {
            use uor_matmul_validate::oracle::{FloatOracle, MatrixMultiply};
            let t = best(|| {
                let c = MatrixMultiply::product_f32(m, k, n, &a, &b);
                assert_eq!(c.len(), m * n);
                std::hint::black_box(&c);
            });
            paths.push(Path {
                secs: t,
                bytes: (m * k * 4 + k * n * 4 + m * n * 4) as f64,
            });
        }

        let mut line = format!("{:>15} {:>10}", format!("{m}x{k}x{n}"), n * k);
        for p in &paths {
            let gbs = p.bytes / p.secs / 1e9;
            line.push_str(&format!(
                " {:>13}",
                format!("{gbs:.2} ({:.0}%)", 100.0 * gbs * 1e9 / stream)
            ));
        }
        println!("{line}");
        let macs = (m as f64) * (k as f64) * (n as f64);
        let mut nominal = format!("{:>15} {:>10}", "Gmac/s", "");
        for p in &paths {
            nominal.push_str(&format!(" {:>13}", format!("{:.3}", macs / p.secs / 1e9)));
        }
        println!("{nominal}");
    }
}
