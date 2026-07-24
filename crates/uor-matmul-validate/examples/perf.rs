//! A quick throughput probe, for development.
//!
//! Not a claim: `CG-*` and `just scaling` are where the measured figures live,
//! with confidence intervals. This is for looking at a change.

/// Probe every path.
fn main() {
    use std::time::Instant;
    use uor_matmul::prelude::*;
    use uor_matmul_core::EncodeMode;

    // Dense i8 -> i32, generic driver vs kernel-driven.
    for n in [64usize, 128, 256] {
        let a = vec![1i8; n * n];
        let b = vec![1i8; n * n];
        let mut c = vec![0i32; n * n];
        let mut scratch = vec![Alphabet::<i8, Full<i8>>::ZERO; 1 << 16];

        let t0 = Instant::now();
        {
            let av = MatView::row_major(as_alphabet_full(&a), n, n).unwrap();
            let bv = MatView::row_major(as_alphabet_full(&b), n, n).unwrap();
            let cv = MatViewMut::row_major(&mut c, n, n).unwrap();
            let mut t = Triple::new(av, bv, cv).unwrap();
            gemm(
                &mut t,
                &Linear::OVERWRITE,
                GemmOptions {
                    encode: EncodeMode::Wrapping,
                    ..Default::default()
                },
                &mut Scratch::none(),
            );
        }
        let generic = t0.elapsed().as_secs_f64();

        let t1 = Instant::now();
        {
            let av = MatView::row_major(as_alphabet_full(&a), n, n).unwrap();
            let bv = MatView::row_major(as_alphabet_full(&b), n, n).unwrap();
            let cv = MatViewMut::row_major(&mut c, n, n).unwrap();
            let mut t = Triple::new(av, bv, cv).unwrap();
            uor_matmul::gemm_w8a8(
                &mut t,
                &Linear::OVERWRITE,
                GemmOptions {
                    encode: EncodeMode::Wrapping,
                    ..Default::default()
                },
                &mut Scratch::new(&mut scratch),
            );
        }
        let packed = t1.elapsed().as_secs_f64();
        let gmacs = (n as f64).powi(3) / 1e9;
        println!(
            "dense {n}^3: generic {:.4}s ({:.2} Gmac/s)  packed {:.4}s ({:.2} Gmac/s)",
            generic,
            gmacs / generic,
            packed,
            gmacs / packed
        );
    }

    // Coded: the scaling question.
    let table: [Alphabet<i8, Full<i8>>; 16] = core::array::from_fn(|i| Alphabet::of((i as i8) - 8));
    let grid = Grid::new(&table);
    for n in [64usize, 128, 256] {
        let acts = vec![1i8; n * n];
        let codes: Vec<u16> = vec![9u16; n * n];
        let coded = CodedMatrix::new(grid, n, n, &codes).unwrap();
        let mut c = vec![0i32; n * n];
        let t = Instant::now();
        {
            let av = MatView::row_major(as_alphabet_full(&acts), n, n).unwrap();
            let cv = MatViewMut::row_major(&mut c, n, n).unwrap();
            let mut tr = uor_matmul::CodedTriple::new(av, coded, cv).unwrap();
            uor_matmul::coded_gemm(
                &mut tr,
                &Linear::OVERWRITE,
                GemmOptions {
                    encode: EncodeMode::Wrapping,
                    ..Default::default()
                },
            );
        }
        let e = t.elapsed().as_secs_f64();
        println!(
            "coded {n}^3: {:.4}s ({:.3} Gmac/s)",
            e,
            (n as f64).powi(3) / 1e9 / e
        );
    }

    // Float.
    for n in [32usize, 64, 128] {
        let a = vec![1.0f32; n * n];
        let b = vec![1.0f32; n * n];
        let mut c = vec![0.0f32; n * n];
        let t = Instant::now();
        {
            let av = MatView::row_major(&a, n, n).unwrap();
            let bv = MatView::row_major(&b, n, n).unwrap();
            let cv = MatViewMut::row_major(&mut c, n, n).unwrap();
            let mut tr = Triple::new(av, bv, cv).unwrap();
            uor_matmul::gemm_float(&mut tr, &Linear::OVERWRITE, GemmOptions::default());
        }
        let e = t.elapsed().as_secs_f64();
        println!(
            "f32   {n}^3: {:.4}s ({:.3} Gmac/s)",
            e,
            (n as f64).powi(3) / 1e9 / e
        );
    }
}
