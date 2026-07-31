//! The `CK-14` symbol corpus: a committed f32 weight artifact whose values are
//! drawn from at most 256 distinct bit patterns --- the realistic case the
//! `u8` arena width exists for, a dequantized 8-bit artifact.
//!
//! The artifacts are committed with per-file SHA-256 recorded in
//! `oracles/symbols/manifest.json`, and this harness verifies each digest
//! before believing the bytes, as the NumPy oracle's does. What it then
//! asserts is the *fitting*: the artifact's distinct patterns, canonicalized,
//! are the committed codebook; the committed dense spelling is the codebook
//! read through the codes; and a product driven by the `u8` codes gives the
//! dense float driver's bytes.

use std::path::PathBuf;

use uor_matmul_codec::{canonicalize, Arena, CodedMatrix};
use uor_matmul_core::{as_alphabet_whole, Alphabet, MatView, MatViewMut, Triple, Whole};
use uor_matmul_gemm::{
    gemm_float, gemm_tabulated, Collapse, GemmOptions, Linear, Scratch, TabulatedTriple, Tabulation,
};
use uor_matmul_validate::corpus::sha256_hex;

fn corpus_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/uor-matmul-validate is two below the root")
        .join("oracles/symbols")
}

fn read_f32(path: &std::path::Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("four bytes")))
        .collect()
}

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

#[test]
fn the_committed_symbol_corpus_fits_a_byte() {
    let dir = corpus_dir();
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).expect("manifest"))
            .expect("manifest parses");
    assert_eq!(manifest["id"], "CK-14");
    assert_eq!(manifest["codebook_symbols"], 256);

    // R11: every artifact is checked against its recorded digest before it is
    // believed.
    let codebook_path = dir.join("codebook.f32.bin");
    assert_eq!(
        format!(
            "sha256:{}",
            sha256_hex(&std::fs::read(&codebook_path).unwrap())
        ),
        manifest["codebook_sha256"].as_str().unwrap(),
        "the codebook has drifted from its recorded digest"
    );
    let codebook = read_f32(&codebook_path);
    assert_eq!(codebook.len(), 256);
    // Canonical order is the unsigned order on bit patterns (CK-10).
    assert!(
        codebook.windows(2).all(|w| w[0].to_bits() < w[1].to_bits()),
        "the committed codebook is in canonical order"
    );

    for case in manifest["cases"].as_array().expect("cases") {
        let stem = case["stem"].as_str().expect("stem");
        let n = case["shape"]["n"].as_u64().unwrap() as usize;
        let k = case["shape"]["k"].as_u64().unwrap() as usize;
        let codes_path = dir.join(format!("{stem}.codes.bin"));
        let weights_path = dir.join(format!("{stem}.weights.f32.bin"));
        for (field, path) in [
            ("codes_sha256", &codes_path),
            ("weights_sha256", &weights_path),
        ] {
            assert_eq!(
                format!("sha256:{}", sha256_hex(&std::fs::read(path).unwrap())),
                case[field].as_str().unwrap(),
                "{} has drifted from its recorded digest",
                path.display()
            );
        }
        let codes = std::fs::read(&codes_path).unwrap();
        let mut weights = read_f32(&weights_path);
        assert_eq!(codes.len(), n * k);
        assert_eq!(weights.len(), n * k);

        // The fitting, demonstrated rather than described: the artifact's
        // distinct bit patterns, canonicalized, *are* the committed codebook.
        let distinct = canonicalize(&mut weights);
        assert_eq!(distinct, 256, "{stem}: the artifact fills the byte exactly");
        assert!(
            weights[..distinct]
                .iter()
                .zip(&codebook)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "{stem}: canonicalizing the artifact reproduces the codebook"
        );

        // And the committed dense spelling is the tier's decode written out:
        // the codebook read through the codes, element for element.
        let mut decoded = read_f32(&weights_path);
        for (at, w) in decoded.iter_mut().enumerate() {
            *w = codebook[codes[at] as usize];
        }
        let committed = read_f32(&weights_path);
        assert!(
            decoded
                .iter()
                .zip(&committed)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "{stem}: the dense spelling is the codebook read through the codes"
        );

        // The fixture's product: the `u8` codes drive the tabulated traversal
        // (which declines the one-element block to the dense route, or streams
        // it), and the bytes are the dense float driver's over the committed
        // dense spelling. `CD-18` sweeps shapes and offers; this asserts the
        // committed artifact takes the same identity.
        let table: &[Alphabet<f32, Whole<f32>>; 256] =
            as_alphabet_whole(&codebook).try_into().unwrap();
        for m in [1usize, 3] {
            let a: Vec<f32> = fill(m * k, 0xa11, |x| {
                (x % 2048) as f32 * 2.0f32.powi((x % 19) as i32 - 9)
            });
            // The dense reference: `B` is `W^T`, and the corpus stores `W`
            // `n x k`, so `B[p][j]` is `weights[j * k + p]`.
            let b: Vec<f32> = (0..k * n)
                .map(|at| committed[(at % n) * k + at / n])
                .collect();
            let mut want = vec![0.0f32; m * n];
            {
                let av = MatView::row_major(&a, m, k).unwrap();
                let bv = MatView::row_major(&b, k, n).unwrap();
                let cv = MatViewMut::row_major(&mut want, m, n).unwrap();
                let mut t = Triple::new(av, bv, cv).unwrap();
                gemm_float(&mut t, &Linear::OVERWRITE, GemmOptions::default());
            }
            let mut got = vec![0.0f32; m * n];
            {
                let av = MatView::row_major(as_alphabet_whole(&a), m, k).unwrap();
                let cv = MatViewMut::row_major(&mut got, m, n).unwrap();
                let w = CodedMatrix::new(Arena::new(table), n, k, &codes)
                    .expect("the codes describe n x k");
                let mut tr = TabulatedTriple::new(av, w, cv).unwrap();
                let mut panel = vec![Alphabet::<f32, Whole<f32>>::ZERO; k];
                gemm_tabulated(
                    &mut tr,
                    &Linear::OVERWRITE,
                    GemmOptions::default(),
                    &mut Scratch::new(&mut panel),
                    &mut Tabulation::none(),
                    &mut Collapse::none(),
                );
            }
            assert!(
                want.iter()
                    .zip(&got)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{stem} m={m}: the coded product must be the dense driver's bytes"
            );
        }
    }
}
