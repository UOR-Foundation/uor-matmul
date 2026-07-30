//! `CX-10`: byte equality with NumPy `int64`, out of process.
//!
//! Every other oracle in this repository is a Rust crate. They share a
//! language, a compiler, and in more than one case a kernel author, so
//! agreement among them is weaker evidence than it looks. NumPy shares none of
//! those, which is why it is registered separately and why its outputs are
//! *committed artifacts* rather than a live dependency: the claim is
//! reproducible without a Python toolchain, and `oracles/numpy/generate.py`
//! records exactly how the artifacts were made.
//!
//! `int64` accumulation wraps where a two's complement accumulator does, so
//! this is byte equality with no exempted region (§3.4).

use std::path::PathBuf;

use uor_matmul::prelude::*;
use uor_matmul_core::EncodeMode;
use uor_matmul_validate::corpus::sha256_hex;

fn oracle_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/uor-matmul-validate is two below the root")
        .join("oracles/numpy")
}

fn read_i64(path: &std::path::Path) -> Vec<i64> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().expect("eight bytes")))
        .collect()
}

/// Our `i64` product, encoded with wrapping semantics to match NumPy's `int64`.
fn ours_i64(m: usize, k: usize, n: usize, a: &[i64], b: &[i64]) -> Vec<i64> {
    let mut c = vec![0i64; m * n];
    if m == 0 || n == 0 {
        return c;
    }
    let av = MatView::row_major(as_alphabet_full(a), m, k).unwrap();
    let bv = MatView::row_major(as_alphabet_full(b), k, n).unwrap();
    let cv = MatViewMut::row_major(&mut c, m, n).unwrap();
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
    c
}

/// `CX-10`: byte-identical with NumPy `int64` over every committed case.
#[test]
fn numpy_int64_is_byte_identical_cx_10() {
    let dir = oracle_dir();
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).expect("manifest"))
            .expect("manifest parses");

    assert_eq!(manifest["id"], "CX-10");
    assert_eq!(manifest["dtype"], "int64");
    eprintln!(
        "CX-10: numpy {} via {}",
        manifest["numpy_version"].as_str().unwrap_or("?"),
        manifest["entry"].as_str().unwrap_or("?")
    );

    let cases = manifest["cases"].as_array().expect("cases");
    assert!(!cases.is_empty(), "the oracle must have cases");

    for case in cases {
        let stem = case["stem"].as_str().expect("stem");
        let m = case["shape"]["m"].as_u64().unwrap() as usize;
        let k = case["shape"]["k"].as_u64().unwrap() as usize;
        let n = case["shape"]["n"].as_u64().unwrap() as usize;

        // R11: the artifact is checked against its recorded digest before it is
        // believed. An oracle whose bytes have drifted is not an oracle.
        for (field, name) in [
            ("a_sha256", "a"),
            ("b_sha256", "b"),
            ("expected_sha256", "expected"),
        ] {
            let path = dir.join(format!("{stem}.{name}.bin"));
            let actual = format!("sha256:{}", sha256_hex(&std::fs::read(&path).unwrap()));
            assert_eq!(
                actual,
                case[field].as_str().unwrap(),
                "{} has drifted from its recorded digest",
                path.display()
            );
        }

        let a = read_i64(&dir.join(format!("{stem}.a.bin")));
        let b = read_i64(&dir.join(format!("{stem}.b.bin")));
        let expected = read_i64(&dir.join(format!("{stem}.expected.bin")));

        assert_eq!(
            ours_i64(m, k, n, &a, &b),
            expected,
            "CX-10 mismatch at {stem}"
        );
    }
}

/// The checksum routine must be able to fail, or the verification above is
/// decoration.
#[test]
fn the_committed_checksum_is_falsifiable() {
    // The empty string's SHA-256 is a published constant.
    assert_eq!(
        sha256_hex(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_ne!(sha256_hex(b"abc"), sha256_hex(b"abd"));
}
