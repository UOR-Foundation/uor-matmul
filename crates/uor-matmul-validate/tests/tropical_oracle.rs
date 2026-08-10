//! `CX-11`: byte equality with a NumPy max-plus product, out of process.
//!
//! The sibling of `CX-10`, and it is here for the same reason (D-1): every
//! other oracle in this repository is a Rust crate, and they share a language, a
//! compiler and sometimes a kernel author, so agreement among them is weaker
//! evidence than it looks. NumPy shares none of those.
//!
//! What is different is what *exact* is worth here. A float oracle computes a
//! different function and its deviation is the measurement (N1). An `int64` ring
//! oracle agrees byte for byte because reduction modulo `2^64` is a ring
//! homomorphism --- true, but an argument. A `(max, +)` product needs neither
//! argument nor exemption: `max` returns one of the values offered to it and `+`
//! is exact at this range, so **there is nothing for two implementations to
//! disagree about**, and the claim is byte identity at `build`.
//!
//! Three things are checked before the product is, and each of them is a way
//! this claim could be true of nothing:
//!
//! - Every artifact is checked against its recorded SHA-256 (R11).
//! - The manifest's corpus parameters are checked against this crate's own
//!   constants, so an artifact drawn at another span or another mask is refused
//!   rather than compared.
//! - The operands are checked against `Corpus::tropical`'s own draw, so the
//!   claim that the Python generator and the Rust corpus are *one* recorded fill
//!   in two spellings is asserted rather than asserted-in-a-comment.

use std::path::PathBuf;

use uor_matmul_core::{as_alphabet_tropical, EncodeMode, MatView, MatViewMut, Triple, Trop};
use uor_matmul_gemm::{gemm, GemmOptions, MaxPlus, Scratch};
use uor_matmul_validate::corpus::{sha256_hex, Corpus, Mask, TROPICAL_SPAN};
use uor_matmul_validate::{bytes_equal, Case};

/// The seed `oracles/tropical/generate.py` was run at, and the one every
/// selection gate walks the corpus at.
const SEED: u64 = 20_260_805;

/// The artifact's spelling of the semiring zero.
///
/// A property of the *file*, not of the element type: a byte stream needs some
/// spelling for `-inf`, and this one is decoded into `Trop`'s own discriminant
/// on the way in. That is the whole reason `Trop` carries a separate field
/// instead of reserving a pattern inside its element (§5.2, R8) --- a library
/// that had reserved `i64::MIN` could not read this file back into an element
/// type that admits `i64::MIN`.
const NEG_INF_SPELLING: i64 = i64::MIN;

fn oracle_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/uor-matmul-validate is two below the root")
        .join("oracles/tropical")
}

fn read_i64(path: &std::path::Path) -> Vec<i64> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().expect("eight bytes")))
        .collect()
}

/// The artifact's spelling, decoded into the element type.
fn decode(xs: &[i64]) -> Vec<Trop<i64>> {
    xs.iter()
        .map(|&v| {
            if v == NEG_INF_SPELLING {
                Trop::NEG_INF
            } else {
                Trop::finite(v)
            }
        })
        .collect()
}

/// The element type, spelled back into the artifact's bytes.
fn encode(xs: &[Trop<i64>]) -> Vec<i64> {
    xs.iter()
        .map(|x| x.get().unwrap_or(NEG_INF_SPELLING))
        .collect()
}

/// Our `(max, +)` product, through the **dense driver** --- not through a
/// second one.
///
/// This is the whole of `CD-29` from the outside: the call below differs from
/// `ours_i64`'s in `tests/numpy.rs` in the element type it is instantiated at
/// and in nothing else. Same `gemm`, same options, same offer.
fn ours_tropical(m: usize, k: usize, n: usize, a: &[Trop<i64>], b: &[Trop<i64>]) -> Vec<Trop<i64>> {
    let mut c = vec![Trop::<i64>::NEG_INF; m * n];
    if m == 0 || n == 0 {
        return c;
    }
    {
        let av = MatView::row_major(as_alphabet_tropical(a), m, k).expect("A fits its buffer");
        let bv = MatView::row_major(as_alphabet_tropical(b), k, n).expect("B fits its buffer");
        let cv = MatViewMut::row_major(&mut c, m, n).expect("C fits its buffer");
        let mut t = Triple::new(av, bv, cv).expect("the product exists");
        gemm(
            &mut t,
            &MaxPlus::OVERWRITE,
            GemmOptions {
                encode: EncodeMode::Wrapping,
                ..Default::default()
            },
            &mut Scratch::none(),
        );
    }
    c
}

/// `CX-11`: byte-identical with NumPy's max-plus product over every committed
/// case.
#[test]
fn numpy_max_plus_is_byte_identical_cx_11() {
    let dir = oracle_dir();
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).expect("manifest"))
            .expect("manifest parses");

    assert_eq!(manifest["id"], "CX-11");
    assert_eq!(manifest["dtype"], "int64");
    assert_eq!(manifest["semiring"], "(max, +)");
    // The artifact was drawn under this crate's corpus parameters, or it is a
    // corpus of something else and comparing against it would say nothing.
    assert_eq!(manifest["seed"].as_u64(), Some(SEED));
    assert_eq!(manifest["span"].as_i64(), Some(TROPICAL_SPAN));
    assert_eq!(manifest["neg_inf"].as_i64(), Some(NEG_INF_SPELLING));
    for (field, mask) in [("mask_a", Mask::LEFT), ("mask_b", Mask::RIGHT)] {
        assert_eq!(manifest[field]["period"].as_u64(), Some(mask.period as u64));
        assert_eq!(manifest[field]["phase"].as_u64(), Some(mask.phase as u64));
    }
    eprintln!(
        "CX-11: numpy {} via {}",
        manifest["numpy_version"].as_str().unwrap_or("?"),
        manifest["entry"].as_str().unwrap_or("?")
    );

    let manifest_cases = manifest["cases"].as_array().expect("cases");
    let corpus = Corpus::tropical(SEED);
    assert_eq!(
        manifest_cases.len(),
        corpus.cases.len(),
        "the artifact and `Corpus::tropical` must be the same corpus"
    );

    let mut masked_outputs = 0usize;
    for (entry, case) in manifest_cases.iter().zip(&corpus.cases) {
        let stem = entry["stem"].as_str().expect("stem");
        let Case { m, k, n, .. } = *case;
        assert_eq!(stem, format!("{m}x{k}x{n}"), "corpus order");

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
                entry[field].as_str().unwrap(),
                "{} has drifted from its recorded digest",
                path.display()
            );
        }

        let a = read_i64(&dir.join(format!("{stem}.a.bin")));
        let b = read_i64(&dir.join(format!("{stem}.b.bin")));
        let expected = read_i64(&dir.join(format!("{stem}.expected.bin")));

        // One recorded fill, two spellings. If the Python re-spelling of the
        // xorshift ever drifts from the Rust one, the comparison below would
        // still pass --- against a corpus nothing else in this repository
        // walks.
        assert_eq!(
            encode(&case.fill_tropical_i64(m * k, 1, Mask::LEFT)),
            a,
            "{stem}: A is not `Corpus::tropical`'s own draw"
        );
        assert_eq!(
            encode(&case.fill_tropical_i64(k * n, 2, Mask::RIGHT)),
            b,
            "{stem}: B is not `Corpus::tropical`'s own draw"
        );

        let ours = encode(&ours_tropical(m, k, n, &decode(&a), &decode(&b)));
        bytes_equal(&ours, &expected).unwrap_or_else(|e| panic!("CX-11 mismatch at {stem}: {e}"));

        masked_outputs += expected.iter().filter(|&&v| v == NEG_INF_SPELLING).count();
    }

    // A-6 is two claims and the mask is half of it: a corpus whose every output
    // was finite would have tested the other half twice. The count is exact
    // because both sides of it are committed --- the corpus and the artifact
    // --- so a shape or a mask that changed without anyone noticing is what
    // this number is here to refuse.
    assert_eq!(
        masked_outputs, 409,
        "the committed artifact's masked-output count changed"
    );
    eprintln!("CX-11: {masked_outputs} committed output cells at the semiring zero");
}

/// The comparison must be able to fail, or the test above is decoration.
///
/// Two ways, because the harness has two halves that could each be vacuous: the
/// decode could collapse the semiring zero into a finite value, and the product
/// could be compared against itself.
#[test]
fn the_tropical_oracle_comparison_is_falsifiable() {
    // The artifact's spelling is not a finite value, and no finite value the
    // draw produces is near it.
    assert_eq!(decode(&[NEG_INF_SPELLING])[0], Trop::<i64>::NEG_INF);
    assert_eq!(decode(&[0])[0], Trop::finite(0));
    assert_ne!(decode(&[0])[0], Trop::<i64>::NEG_INF);

    // A one-term product whose answer is known by hand, and the same product
    // with one lane masked, which is a different answer and not a smaller one.
    let live = [Trop::finite(1i64), Trop::finite(2)];
    let right = [Trop::finite(4i64), Trop::finite(0)];
    assert_eq!(
        ours_tropical(1, 2, 1, &live, &right),
        [Trop::finite(5)],
        "max(1+4, 2+0) = 5"
    );
    let masked = [Trop::<i64>::NEG_INF; 2];
    assert_eq!(
        ours_tropical(1, 2, 1, &live, &masked),
        [Trop::NEG_INF],
        "every term absorbed"
    );
    assert!(bytes_equal(&encode(&[Trop::finite(5i64)]), &[5]).is_ok());
    assert!(bytes_equal(&encode(&[Trop::<i64>::NEG_INF]), &[5]).is_err());
}
