//! R11: every provenance record points at a harness that exists.
//!
//! `oracles/provenance.toml` binds each external claim to the function that
//! makes it. Nothing read that binding. `xtask verify-oracles` substring-matches
//! the record's *id* against the file's text and never looks at any other field,
//! and the register's ID-to-test rule (`CM-02`) matches a test to an ID by the
//! *suffix of its name*, which the real functions satisfy whatever the record
//! says. So four of eleven `harness` values named functions that existed nowhere
//! in the repository, both gates stayed green, and the only way to notice was to
//! grep each name by hand.
//!
//! This file closes that. It resolves every `harness` and every `generator` path
//! in the record file and fails on any that cannot be found --- and it resolves
//! them the way a reader would, by opening the named file and looking for the
//! named declaration, so a rename that moves a function without updating its
//! record is a red test rather than a citation to a missing paper.
//!
//! The reader below is a line scanner rather than a TOML parse, for the reason
//! `corpus::sha256_hex` is a hand-rolled digest: a check on a committed artifact
//! that needs a dependency to run is one more thing between the claim and its
//! evidence. It is not a lax reader --- it counts the records it saw and refuses
//! a file where the count of `harness` keys disagrees, so a record that lost its
//! harness field fails here too.

use std::path::{Path, PathBuf};

/// The repository root, resolved from this crate's manifest directory so that
/// the test works from any working directory.
fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/uor-matmul-validate is two below the root")
        .to_path_buf()
}

/// One record's pointers into the repository.
#[derive(Debug)]
struct Record {
    id: String,
    harness: String,
    generator: Option<String>,
}

/// The value of a `key = "value"` line, or `None` if this line is not that key.
///
/// Deliberately exact about the shape it accepts: a `harness` value is always a
/// single-quoted scalar in this file, and a reader that also accepted the `"""`
/// multi-line form would silently accept a `note` that happened to start with
/// the word.
fn scalar<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let rest = line.trim().strip_prefix(key)?.trim_start();
    let rest = rest.strip_prefix('=')?.trim();
    if rest.starts_with("\"\"\"") {
        return None;
    }
    rest.strip_prefix('"')?.strip_suffix('"')
}

/// Every record in `oracles/provenance.toml`, in file order.
fn records(text: &str) -> Vec<Record> {
    let mut out: Vec<Record> = Vec::new();
    let mut opened = 0usize;
    let (mut id, mut harness, mut generator) = (None, None, None);
    let flush = |id: &mut Option<String>,
                 harness: &mut Option<String>,
                 generator: &mut Option<String>,
                 out: &mut Vec<Record>| {
        if let (Some(id), Some(harness)) = (id.take(), harness.take()) {
            out.push(Record {
                id,
                harness,
                generator: generator.take(),
            });
        }
    };
    for line in text.lines() {
        if line.trim() == "[[record]]" {
            opened += 1;
            flush(&mut id, &mut harness, &mut generator, &mut out);
            continue;
        }
        if let Some(v) = scalar(line, "id") {
            id = Some(v.to_string());
        } else if let Some(v) = scalar(line, "harness") {
            harness = Some(v.to_string());
        } else if let Some(v) = scalar(line, "generator") {
            generator = Some(v.to_string());
        }
    }
    flush(&mut id, &mut harness, &mut generator, &mut out);
    assert_eq!(
        out.len(),
        opened,
        "every [[record]] must carry both an id and a harness; {opened} records parsed to \
         {} complete ones",
        out.len()
    );
    assert!(opened >= 11, "the record file lost records: {opened}");
    out
}

/// Resolve `<path>::<fn>` against the repository, reporting why not.
///
/// `Ok(())` means the named file exists and declares the named function. The
/// message on the error side is the whole value of this test, so it names the
/// record, the path, and the function separately.
fn resolve(root: &Path, id: &str, harness: &str) -> Result<(), String> {
    let (path, name) = harness
        .split_once("::")
        .ok_or_else(|| format!("{id}: `{harness}` is not `<path>::<fn>`"))?;
    let file = root.join(path);
    let text = std::fs::read_to_string(&file)
        .map_err(|e| format!("{id}: {} names no file: {e}", file.display()))?;
    // A declaration, not a mention: `fn <name>(` at the start of a trimmed line
    // is how every test in this repository is written, and matching the bare
    // name would be satisfied by the record's own text appearing in a comment.
    let declared = text.lines().any(|l| {
        let l = l.trim_start();
        let l = l.strip_prefix("pub ").unwrap_or(l);
        l.strip_prefix("fn ")
            .and_then(|r| r.strip_prefix(name))
            .is_some_and(|r| r.starts_with('('))
    });
    if declared {
        Ok(())
    } else {
        Err(format!(
            "{id}: {path} declares no `fn {name}`; the provenance record points at nothing"
        ))
    }
}

/// R11: every `harness` and every `generator` in the record file resolves.
#[test]
fn the_recorded_harness_names_resolve() {
    let root = root();
    let text = std::fs::read_to_string(root.join("oracles/provenance.toml"))
        .expect("oracles/provenance.toml");
    let records = records(&text);

    let mut broken: Vec<String> = Vec::new();
    for r in &records {
        if let Err(e) = resolve(&root, &r.id, &r.harness) {
            broken.push(e);
        }
        if let Some(g) = &r.generator {
            let path = root.join(g);
            if !path.exists() {
                broken.push(format!("{}: generator {g} names no file", r.id));
            }
        }
    }
    assert!(
        broken.is_empty(),
        "oracles/provenance.toml points at {} thing(s) that do not exist:\n  {}",
        broken.len(),
        broken.join("\n  ")
    );
    eprintln!("R11: {} provenance records resolve", records.len());
}

/// The resolver must be able to fail, or the check above is decoration.
///
/// Both halves are exercised: a function that is absent from a file that
/// exists, and a file that does not exist at all. The first is the defect this
/// test was written for --- all four broken records named a real file.
#[test]
fn the_harness_resolver_is_falsifiable() {
    let root = root();
    let real = "crates/uor-matmul-validate/tests/cross_library.rs";

    // The name the record carried for as long as the file existed.
    let e = resolve(&root, "CX-01", &format!("{real}::ndarray_i32_agrees_cx_01"))
        .expect_err("a function that does not exist must not resolve");
    assert!(
        e.contains("declares no `fn ndarray_i32_agrees_cx_01`"),
        "{e}"
    );

    let e = resolve(
        &root,
        "CX-00",
        "crates/uor-matmul-validate/tests/nowhere.rs::f",
    )
    .expect_err("a file that does not exist must not resolve");
    assert!(e.contains("names no file"), "{e}");

    // And a real one does resolve, or the test above would pass on a resolver
    // that failed everything.
    resolve(
        &root,
        "CX-01",
        &format!("{real}::ndarray_i32_is_byte_identical_cx_01"),
    )
    .expect("the real name resolves");

    // The scanner reads the fields it claims to read, and refuses the
    // multi-line form that a `note` uses.
    assert_eq!(scalar(r#"id = "CX-01""#, "id"), Some("CX-01"));
    assert_eq!(scalar("note = \"\"\"a", "note"), None);
    assert_eq!(scalar(r#"id = "CX-01""#, "harness"), None);
}
