//! `CG-11`: the static issue census names a bottleneck resource for every
//! emitted kernel sequence.
//!
//! The claim is `build`: what is asserted is that the census runs and that no
//! analysed sequence goes without a named bottleneck. The figures the census
//! prints are `llvm-mca` scheduling-model predictions and are never asserted
//! here --- a test that asserted a predicted cycle count would be asserting a
//! model of a machine, not the machine.

use std::path::PathBuf;
use std::process::Command;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("two below the root")
        .to_path_buf()
}

/// The rows of one artifact's table: `(family, bottleneck)` per sequence.
///
/// Read from the artifact rather than restated from the subcommand's stdout,
/// because the artifact is what CI uploads; a report that printed one thing
/// and archived another would pass a stdout check and ship the other.
fn rows(path: &PathBuf) -> Vec<(String, String)> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if !line.starts_with("| ") || line.contains("---") || line.starts_with("| sequence") {
            continue;
        }
        let cells: Vec<&str> = line.split('|').map(str::trim).collect();
        // Leading and trailing empty cells flank the row; the columns between
        // are sequence, family, instructions, cycles, IPC, RThroughput,
        // bottleneck.
        assert!(
            cells.len() == 9,
            "{line}: a census row has seven columns; the artifact's shape is \
             part of the claim"
        );
        out.push((cells[2].to_string(), cells[7].to_string()));
    }
    out
}

/// `CG-11`: every analysed sequence has a named bottleneck, and no shipped
/// ISA family is silently absent from the report.
#[test]
#[ignore = "needs llvm-mca and the cross targets; run via just census"]
fn the_issue_census_names_a_bottleneck_per_kernel_cg_11() {
    let status = Command::new(env!("CARGO"))
        .current_dir(root())
        .args(["run", "-q", "-p", "xtask", "--target-dir"])
        .arg(root().join("target").join("cg11"))
        .args(["--", "issue-census"])
        .status()
        .expect("xtask runs");
    assert!(
        status.success(),
        "CG-11: the issue census must run; a census that cannot run is a claim \
         without an object"
    );

    let dir = root().join("target").join("issue-census");
    let x86 = dir.join("x86-64.md");
    assert!(x86.exists(), "CG-11: {} was not written", x86.display());
    let x86_rows = rows(&x86);
    assert!(
        !x86_rows.is_empty(),
        "CG-11: the x86-64 artifact holds no sequence; the census would pass vacuously"
    );
    for (family, bottleneck) in &x86_rows {
        assert!(
            !bottleneck.is_empty(),
            "CG-11: a {family} sequence is reported with no bottleneck; a row \
             without one is an adjective, not an analysis"
        );
    }

    // Every ISA family the x86-64 asm carries must appear in the report. A
    // family dropped between the disassembly and the table is a report that
    // answers a narrower question than it claims to.
    for family in ["portable", "avx2", "avx512", "table"] {
        assert!(
            x86_rows.iter().any(|(f, _)| f == family),
            "CG-11: no {family} sequence in the x86-64 artifact"
        );
    }

    // The aarch64 half exists when the target is installed; where it exists,
    // the NEON family must appear in it.
    let aarch64 = dir.join("aarch64.md");
    if aarch64.exists() {
        let arm_rows = rows(&aarch64);
        for (family, bottleneck) in &arm_rows {
            assert!(
                !bottleneck.is_empty(),
                "CG-11: a {family} sequence is reported with no bottleneck"
            );
        }
        assert!(
            arm_rows.iter().any(|(f, _)| f == "neon"),
            "CG-11: no neon sequence in the aarch64 artifact"
        );
    }
}
