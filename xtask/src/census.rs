//! `CG-11`: the static issue census.
//!
//! The operation census in `uor-matmul-gemm::tabulated` counts what a run
//! *does*; this census reads what the emitted inner loops *cost*, before any
//! run. `llvm-mca` simulates each kernel sequence against a scheduling model
//! and reports per-port occupancy, the critical path, and a predicted
//! throughput. The figures are the model's, not the machine's --- which is
//! why the claim here is `build` about the analysis and nothing about the
//! numbers: every sequence gets a named bottleneck, and the numbers are
//! reported, never asserted.
//!
//! x86-64 is the required half, modelled at `znver4`, because that model
//! covers both the AVX2 and the AVX-512 VNNI sequences this crate emits ---
//! `znver3` has no AVX-512 model. aarch64 is modelled at `apple-m4` when its
//! target is installed, and skipped with a note when it is not. wasm has no
//! scheduling model at all, so the SIMD128 sequences are absent from the
//! report rather than analysed against the wrong machine.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::audit;
use crate::Fail;

/// One disassembly target the census reads.
struct CensusTarget {
    /// The rustup triple the asm is emitted for.
    triple: &'static str,
    /// The `-march` llvm-mca analyses it as.
    march: &'static str,
    /// The scheduling model. A prediction is only ever about one machine, and
    /// the artifact says which.
    mcpu: &'static str,
    /// The artifact's file name under `target/issue-census/`.
    artifact: &'static str,
}

const X86: CensusTarget = CensusTarget {
    triple: "x86_64-unknown-linux-gnu",
    march: "x86-64",
    mcpu: "znver4",
    artifact: "x86-64.md",
};

const AARCH64: CensusTarget = CensusTarget {
    triple: "aarch64-unknown-linux-gnu",
    march: "aarch64",
    mcpu: "apple-m4",
    artifact: "aarch64.md",
};

/// One emitted function: its symbol and the trimmed lines of its body, local
/// labels included.
struct Emitted {
    symbol: String,
    lines: Vec<String>,
}

/// A function worth analysing: an inner loop, not a stub.
struct Sequence {
    name: String,
    family: &'static str,
    instructions: Vec<String>,
}

/// What llvm-mca says about one sequence, per simulated iteration.
struct Analysis {
    instructions: u64,
    cycles: f64,
    ipc: f64,
    rthroughput: f64,
    bottleneck: String,
}

/// `CG-11`: run the census and write the artifacts.
pub fn issue_census(root: &Path) -> Result<(), Fail> {
    let mca = find_llvm_mca()?;
    let version = mca_version(&mca)?;
    let out_dir = root.join("target").join("issue-census");
    std::fs::create_dir_all(&out_dir)?;

    let mut targets = vec![X86];
    if target_installed(AARCH64.triple) {
        targets.push(AARCH64);
    } else {
        println!(
            "issue-census: {} is not an installed rustup target; the NEON sequences are \
             absent from this run rather than analysed",
            AARCH64.triple
        );
    }

    let mut total = 0usize;
    for target in &targets {
        let asm_dir = out_dir.join("asm").join(target.triple);
        emit_asm(root, target, &asm_dir)?;

        let mut sequences = Vec::new();
        let mut skipped = 0usize;
        let mut objects = 0usize;
        for path in audit::asm_files(&asm_dir)? {
            let file = path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            if !file.starts_with("uor_matmul_kernels") {
                continue;
            }
            objects += 1;
            for f in emitted_functions(&std::fs::read_to_string(&path)?) {
                let instructions: Vec<String> = f
                    .lines
                    .iter()
                    .filter(|l| is_instruction(l))
                    .cloned()
                    .collect();
                // Fewer than five instructions, or no backward branch: a stub
                // or a straight-line helper, and a census of inner loops has
                // nothing to say about either.
                if instructions.len() < 5 || !has_backward_branch(&f.lines) {
                    skipped += 1;
                    continue;
                }
                // `{:#}` drops the crate disambiguator, which is a build
                // hash: it would make every row differ between two runs of
                // the same source.
                let name = format!("{:#}", rustc_demangle::demangle(&f.symbol));
                sequences.push(Sequence {
                    family: family(&name),
                    name,
                    instructions,
                });
            }
        }
        if objects == 0 {
            return Err(format!(
                "no uor-matmul-kernels object was emitted for {}, so the census would \
                 report on nothing. `--emit asm` writes a `.s` only when the crate is \
                 actually compiled; the emit step above must have read a warm directory \
                 (CG-11)",
                target.triple
            )
            .into());
        }
        // The directory walk's order is the filesystem's, so sort: a report
        // that shuffles between runs gets read as a diff that means nothing.
        sequences.sort_by(|a, b| a.name.cmp(&b.name));

        let mut rows: Vec<(&Sequence, Analysis)> = Vec::new();
        for seq in &sequences {
            rows.push((seq, analyse(&mca, target, seq)?));
        }
        if let Some((seq, _)) = rows.iter().find(|(_, a)| a.bottleneck.is_empty()) {
            return Err(format!(
                "`{}` was analysed with no bottleneck named; a row without one is an \
                 adjective, not an analysis (CG-11)",
                seq.name
            )
            .into());
        }

        let content = render(target, &version, &rows, skipped);
        std::fs::write(out_dir.join(target.artifact), &content)?;
        print!("{content}");
        total += rows.len();
    }

    if total == 0 {
        return Err(
            "issue-census analysed nothing; the census would pass vacuously (CG-11)".into(),
        );
    }
    println!("issue-census: {total} sequences analysed, every one with a named bottleneck (CG-11)");
    Ok(())
}

/// The llvm-mca to run, or a failure that says everywhere we looked. A census
/// that passes without its tool is worse than no census: the table it prints
/// is empty, and an empty table reads as "nothing to report".
fn find_llvm_mca() -> Result<PathBuf, Fail> {
    let mut looked: Vec<String> = Vec::new();
    if let Some(set) = std::env::var_os("LLVM_MCA") {
        // Trusted as given; a bogus value fails loudly at run time, naming
        // itself in the error.
        return Ok(PathBuf::from(set));
    }
    if let Some(paths) = std::env::var_os("PATH") {
        looked.push("PATH".to_string());
        for dir in std::env::split_paths(&paths) {
            let candidate = dir.join("llvm-mca");
            if candidate.is_file() {
                return Ok(candidate);
            }
        }
    }
    for fixed in ["/opt/homebrew/opt/llvm/bin/llvm-mca", "/usr/bin/llvm-mca"] {
        looked.push(fixed.to_string());
        if Path::new(fixed).is_file() {
            return Ok(PathBuf::from(fixed));
        }
    }
    // Ubuntu's `llvm` package keeps the tools versioned under /usr/lib.
    if let Ok(entries) = std::fs::read_dir("/usr/lib") {
        for entry in entries.flatten() {
            if !entry.file_name().to_string_lossy().starts_with("llvm-") {
                continue;
            }
            let candidate = entry.path().join("bin").join("llvm-mca");
            looked.push(candidate.display().to_string());
            if candidate.is_file() {
                return Ok(candidate);
            }
        }
    }
    Err(format!(
        "llvm-mca was not found; looked at LLVM_MCA (unset), {}. A Rust toolchain does \
         not ship llvm-mca, so the census names where it looked rather than reporting \
         nothing (CG-11)",
        looked.join(", ")
    )
    .into())
}

/// The first line of `llvm-mca --version`, for the artifact header: a
/// prediction is about one version of one model.
fn mca_version(mca: &Path) -> Result<String, Fail> {
    let out = Command::new(mca).arg("--version").output().map_err(|e| {
        format!(
            "could not run `{}` (from LLVM_MCA or the search path): {e}",
            mca.display()
        )
    })?;
    if !out.status.success() {
        return Err(format!(
            "`{} --version` failed: {}. LLVM_MCA names a tool that is not llvm-mca; the \
             census does not run without its tool (CG-11)",
            mca.display(),
            String::from_utf8_lossy(&out.stderr)
        )
        .into());
    }
    Ok(String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()
        .unwrap_or("unknown llvm-mca")
        .trim()
        .to_string())
}

/// Is `triple` an installed rustup target? Asked rather than attempted, so
/// the absence of the aarch64 half says what it is instead of surfacing as a
/// compiler error about a missing std.
fn target_installed(triple: &str) -> bool {
    Command::new("rustup")
        .args(["target", "list", "--installed"])
        .output()
        .map(|out| {
            String::from_utf8_lossy(&out.stdout)
                .lines()
                .any(|l| l.trim() == triple)
        })
        .unwrap_or(false)
}

/// Emit the kernels crate's assembly for one target into a fresh directory.
fn emit_asm(root: &Path, target: &CensusTarget, dir: &Path) -> Result<(), Fail> {
    // Removed, not reused: `--emit asm` writes a `.s` only when the crate is
    // actually compiled, so a warm directory makes `cargo rustc` answer
    // "Finished" and leave last run's assembly --- or none at all. The defect
    // is `audit_disassembly`'s to document; the census inherits the fix.
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir)?;
    let status = Command::new(env!("CARGO"))
        .current_dir(root)
        .args([
            "rustc",
            "--target",
            target.triple,
            "-p",
            "uor-matmul-kernels",
            "--lib",
            "--release",
            "--target-dir",
        ])
        .arg(dir)
        .args(["--", "--emit", "asm", "-C", "debuginfo=0"])
        .status()?;
    if !status.success() {
        return Err(format!(
            "could not emit {} assembly for uor-matmul-kernels (CG-11)",
            target.triple
        )
        .into());
    }
    Ok(())
}

/// Every emitted function in an `.s`, with the local labels kept: telling a
/// backward branch (a loop) from a forward one (an early exit) takes them,
/// and `audit::function_bodies` discards them. The end-of-body framing is the
/// same three markers --- `.Lfunc_end`, `.size`, `.cfi_endproc` --- for the
/// reason that gate records: a body that runs to end-of-file attributes the
/// rest of the crate to one function.
fn emitted_functions(text: &str) -> Vec<Emitted> {
    let lines: Vec<&str> = text.lines().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let trimmed = lines[i].trim();
        if !audit::is_function_label(trimmed) {
            i += 1;
            continue;
        }
        let symbol = trimmed.trim_end_matches(':').to_string();
        let mut body = Vec::new();
        i += 1;
        while i < lines.len() {
            let inner = lines[i].trim();
            if inner.starts_with(".Lfunc_end")
                || inner.starts_with(".size")
                || inner.starts_with(".cfi_endproc")
            {
                break;
            }
            body.push(inner.to_string());
            i += 1;
        }
        out.push(Emitted {
            symbol,
            lines: body,
        });
    }
    out
}

/// Is this body line an instruction, as opposed to a directive, a comment, or
/// a label? Same test as `audit::float_opcode` draws, because both gates feed
/// on instructions and only on instructions.
fn is_instruction(line: &str) -> bool {
    !(line.is_empty()
        || line.starts_with('.')
        || line.starts_with('#')
        || line.starts_with("//")
        || line.ends_with(':'))
}

/// Does the body branch to one of its own *earlier* local labels? That is
/// what makes it a loop; a jump to a later label is an early exit.
fn has_backward_branch(lines: &[String]) -> bool {
    let mut labels: Vec<&str> = Vec::new();
    for line in lines {
        if let Some(label) = line.strip_suffix(':') {
            if label.starts_with('.') {
                labels.push(label);
            }
            continue;
        }
        if !is_instruction(line) {
            continue;
        }
        let Some(mnemonic) = line.split_whitespace().next() else {
            continue;
        };
        if !is_branch(mnemonic) {
            continue;
        }
        // The target is the last operand: `.LBB0_2` on both ISAs, or a jump
        // table's `*.LJTI0_0(,%rax,8)`, which names a rodata label and so is
        // never in `labels`.
        let target = line
            .split_whitespace()
            .next_back()
            .unwrap_or("")
            .trim_start_matches('*')
            .split(['(', ','])
            .next()
            .unwrap_or("");
        if labels.contains(&target) {
            return true;
        }
    }
    false
}

/// Branch mnemonics on the two ISAs the census reads: x86 conditions all
/// start with `j`, aarch64's are `b`, `b.<cond>`, and the compare-and-test
/// branches. The lists do not collide --- x86's `bt*` are bit tests, not
/// `tb*` --- and a stray hit is harmless anyway, because a non-branch names
/// no local label.
fn is_branch(mnemonic: &str) -> bool {
    mnemonic.starts_with('j')
        || mnemonic.starts_with("loop")
        || mnemonic == "b"
        || mnemonic.starts_with("b.")
        || mnemonic.starts_with("cb")
        || mnemonic.starts_with("tb")
}

/// The ISA family a sequence belongs to, out of its demangled module path.
/// `isa::x86` splits the way the register's `CB-02`/`CB-03` claims split it:
/// the VNNI dot products and the `a5_*` table sequences are AVX-512, the rest
/// is AVX2.
fn family(name: &str) -> &'static str {
    if name.contains("isa::arm::") {
        "neon"
    } else if name.contains("isa::portable::") {
        "portable"
    } else if name.contains("isa::x86::") {
        if name.contains("vnni")
            || name.contains("dpbusd")
            || name.contains("dpwssd")
            || name.contains("a5_")
        {
            "avx512"
        } else {
            "avx2"
        }
    } else if name.contains("::table::") {
        "table"
    } else {
        "other"
    }
}

/// Run llvm-mca over one sequence's instructions and parse what it prints.
fn analyse(mca: &Path, target: &CensusTarget, seq: &Sequence) -> Result<Analysis, Fail> {
    let mut child = Command::new(mca)
        .args(["-march", target.march, "-mcpu", target.mcpu])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("could not run `{}`: {e}", mca.display()))?;
    let mut stdin = child.stdin.take().expect("stdin was piped");
    let _ = stdin.write_all(seq.instructions.join("\n").as_bytes());
    drop(stdin);
    let out = child.wait_with_output()?;
    if !out.status.success() {
        return Err(format!(
            "llvm-mca rejected `{}`: {}",
            seq.name,
            String::from_utf8_lossy(&out.stderr)
        )
        .into());
    }
    parse_analysis(&String::from_utf8_lossy(&out.stdout), seq)
}

/// Parse llvm-mca's summary view: the headline counters, and the bottleneck
/// as the resource with the highest pressure in the "Resource pressure per
/// iteration" table, named as llvm-mca names it. Any deviation from the
/// expected shape is an error, because a census that misreads its tool
/// reports garbage with confidence.
fn parse_analysis(text: &str, seq: &Sequence) -> Result<Analysis, Fail> {
    let unexpected = |what: &str| -> Fail {
        format!(
            "llvm-mca's output for `{}` is not the shape this parser expects ({what}); \
             the census reports nothing it cannot read (CG-11)",
            seq.name
        )
        .into()
    };

    let iterations = counter(text, "Iterations:").ok_or_else(|| unexpected("no Iterations"))?;
    let instructions =
        counter(text, "Instructions:").ok_or_else(|| unexpected("no Instructions"))?;
    let total_cycles =
        counter(text, "Total Cycles:").ok_or_else(|| unexpected("no Total Cycles"))?;
    if iterations == 0 || instructions % iterations != 0 {
        return Err(unexpected(
            "Instructions is not Iterations times a whole block",
        ));
    }
    let ipc = ratio(text, "IPC:").ok_or_else(|| unexpected("no IPC"))?;
    let rthroughput =
        ratio(text, "Block RThroughput:").ok_or_else(|| unexpected("no Block RThroughput"))?;

    // The Resources section names each column of the pressure tables; the
    // per-iteration table that follows gives one value per column, wrapped
    // into further row blocks when the model is wide.
    let mut resources: Vec<(String, String)> = Vec::new();
    let mut in_resources = false;
    let mut in_pressure = false;
    let mut header: Vec<String> = Vec::new();
    let mut best: Option<(f64, String)> = None;
    for line in text.lines() {
        let t = line.trim();
        if t == "Resources:" {
            in_resources = true;
            continue;
        }
        if t == "Resource pressure per iteration:" {
            in_resources = false;
            in_pressure = true;
            continue;
        }
        if t.starts_with("Resource pressure by instruction") {
            in_pressure = false;
        }
        if in_resources {
            if let Some(rest) = t.strip_prefix('[') {
                let Some((col, rest)) = rest.split_once(']') else {
                    return Err(unexpected("a malformed Resources row"));
                };
                let name = rest.trim().trim_start_matches('-').trim().to_string();
                if name.is_empty() {
                    return Err(unexpected("an unnamed resource"));
                }
                resources.push((format!("[{col}]"), name));
            }
            continue;
        }
        if !in_pressure || t.is_empty() {
            continue;
        }
        let tokens: Vec<&str> = t.split_whitespace().collect();
        if tokens.iter().all(|tok| tok.starts_with('[')) {
            header = tokens.iter().map(|s| s.to_string()).collect();
            continue;
        }
        if tokens.len() != header.len() || header.is_empty() {
            return Err(unexpected("a pressure row that does not match its header"));
        }
        for (tok, col) in tokens.iter().zip(&header) {
            if *tok == "-" {
                continue;
            }
            let pressure: f64 = tok
                .parse()
                .map_err(|_| unexpected("an unparseable pressure value"))?;
            let name = resources
                .iter()
                .find(|(c, _)| c == col)
                .map(|(_, n)| n.clone())
                .ok_or_else(|| unexpected("a pressure column the Resources section never named"))?;
            if best.as_ref().is_none_or(|(b, _)| pressure > *b) {
                best = Some((pressure, name));
            }
        }
    }
    let Some((_, bottleneck)) = best else {
        return Err(unexpected("no positive resource pressure"));
    };

    Ok(Analysis {
        instructions: instructions / iterations,
        cycles: total_cycles as f64 / iterations as f64,
        ipc,
        rthroughput,
        bottleneck,
    })
}

/// The integer after a `Key: value` line in llvm-mca's summary.
fn counter(text: &str, key: &str) -> Option<u64> {
    text.lines()
        .find(|l| l.trim_start().starts_with(key))?
        .trim_start()[key.len()..]
        .split_whitespace()
        .next()?
        .parse()
        .ok()
}

/// The float after a `Key: value` line in llvm-mca's summary.
fn ratio(text: &str, key: &str) -> Option<f64> {
    text.lines()
        .find(|l| l.trim_start().starts_with(key))?
        .trim_start()[key.len()..]
        .split_whitespace()
        .next()?
        .parse()
        .ok()
}

/// The artifact: a header that says what the figures are and are not, then
/// the table. The same text is printed, so the console and the archive cannot
/// drift apart.
fn render(
    target: &CensusTarget,
    version: &str,
    rows: &[(&Sequence, Analysis)],
    skipped: usize,
) -> String {
    // Why this model and not another. Only x86-64 has a choice to defend;
    // aarch64 names apple-m4 and there is no second candidate in the crate.
    let model_note = if target.march == "x86-64" {
        " znver4 is the x86-64 model because it covers both the AVX2 and the AVX-512 VNNI \
         sequences; znver3 has no AVX-512 model."
    } else {
        ""
    };
    let mut out = String::new();
    out.push_str(&format!(
        "# Issue census: {} (CG-11)\n\
         \n\
         Static issue analysis of the emitted kernel inner loops, one row per sequence.\n\
         \n\
         - Tool: `llvm-mca`, {version}.\n\
         - Scheduling model: `-march={} -mcpu={}`.{model_note}\n\
         - Every figure is an `llvm-mca` scheduling-model prediction, not a measurement on \
         hardware. Cycles are llvm-mca's Total Cycles over its 100 simulated iterations, \
         divided by 100; IPC and Block RThroughput are as the tool prints them.\n\
         - The bottleneck is the resource with the highest pressure in llvm-mca's \
         \"Resource pressure per iteration\" table, named as llvm-mca names it.\n\
         - wasm has no llvm-mca scheduling model, so the SIMD128 sequences are absent from \
         this census rather than analysed against the wrong machine.\n\
         - {skipped} emitted functions are not below: fewer than five instructions or no \
         backward branch, so they are stubs or straight-line helpers, not inner loops.\n\
         \n\
         | sequence | family | instructions | cycles | IPC | block RThroughput | bottleneck |\n\
         | --- | --- | --- | --- | --- | --- | --- |\n",
        target.march, target.march, target.mcpu
    ));
    for (seq, a) in rows {
        out.push_str(&format!(
            "| `{}` | {} | {} | {:.1} | {:.2} | {:.2} | {} |\n",
            seq.name, seq.family, a.instructions, a.cycles, a.ipc, a.rthroughput, a.bottleneck
        ));
    }
    out
}
