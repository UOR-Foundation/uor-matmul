//! `CB-11`: the Cortex-M executor is *run*, not merely built.
//!
//! The executor's ELF prints `CB-11: PASS` over semihosting only after every
//! parity check has held on the device; this test runs that ELF under
//! `qemu-system-arm` and fails if the marker is absent. A compile would assert
//! nothing: the aarch64 lesson (`cross-run`'s comment in the Justfile) was a
//! kernel that compiled, passed every gate, and was wrong on every ARM host.
//! No user-mode emulator covers Cortex-M, which is why this is a system
//! emulator and not another entry in `cross-run`.
//!
//! Ignored in the ordinary suite because it needs two things `just test`
//! cannot provide: the executors, which are built for their own targets by
//! `just cortex-m-run` (a nested `cargo build` would block on the target
//! directory lock --- see `bdd.rs`), and `qemu-system-arm`, which CI installs
//! and a laptop may not have. Absent qemu is a failure here, not a skip: in
//! the job that runs this, qemu's absence is the misconfiguration.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// The marker the executor prints after the last check holds. Printed and
/// matched as one line, so a crash mid-suite --- which prints nothing --- is
/// as much a failure as a printed `FAIL`.
const PASS: &str = "CB-11: PASS";

/// `(target triple, qemu machine)` for each executor. The v6m run is the one
/// `CB-11` exists for; v7em rides the same harness because the `SMLAD` lane a
/// thumbv7em sequence would take is the next register this crate grows, and an
/// executor that already runs there is what makes adding it a kernels change
/// and nothing else.
const TARGETS: &[(&str, &str)] = &[
    ("thumbv6m-none-eabi", "mps2-an385"),
    ("thumbv7em-none-eabihf", "mps2-an386"),
];

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/uor-matmul-conformance is two below the root")
        .to_path_buf()
}

fn elf(root: &Path, triple: &str) -> PathBuf {
    root.join("target")
        .join(triple)
        .join("release")
        .join("uor-matmul-executor")
}

/// `CB-11`: every sequence a Cortex-M build offers equals its reference, run
/// under qemu-system on an mps2 machine.
#[test]
#[ignore = "run by `just cortex-m-run`, which builds the executors first"]
fn cortex_m_parity_runs_under_qemu_cb_11() {
    let root = root();
    for &(triple, machine) in TARGETS {
        let elf = elf(&root, triple);
        assert!(
            elf.is_file(),
            "CB-11: {} is not built; run `just cortex-m-run`",
            elf.display()
        );
        let output = Command::new("qemu-system-arm")
            .args([
                "-M",
                machine,
                "-nographic",
                "-semihosting-config",
                "enable=on,target=native",
                "-kernel",
            ])
            .arg(&elf)
            // Semihosting writes to the console; a dead executor must not hang
            // the suite, so stderr is inherited and only stdout is captured.
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .output()
            .unwrap_or_else(|e| {
                panic!("CB-11: qemu-system-arm did not start: {e}; CI installs it, so its absence is a failure")
            });
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.lines().any(|line| line.trim() == PASS),
            "CB-11: the {triple} executor did not print `{PASS}` on {machine}\n\
             exit status: {}\n--- semihosting output ---\n{stdout}",
            output.status
        );
        eprintln!("CB-11: {triple} on {machine}: {PASS}");
    }
}
