# `just vv` is the normative acceptance gate. Everything else is a slice of it.

default: vv

# The whole gate.
vv: fmt-check model lint test features purity no-alloc bdd
    @echo "vv: the acceptance gate passed"

# R1, R2, R8, R10, R11, R13, R15 --- the repository gates, each falsifiable.
model:
    cargo run -q -p xtask -- validate

# Regenerate everything the model owns: the Rust consts and CONFORMANCE.md.
model-write:
    cargo run -q -p xtask -- check-model --write

fmt:
    cargo fmt --all

fmt-check:
    cargo fmt --all -- --check

lint:
    cargo clippy --workspace --all-targets -- -D warnings

test:
    # Keep Cargo and the nested target-build commands on the pinned toolchain.
    # Some hosts put Homebrew Rust ahead of rustup; the conformance tests spawn
    # Cargo for thumbv7em, whose standard library belongs to the pinned toolchain.
    PATH="$(dirname "$(rustup which cargo --toolchain "$(rustup show active-toolchain | sed 's/ .*//')")"):$PATH" \
        cargo test --workspace

# A recipe because `kappa` did not compile. `address_into` read
# `AddressOutcome::label`, a field `uor-addr-1` does not have --- it is
# `.address` --- and nothing in the workspace ever built the feature, so the
# error existed in `main` without a single gate having an opinion about it.
# `just test` and `just lint` both run at default features, and `kappa` is off by
# default; `cargo deny --all-features` reads the dependency graph and compiles
# nothing. A feature only its author has built is a feature that does not work.
#
# `--all-targets` because the tests behind a feature flag are code too, and they
# are the half most likely to be left behind by an upstream rename.
#
# Every optional feature compiles, and the ones with tests run them.
features:
    cargo check --workspace --all-features --all-targets
    cargo test -p uor-matmul-codec --all-features
    # `CX-08`'s harness sits behind `ref-gemm`, which is not a default feature,
    # so `cargo test --workspace` compiles it and never runs it. The strengthened
    # `CM-02` found that: a test that is compiled out is not evidence, it is a
    # name. This line is what makes the oracle's row honest.
    cargo test -p uor-matmul-validate --all-features

# CT-02: the whole corpus in a build where every accumulator operation is
# checked and any overflow panics. The width is derived so that this is
# unreachable; the profile exists to witness that, not to guard it.
#
# The whole corpus with every accumulator operation checked.
checked:
    cargo test --workspace --profile checked

# R2, R3, R13, and CU-01's disassembly.
purity:
    cargo run -q -p xtask -- audit-purity
    cargo run -q -p xtask -- audit-uor-float
    cargo run -q -p xtask -- audit-disassembly

# R7, C1, CA-03: every shipped crate builds for a target with no allocator at
# all. A crate that had picked up an `alloc` dependency cannot link here.
#
# Build the shipped crates for targets with no allocator at all.
no-alloc:
    cargo build -p uor-matmul --no-default-features --target thumbv7em-none-eabihf
    # `.cargo/config.toml` pins `+simd128` for this target, so the plain build
    # below is the SIMD128-*on* one and the second has to say `-simd128` to be a
    # second configuration at all. It used to say `+simd128` again by way of
    # `build.rustflags`, which `target.<triple>.rustflags` outranks --- so the
    # pair compiled the same thing twice and the comment claimed otherwise.
    # Whether the two *agree* is `CB-05`, and that is asserted by running them
    # under `wasmtime` in `cross-run`; what this pair asserts is that neither
    # links an allocator.
    cargo build -p uor-matmul --no-default-features --target wasm32-unknown-unknown
    cargo build -p uor-matmul --no-default-features --target wasm32-unknown-unknown \
        --config 'target.wasm32-unknown-unknown.rustflags=["-C","target-feature=-simd128"]'
    cargo check -p uor-matmul --target aarch64-unknown-linux-gnu

# The crates that carry an ISA sequence or a width that a 32-bit `usize` can
# change. `uor-matmul-conformance` and `-validate` are absent because their tests
# read the repository's own files, which a WASI sandbox does not hand them.
#
# `uor-matmul` is here for the raw-pointer face: `low` and `span` compute a
# strided view's window in `isize`, which is 32 bits on `wasm32` and 64 here, and
# a window is exactly the kind of arithmetic that changes with the width. Two
# tests that assumed a 64-bit `usize` were already found this way.
cross_crates := "-p uor-matmul-core -p uor-matmul-codec -p uor-matmul-kernels -p uor-matmul-gemm -p uor-matmul"

# CB-04, CB-05: execute the sequences no register here can run.
#
# This exists because the alternative was believing them. A NEON `i16` build that
# re-read four rows and never read the last four compiled cleanly, passed every
# gate, and was wrong on every ARM host; one `cargo test` under `qemu-aarch64`
# found it in a quarter of a second. Two tests that asserted a 64-bit `usize`
# went the same way under `wasmtime`.
#
# `--release` because a debug NEON build takes minutes under emulation and
# proves the same thing.
#
# The linker and runner are set here rather than in `.cargo/config.toml` because
# `[target.aarch64-unknown-linux-gnu]` there would also apply on a machine where
# aarch64 is the *host* --- `cross.yml` has one, and a `runner` in the config sent
# its native test binaries through a `qemu-aarch64` it does not have. Set per
# invocation, they reach this cross-run and nothing else.
#
# Run the aarch64 and wasm sequences, on the emulators, for real.
cross-run:
    CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
    CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUNNER="qemu-aarch64 -L /usr/aarch64-linux-gnu" \
        cargo test --release --target aarch64-unknown-linux-gnu {{cross_crates}}
    # CB-05 is the claim that the two wasm configurations agree, so both are
    # *run*. Neither is implied: with SIMD128 off the selector offers only the
    # portable sequence, and the comparison the parity tests make is a different
    # one --- measured, 525 sequence comparisons against 657.
    cargo test --release --target wasm32-wasip1 {{cross_crates}}
    RUSTFLAGS="-C target-feature=+simd128" \
        cargo test --release --target wasm32-wasip1 {{cross_crates}}

# CB-11: the parity checks, run on a Cortex-M target.
#
# This exists for the same reason `cross-run` does: the alternative was
# believing them. No user-mode emulator covers Cortex-M, so the sequences a
# thumbv6m build offers were asserted by a compile --- which is no assertion at
# all, as the NEON kernel that compiled, passed every gate, and was wrong on
# every ARM host established. The executor is the *same* check bodies the host
# test runs, on the reduced corpus, and the run is the claim.
#
# The linker script reaches the build here rather than in `.cargo/config.toml`
# for exactly the reason `cross-run`'s comment records: a `[target.thumbv*]`
# section there would apply to the `no-alloc` library builds too, and a linker
# script named for an executor binary has no business in those. Set per
# invocation, it reaches this build and nothing else.
#
# The ignored test is what asserts the marker: it runs each ELF under
# qemu-system and fails if `CB-11: PASS` is absent, so a crash mid-suite is a
# failure and not a pass-by-silence.
#
# Run the Cortex-M parity executor, on the system emulator, for real.
cortex-m-run:
    RUSTFLAGS="-C link-arg=-Tlink.x" \
        cargo build --release -p uor-matmul-executor --target thumbv6m-none-eabi
    RUSTFLAGS="-C link-arg=-Tlink.x" \
        cargo build --release -p uor-matmul-executor --target thumbv7em-none-eabihf
    cargo test -p uor-matmul-conformance --test cortex_m -- --ignored --nocapture

# `CU-01`'s companion: the crate with the `unsafe` in it, under Miri.
#
# A recipe because it was only ever run in CI, and in CI it never once reported:
# every run in its history is a `failure` on the toolchain pin or a `cancelled` at
# GitHub's six-hour ceiling. It was also pointed at the three crates that
# `#![forbid(unsafe_code)]` rather than the one with the `unsafe` in it.
#
# Needs a nightly toolchain with the `miri` component, which is why it is not part
# of `just vv`; `RUSTUP_TOOLCHAIN` is how the nightly wins against the pin in
# `rust-toolchain.toml`, exactly as `cargo +nightly` does for `just fuzz`.
#
# `CU-07` reads the `-p` list out of this line *and* out of `miri.yml` and asserts
# the two are equal, so a local run and the CI run cannot come to mean different
# things.
#
# Undefined behaviour, in the crate that has the `unsafe`.
miri:
    MIRIFLAGS=-Zmiri-strict-provenance RUSTUP_TOOLCHAIN=nightly \
        cargo miri test -p uor-matmul-kernels -p uor-matmul -p uor-matmul-core -p uor-matmul-codec

# R11: the oracle crates are dev-dependencies and nothing shipped depends on
# them. `cargo deny` is what says so about the *graph* rather than about the
# source, and it also refuses a wildcard version requirement and an advisory
# against anything in the tree.
#
# A recipe because it was only ever run in CI, and three wildcard requirements and
# an unmaintained advisory sat in `main` unseen: `advisories FAILED, bans FAILED`
# on every push. Needs `cargo install cargo-deny`, which is why it is not in
# `just vv`.
#
# Advisories, bans, licences and sources, over the dependency graph.
deny:
    cargo deny --all-features check

# CA-02: the corpus digest is the same on every target.
#
# Every off-host axis: no-alloc, the emulators, and the corpus digest.
cross: no-alloc cross-run
    cargo test -p uor-matmul-conformance --test environment

# The bench half of `scaling`, on its own for the quick loop: one criterion
# group per element type times this library beside every enabled oracle at
# three shapes; this includes tropical lane and witness scaling. Byte-equality
# is asserted inside each timed closure. A regression
# against your own last run is a criterion baseline away ---
# `--save-baseline before` before the change, `--baseline before` after.
#
# "Are we faster?", beside every enabled oracle. Seconds. The completed
# Criterion measurements are consolidated into target/criterion/REPORT.md and
# target/criterion/REPORT.html after the run.
bench:
    cargo bench -p uor-matmul-validate
    cargo run --release -p uor-matmul-validate --bin benchmark_report

# CG-23's native lookup protocol is deliberately absent from ordinary Criterion
# runs. It collects and emits 256 paired 50 ms epochs for all seven cases on one
# pinned CPU. Exact linked-ELF inspection on the recorded compiler/host found
# the MR1 reduction and unchanged NR16 tiles normalized-static-equivalent to
# their controls, with the production alphabet 64-byte aligned and addressed by
# direct LEA. Their clocks are labeled open/static-control and never asserted as
# build truth. The four structurally changed cases retain the preregistered
# demonstrated-superiority rule `upper95 <= 1`; poison, complete-output checks,
# and identical safe wrappers apply to both classes.
native-lookup-acceptance:
    @mkdir -p target/measurements
    bash -o pipefail -c 'taskset -c 0 cargo bench -p uor-matmul-validate --bench scaling -- __native_lookup_acceptance_only__ --noplot 2>&1 | tee target/measurements/native-lookup-acceptance-2026-08-08.log'

# Regenerate the comparison report from an existing Criterion directory.
bench-report:
    cargo run --release -p uor-matmul-validate --bin benchmark_report

# CG-*: scaling is a V&V axis, not a benchmark. Every performance claim is a
# fitted exponent with a confidence interval, against the same fit for the
# oracle. Every figure it prints is `open`.
# `--release` is not optional here. `cargo test` builds at `opt-level = 0`, and a
# throughput figure from an unoptimised build is not a figure --- measured, the
# same shapes read two hundred times slower. The timed tests say so themselves if
# they are run without it.
#
# Fitted scaling exponents against the oracle's, every figure `open`.
scaling: bench
    cargo test --release -p uor-matmul-validate --test scaling_report -- --nocapture

# CG-21: both IEEE widths walk the same structural shapes against the retained
# exact reference, matrixmultiply, and faer, followed by matched integer and
# tropical controls.  Every calibrated batch is poisoned before timing and
# checks all result bytes after timing; latency, throughput, and logical caller
# traffic carry 95% confidence intervals, and every raw calibrated duration is
# emitted with its width, case, route, round, and batch.  The
# oracle features are intentionally enabled: a performance recipe that silently
# omitted its comparison would not discharge the claim.
uor-float-sweep:
    cargo test --release -p uor-matmul-gemm --lib float::tests::every_atlas_candidate_is_measurable_with_byte_checks_cg_21 -- --ignored --exact --nocapture --test-threads=1
    cargo test --release -p uor-matmul-validate --test uor_float_sweep -- --ignored --nocapture --test-threads=1

# CG-09: throughput against the degeneracy of the operand, over the large shapes
# `just vv` has no time for. Minutes.
#
# Throughput against operand degeneracy, over the large shapes. Minutes.
collapse:
    cargo run --release -p uor-matmul-validate --example collapse_sweep

# CT-06, CG-08: super-massive input. Minutes, and gigabytes of operands, so it is
# its own recipe rather than part of `vv`.
#
# Super-massive input: operands past the last level of cache. Minutes.
massive:
    cargo test --release -p uor-matmul-validate --test massive -- \
        --ignored --nocapture --test-threads=1

# R4's meta-gate: no `open` claim is asserted as established, and no cited
# authority is presented as this repository's own result.
#
# R4's meta-gate: no `open` claim asserted as established.
honesty: bdd
    cargo run -q -p xtask -- check-model

# R9: every capability begins as a Gherkin scenario, and every scenario has a
# test whose name ends in its ID.
bdd:
    cargo test -p uor-matmul-conformance

# CT-01, CT-03, CK-06, CT-08, CD-25: the fuzz targets. Needs `cargo install
# cargo-fuzz` and a nightly toolchain, which is why it is not part of `just vv`.
#
# The `tropical` target carries a differential as well as a totality claim: it
# runs both witness mechanisms on the operands the fuzzer drew and compares
# their bytes, because a corpus is a list of shapes someone thought of.
#
# Totality over unstructured input, on all four targets.
fuzz duration="60":
    cargo +nightly fuzz run totality -- -max_total_time={{duration}}
    cargo +nightly fuzz run float_decode -- -max_total_time={{duration}}
    cargo +nightly fuzz run codec_shapes -- -max_total_time={{duration}}
    cargo +nightly fuzz run tropical -- -max_total_time={{duration}}

# Regenerate the committed oracle and corpus artifacts. Run only when a corpus
# changes; the outputs are committed and their digests are recorded.
#
# Regenerate the NumPy oracles and the symbol corpus. Only when a corpus changes.
oracles:
    python3 oracles/numpy/generate.py
    python3 oracles/tropical/generate.py
    python3 oracles/symbols/generate.py

# CG-11: the static issue census --- llvm-mca over the emitted inner loops, one
# named bottleneck per kernel sequence, reported as scheduling-model predictions
# and never asserted as measurements. Not part of `just vv`: a Rust toolchain
# does not ship llvm-mca, so a gate that required it would fail on a clean
# checkout for a reason that has nothing to do with the code --- the same
# reason `just miri` and `just fuzz` are their own recipes.
#
# Static issue analysis over the emitted inner loops. Needs llvm-mca.
census:
    cargo run -q -p xtask -- issue-census
    cargo test -p uor-matmul-conformance --test issue_census -- --ignored --nocapture

# CG-10: the tabulated traversal against this library's own packed kernels, over
# shapes large enough for the table to amortize. Minutes.
#
# The tabulated traversal against the packed kernels. Minutes.
tabulation:
    cargo run --release -p uor-matmul-validate --example tabulation_sweep

# The recorded break-even against the clock and the census, on this host.
# Minutes, and every figure is `open`.
#
# The tabulation break-even, measured. Minutes.
breakeven:
    cargo run --release -p uor-matmul-validate --example tabulation_breakeven

# CG-14: the u8 symbol path's achieved bytes/second on gemv and skinny GEMM,
# against the host's own STREAM triad number measured in the same harness, with
# byte-identity asserted inside every timed run. Minutes, and every figure is
# `open`. `--release` is not optional: a throughput figure from an unoptimised
# build is not a figure.
#
# The symbol path against the bus, measured. Minutes.
symbol-bandwidth:
    cargo test --release -p uor-matmul-validate --test symbol_bandwidth -- \
        --ignored --nocapture --test-threads=1

# CG-15: the retained default and compatibility float-workspace spellings over
# one Atlas operation, on the recorded workspace-sweep shapes, with byte
# identity asserted inside every timed run. Minutes, and every figure is
# `open`. `--release` is not optional: an unoptimised throughput figure is not
# a figure.
#
# The retained float-workspace protocol over the one Atlas operation. Minutes.
bridge-sweep:
    cargo test --release -p uor-matmul-validate --test bridge_sweep -- \
        --ignored --nocapture --test-threads=1

# The block-16 pricing sweep: `Book<256, 16>` against `Book<256, 8>`, in both
# code widths, on the tabulation-sweep shapes and through the break-even, with
# byte-identity asserted inside every timed run. Minutes, and every figure is
# `open`. `--release` is not optional: a throughput figure from an unoptimised
# build is not a figure. `BLOCK_SWEEP_CHECK=1` is the correctness dry run ---
# one iteration, the census ratios asserted.
#
# The block-16 pricing sweep, measured. Minutes.
block-sweep:
    cargo test --release -p uor-matmul-validate --test block_sweep -- \
        --ignored --nocapture --test-threads=1

# The scalar-port co-issue experiment: the AVX2 `i8` tile kernel beside a
# scalar Kronecker broadcast stream, one exact accumulator, the ratio printed
# per configuration. x86-only by construction --- the thesis is about x86
# ports --- and off-x86 the harness prints its decline and why. Minutes, and
# every figure is `open`. `--release` is not optional: a throughput figure
# from an unoptimised build is not a figure.
#
# The scalar-port co-issue experiment. Minutes, x86 only.
coissue:
    cargo test --release -p uor-matmul-validate --lib coissue -- \
        --ignored --nocapture --test-threads=1

# CG-16: the symbol-tabulated Atlas traversal against the compatibility
# spelling, dense Atlas, and the oracle, on gemv, skinny, and tabulation-sweep
# shapes, with byte identity asserted inside every timed run. Minutes, and
# every figure is `open`. `--release` is not optional: an unoptimised
# throughput figure is not a figure.
#
# Atlas symbol tabulation against dense Atlas and the bus. Minutes.
symbol-tabulated:
    cargo test --release -p uor-matmul-validate --test symbol_tabulated_sweep -- \
        --ignored --nocapture --test-threads=1

# CG-12: the sub-cubic recursion against the cubic packed walk on the
# i32-exact lane, swept through the crossover, with the host's fastest
# sustained product rate on the same axes and byte-identity asserted inside
# every timed run. Minutes, and every figure is `open`. `--release` is not
# optional: a throughput figure from an unoptimised build is not a figure.
#
# The recursion against the cubic walk, measured. Minutes.
strassen-sweep:
    cargo test --release -p uor-matmul-validate --test strassen_sweep -- \
        --ignored --nocapture --test-threads=1

# CG-19: the selection lane against the accumulation lane at matched shapes ---
# one driver at two instantiations of `E`, so what separates the rates is the
# arithmetic and not the traversal --- with the packed ring lane beside them for
# scale, and the two witness mechanisms at both ends of the compare pass's cost.
# Byte-identity with a reference computed off the timed path is asserted inside
# every timed run. Seconds, and every figure is `open`. `--release` is not
# optional: a throughput figure from an unoptimised build is not a figure.
#
# The selection lane against the ring lane, measured. Seconds.
tropical-sweep:
    cargo test --release -p uor-matmul-validate --test tropical_sweep -- \
        --ignored --nocapture --test-threads=1

# CG-17: the i64x2 SWAR broadcast sequence against the i32x4 dot-with-extends
# sequence and the portable reference, on wasm32-wasip1 under wasmtime, per
# panel depth and bound, with byte-identity asserted inside every timed run.
# Every figure is `open`. `--release` is not optional: a throughput figure
# from an unoptimised build is not a figure.
#
# The SWAR sequence against the dot incumbent, measured under wasmtime.
swar-sweep:
    RUSTFLAGS="-C target-feature=+simd128" \
        cargo test --release --target wasm32-wasip1 -p uor-matmul-kernels --test swar_sweep -- \
        --ignored --nocapture
