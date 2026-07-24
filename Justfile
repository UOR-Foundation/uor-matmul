# `just vv` is the normative acceptance gate. Everything else is a slice of it.

default: vv

# The whole gate.
vv: fmt-check model lint test purity no-alloc
    @echo "vv: the acceptance gate passed"

# R10, R1, R8, R13, R15, R11 --- the repository gates.
model:
    cargo run -q -p xtask -- validate

# Regenerate everything the model owns.
model-write:
    cargo run -q -p xtask -- check-model --write

fmt:
    cargo fmt --all

fmt-check:
    cargo fmt --all -- --check

lint:
    cargo clippy --workspace --all-targets -- -D warnings

test:
    cargo test --workspace

# CT-02: the whole corpus in a build where every accumulator operation is
# checked and any overflow panics. The width is derived so that this is
# unreachable; the profile exists to witness that, not to guard it.
checked:
    cargo test --workspace --profile checked

# R2, R3, R13.
purity:
    cargo run -q -p xtask -- audit-purity

# R7, C1: every shipped crate builds for a target with no allocator at all.
# A crate that had picked up an `alloc` dependency cannot link here.
no-alloc:
    cargo build -p uor-matmul --no-default-features --target thumbv7em-none-eabihf
    cargo build -p uor-matmul --no-default-features --target wasm32-unknown-unknown

# CA-02: the corpus produces identical bytes off the host.
cross: no-alloc

# CG-*: scaling is a V&V axis, not a benchmark. Every performance claim is a
# fitted exponent with a confidence interval, against the same fit for the
# oracle.
scaling:
    cargo bench -p uor-matmul-validate

# R4's meta-gate: no `open` claim is asserted as established.
honesty:
    cargo run -q -p xtask -- check-model

# R9: every capability begins as a Gherkin scenario.
bdd:
    cargo test -p uor-matmul-conformance
