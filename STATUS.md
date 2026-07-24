# STATUS

What of the build specification is implemented, and what is not.

This file exists because the plan's R15 says *nothing is deferred*, and that is
not yet true. Recording which parts are missing is the only honest alternative
to pretending otherwise. R15 is a release condition, and `v0.1.0` is not
releasable until no row of `model/ids.toml` reads `state = "pending"`.

`cargo run -p xtask -- audit-deferral` enforces R15 inside the shipped crates:
no `TODO`, no stub, no `unimplemented!`, no placeholder section. It passes. What
it cannot see is a capability that was never started, which is what the ID
register's `state` column and this file record instead.

`CONFORMANCE.md` is generated from `model/ids.toml` and carries the same
information per ID.

## Implemented and green

| Plan section | What exists | Evidence |
| --- | --- | --- |
| §0.1 C1 | `no_std`, no `alloc` dependency in any shipped crate; builds for `thumbv7em-none-eabihf` and `wasm32-unknown-unknown` | `just no-alloc`, `CA-03` |
| §0.1 C4 | parametric over `(Element, Bound, Codec, MaxBlock)`; the three element families of §5.2b | `uor-matmul-core` |
| §0.1 C6 | the operation is total; `gemm` returns `()` | `CT-01`, `CT-03` |
| §2.1 | the ID register, and `CONFORMANCE.md` generated from it | `CM-02` |
| §3.2 | the accumulator that cannot overflow, derived from the *declared* `MAX_K_BITS` | `CM-01`, `CS-07`, `CT-02` |
| §3.3 | the float codec: `FloatElement::decode`, total on all bit patterns; the complete accumulator with sticky non-finite flags | `CT-03`, `CU-04` |
| §3.4 | integer oracles bit-identical **everywhere** under `EncodeMode::Wrapping`, including past the depth at which they wrap | `CX-01` .. `CX-04` |
| §4 | repository layout, `Justfile`, `clippy.toml`, `deny.toml`, MSRV | — |
| §5.1 | `acc_bits`, `fits_narrow` as `#[doc(hidden)] pub`, `NARROW_CAPS`, `narrow_cap_for` | `CS-07`, `CU-02` |
| §5.2 | `Element` / `IntegerElement` / `FloatElement`, `Bound`, `Full`, `Bnd`, `ObservedBound` | `CT-01`, `CS-03` |
| §5.2b | all three element families, complex accumulating as a pair | `CK-01` |
| §5.3 | `Limbs<L>`, `Complete<L, MIN_EXP>`, the single encode step | `CT-02`, `CD-05` |
| §5.4 | `dot_ref`, `dot_wide`, `dot_instrumented` | `CD-03`, `CD-09`, `CU-02` |
| §5.5 | `Shape`, `Strides`, views, `Triple`, `NotAProduct` | `CS-02`, `CS-03`, `CS-06` |
| §6 | the `Codec` trait with `MAX_BLOCK` and variable-length `decode_len`; all seven tiers; `CodedMatrix`; the canonical kappa manifest | `CK-01` .. `CK-06` |
| §8 (partial) | the driver: traversals, scratch as an offer, epilogues, coded GEMM | `CD-04`, `CD-10`, `CS-04` |
| §10 (partial) | external validation against `ndarray` and `nalgebra`, byte-equality comparator, awkward-shape corpus, an independently written wrapping oracle for the deep half | `CX-01` .. `CX-04` |
| R1, R2, R8, R10, R11, R13, R15 | the gates, each falsifiable | `cargo xtask validate` |

Two defects were found by the suite rather than by inspection, which is the
point of having one:

- The awkward-shape corpus reported a `5 x 0` row-major output as
  self-aliasing, because its row stride is zero. An empty output has no two
  distinct coordinates to collide. Fixed, and pinned by a test.
- The first attempt at the deep half of `CX-01` asserted that a wrapping and a
  saturating encode must differ past the `i32` worst-case bound. They need not:
  random signed data cancels, so the *actual* sum can stay inside `i32` where
  the worst case does not. The witness is now a same-sign constant fill, whose
  exact value is provably past `i32`.

## Not implemented

Each row is a capability the specification requires and this repository does not
yet have. None is a design disagreement; all are unfinished work. The
corresponding IDs read `state = "pending"` in `model/ids.toml`.

| Plan section | Missing | IDs blocked |
| --- | --- | --- |
| §5.3, §3.3 | `EncodeFrom<Complete<L, MIN_EXP>>` for `f32` / `f64`: the correctly-rounded encode out of the complete accumulator. The decode, the accumulator, and the non-finite propagation exist and are tested; nothing rounds back out yet, so there is no float GEMM. | `CX-05` .. `CX-09` |
| §7 | `uor-matmul-kernels`: `KernelSpec`, and the AVX2 / AVX-512 VNNI / NEON / NEON-dotprod / wasm-SIMD128 microkernels. The portable reference exists in `uor-matmul-core` and the driver runs it; `Backend` is accepted and ignored, which the driver says in a comment. | `CB-02` .. `CB-05`, `CD-01`, `CU-01`, `CU-03` |
| §5.5, `CS-05` | the raw-pointer entry points, signature-identical to `matrixmultiply` | `CS-05` |
| §12, S12 | caller-driven parallelism over a declared tile partition | `CG-06` |
| §13, R12 | the scaling harness: fitted exponents with confidence intervals, ours against each oracle's | `CG-01` .. `CG-07` |
| §14 | `uor-matmul-conformance`: the cucumber runner and the honesty meta-gate; `features/suites/*.feature` | R9, and R4's behavioural half |
| §14 | `.github/workflows/*`, `benches/`, `fuzz/` | CI does not run |
| §3.3, `CX-10` | the NumPy `int64` out-of-process oracle and its committed artifacts | `CX-10` |
| §2.1 | a counting global allocator; a big-endian target in CI; the randomized differential at the derived sample size | `CA-01`, `CA-02`, `CD-06`, `CP-01` |
| §4 | `ARCHITECTURE.md`, `VERIFICATION.md`, `ANALYSIS.md`, `VALIDATION.md`, `AGENTS.md`, `docs/` | — |
| §6.2 | the E8 codebook as committed data. `Book<256, 8>` is implemented and tested; the table itself is not in the repository. | S4 |

## Resolved by the revised specification

Three of the four deviations the previous turn flagged are gone, because the
revision answered them. Recorded here so the history is legible.

| Was | Now |
| --- | --- |
| `acc_bits` derived from `usize::BITS` disagreed with the plan's resolved table for `i16`/`i32`/`i64` | §3.2 declares `MAX_K_BITS = 64` in the model, giving exactly 79 / 95 / 127 / 191. The `bits_loose` column is gone and `CM-02` now means what §2.1 says it means. |
| R2 forbade the `f32` token in core, but §5.2 wanted `impl Element for f32`, which the orphan rule puts nowhere else | R2 now forbids float *arithmetic*, not the float *types*. `f32` and `f64` implement `Element` and `FloatElement` directly in `uor-matmul-core`, and the planned `uor-matmul-float` crate is not needed. |
| `oracle_inexact` was recorded for integer oracles past their ceiling | §3.4's ring-homomorphism argument removes the exemption. Integer oracles agree byte for byte at every depth under `EncodeMode::Wrapping`, and `oracle_inexact` is now reachable only by a float oracle. |
| `Codec::BLOCK` was a fixed width, which made run coding awkward | `MAX_BLOCK` plus `decode_len` makes variable-length tiers first-class, and `CK-06` is the invariant that keeps them inside the trait. |

## Open questions for the specification

The plan text available to this implementation was truncated partway through
§5.4 in this revision, and partway through §7.3 in the previous one. Sections §6
onward, and Appendices A to D, have not been received in the revised form. The
following were decided here on the plan's own principles and should be confirmed.

1. **§8, the driver.** Packing formats and the blocked traversal's panel
   geometry. What exists is a correct, exact, traversal-invariant driver; what
   it is not yet is the cache-blocked one §8 describes. `CD-04` holds either
   way, because it asserts the output is invariant under the choice.
2. **D-12**, cited by §3.3 for the complete accumulator's traversal.
3. **D-15**, cited by §7.3. The previous revision contradicted itself: the §7.2
   table said the `dpbusd` sequence is "Off by default", the §7.3 prose said it
   *is* the default. The model records all three thresholds either way.
4. **D-16**, cited by §6.2 for `Runs`. Implemented as a variable-length tier:
   a code is a run index, `decode_len` is the run's length, and `CK-06` is the
   invariant that keeps the arithmetic downstream dense.
5. **D-7**, cited by §5.2 and followed: the bound is a sealed trait with an
   associated const.
6. **§15**, which defines the release state, has not been received.

## Remaining deviations, and why

| Where | Specification | Implemented | Reason |
| --- | --- | --- | --- |
| §5.3 | `Accumulator::encode<O: Element>` | `encode<O: Element + EncodeFrom<Self>>` | The encode step is a relation between an accumulator and an output type. On `Element` alone it cannot express `Complete<10, -298> -> f32` or `ComplexAcc<i128> -> Complex<i32>`, which would have forced a second method. |
| §5.5 | `MatView` construction is infallible | `MatView::new` returns `Option` | With arbitrary strides and `forbid(unsafe_code)`, a view whose coordinates fall outside its buffer must be rejected somewhere. It is the same category as `NotAProduct` --- non-existence, decided before any arithmetic --- and it cannot be provoked by the values in the buffer. |
| §6.1 | `Codec::decode_into` is the required method | `Codec::decode_element` is required; `decode_into` and `decode_seq` are provided | A composing tier (`Offset`, `Runs`, `Transcode`) would otherwise need a scratch buffer the size of a block, which with no `alloc` means either a hardcoded maximum block size or a heap. Both are arbitrary limitations (R7, R8). |
| §6.2 | `Transcode<C1, C2>` | `Transcode<M, C, In>` with `M: CodeMap` | A decode into the alphabet cannot be the input of another decode. The first half of the composite is a relabelling, which is what `CodeMap` names. |
| §6.2 | `Offset<C> { zero: E }` | `Offset<E, BdIn, BdOut, C>`, with an O(1) check at construction | `d(c) - z` widens the alphabet from `B` to `B + \|z\|`. Carrying one bound would have made the decode either lossy or overflowing. |
| §6.4 | `kappa` is `no_std` and heap-free | the manifest writer is; `uor-addr-1` brings its own allocation | `uor_addr_1::address` takes a `Vec`. The feature is off by default, and `Manifest::write_canonical_json` --- the normative part --- allocates nothing. |
| §5.3 | `Complete<const L: usize>` | `Complete<const L: usize, const MIN_EXP: i32>` | The exponent origin is carried rather than inferred, so two `Complete` values of different origins cannot be combined. |
| R3 | grep for saturating instructions | a saturating operation outside the encode step must carry an `R3-ok:` note | A grep cannot tell a buffer cursor from an accumulator. Making the author say which it is keeps the rule enforceable without making it wrong. |
| R5 | wrapping arithmetic is explicit | it is, *and* the accumulator's arithmetic is deliberately plain | Spelling `wrapping_add` in `acc.rs` would assert that a wrap is possible, when §3.2's whole point is that it is not. The witness is `CT-02`, the overflow-checked build. |
| R2 | static gate | the gate is a source grep; `CU-01` is the definitive one | A source grep cannot see what the optimizer emitted. The grep is falsifiable --- a planted `x + 1.0` in a shipped crate fails it --- but `CU-01`'s disassembly is the real rule, and it is pending. |
