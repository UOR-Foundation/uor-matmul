# VERIFICATION

Which axis of `just vv` discharges which class of claim.

| `just` recipe | Enforces | ID classes |
| --- | --- | --- |
| `just fmt-check` | the diff is reviewable | --- |
| `just model` | R1, R8, R10, R11, R13, R15 | `CM-01`, and the absence of `CN-*` |
| `just lint` | clippy at `-D warnings`, including the unsafe-documentation lints | --- |
| `just test` | the whole suite | `CS-*`, `CT-*`, `CD-*`, `CB-*`, `CK-*`, `CX-*`, `CA-01`, `CP-01`, `CU-02` .. `CU-05`, `CU-07`, `CG-*` |
| `just purity` | R2 by scope-tracking the source, R3 and R13 by grep, and `CU-01` by disassembly | `CU-01` |
| `just no-alloc` | R7, C1 | `CA-03` |
| `just bdd` | R9 and R4's behavioural half | `CM-02`, `CM-03` |
| `just checked` | every accumulator operation checked, no overflow | `CT-02` |
| `just cross` | the corpus digest off the host | `CA-02` |
| `just scaling` | fitted exponents, reported never asserted | `CG-01` .. `CG-07` |
| `just features` | every optional feature compiles, and `kappa`'s tests run | --- |
| `just fuzz` | totality over unstructured input | `CT-01`, `CT-03`, `CK-06` |
| `just miri` | undefined behaviour in the crate that has the `unsafe`; `CU-07`, under `just test`, is what asserts it is pointed there | --- |

## The two halves of R2, and why both are needed

`audit-purity` reads the source and `CU-01` reads the emitted instructions.
Neither covers the other, and that was measured rather than assumed:

- `CU-01` sees only what was codegen'd. An uncalled `pub fn` is not in the rlib
  at all --- the symbol is absent from a 1.9 MB rlib and from every `.s` the gate
  reads --- so a float add sitting in one is invisible here, and codegen'd in a
  *downstream* build where these gates do not run.
- `audit-purity` cannot see what the optimizer emitted, and cannot type-check.
  It tracks which names carry floats: parameters and `let` bindings whose
  declared type mentions `f32` or `f64`, float literals, `as` casts, the elements
  of a float slice bound by a `for`, and aliasing chains between them.

Both are run by `just purity`, and both are on the falsifiable list below.

## Every gate is falsifiable

A gate nobody has seen fail is indistinguishable from a gate that cannot. Each
of these has been checked by planting the defect it exists to catch:

| Gate | Planted defect | Reported |
| --- | --- | --- |
| `check-constants` (R1) | a model numeral restated in a shipped crate | yes |
| `audit-limits` (R8) | a `Result` over an unsanctioned error | yes |
| `audit-purity` (R2) | eight float-arithmetic plants: two typed params added, a `f64` multiply, a typed `let`, a `for` over `&[f32]`, `mul_add`, an `as f32` cast, two `*const f64` derefs, and a two-step aliasing chain | yes |
| `audit-disassembly` (`CU-01`) | a `black_box`-guarded float add inside an emitted function, reported as `addss` | yes |
| `audit-disassembly` (`CU-06`) | a `black_box`-guarded value multiply in `gather_run`'s column loop, reported as `mul` in both `gather_reference_*` bodies. The Mach-O framing defect this exposed --- every body ran to end-of-file, so a dense NEON kernel's `sdot` was attributed to the gather --- is the one the gate actually shipped | yes |
| `audit-deferral` (R15) | a `TODO` in a shipped crate | yes |
| the honesty meta-gate (R4) | an ID with no test | yes |
| the differential comparator | a one-element difference | yes |
| the ulp metric | `+0.0` against `-0.0`, and a known one-ulp pair | yes |
| the scaling fit | a known exponent, and too few points | yes |
| the NumPy digest check | a published SHA-256 vector | yes |
| `just features` | `AddressOutcome::label` for `.address` --- the defect `kappa` actually shipped, unbuilt by any gate | yes |
| the README examples | a wrong asserted value in a fenced block; they are doctests now | yes |
| `just miri` | the raw face's window made one element too long: passes the native run, Undefined Behaviour under Miri | yes |
| the raw-window tests (`CS-05`) | the same window made one element too short, caught by all three | yes |
| `CU-07` | `-p uor-matmul-kernels` dropped from the job, from the recipe, and from both --- the last is the defect this workspace actually shipped | yes |

`audit-disassembly` deliberately does *not* catch a float add the optimizer
removed, because such an add is not in the shipped kernel. The gate reports the
binary rather than the source, which is the only thing worth reporting.

## What the suite does not establish

The upstream formalization's theorems. `CL-MM01` .. `CL-MM04` are cited, never
re-derived, and a `CX-*` result is evidence that the kernels realize the
identity --- not a proof of it. `model/authorities.toml` records what is cited
and from where; the meta-gate fails the build if any document blurs the two
registers.
