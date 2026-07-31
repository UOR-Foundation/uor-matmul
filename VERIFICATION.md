# VERIFICATION

Which axis of `just vv` discharges which class of claim.

| `just` recipe | Enforces | ID classes |
| --- | --- | --- |
| `just fmt-check` | the diff is reviewable | --- |
| `just model` | R1, R8, R10, R11, R13, and R15 --- the last suspended for the representation-width phase; `model/ledger.toml` records the suspension and `audit-deferral` reads its exemption from there | `CM-01`, and the absence of `CN-*` |
| `just lint` | clippy at `-D warnings`, including the unsafe-documentation lints | --- |
| `just test` | the whole suite | `CS-*`, `CT-*`, `CD-*`, `CB-*`, `CK-*`, `CX-*`, `CA-01`, `CP-01`, `CU-02` .. `CU-05`, `CU-07`, `CG-*` |
| `just purity` | R2 by scope-tracking the source, R3 and R13 by grep, and `CU-01` by disassembly | `CU-01` |
| `just no-alloc` | R7, C1 | `CA-03` |
| `just bdd` | R9 and R4's behavioural half | `CM-02`, `CM-03` |
| `just checked` | every accumulator operation checked, no overflow | `CT-02` |
| `just cross` | the corpus digest off the host | `CA-02` |
| `just cortex-m-run` | the parity checks run on a Cortex-M target under qemu-system, not merely compiled | `CB-11` |
| `just scaling` | fitted exponents, reported never asserted | `CG-01` .. `CG-07` |
| `just census` | the static issue census names a bottleneck per kernel sequence | `CG-11` |
| `just symbol-bandwidth` | the symbol path's achieved bytes/second against the host's own STREAM number, reported never asserted, with byte-identity asserted inside the timed region | `CG-14` |
| `just bridge-sweep` | the float placement bridge's achieved MACs/second against the scalar scaled lanes, reported never asserted, with byte-identity asserted inside the timed region | `CG-15` |
| `just symbol-tabulated` | the symbol tabulated traversal's achieved MACs/second in the scaled integer lane against the bridge, the dense driver, and the oracle, reported never asserted, with byte-identity asserted inside the timed region | `CG-16` |
| `just strassen-sweep` | the sub-cubic recursion's achieved MACs/second against the cubic packed walk on the i32-exact lane, reported never asserted, with byte-identity asserted inside the timed region | `CG-12` |
| `just swar-sweep` | the i64x2 SWAR broadcast sequence's achieved MACs/second against the dot-with-extends sequence and the portable reference, on wasm32-wasip1 under wasmtime, reported never asserted, with byte-identity asserted inside the timed region | `CG-17` |
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
| the `f64_exact` exemption, both directions | an emitted `fmul` inside a function holding the `F64Exact` witness: not reported. The same body without the marker: reported by both halves, as arithmetic on a float in the source and as `fmul` in the disassembly | yes |
| the `f64_exact` marker without the witness | a function named `*_f64_exact` that never mentions `F64Exact`: reported by the source half, which is the half that can see the type | yes |
| the `F64Exact` precondition | `F64Exact::<Bnd<{1 << 26}>, 3>::new()` --- 3 * 2^52 exceeds 2^53 --- fails to compile with the assertion's own message. Checked at monomorphization, so `cargo check` alone does not see it; a test build does | yes |
| `audit-disassembly` (`CU-06`) | a `black_box`-guarded value multiply in `gather_run`'s column loop, reported as `mul` in both `gather_reference_*` bodies. The Mach-O framing defect this exposed --- every body ran to end-of-file, so a dense NEON kernel's `sdot` was attributed to the gather --- is the one the gate actually shipped | yes |
| `audit-deferral` (R15) | a `TODO` in a shipped crate | yes |
| the R15 suspension's exemption | a `TODO` in the admitted log: not reported. The same marker in `README.md`: reported. An `admits` path under `crates/`: rejected by the model's own check before any gate runs | yes |
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
| `CK-10` (arena canonicalization) | duplicates not collapsed by `canonicalize`; a sign-masked comparison is *not* reported --- pattern order never makes `-0.0`/`+0.0` adjacent, and the row says so | yes |
| `CK-09` on the arena tier | an off-by-one `index_of` | yes |
| the `CK-08` arena spelling | `Arena` spelled `Grid` in the canonical manifest | yes |
| `CD-14` | rows and column group swapped in the wide-lane gather | yes |
| `CB-09` | the modular lane's `place` written into the wrong limb (`<< 32` for `Mod32`, `<< 64` for `Mod64`), one at a time --- each failed its own family's end-to-end byte comparison and not the other's | yes |
| `CU-08` | the modular table lane admitted under `Saturating` as well as `Wrapping` --- the census half caught the table running where the exact lane must stream | yes |
| `CB-10` | a sign inversion inside the bound-1 build (the `w == 1` and `w == -1` arms swapped, in both row-specialized and any-row forms) --- the parity sweep failed at the first tile, and the census test's byte assertion failed with the negated product. The AVX2 twin is gated by the same parity test on x86 CI. Also: the bound-1 build's `max_bound` raised to admit bound 2 --- the parity test failed on the missing bound-1 declaration, with the selection half's `choose_table` assertions behind it | yes |
| `CK-11` | the `Sign` decode's bit convention inverted (set is `-1`, clear is `+1`) --- both `CK-11` tests failed, the codec-level one on the first exhaustive code and the gemm-level one on the first shape. `CK-09` did *not* fire: the inversion permutes the table, which the enumeration laws are blind to by design --- the cross-spelling byte identity is the gate for the convention, and it held | yes |
| `CK-12` | the `Ternary` decode's digit convention disturbed, twice: the `-1`/`+1` arms swapped, and the dead digit 3 decoded to `+1` --- each failed the codec-level `CK-12` test on an exhaustive code and the gemm-level one on the first shape; the adds-only bound-1 half did *not* fire under either, because its dense reference shares the tier's decode and only the cross-spelling comparison sees the convention. `CK-09` did not fire under the swap either, for the same reason as under `CK-11`'s | yes |
| `CD-17` | numeric `==` as the float rows' identity (both the element verdict and the contiguous-run comparison) --- all three `CD-17` tests failed: the unit pass counted five rows where bit identity sees two (NaN rows refuse to collapse under `==`), the bit-witness unit test counted five symbols for four, and the end-to-end census left the per-distinct-row closed form | yes |
| `CG-11`, a dropped family | the `table` family classified as `other` in the census --- the conformance test failed on `no table sequence in the x86-64 artifact` | yes |
| `CG-11`, a tool that is not llvm-mca | `LLVM_MCA` pointed at `/usr/bin/false` and at a path that does not exist --- the subcommand failed nonzero, naming the tool it was told to run. Also an impostor that prints the summary counters but no pressure table: the parser failed with `no positive resource pressure` rather than reporting a bottleneck it invented | yes |
| `CG-13` | the materialization built each family's cached list without its first entry (`skip(1)` on the full walk) --- the parity test failed on `i8 tile: the cached list differs from the full walk`, the first family's list comparison | yes |
| `CD-18` (and the arena decode generally) | an off-by-one index in the arena's `decode_element`. The first planting failed only `CK-10` --- `CD-14`, `CD-17`, and `CD-18` all passed, because the harness's reference decoded the operand *through the codec under test*, so a wrong decode shifted both sides of the comparison. The reference now reads the codebook directly with the modulo written out, and the same plant fails `CD-14`, `CD-17`, and `CD-18` on the first shape. `CK-14`'s cross-width test does *not* fire under it: the two widths share the one decode, so a shift shifts both --- that test is consistency between spellings, and the absolute pin is the direct-table reference plus `CK-10`'s fixed positions | yes |
| `CG-14`'s timed-region assertion | one code corrupted after the reference operand was built from the stream --- the byte-identity assertion fired on the first shape (`sym walk 1024x1024x1: the symbol run must give the dense driver's bytes`). Planted one line earlier, before the reference was built, it passed: the reference shares the stream it is derived from, and the row records where the assertion's teeth are | yes |
| `CD-19` | a one-exponent error in the placement scale (`base_a + base_b + 1` handed to the `Scaled` epilogue) --- the byte-equality test failed on the first bridged shape. And the two panels' scale bases swapped in the reification (`A` scaled to `base_b`, `B` to `base_a`) --- failed at the first asymmetric-span case; the one-exponent fill is blind to the swap because its two bases are equal, and the row says so, which is why the sweep carries asymmetric spans. The default driver's auto-selection, twice: `bridge_possible` planted to decline at every type --- the test failed on the type pin itself; and the offer question planted out at the call site --- the byte assertions *passed* (the scalar lanes give the same bytes, which is exactly the vacuity the row exists against) and the side-effect pin failed instead: the offer held packed codes where the table lane would have left the reified operands. The lane-depth term is pinned the same way, as the table's own capacity arithmetic (`lane_depth(2^27) == 511`) plus a depth-past-the-lane call whose offer must not hold the reification | yes |
| `CD-20` | a one-exponent error in the lane's placement (`exponent + 1` in `Scaled64::place_scaled`) --- failed on the first shape, the tabulated bytes doubled against the reference. And a lane that over-runs its capacity (the run depth planted at four times the walk's bound, so a reduction of 40000 at-bound products sums in one run instead of two) --- failed the worst-case fill, which exists because the mixed-sign fills cancel away from the bound and a capacity lie is invisible under them; the row records that the capacity plant only has teeth where the data reaches `2^63` | yes |
| `CG-16`'s census assertion | the sweep was first written reading the census after the timed reps, so the printed counts were `reps x` the per-call counts --- caught by comparing against the closed forms (`decodes` is `m * k + 2 * code_space` exactly) before any figure was recorded; the census is now snapshot after one untimed run, as `CG-14`'s | yes |
| `CB-12` | a field overlap one bit wide (the field-1 spread shift planted as 12 for 13, so two fields share bit 20) and a halved compensation term (`<< 13` for `<< 14` in the extraction's `16384 * d`), one at a time --- each failed the parity test on the first packed depth under the wasmtime run (`SWAR broadcast disagrees at kc=1`). The tile-contract half records a real bring-up defect: the sweep's first draft modelled the tile as accumulate-into and the kernel's first draft accumulated into `acc`, where every kernel's contract is overwrite --- the timed-region assertion fired on the second rep at 2x the one-call sum, and the explicit garbage-filled-tile assert now covers the kernel side, which the zeroed-`acc` parity loops cannot see | yes |
| `CD-21` | a wrong sub-product assembly (`M2` and `M5` swapped in the combination) --- five of the six `CD-21` unit tests failed, the hand-computed `2x2` first; only the plan test passed, which is about level counting, not values. And the offer-decline rule, planted in layers: the per-product decline alone did *not* fire, because the plan's offer rule stops a starved offer upstream (the plan test asserts exactly that, and fired when the plan's rule was planted out); with the top-level guard out as well, the offer-ladder byte test failed at the first starved offer --- a `split_at_mut` out of range, which is the corruption a safe language can write | yes |
| `CD-22` | the facade pointed back at the reference (the state the claim exists against): the byte assertions *passed* --- the reference's bytes are the same bytes, which is exactly the vacuity the row exists against --- and the sentinel-offer pin failed instead (`the suggested offer was never written`), because the packed traversal packs its panels into the offer and the reference never touches it. And the census planted silent (`RouteCensus::routed` with an empty body): both census-reading tests failed, the kernel-route one on `must run a table kernel, got None`, the zero-scratch one on the same absence --- a witness that records nothing fails as loudly as a wrong route | yes |
| `CG-12`'s harness, twice | the first draft's "exact cubic" baseline was the modular lane (`EncodeMode::Wrapping` selects it), so the recursion was being compared against a lane it does not factorize --- caught when the exact lane's rate was measured directly at 18 Gmac/s against the column's 26; the exact baseline now runs under `Saturating`, which declines the modular arm. And the first draft's base case ran the per-tile streaming path at deep levels: the plan reserved `suggested_accumulators`, which answers zero whenever `k <= KC`, but the grown bounds make the base case's *lane* shallower than its `k` --- found by an overhead that scaled with `n^3` (L=3 at 2048 read 8.2 Gmac/s where the products alone price at 16), fixed by offering the output block of accumulators unconditionally | yes |
| `CG-18` | a multiply charged per gather read (`ledger.multiplied(computed * depth * rows)` beside the gather's `read`) --- the test failed on the spot (`the only multiplies are the build's (\`code_space * block * rows\` per slot)`), because the running table's multiply count is a closed form and one more is visible exactly. The selection half was falsified the same way the derivation already is: the boundary is recomputed from the host's own declarations, so a hardcoded break-even fails on the ISA it was not derived for --- that failure is the reason the test carries no number. And a real bring-up defect, caught by `CB-05`'s wasm leg rather than by planting: the distinct-column stream built its base-256 digits as `j / space.pow(p)`, and `pow` overflows a 32-bit `usize` at `p = 4`, reading as a divide by zero under wasmtime; the digits are now taken by repeated division | yes |
| `CB-11` (the Cortex-M executor) | the portable tile's row index written `i ^ 1` in `isa/portable.rs` --- the thumbv6m executor printed `CB-11: FAIL` with the assert's location and qemu exited 1, so `just cortex-m-run` failed at the first family, before any PASS marker | yes |

`audit-disassembly` deliberately does *not* catch a float add the optimizer
removed, because such an add is not in the shipped kernel. The gate reports the
binary rather than the source, which is the only thing worth reporting.

## What the suite does not establish

The upstream formalization's theorems. `CL-MM01` .. `CL-MM04` are cited, never
re-derived, and a `CX-*` result is evidence that the kernels realize the
identity --- not a proof of it. `model/authorities.toml` records what is cited
and from where; the meta-gate fails the build if any document blurs the two
registers.
