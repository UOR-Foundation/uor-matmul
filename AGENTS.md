# AGENTS

The standing brief for anyone --- human or otherwise --- changing this
repository.

## Read first

`README.md` for the thesis and the non-goals, then `ARCHITECTURE.md` §"The one
sentence". Everything else follows from those two.

## The rules, in the order they are most often broken

1. **The model is the single source (R10).** Every constant lives in
   `model/*.toml`. `crates/uor-matmul-core/src/generated.rs` and
   `CONFORMANCE.md` are *generated*; editing either is a mistake the gate
   catches. Run `just model-write` after changing the model. Constants have two
   honest kinds: derived, and discovered by measurement. A measured constant (a
   blocking factor, a prefetch distance) is recorded with what measured it;
   inventing a derivation for one is fiction (R1's allowlist).

2. **Nothing is deferred (R15).** No `TODO`, no `stub`, no `placeholder` section, no
   capability behind a flag that turns it off. If a change cannot be finished,
   it should not be started --- and `cargo xtask audit-deferral` will say so.

3. **One answer, many factorizations (R13, C5).** There is no fast path and no
   careful path. Lanes, traversals, tile and reduce kernels, narrow and wide
   panels, table and dense sequences are *factorizations* of one identity: they
   must be differentially tested against the portable reference, and the
   reference is never optimized (R6). What makes them one method is that they
   produce the same bytes, which the `CD-*` gates assert. If a change
   introduces a second way of computing the same thing that can give a
   different answer, it is wrong even if both ways agree today.

4. **The operation is total (R14, C6).** `gemm` returns `()`. If a change wants
   to return a `Result`, the question to ask is "what object does not exist?" ---
   and if the answer is anything about the data's size, depth, or magnitude, the
   change is wrong.

5. **Levels are load-bearing (R4).** A claim is `some-true` (reproduced from an
   authority), `build` (constructed here and validated), or `open` (measured and
   reported). Writing "proves" about an `open` claim fails the meta-gate, and it
   should.

## Adding a capability

In this order, because the order is the discipline:

1. A row in `model/ids.toml`, with its level.
2. A scenario in `features/suites/`, tagged with the ID.
3. A failing test whose name **ends in the ID**, lowercased with underscores.
4. The parametric implementation.
5. `just vv`.

Steps 1--3 before step 4 is R9. The meta-gate enforces all of it: an ID with no
scenario, a scenario with no ID, or an ID with no test all fail `just bdd`.

## Adding a backend

Add a module under `crates/uor-matmul-kernels/src/isa/` exporting a
`KernelSpec`, and add it to the family's entry list in `spec.rs` --- one line
in the `family!` invocation behind `available_i8`, `available_i16_modular`,
`available_i32_exact` and so on; there is no single `available()`. The one
list generates both the full walk and the cached walk the driver selects from
(`CG-13`), so a line added once appears in both. Touch no driver code --- if
you need to, the abstraction is wrong and that is the thing to fix. The
differential test picks it up automatically.

## Adding an element type

An `impl Element`, plus **one of** `IntegerElement`, `FloatElement`, or neither.
The third arm is not an oversight: `Trop<E>` implements `Element` alone, and
that absence is what excludes the sub-cubic recursion (which needs
`IntegerElement::sub`) and the `Linear` epilogue (whose `beta * C` has no
reading under `(max, +)`) *by construction* rather than by a refusal. Before you
reach for a trait, ask which operations the new type genuinely has; a trait it
does not implement is a capability it cannot be handed.

If you find yourself adding a branch anywhere else, the type is not being added
parametrically.

## Adding a semiring instance

Rarely. There are two, and a third would need a reason from the operation
census, not from convenience. If there is one:

1. A new `Element` type carrying the algebra, with its own `Acc`. Do **not** add
   a semiring parameter to a traversal --- `Element::Acc` is *not a parameter and
   not a choice*, and a parameter beside the element type would make it a
   function of two things.
2. A `Semiring` marker in `uor-matmul-core`, declaring `IDEMPOTENT` and its
   witnesses, so that `CK-16`'s one body quantifies over the new instance too.
   The declaration is compared against the measurement; do not derive one from
   the other.
3. An `EncodeFrom` for the output alphabet, routed through the *same*
   `encode_i128_into` the ring family uses. A second encode step would be a
   second method (R13).
4. An epilogue, if `⊗` by a scalar means anything in the new algebra. It gets
   its own trait for `⊗`, so that an accumulator of one algebra cannot be handed
   the other's epilogue.
5. A derived accumulator width with its own gate. State which terms are present
   and which are absent, and why: the tropical width has no `MAX_K_BITS` term
   and `CA-04` is the row that says so.

## Writing a gate

A gate that cannot fail is worse than no gate, because it reads as evidence.
Before adding one, plant the defect it exists to catch and confirm it fires;
then record that in `VERIFICATION.md`'s falsifiability table, which is the
running list. Gates in this repository have been found vacuous repeatedly and in
every flavour: a differential test comparing the reference against itself, a
claim discharged by a compile rather than a run, a job whose `-p` list omitted
the crate it was named for, a feature nothing ever built, and examples in the
`README` that nothing compiled. Assume yours is one until you have watched it
fail.

## Comments

Explain *why*, not *what*. The code says what it does. A comment earns its place
by recording the reason a decision went one way when it could plausibly have
gone another --- a bound that is derived rather than chosen, an ordering that is
normative rather than conventional, a saturation that is inside the encode step
rather than in an accumulation.
