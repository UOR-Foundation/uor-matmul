# uor-matmul

**Decode the code, accumulate exactly, encode once.**

MatMul as an operation on *coded* operands. A weight tier is a codec
`d : Code -> Alphabet(B)`; the operation is the exact accumulation
`sum a_i * d(c_i)`; the result is encoded once into whatever the caller asked
for. The codec is not an argument of the arithmetic, which is why the same
result survives a change of tier, a change of reduction schedule, and a change
of substrate.

There is one answer, and it has many factorizations. Not one method for
integers and another for floats; not a fast path and a careful path; not a SIMD
kernel and a scalar fallback. Every entry point is that sentence at a different
instantiation, and the tree is full of legitimate reorganizations of it ---
three lanes, three traversals, tile and reduce kernels, narrow and wide panels,
table and dense sequences. What makes them one method rather than many is that
they all produce the same bytes, which the `CD-*` gates assert rather than
argue. The library holds nothing in reserve for cases it finds hard, because
the exact accumulation has no hard cases.

In the shape every GEMM is spelled --- `(m, k, n)`, leading dimensions, `alpha`
and `beta`:

```rust
use uor_matmul::{slice, suggested_scratch, Shape};

let a = [1i8, 2, 3, 4];
let b = [5i8, 6, 7, 8];
let mut c = [0i32; 4];

// The panel buffer is the caller's, because the library never allocates.
// `&mut []` is valid and gives the same bytes, more slowly.
// `workspace_report` names the plans in bytes: the full-depth suggested
// offer, and the bounded one, which stops growing with `k`.
let mut scratch = vec![0i8; suggested_scratch(Shape { m: 2, k: 2, n: 2 })];

slice::gemm(2, 2, 2, &a, &b, &mut c, &mut scratch).unwrap();
assert_eq!(c, [19, 22, 43, 50]);

// `C := alpha*A*B + beta*C`, with leading dimensions, when you want them.
slice::gemm_ex(2, 2, 2, 3, &a, 2, &b, 2, -1, &mut c, 2, &mut scratch).unwrap();
```

`slice::gemm_float` and `slice::gemm_float_ex` are the same call over `f32` and
`f64`. `raw::sgemm` and `raw::dgemm` are signature-identical to
`matrixmultiply`, for code that already has a call site. And the view API
underneath takes arbitrary strides --- negative, zero, transposed --- because
transposition is a stride and not a mode:

```rust
use uor_matmul::prelude::*;

let a = [1i8, 2, 3, 4];
let b = [5i8, 6, 7, 8];
let mut c = [0i32; 4];

let av = MatView::row_major(as_alphabet_full(&a), 2, 2).unwrap();
let bv = MatView::row_major(as_alphabet_full(&b), 2, 2).unwrap();
let cv = MatViewMut::row_major(&mut c, 2, 2).unwrap();

// The one fallible step: does this product exist?
let mut t = Triple::new(av, bv, cv).unwrap();

// The operation itself cannot fail, so it returns `()`.
gemm(&mut t, &Linear::OVERWRITE, GemmOptions::default(), &mut Scratch::none());
assert_eq!(c, [19, 22, 43, 50]);
```

The `slice::gemm` and `gemm` those examples call are the auto-selecting
entry: at the caller's offer and the host's declarations they run the
kernelized factorization the offer admits --- a kernel from the CG-13 table at
the suggested scratch, the streaming reference at none --- and every route
returns the same bytes at every offer (`CD-22`). The reference traversal the
kernels are validated against is never hidden: it stays directly callable as
`uor_matmul_gemm::gemm` (R6), and the route the selection took is readable
through `gemm_auto_counted`'s census rather than inferred from a clock.

## The six hard constraints

| # | Constraint |
| --- | --- |
| C1 | `no_std`, zero heap allocation, compiles for wasm and embedded |
| C2 | The UTQC method informs the implementation: docs-as-code, BDD-first, committed oracles, a three-level honesty ledger with a meta-gate |
| C3 | Scaling is compared against the oracle's scaling, as a fitted exponent with a confidence interval |
| C4 | No arbitrary limitations; completely parametric; arbitrary sizes and inputs |
| C5 | One answer, many factorizations; no classical approach and no fallback to a lesser method |
| C6 | Never errors on valid input; arbitrary means arbitrary |

## The library has no envelope

> For every input the machine can represent, this library returns the exact
> value of the sum, encoded once into the caller's requested output type. There
> is no domain in which it approximates, no depth at which it wraps, and no
> shape at which it declines.

The accumulator's width is a compile-time function of the element type alone,
sized against the largest `k` the machine can address, so overflow is
*unreachable* rather than guarded:

| `E` | worst-case bits | accumulator |
| --- | --- | --- |
| `i8` | 79 | `i128` |
| `i16` | 95 | `i128` |
| `i32` | 127 | `i128` |
| `i64` | 191 | `Limbs<3>` (192 bits) |
| `Trop<i8>` | 10 | `TropAcc<10>` |
| `Trop<i64>` | 66 | `TropAcc<66>` |

`MAX_K_BITS` is *declared* in `model/constants.toml`, not probed from the host,
so this table is the same table on a 64-bit host, a 32-bit host, and wasm32. A
32-bit host cannot reach that depth, which makes the width conservative there
and never wrong anywhere.

The last two rows have **no depth term at all** --- ten bits where the ring's
`i8` needs seventy-nine --- and their absence is the arithmetic content of the
selection half. A sum grows with the reduction; a maximum does not.
`max_p (a_p + b_p)` is bounded by `2B` whatever `k` is, so `MAX_K_BITS` never
enters the derivation and `CA-04` is the gate that says the width is the same
number at depth one and at the deepest reduction the machine can address.

There is no ladder, no policy, no promotion, and no `k_max` in the public API.
`fits_narrow` survives only as an internal predicate answering one question:
*may this tile be accumulated in a narrower register without changing the
answer?* Both sides compute the same integer, so the choice is invisible, has no
failure mode, and is never surfaced to the caller. That is what separates an
optimization from a fallback: a fallback changes the answer or the guarantee,
and this changes neither.

## The error surface, in full

Two variants, both meaning *the requested object does not exist*, both reported
at view construction, before any arithmetic is named:

- `Nonconformant` — A is `m x k` and B is `p x n` with `k != p`.
- `OutputAliasesItself` — the output strides map two coordinates onto one cell.

Neither can be caused by the size, depth, magnitude, or distribution of the
data, nor by the host. Deliberately absent, each absence load-bearing:

| Absent | Why |
| --- | --- |
| a depth or size limit | the accumulator cannot overflow |
| an alphabet violation | every value of `E` is in `Alphabet<E, Full<E>>` |
| a scratch error | scratch is an offer; too little means a different traversal, not a failure |
| an accumulator-policy error | there is no policy |
| an epilogue capacity error | the complete width derives arbitrary `i64` scalar growth and both terminal terms before the one encode |
| a backend-unavailable error | the portable backend is always present and always correct |
| a non-finite-input error | non-finite floats are codes with IEEE-defined behaviour |

`i8::MIN` is not rejected. It is an element of `Alphabet<i8, Full<i8>>`, whose
bound is 128. Refusing a representable value would be an arbitrary limitation
dressed as rigour.

## Floats are codes

An IEEE 754 value is a bit pattern naming an exact dyadic rational. That is a
codec, with the same shape as a codebook, so the float path is not a second
method: decode exactly, accumulate into a *complete accumulator* spanning the
entire product exponent range and the integer-scaled terminal expression, and
round once at the end.

Between decode and encode, finite values stay in the UOR Atlas. The canonical
reference normalizes each dyadic to the unique finite non-adjacent Laurent word
in `Z[X, X^-1]/(X - 2)`, reads sign as the Atlas modality involution, and gives
every signed grade the mixed-radix address `(word, scope, context)`. `CK-19` and
`CK-20` execute that construction as formal witnesses; they are not runtime
buffers or a second implementation.

The optimized embedding projects the same normalized coefficient directly into
balanced signed octets, the coordinates of the complete signed-`i8` lookup
alphabet. Projection repeats the identical quotient step until the coefficient
is zero. More required precision therefore means another self-similar word,
not a format arm, cutoff, or representation change. `CD-30` differentially
pins this optimized contraction to the canonical finite-NAF reference for both
float formats and every public entry.

There is one shipped float arithmetic, exposed through several factorizations.
At each reduction position it contracts equal `u+v` Laurent diagonals by lookup
and addition into one bounded product carrier per output cell. Only after every
diagonal of that mathematical source product has arrived does the carrier
resolve its sign and magnitude;
one Euclidean fracture in the signed-place radix `i128::MAX + 1` then yields a
low digit and at most one unit high digit, placed at the base grade and its
radix-successor grade. The only selector globally compares every eligible
group-one tile, narrow, and reduce lookup declaration by model-derived executed
work, including exact output-cell residency, live-only product initialization,
the fixed Atlas workspace, and full tiles plus row, column, and corner edges.
One exact frame owns every live cell of a tile for its entire reduction; its
streamed source cell is one six-state boundary quotient plus the finite payload,
and reused coordinate words clear only their retired suffix. A model-generated
contiguous capacity dispatch gives the frame exactly `rows * columns` cells,
with no replay or maximum-tile allocation. There is no scalar support mask, population count,
common-gauge route, tuned threshold, significand multiply, float arithmetic,
whole-operand integer reification, or reserve route (`CU-11`, `CG-22`). The
interval and projector objects used by `CD-31` and `CK-20` remain borrowed
theorem certificates, not objects materialized by the hot loop. Each offered
sixteen-byte `PackedCode` slot becomes the source's balanced-octet/grade
projection in place, so panels reuse both decode and projection without a
second buffer or copy. An empty offer streams one reduction position through
the same fixed workspace and changes reuse but never arithmetic, allocation,
or output bytes (`CA-05`, `CD-19`, `CD-30`).

The public coded traversal can also force this arithmetic through its
block-one q table, but automatic selection does not speculate on that route.
The final current-source `CG-16` fit predicted the table for two holdouts with the
identical pre-admission structural key; their unlike values produced opposite
decisive clock outcomes (`0.1821 +/- 0.0397` and `2.9348 +/- 0.3215`
table/decline).
`CS-10` forbids inspecting those values to choose a route, so that candidate is
rejected and the value-blind block-one default remains the coded Atlas decline.
Forced tabulation remains available and byte-identical for a caller that names
it; no rejected clock rule becomes a model constant.

Consequences, stated plainly:

- The `f32` result is the correctly-rounded value of the exact sum. It is
  schedule-independent, tile-independent, and substrate-independent, which no
  classical `f32` GEMM is.
- It is therefore **not** bit-identical to `matrixmultiply`, `faer`, or BLAS in
  general. Where they differ, they differ by their own rounding error, and this
  repository reports that error in ulps against the exact value rather than
  matching it.
- The integer paths are bit-identical to the integer oracles **everywhere**,
  including past the depth at which those oracles wrap. Reduction modulo `2^w`
  is a ring homomorphism, so reducing the exact sum once at the end equals
  reducing at every step: `EncodeMode::Wrapping` into a `w`-bit output
  reproduces a classical wrapping accumulator's bytes exactly, at every depth.
  There is no exempted region. A caller who wants the mathematical value
  instead asks for `EncodeMode::Saturating`, and both answers come from one
  accumulation.
- Non-finite inputs are codes too. NaN and infinity propagate by the IEEE rules;
  they are not an error condition.

## Max is a code operation too

The operation census this library is written against names two products, not
one: *matrix products under complete accumulation*, and *max*. The second is the
`(max, +)` semiring --- `⊕` is `max`, `⊗` is addition --- and it is where a
transformer's selection lives once the softmax is gone.

It is not a second method, and it is not a second driver. The semiring is
carried by the **element type**: `Trop<E>` is the alphabet `E ∪ {-inf}`, its
`mac` is `acc = max(acc, a + w)`, and its accumulator's `combine` is `max`. The
*reference traversal* names `E::mac` and `combine` and nothing else, so
instantiating `E` at `Trop<i8>` is the whole of the change there:

```rust
# use uor_matmul::prelude::*;
# use uor_matmul::{Scratch, Triple};
let a = [Trop::finite(1i8), Trop::finite(2), Trop::finite(3), Trop::finite(4)];
let b = [Trop::finite(5i8), Trop::finite(6), Trop::finite(7), Trop::finite(8)];
let mut c = [Trop::<i32>::NEG_INF; 4];

let av = MatView::row_major(as_alphabet_tropical(&a), 2, 2).unwrap();
let bv = MatView::row_major(as_alphabet_tropical(&b), 2, 2).unwrap();
let cv = MatViewMut::row_major(&mut c, 2, 2).unwrap();
let mut t = Triple::new(av, bv, cv).unwrap();

uor_matmul::driver::gemm(&mut t, &MaxPlus::OVERWRITE, GemmOptions::default(), &mut Scratch::none());
// max(1+5, 2+7) = 9, and so on.
assert_eq!(c, [Trop::finite(9), Trop::finite(10), Trop::finite(11), Trop::finite(12)]);
```

That is `uor_matmul::driver::gemm` --- the reference traversal, the one R6 keeps
unoptimized and every other factorization is validated against. It is *not* the
`gemm` the first example on this page calls: that one is `gemm_auto`, which
selects a kernelized factorization, and every such factorization is bounded on
`Kernelized`, which `Trop<E>` deliberately does not implement. So the census's
second product runs on the reference traversal and on the packed `(max, +)`
sequences (`CB-13`), and not through the auto-selecting front door. Saying which
is not a caveat --- it is the difference between a claim about *the* driver and
a claim about *every* driver, and the second one is false.

`CD-29` asserts the first: the reference traversal computes both products, and
it asserts the *structure* as well as the value --- a planted branch on the
accumulator's width, which is exactly a branch on which semiring the caller is
in, fails it. `CK-16` asserts that the semiring laws hold at *every* instance
while idempotence holds precisely at the tropical one and fails precisely at the
ring.

Three consequences, each load-bearing:

- **The semiring zero is the pad and the mask.** A padded position decodes to
  `Element::ZERO`, which here is `-inf`; a masked position is the semiring zero.
  They coincide by construction, so a masked shape and a zero-padded shape are
  byte-identical and there is nothing to reconcile (`CK-17`).
- **`-inf` is a value of `Trop<E>`, not a reserved pattern inside `E`.**
  `i8::MIN` remains a perfectly ordinary finite tropical element, exactly as it
  remains an ordinary member of its own alphabet in the ring family. A sentinel
  would also have made `⊗` at the semiring zero an `i8::MIN + a` that overflows,
  which the checked build traps and C6 forbids --- so the absorbing law is
  spelled as *no arithmetic is performed at all* (`CT-08`).
- **`Trop<E>` is deliberately not an `IntegerElement`.** That is what excludes
  the sub-cubic recursion, which needs additive inverses the semiring does not
  have, and the `Linear` epilogue, whose `beta * C` has no reading under
  `(max, +)`. Excluded by construction, not by a runtime check.

A selection also answers a question a sum cannot: *which* term achieved it.
`gemm_selected` writes that witness beside the product, under D-6's tie-break
--- the smallest index --- by either of two mechanisms whose bytes are identical
at every shape, degeneracy and offer including none (`CD-24`, `CD-25`).

## When the operand is a code, the product is a table read

The weights arrive as codes. Decoding them and multiplying issues the same
`m*k*n` products a dense GEMM does, plus a decode. Indexing a table *by* the code
issues far fewer:

```text
T[i][p][c] = sum over t < Bk of  A[i, p*Bk + t] * decode(c, t)
C[i][j]    = sum over p       of  T[i][p][ index_of(w[p][j]) ]
```

The column loop under that is one table read and one add per code, covering a
whole codeword, with **no multiply in it at all** — asserted by the operation
census and by reading the emitted instructions (`CU-06`).

This is exact for the same reason tiling is: a sum is a function of the multiset
of its products, so regrouping them changes nothing, and `CD-13` asserts the
bytes. A classical `sgemm` cannot do it at all — its `T[c]` would carry its own
rounding error and reusing it across `n` columns would propagate that error `n`
times. **Tabulation is available only to an exact library.**

### The density, which is the claim

One 256-bit add covers `8 * block` products at a 32-bit lane: eight output rows,
each carrying a whole codeword. At `Book<256,8>` that is **64 products per
arithmetic instruction**. `vpdpbusd`, the densest integer instruction x86 has, is
32 and cannot be told to cover more. The table's density is a property of the
*codec*: a codebook naming a longer block is a denser instruction, with no change
to the hardware.

Measured on one AVX2 core (AMD EPYC 7763, no VNNI), `Book<256,8>`, against this
library's own packed AVX2 tile path handed the weights *already decoded*:

| `m x k x n` | table | packed | |
| --- | --- | --- | --- |
| `64x1024x4096` | **63.9** | 24.9 | 2.6x |
| `64x1024x16384` | **57.3** | 24.0 | 2.4x |
| `64x4096x4096` | **48.9** | 14.4 | 3.4x |
| `256x1024x4096` | **52.0** | 33.6 | 1.6x |
| `17x1032x1021` | **31.0** | 13.8 | 2.3x |
| `1x1024x4096` | **8.4** | 1.1 | 7.2x |

Gmac/s. Every figure is `open`. Where the kernels win the library hands them the
work — `1000x512x512` is 39.0 against 39.8, and `1x8192x1` is `n*k` decodes for
`n*k` products, which no method beats. `ANALYSIS.md` carries every shape, and
carries what is still short and what has not been attributed.

## Non-goals

| # | Non-goal | Kind | Reason |
| --- | --- | --- | --- |
| N1 | Reproducing another library's float rounding | mathematics | this library computes the correctly-rounded result of the exact sum. A classical GEMM computes an order-dependent approximation of it. Where they differ, the difference is the other library's rounding error, and this repository measures it rather than matching it. |
| N2 | A proof development in this repository | scope | the formalization is upstream and is cited. This repo is a Rust library and is judged on the library. |
| N3 | Any quality claim about a codebook | discipline | VQ quality is measured per (model, codebook) and reported `open`, never asserted |
| N4 | A second method for any case, however hard | design | there is nothing this library does not do with decode-accumulate-encode, so there is nothing left over for a second method to cover |

Note what is **not** on that list. **Throughput is not a non-goal.** Float work
is admitted only through the pure-UOR operation census, and route selection is
derived from that census rather than from a hand-restated size threshold
(`CG-22`). Performance claims remain measurements: `CG-21` reports latency,
traffic and throughput with output poisoning before and complete byte checks
after every calibrated batch; only production calls are inside the timer.

Asymmetric quantization is not a non-goal: a zero point is the codec `d(c) = c - z`, expressed as `Offset<C>`. A reduction
depth is not a non-goal: the accumulator cannot overflow. An unaligned or prime
shape is not a non-goal: padding with the alphabet's zero is exact. A float
input is not a non-goal: it is a code.

## Claim discipline

Every claim carries one of three honesty levels, and the build fails if the two
registers are blurred:

| Level | Meaning |
| --- | --- |
| `some-true` | reproduced from an authority — the upstream formalization, a published instruction semantics, an IEEE 754 rule. **Not established here.** |
| `build` | constructed here and validated against its oracle. Evidence that the kernels realize the identity, **not a proof of it**. |
| `open` | measured and reported, **never asserted**. |

A cross-library agreement is `build`, never `some-true`: it is evidence that the
kernels realize the identity, not a proof of it. The identity itself is cited
from upstream and says nothing about any binary in this repository.

`CONFORMANCE.md` is generated from `model/ledger.toml`, so a claim cannot exist
in the documentation without a ledger row, or in the ledger without appearing in
the documentation.

## Repository layout

| Path | What it is |
| --- | --- |
| `model/` | the single source of every constant, tier, oracle, and claim |
| `features/suites/` | one Gherkin scenario per conformance ID |
| `oracles/` | committed external artifacts, with provenance and checksums |
| `crates/uor-matmul-core` | alphabet, borrowed Atlas carrier/projectors, accumulator, reference accumulation, views. `no_std`, no `alloc`, `forbid(unsafe_code)`, no float arithmetic |
| `crates/uor-matmul-codec` | the `Codec` trait, every tier, and the E8 codebook |
| `crates/uor-matmul-kernels` | one module per ISA: the dense tile sequences and the table sequences. The only crate that writes `#[target_feature]`, which is why every sequence lives here |
| `crates/uor-matmul-gemm` | the driver: traversal, scratch, epilogue, tile partition |
| `crates/uor-matmul` | the facade, and the raw-pointer face |
| `crates/uor-matmul-executor` | dev/CI only: the Cortex-M runner --- CB parity executed under qemu-system, because parity is a run, not a compile |
| `crates/uor-matmul-model` | build-time: parses `model/*.toml`, generates the Rust consts and `CONFORMANCE.md` |
| `crates/uor-matmul-validate` | dev/CI only: oracle adapters, the differential harness, the scaling fits |
| `crates/uor-matmul-conformance` | dev/CI only: the BDD runner and the honesty meta-gate |
| `xtask/` | the gates |
| `fuzz/` | totality targets |

`just vv` is the normative acceptance gate. `VERIFICATION.md` maps each of its
axes to the claims it discharges, and lists the defect that was planted to prove
each gate can fail. `VALIDATION.md` is how a third party reproduces all of it
without trusting this repository.

## The `std` feature and which kernel runs

`std` buys runtime CPU feature detection and nothing else in the numerical
path. The facade's `std` forwards to both numerical crates; a consumer of
`uor-matmul-gemm` directly enables *its* `std` for the same thing. Without it a
backend is available only if the *build* declared its target
feature, so a hosted build with neither `std` nor `-C target-cpu=native` runs
the portable kernel --- correctly, and several times slower than the machine can
manage. On an embedded target that is right; on a server it is not what was
meant. `uor_matmul::kernels::available_i8()` says which kernels a build can run.

## Performance

`ANALYSIS.md` §"Against the oracles" has the integer sweep --- throughput,
latency, and fitted exponents from `n = 1` to `n = 1024`, beside every oracle.
`MEASUREMENT-LOG.md` §"Current pure-Atlas CG-16 and CG-21 measurement record"
records the completed block-one selector experiment: its same-key H01/H02
holdout rejected the fitted `CG-16` candidate, so the value-blind default still
declines. The final current-source `CG-21` sweep records the current one-frame
implementation: all exact-byte guards passed; f32 one-grade measured
`1.089 +/- 0.313` us and full finite range measured `221.9 +/- 23.9` us with an
offer, while the f64 full finite range measured `554.3 +/- 63.8` us without
one. The two f32 full-range intervals are faster than the preceding
source-pinned run despite no reachable x86 production change. They are retained
as `open` host observations, not an attributed
regression or a selector threshold.
Two classes remain separate, because confusing them is the oldest mistake in
this repository's
performance prose: **arbitrary-data results** (dense kernels, exponents,
fraction of peak --- caller-supplied seed, operands generated at runtime, no
structural assumption) and **structured-data results** (tabulation, collapse,
the symbol path --- these win because the data has a small effective alphabet,
and every figure comes with its corpus).

The following integer table is the pre-Atlas oracle sweep on a two-core AVX2
runner. Its former `f32` row is intentionally omitted: the bridge/scalar
implementation it measured has been removed, so that number is historical
evidence rather than a description of this implementation.

| | uor-matmul | oracle | |
| --- | --- | --- | --- |
| `i8`, `n = 1024` | 37.7 Gmac/s | matrixmultiply `f32` 43.3 | 1.1x behind, across element types |
| `i32`, `n = 1024` | 29.1 Gmac/s | ndarray 0.21 | **139x ahead** |
| `i32`, `n = 1024` | 29.1 Gmac/s | nalgebra 4.58 | **6.3x ahead** |
| `i8`, `1024x1024x1` | 37.5 Gmac/s | ndarray 3.24 | **12x ahead** |
| `i8`, `1x1048576x1` | 40.1 Gmac/s | ndarray 2.41 | **17x ahead** |
| latency at `n = 1` | 22.5 ns (`i8`) | ndarray 60 ns (measured on an M4 Max; the x86 runner's figure was 140 ns before the resolved-kernel cache) | |

The integer paths in that sweep are ahead of both integer oracles at every size that is not
latency-bound, and hold their throughput from `n = 128` upward while `ndarray`
falls away --- with the caveat, stated wherever these figures appear, that the
integer oracles have no integer BLAS and their paths are generic kernels. These
rows therefore claim only the recorded comparisons, not parity with an unnamed
production baseline, and every integer figure remains `open`.

The historical pre-one-frame pure-UOR body, in the 2026-08-06 shared-host
census, measured
the offered `f32` Atlas route at `0.0160 +/- 0.0099` Gproduct/s on the
few-grades `32^3` case, against `0.0003 +/- 0.0001` for the deliberately
unoptimized exact reference and `8.609 +/- 1.775` for `matrixmultiply`.
The analogous offered `f64` rate was `0.0062 +/- 0.0008`, against `0.0001`
for the exact reference and `6.242 +/- 1.032` for `matrixmultiply`. These are
`open` observations, not acceptance thresholds. That run identified full-range
gauge projection as slower than the exact reference at its V&V shape and drove
the direct one-pass refactor. The log preserves the finding as a baseline
without turning it into a fallback or weakening the purity and byte-identity
gates; it does not claim those rates for the frozen implementation.

`ANALYSIS.md` §"The constraint that is nobody's" has the structured-data
figures: the collapse traversal charges per *distinct* row of `A` rather than
per row, so at `4096 x 512 x 512` with one distinct row it runs at **715
Gmac/s** --- 17.8x this library's own packed traversal and 1350x `ndarray` ---
on an answer asserted byte for byte against both, and an operand with no
repeated rows pays 4% for the question and is told so in the same table. The
second such figure is tabulation: a partial sum of one row of `A` against every
codeword is computed once per block of the reduction and then *read*; the
column loop below it is one table read and one add per code, covering
`MAX_BLOCK` weights, with no multiply in it at all. Over the E8 codebook that
is **16x fewer multiplies and 5.3x fewer operations** than the dense traversal
at `n = 4096`, counted by an operation census rather than timed, and the census
and the clock cross the derived break-even at the same `n`. Against this
library's own packed AVX2 path over the *already decoded* weights it is **8.8x
ahead at `m = 1`**, **3.6x at `m = 8`**, and **1.1x to 1.7x** through `m = 64`
and `k = 4096`. And the third: one level of the sub-cubic recursion, which over
the integers is add, subtract, and multiply only, so it returns the same
integer the naive loop returns, bit for bit --- `CD-21` pins the bytes, and at
`4096` cubed on the measured host it runs **+56%** over the lane it factorizes
with a fitted time exponent of `2.87 +/- 0.01`, the interval excluding 3.0 on
three host classes.

The weights' residency halves too: a 256-entry codebook's index stream is one
byte per code, not two (`CK-15`) --- E8 at **0.125 bytes per decoded weight**,
`Sign<8>` and `Ternary<4>` halved likewise, with the gather dispatched once per
call between monomorphic `u8` and `u16` sequences. A longer codeword is a
denser instruction and a smaller stream: `Book<256,16>` stores **0.0625**
bytes per weight and reads 1.5--1.8x faster than `Book<256,8>` at the shapes
where the table pays, at an identical slab and plan.

Where the table does not pay --- a shape below its break-even --- a caller who
offers room for the decoded operand gets the tile kernels instead, at **parity
with a dense operand handed over free**. Factorizations of one identity, chosen
by the offer and the shape, byte-identical under `CD-13`. And the operand's
*columns* collapse too: two columns whose code streams agree accumulate
identically, so an operand with repeated columns is charged for the ones it has
--- **2.06x** at high degeneracy, 4% for the question when there are none.
`ANALYSIS.md` §"The other constraint that is nobody's" has the nine-row table
of what each removal was worth and the two tuning constants that changed sign
when the loop around them changed.

## Licence

Apache-2.0.
