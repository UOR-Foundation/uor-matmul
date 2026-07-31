# MEASUREMENT LOG

The working log of the representation-width performance phase. This file is
the one place a named next step may live while `SUSP-R15-WIDTH-PHASE` stands
in `model/ledger.toml`: performance work is iterative, and a rule forbidding a
named next step makes this log unwritable. Everywhere else R15 stands as
written, and when the suspension's ledger row is removed this file is held to
the rule like every other document.

Three axes, kept separate because confusing them decides what each item may
claim:

1. **Bytes crossing the bus.** Set by representation width; lowered only by
   narrowing the symbol. The machine's STREAM number is a floor on time; no
   item here claims to beat it, only to lower it and to sit near it.
2. **Products performed.** Lowered by tabulation, row collapse, and
   Strassen.
3. **Products per issued instruction, and idle-port occupancy.** Lowered by
   SWAR and by port multiplexing. Constant-factor only; it cannot cross an
   asymptotic line.

The bandwidth floor binds only where arithmetic intensity is O(1) ---
matrix-vector and skinny GEMM. At large dense square sizes compute sits far
above the floor, so the existing gemv margins against ndarray and
matrixmultiply are a *residency* result, which the symbol-bandwidth
measurement below says in numbers.

## The queue

In landing order. An entry leaves this list by shipping with its IDs and
gates, or by being measured and recorded below as a result.

The phase's eight items are all landed. What remains are the follow-ups the
measurements named:

1. **The multi-host pass.** Wired, not yet read: the `scaling` workflow now
   runs the phase sweeps (CG-12, CG-14, CG-15, CG-16, and CG-17 under
   wasmtime) on the x86 runner and on a native aarch64 runner, publishing the
   logs as the `measurements-x86` and `measurements-aarch64` artifacts. What
   remains is reading a run: the Strassen crossover, the bridge factor, and
   the SWAR co-issue experiment (the CG-11 census has the AVX2 i8 tile bound
   on Zn4FP2 with scalar ports idle in the model) are x86 questions this host
   cannot answer locally.
2. **The op-kind cost model for symbol-table selection.** The measured
   boundary is sharp --- the table wins under 2.2x and loses 40--175x --- so
   this is measured per-op-kind constants in `model/tiers.toml` with the
   CM-04 trail, or nothing.
3. **A Cortex-M executor in CI** (qemu-system mps2/mps3 or renode) before
   any thumbv6m sequence; CB parity is a run, and no user-mode emulator
   covers Cortex-M.
4. **Mantissa slicing, conditionally.** The analysis (ANALYSIS.md
   §"Mantissa slicing, and the RNS beside it") is recorded: nine 8-bit
   passes, shift recombination, a wash where the narrow-to-wide ratio is
   two, worth it at eight or better (VNNI, NEON dotprod, tensor units). The
   trigger is a host whose measured ratio is eight or better, and the first
   composition to measure is the sliced symbol.

## Measurements

**A NEON sequence for the `i32`-exact lane (CB-02's family, shipped; figures
`open`).** The queue's NEON item, landed: `NEON_I32_I64` in
`crates/uor-matmul-kernels/src/isa/arm.rs`, `mr 4 nr 8`, the signed
`32 x 32 -> 64` widening multiply-accumulate `vmlal_s32` --- the family's
whole arithmetic in one instruction, two adjacent columns at a time, so the
accumulators store back in column order where AVX2's even/odd split
deinterleaves. One `family!` line in `available_i32_exact`; the differential
walk (`CB-02`, `CB-04`, `CB-07`) and `CG-13` cover it with no test edits, and
no driver code changed. Measured on an Apple M4 Max (dev machine,
aarch64-apple-darwin), 2026-07-30, `just bridge-sweep`, Gmac/s, byte-identity
asserted inside every timed run, as a same-state A/B --- the family line
removed and restored around two adjacent runs, with the oracle column holding
at 39.4--39.7 across both, so the runs are comparable:

| fill | `m x k x n` | portable | NEON | |
| --- | --- | --- | --- | --- |
| one exponent | `512` cubed explicit | 8.464 | 9.876 | 1.17x |
| one exponent | `1024` cubed explicit | 10.750 | 13.153 | 1.22x |
| one exponent | `509x1021x257` explicit | 8.809 | 10.240 | 1.16x |
| a few binades (3/4) | `1024` cubed explicit | 9.806 | 12.055 | 1.23x |

The reading: a real sequence closes the gap to matrixmultiply on this host
from 3.7x to 3.0x (39.6 against 13.2 at `1024` cubed) --- partway through the
3.8x the queue item named, not through it. What remains is structural:
`vmlal_s32` covers two columns an instruction where the f32 FMA path covers
four lanes across two units, so this lane's ceiling sits under half the float
path's on this silicon, and closing the rest is blocking and pack cost
(`CG-15`'s remaining distance), not a wider instruction. The recursion's
sweep re-run with the lane in place (`just strassen-sweep`, same host) reads
a fitted exponent of `2.9031 +/- 0.0432` against `n` --- the interval still
excludes `3.0`, so `CG-12`'s honesty condition is unaffected by the lane
change. A same-state Strassen A/B was attempted and discarded: an external
load generator on the shared host (load average 152, processes unrelated to
this repository) made the baseline run's figures unusable, and a contaminated
A/B is not a result. Named follow-ups: an `M1` variant for one-row panels ---
gemv shapes fall back to the portable kernel today, the gap the x86 family's
`_M1` entries already close on that ISA --- and the Strassen A/B re-run on a
quiet machine.

**The sub-cubic recursion on the i32-exact lane (CD-21, CG-12, shipped;
CG-12 figures `open`).** Winograd's form of Strassen's recursion above the
packed traversal: eight block sums and seven products per level, the sums in
the panel offer as bare elements (they outgrow the declared bound by
construction; the grown bound travels as a value to the kernel boundary), the
products in the accumulator offer in the accumulator's own width, and the
encode step exactly once, on the combination, at the top. The level count is
a pure function of declarations --- shape evenness (odd declines rather than
pads), the bound's headroom (`4^L * B <= 2^31 - 1`, because Winograd's
cross-term sums are four block terms at worst; at full `i8` range that is
zero levels and at full `i32` range it is the same zero), the offer, and the
measured crossover `strassen_min_extent` in the `[blocking]` table. `CD-21`
asserts the bytes against the `CX-01` wrapping oracle and `ndarray` at every
corpus size, every requested level count, and every offer including none; the
identity argument has no float half, because over the integers the recursion
uses only add, subtract, and multiply. The one sign in the combination is
folded into a sum temporary, so the accumulator only ever adds. The `i16`
lane's analysis --- per-operation penalty about two, break-even near five
levels, which `4^5 * B <= 2^15` admits no useful bound for --- is recorded,
not implemented. Measured on an Apple M4 Max (dev machine,
aarch64-apple-darwin), 2026-07-30, `just strassen-sweep`, nominal Gmac/s
(`m*k*n` per second) on random dense `i32` at a declared `2^24` bound, seed
`20260730`, best of a 0.35 s budget per point, byte-identity against the
cubic walk at the same encode asserted inside every timed run:

| `n` | default (modular) | exact cubic | `L=1` | `L=2` | `L=3` | levels taken |
| --- | --- | --- | --- | --- | --- | --- |
| 256 | 26.571 | 17.677 | 15.900 | 13.100 | 9.522 | 0 |
| 384 | 28.834 | 18.653 | 17.780 | 15.602 | 12.171 | 0 |
| 512 | 29.024 | 18.884 | 18.461 | 17.360 | 13.641 | 0 |
| 768 | 27.859 | 18.718 | 19.934 | 19.454 | 16.530 | 1 |
| 1024 | 28.152 | 18.393 | 20.592 | 20.347 | 18.096 | 1 |
| 1536 | 27.175 | 18.069 | 20.332 | 21.686 | 20.121 | 2 |
| 2048 | 26.186 | 17.297 | 20.026 | 21.485 | 20.378 | 2 |
| 3072 | 24.357 | 16.587 | 19.996 | 22.294 | 22.537 | 3 |
| 4096 | 22.511 | 15.534 | 19.088 | 22.225 | 23.203 | 3 |

Fitted exponents (geometric spacing, nine samples, 1.96 standard errors):
exact cubic against `n` `3.0514 +/- 0.0334`; recursion at the auto-selected
levels against `n` `2.9037 +/- 0.0092` --- the interval `[2.894, 2.913]`
excludes `3.0`, which was the queue item's honesty condition --- and against
MAC count `0.9679 +/- 0.0031`, which excludes `1.0`. The fastest sustained
product rate this library reaches on this host, measured in the same harness
(the `i8` lane at `4096` cubed), is 63.1 Gmac/s; the recursion's nominal
rate does not cross it at any measured size, and it does not cross the
modular lane's rate below `4096`, where it matches at 23.2 against 22.5.
What it beats is the lane it factorizes: the exact lane's cubic walk, from
+6% at `n = 768` to +49% at `4096`. The crossover the x86 figures predicted
at 1024--2048 sits at `n = 768` on this host, because the exact lane runs at
16--19 Gmac/s here (the portable kernel, auto-vectorized) where the
prediction priced a different lane penalty. On `f32` it rides the bridge, as
the queue entry said: `just bridge-sweep` re-measured with the recursion in
place reads 16.239 Gmac/s bridged at one exponent `1024` cubed (was 15.309),
unchanged at `512` (the threshold declines), wide spans declining as before
--- `CG-15`'s figures are re-reported, not restated; the earlier run was a
different program. Two harness defects the measurement found are recorded in
VERIFICATION.md: the first baseline column was the modular lane in exact
clothing, and the base case streamed per tile at deep levels until the plan
offered the output block of accumulators unconditionally. What remains: an
x86 runner measuring the same sweep (the work order's economics were priced
there); a depth-aware level rule (at `1024`, `L=2` is a wash over the `L=1`
the plan takes); and the `i16` lane, whose analysis says decline.

**SWAR / Kronecker substitution (CB-12, CG-17, shipped as a measured decline;
CG-17 figures `open`).** The broadcast form on baseline wasm SIMD128: three
`B`-row elements packed at 21-bit spacing in each 64-bit lane, both operands
biased to unsigned (the `dpbusd` offset identity applied on both sides), one
splatted scalar per row per step, so an `i64x2.mul` produces six products in
disjoint fields with five guard bits a field. The packed accumulator absorbs a
chunk of 32 products a field --- `floor((2^21 - 1) / (255 * 255))`, recorded
as the `wasm_swar_field_w8a8` model row --- before extraction and the
two-sided compensation land the chunk in the `i32` lanes; the driver's lane
capacity composes on top of the chunk unchanged. Measured on an Apple M4 Max
(dev machine, aarch64-apple-darwin) under wasmtime 45.0.0, 2026-07-30, `just
swar-sweep`, padded-panel Mmac/s in a hot loop --- the ratio is the finding,
the absolutes are the harness's --- with byte-identity against the portable
reference asserted inside every timed run:

| k | fill | portable (4x4) | dot (4x8) | swar (4x12) | swar / dot |
| --- | --- | --- | --- | --- | --- |
| 64 | full i8 | 13141.0 | 20289.9 | 7615.9 | 0.38x |
| 64 | W4A8 (bound 7) | 13139.3 | 19855.7 | 7566.4 | 0.38x |
| 1024 | full i8 | 18423.8 | 26859.7 | 8707.9 | 0.32x |
| 1024 | W4A8 (bound 7) | 18424.7 | 26861.5 | 8729.0 | 0.32x |
| 16384 | full i8 | 18907.5 | 27449.3 | 8776.2 | 0.32x |
| 16384 | W4A8 (bound 7) | 18907.4 | 27456.7 | 8787.1 | 0.32x |

The reading: the extending dot wins by 2.6--3.1x at every depth and both
bounds, and the instruction count says why --- the incumbent's widening is one
extend instruction per sixteen bytes and its dot is eight products, while the
SWAR form pays eleven instructions per six columns per step to build fields
the ISA has no instruction for, and the multiply that buys (six products
against eight, no extends) never recovers it. The narrower family does not
rescue it: at the W4A8 bound the fields shrink to ten bits and the chunk to
five products, so the extraction the pack was amortizing against returns every
fifth step. This is the outcome OpenBLAS's decline predicts --- the trick
loses on throughput, not numerics, here as there. So no family list carries
the sequence: a listed sequence is one `Auto` may select, and selecting this
one would be a measured regression. The spec stays exported and `CB-12` pins
both halves, the bytes and the absence. The x86 co-issue experiment is
declined on this host: the `CG-11` census reads the AVX2 `i8` tile bound on
`Zn4FP2` (85 instructions, 14.5 cycles a tile-step, IPC 5.88 ---
scheduling-model predictions, reported as such), which leaves the scalar
ports idle in the model and makes the single-digit co-issue gain plausible,
but the x86 kernels do not run on this aarch64 host and a gain that cannot be
measured cannot ship; an x86 runner measuring the co-issued pair is what would
change that. The Cortex-M half is analysis only, in ANALYSIS.md §"The SWAR
broadcast, measured and declined": thumbv7em has `SMLAD`, so the
multiplexing is already in silicon there, and thumbv6m has no executor in
this repository --- qemu user-mode does not cover Cortex-M, and `CB` parity
is a run, not a compile --- so no Cortex-M sequence is registered.

**The symbol table in the scaled lane (CD-20, CG-16, shipped; CG-16 figures
`open`).** The remainder of the symbol item above: a `Tabulated` lane for
`f32` whose lane word is an `i64` of pre-scaled significands (`Scaled64`)
rather than the 80-byte `Complete`, so the 256-entry slab is 2 KiB a row and
fits L1, which is the answer `tabulation_fits` was measured to decline for
the wide lane. The scale channel is a per-call declaration, not lane state:
a walk of `A` and of the codebook --- `m * k + code_space` decodes, the
coded stream never read, charged to the census --- asked only after the
table is selected, so a call the predicate declines never pays it. The walk
is the bridge's own span walk and admission (`24 + span <= 31`, finite codes
only); the run depth is derived per side, `2^63 / 2^(48 + wa + wb)`,
because the lane holds one element of each panel and is not bound by the
kernel table's one-alphabet interface. `CD-20` asserts byte-identity with
the dense float driver at every shape and every offer, the walk admitted
and declined alike, including a worst-case fill that drives a run to one
product short of `2^63` across a chunk boundary; `f64` declines by
declaration --- its `Tabulated` lane is the complete accumulator, because a
53-bit significand is not an `i32` at any span. Measured on an Apple M4 Max
(dev machine, aarch64-apple-darwin), 2026-07-30, `just symbol-tabulated`,
Gmac/s, byte-identity asserted inside every timed run, STREAM 135.10 GB/s in
the same harness:

| fill | `m x k x n` | sym table | bridge | uor `f32` | `matrixmultiply` |
| --- | --- | --- | --- | --- | --- |
| one exponent | `1024x1024x1` | 0.004 | 0.696 | 0.706 | 4.286 |
| one exponent | `1x1024x1024` | **0.973** | 0.533 | 0.540 | 4.455 |
| one exponent | `1x1048576x1` | 0.003 | 0.474 | 0.470 | 0.405 |
| one exponent | `2048x8x2048` | 0.409 | 0.353 | 0.525 | 16.520 |
| one exponent | `8x262144x8` | 0.032 | 0.835 | 1.414 | 21.384 |
| one exponent | `64x1024x4096` | **3.986** | 1.426 | 3.150 | 54.272 |
| one exponent | `64x4096x4096` | **3.785** | 1.356 | 2.339 | 51.341 |
| a few binades (3) | `1x1024x1024` | **1.135** | 0.527 | 0.518 | 4.458 |
| a few binades (3) | `64x1024x4096` | 2.784 | 1.405 | 3.132 | 53.638 |
| a few binades (3) | `64x4096x4096` | **2.716** | 1.358 | 2.293 | 50.646 |
| wide spans (~18, declines) | `64x4096x4096` | 0.187 | 1.879 | 2.331 | 50.737 |

The reading: the lever the `CG-14` reading priced is real and narrower than
priced. The table wins 1.8--2.2x over the dense driver at `1x1024x1024` ---
a gemv the bridge refuses to walk --- and 1.2--1.6x at the tabulation-sweep
shapes, where it is 2.8x the bridge, whose per-call reification of the
`k x n` operand amortizes over `m = 64` rows rather than a cube's thousand.
It loses wherever the build's `code_space` products per reduction element do
not amortize over `n`, and the losses are catastrophic, not marginal: 175x
at `1024x1024x1`, 44x at `8x262144x8`. So `tabulation_pays` is unchanged
and selection keeps declining the one-element block: the win is an op-*kind*
difference (a read and an add against a decode and a placement), which the
instruction-count predicate cannot see, and the boundary would have to be
built from measured per-op-kind constants --- a model change with a `CM-04`
trail --- to select a 1.2--2.2x win at the risk of a 40--175x loss. The
table is reachable under `Traversal::Tabulated` for a caller who knows its
shape, and the full argument with the per-shape census is in ANALYSIS.md
§"The symbol table in the scaled lane".

**Float placement bridge (CD-19, CG-15, shipped; CG-15 figures `open`).** The
float driver's scaled panels are exact integers, so their product is an exact
integer dot product at one known scale, `2^-(base_a + base_b)`. The bridge
reifies the panels as the `i32` alphabet they already are, hands the reduction
to the integer kernel table, and places the table's exact `i128` sum into the
complete accumulator at that scale. The scale channel is a placement epilogue
--- `Scaled`, over a `PlaceAt` accumulator trait whose one impl is
`Complete::add_scaled` --- not a parameter on the `Epilogue` trait, because
the scale is a fact of one call's panels and a wrapper carries it without
touching the contract every other epilogue implements; `EncodeFrom<i128>` for
the float formats is the scale-zero placement the traversal bound asks for.
Admission is a declaration from the panels' measured spans: a scaled
significand must fit the `i32` alphabet (`24 + span <= 31` at `f32`), and the
reduction's depth is no term, because the table's lane capacity chunks it.
`CD-19` asserts byte-identity with the streaming traversal at every shape and
every offer, including none. The prediction, written into ANALYSIS.md before
the sweep ran: 4--7x on the AVX2 runner, less on this host, where the
`i32`-exact family has no hand-written NEON sequence and the bridge runs the
portable kernel. Measured on an Apple M4 Max (dev machine,
aarch64-apple-darwin), 2026-07-30, `just bridge-sweep`, Gmac/s, with
byte-identity asserted inside every timed run:

| fill | `m x k x n` | scalar | bridged | | `matrixmultiply` |
| --- | --- | --- | --- | --- | --- |
| one exponent | `512` cubed | 4.326 | **13.082** | 3.0x | 61.277 |
| one exponent | `1024` cubed | 4.505 | **15.309** | 3.4x | 58.589 |
| one exponent | `509x1021x257` | 4.421 | **13.580** | 3.1x | 58.461 |
| a few binades (3/4) | `512` cubed | 4.255 | **11.964** | 2.8x | 60.529 |
| a few binades (3/4) | `1024` cubed | 4.039 | **14.233** | 3.5x | 58.440 |
| a few binades (3/4) | `509x1021x257` | 3.994 | **12.822** | 3.2x | 58.714 |
| wide spans (18/22, declines) | `1024` cubed | 4.016 | 4.036 | 1.0x | 58.109 |

The reading against the prediction: the auto-vectorizer found the widening
multiply-accumulate, so the factor is 3.0--3.5x at the two cubes --- below the
x86-oriented 4--7x, because the four-`i64`-lanes arithmetic is an AVX2 sentence
and this host's family entry is a portable loop the compiler vectorized --- and
the gap to `matrixmultiply` closed to 3.8x at `1024` cubed on this host, from
13x, further than the predicted 6--10x because this host's `sgemm` posts 58--61
Gmac/s where the prediction priced the x86 runner's. The boundary is reported,
not smoothed over: spans past seven binades decline (the wide fill's row), every
`f64` declines (a 53-bit significand is not an `i32` at any span, and the
`i64`-element family's `i128` lane has no SIMD multiply on any supported
target), and an asymmetric span declares the wider panel's bound, which can
collapse the lane depth to one product --- exact, and no faster than the scalar
lane. What the bridge unblocked was the narrow float-symbol tabulation lane
the `CG-14` reading priced, and that lane is the entry above this one:
`Scaled64`, the same span walk and the same declaration, the exact integer
sum placed at the panel's scale through the accumulator's own `add_scaled`.

**The bridge as the default path (CD-19's lane, CG-15 re-read; figures
`open`).** The bridge shipped behind its own entry point, and the bench timed
the other one: `just bench`'s `gemm_f32` group calls `gemm_float_packed`,
which never took the table. The default driver now auto-selects, under the
library's selection doctrine --- one identity, the factorization chosen by the
shape and the offer, the bytes never (`CD-19`). A `PackedCode` is sixteen
bytes, so a panel offer re-reads as four `i32` words a code (the layout's
padding word is named, which is what makes the re-read a safe cast); when the
offer holds the reified operands (`k * (m + n)` words) plus a full-depth
kernel panel pair, the spans admit the `i32` alphabet, and the declared lane
holds the whole depth, the packed entry hands the reduction to the kernel
table. `suggested_float_panels` names the offer that admits every
factorization the shape supports, the slice face documents it, and the raw
face documents that a raw pointer offers none. `gemm_float` keeps its old
bounds and its scalar traversal: with no panels offered the offer question is
answered before it is asked, so the `EncodeFrom<i128>` bound would be a lie
about a lane that cannot run.

Two findings the re-read of `CG-15` surfaced, both recorded because they
changed the rule. First, the sweep's `scalar` column was no longer the scalar
lanes: at the panel offer every caller has, the default took the bridge at
every admitted fill, which is the change working, and the columns are now
named `default` and `explicit` with the fill deciding which factorization
each is. Second, a regression the bench's constant fill had hidden: past the
lane's depth (`k > lane_cap / bound^2` --- 511 products at the few-binades
fill's declared `2^27`), the table chunks the reduction, the chunked
traversal's exact partial sums want an accumulator offer a `PackedCode`
panel cannot spell (eight-byte aligned against `i128`'s sixteen), and the
per-tile chunk traversal it runs without one measured 2.5 Gmac/s at `512`
cubed against the scalar scaled lanes' 4.3. The auto-selection now carries
the lane's depth as a term --- the table's own capacity arithmetic, not a
threshold --- and declines past it; the explicit entry, whose offers can
spell the accumulator room, is where a deep reduction gets the table
(10.8--13.4 Gmac/s at the three deep shapes, against the scalar lanes' 4.0).
No size floor was needed: measured on an Apple M4 Max (dev machine,
aarch64-apple-darwin), 2026-07-30, `cargo bench -p uor-matmul-validate --
gemm_f32`, the default path before against after (criterion means):

| `m x k x n` | before | after | |
| --- | --- | --- | --- |
| `16` cubed | 4.130 us | **3.890 us** | 1.06x |
| `128` cubed | 697.3 us | **338.9 us** | 2.06x |
| `64x512x1024` | 9.788 ms | **3.812 ms** | 2.57x |

The smallest shape is a wash with a slight edge --- the reification costs
about what the table saves at 4096 products, and the doctrine's requirement
was only that it not regress --- so no measured constant entered the model.
The two larger shapes keep about two-thirds of the explicit entry's factor:
the default's offer cannot spell the accumulator room the chunked traversal
and the sub-cubic level want, which is the remaining distance to the
`CG-15` explicit column (14.95 against 13.89 at one-exponent `1024` cubed)
and to the oracle (3.8x at `1024` cubed, from 13x).

**f32 as a symbol (CK-14, CD-18, shipped; CG-14 figures `open`).** The arena
tier's code width is a parameter now: `Arena<'_, E, N, u8>` stores one byte a
symbol against the dense float's four, decodes the same stream as the `u16`
spelling (`CK-14`), and drives the same exact accumulation --- the traversal
declines the one-element block, as it always does, and the stream below the
decline is the coded float path, byte-identical to `gemm_float` at every
shape and offer (`CD-18`). The committed corpus (`oracles/symbols/`,
digest-pinned, generator beside it) is the realistic case: an f32 artifact of
exactly 256 distinct bit patterns, a dequantized 8-bit grid. The acceptance
sweep (`just symbol-bandwidth`) charges each path its operand bytes --- `A`
plus the stored weights plus `C`, the 1 KiB codebook on the symbol side ---
and measures the host's STREAM number in the same harness: a triad
`a[i] = b[i] + 3*c[i]` over 3 x 2^25 `f32` (384 MiB, past every cache), best
of ten, 12 bytes counted per element (two reads and the write; the
write-allocate read is the machine's, stated not counted). On an Apple M4 Max
(dev machine, aarch64-apple-darwin), 2026-07-30, STREAM measured 134.19 GB/s
and the sweep read:

| m x k x n | W stored (B) | sym walk GB/s | sym panel GB/s | uor f32 GB/s | matrixmultiply GB/s |
| --- | --- | --- | --- | --- | --- |
| 1024x1024x1 | 1024 | 0.97 (1%) | 1.01 (1%) | 0.95 (1%) | 17.19 (13%) |
| 1x1024x1024 | 1048576 | 0.17 (0%) | 0.16 (0%) | 0.94 (1%) | 17.86 (13%) |
| 1x1048576x1 | 1048576 | 0.84 (1%) | 0.77 (1%) | 1.65 (1%) | 3.24 (2%) |
| 2048x8x2048 | 16384 | 0.08 (0%) | 0.08 (0%) | 0.28 (0%) | 8.34 (6%) |
| 8x262144x8 | 2097152 | 0.10 (0%) | 0.10 (0%) | 1.37 (1%) | 21.38 (16%) |

The reading, and it is the outcome the queue entry said to report if it came:
the symbol path sits at about 1% of the bus, not near it. At 0.08-0.25 Gmac/s
the traversal is bound by the scalar exact accumulation --- one product per
step, the same accumulation the dense float driver runs, which reads four
times the weight bytes and sits at the same 1%. The residency is real (the
census counts one decode per stored byte and the W column is a quarter of the
dense spelling's) and buys nothing here, because nothing at these shapes is
waiting on the bus: the bottleneck is per-element work, decode and placement,
not bandwidth. That is the finding this item promised to report either way,
and it prices the rest of the queue: what closes the gap is products per
step, which is the bridge (shipped above) and the narrow lane it unblocks, not
a narrower stream. The oracle's 13-16% of the bus is the inexact path's
figure, reported for scale; `CX-05` records what its bytes deviate by.

**Port and issue census (CG-11, shipped).** `cargo xtask issue-census` emits
the kernels crate's assembly for x86-64 (and aarch64 where the target is
installed), runs `llvm-mca` over every emitted inner loop --- 136 sequences on
the development host --- and reports per-port occupancy, the critical path,
predicted throughput, and a named bottleneck resource per kernel, as
`target/issue-census/*.md` artifacts and a `census` CI job that uploads them.
Every figure is an `llvm-mca` scheduling-model prediction (`znver4`,
`apple-m4`), reported and never asserted as a measurement; wasm has no
scheduling model and is absent rather than mis-analysed. First reading, as
predictions only: the portable MAC loops are store-bound (`Zn4Store`), the
wide portable reduces ALU-bound (`Zn4ALU0`), and the NEON builds issue-port
bound (`CyUnitIS`). "We used more of the processor" is now a table, not an
adjective.

**Resolved-kernel cache (CG-13, shipped; figures `open`).** Each element
family's availability list is materialized once per process into a `OnceLock`
under `std` --- the only configuration with anything runtime to resolve ---
and the driver selects from the resolved slice; the bound filter and the
panel-height choice are the caller's declarations and are still answered per
call. Latency at `n = 1` through `gemm_packed` with the wrapping encode, best
of batches of two thousand calls, on an Apple M4 Max (dev machine,
aarch64-apple-darwin): `i8` 36.0 ns before, 22.5 ns after; `i32` 51.6 ns
before, 39.5 ns after. The 140/60 ns figures in `ANALYSIS.md` are from a
two-core shared x86 runner, so these are a different machine's story --- what
transfers is the delta, not the absolute. Two honest caveats about this host:
on aarch64-apple-darwin every availability predicate is a compile-time
constant (`neon` and `dotprod` are baseline target features), so the walk the
cache replaces was already folded and what the delta measures is the resolved
slice against the closure chain; and a single call at `n = 1` lands inside
one tick of this host's clock, which is why the harness amortizes over a
batch rather than timing one call.
