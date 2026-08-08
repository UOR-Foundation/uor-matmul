# MEASUREMENT LOG

**The finite i8 lookup route (first host measurement; open).** The complete
signed i8 product alphabet is now a static 256 KiB lookup table, and selected
i8 dense, reduce, and table entries use lookup plus accumulation on every ISA
family. The table is generated at compile time by an independent shift-add
constructor; shipped calls do not allocate. On the development host (Apple M4
Max, aarch64), the existing `gemm/i8_i32_128cubed` Criterion case measured
**992.81 us--1.0875 ms**, midpoint **1.0350 ms** over ten samples. Criterion
compared this with the prior route at **+5.58%--+12.44%** (midpoint **+8.94%**),
so this correctness-complete lookup route was a small regression. Byte identity
and the zero-multiply census passed. The follow-up native vector-add/gather
route retains the same table identity: on the same host and shape it measured
**880.00 us--889.39 us**, midpoint **883.99 us**, or **-15.14% to -8.15%**
(midpoint **-11.85%**) against the scalar lookup route. ARM's table reads remain
scalar because NEON has no general i32 gather; its accumulation is now native
four-lane add. The AVX2 and AVX-512 table builders now use native i32 gather
plus add as well. The pinned Rust toolchain cross-checks both x86-64 and Wasm
kernel builds; no non-CPU path was added in this step. The reduce factorization
now has the same native vectorized lookup accumulation on all three ISA
families. On the development host, `workspace/none[0B]/1x1048576x1` measured **383.69 us--386.95 us**,
midpoint **385.33 us**, or **-8.74% to -7.03%** (midpoint **-7.89%**) against
the prior reduce route. Byte identity was asserted in the timed path. The
isolated finite-i8 table-build benchmark now covers the full-alphabet
builder at `space=4096`, `block=16`, `rows=16`: portable measured
**383.21 us--398.29 us**, midpoint **391.98 us**, while the native NEON build
measured **379.36 us--385.02 us**, midpoint **381.70 us** (about **2.62%**
faster at the midpoint, ten samples). A CPU byte-sliced i16 table-build
experiment was rejected: portable measured **105.04 us** versus **1.758 ms**
for native NEON, about **16.7x slower**; that path was reverted and does not
ship. The AVX2 and AVX-512 builders are complete registered sequences whose
bytes and operation declarations are covered by the backend differential and
CM-04 gates. Host timing is a separate optional `open` claim, never a
completion condition for either sequence or its declaration-derived boundary.

## Closed representation-width investigation

This log is held to R15 exactly like every other workspace file. The original
representation-width investigation is closed: each implementation capability
either ships with its IDs and gates or was measured and declined. Rows that
compare instruction declarations are `build` evidence and do not become
incomplete merely because a particular ISA was absent from the measurement
host.

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

### Disposition

The original items have final dispositions:

| Item | Disposition |
| --- | --- |
| VNNI declaration boundary | **Closed as `build`.** `model/tiers.toml` records the AVX-512/VNNI sequence pair's declared build, gather, and dense densities; CM-04 recomputes each result without consulting a clock. The runner's lack of AVX-512 only means its open artifact exercised the AVX2 pair. It creates no missing performance capability. |
| The co-issue thesis | **Answered and refuted.** 2.6--11.2x slower at every recorded configuration; no kernel ships. The retained harness is a reproducible falsification instrument, not an incomplete capability. |
| Mantissa slicing | **Answered and declined.** Every measured host sits at a narrow-to-wide ratio of four to five, below the proposed eightfold trigger. More importantly, the direct balanced-octet Atlas projection exposes the narrow alphabet without nine materialized operand passes or nine traditional integer GEMMs. ANALYSIS.md retains the rejected derivation so it is not rediscovered. |
| The multi-host remainder | **Closed.** The `CG-17` x86 figure landed: 0.30x at `k = 64` and 0.20x at `k = 1024` and `16384`, the same decline on a second host. |

1. **The multi-host result.** The CG-17 x86 figure: the
   SWAR broadcast at 0.30x of the dot sequence at `k = 64` and 0.20x at
   `k = 1024` and `16384` on the x86 runner --- worse than the M4 Max's
   0.32--0.38x, the same decline on a second host. The VNNI break-even
   artifact resolved the AVX2 pair because the runner declares no AVX-512; it
   therefore observed that pair rather than the separately complete VNNI
   declaration rows. The co-issue experiment is answered: with the
   packing defect removed, the co-issued form is 2.6--11.2x slower than the
   vector-only tile at every configuration --- the thesis is refuted, no
   kernel ships, and the retained instrument reproduces that disposition.
2. **Mantissa slicing, declined.** The analysis (ANALYSIS.md
   §"Mantissa slicing, and the RNS beside it") records nine 8-bit passes,
   shift recombination, and the proposed narrow-to-wide trigger so the cost
   argument is not rediscovered. This host's measured ratio --- i8 dotprod at
   63.1 Gmac/s against the `i32`-exact lane's 13--16 --- is about four to five,
   below the proposed eightfold trigger. The direct balanced-octet projection
   also makes the legacy decomposition unnecessary: it reaches the same narrow
   alphabet without materializing nine operands or invoking nine integer GEMMs.

## Measurements

**Float-history notice.** Every float result below that names the placement
bridge, scaled/scalar lanes, whole-operand reification, or a traditional
integer product records the implementation that existed when the measurement
was taken. Those numbers are preserved as historical evidence; they do not
describe the current call graph. The `CG-21` section at the end distinguishes
its logged pre-one-frame baseline from the fresh sweep required for the frozen
pure-Atlas refactor; neither status retroactively rewrites an older result.

**The selection lane against the ring lane (`CG-19`, measured; every figure
`open`).** Measured on the CI-class development host (x86_64-unknown-linux-gnu,
AVX2, no AVX-512), 2026-08-06, `just tropical-sweep` in release, seed 20260805,
best of a 0.35 s budget per point, with byte-identity against the reference
traversal asserted inside every timed run.

At `64^3`: the `(max, +)` reference lane reads **0.683 Gmac/s**, the ring
reference lane 0.995, and the ring's *packed* lane 11.144. The first two are the
comparison the row is about --- the same traversal at two instantiations --- and
they sit within a factor of one and a half, which is what a traversal that
branches on nothing should look like. The third is not a tropical figure and is
recorded only so the first two are not read as this library's throughput: the
packed ring lane is an order above both, because it is kernelized and the
reference is not. The tropical sequences that would close that gap are pinned by
`CB-13` and are not what this row times.

The two witness mechanisms, at `16x4096x16`: on a tie-dense fill, lexicographic
**0.316 Gmac/s** against compare-pass **0.854**; on a fill whose maximum falls
last, lexicographic **0.388** against **0.316**. The order reverses, which is why
both fills are timed and why neither mechanism is a default in the sense of being
faster --- `Witness::Lexicographic` is the default because its invariance is a
property of the order rather than of the loop (`CD-24`), and `CD-25` asserts the
two write the same bytes whatever the clock says.

Nothing here is asserted as a property of the library. A reader who reruns
`just tropical-sweep` on another host should expect different numbers and the
same two orderings, and a disagreement about the *orderings* at this host, seed
and shape is what would refute the row.

**The public-path and workspace benches (CD-22 and the Phase B plans,
measured; figures `open`).** On the development host (Apple M4 Max, quiet
window), 2026-07-31, `cargo bench -p uor-matmul-validate`, criterion means
over 100 samples. The public safe integer API is the kernelized path now,
measured rather than asserted: `slice::gemm` against `gemm_packed` at 16/64/128
cubed reads 796/800 ns, 13.29/13.35 us, 60.32/60.22 us --- the same path
within noise at every size, which is `CD-22`'s claim with a number on it.
The workspace groups, against the work order's shapes: at `16x400000x16`
the bounded offer matches the suggested one (16.98 against 16.95 ms) at
12 KiB against 12.8 MiB of caller memory, a thousandth of the footprint at
no measurable cost; at `8x262144x8` the bounded offer costs 1.7x against
the suggested (4.51 against 2.66 ms --- the 1024-deep chunk against a
full-depth panel) while still beating no offer (6.17 ms) at 800x less
memory; at `1x1048576x1` the packed path does not pay at all (suggested and
bounded both 1.80 ms against the streaming reference's 384 us), which is
the cost model declining a tile at `m = 1` exactly as it should. Against
the oracles: `gemm_i32` wins at every measured shape (128 cubed 107 us
against nalgebra's 133 and ndarray's 771; `64x512x1024` 1.42 ms against
3.64 and 29.8); `gemm_f32` keeps the honest gap --- `64x512x1024` at 3.19 ms
against matrixmultiply's 696 us (4.6x) and faer's 1.20 ms (2.7x), the
expected state, not a regression.

**The multi-host pass, first read (CG-12, CG-14, CG-15, CG-16 on the CI
runners; figures `open`).** The `scaling` workflow's first measurement run
(`#30657176730`, 2026-07-31, artifacts `measurements-x86` and
`measurements-aarch64`), read against the single-host figures above. The
sub-cubic recursion, three host classes now: the x86 runner crosses at
`n = 1024` (`L=1` 15.522 against the cubic walk's 15.157), inside the work
order's priced 1024--2048, where the M4 Max crossed at 768; the fitted
exponent against `n` is `2.9042 +/- 0.0211` on x86 and `2.8767 +/- 0.0099`
on the CI aarch64 runner, both intervals excluding `3.0`, so `CG-12`'s
honesty condition now holds on three host classes. The recursion's margin
over its own lane at `L=3`, `4096`: +30% on x86, +40% on the CI aarch64,
+56% on the M4 Max --- and it crosses the *modular* lane only on the M4 Max
(29.4 against 22.0), which is a property of that machine's NEON lane, not
of the recursion. The bridge: x86 explicit `1024` cubed 9.696 Gmac/s
against matrixmultiply's 43.2 (a 4.5x gap), CI aarch64 5.911 against 25.1
(4.2x), M4 Max 13.15 against 39.6 (3.0x) --- the work order's 4--7x factor
over the scalar lanes remains an M4 Max measurement (3.0--3.5x pre-NEON),
because the sweep's columns no longer print the scalar baseline, and that
is noted rather than re-derived. The symbol path against the bus on x86
(`CG-14`): STREAM 31.79 GB/s, the symbol walk at 0--1% of it --- the
decode-bound finding, not the bandwidth one, on a second host. The symbol
table's boundary on x86 (`CG-16`): the same shape as the M4 Max's --- a
2.45x win at `1x1024x1024`, 1.7x at the `64x4096` shapes, catastrophic
losses at the same one-column shapes --- so the historical op-kind-fit verdict
recorded below is not one machine's weather. Two harness defects the first run exposed,
fixed in this branch: the swar step's toolchain had no wasip1 std (the
pinned channel's target list does not name it and the action's `targets:`
installs for stable), and the step's pipe swallowed the failure, so the
artifact was empty and the job green; the workflow now adds the target
explicitly and every measurement step runs under `pipefail`, which is the
honesty rule applied to a workflow --- a measurement that cannot fail is
not a measurement. At the time of this first read, `CG-17`'s x86 figure had not
landed; its later result is recorded in the closed disposition above.

**The quiet-host batch: the M1 gemv, the Strassen A/B, and the historical
op-kind feasibility fit (CG-12, CG-16; figures `open`).** Three then-open
measurements from the entries below, taken in one quiet window on the same host
(Apple M4 Max, dev machine, aarch64-apple-darwin), 2026-07-30 late, byte-identity
asserted inside every timed run. The M1 variant's gemv effect, against the
`CG-16` table above: none. At `1x1024x1024` the bridge reads 0.517 against
the pre-M1 0.533 --- a wash, because those shapes are bound by decode and
placement, not by the MAC sequence (`CG-14`'s finding); the one shape with
kernel work to save, `8x262144x8`, moved 0.835 to 0.946 (+13%). The
one-row panel is the right shape for the family regardless --- it removes a
four-row panel's padding arithmetic wherever the lane *is* the bottleneck
--- and the differential walk covers it; it is simply not a gemv win on this
host, which is what the measurement was for. The Strassen same-state A/B,
attempted once and discarded under external load, reran clean (the i8 peak
read 59.204 and 59.348 Gmac/s in the two runs, 0.2% apart): the exact cubic
lane gains 1.21--1.29x from the NEON sequence at every size (17.57 to 22.59
at `1024`, 14.98 to 18.80 at `4096`), and the recursion's margin over its
own lane widens to +56% at `L=3`, `4096` --- where it now crosses the
modular lane's rate (29.4 against 22.0), not merely matches it. The fitted
exponent is `2.8721 +/- 0.0102` against `n`: the interval still excludes
`3.0`, and the MAC-count fit `0.9574 +/- 0.0034` still excludes `1.0`.

**The op-kind cost model, measured, rejected, and removed (CG-16).** The
now-removed instrument fit per-op-kind nanosecond constants over the historical
`CG-16` grid --- 43 points, both sides timed
with their censuses --- and the model does not fit: the decode coefficient
comes out negative (-2.14 ns, least-squares noise across collinear counts,
not physics), the dense side's mean relative residual is 126%, and the
fitted model predicts a non-positive time at 8 of the 14 table-versus-dense
points, which are exactly the build-dominated shapes the boundary would have
to get right. A margin swept over the survivors is a boundary read from a
subset that excludes the catastrophic losses, and that is a fiction, not a
conservative rule --- the recorded separation is against the bridge only,
where all 8 points price (any margin in `[0.32, 1.39)` separates), and it
changes nothing: the bridge is not the path the table would displace. At that
time automatic selection declined the symbol table and
`Traversal::Tabulated` remained the entry for a caller who knew its shape. The
instrument stayed in the tree for a second host,
whose op costs could compose differently; that second-host result is recorded
above. The instrument was removed with the placement bridge because its dense
count vector named reification and scalar-lane routes that no longer exist;
retaining it under the new Atlas body would fabricate counts for dead
operations. `CG-16` retains the current symbol-table timing and live census,
while `CG-21` measures dense Atlas symmetrically. Two harness defects
the first run exposed are fixed and worth recording: the fitted constants
were seconds per op where the printer and the predictor read nanoseconds
(every constant printed 0.0000 and every residual 1.0 --- a vacuous fit
presenting as a verdict), and the boundary's drop rule was silent, so the
first verdict covered 6 of 14 points without saying so.

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
3.8x the historical item named, not through it. The measured residual is structural:
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
A/B is not a result. Both follow-ups this paragraph named --- the `M1`
variant and the quiet-machine A/B --- landed and are the entry above this
one; the `M1` is no gemv win on this host, and the clean A/B is the 1.2x
this lane's bridge measurement already priced.

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
and the lane is excluded by that declaration-derived result. Measured on an Apple M4 Max (dev machine,
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
offered the output block of accumulators unconditionally. At that point the
recorded follow-ups were an x86 run of the same sweep, a depth-aware level rule,
and the `i16` lane. Later sections record the multi-host result; the bridge that
motivated the other two questions has since been removed, and the `i16`
analysis remains a measured decline rather than an open capability.

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
both halves, the bytes and the absence. The `CG-11` census reads the AVX2 `i8`
tile bound on `Zn4FP2` (85 instructions, 14.5 cycles a tile-step, IPC 5.88 ---
scheduling-model predictions, reported as such), which leaves the scalar ports
idle in the model. The later x86 co-issue experiment recorded below tested that
hypothesis and rejected the sequence across its complete grid; no co-issue
measurement obligation remains. The Cortex-M half is analysis only, in
ANALYSIS.md §"The SWAR
broadcast, measured and declined": thumbv7em has `SMLAD`, so the
multiplexing is already in silicon there, and thumbv6m has no executor in
this repository --- qemu user-mode does not cover Cortex-M, and `CB` parity
is a run, not a compile --- so no Cortex-M sequence is registered.

**The symbol table in the Atlas lanes (CD-20 build evidence; historical CG-16
figures `open`).** The live `f32` lane is `Scaled64`, an eight-byte exact Atlas
carrier. Its contextual projection and builder contract balanced signed octets
through lookup and addition. The one-product contraction follows the occupied
octet extent: its dedicated q observer records that variable work, while the
generic table Census names one opaque contraction presentation rather than
assigning it a fabricated fixed add count. For a pointwise block-one code stream
shorter than the declared enumeration, the scale walk visits `A` and the
addressed symbol visits only. The decoded book and each slab likewise fill only
addressed canonical entries for a column block; unused symbols cannot widen the
span or charge a build. Duplicate visits are permitted deliberately, avoiding a
bitmap or a parallel carrier allocation.

The caller's panel offer holds the decoded book and one sixteen-row activation
tile. Any remaining complete `m * k` row spans are reinterpreted in place as a
zero-copy activation-projection cache: cached rows project once per call, while
uncached rows project once per column block. An exact-sized offer has no cache
tail and still executes the same table arithmetic. The census counts the span
observation, codec decode, contextual projection, opaque demand-build
presentation, gather, and table add at those actual presentation counts
(`CD-20`); `CD-32` owns the occupied-extent observation.

`f64` is not categorically declined. Its public table lane is the complete
`Wide<Complete>` carrier, and a forced block-one `Arena<8>` call executes a real
eight-symbol table with nonzero table reads and byte identity against dense
Atlas. An independently declared block-two enumerable `f64` codec also admits
the same lane from its own declarations. Precision changes carrier occupancy
and residency, not whether a table object exists.

The following 2026-07-30 Apple M4 Max table is a historical measurement of the
former full-codebook `f32` build and removed placement bridge. It remains an
`open` record and is not a performance claim for the demand-built Atlas table.
That harness asserted byte identity inside every timed run and measured STREAM
at 135.10 GB/s:

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

The historical reading: the lever the `CG-14` reading priced was real and narrower than
priced. The table wins 1.8--2.2x over the dense driver at `1x1024x1024` ---
a gemv the bridge refuses to walk --- and 1.2--1.6x at the tabulation-sweep
shapes, where it is 2.8x the bridge, whose per-call reification of the
`k x n` operand amortizes over `m = 64` rows rather than a cube's thousand.
It loses wherever that build's `code_space` products per reduction element do
not amortize over `n`, and the losses are catastrophic, not marginal: 175x
at `1024x1024x1`, 44x at `8x262144x8`. At that revision `tabulation_pays` was
unchanged and selection declined the one-element block: the win was an op-*kind*
difference (a read and an add against a decode and a placement), which the
instruction-count predicate cannot see, and the boundary would have to be
built from measured per-op-kind constants --- a model change with a `CM-04`
trail --- to select a 1.2--2.2x win at the risk of a 40--175x loss. The
historical full argument is preserved in ANALYSIS.md §"The symbol table in
the scaled lane". The live selector boundary is measured by the current CG-16
instrument and is recorded only with that instrument's demand-build census.

**Float placement bridge (CD-19, CG-15, historical and removed; CG-15 figures
`open`).** The
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
not smoothed over: spans past seven binades declined that bridge (the wide fill's
row), and every `f64` bridge call declined it (a 53-bit significand is not an `i32` at any span, and the
`i64`-element family's `i128` lane has no SIMD multiply on any supported
target), and an asymmetric span declares the wider panel's bound, which can
collapse the lane depth to one product --- exact, and no faster than the scalar
lane. This was a property of the removed bridge, not of the live complete `f64`
table described above. What the bridge unblocked was the narrow float-symbol tabulation lane
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

**Deep default bridge re-read (CG-15, open).** The panel-only float entry had
enough room for the bridge's reified operands and kernel panels, but no exact
accumulator offer. At a depth past the selected lane's capacity it therefore
declined to the scalar scaled loop, even though the same kernel traversal was
faster when [`gemm_float_bridged`] supplied accumulator room. The default now
keeps one tile-sized `i128` accumulator block on the stack and lets the kernel
chunk the reduction; the byte-identical test now pins the bridge at both shallow
and deep admitted depths. This adds no model constant and keeps the no-allocation
API unchanged.

On the same aarch64-macos host, `just bridge-sweep` (best of a 0.35 s budget per
point, 2026-08-05) read the retained version as follows:

| fill | default at `512` cubed | explicit | default at `1024` cubed | explicit |
| --- | ---: | ---: | ---: | ---: |
| one exponent | 14.208 | 13.905 | 15.271 | 17.285 |
| a few binades (3/4) | 4.552 | 12.369 | 4.888 | 15.202 |
| wide spans (18/22) | 3.602 | 3.694 | 3.658 | 3.672 |

The accumulator closes part of the default deep-span gap, but the large
explicit offer still has more output-block room and can admit richer factoring.
The wide-span row is unchanged: its exponent range does not fit the `i32`
bridge, so both entries correctly use the scalar scaled lane.

**Wide-span compact-band probe (CG-15, open; rejected).** A caller-owned
experiment compacted each finite `f32` code to a band-local `i32` value and
bucketed each output by `(A-band, B-band)`, placing each non-empty bucket once.
It was byte-identical to the scalar lane, but on the same aarch64-macos host
it reached only `0.415`, `0.431`, `0.434`, `0.342`, and `0.401` Gmac/s at
`32`, `256`, `512`, `1024` cubed, and `509x1021x257`, respectively, against
`1.649`, `3.414`, `3.711`, `3.837`, and `3.819` for the scalar lane. The
per-output bucket bookkeeping and wide integer accumulation cost more than the
placements removed, so the experiment is not part of the shipped API. The
result closes the scalar-band proposal as a decline. The direct Atlas
contraction recorded later supplies the representation-level grouping without
shipping this bucket mechanism.

**Portable tropical baseline (CG-19, open).** The first completed slice of
the plan's D-8 selection half is the exact portable `(max, +)` reference:
finite inputs use a one-step wider signed sum, `-inf` is the semiring zero,
and reduction combines by max. The release sweep on the development host
(Apple M4 Max, aarch64-apple-darwin; 2026-08-05) measured:

| shape | portable reference (Gop/s) |
| --- | ---: |
| 32x32x32 | 1.504 |
| 128x128x128 | 1.557 |
| 256x256x256 | 1.478 |

The timed result is byte-identical to an independently written scalar oracle.
This is a baseline, not a performance claim.

**Native tropical lane (CG-19, open; shipped).** The plan's first native
factorization is a separate tropical kernel family: an 8x16 stack block packs
each operand once, then invokes a 4x8 NEON tile whose interleaved loads decode
eight `Trop<i8>` records at once and reuse that B vector across four A rows.
The tile applies `smax`/`add` and processes depth in stack chunks. It is
byte-identical to the portable result, including `-inf`, tails, and a depth
that crosses a chunk boundary. On the same host and release harness it
measured:

| shape | portable reference (Gop/s) | native NEON (Gop/s) |
| --- | ---: | ---: |
| 32x32x32 | 1.542 | 6.292 |
| 128x128x128 | 1.595 | 6.727 |
| 256x256x256 | 1.498 | 6.235 |

The native blocked path is about 4.0--4.2x here. The representation-aware
load, four-row reuse, and 8x16 stack block remove the prior scalar conversion,
repeated-B decode, and panel-copy costs while keeping one semantic data path.
This record establishes the shipped NEON block and makes no claim for an
alternate cache geometry. AVX2 and wasm are separate registered backend
factorizations under the same tropical contract.

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
and it established that products per step, rather than a narrower stream, was
the relevant axis for the later Atlas refactor. The oracle's 13-16% of the bus is the inexact path's
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

**The Gray-walk sign build (shipped; measured verdict: ship).** The
bound-1 table build was per-codeword independent sums: `code_space * block`
adds per row of the table (2048 at `Sign<8>`), zero multiplies, charged to
the census as `adds`. The Gray construction walks the code space in reflected
Gray-code order --- consecutive codes differ in exactly one bit `q`, so
`T[next] = T[cur] +- 2 * A[q]` --- and stores at the binary code index, not
the walk ordinal: `2 * block + space - 1` adds per row (271 at `Sign<8>`),
still zero multiplies, the 256 stores both builds share as their floor. The
spec is the bound-1 spec the host would otherwise select with only the build
swapped --- the incumbent keeps its own gathers, so the comparison prices the
build and not a portable gather --- and the walk has no ISA variant because a
serial dependency chain does not vectorize. It shipped wired behind the
codec's own declaration: `Enumerable::SIGN_BIT_BOOK` (`Sign` alone answers
true --- `Ternary` declares the same bound and its book is not the bit
decomposition), read at bound 1 under `Auto`; a named backend gets the
per-codeword build it names. The census's bound-1 charge is now the spec's
own `build_adds`: every product for the per-codeword build, the walk's count
for the Gray build. Parity is the differential test in
`crates/uor-matmul-kernels/tests/parity.rs` (every code, the alphabet's
extremes, every tile height, slot for slot against the per-codeword build and
spot-checked against the model) plus the existing bound-1 suite, which now
runs the Gray build end to end.

*Verdict rule, pre-registered:* the Gray build stays only if the isolated
build wins *and* the end-to-end run does not regress; an isolated win with an
end-to-end regression is a store-bound decline, the table stays per-codeword,
and the outcome is recorded here, exactly like the SWAR item. The harness is
the `gray_sign` group in `crates/uor-matmul-validate/benches/scaling.rs`:
isolated builds at `Sign<8>` and `Sign<16>`, and the tabulated gemm at
`8x1024x2100` and `8x4096x2100` (small `m`, wide `n`, the build a visible
fraction), `Auto` against the named per-codeword builds.

*Outcome, measured 2026-07-31 on the development host (Apple M4 Max, quiet
window; every figure `open`):* **ship**. The isolated build wins at both
widths --- 1.87 us to 0.35 us at `Sign<8>` (5.3x) and 2.45 ms to 88.7 us at
`Sign<16>` (27.6x) --- and the end-to-end run does not regress: a wash at
`8x1024x2100` (214.7 us against the incumbent ISA build's 214.6) and a small
win at `8x4096x2100` (656.7 against 673.8, +2.6%), byte-identity asserted
inside every timed closure. The build is a small term at these shapes, so
the e2e figure is close to parity by construction; the rule asked for no
regression, and there is none.

**The block-16 pricing sweep (measured; parametric implementation retained).** A
`Book<256, 16>` tier is expressible today (Phase C's width-parameterized
`Book`), and the density argument in `model/tiers.toml` says a longer
codeword improves everything except the codebook: stored codes halve per
decoded weight (0.125 B/w at `u16`, 0.0625 at `u8`), table reads per decoded
weight halve, build products stay `code_space * k` per column block, and the
break-even moves down. One part of that argument needed checking before any
of it: whether the slab scales with block. It does not --- a slab is
`slab_codes * rows` lane words whatever the codeword length (one entry per
addressable code per row of the tile), so the residency question is the
stack's `depth` slots, and the plan the driver resolves is identical at
block 8 and block 16 on every shape in the grid. What doubles is the build
products per slab and the codebook; what halves is the slots.

The harness is `crates/uor-matmul-validate/tests/block_sweep.rs`, run with
`just block-sweep` (release, single-threaded, `open` figures): four
configurations (both block widths in both code widths) over the
tabulation-sweep shapes, printing stored bytes per decoded weight, codebook
bytes, the resolved plan with the term that binds its depth, slab and stack
bytes, build / gather / end-to-end times, the census, and the break-even
recomputed from the host's own declarations. The 16-wide book is built so
both blocks price the same product (`book16[c] = book8[c] ++ book8[(c + 128)
% 256]` with streams to match), and byte-identity between the two blocks'
outputs is asserted inside every timed run. The correctness dry run
(`BLOCK_SWEEP_CHECK=1`) passed on the development host: the predicted census
ratios held exactly --- reads halve, build products constant, codebook
decodes double, plans identical, and the `u8` census equal to the `u16` one
field for field --- and the derived break-even moved from `n = 2049` at
block 8 to `n = 1366` at block 16, the direction the density argument
predicts. The first attempted timed run was discarded because the host was
loaded; a contaminated measurement is not a result. The quiet-host outcome
that governs the decision is recorded immediately below.

*Outcome, measured 2026-07-31 on the development host (Apple M4 Max, quiet
window; every figure `open`, byte-identity asserted inside every timed run):*
the longer codeword wins everywhere the table pays. End-to-end at
`64x1024x4096`: 8.69--8.88 ms at block 16 against 13.14--15.74 at block 8
(1.5--1.8x); at `64x4096x4096`: 25.4--25.7 against 39.1--39.4 (1.5x); at
`8x262144x8`: 419--424 against 628--631 (1.5x, the build-dominated shape,
where the halved slot count is what pays). The census held the predicted
ratios exactly (reads halve, build products constant, decodes double), the
resolved plan is identical at both widths on every shape, and the recomputed
break-even is `n = 1366` against block 8's `2049`. The stored residency is
the other half of the result: 0.0625 B/weight at `u8`, 0.125 at `u16`. No
tier is added: the machinery is already parametric, a caller with a 16-wide
codebook uses it today, and whether real weight artifacts have reachable
16-wide codebooks is the per-model question this sweep does not ask.

**The one-level modular bilinear factorization (shipped; auto crossover measured).** Winograd's form at one level over the quotient `Z/2^w`: eight block sums, seven base block products against the classical decomposition's eight, the base products on the modular packed kernels. The quotient is a ring, so the sums and the combination are exact in it by definition --- the exact lane's `4^L * B` headroom bookkeeping has no modular analogue, and there is no bound to track. One level is `Theta(n^3)`; this is a bilinear factorization, not a subcubic implementation, and there is no recursion below it. It ships as `gemm_strassen_modular`, an explicit entry the caller names: admitted by a `Wrapping` encode, a modular lane for the output width, even extents, and the offer (`modular_level_needs`), declining to the direct packed modular walk at the same bytes otherwise --- `CD-23` pins the bytes at every shape and offer, and the route census counts the seven products (`Route::StrassenModular`). strassen.rs's machinery is shared, not duplicated: the sums formation and the combination are the exact recursion's own loops (`E::add`/`E::sub` are wrapping spellings, which are the ring's operations here), and only the base product is new (`gemm_packed_modular_raw`, the direct modular arm's own traversal at a raw sink).

*Pre-registered verdict rule (historical):* auto-selection would take the modular level only if the factorization beat the direct packed modular walk end to end at a measured crossover. Before the outcome below existed, the model recorded `strassen_modular_min_extent = usize::MAX` and the arm declined everywhere; the harness was the `modular_strassen` group in `crates/uor-matmul-validate/benches/scaling.rs` (squares 512--4096 on the `i8`-into-`i32` and `i32`-into-`i32` rings, the explicit entry against `gemm_packed` at `Wrapping`, packing and recombination included, byte-identity asserted inside the timed closures).

*Outcome, measured 2026-07-31 on the development host (Apple M4 Max, quiet window; every figure `open`, byte-identity asserted inside every timed closure):* the crossover exists and the arm takes the level. At 512 cubed the level loses on both rings --- 2.09 ms against the direct walk's 1.25 at `i8` (0.60x), 5.15 against 4.95 at `i32` (0.96x), the sums' cost against one level's saved product. At 1024 cubed it wins (9.18 against 10.46 at `i8`, 1.14x; 36.2 against 40.1 at `i32`, 1.11x), and the win widens with size: 1.41x at 2048 and 1.57x at 4096 on the `i8` ring, 1.20x and 1.31x on `i32`. The model's `strassen_modular_min_extent` is now the smallest measured winner, 1024, recorded in `model/constants.toml` with this measurement; the explicit entry is unchanged, and at shapes below the threshold the arm declines to the direct walk exactly as before.

**The scalar-port co-issue experiment (measured and rejected, x86).** The CG-11 census reads the AVX2 `i8` tile kernel bound on `Zn4FP2` (85 instructions, ~14.5 cycles a tile-step, IPC ~5.88, scheduling-model predictions) with the scalar integer ports idle in the model, so the port-multiplexing thesis says a scalar integer stream should co-issue beside the kernel for a single-digit-percent throughput gain. The census's own caution is the reason this is an experiment and not a kernel: if both streams bind the same port, one bottleneck has been split into two queues for it, and only a measurement answers which. The harness is `crates/uor-matmul-validate/src/coissue.rs`, a dev-only instrument in the FloatBook mould: the vector stream is the AVX2 tile kernel itself, called through the kernels crate's public `KernelSpec` interface (never copied); the scalar stream is Kronecker substitution over the integers in safe-Rust `u64` arithmetic --- three biased fields at 21-bit spacing to a word, one multiply producing three products, guard bits absorbing a chunk of 32 products a field, the `dpbusd` offset identity paid at extraction (the wasm sequence's own construction, `model/constants.toml`'s `wasm_swar_field_w8a8` row) --- interleaved with the kernel in one depth loop, not two passes, because the co-issue is the point. Column splits 48|16 and 32|32 of `n = 64`, `m` in {4, 8, 16}, `k` in {64, 1024, 16384}, at full `i8` range and at W4A8; byte-identity against the schoolbook reference is asserted inside every timed run.

*Pre-registered verdict rule (historical):* the co-issued form was evidence for or against port multiplexing, and either result would answer the thesis. A kernel would ship only for a gain above **5% sustained across the configuration grid** (every split, both bounds, all three depths; a win at one configuration and a loss at another would be a split result). The development host was aarch64 and could not answer that question; the x86 CI outcome immediately below did. The dry-run correctness half also established that the scalar stream and co-issue were byte-identical to the schoolbook reference at the alphabet's extremes, at W4A8, at all three depths and every `m`, both splits.

*Outcome, measured 2026-08-01 on the CI x86 runner (scaling run `#30719729460`; every figure `open`, byte-identity asserted inside every timed run):* **the thesis is refuted, and the wire stays out.** With the packing defect removed, the co-issued form is slower than the vector-only tile at *every* configuration --- 2.7--3.4x at the 192|64 split and 4.2--5.8x at 128|128 at `k = 64`, rising to 2.6--11.2x at `k = 16384`, at both bounds --- because the scalar stream's density is two orders below the tile's: three products per `u64` multiply against sixteen products per instruction at several instructions a cycle, so no port the scheduler leaves free is capacity worth using. The census's "scalar ports idle in the model" is true and useless here: the ports are idle because the tile is dense, not because work could profitably move. The verdict rule's answer is decline, and no kernel ships; the retained instrument makes that host-scoped `open` result reproducible. (The workflow now tees stderr as well as stdout, so the artifact carries the figures the CI log carried this time.)

**The coissue harness's first artifact measured packing, not ports (a harness defect, recorded and superseded).** The first x86 run of the co-issue sweep came back with the split winning 10--40% everywhere, at per-call times around 0.7 Gmac/s --- thirty times below the AVX2 tile's real rate. That is the number that gives the defect away: the timed region was allocating and packing panels per tile per rep, and the split "won" because it packed fewer tiles. A measurement that answers a different question than it asked is the falsifiability table's own tradition, and the artifact is recorded here as one, not cited anywhere. The harness was rebuilt on the driver's structure: `Coissue::prepare` packs every panel and owns every buffer once, and `Coissue::run` --- the timed path --- borrows all of it (per-chunk contiguous slices of the pre-packed panels, the scalar stream's stack arrays, one reused output buffer), with byte-identity asserted inside the timed rep against that buffer and a counting-allocator test asserting the timed path's allocation count is zero. The width was also raised so the vector part is kernel-dominated: `n = 256` at splits 192|64 and 128|128, with the `n = 64` configurations kept as a second group labelled small-tile regime. The verdict rule remained unchanged: ship only if the co-issued form cleared **> 5% sustained across the grid**. The first artifact is void; the corrected fixed run recorded above supersedes it.

**The breakeven artifact's 683 is the AVX2 pair's crossing --- the runner declares no AVX-512 (diagnosis, case (a)).** The instrument flipped kernels-to-table between `n = 512` and `n = 683` on the x86 CI runner, matching the AVX2 declaration pair. The code's answer is direct: the table family resolves `avx512_table_i8_i32` only when `avx512_available()` reports `avx512f` and `avx512bw`, while the dense side resolves `AVX512_DPBUSD_I8_I32` only with VNNI. This runner resolved neither, so both sides used the AVX2 declarations. The artifact therefore says nothing about an AVX-512/VNNI clock. Those model rows are already complete `build` declarations whose boundaries CM-04 recomputes from registered sequence densities; no unavailable-host observation is part of their claim. The resolved-pair print makes the host-scoped `open` artifact explicit, while the declaration comparison remains host-independent.

**Historical pre-one-frame pure-UOR float performance census (CG-21 baseline,
open; measured 2026-08-06).** The then-current `just uor-float-sweep` completed
in 25.80 s after compilation on the shared x86-64 Codespace host (AMD EPYC
7763, Linux). Each point used nine independently timed batches targeted at 4 ms
and a two-sided 95% Student interval (eight degrees of freedom). That harness
version poisoned C and compared every output code inside each timed invocation:
both Atlas routes and the incumbent exact reference matched the independently
accumulated exact bytes throughout. The external routes were compared
byte-for-byte with their own pre-run output in the timed closure and separately
reported against the exact result in ulps, so their inexact answers could
neither fail nor flatter the timing. These intervals consequently include the
guard traversals and are preserved as the baseline they actually measured.

The symmetric f32/f64 grid was `(m,k,n,fill,seed)` =
`(1,1,1,one-grade,101)`, `(32,32,32,few-grades,102)`,
`(16,128,16,inverse-gauge,103)`, `(7,31,5,full-finite-range,104)`,
`(1,65536,1,dense-significand,105)`, and
`(128,8,128,sparse-significand,106)`.  Those are V&V workload sizes, not an
implementation envelope: their purpose is to retain every structural axis
while letting the deliberately unoptimized reference compute every expected
byte.  Traffic below is the explicitly named logical lower bound A-read +
B-read + C-write; the timed poison and validation add two C traversals, printed
by the harness but not misreported as algorithm traffic, and no hardware-counter
claim is made.

Representative intervals (latency, nominal product rate, logical caller
traffic) were:

| format / case / route | latency (us) | Gproduct/s | logical GB/s |
| --- | ---: | ---: | ---: |
| f32, few-grades 32 cubed, Atlas offered | 3339.6 +/- 1454.7 | 0.0160 +/- 0.0099 | 0.0060 +/- 0.0037 |
| f32, few-grades 32 cubed, Atlas no-offer | 3947.1 +/- 2302.8 | 0.0121 +/- 0.0057 | 0.0045 +/- 0.0021 |
| f32, few-grades 32 cubed, exact reference | 133450.8 +/- 42767.3 | 0.0003 +/- 0.0001 | 0.0001 +/- 0.0000 |
| f32, few-grades 32 cubed, matrixmultiply | 4.19 +/- 1.26 | 8.609 +/- 1.775 | 3.228 +/- 0.666 |
| f32, few-grades 32 cubed, faer | 4.86 +/- 2.23 | 7.772 +/- 1.499 | 2.915 +/- 0.562 |
| f64, few-grades 32 cubed, Atlas offered | 5446.7 +/- 752.0 | 0.0062 +/- 0.0008 | 0.0046 +/- 0.0006 |
| f64, few-grades 32 cubed, Atlas no-offer | 5254.3 +/- 2207.4 | 0.0076 +/- 0.0023 | 0.0057 +/- 0.0017 |
| f64, few-grades 32 cubed, exact reference | 306430.6 +/- 41483.6 | 0.0001 +/- 0.0000 | 0.0001 +/- 0.0000 |
| f64, few-grades 32 cubed, matrixmultiply | 5.72 +/- 1.89 | 6.242 +/- 1.032 | 4.682 +/- 0.774 |
| f64, few-grades 32 cubed, faer | 6.30 +/- 1.65 | 5.518 +/- 0.814 | 4.138 +/- 0.610 |
| i8 control, 32x256x32, packed | 424.2 +/- 76.0 | 0.650 +/- 0.119 | 0.0507 +/- 0.0093 |
| tropical control, 32x256x32, portable | 1821.9 +/- 195.2 | 0.147 +/- 0.018 | 0.0229 +/- 0.0028 |

The historical performance finding is mixed and therefore useful. On coherent grades
the Atlas factorization is tens of times faster than the element-by-element
exact incumbent, but it remains orders of magnitude behind both inexact float
oracles. More importantly, **full-range f64 gauge handling was the bottleneck in
that revision**: at only `7x31x5` (1,085 products), offered and no-offer Atlas took
`71.17 +/- 21.61` ms and `65.60 +/- 15.37` ms, respectively, while the exact
reference took `26.61 +/- 9.18` ms and the two classical oracles took about
1--2 us.  Full-range f32 shows the same inversion (14.92 and 10.55 ms Atlas
against 1.76 ms exact). The census therefore rejects any claim that the
then-current full-range gauge projection was optimal; the direct contraction
removed that route. The wide intervals
record contention on this shared host rather than hiding it, and all values
remain `open` observations, never acceptance thresholds.

## Current pure-Atlas CG-16 and CG-21 measurement record

Current performance evidence belongs under this heading as an append-only
record carrying the command, host declaration, source identity, raw samples,
confidence intervals, route Census, and complete byte guards. Historical
figures above do not describe the live call graph.

### CG-16 preliminary value-blind block-one candidate --- rejected

**Provenance.** This `open` measurement completed on 2026-08-08 at
12:54:43 UTC. The host was Linux 6.8.0-1052-azure x86-64 on an AMD EPYC 7763
64-Core Processor. The release compiler was rustc 1.97.1
(`8bab26f4f68e0e26f0bb7960be334d5b520ea452`, LLVM 22.1.6). The working tree
was based on `54cca2ad51fae469e5a342902b034196c34cc3bf` on
`agent/pure-uor-float-fractal`. Because the run preceded the final commit, these
SHA-256 content identities name the measured sources exactly:

| measured input | SHA-256 |
| --- | --- |
| `crates/uor-matmul-validate/tests/symbol_tabulated_sweep.rs` | `3a5114ac97d136eaa2a1995fb00edfb16541e51ecfed550ae1bfd7bedd9f7a2b` |
| `crates/uor-matmul-gemm/src/tabulated.rs` | `d42be31c94213e63fc39f6f5b2b232a9beea15f303c1370fd7d29e8f5037bf81` |
| `model/constants.toml` | `306d0914490e07c8fe462ce38f7fa6d9588bb9b97937623f12a775f5f488fb76` |
| `model/widths.toml` | `61d95e052695a60da927ef5e3c3eeb2701a065f3f9f1ea9a7e529f2231578c75` |

The exact release invocation was retained without changing its test semantics:

```text
set -o pipefail
cargo test --release -p uor-matmul-validate --test symbol_tabulated_sweep symbol_tabulated_value_blind_boundary_cg_16 -- --exact --ignored --nocapture --test-threads=1 2>&1 | tee target/measurements/cg16-pure-atlas-selector-2026-08-08.log
```

The transcript is 938 lines and 100,848 bytes; its SHA-256 is
`9e2e852831bcf00f1fa03ac8317ca8cc9c397c893cf01c21bf07fec2ee6bebf2`.
The release test passed 1 test with 2 filtered out and no failures; the measured
body finished in 5.22 seconds after the release build.

Before that clock, the exact structural/plants gate and the nonignored live
replay-to-Census reconciliation were run separately and passed:

```text
cargo test -p uor-matmul-validate --test symbol_tabulated_sweep symbol_tabulated_selector_structure_and_plants_cg_16 -- --exact --nocapture --test-threads=1
cargo test -p uor-matmul-validate --test symbol_tabulated_sweep symbol_tabulated_replay_reconciles_live_census_cg_16 -- --exact --nocapture --test-threads=1
```

Each command passed its one selected test with 2 filtered out. The second gate
reconciled one representative block-one compact-q case and one block-three
scalar-fracture control against the live Census and complete output bytes.

**Matrix and samples.** The source-pinned corpus contains 32 calibration
identities: 28 distinct value-blind `StructuralWork` keys and four additional
value-twin rows. Their adversarial envelopes form a `28 x 14` table design of
exact rank 14 and a `28 x 6` decline design of exact rank 6; deleting any named
coordinate lowers its rank. The holdout has 12 block-one identities over 11
unseen structural keys, plus the block-three H13 and block-five H14 controls.
Every one of those 46 identities emitted nine immutable paired rounds for both
routes, for 828 raw `SAMPLE` rows labelled by split, identity, structural key,
round, and route. All untimed replay/Census, poison, and complete-byte guards
passed before any observation was reported.

The exhaustive active-face nonnegative fit selected four active table
coordinates with RSS `5.990364769142e-6` and two active decline coordinates with
RSS `1.582258120540e-5`. Its conservative non-overlap rule predicted table for
9 block-one holdouts and decline for 3. The paired ratios below are table time
divided by decline time; each uncertainty is the reported paired 95% interval
half-width:

| holdout | candidate route | table / decline | measured side |
| --- | --- | ---: | --- |
| H01 | table | `0.1792 +/- 0.0278` | table |
| H02 | table | `3.2966 +/- 0.4884` | decline |
| H03 | table | `0.0443 +/- 0.0022` | table |
| H04 | table | `0.0340 +/- 0.0020` | table |
| H05 | decline | `1.3257 +/- 0.2759` | decline |
| H06 | decline | `1.1626 +/- 0.0728` | decline |
| H07 | table | `0.2283 +/- 0.0050` | table |
| H08 | decline | `11.8425 +/- 1.8207` | decline |
| H09 | table | `0.1274 +/- 0.0134` | table |
| H10 | table | `0.1510 +/- 0.0116` | table |
| H11 | table | `0.8369 +/- 0.0618` | table |
| H12 | table | `0.3214 +/- 0.0519` | table |

The scalar-fracture controls reported `1.0197 +/- 0.3579` at H13/B3 and
`0.7547 +/- 0.0940` at H14/B5. They validate the parametric control paths; they
do not enter the block-one fit.

**Disposition.** H01 and H02 are the decisive falsifier. Their structural keys
are field-for-field equal before admission and the harness asserts that the
candidate route is equal, yet their unlike values put the paired 95% intervals
strictly on opposite sides of one. A selector satisfying `CS-10` cannot inspect
those values, so it cannot choose the winning route for both presentations of
that key. The fitted candidate is rejected. No coefficient or route was written
to the model or production selector; automatic block-one selection remains the
current value-blind decline, while forced tabulation remains byte-identical and
available to an informed caller.

#### Post-native source-frozen rerun

The authoritative current-source rerun completed later on 2026-08-08 with the
same host and compiler. It grouped the exact structural/plants gate, the live
Census reconciliation, and the ignored release clock in one transcript:

```text
target/measurements/cg16-pure-atlas-selector-post-native-2026-08-08.log
```

The artifact is 1,204 lines and 120,985 bytes with SHA-256
`b63ceb82416148b7801e501de35f70ad7ea5537fb365ca33986b2d5c4be45a34`.
Its sorted full-source manifest is
`88f1ee00d2e4e1a60b70a8568f6b4aadcaca40ae4ba2044298436d4602dbeed1`
both before and after the run. Both non-release gates passed, the release gate
passed, all poison/byte/Census guards passed, and the same 46 identities emitted
828 raw samples in 92 route groups with exactly nine rounds each.

The fresh table fit has five active coordinates and RSS
`2.962146178607e-6`; the fresh decline fit has three active coordinates and RSS
`1.282755807339e-5`. It again predicted table for 9 block-one holdouts and
decline for 3. The decisive same-key twins remained contradictory:

| holdout | candidate route | table / decline | measured side |
| --- | --- | ---: | --- |
| H01 | table | `0.1792 +/- 0.0318` | table |
| H02 | table | `3.9968 +/- 1.7006` | decline |

The B3 and B5 controls reported `0.8164 +/- 0.0610` and
`0.8194 +/- 0.0900`. This post-native rerun therefore confirms rather than
changes the disposition above: the fitted value-blind block-one candidate is
rejected, the automatic default remains decline, and forced exact tabulation
remains available.

### CG-21 UOR-NAF cleanup sweep --- completed, open

The frozen implementation uses the direct one-pass diagonal contraction, one
exact live-cell frame, and in-place Atlas projection caches described in
ARCHITECTURE.md. The cleanup additionally initializes only live product
carriers, clears only retired coordinate suffixes, and decodes each streamed
source once into its six-state boundary quotient plus finite payload. These are
storage/work refinements of the same contraction, not another numerical route.

The live `CG-21` harness poisons with expected-derived distinct values before
every calibrated batch and completely checks lengths and bytes after it, so
only real production calls are timed. The authoritative post-V&V grouped run
began at 22:26:17 UTC on the EPYC 7763 host with rustc 1.97.1. Its retained
transcript is:

```text
target/measurements/cg21-uor-naf-final-post-vv-2026-08-08.log
```

The artifact is 4,252 lines and 1,199,070 bytes with SHA-256
`1c4ed9a5bf7e837f04bf57af786247875491084d037ebe5701cc96596217cb3c`.
Its complete sorted Rust/TOML/toolchain source manifest is identical before and
after the commands:
`0960aef62004bd3a8617dfdf18766306c678c6f9fd6355c46ad0777519119879`.
The exact timer/plant test passed, the live audit reached 11 roots, 92 functions,
and 115 call edges, and both ignored release sweeps passed.

The public sweep retained 558 `CG21_SAMPLE` rows: 62 width/case/route groups,
each with rounds 0 through 8 exactly once. The forced-candidate supplement
retained 2,916 more rows: 324 f32/f64 case/offer/route groups, again with nine
rounds each. Every poison, length, and expected-byte comparison passed. Selected
current public latencies are:

| corpus | offered | no offer | exact incumbent |
| --- | ---: | ---: | ---: |
| f32 one grade, `1x1x1` | `1.350 +/- 0.329` us | `1.115 +/- 0.214` us | `3.954 +/- 0.491` us |
| f32 few grades, `32^3` | `6526.5 +/- 226.7` us | `5820.2 +/- 175.1` us | `134628.6 +/- 9351.3` us |
| f32 full finite range, `7x31x5` | `219.3 +/- 30.5` us | `183.3 +/- 26.5` us | `5327.4 +/- 185.6` us |
| f32 sparse significand, `128x8x128` | `16173.8 +/- 1248.4` us | `19638.6 +/- 3759.4` us | `41527.6 +/- 5204.0` us |
| f64 few grades, `32^3` | `12201.6 +/- 930.5` us | `11245.8 +/- 196.5` us | `717894.3 +/- 12559.4` us |
| f64 full finite range, `7x31x5` | `479.4 +/- 59.4` us | `485.4 +/- 61.8` us | `25712.3 +/- 3833.1` us |

Against the preceding retained run, the 95% intervals show no clear regression.
They show clear wins for f32 one-grade with an offer, both f32 sparse offer
states, and f64 one-grade without an offer; the remaining pure-UOR intervals
overlap. The float oracles remain much faster and generally produce different
codes because they round intermediate products while this method returns the
correctly rounded exact sum. These are host-scoped `open` figures, not
acceptance thresholds, selector constants, or an unqualified global-optimality
claim. The build-level optimum remains exactly the model-derived minimum over
the complete eligible group-one lookup/add universe named by `CG-22`.

## Column dictionary radix recurrence

**Outcome, measured 2026-08-08 on the development host (`open`).** The pure
ternary column-dictionary filter passed the pre-registered no-regression rule
against the immutable pre-refactor comparator at every measured reduction
depth. The host was Linux 6.8.0-1052-azure x86-64 on an AMD EPYC 7763 64-Core
Processor. The release compiler was rustc 1.97.1
(`8bab26f4f68e0e26f0bb7960be334d5b520ea452`, LLVM 22.1.6). The base revision
was `4896156face89599fe03406bc7f7be1780d3fbcf`. The final current-source
rerun records SHA-256 identities for `tabulated.rs`
`b97ff23b206a29db0ce1cbc7051a30c6bfe5d790f1e1a229cc1076d2e24bbc85`,
model derivation
`a75b8699d2581cb26b0fbf697e1b3a354c3f572706fa30496d1be68ee41805b8`,
model constants
`306d0914490e07c8fe462ce38f7fa6d9588bb9b97937623f12a775f5f488fb76`,
the governing audit
`92c9651196441ed13600ac8ab9b1739b4879b5148591b3f54a863dd9cb1b8f2c`,
and `Cargo.lock`
`287d7784819bfdaf257aa5435fc92889ee6504228819f2b03e45e2a73aa67bc3`.
The retained transcript is
`target/measurements/column-hash-uor-naf-final-2026-08-08.log`: 35 lines,
2,338 bytes, SHA-256
`6f8dc798d9c3041bc6931b940b7ef8a89b56da820e7ded3af30156351f1e39f8`.

The exact command was:

```text
cargo test -p uor-matmul-gemm --release --lib tabulated::tests::ternary_radix_column_collapse_does_not_regress_the_retained_legacy_clock_cu_11 -- --exact --ignored --nocapture
```

The immutable corpus has 257 columns over `Arena<f32, 251, u8>`, at depths 1,
16, 64, and 256. Column `j` uses representative `j - 1` exactly when
`j % 5 == 1`; coordinate `p` is `(representative * 29 + p * 17) % 251`.
Before timing, both arms were required to return the same distinct count and
the same complete first-occurrence map. The candidate was the live
`h <- h + h + h + index` recurrence over the model-owned 16-coordinate prefix,
initialized by the complete source length and reduced only once at the final
dictionary modulus. The comparator retained the former seeded FNV multiply,
rotate/XOR mixing, and masked linear probe verbatim inside the ignored test; it
is evidence, not a shipped alternate route.

Each depth calibrated one common power-of-two batch until the faster of the two
arms occupied at least 20 ms. It then collected 64 paired samples, alternating
both arms in 32-call chunks and reversing the first arm between samples. Each
paired sample was poisoned before either clock, and both distinct counts and
complete maps were checked after both clocks. Ratios are the
geometric mean of the 64 paired `candidate / retained` duration ratios; the 95%
interval is the paired log mean plus or minus two standard errors (63 degrees
of freedom, conservatively wider than Student `t_0.975,63`). The registered
acceptance rule was exact: every upper endpoint must be at most 1.0.

Two unchanged-source preliminary runs exposed full-arm order drift at depth
256. Rather than select a favourable rerun, the harness was made symmetric
inside every sample by alternating fixed 32-call chunks; a mutation that
removes that alternation is required to fail the audit. The final transcript
below is the first run of that frozen protocol, with the same corpus, calibrated
batch rule, complete guards, and exact upper-endpoint criterion.

| depth | common batch | paired ratio | paired 95% interval | verdict |
| ---: | ---: | ---: | ---: | --- |
| 1 | 32768 | 0.5327 | [0.5265, 0.5389] | pass |
| 16 | 8192 | 0.7988 | [0.7954, 0.8023] | pass |
| 64 | 4096 | 0.8291 | [0.8263, 0.8318] | pass |
| 256 | 2048 | 0.9095 | [0.8995, 0.9196] | pass |

The timing claim is deliberately only host-scoped `open` evidence. Correctness
does not depend on it: canonical index-stream equality remains the dictionary
authority, and the differential separately proves that removing the redundant
initial remainder preserves every final address. The model independently
derives the widest live unreduced accumulator from a full 64-bit source-length
coordinate and 16 full 64-bit indices at radix three:
`1_191_107_759_025_695_718_254_230_815`, exactly 90 bits. The exact model test,
the CU-11 mutation gate, and the live source audit passed before this clock; the
live audit reached 11 roots, 92 functions, and 115 edges, including 23 Atlas
functions and 57 Atlas edges, so the measurement is not attached to a vacuous
or stale graph.

## Native lookup factorization (`CG-23`)

The final host-scoped protocol distinguishes four changed lookup bodies from
three mechanically normalized static controls before timing. Changed cases keep
the preregistered demonstrated-superiority assertion (`upper95 <= 1.0`); static
controls retain the same 256 paired samples and report their intervals without
turning a true equality into an almost-always-failing superiority test. Raw and
resolved arms traverse the same safe `KernelSpec` or `TableSpec` wrapper.

The source/model/scenario protocol test was captured red before the
classification existed, then passed with mutation plants for a changed-to-
control relabel and wrapper asymmetry. Model regeneration/check, seven native
audit/plant tests, the live 11-root/87-function/108-edge audit, and five release
parity tests passed before the clock. The linked x86-64 artifact contained one
64-byte-aligned, local-hidden 262,144-byte production product alphabet with no
dynamic relocation; the MR1 reduction used a direct RIP-relative address and
the normalized 19-instruction recurrence/backedge.

The retained CPU-0 transcript is
`target/measurements/native-lookup-acceptance-2026-08-08.log`: 17 LF lines,
41,647 bytes, SHA-256
`0f7f1c91aaa5d23bec6360abb974c88942fd0ecc6c4a2eaec7e5aaa99854a435`.
Every case emitted 256 paired observations:

| class | case | resolved / raw | upper 95% endpoint | disposition |
| --- | --- | ---: | ---: | --- |
| changed | tile MR1/NR8/KG1 | 0.976319 | 0.980664 | pass |
| changed | tile MR6/NR8/KG1 | 0.947196 | 0.950665 | pass |
| static control | reduce MR1/NR1/KG1 | 0.999832 | 1.001504 | report |
| changed | reduce MR4/NR1/KG1 | 0.893418 | 0.897898 | pass |
| changed | table rows16/KG2 | 0.784419 | 0.797142 | pass |
| static control | tile MR1/NR16/KG1 | 1.001143 | 1.003846 | report |
| static control | tile MR6/NR16/KG1 | 0.960139 | 0.962730 | report |

All four changed factorizations passed their exact rule. The three normalized
controls remain `open` reports, as required by R4; their clocks are evidence of
the host observation, while emitted-code and byte parity discharge the build
claim.
