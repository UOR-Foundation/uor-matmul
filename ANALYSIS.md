# ANALYSIS

Where the numbers come from. Nothing here is a preference; each figure is
derived, and `cargo xtask check-model` re-derives it.

## The accumulator widths

`MAX_K_BITS = 64` is **declared**, not probed. The alternative --- deriving it
from `usize::BITS / size_of::<E>()` on the host --- gives a *narrower* bound
(79 / 94 / 125 / 188 rather than 79 / 95 / 127 / 191) and resolves to the same
accumulator type for every element type. Declaring it is still the right choice,
because it makes the table target-independent: `CD-06` and `CA-02` then compare
one function against itself rather than two functions that happen to agree.

The complex rows carry one extra bit, for the two element-products the real part
sums. `log2(PRODUCT_TERMS)` is read from the element type, so a complex alphabet
needs no separate table and no branch.

## The complete accumulator widths

For `f32`: the minimum subnormal is `2^-149`, so the minimum product exponent is
`2 * -149 = -298`. The maximum finite value is below `2^128`, so the maximum
product exponent is `256`. The span is `256 - (-298) = 554` bits. Add
`MAX_K_BITS = 64` for the depth and one for the sign: `619`, which is 10 limbs
of 64 bits, 80 bytes.

For `f64`: `2 * -1074 = -2148` and `2 * 1024 = 2048`, a span of `4196`; plus 64
plus 1 is `4261`, which is 67 limbs, 536 bytes.

The 536 bytes per output element is the real cost of exactness at `f64`, and it
is not hidden: a large register-blocked tile is expensive on a small target. The
mitigation is a traversal choice, not a method choice.

## The narrow-register thresholds

Each is `floor(cap / per_step)`, where `per_step` is the worst-case magnitude
one step of that instruction sequence contributes.

| Sequence | per step | cap | threshold |
| --- | --- | --- | --- |
| plain `i32` tile, W8A8 | `127 * 127 = 16129` | `i32::MAX` | 133144 |
| `vpdpbusd` offset, W8A8 | `255 * 127 = 32385` | `i32::MAX` | 66311 |
| `vpdpbusd` compensation | `128 * 127 = 16256` | `i32::MAX` | 132104 |
| `_mm256_madd_epi16` pair | `2 * 127 * 127 = 32258` | `i32::MAX` | 66572 steps |
| `vmull_s8` in `i16` | `127 * 127 = 16129` | `i16::MAX` | 2 |

The last row is why the NEON kernel widens to `i32` immediately: three `i16`
products do not fit (`3 * 127^2 = 48387 > i16::MAX`), so accumulating even two
of them in `i16` would be a saturation waiting to happen.

None of these is a limit. A tile past its threshold takes a wider lane and
computes the same integer, and `CU-03` asserts agreement at depths straddling
each one.

## The blocking parameters

`mc = 128`, `kc = 256`, `nc = 1024`. Cache-shaped, explicitly allowlisted out of
R1, and asserted invariant by `CD-01`: changing one cannot change any output
byte, only which traversal produces it. They are the one table in the model that
is *tuning* rather than derivation, and saying so is the point of the allowlist.

## The property-suite sample size

`CP-01` runs 4096 randomized cases per run, at a recorded seed.

The number is derived from what the suite has to catch. A defect confined to one
position of a packed tile survives a random case with probability
`1 - 1/(MR * NR)`. The widest shipped kernel is `8 x 32 = 256` positions, so
after `n` independent cases the escape probability is `(1 - 1/256)^n`. Requiring
that below `10^-6` gives

```text
n >= 256 * ln(10^6) ~= 256 * 13.8 ~= 3538
```

and 4096 is the next power of two above it. The seed is recorded so a failure is
reproducible; a suite whose failures cannot be reproduced is a lottery.

## The corpus shapes

Zero, one, primes, powers of two plus and minus one, and rectangular extremes.
Chosen so that no block size divides anything and every tail path is reached. A
corpus of round numbers would agree with almost any implementation, which is the
failure mode this list exists to avoid --- and it earned its keep immediately, by
finding a `5 x 0` output that the aliasing check reported as self-aliasing
because a row-major view over zero columns has a row stride of zero.

## The scaling fits

Least squares in log-log space, over geometrically spaced points. Geometric
rather than linear because a fit over linearly spaced sizes is dominated by the
largest few and reports the asymptote of the tail rather than of the curve.

Fewer than three points do not fit: two points determine a line exactly and
would report a confidence interval of zero, which would be a lie rather than a
measurement. The reported interval is `1.96` standard errors; with few points
that is optimistic, which is why the sample count is reported beside it.

Every figure is `open`. `CG-01` on a two-core shared runner measured an exponent
of about `0.98 +/- 0.06` against MAC count, which is what an `O(m k n)`
implementation should give --- reported, not asserted.

## Where the time goes

Measured on a two-core shared runner with AVX2 and no AVX-512. Every figure is
`open`. What follows is what measuring found, in the order it was found, because
every one of them was a copy or a traversal rather than arithmetic --- and none
of them was visible by reading.

**The accumulator was passed by value.** `Element::mac` took and returned
`Self::Acc`. For an `i128` that is free; for `Complete<10>` it is 88 bytes in
and 88 bytes out, *twice* per product once `add_scaled` did the same. Making
both take `&mut` was worth 4x on `f32` and 1.6x on the packed integer path,
because the integer kernels were paying for it too through the shared trait.

**The complete accumulator added at full width.** `add_scaled` built a
full-width value and combined it, which is O(L) per product --- 67 limbs for
`f64`. Adding the three spread limbs in place, with a carry that stops as soon
as a limb absorbs it, makes it O(1) in the common case. `Limbs::add_i128` had
the same shape and got the same fix.

**Coded random access walked the row.** `CodedMatrix::at` found row `r` by
walking every preceding row's codes. That is O(row) per element, which makes a
driver reading one element at a time O(k^2 n) rather than O(m k n) --- not a
constant factor, a different curve. It showed up as a benchmark that did not
finish. Fixed-width tiers now index by arithmetic; only a run codec walks, and
only when a caller uses random access on one.

**The packing loop indexed instead of walking.** `origin + i * rs + j * cs` per
element is two multiplies where a walk is one add, and the panel is a million
elements. `MatView::row_walk` and `column_walk`, with the full panel split from
the edge, was worth 1.7x on `i32`.

**Panels were packed at microkernel granularity.** A panel is `mr` rows or `nr`
columns wide, and packing at that granularity repacks `A` once per column
*panel*: `m*n*k/nr` element copies against `m*n*k` of arithmetic. At the shipped
`nr` that is not a rounding error --- it was the same order as the arithmetic,
and it was the whole distance between this driver and the instruction ceiling of
its own kernels. Packing in cache-shaped *blocks* instead was worth 2.1x on `i8`
and 2.3x on `i32`.

**`k_group` was declared and ignored.** Every SIMD kernel rebuilt its `k`-group
in a stack buffer once per step, because the packed layout did not group. Making
the layout honour it --- `k`-major in groups, lane-major within one --- lets
`B`'s group widen straight into the vectors `madd`, `dpwssd`, `sdot` and `dot`
consume, and `A`'s group be one unaligned load and a broadcast. It also removed
every kernel's `k`-tail. Worth 1.35x on `i8`.

**The inner loop held the larger panel.** With the column panel as the inner
loop's invariant, `nr * kpad` is what has to stay in L1 --- and at a panel whose
depth is the whole of `k` that is 32 KiB at `n = 2048`. The row panel is
`mr * kpad`, which is smaller, so it is the one to hold. Worth 1.3x at
`n = 2048` and nothing below it, which is what a cache-residency fix should look
like.

**The block extents read the blocking as a column count.** `NC` columns of `B`
at a panel depth of the whole `k` is a `k/KC`-times-oversized panel, which at
`k = 1024` is the difference between a panel in L2 and a panel in memory. Read
as a working *set* --- `KC * NC` elements of `B`, `MC * KC` of `A` --- the block
narrows as the depth grows and what stays resident stays the same size.

**The working set shrank the blocks past where it could help.** `nc` was capped
at `KC * NC / kpad`, which at `k = 262144` is one column --- and then `A` is read
once per column of `B`. But once a single microkernel panel at that depth already
exceeds the working set, shrinking the block gains nothing, because the panel is
out of that cache either way. The cap now applies only while it can be satisfied.

**A narrow output paid for columns that were not there.** A tile kernel produces
`nr` columns per call whether the output has them or not; at `n = 1` that is one
useful lane in ninety-six. The reduce factorization puts the lanes on `k`, where
there is always more. Worth 50x on a matrix-vector product.

**The packing walked the wrong stride.** `pack_columns` looped lanes outside and
depth inside, so it read a row-major `B` down its columns --- a new cache line
per element. Which loop is inner decides which stride the reads walk, and the
right answer is whichever stride is smaller, which is a question about the
*declared* strides. It is invisible on a square, where the packing is amortized
over the row blocks, and it is the whole cost when `m` is small: worth 1.8x on
`1x1024x1024` and 2.0x on `4096x2x4096`. It also has to be *off* for a
one-lane reduce panel, where an inner loop of one element is all overhead ---
which measuring caught, at a 2x regression on `8x262144x8`.

**A packed panel is a copy, and some panels were already there.** For a
`LaneLayout::Contiguous` kernel the panel *is* a run of rows or of columns, so a
row-major `A` and a column-major `B` already hold it --- which is not an exotic
case, it is a matrix-vector product with the ordinary layout, where the panel
would otherwise be one element written per multiply-accumulate and double the
whole product's memory traffic. `MatView::row_block` and `column_block` hand back
the operand's own memory when the strides already say so. Worth a further 37x on
a single deep dot product, which then needs no working memory at all.

**The driver had no way to compare its two traversals.** The packed traversal
pads the shape up to whole panels and copies them; the streaming reference does
exactly the products the shape names and copies nothing. Below `n = 13` the
reference is *faster* --- by 7x at `n = 1` --- and nothing in the kernel table
said so, because no sequence declared how much a padded product costs it.
`products_per_step` is that declaration, and the driver now counts both
traversals: element accesses (two per packed element, and the *padded* extents,
because a shape narrower than a panel pays for the whole panel) and instruction
issues (one per `products_per_step` padded products, against one per streaming
product plus its two operand reads). Worth 2.6x at `n = 1` and 2x at `n = 4`.

The counts are exact; what they omit is the packed traversal's fixed setup, which
is proportional to nothing. Measured, the model is right at every size except
`n = 6` and `n = 8`, where it takes the packed traversal and the reference is
1.2x to 1.8x faster --- a residual of a few hundred nanoseconds, reported rather
than closed with a tuned constant, because a tuned constant is the thing R8 does
not permit.

**The float encode formed the magnitude once per bit.** `magnitude_bit` negated
the whole register on every call: 10 limbs for `f32`, 67 for `f64`, once per
significand bit. That is O(P*L) where O(P + L) does. Forming the magnitude once
and reading a `P`-bit window out of it was worth 2.9x on `f32` at `k = 2`, where
the encode is nearly the whole cost, and nothing at large `k`, where the
accumulate loop is.

**The float loop asked three questions per product that belonged elsewhere.**
Whether a code is finite is a fact about the *panel*, settled while its codes are
walked anyway. Whether a product of two significands fits an `i64` is a fact
about the *element type* --- `2 * 24 <= 63` for `f32`. And the limb window flushed
by cutting its `i128` into four 63-bit pieces, where `add_scaled` places a
magnitude at a scale in one three-limb spread. Together worth 1.35x at
`n = 1024`, 1.4x at `n = 512`, and 1.4x on the rectangular shapes.

## The constraint that is not ours

A classical GEMM chunks the reduction so its panels fit cache, and adds the
chunks into `C` as it goes. It can do that because its accumulator and its output
are the same width --- and it pays for it with an answer that depends on the
chunking, which is why no two classical `sgemm` implementations agree bit for bit.

This library cannot write partial sums into `C`, because `C` is the *encoded*
output and a partial sum encoded is a partial sum rounded. So without somewhere to
keep exact partial sums, the panels must hold the whole of `k` --- and then the
offer grows with the depth, and a caller with an astronomical `k` either supplies
an astronomical buffer or gets an unblocked traversal.

[`Scratch::with_accumulators`] is that somewhere. With a block of exact
accumulators the reduction is chunked to whatever the cache holds while every
partial sum stays full width, and neither offer grows with `k`:
`KC * (MC + NC)` of panel room and `MC * NC` of accumulators, for any depth at
all. The chunking is invisible in the result, and that is not a hope --- it is
what an exact sum *means*: the sum is order-independent, so it may be split any
way the machine prefers and recombined with no consequence. `CD-10` asserts the
depth-chunked traversal byte-identical to the full-depth one and to the streaming
reference, over depths straddling the chunk boundary.

Measured, the full-depth traversal is still the faster of the two wherever a
caller can afford its offer, so that is what `suggested_scratch` suggests. The
accumulator offer is not a faster traversal; it is the removal of a ceiling ---
the one place where the amount of memory a caller had to find scaled with the
problem.

What remains, and why:

- `A` is still repacked once per column *block*, which at the suggested offer is
  once. Removing the last of it needs a `C` accumulator panel of exact
  accumulators, which is memory the library is not allowed to own (R7). A caller
  who wants it can partition and reuse.
- The generic and coded drivers walk `k` innermost, so `B` is read with stride
  `n`. Fixing that needs an accumulator row, which is again memory the library
  cannot own. The kernel-driven path is the one that packs.
- `m = 1` against a wide `n` still fills a `6`-row panel with one real row, so
  the kernel does six times the arithmetic the product needs. It is now level
  with `matrixmultiply` and 4.6x ahead of `ndarray` on that shape, because the
  packing rather than the arithmetic was the cost; a narrower tile panel is the
  one thing left that would buy something there.
- A deep, thin shape --- `16 x 400000 x 16` --- sustains 5.8 Gmac/s where the
  microkernel alone runs at 38. With full-depth panels nothing is resident: the
  `A` block is 1.6 MB and the `B` block 6.4 MB. The depth-chunked traversal makes
  them resident but produces one kernel call per output column at that shape, and
  measured the two land within 30% of each other. Closing it needs the traversal
  choice to know the chunk depth before it picks the panel shape, which is a
  circularity this driver does not yet break. It is reported, not hidden: `CG-08`
  prints it per pass.
- `f32` is far slower than the integer paths. The size of that gap is measured
  against the *oracles* below, not against our own integer path, because
  comparing a library to itself says nothing about whether the cost is
  reasonable.

## The constraint that is nobody's

The section above is about a constraint a classical GEMM has and this library
does not. This one is about a constraint neither has noticed.

Every traversal in this library --- and every classical GEMM --- issues
`m * k * n` products whatever the operands hold. That is not a property of the
identity. It is a property of how the identity has been walked. Two equal rows of
`A` name the same sum against every column of `B`, so a driver that computes both
has computed one thing twice, and no amount of blocking or vectorisation
recovers it: the arithmetic was already issued.

The upstream Atlas measures the same thing from the other side. Its finite sector
has `3^7 = 2187` length-seven braid words and **26** distinct canonical states ---
a degeneracy of 84 --- and it charges per distinct state, content-addressed by
`kappa`, rather than per word. What it says about this library is that the number
of *expressions* an operand is written in and the number of *meanings* it carries
are different numbers, and only one of them is worth paying for.

[`Collapse`] is that here. One pass numbers each row of `A` by the first row
equal to it; the product is taken over the distinct rows only; the output is
expanded in place. `CD-12` asserts it byte-identical to the packed and streaming
traversals at every degeneracy from one meaning to all of them, and at every
offer including none.

**Why this library may do it and a classical one may not.** Sharing a result
between two rows is sound exactly when the two rows name the same value. Here
they do: the sum is over a declared alphabet, taken exactly, with no rounding
between the products and the single encode, so equal operands give equal sums by
definition. A classical `sgemm` sharing a row would additionally have to argue
that the *order* of its additions was the same, because its answer depends on
that order --- and it is the same argument it cannot make about chunking.

### Throughput against degeneracy, Gmac/s

Against the nominal `m * k * n`, which is what the caller asked for. Reporting
against the products actually issued would print the same number in every row and
hide the whole effect. Every figure is `open`.

| `m x k x n` | `d = 1` | `m/8` | `m/2` | `d = m` | uor packed | ndarray |
| --- | --- | --- | --- | --- | --- | --- |
| 4096x512x512 | 715 | 217 | -- | 38.6 | 40.2 | 0.53 |
| 4096x64x64 | 76.0 | 42.6 | -- | 12.0 | 13.7 | 1.85 |
| 65536x128x128 | 166 | 95.3 | 29.8 | 17.6 | 19.8 | 0.70 |

At `4096 x 512 x 512` with one distinct row that is **17.8x** this library's own
packed traversal and **1350x** `ndarray`, on an answer asserted byte for byte
against both.

### The price of looking

The last two columns are the row to read second. An operand whose rows are
pairwise distinct pays the pass and gets nothing: 4% at `k = 512`, 12% at
`k = 64`, 11% at `65536 x 128 x 128`. That is the honest cost of the question,
and it is why the traversal is entered through an *offer* rather than always.
`Scratch` set the precedent --- a caller who cannot spare the memory gets the
same bytes from a different walk --- and the same rule applies here: a caller who
knows their `A` has no repeated rows offers nothing and pays nothing.

Three things were measured and fixed before that price was 4% rather than 25%:

**The pass hashed through the wrapper.** `Alphabet`'s derived `Hash` and
`PartialEq` ask for one on the *bound*, which is a marker with nothing to
compare, so the comparison was element by element where the peeled slices are one
`memcmp`. At one distinct row --- every row compared against the representative
--- that was half the pass.

**One hash lane is a serial multiply chain.** FNV's mix has five-cycle latency
and one-cycle throughput, so a single accumulator runs at a fifth of the
multiplier's rate. Eight independent lanes over `chunks_exact(8)`, destructured
rather than indexed so the bounds check does not survive, took the pass from
2.4 ns per element to 0.6.

**The fold cost more than a twelfth of the row.** Folding eight lanes by running
the byte mix over all sixty-four of their bytes is sixty-four *serial*
multiplies per row --- at `k = 512` that is a twelfth of the elements and a third
of the time. One rotate-xor per lane and a multiply-shift finisher does it in
sixteen operations.

**The expansion wrote cells.** Replicating an output row is `n` strided writes
each costing two index computations, where a row-major output already holds the
row as a run. [`MatViewMut::copy_row`] is [`MatView::row_block`]'s mutable twin
and moves it. It is also where the earlier measurement was wrong: the first
version read 0.64 GB/s and the second 24 GB/s, and the difference was entirely
the first touch of an 8 MB output. A copy rate measured on unfaulted pages is a
page-fault rate.

### What it is not

The upstream paper is careful, and this section is careful in the same way. The
Atlas claims polynomial-time evaluation for the *finite* sector's invariant
decisions and explicitly disclaims subverting the `#P`-hardness of general tensor
contraction. The analogue here is exactly as narrow: a dense product of two
operands with no repeated content costs what it costs, and the row above says so.
What the collapse traversal removes is the assumption that the cost of a product
is a function of its *shape* --- and for a batch over a vocabulary, a one-hot or
gather product, a padded batch, or a low-bit quantised operand, the shape and the
content say very different things.

One thing it does not do, decided by a declaration rather than by the data: an
epilogue that reads `C` gets the packed traversal, because two rows with equal
rows of `A` still have different outputs when the `C` they read differs.

### Columns, and why the layout decides what they cost

Sharing is not always on the `A` side, and the other side needs no second
traversal. `(A * B)^T = B^T * A^T`, transposition is a stride, and equal columns
of `B` are equal rows of `B^T` --- so [`Triple::transposed`] is the whole of it,
and `gemm_collapsed` on that triple runs the same pass, the same compaction, and
the same expansion. `CD-12` asserts that too.

What it costs is not the same, and the reason is worth stating because it is the
one thing about this traversal that the *caller's* declaration decides. The pass
has to read the axis it is collapsing. At `512 x 512 x 4096`, one distinct
column:

| `B`'s layout | `d = 1` | `d = n/8` | `d = n` | uor packed |
| --- | --- | --- | --- | --- |
| column-major | 194 | 118 | 34.0 | 37.1 |
| row-major | 54.2 | 44.6 | 29.7 | 38.0 |

A column of a column-major `B` is a run, so the pass reads it the way
[`MatView::row_block`] hands it over and compares it with one `memcmp`. A column
of a row-major `B` is `k` reads on `k` different cache lines, and there are `n`
of them --- `Theta(k * n)` cache misses, which is `1/m` of the product's work at
thirty times the cost per access. That is the whole distance between 5.2x and
1.4x. Nothing about the answer differs, and `CD-12` covers both.

The expansion has the same shape and the same answer: which loop is inner is
decided by the output's strides and not by the order the rows were written in.
Every column of the expansion is independent, so the descending walk that makes
it safe is preserved either way, and the inner loop is free to be whichever axis
is the near one. Getting that wrong is expensive and was: walking the rows of
`C^T` outermost over a row-major `C` reads a cache line per cell, and it held the
column figure at parity until the loops were exchanged.

## The other constraint that is nobody's

The collapse traversal is about the operand having fewer *meanings* than
expressions. This one is about the operand being a *code* at all.

`crates/uor-matmul-gemm/src/coded.rs` used to describe itself accurately: the
weights arrive as codes, they are decoded, and from there it is the same
accumulation the dense driver runs. That is decode-then-multiply. It issues the
same `m*k*n` products and adds a decode on top, so the codec buys residency and
pays for it in throughput --- the wrong direction, because the codec is the thing
that should make the arithmetic cheaper.

When the operand is a code, the product is a table read:

```text
T[i][p][c] = sum over t < Bk of  A[i, p*Bk + t] * decode(c, t)
C[i][j]    = sum over p       of  T[i][p][ index_of(w[p][j]) ]
```

The table is built once per row tile and per block of the reduction, and read `n`
times. Its column loop is one read and one add per code, covering `Bk` weights,
and it contains no multiply.

### Why it is available here and nowhere classical

`T[i][p][c]` is a partial sum of the same products, and the total is the sum of
those partial sums. A sum is a function of the multiset of its products, so
regrouping them changes nothing --- the same licence tiling already uses.

A classical `sgemm` cannot do this at all. Its `T[c]` would carry its own
rounding error, and reusing it across `n` columns would propagate that error `n`
times. It would have to argue about the *order* of its additions, which is
exactly the thing it cannot do. Tabulation is available only to a library whose
sum is exact, and that is the sense in which it is not a GEMM trick borrowed from
elsewhere.

### The op counts, and where they cross

`m*(k/Bk)` tables, each `S*Bk` products to build and read `n` times, against
`m*k*n` products:

```text
tabulated = m*k*S + m*k*n/Bk        dense = m*k*n
```

so tabulation is cheaper exactly when `n*(Bk - 1) > S*Bk`. `model/tiers.toml`
records the crossing per codec and `CM-04` recomputes it. Measured at
`8 x 256 x n` over `Book<256,8>`, whose crossing is `n = 293`:

| `n` | ops vs dense | multiplies vs dense | wall clock vs streaming |
| --- | --- | --- | --- |
| 64 | 0.24x | 0.25x | 0.26x |
| 256 | 0.89x | 1.00x | 1.00x |
| 512 | **1.60x** | 2.00x | 1.82x |
| 4096 | **5.33x** | 16.0x | 4.94x |

The census and the clock cross in the same place, and the ratios at `n = 4096`
are the derivation's `16x` and `5.33x` exactly. `CG-10` prints this; `CU-06`
asserts the closed forms behind it --- `adds == table_reads == m*n*(k/Bk)` and
`multiplies == m*k*S`, so every multiply the traversal issues is in the build.

### Against the kernels, which is the harder question

Beating the streaming traversal is not the bar. The bar is this library's own
packed AVX2 tile path over the *decoded* weights --- which is handed its operand
already dense, so the comparison is generous to it. Over `Book<256,8>`, running
`Traversal::Blocked`, which is the default and therefore what a caller gets. The
`picked` column is read from the census, not recomputed from the predicate.

| `m x k x n` | default | packed | vs packed | picked |
| --- | --- | --- | --- | --- |
| `1x1024x4096` | **8.35** | 1.15 | **7.24x** | table |
| `1x1024x8192` | **9.27** | 1.15 | **8.08x** | table |
| `8x1024x4096` | **35.94** | 6.95 | **5.18x** | table |
| `64x1024x4096` | **63.85** | 24.94 | **2.56x** | table |
| `64x4096x4096` | **48.93** | 14.38 | **3.41x** | table |
| `256x1024x4096` | **51.96** | 33.60 | **1.55x** | table |
| `64x1024x16384` | **57.26** | 24.01 | **2.39x** | table |
| `1000x512x512` | 38.99 | 39.81 | 0.98x | kernels |
| `1x8192x1` | 7.24 | 21.01 | 0.34x | kernels |
| `3x1024x4093` | **11.79** | 3.49 | **3.38x** | table |
| `17x1032x1021` | **30.96** | 13.79 | **2.25x** | table |

The last four shapes divide nothing --- a shape below the break-even, a degenerate
dot product, a ragged row tile, a prime column count --- and they are in the sweep
because a traversal with a cliff at an awkward size would show it there.

### The traversal issued no vector instruction

The number above the previous one was `28.04` at `64x1024x4096`, and the whole
distance to `52.24` is this: **the tabulated traversal compiled at the target's
baseline**. Every AVX2 sequence in this workspace is behind `#[target_feature]`
in `uor-matmul-kernels`, because that is the only crate permitted `unsafe`, and
the table's column loop and its build were written in `uor-matmul-gemm`. Counted
off the disassembly of `tabulate`, the traversal contained zero `vpaddd` and zero
`vpaddq`. It was not slow because the construction was wrong; it was slow because
nothing had told the compiler it could use the machine.

Isolated on one row tile of `4096` columns and `k = 1024` over `Book<256,8>`, so
that the two halves can be read apart:

| | today | in vectors | |
| --- | --- | --- | --- |
| column loop | 17.6 Gmac/s | **86.7** | 4.9x |
| table build | 2.1 Gprod/s | **25.1** | 11.7x |

Every SIMD target this workspace supports now has both: AVX2 for the 32-bit and
the 64-bit lane, NEON through `vmlal_s16`, and SIMD128 through
`i32x4_extmul_*_i16x8`. The last two are compile-checked against their targets
here and pinned to the reference by `CB-08` when the `cross` job runs them
natively --- the same standing the NEON dense kernels have always had, and stated
rather than left to be discovered.

The column-loop figure is two eliminations, not one, and they are independent:

- **Vectors**: 17.6 to 44.7 Gmac/s. Sixteen `i32` lanes are two 256-bit registers
  and the entry is one cache line, so the step is two fused load-adds.
- **The exact accumulator, out of the reduction**: 44.7 to 86.7. This one is not
  about instructions at all. A narrow lane holds `capacity` products exactly ---
  133144 of them at `(i8, 128)` against a `k` of 1024 --- so it carries the
  *whole* reduction and `AccOf<E>` is touched once per output element. The lane
  had been folded into the exact accumulator once per chunk, which is
  `m*n*(k/Bk)/depth` reads and writes of a 16-byte word to save 4-byte ones. That
  is what "encode once" already said, and the traversal was not doing it.

Simply enabling AVX2 for the whole build --- `-C target-feature=+avx2` --- was
worth 15%, not 2.9x. The loop structure had to change with it, which is why the
sequences moved to the crate that owns instruction selection rather than the flag
being turned on.

#### The arithmetic density, which is the actual claim

One 256-bit add covers `8 * block` products at a 32-bit lane: eight output rows,
each carrying a whole codeword. At `Book<256,8>` that is **64 products per
arithmetic instruction**.

Nothing dense is the same shape. `vpmaddubsw` plus `vpmaddwd` --- the best an
AVX2 host without VNNI has --- is about 10 products per instruction. `vpdpbusd`,
the densest integer instruction x86 has at all, is 32 and cannot be told to cover
more. The table's density is a property of the *codec*: a codebook that names a
longer block is a denser instruction, without any change to the hardware or to
this code. That is the paradigm difference, and it is measurable rather than
rhetorical.

What the table pays for it is traffic: `lane_bytes / block` bytes per product,
which is 0.5 at a 32-bit lane and a block of eight, against about 0.23 for a
dense tile reusing packed panels across `MR` rows and `NR` columns. So the table
is instruction-cheap and bandwidth-expensive, and the crossover moves with the
codec's block --- a block of sixteen halves the traffic per product and leaves the
instruction count alone.

#### What the kernel boundary cost, and what it took to stop paying it

The first version of this cost a factor of three at a one-row tile ---
`1x1024x4096` fell from 9.93 to 3.37 --- and the cause was not the vectors. It
was three things the boundary made explicit that had been implicit before, and
each had to be taken back:

**The gather took an index stream, so the driver built one.** One `u32` per code
is `4 / (rows * block)` bytes per product: a thirty-second of the entry traffic
at the widest tile, and exactly as wide as the entry it addresses at a one-row
tile. The answer is that there was nothing to build. `Enumerable::as_index_stream`
is the codec saying its stored codes already address its enumeration ---
`index_of(c) == c & (CODE_SPACE - 1)`, which holds when the space is a power of
two and the enumeration is the code type's own order --- and then the operand's
own memory *is* the stream. That is the rule `MatView::row_block` already follows
on the dense side: borrow when the layout holds what is wanted, copy otherwise.
`Packed` still copies, because its index is a mixed-radix decomposition of its
byte and not the byte.

**Both reference sequences took a runtime row count.** With `rows` runtime the
entry is a slice of unknown length and its accumulation is a chunked iterator
around what should be `rows` registers; the compiler also re-derives each lane's
address as a multiply. Const-generic in the tile height, as the gather already
was, the reference build went from half the traversal at a one-row tile to a
fraction of it: `1x1024x4096` moved 3.57 to 6.19 on that change alone.

**The frame size was bounding the reduction.** `GATHER_SLOTS` sizes the buffer an
index run is built in, and it was also capping the stack depth. At a one-row tile
the depth that pays is 128 and it was capped at 32. A chunk deeper than the buffer
is now walked in windows of it, so a frame size bounds a frame and nothing else
(R8).

One thing was tried and *removed* by measurement. Padding a narrow tile up to the
narrowest vector sequence --- exact, because the alphabet's zero contributes
nothing --- was worth 1.86x at `17x1032x1021` against a traversal whose narrow
tiles were framing-bound. Against one whose are not, it loses everywhere it used
to win: 25.0 against 29.6 there, 5.9 against 6.2 at `m = 1`. It was compensating
for a defect, and when the defect went it went with it. A knob that stops paying
is a knob that goes.

### The narrow tiles, and the exponent that was hiding in a register

Two shapes came out of the refactor short of where they started:

| `m x k x n` | before | after | closed | |
| --- | --- | --- | --- | --- |
| `1x1024x4096` | 8.9 | 5.35 | **8.35** | 0.60x -> 0.94x |
| `3x1024x4093` | 12.1 | 8.02 | **11.79** | 0.66x -> 0.97x |

Three measurements said where it was not.

**Not the column loop.** Isolated --- one row, 4096 columns, 128 slots, the same
128 KiB stack and 1 MiB code stream the traversal walks --- the shipped shape
runs at **15.09 Gmac/s**, and it is the fastest of five forms tried. The
pre-refactor shape, two columns with the accumulation in registers and the codes
read inline, is **13.23** on the same data. The column group barely moves it:
15.09, 14.68, 14.41, 14.30, 14.51 at groups of 1, 2, 4, 8, 16.

**Not the build, and not the setup.** `1x1024x4096` and `1x1024x8192` are in the
sweep as a pair for this: the build is `k/block * S * block * rows` and does not
move with `n`, so the two times solve for both terms. They give an
`n`-independent cost of **0.086 ms** against a total of 0.784 --- eleven percent.

**Not the collapse.** Disabling the column-collapse pass entirely leaves
`1x1024x4096` at 5.35, unchanged to two figures.

So the per-column path in the traversal ran at about **3.3 cycles per code-step
where the same loop in isolation ran at 1.3**, and what was left to explain was
framing the isolated form folded and the shipped one did not.

**It is the slab, and the probe says so.** The isolated loop was rewritten with
the slab and the depth as runtime values --- what actually crosses the
`TableSpec` boundary --- and measured against the same loop with them as
literals, on the same data:

| binding | group 1 | group 4 | group 16 |
| --- | --- | --- | --- |
| slab and depth runtime | 3.84 | 5.72 | 6.92 |
| slab a literal, depth runtime | 9.40 | 6.96 | 7.89 |
| both literal | 14.92 | 10.62 | 9.54 |

The shipped traversal sat at 5.35, inside the runtime band. Depth can stay a
register; the slab cannot. And the difference is not the mask --- a literal slab
makes every slot's base a constant displacement, so the slot loop unrolls and
the cursor disappears.

**The ceiling was a sentence, not a boundary.** What stood here before said
closing it meant carrying the code space into the sequence selection, which
would multiply the monomorphizations by the number of codecs. That was wrong on
its face, and it is worth saying why, because the shape of the error is the
usual one. The codec never reaches the column step. It contributes exactly one
thing: the slab, which is `slab_codes(CODE_SPACE) * rows`, which the boundary
has *already asserted* is a power of two, and which is *already an argument* of
the sequence call. So there was no plumbing to add and no boundary to move. The
free value was never a codec --- it was a single exponent, bounded by sixteen
because a code is a `u16` and `2^16` codes is every code there can be.

Enumerating it is one `match` (`dispatch_slab!`), nested inside the `(rows,
group)` dispatch that was already there. The wildcard arm binds the constant to
zero, and the runs read zero as "the caller did not know it" and take the slab
from their argument --- the same body at a different binding, so this is one
sequence and not two (R13), and a code space the list does not name is computed
rather than refused (R8). `shift` fell out entirely: the boundary derives it as
`rows.trailing_zeros()`, so at a compile-time tile height it is the tile
height's.

That closed both shapes: `1x1024x4096` from 5.35 to 8.35 and `3x1024x4093` from
8.02 to 11.79, with no other shape moved outside run-to-run noise.

**What is still short, stated as what it is.** Both shapes are a few percent
under their pre-refactor peak, and the isolated loop still runs faster than the
traversal reaches. Some of that is the table itself and is not recoverable: the
build costs `CODE_SPACE * k * rows` products whatever `n` is, which at `S = 256`,
`k = 1024` and one row is 262144 products against 4.19M of useful work. That is
the `n`-independent 0.086 ms measured above, and it is now seventeen percent of a
much smaller total rather than eleven percent of a larger one. It amortizes with
`n` exactly as the algebra says it should, which is why `1x1024x8192` reads 9.3
where `1x1024x4096` reads 8.4. The remainder is unattributed and is not claimed
as anything.

One more hypothesis was checked and does not hold. `tabulation_depth` sizes the
table against L2, and at a one-row tile the slab is 1 KiB, so 128 slots is a
128 KiB table and every lookup is an L2 hit *by construction* --- L1 is 32 KiB.
Splitting the reduction into passes that each fit L1 costs almost no extra
traffic, because each code is still read exactly once and only the 16 KiB output
lane is re-touched per pass. It was measured at depths of 128, 64, 32, 16 and 8
and there is no signal: at a group of sixteen the L2-sized depth reads 15.13,
15.22 and 15.18 across three runs, which is the steadiest figure in the table,
while a 32 KiB depth reads 16.99 and then 9.79. The spread between neighbouring
configurations is larger than any trend across them. The existing sizing stands,
and a knob is not added to chase variance.

### Three factorizations, and the offer decides which

A coded operand has three of them, all computing the same bytes:

- **The table**, when the codec's block is long enough to repay building it. It
  never materializes the dense weights, which is what the codec is for.
- **The tile kernels**, when the caller's offer holds the whole decoded operand
  *and* room for the kernels' own panels. That is the caller declaring it can
  afford the dense weights, so the route is never taken behind its back: no offer,
  no route. `1000x512x512` is where it earns its place --- `1.4` Gmac/s streamed,
  `39.2` through the kernels, against `39.9` for a dense operand handed over free.
- **The stream**, which needs nothing at all and runs where neither of the above
  can. It is what makes the traversal total on a target whose RAM cannot hold a
  decoded row.

`1x8192x1` is the shape that cannot be won: `n*k` decodes for `n*k` products, so
no method beats one that is given the decode for free. It reaches `0.36x` of a
dense kernel and the row exists to say what the decode costs, not to claim
otherwise.

### The third degeneracy: distinct columns

Collapse charges per distinct *row* of `A`. Tabulation charges per distinct
*code*. Neither charges per distinct **column of the coded operand** --- and two
columns whose index streams agree read the same table entries in the same order,
so their accumulations are equal. Not nearly: identically, and for the same reason
the rows are.

At `16x1024x4096` over `Book<256,8>`, with column `j` repeating column `j % d`:

| `d` | degeneracy | collapsed | uncollapsed | |
| --- | --- | --- | --- | --- |
| 1 | 4096x | **58.70** | 47.04 | **1.25x** |
| 8 | 512x | **98.63** | 38.22 | **2.58x** |
| 64 | 64x | **59.05** | 47.60 | **1.24x** |
| 512 | 8x | **41.49** | 31.10 | **1.33x** |
| 4096 | 1x | 17.14 | 28.73 | 0.60x |

Read those ratios and not those figures. This sweep swings by a factor of two
between runs on a two-core shared runner --- `d = 64` has measured 45.30, 59.05
and 94.89 for the same code --- where the shape table above repeats to within 1%.
The collapse is worth something at every degeneracy and the last row is the price
of looking; how much of each is not a number this machine can settle.

The ceiling at `d = 1` is the table *build*, which does not collapse: it is
`m*k*S` products whatever the columns do. That is why 4096-fold degeneracy buys
2x and not 4096x, and it is the honest shape of the construction rather than a
disappointment.

One thing this cost to get right. Naming the columns instead of counting from a
first one turns the code base and the accumulator base from induction variables
into indexed loads, and measured that **halved every shape, including the ones
with nothing to collapse**. So there are two column loops: the consecutive one,
unchanged, and an indexed one reached only when the collapse has something to do.
A specialization in the direction that costs something is a second function, not a
parameter.

### Everything between 2.95 and 28 was overhead

The first working version of this traversal reached `2.95` on the fourth row.
None of what followed was arithmetic:

| what was removed | `64x1024x4096` |
| --- | --- |
| the first working version | 2.95 |
| the exact accumulator out of the block loop, into a narrow lane | 5.51 |
| the row count made a compile-time one | 9.03 |
| the step's addressing walked instead of indexed | 13.09 |
| the table's stride made the compile-time row count | 13.90 |
| the column loop given more than one load in flight | 20.19 |
| the codebook decoded once per call rather than once per tile | 25.33 |
| the build's entry kept in registers across its codeword | 26.44 |
| the row tile widened, once the above made it pay | **27.91** |

Three lessons, and they generalise past this module:

**A runtime length where a compile-time one belongs is an order of magnitude.**
An `R`-word slice of unknown length compiles to a bounds check, a scalar prologue
and a vector epilogue wrapped around eight adds. With `R` a constant the
accumulation is registers, the entry is one cache line, and its address is a
shift. Four of the nine rows above are that one bug in four places.

**Traffic is not arithmetic and does not look like it.** The exact accumulator is
sixteen bytes and the naive loop touched one per output cell per block ---
`m*n*k/Bk` touches, a gigabyte moved to compute a quarter of a billion products.
A narrow lane fixes it the way the float path's did: a chunk of the reduction is a
sum of `Bk`-product entries, so a 32-bit register holds it exactly for a depth
`fits_narrow` states. The build had the same disease in a different place --- it
read and wrote every entry once per element of its codeword, `Bk` times the
traffic --- and the fix is the same shape: keep the entry in registers and write
it once.

**Every tuning constant here was measured, and two of them changed sign.** A
stack at a quarter of L2 beat one at a half while the column loop had one load in
flight, and lost to it once the loop had two. A sixteen-row tile lost to an
eight-row tile while the build re-read every entry, and won once it did not.
Neither is a fact about caches; both are facts about what else the loop was doing,
and neither would have been found without re-running the sweep after each change.

### Both sides of that comparison were describing something else

The predicate above compares instruction counts, and until this was measured
neither of its two terms described a sequence that has ever shipped.

The table's side was `MAX_BLOCK * rows` products per instruction. A tile of
`rows` is `rows / lanes_per_add` instructions, so pricing it as one over-states
the table by the register count --- a factor of two at a sixteen-row tile and a
32-bit lane. The dense side was a model constant of 32, which is `vpdpbusd`'s
density; the AVX2 `i8` tile this workspace ships declares 16, and the host these
figures come from has no `vpdpbusd` at all.

Two errors, both a factor of two, in opposite directions. They cancel exactly,
and every recorded `break_even_n` is the same number under the corrected form ---
which is how the mistake survived. **Two errors that cancel are not a
derivation.** Where they stop cancelling is a host with VNNI, whose dense tile is
four times denser per instruction while the table is not: the old form takes the
table from `n = 683` where the corrected one says no `n` pays at all. `CM-04`
asserts that case now, so the claim is falsifiable on a machine nobody here has.

Both numbers are read off the two specs at run time, so a host is priced at its
own sequences. The recorded rows in `model/tiers.toml` carry the register width
they are written for, because a break-even without one is a number about nothing.

One term was tried in the honest direction and put back. Reading the *chosen
tile's* `mr` for the dense row count says a one-row kernel wastes no lanes, which
is true --- and it declined the table at `1x1024x4096`, where the table is 7.2x
the dense path. The reason the dense path is weak at small `m` is not lane waste:
it packs `n * k` elements to compute `m * n * k` products, so at `m = 1` the copy
is the same order as the arithmetic. The blocking row count stands for that, and
now says so.

### The predicate is a comparison of instructions, not of operations

The op-count model in the previous section --- `S + n/Bk` against `n` --- prices a
build multiply and a kernel multiply the same, and they are not the same. A
tabulated lane operation covers `Bk * rows` products; one instruction of a dense
tile covers `KERNEL_PRODUCTS_PER_STEP` of them. Counting both as one apiece
selected the table at `1000x512x512`, where the kernels are four times faster per
product. The corrected predicate is

```text
cols * (block*rows - kernel_step) > code_space * kernel_step * block
```

and there is a second term without which it is wrong in the other direction: a
dense tile issues its products per instruction only when it has `KERNEL_ROWS` rows
to fill. At `m = 1` it has one useful row in six, and scaling the dense side by
the rows actually present is what keeps the table selected there --- where it is
`8.8x` faster and the first version of the predicate said no. Both terms are in
`model/constants.toml`, both change which traversal produces a byte and never
which byte, and `CM-04` recomputes every recorded break-even from them.

### What the last gap cost to close

`1000x512x512` ran at `1.4` Gmac/s against the kernels' `39.9` and the reason was
not the lane. Output-major with a length-512 dot re-reads every row of `A` once
per column: `m*n*k` element reads, and at that shape the traversal is bound by
memory, not by arithmetic. No amount of vectorizing the dot changes it --- measured,
the 32-bit lane and the 64-bit lane are within 5% of each other there, and both
are within noise of where it started. Blocking is what fixes it, and the library
already has blocking: it is `gemm_packed`.

Two things were in the way and both are gone. `MatViewMut` is not `Copy`, so a
driver holding one inside a larger value could not hand it to a constructor that
takes ownership; it now has `reborrow`, which is nine lines and useful anywhere one
traversal is expressed as another over the same output. And the traversal's
boundary now asks for `Kernelized` --- the marker meaning "this element family has
microkernels" --- which is exactly the right thing to require of a traversal that
competes with them.

### What an adversarial review found

The construction above was reviewed against its own claims rather than its own
output: for each subsystem, what would still pass if the code were wrong. Eight
defects survived checking, and every one of them is reachable from documented
public API. They are recorded because the pattern is more useful than the list.

| where | what | how it showed |
| --- | --- | --- |
| NEON `i16` build | rows 8--15 read the wrong activations; rows 12--15 never read | `cargo test` under `qemu-aarch64` |
| tabulated driver | a narrowed column block skipped repeated columns and never filled them | 896 of 1024 cells wrong |
| table selection | a named backend, or an odd codeword width, panicked inside `gemm` | 246 of 250 selections returned `None` |
| `gemm_packed` | did not terminate at a bound that shrinks the lane below `k_group` | `timeout` killed it |
| `gemm_float` | divided by zero at `k == 0` | panic on every build |
| `admits` | shifted a panel past its slot when the other panel was all zeros | panic in the checked profile |
| `MatView::new` | accepted a view whose cells are outside the buffer | `wrapping_mul` in the reach |
| every `gather` | masked the entry's base and not the `rows` lanes read from it | safe method, unsafe read |

**Six of the eight were invisible to a gate that claimed the ground.** That is
the part worth keeping:

- `CD-13` swept the scratch offers with *one* shared fraction, so the pair that
  breaks the traversal --- an index long enough to collapse against an
  accumulator offer too small for the whole output width --- was unreachable by
  construction.
- `CD-01` checked that selection cannot fail for every backend, for the *dense*
  kernels only. The table half did not exist.
- `uor-matmul-gemm` did not enable `uor-matmul-kernels/std`, so its harness
  linked the kernels with runtime detection off and every table sequence resolved
  to the portable one. `k_group` was always one, no vector layout was ever
  packed, and the driver's own suite asserted against a sequence no consumer on
  the host will run. `uor-matmul-kernels` had already learned this for `CB-02`
  and left the note; the driver never applied it.
- `CB-05` --- "the two wasm configurations agree" --- was asserted by a `cargo
  build`. The CI job that names it ran neither `uor-matmul-kernels` nor
  `uor-matmul-gemm`, which is every line of SIMD128 and every driver claim, and
  enabled SIMD128 only for the build steps. The `no-alloc` recipe's "with and
  without SIMD128" pair compiled the same configuration twice, because
  `target.<triple>.rustflags` in `.cargo/config.toml` outranks the
  `build.rustflags` the second arm passed.
- `CU-01` reused `target/cu01-asm`. `--emit asm` writes a `.s` only when the
  crate actually compiles, so a warm directory made `cargo rustc` answer
  "Finished" and leave nothing: it reported three objects on a run where two
  files existed, and the absent one was `uor-matmul-gemm`.
- R3's note requirement exempted every path containing `gemm` --- the whole
  driver, which is where the accumulation lives and therefore the only crate the
  rule is about. Fifty sites were behind it.

Every gate above now asserts what it names, and each fix was checked by putting
the defect back and watching the gate fail.

**The narrow rows do not want a vector sequence, and that is measured now rather
than assumed.** The vector table sequences exist only at eight and sixteen rows;
at four, two and one the reference carries the tile. What stood here said four
`i32` is exactly one 128-bit register, so the absence was "unwritten work rather
than an impossibility, and the next thing to measure". It has been measured. A
128-bit column step at four rows --- the entry as one `__m128i`, four column
accumulators as four more, the whole step four `paddd` instead of sixteen scalar
adds --- runs at **46.9 Gmac/s against the scalar 46.4**, three times in a row:

| | Gmac/s |
| --- | --- |
| scalar, rows 4, group 4 | 46.42 / 46.49 / 46.67 |
| 128-bit vector, same shape | 46.86 / 46.66 / 46.91 |

1.01x, which is nothing. The reason is worth keeping: at four rows the stack is
`256 * 4 * 4` bytes per slot and 512 KiB over the run, so the column step is
bound by the entry loads and not by the adds it issues. Vectorizing an add
cannot speed up a loop that is waiting for memory. At one row the same argument
is stronger --- the entry is four bytes, so a vector step would need a gather,
and `vpgatherdd` on this microarchitecture is slower per element than the scalar
load it replaces. The reference is not carrying those tiles as a stopgap; it is
carrying them because there is nothing to win.

The same question at *eight* rows answers the one remaining registration gap.
AVX-512 registers `(16, 1)` and `(16, 2)`; AVX2 registers those and the two
eight-row pairs, so an eight-row tile on an AVX-512 host runs a 256-bit sequence.
Whether a 512-bit one would beat it cannot be timed on this host, but the ceiling
can be:

| rows 8 | Gmac/s |
| --- | --- |
| scalar, group 1 | 54.86 |
| 256-bit, group 1 | 57.82 |
| scalar, group 2 | 39.20 |
| 256-bit, group 2 | 57.73 |

The vector form reaches 57.8 at *both* groups. Group two issues twice the adds
per slot, so if the adder were the limit it would be the slower of the two; it is
not, which places the limit on the entry loads. A 512-bit step issues the same
loads from the same addresses, so it cannot pass a ceiling the 256-bit step
already reaches. The eight-row pairs are absent from the AVX-512 table for the
same reason the narrow rows are absent from all of them.

**Both of the things that were once recorded as unfixed are now fixed, and the
first of them is worth keeping because of what it says about gates.**

`audit-purity`'s R2 half matched a token list and a float *literal*. So
`let mut t = a; t = t + b;` on two `f32` parameters --- the most direct violation
the rule has --- passed it, and it described itself as "deliberately crude" and
pointed at `CU-01` as definitive.

`CU-01` is definitive for what it can see, and that was measured rather than
assumed: a `black_box`-guarded float add placed inside a function that is
certainly emitted is caught, `addss` and all. But it sees only what was
*codegen'd*, and an uncalled `pub fn` is not in the rlib at all --- measured, the
symbol is absent from a 1.9 MB rlib and from every `.s` file the gate reads. Such
a function is codegen'd in a *downstream* build, where this repository's gates do
not run. Neither half covered the other, and the source half was the one with the
gap.

It now tracks which values are floats rather than matching tokens: parameters and
`let` bindings whose declared type mentions `f32` or `f64`, float literals, `as`
casts, aliasing chains, and the elements of a float slice bound by a `for`. Eight
plants, each caught:

| plant | |
| --- | --- |
| `let mut t = a; t = t + b;` on two `f32` params | caught |
| `a * b` on two `f64` params | caught |
| `let y: f32 = x; y - x` | caught |
| `for x in v { s = s + *x }` over `&[f32]` | caught |
| `a.mul_add(b, b)` | caught |
| `let z = n as f32; z / 2.0` | caught |
| `*p + *q` on two `*const f64` | caught |
| `let b = a; let c = b; c + c` | caught |

and the shipped crates are clean under it.

The second was the declared alphabet bound. `Alphabet::new` checks it,
`as_alphabet` checks it, and `Alphabet::of` exists only for `Full<E>`, which
admits everything --- so every constructor the library offers establishes it. What
does not is `bytemuck::TransparentWrapper::wrap`, and that derive is not
removable: `uor-matmul-core` is `#![forbid(unsafe_code)]`, and the trait is how it
wraps a slice zero-copy at all. Removing it would remove the checked zero-copy
tight-bound path with it.

The bound reaches an answer in exactly one place --- the narrow run length in
`dot_ref` --- so that is where it is now verified. Free in release, and a panic
under the checked profile, which `CT-02` runs over the whole corpus. `CD-03`
asserts both profiles by name, so the test cannot pass by not running:
`Alphabet::wrap(100)` under a declared bound of one is loud where it used to be a
quietly wrong integer.

## Against the oracles

C3 is a hard constraint: scaling is compared against the oracle's scaling. Both
sides are measured in one process, over one sweep spanning ten orders of
magnitude in MAC count, with the answer asserted inside the timed region --- a
speed measured on the wrong bytes is not a measurement.

Two-core shared runner with AVX2 and no AVX-512. Best of as many repetitions as
fit a fixed wall-clock budget, so the large end gets more than two samples.
Operands are a recorded pseudo-random fill: they used to be all ones, which
flattered this library twice, because then every `f32` product shares one
exponent and the complete accumulator's limb window never once has to flush.
Every figure is `open`.

### Throughput on squares, Gmac/s

| `n` | uor `i8` | uor `i32` | ndarray `i32` | nalgebra `i32` | uor `f32` exact | matrixmultiply `f32` |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.007 | 0.010 | 0.017 | 0.014 | 0.017 | 0.013 |
| 4 | 0.27 | 0.36 | 0.46 | 0.46 | 0.110 | 0.64 |
| 8 | 0.53 | 1.16 | 1.02 | 1.55 | 0.206 | 4.66 |
| 16 | 4.09 | 4.01 | 1.50 | 3.15 | 0.338 | 14.6 |
| 32 | 6.44 | 6.21 | 1.74 | 4.17 | 0.532 | 26.8 |
| 128 | 19.7 | 17.1 | 0.78 | 4.50 | 0.940 | 28.9 |
| 512 | 38.5 | 29.9 | 0.53 | 4.71 | 1.103 | 43.2 |
| 1024 | 37.7 | 29.1 | 0.21 | 4.58 | 1.118 | 43.3 |
| 2048 | 32.9 | 26.4 | 0.15 | 4.54 | 0.873 | 41.2 |

At `n = 2048` a single one of our repetitions already exceeds the wall-clock
budget, so that row is a best-of-one and moves by up to a third between runs on a
shared machine. It is reported rather than smoothed, and the row above it is the
one to read.

The three smallest rows are where an oracle is ahead, and they are the rows where
a call is a few hundred nanoseconds and dominated by what happens before the
arithmetic. `ndarray`'s figure there includes a heap allocation and ours includes
none; what ours includes instead is walking the kernel table and weighing two
traversals, which is what buys the rest of the table.

### Throughput on shapes that are not squares, Gmac/s

A square is one use-case and not the interesting one. Each row below stresses a
different half of the driver.

| `m x k x n` | uor `i8` | uor `i32` | ndarray `i32` | nalgebra `i32` | uor `f32` | matrixmultiply `f32` |
| --- | --- | --- | --- | --- | --- | --- |
| 1024x1024x1 | 37.5 | 16.7 | 3.24 | 0.18 | 0.217 | 2.25 |
| 1x1024x1024 | 1.09 | 0.94 | 0.20 | 0.14 | 0.153 | 1.18 |
| 8x262144x8 | 6.64 | 4.52 | 1.70 | 1.82 | 0.471 | 16.0 |
| 1x1048576x1 | 40.1 | 10.2 | 2.41 | 0.34 | 0.152 | 0.31 |
| 2048x8x2048 | 9.24 | 8.34 | 1.08 | 0.82 | 0.238 | 13.4 |
| 4096x2x4096 | 2.57 | 2.51 | 0.41 | 0.17 | 0.068 | 0.86 |
| 509x1021x257 | 26.8 | 24.5 | 1.18 | 5.54 | 1.113 | 41.6 |

### Latency at the smallest shapes, nanoseconds per call

| `n` | uor `i8` | uor `i32` | ndarray | matrixmultiply |
| --- | --- | --- | --- | --- |
| 1 | 140 | 100 | 60 | 80 |
| 2 | 150 | 110 | 69 | 80 |
| 3 | 180 | 130 | 89 | 89 |
| 4 | 240 | 180 | 140 | 100 |

### Fitted exponent of throughput against MAC count

| | exponent |
| --- | --- |
| uor `i8` | 0.35 |
| uor `i32` | 0.32 |
| ndarray | 0.04 |
| nalgebra | 0.20 |
| uor `f32` | 0.28 |
| matrixmultiply | 0.32 |

A positive exponent means throughput still improving with size: the small end is
latency-dominated for everyone. `ndarray`'s 0.04 is the opposite story --- it
starts fast and *degrades*, from 1.9 Gmac/s at `n = 64` to 0.15 at `n = 2048`,
which is a cache story rather than an arithmetic one. Ours tracks
`matrixmultiply`'s almost exactly, which is what a driver with the same blocking
structure should do.

### What this says

**On integers, this library is ahead of both oracles at every size that is not
latency-bound.** At `n = 512` it is 73x `ndarray` and 8.2x `nalgebra`; at
`n = 1024`, 184x and 8.2x; at `n = 2048`, 225x and 7.2x. Nothing about that is a compromise on exactness: the
integer result is the exact sum encoded once, and `CX-01` .. `CX-04` and `CX-10`
assert it byte for byte against four independent implementations, including one
outside the Rust ecosystem.

**On integer squares it is now within 1.2x of `matrixmultiply`'s `f32`.** That
comparison is across element types and is not a like-for-like claim; it is here
because `matrixmultiply` is the fastest thing in the process and it says how much
of the machine the driver is reaching.

**On a matrix-vector product it is 10x ahead of `ndarray` and 15x ahead of
`matrixmultiply`.** That shape used to be this library's worst, at 0.70 Gmac/s,
and the reason was structural: the tile kernel produced sixteen columns for a
product that has one. The reduce factorization and panel borrowing are what
closed it, and both are the same identity, factored differently.

**Latency at `n = 1` is 140 ns against `ndarray`'s 60.** The difference is view
construction, walking the kernel table, and weighing the two traversals --- all
once per call and none of it scaling. Measured, the table walk is 20 ns per
family and there are two, which is the largest single piece. It is one of two
places an oracle is ahead, and the sweep says so.

**On floats the library is roughly 39x behind `matrixmultiply`.** It was 134x,
and calling that a trade was wrong: most of it was not the price of
exactness, it was a placement done once per product that can be done once per
reduction. See "The float placement" below. What remains is a real gap and it is
still the largest one in this document.

### The modular factorization

`i32` reaching 27 Gmac/s is not a `wrapping_mul` shortcut. When the caller asks
to encode by wrapping into a `w`-bit output, reduction modulo `2^w` is a ring
homomorphism, so accumulating in `Z/2^w` *is* the exact accumulation seen in the
quotient the caller named. `_mm256_mullo_epi32` then gives eight products per
instruction where the exact `i64` lane gives four, with no widening, because in
the quotient there is nothing to widen to.

Which factorization runs is decided by two *declarations* --- the encode mode
and the output width --- and never by inspecting the data. `CD-05` asserts that
both give the value their mode asks for, and that past `i32` they disagree,
which is what makes the choice observable rather than cosmetic.

### The reduce factorization

A tile kernel's vector lanes are columns of `C`. A reduce kernel's are steps of
`k`. Both compute the same integer, and `CB-06` asserts it; which one runs is
decided by the shape against the tile, which is a declaration the caller made
when they constructed the view.

It needs no second driver. A reduce panel is the *same* layout function at
`group = kpad`, so `packed_slot` reduces to `lane * kpad + p` and one `usize` in
the packer covers both. The table carries a reduce sequence for every family on
every ISA, each at a four-row and a one-row panel, because a panel wider than the
output is zero-padded and for a reduce kernel that padding is copied at `k`
elements a row.

### The narrow panel

`i8` at `n = 8` ran *slower* than the same kernel at `n = 16` --- 0.51 Gmac/s
against 4.05. That is not a property of the shape. The AVX2 `i8` tile is
`6 x 16`, so an output eight columns wide paid for sixteen: twice the arithmetic,
twice the packing, and a padded panel that fills half its lanes with zeros.

The table already had the answer on the other axis. `AVX2_I8_I32_M1` exists
because a panel *taller* than the output does the padding's arithmetic at `m = 1`;
`AVX2_I8_I32_N8` is the same argument on the columns. Measured, `8 x 8 x 8` went
from 1001 ns to 490 ns, a factor of two, and nothing above `n = 16` moved.

Three things had to be got right, and two of them were got wrong first.

**A width parameter costs the full-width panel its registers.** Folding both
widths into one kernel behind a `const V` --- an accumulator array indexed by the
parameter --- took `n = 2048` from 31.3 Gmac/s to 24.9. The array stops being
register-allocated and starts being memory. Two panel widths are two sequences,
which is what a table of sequences is for, and `CB-01` .. `CB-06` hold both to the
same integer. They are written out separately and the differential net sees both
--- `every_i8_tile()` chains the two lists, because a sequence outside the net is
a sequence nothing checks.

**The narrow panels are a second list, not more entries in the first.** Putting
them in `available_i8` lengthened *every* selection, including the calls that
could never use one: measured, a hundred nanoseconds on a shape whose whole cost
is a hundred and twenty. The driver only asks when `shape.n < tile.nr`, which is
the same condition under which it already asks for a reduce sequence, so a
product wider than its panel pays nothing for the question. What it does cost, on
a shape narrower than the tile that then does not want one, is one table walk:
20 ns, against the 511 ns it buys at `n = 8`. `n <= 4` and `n = 12` are 40 ns
worse and every size from 6 up is better, which is the trade and it is reported
rather than averaged away.

**The cost model priced a traversal the offer could not run.** `packed_cost`
counted the blocked traversal's packing --- each operand packed once per block ---
whatever the caller offered. But `run` takes the blocked traversal only when a
block of both panels fits the offer; below that it repacks *both* panels for every
output tile, which is a different count entirely. Narrowing the panel shrank
`mr + nr` enough for `4 x 4 x 4` to slip past the scratch guard into that
per-tile path, and the model --- still pricing the blocked one --- called it
cheaper than the streaming reference while it ran three and a half times slower.
Costing the traversal the offer actually buys fixed it, and it is the count that
was wrong rather than a constant that needed tuning: `calls * (mr + nr) * kpad`
against `(rows + cols) * kpad`.

### The float placement

A complete accumulator is exact because every product lands at *its own*
position in the register, and finding that position is a shift, a limb index and
a carry --- per product. Measured on a `4096`-deep panel, that placement was the
entire distance between what the arithmetic costs and what the loop cost:

| | ns per product |
| --- | --- |
| the mantissa product alone, into one `i64` | 0.43 |
| the shipped window, exponents inside one binade | 1.98 |
| the shipped window, exponents over ~60 binades | 2.49 |
| ten `i128` buckets with the carry deferred | 2.33 |

The bucket form is the natural alternative --- one accumulator per limb, nothing
ever flushed --- and it is worth 6%. Two independent scalar designs landing
within 6% of each other is what a structural ceiling looks like: the cost is the
placement itself, not the way the placement is organised.

So the placement has to leave the loop. Write `a * 2^(ea - base_a)` into the
panel instead of `(a, ea)`, and likewise for `b`, and

```text
  sum_p (a_p 2^(ea_p - base_a))(b_p 2^(eb_p - base_b))
    = 2^-(base_a + base_b) * sum_p a_p b_p 2^(ea_p + eb_p)
```

--- the float dot product *is* an integer dot product, at one known scale,
placed into the register once for the whole reduction. Nothing is approximated:
the scaled significands are exact integers and so is their sum. This is the same
move the modular factorization makes for integers, and it is legitimate for the
same kind of reason: an identity, not an approximation.

What it costs is width, which is what makes it a declaration rather than a mode.
A significand of `P` bits scaled across a span of `w` exponents needs `P + w`
bits; a product needs `2P + wa + wb`; a sum of `k` of them needs `ceil(log2 k)`
more. Three lanes follow, and every term in the choice is a count of bits taken
from the element type and the panels:

| when | the reduction is | measured |
| --- | --- | --- |
| `2P + wa + wb + ceil(log2 k) <= 62` | an `i64` dot product, which vectorizes | 0.50 ns |
| `... <= 126` | one wide multiply per product | 0.89 ns |
| otherwise | the per-product placement | 2.5 ns |

`CU-04` asserts all three byte-identical to the streaming traversal, over
operands chosen to reach each one, at every panel offer; a second test asserts
that the spans do select the lane they were chosen to select, because a
differential test over operands that all take one path is a test of one path.
Planting a one-exponent error in either scaled lane fails it.

Measured end to end, at `f32`:

| `m x k x n` | before | after | |
| --- | --- | --- | --- |
| `512` cubed | 0.325 | **1.103** | 3.4x |
| `1024` cubed | 0.341 | **1.118** | 3.3x |
| `509x1021x257` | 0.313 | **1.113** | 3.6x |
| `256` cubed | 0.346 | 1.043 | 3.0x |
| `32` cubed | 0.288 | 0.532 | 1.8x |
| `8x262144x8` | 0.307 | 0.471 | 1.5x |
| one exponent throughout, `512` cubed | 0.33 | **2.02** | 6.1x |

Two things about the shape of that.

**The spans are settled for the whole call, not per panel.** They have to be: the
scaling is written *into* the panels, so a block scaled for one row and read
unscaled by another would be read wrong --- which is what the first version did.
Deciding once costs one walk of each operand's exponents.

**That walk is not always worth taking.** It reads `(m + n) * k` codes and saves
one placement from each of `m * n * k` products, and a decode and a placement are
the same order of work, so it pays exactly when `m * n > m + n`. That is false
for a matrix-vector product, where it would more than double the whole call ---
`1x1048576x1` lost a third before the question was asked --- and true for
everything with two real dimensions. It is a comparison of counts, like the one
`pick` makes, and not a threshold.

What is left is the `39x`. The scaled lanes are scalar; the `i64` lane's inner
loop is a plain integer dot product, which is exactly what the AVX2 `i32` tile
kernel already computes at an order of magnitude more throughput. Reaching it
means expressing the scaled panels as an integer alphabet and handing them to the
kernel table, and that is not done: it wants an epilogue that can place a raw
`i128` accumulator at a scale, which the integer driver's epilogue contract does
not currently express.

### The declared alphabet

`_mm256_madd_epi16` sums two products into an `i32`, and two full-magnitude
`i16` products are `2 * 2^30` --- one bit past it. So that sequence is exact
exactly while the alphabet bound is at most `32767`, and at the full `i16`
alphabet it returned `-2^31` where the exact sum is `+2^31`. A random fill
reaches that input with probability `2^-64`, which is why the parity tests did
not find it and now ask for the extremes by name.

The fix is a declaration, not a guard: a `KernelSpec` carries `max_bound`, the
widest alphabet its sequence is exact on, and selection does not consider a
sequence outside it. The paired sequence declares `32767` and keeps its
instruction count; a widening sequence declares every `i16` and issues more.
Both are exact on what they declare, which is what makes them two factorizations
rather than a fast one and a safe one --- outside its alphabet the paired
sequence does not compute this identity at all.
