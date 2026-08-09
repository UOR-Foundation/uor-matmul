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
of 64 bits. An arbitrary `i64` scale adds 63 exponent-growth bits and the two
terms of `alpha * sum + beta * C` add one: the terminal expression needs 683
signed bits.

For `f64`: `2 * -1074 = -2148` and `2 * 1024 = 2048`, a span of `4196`; plus 64
plus 1 is `4261`; terminal scalar application raises that to 4325 bits.

The old three boolean non-finite flags already rounded to one aligned word.
That word is now the signed extension limb, with seven extreme values reserved
for every nonempty union of those flags; canonical NaN and the two signed
infinities occupy three of them. Physical storage is therefore unchanged at 88
bytes for `f32` and 544 for `f64`: 704 and 4352 bits respectively. The finite
tail needs only 43 and 37 signed bits, leaving all seven sentinels unreachable
by the model-derived terminal expression. The 544 bytes per output element is
the real cost of exactness at `f64`, and it is not hidden: a large
register-blocked tile is expensive on a small target. The mitigation is a
traversal choice, not a method choice.

## The tropical accumulator widths, and the term that is missing

The ring's derivation is `1 + MAX_K_BITS + 2 * (E::BITS - 1) + log2(PRODUCT_TERMS)`.
The `(max, +)` derivation is

```text
trop_acc_bits(E) = 1 + (E::BITS + 1)
```

and the whole content of this section is which terms are *absent* and why.

`MAX_K_BITS` is absent because a maximum does not grow with the reduction.
`max_p (a_p + b_p)` is bounded by `max_p a_p + max_p b_p` whatever `k` is, so
there is no depth for a depth term to be a function of. That is not a tighter
bound on the same quantity --- it is a different quantity, and `CA-04` is the
gate that says the derived width is the same number at depth one and at
`2^MAX_K_BITS`.

`2 * (E::BITS - 1)` is absent because `⊗` is addition, not multiplication. Two
magnitudes at most `2^(BITS-1)` multiply to `2^(2*BITS-2)` and *add* to at most
`2^BITS`. Representing that bound needs `BITS + 1` bits rather than `BITS` ---
one more than its logarithm, because the bound is attained --- and the sign
makes `BITS + 2`.

`log2(PRODUCT_TERMS)` is absent because there is no complex tropical element:
`max` on the complex numbers is not an order, and inventing one would be an
arbitrary choice this repository has nowhere to derive from.

| `E` | worst-case bits | accumulator |
| --- | --- | --- |
| `Trop<i8>` | 10 | `TropAcc<10>` |
| `Trop<i16>` | 18 | `TropAcc<18>` |
| `Trop<i32>` | 34 | `TropAcc<34>` |
| `Trop<i64>` | 66 | `TropAcc<66>` |

Ten bits against the ring's seventy-nine at the same element type. The register
is physically an `i128` in every case --- there is no narrower machine integer
worth the trouble, and `TropAcc` is `repr(transparent)` over it --- so the width
above is what the *derivation* answers and `Accumulator::BITS` reports, not the
bytes the register occupies. The distinction matters for exactly one reason:
`i128::MIN` is then unreachable as a finite accumulation at every element type,
which is what lets it name the semiring zero without excluding any value. That
is a derivation, and if it ever stopped holding the accumulator would need a
flag exactly as `Complete` needs one for its non-finite states.

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
- The historical `m = 1` six-row padding defect is closed by the registered
  one-row family entries. The selector reads their declared height, and CG-22
  pins their ordering so a wider equal-bound entry cannot shadow them. On the
  recorded host that removed wasted arithmetic without moving the decode-bound
  gemv clock; both facts are retained rather than treating a zero timing margin
  as permission to restore padding.
- A deep, thin shape --- `16 x 400000 x 16` --- sustains 5.9 Gmac/s where the
  microkernel alone runs at 38. The gap is the *pack*, and it is arithmetic
  rather than a tuning choice: the operands are 12.8 M elements and the product
  is 102.4 M multiply-accumulates, so every element copied into a panel serves
  eight of them. The 38 figure is measured on panels that are already packed. At
  a 16-wide output there is nothing to amortize the copy against --- an `A`
  element is used by 16 columns, a `B` element by 16 rows --- and no traversal
  reads fewer elements than the operands contain.

  This used to be recorded here as a blocking choice: that the depth-chunked
  traversal could not pick its panel shape without knowing its chunk depth, a
  circularity left unbroken. It is not that. Measured, best of seven passes, with
  the accumulator offer swept from the suggested 256 up to 2048 --- which is what
  moves the block extents --- the shape runs 5.88, 5.89, 5.88, 5.89, 5.94, 5.90
  Gmac/s. Flat. The block is not what is costing it, and it is worth saying so
  where the wrong reason used to be: a single pass differed from another by 40% on
  this host, which is how a 6% difference in a median came to look like a finding.

  Nor is it bandwidth: 12.8 MB in 17.4 ms is 0.74 GB/s, two orders below what the
  machine will do. It is the copy itself, at a shape with nothing to spread it
  over. `CG-08` prints it per pass.
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
in `uor-matmul-kernels`, because that attribute requires `unsafe` and
`uor-matmul-gemm` forbids it --- and the table's column loop and its build were
written in `uor-matmul-gemm`. Counted
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

The same sweep reports a `narrow block` column: the collapsed traversal with
the accumulator offer halved, so the column block resolves to half the output
width and the collapse that runs is `CD-16`'s block-local one or none.
Measured on an aarch64 dev machine --- not the runner the table above came
from, so the absolute figures are its own and only the shape transfers:

| `d` | collapsed | narrow block |
| --- | --- | --- |
| 1 | 62.62 | 46.11 |
| 8 | 56.70 | 43.85 |
| 64 | 53.12 | 44.39 |
| 512 | 50.64 | 34.70 |
| 4096 | 31.77 | 23.31 |

Two readings. The collapse survives narrowing: every degenerate row sits well
above the nothing-to-collapse row of the same column (`d = 4096`), where
before `CD-16` a halved offer disabled the collapse outright. And the drop
from the full-offer figure is the build, which does not collapse and now runs
once per column block --- halving the block doubles it, the same ceiling the
`d = 1` row of the main table already names.

One thing this cost to get right. Naming the columns instead of counting from a
first one turns the code base and the accumulator base from induction variables
into indexed loads, and measured that **halved every shape, including the ones
with nothing to collapse**. So there are two column loops: the consecutive one,
unchanged, and an indexed one reached only when the collapse has something to do.
A specialization in the direction that costs something is a second function, not a
parameter.

### What the missing index stream costs the sign composition

The sign tier is spelled as `Packed<Grid<2>,8>` (`CK-13`): a code space of 256
and a block of 8, `Book<256,8>`'s numbers exactly, with one gather-path
difference --- `Packed` cannot answer `as_index_stream`, because a packed
byte's index is a mixed-radix decomposition and not the byte. So the
composition builds its index stream where the book borrows the operand's own
memory. This is the price of that, measured on an Apple M4 Max (aarch64),
2026-07-27, `Traversal::Tabulated` at the full offer, each side asserted
against its own dense reference. Every figure is `open`.

`sign, Full` is the composition over `Full<i8>` with the general build, so its
ratio against the book isolates the gather. `sign, Bnd<1>` is the tier as it
stands: activations and weights both in `{-1,+1}`, the bound-1 build
admissible, and the census's build multiplies going from `m*k*256` to zero.

| `m x k x n` | `Book<256,8>` | sign, `Full` | sign, `Bnd<1>` | sign/book | b1/book | b1/full | build mul |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `1x1024x1024` | 15.32 | 12.15 | 4.38 | 0.79x | 0.29x | 0.36x | 262144 -> 0 |
| `1x1024x4096` | 20.70 | 13.51 | 8.27 | 0.65x | 0.40x | 0.61x | 262144 -> 0 |
| `1x4096x1024` | 11.68 | 8.62 | 3.30 | 0.74x | 0.28x | 0.38x | 1048576 -> 0 |
| `1x4096x4096` | 14.69 | 10.36 | 7.13 | 0.71x | 0.49x | 0.69x | 1048576 -> 0 |
| `4x1024x1024` | 38.00 | 28.94 | 14.25 | 0.76x | 0.38x | 0.49x | 1048576 -> 0 |
| `4x1024x4096` | 59.06 | 39.56 | 28.44 | 0.67x | 0.48x | 0.72x | 1048576 -> 0 |
| `4x4096x1024` | 41.11 | 29.88 | 13.74 | 0.73x | 0.33x | 0.46x | 4194304 -> 0 |
| `4x4096x4096` | 62.90 | 41.04 | 28.25 | 0.65x | 0.45x | 0.69x | 4194304 -> 0 |
| `16x1024x1024` | 60.03 | 55.69 | 34.88 | 0.93x | 0.58x | 0.63x | 4194304 -> 0 |
| `16x1024x4096` | 92.14 | 84.86 | 59.19 | 0.92x | 0.64x | 0.70x | 4194304 -> 0 |
| `16x4096x1024` | 66.02 | 58.55 | 36.25 | 0.89x | 0.55x | 0.62x | 16777216 -> 0 |
| `16x4096x4096` | 93.60 | 85.34 | 56.97 | 0.91x | 0.61x | 0.67x | 16777216 -> 0 |

Three runs of the sweep put the one-row ratios at 0.65--0.79 every time and the
sixteen-row ratios at 0.87--1.00, with one cell (`16x4096x4096`) swinging to
0.66 in a single run; the one-row and four-row figures are the stable ones.
The gather the composition cannot borrow costs it **a fifth to a third at a
one-row tile, about a quarter to a third at four rows, and under a tenth at
sixteen** --- real, and shrinking exactly as the build and the row count
amortize it.

That measurement is what demanded the dedicated `Sign` tier (`CK-11`): the
same decode with the `u16` code *being* the index, so the traversal borrows
the operand's own memory exactly as the book does. Re-measured on the same
host the day after (2026-07-28), same discipline, the composition column
reproduced and the tier column new. Every figure is `open`.

| `m x k x n` | `Book<256,8>` | sign, `Full` | `Sign<8>` | sign, `Bnd<1>` | sign/book | tier/book | b1/book | b1/full | build mul |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `1x1024x1024` | 14.99 | 10.99 | 14.89 | 4.57 | 0.73x | 0.99x | 0.30x | 0.42x | 262144 -> 0 |
| `1x1024x4096` | 20.59 | 13.40 | 19.98 | 8.79 | 0.65x | 0.97x | 0.43x | 0.66x | 262144 -> 0 |
| `1x4096x1024` | 11.86 | 8.46 | 11.41 | 3.40 | 0.71x | 0.96x | 0.29x | 0.40x | 1048576 -> 0 |
| `1x4096x4096` | 14.52 | 9.03 | 13.41 | 6.55 | 0.62x | 0.92x | 0.45x | 0.73x | 1048576 -> 0 |
| `4x1024x1024` | 37.62 | 27.88 | 37.62 | 14.28 | 0.74x | 1.00x | 0.38x | 0.51x | 1048576 -> 0 |
| `4x1024x4096` | 58.34 | 38.69 | 58.43 | 28.17 | 0.66x | 1.00x | 0.48x | 0.73x | 1048576 -> 0 |
| `4x4096x1024` | 41.00 | 27.92 | 39.39 | 14.70 | 0.68x | 0.96x | 0.36x | 0.53x | 4194304 -> 0 |
| `4x4096x4096` | 63.01 | 40.11 | 63.15 | 28.99 | 0.64x | 1.00x | 0.46x | 0.72x | 4194304 -> 0 |
| `16x1024x1024` | 60.39 | 55.51 | 64.26 | 37.49 | 0.92x | 1.06x | 0.62x | 0.68x | 4194304 -> 0 |
| `16x1024x4096` | 85.72 | 79.92 | 86.79 | 66.27 | 0.93x | 1.01x | 0.77x | 0.83x | 4194304 -> 0 |
| `16x4096x1024` | 62.70 | 58.92 | 64.32 | 39.54 | 0.94x | 1.03x | 0.63x | 0.67x | 16777216 -> 0 |
| `16x4096x4096` | 94.52 | 87.43 | 100.36 | 69.98 | 0.92x | 1.06x | 0.74x | 0.80x | 16777216 -> 0 |

The reading the tier was built to produce: **the gap is closed.** At one and
four rows the composition sits at 0.62--0.74 of the book while the tier sits
at 0.92--1.00 --- noise, against the fifth-to-a-third the composition pays.
At sixteen rows the tier edges the book itself (1.01--1.06), which is the
decodes differing and not the gathers: the tier's codebook is a bit test, the
book's is a 2048-byte copy of E8, and at that row count the build is the
amortized cost either way. What did not move, on that day, is the `Bnd<1>`
column, and it should not have: its cost was selection (no NEON bound-1 spec
on this host yet), not the gather, and the tier changes nothing about which
build is admissible.

The `Bnd<1>` column wanted a careful reading then, and the reading dated it.
Its build issues no multiply --- the census says so at every shape --- and it
was still the slowest column by a wide margin. The reason was selection, not
arithmetic: at bound 1 the only admissible spec on this host was the portable
reference, whose gathers are the reference's own, while the `Full` column runs
the NEON build and the NEON gathers. The adds-only build's win was real but it
was a scalar build against a vector one, and the gathers move with the spec.

The NEON bound-1 build now exists (a `neon_table_i8_i32_bound1` spec beside
the AVX2 one, same shapes, same gathers, `((a & keep) ^ sign) - sign` with the
masked negation computed in the `i16` lane and folded into the accumulation by
a widening add). The first spelling computed the masks in the `i32` lane and
measured *slower* than the reference it replaced --- 10.3 against 18.0 Gprod/s
at a sixteen-row tile --- because the autovectorizer already writes the
reference in NEON; the widening-add spelling is what overtook it (18.3 and
14.8 Gprod/s at sixteen and eight rows, against the reference's 15.7 and
11.2). Re-measured on the same host later the same day (2026-07-28), same
discipline, two runs; the table is the first. Every figure is `open`.

| `m x k x n` | `Book<256,8>` | sign, `Full` | `Sign<8>` | sign, `Bnd<1>` | sign/book | tier/book | b1/book | b1/full | build mul |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `1x1024x1024` | 14.75 | 10.98 | 14.99 | 4.31 | 0.74x | 1.02x | 0.29x | 0.39x | 262144 -> 0 |
| `1x1024x4096` | 20.73 | 13.42 | 19.91 | 8.28 | 0.65x | 0.96x | 0.40x | 0.62x | 262144 -> 0 |
| `1x4096x1024` | 11.28 | 8.34 | 11.36 | 3.27 | 0.74x | 1.01x | 0.29x | 0.39x | 1048576 -> 0 |
| `1x4096x4096` | 14.45 | 9.49 | 14.56 | 7.04 | 0.66x | 1.01x | 0.49x | 0.74x | 1048576 -> 0 |
| `4x1024x1024` | 37.45 | 28.65 | 37.56 | 13.36 | 0.77x | 1.00x | 0.36x | 0.47x | 1048576 -> 0 |
| `4x1024x4096` | 61.37 | 37.85 | 58.47 | 27.09 | 0.62x | 0.95x | 0.44x | 0.72x | 1048576 -> 0 |
| `4x4096x1024` | 41.12 | 29.58 | 39.21 | 13.23 | 0.72x | 0.95x | 0.32x | 0.45x | 4194304 -> 0 |
| `4x4096x4096` | 62.87 | 41.68 | 66.65 | 27.52 | 0.66x | 1.06x | 0.44x | 0.66x | 4194304 -> 0 |
| `16x1024x1024` | 58.75 | 56.00 | 60.17 | 37.40 | 0.95x | 1.02x | 0.64x | 0.67x | 4194304 -> 0 |
| `16x1024x4096` | 82.94 | 85.13 | 94.56 | 69.35 | 1.03x | 1.14x | 0.84x | 0.81x | 4194304 -> 0 |
| `16x4096x1024` | 61.31 | 58.49 | 66.14 | 40.03 | 0.95x | 1.08x | 0.65x | 0.68x | 16777216 -> 0 |
| `16x4096x4096` | 74.63 | 72.38 | 93.29 | 70.51 | 0.97x | 1.25x | 0.94x | 0.97x | 16777216 -> 0 |

The column now splits exactly where the spec does. At one and four rows
nothing moved, and nothing should have: the NEON table sequences spell tiles
of eight and sixteen rows --- the same coverage the full-alphabet spec has ---
so below eight rows the build is still the portable one and the column
reproduces (0.29--0.49 across both runs, against 0.29--0.48 before). At
sixteen rows the build is the NEON one and the column moved: `b1/full` is
0.67--0.97 where it was 0.62--0.83, with the second run putting the four
sixteen-row cells at 0.63--0.94. The widest cell is also the noisiest --- the
book column itself swung 74--101 across the two runs --- so the statement that
survives the noise is the build probe's above, measured directly, not any one
cell of this table.

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
asserts that declaration pair directly, so the claim is falsifiable without
depending on which ISA the test host happens to expose.

Both numbers are read off the two specs at run time, so a host is priced at its
own sequences. The recorded rows in `model/tiers.toml` carry the register width
they are written for, because a break-even without one is a number about nothing.

The rows are now recorded per pair, and one non-AVX2 pair is measured.
`model/tiers.toml` carries a row per enumerable codec per instruction set ---
the AVX2 pair first, then NEON (four lanes against `NEON_DOTPROD_I8_I32`'s
sixteen), AVX-512 VNNI (sixteen lanes against `vpdpbusd`'s sixty-four) and
wasm SIMD128 (four against eight) --- each recomputed by `CM-04` from the
declarations. The crossings move with the pair: at NEON, `Book<256,8>` pays
from `n = 2049` rather than 683, and the block-4 codecs never pay at all,
because one 128-bit table add covers exactly what one `sdot` covers. Measured
on an Apple M4 Max (aarch64, NEON with the dot-product extension), 2026-07-29,
two runs of the `tabulation_breakeven` example: the shipped predicate flips
from the kernels to the table between `n = 2048` and `n = 2049` at a
sixteen-row tile --- exactly the derived crossing --- and between 512 and 683
at a one-row tile, where no shipped ISA has a vector table sequence and the
reference gather's own 683 is the right number again. The clock tells its own
story: the tabulated traversal is ahead of the packed kernels at every width
measured (1.2x at `n = 512`, rising to about 3x at 8192), so on this host the
instruction count is the conservative claim, and the timing figures are `open`.
The VNNI and wasm rows are complete derived `build` declarations, not clock
claims. CM-04 recomputes them from the registered sequence densities; host
timings, when reported, remain separate `open` observations.

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

### The gate that never reported

`miri` is the gate that checks for undefined behaviour, and it had never once
finished. Every run in its history is `failure` --- the toolchain pin outranked
the nightly the job installed, so `cargo miri` resolved to stable and reported
that the component was unavailable --- or, once that was fixed, `cancelled` at
exactly six hours, which is GitHub's ceiling on a job.

Fixing the toolchain made it *run*, and running exposed what it was running.
The workflow's own header says `uor-matmul-kernels` is the only crate with
`unsafe` and that its portable module "is validated here". The job's command was

    cargo miri test -p uor-matmul-core -p uor-matmul-codec -p uor-matmul-gemm

--- three crates that each `#![forbid(unsafe_code)]`, and not the one that holds
every `unsafe` block in the workspace. So it spent six hours emulating code that
cannot contain the thing it was looking for, and then reported nothing at all.

Its whole history is four `failure` and four `cancelled`. Not one success.

Three changes, and the job finishes:

- It runs `uor-matmul-kernels` now. Under Miri no feature-detection predicate
  answers true, so the portable sequences are what execute --- which is exactly
  what the header always claimed was being validated.
- `uor-matmul-gemm` is out. It forbids `unsafe`, and its tests are whole
  matmuls; `CD-13` alone sweeps 189 offer pairs. Its soundness surface is the
  crates it calls, and those are in. `-core` and `-codec` stay, because both do
  `bytemuck` casts and slice arithmetic that provenance checking has something
  to say about.
- The heavy differential sweeps take a smaller corpus under `cfg!(miri)`.
  `CB-08` runs 4 spaces x 3 blocks x 5 rows x 5 groups natively; under Miri it
  runs two of each. That reduces the number of *instances*, not the set of
  *paths*: every arm of `dispatch_run!` and `dispatch_slab!` is one macro body
  at different constants, so a narrow arm and a wide arm exercise the code, and
  keeping a code space that is not a power of two keeps the padding claim.

And `timeout-minutes: 45`, so that a run which stops finishing *fails* rather
than occupying a runner for six hours and reporting `cancelled` --- which reads
like an infrastructure hiccup instead of the regression it would be.

### The partition, and the case it was missing

Fixing the list is not the same as making the list checkable. The reason a job
could point away from every `unsafe` block in the workspace and nobody notice is
that nothing anywhere asserted where it pointed. `CU-07` does:

> Every shipped crate either forbids `unsafe_code` outright or is a crate the
> Miri job runs.

Two cases, no third one. Either the compiler rules the question out, or Miri
answers it. The test reads the `-p` list out of `.github/workflows/miri.yml`
rather than restating it --- a restated list would agree with itself while the
job ran something else, which is the failure --- and it reads the same list out
of the `Justfile` and requires the two to be equal, so a local `just miri` and
the CI job cannot come to mean different things.

Writing it down found two more cases the corrected list still did not cover.

`uor-matmul-model` neither forbids `unsafe_code` nor is run under Miri, and it
had no `publish = false` --- so a crate the workspace lint table already calls
"build-time and CI infrastructure", whose own module doc says it "is not a
dependency of any shipped crate", was set to ship to crates.io at `0.1.0`, model
reader and `serde` and `toml` and all. Its two dependents are `xtask` and
`uor-matmul-conformance`, both `publish = false`; nothing a consumer of
`uor-matmul` builds reaches it, because R10's generated consts are committed
rather than produced at build time. It is `publish = false` now, which is what
the other two infrastructure crates always said.

`uor-matmul` is the harder one. The facade cannot `forbid(unsafe_code)` and says
so in a comment: it carries the raw-pointer face (`CS-05`), whose contract is a
caller obligation. It was not in the Miri list --- and adding it would have run
nothing, because the crate had no tests at all. What tests it had lived in
`uor-matmul-validate`, which is infrastructure the Miri job does not run, and
they called `sgemm` with `k as isize` and `1`.

Both positive. So of the two functions in `raw.rs` --- `low`, which finds the
lowest offset a strided view reaches, and `span`, which counts elements from
there --- `low` returned zero on every call ever made of it, and `span` returned
`rows * cols`. The negative-stride arithmetic those two functions exist for had
never been executed, by any test, on any target.

`crates/uor-matmul/tests/raw.rs` executes it: every combination of stride sign,
including column-major and padded, over four shapes, against the safe API on the
same window --- 84 calls, plus the degenerate extents and the `f64`
monomorphisation. Each buffer is allocated at *exactly* the reachable span and
the caller's base pointer is `-lo` elements into it, so an off-by-one in either
direction leaves the window rather than reading a neighbour. Unreached cells hold
`NaN`, which an exact sum propagates, so a stray read *inside* the buffer is a
`NaN` in the output rather than a plausible number.

The two directions fail differently, and that difference is the argument for the
Miri job existing at all. A window one element too *short* fails natively, in all
three tests. A window one element too *long* passes the native run --- and is
Undefined Behaviour under Miri, reported as a dangling reference going beyond the
bounds of its allocation. Both measured, by planting them.

The job now runs `-p uor-matmul-kernels -p uor-matmul -p uor-matmul-core -p
uor-matmul-codec`, and `CU-07` holds it there. Locally, on two cores: 25m56s, 63
tests, exit 0. The kernels' parity suite is 1166s of that, the codec suite 236s,
`-core` 24s.

In CI, on `ba64fcf`: **18m07s, success**. That is the first conclusion this gate
has reported. Its nine previous runs are four `failure` --- 30s to 7m, all of them
the toolchain pin --- and five `cancelled`, each of which sat on a runner for
exactly six hours and then reported nothing at all:

| commit | conclusion | wall clock |
| --- | --- | --- |
| `ba64fcf` | `success` | 18m07s |
| `655dc38` | `cancelled` | 6h00m |
| `80204e8` | `cancelled` | 6h00m |
| `3d36fea` | `cancelled` | 6h00m |
| `b081b51` | `cancelled` | 6h00m |
| `02eaf46` .. `f2637bf` | `failure` x4 | 30s .. 7m |

Eighteen minutes against a 45-minute ceiling, so a regression that doubled the
cost would still report, and one that grew tenfold would *fail* rather than be
cancelled --- which was the whole point of setting the ceiling in the first place.

## Against the oracles

> The float rows in this oracle sweep predate the pure-UOR Atlas refactor and
> are historical baselines. Integer rows retain their original reading. "The
> pure-UOR float Atlas (current)" below describes the live architecture; the
> current measurement section of `MEASUREMENT-LOG.md` records the completed
> post-native `CG-16` candidate rejection and distinguishes `CG-21`'s
> historical timing baseline from the completed current-source sweep.

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
reduction. See "The float placement" below. The next section, "The float
placement bridge", measures what closed most of what remained on the
development host: the same placement, handed to the kernel table.

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

### The pure-UOR float Atlas (current)

The float operation between IEEE decode and the single encode is now entirely
the declared UOR census. The canonical reference takes the unique finite
non-adjacent Laurent section of `Z[X, X^-1]/(X - 2)`: valuation is grade, sign
is the modality involution, and Euclidean mixed-radix addressing maps every
signed grade across `(word, scope=4, context=8)` without wrap. Its carrier,
four projectors, and minimum-gauge theorem are executable formal certificates.
They deliberately do not pretend to be objects reached by the optimized call
graph.

Execution removes the same valuation and projects the remaining odd coefficient
directly into balanced radix-256 coordinates derived from the signed-`i8`
lookup alphabet. The quotient step repeats until the coefficient is zero:
precision determines the number of repetitions, never a representation arm.
For each reduction position the one direct factorization contracts all occupied
coordinate pairs on their `u+v` Laurent diagonal, using only signed-octet table
lookup and addition. Each diagonal enters one bounded three-limb carrier for the
mathematical source product. After all diagonals have arrived, the carrier
resolves its sign and magnitude once. One Euclidean fracture in the
signed-place radix `i128::MAX + 1` places its low digit at the product grade and
its possible unit high digit at the radix-successor grade. `CD-30`
differentially pins those bytes to the canonical finite-NAF reference.

The evaluated common-gauge and packed-support selectors do not ship. Their
scalar masks, population counts, and second projection pass added work and
regressed the measured grid; keeping them as a conditional route would retain
both a lesser implementation and a classical bookkeeping mechanism. The only
remaining choice *inside the dense Atlas factorization* is among eligible
group-one tile, narrow, and reduce lookup/add orientations, globally priced by
model-derived executed work. It does not decide whether the coded driver admits
block-one tabulation. That internal price
includes each declaration's contraction work, exact output-cell residency after
the fixed Atlas workspace is charged to L1, live-only product-carrier
initialization, and full tiles plus every edge orientation; it has no tuned
threshold. One exact frame owns all live cells of each tile for the complete
reduction, and the model-generated contiguous capacity dispatch instantiates
its exact live extent rather than a manually maintained maximum frame. Offered
`PackedCode` words become balanced-octet/grade projections in place, so reuse
covers projection as well as decode with no second buffer. Streamed cells decode
once into the six-state boundary quotient and finite payload; reused coordinate
words clear only their retired suffix. Panels and the fixed source workspace are
reused across arbitrary reduction depth. Thus neither empty workspace nor wider
exponent support causes whole-operand reification, allocation, a traditional
multiply, a scalar support mask, or a reserve arithmetic path (`CK-19`,
`CK-20`, `CD-31`, `CA-05`, `CU-11`, `CG-22`).

The final current-source 2026-08-09 `CG-16` release instrument fitted a value-blind
block-one table candidate on 28 unique structural calibration envelopes and
held out 12 block-one identities plus block-three and block-five scalar-fracture
controls. The candidate routed 9 of the 12 block-one holdouts to the table and 3
to the coded Atlas decline. H01 and H02 were deliberately unlike-valued twins
with exactly the same pre-admission `StructuralWork`, so `CS-10` required the
candidate to route them alike. Their paired table/decline ratios were instead
`0.1821 +/- 0.0397` and `2.9348 +/- 0.3215`: the table decisively won one and
decisively lost the other. The same-key holdout therefore falsified the fitted
candidate. No value-dependent repair is admissible, no fit coefficient became
a model constant, and the current value-blind block-one default remains the
decline. A caller may still force the exact table factorization.

The logged 2026-08-06 `CG-21` run on the shared x86-64 host is the historical
pre-one-frame baseline. It exposed the decisive defect at full finite range:
`f64 7x31x5` took 65--71 ms. During the refactor, an intermediate local run after
the direct diagonal factorization, bounded selector scan, and removal of
redundant kernel-lane clears observed `1.010 +/- 0.024` / `1.517 +/- 0.673` ms
for the same offered/no-offer `f64` calls; `f32` observed
`0.913 +/- 0.181` / `1.621 +/- 0.376` ms. It also observed few-grades `f32 32^3`
rates of `0.0179 +/- 0.0055` / `0.0221 +/- 0.0024` Gproduct/s and reported exact
bytes plus unchanged integer and tropical controls.

Those intermediate numbers have no retained command artifact and are not the
authoritative result. The final current-source UOR-NAF cleanup sweep does: its source
manifest is identical before and after the run, it retains every raw sample,
and the live harness poisons before and completely checks after each calibrated
batch, leaving only production calls inside the timer. The current full
finite-range calls are `221.912 +/- 23.949` / `176.396 +/- 16.811` us for f32
offered/no offer and `481.324 +/- 60.387` / `554.315 +/- 63.761` us for f64.
The f32 intervals are faster than the preceding source-pinned run despite no
reachable x86 production change, so the record does not attribute that host
fluctuation to the implementation. These figures remain
`open`; the build claims are exact bytes and census correspondence over the
complete eligible group-one universe, not a host-independent nanosecond rate,
an optimum outside that universe, or a new selector constant.

### The float placement (historical, superseded)

> Historical measurement record. The bridge/scalar implementation described
> from here through "The float placement bridge" has been removed. Its numeric
> evidence is preserved verbatim and must not be read as the current call graph
> or current float performance. The section above supersedes its call graph;
> `CG-21` records which timing evidence predates the replacement.

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
kernel table. That is the float placement bridge, and it is the next section.

### The float placement bridge (historical, superseded)

The scaled panels are exact integers, so the reduction over them is an integer
dot product at one known scale, `2^-(base_a + base_b)`. The bridge reifies the
scaled panels as the `i32` alphabet they already are, hands them to the kernel
table, and places the table's exact `i128` sum into the float accumulator at
that scale --- through `Complete::add_scaled`, the decode's own primitive, so
nothing is rounded on the way in. The scale channel is a placement epilogue,
`Scaled`, not a parameter on the `Epilogue` trait: the scale is a fact of one
call's panels, and a wrapping epilogue carries it without touching the
contract every other epilogue implements. `CD-19` asserts the bytes are the
streaming traversal's at every shape and every offer; a span that does not fit
the `i32` alphabet --- more than seven binades at a 24-bit significand, any
`f64` --- takes the scalar scaled lanes, and a wider one the per-product
placement, exactly as before.

**The prediction, written before the measurement.** A 24-bit significand
forces products into the 64-bit lane, and AVX2 offers four `i64` lanes against
eight `f32` FMA lanes across two units, so on the x86 runner the realistic
ceiling is single-digit Gmac/s: roughly 4--7x over the 1.10--1.12 the scalar
lanes post at 512--1024 cubed, the gap to `matrixmultiply` closing to
something like 6--10x, not to parity. On this host --- an Apple M4 Max,
aarch64, not the x86 runner the rest of this document's figures come from ---
the `i32`-exact family has no hand-written NEON sequence: the table's entries
for it are the portable kernel and the two AVX2 ones, so the bridge runs the
portable kernel at the aarch64 baseline and the figure is whatever LLVM's
auto-vectorizer makes of a portable `i32 x i32 -> i64` loop. That predicts a
smaller factor here: measurably above the scalar lanes at the two cubes if the
auto-vectorizer finds the widening multiply-accumulate, a wash if it does not
--- the portable loop issues the same per-product multiply the scalar lane
does, and the bridge adds a decode and a pack on top. What the machine
actually said is below.

**The measurement.** Every figure in the tables is `open`: measured on an
Apple M4 Max (dev machine, aarch64-apple-darwin), 2026-07-30, by `just
bridge-sweep` (`CG-15`), best of a 0.35 s budget per point, with byte-identity
against the scalar lanes asserted inside every timed run. The baseline figures
quoted from elsewhere in this document are an x86 runner's; the scalar column
here is the same code measured on this host, and it is three to four times the
x86 runner's number before the bridge does anything --- the two machines'
figures are not interchangeable, which is why the sweep remeasures the
baseline it compares against. Gmac/s:

| fill | `m x k x n` | scalar | bridged | | `matrixmultiply` |
| --- | --- | --- | --- | --- | --- |
| one exponent | `512` cubed | 4.326 | **13.082** | 3.0x | 61.277 |
| one exponent | `1024` cubed | 4.505 | **15.309** | 3.4x | 58.589 |
| one exponent | `509x1021x257` | 4.421 | **13.580** | 3.1x | 58.461 |
| one exponent | `256` cubed | 3.918 | 9.643 | 2.5x | 60.659 |
| one exponent | `32` cubed | 1.615 | 2.280 | 1.4x | 52.429 |
| a few binades (3/4) | `512` cubed | 4.255 | **11.964** | 2.8x | 60.529 |
| a few binades (3/4) | `1024` cubed | 4.039 | **14.233** | 3.5x | 58.440 |
| a few binades (3/4) | `509x1021x257` | 3.994 | **12.822** | 3.2x | 58.714 |
| wide spans (18/22) | `512` cubed | 3.945 | 3.900 | 1.0x | 62.096 |
| wide spans (18/22) | `1024` cubed | 4.016 | 4.036 | 1.0x | 58.109 |

Read against the prediction. The direction was right and the size was not, in
both halves. The auto-vectorizer did find the widening multiply-accumulate ---
the portable `i32`-exact kernel runs at 13--15 Gmac/s where the scalar lane
posts 4.0--4.5, so the bridge buys 3.0--3.5x at the two cubes, below the
4--7x the x86-oriented prediction said, because the four-lanes-against-eight
arithmetic is an AVX2 sentence and this host's family entry is a portable loop
the compiler vectorized. And the gap to `matrixmultiply` closed further than
predicted --- to 3.8x at `1024` cubed, from 13x --- because the prediction
priced the oracle from the x86 runner's figure and this host's `sgemm` posts
58--61, not more. The wide-span rows are the boundary, reported rather than
smoothed over: past seven binades the scaled significand is not an `i32`, the
bridge declines, and the two columns are the same walk with the span
question's price on top --- about 1%, the walk being `(m + n) * k` packs
against `m * k * n` products.

Two more honest edges. `f64` never crosses the bridge: a 53-bit significand
is not an `i32` at any span, and the `i64`-element family whose lane is an
`i128` has no SIMD multiply on any target this library supports, so the scalar
scaled lanes are `f64`'s answer until a wider alphabet earns a family. And the
declared bound is the wider of the two panels, so an asymmetric span --- seven
binades on one side, none on the other --- declares `2^31` and the lane depth
collapses to one product: exact, chunked by the table's own machinery, and no
faster than the scalar lane it replaced. That case is the alphabet's edge, not
its center; the center is a significand and a few binades, and it is measured
above.

**The bridge is the default path now.** As first shipped, the bridge sat
behind its own entry point and the default float driver never took it ---
`cargo bench` timed the scalar lanes, which is the measurement that forced
the issue. The selection is the library's usual doctrine: a `PackedCode`
panel offer re-reads as four `i32` words a code (the layout's padding word is
named, so the re-read is a safe cast, not a transmute), and the packed entry
takes the table when the offer holds the reified operands plus a full-depth
kernel panel pair, the spans admit the `i32` alphabet, and the declared lane
holds the whole depth. The last term is the one the first draft lacked, and
the sweep caught it: past the lane's depth the table chunks, the chunked
traversal's partial sums want an accumulator offer a panel buffer cannot
spell (eight-byte aligned against `i128`'s sixteen), and the per-tile chunk
traversal it falls to measured 2.5 Gmac/s at `512` cubed against the scalar
lanes' 4.3. So a deep reduction declines to the scalar lanes by default and
reaches the table through the explicit entry, whose offers can spell the
accumulator room. Every figure here is `open`, measured on an Apple M4 Max
(dev machine, aarch64-apple-darwin), 2026-07-30, `cargo bench -p
uor-matmul-validate -- gemm_f32`, before against after, criterion means:
`16` cubed 4.130 us against 3.890 (a wash, kept because the doctrine forbids
a regression there, and no floor constant entered the model for it); `128`
cubed 697.3 us against 338.9 (2.06x); `64x512x1024` 9.788 ms against 3.812
(2.57x). The remaining distance to the explicit entry is the accumulator
room its offer can spell --- the chunked traversal and the sub-cubic level
both live there --- and `suggested_float_panels` is the query a caller asks
for the offer that admits every factorization the shape supports.

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

## Float tabulation at a tiny code space

> Historical pre-Atlas experiment. The numeric measurements are retained as
> evidence of what that prototype did; its scalar/placement operation model is
> not part of the current pure-UOR implementation.

Measured 2026-07-29 on an Apple M4 Max (aarch64), release build, best of a
0.30 s budget per cell. Every figure in this section is `open`. The instrument
is `FloatBook<S, BLK>` in `crates/uor-matmul-validate/src/float_tab.rs` ---
dev-only scaffolding, not a tier: `S` codewords of `BLK` `f32` symbols each,
`Code = u16`, `S` a power of two so the stored code *is* the index and
`as_index_stream` borrows the operand's memory. Nothing here adds a model row,
an ID, or shipped code; the measurement decides whether building the tier
properly (R9) is worth starting.

The question exists because the only shipped float codec, the arena tier, is
`MAX_BLOCK = 1`, which `tabulation_pays` refuses outright: the float table
route had never carried a measurement. The structural argument said it should
not pay --- the integer table's win is the narrow lane, and a float has none:
`AccOf<f32>` is 88 bytes on this host and the lane *is* the complete
accumulator. The counter-argument is that at `S` of 4 or 16 the slab
(`S * rows * 88` bytes, 5.6 KB to 22.5 KB at the widest row tile) sits in L1,
and the column loop trades one exact float mac per product for one table read
and one exact accumulator combine per `BLK` products.

### The op costs

One primitive per iteration over `k = 4096`, same budget. `mac` is
`Element::mac` for `f32` --- the encode-to-fixed-point plus placement the
dense float driver issues once per product. `gather+add` is the column loop's
step: read one 88-byte table entry by code, `combine` it into the running
accumulator, covering `BLK` products.

| op | per step | per product covered |
| --- | --- | --- |
| gather+add, `S = 4`, `BLK = 4` | 4.19 ns | 1.048 ns |
| gather+add, `S = 4`, `BLK = 8` | 4.07 ns | 0.509 ns |
| gather+add, `S = 16`, `BLK = 4` | 4.19 ns | 1.048 ns |
| gather+add, `S = 16`, `BLK = 8` | 4.15 ns | 0.519 ns |
| exact float mac | 3.42 ns | 3.42 ns |

Two readings. The combine and the mac cost about the same per *step* --- both
are an 88-byte multi-limb exact add, and the gather's L1 read is hidden under
it --- so the table's whole advantage is the block: the same step amortized
over `BLK` products is 3.3x cheaper at `BLK = 4` and 6.7x at `BLK = 8`. And
`S` is invisible in these numbers: 4 and 16 entries both sit in L1, so the
code space costs nothing until it doesn't fit. The build is `m * S * k`
products once per column block against a column loop of `m * n * k / BLK`
combines; at `n >= 1024` and `S <= 16` it is under 2% and does not appear in
what follows.

### End to end

Gmac/s against the nominal `m * k * n`, three routes over the same operands.
`stream` is `gemm_float` over the decoded weights with no panel offer, which
is exactly what the tabulated driver's dense decline route runs for `f32`
(`Tabulated::dense_gemm` calls it with the leftover offer unused). `packed` is
the same driver with panels --- the placement bridge, both panels prescaled to
a common base so the inner loop is an integer dot product --- offered
`m * k` and `k * min(n, 512)` codes. `table` is `Traversal::Tabulated` forced
at the offer `suggested_tabulation` sizes; every cell is asserted
byte-identical (`symbol_bits`) to the streaming driver's output, and the
census asserts the table ran (`table_reads = m*n*k/BLK`, `adds` likewise).
`picked` is the default `Traversal::Blocked`'s own choice at the same offer,
read from the census.

#### `FloatBook<4, 4>` --- slab 5632 bytes at the widest tile

| `m x k x n` | stream | packed | table | vs stream | vs packed | picked |
| --- | --- | --- | --- | --- | --- | --- |
| `1x1024x1024` | 0.113 | 0.484 | 0.975 | 8.60x | 2.01x | table |
| `1x1024x4096` | 0.086 | 0.309 | 0.985 | 11.51x | 3.19x | table |
| `1x4096x1024` | 0.107 | 0.256 | 0.932 | 8.68x | 3.64x | table |
| `1x4096x4096` | 0.045 | 0.125 | 0.932 | 20.53x | 7.44x | table |
| `4x1024x1024` | 0.109 | 0.799 | 1.080 | 9.89x | 1.35x | table |
| `4x1024x4096` | 0.086 | 0.544 | 1.118 | 12.94x | 2.05x | table |
| `4x4096x1024` | 0.105 | 0.481 | 0.935 | 8.94x | 1.95x | table |
| `4x4096x4096` | 0.057 | 0.215 | 1.010 | 17.78x | 4.69x | table |
| `16x1024x1024` | 0.092 | 2.156 | 1.112 | 12.03x | 0.52x | table |
| `16x1024x4096` | 0.073 | 1.533 | 1.145 | 15.63x | 0.75x | table |
| `16x4096x1024` | 0.097 | 1.389 | 1.061 | 10.91x | 0.76x | table |
| `16x4096x4096` | 0.055 | 0.922 | 1.093 | 19.95x | 1.19x | table |

#### `FloatBook<4, 8>` --- slab 5632 bytes at the widest tile

| `m x k x n` | stream | packed | table | vs stream | vs packed | picked |
| --- | --- | --- | --- | --- | --- | --- |
| `1x1024x1024` | 0.111 | 0.468 | 1.908 | 17.16x | 4.08x | table |
| `1x1024x4096` | 0.089 | 0.321 | 1.996 | 22.55x | 6.22x | table |
| `1x4096x1024` | 0.109 | 0.343 | 1.912 | 17.56x | 5.58x | table |
| `1x4096x4096` | 0.057 | 0.151 | 1.949 | 34.15x | 12.88x | table |
| `4x1024x1024` | 0.113 | 1.113 | 2.139 | 19.00x | 1.92x | table |
| `4x1024x4096` | 0.089 | 0.584 | 2.143 | 24.10x | 3.67x | table |
| `4x4096x1024` | 0.109 | 0.711 | 2.147 | 19.75x | 3.02x | table |
| `4x4096x4096` | 0.057 | 0.298 | 2.224 | 38.99x | 7.46x | table |
| `16x1024x1024` | 0.113 | 2.424 | 2.187 | 19.35x | 0.90x | table |
| `16x1024x4096` | 0.089 | 1.626 | 2.300 | 25.89x | 1.41x | table |
| `16x4096x1024` | 0.109 | 1.794 | 2.181 | 20.05x | 1.22x | table |
| `16x4096x4096` | 0.057 | 0.961 | 2.315 | 40.52x | 2.41x | table |

#### `FloatBook<16, 4>` --- slab 22528 bytes at the widest tile

| `m x k x n` | stream | packed | table | vs stream | vs packed | picked |
| --- | --- | --- | --- | --- | --- | --- |
| `1x1024x1024` | 0.113 | 0.484 | 0.887 | 7.82x | 1.83x | table |
| `1x1024x4096` | 0.090 | 0.344 | 0.931 | 10.39x | 2.71x | table |
| `1x4096x1024` | 0.110 | 0.361 | 0.884 | 8.04x | 2.45x | table |
| `1x4096x4096` | 0.057 | 0.152 | 0.949 | 16.64x | 6.24x | table |
| `4x1024x1024` | 0.112 | 1.107 | 0.875 | 7.79x | 0.79x | table |
| `4x1024x4096` | 0.089 | 0.594 | 0.948 | 10.71x | 1.60x | table |
| `4x4096x1024` | 0.109 | 0.693 | 0.845 | 7.77x | 1.22x | table |
| `4x4096x4096` | 0.057 | 0.298 | 0.930 | 16.34x | 3.12x | table |
| `16x1024x1024` | 0.111 | 2.443 | 0.931 | 8.40x | 0.38x | table |
| `16x1024x4096` | 0.087 | 1.571 | 1.040 | 11.91x | 0.66x | table |
| `16x4096x1024` | 0.108 | 1.826 | 0.926 | 8.60x | 0.51x | table |
| `16x4096x4096` | 0.057 | 0.970 | 1.029 | 18.01x | 1.06x | table |

#### `FloatBook<16, 8>` --- slab 22528 bytes at the widest tile

| `m x k x n` | stream | packed | table | vs stream | vs packed | picked |
| --- | --- | --- | --- | --- | --- | --- |
| `1x1024x1024` | 0.114 | 0.481 | 1.595 | 13.98x | 3.32x | table |
| `1x1024x4096` | 0.088 | 0.343 | 1.817 | 20.54x | 5.30x | table |
| `1x4096x1024` | 0.110 | 0.357 | 1.593 | 14.49x | 4.47x | table |
| `1x4096x4096` | 0.057 | 0.153 | 1.818 | 31.79x | 11.90x | table |
| `4x1024x1024` | 0.114 | 1.089 | 1.579 | 13.88x | 1.45x | table |
| `4x1024x4096` | 0.089 | 0.595 | 1.791 | 20.08x | 3.01x | table |
| `4x4096x1024` | 0.108 | 0.715 | 1.537 | 14.19x | 2.15x | table |
| `4x4096x4096` | 0.057 | 0.298 | 1.817 | 31.77x | 6.09x | table |
| `16x1024x1024` | 0.112 | 2.375 | 1.668 | 14.94x | 0.70x | table |
| `16x1024x4096` | 0.089 | 1.623 | 2.006 | 22.62x | 1.24x | table |
| `16x4096x1024` | 0.109 | 1.816 | 1.658 | 15.23x | 0.91x | table |
| `16x4096x4096` | 0.057 | 0.976 | 1.992 | 34.77x | 2.04x | table |

### The reading

**The structural argument is wrong at tiny code spaces, and the numbers say
why.** The float lane is the complete accumulator, so there is no narrow-word
win --- but the table's win here was never the lane width. It is the *block*:
one exact accumulator combine per `BLK` products instead of one exact float
mac per product, and the two steps cost the same (op table above). The
premise's L1-fit argument holds exactly as stated: `S` of 4 and 16 are
indistinguishable in throughput, and the column loop runs at a flat ~0.9--1.15
Gmac/s at `BLK = 4` and ~1.5--2.3 at `BLK = 8` whatever the shape --- the
signature of a loop bound by the 88-byte combine at ~4.1 ns each, nothing
else.

**Against the route the driver would actually decline to, the table wins
everywhere measured:** 7.8x to 40.5x against `stream`, at all 48 cells. The
dense factorization `Tabulated::dense_gemm` names for `f32` is the streaming
traversal, and against it there is no contested region at all.

**Against the placement bridge the win is regional.** `packed` prescales both
panels and turns the inner loop into an integer dot product; it is the real
competitor, and the regions are:

- `m = 1`: the table wins at every cell, 1.8x--12.9x. The bridge's own
  costing declines the prescaling walk at `m * n <= m + n`, so a single row
  gets neither the walk nor the lane it buys --- and the table's amortization
  is over `n`, which needs no rows.
- `m = 4`: the table wins at `BLK = 8` everywhere (1.45x--7.46x); at `BLK = 4`
  it wins except the smallest cell (`0.79x` at `4x1024x1024`, `S = 16`).
- `m = 16`: the bridge wins the narrow cells (`0.38x`--`0.91x` at `n = 1024`,
  and `16x1024x4096` at `BLK = 4`), the table wins the wide ones, and the
  cross sits near `n * (BLK - 1) ≈ k`: at a wide tile the build's `m * S * k`
  products are repeated per 512-column packed block on the dense side but only
  amortized once per output on the table side.

**The predicate already admits all of this.** `picked` is `table` at all 48
cells under the default `Traversal::Blocked`: with `block > 1`,
`tabulation_pays`'s terms --- a table step of `BLK` products against the
float `dense_steps` declaration of one product per step, and the L1 fit ---
admit every shape tried. That cuts both ways, and the way that matters for a
decision is the negative one: the predicate prices the table against the
*streaming* declaration (`dense: 1`), so at `m = 16, n = 1024` it selects the
table where the placement bridge is 1.1x--2.6x faster. The dense side of the
float comparison is declared by the weakest float route, not the strongest.

### What this does not say

- The packed column was offered `k * min(n, 512)` panel codes; a caller
  offering the full `k * n` re-packs `B` less often at `n = 4096`, so the
  `packed` figures at that width are, if anything, understated.
- The codebook here is 4--16 arbitrary symbols chosen for the experiment. A
  real tier's codebook is the artifact's own distinct blocks. This experiment
  prices the traversal and makes no compression or artifact-frequency claim;
  N3 classifies such quality observations per artifact rather than as a library
  capability.
- Byte-identity held at all 48 cells (`symbol_bits` against the streaming
  driver, with an infinity-free, NaN-free fill). The codec itself is total by
  the same mask argument the shipped codecs use; the traversal's totality over
  non-finite symbols is the arena tier's already-shipped claim, unchanged.

The decision-relevant summary: at `BLK = 8` the table beats the *best* dense
float route by 1.2x--12.9x at `m <= 4` and by 1.2x--2.4x at `m = 16` with wide
outputs, and loses only at tall-`m`, narrow-`n` shapes (down to 0.38x) ---
with the caveat that the selection predicate currently cannot see that region,
because it prices the dense side at the streaming declaration.

A note on the fill, added when the scaled lane landed (`CD-20`): the figures
above were measured with a fill spanning about ten binades, which the scaled
lane's admission (`24 + span <= 31`) declines by design. The instrument's
fills now stay inside one five-binade band and its offer carries the scaled
lane's `i64` words (`suggested_tabulation_lanes`), so the forced table runs
under the merged semantics; the measured economics above are the wide-lane
tree's, and re-running the instrument today prices the scaled lane instead.

## The SWAR broadcast, measured and declined

Every figure in this section is `open`: measured on one host under one runtime
and reported, never asserted. `CG-17` is the claim these numbers belong to;
`CB-12` pins the bytes.

Kronecker substitution --- packing several small integers into one machine
word's bit fields so one multiply produces several products --- is a plain
identity over the integers, available to any library, and nothing about it is
exactness-gated: disjoint fields cannot carry into each other inside their
guard bits, and that is a fact of arithmetic, not of this library. OpenBLAS
declines the trick for throughput reasons, not numerical ones. The question
here was narrower: on baseline wasm SIMD128 there is `i64x2.mul` and no
byte-width dot product --- relaxed-simd's `i8x16.dot_i8x16_i7x8_s` is
specified non-deterministic, its intermediate precision
implementation-defined, so this library cannot use it regardless of
availability --- and the incumbent is `i32x4.dot_i16x8_s` at eight products an
instruction but paying two extends per sixteen bytes of operand. A
six-products-per-multiply form with no extends is a plausible win, and cheap
to measure. It was measured.

The form is the broadcast one, because it is the only one: multiplying two
packed vectors convolves their fields, so one side must be scalar. Pack three
elements of a `B` row at 21-bit spacing in each 64-bit lane --- two lanes to
a `v128`, six columns a register --- and multiply by one splatted `A` scalar.
Both operands are biased to unsigned first, the `dpbusd` offset identity
applied on both sides, so a product reaches `255 * 255` (sixteen bits) and
five guard bits a field remain; the compensation
`sum(a*b) = sum(a'*b') - 128*sum(a') - 128*sum(b') + 16384*k` is paid in exact
integers at extraction. The guard bits absorb `floor((2^21 - 1) / (255*255)) =
32` products a field, which is the packed accumulator's chunk; extraction and
compensation run once a chunk, and the driver's lane capacity composes on top
unchanged --- declare it, as the chunk, rather than inventing a second
extraction mechanism. The spacing is the one choice: sixteen bits is the
product width exactly and leaves no guard bit, so every product would be
extracted; twenty-four fits only two fields a lane; twenty-one is the widest
spacing that still holds three, and it is the one the numeral derives. The
bytes are pinned against the portable reference at every packed depth, at the
alphabet's extremes, and at the W4A8 bound (`CB-12`), including the extremes
fill that drives a field to its guard bits inside one chunk.

Measured on an Apple M4 Max (dev machine, aarch64-apple-darwin) under
wasmtime 45.0.0, 2026-07-30, `just swar-sweep`, padded-panel Mmac/s in a hot
loop --- the ratio is the finding, the absolutes are the harness's --- with
byte-identity asserted inside every timed run:

| k | fill | portable (4x4) | dot (4x8) | swar (4x12) | swar / dot |
| --- | --- | --- | --- | --- | --- |
| 64 | full i8 | 13141.0 | 20289.9 | 7615.9 | 0.38x |
| 64 | W4A8 (bound 7) | 13139.3 | 19855.7 | 7566.4 | 0.38x |
| 1024 | full i8 | 18423.8 | 26859.7 | 8707.9 | 0.32x |
| 1024 | W4A8 (bound 7) | 18424.7 | 26861.5 | 8729.0 | 0.32x |
| 16384 | full i8 | 18907.5 | 27449.3 | 8776.2 | 0.32x |
| 16384 | W4A8 (bound 7) | 18907.4 | 27456.7 | 8787.1 | 0.32x |

The reading, and it is the one the instruction count predicts. The incumbent's
widening is one extend per sixteen bytes of `B` and its dot is eight products;
the broadcast form pays eleven instructions per six columns per step to spread
bytes into fields the ISA has no instruction for --- a shuffle, three
mask-and-shifts, two ors, the bias add and mask --- and the multiply that buys
(six products against eight, no extends) never recovers it. The loss grows
with depth because the dot kernel's per-step cost is already amortized while
the pack is paid on every step. The narrower alphabet does not rescue the
form: at the W4A8 bound a `+-7` bias fits the biased product in eight bits,
so six ten-bit fields fit a lane --- but the guard bits shrink to two, the
chunk to five products, and the extraction the pack was amortized against
returns every fifth step. The field arithmetic is exact in every case and the
parity net says so; the loss is throughput, which is OpenBLAS's reason for
declining the trick, arrived at here independently and on a different ISA.

So selection declines it, and the decline is the kind of fact this repository
keeps as a gate: the sequence is not in any family's availability list,
because a listed sequence is one `Auto` may select and selecting this one
would be a measured 2.6--3.1x regression on every wasm panel shape. The spec
stays exported --- a caller who knows its shape can name it --- and `CB-12`
asserts both halves: the bytes, where the sequence can run, and the absence
from every list, so the decline cannot quietly revert.

**The x86 half began as a census reading and ended as a measured decline.** The
`CG-11` census reads the AVX2 `i8` tile (`avx2_i8_inner::<6>`) bound on
`Zn4FP2` --- 85 instructions, 14.5 cycles a tile-step, IPC 5.88, all of it
`llvm-mca` scheduling-model prediction on `znver4` and reported as such. The
vector pipes are the binding resource in the model and the scalar ALU ports
are not, which is exactly the condition under which a scalar Kronecker stream
co-issued into the same accumulator might have bought a single-digit gain.
The dedicated x86 experiment retained later in the measurement log tested the
co-issued pair against the vector kernel alone and rejected it across the
complete grid. No x86 sequence was added, and the measurement question is
closed rather than left as an unavailable-host obligation.

**The Cortex-M half is analysis, for two different reasons on the two
families.** The repository's embedded target is thumbv7em-none-eabihf
(Cortex-M4/M7), which has the DSP extension: `SMLAD` is two 16-bit
multiply-accumulates in one instruction, the multiplexing already in silicon,
so a Kronecker sequence gains nothing there and none is registered --- the
same rule that keeps one off x86. thumbv6m (M0/M0+/M23) is where the trick
would matter: no SIMD, one 32-bit multiplier, and on the base M0 that
multiply is the slow iterative one. There the honest arithmetic is two biased
bytes at sixteen-bit spacing in a `u32` --- no guard bits, so extraction on
every multiply --- against two scalar MACs, and the pack and extract dominate
exactly as they did on wasm; the plausible outcome is a wash. But the
decisive fact is the declared backend set: the workspace's embedded execution
target is thumbv7em, not thumbv6m. A sequence no registered executor runs would
be an unchecked claim, so no thumbv6m sequence is a capability of this backend
family. The portable implementation remains total there; this paragraph records
why the Kronecker sequence was not registered, rather than an implementation
item outside the declared target set.

## The symbol path against the bus

> Historical pre-Atlas measurement. Residency figures remain valid for the
> recorded artifact, while every description of scalar exact accumulation or
> placement as the shipped float body is superseded by "The pure-UOR float
> Atlas (current)" above.

Every figure in this section is `open`: measured on one host and reported,
never asserted. `CG-14` is the claim these numbers belong to.

The arena tier at a `u8` code width stores one byte per weight where the
dense float driver reads four, over the same exact accumulation --- `CK-14`
and `CD-18` pin the bytes, so what is left to ask is the economics. The
harness (`just symbol-bandwidth`) measures gemv and skinny GEMM shapes,
charges each path its operand bytes --- `A` plus the stored weights plus `C`,
the 1 KiB codebook on the symbol side --- and takes the host's STREAM number
in the same process: a triad `a[i] = b[i] + 3*c[i]` over 3 x 2^25 `f32`
(384 MiB, past every cache this host has), best of ten, 12 bytes counted per
element, with the write-allocate read stated and not counted. Byte-identity
with the dense float driver is asserted inside every timed run, and the
census printed per shape records which factorization ran. On an Apple M4 Max
(dev machine, aarch64-apple-darwin), 2026-07-30, STREAM measured 134.19 GB/s
and the sweep read:

| `m x k x n` | W stored (B) | sym walk GB/s | sym panel GB/s | uor `f32` GB/s | matrixmultiply GB/s |
| --- | --- | --- | --- | --- | --- |
| 1024x1024x1 | 1024 | 0.97 (1%) | 1.01 (1%) | 0.95 (1%) | 17.19 (13%) |
| 1x1024x1024 | 1048576 | 0.17 (0%) | 0.16 (0%) | 0.94 (1%) | 17.86 (13%) |
| 1x1048576x1 | 1048576 | 0.84 (1%) | 0.77 (1%) | 1.65 (1%) | 3.24 (2%) |
| 2048x8x2048 | 16384 | 0.08 (0%) | 0.08 (0%) | 0.28 (0%) | 8.34 (6%) |
| 8x262144x8 | 2097152 | 0.10 (0%) | 0.10 (0%) | 1.37 (1%) | 21.38 (16%) |

The reading is the recorded outcome. Every exact
path sits near 1% of the bus; the inexact oracle sits at 13-16% of it. The
symbol path's fourfold residency advantage is real --- the census counts one
decode per stored byte, and the stored-weights column is a quarter of the
dense spelling's --- and buys nothing at these shapes, because nothing here
is waiting on the bus: at 0.08-0.25 Gmac/s the traversal is bound by the
scalar exact accumulation, one product per step with a decode and a placement
per element, and the dense driver reading four times the bytes posts the same
figure for the same reason. Decode-and-place latency, not bandwidth, is the
bottleneck at O(1) arithmetic intensity, which prices the rest of the phase's
queue: the items that can move these rows are the ones that attack products
per step --- the float placement bridge and the narrow lane it unblocks ---
and a narrower stream is not among the levers left.

### The symbol table in the scaled lane

The live `CD-20` body is demand-built. At block one, when the stored stream
addresses fewer symbols than the declared enumeration, the span walk, decoded
book, and table build visit those addressed symbols rather than materializing
the unused alphabet. Each table entry is the contextual `Scaled64` q
contraction. Its lookup/add work follows the occupied balanced-octet extent;
the dedicated q observer verifies that variable work, while the generic table
Census deliberately reports one opaque contraction presentation per product
instead of fabricating a fixed operation weight. The public panel offer holds
the book and activation tile; complete activation rows in its remaining tail
become their projected cells in place. That zero-copy cache projects those rows
once per call, while rows the tail does not hold are projected once per column
block. The exact-sized offer is the same protocol with zero cached rows, not
another route.

The public `f64` lane remains `Wide<Complete>`, but it is executable rather than
a categorical refusal: forced `Arena<8>` block-one tabulation performs nonzero
table reads and matches dense Atlas byte for byte. An independently declared
block-two `f64` codec exercises automatic admission through those same public
declarations. `CD-20` exercises both formats; its Census observes actual span,
decode, projection, demand-build, gather, and table-add presentations, while
the q observer owns the value-dependent occupied-extent work.

> The remainder of this section is a historical `CG-16` measurement. It priced
> the former full-codebook `f32` build against the removed placement bridge.
> Its numbers remain `open` evidence for that revision, not performance claims
> for the demand-built implementation.

Every figure in this section is `open`: measured on one host and reported,
never asserted. `CG-16` is the claim these numbers belong to.

The previous section priced the lever: at O(1) arithmetic intensity the
symbol path is bound by per-element work, so what moves it is products per
step. The scaled lane is the bridge's identity with the table doing the
reduction. The slab arithmetic is the whole reason it exists: a table entry
is `rows` lane words, and at `S = 256` codes the slab is

```text
  256 * rows * 88 bytes  =  88 KiB at rows = 4   (the complete accumulator)
  256 * rows * 8  bytes  =   8 KiB at rows = 4   (the scaled lane)
```

--- the 88 KiB slab fits no L1 at any tile, which is what `tabulation_fits`
answered and what made even the forced traversal decline; the 8 KiB slab
holds at `rows = 8` (32 KiB against this host's 32 KiB budget term, the
factor of two inside `tabulation_fits` included). The lane's inputs are the
codebook and the activation tile pre-scaled to the panels' measured base
exponents --- the bridge's span walk, over `A` and over the *codebook*, which
is `W`'s whole alphabet materialized: `m * k + 256` decodes, never `n * k`
--- and the walk is asked only after the table is selected, so a call the
predicate declines never pays it. Admission is the bridge's own declaration
(`24 + span <= 31`, finite codes only); the lane's run depth is derived from
the per-side spans, `2^63 / 2^(48 + wa + wb)`, because the lane holds a
product of one element of each panel and is not bound by the kernel table's
one-alphabet interface. At the corpus codebook (seven binades, the
alphabet's edge) against a one-binade `A` the run is 127 products; against a
one-exponent `A`, 255.

Measured on an Apple M4 Max (dev machine, aarch64-apple-darwin),
2026-07-30, `just symbol-tabulated`, Gmac/s, best of a 0.35 s budget per
point, byte-identity with the dense float driver asserted inside every timed
run, the census confirming the table ran; STREAM in the same harness
measured 135.10 GB/s:

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

The historical reading, and it is not the one the work order priced. The table won
where the build's `code_space` products per reduction element amortize over
a wide `n` *and* the runs stay deep: 1.8--2.2x over the dense driver at
`1x1024x1024` --- a gemv the bridge's `m * n > m + n` question refuses to
walk --- and 1.2--1.6x at the tabulation-sweep shapes, where it also posts
2.8x over the bridge, whose per-call reification of the `k x n` operand
amortized over `m = 64` rows rather than a cube's thousand. It lost
everywhere the amortization is absent, and the losses are not marginal:
175x at `1024x1024x1`, where the build issues 256 products per reduction
element to serve one output column; 44x at `8x262144x8`, where the census
counts 537 million build products against 17 million gathered; and a
0.76--0.78x wash at `2048x8x2048`, where `k = 8` leaves one placement per
output and nothing for the table to share. The span narrows it further:
the fill's three exponents --- a span of two --- against the seven-binade
book cut the run from 255 to 63, and `64x1024x4096` falls from a 1.27x win
to a 0.89x loss.

At that revision the predicate stayed as it was. `tabulation_pays` counted
*instructions*: one table read against one dense product, which at `block = 1`
was never a win. The table's win was an op-*kind* difference --- a read and an add against a decode
and a placement, with the build's price and the run's depth as terms ---
and expressing it would have taken per-op-kind costs, which are measured
constants of one host, into a declaration-only predicate. The asymmetry in
that artifact was the disposition: the winning
margin is under 2.2x and the losing region is 40--175x, so a selection rule
that guesses the boundary wrong is catastrophic where a missing one is
merely leaving 1.2--2.2x on shapes the caller knows. The table is reachable
under `Traversal::Tabulated` for exactly that caller, and `CD-20` pins its
bytes. The wide-fill rows were
the boundary the other way: past seven binades the lane declines by the
bridge's own declaration and the dense route answers, one span walk (read
off the census's decode count) poorer. Demand building and the zero-copy
projection cache change precisely those counts, so this historical clock does
not establish the live boundary. The 2026-08-09 `CG-16` paired instrument
tested a live value-blind candidate and rejected it: H01 and H02 had the same
structural key and candidate table route, yet their unlike values produced
decisive table/decline ratios of `0.1821 +/- 0.0397` and
`2.9348 +/- 0.3215`. Forced tabulation remains available; automatic block-one
selection remains the decline.

## The sub-cubic recursion, on integers

Every figure in this section is `open`: measured on one host and reported,
never asserted. `CG-12` is the claim the scaling numbers belong to; `CD-21`
pins the bytes and is `build`.

The standing objection to everything above is that the library's wins come
from data that is quantized or structured: a narrower alphabet, a codebook, a
collapsed row. The sub-cubic recursion is the answer that needs no structure
at all. Winograd's form of Strassen's algorithm regroups one product into
seven products of half the extent plus eighteen block additions, and applied
for `L` levels it does `(7/8)^L` of the products. Over the integers the
regrouping uses only add, subtract, and multiply, so the regrouped sum is the
same integer the naive loop returns, bit for bit. A float library declines
Strassen because intermediate cancellation degrades its norm bounds with
depth; there is no norm here and nothing degrades, which is why the
`CD-*` byte-equality discipline covers the recursion with no new argument ---
exactly as it covers tabulation --- and why no classical `sgemm` can make the
same claim.

### Why the i32 lane, and the bound bookkeeping

The recursion runs where the block sums stay inside the element type.
Winograd's cross-term sums are four block terms at worst
(`S4 = A12 - A21 - A22 + A11`), so a level taken at operand bound `B` forms
sums of magnitude at most `4B`, and those sums are the next level's operands.
At full `i8` range one level already leaves the alphabet (`254 > 127`), so
the `i8` lane has zero free levels --- staying in `i8` would take a declared
bound of 63 for one level, which is quantization and is excluded. On the
`i32` lane each level costs two bits of a thirty-one that bounded data is not
using: at a declared `2^24` --- the bound the float placement bridge's scaled
alphabet already declares --- three levels fit (`4^3 * 2^24 = 2^30`), and the
lane's chunking is exact at any depth, so a shallow lane at a grown bound is
a few percent of rate and not a correctness question (the figures below
include it). At the full `i32` alphabet there are zero
free levels, the same zero `i8` has: a sum of two full-range values is not an
`i32`, and the plan says so and declines. The accumulator is not a constraint
at any admitted depth either: a product temporary at level `l` is bounded by
`9 * 4^2l * B^2 * k / 2^l`, which the headroom rule keeps under `9/16` of the
worst case the accumulator's width was derived against. The `i16` lane's
analysis is recorded rather than implemented: its per-operation penalty is
about two, so break-even sits near five levels, and `4^5 * B <= 2^15` admits
no useful bound.

The one sign in the combination is folded into a sum temporary (`T4` is
`B21 - T2` where the textbook writes `T2 - B21`), so the accumulator only
ever adds: there is no subtraction in the accumulator width to define, and
the byte argument stays one sentence. The sums live in the panel offer, as
bare elements --- they outgrow the declared bound by construction, and the
grown bound travels as a value to the kernel boundary, which is where the
alphabet hypothesis is discharged (the bridge's measured bound is the same
kind of declaration). The seven products live in the accumulator offer, in
the accumulator's own width, and no epilogue runs on one: the encode step
runs exactly once, on the combination, at the recursion's top.

### The plan, and what declines

The level count is a pure function of declarations, never of data: the
shape's evenness (odd extents decline rather than pad --- padding would be
exact, `CK-03`'s precedent, but it would materialize a padded copy of both
operands to buy a shape one halving, and declining keeps the mechanism one
code path), the bound's headroom above, the offer (a level whose sums or
product temporaries the offer cannot hold is declined, `CD-10`'s rule one
traversal up), and the measured crossover `strassen_min_extent` in the
`[blocking]` table. An explicit `gemm_strassen(t, e, o, s, levels)` request
is capped only by the exactness rules --- a caller declaring levels is in
`Traversal::Tabulated`'s position and knows its shape. `CD-21` asserts the
bytes against the `CX-01` wrapping oracle and `ndarray` at every corpus size,
every requested level count, and every offer including none, so the plan
decides which instructions run and nothing else.

### The measurement

Measured on an Apple M4 Max (dev machine, aarch64-apple-darwin), 2026-07-30,
`just strassen-sweep`, nominal Gmac/s (`m*k*n` per second) on random dense
`i32` at a declared `2^24` bound, seed `20260730`, best of a 0.35 s budget
per point, byte-identity against the cubic walk at the same encode asserted
inside every timed run. Two baselines, because the library has two cubic
walks a caller can mean: the modular lane, which a wrapping `i32 -> i32`
call selects by default, and the exact lane the recursion factorizes, which
a saturating call, a wider output, or the float bridge runs. The recursion's
columns are measured under `EncodeMode::Saturating`, so the comparison is
within one lane:

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

The fitted exponents, `time = c * x^e`, geometric spacing, nine samples each,
1.96 standard errors: exact cubic against MAC count `1.0171 +/- 0.0111`;
recursion at the auto-selected levels against MAC count `0.9679 +/- 0.0031`;
exact cubic against `n` `3.0514 +/- 0.0334`; recursion against `n`
`2.9037 +/- 0.0092`, whose interval `[2.894, 2.913]` excludes `3.0`. The
fastest sustained product rate this library reaches on this host, measured
in the same harness (the `i8` lane at `4096` cubed), is 63.1 Gmac/s; the
machine's peak arithmetic throughput is the bar the queue set, because a
nominal rate above it would be impossible for any implementation performing
`Theta(m*k*n)` products. The recursion's nominal rate does not cross that
line at any measured size, and the exponent says why it does not need to:
the win is a smaller exponent, not a bigger constant.

### The honest reading

The recursion beats the lane it factorizes from `n = 768` (+6%) and the
margin grows with size: +18--24% at `1536--2048`, +36% at `3072`, +49% at
`4096`, with the fitted exponent's interval excluding 3.0. That is the
sub-cubic claim on arbitrary random dense input, and it needs no structure
in the data.

Two boundaries, stated rather than smoothed. A wrapping `i32 -> i32` caller
is not served: that call's default is the modular lane, which reads
`Z/2^32` off the encode declaration and issues eight products per instruction
where the exact lane's widening issues four --- on this host 22--30 Gmac/s
against the exact lane's 16--19, and the recursion only matches the modular
lane at `4096`. The recursion serves the exact arm: saturating encodes,
wider outputs, and the float placement bridge, whose scaled alphabet is the
`2^24` bound the sweep declares. On `f32` the composition is measured by
re-running `just bridge-sweep` with the recursion in place: one exponent at
`1024` cubed reads 16.239 Gmac/s where the same harness recorded 15.309 the
day before (+6%, the one level the threshold admits there), `512` cubed is
unchanged at 13.105 (the threshold declines), and the wide-span fills decline
exactly as before. And the x86 economics the queue priced --- the `i32` lane
at 1.29x per operation, two levels to break even --- are a different
machine's story: on this host the exact lane's penalty against the modular
lane is about 1.5x per operation, the crossover sits at `n = 768` rather
than the predicted 1024--2048. These are host-scoped `open` figures; no x86
clock is part of the shipped build claim or an outstanding capability.

Two harness defects the measurement itself found, both recorded in
VERIFICATION.md's falsifiability table: the first baseline column was the
modular lane in exact clothing (`EncodeMode::Wrapping` selects it, so the
recursion was being priced against a lane it does not factorize), and the
base case streamed per tile at deep levels because the plan reserved a
shape-only accumulator suggestion that answers zero whenever `k <= KC` ---
while the grown bounds make the base case's *lane* shallower than its `k`,
which only an accumulator block repairs. The second is why the L=3 column at
`2048` read 8.2 Gmac/s in the first sweep and 20.9 in this one.

## Mantissa slicing, and the RNS beside it --- the analysis, recorded

> Historical design analysis, superseded as an implementation direction by the
> canonical recursive Atlas word. It is retained to preserve the rejected
> cost derivation, not as deferred work or a description of shipped arithmetic.

This section is a derivation, not a measurement: no figure in it is a claim
about a machine, and nothing here is implemented. It is recorded so that the
arithmetic is not rediscovered, and its precondition --- the placement bridge
and the SWAR investigation, both landed above --- is exactly what makes the
arithmetic short.

The idea: a 24-bit `f32` significand is three 8-bit slices. A product of two
significands is nine slice-products, each an exact integer at a known shift,
so nine integer GEMMs over the sliced operands --- each byte-identical to the
dense integer path it factorizes, because slicing is a regrouping of the same
products --- and a recombination that is a shift and an add into the wide
accumulator. No error analysis arises anywhere: the slices are integers, the
shifts are exact, and the sum is the same sum. This is the placement bridge's
identity read one level down, and like the bridge it is a factorization, not
a method.

The count says where it pays. Nine passes at the `i8` lane's rate against one
pass at the `i64` lane's: on baseline AVX2 the `i8` tile runs about twice the
`i64` lane's products per instruction, so nine passes at `38.5` read as about
`4.3` effective against the `i64` lane --- a wash, and a wash is a decline.
The condition that flips it is a large narrow-to-wide throughput ratio: VNNI
issues 64 `i8` MACs per instruction against the `i64` lane's four, and
`64 / 9 ~ 7` beats `4` by about 1.8x. NEON's dotprod and the `i8` tensor
units are the same shape. Where the ratio is two, slicing is dead; where it
is eight or sixteen, slicing is the widest exact lane the machine has.

Two apparent compositions were considered and are rejected with the slicing
scheme. Feeding the slices to SWAR or symbol tabulation would still require
nine materialized operand passes and nine traditional integer GEMMs. That is a
second arithmetic route and loses the direct Atlas projection's zero-copy,
precision-directed recurrence; a favourable instruction-width ratio cannot
repair the semantic and traffic regression.

The adjacent idea, recorded so it is not rediscovered: full RNS --- several
coprime moduli, CRT reconstruction. Covering the `i8` lane's 79-bit worst
case takes about six 16-bit channels, giving about `6 / (2 * 3)` ~ 1.3
effective MACs per instruction against the `i64` lane's four, or eleven
8-bit channels for VNNI at about `64 / 11 ~ 5.8`. Marginal where it is
available at all, and it turns favourable only under the same condition
mantissa slicing needs --- a large narrow-to-wide ratio --- where slicing
wins on simplicity: nine passes and a shift, against a basis conversion, a
CRT reconstruction, and a carry argument the slicing never has to make.

The verdict is closed: neither mantissa slicing nor RNS is part of the float
implementation. The direct balanced-octet Atlas contraction already exposes
the narrow lookup alphabet without materializing slices, changing arithmetic
families, or adding reconstruction machinery. This section records why the
two legacy decompositions were declined; it names no future implementation
work.
