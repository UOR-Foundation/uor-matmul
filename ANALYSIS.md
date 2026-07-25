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

Measured on a two-core shared runner with AVX2 and no AVX-512, at `i8 x i8 ->
i32` under `EncodeMode::Wrapping`. Every figure is `open`.

| Path | Gmac/s |
| --- | --- |
| generic driver, 256^3 | 0.93 |
| kernel-driven, 256^3 | 2.09 |
| coded (`Grid<16>`), 256^3 | 0.67 |
| `f32`, 128^3 | 0.08 |

Four things dominated, and each was a copy or a traversal rather than
arithmetic:

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

**The `B` panel was packed once per output block.** The panels are the only
copies the kernel driver makes, and with the row block outermost `B` was
repacked `m/mr` times over: `m*n*k/mr` element copies against `m*n*k` of
arithmetic, about a sixth of the work. Hoisting the `B` pack out of the row loop
--- column block outermost, whole depth in one chunk --- replaced that with
`n*k` copies and left `A` repacked `m*n*k/nr` times, which is the cheaper way
round because `nr > mr`. Worth 27%.

What remains, and why it is not a defect:

- `A` is still repacked per column block, at `m*n*k/nr` ~ 6% of the arithmetic.
  Removing it needs a `C` accumulator panel of `m x nr` exact accumulators,
  which is memory the library is not allowed to own (R7). A caller who wants it
  can partition and reuse.
- The generic and coded drivers walk `k` innermost, so `B` is read with stride
  `n`. Fixing that needs an accumulator row, which is again memory the library
  cannot own. The kernel-driven path is the one that packs, and it is 2.2x
  faster for exactly this reason.
- `f32` is far slower than the integer paths. The size of that gap is measured
  against the *oracles* below, not against our own integer path, because
  comparing a library to itself says nothing about whether the cost is
  reasonable.

## Against the oracles

C3 is a hard constraint: scaling is compared against the oracle's scaling. Both
sides are measured in one process, over one sweep spanning nine orders of
magnitude in MAC count, with the answer asserted inside the timed region --- a
speed measured on the wrong bytes is not a measurement.

Two-core shared runner with AVX2 and no AVX-512. Best of N per point; every
figure is `open`.

### Throughput, Gmac/s

| `n` | uor `i8` | uor `i32` | ndarray `i32` | nalgebra `i32` | uor `f32` exact | matrixmultiply `f32` |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 1.38 | 1.60 | 1.00 | 1.51 | 0.08 | 4.27 |
| 32 | 7.42 | 7.82 | 1.72 | 4.13 | 0.20 | 25.4 |
| 128 | 11.2 | 11.2 | 0.70 | 4.22 | 0.31 | 28.4 |
| 512 | 12.9 | 13.1 | 0.38 | 3.86 | 0.36 | 35.9 |
| 1024 | 12.5 | 12.3 | 0.21 | 4.68 | 0.36 | 42.5 |

### Latency at `n = 1`, nanoseconds per call

| | ns |
| --- | --- |
| uor `i8` | 100 |
| ndarray | 69 |
| matrixmultiply | 89 |

### Fitted exponent against MAC count, whole sweep

| | exponent |
| --- | --- |
| uor `i8` | 0.68 |
| uor `i32` | 0.69 |
| ndarray | 0.92 |
| nalgebra | 0.77 |
| uor `f32` | 0.82 |
| matrixmultiply | 0.64 |

An exponent below 1 means throughput still improving with size: the small end is
latency-dominated. `ndarray`'s 0.92 is the opposite story --- it starts fast and
*degrades*, from 1.9 Gmac/s at `n = 64` to 0.21 at `n = 1024`, which is a cache
story rather than an arithmetic one.

### What this says

**On integers, this library is ahead of both oracles.** At `n = 1024` it is 59x
`ndarray` and 2.7x `nalgebra`, and it holds a flat 12--13 Gmac/s from `n = 128`
upward while `ndarray` falls away. Nothing about that is a compromise on
exactness: the integer result is the exact sum encoded once, and `CX-01` .. `CX-04`
and `CX-10` assert it byte for byte against four independent implementations,
including one outside the Rust ecosystem.

**Latency at `n = 1` is 100 ns against `ndarray`'s 69.** The difference is the
view construction and the kernel selection, both of which happen once per call
and neither of which scales. It is the one place an oracle is ahead and the sweep
says so.

**On floats the library is 120x behind `matrixmultiply`, and that is the trade
N4 names.** A classical `sgemm` issues one fused multiply-add per element. This
one decodes two IEEE bit patterns, multiplies their significands as integers,
and places the exact product into a 619-bit fixed-point register --- and never
rounds until the end. Counting the scalar operations that requires against one
FMA over eight lanes on two ports gives a floor near 100x, and the measurement
sits just above it. The gap is the price of an answer that does not depend on
the order of the additions, which no figure in the `matrixmultiply` column has.

### What it took to get here

Four structural defects, each found by measuring rather than by reading:

**The SIMD kernels were never selected.** `uor-matmul-validate` depended on
`uor-matmul` with default features, so `std` was off, runtime detection fell
back to `cfg!(target_feature = ...)`, and every earlier figure was the portable
kernel while claiming to be the machine. Enabling `std` was worth 5.8x on `i8`.
The sweep now prints which kernels a build can run, first, so this cannot recur
silently. `README.md` says the same thing where a user will see it.

**Only `i8` had kernels.** `i32` went through the generic driver, which is the
same identity but not the same instructions --- and benchmarking it against
`nalgebra` was benchmarking the wrong program. Every integer family now has a
kernel: `i16` and `i32` exact, `i16`, `i32`, and `i64` modular, and `i64` exact
in an `i128` lane, which is the only width that holds an `i64` product and the
reason no SIMD reaches it.

**The packing loop indexed instead of walking.** `origin + i * rs + j * cs` per
element is two multiplies where a walk is one add, and the panel is a million
elements. Adding `MatView::row_walk` and `column_walk` and splitting the full
tile from the edge was worth 1.7x on `i32`.

**The accumulator was passed by value.** `Element::mac` took and returned
`Self::Acc` --- 88 bytes in and out, twice per product for a float. `&mut` was
worth 4x on `f32` and 1.6x on the packed integer path.

### The modular factorization

`i32` reaching 12 Gmac/s is not a `wrapping_mul` shortcut. When the caller asks
to encode by wrapping into a `w`-bit output, reduction modulo `2^w` is a ring
homomorphism, so accumulating in `Z/2^w` *is* the exact accumulation seen in the
quotient the caller named. `_mm256_mullo_epi32` then gives eight products per
instruction where the exact `i64` lane gives four, with no widening, because in
the quotient there is nothing to widen to.

Which factorization runs is decided by two *declarations* --- the encode mode
and the output width --- and never by inspecting the data. `CD-05` asserts that
both give the value their mode asks for, and that past `i32` they disagree,
which is what makes the choice observable rather than cosmetic.
