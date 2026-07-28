# ARCHITECTURE

Normative. Where this document and the code disagree, one of them is a bug.

## The one sentence

> Decode the code, accumulate exactly, encode once.

Every entry point is that sentence at a different instantiation. The parameters
are `(Element, Bound, Codec, MaxBlock, Backend, Traversal)`, and W8A8 is the
instantiation `(i8, 127, Identity, 1, *, *)`. Nothing privileges it beyond its
having the most instruction support and the most external oracles.

## The crates, and why they are separate

| Crate | Contains | `unsafe` | `alloc` | float arithmetic |
| --- | --- | --- | --- | --- |
| `uor-matmul-core` | alphabet, accumulator, reference accumulation, views, the error surface | forbidden | none | none |
| `uor-matmul-codec` | the `Codec` trait, seven tiers, `CodedMatrix`, the kappa manifest, the E8 table | forbidden | none | none |
| `uor-matmul-kernels` | one module per ISA, each a `KernelSpec` value | permitted, documented | none | none |
| `uor-matmul-gemm` | the driver: traversals, scratch, epilogue, partition | none | none | none |
| `uor-matmul` | the facade and the raw-pointer face | the raw face only | none | none |
| `uor-matmul-model` | the typed model and the generators | none | freely | n/a |
| `uor-matmul-validate` | oracle adapters, corpus, scaling harness | documented | freely | n/a |
| `uor-matmul-conformance` | the BDD runner and the honesty meta-gate | none | freely | n/a |

The split is not organizational. `uor-matmul-kernels` is separate because it is
where `#[target_feature]` is written, which requires `unsafe`; the three
numerical crates above it `#![forbid(unsafe_code)]`, so the boundary is a thing a
compiler checks rather than a thing a reviewer remembers. The facade's raw face
is the one other place `unsafe` appears, and it appears there because a caller's
pointer carries no shape to check --- `CU-07` asserts that both of them, and only
they, are what the Miri job runs. `uor-matmul-validate` is separate because the
oracles must not be reachable from a shipped crate, and a dependency edge is the
only way to enforce that.

## The accumulator model

The worst-case magnitude of an integer accumulation is `k * B_a * B_w`. Both
bounds are properties of the element type; `k` is bounded by
`MAX_K_BITS`, which is **declared** in `model/constants.toml` rather than probed
from the host. So:

```text
acc_bits(E) = 1 + MAX_K_BITS + 2 * (E::BITS - 1) + log2(products per mac)
```

| `E` | bits | accumulator |
| --- | --- | --- |
| `i8` | 79 | `i128` |
| `i16` | 95 | `i128` |
| `i32` | 127 | `i128` |
| `i64` | 191 | `Limbs<3>` |
| `Complex<i32>` | 128 | pair of `i128` |
| `Complex<i64>` | 192 | pair of `Limbs<3>` |

Declaring `MAX_K_BITS` rather than probing it is what makes this one table
rather than one per target. A 32-bit host cannot reach that depth, so the width
is conservative there and wrong nowhere --- and `CD-06` and `CA-02` become
comparisons of one function rather than of two that happen to agree.

For a float element the accumulator is `Complete<L, MIN_EXP>`: a fixed-point
register spanning the entire product exponent range, plus sticky flags for the
non-finite states.

| `E` | product exponent span | limbs | bytes |
| --- | --- | --- | --- |
| `f32` | `2^-298 .. 2^256` | 10 | 80 |
| `f64` | `2^-2148 .. 2^2048` | 67 | 536 |

The non-finite state is a *flag*, not a value. A value would have to take part
in the fixed-point addition, and then the answer would depend on where in the
accumulation the infinity arrived. IEEE 754 clause 6 is about the value, not the
schedule.

## The tile and reduce factorizations

A tile kernel's vector lanes are columns of `C`; a reduce kernel's are steps of
`k`. A tile kernel produces `nr` columns per call whether the output has them or
not, so a shape narrower than `nr` pays for the ones that are not there --- at
`n = 1` that is one useful lane in ninety-six. A reduce kernel has no such
problem, because there is always more `k`.

Both compute the same integer, and `CB-06` asserts it. Which one runs is decided
by the shape against the tile, and the shape is a declaration the caller made
when they constructed the view. Within the reduce family the same rule picks the
panel *height*: the widest one the rows fill, because a panel wider than the
output is zero-padded and for a reduce kernel that padding is copied.

## Tabulation: when the operand is a code, the product is a table read

Every traversal above issues one product per `(i, p, j)` whatever the operands
hold. That is a property of the traversal, not of the identity. When the weights
are coded, a cheaper grouping exists:

```text
T[i][p][c] = sum over t < Bk of  A[i, p*Bk + t] * decode(c, t)
C[i][j]    = sum over p       of  T[i][p][ index_of(w[p][j]) ]
```

The table is built once per row tile and per block of the reduction and then read
`n` times. Its column loop is one read and one add per code, covering `Bk`
weights, and it contains no multiply --- asserted twice, by the operation census
(`Census`) and by reading the emitted instructions of the loop itself (`CU-06`).

The op counts follow from the two shapes: `m*k*S + m*k*n/Bk` against `m*k*n`, so
tabulation is cheaper exactly when `n * (Bk - 1) > S * Bk`. For `Book<256,8>` that
is `n > 292`, which `model/tiers.toml` records and `CM-04` recomputes.

The table is not the only sharing in this driver, and the other two axes are
the collapse traversal's own move applied to its two operands. Two equal rows
of `A` name the same sum against every column of `W`, and two equal columns of
`W` read the same table entries in the same order, so with the caller's offers
the driver charges per *distinct* row and per *distinct* column. The row side
is literally `collapse.rs`'s pass, compaction, and expansion --- `A` is always
dense here, so nothing about them changes --- with the compacted `d x k`
product planned as a product in its own right and then expanded into the full
output; the column side is a first-occurrence map over the code stream, made
block-local so it holds at any column-block width. Both follow the offer
discipline of the dense collapse: an epilogue that reads `C` declines the row
side outright, and a short offer, or an operand with nothing to share, gives
the same bytes from the uncollapsed traversal. `CD-15` and `CD-14` assert the
bytes at every degeneracy, and the census asserts the charge actually moved.

Three things make this a factorization of the same identity rather than a
different algorithm:

- **It is exact for the same reason tiling is.** A sum is a function of the
  multiset of its products, so regrouping them changes nothing. A classical
  `sgemm` cannot do this at all: its `T[c]` would carry its own rounding error and
  reusing it across `n` columns would propagate that error `n` times. Tabulation
  is available *only* to an exact library.
- **Which codecs admit it is a type-level fact.** `Enumerable` adds the code
  *space* that `Codec` never exposed. `Identity` and `Runs` do not implement it,
  for stated reasons, so a codec that cannot be tabulated cannot be handed to the
  traversal.
- **The reduction must run along the code block.** A code whose `Bk` elements land
  in `Bk` different output columns contributes one product to each and no partial
  sum to any. So the coded operand is `n x k` --- one row per output column --- and
  the product is `C := A * W^T`. `coded.rs`'s `k x n` orientation is the streaming
  one and needs no offer at all.

### The sequences live with the other sequences

`TableSpec` is `KernelSpec`'s shape for the two things a table needs --- fill one
slot, reduce one column group --- and it lives in `uor-matmul-kernels` for the same
reason every other sequence does: it is the only crate that writes
`#[target_feature]` --- 39 of them, against none anywhere else --- and that
attribute requires `unsafe`, which the other numerical crates forbid outright.
A sequence written
anywhere else compiles at the target's baseline. Measured, that was the whole
difference between 17.6 and 86.7 Gmac/s on the column loop.

The reference is generic over the element *and* the lane, including the lane that
is the exact accumulator, so there is one traversal in `gemm` and not one per
family. `CB-08` pins every other sequence to it lane for lane.

Three facts the loop rests on, all stated where they are relied upon:

- **The slab is a power of two and the read is masked.** `index_of` is total below
  `CODE_SPACE` --- `Enumerable`'s law, asserted by `CK-09` --- so masking changes no
  value the traversal can reach. What it changes is that every read is in-slab
  *whatever* the index holds, so the step needs no comparison and no branch. Safety
  holds unconditionally; correctness is the law.
- **The offsets arrive pre-scaled.** The driver multiplies the index by the tile
  height while it walks the code stream anyway, so the column loop has one `and`
  and one fused load-add per code and no multiply of any kind. That is the same
  discipline the packed panel follows on the dense side: the layout carries the
  address, so the inner loop walks and never indexes.
- **The codec reaches the column step as one exponent, and it is enumerated.** A
  slab is `slab_codes(CODE_SPACE) * rows` lane words and is asserted a power of
  two at the boundary, so the only thing a codec contributes to the step is
  `log2` of its rounded code space --- at most sixteen, because a code is a
  `u16`. `dispatch_slab!` matches it to a constant inside the `(rows, group)`
  dispatch that was already there, which turns every slot's base into a constant
  displacement; the wildcard arm binds zero and the runs take the slab from their
  argument instead, so it is one body at two bindings rather than two sequences
  (R13) and an unlisted code space is computed rather than refused (R8). Measured
  on a one-row tile, that binding is the difference between 3.8 and 9.4 Gmac/s.

### The lane, and where the exact accumulator is

Once, at the end. `Lane::capacity` says how many products one narrow word holds
exactly --- 133144 at `(i8, 128)` --- so the lane carries the *whole* reduction and
`AccOf<E>` is touched once per output element rather than once per chunk. A `k`
past the capacity is cut into runs and each run is placed once, which is the same
chunking `fits_narrow` already licenses for the tile kernels, and never a limit on
`k`.

Which lane a family uses is `Tabulated::Lane`, an associated type: which register
holds a run of products is a property of the element type, like `AccOf<E>`. It was
a per-shape search, which was a mechanism with one answer.

There is a second lane, chosen by declaration rather than by shape. When the
caller asks to encode by wrapping into an output no wider than `w` bits, the
table can run in `Z/2^w` --- the same ring-homomorphism factorization the dense
side cashes in (`ANALYSIS.md` §"The modular factorization"), with the same
two-declaration admissibility: the encode mode is asked at the traversal
boundary, the output width is `Tabulated::modular_table_admitted`. For `i32`
the `Mod32` word replaces scalar widening macs with eight-wide `mullo_epi32` on
the build and scalar 128-bit gather adds with `vpaddd` on the column loop, at a
quarter of the lane traffic; `CB-09` pins every modular sequence to the
portable modular reference lane for lane, and `CU-08` pins when the lane may
run and that its depth is unbounded at every bound. The lane is read out of the
accumulator offer, relabelled several words to the word, so an offer sized for
the exact lane already holds it. For `i64` the build's multiply has no SIMD
instruction, so the modular lane is the portable sequence alone --- the same
reason the dense `i64` modular family is portable-only. `i8` and `i16` offer
none: their exact lane already holds every depth a weight row reaches, so a
quotient read would buy nothing.

## Borrowing instead of packing

A packed panel is a copy, so a copy of something already in the panel's shape is
pure cost. For a `LaneLayout::Contiguous` kernel the panel *is* a run of rows or
of columns, so a row-major `A` and a column-major `B` already hold it.
`MatView::row_block` and `MatView::column_block` hand back the operand's own
memory when the strides say so, and `None` otherwise --- and when both sides
borrow, the traversal needs no working memory at all, so a matrix-vector product
runs blocked on an empty `Scratch`.

## The narrow/wide factorization

`fits_narrow(b, cap, k)` answers one question: may this tile be accumulated in a
narrower register without changing the answer? `narrow_cap_for` returns the
**narrowest** lane that suffices, scanning `NARROW_CAPS` from the narrow end,
because a wider cap is easier to satisfy and scanning from the wide end would
always return the widest.

Both sides compute the same integer. That is what separates an optimization from
a fallback: a fallback changes the answer or the guarantee, and this changes
neither. `CD-09` asserts it, and `CU-02` measures that the narrow path is
actually taken.

## The declared alphabet

A `KernelSpec` also declares `max_bound`: the widest alphabet bound at which its
sequence is exact. `lane_cap` bounds the *depth* and the driver answers that by
chunking; this is the other question, and chunking cannot answer it. A sequence
with an intermediate narrower than its lane is wrong past some magnitude however
shallow the chunk --- `_mm256_madd_epi16` sums two products into an `i32`, and
two full-magnitude `i16` products are one bit past it.

So selection does not consider a sequence outside its declared alphabet. Not
because it is riskier there: there it computes a different number, so it is not a
factorization of this identity at all. `CB-07` asserts both halves --- that every
sequence is exact at the extremes it declares, and that selection never offers
one outside them.

## The packed panel format

`KernelSpec` declares `mr`, `nr`, `k_group`, `lane_layout`, `lane_cap`, and
`max_bound`. One function gives the layout:

```text
packed_slot(p, lane, lanes, group) = (p / group) * (lanes * group)
                                   + lane * group
                                   + (p % group)
```

with `group = k_group` for `LaneLayout::Interleaved` and `group = kpad` for
`LaneLayout::Contiguous`. Interleaved is what a kernel wants when its vector
lanes are output columns: one load brings in the same `k`-group for every lane.
Contiguous is what a kernel wants when its lanes are steps of the reduction: one
load brings in a run of `k` for a single lane.

The driver's packer *is* that function, every kernel's index arithmetic is it
specialized to its own constants, and the parity tests read panels through it ---
so a kernel that disagrees with the layout disagrees with the reference.

Rows past `m`, columns past `n`, and depth past `k` up to the group multiple are
packed with the alphabet's zero. Zero padding contributes nothing to the sum and
nothing to the lane, which is why an unaligned or prime shape takes this path and
not a different one, and why no kernel has a `k`-tail. `CK-03` asserts that two tiers with equal decodes pack
byte-identical panels --- stronger than equal products, and the check that would
catch a codec whose decode is right but whose ordering is not.

## The canonical weight manifest

A weight artifact's identity is the kappa label of its manifest, not of its code
bytes, so that a transcode between tiers is visibly a different artifact with a
provably identical decoded stream.

```json
{"block":8,"bound":127,"codebook_sha256":"sha256:...","codes_sha256":"sha256:...",
 "cols":4096,"rows":4096,"spec":"uor-matmul/1","tier":"Book"}
```

JCS-RFC8785 canonical form: members in lexicographic key order, no whitespace,
no escapes. Bulk arrays are referenced by digest rather than inlined, so the
manifest stays inside `uor-addr-1`'s ceilings and the address stays cheap.

## The error surface

Two variants, both meaning *the requested object does not exist*, both reported
at view construction, before any arithmetic is named. See `README.md` for the
table of what is deliberately absent and why each absence is load-bearing.
