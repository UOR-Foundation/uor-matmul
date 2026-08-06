# ARCHITECTURE

Normative. Where this document and the code disagree, one of them is a bug.

## The one sentence

> Decode the code, accumulate exactly, encode once.

Every entry point is that sentence at a different instantiation. The parameters
are `(Element, Bound, Codec, MaxBlock, Backend, Traversal)`, and W8A8 is the
instantiation `(i8, 127, Identity, 1, *, *)`. Nothing privileges it beyond its
having the most instruction support and the most external oracles.

The **algebra is not a seventh parameter**. It is carried inside the first one:
`Trop<i8>` is a different `Element` from `i8`, with a different `Element::Acc`,
and that is the whole of how the `(max, +)` semiring reaches the code. See "The
two semirings, and why one of them is an element type" below.

## The crates, and why they are separate

| Crate                    | Contains                                                                      | `unsafe`              | `alloc` | float arithmetic |
| ------------------------ | ----------------------------------------------------------------------------- | --------------------- | ------- | ---------------- |
| `uor-matmul-core`        | alphabet, accumulator, reference accumulation, views, the error surface       | forbidden             | none    | none             |
| `uor-matmul-codec`       | the `Codec` trait, ten tiers, `CodedMatrix`, the kappa manifest, the E8 table | forbidden             | none    | none             |
| `uor-matmul-kernels`     | one module per ISA, each a `KernelSpec` value                                 | permitted, documented | none    | none             |
| `uor-matmul-gemm`        | the driver: traversals, scratch, epilogue, partition                          | none                  | none    | none             |
| `uor-matmul`             | the facade and the raw-pointer face                                           | the raw face only     | none    | none             |
| `uor-matmul-model`       | the typed model and the generators                                            | none                  | freely  | n/a              |
| `uor-matmul-validate`    | oracle adapters, corpus, scaling harness                                      | documented            | freely  | n/a              |
| `uor-matmul-conformance` | the BDD runner and the honesty meta-gate                                      | none                  | freely  | n/a              |

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

| `E`            | bits | accumulator        |
| -------------- | ---- | ------------------ |
| `i8`           | 79   | `i128`             |
| `i16`          | 95   | `i128`             |
| `i32`          | 127  | `i128`             |
| `i64`          | 191  | `Limbs<3>`         |
| `Complex<i32>` | 128  | pair of `i128`     |
| `Complex<i64>` | 192  | pair of `Limbs<3>` |

Declaring `MAX_K_BITS` rather than probing it is what makes this one table
rather than one per target. A 32-bit host cannot reach that depth, so the width
is conservative there and wrong nowhere --- and `CD-06` and `CA-02` become
comparisons of one function rather than of two that happen to agree.

For a float element the accumulator is `Complete<L, MIN_EXP>`: a fixed-point
register spanning the entire product exponent range, plus sticky flags for the
non-finite states.

| `E`   | product exponent span | limbs | bytes |
| ----- | --------------------- | ----- | ----- |
| `f32` | `2^-298 .. 2^256`     | 10    | 80    |
| `f64` | `2^-2148 .. 2^2048`   | 67    | 536   |

The non-finite state is a *flag*, not a value. A value would have to take part
in the fixed-point addition, and then the answer would depend on where in the
accumulation the infinity arrived. IEEE 754 clause 6 is about the value, not the
schedule.

## The two semirings, and why one of them is an element type

The operation census this library realizes names two products: *matrix products
under complete accumulation*, and *max*. The second is the `(max, +)` semiring
--- `⊕` is `max`, `⊗` is addition, the additive identity is `-inf` and the
multiplicative identity is `0`.

It reaches the code as an **element type**, `Trop<E>`, and not as a parameter on
the driver. The reason is `Element::Acc`, which is documented as *not a
parameter and not a choice*: exactly one accumulator per element type. The two
semirings do not share one --- the ring's is seventy-nine bits at `i8` and this
one's is ten --- so a semiring parameter beside the element type would make
`Acc` a function of two things and contradict its own contract.

Carrying it in the element type buys the property this document opens with. The
dense driver's body names `E::mac` and `Accumulator::combine` and nothing else,
so it computes a ring product over `Alphabet<i8, _>` and a `(max, +)` product
over `Alphabet<Trop<i8>, _>` with no branch and no second traversal (`CD-29`).
`gemm`'s bound is `E: Element`, not `E: IntegerElement`, for exactly this
reason.

| | ring | tropical |
| --- | --- | --- |
| element | `i8` .. `i64`, `Complex<_>`, `f32`, `f64` | `Trop<i8>` .. `Trop<i64>` |
| `⊕` | `+` | `max` |
| `⊗` | `*` | `+` |
| additive identity | `0` | `-inf` |
| accumulator | `acc_bits(E)`, with a `MAX_K_BITS` term | `trop_acc_bits(E)`, with **no** depth term |
| `⊕` idempotent | no | yes |
| epilogue | `Linear { alpha, beta }` (`CS-05`) | `MaxPlus { alpha, beta }` (`CS-12`) |
| sub-cubic recursion | available | absent: no additive inverses |

The last two rows are enforced by the type system rather than by a check.
`Trop<E>` does not implement `IntegerElement`, so `gemm_strassen` --- which needs
`IntegerElement::sub` --- does not exist at a tropical instantiation; and the
`Linear` impl requires `AccOf<E>: ScaleExact`, which `TropAcc` does not
implement, so `beta * C` cannot be written where it has no meaning. Neither
exclusion can be reached at run time, which is the difference between excluding
by construction and refusing by a branch (U-ii).

`Semiring` is the declaration that makes this checkable: two zero-sized markers,
`Ring<E>` and `Tropical<E>`, each declaring whether its `⊕` is idempotent, so
that one gate body quantifies over both. Nothing in a traversal reads it and no
traversal signature mentions it; removing it would change no output byte. What
it changes is that `CK-16` can say *the laws hold at every instance, and
idempotence holds precisely at one of them* --- a sentence that is not
expressible as two independent tests.

### The selection witness

A `(max, +)` product answers a question a sum cannot: which term achieved the
maximum. `SelectedTriple` carries a witness matrix beside `C`, validated at
construction against the same two conditions `Triple::new` reports and no
others, and `gemm_selected` writes both. The tie-break is the smallest index; it
is cited (`AUTH-TF1-D6`), not chosen here.

Two mechanisms produce it, and they are factorizations rather than alternatives.
`Witness::Lexicographic` reduces `(value, index)` under an order that is total
on the pair, so invariance under partition and order is a property of the order
and not of the loop. `Witness::ComparePass` reduces the value and takes a
compare-only second pass for the least index attaining it. `CD-25` asserts their
bytes are identical at every shape, degeneracy and offer including none.

### The two gauge sections

`recenter` is the canonical section of the shift gauge: `(max, +)` is invariant
under `⊗ c`, and the representative is the one whose maximum is exactly zero. It
is taken in the accumulator's width, where it is exact for every input --- a
difference of two alphabet elements has magnitude at most `2^BITS`, which is
precisely what `trop_acc_bits` derives the width against.

`dyadic` is the canonical section of the dyadic gauge, and it is a *placement*
rather than a division: `Complete::add_scaled` takes a signed exponent, so
writing a value at `2^-k` in the fixed-point register **is** division by `2^k`
with every bit preserved. No division opcode is emitted and none is needed,
which is why the census has no division in it.

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

The build is the one place a multiply remains, and at bound 1 not even there.
The sign spelling (`Packed<Grid<2>,8>` over `Bnd<1>`, table `[-1,+1]`) and the
ternary one (`Packed<Grid<4>,4>`, table `[-1,0,+1,dead]`) --- both declared by
`CK-13` as compositions of codecs that already existed --- put every book word
in `{-1, 0, +1}`, where the product is the activation, its negation, or zero:
`T[c][i]` is adds and subtracts, with the negation an XOR against the sign mask
and the mask subtracted back, two's complement's own spelling of `-a`. So the
builds that declare `max_bound = 1` issue no multiply at all. This is selection
by declaration, not a second method: the bound-1 builds sit after every
full-alphabet sequence in the available list, `choose_table` hands them exactly
the alphabet they declare, and only the build differs --- the gathers are
bound-independent and are the same function pointers, shared rather than
duplicated. `CB-10` pins every bound-1 build to the model slot for slot and the
census to what was issued: zero multiplies on a bound-1 tabulated run.

The table is not the only sharing in this driver, and the other two axes are
the collapse traversal's own move applied to its two operands. Two equal rows
of `A` name the same sum against every column of `W`, and two equal columns of
`W` read the same table entries in the same order, so with the caller's offers
the driver charges per *distinct* row and per *distinct* column. For the float
families the row side reads *bit*-distinct: the symbol is the bit pattern ---
the arena tier's canonical-codebook semantics (`CK-10`) applied to `A` --- so
two rows that differ only in the sign of a zero or in a NaN payload are
charged as the two rows they are, and rows identical bit for bit, NaNs
included, are charged once (`CD-17`). The row side
is literally `collapse.rs`'s pass, compaction, and expansion --- `A` is always
dense here, so nothing about them changes --- with the compacted `d x k`
product planned as a product in its own right and then expanded into the full
output; the column side is a first-occurrence map over the code stream, made
block-local so it holds at any column-block width. Both follow the offer
discipline of the dense collapse: an epilogue that reads `C` declines the row
side outright, and a short offer, or an operand with nothing to share, gives
the same bytes from the uncollapsed traversal. `CD-15`, `CD-16`, and `CD-17`
assert the
bytes at every degeneracy, and the census asserts the charge actually moved.

### Addressing is a declaration, not an exclusion

Whether this grouping exists for an operand is decided by what the operand
*declares*, never by what it holds. `Addressing::of(tier, block, bound)` reads
exactly the three manifest fields that describe the code --- and neither of the
two that describe the artifact's bytes --- and answers whether a code addresses
an element at all, and whether it addresses a *run*. `Identity` and `Runs`
address nothing, and take the dense traversal; that is not a refusal, it is what
their declarations say, and the dense traversal is the census's first product
rather than a fallback from this one.

Orientation is the second half of the same question. The reduction must run
along the code block, so the coded operand must be stored `n x k` --- which is
`rows` and `cols` in the canonical manifest. `Manifest::reduces_along_the_block`
asks the declaration; nothing probes the stream. `CS-10` asserts both directions:
at a fixed manifest the selection does not move when the operand's values change,
and it does move when the declaration changes.

A consequence worth stating rather than discovering: `tabulation_pays` refuses at
`MAX_BLOCK == 1`, and `Addressing`'s `addresses_a_run` is false at exactly the
same point. The arena tier's route back to the dense walk is therefore *stated*
by the derivation rather than contradicted by it.

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
`#[target_feature]` --- 40 of them, against none anywhere else --- and that
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

### The arena tier: a float is a symbol with an address

A float was always a code here --- `FloatElement::decode` reads the name of a
dyadic rational, and the accumulation below it is the same integer accumulation
every other element family uses. The arena tier carries that one level up: the
distinct bit patterns of a weight artifact are its codebook, canonicalized by
`canonicalize` into unsigned pattern order, so the artifact's identity at the
kappa level is its manifest and its elements are `u16` indices. Two artifacts
holding the same values share a codebook whatever order their streams stored
them in. `CK-10` pins the construction --- signed zeros and NaN payloads are
distinct symbols, because identity is pattern equality, not value equality ---
and `CK-08` pins the manifest spelling, `bound` recording `Whole`'s `u128::MAX`
because a float alphabet has no magnitude to declare.

Nothing below the identity level changes. `Codec` and `CodedMatrix` are generic
over `Element`, the arena's decode is the table read `Grid` performs, and the
table's lane is a register that holds the products exactly, so tabulation is
exact for the same reason it is for integer codes: the table entries are exact
partial sums and reusing them is regrouping, not rounding. Which register ---
the complete accumulator, or the scaled integer word the next paragraph names
--- is the family's declaration, and `CD-14` asserts the whole traversal
byte-equal to the dense float driver, forced and declined alike.

What the tier does not claim is a traversal win, and the model records why.
`MAX_BLOCK` is one --- one symbol per code --- and `tabulation_pays` refuses a
one-element block on op count, exactly as it refuses `Grid<16>`: the table
saves the multiply and no adds, so selection declines and the byte-identical
dense route runs. The arena's claim is identity and residency; `CG-03` measures
residency like any other codec's, and `CG-14` reports what the residency buys
against the bus.

The code width is a parameter of the one tier, not a second tier:
`Arena<'_, E, N>` addresses the codebook with a `u16`, and
`Arena<'_, E, N, u8>` with a byte (`CK-14`). At the byte width a codebook of
256 distinct patterns stores one byte a symbol against the dense float's four,
and the coded operand drives the same exact accumulation: the traversal
declines the one-element block, as it always does, and the stream below the
decline is the coded float path --- `CD-18` pins it byte-identical to
`gemm_float` at every shape and every offer, under both epilogues. A `u8`
stream is never the index stream the gather borrows --- the gather's index
type is `u16` --- so the narrow spelling re-spells its codes as indices
wherever the traversal builds one, at the same bytes.

What the byte width does not buy by itself is the table's arithmetic: a table
entry is `rows` lane words, and over the 80-byte complete accumulator a
256-entry table holds no L1 slab at any tile. The lane question is answered
one level up, not in the codec: `f32`'s table lane is `Scaled64`, an `i64` of
significands pre-scaled to the panels' measured base exponents --- the
placement bridge's span walk and admission, over `A` and over the codebook,
asked once per call and only when the table was selected. At eight bytes a
word the same slab holds at a four-row tile, and the forced traversal
tabulates: the build decodes each symbol into the scaled alphabet once, the
gather accumulates the narrow words, and the exact sum is placed at
`2^(base_a + base_b)` through the accumulator's own `add_scaled` (`CD-20`).
The exact accumulator is still the lane the streaming decline accumulates
in, because the stream walks raw elements with no walk ahead of them; and it
is still the whole answer for `f64`, whose 53-bit significand is not an
`i32` at any span. Selection is a separate question, and `CG-16` measured
it: the win is an op-kind one the instruction-count predicate cannot see,
so the costed traversal still declines the one-element block. A float block
codebook --- `MAX_BLOCK` past one --- remains a further step: `Book` is
bounded to integer elements, so such a codebook would be a generalization of
`Book`, not an instantiation of anything here.

One implementation note is load-bearing. The portable gather staged a column
group of lane words in a compile-time array, which pays when the words can sit
in registers and is a frame of pure copy when they cannot --- 80 or 536 bytes
of limbs never can. `portable_table` chooses at compile time: words of sixteen
bytes or fewer take the staged gather, wider words accumulate in place, and
both are the same reads and the same adds (`CB-08`).
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
