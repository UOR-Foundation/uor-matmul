Feature: Determinism

  Same inputs, same bytes, whatever the schedule.

  Every scenario below names the conformance ID it discharges, and a test
  whose name ends in that ID runs it. `cargo xtask check-model` fails if an
  ID here has no register row, or a register row has no scenario (CM-02).

  @CD-01 @build
  Scenario: Backend choice does not change output bytes
    Given the standing corpus
    When the suite exercises CD-01
    Then the claim holds byte for byte

  @CD-02 @build
  Scenario: Tile count and completion order do not change output bytes, over `{1,2,3,5,8,64,m*n}` and shuffled orders
    Given the standing corpus
    When the suite exercises CD-02
    Then the claim holds byte for byte

  @CD-03 @build
  Scenario: Every partial sum stays inside the accumulator, measured by `dot_instrumented`, never inferred
    Given the standing corpus
    When the suite exercises CD-03
    Then the claim holds byte for byte

  @CD-04 @build
  Scenario: Traversal and blocking preset do not change output bytes, including deliberately bad presets
    Given the standing corpus
    When the suite exercises CD-04
    Then the claim holds byte for byte

  @CD-05 @build
  Scenario: Encode mode is the only thing that changes the output bytes for a fixed accumulation
    Given the standing corpus
    When the suite exercises CD-05
    Then the claim holds byte for byte

  @CD-06 @build
  Scenario: Big-endian and little-endian targets agree on the serialized output
    Given the standing corpus
    When the suite exercises CD-06
    Then the claim holds byte for byte

  @CD-07 @build
  Scenario: Splitting the operand stream at any index sums the parts
    Given the standing corpus
    When the suite exercises CD-07
    Then the claim holds byte for byte

  @CD-08 @build
  Scenario: Two tiles reduce in either order
    Given the standing corpus
    When the suite exercises CD-08
    Then the claim holds byte for byte

  @CD-09 @build
  Scenario: The narrow-register tile path agrees with `dot_wide`
    Given the standing corpus
    When the suite exercises CD-09
    Then the claim holds byte for byte

  @CD-10 @build
  Scenario: `Scratch::None`, one byte, `suggested_scratch - 1`, `suggested_scratch`, and ten times it all give the same bytes
    Given the standing corpus
    When the suite exercises CD-10
    Then the claim holds byte for byte

  @CD-11 @build
  Scenario: Forcing a wider accumulator than necessary changes nothing but the room
    Given the standing corpus
    When the suite exercises CD-11
    Then the claim holds byte for byte

  @CD-12 @build
  Scenario: Collapsing equal rows of A cannot change a byte, at every degeneracy and every offer
    Given the standing corpus
    When the suite exercises CD-12
    Then the claim holds byte for byte

  @CD-13 @build
  Scenario: Tabulated, Blocked, and OutputMajor produce byte-identical output at every shape
    Given the standing corpus
    When the suite exercises CD-13
    Then the claim holds byte for byte

  @CD-14 @build
  Scenario: An arena-coded float weight matrix gives byte-identical output to the dense float driver at every shape
    Given the standing corpus
    When the suite exercises CD-14
    Then the claim holds byte for byte

  @CD-16 @build
  Scenario: Collapsing equal columns of the coded operand cannot change a byte at any column-block width, and a repeated column is never charged twice within its block
    Given the standing corpus
    When the suite exercises CD-16
    Then the claim holds byte for byte

  @CD-15 @build
  Scenario: Collapsing equal rows of A in the tabulated traversal cannot change a byte, at every degeneracy and every offer
    Given the standing corpus
    When the suite exercises CD-15
    Then the claim holds byte for byte

  @CD-17 @build
  Scenario: Collapsing bit-identical rows of A in the float tabulated traversal cannot change a byte, at every degeneracy and every offer; rows differing only in the sign of zero or in a NaN payload are distinct
    Given the standing corpus
    When the suite exercises CD-17
    Then the claim holds byte for byte

  @CD-18 @build
  Scenario: A u8-symbol-coded float weight matrix gives byte-identical output to the dense float driver at every shape and every offer
    Given the standing corpus
    When the suite exercises CD-18
    Then the claim holds byte for byte

  @CD-19 @build
  Scenario: Every historical float workspace spelling is the same zero-copy Atlas-octet reduction at every shape, offer, and initial workspace pattern
    Given the standing corpus and caller-owned workspace filled with distinct poison patterns
    When the suite exercises CD-19 without interpreting post-call workspace residue
    Then the outputs are byte-identical, backing storage is retained, and neither initial bytes nor unspecified residue become operand data

  @CD-20 @build
  Scenario: Symbol-coded floats match dense Atlas bytes through a contextual f32 octet pair and codec-parametric f64 complete tables
    Given the standing corpus
    When the suite composes f32 projection with Scaled64 consumption and exercises both block-one and downstream block-two f64 codecs for CD-20
    Then the claim holds byte for byte

  @CD-21 @build
  Scenario: The sub-cubic recursion is byte-identical to the cubic packed walk at every shape, level count, and offer, and to the CX-01 oracle at every corpus size; an unadmitted level is declined and declining changes no byte
    Given the standing corpus
    When the suite exercises CD-21
    Then the claim holds byte for byte

  @CD-22 @build
  Scenario: The documented default integer entry point selects the kernelized factorization the caller's offer and the host's declarations admit, witnessed by the route census; every route returns the reference traversal's bytes at every offer including none, and the reference remains directly callable
    Given the standing corpus
    When the suite exercises CD-22
    Then the claim holds byte for byte

  @CD-23 @build
  Scenario: The one-level modular bilinear factorization returns the direct packed modular walk's bytes at every shape and every offer, with seven base products per level counted by the census, and declines to the direct walk wherever the shape, encode, lane, or offer does not admit it
    Given the standing corpus
    When the suite exercises CD-23
    Then the claim holds byte for byte

  @CD-24 @build
  Scenario: The selection witness is invariant under partition, count and order, because the order on `(value, index)` is total
    Given the standing corpus
    When the suite exercises CD-24
    Then the claim holds byte for byte

  @CD-25 @build
  Scenario: Both witness mechanisms give identical bytes at every shape, degeneracy, and offer including none
    Given the standing corpus
    When the suite exercises CD-25
    Then the claim holds byte for byte

  @CD-26 @build
  Scenario: Recentring is the canonical section of the shift gauge: gauge-invariant, idempotent, and its representative's maximum is exactly zero
    Given the standing corpus
    When the suite exercises CD-26
    Then the claim holds byte for byte

  @CD-27 @build
  Scenario: The dyadic section is exact, and equals the arithmetic shift at every `k` where that shift exists
    Given the standing corpus
    When the suite exercises CD-27
    Then the claim holds byte for byte

  @CD-28 @build
  Scenario: Within one element type, encode mode is the only thing that changes the output bytes
    Given the standing corpus
    When the suite exercises CD-28
    Then the claim holds byte for byte

  @CD-29 @build
  Scenario: One traversal computes both products of the operation census: the dense driver is parametric in the semiring
    Given the standing corpus
    When the suite exercises CD-29
    Then the claim holds byte for byte

  @CD-30 @build
  Scenario: Every pure-UOR float factorization returns the exact reference bytes through every public workspace spelling and offer
    Given the standing float corpus over both formats, every IEEE code class, and every offer including none
    When the suite exercises every pure-UOR float factorization and public entry for CD-30
    Then every output is byte-identical to the independently computed exact reference

  @CD-31 @build
  Scenario: The common-base interval theorem certifies admission and minimum grouping without becoming a runtime route
    Given formal variable-capacity intervals small enough for exhaustive comparison
    When the suite derives admission, grouping, and the greatest common base for CD-31
    Then direct construction admits exactly those bases, exhaustive search finds no smaller grouping, and no smaller base has more headroom
    And the certificate declares no second float execution route

  @CD-32 @build
  Scenario: The total binary32 q-carrier tags every boundary and scalar-fractures every short run without changing traversal selection
    Given the compact binary32 corpus, every IEEE code class and exponent-span boundary, all seven Complete non-finite unions, an empty reduction, and codec blocks and code spaces 2, 3, and 5 including a block wider than a one-product lane
    When the suite executes resident forced tables and equal-shape automatic calls through the contextual q-carrier across repeated columns, reduction and column tails, non-unit source strides, and absent, short, and complete offers for CD-32
    Then the 48-bit product, 507 relative grades, 1021 states, 10-bit state, 58-bit tag payload and top-positive interval, and zero-span capacity 31744 equal their model derivations
    And every forced table executes with zero traditional multiplies, the exact declared Census, and reference-identical bytes; the empty reduction has zero work; every non-finite product is placed immediately in source order as a one-product run; and a capacity below the codec block scalar-fractures with the original block stride instead of declining
    And compact products retain their complete bytes while the maximal-prefix recurrence groups already-projected source slots by the least nonnegative per-slot L-infinity certificate rather than by a global span
    And exhaustive ordered partitions find no fewer common-boundary groups under those certificates, without claiming cancellation-sensitive or independent-lane optimality
    And equal shapes select the same traversal independently of their values
