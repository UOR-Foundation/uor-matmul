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
  Scenario: The float driver's scaled lanes give byte-identical output whether the integer dot product runs as the scalar loop or on the integer kernel table, at every shape and every offer
    Given the standing corpus
    When the suite exercises CD-19
    Then the claim holds byte for byte

  @CD-20 @build
  Scenario: A u8-symbol-coded float weight matrix tabulated in the scaled integer lane gives byte-identical output to the dense float driver at every shape and every offer, with the span walk admitted and declined alike
    Given the standing corpus
    When the suite exercises CD-20
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
