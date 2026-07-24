Feature: Backend parity

  Every backend is a factorization of one identity.

  Every scenario below names the conformance ID it discharges, and a test
  whose name ends in that ID runs it. `cargo xtask check-model` fails if an
  ID here has no register row, or a register row has no scenario (CM-02).

  @CB-01 @build
  Scenario: Portable equals `dot_ref` on the whole corpus
    Given the standing corpus
    When the suite exercises CB-01
    Then the claim holds byte for byte

  @CB-02 @build
  Scenario: AVX2 equals portable
    Given the standing corpus
    When the suite exercises CB-02
    Then the claim holds byte for byte

  @CB-03 @build
  Scenario: AVX-512 VNNI equals portable, on all three of its sequences
    Given the standing corpus
    When the suite exercises CB-03
    Then the claim holds byte for byte

  @CB-04 @build
  Scenario: NEON and NEON dotprod equal portable
    Given the standing corpus
    When the suite exercises CB-04
    Then the claim holds byte for byte

  @CB-05 @build
  Scenario: wasm SIMD128 equals portable, and SIMD128-off equals SIMD128-on
    Given the standing corpus
    When the suite exercises CB-05
    Then the claim holds byte for byte
