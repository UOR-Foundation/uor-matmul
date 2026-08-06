Feature: Purity

  One method. No classical path, no fallback.

  Every scenario below names the conformance ID it discharges, and a test
  whose name ends in that ID runs it. `cargo xtask check-model` fails if an
  ID here has no register row, or a register row has no scenario (CM-02).

  @CU-01 @build
  Scenario: No float arithmetic opcode in any shipped kernel's disassembly, except where the F64Exact witness proves the opcode exact
    Given the standing corpus
    When the suite exercises CU-01
    Then the claim holds byte for byte

  @CU-02 @build
  Scenario: The instrumented count of narrow-path tiles matches `fits_narrow` exactly, so no tile takes an unintended path
    Given the standing corpus
    When the suite exercises CU-02
    Then the claim holds byte for byte

  @CU-03 @build
  Scenario: Every instruction sequence agrees at depths straddling its own threshold
    Given the standing corpus
    When the suite exercises CU-03
    Then the claim holds byte for byte

  @CU-04 @build
  Scenario: Float accumulation is order-independent: shuffled tiles and every backend agree bit for bit, including on catastrophic-cancellation cases
    Given the standing corpus
    When the suite exercises CU-04
    Then the claim holds byte for byte

  @CU-05 @build
  Scenario: There is exactly one accumulation path per element family, asserted by `audit-purity` over the call graph
    Given the standing corpus
    When the suite exercises CU-05
    Then the claim holds byte for byte

  @CU-06 @build
  Scenario: The tabulated inner loop issues no multiply
    Given the standing corpus
    When the suite exercises CU-06
    Then the claim holds byte for byte

  @CU-07 @build
  Scenario: Every shipped crate either forbids `unsafe_code` outright or is a crate the Miri job runs
    Given the standing corpus
    When the suite exercises CU-07
    Then the claim holds byte for byte

  @CU-08 @build
  Scenario: The modular table lane runs exactly when the encode is `Wrapping` and the output is no wider than the lane; its depth is unbounded at every bound
    Given the standing corpus
    When the suite exercises CU-08
    Then the claim holds byte for byte

  @CU-10 @build
  Scenario: No tropical sequence issues a multiply, read from the emitted instructions
    Given the standing corpus
    When the suite exercises CU-10
    Then the claim holds byte for byte
