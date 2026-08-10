Feature: Allocation and environment

  Zero heap, no_std, and the same bytes off the host.

  Every scenario below names the conformance ID it discharges, and a test
  whose name ends in that ID runs it. `cargo xtask check-model` fails if an
  ID here has no register row, or a register row has no scenario (CM-02).

  @CA-01 @build
  Scenario: Zero allocations during any call, on every hosted target
    Given the standing corpus
    When the suite exercises CA-01
    Then the claim holds byte for byte

  @CA-02 @build
  Scenario: Identical bytes on `thumbv7em-none-eabihf` and both wasm targets as on x86-64
    Given the standing corpus
    When the suite exercises CA-02
    Then the claim holds byte for byte

  @CA-03 @build
  Scenario: No shipped crate links an allocator symbol on a `no_std` target
    Given the standing corpus
    When the suite exercises CA-03
    Then the claim holds byte for byte

  @CA-04 @build
  Scenario: The tropical accumulator's width is independent of `k`, and is the same width at depth one and at the deepest reduction the machine can address
    Given the standing corpus
    When the suite exercises CA-04
    Then the claim holds byte for byte

  @CA-05 @build
  Scenario: The Atlas carrier and projector layer is a zero-copy view over caller-owned storage
    Given caller-owned panels at every offer including none
    When the suite exercises the carrier and projector layer for CA-05
    Then no carrier is owned or allocated, the backing address is preserved, and the dyadic denominator remains implicit
