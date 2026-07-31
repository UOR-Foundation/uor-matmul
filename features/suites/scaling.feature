Feature: Scaling

  Fitted exponents. Measured and reported, never asserted.

  Every scenario below names the conformance ID it discharges, and a test
  whose name ends in that ID runs it. `cargo xtask check-model` fails if an
  ID here has no register row, or a register row has no scenario (CM-02).

  @CG-01 @open
  Scenario: Arithmetic scaling exponent, this library and every oracle
    Given the standing sweep and its recorded seed
    When the harness measures CG-01
    Then the figure is reported with its confidence interval
    And nothing asserts it as established

  @CG-02 @open
  Scenario: Per-axis scaling exponents for `m`, `n`, `k` separately
    Given the standing sweep and its recorded seed
    When the harness measures CG-02
    Then the figure is reported with its confidence interval
    And nothing asserts it as established

  @CG-03 @open
  Scenario: Residency scaling: bytes of weight storage touched, per codec, against every oracle
    Given the standing sweep and its recorded seed
    When the harness measures CG-03
    Then the figure is reported with its confidence interval
    And nothing asserts it as established

  @CG-04 @open
  Scenario: Working-set scaling, `suggested_scratch` against each oracle's measured internal allocation
    Given the standing sweep and its recorded seed
    When the harness measures CG-04
    Then the figure is reported with its confidence interval
    And nothing asserts it as established

  @CG-05 @open
  Scenario: Allocation count and peak bytes: zero here, whatever the oracle does there
    Given the standing sweep and its recorded seed
    When the harness measures CG-05
    Then the figure is reported with its confidence interval
    And nothing asserts it as established

  @CG-06 @open
  Scenario: Parallel speedup against tile count, with byte-equality asserted inside the timed harness
    Given the standing sweep and its recorded seed
    When the harness measures CG-06
    Then the figure is reported with its confidence interval
    And nothing asserts it as established

  @CG-07 @open
  Scenario: Small-shape latency, where a heavyweight prologue costs more than an asymptote
    Given the standing sweep and its recorded seed
    When the harness measures CG-07
    Then the figure is reported with its confidence interval
    And nothing asserts it as established

  @CG-08 @open
  Scenario: Sustained throughput on super-massive input, reported per pass, against every oracle that finishes
    Given the standing corpus
    When the suite exercises CG-08
    Then the figure is reported and not asserted

  @CG-09 @open
  Scenario: Throughput against the degeneracy of the operand, including the price of looking when there is none
    Given the standing corpus
    When the suite exercises CG-09
    Then the figure is reported and not asserted

  @CG-10 @open
  Scenario: Operation census and wall time for Tabulated against Blocked, swept over n through the break-even
    Given the standing corpus
    When the suite exercises CG-10
    Then the figure is reported and not asserted

  @CG-11 @build
  Scenario: Static issue analysis over the emitted inner loops names a bottleneck resource for every kernel sequence
    Given the emitted assembly of the kernels crate and llvm-mca's scheduling models
    When the suite exercises CG-11
    Then every analysed sequence is reported with a named bottleneck
    And the figures are scheduling-model predictions, never asserted as measurements

  @CG-12 @open
  Scenario: Achieved MACs per second of the sub-cubic recursion against the cubic packed walk on the i32-exact lane, swept through the crossover, with the host's fastest sustained product rate on the same axes
    Given the standing sweep and its recorded seed
    When the harness measures CG-12
    Then the figure is reported and not asserted

  @CG-13 @build
  Scenario: The resolved kernel sequence is cached per element family, and a cached selection returns the sequence the full walk returns
    Given the kernel table and a host whose feature bits cannot change while it runs
    When the suite exercises CG-13
    Then a cached selection equals the full walk for every family, bound, and panel height
    And the latency this buys is reported under CG-07 and never asserted

  @CG-14 @open
  Scenario: Achieved bytes per second for a u8-symbol-coded gemv and skinny GEMM against an f32 oracle and the host's measured STREAM number
    Given the standing sweep and its recorded seed
    When the harness measures CG-14
    Then the figure is reported and not asserted

  @CG-15 @open
  Scenario: Achieved MACs per second of the float placement bridge, the default auto-selection against the explicit entry, on the shapes ANALYSIS tables
    Given the standing sweep and its recorded seed
    When the harness measures CG-15
    Then the figure is reported and not asserted

  @CG-16 @open
  Scenario: Achieved MACs per second of the symbol tabulated traversal in the scaled integer lane against the float placement bridge, the dense float driver, and an f32 oracle, on gemv, skinny, and tabulation-sweep shapes
    Given the standing sweep and its recorded seed
    When the harness measures CG-16
    Then the figure is reported and not asserted

  @CG-17 @open
  Scenario: Achieved MACs per second of the i64x2 SWAR broadcast sequence against the i32x4 dot-with-extends sequence and the portable reference, on wasm32-wasip1 under wasmtime
    Given the standing sweep and its recorded seed
    When the harness measures CG-17
    Then the figure is reported and not asserted

  @CG-18 @build
  Scenario: The operation census is the performance gate: selection is the derivation at the break-even, and the running gather issues no multiplies beyond the build's
    Given the break-even recomputed at test time from the sequences' own per-ISA declarations
    When the suite exercises CG-18
    Then below the break-even the census shows the dense route, and at and above it the table
    And the table's multiplies are exactly the build's charge, and a block-1 codec is declined at every size
