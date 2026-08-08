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
  Scenario: Historical float workspace spellings execute one Atlas operation
    Given the retained workspace-sweep shapes and exponent-span fills
    When the harness measures CG-15
    Then byte identity holds inside every timed region and the product rates are reported but not asserted

  @CG-16 @open
  Scenario: Public block-one Atlas table crossover against the public coded Atlas decline
    Given the recorded code spaces, shapes, exponent spans, seeds, and caller offers
    When paired calibrated batches run only the named production calls between the clock reads
    Then poison and complete byte identity bracket every timed batch, counted calls establish both routes and a multiply-free table, and the paired intervals are reported but not asserted

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
    And the table's multiplies are exactly the build's charge, and an ordinary product-build block-1 codec is declined at every size

  @CG-19 @open
  Scenario: Tropical selection throughput against the ring lane at matched shapes, and witness mechanism A against B
    Given the standing sweep and its recorded seed
    When the harness measures CG-19
    Then the figure is reported with its confidence interval
    And nothing asserts it as established

  @CG-21 @open
  Scenario: Pure-UOR float throughput, traffic, and latency against the incumbent and oracles, with non-float controls
    Given the standing f32, f64, integer, and tropical sweeps and their recorded seed
    When the harness brackets each CG-21 batch with poison and complete byte identity outside a timer containing only the real production call
    Then every figure is reported with its confidence interval
    And nothing asserts it as established

  @CG-22 @build
  Scenario: The direct pure-UOR traversal chooses the global model-derived executed-work lookup orientation and leaves non-float routes unchanged
    Given every eligible group-one lookup-add family, both float accumulator sizes, exact-cell residency, fixed Atlas workspace, full and edge subcells, the direct coordinate census, and captured integer and tropical control counts
    When the suite exercises shapes selecting each executed-work orientation for CG-22
    Then the route census names the global model-derived minimum, shipped storage and work equal their independent model twins, and every non-float route and operation count is unchanged

  @CG-23 @open
  Scenario: Native lookup changes retain demonstrated superiority while linked static-equivalent controls remain open
    Given exact ELF inspection of every linked native lookup body and the aligned direct production alphabet address on the recorded compiler and host
    When the sentinel measures each changed case and each static-equivalent control through identical safe wrappers with poison and complete byte checks
    Then every case emits 256 interleaved paired samples and its confidence interval
    And only the structurally changed cases assert the preregistered upper 95% endpoint at or below one
    And every static-equivalent timing is labeled open static-control and never asserted as build truth
