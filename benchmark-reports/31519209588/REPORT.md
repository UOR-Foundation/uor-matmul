# Comparison Benchmark Report

Generated from Criterion artifacts.

## Run context

- Host: `GitHub Actions 1000031976` (Linux/X64)
- Workflow: `scaling` (run `31519209588`)
- Repository revision: `035e615649831100a1d52626e9662b8f2a9181b7`

- Completed Criterion measurements: **75**
- Required same-shape comparison measurements: **75 / 75**
- Timing unit: Criterion estimates converted for readability; intervals are 95% confidence intervals.

Ratios below are competitor time divided by the named primary. Values above 1× mean the primary is faster.


## Direct comparisons

These tables compare identical shapes. The speedup columns are competitor time divided by the named primary; a value above 1× means the primary operation is faster. GEMM sections also include a scaling-efficiency chart: normalized time per MAC stays flat at 1.00× for linear work scaling, while a rising line shows worsening efficiency as the problem grows.

**Scaling highlighted:** tropical results are reported in three views below: lane scaling, tie-dense witness scaling, and max-last witness scaling.

### i32 GEMM

Ratios are competitor time divided by uor-matmul time; greater than 1x means uor-matmul is faster.

| Shape | uor-matmul (ours) | handwritten | uor-matmul speedup vs handwritten | ndarray | uor-matmul speedup vs ndarray | nalgebra | uor-matmul speedup vs nalgebra |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16x16x16 | 1.377 µs | 3.270 µs | 2.37× | 2.876 µs | 2.09× | 1.510 µs | 1.10× |
| 128x128x128 | 148.9 µs | 3.036 ms | 20.39× | 2.798 ms | 18.79× | 488.7 µs | 3.28× |
| 32x256x512 | 288.1 µs | 5.148 ms | 17.87× | 7.674 ms | 26.63× | 1.210 ms | 4.20× |

### f32 GEMM

Ratios are competitor time divided by uor-matmul time; greater than 1x means uor-matmul is faster.

| Shape | uor-matmul (ours) | handwritten | uor-matmul speedup vs handwritten | matrixmultiply | uor-matmul speedup vs matrixmultiply | faer | uor-matmul speedup vs faer |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16x16x16 | 324.6 µs | 2.734 µs | 0.008× | 499.2 ns | 0.002× | 1.611 µs | 0.005× |
| 128x128x128 | 155.5 ms | 3.014 ms | 0.02× | 62.86 µs | 4.0e-4× | 125.0 µs | 8.0e-4× |
| 32x256x512 | 314.6 ms | 6.478 ms | 0.02× | 159.8 µs | 5.1e-4× | 388.4 µs | 0.001× |

### f64 GEMM

Ratios are competitor time divided by uor-matmul time; greater than 1x means uor-matmul is faster.

| Shape | uor-matmul (ours) | handwritten | uor-matmul speedup vs handwritten | matrixmultiply | uor-matmul speedup vs matrixmultiply | faer | uor-matmul speedup vs faer |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16x16x16 | 324.3 µs | 2.734 µs | 0.008× | 659.6 ns | 0.002× | 1.516 µs | 0.005× |
| 128x128x128 | 144.8 ms | 3.088 ms | 0.02× | 125.4 µs | 8.7e-4× | 173.3 µs | 0.001× |
| 32x256x512 | 289.5 ms | 12.56 ms | 0.04× | 417.1 µs | 0.001× | 644.0 µs | 0.002× |

### Tropical lane scaling

Ratios are competitor time divided by tropical lane time; greater than 1x means tropical lane is faster.

| Shape | tropical lane (ours) | ring lane | tropical lane speedup vs ring lane | ring packed | tropical lane speedup vs ring packed |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64x64x64 | 455.7 µs | 348.2 µs | 0.76× | 52.93 µs | 0.12× |
| 128x128x128 | 3.514 ms | 2.784 ms | 0.79× | 338.0 µs | 0.10× |
| 16x4096x16 | 1.648 ms | 1.314 ms | 0.80× | 276.0 µs | 0.17× |

### Tropical witness scaling · tie-dense

Ratios are competitor time divided by lexicographic time; greater than 1x means lexicographic is faster.

| Shape | lexicographic (ours) | compare-pass | lexicographic speedup vs compare-pass |
| ---: | ---: | ---: | ---: |
| tie-dense/64x64x64 | 998.4 µs | 532.1 µs | 0.53× |
| tie-dense/128x128x128 | 7.367 ms | 3.574 ms | 0.49× |
| tie-dense/16x4096x16 | 3.333 ms | 1.447 ms | 0.43× |

### Tropical witness scaling · max-last

Ratios are competitor time divided by lexicographic time; greater than 1x means lexicographic is faster.

| Shape | lexicographic (ours) | compare-pass | lexicographic speedup vs compare-pass |
| ---: | ---: | ---: | ---: |
| max-last/64x64x64 | 741.0 µs | 914.2 µs | 1.23× |
| max-last/128x128x128 | 5.702 ms | 6.985 ms | 1.22× |
| max-last/16x4096x16 | 2.728 ms | 3.283 ms | 1.20× |

### Public route

Ratios are competitor time divided by slice::gemm time; greater than 1x means slice::gemm is faster.

| Shape | slice::gemm (ours) | gemm_packed | slice::gemm speedup vs gemm_packed |
| ---: | ---: | ---: | ---: |
| 16x16x16 | 1.753 µs | 1.770 µs | 1.01× |
| 64x64x64 | 52.91 µs | 52.95 µs | 1.00× |
| 128x128x128 | 340.6 µs | 339.9 µs | 1.00× |

### Finite i8 lookup-table build

Ratios are competitor time divided by cpu-native time; greater than 1x means cpu-native is faster.

| Shape | cpu-native (ours) | portable | cpu-native speedup vs portable |
| ---: | ---: | ---: | ---: |
| space4096-blk16 | 1.009 ms | 1.082 ms | 1.07× |

### Modular-Strassen routes

Ratios are competitor time divided by packed time; greater than 1x means packed is faster.

| Shape | packed (ours) | level | packed speedup vs level |
| ---: | ---: | ---: | ---: |
| i8/512cubed | 16.92 ms | 16.75 ms | 0.99× |
| i8/1024cubed | 135.3 ms | 122.3 ms | 0.90× |
| i8/2048cubed | 1.098 s | 951.9 ms | 0.87× |
| i32/512cubed | 5.109 ms | 6.297 ms | 1.23× |
| i32/1024cubed | 41.60 ms | 40.36 ms | 0.97× |

## gemm_i32

| Benchmark | Mean | 95% confidence interval | Raw estimate |
| --- | ---: | ---: | --- |

| handwritten/128x128x128 | 3.036 ms | 3.032 ms - 3.040 ms | [estimates.json](criterion/gemm_i32/handwritten_128x128x128/new/estimates.json) |
| handwritten/16x16x16 | 3.270 µs | 3.249 µs - 3.300 µs | [estimates.json](criterion/gemm_i32/handwritten_16x16x16/new/estimates.json) |
| handwritten/32x256x512 | 5.148 ms | 5.144 ms - 5.153 ms | [estimates.json](criterion/gemm_i32/handwritten_32x256x512/new/estimates.json) |
| nalgebra/128x128x128 | 488.7 µs | 487.6 µs - 490.5 µs | [estimates.json](criterion/gemm_i32/nalgebra_128x128x128/new/estimates.json) |
| nalgebra/16x16x16 | 1.510 µs | 1.507 µs - 1.513 µs | [estimates.json](criterion/gemm_i32/nalgebra_16x16x16/new/estimates.json) |
| nalgebra/32x256x512 | 1.210 ms | 1.209 ms - 1.210 ms | [estimates.json](criterion/gemm_i32/nalgebra_32x256x512/new/estimates.json) |
| ndarray/128x128x128 | 2.798 ms | 2.791 ms - 2.804 ms | [estimates.json](criterion/gemm_i32/ndarray_128x128x128/new/estimates.json) |
| ndarray/16x16x16 | 2.876 µs | 2.873 µs - 2.879 µs | [estimates.json](criterion/gemm_i32/ndarray_16x16x16/new/estimates.json) |
| ndarray/32x256x512 | 7.674 ms | 7.672 ms - 7.676 ms | [estimates.json](criterion/gemm_i32/ndarray_32x256x512/new/estimates.json) |
| uor-matmul/128x128x128 | 148.9 µs | 148.6 µs - 149.4 µs | [estimates.json](criterion/gemm_i32/uor-matmul_128x128x128/new/estimates.json) |
| uor-matmul/16x16x16 | 1.377 µs | 1.375 µs - 1.380 µs | [estimates.json](criterion/gemm_i32/uor-matmul_16x16x16/new/estimates.json) |
| uor-matmul/32x256x512 | 288.1 µs | 287.6 µs - 288.9 µs | [estimates.json](criterion/gemm_i32/uor-matmul_32x256x512/new/estimates.json) |

## gemm_f32

| Benchmark | Mean | 95% confidence interval | Raw estimate |
| --- | ---: | ---: | --- |

| faer/128x128x128 | 125.0 µs | 124.7 µs - 125.3 µs | [estimates.json](criterion/gemm_f32/faer_128x128x128/new/estimates.json) |
| faer/16x16x16 | 1.611 µs | 1.610 µs - 1.612 µs | [estimates.json](criterion/gemm_f32/faer_16x16x16/new/estimates.json) |
| faer/32x256x512 | 388.4 µs | 388.1 µs - 388.7 µs | [estimates.json](criterion/gemm_f32/faer_32x256x512/new/estimates.json) |
| handwritten/128x128x128 | 3.014 ms | 3.010 ms - 3.019 ms | [estimates.json](criterion/gemm_f32/handwritten_128x128x128/new/estimates.json) |
| handwritten/16x16x16 | 2.734 µs | 2.723 µs - 2.745 µs | [estimates.json](criterion/gemm_f32/handwritten_16x16x16/new/estimates.json) |
| handwritten/32x256x512 | 6.478 ms | 6.477 ms - 6.480 ms | [estimates.json](criterion/gemm_f32/handwritten_32x256x512/new/estimates.json) |
| matrixmultiply/128x128x128 | 62.86 µs | 61.99 µs - 64.08 µs | [estimates.json](criterion/gemm_f32/matrixmultiply_128x128x128/new/estimates.json) |
| matrixmultiply/16x16x16 | 499.2 ns | 495.9 ns - 502.1 ns | [estimates.json](criterion/gemm_f32/matrixmultiply_16x16x16/new/estimates.json) |
| matrixmultiply/32x256x512 | 159.8 µs | 159.7 µs - 159.9 µs | [estimates.json](criterion/gemm_f32/matrixmultiply_32x256x512/new/estimates.json) |
| uor-matmul/128x128x128 | 155.5 ms | 155.5 ms - 155.6 ms | [estimates.json](criterion/gemm_f32/uor-matmul_128x128x128/new/estimates.json) |
| uor-matmul/16x16x16 | 324.6 µs | 322.5 µs - 327.6 µs | [estimates.json](criterion/gemm_f32/uor-matmul_16x16x16/new/estimates.json) |
| uor-matmul/32x256x512 | 314.6 ms | 312.5 ms - 316.7 ms | [estimates.json](criterion/gemm_f32/uor-matmul_32x256x512/new/estimates.json) |

## gemm_f64

| Benchmark | Mean | 95% confidence interval | Raw estimate |
| --- | ---: | ---: | --- |

| faer/128x128x128 | 173.3 µs | 172.9 µs - 173.8 µs | [estimates.json](criterion/gemm_f64/faer_128x128x128/new/estimates.json) |
| faer/16x16x16 | 1.516 µs | 1.515 µs - 1.517 µs | [estimates.json](criterion/gemm_f64/faer_16x16x16/new/estimates.json) |
| faer/32x256x512 | 644.0 µs | 643.6 µs - 644.4 µs | [estimates.json](criterion/gemm_f64/faer_32x256x512/new/estimates.json) |
| handwritten/128x128x128 | 3.088 ms | 3.087 ms - 3.089 ms | [estimates.json](criterion/gemm_f64/handwritten_128x128x128/new/estimates.json) |
| handwritten/16x16x16 | 2.734 µs | 2.721 µs - 2.751 µs | [estimates.json](criterion/gemm_f64/handwritten_16x16x16/new/estimates.json) |
| handwritten/32x256x512 | 12.56 ms | 12.51 ms - 12.66 ms | [estimates.json](criterion/gemm_f64/handwritten_32x256x512/new/estimates.json) |
| matrixmultiply/128x128x128 | 125.4 µs | 125.4 µs - 125.5 µs | [estimates.json](criterion/gemm_f64/matrixmultiply_128x128x128/new/estimates.json) |
| matrixmultiply/16x16x16 | 659.6 ns | 651.9 ns - 666.6 ns | [estimates.json](criterion/gemm_f64/matrixmultiply_16x16x16/new/estimates.json) |
| matrixmultiply/32x256x512 | 417.1 µs | 416.5 µs - 417.6 µs | [estimates.json](criterion/gemm_f64/matrixmultiply_32x256x512/new/estimates.json) |
| uor-matmul/128x128x128 | 144.8 ms | 144.6 ms - 145.1 ms | [estimates.json](criterion/gemm_f64/uor-matmul_128x128x128/new/estimates.json) |
| uor-matmul/16x16x16 | 324.3 µs | 323.9 µs - 324.8 µs | [estimates.json](criterion/gemm_f64/uor-matmul_16x16x16/new/estimates.json) |
| uor-matmul/32x256x512 | 289.5 ms | 289.4 ms - 289.7 ms | [estimates.json](criterion/gemm_f64/uor-matmul_32x256x512/new/estimates.json) |

## tropical

| Benchmark | Mean | 95% confidence interval | Raw estimate |
| --- | ---: | ---: | --- |

| lane/ring-packed/128x128x128 | 338.0 µs | 337.6 µs - 338.8 µs | [estimates.json](criterion/tropical/lane_ring-packed_128x128x128/new/estimates.json) |
| lane/ring-packed/16x4096x16 | 276.0 µs | 275.8 µs - 276.3 µs | [estimates.json](criterion/tropical/lane_ring-packed_16x4096x16/new/estimates.json) |
| lane/ring-packed/64x64x64 | 52.93 µs | 52.72 µs - 53.25 µs | [estimates.json](criterion/tropical/lane_ring-packed_64x64x64/new/estimates.json) |
| lane/ring/128x128x128 | 2.784 ms | 2.782 ms - 2.789 ms | [estimates.json](criterion/tropical/lane_ring_128x128x128/new/estimates.json) |
| lane/ring/16x4096x16 | 1.314 ms | 1.311 ms - 1.320 ms | [estimates.json](criterion/tropical/lane_ring_16x4096x16/new/estimates.json) |
| lane/ring/64x64x64 | 348.2 µs | 347.9 µs - 348.7 µs | [estimates.json](criterion/tropical/lane_ring_64x64x64/new/estimates.json) |
| lane/tropical/128x128x128 | 3.514 ms | 3.498 ms - 3.541 ms | [estimates.json](criterion/tropical/lane_tropical_128x128x128/new/estimates.json) |
| lane/tropical/16x4096x16 | 1.648 ms | 1.642 ms - 1.657 ms | [estimates.json](criterion/tropical/lane_tropical_16x4096x16/new/estimates.json) |
| lane/tropical/64x64x64 | 455.7 µs | 454.7 µs - 457.2 µs | [estimates.json](criterion/tropical/lane_tropical_64x64x64/new/estimates.json) |
| witness/compare-pass/max-last/128x128x128 | 6.985 ms | 6.976 ms - 6.998 ms | [estimates.json](criterion/tropical/witness_compare-pass_max-last_128x128x128/new/estimates.json) |
| witness/compare-pass/max-last/16x4096x16 | 3.283 ms | 3.280 ms - 3.286 ms | [estimates.json](criterion/tropical/witness_compare-pass_max-last_16x4096x16/new/estimates.json) |
| witness/compare-pass/max-last/64x64x64 | 914.2 µs | 913.4 µs - 915.0 µs | [estimates.json](criterion/tropical/witness_compare-pass_max-last_64x64x64/new/estimates.json) |
| witness/compare-pass/tie-dense/128x128x128 | 3.574 ms | 3.571 ms - 3.578 ms | [estimates.json](criterion/tropical/witness_compare-pass_tie-dense_128x128x128/new/estimates.json) |
| witness/compare-pass/tie-dense/16x4096x16 | 1.447 ms | 1.443 ms - 1.453 ms | [estimates.json](criterion/tropical/witness_compare-pass_tie-dense_16x4096x16/new/estimates.json) |
| witness/compare-pass/tie-dense/64x64x64 | 532.1 µs | 530.5 µs - 534.5 µs | [estimates.json](criterion/tropical/witness_compare-pass_tie-dense_64x64x64/new/estimates.json) |
| witness/lexicographic/max-last/128x128x128 | 5.702 ms | 5.698 ms - 5.710 ms | [estimates.json](criterion/tropical/witness_lexicographic_max-last_128x128x128/new/estimates.json) |
| witness/lexicographic/max-last/16x4096x16 | 2.728 ms | 2.726 ms - 2.732 ms | [estimates.json](criterion/tropical/witness_lexicographic_max-last_16x4096x16/new/estimates.json) |
| witness/lexicographic/max-last/64x64x64 | 741.0 µs | 739.8 µs - 743.0 µs | [estimates.json](criterion/tropical/witness_lexicographic_max-last_64x64x64/new/estimates.json) |
| witness/lexicographic/tie-dense/128x128x128 | 7.367 ms | 7.356 ms - 7.386 ms | [estimates.json](criterion/tropical/witness_lexicographic_tie-dense_128x128x128/new/estimates.json) |
| witness/lexicographic/tie-dense/16x4096x16 | 3.333 ms | 3.331 ms - 3.338 ms | [estimates.json](criterion/tropical/witness_lexicographic_tie-dense_16x4096x16/new/estimates.json) |
| witness/lexicographic/tie-dense/64x64x64 | 998.4 µs | 997.6 µs - 999.7 µs | [estimates.json](criterion/tropical/witness_lexicographic_tie-dense_64x64x64/new/estimates.json) |

## public_api

| Benchmark | Mean | 95% confidence interval | Raw estimate |
| --- | ---: | ---: | --- |

| gemm_packed/128x128x128 | 339.9 µs | 339.5 µs - 340.8 µs | [estimates.json](criterion/public_api/gemm_packed_128x128x128/new/estimates.json) |
| gemm_packed/16x16x16 | 1.770 µs | 1.768 µs - 1.772 µs | [estimates.json](criterion/public_api/gemm_packed_16x16x16/new/estimates.json) |
| gemm_packed/64x64x64 | 52.95 µs | 52.87 µs - 53.04 µs | [estimates.json](criterion/public_api/gemm_packed_64x64x64/new/estimates.json) |
| slice::gemm/128x128x128 | 340.6 µs | 340.2 µs - 341.1 µs | [estimates.json](criterion/public_api/slice__gemm_128x128x128/new/estimates.json) |
| slice::gemm/16x16x16 | 1.753 µs | 1.749 µs - 1.759 µs | [estimates.json](criterion/public_api/slice__gemm_16x16x16/new/estimates.json) |
| slice::gemm/64x64x64 | 52.91 µs | 52.83 µs - 52.99 µs | [estimates.json](criterion/public_api/slice__gemm_64x64x64/new/estimates.json) |

## lookup_build

| Benchmark | Mean | 95% confidence interval | Raw estimate |
| --- | ---: | ---: | --- |

| build/cpu-native/space4096-blk16 | 1.009 ms | 1.008 ms - 1.009 ms | [estimates.json](criterion/lookup_build/build_cpu-native_space4096-blk16/new/estimates.json) |
| build/portable/space4096-blk16 | 1.082 ms | 1.081 ms - 1.082 ms | [estimates.json](criterion/lookup_build/build_portable_space4096-blk16/new/estimates.json) |

## modular_strassen

| Benchmark | Mean | 95% confidence interval | Raw estimate |
| --- | ---: | ---: | --- |

| level/i32/1024cubed | 40.36 ms | 40.17 ms - 40.61 ms | [estimates.json](criterion/modular_strassen/level_i32_1024cubed/new/estimates.json) |
| level/i32/512cubed | 6.297 ms | 6.290 ms - 6.307 ms | [estimates.json](criterion/modular_strassen/level_i32_512cubed/new/estimates.json) |
| level/i8/1024cubed | 122.3 ms | 121.9 ms - 122.9 ms | [estimates.json](criterion/modular_strassen/level_i8_1024cubed/new/estimates.json) |
| level/i8/2048cubed | 951.9 ms | 950.8 ms - 953.3 ms | [estimates.json](criterion/modular_strassen/level_i8_2048cubed/new/estimates.json) |
| level/i8/512cubed | 16.75 ms | 16.74 ms - 16.76 ms | [estimates.json](criterion/modular_strassen/level_i8_512cubed/new/estimates.json) |
| packed/i32/1024cubed | 41.60 ms | 41.32 ms - 41.93 ms | [estimates.json](criterion/modular_strassen/packed_i32_1024cubed/new/estimates.json) |
| packed/i32/512cubed | 5.109 ms | 5.099 ms - 5.120 ms | [estimates.json](criterion/modular_strassen/packed_i32_512cubed/new/estimates.json) |
| packed/i8/1024cubed | 135.3 ms | 135.1 ms - 135.5 ms | [estimates.json](criterion/modular_strassen/packed_i8_1024cubed/new/estimates.json) |
| packed/i8/2048cubed | 1.098 s | 1.095 s - 1.102 s | [estimates.json](criterion/modular_strassen/packed_i8_2048cubed/new/estimates.json) |
| packed/i8/512cubed | 16.92 ms | 16.91 ms - 16.94 ms | [estimates.json](criterion/modular_strassen/packed_i8_512cubed/new/estimates.json) |

