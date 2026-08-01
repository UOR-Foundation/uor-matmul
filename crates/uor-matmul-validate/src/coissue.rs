//! Experiment scaffolding: the scalar-port co-issue experiment.
//!
//! This is a measurement instrument, not a kernel --- the same standing
//! [`crate::float_tab`] has. No ID, no scenario, no shipped code, and nothing
//! here is a claim yet.
//!
//! The thesis under test is port multiplexing. The CG-11 census (llvm-mca,
//! znver4) reads the AVX2 `i8` tile kernel bound on `Zn4FP2` at IPC ~5.88
//! with the scalar integer ports idle in the model, so a scalar integer
//! stream summing into the same exact accumulator should co-issue beside it
//! for a single-digit-percent gain. The census's own caution is the reason
//! this is an experiment and not a kernel: if the two streams bind the same
//! port, the experiment has split one bottleneck into two queues for it, and
//! the only honest answer is a measurement.
//!
//! The scalar stream is Kronecker substitution over the integers, the
//! construction `isa/wasm.rs` documents for SIMD128, transcribed to `u64`
//! registers and written in safe Rust --- the experiment is whether the
//! out-of-order scheduler co-issues it, so the scheduling is left to LLVM:
//!
//! - Three elements of a `B` row are packed at [`SWAR_SPACING`]-bit fields of
//!   one `u64`, both operands biased to unsigned --- the same `+128` offset
//!   identity `AVX512_DPBUSD_I8_I32` uses, with the compensation paid at
//!   extraction: `sum(a*b) = sum(a'*b') - 128*sum(a') - 128*sum(b') +
//!   16384*d`.
//! - One `u64` multiply against a splatted scalar produces three products in
//!   disjoint fields, and [`SWAR_CHUNK`] products a field fit under the five
//!   guard bits --- the `wasm_swar_field_w8a8` row of `model/constants.toml`
//!   records the same derivation (`33 * 255^2` carries, so the chunk is 32).
//! - A column group's tail is padded with raw zeros, which bias to 128 and
//!   cancel exactly in the same identity, so any scalar width works.
//!
//! # What the timed path measures
//!
//! The first version of this harness packed panels and allocated per rep,
//! and the x86 artifact it produced measured *packing*, not ports ---
//! recorded as a harness defect in `MEASUREMENT-LOG.md`. The structure now
//! is the driver's: [`Coissue::prepare`] packs every panel and allocates
//! every buffer once, and [`Coissue::run`] --- the timed path --- borrows all
//! of it and allocates nothing, which a counting-allocator test asserts. The
//! byte-identity assertion stays inside the timed rep, against a reused
//! output buffer. The vector stream is the AVX2 `i8` tile kernel itself,
//! called through the kernels crate's public [`KernelSpec`] interface ---
//! not copied, never reimplemented. The measurement path is x86-only: an
//! experiment about x86 ports cannot run on aarch64, and it says so. The
//! byte-identity half runs everywhere, on whatever tile the host offers.

#[cfg(test)]
use uor_matmul::kernels::cached;
use uor_matmul::kernels::packed_slot;
use uor_matmul::kernels::KernelSpec;
#[cfg(test)]
use uor_matmul::Backend;

/// The field spacing, in bits: three fields to a `u64`, five guard bits a
/// field over the widest biased product. The derivation is the wasm
/// sequence's; `model/constants.toml`'s `wasm_swar_field_w8a8` row records it.
const SWAR_SPACING: u32 = 21;

/// The fields per `u64` multiplicand.
const SWAR_T: usize = 3;

/// The deepest run of products one field absorbs before extraction:
/// `floor(((1 << SWAR_SPACING) - 1) / (255 * 255))`, which is 32.
const SWAR_CHUNK: usize = 32;

/// One field's mask: `(1 << SWAR_SPACING) - 1`.
const FIELD_MASK: u64 = (1 << SWAR_SPACING) - 1;

/// The vector tile this experiment reads: the host's AVX2 `i8` spec where the
/// host has one, and the list's first --- the portable reference --- where it
/// does not. The bytes do not depend on the choice (every spec computes the
/// same integer); the measurement does, and it insists on the AVX2 one.
#[cfg(test)]
fn vector_spec() -> KernelSpec<i8, i32> {
    cached::available_i8()
        .find(|s| s.backend == Backend::Avx2)
        .or_else(|| cached::available_i8().next())
        .expect("the portable kernel is always present")
}

/// Pack the `rows x kc` tile at `(r0, p0)` of a row-major operand into the
/// layout the tile kernel reads, with the alphabet's zero past the operand's
/// extents --- padding is exact, which is `S8`'s whole point.
#[allow(clippy::too_many_arguments)]
fn pack_tile(
    src: &[i8],
    src_rows: usize,
    src_cols: usize,
    r0: usize,
    p0: usize,
    rows: usize,
    kc: usize,
    k_group: usize,
    out: &mut [i8],
) {
    for i in 0..rows {
        for p in 0..kc {
            let v = if r0 + i < src_rows && p0 + p < src_cols {
                src[(r0 + i) * src_cols + p0 + p]
            } else {
                0
            };
            out[packed_slot(p, i, rows, k_group)] = v;
        }
    }
}

/// One chunk of the scalar Kronecker stream over columns `[nv, n)`: three
/// fields to a `u64`, a chunk of `d` depths per extraction, accumulating into
/// `acc` at row stride `n`.
///
/// Everything is `u64` arithmetic on stack arrays: no allocation, no SIMD
/// intrinsics, no unsafe. Whether these instructions land on ports the vector
/// kernel leaves idle is the question the measurement answers; the
/// construction itself is an exact integer identity on any host.
#[allow(clippy::too_many_arguments)]
fn scalar_chunk(
    a: &[i8],
    b: &[i8],
    m: usize,
    k: usize,
    n: usize,
    nv: usize,
    c0: usize,
    d: usize,
    acc: &mut [i32],
) {
    let ns = n - nv;
    let groups = ns.div_ceil(SWAR_T);
    for g in 0..groups {
        let cols = SWAR_T.min(ns - g * SWAR_T);
        // One packed accumulator per A row, the chunk's field sums of the
        // biased multiplicand, and the chunk's biased scalar sums --- the
        // three terms the offset identity corrects with.
        let mut packed = [0u64; 16];
        let mut bsum = [0i64; SWAR_T];
        let mut asum = [0i64; 16];
        debug_assert!(m <= 16, "the instrument's row window is 16");
        for p in c0..c0 + d {
            // The multiplicand: three biased fields at `SWAR_SPACING`. A
            // column past the group's tail reads as raw zero --- biased 128,
            // which the correction cancels exactly, so the group needs no
            // remainder case.
            let mut w = 0u64;
            for (f, bs) in bsum.iter_mut().enumerate() {
                let raw = if f < cols {
                    i64::from(b[(nv + g * SWAR_T + f) * k + p]) + 128
                } else {
                    128
                };
                *bs += raw;
                w |= (raw as u64) << (SWAR_SPACING * f as u32);
            }
            for i in 0..m {
                let av = i64::from(a[i * k + p]) + 128;
                asum[i] += av;
                packed[i] = packed[i].wrapping_add(w.wrapping_mul(av as u64));
            }
        }
        // Extraction: `sum(a*b) = sum(a'*b') - 128*sum(a') - 128*sum(b') +
        // 16384*d`, every term an exact integer. A field holds at most
        // `CHUNK * 255^2 < 2^21`, a corrected chunk sum at most
        // `CHUNK * 128^2 < 2^31`, so the cast is the value.
        let step_bias = (d as i64) << 14;
        for i in 0..m {
            for f in 0..cols {
                let field = (packed[i] >> (SWAR_SPACING * f as u32)) & FIELD_MASK;
                let corrected = field as i64 - (asum[i] << 7) - (bsum[f] << 7) + step_bias;
                acc[i * n + nv + g * SWAR_T + f] += corrected as i32;
            }
        }
    }
}

/// One configuration, prepared: every panel packed once, every buffer owned
/// once. [`Coissue::run`] --- the timed path --- borrows all of it.
///
/// The vector panels are packed for the whole depth at preparation, and the
/// run reads per-chunk slices of them: the packed layout is `k`-major in
/// `k_group` runs, so a chunk aligned to `k_group` is one contiguous slice,
/// and the interleave stays one depth loop without any packing inside it.
pub struct Coissue {
    spec: KernelSpec<i8, i32>,
    m: usize,
    k: usize,
    n: usize,
    nv: usize,
    a: Vec<i8>,
    b: Vec<i8>,
    tiles_a: Vec<i8>,
    tiles_b: Vec<i8>,
    tile: Vec<i32>,
    out: Vec<i32>,
}

impl Coissue {
    /// Pack every panel and own every buffer, once. The vector columns are
    /// `[0, nv)`; `nv` must be a whole number of the tile's columns (the
    /// experiment's splits are chosen so), and the scalar stream takes the
    /// rest, padding its own tail exactly.
    pub fn prepare(
        spec: &KernelSpec<i8, i32>,
        a: &[i8],
        b: &[i8],
        m: usize,
        k: usize,
        n: usize,
        nv: usize,
    ) -> Self {
        assert_eq!(a.len(), m * k, "a is m x k");
        assert_eq!(b.len(), n * k, "b is n x k");
        assert!(
            nv <= n && nv.is_multiple_of(spec.nr),
            "the split must be a whole number of vector tiles"
        );
        let (mr, nr, kg) = (spec.mr, spec.nr, spec.k_group.max(1));
        let k_pad = k.div_ceil(kg) * kg;
        let m_tiles = m.div_ceil(mr);
        let v_tiles = nv / nr;
        let mut tiles_a = vec![0i8; m_tiles * mr * k_pad];
        for mt in 0..m_tiles {
            pack_tile(
                a,
                m,
                k,
                mt * mr,
                0,
                mr,
                k_pad,
                kg,
                &mut tiles_a[mt * mr * k_pad..][..mr * k_pad],
            );
        }
        let mut tiles_b = vec![0i8; v_tiles * nr * k_pad];
        for nt in 0..v_tiles {
            pack_tile(
                b,
                n,
                k,
                nt * nr,
                0,
                nr,
                k_pad,
                kg,
                &mut tiles_b[nt * nr * k_pad..][..nr * k_pad],
            );
        }
        Self {
            spec: *spec,
            m,
            k,
            n,
            nv,
            a: a.to_vec(),
            b: b.to_vec(),
            tiles_a,
            tiles_b,
            tile: vec![0i32; mr * nr],
            out: vec![0i32; m * n],
        }
    }

    /// One full pass over the reduction: the vector tile kernel over the
    /// pre-packed panels, the scalar Kronecker stream over the rest, in one
    /// depth loop.
    ///
    /// The timed path. It borrows every byte it touches and allocates none ---
    /// the counting-allocator test asserts the count is zero. The output is
    /// zeroed per pass, the same contract the epilogue's `beta = 0` keeps.
    pub fn run(&mut self) {
        self.out.fill(0);
        let (mr, nr, kg) = (self.spec.mr, self.spec.nr, self.spec.k_group.max(1));
        let k_pad = self.k.div_ceil(kg) * kg;
        let m_tiles = self.tiles_a.len() / (mr * k_pad);
        let v_tiles = self.tiles_b.len() / (nr * k_pad);
        let mut c0 = 0usize;
        while c0 < self.k {
            let d = SWAR_CHUNK.min(self.k - c0);
            let kc = d.div_ceil(kg) * kg;
            for mt in 0..m_tiles {
                // One contiguous slice of the pre-packed panel per chunk:
                // `packed_slot` is `k`-major in `k_group` runs, so depths
                // `[c0, c0 + kc)` live at `[c0 * mr, (c0 + kc) * mr)`.
                let pa = &self.tiles_a[mt * mr * k_pad + c0 * mr..][..mr * kc];
                for nt in 0..v_tiles {
                    let pb = &self.tiles_b[nt * nr * k_pad + c0 * nr..][..nr * kc];
                    self.spec.mac_tile(kc, pa, pb, &mut self.tile);
                    for i in 0..mr {
                        for j in 0..nr {
                            if mt * mr + i < self.m {
                                self.out[(mt * mr + i) * self.n + nt * nr + j] +=
                                    self.tile[i * nr + j];
                            }
                        }
                    }
                }
            }
            scalar_chunk(
                &self.a,
                &self.b,
                self.m,
                self.k,
                self.n,
                self.nv,
                c0,
                d,
                &mut self.out,
            );
            c0 += d;
        }
    }

    /// The output of the last [`Coissue::run`].
    pub fn out(&self) -> &[i32] {
        &self.out
    }
}

/// The scalar-only configuration, for the correctness half that runs on any
/// host: no tile kernel at all.
pub fn scalar_only(a: &[i8], b: &[i8], m: usize, k: usize, n: usize) -> Vec<i32> {
    let mut acc = vec![0i32; m * n];
    let mut c0 = 0usize;
    while c0 < k {
        let d = SWAR_CHUNK.min(k - c0);
        scalar_chunk(a, b, m, k, n, 0, c0, d, &mut acc);
        c0 += d;
    }
    acc
}

/// The schoolbook reference, for the byte-identity assertions.
#[cfg(test)]
fn reference(a: &[i8], b: &[i8], m: usize, k: usize, n: usize) -> Vec<i32> {
    let mut acc = vec![0i32; m * n];
    for i in 0..m {
        for j in 0..n {
            acc[i * n + j] = (0..k)
                .map(|p| i32::from(a[i * k + p]) * i32::from(b[j * k + p]))
                .sum();
        }
    }
    acc
}

/// A deterministic fill at the full `i8` range, extremes included.
#[cfg(test)]
fn fill_full(len: usize, salt: u64) -> Vec<i8> {
    let mut s = salt | 1;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 16) as i8
        })
        .collect()
}

/// The same, clamped to the W4A8 bound.
#[cfg(test)]
fn fill_w4a8(len: usize, salt: u64) -> Vec<i8> {
    fill_full(len, salt)
        .into_iter()
        .map(|x| (x % 8).clamp(-7, 7))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    // What the timed path allocates, counted. The instrument's discipline is
    // the driver's: everything the run needs exists before the run.
    #[global_allocator]
    static COUNTING: crate::counting::Counting = crate::counting::Counting;

    /// Byte-identity for the scalar stream alone and for the interleaved
    /// co-issue, at the alphabet's extremes and at three depths. Runs on any
    /// host: the construction is exact integer arithmetic, and the vector
    /// part reads whatever tile the host offers, which is exact by `CB-01`.
    #[test]
    fn the_streams_agree_with_the_reference_at_the_extremes() {
        for (label, fill) in [
            ("full", fill_full as fn(usize, u64) -> Vec<i8>),
            ("w4a8", fill_w4a8),
        ] {
            for &k in &[64usize, 1024, 16384] {
                for &m in &[4usize, 8, 16] {
                    let n = 64usize;
                    let a = fill(m * k, (m * k) as u64 + 1);
                    let b = fill(n * k, (n * k) as u64 + 2);
                    let want = reference(&a, &b, m, k, n);
                    assert_eq!(
                        scalar_only(&a, &b, m, k, n),
                        want,
                        "{label} {m}x{k}x{n}: the scalar stream disagrees"
                    );
                    let spec = vector_spec();
                    for nv in [48usize, 32] {
                        let mut run = Coissue::prepare(&spec, &a, &b, m, k, n, nv);
                        run.run();
                        assert_eq!(
                            run.out(),
                            &want[..],
                            "{label} {m}x{k}x{n} split {nv}|{}: the co-issue disagrees",
                            n - nv
                        );
                        let mut vec_run = Coissue::prepare(&spec, &a, &b, m, k, n, n);
                        vec_run.run();
                        assert_eq!(
                            vec_run.out(),
                            &want[..],
                            "{label} {m}x{k}x{n}: the vector-only run disagrees"
                        );
                    }
                }
            }
        }
    }

    /// The co-issue at the kernel-dominated width: `n = 256`, splits
    /// 192|64 and 128|128, at two depths.
    #[test]
    fn the_co_issue_agrees_at_the_wide_split() {
        let spec = vector_spec();
        for &k in &[64usize, 1024] {
            for &m in &[4usize, 16] {
                let n = 256usize;
                let a = fill_full(m * k, (m * k) as u64 + 3);
                let b = fill_full(n * k, (n * k) as u64 + 4);
                let want = reference(&a, &b, m, k, n);
                for nv in [192usize, 128] {
                    let mut run = Coissue::prepare(&spec, &a, &b, m, k, n, nv);
                    run.run();
                    assert_eq!(
                        run.out(),
                        &want[..],
                        "{m}x{k}x{n} split {nv}|{}: the co-issue disagrees",
                        n - nv
                    );
                }
            }
        }
    }

    /// The timed path allocates nothing, counted.
    ///
    /// The first version of this harness packed and allocated per rep, and
    /// measured the packing. This is the assertion that keeps the fixed
    /// structure honest: everything the run needs exists before the run, so
    /// the clock reads the reduction and nothing else.
    #[test]
    fn the_timed_path_allocates_nothing() {
        let (m, k, n) = (4usize, 1024usize, 64usize);
        let a = fill_full(m * k, 1);
        let b = fill_full(n * k, 2);
        let spec = vector_spec();
        let mut run = Coissue::prepare(&spec, &a, &b, m, k, n, 48);
        let ((), reading) = crate::counting::measure(|| run.run());
        assert_eq!(
            reading.allocations, 0,
            "the timed path must allocate nothing ({reading:?})"
        );
        assert_eq!(reading.bytes, 0);
    }

    /// The measurement: vector-only against the co-issued split, at i8 full
    /// range and at W4A8, printing the ratio per configuration.
    ///
    /// x86-only: the thesis is about x86 ports, and off-x86 this declines with
    /// the reason printed. The figures are `open` --- measured, reported,
    /// never asserted; the assertions inside are correctness, not time.
    #[test]
    #[ignore = "a measurement harness: `just coissue`"]
    fn coissue_sweep_times_the_split_streams() {
        if !cfg!(target_arch = "x86_64") {
            eprintln!(
                "coissue: this experiment measures whether a scalar integer stream \
                 co-issues beside the AVX2 tile kernel on x86 ports; there is no \
                 x86 here, so the measurement declines. The scalar stream's \
                 byte-identity is covered by the always-on tests."
            );
            return;
        }
        let Some(spec) = cached::available_i8().find(|s| s.backend == Backend::Avx2) else {
            eprintln!(
                "coissue: no AVX2 on this x86 host; the measurement declines \
                 (the thesis is about the AVX2 tile's ports)."
            );
            return;
        };
        let groups: [(&str, usize, [usize; 2]); 2] = [
            ("n=256 (kernel-dominated)", 256, [192, 128]),
            ("n=64 (small-tile regime)", 64, [48, 32]),
        ];
        for (group_name, n, splits) in groups {
            for (label, fill) in [
                ("full", fill_full as fn(usize, u64) -> Vec<i8>),
                ("w4a8", fill_w4a8),
            ] {
                for &k in &[64usize, 1024, 16384] {
                    for &m in &[4usize, 8, 16] {
                        let a = fill(m * k, (m * k) as u64 + 1);
                        let b = fill(n * k, (n * k) as u64 + 2);
                        let want = reference(&a, &b, m, k, n);
                        // Enough reps to see past the noise floor at the small
                        // shapes, bounded at the large ones.
                        let reps = (50_000_000usize / (m * k * n).max(1)).max(2);
                        for nv in splits {
                            let mut vec_run = Coissue::prepare(&spec, &a, &b, m, k, n, n);
                            let started = Instant::now();
                            for _ in 0..reps {
                                vec_run.run();
                                assert_eq!(
                                    vec_run.out(),
                                    &want[..],
                                    "the timed run must be correct"
                                );
                            }
                            let vec_t = started.elapsed();
                            let mut co_run = Coissue::prepare(&spec, &a, &b, m, k, n, nv);
                            let started = Instant::now();
                            for _ in 0..reps {
                                co_run.run();
                                assert_eq!(
                                    co_run.out(),
                                    &want[..],
                                    "the timed run must be correct"
                                );
                            }
                            let co_t = started.elapsed();
                            eprintln!(
                                "coissue {group_name} {label} {m}x{k}x{n} split {nv}|{}: \
                                 vector {vec_t:?}, co-issued {co_t:?}, ratio {:.3} ({reps} reps)",
                                n - nv,
                                co_t.as_secs_f64() / vec_t.as_secs_f64(),
                            );
                        }
                    }
                }
            }
        }
    }
}
