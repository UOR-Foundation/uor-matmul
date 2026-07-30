//! `CG-17` (open): achieved MACs per second of the `i64x2` SWAR broadcast
//! sequence against the `i32x4` dot-with-extends sequence and the portable
//! reference, on wasm32-wasip1 under wasmtime.
//!
//! The SWAR sequence packs three biased `B` elements at 21-bit spacing in each
//! 64-bit lane and multiplies by one broadcast scalar: six products per
//! `i64x2.mul` against the dot sequence's eight per `i32x4.dot_i16x8_s`, but
//! with no widening extends on the operand path. Which side of that trade the
//! field packing lands on is the thing measured here, and either answer is the
//! finding --- the sequence is an ordinary integer identity, not something
//! this library is needed for, and if it loses, selection keeps the incumbent
//! and the loss is the record.
//!
//! Every figure is `open`: printed, never asserted. What *is* asserted, inside
//! each timed run, is byte-identity with the portable reference --- a speed
//! measured on the wrong bytes is not a measurement.
//!
//! wasm has no `llvm-mca` scheduling model and no native runner on the
//! development host, so this harness is the only measurement of these
//! sequences there is: `just swar-sweep` runs it under `wasmtime`, in release,
//! where a throughput figure means something.

use std::time::Instant;

use uor_matmul_core::{as_alphabet_full, dot_ref};
use uor_matmul_kernels::{packed_slot, portable_i8, KernelSpec};

/// Deterministic fill, the same recorded generator the parity tests use.
fn fill<T, F: Fn(i64) -> T>(len: usize, salt: u64, map: F) -> Vec<T> {
    let mut s = 0x243F_6A88_85A3_08D3u64 ^ salt;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            map((s >> 33) as i64)
        })
        .collect()
}

/// The lanes' reference: the core's own accumulation, read through the spec's
/// panel layout, so a kernel that misreads its panel is measured on nothing.
fn reference(spec: &KernelSpec<i8, i32>, kc: usize, pa: &[i8], pb: &[i8]) -> Vec<i32> {
    let mut out = vec![0i32; spec.mr * spec.nr];
    for i in 0..spec.mr {
        let a: Vec<i8> = (0..kc)
            .map(|p| pa[packed_slot(p, i, spec.mr, spec.k_group)])
            .collect();
        for j in 0..spec.nr {
            let b: Vec<i8> = (0..kc)
                .map(|p| pb[packed_slot(p, j, spec.nr, spec.k_group)])
                .collect();
            out[i * spec.nr + j] = dot_ref(as_alphabet_full(&a), as_alphabet_full(&b)) as i32;
        }
    }
    out
}

/// Pack the dense fills into one spec's panel layout, at one alphabet.
fn pack(spec: &KernelSpec<i8, i32>, kc: usize, bound7: bool, salt: u64) -> (Vec<i8>, Vec<i8>) {
    let map = move |v: i64| {
        if bound7 {
            ((v as u8 % 15) as i8) - 7
        } else {
            v as i8
        }
    };
    let rows: Vec<Vec<i8>> = (0..spec.mr)
        .map(|i| fill(kc, salt ^ i as u64, map))
        .collect();
    let cols: Vec<Vec<i8>> = (0..spec.nr)
        .map(|j| fill(kc, salt ^ 0x5A ^ j as u64, map))
        .collect();
    let mut pa = vec![0i8; spec.mr * kc];
    let mut pb = vec![0i8; spec.nr * kc];
    for p in 0..kc {
        for (i, row) in rows.iter().enumerate() {
            pa[packed_slot(p, i, spec.mr, spec.k_group)] = row[p];
        }
        for (j, col) in cols.iter().enumerate() {
            pb[packed_slot(p, j, spec.nr, spec.k_group)] = col[p];
        }
    }
    (pa, pb)
}

/// Best of as many repetitions as fit a fixed budget, in seconds per rep.
///
/// The minimum is the run least interfered with; the budget is wall-clock so a
/// fast point and a slow one both get enough samples. The same discipline the
/// other sweeps use, at a shorter budget: the sweep has eighteen points and
/// the runner's start-up dominates anything shorter.
fn best(mut run: impl FnMut()) -> f64 {
    const BUDGET: f64 = 0.15;
    let mut best = f64::INFINITY;
    let mut spent = 0.0;
    loop {
        let s = Instant::now();
        run();
        let t = s.elapsed().as_secs_f64();
        best = best.min(t);
        spent += t;
        if spent >= BUDGET {
            return best;
        }
    }
}

/// One spec on one packed panel pair: seconds per call.
///
/// The tile contract is overwrite, so every call must leave the one-call sum
/// --- asserted against the reference inside the timed region, so the number
/// is for the right bytes.
fn time_spec(spec: &KernelSpec<i8, i32>, kc: usize, pa: &[i8], pb: &[i8], want: &[i32]) -> f64 {
    // Calls per batch, so one batch is the same product count at every depth.
    let reps = ((1usize << 24) / (spec.mr * spec.nr * kc).max(1)).max(1);
    let mut acc = vec![0i32; spec.mr * spec.nr];
    let secs = best(|| {
        for _ in 0..reps {
            spec.mac_tile(kc, pa, pb, &mut acc);
            // Byte-identity with the reference, inside the timed region: the
            // tile contract is overwrite, so every call must leave exactly the
            // one-call sum.
            assert_eq!(acc, want, "a timed call must give the reference's bytes");
        }
    });
    secs / reps as f64
}

#[test]
#[ignore = "a release-mode sweep under wasmtime: `just swar-sweep`"]
fn i64x2_swar_against_dot_and_portable_cg_17() {
    if !uor_matmul_kernels::isa::wasm::simd128_available() {
        eprintln!(
            "CG-17: no baseline SIMD128 on this host; the sweep only measures on wasm32-wasip1"
        );
        return;
    }
    let swar = uor_matmul_kernels::isa::wasm::SIMD128_SWAR_I8_I32;
    let dot = uor_matmul_kernels::isa::wasm::SIMD128_I8_I32;
    let portable = portable_i8();
    let mmacs = |spec: &KernelSpec<i8, i32>, kc: usize, secs: f64| {
        (spec.mr * spec.nr * kc) as f64 / secs / 1e6
    };
    println!();
    println!("# CG-17 (open): i64x2 SWAR broadcast vs i32x4 dot-with-extends vs portable, Mmac/s");
    println!(
        "# host: {}-{} under the configured runner; best of a 0.15 s budget per point;",
        std::env::consts::ARCH,
        std::env::consts::OS
    );
    println!("# byte-identity with the portable reference is asserted inside every timed run;");
    println!(
        "# the portable kernel is never optimized (R6): its column is the baseline, not a target"
    );
    for kc in [64usize, 1024, 16384] {
        println!();
        println!(
            "| k = {kc} | portable ({}x{}) | dot ({}x{}) | swar ({}x{}) | swar / dot |",
            portable.mr, portable.nr, dot.mr, dot.nr, swar.mr, swar.nr
        );
        println!("| --- | --- | --- | --- | --- |");
        for (label, bound7) in [("full i8", false), ("W4A8 (bound 7)", true)] {
            let mut row = [0.0f64; 3];
            for (slot, spec) in [(0usize, &portable), (1, &dot), (2, &swar)] {
                let kc = kc.div_ceil(spec.k_group) * spec.k_group;
                let (pa, pb) = pack(spec, kc, bound7, kc as u64);
                let want = reference(spec, kc, &pa, &pb);
                row[slot] = time_spec(spec, kc, &pa, &pb, &want);
            }
            println!(
                "| {label} | {:.1} | {:.1} | {:.1} | {:.2}x |",
                mmacs(&portable, kc, row[0]),
                mmacs(&dot, kc, row[1]),
                mmacs(&swar, kc, row[2]),
                mmacs(&swar, kc, row[2]) / mmacs(&dot, kc, row[1]),
            );
        }
    }
}
