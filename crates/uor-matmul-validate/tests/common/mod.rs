//! What the symbol-path timing harnesses share: the recorded fills, the
//! timing discipline, and the shape grid. One copy, because a second spelling
//! of the same generator would let the harnesses drift apart while claiming
//! to measure the same points.

use std::time::Instant;

/// Deterministic fill, the same recorded generator the other harnesses use.
pub fn fill<T, F: Fn(u64) -> T>(len: usize, salt: u64, map: F) -> Vec<T> {
    let mut s = 0x243F_6A88_85A3_08D3u64 ^ salt;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            map(s >> 33)
        })
        .collect()
}

/// Best of as many repetitions as fit a fixed budget, in seconds per call.
///
/// The minimum is the run least interfered with; the budget is wall-clock so a
/// fast point and a slow one both get enough samples for the minimum to mean
/// something. The same discipline `oracle_sweep` uses.
pub fn best(mut run: impl FnMut()) -> f64 {
    const BUDGET: f64 = 0.35;
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

/// A fill whose decoded exponent span is exactly `span` binades, as the bridge
/// sweep's: significands drawn from `[2^23, 2^24)`, so the decode does not add
/// a bit of its own.
pub fn spanned(len: usize, salt: u64, span: i32) -> Vec<f32> {
    fill(len, salt, |v| {
        let s = if span == 0 {
            0
        } else {
            (v as i64 % span as i64) as i32
        };
        let m = 8_388_608 + v % 8_388_607;
        let x = m as f32 * 2.0f32.powi(s - span / 2);
        if v & 1 == 0 {
            -x
        } else {
            x
        }
    })
}

/// The `CG-14` sweep's fill: eighteen binades of span, which neither the lane
/// nor the bridge admits. Both decline it; the fill is here so the report
/// shows the boundary rather than silently measuring only the admitted case.
pub fn wide(len: usize, salt: u64) -> Vec<f32> {
    fill(len, salt, |v| {
        (v % 2048) as f32 * 2.0f32.powi((v % 19) as i32 - 9)
    })
}

/// The shapes: `CG-14`'s gemv and skinny five, where the bandwidth floor
/// binds, and the tabulation sweep's two, where the build's amortization over
/// `n` is what the table exists for.
pub const SHAPES: &[(usize, usize, usize)] = &[
    (1024, 1024, 1),
    (1, 1024, 1024),
    (1, 1_048_576, 1),
    (2048, 8, 2048),
    (8, 262_144, 8),
    (64, 1024, 4096),
    (64, 4096, 4096),
];
