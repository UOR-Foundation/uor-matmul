//! `CS-05`: the raw-pointer face, over the strides that make it raw.
//!
//! The differential test in `uor-matmul-validate` compares this face against the
//! safe API on row-major operands. That leaves the two functions whose whole job
//! is the other cases --- `low` and `span`, which compute the window a strided
//! view reaches --- asserted by nothing, and it leaves them outside Miri, because
//! `uor-matmul-validate` is infrastructure that the Miri job does not run.
//!
//! These live in the crate that owns the `unsafe`, so `just miri` reaches them
//! (`CU-07`). Every case allocates a buffer of *exactly* the reachable span and
//! hands `sgemm` a pointer into it, so a window that is off by one element in
//! either direction is an out-of-bounds access rather than a silent read of a
//! neighbouring allocation. Under Miri that is reported; natively it is caught
//! only when the answer happens to change, which is why the buffers are exact.

use uor_matmul::{gemm_float, GemmOptions, Linear, MatView, MatViewMut, Strides, Triple};

/// The window an `rows x cols` view with these strides reaches: the lowest
/// offset (as a negative adjustment) and the element count from there.
///
/// Computed here independently of `raw.rs`'s own `low`/`span` --- by walking
/// every coordinate rather than by reasoning about signs --- so that the two
/// disagreeing is a test failure rather than a shared mistake.
fn window(rs: isize, cs: isize, rows: usize, cols: usize) -> (isize, usize) {
    if rows == 0 || cols == 0 {
        return (0, 0);
    }
    let mut lo = isize::MAX;
    let mut hi = isize::MIN;
    for i in 0..rows {
        for j in 0..cols {
            let at = i as isize * rs + j as isize * cs;
            lo = lo.min(at);
            hi = hi.max(at);
        }
    }
    (lo, (hi - lo) as usize + 1)
}

/// A buffer holding exactly the window, together with the strides that describe
/// it and the pointer the caller passes.
///
/// The buffer is indexed by *offset from the lowest one the strides reach*, which
/// is the layout `MatView::new` expects: it places its own origin at `-lo`, the
/// same number `raw_gemm` applies to the caller's pointer. So the safe face and
/// the raw face are handed the same window, not merely the same numbers --- and
/// the caller's base pointer for a negative stride is *inside* the allocation
/// rather than at its start.
///
/// Carrying `rs` and `cs` here rather than beside every call is what keeps the
/// signatures short enough to read; a coordinate means nothing without them.
struct Strided {
    buf: Vec<f32>,
    lo: isize,
    rs: isize,
    cs: isize,
}

impl Strided {
    /// Fill the window so that coordinate `(i, j)` holds `f(i, j)`, and leave
    /// every offset the view does not reach at a value that would be wrong if it
    /// were read: `NaN` propagates through an exact sum, so a stray read shows up
    /// as a `NaN` in the output rather than as a plausible number.
    fn new(
        rs: isize,
        cs: isize,
        rows: usize,
        cols: usize,
        mut f: impl FnMut(usize, usize) -> f32,
    ) -> Self {
        let (lo, span) = window(rs, cs, rows, cols);
        let mut buf = vec![f32::NAN; span];
        for i in 0..rows {
            for j in 0..cols {
                let at = (i as isize * rs + j as isize * cs - lo) as usize;
                buf[at] = f(i, j);
            }
        }
        Self { buf, lo, rs, cs }
    }

    /// A buffer of exactly the window, all zeros --- for an output view, where
    /// every reachable cell is written and no cell is read before it is.
    fn zeroed(rs: isize, cs: isize, rows: usize, cols: usize) -> Self {
        let (lo, span) = window(rs, cs, rows, cols);
        Self {
            buf: vec![0.0f32; span],
            lo,
            rs,
            cs,
        }
    }

    /// The strides, as the safe API names them.
    fn strides(&self) -> Strides {
        Strides {
            rs: self.rs,
            cs: self.cs,
        }
    }

    /// The pointer coordinate `(0, 0)` sits at: the base the raw face is handed.
    fn origin(&self) -> *const f32 {
        // `wrapping_offset` rather than `offset`: `-lo` is inside the allocation
        // by construction, but writing it as the wrapping form says so without
        // asking the reader to re-derive it.
        self.buf.as_ptr().wrapping_offset(-self.lo)
    }

    /// As [`Self::origin`], mutably.
    fn origin_mut(&mut self) -> *mut f32 {
        self.buf.as_mut_ptr().wrapping_offset(-self.lo)
    }

    /// The value at a matrix coordinate.
    fn at(&self, i: usize, j: usize) -> f32 {
        self.buf[(i as isize * self.rs + j as isize * self.cs - self.lo) as usize]
    }
}

/// The safe API's answer for the same operands and the same strides.
///
/// The comparison is against the views rather than against a recomputed sum,
/// because the raw face is not a second method (R13): it is a different way of
/// naming the same operands, and what `CS-05` claims is that the bytes agree.
///
/// Row-major output here, because what this compares is the *input* windows; the
/// strided-output case is compared cell by cell at the call site.
fn safely(shape: (usize, usize, usize), a: &Strided, b: &Strided, linear: &Linear) -> Vec<f32> {
    let (m, k, n) = shape;
    let mut out = vec![0.0f32; m * n];
    if m == 0 || n == 0 {
        return out;
    }
    let av = MatView::new(&a.buf, m, k, a.strides()).expect("the window fits");
    let bv = MatView::new(&b.buf, k, n, b.strides()).expect("the window fits");
    let cv = MatViewMut::row_major(&mut out, m, n).expect("row-major fits");
    let mut t = Triple::new(av, bv, cv).expect("conformant");
    gemm_float(&mut t, linear, GemmOptions::default());
    out
}

/// Every combination of stride sign, on operands whose buffers are exactly the
/// window they reach.
///
/// `low` and `span` exist for the negative cases, and before this test nothing
/// ran them: the differential test in `uor-matmul-validate` passes `k as isize`
/// and `1`, both positive, so `low` returned zero every time it was called and
/// `span` returned `rows * cols`.
///
/// Under Miri the *shapes* shrink and the stride pairs do not. That split is the
/// whole of the reduction and it is the one that keeps the claim: a stride sign
/// is a distinct path through `low` and `span` --- which branch on exactly that
/// --- while a shape only changes how many times the same path runs. All 21
/// stride pairs still run; two shapes carry them instead of four, and the two
/// kept are the degenerate `1x1x1` and one with every extent distinct, which is
/// what a transposed or off-by-one window would show up in.
///
/// The two dropped are the two largest, and under Miri a shape costs what its
/// whole product costs: `13x17x19` is forty times the multiply-accumulates of
/// `3x7x5`. Measured: all four shapes cost 680 s for these three tests alone,
/// against 1556 s for the rest of the Miri job and a 45-minute ceiling on the
/// runner --- so the facade would have been two fifths of the budget. Two shapes
/// cost 18.5 s, and the same three tests pass. That is the margin between a gate
/// that reports and the six-hour cancellations this one used to produce.
#[test]
fn every_stride_sign_reaches_exactly_its_window_cs_05() {
    let shapes: &[(usize, usize, usize)] = if cfg!(miri) {
        &[(1, 1, 1), (3, 7, 5)]
    } else {
        &[(1, 1, 1), (3, 7, 5), (4, 16, 6), (13, 17, 19)]
    };
    let mut checked = 0usize;
    for &(m, k, n) in shapes {
        for &(rsa, csa) in &[
            (k as isize, 1isize),   // row-major
            (1, m as isize),        // column-major
            (-(k as isize), 1),     // rows descending
            (k as isize, -1),       // columns descending
            (-(k as isize), -1),    // both
            (2 * k as isize, 2),    // padded, positive
            (-(2 * k as isize), 2), // padded, rows descending
        ] {
            for &(rsc, csc) in &[(n as isize, 1isize), (-(n as isize), 1), (n as isize, -1)] {
                let a = Strided::new(rsa, csa, m, k, |i, j| (i * k + j) as f32 * 0.5 - 3.0);
                let b = Strided::new(n as isize, 1, k, n, |i, j| (i * n + j) as f32 * -0.25 + 1.5);
                let mut c = Strided::zeroed(rsc, csc, m, n);

                let expected = safely((m, k, n), &a, &b, &Linear::OVERWRITE);

                // SAFETY: each buffer is exactly the window its strides reach and
                // the origin pointer is inside it; `c` is a separate allocation
                // from `a` and `b`; and no stride pair here maps two output
                // coordinates onto one cell, because each is a permutation of a
                // dense layout.
                unsafe {
                    uor_matmul::raw::sgemm(
                        m,
                        k,
                        n,
                        1,
                        a.origin(),
                        rsa,
                        csa,
                        b.origin(),
                        b.rs,
                        b.cs,
                        0,
                        c.origin_mut(),
                        rsc,
                        csc,
                    );
                }

                for i in 0..m {
                    for j in 0..n {
                        let got = c.at(i, j);
                        assert_eq!(
                            got,
                            expected[i * n + j],
                            "({m}x{k}x{n}) a=({rsa},{csa}) c=({rsc},{csc}) at ({i},{j})"
                        );
                    }
                }
                checked += 1;
            }
        }
    }
    // Seven `A` stride pairs by three `C` stride pairs, per shape. Written as the
    // product rather than as a literal so that adding a stride pair and
    // forgetting to run it cannot pass: the count has to move with the lists.
    assert_eq!(
        checked,
        shapes.len() * 7 * 3,
        "every shape crossed with every stride pair"
    );
}

/// A degenerate extent asks for a window of nothing, and gets one.
///
/// `m == 0` and `n == 0` return before the slices are built. `k == 0` does not:
/// it builds a zero-length slice for `a` and for `b`, which is only sound if the
/// pointer is non-null and aligned, and then multiplies a zero-width `a` by a
/// zero-height `b` --- whose exact sum is the empty sum, so `C` becomes `beta * C`
/// and nothing else. Miri checks the zero-length slices; the assertions check the
/// answer.
///
/// Every call below satisfies the documented contract on its own terms, which at
/// a degenerate extent is the whole difficulty: it is easy to write a test that
/// hands the raw face a buffer too small for what the arguments declare and then
/// passes because nothing was read. That asserts the early return by assuming it.
#[test]
fn a_degenerate_extent_is_a_window_of_nothing_cs_05() {
    // `k == 0`: the empty sum, so `alpha * 0 + beta * C`.
    let empty: [f32; 0] = [];
    let mut c = [1.0f32, 2.0, 3.0, 4.0];
    // SAFETY: `empty.as_ptr()` is non-null and aligned even at length zero,
    // which is what a zero-`k` call reads; `c` holds its 2x2 window exactly.
    unsafe {
        uor_matmul::raw::sgemm(
            2,
            0,
            2,
            1,
            empty.as_ptr(),
            0,
            1,
            empty.as_ptr(),
            2,
            1,
            3,
            c.as_mut_ptr(),
            2,
            1,
        );
    }
    assert_eq!(c, [3.0, 6.0, 9.0, 12.0], "beta absorbs the prior C");

    // `m == 0` and `n == 0` write nothing at all, and must not touch `c`.
    //
    // The operands are real here rather than empty. A zero-`n` call still
    // declares `a` to be a readable `m x 4`, and handing it a zero-length buffer
    // would be a contract violation that passes only because the implementation
    // returns before reading --- which is the claim under test, asserted by
    // assuming it. Eight elements cover every `(m, n)` below at `k = 4`.
    let operand = [1.0f32; 8];
    let mut untouched = [7.0f32; 4];
    for (m, n) in [(0usize, 2usize), (2, 0), (0, 0)] {
        // SAFETY: `operand` is a readable row-major `m x 4` and `4 x n` at every
        // extent below, since both need at most eight elements; `untouched` holds
        // the `m x n` output, which is empty. No output coordinate exists, so no
        // two of them alias.
        unsafe {
            uor_matmul::raw::sgemm(
                m,
                4,
                n,
                1,
                operand.as_ptr(),
                4,
                1,
                operand.as_ptr(),
                n as isize,
                1,
                0,
                untouched.as_mut_ptr(),
                n as isize,
                1,
            );
        }
        assert_eq!(untouched, [7.0f32; 4], "m={m} n={n} wrote nothing");
    }
}

/// `dgemm` is the same body at a different width, and says so.
///
/// Not a restatement of the test above: what it asserts is that the `f64`
/// instantiation reaches the same window arithmetic, which is a separate
/// monomorphisation and therefore a separate thing for Miri to check.
#[test]
fn the_f64_face_reaches_the_same_window_cs_05() {
    let (m, k, n) = (3usize, 5usize, 4usize);
    let rsa = -(k as isize);
    let (lo, span) = window(rsa, 1, m, k);
    let mut a = vec![f64::NAN; span];
    for i in 0..m {
        for j in 0..k {
            a[(i as isize * rsa + j as isize - lo) as usize] = (i * k + j) as f64 * 0.25 - 1.0;
        }
    }
    let b: Vec<f64> = (0..k * n).map(|i| i as f64 * -0.5 + 2.0).collect();
    let mut c = vec![0.0f64; m * n];

    let mut expected = vec![0.0f64; m * n];
    {
        let av = MatView::new(&a, m, k, Strides { rs: rsa, cs: 1 }).expect("the window fits");
        let bv = MatView::row_major(&b, k, n).expect("row-major fits");
        let cv = MatViewMut::row_major(&mut expected, m, n).expect("row-major fits");
        let mut t = Triple::new(av, bv, cv).expect("conformant");
        gemm_float(&mut t, &Linear::OVERWRITE, GemmOptions::default());
    }

    // SAFETY: `a` is exactly the window `(rsa, 1)` reaches over `m x k` and the
    // origin is `-lo` elements in; `b` and `c` are dense and exactly sized; all
    // three are distinct allocations.
    unsafe {
        uor_matmul::raw::dgemm(
            m,
            k,
            n,
            1,
            a.as_ptr().wrapping_offset(-lo),
            rsa,
            1,
            b.as_ptr(),
            n as isize,
            1,
            0,
            c.as_mut_ptr(),
            n as isize,
            1,
        );
    }
    assert_eq!(c, expected, "the f64 face agrees with the safe API");
}
