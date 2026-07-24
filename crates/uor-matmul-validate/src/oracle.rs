//! Oracle adapters and the agreement verdict.

/// What a comparison against an external library established.
///
/// Three outcomes, not two. The third applies to *float* oracles only: a
/// classical `f32` GEMM computes an order-dependent approximation of the value
/// we compute exactly, so recording how far off it is beats both failing the
/// gate and quietly skipping. An **integer** oracle never reaches it --- under
/// `EncodeMode::Wrapping` there is no depth at which it may differ (§3.4).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Agreement {
    /// The oracle is exact for this case and returned our bytes.
    Exact,
    /// The oracle is not exact for this case. The deviation from our exact
    /// value is recorded and reported; this never fails the gate.
    OracleInexact {
        /// The largest absolute deviation observed.
        max_deviation: i128,
        /// Where it was.
        at_index: usize,
    },
    /// The oracle is exact for this case and did **not** return our bytes.
    ///
    /// A bug in one of the two libraries. This always fails the gate.
    Mismatch(String),
}

impl Agreement {
    /// Does this verdict fail the acceptance gate?
    pub fn is_failure(&self) -> bool {
        matches!(self, Self::Mismatch(_))
    }
}

/// One external library, adapted to a common shape.
pub trait Oracle {
    /// The conformance ID this oracle discharges.
    const ID: &'static str;
    /// The crate name, for the report.
    const CRATE: &'static str;

    /// The oracle's own `i32` product, row-major.
    fn product_i32(m: usize, k: usize, n: usize, a: &[i32], b: &[i32]) -> Vec<i32>;
}

/// `CX-01`, `CX-03`: `ndarray`.
///
/// `i32` satisfies `LinalgScalar`, and ndarray dispatches integer types to its
/// own generic kernel rather than to `matrixmultiply`, which is what makes this
/// an independent witness for the integer path.
#[cfg(feature = "ref-ndarray")]
pub struct NdArray;

#[cfg(feature = "ref-ndarray")]
impl Oracle for NdArray {
    const ID: &'static str = "CX-01";
    const CRATE: &'static str = "ndarray";

    fn product_i32(m: usize, k: usize, n: usize, a: &[i32], b: &[i32]) -> Vec<i32> {
        use ndarray::Array2;
        let a = Array2::from_shape_vec((m, k), a.to_vec()).expect("A conforms");
        let b = Array2::from_shape_vec((k, n), b.to_vec()).expect("B conforms");
        a.dot(&b).into_raw_vec_and_offset().0
    }
}

/// `CX-02`, `CX-04`: `nalgebra`.
///
/// An implementation lineage independent of ndarray's, which is why both are
/// registered rather than one.
#[cfg(feature = "ref-nalgebra")]
pub struct Nalgebra;

#[cfg(feature = "ref-nalgebra")]
impl Oracle for Nalgebra {
    const ID: &'static str = "CX-02";
    const CRATE: &'static str = "nalgebra";

    fn product_i32(m: usize, k: usize, n: usize, a: &[i32], b: &[i32]) -> Vec<i32> {
        use nalgebra::DMatrix;
        // nalgebra is column-major; transposing on the way in and out keeps the
        // comparison row-major on both sides without touching the arithmetic.
        let a = DMatrix::from_row_slice(m, k, a);
        let b = DMatrix::from_row_slice(k, n, b);
        let c = a * b;
        let mut out = vec![0i32; m * n];
        for i in 0..m {
            for j in 0..n {
                out[i * n + j] = c[(i, j)];
            }
        }
        out
    }
}

/// Compare our product against an integer oracle's.
///
/// There is no exactness parameter, because an integer oracle has no permitted
/// difference: under `EncodeMode::Wrapping` we reproduce its bytes at every
/// depth, by ring homomorphism (§3.4). A mismatch is a bug in one of the two
/// libraries.
pub fn compare(ours: &[i32], theirs: &[i32]) -> Agreement {
    match crate::bytes_equal(ours, theirs) {
        Ok(()) => Agreement::Exact,
        Err(detail) => Agreement::Mismatch(detail),
    }
}

/// Compare our exact float product against a classical GEMM's, and measure the
/// deviation.
///
/// Never a failure. A classical `f32` GEMM computes an order-dependent
/// approximation of the value we compute exactly, so what is interesting is how
/// far off it is, not whether it differs (N1, `CX-05` .. `CX-09`).
pub fn measure_deviation(ours: &[i32], theirs: &[i32]) -> Agreement {
    let mut max_deviation = 0i128;
    let mut at_index = 0usize;
    for (i, (x, y)) in ours.iter().zip(theirs).enumerate() {
        let d = (*x as i128 - *y as i128).abs();
        if d > max_deviation {
            max_deviation = d;
            at_index = i;
        }
    }
    Agreement::OracleInexact {
        max_deviation,
        at_index,
    }
}
