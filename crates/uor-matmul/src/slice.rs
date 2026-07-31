//! The slice face (`CS-08`).
//!
//! A GEMM in the shape every other library spells it --- `(m, k, n)`, leading
//! dimensions, `alpha` and `beta` --- over safe slices, so that calling this
//! library does not require knowing what an [`Alphabet`], a [`MatView`] or a
//! [`Triple`] is.
//!
//! It is not a second path (R13). Every function here builds the same two views
//! and the same triple the view API takes and calls the same driver, which is
//! what `CS-08` asserts byte for byte. What it saves is the ceremony, not the
//! arithmetic.
//!
//! # Leading dimensions
//!
//! `lda`, `ldb` and `ldc` are elements between consecutive rows, as in a
//! row-major BLAS call: `lda == k` is the contiguous case, and `lda > k` names a
//! sub-matrix of a wider buffer. A slice must therefore hold
//! `(rows - 1) * ld + cols` elements. Arbitrary *strides* --- negative, zero,
//! transposed --- are the view API's business and remain available there (S7,
//! `CS-02`); this face is the rectangular subset every GEMM caller already has a
//! word for.
//!
//! # Scratch
//!
//! The library never allocates (R7), so the working memory is the caller's here
//! as everywhere. The offer decides which factorization of the one identity
//! runs --- never what the bytes are (`CD-22`). There are two offers and two
//! named plans ([`uor_matmul_gemm::workspace_report`] prices both, in bytes):
//!
//! - The **suggested** plan: [`suggested_scratch`] panel elements, no
//!   accumulators. The panels hold the whole of `k`, so the blocked traversal
//!   runs full-depth wherever the lane reaches --- and the panel count grows
//!   with `k`.
//! - The **bounded** plan: one `KC` chunk of panels plus the accumulator block
//!   [`suggested_accumulators`] sizes. The reduction is walked in chunks and
//!   the exact partial sums live in the accumulator block between chunks, so
//!   the cost stops growing with `k`. [`gemm_ex_full`] and [`gemm_full`] spell
//!   both halves; [`gemm`] and [`gemm_ex`] are the panel-only half.
//!
//! `&mut []` everywhere is valid and gives the same bytes (`CD-04`, `CD-10`);
//! only the chunking changes.
//!
//! [`suggested_scratch`]: uor_matmul_gemm::suggested_scratch
//! [`suggested_accumulators`]: uor_matmul_gemm::suggested_accumulators
//!
//! ```
//! # use uor_matmul::{slice, Shape, suggested_scratch};
//! let a = [1i8, 2, 3, 4];
//! let b = [5i8, 6, 7, 8];
//! let mut c = [0i32; 4];
//! let mut scratch = vec![0i8; suggested_scratch(Shape { m: 2, k: 2, n: 2 })];
//!
//! slice::gemm(2, 2, 2, &a, &b, &mut c, &mut scratch).unwrap();
//! assert_eq!(c, [19, 22, 43, 50]);
//! ```

use uor_matmul_core::{
    as_alphabet_full, as_alphabet_full_mut, AccOf, Element, EncodeFrom, FloatElement, MatView,
    MatViewMut, NotAProduct, PackedCode, Strides, Triple,
};
use uor_matmul_gemm::epilogue::{AbsorbPrior, PlaceAt, ScaleExact};
use uor_matmul_gemm::float::SignedPlace;
use uor_matmul_gemm::{
    gemm_auto as view_gemm, gemm_float_packed, GemmOptions, Kernelized, Linear, Scratch,
};

/// Which operand a message is about.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Operand {
    A,
    B,
    C,
}

impl Operand {
    const fn as_str(self) -> &'static str {
        match self {
            Self::A => "a",
            Self::B => "b",
            Self::C => "c",
        }
    }
}

/// The row-major strides a leading dimension names.
const fn ld(n: usize) -> Strides {
    Strides {
        rs: n as isize,
        cs: 1,
    }
}

/// A slice that does not hold the matrix its caller described.
///
/// This panics rather than returning, and the distinction is the library's error
/// doctrine rather than convenience. The reportable conditions are the ones where
/// the requested *object* does not exist, and there are two of them
/// ([`NotAProduct`], `CT-05`); a buffer that is shorter than the dimensions its
/// caller passed alongside it is neither of those --- it is the same class of
/// disagreement as handing a kernel a panel of the wrong length, which this
/// library has always treated as a programming error at the boundary and never as
/// a condition of the data (R8, C6).
#[cold]
#[inline(never)]
fn does_not_fit(operand: Operand, rows: usize, cols: usize, l: usize, len: usize) -> ! {
    panic!(
        "`{}` was declared {rows}x{cols} with a leading dimension of {l}, which needs \
         {} elements, and the slice has {len}",
        operand.as_str(),
        // R3-ok: a length in a message, on each line of the expression below
        rows.saturating_sub(1) // R3-ok: a length in a message, not an accumulation
            .saturating_mul(l) // R3-ok: a length in a message, not an accumulation
            .saturating_add(cols) // R3-ok: a length in a message, not an accumulation
    )
}

/// `C := alpha * A * B + beta * C` over integer operands, exactly.
///
/// `A` is `m x k` with leading dimension `lda`, `B` is `k x n` with `ldb`, `C` is
/// `m x n` with `ldc`, all row-major. `alpha` and `beta` are exact integer
/// scalars applied inside the accumulator, so neither can round (`CS-01`).
///
/// This is the documented default entry point: it runs
/// [`uor_matmul_gemm::gemm_auto`], which selects the kernelized factorization
/// the offer and the host's declarations admit and declines to the reference
/// traversal when they admit none --- the same bytes either way (`CD-22`).
///
/// `scratch` is the panel buffer; see the module docs. `&mut []` is valid and
/// gives the same bytes.
///
/// # Errors
///
/// [`NotAProduct`], and only that: a non-conformant shape or an output whose
/// strides map two coordinates onto one cell. Never on the values in the buffers,
/// at any magnitude or depth (R14).
///
/// # Panics
///
/// If a slice does not hold the matrix its dimensions and leading dimension
/// describe. That is a disagreement between two things the caller passed, not a
/// property of the data --- see [`does_not_fit`].
#[allow(clippy::too_many_arguments)]
pub fn gemm_ex<E, O>(
    m: usize,
    k: usize,
    n: usize,
    alpha: i64,
    a: &[E],
    lda: usize,
    b: &[E],
    ldb: usize,
    beta: i64,
    c: &mut [O],
    ldc: usize,
    scratch: &mut [E],
) -> Result<(), NotAProduct>
where
    E: Kernelized,
    O: Element + EncodeFrom<AccOf<E>> + Copy,
    AccOf<E>: ScaleExact + AbsorbPrior<O>,
{
    let (a_len, b_len, c_len) = (a.len(), b.len(), c.len());
    let av = MatView::new(as_alphabet_full(a), m, k, ld(lda))
        .unwrap_or_else(|| does_not_fit(Operand::A, m, k, lda, a_len));
    let bv = MatView::new(as_alphabet_full(b), k, n, ld(ldb))
        .unwrap_or_else(|| does_not_fit(Operand::B, k, n, ldb, b_len));
    let cv = MatViewMut::new(c, m, n, ld(ldc))
        .unwrap_or_else(|| does_not_fit(Operand::C, m, n, ldc, c_len));
    let mut triple = Triple::new(av, bv, cv)?;
    view_gemm(
        &mut triple,
        &Linear { alpha, beta },
        GemmOptions::default(),
        &mut Scratch::new(as_alphabet_full_mut(scratch)),
    );
    Ok(())
}

/// `C := A * B` over contiguous row-major integer operands.
///
/// [`gemm_ex`] at `alpha = 1`, `beta = 0`, and the leading dimensions the shape
/// implies --- which is the call almost everyone wants.
pub fn gemm<E, O>(
    m: usize,
    k: usize,
    n: usize,
    a: &[E],
    b: &[E],
    c: &mut [O],
    scratch: &mut [E],
) -> Result<(), NotAProduct>
where
    E: Kernelized,
    O: Element + EncodeFrom<AccOf<E>> + Copy,
    AccOf<E>: ScaleExact + AbsorbPrior<O>,
{
    gemm_ex(m, k, n, 1, a, k, b, n, 0, c, n, scratch)
}

/// `C := alpha * A * B + beta * C` over integer operands, exactly, with both
/// offers spelled.
///
/// The same call as [`gemm_ex`] with the working memory cut in two: `panels`
/// is the panel buffer, `accumulators` a block of exact accumulators. The two
/// are independent offers --- either may be empty --- and together they name
/// the two plans [`uor_matmul_gemm::workspace_report`] distinguishes: the
/// *suggested* plan (the full-depth panels [`suggested_scratch`] sizes, no
/// accumulators) and the *bounded* plan (one `KC` chunk of panels plus the
/// accumulator block [`suggested_accumulators`] sizes), whose cost does not
/// grow with `k`. The bytes are identical on every rung between none and both
/// (`CD-04`, `CD-10`); only the chunking changes.
///
/// [`suggested_scratch`]: uor_matmul_gemm::suggested_scratch
/// [`suggested_accumulators`]: uor_matmul_gemm::suggested_accumulators
///
/// # Errors and panics
///
/// As [`gemm_ex`].
#[allow(clippy::too_many_arguments)]
pub fn gemm_ex_full<E, O>(
    m: usize,
    k: usize,
    n: usize,
    alpha: i64,
    a: &[E],
    lda: usize,
    b: &[E],
    ldb: usize,
    beta: i64,
    c: &mut [O],
    ldc: usize,
    panels: &mut [E],
    accumulators: &mut [AccOf<E>],
) -> Result<(), NotAProduct>
where
    E: Kernelized,
    O: Element + EncodeFrom<AccOf<E>> + Copy,
    AccOf<E>: ScaleExact + AbsorbPrior<O>,
{
    let (a_len, b_len, c_len) = (a.len(), b.len(), c.len());
    let av = MatView::new(as_alphabet_full(a), m, k, ld(lda))
        .unwrap_or_else(|| does_not_fit(Operand::A, m, k, lda, a_len));
    let bv = MatView::new(as_alphabet_full(b), k, n, ld(ldb))
        .unwrap_or_else(|| does_not_fit(Operand::B, k, n, ldb, b_len));
    let cv = MatViewMut::new(c, m, n, ld(ldc))
        .unwrap_or_else(|| does_not_fit(Operand::C, m, n, ldc, c_len));
    let mut triple = Triple::new(av, bv, cv)?;
    view_gemm(
        &mut triple,
        &Linear { alpha, beta },
        GemmOptions::default(),
        &mut Scratch::with_accumulators(as_alphabet_full_mut(panels), accumulators),
    );
    Ok(())
}

/// `C := A * B` over contiguous row-major integer operands, with both offers.
///
/// [`gemm_ex_full`] at `alpha = 1`, `beta = 0`, and the implied leading
/// dimensions.
#[allow(clippy::too_many_arguments)]
pub fn gemm_full<E, O>(
    m: usize,
    k: usize,
    n: usize,
    a: &[E],
    b: &[E],
    c: &mut [O],
    panels: &mut [E],
    accumulators: &mut [AccOf<E>],
) -> Result<(), NotAProduct>
where
    E: Kernelized,
    O: Element + EncodeFrom<AccOf<E>> + Copy,
    AccOf<E>: ScaleExact + AbsorbPrior<O>,
{
    gemm_ex_full(m, k, n, 1, a, k, b, n, 0, c, n, panels, accumulators)
}

/// `C := alpha * A * B + beta * C` over float operands, exactly.
///
/// The same operation and the same shape as [`gemm_ex`]. The library never adds
/// two floats: each is a code, it decodes to an exact dyadic rational, the
/// products land in a complete accumulator, and the result is rounded once at the
/// end. That is why this is *not* bit-identical to a classical `sgemm` (N1) --- it
/// is the correctly-rounded value of the exact sum.
///
/// `pa` and `pb` are decode panels, the float analogue of `scratch`, and the
/// offer decides which factorization of the one identity runs --- never what
/// the bytes are (`CD-19`). `&mut []` runs the streaming traversal. `k` and
/// `k * n` elements let each operand be decoded once instead of once per row.
/// [`uor_matmul_gemm::suggested_float_panels`] names the offer that also
/// admits the float placement bridge, which hands the reduction to the
/// integer kernel table; it is the fast path a caller who follows the
/// suggestion gets.
///
/// # Errors and panics
///
/// As [`gemm_ex`].
#[allow(clippy::too_many_arguments)]
pub fn gemm_float_ex<E, O>(
    m: usize,
    k: usize,
    n: usize,
    alpha: i64,
    a: &[E],
    lda: usize,
    b: &[E],
    ldb: usize,
    beta: i64,
    c: &mut [O],
    ldc: usize,
    pa: &mut [PackedCode],
    pb: &mut [PackedCode],
) -> Result<(), NotAProduct>
where
    E: FloatElement,
    O: Element + EncodeFrom<AccOf<E>> + EncodeFrom<i128> + Copy,
    AccOf<E>: SignedPlace + PlaceAt + ScaleExact + AbsorbPrior<O>,
{
    let (a_len, b_len, c_len) = (a.len(), b.len(), c.len());
    let av = MatView::new(a, m, k, ld(lda))
        .unwrap_or_else(|| does_not_fit(Operand::A, m, k, lda, a_len));
    let bv = MatView::new(b, k, n, ld(ldb))
        .unwrap_or_else(|| does_not_fit(Operand::B, k, n, ldb, b_len));
    let cv = MatViewMut::new(c, m, n, ld(ldc))
        .unwrap_or_else(|| does_not_fit(Operand::C, m, n, ldc, c_len));
    let mut triple = Triple::new(av, bv, cv)?;
    gemm_float_packed(
        &mut triple,
        &Linear { alpha, beta },
        GemmOptions::default(),
        pa,
        pb,
    );
    Ok(())
}

/// `C := A * B` over contiguous row-major float operands, exactly.
///
/// [`gemm_float_ex`] at `alpha = 1`, `beta = 0`, and the implied leading
/// dimensions.
#[allow(clippy::too_many_arguments)]
pub fn gemm_float<E, O>(
    m: usize,
    k: usize,
    n: usize,
    a: &[E],
    b: &[E],
    c: &mut [O],
    pa: &mut [PackedCode],
    pb: &mut [PackedCode],
) -> Result<(), NotAProduct>
where
    E: FloatElement,
    O: Element + EncodeFrom<AccOf<E>> + EncodeFrom<i128> + Copy,
    AccOf<E>: SignedPlace + PlaceAt + ScaleExact + AbsorbPrior<O>,
{
    gemm_float_ex(m, k, n, 1, a, k, b, n, 0, c, n, pa, pb)
}
