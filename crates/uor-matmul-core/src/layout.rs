//! Shapes, strides, borrowed views, and the conformant triple (§5.5).
//!
//! Strides are arbitrary: any row stride, any column stride, negative, zero, or
//! larger than the matrix. Transposition is a stride, not a mode, so a
//! transposed operand takes the same path as any other and not a different one
//! (S7, `CS-06`).

use crate::alphabet::IntegerElement;
use crate::error::NotAProduct;

/// The three dimensions of a product.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct Shape {
    /// Rows of A and of C.
    pub m: usize,
    /// Columns of A and rows of B: the accumulation depth.
    pub k: usize,
    /// Columns of B and of C.
    pub n: usize,
}

/// A row stride and a column stride, in elements.
///
/// Both may be negative and either may be zero on an input, which is how a
/// broadcast row or a reversed traversal is expressed without a mode flag.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Strides {
    /// Elements between consecutive rows.
    pub rs: isize,
    /// Elements between consecutive columns.
    pub cs: isize,
}

impl Strides {
    /// Row-major over `cols` columns.
    pub const fn row_major(cols: usize) -> Self {
        Self {
            rs: cols as isize,
            cs: 1,
        }
    }

    /// Column-major over `rows` rows.
    pub const fn col_major(rows: usize) -> Self {
        Self {
            rs: 1,
            cs: rows as isize,
        }
    }

    /// The element offset of coordinate `(i, j)` from the view's origin.
    pub const fn offset(&self, i: usize, j: usize) -> isize {
        (i as isize)
            .wrapping_mul(self.rs)
            .wrapping_add((j as isize).wrapping_mul(self.cs))
    }

    /// The lowest and highest offsets an `rows x cols` view reaches.
    const fn extent(&self, rows: usize, cols: usize) -> (isize, isize) {
        if rows == 0 || cols == 0 {
            return (0, 0);
        }
        let last_r = (rows - 1) as isize;
        let last_c = (cols - 1) as isize;
        let r_lo = if self.rs < 0 {
            last_r.wrapping_mul(self.rs)
        } else {
            0
        };
        let r_hi = if self.rs > 0 {
            last_r.wrapping_mul(self.rs)
        } else {
            0
        };
        let c_lo = if self.cs < 0 {
            last_c.wrapping_mul(self.cs)
        } else {
            0
        };
        let c_hi = if self.cs > 0 {
            last_c.wrapping_mul(self.cs)
        } else {
            0
        };
        (r_lo.wrapping_add(c_lo), r_hi.wrapping_add(c_hi))
    }

    /// Do two distinct coordinates of an `rows x cols` view land on the same
    /// cell?
    ///
    /// Exact, not conservative. Two coordinates collide iff there are
    /// `(di, dj)` not both zero with `di * rs + dj * cs == 0`, `|di| < rows`,
    /// and `|dj| < cols`. When both strides are non-zero the smallest such
    /// solution is `(cs / g, -rs / g)` with `g = gcd(|rs|, |cs|)`, so the whole
    /// question reduces to two comparisons.
    const fn self_aliases(&self, rows: usize, cols: usize) -> bool {
        if rows == 0 || cols == 0 {
            // An empty output has no two distinct coordinates, so nothing can
            // collide. Found by the awkward-shape corpus: a 5x0 row-major view
            // has a row stride of zero, which for a non-empty output would
            // genuinely alias.
            return false;
        }
        if rows <= 1 && cols <= 1 {
            return false;
        }
        if rows > 1 && self.rs == 0 {
            return true;
        }
        if cols > 1 && self.cs == 0 {
            return true;
        }
        if rows <= 1 || cols <= 1 {
            return false;
        }
        let a = self.rs.unsigned_abs();
        let b = self.cs.unsigned_abs();
        let g = gcd(a, b);
        // g is non-zero here: both strides are non-zero.
        (b / g) < rows && (a / g) < cols
    }
}

const fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// A borrowed, arbitrarily strided input matrix.
///
/// The library never owns a matrix and never copies one to look at it. A view
/// is a slice, an origin, a shape, and two strides.
#[derive(Clone, Copy, Debug)]
pub struct MatView<'a, E> {
    data: &'a [E],
    origin: usize,
    rows: usize,
    cols: usize,
    strides: Strides,
}

/// The mutable twin of [`MatView`], for the output.
#[derive(Debug)]
pub struct MatViewMut<'a, E> {
    data: &'a mut [E],
    origin: usize,
    rows: usize,
    cols: usize,
    strides: Strides,
}

/// Compute the origin index that places an `rows x cols` strided view inside
/// `len` elements, or report that no such view exists.
const fn place(len: usize, rows: usize, cols: usize, strides: Strides) -> Option<usize> {
    if rows == 0 || cols == 0 {
        return Some(0);
    }
    let (lo, hi) = strides.extent(rows, cols);
    // The origin sits at `-lo`, so that the lowest offset reached is index 0.
    if lo > 0 {
        return None;
    }
    let origin = lo.unsigned_abs();
    let Some(top) = origin.checked_add(hi.unsigned_abs()) else {
        return None;
    };
    if top < len {
        Some(origin)
    } else {
        None
    }
}

impl<'a, E> MatView<'a, E> {
    /// Borrow `data` as an `rows x cols` matrix with the given strides.
    ///
    /// The origin is placed so that the lowest offset the view reaches is index
    /// zero, which is what makes a negative stride mean "walk backwards through
    /// this buffer" rather than "read before it".
    ///
    /// `None` means no such view exists inside a buffer of this length. That is
    /// the same kind of condition as [`NotAProduct`]: non-existence of the
    /// requested object, decided before any arithmetic is named, and impossible
    /// to provoke with the *values* in the buffer.
    pub const fn new(data: &'a [E], rows: usize, cols: usize, strides: Strides) -> Option<Self> {
        match place(data.len(), rows, cols, strides) {
            Some(origin) => Some(Self {
                data,
                origin,
                rows,
                cols,
                strides,
            }),
            None => None,
        }
    }

    /// Borrow a contiguous row-major `rows x cols` matrix. Infallible when the
    /// slice is exactly the right length.
    pub const fn row_major(data: &'a [E], rows: usize, cols: usize) -> Option<Self> {
        Self::new(data, rows, cols, Strides::row_major(cols))
    }

    /// Rows.
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Columns.
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// The strides.
    pub const fn strides(&self) -> Strides {
        self.strides
    }

    /// The element at `(i, j)`.
    ///
    /// In bounds for every `i < rows` and `j < cols` by construction, so this
    /// cannot fail and does not return an `Option`.
    pub fn at(&self, i: usize, j: usize) -> &E {
        &self.data[self.index(i, j)]
    }

    fn index(&self, i: usize, j: usize) -> usize {
        let off = self.strides.offset(i, j);
        // `origin` was chosen so that `origin + off` is a valid index for every
        // coordinate the view names; `place` established that at construction.
        self.origin.wrapping_add_signed(off)
    }

    /// A `rows`-long walk down column `j`, starting at row `i`.
    ///
    /// The packing loop's inner step. Computing `origin + i * rs + j * cs` per
    /// element costs two multiplies and an add; walking costs one add, and at
    /// a million elements per panel that is the difference between the packing
    /// being a rounding error and being a third of the work.
    pub fn column_walk(&self, i: usize, j: usize, rows: usize) -> Walk<'_, E> {
        Walk {
            data: self.data,
            at: self.index(i, j),
            step: self.strides.rs,
            left: rows,
        }
    }

    /// A `cols`-long walk along row `i`, starting at column `j`.
    pub fn row_walk(&self, i: usize, j: usize, cols: usize) -> Walk<'_, E> {
        Walk {
            data: self.data,
            at: self.index(i, j),
            step: self.strides.cs,
            left: cols,
        }
    }
}

/// A strided walk through a view, one add per element.
///
/// Yields exactly the elements [`MatView::at`] would, in the same order. It is
/// not a different way of reading the matrix --- it is the same reads with the
/// index arithmetic strength-reduced, which is why nothing downstream can tell
/// which was used.
#[derive(Clone, Debug)]
pub struct Walk<'a, E> {
    data: &'a [E],
    at: usize,
    step: isize,
    left: usize,
}

impl<'a, E> Iterator for Walk<'a, E> {
    type Item = &'a E;

    fn next(&mut self) -> Option<&'a E> {
        if self.left == 0 {
            return None;
        }
        self.left -= 1;
        let here = self.at;
        self.at = self.at.wrapping_add_signed(self.step);
        self.data.get(here)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.left, Some(self.left))
    }
}

impl<E> ExactSizeIterator for Walk<'_, E> {}

impl<'a, E> MatViewMut<'a, E> {
    /// Borrow `data` mutably as an `rows x cols` matrix. See [`MatView::new`].
    pub fn new(data: &'a mut [E], rows: usize, cols: usize, strides: Strides) -> Option<Self> {
        let origin = place(data.len(), rows, cols, strides)?;
        Some(Self {
            data,
            origin,
            rows,
            cols,
            strides,
        })
    }

    /// Borrow a contiguous row-major `rows x cols` matrix mutably.
    pub fn row_major(data: &'a mut [E], rows: usize, cols: usize) -> Option<Self> {
        Self::new(data, rows, cols, Strides::row_major(cols))
    }

    /// Rows.
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Columns.
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// The strides.
    pub const fn strides(&self) -> Strides {
        self.strides
    }

    /// The element at `(i, j)`.
    pub fn at(&self, i: usize, j: usize) -> &E {
        &self.data[self.index(i, j)]
    }

    /// The element at `(i, j)`, mutably.
    pub fn at_mut(&mut self, i: usize, j: usize) -> &mut E {
        let idx = self.index(i, j);
        &mut self.data[idx]
    }

    fn index(&self, i: usize, j: usize) -> usize {
        self.origin.wrapping_add_signed(self.strides.offset(i, j))
    }
}

/// The conformant triple.
///
/// This is where non-existence is reported, once, before any arithmetic is
/// named. `gemm` then takes a `Triple` and cannot fail: there is no second
/// place for a condition to arise, because every remaining question --- depth,
/// magnitude, alignment, host capability --- has an answer for every input
/// (R14, C6).
#[derive(Debug)]
pub struct Triple<'a, 'b, 'c, E, O> {
    a: MatView<'a, E>,
    b: MatView<'b, E>,
    c: MatViewMut<'c, O>,
}

impl<'a, 'b, 'c, E, O> Triple<'a, 'b, 'c, E, O> {
    /// One of the library's two fallible constructors. The other is
    /// `CodedTriple::new`, with the same signature and the same two failures.
    pub fn new(
        a: MatView<'a, E>,
        b: MatView<'b, E>,
        c: MatViewMut<'c, O>,
    ) -> Result<Self, NotAProduct> {
        if a.cols() != b.rows() {
            return Err(NotAProduct::Nonconformant {
                a: (a.rows(), a.cols()),
                b: (b.rows(), b.cols()),
            });
        }
        if c.rows() != a.rows() || c.cols() != b.cols() {
            return Err(NotAProduct::Nonconformant {
                a: (a.rows(), a.cols()),
                b: (b.rows(), b.cols()),
            });
        }
        let s = c.strides();
        if s.self_aliases(c.rows(), c.cols()) {
            return Err(NotAProduct::OutputAliasesItself {
                m: c.rows(),
                n: c.cols(),
                rs: s.rs,
                cs: s.cs,
            });
        }
        Ok(Self { a, b, c })
    }

    /// The shape of the product, which exists because this value does.
    pub fn shape(&self) -> Shape {
        Shape {
            m: self.a.rows(),
            k: self.a.cols(),
            n: self.b.cols(),
        }
    }

    /// The left operand.
    pub fn a(&self) -> &MatView<'a, E> {
        &self.a
    }

    /// The right operand.
    pub fn b(&self) -> &MatView<'b, E> {
        &self.b
    }

    /// The output, mutably.
    pub fn c_mut(&mut self) -> &mut MatViewMut<'c, O> {
        &mut self.c
    }

    /// The operands and the output at once, so a driver can borrow all three.
    pub fn parts(&mut self) -> (&MatView<'a, E>, &MatView<'b, E>, &mut MatViewMut<'c, O>) {
        (&self.a, &self.b, &mut self.c)
    }
}

impl<E: IntegerElement> MatView<'_, E> {
    /// The alphabet's zero, which is what a position past the end of a panel
    /// decodes to. Zero padding is exact, which is why an unaligned or prime
    /// shape takes the same path and not a different one (S8).
    pub const PAD: E = E::ZERO;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CS-02: negative and zero input strides are honoured, not rejected.
    #[test]
    fn negative_and_zero_input_strides_cs_02() {
        let data = [1i32, 2, 3, 4, 5, 6];

        // A reversed 2x3 view: row stride -3 walks backwards through the buffer.
        let v = MatView::new(&data, 2, 3, Strides { rs: -3, cs: 1 }).unwrap();
        assert_eq!(*v.at(0, 0), 4);
        assert_eq!(*v.at(1, 0), 1);
        assert_eq!(*v.at(1, 2), 3);

        // A broadcast row: zero row stride reads the same values for every row.
        let v = MatView::new(&data, 4, 3, Strides { rs: 0, cs: 1 }).unwrap();
        assert_eq!(*v.at(0, 1), *v.at(3, 1));
    }

    /// CS-06: transposition is a stride, so it is not a mode and not a branch.
    #[test]
    fn transposition_is_a_stride_cs_06() {
        let data = [1i32, 2, 3, 4, 5, 6];
        let row = MatView::new(&data, 2, 3, Strides::row_major(3)).unwrap();
        let transposed = MatView::new(&data, 3, 2, Strides { rs: 1, cs: 3 }).unwrap();
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(row.at(i, j), transposed.at(j, i));
            }
        }
    }

    /// CS-03: a self-aliasing output is reported once, before any arithmetic.
    /// The detection is exact, not conservative.
    #[test]
    fn self_aliasing_output_is_reported_cs_03() {
        // A zero row stride on an output collapses every row onto one.
        assert!(Strides { rs: 0, cs: 1 }.self_aliases(4, 4));
        // A zero column stride does the same to columns.
        assert!(Strides { rs: 4, cs: 0 }.self_aliases(4, 4));
        // (2, 3) with a 4x4 shape: (di, dj) = (3, -2) collides, and |3| < 4.
        assert!(Strides { rs: 2, cs: 3 }.self_aliases(4, 4));
        // The same strides on a 2x2 shape do not: no small solution fits.
        assert!(!Strides { rs: 2, cs: 3 }.self_aliases(2, 2));
        // Ordinary row-major never aliases.
        assert!(!Strides::row_major(4).self_aliases(4, 4));
        assert!(!Strides::col_major(4).self_aliases(4, 4));
        // A single row or column cannot alias, whatever the strides.
        assert!(!Strides { rs: 0, cs: 0 }.self_aliases(1, 1));
        // Neither can an empty output, which has no two coordinates at all ---
        // even though a 5x0 row-major view has a row stride of zero.
        assert!(!Strides::row_major(0).self_aliases(5, 0));
        assert!(!Strides::row_major(7).self_aliases(0, 7));
        // But a genuine collapse of five rows onto one cell does alias.
        assert!(Strides { rs: 0, cs: 1 }.self_aliases(5, 3));
    }

    /// CS-03, CT-05: the two reportable conditions are reported here, and
    /// nowhere else.
    #[test]
    fn nonconformant_shapes_are_reported_at_construction_cs_03() {
        let a = [0i32; 6];
        let b = [0i32; 6];
        let mut c = [0i32; 4];
        let av = MatView::row_major(&a, 2, 3).unwrap();
        let bv = MatView::row_major(&b, 2, 3).unwrap();
        let cv = MatViewMut::row_major(&mut c, 2, 2).unwrap();
        assert_eq!(
            Triple::new(av, bv, cv).unwrap_err(),
            NotAProduct::Nonconformant {
                a: (2, 3),
                b: (2, 3)
            }
        );
    }

    /// CT-04: a zero dimension is not a special case and constructs fine.
    #[test]
    fn zero_dimensions_construct_ct_04() {
        let a: [i32; 0] = [];
        let b: [i32; 0] = [];
        let mut c: [i32; 0] = [];
        let av = MatView::row_major(&a, 0, 0).unwrap();
        let bv = MatView::row_major(&b, 0, 0).unwrap();
        let cv = MatViewMut::row_major(&mut c, 0, 0).unwrap();
        let t = Triple::new(av, bv, cv).unwrap();
        assert_eq!(t.shape(), Shape { m: 0, k: 0, n: 0 });
    }
}
