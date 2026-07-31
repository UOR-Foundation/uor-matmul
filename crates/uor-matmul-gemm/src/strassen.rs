//! The sub-cubic factorization: Winograd's form of Strassen's recursion.
//!
//! Over the integers the recursion uses only add, subtract, and multiply, so
//! the regrouped sum is the same integer the naive loop returns, bit for bit.
//! A float library declines Strassen because intermediate cancellation
//! degrades its norm bounds with depth; there is no norm here and nothing
//! degrades, so the exposure that keeps it out of every classical `sgemm`
//! does not exist. The `CD-*` byte-equality discipline covers the recursion
//! with no new argument --- exactly as it covers tabulation --- and `CD-21`
//! asserts it: at every shape, depth, requested level count, and offer
//! including none, the recursion's bytes are the cubic walk's.
//!
//! # The variant
//!
//! One level halves all three extents and forms eight block sums and seven
//! products, in Winograd's arrangement (fifteen block additions against
//! Strassen's eighteen --- the additions are the recursion's `O(n^2)`
//! overhead, so the cheaper arrangement is the one implemented):
//!
//! ```text
//! S1 = A21 + A22    T1 = B12 - B11
//! S2 = S1 - A11     T2 = B22 - T1
//! S3 = A11 - A21    T3 = B22 - B12
//! S4 = A12 - S2     T4 = B21 - T2
//!
//! M1 = A11 * B11   M2 = A12 * B21   M3 = S4 * B22   M4 = A22 * T4
//! M5 = S1 * T1     M6 = S2 * T2     M7 = S3 * T3
//!
//! C11 = M1 + M2
//! C12 = M1 + M6 + M5 + M3
//! C21 = M1 + M6 + M7 + M4
//! C22 = M1 + M6 + M7 + M5
//! ```
//!
//! The textbook form writes `T4 = T2 - B21` and `C21 = U3 - M4`. Here the one
//! negation in the combination is folded into the sum temporary, so the
//! accumulator only ever adds: there is no subtraction in the accumulator
//! width to define, and the byte-equality argument stays one sentence.
//!
//! # The bound bookkeeping
//!
//! The cross-term sums are four block terms at worst (`S4 = A12 - A21 - A22 +
//! A11`), so a level taken at operand bound `b` forms sums of magnitude at
//! most `4b`, and those sums are the next level's operands. The plan admits a
//! level only while `4^L * B` stays inside the element type --- `E::FULL - 1`,
//! not `E::FULL`, because the type's largest magnitude is reachable with one
//! sign only and a sum's sign is the data's --- and each product carries its
//! operands' exact grown bounds down to the lane-depth declaration at the
//! base case. This is why the recursion runs on the `i32` lane and not on
//! `i8`: at full `i8` range one level's sums already leave the alphabet
//! (`254 > 127`), while on `i32` each level costs two bits of a thirty-one
//! that full-range data is not using. At the full `i32` alphabet there are
//! zero free levels, the same zero `i8` has --- the plan says so and declines.
//!
//! The accumulator is not a constraint at any admitted depth: a product
//! temporary at level `l` is bounded by `9 * 4^2l * B^2 * k / 2^l`, which the
//! headroom rule `4^L * B <= FULL` keeps under `(9/16)` of the worst case the
//! accumulator's width was derived against (§3.2). `CT-02` witnesses it: the
//! whole corpus runs under checked arithmetic, and the recursion's sums,
//! lanes, and combinations are the interesting overflow witness it was built
//! for.
//!
//! # The offer decides
//!
//! The sums and the seven product temporaries are scratch, offered by the
//! caller like every other buffer in the library: the sums from the panel
//! offer (bare elements --- they outgrow the declared bound by construction,
//! and the grown bound is re-declared at the kernel boundary as a value), the
//! products from the accumulator offer, which is exactly the width an exact
//! partial sum needs. A level the offer cannot hold is declined, and a
//! decline is the cubic walk: same bytes, more products (`CD-10`'s rule, one
//! traversal up).
//!
//! Odd extents decline rather than pad. Zero padding would be exact ---
//! `CK-03`'s precedent --- but it would materialize a padded copy of both
//! operands, `O(m*k + k*n)` of traffic and a second buffer discipline, to buy
//! a shape one halving; declining keeps every shape exact by construction and
//! the mechanism one code path, and the shapes the recursion is measured to
//! pay at are even either way.

use uor_matmul_core::{
    as_alphabet_full, as_alphabet_full_mut, AccOf, Accumulator, Alphabet, Backend, Bound, Element,
    EncodeFrom, Full, IntegerElement, MatView, Shape, Triple,
};

use crate::driver::GemmOptions;
use crate::epilogue::Epilogue;
use crate::kernel::Kernelized;
use crate::scratch::Scratch;

/// Panels and accumulators one level at `(m, k, n)` carves: four A-side sums
/// of `m/2 x k/2`, four B-side sums of `k/2 x n/2`, and seven product
/// temporaries of `m/2 x n/2` in the accumulator's width.
///
/// Shared by the plan and the carve, so the two cannot disagree.
fn level_cost(m: usize, k: usize, n: usize) -> (u128, u128) {
    let (mh, kh, nh) = (m as u128 / 2, k as u128 / 2, n as u128 / 2);
    let sums = mh.saturating_mul(kh).saturating_add(kh.saturating_mul(nh)); // R3-ok: a scratch size question
    let products = mh.saturating_mul(nh); // R3-ok: a scratch size question
    (
        sums.saturating_mul(4),     // R3-ok: a scratch size question
        products.saturating_mul(7), // R3-ok: a scratch size question
    )
}

/// What the recursion wants for `l` levels at `shape`: every level's sums and
/// products, held while the level below runs, plus the base case's own
/// packed-traversal offer at the bottom.
pub(crate) fn needs(shape: Shape, l: usize) -> (u128, u128) {
    let (mut panels, mut accs) = (0u128, 0u128);
    let (mut m, mut k, mut n) = (shape.m, shape.k, shape.n);
    for _ in 0..l {
        let (p, q) = level_cost(m, k, n);
        panels = panels.saturating_add(p); // R3-ok: a scratch size question
        accs = accs.saturating_add(q); // R3-ok: a scratch size question
        m /= 2;
        k /= 2;
        n /= 2;
    }
    let base = Shape { m, k, n };
    // The base case's own traversal: a full-depth panel, and one output block
    // of accumulators. The block is offered unconditionally rather than
    // through `suggested_accumulators`, which answers `0` whenever `k <= KC`
    // --- the base case chunks whenever its *lane* is shallower than `k`, and
    // the lane's depth is a fact of the grown bound, which a shape-only query
    // cannot see. Without the block the base case streams per tile, which is
    // the one traversal measurably slower than the products it replaces.
    use uor_matmul_core::generated::blocking;
    let base_accs =
        (base.m.min(blocking::MC) as u128).saturating_mul(base.n.min(blocking::NC) as u128); // R3-ok: a scratch size question
    (
        panels.saturating_add(crate::suggested_scratch(base) as u128), // R3-ok: a scratch size question
        accs.saturating_add(base_accs), // R3-ok: a scratch size question
    )
}

/// What the recursion wants for `levels` levels at `shape`: panel elements
/// and accumulators.
///
/// A *query*, like [`crate::suggested_scratch`]: offering less declines
/// levels, it never fails and never changes a byte (`CD-21`).
pub fn strassen_scratch(shape: Shape, levels: usize) -> (usize, usize) {
    let (p, q) = needs(shape, levels);
    (
        p.min(usize::MAX as u128) as usize,
        q.min(usize::MAX as u128) as usize,
    )
}

/// What the recursion wants at this shape when its size and evenness admit
/// levels, `(0, 0)` where they admit none.
///
/// Blind to the declared bound, which a shape-only query cannot see: a caller
/// whose alphabet leaves the sums no headroom declines the recursion, and
/// what is offered buys the same bytes from the cubic walk (`CD-10`). The
/// threshold is the `i32` lane's, the only lane with a measurement.
pub(crate) fn wants(shape: Shape) -> (u128, u128) {
    use uor_matmul_core::generated::blocking;
    let (mut m, mut k, mut n) = (shape.m, shape.k, shape.n);
    let mut l = 0;
    while m % 2 == 0
        && k % 2 == 0
        && n % 2 == 0
        && m / 2 >= blocking::STRASSEN_MIN_EXTENT
        && k / 2 >= blocking::STRASSEN_MIN_EXTENT
        && n / 2 >= blocking::STRASSEN_MIN_EXTENT
    {
        l += 1;
        m /= 2;
        k /= 2;
        n /= 2;
    }
    if l == 0 {
        return (0, 0);
    }
    needs(shape, l)
}

/// How many of the `requested` levels the declarations admit at this shape,
/// bound, and offer.
///
/// A level is admitted exactly while all of these hold, each a declaration
/// and none of them a look at the data:
///
/// - **The shape halves.** All three extents stay even through the level.
///   Odd extents decline; see the module documentation for why there is no
///   padding path.
/// - **The sums stay in the element.** The level's block sums reach `4^L *
///   bound`, which must not exceed `E::FULL - 1`.
/// - **The offer holds the temporaries.** [`needs`] for the level count must
///   fit the offered panels and accumulators.
///
/// `requested == usize::MAX` is the auto-selection the packed driver makes,
/// and it is additionally capped by the measured crossover
/// ([`Kernelized::strassen_min_extent`]): the base case's smallest extent must
/// stay at or above the extent a level is measured to pay at. A finite
/// request is a caller declaring its levels, the same position
/// `Traversal::Tabulated` takes --- the caller knows its shape --- and it is
/// capped only by the three exactness rules above.
pub fn levels<E: Kernelized>(
    shape: Shape,
    bound: u128,
    panels: usize,
    accs: usize,
    requested: usize,
) -> usize {
    let mut cap = requested;
    if requested == usize::MAX {
        let Some(min_extent) = E::strassen_min_extent() else {
            return 0;
        };
        let (mut m, mut k, mut n) = (shape.m, shape.k, shape.n);
        cap = 0;
        while m / 2 >= min_extent && k / 2 >= min_extent && n / 2 >= min_extent {
            cap += 1;
            m /= 2;
            k /= 2;
            n /= 2;
        }
    }
    let mut grown = bound;
    let mut taken = 0;
    let (mut m, mut k, mut n) = (shape.m, shape.k, shape.n);
    while taken < cap {
        if m % 2 != 0 || k % 2 != 0 || n % 2 != 0 {
            break;
        }
        // The next level's sums reach four times the bound this level's
        // operands carry. Saturating rather than checked: past `2^126` the
        // bound is past every element type's `FULL`, and the decline below is
        // the right answer there too.
        grown = grown.saturating_mul(4); // R3-ok: a bound declaration, saturating towards decline
        if grown > E::FULL - 1 {
            break;
        }
        let (p, q) = needs(shape, taken + 1);
        if p > panels as u128 || q > accs as u128 {
            break;
        }
        taken += 1;
        m /= 2;
        k /= 2;
        n /= 2;
    }
    taken
}

/// `C := epilogue(A * B, C)` through the sub-cubic recursion.
///
/// `levels` is a *request*: the declarations admit some count `L <= levels`
/// (see [`levels`]), the recursion runs `L` levels deep, and the products the
/// regrouping does not take run as the ordinary packed traversal. At `L = 0`
/// this is the packed walk itself. Every one of those is the same integer, so
/// the request decides which instructions run and nothing else (`CD-21`).
///
/// Returns `()`, for the same reason [`crate::gemm`] does.
pub fn gemm_strassen<E, Bd, O, Ep>(
    triple: &mut Triple<'_, '_, '_, Alphabet<E, Bd>, O>,
    epilogue: &Ep,
    options: GemmOptions,
    scratch: &mut Scratch<'_, E, Bd>,
    levels_requested: usize,
) where
    E: Kernelized,
    Bd: Bound,
    O: Element + EncodeFrom<AccOf<E>>,
    Ep: Epilogue<E, O>,
{
    let shape = triple.shape();
    if shape.m == 0 || shape.n == 0 {
        return;
    }
    let take = levels::<E>(
        shape,
        Bd::VALUE,
        scratch.len(),
        scratch.accumulators(),
        levels_requested,
    );
    if take == 0 {
        // No level the declarations admit: the packed walk, with the recursion
        // switched off. A caller who asked for zero levels gets zero levels ---
        // the packed walk's own auto-selection would otherwise take them, which
        // is the same bytes (`CD-21`) and not what was asked.
        crate::kernel::gemm_packed_impl(triple, epilogue, options, scratch, false);
        return;
    }
    run(triple, epilogue, options, scratch, Bd::VALUE, take);
}

/// The recursion, `depth` levels, over the exact lanes. The caller --- the
/// packed driver's exact arm, or [`gemm_strassen`] --- has already counted
/// the levels the declarations admit.
pub(crate) fn run<E, Bd, O, Ep>(
    triple: &mut Triple<'_, '_, '_, Alphabet<E, Bd>, O>,
    epilogue: &Ep,
    options: GemmOptions,
    scratch: &mut Scratch<'_, E, Bd>,
    bound: u128,
    depth: usize,
) where
    E: Kernelized,
    Bd: Bound,
    O: Element + EncodeFrom<AccOf<E>>,
    Ep: Epilogue<E, O>,
{
    let shape = triple.shape();
    // The plan admitted this depth against this offer, so a short pool here is
    // a plan/carve disagreement, not an input condition. Written as a decline
    // rather than an assertion because the cubic walk is right there: same
    // bytes, more products (`CD-21`).
    let (need_panels, need_accs) = level_cost(shape.m, shape.k, shape.n);
    if (scratch.len() as u128) < need_panels || (scratch.accumulators() as u128) < need_accs {
        crate::kernel::gemm_packed_cubic_at(triple, epilogue, options, scratch, bound);
        return;
    }
    // The operands, as bare elements and back as the full alphabet: the sums
    // the recursion forms outgrow `Bd` by construction, and the grown bound
    // travels as a value (`bound`, and its multiples below) to the kernel
    // boundary, which is where the alphabet hypothesis is discharged.
    let av = triple.a().peeled().full_alphabet();
    let bv = triple.b().peeled().full_alphabet();

    let (panel_count, acc_count) = (scratch.len(), scratch.accumulators());
    let (panel_offer, acc_offer) = scratch.split(panel_count, acc_count);
    let panels: &mut [E] = bytemuck::TransparentWrapper::peel_slice_mut(panel_offer);
    let accs: &mut [AccOf<E>] = acc_offer;

    let reads_c = epilogue.reads_c();
    let encode = options.encode;
    let (m2, n2) = (shape.m / 2, shape.n / 2);
    // The top level's combination is output: the epilogue runs on it, exactly
    // once per element, as on every other traversal. Below the top the
    // combination lands in a product temporary and no epilogue runs at all.
    let mut sink = |i: usize, j: usize, q: [AccOf<E>; 4]| {
        let c = triple.c_mut();
        for (ci, cj, acc) in [
            (i, j, q[0]),
            (i, j + n2, q[1]),
            (i + m2, j, q[2]),
            (i + m2, j + n2, q[3]),
        ] {
            let prior = if reads_c { Some(*c.at(ci, cj)) } else { None };
            *c.at_mut(ci, cj) = epilogue.finish(acc, prior, encode);
        }
    };
    winograd(
        &av,
        bound,
        &bv,
        bound,
        depth,
        &mut sink,
        panels,
        accs,
        options.backend,
    );
}

/// One Winograd level over `a * b`, with `level - 1` levels below it.
///
/// Carves the level's sums and product temporaries from the pools, forms the
/// sums, computes the seven products, and hands each output element's four
/// quadrant sums to `sink`. The pools were admitted by the plan; the carve
/// amounts are [`level_cost`]'s, which the plan read through [`needs`].
#[allow(clippy::too_many_arguments)]
fn winograd<E: Kernelized>(
    a: &MatView<'_, Alphabet<E, Full<E>>>,
    ab: u128,
    b: &MatView<'_, Alphabet<E, Full<E>>>,
    bb: u128,
    level: usize,
    sink: &mut impl FnMut(usize, usize, [AccOf<E>; 4]),
    mut panels: &mut [E],
    mut accs: &mut [AccOf<E>],
    backend: Backend,
) {
    let (m, k, n) = (a.rows(), a.cols(), b.cols());
    let (m2, k2, n2) = (m / 2, k / 2, n / 2);

    let s1 = carve(&mut panels, m2 * k2);
    let s2 = carve(&mut panels, m2 * k2);
    let s3 = carve(&mut panels, m2 * k2);
    let s4 = carve(&mut panels, m2 * k2);
    let t1 = carve(&mut panels, k2 * n2);
    let t2 = carve(&mut panels, k2 * n2);
    let t3 = carve(&mut panels, k2 * n2);
    let t4 = carve(&mut panels, k2 * n2);
    let m1 = carve(&mut accs, m2 * n2);
    let m2b = carve(&mut accs, m2 * n2);
    let m3 = carve(&mut accs, m2 * n2);
    let m4 = carve(&mut accs, m2 * n2);
    let m5 = carve(&mut accs, m2 * n2);
    let m6 = carve(&mut accs, m2 * n2);
    let m7 = carve(&mut accs, m2 * n2);

    let (a11, a12) = (quadrant(a, 0, 0, m2, k2), quadrant(a, 0, k2, m2, k2));
    let (a21, a22) = (quadrant(a, m2, 0, m2, k2), quadrant(a, m2, k2, m2, k2));
    let (b11, b12) = (quadrant(b, 0, 0, k2, n2), quadrant(b, 0, n2, k2, n2));
    let (b21, b22) = (quadrant(b, k2, 0, k2, n2), quadrant(b, k2, n2, k2, n2));

    // The eight sums. `E::add`/`E::sub` are the spelled-wrapping spellings of
    // arithmetic that cannot wrap: the plan admitted this level only while
    // `4^level * bound` stays inside the element type. The reads are the
    // views' own walks --- one add per element rather than an index
    // recomputed per quadrant --- because at the sizes the recursion pays,
    // this loop is the whole of the level's `O(n^2)` overhead.
    for i in 0..m2 {
        let rows = (
            &mut s1[i * k2..][..k2],
            &mut s2[i * k2..][..k2],
            &mut s3[i * k2..][..k2],
            &mut s4[i * k2..][..k2],
        );
        let walks = (
            a11.row_walk(i, 0, k2),
            a12.row_walk(i, 0, k2),
            a21.row_walk(i, 0, k2),
            a22.row_walk(i, 0, k2),
        );
        for (j, (((x11, x12), x21), x22)) in
            walks.0.zip(walks.1).zip(walks.2).zip(walks.3).enumerate()
        {
            let (x11, x12, x21, x22) = (x11.get(), x12.get(), x21.get(), x22.get());
            let s1v = E::add(x21, x22);
            rows.0[j] = s1v;
            rows.1[j] = E::sub(s1v, x11);
            rows.2[j] = E::sub(x11, x21);
            rows.3[j] = E::sub(x12, E::sub(s1v, x11));
        }
    }
    for i in 0..k2 {
        let rows = (
            &mut t1[i * n2..][..n2],
            &mut t2[i * n2..][..n2],
            &mut t3[i * n2..][..n2],
            &mut t4[i * n2..][..n2],
        );
        let walks = (
            b11.row_walk(i, 0, n2),
            b12.row_walk(i, 0, n2),
            b21.row_walk(i, 0, n2),
            b22.row_walk(i, 0, n2),
        );
        for (j, (((y11, y12), y21), y22)) in
            walks.0.zip(walks.1).zip(walks.2).zip(walks.3).enumerate()
        {
            let (y11, y12, y21, y22) = (y11.get(), y12.get(), y21.get(), y22.get());
            let t1v = E::sub(y12, y11);
            let t2v = E::sub(y22, t1v);
            rows.0[j] = t1v;
            rows.1[j] = t2v;
            rows.2[j] = E::sub(y22, y12);
            // `B21 - T2`, negated against the textbook's `T2 - B21`: the one
            // subtraction in the combination is folded into this temporary.
            rows.3[j] = E::sub(y21, t2v);
        }
    }

    // The seven products. The sums a product reads are formed already; the
    // pools' remainder is each child's own scratch, and the products are
    // sequential, so the children share it one at a time.
    recurse(&a11, ab, &b11, bb, level - 1, m1, n2, panels, accs, backend);
    recurse(
        &a12,
        ab,
        &b21,
        bb,
        level - 1,
        m2b,
        n2,
        panels,
        accs,
        backend,
    );
    let (s4v, t4v) = (sums_view(s4, m2, k2), sums_view(t4, k2, n2));
    recurse(
        &s4v,
        4 * ab,
        &b22,
        bb,
        level - 1,
        m3,
        n2,
        panels,
        accs,
        backend,
    );
    recurse(
        &a22,
        ab,
        &t4v,
        4 * bb,
        level - 1,
        m4,
        n2,
        panels,
        accs,
        backend,
    );
    let (s1v, t1v) = (sums_view(s1, m2, k2), sums_view(t1, k2, n2));
    recurse(
        &s1v,
        2 * ab,
        &t1v,
        2 * bb,
        level - 1,
        m5,
        n2,
        panels,
        accs,
        backend,
    );
    let (s2v, t2v) = (sums_view(s2, m2, k2), sums_view(t2, k2, n2));
    recurse(
        &s2v,
        3 * ab,
        &t2v,
        3 * bb,
        level - 1,
        m6,
        n2,
        panels,
        accs,
        backend,
    );
    let (s3v, t3v) = (sums_view(s3, m2, k2), sums_view(t3, k2, n2));
    recurse(
        &s3v,
        2 * ab,
        &t3v,
        2 * bb,
        level - 1,
        m7,
        n2,
        panels,
        accs,
        backend,
    );

    // The combination. Every operand is a complete exact sum and `combine` is
    // the accumulator's own addition, so the regrouping is invisible in the
    // result --- which is the whole of `CD-21`'s claim.
    for i in 0..m2 {
        for j in 0..n2 {
            let at = i * n2 + j;
            let u2 = m1[at].combine(m6[at]);
            let u3 = u2.combine(m7[at]);
            sink(
                i,
                j,
                [
                    m1[at].combine(m2b[at]),
                    u2.combine(m5[at]).combine(m3[at]),
                    u3.combine(m4[at]),
                    u3.combine(m5[at]),
                ],
            );
        }
    }
}

/// One product of the recursion: `levels` more levels of regrouping, or the
/// packed traversal when they run out --- or when the pools cannot hold this
/// level's temporaries, which the plan's admission makes unreachable and the
/// decline makes harmless.
#[allow(clippy::too_many_arguments)]
fn recurse<E: Kernelized>(
    a: &MatView<'_, Alphabet<E, Full<E>>>,
    ab: u128,
    b: &MatView<'_, Alphabet<E, Full<E>>>,
    bb: u128,
    level: usize,
    out: &mut [AccOf<E>],
    ldc: usize,
    panels: &mut [E],
    accs: &mut [AccOf<E>],
    backend: Backend,
) {
    let (need_panels, need_accs) = level_cost(a.rows(), a.cols(), b.cols());
    if level == 0 || (panels.len() as u128) < need_panels || (accs.len() as u128) < need_accs {
        let mut sub = Scratch::with_accumulators(as_alphabet_full_mut(panels), accs);
        crate::kernel::gemm_packed_exact_raw(a, b, backend, &mut sub, ab.max(bb), out, ldc);
        return;
    }
    let (m2, n2) = (a.rows() / 2, b.cols() / 2);
    let mut sink = |i: usize, j: usize, q: [AccOf<E>; 4]| {
        out[i * ldc + j] = q[0];
        out[i * ldc + n2 + j] = q[1];
        out[(i + m2) * ldc + j] = q[2];
        out[(i + m2) * ldc + n2 + j] = q[3];
    };
    winograd(a, ab, b, bb, level, &mut sink, panels, accs, backend);
}

/// Bump-allocate `n` elements from the front of `pool`.
///
/// The plan admitted exactly these amounts, so the carve never wants more
/// than the pool holds; `recurse` checks before descending, which is what
/// keeps this a `split_at_mut` rather than a fallback.
fn carve<'p, T>(pool: &mut &'p mut [T], n: usize) -> &'p mut [T] {
    let (head, tail) = core::mem::take(pool).split_at_mut(n);
    *pool = tail;
    head
}

/// A quadrant of an operand, as a view in its own right.
///
/// `i + r <= rows` and `j + c <= cols` by the halving, so the block exists;
/// [`MatView::subview`]'s `None` is for a caller who names a block outside
/// the view, which the plan's evenness rule makes unreachable here.
fn quadrant<'v, E: IntegerElement>(
    v: &MatView<'v, Alphabet<E, Full<E>>>,
    i: usize,
    j: usize,
    r: usize,
    c: usize,
) -> MatView<'v, Alphabet<E, Full<E>>> {
    v.subview(i, j, r, c).expect("the plan halves even extents")
}

/// A sum temporary as a row-major operand view. The carve is exact, so the
/// view exists.
fn sums_view<E: IntegerElement>(s: &[E], r: usize, c: usize) -> MatView<'_, Alphabet<E, Full<E>>> {
    MatView::row_major(as_alphabet_full(s), r, c).expect("the carve is the view's extent")
}

#[cfg(test)]
// R7 governs the library, not its tests: these build operands on the heap so
// that awkward shapes can be generated. `CA-01` witnesses the library's own
// zero-allocation claim with a counting allocator instead.
#[allow(clippy::disallowed_types)]
mod tests {
    use super::*;
    use crate::epilogue::Linear;
    use std::vec;
    use std::vec::Vec;
    use uor_matmul_core::{as_alphabet, Bnd, EncodeMode, Full, MatViewMut, Strides};

    /// A deterministic fill with entries in `[-bound, bound]`.
    fn fill(len: usize, seed: u64, bound: i32) -> Vec<i32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 33) as i32).wrapping_abs() % (bound + 1) * if s & 1 == 0 { -1 } else { 1 }
            })
            .collect()
    }

    /// The reference answer: the streaming driver, no scratch, no kernels.
    fn reference(
        m: usize,
        k: usize,
        n: usize,
        a: &[i32],
        b: &[i32],
        encode: EncodeMode,
    ) -> Vec<i32> {
        let mut c = vec![0i32; m * n];
        let av = MatView::row_major(as_alphabet::<i32, Bnd<128>>(a).unwrap(), m, k).unwrap();
        let bv = MatView::row_major(as_alphabet::<i32, Bnd<128>>(b).unwrap(), k, n).unwrap();
        let cv = MatViewMut::row_major(&mut c, m, n).unwrap();
        let mut t = Triple::new(av, bv, cv).unwrap();
        crate::gemm(
            &mut t,
            &Linear::OVERWRITE,
            GemmOptions {
                encode,
                ..Default::default()
            },
            &mut Scratch::none(),
        );
        c
    }

    /// The recursion at `levels` requested levels over `panels`/`accs` of offer.
    #[allow(clippy::too_many_arguments)]
    fn recursive(
        m: usize,
        k: usize,
        n: usize,
        a: &[i32],
        b: &[i32],
        levels: usize,
        panels: usize,
        accs: usize,
        encode: EncodeMode,
    ) -> Vec<i32> {
        let mut c = vec![0i32; m * n];
        let mut panel = vec![Alphabet::<i32, Bnd<128>>::ZERO; panels];
        let mut acc_buf = vec![0i128; accs];
        let av = MatView::row_major(as_alphabet::<i32, Bnd<128>>(a).unwrap(), m, k).unwrap();
        let bv = MatView::row_major(as_alphabet::<i32, Bnd<128>>(b).unwrap(), k, n).unwrap();
        let cv = MatViewMut::row_major(&mut c, m, n).unwrap();
        let mut t = Triple::new(av, bv, cv).unwrap();
        gemm_strassen(
            &mut t,
            &Linear::OVERWRITE,
            GemmOptions {
                encode,
                ..Default::default()
            },
            &mut Scratch::with_accumulators(&mut panel, &mut acc_buf),
            levels,
        );
        c
    }

    /// CD-21: at every shape --- even, odd, prime, degenerate --- at every
    /// requested level count, and at every offer including none, the
    /// recursion's bytes are the reference driver's.
    #[test]
    fn recursion_is_byte_identical_to_the_cubic_walk_cd_21() {
        let shapes = [
            (0, 0, 0),
            (1, 1, 1),
            (2, 2, 2),
            (4, 6, 8),
            (5, 7, 9),
            (12, 18, 20),
            (13, 17, 19),
            (96, 64, 80),
            (2, 4096, 2),
        ];
        for &(m, k, n) in &shapes {
            let a = fill(m * k, m as u64 + 1, 100);
            let b = fill(k * n, n as u64 + 2, 100);
            for encode in [EncodeMode::Wrapping, EncodeMode::Saturating] {
                let want = reference(m, k, n, &a, &b, encode);
                for levels in [0, 1, 2, 3, 5] {
                    let (panels, accs) = strassen_scratch(Shape { m, k, n }, levels);
                    for offer in [0, panels / 3, panels] {
                        let got = recursive(m, k, n, &a, &b, levels, offer, accs, encode);
                        assert_eq!(
                            got, want,
                            "{m}x{k}x{n} levels {levels} offer {offer} encode {encode:?}"
                        );
                    }
                }
            }
        }
    }

    /// CD-21: the recursion reaches every product the cubic walk does. The
    /// three single-product quadrants of a 2x2 blocking are the smallest
    /// shapes that distinguish a wrong assembly from a right one, and the
    /// worst-magnitude fills are the ones whose block sums reach the grown
    /// bound.
    #[test]
    fn the_seven_products_assemble_the_identity_cd_21() {
        // Hand-computed 2x2: C = A * B with A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]].
        let a = [1i32, 2, 3, 4];
        let b = [5i32, 6, 7, 8];
        let want = [19i32, 22, 43, 50];
        let (panels, accs) = strassen_scratch(Shape { m: 2, k: 2, n: 2 }, 1);
        let got = recursive(2, 2, 2, &a, &b, 1, panels, accs, EncodeMode::Wrapping);
        assert_eq!(
            got, want,
            "one level over 2x2 must be the schoolbook answer"
        );

        // Full-magnitude entries, so the block sums reach 4 * bound exactly.
        for &(m, k, n) in &[(16, 16, 16), (32, 16, 64)] {
            let a = vec![127i32; m * k];
            let b = vec![-127i32; k * n];
            let want = reference(m, k, n, &a, &b, EncodeMode::Wrapping);
            for levels in [1, 2, 3] {
                let (panels, accs) = strassen_scratch(Shape { m, k, n }, levels);
                let got = recursive(m, k, n, &a, &b, levels, panels, accs, EncodeMode::Wrapping);
                assert_eq!(got, want, "{m}x{k}x{n} levels {levels} at the bound's edge");
            }
        }
    }

    /// CD-21: a level that reads `C` is exact too --- the epilogue runs once
    /// per element on the combination, with the prior value, exactly as on
    /// the cubic walk.
    #[test]
    fn an_epilogue_that_reads_c_runs_once_cd_21() {
        let (m, k, n) = (16, 32, 24);
        let a = fill(m * k, 7, 100);
        let b = fill(k * n, 8, 100);
        let prior = fill(m * n, 9, 50);
        let run = |levels: usize| {
            let mut c = prior.clone();
            let (panels, accs) = strassen_scratch(Shape { m, k, n }, levels);
            let mut panel = vec![Alphabet::<i32, Bnd<128>>::ZERO; panels];
            let mut acc_buf = vec![0i128; accs];
            let av = MatView::row_major(as_alphabet::<i32, Bnd<128>>(&a).unwrap(), m, k).unwrap();
            let bv = MatView::row_major(as_alphabet::<i32, Bnd<128>>(&b).unwrap(), k, n).unwrap();
            let cv = MatViewMut::row_major(&mut c, m, n).unwrap();
            let mut t = Triple::new(av, bv, cv).unwrap();
            gemm_strassen(
                &mut t,
                &Linear { alpha: 3, beta: -2 },
                GemmOptions {
                    encode: EncodeMode::Wrapping,
                    ..Default::default()
                },
                &mut Scratch::with_accumulators(&mut panel, &mut acc_buf),
                levels,
            );
            c
        };
        assert_eq!(run(0), run(2));
        assert_eq!(run(1), run(3));
    }

    /// CD-21: strides are invisible. A transposed operand and a transposed
    /// output are the same product under the recursion as under the cubic
    /// walk, because a quadrant of a strided view is a strided view.
    #[test]
    fn strided_operands_and_output_are_invisible_cd_21() {
        let (m, k, n) = (16, 32, 24);
        let a = fill(m * k, 11, 100);
        let b = fill(k * n, 12, 100);
        let want = reference(m, k, n, &a, &b, EncodeMode::Wrapping);

        // A read through its transpose's strides; C written through its own.
        // `at` is A stored as its transpose, so the strided view reads A.
        let mut at = vec![0i32; m * k];
        for i in 0..m {
            for j in 0..k {
                at[j * m + i] = a[i * k + j];
            }
        }
        let (panels, accs) = strassen_scratch(Shape { m, k, n }, 2);
        let mut panel = vec![Alphabet::<i32, Bnd<128>>::ZERO; panels];
        let mut acc_buf = vec![0i128; accs];
        let mut ct = vec![0i32; m * n];
        let av = MatView::new(
            as_alphabet::<i32, Bnd<128>>(&at).unwrap(),
            m,
            k,
            Strides {
                rs: 1,
                cs: m as isize,
            },
        )
        .unwrap();
        let bv = MatView::row_major(as_alphabet::<i32, Bnd<128>>(&b).unwrap(), k, n).unwrap();
        let cv = MatViewMut::new(
            &mut ct,
            m,
            n,
            Strides {
                rs: 1,
                cs: m as isize,
            },
        )
        .unwrap();
        let mut t = Triple::new(av, bv, cv).unwrap();
        gemm_strassen(
            &mut t,
            &Linear::OVERWRITE,
            GemmOptions {
                encode: EncodeMode::Wrapping,
                ..Default::default()
            },
            &mut Scratch::with_accumulators(&mut panel, &mut acc_buf),
            2,
        );
        // `ct` holds C^T row-major: `ct[j * m + i] == C[i][j]`.
        for i in 0..m {
            for j in 0..n {
                assert_eq!(
                    ct[j * m + i],
                    want[i * n + j],
                    "({i}, {j}) through transposed strides"
                );
            }
        }
    }

    /// CD-21: the same identity on the `i8` lane. At full `i8` range the
    /// headroom admits no level; at a declared bound of 7 it admits one ---
    /// and the byte-equality discipline covers both, because it does not care
    /// which lane the products ran on.
    #[test]
    fn the_i8_lane_is_covered_by_the_same_argument_cd_21() {
        let (m, k, n) = (16, 32, 24);
        let a: Vec<i8> = fill(m * k, 13, 6).into_iter().map(|x| x as i8).collect();
        let b: Vec<i8> = fill(k * n, 14, 6).into_iter().map(|x| x as i8).collect();
        let run = |levels: usize, full: bool| {
            let mut c = vec![0i32; m * n];
            let (panels, accs) = strassen_scratch(Shape { m, k, n }, levels);
            if full {
                let mut panel = vec![Alphabet::<i8, Full<i8>>::ZERO; panels];
                let mut acc_buf = vec![0i128; accs];
                let av = MatView::row_major(as_alphabet_full(&a), m, k).unwrap();
                let bv = MatView::row_major(as_alphabet_full(&b), k, n).unwrap();
                let cv = MatViewMut::row_major(&mut c, m, n).unwrap();
                let mut t = Triple::new(av, bv, cv).unwrap();
                gemm_strassen(
                    &mut t,
                    &Linear::OVERWRITE,
                    GemmOptions {
                        encode: EncodeMode::Wrapping,
                        ..Default::default()
                    },
                    &mut Scratch::with_accumulators(&mut panel, &mut acc_buf),
                    levels,
                );
            } else {
                let mut panel = vec![Alphabet::<i8, Bnd<7>>::ZERO; panels];
                let mut acc_buf = vec![0i128; accs];
                let av = MatView::row_major(as_alphabet::<i8, Bnd<7>>(&a).unwrap(), m, k).unwrap();
                let bv = MatView::row_major(as_alphabet::<i8, Bnd<7>>(&b).unwrap(), k, n).unwrap();
                let cv = MatViewMut::row_major(&mut c, m, n).unwrap();
                let mut t = Triple::new(av, bv, cv).unwrap();
                gemm_strassen(
                    &mut t,
                    &Linear::OVERWRITE,
                    GemmOptions {
                        encode: EncodeMode::Wrapping,
                        ..Default::default()
                    },
                    &mut Scratch::with_accumulators(&mut panel, &mut acc_buf),
                    levels,
                );
            }
            c
        };
        // The bound-7 alphabet admits one level; full i8 admits none. All of
        // them are the same bytes.
        let want = run(0, false);
        assert_eq!(want, run(1, false));
        assert_eq!(want, run(3, false));
        assert_eq!(want, run(2, true));
    }

    /// CD-21: the plan is a pure function of the declarations, and each rule
    /// declines exactly what it says. These assertions are what keep the byte
    /// comparisons above honest: without them, "identical at every offer"
    /// could pass with the recursion never running.
    #[test]
    fn the_plan_says_what_it_takes_cd_21() {
        let shape = Shape {
            m: 64,
            k: 64,
            n: 64,
        };
        let (p2, a2) = strassen_scratch(shape, 2);
        // Even shape, headroom to spare, full offer: the request is taken.
        assert_eq!(levels::<i32>(shape, 128, p2, a2, 2), 2);
        assert_eq!(
            levels::<i32>(shape, 128, p2, a2, 5),
            2,
            "the offer caps the request"
        );
        // A starved offer admits strictly less.
        assert!(levels::<i32>(shape, 128, p2 / 4, a2 / 4, 2) < 2);
        assert_eq!(levels::<i32>(shape, 128, 0, 0, 2), 0, "no offer, no levels");
        // An odd extent declines, however good the offer.
        assert_eq!(
            levels::<i32>(
                Shape {
                    m: 63,
                    k: 64,
                    n: 64
                },
                128,
                p2,
                a2,
                2
            ),
            0
        );
        // The headroom rule: 4 * 2^29 > 2^31 - 1, so bound 2^29 admits no level;
        // 4 * 2^28 <= 2^31 - 1 admits one.
        let tight = Shape {
            m: 64,
            k: 64,
            n: 64,
        };
        assert_eq!(levels::<i32>(tight, 1 << 29, p2, a2, 1), 0);
        assert_eq!(levels::<i32>(tight, 1 << 28, p2, a2, 1), 1);
        // The full i32 alphabet admits none: a sum of two full-range values is
        // not an i32.
        assert_eq!(levels::<i32>(shape, Full::<i32>::VALUE, p2, a2, 1), 0);
        // Auto-selection is capped by the measured threshold; an explicit
        // request is not. At 64 the threshold (256) admits nothing.
        assert_eq!(levels::<i32>(shape, 128, p2, a2, usize::MAX), 0);
        assert_eq!(levels::<i32>(shape, 128, p2, a2, 2), 2);
        // i8 has no measured crossover, so auto-selection never fires there.
        assert_eq!(levels::<i8>(shape, 7, p2, a2, usize::MAX), 0);
        assert_eq!(
            levels::<i8>(shape, 7, p2, a2, 1),
            1,
            "an explicit request is a declaration"
        );
    }
}
