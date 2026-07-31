//! GEMM over float operands (§3.3, S6).
//!
//! Not a second driver, and not a second method. A float is a code; it decodes
//! to an exact dyadic rational; the products land in a complete accumulator
//! where every add is exact; and the result is rounded once, at the end. That
//! is the same sentence [`crate::gemm`] realizes, at a different instantiation
//! of the same three steps.
//!
//! The library never adds two floats. Every addition below happens in a
//! fixed-point integer register, which is why the result is
//! schedule-independent, tile-independent, and substrate-independent --- and
//! why it is *not* bit-identical to a classical `sgemm` (N1).

use uor_matmul_core::{
    as_alphabet_full, as_alphabet_full_mut, AccOf, Accumulator, Complete, Element, EncodeFrom,
    FloatElement, Full, MatView, PackedCode, Shape, Triple,
};

use crate::driver::GemmOptions;
use crate::epilogue::{Epilogue, PlaceAt, Scaled};
use crate::kernel::Kernelized;
use crate::scratch::Scratch;

/// `C := epilogue(A * B, C)`, over float operands, computed exactly.
///
/// Returns `()`, for the same reason [`crate::gemm`] does: the requested
/// product exists, because a [`Triple`] exists (R14, C6). Non-finite inputs are
/// codes and propagate by the IEEE rules; they are not an error condition
/// (`CT-03`).
///
/// With no panels to offer, the offer question [`gemm_float_packed`] asks is
/// answered before it is asked: the placement bridge's reification has nowhere
/// to live, so the scalar lanes run --- at the same bytes (`CD-19`). That is
/// why this entry carries no `EncodeFrom<i128>` bound: the kernel-table lane
/// is unreachable through it, and a caller with panels to offer gets it from
/// [`gemm_float_packed`].
pub fn gemm_float<E, O, Ep>(
    triple: &mut Triple<'_, '_, '_, E, O>,
    epilogue: &Ep,
    options: GemmOptions,
) where
    E: FloatElement,
    O: EncodeFrom<AccOf<E>> + Copy,
    Ep: Epilogue<E, O>,
    AccOf<E>: SignedPlace,
{
    gemm_float_scalar(triple, epilogue, options, &mut [], &mut [], None)
}

/// The same operation, with somewhere to decode into.
///
/// A float is a code, and decoding it is real work: a bit test, two shifts, a
/// mask, and a branch. The naive traversal decodes `B[p][j]` once for every row
/// of `A`, so every element of `B` is decoded `m` times and every element of
/// `A` is decoded `n` times. Decoding once into a panel and multiplying many
/// times removes both factors, which is the same structural point the integer
/// driver makes by packing --- and here it is worth more, because a decode
/// costs far more than a copy.
///
/// The panels are the caller's, so this still allocates nothing. Offering none
/// runs the streaming traversal, which decodes per element and gives the same
/// bytes (S13, `CD-04`).
///
/// # Which factorization runs
///
/// The offer decides, and the bytes never do (`CD-19`). A `PackedCode` is
/// sixteen bytes, so a panel offer re-reads as four `i32` words per code; when
/// the offer holds the placement bridge's reified operands plus a full-depth
/// kernel panel pair, the panels' measured spans admit the `i32` alphabet,
/// and the declared lane holds the whole depth, the reduction runs on the
/// integer kernel table --- the same identity, walked by the table's
/// instructions rather than the scalar lanes (R13). [`suggested_float_panels`]
/// names the offer that admits every factorization the shape supports;
/// anything short of it runs the widest lane the offer and the spans do
/// admit, at the same bytes. A depth past the lane declines to the scalar
/// lanes: the chunked traversal the table would run instead keeps its exact
/// partial sums in an accumulator offer a panel buffer cannot spell, and
/// [`gemm_float_bridged`] is the entry whose offers can.
pub fn gemm_float_packed<E, O, Ep>(
    triple: &mut Triple<'_, '_, '_, E, O>,
    epilogue: &Ep,
    options: GemmOptions,
    pa: &mut [PackedCode],
    pb: &mut [PackedCode],
) where
    E: FloatElement,
    O: Element + EncodeFrom<AccOf<E>> + EncodeFrom<i128> + Copy,
    Ep: Epilogue<E, O>,
    AccOf<E>: SignedPlace + PlaceAt,
{
    let shape = triple.shape();
    if shape.m == 0 || shape.n == 0 {
        return;
    }

    // The offer question is free and is asked before the walk, which is not:
    // the walk costs `(m + n) * k` decodes against one placement saved from
    // each of `m * n * k` products, so it pays exactly when `m * n > m + n`.
    // A matrix-vector product takes the scalar path without paying the walk.
    let worth_asking = shape.m.saturating_mul(shape.n) > shape.m.saturating_add(shape.n); // R3-ok: a shape predicate, not a value
    let mut spans = None;
    if shape.k != 0 && worth_asking && bridge_possible::<E>() {
        // The reification fits, the first half of the offer question.
        if bridge_room(shape, 0, pb.len()) {
            let walked = {
                let (a, b) = (triple.a(), triple.b());
                measure_spans(a, b, shape)
            };
            spans = Some(walked);
            let (finite, a_span, b_span) = walked;
            if let Some(bridge) = finite.then(|| admits_bridge::<E>(a_span, b_span)).flatten() {
                // Past the reified operands the kernel table wants a
                // full-depth panel pair; below that the traversal it would run
                // is the per-tile one, and the scalar scaled lanes are the
                // faster factorization of the same identity. The tile's own
                // `mr` and `nr` are the question, at the bound the spans just
                // declared.
                let spec = <i32 as Kernelized>::exact_spec(options.backend, bridge.bound, shape.m);
                // And the lane must hold the whole depth. Past it the table
                // chunks the reduction, and the chunked traversal's exact
                // partial sums live in an accumulator offer this face cannot
                // spell --- a `PackedCode` panel is eight-byte aligned, an
                // `i128` accumulator sixteen. Without one the table runs its
                // per-tile chunk traversal, which packs both panels afresh
                // per output tile per chunk: measured (`CG-15`, a fill of a
                // few binades at `512` cubed) 2.5 Gmac/s against the scalar
                // scaled lanes' 4.3, so the decline is the faster
                // factorization of the same identity. The lane's depth is
                // the table's own capacity arithmetic at the declared bound,
                // not a threshold; a caller with accumulator room gets the
                // chunked traversal from [`gemm_float_bridged`], whose offers
                // can spell it.
                if shape.k <= spec.lane_depth(bridge.bound)
                    && bridge_room(shape, spec.mr + spec.nr, pb.len())
                {
                    let admitted = {
                        let words = bytemuck::cast_slice_mut::<PackedCode, i32>(pb);
                        let (scaled, rest) = words.split_at_mut(suggested_bridge_scaled(shape));
                        let mut scratch = Scratch::new(as_alphabet_full_mut(rest));
                        run_bridge(triple, epilogue, options, bridge, scaled, &mut scratch)
                    };
                    if admitted {
                        return;
                    }
                }
            }
        }
    }
    gemm_float_scalar(triple, epilogue, options, pa, pb, spans)
}

/// The offer question the default driver asks, in two asks because the
/// second term is the running tile's own: an offer of `codes` panel codes
/// re-reads as `WORDS_PER_CODE` words a code, and the bridge needs the
/// reification (`k * (m + n)` words) plus a full-depth panel pair for a tile
/// of `per_step` rows and columns (`k * per_step` more). Asked with
/// `per_step = 0` before the walk, when the tile is not yet known, it is the
/// reification half alone. The same kind of question the scratch suggestions
/// answer, and never a look at the data (R13).
pub(crate) fn bridge_room(shape: Shape, per_step: usize, codes: usize) -> bool {
    let words = codes.saturating_mul(WORDS_PER_CODE); // R3-ok: an offer size question
    let need = suggested_bridge_scaled(shape).saturating_add(shape.k.saturating_mul(per_step)); // R3-ok: an offer size question
    words >= need
}

/// The scalar lanes: the streaming and panel traversals, which is what every
/// decline --- no offer, a narrow offer, a span past the `i32` alphabet, an
/// element type whose significand never fits one --- runs, at the same bytes
/// (`CD-04`, `CD-19`).
///
/// `spans` is the walk's answer when the caller already paid for it, so a
/// bridge that measured and declined does not measure again: the prescaling
/// question below is asked of the same spans the admission was.
fn gemm_float_scalar<E, O, Ep>(
    triple: &mut Triple<'_, '_, '_, E, O>,
    epilogue: &Ep,
    options: GemmOptions,
    pa: &mut [PackedCode],
    pb: &mut [PackedCode],
    spans: Option<(bool, Span, Span)>,
) where
    E: FloatElement,
    O: EncodeFrom<AccOf<E>> + Copy,
    Ep: Epilogue<E, O>,
    AccOf<E>: SignedPlace,
{
    let shape = triple.shape();
    if shape.m == 0 || shape.n == 0 {
        return;
    }
    let reads_c = epilogue.reads_c();
    let (a, b, c) = triple.parts();

    // One row of `A` and one column of `B` is the smallest offer that removes a
    // whole factor of redundant decoding; below it, the streaming traversal
    // runs. The same identity, walked differently (S13).
    // `k == 0` takes this path too, and not as a special case: the sum over an
    // empty reduction is zero, the loop below computes exactly that, and the
    // epilogue still runs. It is named because the comparison does not catch it
    // --- `pa.len() < 0` is false for every `usize` --- and what followed was
    // `pb.len() / shape.k`, which panics on every build. `gemm_float` returns
    // `()` and has no failure to report (R14, `CT-04`).
    if shape.k == 0 || pa.len() < shape.k || pb.len() < shape.k {
        for i in 0..shape.m {
            for j in 0..shape.n {
                let mut acc = <AccOf<E> as Accumulator>::ZERO;
                for p in 0..shape.k {
                    acc.accumulate_one(a.at(i, p).pack(), b.at(p, j).pack());
                }
                let prior = if reads_c { Some(*c.at(i, j)) } else { None };
                *c.at_mut(i, j) = epilogue.finish(acc, prior, options.encode);
            }
        }
        return;
    }

    // Columns of `B` that fit the offer at once. Every column decodes `B`'s
    // whole depth once and then serves every row of `A`.
    let block = (pb.len() / shape.k).min(shape.n).max(1);

    // The element type's significand width decides whether a product of two of
    // them can leave an `i64`. It is a constant of the type, not of the data.
    let product_fits = 2 * E::SIGNIFICAND_BITS <= 63;

    // Whether both panels can be scaled to a common base is one decision for the
    // whole call, and it has to be: the scaling is written *into* the panels, so a
    // block scaled for one row and read unscaled by another would be read wrong.
    // Deciding it globally costs one walk of each operand's exponents --- `m * k`
    // and `k * n` against `m * k * n` products --- and makes the two panel
    // formats impossible to mix.
    // The walk costs `(m + n) * k` decodes and the scaling saves one placement
    // from each of `m * n * k` products. A decode and a placement are the same
    // order of work, so it pays exactly when `m * n > m + n` --- which is false
    // for a matrix-vector product, where the walk would more than double the
    // whole call, and true for everything with two real dimensions.
    let worth_asking = shape.m.saturating_mul(shape.n) > shape.m.saturating_add(shape.n); // R3-ok: a shape predicate, not a value
    let prescaled = if !worth_asking {
        None
    } else {
        let (finite, a_span, b_span) = spans.unwrap_or_else(|| measure_spans(a, b, shape));
        finite
            .then(|| admits::<E>(shape.k, a_span, b_span))
            .flatten()
            .map(|scale| (scale, a_span.base(), b_span.base()))
    };

    let mut j0 = 0;
    while j0 < shape.n {
        let cols = block.min(shape.n - j0);
        // Whether this block of `B` is finite is settled here, while its codes
        // are being walked anyway, rather than once per product afterwards.
        let mut b_finite = true;
        for (jj, j) in (j0..j0 + cols).enumerate() {
            let dst = &mut pb[jj * shape.k..jj * shape.k + shape.k];
            for (slot, v) in dst.iter_mut().zip(b.column_walk(0, j, shape.k)) {
                *slot = v.pack();
                b_finite &= slot.is_finite();
            }
            if let Some((_, _, base_b)) = prescaled {
                rescale(dst, base_b);
            }
        }

        for i in 0..shape.m {
            // Decode row `i` of `A` once, and serve every column of this block
            // from it. Walking rather than indexing, for the reason the integer
            // packer walks: two multiplies per element is most of the cost of a
            // decode that is otherwise a handful of bit operations.
            let mut a_finite = true;
            for (slot, v) in pa[..shape.k].iter_mut().zip(a.row_walk(i, 0, shape.k)) {
                *slot = v.pack();
                a_finite &= slot.is_finite();
            }
            if let Some((_, base_a, _)) = prescaled {
                rescale(&mut pa[..shape.k], base_a);
            }
            let panels = PanelFacts {
                finite: a_finite && b_finite,
                product_fits,
                prescaled: prescaled.map(|(scale, _, _)| scale),
            };
            for (jj, j) in (j0..j0 + cols).enumerate() {
                let mut acc = <AccOf<E> as Accumulator>::ZERO;
                acc.accumulate_panels(
                    &pa[..shape.k],
                    &pb[jj * shape.k..jj * shape.k + shape.k],
                    panels,
                );
                let prior = if reads_c { Some(*c.at(i, j)) } else { None };
                *c.at_mut(i, j) = epilogue.finish(acc, prior, options.encode);
            }
        }
        j0 += cols;
    }
}

/// How much `i32` room [`gemm_float_bridged`] can use for this shape: the two
/// reified integer operands, `k * (m + n)` elements.
///
/// A *query*, like [`crate::suggested_scratch`]. Offering less is not an
/// error; it selects the panel traversal, at the same bytes (`CD-19`).
pub fn suggested_bridge_scaled(shape: Shape) -> usize {
    shape.k.saturating_mul(shape.m.saturating_add(shape.n)) // R3-ok: a scratch size query
}

/// The `i32` words one [`PackedCode`] re-reads as. Sixteen bytes against
/// four: the layout named its padding word so that this is a fact of the
/// type rather than of a build.
const WORDS_PER_CODE: usize = size_of::<PackedCode>() / size_of::<i32>();

/// The panel offer that lets the default float driver run the widest
/// factorization this shape admits: `(pa, pb)`, in [`PackedCode`] elements.
///
/// A *query*, like [`crate::suggested_scratch`] --- offering less is not an
/// error, it selects a narrower factorization at the same bytes (`CD-04`,
/// `CD-19`). `pa` is one row of `A`, the offer that removes a factor of
/// redundant decoding in the scalar lanes. `pb` is the larger of the two
/// wants the driver can have at this shape: the scalar lanes' decode panels
/// (`k * n` codes), and the placement bridge's --- the reified `i32` operands
/// plus the kernel table's own suggestion, at [`WORDS_PER_CODE`] words to a
/// code.
pub fn suggested_float_panels(shape: Shape) -> (usize, usize) {
    let scalar = shape.k.saturating_mul(shape.n); // R3-ok: a scratch size query
    let bridged = suggested_bridge_scaled(shape)
        .saturating_add(crate::suggested_scratch(shape)) // R3-ok: a scratch size query
        .div_ceil(WORDS_PER_CODE);
    (shape.k, scalar.max(bridged))
}

/// Whether any span of this element type can admit the bridge.
///
/// The admission's first term (`admits_bridge`), asked of the type's own
/// width before the walk prices it: a scaled significand is an `i32` only if
/// `p + w <= 31`, and `w` is never negative, so a significand past 31 bits
/// declines at every span and the walk is not paid to learn it. An `f64`
/// never crosses; that is the element type's arithmetic answering, not a
/// branch on which float it is.
const fn bridge_possible<E: FloatElement>() -> bool {
    // One bit of the `i32` is the sign, so the alphabet's magnitude is 31
    // bits: `p + w <= 31` with `w` never negative needs `p < 32`.
    E::SIGNIFICAND_BITS < i32::BITS
}

/// `C := epilogue(A * B, C)` over float operands, with the reduction handed to
/// the integer kernel table when the panels' exponent spans admit one.
///
/// Not a second driver, and not a second method (R13). The scaled panels are
/// exact integers, so their product is an exact integer dot product at one
/// known scale ---
///
/// ```text
///   sum_p (a_p 2^(ea_p - base_a)) (b_p 2^(eb_p - base_b))
///     = 2^-(base_a + base_b) * sum_p a_p b_p 2^(ea_p + eb_p)
/// ```
///
/// --- and the kernel table is what computes integer dot products in this
/// library, at an order of magnitude more throughput than a scalar loop. The
/// bridge is the two ends of that sentence: the panels reified as the integer
/// alphabet they already are, and the table's exact `i128` sum placed into the
/// float accumulator at `2^-(base_a + base_b)` by [`Scaled`], through the
/// decode's own primitive. Nothing is approximated, which is what `CD-19`
/// asserts byte for byte against the streaming traversal.
///
/// The admission is a declaration from the panels' measured spans, exactly as
/// [`Prescaled`]'s is: each scaled significand must be an element of the
/// alphabet the table multiplies, which is `i32`. What does *not* appear in it
/// is the reduction's depth --- the table's own lane-capacity machinery chunks
/// a deep reduction and folds the chunks into the exact accumulator, the same
/// machinery the integer driver has always used.
///
/// This is the explicit entry, for the caller who holds the bridge's offers
/// by name. The default driver ([`gemm_float_packed`]) runs the same body
/// when its panel offer re-reads as the room the bridge needs, which is what
/// [`suggested_float_panels`] names; the two are one factorization, not two
/// implementations (R13).
///
/// # Offers
///
/// `scaled` is where the reified operands live: [`suggested_bridge_scaled`]
/// elements. `scratch` is the kernel table's ordinary offer, accumulators
/// included for a deep `k`. `pa` and `pb` are the decode panels the scalar
/// lanes read on a decline. Every one is an offer: too little, or none, runs
/// the scalar lanes at the same bytes (`CD-19`, S13).
///
/// A matrix-vector product is not walked: the span walk costs `(m + n) * k`
/// decodes against `m * n * k` products saved, which is false when one
/// dimension is one, so the question is not asked --- the same comparison the
/// panel traversal makes.
#[allow(clippy::too_many_arguments)]
pub fn gemm_float_bridged<E, O, Ep>(
    triple: &mut Triple<'_, '_, '_, E, O>,
    epilogue: &Ep,
    options: GemmOptions,
    pa: &mut [PackedCode],
    pb: &mut [PackedCode],
    scaled: &mut [i32],
    scratch: &mut Scratch<'_, i32, Full<i32>>,
) where
    E: FloatElement,
    O: Element + EncodeFrom<AccOf<E>> + EncodeFrom<i128> + Copy,
    Ep: Epilogue<E, O>,
    AccOf<E>: SignedPlace + PlaceAt,
{
    let shape = triple.shape();
    if shape.m == 0 || shape.n == 0 {
        return;
    }

    // The offer question is free and is asked before the walk, which is not.
    let worth_asking = shape.m.saturating_mul(shape.n) > shape.m.saturating_add(shape.n); // R3-ok: a shape predicate, not a value
    if shape.k == 0 || !worth_asking || scaled.len() < suggested_bridge_scaled(shape) {
        gemm_float_scalar(triple, epilogue, options, pa, pb, None);
        return;
    }

    let walked = {
        let (a, b) = (triple.a(), triple.b());
        measure_spans(a, b, shape)
    };
    let (finite, a_span, b_span) = walked;
    let Some(bridge) = finite.then(|| admits_bridge::<E>(a_span, b_span)).flatten() else {
        // The spans do not fit the integer alphabet --- a non-finite code, an
        // `f64` significand, or a span past 31 bits. The scalar lanes' own
        // scaled lanes reach further, and the per-product placement reaches
        // further still. The walk's answer goes with the decline, so it is
        // not paid twice.
        gemm_float_scalar(triple, epilogue, options, pa, pb, Some(walked));
        return;
    };

    if !run_bridge(triple, epilogue, options, bridge, scaled, scratch) {
        gemm_float_scalar(triple, epilogue, options, pa, pb, Some(walked));
    }
}

/// The bridge's far half, shared by the explicit entry and the default
/// driver's auto-selection so that there is one body for the table
/// factorization, not two (R13): the panels reified as the `i32` alphabet
/// they already are, and the reduction handed to the kernel table, with the
/// table's exact sum placed at the bridge's one scale by [`Scaled`].
///
/// Returns `false` only when the reified triple cannot be formed, which is
/// unreachable --- the shapes are the caller's own and the buffers are
/// exactly `m * k` and `k * n` --- and the caller then walks the scalar
/// lanes, at the same bytes (`CD-19`). A decline rather than an assertion,
/// because the operation returns `()` and has no failure to report (R14).
fn run_bridge<E, O, Ep>(
    triple: &mut Triple<'_, '_, '_, E, O>,
    epilogue: &Ep,
    options: GemmOptions,
    bridge: Bridge,
    scaled: &mut [i32],
    scratch: &mut Scratch<'_, i32, Full<i32>>,
) -> bool
where
    E: FloatElement,
    O: Element + EncodeFrom<AccOf<E>> + EncodeFrom<i128> + Copy,
    Ep: Epilogue<E, O>,
    AccOf<E>: SignedPlace + PlaceAt,
{
    let shape = triple.shape();

    // Reify the scaled panels as the integer matrices the identity says they
    // are. After this their exponents are spent: every significand carries its
    // own as magnitude, and the one exponent left is the bridge's.
    let (ia, ib) = scaled.split_at_mut(shape.m * shape.k);
    {
        let (a, b) = (triple.a(), triple.b());
        for i in 0..shape.m {
            let row = &mut ia[i * shape.k..(i + 1) * shape.k];
            for (slot, v) in row.iter_mut().zip(a.row_walk(i, 0, shape.k)) {
                *slot = scaled_i32(v.pack(), bridge.base_a);
            }
        }
        for p in 0..shape.k {
            let row = &mut ib[p * shape.n..(p + 1) * shape.n];
            for (slot, v) in row.iter_mut().zip(b.row_walk(p, 0, shape.n)) {
                *slot = scaled_i32(v.pack(), bridge.base_b);
            }
        }
    }

    let integer = {
        let av = MatView::row_major(as_alphabet_full(ia), shape.m, shape.k);
        let bv = MatView::row_major(as_alphabet_full(ib), shape.k, shape.n);
        match (av, bv) {
            (Some(av), Some(bv)) => Triple::new(av, bv, triple.c_mut().reborrow()).ok(),
            _ => None,
        }
    };
    let Some(mut integer) = integer else {
        return false;
    };

    let placed = Scaled::<E, Ep>::new(epilogue, bridge.base());
    crate::kernel::gemm_packed_exact_at(
        &mut integer,
        &placed,
        options,
        scratch,
        bridge.bound,
        true,
    );
    true
}

/// One walk of each operand's exponents: whether every code is finite, and
/// the span each panel's exponents cover.
///
/// The prescaling decision and the bridge's admission are both settled for the
/// whole call, not per panel, because the scaling is written *into* the panels
/// --- a block scaled for one row and read unscaled by another would be read
/// wrong. Deciding once costs this walk: `m * k` and `k * n` packs against
/// `m * k * n` products.
fn measure_spans<E: FloatElement>(
    a: &MatView<'_, E>,
    b: &MatView<'_, E>,
    shape: Shape,
) -> (bool, Span, Span) {
    let mut finite = true;
    let mut a_span = Span::EMPTY;
    let mut b_span = Span::EMPTY;
    for i in 0..shape.m {
        for v in a.row_walk(i, 0, shape.k) {
            let code = v.pack();
            finite &= code.is_finite();
            a_span.see(code);
        }
    }
    for j in 0..shape.n {
        for v in b.column_walk(0, j, shape.k) {
            let code = v.pack();
            finite &= code.is_finite();
            b_span.see(code);
        }
    }
    (finite, a_span, b_span)
}

/// The bridge's admission, as a declaration from the measured spans.
///
/// The whole question is whether each scaled significand is an element of the
/// alphabet the kernel table multiplies: `i32`, whose magnitude reaches
/// `2^31`. A `f32` significand is 24 bits, so a panel admits seven binades of
/// span; an `f64` significand is 53 and admits none, which is the element
/// type's own arithmetic answering, not a branch on which float it is. The
/// reduction's depth is no term here: the table's lane capacity chunks it, as
/// it does for every integer caller.
///
/// `pub(crate)` for the symbol lane (`CD-20`), whose alphabet is the same
/// scaled significand and whose admission is therefore the same verdict.
pub(crate) struct Bridge {
    /// The base `A`'s significands were scaled to: the panel's lowest exponent.
    pub(crate) base_a: i32,
    /// The same for `B`.
    pub(crate) base_b: i32,
    /// The declared alphabet bound: the widest scaled significand either panel
    /// can hold. The lane depth is derived from it, so it is the honest
    /// maximum and not a tuned one --- a bound below the data's true magnitude
    /// is a wrong answer, not a fast one.
    pub(crate) bound: u128,
}

impl Bridge {
    /// The exponent of bit 0 of every scaled product: `base_a + base_b`.
    fn base(&self) -> i32 {
        self.base_a.saturating_add(self.base_b) // R3-ok: an exponent base, not an accumulation
    }
}

/// The measured spans, read as an admission verdict.
pub(crate) fn admits_bridge<E: FloatElement>(a: Span, b: Span) -> Option<Bridge> {
    let p = E::SIGNIFICAND_BITS;
    let (wa, wb) = (a.width(), b.width());
    // One bit of the `i32` is the sign, so the alphabet's magnitude is 31 bits.
    const ALPHABET: u32 = i32::BITS - 1;
    if p.checked_add(wa)? > ALPHABET || p.checked_add(wb)? > ALPHABET {
        return None;
    }
    Some(Bridge {
        base_a: a.base(),
        base_b: b.base(),
        bound: 1u128 << (p + wa.max(wb)),
    })
}

/// Decode one code with its significand scaled to `base`, as the `i32` the
/// bridge's admission just proved it is.
///
/// The `as` cannot truncate: `admits_bridge` measured the panel's span and
/// admitted only `p + w <= 31`, so the scaled magnitude is below `2^31` by that
/// arithmetic (R5). A zero significand carries no exponent and scales to zero,
/// exactly as [`rescale`] leaves it.
#[inline(always)]
fn scaled_i32(code: PackedCode, base: i32) -> i32 {
    if code.mantissa == 0 {
        0
    } else {
        (code.mantissa << code.exp.wrapping_sub(base) as u32) as i32
    }
}

/// Decode one code with its significand scaled to `base`, kept in the
/// element's own storage: the `f32` whose value is `mantissa * 2^(exp -
/// base)`.
///
/// The symbol tabulation lane's half of the reification [`scaled_i32`] does
/// for the bridge. The table build's element type is the family's own ---
/// the `Lane<E>` contract reads `E` --- so the scaled alphabet travels as
/// honest `f32` values and the lane's mac decodes them again, where the
/// bridge hands the kernel table a reified `i32`. The value is exact: the
/// admission that produced `base` measured the panel's span, so `exp - base`
/// lies in `[0, 31 - p]` and the scaled significand needs at most 31 bits'
/// worth of magnitude, which a 24-bit significand at a shifted exponent
/// spells without rounding. The construction is bit surgery, not float
/// arithmetic: a zero significand scales to a positive zero (its sign could
/// not survive the integer alphabet either), and anything else is a normal
/// float whose exponent field cannot reach the non-finite encodings.
pub(crate) fn scaled_f32(code: PackedCode, base: i32) -> f32 {
    if code.mantissa == 0 {
        return 0.0;
    }
    let negative = code.mantissa < 0;
    let mag = code.mantissa.unsigned_abs();
    // `top` is the position of the highest set bit, so `mag = 1.f * 2^top`
    // with `f` the stored fraction: `mag * 2^(exp - base)` is the float whose
    // biased exponent is `127 + top + exp - base`.
    let top = 31 - (mag as u32).leading_zeros() as i32; // R3-ok: a bit position, not an accumulation
    let shift = top + code.exp.wrapping_sub(base); // R3-ok: an exponent, not an accumulation
                                                   // Admission bounds the shift to `[0, 31 - p]`, so the field lands in
                                                   // `[127 + 0, 127 + 23 + 7]`: a normal float, never a subnormal encoding
                                                   // and never the non-finite one.
    let biased = (127 + shift) as u32; // R3-ok: an exponent field, not an accumulation
    let fraction = ((mag as u32) << (23 - top)) & 0x7F_FFFF;
    f32::from_bits(((negative as u32) << 31) | (biased << 23) | fraction)
}

/// A limb window over a complete accumulator (D-12).
///
/// Every product must land at its own position for the sum to be exact, and
/// that is what makes a complete accumulator expensive: a spread across limbs
/// and a carry, per product.
///
/// But a *run* of products whose exponents share a limb can be summed in one
/// 128-bit register first and placed once. For weights and activations of
/// similar magnitude --- which is the ordinary case, and the whole reason
/// quantization works --- that is nearly every product, and the cost per
/// product falls to one shift and one add.
///
/// It is not an approximation and not a fast path. The window holds an exact
/// integer at a known scale; flushing it adds that integer at that scale. What
/// changes is how often the wide register is touched, and nothing else.
struct Window<'a, const L: usize, const MIN_EXP: i32> {
    acc: &'a mut Complete<L, MIN_EXP>,
    /// The limb the window sits at, or `usize::MAX` when it is empty.
    at: usize,
    /// The window's contents, at scale `64 * at`.
    bits: i128,
}

impl<const L: usize, const MIN_EXP: i32> Window<'_, L, MIN_EXP> {
    #[inline(always)]
    fn place(&mut self, mantissa: i64, exp: i32) {
        if mantissa == 0 {
            return;
        }
        let shift = exp.wrapping_sub(MIN_EXP);
        if shift < 0 {
            // Below the register's floor. Unreachable for any pair of finite
            // values of the element type this register was sized for.
            return;
        }
        let at = (shift as u32 / 64) as usize;
        let bit = shift as u32 % 64;
        let value = i128::from(mantissa) << bit;

        if at == self.at {
            // The window's own capacity decides when to flush, and it is asked
            // rather than assumed. A term reaches `2^125` for `f64`, so any
            // fixed count would be either wrong or needlessly small --- and a
            // fixed count is an arbitrary ceiling, which R8 does not permit.
            // `checked_add` is the exact question, and its `None` branch is
            // taken about once per `2^60` products for realistic data.
            if let Some(sum) = self.bits.checked_add(value) {
                self.bits = sum;
                return;
            }
        }
        self.flush();
        self.at = at;
        self.bits = value;
    }

    #[inline]
    fn flush(&mut self) {
        if self.at == usize::MAX || self.bits == 0 {
            return;
        }
        // The window holds an exact integer at scale `64 * at`, and placing a
        // magnitude at a scale is exactly what `add_scaled` does: one three-limb
        // spread and one carry, against the four separate `add_signed` calls
        // this used to make by cutting the window into 63-bit pieces. Realistic
        // data flushes whenever the product exponent crosses a limb boundary, so
        // this is not a rare path.
        self.acc.add_scaled(
            self.bits.unsigned_abs(),
            MIN_EXP + (self.at as i32) * 64,
            self.bits < 0,
        );
        self.bits = 0;
    }
}

/// Accumulate a whole dot product of two decoded panels.
///
/// Straight-line for the finite case, which is every product in every ordinary
/// matrix, with one predictable branch guarding the IEEE clause 6 rules.
#[inline]
fn accumulate_run<const L: usize, const MIN_EXP: i32>(
    acc: &mut Complete<L, MIN_EXP>,
    pa: &[PackedCode],
    pb: &[PackedCode],
    panels: PanelFacts,
) {
    let mut window = Window {
        acc,
        at: usize::MAX,
        bits: 0,
    };

    // Both panels finite, and the element type's significands narrow enough that
    // no product can leave an `i64`: then the loop is the three lines it should
    // be, with no per-product test of anything. Both are facts about the *panel*
    // and the *type*, established once, so this is the same arithmetic asked
    // fewer questions --- not a second method (R13). `CU-04` compares it against
    // the per-product traversal on the same operands.
    // Both panels scaled to one base: the reduction is an integer dot product
    // and the register is touched once. The 64-bit lane is the one that
    // vectorizes; the 128-bit lane is one wide multiply per product. Which is
    // admissible was decided by the panels' spans, and both are the same sum
    // (`CU-04`).
    if let Some(scale) = panels.prescaled {
        if scale.wide {
            let mut sum = 0i128;
            for (a, b) in pa.iter().zip(pb) {
                sum = sum.wrapping_add(i128::from(a.mantissa) * i128::from(b.mantissa));
            }
            window
                .acc
                .add_scaled(sum.unsigned_abs(), scale.base, sum < 0);
        } else {
            let mut sum = 0i64;
            for (a, b) in pa.iter().zip(pb) {
                sum = sum.wrapping_add(a.mantissa.wrapping_mul(b.mantissa));
            }
            window
                .acc
                .add_scaled(u128::from(sum.unsigned_abs()), scale.base, sum < 0);
        }
        return;
    }

    if panels.finite && panels.product_fits {
        for (a, b) in pa.iter().zip(pb) {
            window.place(a.mantissa * b.mantissa, a.exp + b.exp);
        }
        window.flush();
        return;
    }

    for (a, b) in pa.iter().zip(pb) {
        if a.is_finite() && b.is_finite() {
            // Does the product fit a signed 64-bit mantissa? For `f32` it
            // always does --- two 24-bit significands make 48 bits --- and for
            // `f64` it sometimes does. `checked_mul` asks exactly that
            // question, so there is no width constant to get wrong and no
            // element type this branch is tuned for.
            if let Some(product) = a.mantissa.checked_mul(b.mantissa) {
                window.place(product, a.exp + b.exp);
            } else {
                let sign = (a.mantissa < 0) != (b.mantissa < 0);
                let (ua, ub) = (a.mantissa.unsigned_abs(), b.mantissa.unsigned_abs());
                let (lo, hi) = (ua & 0xFFFF_FFFF, ua >> 32);
                let sgn = |v: u128| {
                    if sign {
                        -((v & ((1 << 62) - 1)) as i64)
                    } else {
                        (v & ((1 << 62) - 1)) as i64
                    }
                };
                let l = u128::from(lo) * u128::from(ub);
                let h = u128::from(hi) * u128::from(ub);
                let e = a.exp + b.exp;
                window.place(sgn(l), e);
                window.place(sgn(l >> 62), e + 62);
                window.place(sgn(h), e + 32);
                window.place(sgn(h >> 62), e + 94);
            }
            continue;
        }
        window.flush();
        if a.is_nan() || b.is_nan() {
            window.acc.set_nan();
            continue;
        }
        // An infinity times a zero is a NaN, by IEEE 754 clause 7.2; otherwise
        // the sign is the product of the two mantissa signs, which the packing
        // already arranged.
        let (inf, other) = if a.is_infinite() { (a, b) } else { (b, a) };
        if other.is_finite() && other.mantissa == 0 {
            window.acc.set_nan();
        } else {
            window
                .acc
                .set_infinity((inf.mantissa < 0) != (other.mantissa < 0));
        }
    }
    window.flush();
}

/// What a caller has already established about a pair of decoded panels.
///
/// Neither field can change a value. `finite` says the IEEE clause 6 rules have
/// nothing to do here, and `product_fits` is a property of the element type's
/// significand width --- `2 * 24 <= 63` for `f32`. Both are asked once instead
/// of once per product.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PanelFacts {
    /// Every code in both panels is finite.
    pub finite: bool,
    /// Every product of two significands of this element type fits an `i64`.
    pub product_fits: bool,
    /// Both panels hold significands already scaled to a common base, so the
    /// whole dot product is one integer sum placed once. See [`Prescaled`].
    pub prescaled: Option<Prescaled>,
}

impl PanelFacts {
    /// Establish nothing. Always admissible, and always the same answer.
    pub const UNKNOWN: Self = Self {
        finite: false,
        product_fits: false,
        prescaled: None,
    };
}

/// Both panels' significands scaled to one common base.
///
/// This is what removes the placement from the inner loop, and it is the whole
/// of why the float path costs what it costs. A complete accumulator is exact
/// because every product lands at *its own* position, and finding that position
/// is a shift, a limb index, and a carry --- per product. Measured, that
/// placement is the entire distance between 0.43 ns per product and 2.5.
///
/// It does not have to be per product. Write `a * 2^(ea - base_a)` into the
/// panel instead of `(a, ea)`, and likewise for `b`, and then
///
/// ```text
///   sum_p (a_p 2^(ea_p - base_a)) (b_p 2^(eb_p - base_b))
///     = 2^-(base_a + base_b) * sum_p a_p b_p 2^(ea_p + eb_p)
/// ```
///
/// so the float dot product *is* an integer dot product, at one known scale,
/// placed into the register once for the whole reduction. Nothing is
/// approximated: the scaled significands are exact integers and so is their
/// sum.
///
/// What it costs is width, and that is what makes it a declaration rather than a
/// mode. A significand of `P` bits scaled across a span of `w` exponents needs
/// `P + w` bits, a product needs `2P + wa + wb`, and a sum of `k` of them needs
/// `ceil(log2 k)` more. When that fits a signed 64-bit lane the loop is a plain
/// integer dot product and vectorizes; when it needs 128 the loop is one wide
/// multiply per product; when it needs more, the per-product placement is the
/// only sequence that computes this identity and it runs. All three are the same
/// sum --- `CU-04` asserts it --- and which one runs is decided by the panels'
/// exponent span, established while their codes are walked anyway, exactly as
/// [`PanelFacts::finite`] is.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Prescaled {
    /// The exponent of bit 0 of every scaled product: `base_a + base_b`.
    pub base: i32,
    /// The sum needs a 128-bit lane rather than a 64-bit one.
    pub wide: bool,
}

/// The exponent span of a panel, and the base its scaling starts from.
///
/// Zero significands are not seen: a zero contributes nothing to the sum at any
/// scale, so it cannot widen the span and must not.
///
/// `pub(crate)` for the symbol lane's span walk (`CD-20`): the tabulated
/// traversal measures the same spans by the same walk, and admission is the
/// same declaration.
#[derive(Clone, Copy)]
pub(crate) struct Span {
    min: i32,
    max: i32,
    any: bool,
}

impl Span {
    pub(crate) const EMPTY: Self = Self {
        min: i32::MAX,
        max: i32::MIN,
        any: false,
    };

    #[inline(always)]
    pub(crate) fn see(&mut self, code: PackedCode) {
        if code.mantissa != 0 {
            self.min = self.min.min(code.exp);
            self.max = self.max.max(code.exp);
            self.any = true;
        }
    }

    pub(crate) fn base(&self) -> i32 {
        if self.any {
            self.min
        } else {
            0
        }
    }

    /// How many bits a significand of this panel gains from the scaling.
    pub(crate) fn width(&self) -> u32 {
        if self.any {
            self.max.wrapping_sub(self.min) as u32
        } else {
            0
        }
    }
}

/// Does scaling both panels to a common base keep every intermediate exact, and
/// in which lane?
///
/// Every term is a count of bits, and every count comes from the element type or
/// from the panels themselves. There is no tuned constant and no threshold to
/// choose: the question is whether the widest value that can arise fits the lane,
/// and it is asked arithmetically.
fn admits<E: FloatElement>(k: usize, a: Span, b: Span) -> Option<Prescaled> {
    // An all-zero panel used to return here, on the argument that the sum is
    // zero at any base and the scaling is therefore exact. The sum is --- but the
    // *other* panel is still rescaled, and returning before the width guard below
    // meant its significands were shifted by however wide its exponent span
    // happened to be. Measured: a zero `A` against a `B` holding `1e-30` and
    // `1e30` panicked with "attempt to shift left with overflow", and in release
    // the shift is masked and the answer silently wrong.
    //
    // No special case is needed to fix it. `Span::base` and `Span::width` are
    // already zero for an empty span, so falling through gives the empty side a
    // width of zero and asks the guard about the side that has one.
    let p = E::SIGNIFICAND_BITS;
    let (wa, wb) = (a.width(), b.width());
    // Each scaled significand must itself stay inside a signed 64-bit slot,
    // because that is what the panel holds.
    if p.checked_add(wa)? > 62 || p.checked_add(wb)? > 62 {
        return None;
    }
    // `k` terms, so `ceil(log2 k)` carry bits above the widest product.
    let depth = if k <= 1 { 0 } else { (k - 1).ilog2() + 1 };
    let need = (2 * p)
        .checked_add(wa)?
        .checked_add(wb)?
        .checked_add(depth)?;
    let base = a.base().saturating_add(b.base()); // R3-ok: an exponent base, not an accumulation
    if need <= 62 {
        Some(Prescaled { base, wide: false })
    } else if need <= 126 {
        Some(Prescaled { base, wide: true })
    } else {
        None
    }
}

/// Scale a packed panel's significands to `base`, in place.
///
/// After this the panel's `exp` fields are spent: every significand carries its
/// own exponent as magnitude, and the one exponent left is the caller's `base`.
fn rescale(panel: &mut [PackedCode], base: i32) {
    for code in panel {
        if code.mantissa != 0 {
            code.mantissa <<= code.exp.wrapping_sub(base) as u32;
        }
    }
}

/// What the packed float loop needs from an accumulator.
///
/// A trait rather than an inherent method so that `gemm_float_packed` stays
/// generic over the element type while the hot path stays monomorphic.
pub trait SignedPlace {
    /// Accumulate a whole dot product of two decoded panels, exactly.
    fn accumulate_panels(&mut self, pa: &[PackedCode], pb: &[PackedCode], panels: PanelFacts);
    /// Accumulate one product of two decoded codes.
    fn accumulate_one(&mut self, a: PackedCode, b: PackedCode);
}

impl<const L: usize, const MIN_EXP: i32> SignedPlace for Complete<L, MIN_EXP> {
    #[inline]
    fn accumulate_panels(&mut self, pa: &[PackedCode], pb: &[PackedCode], panels: PanelFacts) {
        accumulate_run(self, pa, pb, panels);
    }

    #[inline]
    fn accumulate_one(&mut self, a: PackedCode, b: PackedCode) {
        accumulate_run(self, &[a], &[b], PanelFacts::UNKNOWN);
    }
}

#[cfg(test)]
// R7 governs the library, not its tests: these build operands on the heap so
// that awkward shapes and long reduction orders can be generated. `CA-01`
// witnesses the library's own zero-allocation claim with a counting allocator.
#[allow(clippy::disallowed_types)]
mod tests {
    use super::*;
    use crate::epilogue::Linear;
    use std::vec;
    use std::vec::Vec;
    use uor_matmul_core::{EncodeMode, MatView, MatViewMut};

    fn product(m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        {
            let av = MatView::row_major(a, m, k).unwrap();
            let bv = MatView::row_major(b, k, n).unwrap();
            let cv = MatViewMut::row_major(&mut c, m, n).unwrap();
            let mut t = Triple::new(av, bv, cv).unwrap();
            gemm_float(&mut t, &Linear::OVERWRITE, GemmOptions::default());
        }
        c
    }

    /// The same product with panels offered, which is the only way the prescaling
    /// path is reached: [`gemm_float`] offers none and therefore always streams.
    fn product_packed(m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        let mut pa = vec![PackedCode::default(); k];
        let mut pb = vec![PackedCode::default(); k * n];
        {
            let av = MatView::row_major(a, m, k).unwrap();
            let bv = MatView::row_major(b, k, n).unwrap();
            let cv = MatViewMut::row_major(&mut c, m, n).unwrap();
            let mut t = Triple::new(av, bv, cv).unwrap();
            gemm_float_packed(
                &mut t,
                &Linear::OVERWRITE,
                GemmOptions::default(),
                &mut pa,
                &mut pb,
            );
        }
        c
    }

    /// The same product through the bridge, with every offer named separately
    /// so the sweep can starve each in turn: the decode panels the fallback
    /// reads, the `i32` room the reified operands need, and the kernel table's
    /// own scratch and accumulator offers.
    #[allow(clippy::too_many_arguments)]
    fn product_bridged(
        m: usize,
        k: usize,
        n: usize,
        a: &[f32],
        b: &[f32],
        scaled: usize,
        panels: usize,
        kernel: usize,
        accs: usize,
    ) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        let mut pa = vec![PackedCode::default(); k.max(1)];
        let mut pb = vec![PackedCode::default(); panels];
        let mut scaled_buf = vec![0i32; scaled];
        let mut kernel_buf = vec![uor_matmul_core::Alphabet::of(0i32); kernel];
        let mut acc_buf = vec![0i128; accs];
        {
            let av = MatView::row_major(a, m, k).unwrap();
            let bv = MatView::row_major(b, k, n).unwrap();
            let cv = MatViewMut::row_major(&mut c, m, n).unwrap();
            let mut t = Triple::new(av, bv, cv).unwrap();
            gemm_float_bridged(
                &mut t,
                &Linear::OVERWRITE,
                GemmOptions::default(),
                &mut pa,
                &mut pb,
                &mut scaled_buf,
                &mut crate::scratch::Scratch::with_accumulators(&mut kernel_buf, &mut acc_buf),
            );
        }
        c
    }

    /// `CD-19`: the integer kernel table's lane gives the float driver's bytes.
    ///
    /// The bridge reifies the scaled panels as an `i32` alphabet and hands the
    /// reduction to the kernel table; the placed sum must be the same integer
    /// the scalar scaled lanes compute, at the same scale, so the output is
    /// the same bytes. The reference is the streaming traversal, which knows
    /// nothing about spans, panels, or kernels. The operands are chosen to
    /// reach every admission verdict --- the bridge, the 128-bit scalar lane,
    /// and the per-product placement --- because a differential test over
    /// operands that all take one path is a test of one path, and
    /// [`the_spans_select_the_bridge_cd_19`] says which is which.
    ///
    /// The significands are drawn from `[2^23, 2^24)`, so the decoded exponent
    /// is exactly the one the generator names: a 24-bit significand is the
    /// element type's own, and the span is the generator's and nothing else.
    #[test]
    fn the_kernel_table_lane_cannot_change_a_byte_cd_19() {
        for (label, span_a, span_b) in [
            ("one exponent", 0i32, 0i32),
            ("a few binades", 3, 4),
            ("asymmetric, at the alphabet's edge", 7, 0),
            ("too wide for the i32 alphabet", 8, 8),
            ("past every scaled lane", 90, 90),
        ] {
            for &(m, k, n) in &[
                (1usize, 1usize, 1usize),
                (2, 3, 5),
                (5, 7, 3),
                (16, 64, 8),
                (3, 129, 17),
            ] {
                let mut seed = 0x9E37_79B9_7F4A_7C15u64 ^ (span_a as u64) << 8 ^ (k as u64);
                let mut next = move || {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    (seed >> 33) as i64
                };
                let gen = |next: &mut dyn FnMut() -> i64, len: usize, span: i32| -> Vec<f32> {
                    (0..len)
                        .map(|_| {
                            let v = next();
                            let s = if span == 0 {
                                0
                            } else {
                                (v % span as i64) as i32
                            };
                            let m = 8_388_608 + v.unsigned_abs() % 8_388_607;
                            let x = m as f32 * 2.0f32.powi(s - span / 2);
                            if v & 1 == 0 {
                                -x
                            } else {
                                x
                            }
                        })
                        .collect()
                };
                let av = gen(&mut next, m * k, span_a);
                let bv = gen(&mut next, k * n, span_b);

                let want = product(m, k, n, &av, &bv);
                let suggested = suggested_bridge_scaled(uor_matmul_core::Shape { m, k, n });
                let kernel_full = crate::suggested_scratch(uor_matmul_core::Shape { m, k, n });
                // Every combination of starving the three offers: the scaled
                // room, the kernel table's panel scratch, and the decode panels
                // the fallback reads. All of them are the same bytes.
                for scaled in [0, suggested] {
                    for kernel in [0, kernel_full] {
                        for (panels, accs) in [(0, 0), (k * n, m * n)] {
                            let got =
                                product_bridged(m, k, n, &av, &bv, scaled, panels, kernel, accs);
                            assert_eq!(
                                got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                                want.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                                "{label}, {m}x{k}x{n}, scaled {scaled}, kernel {kernel}, \
                                 panels {panels}"
                            );
                        }
                    }
                }
            }
        }

        // A deep reduction past the lane's capacity: the kernel table chunks
        // and folds, the scalar lane would have to be the wide one, and the
        // bytes are still the streaming traversal's. Span `(2, 0)` at
        // `k = 4096` admits the bridge with a declared bound of `2^26`, whose
        // lane holds 2048 products --- half the depth.
        let (m, k, n) = (4usize, 4096usize, 6usize);
        let av: Vec<f32> = (0..m * k)
            .map(|i| (8_388_608 + (i as u64 * 37) % 8_388_607) as f32)
            .collect();
        let bv: Vec<f32> = (0..k * n)
            .map(|i| {
                let x = (8_388_608 + (i as u64 * 53) % 8_388_607) as f32;
                if i % 3 == 0 {
                    -x
                } else {
                    x
                }
            })
            .collect();
        // A span of two binades on one side only.
        let av: Vec<f32> = av
            .iter()
            .enumerate()
            .map(|(i, &x)| x * 2.0f32.powi((i % 3) as i32 - 1))
            .collect();
        let want = product(m, k, n, &av, &bv);
        let suggested = suggested_bridge_scaled(uor_matmul_core::Shape { m, k, n });
        let kernel_full = crate::suggested_scratch(uor_matmul_core::Shape { m, k, n });
        for (kernel, accs) in [(0, 0), (kernel_full, 0), (kernel_full, m * n)] {
            let got = product_bridged(m, k, n, &av, &bv, suggested, 0, kernel, accs);
            assert_eq!(
                got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                want.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                "deep and chunked, kernel {kernel}, accs {accs}"
            );
        }

        // A non-finite code declines the bridge outright and propagates by the
        // IEEE rules, through the same fallback the unoffered call takes.
        let av = [1.0f32, f32::INFINITY, 2.0, 3.0, f32::NAN, 4.0];
        let bv = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let want = product(2, 3, 2, &av, &bv);
        let got = product_bridged(2, 3, 2, &av, &bv, 3 * (2 + 2), 0, 1024, 0);
        assert_eq!(
            got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "non-finite codes decline the bridge and keep the driver's bytes"
        );
    }

    /// The verdicts the byte test above relies on, asserted rather than
    /// trusted: which spans the bridge admits, and at what declared bound.
    ///
    /// A differential test over operands that all decline would pass with the
    /// bridge dead, so the admission is pinned directly --- the same reason
    /// `each_scaled_lane_is_reached_cu_04` exists beside its own sweep.
    #[test]
    fn the_spans_select_the_bridge_cd_19() {
        let span = |exps: &[i32]| {
            let mut s = Span::EMPTY;
            for &e in exps {
                s.see(PackedCode {
                    mantissa: 1,
                    exp: e,
                    _pad: 0,
                });
            }
            s
        };
        // 24-bit significands: a scaled significand is an `i32` while the span
        // keeps it at or under 31 bits, and the declared bound is the widest
        // scaled significand either panel can hold.
        let tight = span(&[0]);
        assert_eq!(
            admits_bridge::<f32>(tight, tight).map(|b| b.bound),
            Some(1 << 24),
            "no span: the bare significand, at its own bound"
        );
        let some = span(&[0, 3]);
        let other = span(&[0, 4]);
        assert_eq!(
            admits_bridge::<f32>(some, other).map(|b| b.bound),
            Some(1 << 28),
            "the bound is the wider of the two panels"
        );
        let edge = span(&[0, 7]);
        assert!(
            admits_bridge::<f32>(edge, tight).is_some(),
            "31-bit scaled significands are still `i32`"
        );
        let past = span(&[0, 8]);
        assert!(
            admits_bridge::<f32>(past, tight).is_none(),
            "32-bit scaled significands are not"
        );
        let huge = span(&[0, 100]);
        assert!(
            admits_bridge::<f32>(huge, huge).is_none(),
            "a span past the alphabet declines"
        );
        // The scale is the two bases summed, and it is where the integer sum
        // is placed: one exponent off here is a wrong answer (`CD-19`).
        let low = span(&[-30, -27]);
        let high = span(&[40, 43]);
        let b = admits_bridge::<f32>(low, high).expect("narrow spans admit");
        assert_eq!(b.base(), -30 + 40);
        // `f64`'s 53-bit significand never fits the `i32` alphabet: the same
        // question, answered by the element type's own width.
        assert!(admits_bridge::<f64>(tight, tight).is_none());
    }

    /// The default driver's half of `CD-19`: the packed entry takes the
    /// kernel-table lane itself when the offer admits it, at the streaming
    /// traversal's bytes --- and the byte claim is only as good as the pin
    /// that the lane really ran, so the offer question and the lane's
    /// footprint are asserted alongside it.
    #[test]
    fn the_default_driver_takes_the_table_cd_19() {
        // The offer rule, pinned at its boundaries: the reification is
        // `k * (m + n)` words at four words a code, and the full-depth panel
        // pair is `k * per_step` more. `per_step = 8` is the portable tile's
        // `mr + nr`; a wider tile only asks for more.
        let s16 = Shape {
            m: 16,
            k: 16,
            n: 16,
        };
        assert!(bridge_room(s16, 8, 160), "640 words exactly admits");
        assert!(!bridge_room(s16, 8, 159), "one word short declines");
        assert!(bridge_room(s16, 0, 128), "the reification alone fits");
        assert!(!bridge_room(s16, 0, 127), "one code short of it does not");
        // The suggested offer admits the table at every shape the bench and
        // the sweep measure.
        for &(m, k, n) in &[
            (16usize, 16usize, 16usize),
            (128, 128, 128),
            (64, 512, 1024),
            (509, 1021, 257),
        ] {
            let shape = Shape { m, k, n };
            let (_, pb) = suggested_float_panels(shape);
            assert!(
                bridge_room(shape, 8, pb),
                "{m}x{k}x{n}: the suggestion admits"
            );
        }
        // A hand-sized `k * n` offer declines where the reification cannot
        // fit: a tall product's `A` is the bigger operand.
        assert!(!bridge_room(
            Shape {
                m: 1000,
                k: 8,
                n: 8
            },
            8,
            64
        ));
        // The element type's own width answers before the walk is paid for:
        // an `f32` can cross, an `f64` never does.
        assert!(bridge_possible::<f32>());
        assert!(!bridge_possible::<f64>());

        // The bytes: spans that admit the bridge, the packed entry offered
        // the suggestion, against the streaming traversal that knows nothing
        // about spans.
        let (m, k, n) = (16usize, 64usize, 8usize);
        let mut seed = 0x243F_6A88_85A3_08D3u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 33) as i64
        };
        let mut gen = |len: usize, span: i32| -> Vec<f32> {
            (0..len)
                .map(|_| {
                    let v = next();
                    let s = (v % span as i64) as i32;
                    let sig = 8_388_608 + v.unsigned_abs() % 8_388_607;
                    let x = sig as f32 * 2.0f32.powi(s - span / 2);
                    if v & 1 == 0 {
                        -x
                    } else {
                        x
                    }
                })
                .collect()
        };
        let av = gen(m * k, 3);
        let bv = gen(k * n, 4);
        let want = product(m, k, n, &av, &bv);

        let shape = Shape { m, k, n };
        let (pa_len, pb_len) = suggested_float_panels(shape);
        let mut pa = vec![PackedCode::default(); pa_len];
        let mut pb = vec![PackedCode::default(); pb_len];
        let mut got = vec![0.0f32; m * n];
        {
            let a = MatView::row_major(&av, m, k).unwrap();
            let b = MatView::row_major(&bv, k, n).unwrap();
            let c = MatViewMut::row_major(&mut got, m, n).unwrap();
            let mut t = Triple::new(a, b, c).unwrap();
            gemm_float_packed(
                &mut t,
                &Linear::OVERWRITE,
                GemmOptions::default(),
                &mut pa,
                &mut pb,
            );
        }
        assert_eq!(
            got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "the default driver's table lane changed a byte"
        );

        // And the table lane really ran: the reified operands are still
        // sitting in the offer it was handed. Had the scalar lanes run, `pb`
        // would hold packed codes rather than these words, which is the pin
        // that keeps the byte assertion above honest.
        let (finite, a_span, b_span) = {
            let a = MatView::row_major(&av, m, k).unwrap();
            let b = MatView::row_major(&bv, k, n).unwrap();
            measure_spans(&a, &b, shape)
        };
        let bridge = admits_bridge::<f32>(a_span, b_span).expect("these spans admit");
        assert!(finite, "this fill is finite");
        let mut reified: Vec<i32> = av
            .iter()
            .map(|v| scaled_i32(v.pack(), bridge.base_a))
            .collect();
        reified.extend(bv.iter().map(|v| scaled_i32(v.pack(), bridge.base_b)));
        let words: &[i32] = bytemuck::cast_slice(&pb);
        assert_eq!(
            &words[..reified.len()],
            reified.as_slice(),
            "the offer holds the reified operands, so the table lane ran"
        );

        // The depth term, pinned as the table's own capacity arithmetic: a
        // declared bound of `2^27` holds 511 products in the `i64` lane, so a
        // deeper reduction declines to the scalar lanes rather than chunking
        // without accumulator room. The figure is the family's `i64` lane cap
        // read against the bound, identical on every backend the family has.
        let spec = <i32 as Kernelized>::exact_spec(GemmOptions::default().backend, 1 << 27, 8);
        assert_eq!(spec.lane_depth(1 << 27), 511, "the lane's depth at 2^27");

        // A depth past the lane declines: spans that admit the alphabet but a
        // `k` past the lane. The bytes are the streaming traversal's, and the
        // offer shows the scalar lanes ran --- it holds packed codes, not the
        // reified operands the table lane would have left.
        let (m, k, n) = (16usize, 1024usize, 8usize);
        let av = gen(m * k, 3);
        let bv = gen(k * n, 4);
        let want = product(m, k, n, &av, &bv);
        let shape = Shape { m, k, n };
        let (pa_len, pb_len) = suggested_float_panels(shape);
        let mut pa = vec![PackedCode::default(); pa_len];
        let mut pb = vec![PackedCode::default(); pb_len];
        let mut got = vec![0.0f32; m * n];
        {
            let a = MatView::row_major(&av, m, k).unwrap();
            let b = MatView::row_major(&bv, k, n).unwrap();
            let c = MatViewMut::row_major(&mut got, m, n).unwrap();
            let mut t = Triple::new(a, b, c).unwrap();
            gemm_float_packed(
                &mut t,
                &Linear::OVERWRITE,
                GemmOptions::default(),
                &mut pa,
                &mut pb,
            );
        }
        assert_eq!(
            got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "a depth past the lane must still give the streaming bytes"
        );
        let (finite, a_span, b_span) = {
            let a = MatView::row_major(&av, m, k).unwrap();
            let b = MatView::row_major(&bv, k, n).unwrap();
            measure_spans(&a, &b, shape)
        };
        assert!(finite);
        let bridge = admits_bridge::<f32>(a_span, b_span).expect("the alphabet admits");
        let mut reified: Vec<i32> = av
            .iter()
            .map(|v| scaled_i32(v.pack(), bridge.base_a))
            .collect();
        reified.extend(bv.iter().map(|v| scaled_i32(v.pack(), bridge.base_b)));
        let words: &[i32] = bytemuck::cast_slice(&pb);
        assert_ne!(
            &words[..reified.len()],
            reified.as_slice(),
            "a depth past the lane must decline: the table lane did not run"
        );
    }

    /// `CS-05`, R14: an all-zero panel against one with a wide exponent span.
    ///
    /// `admits` returned early for an empty span, on the argument that the sum is
    /// zero at any base --- which is true of the sum and false of the *rescale*:
    /// the other panel's significands were still shifted by its whole exponent
    /// width, before the guard that keeps a scaled significand inside its slot.
    /// `1e-30` against `1e30` panicked with "attempt to shift left with
    /// overflow".
    ///
    /// It is a panic and not a wrong answer, and the difference is worth stating:
    /// the early return fires only when a whole panel is zero, and then every
    /// product is zero however the other panel's significands were mangled, so
    /// release --- where the shift is masked rather than checked --- returns the
    /// right bytes. A panic inside an operation that returns `()` is still a
    /// failure it has no way to report (R14), and this test runs in the profile
    /// `just vv` uses, which is the one that checks the shift.
    #[test]
    fn a_zero_panel_against_a_wide_span_is_exact_cs_05() {
        let wide = [1e-30f32, 1e30, 1.0, 1e-20, 1e20, 2.0, 1e-10, 1e10, 3.0];
        let zeros = [0.0f32; 9];
        assert_eq!(product_packed(3, 3, 3, &zeros, &wide), vec![0.0; 9]);
        assert_eq!(product_packed(3, 3, 3, &wide, &zeros), vec![0.0; 9]);
        // And a wide span on both sides, where no scaling can be admitted at all:
        // the traversal has to fall back to placing each product, not to a shift
        // that does not fit.
        assert_eq!(
            product_packed(3, 3, 3, &wide, &wide),
            product(3, 3, 3, &wide, &wide),
            "the packed and streaming traversals must agree on a wide span"
        );
    }

    /// `CT-04`: the degenerate shapes are shapes, not error conditions.
    ///
    /// `k == 0` is the one that mattered: the sum over an empty reduction is
    /// zero and the epilogue still runs, but the offer guard reads
    /// `pa.len() < shape.k`, which is false for every `usize` when `k` is zero,
    /// so control reached `pb.len() / shape.k` and `gemm_float` panicked with
    /// "attempt to divide by zero" --- on every build, release included, since
    /// integer division by zero is not a debug assertion. `gemm_float` returns
    /// `()` and has no failure to report (R14).
    #[test]
    fn degenerate_shapes_are_shapes_ct_04() {
        for &(m, k, n) in &[
            (2usize, 0usize, 2usize),
            (0, 2, 2),
            (2, 2, 0),
            (0, 0, 0),
            (1, 1, 1),
            (1, 0, 1),
            (3, 0, 7),
        ] {
            let a = vec![1.0f32; m * k];
            let b = vec![1.0f32; k * n];
            let got = product(m, k, n, &a, &b);
            assert_eq!(got.len(), m * n, "{m}x{k}x{n}");
            // Every cell is the empty sum where `k` is zero, and `k` products of
            // ones otherwise.
            assert!(
                got.iter().all(|&x| x == k as f32),
                "{m}x{k}x{n} gave {got:?}"
            );
        }
    }

    /// The ordinary case is ordinary: an exact small product is exactly right.
    #[test]
    fn small_exact_products_are_exact_cs_01() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [5.0f32, 6.0, 7.0, 8.0];
        assert_eq!(product(2, 2, 2, &a, &b), vec![19.0, 22.0, 43.0, 50.0]);
    }

    /// `CU-04`: the classic catastrophic-cancellation case. A classical `sgemm`
    /// gets `0.0` or `1.0` here depending on the order it happens to add in;
    /// the exact sum is `1.0` and there is no order in which this returns
    /// anything else.
    #[test]
    fn catastrophic_cancellation_is_exact_cu_04() {
        let big = 1.0e30f32;
        // 1e30 + 1.0 - 1e30. In f32, `1e30 + 1.0` rounds back to `1e30`, so a
        // classical accumulator loses the 1.0 entirely.
        let a = [1.0f32, 1.0, 1.0];
        let b = [big, 1.0, -big];
        assert_eq!(product(1, 3, 1, &a, &b), vec![1.0f32]);

        // And the naive float sum really does lose it, so this is not a
        // vacuous comparison.
        let naive = a.iter().zip(b.iter()).fold(0.0f32, |s, (x, y)| s + x * y);
        assert_eq!(naive, 0.0f32, "a classical accumulator loses the 1.0");
    }

    /// `CU-04`: shuffling the reduction order cannot change the answer, because
    /// nothing is rounded until the end. No classical `f32` GEMM has this
    /// property.
    #[test]
    fn float_accumulation_is_order_independent_cu_04() {
        let k = 257;
        let a: Vec<f32> = (0..k).map(|i| ((i * 37 % 1000) as f32) * 1.0e-3).collect();
        let b: Vec<f32> = (0..k).map(|i| ((i * 53 % 997) as f32) * 1.0e3).collect();
        let reference = product(1, k, 1, &a, &b);

        // The same terms in a different order.
        let mut order: Vec<usize> = (0..k).collect();
        for round in 0..8 {
            order.rotate_left(round * 7 + 1);
            let a2: Vec<f32> = order.iter().map(|&i| a[i]).collect();
            let b2: Vec<f32> = order.iter().map(|&i| b[i]).collect();
            assert_eq!(product(1, k, 1, &a2, &b2), reference, "round {round}");
        }
    }

    /// `CT-03`: non-finite codes propagate by the IEEE rules and never error.
    #[test]
    fn non_finite_codes_propagate_ct_03() {
        let inf = f32::INFINITY;
        assert!(product(1, 1, 1, &[inf], &[2.0])[0].is_infinite());
        assert!(product(1, 1, 1, &[-inf], &[2.0])[0] == f32::NEG_INFINITY);
        // Infinity times zero is a NaN, by clause 7.2.
        assert!(product(1, 1, 1, &[inf], &[0.0])[0].is_nan());
        // Opposite infinities in one sum are a NaN, whatever order they arrive.
        assert!(product(1, 2, 1, &[1.0, 1.0], &[inf, -inf])[0].is_nan());
        assert!(product(1, 2, 1, &[1.0, 1.0], &[-inf, inf])[0].is_nan());
        // A NaN anywhere absorbs.
        assert!(product(1, 3, 1, &[1.0, 1.0, 1.0], &[1.0, f32::NAN, 1.0])[0].is_nan());
    }

    /// `CD-05`: the encode mode is the only thing that changes the bytes for a
    /// fixed accumulation. `Nearest` rounds half to even; `TowardZero` truncates.
    ///
    /// The discriminating case has to be a sum that is *not* a tie and *not*
    /// representable, and this test had neither: it compared the two modes on a
    /// sum exactly half an ulp past 1.0, where nearest-even and truncation both
    /// keep 1.0 --- its own comment says so --- and then ran a second case that
    /// was a whole ulp, hence exact, hence unrounded, and ran it under `Nearest`
    /// only. Nothing here could fail on the encode mode, which is the one thing
    /// the ID names.
    ///
    /// Three quarters of an ulp is the case that separates them.
    #[test]
    fn the_encode_mode_is_the_only_variable_cd_05() {
        fn at_mode(a: &[f32], b: &[f32], encode: EncodeMode) -> f32 {
            let k = b.len();
            let mut c = [0.0f32];
            let av = MatView::row_major(a, 1, k).unwrap();
            let bv = MatView::row_major(b, k, 1).unwrap();
            let cv = MatViewMut::row_major(&mut c, 1, 1).unwrap();
            let mut t = Triple::new(av, bv, cv).unwrap();
            gemm_float(
                &mut t,
                &Linear::OVERWRITE,
                GemmOptions {
                    encode,
                    ..Default::default()
                },
            );
            c[0]
        }

        const HALF_ULP: f32 = f32::from_bits(0x3380_0000); // 2^-24
        const QUARTER_ULP: f32 = f32::from_bits(0x3300_0000); // 2^-25

        // `1 + 3/4 ulp`: above the midpoint, below the next value, and not a tie.
        // Nearest goes up, truncation stays. This is the assertion the ID is about.
        let ones = [1.0f32, 1.0, 1.0];
        let three_quarters = [1.0f32, HALF_ULP, QUARTER_ULP];
        let nearest = at_mode(&ones, &three_quarters, EncodeMode::Nearest);
        let toward_zero = at_mode(&ones, &three_quarters, EncodeMode::TowardZero);
        assert_eq!(
            nearest,
            1.0 + f32::EPSILON,
            "nearest rounds up past halfway"
        );
        assert_eq!(toward_zero, 1.0, "truncation keeps the smaller neighbour");
        assert_ne!(
            nearest, toward_zero,
            "the encode mode has to be able to change the bytes, or this ID \
             asserts nothing"
        );

        // And the tie, where they agree: half to even keeps the even mantissa and
        // truncation keeps the same value for its own reason. Recorded because it
        // is the case that *cannot* discriminate, which is what made this test
        // vacuous when it was the only case.
        let pair = [1.0f32, 1.0];
        let tie = [1.0f32, HALF_ULP];
        assert_eq!(at_mode(&pair, &tie, EncodeMode::Nearest), 1.0);
        assert_eq!(at_mode(&pair, &tie, EncodeMode::TowardZero), 1.0);

        // A sum that is exactly representable is unrounded, so both modes give it
        // back unchanged --- the third case, and the reason a full ulp could not
        // discriminate either.
        let exact = [1.0f32, f32::EPSILON];
        assert_eq!(
            at_mode(&pair, &exact, EncodeMode::Nearest),
            1.0 + f32::EPSILON
        );
        assert_eq!(
            at_mode(&pair, &exact, EncodeMode::TowardZero),
            1.0 + f32::EPSILON
        );
    }

    /// Subnormals are not a special case: gradual underflow falls out of the
    /// same clamp that puts the least significant bit at its floor.
    #[test]
    fn subnormals_round_correctly_ct_03() {
        let tiny = f32::from_bits(1); // 2^-149, the smallest subnormal
                                      // tiny * 1.0 is exactly tiny.
        assert_eq!(product(1, 1, 1, &[tiny], &[1.0]), vec![tiny]);
        // Four of them sum to 4 * 2^-149, still subnormal and still exact.
        let a = [1.0f32; 4];
        let b = [tiny; 4];
        assert_eq!(product(1, 4, 1, &a, &b), vec![f32::from_bits(4)]);
        // A product below the subnormal floor rounds to zero, once.
        let half_tiny = product(1, 1, 1, &[tiny], &[0.25]);
        assert_eq!(half_tiny, vec![0.0f32]);
    }

    /// Overflow reaches infinity under `Nearest` and clamps under the directed
    /// modes. Either way it happens once, in the encode step.
    #[test]
    fn overflow_happens_once_in_the_encode_step_cs_05() {
        let big = f32::MAX;
        let a = [1.0f32, 1.0];
        let b = [big, big];
        assert_eq!(product(1, 2, 1, &a, &b), vec![f32::INFINITY]);

        let mut c = [0.0f32];
        let av = MatView::row_major(&a, 1, 2).unwrap();
        let bv = MatView::row_major(&b, 2, 1).unwrap();
        let cv = MatViewMut::row_major(&mut c, 1, 1).unwrap();
        let mut t = Triple::new(av, bv, cv).unwrap();
        gemm_float(
            &mut t,
            &Linear::OVERWRITE,
            GemmOptions {
                encode: EncodeMode::Saturating,
                ..Default::default()
            },
        );
        assert_eq!(c[0], f32::MAX);
    }

    /// f64 takes the same path at a different instantiation.
    #[test]
    fn f64_takes_the_same_path_cu_05() {
        let a = [1.0f64, 1.0, 1.0];
        let b = [1.0e300f64, 1.0, -1.0e300];
        let mut c = [0.0f64];
        let av = MatView::row_major(&a, 1, 3).unwrap();
        let bv = MatView::row_major(&b, 3, 1).unwrap();
        let cv = MatViewMut::row_major(&mut c, 1, 1).unwrap();
        let mut t = Triple::new(av, bv, cv).unwrap();
        gemm_float(&mut t, &Linear::OVERWRITE, GemmOptions::default());
        assert_eq!(c[0], 1.0f64);
    }
}

#[cfg(test)]
#[allow(clippy::disallowed_types)]
mod window_tests {
    use super::*;
    use std::vec;
    use std::vec::Vec;
    use uor_matmul_core::{MatView, MatViewMut};

    /// The window is exact for adversarial exponents.
    ///
    /// A window holds an exact integer at a known scale, and a term reaches
    /// `2^125` for `f64`. Any fixed flush count would be wrong for some input;
    /// this is the input that finds it. The exponents are chosen so that every
    /// product lands in the same limb with the largest possible shift within
    /// it, which is the worst case for the window's capacity.
    #[test]
    fn the_window_is_exact_at_full_width_cu_04() {
        for k in [1usize, 2, 3, 4, 5, 8, 17, 64, 1000] {
            // Significands with every bit set, so each product is full width.
            let a: Vec<f64> = (0..k)
                .map(|i| f64::from_bits(0x433F_FFFF_FFFF_FFFF - (i as u64 % 3)))
                .collect();
            let b: Vec<f64> = (0..k)
                .map(|i| f64::from_bits(0x433F_FFFF_FFFF_FFFF - (i as u64 % 5)))
                .collect();

            let mut packed = [0.0f64];
            {
                let av = MatView::row_major(&a, 1, k).unwrap();
                let bv = MatView::row_major(&b, k, 1).unwrap();
                let cv = MatViewMut::row_major(&mut packed, 1, 1).unwrap();
                let mut t = Triple::new(av, bv, cv).unwrap();
                let mut pa = vec![uor_matmul_core::PackedCode::default(); k];
                let mut pb = vec![uor_matmul_core::PackedCode::default(); k];
                gemm_float_packed(
                    &mut t,
                    &crate::epilogue::Linear::OVERWRITE,
                    GemmOptions::default(),
                    &mut pa,
                    &mut pb,
                );
            }

            // The streaming traversal, which places every product on its own
            // and never fills a window. If the two disagree, the window lost a
            // carry.
            let mut streamed = [0.0f64];
            {
                let av = MatView::row_major(&a, 1, k).unwrap();
                let bv = MatView::row_major(&b, k, 1).unwrap();
                let cv = MatViewMut::row_major(&mut streamed, 1, 1).unwrap();
                let mut t = Triple::new(av, bv, cv).unwrap();
                gemm_float_packed(
                    &mut t,
                    &crate::epilogue::Linear::OVERWRITE,
                    GemmOptions::default(),
                    &mut [],
                    &mut [],
                );
            }
            assert_eq!(
                packed, streamed,
                "k={k}: the window disagreed with per-product placement"
            );
        }
    }

    /// `CU-04`: what a panel establishes cannot change what it computes.
    ///
    /// The packed traversal settles two facts once per panel --- that every code
    /// is finite, and that this element type's products fit an `i64` --- and then
    /// runs a loop with no per-product test. The streaming traversal establishes
    /// neither and tests everything. They must agree on every input, including
    /// the ones where the facts are false: a non-finite code anywhere, and `f64`
    /// significands wide enough that a product leaves an `i64`.
    /// `CU-04`: scaling both panels to a common base cannot change a byte.
    ///
    /// Three sequences compute this sum --- the 64-bit scaled lane, the 128-bit
    /// scaled lane, and the per-product placement --- and which one runs is
    /// decided by the operands' exponent spans. So the operands below are chosen
    /// to reach each of the three, and every one of them is compared against the
    /// streaming traversal, which always places per product.
    #[test]
    fn scaling_both_panels_cannot_change_a_byte_cu_04() {
        // Spans chosen to land on each lane: none, a significand's worth, a
        // decade, and far too much for any scaling.
        for (label, span_a, span_b) in [
            ("one exponent", 0i32, 0i32),
            ("a few binades", 3, 4),
            ("a decade each", 19, 23),
            ("past every lane", 90, 90),
        ] {
            for &(m, k, n) in &[
                (1usize, 1usize, 1usize),
                (5, 7, 3),
                (16, 64, 8),
                (3, 129, 17),
            ] {
                let mut seed = 0x9E37_79B9_7F4A_7C15u64 ^ (span_a as u64) << 8 ^ (k as u64);
                let mut next = move || {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    (seed >> 33) as i64
                };
                let gen = |next: &mut dyn FnMut() -> i64, len: usize, span: i32| -> Vec<f32> {
                    (0..len)
                        .map(|_| {
                            let v = next();
                            let s = if span == 0 {
                                0
                            } else {
                                (v % span as i64) as i32
                            };
                            (1 + v % 8_388_607) as f32 * 2.0f32.powi(s - span / 2)
                        })
                        .collect()
                };
                let av = gen(&mut next, m * k, span_a);
                let bv = gen(&mut next, k * n, span_b);

                // The reference: the streaming traversal, which places every
                // product individually and knows nothing about spans.
                let mut want = vec![0.0f32; m * n];
                {
                    let a = MatView::row_major(&av, m, k).unwrap();
                    let b = MatView::row_major(&bv, k, n).unwrap();
                    let c = MatViewMut::row_major(&mut want, m, n).unwrap();
                    let mut t = Triple::new(a, b, c).unwrap();
                    gemm_float(
                        &mut t,
                        &crate::epilogue::Linear::OVERWRITE,
                        GemmOptions::default(),
                    );
                }

                // And every panel offer, because the offer decides the block and
                // the block must not decide the answer.
                for offer in [0usize, 1, k, k * n] {
                    let mut got = vec![0.0f32; m * n];
                    let mut qa = vec![
                        PackedCode {
                            mantissa: 0,
                            exp: 0,
                            _pad: 0
                        };
                        k.max(1)
                    ];
                    let mut qb = vec![
                        PackedCode {
                            mantissa: 0,
                            exp: 0,
                            _pad: 0
                        };
                        offer
                    ];
                    let a = MatView::row_major(&av, m, k).unwrap();
                    let b = MatView::row_major(&bv, k, n).unwrap();
                    let c = MatViewMut::row_major(&mut got, m, n).unwrap();
                    let mut t = Triple::new(a, b, c).unwrap();
                    gemm_float_packed(
                        &mut t,
                        &crate::epilogue::Linear::OVERWRITE,
                        GemmOptions::default(),
                        &mut qa,
                        &mut qb,
                    );
                    assert_eq!(
                        got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        want.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        "{label}, {m}x{k}x{n}, offer {offer}"
                    );
                }
            }
        }
    }

    /// The three lanes are all reached by the test above, and this says which.
    ///
    /// A differential test over operands that all take one path is a test of one
    /// path, so the spans are asserted to select what they were chosen to select.
    #[test]
    fn each_scaled_lane_is_reached_cu_04() {
        let span = |exps: &[i32]| {
            let mut s = Span::EMPTY;
            for &e in exps {
                s.see(PackedCode {
                    mantissa: 1,
                    exp: e,
                    _pad: 0,
                });
            }
            s
        };
        // 24-bit significands, so a 64-bit lane has 62 - 48 = 14 bits to spend on
        // the two spans and the depth together.
        let tight = span(&[0]);
        assert_eq!(
            admits::<f32>(64, tight, tight),
            Some(Prescaled {
                base: 0,
                wide: false
            }),
            "no span and a shallow depth is the 64-bit lane"
        );
        let some = span(&[0, 20]);
        assert!(
            matches!(
                admits::<f32>(64, some, some),
                Some(Prescaled { wide: true, .. })
            ),
            "a span past the 64-bit lane is the 128-bit lane"
        );
        let huge = span(&[0, 100]);
        assert_eq!(
            admits::<f32>(64, huge, huge),
            None,
            "a span past every lane is the per-product placement"
        );
    }

    #[test]
    fn what_a_panel_establishes_cannot_change_it_cu_04() {
        fn both_ways_agree_f32(k: usize, a: &[f32], b: &[f32]) {
            let mut packed = vec![0.0f32; 1];
            let mut streamed = vec![0.0f32; 1];
            let mut pa = vec![uor_matmul_core::PackedCode::default(); k];
            let mut pb = vec![uor_matmul_core::PackedCode::default(); k];
            for (out, offer) in [(&mut packed, true), (&mut streamed, false)] {
                let av = MatView::row_major(a, 1, k).unwrap();
                let bv = MatView::row_major(b, k, 1).unwrap();
                let cv = MatViewMut::row_major(out.as_mut_slice(), 1, 1).unwrap();
                let mut t = Triple::new(av, bv, cv).unwrap();
                let (x, y): (&mut [_], &mut [_]) = if offer {
                    (&mut pa, &mut pb)
                } else {
                    (&mut [], &mut [])
                };
                gemm_float_packed(
                    &mut t,
                    &crate::epilogue::Linear::OVERWRITE,
                    GemmOptions::default(),
                    x,
                    y,
                );
            }
            assert_eq!(
                packed[0].to_bits(),
                streamed[0].to_bits(),
                "the panel facts changed the answer"
            );
        }

        let k = 64usize;
        // Finite, full significand width, exponents spread across limbs.
        let a: Vec<f32> = (0..k)
            .map(|i| f32::from_bits(0x4B7F_FFFF - (i as u32 % 7) - ((i as u32 % 11) << 23)))
            .collect();
        let b: Vec<f32> = (0..k)
            .map(|i| f32::from_bits(0x3F7F_FFFF - (i as u32 % 5) - ((i as u32 % 13) << 23)))
            .collect();
        both_ways_agree_f32(k, &a, &b);

        // A non-finite code makes `finite` false, so the general loop runs; it
        // must still agree with the streaming one it is compared against.
        for at in [0usize, 1, 31, 63] {
            for bad in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 0.0] {
                let mut a2 = a.clone();
                a2[at] = bad;
                both_ways_agree_f32(k, &a2, &b);
                let mut b2 = b.clone();
                b2[at] = bad;
                both_ways_agree_f32(k, &a, &b2);
            }
        }

        // `f64`, where two significands make 106 bits and no product fits an
        // `i64`, so `product_fits` is false for the whole type.
        let a: Vec<f64> = (0..k)
            .map(|i| f64::from_bits(0x433F_FFFF_FFFF_FFFF - (i as u64 % 7)))
            .collect();
        let b: Vec<f64> = (0..k)
            .map(|i| f64::from_bits(0x3FEF_FFFF_FFFF_FFFF - (i as u64 % 5)))
            .collect();
        let mut packed = [0.0f64];
        let mut streamed = [0.0f64];
        let mut pa = vec![uor_matmul_core::PackedCode::default(); k];
        let mut pb = vec![uor_matmul_core::PackedCode::default(); k];
        for (out, offer) in [(&mut packed, true), (&mut streamed, false)] {
            let av = MatView::row_major(&a, 1, k).unwrap();
            let bv = MatView::row_major(&b, k, 1).unwrap();
            let cv = MatViewMut::row_major(out.as_mut_slice(), 1, 1).unwrap();
            let mut t = Triple::new(av, bv, cv).unwrap();
            let (x, y): (&mut [_], &mut [_]) = if offer {
                (&mut pa, &mut pb)
            } else {
                (&mut [], &mut [])
            };
            gemm_float_packed(
                &mut t,
                &crate::epilogue::Linear::OVERWRITE,
                GemmOptions::default(),
                x,
                y,
            );
        }
        assert_eq!(packed[0].to_bits(), streamed[0].to_bits());
    }

    /// The window carries exactly across a limb boundary, where a term in one
    /// limb and a term in the next must not be added to each other.
    #[test]
    fn the_window_carries_across_limbs_cu_04() {
        // Exponents 64 apart put consecutive products in different limbs.
        let k = 128usize;
        let a: Vec<f64> = (0..k).map(|i| (2.0f64).powi(i as i32 * 8 - 500)).collect();
        let b: Vec<f64> = vec![1.0; k];

        let mut packed = [0.0f64];
        let mut pa = vec![uor_matmul_core::PackedCode::default(); k];
        let mut pb = vec![uor_matmul_core::PackedCode::default(); k];
        {
            let av = MatView::row_major(&a, 1, k).unwrap();
            let bv = MatView::row_major(&b, k, 1).unwrap();
            let cv = MatViewMut::row_major(&mut packed, 1, 1).unwrap();
            let mut t = Triple::new(av, bv, cv).unwrap();
            gemm_float_packed(
                &mut t,
                &crate::epilogue::Linear::OVERWRITE,
                GemmOptions::default(),
                &mut pa,
                &mut pb,
            );
        }
        let mut streamed = [0.0f64];
        {
            let av = MatView::row_major(&a, 1, k).unwrap();
            let bv = MatView::row_major(&b, k, 1).unwrap();
            let cv = MatViewMut::row_major(&mut streamed, 1, 1).unwrap();
            let mut t = Triple::new(av, bv, cv).unwrap();
            gemm_float_packed(
                &mut t,
                &crate::epilogue::Linear::OVERWRITE,
                GemmOptions::default(),
                &mut [],
                &mut [],
            );
        }
        assert_eq!(packed, streamed);
    }
}
