//! The packed, kernel-driven traversal (§7, §8).
//!
//! Same identity, different instructions. This module packs `A` and `B` into
//! the panel format a [`KernelSpec`] reads, runs the microkernel over chunks
//! the lane can hold, and folds the chunks into the accumulator that cannot
//! overflow. Nothing about the value depends on any of it, which is what
//! `CD-01` asserts byte for byte.
//!
//! # Every instantiation, not one
//!
//! The dispatch is a trait, [`Kernelized`], implemented for every element type
//! that has instructions. An element type without one is not a hole: its
//! `Lane` is its own exact accumulator and the reference kernel runs, which is
//! the same identity at the same exactness. What would be a hole --- and was,
//! before this --- is an instantiation whose *fast* factorization exists in the
//! ISA and is not reached, because then a measurement of that instantiation is
//! a measurement of the wrong program.

use uor_matmul_core::{
    AccOf, Accumulator, Alphabet, Backend, Bound, Element, EncodeFrom, EncodeMode, IntegerElement,
    Triple,
};
use uor_matmul_kernels::{
    available_i16, available_i16_modular, available_i32_exact, available_i32_modular,
    available_i64_exact, available_i64_modular, available_i8, choose, Factorization, KernelSpec,
    MAX_TILE_LANES,
};

use crate::driver::GemmOptions;
use crate::epilogue::Epilogue;
use crate::scratch::Scratch;

/// An element type with microkernels, and how to fold their lanes.
///
/// The two factorizations are chosen by *declaration*, never by inspecting the
/// data: the alphabet bound decides how deep an exact lane may go, and the
/// encode mode decides whether the modular lane is admissible at all.
pub trait Kernelized: IntegerElement {
    /// The lane the exact kernels accumulate in.
    type Exact: Copy + Default + 'static;
    /// The lane the modular kernels accumulate in, when one exists.
    type Modular: Copy + Default + 'static;

    /// The exact kernel for this backend. Always present.
    fn exact_spec(backend: Backend) -> KernelSpec<Self, Self::Exact>;

    /// The modular kernel for this backend, if the output width admits it.
    ///
    /// `out_bits` is the width of the type the caller is encoding into. The
    /// modular lane is legitimate exactly when the caller asked to *wrap* into
    /// a type no wider than the lane: reduction modulo `2^w` is a ring
    /// homomorphism, so the lane's own wrap is the encode, and nothing is lost
    /// that the caller did not ask to lose (§3.4).
    fn modular_spec(backend: Backend, out_bits: u32) -> Option<KernelSpec<Self, Self::Modular>>;

    /// Fold an exact lane into the accumulator that cannot overflow.
    fn fold_exact(acc: &mut Self::Acc, lane: Self::Exact);

    /// Combine two modular lanes. The ring's own addition.
    fn add_modular(acc: Self::Modular, lane: Self::Modular) -> Self::Modular;

    /// The modular lane, as the value the encode step would have produced.
    fn modular_as_acc(lane: Self::Modular) -> Self::Acc;
}

impl Kernelized for i8 {
    type Exact = i32;
    type Modular = i32;

    fn exact_spec(backend: Backend) -> KernelSpec<Self, i32> {
        choose(available_i8(), backend).expect("the portable kernel is always present")
    }

    fn modular_spec(backend: Backend, out_bits: u32) -> Option<KernelSpec<Self, i32>> {
        if out_bits > 32 {
            return None;
        }
        // The `i8` kernels already accumulate in a 32-bit lane. Read as exact,
        // that lane fills at `k = 133144` and the driver chunks; read as
        // `Z/2^32` --- which is what the caller asked for --- its wrap *is* the
        // encode, so the same kernel carries any depth in one chunk. Same
        // instructions, same bytes, one fewer fold: that is the homomorphism
        // being cashed in rather than a second kernel being written.
        let spec = choose(available_i8(), backend)?;
        Some(KernelSpec {
            factorization: Factorization::Modular,
            ..spec
        })
    }

    fn fold_exact(acc: &mut Self::Acc, lane: i32) {
        *acc += i128::from(lane);
    }

    fn add_modular(acc: i32, lane: i32) -> i32 {
        acc.wrapping_add(lane)
    }

    fn modular_as_acc(lane: i32) -> Self::Acc {
        i128::from(lane)
    }
}

impl Kernelized for i16 {
    type Exact = i64;
    type Modular = i32;

    fn exact_spec(backend: Backend) -> KernelSpec<Self, i64> {
        choose(available_i16(), backend).expect("the portable kernel is always present")
    }

    fn modular_spec(backend: Backend, out_bits: u32) -> Option<KernelSpec<Self, i32>> {
        if out_bits > 32 {
            return None;
        }
        // Not a shortcut for the exact lane's benefit --- that lane already
        // reaches every addressable `k` for `i16`. It is twice the columns per
        // instruction, because in `Z/2^32` there is nothing to widen to.
        choose(available_i16_modular(), backend)
    }

    fn fold_exact(acc: &mut Self::Acc, lane: i64) {
        *acc += i128::from(lane);
    }

    fn add_modular(acc: i32, lane: i32) -> i32 {
        acc.wrapping_add(lane)
    }

    fn modular_as_acc(lane: i32) -> Self::Acc {
        i128::from(lane)
    }
}

impl Kernelized for i32 {
    type Exact = i64;
    type Modular = i32;

    fn exact_spec(backend: Backend) -> KernelSpec<Self, i64> {
        choose(available_i32_exact(), backend).expect("the portable kernel is always present")
    }

    fn modular_spec(backend: Backend, out_bits: u32) -> Option<KernelSpec<Self, i32>> {
        if out_bits > 32 {
            return None;
        }
        choose(available_i32_modular(), backend)
    }

    fn fold_exact(acc: &mut Self::Acc, lane: i64) {
        *acc += i128::from(lane);
    }

    fn add_modular(acc: i32, lane: i32) -> i32 {
        acc.wrapping_add(lane)
    }

    fn modular_as_acc(lane: i32) -> Self::Acc {
        i128::from(lane)
    }
}

impl Kernelized for i64 {
    type Exact = i128;
    type Modular = i64;

    fn exact_spec(backend: Backend) -> KernelSpec<Self, i128> {
        // An `i64 x i64` product needs 128 bits, so the lane is an `i128`. No
        // SIMD integer multiply reaches that width on any supported target, so
        // this is not a placeholder --- it is the whole of what the hardware
        // offers, and the packing still buys it the locality every other family
        // gets.
        choose(available_i64_exact(), backend).expect("the portable kernel is always present")
    }

    fn modular_spec(backend: Backend, out_bits: u32) -> Option<KernelSpec<Self, i64>> {
        if out_bits > 64 {
            return None;
        }
        choose(available_i64_modular(), backend)
    }

    fn fold_exact(acc: &mut Self::Acc, lane: i128) {
        acc.add_i128_in_place(lane);
    }

    fn add_modular(acc: i64, lane: i64) -> i64 {
        acc.wrapping_add(lane)
    }

    fn modular_as_acc(lane: i64) -> Self::Acc {
        <Self::Acc as Accumulator>::ZERO.add_i128(i128::from(lane))
    }
}

/// The tile buffer, sized by the kernels rather than by a number chosen here.
///
/// Every kernel carries a `const` assertion that its own tile fits
/// [`MAX_TILE_LANES`], so a kernel too large for this buffer fails the *build*.
/// That is what keeps it a derivation rather than a ceiling: no input can reach
/// it, and no kernel can quietly exceed it (R8).
const MAX_TILE: usize = MAX_TILE_LANES;

/// `C := epilogue(A * B, C)` through a microkernel, for any element type that
/// has one.
///
/// Returns `()`, for the same reason [`crate::gemm`] does.
pub fn gemm_packed<E, Bd, O, Ep>(
    triple: &mut Triple<'_, '_, '_, Alphabet<E, Bd>, O>,
    epilogue: &Ep,
    options: GemmOptions,
    scratch: &mut Scratch<'_, E, Bd>,
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

    // The modular lane is admissible exactly when the caller asked to wrap into
    // a type no wider than it. That is a question about two declarations --- the
    // encode mode and the output type --- and about nothing else.
    let modular = matches!(options.encode, EncodeMode::Wrapping)
        .then(|| E::modular_spec(options.backend, O::BITS))
        .flatten();

    match modular {
        Some(spec) => run::<E, Bd, O, Ep, E::Modular>(
            triple,
            epilogue,
            options,
            scratch,
            spec,
            E::add_modular,
            |acc, lane| *acc = E::modular_as_acc(lane),
            usize::MAX,
        ),
        None => {
            let spec = E::exact_spec(options.backend);
            let depth = spec.lane_depth(Bd::VALUE);
            run::<E, Bd, O, Ep, E::Exact>(
                triple,
                epilogue,
                options,
                scratch,
                spec,
                |_, lane| lane,
                E::fold_exact,
                depth,
            )
        }
    }
}

/// The one traversal, shared by both factorizations.
///
/// `combine_lane` folds two lanes of the same chunk --- the ring's addition for
/// a modular lane, and unreachable for an exact one, which never sees two
/// chunks in the same lane. `fold` moves a finished lane into the accumulator.
#[allow(clippy::too_many_arguments)]
fn run<E, Bd, O, Ep, L>(
    triple: &mut Triple<'_, '_, '_, Alphabet<E, Bd>, O>,
    epilogue: &Ep,
    options: GemmOptions,
    scratch: &mut Scratch<'_, E, Bd>,
    spec: KernelSpec<E, L>,
    combine_lane: impl Fn(L, L) -> L,
    fold: impl Fn(&mut AccOf<E>, L),
    lane_depth: usize,
) where
    E: Kernelized,
    Bd: Bound,
    O: Element + EncodeFrom<AccOf<E>>,
    Ep: Epilogue<E, O>,
    L: Copy + Default + 'static,
{
    let shape = triple.shape();
    let (mr, nr) = (spec.mr, spec.nr);
    let per_step = mr + nr;
    debug_assert!(
        mr * nr <= MAX_TILE,
        "the tile buffer covers every shipped kernel"
    );

    // With less than one `k`-step of room there is no panel to pack, so the
    // streaming traversal runs instead --- the same identity, walked
    // differently (S13).
    if scratch.len() < per_step {
        crate::gemm(triple, epilogue, options, scratch);
        return;
    }

    let kc = (scratch.len() / per_step).min(lane_depth).max(1);
    let reads_c = epilogue.reads_c();

    // Pack `B` once per column block, not once per output block.
    //
    // The panels are the only copies this driver makes, and which loop they
    // sit in decides how many times each byte is copied. With the row block
    // outermost, `B` is repacked `m/mr` times over: `m*n*k/mr` element copies
    // against `m*n*k` of arithmetic. Putting the column block outermost and
    // hoisting the `B` pack out of the row loop replaces that with `n*k`
    // copies, and leaves `A` repacked `m*n*k/nr` times --- the cheaper way
    // round, because `nr > mr`.
    //
    // Hoisting needs the whole depth in one chunk, because a chunked
    // accumulation revisits each output block per chunk. When the offer is too
    // small the chunked traversal below runs instead, and `CD-01` and `CD-10`
    // assert the bytes are the same either way.
    let hoist = shape.k <= kc && scratch.len() >= per_step * shape.k;
    let mut tile = [L::default(); MAX_TILE];
    let (a, b, c) = triple.parts();

    if hoist {
        let depth = shape.k;
        let buf = scratch.take(per_step * depth);
        let (pa_buf, pb_buf) = buf.split_at_mut(mr * depth);

        let mut j0 = 0;
        while j0 < shape.n {
            pack_columns(pb_buf, nr, depth, b, 0, j0, shape.n);

            let mut i0 = 0;
            while i0 < shape.m {
                pack_rows(pa_buf, mr, depth, a, i0, 0, shape.m);

                let pa: &[E] = bytemuck::TransparentWrapper::peel_slice(&*pa_buf);
                let pb: &[E] = bytemuck::TransparentWrapper::peel_slice(&*pb_buf);
                spec.mac_tile(depth, pa, pb, &mut tile[..mr * nr]);

                for i in 0..mr.min(shape.m - i0) {
                    for j in 0..nr.min(shape.n - j0) {
                        let mut acc = <AccOf<E> as Accumulator>::ZERO;
                        fold(&mut acc, tile[i * nr + j]);
                        let prior = if reads_c {
                            Some(*c.at(i0 + i, j0 + j))
                        } else {
                            None
                        };
                        *c.at_mut(i0 + i, j0 + j) = epilogue.finish(acc, prior, options.encode);
                    }
                }
                i0 += mr;
            }
            j0 += nr;
        }
        return;
    }

    let mut i0 = 0;
    while i0 < shape.m {
        let mut j0 = 0;
        while j0 < shape.n {
            let mut wide = [<AccOf<E> as Accumulator>::ZERO; MAX_TILE];
            let mut carried = [L::default(); MAX_TILE];

            let mut p0 = 0;
            while p0 < shape.k {
                let depth = kc.min(shape.k - p0);
                {
                    let buf = scratch.take(per_step * depth);
                    let (pa_buf, pb_buf) = buf.split_at_mut(mr * depth);
                    pack_rows(pa_buf, mr, depth, a, i0, p0, shape.m);
                    pack_columns(pb_buf, nr, depth, b, p0, j0, shape.n);
                }

                let buf = scratch.take(per_step * depth);
                let raw: &[E] = bytemuck::TransparentWrapper::peel_slice(&*buf);
                let (pa, pb) = raw.split_at(mr * depth);
                spec.mac_tile(depth, pa, pb, &mut tile[..mr * nr]);

                if matches!(spec.factorization, Factorization::Modular) {
                    // The ring's own addition: chunks combine in the quotient.
                    for (acc, lane) in carried[..mr * nr].iter_mut().zip(&tile[..mr * nr]) {
                        *acc = combine_lane(*acc, *lane);
                    }
                } else {
                    // Each chunk's lane is exact; the accumulator absorbs them.
                    for (w, lane) in wide[..mr * nr].iter_mut().zip(&tile[..mr * nr]) {
                        fold(w, *lane);
                    }
                }
                p0 += depth;
            }

            if matches!(spec.factorization, Factorization::Modular) {
                for (w, lane) in wide[..mr * nr].iter_mut().zip(&carried[..mr * nr]) {
                    fold(w, *lane);
                }
            }

            for i in 0..mr.min(shape.m - i0) {
                for j in 0..nr.min(shape.n - j0) {
                    let prior = if reads_c {
                        Some(*c.at(i0 + i, j0 + j))
                    } else {
                        None
                    };
                    *c.at_mut(i0 + i, j0 + j) =
                        epilogue.finish(wide[i * nr + j], prior, options.encode);
                }
            }
            j0 += nr;
        }
        i0 += mr;
    }
}

/// Pack `mr` rows of `A`, `k`-major.
///
/// Walks each row once rather than indexing per element, and splits the full
/// tile from the edge so the common case carries no bounds branch. Rows past
/// the matrix take the alphabet's zero; zero padding is exact, so an unaligned
/// shape takes this path and not a different one (S8).
fn pack_rows<E: IntegerElement, Bd: Bound>(
    out: &mut [Alphabet<E, Bd>],
    lanes: usize,
    depth: usize,
    a: &uor_matmul_core::MatView<'_, Alphabet<E, Bd>>,
    row0: usize,
    col0: usize,
    rows: usize,
) {
    let full = lanes.min(rows.saturating_sub(row0));
    for lane in 0..full {
        for (p, v) in a.row_walk(row0 + lane, col0, depth).enumerate() {
            out[p * lanes + lane] = *v;
        }
    }
    for lane in full..lanes {
        for p in 0..depth {
            out[p * lanes + lane] = Alphabet::ZERO;
        }
    }
}

/// Pack `nr` columns of `B`, `k`-major. See [`pack_rows`].
fn pack_columns<E: IntegerElement, Bd: Bound>(
    out: &mut [Alphabet<E, Bd>],
    lanes: usize,
    depth: usize,
    b: &uor_matmul_core::MatView<'_, Alphabet<E, Bd>>,
    row0: usize,
    col0: usize,
    cols: usize,
) {
    let full = lanes.min(cols.saturating_sub(col0));
    for lane in 0..full {
        for (p, v) in b.column_walk(row0, col0 + lane, depth).enumerate() {
            out[p * lanes + lane] = *v;
        }
    }
    for lane in full..lanes {
        for p in 0..depth {
            out[p * lanes + lane] = Alphabet::ZERO;
        }
    }
}

#[cfg(test)]
// R7 governs the library, not its tests: these build operands on the heap so
// that awkward shapes can be generated. `CA-01` witnesses the library's own
// zero-allocation claim with a counting allocator.
#[allow(clippy::disallowed_types)]
mod tests {
    use super::*;
    use crate::epilogue::Linear;
    use std::vec;
    use std::vec::Vec;
    use uor_matmul_core::{as_alphabet_full, Full, MatView, MatViewMut};

    /// Run one instantiation through the packed driver.
    #[allow(clippy::too_many_arguments)]
    fn packed<E, O>(
        m: usize,
        k: usize,
        n: usize,
        a: &[E],
        b: &[E],
        backend: Backend,
        mode: EncodeMode,
        scratch_len: usize,
    ) -> Vec<O>
    where
        E: Kernelized,
        O: Element + EncodeFrom<AccOf<E>> + Default + Clone,
        Linear: Epilogue<E, O>,
    {
        let mut c = vec![O::default(); m * n];
        let mut buf = vec![Alphabet::<E, Full<E>>::ZERO; scratch_len];
        {
            let av = MatView::row_major(as_alphabet_full(a), m, k).unwrap();
            let bv = MatView::row_major(as_alphabet_full(b), k, n).unwrap();
            let cv = MatViewMut::row_major(&mut c, m, n).unwrap();
            let mut t = Triple::new(av, bv, cv).unwrap();
            gemm_packed(
                &mut t,
                &Linear::OVERWRITE,
                GemmOptions {
                    backend,
                    encode: mode,
                    ..Default::default()
                },
                &mut Scratch::new(&mut buf),
            );
        }
        c
    }

    /// The generic driver's answer, which consults no kernel at all.
    fn generic<E, O>(m: usize, k: usize, n: usize, a: &[E], b: &[E], mode: EncodeMode) -> Vec<O>
    where
        E: IntegerElement,
        O: Element + EncodeFrom<AccOf<E>> + Default + Clone,
        Linear: Epilogue<E, O>,
    {
        let mut c = vec![O::default(); m * n];
        {
            let av = MatView::row_major(as_alphabet_full(a), m, k).unwrap();
            let bv = MatView::row_major(as_alphabet_full(b), k, n).unwrap();
            let cv = MatViewMut::row_major(&mut c, m, n).unwrap();
            let mut t = Triple::new(av, bv, cv).unwrap();
            crate::gemm(
                &mut t,
                &Linear::OVERWRITE,
                GemmOptions {
                    encode: mode,
                    ..Default::default()
                },
                &mut Scratch::none(),
            );
        }
        c
    }

    fn fill<T, F: Fn(i64) -> T>(len: usize, salt: u64, map: F) -> Vec<T> {
        let mut s = 0x9E37_79B9_7F4A_7C15u64 ^ salt;
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                map((s >> 33) as i64)
            })
            .collect()
    }

    const SHAPES: &[(usize, usize, usize)] = &[
        (1, 1, 1),
        (3, 7, 11),
        (13, 17, 19),
        (8, 256, 32),
        (5, 97, 7),
        (33, 5, 41),
    ];

    /// `CD-01`: every backend, every scratch amount, and every factorization
    /// give the generic driver's bytes --- at every instantiation that has a
    /// kernel, not just at the one the instructions were written for.
    #[test]
    fn every_instantiation_matches_the_generic_driver_cd_01() {
        for &(m, k, n) in SHAPES {
            let a8 = fill(m * k, 1, |v| v as i8);
            let b8 = fill(k * n, 2, |v| v as i8);
            let a16 = fill(m * k, 3, |v| v as i16);
            let b16 = fill(k * n, 4, |v| v as i16);
            let a32 = fill(m * k, 5, |v| v as i32);
            let b32 = fill(k * n, 6, |v| v as i32);
            let a64 = fill(m * k, 7, |v| v.wrapping_mul(0x1000_0000_01B3));
            let b64 = fill(k * n, 8, |v| v.wrapping_mul(0x9E37_79B9));

            let want8: Vec<i32> = generic(m, k, n, &a8, &b8, EncodeMode::Wrapping);
            let want16: Vec<i32> = generic(m, k, n, &a16, &b16, EncodeMode::Wrapping);
            let want32: Vec<i32> = generic(m, k, n, &a32, &b32, EncodeMode::Wrapping);
            let want64: Vec<i64> = generic(m, k, n, &a64, &b64, EncodeMode::Wrapping);

            for backend in Backend::ALL {
                for scratch_len in [0usize, 1, 64, 4096, 1 << 18] {
                    assert_eq!(
                        packed::<i8, i32>(
                            m,
                            k,
                            n,
                            &a8,
                            &b8,
                            backend,
                            EncodeMode::Wrapping,
                            scratch_len
                        ),
                        want8,
                        "i8 {m}x{k}x{n} on {} with {scratch_len}",
                        backend.as_str()
                    );
                    assert_eq!(
                        packed::<i16, i32>(
                            m,
                            k,
                            n,
                            &a16,
                            &b16,
                            backend,
                            EncodeMode::Wrapping,
                            scratch_len
                        ),
                        want16,
                        "i16 {m}x{k}x{n} on {}",
                        backend.as_str()
                    );
                    assert_eq!(
                        packed::<i32, i32>(
                            m,
                            k,
                            n,
                            &a32,
                            &b32,
                            backend,
                            EncodeMode::Wrapping,
                            scratch_len
                        ),
                        want32,
                        "i32 {m}x{k}x{n} on {}",
                        backend.as_str()
                    );
                    assert_eq!(
                        packed::<i64, i64>(
                            m,
                            k,
                            n,
                            &a64,
                            &b64,
                            backend,
                            EncodeMode::Wrapping,
                            scratch_len
                        ),
                        want64,
                        "i64 {m}x{k}x{n} on {}",
                        backend.as_str()
                    );
                }
            }
        }
    }

    /// `CD-05`: the encode mode selects the factorization, and both give the
    /// value that mode asks for.
    ///
    /// Under `Saturating` the exact lane runs and the answer is the clamped
    /// mathematical value; under `Wrapping` the modular lane runs and the
    /// answer is the same bytes the exact accumulation would have wrapped to.
    /// One identity, two quotients, and the caller names which.
    #[test]
    fn the_encode_mode_selects_the_factorization_cd_05() {
        let (m, k, n) = (4usize, 500usize, 4usize);
        let a = vec![i8::MAX; m * k];
        let b = vec![i8::MAX; k * n];

        let wrapping: Vec<i32> = packed(
            m,
            k,
            n,
            &a,
            &b,
            Backend::Auto,
            EncodeMode::Wrapping,
            1 << 16,
        );
        assert_eq!(
            wrapping,
            generic::<i8, i32>(m, k, n, &a, &b, EncodeMode::Wrapping)
        );

        let saturating: Vec<i32> = packed(
            m,
            k,
            n,
            &a,
            &b,
            Backend::Auto,
            EncodeMode::Saturating,
            1 << 16,
        );
        assert_eq!(
            saturating,
            generic::<i8, i32>(m, k, n, &a, &b, EncodeMode::Saturating)
        );

        // The exact value is 500 * 127 * 127, which fits i32, so the two agree
        // here. Deeper, they must not --- and that is the next assertion.
        assert_eq!(wrapping[0], 500 * 127 * 127);

        let deep = 200_000usize;
        let a = vec![i8::MAX; deep];
        let b = vec![i8::MAX; deep];
        let w: Vec<i32> = packed(
            1,
            deep,
            1,
            &a,
            &b,
            Backend::Auto,
            EncodeMode::Wrapping,
            1 << 16,
        );
        let s: Vec<i32> = packed(
            1,
            deep,
            1,
            &a,
            &b,
            Backend::Auto,
            EncodeMode::Saturating,
            1 << 16,
        );
        assert_eq!(
            w[0],
            ((deep as i64) * 127 * 127) as i32,
            "wrapping truncates"
        );
        assert_eq!(s[0], i32::MAX, "saturating clamps");
        assert_ne!(w, s, "past i32 the two encode modes must disagree");
    }

    /// `CT-01`: a depth past every lane's reach is still exact, because the
    /// chunking is a register question and the chunks combine exactly.
    #[test]
    fn depth_past_every_lane_is_exact_ct_01() {
        let k = 300_000;
        let a = vec![i8::MIN; k];
        let b = vec![i8::MIN; k];
        let expected = ((k as i128) * 128 * 128) as i32;
        for backend in Backend::ALL {
            let got: Vec<i32> = packed(1, k, 1, &a, &b, backend, EncodeMode::Wrapping, 1 << 16);
            assert_eq!(got, vec![expected], "{}", backend.as_str());
        }
    }
}
