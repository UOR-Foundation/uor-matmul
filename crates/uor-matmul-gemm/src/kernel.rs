//! The packed, kernel-driven traversal (§7, §8).
//!
//! Same identity, different instructions. This module packs `A` and `B` into
//! the panel format [`KernelSpec`] reads, runs the microkernel over `kc`-deep
//! chunks, and combines the chunks into the wide accumulator. Nothing about the
//! value depends on any of that, which is what `CD-01` asserts byte for byte.
//!
//! # Why a `kc` at all
//!
//! A microkernel accumulates in a 32-bit lane. `kc` is chosen so the lane
//! cannot overflow --- [`narrow_cap_for`] answers that from the alphabet bound
//! and the kernel's declared `lane_cap` --- and the chunks are combined in the
//! accumulator that cannot overflow at all. So the depth of a chunk is a
//! question about a register, and the number of chunks is invisible in the
//! answer (§5.1, R13).

use uor_matmul_core::{AccOf, Accumulator, Alphabet, Bound, Element, Triple};
use uor_matmul_kernels::{select, KernelSpec};

use crate::driver::GemmOptions;
use crate::epilogue::Epilogue;
use crate::scratch::Scratch;

/// `C := epilogue(A * B, C)` for the W8A8 instantiation, through a microkernel.
///
/// Returns `()`, for the same reason [`crate::gemm`] does.
///
/// This is `(i8, i32)` because that is the instantiation the instructions
/// exist for --- `vpdpbusd`, `vdotq_s32`, `i32x4_dot_i16x8` all name it. It is
/// not a privileged path: [`crate::gemm`] computes the same integer for this
/// instantiation and for every other one, and `CD-01` asserts the two agree.
pub fn gemm_w8a8<Bd, Ep>(
    triple: &mut Triple<'_, '_, '_, Alphabet<i8, Bd>, i32>,
    epilogue: &Ep,
    options: GemmOptions,
    scratch: &mut Scratch<'_, i8, Bd>,
) where
    Bd: Bound,
    Ep: Epilogue<i8, i32>,
{
    let spec = select(options.backend);
    let shape = triple.shape();
    if shape.m == 0 || shape.n == 0 {
        return;
    }

    // The panel pair needs `(mr + nr) * kc` elements. With less than one
    // `k`-step of room there is no panel to pack, so the streaming traversal
    // runs instead --- the same identity, walked differently (S13).
    let per_step = spec.mr + spec.nr;
    if scratch.len() < per_step {
        crate::gemm(triple, epilogue, options, scratch);
        return;
    }

    // The deepest chunk this kernel's lane can hold, and that the offer fits.
    let kc = (scratch.len() / per_step)
        .min(lane_depth(&spec, Bd::VALUE))
        .max(1);

    let reads_c = epilogue.reads_c();
    let (a, b, c) = triple.parts();

    let mut tile = [0i32; MAX_TILE];
    let mr = spec.mr;
    let nr = spec.nr;
    debug_assert!(
        mr * nr <= MAX_TILE,
        "the tile buffer covers every shipped kernel"
    );

    // Pack `B` once per column block, not once per output block.
    //
    // The panels are the only copies this driver makes, and which loop they sit
    // in decides how many times each byte is copied. With the row block
    // outermost, `B` is repacked for every row block --- `m/mr` times over ---
    // which costs `m*n*k/mr` element copies against `m*n*k` of arithmetic, so
    // about a sixth of the work is spent copying `B` alone. Putting the column
    // block outermost and hoisting the `B` pack out of the row loop replaces
    // that with `n*k` copies once, and leaves `A` repacked `m*n*k/nr` times.
    // `nr` is the wider of the two, so this is the cheaper way round.
    //
    // Hoisting `B` requires the whole depth in one chunk, because a chunked
    // accumulation has to revisit each output block per chunk. When the offer
    // is too small for that, the chunked traversal below runs instead --- the
    // same identity, walked differently, and `CD-01` and `CD-10` assert the
    // bytes are the same either way (S13).
    let hoist = shape.k <= kc && scratch.len() >= per_step * shape.k;

    if hoist {
        let depth = shape.k;
        let buf = scratch.take(per_step * depth);
        let (pa_buf, pb_buf) = buf.split_at_mut(mr * depth);

        let mut j0 = 0;
        while j0 < shape.n {
            for p in 0..depth {
                for j in 0..nr {
                    pb_buf[p * nr + j] = if j0 + j < shape.n {
                        *b.at(p, j0 + j)
                    } else {
                        Alphabet::ZERO
                    };
                }
            }

            let mut i0 = 0;
            while i0 < shape.m {
                for p in 0..depth {
                    for i in 0..mr {
                        pa_buf[p * mr + i] = if i0 + i < shape.m {
                            *a.at(i0 + i, p)
                        } else {
                            Alphabet::ZERO
                        };
                    }
                }

                let pa: &[i8] = bytemuck::TransparentWrapper::peel_slice(&*pa_buf);
                let pb: &[i8] = bytemuck::TransparentWrapper::peel_slice(&*pb_buf);
                spec.mac_tile(depth, pa, pb, &mut tile[..mr * nr]);

                // One chunk covers the whole depth, so the lane already holds
                // the complete accumulation and there is nothing to combine.
                for i in 0..mr.min(shape.m - i0) {
                    for j in 0..nr.min(shape.n - j0) {
                        let acc = i8::combine_narrow(
                            <AccOf<i8> as Accumulator>::ZERO,
                            tile[i * nr + j] as i64,
                        );
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
            // One output block, accumulated exactly across every chunk.
            let mut wide = [<AccOf<i8> as Accumulator>::ZERO; MAX_TILE];

            let mut p0 = 0;
            while p0 < shape.k {
                let depth = kc.min(shape.k - p0);
                {
                    let buf = scratch.take(per_step * depth);
                    let (pa, pb) = buf.split_at_mut(mr * depth);
                    // Pack `A` k-major, zero-padding rows past `m`. Zero
                    // padding is exact, so an unaligned shape takes this path
                    // and not a different one (S8).
                    for p in 0..depth {
                        for i in 0..mr {
                            pa[p * mr + i] = if i0 + i < shape.m {
                                *a.at(i0 + i, p0 + p)
                            } else {
                                Alphabet::ZERO
                            };
                        }
                        for j in 0..nr {
                            pb[p * nr + j] = if j0 + j < shape.n {
                                *b.at(p0 + p, j0 + j)
                            } else {
                                Alphabet::ZERO
                            };
                        }
                    }
                }

                let buf = scratch.take(per_step * depth);
                let raw: &[i8] = bytemuck::TransparentWrapper::peel_slice(&*buf);
                let (pa, pb) = raw.split_at(mr * depth);
                spec.mac_tile(depth, pa, pb, &mut tile[..mr * nr]);

                // Combine the chunk into the accumulator that cannot overflow.
                for (w, t) in wide[..mr * nr].iter_mut().zip(&tile[..mr * nr]) {
                    *w = i8::combine_narrow(*w, *t as i64);
                }
                p0 += depth;
            }

            for i in 0..mr.min(shape.m - i0) {
                for j in 0..nr.min(shape.n - j0) {
                    let acc = wide[i * nr + j];
                    let prior = if reads_c {
                        Some(*c.at(i0 + i, j0 + j))
                    } else {
                        None
                    };
                    *c.at_mut(i0 + i, j0 + j) = epilogue.finish(acc, prior, options.encode);
                }
            }
            j0 += nr;
        }
        i0 += mr;
    }
}

/// The largest tile any shipped kernel produces, so the driver needs no heap.
///
/// Derived from the widest `mr * nr` in `uor-matmul-kernels`; a kernel larger
/// than this trips the `debug_assert` above rather than corrupting anything,
/// and adding one means raising this constant in the same commit.
const MAX_TILE: usize = 8 * 32;

/// The deepest chunk this kernel's lane can hold for an alphabet bounded by
/// `bound`.
///
/// A question about a register, not a limit on `k`: a deeper accumulation is
/// simply split into more chunks, and the chunks combine exactly.
fn lane_depth(spec: &KernelSpec, bound: u128) -> usize {
    let per_step = bound.saturating_mul(bound);
    if per_step == 0 {
        return usize::MAX;
    }
    usize::try_from(spec.lane_cap / per_step)
        .unwrap_or(usize::MAX)
        .max(1)
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
    use uor_matmul_core::{as_alphabet_full, Backend, EncodeMode, Full, MatView, MatViewMut};

    fn run(
        m: usize,
        k: usize,
        n: usize,
        a: &[i8],
        b: &[i8],
        backend: Backend,
        scratch_len: usize,
    ) -> Vec<i32> {
        let mut c = vec![0i32; m * n];
        let mut buf = vec![Alphabet::<i8, Full<i8>>::ZERO; scratch_len];
        {
            let av = MatView::row_major(as_alphabet_full(a), m, k).unwrap();
            let bv = MatView::row_major(as_alphabet_full(b), k, n).unwrap();
            let cv = MatViewMut::row_major(&mut c, m, n).unwrap();
            let mut t = Triple::new(av, bv, cv).unwrap();
            gemm_w8a8(
                &mut t,
                &Linear::OVERWRITE,
                GemmOptions {
                    backend,
                    encode: EncodeMode::Wrapping,
                    ..Default::default()
                },
                &mut Scratch::new(&mut buf),
            );
        }
        c
    }

    fn fill(len: usize, salt: u64) -> Vec<i8> {
        let mut s = 0x9E37_79B9_7F4A_7C15u64 ^ salt;
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 33) as i8
            })
            .collect()
    }

    /// `CD-01`: the backend a caller names never changes the output bytes, and
    /// neither does the scratch amount that decides the chunk depth.
    #[test]
    fn backend_and_chunking_are_invisible_cd_01() {
        for (m, k, n) in [
            (1, 1, 1),
            (3, 7, 11),
            (13, 17, 19),
            (8, 256, 32),
            (5, 97, 7),
        ] {
            let a = fill(m * k, (m * k) as u64);
            let b = fill(k * n, (k * n) as u64 ^ 0x33);

            // The generic driver is the reference: it consults no kernel.
            let mut expected = vec![0i32; m * n];
            {
                let av = MatView::row_major(as_alphabet_full(&a), m, k).unwrap();
                let bv = MatView::row_major(as_alphabet_full(&b), k, n).unwrap();
                let cv = MatViewMut::row_major(&mut expected, m, n).unwrap();
                let mut t = Triple::new(av, bv, cv).unwrap();
                crate::gemm(
                    &mut t,
                    &Linear::OVERWRITE,
                    GemmOptions {
                        encode: EncodeMode::Wrapping,
                        ..Default::default()
                    },
                    &mut Scratch::none(),
                );
            }

            for backend in Backend::ALL {
                for scratch_len in [0usize, 1, 40, 41, 4096, 100_000] {
                    assert_eq!(
                        run(m, k, n, &a, &b, backend, scratch_len),
                        expected,
                        "{}x{}x{} on {} with {} scratch",
                        m,
                        k,
                        n,
                        backend.as_str(),
                        scratch_len
                    );
                }
            }
        }
    }

    /// `CD-01`, `CT-01`: a depth past every lane's reach still gives the exact
    /// value, because the chunking is a register question and the chunks
    /// combine in an accumulator that cannot overflow.
    #[test]
    fn depth_past_every_lane_is_exact_ct_01() {
        let k = 300_000;
        let a = vec![i8::MIN; k];
        let b = vec![i8::MIN; k];
        let expected = ((k as i128) * 128 * 128) as i32; // wraps, and must
        for backend in Backend::ALL {
            assert_eq!(run(1, k, 1, &a, &b, backend, 8192), vec![expected]);
        }
    }
}
