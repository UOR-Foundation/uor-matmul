//! The shape and data corpus every differential test walks.
//!
//! Shapes are chosen to be awkward on purpose: zero, one, primes, powers of two
//! plus one, and depths on both sides of every narrow-register threshold. A
//! corpus of round numbers would agree with anything.
//!
//! # Why the tropical corpus is drawn from three values and not from the alphabet
//!
//! The ring corpus wants a *wide* draw: a fill that never leaves `-8..8` would
//! agree with a kernel that silently narrowed its accumulator. The selection
//! corpus wants the opposite, and for a reason that is specific to what it
//! gates. A `(max, +)` reduction's witness is only observable where two terms
//! **tie** --- everywhere else every tie-break convention, every partition and
//! every mechanism agrees, so a sweep with no ties in it exercises the witness
//! machinery not at all while producing a green row that reads like evidence.
//! [`Corpus::tropical`] therefore draws from three values, where a reduction of
//! any depth past sixteen is tied in more than half its cells, and
//! `the_narrow_draw_forces_ties_and_the_wide_one_does_not` below is the
//! two-sided measurement of that: the same shapes, the same seeds, the ring
//! corpus's own wide fill through the same masks, and the tie count falls by
//! nearly two orders of magnitude.
//!
//! Every tropical fill carries **masked lanes** --- positions at the semiring
//! zero --- because A-6 is two claims and the mask is half of it. They are
//! placed structurally rather than drawn, at coprime periods in the two
//! operands, so that the masks never align into the same column of every cell
//! and so that a case either provably has one or provably is too small to.

use uor_matmul_core::Trop;

/// One case: a shape and a deterministic fill.
#[derive(Clone, Copy, Debug)]
pub struct Case {
    /// Rows of A.
    pub m: usize,
    /// Depth.
    pub k: usize,
    /// Columns of B.
    pub n: usize,
    /// The seed the fill is derived from, recorded so a failure is reproducible.
    pub seed: u64,
}

impl Case {
    /// Deterministic `i8` fill.
    ///
    /// A small xorshift rather than a dependency: the sample is recorded by its
    /// seed, and a generator whose behaviour could change with a minor version
    /// bump would make that record worthless.
    pub fn fill_i8(&self, len: usize, salt: u64) -> Vec<i8> {
        let mut s = self.seed ^ salt.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 33) as i8
            })
            .collect()
    }

    /// The same fill widened, for an oracle that only speaks `i32`.
    pub fn widen(xs: &[i8]) -> Vec<i32> {
        xs.iter().map(|&x| x as i32).collect()
    }

    /// The tropical draw: `None` is the semiring zero, `Some(v)` a finite
    /// element with `v` in [`TROPICAL_RANGE`].
    ///
    /// The same recorded xorshift the ring fill uses, so one generator covers
    /// both halves of the census and the seed is still the whole record of the
    /// sample. What differs is the *width of the draw* and the mask, and both
    /// differences are the point rather than a convenience --- see this
    /// module's header.
    ///
    /// Re-spelled in Python in `oracles/tropical/generate.py`, which is how the
    /// committed artifact and this fill are the same numbers (`CX-11`).
    pub fn draw_tropical(&self, len: usize, salt: u64, mask: Mask) -> Vec<Option<i64>> {
        let mut s = self.seed ^ salt.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        (0..len)
            .map(|i| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                if mask.masks(i) {
                    None
                } else {
                    Some((s >> 33) as i64 % TROPICAL_SPAN - TROPICAL_SPAN / 2)
                }
            })
            .collect()
    }

    /// The draw at `Trop<i8>`, the width the selection benchmarks and the
    /// witness gates run at.
    pub fn fill_tropical_i8(&self, len: usize, salt: u64, mask: Mask) -> Vec<Trop<i8>> {
        self.draw_tropical(len, salt, mask)
            .into_iter()
            .map(|v| v.map_or(Trop::NEG_INF, |x| Trop::finite(x as i8)))
            .collect()
    }

    /// The draw at `Trop<i64>`, the width the committed NumPy witness speaks.
    ///
    /// The same numbers as [`Case::fill_tropical_i8`], not a second draw: every
    /// value lands in [`TROPICAL_RANGE`], which both element types hold
    /// exactly, so the two spellings agree element for element.
    pub fn fill_tropical_i64(&self, len: usize, salt: u64, mask: Mask) -> Vec<Trop<i64>> {
        self.draw_tropical(len, salt, mask)
            .into_iter()
            .map(|v| v.map_or(Trop::NEG_INF, Trop::finite))
            .collect()
    }
}

/// How many distinct finite values a tropical draw takes.
///
/// Three: the smallest span that still carries a negative, a zero and a
/// positive, so that a sign error in a decode is visible and a tie is the
/// common case rather than a rarity. The constant is here rather than at a call
/// site because the Python generator re-spells it, and a second copy of a
/// corpus parameter is a second corpus.
pub const TROPICAL_SPAN: i64 = 3;

/// The closed range every tropical draw lands in: `-1..=1`.
pub const TROPICAL_RANGE: core::ops::RangeInclusive<i64> =
    (-(TROPICAL_SPAN / 2))..=(TROPICAL_SPAN / 2);

/// Which lanes of an operand are at the semiring zero.
///
/// Structural rather than drawn, for two reasons. A mask that came out of the
/// generator would be present with high probability rather than *present*, so a
/// case could quietly carry none; and A-6's mask is a property of the operand's
/// shape --- a padded tail, a dropped head, an unused expert --- not a property
/// of its values.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Mask {
    /// Every `period`-th lane is masked.
    pub period: usize,
    /// Counting from this one.
    pub phase: usize,
}

impl Mask {
    /// The left operand's mask: every seventh lane from the fourth.
    pub const LEFT: Self = Self {
        period: 7,
        phase: 3,
    };

    /// The right operand's: every fifth from the third.
    ///
    /// Coprime with [`Mask::LEFT`]'s period, so the two masks do not align into
    /// the same column of every cell --- which would make one operand's mask
    /// unobservable behind the other's.
    pub const RIGHT: Self = Self {
        period: 5,
        phase: 2,
    };

    /// No masked lane at all, for the half of a comparison that has to isolate
    /// the mask's effect.
    pub const NONE: Self = Self {
        period: 0,
        phase: 0,
    };

    /// Is lane `i` at the semiring zero?
    pub const fn masks(&self, i: usize) -> bool {
        self.period != 0 && i % self.period == self.phase
    }

    /// The smallest operand length this mask is present in.
    ///
    /// A case shorter than this carries no masked lane, which is a fact about
    /// the shape and not a gap: it is what lets a test assert *presence* rather
    /// than assert a probability.
    pub const fn shortest_masked(&self) -> usize {
        if self.period == 0 {
            usize::MAX
        } else {
            self.phase + 1
        }
    }
}

/// The standing corpus.
#[derive(Clone, Debug)]
pub struct Corpus {
    /// Every case.
    pub cases: Vec<Case>,
}

impl Corpus {
    /// The corpus every `CX-*` and `CB-*` test walks.
    pub fn standard(seed: u64) -> Self {
        let shapes: &[(usize, usize, usize)] = &[
            // Degenerate. Not special cases; they take the same path (CT-04).
            (0, 0, 0),
            (0, 5, 7),
            (5, 0, 7),
            (5, 7, 0),
            (1, 1, 1),
            // Primes, so no block size divides anything.
            (3, 7, 11),
            (13, 17, 19),
            (23, 29, 31),
            (1, 97, 1),
            (97, 1, 97),
            // One either side of a power of two, where a padded kernel would
            // show its seams.
            (7, 8, 9),
            (15, 16, 17),
            (31, 32, 33),
            (63, 64, 65),
            // Rectangular extremes.
            (1, 1024, 1),
            (128, 3, 128),
            (2, 4096, 2),
        ];
        Self {
            cases: shapes
                .iter()
                .enumerate()
                .map(|(i, &(m, k, n))| Case {
                    m,
                    k,
                    n,
                    seed: seed.wrapping_add(i as u64),
                })
                .collect(),
        }
    }

    /// The corpus every selection gate walks: `CX-11`, `CD-24`, `CD-25`.
    ///
    /// A second constructor rather than an extension of [`Corpus::standard`],
    /// and not for tidiness: `CA-02` pins a digest over the standing corpus's
    /// output bytes on every target, so a shape added there is a shape that
    /// breaks the digest on all of them.
    ///
    /// The shapes answer the questions a *selection* has and a sum does not.
    /// The witness of a cell is an index, so what has to be swept is where the
    /// index can come from: a depth of zero, where the answer is the
    /// past-the-end index; a depth of one, where the whole cell is either the
    /// one term or the semiring zero; depths that are not multiples of any
    /// panel length, so a tie is split across a partition boundary; and a depth
    /// far past any panel, where the winning index is nowhere near either end.
    pub fn tropical(seed: u64) -> Self {
        let shapes: &[(usize, usize, usize)] = &[
            // Depth zero: no term, so the witness is `k` --- and `k` is zero.
            (1, 0, 1),
            (3, 0, 4),
            // No output at all. The same path, said once (CT-04).
            (0, 0, 0),
            (0, 5, 7),
            (5, 7, 0),
            // Depth one, where a masked lane in either operand *is* the whole
            // reduction and the cell is the semiring zero. At this depth A's
            // lane index is the row and B's is the column, so extents that are
            // multiples of the two mask periods --- seven and five --- put
            // whole-masked cells in the corpus rather than hoping for them.
            (1, 1, 1),
            (7, 1, 5),
            (35, 1, 35),
            // Primes, so no panel length and neither mask period divides
            // anything, and a tie lands across a partition boundary.
            (3, 7, 11),
            (13, 17, 19),
            (23, 29, 31),
            // Deep and narrow: the winner is far from both ends of the
            // reduction, which is where a mechanism that quietly kept the last
            // index rather than the first would still look right at k = 2.
            (2, 97, 3),
            (1, 1024, 1),
            (4, 4093, 4),
            // Wide and shallow, the gemv-shaped selection.
            (128, 3, 128),
            // A power of two on every axis, where a panel divides the depth
            // exactly and every partition boundary falls in the same place.
            (16, 64, 16),
        ];
        Self {
            cases: shapes
                .iter()
                .enumerate()
                .map(|(i, &(m, k, n))| Case {
                    m,
                    k,
                    n,
                    seed: seed.wrapping_add(i as u64),
                })
                .collect(),
        }
    }
}

/// A small SHA-256, so that verifying a committed artifact needs no
/// dependency. R11 asks for a checksum; a checksum that required a crate to
/// check would be one more thing to trust.
///
/// Lives here rather than in any one harness because two committed corpora
/// verify digests --- the NumPy oracle's and the symbol corpus's --- and one
/// checksum routine is one thing to be wrong.
pub fn sha256_hex(bytes: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut msg = bytes.to_vec();
    let bit_len = (bytes.len() as u64) * 8;
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    for block in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, c) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes(c.try_into().unwrap());
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let mut v = h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ ((!v[4]) & v[6]);
            let t1 = v[7]
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v = [
                t1.wrapping_add(t2),
                v[0],
                v[1],
                v[2],
                v[3].wrapping_add(t1),
                v[4],
                v[5],
                v[6],
            ];
        }
        for (dst, src) in h.iter_mut().zip(v.iter()) {
            *dst = dst.wrapping_add(*src);
        }
    }
    h.iter().map(|w| format!("{w:08x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The seed the selection gates walk the tropical corpus at.
    const SEED: u64 = 20_260_805;

    /// How many cells of `A ⊗ B` have their maximum attained by two or more
    /// terms, and how many cells there are.
    ///
    /// A cell whose every term is the semiring zero counts as tied: every term
    /// attains the maximum there, which is exactly the case D-6's tie-break has
    /// to decide.
    fn ties(m: usize, k: usize, n: usize, a: &[Option<i64>], b: &[Option<i64>]) -> (usize, usize) {
        let (mut tied, mut cells) = (0, 0);
        for i in 0..m {
            for j in 0..n {
                cells += 1;
                let (mut best, mut count) = (None::<i64>, 0usize);
                for p in 0..k {
                    let term = match (a[i * k + p], b[p * n + j]) {
                        (Some(x), Some(y)) => Some(x + y),
                        _ => None,
                    };
                    match (term, best) {
                        (Some(v), Some(w)) if v > w => (best, count) = (term, 1),
                        (Some(v), Some(w)) if v == w => count += 1,
                        (Some(_), None) => (best, count) = (term, 1),
                        (None, None) => count += 1,
                        _ => {}
                    }
                }
                if count >= 2 {
                    tied += 1;
                }
            }
        }
        (tied, cells)
    }

    /// The ring corpus's own wide fill, masked identically --- the control.
    fn wide(case: &Case, len: usize, salt: u64, mask: Mask) -> Vec<Option<i64>> {
        case.fill_i8(len, salt)
            .into_iter()
            .enumerate()
            .map(|(i, v)| if mask.masks(i) { None } else { Some(v as i64) })
            .collect()
    }

    /// The narrow draw forces ties; the wide one, at the same shapes, the same
    /// seeds and the same masks, does not.
    ///
    /// This is the whole justification for a second corpus, and it is written
    /// two-sided because the one-sided half is worthless: a tie-break gate over
    /// a wide randomized sweep reports green while deciding nothing, and reads
    /// exactly like a gate that decided something.
    #[test]
    fn the_narrow_draw_forces_ties_and_the_wide_one_does_not() {
        let (mut narrow_tied, mut narrow_cells) = (0, 0);
        let (mut wide_tied, mut wide_cells) = (0, 0);
        for case in Corpus::tropical(SEED).cases {
            let Case { m, k, n, .. } = case;
            // Depth zero has no term and so no tie to force; it is in the
            // corpus for the past-the-end witness, not for this measurement.
            if k < 2 || m * n == 0 {
                continue;
            }
            let (a, b) = (
                case.draw_tropical(m * k, 1, Mask::LEFT),
                case.draw_tropical(k * n, 2, Mask::RIGHT),
            );
            let (t, c) = ties(m, k, n, &a, &b);
            narrow_tied += t;
            narrow_cells += c;

            // Every case, not the corpus in aggregate: a corpus whose ties all
            // sat in one deep shape would leave every other shape's witness
            // undecided while the total looked healthy.
            assert!(t > 0, "{m}x{k}x{n}: the narrow draw tied nothing");
            // Past depth sixteen the tie stops being an event and becomes the
            // common case. Measured: 63%, 76%, 100%, 100%, 100%, 98%.
            if k >= 16 {
                assert!(
                    t * 2 >= c,
                    "{m}x{k}x{n}: only {t} of {c} cells tied at depth {k}"
                );
            }

            let (a, b) = (
                wide(&case, m * k, 1, Mask::LEFT),
                wide(&case, k * n, 2, Mask::RIGHT),
            );
            let (t, c) = ties(m, k, n, &a, &b);
            wide_tied += t;
            wide_cells += c;
        }
        assert_eq!(narrow_cells, wide_cells, "the same cells, twice");
        assert!(narrow_cells > 0, "the corpus must have cells");
        // The control, and the whole justification for a second corpus.
        // Measured on the shapes above: 4586 tied cells against 52, which is
        // the difference between a tie-break gate and a green row.
        assert!(
            narrow_tied >= 20 * wide_tied,
            "the draw, not the shapes, must be what forces the ties: \
             narrow {narrow_tied}, wide {wide_tied} of {wide_cells}"
        );
        eprintln!(
            "forced ties: narrow {narrow_tied}/{narrow_cells}, wide {wide_tied}/{wide_cells}"
        );
    }

    /// Every case large enough to hold one carries a masked lane, in both
    /// operands --- A-6 is half the claim, and a corpus that masked nothing
    /// would test the other half twice.
    #[test]
    fn every_tropical_case_carries_a_masked_lane() {
        let mut with_mask = 0;
        for case in Corpus::tropical(SEED).cases {
            let Case { m, k, n, .. } = case;
            for (len, salt, mask) in [(m * k, 1, Mask::LEFT), (k * n, 2, Mask::RIGHT)] {
                let drawn = case.draw_tropical(len, salt, mask);
                let masked = drawn.iter().filter(|v| v.is_none()).count();
                if len >= mask.shortest_masked() {
                    assert!(masked > 0, "{m}x{k}x{n}: an operand of {len} with no mask");
                    with_mask += 1;
                } else {
                    assert_eq!(masked, 0, "{m}x{k}x{n}: a mask below the shortest length");
                }
            }
        }
        // A corpus of nothing but degenerate shapes would satisfy the loop
        // above vacuously.
        assert!(with_mask >= 20, "too few masked operands: {with_mask}");

        // And the unmasked spelling masks nothing, which is what lets a
        // comparison isolate the mask's effect.
        let case = Corpus::tropical(SEED).cases[8];
        assert!(case
            .draw_tropical(64, 1, Mask::NONE)
            .iter()
            .all(Option::is_some));
    }

    /// The draw lands where it says it lands, so the two element spellings are
    /// the same numbers and the artifact's `-2^63` spelling of the semiring
    /// zero is unambiguous.
    #[test]
    fn the_tropical_draw_stays_in_its_declared_range() {
        let case = Corpus::tropical(SEED).cases[9];
        let drawn = case.draw_tropical(4096, 1, Mask::LEFT);
        assert!(drawn.iter().flatten().all(|v| TROPICAL_RANGE.contains(v)));
        // Every value in the range is drawn, or the span is a claim about a
        // generator that does not make it.
        for want in TROPICAL_RANGE {
            assert!(
                drawn.contains(&Some(want)),
                "the draw never produced {want}"
            );
        }
        let (i8s, i64s) = (
            case.fill_tropical_i8(64, 1, Mask::LEFT),
            case.fill_tropical_i64(64, 1, Mask::LEFT),
        );
        for (x, y) in i8s.iter().zip(&i64s) {
            assert_eq!(x.get().map(i64::from), y.get(), "one draw, two spellings");
        }
    }
}
