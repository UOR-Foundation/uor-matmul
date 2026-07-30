//! The shape and data corpus every differential test walks.
//!
//! Shapes are chosen to be awkward on purpose: zero, one, primes, powers of two
//! plus one, and depths on both sides of every narrow-register threshold. A
//! corpus of round numbers would agree with anything.

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
