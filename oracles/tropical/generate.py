#!/usr/bin/env python3
"""Generate the CX-11 oracle artifacts: the `(max, +)` half of the census.

Run once; the outputs are committed. This is CX-10's sibling and it exists for
the same reason (D-1): NumPy is outside the Rust ecosystem entirely, so
agreement with it is evidence of a different kind from agreement with a crate
that shares a language, a compiler, and sometimes a kernel author.

What is different here is what "exact" is worth. A float oracle and this library
compute different functions and the deviation is the measurement (N1). An int64
ring oracle agrees byte for byte because reduction modulo 2^64 is a ring
homomorphism, which is an argument. A max-plus product needs neither argument
nor exemption: `max` selects one of the values offered to it and `+` is exact at
this range, so there is no rounding for two implementations to disagree about
and byte equality is the only defensible comparison.

The semiring zero is spelled -2^63 in the artifact. That is a property of the
*file* and not of the element type: a byte stream needs some spelling for `-inf`,
and the Rust side decodes it into `Trop`'s own discriminant on the way in, which
is exactly why `Trop` carries a separate field rather than reserving a pattern
inside its element (S5.2, R8). The draw below never produces a value near it.

The reduction here does use a sentinel, and deliberately so: it is the way an
outside implementation would write this, in the library it would write it in,
and an oracle that reproduced this repository's own construction would be a
weaker witness than one that did not. It is exact --- every finite term has
magnitude at most 2 and the sentinel is -2^40, so the two are separated by
thirty-nine bits and no finite sum can be mistaken for a masked one.
"""

import hashlib
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent

# `Corpus::tropical`'s seed in crates/uor-matmul-validate/tests/tropical_oracle.rs.
# Case `i` draws from SEED + i, exactly as the Rust corpus does.
SEED = 20260805

# `corpus::TROPICAL_SPAN`: three values, so that a tie is the common case rather
# than a rarity. The measurement that says so is
# `the_narrow_draw_forces_ties_and_the_wide_one_does_not`.
SPAN = 3

# `corpus::Mask::LEFT` and `corpus::Mask::RIGHT`: (period, phase). Coprime, so
# the two masks never align into the same column of every cell. Recorded in the
# manifest so the Rust harness can refuse an artifact drawn under other ones.
MASK_A = (7, 3)
MASK_B = (5, 2)

# `corpus::Corpus::tropical`'s shapes, in order. Depth zero for the
# past-the-end witness, depth one for the whole-cell mask, primes so no panel
# length divides anything, and depths far past any panel.
SHAPES = [
    (1, 0, 1), (3, 0, 4),
    (0, 0, 0), (0, 5, 7), (5, 7, 0),
    (1, 1, 1), (7, 1, 5), (35, 1, 35),
    (3, 7, 11), (13, 17, 19), (23, 29, 31),
    (2, 97, 3), (1, 1024, 1), (4, 4093, 4),
    (128, 3, 128),
    (16, 64, 16),
]

MASK64 = 0xFFFFFFFFFFFFFFFF
GOLDEN = 0x9E3779B97F4A7C15
NEG_INF = -(1 << 63)
SENTINEL = -(1 << 40)


def draw(seed, length, salt, mask):
    """The Rust corpus's `Case::draw_tropical`, re-spelled.

    The same xorshift the ring corpus uses, so one recorded generator covers
    both halves of the census; what differs is the width of the draw and the
    structural mask. Returns the artifact spelling: -2^63 for the semiring zero.
    """
    period, phase = mask
    state = seed ^ ((salt * GOLDEN) & MASK64)
    out = []
    for i in range(length):
        state ^= (state << 13) & MASK64
        state ^= state >> 7
        state ^= (state << 17) & MASK64
        if i % period == phase:
            out.append(NEG_INF)
        else:
            out.append((state >> 33) % SPAN - SPAN // 2)
    return np.array(out, dtype=np.int64)


def max_plus(a, b, m, k, n):
    """`C[i, j] = max_p (A[i, p] + B[p, j])`, over `E u {-inf}`.

    Masked lanes are carried through a sentinel far below any finite sum, so
    the whole reduction is one broadcast add and one `max` --- the obvious
    NumPy spelling, which is the point of using it as the witness.
    """
    if k == 0:
        # No term at all, so the reduction is the identity of `max`, which is
        # the semiring zero. Not a special case: it is what the general formula
        # says, and NumPy declines to reduce an empty axis without an identity.
        return np.full((m, n), NEG_INF, dtype=np.int64)
    aa = np.where(a == NEG_INF, SENTINEL, a).reshape(m, k)
    bb = np.where(b == NEG_INF, SENTINEL, b).reshape(k, n)
    terms = aa[:, :, None] + bb[None, :, :]
    c = terms.max(axis=1)
    # Anything that touched a masked lane is at most SENTINEL + 1; anything
    # that did not is at least -2. Half the sentinel separates them.
    return np.where(c <= SENTINEL // 2, NEG_INF, c).astype(np.int64)


def main():
    cases = []
    for i, (m, k, n) in enumerate(SHAPES):
        seed = SEED + i
        a = draw(seed, m * k, 1, MASK_A)
        b = draw(seed, k * n, 2, MASK_B)
        c = max_plus(a, b, m, k, n)

        stem = f"{m}x{k}x{n}"
        for name, arr in (("a", a), ("b", b), ("expected", c)):
            (HERE / f"{stem}.{name}.bin").write_bytes(arr.astype("<i8").tobytes())
        cases.append({
            "shape": {"m": m, "k": k, "n": n},
            "a_sha256": "sha256:" + hashlib.sha256((HERE / f"{stem}.a.bin").read_bytes()).hexdigest(),
            "b_sha256": "sha256:" + hashlib.sha256((HERE / f"{stem}.b.bin").read_bytes()).hexdigest(),
            "expected_sha256": "sha256:" + hashlib.sha256(
                (HERE / f"{stem}.expected.bin").read_bytes()).hexdigest(),
            "stem": stem,
        })

    manifest = {
        "id": "CX-11",
        "oracle": "numpy",
        "numpy_version": np.__version__,
        "dtype": "int64",
        "byte_order": "little-endian",
        "entry": "(A[:, :, None] + B[None, :, :]).max(axis=1)",
        "semiring": "(max, +)",
        "neg_inf": NEG_INF,
        "span": SPAN,
        "mask_a": {"period": MASK_A[0], "phase": MASK_A[1]},
        "mask_b": {"period": MASK_B[0], "phase": MASK_B[1]},
        "seed": SEED,
        "spec": "uor-matmul/1",
        "note": (
            "The (max, +) half of the operation census, witnessed out of process "
            "and outside the Rust ecosystem. A max-plus product has no rounding "
            "for two implementations to disagree about, so the comparison is byte "
            "equality with no exempted region. The semiring zero is spelled -2^63 "
            "in the artifact and decoded into the element type's own discriminant "
            "on the way in, so no sentinel enters the element type."
        ),
        "cases": cases,
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(cases)} cases against numpy {np.__version__}")


if __name__ == "__main__":
    main()
