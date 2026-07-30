#!/usr/bin/env python3
"""Generate the CK-14 symbol-corpus artifacts.

The realistic case the `u8` arena width exists for: an f32 weight artifact
whose values are drawn from at most 256 distinct bit patterns --- a
dequantized 8-bit artifact. The codebook is the distinct patterns in
canonical arena order (unsigned order on the bit patterns, duplicates
collapsed --- the discipline `canonicalize` applies, `CK-10`), and the codes
address it one byte a symbol. The dense spelling is committed beside the
codes so the fixture's reference is the artifact itself, not a re-derivation
of it.

Run once; the outputs are committed. The codebook is data with no quality
claim attached (N3): it is an input the library consumes, and fitting one
beyond this trivial case is not the library's business.
"""

import hashlib
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent
SEED = 20260730

# One awkward small case (prime in both extents) and one the size of a real
# layer, both stored `n x k`: one coded row per output column, the
# orientation the tabulated traversal takes.
SHAPES = [(13, 257), (1024, 1024)]


def xorshift(state):
    """The same generator the Rust corpus uses, so the fills are comparable."""
    while True:
        state ^= (state << 13) & 0xFFFFFFFFFFFFFFFF
        state ^= state >> 7
        state ^= (state << 17) & 0xFFFFFFFFFFFFFFFF
        yield (state >> 33) % 256


def main():
    # The codebook: an exact dequantization grid, (q - 127.5) * 2**-6. Every
    # value is exactly representable in f32 and the 256 bit patterns are
    # distinct by construction; the assertion asks rather than assumes.
    values = (np.arange(256, dtype=np.float32) - np.float32(127.5)) * np.float32(2.0**-6)
    bits = values.view(np.uint32)
    assert len(np.unique(bits)) == 256, "the grid must be 256 distinct bit patterns"
    # Canonical arena order: the unsigned order on bit patterns (CK-10).
    codebook = values[np.argsort(bits, kind="stable")]
    (HERE / "codebook.f32.bin").write_bytes(codebook.astype("<f4").tobytes())
    codebook_sha = "sha256:" + hashlib.sha256((HERE / "codebook.f32.bin").read_bytes()).hexdigest()

    gen = xorshift(SEED)
    cases = []
    for (n, k) in SHAPES:
        codes = np.array([next(gen) for _ in range(n * k)], dtype=np.uint8)
        # The dense spelling of the same artifact: the codebook read through
        # the codes, which is the tier's decode written out.
        weights = codebook[codes]
        stem = f"{n}x{k}"
        (HERE / f"{stem}.codes.bin").write_bytes(codes.tobytes())
        (HERE / f"{stem}.weights.f32.bin").write_bytes(weights.astype("<f4").tobytes())
        cases.append({
            "shape": {"n": n, "k": k},
            "codes_sha256": "sha256:" + hashlib.sha256(
                (HERE / f"{stem}.codes.bin").read_bytes()).hexdigest(),
            "weights_sha256": "sha256:" + hashlib.sha256(
                (HERE / f"{stem}.weights.f32.bin").read_bytes()).hexdigest(),
            "stem": stem,
        })

    manifest = {
        "id": "CK-14",
        "oracle": "none: a self-describing artifact, not an outside witness",
        "numpy_version": np.__version__,
        "dtype": "f32",
        "byte_order": "little-endian",
        "entry": "dequantized-8-bit-style f32 weights, <= 256 distinct bit patterns",
        "seed": SEED,
        "spec": "uor-matmul/1",
        "note": (
            "The codebook is the artifact's distinct bit patterns in canonical "
            "arena order; the codes address it one byte a symbol; the dense "
            "spelling is committed beside them so the fixture's reference is "
            "the artifact itself. No quality claim attaches to the codebook (N3)."
        ),
        "codebook_sha256": codebook_sha,
        "codebook_symbols": 256,
        "cases": cases,
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(cases)} cases, codebook of 256 symbols")


if __name__ == "__main__":
    main()
