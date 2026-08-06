//! Content addressing (§6.4).
//!
//! A weight artifact's identity is the kappa label of its **canonical
//! manifest**, not of its code bytes, so that a transcode between tiers is
//! visibly a different artifact with a provably identical decoded stream. That
//! is `CL-MM01` with an address attached to each side, and `CK-05` is the test
//! that says it.
//!
//! Bulk arrays are referenced by digest rather than inlined, so the manifest
//! stays inside `uor-addr-1`'s depth and width ceilings and the address stays
//! cheap.
//!
//! # Allocation
//!
//! [`Manifest::write_canonical_json`] writes into a caller-supplied buffer and
//! allocates nothing, so the default build of this crate remains heap-free
//! (R7). The `kappa` feature additionally pulls in `uor-addr-1` to turn that
//! JSON into a label; that crate owns its own allocation, which is why the
//! feature is off by default and why the manifest writer is usable without it.

use uor_matmul_core::{Bound, Shape, NARROW_CAP};

use crate::tier::TierId;

/// Bytes in a kappa label.
pub const ADDRESS_LABEL_BYTES: usize = 71;

/// A canonical weight-artifact manifest.
///
/// The field set and its JSON spelling are normative and are restated in
/// `ARCHITECTURE.md`. Changing either changes every artifact's identity, which
/// is why the schema carries a `spec` tag.
///
/// There is deliberately no code-width field. The width is a property of the
/// code *bytes*, which `codes_sha256` already distinguishes: a `u8` spelling
/// and a `u16` spelling of one tier decode alike and digest differently, so
/// they are two artifacts with two addresses --- the same rule `CK-05` states
/// for equal-decoding codecs generally. Nothing downstream of the manifest
/// reads the codes width-sensitively: the only reader is the decoder, which
/// learns the width from the artifact's own type.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Manifest<'a> {
    /// Which tier decodes the codes.
    pub tier: TierId,
    /// The alphabet bound the decoded stream satisfies.
    pub bound: u128,
    /// Rows of the decoded matrix.
    pub rows: usize,
    /// Decoded elements per row.
    pub cols: usize,
    /// Alphabet elements produced per code.
    pub block: usize,
    /// `sha256:<64hex>` of the codebook, or of the empty table for a codec that
    /// has none.
    pub codebook_sha256: &'a str,
    /// `sha256:<64hex>` of the code bytes.
    pub codes_sha256: &'a str,
    /// The schema tag.
    pub spec: &'a str,
}

/// What a manifest says about addressing the artifact it describes.
///
/// Derived, never declared. [`Manifest`]'s field set is normative and carries
/// no addressing field; it gains none here, because a field would be a second
/// source for what the tier, the block and the bound already fix (R10) as well
/// as a change to every artifact's identity.
///
/// Those three are the manifest's fields that describe the *code*. The two that
/// describe the artifact's *bytes* --- `codes_sha256` and `codebook_sha256` ---
/// are not read, and that absence is the whole of `CS-10`: two artifacts of one
/// tier, one block and one bound address alike however far apart their code
/// bytes are, so a traversal chosen from this cannot have probed either one.
///
/// It says nothing about whether a *table* over the code space exists. That is
/// [`crate::Enumerable`]'s question, it is answered by the type at the
/// tabulated traversal's boundary rather than by a token, and a composing tier
/// --- [`crate::Packed`], [`crate::Offset`], [`crate::Transcode`] --- reports
/// its own tier while inheriting its inner codec's enumeration. What is stated
/// here is the block: how many elements one code names, and how far a lane
/// carries their partial sums.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Addressing {
    /// No code of this artifact indexes a decode.
    ///
    /// Either the manifest names no element per code at all, or the tier is one
    /// of the two with nothing between a code and an element:
    /// [`TierId::Identity`], whose codes *are* the alphabet, so its code space
    /// is as wide as the element type; and [`TierId::Runs`], whose code widths
    /// are the data, so there is no `p`-th block for anything to be built per.
    /// Those are the two tiers `ARCHITECTURE.md` names as admitting no table;
    /// neither implements [`crate::Enumerable`], and no composition of them can,
    /// because a composing tier reports its own token and its own enumeration.
    ///
    /// Reading the tier for this is not a dispatch on the *answer*: two codecs
    /// with different tiers and equal decodes still write byte-identical output
    /// (`CK-05`), and the table and the stream are held to the same bytes
    /// either way (`CD-13`). What it decides is which factorizations exist.
    Nothing,
    /// A code names `elements` consecutive elements of one row, so a table
    /// indexed by the code space can carry their partial sum against an
    /// activation block of the same length.
    ARunOf {
        /// Consecutive elements of a row one code names.
        ///
        /// One is well formed and is what every scalar tier declares. It is the
        /// block over which a table sums nothing --- one code, one product ---
        /// which is why `tabulation_pays` refuses it on op count and routes the
        /// arena tier back to the dense traversal.
        elements: usize,
        /// Partial sums of one such run that a narrow lane word holds exactly.
        ///
        /// A product of two alphabet elements has magnitude at most `bound^2`,
        /// and a run of `elements` of them at most `elements * bound^2`, so a
        /// lane holding [`NARROW_CAP`] holds this many runs and no more.
        ///
        /// `None` when no run fits one at all: a bound wide enough that a single
        /// block already exceeds the lane, and in particular the `u128::MAX` a
        /// float codebook declares through `Whole`, which is not a magnitude.
        /// The reduction is then carried in the exact accumulator --- where a
        /// family with no narrow register was always going to carry it.
        lane_run: Option<usize>,
    },
}

impl Addressing {
    /// The addressing a tier, a block and a bound declare. The whole of the
    /// derivation, and its only entry point.
    pub const fn of(tier: TierId, block: usize, bound: u128) -> Self {
        match tier {
            TierId::Identity | TierId::Runs => Self::Nothing,
            // A code that names no element addresses nothing, whatever its tier.
            // `CodedMatrix::new` refuses such a codec outright, so this is the
            // same non-existence stated one step earlier.
            _ if block == 0 => Self::Nothing,
            _ => Self::ARunOf {
                elements: block,
                lane_run: lane_run(block, bound),
            },
        }
    }

    /// Does one code name an element at all?
    pub const fn addresses_an_element(self) -> bool {
        matches!(self, Self::ARunOf { .. })
    }

    /// Does one code name a *run*, so that a table entry is a partial sum of
    /// more than one product?
    ///
    /// This is the term the tabulated traversal's break-even turns on, and it
    /// is false at `MAX_BLOCK == 1` for the reason stated on `elements` above:
    /// a table of one product per entry repays no build at any width.
    pub const fn addresses_a_run(self) -> bool {
        matches!(self, Self::ARunOf { elements, .. } if elements > 1)
    }
}

/// Partial sums of a `block`-long run that one narrow lane word holds exactly.
///
/// The same derivation `uor_matmul_core`'s narrow run is, one level up: there it
/// is products that are counted against [`NARROW_CAP`] and here it is blocks of
/// them, so the per-code magnitude carries an extra factor of `block` and
/// nothing else changes. The cap is read from core rather than restated,
/// because a constant with two sources is a constant with none (R10).
const fn lane_run(block: usize, bound: u128) -> Option<usize> {
    if bound == 0 || block == 0 {
        // An alphabet of one value, or a code that names no element: neither
        // moves the lane, so no run of them ever fills it. The same answer core's
        // own run derivation gives at a bound of zero, for the same reason.
        return Some(usize::MAX);
    }
    let square = match bound.checked_mul(bound) {
        Some(v) => v,
        // A bound that cannot be squared is not a magnitude: it is `Whole`'s,
        // which declares that the codebook itself is the alphabet.
        None => return None,
    };
    let per_code = match square.checked_mul(block as u128) {
        Some(v) => v,
        None => return None,
    };
    let run = NARROW_CAP / per_code;
    if run == 0 {
        return None;
    }
    // A run wider than the machine can index is the machine's limit, not the
    // lane's, and clamping says so without inventing a smaller one (R8).
    if run > usize::MAX as u128 {
        Some(usize::MAX)
    } else {
        Some(run as usize)
    }
}

/// The manifest could not be rendered or addressed.
///
/// Neither variant can be caused by the *values* in a matrix, only by a
/// manifest that does not describe an artifact. Like [`uor_matmul_core::
/// NotAProduct`], this is non-existence, decided before any arithmetic.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[non_exhaustive]
pub enum KappaError {
    /// The caller's buffer is too small for the canonical JSON.
    BufferTooSmall {
        /// Bytes the manifest needs.
        needed: usize,
        /// Bytes the caller offered.
        offered: usize,
    },
    /// A digest field is not a `sha256:<64hex>` string.
    MalformedDigest,
    /// The addressing transform rejected the manifest.
    NotAddressable,
}

impl core::fmt::Display for KappaError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::BufferTooSmall { needed, offered } => {
                write!(
                    f,
                    "the canonical manifest needs {needed} bytes, {offered} offered"
                )
            }
            Self::MalformedDigest => write!(f, "a digest field is not sha256:<64hex>"),
            Self::NotAddressable => write!(f, "the addressing transform rejected the manifest"),
        }
    }
}

/// A cursor that writes into a caller's buffer and never allocates.
struct Out<'a> {
    buf: &'a mut [u8],
    at: usize,
    overflowed: bool,
}

impl Out<'_> {
    fn push(&mut self, bytes: &[u8]) {
        let end = self.at.saturating_add(bytes.len()); // R3-ok: a buffer cursor, checked below
        if end > self.buf.len() {
            self.overflowed = true;
            self.at = end;
            return;
        }
        self.buf[self.at..end].copy_from_slice(bytes);
        self.at = end;
    }

    fn push_u128(&mut self, mut v: u128) {
        // 39 is the decimal width of `u128::MAX`, so this buffer cannot be
        // outrun by any input. A derivation, not a choice (R8).
        let mut digits = [0u8; 39];
        let mut n = 0;
        if v == 0 {
            self.push(b"0");
            return;
        }
        while v > 0 {
            digits[n] = b'0' + (v % 10) as u8;
            v /= 10;
            n += 1;
        }
        for i in (0..n).rev() {
            self.push(&digits[i..=i]);
        }
    }
}

impl Manifest<'_> {
    /// What this manifest says about addressing the artifact it describes.
    ///
    /// Read from the tier, the block and the bound, and from neither digest.
    /// See [`Addressing`] for why that absence is the claim rather than an
    /// omission (`CS-10`).
    pub const fn addressing(&self) -> Addressing {
        Addressing::of(self.tier, self.block, self.bound)
    }

    /// Does this artifact stand as the coded operand of `shape` with the
    /// reduction running *along* its code blocks?
    ///
    /// `rows == n` and `cols == k`: one coded row per output column, so a code
    /// block is a run of the reduction, a table entry is a partial sum of it,
    /// and the product is `C := A * W^T`. That is the orientation
    /// `uor_matmul_gemm::TabulatedTriple` takes, and this asks its constructor's
    /// question of the *declaration*, before there is anything to construct.
    ///
    /// Two queries rather than one enum, because at `k == n` a square coded
    /// operand satisfies both and an enum would have to pick --- inventing a
    /// distinction the declaration does not make. Which of the two products is
    /// meant is then named by the triple the caller builds, and neither answer
    /// is wrong.
    pub const fn reduces_along_the_block(&self, shape: Shape) -> bool {
        self.rows == shape.n && self.cols == shape.k
    }

    /// Does this artifact stand as the coded operand of `shape` with the
    /// reduction running *across* its code blocks?
    ///
    /// `rows == k` and `cols == n`: one coded row per step of the reduction, so
    /// a code block is a run of `MAX_BLOCK` different *output columns* and there
    /// is nothing for a partial sum to be a partial sum of. That is the
    /// orientation `uor_matmul_gemm::CodedTriple` takes --- the streaming one,
    /// which needs no offer at all. Not a lesser orientation and not a fallback:
    /// it is the one a `k x n` quantized weight is already stored in.
    pub const fn reduces_across_the_block(&self, shape: Shape) -> bool {
        self.rows == shape.k && self.cols == shape.n
    }

    /// Write the JCS-RFC8785 canonical JSON for this manifest.
    ///
    /// Returns the number of bytes written. Object members are emitted in
    /// lexicographic order of their keys, with no whitespace and no escapes,
    /// which is what JCS requires and what makes two independently produced
    /// manifests of the same artifact byte-identical.
    ///
    /// Allocates nothing. If `out` is too small the needed length is reported
    /// rather than a partial write being passed off as a manifest.
    pub fn write_canonical_json(&self, out: &mut [u8]) -> Result<usize, KappaError> {
        for d in [self.codebook_sha256, self.codes_sha256] {
            if !is_sha256(d) {
                return Err(KappaError::MalformedDigest);
            }
        }
        let mut w = Out {
            buf: out,
            at: 0,
            overflowed: false,
        };

        // Lexicographic key order: block, bound, codebook_sha256,
        // codes_sha256, cols, rows, spec, tier.
        w.push(b"{\"block\":");
        w.push_u128(self.block as u128);
        w.push(b",\"bound\":");
        w.push_u128(self.bound);
        w.push(b",\"codebook_sha256\":\"");
        w.push(self.codebook_sha256.as_bytes());
        w.push(b"\",\"codes_sha256\":\"");
        w.push(self.codes_sha256.as_bytes());
        w.push(b"\",\"cols\":");
        w.push_u128(self.cols as u128);
        w.push(b",\"rows\":");
        w.push_u128(self.rows as u128);
        w.push(b",\"spec\":\"");
        w.push(self.spec.as_bytes());
        w.push(b"\",\"tier\":\"");
        w.push(self.tier.as_str().as_bytes());
        w.push(b"\"}");

        if w.overflowed {
            return Err(KappaError::BufferTooSmall {
                needed: w.at,
                offered: w.buf.len(),
            });
        }
        Ok(w.at)
    }

    /// The manifest for a coded matrix, given the two digests the caller has
    /// computed over the bulk arrays.
    pub fn of<E, Bd, C>(
        matrix: &crate::CodedMatrix<'_, E, Bd, C>,
        codebook_sha256: &'static str,
        codes_sha256: &'static str,
        spec: &'static str,
    ) -> Manifest<'static>
    where
        E: uor_matmul_core::Element,
        Bd: Bound,
        C: crate::Codec<E, Bd>,
    {
        Manifest {
            tier: C::TIER,
            bound: Bd::VALUE,
            rows: matrix.rows(),
            cols: matrix.cols(),
            block: C::MAX_BLOCK,
            codebook_sha256,
            codes_sha256,
            spec,
        }
    }
}

fn is_sha256(s: &str) -> bool {
    let Some(hex) = s.strip_prefix("sha256:") else {
        return false;
    };
    hex.len() == 64
        && hex
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

/// Address a manifest, writing the label into a caller-supplied buffer.
///
/// The label is `uor-addr-1`'s JCS-RFC8785 + NFC + SHA-256 transform applied to
/// [`Manifest::write_canonical_json`]'s output.
#[cfg(feature = "kappa")]
pub fn address_into(
    manifest: &Manifest<'_>,
    scratch: &mut [u8],
    out: &mut [u8; ADDRESS_LABEL_BYTES],
) -> Result<(), KappaError> {
    let n = manifest.write_canonical_json(scratch)?;
    let outcome = uor_addr_1::address(&scratch[..n]).map_err(|_| KappaError::NotAddressable)?;
    // `AddressOutcome::address`, not `.label`: the field is the ASCII wire form,
    // `sha256:<64 lowercase hex>`, which is the 71 bytes `ADDRESS_LABEL_BYTES`
    // names. This read `.label`, a field the crate does not have, and said so
    // only when something built the `kappa` feature --- which nothing did.
    let label = outcome.address.as_bytes();
    if label.len() != ADDRESS_LABEL_BYTES {
        return Err(KappaError::NotAddressable);
    }
    out.copy_from_slice(label);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const D0: &str = "sha256:0000000000000000000000000000000000000000000000000000000000000000";
    const D1: &str = "sha256:1111111111111111111111111111111111111111111111111111111111111111";
    const D2: &str = "sha256:2222222222222222222222222222222222222222222222222222222222222222";

    /// CK-08: the manifest is canonical --- keys in lexicographic order, no
    /// whitespace --- so two independent producers of the same artifact write
    /// the same bytes and therefore mint the same label.
    #[test]
    fn canonical_json_is_byte_stable_ck_08() {
        let m = Manifest {
            tier: TierId::Book,
            bound: 127,
            rows: 4096,
            cols: 4096,
            block: 8,
            codebook_sha256: D0,
            codes_sha256: D1,
            spec: "uor-matmul/1",
        };
        let mut buf = [0u8; 512];
        let n = m.write_canonical_json(&mut buf).unwrap();
        let text = core::str::from_utf8(&buf[..n]).unwrap();
        assert_eq!(
            text,
            concat!(
                "{\"block\":8,\"bound\":127,",
                "\"codebook_sha256\":\"sha256:00000000000000000000000000000000",
                "00000000000000000000000000000000\",",
                "\"codes_sha256\":\"sha256:11111111111111111111111111111111",
                "11111111111111111111111111111111\",",
                "\"cols\":4096,\"rows\":4096,\"spec\":\"uor-matmul/1\",\"tier\":\"Book\"}"
            )
        );
    }

    /// A short buffer reports what it needed rather than truncating, because a
    /// truncated manifest would address a different artifact.
    #[test]
    fn a_short_buffer_reports_the_need_ck_08() {
        let m = Manifest {
            tier: TierId::Identity,
            bound: 1,
            rows: 1,
            cols: 1,
            block: 1,
            codebook_sha256: D0,
            codes_sha256: D1,
            spec: "uor-matmul/1",
        };
        let mut buf = [0u8; 8];
        match m.write_canonical_json(&mut buf) {
            Err(KappaError::BufferTooSmall { needed, offered }) => {
                assert!(needed > 8);
                assert_eq!(offered, 8);
            }
            other => panic!("expected BufferTooSmall, got {other:?}"),
        }
    }

    /// A malformed digest is rejected before anything is written: an artifact
    /// whose bulk arrays are not addressed is not an addressable artifact.
    #[test]
    fn a_malformed_digest_is_rejected_ck_08() {
        let m = Manifest {
            tier: TierId::Identity,
            bound: 1,
            rows: 1,
            cols: 1,
            block: 1,
            codebook_sha256: "not-a-digest",
            codes_sha256: D1,
            spec: "uor-matmul/1",
        };
        let mut buf = [0u8; 512];
        assert_eq!(
            m.write_canonical_json(&mut buf),
            Err(KappaError::MalformedDigest)
        );
    }

    /// CK-08: the arena tier's spelling is pinned like every other token. A
    /// float alphabet has no magnitude, so the bound field is `Whole`'s value,
    /// recorded as itself --- and a new token mints new addresses, which is the
    /// point of the tier (CL-MM01).
    #[test]
    fn arena_manifest_spelling_is_byte_stable_ck_08() {
        let m = Manifest {
            tier: TierId::Arena,
            bound: u128::MAX,
            rows: 4096,
            cols: 4096,
            block: 1,
            codebook_sha256: D0,
            codes_sha256: D1,
            spec: "uor-matmul/1",
        };
        let mut buf = [0u8; 512];
        let n = m.write_canonical_json(&mut buf).unwrap();
        let text = core::str::from_utf8(&buf[..n]).unwrap();
        assert_eq!(
            text,
            concat!(
                "{\"block\":1,\"bound\":340282366920938463463374607431768211455,",
                "\"codebook_sha256\":\"sha256:00000000000000000000000000000000",
                "00000000000000000000000000000000\",",
                "\"codes_sha256\":\"sha256:11111111111111111111111111111111",
                "11111111111111111111111111111111\",",
                "\"cols\":4096,\"rows\":4096,\"spec\":\"uor-matmul/1\",\"tier\":\"Arena\"}"
            )
        );
    }

    /// `CS-10`: addressing is derived from the declaration, and the two fields
    /// that move with the artifact's bytes are the two it does not read.
    ///
    /// Both directions, at the declaration level. A one-sided version --- only
    /// that equal declarations address alike --- passes for a derivation that
    /// returns a constant, so the second half asserts that each of the three
    /// fields it *does* read moves the answer.
    #[test]
    fn addressing_is_read_from_the_declaration_cs_10() {
        let e8 = Manifest {
            tier: TierId::Book,
            bound: 128,
            rows: 4096,
            cols: 4096,
            block: 8,
            codebook_sha256: D0,
            codes_sha256: D1,
            spec: "uor-matmul/1",
        };

        // Two artifacts, one declaration. Only the digests moved --- which is
        // exactly what "the values changed" means to a manifest --- and the
        // addressing did not.
        let other = Manifest {
            codebook_sha256: D2,
            codes_sha256: D2,
            ..e8
        };
        assert_ne!(e8, other, "different bytes are a different artifact");
        let mut lhs = [0u8; 512];
        let mut rhs = [0u8; 512];
        let ln = e8.write_canonical_json(&mut lhs).unwrap();
        let rn = other.write_canonical_json(&mut rhs).unwrap();
        assert_ne!(lhs[..ln], rhs[..rn], "and a different canonical manifest");
        assert_eq!(e8.addressing(), other.addressing());

        // The run and the lane, recomputed here rather than recalled: eight
        // products of two elements of magnitude 128 apiece, against the cap one
        // narrow word holds, clamped where a 32-bit machine cannot index that
        // far.
        let want_run = (NARROW_CAP / (8 * 128 * 128)).min(usize::MAX as u128) as usize;
        assert_eq!(
            e8.addressing(),
            Addressing::ARunOf {
                elements: 8,
                lane_run: Some(want_run),
            }
        );
        assert!(e8.addressing().addresses_a_run());
        assert!(e8.addressing().addresses_an_element());

        // The block moves it. One element per code addresses an element and not
        // a run, which is the term the tabulated break-even refuses.
        let scalar = Manifest { block: 1, ..e8 };
        assert!(scalar.addressing().addresses_an_element());
        assert!(!scalar.addressing().addresses_a_run());
        assert_ne!(scalar.addressing(), e8.addressing());
        assert_eq!(Addressing::of(TierId::Book, 0, 128), Addressing::Nothing);

        // The bound moves it. `Whole`'s `u128::MAX` is not a magnitude, so no
        // run of any length fits a narrow word and the partial sums are carried
        // in the exact accumulator --- which is what the arena tier declares.
        assert_eq!(
            Addressing::of(TierId::Arena, 1, u128::MAX),
            Addressing::ARunOf {
                elements: 1,
                lane_run: None,
            }
        );
        // And so does a bound one block of which already exceeds the lane.
        assert_eq!(
            Addressing::of(TierId::Book, 8, 1u128 << 40),
            Addressing::ARunOf {
                elements: 8,
                lane_run: None,
            }
        );
        // A bound of zero is the alphabet `{0}`: nothing fills the lane, ever.
        assert_eq!(
            Addressing::of(TierId::Book, 8, 0),
            Addressing::ARunOf {
                elements: 8,
                lane_run: Some(usize::MAX),
            }
        );

        // The tier moves it, for the two tiers with nothing between a code and
        // an element --- whatever their block and bound say.
        assert_eq!(
            Addressing::of(TierId::Identity, 1, 128),
            Addressing::Nothing
        );
        assert_eq!(Addressing::of(TierId::Runs, 8, 128), Addressing::Nothing);

        // Orientation, read from `rows` and `cols` and from nothing else, at a
        // shape where `k != n` so the two are distinguishable.
        let shape = Shape { m: 3, k: 64, n: 40 };
        let along = Manifest {
            rows: 40,
            cols: 64,
            ..e8
        };
        let across = Manifest {
            rows: 64,
            cols: 40,
            ..e8
        };
        assert!(along.reduces_along_the_block(shape));
        assert!(!along.reduces_across_the_block(shape));
        assert!(across.reduces_across_the_block(shape));
        assert!(!across.reduces_along_the_block(shape));
        // The two differ in `rows` and `cols` alone, so the orientation came
        // from the declaration and the code declaration is untouched by it.
        assert_eq!(along.addressing(), across.addressing());

        // A square coded operand answers both, because at `k == n` the
        // declaration names no difference and neither answer is wrong.
        let square = Shape { m: 3, k: 64, n: 64 };
        let s = Manifest {
            rows: 64,
            cols: 64,
            ..e8
        };
        assert!(s.reduces_along_the_block(square));
        assert!(s.reduces_across_the_block(square));
    }
}
