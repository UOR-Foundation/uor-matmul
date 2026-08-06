//! The typed shape of `model/*.toml`.
//!
//! Nothing here interprets the model; [`crate::Model::check`] does that. These
//! types exist so that a malformed model is a parse error rather than a
//! silently wrong constant.

use serde::{Deserialize, Deserializer};

use crate::ModelError;

/// Deserialize a `u128` written either as a TOML integer or as a string.
///
/// TOML integers are `i64`, so `|i64::MIN|` --- the `FULL` of the `i64`
/// alphabet --- has no integer spelling. Writing it as a string is not a
/// workaround: it is the model saying that this value is a magnitude, not a
/// machine integer, which is exactly why the accumulator is wider than one.
fn u128_flex<'de, D: Deserializer<'de>>(d: D) -> Result<u128, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Repr {
        // TOML's integer type is i64; anything wider arrives as a string.
        Int(i64),
        Str(String),
    }
    match Repr::deserialize(d)? {
        Repr::Int(i) => u128::try_from(i).map_err(serde::de::Error::custom),
        Repr::Str(s) => s.trim().parse().map_err(serde::de::Error::custom),
    }
}

/// [`u128_flex`] for an optional field.
fn opt_u128_flex<'de, D: Deserializer<'de>>(d: D) -> Result<Option<u128>, D::Error> {
    Option::<toml::Value>::deserialize(d)?
        .map(|v| u128_flex(v).map_err(serde::de::Error::custom))
        .transpose()
}

/// One of the three UTQC honesty levels (R4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Level {
    /// A fact reproduced from an authority. Not established here.
    SomeTrue,
    /// Constructed here and validated against its oracle.
    Build,
    /// Measured and reported, never asserted.
    Open,
}

impl Level {
    /// The token used in `model/*.toml` and in generated documentation.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SomeTrue => "some-true",
            Self::Build => "build",
            Self::Open => "open",
        }
    }
}

/// `model/constants.toml`.
#[derive(Debug, Clone, Deserialize)]
pub struct Constants {
    /// The schema tag; must be `uor-matmul/1`.
    pub spec: String,
    /// The largest reduction depth any addressable machine can present.
    ///
    /// Declared, not probed, so the accumulator type is the same on a 64-bit
    /// host, a 32-bit host, and wasm32 (§3.2).
    pub max_k_bits: u32,
    /// Machine integer types usable as alphabet elements.
    pub element: Vec<Element>,
    /// Named `(element, bound)` pairs. Exactly one is canonical.
    pub instantiation: Vec<Instantiation>,
    /// Narrow-register thresholds. Not limits (§5.1).
    pub narrow: Narrow,
    /// Cache-shaped tuning parameters, allowlisted out of R1.
    pub blocking: Blocking,
}

/// A machine integer type usable as an alphabet element and as a coded weight.
#[derive(Debug, Clone, Deserialize)]
pub struct Element {
    /// The Rust type name.
    pub name: String,
    /// Width in bits.
    pub bits: u32,
    /// The largest magnitude the type can hold, so that `Full<E>` admits every
    /// value of `E` and the default path cannot reject anything.
    #[serde(deserialize_with = "u128_flex")]
    pub full: u128,
    /// Size of one value in bytes. For a complex type this is twice the
    /// component size, which is why the derivation reads it separately from
    /// `bits`.
    pub bytes: u32,
    /// Element-products summed by one `mac`. One for a scalar type, two for a
    /// complex one.
    #[serde(default = "one")]
    pub product_terms: u32,
    /// The resolved accumulator type name.
    pub accumulator: String,
}

const fn one() -> u32 {
    1
}

/// A named `(element, bound)` pair. W8A8 is the canonical one.
#[derive(Debug, Clone, Deserialize)]
pub struct Instantiation {
    /// The name used in documentation and test IDs.
    pub name: String,
    /// The element type this instantiates over.
    pub element: String,
    /// The declared alphabet bound `B`.
    #[serde(deserialize_with = "u128_flex")]
    pub bound: u128,
    /// Whether this is *the* canonical instantiation. Exactly one is.
    pub canonical: bool,
    /// Free-form provenance.
    #[serde(default)]
    pub note: String,
}

/// Narrow-register thresholds (§5.1, §7.3).
#[derive(Debug, Clone, Deserialize)]
pub struct Narrow {
    /// `i32::MAX`, the default cap for a 32-bit lane.
    #[serde(deserialize_with = "u128_flex")]
    pub cap_i32: u128,
    /// One row per instruction sequence.
    pub threshold: Vec<Threshold>,
}

/// The `k` at which one instruction sequence stops being usable for a tile.
///
/// This is not a limit on anything. A tile past its threshold uses a wider
/// register and computes the same integer.
#[derive(Debug, Clone, Deserialize)]
pub struct Threshold {
    /// Stable identifier, used in generated const names.
    pub name: String,
    /// The instruction sequence and its worst-case per-step magnitude.
    pub sequence: String,
    /// The worst-case magnitude contributed by one step.
    #[serde(deserialize_with = "u128_flex")]
    pub per_step: u128,
    /// `floor(cap / per_step)`, pinned so R1's gate has something to check.
    #[serde(deserialize_with = "u128_flex")]
    pub k_max: u128,
    /// A cap other than `cap_i32`, for sequences that accumulate in i16.
    #[serde(default, deserialize_with = "opt_u128_flex")]
    pub cap_override: Option<u128>,
}

/// Constants discovered by measurement: cache-shaped tuning, found empirically.
/// Changing one cannot change any output byte, only which traversal produces it
/// (`CD-01`). A derivation story for such a constant is fiction, so the table
/// records the measurement instead.
#[derive(Debug, Clone, Deserialize)]
pub struct Blocking {
    /// Asserts the R1 allowlist applies to this table.
    pub allowlisted_from_r1: bool,
    /// Why the allowlist is sound.
    pub reason: String,
    /// Rows of A per block.
    pub mc: usize,
    /// Depth per panel.
    pub kc: usize,
    /// Columns of B per block.
    pub nc: usize,
    /// The machine's cache line, in bytes. Sizes the transpose a reduce panel
    /// needs so that the operand is read once rather than once per column.
    pub cache_line_bytes: usize,
    /// The first level of data cache, in bytes. Sizes the tabulation table, which
    /// is read once per output column and must be resident to pay.
    pub l1_bytes: usize,
    /// The last level of cache a per-column read can afford, in bytes. Sizes the
    /// depth of a tabulation stack, which is what keeps the exact accumulator out
    /// of the inner loop.
    pub l2_bytes: usize,
    /// Products one instruction of a dense tile kernel issues. Makes the
    /// tabulation predicate a comparison of instructions rather than of
    /// operations.
    pub kernel_products_per_step: usize,
    /// Rows of the output a dense tile kernel produces per call. Below this many
    /// rows the tile pays for lanes that are not there, and the predicate scales
    /// the dense side accordingly.
    pub kernel_rows: usize,
    /// The smallest base-case extent at which one level of the sub-cubic
    /// recursion pays on the `i32`-exact lane, measured by `just strassen-sweep`.
    /// Below it a level's block sums and temporary traffic cost more than its
    /// product count saves.
    pub strassen_min_extent: usize,
    /// The smallest base-case extent at which one level of the modular bilinear
    /// factorization pays on a modular lane. `usize::MAX` while the lane has no
    /// measurement: the modular arm consults it and declines until the
    /// pre-registered sweep in `MEASUREMENT-LOG.md` records one.
    pub strassen_modular_min_extent: usize,
}

/// `model/widths.toml`.
#[derive(Debug, Clone, Deserialize)]
pub struct Widths {
    /// The schema tag.
    pub spec: String,
    /// One row per integer element type.
    pub width: Vec<Width>,
    /// One row per float element type.
    pub complete: Vec<Complete>,
}

/// The derived accumulator for one integer element type.
#[derive(Debug, Clone, Deserialize)]
pub struct Width {
    /// The element type name.
    pub element: String,
    /// `acc_bits(E)`: the worst case any addressable machine can express.
    pub bits: u32,
    /// The resolved accumulator type name.
    pub accumulator: String,
    /// The accumulator's actual width.
    pub acc_bits: u32,
    /// The accumulator's size in bytes.
    pub bytes: usize,
}

/// The derived complete accumulator for one float element type (§3.3).
#[derive(Debug, Clone, Deserialize)]
pub struct Complete {
    /// The element type name.
    pub element: String,
    /// Twice the minimum subnormal exponent.
    pub min_product_exp: i64,
    /// Twice the maximum finite exponent.
    pub max_product_exp: i64,
    /// The product exponent range.
    pub span_bits: i64,
    /// `log2` of the largest `k` the machine can address.
    pub guard_bits: u32,
    /// One.
    pub sign_bits: u32,
    /// Span plus guard plus sign.
    pub total_bits: i64,
    /// 64-bit limbs sufficient for `total_bits`.
    pub limbs: usize,
    /// The accumulator's size in bytes.
    pub bytes: usize,
}

/// `model/tiers.toml`.
#[derive(Debug, Clone, Deserialize)]
pub struct Tiers {
    /// The schema tag.
    pub spec: String,
    /// One row per codec tier.
    pub tier: Vec<Tier>,
    /// One row per shipped codebook.
    pub codebook: Vec<Codebook>,
    /// One row per enumerable codec: the codecs a table can be indexed by.
    pub tabulation: Vec<Tabulation>,
}

/// One instantiation of the `Codec` trait.
#[derive(Debug, Clone, Deserialize)]
pub struct Tier {
    /// The `TierId` discriminant name.
    pub id: String,
    /// The Rust type, with its parameters.
    pub rust: String,
    /// The stored code type.
    pub code: String,
    /// Alphabet elements produced per code, as a literal or an expression.
    pub block: toml::Value,
    /// Where the decode table lives. Always borrowed, never owned.
    pub table: String,
    /// Free-form notes, including anything normative about the encoding.
    #[serde(default)]
    pub note: String,
    /// The honesty level of the tier's claim.
    pub honesty: Level,
}

/// The pair the first-recorded rows are written for: `AVX2_TABLE`'s eight
/// lanes against `AVX2_I8_I32`'s sixteen products a step.
fn default_isa() -> String {
    "avx2".to_string()
}

/// One enumerable codec at one named pair of sequences, with the break-even
/// the op counts put it at.
///
/// The numbers here are *derivations*, not measurements: `break_even_n` is where
/// the tabulated op count crosses the dense one, and `CM-04` recomputes it rather
/// than trusting it. What it costs in wall time is `CG-10` and is `open`.
#[derive(Debug, Clone, Deserialize)]
pub struct Tabulation {
    /// The codec, spelled as the caller writes it.
    pub codec: String,
    /// The instruction set the row's pair of sequences is written for. Both
    /// sides are declarations a sequence makes about itself, so a break-even
    /// without one is a number about nothing. Rows that name nothing are the
    /// AVX2 pair the table was first recorded for.
    #[serde(default = "default_isa")]
    pub isa: String,
    /// Distinct codes: `Enumerable::CODE_SPACE`.
    pub code_space: usize,
    /// Alphabet elements one code names: `Codec::MAX_BLOCK`.
    pub block: usize,
    /// Rows of `A` one table entry covers at the canonical lane: the row tile
    /// `tabulation_rows` resolves to for this code space.
    pub rows: usize,
    /// Lane words one issued add covers: the register width the recorded
    /// break-even is written for.
    pub lanes_per_add: usize,
    /// Products one instruction of the dense tile issues on this row's ISA:
    /// the `products_per_step` of the `KernelSpec` the dense path would run
    /// there. Absent on the AVX2 rows, which read the model's
    /// `kernel_products_per_step` --- that constant *is* `AVX2_I8_I32`'s
    /// declaration. The dense row count is the blocking constant
    /// `kernel_rows` on every ISA, because the shipped predicate reads it
    /// rather than the chosen tile's `mr`.
    #[serde(default)]
    pub kernel_products_per_step: Option<usize>,
    /// The first `n` at which tabulation issues fewer operations than blocking.
    ///
    /// Absent when `block == 1`, where no such `n` exists: one code names one
    /// element, so the table replaces every multiply with a read and removes no
    /// add at all.
    #[serde(default)]
    pub break_even_n: Option<usize>,
    /// Free-form notes.
    #[serde(default)]
    pub note: String,
    /// The honesty level of the row's claim.
    pub honesty: Level,
}

/// A shipped codebook. Data, borrowed from the caller, never learned here.
#[derive(Debug, Clone, Deserialize)]
pub struct Codebook {
    /// Stable identifier.
    pub id: String,
    /// Number of entries.
    pub entries: usize,
    /// Alphabet elements per entry.
    pub block: usize,
    /// The element type the entries are drawn from.
    pub element: String,
    /// Size of the table in bytes.
    pub bytes: usize,
    /// Where the table came from.
    pub source: String,
    /// The honesty level of the codebook's own claim.
    pub honesty: Level,
    /// The quality claim, which is always "none" (N3).
    pub quality_claim: String,
}

/// `model/oracles.toml`.
#[derive(Debug, Clone, Deserialize)]
pub struct Oracles {
    /// The schema tag.
    pub spec: String,
    /// External crates used as oracles.
    pub oracle: Vec<Oracle>,
    /// Cited authorities. Not executed, never re-derived (§12, N2).
    pub authority: Vec<Authority>,
}

/// An external crate used as an oracle.
#[derive(Debug, Clone, Deserialize)]
pub struct Oracle {
    /// The conformance ID this oracle discharges.
    pub id: String,
    /// The crate name.
    #[serde(rename = "crate")]
    pub krate: String,
    /// The `uor-matmul-validate` feature that enables it.
    pub feature: String,
    /// The element type compared.
    pub element: String,
    /// The oracle's entry point.
    pub entry: String,
    /// `exact` or `inexact` (§3.4).
    pub exactness: String,
    /// Whether this oracle's lineage is independent of the others'.
    pub independent: bool,
    /// Whether the oracle is behind an opt-in feature.
    #[serde(default)]
    pub optional: bool,
    /// The honesty level of the resulting claim.
    pub honesty: Level,
    /// Free-form provenance.
    #[serde(default)]
    pub note: String,
}

/// A cited authority. It tells the implementation what identity to realize, and
/// it is never re-derived, vendored, or gated on.
#[derive(Debug, Clone, Deserialize)]
pub struct Authority {
    /// Stable identifier, e.g. `CL-MM01`.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Where the authority lives.
    pub source: String,
    /// Always `some-true`.
    pub honesty: Level,
    /// What the authority says.
    pub statement: String,
}

/// `model/ledger.toml`.
#[derive(Debug, Clone, Deserialize)]
pub struct Ledger {
    /// The schema tag.
    pub spec: String,
    /// The words `audit-deferral` reads as a deferral (R15).
    ///
    /// Here rather than inside the gate for the reason `admits` is: the gate
    /// scans the whole workspace including its own source, so a marker table
    /// written as Rust string literals would make the gate fail on itself. The
    /// alternative was exempting the gate's own file, which is a hole in exactly
    /// the place nobody looks. As model data the vocabulary is reviewable, and
    /// the gate's source contains no marker to trip over (R10).
    pub deferral_markers: Vec<String>,
    /// One row per claim.
    pub claim: Vec<Claim>,
    /// One row per suspended rule. Empty in the steady state: a suspension is
    /// an event, not a condition.
    #[serde(default)]
    pub suspension: Vec<Suspension>,
}

/// A working rule deliberately suspended for one named scope.
///
/// The rules in §0 are the repository's promises, so suspending one is itself
/// a claim and belongs in the ledger, where `CONFORMANCE.md` renders it: a
/// suspension that lived only in a gate's exemption list would be a hole
/// nobody reviews. What the suspension *admits* is enumerated here and read by
/// the gate from this file, so the exemption is model data (R10) rather than a
/// constant inside the gate.
#[derive(Debug, Clone, Deserialize)]
pub struct Suspension {
    /// Stable identifier, e.g. `SUSP-R15-WIDTH-PHASE`.
    pub id: String,
    /// The rule suspended, e.g. `R15`.
    pub rule: String,
    /// The scope the suspension is bounded to. A suspension without a named
    /// scope is a repeal, and that is a different decision.
    pub scope: String,
    /// Why the rule blocks the work and why suspending it is honest here.
    pub reason: String,
    /// The repo-relative paths the suspension exempts from the rule's gate.
    /// Never a crate: a deferral does not become admissible by being parked in
    /// shipped code, so an admitted path under `crates/` is rejected.
    #[serde(default)]
    pub admits: Vec<String>,
    /// How the suspension ends.
    pub ends: String,
}

/// One claim, at exactly one honesty level.
#[derive(Debug, Clone, Deserialize)]
pub struct Claim {
    /// The conformance ID, or an `AUTH-`/`OPEN-` prefixed identifier.
    pub id: String,
    /// The honesty level. Untagged claims do not ship (R4).
    pub level: Level,
    /// What is claimed.
    pub statement: String,
    /// The observation that would refute the claim.
    ///
    /// A `some-true` row is refuted by the cited text not carrying it at the
    /// citation given; an `open` row by its figure not reproducing at the host,
    /// seed, and harness it records. Both are conditions a third party can
    /// check without this repository's cooperation, which is what makes a
    /// citation different from an assertion.
    pub refuted_by: String,
    /// The Gherkin file carrying the scenario (R9).
    #[serde(default)]
    pub feature: Option<String>,
    /// The authority a `some-true` claim is reproduced from.
    #[serde(default)]
    pub authority: Option<String>,
    /// Recorded sample size, for a `CP-*` claim.
    #[serde(default)]
    pub sample_size: Option<u64>,
    /// Recorded seed, for a `CP-*` claim.
    #[serde(default)]
    pub seed: Option<u64>,
}

impl Ledger {
    /// The meta-gate's structural half: every claim is well formed for its
    /// level (R4), and every suspension names what it exempts.
    ///
    /// The behavioural half --- that no test asserts an `open` claim as
    /// established --- lives in `uor-matmul-conformance`, because it needs the
    /// test names, not the model.
    pub fn check(&self) -> Result<(), ModelError> {
        for s in &self.suspension {
            for (field, value) in [
                ("id", &s.id),
                ("rule", &s.rule),
                ("scope", &s.scope),
                ("reason", &s.reason),
                ("ends", &s.ends),
            ] {
                if value.trim().is_empty() {
                    return Err(ModelError::Inconsistent(format!(
                        "{}: a suspension with an empty {field} is a repeal written \
                         carelessly; every field is the claim",
                        s.id
                    )));
                }
            }
            for path in &s.admits {
                if path.starts_with("crates/") || path.starts_with("xtask/") {
                    return Err(ModelError::Inconsistent(format!(
                        "{}: `{path}` is code, and a suspension never admits shipped code or \
                         a gate --- a deferral does not become admissible by being parked \
                         where the rule matters most",
                        s.id
                    )));
                }
            }
        }
        for c in &self.claim {
            // R4: a claim that cannot be broken is not at any honesty level;
            // it is outside the register the three levels partition.
            if c.refuted_by.trim().is_empty() {
                return Err(ModelError::Inconsistent(format!(
                    "{}: a claim with no refutation condition is not falsifiable; every row \
                     states what would refute it",
                    c.id
                )));
            }
            match c.level {
                Level::SomeTrue => {
                    if c.authority.is_none() {
                        return Err(ModelError::Inconsistent(format!(
                            "{}: a some-true claim must name the authority it is \
                             reproduced from",
                            c.id
                        )));
                    }
                }
                Level::Build => {
                    if c.feature.is_none() {
                        return Err(ModelError::Inconsistent(format!(
                            "{}: a build claim must name the Gherkin scenario that \
                             validates it (R9)",
                            c.id
                        )));
                    }
                    if c.authority.is_some() {
                        return Err(ModelError::Inconsistent(format!(
                            "{}: a build claim is evidence, not a reproduction of an \
                             authority; it must not name one",
                            c.id
                        )));
                    }
                }
                Level::Open => {
                    if c.authority.is_some() {
                        return Err(ModelError::Inconsistent(format!(
                            "{}: an open claim is a measurement and cannot cite an \
                             authority for its value",
                            c.id
                        )));
                    }
                }
            }
            // R4: every CP-* claim records its sample size and seed.
            if c.id.starts_with("CP-") && (c.sample_size.is_none() || c.seed.is_none()) {
                return Err(ModelError::Inconsistent(format!(
                    "{}: a statistical claim must record its sample size and seed",
                    c.id
                )));
            }
            // §2: every CG-* claim is open. A fitted exponent is a measurement.
            if c.id.starts_with("CG-") && c.level != Level::Open {
                return Err(ModelError::Inconsistent(format!(
                    "{}: a scaling exponent is measured and reported, never asserted",
                    c.id
                )));
            }
            // §2: there is no CN-* class. Negative testing presupposes inputs
            // the library rejects, and by C6 there are none.
            if c.id.starts_with("CN-") {
                return Err(ModelError::Inconsistent(format!(
                    "{}: there is no CN-* class; the library rejects nothing beyond \
                     non-existence of the requested object, which CS-* covers",
                    c.id
                )));
            }
        }
        Ok(())
    }

    /// Look up a claim by conformance ID.
    pub fn get(&self, id: &str) -> Option<&Claim> {
        self.claim.iter().find(|c| c.id == id)
    }
}

/// `model/ids.toml` --- the conformance ID register (§2.1).
#[derive(Debug, Clone, Deserialize)]
pub struct Ids {
    /// The schema tag.
    pub spec: String,
    /// One row per conformance ID.
    pub id: Vec<IdRow>,
}

/// One registered conformance ID.
#[derive(Debug, Clone, Deserialize)]
pub struct IdRow {
    /// The ID, e.g. `CS-04`.
    pub id: String,
    /// The honesty level of the claim (R4).
    pub level: Level,
    /// The Gherkin suite the scenario belongs to.
    pub suite: String,
    /// What the ID claims.
    pub statement: String,
    /// The observation that would refute the claim.
    ///
    /// The upstream formalization attaches a *Refuted by* line to every result,
    /// and the reason it does is the reason a gate is worth having: a claim
    /// nobody can say how to break is not a claim about the world. Required
    /// rather than optional, because an absent condition would read as "there
    /// is none", and that is the one thing a falsifiable register may not say
    /// by omission.
    pub refuted_by: String,
}

impl Ids {
    /// Look up a row.
    pub fn get(&self, id: &str) -> Option<&IdRow> {
        self.id.iter().find(|r| r.id == id)
    }
}

/// `model/authorities.toml` --- what this repository cites (§12, CM-03).
#[derive(Debug, Clone, Deserialize)]
pub struct Authorities {
    /// The schema tag.
    pub spec: String,
    /// One row per cited authority.
    pub authority: Vec<AuthorityRow>,
}

/// A cited authority. Never re-derived, vendored, or gated on.
#[derive(Debug, Clone, Deserialize)]
pub struct AuthorityRow {
    /// Stable identifier, e.g. `CL-MM01`.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// What a third party needs to find the source.
    pub citation: String,
    /// A checksum over the committed artifact, or `none`.
    pub checksum: String,
    /// Why there is no checksum, when there is none.
    #[serde(default)]
    pub checksum_reason: String,
    /// What the authority says.
    pub statement: String,
    /// The conformance IDs that are evidence this library realizes it.
    #[serde(default)]
    pub realized_by: Vec<String>,
}
