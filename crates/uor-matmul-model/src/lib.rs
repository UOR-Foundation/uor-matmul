//! Typed registries parsed from `model/*.toml`.
//!
//! The model is authored once, as prose in `docs/conceptual-model/` and as
//! typed data here (R10). Constants, tiers, envelopes, oracles, and the claim
//! ledger have exactly one source, and the Rust consts in the shipped crates
//! are generated from it by [`codegen`].
//!
//! This crate is build-time and CI infrastructure. It is not a dependency of
//! any shipped crate, and it may use `std`.

#![deny(missing_docs)]

pub mod codegen;
pub mod derive;
pub mod registry;

pub use registry::{
    Atlas, Authorities, Authority, AuthorityRow, Blocking, Claim, Codebook, ColumnHash, Complete,
    Constants, Element, IdRow, Ids, Instantiation, KernelCapacity, Ledger, Level, Narrow, Oracle,
    Oracles, Suspension, Threshold, Tier, Tiers, Width, Widths,
};

use std::path::{Path, PathBuf};

/// Everything `model/*.toml` says, parsed and cross-checked.
#[derive(Debug, Clone)]
pub struct Model {
    /// `model/constants.toml`.
    pub constants: Constants,
    /// `model/widths.toml`.
    pub widths: Widths,
    /// `model/tiers.toml`.
    pub tiers: Tiers,
    /// `model/oracles.toml`.
    pub oracles: Oracles,
    /// `model/ledger.toml`.
    pub ledger: Ledger,
    /// `model/ids.toml`: the conformance ID register (§2.1).
    pub ids: Ids,
    /// `model/authorities.toml`: what this repository cites (§12).
    pub authorities: Authorities,
}

/// A failure to load or to cross-check the model.
#[derive(Debug)]
pub enum ModelError {
    /// A model file could not be read.
    Io(PathBuf, std::io::Error),
    /// A model file could not be parsed.
    Parse(PathBuf, toml::de::Error),
    /// The model disagrees with itself, or with a derivation (CM-01).
    Inconsistent(String),
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(p, e) => write!(f, "reading {}: {e}", p.display()),
            Self::Parse(p, e) => write!(f, "parsing {}: {e}", p.display()),
            Self::Inconsistent(m) => write!(f, "model is inconsistent: {m}"),
        }
    }
}

impl std::error::Error for ModelError {}

/// The CG-* register rows whose level is `build` rather than `open`.
///
/// §2 forces every CG-* row `open`, because a scaling exponent is a
/// measurement: fitted, reported, never asserted. A census is not a
/// measurement. CG-11's claim is that the static issue analysis runs over the
/// emitted inner loops and names a bottleneck resource for every kernel
/// sequence --- that is constructed here and validated, which is what `build`
/// means. The figures the census prints remain `llvm-mca` scheduling-model
/// predictions: reported, never asserted as measurements, exactly as the row's
/// statement says. The exception lives in the ID register alone; a CG-* claim
/// in the *ledger* is still forced `open` by `Ledger::check`, because a ledger
/// claim about scaling is a measurement and nothing else.
///
/// CG-13's is the same shape of claim about dispatch: that the resolved kernel
/// sequence is cached per element family and that a cached selection returns
/// the sequence the full walk returns is constructed here and asserted, which
/// is `build`. What the cache buys in nanoseconds is a measurement and stays
/// `open` under CG-07, exactly as the row's statement says.
///
/// CG-18's is the same shape again, about selection: that auto-selection is
/// the break-even derivation and that the running gather issues no multiplies
/// beyond the build's is read off the operation census, which is a count and
/// not a clock --- constructed here and asserted, which is `build`. What the
/// boundary buys in nanoseconds stays `open` under CG-10.
///
/// CG-22 likewise asserts a route/census correspondence rather than a clock:
/// the selected float factorization is the model's derivation, and the
/// non-float controls retain their route and counts. The achieved rates remain
/// `open` under CG-21.
const CG_BUILD_ROWS: &[&str] = &["CG-11", "CG-13", "CG-18", "CG-22"];

impl Model {
    /// Load every model file from a `model/` directory.
    pub fn load(dir: &Path) -> Result<Self, ModelError> {
        Ok(Self {
            constants: read(dir, "constants.toml")?,
            widths: read(dir, "widths.toml")?,
            tiers: read(dir, "tiers.toml")?,
            oracles: read(dir, "oracles.toml")?,
            ledger: read(dir, "ledger.toml")?,
            ids: read(dir, "ids.toml")?,
            authorities: read(dir, "authorities.toml")?,
        })
    }

    /// Load the model from the repository root, resolved from this crate's
    /// manifest directory so that it works from any working directory.
    pub fn load_from_repo_root() -> Result<Self, ModelError> {
        Self::load(&repo_root().join("model"))
    }

    /// Cross-check every derivable numeral in the model against the `const fn`
    /// that derives it (`CM-01`, `CM-02`, R1).
    ///
    /// This is the whole of R1's enforcement: a numeral that survives this
    /// check is either derived or explicitly allowlisted, and there is no third
    /// category.
    pub fn check(&self) -> Result<(), ModelError> {
        let bad = |m: String| ModelError::Inconsistent(m);
        let max_k_bits = self.constants.max_k_bits;

        // The Atlas tuple is source data; every other Atlas numeral is a
        // derivation. Scope indexes carrier copies and modality names a value
        // at an ordered grade site, so neither may be folded into the other's
        // dimension without this check noticing.
        let atlas = &self.constants.atlas;
        if atlas.scope == 0 || atlas.modality < 3 || atlas.context < 2 {
            return Err(bad(format!(
                "Atlas dimensions must admit scope, the negative/zero/positive modalities, and a centered context; got ({}, {}, {})",
                atlas.scope, atlas.modality, atlas.context
            )));
        }
        if atlas.source.trim().is_empty() {
            return Err(bad(
                "the Atlas source tuple has no clause provenance".to_string()
            ));
        }
        let carrier = derive::atlas_carrier_dim(atlas.modality, atlas.context);
        let ranks = derive::atlas_projector_ranks(atlas.modality, atlas.context);
        if ranks.into_iter().sum::<u128>() != u128::from(carrier) {
            return Err(bad(format!(
                "Atlas projector ranks {ranks:?} do not exhaust carrier dimension {carrier}"
            )));
        }
        let classes = derive::atlas_class_count(atlas.scope, atlas.modality, atlas.context);
        if classes != u128::from(atlas.scope) * u128::from(carrier) {
            return Err(bad(format!(
                "Atlas class count {classes} is not scope {} times carrier {carrier}",
                atlas.scope
            )));
        }
        let page_sites = derive::atlas_page_sites(atlas.scope, atlas.context);
        if page_sites == 0 {
            return Err(bad(
                "an Atlas address page has no ordered grade sites".to_string()
            ));
        }
        let refinement = derive::atlas_refinement_leaves(atlas.context);
        let refinement_power = atlas.context - 1;
        if refinement.coefficient() != 1 || refinement.power() != refinement_power {
            return Err(bad(format!(
                "Atlas refinement {refinement} is not the exact power 2^{refinement_power}"
            )));
        }
        let alphabet = derive::atlas_alphabet(atlas.scope, atlas.modality, atlas.context);
        if alphabet.coefficient() != classes || alphabet.power() != refinement_power {
            return Err(bad(format!(
                "Atlas alphabet {alphabet} is not class count {classes} times the exact refinement power"
            )));
        }

        // `repr(align(N))` accepts exactly nonzero powers of two through
        // 2^29. The cache line remains an honestly measured blocking value;
        // this check proves only that its generated private layout witness is
        // a total Rust representation on every target.
        let blocking = &self.constants.blocking;
        if !blocking.allowlisted_from_r1 || blocking.reason.trim().is_empty() {
            return Err(bad(
                "the measured cache-shaped constants lack their R1 allowlist provenance"
                    .to_string(),
            ));
        }
        if !blocking.cache_line_bytes.is_power_of_two() {
            return Err(bad(format!(
                "cache line {} is not a nonzero power-of-two Rust representation alignment",
                blocking.cache_line_bytes
            )));
        }
        const RUST_MAX_REPR_ALIGN: usize = 1usize << 29;
        if blocking.cache_line_bytes > RUST_MAX_REPR_ALIGN {
            return Err(bad(format!(
                "cache line {} exceeds Rust's maximum representation alignment {RUST_MAX_REPR_ALIGN}",
                blocking.cache_line_bytes
            )));
        }

        let capacity = &self.constants.kernel_capacity;
        if capacity.max_tile_lanes == 0 {
            return Err(bad(
                "the declared kernel-family maximum has no output cells".to_string(),
            ));
        }
        if capacity.max_source_sites == 0 {
            return Err(bad(
                "the declared kernel-family maximum has no source sites".to_string(),
            ));
        }
        if capacity.source.trim().is_empty() {
            return Err(bad(
                "the declared kernel-family maximum has no derivation provenance".to_string(),
            ));
        }

        // The hash is not an authority: canonical stream equality is. Its
        // prefix is therefore an honestly open work measurement, while the
        // u128 carrier proof is exact geometry over the largest dictionary
        // and canonical index that a 64-bit address space can present.
        let hash = &self.constants.column_hash;
        if hash.level != Level::Open {
            return Err(bad(
                "the column-hash prefix is measured and must remain open".to_string(),
            ));
        }
        if hash.prefix == 0 {
            return Err(bad(
                "the measured column-hash prefix must observe at least one coordinate".to_string(),
            ));
        }
        if hash.source.trim().is_empty() || !hash.source.contains("MEASUREMENT-LOG.md") {
            return Err(bad(
                "the measured column-hash prefix lacks its retained-clock provenance in MEASUREMENT-LOG.md"
                    .to_string(),
            ));
        }
        let hash_bound =
            derive::column_hash_accumulator_bound(max_k_bits, atlas.modality, hash.prefix)
                .ok_or_else(|| {
                    bad(format!(
                "the measured {}-coordinate Atlas column hash does not fit its u128 carrier",
                hash.prefix
            ))
                })?;
        let hash_bits = derive::unsigned_bits(hash_bound);
        if hash.accumulator_bits != hash_bits {
            return Err(bad(format!(
                "the measured {}-coordinate Atlas column hash needs {hash_bits} accumulator bits, model says {}",
                hash.prefix, hash.accumulator_bits
            )));
        }

        // CM-01: every accumulator width is `acc_bits` of its element type.
        for w in &self.widths.width {
            let e = self
                .constants
                .element
                .iter()
                .find(|e| e.name == w.element)
                .ok_or_else(|| bad(format!("widths.toml names unknown element {}", w.element)))?;

            let bits = derive::acc_bits(max_k_bits, e.bits, e.product_terms);
            if bits != w.bits {
                return Err(bad(format!(
                    "acc_bits({}) = {bits}, but widths.toml says {}",
                    e.name, w.bits
                )));
            }
            if derive::accumulator_for(bits) != w.accumulator {
                return Err(bad(format!(
                    "acc_bits({}) = {bits} resolves to {}, but widths.toml says {}",
                    e.name,
                    derive::accumulator_for(bits),
                    w.accumulator
                )));
            }
            if w.acc_bits != (w.bytes as u32).saturating_mul(8) {
                return Err(bad(format!(
                    "{}: acc_bits {} disagrees with bytes {}",
                    e.name, w.acc_bits, w.bytes
                )));
            }
            if w.acc_bits < w.bits {
                return Err(bad(format!(
                    "{}: the accumulator has {} bits but the worst case needs {}",
                    e.name, w.acc_bits, w.bits
                )));
            }
        }

        // `full` is a property of the type: |E::MIN| = 2^(BITS-1).
        for e in &self.constants.element {
            let expect = 1u128 << (e.bits - 1);
            if e.bytes.saturating_mul(8) != e.bits.saturating_mul(e.product_terms) {
                return Err(bad(format!(
                    "{}: {} bytes cannot hold {} components of {} bits",
                    e.name, e.bytes, e.product_terms, e.bits
                )));
            }
            if e.full != expect {
                return Err(bad(format!(
                    "{}: FULL should be 2^{} = {expect}, model says {}",
                    e.name,
                    e.bits - 1,
                    e.full
                )));
            }
        }

        // R1: every narrow-register threshold is floor(cap / per_step).
        for t in &self.constants.narrow.threshold {
            let cap = t.cap_override.unwrap_or(self.constants.narrow.cap_i32);
            let expect = cap / t.per_step;
            if expect != t.k_max {
                return Err(bad(format!(
                    "narrow threshold {}: floor({cap} / {}) = {expect}, model says {}",
                    t.name, t.per_step, t.k_max
                )));
            }
        }

        // Complete accumulators: low limbs cover the product reduction, while
        // the existing tail word covers arbitrary i64 scaling and the two-term
        // Linear expression without changing the associated accumulator type.
        let complete_state = &self.widths.complete_state;
        if complete_state.scalar_bits != derive::integer_scalar_growth_bits(i64::BITS) {
            return Err(bad(format!(
                "complete scalar headroom is {}, but i64 scaling needs {} bits",
                complete_state.scalar_bits,
                derive::integer_scalar_growth_bits(i64::BITS)
            )));
        }
        let terminal_terms_bits = derive::sum_terms_bits(complete_state.terminal_terms);
        if complete_state.terminal_terms_bits != terminal_terms_bits {
            return Err(bad(format!(
                "{} terminal terms need {terminal_terms_bits} bits, model says {}",
                complete_state.terminal_terms, complete_state.terminal_terms_bits
            )));
        }
        if complete_state.extension_bits != i64::BITS {
            return Err(bad(format!(
                "complete extension is {} bits, but its i64 word has {}",
                complete_state.extension_bits,
                i64::BITS
            )));
        }
        if complete_state.nonfinite_flag_count != 3 {
            return Err(bad(format!(
                "a complete accumulator preserves three former non-finite flags, model says {}",
                complete_state.nonfinite_flag_count
            )));
        }
        let nonfinite_states =
            derive::complete_nonfinite_states(complete_state.nonfinite_flag_count);
        if complete_state.nonfinite_states != nonfinite_states {
            return Err(bad(format!(
                "{} complete non-finite flags have {nonfinite_states} nonempty unions, model says {}",
                complete_state.nonfinite_flag_count, complete_state.nonfinite_states
            )));
        }

        // CD-32's f32 table word has one exact compact-or-tagged spelling. The
        // top-positive interval is a consequence of its product and state
        // widths, not a second capacity threshold: values below it are compact
        // coefficients and values inside it are self-describing one-product
        // tokens.
        let q = &self.widths.f32_q_carrier;
        let product_magnitude_bits = q
            .significand_bits
            .checked_mul(2)
            .ok_or_else(|| bad("the f32 q-carrier product width overflows u32".to_string()))?;
        if q.product_magnitude_bits != product_magnitude_bits {
            return Err(bad(format!(
                "two {}-bit coefficients need {product_magnitude_bits} product bits, model says {}",
                q.significand_bits, q.product_magnitude_bits
            )));
        }
        let product_bound = derive::f32_q_product_bound(q.significand_bits)
            .ok_or_else(|| bad("the f32 q-carrier product bound exceeds u128".to_string()))?;
        if u128::from(q.product_bound) != product_bound {
            return Err(bad(format!(
                "the f32 q-carrier product bound is {product_bound}, model says {}",
                q.product_bound
            )));
        }
        let factor_span = q
            .max_factor_exp
            .checked_sub(q.min_factor_exp)
            .ok_or_else(|| bad("the f32 q-carrier factor-grade span underflows".to_string()))?;
        let relative_grade_count = u32::try_from(factor_span)
            .ok()
            .and_then(|span| span.checked_mul(2))
            .and_then(|span| span.checked_add(1))
            .ok_or_else(|| bad("the f32 q-carrier relative-grade count overflows".to_string()))?;
        if q.relative_grade_count != relative_grade_count {
            return Err(bad(format!(
                "the f32 q-carrier has {relative_grade_count} relative product grades, model says {}",
                q.relative_grade_count
            )));
        }
        let signed_finite_states = q
            .relative_grade_count
            .checked_mul(2)
            .ok_or_else(|| bad("the f32 q-carrier finite-state count overflows".to_string()))?;
        if q.signed_finite_states != signed_finite_states {
            return Err(bad(format!(
                "{} grades at two signs make {signed_finite_states} finite states, model says {}",
                q.relative_grade_count, q.signed_finite_states
            )));
        }
        let state_count = q
            .signed_finite_states
            .checked_add(complete_state.nonfinite_states)
            .ok_or_else(|| bad("the f32 q-carrier total state count overflows".to_string()))?;
        if q.state_count != state_count {
            return Err(bad(format!(
                "{} finite states plus {} Complete unions make {state_count} states, model says {}",
                q.signed_finite_states, complete_state.nonfinite_states, q.state_count
            )));
        }
        let state_bits = derive::state_bits(q.state_count);
        if q.state_bits != state_bits {
            return Err(bad(format!(
                "{} f32 q-carrier states need {state_bits} bits, model says {}",
                q.state_count, q.state_bits
            )));
        }
        let tag_payload_bits = q
            .product_magnitude_bits
            .checked_add(q.state_bits)
            .ok_or_else(|| bad("the f32 q-carrier tag payload width overflows".to_string()))?;
        if q.tag_payload_bits != tag_payload_bits {
            return Err(bad(format!(
                "product magnitude plus state needs {tag_payload_bits} tag bits, model says {}",
                q.tag_payload_bits
            )));
        }
        let tag_interval = derive::power_of_two(q.tag_payload_bits)
            .ok_or_else(|| bad("the f32 q-carrier tag interval exceeds u128".to_string()))?;
        if u128::from(q.tag_interval) != tag_interval {
            return Err(bad(format!(
                "the f32 q-carrier tag interval is {tag_interval}, model says {}",
                q.tag_interval
            )));
        }
        let tag_base =
            derive::top_positive_interval_base(complete_state.extension_bits, q.tag_payload_bits)
                .ok_or_else(|| {
                bad("the f32 q-carrier tag interval does not fit positive i64".to_string())
            })?;
        if u128::from(q.tag_base) != tag_base {
            return Err(bad(format!(
                "the f32 q-carrier tag base is {tag_base:#x}, model says {:#x}",
                q.tag_base
            )));
        }
        let compact_ceiling = tag_base - 1;
        if u128::from(q.compact_ceiling) != compact_ceiling {
            return Err(bad(format!(
                "the f32 q-carrier compact ceiling is {compact_ceiling}, model says {}",
                q.compact_ceiling
            )));
        }
        let zero_span_capacity =
            derive::f32_q_lane_capacity(compact_ceiling, product_bound, 0, 0, false);
        if u128::from(q.zero_span_capacity) != zero_span_capacity {
            return Err(bad(format!(
                "the f32 q-carrier zero-span capacity is {zero_span_capacity}, model says {}",
                q.zero_span_capacity
            )));
        }
        for c in &self.widths.complete {
            let span = c
                .max_product_exp
                .checked_sub(c.min_product_exp)
                .ok_or_else(|| bad(format!("{}: product exponent range underflows", c.element)))?;
            if span != c.span_bits {
                return Err(bad(format!(
                    "{}: product exponent span is {span}, model says {}",
                    c.element, c.span_bits
                )));
            }
            let accumulation = c.span_bits + c.guard_bits as i64 + c.sign_bits as i64;
            if accumulation != c.accumulation_bits {
                return Err(bad(format!(
                    "{}: accumulation bits should be {accumulation}, model says {}",
                    c.element, c.accumulation_bits
                )));
            }
            let total = c.accumulation_bits
                + complete_state.scalar_bits as i64
                + complete_state.terminal_terms_bits as i64;
            if total != c.total_bits {
                return Err(bad(format!(
                    "{}: terminal-expression bits should be {total}, model says {}",
                    c.element, c.total_bits
                )));
            }
            let accumulation_bits = u32::try_from(c.accumulation_bits).map_err(|_| {
                bad(format!(
                    "{}: accumulation_bits {} cannot size a finite NAF carrier",
                    c.element, c.accumulation_bits
                ))
            })?;
            let sites = derive::naf_sites(accumulation_bits);
            if sites != c.naf_sites {
                return Err(bad(format!(
                    "{}: {accumulation_bits} accumulation bits needs {sites} NAF sites, model says {}",
                    c.element, c.naf_sites
                )));
            }
            let pages = derive::atlas_pages(sites, page_sites);
            if pages != c.atlas_pages {
                return Err(bad(format!(
                    "{}: {sites} NAF sites at {page_sites} sites per Atlas word needs {pages} pages, model says {}",
                    c.element, c.atlas_pages
                )));
            }
            let limbs = derive::limbs_for(accumulation_bits);
            if limbs != c.limbs {
                return Err(bad(format!(
                    "{}: {} accumulation bits needs {limbs} low limbs, model says {}",
                    c.element, c.accumulation_bits, c.limbs
                )));
            }
            let low_bits = (c.limbs as u32).saturating_mul(u64::BITS);
            let physical_bits = low_bits.saturating_add(complete_state.extension_bits);
            if c.physical_bits != physical_bits {
                return Err(bad(format!(
                    "{}: low limbs plus extension are {physical_bits} bits, model says {}",
                    c.element, c.physical_bits
                )));
            }
            if c.total_bits > i64::from(c.physical_bits) {
                return Err(bad(format!(
                    "{}: {} terminal bits exceed {} physical bits",
                    c.element, c.total_bits, c.physical_bits
                )));
            }
            let finite_state_bits = derive::extension_value_bits(c.total_bits, low_bits);
            if c.finite_state_bits != finite_state_bits {
                return Err(bad(format!(
                    "{}: terminal expression needs {finite_state_bits} signed extension bits, model says {}",
                    c.element, c.finite_state_bits
                )));
            }
            if !derive::sentinels_outside_signed_width(
                finite_state_bits,
                complete_state.extension_bits,
                complete_state.nonfinite_states,
            ) {
                return Err(bad(format!(
                    "{}: a finite terminal expression can alias a non-finite state",
                    c.element
                )));
            }
            let bytes = (c.physical_bits as usize).div_ceil(u8::BITS as usize);
            if c.bytes != bytes {
                return Err(bad(format!(
                    "{}: {} physical bits is {bytes} bytes, model says {}",
                    c.element, c.physical_bits, c.bytes
                )));
            }
        }

        // Exactly one canonical instantiation, and every instantiation's bound
        // is admissible for its element type.
        let canonical: Vec<_> = self
            .constants
            .instantiation
            .iter()
            .filter(|i| i.canonical)
            .collect();
        if canonical.len() != 1 {
            return Err(bad(format!(
                "expected exactly one canonical instantiation, found {}",
                canonical.len()
            )));
        }
        for i in &self.constants.instantiation {
            let e = self
                .constants
                .element
                .iter()
                .find(|e| e.name == i.element)
                .ok_or_else(|| bad(format!("instantiation {} names unknown element", i.name)))?;
            if i.bound > e.full {
                return Err(bad(format!(
                    "instantiation {}: bound {} exceeds {}'s FULL of {}",
                    i.name, i.bound, e.name, e.full
                )));
            }
        }

        // The complete accumulator's guard is the same declared depth the
        // integer accumulators use. If the two ever diverged, a float
        // accumulation and an integer one would be sized against different
        // machines (§3.3).
        for c in &self.widths.complete {
            if c.guard_bits != max_k_bits {
                return Err(bad(format!(
                    "{}: guard_bits is {} but MAX_K_BITS is {max_k_bits}; a float and an \
                     integer accumulation must be sized against the same machine",
                    c.element, c.guard_bits
                )));
            }
        }

        // `CM-04`: every recorded break-even is the derivation, recomputed ---
        // at the row's own pair of sequences, because both sides of the count
        // are ISA declarations and a host is priced at its own.
        for t in &self.tiers.tabulation {
            let Some(table_step) = t.block.checked_mul(t.lanes_per_add) else {
                return Err(bad(format!(
                    "tabulation {} ({}): block {} times lanes_per_add {} exceeds usize",
                    t.codec, t.isa, t.block, t.lanes_per_add
                )));
            };
            let expect = derive::tabulation_break_even(
                t.code_space,
                t.block,
                t.rows,
                table_step,
                t.build_products_per_step,
                t.kernel_products_per_step
                    .unwrap_or(self.constants.blocking.kernel_products_per_step),
                self.constants.blocking.kernel_rows,
            );
            if expect != t.break_even_n {
                return Err(bad(format!(
                    "tabulation {} ({}): break-even of code_space {} over block {} at {} rows \
                     and build density {} is {:?}, but tiers.toml says {:?}",
                    t.codec,
                    t.isa,
                    t.code_space,
                    t.block,
                    t.rows,
                    t.build_products_per_step,
                    expect,
                    t.break_even_n
                )));
            }
            if t.code_space == 0 {
                return Err(bad(format!(
                    "tabulation {}: a codec that enumerates nothing has no table",
                    t.codec
                )));
            }
        }

        self.ledger.check()?;
        self.check_oracle_ledger_agreement()?;
        self.check_ids()?;
        self.check_authorities()?;
        Ok(())
    }

    /// `CM-02`: every registered ID is well formed, and every class obeys the
    /// rules §2 states about it.
    fn check_ids(&self) -> Result<(), ModelError> {
        let bad = |m: String| ModelError::Inconsistent(m);
        let mut seen: Vec<&str> = Vec::new();
        for row in &self.ids.id {
            if seen.contains(&row.id.as_str()) {
                return Err(bad(format!("{}: registered twice", row.id)));
            }
            seen.push(&row.id);

            // §2: there is no CN-* class. Negative testing presupposes inputs
            // the library rejects, and by C6 there are none.
            if row.id.starts_with("CN-") {
                return Err(bad(format!(
                    "{}: there is no CN-* class; the library rejects nothing beyond \
                     non-existence of the requested object, which CS-03 covers",
                    row.id
                )));
            }
            // §2: a fitted exponent is measured and reported, never asserted.
            if row.id.starts_with("CG-")
                && row.level != Level::Open
                && !CG_BUILD_ROWS.contains(&row.id.as_str())
            {
                return Err(bad(format!("{}: a scaling exponent must be open", row.id)));
            }
            // §2, R4: a cross-library *result* is evidence that the kernels
            // realize the identity, not a proof of it. It is never some-true.
            if row.id.starts_with("CX-") && row.level == Level::SomeTrue {
                return Err(bad(format!(
                    "{}: a cross-library result is `build`, never `some-true`",
                    row.id
                )));
            }
            if row.statement.trim().is_empty() {
                return Err(bad(format!(
                    "{}: an untagged claim does not ship (R4)",
                    row.id
                )));
            }
            if row.refuted_by.trim().is_empty() {
                return Err(bad(format!(
                    "{}: a claim with no refutation condition is not falsifiable; every row \
                     states what would refute it",
                    row.id
                )));
            }
            // R10, CM-02: `CONFORMANCE.md` groups the register by class, and
            // the generator skips a class it does not know. Left to the
            // generator, that skip is silent: the row stays gated by every
            // rule here and by the meta-gate, and reaches the published index
            // nowhere. No byte comparison can catch it either, because the
            // rendered bytes and the committed bytes agree on the absence. So
            // the register refuses the row, which is the one place the
            // omission is visible.
            if !codegen::CLASSES.iter().any(|(p, _)| row.id.starts_with(p)) {
                return Err(bad(format!(
                    "{}: `CONFORMANCE.md` renders no class this ID belongs to, so the row \
                     would be gated everywhere and published nowhere; add its class to \
                     `codegen::CLASSES`, or register the ID under a class that exists",
                    row.id
                )));
            }
        }

        // §3.4: the integer oracles agree byte for byte over the whole corpus
        // with no exempted region, so none of them may be `open`.
        for id in ["CX-01", "CX-02", "CX-03", "CX-04", "CX-10"] {
            let row = self
                .ids
                .get(id)
                .ok_or_else(|| bad(format!("{id}: an integer oracle with no register row")))?;
            if row.level != Level::Build {
                return Err(bad(format!(
                    "{id}: an integer oracle is bit-identical everywhere by ring \
                     homomorphism, so its claim is `build` and not a measurement"
                )));
            }
        }
        Ok(())
    }

    /// `CM-03`: every `some-true` claim has a row in `model/authorities.toml`
    /// with a citation, and every authority names IDs that exist.
    fn check_authorities(&self) -> Result<(), ModelError> {
        let bad = |m: String| ModelError::Inconsistent(m);
        for a in &self.authorities.authority {
            if a.citation.trim().is_empty() {
                return Err(bad(format!("{}: an authority with no citation", a.id)));
            }
            if a.checksum == "none" && a.checksum_reason.trim().is_empty() {
                return Err(bad(format!(
                    "{}: no checksum and no reason. A missing checksum must be a stated \
                     fact, not an omission (R11)",
                    a.id
                )));
            }
            for id in &a.realized_by {
                if self.ids.get(id).is_none() {
                    return Err(bad(format!("{}: realized_by names unknown ID {id}", a.id)));
                }
            }
        }
        // Every some-true claim in the ledger names a known authority.
        for c in &self.ledger.claim {
            if c.level != Level::SomeTrue {
                continue;
            }
            let Some(name) = &c.authority else {
                return Err(bad(format!(
                    "{}: a some-true claim must name an authority",
                    c.id
                )));
            };
            if !self.authorities.authority.iter().any(|a| &a.id == name) {
                return Err(bad(format!(
                    "{}: cites {name}, which has no row in model/authorities.toml (CM-03)",
                    c.id
                )));
            }
        }
        Ok(())
    }

    /// Every oracle row's honesty level agrees with its ledger claim, and every
    /// `inexact` oracle is `open` rather than `build` (§3.4, R4).
    fn check_oracle_ledger_agreement(&self) -> Result<(), ModelError> {
        for o in &self.oracles.oracle {
            let claim = self.ids.get(&o.id).ok_or_else(|| {
                ModelError::Inconsistent(format!("oracle {} has no register row", o.id))
            })?;
            if claim.level != o.honesty {
                return Err(ModelError::Inconsistent(format!(
                    "oracle {}: registry says {:?}, ledger says {:?}",
                    o.id, o.honesty, claim.level
                )));
            }
            // §3.4: an oracle that is not exact for a case cannot be expected
            // to return our bytes, so its claim is a measurement, not a result.
            let want = if o.exactness == "exact" {
                Level::Build
            } else {
                Level::Open
            };
            if claim.level != want {
                return Err(ModelError::Inconsistent(format!(
                    "oracle {} is {} so its claim must be {want:?}, not {:?}",
                    o.id, o.exactness, claim.level
                )));
            }
            // R4: a CX-* result is `build`, never `some-true`.
            if claim.level == Level::SomeTrue {
                return Err(ModelError::Inconsistent(format!(
                    "oracle {} is some-true; a cross-library result is evidence that the \
                     kernels realize the identity, not a proof of it",
                    o.id
                )));
            }
        }
        Ok(())
    }
}

fn read<T: serde::de::DeserializeOwned>(dir: &Path, name: &str) -> Result<T, ModelError> {
    let path = dir.join(name);
    let text = std::fs::read_to_string(&path).map_err(|e| ModelError::Io(path.clone(), e))?;
    toml::from_str(&text).map_err(|e| ModelError::Parse(path, e))
}

/// The repository root, resolved from this crate's manifest directory.
pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/uor-matmul-model is two levels below the repository root")
        .to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CM-01: the model is self-consistent and every numeral in it derives.
    #[test]
    fn model_is_consistent_cm_01() {
        let model = Model::load_from_repo_root().expect("model loads");
        model.check().expect("model checks");
    }

    /// The Atlas dimensions and complete-carrier depth are derivations rather
    /// than unchecked numerals in the model.
    #[test]
    fn atlas_and_naf_width_pins_are_falsifiable_cm_01() {
        let mut model = Model::load_from_repo_root().expect("model loads");
        model.check().expect("the committed derivations agree");

        model.widths.complete[0].atlas_pages += 1;
        let e = model
            .check()
            .expect_err("an Atlas page count that is one too large must fail");
        assert!(e.to_string().contains("Atlas word"), "{e}");
        model.widths.complete[0].atlas_pages -= 1;

        // Refinement cardinality stays symbolic.  A context wider than a
        // machine integer therefore remains a valid model instance; only the
        // finite source support determines how many address words it visits.
        model.constants.atlas.context = 512;
        let page_sites =
            derive::atlas_page_sites(model.constants.atlas.scope, model.constants.atlas.context);
        for complete in &mut model.widths.complete {
            complete.atlas_pages = derive::atlas_pages(complete.naf_sites, page_sites);
        }
        model
            .check()
            .expect("symbolic refinement has no machine-word context ceiling");

        model.constants.atlas.context = 0;
        let e = model
            .check()
            .expect_err("a context with no centered block must fail");
        assert!(e.to_string().contains("Atlas dimensions"), "{e}");
    }

    /// The capacity pin owns both generated consumers and refuses an empty or
    /// unproven declaration. Its equality with the live family maximum is the
    /// independent CG-22 differential in `uor-matmul-kernels`.
    #[test]
    fn kernel_capacity_generates_one_complete_dispatch_interval_cm_01() {
        let mut model = Model::load_from_repo_root().expect("model loads");
        model.check().expect("the committed capacity is valid");

        let maximum = model.constants.kernel_capacity.max_tile_lanes;
        let dispatch = codegen::render_atlas_dispatch(&model);
        for capacity in 1..maximum {
            let arm = format!("{capacity} => $execute!({capacity}),");
            assert_eq!(
                dispatch.matches(&arm).count(),
                1,
                "every nonterminal capacity is emitted exactly once"
            );
        }
        assert_eq!(
            dispatch
                .matches("MAX_TILE_LANES => $execute!(MAX_TILE_LANES),")
                .count(),
            1
        );
        assert!(codegen::render_kernel_capacity(&model)
            .contains(&format!("MAX_TILE_LANES: usize = {maximum};")));
        let source_sites = model.constants.kernel_capacity.max_source_sites;
        assert!(codegen::render_kernel_capacity(&model)
            .contains(&format!("MAX_ATLAS_SOURCE_SITES: usize = {source_sites};")));
        assert!(codegen::render_atlas_dispatch(&model)
            .contains(&format!("MAX_ATLAS_SOURCE_SITES: usize = {source_sites};")));

        model.constants.kernel_capacity.max_tile_lanes = 0;
        let error = model
            .check()
            .expect_err("an empty generated interval cannot dispatch a physical tile");
        assert!(error.to_string().contains("no output cells"), "{error}");

        model.constants.kernel_capacity.max_tile_lanes = maximum;
        model.constants.kernel_capacity.max_source_sites = 0;
        let error = model
            .check()
            .expect_err("an empty generated source frame cannot hold a kernel");
        assert!(error.to_string().contains("no source sites"), "{error}");

        model.constants.kernel_capacity.max_source_sites = source_sites;
        model.constants.kernel_capacity.source.clear();
        let error = model
            .check()
            .expect_err("a derived capacity without provenance is not a model value");
        assert!(
            error.to_string().contains("no derivation provenance"),
            "{error}"
        );
    }

    /// The generated kernel artifact may use the measured cache line only as
    /// a private layout alignment. Invalid Rust alignments and missing
    /// measurement provenance each fail before code generation can make them
    /// source literals.
    #[test]
    fn kernel_capacity_generates_model_owned_cache_alignment_cm_01() {
        let model = Model::load_from_repo_root().expect("model loads");
        model.check().expect("the committed cache alignment checks");
        let alignment = model.constants.blocking.cache_line_bytes;
        let generated = codegen::render_kernel_capacity(&model);
        assert_eq!(
            generated
                .matches(&format!("#[repr(align({alignment}))]"))
                .count(),
            1
        );
        assert_eq!(
            generated
                .matches(&format!("CACHE_LINE_BYTES: usize = {alignment};"))
                .count(),
            1
        );
        assert_eq!(
            generated
                .matches("pub(crate) struct CacheAligned<T>(pub(crate) T);")
                .count(),
            1
        );
        assert_eq!(
            generated
                .matches("impl<T> core::ops::Deref for CacheAligned<T>")
                .count(),
            1
        );
        assert!(generated.contains("#[inline(always)]\n    fn deref(&self) -> &Self::Target"));
        assert!(generated.contains("[blocking] cache_line_bytes"));

        for invalid in [0usize, 63] {
            let mut model = Model::load_from_repo_root().expect("model loads");
            model.constants.blocking.cache_line_bytes = invalid;
            let error = model
                .check()
                .expect_err("a zero or non-radix cache alignment must fail");
            assert!(
                error
                    .to_string()
                    .contains("power-of-two Rust representation alignment"),
                "{error}"
            );
        }

        let mut model = Model::load_from_repo_root().expect("model loads");
        model.constants.blocking.cache_line_bytes = 1usize << 30;
        let error = model
            .check()
            .expect_err("an alignment beyond Rust's representation range must fail");
        assert!(
            error
                .to_string()
                .contains("maximum representation alignment"),
            "{error}"
        );

        let mut model = Model::load_from_repo_root().expect("model loads");
        model.constants.blocking.allowlisted_from_r1 = false;
        let error = model
            .check()
            .expect_err("a measured alignment without its allowlist provenance must fail");
        assert!(
            error.to_string().contains("allowlist provenance"),
            "{error}"
        );

        let mut model = Model::load_from_repo_root().expect("model loads");
        model.constants.blocking.reason.clear();
        let error = model
            .check()
            .expect_err("a measured alignment without an allowlist reason must fail");
        assert!(
            error.to_string().contains("allowlist provenance"),
            "{error}"
        );

        let mut model = Model::load_from_repo_root().expect("model loads");
        model.constants.blocking.cache_line_bytes = 128;
        model
            .check()
            .expect("another valid measured alignment is parametric");
        let generated = codegen::render_kernel_capacity(&model);
        assert!(generated.contains("CACHE_LINE_BYTES: usize = 128;"));
        assert!(generated.contains("#[repr(align(128))]"));
    }

    /// CU-11's prefix is an honest measurement, while its unreduced carrier
    /// width is derived. Every field has an independently failing mutation so
    /// neither the open provenance nor the overflow proof can become inert.
    #[test]
    fn measured_column_hash_prefix_is_open_generated_and_falsifiable_cu_11() {
        let mut model = Model::load_from_repo_root().expect("model loads");
        model
            .check()
            .expect("the committed hash measurement checks");
        assert_eq!(model.constants.column_hash.level, Level::Open);
        assert_eq!(model.constants.column_hash.prefix, 16);
        assert_eq!(model.constants.column_hash.accumulator_bits, 90);
        assert!(codegen::render_atlas_dispatch(&model)
            .contains("pub(crate) const COLUMN_HASH_PREFIX: usize = 16;"));

        model.constants.column_hash.level = Level::Build;
        let error = model
            .check()
            .expect_err("a measured prefix cannot claim build honesty");
        assert!(error.to_string().contains("must remain open"), "{error}");

        let mut model = Model::load_from_repo_root().expect("model loads");
        model.constants.column_hash.prefix = 0;
        let error = model
            .check()
            .expect_err("an empty measured prefix cannot filter a coordinate");
        assert!(
            error.to_string().contains("at least one coordinate"),
            "{error}"
        );

        let mut model = Model::load_from_repo_root().expect("model loads");
        model.constants.column_hash.source.clear();
        let error = model
            .check()
            .expect_err("a measured prefix needs retained-clock provenance");
        assert!(error.to_string().contains("provenance"), "{error}");

        let mut model = Model::load_from_repo_root().expect("model loads");
        model.constants.column_hash.accumulator_bits -= 1;
        let error = model
            .check()
            .expect_err("a narrowed recurrence carrier pin must fail");
        assert!(error.to_string().contains("accumulator bits"), "{error}");

        let mut model = Model::load_from_repo_root().expect("model loads");
        model.constants.column_hash.prefix = 64;
        let error = model
            .check()
            .expect_err("a prefix whose exact recurrence exceeds u128 must fail");
        assert!(
            error.to_string().contains("does not fit its u128 carrier"),
            "{error}"
        );
    }

    /// The shared tail preserves the former three independent public flag
    /// fields, so its reserved state count is the number of their nonempty
    /// unions rather than a hand-picked list of canonical IEEE results.
    #[test]
    fn complete_tail_nonfinite_state_space_is_derived_cs_13() {
        let mut model = Model::load_from_repo_root().expect("model loads");
        model
            .check()
            .expect("the committed tail state space is valid");

        let state = &model.widths.complete_state;
        assert_eq!(state.nonfinite_flag_count, 3);
        assert_eq!(
            state.nonfinite_states,
            derive::complete_nonfinite_states(state.nonfinite_flag_count)
        );
        let generated = codegen::render(&model);
        for witness in [
            "const BASE: i64 = i64::MIN;",
            "const NAN_MASK: u8 = 1;",
            "const POS_INF_MASK: u8 = 2;",
            "const NEG_INF_MASK: u8 = 4;",
            "const COUNT: u32 = 7;",
        ] {
            assert!(
                generated.contains(witness),
                "the generated tail omits `{witness}`"
            );
        }

        model.widths.complete_state.nonfinite_states = 3;
        let error = model
            .check()
            .expect_err("the planted canonical-only state count must fail");
        assert!(error.to_string().contains("nonempty unions"), "{error}");

        model.widths.complete_state.nonfinite_states = 7;
        model.widths.complete_state.nonfinite_flag_count = 2;
        let error = model
            .check()
            .expect_err("the planted dropped public flag must fail");
        assert!(
            error
                .to_string()
                .contains("preserves three former non-finite flags"),
            "{error}"
        );
    }

    /// CD-32's carrier constants are consequences of the binary32 coefficient
    /// and grade ranges plus Complete's seven non-finite unions. Each mutation
    /// below leaves a syntactically valid model and must still be rejected.
    #[test]
    fn total_f32_q_carrier_pins_are_falsifiable_cd_32() {
        let mut model = Model::load_from_repo_root().expect("model loads");
        model
            .check()
            .expect("the committed carrier geometry checks");
        let q = &model.widths.f32_q_carrier;
        assert_eq!(q.product_bound, 281_474_943_156_225);
        assert_eq!(q.relative_grade_count, 507);
        assert_eq!(q.signed_finite_states, 1_014);
        assert_eq!(q.state_count, 1_021);
        assert_eq!(q.state_bits, 10);
        assert_eq!(q.tag_payload_bits, 58);
        assert_eq!(q.tag_base, 0x7c00_0000_0000_0000);
        assert_eq!(q.compact_ceiling, q.tag_base - 1);
        assert_eq!(q.zero_span_capacity, 31_744);

        let kernels = codegen::render_kernel_capacity(&model);
        let gemm = codegen::render_atlas_dispatch(&model);
        for witness in [
            "pub(crate) mod f32_q",
            "const PRODUCT_BOUND: u64 = 281474943156225;",
            "const STATE_COUNT: u32 = 1021;",
            "const TAG_BASE: u64 = 8935141660703064064;",
            "const COMPACT_CEILING: u64 = 8935141660703064063;",
            "const ZERO_SPAN_CAPACITY: u64 = 31744;",
            "const MAGNITUDE_RADIX: u64 = 281474976710656;",
            "const SPLIT_STATE: u32 = 1021;",
        ] {
            assert!(kernels.contains(witness), "kernel view lacks `{witness}`");
            assert!(gemm.contains(witness), "gemm view lacks `{witness}`");
        }

        model.widths.f32_q_carrier.product_bound -= 1;
        let error = model
            .check()
            .expect_err("a narrowed coefficient product bound must fail");
        assert!(error.to_string().contains("product bound"), "{error}");

        let mut model = Model::load_from_repo_root().expect("model loads");
        model.widths.f32_q_carrier.state_count -= 1;
        let error = model
            .check()
            .expect_err("dropping one Complete state must fail");
        assert!(error.to_string().contains("Complete unions"), "{error}");

        let mut model = Model::load_from_repo_root().expect("model loads");
        model.widths.f32_q_carrier.tag_base += 1;
        let error = model
            .check()
            .expect_err("moving the top-positive interval must fail");
        assert!(error.to_string().contains("tag base"), "{error}");

        let mut model = Model::load_from_repo_root().expect("model loads");
        model.widths.f32_q_carrier.zero_span_capacity -= 1;
        let error = model
            .check()
            .expect_err("an inexact compact capacity must fail");
        assert!(error.to_string().contains("zero-span capacity"), "{error}");
    }

    /// CM-02: every registered ID is unique and well formed.
    #[test]
    fn the_id_register_is_well_formed_cm_02() {
        let model = Model::load_from_repo_root().expect("model loads");
        model.check().expect("model checks");
        assert!(model.ids.id.len() > 50, "the register is the whole of §2.1");
    }

    /// CM-02: a row whose class `CONFORMANCE.md` does not render is refused
    /// rather than skipped.
    ///
    /// This is the one omission the index cannot report about itself: an
    /// unrendered row leaves the rendered bytes and the committed bytes
    /// equally silent, so `check-model`'s byte comparison passes over it. The
    /// refusal is therefore asserted here, with the committed register as the
    /// control that the rule admits every class actually in use.
    #[test]
    fn a_class_the_index_does_not_render_is_refused_cm_02() {
        let mut model = Model::load_from_repo_root().expect("model loads");
        model
            .check()
            .expect("every committed row names a rendered class");

        let mut planted = model.ids.id[0].clone();
        planted.id = "CZ-01".to_string();
        model.ids.id.push(planted);
        let e = model
            .check()
            .expect_err("a class the index does not render is refused");
        let text = e.to_string();
        assert!(text.contains("CZ-01"), "{text}");
        assert!(text.contains("codegen::CLASSES"), "{text}");
    }

    /// CM-02, R4: a row that names no refutation condition does not ship,
    /// in the ID register and in the ledger alike.
    ///
    /// Whitespace rather than an empty string, because a condition spelled as
    /// a space is exactly the omission that would otherwise read as a value.
    #[test]
    fn a_claim_with_no_refutation_condition_is_refused_cm_02() {
        let mut model = Model::load_from_repo_root().expect("model loads");
        model.check().expect("every committed row states one");

        let restore = model.ids.id[0].refuted_by.clone();
        model.ids.id[0].refuted_by = "   ".to_string();
        let e = model.check().expect_err("an ID row with none is refused");
        assert!(e.to_string().contains("not falsifiable"), "{e}");
        model.ids.id[0].refuted_by = restore;

        model.ledger.claim[0].refuted_by = String::new();
        let e = model
            .check()
            .expect_err("a ledger row with none is refused");
        assert!(e.to_string().contains("not falsifiable"), "{e}");
    }

    /// CM-03: every `some-true` claim cites an authority that exists.
    #[test]
    fn every_some_true_claim_cites_an_authority_cm_03() {
        let model = Model::load_from_repo_root().expect("model loads");
        for c in &model.ledger.claim {
            if c.level == Level::SomeTrue {
                let name = c
                    .authority
                    .as_ref()
                    .expect("a some-true claim names its authority");
                assert!(
                    model.authorities.authority.iter().any(|a| &a.id == name),
                    "{name}"
                );
            }
        }
    }
}
