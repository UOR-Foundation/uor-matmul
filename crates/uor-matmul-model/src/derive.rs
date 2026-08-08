//! The derivations. Every numeral in `model/*.toml` that could be computed is
//! computed here, and [`crate::Model::check`] fails if the model disagrees.
//!
//! These are the host-side twins of the `const fn`s in `uor-matmul-core`. The
//! two must agree, which is what `CM-01` checks; keeping them as separate
//! implementations is deliberate, because a generated constant checked against
//! the generator that produced it checks nothing.

/// Bits sufficient for any accumulation this machine can express:
///
/// ```text
/// sign + log2(max k) + log2(B_a) + log2(B_w) + log2(products per mac)
/// ```
///
/// with `max k` bounded by `usize::MAX / size_of::<E>()`, because `a` and `w`
/// must both exist in memory.
///
/// This is a function of the element type alone. There is no ladder, no policy,
/// no promotion, and no `k_max` in the public API (§3.2).
pub const fn acc_bits(max_k_bits: u32, element_bits: u32, product_terms: u32) -> u32 {
    1 + max_k_bits + 2 * (element_bits - 1) + product_terms.ilog2()
}

/// The unique accumulator type with at least `bits` bits.
///
/// Not a parameter, not a policy, not a ladder: exactly one type per width.
pub fn accumulator_for(bits: u32) -> String {
    if bits <= 128 {
        "i128".to_string()
    } else {
        format!("Limbs<{}>", limbs_for(bits))
    }
}

/// 64-bit limbs sufficient for `bits` bits.
pub const fn limbs_for(bits: u32) -> usize {
    (bits as usize).div_ceil(64)
}

/// Dimension of one Atlas carrier, excluding scope.
///
/// Both factors are model coordinates.  The product is widened before it is
/// formed so the derivation is exact over their complete declared domains.
pub const fn atlas_carrier_dim(modality: u32, context: u32) -> u64 {
    modality as u64 * context as u64
}

/// Number of scoped classes in an Atlas instance.
pub const fn atlas_class_count(scope: u32, modality: u32, context: u32) -> u128 {
    scope as u128 * atlas_carrier_dim(modality, context) as u128
}

/// Ordered grade sites in one address page.
///
/// Modality is the value stored at a site, not another grade axis, so a page is
/// `scope * context` sites rather than `scope * modality * context` sites.
pub const fn atlas_page_sites(scope: u32, context: u32) -> u64 {
    scope as u64 * context as u64
}

/// Independent sign coordinates in the centered context block.
pub const fn atlas_refinement_bits(context: u32) -> u32 {
    context - 1
}

/// An exact finite cardinality represented without materializing `2^power`.
///
/// Atlas refinement is exponential in the context coordinate.  Keeping that
/// exponent symbolic is the mathematical representation: a machine integer
/// is merely one possible rendering of it, and cannot impose a context limit
/// on the model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AtlasCardinality {
    coefficient: u128,
    power: u32,
}

impl AtlasCardinality {
    /// Construct the exact cardinality `coefficient * 2^power`.
    pub const fn new(coefficient: u128, power: u32) -> Self {
        Self { coefficient, power }
    }

    /// The finite coefficient multiplying the power of two.
    pub const fn coefficient(self) -> u128 {
        self.coefficient
    }

    /// The power of two, retained symbolically at every magnitude.
    pub const fn power(self) -> u32 {
        self.power
    }

    /// Materialize the cardinality only when it fits the requested machine
    /// representation.
    pub const fn as_u128(self) -> Option<u128> {
        if self.power >= u128::BITS {
            None
        } else {
            self.coefficient.checked_mul(1u128 << self.power)
        }
    }
}

impl core::fmt::Display for AtlasCardinality {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} * 2^{}", self.coefficient, self.power)
    }
}

/// Refinement leaves in the Atlas alphabet.
pub const fn atlas_refinement_leaves(context: u32) -> AtlasCardinality {
    AtlasCardinality::new(1, atlas_refinement_bits(context))
}

/// Full Atlas alphabet cardinality.
pub const fn atlas_alphabet(scope: u32, modality: u32, context: u32) -> AtlasCardinality {
    AtlasCardinality::new(
        atlas_class_count(scope, modality, context),
        atlas_refinement_bits(context),
    )
}

/// Largest unreduced column-hash coordinate admitted by one measured prefix.
///
/// The unreduced initial coordinate is the complete source length, so it may
/// occupy every one of the `address_bits`; reducing it before the recurrence
/// would add work without changing the final dictionary residue. A canonical
/// index has the same width. Repeating `h' = modality * h + index` derives the
/// exact inclusive maximum; `None` means that maximum cannot be represented in
/// the production `u128` carrier.
pub const fn column_hash_accumulator_bound(
    address_bits: u32,
    modality: u32,
    prefix: usize,
) -> Option<u128> {
    if address_bits < 2 || modality == 0 {
        return None;
    }
    let Some(index_cardinality) = power_of_two(address_bits) else {
        return None;
    };
    let coordinate = index_cardinality - 1;
    let mut bound = coordinate;
    let mut observed = 0usize;
    while observed < prefix {
        let Some(radix_term) = bound.checked_mul(modality as u128) else {
            return None;
        };
        let Some(next) = radix_term.checked_add(coordinate) else {
            return None;
        };
        bound = next;
        observed += 1;
    }
    Some(bound)
}

/// Unsigned bits required to spell `value`, with zero requiring no bits.
pub const fn unsigned_bits(mut value: u128) -> u32 {
    let mut bits = 0u32;
    while value != 0 {
        value /= 2;
        bits += 1;
    }
    bits
}

/// Ranks of the global, modality, context, and interaction projectors.
pub const fn atlas_projector_ranks(modality: u32, context: u32) -> [u128; 4] {
    let modality = modality as u128;
    let context = context as u128;
    [1, modality - 1, context - 1, (modality - 1) * (context - 1)]
}

/// NAF sites sufficient for a signed complete accumulator of `total_bits`.
///
/// A width-`b` signed integer has magnitude below `2^(b-1)` or is exactly
/// `-2^(b-1)`. Its canonical non-adjacent form therefore has no nonzero site
/// above `b-1`, including the carry representation of a positive run of ones.
pub const fn naf_sites(total_bits: u32) -> usize {
    total_bits as usize
}

/// Exponent growth from multiplying by the largest magnitude of a signed
/// scalar with `bits` bits. `i64::MIN` has magnitude `2^63`, hence 63.
pub const fn integer_scalar_growth_bits(bits: u32) -> u32 {
    bits.saturating_sub(1)
}

/// Headroom required to sum `terms` values of the same signed width.
pub const fn sum_terms_bits(terms: u32) -> u32 {
    if terms <= 1 {
        0
    } else {
        u32::BITS - (terms - 1).leading_zeros()
    }
}

/// The nonempty subsets of `flag_count` independent non-finite flags.
///
/// This is multiplication rather than a host bit shift because the derivation
/// is a cardinality, independent of its eventual compact representation.
pub const fn complete_nonfinite_states(flag_count: u32) -> u32 {
    let mut cardinality = 1u32;
    let mut flag = 0u32;
    while flag < flag_count {
        let Some(next) = cardinality.checked_mul(2) else {
            return u32::MAX;
        };
        cardinality = next;
        flag += 1;
    }
    cardinality - 1
}

/// The exact cardinality `2^bits`, or `None` when it does not fit in `u128`.
pub const fn power_of_two(bits: u32) -> Option<u128> {
    let mut value = 1u128;
    let mut bit = 0u32;
    while bit < bits {
        let Some(next) = value.checked_mul(2) else {
            return None;
        };
        value = next;
        bit += 1;
    }
    Some(value)
}

/// Largest product of two `significand_bits`-wide unsigned coefficients.
pub const fn f32_q_product_bound(significand_bits: u32) -> Option<u128> {
    let Some(cardinality) = power_of_two(significand_bits) else {
        return None;
    };
    let maximum = cardinality - 1;
    maximum.checked_mul(maximum)
}

/// Bits required to distinguish `states` states, with zero states using none.
pub const fn state_bits(states: u32) -> u32 {
    let mut bits = 0u32;
    let mut capacity = 1u128;
    while capacity < states as u128 {
        let Some(next) = capacity.checked_mul(2) else {
            return u32::MAX;
        };
        capacity = next;
        bits += 1;
    }
    bits
}

/// First value in a `payload_bits`-wide interval at the top of positive
/// `extension_bits`-wide signed storage.
pub const fn top_positive_interval_base(extension_bits: u32, payload_bits: u32) -> Option<u128> {
    let Some(positive_bits) = extension_bits.checked_sub(1) else {
        return None;
    };
    let Some(positive_end) = power_of_two(positive_bits) else {
        return None;
    };
    let Some(interval) = power_of_two(payload_bits) else {
        return None;
    };
    positive_end.checked_sub(interval)
}

/// Exact run capacity of the compact binary32 product token.
///
/// A non-finite factor is an immediate-placement token. Finite products use
/// `floor(compact_ceiling / (product_bound * 2^(wa + wb)))`; an unrepresentable
/// denominator likewise has no multi-product compact run and therefore returns
/// the total one-product capacity.
pub const fn f32_q_lane_capacity(
    compact_ceiling: u128,
    product_bound: u128,
    wa: u32,
    wb: u32,
    nonfinite: bool,
) -> u128 {
    if nonfinite || product_bound == 0 {
        return 1;
    }
    let Some(width) = wa.checked_add(wb) else {
        return 1;
    };
    let Some(scale) = power_of_two(width) else {
        return 1;
    };
    let Some(denominator) = product_bound.checked_mul(scale) else {
        return 1;
    };
    let capacity = compact_ceiling / denominator;
    if capacity == 0 {
        1
    } else {
        capacity
    }
}

/// Signed bits occupied in the extension word after `low_bits` low sites.
pub const fn extension_value_bits(total_bits: i64, low_bits: u32) -> u32 {
    if total_bits <= low_bits as i64 {
        1
    } else {
        (total_bits - low_bits as i64) as u32
    }
}

/// Whether extreme extension-word sentinels lie strictly outside every value
/// in a derived signed finite width.
pub const fn sentinels_outside_signed_width(
    finite_bits: u32,
    extension_bits: u32,
    sentinel_count: u32,
) -> bool {
    if finite_bits == 0
        || finite_bits > extension_bits
        || extension_bits == 0
        || extension_bits >= i128::BITS
        || sentinel_count == 0
    {
        return false;
    }
    let sentinel_floor = -(1i128 << (extension_bits - 1));
    let sentinel_ceiling = sentinel_floor + sentinel_count as i128 - 1;
    let finite_floor = -(1i128 << (finite_bits - 1));
    finite_floor > sentinel_ceiling
}

/// Address words sufficient for `sites`, with no fixed ceiling.
pub const fn atlas_pages(sites: usize, page_sites: u64) -> usize {
    if sites == 0 {
        0
    } else {
        // The quotient cannot exceed `sites`, so converting only the result
        // preserves every host-addressable source even when one model page is
        // wider than the host's address type.
        (1 + (sites as u128 - 1) / page_sites as u128) as usize
    }
}

#[derive(Clone, Copy)]
enum AtlasCensusFactor {
    Rows,
    Depth,
    Columns,
    PhysicalTile,
}

const ATLAS_CENSUS_WORDS: usize = [
    AtlasCensusFactor::Rows,
    AtlasCensusFactor::Depth,
    AtlasCensusFactor::Columns,
    AtlasCensusFactor::PhysicalTile,
]
.len();
const ATLAS_CENSUS_RADIX: u128 = u64::MAX as u128 + (u64::MAX != u64::MIN) as u128;

const fn atlas_count_from_u128(mut value: u128) -> [u64; ATLAS_CENSUS_WORDS] {
    let mut words = [0; ATLAS_CENSUS_WORDS];
    let mut remaining = ATLAS_CENSUS_WORDS;
    while remaining != 0 {
        remaining -= 1;
        words[remaining] = (value % ATLAS_CENSUS_RADIX) as u64;
        value /= ATLAS_CENSUS_RADIX;
    }
    words
}

const fn atlas_count_add(
    left: [u64; ATLAS_CENSUS_WORDS],
    right: [u64; ATLAS_CENSUS_WORDS],
) -> [u64; ATLAS_CENSUS_WORDS] {
    let mut words = [0; ATLAS_CENSUS_WORDS];
    let mut carry = 0u128;
    let mut remaining = ATLAS_CENSUS_WORDS;
    while remaining != 0 {
        remaining -= 1;
        let sum = left[remaining] as u128 + right[remaining] as u128 + carry;
        words[remaining] = (sum % ATLAS_CENSUS_RADIX) as u64;
        carry = sum / ATLAS_CENSUS_RADIX;
    }
    words
}

const fn atlas_count_multiply(
    value: [u64; ATLAS_CENSUS_WORDS],
    factor: usize,
) -> [u64; ATLAS_CENSUS_WORDS] {
    let mut words = [0; ATLAS_CENSUS_WORDS];
    let mut carry = 0u128;
    let mut remaining = ATLAS_CENSUS_WORDS;
    while remaining != 0 {
        remaining -= 1;
        let product = value[remaining] as u128 * factor as u128 + carry;
        words[remaining] = (product % ATLAS_CENSUS_RADIX) as u64;
        carry = product / ATLAS_CENSUS_RADIX;
    }
    words
}

const fn atlas_count_product(left: usize, right: usize) -> Option<[u64; ATLAS_CENSUS_WORDS]> {
    let Some(product) = (left as u128).checked_mul(right as u128) else {
        return None;
    };
    Some(atlas_count_from_u128(product))
}

const fn atlas_count_coordinates(words: [u64; ATLAS_CENSUS_WORDS]) -> [u128; ATLAS_CENSUS_WORDS] {
    let mut coordinates = [0; ATLAS_CENSUS_WORDS];
    let mut index = 0;
    while index < ATLAS_CENSUS_WORDS {
        coordinates[index] = words[index] as u128;
        index += 1;
    }
    coordinates
}

const fn invalid_atlas_census() -> [[u128; ATLAS_CENSUS_WORDS]; 4] {
    [[u128::MAX; ATLAS_CENSUS_WORDS]; 4]
}

/// Exact candidate census for the one-projection Atlas tile.
///
/// The four rows are projection sites, actual source decodes under the caller
/// offer, issued lookup steps including padded edges, and peak live bytes.
/// Each count is a most-significant-first radix-`2^64` word so all three shape
/// dimensions and the physical-tile factor remain exact on every supported
/// address width. This is the independent model twin of the shipped selector.
#[allow(clippy::too_many_arguments)]
pub const fn atlas_executed_work(
    m: usize,
    k: usize,
    n: usize,
    mr: usize,
    nr: usize,
    products_per_step: usize,
    accumulator_bytes: usize,
    workspace_bytes: usize,
    pa_codes: usize,
    pb_codes: usize,
    max_tile_lanes: usize,
) -> [[u128; ATLAS_CENSUS_WORDS]; 4] {
    let Some(physical_outputs) = mr.checked_mul(nr) else {
        return invalid_atlas_census();
    };
    if physical_outputs == 0
        || physical_outputs > max_tile_lanes
        || products_per_step == 0
        || max_tile_lanes == 0
    {
        return invalid_atlas_census();
    }
    if m == 0 || k == 0 || n == 0 {
        return [[0; ATLAS_CENSUS_WORDS]; 4];
    }

    let a_offer_rows = match pa_codes.checked_div(k) {
        Some(rows) => rows,
        None => 0,
    };
    let offered_b_cols = match pb_codes.checked_div(k) {
        Some(cols) => cols,
        None => 0,
    };
    let b_offer_cols = if offered_b_cols < n {
        offered_b_cols
    } else {
        n
    };
    let streamed_cols = if nr < n { nr } else { n };
    let streamed_cols = if streamed_cols == 0 { 1 } else { streamed_cols };
    let block_width = if b_offer_cols == 0 {
        streamed_cols
    } else {
        b_offer_cols
    };
    let full_blocks = n / block_width;
    let tail_cols = n % block_width;
    let block_count = full_blocks + if tail_cols == 0 { 0 } else { 1 };
    let full_block_tiles = block_width.div_ceil(nr);
    let Some(mut column_tiles) = full_blocks.checked_mul(full_block_tiles) else {
        return invalid_atlas_census();
    };
    if tail_cols != 0 {
        let Some(total) = column_tiles.checked_add(tail_cols.div_ceil(nr)) else {
            return invalid_atlas_census();
        };
        column_tiles = total;
    }
    let row_tiles = m.div_ceil(mr);

    let full_row_tiles = m / mr;
    let tail_rows = m % mr;
    let cached_full_rows = if a_offer_rows < mr { a_offer_rows } else { mr };
    let Some(mut cached_rows) = full_row_tiles.checked_mul(cached_full_rows) else {
        return invalid_atlas_census();
    };
    let cached_tail_rows = if a_offer_rows < tail_rows {
        a_offer_rows
    } else {
        tail_rows
    };
    let Some(total_cached_rows) = cached_rows.checked_add(cached_tail_rows) else {
        return invalid_atlas_census();
    };
    cached_rows = total_cached_rows;

    let Some(decoded_a_cached) = atlas_count_product(block_count, cached_rows) else {
        return invalid_atlas_census();
    };
    let Some(decoded_a_direct) = atlas_count_product(column_tiles, m - cached_rows) else {
        return invalid_atlas_census();
    };
    let decoded_a = atlas_count_add(decoded_a_cached, decoded_a_direct);
    let decoded_b = if b_offer_cols == 0 {
        let Some(value) = atlas_count_product(n, row_tiles) else {
            return invalid_atlas_census();
        };
        value
    } else {
        atlas_count_from_u128(n as u128)
    };
    let decodes = atlas_count_multiply(atlas_count_add(decoded_a, decoded_b), k);
    let projections = decodes;

    let Some(tile_count) = atlas_count_product(row_tiles, column_tiles) else {
        return invalid_atlas_census();
    };
    let issued = atlas_count_multiply(
        atlas_count_multiply(tile_count, physical_outputs.div_ceil(products_per_step)),
        k,
    );

    let live_rows = if mr < m { mr } else { m };
    let live_cols = {
        let cols = if nr < block_width { nr } else { block_width };
        if cols < n {
            cols
        } else {
            n
        }
    };
    let Some(live_cells) = live_rows.checked_mul(live_cols) else {
        return invalid_atlas_census();
    };
    let Some(cell_bytes) = (live_cells as u128).checked_mul(accumulator_bytes as u128) else {
        return invalid_atlas_census();
    };
    let Some(live_bytes) = (workspace_bytes as u128).checked_add(cell_bytes) else {
        return invalid_atlas_census();
    };

    [
        atlas_count_coordinates(projections),
        atlas_count_coordinates(decodes),
        atlas_count_coordinates(issued),
        atlas_count_coordinates(atlas_count_from_u128(live_bytes)),
    ]
}

/// Exact semantic census for a uniform nonzero direct Atlas product.
///
/// Each reduction position is one self-similar coordinate contraction. The
/// tuple is `(coordinate products, kernel calls, factor projections, dyadic
/// placements, terminal encodes)`. Padding is included in coordinate products
/// because it is arithmetic the selected kernel really executes; projections,
/// placements, and encodes name only caller data. `panel_depth` remains part of
/// the public model query because execution panels factor storage, not meaning.
pub const fn atlas_uniform_census(
    m: usize,
    k: usize,
    n: usize,
    mr: usize,
    nr: usize,
    _panel_depth: usize,
) -> [usize; 5] {
    if m == 0 || n == 0 {
        return [0; 5];
    }
    let tiles_m = if mr == 0 { 0 } else { m.div_ceil(mr) };
    let tiles_n = if nr == 0 { 0 } else { n.div_ceil(nr) };
    let tile_count = tiles_m.saturating_mul(tiles_n);
    let coordinate_products = tile_count
        .saturating_mul(mr)
        .saturating_mul(nr)
        .saturating_mul(k);
    let kernel_calls = tile_count.saturating_mul(k);
    let projections = if k == 0 {
        0
    } else {
        tiles_n
            .saturating_mul(m)
            .saturating_mul(k)
            .saturating_add(tiles_m.saturating_mul(n).saturating_mul(k))
    };
    let placements = m.saturating_mul(n).saturating_mul(k);
    let encodes = m.saturating_mul(n);
    [
        coordinate_products,
        kernel_calls,
        projections,
        placements,
        encodes,
    ]
}

/// Exact operation census of the algebra-parametric portable traversal.
///
/// The tuple is `(element products, accumulator combines, terminal encodes)`.
/// `panel == 0` is output-major streaming; a nonzero panel combines one exact
/// partial accumulator per panel. Ring and tropical instantiations must return
/// this same tuple because the traversal cannot inspect which algebra it runs.
pub const fn dense_reference_census(m: usize, k: usize, n: usize, panel: usize) -> [usize; 3] {
    if m == 0 || n == 0 {
        return [0; 3];
    }
    let outputs = m.saturating_mul(n);
    let products = outputs.saturating_mul(k);
    let combines = if panel == 0 || k == 0 {
        0
    } else {
        outputs.saturating_mul(k.div_ceil(panel))
    };
    [products, combines, outputs]
}

/// May this tile be accumulated in a narrower register without changing the
/// answer? (§5.1)
///
/// A `false` selects the wide register. It never selects a different method and
/// never reaches the caller: both sides compute the same integer, so the choice
/// is invisible and has no failure mode. That is what separates an optimization
/// from a fallback (R13).
pub const fn fits_narrow(b: u128, cap: u128, k: u128) -> bool {
    k <= cap / (b * b)
}

/// The largest `k` for which a sequence contributing `per_step` per step stays
/// inside `cap`.
pub const fn threshold(cap: u128, per_step: u128) -> u128 {
    cap / per_step
}

/// The first `n` at which tabulating a `code_space`-wide enumeration of `block`
/// elements issues fewer *instructions* than the dense tile.
///
/// Instructions, not operations, and both sides are declarations the sequences
/// make about themselves. `table_step` is `block * lanes_per_add`: one register
/// of lanes, each carrying a whole codeword. `build_step` is the independent
/// density of the sequence that fills an entry. `kernel_step` is what one dense
/// tile instruction covers, and `kernel_rows` how many rows it needs to issue
/// that many.
///
/// It is not `block * rows`. A tile of `rows` is `rows / lanes_per_add`
/// instructions, so pricing it as one over-states the table by the register
/// count. On an AVX2 host that error cancelled against a `kernel_step` naming
/// `vpdpbusd`, which that host does not have; the two together gave the right
/// number for no reason. Written as two declarations it gives the same number
/// here and a different, correct one wherever the two sides do not scale
/// together.
///
/// ```text
/// tabulated instructions = m*k*S/build_step + m*n*k/table_step
/// dense instructions     = m*k*n/kernel_step
/// ```
///
/// so the table is cheaper exactly when
/// `n * build_step * (table_step - effective) > S * effective * table_step`.
///
/// `None` when no `n` satisfies it: `block == 1` names one element per code, and
/// `table_step <= effective` means one table instruction does not cover what one
/// dense instruction does, so the build is never repaid.
pub const fn tabulation_break_even(
    code_space: usize,
    block: usize,
    rows: usize,
    table_step: usize,
    build_step: usize,
    kernel_step: usize,
    kernel_rows: usize,
) -> Option<usize> {
    if block <= 1
        || rows == 0
        || kernel_step == 0
        || kernel_rows == 0
        || table_step == 0
        || build_step == 0
    {
        return None;
    }
    // A dense tile issues `kernel_step` products per instruction only when it
    // has `kernel_rows` rows to fill; with fewer it pays for the lanes that are
    // not there. Without this term the predicate declined the table at `m = 1`,
    // where the table was three times faster and the tile had one useful row in
    // six.
    let present = if rows < kernel_rows {
        rows
    } else {
        kernel_rows
    };
    let effective = (kernel_step as u128 * present as u128 / kernel_rows as u128) as usize;
    if effective == 0 || table_step <= effective {
        return None;
    }
    let right = CostProduct::of(code_space, effective, table_step);
    if !tabulation_cost_pays(usize::MAX, build_step, table_step - effective, right) {
        return None;
    }
    // Zero cannot beat the nonzero build. Binary search the monotone exact
    // inequality, so a 192-bit numerator never has to be divided by a 128-bit
    // denominator or narrowed through a saturating intermediate.
    let mut below = 0usize;
    let mut at_or_above = usize::MAX;
    while at_or_above - below > 1 {
        let middle = below + (at_or_above - below) / 2;
        if tabulation_cost_pays(middle, build_step, table_step - effective, right) {
            at_or_above = middle;
        } else {
            below = middle;
        }
    }
    Some(at_or_above)
}

const fn tabulation_cost_pays(
    columns: usize,
    build_step: usize,
    advantage: usize,
    right: CostProduct,
) -> bool {
    CostProduct::of(columns, build_step, advantage).greater_than(right)
}

#[derive(Clone, Copy)]
struct CostProduct([u64; 3]);

impl CostProduct {
    const RADIX: u128 = u64::MAX as u128 + 1;

    const fn of(a: usize, b: usize, c: usize) -> Self {
        let factors = [a as u64, b as u64, c as u64];
        let mut limbs = [1u64, 0, 0];
        let mut factor = 0usize;
        while factor < factors.len() {
            let mut carry = 0u128;
            let mut limb = 0usize;
            while limb < limbs.len() {
                let wide = limbs[limb] as u128 * factors[factor] as u128 + carry;
                limbs[limb] = (wide % Self::RADIX) as u64;
                carry = wide / Self::RADIX;
                limb += 1;
            }
            debug_assert!(carry == 0);
            factor += 1;
        }
        Self(limbs)
    }

    const fn greater_than(self, other: Self) -> bool {
        if self.0[2] != other.0[2] {
            self.0[2] > other.0[2]
        } else if self.0[1] != other.0[1] {
            self.0[1] > other.0[1]
        } else {
            self.0[0] > other.0[0]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The W8A8 threshold, as an explicit pin. R1 permits `133144` to appear
    /// exactly here and nowhere else in the workspace.
    #[test]
    fn w8a8_i32_threshold_is_pinned() {
        const CAP_I32: u128 = 2147483647;
        assert_eq!(threshold(CAP_I32, 127 * 127), 133144);
        assert!(fits_narrow(127, CAP_I32, 133144));
        assert!(!fits_narrow(127, CAP_I32, 133145));
    }

    /// §3.2's resolved table, on a 64-bit host.
    #[test]
    fn accumulator_widths_resolve_as_documented() {
        assert_eq!(acc_bits(64, 8, 1), 79);
        assert_eq!(acc_bits(64, 16, 1), 95);
        assert_eq!(acc_bits(64, 32, 1), 127);
        assert_eq!(acc_bits(64, 64, 1), 191);
        // A complex element sums two products per mac, so it gets one more bit.
        assert_eq!(acc_bits(64, 32, 2), 128);
        assert_eq!(acc_bits(64, 64, 2), 192);

        assert_eq!(accumulator_for(acc_bits(64, 8, 1)), "i128");
        assert_eq!(accumulator_for(acc_bits(64, 16, 1)), "i128");
        assert_eq!(accumulator_for(acc_bits(64, 32, 1)), "i128");
        assert_eq!(accumulator_for(acc_bits(64, 64, 1)), "Limbs<3>");
    }

    /// §3.3's complete accumulator widths.
    #[test]
    fn complete_accumulator_widths_resolve_as_documented() {
        assert_eq!(limbs_for(619), 10);
        assert_eq!(limbs_for(4261), 67);
        assert_eq!(naf_sites(619), 619);
        assert_eq!(naf_sites(4261), 4261);
        assert_eq!(atlas_pages(619, 32), 20);
        assert_eq!(atlas_pages(4261, 32), 134);
        assert_eq!(integer_scalar_growth_bits(i64::BITS), 63);
        assert_eq!(sum_terms_bits(2), 1);
        assert_eq!(complete_nonfinite_states(3), 7);
        assert_eq!(extension_value_bits(683, 10 * u64::BITS), 43);
        assert_eq!(extension_value_bits(4325, 67 * u64::BITS), 37);
        assert!(sentinels_outside_signed_width(
            43,
            i64::BITS,
            complete_nonfinite_states(3)
        ));
        assert!(sentinels_outside_signed_width(
            37,
            i64::BITS,
            complete_nonfinite_states(3)
        ));
        assert!(!sentinels_outside_signed_width(
            i64::BITS,
            i64::BITS,
            complete_nonfinite_states(3)
        ));
    }

    /// CD-32's compact/tag boundary is exact integer geometry. The global
    /// width sum is the totality bound; runtime grouping may use the tighter
    /// already-projected per-product capacities without changing these pins.
    #[test]
    fn total_f32_q_carrier_arithmetic_is_exact_cd_32() {
        let product = f32_q_product_bound(24).expect("a binary32 product fits u128");
        assert_eq!(product, 281_474_943_156_225);
        assert_eq!(state_bits(2 * 507 + complete_nonfinite_states(3)), 10);
        let tag_base =
            top_positive_interval_base(i64::BITS, 58).expect("the tag interval fits positive i64");
        assert_eq!(tag_base, 0x7c00_0000_0000_0000);
        let compact_ceiling = tag_base - 1;
        assert_eq!(
            f32_q_lane_capacity(compact_ceiling, product, 0, 0, false),
            31_744
        );
        assert_eq!(
            f32_q_lane_capacity(compact_ceiling, product, 7, 7, false),
            1
        );
        assert_eq!(f32_q_lane_capacity(compact_ceiling, product, 0, 0, true), 1);
        assert_eq!(
            f32_q_lane_capacity(compact_ceiling, product, u32::MAX, 1, false),
            1
        );
    }

    /// The measured 16-coordinate dictionary filter has an exact 90-bit
    /// unreduced carrier on the widest supported address space. This is an
    /// arithmetic proof independent of the clock that selected the prefix.
    #[test]
    fn measured_ternary_column_hash_fits_u128_cu_11() {
        let bound = column_hash_accumulator_bound(64, 3, 16)
            .expect("the measured ternary prefix fits u128 exactly");
        assert_eq!(bound, 1_191_107_759_025_695_718_254_230_815);
        assert_ne!(
            bound, 794_071_836_276_006_466_529_705_247,
            "the former pre-reduced length bound is not the live recurrence"
        );
        assert_eq!(unsigned_bits(bound), 90);
        assert!(bound < power_of_two(90).expect("2^90 fits u128"));
        assert!(column_hash_accumulator_bound(64, 3, 64).is_none());
    }

    /// CD-32's grouping theorem is deliberately qualified: it is the minimum
    /// source-ordered common-boundary partition for the least nonnegative
    /// scalar certificates, not a cancellation-sensitive bin packing or a set
    /// of independent lane schedules.
    #[test]
    fn maximal_prefix_q_groups_equal_exhaustive_ordered_partitions_cd_32() {
        fn greedy(bounds: &[u128], ceiling: u128) -> usize {
            let mut groups = 0usize;
            let mut headroom_used = 0u128;
            for &bound in bounds {
                assert!(
                    bound <= ceiling,
                    "tags are singleton before compact grouping"
                );
                if groups == 0 || bound > ceiling - headroom_used {
                    groups += 1;
                    headroom_used = 0;
                }
                headroom_used += bound;
            }
            groups
        }

        fn exhaustive(bounds: &[u128], ceiling: u128) -> usize {
            if bounds.is_empty() {
                return 0;
            }
            let boundaries = bounds.len() - 1;
            let mut best = usize::MAX;
            for cuts in 0usize..(1usize << boundaries) {
                let mut groups = 1usize;
                let mut sum = 0u128;
                let mut valid = true;
                for (slot, &bound) in bounds.iter().enumerate() {
                    if slot > 0 && cuts & (1usize << (slot - 1)) != 0 {
                        groups += 1;
                        sum = 0;
                    }
                    let Some(next) = sum.checked_add(bound) else {
                        valid = false;
                        break;
                    };
                    if next > ceiling {
                        valid = false;
                        break;
                    }
                    sum = next;
                }
                if valid {
                    best = best.min(groups);
                }
            }
            best
        }

        const Q: u128 = 5;
        for encoded in 0u32..6u32.pow(5) {
            let mut number = encoded;
            let bounds: [u128; 5] = core::array::from_fn(|_| {
                let value = number % 6;
                number /= 6;
                u128::from(value)
            });
            assert_eq!(greedy(&bounds, Q), exhaustive(&bounds, Q), "{bounds:?}");
        }

        // Signed cancellation could fit these in one scalar sum, but the
        // nonnegative certificate schedule correctly claims only two groups.
        assert_eq!(greedy(&[Q, Q], Q), 2);
        assert_eq!(i128::try_from(Q).unwrap() - i128::try_from(Q).unwrap(), 0);

        // Two independent lanes could each keep one common source interval;
        // their least per-slot L-infinity certificates cannot. This is the
        // counterexample that prevents a stronger cross-lane optimality claim.
        let vectors = [[Q, 0u128], [0, Q]];
        let certificates = vectors.map(|slot| slot[0].max(slot[1]));
        assert_eq!(greedy(&certificates, Q), 2);
        assert!((0..2).all(|lane| vectors.iter().map(|slot| slot[lane]).sum::<u128>() <= Q));
    }

    /// `CM-04`: build density is an independent declaration, and the model's
    /// exact three-factor comparison remains total at the address boundary.
    #[test]
    fn tabulation_break_even_prices_the_declared_build_density_cm_04() {
        // AVX2's general four-element lookup build emits one product per step;
        // the bound-one builder emits eight. Equal gather density therefore
        // cannot give them the same boundary.
        assert_eq!(tabulation_break_even(256, 4, 16, 32, 1, 16, 6), Some(8193));
        assert_eq!(tabulation_break_even(256, 4, 16, 32, 8, 16, 6), Some(1025));
        assert_eq!(tabulation_break_even(256, 4, 16, 32, 0, 16, 6), None);

        // `2 * usize::MAX` does not fit an address word and may not saturate
        // into a false tie. Three build products per step put the exact first
        // winner at floor(2*MAX/3)+1.
        let boundary = ((usize::MAX as u128 * 2) / 3 + 1) as usize;
        assert_eq!(
            tabulation_break_even(usize::MAX, 2, 1, 2, 3, 1, 1),
            Some(boundary)
        );
        assert_eq!(tabulation_break_even(usize::MAX, 2, 1, 2, 2, 1, 1), None);
    }

    /// The canonical tuple resolves scope beside the carrier rather than
    /// silently turning the 96 classes into a 96-dimensional carrier.
    #[test]
    fn atlas_dimensions_and_projectors_are_derived_cm_01() {
        assert_eq!(atlas_carrier_dim(3, 8), 24);
        assert_eq!(atlas_class_count(4, 3, 8), 96);
        assert_eq!(atlas_page_sites(4, 8), 32);
        assert_eq!(atlas_refinement_bits(8), 7);
        assert_eq!(atlas_refinement_leaves(8).as_u128(), Some(128));
        assert_eq!(atlas_alphabet(4, 3, 8).as_u128(), Some(12_288));
        assert_eq!(atlas_projector_ranks(3, 8), [1, 2, 7, 14]);
        assert_eq!(atlas_projector_ranks(3, 8).into_iter().sum::<u128>(), 24);

        let unmaterialized = atlas_refinement_leaves(u32::MAX);
        assert_eq!(unmaterialized.coefficient(), 1);
        assert_eq!(unmaterialized.power(), u32::MAX - 1);
        assert_eq!(unmaterialized.as_u128(), None);

        assert_eq!(
            atlas_carrier_dim(u32::MAX, u32::MAX),
            u64::from(u32::MAX) * u64::from(u32::MAX)
        );
        assert_eq!(
            atlas_class_count(u32::MAX, u32::MAX, u32::MAX),
            u128::from(u32::MAX).pow(3)
        );
    }

    #[test]
    fn atlas_route_and_uniform_census_are_derived() {
        let f64_accumulator_bytes = 67 * core::mem::size_of::<u64>() + core::mem::size_of::<i64>();
        let workspace_bytes = 16_384;
        let full = atlas_executed_work(
            6,
            7,
            16,
            6,
            16,
            16,
            f64_accumulator_bytes,
            workspace_bytes,
            0,
            0,
            128,
        );
        assert_eq!(full[0], [0, 0, 0, 154]);
        assert_eq!(full[1], full[0]);
        assert_eq!(full[2], [0, 0, 0, 42]);
        assert_eq!(
            full[3],
            [
                0,
                0,
                0,
                (workspace_bytes + 96 * f64_accumulator_bytes) as u128,
            ]
        );
        let edged = atlas_executed_work(
            7,
            7,
            17,
            6,
            16,
            16,
            f64_accumulator_bytes,
            workspace_bytes,
            0,
            0,
            128,
        );
        assert_eq!(edged[0], [0, 0, 0, 336]);
        assert_eq!(edged[1], edged[0]);
        assert_eq!(edged[2], [0, 0, 0, 168]);
        assert_eq!(
            atlas_executed_work(
                7,
                0,
                17,
                6,
                16,
                16,
                f64_accumulator_bytes,
                workspace_bytes,
                usize::MAX,
                usize::MAX,
                128,
            ),
            [[0; ATLAS_CENSUS_WORDS]; 4]
        );
        assert_eq!(
            atlas_uniform_census(5, 7, 3, 4, 1, 16),
            [168, 42, 147, 105, 15]
        );
        assert_eq!(dense_reference_census(5, 7, 3, 0), [105, 0, 15]);
        assert_eq!(dense_reference_census(5, 7, 3, 3), [105, 45, 15]);
    }
}
