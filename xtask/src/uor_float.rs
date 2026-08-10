//! The live float-route purity and materialization gate.
//!
//! This is deliberately a source call-graph audit rather than a vocabulary
//! grep over the whole repository.  Float decoding and final IEEE encoding are
//! legitimate boundaries; `CU-11` governs the region reachable between them.
//! Starting from every shipped definition of the three stable float entry
//! points makes a legacy helper visible even when a wrapper or generic layer
//! separates it from the public function.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::{Path, PathBuf};

use uor_matmul_model::{codegen, Model};

use crate::Fail;

const ROOTS: &[&str] = &[
    "gemm_float",
    "gemm_float_packed",
    "gemm_float_bridged",
    "gemm_float_ex",
    "gemm_float_ex_full",
    "gemm_float_full",
    "sgemm",
    "dgemm",
    "gemm_tabulated",
    "gemm_tabulated_counted",
];
const ATLAS_REFERENCE_FILE: &str = "crates/uor-matmul-core/src/float_atlas.rs";
const ACCUMULATOR_FILE: &str = "crates/uor-matmul-core/src/acc.rs";
const CORE_GENERATED_FILE: &str = "crates/uor-matmul-core/src/generated.rs";
const ATLAS_ENGINE_FILE: &str = "crates/uor-matmul-gemm/src/float.rs";
const LOOKUP_FILE: &str = "crates/uor-matmul-kernels/src/lookup.rs";
const KERNEL_SPEC_FILE: &str = "crates/uor-matmul-kernels/src/spec.rs";
const KERNEL_CAPACITY_FILE: &str = "crates/uor-matmul-kernels/src/generated_capacity.rs";
const ATLAS_DISPATCH_FILE: &str = "crates/uor-matmul-gemm/src/generated_atlas_dispatch.rs";
const TABLE_FILE: &str = "crates/uor-matmul-kernels/src/table.rs";
const TABULATED_FILE: &str = "crates/uor-matmul-gemm/src/tabulated.rs";
const FLOAT_CORPUS_FILE: &str = "crates/uor-matmul-validate/src/float_corpus.rs";
const FLOAT_SWEEP_FILE: &str = "crates/uor-matmul-validate/tests/uor_float_sweep.rs";
const NATIVE_LOOKUP_BENCH_FILE: &str =
    "crates/uor-matmul-validate/benches/scaling/native_lookup.rs";
const MODEL_DERIVE_FILE: &str = "crates/uor-matmul-model/src/derive.rs";
const JUSTFILE: &str = "Justfile";

/// Names whose semantics are the superseded per-product or integer-multiply
/// routes.  This is a semantic list, not an instruction list: group-level
/// complete dyadic placement in the Atlas engine remains permitted.
const FORBIDDEN_CALLS: &[(&str, &str)] = &[
    (
        "gemm_float_scalar",
        "the scalar significand-product accumulation",
    ),
    ("naf_product", "scalar significand multiplication"),
    ("accumulate_run", "per-product exponent-window placement"),
    ("run_bridge", "the full-operand reification bridge"),
    ("scaled_i32", "a reified scaled integer operand"),
    ("scaled_f64", "a reified scaled float operand"),
    ("gemm_packed", "a traditional integer-multiply route"),
    ("gemm_auto", "a traditional integer-multiply route"),
    ("gemm_auto_counted", "a traditional integer-multiply route"),
    ("gemm_selected", "a traditional integer-multiply route"),
];

/// Types and offers that materialize an operand or place each product into an
/// exponent window.  They are checked only in reachable float functions, so an
/// integer API may keep its own scratch vocabulary without being implicated.
const FORBIDDEN_REACHABLE_TOKENS: &[(&str, &str)] = &[
    ("Window", "per-product exponent-window placement"),
    (
        "suggested_bridge_scaled",
        "an operand-sized reification offer",
    ),
    (
        "count_ones",
        "a scalar packed-support population count selecting coordinate work",
    ),
    (
        "support_mask",
        "a scalar packed-support mask standing in for Atlas coordinates",
    ),
    (
        "factor_support",
        "a scalar packed factor-support route beside direct refinement",
    ),
    (
        "common_factor_gauge",
        "the rejected common-gauge factorization route",
    ),
];

/// An Atlas carrier is borrowed and lazy.  A fixed local representation of one
/// atom is fine; an owned growable collection is not a carrier view.
const OWNED_CARRIER_TOKENS: &[&str] = &[
    "Vec<",
    "Vec :: <",
    "vec![",
    "Box<",
    "String",
    "Cow<",
    "Rc<",
    "Arc<",
    "to_vec(",
    "into_boxed",
    "collect::<Vec",
    "alloc::",
];

/// Stack ownership is ownership too.  These spellings are forbidden in the
/// carrier/projector bodies even though they never reach the allocator counted
/// by the runtime half of CA-05.
const MATERIALIZED_CARRIER_TOKENS: &[&str] = &[
    "[i128;",
    "[i64; ATLAS_CARRIER_DIM]",
    "core::array::from_fn",
    "coordinates:",
];

const REQUIRED_ENGINE: &[&str] = &[
    "accumulate_atlas",
    "accumulate_direct_atlas_tile",
    "atlas_atom",
    "atlas_dot_spec",
    "atlas_executed_work",
    "atlas_tile_spec",
    "atlas_word",
];

const REQUIRED_REFERENCE_TYPES: &[&str] = &[
    "AtlasDigit",
    "FiniteNaf",
    "LaurentDigit",
    "NafDigits",
    "NafAtoms",
    "GradeAddress",
    "AtlasAddress",
    "AtlasCarrier",
    "AtlasBlocks",
    "GradeInterval",
    "MinimumGauges",
];

#[derive(Clone, Debug)]
struct Function {
    name: String,
    rel: String,
    line: usize,
    body: String,
    code: String,
}

#[derive(Default, Debug)]
struct Census {
    roots: usize,
    reachable: usize,
    edges: usize,
    atlas_functions: usize,
    atlas_edges: usize,
    lookups: usize,
    complete_adds: usize,
    dyadic_placements: usize,
    tabulated_functions: usize,
    scaled_lane_lookups: usize,
}

/// `CU-11`, with the static half of `CA-05`: every live float root reaches the
/// Atlas lookup/add engine and no live path reaches a legacy multiplication or
/// full-operand representation.
pub fn audit_uor_float(root: &Path) -> Result<(), Fail> {
    let functions = shipped_functions(root)?;
    let (tabulated_functions, scaled_lane_lookups) = audit_support_files(root, &functions)?;
    let mut result = inspect(&functions)?;
    result.tabulated_functions = tabulated_functions;
    result.scaled_lane_lookups = scaled_lane_lookups;
    println!(
        "audit-uor-float: {} roots, {} reachable functions, {} call edges; {} Atlas \
         functions/{} Atlas edges, {} lookups, {} complete additions, {} dyadic placements; \
         {} tabulated functions, {} scaled-lane lookups \
         (CA-05, CU-11)",
        result.roots,
        result.reachable,
        result.edges,
        result.atlas_functions,
        result.atlas_edges,
        result.lookups,
        result.complete_adds,
        result.dyadic_placements,
        result.tabulated_functions,
        result.scaled_lane_lookups,
    );
    Ok(())
}

/// The scalar Atlas terminus is a lookup through a generated row-address
/// alphabet. Keeping the address projection separate from the table read makes
/// both halves falsifiable: a packed shift/mask or a widening product cannot be
/// hidden inside an otherwise canonical-looking `I8_PRODUCTS[...]` access.
fn is_canonical_i8_product_accessor(function: &Function) -> bool {
    function.rel == LOOKUP_FILE
        && function.name == "i8_products"
        && function.body.split_whitespace().collect::<String>() == "&I8_PRODUCTS"
}

fn audit_i8_product_lookup(functions: &[Function], violations: &mut Vec<String>) {
    let Some(products) = functions
        .iter()
        .find(|function| function.rel == LOOKUP_FILE && function.name == "i8_products")
    else {
        violations.push(format!(
            "`{LOOKUP_FILE}` has no production `i8_products`; the canonical table borrow is unaudited"
        ));
        return;
    };
    if !is_canonical_i8_product_accessor(products) {
        violations.push(format!(
            "{}:{}: `i8_products` does not borrow exactly the canonical aligned product alphabet",
            products.rel, products.line
        ));
    }

    let Some(indexed_lookup) = functions
        .iter()
        .find(|function| function.rel == LOOKUP_FILE && function.name == "i8_product_from")
    else {
        violations.push(format!(
            "`{LOOKUP_FILE}` has no production `i8_product_from`; the borrowed native lookup is unaudited"
        ));
        return;
    };
    if indexed_lookup.body.split_whitespace().collect::<String>()
        != "products[i8_product_address(a,b)asusize]"
    {
        violations.push(format!(
            "{}:{}: `i8_product_from` does not read its borrowed alphabet through the canonical row address",
            indexed_lookup.rel, indexed_lookup.line
        ));
    }

    let Some(lookup) = functions
        .iter()
        .find(|function| function.rel == LOOKUP_FILE && function.name == "i8_product")
    else {
        violations.push(format!(
            "`{LOOKUP_FILE}` has no production `i8_product`; the lookup census is empty"
        ));
        return;
    };
    let lookup_compact = lookup.body.split_whitespace().collect::<String>();
    if !lookup_compact.contains("i8_products()[i8_product_address(a,b)asusize]") {
        violations.push(format!(
            "{}:{}: `i8_product` does not read the canonical table accessor through the row-address alphabet",
            lookup.rel, lookup.line
        ));
    }

    let Some(address) = functions
        .iter()
        .find(|function| function.rel == LOOKUP_FILE && function.name == "i8_product_address")
    else {
        violations.push(format!(
            "`{LOOKUP_FILE}` has no production `i8_product_address`; the lookup address is unaudited"
        ));
        return;
    };
    let address_compact = address.body.split_whitespace().collect::<String>();
    if !address_compact.contains("I8_PRODUCT_ROW_ADDRESSES[aasu8asusize]+basu8asi32") {
        violations.push(format!(
            "{}:{}: the signed-octet pair is not projected by generated row address plus coordinate",
            address.rel, address.line
        ));
    }
    for (function, label) in [
        (lookup, "lookup"),
        (indexed_lookup, "borrowed lookup"),
        (address, "address projection"),
    ] {
        let compact = function.body.split_whitespace().collect::<String>();
        if compact.contains("wrapping_mul")
            || compact.contains("checked_mul")
            || compact.contains("saturating_mul")
            || compact.contains("<<")
            || compact.contains(">>")
            || compact.contains('&')
            || contains_call(&function.body, "naf_product")
        {
            violations.push(format!(
                "{}:{}: the runtime product {label} contains multiply or packed bit-address arithmetic",
                function.rel, function.line
            ));
        }
    }
}

const I8_PRODUCTS_ELF_SYMBOL: &str = "__uor_matmul_kernels_v0_1_0_i8_products";

/// Linux x86-64 binds the one shared alphabet locally so every inlined lookup
/// can use a direct RIP-relative address. Other targets retain Rust's ordinary
/// private symbol because both the export name and ELF directive are cfg-local.
fn audit_i8_product_elf_visibility(source: &str, violations: &mut Vec<String>) {
    let compact = source.split_whitespace().collect::<String>();
    let export = format!(
        "#[cfg_attr(all(target_arch=\"x86_64\",target_os=\"linux\"),unsafe(export_name=\"{I8_PRODUCTS_ELF_SYMBOL}\"))]staticI8_PRODUCTS:"
    );
    let hidden = format!(
        "#[cfg(all(target_arch=\"x86_64\",target_os=\"linux\"))]core::arch::global_asm!(\".hidden{I8_PRODUCTS_ELF_SYMBOL}\");"
    );
    if !compact.contains(&export)
        || !compact.contains(&hidden)
        || source.matches(I8_PRODUCTS_ELF_SYMBOL).count() != 2
    {
        violations.push(format!(
            "`{LOOKUP_FILE}` does not give the one Linux x86-64 product alphabet the exact hidden ELF symbol `{I8_PRODUCTS_ELF_SYMBOL}`"
        ));
    }
}

/// The native seam may only reify the hidden table's address. The fallback is
/// the ordinary inline borrow, so non-Linux x86 keeps identical semantics.
fn audit_i8_product_native_address_seam(
    source: &str,
    _functions: &[Function],
    violations: &mut Vec<String>,
) {
    let source_compact = source.split_whitespace().collect::<String>();
    let exact_asm = "core::arch::asm!(\"lea{address},[rip+{table}]\",address=out(reg)address,table=symI8_PRODUCTS,options(nostack,readonly,preserves_flags));";
    let cfg_linux = "#[cfg(all(target_arch=\"x86_64\",target_os=\"linux\"))]#[inline(always)]pub(crate)fni8_products_native()->&'static[i32;I8_PRODUCT_ENTRIES]{";
    let cfg_fallback = "#[cfg(all(target_arch=\"x86_64\",not(target_os=\"linux\")))]#[inline(always)]pub(crate)fni8_products_native()->&'static[i32;I8_PRODUCT_ENTRIES]{i8_products()}";
    let linux_at = source_compact.find(cfg_linux);
    let fallback_at = source_compact.find(cfg_fallback);
    let linux_body = linux_at
        .zip(fallback_at)
        .filter(|(linux, fallback)| linux < fallback)
        .map(|(linux, fallback)| &source_compact[linux..fallback]);
    let computational = linux_body.is_some_and(|body| {
        [
            "<<",
            ">>",
            "wrapping_mul",
            "i8_product_address",
            "_mm",
            "gather",
        ]
        .iter()
        .any(|token| body.contains(token))
    });
    if linux_body.is_none_or(|body| {
        !body.contains("letaddress:*constCacheAligned<[i32;I8_PRODUCT_ENTRIES]>;")
            || !body.contains(exact_asm)
            || !body.contains("&(*address).0}")
    }) || source.matches("core::arch::asm!(").count() != 1
        || !source_compact.contains(cfg_fallback)
        || computational
    {
        violations.push(format!(
            "`{LOOKUP_FILE}` native product address seam is not exactly one address-only RIP-relative LEA plus the regular non-Linux x86 borrow"
        ));
    }
}

fn audit_x86_native_product_borrows(functions: &[Function], violations: &mut Vec<String>) {
    const X86_FILE: &str = "crates/uor-matmul-kernels/src/isa/x86.rs";
    for name in [
        "avx512_lookup_i8",
        "avx2_lookup_reduce_i8",
        "avx2_table_build_lookup",
        "a5_build_lookup8",
    ] {
        let Some(function) = functions
            .iter()
            .find(|function| function.rel == X86_FILE && function.name == name)
        else {
            violations.push(format!("`{X86_FILE}` has no native lookup `{name}`"));
            continue;
        };
        let compact = function.body.split_whitespace().collect::<String>();
        let binding_at = function.body.find("let product_alphabet");
        let first_loop = [function.body.find("for "), function.body.find("while ")]
            .into_iter()
            .flatten()
            .min();
        if compact
            .matches("crate::lookup::i8_products_native()")
            .count()
            != 1
            || !compact.contains("letproduct_alphabet=crate::lookup::i8_products_native();")
            || first_loop
                .is_some_and(|loop_at| binding_at.is_none_or(|borrow_at| borrow_at > loop_at))
        {
            violations.push(format!(
                "{}:{}: `{name}` does not bind the native alphabet exactly once before its first loop",
                function.rel, function.line
            ));
        }
    }
}

/// The one-row AVX2 reduction presents both adjacent native coordinates to one
/// ordered additive chain. The second half is load-bearing: omitting it or
/// presenting the first half twice changes every complete 16-depth object even
/// though an eight-depth-only alphabet census would still pass.
fn audit_avx2_lookup_reduce_pairing(functions: &[Function], violations: &mut Vec<String>) {
    const X86_FILE: &str = "crates/uor-matmul-kernels/src/isa/x86.rs";
    let Some(octet) = functions
        .iter()
        .find(|function| function.rel == X86_FILE && function.name == "avx2_lookup_reduce_octet")
    else {
        violations.push(format!(
            "`{X86_FILE}` has no single native-octet reduction recurrence"
        ));
        return;
    };
    let octet = octet.body.split_whitespace().collect::<String>();
    for witness in [
        "letmutaddresses=_mm256_cvtepu8_epi32(left_octets);",
        "letright_coordinates=_mm256_cvtepu8_epi32(right_octets);",
        "whiledigit<u8::BITS{addresses=_mm256_add_epi32(addresses,addresses);digit+=1;}",
        "letindices=_mm256_add_epi32(addresses,right_coordinates);",
        "_mm256_i32gather_epi32(product_alphabet.as_ptr(),indices,4)",
    ] {
        if !octet.contains(witness) {
            violations.push(format!(
                "`{X86_FILE}` native-octet reduction recurrence lacks `{witness}`"
            ));
        }
    }
    if ["<<", ">>", "wrapping_mul", "i8_product("]
        .iter()
        .any(|token| octet.contains(token))
    {
        violations.push(format!(
            "`{X86_FILE}` native-octet reduction recurrence contains a packed-bit or alternate product operation"
        ));
    }

    let Some(reduce) = functions
        .iter()
        .find(|function| function.rel == X86_FILE && function.name == "avx2_lookup_reduce_i8")
    else {
        violations.push(format!("`{X86_FILE}` has no AVX2 lookup reduction"));
        return;
    };
    let reduce = reduce.body.split_whitespace().collect::<String>();
    for witness in [
        "constNATIVE_LANES:usize=core::mem::size_of::<__m256i>()/core::mem::size_of::<i32>();",
        "constPAIRED_DEPTH:usize=2*NATIVE_LANES;",
        "letmutsum=_mm256_setzero_si256();",
        "letpaired_end=kc-kc%PAIRED_DEPTH;",
        "whilep!=paired_end{",
        "letproducts0=unsafe{avx2_lookup_reduce_octet(product_alphabet,left_octets0,right_octets0)};",
        "sum=_mm256_add_epi32(sum,products0);",
        "letproducts1=unsafe{avx2_lookup_reduce_octet(product_alphabet,left_octets1,right_octets1)};",
        "sum=_mm256_add_epi32(sum,products1);",
        "p+=PAIRED_DEPTH;",
        "letvector_end=kc-kc%NATIVE_LANES;",
        "ifp<vector_end{",
        "sum=_mm256_add_epi32(sum,products);",
    ] {
        if !reduce.contains(witness) {
            violations.push(format!(
                "`{X86_FILE}` AVX2 lookup reduction does not preserve both ordered paired halves: missing `{witness}`"
            ));
        }
    }
    let second_load = reduce.find("let(left_octets1,right_octets1)=unsafe{");
    let advance = reduce.find("p+=PAIRED_DEPTH;");
    let second_refinement = reduce.find("letproducts1=unsafe{");
    let cursor_is_early = second_load
        .zip(advance)
        .zip(second_refinement)
        .is_some_and(|((load, advance), refinement)| load < advance && advance < refinement);
    if !cursor_is_early || reduce.contains("sum0") || reduce.contains("sum1") {
        violations.push(format!(
            "`{X86_FILE}` AVX2 lookup reduction does not advance its byte cursor between the second-half load and refinement on one accumulator"
        ));
    }
    if reduce.matches("avx2_lookup_reduce_octet(").count() != 3 {
        violations.push(format!(
            "`{X86_FILE}` AVX2 lookup reduction does not use one identical native-octet recurrence for both paired halves and the terminal vector"
        ));
    }
}

/// The retained and resolved native clocks differ only at the function pointer
/// inside an otherwise identical spec. A direct raw call would omit safe-wrapper
/// extent and k-group checks from one side of the ratio and measure the harness
/// asymmetry rather than the address factorization.
fn audit_native_lookup_clock_wrappers(source: &str, violations: &mut Vec<String>) {
    let functions = extract_functions(source, NATIVE_LOOKUP_BENCH_FILE);
    for (name, exact_body) in [
        ("clock_kernel", "spec.mac_tile(kc,pa,pb,acc);"),
        ("clock_table", "spec.build(space,block,book,acts,out);"),
    ] {
        let Some(function) = functions.iter().find(|function| function.name == name) else {
            violations.push(format!(
                "`{NATIVE_LOOKUP_BENCH_FILE}` has no shared `{name}` measurement wrapper"
            ));
            continue;
        };
        if function.body.split_whitespace().collect::<String>() != exact_body {
            violations.push(format!(
                "`{NATIVE_LOOKUP_BENCH_FILE}` `{name}` is not exactly the common safe spec wrapper"
            ));
        }
    }

    let compact = source.split_whitespace().collect::<String>();
    let kernel_pair = "|output|clock_kernel(&raw_spec,kc,&pa,&pb,output),|output|clock_kernel(&spec,kc,&pa,&pb,output),";
    let table_pair = "|output|clock_table(&raw_spec,space,block,&book,&acts,output),|output|clock_table(&spec,space,block,&book,&acts,output),";
    if compact.matches(kernel_pair).count() != 2
        || compact.matches(table_pair).count() != 1
        || compact.matches(".mac_tile(").count() != 1
        || compact.matches(".build(").count() != 1
        || !compact.contains("#[inline(never)]fnclock_kernel(")
        || !compact.contains("#[inline(never)]fnclock_table(")
        || !compact.contains("KernelSpec{mac_tile,..spec}")
        || !compact.contains("TableSpec{build:raw_table_build,..spec}")
        || compact.contains("raw_tile_dispatch")
        || compact.contains("raw_reduce_dispatch")
    {
        violations.push(format!(
            "`{NATIVE_LOOKUP_BENCH_FILE}` acceptance closures do not traverse identical safe wrappers around raw and resolved function pointers"
        ));
    }
}

/// CG-23 keeps a clock at its honest R4 level. Only bodies changed by the
/// native lookup refactor retain the preregistered superiority decision;
/// compiler-linked static equivalents are complete open observations. The
/// classification is closed and source-visible so a changed case cannot be
/// moved behind the reporting-only arm by an innocent-looking benchmark edit.
fn audit_native_lookup_acceptance_protocol(source: &str, violations: &mut Vec<String>) {
    let compact = source.split_whitespace().collect::<String>();
    let exact_classification = "constACCEPTANCE_CASES:[(AcceptanceCase,AcceptanceClass);7]=[(AcceptanceCase::Tile{rows:1,columns:8,},AcceptanceClass::Changed,),(AcceptanceCase::Tile{rows:6,columns:8,},AcceptanceClass::Changed,),(AcceptanceCase::Reduction{rows:4},AcceptanceClass::Changed,),(AcceptanceCase::Table,AcceptanceClass::Changed),(AcceptanceCase::Reduction{rows:1},AcceptanceClass::StaticEquivalent,),(AcceptanceCase::Tile{rows:1,columns:16,},AcceptanceClass::StaticEquivalent,),(AcceptanceCase::Tile{rows:6,columns:16,},AcceptanceClass::StaticEquivalent,),];";
    if !compact.contains(exact_classification)
        || !compact.contains("enumAcceptanceClass{Changed,StaticEquivalent,}")
        || !compact.contains(
            "AcceptanceClass::Changed=>\"changed\",AcceptanceClass::StaticEquivalent=>\"open/static-control\"",
        )
    {
        violations.push(format!(
            "`{NATIVE_LOOKUP_BENCH_FILE}` does not contain CG-23's exact closed changed/static-control classification"
        ));
    }

    let functions = extract_functions(source, NATIVE_LOOKUP_BENCH_FILE);
    let Some(measurement) = functions
        .iter()
        .find(|function| function.name == "measure_paired_acceptance")
    else {
        violations.push(format!(
            "`{NATIVE_LOOKUP_BENCH_FILE}` has no CG-23 paired measurement protocol"
        ));
        return;
    };
    let measurement = measurement.body.split_whitespace().collect::<String>();
    for witness in [
        "letclass=acceptance_class(case);",
        "letmutpaired_log_ratios=[0.0f64;ACCEPTANCE_SAMPLES];",
        "ifclass==AcceptanceClass::Changed{assert!(upper_95<=1.0,",
    ] {
        if !measurement.contains(witness) {
            violations.push(format!(
                "`{NATIVE_LOOKUP_BENCH_FILE}` CG-23 paired measurement lacks `{witness}`"
            ));
        }
    }
    if measurement.matches("upper_95<=1.0").count() != 1
        || measurement.contains("AcceptanceClass::StaticEquivalent{assert!")
        || compact.contains("assert_paired_non_regression")
        || !compact.contains("class={class_label}{case_label}:samples={ACCEPTANCE_SAMPLES}")
    {
        violations.push(format!(
            "`{NATIVE_LOOKUP_BENCH_FILE}` does not isolate the hard timing decision to CG-23 changed cases"
        ));
    }

    for case in [
        "AcceptanceCase::Tile{rows:spec.mr,columns:spec.nr,}",
        "AcceptanceCase::Reduction{rows:spec.mr}",
        "AcceptanceCase::Table",
    ] {
        if !compact.contains(&format!("measure_paired_acceptance({case},")) {
            violations.push(format!(
                "`{NATIVE_LOOKUP_BENCH_FILE}` does not route `{case}` through CG-23's classified paired measurement"
            ));
        }
    }
}

/// The native nibble projectors borrow one complete generated row selected by
/// the signed-octet code. The two-dimensional table makes the safe index itself
/// the bounds witness; flattening, masking, wrapping, or fixing a row would
/// change the projector alphabet before any native lane observes it.
fn audit_i8_nibble_lookup(functions: &[Function], violations: &mut Vec<String>) {
    let Some(lookup) = functions
        .iter()
        .find(|function| function.rel == LOOKUP_FILE && function.name == "i8_nibble_products")
    else {
        violations.push(format!(
            "`{LOOKUP_FILE}` has no production `i8_nibble_products`; native projector \
             termination is unaudited"
        ));
        return;
    };
    let compact = lookup.body.split_whitespace().collect::<String>();
    if compact != "&I8_NIBBLE_PRODUCTS[aasu8asusize]" {
        violations.push(format!(
            "{}:{}: `i8_nibble_products` does not borrow exactly the canonical safe row indexed by its signed-octet code",
            lookup.rel, lookup.line
        ));
    }
}

fn audit_support_files(root: &Path, functions: &[Function]) -> Result<(usize, usize), Fail> {
    let reference_path = root.join(ATLAS_REFERENCE_FILE);
    let reference_raw = std::fs::read_to_string(&reference_path).map_err(|error| {
        format!(
            "CU-11 cannot read its finite Atlas reference `{}`: {error}",
            reference_path.display()
        )
    })?;
    let reference = mask_comments_strings_and_tests(&reference_raw);
    let engine_path = root.join(ATLAS_ENGINE_FILE);
    let engine_raw = std::fs::read_to_string(&engine_path).map_err(|error| {
        format!(
            "CU-11 cannot read its live Atlas engine `{}`: {error}",
            engine_path.display()
        )
    })?;
    let engine = mask_comments_strings_and_tests(&engine_raw);
    let dispatch_raw =
        std::fs::read_to_string(root.join(ATLAS_DISPATCH_FILE)).map_err(|error| {
            format!("CU-11 cannot read generated Atlas dispatcher `{ATLAS_DISPATCH_FILE}`: {error}")
        })?;
    let dispatch = mask_comments_strings_and_tests(&dispatch_raw);
    let mut violations = Vec::new();
    for stale in [
        "struct AtlasScale",
        "admits_atlas_scale",
        "project_common_grade_f32",
    ] {
        if engine.contains(stale) {
            violations.push(format!(
                "`{ATLAS_ENGINE_FILE}` retains superseded pre-q carrier `{stale}`"
            ));
        }
    }
    audit_float_radix_sources(
        &reference_raw,
        &std::fs::read_to_string(root.join(ACCUMULATOR_FILE))?,
        &mut violations,
    );
    audit_complete_tail_compatibility(root, &mut violations)?;
    let model_derive = std::fs::read_to_string(root.join(MODEL_DERIVE_FILE))?;
    audit_column_hash_model_source(&model_derive, &mut violations);
    audit_candidate_performance_measurement(root, &engine_raw, &mut violations)?;
    let (tabulated_functions, scaled_lane_lookups) =
        audit_tabulated_float(root, functions, &mut violations)?;
    for name in REQUIRED_REFERENCE_TYPES {
        if !reference.contains(&format!("struct {name}"))
            && !reference.contains(&format!("enum {name}"))
        {
            violations.push(format!(
                "`{ATLAS_REFERENCE_FILE}` has no production `{name}` declaration; the \
                 NAF/address/projector/gauge foundation is incomplete"
            ));
        }
    }
    audit_borrowed_carrier_declarations(&reference, &mut violations);
    for token in OWNED_CARRIER_TOKENS {
        if reference.contains(token) {
            violations.push(format!(
                "`{ATLAS_REFERENCE_FILE}` contains owned-carrier token `{token}`; CA-05 \
                 requires a fixed atom or borrowed view, never a materialized operand"
            ));
        }
    }

    audit_i8_product_lookup(functions, &mut violations);
    let lookup_source = std::fs::read_to_string(root.join(LOOKUP_FILE))?;
    audit_i8_product_elf_visibility(&lookup_source, &mut violations);
    audit_i8_product_native_address_seam(&lookup_source, functions, &mut violations);
    audit_x86_native_product_borrows(functions, &mut violations);
    audit_avx2_lookup_reduce_pairing(functions, &mut violations);
    let native_lookup_clock = std::fs::read_to_string(root.join(NATIVE_LOOKUP_BENCH_FILE))?;
    audit_native_lookup_clock_wrappers(&native_lookup_clock, &mut violations);
    audit_native_lookup_acceptance_protocol(&native_lookup_clock, &mut violations);
    audit_i8_nibble_lookup(functions, &mut violations);

    for (selector, family, portable) in [(
        "resolve_atlas_dot_spec",
        "available_reduce_i8",
        &["R1_I8_I32", "R_I8_I32"][..],
    )] {
        let Some(selector_fn) = functions
            .iter()
            .find(|function| function.rel == ATLAS_ENGINE_FILE && function.name == selector)
        else {
            violations.push(format!(
                "`{ATLAS_ENGINE_FILE}` has no `{selector}`; `{family}` is not bound to the Atlas"
            ));
            continue;
        };
        let compact = selector_fn.body.split_whitespace().collect::<String>();
        if !contains_call(&selector_fn.body, family)
            || !selector_fn.body.contains(".filter(")
            || !compact.contains("spec.k_group==1")
            || !contains_call(&selector_fn.body, "choose_for_rows")
        {
            violations.push(format!(
                "{}:{}: `{selector}` is not the audited group-one selection over `{family}`",
                selector_fn.rel, selector_fn.line
            ));
        }
        audit_group_one_family(root, functions, family, portable, &mut violations)?;
    }
    audit_atlas_dot_selector_cache(&engine, &engine_raw, functions, &mut violations);

    let tile_families = [
        ("available_i8", &["I8_I32"][..]),
        ("available_i8_narrow", &[][..]),
        ("available_reduce_i8", &["R1_I8_I32", "R_I8_I32"][..]),
    ];
    audit_atlas_tile_selector(&engine, functions, &mut violations);
    let test_functions = extract_functions_including_tests(&engine_raw, ATLAS_ENGINE_FILE);
    audit_model_storage_differential(&test_functions, &mut violations);
    for (family, portable) in tile_families {
        audit_group_one_family(root, functions, family, portable, &mut violations)?;
    }

    let max_tile_lanes = std::fs::read_to_string(root.join(KERNEL_CAPACITY_FILE))
        .ok()
        .and_then(|source| constant_initializer(&source, "MAX_TILE_LANES").map(str::to_string))
        .and_then(|initializer| parse_usize_product(&initializer));
    let max_tile_lanes = max_tile_lanes.unwrap_or_else(|| {
        violations.push(format!(
            "`{KERNEL_CAPACITY_FILE}` has no generated `MAX_TILE_LANES`; exact frame dispatch is \
             unaudited"
        ));
        0
    });
    let max_source_sites = std::fs::read_to_string(root.join(KERNEL_CAPACITY_FILE))
        .ok()
        .and_then(|source| {
            constant_initializer(&source, "MAX_ATLAS_SOURCE_SITES").map(str::to_string)
        })
        .and_then(|initializer| parse_usize_product(&initializer));
    let max_source_sites = max_source_sites.unwrap_or_else(|| {
        violations.push(format!(
            "`{KERNEL_CAPACITY_FILE}` has no generated `MAX_ATLAS_SOURCE_SITES`; the exact source workspace is unaudited"
        ));
        0
    });
    audit_generated_kernel_capacity(root, max_tile_lanes, max_source_sites, &mut violations)?;
    audit_panel_execution_storage(
        &engine,
        &dispatch,
        functions,
        max_tile_lanes,
        max_source_sites,
        &mut violations,
    );

    if !violations.is_empty() {
        return Err(format!(
            "CA-05, CU-11 support audit failed before live reachability:\n\n{}",
            violations.join("\n")
        )
        .into());
    }
    if tabulated_functions == 0 || scaled_lane_lookups == 0 {
        return Err("CU-11's tabulated float census is empty".into());
    }
    Ok((tabulated_functions, scaled_lane_lookups))
}

/// The IEEE boundary may name a binary radix, but the finite Atlas section,
/// complete carrier, and one terminal encode must express that radix through
/// quotient/remainder and additive recurrence.  A source-level guard is
/// intentional here: a differential test cannot distinguish a radix identity
/// from a shift/mask spelling that happens to return the same bytes.
fn audit_float_radix_sources(atlas_raw: &str, accumulator_raw: &str, violations: &mut Vec<String>) {
    let atlas = mask_comments_strings_and_tests(atlas_raw);
    let accumulator = mask_comments_strings_and_tests(accumulator_raw);
    let complete = accumulator
        .find("pub struct Complete")
        .map_or(accumulator.as_str(), |start| &accumulator[start..]);

    for (family, source) in [
        ("finite Atlas section", atlas.as_str()),
        ("Complete/encode", complete),
    ] {
        if family == "finite Atlas section" && contains_packed_shift(source) {
            violations.push(format!(
                "`{family}` contains legacy bitwise/scan token `shift operator`; CU-11 requires radix quotient/remainder or additive recurrence"
            ));
        }
        for token in [
            "<<=",
            ">>=",
            "leading_zeros",
            "trailing_zeros",
            "count_ones",
        ] {
            if source.contains(token) {
                violations.push(format!(
                    "`{family}` contains legacy bitwise/scan token `{token}`; CU-11 requires radix quotient/remainder or additive recurrence"
                ));
            }
        }
    }

    audit_complete_radix_call_graph(accumulator_raw, violations);

    let atlas_compact = atlas.split_whitespace().collect::<String>();
    for witness in ["whileunit%2==0", "unit/=2", "self.rest%2", "self.rest%4"] {
        if !atlas_compact.contains(witness) {
            violations.push(format!(
                "`{ATLAS_REFERENCE_FILE}` lacks pure-radix NAF witness `{witness}`"
            ));
        }
    }

    let complete_compact = complete.split_whitespace().collect::<String>();
    let accumulator_compact = accumulator.split_whitespace().collect::<String>();
    for token in [
        "mask&COMPLETE_",
        "|COMPLETE_",
        "mask&!(COMPLETE_",
        "!(self.state",
        "sig&1",
    ] {
        if complete_compact.contains(token) {
            violations.push(format!(
                "`{ACCUMULATOR_FILE}` Complete/encode body retains bit-field spelling `{token}`"
            ));
        }
    }
    for witness in [
        "fnradix_binary_width(",
        "fnradix_window(",
        "fnradix_neg_limbs<",
        "fnradix_spread_u128(",
        "fncompose_ieee_bits(",
    ] {
        if !accumulator_compact.contains(witness) {
            violations.push(format!(
                "`{ACCUMULATOR_FILE}` lacks Complete/encode radix witness `{witness}`"
            ));
        }
    }

    let accumulator_functions = extract_functions(accumulator_raw, ACCUMULATOR_FILE);
    if let Some(spread) = accumulator_functions
        .iter()
        .find(|function| function.name == "radix_spread_u128")
    {
        let compact = spread.body.split_whitespace().collect::<String>();
        for witness in [
            "whileremaining!=0",
            "forwordin&mutwords",
            "letdoubled=u128::from(*word)+u128::from(*word)+carry",
            "carry=doubled/COMPLETE_LIMB_RADIX",
        ] {
            if !compact.contains(witness) {
                violations.push(format!(
                    "`{ACCUMULATOR_FILE}` radix spread lacks pure recurrence witness `{witness}`"
                ));
            }
        }
    } else {
        violations.push(format!(
            "`{ACCUMULATOR_FILE}` has no production `radix_spread_u128`"
        ));
    }
}

/// Follow every production helper reached by `Complete` and its one float
/// encoder. This closes the hole left by scanning only the declaration tail:
/// a helper moved above the struct, or a call back into a packed `Limbs`
/// observer, remains part of the audited arithmetic graph.
fn audit_complete_radix_call_graph(accumulator_raw: &str, violations: &mut Vec<String>) {
    let functions = extract_functions(accumulator_raw, ACCUMULATOR_FILE);
    let Some(public_start) = accumulator_raw
        .find("impl<const L: usize, const MIN_EXP: i32> core::fmt::Debug for Complete<L, MIN_EXP>")
    else {
        violations.push(format!(
            "`{ACCUMULATOR_FILE}` has no auditable public `Complete` implementation"
        ));
        return;
    };
    let Some(test_offset) = accumulator_raw[public_start..].find("#[cfg(test)]") else {
        violations.push(format!(
            "`{ACCUMULATOR_FILE}` has no boundary between production `Complete` and its tests"
        ));
        return;
    };
    let test_start = public_start + test_offset;
    let Some(encode_start) = accumulator_raw.find("macro_rules! impl_encode_into_float") else {
        violations.push(format!(
            "`{ACCUMULATOR_FILE}` has no auditable terminal float encoder"
        ));
        return;
    };
    let line_at = |byte: usize| {
        accumulator_raw[..byte]
            .bytes()
            .filter(|&value| value == b'\n')
            .count()
            + 1
    };
    let public_start = line_at(public_start);
    let test_start = line_at(test_start);
    let encode_start = line_at(encode_start);

    let mut by_name: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (index, function) in functions.iter().enumerate() {
        by_name.entry(&function.name).or_default().push(index);
    }
    let names: Vec<&str> = by_name.keys().copied().collect();
    let mut reached = BTreeSet::new();
    let mut queue = VecDeque::new();
    for (index, function) in functions.iter().enumerate() {
        if (public_start <= function.line && function.line < test_start)
            || encode_start <= function.line
        {
            reached.insert(index);
            queue.push_back(index);
        }
    }

    while let Some(caller) = queue.pop_front() {
        for name in &names {
            if !contains_call(&functions[caller].body, name) {
                continue;
            }
            let candidates = &by_name[*name];
            // An overloaded method name does not identify a receiver type in
            // this deliberately small source parser. Every helper introduced
            // for the radix recurrence is a free, uniquely named function;
            // packed `Limbs` observer names are rejected textually below.
            if candidates.len() != 1 {
                continue;
            }
            let callee = candidates[0];
            if reached.insert(callee) {
                queue.push_back(callee);
            }
        }
    }

    for required in [
        "add_at",
        "sub_at",
        "radix_neg_limbs",
        "radix_binary_width",
        "radix_spread_u128",
        "radix_window",
        "compose_ieee_bits",
    ] {
        if !reached
            .iter()
            .any(|&index| functions[index].name == required)
        {
            violations.push(format!(
                "`{ACCUMULATOR_FILE}` live Complete/encode graph does not reach `{required}`"
            ));
        }
    }

    for &index in &reached {
        let function = &functions[index];
        if contains_packed_shift(&function.body) {
            violations.push(format!(
                "{}:{}: live Complete/encode helper `{}` contains a legacy shift operator",
                function.rel, function.line, function.name
            ));
        }
        for method in [
            "bit",
            "window",
            "any_below",
            "neg",
            "leading_zeros",
            "trailing_zeros",
            "count_ones",
            "magnitude_low_u128",
            "low_i128",
            "add_i128",
            "add_i128_in_place",
        ] {
            if contains_call(&function.body, method) {
                violations.push(format!(
                    "{}:{}: live Complete/encode helper `{}` calls packed generic Limbs operation `{method}`",
                    function.rel, function.line, function.name
                ));
            }
        }
    }
}

/// Detect `<<` and `>>` after comments, strings, and arbitrary whitespace are
/// removed. The one nested associated-type spelling in the integer-to-float
/// bridge is type syntax rather than an operator and is erased explicitly.
fn contains_packed_shift(source: &str) -> bool {
    let compact = source.split_whitespace().collect::<String>().replace(
        "<$tasEncodeFrom<<$tasElement>::Acc>>::encode_from",
        "EncodeFromElementAcc::encode_from",
    );
    compact.contains("<<") || compact.contains(">>")
}

/// The chosen declaration minimizes the complete structural census over one
/// global compatible candidate set. Projection, decode, lookup issue, and peak
/// live storage are exact independent coordinates; omitting one is not an
/// optimization model but an unpriced operation.
fn audit_atlas_tile_selector(engine: &str, functions: &[Function], violations: &mut Vec<String>) {
    let compact_engine = engine.split_whitespace().collect::<String>();
    for witness in [
        "enumAtlasCountFactor{Rows,Depth,Columns,PhysicalTile,}",
        "constATLAS_COUNT_WORDS:usize=[AtlasCountFactor::Rows,AtlasCountFactor::Depth,AtlasCountFactor::Columns,AtlasCountFactor::PhysicalTile,].len()",
        "constATLAS_COUNT_RADIX:u128=u64::MAXasu128+(u64::MAX!=u64::MIN)asu128",
        "#[derive(Clone,Copy,Debug,PartialEq,Eq,PartialOrd,Ord)]structAtlasCount([u64;ATLAS_COUNT_WORDS]);",
        "#[derive(Clone,Copy,Debug,PartialEq,Eq,PartialOrd,Ord)]structAtlasWork{projections:AtlasCount,decodes:AtlasCount,issued:AtlasCount,product_initializations:AtlasCount,live_bytes:AtlasCount,}",
    ] {
        if !compact_engine.contains(witness) {
            violations.push(format!(
                "`{ATLAS_ENGINE_FILE}` lacks exact multi-coordinate work witness `{witness}`"
            ));
        }
    }
    let tile_selector = functions
        .iter()
        .find(|function| function.rel == ATLAS_ENGINE_FILE && function.name == "atlas_tile_spec");
    if let Some(selector) = tile_selector {
        let compact = selector.body.split_whitespace().collect::<String>();
        for family in ["available_i8", "available_i8_narrow", "available_reduce_i8"] {
            if !contains_call(&selector.body, family) {
                violations.push(format!(
                    "{}:{}: `atlas_tile_spec` omits `{family}` from the global Atlas candidate walk",
                    selector.rel, selector.line
                ));
            }
        }
        for witness in [
            ".filter(",
            "spec.k_group==1",
            "Factorization::Exact",
            "spec.max_bound>=",
            "atlas_executed_work::<A>(spec,shape,pa_codes,pb_codes)<=atlas_executed_work::<A>(incumbent,shape,pa_codes,pb_codes)",
        ] {
            if !compact.contains(witness) {
                violations.push(format!(
                    "{}:{}: `atlas_tile_spec` lacks global compatibility/minimum-work witness `{witness}`",
                    selector.rel, selector.line
                ));
            }
        }
        if compact.matches("atlas_executed_work::<A>(").count() < 2 {
            violations.push(format!(
                "{}:{}: `atlas_tile_spec` does not compare the candidate and incumbent by \
                 their actual projection/decode/issue/storage work",
                selector.rel, selector.line
            ));
        }
        if contains_call(&selector.body, "choose_for_rows") {
            violations.push(format!(
                "{}:{}: `atlas_tile_spec` preselects a family through `choose_for_rows` instead of comparing every compatible declaration",
                selector.rel, selector.line
            ));
        }
    } else {
        violations.push(format!(
            "`{ATLAS_ENGINE_FILE}` has no `atlas_tile_spec`; the three-family minimum-work selector is unaudited"
        ));
    }

    let executed_work = functions.iter().find(|function| {
        function.rel == ATLAS_ENGINE_FILE && function.name == "atlas_executed_work"
    });
    if let Some(executed_work) = executed_work {
        let compact = executed_work.body.split_whitespace().collect::<String>();
        for witness in [
            "ifshape.m==0||shape.k==0||shape.n==0{returnAtlasWork::ZERO;}",
            "letb_offer_cols=pb_codes.checked_div(shape.k).unwrap_or(0).min(shape.n)",
            "letblock_width=ifb_offer_cols==0{streamed_cols}else{b_offer_cols}",
            "letblock_count=full_blocks+usize::from(tail_cols!=0)",
            "letcolumn_tiles=full_blocks.checked_mul(full_block_tiles)",
            "letrow_tiles=shape.m.div_ceil(spec.mr)",
            "letdecoded_a=atlas_product_count(block_count,cached_rows).add(atlas_product_count(",
            "column_tiles,shape.m-cached_rows",
            "letdecoded_b=ifb_offer_cols==0{atlas_product_count(shape.n,row_tiles)}else{AtlasCount::from_u128(shape.nasu128)}",
            "letprojection_sites=decodes",
            "physical_outputs.div_ceil(spec.products_per_step)",
            "letissued=atlas_product_count(row_tiles,column_tiles).multiply(steps).multiply(shape.k)",
            "letproduct_initializations=atlas_product_count(shape.m,shape.n).multiply(shape.k)",
            "letlive_cells=live_rows.checked_mul(live_cols)",
            "ATLAS_TILE_WORK_BYTESasu128",
            "(live_cellsasu128).checked_mul(core::mem::size_of::<A>()asu128)",
            "AtlasWork{projections:projection_sites,decodes,issued,product_initializations,live_bytes,}",
        ] {
            if !compact.contains(witness) {
                violations.push(format!(
                    "{}:{}: `atlas_executed_work` omits independent structural-cost witness \
                     `{witness}`",
                    executed_work.rel, executed_work.line
                ));
            }
        }
        if compact.contains("uor_matmul_model::derive::") {
            violations.push(format!(
                "{}:{}: shipped `atlas_executed_work` delegates to its model twin, making \
                 CG-22's equality vacuous",
                executed_work.rel, executed_work.line
            ));
        }
        if compact.contains("saturating_")
            || compact.contains("atlas_retained_cells")
            || compact.contains("blocking::L1_BYTES")
        {
            violations.push(format!(
                "{}:{}: `atlas_executed_work` saturates or retains the removed output-window replay model",
                executed_work.rel, executed_work.line
            ));
        }
    } else {
        violations.push(format!(
            "`{ATLAS_ENGINE_FILE}` has no `atlas_executed_work`; selection cannot account for \
            every operation its one-pass frame actually executes"
        ));
    }

    for (name, witnesses) in [
        (
            "from_u128",
            &[
                "*word=(value%ATLAS_COUNT_RADIX)asu64",
                "value/=ATLAS_COUNT_RADIX",
                "debug_assert_eq!(value,0",
            ][..],
        ),
        (
            "add",
            &[
                "u128::from(self.0[index])+u128::from(other.0[index])+carry",
                "words[index]=(sum%ATLAS_COUNT_RADIX)asu64",
                "carry=sum/ATLAS_COUNT_RADIX",
            ][..],
        ),
        (
            "multiply",
            &[
                "u128::from(self.0[index])*factorasu128+carry",
                "words[index]=(product%ATLAS_COUNT_RADIX)asu64",
                "carry=product/ATLAS_COUNT_RADIX",
            ][..],
        ),
    ] {
        let exact = functions.iter().find(|function| {
            function.rel == ATLAS_ENGINE_FILE
                && function.name == name
                && function.body.contains("ATLAS_COUNT_RADIX")
        });
        let Some(exact) = exact else {
            violations.push(format!(
                "`{ATLAS_ENGINE_FILE}` has no exact AtlasCount `{name}` constructor"
            ));
            continue;
        };
        let compact = exact.body.split_whitespace().collect::<String>();
        for witness in witnesses {
            if !compact.contains(witness) {
                violations.push(format!(
                    "{}:{}: exact AtlasCount `{name}` lacks `{witness}`",
                    exact.rel, exact.line
                ));
            }
        }
        if compact.contains("saturating_") || compact.contains("<<") || compact.contains(">>") {
            violations.push(format!(
                "{}:{}: exact AtlasCount `{name}` uses saturating or bitwise arithmetic",
                exact.rel, exact.line
            ));
        }
    }
}

/// Dot-kernel resolution is shape-independent and therefore paid once per
/// requested backend under `std`. The no-std spelling remains the same direct
/// declaration resolver so the cache is an overhead elimination, not another
/// selection policy.
fn audit_atlas_dot_selector_cache(
    engine: &str,
    _engine_raw: &str,
    functions: &[Function],
    violations: &mut Vec<String>,
) {
    let compact_engine = engine.split_whitespace().collect::<String>();
    for witness in [
        "staticATLAS_DOT_AUTO_SPEC:std::sync::OnceLock<KernelSpec<i8,i32>>=std::sync::OnceLock::new();",
        "staticATLAS_DOT_NAMED_SPECS:[std::sync::OnceLock<KernelSpec<i8,i32>>;uor_matmul_core::Backend::ALL.len()]=[const{std::sync::OnceLock::new()};uor_matmul_core::Backend::ALL.len()];",
    ] {
        if !compact_engine.contains(witness) {
            violations.push(format!(
                "`{ATLAS_ENGINE_FILE}` lacks per-backend dot-selector cache witness `{witness}`"
            ));
        }
    }

    let Some(accessor) = functions
        .iter()
        .find(|function| function.rel == ATLAS_ENGINE_FILE && function.name == "atlas_dot_spec")
    else {
        violations.push(format!(
            "`{ATLAS_ENGINE_FILE}` has no cached `atlas_dot_spec` accessor"
        ));
        return;
    };
    let compact = accessor.body.split_whitespace().collect::<String>();
    for witness in [
        "letslot=ifbackend==uor_matmul_core::Backend::Auto{&ATLAS_DOT_AUTO_SPEC}else{&ATLAS_DOT_NAMED_SPECS[atlas_dot_backend_index(backend)]}",
        "slot.get_or_init(||{",
        "resolve_atlas_dot_spec(backend)",
    ] {
        if !compact.contains(witness) {
            violations.push(format!(
                "{}:{}: cached dot-selector accessor lacks `{witness}`",
                accessor.rel, accessor.line
            ));
        }
    }
    let initializer = compact.find("slot.get_or_init(||{");
    let first_resolution = compact.find("resolve_atlas_dot_spec(backend)");
    if !matches!((initializer, first_resolution), (Some(initializer), Some(resolution)) if initializer < resolution)
        || compact.matches("resolve_atlas_dot_spec(backend)").count() != 2
        || compact.contains("available_reduce_i8(")
    {
        violations.push(format!(
            "{}:{}: std resolution is not initializer-only or no-std does not share exactly the same resolver",
            accessor.rel, accessor.line
        ));
    }
    if !compact.contains("#[cfg(not(feature=))]{resolve_atlas_dot_spec(backend)}") {
        violations.push(format!(
            "`{ATLAS_ENGINE_FILE}` lacks the direct no-std dot-selector resolution"
        ));
    }
}

/// The shipped residency/work equations and the model derivation are
/// independent twins. CG-22 must execute an equality comparison between both;
/// a production call into the model or a test comparing either side to itself
/// would turn the claimed verification into a tautology.
fn audit_model_storage_differential(functions: &[Function], violations: &mut Vec<String>) {
    let Some(compare) = functions
        .iter()
        .find(|function| function.rel == ATLAS_ENGINE_FILE && function.name == "assert_model_work")
    else {
        violations.push(format!(
            "`{ATLAS_ENGINE_FILE}` has no CG-22 shipped/model storage-work differential"
        ));
        return;
    };
    let compact = compare.body.split_whitespace().collect::<String>();
    let comparison = "assert_eq!(atlas_executed_work::<A>(spec,shape,pa_codes,pb_codes).coordinates(),uor_matmul_model::derive::atlas_executed_work(shape.m,shape.k,shape.n,spec.mr,spec.nr,spec.products_per_step,bytes,ATLAS_TILE_WORK_BYTES,pa_codes,pb_codes,MAX_TILE_LANES,),";
    if !compact.contains(comparison) {
        violations.push(format!(
            "{}:{}: CG-22 lacks non-vacuous shipped/model comparison `{comparison}`",
            compare.rel, compare.line
        ));
    }
    if compact
        .matches("uor_matmul_model::derive::atlas_executed_work(")
        .count()
        != 1
        || compact
            .matches("atlas_executed_work::<A>(spec,shape,pa_codes,pb_codes)")
            .count()
            != 1
    {
        violations.push(format!(
            "{}:{}: CG-22 must compare one shipped census with one independently derived model census",
            compare.rel, compare.line
        ));
    }
    for removed in ["atlas_l1_cells", "atlas_cell_lanes", "blocking::L1_BYTES"] {
        if compact.contains(removed) {
            violations.push(format!(
                "{}:{}: CG-22 retains removed output-window model token `{removed}`",
                compare.rel, compare.line
            ));
        }
    }

    let coverage = functions.iter().find(|function| {
        function.rel == ATLAS_ENGINE_FILE
            && function.name == "atlas_route_and_operation_census_follow_the_model_cg_22"
    });
    if let Some(coverage) = coverage {
        let compact = coverage.body.split_whitespace().collect::<String>();
        for witness in [
            "available_i8()",
            "available_i8_narrow()",
            "available_reduce_i8()",
            "assert_model_work::<AccOf<f32>>(spec,shape,0,0)",
            "assert_model_work::<AccOf<f64>>(spec,shape,0,0)",
            "usize::MAX",
            "Strides{rs:0,cs:0}",
            ".take(ATLAS_COUNT_WORDS-1).any(|word|word!=0)",
            "assert_model_work::<AccOf<f64>>(spec,offered_shape,pa_codes,pb_codes)",
            "assert_eq!(census.projections,exact_usize(work.projections))",
            "assert_eq!(census.decoded_a+census.decoded_b,exact_usize(work.decodes))",
            "assert_eq!(census.issued_steps,exact_usize(work.issued))",
            "assert_eq!(census.product_initializations,exact_usize(work.product_initializations),",
            "letsource_lower_bound=(shape.m+shape.n)*shape.k",
            "letempty_depth=Shape{m:3,k:0,n:5}",
            "assert_eq!(census.route,None)",
            "assert_eq!(census.encodes,empty_depth.m*empty_depth.n)",
            "assert_model_work::<AccOf<f64>>(spec,empty_depth,7,11)",
        ] {
            if !compact.contains(witness) {
                violations.push(format!(
                    "{}:{}: CG-22 does not exercise `{witness}` across its candidate sweep",
                    coverage.rel, coverage.line
                ));
            }
        }
    } else {
        violations.push(format!(
            "`{ATLAS_ENGINE_FILE}` has no executable CG-22 candidate/model coverage test"
        ));
    }
}

#[derive(Debug, PartialEq, Eq)]
struct PerformanceRow {
    m: usize,
    k: usize,
    n: usize,
    fill: String,
    seed: u64,
}

fn audit_candidate_performance_measurement(
    root: &Path,
    float: &str,
    violations: &mut Vec<String>,
) -> Result<(), Fail> {
    audit_compute_only_float_sweep(root, violations)?;
    let corpus = std::fs::read_to_string(root.join(FLOAT_CORPUS_FILE))?;
    let candidate_rows = performance_rows(float, "CANDIDATE_CASES", "CandidateCase");
    let validation_rows = performance_rows(&corpus, "PERFORMANCE_CASES", "FloatCase");
    if candidate_rows.len() != 6 || validation_rows.len() != 6 {
        violations.push(format!(
            "CG-21 candidate/validation corpus is not the declared six rows (candidate {}, validation {})",
            candidate_rows.len(),
            validation_rows.len()
        ));
    }
    if candidate_rows != validation_rows {
        violations.push(format!(
            "CG-21 forced-candidate rows diverge from `{FLOAT_CORPUS_FILE}` PERFORMANCE_CASES: candidate {candidate_rows:?}, validation {validation_rows:?}"
        ));
    }
    audit_compute_only_candidate_sweep_source(float, violations);

    let justfile = std::fs::read_to_string(root.join(JUSTFILE))?;
    let compact_just = justfile.split_whitespace().collect::<String>();
    if !compact_just.contains(
        "cargotest--release-puor-matmul-gemm--libfloat::tests::every_atlas_candidate_is_measurable_with_byte_checks_cg_21----ignored--exact--nocapture--test-threads=1",
    ) {
        violations.push(
            "`just uor-float-sweep` does not run the forced-candidate release measurement"
                .to_string(),
        );
    }
    Ok(())
}

/// Keep the ignored selector instrument honest too. A forced candidate is a
/// production engine call with a supplied selector, not an excuse to price
/// output poisoning, public boundary construction, or verification as kernel
/// work. Interleaving and pairing preserve the caller-visible comparison when
/// clock rate or machine load drifts across the release run.
fn audit_compute_only_candidate_sweep_source(source: &str, violations: &mut Vec<String>) {
    let functions = extract_functions_including_tests(source, ATLAS_ENGINE_FILE);
    let Some(poison) = functions
        .iter()
        .find(|function| function.name == "poison_candidate_output")
    else {
        violations.push(format!(
            "`{ATLAS_ENGINE_FILE}` has no expected-derived CG-21 candidate poison"
        ));
        return;
    };
    let compact = poison.body.split_whitespace().collect::<String>();
    for witness in [
        "output.len(),expected.len()",
        "expected.symbol_bits()^1",
        "assert_ne!(poisoned.symbol_bits(),expected.symbol_bits()",
        "*output=poisoned",
    ] {
        if !compact.contains(witness) {
            violations.push(format!(
                "{}:{}: CG-21 candidate poison is not distinct and expected-derived at every cell; missing `{witness}`",
                poison.rel, poison.line
            ));
        }
    }

    let Some(comparator) = functions
        .iter()
        .find(|function| function.name == "assert_candidate_bytes")
    else {
        violations.push(format!(
            "`{ATLAS_ENGINE_FILE}` has no complete CG-21 candidate byte comparator"
        ));
        return;
    };
    let compact = comparator.body.split_whitespace().collect::<String>();
    for witness in [
        "actual.len(),expected.len()",
        "actual.iter().zip(expected).enumerate()",
        "actual.symbol_bits(),expected.symbol_bits()",
    ] {
        if !compact.contains(witness) {
            violations.push(format!(
                "{}:{}: CG-21 candidate comparator is not a complete byte comparison; missing `{witness}`",
                comparator.rel, comparator.line
            ));
        }
    }

    let Some(batch) = functions
        .iter()
        .find(|function| function.name == "candidate_timed_batch")
    else {
        violations.push(format!(
            "`{ATLAS_ENGINE_FILE}` has no forced-candidate CG-21 batch"
        ));
        return;
    };
    let compact = batch.body.split_whitespace().collect::<String>();
    let Some((start, end)) = compact
        .find("letstart=std::time::Instant::now();")
        .zip(compact.find("start.elapsed()"))
        .filter(|(start, end)| start < end)
    else {
        violations.push(format!(
            "{}:{}: CG-21 candidate batch has no ordered production interval",
            batch.rel, batch.line
        ));
        return;
    };
    let prefix = &compact[..start];
    let interval = &compact[start..end];
    let suffix = &compact[end..];
    for witness in [
        "poison_candidate_output(&mutmeasured.output,expected)",
        "MatView::row_major",
        "MatViewMut::row_major",
        "Triple::new",
        "GemmOptions::default",
        "letselect=",
        "letmutledger=()",
    ] {
        if !prefix.contains(witness) {
            violations.push(format!(
                "{}:{}: CG-21 candidate batch does not prepare `{witness}` before timing",
                batch.rel, batch.line
            ));
        }
    }
    if !contains_call(interval, "gemm_float_tiles_with_selector") {
        violations.push(format!(
            "{}:{}: CG-21 candidate timer excludes the real `gemm_float_tiles_with_selector` call",
            batch.rel, batch.line
        ));
    }
    for contamination in [
        "poison_candidate_output(",
        "assert_candidate_bytes(",
        "MatView::",
        "MatViewMut::",
        "Triple::new",
        "GemmOptions::",
        "Vec::",
        "vec![",
    ] {
        if interval.contains(contamination) {
            violations.push(format!(
                "{}:{}: CG-21 candidate timer is contaminated by `{contamination}`",
                batch.rel, batch.line
            ));
        }
    }
    if !suffix.contains("assert_candidate_bytes(&measured.output,expected)") {
        violations.push(format!(
            "{}:{}: CG-21 candidate batch lacks its complete post-timer byte check",
            batch.rel, batch.line
        ));
    }

    let Some(sweep) = functions
        .iter()
        .find(|function| function.name == "candidate_release_sweep")
    else {
        violations.push(format!(
            "`{ATLAS_ENGINE_FILE}` has no forced-candidate CG-21 measurement body"
        ));
        return;
    };
    let compact = sweep.body.split_whitespace().collect::<String>();
    let source_compact = source.split_whitespace().collect::<String>();
    for witness in [
        "available_i8()",
        "available_i8_narrow()",
        "available_reduce_i8()",
        "same_candidate_route(candidate,spec)",
        "forcaseinCANDIDATE_CASES",
        "(,0,0)",
        "(,shape.k,shape.k)",
        "(,suggested.0,suggested.1)",
        "CandidateMeasurement::new(spec,shape,pa_codes,pb_codes)",
        "Duration::from_millis(4)",
        "forroundin0..CANDIDATE_SAMPLES",
        "letat=(round+offset)%measurements.len()",
        "candidate_timed_batch(",
        "atlas_tile_spec::<AccOf<E>>(Backend::Auto,shape,pa_codes,pb_codes)",
        "letselected_seconds=measurements[selected_at].seconds",
        "measured.seconds[at]/selected_seconds[at]",
        "candidate_estimate(&ratios)",
        "ratio_half_width,ratios,CANDIDATE_SAMPLES",
        "measured.elapsed_ns[round]=elapsed.as_nanos()",
        "for(route,measured)inmeasurements.iter().enumerate()",
        "for(round,elapsed_ns)inmeasured.elapsed_ns.iter().enumerate()",
        "E::LABEL",
        "route,measured.spec.backend,measured.spec.factorization",
        "round,measured.batch,elapsed_ns",
    ] {
        if !compact.contains(witness) {
            violations.push(format!(
                "{}:{}: CG-21 candidate measurement lacks `{witness}`",
                sweep.rel, sweep.line
            ));
        }
    }
    if !source_compact.contains(
        "CG21_SAMPLEphase=candidatewidth={}case={}m={}k={}n={}fill={:?}offer={}route={}backend={:?}factorization={:?}mr={}nr={}layout={:?}k_group={}products_per_step={}lane_cap={}max_bound={}round={}batch={}elapsed_ns={}",
    ) {
        violations.push(format!(
            "{}:{}: CG-21 candidate measurement lacks its machine-readable width/case/route/round/batch/elapsed schema",
            sweep.rel, sweep.line
        ));
    }
}

/// Keep correctness guards and boundary construction complete without allowing
/// either to masquerade as implementation throughput. Each measured route owns
/// a batch function: poison, views, conformant triple, options, and offers are
/// established before its clock; the interval repeats one production API; the
/// complete comparison follows the captured elapsed duration.
fn audit_compute_only_float_sweep(root: &Path, violations: &mut Vec<String>) -> Result<(), Fail> {
    let raw = std::fs::read_to_string(root.join(FLOAT_SWEEP_FILE))?;
    audit_compute_only_float_sweep_source(&raw, violations);
    Ok(())
}

fn audit_compute_only_float_sweep_source(source: &str, violations: &mut Vec<String>) {
    let functions = extract_functions_including_tests(source, FLOAT_SWEEP_FILE);
    let Some(poison) = functions
        .iter()
        .find(|function| function.name == "poison_from_expected")
    else {
        violations.push(format!(
            "`{FLOAT_SWEEP_FILE}` has no expected-derived CG-21 float poison"
        ));
        return;
    };
    let compact = poison.body.split_whitespace().collect::<String>();
    for witness in [
        "E::from_corpus_bits(expected.corpus_bits()^1)",
        "assert_ne!(poisoned.corpus_bits(),expected.corpus_bits()",
    ] {
        if !compact.contains(witness) {
            violations.push(format!(
                "{}:{}: CG-21 float poison is not proved distinct from each expected code; missing `{witness}`",
                poison.rel, poison.line
            ));
        }
    }

    let Some(poison_output) = functions
        .iter()
        .find(|function| function.name == "poison_output")
    else {
        violations.push(format!(
            "`{FLOAT_SWEEP_FILE}` has no complete CG-21 output poison"
        ));
        return;
    };
    let compact = poison_output.body.split_whitespace().collect::<String>();
    for witness in [
        "out.len(),expected.len()",
        "out.iter_mut().zip(expected)",
        "*out=poison_from_expected(expected)",
    ] {
        if !compact.contains(witness) {
            violations.push(format!(
                "{}:{}: CG-21 output poison does not cover every expected cell; missing `{witness}`",
                poison_output.rel, poison_output.line
            ));
        }
    }

    let Some(comparator) = functions
        .iter()
        .find(|function| function.name == "assert_bits")
    else {
        violations.push(format!(
            "`{FLOAT_SWEEP_FILE}` has no complete CG-21 byte comparator"
        ));
        return;
    };
    let compact = comparator.body.split_whitespace().collect::<String>();
    for witness in [
        "got.len(),want.len()",
        "got.iter().zip(want).enumerate()",
        "got.corpus_bits(),want.corpus_bits()",
    ] {
        if !compact.contains(witness) {
            violations.push(format!(
                "{}:{}: CG-21 comparator is not a complete byte comparison; missing `{witness}`",
                comparator.rel, comparator.line
            ));
        }
    }

    let Some(samples) = functions.iter().find(|function| function.name == "samples") else {
        violations.push(format!(
            "`{FLOAT_SWEEP_FILE}` has no calibrated sample loop"
        ));
        return;
    };
    let compact = samples.body.split_whitespace().collect::<String>();
    for witness in [
        "measured_batch(1);letpilot=measured_batch(1);",
        "letmutelapsed=[Duration::ZERO;SAMPLE_COUNT];forsamplein&mutelapsed{*sample=measured_batch(batch);}",
        "letbatch=SAMPLE_TARGET",
    ] {
        if !compact.contains(witness) {
            violations.push(format!(
                "{}:{}: compute-only calibration lacks `{witness}`",
                samples.rel, samples.line
            ));
        }
    }
    let source_compact = source.split_whitespace().collect::<String>();
    if !source_compact.contains("constSAMPLE_TARGET:Duration=Duration::from_millis(4);") {
        violations.push(format!(
            "`{FLOAT_SWEEP_FILE}` does not declare its bounded calibration target"
        ));
    }
    if compact.contains("Instant::now") {
        violations.push(format!(
            "{}:{}: generic calibration owns a clock instead of the prepared route batch",
            samples.rel, samples.line
        ));
    }

    let Some(printer) = functions
        .iter()
        .find(|function| function.name == "print_metrics")
    else {
        violations.push(format!(
            "`{FLOAT_SWEEP_FILE}` has no CG-21 raw-sample printer"
        ));
        return;
    };
    let compact = printer.body.split_whitespace().collect::<String>();
    for witness in [
        "for(round,elapsed_ns)inmeasured.elapsed_ns.iter().enumerate()",
        "shape.m,shape.k,shape.n,measured.batch",
    ] {
        if !compact.contains(witness) {
            violations.push(format!(
                "{}:{}: CG-21 public measurement lacks machine-readable raw sample field `{witness}`",
                printer.rel, printer.line
            ));
        }
    }
    if !source_compact.contains(
        "CG21_SAMPLEphase=publicwidth={width}case={case}m={}k={}n={}fill={fill}route={route_id}round={round}batch={}elapsed_ns={elapsed_ns}",
    ) {
        violations.push(format!(
            "{}:{}: CG-21 public measurement lacks machine-readable raw sample fields",
            printer.rel, printer.line
        ));
    }
    if !source_compact.contains("elapsed_ns:elapsed.map(|elapsed|elapsed.as_nanos())") {
        violations.push(format!(
            "`{FLOAT_SWEEP_FILE}` does not retain each raw batch duration"
        ));
    }

    let faer_poisoners: Vec<_> = functions
        .iter()
        .filter(|function| function.name == "poison_faer_output")
        .collect();
    if faer_poisoners.len() != 2 {
        violations.push(format!(
            "`{FLOAT_SWEEP_FILE}` has {} faer poison implementations rather than the two measured IEEE widths",
            faer_poisoners.len()
        ));
    }
    for function in faer_poisoners {
        let compact = function.body.split_whitespace().collect::<String>();
        for witness in [
            "expected.len(),case.m*case.n",
            "state.c[(i,j)]=poison_from_expected(expected[i*case.n+j])",
        ] {
            if !compact.contains(witness) {
                violations.push(format!(
                    "{}:{}: faer's real C state is not expected-derived and poisoned at every cell; missing `{witness}`",
                    function.rel, function.line
                ));
            }
        }
    }

    type RouteAudit<'a> = (
        &'a str,
        &'a str,
        &'a [&'a str],
        &'a [&'a str],
        &'a [&'a str],
    );
    let routes: &[RouteAudit<'_>] = &[
        (
            "timed_float_packed_batch",
            "uor_matmul::gemm_float_packed",
            &["poison_output(out,expected)"],
            &[
                "MatView::row_major",
                "MatViewMut::row_major",
                "Triple::new",
                "GemmOptions::default",
            ],
            &["assert_bits(out,expected)"],
        ),
        (
            "timed_float_no_offer_batch",
            "uor_matmul::gemm_float",
            &["poison_output(out,expected)"],
            &[
                "MatView::row_major",
                "MatViewMut::row_major",
                "Triple::new",
                "GemmOptions::default",
            ],
            &["assert_bits(out,expected)"],
        ),
        (
            "timed_incumbent_batch",
            "uor_matmul_gemm::gemm",
            &["poison_output(out,expected)"],
            &[
                "MatView::row_major",
                "MatViewMut::row_major",
                "Triple::new",
                "Scratch::none",
                "GemmOptions::default",
            ],
            &["assert_bits(out,expected)"],
        ),
        (
            "timed_matrixmultiply_batch",
            "E::matrixmultiply",
            &["poison_output(out,expected)"],
            &[],
            &["assert_bits(out,expected)"],
        ),
        (
            "timed_faer_batch",
            "E::faer_compute",
            &[
                "poison_output(out,expected)",
                "E::poison_faer_output(state,case,expected)",
            ],
            &[],
            &[
                "E::copy_faer_output(state,case,out)",
                "assert_bits(out,expected)",
            ],
        ),
        (
            "timed_integer_batch",
            "uor_matmul::gemm_packed",
            &[
                "out.len(),expected.len()",
                "out.iter_mut().zip(expected)",
                "*out=expected^1",
                "assert_ne!(*out,expected)",
            ],
            &[
                "MatView::row_major",
                "MatViewMut::row_major",
                "Triple::new",
                "Scratch::new",
                "letoptions=GemmOptions",
            ],
            &["assert_eq!(out,expected"],
        ),
        (
            "timed_tropical_batch",
            "uor_matmul_gemm::gemm",
            &[
                "out.len(),expected.len()",
                "out.iter_mut().zip(expected)",
                "assert_ne!(*out,expected)",
            ],
            &[
                "MatView::row_major",
                "MatViewMut::row_major",
                "Triple::new",
                "Scratch::none",
                "GemmOptions::default",
            ],
            &["assert_eq!(out,expected"],
        ),
    ];
    for &(function_name, production_call, poison, setup, checks) in routes {
        let Some(function) = functions
            .iter()
            .find(|function| function.name == function_name)
        else {
            violations.push(format!(
                "`{FLOAT_SWEEP_FILE}` has no `{function_name}` measurement"
            ));
            continue;
        };
        let compact = function.body.split_whitespace().collect::<String>();
        let Some((start, end)) = compact
            .find("letstart=Instant::now();")
            .zip(compact.find("start.elapsed()"))
            .filter(|(start, end)| start < end)
        else {
            violations.push(format!(
                "{}:{}: `{function_name}` has no ordered production interval",
                function.rel, function.line
            ));
            continue;
        };
        let prefix = &compact[..start];
        let interval = &compact[start..end];
        let suffix = &compact[end..];
        for &witness in poison {
            if !prefix.contains(witness) {
                violations.push(format!(
                    "{}:{}: `{function_name}` does not prepare expected-derived poison `{witness}` before timing",
                    function.rel, function.line
                ));
            }
        }
        for &witness in setup {
            if !prefix.contains(witness) {
                violations.push(format!(
                    "{}:{}: `{function_name}` does not prepare `{witness}` before timing",
                    function.rel, function.line
                ));
            }
        }
        if !contains_call(interval, production_call) {
            violations.push(format!(
                "{}:{}: `{function_name}` timer excludes the real `{production_call}` call",
                function.rel, function.line
            ));
        }
        for contamination in [
            "out.fill(",
            "poison_output(",
            "iter_mut().zip(expected)",
            "assert_bits(",
            "assert_eq!(",
            "MatView::",
            "MatViewMut::",
            "Triple::new",
            "Scratch::",
            "GemmOptions::",
            "poison_faer_output",
            "copy_faer_output",
        ] {
            if interval.contains(contamination) {
                violations.push(format!(
                    "{}:{}: `{function_name}` timer is contaminated by `{contamination}`",
                    function.rel, function.line,
                ));
            }
        }
        for &check in checks {
            if !suffix.contains(check) {
                violations.push(format!(
                    "{}:{}: `{function_name}` lacks post-timer `{check}`",
                    function.rel, function.line
                ));
            }
        }
    }
}

fn performance_rows(source: &str, symbol: &str, constructor: &str) -> Vec<PerformanceRow> {
    let source = mask_comments_and_strings(source);
    let Some(array) = const_array_initializer(&source, symbol) else {
        return Vec::new();
    };
    let mut rows = Vec::new();
    let needle = format!("{constructor} {{");
    let mut from = 0usize;
    while let Some(offset) = array[from..].find(&needle) {
        let open = from + offset + constructor.len() + 1;
        let bytes = array.as_bytes();
        let mut depth = 1usize;
        let mut end = open + 1;
        while end < bytes.len() && depth != 0 {
            match bytes[end] {
                b'{' => depth += 1,
                b'}' => depth -= 1,
                _ => {}
            }
            end += 1;
        }
        if depth != 0 {
            break;
        }
        let compact = array[open + 1..end - 1]
            .split_whitespace()
            .collect::<String>();
        let fill = ["CandidateFill::", "FloatFill::"]
            .into_iter()
            .find_map(|prefix| identifier_after(&compact, &format!("fill:{prefix}")));
        if let (Some(m), Some(k), Some(n), Some(fill), Some(seed)) = (
            decimal_after(&compact, "m:"),
            decimal_after(&compact, "k:"),
            decimal_after(&compact, "n:"),
            fill,
            decimal_after(&compact, "seed:"),
        ) {
            rows.push(PerformanceRow {
                m: m as usize,
                k: k as usize,
                n: n as usize,
                fill,
                seed,
            });
        }
        from = end;
    }
    rows
}

fn const_array_initializer<'a>(source: &'a str, symbol: &str) -> Option<&'a str> {
    let declaration = source.find(&format!("const {symbol}"))?;
    let equals = declaration + source[declaration..].find('=')?;
    let open = equals + source[equals..].find('[')?;
    let bytes = source.as_bytes();
    let mut depth = 1usize;
    let mut end = open + 1;
    while end < bytes.len() && depth != 0 {
        match bytes[end] {
            b'[' => depth += 1,
            b']' => depth -= 1,
            _ => {}
        }
        end += 1;
    }
    (depth == 0).then_some(&source[open + 1..end - 1])
}

fn decimal_after(source: &str, prefix: &str) -> Option<u64> {
    let start = source.find(prefix)? + prefix.len();
    let digits: String = source[start..]
        .chars()
        .take_while(|character| character.is_ascii_digit() || *character == '_')
        .filter(|character| *character != '_')
        .collect();
    (!digits.is_empty()).then(|| digits.parse().ok()).flatten()
}

fn identifier_after(source: &str, prefix: &str) -> Option<String> {
    let start = source.find(prefix)? + prefix.len();
    let identifier: String = source[start..]
        .chars()
        .take_while(|character| character.is_ascii_alphanumeric() || *character == '_')
        .collect();
    (!identifier.is_empty()).then_some(identifier)
}

/// Bind the model-derived seven-state tail to the public-compatibility
/// regression that exercises every formerly observable flag union.
///
/// Runtime assertions establish behavior; this structural half prevents a
/// weakened loop, missing Debug/Hash check, or canonical-only model pin from
/// continuing to look like that evidence.
fn audit_complete_tail_compatibility(
    root: &Path,
    violations: &mut Vec<String>,
) -> Result<(), Fail> {
    let model = Model::load(&root.join("model"))?;
    model.check()?;
    let generated = std::fs::read_to_string(root.join(CORE_GENERATED_FILE))?;
    let state = &model.widths.complete_state;
    let expected_states =
        uor_matmul_model::derive::complete_nonfinite_states(state.nonfinite_flag_count);
    for witness in [
        "pub(crate) const BASE: i64 = i64::MIN;".to_string(),
        "pub(crate) const NAN_MASK: u8 = 1;".to_string(),
        "pub(crate) const POS_INF_MASK: u8 = 2;".to_string(),
        "pub(crate) const NEG_INF_MASK: u8 = 4;".to_string(),
        format!("pub(crate) const COUNT: u32 = {expected_states};"),
    ] {
        if !generated.contains(&witness) {
            violations.push(format!(
                "`{CORE_GENERATED_FILE}` lacks model-derived Complete tail witness `{witness}`"
            ));
        }
    }

    let acc = std::fs::read_to_string(root.join(ACCUMULATOR_FILE))?;
    let production = mask_comments_strings_and_tests(&acc)
        .split_whitespace()
        .collect::<String>();
    for witness in [
        "#[derive(Clone,Copy,PartialEq,Eq)]pubstructComplete",
        "impl<constL:usize,constMIN_EXP:i32>core::fmt::DebugforComplete<L,MIN_EXP>",
        "impl<constL:usize,constMIN_EXP:i32>core::hash::HashforComplete<L,MIN_EXP>",
        "letmask=radix_union_nonfinite_masks(self.nonfinite_mask().unwrap_or(0),COMPLETE_NAN_MASK)",
        "letmask=radix_union_nonfinite_masks(self.nonfinite_mask().unwrap_or(0),bit)",
        "letmask=radix_union_nonfinite_masks(self.nonfinite_mask().unwrap_or(0),other.nonfinite_mask().unwrap_or(0),)",
        "limbs:self.limbs.combine(other.limbs)",
    ] {
        if !production.contains(witness) {
            violations.push(format!(
                "`{ACCUMULATOR_FILE}` lacks former-tail compatibility witness `{witness}`"
            ));
        }
    }
    if production.contains("Debug)]pubstructComplete")
        || production.contains("Hash,Debug)]pubstructComplete")
    {
        violations.push(format!(
            "`{ACCUMULATOR_FILE}` exposes the replacement `state` field through derived Debug/Hash"
        ));
    }

    let tests = extract_functions_including_tests(&acc, ACCUMULATOR_FILE);
    let regression = tests
        .iter()
        .find(|function| function.name == "complete_tail_preserves_former_flag_observations_cs_13");
    if let Some(regression) = regression {
        let compact = regression.body.split_whitespace().collect::<String>();
        for witness in [
            "formaskin1..=COMPLETE_NONFINITE_STATE_COUNTasu8",
            "assert_eq!(*value.raw(),low",
            "assert_eq!(format!(),format!())",
            "digest(&Former{limbs:low,nan,pos_inf,neg_inf,})",
            "assert_eq!(states[left]==states[right],left==right)",
            "assert!(negative.is_negative())",
            "assert_eq!(negative.magnitude(),negative_low.neg())",
            "assert_eq!(*combined.raw(),low.combine(low))",
            "opposite.set_infinity(false)",
            "opposite.set_infinity(true)",
        ] {
            if !compact.contains(witness) {
                violations.push(format!(
                    "{}:{}: CS-13 Complete-tail regression lacks `{witness}`",
                    regression.rel, regression.line
                ));
            }
        }
    } else {
        violations.push(format!(
            "`{ACCUMULATOR_FILE}` has no `complete_tail_preserves_former_flag_observations_cs_13` regression"
        ));
    }
    Ok(())
}

/// Audit the generic public tabulation roots at their two float
/// instantiations. A source call graph cannot resolve an associated method such
/// as `E::dense_gemm`; following all six same-named implementations would
/// invent integer edges. Reading the two concrete impl blocks is the exact
/// type-directed counterpart: `f32` must terminate in the q-cell lookup lane,
/// while the API-locked `f64` complete lane must be inert and its
/// only executable spelling must re-enter the Atlas float driver.
fn audit_tabulated_float(
    root: &Path,
    functions: &[Function],
    violations: &mut Vec<String>,
) -> Result<(usize, usize), Fail> {
    let tabulated_raw = std::fs::read_to_string(root.join(TABULATED_FILE))?;
    let table_raw = std::fs::read_to_string(root.join(TABLE_FILE))?;
    Ok(audit_tabulated_float_sources(
        &tabulated_raw,
        &table_raw,
        functions,
        violations,
    ))
}

/// Both whole-block and scalar-fractured builds consume the same addressed set.
/// The set owns deduplication; each builder must walk that set exactly once and
/// retain the complete-book presentation as the total short-offer fallback.
fn audit_addressed_entry_build(functions: &[Function], violations: &mut Vec<String>) {
    let Some(collect) = functions
        .iter()
        .find(|function| function.rel == TABULATED_FILE && function.name == "collect")
    else {
        violations
            .push("the addressed-entry set has no call-wide collection operation".to_string());
        return;
    };
    let compact = collect.body.split_whitespace().collect::<String>();
    for witness in [
        "self.insert(C::index_of(code))",
        "EntryInsert::Full",
        "self.clear()",
        "letcount=self.used",
        "self.occupied[at]=self.seen[slot]-1",
        "self.seen[slot]=0",
    ] {
        if !compact.contains(witness) {
            violations.push(format!(
                "{}:{}: addressed-entry collection lacks exact deduplication/rollback witness `{witness}`",
                collect.rel, collect.line
            ));
        }
    }

    for (name, entry_call) in [
        ("build_source_block", "table.build_entry("),
        ("build_source_scalar", "table.build_cell("),
    ] {
        let Some(build) = functions
            .iter()
            .find(|function| function.rel == TABULATED_FILE && function.name == name)
        else {
            violations.push(format!(
                "`{TABULATED_FILE}` has no `{name}`; addressed q builds are unaudited"
            ));
            continue;
        };
        let compact = build.body.split_whitespace().collect::<String>();
        for witness in [
            "set.insert(C::index_of(code))",
            "ifcomplete&&set.len()<table.code_space()",
            "foratin0..set.len()",
            "set.index(at)",
            "set.clear()",
        ] {
            if !compact.contains(witness) {
                violations.push(format!(
                    "{}:{}: `{name}` does not issue exactly one presentation per distinct addressed index; missing `{witness}`",
                    build.rel, build.line
                ));
            }
        }
        if !compact.contains(&entry_call.split_whitespace().collect::<String>()) {
            violations.push(format!(
                "{}:{}: `{name}` does not terminate in `{entry_call}`",
                build.rel, build.line
            ));
        }
    }
}

fn audit_tabulated_float_sources(
    tabulated_raw: &str,
    table_raw: &str,
    functions: &[Function],
    violations: &mut Vec<String>,
) -> (usize, usize) {
    audit_tabulated_atlas_stream(
        &mask_comments_strings_and_tests(tabulated_raw),
        functions,
        violations,
    );
    audit_tabulated_radix_addresses(functions, violations);
    let tabulated_tests = extract_functions_including_tests(tabulated_raw, TABULATED_FILE);
    audit_tabulated_stream_differentials(&tabulated_tests, violations);
    let mut audited = 0usize;

    let run_lane = functions
        .iter()
        .find(|function| function.rel == TABULATED_FILE && function.name == "run_lane");
    if let Some(run_lane) = run_lane {
        let compact = run_lane.body.split_whitespace().collect::<String>();
        let admission = compact.find("if!admits(");
        let column_pass = compact.find("distinct_columns::<E,Bd,C>(");
        if admission.is_none() || column_pass.is_none() || column_pass <= admission {
            violations.push(format!(
                "{}:{}: a structurally declined table pays the column hash pass before admission",
                run_lane.rel, run_lane.line
            ));
        }
        for witness in [
            "column_workspace(index,shape.n,plan.cols,repeated,need_entries)",
            "letneed_entries=block==1&&!C::SIGN_BIT_BOOK&&space>1;",
            "set.collect::<E,Bd,C>(triple.w.codes())",
            "addressed_lane_scale(&triple.a,&triple.w,addressed,ledger)",
        ] {
            if !compact.contains(witness) {
                violations.push(format!(
                    "{}:{}: the admitted block-one route lacks distinct-address witness `{witness}`",
                    run_lane.rel, run_lane.line
                ));
            }
        }
    } else {
        violations.push(format!(
            "`{TABULATED_FILE}` has no `run_lane`; column-pass admission order is unaudited"
        ));
    }

    let column_workspace = functions
        .iter()
        .find(|function| function.rel == TABULATED_FILE && function.name == "column_workspace");
    if column_workspace.is_none_or(|function| {
        let compact = function.body.split_whitespace().collect::<String>();
        !compact.contains("if!need_entries{return(None,None);")
            || !compact.contains("if!need_entries{return(collapse,None);")
    }) {
        violations.push(
            "non-pointwise/sign-book/one-coordinate calls construct or clear an unused addressed-entry set"
                .to_string(),
        );
    }

    audit_addressed_entry_build(functions, violations);

    for (element, lane, stream_lane) in [
        ("f32", "Scaled64", "Wide<AccOf<f32>>"),
        ("f64", "Wide<AccOf<f64>>", "Wide<AccOf<f64>>"),
    ] {
        let declaration = format!("impl Tabulated for {element}");
        let Some(block) = impl_block(tabulated_raw, &declaration) else {
            violations.push(format!(
                "`{TABULATED_FILE}` has no `{declaration}`; public float tabulation is unaudited"
            ));
            continue;
        };
        audited += 1;
        if !block
            .split_whitespace()
            .collect::<String>()
            .contains(&format!("typeLane={lane};"))
        {
            violations.push(format!(
                "`{declaration}` does not declare the audited `{lane}` lane"
            ));
        }
        if !block
            .split_whitespace()
            .collect::<String>()
            .contains(&format!("typeStreamLane={stream_lane};"))
        {
            violations.push(format!(
                "`{declaration}` does not bind its existing stream extension point to private \
                 `{stream_lane}`"
            ));
        }
        let methods = extract_functions(block, TABULATED_FILE);
        let Some(dense) = methods.iter().find(|method| method.name == "dense_gemm") else {
            violations.push(format!("`{declaration}` has no `dense_gemm` method"));
            continue;
        };
        audit_tabulated_one_dot(&declaration, dense, violations);
        for token in OWNED_CARRIER_TOKENS {
            if methods.iter().any(|method| method.code.contains(token)) {
                violations.push(format!(
                    "`{declaration}` owns `{token}` storage instead of borrowing its offers"
                ));
            }
        }

        if element == "f32" {
            let table_spec = methods.iter().find(|method| method.name == "table_spec");
            if table_spec.is_none_or(|method| {
                let compact = method.body.split_whitespace().collect::<String>();
                !compact.contains("letmutspec=portable_table::<f32,Scaled64>(rows,group)")
                    || compact.matches("portable_table::<f32,Scaled64>(").count() != 1
                    || !compact.contains("let_=(backend,bound,block)")
                    || !compact.contains("spec.build_multiplies=false")
                    || !compact.contains("spec.build_adds=f32_q_build_presentations")
                    || !compact.contains("spec.lane_cap=u128::from(f32_q::COMPACT_CEILING)")
                    || compact.contains("choose_table(")
                    || compact.contains("gray_sign_table(")
                    || compact.contains("table_spec_modular(")
            }) {
                violations.push(
                    "`impl Tabulated for f32` does not bind its total q lane directly to the one portable TableBuild/gather graph"
                        .to_string(),
                );
            }
            let lane_scale = methods.iter().find(|method| method.name == "lane_scale");
            if lane_scale.is_none_or(|method| {
                let compact = method.body.split_whitespace().collect::<String>();
                [
                    "letmutnonfinite=false",
                    "a_span.see(code)",
                    "b_span.see(code)",
                    "ledger.decoded(visits)",
                    "ifnonfinite{u128::from(f32_q::COMPACT_CEILING)+1}",
                    "elseifk==1&&block==1",
                    "<Scaled64asLane<f32>>::mac(",
                    "project_f32_q(max_a.1,a_span.base())",
                    "project_f32_q(max_b.1,b_span.base())",
                    "f32_q_step_bound(wa+wb)",
                    "Some(LaneScale{base_a:a_span.base(),base_b:b_span.base(),per_step,})",
                ]
                .iter()
                .any(|witness| !compact.contains(witness))
                    || compact.contains("admits_atlas_scale")
                    || compact.contains("returnNone")
            }) {
                violations.push(
                    "`impl Tabulated for f32` does not derive a total exact-Q scale from every finite span and special singleton"
                        .to_string(),
                );
            }
            let prescale = methods.iter().find(|method| method.name == "prescale");
            if prescale.is_none_or(|method| {
                let compact = method.body.split_whitespace().collect::<String>();
                let signature = method.code.split_whitespace().collect::<String>();
                !signature.contains("fnprescale(x:Self,base:i32)->Self")
                    || !compact.contains("project_f32_q(x,base)")
                    || compact.contains("project_common_grade_f32")
            }) {
                violations.push(
                    "`impl Tabulated for f32` does not relabel each existing four-byte panel cell with its total contextual q projection"
                        .to_string(),
                );
            }
            let lanes = methods.iter().find(|method| method.name == "lanes");
            if lanes.is_none_or(|method| !method.body.contains("wrap_i64s_mut")) {
                violations.push(
                    "`impl Tabulated for f32` does not borrow Scaled64 words from the caller's \
                     narrow table offer"
                        .to_string(),
                );
            }
        } else if methods
            .iter()
            .any(|method| matches!(method.name.as_str(), "probe_capacity" | "lane_scale"))
        {
            violations.push(
                "`impl Tabulated for f64` categorically overrides the generic capacity/scale protocol instead of pricing downstream Enumerable codecs from their declarations"
                    .to_string(),
            );
        }
    }

    let scaled_impl = impl_block(table_raw, "impl Lane<f32> for Scaled64");
    let scaled_methods = scaled_impl
        .map(|block| extract_functions(block, TABLE_FILE))
        .unwrap_or_default();
    let scaled_mac = scaled_methods.iter().find(|method| method.name == "mac");
    if scaled_mac.is_none_or(|method| {
        let compact = method.body.split_whitespace().collect::<String>();
        let signature = method.code.split_whitespace().collect::<String>();
        !signature.contains("fnmac(self,a:f32,w:f32)->Self")
            || !compact
                .contains("letproduct=f32_q_product(decode_f32_q_factor(a),decode_f32_q_factor(w))")
            || !compact.contains("f32_q_add_words(self,product)")
            || compact.contains(".pack()")
            || compact.contains(".decode()")
    }) {
        violations.push(
            "`Scaled64::mac` does not consume the two in-place q cells through the total Atlas token product"
                .to_string(),
        );
    }
    if !tabulated_raw.contains("contextual q carrier")
        || !tabulated_raw.contains("self-describing finite/boundary tag")
        || !table_raw.contains("total contextual Atlas lane")
        || !table_raw.contains("not standalone IEEE operands")
        || !table_raw.contains("only that producer/consumer")
    {
        violations.push(
            "the f32 q producer/Scaled64 public trait signatures are not documented as one total contextual protocol"
                .to_string(),
        );
    }

    let product = functions.iter().find(|function| {
        function.rel == TABLE_FILE && function.name == "atlas_f32_q_magnitude_product_observed"
    });
    let mut scaled_lookups = 0usize;
    if let Some(product) = product {
        let compact = product.body.split_whitespace().collect::<String>();
        let terminus = functions.iter().find(|function| {
            function.rel == TABLE_FILE && function.name == "atlas_f32_q_magnitude_product"
        });
        scaled_lookups = usize::from(terminus.is_some_and(|function| {
            contains_call(&function.body, "atlas_f32_q_magnitude_product_observed")
                && function.body.contains("crate::lookup::i8_product")
        }));
        for witness in [
            "letleft=atlas_balanced_u32_octets(left)",
            "letright=atlas_balanced_u32_octets(right)",
            "ifleft.extent==0||right.extent==0{return0;}",
            "left.coordinates[..left.extent]",
            "right.coordinates[..right.extent]",
            "letgrade=left_grade+right_grade",
            "grades[grade]",
            "lookup(left_coordinate,right_coordinate)",
            "letgrade_extent=left.extent+right.extent-1",
            "letmutcoordinates=grades[..grade_extent].iter().rev()",
            ".next().expect(",
            "grade_observed()",
            "for&coordinateincoordinates",
            "for_in0..i8::BITS",
            "product.wrapping_add(product)",
            "product=product.wrapping_add(coordinate)",
        ] {
            if !compact.contains(witness) {
                violations.push(format!(
                    "{}:{}: extent-fractal q contraction lacks lookup/Horner witness `{witness}`",
                    product.rel, product.line
                ));
            }
        }
        if scaled_lookups == 0 {
            violations.push(format!(
                "{}:{}: the production q contraction does not terminate in the canonical i8 address lookup",
                product.rel, product.line
            ));
        }
        if compact.contains("letmutproduct=0")
            || compact.contains("grades.iter().rev()")
            || compact.contains("coordinates.iter()")
            || compact.matches("grade_observed()").count() != 2
        {
            violations.push(format!(
                "{}:{}: q contraction reintroduced a zero-initialized or fixed-extent Horner prefix",
                product.rel, product.line
            ));
        }
        for token in ["wrapping_mul", "checked_mul", "saturating_mul"] {
            if product.code.contains(token) {
                violations.push(format!(
                    "{}:{}: extent-fractal q contraction contains runtime value multiply `{token}`",
                    product.rel, product.line
                ));
            }
        }
    } else {
        violations.push(format!(
            "`{TABLE_FILE}` has no production extent-fractal q contraction; Scaled64's census is empty"
        ));
    }

    let projection = functions
        .iter()
        .find(|function| function.rel == TABULATED_FILE && function.name == "project_f32_q");
    if let Some(projection) = projection {
        let compact = projection.body.split_whitespace().collect::<String>();
        for witness in [
            "letraw:u32=bytemuck::cast::<f32,u32>(x)",
            "letsource_q=unsigned/fraction_place",
            "letfraction=unsigned%fraction_place",
            "letrelative=x.pack().exp.checked_sub(base)",
            "letq_field=atlas_double_u32(q,f32_q::SIGNIFICAND_BITS-1)",
            "bytemuck::cast::<u32,f32>(sign+q_field+fraction)",
        ] {
            if !compact.contains(witness) {
                violations.push(format!(
                    "{}:{}: total in-place q projection lacks `{witness}`",
                    projection.rel, projection.line
                ));
            }
        }
        for token in [
            "f32::from_bits",
            "f32::to_bits",
            "leading_zeros",
            "trailing_zeros",
            "wrapping_shl",
            "<<",
            ">>",
            " & ",
            " | ",
        ] {
            if projection.code.contains(token) {
                violations.push(format!(
                    "{}:{}: q projection retains traditional bit-field arithmetic `{token}`",
                    projection.rel, projection.line
                ));
            }
        }
    } else {
        violations.push(format!(
            "`{TABULATED_FILE}` has no direct in-place `project_f32_q`"
        ));
    }
    let compact_float = functions
        .iter()
        .filter(|function| function.rel == ATLAS_ENGINE_FILE)
        .map(|function| function.code.as_str())
        .collect::<Vec<_>>()
        .concat();
    if compact_float.contains("atlas_regrade_f32")
        || compact_float.contains("project_common_grade_f32")
        || compact_float.contains("admits_atlas_scale")
        || compact_float.contains("struct AtlasScale")
    {
        violations.push(format!(
            "`{ATLAS_ENGINE_FILE}` retains a superseded common-grade carrier beside total q"
        ));
    }

    let f32_block = impl_block(tabulated_raw, "impl Tabulated for f32").unwrap_or_default();
    if !f32_block
        .split_whitespace()
        .collect::<String>()
        .contains("spec.build_multiplies=false")
    {
        violations.push(
            "the specialized f32 table declaration does not price its Atlas lookup/add build as multiply-free"
                .to_string(),
        );
    }

    let tabulated_tests = extract_functions_including_tests(tabulated_raw, TABULATED_FILE);
    let table_tests = extract_functions_including_tests(table_raw, TABLE_FILE);
    let collapse_clock = tabulated_tests.iter().find(|function| {
        function.name
            == "ternary_radix_column_collapse_does_not_regress_the_retained_legacy_clock_cu_11"
    });
    if collapse_clock.is_none_or(|function| {
        let compact = function.body.split_whitespace().collect::<String>();
        [
            "assert_eq!(radix,legacy)",
            "assert_eq!(&radix_index[..n],&legacy_index[..n])",
            "constPAIRED_SAMPLES:usize=64",
            "constPAIRED_CHUNK_CALLS:usize=32",
            "constMIN_BATCH_TIME:std::time::Duration=std::time::Duration::from_millis(20)",
            "letmutbatch=1usize;loop{letradix_batch=measure_radix(&mutradix_index,batch);letlegacy_batch=measure_legacy(&mutlegacy_index,batch);ifradix_batch.min(legacy_batch)>=MIN_BATCH_TIME{break;}",
            "batch=batch.checked_mul(2)",
            "index.fill(poison);letmutobserved=expected_distinct;letstart=std::time::Instant::now();for_in0..batch{observed=std::hint::black_box(distinct_columns::<",
            "index.fill(poison);letmutobserved=expected_distinct;letstart=std::time::Instant::now();for_in0..batch{observed=std::hint::black_box(legacy_distinct_columns::<",
            "letelapsed=start.elapsed();assert_eq!(observed,expected_distinct);assert_eq!(&index[..n],expected_map.as_slice());elapsed",
            "forsamplein0..PAIRED_SAMPLES{radix_index.fill(poison);legacy_index.fill(poison)",
            "whilecompleted<batch{letcalls=(batch-completed).min(PAIRED_CHUNK_CALLS);letradix_first=(sample+chunk).is_multiple_of(2)",
            "run_radix(&mutradix_index,calls)",
            "run_legacy(&mutlegacy_index,calls)",
            "assert_eq!(radix_observed,expected_distinct);assert_eq!(legacy_observed,expected_distinct);assert_eq!(&radix_index[..n],expected_map.as_slice());assert_eq!(&legacy_index[..n],expected_map.as_slice())",
            "letlog_ratios=radix_samples.iter().zip(&legacy_samples)",
            "letmargin=2.0*(variance/count).sqrt()",
            "letupper_95=(mean_log+margin).exp()",
            "upper_95<=1.0",
        ]
        .iter()
        .any(|witness| !compact.contains(witness))
            || compact.matches("legacy_distinct_columns::<").count() < 3
            || compact.matches("index.fill(poison)").count() != 4
            || compact.matches("assert_eq!(observed,expected_distinct)").count() != 2
            || compact
                .matches("assert_eq!(&index[..n],expected_map.as_slice())")
                .count()
                != 2
            || compact.matches("for_in0..batch{").count() != 2
            || compact.matches("for_in0..calls{").count() != 2
            || compact.matches("measure_radix(&mutradix_index,batch)").count() != 1
            || compact.matches("measure_legacy(&mutlegacy_index,batch)").count() != 1
            || compact.matches("run_radix(&mutradix_index,calls)").count() != 2
            || compact.matches("run_legacy(&mutlegacy_index,calls)").count() != 2
            || compact
                .matches("letelapsed=start.elapsed();assert_eq!(observed,expected_distinct);assert_eq!(&index[..n],expected_map.as_slice());elapsed")
                .count()
                != 2
            || compact.contains("Duration::ZERO")
    }) {
        violations.push(
            "the retained column-collapse clock is not a paired, poisoned, post-checked 95% interval against the immutable legacy arm"
                .to_string(),
        );
    }
    let retained_hash = tabulated_tests
        .iter()
        .find(|function| function.name == "legacy_column_hash");
    let retained_collapse = tabulated_tests
        .iter()
        .find(|function| function.name == "legacy_distinct_columns");
    let retained_changed = retained_hash.is_none_or(|function| {
        let compact = function.body.split_whitespace().collect::<String>();
        [
            "constHASH_PREFIX:usize=16",
            "constSEED:u64=0xcbf2_9ce4_8422_2325",
            "constPRIME:u64=0x0000_0100_0000_01b3",
            "wrapping_mul(PRIME)",
            "rotate_left",
            "x^=x>>30",
        ]
        .iter()
        .any(|witness| !compact.contains(witness))
    }) || retained_collapse.is_none_or(|function| {
        let compact = function.body.split_whitespace().collect::<String>();
        !compact.contains("letmask=table-1")
            || !compact.contains("letmutprobe=hash&mask")
            || !compact.contains("probe=(probe+1)&mask")
            || !compact.contains("columns_equal::<E,Bd,C>(run,other)")
    });
    if retained_changed {
        violations.push(
            "the retained column-collapse comparator no longer has the immutable pre-refactor spelling"
                .to_string(),
        );
    }
    let radix_parity = table_tests.iter().find(|function| {
        function.name == "portable_radix_addresses_match_retained_bit_oracle_cu_11"
    });
    if radix_parity.is_none_or(|function| {
        let compact = function.body.split_whitespace().collect::<String>();
        !compact.contains("table_entry_address(offset,slab,rows)")
            || !compact.contains("offset&(slab-rows)")
            || !compact.contains("table_code_address(code,code_space,rows)")
            || !compact.contains("(code&(code_space-1))*rows")
            || !compact.contains("table_row_grade(rows),rows.trailing_zeros()")
            || !compact.contains("usize::MAX")
    }) {
        violations.push(
            "the portable q address graph lacks its independent retained-bit boundary differential"
                .to_string(),
        );
    }
    audit_f32_carrier_composition(&tabulated_tests, &table_tests, violations);
    audit_cd32_total_q_contract(
        tabulated_raw,
        functions,
        &tabulated_tests,
        &table_tests,
        violations,
    );
    let query = tabulated_tests.iter().find(|function| {
        function.name == "forced_f64_symbol_traversal_uses_the_complete_atlas_table_cd_20"
    });
    if query.is_none_or(|function| {
        let compact = function.body.split_whitespace().collect::<String>();
        !compact.contains("letsuggested=suggested_tabulation::<f64,Whole<f64>>(shape,8,1)")
            || !compact.contains("assert!(suggested>0")
            || !compact
                .contains("assert_eq!(suggested_tabulation_lanes::<f64,Whole<f64>>(shape,8,1),0")
            || !compact.contains("assert!(census.table_reads>0")
            || !compact.contains("assert_eq!(census.kernel_calls,0")
            || !compact.contains("assert_eq!(census.multiplies,0")
    }) {
        violations.push(
            "the parametric f64 complete-table declaration lacks an executable workspace/query/route coherence differential"
                .to_string(),
        );
    }

    let downstream = tabulated_tests.iter().find(|function| {
        function.name == "downstream_block_two_f64_codec_is_not_categorically_declined_cd_20"
    });
    if downstream.is_none_or(|function| {
        let compact = function.body.split_whitespace().collect::<String>();
        !compact.contains("letshape=Shape{m,k,n}")
            || !compact.contains("PairF64::MAX_BLOCK")
            || !compact.contains("suggested_tabulation::<f64,Whole<f64>>(shape,2,2)")
            || !compact.contains("traversal:Traversal::Blocked")
            || !compact.contains("assert!(census.table_reads>0")
            || !compact.contains("assert_eq!(census.kernel_calls,0")
            || !compact.contains("assert_eq!(census.multiplies,0")
            || compact.matches(".map(|value|value.to_bits())").count() < 2
    }) {
        violations.push(
            "the f64 table protocol lacks a downstream block-two codec witness against categorical decline"
                .to_string(),
        );
    }

    let repeated = tabulated_tests.iter().find(|function| {
        function.name == "repeated_block_one_symbols_are_built_once_per_addressed_index_cg_16"
    });
    if repeated.is_none_or(|function| {
        let compact = function.body.split_whitespace().collect::<String>();
        !compact.contains("fornin[D-1,D,D+1]")
            || !compact.contains("Tabulation::with_index(&mutlanes,&mutindex)")
            || !compact.contains("letcolumn_blocks=n.div_ceil(2)")
            || !compact.contains("assert_eq!(census.table_reads,contractions)")
            || !compact.contains("census.adds")
            || !compact.contains("census.decodes")
    }) {
        violations.push(
            "the distinct-address table route lacks its below/equal/above-space adversarial census"
                .to_string(),
        );
    }
    let shared_slot = tabulated_tests.iter().find(|function| {
        function.name == "shared_slot_indices_are_deduplicated_without_collapsing_columns_cg_16"
    });
    if shared_slot.is_none_or(|function| {
        let compact = function.body.split_whitespace().collect::<String>();
        !compact.contains("flat_map(|column|[3u8,columnasu8])")
            || !compact.contains("letbuilds=1+nasu64")
            || !compact.contains("letgathers=(k*n)asu64")
            || !compact.contains("assert_eq!(census.table_reads,gathers)")
            || !compact.contains("builds*f32_q_build_presentations(1,1,1)+gathers")
    }) {
        violations.push(
            "the addressed-entry set lacks a shared-slot witness independent of column collapse"
                .to_string(),
        );
    }
    let no_unused_set = tabulated_tests.iter().find(|function| {
        function.name == "non_pointwise_books_do_not_construct_an_unused_entry_set_cg_16"
    });
    if no_unused_set.is_none_or(|function| {
        let compact = function.body.split_whitespace().collect::<String>();
        !compact.contains("column_workspace(&mutdistinct,n,cols,false,false)")
            || !compact.contains("column_workspace(&mutrepeated,n,cols,true,false)")
            || !compact.contains("repeated[untouched_probe],marker")
    }) {
        violations.push(
            "the addressed-entry workspace lacks a no-work/no-clear witness for inapplicable books"
                .to_string(),
        );
    }
    let overflow_reuse = tabulated_tests.iter().find(|function| {
        function.name == "addressed_entry_set_overflow_collision_and_reuse_are_exact_cg_16"
    });
    if overflow_reuse.is_none_or(|function| {
        let compact = function.body.split_whitespace().collect::<String>();
        !compact.contains("&[0,1,2]).is_none()")
            || !compact.contains("&[3,3])")
            || !compact.contains("colliding.insert(9)")
            || !compact.contains("colliding.seen.iter().all")
    }) {
        violations.push(
            "the addressed-entry set lacks overflow-clear, collision, and immediate-reuse evidence"
                .to_string(),
        );
    }
    let short_index = tabulated_tests.iter().find(|function| {
        function.name == "short_index_offers_keep_duplicate_entry_work_truthful_cg_16"
    });
    if short_index.is_none_or(|function| {
        let compact = function.body.split_whitespace().collect::<String>();
        !compact.contains("run(None)")
            || !compact.contains("run(Some(1))")
            || !compact.contains("run(Some(suggested_tabulation_index(shape)))")
            || !compact.contains("assert_eq!(census.table_reads,4)")
            || !compact.contains("assert_eq!(complete_census.table_reads,1)")
    }) {
        violations.push(
            "the total no-index factorization lacks an absent/short/full duplicate-work census"
                .to_string(),
        );
    }
    let permuted = tabulated_tests.iter().find(|function| {
        function.name == "addressed_codec_preserves_a_nonidentity_enumeration_cg_16"
    });
    if permuted.is_none_or(|function| {
        let compact = function.body.split_whitespace().collect::<String>();
        !compact.contains("PermutedF32(table)")
            || !compact.contains("Some(&[0,1])")
            || !compact.contains("assert_eq!(scale.base_b")
    }) {
        violations.push(
            "the addressed-codec relabel lacks a nonidentity code_at/index_of witness".to_string(),
        );
    }

    (audited, scaled_lookups)
}

/// The f32 carrier has meaning only as one paired protocol. These executable
/// laws cross both public trait signatures, the private projection/product
/// helpers, final Laurent placement, and an independent product oracle.
fn audit_f32_carrier_composition(
    tabulated_tests: &[Function],
    table_tests: &[Function],
    violations: &mut Vec<String>,
) {
    let composition = tabulated_tests.iter().find(|function| {
        function.name == "f32_q_projection_and_scaled64_compose_as_one_atlas_protocol_cd_20"
    });
    if composition.is_none_or(|function| {
        let compact = function.body.split_whitespace().collect::<String>();
        [
            "<f32asTabulated>::prescale(a,base_a)",
            "<f32asTabulated>::prescale(w,base_b)",
            "<Scaled64asLane<f32>>::mac(Scaled64(0),projected_a,projected_w)",
            "<Scaled64asLane<f32>>::place_scaled(lane,<AccOf<f32>asAccumulator>::ZERO,base_a+base_b,)",
            "<f32asElement>::mac(&mutwant,a,w)",
            "assert_eq!(got,want",
            "f32_q_build_presentations(1,1,1),1",
            "f32_q_build_presentations(3,5,7),105",
        ]
        .iter()
        .any(|witness| !compact.contains(witness))
    }) {
        violations.push(
            "the contextual f32 q-projection/Scaled64 pair lacks its complete Laurent composition law or one-presentation census law"
                .to_string(),
        );
    }

    let boundary = table_tests.iter().find(|function| {
        function.name == "balanced_octets_match_the_independent_wide_product_cd_20"
    });
    if boundary.is_none_or(|function| {
        let compact = function.body.split_whitespace().collect::<String>();
        !compact.contains("forleftini8::MIN..=i8::MAX")
            || !compact.contains("forrightini8::MIN..=i8::MAX")
            || !compact.contains("atlas_octet_product([left,0,0,0],[right,0,0,0],)")
            || !compact.contains("i64::from(left)*i64::from(right)")
            || !compact.contains("i64::from(i32::MIN)")
            || !compact.contains("i64::from(i32::MAX)")
    }) {
        violations.push(
            "the contextual f32 carrier lacks an exhaustive signed-octet and coefficient-boundary product law"
                .to_string(),
        );
    }
}

/// Bind CD-32's executable laws to the shipped total-q producer and scheduler.
/// Model arithmetic, kernel token laws, driver bytes, and the source recurrence
/// are independent witnesses; omitting any one would leave a self-comparison.
fn audit_cd32_total_q_contract(
    tabulated_raw: &str,
    functions: &[Function],
    tabulated_tests: &[Function],
    table_tests: &[Function],
    violations: &mut Vec<String>,
) {
    let lane = tabulated_tests
        .iter()
        .find(|function| function.name == "total_f32_lane_scale_uses_the_exact_q_capacity_cd_32");
    let lane_witnesses = [
        "scale.per_step,u128::from(f32_q::PRODUCT_BOUND)",
        "<f32asTabulated>::lane_run::<Scaled64>(0,&scale),Some(usize::try_from(f32_q::ZERO_SPAN_CAPACITY).unwrap())",
        "let(scale,census)=observe(&[1.0],&[0],&[f32::NAN])",
        "<f32asTabulated>::lane_run::<Scaled64>(0,&scale),Some(1)",
        "letextremes=[f32::from_bits(1),f32::MAX]",
        "let(scale,census)=observe(&extremes,&[0,1],&extremes)",
    ];
    let missing_lane = lane.and_then(|function| {
        let compact = function.body.split_whitespace().collect::<String>();
        lane_witnesses
            .iter()
            .find(|witness| !compact.contains(**witness))
            .copied()
    });
    let runtime_model =
        lane.is_some_and(|function| function.body.contains("Model::load_from_repo_root"));
    if lane.is_none() || missing_lane.is_some() || runtime_model {
        violations.push(format!(
            "CD-32 lacks the production-side exact-Q, non-finite, and full-span lane-capacity differential{}",
            missing_lane.map_or_else(
                || {
                    if runtime_model {
                        " (the target test reopens the repository model)".to_string()
                    } else {
                        String::new()
                    }
                },
                |witness| format!(" `{witness}`"),
            )
        ));
    }

    let total = tabulated_tests
        .iter()
        .find(|function| function.name == "total_f32_q_carrier_executes_every_ieee_boundary_cd_32");
    if total.is_none_or(|function| {
        let compact = function.body.split_whitespace().collect::<String>();
        [
            "letexponent_symbols:[f32;256]=core::array::from_fn",
            "letexponent_codes:Vec<u8>=(0u8..=u8::MAX).collect()",
            "letunion_symbols=[f32::INFINITY,f32::NEG_INFINITY,f32::NAN,0.0]",
            "0u8,3,3",
            "3,1,3",
            "3,3,2",
            "0,1,3",
            "0,3,2",
            "3,1,2",
            "0,1,2",
            "census.table_reads>0",
            "assert_eq!(census.kernel_calls,0",
            "assert_eq!(census.multiplies,0",
        ]
        .iter()
        .any(|witness| !compact.contains(witness))
    }) {
        violations.push(
            "CD-32 lacks its complete exponent-field and seven-union resident-table differential"
                .to_string(),
        );
    }

    let special = tabulated_tests.iter().find(|function| {
        function.name == "f32_q_special_atoms_are_immediate_source_order_singletons_cd_32"
    });
    if special.is_none_or(|function| {
        let compact = function.body.split_whitespace().collect::<String>();
        [
            "letcodes=[0u8,1,2,3,0]",
            "for(&activation,&code)inactivations.iter().zip(&codes)",
            "f32::mac(&mutexpected,activation,symbols[usize::from(code)])",
            "assert_eq!(captured.get(),expected",
            "assert_eq!(census.table_reads,5",
            "assert_eq!(census.kernel_calls,0",
            "assert_eq!(census.multiplies,0",
        ]
        .iter()
        .any(|witness| !compact.contains(witness))
    }) {
        violations.push(
            "CD-32 lacks its source-ordered finite/special/finite Complete-accumulator differential"
                .to_string(),
        );
    }

    let raw = tabulated_raw.split_whitespace().collect::<String>();
    let fracture = tabulated_tests
        .iter()
        .find(|function| function.name == "f32_q_lane_scalar_fractures_a_wider_codec_block_cd_32");
    if !raw.contains("structFractureF32")
        || !raw.contains("implCodec<f32,Whole<f32>>forFractureF32")
        || !raw.contains("constMAX_BLOCK:usize=2")
        || !raw.contains("constCODE_SPACE:usize=2")
        || fracture.is_none_or(|function| {
            let compact = function.body.split_whitespace().collect::<String>();
            function.body.contains("Model::load_from_repo_root")
                || [
                    "letlow=f32::from_bits(0x3fff_ffff)",
                    "lethigh=f32::from_bits(0x437f_ffff)",
                    "letcompact_ceiling=u128::from(f32_q::COMPACT_CEILING)",
                    "letproduct_bound=u128::from(f32_q::PRODUCT_BOUND)",
                    "f32_q_lane_capacity",
                    "1<FractureF32::MAX_BLOCK",
                    "is_some_and(|whole|whole>compact_ceiling)",
                    "Traversal::Tabulated",
                    "forced.table_reads,(m*FractureF32::CODE_SPACE*k)asu64",
                    "assert_eq!(forced.adds,24",
                    "assert_eq!(forced.decodes,44",
                    "assert_eq!(forced.kernel_calls,0",
                    "assert_eq!(forced.multiplies,0",
                    "assert_eq!(finite_signature,special_signature",
                    "assert!(finite_signature.0",
                ]
                .iter()
                .any(|witness| !compact.contains(witness))
        })
    {
        violations.push(
            "CD-32 lacks the unsafe-whole-block scalar-fracture and nonvacuous value-blind route differential"
                .to_string(),
        );
    }

    let generated_kernel_model = table_tests
        .iter()
        .find(|function| function.name == "generated_model");
    if table_tests
        .iter()
        .any(|function| function.body.contains("Model::load_from_repo_root"))
        || generated_kernel_model.is_none_or(|function| {
            let compact = function.body.split_whitespace().collect::<String>();
            [
                "F32QCarrier{",
                "significand_bits:f32_q::SIGNIFICAND_BITS",
                "product_bound:f32_q::PRODUCT_BOUND",
                "state_count:f32_q::STATE_COUNT",
                "compact_ceiling:f32_q::COMPACT_CEILING",
                "zero_span_capacity:f32_q::ZERO_SPAN_CAPACITY",
                "nonfinite_states:f32_q::STATE_COUNT-f32_q::SIGNED_FINITE_STATES",
            ]
            .iter()
            .any(|witness| !compact.contains(witness))
        })
    {
        violations.push(
            "CD-32 kernel differentials do not use the generated target-independent q model"
                .to_string(),
        );
    }

    for (test, label, witnesses) in [
        (
            "empty_f32_q_reduction_has_zero_work_cd_32",
            "zero-depth totality/Census",
            &[
                "Shape{m:3,k:0,n:5}",
                "assert_eq!(census,Census::default())",
                "assert_eq!(got,vec![0.0;15])",
            ][..],
        ),
        (
            "parametric_nonpower_q_blocks_preserve_bytes_strides_offers_and_census_cd_32",
            "non-power block/space, tail, stride, and offer",
            &[
                "exercise::<3,3>()",
                "exercise::<5,5>()",
                "Shape{m:3,k:2*B,n:7,}",
                "letblocks=shape.k/B",
                "u8::try_from((j+p)%D)",
                "forofferin[0,1,usize::MAX]",
                "Strides{rs:row_strideasisize,cs:1,}",
                "got.iter().map(|value|value.to_bits())",
                "want.iter().map(|value|value.to_bits())",
                "assert_eq!(census.kernel_calls,0",
                "assert_eq!(census.multiplies,0",
                "assert!(census.table_reads>0",
            ][..],
        ),
    ] {
        let found = tabulated_tests
            .iter()
            .find(|function| function.name == test);
        let lacks_parametric_codec = test
            == "parametric_nonpower_q_blocks_preserve_bytes_strides_offers_and_census_cd_32"
            && (!raw.contains("structParametricFractureF32<constB:usize,constD:usize>")
                || !raw.contains("impl<constB:usize,constD:usize>Codec<f32,Whole<f32>>forParametricFractureF32<B,D>")
                || !raw.contains("constMAX_BLOCK:usize=B")
                || !raw.contains("constCODE_SPACE:usize=D"));
        if lacks_parametric_codec
            || found.is_none_or(|function| {
                let compact = function.body.split_whitespace().collect::<String>();
                witnesses.iter().any(|witness| !compact.contains(*witness))
            })
        {
            violations.push(format!("CD-32 lacks its {label} differential"));
        }
    }

    for (test, label, witnesses) in [
        (
            "q_precision_fractal_matches_wide_product_and_exact_work_cd_32",
            "extent-fractal lookup/Horner work",
            &[
                "letmodel=generated_model()",
                "expected_lookups=left_extent*right_extent",
                "left_extent+right_extent-1",
                "(one_lookups.get(),one_grades.get()),(1,1)",
                "(max_lookups.get(),max_grades.get()),(16,7)",
            ][..],
        ),
        (
            "mixed_nonfinite_and_finite_words_scalar_fracture_cd_32",
            "mixed finite/special split and special union",
            &[
                "letmodel=generated_model()",
                "LaneWord>::add(special,finite)",
                "LaneWord>::add(finite,special)",
                "q.signed_finite_states+union-1",
            ][..],
        ),
        (
            "scaled64_zero_is_raw_identity_for_every_token_class_cd_32",
            "raw public-lane zero identity",
            &[
                "letmodel=generated_model()",
                "LaneWord>::add(zero,word)",
                "LaneWord>::add(word,zero)",
                "Scaled64(i64::MIN)",
                "Scaled64(i64::MAX)",
            ][..],
        ),
    ] {
        let found = table_tests.iter().find(|function| function.name == test);
        if found.is_none_or(|function| {
            let compact = function.body.split_whitespace().collect::<String>();
            witnesses.iter().any(|witness| !compact.contains(*witness))
        }) {
            violations.push(format!("CD-32 lacks its kernel {label} differential"));
        }
    }

    let required_production = [
        (
            "run_lane",
            "dynamic q lane seam",
            &[
                "letobserved_run=E::lane_run::<L>(Bd::VALUE,&scale)",
                "letdata_dependent_lane=lane_capacity.is_none()&&observed_run.is_some()",
                "letlocal_envelopes=data_dependent_lane&&observed_run.is_some_and(|run|run<shape.k)",
                "letprojection_decodes=data_dependent_lane",
            ][..],
        ),
        (
            "scalar_envelope",
            "least per-slot L-infinity certificate",
            &[
                "subview(row0,source,rows,1)",
                "ifcollapsed.is_some_and(|first|first[j]!=j){continue;}",
                "CodedMatrix::new(scalar,1,1,core::slice::from_ref(code))",
                "envelope=envelope.max(regrade_envelope(local,call_scale,cap))",
            ][..],
        ),
        (
            "regrade_envelope",
            "common-base certificate regrading",
            &[
                "letsingleton=cap+1",
                "local.base_a.checked_sub(call.base_a)",
                "local.base_b.checked_sub(call.base_b)",
                "ifbound>cap||bound>cap-bound{returnsingleton;}",
                "bound+=bound",
            ][..],
        ),
        (
            "row_tile",
            "source-ordered maximal-prefix scheduler",
            &[
                "letmutheight=0u128",
                "letmutpending=false",
                "ifbound>cap||block_bound>cap-bound{block_bound=cap+1;break;}",
                "ifpending&&block_bound>cap-height{place(carried,acc,placed,scale.exponent());",
                "height+=block_bound",
                "letsingleton=bound>cap",
                "ifpending&&(singleton||bound>cap-height)",
                "ifsingleton{",
                "pending=false",
                "height=0",
                "height+=bound",
            ][..],
        ),
    ];
    for (name, label, witnesses) in required_production {
        let found = functions
            .iter()
            .find(|function| function.rel == TABULATED_FILE && function.name == name);
        if found.is_none_or(|function| {
            let compact = function.body.split_whitespace().collect::<String>();
            witnesses.iter().any(|witness| !compact.contains(*witness))
        }) {
            violations.push(format!("CD-32 production lacks its {label}"));
        }
    }

    for forbidden in ["[u128;", "Vec<u128>", "MAX_FRACTURE", "FRACTURE_LIMIT"] {
        for function in functions.iter().filter(|function| {
            function.rel == TABULATED_FILE
                && matches!(
                    function.name.as_str(),
                    "scalar_envelope" | "regrade_envelope" | "row_tile" | "build_source_scalar"
                )
        }) {
            if function.code.contains(forbidden) {
                violations.push(format!(
                    "{}:{}: scalar fracture retains arbitrary bound storage/limit `{forbidden}`",
                    function.rel, function.line
                ));
            }
        }
    }
}

/// The model bounds the same unreduced recurrence production executes. The
/// full source length is the initial address-sized coordinate; the former
/// dictionary residue would prove a smaller object even though both spellings
/// happen to reach the same final modulus.
fn audit_column_hash_model_source(model_raw: &str, violations: &mut Vec<String>) {
    let functions = extract_functions(model_raw, MODEL_DERIVE_FILE);
    let bound = functions
        .iter()
        .find(|function| function.name == "column_hash_accumulator_bound");
    if bound.is_none_or(|function| {
        let compact = function.body.split_whitespace().collect::<String>();
        [
            "letSome(index_cardinality)=power_of_two(address_bits)",
            "letcoordinate=index_cardinality-1",
            "letmutbound=coordinate",
            "bound.checked_mul(modalityasu128)",
            "radix_term.checked_add(coordinate)",
        ]
        .iter()
        .any(|witness| !compact.contains(witness))
            || compact.contains("dictionary_cardinality")
    }) {
        violations.push(
            "the column-hash model does not bound the full unreduced usize length coordinate"
                .to_string(),
        );
    }

    let compact = model_raw.split_whitespace().collect::<String>();
    for witness in [
        "assert_eq!(bound,1_191_107_759_025_695_718_254_230_815)",
        "assert_ne!(bound,794_071_836_276_006_466_529_705_247",
        "assert_eq!(unsigned_bits(bound),90)",
    ] {
        if !compact.contains(witness) {
            violations.push(format!(
                "the column-hash model regression lacks exact full-length witness `{witness}`"
            ));
        }
    }
}

/// The runtime census is the independent witness for page reuse and attempted
/// dense presentations. Static placement proves where the counter moves; these
/// differentials prove its closed form over caller pages, the model page, a
/// first decline, and the zero-depth public stream.
/// Float q reaches the column dictionary and the portable table gathers. These
/// addresses are part of the live arithmetic graph, so clean projection and
/// contraction are insufficient if hashing or wrapping reintroduces packed
/// shifts, masks, XOR mixing, or a fitted multiplier downstream.
fn audit_tabulated_radix_addresses(functions: &[Function], violations: &mut Vec<String>) {
    let required = [
        (
            TABULATED_FILE,
            "column_hash",
            &[
                "letmuthash=run.len()asu128",
                "letmeasured=run.len().min(crate::float::COLUMN_HASH_PREFIX)",
                "for&codein&run[..measured]",
                "letdoubled=hash+hash",
                "hash=doubled+hash+C::index_of(code)asu128",
                "(hash%modulusasu128)asusize",
            ][..],
        ),
        (
            TABULATED_FILE,
            "distinct_columns",
            &[
                "letmutprobe=hash",
                "probe+=1",
                "ifprobe==table{probe=0;}",
                "key[probe]==hash&&columns_equal",
            ][..],
        ),
        (
            TABULATED_FILE,
            "insert",
            &[
                "letextent=self.seen.len()",
                "letmutprobe=ifindex<extent{index}else{index%extent}",
                "probe+=1",
                "ifprobe==extent{probe=0;}",
            ][..],
        ),
        (
            TABLE_FILE,
            "table_entry_address",
            &["letwithin=offset%slab", "within-within%rows"][..],
        ),
        (
            TABLE_FILE,
            "table_code_address",
            &["(code%code_space)*rows"][..],
        ),
        (
            TABLE_FILE,
            "table_row_grade",
            &["whilerows>1{rows/=2;grade+=1;}"][..],
        ),
        (
            TABLE_FILE,
            "codes_any",
            &[
                "letcode_space=slab/rows",
                "table_code_address(codes[at].into(),code_space,rows)",
            ][..],
        ),
        (
            TABLE_FILE,
            "gather_run",
            &["table_entry_address(atasusize,slab,R)"][..],
        ),
        (
            TABLE_FILE,
            "gather_any",
            &["table_entry_address(atasusize,slab,rows)"][..],
        ),
        (
            TABLE_FILE,
            "codes_run",
            &[
                "(slab_arg,slab_arg/R)",
                "table_code_address(codes[at].into(),code_space,R)",
            ][..],
        ),
        (
            TABLE_FILE,
            "portable_table",
            &[
                "build:portable_build::<E,L>",
                "portable_gather::<L>",
                "portable_gather_codes::<L,u16>",
                "portable_gather_codes::<L,u8>",
                "portable_gather_wide::<L>",
                "portable_gather_codes_wide::<L,u16>",
                "portable_gather_codes_wide::<L,u8>",
            ][..],
        ),
        (
            TABLE_FILE,
            "portable_build",
            &["build_run::<1,E,L>", "build_any(rows,block,book,acts,out)"][..],
        ),
        (TABLE_FILE, "build_run", &["*cell=cell.mac(a,w)"][..]),
        (TABLE_FILE, "build_any", &["*cell=cell.mac(a,w)"][..]),
        (
            TABLE_FILE,
            "portable_gather",
            &[
                "gather_any(rows,group,slab,stack,off,lane)",
                "gather_run::<C,R,G,L>",
            ][..],
        ),
        (
            TABLE_FILE,
            "portable_gather_codes",
            &[
                "codes_any(rows,depth,slab,shift,stack,codes,stride,lane)",
                "codes_run::<C,R,G,L,K>",
            ][..],
        ),
        (
            TABLE_FILE,
            "portable_gather_wide",
            &["gather_any(rows,group,slab,stack,off,lane)"][..],
        ),
        (
            TABLE_FILE,
            "portable_gather_codes_wide",
            &["codes_any(rows,depth,slab,shift,stack,codes,stride,lane)"][..],
        ),
    ];
    for (rel, name, witnesses) in required {
        let found = functions
            .iter()
            .find(|function| function.rel == rel && function.name == name);
        if found.is_none_or(|function| {
            let compact = function.body.split_whitespace().collect::<String>();
            witnesses.iter().any(|witness| !compact.contains(*witness))
        }) {
            violations.push(format!(
                "the float-reachable radix address graph lacks `{rel}::{name}`"
            ));
        }
    }

    for (rel, name) in [
        (TABULATED_FILE, "column_hash"),
        (TABULATED_FILE, "distinct_columns"),
        (TABULATED_FILE, "insert"),
        (TABLE_FILE, "table_entry_address"),
        (TABLE_FILE, "table_code_address"),
        (TABLE_FILE, "table_row_grade"),
        (TABLE_FILE, "codes_any"),
        (TABLE_FILE, "gather_run"),
        (TABLE_FILE, "gather_any"),
        (TABLE_FILE, "codes_run"),
        (TABLE_FILE, "portable_table"),
        (TABLE_FILE, "portable_build"),
        (TABLE_FILE, "build_run"),
        (TABLE_FILE, "build_any"),
        (TABLE_FILE, "portable_gather"),
        (TABLE_FILE, "portable_gather_codes"),
        (TABLE_FILE, "portable_gather_wide"),
        (TABLE_FILE, "portable_gather_codes_wide"),
    ] {
        if let Some(function) = functions
            .iter()
            .find(|function| function.rel == rel && function.name == name)
        {
            let compact = function.body.split_whitespace().collect::<String>();
            for token in [
                "wrapping_mul",
                "rotate_left",
                "rotate_right",
                "trailing_zeros",
                "SEED",
                "PRIME",
                "min(16)",
                "&mask",
                ">&mask",
                "&=",
                "&(slab",
                "&(code_space",
                "^",
                "<<",
                ">>",
                "hash*",
                "*hash",
            ] {
                if compact.contains(token) {
                    violations.push(format!(
                        "{}:{}: float-reachable address `{name}` retains legacy token `{token}`",
                        function.rel, function.line
                    ));
                }
            }
        }
    }

    for name in ["column_hash", "insert"] {
        if let Some(function) = functions
            .iter()
            .find(|function| function.rel == TABULATED_FILE && function.name == name)
        {
            if function.body.contains('*') {
                violations.push(format!(
                    "{}:{}: float-reachable address `{name}` retains a traditional multiply",
                    function.rel, function.line
                ));
            }
        }
    }
}

fn audit_tabulated_stream_differentials(functions: &[Function], violations: &mut Vec<String>) {
    let required = [
        (
            "every_float_panel_offer_is_atlas",
            &[
                "letpacked=traversal!=Traversal::OutputMajor&&offer==n*k+k",
                "letsource_page=ifoffer==0{blocking::KC}else{offer}",
                "elseifsource_page>=k{n*m.div_ceil(ROW_TILES[0])}",
                "m*n*k.div_ceil(source_page)",
                "assert_eq!(census.kernel_calls,expected_callsasu64",
            ][..],
        ),
        (
            "an_accepted_first_partial_is_not_recomputed_cd_20",
            &[
                "m*n*k.div_ceil(blocking::KC)",
                "assert_eq!(census.kernel_calls,(m*n*k.div_ceil(blocking::KC))asu64)",
                "assert_eq!(census.adds,(m*n*(k.div_ceil(blocking::KC)-1))asu64",
                "assert_eq!(census.multiplies,0)",
            ][..],
        ),
        (
            "a_declined_first_partial_preserves_the_ordinary_stream_cd_20",
            &[
                "assert_eq!(DENSE_CALLS.load(Ordering::Relaxed),1",
                "assert_eq!(census.kernel_calls,1",
            ][..],
        ),
        (
            "an_empty_float_reduction_uses_the_public_stream_zero_cd_20",
            &[
                "assert_eq!(census.decodes,0)",
                "assert_eq!(census.multiplies,0)",
                "assert_eq!(census.kernel_calls,0)",
            ][..],
        ),
        (
            "atlas_stream_retains_wide_and_crosses_pages_exactly_cd_20",
            &[
                "family!(f32)",
                "family!(f64)",
                "crate::float::atlas_dot_resolutions(GemmOptions::default().backend),1",
            ][..],
        ),
    ];
    for (name, witnesses) in required {
        let Some(function) = functions
            .iter()
            .find(|function| function.rel == TABULATED_FILE && function.name == name)
        else {
            violations.push(format!(
                "`{TABULATED_FILE}` has no executable page-ledger differential `{name}`"
            ));
            continue;
        };
        let compact = function.body.split_whitespace().collect::<String>();
        for witness in witnesses {
            if !compact.contains(witness) {
                violations.push(format!(
                    "{}:{}: page-ledger differential `{name}` lacks `{witness}`",
                    function.rel, function.line
                ));
            }
        }
    }

    let reuse = functions.iter().find(|function| {
        function.rel == TABULATED_FILE
            && function.name == "table_padding_is_zeroed_once_and_reused_at_the_same_geometry_cd_13"
    });
    if reuse.is_none_or(|function| {
        let compact = function.body.split_whitespace().collect::<String>();
        !compact.contains("letmutwords=vec![-1i32;slab*depth]")
            || !compact.contains("Table::new(&mutwords,space,rows,depth)")
            || !compact.contains("Table::reuse_zeroed(&mutwords,space,rows,depth)")
            || !compact.contains(".all(|&word|word!=0)")
            || compact.matches(".all(|&word|word==0)").count() < 2
    }) {
        violations.push(
            "the private table-stack reuse lacks its poisoned-live/zero-padding executable law"
                .to_string(),
        );
    }

    let tabulate = functions
        .iter()
        .find(|function| function.rel == TABULATED_FILE && function.name == "tabulate");
    if tabulate.is_none_or(|function| {
        let compact = function.body.split_whitespace().collect::<String>();
        !compact.contains("letmutzeroed_rows=None")
            || !compact.contains(
                "ifzeroed_rows==Some(rows){Table::reuse_zeroed(stack,space,rows,plan.depth)",
            )
            || !compact.contains("lettable=Table::new(stack,space,rows,plan.depth)")
            || !compact.contains("zeroed_rows=Some(rows)")
    }) {
        violations.push(
            "the resident table stack is not reborrowed only at identical row geometry".to_string(),
        );
    }
}

fn audit_tabulated_one_dot(declaration: &str, dense: &Function, violations: &mut Vec<String>) {
    if !contains_call(&dense.body, "gemm_float") {
        violations.push(format!(
            "`{declaration}`'s dense factorization does not re-enter `gemm_float`"
        ));
    }
    let compact = dense.body.split_whitespace().collect::<String>();
    for witness in [
        "ifdense.shape().m==1&&dense.shape().n==1",
        "letmutacc=<AccOf<Self>asAccumulator>::ZERO",
        "accumulate_atlas_dot(",
        "dense.shape().k",
        "PanelFacts::UNKNOWN",
        "options.backend",
        "epilogue.finish(acc,prior,options.encode)",
        "returntrue",
    ] {
        if !compact.contains(witness) {
            violations.push(format!(
                "`{declaration}`'s API-neutral one-dot bridge lacks `{witness}`"
            ));
        }
    }
    let atlas_at = compact.find("accumulate_atlas_dot(");
    let tiled_at = compact.rfind("gemm_float(");
    if atlas_at.is_none()
        || tiled_at.is_none()
        || atlas_at >= tiled_at
        || compact.matches("accumulate_atlas_dot(").count() != 1
    {
        violations.push(format!(
            "`{declaration}` does not enter Atlas exactly once for a one-output dot before its \
             tiled dense spelling"
        ));
    }
    for forbidden in ["E::mac(", "Self::mac(", "dot_ref(", "dot_lane("] {
        if compact.contains(forbidden) {
            violations.push(format!(
                "`{declaration}`'s one-dot bridge reaches scalar escape `{forbidden}`"
            ));
        }
    }
}

/// The stable public `Tabulated` API keeps its concrete `Wide<Complete>` stream
/// lane. The optimized float decline borrows the family's existing dense Atlas
/// factorization through its first real source partial: acceptance retains that
/// useful partial, rejection happens before caller `C` is touched, and no dummy
/// capability product or private associated-type replacement exists.
fn audit_tabulated_atlas_stream(
    source: &str,
    functions: &[Function],
    violations: &mut Vec<String>,
) {
    let compact_source = source.split_whitespace().collect::<String>();
    for removed in [
        "modatlas_stream",
        "AtlasStreamLedger",
        "DenseCapabilityProbe",
        "letprobe=",
        "dense_chunked_route",
        "structCapture<",
    ] {
        if compact_source.contains(removed) {
            violations.push(format!(
                "`{TABULATED_FILE}` retains rejected private/dummy stream adapter `{removed}`"
            ));
        }
    }
    for element in ["f32", "f64"] {
        let declaration = format!("impl Tabulated for {element}");
        let Some(block) = impl_block(source, &declaration) else {
            violations.push(format!("`{TABULATED_FILE}` has no `{declaration}`"));
            continue;
        };
        let compact = block.split_whitespace().collect::<String>();
        let stream_lane = format!("Wide<AccOf<{element}>>");
        if !compact.contains(&format!("typeStreamLane={stream_lane};")) {
            violations.push(format!(
                "`{declaration}` does not preserve its public stream extension point as \
                 `{stream_lane}`"
            ));
        }
    }

    let Some(decline) = functions
        .iter()
        .find(|function| function.rel == TABULATED_FILE && function.name == "decline")
    else {
        violations.push(format!(
            "`{TABULATED_FILE}` has no `decline`; empty float offers are unaudited"
        ));
        return;
    };
    let decline_compact = decline.body.split_whitespace().collect::<String>();
    let packed_at = decline_compact.find("packed_route(");
    let atlas_at = decline_compact.find("atlas_stream_route(");
    let stream_at = decline_compact.rfind("stream(");
    if packed_at.is_none()
        || atlas_at.is_none()
        || stream_at.is_none()
        || packed_at >= atlas_at
        || atlas_at >= stream_at
        || decline_compact.matches("atlas_stream_route(").count() != 1
    {
        violations.push(format!(
            "{}:{}: `decline` does not preserve packed admission, one Atlas capability question, \
             and the total stream in that order",
            decline.rel, decline.line
        ));
    }

    let Some(route) = functions
        .iter()
        .find(|function| function.rel == TABULATED_FILE && function.name == "atlas_stream_route")
    else {
        violations.push(format!(
            "`{TABULATED_FILE}` has no API-neutral Atlas stream capability route"
        ));
        return;
    };
    let route_compact = route.body.split_whitespace().collect::<String>();
    for witness in [
        "ifBd::VALUE!=u128::MAX||triple.shape().k==0{returnfalse;}",
        "dense_stream(triple,epilogue,options,panel,ledger)",
    ] {
        if !route_compact.contains(witness) {
            violations.push(format!(
                "{}:{}: the Atlas stream route lacks first-real-partial witness `{witness}`",
                route.rel, route.line
            ));
        }
    }
    let prefilter_at =
        route_compact.find("ifBd::VALUE!=u128::MAX||triple.shape().k==0{returnfalse;}");
    let partial_at = route_compact.find("dense_stream(");
    if prefilter_at.is_none() || partial_at.is_none() || prefilter_at >= partial_at {
        violations.push(format!(
            "{}:{}: the type-level bound prefilter must precede the real dense partial walk",
            route.rel, route.line
        ));
    }
    for forbidden in [
        "E::mac(",
        "E::dense_gemm(",
        "Alphabet::<E,Bd>::ZERO",
        "ledger.kernelled(",
        "ledger.multiplied(",
    ] {
        if route_compact.contains(forbidden) {
            violations.push(format!(
                "{}:{}: the Atlas route contains dummy or synthetic operation \
                 `{forbidden}`",
                route.rel, route.line
            ));
        }
    }

    if !compact_source.contains("structDenseCapture<'a,A>(&'acore::cell::Cell<A>);") {
        violations.push("the dense-partial capture is not one borrowed `Cell`".to_string());
    }
    let Some(capture_impl) = impl_block(
        source,
        "impl<E: Element, O: Element> Epilogue<E, O> for DenseCapture",
    ) else {
        violations.push("the dense-partial capture has no exact epilogue".to_string());
        return;
    };
    let capture_methods = extract_functions(capture_impl, TABULATED_FILE);
    let finish = capture_methods
        .iter()
        .find(|method| method.name == "finish");
    let reads_c = capture_methods
        .iter()
        .find(|method| method.name == "reads_c");
    if finish.is_none_or(|method| {
        method.body.split_whitespace().collect::<String>() != "self.0.set(acc);O::ZERO"
    }) || reads_c.is_none_or(|method| method.body.trim() != "false")
    {
        violations.push(
            "the dense-partial capture does not transfer exactly one accumulator without \
             reading or encoding caller output"
                .to_string(),
        );
    }

    let Some(dot) = functions
        .iter()
        .find(|function| function.rel == TABULATED_FILE && function.name == "dense_stream_dot")
    else {
        violations.push(format!(
            "`{TABULATED_FILE}` has no one-partial dense stream adapter"
        ));
        return;
    };
    let dot_compact = dot.body.split_whitespace().collect::<String>();
    for witness in [
        "ifb.is_empty(){returnSome(<AccOf<E>asAccumulator>::ZERO);}",
        "letright=MatView::row_major(b,b.len(),1)",
        "letcaptured=core::cell::Cell::new(<AccOf<E>asAccumulator>::ZERO)",
        "ledger.kernelled();",
        "E::dense_gemm(a,right,output,&DenseCapture(&captured),options,&mut[])",
        "ran.then(||captured.into_inner())",
    ] {
        if !dot_compact.contains(witness) {
            violations.push(format!(
                "{}:{}: the one-partial adapter lacks real-source witness `{witness}`",
                dot.rel, dot.line
            ));
        }
    }
    for forbidden in ["E::mac(", "dot_ref(", "letprobe=", "DenseCapabilityProbe"] {
        if dot_compact.contains(forbidden) {
            violations.push(format!(
                "{}:{}: the one-partial adapter reaches dummy/scalar operation `{forbidden}`",
                dot.rel, dot.line
            ));
        }
    }
    let empty_at = dot_compact.find("ifb.is_empty(){returnSome(");
    let ledger_at = dot_compact.find("ledger.kernelled();");
    let dense_at = dot_compact.find("E::dense_gemm(");
    if !matches!((empty_at, ledger_at, dense_at), (Some(empty), Some(ledger), Some(dense)) if empty < ledger && ledger < dense)
        || dot_compact.matches("ledger.kernelled();").count() != 1
    {
        violations.push(format!(
            "{}:{}: the actual dense presentation is not counted exactly once, after the \
             zero-depth return and immediately before `E::dense_gemm`",
            dot.rel, dot.line
        ));
    }

    let Some(dense) = functions
        .iter()
        .find(|function| function.rel == TABULATED_FILE && function.name == "dense_stream")
    else {
        violations.push(format!(
            "`{TABULATED_FILE}` has no bounded first-real-partial stream"
        ));
        return;
    };
    let dense_compact = dense.body.split_whitespace().collect::<String>();
    for witness in [
        "if!panel.is_empty(){returndense_stream_with_page(triple,epilogue,options,panel,ledger);}",
        "letmutpage=[Alphabet::<E,Bd>::ZERO;blocking::KC]",
        "dense_stream_with_page(triple,epilogue,options,&mutpage,ledger)",
    ] {
        if !dense_compact.contains(witness) {
            violations.push(format!(
                "{}:{}: the bounded dense stream lacks retained-partial witness `{witness}`",
                dense.rel, dense.line
            ));
        }
    }
    for forbidden in [
        "panel.copy_from_slice",
        "decode_row_into",
        "dense_stream_dot",
        "ledger.kernelled",
    ] {
        if dense_compact.contains(forbidden) {
            violations.push(format!(
                "{}:{}: the page chooser performs work instead of forwarding the caller page: \
                 `{forbidden}`",
                dense.rel, dense.line
            ));
        }
    }

    let Some(walk) = functions.iter().find(|function| {
        function.rel == TABULATED_FILE && function.name == "dense_stream_with_page"
    }) else {
        violations.push(format!(
            "`{TABULATED_FILE}` has no persistent caller-page dense stream"
        ));
        return;
    };
    let walk_compact = walk.body.split_whitespace().collect::<String>();
    for witness in [
        "debug_assert!(shape.k==0||!page.is_empty())",
        "letborrowed=page.len()>=shape.k",
        "letmutaccepted=false",
        "triple.w.decode_row_into(j,&mutpage[..shape.k])",
        "letright=MatView::row_major(&page[..shape.k],shape.k,1)",
        "letmutoutput=[O::ZERO;ROW_TILES[0]]",
        "letrows=(shape.m-row0).min(ROW_TILES[0])",
        "letsink=MatViewMut::row_major(&mutoutput[..rows],rows,1)",
        "ledger.kernelled();",
        "E::dense_gemm(left,right,sink,epilogue,options,&mut[])",
        "*triple.c.at_mut(row0+i,j)=cell",
        "letdepth=(shape.k-start).min(page.len())",
        "*cell=triple.w.at(j,start+p)",
        "dense_stream_dot::<E,Bd,O,Lg>(left,&page[..depth],options,ledger",
        "Some(partial)=>{accepted=true;partial}",
        "Noneifaccepted=>{panic!(",
        "None=>returnfalse",
        "iffirst{acc=partial;first=false;}else{ledger.added(1);acc=acc.combine(partial);}",
        "epilogue.finish(acc,prior,options.encode)",
    ] {
        if !walk_compact.contains(witness) {
            violations.push(format!(
                "{}:{}: the persistent dense stream lacks real-page witness `{witness}`",
                walk.rel, walk.line
            ));
        }
    }
    if walk_compact.contains("dense_stream_dot::<E,Bd,O,Lg>(left,&page[..shape.k]") {
        violations.push(format!(
            "{}:{}: a full decoded source row is still presented once per output cell instead \
             of once per bounded row tile",
            walk.rel, walk.line
        ));
    }
    let batched_ledger = walk_compact.find("ledger.kernelled();");
    let batched_dense = walk_compact.find("E::dense_gemm(left,right,sink,epilogue,options,&mut[])");
    if !matches!((batched_ledger, batched_dense), (Some(ledger), Some(dense)) if ledger < dense)
        || walk_compact.matches("ledger.kernelled();").count() != 1
    {
        violations.push(format!(
            "{}:{}: the full-row dense presentation is not counted exactly once per bounded \
             row tile immediately before the family call",
            walk.rel, walk.line
        ));
    }
    let first_partial = walk_compact.find("dense_stream_dot::<E,Bd,O,Lg>(");
    let prior = walk_compact.find("letprior=ifreads_c");
    let write = walk_compact.find("*triple.c.at_mut(i,j)=epilogue.finish(");
    if first_partial.is_none()
        || prior.is_none()
        || write.is_none()
        || first_partial >= prior
        || prior >= write
    {
        violations.push(format!(
            "{}:{}: the first real partial is not accepted before caller `C` is observed",
            walk.rel, walk.line
        ));
    }
    for forbidden in [
        "E::mac(",
        "dot_ref(",
        "dot_lane(",
        "ledger.multiplied(",
        "DenseCapabilityProbe",
    ] {
        if walk_compact.contains(forbidden) {
            violations.push(format!(
                "{}:{}: the persistent dense stream reaches scalar/synthetic operation \
                 `{forbidden}`",
                walk.rel, walk.line
            ));
        }
    }

    let Some(stream) = functions
        .iter()
        .find(|function| function.rel == TABULATED_FILE && function.name == "stream")
    else {
        violations.push(format!(
            "`{TABULATED_FILE}` has no `stream`; empty and short float offers are unaudited"
        ));
        return;
    };
    let compact = stream.body.split_whitespace().collect::<String>();
    for witness in [
        "<E::StreamLaneasLane<E>>::capacity(Bd::VALUE)",
        "dot_lane::<E,Bd,E::StreamLane>(",
        "dot_walk::<E,E::StreamLane,_>(",
        "epilogue.finish(acc,prior,options.encode)",
    ] {
        if !compact.contains(witness) {
            violations.push(format!(
                "{}:{}: the total stream lacks associated-lane witness `{witness}`",
                stream.rel, stream.line
            ));
        }
    }
    if !compact.contains("ledger.multiplied(count_factor(shape.k));letprior=ifreads_c") {
        violations.push(format!(
            "{}:{}: the stream lacks one unconditional per-output empty-dot ledger notification",
            stream.rel, stream.line
        ));
    }
    for forbidden in [
        "E::mac(",
        "dot_ref(",
        "dense_chunked_route(",
        "Capture::<",
        "ledger.kernelled(",
    ] {
        if compact.contains(forbidden) {
            violations.push(format!(
                "{}:{}: the float-capable stream reaches forbidden scalar/page operation \
                 `{forbidden}`",
                stream.rel, stream.line
            ));
        }
    }

    for name in ["dot_lane", "dot_walk"] {
        let Some(dot) = functions
            .iter()
            .find(|function| function.rel == TABULATED_FILE && function.name == name)
        else {
            violations.push(format!(
                "`{TABULATED_FILE}` has no `{name}` associated-lane dot walk"
            ));
            continue;
        };
        let compact = dot.body.split_whitespace().collect::<String>();
        for witness in ["letmutlane=L::ZERO", "lane=lane.mac(", "lane.place(acc)"] {
            if !compact.contains(witness) {
                violations.push(format!(
                    "{}:{}: `{name}` bypasses associated lane witness `{witness}`",
                    dot.rel, dot.line
                ));
            }
        }
        // `dot_lane` and `dot_walk` are the ordinary total stream for finite
        // integer/custom alphabets, including a truthful zero-capacity lane.
        // The preceding type-bound and nonzero-depth route proves public
        // floats cannot reach them with a product to issue; rejecting their
        // generic `E::mac` arm would silently outlaw the non-float API.
    }
}

/// Return one complete impl block, retaining its declaration and braces.
fn impl_block<'a>(source: &'a str, declaration: &str) -> Option<&'a str> {
    let start = source.find(declaration)?;
    let open = start + source[start..].find('{')?;
    let bytes = source.as_bytes();
    let mut depth = 1usize;
    let mut end = open + 1;
    while end < bytes.len() && depth != 0 {
        match bytes[end] {
            b'{' => depth += 1,
            b'}' => depth -= 1,
            _ => {}
        }
        end += 1;
    }
    (depth == 0).then_some(&source[start..end])
}

/// Inspect the representation itself, not merely its allocator behaviour.
/// This is deliberately exact about the two source spellings because changing
/// either one changes what CA-05 claims is borrowed.
fn audit_borrowed_carrier_declarations(reference: &str, violations: &mut Vec<String>) {
    let Some(source) = declaration_body(reference, "enum AtlasCarrierSource") else {
        violations.push("the Atlas carrier has no auditable borrowed source declaration".into());
        return;
    };
    if !source.contains("Lattice(&'a [")
        || !source.contains("Word(&'a [")
        || source.matches("&'a [").count() != 2
        || source.matches('(').count() != 2
    {
        violations.push(
            "`AtlasCarrierSource` is not exactly the two borrowed caller lattice and canonical word sources"
                .into(),
        );
    }

    let Some(carrier) = declaration_body(reference, "struct AtlasCarrier<'a>") else {
        violations.push("the Atlas carrier view declaration is missing its source lifetime".into());
        return;
    };
    if !carrier.contains("source: AtlasCarrierSource<'a>")
        || carrier.contains('[')
        || carrier.matches(':').count() != 1
    {
        violations.push(
            "`AtlasCarrier` is not solely a tagged borrowed-source view; an inline carrier may be owned"
                .into(),
        );
    }

    let Some(blocks) = declaration_body(reference, "struct AtlasBlocks<'a>") else {
        violations
            .push("the Atlas projector view declaration is missing its carrier lifetime".into());
        return;
    };
    if !blocks.contains("carrier: AtlasCarrier<'a>")
        || blocks.contains('[')
        || blocks.matches(':').count() != 1
    {
        violations.push(
            "`AtlasBlocks` is not solely a carrier view; projected coordinates may be materialized"
                .into(),
        );
    }
}

/// Prove every declaration that can win an Atlas selection terminates in
/// lookup/add arithmetic.
///
/// A group-one declaration need not itself be a possible winner: the ordinary
/// NEON `vmull` tile is earlier than a selector-equivalent lookup tile, and the
/// Atlas selector's later-on-`<=` rule makes the former unreachable. That
/// dominance is proved field by field below. Any other non-lookup group-one
/// declaration remains a possible winner and fails this audit.
fn audit_group_one_family(
    root: &Path,
    functions: &[Function],
    family_name: &str,
    portable_symbols: &[&str],
    violations: &mut Vec<String>,
) -> Result<(), Fail> {
    let spec_raw = std::fs::read_to_string(root.join(KERNEL_SPEC_FILE))?;
    let family = family_named(&spec_raw, family_name).ok_or_else(|| {
        format!(
            "CU-11 cannot find `{family_name}` in `{KERNEL_SPEC_FILE}`; the group-one \
             selector audit would be empty"
        )
    })?;
    let portable_raw =
        std::fs::read_to_string(root.join("crates/uor-matmul-kernels/src/isa/portable.rs"))?;
    let arm_raw = std::fs::read_to_string(root.join("crates/uor-matmul-kernels/src/isa/arm.rs"))?;
    let x86_raw = std::fs::read_to_string(root.join("crates/uor-matmul-kernels/src/isa/x86.rs"))?;
    if family_name == "available_i8" {
        audit_neon_vmull_shadow(family, &arm_raw, violations);
        audit_avx2_lookup_m1_order(family, &spec_raw, &x86_raw, violations);
    }
    let mut entries = 0usize;
    let mut group_one = 0usize;
    let mut traditional = 0usize;
    let mut shadowed = 0usize;
    for line in family.lines().filter(|line| line.contains("=>")) {
        entries += 1;
        let symbol = line
            .rsplit_once("::")
            .map(|(_, symbol)| symbol)
            .unwrap_or("")
            .trim()
            .trim_end_matches(',');
        if portable_symbols.contains(&symbol) {
            group_one += 1;
            let marker = format!("{symbol},");
            let Some(at) = portable_raw.find(&marker) else {
                violations.push(format!(
                    "portable group-one entry `{symbol}` has no source declaration"
                ));
                continue;
            };
            let tail = &portable_raw[at..portable_raw.len().min(at + 300)];
            if !tail.contains("crate::lookup::i8_product") {
                violations.push(format!(
                    "portable group-one entry `{symbol}` is not backed by `i8_product`"
                ));
            }
            continue;
        }

        let (rel, raw) = if symbol.starts_with("AVX") {
            let rel = "crates/uor-matmul-kernels/src/isa/x86.rs";
            (rel, x86_raw.clone())
        } else if symbol.starts_with("NEON") {
            let rel = "crates/uor-matmul-kernels/src/isa/arm.rs";
            (rel, arm_raw.clone())
        } else if symbol.starts_with("SIMD128") {
            let rel = "crates/uor-matmul-kernels/src/isa/wasm.rs";
            (rel, std::fs::read_to_string(root.join(rel))?)
        } else {
            violations.push(format!(
                "unclassified `{family_name}` member `{symbol}`; CU-11 refuses to \
                 assume whether the group-one filter admits it"
            ));
            continue;
        };
        let Some(initializer) = constant_initializer(&raw, symbol) else {
            violations.push(format!(
                "`{rel}` has no initializer for family member `{symbol}`"
            ));
            continue;
        };
        let (group, backing) = if let Some(group) = parse_k_group(initializer) {
            (Some(group), initializer.to_string())
        } else {
            let helper = initializer
                .split('=')
                .nth(1)
                .unwrap_or("")
                .trim()
                .chars()
                .take_while(|character| character.is_ascii_alphanumeric() || *character == '_')
                .collect::<String>();
            let Some(function) = functions
                .iter()
                .find(|function| function.rel == rel && function.name == helper)
            else {
                violations.push(format!(
                    "cannot resolve `{symbol}`'s spec constructor `{helper}` in `{rel}`"
                ));
                continue;
            };
            (parse_k_group(&function.body), function.body.clone())
        };
        let Some(group) = group else {
            violations.push(format!(
                "`{symbol}` declares no readable `k_group`; the structural filter cannot be \
                 audited"
            ));
            continue;
        };
        let names_lookup = symbol.contains("LOOKUP");
        if group == 1 {
            group_one += 1;
            if symbol == "NEON_I8_I32" && family_name == "available_i8" {
                shadowed += 1;
                continue;
            }
            if !names_lookup || !backing.to_ascii_lowercase().contains("lookup") {
                violations.push(format!(
                    "group-one family member `{symbol}` is a possible Atlas winner without a \
                     proved later lookup dominator"
                ));
            } else {
                let kernel = backing
                    .split("mac_tile:")
                    .nth(1)
                    .unwrap_or("")
                    .trim_start()
                    .chars()
                    .take_while(|character| character.is_ascii_alphanumeric() || *character == '_')
                    .collect::<String>();
                let runtime = functions
                    .iter()
                    .find(|function| function.rel == rel && function.name == kernel);
                let canonical = runtime.is_some_and(|function| {
                    audit_group_one_kernel_terminus(functions, function, violations)
                });
                if !canonical {
                    violations.push(format!(
                        "group-one member `{symbol}` does not terminate in the canonical i8 \
                         product table; resolved kernel was `{kernel}`"
                    ));
                }
            }
        } else {
            traditional += 1;
            if names_lookup {
                violations.push(format!(
                    "lookup family member `{symbol}` declares `k_group: {group}`, so the \
                     structural selector silently drops it"
                ));
            }
        }
    }
    if entries == 0 || group_one == 0 || traditional == 0 {
        violations.push(format!(
            "the `{family_name}` group-one audit is vacuous: {entries} entries, {group_one} \
             admitted ({shadowed} shadowed), {traditional} grouped controls"
        ));
    }
    Ok(())
}

/// Audit the exact helper graph of each native lookup entry. Only the grouped
/// AVX2 tile has the `avx2_lookup_i8_depth` boundary; reduce and AVX-512 entries
/// retain their direct table terminus and must not acquire a fictitious edge to
/// a helper they do not call.
fn audit_group_one_kernel_terminus(
    functions: &[Function],
    runtime: &Function,
    violations: &mut Vec<String>,
) -> bool {
    match (runtime.rel.as_str(), runtime.name.as_str()) {
        ("crates/uor-matmul-kernels/src/isa/x86.rs", "avx2_lookup_i8") => {
            audit_native_nibble_terminus(functions, runtime, "avx2_lookup_i8_depth", violations)
        }
        ("crates/uor-matmul-kernels/src/isa/arm.rs", _) => {
            audit_native_nibble_terminus(functions, runtime, "neon_nibble_products", violations)
        }
        ("crates/uor-matmul-kernels/src/isa/wasm.rs", _) => {
            audit_native_nibble_terminus(functions, runtime, "simd128_nibble_products", violations)
        }
        _ => {
            let direct_table = contains_call(&runtime.body, "i8_products")
                || contains_call(&runtime.body, "i8_products_native");
            let canonical_table = functions.iter().any(is_canonical_i8_product_accessor);
            (!direct_table || canonical_table)
                && (direct_table
                    || contains_call(&runtime.body, "i8_product")
                    || contains_call(&runtime.body, "i8_product_from")
                    || contains_call(&runtime.body, "i8_nibble_products"))
        }
    }
}

/// The historical NEON tile consumes one coordinate per step, but cannot win
/// Atlas selection because its later lookup twin has the same backend, shape,
/// factorization, density, and bound. Altering any one of those fields removes
/// the proof; name similarity is deliberately insufficient.
fn audit_neon_vmull_shadow(family: &str, arm: &str, violations: &mut Vec<String>) {
    const VMULL: &str = "crate::isa::arm::NEON_I8_I32";
    const LOOKUP: &str = "crate::isa::arm::NEON_LOOKUP_I8_I32";
    let vmull_at = family.find(VMULL);
    let lookup_at = family.find(LOOKUP);
    if vmull_at.is_none() || lookup_at.is_none() || vmull_at >= lookup_at {
        violations.push(
            "the NEON vmull group-one tile is not shadowed by its later lookup twin".to_string(),
        );
        return;
    }

    let Some(vmull) = constant_initializer(arm, "NEON_I8_I32") else {
        violations.push("the NEON vmull shadow source declaration is missing".to_string());
        return;
    };
    let vmull = vmull.split_whitespace().collect::<String>();
    for witness in [
        "backend:Backend::Neon",
        "factorization:Factorization::Exact",
        "mr:4",
        "nr:8",
        "k_group:1",
        "products_per_step:8",
        "max_bound:u128::MAX",
    ] {
        if !vmull.contains(witness) {
            violations.push(format!(
                "the NEON vmull shadow lacks selector field `{witness}`"
            ));
        }
    }

    let lookup = constant_initializer(arm, "NEON_LOOKUP_I8_I32")
        .map(|source| source.split_whitespace().collect::<String>());
    if lookup.as_deref()
        != Some(
            "pubconstNEON_LOOKUP_I8_I32:KernelSpec<i8,i32>=neon_lookup_spec::<4,8>(Backend::Neon);",
        )
    {
        violations.push(
            "the later NEON lookup is not selector-equivalent to the 4x8 vmull tile".to_string(),
        );
    }
    let helper = declaration_body(arm, "const fn neon_lookup_spec")
        .map(|source| source.split_whitespace().collect::<String>());
    for witness in [
        "backend,",
        "factorization:Factorization::Exact",
        "mr:MR",
        "nr:NR",
        "k_group:1",
        "products_per_step:NR",
        "max_bound:u128::MAX",
        "mac_tile:neon_lookup_i8::<MR,NR>",
    ] {
        if helper
            .as_deref()
            .is_none_or(|source| !source.contains(witness))
        {
            violations.push(format!(
                "the NEON lookup shadow lacks selector-equivalent field `{witness}`"
            ));
        }
    }
}

/// The added one-row AVX2 Atlas tile precedes the established grouped M1 tile.
/// At every GEMM row count (`rows > 0`) the public chooser's later-on-equal
/// rule therefore restores the historical native winner. At `rows == 0`, which
/// GEMM rejects before selection, neither height fills and the strict-smaller
/// rule leaves the lookup visible through the public chooser; both sides of
/// that boundary are intentional and audited.
fn audit_avx2_lookup_m1_order(family: &str, spec: &str, x86: &str, violations: &mut Vec<String>) {
    const LOOKUP: &str = "crate::isa::x86::AVX2_LOOKUP_I8_I32_M1";
    const NATIVE: &str = "crate::isa::x86::AVX2_I8_I32_M1";
    let lookup_at = family.find(LOOKUP);
    let native_at = family.find(NATIVE);
    if lookup_at.is_none() || native_at.is_none() || lookup_at >= native_at {
        violations.push(
            "the AVX2 one-row lookup does not precede the historical native M1 winner".to_string(),
        );
    }

    let lookup = constant_initializer(x86, "AVX2_LOOKUP_I8_I32_M1")
        .map(|source| source.split_whitespace().collect::<String>());
    if lookup.as_deref()
        != Some(
            "pubconstAVX2_LOOKUP_I8_I32_M1:KernelSpec<i8,i32>=avx2_lookup_spec::<1,16>(Backend::Avx2);",
        )
    {
        violations.push("the AVX2 one-row lookup is not the audited 1x16 tile".to_string());
    }
    let native = constant_initializer(x86, "AVX2_I8_I32_M1")
        .map(|source| source.split_whitespace().collect::<String>());
    for witness in [
        "backend:Backend::Avx2",
        "factorization:Factorization::Exact",
        "mr:1",
        "nr:A2_I8_NR",
        "products_per_step:16",
        "max_bound:32767",
    ] {
        if native
            .as_deref()
            .is_none_or(|source| !source.contains(witness))
        {
            violations.push(format!(
                "the historical AVX2 M1 winner lacks selector field `{witness}`"
            ));
        }
    }
    if !x86.contains("const A2_I8_NR: usize = 16;") {
        violations.push("the historical AVX2 M1 width is not 16".to_string());
    }
    let chooser = declaration_body(spec, "pub fn choose_for_rows")
        .map(|source| source.split_whitespace().collect::<String>());
    for witness in [
        "ifb.mr>rows{spec.mr<=rows||spec.mr<b.mr}",
        "spec.mr<=rows&&spec.mr>=b.mr",
    ] {
        if chooser
            .as_deref()
            .is_none_or(|source| !source.contains(witness))
        {
            violations.push(format!(
                "the row chooser lacks the audited positive/zero boundary `{witness}`"
            ));
        }
    }
}

/// Follow one native lookup body through its same-file helpers to the vector
/// projector and canonical nibble row. AVX2 groups a full coordinate vector
/// before presenting each live depth, ARM reaches the projector directly, and
/// wasm names its four row accumulators separately so the emitted tile stays
/// in registers. Auditing the transitive edge preserves those factorizations
/// without mistaking an inlining boundary for a different product operation.
fn audit_native_nibble_terminus(
    functions: &[Function],
    runtime: &Function,
    helper_name: &str,
    violations: &mut Vec<String>,
) -> bool {
    let Some(helper) = functions
        .iter()
        .find(|function| function.rel == runtime.rel && function.name == helper_name)
    else {
        violations.push(format!(
            "{}:{}: `{}` has no resolvable native projector helper `{helper_name}`",
            runtime.rel, runtime.line, runtime.name
        ));
        return false;
    };
    let Some(runtime_index) = functions
        .iter()
        .position(|function| core::ptr::eq(function, runtime))
    else {
        violations.push(format!(
            "{}:{}: `{}` is not a member of the audited function graph",
            runtime.rel, runtime.line, runtime.name
        ));
        return false;
    };
    let helper_index = functions
        .iter()
        .position(|function| core::ptr::eq(function, helper))
        .expect("the projector helper was borrowed from this function graph");

    let mut reached = BTreeSet::from([runtime_index]);
    let mut queue = VecDeque::from([runtime_index]);
    let mut scalar_terminus = None;
    while let Some(caller_index) = queue.pop_front() {
        let caller = &functions[caller_index];
        if contains_call(&caller.body, "i8_products") || contains_call(&caller.body, "i8_product") {
            scalar_terminus = Some(caller);
        }
        for (callee_index, callee) in functions.iter().enumerate() {
            if callee.rel == runtime.rel
                && contains_call(&caller.body, &callee.name)
                && reached.insert(callee_index)
            {
                queue.push_back(callee_index);
            }
        }
    }
    if let Some(scalar) = scalar_terminus {
        violations.push(format!(
            "{}:{}: `{}` reaches scalar i8 product table lookup in `{}` before the native nibble \
             projector terminus",
            runtime.rel, runtime.line, runtime.name, scalar.name
        ));
        return false;
    }
    if !reached.contains(&helper_index) {
        violations.push(format!(
            "{}:{}: `{}` does not reach native projector helper `{helper_name}`",
            runtime.rel, runtime.line, runtime.name
        ));
        return false;
    }
    contains_call(&helper.body, "i8_nibble_products")
}

/// Bind the model-owned capacity to the declarations it sizes.
///
/// The generated bytes establish single-source ownership. The family macro's
/// test-only declaration walk establishes the independent semantic twin: it
/// ignores host feature predicates, visits every entry from the one normative
/// family list, and requires the largest actual `mr * nr` to equal the model
/// value. Both an overflowing tile and unused arbitrary headroom therefore
/// fail.
fn audit_generated_kernel_capacity(
    root: &Path,
    generated_max: usize,
    generated_source_max: usize,
    violations: &mut Vec<String>,
) -> Result<(), Fail> {
    let model = Model::load(&root.join("model"))?;
    model.check()?;
    if model.constants.kernel_capacity.max_tile_lanes != generated_max {
        violations.push(format!(
            "model kernel capacity {} differs from generated `{KERNEL_CAPACITY_FILE}` value \
             {generated_max}",
            model.constants.kernel_capacity.max_tile_lanes
        ));
    }
    if model.constants.kernel_capacity.max_source_sites != generated_source_max {
        violations.push(format!(
            "model Atlas source capacity {} differs from generated `{KERNEL_CAPACITY_FILE}` value {generated_source_max}",
            model.constants.kernel_capacity.max_source_sites
        ));
    }

    let capacity = std::fs::read_to_string(root.join(KERNEL_CAPACITY_FILE))?;
    let dispatch = std::fs::read_to_string(root.join(ATLAS_DISPATCH_FILE))?;
    audit_generated_capacity_artifacts(&model, &capacity, &dispatch, violations);

    let spec = std::fs::read_to_string(root.join(KERNEL_SPEC_FILE))?;
    let kernels_lib = std::fs::read_to_string(root.join("crates/uor-matmul-kernels/src/lib.rs"))?;
    audit_kernel_capacity_sources(&spec, &kernels_lib, violations);
    let (declared_max, declared_source_max) = declared_source_capacity_maxima(root, violations)?;
    if declared_max != generated_max {
        violations.push(format!(
            "the independently read ISA tile maximum is {declared_max}, but the model-generated \
             capacity is {generated_max}; spare headroom and overflow are both forbidden"
        ));
    }
    if declared_source_max != generated_source_max {
        violations.push(format!(
            "the independently read ISA source-site maximum is {declared_source_max}, but the model-generated capacity is {generated_source_max}; spare headroom and overflow are both forbidden"
        ));
    }
    Ok(())
}

fn audit_generated_capacity_artifacts(
    model: &Model,
    capacity: &str,
    dispatch: &str,
    violations: &mut Vec<String>,
) {
    let expected_capacity = codegen::render_kernel_capacity(model);
    if capacity != expected_capacity {
        violations.push(format!(
            "`{KERNEL_CAPACITY_FILE}` is not the exact model-generated capacity artifact"
        ));
    }
    let expected_dispatch = codegen::render_atlas_dispatch(model);
    if dispatch != expected_dispatch {
        violations.push(format!(
            "`{ATLAS_DISPATCH_FILE}` is not the exact model-generated capacity interval"
        ));
    }
}

fn audit_kernel_capacity_sources(spec: &str, kernels_lib: &str, violations: &mut Vec<String>) {
    let compact_spec = mask_comments_strings_and_tests(spec)
        .split_whitespace()
        .collect::<String>();
    let compact_lib = mask_comments_strings_and_tests(kernels_lib)
        .split_whitespace()
        .collect::<String>();
    if !compact_spec
        .contains("pubconstMAX_TILE_LANES:usize=crate::generated_capacity::MAX_TILE_LANES;")
    {
        violations.push(
            "the stable `spec::MAX_TILE_LANES` is not a direct alias of the private generated value"
                .to_string(),
        );
    }
    if !compact_lib.contains("modgenerated_capacity;") {
        violations.push(
            "the kernel crate does not compile its private generated capacity artifact".to_string(),
        );
    }

    let Some(family_macro) = declaration_body(spec, "macro_rules! family") else {
        violations.push("the kernel registry has no auditable `family!` generator".to_string());
        return;
    };
    let compact_macro = family_macro.split_whitespace().collect::<String>();
    for witness in [
        "$name:ident,$all:ident,$cached:ident,$cache:ident,$E:ty,$L:ty;",
        "assert!($spec.mr*$spec.nr<=MAX_TILE_LANES);",
        "assert!($spec.mr+$spec.nr<=crate::generated_capacity::MAX_ATLAS_SOURCE_SITES);",
        "#[cfg(test)]fn$all()->implIterator<Item=KernelSpec<$E,$L>>",
        "core::iter::empty()$(.chain(core::iter::once($spec)))*",
    ] {
        if !compact_macro.contains(witness) {
            violations.push(format!(
                "the one-source family generator lacks declaration-walk witness `{witness}`"
            ));
        }
    }

    let all_names = declared_family_all_names(spec);
    if all_names.is_empty() {
        violations
            .push("the generated-capacity audit found no declared kernel families".to_string());
        return;
    }
    let unique: BTreeSet<_> = all_names.iter().collect();
    if unique.len() != all_names.len() {
        violations.push(
            "two kernel families share one declaration-walk name, so one can be omitted"
                .to_string(),
        );
    }

    let Some(test) = declaration_body(
        spec,
        "fn declared_host_kernel_families_fit_generated_capacity_cg_22",
    ) else {
        violations.push(
            "the kernel registry has no executable CG-22 exact-capacity differential".to_string(),
        );
        return;
    };
    let compact_test = test.split_whitespace().collect::<String>();
    for witness in [
        ".map(|spec|(spec.mr*spec.nr,spec.mr+spec.nr))",
        ".reduce(|left,right|(left.0.max(right.0),left.1.max(right.1)))",
        ".map(|(cells,_)|cells)",
        ".map(|(_,sources)|sources)",
        "assert!(declared_cells<=MAX_TILE_LANES)",
        "assert!(declared_sources<=crate::generated_capacity::MAX_ATLAS_SOURCE_SITES)",
        "assert_eq!(declared_cells,MAX_TILE_LANES,",
        "assert_eq!(declared_sources,crate::generated_capacity::MAX_ATLAS_SOURCE_SITES,",
    ] {
        if !compact_test.contains(witness) {
            violations.push(format!(
                "the executable family-maximum differential lacks `{witness}`"
            ));
        }
    }
    let witnessed = all_names
        .iter()
        .filter(|name| compact_test.matches(&format!("maxima!({name})")).count() == 1)
        .count();
    if witnessed != all_names.len() {
        violations.push(format!(
            "the executable family-maximum differential covers {witnessed} of {} generated \
             declaration walks exactly once",
            all_names.len()
        ));
    }
    if compact_test.matches("maxima!(").count() != all_names.len() {
        violations.push(
            "the executable family-maximum differential has an extra, duplicate, or omitted family"
                .to_string(),
        );
    }
}

/// Read the architecture declarations rather than the generated capacity.
///
/// Every concrete kernel shape is paired with `tile_fits!`, and the family
/// macro independently asserts the same relation for every registered entry.
/// The largest concrete source witness therefore proves the lower side of
/// exactness while the generated per-entry assertions prove the upper side.
fn declared_source_capacity_maxima(
    root: &Path,
    violations: &mut Vec<String>,
) -> Result<(usize, usize), Fail> {
    let mut tile_maximum = 0usize;
    let mut source_maximum = 0usize;
    for relative in [
        "crates/uor-matmul-kernels/src/isa/portable.rs",
        "crates/uor-matmul-kernels/src/isa/x86.rs",
        "crates/uor-matmul-kernels/src/isa/arm.rs",
        "crates/uor-matmul-kernels/src/isa/wasm.rs",
    ] {
        let source = std::fs::read_to_string(root.join(relative))?;
        let shapes = concrete_tile_fit_shapes(&source);
        if shapes.is_empty() {
            violations.push(format!(
                "`{relative}` has no independently evaluable `tile_fits!` declaration"
            ));
        }
        for (rows, cols) in shapes {
            tile_maximum = tile_maximum.max(rows * cols);
            source_maximum = source_maximum.max(rows + cols);
        }
    }
    Ok((tile_maximum, source_maximum))
}

#[cfg(test)]
fn concrete_tile_fit_products(source: &str) -> Vec<usize> {
    concrete_tile_fit_shapes(source)
        .into_iter()
        .filter_map(|(rows, cols)| rows.checked_mul(cols))
        .collect()
}

fn concrete_tile_fit_shapes(source: &str) -> Vec<(usize, usize)> {
    let source = mask_comments_strings_and_tests(source);
    let mut constants = BTreeMap::<String, usize>::new();
    let mut unresolved = Vec::new();
    for line in source.lines() {
        let line = line.trim();
        let Some(const_at) = line.find("const ") else {
            continue;
        };
        let tail = &line[const_at + "const ".len()..];
        let Some((name, value)) = tail.split_once(": usize =") else {
            continue;
        };
        unresolved.push((
            name.trim().to_string(),
            value.trim().trim_end_matches(';').to_string(),
        ));
    }
    for _ in 0..=unresolved.len() {
        let mut changed = false;
        unresolved.retain(|(name, expression)| {
            if let Some(value) = eval_usize_expression(expression, &constants) {
                constants.insert(name.clone(), value);
                changed = true;
                false
            } else {
                true
            }
        });
        if !changed {
            break;
        }
    }

    let mut shapes = Vec::new();
    let mut from = 0usize;
    while let Some(offset) = source[from..].find("tile_fits!(") {
        let start = from + offset + "tile_fits!(".len();
        let Some(end_offset) = source[start..].find(')') else {
            break;
        };
        let end = start + end_offset;
        let arguments = &source[start..end];
        if !arguments.contains("MAX_TILE_LANES") {
            if let Some((rows, cols)) = arguments.split_once(',') {
                if let (Some(rows), Some(cols)) = (
                    eval_usize_expression(rows, &constants),
                    eval_usize_expression(cols, &constants),
                ) {
                    shapes.push((rows, cols));
                }
            }
        }
        from = end + 1;
    }
    shapes
}

fn eval_usize_expression(expression: &str, constants: &BTreeMap<String, usize>) -> Option<usize> {
    let expression = expression
        .trim()
        .trim_matches(|character| matches!(character, '(' | ')'));
    if let Ok(value) = expression.replace('_', "").parse::<usize>() {
        return Some(value);
    }
    if expression
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
    {
        return constants.get(expression).copied();
    }
    for operator in ['+', '*', '/'] {
        if let Some((left, right)) = expression.rsplit_once(operator) {
            let left = eval_usize_expression(left, constants)?;
            let right = eval_usize_expression(right, constants)?;
            return match operator {
                '+' => left.checked_add(right),
                '*' => left.checked_mul(right),
                '/' => left.checked_div(right),
                _ => None,
            };
        }
    }
    None
}

fn declared_family_all_names(source: &str) -> Vec<String> {
    let source = mask_comments_strings_and_tests(source);
    let bytes = source.as_bytes();
    let mut names = Vec::new();
    let mut from = 0usize;
    while let Some(offset) = source[from..].find("family! {") {
        let start = from + offset;
        let open = start + "family! ".len();
        let mut depth = 1usize;
        let mut end = open + 1;
        while end < bytes.len() && depth != 0 {
            match bytes[end] {
                b'{' => depth += 1,
                b'}' => depth -= 1,
                _ => {}
            }
            end += 1;
        }
        if depth != 0 {
            break;
        }
        let block = &source[open + 1..end - 1];
        let Some(header) = block.split_once(';').map(|(header, _)| header) else {
            from = end;
            continue;
        };
        let fields: Vec<_> = header
            .split(',')
            .map(str::trim)
            .filter(|field| !field.is_empty())
            .collect();
        if fields.len() >= 2 {
            names.push(fields[1].to_string());
        }
        from = end;
    }
    names
}

/// The live carrier is one reduction position plus a representation-sized set
/// of exact output cells, never a depth panel.
///
/// The independently derived maximum `mr + nr` owns the sole source-state
/// store, with no spare headroom inherited from the larger output-cell bound.
/// Every live output of a physical edge tile occupies one exact const-generic
/// frame and the contraction is invoked once: there is no cache window that
/// re-projects the same source. Each source product likewise owns one fixed
/// `AtlasProduct`; all lookup diagonals enter it before its resolved magnitude
/// is fractured by the signed-place radix and placed at the two resulting
/// grades.
fn audit_panel_execution_storage(
    engine: &str,
    dispatch: &str,
    functions: &[Function],
    max_tile_lanes: usize,
    max_source_sites: usize,
    violations: &mut Vec<String>,
) {
    let compact_engine = engine.split_whitespace().collect::<String>();
    let compact_dispatch = dispatch.split_whitespace().collect::<String>();
    let source_bound = format!("constMAX_ATLAS_SOURCE_SITES:usize={max_source_sites};");
    if !compact_dispatch.contains(&source_bound) {
        violations.push(format!(
            "`{ATLAS_DISPATCH_FILE}` does not generate its exact {max_source_sites}-site Atlas source capacity"
        ));
    }
    if compact_engine.contains("constATLAS_SOURCE_SITES:") {
        violations.push(format!(
            "`{ATLAS_ENGINE_FILE}` redeclares the Atlas source capacity instead of consuming its generated model value"
        ));
    }
    for witness in [
        "constATLAS_PRODUCT_LIMBS:usize=(u128::BITS+i32::BITS).div_ceil(u64::BITS)asusize",
        "structAtlasProduct{limbs:[u64;ATLAS_PRODUCT_LIMBS]",
        "structAtlasProjectedCode{exponent:i32,coordinates:[i8;MAX_ATLAS_WORDS],valuation:u8,extent:u8,kind:u8,}",
        "enumAtlasProjectedKind{FiniteZero,FinitePositive,FiniteNegative,PositiveInfinity,NegativeInfinity,NotANumber,}",
        "core::mem::size_of::<AtlasProjectedCode>()==core::mem::size_of::<PackedCode>()",
        "structAtlasTileWorkspace{",
        "source_kinds:[u8;MAX_ATLAS_SOURCE_SITES]",
        "source_finite:[AtlasFiniteSite;MAX_ATLAS_SOURCE_SITES]",
        "source_words:[[i8;MAX_ATLAS_WORDS];MAX_ATLAS_SOURCE_SITES]",
        "source_extents:[u8;MAX_ATLAS_SOURCE_SITES]",
        "products:[AtlasProduct;MAX_TILE_LANES]",
        "source_kinds:[AtlasProjectedKind::FiniteZeroasu8;MAX_ATLAS_SOURCE_SITES]",
        "constATLAS_TILE_WORK_BYTES:usize=core::mem::size_of::<AtlasTileWorkspace>()",
    ] {
        if !compact_engine.contains(witness) {
            violations.push(format!(
                "`{ATLAS_ENGINE_FILE}` lacks derived bounded-frame/product witness `{witness}`"
            ));
        }
    }
    if !compact_engine.contains("include!()")
        || compact_engine.contains("macro_rules!dispatch_atlas_cell_capacity")
    {
        violations.push(format!(
            "`{ATLAS_ENGINE_FILE}` does not consume the model-generated Atlas dispatcher"
        ));
    }
    audit_exact_capacity_dispatch(dispatch, max_tile_lanes, violations);

    let Some(context) = declaration_body(engine, "struct AtlasOutputTileContext") else {
        violations.push(format!(
            "`{ATLAS_ENGINE_FILE}` has no typed output-tile context; exhaustive capacity arms may repeat argument setup"
        ));
        return;
    };
    let context = context.split_whitespace().collect::<String>();
    for witness in [
        "a:&'callMatView<'a,E>",
        "b:&'callMatView<'b,E>",
        "c:&'callmutMatViewMut<'c,O>",
        "epilogue:&'callEp",
        "options:GemmOptions",
        "pa:&'call[PackedCode]",
        "pb:&'call[PackedCode]",
        "shape:Shape",
        "spec:KernelSpec<i8,i32>",
        "i0:usize",
        "j0:usize",
        "block_start:usize",
        "rows:usize",
        "cols:usize",
        "cached_a_rows:usize",
        "b_offer_cols:usize",
        "place:P",
        "workspace:&'callmutAtlasTileWorkspace",
        "ledger:&'callmutLg",
    ] {
        if !context.contains(witness) {
            violations.push(format!(
                "the typed output-tile context lacks borrowed invariant witness `{witness}`"
            ));
        }
    }

    let Some(contract) = functions.iter().find(|function| {
        function.rel == ATLAS_ENGINE_FILE && function.name == "accumulate_direct_atlas_tile"
    }) else {
        violations.push(format!(
            "`{ATLAS_ENGINE_FILE}` has no `accumulate_direct_atlas_tile`; lazy source storage is unaudited"
        ));
        return;
    };
    let contract_compact = contract.body.split_whitespace().collect::<String>();
    let contract_code = contract.code.split_whitespace().collect::<String>();
    if !contract_code.contains("workspace:&mutAtlasTileWorkspace") {
        violations.push(format!(
            "{}:{}: the direct contraction does not borrow its one fixed workspace",
            contract.rel, contract.line
        ));
    }
    if contract_compact.matches("cells.for_each_live(").count() != 2
        || contract_compact.contains("ifi<rows&&j<cols")
        || contract_compact.contains("ifi>=rows||j>=cols")
    {
        violations.push(format!(
            "{}:{}: the Atlas cell view does not make its live-only traversal invariant explicit",
            contract.rel, contract.line
        ));
    }
    for required in [
        "workspace.source_kinds.split_at_mut(spec.mr)",
        "workspace.source_finite.split_at_mut(spec.mr)",
        "&mutremainder[..spec.nr]",
        "workspace.source_extents.split_at_mut(spec.mr)",
        "forpin0..depth",
        "source_a(p,i,ledger)",
        "source_b(p,j,ledger)",
        "atlas_kind_is_boundary(a_kinds[i])",
        "atlas_kind_is_boundary(b_kinds[j])",
        "atlas_boundary_code(a_kinds[i])",
        "atlas_boundary_code(b_kinds[j])",
        "foriin0..rows",
        "letfirst=i*spec.nr",
        "workspace.products[first..first+cols].fill(AtlasProduct::ZERO)",
        "ledger.product_initialized(rows*cols)",
        "workspace.products[physical_lane].add_diagonal(lane,diagonal)",
        "workspace.products[physical_lane].signed_magnitude()",
    ] {
        if !contract_compact.contains(required) {
            violations.push(format!(
                "{}:{}: the direct contraction lacks lazy-cell witness `{required}`",
                contract.rel, contract.line
            ));
        }
    }
    audit_product_resolution(
        contract,
        "workspace.products[physical_lane].add_diagonal(lane,diagonal)",
        "workspace.products[physical_lane].signed_magnitude()",
        "place(accumulator,",
        violations,
    );

    for (name, witnesses) in [
        (
            "atlas_source_state",
            &[
                "ifletSome((negative,magnitude,exponent))=finite_parts(code)",
                "ifmagnitude==0{return(AtlasProjectedKind::FiniteZeroasu8,AtlasFiniteSite::ZERO)",
                "let(unit,valuation)=atlas_odd_section(magnitude)",
                "ifnegative{AtlasProjectedKind::FiniteNegativeasu8}else{AtlasProjectedKind::FinitePositiveasu8}",
                "grade:i64::from(ifprescaled{0}else{exponent})+i64::from(valuation)",
                "elseifcode.is_nan(){(AtlasProjectedKind::NotANumberasu8,AtlasFiniteSite::ZERO)}",
                "elseifcode.mantissa<0{(AtlasProjectedKind::NegativeInfinityasu8,AtlasFiniteSite::ZERO,)}else{(AtlasProjectedKind::PositiveInfinityasu8,AtlasFiniteSite::ZERO,)}",
            ][..],
        ),
        (
            "atlas_kind_is_boundary",
            &[
                "kind==AtlasProjectedKind::PositiveInfinityasu8",
                "kind==AtlasProjectedKind::NegativeInfinityasu8",
                "kind==AtlasProjectedKind::NotANumberasu8",
            ][..],
        ),
        (
            "atlas_kind_is_productive",
            &[
                "kind==AtlasProjectedKind::FinitePositiveasu8",
                "kind==AtlasProjectedKind::FiniteNegativeasu8",
            ][..],
        ),
        (
            "atlas_boundary_code",
            &[
                "kind==AtlasProjectedKind::FiniteZeroasu8",
                "kind==AtlasProjectedKind::FinitePositiveasu8",
                "kind==AtlasProjectedKind::FiniteNegativeasu8",
                "kind==AtlasProjectedKind::PositiveInfinityasu8",
                "kind==AtlasProjectedKind::NegativeInfinityasu8",
                "kind==AtlasProjectedKind::NotANumberasu8",
            ][..],
        ),
    ] {
        let Some(function) = functions
            .iter()
            .find(|function| function.rel == ATLAS_ENGINE_FILE && function.name == name)
        else {
            violations.push(format!(
                "`{ATLAS_ENGINE_FILE}` has no exact six-state boundary quotient `{name}`"
            ));
            continue;
        };
        let compact = function.body.split_whitespace().collect::<String>();
        for witness in witnesses {
            if !compact.contains(witness) {
                violations.push(format!(
                    "{}:{}: `{name}` aliases or omits boundary state `{witness}`",
                    function.rel, function.line
                ));
            }
        }
    }

    let projector = functions.iter().find(|function| {
        function.rel == ATLAS_ENGINE_FILE
            && function.name == "project"
            && function.body.contains("atlas_word")
            && function.body.contains("finite_parts")
    });
    if let Some(projector) = projector {
        let compact = projector.body.split_whitespace().collect::<String>();
        for witness in [
            "letSome((negative,magnitude,exponent))=finite_parts(code)else",
            "let(unit,valuation)=atlas_odd_section(magnitude)",
            "letatom=AtlasAtom{unit,grade:i64::from(exponent)+i64::from(valuation),negative,}",
            "letextent=atlas_word(atom,&mutcoordinates)",
            "coordinates,valuation:valuationasu8,extent:extentasu8",
        ] {
            if !compact.contains(witness) {
                violations.push(format!(
                    "{}:{}: the in-place Atlas projector lacks `{witness}`",
                    projector.rel, projector.line
                ));
            }
        }
        if compact.matches("atlas_word(").count() != 1
            || compact.contains("from_bits")
            || compact.contains("to_bits")
            || compact.contains("trailing_zeros")
        {
            violations.push(format!(
                "{}:{}: one source projection must build one Atlas word directly, without an IEEE round trip",
                projector.rel, projector.line
            ));
        }
    } else {
        violations.push(format!(
            "`{ATLAS_ENGINE_FILE}` has no in-place `AtlasProjectedCode::project`"
        ));
    }

    let odd = functions
        .iter()
        .find(|function| function.rel == ATLAS_ENGINE_FILE && function.name == "atlas_odd_section");
    if odd.is_none_or(|function| {
        let compact = function.body.split_whitespace().collect::<String>();
        !compact.contains("letquotient=magnitude/2")
            || !compact.contains("quotient.wrapping_add(quotient)!=magnitude")
            || !compact.contains("magnitude=quotient")
            || compact.contains("trailing_zeros")
            || compact.contains(">>")
    }) {
        violations.push(
            "the live Atlas odd-section helper does not use quotient/add divisibility".to_string(),
        );
    }
    let signed_place_split = functions.iter().find(|function| {
        function.rel == ATLAS_ENGINE_FILE && function.name == "atlas_split_signed_place"
    });
    if compact_engine
        .matches("constATLAS_SIGNED_PLACE_RADIX:u128=i128::MIN.unsigned_abs()")
        .count()
        != 1
        || signed_place_split.is_none_or(|function| {
            let compact = function.body.split_whitespace().collect::<String>();
            compact
                .matches("magnitude%ATLAS_SIGNED_PLACE_RADIX")
                .count()
                != 1
                || compact
                    .matches("magnitude/ATLAS_SIGNED_PLACE_RADIX")
                    .count()
                    != 1
                || compact.contains("magnitude>>")
                || compact.contains("magnitude<<")
                || compact.contains("magnitude&")
        })
    {
        violations.push(
            "the live signed-place fracture does not use one Euclidean radix quotient and remainder"
                .to_string(),
        );
    }
    let word = functions
        .iter()
        .find(|function| function.rel == ATLAS_ENGINE_FILE && function.name == "atlas_word");
    if word.is_none_or(|function| {
        let compact = function.body.split_whitespace().collect::<String>();
        !compact.contains("letresidue=value.rem_euclid(radix)")
            || !compact.contains("value=(value-digit)/radix")
            || compact.contains(">>ATLAS_DIGIT_BITS")
            || compact.contains("valueasu8")
            || compact.contains("valueasi8")
    }) {
        violations.push(
            "the live Atlas word projector does not use centered Euclidean radix extraction"
                .to_string(),
        );
    }

    let cache = functions.iter().find(|function| {
        function.rel == ATLAS_ENGINE_FILE && function.name == "cache_atlas_source"
    });
    if let Some(cache) = cache {
        let compact = cache.body.split_whitespace().collect::<String>();
        for witness in [
            "let(projected,occupied)=AtlasProjectedCode::project(code)",
            "ifoccupied{ledger.projected();}",
            "projected.into_packed()",
        ] {
            if !compact.contains(witness) {
                violations.push(format!(
                    "{}:{}: the in-place source cache lacks `{witness}`",
                    cache.rel, cache.line
                ));
            }
        }
        if compact.matches("AtlasProjectedCode::project(").count() != 1
            || compact.matches("ledger.projected();").count() != 1
        {
            violations.push(format!(
                "{}:{}: the source cache does not record exactly its one real projection",
                cache.rel, cache.line
            ));
        }
    } else {
        violations.push(format!(
            "`{ATLAS_ENGINE_FILE}` has no in-place `cache_atlas_source` seam"
        ));
    }

    let projection_a = contract_compact.find("letextent=replace_atlas_word(atom,&muta_words[i])");
    let projection_b = contract_compact.find("letextent=replace_atlas_word(atom,&mutb_words[j])");
    let diagonals = contract_compact.find("fordiagonalin0..a_extent+b_extent-1");
    if contract_compact.matches("replace_atlas_word(").count() != 2
        || contract_compact.matches("ledger.projected();").count() != 2
        || !matches!((projection_a, projection_b, diagonals), (Some(a), Some(b), Some(diagonal)) if a < diagonal && b < diagonal)
    {
        violations.push(format!(
            "{}:{}: each live A/B source must be projected exactly once before all of its \
             coordinate diagonals",
            contract.rel, contract.line
        ));
    }
    for witness in [
        "AtlasSource::Raw(code)=>{(a_kinds[i],a_finite[i])=atlas_source_state(code,false);a_extents[i]=0;}",
        "AtlasSource::Projected(projected)=>{a_kinds[i]=projected.kind;ifatlas_kind_is_productive(projected.kind){a_finite[i]=projected.finite_site();}a_words[i]=projected.coordinates;a_extents[i]=projected.extent;}",
        "letextent=ifa_extents[i]!=0{usize::from(a_extents[i])}else{letextent=replace_atlas_word(atom,&muta_words[i]);ledger.projected();extent}",
        "letextent=ifb_extents[j]!=0{usize::from(b_extents[j])}else{letextent=replace_atlas_word(atom,&mutb_words[j]);ledger.projected();extent}",
    ] {
        if !contract_compact.contains(witness) {
            violations.push(format!(
                "{}:{}: cached/raw projection reuse lacks `{witness}`",
                contract.rel, contract.line
            ));
        }
    }

    let replacement = functions.iter().find(|function| {
        function.rel == ATLAS_ENGINE_FILE && function.name == "replace_atlas_word"
    });
    if let Some(replacement) = replacement {
        let compact = replacement.body.split_whitespace().collect::<String>();
        for witness in [
            "letextent=atlas_word(atom,coordinates)",
            "coordinates[extent..].fill(0)",
            "extent",
        ] {
            if !compact.contains(witness) {
                violations.push(format!(
                    "{}:{}: reused Atlas words do not clear exactly their retired suffix; missing `{witness}`",
                    replacement.rel, replacement.line
                ));
            }
        }
        if compact.contains("coordinates.fill(0)") {
            violations.push(format!(
                "{}:{}: reused Atlas words rewrite their live prefix instead of only the retired suffix",
                replacement.rel, replacement.line
            ));
        }
    } else {
        violations.push(format!(
            "`{ATLAS_ENGINE_FILE}` has no exact reused-word replacement seam"
        ));
    }

    let frame = functions.iter().find(|function| {
        function.rel == ATLAS_ENGINE_FILE && function.name == "execute_atlas_output_tile"
    });
    if let Some(frame) = frame {
        let code = frame.code.split_whitespace().collect::<String>();
        let body = frame.body.split_whitespace().collect::<String>();
        for witness in [
            "fnexecute_atlas_output_tile<constCELL_CAP:usize",
            "context:&mutAtlasOutputTileContext",
            "letmutaccumulators=[<AccOf<E>asAccumulator>::ZERO;CELL_CAP]",
            "execute_atlas_output_tile_body(&mutaccumulators,context)",
        ] {
            if !code.contains(witness) && !body.contains(witness) {
                violations.push(format!(
                    "{}:{}: the const-generic output frame lacks bounded witness `{witness}`",
                    frame.rel, frame.line
                ));
            }
        }
        let frame_marker = "#[inline(never)]fnexecute_atlas_output_tile";
        if !compact_engine.contains(frame_marker) {
            violations.push(format!(
                "{}:{}: `execute_atlas_output_tile` is not a non-inlined frame, so unchosen \
                 const extents may inflate its generic caller's stack",
                frame.rel, frame.line
            ));
        }
    } else {
        violations.push(format!(
            "`{ATLAS_ENGINE_FILE}` has no const-generic `execute_atlas_output_tile` frame"
        ));
    }

    let body = functions.iter().find(|function| {
        function.rel == ATLAS_ENGINE_FILE && function.name == "execute_atlas_output_tile_body"
    });
    if let Some(body) = body {
        let compact = body.body.split_whitespace().collect::<String>();
        for witness in [
            "accumulators:&mut[AccOf<E>]",
            "context:&mutAtlasOutputTileContext",
            "debug_assert_eq!(accumulators.len(),tile_outputs,",
            "letworkspace=&mut*context.workspace",
            "letmutcells=WindowAtlasCells{first_logical:0,live_cols:cols,physical_cols:spec.nr,accumulators,}",
            "AtlasSource::Projected(AtlasProjectedCode::from_packed(pa[ii*shape.k+p]))",
            "AtlasSource::Projected(AtlasProjectedCode::from_packed(pb[",
            "(j0+jj-block_start)*shape.k+p",
            "ledger.decoded_a()",
            "ledger.decoded_b()",
            "accumulate_direct_atlas_tile(",
        ] {
            if !body
                .code
                .split_whitespace()
                .collect::<String>()
                .contains(witness)
                && !compact.contains(witness)
            {
                violations.push(format!(
                    "{}:{}: the shared slice execution body lacks exact-frame witness \
                     `{witness}`",
                    body.rel, body.line
                ));
            }
        }
        if compact.matches("accumulate_direct_atlas_tile(").count() != 1
            || compact.contains("cell_start")
            || compact.contains("accumulators[..")
            || compact.contains("ifshape.k!=0")
        {
            violations.push(format!(
                "{}:{}: one post-bypass physical tile is replayed, conditionally skipped, or sliced instead of one live-cell contraction",
                body.rel, body.line
            ));
        }
    } else {
        violations.push(format!(
            "`{ATLAS_ENGINE_FILE}` has no shared `execute_atlas_output_tile_body`"
        ));
    }

    let tiles = functions.iter().find(|function| {
        function.rel == ATLAS_ENGINE_FILE && function.name == "gemm_float_tiles_with_selector"
    });
    if let Some(tiles) = tiles {
        let compact = tiles.body.split_whitespace().collect::<String>();
        for witness in [
            "ifshape.k==0",
            "letreads_c=epilogue.reads_c()",
            "epilogue.finish(<AccOf<E>asAccumulator>::ZERO,prior,options.encode)",
            "ledger.encoded()",
            "letspec=select(options.backend,shape,pa.len(),pb.len())",
            "letmutworkspace=AtlasTileWorkspace::ZERO",
            "letcell_capacity=rows*cols",
            "letmutcontext=AtlasOutputTileContext{",
            "workspace:&mutworkspace",
            "*slot=cache_atlas_source(value.pack(),ledger)",
            "execute_atlas_output_tile::<$cell_cap,E,O,Ep,P,Lg>(&mutcontext",
            "dispatch_atlas_cell_capacity!(cell_capacity,execute)",
        ] {
            if !compact.contains(witness) {
                violations.push(format!(
                    "{}:{}: `gemm_float_tiles` lacks bounded-frame dispatch witness `{witness}`",
                    tiles.rel, tiles.line
                ));
            }
        }
        let empty = compact.find("ifshape.k==0");
        let selected = compact.find("letspec=select(");
        if !matches!((empty, selected), (Some(empty), Some(selected)) if empty < selected)
            || compact[..selected.unwrap_or(0)].contains("ledger.selected(")
            || compact[..selected.unwrap_or(0)].contains("dispatch_atlas_cell_capacity!(")
        {
            violations.push(format!(
                "{}:{}: the empty reduction does not terminate before selector, route ledger, and Atlas frame dispatch",
                tiles.rel, tiles.line
            ));
        }
        let workspace = compact.find("letmutworkspace=AtlasTileWorkspace::ZERO");
        let block_walk = compact.find("whileblock_start<shape.n");
        if compact.matches("AtlasTileWorkspace::ZERO").count() != 1
            || !matches!((workspace, block_walk), (Some(workspace), Some(block)) if workspace < block)
        {
            violations.push(format!(
                "{}:{}: the fixed Atlas workspace is not initialized exactly once before the output-tile walk",
                tiles.rel, tiles.line
            ));
        }
    } else {
        violations.push(format!(
            "`{ATLAS_ENGINE_FILE}` has no `gemm_float_tiles_with_selector`; zero-depth and const-generic frame dispatch are unaudited"
        ));
    }

    let production_tiles = functions
        .iter()
        .find(|function| function.rel == ATLAS_ENGINE_FILE && function.name == "gemm_float_tiles");
    if production_tiles.is_none_or(|function| {
        let compact = function.body.split_whitespace().collect::<String>();
        !compact.contains("gemm_float_tiles_with_selector(")
            || !compact.contains("atlas_tile_spec::<AccOf<E>>(backend,shape,pa_codes,pb_codes)")
    }) {
        violations.push(format!(
            "`{ATLAS_ENGINE_FILE}` production tiled wrapper does not inject exactly the global Atlas selector"
        ));
    }

    let scalar_contract = functions
        .iter()
        .find(|function| function.rel == ATLAS_ENGINE_FILE && function.name == "accumulate_atlas");
    if let Some(scalar_contract) = scalar_contract {
        audit_product_resolution(
            scalar_contract,
            "product.add_diagonal(lane[0],diagonal)",
            "product.signed_magnitude()",
            "acc.place_at_wide(",
            violations,
        );
    }
    audit_total_panel_semantics(functions, violations);

    for forbidden in [
        "ATLAS_PANEL_SITES",
        "atlas_panel_depth",
        "code_storage",
        "blocking::KC",
        "with_atlas_cells",
        "atlas_cell_lanes",
        "atlas_retained_cells",
        "source_projected",
        "source_atoms:",
        "blocking::L1_BYTES",
        "whilecell_start",
        "ATLAS_L1_CELL_CAP",
        "full_tile_fits",
        "l1_cell_array_fits",
        "[AccOf<E>::ZERO;MAX_TILE_LANES]",
        "[A::ZERO;MAX_TILE_LANES]",
    ] {
        if compact_engine.contains(forbidden) {
            violations.push(format!(
                "`{ATLAS_ENGINE_FILE}` retains depth-panel token `{forbidden}`; the source carrier must be one k-independent cell"
            ));
        }
    }

    // A repeated PackedCode initializer is a dormant worst-case code cache.
    // The live source cell retains only the six-state boundary quotient, so no
    // full code array is needed after decode/projection.
    let repeated_code_stores = [
        "[ZERO_CODE;",
        "[UNIT_CODE;",
        "[PackedCode::default();",
        "[PackedCode{",
    ]
    .iter()
    .map(|token| compact_engine.matches(token).count())
    .sum::<usize>();
    if repeated_code_stores != 0 {
        violations.push(format!(
            "`{ATLAS_ENGINE_FILE}` has {repeated_code_stores} repeated PackedCode stores; \
             CA-05 requires one derived boundary-state cell, never a depth carrier or separate \
             A/B full-code arrays"
        ));
    }
}

fn audit_exact_capacity_dispatch(
    dispatch: &str,
    max_tile_lanes: usize,
    violations: &mut Vec<String>,
) {
    if max_tile_lanes == 0 {
        return;
    }
    let Some(dispatch) = declaration_body(dispatch, "macro_rules! dispatch_atlas_cell_capacity")
    else {
        violations.push(format!(
            "`{ATLAS_ENGINE_FILE}` has no exact-capacity frame dispatcher"
        ));
        return;
    };

    let mut arms = Vec::new();
    let mut has_default = false;
    for line in dispatch.lines() {
        let Some((pattern, body)) = line.split_once("=>") else {
            continue;
        };
        let pattern = pattern.trim().trim_end_matches(',');
        let body = body.split_whitespace().collect::<String>();
        if pattern == "_" {
            has_default = body.starts_with("unreachable!(");
        } else if pattern == "MAX_TILE_LANES" || pattern.parse::<usize>().is_ok() {
            arms.push((pattern.to_string(), body));
        }
    }

    if arms.len() != max_tile_lanes {
        violations.push(format!(
            "the Atlas capacity dispatcher has {} admissible arms; `MAX_TILE_LANES` requires \
             exactly {max_tile_lanes}",
            arms.len()
        ));
    }
    for expected in 1..max_tile_lanes {
        let index = expected - 1;
        let expected_pattern = expected.to_string();
        let expected_body = format!("$execute!({expected}),");
        if arms.get(index).map(|arm| (&arm.0, &arm.1)) != Some((&expected_pattern, &expected_body))
        {
            violations.push(format!(
                "the Atlas capacity dispatcher has a missing, duplicate, reordered, or \
                 mismatched arm at capacity {expected}"
            ));
        }
    }
    let terminal = arms.last();
    if terminal.map(|arm| arm.0.as_str()) != Some("MAX_TILE_LANES")
        || terminal.map(|arm| arm.1.as_str()) != Some("$execute!(MAX_TILE_LANES),")
    {
        violations.push(format!(
            "the Atlas capacity dispatch terminal is not the symbolic `MAX_TILE_LANES => \
             $execute!(MAX_TILE_LANES)` after capacities 1..{}",
            max_tile_lanes - 1
        ));
    }
    if !has_default {
        violations.push(
            "the Atlas capacity dispatcher has no unreachable out-of-family control arm"
                .to_string(),
        );
    }
}

/// One product may need two signed-`i128` placement calls because its magnitude
/// has two digits in the `i128::MAX + 1` signed-place radix. The invariant is
/// one carrier resolution after every lookup diagonal, followed by exactly one
/// radix fracture and the two resulting grades.
fn audit_product_resolution(
    function: &Function,
    diagonal: &str,
    resolution: &str,
    placement: &str,
    violations: &mut Vec<String>,
) {
    let compact = function.body.split_whitespace().collect::<String>();
    let diagonal_at = compact.find(diagonal);
    let resolution_at = compact.find(resolution);
    if diagonal_at.is_none() || resolution_at.is_none() || diagonal_at >= resolution_at {
        violations.push(format!(
            "{}:{}: `{}` does not consolidate lookup diagonals into one `AtlasProduct` before \
             resolving the mathematical source product",
            function.rel, function.line, function.name
        ));
        return;
    }
    if compact.matches(resolution).count() != 1 {
        violations.push(format!(
            "{}:{}: `{}` resolves `AtlasProduct` {} times in its source body; exactly one \
             resolution must follow the diagonal walk",
            function.rel,
            function.line,
            function.name,
            compact.matches(resolution).count()
        ));
    }
    let resolved = resolution_at.expect("the guarded resolution exists");
    if compact[..resolved].contains(placement) {
        violations.push(format!(
            "{}:{}: `{}` places a diagonal before the complete product carrier is resolved",
            function.rel, function.line, function.name
        ));
    }
    let tail = &compact[resolved..];
    let fracture_at = tail.find("atlas_split_signed_place(magnitude)");
    let low_at = tail.find("iflow!=0");
    let high_at = tail.find("ifhigh!=0");
    let source_ordered = matches!(
        (fracture_at, low_at, high_at),
        (Some(fracture), Some(low), Some(high)) if fracture < low && low < high
    );
    let high_grade = high_at.is_some_and(|high| {
        tail[high..].contains("i64::from(i128::BITS-1)")
            && tail[high..].contains("ifnegative{-1}else{1}")
    });
    if tail.matches(placement).count() != 2
        || tail.matches("atlas_split_signed_place(magnitude)").count() != 1
        || !source_ordered
        || !high_grade
        || tail.contains("magnitude>>")
        || tail.contains("magnitude<<")
        || tail.contains("magnitude&")
    {
        violations.push(format!(
            "{}:{}: `{}` does not terminate its one resolved product in exactly one \
             signed-place radix fracture and its two source-ordered grades",
            function.rel, function.line, function.name
        ));
    }
}

/// Panels are total objects: a missing coordinate is the float zero code, not
/// a reason to truncate the dot. The non-finite branch must see that synthetic
/// zero before finite atom projection so `infinity * implicit-zero` becomes
/// NaN under the same `accumulate_one` boundary join as an explicit zero.
fn audit_total_panel_semantics(functions: &[Function], violations: &mut Vec<String>) {
    let panels = functions.iter().find(|function| {
        function.rel == ATLAS_ENGINE_FILE && function.name == "accumulate_atlas_panels"
    });
    if let Some(panels) = panels {
        let compact = panels.body.split_whitespace().collect::<String>();
        for witness in [
            "accumulate_atlas_dot(",
            "pa.len().max(pb.len())",
            "pa.get(p).copied().unwrap_or(ZERO_CODE)",
            "pb.get(p).copied().unwrap_or(ZERO_CODE)",
        ] {
            if !compact.contains(witness) {
                violations.push(format!(
                    "{}:{}: unequal Atlas panels lack total zero-extension witness `{witness}`",
                    panels.rel, panels.line
                ));
            }
        }
        if compact.contains("pa.len().min(pb.len())") {
            violations.push(format!(
                "{}:{}: unequal Atlas panels truncate to their common prefix",
                panels.rel, panels.line
            ));
        }
    } else {
        violations.push(format!(
            "`{ATLAS_ENGINE_FILE}` has no `accumulate_atlas_panels`; unequal-panel totality is \
             unaudited"
        ));
    }

    let contract = functions
        .iter()
        .find(|function| function.rel == ATLAS_ENGINE_FILE && function.name == "accumulate_atlas");
    if let Some(contract) = contract {
        let compact = contract.body.split_whitespace().collect::<String>();
        let boundary = compact
            .find("if!a_code.is_finite()||!b_code.is_finite()")
            .zip(compact.find("acc.accumulate_one(a_code,b_code)"));
        let projection = compact.find("atlas_atom(a_code,prescaled)");
        if !matches!((boundary, projection), (Some((branch, join)), Some(project)) if branch < join && join < project)
        {
            violations.push(format!(
                "{}:{}: `accumulate_atlas` does not join non-finite products before finite \
                 projection; implicit-zero IEEE boundary products are not total",
                contract.rel, contract.line
            ));
        }
    }
}

fn constant_initializer<'a>(source: &'a str, symbol: &str) -> Option<&'a str> {
    let public = format!("pub const {symbol}");
    let crate_visible = format!("pub(crate) const {symbol}");
    let start = source
        .find(&public)
        .or_else(|| source.find(&crate_visible))?;
    let tail = &source[start..];
    // A rationale comment inside a declaration may contain a prose semicolon.
    // Find the terminator in the offset-preserving masked source so evidence
    // cannot be truncated merely by explaining why a field has its value.
    let end = mask_comments_strings_and_tests(tail).find(';')? + 1;
    Some(&tail[..end])
}

fn parse_usize_product(initializer: &str) -> Option<usize> {
    let expression = initializer.split_once('=')?.1.trim().trim_end_matches(';');
    expression.split('*').try_fold(1usize, |product, factor| {
        let factor = factor
            .trim()
            .trim_matches(|character| matches!(character, '(' | ')'))
            .replace('_', "")
            .parse::<usize>()
            .ok()?;
        product.checked_mul(factor)
    })
}

fn parse_k_group(source: &str) -> Option<usize> {
    let after = source.split("k_group:").nth(1)?.trim_start();
    let digits: String = after
        .chars()
        .take_while(|character| character.is_ascii_digit())
        .collect();
    (!digits.is_empty()).then(|| digits.parse().ok()).flatten()
}

fn family_named<'a>(source: &'a str, name: &str) -> Option<&'a str> {
    let declaration = format!("\n    {name},");
    let name_at = source.find(&declaration)?;
    let family_at = source[..name_at].rfind("family! {")?;
    let tail = &source[family_at..];
    let end = tail.find("\n}\n")? + 3;
    Some(&tail[..end])
}

fn inspect(functions: &[Function]) -> Result<Census, Fail> {
    if functions.is_empty() {
        return Err("CU-11 enumerated no shipped functions; the gate would pass vacuously".into());
    }

    let mut by_name: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (index, function) in functions.iter().enumerate() {
        by_name.entry(&function.name).or_default().push(index);
    }

    let mut reached = BTreeSet::new();
    let mut queue = VecDeque::new();
    let mut missing_roots = Vec::new();
    let mut census = Census::default();
    for root in ROOTS {
        let Some(indices) = by_name.get(root) else {
            missing_roots.push(*root);
            continue;
        };
        census.roots += indices.len();
        for &index in indices {
            if reached.insert(index) {
                queue.push_back(index);
            }
        }
    }
    if !missing_roots.is_empty() {
        return Err(format!(
            "CU-11 could not find live float root(s): {}. A call-graph gate with a missing \
             root does not govern the public operation.",
            missing_roots.join(", ")
        )
        .into());
    }

    let names: Vec<&str> = by_name.keys().copied().collect();
    let mut edges = BTreeSet::new();
    while let Some(caller) = queue.pop_front() {
        let code = &functions[caller].body;
        for name in &names {
            if !contains_call(code, name) {
                continue;
            }
            let candidates = &by_name[*name];
            // A bare method call does not identify which of several impls was
            // selected.  Following all same-named methods invents edges (and
            // once made the core reference appear live even though its type
            // was unnameable outside the crate).  Unique callees are exact;
            // public roots are seeded independently above; forbidden calls are
            // checked textually below even if their name is overloaded.
            if candidates.len() != 1 {
                continue;
            }
            for &callee in candidates {
                // A recursive edge is still an operation in the census.
                if edges.insert((caller, callee)) && reached.insert(callee) {
                    queue.push_back(callee);
                }
            }
        }
    }
    census.reachable = reached.len();
    census.edges = edges.len();
    if census.edges == 0 {
        return Err("CU-11 found roots but no call edge; the operation census is vacuous".into());
    }

    let mut violations = Vec::new();
    for &index in &reached {
        let function = &functions[index];
        if let Some((_, reason)) = FORBIDDEN_CALLS
            .iter()
            .find(|(name, _)| function.name == *name)
        {
            violations.push(format!(
                "{}:{}: `{}` is reachable: {reason}",
                function.rel, function.line, function.name
            ));
        }
        for (name, reason) in FORBIDDEN_CALLS {
            if contains_call(&function.body, name) {
                violations.push(format!(
                    "{}:{}: reachable `{}` calls `{name}`: {reason}",
                    function.rel, function.line, function.name
                ));
            }
        }
        for (token, reason) in FORBIDDEN_REACHABLE_TOKENS {
            if code_contains_wordish(&function.code, token) {
                violations.push(format!(
                    "{}:{}: `{token}` occurs in reachable `{}`: {reason}",
                    function.rel, function.line, function.name
                ));
            }
        }
    }

    let atlas: Vec<&Function> = functions
        .iter()
        .enumerate()
        .filter(|(index, function)| {
            reached.contains(index)
                && function.rel == ATLAS_ENGINE_FILE
                && (function.name.contains("atlas")
                    || matches!(function.name.as_str(), "gauge_interval" | "belongs"))
        })
        .map(|(_, function)| function)
        .collect();
    census.atlas_functions = atlas.len();
    census.atlas_edges = edges
        .iter()
        .filter(|(caller, callee)| {
            (functions[*caller].rel == ATLAS_ENGINE_FILE
                && functions[*caller].name.contains("atlas"))
                || (functions[*callee].rel == ATLAS_ENGINE_FILE
                    && functions[*callee].name.contains("atlas"))
        })
        .count();

    for required in REQUIRED_ENGINE {
        let live = functions.iter().enumerate().any(|(index, function)| {
            reached.contains(&index)
                && function.rel == ATLAS_ENGINE_FILE
                && function.name == *required
        });
        if !live {
            violations.push(format!(
                "none of the {} live float roots reaches `{required}` in \
                 `{ATLAS_ENGINE_FILE}`",
                census.roots
            ));
        }
    }

    // Read every production function in both the finite reference algebra and
    // the live Atlas engine. A dormant owned alternative is a second method
    // waiting to become reachable, which R13 forbids too.
    let atlas_all: Vec<&Function> = functions
        .iter()
        .filter(|function| {
            function.rel == ATLAS_REFERENCE_FILE
                || (function.rel == ATLAS_ENGINE_FILE && function.name.contains("atlas"))
        })
        .collect();
    if atlas_all.is_empty() {
        violations.push(
            "the Atlas reference and engine expose no production function; CU-11 would be empty"
                .to_string(),
        );
    }
    for function in &atlas_all {
        for token in OWNED_CARRIER_TOKENS {
            if function.code.contains(token) {
                violations.push(format!(
                    "{}:{}: `{}` contains `{token}`; CA-05 requires a borrowed/lazy carrier",
                    function.rel, function.line, function.name
                ));
            }
        }
        if function.rel == ATLAS_REFERENCE_FILE {
            for token in MATERIALIZED_CARRIER_TOKENS {
                if function.body.contains(token) {
                    violations.push(format!(
                        "{}:{}: `{}` contains `{token}`; CA-05 forbids stack-owned carrier or projector materialization too",
                        function.rel, function.line, function.name
                    ));
                }
            }
        }
        for (token, reason) in FORBIDDEN_REACHABLE_TOKENS {
            if code_contains_wordish(&function.code, token) {
                violations.push(format!(
                    "{}:{}: `{}` contains `{token}`: {reason}",
                    function.rel, function.line, function.name
                ));
            }
        }
    }

    // This is an operation census, not merely a reachability assertion.  Each
    // permitted semantic family must occur in reachable Atlas code.  The names
    // are intentionally broad enough for generic implementations while still
    // being plantable one family at a time.
    for function in &atlas {
        let lower = function.code.to_ascii_lowercase();
        census.lookups += count_any(&lower, &["lookup", "table", "address"])
            + usize::from(contains_call(&function.body, "mac_tile"));
        census.complete_adds += count_any(
            &lower,
            &["complete", "accumulate", "add_assign", "wrapping_add", "+="],
        ) + usize::from(contains_call(&function.body, "mac_tile"))
            + usize::from(contains_call(&function.body, "place_at"));
        census.dyadic_placements +=
            count_any(&lower, &["dyadic", "place", "shift", "grade", "laurent"])
                + usize::from(contains_call(&function.body, "place_at"));
    }
    for (count, family) in [
        (census.lookups, "Atlas lookup/address"),
        (census.complete_adds, "complete addition"),
        (census.dyadic_placements, "group-level dyadic placement"),
    ] {
        if count == 0 {
            violations.push(format!(
                "the reachable Atlas call graph contains no {family} operation; CU-11's \
                 operation census would pass without observing the claimed engine"
            ));
        }
    }

    if !violations.is_empty() {
        return Err(format!(
            "CA-05, CU-11: after IEEE decode and before the one IEEE encode, every live float \
             root must terminate in the borrowed Atlas lookup/complete-add engine. \
             Group-level complete dyadic placement is permitted; scalar significand \
             multiplication, per-product windows, reified operands, and traditional integer \
             multiply routes are not. The live call graph violates that boundary:\n\n{}",
            violations.join("\n")
        )
        .into());
    }

    Ok(census)
}

fn count_any(haystack: &str, needles: &[&str]) -> usize {
    needles
        .iter()
        .map(|needle| haystack.match_indices(needle).count())
        .sum()
}

fn code_contains_wordish(code: &str, token: &str) -> bool {
    let mut from = 0;
    while let Some(offset) = code[from..].find(token) {
        let start = from + offset;
        let end = start + token.len();
        let ident = |byte: u8| byte.is_ascii_alphanumeric() || byte == b'_';
        let before = start == 0 || !ident(code.as_bytes()[start - 1]);
        let after = end == code.len() || !ident(code.as_bytes()[end]);
        if before && after {
            return true;
        }
        from = start + 1;
    }
    false
}

fn declaration_body<'a>(source: &'a str, declaration: &str) -> Option<&'a str> {
    let start = source.find(declaration)?;
    let open = start + source[start..].find('{')?;
    let bytes = source.as_bytes();
    let mut depth = 1usize;
    let mut end = open + 1;
    while end < bytes.len() && depth != 0 {
        match bytes[end] {
            b'{' => depth += 1,
            b'}' => depth -= 1,
            _ => {}
        }
        end += 1;
    }
    (depth == 0).then(|| &source[open + 1..end - 1])
}

fn contains_call(code: &str, name: &str) -> bool {
    let bytes = code.as_bytes();
    let mut from = 0;
    while let Some(offset) = code[from..].find(name) {
        let start = from + offset;
        let mut at = start + name.len();
        let ident = |byte: u8| byte.is_ascii_alphanumeric() || byte == b'_';
        if (start > 0 && ident(bytes[start - 1])) || (at < bytes.len() && ident(bytes[at])) {
            from = start + 1;
            continue;
        }
        while at < bytes.len() && bytes[at].is_ascii_whitespace() {
            at += 1;
        }
        if bytes.get(at..at + 3) == Some(b"::<") {
            at += 3;
            let mut depth = 1usize;
            while at < bytes.len() && depth != 0 {
                match bytes[at] {
                    b'<' => depth += 1,
                    b'>' => depth -= 1,
                    _ => {}
                }
                at += 1;
            }
            while at < bytes.len() && bytes[at].is_ascii_whitespace() {
                at += 1;
            }
        }
        if bytes.get(at) == Some(&b'(') {
            return true;
        }
        from = start + 1;
    }
    false
}

fn shipped_functions(root: &Path) -> Result<Vec<Function>, Fail> {
    let mut files = Vec::new();
    for krate in [
        "uor-matmul",
        "uor-matmul-core",
        "uor-matmul-codec",
        "uor-matmul-kernels",
        "uor-matmul-gemm",
    ] {
        let dir = root.join("crates").join(krate).join("src");
        collect_rs(&dir, &mut files)?;
    }
    let mut functions = Vec::new();
    for path in files {
        let raw = std::fs::read_to_string(&path)?;
        let rel = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .display()
            .to_string();
        functions.extend(extract_functions(&raw, &rel));
    }
    Ok(functions)
}

fn collect_rs(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), Fail> {
    if !dir.exists() {
        return Ok(());
    }
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_rs(&path, out)?;
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            out.push(path);
        }
    }
    Ok(())
}

/// Extract production function spans. Comments and `#[cfg(test)]` items are
/// masked first, preserving byte and line positions, then braces delimit the
/// body.  A malformed or body-less declaration is ignored; shipped trait
/// signatures are not executable call-graph nodes.
fn extract_functions(raw: &str, rel: &str) -> Vec<Function> {
    let code = mask_comments_strings_and_tests(raw);
    let bytes = code.as_bytes();
    let mut out = Vec::new();
    let mut at = 0usize;
    while at + 2 <= bytes.len() {
        let Some(offset) = code[at..].find("fn ") else {
            break;
        };
        let start = at + offset;
        let before_ok =
            start == 0 || !(bytes[start - 1].is_ascii_alphanumeric() || bytes[start - 1] == b'_');
        if !before_ok {
            at = start + 1;
            continue;
        }
        let name_start = start + 3;
        let name_end = code[name_start..]
            .find(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_'))
            .map_or(bytes.len(), |offset| name_start + offset);
        if name_end == name_start {
            at = name_start;
            continue;
        }
        let Some(open) = function_body_open(&code, name_end) else {
            break;
        };
        if bytes[open] == b';' {
            at = open + 1;
            continue;
        }
        let mut depth = 1usize;
        let mut end = open + 1;
        while end < bytes.len() && depth != 0 {
            match bytes[end] {
                b'{' => depth += 1,
                b'}' => depth -= 1,
                _ => {}
            }
            end += 1;
        }
        if depth != 0 {
            break;
        }
        out.push(Function {
            name: code[name_start..name_end].to_string(),
            rel: rel.to_string(),
            line: raw[..start].bytes().filter(|&byte| byte == b'\n').count() + 1,
            body: code[open + 1..end - 1].to_string(),
            code: code[start..end].to_string(),
        });
        at = end;
    }
    out
}

fn extract_functions_including_tests(raw: &str, rel: &str) -> Vec<Function> {
    // The production graph must exclude tests, while a falsifiability audit
    // needs to inspect the named CG-22 differential itself. Rename only the
    // scanner's sentinel; the ordinary comment/string masking and brace parser
    // remain identical.
    let visible = raw.replace("#[cfg(test)]", "#[cfg(uor_float_audit_test)]");
    extract_functions(&visible, rel)
}

/// Find the body delimiter without mistaking a semicolon inside an array type
/// (`&mut [i8; N]`) for a body-less trait declaration. The prior flat search
/// silently omitted exactly the fixed-carrier projectors CU-11 exists to
/// inspect.
fn function_body_open(code: &str, start: usize) -> Option<usize> {
    let bytes = code.as_bytes();
    let mut parentheses = 0usize;
    let mut brackets = 0usize;
    for (at, &byte) in bytes.iter().enumerate().skip(start) {
        match byte {
            b'(' => parentheses += 1,
            b')' => parentheses = parentheses.saturating_sub(1),
            b'[' => brackets += 1,
            b']' => brackets = brackets.saturating_sub(1),
            b'{' => return Some(at),
            b';' if parentheses == 0 && brackets == 0 => return Some(at),
            _ => {}
        }
    }
    None
}

/// Replace comments, strings and exact `cfg(test)` items/statements with spaces
/// while retaining newlines and byte offsets. The scanner is intentionally
/// small, but unlike a line grep it cannot be satisfied by a comment naming an
/// operation, and local instrumentation cannot erase following production.
fn mask_comments_strings_and_tests(raw: &str) -> String {
    let masked = mask_comments_and_strings(raw);

    // Test items are removed after comment masking, so a comment containing
    // `#[cfg(test)]` cannot hide production code. Delimiter depth distinguishes
    // a local closure from the body of its attached item.
    let mut bytes = masked.into_bytes();
    let mut search = 0usize;
    while let Some(offset) = String::from_utf8_lossy(&bytes[search..]).find("#[cfg(test)]") {
        let attr = search + offset;
        let attached = attr + "#[cfg(test)]".len();
        let Some(end) = cfg_attached_end(&bytes, attached) else {
            break;
        };
        for index in attr..end {
            blank(&mut bytes, index);
        }
        search = end;
    }
    String::from_utf8(bytes).expect("masking preserves UTF-8 ASCII positions")
}

fn cfg_attached_end(bytes: &[u8], start: usize) -> Option<usize> {
    let mut parentheses = 0usize;
    let mut brackets = 0usize;
    let mut at = start;
    while at < bytes.len() {
        match bytes[at] {
            b'(' => parentheses += 1,
            b')' => parentheses = parentheses.checked_sub(1)?,
            b'[' => brackets += 1,
            b']' => brackets = brackets.checked_sub(1)?,
            b'{' if parentheses == 0 && brackets == 0 => {
                let mut braces = 1usize;
                at += 1;
                while at < bytes.len() && braces != 0 {
                    match bytes[at] {
                        b'{' => braces += 1,
                        b'}' => braces -= 1,
                        _ => {}
                    }
                    at += 1;
                }
                return (braces == 0).then_some(at);
            }
            b';' if parentheses == 0 && brackets == 0 => return Some(at + 1),
            _ => {}
        }
        at += 1;
    }
    None
}

fn mask_comments_and_strings(raw: &str) -> String {
    let mut bytes = raw.as_bytes().to_vec();
    let mut at = 0usize;
    let mut block_depth = 0usize;
    let mut string = false;
    let mut character = false;
    while at < bytes.len() {
        if block_depth != 0 {
            if bytes.get(at..at + 2) == Some(b"/*") {
                blank(&mut bytes, at);
                blank(&mut bytes, at + 1);
                block_depth += 1;
                at += 2;
            } else if bytes.get(at..at + 2) == Some(b"*/") {
                blank(&mut bytes, at);
                blank(&mut bytes, at + 1);
                block_depth -= 1;
                at += 2;
            } else {
                blank(&mut bytes, at);
                at += 1;
            }
            continue;
        }
        if string {
            if bytes[at] == b'\\' {
                blank(&mut bytes, at);
                if at + 1 < bytes.len() {
                    blank(&mut bytes, at + 1);
                }
                at += 2;
            } else {
                let end = bytes[at] == b'"';
                blank(&mut bytes, at);
                at += 1;
                string = !end;
            }
            continue;
        }
        if character {
            if bytes[at] == b'\\' {
                blank(&mut bytes, at);
                if at + 1 < bytes.len() {
                    blank(&mut bytes, at + 1);
                }
                at += 2;
            } else {
                let end = bytes[at] == b'\'';
                blank(&mut bytes, at);
                at += 1;
                character = !end;
            }
            continue;
        }
        if bytes.get(at..at + 2) == Some(b"//") {
            while at < bytes.len() && bytes[at] != b'\n' {
                bytes[at] = b' ';
                at += 1;
            }
        } else if bytes.get(at..at + 2) == Some(b"/*") {
            blank(&mut bytes, at);
            blank(&mut bytes, at + 1);
            block_depth = 1;
            at += 2;
        } else if bytes[at] == b'"' {
            blank(&mut bytes, at);
            string = true;
            at += 1;
        } else if bytes[at] == b'\'' && looks_like_char_literal(&bytes, at) {
            blank(&mut bytes, at);
            character = true;
            at += 1;
        } else {
            at += 1;
        }
    }

    String::from_utf8(bytes).expect("masking preserves UTF-8 ASCII positions")
}

fn blank(bytes: &mut [u8], at: usize) {
    if bytes[at] != b'\n' {
        bytes[at] = b' ';
    }
}

fn looks_like_char_literal(bytes: &[u8], at: usize) -> bool {
    matches!(
        (bytes.get(at + 1), bytes.get(at + 2), bytes.get(at + 3)),
        (Some(b'\\'), Some(_), Some(b'\'')) | (Some(_), Some(b'\''), _)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finite_atlas_and_complete_encode_use_radix_recurrences_cu_11() {
        let atlas = include_str!("../../crates/uor-matmul-core/src/float_atlas.rs");
        let accumulator = include_str!("../../crates/uor-matmul-core/src/acc.rs");
        let mut violations = Vec::new();
        audit_float_radix_sources(atlas, accumulator, &mut violations);
        assert!(violations.is_empty(), "{violations:?}");

        let shifted_naf = plant(atlas, "            unit /= 2;", "            unit=unit>>1;");
        violations.clear();
        audit_float_radix_sources(&shifted_naf, accumulator, &mut violations);
        assert!(
            violations.iter().any(|violation| {
                violation.contains("finite Atlas section` contains legacy bitwise/scan token")
            }),
            "the planted NAF shift was not rejected: {violations:?}"
        );

        let shifted_encode = plant(
            accumulator,
            "                    sig /= 2;",
            "                    sig >>= 1;",
        );
        violations.clear();
        audit_float_radix_sources(atlas, &shifted_encode, &mut violations);
        assert!(
            violations.iter().any(|violation| {
                violation.contains("Complete/encode` contains legacy bitwise/scan token")
            }),
            "the planted terminal-encode shift was not rejected: {violations:?}"
        );

        let hidden_helper = plant(
            accumulator,
            "        word /= 2;",
            "        word = packed_halve(word);",
        )
        .replacen(
            "pub struct Complete",
            "fn packed_halve(word: u64) -> u64 { word>>1 }\npub struct Complete",
            1,
        );
        violations.clear();
        audit_float_radix_sources(atlas, &hidden_helper, &mut violations);
        assert!(
            violations.iter().any(|violation| {
                violation.contains(
                    "live Complete/encode helper `packed_halve` contains a legacy shift operator",
                )
            }),
            "the planted pre-struct helper shift was not rejected: {violations:?}"
        );

        let packed_limbs = plant(
            accumulator,
            "        let mut word = self.limb(limb);",
            "        let mut word = self.low.window(at, n);",
        );
        violations.clear();
        audit_float_radix_sources(atlas, &packed_limbs, &mut violations);
        assert!(
            violations.iter().any(|violation| {
                violation.contains("calls packed generic Limbs operation `window`")
            }),
            "the planted generic Limbs observer was not rejected: {violations:?}"
        );

        let fixed_precision = plant(
            accumulator,
            "    while remaining != 0 {",
            "    for _ in 0..64 {",
        );
        violations.clear();
        audit_float_radix_sources(atlas, &fixed_precision, &mut violations);
        assert!(
            violations.iter().any(|violation| violation
                .contains("radix spread lacks pure recurrence witness `whileremaining!=0`")),
            "the planted fixed-precision fracture was not rejected: {violations:?}"
        );
    }

    fn clean_panel_fixture() -> String {
        include_str!("../../crates/uor-matmul-gemm/src/float.rs").to_string()
    }

    fn clean_dispatch_fixture() -> String {
        include_str!("../../crates/uor-matmul-gemm/src/generated_atlas_dispatch.rs").to_string()
    }

    fn fixture_max_tile_lanes() -> usize {
        let capacity = include_str!("../../crates/uor-matmul-kernels/src/generated_capacity.rs");
        constant_initializer(capacity, "MAX_TILE_LANES")
            .and_then(parse_usize_product)
            .expect("the fixture kernel extent is evaluable")
    }

    fn fixture_max_source_sites() -> usize {
        let capacity = include_str!("../../crates/uor-matmul-kernels/src/generated_capacity.rs");
        constant_initializer(capacity, "MAX_ATLAS_SOURCE_SITES")
            .and_then(parse_usize_product)
            .expect("the fixture source-site extent is evaluable")
    }

    fn panel_fixture_violations(source: &str) -> Vec<String> {
        panel_fixture_violations_at(source, fixture_max_tile_lanes())
    }

    fn panel_fixture_violations_at(source: &str, max_tile_lanes: usize) -> Vec<String> {
        panel_fixture_violations_with_dispatch_at(
            source,
            &clean_dispatch_fixture(),
            max_tile_lanes,
            fixture_max_source_sites(),
        )
    }

    fn panel_fixture_violations_with_dispatch_at(
        source: &str,
        dispatch: &str,
        max_tile_lanes: usize,
        max_source_sites: usize,
    ) -> Vec<String> {
        let engine = mask_comments_strings_and_tests(source);
        let dispatch = mask_comments_strings_and_tests(dispatch);
        let functions = extract_functions(source, ATLAS_ENGINE_FILE);
        let mut violations = Vec::new();
        audit_panel_execution_storage(
            &engine,
            &dispatch,
            &functions,
            max_tile_lanes,
            max_source_sites,
            &mut violations,
        );
        violations
    }

    fn selector_fixture_violations(source: &str) -> Vec<String> {
        let engine = mask_comments_strings_and_tests(source);
        let functions = extract_functions(source, ATLAS_ENGINE_FILE);
        let mut violations = Vec::new();
        audit_atlas_tile_selector(&engine, &functions, &mut violations);
        violations
    }

    fn dot_cache_fixture_violations(source: &str) -> Vec<String> {
        let engine = mask_comments_strings_and_tests(source);
        let functions = extract_functions(source, ATLAS_ENGINE_FILE);
        let mut violations = Vec::new();
        audit_atlas_dot_selector_cache(&engine, source, &functions, &mut violations);
        violations
    }

    fn model_differential_fixture_violations(source: &str) -> Vec<String> {
        let functions = extract_functions_including_tests(source, ATLAS_ENGINE_FILE);
        let mut violations = Vec::new();
        audit_model_storage_differential(&functions, &mut violations);
        violations
    }

    fn tabulated_float_fixture_violations(
        tabulated: &str,
        table: &str,
        float: &str,
    ) -> Vec<String> {
        let mut functions = extract_functions(tabulated, TABULATED_FILE);
        functions.extend(extract_functions(table, TABLE_FILE));
        functions.extend(extract_functions(float, ATLAS_ENGINE_FILE));
        let mut violations = Vec::new();
        audit_tabulated_float_sources(tabulated, table, &functions, &mut violations);
        violations
    }

    fn stream_fixture_violations(tabulated: &str, float: &str) -> Vec<String> {
        let mut functions = extract_functions(tabulated, TABULATED_FILE);
        functions.extend(extract_functions(float, ATLAS_ENGINE_FILE));
        let mut violations = Vec::new();
        audit_tabulated_atlas_stream(
            &mask_comments_strings_and_tests(tabulated),
            &functions,
            &mut violations,
        );
        audit_tabulated_stream_differentials(
            &extract_functions_including_tests(tabulated, TABULATED_FILE),
            &mut violations,
        );
        violations
    }

    fn one_dot_fixture_violations(tabulated: &str, element: &str) -> Vec<String> {
        let declaration = format!("impl Tabulated for {element}");
        let block = impl_block(tabulated, &declaration)
            .unwrap_or_else(|| panic!("the fixture has no `{declaration}`"));
        let methods = extract_functions(block, TABULATED_FILE);
        let dense = methods
            .iter()
            .find(|method| method.name == "dense_gemm")
            .expect("the fixture float impl has dense_gemm");
        let mut violations = Vec::new();
        audit_tabulated_one_dot(&declaration, dense, &mut violations);
        violations
    }

    fn plant(source: &str, from: &str, to: &str) -> String {
        assert!(
            source.contains(from),
            "the falsifiability plant anchor disappeared: {from}"
        );
        source.replacen(from, to, 1)
    }

    #[test]
    fn owned_stack_carriers_make_the_structural_gate_fail_ca_05() {
        let clean = "enum AtlasCarrierSource<'a> { Lattice(&'a [i64; 24]), Word(&'a [D; 8]) } \
                     struct AtlasCarrier<'a> { source: AtlasCarrierSource<'a> } \
                     struct AtlasBlocks<'a> { carrier: AtlasCarrier<'a> }";
        let mut violations = Vec::new();
        audit_borrowed_carrier_declarations(clean, &mut violations);
        assert!(violations.is_empty(), "{violations:?}");

        let owned_carrier = "enum AtlasCarrierSource<'a> { Lattice(&'a [i64; 24]), Word(&'a [D; 8]) } \
                             struct AtlasCarrier<'a> { source: AtlasCarrierSource<'a>, coordinates: [i128; 24] } \
                             struct AtlasBlocks<'a> { carrier: AtlasCarrier<'a> }";
        audit_borrowed_carrier_declarations(owned_carrier, &mut violations);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("inline carrier")),
            "the planted inline carrier was not rejected: {violations:?}"
        );

        violations.clear();
        let owned_blocks = "enum AtlasCarrierSource<'a> { Lattice(&'a [i64; 24]), Word(&'a [D; 8]) } \
                            struct AtlasCarrier<'a> { source: AtlasCarrierSource<'a> } \
                            struct AtlasBlocks<'a> { carrier: AtlasCarrier<'a>, global: [i128; 24] }";
        audit_borrowed_carrier_declarations(owned_blocks, &mut violations);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("materialized")),
            "the planted projector materialization was not rejected: {violations:?}"
        );
    }

    #[test]
    fn scanner_has_clean_and_dirty_teeth() {
        let clean = "fn root() { atlas_lookup(); } fn atlas_lookup() { complete_add(); } \
                     fn complete_add() { dyadic_place(); } fn dyadic_place() {}";
        let functions = extract_functions(clean, ATLAS_REFERENCE_FILE);
        assert_eq!(functions.len(), 4);
        assert!(contains_call(&functions[0].body, "atlas_lookup"));
        assert!(!contains_call(&functions[0].body, "run_bridge"));

        let dirty = "fn root() { run_bridge(); } fn run_bridge() {}";
        let functions = extract_functions(dirty, "crates/uor-matmul-gemm/src/float.rs");
        assert!(contains_call(&functions[0].body, "run_bridge"));
        assert_eq!(functions[1].name, "run_bridge");

        let fixed = "fn project(out: &mut [i8; 9]) { lookup(out); }";
        let functions = extract_functions(fixed, TABLE_FILE);
        assert_eq!(
            functions.len(),
            1,
            "an array semicolon is not a declaration end"
        );
        assert!(contains_call(&functions[0].body, "lookup"));

        let local_instrumentation = "fn production() { #[cfg(test)] instrumentation(|| { observed(); }); atlas_lookup(); } \
                                     #[cfg(test)] mod tests { fn only_test() { legacy(); } }";
        let functions = extract_functions(local_instrumentation, ATLAS_ENGINE_FILE);
        assert_eq!(
            functions
                .iter()
                .map(|function| function.name.as_str())
                .collect::<Vec<_>>(),
            ["production"],
            "a local cfg(test) statement must not erase following production code or expose the test module"
        );
        assert!(contains_call(&functions[0].body, "atlas_lookup"));
    }

    #[test]
    fn native_product_table_accessor_is_the_canonical_aligned_borrow_cu_11() {
        let clean = include_str!("../../crates/uor-matmul-kernels/src/lookup.rs");
        let functions = extract_functions(clean, LOOKUP_FILE);
        let mut violations = Vec::new();
        audit_i8_product_lookup(&functions, &mut violations);
        assert!(violations.is_empty(), "{violations:?}");

        let severed = plant(clean, "    &I8_PRODUCTS\n}", "    &SEVERED_I8_PRODUCTS\n}");
        let functions = extract_functions(&severed, LOOKUP_FILE);
        violations.clear();
        audit_i8_product_lookup(&functions, &mut violations);
        assert!(
            violations.iter().any(|violation| {
                violation.contains("does not borrow exactly the canonical aligned product alphabet")
            }),
            "the severed product-table accessor was not rejected: {violations:?}"
        );
    }

    #[test]
    fn native_product_alphabet_has_the_exact_hidden_elf_symbol_cu_11() {
        let clean = include_str!("../../crates/uor-matmul-kernels/src/lookup.rs");
        let functions = extract_functions(clean, LOOKUP_FILE);
        let mut violations = Vec::new();
        audit_i8_product_elf_visibility(clean, &mut violations);
        audit_i8_product_native_address_seam(clean, &functions, &mut violations);
        assert!(violations.is_empty(), "{violations:?}");

        let exposed = plant(
            clean,
            ".hidden __uor_matmul_kernels_v0_1_0_i8_products",
            ".globl __uor_matmul_kernels_v0_1_0_i8_products",
        );
        violations.clear();
        audit_i8_product_elf_visibility(&exposed, &mut violations);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("exact hidden ELF symbol")),
            "the planted exposed product alphabet was not rejected: {violations:?}"
        );

        let computational = plant(
            clean,
            "lea {address}, [rip + {table}]",
            "mov {address}, [rip + {table}]",
        );
        let functions = extract_functions(&computational, LOOKUP_FILE);
        violations.clear();
        audit_i8_product_native_address_seam(&computational, &functions, &mut violations);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("address-only RIP-relative LEA")),
            "the planted memory-reading native seam was not rejected: {violations:?}"
        );
    }

    #[test]
    fn native_reduction_pairing_preserves_both_halves_cu_11() {
        let clean = include_str!("../../crates/uor-matmul-kernels/src/isa/x86.rs");
        let functions = extract_functions(clean, "crates/uor-matmul-kernels/src/isa/x86.rs");
        let mut violations = Vec::new();
        audit_avx2_lookup_reduce_pairing(&functions, &mut violations);
        assert!(violations.is_empty(), "{violations:?}");

        let omitted_second_half = plant(
            clean,
            "            sum = _mm256_add_epi32(sum, products1);",
            "            let _omitted_products1 = products1;",
        );
        let functions = extract_functions(
            &omitted_second_half,
            "crates/uor-matmul-kernels/src/isa/x86.rs",
        );
        violations.clear();
        audit_avx2_lookup_reduce_pairing(&functions, &mut violations);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("ordered paired halves")),
            "the planted omitted second reduction half was not rejected: {violations:?}"
        );

        let duplicated_first_half = plant(
            clean,
            "                unsafe { avx2_lookup_reduce_octet(product_alphabet, left_octets1, right_octets1) };",
            "                unsafe { avx2_lookup_reduce_octet(product_alphabet, left_octets0, right_octets0) };",
        );
        let functions = extract_functions(
            &duplicated_first_half,
            "crates/uor-matmul-kernels/src/isa/x86.rs",
        );
        violations.clear();
        audit_avx2_lookup_reduce_pairing(&functions, &mut violations);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("ordered paired halves")),
            "the planted duplicated first reduction half was not rejected: {violations:?}"
        );
    }

    #[test]
    fn native_lookup_acceptance_uses_symmetric_safe_wrappers_cu_11() {
        let clean =
            include_str!("../../crates/uor-matmul-validate/benches/scaling/native_lookup.rs");
        let mut violations = Vec::new();
        audit_native_lookup_clock_wrappers(clean, &mut violations);
        assert!(violations.is_empty(), "{violations:?}");

        let asymmetric = plant(
            clean,
            "|output| clock_kernel(&raw_spec, kc, &pa, &pb, output),",
            "|output| raw_spec.mac_tile(kc, &pa, &pb, output),",
        );
        violations.clear();
        audit_native_lookup_clock_wrappers(&asymmetric, &mut violations);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("identical safe wrappers")),
            "the planted asymmetric native clock was not rejected: {violations:?}"
        );
    }

    #[test]
    fn native_lookup_measurement_protocol_distinguishes_changed_and_static_controls_cg_23() {
        let clean =
            include_str!("../../crates/uor-matmul-validate/benches/scaling/native_lookup.rs");
        let mut violations = Vec::new();
        audit_native_lookup_clock_wrappers(clean, &mut violations);
        audit_native_lookup_acceptance_protocol(clean, &mut violations);
        assert!(violations.is_empty(), "{violations:?}");

        let relabeled_changed_case = plant(
            clean,
            "    (\n        AcceptanceCase::Tile {\n            rows: 1,\n            columns: 8,\n        },\n        AcceptanceClass::Changed,\n    ),",
            "    (\n        AcceptanceCase::Tile {\n            rows: 1,\n            columns: 8,\n        },\n        AcceptanceClass::StaticEquivalent,\n    ),",
        );
        violations.clear();
        audit_native_lookup_acceptance_protocol(&relabeled_changed_case, &mut violations);
        assert!(
            violations.iter().any(|violation| violation
                .contains("exact closed changed/static-control classification")),
            "the planted relabeling of a changed case was not rejected: {violations:?}"
        );

        let asymmetric = plant(
            clean,
            "|output| clock_kernel(&raw_spec, kc, &pa, &pb, output),",
            "|output| raw_spec.mac_tile(kc, &pa, &pb, output),",
        );
        violations.clear();
        audit_native_lookup_clock_wrappers(&asymmetric, &mut violations);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("identical safe wrappers")),
            "the planted asymmetric CG-23 wrapper was not rejected: {violations:?}"
        );
    }

    #[test]
    fn native_nibble_lookup_borrows_the_canonical_safe_row_cu_11() {
        let clean = include_str!("../../crates/uor-matmul-kernels/src/lookup.rs");
        let functions = extract_functions(clean, LOOKUP_FILE);
        let mut violations = Vec::new();
        audit_i8_nibble_lookup(&functions, &mut violations);
        assert!(violations.is_empty(), "{violations:?}");

        let fixed_row = plant(
            clean,
            "    &I8_NIBBLE_PRODUCTS[a as u8 as usize]",
            "    &I8_NIBBLE_PRODUCTS[0]",
        );
        let functions = extract_functions(&fixed_row, LOOKUP_FILE);
        violations.clear();
        audit_i8_nibble_lookup(&functions, &mut violations);
        assert!(
            violations.iter().any(|violation| {
                violation.contains("canonical safe row indexed by its signed-octet code")
            }),
            "the planted fixed projector row was not rejected: {violations:?}"
        );
    }

    #[test]
    fn native_lookup_helpers_reach_the_nibble_projector_cu_11() {
        let clean = "fn neon_lookup_i8() { neon_lookup_accumulate(); } \
                     fn neon_lookup_accumulate() { neon_nibble_products(); } \
                     fn neon_nibble_products() { i8_nibble_products(); }";
        let functions = extract_functions(clean, "crates/uor-matmul-kernels/src/isa/arm.rs");
        let mut violations = Vec::new();
        assert!(audit_native_nibble_terminus(
            &functions,
            &functions[0],
            "neon_nibble_products",
            &mut violations,
        ));
        assert!(violations.is_empty(), "{violations:?}");

        let scalar = "fn neon_lookup_i8() { neon_lookup_accumulate(); scalar_detour(); } \
                      fn neon_lookup_accumulate() { neon_nibble_products(); } \
                      fn neon_nibble_products() { i8_nibble_products(); } \
                      fn scalar_detour() { i8_product(); }";
        let functions = extract_functions(scalar, "crates/uor-matmul-kernels/src/isa/arm.rs");
        assert!(!audit_native_nibble_terminus(
            &functions,
            &functions[0],
            "neon_nibble_products",
            &mut violations,
        ));
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("scalar i8 product table")),
            "the planted scalar-table terminus was not rejected: {violations:?}"
        );

        violations.clear();
        let severed = "fn neon_lookup_i8() { unrelated(); } \
                       fn neon_nibble_products() { i8_nibble_products(); }";
        let functions = extract_functions(severed, "crates/uor-matmul-kernels/src/isa/arm.rs");
        assert!(!audit_native_nibble_terminus(
            &functions,
            &functions[0],
            "neon_nibble_products",
            &mut violations,
        ));

        // All four AVX2 lookup declarations share this grouped runtime and
        // terminal projector edge. Severing that one edge must therefore make
        // every declaration fail the live family audit; checking only the
        // helper in isolation would leave declaration coverage assumptive.
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("xtask is directly below the repository root");
        let x86_rel = "crates/uor-matmul-kernels/src/isa/x86.rs";
        let x86 = include_str!("../../crates/uor-matmul-kernels/src/isa/x86.rs");
        let severed_x86 = plant(
            x86,
            "        let row = crate::lookup::i8_nibble_products(left);",
            "        let row = crate::lookup::severed_i8_nibble_products(left);",
        );
        let mut functions = shipped_functions(root).expect("the shipped graph is readable");
        functions.retain(|function| function.rel != x86_rel);
        functions.extend(extract_functions(&severed_x86, x86_rel));
        let grouped = functions
            .iter()
            .find(|function| function.rel == x86_rel && function.name == "avx2_lookup_i8")
            .expect("the grouped AVX2 lookup is in the shipped graph");
        let direct = functions
            .iter()
            .find(|function| function.rel == x86_rel && function.name == "avx2_lookup_reduce_i8")
            .expect("the direct AVX2 reduce lookup is in the shipped graph");
        violations.clear();
        assert!(
            !audit_group_one_kernel_terminus(&functions, grouped, &mut violations),
            "the severed grouped edge was accepted"
        );
        violations.clear();
        assert!(
            audit_group_one_kernel_terminus(&functions, direct, &mut violations),
            "the direct x86 lookup was sent through the grouped helper arm: {violations:?}"
        );
        assert!(violations.is_empty(), "{violations:?}");

        let lookup = include_str!("../../crates/uor-matmul-kernels/src/lookup.rs");
        let severed_products = plant(lookup, "    &I8_PRODUCTS\n}", "    &SEVERED_I8_PRODUCTS\n}");
        let mut severed_product_functions = functions.clone();
        severed_product_functions.retain(|function| function.rel != LOOKUP_FILE);
        severed_product_functions.extend(extract_functions(&severed_products, LOOKUP_FILE));
        let direct = severed_product_functions
            .iter()
            .find(|function| function.rel == x86_rel && function.name == "avx2_lookup_reduce_i8")
            .expect("the direct AVX2 reduce lookup remains in the shipped graph");
        violations.clear();
        assert!(
            !audit_group_one_kernel_terminus(&severed_product_functions, direct, &mut violations,),
            "the direct x86 lookup accepted a severed product-table accessor"
        );

        let failure = audit_support_files(root, &functions)
            .expect_err("the severed AVX2 projector edge must fail the support audit")
            .to_string();
        for symbol in [
            "AVX2_LOOKUP_I8_I32_M1",
            "AVX2_LOOKUP_I8_I32",
            "AVX2_LOOKUP_I8_I32_M1_N8",
            "AVX2_LOOKUP_I8_I32_N8",
        ] {
            assert!(
                failure.contains(&format!(
                    "group-one member `{symbol}` does not terminate in the canonical i8 product table"
                )),
                "the severed shared AVX2 projector edge did not reject `{symbol}`: {failure}"
            );
        }
    }

    #[test]
    fn later_neon_lookup_proves_vmull_cannot_win_cu_11() {
        let spec = include_str!("../../crates/uor-matmul-kernels/src/spec.rs");
        let arm = include_str!("../../crates/uor-matmul-kernels/src/isa/arm.rs");
        let family = family_named(spec, "available_i8").unwrap();
        let mut violations = Vec::new();
        audit_neon_vmull_shadow(family, arm, &mut violations);
        assert!(violations.is_empty(), "{violations:?}");

        let removed = family.replace("crate::isa::arm::NEON_LOOKUP_I8_I32", "REMOVED_LOOKUP");
        audit_neon_vmull_shadow(&removed, arm, &mut violations);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("not shadowed")),
            "the planted removed lookup did not expose the vmull winner: {violations:?}"
        );

        violations.clear();
        let vmull_line = family
            .lines()
            .find(|line| line.contains("=> crate::isa::arm::NEON_I8_I32,"))
            .unwrap();
        let lookup_line = family
            .lines()
            .find(|line| line.contains("=> crate::isa::arm::NEON_LOOKUP_I8_I32,"))
            .unwrap();
        let reordered = family
            .replace(vmull_line, "NEON_ORDER_MARKER")
            .replace(lookup_line, vmull_line)
            .replace("NEON_ORDER_MARKER", lookup_line);
        audit_neon_vmull_shadow(&reordered, arm, &mut violations);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("not shadowed")),
            "the planted earlier lookup did not expose the later vmull: {violations:?}"
        );

        violations.clear();
        let differentiated = arm.replace(
            "neon_lookup_spec::<4, 8>(Backend::Neon)",
            "neon_lookup_spec::<4, 7>(Backend::Neon)",
        );
        audit_neon_vmull_shadow(family, &differentiated, &mut violations);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("not selector-equivalent")),
            "the planted differentiated lookup still shadowed vmull: {violations:?}"
        );
    }

    #[test]
    fn atlas_dot_selection_is_once_per_backend_and_shared_with_no_std_cu_11() {
        let clean = clean_panel_fixture();
        let mut violations = dot_cache_fixture_violations(&clean);
        assert!(violations.is_empty(), "{violations:?}");

        let missing_auto_slot = plant(
            &clean,
            "static ATLAS_DOT_AUTO_SPEC: std::sync::OnceLock<KernelSpec<i8, i32>> = std::sync::OnceLock::new();",
            "static ATLAS_DOT_AUTO_SPEC_WAS_REMOVED: () = ();",
        );
        violations = dot_cache_fixture_violations(&missing_auto_slot);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("per-backend dot-selector cache")),
            "the planted missing Auto cache slot was not rejected: {violations:?}"
        );

        let eager_resolution = plant(
            &clean,
            "        *slot.get_or_init(|| {",
            "        let _ = resolve_atlas_dot_spec(backend);\n        *slot.get_or_init(|| {",
        );
        violations = dot_cache_fixture_violations(&eager_resolution);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("initializer-only")),
            "the planted per-call resolution was not rejected: {violations:?}"
        );

        let split_no_std = plant(
            &clean,
            "        resolve_atlas_dot_spec(backend)\n    }\n}",
            "        uor_matmul_kernels::spec::R1_I8_I32\n    }\n}",
        );
        violations = dot_cache_fixture_violations(&split_no_std);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("direct no-std")),
            "the planted no-std selection split was not rejected: {violations:?}"
        );
    }

    #[test]
    fn avx2_lookup_m1_preserves_positive_row_winner_and_names_zero_boundary_cu_11() {
        let spec = include_str!("../../crates/uor-matmul-kernels/src/spec.rs");
        let x86 = include_str!("../../crates/uor-matmul-kernels/src/isa/x86.rs");
        let family = family_named(spec, "available_i8").unwrap();
        let mut violations = Vec::new();
        audit_avx2_lookup_m1_order(family, spec, x86, &mut violations);
        assert!(violations.is_empty(), "{violations:?}");

        let later_equal_wins = |rows: usize| {
            let incumbent_mr = 1usize;
            let candidate_mr = 1usize;
            if incumbent_mr > rows {
                candidate_mr <= rows || candidate_mr < incumbent_mr
            } else {
                candidate_mr <= rows && candidate_mr >= incumbent_mr
            }
        };
        assert!(
            !later_equal_wins(0),
            "the public zero-row boundary retains lookup"
        );
        assert!(
            later_equal_wins(1),
            "every GEMM row count restores native M1"
        );

        let reversed_family = spec.replace(
            "    crate::isa::x86::avx2_available() => crate::isa::x86::AVX2_LOOKUP_I8_I32_M1,\n    crate::isa::x86::avx2_available() => crate::isa::x86::AVX2_I8_I32_M1,",
            "    crate::isa::x86::avx2_available() => crate::isa::x86::AVX2_I8_I32_M1,\n    crate::isa::x86::avx2_available() => crate::isa::x86::AVX2_LOOKUP_I8_I32_M1,",
        );
        let reversed_family = family_named(&reversed_family, "available_i8").unwrap();
        audit_avx2_lookup_m1_order(reversed_family, spec, x86, &mut violations);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("does not precede")),
            "the planted equal-height family reordering was not rejected: {violations:?}"
        );

        let changed_boundary = spec.replace("spec.mr < b.mr", "spec.mr <= b.mr");
        audit_avx2_lookup_m1_order(family, &changed_boundary, x86, &mut violations);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("positive/zero boundary")),
            "the planted zero-row boundary change was not rejected: {violations:?}"
        );
    }

    #[test]
    fn short_float_offer_stream_has_no_scalar_escape_cu_11() {
        let tabulated = include_str!("../../crates/uor-matmul-gemm/src/tabulated.rs");
        let float = include_str!("../../crates/uor-matmul-gemm/src/float.rs");
        let mut violations = stream_fixture_violations(tabulated, float);
        assert!(violations.is_empty(), "{violations:?}");

        let no_bound_prefilter = plant(
            tabulated,
            "if Bd::VALUE != u128::MAX || triple.shape().k == 0 {",
            "if triple.shape().k == 0 {",
        );
        violations = stream_fixture_violations(&no_bound_prefilter, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("type-level bound prefilter")),
            "the planted missing bounded-alphabet prefilter was not rejected: {violations:?}"
        );

        let no_zero_depth_boundary = plant(
            tabulated,
            "if Bd::VALUE != u128::MAX || triple.shape().k == 0 {",
            "if Bd::VALUE != u128::MAX {",
        );
        violations = stream_fixture_violations(&no_zero_depth_boundary, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("first-real-partial witness")),
            "the planted zero-depth dense presentation was not rejected: {violations:?}"
        );

        let synthetic_route_call = plant(
            tabulated,
            "dense_stream(triple, epilogue, options, panel, ledger)",
            "{ ledger.kernelled(); dense_stream(triple, epilogue, options, panel, ledger) }",
        );
        violations = stream_fixture_violations(&synthetic_route_call, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("dummy or synthetic operation")),
            "the planted synthetic route kernel call was not rejected: {violations:?}"
        );

        let dummy_probe = plant(
            tabulated,
            "dense_stream(triple, epilogue, options, panel, ledger)",
            "{ let probe = [Alphabet::<E, Bd>::ZERO; 1]; let _ = probe; \
             dense_stream(triple, epilogue, options, panel, ledger) }",
        );
        violations = stream_fixture_violations(&dummy_probe, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("dummy or synthetic operation")),
            "the planted dummy capability product was not rejected: {violations:?}"
        );

        let missing_zero_dot_notification = plant(
            tabulated,
            "            ledger.multiplied(count_factor(shape.k));",
            "            if shape.k != 0 { ledger.multiplied(count_factor(shape.k)); }",
        );
        violations = stream_fixture_violations(&missing_zero_dot_notification, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("empty-dot ledger notification")),
            "the planted missing empty-dot notification was not rejected: {violations:?}"
        );

        let discarded_capture = plant(
            tabulated,
            "        self.0.set(acc);",
            "        let _ = acc;",
        );

        let undercounted_pages = plant(
            tabulated,
            "    ledger.kernelled();\n    let ran = E::dense_gemm",
            "    let ran = E::dense_gemm",
        );
        violations = stream_fixture_violations(&undercounted_pages, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("actual dense presentation")),
            "the planted page undercount was not rejected: {violations:?}"
        );

        let undercounted_full_rows = plant(
            tabulated,
            "                ledger.kernelled();\n                let ran = E::dense_gemm",
            "                let ran = E::dense_gemm",
        );
        violations = stream_fixture_violations(&undercounted_full_rows, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("full-row dense presentation")),
            "the planted full-row undercount was not rejected: {violations:?}"
        );

        let removed_full_row_batch = plant(
            tabulated,
            "                let ran = E::dense_gemm(left, right, sink, epilogue, options, &mut []);",
            "                let ran = false;",
        );
        violations = stream_fixture_violations(&removed_full_row_batch, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("real-page witness")),
            "the planted missing full-row batch was not rejected: {violations:?}"
        );

        let narrowed_full_row_batch = plant(
            tabulated,
            "                let rows = (shape.m - row0).min(ROW_TILES[0]);",
            "                let rows = (shape.m - row0).min(1);",
        );
        violations = stream_fixture_violations(&narrowed_full_row_batch, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("real-page witness")),
            "the planted arbitrary full-row batch size was not rejected: {violations:?}"
        );

        let flattened_page_formula = plant(tabulated, "m * n * k.div_ceil(source_page)", "m * n");
        violations = stream_fixture_violations(&flattened_page_formula, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("page-ledger differential")),
            "the planted short-page undercount formula was not rejected: {violations:?}"
        );
        let per_cell_full_row_formula = plant(tabulated, "n * m.div_ceil(ROW_TILES[0])", "m * n");
        violations = stream_fixture_violations(&per_cell_full_row_formula, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("page-ledger differential")),
            "the planted per-cell full-row census was not rejected: {violations:?}"
        );
        violations = stream_fixture_violations(&discarded_capture, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("does not transfer exactly one accumulator")),
            "the planted discarded first partial was not rejected: {violations:?}"
        );

        let combined_first_partial = plant(
            tabulated,
            "                        acc = partial;",
            "                        acc = acc.combine(partial);",
        );
        violations = stream_fixture_violations(&combined_first_partial, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("real-page witness")),
            "the planted redundant first-partial combine was not rejected: {violations:?}"
        );

        let uncharged_page_join = plant(
            tabulated,
            "                        ledger.added(1);\n                        acc = acc.combine(partial);",
            "                        acc = acc.combine(partial);",
        );
        violations = stream_fixture_violations(&uncharged_page_join, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("real-page witness")),
            "the planted uncharged complete-page join was not rejected: {violations:?}"
        );

        let mismatched_table_geometry = plant(
            tabulated,
            "        let table = if zeroed_rows == Some(rows) {",
            "        let table = if zeroed_rows.is_some() {",
        );
        violations = stream_fixture_violations(&mismatched_table_geometry, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("identical row geometry")),
            "the planted cross-geometry stack reuse was not rejected: {violations:?}"
        );

        let repeated_table_clear = plant(
            tabulated,
            "            Table::reuse_zeroed(stack, space, rows, plan.depth)",
            "            Table::new(stack, space, rows, plan.depth)",
        );
        violations = stream_fixture_violations(&repeated_table_clear, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("identical row geometry")),
            "the planted repeated padding clear was not rejected: {violations:?}"
        );

        let private_stream = plant(
            tabulated,
            "type StreamLane = Wide<AccOf<f32>>;",
            "type StreamLane = Scaled64;",
        );
        violations = stream_fixture_violations(&private_stream, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("stream extension point")),
            "the planted non-Wide public stream association was not rejected: {violations:?}"
        );
    }

    #[test]
    fn tabulated_dense_one_dot_enters_atlas_without_api_change_cu_11() {
        let clean = include_str!("../../crates/uor-matmul-gemm/src/tabulated.rs");
        for element in ["f32", "f64"] {
            let violations = one_dot_fixture_violations(clean, element);
            assert!(violations.is_empty(), "{element}: {violations:?}");
        }

        let severed_shape = plant(
            clean,
            "if dense.shape().m == 1 && dense.shape().n == 1",
            "if false",
        );
        let mut violations = one_dot_fixture_violations(&severed_shape, "f32");
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("one-dot bridge lacks")),
            "the planted severed one-dot shape was not rejected: {violations:?}"
        );

        let scalar = plant(
            clean,
            "            accumulate_atlas_dot(\n",
            "            Self::mac(\n",
        );
        violations = one_dot_fixture_violations(&scalar, "f32");
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("scalar escape `Self::mac(`")),
            "the planted scalar one-dot bridge was not rejected: {violations:?}"
        );

        let repeated = plant(
            clean,
            "        gemm_float(&mut dense, epilogue, options);",
            "        accumulate_atlas_dot(&mut acc, 0, PanelFacts::UNKNOWN, options.backend, |_| ZERO, |_| ZERO); \
             gemm_float(&mut dense, epilogue, options);",
        );
        violations = one_dot_fixture_violations(&repeated, "f32");
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("exactly once")),
            "the planted second Atlas entry was not rejected: {violations:?}"
        );
    }

    #[test]
    fn atlas_source_cell_is_single_and_kernel_bounded_ca_05() {
        let clean = clean_panel_fixture();
        let mut violations = panel_fixture_violations(&clean);
        assert!(violations.is_empty(), "{violations:?}");

        let literal_budget = plant(
            &clean,
            "source_kinds: [u8; MAX_ATLAS_SOURCE_SITES]",
            "source_kinds: [u8; 129]",
        );
        violations = panel_fixture_violations(&literal_budget);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("derived bounded-frame/product witness")),
            "the planted literal source-cell budget was not rejected: {violations:?}"
        );

        let twin_stores = clean.replace(
            "source_kinds: [AtlasProjectedKind::FiniteZero as u8; MAX_ATLAS_SOURCE_SITES],",
            "source_kinds: [AtlasProjectedKind::FiniteZero as u8; MAX_ATLAS_SOURCE_SITES], \
             a_worst_case: [ZERO_CODE; MAX_ATLAS_SOURCE_SITES], \
             b_worst_case: [ZERO_CODE; MAX_ATLAS_SOURCE_SITES],",
        );
        violations = panel_fixture_violations(&twin_stores);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("separate A/B full-code arrays")),
            "the planted separate A/B code stores were not rejected: {violations:?}"
        );

        let aliased_negative = plant(
            &clean,
            "            if negative {\n                AtlasProjectedKind::FiniteNegative as u8",
            "            if negative {\n                AtlasProjectedKind::FinitePositive as u8",
        );
        violations = panel_fixture_violations(&aliased_negative);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("aliases or omits boundary state")),
            "the planted negative boundary alias was not rejected: {violations:?}"
        );

        let redundant_projection_flags = plant(
            &clean,
            "    source_extents: [u8; MAX_ATLAS_SOURCE_SITES],",
            "    source_extents: [u8; MAX_ATLAS_SOURCE_SITES],\n    source_projected: [bool; MAX_ATLAS_SOURCE_SITES],",
        );
        violations = panel_fixture_violations(&redundant_projection_flags);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("source_projected")),
            "the planted redundant projection-state array was not rejected: {violations:?}"
        );

        let duplicated_modality = plant(
            &clean,
            "    source_finite: [AtlasFiniteSite; MAX_ATLAS_SOURCE_SITES],",
            "    source_atoms: [Option<AtlasAtom>; MAX_ATLAS_SOURCE_SITES],",
        );
        violations = panel_fixture_violations(&duplicated_modality);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("source_atoms:")),
            "the planted duplicated finite modality was not rejected: {violations:?}"
        );
    }

    #[test]
    fn atlas_source_cell_is_lazy_and_independent_of_depth_cu_11() {
        let clean = clean_panel_fixture();
        let mut violations = panel_fixture_violations(&clean);
        assert!(violations.is_empty(), "{violations:?}");

        let depth_store = clean.replace(
            "[AtlasProjectedKind::FiniteZero as u8; MAX_ATLAS_SOURCE_SITES]",
            "[AtlasProjectedKind::FiniteZero as u8; ATLAS_PANEL_SITES]",
        );
        violations = panel_fixture_violations(&depth_store);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("depth-panel token")),
            "the planted depth-sized source store was not rejected: {violations:?}"
        );

        let materialized = plant(
            &clean,
            "dispatch_atlas_cell_capacity!(cell_capacity, execute);",
            "let mut code_storage = [ZERO_CODE; shape.k]; \
             dispatch_atlas_cell_capacity!(cell_capacity, execute);",
        );
        violations = panel_fixture_violations(&materialized);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("depth-panel token")),
            "the planted materialized depth carrier was not rejected: {violations:?}"
        );

        let severed_source = clean.replace("source_a(p, i, ledger)", "ZERO_CODE");
        violations = panel_fixture_violations(&severed_source);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("source_a(p,i,ledger)")),
            "the planted severed lazy source was not rejected: {violations:?}"
        );
    }

    #[test]
    fn atlas_output_frame_is_const_generic_and_one_pass_ca_05() {
        let clean = clean_panel_fixture();
        let mut violations = panel_fixture_violations(&clean);
        assert!(violations.is_empty(), "{violations:?}");

        let padded_extent = plant(
            &clean,
            "let cell_capacity = rows * cols;",
            "let cell_capacity = MAX_TILE_LANES;",
        );
        violations = panel_fixture_violations(&padded_extent);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("bounded-frame dispatch witness")),
            "the planted padded cell extent was not rejected: {violations:?}"
        );

        let fixed_maximum = plant(
            &clean,
            "let mut accumulators = [<AccOf<E> as Accumulator>::ZERO; CELL_CAP];",
            "let mut accumulators = [<AccOf<E> as Accumulator>::ZERO; MAX_TILE_LANES];",
        );
        violations = panel_fixture_violations(&fixed_maximum);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("const-generic output frame")),
            "the planted maximum-sized frame was not rejected: {violations:?}"
        );

        let repeated_argument_setup = plant(
            &clean,
            "execute_atlas_output_tile::<$cell_cap, E, O, Ep, P, Lg>(&mut context)",
            "execute_atlas_output_tile::<$cell_cap, E, O, Ep, P, Lg>(context.a)",
        );
        violations = panel_fixture_violations(&repeated_argument_setup);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("&mutcontext")),
            "the planted per-arm argument setup was not rejected: {violations:?}"
        );

        let body_start = clean
            .find("fn execute_atlas_output_tile_body")
            .expect("the one-pass output body plant anchor exists");
        let replay_start = body_start
            + clean[body_start..]
                .find("    accumulate_direct_atlas_tile(\n")
                .expect("the one-pass contraction plant anchor exists");
        let replay_tail = &clean[replay_start..];
        let replay_end = replay_tail
            .find("\n    );")
            .expect("the one-pass contraction has a complete call")
            + "\n    );".len();
        let contraction = &replay_tail[..replay_end];
        let replayed_projection =
            clean.replacen(contraction, &format!("{contraction}\n{contraction}"), 1);
        violations = panel_fixture_violations(&replayed_projection);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("post-bypass physical tile")),
            "the planted output-window re-projection was not rejected: {violations:?}"
        );

        let discarded_projection_cache = plant(
            &clean,
            "AtlasSource::Projected(AtlasProjectedCode::from_packed(pa[ii * shape.k + p]))",
            "AtlasSource::Raw(pa[ii * shape.k + p])",
        );
        violations = panel_fixture_violations(&discarded_projection_cache);
        assert!(
            violations.iter().any(
                |violation| violation.contains("cached/raw projection reuse")
                    || violation.contains("shared slice execution body")
            ),
            "the planted cached-source re-projection was not rejected: {violations:?}"
        );

        let dispatch = clean_dispatch_fixture();
        let no_irreducible_frame = plant(
            &dispatch,
            "            1 => $execute!(1),",
            "            1 => $execute!(MAX_TILE_LANES),",
        );
        violations = panel_fixture_violations_with_dispatch_at(
            &clean,
            &no_irreducible_frame,
            fixture_max_tile_lanes(),
            fixture_max_source_sites(),
        );
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("mismatched arm at capacity 1")),
            "the planted missing one-cell frame was not rejected: {violations:?}"
        );
    }

    #[test]
    fn atlas_capacity_dispatch_is_exact_contiguous_and_terminal_cg_22() {
        let clean = clean_panel_fixture();
        let dispatch = clean_dispatch_fixture();
        let mut violations = panel_fixture_violations(&clean);
        assert!(violations.is_empty(), "{violations:?}");

        let missing = plant(&dispatch, "            64 => $execute!(64),\n", "");
        violations = panel_fixture_violations_with_dispatch_at(
            &clean,
            &missing,
            fixture_max_tile_lanes(),
            fixture_max_source_sites(),
        );
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("arm at capacity 64")),
            "the planted missing capacity was not rejected: {violations:?}"
        );

        let duplicate = plant(
            &dispatch,
            "            64 => $execute!(64),",
            "            63 => $execute!(64),",
        );
        violations = panel_fixture_violations_with_dispatch_at(
            &clean,
            &duplicate,
            fixture_max_tile_lanes(),
            fixture_max_source_sites(),
        );
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("arm at capacity 64")),
            "the planted duplicate capacity was not rejected: {violations:?}"
        );

        let mismatched = plant(
            &dispatch,
            "            64 => $execute!(64),",
            "            64 => $execute!(65),",
        );
        violations = panel_fixture_violations_with_dispatch_at(
            &clean,
            &mismatched,
            fixture_max_tile_lanes(),
            fixture_max_source_sites(),
        );
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("mismatched arm at capacity 64")),
            "the planted mismatched const frame was not rejected: {violations:?}"
        );

        let literal_terminal = plant(
            &dispatch,
            "            MAX_TILE_LANES => $execute!(MAX_TILE_LANES),",
            "            128 => $execute!(128),",
        );
        violations = panel_fixture_violations_with_dispatch_at(
            &clean,
            &literal_terminal,
            fixture_max_tile_lanes(),
            fixture_max_source_sites(),
        );
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("dispatch terminal")),
            "the planted stale literal terminal was not rejected: {violations:?}"
        );

        let spec_drift = panel_fixture_violations_at(&clean, 129);
        assert!(
            spec_drift
                .iter()
                .any(|violation| violation.contains("requires exactly 129")),
            "the planted kernel-extent drift was not rejected: {spec_drift:?}"
        );

        let source_sites = fixture_max_source_sites();
        let source_headroom = plant(
            &dispatch,
            &format!("const MAX_ATLAS_SOURCE_SITES: usize = {source_sites};"),
            &format!(
                "const MAX_ATLAS_SOURCE_SITES: usize = {};",
                source_sites + 1
            ),
        );
        violations = panel_fixture_violations_with_dispatch_at(
            &clean,
            &source_headroom,
            fixture_max_tile_lanes(),
            source_sites,
        );
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("Atlas source capacity")),
            "the planted source-site headroom was not rejected: {violations:?}"
        );
    }

    #[test]
    fn kernel_capacity_is_exact_over_every_declared_family_cg_22() {
        let spec = include_str!("../../crates/uor-matmul-kernels/src/spec.rs");
        let kernels_lib = include_str!("../../crates/uor-matmul-kernels/src/lib.rs");
        let mut violations = Vec::new();
        audit_kernel_capacity_sources(spec, kernels_lib, &mut violations);
        assert!(violations.is_empty(), "{violations:?}");

        let model = Model::load_from_repo_root().expect("the fixture model loads");
        let capacity = include_str!("../../crates/uor-matmul-kernels/src/generated_capacity.rs");
        let dispatch = include_str!("../../crates/uor-matmul-gemm/src/generated_atlas_dispatch.rs");
        audit_generated_capacity_artifacts(&model, capacity, dispatch, &mut violations);
        assert!(violations.is_empty(), "{violations:?}");
        let maximum = model.constants.kernel_capacity.max_tile_lanes;
        let drifted = capacity.replacen(
            &format!("MAX_TILE_LANES: usize = {maximum};"),
            &format!("MAX_TILE_LANES: usize = {};", maximum + 1),
            1,
        );
        audit_generated_capacity_artifacts(&model, &drifted, dispatch, &mut violations);
        assert!(
            violations
                .iter()
                .any(|violation| violation
                    .contains("not the exact model-generated capacity artifact")),
            "the planted model/generated capacity drift was not rejected: {violations:?}"
        );

        violations.clear();
        let source_maximum = model.constants.kernel_capacity.max_source_sites;
        let drifted_source = capacity.replacen(
            &format!("MAX_ATLAS_SOURCE_SITES: usize = {source_maximum};"),
            &format!("MAX_ATLAS_SOURCE_SITES: usize = {};", source_maximum + 1),
            1,
        );
        audit_generated_capacity_artifacts(&model, &drifted_source, dispatch, &mut violations);
        assert!(
            violations
                .iter()
                .any(|violation| violation
                    .contains("not the exact model-generated capacity artifact")),
            "the planted model/generated source-site drift was not rejected: {violations:?}"
        );

        let portable = include_str!("../../crates/uor-matmul-kernels/src/isa/portable.rs");
        let x86 = include_str!("../../crates/uor-matmul-kernels/src/isa/x86.rs");
        let arm = include_str!("../../crates/uor-matmul-kernels/src/isa/arm.rs");
        let wasm = include_str!("../../crates/uor-matmul-kernels/src/isa/wasm.rs");
        let declared = [portable, x86, arm, wasm]
            .into_iter()
            .flat_map(concrete_tile_fit_products)
            .max()
            .expect("the concrete ISA declarations contain tiles");
        assert_eq!(declared, maximum);
        let declared_sources = [portable, x86, arm, wasm]
            .into_iter()
            .flat_map(concrete_tile_fit_shapes)
            .map(|(rows, cols)| rows + cols)
            .max()
            .expect("the concrete ISA declarations contain source sites");
        assert_eq!(declared_sources, source_maximum);
        let shrunken_x86 = plant(
            x86,
            "crate::tile_fits!(8, 16);",
            "crate::tile_fits!(7, 16);",
        );
        let shrunken = [portable, shrunken_x86.as_str(), arm, wasm]
            .into_iter()
            .flat_map(concrete_tile_fit_products)
            .max()
            .expect("the planted ISA declarations still contain tiles");
        assert_ne!(
            shrunken, maximum,
            "the planted removal of the exact maximum must expose model headroom"
        );
        let shrunken_sources = [portable, shrunken_x86.as_str(), arm, wasm]
            .into_iter()
            .flat_map(concrete_tile_fit_shapes)
            .map(|(rows, cols)| rows + cols)
            .max()
            .expect("the planted ISA declarations still contain source sites");
        assert_ne!(
            shrunken_sources, source_maximum,
            "the planted removal of the exact source maximum must expose model headroom"
        );

        violations.clear();
        let duplicate_source = plant(spec, "crate::generated_capacity::MAX_TILE_LANES", "128");
        audit_kernel_capacity_sources(&duplicate_source, kernels_lib, &mut violations);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("not a direct alias")),
            "the planted duplicate capacity source was not rejected: {violations:?}"
        );

        violations.clear();
        let omitted = plant(spec, "            maxima!(all_i8),\n", "");
        audit_kernel_capacity_sources(&omitted, kernels_lib, &mut violations);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("covers")),
            "the planted omitted family was not rejected: {violations:?}"
        );

        violations.clear();
        let vacuous = plant(
            spec,
            ".map(|spec| (spec.mr * spec.nr, spec.mr + spec.nr))",
            ".map(|_| (MAX_TILE_LANES, crate::generated_capacity::MAX_ATLAS_SOURCE_SITES))",
        );
        audit_kernel_capacity_sources(&vacuous, kernels_lib, &mut violations);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("spec.mr*spec.nr")),
            "the planted self-fulfilling maximum was not rejected: {violations:?}"
        );

        violations.clear();
        let host_filtered = plant(
            spec,
            "core::iter::empty()$(.chain(core::iter::once($spec)))*",
            "collect![$($cond => $spec),*]",
        );
        audit_kernel_capacity_sources(&host_filtered, kernels_lib, &mut violations);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("declaration-walk witness")),
            "the planted host-filtered maximum was not rejected: {violations:?}"
        );
    }

    #[test]
    fn float_sweep_times_only_real_production_calls_cg_21() {
        let clean = include_str!("../../crates/uor-matmul-validate/tests/uor_float_sweep.rs");
        let mut violations = Vec::new();
        audit_compute_only_float_sweep_source(clean, &mut violations);
        assert!(violations.is_empty(), "{violations:?}");

        let aggregate_only = plant(
            clean,
            "CG21_SAMPLE phase=public width={width}",
            "CG21_AGGREGATE phase=public width={width}",
        );
        violations.clear();
        audit_compute_only_float_sweep_source(&aggregate_only, &mut violations);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("lacks machine-readable raw sample field")),
            "the planted aggregate-only public output was not rejected: {violations:?}"
        );

        let discarded_raw_durations = plant(
            clean,
            "        elapsed_ns: elapsed.map(|elapsed| elapsed.as_nanos()),",
            "        elapsed_ns: [0; SAMPLE_COUNT],",
        );
        violations.clear();
        audit_compute_only_float_sweep_source(&discarded_raw_durations, &mut violations);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("does not retain each raw batch duration")),
            "the planted discarded public durations were not rejected: {violations:?}"
        );

        let fixed_poison = plant(
            clean,
            "E::from_corpus_bits(expected.corpus_bits() ^ 1)",
            "E::from_corpus_bits(u64::MAX)",
        );
        violations.clear();
        audit_compute_only_float_sweep_source(&fixed_poison, &mut violations);
        assert!(
            violations.iter().any(|violation| violation
                .contains("float poison is not proved distinct from each expected code")),
            "the planted fixed public poison was not rejected: {violations:?}"
        );

        let truncated_comparator = plant(clean, "    assert_eq!(got.len(), want.len());\n", "");
        violations.clear();
        audit_compute_only_float_sweep_source(&truncated_comparator, &mut violations);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("comparator is not a complete byte comparison")),
            "the planted length-blind public comparator was not rejected: {violations:?}"
        );

        let self_comparator = plant(
            clean,
            "            want.corpus_bits(),\n            \"timed output differs at {at}\"",
            "            got.corpus_bits(),\n            \"timed output differs at {at}\"",
        );
        violations.clear();
        audit_compute_only_float_sweep_source(&self_comparator, &mut violations);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("comparator is not a complete byte comparison")),
            "the planted self-comparing public guard was not rejected: {violations:?}"
        );

        let view_inside = plant(
            clean,
            "        let start = Instant::now();\n        for _ in 0..repetitions {\n            uor_matmul::gemm_float_packed(",
            "        let start = Instant::now();\n        let _planted = MatView::row_major(a, case.m, case.k);\n        for _ in 0..repetitions {\n            uor_matmul::gemm_float_packed(",
        );
        violations.clear();
        audit_compute_only_float_sweep_source(&view_inside, &mut violations);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("timer is contaminated by `MatView::`")),
            "the planted in-timer view construction was not rejected: {violations:?}"
        );

        let faer_copy_inside = plant(
            clean,
            "    let start = Instant::now();\n    for _ in 0..repetitions {\n        E::faer_compute(state);",
            "    let start = Instant::now();\n    E::copy_faer_output(state, case, out);\n    for _ in 0..repetitions {\n        E::faer_compute(state);",
        );
        violations.clear();
        audit_compute_only_float_sweep_source(&faer_copy_inside, &mut violations);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("timer is contaminated by `copy_faer_output`")),
            "the planted in-timer faer adapter copy was not rejected: {violations:?}"
        );

        let faer_state_not_poisoned = plant(
            clean,
            "    E::poison_faer_output(state, case, expected);\n",
            "",
        );
        violations.clear();
        audit_compute_only_float_sweep_source(&faer_state_not_poisoned, &mut violations);
        assert!(
            violations.iter().any(|violation| violation
                .contains("does not prepare expected-derived poison `E::poison_faer_output(state,case,expected)` before timing")),
            "the planted stale faer C state was not rejected: {violations:?}"
        );

        let fixed_faer_poison = plant(
            clean,
            "state.c[(i, j)] = poison_from_expected(expected[i * case.n + j]);",
            "state.c[(i, j)] = E::from_corpus_bits(u64::MAX);",
        );
        violations.clear();
        audit_compute_only_float_sweep_source(&fixed_faer_poison, &mut violations);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("faer's real C state is not expected-derived")),
            "the planted fixed faer C poison was not rejected: {violations:?}"
        );

        let elided_api = plant(
            clean,
            "            uor_matmul::gemm_float_packed(",
            "            production_call_was_elided(",
        );
        violations.clear();
        audit_compute_only_float_sweep_source(&elided_api, &mut violations);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("real `uor_matmul::gemm_float_packed` call")),
            "the planted elided production call was not rejected: {violations:?}"
        );

        let unchecked = plant(
            clean,
            "    assert_bits(out, expected);\n    elapsed",
            "    elapsed",
        );
        violations.clear();
        audit_compute_only_float_sweep_source(&unchecked, &mut violations);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("lacks post-timer `assert_bits(out,expected)`")),
            "the planted missing byte check was not rejected: {violations:?}"
        );

        let candidate = include_str!("../../crates/uor-matmul-gemm/src/float.rs");
        violations.clear();
        audit_compute_only_candidate_sweep_source(candidate, &mut violations);
        assert!(violations.is_empty(), "{violations:?}");

        let unlabeled_width = plant(
            candidate,
            "CG21_SAMPLE phase=candidate width={}",
            "CG21_SAMPLE phase=candidate format={}",
        );
        violations.clear();
        audit_compute_only_candidate_sweep_source(&unlabeled_width, &mut violations);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("candidate measurement lacks")),
            "the planted width-blind candidate rows were not rejected: {violations:?}"
        );

        let discarded_candidate_durations = plant(
            candidate,
            "                        measured.elapsed_ns[round] = elapsed.as_nanos();",
            "                        measured.elapsed_ns[round] = 0;",
        );
        violations.clear();
        audit_compute_only_candidate_sweep_source(&discarded_candidate_durations, &mut violations);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("candidate measurement lacks")),
            "the planted discarded candidate durations were not rejected: {violations:?}"
        );

        let fixed_candidate_poison = plant(
            candidate,
            "E::from_candidate_bits(expected.symbol_bits() ^ 1)",
            "E::from_candidate_bits(u64::MAX)",
        );
        violations.clear();
        audit_compute_only_candidate_sweep_source(&fixed_candidate_poison, &mut violations);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("candidate poison is not distinct")),
            "the planted fixed candidate poison was not rejected: {violations:?}"
        );

        let truncated_candidate_comparator = plant(
            candidate,
            "            actual.len(),\n            expected.len(),",
            "            expected.len(),\n            expected.len(),",
        );
        violations.clear();
        audit_compute_only_candidate_sweep_source(&truncated_candidate_comparator, &mut violations);
        assert!(
            violations.iter().any(|violation| violation
                .contains("candidate comparator is not a complete byte comparison")),
            "the planted length-blind candidate comparator was not rejected: {violations:?}"
        );

        let self_candidate_comparator = plant(
            candidate,
            "                expected.symbol_bits(),\n                \"candidate changed an output byte at {at}\"",
            "                actual.symbol_bits(),\n                \"candidate changed an output byte at {at}\"",
        );
        violations.clear();
        audit_compute_only_candidate_sweep_source(&self_candidate_comparator, &mut violations);
        assert!(
            violations.iter().any(|violation| violation
                .contains("candidate comparator is not a complete byte comparison")),
            "the planted self-comparing candidate guard was not rejected: {violations:?}"
        );

        let candidate_view_inside = plant(
            candidate,
            "            let start = std::time::Instant::now();\n            for _ in 0..repetitions {",
            "            let start = std::time::Instant::now();\n            let _planted = MatView::row_major(a, shape.m, shape.k);\n            for _ in 0..repetitions {",
        );
        violations.clear();
        audit_compute_only_candidate_sweep_source(&candidate_view_inside, &mut violations);
        assert!(
            violations
                .iter()
                .any(|violation| violation
                    .contains("candidate timer is contaminated by `MatView::`")),
            "the planted candidate view construction was not rejected: {violations:?}"
        );

        let elided_candidate_api = plant(
            candidate,
            "                gemm_float_tiles_with_selector(",
            "                candidate_production_call_was_elided(",
        );
        violations.clear();
        audit_compute_only_candidate_sweep_source(&elided_candidate_api, &mut violations);
        assert!(
            violations.iter().any(|violation| violation.contains(
                "candidate timer excludes the real `gemm_float_tiles_with_selector` call"
            )),
            "the planted elided candidate production call was not rejected: {violations:?}"
        );

        let unchecked_candidate = plant(
            candidate,
            "        assert_candidate_bytes(&measured.output, expected);\n        elapsed",
            "        elapsed",
        );
        violations.clear();
        audit_compute_only_candidate_sweep_source(&unchecked_candidate, &mut violations);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("lacks its complete post-timer byte check")),
            "the planted missing candidate byte check was not rejected: {violations:?}"
        );
    }

    #[test]
    fn tabulated_float_q_carrier_is_in_place_and_total_cu_11() {
        let tabulated = include_str!("../../crates/uor-matmul-gemm/src/tabulated.rs");
        let table = include_str!("../../crates/uor-matmul-kernels/src/table.rs");
        let float = include_str!("../../crates/uor-matmul-gemm/src/float.rs");
        let mut violations = tabulated_float_fixture_violations(tabulated, table, float);
        assert!(violations.is_empty(), "{violations:?}");

        let early_column_pass = plant(
            tabulated,
            "    if !admits(\n",
            "    let _premature = distinct_columns::<E, Bd, C>(\n        triple.w.codes(),\n        triple.w.codes_per_row(),\n        shape.n,\n        index,\n    );\n    if !admits(\n",
        );
        violations = tabulated_float_fixture_violations(&early_column_pass, table, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("pays the column hash pass before admission")),
            "the planted declined-path column pass was not rejected: {violations:?}"
        );

        let duplicate_build = plant(
            tabulated,
            "set.insert(C::index_of(code))",
            "EntryInsert::New",
        );
        violations = tabulated_float_fixture_violations(&duplicate_build, table, float);
        assert!(
            violations.iter().any(|violation| violation
                .contains("exactly one presentation per distinct addressed index")),
            "the planted duplicate block-one construction was not rejected: {violations:?}"
        );

        let severed_addressed_scale = plant(
            tabulated,
            "addressed_lane_scale(&triple.a, &triple.w, addressed, ledger)",
            "E::lane_scale(&triple.a, &triple.w, ledger)",
        );
        violations = tabulated_float_fixture_violations(&severed_addressed_scale, table, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("distinct-address witness")),
            "the planted raw-stream scale walk was not rejected: {violations:?}"
        );

        let missing_boundary_census = plant(
            tabulated,
            "fn repeated_block_one_symbols_are_built_once_per_addressed_index_cg_16()",
            "fn repeated_block_one_symbols_were_rebuilt_per_stream_cell_cg_16()",
        );
        violations = tabulated_float_fixture_violations(&missing_boundary_census, table, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("below/equal/above-space adversarial census")),
            "the planted missing distinct-address boundary census was not rejected: {violations:?}"
        );

        let missing_shared_slot = plant(
            tabulated,
            "fn shared_slot_indices_are_deduplicated_without_collapsing_columns_cg_16()",
            "fn shared_slot_indices_were_rebuilt_without_column_collapse_cg_16()",
        );
        violations = tabulated_float_fixture_violations(&missing_shared_slot, table, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("independent of column collapse")),
            "the planted missing shared-slot census was not rejected: {violations:?}"
        );

        let wasted_singleton_set = plant(
            tabulated,
            "let need_entries = block == 1 && !C::SIGN_BIT_BOOK && space > 1;",
            "let need_entries = block == 1 && !C::SIGN_BIT_BOOK;",
        );
        violations = tabulated_float_fixture_violations(&wasted_singleton_set, table, float);
        assert!(
            violations.iter().any(|violation| {
                violation.contains("distinct-address witness")
                    || violation.contains("unused addressed-entry set")
            }),
            "the planted one-coordinate EntrySet regression was not rejected: {violations:?}"
        );

        let missing_no_clear = plant(
            tabulated,
            "fn non_pointwise_books_do_not_construct_an_unused_entry_set_cg_16()",
            "fn non_pointwise_books_recleared_the_unused_entry_set_cg_16()",
        );
        violations = tabulated_float_fixture_violations(&missing_no_clear, table, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("no-work/no-clear witness")),
            "the planted missing inapplicable-set witness was not rejected: {violations:?}"
        );

        let missing_overflow_reuse = plant(
            tabulated,
            "fn addressed_entry_set_overflow_collision_and_reuse_are_exact_cg_16()",
            "fn addressed_entry_set_overflow_left_stale_probes_cg_16()",
        );
        violations = tabulated_float_fixture_violations(&missing_overflow_reuse, table, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("overflow-clear, collision")),
            "the planted missing EntrySet reuse witness was not rejected: {violations:?}"
        );

        let missing_short_index = plant(
            tabulated,
            "fn short_index_offers_keep_duplicate_entry_work_truthful_cg_16()",
            "fn short_index_offers_hid_duplicate_entry_work_cg_16()",
        );
        violations = tabulated_float_fixture_violations(&missing_short_index, table, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("absent/short/full duplicate-work census")),
            "the planted missing short-index witness was not rejected: {violations:?}"
        );

        let missing_nonidentity = plant(
            tabulated,
            "fn addressed_codec_preserves_a_nonidentity_enumeration_cg_16()",
            "fn addressed_codec_assumed_an_identity_enumeration_cg_16()",
        );
        violations = tabulated_float_fixture_violations(&missing_nonidentity, table, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("nonidentity code_at/index_of witness")),
            "the planted missing nonidentity-codec witness was not rejected: {violations:?}"
        );

        let copied_projection = plant(tabulated, "project_f32_q(x, base)", "x");
        violations = tabulated_float_fixture_violations(&copied_projection, table, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("relabel each existing four-byte panel cell")),
            "the planted IEEE panel round trip was not rejected: {violations:?}"
        );

        let multiplied_build = plant(
            table,
            "let product = f32_q_product(decode_f32_q_factor(a), decode_f32_q_factor(w));",
            "let product = Scaled64((a as i64).wrapping_mul(w as i64));",
        );
        violations = tabulated_float_fixture_violations(tabulated, &multiplied_build, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("total Atlas token product")),
            "the planted traditional product build was not rejected: {violations:?}"
        );

        let severed_lookup = plant(
            table,
            "lookup(left_coordinate, right_coordinate)",
            "i32::from(left_coordinate).wrapping_mul(i32::from(right_coordinate))",
        );
        violations = tabulated_float_fixture_violations(tabulated, &severed_lookup, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("extent-fractal q contraction")),
            "the planted scalar octet multiplication was not rejected: {violations:?}"
        );

        let bit_regrade = plant(
            tabulated,
            "let source_q = unsigned / fraction_place;",
            "let source_q = unsigned >> (f32_q::SIGNIFICAND_BITS - 1);",
        );
        violations = tabulated_float_fixture_violations(&bit_regrade, table, float);
        assert!(
            violations.iter().any(|violation| violation
                .contains("q projection retains traditional bit-field arithmetic")),
            "the planted bitwise regrade was not rejected: {violations:?}"
        );

        let restored_common_grade = plant(
            float,
            "    if let Some((negative, magnitude, exponent)) = finite_parts(code) {",
            "    let _legacy = project_common_grade_f32;\n    if let Some((negative, magnitude, exponent)) = finite_parts(code) {",
        );
        violations = tabulated_float_fixture_violations(tabulated, table, &restored_common_grade);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("superseded common-grade carrier")),
            "the planted dead common-grade carrier was not rejected: {violations:?}"
        );

        let categorical_f64_table = plant(
            tabulated,
            "impl Tabulated for f64 {\n    type Lane = Wide<AccOf<f64>>;\n    type ModLane = Wide<AccOf<f64>>;\n    type StreamLane = Wide<AccOf<f64>>;\n    const LANE_IS_EXACT: bool = true;",
            "impl Tabulated for f64 {\n    type Lane = Wide<AccOf<f64>>;\n    type ModLane = Wide<AccOf<f64>>;\n    type StreamLane = Wide<AccOf<f64>>;\n    const LANE_IS_EXACT: bool = true;\n\n    fn probe_capacity<L: Lane<Self>>(_: u128) -> Option<usize> { Some(0) }",
        );
        violations = tabulated_float_fixture_violations(&categorical_f64_table, table, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("categorically overrides")),
            "the planted categorical f64 refusal was not rejected: {violations:?}"
        );

        let severed_pair = plant(
            tabulated,
            "let lane = <Scaled64 as Lane<f32>>::mac(Scaled64(0), projected_a, projected_w);",
            "let lane = <Scaled64 as Lane<f32>>::mac(Scaled64(0), projected_a, w);",
        );
        violations = tabulated_float_fixture_violations(&severed_pair, table, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("complete Laurent composition law")),
            "the planted severed producer/consumer pair was not rejected: {violations:?}"
        );

        let miscounted_q_presentation = plant(
            tabulated,
            "f32_q_build_presentations(3, 5, 7),\n            105,",
            "f32_q_build_presentations(3, 5, 7),\n            104,",
        );
        violations = tabulated_float_fixture_violations(&miscounted_q_presentation, table, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("one-presentation census law")),
            "the planted q presentation miscount was not rejected: {violations:?}"
        );

        let categorical_downstream = plant(
            tabulated,
            "fn downstream_block_two_f64_codec_is_not_categorically_declined_cd_20()",
            "fn downstream_block_two_f64_codec_was_categorically_declined_cd_20()",
        );
        violations = tabulated_float_fixture_violations(&categorical_downstream, table, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("downstream block-two codec witness")),
            "the planted missing downstream f64 witness was not rejected: {violations:?}"
        );

        let multiplied_spec = plant(
            tabulated,
            "spec.build_multiplies = false;",
            "spec.build_multiplies = true;",
        );
        violations = tabulated_float_fixture_violations(&multiplied_spec, table, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("multiply-free")),
            "the planted mispriced Atlas table build was not rejected: {violations:?}"
        );

        let severed_portable_graph = plant(
            tabulated,
            "let mut spec = portable_table::<f32, Scaled64>(rows, group);",
            "let mut spec = Self::table_spec_modular(backend, bound, rows, group, block);",
        );
        violations = tabulated_float_fixture_violations(&severed_portable_graph, table, float);
        assert!(
            violations
                .iter()
                .any(|violation| { violation.contains("one portable TableBuild/gather graph") }),
            "the planted alternate f32 table graph was not rejected: {violations:?}"
        );
    }

    #[test]
    fn tabulated_radix_hash_and_portable_q_graph_are_falsifiable_cu_11() {
        let tabulated = include_str!("../../crates/uor-matmul-gemm/src/tabulated.rs");
        let table = include_str!("../../crates/uor-matmul-kernels/src/table.rs");
        let float = include_str!("../../crates/uor-matmul-gemm/src/float.rs");
        let model = include_str!("../../crates/uor-matmul-model/src/derive.rs");
        let mut violations = tabulated_float_fixture_violations(tabulated, table, float);
        audit_column_hash_model_source(model, &mut violations);
        assert!(violations.is_empty(), "{violations:?}");

        let reduced_length = plant(
            tabulated,
            "let mut hash = run.len() as u128;",
            "let mut hash = (run.len() % modulus) as u128;",
        );
        violations = tabulated_float_fixture_violations(&reduced_length, table, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("float-reachable radix address graph")),
            "the planted redundant initial modulus was not rejected: {violations:?}"
        );

        let stale_model = plant(
            model,
            "    let mut bound = coordinate;",
            "    let mut bound = power_of_two(address_bits - 1).unwrap() - 1;",
        );
        violations.clear();
        audit_column_hash_model_source(&stale_model, &mut violations);
        assert!(
            violations
                .iter()
                .any(|violation| { violation.contains("full unreduced usize length coordinate") }),
            "the planted stale reduced-length model bound was not rejected: {violations:?}"
        );

        let literal_prefix = plant(tabulated, "crate::float::COLUMN_HASH_PREFIX", "16");
        violations = tabulated_float_fixture_violations(&literal_prefix, table, float);
        assert!(
            violations.iter().any(|violation| {
                violation.contains("float-reachable radix address graph")
                    || violation.contains("legacy token `min(16)`")
            }),
            "the planted non-model hash prefix was not rejected: {violations:?}"
        );

        let even_hash = plant(
            tabulated,
            "hash = doubled + hash + C::index_of(code) as u128;",
            "hash = doubled + C::index_of(code) as u128;",
        );
        violations = tabulated_float_fixture_violations(&even_hash, table, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("float-reachable radix address graph")),
            "the planted even radix was not rejected: {violations:?}"
        );

        let multiplied_hash = plant(
            tabulated,
            "hash = doubled + hash + C::index_of(code) as u128;",
            "hash = hash * 3 + C::index_of(code) as u128;",
        );
        violations = tabulated_float_fixture_violations(&multiplied_hash, table, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("legacy token `hash*`")),
            "the planted hash multiply was not rejected: {violations:?}"
        );

        let masked_probe = plant(
            tabulated,
            "                    probe += 1;\n                    if probe == table {\n                        probe = 0;\n                    }",
            "                    probe += 1;\n                    probe &= table - 1;",
        );
        violations = tabulated_float_fixture_violations(&masked_probe, table, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("float-reachable radix address graph")),
            "the planted masked column probe was not rejected: {violations:?}"
        );

        let masked_entry = plant(
            table,
            "    let within = offset % slab;",
            "    let within = offset & (slab - 1);",
        );
        violations = tabulated_float_fixture_violations(tabulated, &masked_entry, float);
        assert!(
            violations.iter().any(|violation| {
                violation.contains("float-reachable radix address graph")
                    || violation.contains("legacy token `&(slab`")
            }),
            "the planted masked portable gather address was not rejected: {violations:?}"
        );

        let scanned_grade = plant(
            table,
            "    let mut grade = 0u32;\n    while rows > 1 {\n        rows /= 2;\n        grade += 1;\n    }\n    grade",
            "    rows.trailing_zeros()",
        );
        violations = tabulated_float_fixture_violations(tabulated, &scanned_grade, float);
        assert!(
            violations.iter().any(|violation| {
                violation.contains("float-reachable radix address graph")
                    || violation.contains("legacy token `trailing_zeros`")
            }),
            "the planted portable row scan was not rejected: {violations:?}"
        );

        let bitwise_build = plant(
            table,
            "    match rows {\n        1 => build_run::<1, E, L>(block, book, acts, out),",
            "    let _legacy = rows >> 1;\n    match rows {\n        1 => build_run::<1, E, L>(block, book, acts, out),",
        );
        violations = tabulated_float_fixture_violations(tabulated, &bitwise_build, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("legacy token `>>`")),
            "the planted bitwise TableBuild body was not rejected: {violations:?}"
        );

        let missing_address_parity = plant(
            table,
            "fn portable_radix_addresses_match_retained_bit_oracle_cu_11()",
            "fn portable_radix_addresses_lacked_a_retained_oracle_cu_11()",
        );
        violations = tabulated_float_fixture_violations(tabulated, &missing_address_parity, float);
        assert!(
            violations.iter().any(|violation| {
                violation.contains("independent retained-bit boundary differential")
            }),
            "the planted missing portable address parity was not rejected: {violations:?}"
        );

        let unpoisoned_clock = plant(tabulated, "                index.fill(poison);", "");
        violations = tabulated_float_fixture_violations(&unpoisoned_clock, table, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("retained column-collapse clock")),
            "the planted unpoisoned hash clock was not rejected: {violations:?}"
        );

        let unchecked_clock = plant(
            tabulated,
            "                assert_eq!(&index[..n], expected_map.as_slice());",
            "",
        );
        violations = tabulated_float_fixture_violations(&unchecked_clock, table, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("retained column-collapse clock")),
            "the planted unchecked hash clock was not rejected: {violations:?}"
        );

        let unbatched_clock = plant(tabulated, "for _ in 0..batch {", "for _ in 0..1 {");
        violations = tabulated_float_fixture_violations(&unbatched_clock, table, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("retained column-collapse clock")),
            "the planted unbatched hash clock was not rejected: {violations:?}"
        );

        let uninterleaved_clock = plant(
            tabulated,
            "let radix_first = (sample + chunk).is_multiple_of(2);",
            "let radix_first = sample.is_multiple_of(2);",
        );
        violations = tabulated_float_fixture_violations(&uninterleaved_clock, table, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("retained column-collapse clock")),
            "the planted unchunked hash clock was not rejected: {violations:?}"
        );

        let guarded_timer = plant(
            tabulated,
            "                let elapsed = start.elapsed();\n                assert_eq!(observed, expected_distinct);\n                assert_eq!(&index[..n], expected_map.as_slice());",
            "                assert_eq!(observed, expected_distinct);\n                let elapsed = start.elapsed();\n                assert_eq!(&index[..n], expected_map.as_slice());",
        );
        violations = tabulated_float_fixture_violations(&guarded_timer, table, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("retained column-collapse clock")),
            "the planted in-timer result guard was not rejected: {violations:?}"
        );

        let aggregate_verdict = plant(
            tabulated,
            "                upper_95 <= 1.0,",
            "                ratio <= 1.0,",
        );
        violations = tabulated_float_fixture_violations(&aggregate_verdict, table, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("retained column-collapse clock")),
            "the planted non-interval hash verdict was not rejected: {violations:?}"
        );

        let self_compared_clock = plant(
            tabulated,
            "            let legacy = legacy_distinct_columns::<f32, Whole<f32>, A251>(",
            "            let legacy = distinct_columns::<f32, Whole<f32>, A251>(",
        );
        violations = tabulated_float_fixture_violations(&self_compared_clock, table, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("retained column-collapse clock")),
            "the planted self-compared hash clock was not rejected: {violations:?}"
        );

        let mutated_legacy = plant(
            tabulated,
            "        const HASH_PREFIX: usize = 16;",
            "        const HASH_PREFIX: usize = 15;",
        );
        violations = tabulated_float_fixture_violations(&mutated_legacy, table, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("immutable pre-refactor spelling")),
            "the planted mutation of the retained comparator was not rejected: {violations:?}"
        );

        let zero_horner = plant(
            table,
            "    let mut product = *coordinates\n        .next()\n        .expect(\"two nonempty factors have a nonempty product word\");",
            "    let _highest = coordinates.next();\n    let mut product = 0;",
        );
        violations = tabulated_float_fixture_violations(tabulated, &zero_horner, float);
        assert!(
            violations.iter().any(|violation| {
                violation.contains("zero-initialized or fixed-extent Horner prefix")
            }),
            "the planted zero-initialized full-grade Horner walk was not rejected: {violations:?}"
        );
    }

    #[test]
    fn atlas_projectors_reject_bit_scans_shifts_and_low_byte_casts_cu_11() {
        let clean = clean_panel_fixture();
        let mut violations = panel_fixture_violations(&clean);
        assert!(violations.is_empty(), "{violations:?}");

        let shifted_odd = plant(
            &clean,
            "        let quotient = magnitude / 2;",
            "        let quotient = magnitude >> 1;",
        );
        violations = panel_fixture_violations(&shifted_odd);
        assert!(
            violations.iter().any(|violation| {
                violation.contains("odd-section helper does not use quotient/add divisibility")
            }),
            "the planted valuation shift was not rejected: {violations:?}"
        );

        let low_byte = plant(
            &clean,
            "        let residue = value.rem_euclid(radix);",
            "        let residue = i128::from(value as u8);",
        );
        violations = panel_fixture_violations(&low_byte);
        assert!(
            violations.iter().any(|violation| {
                violation
                    .contains("word projector does not use centered Euclidean radix extraction")
            }),
            "the planted low-byte cast was not rejected: {violations:?}"
        );

        let bit_split = plant(
            &clean,
            "        magnitude / ATLAS_SIGNED_PLACE_RADIX,",
            "        magnitude >> (i128::BITS - 1),",
        );
        violations = panel_fixture_violations(&bit_split);
        assert!(
            violations.iter().any(|violation| {
                violation.contains(
                    "signed-place fracture does not use one Euclidean radix quotient and remainder",
                )
            }),
            "the planted signed-place bit split was not rejected: {violations:?}"
        );

        let wrong_split_radix = plant(
            &clean,
            "const ATLAS_SIGNED_PLACE_RADIX: u128 = i128::MIN.unsigned_abs();",
            "const ATLAS_SIGNED_PLACE_RADIX: u128 = i128::MAX as u128;",
        );
        violations = panel_fixture_violations(&wrong_split_radix);
        assert!(
            violations.iter().any(|violation| {
                violation.contains(
                    "signed-place fracture does not use one Euclidean radix quotient and remainder",
                )
            }),
            "the planted noncanonical signed-place radix was not rejected: {violations:?}"
        );
    }

    #[test]
    fn total_f32_q_contract_is_structurally_falsifiable_cd_32() {
        let tabulated = include_str!("../../crates/uor-matmul-gemm/src/tabulated.rs");
        let table = include_str!("../../crates/uor-matmul-kernels/src/table.rs");
        let float = include_str!("../../crates/uor-matmul-gemm/src/float.rs");
        let mut violations = tabulated_float_fixture_violations(tabulated, table, float);
        assert!(violations.is_empty(), "{violations:?}");

        let missing_capacity = plant(
            tabulated,
            "fn total_f32_lane_scale_uses_the_exact_q_capacity_cd_32()",
            "fn legacy_f32_lane_scale_used_the_power_of_two_capacity_cd_32()",
        );
        violations = tabulated_float_fixture_violations(&missing_capacity, table, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("production-side exact-Q")),
            "the planted missing exact-Q differential was not rejected: {violations:?}"
        );

        let runtime_model = plant(
            tabulated,
            "fn total_f32_lane_scale_uses_the_exact_q_capacity_cd_32() {",
            "fn total_f32_lane_scale_uses_the_exact_q_capacity_cd_32() {\n        let _ = uor_matmul_model::Model::load_from_repo_root();",
        );
        violations = tabulated_float_fixture_violations(&runtime_model, table, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("reopens the repository model")),
            "the planted target-time model reload was not rejected: {violations:?}"
        );

        let missing_union = plant(
            tabulated,
            "            0, 1, 2, // +Inf | -Inf | NaN",
            "            0, 1, 3, // planted missing three-flag union",
        );
        violations = tabulated_float_fixture_violations(&missing_union, table, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("seven-union")),
            "the planted missing Complete union was not rejected: {violations:?}"
        );

        let missing_fracture = plant(
            tabulated,
            "fn f32_q_lane_scalar_fractures_a_wider_codec_block_cd_32()",
            "fn f32_q_lane_declined_a_wider_codec_block_cd_32()",
        );
        violations = tabulated_float_fixture_violations(&missing_fracture, table, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("scalar-fracture")),
            "the planted missing fracture differential was not rejected: {violations:?}"
        );

        let inexact_fracture_census = plant(
            tabulated,
            "            forced.adds, 24,",
            "            forced.adds, 23,",
        );
        violations = tabulated_float_fixture_violations(&inexact_fracture_census, table, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("scalar-fracture")),
            "the planted inexact fracture Census was not rejected: {violations:?}"
        );

        let missing_empty = plant(
            tabulated,
            "fn empty_f32_q_reduction_has_zero_work_cd_32()",
            "fn empty_f32_q_reduction_was_not_observed_cd_32()",
        );
        violations = tabulated_float_fixture_violations(&missing_empty, table, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("zero-depth totality/Census")),
            "the planted missing k-zero differential was not rejected: {violations:?}"
        );

        let missing_nonpower = plant(
            tabulated,
            "fn parametric_nonpower_q_blocks_preserve_bytes_strides_offers_and_census_cd_32()",
            "fn power_two_q_blocks_only_cd_32()",
        );
        violations = tabulated_float_fixture_violations(&missing_nonpower, table, float);
        assert!(
            violations.iter().any(|violation| {
                violation.contains("non-power block/space, tail, stride, and offer")
            }),
            "the planted missing parametric non-power differential was not rejected: {violations:?}"
        );

        let runtime_kernel_model = plant(
            table,
            "fn generated_model() -> GeneratedModel {",
            "fn generated_model() -> GeneratedModel {\n        let _ = uor_matmul_model::Model::load_from_repo_root();",
        );
        violations = tabulated_float_fixture_violations(tabulated, &runtime_kernel_model, float);
        assert!(
            violations
                .iter()
                .any(|violation| { violation.contains("generated target-independent q model") }),
            "the planted kernel target-time model reload was not rejected: {violations:?}"
        );

        let missing_special_order = plant(
            tabulated,
            "fn f32_q_special_atoms_are_immediate_source_order_singletons_cd_32()",
            "fn f32_q_special_atoms_were_reordered_cd_32()",
        );
        violations = tabulated_float_fixture_violations(&missing_special_order, table, float);
        assert!(
            violations
                .iter()
                .any(|violation| { violation.contains("source-ordered finite/special/finite") }),
            "the planted missing special-source-order differential was not rejected: {violations:?}"
        );

        for (name, replacement, label) in [
            (
                "q_precision_fractal_matches_wide_product_and_exact_work_cd_32",
                "q_precision_fractal_skipped_exact_work_cd_32",
                "extent-fractal lookup/Horner work",
            ),
            (
                "mixed_nonfinite_and_finite_words_scalar_fracture_cd_32",
                "mixed_nonfinite_words_skipped_scalar_fracture_cd_32",
                "mixed finite/special split and special union",
            ),
            (
                "scaled64_zero_is_raw_identity_for_every_token_class_cd_32",
                "scaled64_zero_identity_was_sampled_cd_32",
                "raw public-lane zero identity",
            ),
        ] {
            let missing = plant(
                table,
                &format!("fn {name}()"),
                &format!("fn {replacement}()"),
            );
            violations = tabulated_float_fixture_violations(tabulated, &missing, float);
            assert!(
                violations.iter().any(|violation| violation.contains(label)),
                "the planted missing kernel {label} differential was not rejected: {violations:?}"
            );
        }

        let self_compared_route = plant(
            tabulated,
            "            finite_signature, special_signature,\n            \"selection is value-blind\"",
            "            finite_signature, finite_signature,\n            \"planted self-comparison\"",
        );
        violations = tabulated_float_fixture_violations(&self_compared_route, table, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("value-blind route")),
            "the planted selector self-comparison was not rejected: {violations:?}"
        );

        let vacuous_route = plant(
            tabulated,
            "            finite_signature.0,\n            \"the shared automatic route is nonvacuously tabulated\"",
            "            true,\n            \"planted vacuous route\"",
        );
        violations = tabulated_float_fixture_violations(&vacuous_route, table, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("value-blind route")),
            "the planted vacuous selector route was not rejected: {violations:?}"
        );

        let severed_dynamic_seam = plant(
            tabulated,
            "    let local_envelopes = data_dependent_lane && observed_run.is_some_and(|run| run < shape.k);",
            "    let local_envelopes = false;",
        );
        violations = tabulated_float_fixture_violations(&severed_dynamic_seam, table, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("dynamic q lane seam")),
            "the planted severed dynamic q seam was not rejected: {violations:?}"
        );

        let nonleast_certificate = plant(
            tabulated,
            "        envelope = envelope.max(regrade_envelope(local, call_scale, cap));",
            "        envelope = cap;",
        );
        violations = tabulated_float_fixture_violations(&nonleast_certificate, table, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("least per-slot L-infinity certificate")),
            "the planted nonleast scalar certificate was not rejected: {violations:?}"
        );

        let unregraded_certificate = plant(
            tabulated,
            "    for _ in 0..distance {\n        if bound > cap || bound > cap - bound {\n            return singleton;\n        }\n        bound += bound;\n    }\n    bound",
            "    for _ in 0..distance {\n        if bound > cap || bound > cap - bound {\n            return singleton;\n        }\n        bound += 1;\n    }\n    bound",
        );
        violations = tabulated_float_fixture_violations(&unregraded_certificate, table, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("common-base certificate regrading")),
            "the planted non-radix certificate regrade was not rejected: {violations:?}"
        );

        let nonmaximal_prefix = plant(
            tabulated,
            "            if pending && (singleton || bound > cap - height) {",
            "            if pending {",
        );
        violations = tabulated_float_fixture_violations(&nonmaximal_prefix, table, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("source-ordered maximal-prefix scheduler")),
            "the planted eager fracture scheduler was not rejected: {violations:?}"
        );

        let bounded_fracture = plant(
            tabulated,
            "    let block = C::MAX_BLOCK;\n    let source = p * block + coordinate;",
            "    let _planted_bound: Option<[u128; 1]> = None;\n    let block = C::MAX_BLOCK;\n    let source = p * block + coordinate;",
        );
        violations = tabulated_float_fixture_violations(&bounded_fracture, table, float);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("arbitrary bound storage/limit")),
            "the planted fixed fracture bound was not rejected: {violations:?}"
        );
    }

    #[test]
    fn atlas_selector_minimizes_actual_global_work_cu_11() {
        let clean = clean_panel_fixture();
        let mut violations = selector_fixture_violations(&clean);
        assert!(violations.is_empty(), "{violations:?}");

        let missing_count_factor = plant(&clean, "    AtlasCountFactor::PhysicalTile,\n]", "]");
        violations = selector_fixture_violations(&missing_count_factor);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("AtlasCountFactor::PhysicalTile")),
            "the planted undersized exact-count derivation was not rejected: {violations:?}"
        );

        let non_radix_cardinality = plant(
            &clean,
            "u64::MAX as u128 + (u64::MAX != u64::MIN) as u128",
            "u64::MAX as u128",
        );
        violations = selector_fixture_violations(&non_radix_cardinality);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("ATLAS_COUNT_RADIX")),
            "the planted non-cardinality radix was not rejected: {violations:?}"
        );

        let severed_comparison = plant(
            &clean,
            "atlas_executed_work::<A>(spec, shape, pa_codes, pb_codes)",
            "AtlasWork::ZERO",
        );
        violations = selector_fixture_violations(&severed_comparison);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("minimum-work witness")),
            "the planted severed work comparison was not rejected: {violations:?}"
        );

        let family_first = plant(
            &clean,
            ".chain(uor_matmul_kernels::cached::available_i8_narrow())",
            ".chain(core::iter::empty())",
        );
        violations = selector_fixture_violations(&family_first);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("omits `available_i8_narrow`")),
            "the planted missing global family was not rejected: {violations:?}"
        );

        let missing_projection_cost = plant(
            &clean,
            "        projections: projection_sites,",
            "        projections: AtlasCount::ZERO,",
        );
        violations = selector_fixture_violations(&missing_projection_cost);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("structural-cost witness")),
            "the planted projection-cost omission was not rejected: {violations:?}"
        );

        let missing_live_cells = plant(&clean, "(live_cells as u128)", "0u128");
        violations = selector_fixture_violations(&missing_live_cells);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("structural-cost witness")),
            "the planted live-storage omission was not rejected: {violations:?}"
        );

        let missing_product_initializations = plant(
            &clean,
            "        product_initializations,",
            "        product_initializations: AtlasCount::ZERO,",
        );
        violations = selector_fixture_violations(&missing_product_initializations);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("structural-cost witness")),
            "the planted live carrier-initialization omission was not rejected: {violations:?}"
        );

        let priced_empty_depth = plant(
            &clean,
            "if shape.m == 0 || shape.k == 0 || shape.n == 0",
            "if shape.m == 0 || shape.n == 0",
        );
        violations = selector_fixture_violations(&priced_empty_depth);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("shape.k==0")),
            "the planted nonzero k-zero work census was not rejected: {violations:?}"
        );
    }

    #[test]
    fn shipped_and_model_storage_work_twins_are_compared_cg_22() {
        let clean = clean_panel_fixture();
        let mut violations = model_differential_fixture_violations(&clean);
        assert!(violations.is_empty(), "{violations:?}");

        let missing_workspace = plant(
            &clean,
            "                bytes,\n                ATLAS_TILE_WORK_BYTES,",
            "                bytes,\n                0,",
        );
        violations = model_differential_fixture_violations(&missing_workspace);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("shipped/model comparison")),
            "the planted omitted workspace charge was not rejected: {violations:?}"
        );

        let self_comparison = plant(
            &clean,
            "            uor_matmul_model::derive::atlas_executed_work(",
            "            atlas_executed_work::<A>(",
        );
        violations = model_differential_fixture_violations(&self_comparison);
        assert!(
            violations.iter().any(|violation| {
                violation.contains("shipped/model comparison")
                    || violation.contains("one shipped census")
            }),
            "the planted model self-comparison was not rejected: {violations:?}"
        );

        let missing_f64 = plant(
            &clean,
            "                        assert_model_work::<AccOf<f64>>(spec, shape, 0, 0);",
            "                        assert_model_work::<AccOf<f32>>(spec, shape, 0, 0);",
        );
        violations = model_differential_fixture_violations(&missing_f64);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("AccOf<f64>")),
            "the planted missing f64 differential was not rejected: {violations:?}"
        );

        let missing_empty_depth_twin = plant(
            &clean,
            "        assert_model_work::<AccOf<f64>>(spec, empty_depth, 7, 11);",
            "",
        );
        violations = model_differential_fixture_violations(&missing_empty_depth_twin);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("empty_depth")),
            "the planted missing k-zero model twin was not rejected: {violations:?}"
        );
    }

    #[test]
    fn atlas_product_resolves_after_all_lookup_diagonals_cu_11() {
        let clean = clean_panel_fixture();
        let mut violations = panel_fixture_violations(&clean);
        assert!(violations.is_empty(), "{violations:?}");

        let hidden_live_cell_invariant = plant(&clean, "cells.for_each_live", "cells.for_each");
        violations = panel_fixture_violations(&hidden_live_cell_invariant);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("live-only traversal invariant")),
            "the planted non-live cell traversal was not rejected: {violations:?}"
        );

        let padded_product_clear = plant(
            &clean,
            "        for i in 0..rows {\n            let first = i * spec.nr;\n            workspace.products[first..first + cols].fill(AtlasProduct::ZERO);\n        }\n        ledger.product_initialized(rows * cols);",
            "        workspace.products[..spec.mr * spec.nr].fill(AtlasProduct::ZERO);\n        ledger.product_initialized(spec.mr * spec.nr);",
        );
        violations = panel_fixture_violations(&padded_product_clear);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("direct contraction lacks lazy-cell witness")),
            "the planted padded carrier clear was not rejected: {violations:?}"
        );

        let full_word_clear = plant(
            &clean,
            "    coordinates[extent..].fill(0);",
            "    coordinates.fill(0);",
        );
        violations = panel_fixture_violations(&full_word_clear);
        assert!(
            violations.iter().any(|violation| {
                violation.contains("rewrite their live prefix")
                    || violation.contains("retired suffix")
            }),
            "the planted full coordinate-word clear was not rejected: {violations:?}"
        );

        let tile_early_place = plant(
            &clean,
            "workspace.products[physical_lane].add_diagonal(lane, diagonal);",
            "workspace.products[physical_lane].add_diagonal(lane, diagonal); \
             place(accumulator, i128::from(lane), 0);",
        );
        violations = panel_fixture_violations(&tile_early_place);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("places a diagonal before")),
            "the planted tiled diagonal placement was not rejected: {violations:?}"
        );

        let dot_early_place = plant(
            &clean,
            "product.add_diagonal(lane[0], diagonal);",
            "product.add_diagonal(lane[0], diagonal); \
             acc.place_at_wide(i128::from(lane[0]), 0);",
        );
        violations = panel_fixture_violations(&dot_early_place);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("places a diagonal before")),
            "the planted one-dot diagonal placement was not rejected: {violations:?}"
        );

        let missing_radix_fracture = plant(
            &clean,
            "            let (low, high) = atlas_split_signed_place(magnitude);",
            "            let low = magnitude;\n            let high = 0;",
        );
        violations = panel_fixture_violations(&missing_radix_fracture);
        assert!(
            violations.iter().any(|violation| {
                violation.contains("signed-place radix fracture and its two source-ordered grades")
            }),
            "the planted missing signed-place fracture was not rejected: {violations:?}"
        );

        let rescaled_radix_diagonal = plant(
            &clean,
            "product.add_diagonal(lane[0], diagonal);",
            "product.add_diagonal(lane[0], diagonal * ATLAS_DIGIT_BITS as usize);",
        );
        violations = panel_fixture_violations(&rescaled_radix_diagonal);
        assert!(
            violations.iter().any(|violation| {
                violation.contains("does not consolidate lookup diagonals into one `AtlasProduct`")
            }),
            "the planted rescaling of a radix diagonal was not rejected: {violations:?}"
        );
    }

    #[test]
    fn unequal_float_panels_are_total_zero_extended_objects_cu_11() {
        let clean = clean_panel_fixture();
        let mut violations = panel_fixture_violations(&clean);
        assert!(violations.is_empty(), "{violations:?}");

        let truncated = plant(&clean, "pa.len().max(pb.len())", "pa.len().min(pb.len())");
        violations = panel_fixture_violations(&truncated);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("truncate to their common prefix")),
            "the planted unequal-panel truncation was not rejected: {violations:?}"
        );

        let unchecked = plant(&clean, "pa.get(p).copied().unwrap_or(ZERO_CODE)", "pa[p]");
        violations = panel_fixture_violations(&unchecked);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("total zero-extension witness")),
            "the planted unchecked panel read was not rejected: {violations:?}"
        );

        let skipped_boundary = plant(&clean, "acc.accumulate_one(a_code, b_code);", "continue;");
        violations = panel_fixture_violations(&skipped_boundary);
        assert!(
            violations
                .iter()
                .any(|violation| violation.contains("implicit-zero IEEE boundary")),
            "the planted skipped non-finite join was not rejected: {violations:?}"
        );
    }

    /// The exact claim is over the live shipped roots.  It remains red while a
    /// root still reaches any superseded path; a fixture-only test would make
    /// the scenario green without governing the implementation.
    #[test]
    fn post_decode_pre_encode_call_graph_is_uor_only_cu_11() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("xtask is directly below the repository root");
        audit_uor_float(root).unwrap();
    }
}
