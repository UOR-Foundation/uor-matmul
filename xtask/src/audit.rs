//! The grep-shaped gates.
//!
//! These are crude on purpose. A gate that needs interpretation is a gate that
//! gets argued with; these ones read the source, find a token, and fail. Each
//! carries the rule it enforces in its failure message, because the point of a
//! red gate is to name the promise that was broken.

use std::path::{Path, PathBuf};

use uor_matmul_model::Model;

use crate::Fail;

/// The crates that ship. The rules below apply to these and not to the
/// dev-and-CI-only crates, which may use `std`, `alloc`, and floats freely.
pub(crate) const SHIPPED: &[&str] = &[
    "uor-matmul",
    "uor-matmul-core",
    "uor-matmul-codec",
    "uor-matmul-kernels",
    "uor-matmul-gemm",
];

/// Files where a numeral may appear without deriving from the model:
/// the generated output itself, and the model-facing gates.
const NUMERAL_ALLOWLIST: &[&str] = &["generated.rs"];

struct Source {
    path: PathBuf,
    rel: String,
    text: String,
}

fn shipped_sources(root: &Path) -> Result<Vec<Source>, Fail> {
    let mut out = Vec::new();
    for name in SHIPPED {
        let dir = root.join("crates").join(name).join("src");
        if !dir.exists() {
            continue;
        }
        collect(&dir, root, &mut out)?;
    }
    Ok(out)
}

fn collect(dir: &Path, root: &Path, out: &mut Vec<Source>) -> Result<(), Fail> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            collect(&path, root, out)?;
        } else if path.extension().is_some_and(|e| e == "rs") {
            let text = std::fs::read_to_string(&path)?;
            let rel = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .display()
                .to_string();
            out.push(Source { path, rel, text });
        }
    }
    Ok(())
}

/// Lines of `text` outside comments and outside `#[cfg(test)]` modules.
///
/// A rule about what the library *does* must not be tripped by a doc comment
/// explaining what it does not do, nor by a test that deliberately constructs
/// the forbidden thing to prove it is caught.
fn effective_lines(text: &str) -> Vec<(usize, &str)> {
    let mut out = Vec::new();
    let mut in_test = false;
    let mut test_depth = 0i32;
    let mut depth = 0i32;
    for (i, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.starts_with("//") || line.starts_with("#!") {
            // A `#![deny(...)]` or a comment states policy; it never is the
            // behaviour the gate is looking for.
            continue;
        }
        let opens = raw.matches('{').count() as i32;
        let closes = raw.matches('}').count() as i32;
        if line.starts_with("#[cfg(test)]") {
            in_test = true;
            test_depth = depth;
        }
        if !in_test {
            let code = line.split("//").next().unwrap_or("");
            if !code.trim().is_empty() {
                out.push((i + 1, code));
            }
        }
        depth += opens - closes;
        if in_test && depth <= test_depth && closes > 0 {
            in_test = false;
        }
    }
    out
}

/// R1: no magic numeral. Every constant derives by `const fn` from the declared
/// tuple, or is recorded in the model as discovered by measurement; `133144`
/// appears only inside a `const _: () = assert!(...)` pin.
pub fn check_constants(root: &Path) -> Result<(), Fail> {
    let model = Model::load(&root.join("model"))?;
    let sources = shipped_sources(root)?;

    // Every narrow threshold's numeral, and every instantiation bound, may
    // appear only in the generated file or inside a const-assert pin.
    let mut pinned: Vec<String> = Vec::new();
    for t in &model.constants.narrow.threshold {
        pinned.push(t.k_max.to_string());
        pinned.push(t.per_step.to_string());
    }
    let atlas = &model.constants.atlas;
    let refinement = uor_matmul_model::derive::atlas_refinement_leaves(atlas.context);
    let alphabet =
        uor_matmul_model::derive::atlas_alphabet(atlas.scope, atlas.modality, atlas.context);
    let atlas_pinned = [
        atlas.scope.to_string(),
        atlas.modality.to_string(),
        atlas.context.to_string(),
        uor_matmul_model::derive::atlas_carrier_dim(atlas.modality, atlas.context).to_string(),
        uor_matmul_model::derive::atlas_class_count(atlas.scope, atlas.modality, atlas.context)
            .to_string(),
        uor_matmul_model::derive::atlas_page_sites(atlas.scope, atlas.context).to_string(),
        uor_matmul_model::derive::atlas_refinement_bits(atlas.context).to_string(),
        refinement.coefficient().to_string(),
        refinement.power().to_string(),
        alphabet.coefficient().to_string(),
        alphabet.power().to_string(),
    ];

    let mut violations = Vec::new();
    for src in &sources {
        if NUMERAL_ALLOWLIST.iter().any(|a| src.path.ends_with(a)) {
            continue;
        }
        let mut in_atlas_const = false;
        for (line_no, line) in effective_lines(&src.text) {
            if line.contains("const _: () = assert!") {
                continue;
            }
            // The canonical Atlas tuple is necessarily made from small
            // numerals. Scanning every `3`, `4`, or `8` in a numerical crate
            // would be noise, but restating one beside an `ATLAS_` declaration
            // is precisely the R10 defect the private generated views prevent.
            if line.contains("const ATLAS_") {
                in_atlas_const = true;
            }
            if in_atlas_const {
                let right_hand_side = line.split_once('=').map_or(line, |(_, rhs)| rhs);
                for numeral in &atlas_pinned {
                    if contains_literal_numeral(right_hand_side, numeral) {
                        violations.push(format!(
                            "{}:{line_no}: the Atlas numeral {numeral} is model-owned\n    {}",
                            src.rel,
                            line.trim()
                        ));
                    }
                }
                if line.contains(';') {
                    in_atlas_const = false;
                }
            }
            for numeral in &pinned {
                // Numerals below four digits are ubiquitous and carry no claim
                // (`2`, `64`, `128` are widths, not model constants).
                if numeral.len() < 4 {
                    continue;
                }
                if contains_numeral(line, numeral) {
                    violations.push(format!(
                        "{}:{line_no}: the numeral {numeral} is a model constant\n    {}",
                        src.rel,
                        line.trim()
                    ));
                }
            }
        }
    }

    if !violations.is_empty() {
        return Err(format!(
            "R1: no magic numeral. Every constant derives by const fn from the declared \
             tuple, or is recorded in the model as discovered by measurement --- a \
             derivation story for a measured constant is fiction. A model numeral appears \
             only in the generated file or inside a `const _: () = assert!(...)` pin.\n\n{}",
            violations.join("\n")
        )
        .into());
    }
    println!("check-constants: no model numeral is restated in the shipped crates (R1)");
    Ok(())
}

/// Does `line` contain `numeral` as a whole number rather than as a digit run
/// inside a longer one?
fn contains_numeral(line: &str, numeral: &str) -> bool {
    let bytes = line.as_bytes();
    let mut from = 0;
    while let Some(pos) = line[from..].find(numeral) {
        let start = from + pos;
        let end = start + numeral.len();
        let before_ok = start == 0 || !bytes[start - 1].is_ascii_digit();
        let after_ok = end >= bytes.len() || !bytes[end].is_ascii_digit();
        if before_ok && after_ok {
            return true;
        }
        from = start + 1;
    }
    false
}

/// As [`contains_numeral`], but reject digits that are part of an identifier
/// such as the `128` in `i128`.
fn contains_literal_numeral(line: &str, numeral: &str) -> bool {
    let bytes = line.as_bytes();
    let mut from = 0;
    while let Some(pos) = line[from..].find(numeral) {
        let start = from + pos;
        let end = start + numeral.len();
        let before_ok =
            start == 0 || (!bytes[start - 1].is_ascii_alphanumeric() && bytes[start - 1] != b'_');
        let after_ok = end >= bytes.len() || !bytes[end].is_ascii_digit();
        if before_ok && after_ok {
            return true;
        }
        from = start + 1;
    }
    false
}

/// R8: no arbitrary limitation. Every bound is a property of the caller's
/// chosen instantiation, never of the library.
///
/// Concretely: no shipped crate may return an error the model does not
/// sanction. §2's "there is no `CN-*` class" is checked here --- the absence of
/// negative testing is only honest if there is nothing to test negatively.
pub fn audit_limits(root: &Path) -> Result<(), Fail> {
    let sources = shipped_sources(root)?;
    // The only error type the library may name, plus the declaration check at
    // the alphabet boundary, which is not the operation failing (§5.2).
    let sanctioned = ["NotAProduct", "ObservedBound", "KappaError"];

    let mut violations = Vec::new();
    for src in &sources {
        for (line_no, line) in effective_lines(&src.text) {
            let Some(pos) = line.find("Result<") else {
                continue;
            };
            let tail = &line[pos..];
            if sanctioned.iter().any(|s| tail.contains(s)) {
                continue;
            }
            violations.push(format!("{}:{line_no}:{}", src.rel, line.trim()));
        }
    }
    if !violations.is_empty() {
        return Err(format!(
            "R8: every bound in the library is derived from declared parameters and is a \
             property of the caller's chosen instantiation. C6: the only reportable \
             condition is that the requested object does not exist, reported at view \
             construction. A `Result` over anything else is a limitation the model does \
             not sanction.\n\n{}",
            violations.join("\n")
        )
        .into());
    }
    println!("audit-limits: no shipped crate returns an unsanctioned error (R8, C6)");
    Ok(())
}

/// R13, `CU-01`, `CU-03`: one method. No float addition, no saturating or
/// rounding instruction in an accumulation, no lesser method held in reserve.
pub fn audit_purity(root: &Path) -> Result<(), Fail> {
    let sources = shipped_sources(root)?;

    // R2: no float *arithmetic* anywhere. `f32` and `f64` are permitted as
    // element types being decoded and as the target of the encode step; what is
    // forbidden is adding, subtracting, multiplying, or fusing two floats.
    //
    // This half reads the source and `CU-01` reads the emitted instructions, and
    // both are needed, because neither covers the other. `CU-01` sees only what
    // was codegen'd: measured, an uncalled `pub fn` is not in the rlib at all, so
    // a float add sitting in one is invisible to it here --- and codegen'd in a
    // *downstream* build, where this repository's gates do not run. This half is
    // what closes that, so it tracks which values are floats rather than matching
    // a token list: `let mut t = a; t = t + b;` on two `f32` parameters passed
    // both halves until it did.
    //
    // The one exemption is a function named with the `f64_exact` marker, which
    // must hold the `F64Exact` witness: float arithmetic whose operands and
    // results are integers below 2^53 *is* integer arithmetic, so the property
    // --- exactness --- survives it. The marker without the witness is a
    // violation here; the witness without the bound does not compile; the
    // opcode without the marker is a violation in `audit_disassembly`.
    let mut violations = Vec::new();
    for src in &sources {
        let mut floats = FloatScope::default();
        // Set inside a function named with the `f64_exact` marker: the one
        // place float arithmetic is permitted, because such a function must
        // hold the `F64Exact` witness (checked just below), whose constructor
        // does not compile unless the declared bound and depth keep every
        // value inside binary64's exact-integer range. The exemption is a
        // type the compiler checked, not a comment this gate trusts (`CU-01`
        // as restated). R3 and R13 still apply inside it: the witness
        // licenses exact float arithmetic, not saturation and not a second
        // method.
        let mut in_f64_exact = false;
        let raw_lines: Vec<&str> = src.text.lines().collect();
        for (line_no, line) in effective_lines(&src.text) {
            if is_fn_start(line.trim_start()) {
                in_f64_exact = line.contains(F64_EXACT_MARKER);
                if in_f64_exact && !function_span_mentions(&raw_lines, line_no - 1, "F64Exact") {
                    violations.push(format!(
                        "CU-01: {}:{line_no}: the `f64_exact` marker licenses float arithmetic \
                         only where the `F64Exact` witness is held; this function names neither \
                         the type nor its constructor\n    {}",
                        src.rel,
                        line.trim()
                    ));
                }
            }
            if !in_f64_exact {
                for tok in FLOAT_ARITHMETIC {
                    if line.contains(tok) {
                        violations.push(format!(
                            "R2: {}:{line_no}: `{tok}` is float arithmetic\n    {}",
                            src.rel,
                            line.trim()
                        ));
                    }
                }
                if let Some(op) = float_literal_arithmetic(line) {
                    violations.push(format!(
                        "R2: {}:{line_no}: `{op}` applied to a float literal\n    {}",
                        src.rel,
                        line.trim()
                    ));
                }
                // A binding whose type is a float, and an operator next to it.
                floats.observe(line);
                if let Some(name) = floats.arithmetic_on_a_float(line) {
                    violations.push(format!(
                        "R2: {}:{line_no}: arithmetic on `{name}`, which is a float\n    {}",
                        src.rel,
                        line.trim()
                    ));
                }
            }
            // R3: no saturating or rounding instruction in an accumulation.
            // The single encode step is the only place information is
            // discarded. A saturating operation anywhere else must carry an
            // `R3-ok:` note saying why it is not an accumulation --- cursor
            // arithmetic on an index cannot change an output value, but a grep
            // cannot tell an index from an accumulator, so the author says so.
            for tok in ["saturating_add", "saturating_mul", "saturating_sub"] {
                if !line.contains(tok) {
                    continue;
                }
                // The note is a trailing comment, which `effective_lines` strips,
                // so the raw line is what carries it.
                let raw = src
                    .text
                    .lines()
                    .nth(line_no.saturating_sub(1))
                    .unwrap_or("");
                // The note, and nothing else. This read
                // `src.rel.contains("gemm") || raw.contains("R3-ok:")`, and the
                // first half exempts every path under `crates/uor-matmul-gemm/`
                // --- the whole driver, which is where the accumulation lives and
                // therefore the one crate the rule is about. R3 was unenforced
                // exactly where it matters, and three sites in `float.rs` had no
                // note. The author says why a saturating operation is not an
                // accumulation; a directory name cannot say it for them.
                if !raw.contains("R3-ok:") {
                    violations.push(format!(
                        "R3: {}:{line_no}: `{tok}` with no `R3-ok:` note\n    {}",
                        src.rel,
                        line.trim()
                    ));
                }
            }
            // R13: the vocabulary of a second method.
            for tok in ["fallback", "fast_path", "approximate", "good_enough"] {
                if contains_word(line, tok) {
                    violations.push(format!(
                        "R13: {}:{line_no}: `{tok}` names a second method\n    {}",
                        src.rel,
                        line.trim()
                    ));
                }
            }
        }
    }

    if !violations.is_empty() {
        return Err(format!(
            "R13: one answer, many factorizations, no fallback. Every path in the library \
             is decode-then-accumulate-exactly; backends and traversals are factorizations \
             of one identity, byte-identical under the CD-* gates, not a quality hierarchy.\n\n{}",
            violations.join("\n")
        )
        .into());
    }
    println!("audit-purity: one method; no float arithmetic, no second-best (R2, R3, R13)");
    Ok(())
}

/// A line that opens a function definition.
///
/// Shared by `audit_purity`'s two function-scoped trackers --- the float scope
/// and the `f64_exact` exemption --- because the two must agree on where a
/// function begins: if they drifted, an exemption could outlive the function
/// that earned it.
fn is_fn_start(t: &str) -> bool {
    t.starts_with("fn ")
        || t.starts_with("pub fn ")
        || t.starts_with("unsafe fn ")
        || t.starts_with("pub unsafe fn ")
        || t.starts_with("pub(crate) fn ")
        || t.starts_with("const fn ")
        || t.starts_with("pub const fn ")
}

/// Does the function whose header is at raw line `start` (0-based) mention
/// `needle` anywhere from its signature through its body?
///
/// Crude in the house style: brace counting, confused by a brace inside a
/// string literal and by nothing else these crates write. The span ends where
/// the body's brace depth returns to zero.
fn function_span_mentions(lines: &[&str], start: usize, needle: &str) -> bool {
    let mut depth = 0i32;
    let mut opened = false;
    for line in &lines[start..] {
        if line.contains(needle) {
            return true;
        }
        depth += line.matches('{').count() as i32;
        depth -= line.matches('}').count() as i32;
        opened |= line.contains('{');
        if opened && depth <= 0 {
            break;
        }
    }
    false
}

/// Method names that can only be float arithmetic.
const FLOAT_ARITHMETIC: &[&str] = &[
    "mul_add",
    "powi",
    "powf",
    ".sqrt()",
    "f32::EPSILON",
    "f64::EPSILON",
    "to_degrees",
    "to_radians",
];

/// Which names in the function being read are floats.
///
/// The R2 half of `audit-purity` used to match a token list and a float
/// *literal*, so `let mut t = a; t = t + b;` on two `f32` parameters --- the most
/// direct violation the rule has --- passed it. A grep cannot type-check, but it
/// can follow the declarations that are written down, and in these crates every
/// float that reaches an operator is written down: as a parameter, as a `let`
/// with a type, as a float literal, or as an `as f32`.
///
/// Scoped per function, because that is where a name means one thing. Anything
/// this misses `CU-01` still catches once it is codegen'd; anything `CU-01`
/// misses --- an uncalled `pub fn`, which is codegen'd only in a downstream build
/// --- this catches here.
#[derive(Default)]
struct FloatScope {
    names: Vec<String>,
}

impl FloatScope {
    /// Read one line for float declarations, and reset at a function boundary.
    fn observe(&mut self, line: &str) {
        let t = line.trim();
        if is_fn_start(t) {
            self.names.clear();
        }
        // Any binding whose declared type *mentions* a float: `x: f32`,
        // `v: &[f32]`, `p: *const f64`. Over-approximating on purpose --- a
        // `&[f32]` cannot itself be added, but an element of one can, and a name
        // that carries floats is the thing worth watching. Several may appear on
        // a line, as a signature does.
        for (at, _) in line.match_indices(':') {
            if line[at..].starts_with("::") || (at > 0 && line.as_bytes()[at - 1] == b':') {
                continue;
            }
            let ty_end = line[at + 1..]
                .find([',', ')', '=', '{', ';'])
                .map_or(line.len(), |e| at + 1 + e);
            let ty = &line[at + 1..ty_end];
            if ty.contains("f32") || ty.contains("f64") {
                if let Some(name) = ident_before(&line[..at]) {
                    self.push(name);
                }
            }
        }
        // `let name = <float literal>`, `let name = ... as f32`, anywhere on the
        // line --- a one-line function body puts the `let` in the middle of it.
        for (at, _) in line.match_indices("let ") {
            let after = &line[at + 4..];
            let after = after.strip_prefix("mut ").unwrap_or(after);
            let Some(eq) = after.find('=') else { continue };
            let binding = after[..eq].trim();
            let name = binding.split(':').next().unwrap_or("").trim().to_string();
            let stop = after[eq + 1..]
                .find(';')
                .map_or(after.len(), |e| eq + 1 + e);
            let rhs = &after[eq + 1..stop];
            let declared = binding.split_once(':').map(|(_, ty)| ty.trim());
            let is_float = declared.map_or_else(
                || {
                    rhs.contains("as f32")
                        || rhs.contains("as f64")
                        || rhs.contains("f32::")
                        || rhs.contains("f64::")
                        || has_float_literal(rhs)
                        || self.names.iter().any(|n| rhs.contains(n.as_str()))
                },
                |ty| ty.contains("f32") || ty.contains("f64"),
            );
            if is_float && is_ident(&name) {
                self.push(name);
            }
        }
        // `for x in v`, where `v` already carries floats: the element does too.
        for (at, _) in line.match_indices("for ") {
            let after = &line[at + 4..];
            let Some(in_at) = after.find(" in ") else {
                continue;
            };
            let name = after[..in_at].trim().trim_start_matches("&").to_string();
            let src = &after[in_at + 4..];
            if self.names.iter().any(|n| src.contains(n.as_str())) {
                self.push(name);
            }
        }
    }

    fn push(&mut self, name: String) {
        if is_ident(&name) && !self.names.contains(&name) {
            self.names.push(name);
        }
    }

    /// An arithmetic operator with one of this scope's floats beside it.
    fn arithmetic_on_a_float(&self, line: &str) -> Option<String> {
        for name in &self.names {
            for (i, _) in line.match_indices(name.as_str()) {
                // Whole word only: `t` must not match inside `tile`.
                let before = line[..i].chars().next_back();
                let after = line[i + name.len()..].chars().next();
                if before.is_some_and(is_ident_char) || after.is_some_and(is_ident_char) {
                    continue;
                }
                if operator_beside(line, i, i + name.len()) {
                    return Some(name.clone());
                }
            }
        }
        None
    }
}

fn is_ident_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

fn is_ident(s: &str) -> bool {
    !s.is_empty()
        && s.chars().all(is_ident_char)
        && !s.chars().next().is_some_and(|c| c.is_ascii_digit())
}

/// The identifier immediately before a `:` type annotation.
fn ident_before(head: &str) -> Option<String> {
    let name: String = head
        .chars()
        .rev()
        .take_while(|&c| is_ident_char(c))
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    is_ident(&name).then_some(name)
}

/// A float literal: digits with a point, or an `f32`/`f64` suffix.
fn has_float_literal(s: &str) -> bool {
    let b: Vec<char> = s.chars().collect();
    for i in 0..b.len() {
        if !b[i].is_ascii_digit() {
            continue;
        }
        if i > 0 && is_ident_char(b[i - 1]) {
            continue;
        }
        let mut j = i;
        while j < b.len() && (b[j].is_ascii_digit() || b[j] == '_') {
            j += 1;
        }
        if j < b.len() && b[j] == '.' && b.get(j + 1).is_some_and(|c| c.is_ascii_digit()) {
            return true;
        }
        if s[..]
            .get(j..j + 3)
            .is_some_and(|t| t == "f32" || t == "f64")
        {
            return true;
        }
    }
    false
}

/// Is `+`, `-`, `*` or `/` immediately beside the span `[lo, hi)`?
///
/// `->` is a return type, `*const`/`*mut` is a pointer type, and `//` was already
/// stripped, so none of those counts.
fn operator_beside(line: &str, lo: usize, hi: usize) -> bool {
    let b = line.as_bytes();
    let mut i = lo;
    while i > 0 && b[i - 1] == b' ' {
        i -= 1;
    }
    if i > 0 {
        let c = b[i - 1];
        let prev = if i > 1 { b[i - 2] } else { b' ' };
        if (c == b'+' || c == b'*' || c == b'/') && prev != b'*' && prev != b'/' {
            return true;
        }
        if c == b'-' && prev != b'-' && prev != b'<' {
            return true;
        }
    }
    let mut j = hi;
    while j < b.len() && b[j] == b' ' {
        j += 1;
    }
    if j < b.len() {
        let c = b[j];
        let next = b.get(j + 1).copied().unwrap_or(b' ');
        if (c == b'+' || c == b'/') && next != b'/' {
            return true;
        }
        if c == b'*' && next != b'c' && next != b'm' {
            return true;
        }
        if c == b'-' && next != b'>' && next != b'-' {
            return true;
        }
    }
    false
}

/// An arithmetic operator adjacent to a float literal.
///
/// Catches `x + 1.0` and `2.0 * y` without needing to know any types. A float
/// literal in the shipped crates is only ever a constant --- `0.0` for a zero
/// element --- so an operator next to one is arithmetic on a float.
fn float_literal_arithmetic(line: &str) -> Option<&'static str> {
    let bytes = line.as_bytes();
    for (i, w) in bytes.windows(2).enumerate() {
        // A float literal is a digit, a dot, then a digit.
        if w[0] != b'.' || !w[1].is_ascii_digit() {
            continue;
        }
        if i == 0 || !bytes[i - 1].is_ascii_digit() {
            continue;
        }
        // Scan outwards for an arithmetic operator on either side.
        // Skip back over the literal's integer part, or `before` ends with a
        // digit and no operator is ever seen.
        let mut b = i;
        while b > 0 && (bytes[b - 1].is_ascii_digit() || bytes[b - 1] == b'_') {
            b -= 1;
        }
        let before = line[..b].trim_end();
        let after = {
            let mut j = i + 1;
            while j < bytes.len() && (bytes[j].is_ascii_digit() || bytes[j] == b'_') {
                j += 1;
            }
            line[j.min(line.len())..].trim_start()
        };
        for (op, name) in [('+', "+"), ('-', "-"), ('*', "*"), ('/', "/")] {
            if before.ends_with(op) || after.starts_with(op) {
                return Some(name);
            }
        }
    }
    None
}

fn contains_word(line: &str, word: &str) -> bool {
    let bytes = line.as_bytes();
    let mut from = 0;
    while let Some(pos) = line[from..].find(word) {
        let start = from + pos;
        let end = start + word.len();
        let ident = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
        let before_ok = start == 0 || !ident(bytes[start - 1]);
        let after_ok = end >= bytes.len() || !ident(bytes[end]);
        if before_ok && after_ok {
            return true;
        }
        from = start + 1;
    }
    false
}

/// R15: nothing is deferred: every declared capability is complete and present
/// in the one release.
///
/// The vocabulary this reads as a deferral is `model/ledger.toml`'s
/// `deferral_markers`, not a table written here. That is not tidiness: this
/// gate walks `crates/`, `xtask/` and `fuzz/`, so it reads *its own source*, and
/// a marker table spelled as Rust string literals would make it fail on itself.
/// The alternative was exempting this file, which is a hole in exactly the place
/// nobody looks (R10).
///
/// The steady-state ledger has no suspension. If a bounded suspension is ever
/// recorded, its admitted paths are read from the ledger rather than hard-coded
/// here: a gate with a hard-coded hole is a hole nobody reviews. The model also
/// refuses any admitted path under `crates/` or `xtask/`, so shipped code and
/// gates remain covered unconditionally.
pub fn audit_deferral(root: &Path) -> Result<(), Fail> {
    let ledger = Model::load(&root.join("model"))?.ledger;
    let markers: Vec<String> = ledger.deferral_markers.clone();
    if markers.is_empty() {
        return Err(
            "R15's marker vocabulary is empty, so the gate would pass vacuously; \
                    `model/ledger.toml` owns it"
                .into(),
        );
    }
    let admitted: Vec<String> = ledger
        .suspension
        .iter()
        .filter(|s| s.rule == "R15")
        .flat_map(|s| s.admits.clone())
        .collect();
    let mut violations = Vec::new();

    let mut files: Vec<PathBuf> = Vec::new();
    // R15 binds the whole workspace, not the shipped half of it. The gather
    // used to walk `SHIPPED` only, so the gate that forbids deferral read
    // neither `xtask/` --- its own source --- nor the model, validation,
    // conformance and executor crates, nor `fuzz/`, which is outside the
    // workspace entirely. A rule enforced everywhere except where the enforcer
    // lives is a rule with a hole in exactly the place nobody looks.
    for dir in [
        "crates", "xtask", "fuzz", "docs", "model", "features", "oracles", ".github",
    ] {
        let p = root.join(dir);
        if p.exists() {
            gather_all(&p, &mut files)?;
        }
    }
    // Every top-level document, found rather than listed. A hard-coded list is
    // how `docs/conceptual-model/README.md` came to be edited by a change that
    // the gate then did not read: the list was the coverage, and nobody
    // remembered to grow it. Reading the directory means a new document is
    // covered on the day it is written.
    for entry in std::fs::read_dir(root)? {
        let p = entry?.path();
        if p.is_file() && p.extension().is_some_and(|e| e == "md" || e == "toml") {
            files.push(p);
        }
    }
    // The recipes are prose to a reader and a program to `just`, and a deferral
    // parked in a comment there is as deferred as one in a crate.
    //
    // `MEASUREMENT-LOG.md` is reached by the directory read above and is held to
    // the rule like every other document.
    let recipes = root.join("Justfile");
    if recipes.exists() {
        files.push(recipes);
    }

    for path in files {
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let rel = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .display()
            .to_string();
        if admitted.contains(&rel) {
            continue;
        }
        for (i, line) in text.lines().enumerate() {
            for marker in &markers {
                if !line.contains(marker.as_str()) {
                    continue;
                }
                // The vocabulary declares itself, and a list cannot list itself
                // without naming its own members. The exemption is exactly one
                // shape --- a line that is nothing but a quoted marker and a
                // comma, in the file the vocabulary lives in --- so a real
                // deferral written anywhere else in that file, in a comment or
                // in any other field, is still reported.
                if rel == "model/ledger.toml" && line.trim() == format!("\"{marker}\",") {
                    continue;
                }
                // A backticked marker is a *mention*, not a use: the
                // documentation has to be able to name what the gate catches
                // without tripping it. Anything outside code spans is real.
                if outside_code_spans(line, marker) {
                    violations.push(format!("{rel}:{}: {}", i + 1, line.trim()));
                }
            }
        }
    }

    if !violations.is_empty() {
        return Err(format!(
            "R15: nothing is deferred. Every capability is complete and ships in the one release. \
             The vocabulary is `model/ledger.toml`'s `deferral_markers`.\n\n{}",
            violations.join("\n")
        )
        .into());
    }
    if admitted.is_empty() {
        println!("audit-deferral: nothing is deferred (R15)");
    } else {
        println!(
            "audit-deferral: nothing is deferred outside the suspension recorded in \
             model/ledger.toml (R15 suspended; admits: {})",
            admitted.join(", ")
        );
    }
    Ok(())
}

/// Does `marker` occur in `line` outside every backtick-delimited span?
fn outside_code_spans(line: &str, marker: &str) -> bool {
    let mut rest = line;
    let mut at = 0usize;
    while let Some(pos) = rest.find(marker) {
        let absolute = at + pos;
        // An odd number of backticks before this occurrence means it is inside
        // a span.
        if line[..absolute].matches('`').count().is_multiple_of(2) {
            return true;
        }
        at = absolute + marker.len();
        rest = &line[at..];
    }
    false
}

fn gather_all(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), Fail> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            if path.file_name().is_some_and(|n| n == "target") {
                continue;
            }
            gather_all(&path, out)?;
        } else if path.extension().is_some_and(|e| {
            // Source, prose, model data, the workflows, the Gherkin suites, and
            // the oracle generators. A deferral parked in a CI step or in a
            // scenario is a deferral; extension-filtering to `.rs` is how a
            // marker in `ci.yml` used to pass a gate whose own message said
            // "anywhere in the workspace".
            e == "rs"
                || e == "md"
                || e == "toml"
                || e == "yml"
                || e == "yaml"
                || e == "feature"
                || e == "py"
                || e == "json"
        }) {
            out.push(path);
        }
    }
    Ok(())
}

/// `CU-01`, R2: no float add, subtract, multiply, or FMA opcode appears in any
/// shipped kernel's disassembly --- except inside a function that holds the
/// [`F64Exact`] witness, where the operand bound and the depth prove the opcode
/// computes the same integer the exact accumulator does.
///
/// The definitive form of R2. The source grep in [`audit_purity`] cannot see
/// what the optimizer emitted; this reads the assembly the compiler actually
/// produced and looks for the instructions themselves.
///
/// A kernel that computed a float sum would fail here even if it did so through
/// a helper the grep never saw, and that is the point: the rule is about the
/// machine code, so the check should be too.
///
/// What it deliberately does *not* catch is a float add the optimizer removed,
/// because such an add is not in the shipped kernel. That is the gate reporting
/// the binary rather than the source, which is the only thing worth reporting.
///
/// The exemption is not a comment the gate trusts and it is not a review
/// judgement. A function whose emitted name carries `f64_exact` must mention
/// the `F64Exact<Bd, DEPTH>` witness --- the source half of this gate requires
/// it --- and the witness's only constructor does not compile unless
/// `DEPTH * Bd::VALUE^2 <= 2^53`, the range in which binary64 arithmetic on
/// integers *is* integer arithmetic. The disassembly gate checks the name, the
/// source gate checks the type, and the compiler checks the bound; no hop is a
/// judgement call.
pub fn audit_disassembly(root: &Path) -> Result<(), Fail> {
    let out_dir = root.join("target").join("cu01-asm");
    // Removed, not reused. `--emit asm` writes a `.s` only when the crate is
    // actually compiled, so a warm directory makes `cargo rustc` answer
    // "Finished" and leave last run's assembly --- or none at all. Observed: the
    // gate reported three objects on a run where only two `.s` files existed, and
    // the missing one was `uor-matmul-gemm`, the crate the float path lives in.
    // A gate that silently reads fewer objects than it names is the vacuity this
    // file exists to prevent.
    let _ = std::fs::remove_dir_all(&out_dir);
    std::fs::create_dir_all(&out_dir)?;

    let mut checked = 0usize;
    // Per crate, so "how many objects" stops being a number nobody reconciles.
    let mut per_crate: Vec<(&str, usize)> = Vec::new();
    let mut violations = Vec::new();
    // `CU-06`: the tabulated column loop, found by symbol and read instruction by
    // instruction. A source grep cannot see what the optimizer emitted, and the
    // whole claim of the tabulated traversal is about what it emitted.
    let mut column_loops = 0usize;
    let mut tropical_bodies = 0usize;

    for krate in ["uor-matmul-kernels", "uor-matmul-core", "uor-matmul-gemm"] {
        let before = checked;
        let status = std::process::Command::new(env!("CARGO"))
            .current_dir(root)
            .args(["rustc", "--release", "-p", krate, "--lib", "--target-dir"])
            .arg(&out_dir)
            .args(["--", "--emit", "asm", "-C", "debuginfo=0"])
            .status()?;
        if !status.success() {
            return Err(format!("could not build {krate} for disassembly").into());
        }

        for path in asm_files(&out_dir)? {
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            let name = path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            if !name.starts_with(&krate.replace('-', "_")) {
                continue;
            }
            checked += 1;
            // The function each instruction belongs to, so that an opcode can
            // be attributed: an exemption applies to a function holding the
            // witness, never to a crate or a file. Local labels (`.L*` on ELF,
            // `LBB*`/`Ltmp*` on Mach-O) are not functions and do not move the
            // attribution.
            let mut enclosing = String::new();
            for (i, line) in text.lines().enumerate() {
                let trimmed = line.trim();
                if is_function_label(trimmed) {
                    enclosing = trimmed.trim_end_matches(':').to_string();
                }
                if let Some(op) = float_opcode(trimmed) {
                    if enclosing.contains(F64_EXACT_MARKER) {
                        continue;
                    }
                    violations.push(format!("{name}:{}: `{op}`\n    {}", i + 1, trimmed));
                }
            }
            for (label, body) in function_bodies(&text, TABULATED_COLUMN_LOOP) {
                column_loops += 1;
                for (at, &(i, line)) in body.iter().enumerate() {
                    if let Some(op) = integer_multiply_opcode(line) {
                        if op == "msub" && at > 0 && is_aarch64_remainder_pair(body[at - 1].1, line)
                        {
                            continue;
                        }
                        violations.push(format!(
                            "CU-06: {name}:{i}: `{op}` in `{label}`\n    {line}"
                        ));
                    }
                }
            }
            for (label, body) in function_bodies(&text, TROPICAL_SEQUENCE) {
                tropical_bodies += 1;
                for (i, line) in body {
                    if let Some(op) = integer_multiply_opcode(line) {
                        violations.push(format!(
                            "CU-10: {name}:{i}: `{op}` in `{label}`\n    {line}"
                        ));
                    }
                }
            }
        }
        per_crate.push((krate, checked - before));
    }

    if checked == 0 {
        return Err("CU-01 disassembled nothing; the gate would pass vacuously".into());
    }
    if let Some((krate, _)) = per_crate.iter().find(|&&(_, n)| n == 0) {
        return Err(format!(
            "CU-01 disassembled no object for `{krate}`, so every claim about that crate's \
             emitted instructions passed without being read. The gate names three crates and \
             must read all three: `--emit asm` writes a `.s` only when the crate is actually \
             compiled, so this is what a stale `target/cu01-asm` looks like."
        )
        .into());
    }
    if column_loops == 0 {
        return Err(format!(
            "CU-06 found no symbol containing `{TABULATED_COLUMN_LOOP}`, so the claim that \
             the tabulated column loop issues no multiply would pass vacuously. The loop is \
             generic; `uor_matmul_gemm::tabulated` names one concrete instantiation of it \
             precisely so that this gate has instructions to read."
        )
        .into());
    }
    if tropical_bodies == 0 {
        return Err(format!(
            "CU-10 found no symbol containing `{TROPICAL_SEQUENCE}`, so the claim that no \
             tropical sequence multiplies would pass vacuously. Every entry point in the \
             family carries that fragment by convention, and a rename that dropped it is \
             exactly what this catches."
        )
        .into());
    }
    if !violations.is_empty() {
        return Err(format!(
            "R2, CU-01: the library never adds two floats. A float arithmetic opcode in a \
             shipped kernel means an accumulation is not exact, which breaks every claim in \
             §3. The one exemption is a function holding the `F64Exact` witness, where the \
             declared bound and depth keep every value below 2^53 and the opcode computes \
             the same integer --- any other float opcode is a defect.\n\
             CU-06: the tabulated column loop never multiplies. A multiply there means a \
             product the table was supposed to have already computed, which is the whole \
             content of the traversal.\n\
             CU-10: a tropical sequence never multiplies. `⊗` in the `(max, +)` semiring is \
             an addition, so a multiply there is a different operation and not a faster \
             spelling of this one.\n\n{}",
            violations.join("\n")
        )
        .into());
    }
    println!(
        "audit-disassembly: no float arithmetic opcode in {checked} shipped object(s) (CU-01), \
         no multiply in {column_loops} tabulated column loop(s) (CU-06), and none in \
         {tropical_bodies} tropical sequence(s) (CU-10)"
    );
    Ok(())
}

/// The symbol fragment that names the tabulated column loop's accumulation.
///
/// The loop itself is a [`uor_matmul_kernels::TableSpec`] sequence and generic,
/// so it emits no code until something instantiates it;
/// `uor_matmul_gemm::tabulated::gather_reference_*` names the reference
/// instantiation precisely so this gate has instructions to read.
const TABULATED_COLUMN_LOOP: &str = "gather_reference";

/// The symbol fragment that names a packed tropical `(max, +)` sequence.
///
/// `⊗` in that semiring is an addition, so a multiply in one of these bodies is
/// a different operation and not a faster spelling of this one. The convention
/// is the scope, exactly as [`TABULATED_COLUMN_LOOP`] is:
/// `uor_matmul_kernels::tropical`'s header names every entry point that carries
/// it, so adding a sequence enrols it and dropping the fragment in a rename is
/// visible in the source (`CU-10`).
const TROPICAL_SEQUENCE: &str = "trop_i16";

/// The name fragment marking a function that holds the `F64Exact` witness ---
/// the one place a float arithmetic opcode may appear in shipped code (`CU-01`
/// as restated: the property is exactness, and the witness is its proof). The
/// source half of the gate requires every function so named to mention
/// `F64Exact`, whose only constructor refuses to compile outside the exact
/// range, so the marker cannot be worn without the proof.
const F64_EXACT_MARKER: &str = "f64_exact";

/// Is this assembly line a *function* label, as opposed to a local one?
///
/// ELF locals start with `.L` and Mach-O locals are `LBB*` / `Ltmp*`, so the
/// filters differ; a mangled function symbol on either starts with `_` or a
/// letter. Attributing an instruction to the local label above it would lose
/// the function it belongs to, which is the same failure class as the Mach-O
/// framing defect `function_bodies` records below.
pub(crate) fn is_function_label(trimmed: &str) -> bool {
    trimmed.ends_with(':')
        && !trimmed.starts_with('.')
        && !trimmed.starts_with("LBB")
        && !trimmed.starts_with("Ltmp")
}

/// Every emitted function whose symbol contains `fragment`, as `(label, body)`,
/// where the body is the `(line number, instruction)` pairs between the label and
/// the function's end marker.
///
/// Read from the assembler's own framing rather than by a window around a grep
/// hit: a multiply three lines after a label may belong to the next function, and
/// a gate that could not tell would be reporting on the wrong code.
fn function_bodies<'t>(text: &'t str, fragment: &str) -> Vec<(String, Vec<(usize, &'t str)>)> {
    let mut out = Vec::new();
    let lines: Vec<&str> = text.lines().collect();
    for (start, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        // A label, not a directive mentioning the symbol.
        let Some(label) = trimmed.strip_suffix(':') else {
            continue;
        };
        if label.starts_with('.') || !label.contains(fragment) {
            continue;
        }
        let mut body = Vec::new();
        for (i, inner) in lines.iter().enumerate().skip(start + 1) {
            let inner = inner.trim();
            // ELF ends a function with `.Lfunc_end` and a `.size` directive;
            // Mach-O has neither and ends one with `.cfi_endproc`. Reading only
            // the ELF framing made every body run to end-of-file on macOS, so
            // the gate attributed the *rest of the crate's* instructions ---
            // including a dense NEON kernel's `sdot` --- to the gather it was
            // meant to be reading.
            if inner.starts_with(".Lfunc_end")
                || inner.starts_with(".size")
                || inner.starts_with(".cfi_endproc")
            {
                break;
            }
            if inner.is_empty()
                || inner.starts_with('.')
                || inner.starts_with('#')
                || inner.starts_with("//")
                || inner.ends_with(':')
            {
                continue;
            }
            body.push((i + 1, inner));
        }
        out.push((label.to_string(), body));
    }
    out
}

/// Integer multiply opcodes, across the ISAs this workspace targets.
///
/// Shifts are *not* here. A shift by a constant is how a compiler scales an
/// index, and index arithmetic cannot change an output value --- which is the
/// same `R3-ok` distinction the source-level gate draws. What is forbidden is a
/// multiply of two *values*, which in the column loop would mean a product that
/// the table was supposed to have already computed.
const INTEGER_MULTIPLY_OPCODES: &[&str] = &[
    // x86
    "imul",
    "mul",
    "mulx",
    "pmul",
    "vpmul",
    "pmadd",
    "vpmadd",
    "pmull",
    "vpdp",
    // aarch64
    "madd",
    "msub",
    "mneg",
    "smull",
    "umull",
    "smlal",
    "umlal",
    "mla",
    "mls",
    "sdot",
    "udot",
    // wasm
    "i32.mul",
    "i64.mul",
    "i32x4.mul",
    "i16x8.mul",
    "i32x4.dot",
];

/// The integer multiply opcode on this instruction line, if any.
fn integer_multiply_opcode(line: &str) -> Option<&'static str> {
    let mnemonic = line.split_whitespace().next()?;
    INTEGER_MULTIPLY_OPCODES
        .iter()
        .copied()
        .find(|op| mnemonic == *op || mnemonic.starts_with(op))
}

/// Whether two adjacent AArch64 instructions are the compiler's exact
/// quotient/remainder identity: `q = n / d; r = n - q*d`.
///
/// `msub` by itself remains a forbidden value multiply. The exception is
/// deliberately operand-checked and adjacent, so neither an unrelated divide
/// nor a product merely spelled as multiply-subtract can satisfy it.
fn is_aarch64_remainder_pair(divide: &str, remainder: &str) -> bool {
    fn operands<'a>(line: &'a str, mnemonic: &str) -> Option<Vec<&'a str>> {
        let mut parts = line.splitn(2, char::is_whitespace);
        if parts.next()? != mnemonic {
            return None;
        }
        let operands: Vec<_> = parts
            .next()?
            .split(',')
            .map(str::trim)
            .filter(|operand| !operand.is_empty())
            .collect();
        Some(operands)
    }

    fn register_number(register: &str) -> Option<&str> {
        let suffix = register
            .strip_prefix('w')
            .or_else(|| register.strip_prefix('x'))?;
        (!suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_digit())).then_some(suffix)
    }

    let Some(divide) = operands(divide, "udiv") else {
        return false;
    };
    let Some(remainder) = operands(remainder, "msub") else {
        return false;
    };
    if divide.len() != 3 || remainder.len() != 4 {
        return false;
    }
    register_number(divide[0]) == register_number(remainder[1])
        && register_number(divide[1]) == register_number(remainder[3])
        && register_number(divide[2]) == register_number(remainder[2])
}

/// Float arithmetic opcodes, across the ISAs this workspace targets.
///
/// Conversions (`cvtsi2ss`, `scvtf`) and moves (`movss`, `fmov`) are *not*
/// here: decoding a float and encoding one back are exactly what the library
/// does, and forbidding them would forbid the operation rather than the defect.
/// What is forbidden is arithmetic *between* two floats.
const FLOAT_OPCODES: &[&str] = &[
    // x86
    "addss", "addsd", "addps", "addpd", "subss", "subsd", "subps", "subpd", "mulss", "mulsd",
    "mulps", "mulpd", "divss", "divsd", "divps", "divpd", "vaddss", "vaddsd", "vaddps", "vaddpd",
    "vsubss", "vsubsd", "vsubps", "vsubpd", "vmulss", "vmulsd", "vmulps", "vmulpd", "vfmadd",
    "vfmsub", "vfnmadd", "haddps", "haddpd", // aarch64
    "fadd", "fsub", "fmul", "fdiv", "fmadd", "fmsub", "fnmadd", "fmla", "fmls",
    // wasm
    "f32.add", "f32.sub", "f32.mul", "f64.add", "f64.sub", "f64.mul",
];

/// The float opcode on this assembly line, if any.
fn float_opcode(line: &str) -> Option<&'static str> {
    // Labels, directives, and comments are not instructions.
    if line.is_empty()
        || line.starts_with('.')
        || line.starts_with('#')
        || line.starts_with("//")
        || line.ends_with(':')
    {
        return None;
    }
    let mnemonic = line.split_whitespace().next()?;
    FLOAT_OPCODES
        .iter()
        .copied()
        .find(|op| mnemonic == *op || mnemonic.starts_with(op))
}

pub(crate) fn asm_files(dir: &Path) -> Result<Vec<PathBuf>, Fail> {
    let mut out = Vec::new();
    let mut stack = std::vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else {
            continue;
        };
        for entry in entries {
            let path = entry?.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "s") {
                out.push(path);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atlas_literals_are_not_integer_type_names_cm_01() {
        assert!(contains_literal_numeral("3;", "3"));
        assert!(contains_literal_numeral("-3i32;", "3"));
        assert!(!contains_literal_numeral(
            "atlas::PROJECTOR_INPUT_SCALE as i128;",
            "128"
        ));
        assert!(!contains_literal_numeral("atlas::CONTEXT as u32;", "32"));
    }

    #[test]
    fn an_explicit_integer_cast_boundary_stops_float_taint_cu_01() {
        let mut scope = FloatScope::default();
        scope.observe("fn relabel(x: f32) -> u32 {");
        scope.observe("let raw: u32 = bytemuck::cast::<f32, u32>(x);");
        assert_eq!(scope.arithmetic_on_a_float("raw + raw"), None);

        scope.observe("let planted: f32 = x;");
        assert_eq!(
            scope.arithmetic_on_a_float("planted + planted"),
            Some("planted".to_string())
        );
    }

    /// `CU-06`: AArch64 may fuse a permitted remainder into `msub`, but only
    /// the exact adjacent quotient identity is address arithmetic. A bare or
    /// operand-mismatched `msub` remains a value multiply.
    #[test]
    fn aarch64_remainder_lowering_is_not_a_value_product_cu_06() {
        let clean = ("udiv\tx2, x8, x0", "msub\tw8, w2, w0, w8");
        assert!(is_aarch64_remainder_pair(clean.0, clean.1));
        assert_eq!(integer_multiply_opcode(clean.1), Some("msub"));

        for dirty in [
            ("add\tx2, x8, x0", "msub\tw8, w2, w0, w8"),
            ("udiv\tx2, x8, x0", "msub\tw8, w3, w0, w8"),
            ("udiv\tx2, x8, x0", "msub\tw8, w2, w1, w8"),
        ] {
            assert!(
                !is_aarch64_remainder_pair(dirty.0, dirty.1),
                "a non-remainder multiply must stay visible: {dirty:?}"
            );
        }
    }

    /// `CU-10`: no tropical sequence issues a multiply.
    ///
    /// The substance of the claim is read from the emitted instructions by
    /// [`audit_disassembly`], which `just purity` runs. What a test can pin is
    /// the two ways that reading goes vacuous, and both are pinned here
    /// two-sidedly --- because a gate that reads the wrong bodies, or that reads
    /// the right ones with a blind scanner, passes exactly as loudly as one that
    /// works.
    #[test]
    fn no_tropical_sequence_issues_a_multiply_cu_10() {
        // One: the *scope*. Every entry point in the family carries
        // `TROPICAL_SEQUENCE` in its symbol by convention, so a rename that
        // dropped the fragment would silently empty the gate. The source is what
        // says so, and it says so for every one of them.
        let sources = [
            include_str!("../../crates/uor-matmul-kernels/src/isa/portable.rs"),
            include_str!("../../crates/uor-matmul-kernels/src/isa/x86.rs"),
            include_str!("../../crates/uor-matmul-kernels/src/isa/arm.rs"),
            include_str!("../../crates/uor-matmul-kernels/src/isa/wasm.rs"),
        ];
        let mut entry_points = 0usize;
        for text in sources {
            for line in text.lines() {
                let Some(rest) = line.trim().split("fn ").nth(1) else {
                    continue;
                };
                let name: String = rest
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                // A sequence is tropical exactly when it names the semiring. The
                // check is the other way round from the obvious one: it does not
                // ask whether every `trop` function carries the fragment --- it
                // asks whether anything the family would recognise as tropical
                // fails to, which is the rename this catches.
                if name.contains("trop") {
                    assert!(
                        name.contains(TROPICAL_SEQUENCE),
                        "`{name}` is a tropical sequence whose symbol does not carry \
                         `{TROPICAL_SEQUENCE}`, so `audit-disassembly` would never read it"
                    );
                    entry_points += 1;
                }
            }
        }
        assert!(
            entry_points >= 4,
            "one sequence per ISA at least, or the scope check is itself vacuous; found \
             {entry_points}"
        );

        // Two: the *scanner*. A body of the shape these sequences emit ---
        // saturating adds and signed maxima --- must read clean, and the same
        // body with one multiply in it must not. Without the second half a
        // scanner that recognised nothing would pass.
        let clean = "\
_ZN4trop_i16_probe:
    vpaddsw %ymm1, %ymm2, %ymm0
    vpmaxsw %ymm0, %ymm3, %ymm3
    ret
";
        let dirty = "\
_ZN4trop_i16_probe:
    vpaddsw %ymm1, %ymm2, %ymm0
    imull $3, %eax, %eax
    vpmaxsw %ymm0, %ymm3, %ymm3
    ret
";
        for (text, want) in [(clean, false), (dirty, true)] {
            let bodies = function_bodies(text, TROPICAL_SEQUENCE);
            assert_eq!(bodies.len(), 1, "the probe is one body");
            let found = bodies[0]
                .1
                .iter()
                .any(|(_, line)| integer_multiply_opcode(line).is_some());
            assert_eq!(
                found, want,
                "the scanner must report a multiply in a tropical body and only there"
            );
        }
    }
}
