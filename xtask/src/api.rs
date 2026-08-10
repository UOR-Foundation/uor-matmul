//! The exact public API lock.
//!
//! The gate began as a one-way promise that no public item was removed or
//! renamed. This refactor strengthens that contract to exact name-level
//! equality: additions are changes too. The flat re-export lists in
//! `crates/uor-matmul/src/lib.rs` name about ninety items,
//! which is what a reader counts --- but the facade also re-exports four whole
//! crates as modules (`pub use uor_matmul_codec as codec;` and three more), so
//! a caller can name every public item of all five crates, and *that* is the
//! surface a promise not to remove anything is about.
//!
//! `cargo public-api` is the tool for this and it is not available here: it
//! needs a nightly toolchain to build rustdoc's JSON, and the workspace pins
//! stable 1.97.1 (`rust-toolchain.toml`), on which `cargo doc
//! --output-format json` is not even a recognised argument --- measured, not
//! assumed. `cargo doc`'s JSON was the first thing tried for that reason. So
//! the surface is read from the source, and the enumeration below says exactly
//! what it covers, because a lock whose coverage is vague locks nothing.
//!
//! # What is enumerated
//!
//! For each of the five shipped crates, every declaration that a caller outside
//! the crate can name:
//!
//! - `pub` items --- `mod`, `fn`, `struct`, `enum`, `union`, `trait`, `type`,
//!   `const`, `static` --- in modules that are themselves publicly reachable;
//! - every name a `pub use` introduces, including a whole-crate re-export such
//!   as `pub use uor_matmul_codec as codec`;
//! - `pub` fields of a `pub` struct, and every variant of a `pub` enum, both of
//!   which a caller names directly;
//! - the associated items of a `pub` trait, which are public without saying so;
//! - `pub` associated items of an inherent `impl` on a type this crate makes
//!   public;
//! - `#[macro_export] macro_rules!` declarations, at the crate-root path Rust
//!   gives them.
//!
//! Everything under a `#[cfg(test)]` is excluded, and so is anything inside a
//! function body. Items in private modules are excluded except
//! `#[macro_export]`, whose Rust path is at the crate root and is therefore
//! nameable from outside.
//!
//! # What this locks, and what it does not
//!
//! What is recorded is a *path and a kind*, so what is locked is the exact set
//! of publicly nameable declarations: a removal fails, a rename fails as a
//! removal plus an addition, and an addition fails. The baseline moves only
//! under an explicit `--write`, so accepting any surface change is a reviewable
//! artifact change rather than an unnoticed widening.
//!
//! **A narrowing does not fail this gate.** The constraint the refactor is held
//! to has three words in it --- removed, renamed, or *narrowed* --- and this
//! covers two. Narrowing a bound (`E: Element` back to `E: IntegerElement`),
//! narrowing a parameter type, or adding a supertrait leaves every path and
//! kind exactly where it was, and all three were planted and all three passed.
//! The type checker is not a backstop either: it catches a narrowing only when
//! some caller inside this workspace happens to exercise the widened part, which
//! is a coincidence and not a check.
//!
//! That is stated here rather than fixed, because fixing it means recording
//! normalized signatures --- which is a different and much larger gate, one
//! whose false-positive rate on formatting alone would make it a nuisance
//! rather than a lock. What holds the third word is review, and this doc is
//! what tells a reviewer that it is theirs to hold. A gate whose coverage is
//! overstated is worse than a narrow one, because it is read as the whole
//! promise.
//!
//! # What is not enumerated, and why
//!
//! Trait *implementations*. Roughly half of the ones these crates provide are
//! written by `macro_rules!` --- `impl_element_for_signed!`,
//! `impl_encode_from_limbs!`, `impl_tropical_element!` --- and expand at compile
//! time, so a source-level enumeration of impls would be a list that is
//! silently partial, which is worse than no list: it would read as coverage. The
//! trait itself, its associated items, and the types it is implemented for are
//! all locked here, and the impls themselves are what the `CB-*` and `CD-*`
//! parity gates exercise directly.
//!
//! # The direction of the check
//!
//! Equality is symmetric. This repository's current refactor contract is that
//! the public API does not change, so an unreviewed addition is a failure for the
//! same reason as an unreviewed removal. The baseline is rewritten only under an
//! explicit `--write`, exactly as `check-model --write` works.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use crate::audit::SHIPPED;
use crate::Fail;

/// Where the accepted surface is recorded, relative to the repository root.
///
/// It lives with the conformance crate rather than with this one because that
/// is where the repository keeps a gate's *evidence*: `xtask` holds gate code
/// and has never held a record, while every other thing a gate compares against
/// --- `CONFORMANCE.md`, `crates/uor-matmul-core/src/generated.rs`,
/// `model/*.toml`, `oracles/` --- is a committed artifact outside it. The
/// conformance crate is `publish = false`, so nothing a consumer builds carries
/// the file.
pub const BASELINE_PATH: &str = "crates/uor-matmul-conformance/api-baseline.txt";

/// The header the generated baseline carries.
const HEADER: &str = "\
# The governed public surface of the shipped crates, one declaration per line.
#
# Generated by `cargo run -p xtask -- check-api --write`; do not edit by hand.
# `cargo run -p xtask -- check-api` fails if this set and the source's set differ
# in either direction. Regenerating is how a deliberate surface change is
# accepted, and the diff is then a reviewable edit to this file.
";

/// The public surface equals the committed surface exactly.
pub fn check_api(root: &Path, write: bool) -> Result<(), Fail> {
    let current = surface(root)?;
    let path = root.join(BASELINE_PATH);

    let rendered = render(&current);
    if write {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, &rendered)?;
        println!("wrote {} ({} declarations)", path.display(), current.len());
        return Ok(());
    }

    let committed = std::fs::read_to_string(&path).map_err(|e| {
        format!(
            "{}: {e}\nThe public API lock has no baseline, so nothing constrains a removal. \
             Run `cargo run -p xtask -- check-api --write`.",
            path.display()
        )
    })?;
    let baseline: BTreeSet<String> = committed
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(str::to_string)
        .collect();
    if baseline.is_empty() {
        return Err(format!(
            "{} lists no declarations, so the lock would pass vacuously --- the surface it \
             is supposed to hold is {} items wide.",
            path.display(),
            current.len()
        )
        .into());
    }

    exact_surface(&baseline, &current)?;
    println!(
        "check-api: {} public declarations across {} shipped crates, exactly equal to the \
         committed surface",
        current.len(),
        SHIPPED.len()
    );
    Ok(())
}

/// Compare the two sets in both directions.
///
/// Kept separate from source enumeration so both teeth can be falsified without
/// manufacturing a temporary crate tree in a unit test.
fn exact_surface(baseline: &BTreeSet<String>, current: &BTreeSet<String>) -> Result<(), Fail> {
    let removed: Vec<&String> = baseline.difference(current).collect();
    let added: Vec<&String> = current.difference(baseline).collect();
    if removed.is_empty() && added.is_empty() {
        return Ok(());
    }

    let render = |prefix: char, entries: &[&String]| {
        entries
            .iter()
            .map(|entry| format!("{prefix} {entry}"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    Err(format!(
        "the public API differs from its committed surface: {} removed, {} added. The \
         refactor contract is exact API equality; a rename appears in both columns. If a \
         surface change is deliberate, regenerate with `cargo run -p xtask -- check-api \
         --write` so it is an explicit artifact diff to {}. Signature narrowing remains the \
         separately documented review boundary.\n\nremoved:\n{}\n\nadded:\n{}",
        removed.len(),
        added.len(),
        BASELINE_PATH,
        render('-', &removed),
        render('+', &added),
    )
    .into())
}

/// The baseline file's text.
fn render(surface: &BTreeSet<String>) -> String {
    let mut out = String::from(HEADER);
    out.push('\n');
    for entry in surface {
        out.push_str(entry);
        out.push('\n');
    }
    out
}

/// Every publicly nameable declaration of the shipped crates.
fn surface(root: &Path) -> Result<BTreeSet<String>, Fail> {
    let mut out = BTreeSet::new();
    for krate in SHIPPED {
        let dir = root.join("crates").join(krate).join("src");
        if !dir.is_dir() {
            return Err(format!(
                "{} has no `src`, so its public surface would be enumerated as empty and \
                 every removal from it would pass",
                dir.display()
            )
            .into());
        }
        let mut files = Vec::new();
        gather_rs(&dir, &mut files)?;
        files.sort();

        let mut modules = Modules::default();
        for file in &files {
            let text = std::fs::read_to_string(file)?;
            modules.read(&dir, file, &text);
        }

        // Which types this crate makes public, so that an inherent `impl` on a
        // private helper does not contribute methods no caller can reach.
        let mut public_types = BTreeSet::new();
        for file in &files {
            let text = std::fs::read_to_string(file)?;
            collect_public_types(&text, &mut public_types);
        }

        let ident = krate.replace('-', "_");
        let mut found = 0usize;
        for file in &files {
            let Some(module) = modules.path_of(&dir, file) else {
                continue;
            };
            let text = std::fs::read_to_string(file)?;
            for entry in declarations(&text, &public_types) {
                let mut path = ident.clone();
                let declared_module = if entry.crate_root { &[][..] } else { &module };
                for seg in declared_module.iter().chain(entry.path.iter()) {
                    path.push_str("::");
                    path.push_str(seg);
                }
                out.insert(format!("{path} [{}]", entry.kind));
                found += 1;
            }
        }
        if found == 0 {
            return Err(format!(
                "no public declaration was read from `{krate}`, so every claim about its \
                 surface would pass without anything being read"
            )
            .into());
        }
    }
    Ok(out)
}

fn gather_rs(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), Fail> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            gather_rs(&path, out)?;
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
    Ok(())
}

/// The module tree, as the `mod` declarations spell it.
///
/// A file's module path is usually its path under `src`, but not always: this
/// workspace selects a per-architecture implementation with
/// `#[path = "absent_x86.rs"] pub mod x86;`, and filing that file's items under
/// `isa::absent_x86` would name a module that does not exist. The declarations
/// are therefore read rather than inferred, and only the file layout is used
/// where no declaration overrides it.
#[derive(Default)]
struct Modules {
    /// Absolute file path to the module path it is compiled as.
    overrides: BTreeMap<PathBuf, Vec<String>>,
    /// Module paths that are declared without `pub`, whose items no outside
    /// caller can name.
    private: BTreeSet<Vec<String>>,
}

impl Modules {
    fn read(&mut self, src: &Path, file: &Path, text: &str) {
        let Some(here) = file_module(src, file) else {
            return;
        };
        let dir = file.parent().unwrap_or(src);
        let mut run: Vec<&str> = Vec::new();
        for raw in text.lines() {
            let line = code_of(raw).trim().to_string();
            if line.starts_with("#[") {
                run.push(raw.trim());
                continue;
            }
            if line.is_empty() {
                continue;
            }
            if let Some((vis, name)) = mod_declaration(&line) {
                let mut path = here.clone();
                path.push(name.clone());
                if vis != Vis::Pub {
                    self.private.insert(path.clone());
                }
                if let Some(target) = run.iter().find_map(|a| path_attribute(a)) {
                    self.overrides.insert(dir.join(target), path);
                }
            }
            run.clear();
        }
    }

    /// The module path a file's items belong to, or `None` if no caller can
    /// name them.
    fn path_of(&self, src: &Path, file: &Path) -> Option<Vec<String>> {
        let path = self
            .overrides
            .get(file)
            .cloned()
            .or_else(|| file_module(src, file))?;
        for n in 0..path.len() {
            if self.private.contains(&path[..=n]) {
                return None;
            }
        }
        Some(path)
    }
}

/// The module path a file occupies by its position under `src`.
fn file_module(src: &Path, file: &Path) -> Option<Vec<String>> {
    let rel = file.strip_prefix(src).ok()?;
    let mut segs: Vec<String> = rel
        .components()
        .map(|c| c.as_os_str().to_string_lossy().to_string())
        .collect();
    let last = segs.pop()?;
    let stem = last.strip_suffix(".rs")?;
    if stem != "lib" && stem != "mod" {
        segs.push(stem.to_string());
    }
    Some(segs)
}

/// The file a `#[path = "..."]` attribute names.
fn path_attribute(attr: &str) -> Option<String> {
    let inner = attr.strip_prefix("#[")?.strip_suffix(']')?.trim();
    let rest = inner.strip_prefix("path")?.trim().strip_prefix('=')?.trim();
    Some(rest.trim_matches('"').to_string())
}

/// A `mod name;` or `mod name {` declaration, with its visibility.
fn mod_declaration(line: &str) -> Option<(Vis, String)> {
    let (vis, rest) = visibility(line);
    let name = rest.strip_prefix("mod ")?;
    let name: String = name
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    (!name.is_empty()).then_some((vis, name))
}

/// How widely an item is declared.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Vis {
    /// `pub`, with no restriction: nameable from outside the crate.
    Pub,
    /// `pub(crate)`, `pub(super)`, `pub(in ...)`, or nothing at all.
    Inside,
}

/// Split a declaration's visibility from the rest of it.
fn visibility(line: &str) -> (Vis, &str) {
    let Some(rest) = line.strip_prefix("pub") else {
        return (Vis::Inside, line);
    };
    if let Some(restricted) = rest.strip_prefix('(') {
        let Some(end) = restricted.find(')') else {
            return (Vis::Inside, line);
        };
        return (Vis::Inside, restricted[end + 1..].trim_start());
    }
    match rest.strip_prefix(' ') {
        Some(r) => (Vis::Pub, r.trim_start()),
        None => (Vis::Inside, line),
    }
}

/// One declaration on the public surface.
struct Decl {
    /// Segments below the file's module: an enclosing type or trait, then the
    /// item's own name.
    path: Vec<String>,
    /// What kind of thing it is, so a `struct` replaced by a `fn` of the same
    /// name reads as the removal it is.
    kind: &'static str,
    /// `#[macro_export]` places the declaration at the crate root regardless
    /// of which source module contains it.
    crate_root: bool,
}

/// What a block being tracked contributes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Scope {
    /// An inline `pub mod`, a `pub struct`, `pub enum`, `pub trait`, or an
    /// inherent `impl` on a public type: items directly inside it are surface.
    Open,
    /// A function body, a `macro_rules!`, a `#[cfg(test)]` block, a private
    /// module, a trait implementation: nothing inside is separately nameable.
    Closed,
}

/// A block the scan is inside.
struct Frame {
    /// Brace depth immediately inside the block.
    depth: i32,
    /// The path segment it contributes, if any.
    seg: Option<String>,
    scope: Scope,
    /// This block is test-only. Unlike an ordinary private block it must also
    /// hide a nested `#[macro_export]`, whose path would otherwise escape to
    /// the crate root.
    test_hidden: bool,
    /// What kind of body it is, which decides how its unprefixed lines read: a
    /// trait's are associated items, an enum's are variants.
    body: Body,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Body {
    /// Braces holding items that must say `pub` to be surface.
    Items,
    /// A `pub trait`: its associated items are public without saying so.
    Trait,
    /// A `pub enum`: every variant is public.
    Enum,
}

/// Every public type name a crate declares, used to tell an inherent `impl` on
/// a public type from one on a private helper.
fn collect_public_types(text: &str, out: &mut BTreeSet<String>) {
    for raw in text.lines() {
        let line = code_of(raw);
        let line = line.trim();
        let (vis, rest) = visibility(line);
        if vis != Vis::Pub {
            continue;
        }
        for keyword in ["struct ", "enum ", "union ", "trait ", "type "] {
            let Some(tail) = rest.strip_prefix(keyword) else {
                continue;
            };
            let name: String = tail
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if !name.is_empty() {
                out.insert(name);
            }
            break;
        }
    }
}

/// The public declarations one file contributes, below its own module path.
fn declarations(text: &str, public_types: &BTreeSet<String>) -> Vec<Decl> {
    let mut out = Vec::new();
    let mut frames: Vec<Frame> = Vec::new();
    let mut depth = 0i32;
    let mut run: Vec<String> = Vec::new();
    // A `use` or a signature may span lines; both are joined until they close.
    let mut pending = String::new();

    for raw in text.lines() {
        let line = code_of(raw);
        let line = line.trim();
        if line.starts_with("#[") || line.starts_with("#!") {
            run.push(line.to_string());
            continue;
        }
        if line.is_empty() {
            continue;
        }

        let statement = if pending.is_empty() {
            line.to_string()
        } else {
            let joined = format!("{pending} {line}");
            pending.clear();
            joined
        };
        // A `pub use` is only complete at its semicolon.
        let is_use = visibility(&statement).1.starts_with("use ");
        if is_use && !statement.contains(';') {
            pending = statement;
            run.clear();
            continue;
        }

        let open = frames.iter().all(|f| f.scope == Scope::Open);
        let hidden = run.iter().any(|a| a.starts_with("#[cfg(test)]"));
        let test_hidden = hidden || frames.iter().any(|frame| frame.test_hidden);
        let macro_export = run.iter().any(|attribute| attribute == "#[macro_export]");
        let inside = frames.last().map_or(Body::Items, |f| f.body);
        if !test_hidden && (open || macro_export) {
            emit(&statement, inside, &frames, macro_export, &mut out);
        }

        // Whatever this line opens, it is only surface if everything outside it
        // was, and a `#[cfg(test)]` block is never surface.
        if raw_opens(raw) {
            let frame = frame_for(
                &statement,
                public_types,
                depth,
                hidden || !open,
                test_hidden,
            );
            frames.push(frame);
        }
        run.clear();

        depth += raw.matches('{').count() as i32;
        depth -= raw.matches('}').count() as i32;
        while frames.last().is_some_and(|f| f.depth > depth) {
            frames.pop();
        }
    }
    out
}

/// Does this line open a block?
fn raw_opens(raw: &str) -> bool {
    let code = code_of(raw);
    let opens = code.matches('{').count() as i32;
    let closes = code.matches('}').count() as i32;
    opens > closes
}

/// Record whatever `statement` declares.
fn emit(statement: &str, inside: Body, frames: &[Frame], macro_export: bool, out: &mut Vec<Decl>) {
    let prefix: Vec<String> = frames.iter().filter_map(|f| f.seg.clone()).collect();
    let mut push = |name: String, kind: &'static str| {
        let mut path = prefix.clone();
        path.push(name);
        out.push(Decl {
            path,
            kind,
            crate_root: false,
        });
    };

    let (vis, rest) = visibility(statement);

    // An enum's variants and a trait's associated items are public because the
    // enclosing item is; they never say `pub` themselves.
    if vis != Vis::Pub {
        if macro_export {
            if let Some(("macro", name)) = item_name(statement) {
                out.push(Decl {
                    path: vec![name],
                    kind: "macro",
                    crate_root: true,
                });
            }
            return;
        }
        match inside {
            Body::Enum => {
                if let Some(name) = variant_name(statement) {
                    push(name, "variant");
                }
            }
            Body::Trait => {
                if let Some((kind, name)) = item_name(statement) {
                    push(name, kind);
                }
            }
            Body::Items => {}
        }
        return;
    }

    if let Some(tree) = rest.strip_prefix("use ") {
        let tree = tree.trim_end_matches(';').trim();
        let mut names = Vec::new();
        use_names(tree, "", &mut names);
        for name in names {
            push(name, "use");
        }
        return;
    }
    if let Some((kind, name)) = item_name(rest) {
        push(name, kind);
        return;
    }
    // A `pub` field of a struct: `pub name: Ty,`.
    if inside == Body::Items {
        if let Some((name, _)) = rest.split_once(':') {
            let name = name.trim();
            if !name.is_empty() && name.chars().all(|c| c.is_alphanumeric() || c == '_') {
                push(name.to_string(), "field");
            }
        }
    }
}

/// The kind and name of an item declaration, with its modifiers stripped.
fn item_name(rest: &str) -> Option<(&'static str, String)> {
    let mut s = rest.trim();
    // `default`, `unsafe`, `async`, and `extern "C"` all decorate a function;
    // `const` decorates one too, unless it *is* the item.
    loop {
        let before = s;
        for m in ["default ", "unsafe ", "async "] {
            if let Some(r) = s.strip_prefix(m) {
                s = r.trim_start();
            }
        }
        if let Some(r) = s.strip_prefix("extern ") {
            let r = r.trim_start();
            s = match r.strip_prefix('"') {
                Some(q) => q.split_once('"').map_or(r, |(_, t)| t).trim_start(),
                None => r,
            };
        }
        if let Some(r) = s.strip_prefix("const ") {
            if r.trim_start().starts_with("fn ") {
                s = r.trim_start();
            }
        }
        if s == before {
            break;
        }
    }
    for (keyword, kind) in [
        ("mod ", "mod"),
        ("fn ", "fn"),
        ("struct ", "struct"),
        ("enum ", "enum"),
        ("union ", "union"),
        ("trait ", "trait"),
        ("type ", "type"),
        ("const ", "const"),
        ("static ", "static"),
        ("macro_rules! ", "macro"),
    ] {
        let Some(tail) = s.strip_prefix(keyword) else {
            continue;
        };
        let tail = tail.trim_start().trim_start_matches("mut ");
        let name: String = tail
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if !name.is_empty() {
            return Some((kind, name));
        }
    }
    None
}

/// The variant an enum body line declares.
fn variant_name(line: &str) -> Option<String> {
    let name: String = line
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    let first = name.chars().next()?;
    // A variant starts with an uppercase letter in every one of these crates,
    // and the alternative --- accepting anything --- would read a struct field
    // or a `where` clause as a variant.
    (first.is_uppercase()).then_some(name)
}

/// The names a `use` tree introduces, given the prefix it hangs from.
fn use_names(tree: &str, prefix: &str, out: &mut Vec<String>) {
    let t = tree.trim();
    if let Some(open) = t.find('{') {
        let Some(close) = t.rfind('}') else {
            return;
        };
        let head = t[..open].trim().trim_end_matches("::");
        for part in top_level_commas(&t[open + 1..close]) {
            use_names(part, head, out);
        }
        return;
    }
    if t.is_empty() {
        return;
    }
    let name = match t.rsplit_once(" as ") {
        Some((_, alias)) => alias.trim(),
        None => {
            let leaf = t.rsplit("::").next().unwrap_or(t).trim();
            if leaf == "self" {
                prefix.rsplit("::").next().unwrap_or(prefix).trim()
            } else {
                leaf
            }
        }
    };
    if !name.is_empty() && name != "*" && name != "_" {
        out.push(name.to_string());
    }
}

/// Split on commas outside braces and parentheses.
fn top_level_commas(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    for (i, c) in s.char_indices() {
        match c {
            '{' | '(' | '<' => depth += 1,
            '}' | ')' | '>' => depth -= 1,
            ',' if depth == 0 => {
                out.push(s[start..i].trim());
                start = i + c.len_utf8();
            }
            _ => {}
        }
    }
    let tail = s[start..].trim();
    if !tail.is_empty() {
        out.push(tail);
    }
    out
}

/// The frame a block-opening statement pushes.
fn frame_for(
    statement: &str,
    public_types: &BTreeSet<String>,
    depth: i32,
    closed: bool,
    test_hidden: bool,
) -> Frame {
    let shut = |body| Frame {
        depth: depth + 1,
        seg: None,
        scope: Scope::Closed,
        test_hidden,
        body,
    };
    if closed {
        return shut(Body::Items);
    }
    let (vis, rest) = visibility(statement);

    // An inherent `impl` contributes its type's name; a trait `impl` and an
    // impl on a private type contribute nothing a caller names separately.
    if let Some(subject) = inherent_impl_subject(rest.trim()) {
        return if public_types.contains(&subject) {
            Frame {
                depth: depth + 1,
                seg: Some(subject),
                scope: Scope::Open,
                test_hidden,
                body: Body::Items,
            }
        } else {
            shut(Body::Items)
        };
    }
    if vis != Vis::Pub {
        return shut(Body::Items);
    }
    let Some((kind, name)) = item_name(rest) else {
        return shut(Body::Items);
    };
    let body = match kind {
        "trait" => Body::Trait,
        "enum" => Body::Enum,
        "mod" | "struct" | "union" => Body::Items,
        // A `pub fn`, `pub const`, `pub static`: the braces are a body, and a
        // body holds no surface.
        _ => return shut(Body::Items),
    };
    Frame {
        depth: depth + 1,
        seg: Some(name),
        scope: Scope::Open,
        test_hidden,
        body,
    }
}

/// The type an inherent `impl` block is on, if the statement is one.
fn inherent_impl_subject(statement: &str) -> Option<String> {
    let rest = statement.strip_prefix("impl")?;
    // `impl<'a, E>` --- skip the generic parameter list before the type.
    let rest = match rest.trim_start().strip_prefix('<') {
        Some(after) => {
            let mut depth = 1i32;
            let mut end = None;
            for (i, c) in after.char_indices() {
                match c {
                    '<' => depth += 1,
                    '>' => {
                        depth -= 1;
                        if depth == 0 {
                            end = Some(i);
                            break;
                        }
                    }
                    _ => {}
                }
            }
            &after[end? + 1..]
        }
        None => rest,
    };
    let head = rest.trim();
    // `impl Trait for Type` is not inherent.
    let subject = match head.split_once(" for ") {
        Some(_) => return None,
        None => head,
    };
    let name: String = subject
        .trim_start()
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    (!name.is_empty()).then_some(name)
}

/// A line with its trailing `//` comment removed.
///
/// String literals holding `//` do not occur in a declaration in these crates,
/// and a doc comment is stripped whole because it starts with the marker.
fn code_of(raw: &str) -> &str {
    match raw.find("//") {
        Some(at) => &raw[..at],
        None => raw,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_surface_rejects_both_directions() {
        let baseline = BTreeSet::from(["crate::kept [fn]".to_string()]);
        assert!(exact_surface(&baseline, &baseline).is_ok());

        let added = BTreeSet::from([
            "crate::kept [fn]".to_string(),
            "crate::plant [fn]".to_string(),
        ]);
        let error = exact_surface(&baseline, &added)
            .expect_err("an addition must move the exact surface")
            .to_string();
        assert!(error.contains("0 removed, 1 added"), "{error}");
        assert!(error.contains("+ crate::plant [fn]"), "{error}");

        let empty = BTreeSet::new();
        let error = exact_surface(&baseline, &empty)
            .expect_err("a removal must move the exact surface")
            .to_string();
        assert!(error.contains("1 removed, 0 added"), "{error}");
        assert!(error.contains("- crate::kept [fn]"), "{error}");
    }

    fn entries(src: &str) -> Vec<String> {
        let mut types = BTreeSet::new();
        collect_public_types(src, &mut types);
        declarations(src, &types)
            .into_iter()
            .map(|d| format!("{} [{}]", d.path.join("::"), d.kind))
            .collect()
    }

    /// The shapes these crates actually write, each read as the caller would
    /// name it.
    #[test]
    fn the_scan_reads_the_shapes_the_shipped_crates_write() {
        let got = entries(
            "pub struct Shape {\n    pub m: usize,\n    hidden: usize,\n}\n\
             impl Shape {\n    pub const fn row_major(cols: usize) -> Self { Self { m: 0 } }\n\
             fn helper() {}\n}\n\
             pub trait Element {\n    type Acc;\n    fn mac(a: u8);\n}\n\
             pub enum Route {\n    Reference,\n    Kernelized(u8),\n}\n\
             pub use tropical::{Trop, TropAcc as Acc};\n\
             pub use uor_matmul_codec as codec;\n\
             pub fn dot_ref() {}\n\
             pub(crate) fn hidden_helper() {}\n",
        );
        for expected in [
            "Shape [struct]",
            "Shape::m [field]",
            "Shape::row_major [fn]",
            "Element [trait]",
            "Element::Acc [type]",
            "Element::mac [fn]",
            "Route [enum]",
            "Route::Reference [variant]",
            "Route::Kernelized [variant]",
            "Trop [use]",
            "Acc [use]",
            "codec [use]",
            "dot_ref [fn]",
        ] {
            assert!(
                got.contains(&expected.to_string()),
                "{expected} missing: {got:?}"
            );
        }
        for absent in ["Shape::hidden [field]", "hidden_helper [fn]", "helper [fn]"] {
            assert!(
                !got.contains(&absent.to_string()),
                "{absent} must not be surface"
            );
        }
    }

    /// A test module is not API, and neither is anything inside a function.
    #[test]
    fn nothing_inside_a_test_module_or_a_function_body_is_surface() {
        let got = entries(
            "pub fn outer() {\n    pub struct Inner;\n}\n\
             #[cfg(test)]\nmod tests {\n    pub fn fixture() {}\n    pub struct Fixture;\n}\n\
             pub fn after() {}\n",
        );
        assert_eq!(
            got,
            vec!["outer [fn]".to_string(), "after [fn]".to_string()]
        );
    }

    /// An inherent `impl` on a type the crate does not export contributes
    /// nothing, or renaming a private helper's method would read as an API
    /// break.
    #[test]
    fn an_impl_on_a_private_type_is_not_surface() {
        let got = entries("struct Helper;\nimpl Helper {\n    pub fn step() {}\n}\n");
        assert!(got.is_empty(), "{got:?}");
    }

    /// A multi-line `pub use` introduces every name in its braces.
    #[test]
    fn a_multi_line_reexport_introduces_every_name() {
        let got = entries("pub use alphabet::{\n    as_alphabet, Alphabet,\n    Bnd,\n};\n");
        assert_eq!(
            got,
            vec![
                "as_alphabet [use]".to_string(),
                "Alphabet [use]".to_string(),
                "Bnd [use]".to_string()
            ]
        );
    }

    #[test]
    fn a_macro_export_is_a_crate_root_declaration() {
        let declarations = declarations(
            "#[macro_export]\nmacro_rules! tile_fits { ($m:expr) => {}; }\n",
            &BTreeSet::new(),
        );
        assert_eq!(declarations.len(), 1);
        assert_eq!(declarations[0].path, ["tile_fits"]);
        assert_eq!(declarations[0].kind, "macro");
        assert!(declarations[0].crate_root);
    }

    #[test]
    fn a_macro_export_escapes_a_private_module_but_not_a_test_module() {
        let declarations = declarations(
            "mod private {\n    #[macro_export]\n    macro_rules! visible { () => {}; }\n}\n\
             #[cfg(test)]\nmod tests {\n    #[macro_export]\n    macro_rules! hidden { () => {}; }\n}\n",
            &BTreeSet::new(),
        );
        assert_eq!(declarations.len(), 1);
        assert_eq!(declarations[0].path, ["visible"]);
        assert!(declarations[0].crate_root);
    }

    /// `#[path]` decides a module's name, so an architecture alternative is
    /// filed under the module it is compiled as.
    #[test]
    fn a_path_attribute_decides_the_module_a_file_belongs_to() {
        let src = Path::new("/w/src");
        let mut modules = Modules::default();
        modules.read(
            src,
            &src.join("isa/mod.rs"),
            "#[cfg(not(target_arch = \"x86_64\"))]\n#[path = \"absent_x86.rs\"]\npub mod x86;\n",
        );
        assert_eq!(
            modules.path_of(src, &src.join("isa/absent_x86.rs")),
            Some(vec!["isa".to_string(), "x86".to_string()])
        );
    }

    /// A module declared without `pub` takes its whole file out of the surface.
    #[test]
    fn a_private_module_is_not_reachable() {
        let src = Path::new("/w/src");
        let mut modules = Modules::default();
        modules.read(src, &src.join("lib.rs"), "mod inner;\npub mod shown;\n");
        assert_eq!(modules.path_of(src, &src.join("inner.rs")), None);
        assert_eq!(
            modules.path_of(src, &src.join("shown.rs")),
            Some(vec!["shown".to_string()])
        );
    }
}
