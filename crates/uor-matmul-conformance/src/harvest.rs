//! Which `#[test]` functions the workspace actually *runs*.
//!
//! `CM-02` says every registered ID has a test, and it decides that by matching
//! the ID's slug against a harvested name. A name is not a run. Three ways of
//! keeping the name while losing the run all used to pass silently: an
//! `#[ignore]` between the attribute and the `fn`, a `#[cfg(feature = "...")]`
//! on a feature nothing turns on, and a crate moved out from under the scan.
//! Each leaves the register saying "checked" and the machine checking nothing,
//! which is precisely the vacuity R4 exists to prevent.
//!
//! So the scan classifies rather than collects. A test is [`Reached::Default`]
//! when a plain `cargo test --workspace` runs it, [`Reached::Recipe`] when only
//! a named `just` recipe does --- and that redirection is *verified against the
//! Justfile*, not taken from the `#[ignore]` reason's prose --- and
//! [`Reached::Never`] when nothing this repository defines runs it. Only the
//! first two discharge an ID.
//!
//! The alternative was to shell out to `cargo test -- --list`, which needs no
//! parser and answers the question exactly. It is not available: this code runs
//! *inside* `cargo test`, and a nested invocation blocks on the target
//! directory lock. So the scan reads source text, and every rule below is a
//! rule about text that the compiler would apply to the same item.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// The directories under the repository root that the scan reads.
///
/// A workspace member outside them is a hole in `CM-02`, so
/// [`TestNames::harvest`] refuses to return rather than quietly cover less than
/// the workspace.
pub const SCAN_ROOTS: &[&str] = &["crates", "xtask"];

/// What runs a `#[test]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Reached {
    /// `cargo test --workspace`, at default features, on any host. This is the
    /// pool `just vv` exercises.
    Default,
    /// Only the named `just` recipe --- one whose command line was read and
    /// found to satisfy every gate on this test at once: the features its
    /// `#[cfg]` needs, and `--ignored` if it is ignored.
    Recipe(String),
    /// Nothing the repository defines. The string says which rule decided it.
    Never(String),
}

/// Every `#[test]` in the workspace, and what runs it.
#[derive(Clone, Debug, Default)]
pub struct TestNames {
    /// Bare function name to how it is reached. A name may occur in several
    /// files; the most reachable classification wins, because one running test
    /// is enough to discharge the ID its name claims.
    reached: BTreeMap<String, Reached>,
    /// Where each name was found, for a failure message that can be acted on.
    sites: BTreeMap<String, Vec<String>>,
}

impl TestNames {
    /// The names a `cargo test` invocation this repository defines will run.
    ///
    /// This is the set `CM-02` matches an ID's slug against.
    pub fn running(&self) -> BTreeSet<String> {
        self.reached
            .iter()
            .filter(|(_, r)| !matches!(r, Reached::Never(_)))
            .map(|(n, _)| n.clone())
            .collect()
    }

    /// Every name the scan found, run or not.
    ///
    /// `CM-02`'s third direction --- an ID that exists only in a test name ---
    /// must read this one. A test that never runs still *claims* the ID its
    /// name ends in, and a claim made by a name is exactly what that check is
    /// for; filtering it out would have traded one hole for another.
    pub fn all(&self) -> BTreeSet<String> {
        self.reached.keys().cloned().collect()
    }

    /// Every name that nothing runs, with the rule that decided it and where
    /// it lives.
    ///
    /// A `CM-02` failure whose cause is a gating attribute reads as "no test
    /// name ends in `cx_08`" until this is printed beside it, at which point it
    /// reads as "the test is there and nothing runs it" --- a different defect
    /// with a different fix.
    pub fn unreachable(&self) -> Vec<String> {
        self.reached
            .iter()
            .filter_map(|(name, r)| match r {
                Reached::Never(why) => Some(format!(
                    "{name} ({}): {why}",
                    self.sites
                        .get(name)
                        .map(|s| s.join(", "))
                        .unwrap_or_default()
                )),
                _ => None,
            })
            .collect()
    }

    /// Every unreachable test whose name ends in `slug`, with its reason.
    ///
    /// This is what turns a `CM-02` failure from "write a test" into "the test
    /// exists; nothing runs it".
    pub fn stalled(&self, slug: &str) -> Vec<String> {
        self.unreachable()
            .into_iter()
            .filter(|entry| {
                entry
                    .split_once(' ')
                    .map(|(name, _)| name)
                    .unwrap_or(entry)
                    .ends_with(slug)
            })
            .collect()
    }

    /// Names that run only under a named recipe, with the recipe.
    pub fn by_recipe(&self) -> Vec<String> {
        self.reached
            .iter()
            .filter_map(|(name, r)| match r {
                Reached::Recipe(recipe) => Some(format!("{name}: just {recipe}")),
                _ => None,
            })
            .collect()
    }

    /// Scan the workspace.
    ///
    /// # Panics
    ///
    /// If a workspace member lies outside [`SCAN_ROOTS`], or a scan root does
    /// not exist. Either makes the harvest cover less than the workspace, and a
    /// harvest that covers less than it claims turns `CM-02` into a check on
    /// whichever crates happen to be in the right place.
    pub fn harvest(root: &Path) -> Self {
        let roots = scan_roots(root);
        let recipes = Recipes::read(root);
        let mut out = Self::default();
        for dir in &roots {
            let mut stack = vec![dir.clone()];
            while let Some(dir) = stack.pop() {
                let Ok(entries) = std::fs::read_dir(&dir) else {
                    continue;
                };
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        if path.file_name().is_some_and(|n| n == "target") {
                            continue;
                        }
                        stack.push(path);
                    } else if path.extension().is_some_and(|e| e == "rs") {
                        out.read_file(root, &path, &recipes);
                    }
                }
            }
        }
        out
    }

    fn read_file(&mut self, root: &Path, path: &Path, recipes: &Recipes) {
        let Ok(text) = std::fs::read_to_string(path) else {
            return;
        };
        let Some(target) = CargoTarget::of(path) else {
            return;
        };
        let features = CrateFeatures::of(&target.manifest_dir);
        let rel = path
            .strip_prefix(root)
            .unwrap_or(path)
            .display()
            .to_string();
        for found in tests_in(&text) {
            let reached = classify(&found, &features, &target, recipes);
            self.sites.entry(found.name.clone()).or_default().push(
                if matches!(reached, Reached::Never(_)) {
                    format!("{rel}:{}", found.line)
                } else {
                    rel.clone()
                },
            );
            match self.reached.entry(found.name) {
                std::collections::btree_map::Entry::Vacant(v) => {
                    v.insert(reached);
                }
                std::collections::btree_map::Entry::Occupied(mut o) => {
                    // The most reachable wins: two tests may share a name in
                    // two crates, and one of them running is one run.
                    if rank(&reached) > rank(o.get()) {
                        o.insert(reached);
                    }
                }
            }
        }
    }
}

/// How much a classification counts, for picking between two tests of one name.
fn rank(r: &Reached) -> u8 {
    match r {
        Reached::Default => 2,
        Reached::Recipe(_) => 1,
        Reached::Never(_) => 0,
    }
}

/// The scan roots, resolved and checked against the workspace's own membership.
///
/// # Panics
///
/// If a member directory is not covered. `crates/*` and `xtask` are the roots
/// because those are the members; a new member elsewhere would silently stop
/// contributing test names, and every ID discharged only from there would go on
/// reading as checked.
fn scan_roots(root: &Path) -> Vec<PathBuf> {
    let roots: Vec<PathBuf> = SCAN_ROOTS.iter().map(|r| root.join(r)).collect();
    for (name, dir) in SCAN_ROOTS.iter().zip(&roots) {
        assert!(
            dir.is_dir(),
            "CM-02: the scan root `{name}` does not exist at {}, so every test under it \
             is invisible to the gate and every ID it discharges passes unchecked",
            dir.display()
        );
    }
    let manifest = root.join("Cargo.toml");
    let text = std::fs::read_to_string(&manifest)
        .unwrap_or_else(|e| panic!("CM-02: the workspace manifest {}: {e}", manifest.display()));
    let value: toml::Value = text
        .parse()
        .unwrap_or_else(|e| panic!("CM-02: the workspace manifest does not parse: {e}"));
    let members = value
        .get("workspace")
        .and_then(|w| w.get("members"))
        .and_then(|m| m.as_array())
        .unwrap_or_else(|| {
            panic!(
                "CM-02: {} declares no workspace members",
                manifest.display()
            )
        });
    assert!(
        !members.is_empty(),
        "CM-02: {} declares an empty member list",
        manifest.display()
    );
    for member in members {
        let pattern = member
            .as_str()
            .unwrap_or_else(|| panic!("CM-02: a workspace member is not a string: {member}"));
        // A member may be a glob (`crates/*`). What has to be covered is the
        // literal head of the pattern --- everything before the first component
        // that globs --- because that is the directory the members live in.
        let head: PathBuf = pattern
            .split('/')
            .take_while(|c| !c.contains('*') && !c.contains('?'))
            .collect();
        assert!(
            !head.as_os_str().is_empty(),
            "CM-02: the workspace member pattern `{pattern}` globs from the root, so no \
             fixed directory covers it"
        );
        let dir = root.join(&head);
        assert!(
            roots.iter().any(|r| dir == *r || dir.starts_with(r)),
            "CM-02: the workspace member `{pattern}` lives at {}, which no scan root covers. \
             The roots are {:?}. A `#[test]` in that member would satisfy nothing, and every \
             ID it is the only test for would pass this gate while running nowhere. Add the \
             directory to `SCAN_ROOTS`.",
            head.display(),
            SCAN_ROOTS
        );
    }
    roots
}

/// A `#[test]` the scan found, with the attributes the compiler applies to it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoundTest {
    /// The bare function name.
    pub name: String,
    /// 1-based line of the `fn`.
    pub line: usize,
    /// Every attribute on the item, in source order, whether written above the
    /// `#[test]` or between it and the `fn`. The compiler does not care which
    /// side an attribute sits on, so neither does this: reading only the lines
    /// *after* `#[test]` would have missed every `#[cfg(feature = ...)]` in
    /// this workspace, all of which are written above it.
    pub attributes: Vec<String>,
    /// The `cfg` predicate of an enclosing block, if that block's `cfg` is
    /// anything other than `test`.
    pub enclosing_cfg: Option<String>,
}

/// Every `#[test]` in one source file.
///
/// The recognised shape is the one this workspace writes: `#[test]` alone on
/// its line, then attributes and blank lines, then a line beginning `fn `. A
/// doc comment between the attribute and the `fn` ends the item, because a doc
/// comment there is not an attribute run --- and the workspace's convention,
/// which `cortex_m.rs` follows, is to put doc comments above `#[test]`.
pub fn tests_in(text: &str) -> Vec<FoundTest> {
    let lines: Vec<&str> = text.lines().collect();
    let mut out = Vec::new();
    // Blocks opened under a `cfg` that is not `test`: `(brace depth inside the
    // block, the predicate)`. A `#[cfg(feature = "x")] mod tests { .. }` gates
    // every test in it exactly as if the attribute were on each one.
    let mut gated: Vec<(i32, String)> = Vec::new();
    let mut depth = 0i32;
    // The contiguous attribute run immediately above the current line.
    let mut run: Vec<String> = Vec::new();

    for (i, raw) in lines.iter().enumerate() {
        let line = raw.trim();
        if line.starts_with("#[") {
            run.push(line.to_string());
        } else if line.is_empty() {
            // A blank line inside an attribute run is tolerated by the
            // compiler and by this scan.
        } else {
            if run.iter().any(|a| a == "#[test]") {
                if let Some(name) = fn_name(line) {
                    out.push(FoundTest {
                        name,
                        line: i + 1,
                        attributes: run.clone(),
                        enclosing_cfg: gated.last().map(|(_, p)| p.clone()),
                    });
                }
            }
            if let Some(pred) = block_cfg(&run) {
                // Recorded only when the block opens on this line; a `mod x;`
                // declaration opens nothing here.
                if line.contains('{') {
                    gated.push((depth + 1, pred));
                }
            }
            run.clear();
        }
        // Braces outside comments only. A doc comment writes `Bnd<{ N }>` and a
        // prose brace that never closes would leave a gated block open over the
        // rest of the file, disqualifying tests nothing gates.
        let code = raw.split("//").next().unwrap_or("");
        depth += code.matches('{').count() as i32;
        depth -= code.matches('}').count() as i32;
        while gated.last().is_some_and(|(d, _)| *d > depth) {
            gated.pop();
        }
    }
    out
}

/// The `cfg` predicate an attribute run puts on the block it opens, unless that
/// predicate is exactly `test`.
///
/// `#[cfg(test)] mod tests` is how every unit test in this workspace is
/// written, and it is the one predicate that is *true* in the build the gate
/// cares about, so it gates nothing.
fn block_cfg(run: &[String]) -> Option<String> {
    if run.iter().any(|a| a == "#[test]") {
        return None;
    }
    run.iter()
        .filter_map(|a| cfg_predicate(a))
        .find(|p| p.trim() != "test")
}

/// The function name a `fn ` line declares.
fn fn_name(line: &str) -> Option<String> {
    let rest = line.strip_prefix("fn ")?;
    let name: String = rest
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    (!name.is_empty()).then_some(name)
}

/// The inside of a `#[cfg(...)]`, if the attribute is one.
fn cfg_predicate(attr: &str) -> Option<String> {
    let inner = attr.strip_prefix("#[")?.strip_suffix(']')?.trim();
    let rest = inner.strip_prefix("cfg")?;
    let rest = rest.trim();
    Some(rest.strip_prefix('(')?.strip_suffix(')')?.to_string())
}

/// Decide what runs a found test.
///
/// One rule, applied twice. A `#[cfg]` and an `#[ignore]` both say "not in the
/// default run", and both are admissible on the same condition: some invocation
/// the repository *writes down* satisfies them. So the search is for a recipe
/// that both enables the features the `cfg` needs and passes `--ignored` if the
/// test is ignored --- one invocation, satisfying every gate at once, because
/// two recipes each satisfying half of them run the test never.
fn classify(
    found: &FoundTest,
    features: &CrateFeatures,
    target: &CargoTarget,
    recipes: &Recipes,
) -> Reached {
    let mut predicates: Vec<String> = found.enclosing_cfg.iter().cloned().collect();
    predicates.extend(found.attributes.iter().filter_map(|a| cfg_predicate(a)));
    let ignored = found.attributes.iter().find(|a| is_ignore(a));

    // A predicate that is neither true nor false at default features is a
    // property of the machine, and no invocation settles it.
    if let Some(pred) = predicates
        .iter()
        .find(|p| reach(p, &features.default) == Reach::Host)
    {
        return Reached::Never(format!(
            "`#[cfg({pred})]` holds on some hosts and not others, so a green run on one \
             machine says nothing about the claim on another"
        ));
    }

    let compiled = |enabled: &BTreeSet<String>| {
        predicates
            .iter()
            .all(|p| reach(p, enabled) == Reach::Always)
    };
    if compiled(&features.default) && ignored.is_none() {
        return Reached::Default;
    }

    for run in recipes.test_runs(target) {
        let enabled = run.features.applied(features);
        if compiled(&enabled) && (ignored.is_none() || run.ignored) {
            return Reached::Recipe(run.name);
        }
    }

    // Nothing runs it. Which gate stopped it decides the message, because the
    // two have different fixes: one is a feature nothing turns on, the other a
    // recipe nobody wrote.
    if let Some(pred) = predicates
        .iter()
        .find(|p| reach(p, &features.default) != Reach::Always)
    {
        return Reached::Never(format!(
            "`#[cfg({pred})]` is false for `{}` at its default features ({}), and no Justfile \
             recipe runs `cargo test` for that package with the feature on, so nothing \
             compiles it",
            target.package,
            list(&features.default)
        ));
    }
    let attr = ignored.map_or("#[ignore]", |a| a.as_str());
    Reached::Never(format!(
        "`{attr}` keeps it out of `cargo test --workspace`, and no Justfile recipe runs \
         `cargo test -p {} {} -- --ignored`, so nothing runs it at all",
        target.package, target.selector
    ))
}

/// A feature set, for a message.
fn list(features: &BTreeSet<String>) -> String {
    if features.is_empty() {
        "none".to_string()
    } else {
        features.iter().cloned().collect::<Vec<_>>().join(", ")
    }
}

/// Is this attribute an `ignore`, conditional or not?
///
/// `#[cfg_attr(<c>, ignore)]` counts, because whether it ignores depends on a
/// condition this scan cannot settle, and a gate that guesses in the permissive
/// direction is the vacuity it exists to remove.
fn is_ignore(attr: &str) -> bool {
    attr == "#[ignore]"
        || attr.starts_with("#[ignore =")
        || attr.starts_with("#[ignore(")
        || (attr.starts_with("#[cfg_attr(") && attr.contains("ignore"))
}

/// Whether a `cfg` predicate holds wherever the suite runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Reach {
    /// True in every build the repository's own recipes make.
    Always,
    /// False in all of them.
    Never,
    /// True in some and false in others --- a property of the machine, not of
    /// the workspace.
    Host,
}

/// Evaluate a `cfg` predicate against a crate's default feature set.
///
/// Three-valued on purpose. Two-valued logic gets `not(...)` wrong: if an
/// unrecognised predicate were simply false then `#[cfg(target_arch = "x86_64")]`
/// and `#[cfg(not(target_arch = "x86_64"))]` would both be "does not run", and
/// one of them always does.
fn reach(pred: &str, features: &BTreeSet<String>) -> Reach {
    let p = pred.trim();
    if p == "test" {
        return Reach::Always;
    }
    if let Some(inner) = call(p, "not") {
        return match reach(inner, features) {
            Reach::Always => Reach::Never,
            Reach::Never => Reach::Always,
            Reach::Host => Reach::Host,
        };
    }
    if let Some(inner) = call(p, "all") {
        let parts = top_level_commas(inner);
        if parts.iter().any(|q| reach(q, features) == Reach::Never) {
            return Reach::Never;
        }
        if parts.iter().any(|q| reach(q, features) == Reach::Host) {
            return Reach::Host;
        }
        return Reach::Always;
    }
    if let Some(inner) = call(p, "any") {
        let parts = top_level_commas(inner);
        if parts.iter().any(|q| reach(q, features) == Reach::Always) {
            return Reach::Always;
        }
        if parts.iter().any(|q| reach(q, features) == Reach::Host) {
            return Reach::Host;
        }
        return Reach::Never;
    }
    if let Some(rest) = p.strip_prefix("feature") {
        if let Some(name) = rest.trim().strip_prefix('=') {
            let name = name.trim().trim_matches('"');
            return if features.contains(name) {
                Reach::Always
            } else {
                Reach::Never
            };
        }
    }
    Reach::Host
}

/// The argument of `name(...)`, if `p` is that call.
fn call<'p>(p: &'p str, name: &str) -> Option<&'p str> {
    let rest = p.strip_prefix(name)?.trim_start();
    rest.strip_prefix('(')?.strip_suffix(')')
}

/// Split on commas that are not inside parentheses.
fn top_level_commas(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    for (i, c) in s.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth -= 1,
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

/// The cargo target a source file compiles into, and how to select it.
#[derive(Clone, Debug)]
pub struct CargoTarget {
    /// The package name from the owning `Cargo.toml`.
    pub package: String,
    /// The directory holding that manifest.
    pub manifest_dir: PathBuf,
    /// The `cargo test` flag that selects this target: `--test massive`,
    /// `--lib`, `--bench scaling`.
    pub selector: String,
}

impl CargoTarget {
    /// Resolve the target from a source path, by walking up to its manifest.
    pub fn of(path: &Path) -> Option<Self> {
        let mut dir = path.parent()?;
        loop {
            if dir.join("Cargo.toml").is_file() {
                break;
            }
            dir = dir.parent()?;
        }
        let manifest = std::fs::read_to_string(dir.join("Cargo.toml")).ok()?;
        let value: toml::Value = manifest.parse().ok()?;
        let package = value
            .get("package")?
            .get("name")?
            .as_str()?
            .trim_matches('"')
            .to_string();
        let rel = path.strip_prefix(dir).ok()?;
        let mut parts = rel.components();
        let first = parts.next()?.as_os_str().to_string_lossy().to_string();
        let stem = path.file_stem()?.to_string_lossy().to_string();
        // A `tests/x/helpers.rs` belongs to the `tests/x` target, not to a
        // target named `helpers`; only the directly-nested file names one.
        let direct = rel.components().count() == 2;
        let selector = match first.as_str() {
            "src" => {
                if stem == "main" {
                    format!("--bin {package}")
                } else {
                    "--lib".to_string()
                }
            }
            "tests" if direct => format!("--test {stem}"),
            "benches" if direct => format!("--bench {stem}"),
            "examples" if direct => format!("--example {stem}"),
            _ => return None,
        };
        Some(Self {
            package,
            manifest_dir: dir.to_path_buf(),
            selector,
        })
    }
}

/// A crate's feature sets, read from its manifest.
///
/// `#[cfg(feature = "ref-ndarray")]` on a test is not evidence that the test is
/// skipped: in this workspace `ref-ndarray` is a *default* feature and the test
/// runs. `ref-gemm` is not, and its test does not. Only the manifest can tell
/// the two apart, so the manifest is what is read.
#[derive(Clone, Debug, Default)]
pub struct CrateFeatures {
    /// What `cargo test -p <crate>` turns on, resolved through the graph.
    pub default: BTreeSet<String>,
    /// Every feature the crate declares, which is what `--all-features` means.
    pub all: BTreeSet<String>,
}

impl CrateFeatures {
    /// Read them from the manifest in `manifest_dir`.
    pub fn of(manifest_dir: &Path) -> Self {
        let mut out = Self::default();
        let Ok(text) = std::fs::read_to_string(manifest_dir.join("Cargo.toml")) else {
            return out;
        };
        let Ok(value) = text.parse::<toml::Value>() else {
            return out;
        };
        let Some(table) = value.get("features").and_then(|f| f.as_table()) else {
            return out;
        };
        out.all = table.keys().map(|k| k.to_string()).collect();
        let mut queue: Vec<String> = table
            .get("default")
            .and_then(|d| d.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        while let Some(name) = queue.pop() {
            // `dep:x` and `x/y` enable a dependency, not a feature of this crate.
            if name.starts_with("dep:") || name.contains('/') {
                continue;
            }
            if !out.default.insert(name.clone()) {
                continue;
            }
            if let Some(children) = table.get(&name).and_then(|c| c.as_array()) {
                queue.extend(
                    children
                        .iter()
                        .filter_map(|v| v.as_str().map(str::to_string)),
                );
            }
        }
        out
    }
}

/// What a `cargo test` invocation turns on beyond the default set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Enables {
    /// Just the defaults.
    Default,
    /// `--all-features`.
    All,
    /// `--features a,b`, on top of the defaults unless `--no-default-features`.
    Named(BTreeSet<String>),
}

impl Enables {
    /// The feature set the invocation actually compiles with.
    fn applied(&self, features: &CrateFeatures) -> BTreeSet<String> {
        match self {
            Self::Default => features.default.clone(),
            Self::All => features.all.clone(),
            Self::Named(named) => features.default.union(named).cloned().collect(),
        }
    }
}

/// One `cargo test` invocation a recipe makes.
#[derive(Clone, Debug)]
pub struct TestRun {
    /// The recipe's name, for the report.
    pub name: String,
    /// What it turns on.
    pub features: Enables,
    /// Whether it passes `--ignored` to the harness.
    pub ignored: bool,
}

/// The `just` recipes, each as the list of commands it runs.
///
/// Commands, not one joined body. A recipe's lines are separate processes, and
/// reading them as one string lets flags migrate between them: `just features`
/// runs `cargo check --workspace --all-features` and then
/// `cargo test -p uor-matmul-codec --all-features`, and a joined body says that
/// recipe runs the whole workspace's tests with every feature on. It does not,
/// and believing so made a test nothing runs look reachable.
#[derive(Clone, Debug, Default)]
pub struct Recipes {
    entries: Vec<(String, Vec<String>)>,
}

impl Recipes {
    /// Read the Justfile at the repository root.
    pub fn read(root: &Path) -> Self {
        let Ok(text) = std::fs::read_to_string(root.join("Justfile")) else {
            return Self::default();
        };
        Self::parse(&text)
    }

    /// Parse a Justfile's recipe names and commands.
    pub fn parse(text: &str) -> Self {
        let mut entries: Vec<(String, Vec<String>)> = Vec::new();
        let mut current: Option<(String, Vec<String>)> = None;
        // A line ending in a backslash continues into the next one.
        let mut continued = false;
        for line in text.lines() {
            let indented = line.starts_with(' ') || line.starts_with('\t');
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if indented {
                if let Some((_, commands)) = current.as_mut() {
                    let text = trimmed.trim_end_matches('\\').trim_end();
                    match commands.last_mut().filter(|_| continued) {
                        Some(last) => {
                            last.push(' ');
                            last.push_str(text);
                        }
                        None => commands.push(text.to_string()),
                    }
                }
                continued = trimmed.ends_with('\\');
                continue;
            }
            continued = false;
            if let Some(done) = current.take() {
                entries.push(done);
            }
            // A recipe header is `name:` or `name: deps`, at column zero.
            let Some((head, _)) = trimmed.split_once(':') else {
                continue;
            };
            let name = head.split_whitespace().next().unwrap_or("").trim();
            if !name.is_empty()
                && name
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
            {
                current = Some((name.to_string(), Vec::new()));
            }
        }
        if let Some(done) = current.take() {
            entries.push(done);
        }
        Self { entries }
    }

    /// Every recipe that runs `target` under `cargo test`, with what it enables.
    ///
    /// A recipe qualifies when it runs `cargo test`, names the package, and
    /// either names this target or names no target at all --- `cargo test -p x`
    /// with no selector runs every target `x` has. Anything looser would let a
    /// recipe that merely mentions the crate stand in for one that runs the
    /// test; `cargo check --all-features`, which this workspace does run, is
    /// deliberately not enough, because compiling a test is not running it.
    pub fn test_runs(&self, target: &CargoTarget) -> Vec<TestRun> {
        const SELECTORS: &[&str] = &["--test ", "--bench ", "--example ", "--lib", "--bin "];
        self.entries
            .iter()
            .flat_map(|(name, commands)| {
                commands.iter().filter_map(move |command| {
                    let runs = command.contains("cargo test")
                        && (command.contains(&format!("-p {}", target.package))
                            || command.contains(&format!("--package {}", target.package))
                            || command.contains("--workspace"))
                        && (command.contains(&target.selector)
                            || !SELECTORS.iter().any(|s| command.contains(s)));
                    runs.then(|| TestRun {
                        name: name.clone(),
                        features: enables(command),
                        ignored: command.contains("--ignored"),
                    })
                })
            })
            .collect()
    }
}

/// What a recipe's command line turns on.
fn enables(body: &str) -> Enables {
    if body.contains("--all-features") {
        return Enables::All;
    }
    let mut named = BTreeSet::new();
    for marker in ["--features ", "--features="] {
        let mut rest = body;
        while let Some(at) = rest.find(marker) {
            rest = &rest[at + marker.len()..];
            let list = rest.split_whitespace().next().unwrap_or("");
            for name in list.split(',') {
                let name = name.trim().trim_matches('"');
                if !name.is_empty() {
                    named.insert(name.to_string());
                }
            }
        }
    }
    if named.is_empty() {
        Enables::Default
    } else {
        Enables::Named(named)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Written apart so that this file's own fixtures are not harvested as
    /// tests of the workspace: the scan reads source text, and a fixture
    /// containing a bare `#[test]` line would be indistinguishable from one.
    const TEST: &str = "#[test]";

    fn fixture(body: &str) -> String {
        body.replace("@test", TEST)
    }

    fn only(text: &str) -> FoundTest {
        let found = tests_in(text);
        assert_eq!(found.len(), 1, "one test in the fixture, got {found:?}");
        found.into_iter().next().expect("just asserted")
    }

    fn features(default: &[&str]) -> CrateFeatures {
        let mut all: BTreeSet<String> = default.iter().map(|s| (*s).to_string()).collect();
        all.insert("ref-gemm".to_string());
        CrateFeatures {
            default: default.iter().map(|s| (*s).to_string()).collect(),
            all,
        }
    }

    fn target() -> CargoTarget {
        CargoTarget {
            package: "uor-matmul-validate".to_string(),
            manifest_dir: PathBuf::from("/nonexistent"),
            selector: "--test massive".to_string(),
        }
    }

    /// The attribute run is read on both sides of `#[test]`, because the
    /// compiler applies both to the item.
    #[test]
    fn attributes_are_read_above_and_below_the_test_attribute() {
        let found = only(&fixture(
            "#[cfg(feature = \"ref-gemm\")]\n@test\n#[ignore]\nfn a_fixture_zz_90() {}\n",
        ));
        assert_eq!(found.name, "a_fixture_zz_90");
        assert_eq!(
            found.attributes,
            vec![
                "#[cfg(feature = \"ref-gemm\")]".to_string(),
                TEST.to_string(),
                "#[ignore]".to_string(),
            ]
        );
    }

    /// An `#[ignore]` with no recipe behind it means the test does not run.
    #[test]
    fn an_ignored_test_with_no_recipe_runs_nowhere() {
        let found = only(&fixture("@test\n#[ignore]\nfn a_fixture_zz_91() {}\n"));
        let verdict = classify(&found, &features(&[]), &target(), &Recipes::default());
        match verdict {
            Reached::Never(why) => assert!(
                why.contains("--ignored"),
                "the message must name what is missing: {why}"
            ),
            other => panic!("an ignored test with no recipe must not count: {other:?}"),
        }
    }

    /// An `#[ignore]` a recipe runs is admissible, and the recipe is the one
    /// the Justfile actually spells --- not the one the reason claims.
    #[test]
    fn an_ignored_test_a_recipe_runs_is_admissible() {
        let recipes = Recipes::parse(
            "massive:\n    cargo test --release -p uor-matmul-validate --test massive -- \\\n        --ignored --nocapture\n",
        );
        let found = only(&fixture(
            "@test\n#[ignore = \"claims `just nowhere`\"]\nfn a_fixture_zz_92() {}\n",
        ));
        assert_eq!(
            classify(&found, &features(&[]), &target(), &recipes),
            Reached::Recipe("massive".to_string())
        );

        // The same test, against a recipe that runs the crate but not with
        // `--ignored`, is not run by it.
        let sloppy =
            Recipes::parse("test:\n    cargo test -p uor-matmul-validate --test massive\n");
        assert!(matches!(
            classify(&found, &features(&[]), &target(), &sloppy),
            Reached::Never(_)
        ));
    }

    /// A feature gate is decided by the crate's default set, not by the mere
    /// presence of the word `feature`.
    #[test]
    fn a_feature_gate_is_decided_by_the_default_set() {
        let on = only(&fixture(
            "#[cfg(feature = \"ref-ndarray\")]\n@test\nfn a_fixture_zz_93() {}\n",
        ));
        assert_eq!(
            classify(
                &on,
                &features(&["ref-ndarray"]),
                &target(),
                &Recipes::default()
            ),
            Reached::Default
        );
        let off = only(&fixture(
            "#[cfg(feature = \"ref-gemm\")]\n@test\nfn a_fixture_zz_94() {}\n",
        ));
        match classify(
            &off,
            &features(&["ref-ndarray"]),
            &target(),
            &Recipes::default(),
        ) {
            Reached::Never(why) => assert!(why.contains("ref-gemm"), "{why}"),
            other => panic!("a non-default feature gate must not count: {other:?}"),
        }
    }

    /// A recipe that turns the feature on is what makes a non-default gate
    /// admissible --- and it has to *run* the tests, not merely compile them.
    #[test]
    fn a_recipe_that_enables_the_feature_admits_the_gate() {
        let found = only(&fixture(
            "#[cfg(feature = \"ref-gemm\")]\n@test\nfn a_fixture_zz_99() {}\n",
        ));
        let f = features(&["ref-ndarray"]);

        let named =
            Recipes::parse("oracles:\n    cargo test -p uor-matmul-validate --features ref-gemm\n");
        assert_eq!(
            classify(&found, &f, &target(), &named),
            Reached::Recipe("oracles".to_string())
        );

        let all =
            Recipes::parse("oracles:\n    cargo test -p uor-matmul-validate --all-features\n");
        assert_eq!(
            classify(&found, &f, &target(), &all),
            Reached::Recipe("oracles".to_string())
        );

        // `cargo check` compiles the test and runs nothing --- and the recipe's
        // *other* command does not lend it its flags.
        let checked = Recipes::parse(
            "features:\n    cargo check --workspace --all-features --all-targets\n\
             \x20   cargo test -p uor-matmul-codec --all-features\n",
        );
        assert!(matches!(
            classify(&found, &f, &target(), &checked),
            Reached::Never(_)
        ));

        // A run of a different target does not run this one.
        let elsewhere =
            Recipes::parse("codec:\n    cargo test -p uor-matmul-validate --lib --all-features\n");
        assert!(matches!(
            classify(&found, &f, &target(), &elsewhere),
            Reached::Never(_)
        ));
    }

    /// Two recipes each satisfying half the gates run the test never, so the
    /// search is for one invocation that satisfies them all.
    #[test]
    fn one_invocation_must_satisfy_every_gate_at_once() {
        let found = only(&fixture(
            "#[cfg(feature = \"ref-gemm\")]\n@test\n#[ignore]\nfn a_fixture_zz_89() {}\n",
        ));
        let f = features(&["ref-ndarray"]);

        let split = Recipes::parse(
            "a:\n    cargo test -p uor-matmul-validate --test massive --all-features\n\
             b:\n    cargo test -p uor-matmul-validate --test massive -- --ignored\n",
        );
        assert!(matches!(
            classify(&found, &f, &target(), &split),
            Reached::Never(_)
        ));

        let together = Recipes::parse(
            "both:\n    cargo test -p uor-matmul-validate --test massive --all-features -- --ignored\n",
        );
        assert_eq!(
            classify(&found, &f, &target(), &together),
            Reached::Recipe("both".to_string())
        );
    }

    /// `#[cfg(test)]` gates nothing: it is how every unit test here is written.
    #[test]
    fn the_test_cfg_gates_nothing() {
        let text =
            fixture("#[cfg(test)]\nmod tests {\n    @test\n    fn a_fixture_zz_95() {}\n}\n");
        let found = only(&text);
        assert_eq!(found.enclosing_cfg, None);
        assert_eq!(
            classify(&found, &features(&[]), &target(), &Recipes::default()),
            Reached::Default
        );
    }

    /// A `cfg` on the enclosing module gates every test inside it, and stops
    /// gating where the module closes.
    #[test]
    fn an_enclosing_module_cfg_gates_the_tests_inside_it() {
        let text = fixture(
            "#[cfg(feature = \"ref-gemm\")]\nmod gated {\n    @test\n    fn a_fixture_zz_96() {}\n}\n@test\nfn a_fixture_zz_97() {}\n",
        );
        let found = tests_in(&text);
        assert_eq!(found.len(), 2);
        assert_eq!(
            found[0].enclosing_cfg.as_deref(),
            Some("feature = \"ref-gemm\"")
        );
        assert_eq!(found[1].enclosing_cfg, None);
        assert!(matches!(
            classify(&found[0], &features(&[]), &target(), &Recipes::default()),
            Reached::Never(_)
        ));
    }

    /// A brace in a comment does not close a gated block, or a doc comment
    /// would take every test below it out of the run.
    #[test]
    fn a_brace_in_a_comment_does_not_move_the_block_depth() {
        let text = fixture(
            "#[cfg(feature = \"ref-gemm\")]\nmod gated {\n    /// writes `Bnd<{ N }>`\n    @test\n    fn a_fixture_zz_88() {}\n}\n@test\nfn a_fixture_zz_87() {}\n",
        );
        let found = tests_in(&text);
        assert_eq!(found.len(), 2, "{found:?}");
        assert_eq!(
            found[0].enclosing_cfg.as_deref(),
            Some("feature = \"ref-gemm\"")
        );
        assert_eq!(
            found[1].enclosing_cfg, None,
            "the block closed where its brace did"
        );
    }

    /// The three-valued logic: an unrecognised predicate and its negation
    /// cannot both mean "does not run".
    #[test]
    fn a_host_predicate_and_its_negation_are_both_conditional() {
        let f = features(&["std"]).default;
        assert_eq!(reach("target_arch = \"x86_64\"", &f), Reach::Host);
        assert_eq!(reach("not(target_arch = \"x86_64\")", &f), Reach::Host);
        assert_eq!(reach("all(test, feature = \"std\")", &f), Reach::Always);
        assert_eq!(reach("all(test, feature = \"gone\")", &f), Reach::Never);
        assert_eq!(reach("any(feature = \"gone\", test)", &f), Reach::Always);
        assert_eq!(reach("not(feature = \"gone\")", &f), Reach::Always);
    }

    /// A doc comment between the attribute and the `fn` ends the item, which is
    /// why the house convention puts doc comments above `#[test]`.
    #[test]
    fn a_comment_between_the_attribute_and_the_fn_ends_the_run() {
        let text = fixture("@test\n/// a doc comment\nfn a_fixture_zz_98() {}\n");
        assert!(tests_in(&text).is_empty());
    }
}
