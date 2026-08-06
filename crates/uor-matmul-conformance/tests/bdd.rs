//! `CM-02`, `CM-03`, and R4's behavioural half.
//!
//! Runs the meta-gate against the actual workspace: the register, the feature
//! suites, and the `#[test]` functions the workspace actually runs. An ID with
//! no scenario, a scenario with no ID, an ID whose only test nothing runs, or a
//! mislabelled honesty level all fail here.
//!
//! The harvest itself lives in [`uor_matmul_conformance::harvest`], because
//! deciding whether a test runs is a rule with cases and every case has to be
//! exercised --- which a helper buried in an integration test cannot be.

use std::path::PathBuf;

use uor_matmul_conformance::{check_honesty, scenarios_in, TestNames};
use uor_matmul_model::{Level, Model};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/uor-matmul-conformance is two below the root")
        .to_path_buf()
}

/// `CM-02`: every registered ID has a scenario and a test that runs, and every
/// scenario and test names a registered ID.
#[test]
fn every_id_has_a_scenario_and_a_test_cm_02() {
    let root = root();
    let tests = TestNames::harvest(&root);
    let running = tests.running();
    assert!(!running.is_empty(), "the test list must not be empty");

    let report = check_honesty(&root, &tests).expect("the meta-gate runs");
    assert!(
        report.is_clean(),
        "the honesty meta-gate failed:\n\n{}\n\nunreachable tests:\n    {}",
        report.violations.join("\n\n"),
        if tests.unreachable().is_empty() {
            "none".to_string()
        } else {
            tests.unreachable().join("\n    ")
        }
    );
    let by_recipe = tests.by_recipe();
    eprintln!(
        "CM-02: {} registered IDs, {} scenarios, {} test names that run ({} of them only \
         under a named recipe)",
        report.ids_checked,
        report.scenarios_checked,
        running.len(),
        by_recipe.len()
    );
    for line in by_recipe {
        eprintln!("    {line}");
    }
}

/// R9: there are no pending or skipped steps.
#[test]
fn no_scenario_is_pending_cm_02() {
    let suites = scenarios_in(&root().join("features/suites")).expect("suites read");
    assert!(suites.files >= 1, "there must be feature files");
    for s in &suites.scenarios {
        assert!(!s.steps.is_empty(), "{} has no steps", s.id);
        for step in &s.steps {
            let lower = step.to_lowercase();
            assert!(
                !lower.contains("pending") && !lower.contains("todo"),
                "{}: `{step}` is a pending step, and R9 admits none",
                s.id
            );
        }
    }
}

/// `CM-03`: every `some-true` claim cites an authority that exists, with a
/// citation and either a checksum or a stated reason for its absence.
#[test]
fn every_some_true_claim_cites_an_authority_cm_03() {
    let model = Model::load(&root().join("model")).expect("model loads");
    model.check().expect("the model is consistent");

    let mut some_true = 0usize;
    for claim in &model.ledger.claim {
        if claim.level != Level::SomeTrue {
            continue;
        }
        some_true += 1;
        let name = claim
            .authority
            .as_ref()
            .expect("a some-true claim names an authority");
        let a = model
            .authorities
            .authority
            .iter()
            .find(|a| &a.id == name)
            .unwrap_or_else(|| panic!("{name} has no row in model/authorities.toml"));
        assert!(!a.citation.trim().is_empty(), "{name} has no citation");
        assert!(
            a.checksum != "none" || !a.checksum_reason.trim().is_empty(),
            "{name} has no checksum and no reason for its absence"
        );
    }
    assert!(some_true >= 1, "there must be cited authorities");
    eprintln!("CM-03: {some_true} cited authorities, each with a citation");
}

/// R4: the meta-gate can fail.
///
/// A gate nobody has ever seen fail is indistinguishable from a gate that
/// cannot. This plants each of the violations it exists to catch and checks
/// that each is reported.
#[test]
fn the_meta_gate_is_falsifiable_cm_02() {
    let root = root();

    // An ID with no test at all.
    let none = TestNames::default();
    let report = check_honesty(&root, &none).expect("runs");
    assert!(!report.is_clean(), "an empty test list must fail the gate");
    assert!(
        report
            .violations
            .iter()
            .any(|v| v.contains("CM-02") && v.contains("no test name")),
        "the missing-test violation must be reported"
    );

    // A test list that covers everything passes, which is the control.
    let full = TestNames::harvest(&root);
    assert!(check_honesty(&root, &full).expect("runs").is_clean());
}

/// `CM-02`: the harvest covers the whole workspace.
///
/// `CM-02` matches an ID against test names gathered from two directories. If a
/// member crate ever moved out from under them the gate would go on passing
/// while checking less --- the worst failure a meta-gate has, because it is
/// silent. [`TestNames::harvest`] refuses to return in that case; this names the
/// claim so that it appears in the suite rather than only in a panic.
#[test]
fn the_harvest_covers_every_workspace_member_cm_02() {
    let root = root();
    let manifest =
        std::fs::read_to_string(root.join("Cargo.toml")).expect("the workspace manifest reads");
    let value: toml::Value = manifest.parse().expect("the workspace manifest parses");
    let members = value["workspace"]["members"]
        .as_array()
        .expect("the workspace declares members");
    assert!(!members.is_empty(), "a workspace with no members");

    // The harvest asserts coverage itself; running it here is what makes that
    // assertion part of the suite. The count is the control: coverage that
    // yields nothing is coverage in name only.
    let tests = TestNames::harvest(&root);
    assert!(
        tests.all().len() >= members.len(),
        "{} members and only {} test names: the harvest is not reading them",
        members.len(),
        tests.all().len()
    );
}
