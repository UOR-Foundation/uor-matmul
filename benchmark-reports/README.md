# Committed comparison benchmark reports

Each numeric directory is the complete portable bundle produced by that
GitHub Actions run: Markdown, self-contained HTML graphs, run context, the
benchmark log, and the raw Criterion estimates used to render the report.

`current` is a relative symlink to the newest run committed to the repository.
It is deliberately a link rather than a copied report, so the current view and
the immutable run history cannot drift.

Current committed run: [31519209588/REPORT.md](31519209588/REPORT.md), with the
self-contained graphs in [31519209588/index.html](31519209588/index.html).
GitHub displays the symlink target but does not traverse nested browser paths
through it; a normal checkout resolves `current/` directly.

The workflow prepares this layout with `just bench-save "$GITHUB_RUN_ID"`, then
a separate publisher updates `automation/benchmark-report-current` and adds a
one-click PR link to the run summary. The organization forbids `GITHUB_TOKEN`
from creating the PR itself. Protected `main` means a new run enters this
history through a normal reviewed commit; the benchmark runner has no
branch-protection bypass.
