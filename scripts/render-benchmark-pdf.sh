#!/usr/bin/env bash
set -euo pipefail

report=${1:-target/benchmark-report}
html="$report/index.html"
pdf="$report/REPORT.pdf"

test -f "$html" || {
  echo "missing benchmark HTML: $html" >&2
  exit 1
}

browser=${CHROME_BIN:-}
if [[ -n "$browser" && ! -x "$browser" ]]; then
  echo "CHROME_BIN is not executable: $browser" >&2
  exit 1
fi

if [[ -z "$browser" ]]; then
  for candidate in \
    chrome-headless-shell \
    google-chrome \
    google-chrome-stable \
    chromium \
    chromium-browser \
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
    "/Applications/Chromium.app/Contents/MacOS/Chromium" \
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser" \
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"
  do
    if [[ "$candidate" == */* ]]; then
      if [[ -x "$candidate" ]]; then
        browser=$candidate
        break
      fi
    elif command -v "$candidate" >/dev/null 2>&1; then
      browser=$(command -v "$candidate")
      break
    fi
  done
fi

if [[ -z "$browser" ]]; then
  echo "no Chrome-compatible browser found; install Chrome, Chromium, Brave, or Edge, or set CHROME_BIN" >&2
  exit 1
fi

report_dir=$(cd "$report" && pwd -P)
profile=$(mktemp -d "${TMPDIR:-/tmp}/uor-benchmark-pdf.XXXXXX")
trap 'rm -rf "$profile"' EXIT

"$browser" \
  --headless \
  --disable-gpu \
  --disable-background-networking \
  --no-first-run \
  --no-pdf-header-footer \
  --password-store=basic \
  --timeout=10000 \
  --use-mock-keychain \
  --user-data-dir="$profile" \
  --print-to-pdf="$report_dir/REPORT.pdf" \
  "file://$report_dir/index.html"

test -s "$pdf" || {
  echo "browser did not produce a non-empty PDF: $pdf" >&2
  exit 1
}

printf 'wrote %s\n' "$pdf"
