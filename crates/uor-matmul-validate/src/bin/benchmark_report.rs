//! Consolidate Criterion output into comparison-oriented Markdown and HTML.

use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;

const GROUPS: &[&str] = &[
    "gemm",
    "gemm_i32",
    "gemm_f32",
    "public_api",
    "workspace",
    "lookup_build",
    "gray_sign",
    "modular_strassen",
];

struct Row {
    group: String,
    name: String,
    relative: String,
    mean: f64,
    lower: f64,
    upper: f64,
}

struct Other<'a> {
    label: &'a str,
    prefix: &'a str,
}

struct Comparison<'a> {
    title: &'a str,
    group: &'a str,
    shapes: &'a [&'a str],
    primary_label: &'a str,
    primary_prefix: &'a str,
    others: &'a [Other<'a>],
}

fn json_number(root: &Value, path: &[&str]) -> Result<f64, String> {
    let mut value = root;
    for key in path {
        value = value
            .get(*key)
            .ok_or_else(|| format!("missing estimates field {}", path.join(".")))?;
    }
    value
        .as_f64()
        .ok_or_else(|| format!("non-numeric estimates field {}", path.join(".")))
}

fn display_name(name: &str, group: &str) -> String {
    if group == "public_api" {
        if let Some(rest) = name.strip_prefix("slice__gemm_") {
            return format!("slice::gemm/{rest}");
        }
        if let Some(rest) = name.strip_prefix("gemm_packed_") {
            return format!("gemm_packed/{rest}");
        }
    }
    if matches!(group, "gemm_i32" | "gemm_f32") {
        for prefix in [
            "uor-matmul",
            "handwritten",
            "ndarray",
            "nalgebra",
            "matrixmultiply",
            "faer",
        ] {
            if let Some(rest) = name.strip_prefix(&format!("{prefix}_")) {
                return format!("{prefix}/{rest}");
            }
        }
    }
    if matches!(
        group,
        "workspace" | "lookup_build" | "gray_sign" | "modular_strassen"
    ) {
        return name.replace('_', "/");
    }
    name.to_owned()
}

fn collect_rows(root: &Path) -> Result<Vec<Row>, String> {
    let mut rows = Vec::new();
    for group in GROUPS {
        let group_dir = root.join(group);
        if !group_dir.exists() {
            continue;
        }
        let mut names = fs::read_dir(&group_dir)
            .map_err(|error| format!("read {}: {error}", group_dir.display()))?
            .map(|entry| entry.map(|entry| entry.file_name().to_string_lossy().into_owned()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| format!("read {group}: {error}"))?;
        names.sort();
        for name in names {
            let estimate_path = group_dir.join(&name).join("new/estimates.json");
            if !estimate_path.exists() {
                continue;
            }
            let json: Value = serde_json::from_str(
                &fs::read_to_string(&estimate_path)
                    .map_err(|error| format!("read {}: {error}", estimate_path.display()))?,
            )
            .map_err(|error| format!("parse {}: {error}", estimate_path.display()))?;
            let relative = estimate_path
                .strip_prefix(root)
                .map_err(|error| format!("relative path: {error}"))?
                .to_string_lossy()
                .replace(std::path::MAIN_SEPARATOR, "/");
            rows.push(Row {
                group: (*group).to_owned(),
                name: display_name(&name, group),
                relative,
                mean: json_number(&json, &["mean", "point_estimate"])?,
                lower: json_number(&json, &["mean", "confidence_interval", "lower_bound"])?,
                upper: json_number(&json, &["mean", "confidence_interval", "upper_bound"])?,
            });
        }
    }
    Ok(rows)
}

fn format_ns(mut value: f64) -> String {
    let unit = if value >= 1e9 {
        value /= 1e9;
        "s"
    } else if value >= 1e6 {
        value /= 1e6;
        "ms"
    } else if value >= 1e3 {
        value /= 1e3;
        "µs"
    } else {
        "ns"
    };
    let rendered = if value >= 100.0 {
        format!("{value:.1}")
    } else if value >= 10.0 {
        format!("{value:.2}")
    } else {
        format!("{value:.3}")
    };
    format!("{rendered} {unit}")
}

fn html_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn row_for<'a>(rows: &'a [Row], group: &str, prefix: &str, suffix: &str) -> Option<&'a Row> {
    let name = format!("{prefix}{suffix}");
    rows.iter()
        .find(|row| row.group == group && row.name == name)
}

fn comparison_rows<'a>(
    comparison: &Comparison<'_>,
    rows: &'a [Row],
) -> Vec<(&'a Row, Vec<&'a Row>)> {
    comparison
        .shapes
        .iter()
        .filter_map(|shape| {
            let primary = row_for(rows, comparison.group, comparison.primary_prefix, shape)?;
            let others = comparison
                .others
                .iter()
                .map(|other| row_for(rows, comparison.group, other.prefix, shape))
                .collect::<Option<Vec<_>>>()?;
            Some((primary, others))
        })
        .collect()
}

fn ratio(other: &Row, primary: &Row) -> String {
    format!("{:.2}×", other.mean / primary.mean)
}

fn comparison_markdown(comparison: &Comparison<'_>, rows: &[Row]) -> String {
    let mut headers = vec!["Shape".to_owned(), comparison.primary_label.to_owned()];
    for other in comparison.others {
        headers.push(other.label.to_owned());
        headers.push(format!(
            "{} speedup vs {}",
            comparison.primary_label, other.label
        ));
    }
    let mut output = String::new();
    writeln!(output, "### {}\n", comparison.title).unwrap();
    writeln!(
        output,
        "Ratios are competitor time divided by {} time; greater than 1x means {} is faster.\n",
        comparison.primary_label, comparison.primary_label
    )
    .unwrap();
    writeln!(output, "| {} |", headers.join(" | ")).unwrap();
    writeln!(
        output,
        "| {} |",
        headers
            .iter()
            .map(|_| "---:")
            .collect::<Vec<_>>()
            .join(" | ")
    )
    .unwrap();
    for (primary, others) in comparison_rows(comparison, rows) {
        let shape = primary
            .name
            .strip_prefix(comparison.primary_prefix)
            .unwrap_or(&primary.name);
        write!(output, "| {shape} | {} |", format_ns(primary.mean)).unwrap();
        for other in others {
            write!(
                output,
                " {} | {} |",
                format_ns(other.mean),
                ratio(other, primary)
            )
            .unwrap();
        }
        output.push('\n');
    }
    output
}

fn comparison_html(comparison: &Comparison<'_>, rows: &[Row]) -> String {
    let mut headers = vec!["Shape".to_owned(), comparison.primary_label.to_owned()];
    for other in comparison.others {
        headers.push(other.label.to_owned());
        headers.push(format!(
            "{} speedup vs {}",
            comparison.primary_label, other.label
        ));
    }
    let head = headers
        .iter()
        .map(|header| format!("<th>{}</th>", html_escape(header)))
        .collect::<String>();
    let mut body = String::new();
    for (primary, others) in comparison_rows(comparison, rows) {
        let shape = primary
            .name
            .strip_prefix(comparison.primary_prefix)
            .unwrap_or(&primary.name);
        write!(
            body,
            "<tr><td><code>{}</code></td><td>{}</td>",
            html_escape(shape),
            format_ns(primary.mean)
        )
        .unwrap();
        for other in others {
            write!(
                body,
                "<td>{}</td><td>{}</td>",
                format_ns(other.mean),
                ratio(other, primary)
            )
            .unwrap();
        }
        body.push_str("</tr>\n");
    }
    let chart = comparison_chart_html(comparison, rows);
    format!(
        "<section class=\"comparison\"><h3>{}</h3><p>Ratios are competitor time divided by {} time; <strong>greater than 1× means {} is faster</strong>.</p><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>{chart}</section>",
        html_escape(comparison.title),
        html_escape(comparison.primary_label),
        html_escape(comparison.primary_label)
    )
}

fn comparison_chart_html(comparison: &Comparison<'_>, rows: &[Row]) -> String {
    let points = comparison_rows(comparison, rows);
    if points.is_empty() {
        return String::new();
    }
    let left = 190.0;
    let chart_width = 520.0;
    let row_height = 24.0 + comparison.others.len() as f64 * 12.0;
    let height = row_height * points.len() as f64;
    let max_ratio = points
        .iter()
        .flat_map(|(primary, others)| others.iter().map(|other| other.mean / primary.mean))
        .fold(1.0_f64, f64::max)
        .max(1.0)
        * 1.1;
    let mut svg = String::new();
    write!(
        svg,
        "<div class=\"chart-wrap\"><div class=\"chart-title\">Relative time — competitor ÷ {}</div><svg class=\"chart\" viewBox=\"0 0 760 {:.0}\" role=\"img\" aria-label=\"{} relative benchmark chart\">",
        html_escape(comparison.primary_label),
        height,
        html_escape(comparison.title)
    )
    .unwrap();
    let baseline_x = left + chart_width / max_ratio;
    write!(
        svg,
        "<line x1=\"{baseline_x:.1}\" x2=\"{baseline_x:.1}\" y1=\"0\" y2=\"{height:.1}\" class=\"baseline\"/><text x=\"{baseline_x:.1}\" y=\"12\" class=\"baseline-label\">1×</text>"
    )
    .unwrap();
    for (index, (primary, others)) in points.iter().enumerate() {
        let y = index as f64 * row_height;
        let shape = primary
            .name
            .strip_prefix(comparison.primary_prefix)
            .unwrap_or(&primary.name);
        write!(
            svg,
            "<text x=\"0\" y=\"{:.1}\" class=\"shape-label\">{}</text>",
            y + 16.0,
            html_escape(shape)
        )
        .unwrap();
        for (other_index, other) in others.iter().enumerate() {
            let ratio_value = other.mean / primary.mean;
            let bar_y = y + 21.0 + other_index as f64 * 12.0;
            let width = chart_width * (ratio_value / max_ratio).min(1.0);
            let color = if ratio_value >= 1.0 {
                "#238636"
            } else {
                "#cf222e"
            };
            write!(
                svg,
                "<rect x=\"{left:.1}\" y=\"{bar_y:.1}\" width=\"{width:.1}\" height=\"9\" rx=\"3\" fill=\"{color}\"/><text x=\"{:.1}\" y=\"{:.1}\" class=\"bar-label\">{} {:.2}×</text>",
                left + width + 7.0,
                bar_y + 8.0,
                html_escape(comparison.others[other_index].label),
                ratio_value
            )
            .unwrap();
        }
    }
    svg.push_str("</svg></div>");
    svg
}

fn comparisons() -> Vec<Comparison<'static>> {
    vec![
        Comparison {
            title: "i32 GEMM",
            group: "gemm_i32",
            shapes: &["16x16x16", "128x128x128", "64x512x1024"],
            primary_label: "uor-matmul",
            primary_prefix: "uor-matmul/",
            others: &[
                Other {
                    label: "handwritten",
                    prefix: "handwritten/",
                },
                Other {
                    label: "ndarray",
                    prefix: "ndarray/",
                },
                Other {
                    label: "nalgebra",
                    prefix: "nalgebra/",
                },
            ],
        },
        Comparison {
            title: "f32 GEMM",
            group: "gemm_f32",
            shapes: &["16x16x16", "128x128x128", "64x512x1024"],
            primary_label: "uor-matmul",
            primary_prefix: "uor-matmul/",
            others: &[
                Other {
                    label: "handwritten",
                    prefix: "handwritten/",
                },
                Other {
                    label: "matrixmultiply",
                    prefix: "matrixmultiply/",
                },
                Other {
                    label: "faer",
                    prefix: "faer/",
                },
            ],
        },
        Comparison {
            title: "Public route",
            group: "public_api",
            shapes: &["16x16x16", "64x64x64", "128x128x128"],
            primary_label: "slice::gemm",
            primary_prefix: "slice::gemm/",
            others: &[Other {
                label: "gemm_packed",
                prefix: "gemm_packed/",
            }],
        },
        Comparison {
            title: "Finite i8 lookup-table build",
            group: "lookup_build",
            shapes: &["space4096-blk16"],
            primary_label: "cpu-native",
            primary_prefix: "build/cpu-native/",
            others: &[Other {
                label: "portable",
                prefix: "build/portable/",
            }],
        },
        Comparison {
            title: "Modular-Strassen routes",
            group: "modular_strassen",
            shapes: &[
                "i8/512cubed",
                "i8/1024cubed",
                "i8/2048cubed",
                "i32/512cubed",
                "i32/1024cubed",
            ],
            primary_label: "packed",
            primary_prefix: "packed/",
            others: &[Other {
                label: "level",
                prefix: "level/",
            }],
        },
    ]
}

fn main() -> Result<(), String> {
    let root = env::args().nth(1).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/criterion")
    });
    if !root.exists() {
        return Err(format!(
            "Criterion directory does not exist: {}",
            root.display()
        ));
    }
    let rows = collect_rows(&root)?;
    if rows.is_empty() {
        return Err(format!(
            "No completed Criterion estimates found under {}",
            root.display()
        ));
    }

    let comparisons = comparisons();
    let host = env::var("RUNNER_NAME")
        .or_else(|_| env::var("HOSTNAME"))
        .unwrap_or_else(|_| "local".to_owned());
    let arch = env::var("RUNNER_ARCH").unwrap_or_else(|_| env::consts::ARCH.to_owned());
    let os = env::var("RUNNER_OS").unwrap_or_else(|_| env::consts::OS.to_owned());
    let revision = env::var("GITHUB_SHA")
        .or_else(|_| env::var("GIT_COMMIT"))
        .unwrap_or_else(|_| "unknown".to_owned());
    let workflow = env::var("GITHUB_WORKFLOW").unwrap_or_else(|_| "local".to_owned());
    let run = env::var("GITHUB_RUN_ID").unwrap_or_else(|_| "local".to_owned());
    let mut markdown = String::new();
    writeln!(
        markdown,
        "# Comparison Benchmark Report\n\nGenerated from Criterion artifacts.\n"
    )
    .unwrap();
    writeln!(markdown, "## Run context\n\n- Host: `{host}` ({os}/{arch})\n- Workflow: `{workflow}` (run `{run}`)\n- Repository revision: `{revision}`\n").unwrap();
    writeln!(
        markdown,
        "- Completed Criterion measurements: **{}**",
        rows.len()
    )
    .unwrap();
    markdown.push_str(
        "- Timing unit: Criterion estimates converted for readability; intervals are 95% confidence intervals.\n\n\
         Ratios below are competitor time divided by the named primary. Values above 1× mean the primary is faster.\n\n",
    );
    markdown.push_str(
        "\n## Direct comparisons\n\n\
         These tables compare identical shapes. The speedup columns are competitor time divided by the named primary; a value above 1× means the primary operation is faster.\n\n",
    );
    for comparison in &comparisons {
        markdown.push_str(&comparison_markdown(comparison, &rows));
        markdown.push('\n');
    }
    for group in GROUPS {
        let group_rows = rows
            .iter()
            .filter(|row| row.group == *group)
            .collect::<Vec<_>>();
        if group_rows.is_empty() {
            continue;
        }
        writeln!(
            markdown,
            "## {group}\n\n| Benchmark | Mean | 95% confidence interval | Raw estimate |\n| --- | ---: | ---: | --- |\n"
        )
        .unwrap();
        for row in group_rows {
            writeln!(
                markdown,
                "| {} | {} | {} - {} | [estimates.json]({}) |",
                row.name,
                format_ns(row.mean),
                format_ns(row.lower),
                format_ns(row.upper),
                row.relative
            )
            .unwrap();
        }
        markdown.push('\n');
    }

    let comparison_html_output = comparisons
        .iter()
        .map(|comparison| comparison_html(comparison, &rows))
        .collect::<Vec<_>>()
        .join("\n");
    let mut all_html = String::new();
    for group in GROUPS {
        for row in rows.iter().filter(|row| row.group == *group) {
            writeln!(
                all_html,
                "<tr><td>{}</td><td><code>{}</code></td><td>{}</td><td>{} - {}</td><td><a href=\"{}\">estimates.json</a></td></tr>",
                html_escape(&row.group),
                html_escape(&row.name),
                format_ns(row.mean),
                format_ns(row.lower),
                format_ns(row.upper),
                row.relative
            )
            .unwrap();
        }
    }
    let mut html = String::new();
    html.push_str(
        r##"<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Comparison Benchmark Report</title><style>
        :root{color-scheme:light;--ink:#172033;--muted:#5d6b82;--line:#d8e0ea;--panel:#fff;--wash:#f4f7fb;--accent:#2563eb}
        *{box-sizing:border-box}body{margin:0;background:var(--wash);color:var(--ink);font:15px/1.55 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
        header{background:linear-gradient(120deg,#172554,#2563eb);color:#fff;padding:3rem max(1rem,calc((100% - 1180px)/2)) 2.5rem}main{max-width:1180px;margin:0 auto;padding:1.25rem 1rem 3rem}
        h1{font-size:clamp(2rem,5vw,3.5rem);line-height:1.05;margin:0 0 .6rem}h2{margin:2.5rem 0 1rem}h3{font-size:1.25rem;margin:.2rem 0 .6rem}p{color:var(--muted)}header p{color:#dbeafe}header code{color:#fff;background:#ffffff22}
        .context{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:.75rem;margin-top:1.5rem}.context div{background:#ffffff18;border:1px solid #ffffff30;border-radius:10px;padding:.7rem .85rem}.context small{display:block;color:#bfdbfe;text-transform:uppercase;letter-spacing:.08em;font-size:.7rem}.context code{display:block;overflow-wrap:anywhere}
        nav{position:sticky;top:0;z-index:2;background:#ffffffee;backdrop-filter:blur(8px);border-bottom:1px solid var(--line);padding:.65rem 0}nav a{color:var(--accent);font-weight:650;margin-right:1rem;text-decoration:none}
        .comparison{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:1rem 1.1rem;margin:1rem 0 1.25rem;box-shadow:0 5px 18px #1720330b}.comparison p{margin:.2rem 0 .8rem}.chart-wrap{margin-top:1rem;overflow-x:auto;border-top:1px solid var(--line);padding-top:.85rem}.chart-title{color:var(--muted);font-size:.85rem;margin-bottom:.35rem}.chart{display:block;min-width:700px;width:100%;height:auto}.baseline{stroke:#64748b;stroke-dasharray:4 4}.baseline-label,.shape-label,.bar-label{font:12px system-ui,sans-serif;fill:var(--muted)}.shape-label{fill:var(--ink)}.bar-label{font-weight:650}
        table{border-collapse:separate;border-spacing:0;width:100%;margin:1rem 0;background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden}th,td{border-bottom:1px solid var(--line);padding:.65rem .75rem;text-align:left}th{background:#eef3f9;color:#334155;font-size:.82rem;text-transform:uppercase;letter-spacing:.04em}tr:last-child td{border-bottom:0}tbody tr:nth-child(even){background:#fbfcfe}td:nth-child(n+3),th:nth-child(n+3){text-align:right}code{font-size:.92em;background:#eef2f7;border-radius:4px;padding:.08rem .3rem}header code{background:transparent;padding:0}@media(max-width:700px){td,th{padding:.5rem;font-size:.85rem}table{display:block;overflow-x:auto;white-space:nowrap}}
        </style></head><body><header><main><h1>Comparison Benchmark Report</h1><p>Alternatives are compared at identical shapes; raw timing is included for context.</p><div class="context"><div><small>Host</small><code>"##,
    );
    write!(
        html,
        "{}</code></div><div><small>Platform</small><code>{}/{}</code></div><div><small>Workflow</small><code>{}</code></div><div><small>Run</small><code>{}</code></div><div><small>Revision</small><code>{}</code></div><div><small>Measurements</small><code>{}</code></div></div></main></header><main><nav><a href=\"#comparisons\">Comparisons</a><a href=\"#measurements\">All measurements</a></nav><section id=\"comparisons\"><h2>Direct comparisons</h2><p>Ratios are competitor time divided by the named primary. A ratio above 1× means the primary is faster; green bars favor the primary.</p>{}</section><section id=\"measurements\"><h2>Completed measurements</h2><table><thead><tr><th>Group</th><th>Benchmark</th><th>Mean</th><th>95% confidence interval</th><th>Raw estimate</th></tr></thead><tbody>{}</tbody></table></section></main></body></html>",
        html_escape(&host),
        html_escape(&os),
        html_escape(&arch),
        html_escape(&workflow),
        html_escape(&run),
        html_escape(&revision),
        rows.len(),
        comparison_html_output,
        all_html
    )
    .unwrap();

    fs::write(root.join("REPORT.md"), markdown)
        .map_err(|error| format!("write Markdown: {error}"))?;
    fs::write(root.join("REPORT.html"), html).map_err(|error| format!("write HTML: {error}"))?;
    println!(
        "wrote {} measurements to {}/REPORT.md and {}/REPORT.html",
        rows.len(),
        root.display(),
        root.display()
    );
    Ok(())
}
