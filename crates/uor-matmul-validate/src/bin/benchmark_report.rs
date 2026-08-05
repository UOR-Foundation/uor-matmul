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
    if matches!(group, "workspace" | "gray_sign" | "modular_strassen") {
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
    format!(
        "<h3>{}</h3><p>Ratios are competitor time divided by {} time; <strong>greater than 1x means {} is faster</strong>.</p><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>",
        html_escape(comparison.title),
        html_escape(comparison.primary_label),
        html_escape(comparison.primary_label)
    )
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

    let notable = [
        ("i8 scaling exponent", "1.0220 +/- 0.0533 (95%)"),
        ("i8 64^3", "783.4 us"),
        ("i8 128^3", "6.209 ms"),
        ("i8 256^3", "81.06 ms"),
        ("f32 uor-matmul 128^3", "2.547 ms"),
        ("f32 matrixmultiply 128^3", "443.1 us"),
        ("f32 faer 128^3", "700.1 us"),
        ("i8 modular-Strassen packed 2048^3", "2.3169 s"),
        ("i8 modular-Strassen level 2048^3", "2.3168 s"),
    ];
    let comparisons = comparisons();
    let mut markdown = String::new();
    writeln!(
        markdown,
        "# Raspberry Pi Benchmark Report\n\nGenerated from the Criterion artifacts.\n"
    )
    .unwrap();
    markdown.push_str("## Run context\n\n");
    markdown.push_str(
        "- Host: rpi1 - Raspberry Pi, 4-core Cortex-A72, aarch64\n\
         - Toolchain: Rust 1.97.1\n\
         - Repository revision: 8df6471\n",
    );
    writeln!(
        markdown,
        "- Completed Criterion measurements: **{}**",
        rows.len()
    )
    .unwrap();
    markdown.push_str(
        "- Timing unit: Criterion nanosecond estimates converted for readability; intervals are 95% confidence intervals.\n\n\
         The run was intentionally stopped before the remaining modular-Strassen stress cases: i32 2048^3, and all 4096^3 variants. Those cases are not represented as zeroes or failures here.\n\n\
         ## Notable measurements\n\n| Benchmark | Mean |\n| --- | ---: |\n",
    );
    for (name, value) in notable {
        writeln!(markdown, "| {name} | {value} |").unwrap();
    }
    markdown.push_str(
        "\n## Direct comparisons\n\n\
         These tables compare identical shapes. The speedup columns are competitor time divided by the reference time; a value above 1x means the reference operation is faster.\n\n",
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

    let notable_html = notable
        .iter()
        .map(|(name, value)| {
            format!(
                "<tr><td>{}</td><td>{}</td></tr>",
                html_escape(name),
                html_escape(value)
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
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
    let html = format!(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>Raspberry Pi Benchmark Report</title><style>body{{font:15px/1.5 system-ui,-apple-system,sans-serif;color:#202124;max-width:1200px;margin:2rem auto;padding:0 1rem}}h1{{margin-bottom:.25rem}}h2{{margin-top:2rem}}table{{border-collapse:collapse;width:100%;margin:1rem 0}}th,td{{border:1px solid #d0d7de;padding:.45rem .6rem;text-align:left}}th{{background:#f6f8fa}}td:nth-child(n+3),th:nth-child(n+3){{text-align:right}}code{{font-size:.92em}}li{{margin:.2rem 0}}.note{{background:#fff8c5;border:1px solid #d4a72c;padding:.7rem 1rem}}</style></head><body><h1>Raspberry Pi Benchmark Report</h1><p>Generated from the Criterion artifacts.</p><h2>Run context</h2><ul><li>Host: <code>rpi1</code> - Raspberry Pi, 4-core Cortex-A72, <code>aarch64</code></li><li>Toolchain: Rust <code>1.97.1</code></li><li>Repository revision: <code>8df6471</code></li><li>Completed Criterion measurements: <strong>{}</strong></li><li>Intervals: 95% confidence intervals</li></ul><p class=\"note\">The run was intentionally stopped before modular-Strassen i32 <code>2048^3</code> and all <code>4096^3</code> stress cases. They are not represented as zeroes or failures.</p><h2>Notable measurements</h2><table><thead><tr><th>Benchmark</th><th>Mean</th></tr></thead><tbody>{}</tbody></table><h2>Direct comparisons</h2>{}<h2>Completed measurements</h2><table><thead><tr><th>Group</th><th>Benchmark</th><th>Mean</th><th>95% confidence interval</th><th>Raw estimate</th></tr></thead><tbody>{}</tbody></table></body></html>",
        rows.len(),
        notable_html,
        comparison_html_output,
        all_html
    );

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
