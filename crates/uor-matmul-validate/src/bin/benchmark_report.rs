//! Consolidate Criterion output into comparison-oriented Markdown and HTML.

use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;
use uor_matmul_validate::scaling::{fit, Observation};

const GROUPS: &[&str] = &[
    "gemm",
    "gemm_i32",
    "gemm_f32",
    "gemm_f64",
    "tropical",
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
    if matches!(group, "gemm_i32" | "gemm_f32" | "gemm_f64") {
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
        "workspace" | "lookup_build" | "gray_sign" | "modular_strassen" | "tropical"
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

fn comparison_anchor(title: &str) -> String {
    let mut anchor = String::with_capacity(title.len());
    for character in title.chars() {
        if character.is_ascii_alphanumeric() {
            anchor.push(character.to_ascii_lowercase());
        } else if !anchor.ends_with('-') {
            anchor.push('-');
        }
    }
    anchor.trim_matches('-').to_owned()
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
    if comparison_rows(comparison, rows).is_empty() {
        return String::new();
    }
    let mut headers = vec![
        "Shape".to_owned(),
        format!("{} (ours)", comparison.primary_label),
    ];
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
    if comparison_rows(comparison, rows).is_empty() {
        return String::new();
    }
    let mut headers = vec![
        "Shape".to_owned(),
        format!("{} (ours)", comparison.primary_label),
    ];
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
    let scaling_chart = scaling_chart_html(comparison, rows);
    format!(
        "<section class=\"comparison\" id=\"{}\"><h3>{}</h3><p>Each row uses the same shape. The blue series is our implementation; ratios are alternative time divided by ours, so values above 1× favor ours.</p><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>{chart}{scaling_chart}</section>",
        comparison_anchor(comparison.title),
        html_escape(comparison.title),
    )
}

fn comparison_chart_html(comparison: &Comparison<'_>, rows: &[Row]) -> String {
    let points = comparison_rows(comparison, rows);
    if points.is_empty() {
        return String::new();
    }
    let left = 180.0;
    let chart_width = 600.0;
    let series_count = comparison.others.len() + 1;
    let row_height = 28.0 + series_count as f64 * 16.0;
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
        "<div class=\"chart-wrap\"><div class=\"chart-title\">Relative time · 1× = {} (ours)</div><div class=\"legend\"><span><i class=\"legend-ours\"></i>{} (ours)</span><span><i class=\"legend-other\"></i>alternative</span></div><svg class=\"chart\" viewBox=\"0 0 820 {:.0}\" role=\"img\" aria-label=\"{} relative benchmark chart\">",
        html_escape(comparison.primary_label),
        html_escape(comparison.primary_label),
        height,
        html_escape(comparison.title)
    )
    .unwrap();
    let baseline_x = left + chart_width / max_ratio;
    write!(
        svg,
        "<line x1=\"{baseline_x:.1}\" x2=\"{baseline_x:.1}\" y1=\"0\" y2=\"{height:.1}\" class=\"baseline\"/><text x=\"{baseline_x:.1}\" y=\"12\" class=\"baseline-label\">1× ours</text>"
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
            y + 15.0,
            html_escape(shape)
        )
        .unwrap();
        let mut series = Vec::with_capacity(series_count);
        series.push(("ours", 1.0_f64, true));
        series.extend(others.iter().enumerate().map(|(index, other)| {
            (
                comparison.others[index].label,
                other.mean / primary.mean,
                false,
            )
        }));
        for (series_index, (label, ratio_value, ours)) in series.into_iter().enumerate() {
            let bar_y = y + 25.0 + series_index as f64 * 16.0;
            let width = chart_width * (ratio_value / max_ratio).min(1.0);
            write!(
                svg,
                "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"end\" class=\"series-label\">{}</text><rect x=\"{left:.1}\" y=\"{bar_y:.1}\" width=\"{width:.1}\" height=\"10\" rx=\"2\" class=\"{}\"/><text x=\"{:.1}\" y=\"{:.1}\" class=\"bar-label\">{:.2}×</text>",
                left - 8.0,
                bar_y + 9.0,
                html_escape(label),
                if ours { "bar-ours" } else { "bar-other" },
                left + width + 7.0,
                bar_y + 8.0,
                ratio_value
            )
            .unwrap();
        }
    }
    svg.push_str("</svg></div>");
    svg
}

fn parse_shape_work(shape: &str) -> Option<f64> {
    let mut dimensions = shape.split('x').map(str::parse::<f64>);
    let m = dimensions.next()?.ok()?;
    let k = dimensions.next()?.ok()?;
    let n = dimensions.next()?.ok()?;
    if dimensions.next().is_some() || m <= 0.0 || k <= 0.0 || n <= 0.0 {
        return None;
    }
    Some(m * k * n)
}

fn format_work(work: f64) -> String {
    if work >= 1_000_000_000.0 {
        format!("{:.1}G", work / 1_000_000_000.0)
    } else if work >= 1_000_000.0 {
        format!("{:.1}M", work / 1_000_000.0)
    } else if work >= 1_000.0 {
        format!("{:.1}K", work / 1_000.0)
    } else {
        format!("{work:.0}")
    }
}

fn scaling_chart_html(comparison: &Comparison<'_>, rows: &[Row]) -> String {
    if !matches!(comparison.group, "gemm_i32" | "gemm_f32" | "gemm_f64") {
        return String::new();
    }
    let mut specs = vec![(comparison.primary_label, comparison.primary_prefix)];
    specs.extend(
        comparison
            .others
            .iter()
            .map(|other| (other.label, other.prefix)),
    );

    let mut points = Vec::new();
    for shape in comparison.shapes {
        let Some(work) = parse_shape_work(shape) else {
            return String::new();
        };
        let Some(primary) = row_for(rows, comparison.group, comparison.primary_prefix, shape)
        else {
            return String::new();
        };
        let mut means = vec![primary.mean];
        for other in comparison.others {
            let Some(row) = row_for(rows, comparison.group, other.prefix, shape) else {
                return String::new();
            };
            means.push(row.mean);
        }
        points.push(((*shape).to_owned(), work, means));
    }
    if points.len() < 3 {
        return String::new();
    }

    let normalized = specs
        .iter()
        .enumerate()
        .map(|(series_index, (label, _))| {
            let first_cost = points[0].2[series_index] / points[0].1;
            let values = points
                .iter()
                .map(|(_, work, means)| (means[series_index] / work) / first_cost)
                .collect::<Vec<_>>();
            let fit = fit(&points
                .iter()
                .map(|(_, work, means)| Observation {
                    x: *work,
                    y: means[series_index],
                })
                .collect::<Vec<_>>());
            ((*label).to_owned(), values, fit.map(|fit| fit.exponent))
        })
        .collect::<Vec<_>>();

    let min_value = normalized
        .iter()
        .flat_map(|(_, values, _)| values.iter().copied())
        .chain(std::iter::once(1.0))
        .fold(f64::INFINITY, f64::min)
        .max(0.1);
    let max_value = normalized
        .iter()
        .flat_map(|(_, values, _)| values.iter().copied())
        .chain(std::iter::once(1.0))
        .fold(0.0, f64::max)
        .max(min_value * 1.5);
    let y_min = min_value.log10() - 0.1;
    let y_max = max_value.log10() + 0.1;
    let x_min = points.first().unwrap().1.log10();
    let x_max = points.last().unwrap().1.log10();
    let left = 72.0;
    let right = 205.0;
    let top = 26.0;
    let bottom = 48.0;
    let width = 860.0;
    let height = 300.0;
    let plot_width = width - left - right;
    let plot_height = height - top - bottom;
    let x = |work: f64| left + (work.log10() - x_min) / (x_max - x_min) * plot_width;
    let y = |value: f64| top + (y_max - value.log10()) / (y_max - y_min) * plot_height;

    let mut svg = String::new();
    write!(
        svg,
        "<div class=\"scaling-chart-wrap\"><div class=\"chart-title\">Scaling efficiency · time per MAC, normalized to the smallest shape</div><p class=\"chart-note\">A flat line means linear scaling with work. The fitted slope is time versus MAC count; 1.00 is the linear reference.</p><svg class=\"scaling-chart\" viewBox=\"0 0 {width:.0} {height:.0}\" role=\"img\" aria-label=\"{} scaling efficiency chart\">",
        html_escape(comparison.title)
    )
    .unwrap();

    write!(
        svg,
        "<line x1=\"{left:.1}\" x2=\"{:.1}\" y1=\"{:.1}\" y2=\"{:.1}\" class=\"scale-axis\"/><line x1=\"{left:.1}\" x2=\"{left:.1}\" y1=\"{top:.1}\" y2=\"{:.1}\" class=\"scale-axis\"/>",
        width - right,
        height - bottom,
        height - bottom,
        height - bottom
    )
    .unwrap();

    let mut y_ticks = vec![1.0];
    for tick in [0.25, 0.5, 2.0, 4.0, 8.0, 16.0] {
        if tick > min_value * 0.8 && tick < max_value * 1.25 {
            y_ticks.push(tick);
        }
    }
    y_ticks.sort_by(f64::total_cmp);
    y_ticks.dedup_by(|a, b| (*a - *b).abs() < f64::EPSILON);
    for tick in y_ticks {
        let y_pos = y(tick);
        let class = if (tick - 1.0).abs() < f64::EPSILON {
            "scale-reference"
        } else {
            "scale-grid"
        };
        write!(
            svg,
            "<line x1=\"{left:.1}\" x2=\"{:.1}\" y1=\"{y_pos:.1}\" y2=\"{y_pos:.1}\" class=\"{class}\"/><text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"end\" class=\"scale-label\">{tick:.2}×</text>",
            width - right,
            left - 8.0,
            y_pos + 4.0
        )
        .unwrap();
    }
    for (index, (shape, work, _)) in points.iter().enumerate() {
        let x_pos = x(*work);
        write!(
            svg,
            "<line x1=\"{x_pos:.1}\" x2=\"{x_pos:.1}\" y1=\"{top:.1}\" y2=\"{:.1}\" class=\"scale-grid\"/><text x=\"{x_pos:.1}\" y=\"{:.1}\" text-anchor=\"middle\" class=\"scale-label\">{}</text>",
            height - bottom,
            height - 25.0,
            html_escape(&format!("{} · {}", shape, format_work(*work)))
        )
        .unwrap();
        if index == 0 {
            write!(
                svg,
                "<text x=\"{:.1}\" y=\"{:.1}\" class=\"scale-axis-label\">normalized time / MAC</text>",
                left,
                top - 10.0
            )
            .unwrap();
        }
    }

    let linear_y = y(1.0);
    write!(
        svg,
        "<text x=\"{:.1}\" y=\"{:.1}\" class=\"scale-reference-label\">linear work</text>",
        width - right - 4.0,
        linear_y - 6.0
    )
    .unwrap();
    for (series_index, (_, values, _)) in normalized.iter().enumerate() {
        let class = if series_index == 0 {
            "scale-ours"
        } else {
            "scale-other"
        };
        let mut path = String::new();
        for (point_index, ((_, work, _), value)) in points.iter().zip(values).enumerate() {
            write!(
                path,
                "{} {:.1},{:.1}",
                if point_index == 0 { "M" } else { "L" },
                x(*work),
                y(*value)
            )
            .unwrap();
        }
        write!(svg, "<path d=\"{path}\" class=\"{class}\"/>").unwrap();
        for ((_, work, _), value) in points.iter().zip(values) {
            write!(
                svg,
                "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"3\" class=\"{class}\"/>",
                x(*work),
                y(*value)
            )
            .unwrap();
        }
    }
    for (index, (label, _, exponent)) in normalized.iter().enumerate() {
        let color_class = if index == 0 {
            "scale-legend-ours"
        } else {
            "scale-legend-other"
        };
        let text = match exponent {
            Some(exponent) => format!("{label} · slope {exponent:.2}"),
            None => label.clone(),
        };
        let legend_y = top + index as f64 * 22.0;
        write!(
            svg,
            "<line x1=\"{:.1}\" x2=\"{:.1}\" y1=\"{legend_y:.1}\" y2=\"{legend_y:.1}\" class=\"{color_class}\"/><text x=\"{:.1}\" y=\"{:.1}\" class=\"scale-label\">{}</text>",
            width - right + 12.0,
            width - right + 38.0,
            width - right + 46.0,
            legend_y + 4.0,
            html_escape(&text)
        )
        .unwrap();
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
            title: "f64 GEMM",
            group: "gemm_f64",
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
            title: "Tropical lane scaling",
            group: "tropical",
            shapes: &["64x64x64", "128x128x128", "16x4096x16"],
            primary_label: "tropical lane",
            primary_prefix: "lane/tropical/",
            others: &[
                Other {
                    label: "ring lane",
                    prefix: "lane/ring/",
                },
                Other {
                    label: "ring packed",
                    prefix: "lane/ring-packed/",
                },
            ],
        },
        Comparison {
            title: "Tropical witness scaling · tie-dense",
            group: "tropical",
            shapes: &[
                "tie-dense/64x64x64",
                "tie-dense/128x128x128",
                "tie-dense/16x4096x16",
            ],
            primary_label: "lexicographic",
            primary_prefix: "witness/lexicographic/",
            others: &[Other {
                label: "compare-pass",
                prefix: "witness/compare-pass/",
            }],
        },
        Comparison {
            title: "Tropical witness scaling · max-last",
            group: "tropical",
            shapes: &[
                "max-last/64x64x64",
                "max-last/128x128x128",
                "max-last/16x4096x16",
            ],
            primary_label: "lexicographic",
            primary_prefix: "witness/lexicographic/",
            others: &[Other {
                label: "compare-pass",
                prefix: "witness/compare-pass/",
            }],
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
         These tables compare identical shapes. The speedup columns are competitor time divided by the named primary; a value above 1× means the primary operation is faster. GEMM sections also include a scaling-efficiency chart: normalized time per MAC stays flat at 1.00× for linear work scaling, while a rising line shows worsening efficiency as the problem grows.\n\n",
    );
    if rows.iter().any(|row| row.group == "tropical") {
        markdown.push_str(
            "**Scaling highlighted:** tropical results are reported in three views below: lane scaling, tie-dense witness scaling, and max-last witness scaling.\n\n",
        );
    }
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
    let tropical_highlight = if rows.iter().any(|row| row.group == "tropical") {
        "<aside class=\"highlight\"><strong>Tropical scaling</strong><span>Lane scaling plus tie-dense and max-last witness scaling are included in the comparisons below.</span><a href=\"#tropical-lane-scaling\">View the tropical charts</a></aside>"
    } else {
        ""
    };
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
        r##"<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Comparison benchmark report</title><style>
        :root{color-scheme:light;--ink:#1f2937;--muted:#667085;--line:#d0d5dd;--panel:#fff;--wash:#f8fafc;--accent:#155eef;--ours:#155eef;--other:#64748b}
        *{box-sizing:border-box}body{margin:0;background:var(--wash);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}main{max-width:1120px;margin:0 auto;padding:0 1.25rem 3rem}
        header{background:var(--panel);border-bottom:1px solid var(--line);padding:2.25rem 0 1.5rem}
        h1{font-size:2rem;line-height:1.15;font-weight:650;letter-spacing:-.02em;margin:0 0 .45rem}h2{font-size:1.35rem;margin:2.25rem 0 .8rem}h3{font-size:1.15rem;margin:.1rem 0 .45rem}p{color:var(--muted)}header p{margin:.35rem 0 1.25rem}
        .context{display:flex;flex-wrap:wrap;gap:.35rem 1.25rem;color:var(--muted);font-size:.82rem}.context div{display:flex;gap:.35rem}.context small{color:#475467;font-weight:650}.context code{overflow-wrap:anywhere}
        nav{border-bottom:1px solid var(--line);padding:.7rem 0}nav a{color:var(--accent);font-weight:600;margin-right:1.25rem;text-decoration:none}nav a:hover{text-decoration:underline}.highlight{display:flex;flex-wrap:wrap;align-items:baseline;gap:.45rem 1rem;background:#eff4ff;border:1px solid #b2ccff;border-left:3px solid var(--ours);padding:.75rem .9rem;margin:1rem 0 1.25rem}.highlight strong{color:#123b8e}.highlight span{color:#344054}.highlight a{color:var(--accent);font-weight:600;margin-left:auto}
        .comparison{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--ours);padding:1rem 1.1rem;margin:1rem 0 1.25rem}.comparison p{margin:.2rem 0 .8rem}.comparison table{margin-top:.8rem}.chart-wrap,.scaling-chart-wrap{margin-top:1rem;border-top:1px solid var(--line);padding-top:.8rem;overflow-x:auto}.chart-title{color:var(--ink);font-size:.85rem;font-weight:600;margin-bottom:.35rem}.chart-note{font-size:.8rem;margin:.15rem 0 .45rem}.legend{display:flex;gap:1rem;color:var(--muted);font-size:.8rem;margin-bottom:.35rem}.legend span{display:inline-flex;align-items:center;gap:.35rem}.legend i{display:inline-block;width:.7rem;height:.7rem;border-radius:2px}.legend-ours{background:var(--ours)}.legend-other{background:var(--other)}.chart,.scaling-chart{display:block;min-width:760px;width:100%;height:auto}.baseline{stroke:#98a2b3;stroke-dasharray:3 3}.baseline-label,.shape-label,.series-label,.bar-label,.scale-label{font:12px system-ui,sans-serif;fill:var(--muted)}.shape-label{fill:var(--ink);font-weight:600}.series-label{text-transform:lowercase}.bar-label{font-weight:600}.bar-ours{fill:var(--ours)}.bar-other{fill:var(--other)}.scale-axis{stroke:#667085;stroke-width:1}.scale-grid{stroke:#e4e7ec;stroke-width:1}.scale-reference{stroke:#155eef;stroke-width:1.2;stroke-dasharray:4 3}.scale-reference-label,.scale-axis-label{font:12px system-ui,sans-serif;fill:#155eef;font-weight:600}.scale-ours{fill:none;stroke:var(--ours);stroke-width:2.5}.scale-other{fill:none;stroke:var(--other);stroke-width:2;stroke-dasharray:5 3}.scale-legend-ours{stroke:var(--ours);stroke-width:2.5}.scale-legend-other{stroke:var(--other);stroke-width:2;stroke-dasharray:5 3}
        table{border-collapse:collapse;width:100%;margin:1rem 0;background:var(--panel);border:1px solid var(--line)}th,td{border-bottom:1px solid var(--line);padding:.6rem .7rem;text-align:left}th{background:#f2f4f7;color:#475467;font-size:.78rem;text-transform:uppercase;letter-spacing:.04em}tr:last-child td{border-bottom:0}tbody tr:nth-child(even){background:#fcfcfd}th:nth-child(n+3),td:nth-child(n+3){text-align:right}th:nth-child(2),td:nth-child(2){background:#eff4ff}code{font-size:.92em;background:#f2f4f7;border-radius:3px;padding:.08rem .3rem}@media(max-width:700px){main{padding-left:.75rem;padding-right:.75rem}td,th{padding:.5rem;font-size:.82rem}table{display:block;overflow-x:auto;white-space:nowrap}.comparison{padding:.8rem}}
        </style></head><body><header><main><h1>Comparison benchmark report</h1><p>Same-shape measurements of this implementation and the alternatives. Blue marks our primary route.</p><div class="context"><div><small>Host</small><code>"##,
    );
    write!(
        html,
        "{}</code></div><div><small>Platform</small><code>{}/{}</code></div><div><small>Workflow</small><code>{}</code></div><div><small>Run</small><code>{}</code></div><div><small>Revision</small><code>{}</code></div><div><small>Measurements</small><code>{}</code></div></div></main></header><main><nav><a href=\"#comparisons\">Comparisons</a>{}<a href=\"#measurements\">All measurements</a></nav><section id=\"comparisons\"><h2>Direct comparisons</h2><p>Blue marks our primary route. GEMM scaling charts show normalized time per MAC: flat at 1.00× is linear work scaling; rising means the implementation is losing efficiency as the problem grows.</p>{}{}</section><section id=\"measurements\"><h2>All completed measurements</h2><table><thead><tr><th>Group</th><th>Benchmark</th><th>Mean</th><th>95% confidence interval</th><th>Raw estimate</th></tr></thead><tbody>{}</tbody></table></section></main></body></html>",
        html_escape(&host),
        html_escape(&os),
        html_escape(&arch),
        html_escape(&workflow),
        html_escape(&run),
        html_escape(&revision),
        rows.len(),
        if tropical_highlight.is_empty() {
            "".to_owned()
        } else {
            "<a href=\"#tropical-lane-scaling\">Tropical scaling</a>".to_owned()
        },
        tropical_highlight,
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
