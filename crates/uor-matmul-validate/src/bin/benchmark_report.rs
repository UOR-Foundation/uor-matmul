//! Consolidate Criterion output into comparison-oriented Markdown and HTML.

use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

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
    modified: SystemTime,
}

struct RunContext {
    host: String,
    arch: String,
    os: String,
    revision: String,
    workflow: String,
    run: String,
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
                modified: fs::metadata(&estimate_path)
                    .and_then(|metadata| metadata.modified())
                    .map_err(|error| {
                        format!("read timestamp {}: {error}", estimate_path.display())
                    })?,
            });
        }
    }
    Ok(rows)
}

fn context_from_environment() -> RunContext {
    RunContext {
        host: env::var("RUNNER_NAME")
            .or_else(|_| env::var("HOSTNAME"))
            .unwrap_or_else(|_| "local".to_owned()),
        arch: env::var("RUNNER_ARCH").unwrap_or_else(|_| env::consts::ARCH.to_owned()),
        os: env::var("RUNNER_OS").unwrap_or_else(|_| env::consts::OS.to_owned()),
        revision: env::var("GITHUB_SHA")
            .or_else(|_| env::var("GIT_COMMIT"))
            .unwrap_or_else(|_| "unknown".to_owned()),
        workflow: env::var("GITHUB_WORKFLOW").unwrap_or_else(|_| "local".to_owned()),
        run: env::var("GITHUB_RUN_ID").unwrap_or_else(|_| "local".to_owned()),
    }
}

fn context_string(json: &Value, key: &str) -> Result<String, String> {
    json.get(key)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| format!("missing or non-string report context field {key}"))
}

fn read_context(path: &Path) -> Result<RunContext, String> {
    let json: Value = serde_json::from_str(
        &fs::read_to_string(path)
            .map_err(|error| format!("read report context {}: {error}", path.display()))?,
    )
    .map_err(|error| format!("parse report context {}: {error}", path.display()))?;
    Ok(RunContext {
        host: context_string(&json, "host")?,
        arch: context_string(&json, "arch")?,
        os: context_string(&json, "os")?,
        revision: context_string(&json, "revision")?,
        workflow: context_string(&json, "workflow")?,
        run: context_string(&json, "run")?,
    })
}

fn report_context(source: &Path, output: &Path) -> Result<RunContext, String> {
    let bundled_source = output.join("criterion");
    let context_path = output.join("context.json");
    if source == bundled_source && context_path.exists() {
        read_context(&context_path)
    } else {
        Ok(context_from_environment())
    }
}

fn write_context(
    output: &Path,
    context: &RunContext,
    measurements: usize,
    required_comparisons: usize,
) -> Result<(), String> {
    let json = serde_json::json!({
        "schema": 1,
        "source": "Criterion new/estimates.json",
        "host": context.host,
        "arch": context.arch,
        "os": context.os,
        "revision": context.revision,
        "workflow": context.workflow,
        "run": context.run,
        "measurements": measurements,
        "required_comparisons": required_comparisons,
    });
    fs::write(
        output.join("context.json"),
        serde_json::to_string_pretty(&json)
            .map_err(|error| format!("serialize report context: {error}"))?,
    )
    .map_err(|error| format!("write report context: {error}"))
}

fn bundle_estimates(source: &Path, output: &Path, rows: &mut [Row]) -> Result<(), String> {
    let bundled_root = output.join("criterion");
    if source != bundled_root && bundled_root.exists() {
        fs::remove_dir_all(&bundled_root)
            .map_err(|error| format!("clear {}: {error}", bundled_root.display()))?;
    }
    for row in rows {
        let relative = PathBuf::from(&row.relative);
        let source_path = source.join(&relative);
        let bundled_path = bundled_root.join(&relative);
        if source_path != bundled_path {
            let parent = bundled_path
                .parent()
                .ok_or_else(|| format!("no parent for {}", bundled_path.display()))?;
            fs::create_dir_all(parent)
                .map_err(|error| format!("create {}: {error}", parent.display()))?;
            fs::copy(&source_path, &bundled_path).map_err(|error| {
                format!(
                    "copy {} to {}: {error}",
                    source_path.display(),
                    bundled_path.display()
                )
            })?;
        }
        row.relative = Path::new("criterion")
            .join(relative)
            .to_string_lossy()
            .replace(std::path::MAIN_SEPARATOR, "/");
    }
    Ok(())
}

fn retain_current_run(root: &Path, rows: &mut Vec<Row>) -> Result<(), String> {
    let marker = root.join(".comparison-run-start");
    if !marker.exists() {
        return Ok(());
    }
    let started = fs::metadata(&marker)
        .and_then(|metadata| metadata.modified())
        .map_err(|error| format!("read timestamp {}: {error}", marker.display()))?;
    retain_rows_modified_since(rows, started);
    Ok(())
}

fn retain_rows_modified_since(rows: &mut Vec<Row>, started: SystemTime) {
    rows.retain(|row| row.modified >= started);
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

fn required_comparison_count(comparisons: &[Comparison<'_>]) -> usize {
    comparisons
        .iter()
        .map(|comparison| comparison.shapes.len() * (comparison.others.len() + 1))
        .sum()
}

fn missing_comparison_rows(comparisons: &[Comparison<'_>], rows: &[Row]) -> Vec<String> {
    let mut missing = Vec::new();
    for comparison in comparisons {
        for shape in comparison.shapes {
            if row_for(rows, comparison.group, comparison.primary_prefix, shape).is_none() {
                missing.push(format!(
                    "{}/{}{}",
                    comparison.group, comparison.primary_prefix, shape
                ));
            }
            for other in comparison.others {
                if row_for(rows, comparison.group, other.prefix, shape).is_none() {
                    missing.push(format!("{}/{}{}", comparison.group, other.prefix, shape));
                }
            }
        }
    }
    missing
}

fn ratio(other: &Row, primary: &Row) -> String {
    format_ratio(other.mean / primary.mean)
}

fn format_ratio(value: f64) -> String {
    if value >= 100.0 {
        format!("{value:.1}×")
    } else if value >= 0.01 {
        format!("{value:.2}×")
    } else if value >= 0.001 {
        format!("{value:.3}×")
    } else {
        format!("{value:.1e}×")
    }
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
    // Series names and values live in separate fixed columns. Values once
    // followed the logarithmic bar endpoint, which made the smallest ratios
    // collide with long names such as `matrixmultiply`.
    let series_label_x = 150.0;
    let value_label_x = 240.0;
    let plot_left = 260.0;
    let chart_width = 640.0;
    let view_width = 920.0;
    let series_count = comparison.others.len() + 1;
    let series_step = 20.0;
    let row_height = 32.0 + series_count as f64 * series_step;
    let height = row_height * points.len() as f64;
    // Ratios in this report span several orders of magnitude. A linear axis
    // makes a fast alternative disappear at zero, so center the chart at 1×
    // and give equal space to reciprocal factors on a base-2 logarithmic axis.
    let log_extent = points
        .iter()
        .flat_map(|(primary, others)| {
            others
                .iter()
                .map(|other| (other.mean / primary.mean).log2().abs())
        })
        .fold(1.0_f64, f64::max)
        * 1.1;
    let x = |ratio: f64| plot_left + (ratio.log2() + log_extent) / (2.0 * log_extent) * chart_width;
    let mut svg = String::new();
    write!(
        svg,
        "<div class=\"chart-wrap\"><div class=\"chart-title\">Relative time · logarithmic · 1× = {} (ours)</div><div class=\"legend\"><span><i class=\"legend-ours\"></i>{} (ours)</span><span><i class=\"legend-other\"></i>alternative</span></div><svg class=\"chart\" viewBox=\"0 0 {view_width:.0} {:.0}\" role=\"img\" aria-label=\"{} logarithmic relative benchmark chart\">",
        html_escape(comparison.primary_label),
        html_escape(comparison.primary_label),
        height,
        html_escape(comparison.title)
    )
    .unwrap();
    let baseline_x = x(1.0);
    write!(
        svg,
        "<line x1=\"{baseline_x:.1}\" x2=\"{baseline_x:.1}\" y1=\"0\" y2=\"{height:.1}\" class=\"baseline\"/><text x=\"{plot_left:.1}\" y=\"12\" class=\"baseline-label\">alternative faster ←</text><text x=\"{baseline_x:.1}\" y=\"12\" text-anchor=\"middle\" class=\"baseline-label\">1×</text><text x=\"{:.1}\" y=\"12\" text-anchor=\"end\" class=\"baseline-label\">→ ours faster</text>",
        plot_left + chart_width
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
            let bar_y = y + 25.0 + series_index as f64 * series_step;
            let ratio_x = x(ratio_value);
            let (bar_x, width) = if ours {
                (baseline_x - 2.0, 4.0)
            } else {
                (
                    ratio_x.min(baseline_x),
                    (ratio_x - baseline_x).abs().max(2.0),
                )
            };
            write!(
                svg,
                "<text x=\"{series_label_x:.1}\" y=\"{:.1}\" text-anchor=\"end\" class=\"series-label\">{}</text><text x=\"{value_label_x:.1}\" y=\"{:.1}\" text-anchor=\"end\" class=\"bar-label\">{}</text><rect x=\"{bar_x:.1}\" y=\"{bar_y:.1}\" width=\"{width:.1}\" height=\"10\" rx=\"2\" class=\"{}\"/>",
                bar_y + 9.0,
                html_escape(label),
                bar_y + 8.0,
                format_ratio(ratio_value),
                if ours { "bar-ours" } else { "bar-other" }
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
    let bottom = 70.0;
    let width = 860.0;
    let height = 320.0;
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
        let label_y = if index % 2 == 0 {
            height - 36.0
        } else {
            height - 16.0
        };
        write!(
            svg,
            "<line x1=\"{x_pos:.1}\" x2=\"{x_pos:.1}\" y1=\"{top:.1}\" y2=\"{:.1}\" class=\"scale-grid\"/><text x=\"{x_pos:.1}\" y=\"{label_y:.1}\" text-anchor=\"middle\" class=\"scale-label\">{}</text>",
            height - bottom,
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
        "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"end\" class=\"scale-reference-label\">linear work</text>",
        width - right - 8.0,
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
            shapes: &["16x16x16", "128x128x128", "32x256x512"],
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
            shapes: &["16x16x16", "128x128x128", "32x256x512"],
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
            shapes: &["16x16x16", "128x128x128", "32x256x512"],
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
    let mut args = env::args().skip(1);
    let source = args.next().map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/criterion")
    });
    let output = args.next().map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/benchmark-report")
    });
    if let Some(unexpected) = args.next() {
        return Err(format!(
            "unexpected argument {unexpected}; usage: benchmark_report [criterion-directory] [report-directory]"
        ));
    }
    if !source.exists() {
        return Err(format!(
            "Criterion directory does not exist: {}",
            source.display()
        ));
    }
    let mut rows = collect_rows(&source)?;
    retain_current_run(&source, &mut rows)?;
    if rows.is_empty() {
        return Err(format!(
            "No completed Criterion estimates found under {}",
            source.display()
        ));
    }

    let comparisons = comparisons();
    let missing = missing_comparison_rows(&comparisons, &rows);
    if !missing.is_empty() {
        return Err(format!(
            "Comparison benchmark report is incomplete; missing {} required estimates:\n- {}",
            missing.len(),
            missing.join("\n- ")
        ));
    }
    let required_comparisons = required_comparison_count(&comparisons);
    let context = report_context(&source, &output)?;
    fs::create_dir_all(&output)
        .map_err(|error| format!("create report directory {}: {error}", output.display()))?;
    bundle_estimates(&source, &output, &mut rows)?;
    write_context(&output, &context, rows.len(), required_comparisons)?;
    let mut markdown = String::new();
    writeln!(
        markdown,
        "# Comparison Benchmark Report\n\nGenerated from Criterion artifacts.\n"
    )
    .unwrap();
    writeln!(markdown, "## Run context\n\n- Host: `{}` ({}/{})\n- Workflow: `{}` (run `{}`)\n- Repository revision: `{}`\n", context.host, context.os, context.arch, context.workflow, context.run, context.revision).unwrap();
    writeln!(
        markdown,
        "- Completed Criterion measurements: **{}**",
        rows.len()
    )
    .unwrap();
    writeln!(
        markdown,
        "- Required same-shape comparison measurements: **{required_comparisons} / {required_comparisons}**"
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
        .comparison{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--ours);padding:1rem 1.1rem;margin:1rem 0 1.25rem}.comparison p{margin:.2rem 0 .8rem}.comparison table{margin-top:.8rem}.chart-wrap,.scaling-chart-wrap{margin-top:1rem;border-top:1px solid var(--line);padding-top:.8rem;overflow-x:auto}.chart-title{color:var(--ink);font-size:.85rem;font-weight:600;margin-bottom:.35rem}.chart-note{font-size:.8rem;margin:.15rem 0 .45rem}.legend{display:flex;gap:1rem;color:var(--muted);font-size:.8rem;margin-bottom:.35rem}.legend span{display:inline-flex;align-items:center;gap:.35rem}.legend i{display:inline-block;width:.7rem;height:.7rem;border-radius:2px}.legend-ours{background:var(--ours)}.legend-other{background:var(--other)}.chart,.scaling-chart{display:block;width:100%;height:auto}.chart{min-width:900px}.scaling-chart{min-width:760px}.baseline{stroke:#98a2b3;stroke-dasharray:3 3}.baseline-label,.shape-label,.series-label,.bar-label,.scale-label{font:12px system-ui,sans-serif;fill:var(--muted)}.shape-label{fill:var(--ink);font-weight:600}.series-label{text-transform:lowercase}.bar-label{font-weight:600}.bar-ours{fill:var(--ours)}.bar-other{fill:var(--other)}.scale-axis{stroke:#667085;stroke-width:1}.scale-grid{stroke:#e4e7ec;stroke-width:1}.scale-reference{stroke:#155eef;stroke-width:1.2;stroke-dasharray:4 3}.scale-reference-label,.scale-axis-label{font:12px system-ui,sans-serif;fill:#155eef;font-weight:600}.scale-ours{fill:none;stroke:var(--ours);stroke-width:2.5}.scale-other{fill:none;stroke:var(--other);stroke-width:2;stroke-dasharray:5 3}.scale-legend-ours{stroke:var(--ours);stroke-width:2.5}.scale-legend-other{stroke:var(--other);stroke-width:2;stroke-dasharray:5 3}
        table{border-collapse:collapse;width:100%;margin:1rem 0;background:var(--panel);border:1px solid var(--line)}th,td{border-bottom:1px solid var(--line);padding:.6rem .7rem;text-align:left}th{background:#f2f4f7;color:#475467;font-size:.78rem;text-transform:uppercase;letter-spacing:.04em}tr:last-child td{border-bottom:0}tbody tr:nth-child(even){background:#fcfcfd}th:nth-child(n+3),td:nth-child(n+3){text-align:right}th:nth-child(2),td:nth-child(2){background:#eff4ff}code{font-size:.92em;background:#f2f4f7;border-radius:3px;padding:.08rem .3rem}@media(max-width:700px){main{padding-left:.75rem;padding-right:.75rem}td,th{padding:.5rem;font-size:.82rem}table{display:block;overflow-x:auto;white-space:nowrap}.comparison{padding:.8rem}}
        </style></head><body><header><main><h1>Comparison benchmark report</h1><p>Same-shape measurements of this implementation and the alternatives. Blue marks our primary route.</p><div class="context"><div><small>Host</small><code>"##,
    );
    write!(
        html,
        "{}</code></div><div><small>Platform</small><code>{}/{}</code></div><div><small>Workflow</small><code>{}</code></div><div><small>Run</small><code>{}</code></div><div><small>Revision</small><code>{}</code></div><div><small>Measurements</small><code>{}</code></div></div></main></header><main><nav><a href=\"#comparisons\">Comparisons</a>{}<a href=\"#measurements\">All measurements</a></nav><section id=\"comparisons\"><h2>Direct comparisons</h2><p>Blue marks our primary route. GEMM scaling charts show normalized time per MAC: flat at 1.00× is linear work scaling; rising means the implementation is losing efficiency as the problem grows.</p>{}{}</section><section id=\"measurements\"><h2>All completed measurements</h2><table><thead><tr><th>Group</th><th>Benchmark</th><th>Mean</th><th>95% confidence interval</th><th>Raw estimate</th></tr></thead><tbody>{}</tbody></table></section></main></body></html>",
        html_escape(&context.host),
        html_escape(&context.os),
        html_escape(&context.arch),
        html_escape(&context.workflow),
        html_escape(&context.run),
        html_escape(&context.revision),
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

    fs::write(output.join("REPORT.md"), markdown)
        .map_err(|error| format!("write Markdown: {error}"))?;
    fs::write(output.join("REPORT.html"), &html).map_err(|error| format!("write HTML: {error}"))?;
    fs::write(output.join("index.html"), html).map_err(|error| format!("write HTML: {error}"))?;
    println!(
        "wrote {} measurements, including all {} required comparisons, to {}/REPORT.md and {}/index.html",
        rows.len(),
        required_comparisons,
        output.display(),
        output.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn complete_comparison_rows(comparisons: &[Comparison<'_>]) -> Vec<Row> {
        let mut rows = Vec::new();
        for comparison in comparisons {
            for shape in comparison.shapes {
                for prefix in std::iter::once(comparison.primary_prefix)
                    .chain(comparison.others.iter().map(|other| other.prefix))
                {
                    let name = format!("{prefix}{shape}");
                    if rows
                        .iter()
                        .any(|row: &Row| row.group == comparison.group && row.name == name)
                    {
                        continue;
                    }
                    rows.push(Row {
                        group: comparison.group.to_owned(),
                        name,
                        relative: "fixture/new/estimates.json".to_owned(),
                        mean: 10.0 + rows.len() as f64,
                        lower: 9.0,
                        upper: 11.0,
                        modified: SystemTime::UNIX_EPOCH,
                    });
                }
            }
        }
        rows
    }

    #[test]
    fn every_configured_comparison_is_required_before_a_report_is_published() {
        let comparisons = comparisons();
        let mut rows = complete_comparison_rows(&comparisons);
        assert!(missing_comparison_rows(&comparisons, &rows).is_empty());

        let removed = rows.remove(0);
        assert_eq!(
            missing_comparison_rows(&comparisons, &rows),
            vec![format!("{}/{}", removed.group, removed.name)]
        );
    }

    #[test]
    fn comparison_html_contains_relative_and_scaling_graphs() {
        let comparisons = comparisons();
        let rows = complete_comparison_rows(&comparisons);
        let html = comparison_html(&comparisons[0], &rows);
        assert!(html.contains("<svg class=\"chart\""));
        assert!(html.contains("Relative time · logarithmic"));
        assert!(html.contains("alternative faster ←"));
        assert!(html.contains("<svg class=\"scaling-chart\""));
    }

    #[test]
    fn relative_chart_keeps_names_values_and_bars_in_separate_columns() {
        let comparisons = comparisons();
        let rows = complete_comparison_rows(&comparisons);
        let html = comparison_chart_html(&comparisons[0], &rows);

        assert!(html.contains("x=\"150.0\" y=\"34.0\" text-anchor=\"end\" class=\"series-label\""));
        assert!(html.contains("x=\"240.0\" y=\"33.0\" text-anchor=\"end\" class=\"bar-label\""));
        assert!(html.contains("x=\"150.0\" y=\"54.0\" text-anchor=\"end\" class=\"series-label\""));
        assert!(html.contains("<rect x=\"578.0\""));
    }

    #[test]
    fn very_small_ratios_are_not_rounded_to_zero() {
        assert_eq!(format_ratio(0.000_14), "1.4e-4×");
        assert_eq!(format_ratio(0.006_2), "0.006×");
        assert_eq!(format_ratio(1.0), "1.00×");
    }

    #[test]
    fn estimates_older_than_the_current_run_marker_are_excluded() {
        let comparisons = comparisons();
        let mut rows = complete_comparison_rows(&comparisons);
        for row in &mut rows {
            row.modified = SystemTime::UNIX_EPOCH + Duration::from_secs(2);
        }
        rows[0].modified = SystemTime::UNIX_EPOCH;

        retain_rows_modified_since(&mut rows, SystemTime::UNIX_EPOCH + Duration::from_secs(1));

        assert_eq!(rows.len(), required_comparison_count(&comparisons) - 1);
    }
}
