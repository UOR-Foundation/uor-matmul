//! `CD-19`: the caller-owned full float workspace reaches the same exact Atlas
//! bytes as the panel-only float entry. Workspace is an in/out offer: its
//! initial bytes are not data and its post-call residue has no public meaning.

use uor_matmul::{
    slice, suggested_accumulators, suggested_bridge_scaled, suggested_float_panels,
    suggested_scratch, PackedCode, Shape,
};

fn full_product_with_poison(
    shape: Shape,
    a: &[f32],
    b: &[f32],
    code_poison: PackedCode,
    narrow_poison: i32,
    exact_poison: i128,
) -> Vec<f32> {
    let (pa_len, pb_len) = suggested_float_panels(shape);
    let mut full_a = vec![code_poison; pa_len];
    let mut full_b = vec![code_poison; pb_len];
    let mut scaled = vec![narrow_poison; suggested_bridge_scaled(shape).max(3)];
    let mut panels = vec![!narrow_poison; suggested_scratch(shape).max(3)];
    let mut accumulators = vec![exact_poison; suggested_accumulators(shape).max(3)];
    let addresses = (
        full_a.as_ptr(),
        full_b.as_ptr(),
        scaled.as_ptr(),
        panels.as_ptr(),
        accumulators.as_ptr(),
    );
    let mut output = vec![0.0f32; shape.m * shape.n];
    slice::gemm_float_full(
        shape.m,
        shape.k,
        shape.n,
        a,
        b,
        &mut output,
        &mut full_a,
        &mut full_b,
        &mut scaled,
        &mut panels,
        &mut accumulators,
    )
    .expect("the full-workspace product exists");

    assert_eq!(
        addresses,
        (
            full_a.as_ptr(),
            full_b.as_ptr(),
            scaled.as_ptr(),
            panels.as_ptr(),
            accumulators.as_ptr(),
        ),
        "the operation borrows caller storage in place"
    );
    output
}

#[test]
fn the_full_float_workspace_is_byte_identical_to_the_panel_entry_cd_19() {
    let (m, k, n) = (16usize, 1024usize, 8usize);
    let shape = Shape { m, k, n };
    let a: Vec<f32> = (0..m * k)
        .map(|i| {
            let exponent = (i as i32 % 3) - 1;
            (8_388_608 + (i % 8_388_607)) as f32 * 2.0f32.powi(exponent)
        })
        .collect();
    let b: Vec<f32> = (0..k * n)
        .map(|i| {
            let exponent = (i as i32 % 4) - 2;
            (8_388_608 + (i % 8_388_607)) as f32 * 2.0f32.powi(exponent)
        })
        .collect();

    let (pa_len, pb_len) = suggested_float_panels(shape);
    let mut panel_a = vec![PackedCode::default(); pa_len];
    let mut panel_b = vec![PackedCode::default(); pb_len];
    let mut panel_out = vec![0.0f32; m * n];
    slice::gemm_float(m, k, n, &a, &b, &mut panel_out, &mut panel_a, &mut panel_b)
        .expect("the panel product exists");

    let first = full_product_with_poison(
        shape,
        &a,
        &b,
        PackedCode {
            mantissa: 0,
            exp: 0,
            _pad: i32::from_ne_bytes([0, 0, 0, 5]),
        },
        0x1357_2468,
        0x1357_2468_1357_2468,
    );
    let second = full_product_with_poison(
        shape,
        &a,
        &b,
        PackedCode {
            mantissa: 0,
            exp: 0,
            _pad: i32::from_ne_bytes([0, 0, 0, 3]),
        },
        0x2468_1357,
        -0x2468_1357_2468_1357,
    );

    assert_eq!(
        panel_out
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        first
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "the first poison pattern changed the product"
    );
    assert_eq!(
        first
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        second
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "workspace initial bytes became operand data"
    );
}
