//! `CD-19`: the caller-owned full float workspace reaches the same exact
//! bridge bytes as the panel-only float entry.

use uor_matmul::{
    slice, suggested_accumulators, suggested_bridge_scaled, suggested_float_panels,
    suggested_scratch, PackedCode, Shape,
};

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

    let mut full_a = vec![PackedCode::default(); pa_len];
    let mut full_b = vec![PackedCode::default(); pb_len];
    let mut scaled = vec![0i32; suggested_bridge_scaled(shape)];
    let mut panels = vec![0i32; suggested_scratch(shape)];
    let mut accumulators = vec![0i128; suggested_accumulators(shape)];
    let mut full_out = vec![0.0f32; m * n];
    slice::gemm_float_full(
        m,
        k,
        n,
        &a,
        &b,
        &mut full_out,
        &mut full_a,
        &mut full_b,
        &mut scaled,
        &mut panels,
        &mut accumulators,
    )
    .expect("the full-workspace product exists");

    assert_eq!(
        panel_out
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        full_out
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
}
