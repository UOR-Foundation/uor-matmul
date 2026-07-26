//! probe
use uor_matmul_kernels::{available_table_i8, TableSpec};

#[test]
fn slab_below_rows_gather() {
    let specs: Vec<TableSpec<i8, i32>> = available_table_i8(16, 1).collect();
    println!("specs: {}", specs.len());
    for s in &specs {
        println!("backend {:?} rows {} group {}", s.backend, s.rows, s.group);
    }
    // depth 1, slab 1 (a power of two), stack of 1 lane word, lane of 16.
    let stack = vec![7i32; 1];
    let off = vec![0u32; 1];
    let mut lane = vec![0i32; 16];
    for s in &specs {
        println!("calling gather on {:?}", s.backend);
        s.gather(1, 1, &stack, &off, &mut lane);
        println!("  -> {:?}", &lane[..4]);
    }
}

#[test]
fn slab_below_rows_gather_codes() {
    let specs: Vec<TableSpec<i8, i32>> = available_table_i8(16, 1).collect();
    let stack = vec![7i32; 1];
    let codes = vec![0xffffu16; 1];
    let mut lane = vec![0i32; 16];
    for s in &specs {
        println!("calling gather_codes on {:?}", s.backend);
        s.gather_codes(1, 1, &stack, &codes, 1, &mut lane);
        println!("  -> {:?}", &lane[..4]);
    }
}
