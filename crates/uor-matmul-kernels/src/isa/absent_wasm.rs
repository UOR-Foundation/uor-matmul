//! The WebAssembly kernels, on a target that is not wasm32. See
//! [`crate::isa::x86`]'s absent twin for why the constant still resolves.

use crate::isa::portable;
use crate::spec::KernelSpec;
use crate::table::TableSpec;

/// Is SIMD128 available? Never, on a target that is not wasm32.
pub fn simd128_available() -> bool {
    false
}

/// Unreachable: [`simd128_available`] is `false`.
pub const SIMD128_I8_I32: KernelSpec<i8, i32> = portable::I8_I32;
/// Unreachable: [`simd128_available`] is `false`.
pub const SIMD128_LOOKUP_I8_I32: KernelSpec<i8, i32> = portable::I8_I32;

/// Unreachable: [`simd128_available`] is `false`.
pub const SIMD128_SWAR_I8_I32: KernelSpec<i8, i32> = portable::I8_I32;

/// Absent here; the reference reduce sequence carries this family.
pub const SIMD128_R_I8_I32: KernelSpec<i8, i32> = portable::R_I8_I32;

/// Absent here; the reference reduce sequence carries this family.
pub const SIMD128_R_I8_I32_1: KernelSpec<i8, i32> = portable::R1_I8_I32;
/// Unreachable: [`simd128_available`] is `false`.
pub const SIMD128_LOOKUP_R_I8_I32: KernelSpec<i8, i32> = portable::R_I8_I32;
/// Unreachable: [`simd128_available`] is `false`.
pub const SIMD128_LOOKUP_R_I8_I32_1: KernelSpec<i8, i32> = portable::R1_I8_I32;

/// Unreachable: [`simd128_available`] is `false`.
pub const SIMD128_TROP_I16: KernelSpec<i16, i16> = portable::TROP_I16;

// ---------------------------------------------------------------------------
// The table sequences (§7.3)
// ---------------------------------------------------------------------------

/// The `i8` table sequence. Absent here; the reference carries this family.
///
/// A table's column loop is integer adds and a masked index, so every SIMD
/// target has a sequence for it and this absence is unfinished work rather
/// than a property of the hardware. It is written as `None` rather than as a
/// slower body so that the reference is the one that runs and `CB-*` compares
/// against one sequence and not two.
pub fn simd128_table_i8_i32(_rows: usize, _group: usize) -> Option<TableSpec<i8, i32>> {
    None
}

/// The `i16` table sequence, on a target that is not this one.
pub fn simd128_table_i16_i64(_rows: usize, _group: usize) -> Option<TableSpec<i16, i64>> {
    None
}
