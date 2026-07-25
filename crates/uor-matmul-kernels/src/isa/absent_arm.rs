//! The AArch64 kernels, on a target that is not AArch64. See
//! [`crate::isa::x86`]'s absent twin for why the constants still resolve.

use crate::isa::portable;
use crate::spec::KernelSpec;
use crate::table::TableSpec;

/// Is NEON available? Never, on a target that is not AArch64.
pub fn neon_available() -> bool {
    false
}

/// Is the dot-product extension available? Never, on a target that is not
/// AArch64.
pub fn dotprod_available() -> bool {
    false
}

/// Unreachable: [`neon_available`] is `false`.
pub const NEON_I8_I32: KernelSpec<i8, i32> = portable::I8_I32;
/// Unreachable: [`dotprod_available`] is `false`.
pub const NEON_DOTPROD_I8_I32: KernelSpec<i8, i32> = portable::I8_I32;

/// Absent here; the reference reduce sequence carries this family.
pub const NEON_R_I8_I32: KernelSpec<i8, i32> = portable::R_I8_I32;
/// Absent here; the reference reduce sequence carries this family.
pub const NEON_DOTPROD_R_I8_I32: KernelSpec<i8, i32> = portable::R_I8_I32;

/// Absent here; the reference reduce sequence carries this family.
pub const NEON_R_I8_I32_1: KernelSpec<i8, i32> = portable::R1_I8_I32;
/// Absent here; the reference reduce sequence carries this family.
pub const NEON_DOTPROD_R_I8_I32_1: KernelSpec<i8, i32> = portable::R1_I8_I32;

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
pub fn neon_table_i8_i32(_rows: usize, _group: usize) -> Option<TableSpec<i8, i32>> {
    None
}
