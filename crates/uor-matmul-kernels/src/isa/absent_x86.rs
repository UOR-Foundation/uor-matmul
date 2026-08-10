//! The x86-64 kernels, on a target that is not x86-64.
//!
//! Every predicate answers `false`, so `spec.rs` can list what exists once
//! rather than once per target. The specs still have to *resolve*, because the
//! list is one expression; they resolve to the portable reference, which is the
//! only value that could not be wrong if the predicates ever lied.

use crate::isa::portable;
use crate::spec::KernelSpec;
use crate::table::TableSpec;

/// Is AVX2 available? Never, on a target that is not x86-64.
pub fn avx2_available() -> bool {
    false
}

/// Is AVX-512 VNNI available? Never, on a target that is not x86-64.
pub fn avx512vnni_available() -> bool {
    false
}

/// Unreachable: [`avx2_available`] is `false`.
pub const AVX2_I8_I32: KernelSpec<i8, i32> = portable::I8_I32;
/// Unreachable: [`avx2_available`] is `false`.
pub const AVX2_I16_I64: KernelSpec<i16, i64> = portable::I16_I64;
/// Absent here; the reference sequence carries this alphabet.
pub const AVX2_I16_I64_FULL: KernelSpec<i16, i64> = portable::I16_I64;
/// Unreachable: [`avx2_available`] is `false`.
pub const AVX2_I32_I64: KernelSpec<i32, i64> = portable::I32_I64;
/// Unreachable: [`avx2_available`] is `false`.
pub const AVX2_I32_MOD: KernelSpec<i32, i32> = portable::I32_MOD;
/// Unreachable: [`avx2_available`] is `false`.
pub const AVX2_I16_MOD: KernelSpec<i16, i32> = portable::I16_MOD;
/// Unreachable: [`avx512vnni_available`] is `false`.
pub const AVX512_DPWSSD_I8_I32: KernelSpec<i8, i32> = portable::I8_I32;
/// Unreachable: [`avx512vnni_available`] is `false`.
pub const AVX512_DPBUSD_I8_I32: KernelSpec<i8, i32> = portable::I8_I32;
/// Unreachable: [`avx512vnni_available`] is `false`.
pub const AVX512_LOOKUP_I8_I32: KernelSpec<i8, i32> = portable::I8_I32;

/// Absent here; the reference reduce sequence carries this family.
pub const AVX2_R_I8_I32: KernelSpec<i8, i32> = portable::R_I8_I32;
/// Absent here; the reference reduce sequence carries this family.
pub const AVX2_R_I16_I64: KernelSpec<i16, i64> = portable::R_I16_I64;
/// Absent here; the reference reduce sequence carries this family.
pub const AVX2_R_I16_MOD: KernelSpec<i16, i32> = portable::R_I16_MOD;
/// Absent here; the reference reduce sequence carries this family.
pub const AVX2_R_I32_I64: KernelSpec<i32, i64> = portable::R_I32_I64;
/// Absent here; the reference reduce sequence carries this family.
pub const AVX2_R_I32_MOD: KernelSpec<i32, i32> = portable::R_I32_MOD;

/// Absent here; the reference reduce sequence carries this family.
pub const AVX2_R_I8_I32_1: KernelSpec<i8, i32> = portable::R1_I8_I32;
/// Unreachable: [`avx2_available`] is `false`.
pub const AVX2_LOOKUP_R_I8_I32: KernelSpec<i8, i32> = portable::R_I8_I32;
/// Unreachable: [`avx2_available`] is `false`.
pub const AVX2_LOOKUP_R_I8_I32_1: KernelSpec<i8, i32> = portable::R1_I8_I32;
/// Absent here; the reference reduce sequence carries this family.
pub const AVX2_R_I16_I64_1: KernelSpec<i16, i64> = portable::R1_I16_I64;
/// Absent here; the reference reduce sequence carries this family.
pub const AVX2_R_I16_MOD_1: KernelSpec<i16, i32> = portable::R1_I16_MOD;
/// Absent here; the reference reduce sequence carries this family.
pub const AVX2_R_I32_I64_1: KernelSpec<i32, i64> = portable::R1_I32_I64;
/// Absent here; the reference reduce sequence carries this family.
pub const AVX2_R_I32_MOD_1: KernelSpec<i32, i32> = portable::R1_I32_MOD;

/// Absent here; the reference sequence carries this family.
pub const AVX2_I8_I32_M1: KernelSpec<i8, i32> = portable::I8_I32;

/// Absent here; the reference stands in its place.
pub const AVX2_I8_I32_N8: KernelSpec<i8, i32> = portable::I8_I32;

/// Absent here; the reference stands in its place.
pub const AVX2_I8_I32_M1_N8: KernelSpec<i8, i32> = portable::I8_I32;
/// Unreachable: [`avx2_available`] is `false`.
pub const AVX2_LOOKUP_I8_I32: KernelSpec<i8, i32> = portable::I8_I32;
/// Unreachable: [`avx2_available`] is `false`.
pub const AVX2_LOOKUP_I8_I32_M1: KernelSpec<i8, i32> = portable::I8_I32;
/// Unreachable: [`avx2_available`] is `false`.
pub const AVX2_LOOKUP_I8_I32_N8: KernelSpec<i8, i32> = portable::I8_I32;
/// Unreachable: [`avx2_available`] is `false`.
pub const AVX2_LOOKUP_I8_I32_M1_N8: KernelSpec<i8, i32> = portable::I8_I32;
/// Absent here; the reference sequence carries this family.
pub const AVX2_I16_I64_M1: KernelSpec<i16, i64> = portable::I16_I64;
/// Absent here; the reference sequence carries this family.
pub const AVX2_I16_I64_FULL_M1: KernelSpec<i16, i64> = portable::I16_I64;
/// Absent here; the reference sequence carries this family.
pub const AVX2_I32_I64_M1: KernelSpec<i32, i64> = portable::I32_I64;
/// Absent here; the reference sequence carries this family.
pub const AVX2_I32_MOD_M1: KernelSpec<i32, i32> = portable::I32_MOD;
/// Absent here; the reference sequence carries this family.
pub const AVX2_I16_MOD_M1: KernelSpec<i16, i32> = portable::I16_MOD;

/// Unreachable: [`avx2_available`] is `false`.
pub const AVX2_TROP_I16: KernelSpec<i16, i16> = portable::TROP_I16;
/// Unreachable: [`avx2_available`] is `false`.
pub const AVX2_TROP_I16_M1: KernelSpec<i16, i16> = portable::TROP_I16;

// ---------------------------------------------------------------------------
// The table sequences (§7.3)
// ---------------------------------------------------------------------------

/// The `i8` table sequence. Absent here; the reference carries this family.
///
/// This module is compiled only when x86 is not the target. Returning `None`
/// records that target fact; the x86 build supplies its SIMD sequence, while
/// the portable reference remains the complete sequence on this host.
pub fn avx2_table_i8_i32(_rows: usize, _group: usize) -> Option<TableSpec<i8, i32>> {
    None
}

/// The `i16` table sequence. Absent here; the reference carries this family.
///
/// This module is compiled only when x86 is not the target. Returning `None`
/// records that target fact; the x86 build supplies its SIMD sequence, while
/// the portable reference remains the complete sequence on this host.
pub fn avx2_table_i16_i64(_rows: usize, _group: usize) -> Option<TableSpec<i16, i64>> {
    None
}

/// The `i32` table sequence in `Z/2^32`. Absent here; the reference carries
/// this family, exactly as it does the exact lane's.
pub fn avx2_table_i32_mod32(
    _rows: usize,
    _group: usize,
) -> Option<TableSpec<i32, crate::table::Mod32>> {
    None
}

/// The bound-1 `i8` table sequence. Absent here; the portable bound-1 build
/// carries this alphabet.
pub fn avx2_table_i8_i32_bound1(_rows: usize, _group: usize) -> Option<TableSpec<i8, i32>> {
    None
}

/// Is AVX-512 available? Never, on a target that is not x86-64.
pub fn avx512_available() -> bool {
    false
}

/// The `i8` table sequence at 512 bits, on a target that is not x86-64.
pub fn avx512_table_i8_i32(_rows: usize, _group: usize) -> Option<TableSpec<i8, i32>> {
    None
}

/// The `i16` table sequence at 512 bits, on a target that is not x86-64.
pub fn avx512_table_i16_i64(_rows: usize, _group: usize) -> Option<TableSpec<i16, i64>> {
    None
}
