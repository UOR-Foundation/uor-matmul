//! The x86-64 kernels, on a target that is not x86-64.
//!
//! Every predicate answers `false`, so `spec.rs` can list what exists once
//! rather than once per target. The specs still have to *resolve*, because the
//! list is one expression; they resolve to the portable reference, which is the
//! only value that could not be wrong if the predicates ever lied.

use crate::isa::portable;
use crate::spec::KernelSpec;

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
