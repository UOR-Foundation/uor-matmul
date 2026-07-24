//! One module per ISA. Each exports a [`crate::KernelSpec`] and nothing else.

pub mod portable;

#[cfg(target_arch = "x86_64")]
pub mod avx2;
#[cfg(target_arch = "x86_64")]
pub mod avx512vnni;

#[cfg(target_arch = "aarch64")]
pub mod neon;
#[cfg(target_arch = "aarch64")]
pub mod neon_dotprod;

#[cfg(target_arch = "wasm32")]
pub mod wasm_simd128;
