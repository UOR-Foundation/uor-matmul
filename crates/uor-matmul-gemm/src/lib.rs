//! The driver: the operation with a conventional GEMM face on it.
//!
//! [`gemm`] returns `()`. Not `Result<(), _>`, not `Result<(), Infallible>`:
//! `()`. The type signature carries the totality claim, and there is no second
//! place for a condition to arise, because every question a classical GEMM
//! answers with an error --- depth, magnitude, alignment, host capability ---
//! has an answer here for every input (R14, C6, `CT-05`).
//!
//! Non-existence of the requested object is reported once, at view
//! construction, by [`uor_matmul_core::Triple::new`] and [`CodedTriple::new`].
//! Those are the library's only two fallible constructors and they share the
//! same two failures.
//!
//! # One method
//!
//! Everything below is *decode, accumulate exactly, encode once* at a different
//! instantiation. The traversals differ in which order they walk the output and
//! how much working memory they use; they do not differ in what they compute,
//! and `CD-05` asserts it byte for byte.
//!
//! # Three entries, one identity
//!
//! [`gemm`] is the reference traversal: portable, never optimized, and the
//! oracle every other factorization is validated against (R6). It stays
//! directly callable under its own name. [`gemm_packed`] is the kernelized
//! traversal: the modular lane, the sub-cubic recursion, and the CG-13 kernel
//! table, each taken exactly when the caller's declarations and the offer
//! admit it and declining to [`gemm`] when they do not. [`gemm_auto`] is
//! [`gemm_packed`] under the name the documented API calls --- the default
//! entry the safe faces reach --- and [`gemm_auto_counted`] is the same call
//! recording which factorization ran, which is what `CD-22` asserts: the
//! kernel table is reached, and every route returns the reference's bytes.

#![no_std]
#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(clippy::disallowed_types)]

// The unit tests below build operands on the heap. That is a property of the
// tests, not of the library: nothing in the shipped code path can allocate,
// which is what `CA-01` witnesses with a counting allocator.
#[cfg(test)]
extern crate std;

pub mod coded;
pub mod collapse;
pub mod driver;
pub mod epilogue;
pub mod float;
pub mod kernel;
pub mod partition;
pub mod scratch;
pub mod strassen;
pub mod tabulated;

pub use coded::{coded_gemm, CodedTriple};
pub use collapse::{gemm_collapsed, suggested_collapse_index, suggested_collapse_rows, Collapse};
pub use driver::{gemm, GemmOptions};
pub use epilogue::{Bias, Epilogue, Linear, PlaceAt, Scaled};
pub use float::{
    gemm_float, gemm_float_bridged, gemm_float_packed, suggested_bridge_scaled,
    suggested_float_panels, SignedPlace,
};
pub use kernel::{
    gemm_auto, gemm_auto_counted, gemm_packed, Kernelized, Route, RouteCensus, RouteLedger,
};
pub use partition::{Partition, Tile};
pub use scratch::{suggested_accumulators, suggested_scratch, PanelSplit, Scratch};
pub use strassen::{gemm_strassen, levels as strassen_levels, strassen_scratch};
pub use tabulated::{
    gather_reference_i32, gather_reference_wide, gemm_tabulated, gemm_tabulated_counted,
    suggested_tabulation, suggested_tabulation_index, suggested_tabulation_lanes,
    suggested_tabulation_panel, tabulation_depth, tabulation_fits, tabulation_pays,
    tabulation_rows, Census, Lane, LaneScale, LaneWord, Ledger, Table, TableSpec, Tabulated,
    TabulatedTriple, Tabulation, Wide,
};
