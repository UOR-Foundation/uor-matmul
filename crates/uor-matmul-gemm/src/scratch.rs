//! Scratch is an offer, never a requirement (S13, `CD-04`).
//!
//! Too little, none at all, or a strange amount all produce the same bytes;
//! only the traversal differs. There is no scratch error in the library's error
//! surface, because there is no scratch condition to report.

use uor_matmul_core::{Alphabet, Bound, IntegerElement, Shape};

/// Working memory the caller has offered.
///
/// The library never owns this and never grows it. An empty offer is
/// well-formed and selects the streaming traversal, which is what makes the
/// library run on a target whose RAM cannot hold one decoded row.
#[derive(Debug)]
pub struct Scratch<'s, E: IntegerElement, Bd: Bound> {
    buffer: &'s mut [Alphabet<E, Bd>],
}

impl<'s, E: IntegerElement, Bd: Bound> Scratch<'s, E, Bd> {
    /// Offer a buffer.
    pub fn new(buffer: &'s mut [Alphabet<E, Bd>]) -> Self {
        Self { buffer }
    }

    /// Offer nothing.
    ///
    /// Not a degraded mode and not a fallback: the same identity, walked in a
    /// different order (R13).
    pub fn none() -> Scratch<'static, E, Bd> {
        Scratch { buffer: &mut [] }
    }

    /// How much was offered.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Was nothing offered?
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// The largest panel this offer supports, capped at `want`.
    ///
    /// This is the whole of the library's response to a scratch amount: a
    /// panel length. A short offer shortens the panel; it never changes the
    /// arithmetic and never fails.
    pub fn panel(&self, want: usize) -> usize {
        want.min(self.buffer.len())
    }

    /// The first `n` elements of the offer, for a panel.
    pub fn take(&mut self, n: usize) -> &mut [Alphabet<E, Bd>] {
        let n = n.min(self.buffer.len());
        &mut self.buffer[..n]
    }
}

/// How much scratch would let the blocked traversal run at its intended panel
/// size for this shape.
///
/// A *query*, not a requirement. A caller who offers less gets the same bytes
/// from a shorter panel, and a caller who offers none gets the same bytes from
/// the streaming traversal (`CD-04`).
pub fn suggested_scratch(shape: Shape) -> usize {
    use uor_matmul_core::generated::blocking;
    // One packed k-panel of B, which is what the blocked traversal reuses
    // across the rows of A. Everything else the traversal needs lives in
    // registers or in the caller's own buffers.
    shape.k.min(blocking::KC) * shape.n.min(blocking::NC)
}
