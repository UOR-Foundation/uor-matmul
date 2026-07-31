//! Scratch is an offer, never a requirement (S13, `CD-04`).
//!
//! Too little, none at all, or a strange amount all produce the same bytes;
//! only the traversal differs. There is no scratch error in the library's error
//! surface, because there is no scratch condition to report.

use core::mem::size_of;

use uor_matmul_core::{AccOf, Accumulator, Alphabet, Backend, Bound, Element, Shape};

/// A panel offer cut in two. Named because the pair is a return type and
/// `(&mut [Alphabet<E, Bd>], &mut [Alphabet<E, Bd>])` written out says nothing the
/// name does not.
pub type PanelSplit<'s, E, Bd> = (&'s mut [Alphabet<E, Bd>], &'s mut [Alphabet<E, Bd>]);

/// Working memory the caller has offered.
///
/// The library never owns this and never grows it. An empty offer is
/// well-formed and selects the streaming traversal, which is what makes the
/// library run on a target whose RAM cannot hold one decoded row.
///
/// # Two offers, not one
///
/// The panel buffer is where decoded operands are packed. The *accumulator*
/// buffer is the one that removes a classical constraint, so it deserves saying
/// why it exists.
///
/// A classical GEMM chunks the reduction so that its panels fit cache, and
/// accumulates the chunks into `C` as it goes. It can do that because its
/// accumulator and its output are the same width --- and the price is that the
/// answer depends on the chunking, which is why no two classical `sgemm`
/// implementations agree bit for bit.
///
/// This library cannot write partial sums into `C`, because `C` is the *encoded*
/// output and encoding a partial sum would round it. So without somewhere to keep
/// the exact partial sums, the panels must hold the whole of `k` --- which makes
/// the offer grow with the depth, and a caller with an astronomical `k` either
/// supplies an astronomical buffer or gets the unblocked traversal.
///
/// An accumulator buffer is that somewhere. With one, the reduction is chunked to
/// whatever the cache holds while every partial sum stays exact and full width,
/// and the offer stops growing with `k`. This is the property the exact
/// accumulator buys and a classical library cannot have: the sum is
/// order-independent, so it may be split any way the machine prefers and
/// recombined with no consequence at all.
///
/// Offering one is optional, like every other offer. Offering none gives the same
/// bytes from the full-depth traversal (`CD-04`, `CD-10`).
#[derive(Debug)]
pub struct Scratch<'s, E: Element, Bd: Bound> {
    buffer: &'s mut [Alphabet<E, Bd>],
    accumulators: &'s mut [AccOf<E>],
}

impl<'s, E: Element, Bd: Bound> Scratch<'s, E, Bd> {
    /// Offer a panel buffer.
    pub fn new(buffer: &'s mut [Alphabet<E, Bd>]) -> Self {
        Self {
            buffer,
            accumulators: &mut [],
        }
    }

    /// Offer a panel buffer and a block of exact accumulators.
    ///
    /// The accumulators let the reduction be chunked at whatever depth the panels
    /// fit the cache, instead of the panels having to hold the whole of `k`. See
    /// the type's documentation for why that is a property this library has and a
    /// classical one does not.
    pub fn with_accumulators(
        buffer: &'s mut [Alphabet<E, Bd>],
        accumulators: &'s mut [AccOf<E>],
    ) -> Self {
        Self {
            buffer,
            accumulators,
        }
    }

    /// Offer nothing.
    ///
    /// Not a degraded mode and not a fallback: the same identity, walked in a
    /// different order (R13).
    pub fn none() -> Scratch<'static, E, Bd> {
        Scratch {
            buffer: &mut [],
            accumulators: &mut [],
        }
    }

    /// How many exact accumulators were offered.
    pub fn accumulators(&self) -> usize {
        self.accumulators.len()
    }

    /// The first `n` accumulators, zeroed. The start of a block's reduction.
    pub fn take_accumulators(&mut self, n: usize) -> &mut [AccOf<E>] {
        let block = self.keep_accumulators(n);
        block.fill(<AccOf<E> as Accumulator>::ZERO);
        block
    }

    /// The first `n` accumulators, as they stand. The continuation of one.
    pub fn keep_accumulators(&mut self, n: usize) -> &mut [AccOf<E>] {
        let n = n.min(self.accumulators.len());
        &mut self.accumulators[..n]
    }

    /// Both offers at once: `panel` elements of panel room and `accs`
    /// accumulators.
    ///
    /// They are separate buffers, so a traversal that uses both needs them
    /// together rather than one after the other.
    pub fn split(
        &mut self,
        panel: usize,
        accs: usize,
    ) -> (&mut [Alphabet<E, Bd>], &mut [AccOf<E>]) {
        let p = panel.min(self.buffer.len());
        let a = accs.min(self.accumulators.len());
        (&mut self.buffer[..p], &mut self.accumulators[..a])
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

    /// The offer cut in two: the first `at` elements, and the rest.
    ///
    /// One traversal expressed as another needs both halves at once --- the first
    /// for what it materializes, the second handed on as the inner traversal's own
    /// offer. Clamped like every other response to an amount, so a short offer
    /// gives a short head and an empty tail rather than a failure.
    pub fn split_panel(&mut self, at: usize) -> PanelSplit<'_, E, Bd> {
        let at = at.min(self.buffer.len());
        self.buffer.split_at_mut(at)
    }
}

/// How much scratch would let the blocked traversal run at its intended panel
/// size for this shape.
///
/// A *query*, not a requirement. A caller who offers less gets the same bytes
/// from a shorter panel, and a caller who offers none gets the same bytes from
/// the streaming traversal (`CD-04`).
///
/// This is the larger of two wants. The first is the full-depth traversal's:
/// one block of each operand at the full depth, `MC` rows of `A` and `NC`
/// columns of `B`. That is what the blocked traversal reuses --- the `B` block
/// across every row block, and the `A` block across every column panel --- and
/// it is never more than the operands themselves. A caller who wants less
/// offers less and gets the chunked traversal, at the same bytes (`CD-04`,
/// `CD-10`).
///
/// The second is the sub-cubic recursion's, when the shape's size and
/// evenness admit a level: the block sums it materializes plus the base
/// case's own panel. The query cannot see the declared bound, so it cannot
/// know whether the alphabet leaves the sums headroom; where it does not ---
/// and on every lane but `i32`, which is the only lane with a measured
/// crossover --- the recursion declines and the offer buys the same bytes
/// from the cubic walk (`CD-10`, `CD-21`).
pub fn suggested_scratch(shape: Shape) -> usize {
    use uor_matmul_core::generated::blocking;
    // One block of each operand at the full depth: `MC` rows of A and `NC`
    // columns of B. That is what the blocked traversal reuses --- the B block
    // across every row block, and the A block across every column panel --- and
    // it is the amount at which neither operand is repacked more than the
    // blocking says.
    //
    // It is never more than the operands themselves: `MC <= m` and `NC <= n`
    // after the clamps, so this is at most `k * (m + n)`. A caller who wants
    // less offers less and gets the chunked traversal, at the same bytes
    // (`CD-04`, `CD-10`).
    let cubic = shape
        .k
        .saturating_mul(shape.m.min(blocking::MC) + shape.n.min(blocking::NC)); // R3-ok: a scratch size query
    let recursion = crate::strassen::wants(shape).0.min(usize::MAX as u128) as usize;
    cubic.max(recursion)
}

/// How many exact accumulators would let the chunked-depth traversal run at its
/// intended block for this shape.
///
/// A *query*, like [`suggested_scratch`]. Zero is a valid offer and gives the
/// same bytes.
///
/// This is the larger of two wants. The first is the depth-chunked
/// traversal's: the output block it keeps exact partial sums for, `MC` rows
/// by `NC` columns clamped to the shape, which does not grow with `k` --- the
/// whole point, since a caller with an astronomical depth offers this once
/// and the depth is chunked to fit the cache rather than the buffer.
///
/// The second is the sub-cubic recursion's: its product temporaries are exact
/// sums, held in the accumulator's width while the level below runs. The same
/// bound-blindness applies as in [`suggested_scratch`]: where the declared
/// alphabet admits no level, the offer buys the same bytes from the cubic
/// walk (`CD-21`).
pub fn suggested_accumulators(shape: Shape) -> usize {
    use uor_matmul_core::generated::blocking;
    // Paired with a panel offer of `KC * (MC + NC)` rather than the full-depth
    // one, this is the whole of what the depth-chunked traversal needs --- and
    // neither term grows with `k`.
    let chunked = if shape.k <= blocking::KC {
        // Nothing to chunk: the full-depth traversal already holds every partial
        // sum in a register.
        0
    } else {
        shape
            .m
            .min(blocking::MC)
            .saturating_mul(shape.n.min(blocking::NC)) // R3-ok: a scratch size query
    };
    let recursion = crate::strassen::wants(shape).1.min(usize::MAX as u128) as usize;
    chunked.max(recursion)
}

// ---------------------------------------------------------------------------
// The workspace queries, in bytes
// ---------------------------------------------------------------------------

/// Which traversal an offer buys at a shape.
///
/// The names are the traversal's, not a quality tier's: every one computes the
/// same bytes (`CD-01`, `CD-04`), and which runs is decided by the offer and
/// the declarations, never the data (R13).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Chunking {
    /// The panels hold the whole depth: one pass over the packed panels, the
    /// partial sums in registers. What [`suggested_scratch`] buys wherever the
    /// lane reaches the depth.
    FullDepth,
    /// The reduction is walked in chunks of the named depth, a whole number of
    /// the kernel's `k`-groups; the exact partial sums live in the accumulator
    /// offer between chunks. Bounded independent of `k` --- this is the plan
    /// whose cost stops growing with the depth.
    Chunked {
        /// The chunk depth the offer pays for.
        chunk: usize,
    },
    /// One packed group of panel room: both panels repacked for every output
    /// tile. The smallest offer that buys a packed traversal at all.
    PerTile,
    /// Nothing packed: the streaming reference. Zero offer, the same bytes
    /// (S13).
    Streaming,
}

impl core::fmt::Display for Chunking {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::FullDepth => f.write_str("full-depth"),
            Self::Chunked { chunk } => write!(f, "chunked at {chunk}"),
            Self::PerTile => f.write_str("per-tile"),
            Self::Streaming => f.write_str("streaming"),
        }
    }
}

/// One workspace plan, in bytes, with the traversal it buys.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct WorkspacePlan {
    /// Panel room, in bytes --- `Alphabet<E, Bd>` elements, per call.
    pub panels: usize,
    /// Exact-accumulator room, in bytes --- `AccOf<E>` elements, per call,
    /// shared across depth chunks: the block is zeroed per output block, not
    /// per chunk, so one block serves a reduction of any depth.
    pub accumulators: usize,
    /// `panels + accumulators`.
    pub total: usize,
    /// The traversal this offer buys.
    pub chunking: Chunking,
}

/// What a shape wants, in bytes: the suggested plan and the bounded one.
///
/// The tabulated traversal's offers --- table and lanes --- are a separate
/// offer family with its own queries ([`crate::suggested_tabulation`] and its
/// siblings) and are not restated here.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct WorkspaceReport {
    /// The suggested offer: [`suggested_scratch`] panels plus
    /// [`suggested_accumulators`] accumulators. Full-depth where the lane
    /// reaches the reduction; the panel term grows with `k`.
    pub suggested: WorkspacePlan,
    /// The largest plan whose cost does not grow with `k`: one chunk of the
    /// declared blocking (`KC`) of panels plus the accumulator block. Where
    /// `k <= KC` chunking never engages and this *is* the suggested plan's
    /// cubic half --- the recursion's wants, where a lane admits one, remain
    /// the suggested offer's business (`CD-21`).
    pub bounded: WorkspacePlan,
}

/// The chunking label the driver's own arithmetic gives an offer.
///
/// This is `run()`'s decision read off the same terms it reads --- the
/// resolved tile spec, the lane depth at the declared bound, the declared
/// blocking --- with one deliberate blindness: the strides. A layout the
/// driver can borrow wholesale (a contiguous reduce panel over the whole
/// extent) runs with no panels at all, and the query cannot see that from a
/// [`Shape`], so the label is the plan the offer always buys, never the best a
/// lucky layout might.
fn chunking_of<E>(
    shape: Shape,
    spec: &uor_matmul_kernels::KernelSpec<E, E::Exact>,
    lane: usize,
    panels: usize,
    accumulators: usize,
) -> Chunking
where
    E: crate::kernel::Kernelized,
{
    use uor_matmul_core::generated::blocking;

    let (mr, nr) = (spec.mr, spec.nr);
    let per_step = mr + nr;
    let pad_to = spec.k_group.max(1);

    // `run`'s full-depth question, with the borrow conservatively false.
    let kpad = shape.k.div_ceil(pad_to) * pad_to;
    let kc = (panels / per_step).min(lane) / pad_to * pad_to;
    let budget = panels.checked_div(kpad).unwrap_or(0);
    if shape.k <= lane && shape.k <= kc && budget >= per_step {
        return Chunking::FullDepth;
    }
    // `run`'s chunked gate, and `run_chunked_depth`'s chunk arithmetic.
    if accumulators >= mr * nr && panels >= per_step * pad_to && shape.k > pad_to {
        let chunk = blocking::KC.min(lane).min(panels / per_step).max(1) / pad_to * pad_to;
        // `kc == 0` is the driver's decline to the streaming traversal ---
        // written against the same arithmetic coincidence it guards there.
        return if chunk == 0 {
            Chunking::Streaming
        } else {
            Chunking::Chunked { chunk }
        };
    }
    if panels >= per_step * pad_to {
        return Chunking::PerTile;
    }
    Chunking::Streaming
}

/// What the driver would use at this shape, in bytes, term by term.
///
/// A *query*, like [`suggested_scratch`]: nothing here is a requirement, and
/// every offer from none to the suggested total returns the same bytes
/// (`CD-04`, `CD-10`). The terms are derived from the model's blocking
/// constants and the resolved kernel's own declarations --- `KC`, `MC`, `NC`,
/// the tile's `mr`, `nr` and `k_group`, and the lane's depth at the declared
/// bound --- not restated numerals.
pub fn workspace_report<E, Bd>(shape: Shape) -> WorkspaceReport
where
    E: crate::kernel::Kernelized,
    Bd: Bound,
{
    use uor_matmul_core::generated::blocking;

    let spec = E::exact_spec(Backend::Auto, Bd::VALUE, shape.m);
    let lane = spec.lane_depth(Bd::VALUE);
    let e = size_of::<Alphabet<E, Bd>>();
    let a = size_of::<AccOf<E>>();

    let plan = |panels: usize, accumulators: usize| WorkspacePlan {
        panels: panels.saturating_mul(e), // R3-ok: a workspace size question
        accumulators: accumulators.saturating_mul(a), // R3-ok: a workspace size question
        total: panels
            .saturating_mul(e) // R3-ok: a workspace size question
            .saturating_add(accumulators.saturating_mul(a)), // R3-ok: a workspace size question
        chunking: chunking_of::<E>(shape, &spec, lane, panels, accumulators),
    };

    let block = shape
        .m
        .min(blocking::MC)
        .saturating_add(shape.n.min(blocking::NC)); // R3-ok: a workspace size question
    let bounded_panels = shape.k.min(blocking::KC).saturating_mul(block); // R3-ok: a workspace size question
    let bounded_accumulators = if shape.k > blocking::KC {
        shape
            .m
            .min(blocking::MC)
            .saturating_mul(shape.n.min(blocking::NC)) // R3-ok: a workspace size question
    } else {
        0
    };

    WorkspaceReport {
        suggested: plan(suggested_scratch(shape), suggested_accumulators(shape)),
        bounded: plan(bounded_panels, bounded_accumulators),
    }
}

/// The smallest offer that buys a packed traversal at all: one packed group
/// of panel room, in bytes.
///
/// Below it the streaming reference runs, at the same bytes (S13).
pub fn minimum_workspace<E, Bd>(shape: Shape) -> WorkspacePlan
where
    E: crate::kernel::Kernelized,
    Bd: Bound,
{
    let spec = E::exact_spec(Backend::Auto, Bd::VALUE, shape.m);
    let lane = spec.lane_depth(Bd::VALUE);
    let e = size_of::<Alphabet<E, Bd>>();
    let panels = (spec.mr + spec.nr).saturating_mul(spec.k_group.max(1)); // R3-ok: a workspace size question
    WorkspacePlan {
        panels: panels.saturating_mul(e), // R3-ok: a workspace size question
        accumulators: 0,
        total: panels.saturating_mul(e), // R3-ok: a workspace size question
        chunking: chunking_of::<E>(shape, &spec, lane, panels, 0),
    }
}

/// The largest plan inside a caller's byte budget, and the factorization it
/// lands.
///
/// The ladder is the report's named plans: the suggested plan whole, then the
/// bounded one, then one packed group, then the streaming reference at no
/// offer. A budget between two rungs buys the lower one --- offering the rest
/// is never wrong, only idle, and the bytes do not move either way (`CD-10`).
pub fn workspace_for_budget<E, Bd>(shape: Shape, bytes: usize) -> WorkspacePlan
where
    E: crate::kernel::Kernelized,
    Bd: Bound,
{
    let report = workspace_report::<E, Bd>(shape);
    let minimum = minimum_workspace::<E, Bd>(shape);
    if bytes >= report.suggested.total {
        report.suggested
    } else if bytes >= report.bounded.total {
        report.bounded
    } else if bytes >= minimum.total {
        minimum
    } else {
        WorkspacePlan {
            panels: 0,
            accumulators: 0,
            total: 0,
            chunking: Chunking::Streaming,
        }
    }
}
