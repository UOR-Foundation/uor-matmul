//! `CD-21` (build): the sub-cubic recursion --- Winograd's form of Strassen's,
//! on the exact integer lanes --- is byte-identical to the cubic walk at every
//! shape, depth, requested level count, and offer including none, and to the
//! `CX-01` wrapping oracle at every corpus size.
//!
//! Over the integers the recursion uses only add, subtract, and multiply, so
//! the regrouped sum is the same integer the naive loop returns, bit for bit:
//! the cancellation exposure a float library declines Strassen for does not
//! exist here. A level the shape, the bound's headroom, or the offer does not
//! admit is declined, and declining is the cubic walk --- which is what makes
//! "declined" and "cubic" the same bytes rather than two claims.

use uor_matmul::prelude::*;
use uor_matmul::{driver::Scratch, gemm_strassen, strassen_levels, strassen_scratch, Bound, Shape};
use uor_matmul_validate::{
    bytes_equal, oracle_stays_in_range, reference_wrapping_i32, Case, Corpus,
};

const SEED: u64 = 20_260_730;

/// The product under the recursion at `levels` requested levels, with `offer`
/// panel elements and `accs` accumulators on offer.
#[allow(clippy::too_many_arguments)]
fn recursive_i32(
    m: usize,
    k: usize,
    n: usize,
    a: &[i32],
    b: &[i32],
    levels: usize,
    offer: usize,
    accs: usize,
) -> Vec<i32> {
    let mut c = vec![0i32; m * n];
    let mut panel = vec![Alphabet::<i32, Bnd<128>>::ZERO; offer];
    let mut acc_buf = vec![0i128; accs];
    {
        // The fills are widened i8, so `Bnd<128>` is a true declaration ---
        // `as_alphabet` measures it rather than taking it on trust. The full
        // i32 alphabet would leave the recursion no headroom and decline every
        // level, which is not the half of the claim this helper exists to
        // exercise.
        let av = MatView::row_major(
            as_alphabet::<i32, Bnd<128>>(a).expect("the fill is i8"),
            m,
            k,
        )
        .expect("A fits its buffer");
        let bv = MatView::row_major(
            as_alphabet::<i32, Bnd<128>>(b).expect("the fill is i8"),
            k,
            n,
        )
        .expect("B fits its buffer");
        let cv = MatViewMut::row_major(&mut c, m, n).expect("C fits its buffer");
        let mut t = Triple::new(av, bv, cv).expect("the product exists");
        gemm_strassen(
            &mut t,
            &Linear::OVERWRITE,
            GemmOptions {
                encode: EncodeMode::Wrapping,
                ..Default::default()
            },
            &mut Scratch::with_accumulators(&mut panel, &mut acc_buf),
            levels,
        );
    }
    c
}

/// `CD-21`: at every corpus shape --- even, odd, prime, degenerate --- and at
/// every offer, the recursion's bytes are the wrapping oracle's. Most corpus
/// shapes admit no level; they are here so the decline path is the one under
/// test, which it is: a decline is the cubic walk, and the oracle knows nothing
/// about either.
#[test]
fn recursion_matches_the_wrapping_oracle_cd_21() {
    for case in Corpus::standard(SEED).cases {
        let Case { m, k, n, .. } = case;
        let a = Case::widen(&case.fill_i8(m * k, 21));
        let b = Case::widen(&case.fill_i8(k * n, 22));
        let reference = reference_wrapping_i32(m, k, n, &a, &b);
        let (panels, accs) = strassen_scratch(Shape { m, k, n }, 3);
        for offer in [0, panels / 2, panels] {
            let ours = recursive_i32(m, k, n, &a, &b, 3, offer, accs);
            bytes_equal(&ours, &reference)
                .unwrap_or_else(|d| panic!("CD-21 mismatch at {m}x{k}x{n} offer {offer}: {d}"));
        }
    }
}

/// `CD-21`, the admitted half: even shapes deep enough for real levels, at
/// every requested level count and a starved offer that must decline rather
/// than corrupt. `ndarray` corroborates where it is defined.
#[cfg(feature = "ref-ndarray")]
#[test]
fn admitted_levels_match_ndarray_cd_21() {
    use uor_matmul_validate::oracle::{NdArray, Oracle};

    let mut compared = 0usize;
    for (i, &(m, k, n)) in [
        (256usize, 256usize, 256usize),
        (128, 256, 512),
        (64, 512, 128),
    ]
    .iter()
    .enumerate()
    {
        let case = Case {
            m,
            k,
            n,
            seed: SEED ^ (0x5EED << i),
        };
        let a = Case::widen(&case.fill_i8(m * k, 31));
        let b = Case::widen(&case.fill_i8(k * n, 32));
        let reference = reference_wrapping_i32(m, k, n, &a, &b);
        let shape = Shape { m, k, n };
        for levels in [0, 1, 2, 3, 4] {
            let (panels, accs) = strassen_scratch(shape, levels);
            // The full offer, and a starved one that cannot hold the top
            // level's temporaries: the level must be declined, not corrupted.
            for (offer, accs_offered) in [(panels, accs), (panels / 8, accs / 8), (0, 0)] {
                let ours = recursive_i32(m, k, n, &a, &b, levels, offer, accs_offered);
                bytes_equal(&ours, &reference).unwrap_or_else(|d| {
                    panic!("CD-21 mismatch at {m}x{k}x{n} levels {levels} offer {offer}: {d}")
                });
            }
        }
        if oracle_stays_in_range(k, 127, 127) {
            let theirs = NdArray::product_i32(m, k, n, &a, &b);
            let ours = recursive_i32(
                m,
                k,
                n,
                &a,
                &b,
                3,
                {
                    let (p, _) = strassen_scratch(shape, 3);
                    p
                },
                {
                    let (_, q) = strassen_scratch(shape, 3);
                    q
                },
            );
            if let uor_matmul_validate::Agreement::Mismatch(detail) =
                uor_matmul_validate::oracle::compare(&ours, &theirs)
            {
                panic!("CD-21 ndarray mismatch at {m}x{k}x{n}: {detail}");
            }
            compared += 1;
        }
    }
    assert!(compared > 0, "the oracle must actually have been called");
}

/// `CD-21`, the plan made observable: a starved offer admits strictly fewer
/// levels than a full one, the full `i32` alphabet admits none (a sum of two
/// full-range values is not an `i32`), and the bytes are the oracle's in
/// every case. Without the first two assertions the byte comparisons above
/// could pass with the recursion never running at all.
#[test]
fn the_plan_declines_what_the_offer_and_bound_cannot_hold_cd_21() {
    let shape = Shape {
        m: 256,
        k: 256,
        n: 256,
    };
    let (full_panels, full_accs) = strassen_scratch(shape, 2);
    let admitted_full = strassen_levels::<i32>(shape, 128, full_panels, full_accs, 2);
    assert_eq!(
        admitted_full, 2,
        "a full offer admits both requested levels"
    );
    let admitted_starved = strassen_levels::<i32>(shape, 128, full_panels / 8, full_accs / 8, 2);
    assert!(
        admitted_starved < admitted_full,
        "a starved offer must admit fewer levels, got {admitted_starved}"
    );
    assert_eq!(
        strassen_levels::<i32>(shape, Full::<i32>::VALUE, full_panels, full_accs, 2),
        0,
        "the full i32 alphabet leaves no headroom for even one level's sums"
    );
    // An odd extent admits nothing: the recursion declines rather than pad.
    assert_eq!(
        strassen_levels::<i32>(
            Shape {
                m: 255,
                k: 256,
                n: 256
            },
            128,
            full_panels,
            full_accs,
            2
        ),
        0,
        "an odd extent declines every level"
    );

    // And the bytes agree across the whole offer ladder, including the offers
    // that decline.
    let case = Case {
        m: 256,
        k: 256,
        n: 256,
        seed: SEED ^ 0xBEEF,
    };
    let a = Case::widen(&case.fill_i8(256 * 256, 41));
    let b = Case::widen(&case.fill_i8(256 * 256, 42));
    let reference = reference_wrapping_i32(256, 256, 256, &a, &b);
    let mut offer = full_panels;
    loop {
        let ours = recursive_i32(256, 256, 256, &a, &b, 2, offer, full_accs);
        bytes_equal(&ours, &reference)
            .unwrap_or_else(|d| panic!("CD-21 mismatch at offer {offer}: {d}"));
        if offer == 0 {
            break;
        }
        offer /= 8;
    }
}
