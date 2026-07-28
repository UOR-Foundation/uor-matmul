//! `CK-*`: tier equivalence, transcode, and the identity the whole crate exists
//! to realize.
//!
//! Every test here is `build`, not `some-true`: it is evidence that these types
//! realize `CL-MM01`, not a proof of it. The identity itself is cited from
//! upstream and says nothing about any binary in this repository (§2, R4).

use uor_matmul_codec::{
    Book, Codec, CodedMatrix, Enumerable, Grid, Identity, Offset, Packed, Runs, Sign, Transcode,
};
use uor_matmul_core::{
    as_alphabet_full, dot_ref, Alphabet, Bnd, Bound, Full, IntegerElement, NotAProduct,
};

type A8 = Alphabet<i8, Full<i8>>;

/// The 16-entry i4 grid: the canonical `Grid<16>`, and one table size among
/// many rather than a privileged one.
fn i4_table() -> [A8; 16] {
    core::array::from_fn(|i| Alphabet::of((i as i8) - 8))
}

/// Decode a whole coded matrix into a dense vector, for comparing tiers.
fn decode_all<E, Bd, C>(m: &CodedMatrix<'_, E, Bd, C>) -> Vec<Alphabet<E, Bd>>
where
    E: IntegerElement,
    Bd: Bound,
    C: Codec<E, Bd>,
{
    let mut out = Vec::new();
    for r in 0..m.rows() {
        for c in 0..m.cols() {
            out.push(m.at(r, c));
        }
    }
    out
}

/// CK-01: the identity codec decodes to its input, and decoding is the copy the
/// alphabet wrap already validated.
#[test]
fn identity_decodes_to_its_input_ck_01() {
    let data = [1i8, -2, 3, i8::MIN, i8::MAX];
    let codes = as_alphabet_full(&data);
    let m = CodedMatrix::new(Identity, 1, 5, codes).unwrap();
    assert_eq!(decode_all(&m), codes.to_vec());
    // `i8::MIN` is a code like any other. Nothing rejects it (C6).
    assert_eq!(m.at(0, 3).get(), i8::MIN);
}

/// CK-02: the grid tier equals the identity tier on equal decodes.
///
/// This is `CL-MM01` at its smallest instantiation: two different codecs, two
/// different code streams, one decoded stream, one product.
#[test]
fn grid_equals_identity_on_equal_decodes_ck_02() {
    let table = i4_table();
    let grid = Grid::new(&table);

    // Codes 0..15 under the grid decode to -8..7.
    let grid_codes: Vec<u16> = (0..16u16).collect();
    let coded = CodedMatrix::new(grid, 2, 8, &grid_codes).unwrap();

    // The same decoded stream, stored as itself.
    let dense: Vec<i8> = (0..16).map(|i| (i as i8) - 8).collect();
    let dense_codes = as_alphabet_full(&dense);
    let plain = CodedMatrix::new(Identity, 2, 8, dense_codes).unwrap();

    assert_eq!(decode_all(&coded), decode_all(&plain));

    // And therefore the same product against any activation vector.
    let acts: Vec<i8> = (0..8).map(|i| (i * 13 % 127) as i8).collect();
    let a = as_alphabet_full(&acts);
    for r in 0..2 {
        let mut w1 = vec![Alphabet::ZERO; 8];
        let mut w2 = vec![Alphabet::ZERO; 8];
        assert_eq!(coded.decode_row_into(r, &mut w1), 8);
        assert_eq!(plain.decode_row_into(r, &mut w2), 8);
        assert_eq!(dot_ref(a, &w1), dot_ref(a, &w2));
    }
}

/// CK-03: `Packed` unpacks the **low sub-code first**. Normative, pinned by
/// case `nibble-order`.
///
/// A reader that took the high nibble first would decode a different matrix
/// from the same bytes, so this is fixed here rather than left to convention.
#[test]
fn packed_unpacks_low_sub_code_first_ck_03() {
    let table = i4_table();
    let packed: Packed<Grid<'_, i8, Full<i8>, 16>, 2> = Packed::new(Grid::new(&table)).unwrap();

    // 0x3A: low nibble 0xA = 10 -> 2, high nibble 0x3 = 3 -> -5.
    let bytes = [0x3Au8];
    let m = CodedMatrix::new(packed, 1, 2, &bytes).unwrap();
    assert_eq!(m.at(0, 0).get(), 2, "the low sub-code decodes first");
    assert_eq!(m.at(0, 1).get(), -5, "the high sub-code decodes second");

    // Four two-bit sub-codes per byte: the same rule, a different P.
    let quad_table: [A8; 4] = [
        Alphabet::of(0),
        Alphabet::of(1),
        Alphabet::of(2),
        Alphabet::of(3),
    ];
    let quad: Packed<Grid<'_, i8, Full<i8>, 4>, 4> = Packed::new(Grid::new(&quad_table)).unwrap();
    // 0b11_10_01_00 = 0xE4: low-to-high 0, 1, 2, 3.
    let m = CodedMatrix::new(quad, 1, 4, &[0xE4u8]).unwrap();
    assert_eq!(
        (0..4).map(|c| m.at(0, c).get()).collect::<Vec<_>>(),
        vec![0, 1, 2, 3]
    );
}

/// CK-04: `Transcode` decodes as the composite, and composition is closed.
#[test]
fn transcode_is_the_composite_ck_04() {
    let table = i4_table();
    let grid = Grid::new(&table);

    // A relabelling: outer code `c` names inner code `15 - c`.
    let flip = Transcode::<_, _, u16>::new(|c: u16| 15u16 - (c % 16), grid);

    for c in 0..16u16 {
        let direct = Codec::<i8, Full<i8>>::decode_element(&grid, 15 - c, 0);
        let composed = Codec::<i8, Full<i8>>::decode_element(&flip, c, 0);
        assert_eq!(direct, composed);
    }

    // Closed under composition: transcode a transcode.
    let twice = Transcode::<_, _, u16>::new(|c: u16| 15u16 - (c % 16), flip);
    for c in 0..16u16 {
        assert_eq!(
            Codec::<i8, Full<i8>>::decode_element(&twice, c, 0),
            Codec::<i8, Full<i8>>::decode_element(&grid, c, 0),
        );
    }
}

/// CK-05: two `CodedMatrix` values with different tiers and equal decodes
/// produce byte-identical output.
///
/// This is the load-bearing test of the whole crate. If it ever fails, the
/// codec has become an argument of the arithmetic.
#[test]
fn different_tiers_equal_decodes_identical_product_ck_05() {
    // Tier one: an 8-element codebook, one code per row of 8.
    let book_table: [[A8; 8]; 4] =
        core::array::from_fn(|n| core::array::from_fn(|i| Alphabet::of(((n * 8 + i) as i8) - 16)));
    let book = Book::<i8, Full<i8>, 4, 8>::new(&book_table);
    let book_m = CodedMatrix::new(book, 4, 8, &[0u16, 1, 2, 3]).unwrap();

    // Tier two: the same decoded stream, stored densely.
    let dense: Vec<i8> = (0..32).map(|i| (i as i8) - 16).collect();
    let plain_m = CodedMatrix::new(Identity, 4, 8, as_alphabet_full(&dense)).unwrap();

    assert_eq!(decode_all(&book_m), decode_all(&plain_m));

    // Tier three: a grid over 32 single-element codes.
    let grid_table: [A8; 32] = core::array::from_fn(|i| Alphabet::of((i as i8) - 16));
    let grid_codes: Vec<u16> = (0..32u16).collect();
    let grid_m = CodedMatrix::new(Grid::new(&grid_table), 4, 8, &grid_codes).unwrap();
    assert_eq!(decode_all(&grid_m), decode_all(&plain_m));

    // Three tiers, three block sizes, three code types --- one product.
    let acts: Vec<i8> = (0..8).map(|i| (i as i8) * 7 - 20).collect();
    let a = as_alphabet_full(&acts);
    for r in 0..4 {
        let mut w = vec![Alphabet::ZERO; 8];
        let mut x = vec![Alphabet::ZERO; 8];
        let mut y = vec![Alphabet::ZERO; 8];
        book_m.decode_row_into(r, &mut w);
        plain_m.decode_row_into(r, &mut x);
        grid_m.decode_row_into(r, &mut y);
        let expected = dot_ref(a, &x);
        assert_eq!(dot_ref(a, &w), expected);
        assert_eq!(dot_ref(a, &y), expected);
    }
}

/// CK-01: `Offset` expresses asymmetric quantization exactly, as a composition
/// and not a special case --- so it is two codecs with equal decodes, and it
/// falls under the same claim every other tier does.
#[test]
fn offset_is_asymmetric_quantization_ck_01() {
    // Unsigned 4-bit codes 0..15, zero point 8: the classic asymmetric tier.
    let unsigned: [Alphabet<i8, Bnd<15>>; 16] =
        core::array::from_fn(|i| Alphabet::new(i as i8).unwrap());
    let grid = Grid::new(&unsigned);
    let offset =
        Offset::<i8, Bnd<15>, Bnd<23>, _>::new(grid, 8).expect("15 + |8| = 23 fits Bnd<23> and i8");

    for c in 0..16u16 {
        let decoded = Codec::<i8, Bnd<23>>::decode_element(&offset, c, 0);
        assert_eq!(decoded.get(), (c as i8) - 8);
    }

    // A zero point whose image leaves the declared bound has no such codec, and
    // that is reported at construction, before any arithmetic (C6).
    assert!(Offset::<i8, Bnd<15>, Bnd<20>, _>::new(grid, 8).is_none());
}

/// CK-06: a run codec's returned counts sum to the declared row width on every
/// row, and the decoded stream is the same dense stream the inner codec gives.
///
/// This is the invariant that lets sparse storage be a *tier* rather than a
/// second algorithm: the arithmetic downstream never learns that the weights
/// were stored as runs.
#[test]
fn a_run_codecs_counts_sum_to_the_row_width_ck_06() {
    let table = i4_table();
    let grid = Grid::new(&table);

    // Two rows of twelve: four zeros (code 8 -> 0), four fours (code 12 -> 4),
    // four minus-fives (code 3 -> -5). The zeros are explicit runs, which is
    // what keeps the arithmetic dense.
    let runs = [
        (4u32, 8u16),
        (4u32, 12u16),
        (4u32, 3u16),
        (4u32, 8u16),
        (4u32, 12u16),
        (4u32, 3u16),
    ];
    let sparse: Runs<'_, i8, Full<i8>, _, 8> = Runs::new(grid, &runs).unwrap();

    // The counts sum to the row width. Twice, because there are two rows.
    assert_eq!(sparse.decoded_len(), 24);
    assert_eq!(Codec::<i8, Full<i8>>::decode_len(&sparse, 0), 4);

    let indices: Vec<u32> = (0..6u32).collect();
    let m = CodedMatrix::new(sparse, 2, 12, &indices).expect("the counts sum to 12 per row");

    let expected: Vec<i8> = (0..12)
        .map(|p| match p {
            0..=3 => 0,
            4..=7 => 4,
            _ => -5,
        })
        .collect();
    for r in 0..2 {
        let mut row = vec![Alphabet::ZERO; 12];
        assert_eq!(
            m.decode_row_into(r, &mut row),
            12,
            "the row decodes to exactly `cols`"
        );
        assert_eq!(row.iter().map(|a| a.get()).collect::<Vec<_>>(), expected);
    }

    // `column_walk` agrees with `at`, element for element, and exists because
    // `at` does not scale on this tier: it walks row `r` *and* `row_code_range`
    // walks rows `0..r`, so a driver reading a column one element at a time is
    // O(rows^2). Measured on a 512-row run matrix: 12.4 ms indexed against
    // 0.058 ms walked, a factor of 215 that grows with the row count. The
    // zero-offer coded traversal was that driver.
    for c in 0..12 {
        let walked: Vec<i8> = m.column_walk(c).map(|a| a.get()).collect();
        let indexed: Vec<i8> = (0..2).map(|r| m.at(r, c).get()).collect();
        assert_eq!(walked, indexed, "column {c}");
        assert_eq!(walked.len(), 2, "one element per row");
    }

    // The residency claim is a number, not an adjective: six stored runs
    // against twenty-four decoded elements.
    assert_eq!(m.codec().run_count(), 6);

    // Counts that do not sum to the declared width describe no matrix, and
    // that is reported at construction (C6).
    let short = [(3u32, 8u16), (4u32, 12u16)];
    let bad: Runs<'_, i8, Full<i8>, _, 8> = Runs::new(grid, &short).unwrap();
    assert!(CodedMatrix::new(bad, 1, 12, &[0u32, 1]).is_none());

    // A run longer than the declared maximum, or an empty one, is not a run.
    assert!(Runs::<'_, i8, Full<i8>, _, 8>::new(grid, &[(9u32, 8u16)]).is_none());
    assert!(Runs::<'_, i8, Full<i8>, _, 8>::new(grid, &[(0u32, 8u16)]).is_none());
}

/// A shape a codec cannot describe is reported at construction, and nothing
/// else about the data can make `CodedMatrix::new` fail (§6.3, C6).
#[test]
fn only_a_shape_disagreement_is_reportable_cs_03() {
    let table = i4_table();
    let grid = Grid::new(&table);
    let codes = [0u16; 8];

    // Too few codes for the declared shape: no such matrix.
    assert!(CodedMatrix::new(grid, 2, 8, &codes).is_none());
    // A column count that is not a multiple of the block: no such matrix.
    let book_table: [[A8; 8]; 2] = [[Alphabet::of(0); 8]; 2];
    let book = Book::<i8, Full<i8>, 2, 8>::new(&book_table);
    assert!(CodedMatrix::new(book, 1, 7, &[0u16]).is_none());
    // The conforming shape constructs.
    assert!(CodedMatrix::new(grid, 1, 8, &codes).is_some());

    // The error surface of the whole library, for reference: two variants,
    // both meaning the requested object does not exist.
    let _: fn() -> NotAProduct = || NotAProduct::Nonconformant {
        a: (1, 1),
        b: (2, 2),
    };
}

/// The streaming decode agrees with the row decode, which is what lets a target
/// whose RAM cannot hold one row produce the same bytes (S13, CD-04).
#[test]
fn streaming_decode_equals_row_decode_cd_04() {
    let table = i4_table();
    let codes: Vec<u16> = (0..64u16).map(|c| c % 16).collect();
    let m = CodedMatrix::new(Grid::new(&table), 4, 16, &codes).unwrap();

    for r in 0..4 {
        let mut whole = vec![Alphabet::ZERO; 16];
        m.decode_row_into(r, &mut whole);

        // Three elements at a time, which is not a divisor of the row length.
        let mut streamed = Vec::new();
        let mut buf = [Alphabet::ZERO; 3];
        let mut c = 0;
        while c < 16 {
            let end = (c + 3).min(16);
            m.decode_range_into(r, c..end, &mut buf[..end - c]);
            streamed.extend_from_slice(&buf[..end - c]);
            c = end;
        }
        assert_eq!(whole, streamed);
    }
}

/// `CK-03`: the packed panels themselves are byte-equal across codecs that
/// decode equally --- not merely the products.
///
/// This is stronger than `CK-05` and it is the one that would catch a codec
/// whose decode is right but whose *ordering* is not. Two tiers that agree
/// element for element must fill a packed panel with the same bytes, because
/// the packing reads the decoded stream and nothing else.
#[test]
fn packed_panels_are_byte_equal_across_tiers_ck_03() {
    use uor_matmul_codec::{e8_codec, e8_table, E8_I8};

    // The E8 codebook, and a dense tier holding the same decoded stream.
    let table = e8_table::<Full<i8>>().expect("i8 admits every codeword");
    let book = e8_codec(&table);
    let codes: Vec<u16> = (0..64u16).map(|c| c * 3 % 256).collect();
    let coded = CodedMatrix::new(book, 8, 64, &codes).unwrap();

    let dense: Vec<i8> = codes
        .iter()
        .flat_map(|&c| E8_I8[c as usize % 256].iter().copied())
        .collect();
    let plain = CodedMatrix::new(Identity, 8, 64, as_alphabet_full(&dense)).unwrap();

    // Pack both, k-major, exactly as the driver's kernel path does.
    fn pack<C: Codec<i8, Full<i8>>>(
        m: &CodedMatrix<'_, i8, Full<i8>, C>,
        mr: usize,
        kc: usize,
    ) -> Vec<i8> {
        let mut out = vec![0i8; mr * kc];
        for p in 0..kc {
            for i in 0..mr {
                out[p * mr + i] = m.at(i, p).get();
            }
        }
        out
    }

    for (mr, kc) in [(1usize, 1usize), (4, 8), (6, 16), (8, 64)] {
        assert_eq!(
            pack(&coded, mr, kc),
            pack(&plain, mr, kc),
            "the E8 tier and the dense tier must pack the same bytes at {mr}x{kc}"
        );
    }

    // And the panels really are non-trivial, or the comparison is vacuous.
    let panel = pack(&coded, 8, 64);
    assert!(
        panel.iter().any(|&b| b != 0),
        "the packed panel must not be all zeros"
    );
}

/// Every `u16` code, or a stride through them plus the boundaries under Miri.
///
/// Totality over all 65536 is the claim, and the native run makes it. Under Miri
/// the three laws run 65536 times per call site at something like a hundredfold
/// the cost per iteration, and there are three sites --- one of them `Book<256,
/// 8>`, which decodes eight elements per code. That was the longest thing in the
/// codec suite. A stride keeps what soundness needs --- the two ends, the codes
/// either side of a code space, and a sample across the range --- and the claim
/// stays the native run's.
fn every_code(f: &mut dyn FnMut(u16)) {
    if !cfg!(miri) {
        for c in 0..=u16::MAX {
            f(c);
        }
        return;
    }
    // The boundaries first: zero, one, either side of each code space this is
    // called with --- sixteen and two hundred fifty-six --- and the top of the
    // type, which is past every one of them and so is the in-range check's own
    // worst case.
    for c in [0u16, 1, 15, 16, 17, 255, 256, 257, u16::MAX - 1, u16::MAX] {
        f(c);
    }
    let mut c = 0u32;
    while c <= u32::from(u16::MAX) {
        f(c as u16);
        c += 257; // coprime with 16 and with 256, so it lands on every residue
    }
}

/// `CK-09`: `Enumerable`'s two laws, for every codec that implements it.
///
/// Law 1 is the round trip, checked over the *whole* code space rather than a
/// sample: an enumeration with one hole is an enumeration that indexes a dead
/// table entry, and a sample would miss it.
///
/// Law 2 is totality of `index_of` on the code *type*, not on the enumeration ---
/// checked over every `u8` and every `u16`, including values no encoder would
/// ever produce. That is the law the tabulated traversal cashes in when it reads
/// the table without a bounds check (`CT-07`).
#[test]
fn the_enumeration_round_trips_and_is_total_ck_09() {
    use uor_matmul_codec::Enumerable;

    /// Law 1 and law 2 for one codec, plus the decode agreement that makes an
    /// index a stand-in for a code at all.
    fn laws<C, F>(name: &str, codec: &C, block: usize, every_code: F)
    where
        C: Enumerable<i8, Full<i8>>,
        F: Fn(&mut dyn FnMut(C::Code)),
    {
        assert!(
            C::CODE_SPACE > 0,
            "{name}: a codec that enumerates nothing cannot be tabulated"
        );

        // Law 1, over the whole space.
        for i in 0..C::CODE_SPACE {
            assert_eq!(
                C::index_of(C::code_at(i)),
                i,
                "{name}: index_of(code_at({i})) must be {i}"
            );
        }

        // Law 2, over the whole code type, and the decode agreement with it:
        // two codes at the same index must decode alike, or the table would
        // answer one of them wrongly.
        every_code(&mut |code| {
            let index = C::index_of(code);
            assert!(
                index < C::CODE_SPACE,
                "{name}: index_of is not total; a code landed at {index} of {}",
                C::CODE_SPACE
            );
            let canonical = C::code_at(index);
            for t in 0..block {
                assert_eq!(
                    codec.decode_element(code, t).get(),
                    codec.decode_element(canonical, t).get(),
                    "{name}: two codes share index {index} but decode differently at {t}"
                );
            }
        });

        // Law 3: a codec that says its stored stream *is* the index stream has
        // to mean it. The tabulated traversal reads that stream straight into
        // the table's address, so a codec that answered `Some` wrongly would
        // read a live entry belonging to another code and report a wrong sum
        // with nothing to catch it.
        let sample: Vec<C::Code> = (0..C::CODE_SPACE.min(64)).map(C::code_at).collect();
        if let Some(stream) = C::as_index_stream(&sample) {
            assert!(
                C::CODE_SPACE.is_power_of_two(),
                "{name}: a code addresses the enumeration only if the space is 2^j"
            );
            assert_eq!(
                stream.len(),
                sample.len(),
                "{name}: the index stream is the code stream, not a copy of part of it"
            );
            let mask = C::CODE_SPACE - 1;
            every_code(&mut |code| {
                let word = C::as_index_stream(core::slice::from_ref(&code))
                    .expect("the answer does not depend on the slice")[0];
                assert_eq!(
                    word as usize & mask,
                    C::index_of(code),
                    "{name}: the stored code does not address the enumeration"
                );
            });
        }
    }

    let table = i4_table();
    let grid = Grid::<i8, Full<i8>, 16>::new(&table);
    laws("Grid<16>", &grid, 1, |f: &mut dyn FnMut(u16)| every_code(f));

    let packed = Packed::<_, 2>::new(grid).expect("2 divides 8");
    laws(
        "Packed<Grid<16>, 2>",
        &packed,
        2,
        |f: &mut dyn FnMut(u8)| {
            for c in 0..=u8::MAX {
                f(c)
            }
        },
    );

    let quad_table: [A8; 4] = core::array::from_fn(|i| Alphabet::of((i as i8) - 2));
    let quad = Grid::<i8, Full<i8>, 4>::new(&quad_table);
    let packed4 = Packed::<_, 4>::new(quad).expect("4 divides 8");
    laws(
        "Packed<Grid<4>, 4>",
        &packed4,
        4,
        |f: &mut dyn FnMut(u8)| {
            for c in 0..=u8::MAX {
                f(c)
            }
        },
    );

    // A grid narrower than its own field: two nibbles over a ten-entry table, so
    // the sub-space is ten and the total space is a hundred rather than 256.
    let ten_table: [A8; 10] = core::array::from_fn(|i| Alphabet::of((i as i8) - 5));
    let ten = Grid::<i8, Full<i8>, 10>::new(&ten_table);
    let packed_ten = Packed::<_, 2>::new(ten).expect("2 divides 8");
    assert_eq!(
        <Packed<Grid<'_, i8, Full<i8>, 10>, 2> as Enumerable<i8, Full<i8>>>::CODE_SPACE,
        100,
        "a ten-entry grid packed two to a byte enumerates ten squared"
    );
    laws(
        "Packed<Grid<10>, 2>",
        &packed_ten,
        2,
        |f: &mut dyn FnMut(u8)| {
            for c in 0..=u8::MAX {
                f(c)
            }
        },
    );

    let e8 = uor_matmul_codec::e8_table::<Full<i8>>().expect("i8 holds E8");
    let book = uor_matmul_codec::e8_codec(&e8);
    laws("Book<256, 8>", &book, 8, |f: &mut dyn FnMut(u16)| {
        every_code(f)
    });

    // The sign tier. Its code *is* the index at every block width, so the
    // third law's `Some` arm is the whole point of the tier rather than a
    // power-of-two coincidence.
    let sign = Sign::<i8, Full<i8>, 8>::new().expect("the full alphabet admits +-1");
    laws("Sign<8>", &sign, 8, |f: &mut dyn FnMut(u16)| every_code(f));

    // A zero point relabels the image and not the code space, so the offset
    // tier's enumeration is the inner one and needs no new law.
    // The inner bound has to leave the zero point room, which is the O(1) check
    // `Offset::new` already performs; the enumeration is unaffected either way.
    let narrow: [Alphabet<i8, Bnd<8>>; 16] =
        core::array::from_fn(|i| Alphabet::new((i as i8) - 8).expect("|-8..7| <= 8"));
    let offset =
        Offset::<i8, Bnd<8>, Full<i8>, _>::new(Grid::<i8, Bnd<8>, 16>::new(&narrow), 3).unwrap();
    laws("Offset<Grid<16>>", &offset, 1, |f: &mut dyn FnMut(u16)| {
        every_code(f)
    });
}

/// `CK-11`: the `Sign` tier decodes the same alphabet stream as the
/// `Packed<Grid<2>,8>` spelling, and its index stream is the code stream.
///
/// The decode check is exhaustive over the shared code space --- all 256 codes
/// at block 8, through `decode_into` --- so the bit convention is pinned
/// against the composition's own decode rather than restated alongside it. The
/// index-stream check is the tier's reason to exist: a packed byte's index is
/// a mixed-radix decomposition, so the composition *builds* the stream the
/// tabulated traversal gathers from, where a `Sign` code already is one.
#[test]
fn the_sign_tier_decodes_the_compositions_stream_ck_11() {
    let table: [A8; 2] = [Alphabet::of(-1), Alphabet::of(1)];
    let packed = Packed::<_, 8>::new(Grid::<i8, Full<i8>, 2>::new(&table)).expect("8 divides 8");
    let sign = Sign::<i8, Full<i8>, 8>::new().expect("the full alphabet admits +-1");

    let mut want = [A8::ZERO; 8];
    let mut got = [A8::ZERO; 8];
    for code in 0..256u16 {
        assert_eq!(packed.decode_into(code as u8, &mut want), 8);
        assert_eq!(sign.decode_into(code, &mut got), 8);
        assert_eq!(
            want, got,
            "code {code:#04x} decodes differently under the two spellings"
        );
    }

    // The index stream is the code stream: the same memory, not a copy of it.
    let codes: Vec<u16> = (0..256).collect();
    let stream = Sign::<i8, Full<i8>, 8>::as_index_stream(&codes)
        .expect("a Sign code addresses the enumeration");
    assert!(
        core::ptr::eq(stream.as_ptr(), codes.as_ptr()),
        "the stream must be borrowed, not built"
    );

    // The composition cannot answer it, which is the gap this tier exists to
    // close. If `Packed` ever could, this assertion --- not a benchmark --- is
    // what should be revisited first.
    assert!(
        <Packed<Grid<'_, i8, Full<i8>, 2>, 8> as uor_matmul_codec::Enumerable<
            i8,
            Full<i8>,
        >>::as_index_stream(&[0u8; 4])
        .is_none()
    );

    // An alphabet too narrow for +-1 has no sign codec, and that is decided at
    // construction, before any arithmetic (C6).
    assert!(Sign::<i8, Bnd<0>, 8>::new().is_none());
}
