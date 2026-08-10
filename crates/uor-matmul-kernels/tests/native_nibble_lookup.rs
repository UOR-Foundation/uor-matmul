//! Exhaustive signed-octet parity for the native group-one lookup kernels.
//!
//! The ordinary parity corpus establishes arbitrary-depth accumulation and
//! overwrite semantics. This focused corpus asks the complementary question:
//! does each native lookup primitive reconstruct every member of the
//! complete signed `i8 x i8` product alphabet?

#![cfg(any(
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128"),
    target_arch = "x86_64"
))]

use uor_matmul_core::Backend;
use uor_matmul_kernels::{available_i8, available_i8_narrow, available_reduce_i8};

fn native_backend(backend: Backend) -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        matches!(backend, Backend::Neon | Backend::NeonDotprod)
    }
    #[cfg(target_arch = "wasm32")]
    {
        backend == Backend::WasmSimd128
    }
    #[cfg(target_arch = "x86_64")]
    {
        matches!(backend, Backend::Avx2 | Backend::Avx512Vnni)
    }
}

/// An independent shift/add oracle; the native sequence under test contains
/// no multiply operation and does not read this spelling.
fn exact_product(left: i8, right: i8) -> i32 {
    let mut magnitude = i32::from(right).unsigned_abs();
    let mut addend = i32::from(left).unsigned_abs();
    let mut product = 0u32;
    while magnitude != 0 {
        if magnitude & 1 != 0 {
            product += addend;
        }
        addend += addend;
        magnitude >>= 1;
    }
    let product = product as i32;
    if (left < 0) != (right < 0) {
        0 - product
    } else {
        product
    }
}

#[test]
fn native_group_one_lookup_tiles_exhaust_every_signed_octet_pair_ck_18() {
    #[cfg(target_arch = "x86_64")]
    if !std::arch::is_x86_feature_detected!("avx2") {
        return;
    }

    // The narrow x86 family exercises the same projector with its distinct
    // eight-column load/store arm, so it belongs in the exhaustive alphabet
    // census as well as the ordinary selector family.
    let specs: Vec<_> = available_i8()
        .chain(available_i8_narrow())
        .filter(|spec| native_backend(spec.backend) && spec.k_group == 1)
        .collect();
    assert!(
        !specs.is_empty(),
        "the native target must offer its lookup/add tile"
    );

    for spec in specs {
        for left_bits in 0u16..=u8::MAX as u16 {
            let left = left_bits as u8 as i8;
            for right_start in (0u16..=u8::MAX as u16).step_by(spec.nr) {
                let pa = vec![left; spec.mr];
                let pb: Vec<_> = (0..spec.nr)
                    .map(|lane| right_start.wrapping_add(lane as u16) as u8 as i8)
                    .collect();
                let mut acc = vec![0x5a5a_5a5a; spec.mr * spec.nr];
                spec.mac_tile(1, &pa, &pb, &mut acc);

                for i in 0..spec.mr {
                    for j in 0..spec.nr {
                        assert_eq!(
                            acc[i * spec.nr + j],
                            exact_product(left, pb[j]),
                            "{} lookup tile disagrees at ({left}, {})",
                            spec.backend.as_str(),
                            pb[j]
                        );
                    }
                }
            }
        }
    }
}

/// A narrow native vector contains two consecutive reduction depths. Place
/// every signed-octet pair in each half in turn, then in the odd terminal
/// depth, so a half permutation, omission, or duplicate cannot hide behind an
/// equal neighbouring product.
#[cfg(target_arch = "x86_64")]
#[test]
fn native_narrow_lookup_tiles_exhaust_grouped_and_terminal_depths_ck_18() {
    if !std::arch::is_x86_feature_detected!("avx2") {
        return;
    }

    let specs: Vec<_> = available_i8_narrow()
        .filter(|spec| spec.backend == Backend::Avx2 && spec.k_group == 1 && spec.nr == 8)
        .collect();
    assert!(
        !specs.is_empty(),
        "an AVX2 target must offer its narrow lookup/add tiles"
    );

    const FIXED_LEFT: [i8; 2] = [3, -5];
    const FIXED_RIGHT: [i8; 2] = [-7, 11];
    for spec in specs {
        for left_bits in 0u16..=u8::MAX as u16 {
            let left = left_bits as u8 as i8;
            for right_start in (0u16..=u8::MAX as u16).step_by(spec.nr) {
                let rights: Vec<_> = (0..spec.nr)
                    .map(|lane| right_start.wrapping_add(lane as u16) as u8 as i8)
                    .collect();
                for &(kc, target_depth) in &[(2usize, 0usize), (2, 1), (3, 2)] {
                    let mut pa = Vec::with_capacity(spec.mr * kc);
                    let mut pb = Vec::with_capacity(spec.nr * kc);
                    for depth in 0..kc {
                        let depth_left = if depth == target_depth {
                            left
                        } else {
                            FIXED_LEFT[depth]
                        };
                        pa.extend(core::iter::repeat_n(depth_left, spec.mr));
                        if depth == target_depth {
                            pb.extend_from_slice(&rights);
                        } else {
                            pb.extend(core::iter::repeat_n(FIXED_RIGHT[depth], spec.nr));
                        }
                    }

                    let mut acc = vec![0x5a5a_5a5a; spec.mr * spec.nr];
                    spec.mac_tile(kc, &pa, &pb, &mut acc);
                    for i in 0..spec.mr {
                        for j in 0..spec.nr {
                            let expected = (0..kc).fold(0i32, |sum, depth| {
                                sum.wrapping_add(exact_product(
                                    pa[depth * spec.mr + i],
                                    pb[depth * spec.nr + j],
                                ))
                            });
                            assert_eq!(
                                acc[i * spec.nr + j],
                                expected,
                                "{} narrow lookup disagrees at target depth {target_depth}, pair ({left}, {})",
                                spec.backend.as_str(),
                                rights[j]
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn native_group_one_lookup_reductions_exhaust_every_signed_octet_pair_ck_18() {
    #[cfg(target_arch = "x86_64")]
    if !std::arch::is_x86_feature_detected!("avx2") {
        return;
    }

    let specs: Vec<_> = available_reduce_i8()
        .filter(|spec| native_backend(spec.backend) && spec.k_group == 1)
        .collect();
    assert!(
        !specs.is_empty(),
        "the native target must offer its lookup/add reductions"
    );

    for spec in specs {
        for right_bits in 0u16..=u8::MAX as u16 {
            let right = right_bits as u8 as i8;
            for left_start in (0u16..=u8::MAX as u16).step_by(spec.mr) {
                // At `kc == 1`, contiguous reduction packing is one octet per
                // row, so every live SIMD lane receives a distinct coordinate.
                let pa: Vec<_> = (0..spec.mr)
                    .map(|lane| left_start.wrapping_add(lane as u16) as u8 as i8)
                    .collect();
                let pb = [right];
                let mut acc = vec![0x5a5a_5a5a; spec.mr];
                spec.mac_tile(1, &pa, &pb, &mut acc);

                for i in 0..spec.mr {
                    assert_eq!(
                        acc[i],
                        exact_product(pa[i], right),
                        "{} lookup reduction disagrees at ({}, {right})",
                        spec.backend.as_str(),
                        pa[i]
                    );
                }
            }
        }
    }
}

/// The one-row x86 reduction enters its gather sequence only once eight depth
/// coordinates are live. Repeating each pair across that complete vector
/// isolates every address while making an omitted or duplicated radix
/// refinement observable in the reduction result.
#[test]
fn native_one_row_vector_lookup_reduction_exhausts_every_pair_address_ck_18() {
    #[cfg(target_arch = "x86_64")]
    if !std::arch::is_x86_feature_detected!("avx2") {
        return;
    }

    const VECTOR_DEPTH: usize = 8;
    let specs: Vec<_> = available_reduce_i8()
        .filter(|spec| native_backend(spec.backend) && spec.k_group == 1 && spec.mr == 1)
        .collect();
    assert!(
        !specs.is_empty(),
        "the native target must offer its one-row lookup reduction"
    );

    for spec in specs {
        for left_bits in 0u16..=u8::MAX as u16 {
            let left = left_bits as u8 as i8;
            for right_bits in 0u16..=u8::MAX as u16 {
                let right = right_bits as u8 as i8;
                let pa = [left; VECTOR_DEPTH];
                let pb = [right; VECTOR_DEPTH];
                let mut acc = [0x5a5a_5a5a];
                spec.mac_tile(VECTOR_DEPTH, &pa, &pb, &mut acc);

                let product = exact_product(left, right);
                let mut expected = 0i32;
                for _ in 0..VECTOR_DEPTH {
                    expected = expected.wrapping_add(product);
                }
                assert_eq!(
                    acc[0],
                    expected,
                    "{} vector lookup reduction disagrees at ({left}, {right})",
                    spec.backend.as_str()
                );
            }
        }
    }
}

/// Every depth through four complete paired vectors is visited, including all
/// scalar, terminal-vector, pair-entry, and pair-exit boundaries. Distinct,
/// nonzero half sums make aliasing either accumulator or omitting their merge
/// observably different from the portable product alphabet.
#[cfg(target_arch = "x86_64")]
#[test]
fn native_one_row_lookup_reduction_exhausts_paired_depth_boundaries_ck_18() {
    if !std::arch::is_x86_feature_detected!("avx2") {
        return;
    }

    const NATIVE_LANES: usize =
        core::mem::size_of::<std::arch::x86_64::__m256i>() / core::mem::size_of::<i32>();
    const PAIRED_DEPTH: usize = 2 * NATIVE_LANES;
    const MAX_DEPTH: usize = 4 * PAIRED_DEPTH;
    let specs: Vec<_> = available_reduce_i8()
        .filter(|spec| spec.backend == Backend::Avx2 && spec.k_group == 1 && spec.mr == 1)
        .collect();
    assert!(
        !specs.is_empty(),
        "an AVX2 target must offer its one-row lookup reduction"
    );

    for spec in specs {
        for kc in 0..=MAX_DEPTH {
            let pa: Vec<_> = (0..kc)
                .map(|p| {
                    let coordinate = (p % NATIVE_LANES + 1) as i8;
                    if (p / NATIVE_LANES).is_multiple_of(2) {
                        coordinate
                    } else {
                        -coordinate
                    }
                })
                .collect();
            let pb: Vec<_> = (0..kc)
                .map(|p| {
                    if (p / NATIVE_LANES).is_multiple_of(2) {
                        3
                    } else {
                        5
                    }
                })
                .collect();
            let expected = pa.iter().zip(&pb).fold(0i32, |sum, (&left, &right)| {
                sum.wrapping_add(exact_product(left, right))
            });
            let mut acc = [0x5a5a_5a5a];
            spec.mac_tile(kc, &pa, &pb, &mut acc);
            assert_eq!(
                acc[0],
                expected,
                "{} paired lookup reduction disagrees at depth {kc}",
                spec.backend.as_str()
            );
        }
    }
}
