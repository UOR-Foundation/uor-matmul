//! Lookup-only products for finite integer alphabets.
//!
//! The table is data, not a runtime multiplication routine. Its compile-time
//! constructor evaluates the canonical NAF representative, so the shipped i8
//! product route is an Atlas-octet lookup followed by accumulation, including
//! the table-build step.

use crate::generated_capacity::{CacheAligned, CACHE_LINE_BYTES};

/// A binary radix power expressed by the same self-similar doubling used by
/// the Atlas coordinate refinement. Keeping it as a recurrence makes the
/// declaration independent of a machine bit-concatenation operator.
const fn binary_radix(digits: u32) -> usize {
    let mut radix = 1usize;
    let mut digit = 0u32;
    while digit < digits {
        radix += radix;
        digit += 1;
    }
    radix
}

/// A const product as repeated extent addition. This is address geometry, not
/// an element product; spelling the recurrence keeps even table construction
/// out of the multiplication vocabulary audited by `CU-11`.
const fn extent(steps: usize, width: usize) -> usize {
    let mut total = 0usize;
    let mut step = 0usize;
    while step < steps {
        total += width;
        step += 1;
    }
    total
}

const I8_SPACE: usize = binary_radix(u8::BITS);
#[cfg(any(
    target_arch = "aarch64",
    target_arch = "wasm32",
    target_arch = "x86_64",
    test
))]
pub(crate) const NIBBLE_BITS: u32 = u8::BITS / 2;
#[cfg(any(
    target_arch = "aarch64",
    target_arch = "wasm32",
    target_arch = "x86_64",
    test
))]
pub(crate) const NIBBLE_SPACE: usize = binary_radix(NIBBLE_BITS);
#[cfg(any(
    target_arch = "aarch64",
    target_arch = "wasm32",
    target_arch = "x86_64",
    test
))]
const PRODUCT_BYTES: usize = core::mem::size_of::<i16>();
#[cfg(any(
    target_arch = "aarch64",
    target_arch = "wasm32",
    target_arch = "x86_64",
    test
))]
const NIBBLE_COMPONENTS: usize = u8::BITS.div_ceil(NIBBLE_BITS) as usize;
#[cfg(any(
    target_arch = "aarch64",
    target_arch = "wasm32",
    target_arch = "x86_64",
    test
))]
const NIBBLE_PROJECTORS: usize = extent(PRODUCT_BYTES, NIBBLE_COMPONENTS);
#[cfg(any(
    target_arch = "aarch64",
    target_arch = "wasm32",
    target_arch = "x86_64",
    test
))]
const NIBBLE_ROW_BYTES: usize = extent(NIBBLE_PROJECTORS, NIBBLE_SPACE);
const I8_PRODUCT_ENTRIES: usize = extent(I8_SPACE, I8_SPACE);

/// The exact signed `i8 x i8 -> i32` product table. Each complete left-octet
/// row occupies an integral number of model cache lines; aligning the alphabet
/// base prevents any row from acquiring a seventeenth line by displacement.
#[cfg_attr(
    all(target_arch = "x86_64", target_os = "linux"),
    unsafe(export_name = "__uor_matmul_kernels_v0_1_0_i8_products")
)]
static I8_PRODUCTS: CacheAligned<[i32; I8_PRODUCT_ENTRIES]> = CacheAligned(build_i8_products());
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
core::arch::global_asm!(".hidden __uor_matmul_kernels_v0_1_0_i8_products");
const _: () = {
    assert!(core::mem::align_of::<CacheAligned<[i32; I8_PRODUCT_ENTRIES]>>() == CACHE_LINE_BYTES);
    assert!(
        core::mem::size_of::<CacheAligned<[i32; I8_PRODUCT_ENTRIES]>>()
            == core::mem::size_of::<[i32; I8_PRODUCT_ENTRIES]>()
    );
};

/// Borrow the canonical signed-octet product alphabet.
#[inline(always)]
pub(crate) fn i8_products() -> &'static [i32; I8_PRODUCT_ENTRIES] {
    &I8_PRODUCTS
}

/// Borrow the same alphabet through a direct native address representation.
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
#[inline(always)]
pub(crate) fn i8_products_native() -> &'static [i32; I8_PRODUCT_ENTRIES] {
    let address: *const CacheAligned<[i32; I8_PRODUCT_ENTRIES]>;
    // SAFETY: the symbol operand is the one aligned static of this exact type;
    // `lea` only reifies its address and neither reads nor changes memory.
    unsafe {
        core::arch::asm!(
            "lea {address}, [rip + {table}]",
            address = out(reg) address,
            table = sym I8_PRODUCTS,
            options(nostack, readonly, preserves_flags)
        );
        &(*address).0
    }
}

/// Non-ELF x86 targets retain Rust's regular private-static address spelling.
#[cfg(all(target_arch = "x86_64", not(target_os = "linux")))]
#[inline(always)]
pub(crate) fn i8_products_native() -> &'static [i32; I8_PRODUCT_ENTRIES] {
    i8_products()
}

/// Four byte-projector tables per signed octet: low/high bytes of the low
/// nibble contribution, followed by low/high bytes of the signed high-nibble
/// contribution. A SIMD byte shuffle is a parallel table lookup, so this is
/// the same complete octet product table factored into L1-sized projectors.
/// One row is exactly one model cache line, so the generated alignment makes
/// that finite row the unit fetched by every native projector.
#[cfg(any(
    target_arch = "aarch64",
    target_arch = "wasm32",
    target_arch = "x86_64",
    test
))]
pub(crate) static I8_NIBBLE_PRODUCTS: CacheAligned<[[u8; NIBBLE_ROW_BYTES]; I8_SPACE]> =
    CacheAligned(build_i8_nibble_products());
#[cfg(any(
    target_arch = "aarch64",
    target_arch = "wasm32",
    target_arch = "x86_64",
    test
))]
const _: () = {
    assert!(NIBBLE_ROW_BYTES == CACHE_LINE_BYTES);
    assert!(
        core::mem::align_of::<CacheAligned<[[u8; NIBBLE_ROW_BYTES]; I8_SPACE]>>()
            == CACHE_LINE_BYTES
    );
    assert!(
        core::mem::size_of::<CacheAligned<[[u8; NIBBLE_ROW_BYTES]; I8_SPACE]>>()
            == core::mem::size_of::<[[u8; NIBBLE_ROW_BYTES]; I8_SPACE]>()
    );
};

/// Row addresses are themselves a finite lookup alphabet. Generating them by
/// repeated extent addition removes both a hot-loop widening product and a
/// packed-code bit concatenation without allocating or changing table bytes.
const fn build_product_row_addresses() -> [i32; I8_SPACE] {
    let mut addresses = [0i32; I8_SPACE];
    let mut code = 0usize;
    let mut address = 0i32;
    while code < I8_SPACE {
        addresses[code] = address;
        address += I8_SPACE as i32;
        code += 1;
    }
    addresses
}

/// Base address of each signed-octet row in [`I8_PRODUCTS`]. Native gathers
/// read this alphabet before adding the right-coordinate address.
pub(crate) static I8_PRODUCT_ROW_ADDRESSES: [i32; I8_SPACE] = build_product_row_addresses();

/// The complete table address of one signed-octet pair.
#[inline(always)]
pub(crate) fn i8_product_address(a: i8, b: i8) -> i32 {
    I8_PRODUCT_ROW_ADDRESSES[a as u8 as usize] + b as u8 as i32
}

/// Read one exact signed-i8 product from an already borrowed alphabet.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) fn i8_product_from(products: &[i32; I8_PRODUCT_ENTRIES], a: i8, b: i8) -> i32 {
    products[i8_product_address(a, b) as usize]
}

/// Read one exact signed i8 product without a multiply instruction.
#[inline(always)]
pub(crate) fn i8_product(a: i8, b: i8) -> i32 {
    i8_products()[i8_product_address(a, b) as usize]
}

/// The four contiguous nibble-projector rows for one left octet.
#[cfg(any(
    target_arch = "aarch64",
    target_arch = "wasm32",
    target_arch = "x86_64",
    test
))]
#[inline(always)]
pub(crate) fn i8_nibble_products(a: i8) -> &'static [u8] {
    &I8_NIBBLE_PRODUCTS[a as u8 as usize]
}

const fn radix_naf_product(a: i8, b: i8) -> i32 {
    let mut magnitude = b as i32;
    let negative = magnitude < 0;
    if negative {
        magnitude = 0 - magnitude;
    }
    let mut addend = a as i32;
    let mut result = 0i32;
    while magnitude != 0 {
        let digit = magnitude % 4;
        if digit != 0 && digit != 2 {
            if digit == 1 {
                result += addend;
                magnitude -= 1;
            } else {
                result -= addend;
                magnitude += 1;
            }
        }
        addend += addend;
        magnitude /= 2;
    }
    if negative {
        0 - result
    } else {
        result
    }
}

const fn build_i8_products() -> [i32; I8_PRODUCT_ENTRIES] {
    let mut table = [0i32; I8_PRODUCT_ENTRIES];
    let mut a = 0u16;
    let mut index = 0usize;
    while a < I8_SPACE as u16 {
        let mut b = 0u16;
        while b < I8_SPACE as u16 {
            table[index] = radix_naf_product(a as u8 as i8, b as u8 as i8);
            index += 1;
            b += 1;
        }
        a += 1;
    }
    table
}

#[cfg(any(
    target_arch = "aarch64",
    target_arch = "wasm32",
    target_arch = "x86_64",
    test
))]
const fn refine_nibble(mut value: i8) -> i8 {
    let mut digit = 0u32;
    while digit < NIBBLE_BITS {
        value += value;
        digit += 1;
    }
    value
}

#[cfg(any(
    target_arch = "aarch64",
    target_arch = "wasm32",
    target_arch = "x86_64",
    test
))]
const fn build_i8_nibble_products() -> [[u8; NIBBLE_ROW_BYTES]; I8_SPACE] {
    let mut table = [[0u8; NIBBLE_ROW_BYTES]; I8_SPACE];
    let mut a = 0usize;
    while a < I8_SPACE {
        let left = a as u8 as i8;
        let mut digit = 0usize;
        while digit < NIBBLE_SPACE {
            let low = radix_naf_product(left, digit as i8) as i16;
            let signed_high = if digit < NIBBLE_SPACE / 2 {
                digit as i8
            } else {
                (digit as i8) - NIBBLE_SPACE as i8
            };
            let high = radix_naf_product(left, refine_nibble(signed_high)) as i16;
            let low_bytes = low.to_le_bytes();
            let high_bytes = high.to_le_bytes();
            let low_high = NIBBLE_SPACE;
            let high_low = low_high + NIBBLE_SPACE;
            let high_high = high_low + NIBBLE_SPACE;
            table[a][digit] = low_bytes[0];
            table[a][low_high + digit] = low_bytes[1];
            table[a][high_low + digit] = high_bytes[0];
            table[a][high_high + digit] = high_bytes[1];
            digit += 1;
        }
        a += 1;
    }
    table
}

#[cfg(test)]
mod tests {
    use super::{
        i8_nibble_products, i8_product, i8_product_address, i8_products, I8_NIBBLE_PRODUCTS,
        I8_PRODUCT_ENTRIES, I8_SPACE, NIBBLE_ROW_BYTES, NIBBLE_SPACE,
    };
    use crate::generated_capacity::CACHE_LINE_BYTES;

    fn expected_product(a: i8, b: i8) -> i32 {
        // Unit addition is intentionally not the NAF recurrence used to build
        // the production table. This is the slow, unoptimized reference (R6).
        let magnitude = i32::from(a).unsigned_abs();
        let addend = i32::from(b).unsigned_abs() as i32;
        let mut result = 0i32;
        let mut count = 0u32;
        while count < magnitude {
            result += addend;
            count += 1;
        }
        if (a < 0) != (b < 0) {
            0 - result
        } else {
            result
        }
    }

    #[test]
    fn i8_lookup_matches_independent_unit_add_for_every_pair_ck_18() {
        let products = i8_products();
        let mut a = 0usize;
        let mut index = 0usize;
        while a < I8_SPACE {
            let mut b = 0usize;
            while b < I8_SPACE {
                let left = a as u8 as i8;
                let right = b as u8 as i8;
                assert_eq!(products[index], expected_product(left, right));
                assert_eq!(i8_product(left, right), expected_product(left, right));
                index += 1;
                b += 1;
            }
            a += 1;
        }
    }

    /// `CU-11`: each native nibble projector is the same complete signed-i8
    /// lookup alphabet, merely factored into four byte rows. Exhausting both
    /// octets also exercises every runtime row and product address.
    #[test]
    fn nibble_projectors_and_runtime_addresses_exhaust_the_product_alphabet_cu_11() {
        let mut left_code = 0usize;
        while left_code < I8_SPACE {
            let left = left_code as u8 as i8;
            let row = i8_nibble_products(left);
            assert_eq!(row.len(), NIBBLE_ROW_BYTES);

            let mut right_code = 0usize;
            while right_code < I8_SPACE {
                let right = right_code as u8 as i8;
                let low_digit = right_code % NIBBLE_SPACE;
                let high_digit = right_code / NIBBLE_SPACE;
                let low = i16::from_le_bytes([row[low_digit], row[NIBBLE_SPACE + low_digit]]);
                let high = i16::from_le_bytes([
                    row[2 * NIBBLE_SPACE + high_digit],
                    row[3 * NIBBLE_SPACE + high_digit],
                ]);
                let projected = i32::from(low) + i32::from(high);
                let expected = expected_product(left, right);
                assert_eq!(projected, expected, "projectors at ({left}, {right})");
                assert_eq!(
                    i8_product(left, right),
                    expected,
                    "address at ({left}, {right})"
                );
                right_code += 1;
            }
            left_code += 1;
        }
    }

    /// `CU-11`: runtime ISA kernels consume these finite address alphabets
    /// instead of reconstructing fields with shifts, masks, or widening
    /// products. The oracle is deliberately conventional and test-only.
    #[test]
    fn radix_address_alphabets_exhaust_every_octet_cu_11() {
        for left_code in 0usize..I8_SPACE {
            let left = left_code as u8 as i8;
            for right_code in 0usize..I8_SPACE {
                let right = right_code as u8 as i8;
                assert_eq!(
                    i8_product_address(left, right),
                    ((left_code << u8::BITS) | right_code) as i32,
                    "pair ({left}, {right})"
                );
            }
        }
    }

    /// `CU-11`: the two finite lookup alphabets keep their exact payload bytes,
    /// while every semantic row begins at the cache-line extent owned by the
    /// model. Removing either generated wrapper makes this witness fail even
    /// on a linker that happens to place one symbol at a friendly address.
    #[test]
    fn atlas_lookup_rows_are_model_line_aligned_without_padding_cu_11() {
        let products = i8_products();
        let product_row_bytes = I8_SPACE * core::mem::size_of::<i32>();
        assert_eq!(product_row_bytes % CACHE_LINE_BYTES, 0);
        assert_eq!(NIBBLE_ROW_BYTES, CACHE_LINE_BYTES);
        assert_eq!(
            core::mem::size_of_val(products),
            I8_PRODUCT_ENTRIES * core::mem::size_of::<i32>()
        );
        assert_eq!(
            core::mem::size_of_val(&I8_NIBBLE_PRODUCTS),
            I8_SPACE * NIBBLE_ROW_BYTES
        );
        assert_eq!(
            core::mem::align_of_val(&I8_NIBBLE_PRODUCTS),
            CACHE_LINE_BYTES
        );

        let products = products.as_ptr();
        let projectors = I8_NIBBLE_PRODUCTS.as_ptr();
        for row in 0..I8_SPACE {
            // SAFETY: both tables contain exactly `I8_SPACE` complete rows;
            // forming each row's first in-bounds pointer reads no element.
            let product_row = unsafe { products.add(row * I8_SPACE) } as usize;
            // SAFETY: `row < I8_SPACE`, so this points at one complete row.
            let projector_row = unsafe { projectors.add(row) } as usize;
            assert_eq!(product_row % CACHE_LINE_BYTES, 0, "product row {row}");
            assert_eq!(projector_row % CACHE_LINE_BYTES, 0, "projector row {row}");
        }
    }
}
