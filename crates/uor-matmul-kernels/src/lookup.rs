//! Lookup-only products for finite integer alphabets.
//!
//! The table is data, not a runtime multiplication routine. Its compile-time
//! constructor uses shift/add decomposition so the shipped i8 product route is
//! a lookup followed by accumulation, including the table-build step.

const I8_SPACE: usize = 256;

/// The exact signed `i8 x i8 -> i32` product table.
pub(crate) static I8_PRODUCTS: [i32; I8_SPACE << 8] = build_i8_products();

/// Read one exact signed i8 product without a multiply instruction.
#[inline(always)]
pub(crate) fn i8_product(a: i8, b: i8) -> i32 {
    let index = (a as u8 as usize) << 8 | b as u8 as usize;
    I8_PRODUCTS[index]
}

const fn shift_add_product(a: i8, b: i8) -> i32 {
    let mut magnitude = b as i32;
    let negative = magnitude < 0;
    if negative {
        magnitude = 0 - magnitude;
    }
    let mut addend = a as i32;
    let mut result = 0i32;
    while magnitude != 0 {
        if magnitude & 1 != 0 {
            result += addend;
        }
        addend += addend;
        magnitude >>= 1;
    }
    if negative {
        0 - result
    } else {
        result
    }
}

const fn build_i8_products() -> [i32; I8_SPACE << 8] {
    let mut table = [0i32; I8_SPACE << 8];
    let mut a = 0u16;
    let mut index = 0usize;
    while a < I8_SPACE as u16 {
        let mut b = 0u16;
        while b < I8_SPACE as u16 {
            table[index] = shift_add_product(a as u8 as i8, b as u8 as i8);
            index += 1;
            b += 1;
        }
        a += 1;
    }
    table
}

#[cfg(test)]
mod tests {
    use super::{i8_product, I8_PRODUCTS, I8_SPACE};

    fn expected_product(a: i8, b: i8) -> i32 {
        let mut magnitude = i32::from(a).unsigned_abs() as i32;
        let addend = i32::from(b).unsigned_abs() as i32;
        let mut result = 0i32;
        let mut shifted = addend;
        while magnitude != 0 {
            if magnitude & 1 != 0 {
                result += shifted;
            }
            shifted += shifted;
            magnitude >>= 1;
        }
        if (a < 0) != (b < 0) {
            0 - result
        } else {
            result
        }
    }

    #[test]
    fn i8_lookup_matches_independent_shift_add_for_every_pair_ck_18() {
        let mut a = 0usize;
        let mut index = 0usize;
        while a < I8_SPACE {
            let mut b = 0usize;
            while b < I8_SPACE {
                let left = a as u8 as i8;
                let right = b as u8 as i8;
                assert_eq!(I8_PRODUCTS[index], expected_product(left, right));
                assert_eq!(i8_product(left, right), expected_product(left, right));
                index += 1;
                b += 1;
            }
            a += 1;
        }
    }
}
