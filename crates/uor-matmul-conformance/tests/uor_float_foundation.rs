//! Executable mathematics for the pure-UOR float representation.
//!
//! These checks deliberately live in the conformance crate. They establish the
//! representation and admission laws before a shipped arithmetic body consumes
//! them, while leaving the portable float reference untouched (R6, R9).

use std::collections::BTreeMap;

use uor_matmul_model::derive;

const SCOPE: usize = 4;
const MODALITY: usize = 3;
const CONTEXT: usize = 8;
const CARRIER: usize = MODALITY * CONTEXT;

fn naf(mut n: i128) -> Vec<i8> {
    let mut out = Vec::new();
    while n != 0 {
        let digit = if n.rem_euclid(2) == 0 {
            0
        } else if n.rem_euclid(4) == 1 {
            1
        } else {
            -1
        };
        out.push(digit);
        n = (n - i128::from(digit)) / 2;
    }
    out
}

fn eval_naf(digits: &[i8]) -> i128 {
    digits
        .iter()
        .enumerate()
        .map(|(at, &digit)| i128::from(digit) << at)
        .sum()
}

fn is_canonical(digits: &[i8]) -> bool {
    digits.iter().all(|&d| (-1..=1).contains(&d))
        && digits.windows(2).all(|w| w[0] == 0 || w[1] == 0)
        && digits.last().is_none_or(|&d| d != 0)
}

fn canonical_dyadic(mut coefficient: i128, mut exponent: i32) -> (Vec<i8>, i32) {
    if coefficient == 0 {
        return (Vec::new(), 0);
    }
    while coefficient.rem_euclid(2) == 0 {
        coefficient /= 2;
        exponent += 1;
    }
    (naf(coefficient), exponent)
}

fn address(grade: i64) -> (i64, usize, usize) {
    let page_sites = i64::try_from(SCOPE * CONTEXT).expect("the Atlas page fits i64");
    let word = grade.div_euclid(page_sites);
    let within = usize::try_from(grade.rem_euclid(page_sites))
        .expect("Euclidean remainder is a non-negative page offset");
    (word, within / CONTEXT, within % CONTEXT)
}

fn grade(word: i64, scope: usize, context: usize) -> i64 {
    let page_sites = i64::try_from(SCOPE * CONTEXT).expect("the Atlas page fits i64");
    let within = i64::try_from(scope * CONTEXT + context).expect("the page offset fits i64");
    word * page_sites + within
}

#[test]
fn naf_is_the_canonical_quotient_section_ck_19() {
    for n in -16_384i128..=16_384 {
        let digits = naf(n);
        assert!(is_canonical(&digits), "{n}: {digits:?}");
        assert_eq!(eval_naf(&digits), n, "{n}: evaluation changed");
        assert_eq!(
            naf(eval_naf(&digits)),
            digits,
            "{n}: section is not idempotent"
        );

        let negated: Vec<i8> = digits.iter().map(|d| -*d).collect();
        assert_eq!(naf(-n), negated, "{n}: mu must be coefficient negation");
        for &digit in &digits {
            let row = i16::from(digit) + 1;
            let mirrored = 2 - row;
            assert_eq!(mirrored - 1, -i16::from(digit));
        }
    }

    // A canonical finite-support signed-binary word is unique. Exhaustively
    // enumerate every such word through ten sites and reject a second word at
    // the same quotient value.
    let mut seen: BTreeMap<i128, Vec<i8>> = BTreeMap::new();
    fn enumerate(at: usize, digits: &mut Vec<i8>, seen: &mut BTreeMap<i128, Vec<i8>>) {
        if at == 10 {
            let mut canonical = digits.clone();
            while canonical.last() == Some(&0) {
                canonical.pop();
            }
            if !is_canonical(&canonical) {
                return;
            }
            let value = eval_naf(&canonical);
            if let Some(previous) = seen.insert(value, canonical.clone()) {
                assert_eq!(previous, canonical, "two canonical words represent {value}");
            }
            return;
        }
        for digit in [-1, 0, 1] {
            if digit != 0 && digits.last().is_some_and(|&d| d != 0) {
                continue;
            }
            digits.push(digit);
            enumerate(at + 1, digits, seen);
            digits.pop();
        }
    }
    enumerate(0, &mut Vec::new(), &mut seen);

    // Laurent grades have one further gauge: remove every factor of two from
    // the coefficient and move it into the origin. That makes the global
    // finite-support representative unique, including at negative grades.
    for coefficient in -511i128..=511 {
        for exponent in -12..=12 {
            let (digits, origin) = canonical_dyadic(coefficient, exponent);
            assert!(is_canonical(&digits));
            if coefficient == 0 {
                assert!(digits.is_empty());
                assert_eq!(origin, 0);
            } else {
                assert_ne!(eval_naf(&digits).rem_euclid(2), 0);
                let mut lhs = coefficient;
                let mut rhs = eval_naf(&digits);
                if exponent < origin {
                    rhs <<= (origin - exponent) as usize;
                } else {
                    lhs <<= (exponent - origin) as usize;
                }
                assert_eq!(lhs, rhs);
            }
        }
    }
}

#[test]
fn naf_pages_cross_context_scope_and_depth_ck_19() {
    let span = i64::try_from(5 * SCOPE * CONTEXT).expect("the test span fits i64");
    for grade_value in -span..=span {
        let (word, scope, context) = address(grade_value);
        assert!(scope < SCOPE);
        assert!(context < CONTEXT);
        assert_eq!(grade(word, scope, context), grade_value);
    }
    assert_eq!(address(7), (0, 0, 7));
    assert_eq!(address(8), (0, 1, 0));
    assert_eq!(address(31), (0, 3, 7));
    assert_eq!(address(32), (1, 0, 0));
    assert_eq!(address(-1), (-1, 3, 7));
    assert_eq!(address(-31), (-1, 0, 1));
    assert_eq!(address(-32), (-1, 0, 0));
    assert_eq!(address(-33), (-2, 3, 7));

    let page_sites = u64::try_from(SCOPE * CONTEXT).expect("the test page width fits u64");
    assert_eq!(derive::atlas_pages(619, page_sites), 20);
    assert_eq!(derive::atlas_pages(4_261, page_sites), 134);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Rat {
    n: i128,
    d: i128,
}

impl Rat {
    const ZERO: Self = Self { n: 0, d: 1 };

    fn new(mut n: i128, mut d: i128) -> Self {
        assert_ne!(d, 0);
        if d < 0 {
            n = -n;
            d = -d;
        }
        let divisor = gcd(n.unsigned_abs(), d as u128) as i128;
        Self {
            n: n / divisor,
            d: d / divisor,
        }
    }

    fn add(self, other: Self) -> Self {
        Self::new(self.n * other.d + other.n * self.d, self.d * other.d)
    }

    fn sub(self, other: Self) -> Self {
        Self::new(self.n * other.d - other.n * self.d, self.d * other.d)
    }

    fn div(self, divisor: i128) -> Self {
        Self::new(self.n, self.d * divisor)
    }

    fn is_dyadic(self) -> bool {
        (self.d as u128).is_power_of_two()
    }
}

fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.max(1)
}

type Carrier = [Rat; CARRIER];

fn at(row: usize, column: usize) -> usize {
    row * CONTEXT + column
}

fn from_i64(x: &[i64; CARRIER]) -> Carrier {
    let mut out = [Rat::ZERO; CARRIER];
    for (dst, &src) in out.iter_mut().zip(x) {
        *dst = Rat::new(i128::from(src), 1);
    }
    out
}

fn sub_carrier(a: &Carrier, b: &Carrier) -> Carrier {
    let mut out = [Rat::ZERO; CARRIER];
    for i in 0..CARRIER {
        out[i] = a[i].sub(b[i]);
    }
    out
}

fn add_carrier(a: &Carrier, b: &Carrier) -> Carrier {
    let mut out = [Rat::ZERO; CARRIER];
    for i in 0..CARRIER {
        out[i] = a[i].add(b[i]);
    }
    out
}

fn mean_modality(x: &Carrier) -> Carrier {
    let mut out = [Rat::ZERO; CARRIER];
    for column in 0..CONTEXT {
        let mut sum = Rat::ZERO;
        for row in 0..MODALITY {
            sum = sum.add(x[at(row, column)]);
        }
        let mean = sum.div(MODALITY as i128);
        for row in 0..MODALITY {
            out[at(row, column)] = mean;
        }
    }
    out
}

fn mean_context(x: &Carrier) -> Carrier {
    let mut out = [Rat::ZERO; CARRIER];
    for row in 0..MODALITY {
        let mut sum = Rat::ZERO;
        for column in 0..CONTEXT {
            sum = sum.add(x[at(row, column)]);
        }
        let mean = sum.div(CONTEXT as i128);
        for column in 0..CONTEXT {
            out[at(row, column)] = mean;
        }
    }
    out
}

fn project(which: usize, x: &Carrier) -> Carrier {
    let mt = mean_modality(x);
    let ct = sub_carrier(x, &mt);
    match which {
        0 => mean_context(&mt),
        1 => mean_context(&ct),
        2 => sub_carrier(&mt, &mean_context(&mt)),
        3 => sub_carrier(&ct, &mean_context(&ct)),
        _ => unreachable!(),
    }
}

fn embed(pattern: [i8; CONTEXT]) -> [i64; CARRIER] {
    let mut out = [0i64; CARRIER];
    for (column, digit) in pattern.into_iter().enumerate() {
        let row = usize::try_from(digit + 1).expect("a signed digit names a modality row");
        out[at(row, column)] = 3 * (column as i64 + 1);
    }
    out
}

fn selected(carrier: &[i64; CARRIER]) -> usize {
    carrier
        .iter()
        .enumerate()
        .max_by_key(|&(index, value)| (*value, std::cmp::Reverse(index)))
        .map(|(index, _)| index)
        .expect("the carrier is nonempty")
}

#[test]
fn atlas_projectors_reconstruct_exactly_ck_20() {
    assert_eq!(derive::atlas_refinement_bits(CONTEXT as u32), 7);
    assert_eq!(
        derive::atlas_refinement_leaves(CONTEXT as u32).as_u128(),
        Some(128)
    );
    assert_eq!(
        derive::atlas_alphabet(SCOPE as u32, MODALITY as u32, CONTEXT as u32).as_u128(),
        Some(12_288)
    );

    let raw = embed([-1, 0, 0, 0, 1, 0, 1, 0]);
    let carrier = from_i64(&raw);
    let blocks = [
        project(0, &carrier),
        project(1, &carrier),
        project(2, &carrier),
        project(3, &carrier),
    ];

    let reconstructed = blocks
        .iter()
        .fold([Rat::ZERO; CARRIER], |sum, block| add_carrier(&sum, block));
    assert_eq!(reconstructed, carrier);
    assert!(blocks.iter().flatten().copied().all(Rat::is_dyadic));

    for (i, block) in blocks.iter().enumerate() {
        assert_eq!(project(i, block), *block, "P{i} is not idempotent");
        for j in 0..4 {
            if i != j {
                assert_eq!(
                    project(j, block),
                    [Rat::ZERO; CARRIER],
                    "P{j} P{i} is not zero"
                );
            }
        }
    }
}

#[test]
fn interaction_separates_equal_marginals_ck_20() {
    let a_digits = [-1, 0, 0, 0, 1, 0, 1, 0];
    let b_digits = [-1, 0, 0, 1, 0, 0, 0, 1];
    assert_eq!(eval_naf(&a_digits), 79);
    assert_eq!(eval_naf(&b_digits), 135);
    // The Atlas page has a fixed eight-context extent; a zero after the last
    // supported Laurent grade is padding in the carrier, not part of the NAF
    // word's finite support.
    assert!(is_canonical(&a_digits[..7]));
    assert!(is_canonical(&b_digits));

    let a_raw = embed(a_digits);
    let b_raw = embed(b_digits);
    let a = from_i64(&a_raw);
    let b = from_i64(&b_raw);
    for block in 0..3 {
        assert_eq!(project(block, &a), project(block, &b), "P{block} differs");
    }
    let delta = sub_carrier(&a, &b);
    assert_eq!(project(3, &delta), delta);
    assert_ne!(project(3, &a), project(3, &b));
    assert_ne!(selected(&a_raw), selected(&b_raw));
}

#[derive(Clone, Copy, Debug)]
struct Coefficient {
    odd: u64,
    grade: i32,
}

fn shift_room(cap: u64, odd: u64) -> Option<i32> {
    if odd == 0 || odd > cap {
        return None;
    }
    let quotient = cap / odd;
    Some(i32::try_from(quotient.ilog2()).expect("a u64 bit count fits i32"))
}

fn side_endpoints(values: &[Coefficient], cap: u64) -> Option<(i32, i32)> {
    let low = values.iter().map(|v| v.grade).min()?;
    let high = values
        .iter()
        .map(|v| Some(v.grade - shift_room(cap, v.odd)?))
        .collect::<Option<Vec<_>>>()?
        .into_iter()
        .max()?;
    Some((low, high))
}

fn common_base_interval(
    a: &[Coefficient],
    cap_a: u64,
    b: &[Coefficient],
    cap_b: u64,
) -> Option<(i32, i32)> {
    let (a0, ah) = side_endpoints(a, cap_a)?;
    let (b0, bh) = side_endpoints(b, cap_b)?;
    (ah <= a0 && bh <= b0).then_some((ah + bh, a0 + b0))
}

fn direct_admits(
    a: &[Coefficient],
    cap_a: u64,
    b: &[Coefficient],
    cap_b: u64,
    base: i32,
    transfer: i32,
) -> bool {
    fn side(values: &[Coefficient], cap: u64, offset: i32) -> bool {
        values.iter().all(|v| {
            let shift = v.grade + offset;
            shift >= 0
                && u32::try_from(shift)
                    .ok()
                    .and_then(|s| v.odd.checked_shl(s))
                    .is_some_and(|packed| packed <= cap)
        })
    }
    side(a, cap_a, transfer) && side(b, cap_b, -transfer - base)
}

fn greedy_points(mut intervals: Vec<(i32, i32)>) -> Vec<i32> {
    intervals.sort_by_key(|&(_, high)| high);
    let mut points = Vec::new();
    for (low, high) in intervals {
        if points
            .last()
            .is_none_or(|&point| point < low || point > high)
        {
            points.push(high);
        }
    }
    points
}

fn exhaustive_minimum(intervals: &[(i32, i32)]) -> usize {
    if intervals.is_empty() {
        return 0;
    }
    let candidates: Vec<i32> = intervals.iter().flat_map(|&(a, b)| [a, b]).collect();
    for count in 1..=candidates.len() {
        fn choose(
            candidates: &[i32],
            intervals: &[(i32, i32)],
            count: usize,
            from: usize,
            points: &mut Vec<i32>,
        ) -> bool {
            if points.len() == count {
                return intervals
                    .iter()
                    .all(|&(low, high)| points.iter().any(|&p| low <= p && p <= high));
            }
            for at in from..candidates.len() {
                points.push(candidates[at]);
                if choose(candidates, intervals, count, at + 1, points) {
                    return true;
                }
                points.pop();
            }
            false
        }
        if choose(&candidates, intervals, count, 0, &mut Vec::new()) {
            return count;
        }
    }
    unreachable!("one endpoint per interval always stabs the family")
}

#[test]
fn common_base_intervals_equal_direct_admission_cd_31() {
    let coefficients = [
        Coefficient { odd: 1, grade: -2 },
        Coefficient { odd: 3, grade: 0 },
        Coefficient { odd: 5, grade: 3 },
    ];
    for cap_a in [7, 15, 31] {
        for cap_b in [7, 15, 31] {
            for a_mask in 1usize..(1 << coefficients.len()) {
                for b_mask in 1usize..(1 << coefficients.len()) {
                    let a: Vec<_> = coefficients
                        .iter()
                        .copied()
                        .enumerate()
                        .filter_map(|(i, v)| ((a_mask >> i) & 1 == 1).then_some(v))
                        .collect();
                    let b: Vec<_> = coefficients
                        .iter()
                        .copied()
                        .enumerate()
                        .filter_map(|(i, v)| ((b_mask >> i) & 1 == 1).then_some(v))
                        .collect();
                    let interval = common_base_interval(&a, cap_a, &b, cap_b);
                    for base in -12..=12 {
                        let direct = (-24..=24)
                            .any(|transfer| direct_admits(&a, cap_a, &b, cap_b, base, transfer));
                        let derived =
                            interval.is_some_and(|(low, high)| low <= base && base <= high);
                        assert_eq!(direct, derived, "A={a:?}, B={b:?}, Q={base}");
                    }
                }
            }
        }
    }

    // Regression for the incorrect cross-endpoint formula: these positions
    // admit only different bases and therefore require two groups.
    let singleton_intervals = vec![(10, 10), (6, 6)];
    assert_eq!(greedy_points(singleton_intervals.clone()).len(), 2);
    assert_eq!(exhaustive_minimum(&singleton_intervals), 2);
}

#[test]
fn greedy_grouping_is_minimum_and_max_base_maximizes_headroom_cd_31() {
    let catalogue = [(-3, -1), (-2, 2), (0, 0), (1, 4), (3, 5), (5, 8)];
    for mask in 0usize..(1 << catalogue.len()) {
        let intervals: Vec<_> = catalogue
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(i, interval)| ((mask >> i) & 1 == 1).then_some(interval))
            .collect();
        assert_eq!(
            greedy_points(intervals.clone()).len(),
            exhaustive_minimum(&intervals)
        );
    }

    // At every admissible base, reified product coefficients are integral.
    // Raising the base divides each by a power of two, so the greatest common
    // base minimizes their absolute sum and maximizes remaining headroom.
    let product_grades = [9, 11, 14, 14];
    let common = (5, 9);
    let magnitude = |base: i32| -> u128 {
        product_grades
            .iter()
            .map(|&g| 1u128 << u32::try_from(g - base).expect("the interval preserves integrality"))
            .sum()
    };
    for base in common.0..common.1 {
        assert!(magnitude(base + 1) < magnitude(base));
    }
    assert_eq!(
        magnitude(common.1),
        (common.0..=common.1).map(magnitude).min().unwrap()
    );
}
