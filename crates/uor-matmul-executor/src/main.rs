//! `CB-11`: the parity checks, run on a Cortex-M target under qemu-system.
//!
//! `CB-01` through `CB-10` run wherever `cargo test` runs. A Cortex-M target
//! is not one of those places --- no user-mode emulator covers it --- so the
//! sequences a Cortex-M build offers were, until this binary, believed rather
//! than run. This is the run: the same check bodies the host test calls
//! ([`uor_matmul_kernels::parity`]), on the reduced corpus, with the scratch
//! statically sized, printing one line per family over semihosting and
//! `CB-11: PASS` at the end. The harness (`just cortex-m-run`) fails on the
//! marker's absence, so a crash mid-suite is a failure, and the panic handler
//! prints `CB-11: FAIL` with the location so a disagreement is diagnosable.

#![cfg_attr(all(target_arch = "arm", target_os = "none"), no_std, no_main)]

#[cfg(all(target_arch = "arm", target_os = "none"))]
mod device {
    use cortex_m_rt::entry;
    use cortex_m_semihosting::{debug, hprintln};
    use uor_matmul_kernels::{
        available_i16, available_i16_modular, available_i32_exact, available_i32_modular,
        available_i64_exact, available_i64_modular, available_i8, available_i8_narrow,
        available_reduce_i16, available_reduce_i16_modular, available_reduce_i32_exact,
        available_reduce_i32_modular, available_reduce_i64_exact, available_reduce_i64_modular,
        available_reduce_i8, available_table_i16, available_table_i32_modular,
        available_table_i64_modular, available_table_i8, available_tropical_i16, parity,
        MAX_TILE_LANES, TROP_I16_MAX_BOUND, TROP_ZERO,
    };

    // The scratch geometry. Every bound below is read off the reduced corpus
    // the same way the harness reads it, and the parity functions assert the
    // scratch is big enough before writing a byte --- so a sequence that
    // outgrows one of these constants fails the run loudly rather than
    // overwriting the stack quietly.

    /// The reduced corpus's deepest walk (65) rounded up to a `k_group` of 16.
    const KC: usize = 80;
    /// The widest panel any shipped kernel declares is 16 lanes.
    const PANEL: usize = 16 * KC;
    /// `TABLE_REDUCED`'s largest code space, tile height, block, and group.
    const SPACE: usize = 256;
    const ROWS: usize = 16;
    const BLOCK: usize = 8;
    const DEPTH: usize = 5;
    /// The largest slab: the largest space is a power of two, so `space * rows`.
    const STACK: usize = DEPTH * SPACE * ROWS;
    /// The code stream: `(group - 1) * stride + depth` at `stride = depth + 3`.
    const STREAM: usize = (ROWS - 1) * (DEPTH + 3) + DEPTH;

    /// One tile family's parity sweep, on stack scratch sized for the corpus.
    macro_rules! tile_check {
        ($name:ident, $specs:expr, $salts:expr, $maps:expr, $dot:expr, $e:ty, $l:ty) => {
            #[inline(never)]
            fn $name() -> usize {
                let mut pa = [<$e>::default(); PANEL];
                let mut pb = [<$e>::default(); PANEL];
                let mut lane_a = [<$e>::default(); KC];
                let mut lane_b = [<$e>::default(); KC];
                let mut acc = [<$l>::default(); MAX_TILE_LANES];
                let mut want = [<$l>::default(); MAX_TILE_LANES];
                let mut scratch = parity::TileScratch {
                    pa: &mut pa,
                    pb: &mut pb,
                    lane_a: &mut lane_a,
                    lane_b: &mut lane_b,
                    acc: &mut acc,
                    want: &mut want,
                };
                parity::tile_parity(
                    stringify!($name),
                    $specs,
                    parity::DEPTHS_REDUCED,
                    $salts,
                    $maps,
                    $dot,
                    &mut scratch,
                )
            }
        };
    }

    /// One reduce family's parity sweep.
    macro_rules! reduce_check {
        ($name:ident, $specs:expr, $salts:expr, $maps:expr, $dot:expr, $e:ty, $l:ty) => {
            #[inline(never)]
            fn $name() -> usize {
                let mut pa = [<$e>::default(); PANEL];
                let mut pb = [<$e>::default(); KC];
                let mut acc = [<$l>::default(); 16];
                let mut scratch = parity::ReduceScratch {
                    pa: &mut pa,
                    pb: &mut pb,
                    acc: &mut acc,
                };
                parity::reduce_parity(
                    stringify!($name),
                    $specs,
                    parity::DEPTHS_REDUCED,
                    $salts,
                    $maps,
                    $dot,
                    &mut scratch,
                )
            }
        };
    }

    tile_check!(
        tile_i8,
        available_i8().chain(available_i8_narrow()),
        (0xC0, 0x0D),
        (|v| v as i8, |v| v as i8),
        parity::dot_i8,
        i8,
        i32
    );
    tile_check!(
        tile_i16,
        available_i16(),
        (7, 8),
        (|v| (v * 251) as i16, |v| (v * 251) as i16),
        parity::dot_i16,
        i16,
        i64
    );
    tile_check!(
        tile_i32_exact,
        available_i32_exact(),
        (9, 10),
        (
            |v| ((v & 0xFFFF) - 0x8000) as i32,
            |v| ((v & 0xFFFF) - 0x8000) as i32
        ),
        parity::dot_i32_exact,
        i32,
        i64
    );
    tile_check!(
        tile_i32_modular,
        available_i32_modular(),
        (11, 12),
        (
            |v| (v.wrapping_mul(99_991)) as i32,
            |v| (v.wrapping_mul(65_537)) as i32
        ),
        parity::dot_i32_mod,
        i32,
        i32
    );
    tile_check!(
        tile_i16_modular,
        available_i16_modular(),
        (15, 16),
        (|v| v as i16, |v| v as i16),
        parity::dot_i16_mod,
        i16,
        i32
    );
    tile_check!(
        tile_i64_exact,
        available_i64_exact(),
        (17, 18),
        (
            |v| (v & 0xFFFF_FFFF) - 0x8000_0000,
            |v| (v & 0xFFFF_FFFF) - 0x8000_0000
        ),
        parity::dot_i64_exact,
        i64,
        i128
    );
    tile_check!(
        tile_i64_modular,
        available_i64_modular(),
        (13, 14),
        (
            |v| v.wrapping_mul(0x9E37_79B9_7F4A_7C15u64 as i64),
            |v| v.wrapping_mul(0x1000_0000_01B3)
        ),
        parity::dot_i64_mod,
        i64,
        i64
    );

    reduce_check!(
        reduce_i8,
        available_reduce_i8(),
        (21, 22),
        (|v| v as i8, |v| v as i8),
        parity::dot_i8,
        i8,
        i32
    );
    reduce_check!(
        reduce_i16,
        available_reduce_i16(),
        (23, 24),
        (|v| v as i16, |v| v as i16),
        parity::dot_i16,
        i16,
        i64
    );
    reduce_check!(
        reduce_i32_exact,
        available_reduce_i32_exact(),
        (25, 26),
        (
            |v| ((v & 0xFFFF) - 0x8000) as i32,
            |v| ((v & 0xFFFF) - 0x8000) as i32
        ),
        parity::dot_i32_exact,
        i32,
        i64
    );
    reduce_check!(
        reduce_i32_modular,
        available_reduce_i32_modular(),
        (27, 28),
        (
            |v| v.wrapping_mul(99_991) as i32,
            |v| v.wrapping_mul(65_537) as i32
        ),
        parity::dot_i32_mod,
        i32,
        i32
    );
    reduce_check!(
        reduce_i16_modular,
        available_reduce_i16_modular(),
        (29, 30),
        (|v| v as i16, |v| v as i16),
        parity::dot_i16_mod,
        i16,
        i32
    );
    reduce_check!(
        reduce_i64_exact,
        available_reduce_i64_exact(),
        (31, 32),
        (
            |v| (v & 0xFFFF_FFFF) - 0x8000_0000,
            |v| (v & 0xFFFF_FFFF) - 0x8000_0000
        ),
        parity::dot_i64_exact,
        i64,
        i128
    );
    reduce_check!(
        reduce_i64_modular,
        available_reduce_i64_modular(),
        (33, 34),
        (
            |v| v.wrapping_mul(0x1000_0000_01B3),
            |v| v.wrapping_mul(0x9E37_79B9)
        ),
        parity::dot_i64_mod,
        i64,
        i64
    );

    /// The packed tropical family, at each of the three bounds it declares.
    ///
    /// Hand-written rather than one `tile_check!` because the family is swept
    /// at three declared bounds and that macro carries a single fill: the bound
    /// is what decides where the finite range ends and the absorbed one begins,
    /// so one fill would leave two thirds of `CB-13` unrun here. The bounds,
    /// the salts and the fills are [`parity::TROP_SWEEPS`], the same list the
    /// host harness reads.
    ///
    /// The extremes are in the same function and not a separate one, because on
    /// this target they are the half that has teeth: a Cortex-M build offers
    /// only the reference, so what is being run here is the *representation* ---
    /// the saturating add at two masked operands, which no random fill reaches.
    #[inline(never)]
    fn tile_tropical() -> usize {
        let mut pa = [TROP_ZERO; PANEL];
        let mut pb = [TROP_ZERO; PANEL];
        let mut lane_a = [TROP_ZERO; KC];
        let mut lane_b = [TROP_ZERO; KC];
        let mut acc = [TROP_ZERO; MAX_TILE_LANES];
        let mut want = [TROP_ZERO; MAX_TILE_LANES];
        let mut compared = 0;
        for (salts, map) in parity::TROP_SWEEPS {
            let mut scratch = parity::TileScratch {
                pa: &mut pa,
                pb: &mut pb,
                lane_a: &mut lane_a,
                lane_b: &mut lane_b,
                acc: &mut acc,
                want: &mut want,
            };
            compared += parity::tile_parity(
                "tile_tropical",
                available_tropical_i16(),
                parity::DEPTHS_REDUCED,
                salts,
                (map, map),
                parity::dot_trop_i16,
                &mut scratch,
            );
        }
        let mut scratch = parity::TileScratch {
            pa: &mut pa,
            pb: &mut pb,
            lane_a: &mut lane_a,
            lane_b: &mut lane_b,
            acc: &mut acc,
            want: &mut want,
        };
        compared += parity::tile_extremes(
            "tile_tropical extremes",
            available_tropical_i16(),
            parity::DEPTHS_REDUCED,
            &parity::TROP_EXTREMES,
            parity::trop_expect,
            &mut scratch,
        );
        // The separation the family turns on, read on the target's own
        // arithmetic rather than assumed from the host's: an absorbed lane must
        // lose a `max` against the lowest finite result there is.
        let b = TROP_I16_MAX_BOUND as i16;
        assert!(
            TROP_ZERO.saturating_add(b) < (-b).saturating_add(-b),
            "CB-13: the absorbed range must sit strictly below the finite one"
        );
        compared + 1
    }

    /// The tile sequences at the extremes of their declared alphabets.
    #[inline(never)]
    fn tile_extremes() -> usize {
        let mut pa = [0i16; PANEL];
        let mut pb = [0i16; PANEL];
        let mut lane_a = [0i16; KC];
        let mut lane_b = [0i16; KC];
        let mut acc = [0i64; MAX_TILE_LANES];
        let mut want = [0i64; MAX_TILE_LANES];
        let mut compared = 0;
        for spec in available_i16() {
            let (lo, hi) = parity::extremes(spec.max_bound, 32768);
            let values = [
                (lo as i16, lo as i16),
                (lo as i16, hi as i16),
                (hi as i16, hi as i16),
                (hi as i16, lo as i16),
            ];
            let mut scratch = parity::TileScratch {
                pa: &mut pa,
                pb: &mut pb,
                lane_a: &mut lane_a,
                lane_b: &mut lane_b,
                acc: &mut acc,
                want: &mut want,
            };
            compared += parity::tile_extremes(
                "tile_extremes i16",
                core::iter::once(spec),
                &[2, 4, 8, 16, 64],
                &values,
                |kc, a, b| (kc as i64) * i64::from(a) * i64::from(b),
                &mut scratch,
            );
        }
        compared
    }

    /// The reduce sequences at the extremes, where a whole row lands in one
    /// lane.
    #[inline(never)]
    fn reduce_extremes() -> usize {
        let mut pa = [0i8; 16 * 1024];
        let mut pb = [0i8; 1024];
        let mut acc = [0i32; 16];
        let mut scratch = parity::ReduceScratch {
            pa: &mut pa,
            pb: &mut pb,
            acc: &mut acc,
        };
        parity::reduce_extremes(
            "reduce_extremes i8",
            available_reduce_i8(),
            &[16, 64, 1024],
            &[(i8::MIN, i8::MIN)],
            |kc, a, b| (kc as i32) * i32::from(a) * i32::from(b),
            &mut scratch,
        )
    }

    /// One table family's parity sweep against the transcribed model.
    macro_rules! table_check {
        ($name:ident, $available:expr, $ops:expr, $salts:expr, $map:expr, $stack_map:expr,
         $sentinels:expr, $skip:expr, $ragged:expr, $e:ty, $l:ty, $zero:expr) => {
            #[inline(never)]
            fn $name() -> usize {
                let mut book = [<$e>::default(); SPACE * BLOCK];
                let mut flat = [<$e>::default(); ROWS * BLOCK];
                let mut packed = [<$e>::default(); ROWS * BLOCK];
                let mut model = [$zero; SPACE * ROWS];
                let mut out = [$zero; SPACE * ROWS];
                let mut stack = [$zero; STACK];
                let mut off = [0u32; DEPTH * ROWS];
                let mut stream = [0u16; STREAM];
                let mut lane_a = [$zero; ROWS * ROWS];
                let mut lane_b = [$zero; ROWS * ROWS];
                let mut scratch = parity::TableScratch {
                    book: &mut book,
                    flat: &mut flat,
                    packed: &mut packed,
                    model: &mut model,
                    out: &mut out,
                    stack: &mut stack,
                    off: &mut off,
                    stream: &mut stream,
                    lane_a: &mut lane_a,
                    lane_b: &mut lane_b,
                };
                parity::table_parity(
                    stringify!($name),
                    $available,
                    &parity::TABLE_REDUCED,
                    &$ops,
                    $salts,
                    $map,
                    $stack_map,
                    $sentinels,
                    $skip,
                    $ragged,
                    &mut scratch,
                )
            }
        };
    }

    table_check!(
        table_i8,
        available_table_i8,
        parity::ops_i8(),
        (0xb00c, 0xac75),
        |v| ((v % 255) - 127) as i8,
        parity::stack_value32,
        (7, -3),
        128,
        true,
        i8,
        i32,
        0
    );
    table_check!(
        table_i16,
        available_table_i16,
        parity::ops_i16(),
        (0x16b0, 0x16ac),
        |v| ((v % 65535) - 32767) as i16,
        parity::stack_value64,
        (11, -5),
        0,
        false,
        i16,
        i64,
        0
    );
    table_check!(
        table_mod32,
        available_table_i32_modular,
        parity::ops_mod32(),
        (0xb32c, 0xa32c),
        |v| v.wrapping_mul(0x9E37_79B9) as i32,
        parity::stack_value32,
        (7, -3),
        0,
        true,
        i32,
        uor_matmul_kernels::Mod32,
        uor_matmul_kernels::Mod32(0)
    );
    table_check!(
        table_mod64,
        available_table_i64_modular,
        parity::ops_mod64(),
        (0xb64c, 0xa64c),
        |v| v.wrapping_mul(0x9E37_79B9_7F4A_7C15u64 as i64),
        parity::stack_value64,
        (11, -5),
        0,
        false,
        i64,
        uor_matmul_kernels::Mod64,
        uor_matmul_kernels::Mod64(0)
    );

    /// The bound-1 build sweep and the selection half.
    #[inline(never)]
    fn table_bound1() -> usize {
        let mut book = [0i8; SPACE * BLOCK];
        let mut flat = [0i8; ROWS * BLOCK];
        let mut packed = [0i8; ROWS * BLOCK];
        let mut model = [0i32; SPACE * ROWS];
        let mut out = [0i32; SPACE * ROWS];
        let mut stack = [0i32; 0];
        let mut off = [0u32; 0];
        let mut stream = [0u16; 0];
        let mut lane_a = [0i32; 0];
        let mut lane_b = [0i32; 0];
        let mut scratch = parity::TableScratch {
            book: &mut book,
            flat: &mut flat,
            packed: &mut packed,
            model: &mut model,
            out: &mut out,
            stack: &mut stack,
            off: &mut off,
            stream: &mut stream,
            lane_a: &mut lane_a,
            lane_b: &mut lane_b,
        };
        parity::table_bound1(
            "table_bound1",
            available_table_i8,
            &parity::BOUND1_REDUCED,
            &parity::ops_i8(),
            (0xb10c, 0xb10a),
            |v| ((v % 3) - 1) as i8,
            &mut scratch,
        )
    }

    /// One check's report: the comparison count, and the vacuous case named.
    ///
    /// A sweep that compared nothing would pass while asserting nothing ---
    /// the failure mode `CB-02`'s comment in `tests/parity.rs` records --- so
    /// zero is a panic here, which the handler turns into `FAIL`.
    fn run(label: &str, compared: usize) {
        assert!(compared > 0, "CB-11 {label} compared nothing");
        hprintln!("CB-11 {}: {} comparisons", label, compared);
    }

    #[entry]
    fn main() -> ! {
        run("tile i8", tile_i8());
        run("tile i16", tile_i16());
        run("tile i32 exact", tile_i32_exact());
        run("tile i32 modular", tile_i32_modular());
        run("tile i16 modular", tile_i16_modular());
        run("tile i64 exact", tile_i64_exact());
        run("tile i64 modular", tile_i64_modular());
        run("tile tropical", tile_tropical());
        run("reduce i8", reduce_i8());
        run("reduce i16", reduce_i16());
        run("reduce i32 exact", reduce_i32_exact());
        run("reduce i32 modular", reduce_i32_modular());
        run("reduce i16 modular", reduce_i16_modular());
        run("reduce i64 exact", reduce_i64_exact());
        run("reduce i64 modular", reduce_i64_modular());
        run("tile extremes", tile_extremes());
        run("reduce extremes", reduce_extremes());
        run("table i8", table_i8());
        run("table i16", table_i16());
        run("table mod32", table_mod32());
        run("table mod64", table_mod64());
        run("table bound1", table_bound1());
        hprintln!("CB-11: PASS");
        debug::exit(debug::EXIT_SUCCESS);
        loop {}
    }
}

/// The failure half of the marker protocol: a disagreement is a panic, and a
/// panic is a printed `FAIL` with the location, then a semihosting exit with a
/// failure status --- not a halt, because the diagnosis is the point.
#[cfg(all(target_arch = "arm", target_os = "none"))]
#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    cortex_m_semihosting::hprintln!("CB-11: FAIL");
    if let Some(loc) = info.location() {
        cortex_m_semihosting::hprintln!("CB-11: at {}:{}", loc.file(), loc.line());
    }
    cortex_m_semihosting::debug::exit(cortex_m_semihosting::debug::EXIT_FAILURE);
    loop {}
}

#[cfg(not(all(target_arch = "arm", target_os = "none")))]
fn main() {
    println!(
        "uor-matmul-executor is a Cortex-M binary; `just cortex-m-run` builds it for \
         thumbv6m-none-eabi and thumbv7em-none-eabihf and runs it under qemu-system"
    );
}
