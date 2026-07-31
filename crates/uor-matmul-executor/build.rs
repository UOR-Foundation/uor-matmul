//! Puts the directory holding `memory.x` on the linker search path, which is
//! the standard cortex-m-rt pattern: `-Tlink.x` (passed by `just cortex-m-run`)
//! includes `memory.x` by name.

fn main() {
    let dir = std::env::var("CARGO_MANIFEST_DIR").expect("set by cargo");
    println!("cargo:rustc-link-search={dir}");
    println!("cargo:rerun-if-changed=memory.x");
}
