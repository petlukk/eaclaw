use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let kernel_dir = PathBuf::from(&manifest_dir).join("../../target/kernels");
    let kernel_dir = kernel_dir.canonicalize().unwrap_or_else(|_| {
        eprintln!(
            "Warning: kernel directory not found at {:?}. Run ./build.sh first.",
            kernel_dir
        );
        kernel_dir
    });

    println!(
        "cargo:rustc-link-search=native={}",
        kernel_dir.display()
    );

    let kernels = [
        "byte_classifier",
        "json_scanner",
        "command_router",
        "leak_scanner",
        "sanitizer",
        "fused_safety",
        "search",
    ];

    for name in &kernels {
        println!("cargo:rustc-link-lib=dylib={name}");
    }

    println!("cargo:rerun-if-changed=../../target/kernels");
}
