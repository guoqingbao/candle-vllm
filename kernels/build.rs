use anyhow::Result;
use std::fs::read_to_string;
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::path::PathBuf;

fn read_lines(filename: &str) -> Vec<String> {
    let mut result = Vec::new();

    for line in read_to_string(filename).unwrap().lines() {
        result.push(line.to_string())
    }

    result
}

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/pagedattention.cu");
    println!("cargo:rerun-if-changed=src/copy_blocks_kernel.cu");
    println!("cargo:rerun-if-changed=src/reshape_and_cache_kernel.cu");
    println!("cargo:rerun-if-changed=src/marlin_cuda_kernel.cu");
    println!("cargo:rerun-if-changed=src/gptq_cuda_kernel.cu");
    println!("cargo:rerun-if-changed=src/nonzero_bitwise.cu");
    println!("cargo:rerun-if-changed=src/sort.cu");
    let build_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap_or("".to_string()));
    let mut builder = bindgen_cuda::Builder::default().arg("--expt-relaxed-constexpr");
    let mut is_target_msvc = false;
    if let Ok(target) = std::env::var("TARGET") {
        if target.contains("msvc") {
            is_target_msvc = true;
            builder = builder.arg("-D_USE_MATH_DEFINES");
        }
    }

    if !is_target_msvc {
        builder = builder.arg("-Xcompiler").arg("-fPIC");
    }

    println!("cargo:info={builder:?}");
    builder.build_lib(build_dir.join("libpagedattention.a"));

    let bindings = builder.build_ptx().unwrap();
    bindings.write("src/lib.rs").unwrap();

    let kernel_dir = PathBuf::from("../kernels/");
    let absolute_kernel_dir = std::fs::canonicalize(&kernel_dir).unwrap();

    println!(
        "cargo:rustc-link-search=native={}",
        absolute_kernel_dir.display()
    );
    println!("cargo:rustc-link-search={}", build_dir.display());
    println!("cargo:rustc-link-lib=pagedattention");
    println!("cargo:rustc-link-lib=dylib=cudart");

    let contents = read_lines("src/lib.rs");
    for line in contents {
        if line == "pub mod ffi;" {
            return Ok(());
        }
    }
    let mut file = OpenOptions::new()
        .write(true)
        .append(true)
        .open("src/lib.rs")
        .unwrap();
    //Expose paged attention interface to Rust
    if let Err(e) = writeln!(file, "pub mod ffi;") {
        anyhow::bail!("error while building dependencies: {:?}\n", e,)
    } else {
        Ok(())
    }
}
