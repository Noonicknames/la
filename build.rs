#[cfg(all(feature = "static-intelmkl", feature = "static-openblas"))]
compile_error!("Cannot have multiple statically linked linear algebra libraries.");

fn main() {
    #[cfg(feature = "static-intelmkl")]
    {
        println!("cargo::rustc-link-search=/opt/intel/mkl/lib/intel64/");
        println!("cargo::rustc-link-search=/opt/intel/lib/intel64");

        println!("cargo::rustc-link-lib=static=mkl_intel_lp64");
        println!("cargo::rustc-link-lib=static=mkl_sequential");
        println!("cargo::rustc-link-lib=static=mkl_core");
    }

    #[cfg(feature = "static-openblas")]
    {
        println!("cargo::rustc-link-lib=static=openblas");
    }
}
