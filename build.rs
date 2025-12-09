
#[cfg(all(feature = "static-intelmkl", feature = "static-openblas"))]
compile_error!("Cannot have multiple statically linked linear algebra libraries.");

fn main() {
    #[cfg(feature = "static-intelmkl")]
    println!("cargo::rustc-link-lib=mklrt");

    #[cfg(feature = "static-openblas")]
    println!("cargo::rustc-link-lib=openblas");
}