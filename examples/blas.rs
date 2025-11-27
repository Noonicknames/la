use la::{BlasLib, Threading};

fn main() {
    let blas_lib = BlasLib::new().expect("Failed to include Blas backend.");
    blas_lib.set_threading(Threading::Multithreaded);
    blas_lib.set_num_threads(8);
    let blas = blas_lib.functions();
}
