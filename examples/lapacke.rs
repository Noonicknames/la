use la::{BlasLib, LapackeLib, Threading};

fn main() {
    let blas_lib = BlasLib::new().expect("Failed to include Blas backend.");
    blas_lib.set_threading(Threading::Multithreaded);
    blas_lib.set_num_threads(8);
    let lapacke_lib = LapackeLib::new(&blas_lib).unwrap();
    let lapacke = lapacke_lib.functions();
}
