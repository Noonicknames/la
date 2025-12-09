use la::{BlasLib, LapackeLib, Threading};
use nalgebra::{DMatrix, DVector};

fn main() {
    let blas_lib = BlasLib::new().expect("Failed to include Blas backend.");
    println!("Using backend: {:?}", blas_lib.backend());
    // blas_lib.set_threading(Threading::Multithreaded);
    // blas_lib.set_num_threads(8);
    let lapacke_lib = LapackeLib::new(&blas_lib).unwrap();
    let lapacke = lapacke_lib.functions();

    let mut diagonal = DVector::from_vec(vec![1.0; 10]);
    let mut off_diagonal = DVector::from_vec(vec![1.0; 10]);
    let mut isuppz = DVector::from_vec(vec![0; 10]);
    let mut eig_out = DVector::from_vec(vec![0.0; 10]);
    let mut eig_vec_out = DMatrix::zeros(10, 10);

    lapacke
        .dstemr(
            la::EigenRange::All,
            &mut diagonal,
            &mut off_diagonal,
            &mut eig_out,
            &mut isuppz,
            Some(&mut eig_vec_out),
        )
        .unwrap();

    println!("Eigenvalues: {:.3}", eig_out);
    println!("Eigenvectors: {:.3}", eig_vec_out);
}
