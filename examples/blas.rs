use la::{BlasLib, Threading, Transpose};

fn main() {
    let blas_lib =
        BlasLib::new().expect("Failed to include Blas backend.");
    println!("Using backend: {:?}", blas_lib.backend());
    blas_lib.set_threading(Threading::Multithreaded);
    blas_lib.set_num_threads(8);
    let blas = blas_lib.functions();

    let a = nalgebra::DMatrix::from_iterator(
        10,
        10,
        (0..10)
            .flat_map(|j| (0..10).map(move |i| (i, j)))
            .map(|(i, j)| if i == j { i as f64 + 1.0 } else { 0.0 }),
    );
    let b = nalgebra::DMatrix::from_iterator(
        10,
        10,
        (0..10)
            .flat_map(|j| (0..10).map(move |i| (i, j)))
            .map(|(i, j)| i as f64 + 10.0 * j as f64 ),
    );
    let mut c = nalgebra::DMatrix::zeros(10, 10);

    blas.dgemm(1.0, 0.0, &a, Transpose::None, &b, Transpose::None, &mut c);

    println!("Result: {}", c);
}
