
pub use i32::*;


// Luckily, Complex is #[repr(C)]
pub type LaComplexFloat = nalgebra::Complex<f32>;
pub type LaComplexDouble = nalgebra::Complex<f64>;

mod u32 {
    use core::ffi::{c_uint};
    /// The type used for libraries
    pub type LaInt = c_uint;
}


mod i32 {
    use core::ffi::{c_int};
    /// The type used for libraries
    pub type LaInt = c_int;
}

mod i64 {
    use core::ffi::{c_longlong};
    /// The type used for libraries
    pub type LaInt = c_longlong;
}

mod u64 {
    use core::ffi::{c_ulonglong};
    /// The type used for libraries
    pub type LaInt = c_ulonglong;
}