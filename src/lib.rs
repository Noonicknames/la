//! Includes external libraries.
//! 
//! These are loaded dynamically and include BLAS and LAPACK libraries.

mod blas;
mod lapacke;
mod error;
mod types;
pub(crate) mod util;

pub use blas::*;
pub use lapacke::*;
pub use error::*;
pub(crate) use types::*;