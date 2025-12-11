use std::sync::Arc;

#[cfg(feature = "dynamic")]
use libloading::Library;

use crate::{
    blas::{BlasBackend, BlasLib},
    error::LaError,
};

use super::{LapackeFunctions, LapackeFunctionsStatic};

#[derive(Clone)]
pub struct LapackeLib(Arc<LapackeLibInner>);

impl LapackeLib {
    pub fn new(blas: &BlasLib) -> Result<Self, LaError> {
        Ok(Self(Arc::new(LapackeLibInner::new(blas)?)))
    }

    #[cfg(feature = "dynamic")]
    pub(crate) fn lib(&self) -> Option<&Library> {
        self.0.lib()
    }

    pub fn is_static(&self) -> bool {
        match self.0.as_ref() {
            LapackeLibInner::IntelMkl { .. } | LapackeLibInner::OpenBlas { .. } => false,
            LapackeLibInner::Static => true,
        }
    }

    pub fn functions_static(&self) -> LapackeFunctionsStatic {
        let functions = unsafe { std::mem::transmute(self.functions()) };
        LapackeFunctionsStatic {
            _lib: self.clone(),
            functions,
        }
    }

    pub fn functions(&self) -> LapackeFunctions<'_> {
        LapackeFunctions::from_lib(self)
    }
}

#[derive(Debug)]
pub(super) enum LapackeLibInner {
    IntelMkl {
        #[cfg_attr(not(feature = "dynamic"), allow(unused))]
        blas_lapack_lib: BlasLib,
    },
    OpenBlas {
        blas_lapack_lib: BlasLib,
    },
    Static,
}

impl LapackeLibInner {
    fn new(blas: &BlasLib) -> Result<Self, LaError> {
        Ok(match blas.backend() {
            BlasBackend::IntelMkl => Self::IntelMkl {
                blas_lapack_lib: blas.clone(),
            },
            BlasBackend::OpenBlas => Self::OpenBlas {
                blas_lapack_lib: blas.clone(),
            },
            BlasBackend::Static => Self::Static,
        })
    }
    #[cfg(feature = "dynamic")]
    fn lib(&self) -> Option<&Library> {
        match self {
            Self::IntelMkl { blas_lapack_lib } | Self::OpenBlas { blas_lapack_lib } => {
                blas_lapack_lib.lib()
            }
            Self::Static => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::blas::Threading;

    use super::{BlasLib, LapackeLib};

    #[test]
    fn test_lapacke_sequential() {
        let blas_lib = BlasLib::new().expect("Failed to include Blas.");
        blas_lib.set_threading(Threading::Sequential);
        blas_lib.set_num_threads(8);
        let _blas = blas_lib.functions();
        let lapacke_lib = LapackeLib::new(&blas_lib).expect("Failed to include Lapacke.");
        let _lapacke = lapacke_lib.functions();
    }

    #[test]
    fn test_lapacke_multithreaded() {
        let blas_lib = BlasLib::new().expect("Failed to include Blas.");
        blas_lib.set_threading(Threading::Multithreaded);
        blas_lib.set_num_threads(8);
        let _blas = blas_lib.functions();
        let lapacke_lib = LapackeLib::new(&blas_lib).expect("Failed to include Lapacke.");
        let _lapacke = lapacke_lib.functions();
    }
}
