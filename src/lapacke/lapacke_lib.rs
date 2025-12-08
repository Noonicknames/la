use std::sync::Arc;

use libloading::Library;

use crate::{blas::{BlasBackend, BlasLib, util::find_lib_path}, error::LaError};

use super::{LapackeFunctions, LapackeFunctionsStatic};

#[derive(Clone)]
pub struct LapackeLib(Arc<LapackeLibInner>);

impl LapackeLib {
    pub fn new(blas: &BlasLib) -> Result<Self, LaError> {
        Ok(Self(Arc::new(LapackeLibInner::new(blas)?)))
    }
    pub(crate) fn lib(&self) -> Option<&Library> {
        self.0.lib()
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
        blas_lapack_lib: BlasLib,
    },
    OpenBlas {
        _blas_lib: BlasLib,
        lapack_lib: Arc<Library>,
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
                _blas_lib: blas.clone(),
                lapack_lib: {
                    let lib_path = find_lib_path("lapacke")?;
                    Arc::new(unsafe { Library::new(lib_path) }?)
                },
            },
            BlasBackend::Static => Self::Static,
        })
    }
    fn lib(&self) -> Option<&Library> {
        match self {
            Self::IntelMkl { blas_lapack_lib } => blas_lapack_lib.lib(),
            Self::OpenBlas { lapack_lib, .. } => Some(&lapack_lib),
            Self::Static => None
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
