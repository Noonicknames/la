use std::{path::Path, sync::Arc};

#[cfg(feature = "dynamic")]
use libloading::{Library, Symbol};

use crate::error::LaError;

use super::{
    functions::{BlasFunctions, BlasFunctionsStatic},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BlasBackend {
    IntelMkl,
    OpenBlas,
    Static,
}

impl BlasBackend {
    pub fn lib_name(&self) -> Option<&str> {
        match self {
            Self::IntelMkl => Some("mkl_rt"),
            Self::OpenBlas => Some("openblas"),
            Self::Static => None,
        }
    }
}

#[derive(Debug, Default, Hash, PartialEq, Eq, Clone, Copy)]
pub enum Transpose {
    #[default]
    None,
    Transpose,
    Adjoint,
}

impl Transpose {
    pub fn to_sys(&self) -> cblas_sys::CBLAS_TRANSPOSE {
        match self {
            Self::None => cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
            Self::Transpose => cblas_sys::CBLAS_TRANSPOSE::CblasTrans,
            Self::Adjoint => cblas_sys::CBLAS_TRANSPOSE::CblasConjTrans,
        }
    }
}

#[derive(Clone, Debug)]
pub struct BlasLib(Arc<BlasLibInner>);

#[derive(Debug)]
pub struct BlasLibInner {
    #[cfg(feature = "dynamic")]
    lib: Option<Library>,
    pub backend: BlasBackend,
}

impl Drop for BlasLibInner {
    fn drop(&mut self) {
        #[cfg(feature = "dynamic")]
        if let BlasBackend::OpenBlas = self.backend {}

        #[cfg(feature = "dynamic")]
        if let BlasBackend::IntelMkl = self.backend {
            // If the backend is Intel the library must be loaded in.
            let lib = self.lib.as_ref().unwrap();
            let free_buffers =
                match unsafe { lib.get::<unsafe extern "C" fn()>(b"MKL_Free_Buffers") } {
                    Ok(free_buffers) => free_buffers,
                    Err(why) => {
                        eprintln!("{why}");
                        return;
                    }
                };

            let free_thread_buffers =
                match unsafe { lib.get::<unsafe extern "C" fn()>(b"MKL_Thread_Free_Buffers") } {
                    Ok(free_buffers) => free_buffers,
                    Err(why) => {
                        eprintln!("{why}");
                        return;
                    }
                };

            unsafe {
                free_thread_buffers();
                free_buffers();
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq)]
pub enum Threading {
    #[default]
    Sequential,
    Multithreaded,
}

impl Threading {
    pub fn intel(&self) -> i32 {
        match self {
            Self::Sequential => 1,
            Self::Multithreaded => 0,
        }
    }
}

const BACKEND_PRIORITY: &[BlasBackend] = &[
    #[cfg(feature = "dynamic")]
    BlasBackend::IntelMkl,
    #[cfg(feature = "dynamic")]
    BlasBackend::OpenBlas,
    #[cfg(feature = "static")]
    BlasBackend::Static,
];

impl BlasLib {
    pub fn new() -> Result<Self, LaError> {
        let mut errors = Vec::new();

        for backend in BACKEND_PRIORITY.iter() {
            match Self::with_backend(*backend) {
                Ok(blas_lib) => return Ok(blas_lib),
                Err(err) => errors.push(err),
            }
        }

        Err(LaError::from_iter(errors))
    }

    #[cfg(feature = "dynamic")]
    pub fn lib(&self) -> Option<&Library> {
        self.0.lib.as_ref()
    }

    pub fn with_additional_search_paths<P>(
        additional_search_paths: impl IntoIterator<Item = P> + Clone,
    ) -> Result<Self, LaError>
    where
        P: AsRef<Path>,
    {
        let mut errors = Vec::new();

        for backend in BACKEND_PRIORITY.iter() {
            match Self::with_backend_with_additional_search_paths(
                *backend,
                additional_search_paths.clone(),
            ) {
                Ok(blas_lib) => return Ok(blas_lib),
                Err(err) => errors.push(err),
            }
        }

        Err(LaError::from_iter(errors))
    }

    pub fn set_threading(&self, threading: Threading) {

        match self.backend() {
            #[cfg(feature = "dynamic")]
            BlasBackend::IntelMkl => {
                type MklSetThreadingLayerFn = extern "C" fn(i32);
                let mkl_set_threading_layer: Symbol<MklSetThreadingLayerFn> =
                    unsafe { self.0.lib.as_ref().unwrap().get(b"MKL_Set_Threading_Layer") }
                        .expect("Cannot find MKL_Set_Threading_layer");
                mkl_set_threading_layer(threading.intel())
            }
            #[cfg(not(feature = "dynamic"))]
            BlasBackend::IntelMkl => {
                _ = threading;
            }
            #[cfg(feature = "dynamic")]
            BlasBackend::OpenBlas => {}
            #[cfg(not(feature = "dynamic"))]
            BlasBackend::OpenBlas => {}

            BlasBackend::Static => {}
        }
    }

    pub fn set_num_threads(&self, threads: u32) {
        match self.backend() {
            #[cfg(feature = "dynamic")]
            BlasBackend::IntelMkl => {
                type MklSetNumThreadsFn = extern "C" fn(i32);
                let mkl_set_num_threads: Symbol<MklSetNumThreadsFn> =
                    unsafe { self.lib().unwrap().get(b"MKL_Set_Num_Threads") }
                        .expect("Cannot find MKL_Set_Num_Threads");
                mkl_set_num_threads(threads as i32)
            }
            #[cfg(not(feature = "dynamic"))]
            BlasBackend::IntelMkl => {
                _ = threads;
            }
            #[cfg(feature = "dynamic")]
            BlasBackend::OpenBlas => {
                type OpenBlasSetNumThreadsFn = extern "C" fn(i32);
                let mkl_set_num_threads: Symbol<OpenBlasSetNumThreadsFn> =
                    unsafe { self.lib().unwrap().get(b"openblas_set_num_threads") }
                        .expect("Cannot find openblas_set_num_threads");
                mkl_set_num_threads(threads as i32)
            }
            #[cfg(not(feature = "dynamic"))]
            BlasBackend::OpenBlas => {}
            BlasBackend::Static => {}
        }
    }

    pub fn with_backend(backend: BlasBackend) -> Result<Self, LaError> {
        match backend {
            #[cfg(feature = "dynamic")]
            BlasBackend::IntelMkl | BlasBackend::OpenBlas => {
                use crate::util::find_lib_path;

                let lib_path = find_lib_path(backend.lib_name().unwrap())?;
                let lib = unsafe { Library::new(lib_path) }?;
                Ok(Self(Arc::new(BlasLibInner {
                    lib: Some(lib),
                    backend,
                })))
            }
            #[cfg(not(feature = "dynamic"))]
            BlasBackend::IntelMkl | BlasBackend::OpenBlas => Err(LaError::NoLaLibrary),
            BlasBackend::Static => Ok(Self(Arc::new(BlasLibInner {
                #[cfg(feature = "dynamic")]
                lib: None,
                backend,
            }))),
        }
    }

    pub fn with_backend_with_additional_search_paths<P>(
        backend: BlasBackend,
        additional_search_paths: impl IntoIterator<Item = P>,
    ) -> Result<Self, LaError>
    where
        P: AsRef<Path>,
    {
        #[cfg(not(feature = "dynamic"))]
        {
            _ = additional_search_paths;
        }

        match backend {
            #[cfg(feature = "dynamic")]
            BlasBackend::IntelMkl | BlasBackend::OpenBlas => {
                use crate::util::find_lib_path_with_additional_search_paths;

                let lib_path = find_lib_path_with_additional_search_paths(
                    backend.lib_name().unwrap(),
                    additional_search_paths,
                )?;
                let lib = unsafe { Library::new(lib_path) }?;
                Ok(Self(Arc::new(BlasLibInner {
                    lib: Some(lib),
                    backend,
                })))
            }
            #[cfg(not(feature = "dynamic"))]
            BlasBackend::IntelMkl | BlasBackend::OpenBlas => Err(LaError::NoLaLibrary),
            BlasBackend::Static => Ok(Self(Arc::new(BlasLibInner {
                #[cfg(feature = "dynamic")]
                lib: None,
                backend,
            }))),
        }
    }

    pub fn backend(&self) -> BlasBackend {
        self.0.backend
    }

    pub fn functions_static(&self) -> BlasFunctionsStatic {
        let functions = unsafe { std::mem::transmute(self.functions()) };
        BlasFunctionsStatic {
            _lib: self.clone(),
            functions,
        }
    }

    pub fn functions(&self) -> BlasFunctions<'_> {
        BlasFunctions::from_lib(self)
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{Complex, DMatrix, VectorView};

    use crate::blas::{blas_lib::Transpose, Threading};

    use super::BlasLib;

    #[test]
    fn test_blas_sequential() {
        let blas_lib = BlasLib::new().expect("Failed to include Blas backend.");
        blas_lib.set_threading(Threading::Sequential);
        let blas = blas_lib.functions();

        const DIM: usize = 1024;
        let a = DMatrix::from_vec(
            DIM,
            DIM,
            Vec::from_iter((0..DIM * DIM).map(|x| x as f32 + 1.0)),
        );
        let b = DMatrix::from_vec(
            DIM,
            DIM,
            Vec::from_iter((0..DIM * DIM).map(|x| x as f32 + 1.0)),
        );
        let mut c = DMatrix::zeros(DIM, DIM);
        blas.sgemm(1.0, 1.0, &a, Transpose::None, &b, Transpose::None, &mut c);

        let a_view = VectorView::from_slice_with_strides_generic(
            b.as_slice(),
            nalgebra::Dyn(a.len()),
            nalgebra::U1,
            nalgebra::U1,
            nalgebra::Dyn(a.len()),
        );
        let b_view = VectorView::from_slice_with_strides_generic(
            b.as_slice(),
            nalgebra::Dyn(b.len()),
            nalgebra::U1,
            nalgebra::U1,
            nalgebra::Dyn(a.len()),
        );
        blas.sdot::<_, _, _, _, _, _>(a_view, b_view);

        let complex_a = a.map(|x| Complex::new(x, x));
        let complex_b = b.map(|x| Complex::new(x, x));

        let mut c = DMatrix::zeros(DIM, DIM);
        blas.cgemm(
            Complex::ONE,
            Complex::ONE,
            &complex_a,
            Transpose::None,
            &complex_b,
            Transpose::None,
            &mut c,
        );

        let complex_a_view = VectorView::from_slice_with_strides_generic(
            complex_a.as_slice(),
            nalgebra::Dyn(a.len()),
            nalgebra::U1,
            nalgebra::U1,
            nalgebra::Dyn(a.len()),
        );
        let complex_b_view = VectorView::from_slice_with_strides_generic(
            complex_b.as_slice(),
            nalgebra::Dyn(b.len()),
            nalgebra::U1,
            nalgebra::U1,
            nalgebra::Dyn(a.len()),
        );
        blas.cdotc::<_, _, _, _, _, _>(complex_a_view, complex_b_view);
    }

    #[test]
    fn test_blas_multithreaded() {
        let blas_lib = BlasLib::new().expect("Failed to include Blas backend.");
        blas_lib.set_threading(Threading::Multithreaded);
        blas_lib.set_num_threads(8);
        let blas = blas_lib.functions();

        const DIM: usize = 1024;
        let a = DMatrix::from_vec(
            DIM,
            DIM,
            Vec::from_iter((0..DIM * DIM).map(|x| x as f32 + 1.0)),
        );
        let b = DMatrix::from_vec(
            DIM,
            DIM,
            Vec::from_iter((0..DIM * DIM).map(|x| x as f32 + 1.0)),
        );
        let mut c = DMatrix::zeros(DIM, DIM);
        blas.sgemm(1.0, 0.0, &a, Transpose::None, &b, Transpose::None, &mut c);

        let a_view = VectorView::from_slice_with_strides_generic(
            b.as_slice(),
            nalgebra::Dyn(a.len()),
            nalgebra::U1,
            nalgebra::U1,
            nalgebra::Dyn(a.len()),
        );
        let b_view = VectorView::from_slice_with_strides_generic(
            b.as_slice(),
            nalgebra::Dyn(b.len()),
            nalgebra::U1,
            nalgebra::U1,
            nalgebra::Dyn(a.len()),
        );
        blas.sdot::<_, _, _, _, _, _>(a_view, b_view);
    }
}
