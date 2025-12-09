#[cfg(feature = "static")]
use std::marker::PhantomData;
use std::{
    ffi::{c_double, c_float},
    ops::Deref,
};

#[cfg(feature = "dynamic")]
use libloading::Symbol;
use nalgebra::{
    constraint::{SameDimension, ShapeConstraint},
    Complex, Dim, Matrix, Storage, StorageMut, VectorView,
};

use crate::{BlasBackend, LaComplexDouble, LaComplexFloat, LaInt};

use super::blas_lib::{BlasLib, Transpose};

#[derive(Clone)]
pub enum BlasFunctions<'a> {
    #[cfg(feature = "dynamic")]
    Dynamic {
        sdsdot: Symbol<'a, CBlasSDSDotFn>,
        dsdot: Symbol<'a, CBlasDSDotFn>,
        sdot: Symbol<'a, CBlasSDotFn>,
        ddot: Symbol<'a, CBlasDDotFn>,
        cdotu_sub: Symbol<'a, CBlasCDotUSubFn>,
        cdotc_sub: Symbol<'a, CBlasCDotCSubFn>,
        zdotu_sub: Symbol<'a, CBlasZDotUSubFn>,
        zdotc_sub: Symbol<'a, CBlasZDotCSubFn>,

        dgemm: Symbol<'a, CBlasDGemmFn>,
        sgemm: Symbol<'a, CBlasSGemmFn>,
        cgemm: Symbol<'a, CBlasCGemmFn>,
        zgemm: Symbol<'a, CBlasZGemmFn>,
    },
    #[cfg(feature = "static")]
    Static(PhantomData<&'a ()>),
}

impl<'a> BlasFunctions<'a> {
    pub(crate) fn from_lib(lib: &BlasLib) -> Self {
        #[cfg(feature = "dynamic")]
        macro_rules! functions_dynamic {
            ( $( [$name: ident, $symbol: expr, $fn_signature: ty] ),* $(,)?) => {
                let lib = lib.lib().unwrap();
                $(
                    let $name = unsafe {
                        std::mem::transmute(
                            lib
                                .get::<Symbol<$fn_signature>>($symbol)
                                .expect(&format!("Failed to find `{}`.", std::str::from_utf8($symbol).unwrap())),
                        )
                    };
                )*

                Self::Dynamic {
                    $($name),*
                }
            };
        }

        match lib.backend() {
            #[cfg(feature = "dynamic")]
            BlasBackend::IntelMkl | BlasBackend::OpenBlas => {
                functions_dynamic! {
                    [sdsdot, b"cblas_sdsdot", CBlasSDSDotFn],
                    [dsdot, b"cblas_dsdot", CBlasDSDotFn],
                    [sdot, b"cblas_sdot", CBlasSDotFn],
                    [ddot, b"cblas_ddot", CBlasDDotFn],
                    [cdotu_sub, b"cblas_cdotu_sub", CBlasCDotUSubFn],
                    [cdotc_sub, b"cblas_cdotc_sub", CBlasCDotCSubFn],
                    [zdotu_sub, b"cblas_zdotu_sub", CBlasZDotUSubFn],
                    [zdotc_sub, b"cblas_zdotc_sub", CBlasZDotCSubFn],

                    [dgemm, b"cblas_dgemm", CBlasDGemmFn],
                    [sgemm, b"cblas_sgemm", CBlasSGemmFn],
                    [cgemm, b"cblas_cgemm", CBlasCGemmFn],
                    [zgemm, b"cblas_zgemm", CBlasZGemmFn],
                }
            }
            #[cfg(not(feature = "dynamic"))]
            BlasBackend::IntelMkl | BlasBackend::OpenBlas => {
                panic!("Feature \"dynamic\" not enabled, cannot create BlasFunctions.");
            }
            #[cfg(feature = "static")]
            BlasBackend::Static => Self::Static(PhantomData),
            #[cfg(not(feature = "static"))]
            BlasBackend::Static => panic!("Cannot create blas functions, \"static\" feature not enabled."),
        }
    }
}

impl<'a> BlasFunctions<'a> {
    pub fn raw_sdsdot(&self) -> CBlasSDSDotFn {
        match self {
            #[cfg(feature = "dynamic")]
            Self::Dynamic { sdsdot, .. } => **sdsdot,
            #[cfg(feature = "static")]
            Self::Static(..) => cblas_sdsdot,
        }
    }
    /// Performs `dot(x,y) + alpha`.
    ///
    /// # Why does this exist?
    /// The dot product is accumulated with double precision and alpha is added before converting back to single precision.
    /// This function is somewhat outdated and was originally included to decrease numerical error compared to separately adding a scalar.
    pub fn sdsdot<DX, DY, RStrideX, RStrideY, CStrideX, CStrideY>(
        &self,
        alpha: f32,
        x: VectorView<f32, DX, RStrideX, CStrideX>,
        y: VectorView<f32, DY, RStrideY, CStrideY>,
    ) -> f32
    where
        DX: Dim,
        DY: Dim,
        RStrideX: Dim,
        RStrideY: Dim,
        CStrideX: Dim,
        CStrideY: Dim,
        ShapeConstraint: SameDimension<DX, DY>,
    {
        // assert!(x.len() == y.len());
        let n = x.len() as LaInt;
        let incx = x.strides().0 as LaInt;
        let incy = y.strides().0 as LaInt;
        unsafe { (self.raw_sdsdot())(n, alpha, x.as_ptr(), incx, y.as_ptr(), incy) }
    }

    pub fn raw_dsdot(&self) -> CBlasDSDotFn {
        match self {
            #[cfg(feature = "dynamic")]
            Self::Dynamic { dsdot, .. } => **dsdot,
            #[cfg(feature = "static")]
            Self::Static(..) => cblas_dsdot,
        }
    }
    /// Performs `dot(x,y)`.
    ///
    /// The dot product is accumulated as a `f64` and returned as a `f64`.
    pub fn dsdot<DX, DY, RStrideX, RStrideY, CStrideX, CStrideY>(
        &self,
        x: VectorView<f32, DX, RStrideX, CStrideX>,
        y: VectorView<f32, DY, RStrideY, CStrideY>,
    ) -> f64
    where
        DX: Dim,
        DY: Dim,
        RStrideX: Dim,
        RStrideY: Dim,
        CStrideX: Dim,
        CStrideY: Dim,
        ShapeConstraint: SameDimension<DX, DY>,
    {
        // assert!(x.len() == y.len());
        let n = x.len() as LaInt;
        let incx = x.strides().0 as LaInt;
        let incy = y.strides().0 as LaInt;

        unsafe { (self.raw_dsdot())(n, x.as_ptr(), incx, y.as_ptr(), incy) }
    }

    pub fn raw_sdot(&self) -> CBlasSDotFn {
        match self {
            #[cfg(feature = "dynamic")]
            Self::Dynamic { sdot, .. } => **sdot,
            #[cfg(feature = "static")]
            Self::Static(..) => cblas_sdot,
        }
    }
    /// Performs `dot(x,y)`.
    ///
    /// For a [f64] variant, see [`BlasFunctions::ddot`].
    pub fn sdot<DX, DY, RStrideX, RStrideY, CStrideX, CStrideY>(
        &self,
        x: VectorView<f32, DX, RStrideX, CStrideX>,
        y: VectorView<f32, DY, RStrideY, CStrideY>,
    ) -> f32
    where
        DX: Dim,
        DY: Dim,
        RStrideX: Dim,
        RStrideY: Dim,
        CStrideX: Dim,
        CStrideY: Dim,
        ShapeConstraint: SameDimension<DX, DY>,
    {
        // assert!(x.len() == y.len());
        let n = x.len() as LaInt;
        let incx = x.strides().0 as LaInt;
        let incy = y.strides().0 as LaInt;
        unsafe { (self.raw_sdot())(n, x.as_ptr(), incx, y.as_ptr(), incy) }
    }

    pub fn raw_cdotu_sub(&self) -> CBlasCDotUSubFn {
        match self {
            #[cfg(feature = "dynamic")]
            Self::Dynamic { cdotu_sub, .. } => **cdotu_sub,
            #[cfg(feature = "static")]
            Self::Static(..) => cblas_cdotu_sub,
        }
    }
    /// Performs `dot(x,y)`.
    ///
    /// For a [f64] variant, see [`BlasFunctions::ddot`].
    pub fn cdotu<DX, DY, RStrideX, RStrideY, CStrideX, CStrideY>(
        &self,
        x: VectorView<Complex<f32>, DX, RStrideX, CStrideX>,
        y: VectorView<Complex<f32>, DY, RStrideY, CStrideY>,
    ) -> Complex<f32>
    where
        DX: Dim,
        DY: Dim,
        RStrideX: Dim,
        RStrideY: Dim,
        CStrideX: Dim,
        CStrideY: Dim,
        ShapeConstraint: SameDimension<DX, DY>,
    {
        // assert!(x.len() == y.len());
        let n = x.len() as LaInt;
        let incx = x.strides().0 as LaInt;
        let incy = y.strides().0 as LaInt;

        let mut out = Complex::ZERO;

        unsafe { (self.raw_cdotu_sub())(n, x.as_ptr(), incx, y.as_ptr(), incy, &mut out) };

        out
    }

    pub fn raw_cdotc_sub(&self) -> CBlasCDotCSubFn {
        match self {
            #[cfg(feature = "dynamic")]
            Self::Dynamic { cdotc_sub, .. } => **cdotc_sub,
            #[cfg(feature = "static")]
            Self::Static(..) => cblas_cdotc_sub,
        }
    }
    /// Performs `dot(conj(x),y)`.
    ///
    /// For a [f64] variant, see [`BlasFunctions::ddot`].
    pub fn cdotc<DX, DY, RStrideX, RStrideY, CStrideX, CStrideY>(
        &self,
        x: VectorView<Complex<f32>, DX, RStrideX, CStrideX>,
        y: VectorView<Complex<f32>, DY, RStrideY, CStrideY>,
    ) -> Complex<f32>
    where
        DX: Dim,
        DY: Dim,
        RStrideX: Dim,
        RStrideY: Dim,
        CStrideX: Dim,
        CStrideY: Dim,
        ShapeConstraint: SameDimension<DX, DY>,
    {
        // assert!(x.len() == y.len());
        let n = x.len() as LaInt;
        let incx = x.strides().0 as LaInt;
        let incy = y.strides().0 as LaInt;

        let mut out = Complex::ZERO;

        unsafe { (self.raw_cdotc_sub())(n, x.as_ptr(), incx, y.as_ptr(), incy, &mut out) };

        out
    }

    pub fn raw_zdotu_sub(&self) -> CBlasZDotUSubFn {
        match self {
            #[cfg(feature = "dynamic")]
            Self::Dynamic { zdotu_sub, .. } => **zdotu_sub,
            #[cfg(feature = "static")]
            Self::Static(..) => cblas_zdotu_sub,
        }
    }
    /// Performs `dot(conj(x),y)`.
    ///
    /// For a [f64] variant, see [`BlasFunctions::ddot`].
    pub fn zdotu<DX, DY, RStrideX, RStrideY, CStrideX, CStrideY>(
        &self,
        x: VectorView<Complex<f64>, DX, RStrideX, CStrideX>,
        y: VectorView<Complex<f64>, DY, RStrideY, CStrideY>,
    ) -> Complex<f64>
    where
        DX: Dim,
        DY: Dim,
        RStrideX: Dim,
        RStrideY: Dim,
        CStrideX: Dim,
        CStrideY: Dim,
        ShapeConstraint: SameDimension<DX, DY>,
    {
        // assert!(x.len() == y.len());
        let n = x.len() as LaInt;
        let incx = x.strides().0 as LaInt;
        let incy = y.strides().0 as LaInt;

        let mut out = Complex::ZERO;

        unsafe { (self.raw_zdotu_sub())(n, x.as_ptr(), incx, y.as_ptr(), incy, &mut out) };

        out
    }

    pub fn raw_zdotc_sub(&self) -> CBlasZDotCSubFn {
        match self {
            #[cfg(feature = "dynamic")]
            Self::Dynamic { zdotc_sub, .. } => **zdotc_sub,
            #[cfg(feature = "static")]
            Self::Static(..) => cblas_zdotc_sub,
        }
    }
    /// Performs `dot(conj(x),y)`.
    ///
    /// For a [f64] variant, see [`BlasFunctions::ddot`].
    pub fn zdotc<DX, DY, RStrideX, RStrideY, CStrideX, CStrideY>(
        &self,
        x: VectorView<Complex<f64>, DX, RStrideX, CStrideX>,
        y: VectorView<Complex<f64>, DY, RStrideY, CStrideY>,
    ) -> Complex<f64>
    where
        DX: Dim,
        DY: Dim,
        RStrideX: Dim,
        RStrideY: Dim,
        CStrideX: Dim,
        CStrideY: Dim,
        ShapeConstraint: SameDimension<DX, DY>,
    {
        // assert!(x.len() == y.len());
        let n = x.len() as LaInt;
        let incx = x.strides().0 as LaInt;
        let incy = y.strides().0 as LaInt;

        let mut out = Complex::ZERO;

        unsafe { (self.raw_zdotc_sub())(n, x.as_ptr(), incx, y.as_ptr(), incy, &mut out) };

        out
    }

    pub fn raw_ddot(&self) -> CBlasDDotFn {
        match self {
            #[cfg(feature = "dynamic")]
            Self::Dynamic { ddot, .. } => **ddot,
            #[cfg(feature = "static")]
            Self::Static(..) => cblas_ddot,
        }
    }
    /// Performs `dot(x,y)`.
    ///
    /// For a [f32] variant, see [`BlasFunctions::sdot`].
    pub fn ddot<DX, DY>(&self, x: VectorView<f64, DX>, y: VectorView<f64, DY>) -> f64
    where
        DX: Dim,
        DY: Dim,
        ShapeConstraint: SameDimension<DX, DY>,
    {
        // assert!(x.len() == y.len());
        let n = x.len() as LaInt;
        let incx = x.strides().0 as LaInt;
        let incy = y.strides().0 as LaInt;

        unsafe { (self.raw_ddot())(n, x.as_ptr(), incx, y.as_ptr(), incy) }
    }

    pub fn raw_sgemm(&self) -> CBlasSGemmFn {
        match self {
            #[cfg(feature = "dynamic")]
            Self::Dynamic { sgemm, .. } => **sgemm,
            #[cfg(feature = "static")]
            Self::Static(..) => cblas_sgemm,
        }
    }
    /// Performs c := alpha * trans_a(a) * trans_b(b) + beta*c
    ///
    /// `trans` refers to taking the matrix transpose, adjoint, or no operation, see [Transpose].
    pub fn sgemm<RA, CA, SA, RB, CB, SB, RC, CC, SC>(
        &self,
        alpha: f32,
        beta: f32,
        a: &Matrix<f32, RA, CA, SA>,
        trans_a: Transpose,
        b: &Matrix<f32, RB, CB, SB>,
        trans_b: Transpose,
        c: &mut Matrix<f32, RC, CC, SC>,
    ) where
        RA: Dim,
        CA: Dim,
        SA: Storage<f32, RA, CA>,
        RB: Dim,
        CB: Dim,
        SB: Storage<f32, RB, CB>,
        RC: Dim,
        CC: Dim,
        SC: StorageMut<f32, RC, CC>,
    {
        // assert!(a.major == b.major && c.major == a.major);
        // assert!(a.dims.1 == b.dims.0);

        let m = a.data.shape().0.value() as LaInt;
        let n = b.data.shape().1.value() as LaInt;
        let k = a.data.shape().1.value() as LaInt;
        let lda = a.data.shape().0.value() as LaInt;
        let ldb = b.data.shape().0.value() as LaInt;
        let ldc = c.data.shape().0.value() as LaInt;

        unsafe {
            (self.raw_sgemm())(
                cblas_sys::CblasRowMajor,
                trans_a.to_sys(),
                trans_b.to_sys(),
                m,
                n,
                k,
                alpha,
                a.data.ptr(),
                lda,
                b.data.ptr(),
                ldb,
                beta,
                c.data.ptr_mut(),
                ldc,
            )
        }
    }
    pub fn raw_dgemm(&self) -> CBlasDGemmFn {
        match self {
            #[cfg(feature = "dynamic")]
            Self::Dynamic { dgemm, .. } => **dgemm,
            #[cfg(feature = "static")]
            Self::Static(..) => cblas_dgemm,
        }
    }
    /// Performs c := alpha * op_a(a) * op_b(b) + beta*c
    ///
    /// op(x) is one of op(x) = x, op(x) = x', or op(x) = conjg(x').
    pub fn dgemm<RA, CA, SA, RB, CB, SB, RC, CC, SC>(
        &self,
        alpha: f64,
        beta: f64,
        a: &Matrix<f64, RA, CA, SA>,
        trans_a: Transpose,
        b: &Matrix<f64, RB, CB, SB>,
        trans_b: Transpose,
        c: &mut Matrix<f64, RC, CC, SC>,
    ) where
        RA: Dim,
        CA: Dim,
        SA: Storage<f64, RA, CA>,
        RB: Dim,
        CB: Dim,
        SB: Storage<f64, RB, CB>,
        RC: Dim,
        CC: Dim,
        SC: StorageMut<f64, RC, CC>,
    {
        // assert!(a.major == b.major && c.major == a.major);
        // assert!(a.dims.1 == b.dims.0);

        let m = a.data.shape().0.value() as LaInt;
        let n = b.data.shape().1.value() as LaInt;
        let k = a.data.shape().1.value() as LaInt;
        let lda = a.data.shape().0.value() as LaInt;
        let ldb = b.data.shape().0.value() as LaInt;
        let ldc = c.data.shape().0.value() as LaInt;

        unsafe {
            (self.raw_dgemm())(
                cblas_sys::CblasRowMajor,
                trans_a.to_sys(),
                trans_b.to_sys(),
                m,
                n,
                k,
                alpha,
                a.data.ptr(),
                lda,
                b.data.ptr(),
                ldb,
                beta,
                c.data.ptr_mut(),
                ldc,
            )
        }
    }
    pub fn raw_cgemm(&self) -> CBlasCGemmFn {
        match self {
            #[cfg(feature = "dynamic")]
            Self::Dynamic { cgemm, .. } => **cgemm,
            #[cfg(feature = "static")]
            Self::Static(..) => cblas_cgemm,
        }
    }
    /// Performs c := alpha * trans_a(a) * trans_b(b) + beta*c
    ///
    /// `trans` refers to taking the matrix transpose, adjoint, or no operation, see [Transpose].
    pub fn cgemm<RA, CA, SA, RB, CB, SB, RC, CC, SC>(
        &self,
        alpha: Complex<f32>,
        beta: Complex<f32>,
        a: &Matrix<Complex<f32>, RA, CA, SA>,
        trans_a: Transpose,
        b: &Matrix<Complex<f32>, RB, CB, SB>,
        trans_b: Transpose,
        c: &mut Matrix<Complex<f32>, RC, CC, SC>,
    ) where
        RA: Dim,
        CA: Dim,
        SA: Storage<Complex<f32>, RA, CA>,
        RB: Dim,
        CB: Dim,
        SB: Storage<Complex<f32>, RB, CB>,
        RC: Dim,
        CC: Dim,
        SC: StorageMut<Complex<f32>, RC, CC>,
    {
        // assert!(a.major == b.major && c.major == a.major);
        // assert!(a.dims.1 == b.dims.0);

        let m = a.data.shape().0.value() as LaInt;
        let n = b.data.shape().1.value() as LaInt;
        let k = a.data.shape().1.value() as LaInt;
        let lda = a.data.shape().0.value() as LaInt;
        let ldb = b.data.shape().0.value() as LaInt;
        let ldc = c.data.shape().0.value() as LaInt;

        unsafe {
            (self.raw_cgemm())(
                cblas_sys::CblasRowMajor,
                trans_a.to_sys(),
                trans_b.to_sys(),
                m,
                n,
                k,
                &alpha,
                a.data.ptr(),
                lda,
                b.data.ptr(),
                ldb,
                &beta,
                c.data.ptr_mut(),
                ldc,
            )
        }
    }

    pub fn raw_zgemm(&self) -> CBlasZGemmFn {
        match self {
            #[cfg(feature = "dynamic")]
            Self::Dynamic { zgemm, .. } => **zgemm,
            #[cfg(feature = "static")]
            Self::Static(..) => cblas_zgemm,
        }
    }
    /// Performs c := alpha * trans_a(a) * trans_b(b) + beta*c
    ///
    /// `trans` refers to taking the matrix transpose, adjoint, or no operation, see [Transpose].
    pub fn zgemm<RA, CA, SA, RB, CB, SB, RC, CC, SC>(
        &self,
        alpha: Complex<f64>,
        beta: Complex<f64>,
        a: &Matrix<Complex<f64>, RA, CA, SA>,
        trans_a: Transpose,
        b: &Matrix<Complex<f64>, RB, CB, SB>,
        trans_b: Transpose,
        c: &mut Matrix<Complex<f64>, RC, CC, SC>,
    ) where
        RA: Dim,
        CA: Dim,
        SA: Storage<Complex<f64>, RA, CA>,
        RB: Dim,
        CB: Dim,
        SB: Storage<Complex<f64>, RB, CB>,
        RC: Dim,
        CC: Dim,
        SC: StorageMut<Complex<f64>, RC, CC>,
    {
        // assert!(a.major == b.major && c.major == a.major);
        // assert!(a.dims.1 == b.dims.0);

        let m = a.data.shape().0.value() as LaInt;
        let n = b.data.shape().1.value() as LaInt;
        let k = a.data.shape().1.value() as LaInt;
        let lda = a.data.shape().0.value() as LaInt;
        let ldb = b.data.shape().0.value() as LaInt;
        let ldc = c.data.shape().0.value() as LaInt;

        unsafe {
            (self.raw_zgemm())(
                cblas_sys::CblasRowMajor,
                trans_a.to_sys(),
                trans_b.to_sys(),
                m,
                n,
                k,
                &alpha,
                a.data.ptr(),
                lda,
                b.data.ptr(),
                ldb,
                &beta,
                c.data.ptr_mut(),
                ldc,
            )
        }
    }
}

pub struct BlasFunctionsStatic {
    // This is only to make sure BlasLib is not dropped.
    pub(super) _lib: BlasLib,
    pub(super) functions: BlasFunctions<'static>,
}

impl Deref for BlasFunctionsStatic {
    type Target = BlasFunctions<'static>;
    fn deref(&self) -> &Self::Target {
        &self.functions
    }
}

#[cfg(feature = "static")]
extern "C" {
    fn cblas_sdsdot(
        n: LaInt,
        alpha: c_float,
        x: *const c_float,
        incx: LaInt,
        y: *const c_float,
        incy: LaInt,
    ) -> c_float;
    fn cblas_dsdot(
        n: LaInt,
        x: *const c_float,
        incx: LaInt,
        y: *const c_float,
        incy: LaInt,
    ) -> c_double;
    fn cblas_sdot(
        n: LaInt,
        x: *const c_float,
        incx: LaInt,
        y: *const c_float,
        incy: LaInt,
    ) -> c_float;
    fn cblas_ddot(
        n: LaInt,
        x: *const c_double,
        incx: LaInt,
        y: *const c_double,
        incy: LaInt,
    ) -> c_double;
    fn cblas_cdotu_sub(
        n: LaInt,
        x: *const LaComplexFloat,
        incx: LaInt,
        y: *const LaComplexFloat,
        incy: LaInt,
        ret: *mut LaComplexFloat,
    );
    fn cblas_cdotc_sub(
        n: LaInt,
        x: *const LaComplexFloat,
        incx: LaInt,
        y: *const LaComplexFloat,
        incy: LaInt,
        ret: *mut LaComplexFloat,
    );
    fn cblas_zdotu_sub(
        n: LaInt,
        x: *const LaComplexDouble,
        incx: LaInt,
        y: *const LaComplexDouble,
        incy: LaInt,
        ret: *mut LaComplexDouble,
    );
    fn cblas_zdotc_sub(
        n: LaInt,
        x: *const LaComplexDouble,
        incx: LaInt,
        y: *const LaComplexDouble,
        incy: LaInt,
        ret: *mut LaComplexDouble,
    );

    fn cblas_sgemm(
        layout: cblas_sys::CBLAS_LAYOUT,
        transa: cblas_sys::CBLAS_TRANSPOSE,
        transb: cblas_sys::CBLAS_TRANSPOSE,
        m: LaInt,
        n: LaInt,
        k: LaInt,
        alpha: c_float,
        a: *const c_float,
        lda: LaInt,
        b: *const c_float,
        ldb: LaInt,
        beta: c_float,
        c: *mut c_float,
        ldc: LaInt,
    );

    fn cblas_dgemm(
        layout: cblas_sys::CBLAS_LAYOUT,
        transa: cblas_sys::CBLAS_TRANSPOSE,
        transb: cblas_sys::CBLAS_TRANSPOSE,
        m: LaInt,
        n: LaInt,
        k: LaInt,
        alpha: c_double,
        a: *const c_double,
        lda: LaInt,
        b: *const c_double,
        ldb: LaInt,
        beta: c_double,
        c: *const c_double,
        ldc: LaInt,
    );

    fn cblas_cgemm(
        layout: cblas_sys::CBLAS_LAYOUT,
        trans_a: cblas_sys::CBLAS_TRANSPOSE,
        trans_b: cblas_sys::CBLAS_TRANSPOSE,
        m: LaInt,
        n: LaInt,
        k: LaInt,
        alpha: *const LaComplexFloat,
        a: *const LaComplexFloat,
        lda: LaInt,
        b: *const LaComplexFloat,
        ldb: LaInt,
        beta: *const LaComplexFloat,
        c: *mut LaComplexFloat,
        ldc: LaInt,
    );

    fn cblas_zgemm(
        layout: cblas_sys::CBLAS_LAYOUT,
        trans_a: cblas_sys::CBLAS_TRANSPOSE,
        trans_b: cblas_sys::CBLAS_TRANSPOSE,
        m: LaInt,
        n: LaInt,
        k: LaInt,
        alpha: *const LaComplexDouble,
        a: *const LaComplexDouble,
        lda: LaInt,
        b: *const LaComplexDouble,
        ldb: LaInt,
        beta: *const LaComplexDouble,
        c: *mut LaComplexDouble,
        ldc: LaInt,
    );
}

type CBlasSDSDotFn = unsafe extern "C" fn(
    n: LaInt,
    alpha: c_float,
    x: *const c_float,
    incx: LaInt,
    y: *const c_float,
    incy: LaInt,
) -> c_float;

type CBlasDSDotFn = unsafe extern "C" fn(
    n: LaInt,
    x: *const c_float,
    incx: LaInt,
    y: *const c_float,
    incy: LaInt,
) -> c_double;

type CBlasSDotFn = unsafe extern "C" fn(
    n: LaInt,
    x: *const c_float,
    incx: LaInt,
    y: *const c_float,
    incy: LaInt,
) -> c_float;

type CBlasDDotFn = unsafe extern "C" fn(
    n: LaInt,
    x: *const c_double,
    incx: LaInt,
    y: *const c_double,
    incy: LaInt,
) -> c_double;

type CBlasCDotUSubFn = unsafe extern "C" fn(
    n: LaInt,
    x: *const LaComplexFloat,
    incx: LaInt,
    y: *const LaComplexFloat,
    incy: LaInt,
    ret: *mut LaComplexFloat,
);

type CBlasCDotCSubFn = unsafe extern "C" fn(
    n: LaInt,
    x: *const LaComplexFloat,
    incx: LaInt,
    y: *const LaComplexFloat,
    incy: LaInt,
    ret: *mut LaComplexFloat,
);

type CBlasZDotCSubFn = unsafe extern "C" fn(
    n: LaInt,
    x: *const LaComplexDouble,
    incx: LaInt,
    y: *const LaComplexDouble,
    incy: LaInt,
    ret: *mut LaComplexDouble,
);
type CBlasZDotUSubFn = unsafe extern "C" fn(
    n: LaInt,
    x: *const LaComplexDouble,
    incx: LaInt,
    y: *const LaComplexDouble,
    incy: LaInt,
    ret: *mut LaComplexDouble,
);

type CBlasSGemmFn = unsafe extern "C" fn(
    // Called Order in openblas
    layout: cblas_sys::CBLAS_LAYOUT,
    transa: cblas_sys::CBLAS_TRANSPOSE,
    transb: cblas_sys::CBLAS_TRANSPOSE,
    m: LaInt,
    n: LaInt,
    k: LaInt,
    alpha: c_float,
    a: *const c_float,
    lda: LaInt,
    b: *const c_float,
    ldb: LaInt,
    beta: c_float,
    c: *mut c_float,
    ldc: LaInt,
);
type CBlasDGemmFn = unsafe extern "C" fn(
    layout: cblas_sys::CBLAS_LAYOUT,
    transa: cblas_sys::CBLAS_TRANSPOSE,
    transb: cblas_sys::CBLAS_TRANSPOSE,
    m: LaInt,
    n: LaInt,
    k: LaInt,
    alpha: c_double,
    a: *const c_double,
    lda: LaInt,
    b: *const c_double,
    ldb: LaInt,
    beta: c_double,
    c: *const c_double,
    ldc: LaInt,
);
type CBlasCGemmFn = unsafe extern "C" fn(
    layout: cblas_sys::CBLAS_LAYOUT,
    trans_a: cblas_sys::CBLAS_TRANSPOSE,
    trans_b: cblas_sys::CBLAS_TRANSPOSE,
    m: LaInt,
    n: LaInt,
    k: LaInt,
    alpha: *const LaComplexFloat,
    a: *const LaComplexFloat,
    lda: LaInt,
    b: *const LaComplexFloat,
    ldb: LaInt,
    beta: *const LaComplexFloat,
    c: *mut LaComplexFloat,
    ldc: LaInt,
);
type CBlasZGemmFn = unsafe extern "C" fn(
    layout: cblas_sys::CBLAS_LAYOUT,
    trans_a: cblas_sys::CBLAS_TRANSPOSE,
    trans_b: cblas_sys::CBLAS_TRANSPOSE,
    m: LaInt,
    n: LaInt,
    k: LaInt,
    alpha: *const LaComplexDouble,
    a: *const LaComplexDouble,
    lda: LaInt,
    b: *const LaComplexDouble,
    ldb: LaInt,
    beta: *const LaComplexDouble,
    c: *mut LaComplexDouble,
    ldc: LaInt,
);
