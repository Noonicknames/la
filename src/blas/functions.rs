use std::{
    ffi::{c_double, c_float},
    ops::Deref,
};

use libloading::Symbol;
use nalgebra::{
    constraint::{SameDimension, ShapeConstraint},
    Complex, Dim, Matrix, Storage, StorageMut, VectorView,
};

use crate::{LaComplexDouble, LaComplexFloat, LaInt};

use super::blas_lib::{BlasLib, Transpose};

#[derive(Clone)]
pub struct BlasFunctions<'a> {
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
}

impl<'a> BlasFunctions<'a> {
    pub(crate) fn from_lib(lib: &BlasLib) -> Self {
        macro_rules! functions {
            ( $( [$name: ident, $symbol: expr, $fn_signature: ty] ),* $(,)?) => {
                $(
                    let $name = unsafe {
                        std::mem::transmute(
                            lib.lib()
                                .get::<Symbol<$fn_signature>>($symbol)
                                .expect(&format!("Failed to find `{}`.", str::from_utf8($symbol).unwrap())),
                        )
                    };
                )*

                Self {
                    $($name),*
                }
            };
        }

        functions! {
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
}

impl<'a> BlasFunctions<'a> {
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

        (self.sdsdot)(n, alpha, x.as_ptr(), incx, y.as_ptr(), incy)
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

        (self.dsdot)(n, x.as_ptr(), incx, y.as_ptr(), incy)
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

        (self.sdot)(n, x.as_ptr(), incx, y.as_ptr(), incy)
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

        (self.cdotu_sub)(n, x.as_ptr(), incx, y.as_ptr(), incy, &mut out);

        out
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

        (self.cdotc_sub)(n, x.as_ptr(), incx, y.as_ptr(), incy, &mut out);

        out
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

        (self.zdotu_sub)(n, x.as_ptr(), incx, y.as_ptr(), incy, &mut out);

        out
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

        (self.zdotc_sub)(n, x.as_ptr(), incx, y.as_ptr(), incy, &mut out);

        out
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

        (self.ddot)(n, x.as_ptr(), incx, y.as_ptr(), incy)
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

        (self.sgemm)(
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

        (self.dgemm)(
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

        (self.cgemm)(
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

        (self.zgemm)(
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

type CBlasSDSDotFn = extern "C" fn(
    n: LaInt,
    alpha: c_float,
    x: *const c_float,
    incx: LaInt,
    y: *const c_float,
    incy: LaInt,
) -> c_float;

type CBlasDSDotFn = extern "C" fn(
    n: LaInt,
    x: *const c_float,
    incx: LaInt,
    y: *const c_float,
    incy: LaInt,
) -> c_double;

type CBlasSDotFn = extern "C" fn(
    n: LaInt,
    x: *const c_float,
    incx: LaInt,
    y: *const c_float,
    incy: LaInt,
) -> c_float;

type CBlasDDotFn = extern "C" fn(
    n: LaInt,
    x: *const c_double,
    incx: LaInt,
    y: *const c_double,
    incy: LaInt,
) -> c_double;

type CBlasCDotUSubFn = extern "C" fn(
    n: LaInt,
    x: *const LaComplexFloat,
    incx: LaInt,
    y: *const LaComplexFloat,
    incy: LaInt,
        ret: *mut LaComplexFloat,
);

type CBlasCDotCSubFn = extern "C" fn(
    n: LaInt,
    x: *const LaComplexFloat,
    incx: LaInt,
    y: *const LaComplexFloat,
    incy: LaInt,
    ret: *mut LaComplexFloat,
);

type CBlasZDotCSubFn = extern "C" fn(
    n: LaInt,
    x: *const LaComplexDouble,
    incx: LaInt,
    y: *const LaComplexDouble,
    incy: LaInt,
    ret: *mut LaComplexDouble,
);
type CBlasZDotUSubFn = extern "C" fn(
    n: LaInt,
    x: *const LaComplexDouble,
    incx: LaInt,
    y: *const LaComplexDouble,
    incy: LaInt,
    ret: *mut LaComplexDouble,
);

type CBlasDGemmFn = extern "C" fn(
    cblas_sys::CBLAS_LAYOUT,
    cblas_sys::CBLAS_TRANSPOSE,
    cblas_sys::CBLAS_TRANSPOSE,
    LaInt,
    LaInt,
    LaInt,
    c_double,
    *const c_double,
    LaInt,
    *const c_double,
    LaInt,
    c_double,
    *const c_double,
    LaInt,
);

type CBlasSGemmFn = extern "C" fn(
    cblas_sys::CBLAS_LAYOUT,
    cblas_sys::CBLAS_TRANSPOSE,
    cblas_sys::CBLAS_TRANSPOSE,
    LaInt,
    LaInt,
    LaInt,
    c_float,
    *const c_float,
    LaInt,
    *const c_float,
    LaInt,
    c_float,
    *mut c_float,
    LaInt,
);

type CBlasCGemmFn = extern "C" fn(
    order: cblas_sys::CBLAS_LAYOUT,
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

type CBlasZGemmFn = extern "C" fn(
    order: cblas_sys::CBLAS_LAYOUT,
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
