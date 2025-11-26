use std::{
    ffi::c_float,
    ops::{Bound, Deref, Range, RangeBounds},
    os::raw::c_double,
};

use libloading::Symbol;
use nalgebra::{ArrayStorage, Dim, Matrix, StorageMut, Vector, U1};

use crate::LaInt;

use super::lapacke_lib::LapackeLib;

pub enum EigenRange<T> {
    All,
    Index(Range<usize>),
    Value(Range<T>),
}

impl<T> EigenRange<T> {
    pub fn to_sys_char(&self) -> u8 {
        match self {
            Self::All => b'A',
            Self::Index(_) => b'I',
            Self::Value(_) => b'V',
        }
    }
    pub fn get_sys_index(&self, max_index: usize) -> Option<(LaInt, LaInt)> {
        match self {
            Self::Index(i) => Some((
                match i.start_bound() {
                    Bound::Excluded(&start) => (start + 1) as LaInt,
                    Bound::Included(&start) => start as LaInt,
                    Bound::Unbounded => 0 as LaInt,
                },
                match i.end_bound() {
                    Bound::Excluded(&end) => (end - 1) as LaInt,
                    Bound::Included(&end) => end as LaInt,
                    Bound::Unbounded => max_index as LaInt,
                },
            )),
            _ => None,
        }
    }
}

impl EigenRange<f32> {
    pub fn get_sys_value(&self) -> Option<(c_float, c_float)> {
        match self {
            Self::Value(i) => Some((
                match i.start_bound() {
                    Bound::Excluded(&start) | Bound::Included(&start) => start,
                    Bound::Unbounded => f32::MIN,
                },
                match i.end_bound() {
                    Bound::Excluded(&end) | Bound::Included(&end) => end,
                    Bound::Unbounded => f32::MAX,
                },
            )),
            _ => None,
        }
    }
}

impl EigenRange<f64> {
    pub fn get_sys_value(&self) -> Option<(c_double, c_double)> {
        match self {
            Self::Value(i) => Some((
                match i.start_bound() {
                    Bound::Excluded(&start) | Bound::Included(&start) => start,
                    Bound::Unbounded => f64::MIN,
                },
                match i.end_bound() {
                    Bound::Excluded(&end) | Bound::Included(&end) => end,
                    Bound::Unbounded => f64::MAX,
                },
            )),
            _ => None,
        }
    }
}

#[derive(Clone)]
pub struct LapackeFunctions<'a> {
    ssygvd: Symbol<'a, LapackeSsygvdFn>,
    dsygvd: Symbol<'a, LapackeDsygvdFn>,

    ssbevd: Symbol<'a, LapackeSsbevdFn>,
    dsbevd: Symbol<'a, LapackeDsbevdFn>,

    sstemr: Symbol<'a, LapackeSstemrFn>,
    dstemr: Symbol<'a, LapackeDstemrFn>,
}

impl<'a> LapackeFunctions<'a> {
    pub(crate) fn from_lib(lib: &LapackeLib) -> Self {
        let lib = lib.lib();

        macro_rules! functions {
            ( $( [$name: ident, $symbol: expr, $fn_signature: ty] ),* $(,)?) => {
                $(
                    let $name = unsafe {
                        std::mem::transmute(
                            lib
                                .get::<Symbol<$fn_signature>>($symbol)
                                .expect(&format!("Failed to find `{}`.", std::str::from_utf8($symbol).unwrap())),
                        )
                    };
                )*

                Self {
                    $($name),*
                }
            };
        }

        functions! {
            [ssygvd, b"LAPACKE_ssygvd", LapackeSsygvdFn],
            [dsygvd, b"LAPACKE_dsygvd", LapackeDsygvdFn],

            [ssbevd, b"LAPACKE_ssbevd", LapackeSsbevdFn],
            [dsbevd, b"LAPACKE_dsbevd", LapackeDsbevdFn],

            [sstemr, b"LAPACKE_sstemr", LapackeSstemrFn],
            [dstemr, b"LAPACKE_dstemr", LapackeDstemrFn],
        }
    }
}

impl<'a> LapackeFunctions<'a> {
    pub const DEFAULT_ZF64: Option<&'static mut Matrix<f64, U1, U1, ArrayStorage<f64, 1, 1>>> =
        None;
    pub const DEFAULT_ZF32: Option<&'static mut Matrix<f32, U1, U1, ArrayStorage<f32, 1, 1>>> =
        None;

    /// Solves an eigenvalue problem specified by the `problem_kind` argument.
    pub fn dsygvd<RA, CA, SA, RB, CB, SB, DW, SW>(
        &self,
        problem_kind: EigenProblemKind,
        output_kind: EigenOutputKind,
        uplo: Uplo,
        a: &mut Matrix<f64, RA, CA, SA>,
        b: &mut Matrix<f64, RB, CB, SB>,
        w: &mut Vector<f64, DW, SW>,
    ) where
        RA: Dim,
        CA: Dim,
        SA: StorageMut<f64, RA, CA>,
        RB: Dim,
        CB: Dim,
        SB: StorageMut<f64, RB, CB>,
        DW: Dim,
        SW: StorageMut<f64, DW>,
    {
        // assert!(a.is_square());
        // assert!(b.is_square());
        // assert!(w.len() >= b.dims.0 as usize);

        let n = b.data.shape().1.value() as LaInt;
        let lda = a.shape().0 as LaInt;
        let ldb = b.shape().0 as LaInt;

        (self.dsygvd)(
            cblas_sys::CblasColMajor,
            problem_kind.to_sys(),
            output_kind.to_sys(),
            uplo.to_sys(),
            n,
            a.data.ptr_mut(),
            lda,
            b.data.ptr_mut(),
            ldb,
            w.as_mut_ptr(),
        )
    }
    pub fn ssygvd<RA, CA, SA, RB, CB, SB, DW, SW>(
        &self,
        problem_kind: EigenProblemKind,
        output_kind: EigenOutputKind,
        uplo: Uplo,
        a: &mut Matrix<f32, RA, CA, SA>,
        b: &mut Matrix<f32, RB, CB, SB>,
        w: &mut Vector<f32, DW, SW>,
    ) where
        RA: Dim,
        CA: Dim,
        SA: StorageMut<f32, RA, CA>,
        RB: Dim,
        CB: Dim,
        SB: StorageMut<f32, RB, CB>,
        DW: Dim,
        SW: StorageMut<f32, DW>,
    {
        // assert!(a.major == b.major);
        // assert!(a.is_square());
        // assert!(b.is_square());
        // assert!(w.len() >= b.dims.0 as usize);

        let n = b.data.shape().1.value() as LaInt;
        let lda = a.data.strides().0.value() as LaInt;
        let ldb = b.data.strides().0.value() as LaInt;

        (self.ssygvd)(
            cblas_sys::CblasColMajor,
            problem_kind.to_sys(),
            output_kind.to_sys(),
            uplo.to_sys(),
            n,
            a.data.ptr_mut(),
            lda,
            b.data.ptr_mut(),
            ldb,
            w.as_mut_ptr(),
        )
    }

    pub fn ssbevd<RA, CA, SA, DW, SW, RZ, CZ, SZ>(
        &self,
        uplo: Uplo,
        ab: &mut Matrix<f32, RA, CA, SA>,
        w: &mut Vector<f32, DW, SW>,
        z: Option<&mut Matrix<f32, RZ, CZ, SZ>>,
    ) where
        RA: Dim,
        CA: Dim,
        SA: StorageMut<f32, RA, CA>,
        DW: Dim,
        SW: StorageMut<f32, DW>,
        RZ: Dim,
        CZ: Dim,
        SZ: StorageMut<f32, RZ, CZ>,
    {
        let n = ab.shape().0 as LaInt;
        let kd = (ab.shape().1 - 1) as LaInt;
        let ldab = ab.data.strides().0.value() as LaInt;

        if let Some(z) = z {
            let ldz = z.shape().0 as LaInt;
            (self.ssbevd)(
                cblas_sys::CblasColMajor,
                EigenOutputKind::Value.to_sys(),
                uplo.to_sys(),
                n,
                kd,
                ab.as_mut_ptr(),
                ldab,
                w.as_mut_ptr(),
                z.as_mut_ptr(),
                ldz,
            )
        } else {
            (self.ssbevd)(
                cblas_sys::CblasColMajor,
                EigenOutputKind::Value.to_sys(),
                uplo.to_sys(),
                n,
                kd,
                ab.as_mut_ptr(),
                ldab,
                w.as_mut_ptr(),
                std::ptr::null_mut(),
                0,
            )
        }
    }
    pub fn dsbevd<RA, CA, SA, DW, SW, RZ, CZ, SZ>(
        &self,
        uplo: Uplo,
        ab: &mut Matrix<f64, RA, CA, SA>,
        w: &mut Vector<f64, DW, SW>,
        z: Option<&mut Matrix<f64, RZ, CZ, SZ>>,
    ) where
        RA: Dim,
        CA: Dim,
        SA: StorageMut<f64, RA, CA>,
        DW: Dim,
        SW: StorageMut<f64, DW>,
        RZ: Dim,
        CZ: Dim,
        SZ: StorageMut<f64, RZ, CZ>,
    {
        let n = ab.shape().0 as LaInt;
        let kd = (ab.shape().1 - 1) as LaInt;
        let ldab = ab.data.strides().0.value() as LaInt;

        if let Some(z) = z {
            let ldz = z.shape().0 as LaInt;
            (self.dsbevd)(
                cblas_sys::CblasColMajor,
                EigenOutputKind::Value.to_sys(),
                uplo.to_sys(),
                n,
                kd,
                ab.as_mut_ptr(),
                ldab,
                w.as_mut_ptr(),
                z.as_mut_ptr(),
                ldz,
            )
        } else {
            (self.dsbevd)(
                cblas_sys::CblasColMajor,
                EigenOutputKind::Value.to_sys(),
                uplo.to_sys(),
                n,
                kd,
                ab.as_mut_ptr(),
                ldab,
                w.as_mut_ptr(),
                std::ptr::null_mut(),
                0,
            )
        }
    }

    pub fn sstemr<DD, SD, DE, SE, DW, SW, DS, SS, RZ, CZ, SZ>(
        &self,
        range: EigenRange<f32>,
        d: &mut Vector<f32, DD, SD>,
        e: &mut Vector<f32, DE, SE>,
        w: &mut Vector<f32, DW, SW>,
        isuppz: &mut Vector<LaInt, DS, SS>,
        z: Option<&mut Matrix<f32, RZ, CZ, SZ>>,
    ) where
        DD: Dim,
        SD: StorageMut<f32, DD>,
        DE: Dim,
        SE: StorageMut<f32, DE>,
        DW: Dim,
        SW: StorageMut<f32, DW>,
        DS: Dim,
        SS: StorageMut<LaInt, DS>,
        RZ: Dim,
        CZ: Dim,
        SZ: StorageMut<f32, RZ, CZ>,
    {
        let n = d.shape().0 as LaInt;
        let (il, iu) = range.get_sys_index(n as usize).unwrap_or_default();
        let (vl, vu) = range.get_sys_value().unwrap_or_default();
        let range = range.to_sys_char();

        let mut m = 0 as LaInt;

        if let Some(z) = z {
            let ldz = z.shape().0 as LaInt;
            (self.sstemr)(
                cblas_sys::CblasColMajor,
                EigenOutputKind::ValueVector.to_sys(),
                range,
                n,
                d.as_mut_ptr(),
                e.as_mut_ptr(),
                vl,
                vu,
                il,
                iu,
                &mut m,
                w.as_mut_ptr(),
                z.as_mut_ptr(),
                ldz,
                z.shape().1 as LaInt,
                isuppz.as_mut_ptr(),
                &mut false,
            )
        } else {
            (self.sstemr)(
                cblas_sys::CblasColMajor,
                EigenOutputKind::Value.to_sys(),
                range,
                n,
                d.as_mut_ptr(),
                e.as_mut_ptr(),
                vl,
                vu,
                il,
                iu,
                &mut m,
                w.as_mut_ptr(),
                std::ptr::null_mut(),
                0,
                0,
                isuppz.as_mut_ptr(),
                &mut false,
            )
        }
    }

    ///
    /// # Footguns
    /// Ensure vector e is the same length as d.
    /// Although there is one less element, ensure an extra memory space at the end of e since the routine requires it.
    pub fn dstemr<DD, SD, DE, SE, DW, SW, DS, SS, RZ, CZ, SZ>(
        &self,
        range: EigenRange<f64>,
        diagonal: &mut Vector<f64, DD, SD>,
        off_diagonal: &mut Vector<f64, DE, SE>,
        eig_out: &mut Vector<f64, DW, SW>,
        isuppz: &mut Vector<LaInt, DS, SS>,
        z: Option<&mut Matrix<f64, RZ, CZ, SZ>>,
    ) -> usize
    where
        DD: Dim,
        SD: StorageMut<f64, DD>,
        DE: Dim,
        SE: StorageMut<f64, DE>,
        DW: Dim,
        SW: StorageMut<f64, DW>,
        DS: Dim,
        SS: StorageMut<LaInt, DS>,
        RZ: Dim,
        CZ: Dim,
        SZ: StorageMut<f64, RZ, CZ>,
    {
        assert!(off_diagonal.len() == off_diagonal.len());

        let n = diagonal.shape().0 as LaInt;
        let (il, iu) = range.get_sys_index(n as usize).unwrap_or_default();
        let (vl, vu) = range.get_sys_value().unwrap_or_default();
        let range = range.to_sys_char();

        let mut m = 0 as LaInt;
        let mut tryrac = 0 as i32;

        let range_display = char::from_u32(range as u32).unwrap();

        println!("n = {n}\nil = {il}, iu = {iu}\nvl = {vl}, vu = {vu}\nrange = {range_display}\nm = {m}\ntryrac = {tryrac}");

        if let Some(z) = z {
            let ldz = z.shape().0 as LaInt;
            (self.dstemr)(
                cblas_sys::CblasColMajor,
                EigenOutputKind::ValueVector.to_sys(),
                range,
                n,
                diagonal.as_mut_ptr(),
                off_diagonal.as_mut_ptr(),
                vl,
                vu,
                il,
                iu,
                &mut m,
                eig_out.as_mut_ptr(),
                z.as_mut_ptr(),
                ldz,
                z.shape().1 as LaInt, // nzc, number of columns in z
                isuppz.as_mut_ptr(),
                &mut tryrac,
            )
        } else {
            (self.dstemr)(
                cblas_sys::CblasColMajor,
                EigenOutputKind::Value.to_sys(),
                range,
                n,
                diagonal.as_mut_ptr(),
                off_diagonal.as_mut_ptr(),
                vl,
                vu,
                il,
                iu,
                &mut m,
                eig_out.as_mut_ptr(),
                std::ptr::null_mut(),
                1,
                0,
                isuppz.as_mut_ptr(),
                &mut tryrac,
            )
        }

        m as usize
    }
}

pub struct LapackeFunctionsStatic {
    // This is only to make sure BlasLib is not dropped.
    pub(super) _lib: LapackeLib,
    pub(super) functions: LapackeFunctions<'static>,
}

impl Deref for LapackeFunctionsStatic {
    type Target = LapackeFunctions<'static>;
    fn deref(&self) -> &Self::Target {
        &self.functions
    }
}

/// Specifies a type of eigenvalue problem.
#[derive(Debug, Clone, Copy, Default, Hash, PartialEq, Eq)]
pub enum EigenProblemKind {
    #[default]
    /// A*x = B*lambda*x
    AxEqlBx,
    /// A*B*x = lambda*x
    ABxEqlx,
    /// B*A*x = lambda*x
    BAxEqlx,
}

impl EigenProblemKind {
    pub fn to_sys(&self) -> LaInt {
        match self {
            Self::AxEqlBx => 1,
            Self::ABxEqlx => 2,
            Self::BAxEqlx => 3,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Hash, PartialEq, Eq)]
pub enum Uplo {
    #[default]
    Upper,
    Lower,
}

impl Uplo {
    pub fn to_sys(&self) -> u8 {
        match self {
            Self::Upper => b'U',
            Self::Lower => b'L',
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Hash, PartialEq, Eq)]
pub enum EigenOutputKind {
    #[default]
    Value,
    ValueVector,
}

impl EigenOutputKind {
    pub fn to_sys(&self) -> u8 {
        (match self {
            Self::Value => b'N',
            Self::ValueVector => b'V',
        } as u8)
    }
}

type LapackeSsygvdFn = extern "C" fn(
    matrix_layout: cblas_sys::CBLAS_LAYOUT, // matrix_layout
    itype: LaInt,                           // itype
    jobz: u8,                               // jobz
    uplo: u8,                               // uplo
    n: LaInt,                               // n
    a: *mut c_float,                        // a
    lda: LaInt,                             // lda
    b: *mut c_float,                        // b
    ldb: LaInt,                             // ldb
    w: *mut c_float,                        // w
);
type LapackeDsygvdFn = extern "C" fn(
    matrix_layout: cblas_sys::CBLAS_LAYOUT, // matrix_layout
    itype: LaInt,                           // itype
    jobz: u8,                               // jobz
    uplo: u8,                               // uplo
    n: LaInt,                               // n
    a: *mut c_double,                       // a
    lda: LaInt,                             // lda
    b: *mut c_double,                       // b
    ldb: LaInt,                             // ldb
    w: *mut c_double,                       // w
);

type LapackeSsbevdFn = extern "C" fn(
    matrix_layout: cblas_sys::CBLAS_LAYOUT,
    jobz: u8,
    uplo: u8,
    n: LaInt,
    kd: LaInt,
    ab: *mut c_float,
    ldab: LaInt,
    w: *mut c_float,
    z: *mut c_float,
    ldz: LaInt,
);
type LapackeDsbevdFn = extern "C" fn(
    matrix_layout: cblas_sys::CBLAS_LAYOUT,
    jobz: u8,
    uplo: u8,
    n: LaInt,
    kd: LaInt,
    ab: *mut c_double,
    ldab: LaInt,
    w: *mut c_double,
    z: *mut c_double,
    ldz: LaInt,
);

type LapackeSstemrFn = extern "C" fn(
    matrix_layout: cblas_sys::CBLAS_LAYOUT,
    jobz: u8,
    range: u8,
    n: LaInt,
    d: *mut c_float,
    e: *mut c_float,
    vl: c_float,
    vu: c_float,
    il: LaInt,
    iu: LaInt,
    m: *mut LaInt,
    w: *mut c_float,
    z: *mut c_float,
    ldz: LaInt,
    nzc: LaInt,
    isuppz: *mut LaInt,
    tryrac: *mut bool,
);
type LapackeDstemrFn = extern "C" fn(
    matrix_layout: cblas_sys::CBLAS_LAYOUT,
    jobz: u8,
    range: u8,
    n: LaInt,
    d: *mut c_double,
    e: *mut c_double,
    vl: c_double,
    vu: c_double,
    il: LaInt,
    iu: LaInt,
    m: *mut LaInt,
    w: *mut c_double,
    z: *mut c_double,
    ldz: LaInt,
    nzc: LaInt,
    isuppz: *mut LaInt,
    tryrac: *mut i32, // bool
);
