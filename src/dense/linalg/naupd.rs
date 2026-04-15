//! Non-Symmetric Arnoldi Update
use crate::dense::linalg::native_arpack::{
    naupd_with_state, with_states_c32, with_states_c64, with_states_f32, with_states_f64,
};
use crate::RlstScalar;
#[cfg(not(target_os = "macos"))]
use arpack_ng_sys::{__BindgenComplex, cnaupd_c, dnaupd_c, snaupd_c, znaupd_c};
#[cfg(all(target_os = "macos", feature = "macos-legacy-arpack"))]
use arpack_ng_sys::{__BindgenComplex, cnaupd_c, dnaupd_c, snaupd_c, znaupd_c};
use num::Complex;

/// Select which Arnoldi backend to use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArnoldiBackend {
    /// Use the platform default backend.
    Auto,
    /// Use the native Rust implementation.
    Native,
    /// Use the legacy ARPACK-backed implementation.
    Legacy,
}

impl ArnoldiBackend {
    pub(crate) fn resolve(self) -> Self {
        match self {
            Self::Auto => {
                if cfg!(target_os = "macos") {
                    Self::Native
                } else {
                    Self::Legacy
                }
            }
            other => other,
        }
    }
}

/// Implementation of the Non-Symmetric Arnoldi Update
pub trait NonSymmetricArnoldiUpdate: RlstScalar {
    /// Non-Symmetric Arnoldi Update ARPACK-compatible entry point.
    fn naupd(
        ido: &mut i32,
        bmat: &str,
        dim: i32,
        which: &str,
        nev: i32,
        tol: <Self as RlstScalar>::Real,
        resid: &mut [Self],
        ncv: i32,
        v: &mut [Self],
        ldv: i32,
        iparam: &mut [i32; 11],
        ipntr: &mut [i32; 14],
        workd: &mut [Self],
        workl: &mut [Self],
        lworkl: i32,
        rwork: &mut [<Self as RlstScalar>::Real],
        info: &mut i32,
    ) {
        Self::naupd_with_backend(
            ArnoldiBackend::Auto,
            ido,
            bmat,
            dim,
            which,
            nev,
            tol,
            resid,
            ncv,
            v,
            ldv,
            iparam,
            ipntr,
            workd,
            workl,
            lworkl,
            rwork,
            info,
        );
    }

    /// Non-Symmetric Arnoldi Update with an explicit backend choice.
    fn naupd_with_backend(
        backend: ArnoldiBackend,
        ido: &mut i32,
        bmat: &str,
        dim: i32,
        which: &str,
        nev: i32,
        tol: <Self as RlstScalar>::Real,
        resid: &mut [Self],
        ncv: i32,
        v: &mut [Self],
        ldv: i32,
        iparam: &mut [i32; 11],
        ipntr: &mut [i32; 14],
        workd: &mut [Self],
        workl: &mut [Self],
        lworkl: i32,
        rwork: &mut [<Self as RlstScalar>::Real],
        info: &mut i32,
    );
}

macro_rules! impl_naupd_real {
    ($scalar:ty, $with_states:ident, $naupd_c:expr) => {
        impl NonSymmetricArnoldiUpdate for $scalar {
            #[allow(unused_variables)]
            fn naupd_with_backend(
                backend: ArnoldiBackend,
                ido: &mut i32,
                bmat: &str,
                dim: i32,
                which: &str,
                nev: i32,
                tol: $scalar,
                resid: &mut [$scalar],
                ncv: i32,
                v: &mut [$scalar],
                ldv: i32,
                iparam: &mut [i32; 11],
                ipntr: &mut [i32; 14],
                workd: &mut [$scalar],
                workl: &mut [$scalar],
                lworkl: i32,
                _rwork: &mut [$scalar],
                info: &mut i32,
            ) {
                match backend.resolve() {
                    ArnoldiBackend::Native => {
                        $with_states(|states| {
                            naupd_with_state(
                                states, ido, bmat, dim, which, nev, tol, resid, ncv, v, ldv,
                                iparam, ipntr, workd, info,
                            )
                        });
                    }
                    ArnoldiBackend::Legacy => {
                        #[cfg(any(
                            not(target_os = "macos"),
                            all(target_os = "macos", feature = "macos-legacy-arpack")
                        ))]
                        unsafe {
                            $naupd_c(
                                ido,
                                bmat.as_ptr() as *const i8,
                                dim,
                                which.as_ptr() as *const i8,
                                nev,
                                tol,
                                resid.as_mut_ptr(),
                                ncv,
                                v.as_mut_ptr(),
                                ldv,
                                iparam.as_mut_ptr(),
                                ipntr.as_mut_ptr(),
                                workd.as_mut_ptr(),
                                workl.as_mut_ptr(),
                                lworkl,
                                info,
                            );
                        }
                        #[cfg(not(any(
                            not(target_os = "macos"),
                            all(target_os = "macos", feature = "macos-legacy-arpack")
                        )))]
                        {
                            let _ = (workl, lworkl);
                            panic!(
                                "Legacy ARPACK backend is unavailable on macOS unless feature `macos-legacy-arpack` is enabled."
                            );
                        }
                    }
                    ArnoldiBackend::Auto => unreachable!(),
                }
            }
        }
    };
}

macro_rules! impl_naupd_complex {
    ($real:ty, $with_states:ident, $naupd_c:expr) => {
        impl NonSymmetricArnoldiUpdate for Complex<$real> {
            #[allow(unused_variables)]
            fn naupd_with_backend(
                backend: ArnoldiBackend,
                ido: &mut i32,
                bmat: &str,
                dim: i32,
                which: &str,
                nev: i32,
                tol: $real,
                resid: &mut [Complex<$real>],
                ncv: i32,
                v: &mut [Complex<$real>],
                ldv: i32,
                iparam: &mut [i32; 11],
                ipntr: &mut [i32; 14],
                workd: &mut [Complex<$real>],
                workl: &mut [Complex<$real>],
                lworkl: i32,
                rwork: &mut [$real],
                info: &mut i32,
            ) {
                match backend.resolve() {
                    ArnoldiBackend::Native => {
                        $with_states(|states| {
                            naupd_with_state(
                                states, ido, bmat, dim, which, nev, tol, resid, ncv, v, ldv,
                                iparam, ipntr, workd, info,
                            )
                        });
                    }
                    ArnoldiBackend::Legacy => {
                        #[cfg(any(
                            not(target_os = "macos"),
                            all(target_os = "macos", feature = "macos-legacy-arpack")
                        ))]
                        unsafe {
                            $naupd_c(
                                ido,
                                bmat.as_ptr() as *const i8,
                                dim,
                                which.as_ptr() as *const i8,
                                nev,
                                tol,
                                resid.as_mut_ptr() as *mut __BindgenComplex<$real>,
                                ncv,
                                v.as_mut_ptr() as *mut __BindgenComplex<$real>,
                                ldv,
                                iparam.as_mut_ptr(),
                                ipntr.as_mut_ptr(),
                                workd.as_mut_ptr() as *mut __BindgenComplex<$real>,
                                workl.as_mut_ptr() as *mut __BindgenComplex<$real>,
                                lworkl,
                                rwork.as_mut_ptr(),
                                info,
                            );
                        }
                        #[cfg(not(any(
                            not(target_os = "macos"),
                            all(target_os = "macos", feature = "macos-legacy-arpack")
                        )))]
                        {
                            let _ = (workl, lworkl, rwork);
                            panic!(
                                "Legacy ARPACK backend is unavailable on macOS unless feature `macos-legacy-arpack` is enabled."
                            );
                        }
                    }
                    ArnoldiBackend::Auto => unreachable!(),
                }
            }
        }
    };
}

impl_naupd_real!(f32, with_states_f32, snaupd_c);
impl_naupd_real!(f64, with_states_f64, dnaupd_c);
impl_naupd_complex!(f32, with_states_c32, cnaupd_c);
impl_naupd_complex!(f64, with_states_c64, znaupd_c);
