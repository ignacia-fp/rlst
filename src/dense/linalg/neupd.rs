//! Non-Symmetric Arnoldi Eigenvector Extraction
use crate::dense::linalg::native_arpack::{
    approx_eq_complex, combine_basis_complex, combine_basis_real, extract_selected_eigenpairs,
    with_states_c32, with_states_c64, with_states_f32, with_states_f64,
};
use crate::dense::linalg::naupd::ArnoldiBackend;
use crate::RlstScalar;
#[cfg(not(target_os = "macos"))]
use arpack_ng_sys::{__BindgenComplex, cneupd_c, dneupd_c, sneupd_c, zneupd_c};
#[cfg(all(target_os = "macos", feature = "macos-legacy-arpack"))]
use arpack_ng_sys::{__BindgenComplex, cneupd_c, dneupd_c, sneupd_c, zneupd_c};
use num::{Complex, Float, Zero};

/// Implementation of the Non-Symmetric Arnoldi Eigenvector Extraction
pub trait NonSymmetricArnoldiExtract: RlstScalar {
    /// Non-Symmetric Arnoldi Eigenvector Extraction ARPACK-compatible entry point.
    fn neupd(
        rvec: i32,
        howmny: &str,
        select: &mut [i32],
        d: &mut [<Self as RlstScalar>::Complex],
        z: &mut [Self],
        ldz: i32,
        sigmar: <Self as RlstScalar>::Real,
        sigmai: <Self as RlstScalar>::Real,
        workev: &mut [Self],
        bmat: &str,
        n: i32,
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
        Self::neupd_with_backend(
            ArnoldiBackend::Auto,
            rvec,
            howmny,
            select,
            d,
            z,
            ldz,
            sigmar,
            sigmai,
            workev,
            bmat,
            n,
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

    /// Non-Symmetric Arnoldi Eigenvector Extraction with an explicit backend choice.
    fn neupd_with_backend(
        backend: ArnoldiBackend,
        rvec: i32,
        howmny: &str,
        select: &mut [i32],
        d: &mut [<Self as RlstScalar>::Complex],
        z: &mut [Self],
        ldz: i32,
        sigmar: <Self as RlstScalar>::Real,
        sigmai: <Self as RlstScalar>::Real,
        workev: &mut [Self],
        bmat: &str,
        n: i32,
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

fn zero_output<Item: RlstScalar>(d: &mut [Item::Complex], z: &mut [Item]) {
    for item in d.iter_mut() {
        *item = Item::Complex::zero();
    }
    for item in z.iter_mut() {
        *item = Item::zero();
    }
}

fn store_column<Item: RlstScalar>(z: &mut [Item], ldz: usize, col: usize, values: &[Item]) {
    let start = ldz * col;
    let end = start + values.len();
    if end <= z.len() {
        z[start..end].copy_from_slice(values);
    }
}

macro_rules! impl_neupd_complex {
    ($real:ty, $with_states:ident, $neupd_c:expr) => {
        impl NonSymmetricArnoldiExtract for Complex<$real> {
            #[allow(unused_variables)]
            fn neupd_with_backend(
                backend: ArnoldiBackend,
                rvec: i32,
                howmny: &str,
                select: &mut [i32],
                d: &mut [Complex<$real>],
                z: &mut [Complex<$real>],
                ldz: i32,
                sigmar: $real,
                sigmai: $real,
                workev: &mut [Complex<$real>],
                bmat: &str,
                n: i32,
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
                        zero_output(d, z);

                        if n < 0 || nev < 0 || ldv < n || ldz < n {
                            *info = -1;
                            return;
                        }

                        let key = v.as_ptr() as usize;
                        let mut result = None;
                        $with_states(|states| {
                            result = extract_selected_eigenpairs(states, key, nev as usize);
                        });

                        let Some(selected) = result else {
                            *info = -1;
                            return;
                        };

                        let count = selected.values.len().min(d.len());
                        d[..count].copy_from_slice(&selected.values[..count]);

                        if rvec != 0 {
                            let dim = n as usize;
                            let ldz = ldz as usize;

                            for col in 0..count {
                                let start = selected.arnoldi_dim * col;
                                let end = start + selected.arnoldi_dim;
                                let vector = combine_basis_complex(
                                    v,
                                    ldv as usize,
                                    dim,
                                    selected.arnoldi_dim,
                                    &selected.vectors[start..end],
                                );
                                store_column(z, ldz, col, &vector);
                            }
                        }

                        *info = 0;
                    }
                    ArnoldiBackend::Legacy => {
                        #[cfg(any(
                            not(target_os = "macos"),
                            all(target_os = "macos", feature = "macos-legacy-arpack")
                        ))]
                        {
                            let sigmar_c = __BindgenComplex {
                                re: sigmar,
                                im: sigmai,
                            };
                            unsafe {
                                $neupd_c(
                                    rvec,
                                    howmny.as_ptr() as *const i8,
                                    select.as_mut_ptr(),
                                    d.as_mut_ptr() as *mut __BindgenComplex<$real>,
                                    z.as_mut_ptr() as *mut __BindgenComplex<$real>,
                                    ldz,
                                    sigmar_c,
                                    workev.as_mut_ptr() as *mut __BindgenComplex<$real>,
                                    bmat.as_ptr() as *const i8,
                                    n,
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
                        }
                        #[cfg(not(any(
                            not(target_os = "macos"),
                            all(target_os = "macos", feature = "macos-legacy-arpack")
                        )))]
                        {
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

macro_rules! impl_neupd_real {
    ($scalar:ty, $with_states:ident, $neupd_c:expr) => {
        impl NonSymmetricArnoldiExtract for $scalar {
            #[allow(unused_variables)]
            fn neupd_with_backend(
                backend: ArnoldiBackend,
                rvec: i32,
                howmny: &str,
                select: &mut [i32],
                d: &mut [Complex<Self>],
                z: &mut [Self],
                ldz: i32,
                sigmar: Self,
                sigmai: Self,
                workev: &mut [Self],
                bmat: &str,
                n: i32,
                which: &str,
                nev: i32,
                tol: $scalar,
                resid: &mut [Self],
                ncv: i32,
                v: &mut [Self],
                ldv: i32,
                iparam: &mut [i32; 11],
                ipntr: &mut [i32; 14],
                workd: &mut [Self],
                workl: &mut [Self],
                lworkl: i32,
                _rwork: &mut [Self],
                info: &mut i32,
            ) {
                match backend.resolve() {
                    ArnoldiBackend::Native => {
                        zero_output(d, z);

                        if n < 0 || nev < 0 || ldv < n || ldz < n {
                            *info = -1;
                            return;
                        }

                        let key = v.as_ptr() as usize;
                        let mut result = None;
                        $with_states(|states| {
                            result = extract_selected_eigenpairs(states, key, nev as usize);
                        });

                        let Some(selected) = result else {
                            *info = -1;
                            return;
                        };

                        let count = selected.values.len().min(d.len());
                        d[..count].copy_from_slice(&selected.values[..count]);

                        if rvec != 0 {
                            let dim = n as usize;
                            let ldz = ldz as usize;
                            let total_cols = if ldz == 0 { 0 } else { z.len() / ldz };
                            let pair_tol = tol.max(<$scalar>::epsilon().sqrt());
                            let mut out_col = 0usize;
                            let mut eig_col = 0usize;

                            while eig_col < count && out_col < total_cols {
                                let value = selected.values[eig_col];
                                let start = selected.arnoldi_dim * eig_col;
                                let end = start + selected.arnoldi_dim;
                                let (re, im) = combine_basis_real(
                                    v,
                                    ldv as usize,
                                    dim,
                                    selected.arnoldi_dim,
                                    &selected.vectors[start..end],
                                );

                                store_column(z, ldz, out_col, &re);

                                let has_pair = eig_col + 1 < count
                                    && approx_eq_complex(
                                        selected.values[eig_col + 1],
                                        value.conj(),
                                        pair_tol,
                                    );

                                if value.im.abs() > pair_tol && out_col + 1 < total_cols {
                                    store_column(z, ldz, out_col + 1, &im);
                                }

                                eig_col += if has_pair { 2 } else { 1 };
                                out_col += if value.im.abs() > pair_tol { 2 } else { 1 };
                            }
                        }

                        *info = 0;
                    }
                    ArnoldiBackend::Legacy => {
                        #[cfg(any(
                            not(target_os = "macos"),
                            all(target_os = "macos", feature = "macos-legacy-arpack")
                        ))]
                        {
                            let mut di: Vec<Self> = (0..nev).map(|_| num::Zero::zero()).collect();
                            let mut dr: Vec<Self> = (0..nev).map(|_| num::Zero::zero()).collect();
                            unsafe {
                                $neupd_c(
                                    rvec,
                                    howmny.as_ptr() as *const i8,
                                    select.as_mut_ptr(),
                                    dr.as_mut_ptr(),
                                    di.as_mut_ptr(),
                                    z.as_mut_ptr(),
                                    ldz,
                                    sigmar,
                                    sigmai,
                                    workev.as_mut_ptr(),
                                    bmat.as_ptr() as *const i8,
                                    n,
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

                            for ((d_elem, &re), &im) in d.iter_mut().zip(dr.iter()).zip(di.iter()) {
                                *d_elem = Complex::new(re, im);
                            }
                        }
                        #[cfg(not(any(
                            not(target_os = "macos"),
                            all(target_os = "macos", feature = "macos-legacy-arpack")
                        )))]
                        {
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

impl_neupd_real!(f32, with_states_f32, sneupd_c);
impl_neupd_real!(f64, with_states_f64, dneupd_c);
impl_neupd_complex!(f32, with_states_c32, cneupd_c);
impl_neupd_complex!(f64, with_states_c64, zneupd_c);
