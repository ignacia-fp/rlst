//! Implementation of `?potrs` for solving from Cholesky factors.

use lapack::{cpotrs, dpotrs, spotrs, zpotrs};

use crate::base_types::{LapackResult, c32, c64};
use crate::dense::linalg::lapack::interface::lapack_return;

/// Triangle containing the Cholesky factor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum PotrsUplo {
    /// Upper-triangular factor `U`, with `A = U^H U`.
    Upper = b'U',
    /// Lower-triangular factor `L`, with `A = L L^H`.
    Lower = b'L',
}

/// Solve a Hermitian positive-definite system from existing Cholesky factors.
pub trait Potrs: Sized {
    /// Overwrite `b` with the solution of `A X = B`.
    #[allow(clippy::too_many_arguments)]
    fn potrs(
        uplo: PotrsUplo,
        n: usize,
        nrhs: usize,
        factor: &[Self],
        lda: usize,
        b: &mut [Self],
        ldb: usize,
    ) -> LapackResult<()>;
}

macro_rules! implement_potrs {
    ($scalar:ty, $potrs:expr) => {
        impl Potrs for $scalar {
            fn potrs(
                uplo: PotrsUplo,
                n: usize,
                nrhs: usize,
                factor: &[Self],
                lda: usize,
                b: &mut [Self],
                ldb: usize,
            ) -> LapackResult<()> {
                assert!(lda >= n.max(1));
                assert_eq!(factor.len(), lda * n);
                assert!(ldb >= n.max(1));
                assert_eq!(b.len(), ldb * nrhs);
                let mut info = 0;
                unsafe {
                    $potrs(
                        uplo as u8,
                        n as i32,
                        nrhs as i32,
                        factor,
                        lda as i32,
                        b,
                        ldb as i32,
                        &mut info,
                    );
                }
                lapack_return(info, ())
            }
        }
    };
}

implement_potrs!(f32, spotrs);
implement_potrs!(f64, dpotrs);
implement_potrs!(c32, cpotrs);
implement_potrs!(c64, zpotrs);
