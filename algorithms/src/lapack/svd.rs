//! Implement the SVD
use crate::lapack::LapackCompatible;
use crate::lapack::LapackData;
use crate::traits::svd::{Mode, Svd};
use lapacke;
use num::traits::Zero;
use rlst_common::traits::*;
use rlst_common::types::{c32, c64, RlstError, RlstResult, Scalar};
use rlst_dense::{rlst_mat, MatrixD};

macro_rules! implement_svd {
    ($scalar:ty, $lapack_gesvd:ident) => {
        impl<Mat: LapackCompatible> Svd for LapackData<$scalar, Mat>
        where
            Mat: RawAccess<T = $scalar>,
        {
            type T = $scalar;
            fn svd(
                mut self,
                u_mode: Mode,
                vt_mode: Mode,
            ) -> RlstResult<(
                Vec<<$scalar as Scalar>::Real>,
                Option<MatrixD<$scalar>>,
                Option<MatrixD<$scalar>>,
            )> {
                let m = self.mat.shape().0 as i32;
                let n = self.mat.shape().1 as i32;
                let k = std::cmp::min(m, n);
                let lda = self.mat.stride().1 as i32;

                let mut s_values = vec![<<$scalar as Scalar>::Real as Zero>::zero(); k as usize];
                let mut superb =
                    vec![<<$scalar as Scalar>::Real as Zero>::zero(); (k - 1) as usize];
                let jobu;
                let jobvt;
                let mut u_matrix;
                let mut vt_matrix;
                let ldu;
                let ldvt;

                match u_mode {
                    Mode::All => {
                        jobu = b'A';
                        u_matrix = Some(rlst_mat![$scalar, (m as usize, m as usize)]);
                        ldu = m as i32;
                    }
                    Mode::Slim => {
                        jobu = b'S';
                        u_matrix = Some(rlst_mat![$scalar, (m as usize, k as usize)]);
                        ldu = m as i32;
                    }
                    Mode::None => {
                        jobu = b'N';
                        u_matrix = None;
                        ldu = m as i32;
                    }
                };

                match vt_mode {
                    Mode::All => {
                        jobvt = b'A';
                        vt_matrix = Some(rlst_mat![$scalar, (n as usize, n as usize)]);
                        ldvt = n as i32;
                    }
                    Mode::Slim => {
                        jobvt = b'S';
                        vt_matrix = Some(rlst_mat![$scalar, (k as usize, n as usize)]);
                        ldvt = k as i32;
                    }
                    Mode::None => {
                        jobvt = b'N';
                        vt_matrix = None;
                        ldvt = k as i32;
                    }
                }

                let info = unsafe {
                    lapacke::$lapack_gesvd(
                        lapacke::Layout::ColumnMajor,
                        jobu,
                        jobvt,
                        m,
                        n,
                        self.mat.data_mut(),
                        lda,
                        s_values.as_mut_slice(),
                        u_matrix.as_mut().unwrap().data_mut(),
                        ldu,
                        vt_matrix.as_mut().unwrap().data_mut(),
                        ldvt,
                        superb.as_mut_slice(),
                    )
                };

                match info {
                    0 => return Ok((s_values, u_matrix, vt_matrix)),
                    _ => return Err(RlstError::LapackError(info)),
                }
            }
        }
    };
}

implement_svd!(f64, dgesvd);
implement_svd!(f32, sgesvd);
implement_svd!(c32, cgesvd);
implement_svd!(c64, zgesvd);

#[cfg(test)]
mod test {
    use crate::lapack::AsLapack;
    use approx::assert_relative_eq;
    use rand::SeedableRng;

    use super::*;
    use rand_chacha::ChaCha8Rng;
    use rlst_dense::Dot;

    #[test]
    fn test_thick_svd() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);

        let m = 3;
        let n = 4;
        let k = std::cmp::min(m, n);
        let mut mat = rlst_mat!(f64, (m, n));

        mat.fill_from_equally_distributed(&mut rng);
        let expected = mat.copy();

        let (singular_values, u_matrix, vt_matrix) = mat
            .into_lapack()
            .unwrap()
            .svd(Mode::Slim, Mode::Slim)
            .unwrap();

        let u_matrix = u_matrix.unwrap();
        let vt_matrix = vt_matrix.unwrap();

        let mut sigma = rlst_mat!(f64, (k, k));
        for index in 0..k {
            sigma[[index, index]] = singular_values[index];
        }

        let tmp = sigma.dot(&vt_matrix);

        let actual = u_matrix.dot(&tmp);

        for (a, e) in actual.iter_col_major().zip(expected.iter_col_major()) {
            assert_relative_eq!(a, e, epsilon = 1E-13);
        }
    }

    #[test]
    fn test_thin_svd() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);

        let m = 4;
        let n = 3;
        let k = std::cmp::min(m, n);
        let mut mat = rlst_mat!(f64, (m, n));

        mat.fill_from_equally_distributed(&mut rng);
        let expected = mat.copy();

        let (singular_values, u_matrix, vt_matrix) = mat
            .into_lapack()
            .unwrap()
            .svd(Mode::Slim, Mode::Slim)
            .unwrap();

        let u_matrix = u_matrix.unwrap();
        let vt_matrix = vt_matrix.unwrap();

        let mut sigma = rlst_mat!(f64, (k, k));
        for index in 0..k {
            sigma[[index, index]] = singular_values[index];
        }

        let tmp = sigma.dot(&vt_matrix);

        let actual = u_matrix.dot(&tmp);

        for (a, e) in actual.iter_col_major().zip(expected.iter_col_major()) {
            assert_relative_eq!(a, e, epsilon = 1E-13);
        }
    }

    #[test]
    fn test_full_svd() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);

        let m = 5;
        let n = 5;
        let k = std::cmp::min(m, n);
        let mut mat = rlst_mat!(f64, (m, n));

        mat.fill_from_equally_distributed(&mut rng);
        let expected = mat.copy();

        let (singular_values, u_matrix, vt_matrix) = mat
            .into_lapack()
            .unwrap()
            .svd(Mode::All, Mode::All)
            .unwrap();

        let u_matrix = u_matrix.unwrap();
        let vt_matrix = vt_matrix.unwrap();

        let mut sigma = rlst_mat!(f64, (k, k));
        for index in 0..k {
            sigma[[index, index]] = singular_values[index];
        }

        let tmp = sigma.dot(&vt_matrix);

        let actual = u_matrix.dot(&tmp);

        for (a, e) in actual.iter_col_major().zip(expected.iter_col_major()) {
            assert_relative_eq!(a, e, epsilon = 1E-13);
        }
    }
}
