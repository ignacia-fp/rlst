//! Operators on arrays

pub mod addition;
pub mod other;
pub mod scalar_mult;

pub fn test_simd() {
    use crate::rlst_dynamic_array2;
    use rlst_common::traits::*;

    let shape = [6, 13];
    let mut arr1 = rlst_dynamic_array2!(f32, shape);
    let mut arr2 = rlst_dynamic_array2!(f32, shape);
    let mut res = rlst_dynamic_array2!(f32, shape);

    arr1.fill_from_seed_equally_distributed(0);
    arr2.fill_from_seed_equally_distributed(0);

    // let arr3 = arr1.view() + arr2.view();

    let arr3 = 3.0 * arr1;

    res.fill_from(arr3.view());

    println!("{}", res[[0, 0]]);
}
