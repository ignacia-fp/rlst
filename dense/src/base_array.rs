use crate::array::empty_chunk;
use crate::data_container::{DataContainer, DataContainerMut};
use crate::layout::{
    check_multi_index_in_bounds, convert_1d_nd_from_shape, convert_nd_raw, stride_from_shape,
};
use rlst_common::traits::{ChunkedAccess, UnsafeRandomAccessByValue, UnsafeRandomAccessMut};
use rlst_common::{
    traits::{Shape, UnsafeRandomAccessByRef},
    types::Scalar,
};

pub struct BaseArray<Item: Scalar, Data: DataContainer<Item = Item>, const NDIM: usize> {
    data: Data,
    shape: [usize; NDIM],
    stride: [usize; NDIM],
}

impl<Item: Scalar, Data: DataContainer<Item = Item>, const NDIM: usize>
    BaseArray<Item, Data, NDIM>
{
    pub fn new(data: Data, shape: [usize; NDIM]) -> Self {
        let stride = stride_from_shape(shape);
        Self {
            data,
            shape,
            stride,
        }
    }

    pub fn new_with_stride(data: Data, shape: [usize; NDIM], stride: [usize; NDIM]) -> Self {
        Self {
            data,
            shape,
            stride,
        }
    }
}

impl<Item: Scalar, Data: DataContainer<Item = Item>, const NDIM: usize> Shape<NDIM>
    for BaseArray<Item, Data, NDIM>
{
    fn shape(&self) -> [usize; NDIM] {
        self.shape
    }
}

impl<Item: Scalar, Data: DataContainer<Item = Item>, const NDIM: usize>
    UnsafeRandomAccessByRef<NDIM> for BaseArray<Item, Data, NDIM>
{
    type Item = Item;

    #[inline]
    unsafe fn get_unchecked(&self, multi_index: [usize; NDIM]) -> &Self::Item {
        debug_assert!(check_multi_index_in_bounds(multi_index, self.shape()));
        let index = convert_nd_raw(multi_index, self.stride);
        self.data.get_unchecked(index)
    }
}

impl<Item: Scalar, Data: DataContainer<Item = Item>, const NDIM: usize>
    UnsafeRandomAccessByValue<NDIM> for BaseArray<Item, Data, NDIM>
{
    type Item = Item;

    #[inline]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
        debug_assert!(check_multi_index_in_bounds(multi_index, self.shape()));
        let index = convert_nd_raw(multi_index, self.stride);
        self.data.get_unchecked_value(index)
    }
}

impl<Item: Scalar, Data: DataContainerMut<Item = Item>, const NDIM: usize>
    UnsafeRandomAccessMut<NDIM> for BaseArray<Item, Data, NDIM>
{
    type Item = Item;

    #[inline]
    unsafe fn get_unchecked_mut(&mut self, multi_index: [usize; NDIM]) -> &mut Self::Item {
        debug_assert!(check_multi_index_in_bounds(multi_index, self.shape()));
        let index = convert_nd_raw(multi_index, self.stride);
        self.data.get_unchecked_mut(index)
    }
}

impl<Item: Scalar, Data: DataContainerMut<Item = Item>, const N: usize, const NDIM: usize>
    ChunkedAccess<N> for BaseArray<Item, Data, NDIM>
{
    type Item = Item;

    #[inline]
    fn get_chunk(
        &self,
        chunk_index: usize,
    ) -> Option<rlst_common::types::DataChunk<Self::Item, N>> {
        let nelements = self.shape().iter().product();
        if let Some(mut chunk) = empty_chunk(chunk_index, nelements) {
            for count in 0..chunk.valid_entries {
                unsafe {
                    chunk.data[count] = self.get_value_unchecked(convert_1d_nd_from_shape(
                        chunk.start_index + count,
                        self.shape(),
                    ))
                }
            }
            Some(chunk)
        } else {
            None
        }
    }
}