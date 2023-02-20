use core::ops::RangeBounds;

pub trait Sliceable<T = usize> {
    fn slice(&self, range: impl RangeBounds<T>) -> Self;
    fn slice_(&mut self, range: impl RangeBounds<T>);
}

pub trait Length<T = usize> {
    fn rle_len(&self) -> T;
}
