use std::fmt::Debug;

use crate::rle::HasLength;
use crate::{BTreeTrait, FindResult, Query};

/// A generic length finder
pub struct LengthFinder {
    pub left: usize,
}

impl LengthFinder {
    #[inline(always)]
    pub fn new() -> Self {
        Self { left: 0 }
    }
}

impl Default for LengthFinder {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

pub trait UseLengthFinder<B: BTreeTrait> {
    fn get_len(cache: &B::Cache) -> usize;
}

impl<Elem: 'static + HasLength + Debug, B: BTreeTrait<Elem = Elem> + UseLengthFinder<B>> Query<B>
    for LengthFinder
{
    type QueryArg = usize;

    #[inline(always)]
    fn init(target: &Self::QueryArg) -> Self {
        Self { left: *target }
    }

    #[inline(always)]
    fn find_node(
        &mut self,
        _: &Self::QueryArg,
        child_caches: &[crate::Child<B>],
    ) -> crate::FindResult {
        let mut last_left = self.left;
        for (i, cache) in child_caches.iter().enumerate() {
            let len = B::get_len(&cache.cache);
            if self.left >= len {
                last_left = self.left;
                self.left -= len;
            } else {
                return FindResult::new_found(i, self.left);
            }
        }

        self.left = last_left;
        FindResult::new_missing(child_caches.len() - 1, last_left)
    }

    #[inline(always)]
    fn confirm_elem(
        &mut self,
        _: &Self::QueryArg,
        elem: &<B as BTreeTrait>::Elem,
    ) -> (usize, bool) {
        (self.left, self.left < elem.rle_len())
    }
}
