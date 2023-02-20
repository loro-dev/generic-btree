use crate::{BTreeTrait, FindResult, Query};

/// A generic length finder
pub struct LengthFinder {
    pub left: usize,
}

impl LengthFinder {
    pub fn new() -> Self {
        Self { left: 0 }
    }
}

impl Default for LengthFinder {
    fn default() -> Self {
        Self::new()
    }
}

pub trait UseLengthFinder<B: BTreeTrait> {
    fn get_len(cache: &B::Cache) -> usize;
    fn find_element_by_offset(
        elements: &[<B as BTreeTrait>::Elem],
        offset: usize,
    ) -> crate::FindResult;
}

impl<B: BTreeTrait + UseLengthFinder<B>> Query<B> for LengthFinder {
    type QueryArg = usize;

    fn init(target: &Self::QueryArg) -> Self {
        Self { left: *target }
    }

    fn find_node(
        &mut self,
        _: &Self::QueryArg,
        child_caches: &[crate::Child<B>],
    ) -> crate::FindResult {
        for (i, cache) in child_caches.iter().enumerate() {
            let len = B::get_len(&cache.cache);
            if self.left > len {
                self.left -= len;
            } else {
                return FindResult::new_found(i, self.left);
            }
        }

        FindResult::new_missing(child_caches.len(), self.left)
    }

    fn find_element(
        &mut self,
        _: &Self::QueryArg,
        elements: &[<B as BTreeTrait>::Elem],
    ) -> crate::FindResult {
        B::find_element_by_offset(elements, self.left)
    }
}
