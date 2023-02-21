use crate::{BTreeTrait, FindResult, HeapVec, Query, QueryResult, SmallElemVec};

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

    #[allow(unused)]
    fn finder_delete(
        elements: &mut HeapVec<B::Elem>,
        elem_index: usize,
        offset: usize,
    ) -> Option<B::Elem> {
        if elem_index >= elements.len() {
            return None;
        }

        Some(elements.remove(elem_index))
    }

    #[allow(unused)]
    fn finder_delete_range(
        elements: &mut HeapVec<B::Elem>,
        start: Option<QueryResult>,
        end: Option<QueryResult>,
    ) -> SmallElemVec<B::Elem> {
        match (start, end) {
            (None, None) => elements.drain(..).collect(),
            (None, Some(to)) => elements.drain(..to.elem_index).collect(),
            (Some(from), None) => elements.drain(from.elem_index..).collect(),
            (Some(from), Some(to)) => elements.drain(from.elem_index..to.elem_index).collect(),
        }
    }
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

    #[inline]
    fn find_element(
        &mut self,
        _: &Self::QueryArg,
        elements: &[<B as BTreeTrait>::Elem],
    ) -> crate::FindResult {
        B::find_element_by_offset(elements, self.left)
    }

    fn delete(
        elements: &mut crate::HeapVec<<B as BTreeTrait>::Elem>,
        _: &Self::QueryArg,
        elem_index: usize,
        offset: usize,
    ) -> Option<<B as BTreeTrait>::Elem> {
        B::finder_delete(elements, elem_index, offset)
    }

    fn delete_range(
        elements: &mut crate::HeapVec<<B as BTreeTrait>::Elem>,
        _: &Self::QueryArg,
        _: &Self::QueryArg,
        start: Option<crate::QueryResult>,
        end: Option<crate::QueryResult>,
    ) -> crate::SmallElemVec<<B as BTreeTrait>::Elem> {
        B::finder_delete_range(elements, start, end)
    }
}
