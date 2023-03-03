use crate::{BTreeTrait, FindResult, HeapVec, Query, QueryResult};

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
    fn finder_drain_range(
        elements: &mut HeapVec<B::Elem>,
        start: Option<QueryResult>,
        end: Option<QueryResult>,
    ) -> Box<dyn Iterator<Item = B::Elem> + '_> {
        Box::new(match (start, end) {
            (None, None) => elements.drain(..),
            (None, Some(to)) => elements.drain(..to.elem_index),
            (Some(from), None) => elements.drain(from.elem_index..),
            (Some(from), Some(to)) => elements.drain(from.elem_index..to.elem_index),
        })
    }

    #[allow(unused)]
    fn finder_delete_range(
        elements: &mut HeapVec<B::Elem>,
        start: Option<QueryResult>,
        end: Option<QueryResult>,
    ) {
        match (start, end) {
            (None, None) => {
                elements.clear();
            }
            (None, Some(to)) => {
                elements.drain(..to.elem_index);
            }
            (Some(from), None) => {
                elements.drain(from.elem_index..);
            }
            (Some(from), Some(to)) => {
                elements.drain(from.elem_index..to.elem_index);
            }
        };
    }
}

impl<Elem: 'static, B: BTreeTrait<Elem = Elem> + UseLengthFinder<B>> Query<B> for LengthFinder {
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

    fn drain_range<'a, 'b>(
        elements: &'a mut crate::HeapVec<<B as BTreeTrait>::Elem>,
        _: &'b Self::QueryArg,
        _: &'b Self::QueryArg,
        start: Option<crate::QueryResult>,
        end: Option<crate::QueryResult>,
    ) -> Box<dyn Iterator<Item = <B as BTreeTrait>::Elem> + 'a> {
        B::finder_drain_range(elements, start, end)
    }

    fn delete_range(
        elements: &mut HeapVec<<B as BTreeTrait>::Elem>,
        _: &Self::QueryArg,
        _: &Self::QueryArg,
        start: Option<QueryResult>,
        end: Option<QueryResult>,
    ) {
        B::finder_delete_range(elements, start, end)
    }
}
