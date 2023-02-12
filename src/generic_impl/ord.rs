use std::fmt::Debug;

use crate::{BTreeTrait, Query};

#[derive(Debug)]
pub struct OrdTrait<T> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T> Default for OrdTrait<T> {
    fn default() -> Self {
        Self {
            _phantom: Default::default(),
        }
    }
}

impl<T: Clone + Ord + Debug + 'static> BTreeTrait for OrdTrait<T> {
    type Elem = T;

    type Cache = Option<(T, T)>;

    const MAX_LEN: usize = 4;

    fn element_to_cache(_: &Self::Elem) -> Self::Cache {
        None
    }

    fn update_cache_internal<'a, Iter>(cache: &mut Self::Cache, iter: Iter)
    where
        Self::Cache: 'a,
        Iter: Iterator<Item = &'a Self::Cache>,
    {
    }

    fn update_cache_leaf<'a, Iter>(cache: &mut Self::Cache, elements: &[Self::Elem]) {
        todo!()
    }
}

impl<T: Ord + Clone + Debug + 'static> Query for OrdTrait<T> {
    type Cache = Option<(T, T)>;
    type Elem = T;
    type QueryArg = T;

    fn find_node<'a, 'b, Iter>(
        &mut self,
        target: &'b Self::QueryArg,
        iter: Iter,
    ) -> crate::FindResult
    where
        Iter: Iterator<Item = &'a Self::Cache>,
        Self::Cache: 'a,
    {
        let mut last = 0;
        for (i, elem) in iter.enumerate() {
            last = i;
            if let Some((start, end)) = elem {
                if start <= target && end >= target {
                    return crate::FindResult {
                        index: i,
                        offset: 0,
                        found: true,
                    };
                } else if start > target {
                    return crate::FindResult {
                        index: i,
                        offset: 0,
                        found: false,
                    };
                }
            }
        }

        crate::FindResult {
            index: last + 1,
            offset: 0,
            found: false,
        }
    }

    fn find_element<'a, 'b, Iter>(
        &mut self,
        target: &'b Self::QueryArg,
        iter: Iter,
    ) -> crate::FindResult
    where
        Iter: Iterator<Item = &'a Self::Elem>,
        Self::Elem: 'a,
    {
        let mut last = 0;
        for (i, elem) in iter.enumerate() {
            last = i;
            if elem == target {
                return crate::FindResult {
                    index: i,
                    offset: 0,
                    found: true,
                };
            } else if elem > target {
                return crate::FindResult {
                    index: i,
                    offset: 0,
                    found: false,
                };
            }
        }

        crate::FindResult {
            index: last + 1,
            offset: 0,
            found: false,
        }
    }
}
