use std::fmt::Debug;

use crate::{BTree, BTreeTrait, FindResult, Query};

#[derive(Debug)]
struct OrdTrait<T> {
    _phantom: std::marker::PhantomData<T>,
}

#[derive(Debug)]
pub struct OrdTreeSet<T: Clone + Ord + Debug + 'static>(BTree<OrdTrait<T>>);

impl<T: Clone + Ord + Debug + 'static> OrdTreeSet<T> {
    #[inline(always)]
    pub fn new() -> Self {
        Self(BTree::new())
    }

    #[inline(always)]
    pub fn insert(&mut self, value: T) {
        let result = self.0.query::<OrdTrait<T>>(&value);
        if !result.found {
            self.0.insert_by_query_result(result, value);
        }
    }

    #[inline(always)]
    pub fn delete(&mut self, value: &T) -> bool {
        self.0.delete::<OrdTrait<T>>(value)
    }

    #[inline(always)]
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.0.iter()
    }

    pub(crate) fn check(&self) {
        self.0.check()
    }
}

impl<T: Clone + Ord + Debug + 'static> Default for OrdTreeSet<T> {
    fn default() -> Self {
        Self::new()
    }
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

    const MAX_LEN: usize = 16;

    fn element_to_cache(_: &Self::Elem) -> Self::Cache {
        None
    }

    fn calc_cache_internal(caches: &[crate::Child<Self::Cache>]) -> Self::Cache {
        Some((
            caches[0].cache.as_ref().unwrap().0.clone(),
            caches[caches.len() - 1].cache.as_ref().unwrap().1.clone(),
        ))
    }

    fn calc_cache_leaf(elements: &[Self::Elem]) -> Self::Cache {
        Some((elements[0].clone(), elements[elements.len() - 1].clone()))
    }
}

impl<T: Ord + Clone + Debug + 'static> Query for OrdTrait<T> {
    type Cache = Option<(T, T)>;
    type Elem = T;
    type QueryArg = T;

    fn find_node(
        &mut self,
        target: &Self::QueryArg,
        child_caches: &[crate::Child<Self::Cache>],
    ) -> crate::FindResult {
        for (i, child) in child_caches.iter().enumerate() {
            let (min, max) = child.cache.as_ref().unwrap();
            if target < min {
                return FindResult::new_missing(i, 0);
            }
            if target >= min && target <= max {
                return FindResult::new_found(i, 0);
            }
        }

        FindResult::new_missing(child_caches.len(), 0)
    }

    fn find_element(&mut self, target: &T, elements: &[T]) -> crate::FindResult {
        match elements.binary_search(target) {
            Ok(i) => FindResult::new_found(i, 0),
            Err(i) => FindResult::new_missing(i, 0),
        }
    }
}

#[cfg(test)]
mod test {
    use rand::{Rng, SeedableRng};

    use super::*;

    #[test]
    fn test() {
        let mut tree: OrdTreeSet<u64> = OrdTreeSet::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let mut data: Vec<u64> = (0..10_000).map(|_| rng.gen()).collect();
        for &value in data.iter() {
            tree.insert(value);
        }
        data.sort_unstable();
        assert_eq!(tree.iter().copied().collect::<Vec<_>>(), data);
    }
}
