use std::fmt::Debug;

use crate::{BTree, BTreeTrait, FindResult, Query};

#[derive(Debug)]
struct OrdTrait<Key, Value> {
    _phantom: std::marker::PhantomData<(Key, Value)>,
}

#[derive(Debug)]
pub struct OrdTreeMap<Key: Clone + Ord + Debug + 'static, Value: Clone + Debug>(
    BTree<OrdTrait<Key, Value>>,
);

#[derive(Debug)]
pub struct OrdTreeSet<Key: Clone + Ord + Debug + 'static>(OrdTreeMap<Key, ()>);

impl<Key: Clone + Ord + Debug + 'static, Value: Clone + Debug> OrdTreeMap<Key, Value> {
    #[inline(always)]
    pub fn new() -> Self {
        Self(BTree::new())
    }

    #[inline(always)]
    pub fn insert(&mut self, key: Key, value: Value) {
        let result = self.0.query::<OrdTrait<Key, Value>>(&key);
        if !result.found {
            self.0.insert_by_query_result(result, (key, value));
        }
    }

    #[inline(always)]
    pub fn delete(&mut self, value: &Key) -> bool {
        self.0.delete::<OrdTrait<Key, Value>>(value)
    }

    #[inline(always)]
    pub fn iter(&self) -> impl Iterator<Item = &(Key, Value)> {
        self.0.iter()
    }

    #[inline(always)]
    pub fn iter_key(&self) -> impl Iterator<Item = &Key> {
        self.0.iter().map(|x| &x.0)
    }

    pub(crate) fn check(&self) {
        self.0.check()
    }
}

impl<Key: Clone + Ord + Debug + 'static> OrdTreeSet<Key> {
    #[inline(always)]
    pub fn new() -> Self {
        Self(OrdTreeMap::new())
    }

    #[inline(always)]
    pub fn insert(&mut self, key: Key) {
        self.0.insert(key, ());
    }

    #[inline(always)]
    pub fn delete(&mut self, key: &Key) -> bool {
        self.0.delete(key)
    }

    #[inline(always)]
    pub fn iter(&self) -> impl Iterator<Item = &Key> {
        self.0.iter_key()
    }
}

impl<Key: Clone + Ord + Debug + 'static, Value: Clone + Debug> Default for OrdTreeMap<Key, Value> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Key, Value> Default for OrdTrait<Key, Value> {
    fn default() -> Self {
        Self {
            _phantom: Default::default(),
        }
    }
}

impl<Key: Clone + Ord + Debug + 'static, Value: Clone + Debug> BTreeTrait for OrdTrait<Key, Value> {
    type Elem = (Key, Value);

    type Cache = Option<(Key, Key)>;

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
        Some((
            elements[0].0.clone(),
            elements[elements.len() - 1].0.clone(),
        ))
    }
}

impl<Key: Ord + Clone + Debug + 'static, Value> Query for OrdTrait<Key, Value> {
    type Cache = Option<(Key, Key)>;
    type Elem = (Key, Value);
    type QueryArg = Key;

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

    fn find_element(&mut self, target: &Key, elements: &[Self::Elem]) -> crate::FindResult {
        match elements.binary_search_by_key(&target, |x| &x.0) {
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
