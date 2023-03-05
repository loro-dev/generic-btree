use core::fmt::Debug;

use crate::{BTree, BTreeTrait, FindResult, Query};

#[derive(Debug)]
#[repr(transparent)]
struct OrdTrait<Key, Value> {
    _phantom: core::marker::PhantomData<(Key, Value)>,
}

#[derive(Debug)]
pub struct OrdTreeMap<Key: Clone + Ord + Debug + 'static, Value: Clone + Debug> {
    tree: BTree<OrdTrait<Key, Value>>,
    len: usize,
}

#[derive(Debug)]
pub struct OrdTreeSet<Key: Clone + Ord + Debug + 'static>(OrdTreeMap<Key, ()>);

impl<Key: Clone + Ord + Debug + 'static, Value: Clone + Debug + 'static> OrdTreeMap<Key, Value> {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            tree: BTree::new(),
            len: 0,
        }
    }

    #[inline(always)]
    pub fn insert(&mut self, key: Key, value: Value) {
        let result = self.tree.query::<OrdTrait<Key, Value>>(&key);
        if !result.found {
            self.len += 1;
            self.tree.insert_by_query_result(result, (key, value));
        } else {
            let leaf = self
                .tree
                .nodes
                .get_mut(result.node_path.last().unwrap().arena)
                .unwrap();
            leaf.elements[result.elem_index].1 = value;
        }
    }

    #[inline(always)]
    pub fn delete(&mut self, value: &Key) -> Option<(Key, Value)> {
        match self.tree.delete::<OrdTrait<Key, Value>>(value) {
            Some(v) => {
                self.len -= 1;
                Some(v)
            }
            None => None,
        }
    }

    #[inline(always)]
    pub fn iter(&self) -> impl Iterator<Item = &(Key, Value)> {
        self.tree.iter()
    }

    #[inline(always)]
    pub fn iter_key(&self) -> impl Iterator<Item = &Key> {
        self.tree.iter().map(|x| &x.0)
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub(crate) fn check(&self) {
        self.tree.check()
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
        self.0.delete(key).is_some()
    }

    #[inline(always)]
    pub fn iter(&self) -> impl Iterator<Item = &Key> {
        self.0.iter_key()
    }

    pub fn len(&self) -> usize {
        self.0.len
    }

    pub fn is_empty(&self) -> bool {
        self.0.len == 0
    }

    fn check(&self) {
        self.0.check()
    }
}

impl<Key: Clone + Ord + Debug + 'static> Default for OrdTreeSet<Key> {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl<Key: Clone + Ord + Debug + 'static, Value: Clone + Debug + 'static> Default
    for OrdTreeMap<Key, Value>
{
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl<Key, Value> Default for OrdTrait<Key, Value> {
    #[inline(always)]
    fn default() -> Self {
        Self {
            _phantom: Default::default(),
        }
    }
}

impl<Key: Clone + Ord + Debug + 'static, Value: Clone + Debug> BTreeTrait for OrdTrait<Key, Value> {
    type Elem = (Key, Value);
    type WriteBuffer = ();
    type Cache = Option<(Key, Key)>;

    const MAX_LEN: usize = 32;

    #[inline(always)]
    fn element_to_cache(_: &Self::Elem) -> Self::Cache {
        None
    }

    #[inline(always)]
    fn calc_cache_internal(cache: &mut Self::Cache, caches: &[crate::Child<Self>], _: Option<()>) {
        *cache = Some((
            caches[0].cache.as_ref().unwrap().0.clone(),
            caches[caches.len() - 1].cache.as_ref().unwrap().1.clone(),
        ));
    }

    #[inline(always)]
    fn calc_cache_leaf(cache: &mut Self::Cache, elements: &[Self::Elem]) {
        if elements.is_empty() {
            return;
        }

        *cache = Some((
            elements[0].0.clone(),
            elements[elements.len() - 1].0.clone(),
        ))
    }

    type CacheDiff = ();

    #[inline(always)]
    fn merge_cache_diff(_: &mut Self::CacheDiff, _: &Self::CacheDiff) {}
}

impl<Key: Ord + Clone + Debug + 'static, Value: Clone + Debug + 'static> Query<OrdTrait<Key, Value>>
    for OrdTrait<Key, Value>
{
    type QueryArg = Key;

    fn find_node(
        &mut self,
        target: &Self::QueryArg,
        child_caches: &[crate::Child<OrdTrait<Key, Value>>],
    ) -> crate::FindResult {
        match child_caches.binary_search_by(|x| {
            let (min, max) = x.cache.as_ref().unwrap();
            if target < min {
                core::cmp::Ordering::Greater
            } else if target > max {
                core::cmp::Ordering::Less
            } else {
                core::cmp::Ordering::Equal
            }
        }) {
            Ok(i) => FindResult::new_found(i, 0),
            Err(i) => FindResult::new_missing(i, 0),
        }
    }

    fn find_element(&mut self, target: &Key, elements: &[(Key, Value)]) -> crate::FindResult {
        match elements.binary_search_by_key(&target, |x| &x.0) {
            Ok(i) => FindResult::new_found(i, 0),
            Err(i) => FindResult::new_missing(i, 0),
        }
    }

    #[inline(always)]
    fn init(_target: &Self::QueryArg) -> Self {
        Self::default()
    }
}

#[cfg(test)]
mod test {
    use rand::{Rng, SeedableRng};

    use crate::HeapVec;

    use super::*;

    #[test]
    fn test() {
        let mut tree: OrdTreeSet<u64> = OrdTreeSet::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        let mut data: HeapVec<u64> = (0..1000).map(|_| rng.gen()).collect();
        for &value in data.iter() {
            tree.insert(value);
        }
        data.sort_unstable();
        assert_eq!(tree.iter().copied().collect::<HeapVec<_>>(), data);
        tree.check();
    }

    #[test]
    fn test_delete() {
        let mut tree: OrdTreeSet<u64> = OrdTreeSet::new();
        tree.insert(12);
        tree.delete(&12);
        assert_eq!(tree.len(), 0);
    }
}
