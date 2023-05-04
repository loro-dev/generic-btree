use core::fmt::Debug;

use crate::{BTree, BTreeTrait, FindResult, MoveListener, Query};

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
            let leaf = self.tree.nodes.get_mut(result.leaf).unwrap();
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

    pub fn set_listener(&mut self, listener: Option<MoveListener<(Key, Value)>>) {
        self.tree.set_listener(listener);
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
    type Cache = Option<(Key, Key)>;

    const MAX_LEN: usize = 32;

    #[inline(always)]
    fn calc_cache_internal(
        cache: &mut Self::Cache,
        caches: &[crate::Child<Self>],
        _: Option<()>,
    ) -> Option<()> {
        *cache = Some((
            caches[0].cache.as_ref().unwrap().0.clone(),
            caches[caches.len() - 1].cache.as_ref().unwrap().1.clone(),
        ));
        None
    }

    #[inline(always)]
    fn calc_cache_leaf(cache: &mut Self::Cache, elements: &[Self::Elem], diff: Option<()>) {
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
    use std::cmp::Ordering;

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

    #[test]
    fn test_compare_pos() {
        let mut tree: OrdTreeSet<u64> = OrdTreeSet::new();
        for i in 0..100 {
            tree.insert(i);
        }
        for i in 0..99 {
            let a = tree.0.tree.query::<OrdTrait<u64, ()>>(&i);
            assert_eq!(tree.0.tree.compare_pos(a, a), Ordering::Equal);
            for j in i + 1..100 {
                let b = tree.0.tree.query::<OrdTrait<u64, ()>>(&j);
                assert_eq!(tree.0.tree.compare_pos(a, b), Ordering::Less);
                assert_eq!(tree.0.tree.compare_pos(b, a), Ordering::Greater);
            }
        }
    }

    mod move_event_test {
        use std::{cell::RefCell, collections::HashMap, rc::Rc};

        use thunderdome::Index as ArenaIndex;

        use super::*;
        #[test]
        fn test() {
            let mut tree: OrdTreeMap<u64, usize> = OrdTreeMap::new();
            let record: Rc<RefCell<HashMap<u64, ArenaIndex>>> = Default::default();
            let mut rng = rand::rngs::StdRng::seed_from_u64(123);
            let mut data: HeapVec<u64> = (0..1000).map(|_| rng.gen()).collect();
            let record_clone = record.clone();
            tree.set_listener(Some(Box::new(move |event| {
                if let Some(leaf) = event.target_leaf {
                    let mut record = record.borrow_mut();
                    record.insert(event.elem.0, leaf);
                } else {
                    let mut record = record.borrow_mut();
                    record.remove(&event.elem.0);
                }
            })));
            for &value in data.iter() {
                tree.insert(value, 0);
            }
            {
                let record = record_clone.borrow();
                assert_eq!(record.len(), 1000);
                for &value in data.iter() {
                    let index = record.get(&value).unwrap();
                    let node = tree.tree.get_node(*index);
                    assert!(node.elements.iter().any(|x| x.0 == value));
                }
            }
            for value in data.drain(0..100) {
                tree.delete(&value);
            }
            {
                let record = record_clone.borrow();
                assert_eq!(record.len(), 900);
                assert_eq!(tree.len, 900);
                for &value in data.iter() {
                    let index = record.get(&value).unwrap();
                    let node = tree.tree.get_node(*index);
                    assert!(node.elements.iter().any(|x| x.0 == value));
                }
            }
            for value in data.drain(0..800) {
                tree.delete(&value);
            }
            {
                let record = record_clone.borrow();
                assert_eq!(record.len(), 100);
                assert_eq!(tree.len, 100);
                for &value in data.iter() {
                    let index = record.get(&value).unwrap();
                    let node = tree.tree.get_node(*index);
                    assert!(node.elements.iter().any(|x| x.0 == value));
                }
            }
            tree.tree.check();
            for i in (0..100).rev() {
                tree.delete(&data.pop().unwrap());
                {
                    let record = record_clone.borrow();
                    assert_eq!(record.len(), i);
                    assert_eq!(tree.len, i);
                    for &value in data.iter() {
                        let index = record.get(&value).unwrap();
                        let node = tree.tree.get_node(*index);
                        assert!(node.elements.iter().any(|x| x.0 == value));
                    }
                }
            }
        }
    }
}
