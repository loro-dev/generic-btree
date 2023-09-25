use core::fmt::Debug;
use std::cmp::Ordering;
use std::ops::RangeBounds;

use crate::rle::{HasLength, Mergeable, Sliceable};
use crate::{BTree, BTreeTrait, FindResult, MoveEvent, MoveListener, Query};

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
        let Some(result) = self.tree.query::<OrdTrait<Key, Value>>(&key) else {
            self.len += 1;
            self.tree.push(Unmergeable((key, value)));
            return;
        };

        if !result.found {
            self.len += 1;
            let tree = &mut self.tree;
            let data = Unmergeable((key, value));
            let index = result.leaf;
            let leaf = tree.leaf_nodes.get_mut(index.0).unwrap();
            let parent = leaf.parent();

            let mut is_full = false;
            // Try to merge
            if result.offset == 0 && data.can_merge(&leaf.elem) {
                leaf.elem.merge_left(&data);
                let leaf1 = Some(index);
                if let Some(listener) = tree.element_move_listener.as_ref() {
                    listener(MoveEvent {
                        target_leaf: leaf1,
                        elem: &data,
                    });
                }
            } else if result.offset == leaf.elem.rle_len() && leaf.elem.can_merge(&data) {
                leaf.elem.merge_right(&data);
                let leaf1 = Some(index);
                if let Some(listener) = tree.element_move_listener.as_ref() {
                    listener(MoveEvent {
                        target_leaf: leaf1,
                        elem: &data,
                    });
                }
            } else {
                // Insert new leaf node
                let child = tree.alloc_leaf_child(data, parent.unwrap_internal());
                let (_, parent_index, insert_index) = tree.split_leaf_if_needed(result);
                let parent = tree.in_nodes.get_mut(parent_index).unwrap();
                parent.children.insert(insert_index, child).unwrap();
                is_full = parent.is_full();
            }

            tree.recursive_update_cache(parent, true, None);
            if is_full {
                tree.split(parent);
            }
        } else {
            let leaf = self.tree.get_elem_mut(result.leaf).unwrap();
            leaf.0 .1 = value;
        }
    }

    #[inline(always)]
    pub fn delete(&mut self, key: &Key) -> Option<(Key, Value)> {
        let Some(q) = self.tree.query::<OrdTrait<Key, Value>>(key) else {
            return None;
        };
        match self.tree.remove_leaf(q) {
            Some(v) => {
                self.len -= 1;
                Some(v.0)
            }
            None => None,
        }
    }

    pub fn set_listener(&mut self, listener: Option<MoveListener<Unmergeable<(Key, Value)>>>) {
        self.tree.set_listener(listener);
    }

    #[inline(always)]
    pub fn iter(&self) -> impl Iterator<Item = &(Key, Value)> {
        self.tree.iter().map(|x| &x.0)
    }

    #[inline(always)]
    pub fn iter_key(&self) -> impl Iterator<Item = &Key> {
        self.tree.iter().map(|x| &x.0 .0)
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[allow(unused)]
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

    #[allow(unused)]
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

#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct Unmergeable<T>(T);

impl<T> HasLength for Unmergeable<T> {
    fn rle_len(&self) -> usize {
        1
    }
}

impl<T: Clone> Sliceable for Unmergeable<T> {
    fn slice(&self, range: impl RangeBounds<usize>) -> Self {
        let start = match range.start_bound() {
            std::ops::Bound::Included(x) => *x,
            std::ops::Bound::Excluded(x) => x + 1,
            std::ops::Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            std::ops::Bound::Included(x) => x + 1,
            std::ops::Bound::Excluded(x) => *x,
            std::ops::Bound::Unbounded => 1,
        };

        if end - start != 1 {
            panic!("Invalid range");
        }

        self.clone()
    }
}

impl<T> Mergeable for Unmergeable<T> {
    fn can_merge(&self, _rhs: &Self) -> bool {
        false
    }

    fn merge_right(&mut self, _rhs: &Self) {
        unreachable!()
    }

    fn merge_left(&mut self, _left: &Self) {
        unreachable!()
    }
}

impl<Key: Clone + Ord + Debug + 'static, Value: Clone + Debug> BTreeTrait for OrdTrait<Key, Value> {
    type Elem = Unmergeable<(Key, Value)>;
    type Cache = Option<(Key, Key)>;

    #[inline(always)]
    fn calc_cache_internal(
        cache: &mut Self::Cache,
        caches: &[crate::Child<Self>],
        _: Option<()>,
    ) -> Option<()> {
        if caches.is_empty() {
            return None;
        }

        *cache = Some((
            caches[0].cache.as_ref().unwrap().0.clone(),
            caches[caches.len() - 1].cache.as_ref().unwrap().1.clone(),
        ));
        None
    }

    type CacheDiff = ();

    #[inline(always)]
    fn merge_cache_diff(_: &mut Self::CacheDiff, _: &Self::CacheDiff) {}

    #[inline(always)]
    fn get_elem_cache(elem: &Self::Elem) -> Self::Cache {
        Some((elem.0 .0.clone(), elem.0 .0.clone()))
    }
}

impl<Key: Ord + Clone + Debug + 'static, Value: Clone + Debug + 'static> Query<OrdTrait<Key, Value>>
    for OrdTrait<Key, Value>
{
    type QueryArg = Key;

    #[inline(always)]
    fn init(_target: &Self::QueryArg) -> Self {
        Self::default()
    }

    #[inline]
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
            Err(i) => FindResult::new_missing(
                i.min(child_caches.len() - 1),
                if i == child_caches.len() { 1 } else { 0 },
            ),
        }
    }

    #[inline(always)]
    fn confirm_elem(
        &self,
        q: &Self::QueryArg,
        elem: &<OrdTrait<Key, Value> as BTreeTrait>::Elem,
    ) -> (usize, bool) {
        match q.cmp(&elem.0 .0) {
            Ordering::Less => (0, false),
            Ordering::Equal => (0, true),
            Ordering::Greater => (1, false),
        }
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
            let a = tree.0.tree.query::<OrdTrait<u64, ()>>(&i).unwrap();
            assert_eq!(tree.0.tree.compare_pos(a, a), Ordering::Equal);
            for j in i + 1..100 {
                let b = tree.0.tree.query::<OrdTrait<u64, ()>>(&j).unwrap();
                assert_eq!(tree.0.tree.compare_pos(a, b), Ordering::Less);
                assert_eq!(tree.0.tree.compare_pos(b, a), Ordering::Greater);
            }
        }
    }

    mod move_event_test {
        use std::{
            collections::HashMap,
            sync::{Arc, Mutex},
        };

        use crate::LeafIndex;

        use super::*;

        #[test]
        fn test() {
            let mut tree: OrdTreeMap<u64, usize> = OrdTreeMap::new();
            let record: Arc<Mutex<HashMap<u64, LeafIndex>>> = Default::default();
            let mut rng = rand::rngs::StdRng::seed_from_u64(123);
            let mut data: HeapVec<u64> = (0..1000).map(|_| rng.gen()).collect();
            let record_clone = record.clone();
            tree.set_listener(Some(Box::new(move |event| {
                if let Some(leaf) = event.target_leaf {
                    let mut record = record.lock().unwrap();
                    record.insert(event.elem.0 .0, leaf);
                } else {
                    let mut record = record.lock().unwrap();
                    record.remove(&event.elem.0 .0);
                }
            })));
            for &value in data.iter() {
                tree.insert(value, 0);
            }
            {
                let record = record_clone.lock().unwrap();
                assert_eq!(record.len(), 1000);
                for &value in data.iter() {
                    let index = record.get(&value).unwrap();
                    let node = tree.tree.get_elem(*index).unwrap();
                    assert_eq!(node.0 .0, value);
                }
            }
            for value in data.drain(0..100) {
                tree.delete(&value);
            }
            {
                let record = record_clone.lock().unwrap();
                assert_eq!(record.len(), 900);
                assert_eq!(tree.len, 900);
                for &value in data.iter() {
                    let index = record.get(&value).unwrap();
                    let node = tree.tree.get_elem(*index).unwrap();
                    assert_eq!(node.0 .0, value);
                }
            }
            for value in data.drain(0..800) {
                tree.delete(&value);
            }
            {
                let record = record_clone.lock().unwrap();
                assert_eq!(record.len(), 100);
                assert_eq!(tree.len, 100);
                for &value in data.iter() {
                    let index = record.get(&value).unwrap();
                    let node = tree.tree.get_elem(*index).unwrap();
                    assert_eq!(node.0 .0, value);
                }
            }
            tree.tree.check();
            for i in (0..100).rev() {
                tree.delete(&data.pop().unwrap());
                {
                    let record = record_clone.lock().unwrap();
                    assert_eq!(record.len(), i);
                    assert_eq!(tree.len, i);
                    for &value in data.iter() {
                        let index = record.get(&value).unwrap();
                        let node = tree.tree.get_elem(*index).unwrap();
                        assert_eq!(node.0 .0, value);
                    }
                }
            }
        }
    }
}
