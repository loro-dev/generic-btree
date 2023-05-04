//! # generic-btree
//!
//! It's a safe btree crate for versatile purposes:
//!
//! - Rope
//! - BTreeMap / BTreeSet
//! - Run-length-encoding insert/delete
//! - Range map that uses range as key
//!
//! ## Write buffer
//!
//! This crate provides a write buffer that can be used to store updates on ancestor nodes.
//! For example, we may need to update a range of elements in B-Tree together. Normally, it
//! would take O(n) time to update each element, where n is the number of elements.
//! With write buffer, we can update all elements within O(log n). And the actual updates are
//! done when we need to iterate or query the related elements.
//!
//!

#![forbid(unsafe_code)]

use core::{
    fmt::Debug,
    iter::Map,
    ops::{Deref, Range},
};
use std::{cmp::Ordering, mem::take, ops::RangeBounds};

use fxhash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use thunderdome::Arena;
pub use thunderdome::Index as ArenaIndex;
mod event;
mod generic_impl;
mod iter;
pub use event::{MoveEvent, MoveListener};
pub use generic_impl::*;

use crate::rle::HasLength;
pub mod rle;
pub type SmallElemVec<T> = SmallVec<[T; 8]>;
pub type StackVec<T> = SmallVec<[T; 8]>;
pub type HeapVec<T> = Vec<T>;

pub trait BTreeTrait {
    type Elem: Debug;
    type Cache: Debug + Default + Clone + Eq;
    type CacheDiff: Debug;
    /// Use () if you don't need write buffer.
    /// Associated type default is still unstable so we don't provide default value.
    const MAX_LEN: usize;

    #[allow(unused)]
    #[inline(always)]
    fn insert(elements: &mut HeapVec<Self::Elem>, index: usize, offset: usize, elem: Self::Elem) {
        elements.insert(index, elem);
    }

    #[allow(unused)]
    #[inline(always)]
    fn insert_batch(
        elements: &mut HeapVec<Self::Elem>,
        index: usize,
        offset: usize,
        new_elements: impl IntoIterator<Item = Self::Elem>,
    ) where
        Self::Elem: Clone,
    {
        unimplemented!()
    }

    /// If diff.is_some, return value should be some too
    fn calc_cache_internal(
        cache: &mut Self::Cache,
        caches: &[Child<Self>],
        diff: Option<Self::CacheDiff>,
    ) -> Option<Self::CacheDiff>;
    fn calc_cache_leaf(
        cache: &mut Self::Cache,
        elements: &[Self::Elem],
        diff: Option<Self::CacheDiff>,
    ) -> Self::CacheDiff;
    fn merge_cache_diff(diff1: &mut Self::CacheDiff, diff2: &Self::CacheDiff);
}

pub trait Query<B: BTreeTrait> {
    type QueryArg: Clone;

    fn init(target: &Self::QueryArg) -> Self;

    fn find_node(&mut self, target: &Self::QueryArg, child_caches: &[Child<B>]) -> FindResult;

    fn find_element(&mut self, target: &Self::QueryArg, elements: &[B::Elem]) -> FindResult;

    #[allow(unused)]
    #[inline(always)]
    fn delete(
        elements: &mut HeapVec<B::Elem>,
        query: &Self::QueryArg,
        elem_index: usize,
        offset: usize,
    ) -> Option<B::Elem> {
        if elem_index >= elements.len() {
            return None;
        }

        Some(elements.remove(elem_index))
    }

    #[allow(unused)]
    #[inline(always)]
    fn drain_range<'a, 'b>(
        elements: &'a mut HeapVec<B::Elem>,
        start_query: &'b Self::QueryArg,
        end_query: &'b Self::QueryArg,
        start: Option<QueryResult>,
        end: Option<QueryResult>,
    ) -> Box<dyn Iterator<Item = B::Elem> + 'a> {
        Box::new(match (start, end) {
            (None, None) => elements.drain(..),
            (None, Some(to)) => elements.drain(..to.elem_index),
            (Some(from), None) => elements.drain(from.elem_index..),
            (Some(from), Some(to)) => elements.drain(from.elem_index..to.elem_index),
        })
    }

    #[allow(unused)]
    #[inline(always)]
    fn delete_range(
        elements: &mut HeapVec<B::Elem>,
        start_query: &Self::QueryArg,
        end_query: &Self::QueryArg,
        start: Option<QueryResult>,
        end: Option<QueryResult>,
    ) {
        Self::drain_range(elements, start_query, end_query, start, end);
    }
}

pub struct BTree<B: BTreeTrait> {
    nodes: Arena<Node<B>>,
    root: ArenaIndex,
    root_cache: B::Cache,
    /// this field turn true when things written into write buffer
    need_flush: bool,
    element_move_listener: Option<MoveListener<B::Elem>>,
}

impl<Elem: Clone, B: BTreeTrait<Elem = Elem>> Clone for BTree<B> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            nodes: self.nodes.clone(),
            root: self.root,
            root_cache: self.root_cache.clone(),
            need_flush: false,
            element_move_listener: None,
        }
    }
}

pub struct FindResult {
    pub index: usize,
    pub offset: usize,
    pub found: bool,
}

impl FindResult {
    pub fn new_found(index: usize, offset: usize) -> Self {
        Self {
            index,
            offset,
            found: true,
        }
    }

    pub fn new_missing(index: usize, offset: usize) -> Self {
        Self {
            index,
            offset,
            found: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Idx {
    pub arena: ArenaIndex,
    pub arr: usize,
}

impl Idx {
    #[inline(always)]
    pub fn new(arena: ArenaIndex, arr: usize) -> Self {
        Self { arena, arr }
    }
}

type NodePath = SmallVec<[Idx; 8]>;

struct PathRef<'a>(&'a [Idx]);

impl<'a> Deref for PathRef<'a> {
    type Target = [Idx];

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl<'a> From<&'a StackVec<Idx>> for PathRef<'a> {
    fn from(value: &'a StackVec<Idx>) -> Self {
        Self(value)
    }
}

impl<'a> From<&'a [Idx]> for PathRef<'a> {
    fn from(value: &'a [Idx]) -> Self {
        Self(value)
    }
}

impl<'a> PathRef<'a> {
    pub fn this(&self) -> Idx {
        *self.last().unwrap()
    }

    pub fn parent(&self) -> Option<Idx> {
        if self.len() >= 2 {
            self.get(self.len() - 2).copied()
        } else {
            None
        }
    }

    pub fn parent_path(&self) -> PathRef<'a> {
        Self(&self.0[..self.len() - 1])
    }

    pub fn set_as_parent_path(&mut self) {
        debug_assert!(self.len() > 1);
        self.0 = &self.0[..self.len() - 1];
    }

    pub fn is_root(&self) -> bool {
        self.len() == 1
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub struct QueryResult {
    pub leaf: ArenaIndex,
    pub elem_index: usize,
    pub offset: usize,
    pub found: bool,
}

/// A slice of elements in a leaf node of BTree.
///
/// - `start` is Some((start_index, start_offset)) when the slice is the first slice of the given range. i.e. the first element should be sliced.
/// - `end`   is Some((end_index, end_offset))     when the slice is the last  slice of the given range. i.e. the last  element should be sliced.
#[derive(Debug)]
pub struct MutElemArrSlice<'a, Elem> {
    pub elements: &'a mut HeapVec<Elem>,
    /// start is Some((start_index, start_offset)) when the slice is the first slice of the given range. i.e. the first element should be sliced.
    pub start: Option<(usize, usize)>,
    /// end is Some((end_index, end_offset))     when the slice is the last  slice of the given range. i.e. the last  element should be sliced.
    pub end: Option<(usize, usize)>,
}

/// A slice of element
///
/// - `start` is Some(start_offset) when it's first element of the given range.
/// - `end` is Some(end_offset) when it's last element of the given range.
#[derive(Debug)]
pub struct ElemSlice<'a, Elem> {
    path: QueryResult,
    pub elem: &'a Elem,
    pub start: Option<usize>,
    pub end: Option<usize>,
}

impl<'a, Elem> ElemSlice<'a, Elem> {
    pub fn path(&self) -> &QueryResult {
        &self.path
    }
}

impl QueryResult {
    pub fn elem<'b, Elem: Debug, B: BTreeTrait<Elem = Elem>>(
        &self,
        tree: &'b BTree<B>,
    ) -> Option<&'b Elem> {
        tree.nodes
            .get(self.leaf)
            .and_then(|x| x.elements.get(self.elem_index))
    }
}

// TODO: use enum to save spaces
pub struct Node<B: BTreeTrait> {
    parent: Option<ArenaIndex>,
    parent_slot: u32,
    elements: HeapVec<B::Elem>,
    children: HeapVec<Child<B>>,
}

impl<Cache: Debug, Elem: Debug, B: BTreeTrait<Elem = Elem, Cache = Cache>> Debug for BTree<B> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        fn fmt_node<Cache: Debug, Elem: Debug, B: BTreeTrait<Elem = Elem>>(
            tree: &BTree<B>,
            node: &Node<B>,
            f: &mut core::fmt::Formatter<'_>,
            indent_size: usize,
        ) -> core::fmt::Result {
            if node.is_internal() {
                for child in node.children.iter() {
                    indent(f, indent_size)?;
                    let child_node = tree.get_node(child.arena);
                    f.write_fmt(format_args!(
                        "{} Arena({:?}) Cache: {:?}\n",
                        child_node.parent_slot, &child.arena, &child.cache
                    ))?;
                    fmt_node::<Cache, Elem, B>(tree, child_node, f, indent_size + 1)?;
                }
            } else {
                if node.elements.is_empty() {
                    indent(f, indent_size)?;
                    f.write_fmt(format_args!("EMPTY\n"))?;
                }
                for elem in node.elements.iter() {
                    indent(f, indent_size)?;
                    f.write_fmt(format_args!("Elem: {:?}\n", elem))?;
                }
            }

            Ok(())
        }

        fn indent(f: &mut core::fmt::Formatter<'_>, indent: usize) -> core::fmt::Result {
            for _ in 0..indent {
                f.write_str("    ")?;
            }
            Ok(())
        }

        f.write_str("BTree\n")?;
        indent(f, 1)?;
        f.write_fmt(format_args!(
            "Root Arena({:?}) Cache: {:?}\n",
            &self.root, &self.root_cache
        ))?;
        fmt_node::<Cache, Elem, B>(self, self.nodes.get(self.root).unwrap(), f, 1)
    }
}

impl<Cache: Debug, Elem: Debug, B: BTreeTrait<Elem = Elem, Cache = Cache>> Debug for Node<B> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Node")
            .field("elements", &self.elements)
            .field("children", &self.children)
            .finish()
    }
}

impl<Cache: Debug, Elem: Debug, B: BTreeTrait<Elem = Elem, Cache = Cache>> Debug for Child<B> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Child")
            .field("index", &self.arena)
            .field("cache", &self.cache)
            .finish()
    }
}
impl<Elem: Clone, B: BTreeTrait<Elem = Elem>> Clone for Node<B> {
    fn clone(&self) -> Self {
        Self {
            parent: self.parent,
            parent_slot: u32::MAX,
            elements: self.elements.clone(),
            children: self.children.clone(),
        }
    }
}

pub struct Child<B: ?Sized + BTreeTrait> {
    arena: ArenaIndex,
    pub cache: B::Cache,
}

impl<B: BTreeTrait> Clone for Child<B> {
    fn clone(&self) -> Self {
        Self {
            arena: self.arena,
            cache: self.cache.clone(),
        }
    }
}

impl<B: BTreeTrait> Child<B> {
    #[inline(always)]
    pub fn cache(&self) -> &B::Cache {
        &self.cache
    }

    fn new(arena: ArenaIndex, cache: B::Cache) -> Self {
        Self { arena, cache }
    }
}

impl<B: BTreeTrait> Default for Node<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: BTreeTrait> Node<B> {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            parent: None,
            parent_slot: u32::MAX,
            elements: HeapVec::with_capacity(B::MAX_LEN),
            children: HeapVec::with_capacity(B::MAX_LEN),
        }
    }

    #[inline(always)]
    pub fn is_full(&self) -> bool {
        if self.is_internal() {
            self.children.len() >= B::MAX_LEN
        } else {
            self.elements.len() >= B::MAX_LEN
        }
    }

    #[inline(always)]
    pub fn is_lack(&self) -> bool {
        if self.is_internal() {
            self.children.len() < B::MAX_LEN / 2
        } else {
            self.elements.len() < B::MAX_LEN / 2
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        if self.is_internal() {
            self.children.len()
        } else {
            self.elements.len()
        }
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline(always)]
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    #[inline(always)]
    pub fn is_internal(&self) -> bool {
        !self.children.is_empty()
    }

    /// if diff is not provided, the cache will be calculated from scratch
    #[inline(always)]
    fn calc_cache(&self, cache: &mut B::Cache, diff: Option<B::CacheDiff>) -> Option<B::CacheDiff> {
        if self.is_internal() {
            B::calc_cache_internal(cache, &self.children, diff)
        } else {
            Some(B::calc_cache_leaf(cache, &self.elements, diff))
        }
    }

    pub fn elements(&self) -> &[B::Elem] {
        &self.elements
    }
}

type LeafDirtyMap<Diff> = FxHashMap<ArenaIndex, Option<Diff>>;

struct LackInfo {
    is_parent_lack: bool,
}

impl<B: BTreeTrait> BTree<B> {
    #[inline]
    pub fn new() -> Self {
        let mut arena = Arena::new();
        let root = arena.insert(Node::new());
        Self {
            nodes: arena,
            root,
            root_cache: B::Cache::default(),
            need_flush: false,
            element_move_listener: None,
        }
    }

    /// Register/Unregister an element move event listener.
    ///
    /// This is called when a element is moved from one leaf node to another.
    /// It could happen when:
    ///
    /// - Leaf node split
    /// - Leaf nodes merge
    /// - Elements moving from one leaf node to another to keep the balance of the BTree
    ///
    /// It's useful when you try to track an element's position in the BTree.
    #[inline]
    pub fn set_listener(&mut self, listener: Option<MoveListener<B::Elem>>) {
        self.element_move_listener = listener;
    }

    #[inline]
    pub fn node_len(&self) -> usize {
        self.nodes.len()
    }

    #[inline]
    pub fn insert<Q>(&mut self, tree_index: &Q::QueryArg, data: B::Elem)
    where
        Q: Query<B>,
    {
        let result = self.query::<Q>(tree_index);
        self.insert_by_query_result(result, data)
    }

    /// It will invoke [`BTreeTrait::insert`]
    #[inline]
    pub fn insert_by_query_result(&mut self, result: QueryResult, data: B::Elem) {
        let index = result.leaf;
        self.notify_elem_move(index, &data);
        let node = self.nodes.get_mut(index).unwrap();
        B::insert(&mut node.elements, result.elem_index, result.offset, data);
        let is_full = node.is_full();
        self.recursive_update_cache(index, true, None);
        if is_full {
            self.split(result.leaf);
        }
    }

    /// Insert many elements into the tree at once
    ///
    /// It will invoke [`BTreeTrait::insert_batch`]
    ///
    /// NOTE: Currently this method don't guarantee after inserting many elements the tree is
    /// still balance
    pub fn insert_many_by_query_result(
        &mut self,
        result: &QueryResult,
        data: impl IntoIterator<Item = B::Elem>,
    ) where
        B::Elem: Clone,
    {
        let index = result.leaf;
        let node = self.nodes.get_mut(index).unwrap();
        B::insert_batch(&mut node.elements, result.elem_index, result.offset, data);

        let is_full = node.is_full();
        self.recursive_update_cache(result.leaf, true, None);
        if is_full {
            self.split(result.leaf);
        }
        // TODO: tree may still be unbalanced
    }

    pub fn batch_insert_by_query_result(
        &mut self,
        result: &QueryResult,
        data: impl IntoIterator<Item = B::Elem>,
    ) where
        B::Elem: Clone,
    {
        let index = result.leaf;
        let node = self.nodes.get_mut(index).unwrap();
        if result.found {
            B::insert_batch(
                &mut node.elements,
                result.elem_index,
                result.offset,
                data.into_iter(),
            );
        } else {
            node.elements
                .splice(result.elem_index..result.elem_index, data);
        }

        let is_full = node.is_full();
        self.recursive_update_cache(result.leaf, true, None);
        if is_full {
            self.split(result.leaf);
        }
    }

    pub fn delete<Q>(&mut self, query: &Q::QueryArg) -> Option<B::Elem>
    where
        Q: Query<B>,
    {
        let result = self.query::<Q>(query);
        if !result.found {
            return None;
        }

        let index = result.leaf;
        let node = self.nodes.get_mut(index).unwrap();
        let mut ans = None;
        if result.found {
            ans = Q::delete(&mut node.elements, query, result.elem_index, result.offset);
            if let Some(ans) = &ans {
                if let Some(listener) = self.element_move_listener.as_ref() {
                    listener(MoveEvent::new_del(ans));
                }
            }
        }

        let is_full = node.is_full();
        let is_lack = node.is_lack();
        self.recursive_update_cache(result.leaf, true, None);
        if is_full {
            self.split(result.leaf);
        } else if is_lack {
            let path = self.get_path(index);
            let mut path_ref: PathRef = path.as_ref().into();
            while !path_ref.is_root()
                && self.get_node(path_ref.this().arena).is_lack()
                && self.handle_lack(path_ref.this().arena).is_parent_lack
            {
                path_ref.set_as_parent_path();
            }

            self.try_reduce_levels()
        }
        ans
    }

    #[inline(always)]
    pub fn query<Q>(&self, query: &Q::QueryArg) -> QueryResult
    where
        Q: Query<B>,
    {
        self.query_with_finder_return::<Q>(query).0
    }

    /// Shift by offset 1.
    ///
    /// It will not stay on empty spans but scan forward
    pub fn shift_path_by_one_offset(&self, mut path: QueryResult) -> Option<QueryResult>
    where
        B::Elem: rle::HasLength,
    {
        let mut node = self.nodes.get(path.leaf).unwrap();
        path.offset += 1;
        loop {
            if path.elem_index == node.elements.len() {
                node = self.nodes.get(path.leaf).unwrap();
                if let Some(next) = self.next_same_level_node(path.leaf) {
                    path.elem_index = 0;
                    path.offset = 0;
                    path.leaf = next;
                } else {
                    return None;
                }
            }

            assert!(node.is_leaf() && path.elem_index <= node.elements.len());
            let elem = &node.elements[path.elem_index];
            // skip empty span
            if elem.rle_len() <= path.offset {
                path.offset -= elem.rle_len();
                path.elem_index += 1;
            } else {
                break;
            }
        }

        Some(path)
    }

    pub fn query_with_finder_return<Q>(&self, query: &Q::QueryArg) -> (QueryResult, Q)
    where
        Q: Query<B>,
    {
        let mut finder = Q::init(query);
        let mut node = self.nodes.get(self.root).unwrap();
        let mut index = self.root;
        let mut ans = QueryResult {
            leaf: index,
            elem_index: 0,
            offset: 0,
            found: true,
        };
        while node.is_internal() {
            let result = finder.find_node(query, &node.children);
            let i = result.index;
            let i = i.min(node.children.len() - 1);
            ans.found = ans.found && result.found;
            index = node.children[i].arena;
            node = self.nodes.get(index).unwrap();
            ans.leaf = index;
        }

        let result = finder.find_element(query, &node.elements);
        ans.elem_index = result.index;
        ans.found = ans.found && result.found;
        ans.offset = result.offset;
        (ans, finder)
    }

    #[inline]
    pub fn get_elem(&mut self, q: &QueryResult) -> Option<&B::Elem> {
        if !q.found {
            return None;
        }

        let path = self.get_path(q.leaf);
        let index = q.leaf;
        let node = self.nodes.get(index)?;
        node.elements.get(q.elem_index)
    }

    #[inline]
    pub fn get_elem_mut(&mut self, q: &QueryResult) -> Option<&mut B::Elem> {
        if !q.found {
            return None;
        }

        let path = self.get_path(q.leaf);
        let index = q.leaf;
        let node = self.nodes.get_mut(index)?;
        node.elements.get_mut(q.elem_index)
    }

    pub fn drain<Q>(&mut self, range: Range<Q::QueryArg>) -> iter::Drain<B, Q>
    where
        Q: Query<B>,
    {
        let from = self.query::<Q>(&range.start);
        let to = self.query::<Q>(&range.end);
        iter::Drain::new(self, range.start, range.end, from, to)
    }

    /// Update the elements in place
    ///
    /// F should returns true if the cache need to be updated
    ///
    /// This method may break the balance of the tree
    ///
    /// If the given range has zero length, f will still be called, and the slice will
    /// have same `start` and `end` field
    ///
    /// TODO: need better test coverage
    /// TODO: make range: Range<QueryResult> since it's now a Copy type
    pub fn update<F>(&mut self, range: Range<&QueryResult>, f: &mut F)
    where
        F: FnMut(MutElemArrSlice<'_, B::Elem>) -> (bool, Option<B::CacheDiff>),
    {
        let start = range.start;
        let end = range.end;
        let start_leaf = start.leaf;
        let mut path = self.get_path(start_leaf);
        let end_leaf = end.leaf;
        let mut dirty_map: LeafDirtyMap<B::CacheDiff> = FxHashMap::default();

        loop {
            let current_leaf = path.last().unwrap();
            let slice = self.get_slice(current_leaf.arena, start_leaf, start, end_leaf, end);
            let (should_update_cache, cache_diff) = f(slice);
            if should_update_cache {
                add_leaf_dirty_map(current_leaf.arena, &mut dirty_map, cache_diff);
            }

            if current_leaf.arena == end_leaf {
                break;
            }

            if !self.next_sibling(&mut path) {
                break;
            }
        }

        if !dirty_map.is_empty() {
            self.update_dirty_cache_map(dirty_map);
        } else {
            self.nodes
                .get(self.root)
                .unwrap()
                .calc_cache(&mut self.root_cache, None);
        }
    }

    /// Update the elements in place with filter to skip nodes in advance
    ///
    /// F should returns true if the cache need to be updated
    ///
    /// This method may break the balance of the tree
    ///
    /// If the given range has zero length, f will still be called, and the slice will
    /// have same `start` and `end` field
    ///
    /// TODO: need better test coverage
    /// TODO: make range: Range<QueryResult> since it's now a Copy type
    pub fn update_with_filter<F>(
        &mut self,
        range: Range<&QueryResult>,
        f: &mut F,
        filter: &dyn Fn(&B::Cache) -> bool,
    ) where
        F: FnMut(MutElemArrSlice<'_, B::Elem>) -> (bool, Option<B::CacheDiff>),
    {
        let start = range.start;
        let end = range.end;
        let start_leaf = start.leaf;
        let mut current_leaf = start_leaf;
        let end_leaf = end.leaf;
        let end_path = self.get_path(end_leaf);
        let mut dirty_map: LeafDirtyMap<B::CacheDiff> = FxHashMap::default();

        loop {
            let leaf = self.nodes.get(current_leaf).unwrap();
            let cache = leaf
                .parent
                .map(|x| &self.get_node(x).children[leaf.parent_slot as usize].cache)
                .unwrap_or(&self.root_cache);

            if !filter(cache) {
                if current_leaf == end_leaf {
                    break;
                }

                if let Some(next) =
                    self.next_same_level_node_with_filter(current_leaf, &end_path, filter)
                {
                    current_leaf = next;
                } else {
                    break;
                }

                continue;
            }

            let slice = self.get_slice(current_leaf, start_leaf, start, end_leaf, end);
            let (should_update_cache, cache_diff) = f(slice);
            if should_update_cache {
                add_leaf_dirty_map(current_leaf, &mut dirty_map, cache_diff);
            }

            if current_leaf == end_leaf {
                break;
            }

            if let Some(next) =
                self.next_same_level_node_with_filter(current_leaf, &end_path, filter)
            {
                current_leaf = next;
            } else {
                break;
            }
        }

        if !dirty_map.is_empty() {
            self.update_dirty_cache_map(dirty_map);
        } else {
            self.nodes
                .get(self.root)
                .unwrap()
                .calc_cache(&mut self.root_cache, None);
        }
    }

    /// update leaf node's elements, return true if cache need to be updated
    pub fn update_leaf(
        &mut self,
        node_idx: ArenaIndex,
        f: impl FnOnce(&mut Vec<B::Elem>) -> (bool, Option<B::CacheDiff>),
    ) {
        let node = self.nodes.get_mut(node_idx).unwrap();
        assert!(node.is_leaf());
        let (need_update_cache, diff) = f(&mut node.elements);
        let is_full = node.is_full();
        let is_lack = node.is_lack();
        if need_update_cache {
            self.recursive_update_cache(node_idx, true, diff);
        }
        if is_full {
            self.split(node_idx);
        }
        if is_lack {
            self.handle_lack(node_idx);
        }
    }

    pub fn update2_leaf(
        &mut self,
        a_idx: ArenaIndex,
        b_idx: ArenaIndex,
        mut f: impl FnMut(&mut Vec<B::Elem>, Option<ArenaIndex>) -> bool,
    ) {
        let node = self.nodes.get_mut(a_idx).unwrap();
        assert!(node.is_leaf());
        // apply a
        let need_update_cache = f(
            &mut node.elements,
            if a_idx == b_idx { None } else { Some(a_idx) },
        );
        let is_full = node.is_full();
        let is_lack = node.is_lack();
        let (b_full, b_lack) = if b_idx != a_idx {
            // apply b
            let node = self.nodes.get_mut(b_idx).unwrap();
            assert!(node.is_leaf());
            let need_update_cache = f(&mut node.elements, Some(b_idx));
            let is_full = node.is_full();
            let is_lack = node.is_lack();
            if need_update_cache {
                self.recursive_update_cache(b_idx, true, None);
            }
            (is_full, is_lack)
        } else {
            (false, false)
        };
        if need_update_cache {
            self.recursive_update_cache(a_idx, true, None);
        }
        if is_full {
            self.split(a_idx);
        }
        if b_full {
            self.split(b_idx);
        }
        if is_lack {
            self.handle_lack(a_idx);
        }

        // b may be deleted after a handle_lack
        if b_lack && self.nodes.contains(b_idx) {
            self.handle_lack(b_idx);
        }
    }

    #[inline]
    fn update_root_cache(&mut self) {
        self.nodes
            .get(self.root)
            .unwrap()
            .calc_cache(&mut self.root_cache, None);
    }

    fn get_slice(
        &mut self,
        current_leaf: ArenaIndex,
        start_leaf: ArenaIndex,
        start: &QueryResult,
        end_leaf: ArenaIndex,
        end: &QueryResult,
    ) -> MutElemArrSlice<<B as BTreeTrait>::Elem> {
        let node = self.get_mut(current_leaf);
        MutElemArrSlice {
            elements: &mut node.elements,
            start: if current_leaf == start_leaf {
                Some((start.elem_index, start.offset))
            } else {
                None
            },
            end: if current_leaf == end_leaf {
                Some((end.elem_index, end.offset))
            } else {
                None
            },
        }
    }

    fn update_dirty_cache_map(&mut self, mut dirty_map: LeafDirtyMap<B::CacheDiff>) {
        // Sort by level. Now order is from leaf to root
        let mut diff_map: FxHashMap<ArenaIndex, B::CacheDiff> = FxHashMap::default();
        for (idx, diff) in dirty_map.iter_mut() {
            if let Some(diff) = take(diff) {
                diff_map.insert(*idx, diff);
            }
        }
        let mut visit_set: FxHashSet<ArenaIndex> = dirty_map.keys().copied().collect();
        while !visit_set.is_empty() {
            for child_idx in take(&mut visit_set) {
                let node = self.nodes.get(child_idx).unwrap();
                let Some(parent_idx) = node.parent else { continue };
                let (child, parent) = self.get2_mut(child_idx, parent_idx);
                let cache_diff = child.calc_cache(
                    &mut parent.children[child.parent_slot as usize].cache,
                    diff_map.remove(&child_idx),
                );

                visit_set.insert(parent_idx);
                if let Some(e) = diff_map.get_mut(&parent_idx) {
                    B::merge_cache_diff(e, &cache_diff.unwrap());
                } else {
                    diff_map.insert(parent_idx, cache_diff.unwrap());
                }
            }
        }

        self.nodes
            .get(self.root)
            .unwrap()
            .calc_cache(&mut self.root_cache, None);
    }

    pub fn iter(&self) -> impl Iterator<Item = &B::Elem> + '_ {
        let mut path = self.first_path().unwrap_or(SmallVec::new());
        let idx = path.last().copied().unwrap_or(Idx::new(self.root, 0));
        let node = self.get_node(idx.arena);
        let mut iter = node.elements.iter();
        core::iter::from_fn(move || loop {
            if path.is_empty() {
                return None;
            }

            match iter.next() {
                None => {
                    if !self.next_sibling(&mut path) {
                        return None;
                    }

                    let idx = *path.last().unwrap();
                    let node = self.get_node(idx.arena);
                    iter = node.elements.iter();
                }
                Some(elem) => return Some(elem),
            }
        })
    }

    fn first_path(&self) -> Option<NodePath> {
        let mut path = NodePath::new();
        let mut index = self.root;
        let mut node = self.nodes.get(index).unwrap();
        if node.is_empty() {
            return None;
        }

        while node.is_internal() {
            path.push(Idx::new(index, 0));
            index = node.children[0].arena;
            node = self.nodes.get(index).unwrap();
        }

        path.push(Idx::new(index, 0));
        Some(path)
    }

    fn last_path(&self) -> Option<NodePath> {
        let mut path = NodePath::new();
        let mut index = self.root;
        let mut node = self.nodes.get(index).unwrap();
        if node.is_empty() {
            return None;
        }

        while node.is_internal() {
            path.push(Idx::new(index, node.children.len() - 1));
            index = node.children[node.children.len() - 1].arena;
            node = self.nodes.get(index).unwrap();
        }

        path.push(Idx::new(index, node.elements.len() - 1));
        Some(path)
    }

    pub fn first_leaf(&self) -> ArenaIndex {
        let mut index = self.root;
        let mut node = self.nodes.get(index).unwrap();
        while node.is_internal() {
            index = node.children[0].arena;
            node = self.nodes.get(index).unwrap();
        }

        index
    }

    pub fn last_leaf(&self) -> ArenaIndex {
        let mut index = self.root;
        let mut node = self.nodes.get(index).unwrap();
        while node.is_internal() {
            index = node.children[node.children.len() - 1].arena;
            node = self.nodes.get(index).unwrap();
        }

        index
    }

    #[inline]
    pub fn range<Q>(&self, range: Range<Q::QueryArg>) -> Range<QueryResult>
    where
        Q: Query<B>,
    {
        self.query::<Q>(&range.start)..self.query::<Q>(&range.end)
    }

    pub fn iter_range(
        &self,
        range: impl RangeBounds<QueryResult>,
    ) -> impl Iterator<Item = ElemSlice<'_, B::Elem>> + '_ {
        let start = match range.start_bound() {
            std::ops::Bound::Included(start) => *start,
            std::ops::Bound::Excluded(_) => unreachable!(),
            std::ops::Bound::Unbounded => self.first_full_path(),
        };
        let end = match range.end_bound() {
            std::ops::Bound::Included(end) => *end,
            std::ops::Bound::Excluded(end) => *end,
            std::ops::Bound::Unbounded => self.last_full_path(),
        };
        self._iter_range(start, end)
    }

    fn _iter_range(
        &self,
        start: QueryResult,
        end: QueryResult,
    ) -> impl Iterator<Item = ElemSlice<'_, B::Elem>> + '_ {
        let mut node_iter =
            iter::Iter::new(self, self.get_path(start.leaf), self.get_path(end.leaf));
        let mut elem_iter: Option<Map<_, _>> = None;
        core::iter::from_fn(move || loop {
            if let Some(inner_elem_iter) = &mut elem_iter {
                match inner_elem_iter.next() {
                    Some(elem) => return Some(elem),
                    None => elem_iter = None,
                }
            } else {
                match node_iter.next() {
                    Some((path, node)) => {
                        let idx = path.last().unwrap();
                        let (start_idx, start_offset) = if idx.arena == start.leaf {
                            (start.elem_index, Some(start.offset))
                        } else {
                            (0, None)
                        };
                        let (end_idx, end_offset) = if idx.arena == end.leaf {
                            (
                                (end.elem_index + 1).min(node.elements.len()),
                                Some(end.offset),
                            )
                        } else {
                            (node.elements.len(), None)
                        };

                        elem_iter = Some(node.elements[start_idx..end_idx].iter().enumerate().map(
                            move |(i, x)| ElemSlice {
                                path: QueryResult {
                                    leaf: path.last().unwrap().arena,
                                    elem_index: i + start_idx,
                                    offset: if i == 0 { start_offset.unwrap_or(0) } else { 0 },
                                    found: true,
                                },
                                elem: x,
                                start: if i == 0 { start_offset } else { None },
                                end: if i == end_idx - start_idx - 1 {
                                    end_offset
                                } else {
                                    None
                                },
                            },
                        ));
                    }
                    None => return None,
                }
            }
        })
    }

    pub fn first_full_path(&self) -> QueryResult {
        QueryResult {
            leaf: self.first_leaf(),
            elem_index: 0,
            offset: 0,
            found: false,
        }
    }

    pub fn last_full_path(&self) -> QueryResult {
        let leaf = self.last_leaf();
        let node = self.get_node(leaf);
        let elem_index = node.elements.len();
        QueryResult {
            leaf,
            elem_index,
            offset: 0,
            found: false,
        }
    }

    // at call site the cache at path can be out-of-date.
    // the cache will be up-to-date after this method
    fn split(&mut self, node_idx: ArenaIndex) {
        let node = self.nodes.get_mut(node_idx).unwrap();
        let node_parent = node.parent;
        let node_parent_slot = node.parent_slot;
        let right: Node<B> = Node {
            parent: node.parent,
            parent_slot: u32::MAX,
            elements: Vec::new(),
            children: Vec::new(),
        };

        let mut right_children = Vec::new();
        let mut right_elements = Vec::new();
        // split
        if node.is_internal() {
            let split = node.children.len() / 2;
            right_children = node.children.split_off(split);
        } else {
            let split = node.elements.len() / 2;
            right_elements = node.elements.split_off(split);
        }

        // update cache
        let mut right_cache = B::Cache::default();
        if right_children.is_empty() {
            right.calc_cache(&mut right_cache, None);
        }
        let right_arena_idx = self.nodes.insert(right);
        let this_cache = {
            let node = self.get_mut(node_idx);
            let mut cache = Default::default();
            node.calc_cache(&mut cache, None);
            cache
        };

        if !right_children.is_empty() {
            // update children's parent info
            for (i, child) in right_children.iter().enumerate() {
                let child = self.get_mut(child.arena);
                child.parent = Some(right_arena_idx);
                child.parent_slot = i as u32;
            }
        }
        let right = self.nodes.get_mut(right_arena_idx).unwrap();
        if !right_elements.is_empty() {
            if let Some(listener) = self.element_move_listener.as_mut() {
                for elem in right_elements.iter() {
                    listener((right_arena_idx, elem).into());
                }
            }
        }

        right.elements = right_elements;
        right.children = right_children;
        // update parent cache
        right.calc_cache(&mut right_cache, None);

        self.inner_insert_node(
            node_parent,
            node_parent_slot as usize,
            this_cache,
            Child {
                arena: right_arena_idx,
                cache: right_cache,
            },
        );
        // don't need to recursive update cache
    }

    // call site should ensure the cache is up-to-date after this method
    fn inner_insert_node(
        &mut self,
        parent_idx: Option<ArenaIndex>,
        index: usize,
        new_cache: B::Cache,
        node: Child<B>,
    ) {
        if let Some(parent_idx) = parent_idx {
            let parent = self.get_mut(parent_idx);
            parent.children[index].cache = new_cache;
            parent.children.insert(index + 1, node);
            let is_full = parent.is_full();
            self.update_children_parent_slot_from(parent_idx, index + 1);
            if is_full {
                self.split(parent_idx);
            }
        } else {
            self.split_root(new_cache, node);
        }
    }

    fn update_children_parent_slot_from(&mut self, parent_idx: ArenaIndex, index: usize) {
        let parent = self.get_mut(parent_idx);
        let children = take(&mut parent.children);
        for (i, child) in children[index..].iter().enumerate() {
            let idx = index + i;
            let child = self.get_mut(child.arena);
            child.parent_slot = idx as u32;
        }
        let parent = self.get_mut(parent_idx);
        parent.children = children;
    }

    /// right's cache should be up-to-date
    fn split_root(&mut self, new_cache: B::Cache, right: Child<B>) {
        let root_idx = self.root;
        // set right parent
        let right_node = &mut self.get_mut(right.arena);
        right_node.parent_slot = 1;
        right_node.parent = Some(root_idx);
        let root = self.get_mut(self.root);
        // let left be root
        let mut left_node: Node<B> = core::mem::take(root);
        left_node.parent_slot = 0;
        // set left parent
        left_node.parent = Some(root_idx);

        // push left and right to root.children
        root.children = Vec::with_capacity(B::MAX_LEN);
        let left_children = left_node.children.clone();
        let left_arena = self.nodes.insert(left_node);
        let left = Child::new(left_arena, new_cache);
        let mut cache = std::mem::take(&mut self.root_cache);
        let root = self.get_mut(self.root);
        root.children.push(left);
        root.children.push(right);

        // update new root cache
        root.calc_cache(&mut cache, None);

        if left_children.is_empty() {
            // leaf node
            let left = self.nodes.get(left_arena).unwrap();
            debug_assert!(left.is_leaf());
            if let Some(listener) = self.element_move_listener.as_mut() {
                for elem in left.elements.iter() {
                    listener((left_arena, elem).into());
                }
            }
        } else {
            // update left's children's parent
            for child in left_children {
                self.get_mut(child.arena).parent = Some(left_arena);
            }
        }

        self.root_cache = cache;
    }

    #[inline(always)]
    fn get_mut(&mut self, index: ArenaIndex) -> &mut Node<B> {
        self.nodes.get_mut(index).unwrap()
    }

    #[inline(always)]
    fn get2_mut(&mut self, a: ArenaIndex, b: ArenaIndex) -> (&mut Node<B>, &mut Node<B>) {
        let (a, b) = self.nodes.get2_mut(a, b);
        (a.unwrap(), b.unwrap())
    }

    #[inline(always)]
    pub fn get_node(&self, index: ArenaIndex) -> &Node<B> {
        self.nodes.get(index).unwrap()
    }

    /// The given node is lack of children/elements.
    /// We should merge it into its neighbor or borrow from its neighbor.
    ///
    /// Given a random neighbor is neither full or lack, it's guaranteed
    /// that we can either merge into or borrow from it without breaking
    /// the balance rule.
    ///
    /// - cache should be up-to-date when calling this.
    ///
    /// return is parent lack
    fn handle_lack(&mut self, node_idx: ArenaIndex) -> LackInfo {
        if self.root == node_idx {
            return LackInfo {
                is_parent_lack: false,
            };
        }

        let node = self.get_node(node_idx);
        let parent_idx = node.parent.unwrap();
        let parent = self.get_node(parent_idx);
        debug_assert_eq!(parent.children[node.parent_slot as usize].arena, node_idx,);
        let ans = match self.pair_neighbor(node_idx) {
            Some((a_idx, b_idx)) => {
                let parent = self.get_mut(parent_idx);
                let mut a_cache = std::mem::take(&mut parent.children[a_idx.arr].cache);
                let mut b_cache = std::mem::take(&mut parent.children[b_idx.arr].cache);
                let mut re_parent = FxHashMap::default();

                let (a, b) = self.nodes.get2_mut(a_idx.arena, b_idx.arena);
                let a = a.unwrap();
                let b = b.unwrap();
                let ans = if a.len() + b.len() >= B::MAX_LEN {
                    // move
                    if a.len() < b.len() {
                        // move part of b's children to a
                        let move_len = (b.len() - a.len()) / 2;
                        if b.is_internal() {
                            for child in b.children.drain(..move_len) {
                                re_parent.insert(child.arena, (a_idx.arena, a.children.len()));
                                a.children.push(child);
                            }
                            for (i, child) in b.children.iter().enumerate() {
                                re_parent.insert(child.arena, (b_idx.arena, i));
                            }
                        } else if let Some(listener) = self.element_move_listener.as_ref() {
                            a.elements.extend(b.elements.drain(..move_len).map(|x| {
                                listener((a_idx.arena, &x).into());
                                x
                            }));
                        } else {
                            a.elements.extend(b.elements.drain(..move_len));
                        }
                    } else {
                        // move part of a's children to b
                        let move_len = (a.len() - b.len()) / 2;
                        if a.is_internal() {
                            for (i, child) in b.children.iter().enumerate() {
                                re_parent.insert(child.arena, (b_idx.arena, i + move_len));
                            }
                            b.children.splice(
                                0..0,
                                a.children
                                    .drain(a.children.len() - move_len..)
                                    .enumerate()
                                    .map(|(i, x)| {
                                        re_parent.insert(x.arena, (b_idx.arena, i));
                                        x
                                    }),
                            );
                        } else if let Some(listener) = self.element_move_listener.as_ref() {
                            b.elements.splice(
                                0..0,
                                a.elements.drain(a.elements.len() - move_len..).map(|x| {
                                    listener((b_idx.arena, &x).into());
                                    x
                                }),
                            );
                        } else {
                            b.elements
                                .splice(0..0, a.elements.drain(a.elements.len() - move_len..));
                        }
                    }
                    a.calc_cache(&mut a_cache, None);
                    b.calc_cache(&mut b_cache, None);
                    let parent = self.get_mut(parent_idx);
                    parent.children[a_idx.arr].cache = a_cache;
                    parent.children[b_idx.arr].cache = b_cache;
                    LackInfo {
                        is_parent_lack: parent.is_lack(),
                    }
                } else {
                    // merge
                    let is_parent_lack = if node_idx == a_idx.arena {
                        // merge b to a, delete b
                        if a.is_internal() {
                            for (i, child) in b.children.iter().enumerate() {
                                re_parent.insert(child.arena, (a_idx.arena, a.children.len() + i));
                            }
                            a.children.append(&mut b.children);
                        } else {
                            {
                                // notify element move
                                let leaf = a_idx.arena;
                                let elements: &[B::Elem] = &b.elements;
                                if let Some(listener) = self.element_move_listener.as_ref() {
                                    for elem in elements.iter() {
                                        listener((leaf, elem).into());
                                    }
                                }
                            }
                            a.elements.append(&mut b.elements);
                        }
                        a.calc_cache(&mut a_cache, None);
                        let parent = self.get_mut(parent_idx);
                        parent.children[a_idx.arr].cache = a_cache;
                        parent.children.remove(b_idx.arr);
                        let is_lack = parent.is_lack();
                        self.purge(b_idx.arena);
                        self.update_children_parent_slot_from(parent_idx, b_idx.arr);
                        is_lack
                    } else {
                        // merge a to b, delete a
                        if a.is_internal() {
                            for (i, child) in a.children.iter().enumerate() {
                                re_parent.insert(child.arena, (b_idx.arena, i));
                            }
                            for (i, child) in b.children.iter().enumerate() {
                                re_parent.insert(child.arena, (b_idx.arena, i + a.children.len()));
                            }
                            b.children.splice(0..0, core::mem::take(&mut a.children));
                        } else {
                            {
                                // notify element move
                                let leaf = b_idx.arena;
                                let elements: &[B::Elem] = &a.elements;
                                if let Some(listener) = self.element_move_listener.as_ref() {
                                    for elem in elements.iter() {
                                        listener((leaf, elem).into());
                                    }
                                }
                            }
                            b.elements.splice(0..0, core::mem::take(&mut a.elements));
                        }
                        b.calc_cache(&mut b_cache, None);
                        let parent = self.get_mut(parent_idx);
                        parent.children[b_idx.arr].cache = b_cache;
                        parent.children.remove(a_idx.arr);
                        let is_lack = parent.is_lack();
                        self.purge(a_idx.arena);
                        self.update_children_parent_slot_from(parent_idx, a_idx.arr);
                        is_lack
                    };

                    LackInfo { is_parent_lack }
                };

                for (child, (parent, slot)) in re_parent {
                    let child = self.get_mut(child);
                    child.parent = Some(parent);
                    child.parent_slot = slot as u32;
                }
                ans
            }
            None => LackInfo {
                is_parent_lack: true,
            },
        };
        ans
    }

    fn try_reduce_levels(&mut self) {
        let mut reduced = false;
        while self.get_node(self.root).children.len() == 1 {
            let root = self.get_node(self.root);
            let child_arena = root.children[0].arena;
            let child = self.nodes.remove(child_arena).unwrap();
            let root = self.get_mut(self.root);
            let _ = core::mem::replace(root, child);
            reduced = true;
            // root cache should be the same as child cache because there is only one child
        }
        if reduced {
            let root_idx = self.root;
            let root = self.get_mut(self.root);
            root.parent = None;
            root.parent_slot = u32::MAX;
            if root.is_internal() {
                let children = root.children.clone();
                for child in children {
                    let child = self.get_mut(child.arena);
                    child.parent = Some(root_idx);
                }
            } else {
                let root = self.get_node(root_idx);
                if let Some(listener) = self.element_move_listener.as_ref() {
                    for elem in root.elements.iter() {
                        listener((root_idx, elem).into());
                    }
                }
            }
        }
    }

    fn pair_neighbor(&self, this: ArenaIndex) -> Option<(Idx, Idx)> {
        let node = self.get_node(this);
        let arr = node.parent_slot as usize;
        let parent = self.get_node(node.parent.unwrap());

        if arr == 0 {
            parent
                .children
                .get(1)
                .map(|x| (Idx::new(this, arr), Idx::new(x.arena, 1)))
        } else {
            parent
                .children
                .get(arr - 1)
                .map(|x| (Idx::new(x.arena, arr - 1), Idx::new(this, arr)))
        }
    }

    /// Sometimes we cannot use diff because no only the given node is changed, but also its siblings.
    /// For example, after delete a range of nodes, we cannot use the diff from child to infer the diff of parent.
    pub fn recursive_update_cache(
        &mut self,
        node_idx: ArenaIndex,
        can_use_diff: bool,
        cache_diff: Option<B::CacheDiff>,
    ) {
        let mut this_idx = node_idx;
        let mut node = self.get_mut(node_idx);
        let mut this_arr = node.parent_slot;
        let mut diff = cache_diff;
        if can_use_diff {
            while node.parent.is_some() {
                let parent_idx = node.parent.unwrap();
                let (parent, this) = self.get2_mut(parent_idx, this_idx);
                diff = this.calc_cache(&mut parent.children[this_arr as usize].cache, diff);
                this_idx = parent_idx;
                this_arr = parent.parent_slot;
                node = parent;
            }
        } else {
            while node.parent.is_some() {
                let parent_idx = node.parent.unwrap();
                let (parent, this) = self.get2_mut(parent_idx, this_idx);
                this.calc_cache(&mut parent.children[this_arr as usize].cache, None);
                this_idx = parent_idx;
                this_arr = parent.parent_slot;
                node = parent;
            }
        }

        let mut root_cache = std::mem::take(&mut self.root_cache);
        let root = self.root_mut();
        root.calc_cache(&mut root_cache, diff);
        self.root_cache = root_cache;
    }

    fn purge(&mut self, index: ArenaIndex) {
        let mut stack: SmallVec<[_; 64]> = smallvec::smallvec![index];
        while let Some(x) = stack.pop() {
            let Some(node) = self.nodes.get(x) else { continue };
            if node.is_leaf() {
                if let Some(listener) = &mut self.element_move_listener {
                    for elem in node.elements.iter() {
                        listener(MoveEvent::new_del(elem));
                    }
                }
            } else {
                for x in node.children.iter() {
                    stack.push(x.arena);
                }
            }
            self.nodes.remove(x);
        }
    }

    /// find the next sibling at the same level
    ///
    /// return false if there is no next sibling
    #[must_use]
    fn next_sibling(&self, path: &mut [Idx]) -> bool {
        if path.len() <= 1 {
            return false;
        }

        let depth = path.len();
        let parent_idx = path[depth - 2];
        let this_idx = path[depth - 1];
        let parent = self.get_node(parent_idx.arena);
        match parent.children.get(this_idx.arr + 1) {
            Some(next) => {
                path[depth - 1] = Idx::new(next.arena, this_idx.arr + 1);
            }
            None => {
                if !self.next_sibling(&mut path[..depth - 1]) {
                    return false;
                }

                let parent = self.get_node(path[depth - 2].arena);
                path[depth - 1] = Idx::new(parent.children[0].arena, 0);
            }
        }

        true
    }

    pub fn next_same_level_node(&self, node_idx: ArenaIndex) -> Option<ArenaIndex> {
        let node = self.get_node(node_idx);
        let parent = self.get_node(node.parent?);
        if let Some(next) = parent.children.get(node.parent_slot as usize + 1) {
            Some(next.arena)
        } else if let Some(parent_next) = self.next_same_level_node(node.parent?) {
            let parent_next = self.get_node(parent_next);
            parent_next.children.first().map(|x| x.arena)
        } else {
            None
        }
    }

    fn next_same_level_node_with_filter(
        &self,
        node_idx: ArenaIndex,
        end_path: &[Idx],
        filter: &dyn Fn(&B::Cache) -> bool,
    ) -> Option<ArenaIndex> {
        let node = self.get_node(node_idx);
        let mut parent = self.get_node(node.parent?);
        let mut next_index = node.parent_slot as usize + 1;
        loop {
            if let Some(next) = parent.children.get(next_index) {
                if filter(&next.cache) {
                    return Some(next.arena);
                }
                if next.arena == end_path.last().unwrap().arena {
                    return None;
                }

                next_index += 1;
            } else if end_path[end_path.len() - 2].arena == node.parent.unwrap() {
                return None;
            } else if let Some(parent_next) = self.next_same_level_node_with_filter(
                node.parent?,
                &end_path[..end_path.len() - 1],
                filter,
            ) {
                parent = self.get_node(parent_next);
                next_index = 0;
            } else {
                return None;
            }
        }
    }

    pub fn prev_same_level_node(&self, node_idx: ArenaIndex) -> Option<ArenaIndex> {
        let node = self.get_node(node_idx);
        let parent = self.get_node(node.parent?);
        if node.parent_slot > 0 {
            let Some(next) = parent.children.get(node.parent_slot as usize - 1) else { unreachable!() };
            Some(next.arena)
        } else if let Some(parent_prev) = self.prev_same_level_node(node.parent?) {
            let parent_prev = self.get_node(parent_prev);
            parent_prev.children.last().map(|x| x.arena)
        } else {
            None
        }
    }

    /// find the next sibling at the same level
    ///
    /// return false if there is no next sibling
    #[must_use]
    fn prev_sibling(&self, path: &mut [Idx]) -> bool {
        if path.len() <= 1 {
            return false;
        }

        let depth = path.len();
        let parent_idx = path[depth - 2];
        let this_idx = path[depth - 1];
        let parent = self.get_node(parent_idx.arena);
        if this_idx.arr >= 1 {
            let prev = &parent.children[this_idx.arr - 1];
            path[depth - 1] = Idx::new(prev.arena, this_idx.arr - 1);
        } else {
            if !self.prev_sibling(&mut path[..depth - 1]) {
                return false;
            }

            let parent = self.get_node(path[depth - 2].arena);
            path[depth - 1] = Idx::new(
                parent.children.last().unwrap().arena,
                parent.children.len() - 1,
            );
        }

        true
    }

    /// if index is None, then using the last index
    fn try_get_path_from_indexes(&self, indexes: &[Option<usize>]) -> Option<NodePath> {
        debug_assert_eq!(indexes[0], Some(0));
        let mut path = smallvec::smallvec![Idx::new(self.root, 0)];
        let mut node_idx = self.root;
        for &index in indexes[1..].iter() {
            let node = self.get_node(node_idx);
            if node.children.is_empty() {
                return None;
            }

            let i = match index {
                Some(index) => index,
                None => node.children.len() - 1,
            };
            path.push(Idx::new(node.children.get(i)?.arena, i));
            node_idx = node.children[i].arena;
        }
        Some(path)
    }

    pub fn root_cache(&self) -> &B::Cache {
        &self.root_cache
    }

    /// This method will release the memory back to OS.
    /// Currently, it's just `*self = Self::new()`
    pub fn clear(&mut self) {
        *self = Self::new();
    }

    fn root_mut(&mut self) -> &mut Node<B> {
        self.get_mut(self.root)
    }

    pub fn is_empty(&self) -> bool {
        self.get_node(self.root).is_empty()
    }

    fn get_path(&self, idx: ArenaIndex) -> NodePath {
        let mut path = NodePath::new();
        let mut node_idx = idx;
        while node_idx != self.root {
            let node = self.get_node(node_idx);
            path.push(Idx::new(node_idx, node.parent_slot as usize));
            node_idx = node.parent.unwrap();
        }
        path.push(Idx::new(self.root, 0));
        path.reverse();
        path
    }

    pub fn push(&mut self, elem: B::Elem) {
        let leaf_idx = self.last_leaf();
        if let Some(listener) = self.element_move_listener.as_ref() {
            listener((leaf_idx, &elem).into());
        }
        let leaf = self.get_mut(leaf_idx);
        debug_assert!(leaf.is_leaf());
        leaf.elements.push(elem);
        let is_full = leaf.is_full();
        self.recursive_update_cache(leaf_idx, true, None);
        if is_full {
            self.split(leaf_idx);
        }
    }

    pub fn prepend(&mut self, elem: B::Elem) {
        let leaf_idx = self.first_leaf();
        let leaf = self.nodes.get_mut(leaf_idx).unwrap();
        if let Some(listener) = self.element_move_listener.as_ref() {
            listener((leaf_idx, &elem).into());
        }
        debug_assert!(leaf.is_leaf());
        leaf.elements.insert(0, elem);
        let is_full = leaf.is_full();
        self.recursive_update_cache(leaf_idx, true, None);
        if is_full {
            self.split(leaf_idx);
        }
    }

    #[inline]
    /// This method only works when [`MoveListener`] listener is registered
    pub fn notify_batch_move(&self, leaf: ArenaIndex, elements: &[B::Elem]) {
        if let Some(listener) = self.element_move_listener.as_ref() {
            for elem in elements.iter() {
                listener((leaf, elem).into());
            }
        }
    }

    #[inline]
    pub fn notify_elem_move(&self, leaf: ArenaIndex, elem: &B::Elem) {
        if let Some(listener) = self.element_move_listener.as_ref() {
            listener((leaf, elem).into());
        }
    }

    /// compare the position of a and b
    pub fn compare_pos(&self, a: QueryResult, b: QueryResult) -> Ordering {
        if a.leaf == b.leaf {
            if a.elem_index == b.elem_index {
                return a.offset.cmp(&b.offset);
            }
            return a.elem_index.cmp(&b.elem_index);
        }

        let mut node_a = self.get_node(a.leaf);
        let mut node_b = self.get_node(b.leaf);
        while node_a.parent != node_b.parent {
            node_a = self.get_node(node_a.parent.unwrap());
            node_b = self.get_node(node_b.parent.unwrap());
        }

        node_a.parent_slot.cmp(&node_b.parent_slot)
    }
}

#[inline(always)]
fn add_leaf_dirty_map<T>(leaf: ArenaIndex, dirty_map: &mut LeafDirtyMap<T>, leaf_diff: Option<T>) {
    dirty_map.insert(leaf, leaf_diff);
}

impl<B: BTreeTrait> BTree<B> {
    #[allow(unused)]
    pub fn check(&self) {
        // check cache
        let mut leaf_level = None;
        for (index, node) in self.nodes.iter() {
            if node.is_internal() {
                assert!(!node.is_empty());
                for (i, child_info) in node.children.iter().enumerate() {
                    let child = self.get_node(child_info.arena);
                    let mut cache = Default::default();
                    child.calc_cache(&mut cache, None);
                    assert_eq!(child.parent_slot, i as u32);
                    assert_eq!(child.parent, Some(index));
                    assert_eq!(cache, child_info.cache);
                }
            } else {
                let mut length = 0;
                let mut node_idx = index;
                while node_idx != self.root {
                    let node = self.get_node(node_idx);
                    length += 1;
                    node_idx = node.parent.unwrap();
                }
                match leaf_level {
                    Some(expected) => assert_eq!(length, expected),
                    None => {
                        leaf_level = Some(length);
                    }
                }
            }
            if let Some(parent) = node.parent {
                let parent = self.get_node(parent);
                assert_eq!(parent.children[node.parent_slot as usize].arena, index);
                self.get_path(index);
            } else {
                assert_eq!(index, self.root)
            }

            if index != self.root {
                assert!(!node.is_lack(), "len={}\n", node.len());
            }

            assert!(!node.is_full(), "len={}", node.len());
        }
    }
}

impl<B: BTreeTrait> Default for BTree<B> {
    fn default() -> Self {
        Self::new()
    }
}
