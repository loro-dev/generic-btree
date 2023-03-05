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

#![deny(unsafe_code)]

use core::{
    fmt::Debug,
    iter::Map,
    ops::{Deref, Range},
};

use fxhash::FxHashMap;
use smallvec::SmallVec;
use thunderdome::{Arena, Index as ArenaIndex};
mod generic_impl;
mod iter;
pub use generic_impl::*;

use crate::rle::HasLength;
pub mod rle;
pub type SmallElemVec<T> = SmallVec<[T; 8]>;
pub type StackVec<T> = SmallVec<[T; 8]>;
pub type HeapVec<T> = Vec<T>;

pub trait BTreeTrait {
    type Elem: Debug;
    type Cache: Debug + Default + Clone + Eq;
    type CacheDiff;
    /// Use () if you don't need write buffer.
    /// Associated type default is still unstable so we don't provide default value.
    type WriteBuffer: Debug + Clone;
    const MAX_LEN: usize;

    fn element_to_cache(element: &Self::Elem) -> Self::Cache;
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

    #[allow(unused)]
    #[inline(always)]
    fn apply_write_buffer_to_elements(
        elements: &mut HeapVec<Self::Elem>,
        write_buffer: &Self::WriteBuffer,
    ) {
        unimplemented!()
    }

    #[allow(unused)]
    #[inline(always)]
    fn apply_write_buffer_to_nodes(children: &mut [Child<Self>], write_buffer: &Self::WriteBuffer) {
        unimplemented!()
    }

    fn calc_cache_internal(
        cache: &mut Self::Cache,
        caches: &[Child<Self>],
        diff: Option<Self::CacheDiff>,
    ) -> Self::CacheDiff;
    fn calc_cache_leaf(cache: &mut Self::Cache, elements: &[Self::Elem]) -> Self::CacheDiff;
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
}

impl<Elem: Clone, B: BTreeTrait<Elem = Elem>> Clone for BTree<B> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            nodes: self.nodes.clone(),
            root: self.root,
            root_cache: self.root_cache.clone(),
            need_flush: false,
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

// TODO: can be replaced by a immutable structure?
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

// TODO: should check the path is still valid before using it to update in debug mode
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryResult {
    node_path: NodePath,
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
    pub start: Option<(usize, usize)>,
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
    fn path_ref(&self) -> PathRef {
        self.node_path.as_slice().into()
    }

    pub fn elem<'b, Elem: Debug, B: BTreeTrait<Elem = Elem>>(
        &self,
        tree: &'b BTree<B>,
    ) -> Option<&'b Elem> {
        tree.nodes
            .get(self.path_ref().this().arena)
            .and_then(|x| x.elements.get(self.elem_index))
    }
}

// TODO: use enum to save spaces
struct Node<B: BTreeTrait> {
    parent: Option<ArenaIndex>,
    parent_slot: u32,
    elements: HeapVec<B::Elem>,
    children: HeapVec<Child<B>>,
}

impl<
        W: Debug,
        Cache: Debug,
        Elem: Debug,
        B: BTreeTrait<Elem = Elem, Cache = Cache, WriteBuffer = W>,
    > Debug for BTree<B>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        fn fmt_node<
            W: Debug,
            Cache: Debug,
            Elem: Debug,
            B: BTreeTrait<Elem = Elem, WriteBuffer = W>,
        >(
            tree: &BTree<B>,
            node: &Node<B>,
            f: &mut core::fmt::Formatter<'_>,
            indent_size: usize,
        ) -> core::fmt::Result {
            if node.is_internal() {
                for child in node.children.iter() {
                    indent(f, indent_size)?;
                    let child_node = tree.get(child.arena);
                    f.write_fmt(format_args!(
                        "{} Arena({:?}) Cache: {:?} Buffer: {:?}\n",
                        child_node.parent_slot, &child.arena, &child.cache, &child.write_buffer
                    ))?;
                    fmt_node::<W, Cache, Elem, B>(tree, child_node, f, indent_size + 1)?;
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
        fmt_node::<W, Cache, Elem, B>(self, self.nodes.get(self.root).unwrap(), f, 1)
    }
}

impl<
        W: Debug,
        Cache: Debug,
        Elem: Debug,
        B: BTreeTrait<Elem = Elem, Cache = Cache, WriteBuffer = W>,
    > Debug for Node<B>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Node")
            .field("elements", &self.elements)
            .field("children", &self.children)
            .finish()
    }
}

impl<
        W: Debug,
        Cache: Debug,
        Elem: Debug,
        B: BTreeTrait<Elem = Elem, Cache = Cache, WriteBuffer = W>,
    > Debug for Child<B>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Child")
            .field("index", &self.arena)
            .field("cache", &self.cache)
            .field("write_buffer", &self.write_buffer)
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
    pub write_buffer: Option<B::WriteBuffer>,
}

impl<B: BTreeTrait> Clone for Child<B> {
    fn clone(&self) -> Self {
        Self {
            arena: self.arena,
            cache: self.cache.clone(),
            write_buffer: self.write_buffer.clone(),
        }
    }
}

impl<B: BTreeTrait> Child<B> {
    #[inline(always)]
    pub fn cache(&self) -> &B::Cache {
        &self.cache
    }

    fn new(arena: ArenaIndex, cache: B::Cache) -> Self {
        Self {
            arena,
            cache,
            write_buffer: None,
        }
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
    fn calc_cache(&self, cache: &mut B::Cache, diff: Option<B::CacheDiff>) -> B::CacheDiff {
        if self.is_internal() {
            B::calc_cache_internal(cache, &self.children, diff)
        } else {
            B::calc_cache_leaf(cache, &self.elements)
        }
    }
}

type DirtyMap = FxHashMap<(isize, ArenaIndex), HeapVec<Idx>>;

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
        }
    }

    #[inline]
    pub fn node_len(&self) -> usize {
        self.nodes.len()
    }

    pub fn insert<Q>(&mut self, tree_index: &Q::QueryArg, data: B::Elem)
    where
        Q: Query<B>,
    {
        let result = self.query::<Q>(tree_index);
        self.insert_by_query_result(result, data)
    }

    /// It will invoke [`BTreeTrait::insert`]
    pub fn insert_by_query_result(&mut self, result: QueryResult, data: B::Elem) {
        let index = *result.node_path.last().unwrap();
        let node = self.nodes.get_mut(index.arena).unwrap();
        B::insert(&mut node.elements, result.elem_index, result.offset, data);
        let is_full = node.is_full();
        self.recursive_update_cache(result.path_ref());
        if is_full {
            self.split(result.path_ref());
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
        let index = *result.node_path.last().unwrap();
        let node = self.nodes.get_mut(index.arena).unwrap();
        B::insert_batch(&mut node.elements, result.elem_index, result.offset, data);

        let is_full = node.is_full();
        self.recursive_update_cache(result.path_ref());
        if is_full {
            self.split(result.path_ref());
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
        let index = *result.node_path.last().unwrap();
        let node = self.nodes.get_mut(index.arena).unwrap();
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
        self.recursive_update_cache(result.path_ref());
        if is_full {
            self.split(result.path_ref());
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

        let index = *result.node_path.last().unwrap();
        let node = self.nodes.get_mut(index.arena).unwrap();
        let mut ans = None;
        if result.found {
            ans = Q::delete(&mut node.elements, query, result.elem_index, result.offset);
        }

        let is_full = node.is_full();
        let is_lack = node.is_lack();
        self.recursive_update_cache(result.path_ref());
        if is_full {
            self.split(result.path_ref());
        } else if is_lack {
            let mut path_ref = result.path_ref();
            while self.handle_lack(path_ref.this().arena) {
                path_ref.set_as_parent_path();
            }

            self.try_reduce_levels()
        }
        ans
    }

    #[inline]
    pub fn query<Q>(&self, query: &Q::QueryArg) -> QueryResult
    where
        Q: Query<B>,
    {
        self.query_with_finder_return::<Q>(query).0
    }

    /// Shift by offset 1.
    ///
    /// It will not stay on empty spans but scan forward
    pub fn shift_path_by_one_offset(&self, path: &mut QueryResult)
    where
        B::Elem: rle::HasLength,
    {
        let mut node = self
            .nodes
            .get(path.node_path.last().unwrap().arena)
            .unwrap();
        loop {
            if path.elem_index == node.elements.len() {
                path.elem_index = 0;
                path.offset = 0;

                node = self
                    .nodes
                    .get(path.node_path.last().unwrap().arena)
                    .unwrap();

                if !self.next_sibling(&mut path.node_path) {
                    path.elem_index = node.children.len();
                    path.offset = 0;
                    break;
                }
            }

            assert!(node.is_leaf() && path.elem_index <= node.elements.len());
            let elem = &node.elements[path.elem_index];
            // skip empty span
            if elem.rle_len() >= path.offset {
                path.elem_index += 1;
                path.offset = 0;
            } else {
                path.offset += 1;
                break;
            }
        }
    }

    pub fn query_with_finder_return<Q>(&self, query: &Q::QueryArg) -> (QueryResult, Q)
    where
        Q: Query<B>,
    {
        let mut finder = Q::init(query);
        let mut node = self.nodes.get(self.root).unwrap();
        let mut index = self.root;
        let mut ans = QueryResult {
            node_path: smallvec::smallvec![Idx::new(index, 0)],
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
            ans.node_path.push(Idx::new(index, i));
        }

        let result = finder.find_element(query, &node.elements);
        ans.elem_index = result.index;
        ans.found = ans.found && result.found;
        ans.offset = result.offset;
        (ans, finder)
    }

    #[inline]
    pub fn get_elem(&mut self, q: QueryResult) -> Option<&B::Elem> {
        if !q.found {
            return None;
        }

        self.flush_path(PathRef::from(&q.node_path));
        let index = *q.node_path.last().unwrap();
        let node = self.nodes.get(index.arena)?;
        node.elements.get(q.elem_index)
    }

    pub fn drain<Q>(&mut self, range: Range<Q::QueryArg>) -> iter::Drain<B, Q>
    where
        Q: Query<B>,
    {
        self.flush_write_buffer();
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
    pub fn update<F>(&mut self, range: Range<&QueryResult>, f: &mut F)
    where
        F: FnMut(MutElemArrSlice<'_, B::Elem>) -> bool,
    {
        let start = range.start;
        let end = range.end;
        let start_leaf = start.node_path.last().unwrap();
        let mut path = start.node_path.clone();
        let end_leaf = end.node_path.last().unwrap();
        type Level = isize; // always positive, use isize to avoid subtract overflow
        let mut dirty_map: FxHashMap<(Level, ArenaIndex), HeapVec<Idx>> = FxHashMap::default();

        loop {
            let current_leaf = path.last().unwrap();
            let slice = self.get_slice(&path, start_leaf, start, end_leaf, end);
            let should_update_cache = f(slice);
            if should_update_cache {
                // TODO: TEST THIS
                add_path_to_dirty_map(&path, &mut dirty_map);
            }

            if current_leaf == end_leaf {
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

    /// Update the elements with buffer
    ///
    /// F and G should returns true if the cache need to be updated
    ///
    /// Update is not in the order of the elements
    ///
    /// This method may break the balance of the tree
    pub fn update_with_buffer<F, G>(&mut self, range: Range<QueryResult>, f: &mut F, mut g: G)
    where
        F: FnMut(MutElemArrSlice<'_, B::Elem>) -> bool,
        G: FnMut(&mut Option<B::WriteBuffer>, &B::Cache) -> bool,
    {
        self.need_flush = true;
        let start = range.start;
        let end = range.end;
        let start_leaf = start.node_path.last().unwrap();
        let end_leaf = end.node_path.last().unwrap();
        type Level = isize; // always positive, use isize to avoid overflow
        let mut dirty_map: FxHashMap<(Level, ArenaIndex), HeapVec<Idx>> = FxHashMap::default();
        let max_same_parent_level = start
            .node_path
            .iter()
            .zip(end.node_path.iter())
            .take_while(|(a, b)| a.arena == b.arena)
            .count()
            - 1;
        let leaf_level = start.node_path.len() - 1;

        for path in Some(&start.node_path).iter().chain(
            (if start_leaf == end_leaf {
                None
            } else {
                Some(&end.node_path)
            })
            .iter(),
        ) {
            // all elements are in the same leaf
            let slice = self.get_slice(path, start_leaf, &start, end_leaf, &end);
            let should_update_cache = f(slice);

            if should_update_cache {
                add_path_to_dirty_map(path, &mut dirty_map);
            }
        }

        if max_same_parent_level < leaf_level {
            // write to buffer
            self.write_to_buffer(
                Some(&start),
                Some(&end),
                max_same_parent_level,
                &mut g,
                &mut dirty_map,
            );

            for parent_level in max_same_parent_level + 1..leaf_level {
                self.write_to_buffer(Some(&start), None, parent_level, &mut g, &mut dirty_map);
                self.write_to_buffer(None, Some(&end), parent_level, &mut g, &mut dirty_map);
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

    fn write_to_buffer<G>(
        &mut self,
        start: Option<&QueryResult>,
        end: Option<&QueryResult>,
        parent_level: usize,
        g: &mut G,
        dirty_map: &mut DirtyMap,
    ) where
        G: FnMut(&mut Option<B::WriteBuffer>, &B::Cache) -> bool,
    {
        let path = start
            .map(|x| &x.node_path)
            .unwrap_or_else(|| &end.unwrap().node_path);
        let parent = self.nodes.get_mut(path[parent_level].arena).unwrap();
        debug_assert!(parent.is_internal());
        let target_level = parent_level + 1;
        let mut path: NodePath = path[..=target_level].iter().cloned().collect();
        let start_index = start
            .map(|x| x.node_path[target_level].arr + 1)
            .unwrap_or(0);
        let end_index = end
            .map(|x| x.node_path[target_level].arr)
            .unwrap_or(parent.children.len());
        for (i, child) in parent.children[start_index..end_index]
            .iter_mut()
            .enumerate()
        {
            if g(&mut child.write_buffer, &child.cache) {
                path[target_level] = Idx::new(child.arena, i);
                add_path_to_dirty_map(&path, dirty_map)
            }
        }
    }

    /// Perf: can use dirty mark to speed this up. But it requires a new field in node
    ///
    /// Alternatively, we can reduce the usage of this method
    fn recursive_flush_buffer(&mut self, parent_idx: ArenaIndex) {
        let parent = self.nodes.get_mut(parent_idx).unwrap();
        if parent.is_leaf() {
            return;
        }

        let mut children = core::mem::take(&mut parent.children);
        for (i, child) in children.iter_mut().enumerate() {
            if let Some(buffer) = core::mem::take(&mut child.write_buffer) {
                self.apply_child_buffer(
                    buffer,
                    Idx {
                        arena: child.arena,
                        arr: i,
                    },
                )
            }
        }

        for child in children.iter() {
            self.recursive_flush_buffer(child.arena);
        }
        let parent = self.nodes.get_mut(parent_idx).unwrap();
        parent.children = children;
    }

    /// Apply the write buffer all the way down the path
    fn flush_path(&mut self, path: PathRef) {
        // root cannot have write buffer
        for i in 0..path.len() - 1 {
            let parent = path[i];
            let child_idx = path[i + 1];
            let parent = self.get_mut(parent.arena);
            if let Some(buffer) = core::mem::take(&mut parent.children[child_idx.arr].write_buffer)
            {
                self.apply_child_buffer(buffer, child_idx);
            }
        }
    }

    fn apply_child_buffer(&mut self, write_buffer: B::WriteBuffer, child_idx: Idx) {
        let child = self.nodes.get_mut(child_idx.arena);
        let child = child.unwrap();
        if child.is_internal() {
            B::apply_write_buffer_to_nodes(&mut child.children, &write_buffer)
        } else {
            B::apply_write_buffer_to_elements(&mut child.elements, &write_buffer)
        }
    }

    fn get_slice(
        &mut self,
        path: &NodePath,
        start_leaf: &Idx,
        start: &QueryResult,
        end_leaf: &Idx,
        end: &QueryResult,
    ) -> MutElemArrSlice<<B as BTreeTrait>::Elem> {
        let current_leaf = path.last().unwrap();
        let idx = *path.last().unwrap();
        let node = self.get_mut(idx.arena);
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

    fn update_dirty_cache_map(&mut self, dirty_map: DirtyMap) {
        let mut dirty_set: StackVec<((isize, ArenaIndex), _)> = dirty_map.into_iter().collect();
        dirty_set.sort_unstable_by_key(|x| -x.0 .0);
        let mut diff_map: FxHashMap<ArenaIndex, B::CacheDiff> = FxHashMap::default();
        for ((_, parent_idx), children) in dirty_set {
            for Idx { arena, arr } in children {
                let (child, parent) = self.get2_mut(arena, parent_idx);
                let cache_diff =
                    child.calc_cache(&mut parent.children[arr].cache, diff_map.remove(&arena));

                if let Some(e) = diff_map.get_mut(&parent_idx) {
                    B::merge_cache_diff(e, &cache_diff);
                } else {
                    diff_map.insert(parent_idx, cache_diff);
                }
            }
        }

        self.nodes
            .get(self.root)
            .unwrap()
            .calc_cache(&mut self.root_cache, None);
    }

    pub fn flush_write_buffer(&mut self) {
        if self.need_flush {
            self.recursive_flush_buffer(self.root);
            self.need_flush = false;
        }
    }

    pub fn iter_flushed(&mut self) -> impl Iterator<Item = &B::Elem> + '_ {
        self.recursive_flush_buffer(self.root);
        self.iter()
    }

    pub fn iter(&self) -> impl Iterator<Item = &B::Elem> + '_ {
        let mut path = self.first_path().unwrap_or(SmallVec::new());
        let idx = path.last().copied().unwrap_or(Idx::new(self.root, 0));
        let node = self.get(idx.arena);
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
                    let node = self.get(idx.arena);
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

    #[inline]
    pub fn range<Q>(&self, range: Range<Q::QueryArg>) -> Range<QueryResult>
    where
        Q: Query<B>,
    {
        self.query::<Q>(&range.start)..self.query::<Q>(&range.end)
    }

    /// TODO: performance can be optimized by only unloading the related buffer
    pub fn iter_flushed_range(
        &mut self,
        range: Range<QueryResult>,
    ) -> impl Iterator<Item = ElemSlice<'_, B::Elem>> + '_ {
        self.recursive_flush_buffer(self.root);
        self.iter_range(range)
    }

    pub fn iter_range(
        &self,
        range: Range<QueryResult>,
    ) -> impl Iterator<Item = ElemSlice<'_, B::Elem>> + '_ {
        let start = range.start;
        let end = range.end;
        let mut node_iter = iter::Iter::new(self, start.node_path.clone(), end.node_path.clone());
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
                        let (start_idx, start_offset) =
                            if idx.arena == start.node_path.last().unwrap().arena {
                                (start.elem_index, Some(start.offset))
                            } else {
                                (0, None)
                            };
                        let (end_idx, end_offset) =
                            if idx.arena == end.node_path.last().unwrap().arena {
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
                                    node_path: path.clone(),
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

    // at call site the cache at path can be out-of-date.
    // the cache will be up-to-date after this method
    fn split(&mut self, path: PathRef) {
        let node = self.nodes.get_mut(path.this().arena).unwrap();
        let mut right: Node<B> = Node {
            parent: node.parent,
            parent_slot: u32::MAX,
            elements: Vec::new(),
            children: Vec::new(),
        };

        // TODO: Perf remove this vec?
        let mut right_children = Vec::new();
        // split
        if node.is_internal() {
            let split = node.children.len() / 2;
            right.children = node.children.split_off(split);
            right_children = right.children.clone();
        } else {
            let split = node.elements.len() / 2;
            right.elements = node.elements.split_off(split);
        }

        // update cache
        let mut right_cache = B::Cache::default();
        right.calc_cache(&mut right_cache, None);
        let right_arena_idx = self.nodes.insert(right);
        let this_cache = {
            let node = path.this().arena;
            let node = self.get_mut(node);
            let mut cache = Default::default();
            node.calc_cache(&mut cache, None);
            cache
        };
        for (i, child) in right_children.into_iter().enumerate() {
            let child = self.get_mut(child.arena);
            child.parent = Some(right_arena_idx);
            child.parent_slot = i as u32;
        }

        self.inner_insert_node(
            path.parent_path(),
            path.this().arr,
            this_cache,
            Child {
                arena: right_arena_idx,
                cache: right_cache,
                write_buffer: Default::default(),
            },
        );
        // don't need to recursive update cache
    }

    // call site should ensure the cache is up-to-date after this method
    fn inner_insert_node(
        &mut self,
        parent_path: PathRef,
        index: usize,
        new_cache: B::Cache,
        node: Child<B>,
    ) {
        if parent_path.is_empty() {
            self.split_root(new_cache, node);
        } else {
            let parent_index = *parent_path.last().unwrap();
            let parent = self.get_mut(parent_index.arena);
            parent.children[index].cache = new_cache;
            parent.children.insert(index + 1, node);
            let is_full = parent.is_full();
            self.update_children_parent_slot_from(parent_index.arena, index + 1);
            if is_full {
                self.split(parent_path);
            }
        }
    }

    fn update_children_parent_slot_from(&mut self, parent: ArenaIndex, index: usize) {
        let parent = self.get_mut(parent);
        // TODO: Perf remove this to_vec?
        let children = parent.children[index..].to_vec();
        for (i, child) in children.iter().enumerate() {
            let idx = index + i;
            let child = self.get_mut(child.arena);
            child.parent_slot = idx as u32;
        }
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

        // update left's children's parent
        for child in left_children {
            self.get_mut(child.arena).parent = Some(left_arena);
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
    fn get(&self, index: ArenaIndex) -> &Node<B> {
        self.nodes.get(index).unwrap()
    }

    /// merge into or borrow from neighbor
    ///
    /// - cache should be up-to-date when calling this.
    ///
    /// return is parent lack
    fn handle_lack(&mut self, node_idx: ArenaIndex) -> bool {
        if self.root == node_idx {
            return false;
        }

        let node = self.get(node_idx);
        let parent_idx = node.parent.unwrap();
        let parent = self.get(parent_idx);
        assert_eq!(parent.children[node.parent_slot as usize].arena, node_idx,);
        let ans = match self
            .pair_neighbor(parent_idx, Idx::new(node_idx, node.parent_slot as usize))
        {
            Some((a_idx, b_idx)) => {
                let parent = self.get_mut(parent_idx);
                let mut a_cache = std::mem::take(&mut parent.children[a_idx.arr].cache);
                let mut b_cache = std::mem::take(&mut parent.children[b_idx.arr].cache);
                let mut re_parent = FxHashMap::default();

                let (a, b) = self.nodes.get2_mut(a_idx.arena, b_idx.arena);
                let a = a.unwrap();
                let b = b.unwrap();
                let is_lack = if a.len() + b.len() >= B::MAX_LEN {
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
                    parent.is_lack()
                } else {
                    // merge
                    if node_idx == a_idx.arena {
                        // merge b to a, delete b
                        if a.is_internal() {
                            for (i, child) in b.children.iter().enumerate() {
                                re_parent.insert(child.arena, (a_idx.arena, a.children.len() + i));
                            }
                            a.children.append(&mut b.children);
                        } else {
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
                    }
                };

                for (child, (parent, slot)) in re_parent {
                    let child = self.get_mut(child);
                    child.parent = Some(parent);
                    child.parent_slot = slot as u32;
                }
                is_lack
            }
            None => true,
        };
        ans
    }

    fn try_reduce_levels(&mut self) {
        let mut reduced = false;
        while self.get(self.root).children.len() == 1 {
            let root = self.get(self.root);
            let child_arena = root.children[0].arena;
            let child = self.nodes.remove(child_arena).unwrap();
            let root = self.get_mut(self.root);
            let _ = core::mem::replace(root, child);
            reduced = true;
            // root cache should be the same as child cache because there is only one child
        }
        if reduced {
            let root = self.get_mut(self.root);
            root.parent = None;
            root.parent_slot = u32::MAX;
            let children = root.children.clone();
            let root_idx = self.root;
            for child in children {
                let child = self.get_mut(child.arena);
                child.parent = Some(root_idx);
            }
        }
    }

    fn pair_neighbor(&self, parent: ArenaIndex, mut this_pos: Idx) -> Option<(Idx, Idx)> {
        let parent = self.get(parent);
        if this_pos.arr >= parent.children.len()
            || parent.children[this_pos.arr].arena != this_pos.arena
        {
            // need to search correct this_pos.arr
            let Some(x) = parent
                .children
                .iter()
                .position(|x| x.arena == this_pos.arena) else { return None };
            this_pos.arr = x;
        }

        if this_pos.arr == 0 {
            parent
                .children
                .get(1)
                .map(|x| (this_pos, Idx::new(x.arena, 1)))
        } else {
            parent
                .children
                .get(this_pos.arr - 1)
                .map(|x| (Idx::new(x.arena, this_pos.arr - 1), this_pos))
        }
    }

    fn recursive_update_cache(&mut self, path: PathRef) {
        let mut diff = None;
        for (parent_idx, this_idx) in path.parent_path().iter().rev().zip(path.iter().rev()) {
            let (parent, this) = self.get2_mut(parent_idx.arena, this_idx.arena);
            diff = Some(this.calc_cache(&mut parent.children[this_idx.arr].cache, diff));
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
            for x in node.children.iter() {
                stack.push(x.arena);
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
        let parent = self.get(parent_idx.arena);
        match parent.children.get(this_idx.arr + 1) {
            Some(next) => {
                path[depth - 1] = Idx::new(next.arena, this_idx.arr + 1);
            }
            None => {
                if !self.next_sibling(&mut path[..depth - 1]) {
                    return false;
                }

                let parent = self.get(path[depth - 2].arena);
                path[depth - 1] = Idx::new(parent.children[0].arena, 0);
            }
        }

        true
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
        let parent = self.get(parent_idx.arena);
        if this_idx.arr >= 1 {
            let prev = &parent.children[this_idx.arr - 1];
            path[depth - 1] = Idx::new(prev.arena, this_idx.arr - 1);
        } else {
            if !self.prev_sibling(&mut path[..depth - 1]) {
                return false;
            }

            let parent = self.get(path[depth - 2].arena);
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
            let node = self.get(node_idx);
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
}

fn add_path_to_dirty_map(path: &[Idx], dirty_map: &mut DirtyMap) {
    let mut current = path.last().unwrap();
    let mut parent_level = path.len() as isize - 2;
    while parent_level >= 0 {
        let parent = path.get(parent_level as usize).unwrap();
        if let Some(arr) = dirty_map.get_mut(&(parent_level, parent.arena)) {
            arr.push(*current);
            // parent is already created by others, so all ancestors must also be created
            break;
        } else {
            dirty_map.insert((parent_level, parent.arena), vec![*current]);
            current = parent;
            parent_level -= 1;
        }
    }
}

impl<B: BTreeTrait> BTree<B> {
    #[allow(unused)]
    pub fn check(&self) {
        // check cache
        for (index, node) in self.nodes.iter() {
            if node.is_internal() {
                assert!(!node.is_empty());
                for (i, child_info) in node.children.iter().enumerate() {
                    let child = self.get(child_info.arena);
                    let mut cache = Default::default();
                    child.calc_cache(&mut cache, None);
                    assert_eq!(child.parent_slot, i as u32);
                    assert_eq!(child.parent, Some(index));
                    assert_eq!(cache, child_info.cache);
                }
            }
            if let Some(parent) = node.parent {
                let parent = self.get(parent);
                assert_eq!(parent.children[node.parent_slot as usize].arena, index);
            }

            // FIXME: enable these checking when we have parent link
            // if index != self.root {
            //     assert!(!node.is_lack(), "len={}\n", node.len());
            // }

            // assert!(!node.is_full(), "len={}", node.len());
        }

        // TODO: check leaf at same level
        // TODO: check custom invariants
        // TODO: check purge works correctly
    }
}

impl<B: BTreeTrait> Default for BTree<B> {
    fn default() -> Self {
        Self::new()
    }
}
