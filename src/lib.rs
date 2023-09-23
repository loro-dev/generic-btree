#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

use core::{
    fmt::Debug,
    ops::{Deref, Range},
};
use std::{cmp::Ordering, mem::take, ops::RangeBounds};

use fxhash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use thunderdome::Arena;
use thunderdome::Index as RawArenaIndex;

pub use event::{MoveEvent, MoveListener};
pub use generic_impl::*;

use crate::rle::{HasLength, Mergeable, Sliceable};

mod event;
mod generic_impl;
mod iter;

pub mod rle;

pub type SmallElemVec<T> = SmallVec<[T; 8]>;
pub type StackVec<T> = SmallVec<[T; 8]>;
pub type HeapVec<T> = Vec<T>;

///
/// `Elem` should has length. `offset` in search result should always >= `Elem.rle_len()`
pub trait BTreeTrait {
    type Elem: Debug + HasLength + Sliceable + Mergeable;
    type Cache: Debug + Default + Clone + Eq;
    type CacheDiff: Debug;
    /// Use () if you don't need write buffer.
    /// Associated type default is still unstable so we don't provide default value.
    const MAX_LEN: usize;

    /// If diff.is_some, return value should be some too
    fn calc_cache_internal(
        cache: &mut Self::Cache,
        caches: &[Child<Self>],
        diff: Option<Self::CacheDiff>,
    ) -> Option<Self::CacheDiff>;
    fn merge_cache_diff(diff1: &mut Self::CacheDiff, diff2: &Self::CacheDiff);
    fn get_elem_cache(elem: &Self::Elem) -> Self::Cache;
}

pub trait Query<B: BTreeTrait> {
    type QueryArg: Clone;

    fn init(target: &Self::QueryArg) -> Self;

    fn find_node(&mut self, target: &Self::QueryArg, child_caches: &[Child<B>]) -> FindResult;

    /// Confirm the search result and returns (offset, found)
    ///
    /// If elem is not target, `found=false`
    fn confirm_elem(&self, q: &Self::QueryArg, elem: &B::Elem) -> (usize, bool);
}

pub struct BTree<B: BTreeTrait> {
    /// internal nodes
    in_nodes: Arena<Node<B>>,
    /// leaf nodes
    leaf_nodes: Arena<LeafNode<B::Elem>>,
    // root is always internal nodes
    root: ArenaIndex,
    root_cache: B::Cache,
    element_move_listener: Option<MoveListener<B::Elem>>,
}

impl<Elem: Clone, B: BTreeTrait<Elem=Elem>> Clone for BTree<B> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            in_nodes: self.in_nodes.clone(),
            leaf_nodes: self.leaf_nodes.clone(),
            root: self.root,
            root_cache: self.root_cache.clone(),
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
    pub leaf: LeafIndex,
    pub offset: usize,
    pub found: bool,
}

/// Exposed arena index
///
/// Only exposed arena index of leaf node.
///
///
#[repr(transparent)]
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct LeafIndex(RawArenaIndex);

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
enum ArenaIndex {
    Leaf(RawArenaIndex),
    Internal(RawArenaIndex),
}

impl ArenaIndex {
    fn unwrap(self) -> RawArenaIndex {
        match self {
            ArenaIndex::Leaf(x) => x,
            ArenaIndex::Internal(x) => x,
        }
    }

    fn unwrap_leaf(self) -> RawArenaIndex {
        match self {
            ArenaIndex::Leaf(x) => x,
            ArenaIndex::Internal(_) => panic!("unwrap_leaf on internal node"),
        }
    }

    fn unwrap_internal(self) -> RawArenaIndex {
        match self {
            ArenaIndex::Leaf(_) => panic!("unwrap_internal on leaf node"),
            ArenaIndex::Internal(x) => x,
        }
    }
}

impl From<LeafIndex> for ArenaIndex {
    fn from(value: LeafIndex) -> Self {
        Self::Leaf(value.0)
    }
}

impl From<RawArenaIndex> for LeafIndex {
    fn from(value: RawArenaIndex) -> Self {
        Self(value)
    }
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
    pub fn elem<'b, Elem: Debug, B: BTreeTrait<Elem=Elem>>(
        &self,
        tree: &'b BTree<B>,
    ) -> Option<&'b Elem> {
        tree.leaf_nodes.get(self.leaf.0).map(|x| &x.elem)
    }
}

#[derive(Debug, Clone)]
struct LeafNode<Elem> {
    elem: Elem,
    parent: RawArenaIndex,
}

impl<T> LeafNode<T> {
    fn parent(&self) -> ArenaIndex {
        ArenaIndex::Internal(self.parent)
    }
}

impl<T: Sliceable> LeafNode<T> {
    fn split(&mut self, offset: usize) -> Self {
        let new_elem = self.elem.slice(offset..);
        self.elem.slice_(0..offset);
        Self {
            elem: new_elem,
            parent: self.parent,
        }
    }
}

// TODO: use enum to save spaces
struct Node<B: BTreeTrait> {
    parent: Option<ArenaIndex>,
    parent_slot: u8,
    children: HeapVec<Child<B>>,
    is_child_leaf: bool,
}

impl<Cache: Debug, Elem: Debug, B: BTreeTrait<Elem=Elem, Cache=Cache>> Debug for BTree<B> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        fn fmt_node<Cache: Debug, Elem: Debug, B: BTreeTrait<Elem=Elem>>(
            tree: &BTree<B>,
            node_idx: ArenaIndex,
            f: &mut core::fmt::Formatter<'_>,
            indent_size: usize,
        ) -> core::fmt::Result {
            match node_idx {
                ArenaIndex::Leaf(_) => {}
                ArenaIndex::Internal(_) => {
                    let node = tree.get_internal_node(node_idx);
                    for child in node.children.iter() {
                        indent(f, indent_size)?;
                        if child.is_internal() {
                            let child_node = tree.get_internal_node(child.arena);
                            f.write_fmt(format_args!(
                                "{} Arena({:?}) Cache: {:?}\n",
                                child_node.parent_slot, &child.arena, &child.cache
                            ))?;
                            fmt_node::<Cache, Elem, B>(tree, child.arena, f, indent_size + 1)?;
                        } else {
                            let node = tree.get_leaf_node(child.arena);
                            f.write_fmt(format_args!("Leaf({:?}) Cache: {:?}\n", &node.elem, &child.cache))?;
                        }
                    }
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
        fmt_node::<Cache, Elem, B>(self, self.root, f, 1)
    }
}

impl<Cache: Debug, Elem: Debug, B: BTreeTrait<Elem=Elem, Cache=Cache>> Debug for Node<B> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Node")
            .field("children", &self.children)
            .finish()
    }
}

impl<Cache: Debug, Elem: Debug, B: BTreeTrait<Elem=Elem, Cache=Cache>> Debug for Child<B> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Child")
            .field("index", &self.arena)
            .field("cache", &self.cache)
            .finish()
    }
}

impl<Elem: Clone, B: BTreeTrait<Elem=Elem>> Clone for Node<B> {
    fn clone(&self) -> Self {
        Self {
            parent: self.parent,
            parent_slot: u8::MAX,
            children: self.children.clone(),
            is_child_leaf: self.is_child_leaf,
        }
    }
}

pub struct Child<B: ?Sized + BTreeTrait> {
    arena: ArenaIndex,
    pub cache: B::Cache,
}

impl<B: ?Sized + BTreeTrait> Child<B> {
    #[inline]
    fn is_internal(&self) -> bool {
        matches!(self.arena, ArenaIndex::Internal(_))
    }

    #[inline]
    fn is_leaf(&self) -> bool {
        matches!(self.arena, ArenaIndex::Leaf(_))
    }
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

impl<B: BTreeTrait> Node<B> {
    #[inline(always)]
    pub fn new(is_child_leaf: bool) -> Self {
        Self {
            is_child_leaf,
            parent: None,
            parent_slot: u8::MAX,
            children: HeapVec::with_capacity(B::MAX_LEN),
        }
    }

    #[inline(always)]
    pub fn is_full(&self) -> bool {
        self.children.len() >= B::MAX_LEN
    }

    #[inline(always)]
    pub fn is_lack(&self) -> bool {
        self.children.len() < B::MAX_LEN / 2
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.children.len()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn has_leaves(&self) -> bool {
        self.is_child_leaf
    }

    /// if diff is not provided, the cache will be calculated from scratch
    #[inline(always)]
    fn calc_cache(&self, cache: &mut B::Cache, diff: Option<B::CacheDiff>) -> Option<B::CacheDiff> {
        B::calc_cache_internal(cache, &self.children, diff)
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
        let root = arena.insert(Node::new(true));
        Self {
            in_nodes: arena,
            leaf_nodes: Arena::new(),
            root: ArenaIndex::Internal(root),
            root_cache: B::Cache::default(),
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
        self.in_nodes.len()
    }

    #[inline]
    pub fn insert<Q>(&mut self, tree_index: &Q::QueryArg, data: B::Elem)
        where
            Q: Query<B>,
    {
        let Some(result) = self.query::<Q>(tree_index) else {
            self.push(data);
            return;
        };
        let index = result.leaf;
        let leaf = self.leaf_nodes.get_mut(index.0).unwrap();
        let parent_idx = leaf.parent();

        let mut is_full = false;
        // Try to merge
        if result.offset == 0 && data.can_merge(&leaf.elem) {
            leaf.elem.merge_left(&data);
            let leaf1 = Some(index);
            if let Some(listener) = self.element_move_listener.as_ref() {
                listener(MoveEvent {
                    target_leaf: leaf1,
                    elem: &data,
                });
            }
        } else if result.offset == leaf.elem.rle_len() && leaf.elem.can_merge(&data) {
            leaf.elem.merge_right(&data);
            let leaf1 = Some(index);
            if let Some(listener) = self.element_move_listener.as_ref() {
                listener(MoveEvent {
                    target_leaf: leaf1,
                    elem: &data,
                });
            }
        } else {
            // Insert new leaf node
            let child = self.alloc_leaf_child(data, parent_idx.unwrap());
            let (_, parent, _, insert_index) = self.split_leaf_if_needed(result);
            parent.children.insert(insert_index, child);
            is_full = parent.is_full();
        }

        self.recursive_update_cache(result.leaf.into(), true, None);
        if is_full {
            self.split(parent_idx);
        }
    }

    fn alloc_leaf_child(
        &mut self,
        data: <B as BTreeTrait>::Elem,
        parent_index: RawArenaIndex,
    ) -> Child<B> {
        let elem_cache = B::get_elem_cache(&data);
        let new_leaf_index = self.alloc_new_leaf(LeafNode {
            elem: data,
            parent: parent_index,
        });
        let child = Child {
            arena: new_leaf_index,
            cache: elem_cache,
        };
        child
    }

    /// Split a leaf node at offset if it's not the start/end of the leaf node.
    ///
    /// This method should be called when inserting at target pos.
    ///
    /// Returns new_pos, the parent node, the arena index of the parent node, and the insert pos in parent children.
    fn split_leaf_if_needed(&mut self, pos: QueryResult) -> (Option<QueryResult>, &mut Node<B>, RawArenaIndex, usize) {
        // FIXME: notify leaf move
        let leaf = self.leaf_nodes.get_mut(pos.leaf.0).unwrap();
        let parent_idx = leaf.parent;
        let parent = self.in_nodes.get_mut(leaf.parent).unwrap();
        let mut new_pos = Some(pos);
        let leaf_slot = parent
            .children
            .iter()
            .position(|x| x.arena.unwrap() == pos.leaf.0)
            .unwrap();
        let insert_pos = if pos.offset == 0 {
            leaf_slot
        } else if pos.offset == leaf.elem.rle_len() {
            if leaf_slot + 1 < parent.children.len() {
                new_pos = Some(QueryResult {
                    leaf: parent.children[leaf_slot + 1].arena.unwrap().into(),
                    offset: 0,
                    found: true,
                });
            } else {
                new_pos = self.next_elem(pos);
            }
            leaf_slot + 1
        } else {
            assert!(pos.offset < leaf.elem.rle_len(), "elem.rle_len={} but pos.offset={} Elem:{:?}", leaf.elem.rle_len(), pos.offset, &leaf.elem);
            let new_leaf = leaf.split(pos.offset);
            let cache = B::get_elem_cache(&new_leaf.elem);
            // alloc new leaf node
            let leaf_arena_index = {
                let arena_index = self.leaf_nodes.insert(new_leaf);
                let leaf = self.leaf_nodes.get(arena_index).unwrap();
                let elem = &leaf.elem;
                if let Some(listener) = self.element_move_listener.as_ref() {
                    listener(MoveEvent {
                        target_leaf: Some(arena_index.into()),
                        elem,
                    });
                }
                ArenaIndex::Leaf(arena_index)
            };
            new_pos = Some(QueryResult {
                leaf: leaf_arena_index.unwrap().into(),
                offset: 0,
                found: true,
            });
            parent.children.insert(
                leaf_slot + 1,
                Child {
                    arena: leaf_arena_index,
                    cache,
                },
            );
            leaf_slot + 1
        };
        let parent = self.in_nodes.get_mut(parent_idx).unwrap();
        (new_pos, parent, parent_idx, insert_pos)
    }

    fn alloc_new_leaf(&mut self, leaf: LeafNode<B::Elem>) -> ArenaIndex {
        let arena_index = self.leaf_nodes.insert(leaf);
        let leaf = self.leaf_nodes.get(arena_index).unwrap();
        let elem = &leaf.elem;
        if let Some(listener) = self.element_move_listener.as_ref() {
            listener(MoveEvent {
                target_leaf: Some(arena_index.into()),
                elem,
            });
        }
        ArenaIndex::Leaf(arena_index)
    }

    /// Insert many elements into the tree at once
    ///
    /// It will invoke [`BTreeTrait::insert_batch`]
    ///
    /// NOTE: Currently this method don't guarantee after inserting many elements the tree is
    /// still balance
    pub fn insert_many_by_query_result(&mut self, result: QueryResult, data: Vec<B::Elem>)
        where
            B::Elem: Clone,
    {
        let (_, parent, parent_index, insert_index) = self.split_leaf_if_needed(result);
        let mut children = take(&mut parent.children);
        children.splice(
            insert_index..insert_index,
            data.into_iter().map(|elem| {
                let elem_cache = B::get_elem_cache(&elem);
                let new_leaf_index = self.alloc_new_leaf(LeafNode {
                    elem,
                    parent: parent_index,
                });
                Child {
                    arena: new_leaf_index,
                    cache: elem_cache,
                }
            }),
        );

        let parent = self.in_nodes.get_mut(parent_index).unwrap();
        parent.children = children;
        let is_full = parent.is_full();
        self.recursive_update_cache(result.leaf.into(), true, None);
        if is_full {
            self.split(ArenaIndex::Internal(parent_index));
        }
        // TODO: tree may still be unbalanced
    }

    /// Query the tree by custom query type
    ///
    /// Return None if the tree is empty
    #[inline(always)]
    pub fn query<Q>(&self, query: &Q::QueryArg) -> Option<QueryResult>
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
        let mut leaf = self.leaf_nodes.get(path.leaf.0).unwrap();
        let mut parent_index = leaf.parent;
        let mut parent = self.in_nodes.get(leaf.parent).unwrap();
        let mut elem_index = Self::get_leaf_slot(path.leaf.0, parent);
        path.offset += 1;
        loop {
            if elem_index == parent.children.len() {
                if let Some(next) = self.next_same_level_in_node(ArenaIndex::Internal(parent_index))
                {
                    elem_index = 0;
                    path.offset = 0;
                    parent_index = next.unwrap_internal();
                    parent = self.in_nodes.get(parent_index).unwrap();
                } else {
                    return None;
                }
            }

            let elem = &parent.children[elem_index];
            let leaf = self.leaf_nodes.get(elem.arena.unwrap()).unwrap();
            // skip empty span
            if leaf.elem.rle_len() <= path.offset {
                path.offset -= leaf.elem.rle_len();
                elem_index += 1;
            } else {
                path.leaf = elem.arena.unwrap_leaf().into();
                break;
            }
        }

        Some(path)
    }

    #[inline]
    fn get_leaf_slot(leaf_arena_index: RawArenaIndex, parent: &Node<B>) -> usize {
        parent
            .children
            .iter()
            .position(|x| x.arena.unwrap_leaf() == leaf_arena_index)
            .unwrap()
    }

    pub fn query_with_finder_return<Q>(&self, query: &Q::QueryArg) -> (Option<QueryResult>, Q)
        where
            Q: Query<B>,
    {
        let mut finder = Q::init(query);
        if self.is_empty() {
            return (None, finder);
        }

        let mut node = self.in_nodes.get(self.root.unwrap_internal()).unwrap();
        let mut index = self.root;
        let mut found = true;
        loop {
            let result = finder.find_node(query, &node.children);
            debug_assert!(!node.children.is_empty());
            let i = result.index.min(node.children.len() - 1);
            found = found && result.found;
            index = node.children[i].arena;
            match index {
                ArenaIndex::Leaf(_) => {
                    let (offset, leaf_found) = finder.confirm_elem(
                        query,
                        &self.leaf_nodes.get(index.unwrap_leaf()).unwrap().elem,
                    );
                    return (
                        Some(QueryResult {
                            leaf: index.unwrap_leaf().into(),
                            offset,
                            found: found && leaf_found,
                        }),
                        finder,
                    );
                }
                ArenaIndex::Internal(index) => {
                    node = self.in_nodes.get(index).unwrap();
                }
            }
        }
    }

    #[inline]
    pub fn get_elem_mut(&mut self, leaf: LeafIndex) -> Option<&mut B::Elem> {
        let node = self.leaf_nodes.get_mut(leaf.0)?;
        Some(&mut node.elem)
    }

    pub fn get_elem(&self, leaf: LeafIndex) -> Option<&<B as BTreeTrait>::Elem> {
        self.leaf_nodes.get(leaf.0).map(|x| &x.elem)
    }

    pub fn remove_leaf(&mut self, path: QueryResult) -> Option<B::Elem> {
        let Some(leaf) = self.leaf_nodes.get_mut(path.leaf.0) else {
            return None;
        };
        let parent_idx = leaf.parent();
        let parent = self.in_nodes.get_mut(leaf.parent).unwrap();
        let index = Self::get_leaf_slot(path.leaf.0, parent);
        let child = parent.children.remove(index);
        let is_lack = parent.is_lack();
        let is_empty = parent.is_empty();
        debug_assert_eq!(child.arena.unwrap(), path.leaf.0);
        let elem = self.leaf_nodes.remove(child.arena.unwrap()).unwrap().elem;
        if let Some(listener) = self.element_move_listener.as_ref() {
            listener(MoveEvent {
                target_leaf: None,
                elem: &elem,
            });
        }
        self.recursive_update_cache(parent_idx, true, None);
        if is_empty {
            self.remove_internal_node(parent_idx.unwrap());
        } else if is_lack {
            self.handle_lack(parent_idx);
        }

        Some(elem)
    }

    fn remove_internal_node(&mut self, node: RawArenaIndex) {
        if node == self.root.unwrap() {
            return;
        }

        let node = self.in_nodes.remove(node).unwrap();
        if let Some(parent_idx) = node.parent {
            let parent = self.in_nodes.get_mut(parent_idx.unwrap_internal()).unwrap();
            parent.children.remove(node.parent_slot as usize);
            let is_lack = parent.is_lack();
            let is_empty = parent.is_empty();
            self.update_children_parent_slot_from(parent_idx, node.parent_slot as usize);
            if is_empty {
                self.remove_internal_node(parent_idx.unwrap_internal());
            } else if is_lack {
                self.handle_lack(parent_idx);
            }
        } else {
            // ignore remove root
            unreachable!()
        }
    }

    /// Drain the target range
    ///
    /// The returned value must be dropped, otherwise the tree will be corrupted
    pub fn drain<Q>(&mut self, range: Range<Q::QueryArg>) -> iter::Drain<B>
        where
            Q: Query<B>,
    {
        let from = self.query::<Q>(&range.start);
        let to = self.query::<Q>(&range.end);
        iter::Drain::new(self, from, to)
    }

    /// Update the elements in place
    ///
    /// F should returns `(should_update_cache, cache_diff)`
    ///
    /// This method may break the balance of the tree
    ///
    /// If the given range has zero length, f will still be called, and the slice will
    /// have same `start` and `end` field
    ///
    /// TODO: need better test coverage
    pub fn update<F>(&mut self, range: Range<QueryResult>, f: &mut F)
        where
            F: FnMut(&mut B::Elem) -> (bool, Option<B::CacheDiff>),
    {
        let start = range.start;
        self.split_leaf_node(range.end);
        let end = self.prefer_right(range.end);
        self.split_leaf_node(start);
        let Some(start) = self.prefer_right(start) else { return; };
        let start_leaf = start.leaf;
        let mut path = self.get_path(start_leaf.into());
        let mut dirty_map: LeafDirtyMap<B::CacheDiff> = FxHashMap::default();

        loop {
            let current_leaf = path.last().unwrap();
            if let Some(end) = end {
                if current_leaf.arena.unwrap_leaf() == end.leaf.0 {
                    break;
                }
            }

            let node = self
                .leaf_nodes
                .get_mut(current_leaf.arena.unwrap_leaf())
                .unwrap();
            let (should_update_cache, cache_diff) = f(&mut node.elem);
            if should_update_cache {
                add_leaf_dirty_map(current_leaf.arena, &mut dirty_map, cache_diff);
            }

            if !self.next_sibling(&mut path) {
                break;
            }
        }

        if !dirty_map.is_empty() {
            self.update_dirty_cache_map(dirty_map);
        } else {
            self.in_nodes
                .get(self.root.unwrap_internal())
                .unwrap()
                .calc_cache(&mut self.root_cache, None);
        }
    }

    /// return new leaf arena index
    fn split_leaf_node(&mut self, path: QueryResult) -> Option<ArenaIndex> {
        // this method only changes the children num of parent node
        // caches don't need to be updated
        if path.offset == 0 {
            return None;
        }

        let leaf_idx = path.leaf.0;
        let leaf = self.leaf_nodes.get_mut(leaf_idx).unwrap();
        let parent_idx = leaf.parent;
        let parent = self.in_nodes.get_mut(leaf.parent).unwrap();
        let (leaf_node, index) = (leaf, Self::get_leaf_slot(leaf_idx, parent));
        if leaf_node.elem.rle_len() == path.offset {
            return None;
        }

        let new_leaf = leaf_node.split(path.offset);
        let left_cache = B::get_elem_cache(&leaf_node.elem);
        let right_cache = B::get_elem_cache(&new_leaf.elem);
        let leaf_index = self.alloc_new_leaf(new_leaf);

        let parent = self.in_nodes.get_mut(parent_idx).unwrap();
        parent.children[index].cache = left_cache;
        parent.children.insert(
            index + 1,
            Child {
                arena: leaf_index,
                cache: right_cache,
            },
        );
        Some(leaf_index)
    }

    /// Prefer begin of the next leaf node than end of the current leaf node
    ///
    /// When path.offset == leaf.rle_len(), this method will return
    /// the next leaf node with offset 0
    fn prefer_right(&self, path: QueryResult) -> Option<QueryResult> {
        if path.offset == 0 {
            return Some(path);
        }

        let leaf = self.leaf_nodes.get(path.leaf.0).unwrap();
        if path.offset == leaf.elem.rle_len() {
            self.next_elem(path)
        } else {
            Some(path)
        }
    }

    /// update leaf node's elements, return true if cache need to be updated
    ///
    /// `f` returns (is_cache_updated, cache_diff, new_insert_1, new_insert2)
    pub fn update_leaf(
        &mut self,
        node_idx: LeafIndex,
        f: impl FnOnce(&mut B::Elem) -> (bool, Option<B::CacheDiff>, Option<B::Elem>, Option<B::Elem>),
    ) {
        let node = self.leaf_nodes.get_mut(node_idx.0).unwrap();
        let parent_idx = node.parent();
        let (need_update_cache, diff, new_insert_1, new_insert_2) = f(&mut node.elem);

        if need_update_cache {
            self.recursive_update_cache(node_idx.into(), true, diff);
        }

        if !new_insert_1.is_some() {
            return;
        }

        let new: SmallVec<[_; 2]> = new_insert_1
            .into_iter()
            .chain(new_insert_2.into_iter())
            .map(|elem| self.alloc_leaf_child(elem, parent_idx.unwrap()))
            .collect();

        let parent = self.in_nodes.get_mut(parent_idx.unwrap()).unwrap();
        let is_full = parent.is_full();
        let is_lack = parent.is_lack();
        let slot = Self::get_leaf_slot(node_idx.0, parent);
        parent.children.splice(slot + 1..slot + 1, new);
        if is_full {
            self.split(parent_idx);
        }
        if is_lack {
            self.handle_lack(parent_idx);
        }
    }

    #[inline]
    fn update_root_cache(&mut self) {
        self.in_nodes
            .get(self.root.unwrap_internal())
            .unwrap()
            .calc_cache(&mut self.root_cache, None);
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
                let node = self.in_nodes.get(child_idx.unwrap_internal()).unwrap();
                let Some(parent_idx) = node.parent else {
                    continue;
                };
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

        self.in_nodes
            .get(self.root.unwrap_internal())
            .unwrap()
            .calc_cache(&mut self.root_cache, None);
    }

    /// Removed deleted children. `deleted` means they are removed from the arena.
    fn filter_deleted_children(&mut self, internal_node: ArenaIndex) {
        let node = self
            .in_nodes
            .get_mut(internal_node.unwrap_internal())
            .unwrap();
        // PERF: I hate this pattern...
        let mut children = take(&mut node.children);
        children.retain(|x| match x.arena {
            ArenaIndex::Leaf(leaf) => self.leaf_nodes.contains(leaf),
            ArenaIndex::Internal(index) => self.in_nodes.contains(index),
        });
        let node = self
            .in_nodes
            .get_mut(internal_node.unwrap_internal())
            .unwrap();
        node.children = children;
    }

    pub fn iter(&self) -> impl Iterator<Item=&B::Elem> + '_ {
        let mut path = self.first_path().unwrap_or_default();
        path.pop();
        let idx = path.last().copied().unwrap_or(Idx::new(self.root, 0));
        debug_assert!(matches!(idx.arena, ArenaIndex::Internal(_)));
        let node = self.get_internal_node(idx.arena);
        let mut iter = node.children.iter();
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
                    debug_assert!(matches!(idx.arena, ArenaIndex::Internal(_)));
                    let node = self.get_internal_node(idx.arena);
                    iter = node.children.iter();
                }
                Some(elem) => {
                    let leaf = self.leaf_nodes.get(elem.arena.unwrap_leaf()).unwrap();
                    return Some(&leaf.elem);
                }
            }
        })
    }

    fn first_path(&self) -> Option<NodePath> {
        let mut index = self.root;
        let mut node = self.in_nodes.get(index.unwrap_internal()).unwrap();
        if node.is_empty() {
            return None;
        }

        let mut path = NodePath::new();
        loop {
            path.push(Idx::new(index, 0));
            match index {
                ArenaIndex::Leaf(_) => {
                    break;
                }
                ArenaIndex::Internal(_) => {
                    index = node.children[0].arena;
                    if let ArenaIndex::Internal(i) = index {
                        node = self.in_nodes.get(i).unwrap();
                    };
                }
            }
        }

        Some(path)
    }

    fn last_path(&self) -> Option<NodePath> {
        let mut path = NodePath::new();
        let mut index = self.root;
        let mut node = self.in_nodes.get(index.unwrap_internal()).unwrap();
        let mut pos_in_parent = 0;
        if node.is_empty() {
            return None;
        }

        loop {
            path.push(Idx::new(index, pos_in_parent));
            match index {
                ArenaIndex::Leaf(_) => {
                    break;
                }
                ArenaIndex::Internal(_) => {
                    pos_in_parent = node.children.len() - 1;
                    index = node.children[node.children.len() - 1].arena;
                    if let ArenaIndex::Internal(i) = index {
                        node = self.in_nodes.get(i).unwrap();
                    }
                }
            }
        }

        Some(path)
    }

    pub fn first_leaf(&self) -> ArenaIndex {
        let mut index = self.root;
        let mut node = self.in_nodes.get(index.unwrap_internal()).unwrap();
        loop {
            index = node.children[0].arena;
            if matches!(index, ArenaIndex::Leaf(_)) {
                return index;
            };

            node = self.in_nodes.get(index.unwrap_internal()).unwrap();
        }
    }

    pub fn last_leaf(&self) -> ArenaIndex {
        let mut index = self.root;
        let mut node = self.in_nodes.get(index.unwrap_internal()).unwrap();
        loop {
            index = node.children[node.children.len() - 1].arena;
            if matches!(index, ArenaIndex::Leaf(_)) {
                return index;
            };

            node = self.in_nodes.get(index.unwrap_internal()).unwrap();
        }
    }

    #[inline]
    pub fn range<Q>(&self, range: Range<Q::QueryArg>) -> Option<Range<QueryResult>>
        where
            Q: Query<B>,
    {
        if self.is_empty() {
            return None;
        }

        Some(self.query::<Q>(&range.start).unwrap()..self.query::<Q>(&range.end).unwrap())
    }

    pub fn iter_range(
        &self,
        range: impl RangeBounds<QueryResult>,
    ) -> impl Iterator<Item=ElemSlice<'_, B::Elem>> + '_ {
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
    ) -> impl Iterator<Item=ElemSlice<'_, B::Elem>> + '_ {
        let node_iter = iter::Iter::new(
            self,
            self.get_path(start.leaf.into()),
            self.get_path(end.leaf.into()),
        );
        node_iter.map(move |(path, node)| {
            let leaf = LeafIndex(path.last().unwrap().arena.unwrap_leaf());
            ElemSlice {
                path: QueryResult {
                    leaf,
                    offset: 0,
                    found: true,
                },
                elem: &node.elem,
                start: if start.leaf == leaf {
                    Some(start.offset)
                } else {
                    None
                },
                end: if end.leaf == leaf {
                    Some(end.offset)
                } else {
                    None
                },
            }
        })
    }

    pub fn first_full_path(&self) -> QueryResult {
        QueryResult {
            leaf: self.first_leaf().unwrap_leaf().into(),
            offset: 0,
            found: false,
        }
    }

    pub fn last_full_path(&self) -> QueryResult {
        let leaf = self.last_leaf();
        QueryResult {
            leaf: leaf.unwrap_leaf().into(),
            offset: 0,
            found: false,
        }
    }

    // at call site the cache at path can be out-of-date.
    // the cache will be up-to-date after this method
    fn split(&mut self, node_idx: ArenaIndex) {
        let node = self.in_nodes.get_mut(node_idx.unwrap_internal()).unwrap();
        let node_parent = node.parent;
        let node_parent_slot = node.parent_slot;
        let right: Node<B> = Node {
            parent: node.parent,
            parent_slot: u8::MAX,
            children: Vec::new(),
            is_child_leaf: node.is_child_leaf,
        };

        // split
        let split = node.children.len() / 2;
        let mut right_children = node.children.split_off(split);

        // update cache
        let mut right_cache = B::Cache::default();
        if right_children.is_empty() {
            right.calc_cache(&mut right_cache, None);
        }
        let right_arena_idx = self.in_nodes.insert(right);
        let this_cache = {
            let node = self.get_internal_mut(node_idx);
            let mut cache = Default::default();
            node.calc_cache(&mut cache, None);
            cache
        };

        if !right_children.is_empty() {
            // update children's parent info
            for (i, child) in right_children.iter().enumerate() {
                if matches!(child.arena, ArenaIndex::Internal(_)) {
                    let child = self.get_internal_mut(child.arena);
                    child.parent = Some(ArenaIndex::Internal(right_arena_idx));
                    child.parent_slot = i as u8;
                } else {
                    self.get_leaf_mut(child.arena).parent = right_arena_idx;
                }
            }
        }

        let right = self.in_nodes.get_mut(right_arena_idx).unwrap();
        right.children = right_children;
        // update parent cache
        right.calc_cache(&mut right_cache, None);

        self.inner_insert_node(
            node_parent,
            node_parent_slot as usize,
            this_cache,
            Child {
                arena: ArenaIndex::Internal(right_arena_idx),
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
            let parent = self.get_internal_mut(parent_idx);
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

    /// Update the `parent_slot` fields in `children[index..]`
    fn update_children_parent_slot_from(&mut self, parent_idx: ArenaIndex, index: usize) {
        let parent = self.get_internal_mut(parent_idx);
        // PERF: Is there a way to avoid `take` like this?
        let children = take(&mut parent.children);
        for (i, child) in children[index..].iter().enumerate() {
            let idx = index + i;
            let child = self.get_internal_mut(child.arena);
            child.parent_slot = idx as u8;
        }
        let parent = self.get_internal_mut(parent_idx);
        parent.children = children;
    }

    /// right's cache should be up-to-date
    fn split_root(&mut self, new_cache: B::Cache, right: Child<B>) {
        let root_idx = self.root;
        // set right parent
        let right_node = &mut self.get_internal_mut(right.arena);
        right_node.parent_slot = 1;
        right_node.parent = Some(root_idx);
        let root = self.get_internal_mut(self.root);
        // let left be root
        let mut left_node: Node<B> = core::mem::replace(
            root,
            Node {
                parent: None,
                parent_slot: 0,
                children: vec![],
                is_child_leaf: false,
            },
        );
        left_node.parent_slot = 0;
        // set left parent
        left_node.parent = Some(root_idx);

        // push left and right to root.children
        root.children = Vec::with_capacity(B::MAX_LEN);
        let left_children = left_node.children.clone();
        let left_arena = self.in_nodes.insert(left_node);
        let left = Child::new(ArenaIndex::Internal(left_arena), new_cache);
        let mut cache = std::mem::take(&mut self.root_cache);
        let root = self.get_internal_mut(self.root);
        root.children.push(left);
        root.children.push(right);

        // update new root cache
        root.calc_cache(&mut cache, None);

        for child in left_children {
            if child.is_internal() {
                self.get_internal_mut(child.arena).parent = Some(ArenaIndex::Internal(left_arena));
            } else {
                self.get_leaf_mut(child.arena).parent = left_arena;
            }
        }

        self.root_cache = cache;
    }

    #[inline(always)]
    fn get_internal_mut(&mut self, index: ArenaIndex) -> &mut Node<B> {
        self.in_nodes.get_mut(index.unwrap_internal()).unwrap()
    }

    #[inline(always)]
    fn get_leaf_mut(&mut self, index: ArenaIndex) -> &mut LeafNode<B::Elem> {
        self.leaf_nodes.get_mut(index.unwrap_leaf()).unwrap()
    }

    #[inline(always)]
    fn get2_mut(&mut self, a: ArenaIndex, b: ArenaIndex) -> (&mut Node<B>, &mut Node<B>) {
        let (a, b) = self
            .in_nodes
            .get2_mut(a.unwrap_internal(), b.unwrap_internal());
        (a.unwrap(), b.unwrap())
    }

    /// # Panic
    ///
    /// If the given index is not valid or deleted
    #[inline(always)]
    fn get_internal_node(&self, index: ArenaIndex) -> &Node<B> {
        self.in_nodes.get(index.unwrap_internal()).unwrap()
    }

    #[inline(always)]
    fn get_leaf_node(&self, index: ArenaIndex) -> &LeafNode<B::Elem> {
        self.leaf_nodes.get(index.unwrap_leaf()).unwrap()
    }

    #[inline(always)]
    fn get_in_node_safe(&self, index: ArenaIndex) -> Option<&Node<B>> {
        self.in_nodes.get(index.unwrap_internal())
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

        let node = self.get_internal_node(node_idx);
        let parent_idx = node.parent.unwrap();
        let parent = self.get_internal_node(parent_idx);
        debug_assert_eq!(parent.children[node.parent_slot as usize].arena, node_idx, );
        let ans = match self.pair_neighbor(node_idx) {
            Some((a_idx, b_idx)) => {
                let parent = self.get_internal_mut(parent_idx);
                let mut a_cache = std::mem::take(&mut parent.children[a_idx.arr].cache);
                let mut b_cache = std::mem::take(&mut parent.children[b_idx.arr].cache);
                let mut re_parent = FxHashMap::default();

                let (a, b) = self
                    .in_nodes
                    .get2_mut(a_idx.arena.unwrap_internal(), b_idx.arena.unwrap_internal());
                let a = a.unwrap();
                let b = b.unwrap();
                let ans = if a.len() + b.len() >= B::MAX_LEN {
                    // move
                    if a.len() < b.len() {
                        // move part of b's children to a
                        let move_len = (b.len() - a.len()) / 2;
                        for child in b.children.drain(..move_len) {
                            re_parent.insert(child.arena, (a_idx.arena, a.children.len()));
                            a.children.push(child);
                        }
                        for (i, child) in b.children.iter().enumerate() {
                            re_parent.insert(child.arena, (b_idx.arena, i));
                        }
                    } else {
                        // move part of a's children to b
                        let move_len = (a.len() - b.len()) / 2;
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
                    }
                    a.calc_cache(&mut a_cache, None);
                    b.calc_cache(&mut b_cache, None);
                    let parent = self.get_internal_mut(parent_idx);
                    parent.children[a_idx.arr].cache = a_cache;
                    parent.children[b_idx.arr].cache = b_cache;
                    LackInfo {
                        is_parent_lack: parent.is_lack(),
                    }
                } else {
                    // merge
                    let is_parent_lack = if node_idx == a_idx.arena {
                        // merge b to a, delete b
                        for (i, child) in b.children.iter().enumerate() {
                            re_parent.insert(child.arena, (a_idx.arena, a.children.len() + i));
                        }
                        a.children.append(&mut b.children);
                        a.calc_cache(&mut a_cache, None);
                        let parent = self.get_internal_mut(parent_idx);
                        parent.children[a_idx.arr].cache = a_cache;
                        parent.children.remove(b_idx.arr);
                        let is_lack = parent.is_lack();
                        self.purge(b_idx.arena);
                        self.update_children_parent_slot_from(parent_idx, b_idx.arr);
                        is_lack
                    } else {
                        // merge a to b, delete a
                        for (i, child) in a.children.iter().enumerate() {
                            re_parent.insert(child.arena, (b_idx.arena, i));
                        }
                        for (i, child) in b.children.iter().enumerate() {
                            re_parent.insert(child.arena, (b_idx.arena, i + a.children.len()));
                        }
                        b.children.splice(0..0, core::mem::take(&mut a.children));
                        b.calc_cache(&mut b_cache, None);
                        let parent = self.get_internal_mut(parent_idx);
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
                    match child {
                        ArenaIndex::Leaf(_) => {
                            let child = self.get_leaf_mut(child);
                            child.parent = parent.unwrap_internal();
                        }
                        ArenaIndex::Internal(_) => {
                            let child = self.get_internal_mut(child);
                            child.parent = Some(parent);
                            child.parent_slot = slot as u8;
                        }
                    }
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
        while self.get_internal_node(self.root).children.len() == 1 {
            let root = self.get_internal_node(self.root);
            if root.is_child_leaf {
                break;
            }

            let child_arena = root.children[0].arena;
            let child = self.in_nodes.remove(child_arena.unwrap_internal()).unwrap();
            let root = self.get_internal_mut(self.root);
            let _ = core::mem::replace(root, child);
            reduced = true;
            // root cache should be the same as child cache because there is only one child
        }
        if reduced {
            let root_idx = self.root;
            let root = self.get_internal_mut(self.root);
            root.parent = None;
            root.parent_slot = u8::MAX;
            self.reset_children_parent_pointer(root_idx);
        }
    }

    fn reset_children_parent_pointer(&mut self, parent_idx: ArenaIndex) {
        let parent = self.in_nodes.get(parent_idx.unwrap_internal()).unwrap();
        let children = parent.children.clone();
        for child in children {
            match child.arena {
                ArenaIndex::Leaf(_) => {
                    let child = self.get_leaf_mut(child.arena);
                    child.parent = parent_idx.unwrap_internal();
                }
                ArenaIndex::Internal(_) => {
                    let child = self.get_internal_mut(child.arena);
                    child.parent = Some(parent_idx);
                }
            }
        }
    }

    fn pair_neighbor(&self, this: ArenaIndex) -> Option<(Idx, Idx)> {
        let node = self.get_internal_node(this);
        let arr = node.parent_slot as usize;
        let parent = self.get_internal_node(node.parent.unwrap());

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
        mut node_idx: ArenaIndex,
        can_use_diff: bool,
        cache_diff: Option<B::CacheDiff>,
    ) {
        if let ArenaIndex::Leaf(index) = node_idx {
            let leaf = self.leaf_nodes.get(index).unwrap();
            let cache = B::get_elem_cache(&leaf.elem);
            node_idx = leaf.parent();
            let node = self.get_internal_mut(node_idx);
            node.children.iter_mut().find(|x| x.arena.unwrap_leaf() == index).unwrap().cache = cache;
        }

        let mut this_idx = node_idx;
        let mut node = self.get_internal_mut(node_idx);
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
            if let ArenaIndex::Leaf(index) = x {
                if let Some(leaf) = self.leaf_nodes.remove(index) {
                    if let Some(listener) = &mut self.element_move_listener {
                        listener(MoveEvent::new_del(&leaf.elem));
                    }
                }

                continue;
            }

            let Some(node) = self.in_nodes.remove(x.unwrap()) else {
                continue;
            };

            for x in node.children.iter() {
                stack.push(x.arena);
            }
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
        let parent = self.get_internal_node(parent_idx.arena);
        match parent.children.get(this_idx.arr + 1) {
            Some(next) => {
                path[depth - 1] = Idx::new(next.arena, this_idx.arr + 1);
            }
            None => {
                if !self.next_sibling(&mut path[..depth - 1]) {
                    return false;
                }

                let parent = self.get_internal_node(path[depth - 2].arena);
                path[depth - 1] = Idx::new(parent.children[0].arena, 0);
            }
        }

        true
    }

    fn next_same_level_in_node(&self, node_idx: ArenaIndex) -> Option<ArenaIndex> {
        match node_idx {
            ArenaIndex::Leaf(_) => {
                let leaf_idx = node_idx.unwrap_leaf();
                let leaf1 = self.leaf_nodes.get(leaf_idx).unwrap();
                let parent1 = self.get_internal_node(leaf1.parent());
                let (leaf, parent, index) =
                    (leaf1, parent1, Self::get_leaf_slot(leaf_idx, parent1));
                if index + 1 < parent.children.len() {
                    Some(parent.children[index + 1].arena)
                } else if let Some(parent_next) = self.next_same_level_in_node(leaf.parent()) {
                    let parent_next = self.get_internal_node(parent_next);
                    Some(parent_next.children.first().unwrap().arena)
                } else {
                    None
                }
            }
            ArenaIndex::Internal(_) => {
                let node = self.get_internal_node(node_idx);
                let parent = self.get_internal_node(node.parent?);
                if let Some(next) = parent.children.get(node.parent_slot as usize + 1) {
                    Some(next.arena)
                } else if let Some(parent_next) = self.next_same_level_in_node(node.parent?) {
                    let parent_next = self.get_internal_node(parent_next);
                    parent_next.children.first().map(|x| x.arena)
                } else {
                    None
                }
            }
        }
    }

    fn prev_same_level_in_node(&self, node_idx: ArenaIndex) -> Option<ArenaIndex> {
        match node_idx {
            ArenaIndex::Leaf(leaf_idx) => {
                let leaf = self.leaf_nodes.get(leaf_idx).unwrap();
                let parent = self.get_internal_node(leaf.parent());
                let index = Self::get_leaf_slot(leaf_idx, parent);
                if index > 0 {
                    Some(parent.children[index - 1].arena)
                } else if let Some(parent_next) = self.prev_same_level_in_node(leaf.parent()) {
                    let parent_next = self.get_internal_node(parent_next);
                    Some(parent_next.children.last().unwrap().arena)
                } else {
                    None
                }
            }
            ArenaIndex::Internal(_) => {
                let node = self.get_internal_node(node_idx);
                let parent = self.get_internal_node(node.parent?);
                if node.parent_slot > 0 {
                    let Some(next) = parent.children.get(node.parent_slot as usize - 1) else {
                        unreachable!()
                    };
                    Some(next.arena)
                } else if let Some(parent_prev) = self.prev_same_level_in_node(node.parent?) {
                    let parent_prev = self.get_internal_node(parent_prev);
                    parent_prev.children.last().map(|x| x.arena)
                } else {
                    None
                }
            }
        }
    }

    /// find the next element in the tree
    pub fn next_elem(&self, path: QueryResult) -> Option<QueryResult> {
        self.next_same_level_in_node(path.leaf.into())
            .map(|x| QueryResult {
                leaf: x.unwrap_leaf().into(),
                offset: 0,
                found: true,
            })
    }

    fn next_same_level_node_with_filter(
        &self,
        node_idx: ArenaIndex,
        end_path: &[Idx],
        filter: &dyn Fn(&B::Cache) -> bool,
    ) -> Option<ArenaIndex> {
        let node = self.get_internal_node(node_idx);
        let mut parent = self.get_internal_node(node.parent?);
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
                parent = self.get_internal_node(parent_next);
                next_index = 0;
            } else {
                return None;
            }
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
        let parent = self.get_internal_node(parent_idx.arena);
        if this_idx.arr >= 1 {
            let prev = &parent.children[this_idx.arr - 1];
            path[depth - 1] = Idx::new(prev.arena, this_idx.arr - 1);
        } else {
            if !self.prev_sibling(&mut path[..depth - 1]) {
                return false;
            }

            let parent = self.get_internal_node(path[depth - 2].arena);
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
            let node = self.get_internal_node(node_idx);
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
        self.get_internal_mut(self.root)
    }

    pub fn is_empty(&self) -> bool {
        self.get_internal_node(self.root).is_empty()
    }

    fn get_path(&self, idx: ArenaIndex) -> NodePath {
        let mut path = NodePath::new();
        let mut node_idx = idx;
        while node_idx != self.root {
            match node_idx {
                ArenaIndex::Leaf(inner_node_idx) => {
                    let node = self.leaf_nodes.get(inner_node_idx).unwrap();
                    let parent = self.in_nodes.get(node.parent).unwrap();
                    let index = Self::get_leaf_slot(inner_node_idx, parent);
                    path.push(Idx::new(node_idx, index));
                    node_idx = ArenaIndex::Internal(node.parent);
                }
                ArenaIndex::Internal(_) => {
                    let node = self.get_internal_node(node_idx);
                    path.push(Idx::new(node_idx, node.parent_slot as usize));
                    node_idx = node.parent.unwrap();
                }
            }
        }
        path.push(Idx::new(self.root, 0));
        path.reverse();
        path
    }

    pub fn push(&mut self, elem: B::Elem) {
        let mut is_full = false;
        let mut parent_idx = self.root;
        let mut update_cache_idx = parent_idx;
        if self.is_empty() {
            let data = self.alloc_leaf_child(elem, parent_idx.unwrap());
            let parent = self.in_nodes.get_mut(parent_idx.unwrap()).unwrap();
            parent.children.push(data);
        } else {
            let leaf_idx = self.last_leaf();
            update_cache_idx = leaf_idx;
            let leaf = self.leaf_nodes.get_mut(leaf_idx.unwrap_leaf()).unwrap();
            parent_idx = leaf.parent();
            if leaf.elem.can_merge(&elem) {
                leaf.elem.merge_right(&elem);
                if let Some(listener) = self.element_move_listener.as_ref() {
                    listener(MoveEvent {
                        target_leaf: Some(leaf_idx.unwrap_leaf().into()),
                        elem: &elem,
                    });
                }
            } else {
                let data = self.alloc_leaf_child(elem, parent_idx.unwrap());
                let parent = self.in_nodes.get_mut(parent_idx.unwrap()).unwrap();
                parent.children.push(data);
                is_full = parent.is_full();
            }
        }

        self.recursive_update_cache(update_cache_idx, true, None);
        if is_full {
            self.split(parent_idx);
        }
    }

    pub fn prepend(&mut self, elem: B::Elem) {
        // FIXME: Call move listener
        let leaf_idx = self.first_leaf();
        let leaf = self.leaf_nodes.get_mut(leaf_idx.unwrap_leaf()).unwrap();
        let mut is_full = false;
        if elem.can_merge(&leaf.elem) {
            leaf.elem.merge_left(&elem);
            let leaf1 = Some(leaf_idx.unwrap_leaf().into());
            if let Some(listener) = self.element_move_listener.as_ref() {
                listener(MoveEvent {
                    target_leaf: leaf1,
                    elem: &elem,
                });
            }
        } else {
            let parent_idx = leaf.parent;
            let data = self.alloc_leaf_child(elem, parent_idx);
            let parent = self.in_nodes.get_mut(parent_idx).unwrap();
            parent.children.push(data);
            is_full = parent.is_full();
        }

        self.recursive_update_cache(leaf_idx, true, None);
        if is_full {
            self.split(leaf_idx);
        }
    }

    /// compare the position of a and b
    pub fn compare_pos(&self, a: QueryResult, b: QueryResult) -> Ordering {
        if a.leaf == b.leaf {
            return a.offset.cmp(&b.offset);
        }

        let leaf_a = self.leaf_nodes.get(a.leaf.0).unwrap();
        let leaf_b = self.leaf_nodes.get(b.leaf.0).unwrap();
        let mut node_a = self.get_internal_node(leaf_a.parent());
        if leaf_a.parent == leaf_b.parent {
            for child in node_a.children.iter() {
                if child.arena.unwrap() == a.leaf.0 {
                    return Ordering::Less;
                }
                if child.arena.unwrap() == b.leaf.0 {
                    return Ordering::Greater;
                }
            }
        }

        let mut node_b = self.get_internal_node(leaf_b.parent());
        while node_a.parent != node_b.parent {
            node_a = self.get_internal_node(node_a.parent.unwrap());
            node_b = self.get_internal_node(node_b.parent.unwrap());
        }

        node_a.parent_slot.cmp(&node_b.parent_slot)
    }

    /// Iterate the caches of previous nodes/elements.
    /// This method will visit as less caches as possible.
    /// For example, if all nodes in a subtree need to be visited, we will only visit the root cache.
    ///
    /// f: (node_cache, previous_sibling_elem, (this_elem, offset))
    pub fn visit_previous_caches<F>(&self, cursor: QueryResult, mut f: F)
        where
            F: FnMut(PreviousCache<'_, B>),
    {
        // the last index of path points to the leaf element
        let mut path = self.get_path(cursor.leaf.into());
        let mut path_index = 0;
        let mut child_index = 0;
        let mut node = self.get_internal_node(path[path_index].arena);
        'outer: loop {
            if path_index + 1 >= path.len() {
                break;
            }

            while child_index == path.get(path_index + 1).map(|x| x.arr).unwrap() {
                path_index += 1;
                if path_index + 1 < path.len() {
                    node = self.get_internal_node(path[path_index].arena);
                    child_index = 0;
                } else {
                    break 'outer;
                }
            }

            f(PreviousCache::NodeCache(&node.children[child_index].cache));
            child_index += 1;
        }

        let node = self.leaf_nodes.get(cursor.leaf.0).unwrap();
        f(PreviousCache::ThisElemAndOffset {
            elem: &node.elem,
            offset: cursor.offset,
        });
    }
}

pub enum PreviousCache<'a, B: BTreeTrait> {
    NodeCache(&'a B::Cache),
    PrevSiblingElem(&'a B::Elem),
    ThisElemAndOffset { elem: &'a B::Elem, offset: usize },
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
        for (index, node) in self.in_nodes.iter() {
            if index != self.root.unwrap() {
                assert!(!node.is_empty());
            }
            for (i, child_info) in node.children.iter().enumerate() {
                if matches!( child_info.arena, ArenaIndex::Internal(_)) {
                    let child = self.get_internal_node(child_info.arena);
                    let mut cache = Default::default();
                    child.calc_cache(&mut cache, None);
                    assert_eq!(child.parent_slot, i as u8);
                    assert_eq!(child.parent, Some(ArenaIndex::Internal(index)));
                    assert_eq!(cache, child_info.cache);
                }
            }
            if let Some(parent) = node.parent {
                let parent = self.get_internal_node(parent);
                assert_eq!(
                    parent.children[node.parent_slot as usize].arena,
                    ArenaIndex::Internal(index)
                );
                self.get_path(ArenaIndex::Internal(index));
            } else {
                assert_eq!(index, self.root.unwrap_internal())
            }

            if index != self.root.unwrap() {
                assert!(!node.is_lack(), "len={}\n", node.len());
            }

            assert!(!node.is_full(), "len={}", node.len());
        }

        for (leaf_index, leaf_node) in self.leaf_nodes.iter() {
            let mut length = 1;
            let mut node_idx = leaf_node.parent;
            while node_idx != self.root.unwrap() {
                let node = self.get_internal_node(ArenaIndex::Internal(node_idx));
                length += 1;
                node_idx = node.parent.unwrap().unwrap();
            }
            match leaf_level {
                Some(expected) => assert_eq!(length, expected),
                None => {
                    leaf_level = Some(length);
                }
            }

            let cache = B::get_elem_cache(&leaf_node.elem);
            let parent = self.get_internal_node(leaf_node.parent());
            assert_eq!(parent.children.iter().find(|x| x.arena.unwrap_leaf() == leaf_index).unwrap().cache, cache);
            self.get_path(ArenaIndex::Leaf(leaf_index));
        }
    }
}

impl<B: BTreeTrait> Default for BTree<B> {
    fn default() -> Self {
        Self::new()
    }
}
