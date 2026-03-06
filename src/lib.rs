#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

use core::{fmt::Debug, ops::Range};
use std::collections::{BTreeSet, VecDeque};
use std::ops::AddAssign;
use std::{cmp::Ordering, mem::take, ops::RangeBounds};

pub(crate) use heapless::Vec as HeaplessVec;
use itertools::Itertools;
use rle::{CanRemove, TryInsert};
use rustc_hash::{FxHashMap, FxHashSet};
use thunderdome::Arena;
use thunderdome::Index as RawArenaIndex;

pub use generic_impl::*;

use crate::rle::{HasLength, Mergeable, Sliceable};

mod generic_impl;
pub mod iter;

pub mod rle;

pub type HeapVec<T> = Vec<T>;

const MAX_CHILDREN_NUM: usize = 12;

/// `Elem` should has length. `offset` in search result should always >= `Elem.rle_len()`
pub trait BTreeTrait {
    /// Sometime an [Elem] with length of 0, but it's not empty.
    ///
    /// The empty [Elem]s are the ones that can be safely ignored.
    type Elem: Debug + HasLength + Sliceable + Mergeable + TryInsert + CanRemove;
    type Cache: Debug + Default + Clone + Eq;
    type CacheDiff: Debug + Default + CanRemove;
    // Whether we should use cache diff by default
    const USE_DIFF: bool = true;

    /// If diff.is_some, return value should be some too
    fn calc_cache_internal(cache: &mut Self::Cache, caches: &[Child<Self>]) -> Self::CacheDiff;
    fn apply_cache_diff(cache: &mut Self::Cache, diff: &Self::CacheDiff);
    fn merge_cache_diff(diff1: &mut Self::CacheDiff, diff2: &Self::CacheDiff);
    fn get_elem_cache(elem: &Self::Elem) -> Self::Cache;
    fn new_cache_to_diff(cache: &Self::Cache) -> Self::CacheDiff;
    fn sub_cache(cache_lhs: &Self::Cache, cache_rhs: &Self::Cache) -> Self::CacheDiff;
}

pub trait Query<B: BTreeTrait> {
    type QueryArg: Clone;

    fn init(target: &Self::QueryArg) -> Self;

    fn find_node(&mut self, target: &Self::QueryArg, child_caches: &[Child<B>]) -> FindResult;

    /// Confirm the search result and returns (offset, found)
    ///
    /// If elem is not target, `found=false`
    fn confirm_elem(&mut self, q: &Self::QueryArg, elem: &B::Elem) -> (usize, bool);
}

pub struct BTree<B: BTreeTrait> {
    /// internal nodes
    in_nodes: Arena<Node<B>>,
    /// leaf nodes
    leaf_nodes: Arena<LeafNode<B::Elem>>,
    first_leaf: Option<LeafIndex>,
    last_leaf: Option<LeafIndex>,
    /// root is always a internal node
    /// TODO: we may use a constant as root index
    root: ArenaIndex,
    root_cache: B::Cache,
}

impl<Elem: Clone, B: BTreeTrait<Elem = Elem>> Clone for BTree<B> {
    fn clone(&self) -> Self {
        Self {
            in_nodes: self.in_nodes.clone(),
            leaf_nodes: self.leaf_nodes.clone(),
            first_leaf: self.first_leaf,
            last_leaf: self.last_leaf,
            root: self.root,
            root_cache: self.root_cache.clone(),
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
    pub arr: u8,
}

impl Idx {
    pub fn new(arena: ArenaIndex, arr: u8) -> Self {
        Self { arena, arr }
    }
}

type NodePath = HeaplessVec<Idx, 10>;

#[derive(Debug, Clone, PartialEq, Eq, Copy, Hash)]
pub struct Cursor {
    pub leaf: LeafIndex,
    pub offset: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub struct QueryResult {
    pub cursor: Cursor,
    pub found: bool,
}

/// Exposed arena index
///
/// Only exposed arena index of leaf node.
///
///
#[repr(transparent)]
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub struct LeafIndex(RawArenaIndex);

impl LeafIndex {
    pub fn inner(&self) -> RawArenaIndex {
        self.0
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum ArenaIndex {
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

    pub fn unwrap_leaf(self) -> RawArenaIndex {
        match self {
            ArenaIndex::Leaf(x) => x,
            ArenaIndex::Internal(_) => panic!("unwrap_leaf on internal node"),
        }
    }

    pub fn unwrap_internal(self) -> RawArenaIndex {
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
    cursor: Cursor,
    pub elem: &'a Elem,
    pub start: Option<usize>,
    pub end: Option<usize>,
}

impl<'a, Elem> ElemSlice<'a, Elem> {
    pub fn cursor(&self) -> &Cursor {
        &self.cursor
    }
}

impl QueryResult {
    pub fn elem<'b, Elem: Debug, B: BTreeTrait<Elem = Elem>>(
        &self,
        tree: &'b BTree<B>,
    ) -> Option<&'b Elem> {
        tree.leaf_nodes.get(self.cursor().leaf.0).map(|x| &x.elem)
    }

    #[inline(always)]
    pub fn cursor(&self) -> Cursor {
        self.cursor
    }

    #[inline(always)]
    pub fn leaf(&self) -> LeafIndex {
        self.cursor().leaf
    }

    #[inline(always)]
    pub fn offset(&self) -> usize {
        self.cursor().offset
    }

    #[inline(always)]
    pub fn found(&self) -> bool {
        self.found
    }

    #[inline(always)]
    pub fn arena(&self) -> RawArenaIndex {
        self.cursor.leaf.0
    }
}

#[derive(Debug, Clone)]
pub struct LeafNode<Elem> {
    elem: Elem,
    parent: RawArenaIndex,
    parent_slot: u8,
    prev_leaf: Option<RawArenaIndex>,
    next_leaf: Option<RawArenaIndex>,
}

impl<T> LeafNode<T> {
    pub fn parent(&self) -> ArenaIndex {
        ArenaIndex::Internal(self.parent)
    }

    pub fn elem(&self) -> &T {
        &self.elem
    }

    #[inline(always)]
    fn parent_slot(&self) -> usize {
        self.parent_slot as usize
    }

    #[inline(always)]
    fn prev_leaf(&self) -> Option<LeafIndex> {
        self.prev_leaf.map(LeafIndex)
    }

    #[inline(always)]
    fn next_leaf(&self) -> Option<LeafIndex> {
        self.next_leaf.map(LeafIndex)
    }
}

impl<T: Sliceable> LeafNode<T> {
    fn split(&mut self, offset: usize) -> Self {
        let new_elem = self.elem.split(offset);
        debug_assert!(self.parent_slot < u8::MAX);
        Self {
            elem: new_elem,
            parent: self.parent,
            parent_slot: self.parent_slot + 1,
            prev_leaf: None,
            next_leaf: None,
        }
    }
}

pub struct Node<B: BTreeTrait> {
    parent: Option<ArenaIndex>,
    parent_slot: u8,
    children: HeaplessVec<Child<B>, MAX_CHILDREN_NUM>,
}

#[repr(transparent)]
#[derive(Debug, Default, Clone)]
pub struct SplittedLeaves {
    pub arr: HeaplessVec<LeafIndex, 2>,
}

impl SplittedLeaves {
    #[inline]
    fn push_option(&mut self, leaf: Option<ArenaIndex>) {
        if let Some(leaf) = leaf {
            self.arr.push(leaf.unwrap().into()).unwrap();
        }
    }

    #[inline]
    fn push(&mut self, leaf: ArenaIndex) {
        self.arr.push(leaf.unwrap().into()).unwrap();
    }
}

impl<Cache: Debug, Elem: Debug, B: BTreeTrait<Elem = Elem, Cache = Cache>> Debug for BTree<B> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        fn fmt_node<Cache: Debug, Elem: Debug, B: BTreeTrait<Elem = Elem>>(
            tree: &BTree<B>,
            node_idx: ArenaIndex,
            f: &mut core::fmt::Formatter<'_>,
            indent_size: usize,
        ) -> core::fmt::Result {
            match node_idx {
                ArenaIndex::Leaf(_) => {}
                ArenaIndex::Internal(_) => {
                    let node = tree.get_internal(node_idx);
                    for child in node.children.iter() {
                        indent(f, indent_size)?;
                        if child.is_internal() {
                            let child_node = tree.get_internal(child.arena);
                            f.write_fmt(format_args!(
                                "{} Arena({:?}) Cache: {:?}\n",
                                child_node.parent_slot, &child.arena, &child.cache
                            ))?;
                            fmt_node::<Cache, Elem, B>(tree, child.arena, f, indent_size + 1)?;
                        } else {
                            let node = tree.get_leaf(child.arena);
                            f.write_fmt(format_args!(
                                "Leaf({:?}) Arena({:?}) Parent({:?}) Cache: {:?}\n",
                                &node.elem, child.arena, node.parent, &child.cache
                            ))?;
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

impl<Cache: Debug, Elem: Debug, B: BTreeTrait<Elem = Elem, Cache = Cache>> Debug for Node<B> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Node")
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
            parent_slot: self.parent_slot,
            children: self.children.clone(),
        }
    }
}

pub struct Child<B: ?Sized + BTreeTrait> {
    pub arena: ArenaIndex,
    pub cache: B::Cache,
}

impl<B: ?Sized + BTreeTrait> Child<B> {
    #[inline]
    fn is_internal(&self) -> bool {
        matches!(self.arena, ArenaIndex::Internal(_))
    }

    #[inline]
    #[allow(unused)]
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
    pub fn cache(&self) -> &B::Cache {
        &self.cache
    }

    fn new(arena: ArenaIndex, cache: B::Cache) -> Self {
        Self { arena, cache }
    }
}

impl<B: BTreeTrait> Node<B> {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            parent: None,
            parent_slot: u8::MAX,
            children: HeaplessVec::new(),
        }
    }

    #[inline(always)]
    pub fn is_full(&self) -> bool {
        self.children.len() >= MAX_CHILDREN_NUM
    }

    #[inline(always)]
    pub fn is_lack(&self) -> bool {
        self.children.len() < MAX_CHILDREN_NUM / 2
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.children.len()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline(always)]
    pub fn is_child_leaf(&self) -> bool {
        if self.children.is_empty() {
            return true;
        }

        self.children[0].is_leaf()
    }

    /// if diff is not provided, the cache will be calculated from scratch
    #[inline(always)]
    fn calc_cache(&self, cache: &mut B::Cache, diff: Option<B::CacheDiff>) -> B::CacheDiff {
        match diff {
            Some(inner) => {
                B::apply_cache_diff(cache, &inner);
                inner
            }
            None => B::calc_cache_internal(cache, &self.children),
        }
    }
}

impl<B: BTreeTrait> Default for Node<B> {
    fn default() -> Self {
        Self::new()
    }
}

type LeafDirtyMap<Diff> = FxHashMap<ArenaIndex, Diff>;

/// Whether the parent node is lack of children
#[repr(transparent)]
struct LackInfo {
    /// if Some, the parent node is lack
    parent_lack: Option<ArenaIndex>,
}

impl<B: BTreeTrait> BTree<B> {
    pub fn new() -> Self {
        let mut arena = Arena::new();
        let root = arena.insert(Node::new());
        Self {
            in_nodes: arena,
            leaf_nodes: Arena::new(),
            first_leaf: None,
            last_leaf: None,
            root: ArenaIndex::Internal(root),
            root_cache: B::Cache::default(),
        }
    }

    /// Get the number of nodes in the tree.
    /// It includes all the internal nodes and the leaf nodes.
    #[inline(always)]
    pub fn node_len(&self) -> usize {
        self.in_nodes.len() + self.leaf_nodes.len()
    }

    /// Insert new element to the tree.
    ///
    /// Returns (insert_pos, splitted_leaves)
    pub fn insert<Q>(&mut self, q: &Q::QueryArg, data: B::Elem) -> (Cursor, SplittedLeaves)
    where
        Q: Query<B>,
    {
        let Some(result) = self.query::<Q>(q) else {
            return (self.push(data), Default::default());
        };

        self.insert_by_path(result.cursor, data)
    }

    pub fn insert_by_path(&mut self, cursor: Cursor, data: B::Elem) -> (Cursor, SplittedLeaves) {
        let index = cursor.leaf;
        let leaf = self.leaf_nodes.get_mut(index.0).unwrap();
        let mut parent_idx = leaf.parent();
        let mut update_cache_idx = ArenaIndex::Leaf(index.0);
        let mut can_use_diff = B::USE_DIFF;
        let cache_diff = if B::USE_DIFF {
            Some(B::new_cache_to_diff(&B::get_elem_cache(&data)))
        } else {
            None
        };

        let mut is_full = false;
        let mut splitted: SplittedLeaves = Default::default();
        let ans = match leaf.elem.try_insert(cursor.offset, data) {
            Ok(_) => cursor,
            Err(data) => {
                // Try to merge
                if cursor.offset == 0 && data.can_merge(&leaf.elem) {
                    leaf.elem.merge_left(&data);
                    Cursor {
                        leaf: index,
                        offset: 0,
                    }
                } else if cursor.offset == leaf.elem.rle_len() && leaf.elem.can_merge(&data) {
                    let offset = leaf.elem.rle_len();
                    leaf.elem.merge_right(&data);
                    Cursor {
                        leaf: index,
                        offset,
                    }
                } else {
                    // Insert new leaf node
                    let SplitInfo {
                        parent_idx: parent_index,
                        insert_slot: insert_index,
                        new_leaf,
                        new_pos: _,
                        ..
                    } = self.split_leaf_if_needed(cursor);
                    parent_idx = ArenaIndex::Internal(parent_index);
                    update_cache_idx = parent_idx;
                    can_use_diff = false;
                    let (prev_leaf, next_leaf) =
                        self.leaf_neighbors_for_insert(parent_index, insert_index);
                    let child = self.alloc_leaf_child(
                        data,
                        parent_index,
                        insert_index as u8,
                        prev_leaf,
                        next_leaf,
                    );
                    let ans = child.arena;
                    splitted.push_option(new_leaf);
                    let is_parent_full = {
                        let parent = self.in_nodes.get_mut(parent_index).unwrap();
                        parent.children.insert(insert_index, child).unwrap();
                        parent.is_full()
                    };
                    self.update_children_parent_slot_from(parent_idx, insert_index + 1);
                    is_full = is_parent_full;
                    Cursor {
                        leaf: ans.unwrap().into(),
                        offset: 0,
                    }
                }
            }
        };

        self.recursive_update_cache(
            update_cache_idx,
            can_use_diff,
            if can_use_diff { cache_diff } else { None },
        );
        if is_full {
            self.split(parent_idx);
        }

        (ans, splitted)
    }

    fn alloc_leaf_child(
        &mut self,
        data: <B as BTreeTrait>::Elem,
        parent_index: RawArenaIndex,
        parent_slot: u8,
        prev_leaf: Option<LeafIndex>,
        next_leaf: Option<LeafIndex>,
    ) -> Child<B> {
        let elem_cache = B::get_elem_cache(&data);
        let new_leaf_index = self.alloc_new_leaf(LeafNode {
            elem: data,
            parent: parent_index,
            parent_slot,
            prev_leaf: None,
            next_leaf: None,
        });
        self.link_leaf_between(new_leaf_index.unwrap().into(), prev_leaf, next_leaf);
        Child {
            arena: new_leaf_index,
            cache: elem_cache,
        }
    }

    /// Split a leaf node at offset if it's not the start/end of the leaf node.
    ///
    /// This method should be called when inserting at target pos.
    fn split_leaf_if_needed(&mut self, pos: Cursor) -> SplitInfo {
        let mut new_pos = Some(pos);
        let mut rt_new_leaf = None;
        let mut reindex_from = None;
        let mut new_leaf_link = None;
        let (parent_idx, left_neighbour, insert_pos) = {
            let leaf = self.leaf_nodes.get_mut(pos.leaf.0).unwrap();
            let parent_idx = leaf.parent;
            let parent = self.in_nodes.get_mut(leaf.parent).unwrap();
            let leaf_slot = leaf.parent_slot();
            let left_neighbour = if leaf_slot == 0 {
                None
            } else {
                Some(parent.children[leaf_slot - 1].arena.unwrap().into())
            };
            let insert_pos = if pos.offset == 0 {
                leaf_slot
            } else if pos.offset == leaf.elem.rle_len() {
                if leaf_slot + 1 < parent.children.len() {
                    new_pos = Some(Cursor {
                        leaf: parent.children[leaf_slot + 1].arena.unwrap().into(),
                        offset: 0,
                    });
                } else {
                    new_pos = self.next_elem(pos);
                }
                leaf_slot + 1
            } else {
                assert!(
                    pos.offset < leaf.elem.rle_len(),
                    "elem.rle_len={} but pos.offset={} Elem:{:?}",
                    leaf.elem.rle_len(),
                    pos.offset,
                    &leaf.elem
                );

                if parent.children.len() + 1 >= MAX_CHILDREN_NUM {
                    self.split(ArenaIndex::Internal(parent_idx));
                    return self.split_leaf_if_needed(pos);
                }

                let old_next_leaf = leaf.next_leaf();
                let new_leaf = leaf.split(pos.offset);
                let left_cache = B::get_elem_cache(&leaf.elem);
                let cache = B::get_elem_cache(&new_leaf.elem);
                let leaf_arena_index = {
                    let arena_index = self.leaf_nodes.insert(new_leaf);
                    ArenaIndex::Leaf(arena_index)
                };
                new_leaf_link = Some((
                    leaf_arena_index.unwrap().into(),
                    Some(pos.leaf),
                    old_next_leaf,
                ));
                rt_new_leaf = Some(leaf_arena_index);
                new_pos = Some(Cursor {
                    leaf: leaf_arena_index.unwrap().into(),
                    offset: 0,
                });
                parent.children[leaf_slot].cache = left_cache;
                parent
                    .children
                    .insert(
                        leaf_slot + 1,
                        Child {
                            arena: leaf_arena_index,
                            cache,
                        },
                    )
                    .unwrap();
                reindex_from = Some(leaf_slot + 2);

                leaf_slot + 1
            };
            (parent_idx, left_neighbour, insert_pos)
        };
        if let Some(reindex_from) = reindex_from {
            self.update_children_parent_slot_from(ArenaIndex::Internal(parent_idx), reindex_from);
        }
        if let Some((new_leaf, prev_leaf, next_leaf)) = new_leaf_link {
            self.link_leaf_between(new_leaf, prev_leaf, next_leaf);
        }

        SplitInfo {
            left_neighbour,
            new_pos,
            parent_idx,
            insert_slot: insert_pos,
            new_leaf: rt_new_leaf,
        }
    }

    fn alloc_new_leaf(&mut self, leaf: LeafNode<B::Elem>) -> ArenaIndex {
        let arena_index = self.leaf_nodes.insert(leaf);
        ArenaIndex::Leaf(arena_index)
    }

    #[inline(always)]
    fn next_leaf_index(&self, leaf: LeafIndex) -> Option<LeafIndex> {
        self.leaf_nodes
            .get(leaf.0)
            .and_then(|leaf| leaf.next_leaf())
    }

    #[inline(always)]
    fn prev_leaf_index(&self, leaf: LeafIndex) -> Option<LeafIndex> {
        self.leaf_nodes
            .get(leaf.0)
            .and_then(|leaf| leaf.prev_leaf())
    }

    #[inline(always)]
    fn leaf_neighbors_for_insert(
        &self,
        parent_index: RawArenaIndex,
        insert_slot: usize,
    ) -> (Option<LeafIndex>, Option<LeafIndex>) {
        let parent = self.in_nodes.get(parent_index).unwrap();
        if parent.children.is_empty() {
            return (None, None);
        }

        let prev_leaf = if insert_slot == 0 {
            let next_leaf = parent.children[0].arena.unwrap_leaf();
            self.leaf_nodes.get(next_leaf).unwrap().prev_leaf()
        } else {
            Some(parent.children[insert_slot - 1].arena.unwrap_leaf().into())
        };
        let next_leaf = if insert_slot == parent.children.len() {
            let prev_leaf = parent.children[insert_slot - 1].arena.unwrap_leaf();
            self.leaf_nodes.get(prev_leaf).unwrap().next_leaf()
        } else {
            Some(parent.children[insert_slot].arena.unwrap_leaf().into())
        };
        (prev_leaf, next_leaf)
    }

    #[inline(always)]
    fn link_leaf_between(
        &mut self,
        leaf: LeafIndex,
        prev_leaf: Option<LeafIndex>,
        next_leaf: Option<LeafIndex>,
    ) {
        debug_assert!(
            prev_leaf.is_some() || next_leaf.is_none() || self.first_leaf == next_leaf,
            "link_leaf_between received an invalid head insertion"
        );
        debug_assert!(
            next_leaf.is_some() || prev_leaf.is_none() || self.last_leaf == prev_leaf,
            "link_leaf_between received an invalid tail insertion"
        );
        {
            let leaf_node = self.leaf_nodes.get_mut(leaf.0).unwrap();
            leaf_node.prev_leaf = prev_leaf.map(|x| x.0);
            leaf_node.next_leaf = next_leaf.map(|x| x.0);
        }

        if let Some(prev_leaf) = prev_leaf {
            self.leaf_nodes.get_mut(prev_leaf.0).unwrap().next_leaf = Some(leaf.0);
        } else {
            self.first_leaf = Some(leaf);
        }

        if let Some(next_leaf) = next_leaf {
            self.leaf_nodes.get_mut(next_leaf.0).unwrap().prev_leaf = Some(leaf.0);
        } else {
            self.last_leaf = Some(leaf);
        }

        if self.first_leaf.is_none() {
            self.first_leaf = Some(leaf);
        }

        if self.last_leaf.is_none() {
            self.last_leaf = Some(leaf);
        }
    }

    #[inline(always)]
    fn unlink_leaf(&mut self, leaf: LeafIndex) -> (Option<LeafIndex>, Option<LeafIndex>) {
        let (prev_leaf, next_leaf) = {
            let leaf_node = self.leaf_nodes.get(leaf.0).unwrap();
            (leaf_node.prev_leaf(), leaf_node.next_leaf())
        };

        if let Some(prev_leaf) = prev_leaf {
            self.leaf_nodes.get_mut(prev_leaf.0).unwrap().next_leaf = next_leaf.map(|x| x.0);
        } else {
            self.first_leaf = next_leaf;
        }

        if let Some(next_leaf) = next_leaf {
            self.leaf_nodes.get_mut(next_leaf.0).unwrap().prev_leaf = prev_leaf.map(|x| x.0);
        } else {
            self.last_leaf = prev_leaf;
        }

        if let Some(leaf_node) = self.leaf_nodes.get_mut(leaf.0) {
            leaf_node.prev_leaf = None;
            leaf_node.next_leaf = None;
        }

        (prev_leaf, next_leaf)
    }

    #[inline(always)]
    fn set_child_parent_and_slot(
        &mut self,
        child: ArenaIndex,
        parent_index: RawArenaIndex,
        slot: usize,
    ) {
        let slot = slot as u8;
        match child {
            ArenaIndex::Leaf(leaf) => {
                let child = self.leaf_nodes.get_mut(leaf).unwrap();
                child.parent = parent_index;
                child.parent_slot = slot;
            }
            ArenaIndex::Internal(index) => {
                let child = self.in_nodes.get_mut(index).unwrap();
                child.parent = Some(ArenaIndex::Internal(parent_index));
                child.parent_slot = slot;
            }
        }
    }

    /// Insert many elements into the tree at once
    ///
    /// It will invoke [`BTreeTrait::insert_batch`]
    pub fn insert_many_by_cursor(
        &mut self,
        cursor: Option<Cursor>,
        mut data_iter: impl Iterator<Item = B::Elem>,
    ) {
        let Some(first) = data_iter.next() else {
            return;
        };

        let Some(second) = data_iter.next() else {
            if let Some(c) = cursor {
                self.insert_by_path(c, first);
                return;
            } else {
                self.push(first);
                return;
            }
        };

        let mut data = Vec::with_capacity(data_iter.size_hint().0 + 2);
        data.push(first);
        data.push(second);
        for elem in data_iter {
            data.push(elem);
        }

        merge_adj(&mut data);
        if data.len() == 1 {
            if let Some(c) = cursor {
                self.insert_by_path(c, data.pop().unwrap());
                return;
            } else {
                self.push(data.pop().unwrap());
                return;
            }
        }

        if cursor.is_none() && self.is_empty() {
            assert!(self.is_empty());
            let (new_root, _) = self.create_subtrees_from_elem(data);
            self.in_nodes.remove(self.root.unwrap()).unwrap();
            self.root = new_root;
            self.update_root_cache();
            return;
        }

        // dbg!(cursor, &data);
        // dbg!(&self);
        let cursor = cursor.expect("Cursor must be provided when tree is not empty");
        let SplitInfo {
            new_pos,
            left_neighbour,
            ..
        } = self.split_leaf_if_needed(cursor);
        let mut inserted = 0;
        if let Some(left) = left_neighbour {
            let left_node = self.leaf_nodes.get_mut(left.0).unwrap();
            let mut i = 0;
            while i < data.len() && left_node.elem.can_merge(&data[i]) {
                left_node.elem.merge_right(&data[i]);
                i += 1;
            }

            self.recursive_update_cache(left.into(), B::USE_DIFF, None);
            inserted = i;
        }

        let mut pos = new_pos.unwrap_or(cursor);
        // TODO: PERF this can be optimized further
        for item in data.drain(inserted..).rev() {
            let (p, _) = self.insert_by_path(pos, item);
            pos = p
        }
    }

    /// The returned height starts from 0. Leaf level is 0.
    ///
    /// Returns (newly created subtree's root, height)
    fn create_subtrees_from_elem(&mut self, data: Vec<B::Elem>) -> (ArenaIndex, usize) {
        debug_assert!(self.first_leaf.is_none());
        debug_assert!(self.last_leaf.is_none());
        let mut height = 0;
        let mut nodes = Vec::with_capacity(data.len() / MAX_CHILDREN_NUM + 1);
        let mut prev_leaf = None;
        for elem in data.into_iter().chunks(MAX_CHILDREN_NUM).into_iter() {
            let parent_index = self.in_nodes.insert(Node {
                parent: None,
                parent_slot: 0,
                children: Default::default(),
            });

            nodes.push(parent_index);
            let mut created_leaves = Vec::new();
            for (i, elem) in elem.enumerate() {
                let leaf = {
                    // alloc new leaf child
                    let elem_cache = B::get_elem_cache(&elem);
                    let new_leaf_index = {
                        let leaf = LeafNode {
                            elem,
                            parent: parent_index,
                            parent_slot: i as u8,
                            prev_leaf: None,
                            next_leaf: None,
                        };
                        self.leaf_nodes.insert(leaf)
                    };
                    created_leaves.push(LeafIndex::from(new_leaf_index));
                    Child {
                        arena: ArenaIndex::Leaf(new_leaf_index),
                        cache: elem_cache,
                    }
                };
                let parent = self.in_nodes.get_mut(parent_index).unwrap();
                parent.children.push(leaf).unwrap();
            }
            for leaf_index in created_leaves {
                self.link_leaf_between(leaf_index, prev_leaf, None);
                prev_leaf = Some(leaf_index);
            }
        }

        while nodes.len() > 1 {
            let mut new_nodes = Vec::with_capacity(nodes.len() / MAX_CHILDREN_NUM + 1);
            for chunk in nodes.into_iter().chunks(MAX_CHILDREN_NUM).into_iter() {
                let parent_index = self.in_nodes.insert(Node {
                    parent: None,
                    parent_slot: 0,
                    children: Default::default(),
                });

                new_nodes.push(parent_index);
                for (i, child_idx) in chunk.enumerate() {
                    let (parent, child) = self.in_nodes.get2_mut(parent_index, child_idx);
                    let parent = parent.unwrap();
                    let child = child.unwrap();
                    let mut cache = B::Cache::default();
                    B::calc_cache_internal(&mut cache, &child.children);
                    parent
                        .children
                        .push(Child {
                            arena: ArenaIndex::Internal(child_idx),
                            cache,
                        })
                        .unwrap();
                    child.parent = Some(ArenaIndex::Internal(parent_index));
                    child.parent_slot = i as u8;
                }
            }
            nodes = new_nodes;
            height += 1;
        }

        (ArenaIndex::Internal(nodes[0]), height)
    }

    /// Shift by offset 1.
    ///
    /// It will not stay on empty spans but scan forward
    pub fn shift_path_by_one_offset(&self, mut path: Cursor) -> Option<Cursor>
    where
        B::Elem: rle::HasLength,
    {
        let leaf = self.leaf_nodes.get(path.leaf.0).unwrap();
        if path.offset + 1 < leaf.elem.rle_len() {
            path.offset += 1;
            return Some(path);
        }

        let mut parent_idx = leaf.parent;
        let mut parent = self.in_nodes.get(leaf.parent).unwrap();
        let mut elem_slot_index = leaf.parent_slot();
        path.offset += 1;
        loop {
            if elem_slot_index == parent.children.len() {
                if let Some(next) = self.next_same_level_in_node(ArenaIndex::Internal(parent_idx)) {
                    elem_slot_index = 0;
                    path.offset = 0;
                    parent_idx = next.unwrap_internal();
                    parent = self.in_nodes.get(parent_idx).unwrap();
                } else {
                    return None;
                }
            }

            let elem = &parent.children[elem_slot_index];
            let leaf = self.leaf_nodes.get(elem.arena.unwrap()).unwrap();
            // skip empty span
            if leaf.elem.rle_len() <= path.offset {
                path.offset -= leaf.elem.rle_len();
                elem_slot_index += 1;
            } else {
                path.leaf = elem.arena.unwrap_leaf().into();
                break;
            }
        }

        Some(path)
    }

    /// Query the tree by custom query type
    ///
    /// Return None if the tree is empty
    pub fn query<Q>(&self, query: &Q::QueryArg) -> Option<QueryResult>
    where
        Q: Query<B>,
    {
        self.query_with_finder_return::<Q>(query).0
    }

    pub fn query_with_finder_return<Q>(&self, query: &Q::QueryArg) -> (Option<QueryResult>, Q)
    where
        Q: Query<B>,
    {
        let mut finder = Q::init(query);
        if self.is_empty() {
            return (None, finder);
        }

        let mut node = self.in_nodes.get(self.root.unwrap()).unwrap();
        let mut index;
        let mut found = true;
        loop {
            let result = finder.find_node(query, &node.children);
            debug_assert!(!node.children.is_empty());
            let i = result.index;
            found = found && result.found;
            index = node.children[i].arena;
            match index {
                ArenaIndex::Internal(index) => {
                    node = self.in_nodes.get(index).unwrap();
                }
                ArenaIndex::Leaf(_) => {
                    let (offset, leaf_found) = finder.confirm_elem(
                        query,
                        &self.leaf_nodes.get(index.unwrap_leaf()).unwrap().elem,
                    );
                    return (
                        Some(QueryResult {
                            cursor: Cursor {
                                leaf: index.unwrap_leaf().into(),
                                offset,
                            },
                            found: found && leaf_found,
                        }),
                        finder,
                    );
                }
            }
        }
    }

    pub fn get_elem_mut(&mut self, leaf: LeafIndex) -> Option<&mut B::Elem> {
        let node = self.leaf_nodes.get_mut(leaf.0)?;
        Some(&mut node.elem)
    }

    pub fn get_elem(&self, leaf: LeafIndex) -> Option<&<B as BTreeTrait>::Elem> {
        self.leaf_nodes.get(leaf.0).map(|x| &x.elem)
    }

    /// Remove leaf node from the tree
    ///
    /// If it's already removed, this method will return None
    pub fn remove_leaf(&mut self, path: Cursor) -> Option<B::Elem> {
        let (parent_idx, index) = {
            let leaf = self.leaf_nodes.get(path.leaf.0)?;
            (leaf.parent(), leaf.parent_slot())
        };
        self.unlink_leaf(path.leaf);
        let (is_lack, is_empty) = {
            let parent = self.in_nodes.get_mut(parent_idx.unwrap()).unwrap();
            let child = parent.children.remove(index);
            debug_assert_eq!(child.arena.unwrap(), path.leaf.0);
            (parent.is_lack(), parent.is_empty())
        };
        let elem = self.leaf_nodes.remove(path.leaf.0).unwrap().elem;
        self.update_children_parent_slot_from(parent_idx, index);

        self.recursive_update_cache(parent_idx, B::USE_DIFF, None);
        if is_empty {
            self.remove_internal_node(parent_idx.unwrap());
        } else if is_lack {
            self.handle_lack_recursively(parent_idx);
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
                self.handle_lack_recursively(parent_idx);
            }
        } else {
            // ignore remove root
            unreachable!()
        }
    }

    /// Update the elements in place.
    ///
    /// If the range.start or range.end is in the middle of a leaf node, the leaf node
    /// will be splitted into two leaf nodes. The new leaf nodes will be returned.
    ///
    /// F should returns `Some(cache_diff)` if cache needs to be updated. Otherwise, returns None.
    ///
    /// If the given range has zero length, f will still be called, and the slice will
    /// have same `start` and `end` field
    ///
    /// TODO: need better test coverage
    pub fn update<F>(&mut self, range: Range<Cursor>, f: &mut F) -> SplittedLeaves
    where
        F: FnMut(&mut B::Elem) -> Option<B::CacheDiff>,
    {
        let mut splitted = SplittedLeaves::default();
        let start = range.start;
        let SplitInfo {
            new_pos: end,
            new_leaf,
            ..
        } = self.split_leaf_if_needed(range.end);
        splitted.push_option(new_leaf);
        let SplitInfo {
            new_pos: start,
            new_leaf,
            ..
        } = self.split_leaf_if_needed(start);
        splitted.push_option(new_leaf);
        let Some(start) = start else {
            return splitted;
        };
        let mut dirty_map: LeafDirtyMap<B::CacheDiff> = FxHashMap::default();
        let mut to_remove = Vec::default();
        let mut current_leaf = Some(start.leaf);
        let end_leaf = end.map(|cursor| cursor.leaf);

        while let Some(leaf_idx) = current_leaf {
            if end_leaf == Some(leaf_idx) {
                break;
            }

            let next_leaf = self.next_leaf_index(leaf_idx);
            let node = self.leaf_nodes.get_mut(leaf_idx.0).unwrap();
            let cache_diff = f(&mut node.elem);
            if node.elem.can_remove() {
                to_remove.push(leaf_idx);
            }

            if let Some(diff) = cache_diff {
                add_leaf_dirty_map(leaf_idx.into(), &mut dirty_map, diff);
            }

            current_leaf = next_leaf;
        }

        if !dirty_map.is_empty() {
            self.update_dirty_cache_map(dirty_map);
        } else {
            self.in_nodes
                .get(self.root.unwrap_internal())
                .unwrap()
                .calc_cache(&mut self.root_cache, None);
        }

        for leaf in to_remove {
            self.remove_leaf(Cursor { leaf, offset: 0 });
        }
        splitted
    }

    /// Prefer begin of the next leaf node than end of the current leaf node
    ///
    /// When path.offset == leaf.rle_len(), this method will return
    /// the next leaf node with offset 0
    #[allow(unused)]
    pub fn prefer_right(&self, path: Cursor) -> Option<Cursor> {
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

    /// Prefer end of the previous leaf node than begin of the current leaf node
    ///
    /// When path.offset == 0, this method will return
    /// the previous leaf node with offset leaf.rle_len()
    #[allow(unused)]
    pub fn prefer_left(&self, path: Cursor) -> Option<Cursor> {
        if path.offset != 0 {
            return Some(path);
        }

        let elem = self.prev_elem(path);
        if let Some(elem) = elem {
            let leaf = self.leaf_nodes.get(elem.leaf.0).unwrap();
            Some(Cursor {
                leaf: elem.leaf,
                offset: leaf.elem.rle_len(),
            })
        } else {
            None
        }
    }

    /// Update leaf node's elements.
    ///
    /// `f` returns Option<(cache_diff, new_insert_1, new_insert2)>
    ///
    /// - If returned value is `None`, the cache will not be updated.
    /// - If leaf_node.can_remove(), it will be removed from the tree.
    ///
    /// Returns (path, splitted_leaves), if is is still valid after this method. (If the leaf node is removed, the path will be None)
    pub fn update_leaf_by_search<Q: Query<B>>(
        &mut self,
        q: &Q::QueryArg,
        f: impl FnOnce(
            &mut B::Elem,
            QueryResult,
        ) -> Option<(B::CacheDiff, Option<B::Elem>, Option<B::Elem>)>,
    ) -> (Option<Cursor>, SplittedLeaves) {
        if self.is_empty() {
            panic!("update_leaf_by_search called on empty tree");
        }

        let mut splitted = SplittedLeaves::default();
        let mut finder = Q::init(q);
        let mut path = NodePath::default();
        let mut node_idx = self.root;
        let mut child_arr_pos = 0;
        while let ArenaIndex::Internal(node_idx_inner) = node_idx {
            path.push(Idx {
                arena: ArenaIndex::Internal(node_idx_inner),
                arr: child_arr_pos,
            })
            .unwrap();
            let node = self.in_nodes.get(node_idx_inner).unwrap();
            let result = finder.find_node(q, &node.children);
            child_arr_pos = result.index as u8;
            node_idx = node.children[result.index].arena;
        }

        let (offset, found) = {
            let leaf = self.get_leaf(node_idx);
            finder.confirm_elem(q, &leaf.elem)
        };
        let ans = QueryResult {
            cursor: Cursor {
                leaf: node_idx.unwrap_leaf().into(),
                offset,
            },
            found,
        };
        let (diff, new_insert_1, new_insert_2, deleted, next_leaf) = {
            let leaf = self.get_leaf_mut(node_idx);
            let Some((diff, new_insert_1, new_insert_2)) = f(&mut leaf.elem, ans) else {
                return (Some(ans.cursor), splitted);
            };
            (
                diff,
                new_insert_1,
                new_insert_2,
                leaf.elem.can_remove(),
                leaf.next_leaf(),
            )
        };

        if new_insert_2.is_some() {
            unimplemented!()
        }

        // Delete
        if deleted {
            // handle deletion
            // leaf node should be deleted
            assert!(new_insert_1.is_none());
            assert!(new_insert_2.is_none());
            self.unlink_leaf(ans.cursor.leaf);
            self.leaf_nodes.remove(node_idx.unwrap()).unwrap();
            let mut is_first = true;
            let mut is_child_lack = false;
            let mut child_idx = node_idx;

            // iterate from leaf to root, child to parent
            while let Some(Idx {
                arena: parent_idx,
                arr: parent_arr_pos,
            }) = path.pop()
            {
                let removed_child = is_first;
                let is_lack = {
                    let parent = self.get_internal_mut(parent_idx);
                    if removed_child {
                        parent.children.remove(child_arr_pos as usize);
                    } else {
                        B::apply_cache_diff(
                            &mut parent.children[child_arr_pos as usize].cache,
                            &diff,
                        );
                    }
                    parent.is_lack()
                };
                if removed_child {
                    self.update_children_parent_slot_from(parent_idx, child_arr_pos as usize);
                    is_first = false;
                }

                if is_child_lack {
                    self.handle_lack_single_layer(child_idx);
                }

                is_child_lack = is_lack;
                child_idx = parent_idx;
                child_arr_pos = parent_arr_pos;
            }

            B::apply_cache_diff(&mut self.root_cache, &diff);

            if is_child_lack {
                let root = self.get_internal_mut(self.root);
                if root.children.len() == 1 && !root.is_child_leaf() {
                    self.try_reduce_levels();
                }
            }

            return (None, splitted);
        }

        let mut new_cache_and_child = None;
        if let Some(new_insert_1) = new_insert_1 {
            let cache = {
                let leaf = self.get_leaf(node_idx);
                B::get_elem_cache(&leaf.elem)
            };
            let child = self.alloc_leaf_child(
                new_insert_1,
                path.last().unwrap().arena.unwrap(),
                child_arr_pos + 1,
                Some(ans.cursor.leaf),
                next_leaf,
            );
            splitted.push(child.arena);
            new_cache_and_child = Some((cache, child));
        }

        while let Some(Idx {
            arena: parent_idx,
            arr: parent_arr_pos,
        }) = path.pop()
        {
            match take(&mut new_cache_and_child) {
                Some((cache, child)) => {
                    let insert_pos = child_arr_pos as usize + 1;
                    let (inserted_arena, is_full) = {
                        let parent = self.get_internal_mut(parent_idx);
                        parent.children[child_arr_pos as usize].cache = cache;
                        let arena = child.arena;
                        parent.children.insert(insert_pos, child).unwrap();
                        (arena, parent.is_full())
                    };
                    self.set_child_parent_and_slot(
                        inserted_arena,
                        parent_idx.unwrap_internal(),
                        insert_pos,
                    );
                    self.update_children_parent_slot_from(parent_idx, insert_pos + 1);
                    if is_full {
                        let (_, _, this_cache, right_child) = self.split_node(parent_idx, None);
                        new_cache_and_child = Some((this_cache, right_child));
                    }
                }
                None => {
                    let parent = self.get_internal_mut(parent_idx);
                    B::apply_cache_diff(&mut parent.children[child_arr_pos as usize].cache, &diff);
                }
            }

            child_arr_pos = parent_arr_pos;
        }

        if let Some((cache, child)) = new_cache_and_child {
            self.split_root(cache, child);
        } else {
            B::apply_cache_diff(&mut self.root_cache, &diff);
        }

        (Some(ans.cursor), splitted)
    }

    /// Update leaf node's elements, return true if cache need to be updated
    ///
    /// `f` returns (is_cache_updated, cache_diff, new_insert_1, new_insert2)
    ///
    /// - If leaf_node.can_remove(), it will be removed from the tree.
    ///
    /// Returns true if the node_idx is still valid. (If the leaf node is removed, it will return false).
    pub fn update_leaf(
        &mut self,
        node_idx: LeafIndex,
        f: impl FnOnce(&mut B::Elem) -> (bool, Option<B::Elem>, Option<B::Elem>),
    ) -> (bool, SplittedLeaves) {
        let mut splitted = SplittedLeaves::default();
        let old_cache = {
            let node = self.leaf_nodes.get(node_idx.0).unwrap();
            B::get_elem_cache(&node.elem)
        };
        let node = self.leaf_nodes.get_mut(node_idx.0).unwrap();
        let mut parent_idx = node.parent();
        let (_need_update_cache, mut new_insert_1, mut new_insert_2) = f(&mut node.elem);
        {
            // Normalize returned values
            //
            // If the node can be removed, then both new_insert_1 & new_insert_2 should be None
            // The priority is node.elem > new_insert_1 > new_insert_2
            //
            // And new_insert_1 and new_insert_2 should not match `can_remove` condition
            if let Some(ref new_1) = new_insert_1 {
                if new_1.can_remove() {
                    new_insert_1 = new_insert_2.take();
                    if let Some(ref new_1) = new_insert_1 {
                        if new_1.can_remove() {
                            new_insert_1 = None;
                        }
                    }
                }
            }

            if let Some(ref new_2) = new_insert_2 {
                if new_2.can_remove() {
                    new_insert_2 = None;
                } else if new_insert_1.is_none() {
                    std::mem::swap(&mut new_insert_1, &mut new_insert_2);
                }
            }

            if node.elem.can_remove() {
                if let Some(new_1) = new_insert_1 {
                    node.elem = new_1;
                    new_insert_1 = new_insert_2.take();
                }
            }
        }

        let deleted = node.elem.can_remove();
        let next_leaf = node.next_leaf();
        let current_cache_diff = |tree: &Self| {
            let node = tree.leaf_nodes.get(node_idx.0).unwrap();
            let new_cache = B::get_elem_cache(&node.elem);
            (new_cache != old_cache).then(|| B::sub_cache(&new_cache, &old_cache))
        };

        if deleted {
            debug_assert!(new_insert_1.is_none());
            debug_assert!(new_insert_2.is_none());
            if let Some(diff) = current_cache_diff(self) {
                self.recursive_update_cache(
                    node_idx.into(),
                    B::USE_DIFF,
                    if B::USE_DIFF { Some(diff) } else { None },
                );
            }
            let slot = self.leaf_nodes.get(node_idx.0).unwrap().parent_slot();
            self.unlink_leaf(node_idx);
            self.leaf_nodes.remove(node_idx.0).unwrap();
            let is_lack = {
                let parent = self.in_nodes.get_mut(parent_idx.unwrap()).unwrap();
                parent.children.remove(slot);
                parent.is_lack()
            };
            self.update_children_parent_slot_from(parent_idx, slot);
            if is_lack {
                self.handle_lack_recursively(parent_idx);
            }

            (false, splitted)
        } else if new_insert_1.is_none() {
            debug_assert!(new_insert_2.is_none());
            if let Some(diff) = current_cache_diff(self) {
                self.recursive_update_cache(
                    node_idx.into(),
                    B::USE_DIFF,
                    if B::USE_DIFF { Some(diff) } else { None },
                );
            }
            return (true, splitted);
        } else {
            if new_insert_1.is_some() && new_insert_2.is_none() {
                // try merge new insert to next element
                if let Some(next_idx) = next_leaf {
                    let new = new_insert_1.as_ref().unwrap();
                    let can_merge_next = {
                        let next = self.get_elem(next_idx).unwrap();
                        new.can_merge(next)
                    };
                    if can_merge_next {
                        if let Some(diff) = current_cache_diff(self) {
                            self.recursive_update_cache(
                                node_idx.into(),
                                B::USE_DIFF,
                                if B::USE_DIFF { Some(diff) } else { None },
                            );
                        }
                        let next = self.get_elem_mut(next_idx).unwrap();
                        next.merge_left(new);
                        self.recursive_update_cache(next_idx.into(), B::USE_DIFF, None);
                        splitted.push(next_idx.into());
                        return (true, splitted);
                    }
                }
            }

            if let Some(diff) = current_cache_diff(self) {
                self.recursive_update_cache(
                    node_idx.into(),
                    B::USE_DIFF,
                    if B::USE_DIFF { Some(diff) } else { None },
                );
            }

            let count = if new_insert_1.is_some() { 1 } else { 0 }
                + if new_insert_2.is_some() { 1 } else { 0 };
            let parent = self.in_nodes.get_mut(parent_idx.unwrap()).unwrap();
            let parent = if parent.children.len() + count >= MAX_CHILDREN_NUM {
                self.split(parent_idx);
                let node = self.leaf_nodes.get(node_idx.0).unwrap();
                parent_idx = node.parent();
                self.in_nodes.get_mut(parent_idx.unwrap()).unwrap()
            } else {
                parent
            };
            let slot = self.leaf_nodes.get(node_idx.0).unwrap().parent_slot();
            let mut new: HeaplessVec<Child<B>, 2> = HeaplessVec::new();
            let mut new_leaf_links: HeaplessVec<
                (LeafIndex, Option<LeafIndex>, Option<LeafIndex>),
                2,
            > = HeaplessVec::new();
            let mut prev_leaf = Some(node_idx);
            for (i, elem) in new_insert_1.into_iter().chain(new_insert_2).enumerate() {
                let parent_index = parent_idx.unwrap();
                let elem_cache = B::get_elem_cache(&elem);
                let new_leaf_index = {
                    let leaf = LeafNode {
                        elem,
                        parent: parent_index,
                        parent_slot: (slot + 1 + i) as u8,
                        prev_leaf: None,
                        next_leaf: None,
                    };
                    self.leaf_nodes.insert(leaf)
                };
                let new_leaf = LeafIndex::from(new_leaf_index);
                let next_for_new = if i + 1 == count { next_leaf } else { None };
                new_leaf_links
                    .push((new_leaf, prev_leaf, next_for_new))
                    .unwrap();
                prev_leaf = Some(new_leaf);
                new.push(Child {
                    arena: ArenaIndex::Leaf(new_leaf_index),
                    cache: elem_cache,
                })
                .unwrap();
            }
            for (i, v) in new.into_iter().enumerate() {
                splitted.push(v.arena);
                parent.children.insert(slot + 1 + i, v).unwrap();
            }
            let is_full = parent.is_full();
            let _ = parent;
            self.update_children_parent_slot_from(parent_idx, slot + 1 + count);
            for (leaf, prev_leaf, next_leaf) in new_leaf_links {
                self.link_leaf_between(leaf, prev_leaf, next_leaf);
            }

            assert!(!is_full);
            self.recursive_update_cache(parent_idx, B::USE_DIFF, None);
            (true, splitted)
        }
    }

    /// Update the given leaves with the given function in the given range.
    ///
    /// - The range descibes the range inside the leaf node.
    /// - There can be multiple ranges in the same leaf node.
    /// - The cahce will be recalculated for each affected node
    /// - It doesn't guarantee the applying order
    ///
    /// Currently, the time complexity is O(m^2) for each leaf node,
    /// where m is the number of ranges inside the same leaf node.
    /// If we have a really large m, this function need to be optimized.
    pub fn update_leaves_with_arg_in_ranges<A: Debug>(
        &mut self,
        mut args: Vec<(LeafIndex, Range<usize>, A)>,
        mut f: impl FnMut(&mut B::Elem, &A),
    ) -> Vec<LeafIndex> {
        args.sort_by_key(|x| x.0);
        let mut new_leaves = Vec::new();
        let mut dirty_map: LeafDirtyMap<B::CacheDiff> = Default::default();
        let mut new_elems_at_cursor: FxHashMap<Cursor, Vec<B::Elem>> = Default::default();
        for (leaf, group) in &args.into_iter().group_by(|x| x.0) {
            // This loop doesn't change the shape of the tree.  It only changes each leaf element.
            // A leaf element may be splitted into several parts. The first part stay in the tree,
            // while the rest of them are inserted into `new_elem_at_cursor`, which will be inserted
            // into the tree later.
            let leaf_node = self.leaf_nodes.get_mut(leaf.0).unwrap();
            let len = leaf_node.elem().rle_len();
            let mut split_at = BTreeSet::new();
            // PERF we can avoid this alloc and `group_by`
            let group: Vec<_> = group.into_iter().collect();
            for (_, range, _) in group.iter() {
                split_at.insert(range.start);
                split_at.insert(range.end);
            }

            split_at.remove(&0);
            split_at.remove(&len);

            // leaf_node.elem is the first elem
            let old_cache = B::get_elem_cache(&leaf_node.elem);
            if split_at.is_empty() {
                // doesn't need to split
                for (_, range, a) in group.iter() {
                    assert_eq!(range.start, 0);
                    assert_eq!(range.end, len);
                    f(&mut leaf_node.elem, a);
                }
            } else {
                let mut new_elems = Vec::new();

                let first_split = split_at.first().copied().unwrap();

                // handle first element
                let mut elem = leaf_node.elem.split(first_split);
                for (_, r, a) in group.iter() {
                    if r.start == 0 {
                        f(&mut leaf_node.elem, a);
                    }
                }

                // handle elements in the middle
                let mut last_index = first_split;
                for &index in split_at.iter().skip(1) {
                    let next_elem = elem.split(index - last_index);
                    let cur_range = last_index..index;
                    for (_, r, a) in group.iter() {
                        if r.start <= cur_range.start && cur_range.end <= r.end {
                            f(&mut elem, a);
                        }
                    }

                    new_elems.push(elem);
                    elem = next_elem;
                    last_index = index;
                }

                // handle the last element
                for (_, r, a) in group.iter() {
                    if r.end == len {
                        f(&mut elem, a);
                    }
                }

                new_elems.push(elem);
                new_elems_at_cursor.insert(
                    Cursor {
                        leaf,
                        offset: leaf_node.elem().rle_len(),
                    },
                    new_elems,
                );
            }

            let new_cache = B::get_elem_cache(&leaf_node.elem);
            let diff = B::sub_cache(&new_cache, &old_cache);
            dirty_map.insert(leaf.into(), diff);
        }

        // update cache
        self.update_dirty_cache_map(dirty_map);

        // PERF we can use batch insert to optimize this
        // insert the new leaf nodes
        for (mut cursor, elems) in new_elems_at_cursor {
            for elem in elems.into_iter() {
                // PERF can use insert many to optimize when it's supported
                let result = self.insert_by_path(cursor, elem);
                let len = self.get_elem(result.0.leaf).unwrap().rle_len();
                new_leaves.push(result.0.leaf);
                debug_assert_eq!(result.1.arr.len(), 0);
                cursor = Cursor {
                    leaf: result.0.leaf,
                    offset: len,
                };
            }
        }

        new_leaves
    }

    fn update_root_cache(&mut self) {
        self.in_nodes
            .get(self.root.unwrap_internal())
            .unwrap()
            .calc_cache(&mut self.root_cache, None);
    }

    fn update_dirty_cache_map(&mut self, mut diff_map: LeafDirtyMap<B::CacheDiff>) {
        // diff_map only contains leaf nodes when this function is called
        let mut visit_set: FxHashSet<ArenaIndex> = diff_map.keys().copied().collect();
        while !visit_set.is_empty() {
            for child_idx in take(&mut visit_set) {
                let (parent_idx, cache_diff) = match child_idx {
                    ArenaIndex::Leaf(leaf_idx) => {
                        let node = self.leaf_nodes.get(leaf_idx).unwrap();
                        let parent_idx = node.parent;
                        let parent = self.in_nodes.get_mut(parent_idx).unwrap();
                        let cache_diff = diff_map.remove(&child_idx).unwrap();
                        B::apply_cache_diff(
                            &mut parent.children[node.parent_slot()].cache,
                            &cache_diff,
                        );

                        (ArenaIndex::Internal(parent_idx), cache_diff)
                    }
                    ArenaIndex::Internal(_) => {
                        let node = self.in_nodes.get(child_idx.unwrap_internal()).unwrap();
                        let Some(parent_idx) = node.parent else {
                            continue;
                        };
                        let (child, parent) = self.get2_mut(child_idx, parent_idx);
                        let cache_diff = child.calc_cache(
                            &mut parent.children[child.parent_slot as usize].cache,
                            diff_map.remove(&child_idx),
                        );

                        (parent_idx, cache_diff)
                    }
                };

                visit_set.insert(parent_idx);
                if let Some(e) = diff_map.get_mut(&parent_idx) {
                    B::merge_cache_diff(e, &cache_diff);
                } else {
                    diff_map.insert(parent_idx, cache_diff);
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
        self.update_children_parent_slot_from(internal_node, 0);
    }

    pub fn iter(&self) -> impl Iterator<Item = &B::Elem> + '_ {
        let mut current = self.first_leaf();
        core::iter::from_fn(move || {
            let leaf = current?;
            let node = self.leaf_nodes.get(leaf.0).unwrap();
            current = node.next_leaf();
            Some(&node.elem)
        })
    }

    pub fn drain(&mut self, range: Range<QueryResult>) -> iter::Drain<'_, B> {
        iter::Drain::new(self, Some(range.start), Some(range.end))
    }

    pub fn drain_by_query<Q: Query<B>>(&mut self, range: Range<Q::QueryArg>) -> iter::Drain<'_, B> {
        let start = self.query::<Q>(&range.start);
        let end = self.query::<Q>(&range.end);
        iter::Drain::new(self, start, end)
    }

    pub fn first_leaf(&self) -> Option<LeafIndex> {
        self.first_leaf
    }

    pub fn last_leaf(&self) -> Option<LeafIndex> {
        self.last_leaf
    }

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
        range: impl RangeBounds<Cursor>,
    ) -> impl Iterator<Item = ElemSlice<'_, B::Elem>> + '_ {
        let start = match range.start_bound() {
            std::ops::Bound::Included(start) => *start,
            std::ops::Bound::Excluded(_) => unreachable!(),
            std::ops::Bound::Unbounded => self.start_cursor().unwrap(),
        };
        let (inclusive, end) = match range.end_bound() {
            std::ops::Bound::Included(end) => (true, *end),
            std::ops::Bound::Excluded(end) => (false, *end),
            std::ops::Bound::Unbounded => (true, self.end_cursor().unwrap()),
        };
        self._iter_range(start, end, inclusive)
    }

    fn _iter_range(
        &self,
        start: Cursor,
        end: Cursor,
        inclusive_end: bool,
    ) -> impl Iterator<Item = ElemSlice<'_, B::Elem>> + '_ {
        let node_iter = iter::Iter::new(self, start.leaf, end.leaf);
        node_iter.filter_map(move |(leaf, node)| {
            if end.leaf == leaf && end.offset == 0 && !inclusive_end {
                return None;
            }

            Some(ElemSlice {
                cursor: Cursor { leaf, offset: 0 },
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
            })
        })
    }

    pub fn start_cursor(&self) -> Option<Cursor> {
        Some(Cursor {
            leaf: self.first_leaf()?,
            offset: 0,
        })
    }

    pub fn end_cursor(&self) -> Option<Cursor> {
        let leaf = self.last_leaf()?;
        let node = self.get_leaf(leaf.into());
        Some(Cursor {
            leaf,
            offset: node.elem.rle_len(),
        })
    }

    /// Split the internal node at path into two nodes recursively upwards.
    ///
    // at call site the cache at path can be out-of-date.
    // the cache will be up-to-date after this method
    fn split(&mut self, node_idx: ArenaIndex) {
        self.split_at(node_idx, None)
    }

    fn split_at(&mut self, node_idx: ArenaIndex, at: Option<usize>) {
        let (node_parent, node_parent_slot, this_cache, right_child) =
            self.split_node(node_idx, at);

        self.inner_insert_node(
            node_parent,
            node_parent_slot as usize,
            this_cache,
            right_child,
        );
        // don't need to recursive update cache
    }

    fn split_node(
        &mut self,
        node_idx: ArenaIndex,
        at: Option<usize>,
    ) -> (Option<ArenaIndex>, u8, <B as BTreeTrait>::Cache, Child<B>) {
        let (node_parent, node_parent_slot, right_children) = {
            let node = self.in_nodes.get_mut(node_idx.unwrap_internal()).unwrap();
            let node_parent = node.parent;
            let node_parent_slot = node.parent_slot;

            let split = at.unwrap_or(node.children.len() / 2);
            let mut right_children = HeaplessVec::new();
            while node.children.len() > split {
                right_children.push(node.children.pop().unwrap()).unwrap();
            }
            right_children.reverse();

            (node_parent, node_parent_slot, right_children)
        };

        // update cache
        let mut right_cache = B::Cache::default();
        let right_arena_idx = self.in_nodes.insert(Node {
            parent: node_parent,
            parent_slot: u8::MAX,
            children: right_children,
        });
        let this_cache = {
            let node = self.get_internal_mut(node_idx);
            let mut cache = Default::default();
            node.calc_cache(&mut cache, None);
            cache
        };

        let right = self.in_nodes.get_mut(right_arena_idx).unwrap();
        // update parent cache
        right.calc_cache(&mut right_cache, None);
        self.update_children_parent_slot_from(ArenaIndex::Internal(right_arena_idx), 0);
        let right_child = Child {
            arena: ArenaIndex::Internal(right_arena_idx),
            cache: right_cache,
        };
        (node_parent, node_parent_slot, this_cache, right_child)
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
            let insert_pos = index + 1;
            let (inserted_arena, is_full) = {
                let parent = self.get_internal_mut(parent_idx);
                parent.children[index].cache = new_cache;
                let arena = node.arena;
                parent.children.insert(insert_pos, node).unwrap();
                (arena, parent.is_full())
            };
            self.set_child_parent_and_slot(
                inserted_arena,
                parent_idx.unwrap_internal(),
                insert_pos,
            );
            self.update_children_parent_slot_from(parent_idx, index + 2);
            if is_full {
                self.split(parent_idx);
            }
        } else {
            self.split_root(new_cache, node);
        }
    }

    /// Update the `parent_slot` fields in `children[index..]`
    fn update_children_parent_slot_from(&mut self, parent_idx: ArenaIndex, index: usize) {
        let parent_index = parent_idx.unwrap_internal();
        let mut children: HeaplessVec<ArenaIndex, MAX_CHILDREN_NUM> = HeaplessVec::new();
        {
            let parent = self.get_internal(parent_idx);
            if parent.children.len() <= index {
                return;
            }
            for child in parent.children[index..].iter() {
                children.push(child.arena).unwrap();
            }
        }

        for (offset, child) in children.into_iter().enumerate() {
            self.set_child_parent_and_slot(child, parent_index, index + offset);
        }
    }

    /// right's cache should be up-to-date
    fn split_root(&mut self, new_cache: B::Cache, right: Child<B>) {
        let root_idx = self.root;
        // set right parent
        let right_node = &mut self.get_internal_mut(right.arena);
        right_node.parent_slot = 1;
        right_node.parent = Some(root_idx);
        let left_arena = {
            let root = self.get_internal_mut(self.root);
            // let left be root
            let mut left_node: Node<B> = core::mem::replace(
                root,
                Node {
                    parent: None,
                    parent_slot: 0,
                    children: Default::default(),
                },
            );
            left_node.parent_slot = 0;
            // set left parent
            left_node.parent = Some(root_idx);
            self.in_nodes.insert(left_node)
        };

        let left = Child::new(ArenaIndex::Internal(left_arena), new_cache);
        let mut cache = std::mem::take(&mut self.root_cache);
        let root = self.get_internal_mut(self.root);
        root.children = Default::default();
        root.children.push(left).unwrap();
        root.children.push(right).unwrap();

        // update new root cache
        root.calc_cache(&mut cache, None);
        self.root_cache = cache;
        self.update_children_parent_slot_from(ArenaIndex::Internal(left_arena), 0);
    }

    #[inline]
    pub fn get_internal_mut(&mut self, index: ArenaIndex) -> &mut Node<B> {
        self.in_nodes.get_mut(index.unwrap_internal()).unwrap()
    }

    #[inline]
    pub fn get_leaf_mut(&mut self, index: ArenaIndex) -> &mut LeafNode<B::Elem> {
        self.leaf_nodes.get_mut(index.unwrap_leaf()).unwrap()
    }

    #[inline]
    fn get2_mut(&mut self, a: ArenaIndex, b: ArenaIndex) -> (&mut Node<B>, &mut Node<B>) {
        let (a, b) = self
            .in_nodes
            .get2_mut(a.unwrap_internal(), b.unwrap_internal());
        (a.unwrap(), b.unwrap())
    }

    /// # Panic
    ///
    /// If the given index is not valid or deleted
    #[inline]
    pub fn get_internal(&self, index: ArenaIndex) -> &Node<B> {
        self.in_nodes.get(index.unwrap_internal()).unwrap()
    }

    #[inline]
    pub fn get_leaf(&self, index: ArenaIndex) -> &LeafNode<B::Elem> {
        self.leaf_nodes.get(index.unwrap_leaf()).unwrap()
    }

    /// The given node is lack of children.
    /// We should merge it into its neighbor or borrow from its neighbor.
    ///
    /// Given a random neighbor is neither full or lack, it's guaranteed
    /// that we can either merge into or borrow from it without breaking
    /// the balance rule.
    ///
    /// - The caches in parent's subtree should be up-to-date when calling this.
    /// - The caches in the parent node will be updated
    fn handle_lack_recursively(&mut self, node_idx: ArenaIndex) {
        let mut lack_info = self.handle_lack_single_layer(node_idx);
        while let Some(parent) = lack_info.parent_lack {
            lack_info = self.handle_lack_single_layer(parent);
        }
    }

    /// The given node is lack of children. This method doesn't handle parent's lack.
    ///
    /// - The caches in parent's subtree should be up-to-date when calling this.
    /// - The caches in the parent node will be updated
    fn handle_lack_single_layer(&mut self, node_idx: ArenaIndex) -> LackInfo {
        if self.root == node_idx {
            self.try_reduce_levels();
            return LackInfo { parent_lack: None };
        }

        let node = self.get_internal(node_idx);
        let parent_idx = node.parent.unwrap();
        let parent = self.get_internal(parent_idx);
        debug_assert_eq!(parent.children[node.parent_slot as usize].arena, node_idx);
        if node.children.is_empty() {
            let slot = node.parent_slot as usize;
            self.get_internal_mut(parent_idx).children.remove(slot);
            self.in_nodes.remove(node_idx.unwrap_internal());
            self.update_children_parent_slot_from(parent_idx, slot);
            return LackInfo {
                parent_lack: Some(parent_idx),
            };
        }
        let ans = match self.pair_neighbor(node_idx) {
            Some((a_idx, b_idx)) => {
                let parent = self.get_internal_mut(parent_idx);
                let mut a_cache = std::mem::take(&mut parent.children[a_idx.arr as usize].cache);
                let mut b_cache = std::mem::take(&mut parent.children[b_idx.arr as usize].cache);
                let mut re_parent = FxHashMap::default();

                let (a, b) = self
                    .in_nodes
                    .get2_mut(a_idx.arena.unwrap_internal(), b_idx.arena.unwrap_internal());
                let a = a.unwrap();
                let b = b.unwrap();
                let ans = if a.len() + b.len() >= MAX_CHILDREN_NUM {
                    // move partially
                    if a.len() < b.len() {
                        // move part of b's children to a
                        let move_len = (b.len() - a.len()) / 2;
                        for child in &b.children[..move_len] {
                            re_parent.insert(child.arena, (a_idx.arena, a.children.len()));
                            a.children.push(child.clone()).unwrap();
                        }
                        delete_range(&mut b.children, ..move_len);
                        for (i, child) in b.children.iter().enumerate() {
                            re_parent.insert(child.arena, (b_idx.arena, i));
                        }
                    } else {
                        // move part of a's children to b
                        let move_len = (a.len() - b.len()) / 2;
                        for (i, child) in b.children.iter().enumerate() {
                            re_parent.insert(child.arena, (b_idx.arena, i + move_len));
                        }
                        let mut b_children =
                            HeaplessVec::from_slice(&a.children[a.children.len() - move_len..])
                                .unwrap();
                        for child in take(&mut b.children) {
                            b_children.push(child).unwrap();
                        }
                        b.children = b_children;
                        for (i, child) in b.children.iter().enumerate() {
                            re_parent.insert(child.arena, (b_idx.arena, i));
                        }
                        let len = a.children.len();
                        delete_range(&mut a.children, len - move_len..);
                    }
                    a.calc_cache(&mut a_cache, None);
                    b.calc_cache(&mut b_cache, None);
                    let parent = self.get_internal_mut(parent_idx);
                    parent.children[a_idx.arr as usize].cache = a_cache;
                    parent.children[b_idx.arr as usize].cache = b_cache;
                    LackInfo {
                        parent_lack: if parent.is_lack() {
                            Some(parent_idx)
                        } else {
                            None
                        },
                    }
                } else {
                    // merge
                    let is_parent_lack = if node_idx == a_idx.arena {
                        // merge b to a, delete b
                        for (i, child) in b.children.iter().enumerate() {
                            re_parent.insert(child.arena, (a_idx.arena, a.children.len() + i));
                        }

                        for child in take(&mut b.children) {
                            a.children.push(child).unwrap();
                        }

                        a.calc_cache(&mut a_cache, None);
                        let parent = self.get_internal_mut(parent_idx);
                        parent.children[a_idx.arr as usize].cache = a_cache;
                        parent.children.remove(b_idx.arr as usize);
                        let is_lack = parent.is_lack();
                        self.purge(b_idx.arena);
                        self.update_children_parent_slot_from(parent_idx, b_idx.arr as usize);
                        is_lack
                    } else {
                        // merge a to b, delete a
                        for (i, child) in a.children.iter().enumerate() {
                            re_parent.insert(child.arena, (b_idx.arena, i));
                        }
                        for (i, child) in b.children.iter().enumerate() {
                            re_parent.insert(child.arena, (b_idx.arena, i + a.children.len()));
                        }

                        for child in take(&mut b.children) {
                            a.children.push(child).unwrap();
                        }

                        b.children = take(&mut a.children);
                        b.calc_cache(&mut b_cache, None);
                        let parent = self.get_internal_mut(parent_idx);
                        parent.children[b_idx.arr as usize].cache = b_cache;
                        parent.children.remove(a_idx.arr as usize);
                        let is_lack = parent.is_lack();
                        self.purge(a_idx.arena);
                        self.update_children_parent_slot_from(parent_idx, a_idx.arr as usize);
                        is_lack
                    };

                    LackInfo {
                        parent_lack: if is_parent_lack {
                            Some(parent_idx)
                        } else {
                            None
                        },
                    }
                };

                // FIXME: make this work
                if cfg!(debug_assertions) {
                    // let (a, b) = self
                    //     .in_nodes
                    //     .get2_mut(a_idx.arena.unwrap_internal(), b_idx.arena.unwrap_internal());
                    // if let Some(a) = a {
                    //     assert!(!a.is_lack() && !a.is_full());
                    // }
                    // if let Some(b) = b {
                    //     assert!(!b.is_lack() && !b.is_full());
                    // }
                }

                for (child, (parent, slot)) in re_parent {
                    self.set_child_parent_and_slot(child, parent.unwrap_internal(), slot);
                }
                ans
            }
            None => LackInfo {
                parent_lack: Some(parent_idx),
            },
        };
        ans
    }

    fn try_reduce_levels(&mut self) {
        let mut reduced = false;
        while self.get_internal(self.root).children.len() == 1 {
            let root = self.get_internal(self.root);
            if root.is_child_leaf() {
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
        self.update_children_parent_slot_from(parent_idx, 0);
    }

    fn pair_neighbor(&self, this: ArenaIndex) -> Option<(Idx, Idx)> {
        let node = self.get_internal(this);
        let arr = node.parent_slot as usize;
        let parent = self.get_internal(node.parent.unwrap());

        if arr == 0 {
            parent
                .children
                .get(1)
                .map(|x| (Idx::new(this, arr as u8), Idx::new(x.arena, 1)))
        } else {
            parent
                .children
                .get(arr - 1)
                .map(|x| (Idx::new(x.arena, arr as u8 - 1), Idx::new(this, arr as u8)))
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
            let (cache, parent_idx, slot) = {
                let leaf = self.leaf_nodes.get(index).unwrap();
                (
                    B::get_elem_cache(&leaf.elem),
                    leaf.parent(),
                    leaf.parent_slot(),
                )
            };
            node_idx = parent_idx;
            let node = self.get_internal_mut(node_idx);
            node.children[slot].cache = cache;
        }

        if can_use_diff {
            if let Some(diff) = cache_diff {
                return self.recursive_update_cache_with_diff(node_idx, diff);
            }
        }

        let mut this_idx = node_idx;
        let mut node = self.get_internal_mut(node_idx);
        let mut this_arr = node.parent_slot;
        if can_use_diff {
            if node.parent.is_some() {
                let parent_idx = node.parent.unwrap();
                let (parent, this) = self.get2_mut(parent_idx, this_idx);
                let diff =
                    this.calc_cache(&mut parent.children[this_arr as usize].cache, cache_diff);
                return self.recursive_update_cache_with_diff(parent_idx, diff);
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
        root.calc_cache(
            &mut root_cache,
            if can_use_diff { cache_diff } else { None },
        );
        self.root_cache = root_cache;
    }

    fn recursive_update_cache_with_diff(&mut self, node_idx: ArenaIndex, diff: B::CacheDiff) {
        let mut node = self.get_internal_mut(node_idx);
        let mut this_arr = node.parent_slot;
        while node.parent.is_some() {
            let parent_idx = node.parent.unwrap();
            let parent = self.get_internal_mut(parent_idx);
            B::apply_cache_diff(&mut parent.children[this_arr as usize].cache, &diff);
            this_arr = parent.parent_slot;
            node = parent;
        }

        B::apply_cache_diff(&mut self.root_cache, &diff);
    }

    fn purge(&mut self, index: ArenaIndex) {
        let mut stack = vec![index];
        while let Some(x) = stack.pop() {
            if let ArenaIndex::Leaf(index) = x {
                if self.leaf_nodes.contains(index) {
                    self.unlink_leaf(index.into());
                }
                self.leaf_nodes.remove(index);

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

    fn next_same_level_in_node(&self, node_idx: ArenaIndex) -> Option<ArenaIndex> {
        match node_idx {
            ArenaIndex::Leaf(leaf_idx) => self
                .leaf_nodes
                .get(leaf_idx)
                .unwrap()
                .next_leaf()
                .map(ArenaIndex::from),
            ArenaIndex::Internal(_) => {
                let node = self.get_internal(node_idx);
                let parent = self.get_internal(node.parent?);
                if let Some(next) = parent.children.get(node.parent_slot as usize + 1) {
                    Some(next.arena)
                } else if let Some(parent_next) = self.next_same_level_in_node(node.parent?) {
                    let parent_next = self.get_internal(parent_next);
                    parent_next.children.first().map(|x| x.arena)
                } else {
                    None
                }
            }
        }
    }

    /// find the next element in the tree
    pub fn next_elem(&self, path: Cursor) -> Option<Cursor> {
        self.next_leaf_index(path.leaf)
            .map(|leaf| Cursor { leaf, offset: 0 })
    }

    pub fn prev_elem(&self, path: Cursor) -> Option<Cursor> {
        self.prev_leaf_index(path.leaf)
            .map(|leaf| Cursor { leaf, offset: 0 })
    }

    #[inline(always)]
    pub fn root_cache(&self) -> &B::Cache {
        &self.root_cache
    }

    /// This method will release the memory back to OS.
    /// Currently, it's just `*self = Self::new()`
    #[inline(always)]
    pub fn clear(&mut self) {
        *self = Self::new();
    }

    #[inline(always)]
    fn root_mut(&mut self) -> &mut Node<B> {
        self.get_internal_mut(self.root)
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.get_internal(self.root).is_empty()
    }

    fn get_path(&self, idx: ArenaIndex) -> NodePath {
        let mut path = NodePath::new();
        let mut node_idx = idx;
        while node_idx != self.root {
            match node_idx {
                ArenaIndex::Leaf(inner_node_idx) => {
                    let node = self.leaf_nodes.get(inner_node_idx).unwrap();
                    path.push(Idx::new(node_idx, node.parent_slot as u8))
                        .unwrap();
                    node_idx = ArenaIndex::Internal(node.parent);
                }
                ArenaIndex::Internal(_) => {
                    let node = self.get_internal(node_idx);
                    path.push(Idx::new(node_idx, node.parent_slot)).unwrap();
                    node_idx = node.parent.unwrap();
                }
            }
        }
        path.push(Idx::new(self.root, 0)).unwrap();
        path.reverse();
        path
    }

    pub fn push(&mut self, elem: B::Elem) -> Cursor {
        let mut is_full = false;
        let mut parent_idx = self.root;
        let mut update_cache_idx = parent_idx;
        let mut can_use_diff = B::USE_DIFF;
        let cache = B::get_elem_cache(&elem);
        let ans = if self.is_empty() {
            can_use_diff = false;
            let data = self.alloc_leaf_child(elem, parent_idx.unwrap(), 0, None, None);
            let parent = self.in_nodes.get_mut(parent_idx.unwrap()).unwrap();
            let ans = data.arena;
            parent.children.push(data).unwrap();
            Cursor {
                leaf: ans.unwrap().into(),
                offset: 0,
            }
        } else {
            let leaf_idx = self.last_leaf().unwrap();
            let leaf = self.leaf_nodes.get_mut(leaf_idx.0).unwrap();
            parent_idx = leaf.parent();
            if leaf.elem.can_merge(&elem) {
                update_cache_idx = leaf_idx.into();
                let offset = leaf.elem.rle_len();
                leaf.elem.merge_right(&elem);
                Cursor {
                    leaf: leaf_idx,
                    offset,
                }
            } else {
                let slot = self
                    .in_nodes
                    .get(parent_idx.unwrap())
                    .unwrap()
                    .children
                    .len() as u8;
                let data =
                    self.alloc_leaf_child(elem, parent_idx.unwrap(), slot, self.last_leaf(), None);
                let parent = self.in_nodes.get_mut(parent_idx.unwrap()).unwrap();
                let ans = data.arena;
                update_cache_idx = parent_idx;
                can_use_diff = false;
                parent.children.push(data).unwrap();
                is_full = parent.is_full();
                Cursor {
                    leaf: ans.unwrap().into(),
                    offset: 0,
                }
            }
        };

        self.recursive_update_cache(
            update_cache_idx,
            can_use_diff,
            if can_use_diff {
                Some(B::new_cache_to_diff(&cache))
            } else {
                None
            },
        );
        if is_full {
            self.split(parent_idx);
        }

        ans
    }

    pub fn prepend(&mut self, elem: B::Elem) -> Cursor {
        let Some(leaf_idx) = self.first_leaf() else {
            let parent_idx = self.root;
            let data = self.alloc_leaf_child(elem, parent_idx.unwrap(), 0, None, None);
            let parent = self.in_nodes.get_mut(parent_idx.unwrap()).unwrap();
            let ans = data.arena;
            parent.children.push(data).unwrap();
            self.update_root_cache();
            return Cursor {
                leaf: ans.unwrap().into(),
                offset: 0,
            };
        };
        let leaf = self.leaf_nodes.get_mut(leaf_idx.0).unwrap();
        let parent_idx = leaf.parent();
        let mut is_full = false;
        let ans = if elem.can_merge(&leaf.elem) {
            leaf.elem.merge_left(&elem);
            Cursor {
                leaf: leaf_idx,
                offset: 0,
            }
        } else {
            let parent_idx = leaf.parent;
            let data = self.alloc_leaf_child(elem, parent_idx, 0, None, self.first_leaf());
            let ans = data.arena;
            is_full = {
                let parent = self.in_nodes.get_mut(parent_idx).unwrap();
                parent.children.insert(0, data).unwrap();
                parent.is_full()
            };
            self.update_children_parent_slot_from(ArenaIndex::Internal(parent_idx), 1);
            Cursor {
                leaf: ans.unwrap().into(),
                offset: 0,
            }
        };

        self.recursive_update_cache(leaf_idx.into(), B::USE_DIFF, None);
        if is_full {
            self.split(parent_idx);
        }

        ans
    }

    /// compare the position of a and b
    pub fn compare_pos(&self, a: Cursor, b: Cursor) -> Ordering {
        if a.leaf == b.leaf {
            return a.offset.cmp(&b.offset);
        }

        let leaf_a = self.leaf_nodes.get(a.leaf.0).unwrap();
        let leaf_b = self.leaf_nodes.get(b.leaf.0).unwrap();
        let mut node_a = self.get_internal(leaf_a.parent());
        if leaf_a.parent == leaf_b.parent {
            return leaf_a.parent_slot.cmp(&leaf_b.parent_slot);
        }

        let mut node_b = self.get_internal(leaf_b.parent());
        while node_a.parent != node_b.parent {
            node_a = self.get_internal(node_a.parent.unwrap());
            node_b = self.get_internal(node_b.parent.unwrap());
        }

        node_a.parent_slot.cmp(&node_b.parent_slot)
    }

    /// Iterate the caches of previous nodes/elements.
    /// This method will visit as less caches as possible.
    /// For example, if all nodes in a subtree need to be visited, we will only visit the root cache.
    ///
    /// f: (node_cache, previous_sibling_elem, (this_elem, offset))
    pub fn visit_previous_caches<F>(&self, cursor: Cursor, mut f: F)
    where
        F: FnMut(PreviousCache<'_, B>),
    {
        let node = self.leaf_nodes.get(cursor.leaf.0).unwrap();
        self.visit_previous_caches_before_child(
            ArenaIndex::Internal(node.parent),
            node.parent_slot(),
            &mut f,
        );
        f(PreviousCache::ThisElemAndOffset {
            elem: &node.elem,
            offset: cursor.offset,
        });
    }

    fn visit_previous_caches_before_child<F>(
        &self,
        parent_idx: ArenaIndex,
        child_slot: usize,
        f: &mut F,
    ) where
        F: FnMut(PreviousCache<'_, B>),
    {
        let mut ancestors = NodePath::new();
        let mut current = Some((parent_idx, child_slot as u8));
        while let Some((node_idx, slot)) = current {
            ancestors.push(Idx::new(node_idx, slot)).unwrap();
            let node = self.get_internal(node_idx);
            current = node.parent.map(|parent| (parent, node.parent_slot));
        }

        for ancestor in ancestors.iter().rev() {
            let node = self.get_internal(ancestor.arena);
            for child in node.children[..ancestor.arr as usize].iter() {
                f(PreviousCache::NodeCache(&child.cache));
            }
        }
    }

    pub fn diagnose_balance(&self) {
        let mut size_counter: FxHashMap<usize, usize> = Default::default();
        for (_, node) in self.in_nodes.iter() {
            *size_counter.entry(node.children.len()).or_default() += 1;
        }
        dbg!(size_counter);

        let mut size_counter: FxHashMap<usize, usize> = Default::default();
        for (_, node) in self.leaf_nodes.iter() {
            *size_counter.entry(node.elem.rle_len()).or_default() += 1;
        }
        dbg!(size_counter);
    }

    /// Iterate over the leaf elements in the tree if the filter returns true for all its ancestors' caches, including its own cache.
    pub fn iter_with_filter<'a, R: Default + Copy + AddAssign + 'a>(
        &'a self,
        mut f: impl FnMut(&B::Cache) -> (bool, R) + 'a,
    ) -> impl Iterator<Item = (R, &'a B::Elem)> + 'a {
        let mut queue = VecDeque::new();
        queue.push_back((self.root, R::default()));
        std::iter::from_fn(move || {
            while let Some((node_idx, mut r)) = queue.pop_front() {
                match node_idx {
                    ArenaIndex::Leaf(leaf) => {
                        let node = self.leaf_nodes.get(leaf).unwrap();
                        return Some((r, &node.elem));
                    }
                    ArenaIndex::Internal(idx) => {
                        let node = self.in_nodes.get(idx).unwrap();
                        for child in node.children.iter() {
                            let (drill, new_r) = f(&child.cache);
                            if drill {
                                queue.push_back((child.arena, r));
                            }
                            r += new_r;
                        }
                    }
                }
            }

            None
        })
    }

    /// This method allows users to update the caches and the elements with a filter.
    ///
    /// If `f` returns true for a node, it will drill down into the subtree whose root is the node.
    ///
    /// It's the caller's responsibility to ensure the invariance of caches being up to date.
    pub fn update_cache_and_elem_with_filter<'a>(
        &'a mut self,
        mut f: impl FnMut(&mut B::Cache) -> bool + 'a,
        mut g: impl FnMut(&mut B::Elem) + 'a,
    ) {
        let mut stack = vec![self.root];
        while let Some(node_idx) = stack.pop() {
            match node_idx {
                ArenaIndex::Leaf(leaf) => {
                    let node = self.leaf_nodes.get_mut(leaf).unwrap();
                    g(&mut node.elem);
                }
                ArenaIndex::Internal(idx) => {
                    let node = self.in_nodes.get_mut(idx).unwrap();
                    for child in node.children.iter_mut() {
                        if f(&mut child.cache) {
                            stack.push(child.arena);
                        }
                    }
                }
            }
        }
    }

    pub fn depth(&self) -> usize {
        let mut depth = 0;
        let mut index = self.root;
        let mut node = self.in_nodes.get(index.unwrap_internal()).unwrap();
        loop {
            depth += 1;
            index = node.children.first().unwrap().arena;
            match index {
                ArenaIndex::Leaf(_) => return depth,
                ArenaIndex::Internal(index) => {
                    node = self.in_nodes.get(index).unwrap();
                }
            }
        }
    }

    pub fn internal_avg_children_num(&self) -> f64 {
        let mut sum = 0;
        for (_, node) in self.in_nodes.iter() {
            sum += node.children.len();
        }
        sum as f64 / self.in_nodes.len() as f64
    }
}

fn merge_adj<E: Mergeable + Debug>(data: &mut Vec<E>) {
    // Merge adjacent elements
    let mut i = 0;
    let last = data.len() - 1;
    let mut to_delete_start = 0;
    let mut del_len = 0;
    while i < last {
        if data[i].can_merge(&data[i + 1]) {
            let (a, b) = arref::mut_twice(data.as_mut_slice(), i, i + 1).unwrap();
            a.merge_right(b);
            if del_len == 0 {
                to_delete_start = i + 1;
            }

            data.swap(i + 1, to_delete_start + del_len);
            del_len += 1;
            i += 1;
        }
        i += 1;
    }

    if del_len > 0 {
        data.drain(to_delete_start..to_delete_start + del_len);
    }
}

pub enum PreviousCache<'a, B: BTreeTrait> {
    NodeCache(&'a B::Cache),
    PrevSiblingElem(&'a B::Elem),
    ThisElemAndOffset { elem: &'a B::Elem, offset: usize },
}

#[inline(always)]
fn add_leaf_dirty_map<T>(leaf: ArenaIndex, dirty_map: &mut LeafDirtyMap<T>, leaf_diff: T) {
    dirty_map.insert(leaf, leaf_diff);
}

impl<B: BTreeTrait> BTree<B> {
    pub fn check(&self) {
        // check cache
        let mut leaf_level = None;
        for (index, node) in self.in_nodes.iter() {
            if index != self.root.unwrap() {
                assert!(!node.is_empty());
            }

            for (i, child_info) in node.children.iter().enumerate() {
                if matches!(child_info.arena, ArenaIndex::Internal(_)) {
                    assert!(!node.is_child_leaf());
                    let child = self.get_internal(child_info.arena);
                    let mut cache = Default::default();
                    child.calc_cache(&mut cache, None);
                    assert_eq!(child.parent_slot, i as u8);
                    assert_eq!(child.parent, Some(ArenaIndex::Internal(index)));
                    assert_eq!(
                        cache, child_info.cache,
                        "index={:?} child_index={:?}",
                        index, child_info.arena
                    );
                }
            }

            if let Some(parent) = node.parent {
                let parent = self.get_internal(parent);
                assert_eq!(
                    parent.children[node.parent_slot as usize].arena,
                    ArenaIndex::Internal(index)
                );
                self.get_path(ArenaIndex::Internal(index));
            } else {
                assert_eq!(index, self.root.unwrap_internal())
            }

            // if index != self.root.unwrap() {
            //     assert!(!node.is_lack(), "len={}\n", node.len());
            // }
            //
            // assert!(!node.is_full(), "len={}", node.len());
        }

        let root = self.get_internal(self.root);
        let mut root_cache = Default::default();
        root.calc_cache(&mut root_cache, None);
        assert_eq!(&self.root_cache, &root_cache);

        for (leaf_index, leaf_node) in self.leaf_nodes.iter() {
            let mut length = 1;
            let mut node_idx = leaf_node.parent;
            while node_idx != self.root.unwrap() {
                let node = self.get_internal(ArenaIndex::Internal(node_idx));
                length += 1;
                node_idx = node.parent.unwrap().unwrap();
            }
            match leaf_level {
                Some(expected) => {
                    if length != expected {
                        dbg!(leaf_index, leaf_node);
                        assert_eq!(length, expected);
                    }
                }
                None => {
                    leaf_level = Some(length);
                }
            }

            let cache = B::get_elem_cache(&leaf_node.elem);
            let parent = self.get_internal(leaf_node.parent());
            assert_eq!(parent.children[leaf_node.parent_slot()].cache, cache);
            assert_eq!(
                parent.children[leaf_node.parent_slot()].arena,
                ArenaIndex::Leaf(leaf_index)
            );
            self.get_path(ArenaIndex::Leaf(leaf_index));
        }

        let mut seen = 0;
        let mut prev_leaf = None;
        let mut current = self.first_leaf;
        while let Some(leaf) = current {
            assert!(seen < self.leaf_nodes.len(), "leaf link contains a cycle");
            let node = self.leaf_nodes.get(leaf.0).unwrap();
            assert_eq!(node.prev_leaf(), prev_leaf);
            prev_leaf = Some(leaf);
            current = node.next_leaf();
            seen += 1;
        }
        assert_eq!(prev_leaf, self.last_leaf);
        assert_eq!(seen, self.leaf_nodes.len());

        let mut seen = 0;
        let mut next_leaf = None;
        let mut current = self.last_leaf;
        while let Some(leaf) = current {
            assert!(seen < self.leaf_nodes.len(), "leaf link contains a cycle");
            let node = self.leaf_nodes.get(leaf.0).unwrap();
            assert_eq!(node.next_leaf(), next_leaf);
            next_leaf = Some(leaf);
            current = node.prev_leaf();
            seen += 1;
        }
        assert_eq!(next_leaf, self.first_leaf);
        assert_eq!(seen, self.leaf_nodes.len());
    }
}

impl<B: BTreeTrait, T: Into<B::Elem>> FromIterator<T> for BTree<B> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut tree = Self::new();
        let iter = iter.into_iter();
        let min_size = iter.size_hint().0;
        tree.leaf_nodes.reserve(min_size);
        let max_child_size = MAX_CHILDREN_NUM - 2;

        struct TempInternalNode<B: BTreeTrait> {
            children: HeaplessVec<Child<B>, MAX_CHILDREN_NUM>,
            cache: B::Cache,
            arena_index: RawArenaIndex,
        }

        let parent_num = (min_size + max_child_size - 1) / max_child_size;
        let mut internal_nodes: Vec<TempInternalNode<B>> = Vec::with_capacity(parent_num);
        let index = tree.in_nodes.insert(Default::default());
        internal_nodes.push(TempInternalNode {
            children: Default::default(),
            cache: Default::default(),
            arena_index: index,
        });

        // create all leaf nodes and their parents
        for elem in iter {
            let parent = match internal_nodes.last_mut() {
                Some(last) if last.children.len() < max_child_size => last,
                Some(last) => {
                    // calculate cache
                    B::calc_cache_internal(&mut last.cache, &last.children);
                    let index = tree.in_nodes.insert(Default::default());
                    internal_nodes.push(TempInternalNode {
                        children: Default::default(),
                        cache: Default::default(),
                        arena_index: index,
                    });
                    internal_nodes.last_mut().unwrap()
                }
                _ => unreachable!(),
            };

            let leaf = LeafNode {
                elem: elem.into(),
                parent: parent.arena_index,
                parent_slot: parent.children.len() as u8,
                prev_leaf: None,
                next_leaf: None,
            };

            let cache = B::get_elem_cache(&leaf.elem);
            let leaf_index = tree.leaf_nodes.insert(leaf);
            tree.link_leaf_between(leaf_index.into(), tree.last_leaf, None);
            parent
                .children
                .push(Child {
                    arena: ArenaIndex::Leaf(leaf_index),
                    cache,
                })
                .unwrap();
        }

        // recursively create the internal nodes in higher level, until we reach root
        while internal_nodes.len() > 1 {
            let parent_num = (internal_nodes.len() + max_child_size - 1) / max_child_size;
            let children = std::mem::replace(&mut internal_nodes, Vec::with_capacity(parent_num));
            let index = tree.in_nodes.insert(Default::default());
            internal_nodes.push(TempInternalNode {
                children: Default::default(),
                cache: Default::default(),
                arena_index: index,
            });

            let mut parent_slot = 0;
            // eprintln!(
            //     "children.len={} max_child_size={}",
            //     children.len(),
            //     max_child_size
            // );
            for mut child in children {
                let parent = match internal_nodes.last_mut() {
                    Some(last) if last.children.len() < max_child_size => last,
                    Some(last) => {
                        // calculate cache
                        B::calc_cache_internal(&mut last.cache, &last.children);
                        let index = tree.in_nodes.insert(Default::default());
                        internal_nodes.push(TempInternalNode {
                            children: Default::default(),
                            cache: Default::default(),
                            arena_index: index,
                        });
                        internal_nodes.last_mut().unwrap()
                    }
                    _ => unreachable!(),
                };

                B::calc_cache_internal(&mut child.cache, &child.children);
                let child_node = tree.in_nodes.get_mut(child.arena_index).unwrap();
                child_node.children = child.children;
                child_node.parent = Some(ArenaIndex::Internal(parent.arena_index));
                child_node.parent_slot = parent_slot;
                parent_slot = (parent_slot + 1) % (max_child_size as u8);
                parent
                    .children
                    .push(Child {
                        arena: ArenaIndex::Internal(child.arena_index),
                        cache: child.cache,
                    })
                    .unwrap();
            }

            debug_assert_eq!(parent_num, internal_nodes.len());
        }

        debug_assert_eq!(internal_nodes.len(), 1);
        let node = internal_nodes.remove(0);
        B::calc_cache_internal(&mut tree.root_cache, &node.children);
        tree.in_nodes.remove(tree.root.unwrap());
        tree.root = ArenaIndex::Internal(node.arena_index);
        let root = tree.root.unwrap();
        tree.in_nodes.get_mut(root).unwrap().children = node.children;
        tree
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct MergeElem {
        value: u8,
        len: usize,
    }

    impl HasLength for MergeElem {
        fn rle_len(&self) -> usize {
            self.len
        }
    }

    impl Sliceable for MergeElem {
        fn _slice(&self, range: Range<usize>) -> Self {
            Self {
                value: self.value,
                len: range.len(),
            }
        }

        fn slice_(&mut self, range: impl RangeBounds<usize>)
        where
            Self: Sized,
        {
            let start = match range.start_bound() {
                std::ops::Bound::Included(x) => *x,
                std::ops::Bound::Excluded(x) => x + 1,
                std::ops::Bound::Unbounded => 0,
            };
            let end = match range.end_bound() {
                std::ops::Bound::Included(x) => x + 1,
                std::ops::Bound::Excluded(x) => *x,
                std::ops::Bound::Unbounded => self.len,
            };
            self.len = end - start;
        }
    }

    impl Mergeable for MergeElem {
        fn can_merge(&self, rhs: &Self) -> bool {
            self.value == rhs.value
        }

        fn merge_right(&mut self, rhs: &Self) {
            self.len += rhs.len;
        }

        fn merge_left(&mut self, left: &Self) {
            self.len += left.len;
        }
    }

    impl TryInsert for MergeElem {
        fn try_insert(&mut self, _: usize, elem: Self) -> Result<(), Self>
        where
            Self: Sized,
        {
            Err(elem)
        }
    }

    impl CanRemove for MergeElem {
        fn can_remove(&self) -> bool {
            self.len == 0
        }
    }

    struct MergeElemTreeTrait;

    impl BTreeTrait for MergeElemTreeTrait {
        type Elem = MergeElem;
        type Cache = usize;
        type CacheDiff = isize;

        fn calc_cache_internal(cache: &mut Self::Cache, caches: &[Child<Self>]) -> Self::CacheDiff {
            let new_cache = caches.iter().map(|child| child.cache).sum::<usize>();
            let diff = new_cache as isize - *cache as isize;
            *cache = new_cache;
            diff
        }

        fn apply_cache_diff(cache: &mut Self::Cache, diff: &Self::CacheDiff) {
            *cache = (*cache as isize + diff) as usize;
        }

        fn merge_cache_diff(diff1: &mut Self::CacheDiff, diff2: &Self::CacheDiff) {
            *diff1 += *diff2;
        }

        fn get_elem_cache(elem: &Self::Elem) -> Self::Cache {
            elem.len
        }

        fn new_cache_to_diff(cache: &Self::Cache) -> Self::CacheDiff {
            *cache as isize
        }

        fn sub_cache(cache_lhs: &Self::Cache, cache_rhs: &Self::Cache) -> Self::CacheDiff {
            *cache_lhs as isize - *cache_rhs as isize
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct CountElem {
        len: usize,
    }

    impl HasLength for CountElem {
        fn rle_len(&self) -> usize {
            self.len
        }
    }

    impl Sliceable for CountElem {
        fn _slice(&self, range: Range<usize>) -> Self {
            Self { len: range.len() }
        }

        fn slice_(&mut self, range: impl RangeBounds<usize>)
        where
            Self: Sized,
        {
            let start = match range.start_bound() {
                std::ops::Bound::Included(x) => *x,
                std::ops::Bound::Excluded(x) => x + 1,
                std::ops::Bound::Unbounded => 0,
            };
            let end = match range.end_bound() {
                std::ops::Bound::Included(x) => x + 1,
                std::ops::Bound::Excluded(x) => *x,
                std::ops::Bound::Unbounded => self.len,
            };
            self.len = end - start;
        }
    }

    impl Mergeable for CountElem {
        fn can_merge(&self, _: &Self) -> bool {
            false
        }

        fn merge_right(&mut self, _: &Self) {
            unreachable!()
        }

        fn merge_left(&mut self, _: &Self) {
            unreachable!()
        }
    }

    impl TryInsert for CountElem {
        fn try_insert(&mut self, _: usize, elem: Self) -> Result<(), Self>
        where
            Self: Sized,
        {
            Err(elem)
        }
    }

    impl CanRemove for CountElem {
        fn can_remove(&self) -> bool {
            false
        }
    }

    struct ChildCountTreeTrait;

    impl UseLengthFinder<MergeElemTreeTrait> for MergeElemTreeTrait {
        fn get_len(cache: &usize) -> usize {
            *cache
        }
    }

    impl BTreeTrait for ChildCountTreeTrait {
        type Elem = CountElem;
        type Cache = usize;
        type CacheDiff = isize;

        fn calc_cache_internal(cache: &mut Self::Cache, caches: &[Child<Self>]) -> Self::CacheDiff {
            let new_cache = caches.len();
            let diff = new_cache as isize - *cache as isize;
            *cache = new_cache;
            diff
        }

        fn apply_cache_diff(cache: &mut Self::Cache, diff: &Self::CacheDiff) {
            *cache = (*cache as isize + diff) as usize;
        }

        fn merge_cache_diff(diff1: &mut Self::CacheDiff, diff2: &Self::CacheDiff) {
            *diff1 += *diff2;
        }

        fn get_elem_cache(_: &Self::Elem) -> Self::Cache {
            0
        }

        fn new_cache_to_diff(cache: &Self::Cache) -> Self::CacheDiff {
            *cache as isize
        }

        fn sub_cache(cache_lhs: &Self::Cache, cache_rhs: &Self::Cache) -> Self::CacheDiff {
            *cache_lhs as isize - *cache_rhs as isize
        }
    }

    #[test]
    fn update_leaf_can_merge_into_next_leaf_across_parents() {
        let mut tree: BTree<MergeElemTreeTrait> = (0..11)
            .map(|index| MergeElem {
                value: if index == 10 { 7 } else { index as u8 },
                len: 2,
            })
            .collect();

        let mut leaf = tree.first_leaf().unwrap();
        for _ in 0..9 {
            leaf = tree.next_leaf_index(leaf).unwrap();
        }
        let next_leaf = tree.next_leaf_index(leaf).unwrap();
        assert_ne!(
            tree.get_leaf(leaf.into()).parent(),
            tree.get_leaf(next_leaf.into()).parent()
        );

        let (valid, splitted) = tree.update_leaf(leaf, |elem| {
            elem.len = 1;
            (true, Some(MergeElem { value: 7, len: 1 }), None)
        });
        assert!(valid);
        assert_eq!(splitted.arr.len(), 1);
        assert_eq!(tree.leaf_nodes.len(), 11);
        assert_eq!(tree.get_elem(leaf).unwrap().len, 1);
        assert_eq!(tree.get_elem(next_leaf).unwrap().len, 3);
        tree.check();
    }

    #[test]
    fn structural_insert_recomputes_root_cache() {
        let mut tree = BTree::<ChildCountTreeTrait>::new();
        let cursor = tree.push(CountElem { len: 4 });
        assert_eq!(*tree.root_cache(), 1);

        tree.insert_by_path(
            Cursor {
                leaf: cursor.leaf,
                offset: 2,
            },
            CountElem { len: 1 },
        );

        assert_eq!(*tree.root_cache(), 3);
        tree.check();
    }

    #[test]
    fn prepend_and_insert_many_update_root_cache_on_empty_tree() {
        let mut tree = BTree::<ChildCountTreeTrait>::new();
        tree.prepend(CountElem { len: 1 });
        assert_eq!(*tree.root_cache(), 1);
        tree.check();

        let mut tree = BTree::<ChildCountTreeTrait>::new();
        tree.insert_many_by_cursor(
            None,
            vec![
                CountElem { len: 1 },
                CountElem { len: 1 },
                CountElem { len: 1 },
                CountElem { len: 1 },
            ]
            .into_iter(),
        );
        assert_eq!(*tree.root_cache(), 4);
        tree.check();
    }

    fn previous_len(tree: &BTree<MergeElemTreeTrait>, cursor: Cursor) -> usize {
        let mut len = 0;
        tree.visit_previous_caches(cursor, |cache| match cache {
            PreviousCache::NodeCache(cache) => len += *cache,
            PreviousCache::PrevSiblingElem(elem) => len += elem.len,
            PreviousCache::ThisElemAndOffset { offset, .. } => len += offset,
        });
        len
    }

    #[test]
    fn visit_previous_caches_matches_query_position() {
        let tree: BTree<MergeElemTreeTrait> = [
            MergeElem { value: 0, len: 2 },
            MergeElem { value: 1, len: 3 },
            MergeElem { value: 2, len: 1 },
            MergeElem { value: 3, len: 4 },
            MergeElem { value: 4, len: 2 },
            MergeElem { value: 5, len: 5 },
            MergeElem { value: 6, len: 1 },
            MergeElem { value: 7, len: 3 },
            MergeElem { value: 8, len: 2 },
            MergeElem { value: 9, len: 4 },
            MergeElem { value: 10, len: 2 },
        ]
        .into_iter()
        .collect();
        let total = *tree.root_cache();
        for index in 0..=total {
            let query = tree.query::<LengthFinder>(&index).unwrap();
            assert_eq!(previous_len(&tree, query.cursor()), index);
        }
    }

    #[test]
    fn partial_drain_keeps_tree_consistent_after_drop() {
        let mut tree: BTree<MergeElemTreeTrait> =
            (0..15).map(|value| MergeElem { value, len: 1 }).collect();
        let drain_range = tree.range::<LengthFinder>(3..12).unwrap();
        let mut drain = tree.drain(drain_range);
        assert_eq!(drain.next().unwrap().value, 3);
        drop(drain);

        assert_eq!(*tree.root_cache(), 6);
        assert_eq!(
            tree.iter().map(|elem| elem.value).collect::<Vec<_>>(),
            vec![0, 1, 2, 12, 13, 14]
        );
        tree.check();
    }
}

struct SplitInfo {
    new_pos: Option<Cursor>,
    left_neighbour: Option<LeafIndex>,
    parent_idx: RawArenaIndex,
    insert_slot: usize,
    new_leaf: Option<ArenaIndex>,
}

impl<B: BTreeTrait> Default for BTree<B> {
    fn default() -> Self {
        Self::new()
    }
}

fn delete_range<T: Clone, const N: usize>(
    arr: &mut heapless::Vec<T, N>,
    range: impl RangeBounds<usize>,
) {
    let start = match range.start_bound() {
        std::ops::Bound::Included(x) => *x,
        std::ops::Bound::Excluded(x) => x + 1,
        std::ops::Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
        std::ops::Bound::Included(x) => x + 1,
        std::ops::Bound::Excluded(x) => *x,
        std::ops::Bound::Unbounded => arr.len(),
    };

    if start == end {
        return;
    }

    if end - start == 1 {
        arr.remove(start);
        return;
    }

    let mut ans = heapless::Vec::from_slice(&arr[..start]).unwrap();
    ans.extend_from_slice(&arr[end..]).unwrap();
    *arr = ans;
}
