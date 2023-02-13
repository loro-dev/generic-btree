use std::{
    fmt::Debug,
    ops::{Deref, Range},
};

use thunderdome::{Arena, Index as ArenaIndex};
mod generic_impl;
mod iter;
pub use generic_impl::{OrdTreeMap, OrdTreeSet};
pub mod rle;

pub trait BTreeTrait {
    type Elem: Debug;
    type Cache: Debug + Default + Clone + Eq;
    const MAX_LEN: usize;

    fn element_to_cache(element: &Self::Elem) -> Self::Cache;
    #[allow(unused)]
    fn insert(elements: &mut Vec<Self::Elem>, index: usize, offset: usize, elem: Self::Elem) {
        elements.insert(index, elem);
    }

    fn calc_cache_internal(caches: &[Child<Self::Cache>]) -> Self::Cache;
    fn calc_cache_leaf(elements: &[Self::Elem]) -> Self::Cache;
}

pub trait Query<B: BTreeTrait> {
    type QueryArg: Debug + Clone;

    fn init(target: &Self::QueryArg) -> Self;

    fn find_node(
        &mut self,
        target: &Self::QueryArg,
        child_caches: &[Child<B::Cache>],
    ) -> FindResult;

    fn find_element(&mut self, target: &Self::QueryArg, elements: &[B::Elem]) -> FindResult;

    #[allow(unused)]
    fn delete(
        elements: &mut Vec<B::Elem>,
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
    fn delete_range<'x, 'b>(
        elements: &'x mut Vec<B::Elem>,
        start_query: &'b Self::QueryArg,
        end_query: &'b Self::QueryArg,
        start: Option<QueryResult>,
        end: Option<QueryResult>,
    ) -> Box<dyn Iterator<Item = B::Elem> + 'x> {
        Box::new(match (start, end) {
            (None, None) => elements.drain(..),
            (None, Some(to)) => elements.drain(..to.elem_index),
            (Some(from), None) => elements.drain(from.elem_index..),
            (Some(from), Some(to)) => elements.drain(from.elem_index..to.elem_index),
        })
    }
}

#[derive(Debug)]
pub struct BTree<B: BTreeTrait> {
    nodes: Arena<Node<B>>,
    root: ArenaIndex,
    root_cache: B::Cache,
}

impl<Elem: Clone, B: BTreeTrait<Elem = Elem>> Clone for BTree<B> {
    fn clone(&self) -> Self {
        Self {
            nodes: self.nodes.clone(),
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
    pub arr: usize,
}

impl Idx {
    #[inline(always)]
    pub fn new(arena: ArenaIndex, arr: usize) -> Self {
        Self { arena, arr }
    }
}

type Path = Vec<Idx>;

struct PathRef<'a>(&'a [Idx]);

impl<'a> Deref for PathRef<'a> {
    type Target = [Idx];

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl<'a> From<&'a Vec<Idx>> for PathRef<'a> {
    fn from(value: &'a Vec<Idx>) -> Self {
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

#[derive(Debug, Clone)]
pub struct QueryResult {
    node_path: Path,
    pub elem_index: usize,
    pub offset: usize,
    pub found: bool,
}

impl QueryResult {
    fn path_ref(&self) -> PathRef {
        self.node_path.as_slice().into()
    }
}

// TODO: use enum to save spaces
struct Node<B: BTreeTrait> {
    elements: Vec<B::Elem>,
    children: Vec<Child<B::Cache>>,
}

impl<B: BTreeTrait> Debug for Node<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Node")
            .field("elements", &self.elements)
            .field("children", &self.children)
            .finish()
    }
}

impl<Elem: Clone, B: BTreeTrait<Elem = Elem>> Clone for Node<B> {
    fn clone(&self) -> Self {
        Self {
            elements: self.elements.clone(),
            children: self.children.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Child<Cache> {
    arena: ArenaIndex,
    cache: Cache,
}

impl<Cache> Child<Cache> {
    #[inline(always)]
    pub fn cache(&self) -> &Cache {
        &self.cache
    }

    fn new(arena: ArenaIndex, cache: Cache) -> Self {
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
            elements: Vec::new(),
            children: Vec::new(),
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

    pub fn len(&self) -> usize {
        if self.is_internal() {
            self.children.len()
        } else {
            self.elements.len()
        }
    }

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

    #[must_use]
    fn calc_cache(&self) -> B::Cache {
        if self.is_internal() {
            B::calc_cache_internal(&self.children)
        } else {
            B::calc_cache_leaf(&self.elements)
        }
    }
}

impl<B: BTreeTrait> BTree<B> {
    pub fn new() -> Self {
        let mut arena = Arena::new();
        let root = arena.insert(Node::new());
        Self {
            nodes: arena,
            root,
            root_cache: B::Cache::default(),
        }
    }

    pub fn insert<Q>(&mut self, tree_index: &Q::QueryArg, data: B::Elem)
    where
        Q: Query<B>,
    {
        let result = self.query::<Q>(tree_index);
        self.insert_by_query_result(result, data)
    }

    pub fn insert_by_query_result(&mut self, result: QueryResult, data: B::Elem) {
        let index = *result.node_path.last().unwrap();
        let node = self.nodes.get_mut(index.arena).unwrap();
        if result.found {
            B::insert(&mut node.elements, result.elem_index, result.offset, data);
        } else {
            node.elements.insert(result.elem_index, data);
        }

        let is_full = node.is_full();
        self.recursive_update_cache(result.path_ref());
        if is_full {
            self.split(result.path_ref());
        }
    }

    pub fn delete<Q>(&mut self, query: &Q::QueryArg) -> bool
    where
        Q: Query<B>,
    {
        let result = self.query::<Q>(query);
        if !result.found {
            return false;
        }

        let index = *result.node_path.last().unwrap();
        let node = self.nodes.get_mut(index.arena).unwrap();
        if result.found {
            Q::delete(&mut node.elements, query, result.elem_index, result.offset);
        }

        let is_full = node.is_full();
        let is_lack = node.is_lack();
        self.recursive_update_cache(result.path_ref());
        if is_full {
            self.split(result.path_ref());
        } else if is_lack {
            let mut path_ref = result.path_ref();
            while self.handle_lack(&path_ref) {
                path_ref.set_as_parent_path();
            }
        }
        true
    }

    pub fn query<Q>(&self, query: &Q::QueryArg) -> QueryResult
    where
        Q: Query<B>,
    {
        let mut finder = Q::init(query);
        let mut node = self.nodes.get(self.root).unwrap();
        let mut index = self.root;
        let mut ans = QueryResult {
            node_path: vec![Idx::new(index, 0)],
            elem_index: 0,
            offset: 0,
            found: false,
        };
        while node.is_internal() {
            let result = finder.find_node(query, &node.children);
            let i = result.index;
            let i = i.min(node.children.len() - 1);
            index = node.children[i].arena;
            node = self.nodes.get(index).unwrap();
            ans.node_path.push(Idx::new(index, i));
        }

        let result = finder.find_element(query, &node.elements);
        ans.elem_index = result.index;
        ans.found = result.found;
        ans.offset = result.offset;
        ans
    }

    pub fn drain<Q>(&mut self, range: Range<Q::QueryArg>) -> iter::Drain<B, Q>
    where
        Q: Query<B>,
    {
        let from = self.query::<Q>(&range.start);
        let to = self.query::<Q>(&range.end);
        iter::Drain::new(self, range.start, range.end, from, to)
    }

    pub fn iter_mut<Q>(
        &mut self,
        range: Range<Q::QueryArg>,
    ) -> impl Iterator<Item = &mut B::Elem> + '_
    where
        Q: Query<B>,
    {
        let start = self.query::<Q>(&range.start);
        let end = self.query::<Q>(&range.end);
        let mut node_iter = iter::IterMut::new(self, start.clone(), end.clone());
        let mut elem_iter: Option<std::slice::IterMut<'_, B::Elem>> = None;
        std::iter::from_fn(move || loop {
            if let Some(inner_elem_iter) = &mut elem_iter {
                match inner_elem_iter.next() {
                    Some(elem) => return Some(elem),
                    None => elem_iter = None,
                }
            } else {
                match node_iter.next() {
                    Some((idx, node)) => {
                        let start = if idx.arena == start.node_path.last().unwrap().arena {
                            start.elem_index
                        } else {
                            0
                        };
                        let end = if idx.arena == end.node_path.last().unwrap().arena {
                            end.elem_index
                        } else {
                            node.elements.len()
                        };
                        elem_iter = Some(node.elements[start..end].iter_mut());
                    }
                    None => return None,
                }
            }
        })
    }

    pub fn iter(&self) -> impl Iterator<Item = &B::Elem> + '_ {
        let mut path = self.first_path().unwrap_or(Vec::new());
        let idx = path.last().copied().unwrap_or(Idx::new(self.root, 0));
        let node = self.get(idx.arena);
        let mut iter = node.elements.iter();
        std::iter::from_fn(move || loop {
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

    fn first_path(&self) -> Option<Path> {
        let mut path = Path::new();
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

    pub fn iter_range<Q>(&self, range: Range<Q::QueryArg>) -> impl Iterator<Item = &B::Elem> + '_
    where
        Q: Query<B>,
    {
        let start = self.query::<Q>(&range.start);
        let end = self.query::<Q>(&range.end);
        let mut node_iter = iter::Iter::new(self, start.clone(), end.clone());
        let mut elem_iter: Option<std::slice::Iter<'_, B::Elem>> = None;
        std::iter::from_fn(move || loop {
            if let Some(inner_elem_iter) = &mut elem_iter {
                match inner_elem_iter.next() {
                    Some(elem) => return Some(elem),
                    None => elem_iter = None,
                }
            } else {
                match node_iter.next() {
                    Some((idx, node)) => {
                        let start = if idx.arena == start.node_path.last().unwrap().arena {
                            start.elem_index
                        } else {
                            0
                        };
                        let end = if idx.arena == end.node_path.last().unwrap().arena {
                            end.elem_index
                        } else {
                            node.elements.len()
                        };
                        elem_iter = Some(node.elements[start..end].iter());
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
        let mut right: Node<B> = Node::new();
        // split
        if node.is_internal() {
            let split = node.children.len() / 2;
            right.children.extend(node.children.drain(split..));
        } else {
            let split = node.elements.len() / 2;
            right.elements.extend(node.elements.drain(split..));
        }

        // update cache
        let right_cache = right.calc_cache();
        let right = self.nodes.insert(right);
        let this_cache = self.calc_cache(path.this().arena);

        self.inner_insert_node(
            path.parent_path(),
            path.this().arr,
            this_cache,
            Child {
                arena: right,
                cache: right_cache,
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
        node: Child<B::Cache>,
    ) {
        if parent_path.is_empty() {
            self.split_root(new_cache, node);
        } else {
            let parent_index = *parent_path.last().unwrap();
            let parent = self.get_mut(parent_index.arena);
            parent.children[index].cache = new_cache;
            parent.children.insert(index + 1, node);
            let parent = self.get_mut(parent_index.arena);
            if parent.is_full() {
                self.split(parent_path);
            }
        }
    }

    fn split_root(&mut self, new_cache: B::Cache, new_node: Child<B::Cache>) {
        let root = self.get_mut(self.root);
        let left: Node<B> = std::mem::take(root);
        let right = new_node;
        let left = Child::new(self.nodes.insert(left), new_cache);
        let root = self.get_mut(self.root);
        root.children.push(left);
        root.children.push(right);
        self.root_cache = root.calc_cache();
    }

    fn calc_cache(&mut self, node: ArenaIndex) -> B::Cache {
        let node = self.get_mut(node);
        node.calc_cache()
    }

    #[inline(always)]
    fn get_mut(&mut self, index: ArenaIndex) -> &mut Node<B> {
        self.nodes.get_mut(index).unwrap()
    }

    #[inline(always)]
    fn get(&self, index: ArenaIndex) -> &Node<B> {
        self.nodes.get(index).unwrap()
    }

    /// merge into or borrow from neighbor
    ///
    /// cache should be up-to-date when calling this
    ///
    /// return is parent lack
    fn handle_lack(&mut self, path: &PathRef) -> bool {
        if path.is_root() {
            // ignore the lack of nodes problem for root
            return false;
        }

        match self.pair_neighbor(path.parent().unwrap().arena, path.this()) {
            Some((a_idx, b_idx)) => {
                let (a, b) = self.nodes.get2_mut(a_idx.arena, b_idx.arena);
                let a = a.unwrap();
                let b = b.unwrap();
                if a.len() + b.len() >= B::MAX_LEN {
                    // move
                    if a.len() < b.len() {
                        // move b to a
                        let move_len = (b.len() - a.len()) / 2;
                        if b.is_internal() {
                            a.children.extend(b.children.drain(..move_len));
                        } else {
                            a.elements.extend(b.elements.drain(..move_len));
                        }
                    } else {
                        // move a to b
                        let move_len = (a.len() - b.len()) / 2;
                        if a.is_internal() {
                            b.children
                                .splice(0..0, a.children.drain(a.children.len() - move_len..));
                        } else {
                            b.elements
                                .splice(0..0, a.elements.drain(a.elements.len() - move_len..));
                        }
                    }
                    let a_cache = a.calc_cache();
                    let b_cache = b.calc_cache();
                    let parent = self.get_mut(path.parent().unwrap().arena);
                    parent.children[a_idx.arr].cache = a_cache;
                    parent.children[b_idx.arr].cache = b_cache;
                    parent.is_lack()
                } else {
                    // merge b to a
                    if a.is_internal() {
                        a.children.append(&mut b.children);
                    } else {
                        a.elements.append(&mut b.elements);
                    }
                    let a_cache = a.calc_cache();
                    let parent = self.get_mut(path.parent().unwrap().arena);
                    parent.children.remove(b_idx.arr);
                    parent.children[a_idx.arr].cache = a_cache;
                    let is_lack = parent.is_lack();
                    self.purge(b_idx.arena);
                    is_lack
                }
            }
            None => true,
        }
    }

    fn pair_neighbor(&self, parent: ArenaIndex, index: Idx) -> Option<(Idx, Idx)> {
        let parent = self.get(parent);
        if index.arr == 0 {
            parent
                .children
                .get(1)
                .map(|x| (index, Idx::new(x.arena, 1)))
        } else {
            parent
                .children
                .get(index.arr - 1)
                .map(|x| (Idx::new(x.arena, index.arr - 1), index))
        }
    }

    fn recursive_update_cache(&mut self, path: PathRef) {
        let leaf = self.get(path.this().arena);
        assert!(leaf.is_leaf());
        let mut child_cache = leaf.calc_cache();
        let mut child_index = path.this().arr;
        for idx in path.parent_path().iter().rev() {
            let node = self.get_mut(idx.arena);
            node.children[child_index].cache = child_cache;
            child_cache = node.calc_cache();
            child_index = idx.arr;
        }
        self.root_cache = child_cache;
    }

    fn purge(&mut self, index: ArenaIndex) {
        let mut stack = vec![index];
        while let Some(x) = stack.pop() {
            let node = self.get(x);
            for x in node.children.iter() {
                stack.push(x.arena);
            }
            self.nodes.remove(x);
        }
    }

    /// find the next sibling at the same level
    ///
    /// return false if there is no next sibling
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

    fn get_path_from_indexes(&self, indexes: &[usize]) -> Path {
        debug_assert_eq!(indexes[0], 0);
        let mut path = vec![Idx::new(self.root, 0)];
        let mut node_idx = self.root;
        for &index in indexes[1..].iter() {
            let node = self.get(node_idx);
            path.push(Idx::new(node.children[index].arena, index));
            node_idx = node.children[index].arena;
        }
        path
    }
}

impl<B: BTreeTrait> BTree<B> {
    #[allow(unused)]
    fn check(&self) {
        // check cache
        for (index, node) in self.nodes.iter() {
            if node.is_internal() {
                for child_info in node.children.iter() {
                    let child = self.get(child_info.arena);
                    assert_eq!(child.calc_cache(), child_info.cache);
                }
            }
        }

        // TODO: check leaf at same level
        // TODO: check custom invariants
        // TODO: check children num bound
    }
}

impl<B: BTreeTrait> Default for BTree<B> {
    fn default() -> Self {
        Self::new()
    }
}
