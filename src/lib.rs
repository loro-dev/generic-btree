use std::{
    fmt::Debug,
    ops::{Deref, Range},
};

use thunderdome::{Arena, Index as ArenaIndex};
mod iter;

pub trait BTreeTrait {
    type Elem: Debug;
    type Cache: Debug + Default;
    const MAX_LEN: usize;

    fn element_to_cache(element: &Self::Elem) -> Self::Cache;
    #[allow(unused)]
    fn insert(elements: &mut Vec<Self::Elem>, index: usize, offset: usize, elem: Self::Elem) {
        elements.insert(index, elem);
    }
}

pub trait Query: Default {
    type Cache;
    type Elem;
    type QueryArg: Debug + Clone;

    fn find_node<'a, 'b, Iter>(&mut self, target: &'b Self::QueryArg, iter: Iter) -> FindResult
    where
        Iter: Iterator<Item = &'a Self::Cache>,
        Self::Cache: 'a;

    fn find_element<'a, 'b, Iter>(&mut self, target: &'b Self::QueryArg, iter: Iter) -> FindResult
    where
        Iter: Iterator<Item = &'a Self::Elem>,
        Self::Elem: 'a;

    #[allow(unused)]
    fn delete(
        elements: &mut Vec<Self::Elem>,
        query: &Self::QueryArg,
        elem_index: usize,
        offset: usize,
        found: bool,
    ) {
        if found {
            elements.remove(elem_index);
        }
    }

    #[allow(unused)]
    fn delete_range<'x, 'b>(
        elements: &'x mut Vec<Self::Elem>,
        query: &'b Self::QueryArg,
        from: Option<QueryResult>,
        to: Option<QueryResult>,
    ) -> Box<dyn Iterator<Item = Self::Elem> + 'x> {
        Box::new(match (from, to) {
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
}

pub struct FindResult {
    pub index: usize,
    pub offset: usize,
    pub found: bool,
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

    pub fn parent_path<'b: 'a>(&'b self) -> PathRef<'b> {
        debug_assert!(self.len() > 1);
        Self(&self[..self.len() - 1])
    }

    pub fn is_root(&self) -> bool {
        self.len() == 1
    }

    pub fn is_parent_root(&self) -> bool {
        self.len() == 2
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

#[derive(Debug)]
struct Node<B: BTreeTrait> {
    elements: Vec<B::Elem>,
    children: Vec<ArenaIndex>,
    cache: B::Cache,
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
            cache: B::Cache::default(),
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

    fn update_cache(&mut self) {
        todo!()
    }
}

impl<B: BTreeTrait> BTree<B> {
    pub fn new() -> Self {
        let mut arena = Arena::new();
        let root = arena.insert(Node::new());
        Self { nodes: arena, root }
    }

    pub fn insert<Q>(&mut self, tree_index: &Q::QueryArg, data: B::Elem)
    where
        Q: Query<Cache = B::Cache, Elem = B::Elem>,
    {
        let result = self.query::<Q>(tree_index);
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

    pub fn delete<Q>(&mut self, query: &Q::QueryArg)
    where
        Q: Query<Cache = B::Cache, Elem = B::Elem>,
    {
        let result = self.query::<Q>(query);
        if !result.found {
            return;
        }

        let index = *result.node_path.last().unwrap();
        let node = self.nodes.get_mut(index.arena).unwrap();
        Q::delete(
            &mut node.elements,
            query,
            result.elem_index,
            result.offset,
            result.found,
        );

        let is_full = node.is_full();
        let is_lack = node.is_lack();
        self.recursive_update_cache(result.path_ref());
        if is_full {
            self.split(result.path_ref());
        } else if is_lack {
            self.handle_lack(result.path_ref());
        }
    }

    pub fn query<Q>(&self, query: &Q::QueryArg) -> QueryResult
    where
        Q: Query<Cache = B::Cache, Elem = B::Elem>,
    {
        let mut finder = Q::default();
        let mut node = self.nodes.get(self.root).unwrap();
        let mut index = self.root;
        let mut ans = QueryResult {
            node_path: vec![Idx::new(index, 0)],
            elem_index: 0,
            offset: 0,
            found: false,
        };
        while node.is_internal() {
            let result = finder.find_node(
                query,
                node.children
                    .iter()
                    .map(|n| &self.nodes.get(*n).unwrap().cache),
            );
            let i = result.index;
            let i = i.min(node.children.len() - 1);
            index = node.children[i];
            node = self.nodes.get(index).unwrap();
            ans.node_path.push(Idx::new(index, i));
        }

        let result = finder.find_element(query, node.elements.iter());
        ans.elem_index = result.index;
        ans.found = result.found;
        ans.offset = result.offset;
        ans
    }

    pub fn drain<Q>(&mut self, range: Range<Q::QueryArg>) -> iter::Drain<B>
    where
        Q: Query<Cache = B::Cache, Elem = B::Elem>,
    {
        let from = self.query::<Q>(&range.start);
        let to = self.query::<Q>(&range.end);
        iter::Drain::new(self, from, to)
    }

    pub fn iter_mut<Q>(
        &mut self,
        range: Range<Q::QueryArg>,
    ) -> impl Iterator<Item = &mut B::Elem> + '_
    where
        Q: Query<Cache = B::Cache, Elem = B::Elem>,
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
        right.update_cache();
        let right = self.nodes.insert(right);
        self.update_cache(path.this().arena);

        self.inner_insert_node(path.parent_path(), path.last().unwrap().arr, right);
        // don't need to recursive update cache
    }

    // call site should ensure the cache is up-to-date after this method
    fn inner_insert_node(&mut self, parent_path: PathRef, index: usize, node: ArenaIndex) {
        if parent_path.is_empty() {
            self.split_root(node);
        } else {
            let parent_index = *parent_path.last().unwrap();
            let parent = self.get_mut(parent_index.arena);
            parent.children.insert(index, node);
            if parent.is_full() {
                self.split(parent_path)
            }
        }
    }

    fn split_root(&mut self, node: ArenaIndex) {
        let root = self.get_mut(self.root);
        let left: Node<B> = std::mem::take(root);
        let right = node;
        let left = self.nodes.insert(left);
        let root = self.get_mut(self.root);
        root.children.push(left);
        root.children.push(right);
        root.update_cache();
    }

    fn update_cache(&mut self, node: ArenaIndex) {
        let node = self.get_mut(node);
        node.update_cache();
    }

    #[inline(always)]
    fn get_mut(&mut self, index: ArenaIndex) -> &mut Node<B> {
        self.nodes.get_mut(index).unwrap()
    }

    #[inline(always)]
    fn get(&self, index: ArenaIndex) -> &Node<B> {
        self.nodes.get(index).unwrap()
    }

    // cache should be up-to-date when calling this
    fn handle_lack(&mut self, path: PathRef) {
        if path.is_root() {
            // ignore the lack of nodes problem for root
            return;
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
                    a.update_cache();
                    b.update_cache();
                } else {
                    // merge b to a
                    if a.is_internal() {
                        a.children.append(&mut b.children);
                    } else {
                        a.elements.append(&mut b.elements);
                    }
                    a.update_cache();
                    let parent = self.get_mut(path.parent().unwrap().arena);
                    parent.children.remove(b_idx.arr);
                    let is_lack = parent.is_lack();
                    self.purge(b_idx.arena);
                    if is_lack {
                        return self.handle_lack(path.parent_path());
                    }
                }
            }
            None => {
                return self.handle_lack(path.parent_path());
            }
        }
    }

    fn pair_neighbor(&self, parent: ArenaIndex, index: Idx) -> Option<(Idx, Idx)> {
        let parent = self.get(parent);
        if index.arr == 0 {
            parent.children.get(1).map(|x| (index, Idx::new(*x, 1)))
        } else {
            parent
                .children
                .get(index.arr - 1)
                .map(|x| (Idx::new(*x, index.arr - 1), index))
        }
    }

    fn recursive_update_cache(&mut self, path: PathRef) {
        for idx in path.iter().rev() {
            self.get_mut(idx.arena).update_cache();
        }
    }

    fn purge(&mut self, index: ArenaIndex) {
        let mut stack = vec![index];
        while let Some(x) = stack.pop() {
            let node = self.get(x);
            for x in node.children.iter() {
                stack.push(*x);
            }
            self.nodes.remove(x);
        }
    }

    /// find the next sibling at the same level
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
                path[depth - 1] = Idx::new(*next, this_idx.arr + 1);
            }
            None => {
                if !self.next_sibling(&mut path[..depth - 1]) {
                    return false;
                }

                let parent = self.get(path[depth - 2].arena);
                path[depth - 1] = Idx::new(parent.children[0], 0);
            }
        }

        true
    }
}

impl<B: BTreeTrait> Default for BTree<B> {
    fn default() -> Self {
        Self::new()
    }
}
