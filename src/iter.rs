use crate::{BTree, BTreeTrait, Idx, Node, Path, PathRef, Query, QueryResult};

pub(super) struct IterMut<'a, B: BTreeTrait> {
    tree: &'a mut BTree<B>,
    inclusive_end: QueryResult,
    path: Path,
    done: bool,
}

pub(super) struct Iter<'a, B: BTreeTrait> {
    tree: &'a BTree<B>,
    inclusive_end: QueryResult,
    path: Path,
    done: bool,
}

impl<'a, B: BTreeTrait> IterMut<'a, B> {
    pub fn new(tree: &'a mut BTree<B>, start: QueryResult, inclusive_end: QueryResult) -> Self {
        Self {
            tree,
            inclusive_end,
            path: start.node_path,
            done: false,
        }
    }
}

impl<'a, B: BTreeTrait> Iterator for IterMut<'a, B> {
    type Item = (Idx, &'a mut Node<B>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        if self.inclusive_end.node_path.last() == self.path.last() {
            self.done = true;
        }

        let last = *self.path.last().unwrap();
        if !self.tree.next_sibling(&mut self.path) {
            self.done = true;
        }

        let node = self.tree.get_mut(last.arena);
        Some((last, unsafe { std::mem::transmute(node) }))
    }
}

pub struct Drain<'a, B: BTreeTrait, Q: Query<B>> {
    node_iter: IterMut<'a, B>,
    start_query: Q::QueryArg,
    start_result: QueryResult,
    end_query: Q::QueryArg,
    end_result: QueryResult,

    reversed_elements: Vec<B::Elem>,
}

impl<'a, B: BTreeTrait, Q: Query<B>> Drain<'a, B, Q> {
    pub fn new(
        tree: &'a mut BTree<B>,
        start_query: Q::QueryArg,
        end_query: Q::QueryArg,
        start_result: QueryResult,
        end_result: QueryResult,
    ) -> Self {
        Self {
            start_query,
            end_query,
            node_iter: IterMut::new(tree, start_result.clone(), end_result.clone()),
            start_result,
            end_result,
            reversed_elements: Vec::with_capacity(B::MAX_LEN),
        }
    }
}

impl<'a, B: BTreeTrait, Q: Query<B>> Iterator for Drain<'a, B, Q> {
    type Item = B::Elem;

    fn next(&mut self) -> Option<Self::Item> {
        if self.node_iter.done {
            return None;
        }

        loop {
            if let Some(next_elem) = self.reversed_elements.pop() {
                return Some(next_elem);
            }

            let Some((idx, node)) = self.node_iter.next() else { return None };
            let start = if idx.arena == self.start_result.node_path.last().unwrap().arena {
                Some(self.start_result.clone())
            } else {
                None
            };
            let end = if idx.arena == self.end_result.node_path.last().unwrap().arena {
                Some(self.end_result.clone())
            } else {
                None
            };

            let iter = Q::delete_range(
                &mut node.elements,
                &self.start_query,
                &self.end_query,
                start,
                end,
            );
            self.reversed_elements = iter.collect();
            self.reversed_elements.reverse();
            // TODO: provide offset & use Q's drain method
        }
    }
}

impl<'a, B: BTreeTrait, Q: Query<B>> Drain<'a, B, Q> {
    fn ensure_trim_start_and_end(&mut self) {
        let Some((idx, node)) = self.node_iter.next() else { return };
        let start = if idx.arena == self.start_result.node_path.last().unwrap().arena {
            Some(self.start_result.clone())
        } else {
            None
        };
        let is_last = idx.arena == self.end_result.node_path.last().unwrap().arena;
        let end = if is_last {
            Some(self.end_result.clone())
        } else {
            None
        };

        for _ in Q::delete_range(
            &mut node.elements,
            &self.start_query,
            &self.end_query,
            start,
            end,
        ) {}

        if !is_last {
            let last = self
                .node_iter
                .tree
                .get_mut(self.end_result.node_path.last().unwrap().arena);
            for _ in Q::delete_range(
                &mut last.elements,
                &self.start_query,
                &self.end_query,
                None,
                Some(self.end_result.clone()),
            ) {}
        }
    }
}

impl<'a, B: BTreeTrait, Q: Query<B>> Drop for Drain<'a, B, Q> {
    fn drop(&mut self) {
        self.ensure_trim_start_and_end();
        // leaf nodes can be removed only when their elements are empty
        let start_path = &self.start_result.node_path;
        let end_path = &self.end_result.node_path;
        let mut level = start_path.len() - 1;
        let mut deleted = Vec::new();
        let mut new_path = Vec::with_capacity(start_path.len());
        while start_path[level].arena != end_path[level].arena {
            let start_node = self.node_iter.tree.get(start_path[level].arena);
            let end_node = self.node_iter.tree.get(end_path[level].arena);
            let del_start = if start_node.is_empty() {
                start_path[level].arr
            } else {
                start_path[level].arr + 1
            };
            let del_end = if end_node.is_empty() {
                end_path[level].arr + 1
            } else {
                end_path[level].arr
            };
            new_path.push(del_start.max(1) - 1);
            if start_path[level - 1].arena == end_path[level - 1].arena {
                // parent is the same, delete start..end
                let parent = self.node_iter.tree.get_mut(start_path[level - 1].arena);
                for x in parent.children.drain(del_start..del_end) {
                    deleted.push(x.arena);
                }
            } else {
                // parent is different
                {
                    // delete start..
                    let start_parent = self.node_iter.tree.get_mut(start_path[level - 1].arena);
                    for x in start_parent.children.drain(del_start..) {
                        deleted.push(x.arena);
                    }
                }
                {
                    // delete ..end
                    let end_parent = self.node_iter.tree.get_mut(end_path[level - 1].arena);
                    for x in end_parent.children.drain(..del_end) {
                        deleted.push(x.arena);
                    }
                }
            }

            level -= 1
            // this loop would break since level=0 is guaranteed to be the same
        }

        // release memory
        for x in deleted {
            self.node_iter.tree.purge(x);
        }

        // update cache
        loop {
            new_path.push(start_path[level].arr);
            if level == 0 {
                break;
            }
            level -= 1;
        }

        new_path.reverse(); // now in root to leaf order
        if let Some(path) = self.node_iter.tree.try_get_path_from_indexes(&new_path) {
            // otherwise the path is invalid (e.g. the tree is empty)
            seal(self.node_iter.tree, path);
        }
    }
}

fn seal<B: BTreeTrait>(tree: &mut BTree<B>, path: Path) {
    let mut sibling_path = path.clone();
    let same = !tree.next_sibling(&mut sibling_path);
    tree.recursive_update_cache((&path).into());
    if !same {
        tree.recursive_update_cache((&sibling_path).into());
    }

    for i in 1..path.len() {
        let idx = path[i];
        let node = tree.get(idx.arena);
        let is_lack = node.is_lack();
        let is_full = node.is_full();
        let path_ref: PathRef = path[0..i].into();
        if is_lack {
            tree.handle_lack(&path_ref);
        } else if is_full {
            tree.split(path_ref)
        }
    }
}

impl<'a, B: BTreeTrait> Iter<'a, B> {
    pub fn new(tree: &'a BTree<B>, start: QueryResult, inclusive_end: QueryResult) -> Self {
        Self {
            tree,
            inclusive_end,
            path: start.node_path,
            done: false,
        }
    }
}

impl<'a, B: BTreeTrait> Iterator for Iter<'a, B> {
    type Item = (Idx, &'a Node<B>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        if self.inclusive_end.node_path.last() == self.path.last() {
            self.done = true;
        }

        let last = *self.path.last().unwrap();
        if !self.tree.next_sibling(&mut self.path) {
            self.done = true;
        }

        let node = self.tree.get(last.arena);
        Some((last, unsafe { std::mem::transmute(node) }))
    }
}
