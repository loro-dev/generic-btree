use crate::{BTree, BTreeTrait, Idx, Node, Path, PathRef, QueryResult};

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

pub struct Drain<'a, B: BTreeTrait> {
    node_iter: IterMut<'a, B>,
    start_path: Path,
    start_elem_index: usize,
    end_path: Path,
    end_elem_index: usize,

    reversed_elements: Vec<B::Elem>,
}

impl<'a, B: BTreeTrait> Drain<'a, B> {
    pub fn new(tree: &'a mut BTree<B>, start: QueryResult, end: QueryResult) -> Self {
        Self {
            start_path: start.node_path.clone(),
            start_elem_index: start.elem_index,
            end_path: end.node_path.clone(),
            end_elem_index: end.elem_index,
            node_iter: IterMut::new(tree, start, end),
            reversed_elements: Vec::with_capacity(B::MAX_LEN),
        }
    }
}

impl<'a, B: BTreeTrait> Iterator for Drain<'a, B> {
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
            let start = if idx.arena == self.start_path.last().unwrap().arena {
                self.start_elem_index
            } else {
                0
            };
            let end = if idx.arena == self.end_path.last().unwrap().arena {
                self.end_elem_index
            } else {
                node.elements.len()
            };

            self.reversed_elements = node.elements.drain(start..end).rev().collect();
            // TODO: provide offset & use Q's drain method
        }
    }
}

impl<'a, B: BTreeTrait> Drain<'a, B> {
    fn ensure_trim_start_and_end(&mut self) {
        if self.node_iter.done {
            return;
        }

        let Some((idx, node)) = self.node_iter.next() else { return };
        let is_first = idx.arena == self.start_path.last().unwrap().arena;
        let is_last = idx.arena == self.end_path.last().unwrap().arena;
        let start = if is_first { self.start_elem_index } else { 0 };
        let end = if is_last {
            self.end_elem_index
        } else {
            node.elements.len()
        };

        node.elements.drain(start..end);
        if !is_last {
            let last = self
                .node_iter
                .tree
                .get_mut(self.end_path.last().unwrap().arena);
            last.elements.drain(..self.end_elem_index);
        }
    }
}

impl<'a, B: BTreeTrait> Drop for Drain<'a, B> {
    fn drop(&mut self) {
        self.ensure_trim_start_and_end();
        // leaf nodes can be removed only when their elements are empty
        let mut level = self.start_path.len() - 1;
        let mut deleted = Vec::new();
        let mut new_path = Vec::with_capacity(self.start_path.len());
        while self.start_path[level].arena != self.end_path[level].arena {
            let start_node = self.node_iter.tree.get(self.start_path[level].arena);
            let end_node = self.node_iter.tree.get(self.end_path[level].arena);
            let del_start = if start_node.is_empty() {
                self.start_path[level].arr
            } else {
                self.start_path[level].arr + 1
            };
            let del_end = if end_node.is_empty() {
                self.end_path[level].arr + 1
            } else {
                self.start_path[level].arr
            };
            new_path.push(del_start.max(1) - 1);
            if self.start_path[level - 1].arena == self.end_path[level - 1].arena {
                // parent is the same, delete start..end
                let parent = self
                    .node_iter
                    .tree
                    .get_mut(self.start_path[level - 1].arena);
                for x in parent.children.drain(del_start..del_end) {
                    deleted.push(x.arena);
                }
            } else {
                // parent is different
                {
                    // delete start..
                    let start_parent = self
                        .node_iter
                        .tree
                        .get_mut(self.start_path[level - 1].arena);
                    for x in start_parent.children.drain(del_start..) {
                        deleted.push(x.arena);
                    }
                }
                {
                    // delete ..end
                    let end_parent = self.node_iter.tree.get_mut(self.end_path[level - 1].arena);
                    for x in end_parent.children.drain(..del_end) {
                        deleted.push(x.arena);
                    }
                }
            }
            // this loop would break since level=0 is guaranteed to be the same
        }

        // release memory
        for x in deleted {
            self.node_iter.tree.purge(x);
        }

        // update cache
        loop {
            new_path.push(self.start_path[level].arr);
            if level == 0 {
                break;
            }
            level -= 1;
        }

        new_path.reverse(); // now in root to leaf order
        let path = self.node_iter.tree.get_path_from_indexes(&new_path);
        seal(self.node_iter.tree, path);
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
