//! Encapsulating logic for splitting of normal and root nodes.
use super::{Inner, Node, Tree};
use crate::{
    cache::AddSize,
    data_management::{Dml, HasStoragePreference, ObjectReference},
    size::Size,
    tree::{errors::*, MessageAction},
};
use std::borrow::Borrow;

impl<X, R, M, I> Tree<X, M, I>
where
    X: Dml<Object = Node<R>, ObjectRef = R>,
    R: ObjectReference<ObjectPointer = X::ObjectPointer> + HasStoragePreference,
    M: MessageAction,
    I: Borrow<Inner<X::ObjectRef, M>>,
{
    pub(super) fn split_root_node(&self, mut root_node: X::CacheValueRefMut) {
        self.dml.verify_cache();
        let before = root_node.size();
        let fanout = root_node.fanout();
        let size = root_node.size();
        let actual_size = root_node.actual_size();
        debug!(
            "Splitting root. {}, {:?}, {}, {:?}",
            root_node.kind(),
            fanout,
            size,
            actual_size
        );

        let size_delta = root_node.split_root_mut(|node, pk| {
            debug!(
                "Root split child: {}, {:?}, {}, {:?}",
                node.kind(),
                0, //node.fanout(), //TODO fix it
                0,//node.size(), // TODO fix it
                0,//node.actual_size() // TODO fix it
            );
            self.dml
                .insert(node, self.tree_id(), pk.to_global(self.tree_id()))
        });
        info!("Root split done. {}, {}", root_node.size(), size_delta);
        debug_assert!(before as isize + size_delta == root_node.size() as isize);
        root_node.finish(size_delta);
        self.dml.verify_cache();
    }

    pub(super) fn split_node(
        &self,
        mut node: X::CacheValueRefMut,
        parent: &mut super::nvminternal::TakeChildBuffer<R>,
    ) -> Result<(X::CacheValueRefMut, isize), Error> {
        self.dml.verify_cache();

        let before = node.size();
        let (sibling, pivot_key, size_delta, lpk) = node.split();
        let pk = lpk.to_global(self.tree_id());
        let select_right = sibling.size() > node.size();
        debug!(
            "split {}: {} -> ({}, {}), {}",
            node.kind(),
            before,
            node.size(),
            sibling.size(),
            select_right,
        );
        node.add_size(size_delta);
        let sibling_np = if select_right {
            let (sibling, np) = self.dml.insert_and_get_mut(sibling, self.tree_id(), pk);
            node = sibling;
            np
        } else {
            self.dml.insert(sibling, self.tree_id(), pk)
        };

        let size_delta = parent.split_child(sibling_np, pivot_key, select_right);

        Ok((node, size_delta))
    }
}
