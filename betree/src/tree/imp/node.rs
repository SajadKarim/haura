//! Implementation of the generic node wrapper.
use self::Inner::*;
use super::{
    child_buffer::ChildBuffer,
    internal::{InternalNode, TakeChildBuffer},
    leaf::{LeafNode, LeafNodeMetaData, LeafNodeData},
    packed::PackedMap,
    FillUpResult, KeyInfo, PivotKey, MAX_INTERNAL_NODE_SIZE, MAX_LEAF_NODE_SIZE, MIN_FANOUT,
    MIN_FLUSH_SIZE, MIN_LEAF_NODE_SIZE,
};
use crate::{
    cow_bytes::{CowBytes, SlicedCowBytes},
    data_management::{Dml, HasStoragePreference, Object, ObjectReference},
    database::{DatasetId,RootSpu},
    size::{Size, SizeMut, StaticSize},
    storage_pool::{DiskOffset, StoragePoolLayer},
    tree::{pivot_key::LocalPivotKey, MessageAction, imp::{/*leaf::ArchivedLeafNode,*/ internal::{InternalNodeMetaData, ArchivedInternalNodeMetaData, ArchivedInternalNodeData, InternalNodeData}}},
    StoragePreference,
};
use bincode::{deserialize, serialize_into};
use bitvec::macros::internal::funty::Numeric;
use parking_lot::RwLock;
use std::{
    borrow::Borrow,
    collections::BTreeMap,
    io::{self, Write},
    mem::replace,
};

use rkyv::{
    archived_root,
    ser::{serializers::AllocSerializer, ScratchSpace, Serializer},
    vec::{ArchivedVec, VecResolver},
    with::{ArchiveWith, DeserializeWith, SerializeWith},
    Archive, Archived, Deserialize, Fallible, Infallible, Serialize,
};

//pub(crate) type RootSpu = crate::storage_pool::StoragePoolUnit<crate::checksum::XxHash>;

/// The tree node type.
#[derive(Debug)]
pub struct Node<N: 'static>(Inner<N>); 
//where S: StoragePoolLayer + 'static;

#[derive(Debug)]
enum NodeInnerType {
    Leaf = 1,
    Internal = 2,
}

#[derive(Debug)]
pub(super) enum Inner<N: 'static>
//where S: StoragePoolLayer + 'static
{
    PackedLeaf(PackedMap),
    Leaf(LeafNode),
    Internal(InternalNode<N>),
}

impl<R: HasStoragePreference + StaticSize> HasStoragePreference for Node<R>/*, S> 
where S: StoragePoolLayer + 'static*/
{
    fn current_preference(&mut self) -> Option<StoragePreference> {
        match self.0 {
            PackedLeaf(_) => None,
            Leaf(ref mut leaf) => leaf.current_preference(),
            Internal(ref mut internal) => internal.current_preference(),
        }
    }

    fn recalculate(&mut self) -> StoragePreference {
        match self.0 {
            PackedLeaf(_) => {
                unreachable!("packed leaves are never written back, have no preference")
            }
            Leaf(ref mut leaf) => leaf.recalculate(),
            Internal(ref mut internal) => internal.recalculate(),
        }
    }

    fn recalculate_lazy(&mut self) -> StoragePreference {
        match self.0 {
            PackedLeaf(_) => {
                unreachable!("packed leaves are never written back, have no preference")
            }
            Leaf(ref mut leaf) => leaf.recalculate(),
            Internal(ref mut internal) => internal.recalculate(),
        }
    }

    fn system_storage_preference(&self) -> StoragePreference {
        match self.0 {
            // A packed leaf does not have a storage preference
            PackedLeaf(_) => unreachable!("packed leaf preference cannot be determined"),
            Leaf(ref leaf) => leaf.system_storage_preference(),
            Internal(ref int) => int.system_storage_preference(),
        }
    }

    fn set_system_storage_preference(&mut self, pref: StoragePreference) {
        // NOTE: This generally has a greater impact as leafs need to be
        // unpacked asap. Another solution as proposed by similar approaches is
        // waiting for the next read operation for this leaf.
        self.ensure_unpacked();
        match self.0 {
            PackedLeaf(_) => unreachable!("packed leaves cannot have their preference updated"),
            Leaf(ref mut leaf) => leaf.set_system_storage_preference(pref),
            Internal(ref mut int) => int.set_system_storage_preference(pref),
        }
    }
}

impl<R: ObjectReference + HasStoragePreference + StaticSize> Object<R> for Node<R> 
//where S: StoragePoolLayer + 'static
{
    fn pack<W: Write>(&mut self, mut writer: W) -> Result<(), io::Error> {
        match &mut self.0 {
            //PackedLeaf(ref map) => { println!("..................... pack leaf node"); writer.write_all(map.inner())},
            //Leaf(ref leaf) => { println!("..................... pack leaf node"); PackedMap::pack(leaf, writer)},
            PackedLeaf(ref map) => unreachable!("... commented out PackedLeaf implementation..."),
            Leaf(ref mut leaf) => {

                let mut serializer_meta_data = rkyv::ser::serializers::AllocSerializer::<0>::default();
                serializer_meta_data.serialize_value(&leaf.meta_data).unwrap();
                let bytes_meta_data = serializer_meta_data.into_serializer().into_inner();

                let mut serializer_data = rkyv::ser::serializers::AllocSerializer::<0>::default();
                serializer_data.serialize_value(leaf.get_data().unwrap()).unwrap();
                let bytes_data = serializer_data.into_serializer().into_inner();

                writer.write_all((NodeInnerType::Leaf as u32).to_be_bytes().as_ref())?;
                writer.write_all(bytes_meta_data.len().to_be_bytes().as_ref())?;
                writer.write_all(bytes_data.len().to_be_bytes().as_ref())?;

                writer.write_all(&bytes_meta_data.as_ref())?;
                writer.write_all(&bytes_data.as_ref())?;

                debug!("Leaf node packed successfully"); 

                Ok(())
            },
            /*Leaf(ref leaf) => {
                serialize_into(writer, leaf)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
            },*/
            /*Internal(ref internal) => {
                writer.write_all(&[0xFFu8, 0xFF, 0xFF, 0xFF] as &[u8])?;
                serialize_into(writer, internal)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
            },*/
            Internal(ref internal) => {
                let mut serializer_meta_data = rkyv::ser::serializers::AllocSerializer::<0>::default();
                serializer_meta_data.serialize_value(&internal.meta_data).unwrap();
                let bytes_meta_data = serializer_meta_data.into_serializer().into_inner();

                let mut serializer_data = rkyv::ser::serializers::AllocSerializer::<0>::default();
                serializer_data.serialize_value(&internal.data).unwrap();
                let bytes_data = serializer_data.into_serializer().into_inner();

                writer.write_all((NodeInnerType::Internal as u32).to_be_bytes().as_ref())?;
                writer.write_all(bytes_meta_data.len().to_be_bytes().as_ref())?;
                writer.write_all(bytes_data.len().to_be_bytes().as_ref())?;

                writer.write_all(&bytes_meta_data.as_ref())?;
                writer.write_all(&bytes_data.as_ref())?;

                debug!("Internal node packed successfully"); 

                Ok(())
            },
        }
    }

    fn unpack_at(size: crate::vdev::Block<u32>,checksum: crate::checksum::XxHash, pool: RootSpu, _offset: DiskOffset, d_id: DatasetId, data: Box<[u8]>) -> Result<Self, io::Error> 
    {
        if data[0..4] == (NodeInnerType::Internal as u32).to_be_bytes() {
            let meta_data_len: usize = usize::from_be_bytes(data[4..12].try_into().unwrap());
            let data_len: usize = usize::from_be_bytes(data[12..20].try_into().unwrap());

            let meta_data_start = 4 + 8 + 8;
            let meta_data_end = meta_data_start + meta_data_len;   

            let data_start = meta_data_end;
            let data_end = data_start + data_len;   

            let archivedinternalnodemetadata: &ArchivedInternalNodeMetaData = rkyv::check_archived_root::<InternalNodeMetaData>(&data[meta_data_start..meta_data_end]).unwrap();
            //let archivedinternalnode: &ArchivedInternalNode<ChildBuffer<_>>  = unsafe { archived_root::<InternalNode<ChildBuffer<R>>>(&data[12..len+12]) };
            let meta_data: InternalNodeMetaData = archivedinternalnodemetadata.deserialize(&mut rkyv::de::deserializers::SharedDeserializeMap::new()).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

            //let archivedinternalnodedata: &ArchivedInternalNodeData<ChildBuffer<_>> = rkyv::check_archived_root::<InternalNodeData<ChildBuffer<R>>>(&data[data_start..data_end]).unwrap();
            //let archivedinternalnode: &ArchivedInternalNode<ChildBuffer<_>>  = unsafe { archived_root::<InternalNode<ChildBuffer<R>>>(&data[12..len+12]) };
            //let data: InternalNodeData<ChildBuffer<R>> = archivedinternalnodedata.deserialize(&mut rkyv::de::deserializers::SharedDeserializeMap::new()).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

            debug!("Leaf node packed successfully"); 
            Ok(Node(Internal (InternalNode {
                pool: Some(pool),
                disk_offset: Some(_offset),
                meta_data : meta_data,
                data: None,//Some(data),
                meta_data_size: meta_data_len,
                data_size: data_len,
                data_start: data_start,
                data_end: data_end,
                node_size: size,
                checksum: Some(checksum),                
            }.complete_object_refs(d_id))))

            /*match deserialize::<InternalNode<_>>(&data[4..]) {
                Ok(internal) => Ok(Node(Internal(internal.complete_object_refs(d_id)))),
                Err(e) => Err(io::Error::new(io::ErrorKind::InvalidData, e)),
            }*/
        } else if data[0..4] == (NodeInnerType::Leaf as u32).to_be_bytes() {
            //panic!("hey!");

            // storage_preference is not preserved for packed leaves,
            // because they will not be written back to disk until modified,
            // and every modification requires them to be unpacked.
            // The leaf contents are scanned cheaply during unpacking, which
            // recalculates the correct storage_preference for the contained keys.

            //Ok(Node(PackedLeaf(PackedMap::new(data.into_vec()))))
            /*match deserialize::<LeafNode>(&data[..]) {
                Ok(leaf) => Ok(Node(Leaf(leaf))),
                Err(e) => Err(io::Error::new(io::ErrorKind::InvalidData, e)),
            }*/

            let meta_data_len: usize = usize::from_be_bytes(data[4..12].try_into().unwrap());
            let data_len: usize = usize::from_be_bytes(data[12..20].try_into().unwrap());

            let meta_data_start = 4 + 8 + 8;
            let meta_data_end = meta_data_start + meta_data_len;   

            let data_start = meta_data_end;
            let data_end = data_start + data_len;   

            let archivedleafnodemetadata = rkyv::check_archived_root::<LeafNodeMetaData>(&data[meta_data_start..meta_data_end]).unwrap();
            //let archivedleafnode: &ArchivedLeafNode = unsafe { archived_root::<LeafNode>(&data) };            
            let meta_data:LeafNodeMetaData = archivedleafnodemetadata.deserialize(&mut rkyv::de::deserializers::SharedDeserializeMap::new()).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

            /*
            let archivedleafnodedata = rkyv::check_archived_root::<LeafNodeData>(&data[data_start..data_end]).unwrap();
            //let archivedleafnode: &ArchivedLeafNode = unsafe { archived_root::<LeafNode>(&data) };            
            let data:LeafNodeData = archivedleafnodedata.deserialize(&mut rkyv::de::deserializers::SharedDeserializeMap::new()).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            */

            let mut abc = LeafNode {
                pool: Some(pool),
                disk_offset: Some(_offset),
                meta_data : meta_data,
                data : None,//Some(data),
                meta_data_size: meta_data_len,
                data_size: data_len,
                data_start: data_start,
                data_end: data_end,
                node_size: size,
                checksum: Some(checksum),
            };
            //abc.load_missing_part();

            debug!("Leaf node packed successfully"); 
            Ok(Node(Leaf(abc)))
                
            /* match deserialize::<LeafNode>(&data[..]) {
                Ok(leaf) => Ok(Node(Leaf(leaf))),
                Err(e) => Err(io::Error::new(io::ErrorKind::InvalidData, e)),
            }*/
            //Err(io::Error::new(io::ErrorKind::InvalidData, "as"))
            
        } else {
            panic!("Unkown bytes to unpack. [0..4]: {}", u32::from_be_bytes(data[..4].try_into().unwrap()));
        }
    }

    fn debug_info(&mut self) -> String {
        let actual_size = self.actual_size();
        let size = self.size();
        let fanout = self.fanout();
        format!(
            "{}: {:?}, {}, {:?}",
            self.kind(),
            fanout,
            size,
            actual_size
        )
    }

    fn for_each_child<E, F>(&mut self, mut f: F) -> Result<(), E>
    where
        F: FnMut(&mut R) -> Result<(), E>,
    {
        if let Some(iter) = self.child_pointer_iter_mut() {
            for np in iter {
                f(np)?;
            }
        }
        Ok(())
    }
}

impl<N: StaticSize> Size for Node<N> 
//where S: StoragePoolLayer + 'static
{
    fn size(&self) -> usize {
        match self.0 {
            PackedLeaf(ref map) => map.size(),
            Leaf(ref leaf) => leaf.size(),
            Internal(ref internal) => 4 + internal.size(),
        }
    }

    fn actual_size(&mut self) -> Option<usize> {
        match self.0 {
            PackedLeaf(ref mut map) => map.actual_size(),
            Leaf(ref mut leaf) => leaf.actual_size(),
            Internal(ref mut internal) => internal.actual_size().map(|size| 4 + size),
        }
    }
}

impl<N: StaticSize + HasStoragePreference> Node<N> {
    pub(super) fn try_walk(&mut self, key: &[u8]) -> Option<TakeChildBuffer<N>> {
        match self.0 {
            Leaf(_) | PackedLeaf(_) => None,
            Internal(ref mut internal) => internal.try_walk(key),
        }
    }

    pub(super) fn try_find_flush_candidate(&mut self) -> Option<TakeChildBuffer<N>> {
        match self.0 {
            Leaf(_) | PackedLeaf(_) => None,
            Internal(ref mut internal) => internal.try_find_flush_candidate(
                MIN_FLUSH_SIZE,
                MAX_INTERNAL_NODE_SIZE,
                MIN_FANOUT,
            ),
        }
    }

    pub(super) fn is_too_large(&self) -> bool {
        match self.0 {
            PackedLeaf(ref map) => map.size() > MAX_LEAF_NODE_SIZE,
            Leaf(ref leaf) => leaf.size() > MAX_LEAF_NODE_SIZE,
            Internal(ref internal) => internal.size() > MAX_INTERNAL_NODE_SIZE,
        }
    }
}

impl<N: HasStoragePreference + StaticSize> Node<N>
//where S: StoragePoolLayer + 'static
{
    pub(super) fn kind(&self) -> &str {
        match self.0 {
            PackedLeaf(_) => "packed leaf",
            Leaf(_) => "leaf",
            Internal(_) => "internal",
        }
    }
    pub(super) fn fanout(&mut self) -> Option<usize> {
        match self.0 {
            Leaf(_) | PackedLeaf(_) => None,
            Internal(ref mut internal) => Some(internal.fanout()),
        }
    }

    fn ensure_unpacked(&mut self) -> isize {
        return 0
        // let before = self.size();

        // let leaf = if let PackedLeaf(ref mut map) = self.0 {
        //     map.unpack_leaf()
        // } else {
        //     return 0;
        // };

        // self.0 = Leaf(leaf);
        // let after = self.size();
        // after as isize - before as isize
    }

    fn take(&mut self) -> Self {
        replace(self, Self::empty_leaf())
    }

    pub(super) fn has_too_low_fanout(&mut self) -> bool {
        match self.0 {
            Leaf(_) | PackedLeaf(_) => false,
            Internal(ref mut internal) => internal.fanout() < MIN_FANOUT,
        }
    }

    pub(super) fn is_too_small_leaf(&self) -> bool {
        match self.0 {
            PackedLeaf(ref map) => map.size() < MIN_LEAF_NODE_SIZE,
            Leaf(ref leaf) => leaf.size() < MIN_LEAF_NODE_SIZE,
            Internal(_) => false,
        }
    }

    pub(super) fn is_too_large_leaf(&self) -> bool {
        match self.0 {
            PackedLeaf(ref map) => map.size() > MAX_LEAF_NODE_SIZE,
            Leaf(ref leaf) => leaf.size() > MAX_LEAF_NODE_SIZE,
            Internal(_) => false,
        }
    }

    pub(super) fn is_leaf(&self) -> bool {
        match self.0 {
            Leaf(_) | PackedLeaf(_) => true,
            Internal(_) => false,
        }
    }

    pub(super) fn empty_leaf() -> Self {
        Node(Leaf(LeafNode::new()))
    }

    pub(super) fn level(&self) -> u32 {
        match self.0 {
            Leaf(_) | PackedLeaf(_) => 0,
            Internal(ref internal) => internal.level(),
        }
    }

    pub(super) fn root_needs_merge(&mut self) -> bool {
        match self.0 {
            Leaf(_) | PackedLeaf(_) => false,
            Internal(ref mut internal) => internal.fanout() == 1,
        }
    }
}

impl<N: ObjectReference + StaticSize + HasStoragePreference> Node<N>
//where S: StoragePoolLayer + 'static
{
    pub(super) fn split_root_mut<F>(&mut self, allocate_obj: F) -> isize
    where
        F: Fn(Self, LocalPivotKey) -> N,
    {
        let size_before = self.size();
        self.ensure_unpacked();
        // FIXME: Update this PivotKey, as the index of the node is changing due to the structural change.
        let mut left_sibling = self.take();
        let (right_sibling, pivot_key, cur_level) = match left_sibling.0 {
            PackedLeaf(_) => unreachable!(),
            Leaf(ref mut leaf) => {
                let (right_sibling, pivot_key, _, _pk) =
                    leaf.split(MIN_LEAF_NODE_SIZE, MAX_LEAF_NODE_SIZE);
                (Node(Leaf(right_sibling)), pivot_key, 0)
            }
            Internal(ref mut internal) => {
                let (right_sibling, pivot_key, _, _pk) = internal.split();
                (Node(Internal(right_sibling)), pivot_key, internal.level())
            }
        };
        debug!("Root split pivot key: {:?}", pivot_key);
        *self = Node(Internal(InternalNode::new(
            ChildBuffer::new(allocate_obj(
                left_sibling,
                LocalPivotKey::LeftOuter(pivot_key.clone()),
            )),
            ChildBuffer::new(allocate_obj(
                right_sibling,
                LocalPivotKey::Right(pivot_key.clone()),
            )),
            pivot_key,
            cur_level + 1,
        )));
        let size_after = self.size();
        size_after as isize - size_before as isize
    }
}

pub(super) enum GetResult<'a, N: 'a> {
    Data(Option<(KeyInfo, SlicedCowBytes)>),
    NextNode(&'a RwLock<N>),
}

pub(super) enum ApplyResult<'a, N: 'a> {
    Leaf(Option<KeyInfo>),
    NextNode(&'a mut N),
}

pub(super) enum PivotGetResult<'a, N: 'a> {
    Target(Option<&'a RwLock<N>>),
    NextNode(&'a RwLock<N>),
}

pub(super) enum PivotGetMutResult<'a, N: 'a> {
    Target(Option<&'a mut N>),
    NextNode(&'a mut N),
}

pub(super) enum GetRangeResult<'a, T, N: 'a> {
    Data(T),
    NextNode {
        np: &'a RwLock<N>,
        prefetch_option: Option<&'a RwLock<N>>,
    },
}

impl<N: HasStoragePreference> Node<N>
//where S: StoragePoolLayer + 'static
{
    pub(super) fn get(
        &mut self,
        key: &[u8],
        msgs: &mut Vec<(KeyInfo, SlicedCowBytes)>,
    ) -> GetResult<N> {
        match self.0 {
            PackedLeaf(ref map) => GetResult::Data(map.get(key)),
            Leaf(ref mut leaf) => GetResult::Data(leaf.get_with_info(key)),
            Internal(ref mut internal) => {
                let (child_np, msg) = internal.get(key);
                if let Some(msg) = msg {
                    msgs.push(msg);
                }
                GetResult::NextNode(child_np)
            }
        }
    }

    pub(super) fn get_range<'a>(
        &'a mut self,
        key: &[u8],
        left_pivot_key: &mut Option<CowBytes>,
        right_pivot_key: &mut Option<CowBytes>,
        all_msgs: &mut BTreeMap<CowBytes, Vec<(KeyInfo, SlicedCowBytes)>>,
    ) -> GetRangeResult<Box<dyn Iterator<Item = (&'a [u8], (KeyInfo, SlicedCowBytes))> + 'a>, N>
    where N: ObjectReference
    {
        match self.0 {
            PackedLeaf(ref map) => GetRangeResult::Data(Box::new(map.get_all())),
            Leaf(ref mut leaf) => GetRangeResult::Data(Box::new(
                leaf.entries().iter().map(|(k, v)| (&k[..], v.clone())),
            )),
            Internal(ref mut internal) => {
                // TODO: load the child required in internal.get_range(key..)
                internal.load_data();

                let prefetch_option = if internal.level() == 1 {
                    internal.get_next_node(key)
                } else {
                    None
                };
                let np = internal.get_range(key, left_pivot_key, right_pivot_key, all_msgs);
                GetRangeResult::NextNode {
                    prefetch_option,
                    np,
                }
            }
        }
    }

    pub(super) fn pivot_get(&mut self, pk: &PivotKey) -> Option<PivotGetResult<N>> {
        if pk.is_root() {
            return Some(PivotGetResult::Target(None));
        }
        match self.0 {
            PackedLeaf(_) | Leaf(_) => None,
            Internal(ref mut internal) => Some(internal.pivot_get(pk)),
        }
    }

    pub(super) fn pivot_get_mut(&mut self, pk: &PivotKey) -> Option<PivotGetMutResult<N>> {
        if pk.is_root() {
            return Some(PivotGetMutResult::Target(None));
        }
        match self.0 {
            PackedLeaf(_) | Leaf(_) => None,
            Internal(ref mut internal) => Some(internal.pivot_get_mut(pk)),
        }
    }
}

impl<N: HasStoragePreference + StaticSize> Node<N> 
//where S: StoragePoolLayer + 'static
{
    pub(super) fn insert<K, M>(
        &mut self,
        key: K,
        msg: SlicedCowBytes,
        msg_action: M,
        storage_preference: StoragePreference,
    ) -> isize
    where
        K: Borrow<[u8]> + Into<CowBytes>,
        M: MessageAction,
    {
        let size_delta = self.ensure_unpacked();
        let keyinfo = KeyInfo { storage_preference };
        size_delta
            + (match self.0 {
                PackedLeaf(_) => unreachable!(),
                Leaf(ref mut leaf) => leaf.insert(key, keyinfo, msg, msg_action),
                Internal(ref mut internal) => internal.insert(key, keyinfo, msg, msg_action),
            })
    }

    pub(super) fn insert_msg_buffer<I, M>(&mut self, msg_buffer: I, msg_action: M) -> isize
    where
        I: IntoIterator<Item = (CowBytes, (KeyInfo, SlicedCowBytes))>,
        M: MessageAction,
    {
        let size_delta = self.ensure_unpacked();
        size_delta
            + (match self.0 {
                PackedLeaf(_) => unreachable!(),
                Leaf(ref mut leaf) => leaf.insert_msg_buffer(msg_buffer, msg_action),
                Internal(ref mut internal) => internal.insert_msg_buffer(msg_buffer, msg_action),
            })
    }

    pub(super) fn apply_with_info(
        &mut self,
        key: &[u8],
        pref: StoragePreference,
    ) -> ApplyResult<N> {
        // FIXME: This is bad for performance, what we want to do here is modify
        // the preference in place determine the new preference and write the
        // PACKED leaf as is again. This violates the restriction that they may
        // never be written again, therefore we need a new interface preparing
        // packed leafs for this exact and only purpose.
        self.ensure_unpacked();
        match self.0 {
            // FIXME: see above
            PackedLeaf(_) => unreachable!(),
            Leaf(ref mut leaf) => ApplyResult::Leaf(leaf.apply(key, pref)),
            Internal(ref mut internal) => {
                ApplyResult::NextNode(internal.apply_with_info(key, pref))
            }
        }
    }
}

impl<N: HasStoragePreference> Node<N> 
//where S: StoragePoolLayer + 'static
{
    pub(super) fn child_pointer_iter_mut(&mut self) -> Option<impl Iterator<Item = &mut N> + '_> {
        match self.0 {
            Leaf(_) | PackedLeaf(_) => None,
            Internal(ref mut internal) => Some(
                internal
                    .iter_mut()
                    .map(|child| child.node_pointer.get_mut()),
            ),
        }
    }

    pub(super) fn child_pointer_iter(&mut self) -> Option<impl Iterator<Item = &RwLock<N>> + '_> {
        match &mut self.0 {
            Leaf(_) | PackedLeaf(_) => None,
            Internal(ref mut internal) => Some(internal.iter().map(|child| &child.node_pointer)),
        }
    }

    pub(super) fn drain_children(&mut self) -> Option<impl Iterator<Item = N> + '_> {
        match self.0 {
            Leaf(_) | PackedLeaf(_) => None,
            Internal(ref mut internal) => Some(internal.drain_children()),
        }
    }
}

impl<N: ObjectReference + StaticSize + HasStoragePreference> Node<N> 
//where S: StoragePoolLayer + 'static
{
    pub(super) fn split(&mut self) -> (Self, CowBytes, isize, LocalPivotKey) {
        self.ensure_unpacked();
        match self.0 {
            PackedLeaf(_) => unreachable!(),
            Leaf(ref mut leaf) => {
                let (node, pivot_key, size_delta, pk) =
                    leaf.split(MIN_LEAF_NODE_SIZE, MAX_LEAF_NODE_SIZE);
                (Node(Leaf(node)), pivot_key, size_delta, pk)
            }
            Internal(ref mut internal) => {
                debug_assert!(
                    internal.fanout() >= 2 * MIN_FANOUT,
                    "internal split failed due to low fanout: {}, size: {}, actual_size: {:?}",
                    internal.fanout(),
                    internal.size(),
                    internal.actual_size()
                );
                let (node, pivot_key, size_delta, pk) = internal.split();
                (Node(Internal(node)), pivot_key, size_delta, pk)
            }
        }
    }

    pub(super) fn merge(&mut self, right_sibling: &mut Self, pivot_key: CowBytes) -> isize {
        self.ensure_unpacked();
        right_sibling.ensure_unpacked();
        match (&mut self.0, &mut right_sibling.0) {
            (&mut Leaf(ref mut left), &mut Leaf(ref mut right)) => left.merge(right),
            (&mut Internal(ref mut left), &mut Internal(ref mut right)) => {
                left.merge(right, pivot_key)
            }
            _ => unreachable!(),
        }
    }

    pub(super) fn leaf_rebalance(&mut self, right_sibling: &mut Self) -> FillUpResult {
        self.ensure_unpacked();
        right_sibling.ensure_unpacked();
        match (&mut self.0, &mut right_sibling.0) {
            (&mut Leaf(ref mut left), &mut Leaf(ref mut right)) => {
                left.rebalance(right, MIN_LEAF_NODE_SIZE, MAX_LEAF_NODE_SIZE)
            }
            _ => unreachable!(),
        }
    }

    /*pub(super) fn range_delete(
        &mut self,
        start: &[u8],
        end: Option<&[u8]>,
    ) -> (isize, Option<(&mut N, Option<&mut N>, Vec<N>)>) {
        self.ensure_unpacked();
        match self.0 {
            PackedLeaf(_) => unreachable!(),
            Leaf(ref mut leaf) => {
                let size_delta = leaf.range_delete(start, end);
                (-(size_delta as isize), None)
            }
            Internal(ref mut internal) => {
                let mut dead = Vec::new();
                let (size_delta, l, r) = internal.range_delete(start, end, &mut dead);
                (-(size_delta as isize), Some((l, r, dead)))
            }
        }
    }*/
}

#[derive(serde::Serialize)]
pub struct ChildInfo {
    from: Option<ByteString>,
    to: Option<ByteString>,
    storage: StoragePreference,
    pub pivot_key: PivotKey,
    pub child: NodeInfo,
}

#[derive(serde::Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum NodeInfo {
    Internal {
        level: u32,
        storage: StoragePreference,
        system_storage: StoragePreference,
        children: Vec<ChildInfo>,
    },
    Leaf {
        level: u32,
        storage: StoragePreference,
        system_storage: StoragePreference,
        entry_count: usize,
    },
    Packed {
        entry_count: u32,
        range: Vec<ByteString>,
    },
}

pub struct ByteString(Vec<u8>);

impl serde::Serialize for ByteString {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        static CHARS: &[u8] = b"0123456789ABCDEF";
        let mut s = String::with_capacity(self.0.len() * 2 + self.0.len() / 4);
        for chunk in self.0.chunks(2) {
            for b in chunk {
                let (upper, lower) = (b >> 4, b & 0x0F);
                s.push(CHARS[upper as usize] as char);
                s.push(CHARS[lower as usize] as char);
            }
            s.push(' ');
        }

        serializer.serialize_str(s.trim_end())
    }
}

impl<N: HasStoragePreference + ObjectReference> Node<N>
//where S: StoragePoolLayer + 'static
{
    pub(crate) fn node_info<D>(&mut self, dml: &D) -> NodeInfo
    where
        D: Dml<Object = Node<N>, ObjectRef = N>,
        N: ObjectReference<ObjectPointer = D::ObjectPointer>,
    {
        let _storage = self.correct_preference();
        let _system_storage = self.system_storage_preference();
        let _level = self.level();

        match &mut self.0 {
            Inner::Internal(int) => NodeInfo::Internal {
                storage: _storage,
                system_storage: _system_storage,
                level: _level,
                children: {
                    int.iter_with_bounds()
                        .map(|(maybe_left, child_buf, maybe_right)| {
                            let (mut child, storage_preference, pivot_key) = {
                                let mut np = child_buf.node_pointer.write();
                                let pivot_key = np.index().clone();
                                let storage_preference = np.correct_preference();
                                let child = dml.get(&mut np).unwrap();
                                (child, storage_preference, pivot_key)
                            };

                            let  node_info = child.node_info(dml);
                            drop(child);

                            dml.evict().unwrap();

                            ChildInfo {
                                from: maybe_left.map(|cow| ByteString(cow.to_vec())),
                                to: maybe_right.map(|cow| ByteString(cow.to_vec())),
                                storage: storage_preference,
                                pivot_key,
                                child: node_info,
                            }
                        })
                        .collect()
                },
            },
            Inner::Leaf(leaf) => NodeInfo::Leaf {
                storage: _storage,
                system_storage: _system_storage,
                level: _level,
                entry_count: leaf.entries().len(),
            },
            Inner::PackedLeaf(packed) => {
                let len = packed.entry_count();
                NodeInfo::Packed {
                    entry_count: len,
                    range: if len == 0 {
                        Vec::new()
                    } else {
                        [
                            packed.get_full_by_index(0),
                            packed.get_full_by_index(len - 1),
                        ]
                        .iter()
                        .filter_map(|opt| {
                            if let Some((key, _)) = opt {
                                Some(ByteString(key.to_vec()))
                            } else {
                                None
                            }
                        })
                        .collect()
                    },
                }
            }
        }
    }
}
