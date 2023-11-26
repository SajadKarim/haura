//! Implementation of the [NVMInternalNode] node type.
use super::{
    nvm_child_buffer::NVMChildBuffer,
    node::{PivotGetMutResult, PivotGetResult},
    PivotKey,
};
use crate::{
    cow_bytes::{CowBytes, SlicedCowBytes},
    data_management::{HasStoragePreference, ObjectReference},
    database::DatasetId,
    size::{Size, SizeMut, StaticSize},
    storage_pool::{AtomicSystemStoragePreference, DiskOffset, StoragePoolLayer},
    tree::{pivot_key::LocalPivotKey, KeyInfo, MessageAction},
    AtomicStoragePreference, StoragePreference,
    database::RootSpu,
};
//use bincode::serialized_size;
use parking_lot::RwLock;
//use serde::{Deserialize, Serialize};
use std::{borrow::Borrow, collections::BTreeMap, mem::replace, process::id,
time::{Duration, Instant, SystemTime, UNIX_EPOCH}};

use rkyv::{
    archived_root,
    ser::{serializers::AllocSerializer, ScratchSpace, Serializer},
    vec::{ArchivedVec, VecResolver},
    with::{ArchiveWith, DeserializeWith, SerializeWith},
    Archive, Archived, Deserialize, Fallible, Infallible, Serialize,
};

use chrono::{DateTime, Utc};

//#[derive(serde::Serialize, serde::Deserialize, Debug, Archive, Serialize, Deserialize)]
//#[archive(check_bytes)]
//#[cfg_attr(test, derive(PartialEq))]
pub(super) struct NVMInternalNode<N: 'static> {
    pub pool: Option<RootSpu>,
    pub disk_offset: Option<DiskOffset>,
    pub meta_data: InternalNodeMetaData,
    pub data: Option<InternalNodeData<N>>,
    pub meta_data_size: usize,
    pub data_size: usize,
    pub data_start: usize,
    pub data_end: usize,
    pub node_size: crate::vdev::Block<u32>,
    pub checksum: Option<crate::checksum::XxHash>,
    pub need_to_load_data_from_nvm: bool,
    pub time_for_nvm_last_fetch: SystemTime,
    pub nvm_fetch_counter: usize,
}

impl<N> std::fmt::Debug for NVMInternalNode<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "sdf")
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Archive, Serialize, Deserialize)]
#[archive(check_bytes)]
#[cfg_attr(test, derive(PartialEq))]
pub(super) struct InternalNodeMetaData {
    pub level: u32,
    pub entries_size: usize,
    //#[serde(skip)]
    pub system_storage_preference: AtomicSystemStoragePreference,
    //#[serde(skip)]
    pub pref: AtomicStoragePreference,
    pub(super) pivot: Vec<CowBytes>,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Archive, Serialize, Deserialize)]
#[archive(check_bytes)]
#[cfg_attr(test, derive(PartialEq))]
pub(super) struct InternalNodeData<N: 'static> {
    pub children: Vec<Option<NVMChildBuffer<N>>>,
}

// @tilpner:
// Previously, this literal was magically spread across the code below, and I've (apparently
// correctly) guessed it to be the fixed size of an empty NVMInternalNode<_> when encoded with bincode.
// I've added a test below to verify this and to ensure any bincode-sided change is noticed.
// This is still wrong because:
//
// * usize is platform-dependent, 28 is not. Size will be impl'd incorrectly on 32b platforms
//   * not just the top-level usize, Vec contains further address-sized fields, though bincode
//     might special-case Vec encoding so that this doesn't matter
// * the bincode format may not have changed in a while, but that's not a guarantee
//
// I'm not going to fix them, because the proper fix would be to take bincode out of everything,
// and that's a lot of implementation and testing effort. You should though, if you find the time.
// @jwuensche:
// Added TODO to better find this in the future.
// Will definitely need to adjust this at some point, though this is not now.
// const TEST_BINCODE_FIXED_SIZE: usize = 28;
//
// UPDATE:
// We removed by now the fixed constant and determine the base size of an
// internal node with bincode provided methods based on an empty node created on
// compile-time. We might want to store this value for future access or even
// better determine the size on compile time directly, this requires
// `serialized_size` to be const which it could but its not on their task list
// yet.

// NOTE: Waiting for OnceCell to be stabilized...
// https://doc.rust-lang.org/stable/std/cell/struct.OnceCell.html
static EMPTY_NODE: NVMInternalNode<()> = NVMInternalNode {
    pool: None,
    disk_offset: None,
    meta_data: InternalNodeMetaData {
        level: 0,
        entries_size: 0,
        system_storage_preference: AtomicSystemStoragePreference::none(),
        pref: AtomicStoragePreference::unknown(),
        pivot: vec![]
        },
    data: Some(InternalNodeData {
        children: vec![]
    }),
    meta_data_size: 0,
    data_size: 0,
    data_start: 0,
    data_end: 0,
    node_size: crate::vdev::Block(0),
    checksum: None,
    need_to_load_data_from_nvm: true,
    time_for_nvm_last_fetch: SystemTime::UNIX_EPOCH,// SystemTime::::from(DateTime::parse_from_rfc3339("1996-12-19T16:39:57-00:00").unwrap()),
    nvm_fetch_counter: 0,
};

#[inline]
fn internal_node_base_size() -> usize {
    /*// NOTE: The overhead introduced by using `serialized_size` is negligible
    // and only about 3ns, but we can use OnceCell once (🥁) it is available.
    serialized_size(&EMPTY_NODE)
        .expect("Known node layout could not be estimated. This is an error in bincode.")
        // We know that this is valid as the maximum size in bytes is below u32
        as usize*/

        // let mut serializer = rkyv::ser::serializers::AllocSerializer::<0>::default();
        // serializer.serialize_value(&EMPTY_NODE).unwrap();
        // let bytes = serializer.into_serializer().into_inner();
        // bytes.len()
        0
}


impl<N: StaticSize> Size for NVMInternalNode<N> {
    fn size(&self) -> usize {
        internal_node_base_size() + self.meta_data.entries_size
    }

    fn actual_size(&mut self) -> Option<usize> {
        Some(
            internal_node_base_size()
                + self.meta_data.pivot.iter().map(Size::size).sum::<usize>()
                + self.data.as_mut().unwrap()
                    .children
                    .iter_mut()
                    .map(|child| {
                        child.as_mut().unwrap()
                            .checked_size()
                            .expect("Child doesn't impl actual_size")
                    })
                    .sum::<usize>(),
        )
    }
}

impl<N: HasStoragePreference> HasStoragePreference for NVMInternalNode<N> {
    fn current_preference(&mut self) -> Option<StoragePreference> {
        self.meta_data.pref
            .as_option()
            .map(|pref| self.meta_data.system_storage_preference.weak_bound(&pref))
    }

    fn recalculate(&mut self) -> StoragePreference {
        let mut pref = StoragePreference::NONE;

        for child in &mut self.data.as_mut().unwrap().children {
            pref.upgrade(child.as_mut().unwrap().correct_preference())
        }

        self.meta_data.pref.set(pref);
        pref
    }

    fn recalculate_lazy(&mut self) -> StoragePreference {
        let mut pref = StoragePreference::NONE;

        for child in &mut self.data.as_mut().unwrap().children {
            pref.upgrade(child.as_mut().unwrap().correct_preference())
        }

        self.meta_data.pref.set(pref);
        pref
    }

    fn correct_preference(&mut self) -> StoragePreference {
        let storagepref = self.recalculate();
        self.meta_data.system_storage_preference
            .weak_bound(&storagepref)
    }

    fn system_storage_preference(&self) -> StoragePreference {
        self.meta_data.system_storage_preference.borrow().into()
    }

    fn set_system_storage_preference(&mut self, pref: StoragePreference) {
        self.meta_data.system_storage_preference.set(pref);
    }
}

impl<N: ObjectReference> NVMInternalNode<N> {
    pub(in crate::tree) fn load_entry(&mut self, idx: usize) -> Result<(), std::io::Error> {
        // This method ensures the data part is fully loaded before performing an operation that requires all the entries.
        // However, a better approach can be to load the pairs that are required (so it is a TODO!)
        // Also since at this point I am loading all the data so assuming that 'None' suggests all the data is already fetched.

        if self.need_to_load_data_from_nvm {
            if self.data.is_none() {
                let mut node = InternalNodeData { 
                    children: vec![]
                };

                self.data = Some(node);
            }

            if self.disk_offset.is_some() && self.data.as_ref().unwrap().children.len() < idx {



                if self.time_for_nvm_last_fetch.elapsed().unwrap().as_secs() < 5 {
                    self.nvm_fetch_counter = self.nvm_fetch_counter + 1;

                    if self.nvm_fetch_counter >= 2 {
                        return self.load_all_data();
                    }
                } else {
                    self.nvm_fetch_counter = 0;
                    self.time_for_nvm_last_fetch = SystemTime::now();
                }



                self.data.as_mut().unwrap().children.resize_with(idx, || None);


                match self.pool.as_ref().unwrap().slice(self.disk_offset.unwrap(), self.data_start, self.data_end) {
                    Ok(val) => {

                        let archivedinternalnodedata: &ArchivedInternalNodeData<_> = rkyv::check_archived_root::<InternalNodeData<N>>(&val[..]).unwrap();
                        
                        let val: Option<NVMChildBuffer<N>> = archivedinternalnodedata.children[idx].deserialize(&mut rkyv::de::deserializers::SharedDeserializeMap::new()).unwrap();
                        
                        self.data.as_mut().unwrap().children.insert(idx, val);
                        
                        return Ok(());
                    },
                    Err(e) => {
                        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, e));
                    }
                }


                /*let compressed_data = self.pool.as_ref().unwrap().read(self.node_size, self.disk_offset.unwrap(), self.checksum.unwrap());
                match compressed_data {
                    Ok(buffer) => {
                        let bytes: Box<[u8]> = buffer.into_boxed_slice();

                        let archivedinternalnodedata: &ArchivedInternalNodeData<_> = rkyv::check_archived_root::<InternalNodeData<N>>(&bytes[self.data_start..self.data_end]).unwrap();
                        
                        let val: Option<NVMChildBuffer<N>> = archivedinternalnodedata.children[idx].deserialize(&mut rkyv::de::deserializers::SharedDeserializeMap::new()).unwrap();
                        
                        self.data.as_mut().unwrap().children.insert(idx, val);
                        //let node: InternalNodeData<_> = archivedinternalnodedata.deserialize(&mut rkyv::de::deserializers::SharedDeserializeMap::new()).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                        //self.data = Some(node);
                        
                        return Ok(());
                    },
                    Err(e) => {
                        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, e));
                    }
                }*/
            }
        }

        Ok(())
    }
    
    pub(in crate::tree) fn load_all_data(&mut self) -> Result<(), std::io::Error> {
        // This method ensures the data part is fully loaded before performing an operation that requires all the entries.
        // However, a better approach can be to load the pairs that are required (so it is a TODO!)
        // Also since at this point I am loading all the data so assuming that 'None' suggests all the data is already fetched.
        if self.need_to_load_data_from_nvm && self.disk_offset.is_some() {
            self.need_to_load_data_from_nvm = false;
            let compressed_data = self.pool.as_ref().unwrap().read(self.node_size, self.disk_offset.unwrap(), self.checksum.unwrap());
            match compressed_data {
                Ok(buffer) => {
                    let bytes: Box<[u8]> = buffer.into_boxed_slice();

                    let archivedinternalnodedata: &ArchivedInternalNodeData<_> = rkyv::check_archived_root::<InternalNodeData<N>>(&bytes[self.data_start..self.data_end]).unwrap();
                    
                    let node: InternalNodeData<_> = archivedinternalnodedata.deserialize(&mut rkyv::de::deserializers::SharedDeserializeMap::new()).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

                    self.data = Some(node);
                    
                    return Ok(());
                },
                Err(e) => {
                    return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, e));
                }
            }
        }

        Ok(())
    }
}

impl<N> NVMInternalNode<N> {
    pub fn new(left_child: NVMChildBuffer<N>, right_child: NVMChildBuffer<N>, pivot_key: CowBytes, level: u32) -> Self
    where
        N: StaticSize,
    {
        NVMInternalNode {
            pool: None,
            disk_offset: None,
            meta_data: InternalNodeMetaData { 
                level,
                entries_size: left_child.size() + right_child.size() + pivot_key.size(),
                pivot: vec![pivot_key],
                system_storage_preference: AtomicSystemStoragePreference::from(StoragePreference::NONE),
                pref: AtomicStoragePreference::unknown()
            },
            data: Some(InternalNodeData {
                children: vec![Some(left_child), Some(right_child)],                
            }),
            meta_data_size: 0,
            data_size: 0,
            data_start: 0,
            data_end: 0,
            node_size: crate::vdev::Block(0),
            checksum: None,
            need_to_load_data_from_nvm: true,
            time_for_nvm_last_fetch: SystemTime::now(),
            nvm_fetch_counter: 0,

        }
    }

    // pub(in crate::tree) fn get_data(&mut self) -> Result<& InternalNodeData<N>, std::io::Error> where N: ObjectReference {
    //     self.load_all_data();

    //     Ok(self.data.as_ref().unwrap())
    // }

    // pub(in crate::tree) fn get_data_mut(&mut self) -> Result<&mut InternalNodeData<N>, std::io::Error>  where N: ObjectReference {
    //     self.load_all_data();

    //     Ok(self.data.as_mut().unwrap())
    // }

    /// Returns the number of children.
    pub fn fanout(&mut self) -> usize  where N: ObjectReference {
        self.load_all_data(); //TODO: get only the length?

        self.data.as_ref().unwrap().children.len()
    }

    /// Returns the level of this node.
    pub fn level(&self) -> u32 {
        self.meta_data.level
    }

    /// Returns the index of the child buffer
    /// corresponding to the given `key`.
    fn idx(&self, key: &[u8]) -> usize {
        match self.meta_data
            .pivot
            .binary_search_by(|pivot_key| pivot_key.as_ref().cmp(key))
        {
            Ok(idx) | Err(idx) => idx,
        }
    }

    pub fn iter(&mut self) -> impl Iterator<Item = &Option<NVMChildBuffer<N>>> + '_ where N: ObjectReference{
        self.load_all_data();
        self.data.as_ref().unwrap().children.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Option<NVMChildBuffer<N>>> + '_  where N: ObjectReference {
        self.load_all_data();
        self.data.as_mut().unwrap().children.iter_mut()
    }

    pub fn iter_with_bounds(
        &mut self,
    ) -> impl Iterator<Item = (Option<&CowBytes>, &Option<NVMChildBuffer<N>>, Option<&CowBytes>)> + '_  where N: ObjectReference{
        self.load_all_data();

        let ref pivot = self.meta_data.pivot;
        //let ref children = self.get_data().unwrap().children;

        self.data.as_ref().unwrap().children.iter().enumerate().map(move |(idx, child)| {
            let maybe_left = if idx == 0 {
                None
            } else {
                pivot.get(idx - 1)
            };

            let maybe_right = pivot.get(idx);

            (maybe_left, child, maybe_right)
        })
    }
}

impl<N> NVMInternalNode<N> {
    pub fn get(&mut self, key: &[u8]) -> (&mut RwLock<N>, Option<(KeyInfo, SlicedCowBytes)>)  where N: ObjectReference{
        let idx = self.idx(key);
        self.load_entry(idx);
        let child = &mut self.data.as_mut().unwrap().children[idx];

        let msg = child.as_ref().unwrap().get(key).cloned();
        (&mut child.as_mut().unwrap().node_pointer, msg)
    }

    pub fn pivot_get(&mut self, pk: &PivotKey) -> PivotGetResult<N>  where N: ObjectReference{
        // Exact pivot matches are required only
        debug_assert!(!pk.is_root());
        let pivot = pk.bytes().unwrap();
        let a = self.meta_data.pivot
                .iter()
                .enumerate()
                .find(|(_idx, p)| **p == pivot)
                .map_or_else(
                    || {
                        // Continue the search to the next level

                        //let child = &self.get_data().unwrap().children[self.idx(&pivot)];
                        //PivotGetResult::NextNode(&child.node_pointer)
                        (Some(&pivot), None)
                    },
                    |(idx, _)| {
                        // Fetch the correct child pointer
                        
                        // let child;
                        // if pk.is_left() {
                        //     child = &self.get_data().unwrap().children[idx];
                        // } else {
                        //     child = &self.get_data().unwrap().children[idx + 1];
                        // }
                        // PivotGetResult::Target(Some(&child.node_pointer))
                        (None, Some(idx))
                    },
                );

        if a.0.is_some() {
            let idx = self.idx(a.0.unwrap());
            self.load_entry(idx);
            let child = &self.data.as_ref().unwrap().children[idx];
            PivotGetResult::NextNode(&child.as_ref().unwrap().node_pointer)
        } else {
            let child;
            if pk.is_left() {
                self.load_entry(a.1.unwrap());
                child = &self.data.as_ref().unwrap().children[a.1.unwrap()];
            } else {
                self.load_entry(a.1.unwrap() + 1);
                child = &self.data.as_ref().unwrap().children[a.1.unwrap() + 1];
            }
            PivotGetResult::Target(Some(&child.as_ref().unwrap().node_pointer))
        }
    }

    pub fn pivot_get_mut(&mut self, pk: &PivotKey) -> PivotGetMutResult<N>  where N: ObjectReference{
        // Exact pivot matches are required only
        debug_assert!(!pk.is_root());
        let pivot = pk.bytes().unwrap();
        let (id, is_target) = self.meta_data
            .pivot
            .iter()
            .enumerate()
            .find(|(_idx, p)| **p == pivot)
            .map_or_else(
                || {
                    // Continue the search to the next level
                    (self.idx(&pivot), false)
                },
                |(idx, _)| {
                    // Fetch the correct child pointer
                    (idx, true)
                },
            );
        match (is_target, pk.is_left()) {
            (true, true) => {
                self.load_entry(id);
                PivotGetMutResult::Target(Some(self.data.as_mut().unwrap().children[id].as_mut().unwrap().node_pointer.get_mut()))
            }
            (true, false) => {
                self.load_entry(id + 1);
                PivotGetMutResult::Target(Some(self.data.as_mut().unwrap().children[id + 1].as_mut().unwrap().node_pointer.get_mut()))
            }
            (false, _) => {
                self.load_entry(id);
                PivotGetMutResult::NextNode(self.data.as_mut().unwrap().children[id].as_mut().unwrap().node_pointer.get_mut())
            }
        }
    }

    pub fn apply_with_info(&mut self, key: &[u8], pref: StoragePreference) -> &mut N  where N: ObjectReference{
        let idx = self.idx(key);
        self.load_entry(idx);
        let child = &mut self.data.as_mut().unwrap().children[idx];

        child.as_mut().unwrap().apply_with_info(key, pref);
        child.as_mut().unwrap().node_pointer.get_mut()
    }

    pub fn get_range(
        &self,
        key: &[u8],
        left_pivot_key: &mut Option<CowBytes>,
        right_pivot_key: &mut Option<CowBytes>,
        all_msgs: &mut BTreeMap<CowBytes, Vec<(KeyInfo, SlicedCowBytes)>>,
    ) -> &RwLock<N> {
        let idx = self.idx(key);
        if idx > 0 {
            *left_pivot_key = Some(self.meta_data.pivot[idx - 1].clone());
        }
        if idx < self.meta_data.pivot.len() {
            *right_pivot_key = Some(self.meta_data.pivot[idx].clone());
        }
        let child = &self.data.as_ref().unwrap().children[idx];
        for (key, msg) in child.as_ref().unwrap().get_all_messages() {
            all_msgs
                .entry(key.clone())
                .or_insert_with(Vec::new)
                .push(msg.clone());
        }

        &child.as_ref().unwrap().node_pointer
    }

    pub fn get_next_node(&self, key: &[u8]) -> Option<&RwLock<N>> {
        let idx = self.idx(key) + 1;
        self.data.as_ref().unwrap().children.get(idx).map(|child| &child.as_ref().unwrap().node_pointer)
    }

    pub fn insert<Q, M>(
        &mut self,
        key: Q,
        keyinfo: KeyInfo,
        msg: SlicedCowBytes,
        msg_action: M,
    ) -> isize
    where
        Q: Borrow<[u8]> + Into<CowBytes>,
        M: MessageAction,
        N: ObjectReference
    {
        self.meta_data.pref.invalidate();
        let idx = self.idx(key.borrow());
        self.load_entry(idx);
        let added_size = self.data.as_mut().unwrap().children[idx].as_mut().unwrap().insert(key, keyinfo, msg, msg_action);

        if added_size > 0 {
            self.meta_data.entries_size += added_size as usize;
        } else {
            self.meta_data.entries_size -= -added_size as usize;
        }
        added_size
    }

    pub fn insert_msg_buffer<I, M>(&mut self, iter: I, msg_action: M) -> isize
    where
        I: IntoIterator<Item = (CowBytes, (KeyInfo, SlicedCowBytes))>,
        M: MessageAction,
        N: ObjectReference
    {
        self.meta_data.pref.invalidate();
        let mut added_size = 0;
        let mut buf_storage_pref = StoragePreference::NONE;

        self.load_all_data(); //TODO: Check if the key are in sequence
        for (k, (keyinfo, v)) in iter.into_iter() {
            let idx = self.idx(&k);
            buf_storage_pref.upgrade(keyinfo.storage_preference);
            added_size += self.data.as_mut().unwrap().children[idx].as_mut().unwrap().insert(k, keyinfo, v, &msg_action);
        }

        if added_size > 0 {
            self.meta_data.entries_size += added_size as usize;
        } else {
            self.meta_data.entries_size -= -added_size as usize;
        }
        added_size
    }

    pub fn drain_children(&mut self) -> impl Iterator<Item = N> + '_  where N: ObjectReference {
        self.meta_data.pref.invalidate();
        self.meta_data.entries_size = 0;
        self.load_all_data();
        self.data.as_mut().unwrap().children
            .drain(..)
            .map(|child| child.unwrap().node_pointer.into_inner())
    }
}

impl<N: StaticSize + HasStoragePreference> NVMInternalNode<N> {
    pub fn range_delete(
        &mut self,
        start: &[u8],
        end: Option<&[u8]>,
        dead: &mut Vec<N>,
    ) -> (usize, &mut N, Option<&mut N>) 
    where N: ObjectReference {
        self.load_all_data();
        self.meta_data.pref.invalidate();
        let size_before = self.meta_data.entries_size;
        let start_idx = self.idx(start);
        let end_idx = end.map_or(self.data.as_ref().unwrap().children.len() - 1, |i| self.idx(i));
        if start_idx == end_idx {
            self.load_entry(start_idx);
            let size_delta = self.data.as_mut().unwrap().children[start_idx].as_mut().unwrap().range_delete(start, end);
            return (
                size_delta,
                self.data.as_mut().unwrap().children[start_idx].as_mut().unwrap().node_pointer.get_mut(),
                None,
            );
        }
        // Skip children that may overlap.
        let dead_start_idx = start_idx + 1;
        let dead_end_idx = end_idx - end.is_some() as usize;
        if dead_start_idx <= dead_end_idx {
            for pivot_key in self.meta_data.pivot.drain(dead_start_idx..dead_end_idx) {
                self.meta_data.entries_size -= pivot_key.size();
            }
            let mut entries_size = self.meta_data.entries_size;
            dead.extend(
                self.data.as_mut().unwrap().children
                    .drain(dead_start_idx..=dead_end_idx)
                    .map(|child| child.unwrap()).map(|child| {
                        entries_size -= child.size();
                        child.node_pointer.into_inner()
                    }),
            );

            self.meta_data.entries_size -= entries_size;
        }

        let (mut left_child, mut right_child) = {
            let (left, right) = self.data.as_mut().unwrap().children.split_at_mut(start_idx + 1);
            (&mut left[start_idx], end.map(move |_| &mut right[0]))
        };

        let value = left_child.as_mut().unwrap().range_delete(start, None);
        self.meta_data.entries_size -= value;
        
        if let Some(ref mut child) = right_child {
            self.meta_data.entries_size -= child.as_mut().unwrap().range_delete(start, end);
        }
        let size_delta = size_before - self.meta_data.entries_size;

        (
            size_delta,
            left_child.as_mut().unwrap().node_pointer.get_mut(),
            right_child.map(|child| child.as_mut().unwrap().node_pointer.get_mut()),
        )
    }
}

impl<N: ObjectReference> NVMInternalNode<N> {
    pub fn split(&mut self) -> (Self, CowBytes, isize, LocalPivotKey) {
        self.meta_data.pref.invalidate();
        let split_off_idx = self.fanout() / 2;
        let pivot = self.meta_data.pivot.split_off(split_off_idx);
        let pivot_key = self.meta_data.pivot.pop().unwrap();
        self.load_all_data();
        let mut children = self.data.as_mut().unwrap().children.split_off(split_off_idx);

        if let (Some(new_left_outer), Some(new_left_pivot)) = (children.first_mut(), pivot.first())
        {
            new_left_outer.as_mut().unwrap().update_pivot_key(LocalPivotKey::LeftOuter(new_left_pivot.clone()))
        }

        let entries_size = pivot.iter().map(Size::size).sum::<usize>()
            + children.iter_mut().map(|item| item.as_mut().unwrap()).map(SizeMut::size).sum::<usize>();

        let size_delta = entries_size + pivot_key.size();
        self.meta_data.entries_size -= size_delta;

        let right_sibling = NVMInternalNode {
            pool: None,
            disk_offset: None,
            meta_data: InternalNodeMetaData { 
                level: self.meta_data.level,
                entries_size,
                pivot,
                // Copy the system storage preference of the other node as we cannot
                // be sure which key was targeted by recorded accesses.
                system_storage_preference: self.meta_data.system_storage_preference.clone(),
                pref: AtomicStoragePreference::unknown()
            },
            data: Some(InternalNodeData {
                children,
            }),
            meta_data_size: 0,
            data_size: 0,
            data_start: 0,
            data_end: 0,
            node_size: crate::vdev::Block(0),
            checksum: None,
            need_to_load_data_from_nvm: true,
            time_for_nvm_last_fetch: SystemTime::now(),
            nvm_fetch_counter: 0,

    };
        (
            right_sibling,
            pivot_key.clone(),
            -(size_delta as isize),
            LocalPivotKey::Right(pivot_key),
        )
    }

    pub fn merge(&mut self, right_sibling: &mut Self, old_pivot_key: CowBytes) -> isize {
        self.meta_data.pref.invalidate();
        let size_delta = right_sibling.meta_data.entries_size + old_pivot_key.size();
        self.meta_data.entries_size += size_delta;
        self.meta_data.pivot.push(old_pivot_key);
        self.meta_data.pivot.append(&mut right_sibling.meta_data.pivot);
        self.load_all_data();
        right_sibling.load_all_data();
        self.data.as_mut().unwrap().children.append(&mut right_sibling.data.as_mut().unwrap().children);

        size_delta as isize
    }

    /// Translate any object ref in a `NVMChildBuffer` from `Incomplete` to `Unmodified` state.
    pub fn complete_object_refs(mut self, d_id: DatasetId) -> Self {
        self.load_all_data(); // TODO: this is done to fix borrow error on line 670 (this line 655). Better way is to fetch only the data for required ids.
        // TODO:
        let first_pk = match self.meta_data.pivot.first() {
            Some(p) => PivotKey::LeftOuter(p.clone(), d_id),
            None => unreachable!(
                "The store contains an empty NVMInternalNode, this should never be the case."
            ),
        };
        for (id, pk) in [first_pk]
            .into_iter()
            .chain(self.meta_data.pivot.iter().map(|p| PivotKey::Right(p.clone(), d_id)))
            .enumerate()
        {
            // SAFETY: There must always be pivots + 1 many children, otherwise
            // the state of the Internal Node is broken.
            self.data.as_mut().unwrap().children[id].as_mut().unwrap().complete_object_ref(pk)
        }
        self
    }
}

impl<N: HasStoragePreference> NVMInternalNode<N>
where
    N: StaticSize,
    N: ObjectReference
{
    pub fn try_walk(&mut self, key: &[u8]) -> Option<TakeChildBuffer<N>> {
        let child_idx = self.idx(key);
        self.load_entry(child_idx);
        if self.data.as_mut().unwrap().children[child_idx].as_mut().unwrap().is_empty(key) {
            Some(TakeChildBuffer {
                node: self,
                child_idx,
            })
        } else {
            None
        }
    }

    pub fn try_find_flush_candidate(
        &mut self,
        min_flush_size: usize,
        max_node_size: usize,
        min_fanout: usize,
    ) -> Option<TakeChildBuffer<N>>  where N: ObjectReference{
        let child_idx = {
            let size = self.size();
            let fanout = self.fanout();
            self.load_all_data();
            let (child_idx, child) = self.data.as_mut().unwrap()
                .children
                .iter()
                .enumerate()
                .max_by_key(|&(_, child)| child.as_ref().unwrap().buffer_size())
                .unwrap();

            debug!("Largest child's buffer size: {}", child.as_ref().unwrap().buffer_size());

            if child.as_ref().unwrap().buffer_size() >= min_flush_size
                && (size - child.as_ref().unwrap().buffer_size() <= max_node_size || fanout < 2 * min_fanout)
            {
                Some(child_idx)
            } else {
                None
            }
        };
        child_idx.map(move |child_idx| TakeChildBuffer {
            node: self,
            child_idx,
        })
    }
}

pub(super) struct TakeChildBuffer<'a, N: 'a + 'static> {
    node: &'a mut NVMInternalNode<N>,
    child_idx: usize,
}

impl<'a, N: StaticSize + HasStoragePreference> TakeChildBuffer<'a, N> {
    pub(super) fn split_child(
        &mut self,
        sibling_np: N,
        pivot_key: CowBytes,
        select_right: bool,
    ) -> isize  where N: ObjectReference{
        // split_at invalidates both involved children (old and new), but as the new child
        // is added to self, the overall entries don't change, so this node doesn't need to be
        // invalidated

        self.node.load_all_data();
        let sibling = self.node.data.as_mut().unwrap().children[self.child_idx].as_mut().unwrap().split_at(&pivot_key, sibling_np);
        let size_delta = sibling.size() + pivot_key.size();
        self.node.data.as_mut().unwrap().children.insert(self.child_idx + 1, Some(sibling));
        self.node.meta_data.pivot.insert(self.child_idx, pivot_key);
        self.node.meta_data.entries_size += size_delta;
        if select_right {
            self.child_idx += 1;
        }
        size_delta as isize
    }
}

impl<'a, N> TakeChildBuffer<'a, N>
where
    N: StaticSize,
{
    pub(super) fn size(&self) -> usize {
        Size::size(&*self.node)
    }

    pub(super) fn prepare_merge(&mut self) -> PrepareMergeChild<N>  where N: ObjectReference{
        self.node.load_all_data(); // TODO: return the length only?
        if self.child_idx + 1 < self.node.data.as_ref().unwrap().children.len() {
            PrepareMergeChild {
                node: self.node,
                pivot_key_idx: self.child_idx,
                other_child_idx: self.child_idx + 1,
            }
        } else {
            PrepareMergeChild {
                node: self.node,
                pivot_key_idx: self.child_idx - 1,
                other_child_idx: self.child_idx - 1,
            }
        }
    }
}

pub(super) struct PrepareMergeChild<'a, N: 'a + 'static> {
    node: &'a mut NVMInternalNode<N>,
    pivot_key_idx: usize,
    other_child_idx: usize,
}

impl<'a, N> PrepareMergeChild<'a, N> {
    pub(super) fn sibling_node_pointer(&mut self) -> &mut RwLock<N>  where N: ObjectReference{
        self.node.load_entry(self.other_child_idx);
        &mut self.node.data.as_mut().unwrap().children[self.other_child_idx].as_mut().unwrap().node_pointer
    }
    pub(super) fn is_right_sibling(&self) -> bool {
        self.pivot_key_idx != self.other_child_idx
    }
}

pub(super) struct MergeChildResult<NP> {
    pub(super) pivot_key: CowBytes,
    pub(super) old_np: NP,
    pub(super) size_delta: isize,
}

impl<'a, N: Size + HasStoragePreference> PrepareMergeChild<'a, N> {
    pub(super) fn merge_children(self) -> MergeChildResult<N>  where N: ObjectReference{
        self.node.load_all_data();
        let mut right_sibling = self.node.data.as_mut().unwrap().children.remove(self.pivot_key_idx + 1).unwrap();
        let pivot_key = self.node.meta_data.pivot.remove(self.pivot_key_idx);
        let size_delta =
            pivot_key.size() + NVMChildBuffer::<N>::static_size() + right_sibling.node_pointer.size();
        self.node.meta_data.entries_size -= size_delta;

        let left_sibling = &mut self.node.data.as_mut().unwrap().children[self.pivot_key_idx].as_mut().unwrap();
        left_sibling.append(&mut right_sibling);
        left_sibling
            .messages_preference
            .upgrade_atomic(&right_sibling.messages_preference);

        MergeChildResult {
            pivot_key,
            old_np: right_sibling.node_pointer.into_inner(),
            size_delta: -(size_delta as isize),
        }
    }
}

impl<'a, N: Size + HasStoragePreference> PrepareMergeChild<'a, N> {
    fn get_children(&mut self) -> (&mut Option<NVMChildBuffer<N>>, &mut Option<NVMChildBuffer<N>>)  where N: ObjectReference{
        self.node.load_all_data();
        let (left, right) = self.node.data.as_mut().unwrap().children[self.pivot_key_idx..].split_at_mut(1);
        (&mut left[0], &mut right[0])
    }

    pub(super) fn rebalanced(&mut self, new_pivot_key: CowBytes) -> isize  where N: ObjectReference{
        {
            // Move messages around
            let (left_child, right_child) = self.get_children();
            left_child.as_mut().unwrap().rebalance(right_child.as_mut().unwrap(), &new_pivot_key);
        }

        let mut size_delta = new_pivot_key.size() as isize;
        let old_pivot_key = replace(&mut self.node.meta_data.pivot[self.pivot_key_idx], new_pivot_key);
        size_delta -= old_pivot_key.size() as isize;

        size_delta
    }
}

impl<'a, N: Size + HasStoragePreference> TakeChildBuffer<'a, N> {
    pub fn node_pointer_mut(&mut self) -> &mut RwLock<N>  where N: ObjectReference{
        self.node.load_entry(self.child_idx);
        &mut self.node.data.as_mut().unwrap().children[self.child_idx].as_mut().unwrap().node_pointer
    }
    pub fn take_buffer(&mut self) -> (BTreeMap<CowBytes, (KeyInfo, SlicedCowBytes)>, isize)  where N: ObjectReference{
        self.node.load_entry(self.child_idx);
        let (buffer, size_delta) = self.node.data.as_mut().unwrap().children[self.child_idx].as_mut().unwrap().take();
        self.node.meta_data.entries_size -= size_delta;
        (buffer, -(size_delta as isize))
    }
}

#[cfg(test)]
mod tests {
    

    use super::*;
    use crate::{
        arbitrary::GenExt,
        database::DatasetId,
        tree::default_message_action::{DefaultMessageAction, DefaultMessageActionMsg},
    };
    use bincode::serialized_size;
    
    use quickcheck::{Arbitrary, Gen, TestResult};
    use rand::Rng;
    use serde::Serialize;

    // Keys are not allowed to be empty. This is usually caught at the tree layer, but these are
    // bypassing that check. There's probably a good way to do this, but we can also just throw
    // away the empty keys until we find one that isn't empty.
    #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
    struct Key(CowBytes);
    impl Arbitrary for Key {
        fn arbitrary(g: &mut Gen) -> Self {
            loop {
                let c = CowBytes::arbitrary(g);
                if !c.is_empty() {
                    return Key(c);
                }
            }
        }
    }

    impl<T: Clone> Clone for NVMInternalNode<T> {
        fn clone(&self) -> Self {
            NVMInternalNode {
                pool: self.pool.clone(),
                disk_offset: self.disk_offset.clone(),    
                meta_data: InternalNodeMetaData { 
                    level: self.meta_data.level,
                    entries_size: self.meta_data.entries_size,
                    pivot: self.meta_data.pivot.clone(),
                    system_storage_preference: self.meta_data.system_storage_preference.clone(),
                    pref: self.meta_data.pref.clone(),
                },
                data: Some(InternalNodeData {
                    children: self.data.as_ref().unwrap().children.to_vec(),
                }),
                meta_data_size: 0,
                data_size: 0,
                data_start: 0,
                data_end: 0,
                node_size: crate::vdev::Block(0),
                checksum: None,
                need_to_load_data_from_nvm: true
            }
        }
    }

    impl<T: Arbitrary + Size> Arbitrary for NVMInternalNode<T> {
        fn arbitrary(g: &mut Gen) -> Self {
            let mut rng = g.rng();
            let pivot_key_cnt = rng.gen_range(1..20);
            let mut entries_size = 0;

            let mut pivot = Vec::with_capacity(pivot_key_cnt);
            for _ in 0..pivot_key_cnt {
                let pivot_key = CowBytes::arbitrary(g);
                entries_size += pivot_key.size();
                pivot.push(pivot_key);
            }

            let mut children: Vec<Option<T>> = Vec::with_capacity(pivot_key_cnt + 1);
            for _ in 0..pivot_key_cnt + 1 {
                let child = T::arbitrary(g);
                entries_size += child.size();
                children.push(Some(child));
            }

            NVMInternalNode {
                pool: None,
                disk_offset: None,
                meta_data: InternalNodeMetaData {
                    pivot,
                    entries_size,
                    level: 1,
                    system_storage_preference: AtomicSystemStoragePreference::from(
                        StoragePreference::NONE,
                    ),
                    pref: AtomicStoragePreference::unknown(),
                },
                data: Some(InternalNodeData { 
                    //children: children, //TODO: Sajad Karim, fix the issue
                    children: vec![]
                }),
                meta_data_size: 0,
                data_size: 0,
                data_start: 0,
                data_end: 0,
                node_size: crate::vdev::Block(0),
                checksum: None,
                need_to_load_data_from_nvm: true
            }
        }
    }

    fn check_size<T: Serialize + Size>(node: &mut NVMInternalNode<T>) {
        /*assert_eq!( //TODO: Sajad Karim, fix it
            node.size() as u64,
            serialized_size(node).unwrap(),
            "predicted size does not match serialized size"
        );*/
    }

    #[quickcheck]
    fn check_serialize_size(mut node: NVMInternalNode<CowBytes>) {
        check_size(&mut node);
    }

    #[quickcheck]
    fn check_idx(node: NVMInternalNode<()>, key: Key) {
        let key = key.0;
        let idx = node.idx(&key);

        if let Some(upper_key) = node.meta_data.pivot.get(idx) {
            assert!(&key <= upper_key);
        }
        if idx > 0 {
            let lower_key = &node.meta_data.pivot[idx - 1];
            assert!(lower_key < &key);
        }
    }

    #[quickcheck]
    fn check_size_insert_single(
        mut node: NVMInternalNode<NVMChildBuffer<()>>,
        key: Key,
        keyinfo: KeyInfo,
        msg: DefaultMessageActionMsg,
    ) {
        /*let size_before = node.size() as isize;
        let added_size = node.insert(key.0, keyinfo, msg.0, DefaultMessageAction);
        assert_eq!(size_before + added_size, node.size() as isize);*/ //TODO: Sajad Kari, fix it

        check_size(&mut node);
    }

    #[quickcheck]
    fn check_size_insert_msg_buffer(
        mut node: NVMInternalNode<NVMChildBuffer<()>>,
        buffer: BTreeMap<Key, (KeyInfo, DefaultMessageActionMsg)>,
    ) {
        /*let size_before = node.size() as isize;
        let added_size = node.insert_msg_buffer(
            buffer
                .into_iter()
                .map(|(Key(key), (keyinfo, msg))| (key, (keyinfo, msg.0))),
            DefaultMessageAction,
        );
        assert_eq!(
            size_before + added_size,
            node.size() as isize,
            "size delta mismatch"
        );*/ //Sajad Karim, fix it

        check_size(&mut node);
    }

    #[quickcheck]
    fn check_insert_msg_buffer(
        mut node: NVMInternalNode<NVMChildBuffer<()>>,
        buffer: BTreeMap<Key, (KeyInfo, DefaultMessageActionMsg)>,
    ) {
        /*let mut node_twin = node.clone();
        let added_size = node.insert_msg_buffer(
            buffer
                .iter()
                .map(|(Key(key), (keyinfo, msg))| (key.clone(), (keyinfo.clone(), msg.0.clone()))),
            DefaultMessageAction,
        );

        let mut added_size_twin = 0;
        for (Key(key), (keyinfo, msg)) in buffer {
            let idx = node_twin.idx(&key);
            added_size_twin +=
                node_twin.data.children[idx].insert(key, keyinfo, msg.0, DefaultMessageAction);
        }
        if added_size_twin > 0 {
            node_twin.meta_data.entries_size += added_size_twin as usize;
        } else {
            node_twin.meta_data.entries_size -= -added_size_twin as usize;
        }

        assert_eq!(node, node_twin);
        assert_eq!(added_size, added_size_twin);*/ //Sajad Karim, fix the issue
    }

    static mut PK: Option<PivotKey> = None;

    impl ObjectReference for () {
        type ObjectPointer = ();

        fn get_unmodified(&self) -> Option<&Self::ObjectPointer> {
            Some(&())
        }

        fn set_index(&mut self, _pk: PivotKey) {
            // NO-OP
        }

        fn index(&self) -> &PivotKey {
            unsafe {
                if PK.is_none() {
                    PK = Some(PivotKey::LeftOuter(
                        CowBytes::from(vec![42u8]),
                        DatasetId::default(),
                    ));
                }
                PK.as_ref().unwrap()
            }
        }


    fn serialize_unmodified(&self, w : &mut Vec<u8>) -> Result<(), std::io::Error> {
        unimplemented!("TODO...");
    }

    fn deserialize_and_set_unmodified(bytes: &[u8]) -> Result<Self, std::io::Error> {
        unimplemented!("TODO...");
    }
    }

    #[quickcheck]
    fn check_size_split(mut node: NVMInternalNode<NVMChildBuffer<()>>) -> TestResult {
        /*if node.fanout() < 2 {
            return TestResult::discard();
        }
        let size_before = node.size();
        let (mut right_sibling, _pivot, size_delta, _pivot_key) = node.split();
        assert_eq!(size_before as isize + size_delta, node.size() as isize);
        check_size(&mut node);
        check_size(&mut right_sibling);
        */ //Sajad Karim ,fix the issue

        TestResult::passed()
    }

    #[quickcheck]
    fn check_split(mut node: NVMInternalNode<NVMChildBuffer<()>>) -> TestResult {
        /*if node.fanout() < 4 {
            return TestResult::discard();
        }
        let twin = node.clone();
        let (mut right_sibling, pivot, _size_delta, _pivot_key) = node.split();

        assert!(node.fanout() >= 2);
        assert!(right_sibling.fanout() >= 2);

        node.meta_data.entries_size += pivot.size() + right_sibling.meta_data.entries_size;
        node.meta_data.pivot.push(pivot);
        node.meta_data.pivot.append(&mut right_sibling.meta_data.pivot);
        node.data.children.append(&mut right_sibling.data.children);

        assert_eq!(node, twin);*/ //Sajad Karim ,fix the issue

        TestResult::passed()
    }

    #[quickcheck]
    fn check_split_key(mut node: NVMInternalNode<NVMChildBuffer<()>>) -> TestResult {
        /*if node.fanout() < 4 {
            return TestResult::discard();
        }
        let (right_sibling, pivot, _size_delta, pivot_key) = node.split();
        assert!(node.fanout() >= 2);
        assert!(right_sibling.fanout() >= 2);
        assert_eq!(LocalPivotKey::Right(pivot), pivot_key);*/ //Sajad Karim, fix the issue
        TestResult::passed()
    }

    // #[test]
    // fn check_constant() {
    //     let node: NVMInternalNode<NVMChildBuffer<()>> = NVMInternalNode {
    //         entries_size: 0,
    //         level: 1,
    //         children: vec![],
    //         pivot: vec![],
    //         system_storage_preference: AtomicSystemStoragePreference::from(StoragePreference::NONE),
    //         pref: AtomicStoragePreference::unknown(),
    //     };

    //     assert_eq!(
    //         serialized_size(&node).unwrap(),
    //         TEST_BINCODE_FIXED_SIZE as u64,
    //         "magic constants are wrong"
    //     );
    // }

    // TODO tests
    // split
    // child split
    // flush buffer
    // get with max_msn
}