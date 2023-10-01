//! Implementation of the [LeafNode] node type.
use crate::{
    cow_bytes::{CowBytes, SlicedCowBytes},
    data_management::HasStoragePreference,
    size::Size,
    storage_pool::{AtomicSystemStoragePreference, DiskOffset, StoragePoolLayer},
    tree::{imp::packed, pivot_key::LocalPivotKey, KeyInfo, MessageAction},
    AtomicStoragePreference, StoragePreference,
    database::RootSpu,
};
use std::{borrow::Borrow, collections::BTreeMap, iter::FromIterator};

//use serde::{Deserialize, Serialize};
//use rkyv::{Archive, Deserialize, Serialize};
//use rkyv::ser::{Serializer, serializers::AllocSerializer};
use rkyv::{
    archived_root,
    ser::{serializers::AllocSerializer, ScratchSpace, Serializer},
    vec::{ArchivedVec, VecResolver},
    with::{ArchiveWith, DeserializeWith, SerializeWith},
    Archive, Archived, Deserialize, Fallible, Infallible, Serialize,
};

use std::os::raw::c_void;

use extend::ext;

#[ext]
impl<T> Option<T> {
    fn as_mut_lazy(&mut self) -> &mut T {
        match *self {
            Some(ref mut x) => x,
            None => {
                panic!("TODO... request storagepool for the data..")
            },
        }
    }

    fn as_ref_lazy(&self) -> &T {
        match *self {
            Some(ref x) => x,
            None => {
                panic!("TODO... request storagepool for the data..")
            },
        }
    }
}

/// A leaf node of the tree holds pairs of keys values which are plain data.
#[derive(Clone)]
//#[archive(check_bytes)]
#[cfg_attr(test, derive(PartialEq))]
pub(super) struct LeafNode/*<S> 
where S: StoragePoolLayer + 'static*/
{ 
    //#[with(Skip)]
    pub pool: Option<RootSpu>,
    pub disk_offset: Option<DiskOffset>,
    pub meta_data: LeafNodeMetaData,
    pub data: Option<LeafNodeData>,
    //pub data: LeafNodeData,
    pub meta_data_size: usize,
    pub data_size: usize,
    pub data_start: usize,
    pub data_end: usize,
    pub node_size: crate::vdev::Block<u32>,
    pub checksum: Option<crate::checksum::XxHash>,
    pub need_to_load_data_from_nvm: bool
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Archive, Serialize, Deserialize)]
#[archive(check_bytes)]
#[cfg_attr(test, derive(PartialEq))]
pub(super) struct LeafNodeMetaData {
    pub storage_preference: AtomicStoragePreference,
    /// A storage preference assigned by the Migration Policy
    pub system_storage_preference: AtomicSystemStoragePreference,
    pub entries_size: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Archive, Serialize, Deserialize)]
#[archive(check_bytes)]
#[cfg_attr(test, derive(PartialEq))]

pub struct LeafNodeData {
    #[with(rkyv::with::AsVec)]
    pub entries: BTreeMap<CowBytes, (KeyInfo, SlicedCowBytes)>,
}

impl std::fmt::Debug for LeafNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "todo...")
    }
}

unsafe fn voidp_to_ref<'a, T>(p: *const c_void) -> &'a T
{
    unsafe { &*(p as *const T) }
}

fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}


/// Case-dependent outcome of a rebalance operation.
#[derive(Debug)]
pub(super) enum FillUpResult {
    Rebalanced {
        pivot_key: CowBytes,
        size_delta: isize,
    },
    Merged {
        size_delta: isize,
    },
}

impl Size for LeafNode/*<S> 
where S: StoragePoolLayer + 'static*/
{
    fn size(&self) -> usize {
        packed::HEADER_FIXED_LEN + self.meta_data.entries_size
    }

    fn actual_size(&mut self) -> Option<usize> {
        Some(
            packed::HEADER_FIXED_LEN
                + self.data.as_ref().unwrap()
                    .entries
                    .iter()
                    .map(|(key, (_keyinfo, value))| packed::ENTRY_LEN + key.len() + value.len())
                    .sum::<usize>(),
        )
    }
}

impl HasStoragePreference for LeafNode/*<S>
where S: StoragePoolLayer + 'static*/
{
    fn current_preference(&mut self) -> Option<StoragePreference> {
        self.meta_data.storage_preference
            .as_option()
            .map(|pref| self.meta_data.system_storage_preference.weak_bound(&pref))
    }

    fn recalculate(&mut self) -> StoragePreference {
        let mut pref = StoragePreference::NONE;

        for (keyinfo, _v) in self.get_all_entries().unwrap().entries.values() {
            pref.upgrade(keyinfo.storage_preference);
        }

        self.meta_data.storage_preference.set(pref);
        self.meta_data.system_storage_preference.weak_bound(&pref)
    }

    fn recalculate_lazy(&mut self) -> StoragePreference {
        let mut pref = StoragePreference::NONE;

        for (keyinfo, _v) in self.get_all_entries().unwrap().entries.values() {
            pref.upgrade(keyinfo.storage_preference);
        }

        self.meta_data.storage_preference.set(pref);
        self.meta_data.system_storage_preference.weak_bound(&pref)
    }

    fn system_storage_preference(&self) -> StoragePreference {
        self.meta_data.system_storage_preference.borrow().into()
    }

    fn set_system_storage_preference(&mut self, pref: StoragePreference) {
        self.meta_data.system_storage_preference.set(pref)
    }
}

impl<'a> FromIterator<(&'a [u8], (KeyInfo, SlicedCowBytes))> for LeafNode/*<S> 
where S: StoragePoolLayer + 'static*/
{
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = (&'a [u8], (KeyInfo, SlicedCowBytes))>,
    {
        let mut storage_pref = StoragePreference::NONE;
        let mut entries_size = 0;

        let mut entries = BTreeMap::new();
        let mut needs_second_pass = false;

        for (key, (keyinfo, value)) in iter.into_iter() {
            // pref of overall node is highest pref from keys.
            // We're already looking at every entry here, so finding the overall pref here
            // avoids a full scan later.
            storage_pref.upgrade(keyinfo.storage_preference);
            entries_size += packed::ENTRY_LEN + key.len() + value.len();

            let curr_storage_pref = keyinfo.storage_preference;
            if let Some((ckeyinfo, cvalue)) = entries.insert(CowBytes::from(key), (keyinfo, value))
            {
                // iterator has collisions, try to compensate
                //
                // this entry will no longer be part of the final map, subtract its size
                entries_size -= packed::ENTRY_LEN + key.len() + cvalue.len();

                // In case the old value increased the overall storage priority (faster), and the new
                // value wouldn't have increased it as much, we might need to recalculate the
                // proper preference in a second pass.
                if ckeyinfo.storage_preference != curr_storage_pref {
                    needs_second_pass = true;
                }
            }
        }

        if needs_second_pass {
            storage_pref = StoragePreference::NONE;
            for (keyinfo, _value) in entries.values() {
                storage_pref.upgrade(keyinfo.storage_preference);
            }
        }

        LeafNode {
            pool: None,
            disk_offset: None,
            meta_data: LeafNodeMetaData { 
                storage_preference: AtomicStoragePreference::known(storage_pref),
                system_storage_preference: AtomicSystemStoragePreference::from(StoragePreference::NONE),
                entries_size
            },
            data: Some(LeafNodeData { 
                entries: entries
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

impl LeafNode/*<S>
where S: StoragePoolLayer + 'static*/
{
    /// Constructs a new, empty `LeafNode`.
    pub fn new() -> Self {
        LeafNode {
            pool: None,
            disk_offset: None,
            meta_data: LeafNodeMetaData { 
                storage_preference: AtomicStoragePreference::known(StoragePreference::NONE),
                system_storage_preference: AtomicSystemStoragePreference::from(StoragePreference::NONE),
                entries_size: 0,
            },
            data: Some(LeafNodeData { 
                entries: BTreeMap::new()
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

    pub(in crate::tree) fn get_entry(&mut self, key: &[u8]) -> Result<& LeafNodeData, std::io::Error> {
        if self.need_to_load_data_from_nvm {
            if self.data.is_none() {
                let mut leafnode = LeafNodeData { 
                    entries: BTreeMap::new()
                };

                self.data = Some(leafnode);
            }
            
            if self.disk_offset.is_some() && !self.data.as_ref().unwrap().entries.contains_key(key) {
                let compressed_data = self.pool.as_ref().unwrap().read(self.node_size, self.disk_offset.unwrap(), self.checksum.unwrap());
                match compressed_data {
                    Ok(buffer) => {
                        let bytes: Box<[u8]> = buffer.into_boxed_slice();

                        let archivedleafnodedata: &ArchivedLeafNodeData = rkyv::check_archived_root::<LeafNodeData>(&bytes[self.data_start..self.data_end]).unwrap();
                        
                        for val in archivedleafnodedata.entries.iter() {
                            if val.key.as_ref().cmp(key).is_eq() {
                                let val_1: KeyInfo = val.value.0.deserialize(&mut rkyv::Infallible).unwrap();
                                let val_2: SlicedCowBytes = val.value.1.deserialize(&mut rkyv::de::deserializers::SharedDeserializeMap::new()).unwrap();

                                let key: CowBytes = val.key.deserialize(&mut rkyv::de::deserializers::SharedDeserializeMap::new()).unwrap();

                                self.data.as_mut().unwrap().entries.insert(key, (val_1, val_2));
                            }
                        }

                        return Ok(self.data.as_ref().unwrap());
                    },
                    Err(e) => {
                        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, e));
                    }
                }
            }
        }

        Ok(self.data.as_ref().unwrap())
    }

    pub(in crate::tree) fn get_entry_mut(&mut self, key: &[u8]) -> Result<&mut LeafNodeData, std::io::Error> {
        if self.need_to_load_data_from_nvm {
            if self.data.is_none() {
                let mut leafnode = LeafNodeData { 
                    entries: BTreeMap::new()
                };

                self.data = Some(leafnode);
            }
            
            if self.disk_offset.is_some() && !self.data.as_ref().unwrap().entries.contains_key(key) {
                let compressed_data = self.pool.as_ref().unwrap().read(self.node_size, self.disk_offset.unwrap(), self.checksum.unwrap());
                match compressed_data {
                    Ok(buffer) => {
                        let bytes: Box<[u8]> = buffer.into_boxed_slice();

                        let archivedleafnodedata: &ArchivedLeafNodeData = rkyv::check_archived_root::<LeafNodeData>(&bytes[self.data_start..self.data_end]).unwrap();
                        
                        for val in archivedleafnodedata.entries.iter() {
                            if val.key.as_ref().cmp(key).is_eq() {
                                let val_1: KeyInfo = val.value.0.deserialize(&mut rkyv::Infallible).unwrap();
                                let val_2: SlicedCowBytes = val.value.1.deserialize(&mut rkyv::de::deserializers::SharedDeserializeMap::new()).unwrap();

                                let key: CowBytes = val.key.deserialize(&mut rkyv::de::deserializers::SharedDeserializeMap::new()).unwrap();

                                self.data.as_mut().unwrap().entries.insert(key, (val_1, val_2));
                            }
                        }

                        return Ok(self.data.as_mut().unwrap());
                    },
                    Err(e) => {
                        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, e));
                    }
                }
            }
        }

        Ok(self.data.as_mut().unwrap())
    }

    pub(in crate::tree) fn get_all_entries(&mut self) -> Result<& LeafNodeData, std::io::Error> {
        if self.need_to_load_data_from_nvm && self.disk_offset.is_some() {
            self.need_to_load_data_from_nvm = false; // TODO: What if all the entries are fetched one by one? handle this part as well.
            let compressed_data = self.pool.as_ref().unwrap().read(self.node_size, self.disk_offset.unwrap(), self.checksum.unwrap());
            match compressed_data {
                Ok(buffer) => {
                    let bytes: Box<[u8]> = buffer.into_boxed_slice();

                    let archivedleafnodedata: &ArchivedLeafNodeData = rkyv::check_archived_root::<LeafNodeData>(&bytes[self.data_start..self.data_end]).unwrap();
                    let node:LeafNodeData = archivedleafnodedata.deserialize(&mut rkyv::de::deserializers::SharedDeserializeMap::new()).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

                    self.data = Some(node);
                    return Ok(self.data.as_ref().unwrap());
                },
                Err(e) => {
                    return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, e));
                }
            }
        }

        Ok(self.data.as_ref().unwrap())
    }

    pub(in crate::tree) fn get_all_entries_mut(&mut self) -> Result<&mut LeafNodeData, std::io::Error> {
        if self.need_to_load_data_from_nvm && self.disk_offset.is_some() {
            self.need_to_load_data_from_nvm = false;
            let compressed_data = self.pool.as_ref().unwrap().read(self.node_size, self.disk_offset.unwrap(), self.checksum.unwrap());
            match compressed_data {
                Ok(buffer) => {
                    let bytes: Box<[u8]> = buffer.into_boxed_slice();

                    let archivedleafnodedata: &ArchivedLeafNodeData = rkyv::check_archived_root::<LeafNodeData>(&bytes[self.data_start..self.data_end]).unwrap();
                    let node:LeafNodeData = archivedleafnodedata.deserialize(&mut rkyv::de::deserializers::SharedDeserializeMap::new()).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

                    self.data = Some(node);
                    return Ok(self.data.as_mut().unwrap());

                },
                Err(e) => {
                    return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, e));
                }
            }
        }

        Ok(self.data.as_mut().unwrap())
    }

    pub(in crate::tree) fn set_data(&mut self, obj: LeafNodeData) {
        self.data = Some(obj);
    }

    /// Returns the value for the given key.
    pub fn get(&mut self, key: &[u8]) -> Option<SlicedCowBytes> {
        self.get_entry(key).unwrap().entries.get(key).map(|(_info, data)| data).cloned()
    }

    pub(in crate::tree) fn get_with_info(&mut self, key: &[u8]) -> Option<(KeyInfo, SlicedCowBytes)> {
        self.get_entry(key).unwrap().entries.get(key).cloned()
    }

    pub(in crate::tree) fn entries(&mut self) -> &BTreeMap<CowBytes, (KeyInfo, SlicedCowBytes)> {
        &self.get_all_entries().unwrap().entries
    }

    pub(in crate::tree) fn entry_info(&mut self, key: &[u8]) -> Option<&mut KeyInfo> {
        self.get_entry_mut(key).unwrap().entries.get_mut(key).map(|e| &mut e.0)
    }

    /// Split the node and transfer entries to a given other node `right_sibling`.
    /// Use entries which are, when summed up in-order, above the `min_size` limit.
    /// Returns new pivot key and size delta to the left sibling.
    fn do_split_off(
        &mut self,
        right_sibling: &mut Self,
        min_size: usize,
        max_size: usize,
    ) -> (CowBytes, isize) {
        debug_assert!(self.size() > max_size);
        debug_assert!(right_sibling.meta_data.entries_size == 0);

        let mut sibling_size = 0;
        let mut sibling_pref = StoragePreference::NONE;
        let mut split_key = None;
        for (k, (keyinfo, v)) in self.get_all_entries().unwrap().entries.iter().rev() {
            sibling_size += packed::ENTRY_LEN + k.len() + v.len();
            sibling_pref.upgrade(keyinfo.storage_preference);

            if packed::HEADER_FIXED_LEN + sibling_size >= min_size {
                split_key = Some(k.clone());
                break;
            }
        }
        let split_key = split_key.unwrap();

        right_sibling.get_all_entries_mut().unwrap().entries = self.get_all_entries_mut().unwrap().entries.split_off(&split_key);
        self.meta_data.entries_size -= sibling_size;
        right_sibling.meta_data.entries_size = sibling_size;
        right_sibling.meta_data.storage_preference.set(sibling_pref);

        // have removed many keys from self, no longer certain about own pref, mark invalid
        self.meta_data.storage_preference.invalidate();

        let size_delta = -(sibling_size as isize);

        let pivot_key = self.get_all_entries().unwrap().entries.keys().next_back().cloned().unwrap();
        (pivot_key, size_delta)
    }

    pub fn apply<K>(&mut self, key: K, pref: StoragePreference) -> Option<KeyInfo>
    where
        K: Borrow<[u8]>,
    {
        self.meta_data.storage_preference.invalidate();
        self.get_entry_mut(key.borrow()).unwrap().entries.get_mut(key.borrow()).map(|entry| {
            entry.0.storage_preference = pref;
            entry.0.clone()
        })
    }

    /// Inserts a new message as leaf entry.
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
    {
        let size_before = self.meta_data.entries_size as isize;
        let key_size = key.borrow().len();
        let mut data = self.get(key.borrow());
        msg_action.apply_to_leaf(key.borrow(), msg, &mut data);

        if let Some(data) = data {
            // Value was added or preserved by msg
            self.meta_data.entries_size += data.len();
            self.meta_data.storage_preference.upgrade(keyinfo.storage_preference);

            if let Some((old_info, old_data)) =
                self.get_all_entries_mut().unwrap().entries.insert(key.into(), (keyinfo.clone(), data))
            {
                // There was a previous value in entries, which was now replaced
                self.meta_data.entries_size -= old_data.len();

                // if previous entry was stricter than new entry, invalidate
                if old_info.storage_preference < keyinfo.storage_preference {
                    self.meta_data.storage_preference.invalidate();
                }
            } else {
                // There was no previous value in entries
                self.meta_data.entries_size += packed::ENTRY_LEN;
                self.meta_data.entries_size += key_size;
            }
        } else if let Some((old_info, old_data)) = self.get_entry_mut(key.borrow()).unwrap().entries.remove(key.borrow()) {
            // The value was removed by msg, this may be a downgrade opportunity.
            // The preference of the removed entry can't be stricter than the current node
            // preference, by invariant. That leaves "less strict" and "as strict" as the
            // node preference:
            //
            // - less strict:
            //     If the preference of the removed entry is less strict than the current
            //     node preference, there must be another entry which is preventing a downgrade.
            // - as strict:
            //     The removed entry _may_ have caused the original upgrade to this preference,
            //     we'll have to trigger a scan to find out.
            if self.meta_data.storage_preference.as_option() == Some(old_info.storage_preference) {
                self.meta_data.storage_preference.invalidate();
            }

            self.meta_data.entries_size -= packed::ENTRY_LEN;
            self.meta_data.entries_size -= key_size;
            self.meta_data.entries_size -= old_data.len();
        }
        self.meta_data.entries_size as isize - size_before
    }

    /// Inserts messages as leaf entries.
    pub fn insert_msg_buffer<M, I>(&mut self, msg_buffer: I, msg_action: M) -> isize
    where
        M: MessageAction,
        I: IntoIterator<Item = (CowBytes, (KeyInfo, SlicedCowBytes))>,
    {
        let mut size_delta = 0;
        for (key, (keyinfo, msg)) in msg_buffer {
            size_delta += self.insert(key, keyinfo, msg, &msg_action);
        }
        size_delta
    }

    /// Splits this `LeafNode` into to two leaf nodes.
    /// Returns a new right sibling, the corresponding pivot key, and the size
    /// delta of this node.
    pub fn split(
        &mut self,
        min_size: usize,
        max_size: usize,
    ) -> (Self, CowBytes, isize, LocalPivotKey) {
        // assert!(self.size() > S::MAX);
        let mut right_sibling = LeafNode {
            pool: None,
            disk_offset: None,
            // During a split, preference can't be inherited because the new subset of entries
            // might be a subset with a lower maximal preference.
            meta_data: LeafNodeMetaData { 
                storage_preference: AtomicStoragePreference::known(StoragePreference::NONE),
                system_storage_preference: AtomicSystemStoragePreference::from(StoragePreference::NONE),
                entries_size: 0
            },
            data: Some(LeafNodeData { 
                entries: BTreeMap::new()
            }),
            meta_data_size: 0,
            data_size: 0,
            data_start: 0,
            data_end: 0,
            node_size: crate::vdev::Block(0),
            checksum: None,
            need_to_load_data_from_nvm: true
        };

        // This adjusts sibling's size and pref according to its new entries
        let (pivot_key, size_delta) = self.do_split_off(&mut right_sibling, min_size, max_size);

        (
            right_sibling,
            pivot_key.clone(),
            size_delta,
            LocalPivotKey::Right(pivot_key),
        )
    }

    /// Merge all entries from the *right* node into the *left* node.  Returns
    /// the size change, positive for the left node, negative for the right
    /// node.
    pub fn merge(&mut self, right_sibling: &mut Self) -> isize {
        self.get_all_entries_mut().unwrap().entries.append(&mut right_sibling.get_all_entries_mut().unwrap().entries);
        let size_delta = right_sibling.meta_data.entries_size;
        self.meta_data.entries_size += right_sibling.meta_data.entries_size;

        self.meta_data.storage_preference
            .upgrade_atomic(&right_sibling.meta_data.storage_preference);

        // right_sibling is now empty, reset to defaults
        right_sibling.meta_data.entries_size = 0;
        right_sibling.meta_data
            .storage_preference
            .set(StoragePreference::NONE);

        size_delta as isize
    }

    /// Rebalances `self` and `right_sibling`. Returns `Merged`
    /// if all entries of `right_sibling` have been merged into this node.
    /// Otherwise, returns a new pivot key.
    pub fn rebalance(
        &mut self,
        right_sibling: &mut Self,
        min_size: usize,
        max_size: usize,
    ) -> FillUpResult {
        let size_delta = self.merge(right_sibling);
        if self.size() <= max_size {
            FillUpResult::Merged { size_delta }
        } else {
            // First size_delta is from the merge operation where we split
            let (pivot_key, split_size_delta) =
                self.do_split_off(right_sibling, min_size, max_size);
            FillUpResult::Rebalanced {
                pivot_key,
                size_delta: size_delta + split_size_delta,
            }
        }
    }

    /*pub fn range_delete(&mut self, start: &[u8], end: Option<&[u8]>) -> usize {
        // https://github.com/rust-lang/rust/issues/42849
        let size_before = self.entries_size;
        let range = (
            Bound::Included(start),
            end.map_or(Bound::Unbounded, Bound::Excluded),
        );
        let mut keys = Vec::new();
        for (key, (_keyinfo, value)) in self.entries.range_mut::<[u8], _>(range) {
            self.entries_size -= key.len() + value.len();
            keys.push(key.clone());
        }
        for key in keys {
            self.entries.remove(&key);
        }
        size_before - self.entries_size
    }*/
}

#[cfg(test)]
mod tests {
    use super::{CowBytes, LeafNode, Size};
    use crate::{
        arbitrary::GenExt,
        data_management::HasStoragePreference,
        tree::{
            default_message_action::{DefaultMessageAction, DefaultMessageActionMsg},
            imp::packed::PackedMap,
            KeyInfo,
        },
        StoragePreference,
    };
    use quickcheck::{Arbitrary, Gen, TestResult};
    use rand::Rng;

    impl Arbitrary for KeyInfo {
        fn arbitrary(g: &mut Gen) -> Self {
            let sp = g.rng().gen_range(0..=3);
            KeyInfo {
                storage_preference: StoragePreference::from_u8(sp),
            }
        }
    }

    impl Arbitrary for LeafNode {
        fn arbitrary(g: &mut Gen) -> Self {
            let len = g.rng().gen_range(0..20);
            let entries: Vec<_> = (0..len)
                .map(|_| {
                    (
                        CowBytes::arbitrary(g),
                        DefaultMessageActionMsg::arbitrary(g),
                    )
                })
                .map(|(k, v)| (k, v.0))
                .collect();

            let node: LeafNode = entries
                .iter()
                .map(|(k, v)| (&k[..], (KeyInfo::arbitrary(g), v.clone())))
                .collect();
            node.recalculate();
            node
        }

        fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
            let v: Vec<_> = self.data
                .entries
                .clone()
                .into_iter()
                .map(|(k, (info, v))| (k, (info, CowBytes::from(v.to_vec()))))
                .collect();
            Box::new(v.shrink().map(|entries| {
                entries
                    .iter()
                    .map(|(k, (info, v))| (&k[..], (info.clone(), v.clone().into())))
                    .collect()
            }))
        }
    }

    fn serialized_size(leaf_node: &LeafNode) -> usize {
        let mut data = Vec::new();
        PackedMap::pack(leaf_node, &mut data).unwrap();
        data.len()
    }

    #[quickcheck]
    fn check_actual_size(leaf_node: LeafNode) {
        assert_eq!(leaf_node.actual_size(), Some(serialized_size(&leaf_node)));
    }

    #[quickcheck]
    fn check_serialize_size(leaf_node: LeafNode) {
        let size = leaf_node.size();
        let serialized = serialized_size(&leaf_node);
        if size != serialized {
            eprintln!(
                "leaf {:?}, size {}, actual_size {:?}, serialized_size {}",
                leaf_node,
                size,
                leaf_node.actual_size(),
                serialized
            );
            assert_eq!(size, serialized);
        }
    }

    #[quickcheck]
    fn check_serialization(leaf_node: LeafNode) {
        let mut data = Vec::new();
        PackedMap::pack(&leaf_node, &mut data).unwrap();
        let twin = PackedMap::new(data).unpack_leaf();

        assert_eq!(leaf_node, twin);
    }

    #[quickcheck]
    fn check_size_insert(
        mut leaf_node: LeafNode,
        key: CowBytes,
        key_info: KeyInfo,
        msg: DefaultMessageActionMsg,
    ) {
        let size_before = leaf_node.size();
        let size_delta = leaf_node.insert(key, key_info, msg.0, DefaultMessageAction);
        let size_after = leaf_node.size();
        assert_eq!((size_before as isize + size_delta) as usize, size_after);
        assert_eq!({ serialized_size(&leaf_node) }, size_after);
    }

    const MIN_LEAF_SIZE: usize = 512;
    const MAX_LEAF_SIZE: usize = 2048;

    #[quickcheck]
    fn check_size_split(mut leaf_node: LeafNode) -> TestResult {
        let size_before = leaf_node.size();

        if size_before <= MAX_LEAF_SIZE {
            return TestResult::discard();
        }

        let (sibling, _, size_delta, _pivot_key) = leaf_node.split(MIN_LEAF_SIZE, MAX_LEAF_SIZE);
        assert_eq!({ serialized_size(&leaf_node) }, leaf_node.size());
        assert_eq!({ serialized_size(&sibling) }, sibling.size());
        assert_eq!(
            (size_before as isize + size_delta) as usize,
            leaf_node.size()
        );
        assert!(sibling.size() <= MAX_LEAF_SIZE);
        assert!(sibling.size() >= MIN_LEAF_SIZE);
        assert!(leaf_node.size() >= MIN_LEAF_SIZE);
        TestResult::passed()
    }

    #[quickcheck]
    fn check_split_merge_idempotent(mut leaf_node: LeafNode) -> TestResult {
        if leaf_node.size() <= MAX_LEAF_SIZE {
            return TestResult::discard();
        }
        let this = leaf_node.clone();
        let (mut sibling, ..) = leaf_node.split(MIN_LEAF_SIZE, MAX_LEAF_SIZE);
        leaf_node.recalculate();
        leaf_node.merge(&mut sibling);
        assert_eq!(this, leaf_node);
        TestResult::passed()
    }
}
