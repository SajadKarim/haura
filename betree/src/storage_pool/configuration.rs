//! Storage Pool configuration.
use bindgen_libpmem::libpmem;
use crate::vdev::{self, Dev, Leaf};
use itertools::Itertools;
use libc;
use serde::{Deserialize, Serialize};
use std::{
    fmt, fmt::Write, fs::OpenOptions, io, iter::FromIterator, os::unix::io::AsRawFd, path::PathBuf,
    slice,
};

/// Configuration of a single storage class.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TierConfiguration {
    pub top_level_vdevs: Vec<Vdev>,
}

/// Configuration for the storage pool unit.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct StoragePoolConfiguration {
    /// Storage classes to make use of
    pub tiers: Vec<TierConfiguration>,
    /// The queue length is the product of this factor and the number of disks involved
    pub queue_depth_factor: u32,
    /// Upper limit for concurrent IO operations
    pub thread_pool_size: Option<u32>,
    /// Whether to pin each worker thread to a CPU core
    pub thread_pool_pinned: bool,
}

impl Default for StoragePoolConfiguration {
    fn default() -> Self {
        Self {
            tiers: Vec::new(),
            queue_depth_factor: 20,
            thread_pool_size: None,
            thread_pool_pinned: false,
        }
    }
}

/// Represents a top-level vdev.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged, deny_unknown_fields, rename_all = "lowercase")]
pub enum Vdev {
    /// This vdev is a leaf vdev.
    Leaf(LeafVdev),
    /// This vdev is a mirror vdev.
    Mirror {
        /// Constituent vdevs of this mirror
        mirror: Vec<LeafVdev>,
    },
    /// Parity1 aka RAID5.
    Parity1 {
        /// Constituent vdevs of this parity1 aggregation
        parity1: Vec<LeafVdev>,
    },
}

/// Represents a leaf vdev.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged, deny_unknown_fields, rename_all = "lowercase")]
pub enum LeafVdev {
    /// Backed by a file or disk.
    File(PathBuf),
    /// Customisable file vdev.
    FileWithOpts {
        /// Path to file or block device
        path: PathBuf,
        /// Whether to use direct IO for this file. Defaults to true.
        direct: Option<bool>,
    },
    /// Backed by a memory buffer.
    Memory {
        /// Size of memory vdev in bytes.
        mem: usize,
    },
    PMEMFile(PathBuf),
}

error_chain! {
    errors {
        #[allow(missing_docs)]
        InvalidKeyword
    }
}

impl TierConfiguration {
    /// Returns a new `StorageConfiguration` based on the given top-level vdevs.
    pub fn new(top_level_vdevs: Vec<Vdev>) -> Self {
        TierConfiguration { top_level_vdevs }
    }

    /// Opens file and devices and constructs a `Vec<Vdev>`.
    pub(crate) fn build(&self) -> io::Result<Vec<Dev>> {
        self.top_level_vdevs
            .iter()
            .enumerate()
            .map(|(n, v)| v.build(n))
            .collect()
    }

    /// Parses the configuration from a ZFS-like representation.
    ///
    /// This representation is a sequence of top-level vdevs.
    /// The keywords `mirror` and `parity1` signal
    /// that all immediate following devices shall be grouped in such a vdev.
    ///
    /// # Example
    /// `/dev/sda mirror /dev/sdb /dev/sdc parity1 /dev/sdd /dev/sde /dev/sdf`
    /// results in three top-level vdevs: `/dev/sda`, a mirror vdev, and a
    /// parity1 vdev. The mirror vdev contains `/dev/sdb` and `/dev/sdc`.
    /// The parity1 vdev contains `/dev/sdd`, `/dev/sde`, and `/dev/sdf`.
    pub fn parse_zfs_like<I, S>(iter: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut iter = iter.into_iter().peekable();
        let mut v = Vec::new();
        while let Some(s) = iter.next() {
            let s = s.as_ref();
            if is_path(s) {
                v.push(Vdev::Leaf(LeafVdev::from(s)));
                continue;
            }
            let f = match s {
                "mirror" => |leaves| Vdev::Mirror { mirror: leaves },
                "parity" | "parity1" => |leaves| Vdev::Parity1 { parity1: leaves },
                _ => bail!(ErrorKind::InvalidKeyword),
            };
            let leaves = iter
                .peeking_take_while(is_path)
                .map(|s| LeafVdev::from(s.as_ref()))
                .collect();
            v.push(f(leaves));
        }
        Ok(TierConfiguration { top_level_vdevs: v })
    }

    /// Returns the configuration in a ZFS-like string representation.
    ///
    /// See `parse_zfs_like` for more information.
    pub fn zfs_like(&self) -> String {
        let mut s = String::new();
        for vdev in &self.top_level_vdevs {
            let (keyword, leaves) = match *vdev {
                Vdev::Leaf(ref leaf) => ("", slice::from_ref(leaf)),
                Vdev::Mirror { mirror: ref leaves } => ("mirror ", &leaves[..]),
                Vdev::Parity1 {
                    parity1: ref leaves,
                } => ("parity1 ", &leaves[..]),
            };
            s.push_str(keyword);
            for leaf in leaves {
                match leaf {
                    LeafVdev::File(path) => write!(s, "{} ", path.display()).unwrap(),
                    LeafVdev::FileWithOpts { path, direct } => {
                        write!(s, "{} (direct: {:?}) ", path.display(), direct).unwrap()
                    }
                    LeafVdev::Memory { mem } => write!(s, "memory({}) ", mem).unwrap(),
                    LeafVdev::PMEMFile(path) => write!(s, "{} ", path.display()).unwrap(),
                }
            }
        }
        s.pop();
        s
    }
}

fn is_path<S: AsRef<str> + ?Sized>(s: &S) -> bool {
    match s.as_ref().chars().next() {
        Some('.') | Some('/') => true,
        _ => false,
    }
}

impl FromIterator<Vdev> for TierConfiguration {
    fn from_iter<T: IntoIterator<Item = Vdev>>(iter: T) -> Self {
        TierConfiguration {
            top_level_vdevs: iter.into_iter().collect(),
        }
    }
}

impl Vdev {
    /// Opens file and devices and constructs a `Vdev`.
    fn build(&self, n: usize) -> io::Result<Dev> {
        match *self {
            Vdev::Mirror { mirror: ref vec } => {
                let leaves: io::Result<Vec<Leaf>> = vec.iter().map(LeafVdev::build).collect();
                let leaves: Box<[Leaf]> = leaves?.into_boxed_slice();
                Ok(Dev::Mirror(vdev::Mirror::new(
                    leaves,
                    format!("mirror-{}", n),
                )))
            }
            Vdev::Parity1 { parity1: ref vec } => {
                let leaves: io::Result<Vec<_>> = vec.iter().map(LeafVdev::build).collect();
                let leaves = leaves?.into_boxed_slice();
                Ok(Dev::Parity1(vdev::Parity1::new(
                    leaves,
                    format!("parity-{}", n),
                )))
            }
            Vdev::Leaf(ref leaf) => leaf.build().map(Dev::Leaf),
        }
    }
}

impl LeafVdev {
    fn build(&self) -> io::Result<Leaf> {
        use std::os::unix::fs::OpenOptionsExt;

        match *self {
            LeafVdev::File(_) | LeafVdev::FileWithOpts { .. } => {
                let (path, direct) = match self {
                    LeafVdev::File(path) => (path, true),
                    LeafVdev::FileWithOpts { path, direct } => (path, direct.unwrap_or(true)),
                    LeafVdev::Memory { .. } => unreachable!(),
                    LeafVdev::PMEMFile(path) => unreachable!(),
                };

                let mut file = OpenOptions::new();
                file.read(true).write(true);
                if direct {
                    file.custom_flags(libc::O_DIRECT);
                }
                let file = file.open(&path)?;

                if unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_RANDOM) }
                    != 0
                {
                    return Err(io::Error::last_os_error());
                }

                Ok(Leaf::File(vdev::File::new(
                    file,
                    path.to_string_lossy().into_owned(),
                )?))
            }
            LeafVdev::Memory { mem } => { Ok(Leaf::Memory(vdev::Memory::new(
                mem,
                format!("memory-{}", mem),
            )?)) }
            LeafVdev::PMEMFile(_) => {
                let (path, direct) = match self {
                    LeafVdev::File(path) => unreachable!(),
                    LeafVdev::FileWithOpts { path, direct } => unreachable!(),
                    LeafVdev::Memory { .. } => unreachable!(),
                    LeafVdev::PMEMFile(path) => (path, true),
                };

                let mut is_pmem : i32 = 0;
                let mut mapped_len : u64 = 0;
                let mut pfile = match path.to_str() {
                    Some(x) => libpmem::pmem_file_open(&x, &mut mapped_len, &mut is_pmem),
                    None => panic!(Error)
                };

                //let pmemfile: bindgen_libpmem::file_handle;

                /*let mut file = OpenOptions::new();
                file.read(true).write(true);
                if direct {
                    file.custom_flags(libc::O_DIRECT);
                }
                let file = file.open(&path)?;

                if unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_RANDOM) }
                    != 0
                {
                    return Err(io::Error::last_os_error());
                }
*/
                Ok(Leaf::PMEMFile(vdev::PMEMFile::new(
                    pfile,
                    //file,
                    path.to_string_lossy().into_owned(),
                )?))
            }

        }
    }
}

impl<'a> From<&'a str> for LeafVdev {
    fn from(s: &'a str) -> Self {
        LeafVdev::File(PathBuf::from(s))
    }
}

impl fmt::Display for TierConfiguration {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for vdev in &self.top_level_vdevs {
            vdev.display(0, f)?;
        }
        Ok(())
    }
}

impl Vdev {
    fn display(&self, indent: usize, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Vdev::Leaf(ref leaf) => leaf.display(indent, f),
            Vdev::Mirror { ref mirror } => {
                writeln!(f, "{:indent$}mirror", "", indent = indent)?;
                for vdev in mirror {
                    vdev.display(indent + 4, f)?;
                }
                Ok(())
            }
            Vdev::Parity1 { ref parity1 } => {
                writeln!(f, "{:indent$}parity1", "", indent = indent)?;
                for vdev in parity1 {
                    vdev.display(indent + 4, f)?;
                }
                Ok(())
            }
        }
    }
}

impl LeafVdev {
    fn display(&self, indent: usize, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            LeafVdev::File(path) => {
                writeln!(f, "{:indent$}{}", "", path.display(), indent = indent)
            }
            LeafVdev::FileWithOpts { path, direct } => {
                writeln!(
                    f,
                    "{:indent$}{} (direct: {:?})",
                    "",
                    path.display(),
                    direct,
                    indent = indent
                )
            }
            LeafVdev::Memory { mem } => {
                writeln!(f, "{:indent$}memory({})", "", mem, indent = indent)
            }
            LeafVdev::PMEMFile(path) => {
                writeln!(f, "{:indent$}{}", "", path.display(), indent = indent)
            }
        }
    }
}
