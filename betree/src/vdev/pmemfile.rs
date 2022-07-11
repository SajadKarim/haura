use super::{
    errors::*, AtomicStatistics, Block, Result, ScrubResult, Statistics, Vdev, VdevLeafRead,
    VdevLeafWrite, VdevRead,
};
use crate::{buffer::Buf, checksum::Checksum};
use async_trait::async_trait;
use libc::{c_ulong, ioctl};
use std::{
    fs, io,
    os::unix::{
        fs::{FileExt, FileTypeExt},
        io::AsRawFd,
    },
    sync::atomic::Ordering,
};

/// `LeafVdev` that is backed by a file.
#[derive(Debug)]
pub struct PMEMFile {
    file: fs::File,
    id: String,
    size: Block<u64>,
    stats: AtomicStatistics,
}

impl PMEMFile {
    /// Creates a new `PMEMFile`.
    pub fn new(file: fs::File, id: String) -> io::Result<Self> {
        let file_type = file.metadata()?.file_type();
        let size = if file_type.is_file() {
            Block::from_bytes(file.metadata()?.len())
        } else if file_type.is_block_device() {
            get_block_device_size(&file)?
        } else {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!("Unsupported file type: {:?}", file_type),
            ));
        };
        Ok(PMEMFile {
            file,
            id,
            size,
            stats: Default::default(),
        })
    }
}

#[cfg(target_os = "linux")]
fn get_block_device_size(file: &fs::File) -> io::Result<Block<u64>> {
    const BLKGETSIZE64: c_ulong = 2148012658;
    let mut size: u64 = 0;
    let result = unsafe { ioctl(file.as_raw_fd(), BLKGETSIZE64, &mut size) };

    if result == 0 {
        Ok(Block::from_bytes(size))
    } else {
        Err(io::Error::last_os_error())
    }
}

#[async_trait]
impl VdevRead for PMEMFile {
    async fn read<C: Checksum>(
        &self,
        size: Block<u32>,
        offset: Block<u64>,
        checksum: C,
    ) -> Result<Buf> {
        self.stats.read.fetch_add(size.as_u64(), Ordering::Relaxed);
        let buf = {
            let mut buf = Buf::zeroed(size).into_full_mut();
            if let Err(e) = self.file.read_exact_at(buf.as_mut(), offset.to_bytes()) {
                self.stats
                    .failed_reads
                    .fetch_add(size.as_u64(), Ordering::Relaxed);
                bail!(e)
            }
            buf.into_full_buf()
        };

        match checksum.verify(&buf).map_err(VdevError::from) {
            Ok(()) => Ok(buf),
            Err(e) => {
                self.stats
                    .checksum_errors
                    .fetch_add(size.as_u64(), Ordering::Relaxed);
                Err(e)
            }
        }
    }

    async fn scrub<C: Checksum>(
        &self,
        size: Block<u32>,
        offset: Block<u64>,
        checksum: C,
    ) -> Result<ScrubResult> {
        let data = self.read(size, offset, checksum).await?;
        Ok(ScrubResult {
            data,
            repaired: Block(0),
            faulted: Block(0),
        })
    }

    async fn read_raw(&self, size: Block<u32>, offset: Block<u64>) -> Result<Vec<Buf>> {
        println!("\n.. read_raw for superblock inside PMEMFile vdev.");
        self.stats.read.fetch_add(size.as_u64(), Ordering::Relaxed);
        let mut buf = Buf::zeroed(size).into_full_mut();
        match self.file.read_exact_at(buf.as_mut(), offset.to_bytes()) {
            Ok(()) => Ok(vec![buf.into_full_buf()]),
            Err(e) => {
                self.stats
                    .failed_reads
                    .fetch_add(size.as_u64(), Ordering::Relaxed);
                bail!(e)
            }
        }
    }
}

impl Vdev for PMEMFile {
    fn actual_size(&self, size: Block<u32>) -> Block<u32> {
        size
    }

    fn num_disks(&self) -> usize {
        1
    }

    fn size(&self) -> Block<u64> {
        self.size
    }

    fn effective_free_size(&self, free_size: Block<u64>) -> Block<u64> {
        free_size
    }

    fn id(&self) -> &str {
        &self.id
    }

    fn stats(&self) -> Statistics {
        self.stats.as_stats()
    }

    fn for_each_child(&self, _f: &mut dyn FnMut(&dyn Vdev)) {}
}

#[async_trait]
impl VdevLeafRead for PMEMFile {
    async fn read_raw<T: AsMut<[u8]> + Send>(&self, mut buf: T, offset: Block<u64>) -> Result<T> {
        let size = Block::from_bytes(buf.as_mut().len() as u32);
        self.stats.read.fetch_add(size.as_u64(), Ordering::Relaxed);
        match self.file.read_exact_at(buf.as_mut(), offset.to_bytes()) {
            Ok(()) => Ok(buf),
            Err(e) => {
                self.stats
                    .failed_reads
                    .fetch_add(size.as_u64(), Ordering::Relaxed);
                bail!(e)
            }
        }
    }

    fn checksum_error_occurred(&self, size: Block<u32>) {
        self.stats
            .checksum_errors
            .fetch_add(size.as_u64(), Ordering::Relaxed);
    }
}

static mut cntr : u32 = 0;

#[async_trait]
impl VdevLeafWrite for PMEMFile {
    async fn write_raw<W: AsRef<[u8]> + Send>(
        &self,
        data: W,
        offset: Block<u64>,
        is_repair: bool,
    ) -> Result<()> {

        unsafe { 
            cntr += 1; 

            if cntr == 10 {
                //panic!("...stop here..");
                }
                            }
        println!("\n.... inside write_raw");
        
        
        let block_cnt = Block::from_bytes(data.as_ref().len() as u64).as_u64();
        self.stats.written.fetch_add(block_cnt, Ordering::Relaxed);
        match self
            .file
            .write_all_at(data.as_ref(), offset.to_bytes())
            .map_err(|_| VdevError::Write(self.id.clone()))
        {
            Ok(()) => {
                if is_repair {
                    self.stats.repaired.fetch_add(block_cnt, Ordering::Relaxed);
                }
                Ok(())
            }
            Err(e) => {
                self.stats
                    .failed_writes
                    .fetch_add(block_cnt, Ordering::Relaxed);
                Err(e)
            }
        }
    }
    fn flush(&self) -> Result<()> {
        Ok(self.file.sync_data()?)
    }
}