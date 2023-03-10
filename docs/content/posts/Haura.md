---
title: "What is Haura"
date: 2022-12-08T12:16:16+01:00
draft: false
sidebar: false
pager: false
tags:
  - "Project Details"
---

>_Extract from [[Felix Wiedemann](https://wr.informatik.uni-hamburg.de/_media/research:theses:felix_wiedemann_modern_storage_stack_with_key_value_store_interface_and_snapshots_based_on_copy_on_write_b%CE%B5_trees.pdf)]_
>
> This storage stack uses multiple layers to reduce the complexity of the design and to simplify the reasoning for each component (see Figure 3.1). Many concepts shown in this chapter are heavily based on ZFS which will be presented in detail later on in Section 4.2. 
>
> At the bottom of the storage stack, we have the Vdev Layer (short for virtual device) and the Storage Pool Layer (SPL). The vdevs are responsible for reading and writing data to the disks. A vdev can either be a single disk, a mirror of multiple disks, or a group of disks with parity data – similar to RAID5 and raidz1 in ZFS. The SPL is responsible for striping the data over the vdevs. Like for ZFS, the lower layer already makes use of checksums to provide data integrity and advanced data recovery options. The SPL provides a block device like interface but with checksums. 
>
> On top of this interface is the Data Management Layer (DML). The DML handles generic objects which are identified by an object reference. The DML caches often accessed objects and tracks modifications of objects. Modified objects will be written back when needed. Write back includes compression and allocation of disk space. The DML employs the redirect-on-write technique. For allocation, the DML calls an allocation handler which provides access to the allocation bitmaps. 
> 
> The Tree Layer manages multiple Bε-Trees. Each Bε-Tree node is tracked as a DML object. Each Bε-Tree has a message based key-value interface with byte sequences as keys and values. The length of keys and values and the size of messages is limited to reasonable sizes: A key should be at most 1024 bytes long. Values and messages should be at most 128 KiB of data. Each message may contain arbitrary data and execute arbitrary code on application to data. Additional to querying data for a key and inserting a message for a key, the tree also provides a range query and a range delete operation which are more efficient for large key ranges. 
>
> The top-most Database Layer is built on top of the Bε-Trees. This layer manages multiple data sets which can be individually snapshotted. Each data set provides a key-value store interface and is represented by a Bε-Tree. Likewise, snapshots also provide a (read only) key-value store interface and are represented by a snapshot of the data set’s Bε-Tree. 
>
> There is also a root Bε-Tree in which the allocation bitmaps and the information about the data sets and the snapshots are stored. There are two key concepts of this storage stack: First, we will use the Bε-Tree as the sole index data structure to again reduce the complexity of this design. Having to manage only one data structure greatly reduces the amount of code in the implementation. Second, by applying the path-copying technique [Dri+86], we have copy-on-write Bε-Trees which enable efficient snapshots. 
>
> In contrast to the two presented storage stacks later on, this storage stack does not provide a POSIX compatible filesystem interface. A POSIX filesystem has many operations which must be supported and strict semantics regarding atomicity of concurrent reads and writes. Additionally, userspace filesystem implementations are quite slow due to the context switching and mode switching overhead as the kernel has to switch to our userspace filesystem process for each syscall. See [VTZ17] for a detailed performance analysis of FUSE, the userspace filesystem framework used on Linux. Due to the complexity and the performance issues, this storage stack will instead export a key-value store like interface which is still very similar to a filesystem but with a much smaller and cleaner interface. 
>



Relevant Literature:
1. [Modern Storage Stack with Key-Value Store Interface and Snapshots Based on Copy-On-Write Be-Trees](https://wr.informatik.uni-hamburg.de/_media/research:theses:felix_wiedemann_modern_storage_stack_with_key_value_store_interface_and_snapshots_based_on_copy_on_write_b%CE%B5_trees.pdf)
2. [Design and Implementation of an Object Store with Tiered Storage](https://parcio.ovgu.de/parcio_media/Theses/2021/Design+and+Implementation+of+an+Object+Store+with+Tiered+Storage+%28Till+H%C3%B6ppner%29.pdf)
3. [Data Migration Policies in a Copy-onWrite Tiered Storage Stack - Conception and Implementation](https://parcio.ovgu.de/Thesis+Topics/2022/Data+Migration+Policies+in+a+Copy_on_Write+Tiered+Storage+Stack+_+Conception+and+Implementation-p-274.html)
4. [Supplementing a Modern Storage Engine with Non-volatile Memory](https://wwwiti.cs.uni-magdeburg.de/iti_db/publikationen/ps/auto/thesisKarim23.pdf)

[Main page](/about#haura)


