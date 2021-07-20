#![allow(dead_code)]

use betree_storage_stack::{
    compression::CompressionConfiguration,
    database::AccessMode,
    object::ObjectStore,
    storage_pool::{LeafVdev, TierConfiguration, Vdev},
    Database, DatabaseConfiguration, StoragePoolConfiguration, StoragePreference,
};

use std::io::{BufReader, Read};

use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use insta::assert_json_snapshot;

fn test_db(tiers: u32, mb_per_tier: u32) -> Database<DatabaseConfiguration> {
    let tier_size = mb_per_tier as usize * 1024 * 1024;
    let cfg = DatabaseConfiguration {
        storage: StoragePoolConfiguration {
            tiers: (0..tiers)
                .map(|_| TierConfiguration {
                    top_level_vdevs: vec![Vdev::Leaf(LeafVdev::Memory { mem: tier_size })],
                })
                .collect(),
            ..Default::default()
        },
        compression: CompressionConfiguration::None,
        access_mode: AccessMode::AlwaysCreateNew,
        ..Default::default()
    };

    Database::build(cfg).expect("Database initialisation failed")
}

struct TestDriver {
    name: String,
    database: Database<DatabaseConfiguration>,
    object_store: ObjectStore<DatabaseConfiguration>,
    rng: Xoshiro256PlusPlus,
}

impl TestDriver {
    fn setup(test_name: &str, tiers: u32, mb_per_tier: u32) -> TestDriver {
        let mut database = test_db(tiers, mb_per_tier);

        TestDriver {
            name: String::from(test_name),
            rng: Xoshiro256PlusPlus::seed_from_u64(42),
            object_store: database
                .open_named_object_store(b"test", StoragePreference::FASTEST)
                .expect("Failed to open default object store"),
            database,
        }
    }

    fn checkpoint(&mut self, name: &str) {
        self.database.sync().expect("Failed to sync database");
        assert_json_snapshot!(
            format!("{}/{}", self.name, name),
            self.object_store
                .data_tree()
                .tree_dump()
                .expect("Failed to create data tree dump")
        );
    }

    fn insert_random_at(
        &mut self,
        object_name: &[u8],
        pref: StoragePreference,
        block_size: u64,
        times: u64,
        offset: u64,
    ) {
        let (obj, _info) = self
            .object_store
            .open_or_create_object(object_name)
            .expect("Unable to create object");

        let mut buf = vec![0; block_size as usize];
        let pref = pref.or(StoragePreference::FASTEST);

        for i in 0..times {
            self.rng
                .try_fill(&mut buf[..])
                .expect("Couldn't fill with random data");
            obj.write_at_with_pref(&buf, offset + i * block_size as u64, pref)
                .expect("Failed to write buf");
        }
    }

    fn insert_random(
        &mut self,
        object_name: &[u8],
        pref: StoragePreference,
        block_size: u64,
        times: u64,
    ) {
        self.insert_random_at(object_name, pref, block_size, times, 0);
    }

    fn delete(&self, object_name: &[u8]) {
        if let Ok(Some((obj, _info))) = self.object_store.open_object(object_name) {
            obj.delete().expect("Failed to delete object");
        }
    }

    fn read_for_length(&self, object_name: &[u8]) -> u64 {
        let (obj, _info) = self
            .object_store
            .open_or_create_object(object_name)
            .expect("Unable to create object");

        BufReader::new(obj.cursor()).bytes().count() as u64
    }
}

#[test]
fn insert_single() {
    let mut driver = TestDriver::setup("insert single", 1, 256);

    driver.checkpoint("empty tree");
    driver.insert_random(b"foo", StoragePreference::NONE, 8192, 2000);
    driver.checkpoint("inserted foo");

    for _ in 1..=3 {
        driver.insert_random(b"foo", StoragePreference::NONE, 8192, 2000);
        // intentionally same key as above, to assert that tree structures is not changed by
        // object rewrites of the same size
        driver.checkpoint("inserted foo");
    }

    driver.insert_random(b"foo", StoragePreference::NONE, 8192, 4000);
    driver.checkpoint("rewrote foo, but larger");

    driver.delete(b"foo");
    driver.checkpoint("deleted foo");
    driver.insert_random(b"bar", StoragePreference::NONE, 8192, 3000);
    driver.checkpoint("inserted bar");
}

#[test]
fn delete_single() {
    let mut driver = TestDriver::setup("delete single", 1, 1024);

    driver.checkpoint("empty tree");
    driver.insert_random(b"something", StoragePreference::NONE, 8192 * 8, 2000);
    driver.checkpoint("inserted something");
    driver.delete(b"something");
    driver.checkpoint("deleted something");
}

#[test]
fn downgrade() {
    let mut driver = TestDriver::setup("downgrade", 2, 256);

    driver.checkpoint("empty tree");
    driver.insert_random(b"foo", StoragePreference::FASTEST, 8192, 2000);
    driver.checkpoint("fastest pref");
    driver.insert_random(b"foo", StoragePreference::FAST, 8192, 2100);
    driver.checkpoint("fast pref");
}

#[test]
fn sparse() {
    let mut driver = TestDriver::setup("sparse", 1, 1024);
    let chunk_size = 128 * 1024;

    driver.checkpoint("empty tree");
    driver.insert_random_at(
        b"foo",
        StoragePreference::FASTEST,
        chunk_size,
        200,
        300 * chunk_size,
    );
    driver.checkpoint("sparse write 1");
    driver.insert_random_at(
        b"foo",
        StoragePreference::FASTEST,
        chunk_size,
        300,
        800 * chunk_size,
    );
    driver.checkpoint("sparse write 2");

    assert_eq!(driver.read_for_length(b"foo"), (800 + 300) * chunk_size);
}
