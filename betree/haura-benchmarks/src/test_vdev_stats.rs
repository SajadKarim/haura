use betree_storage_stack::{
    database::{Database, DatabaseConfiguration},
    storage_pool::{StoragePoolConfiguration, TierConfiguration, LeafVdev},
    tree::StorageKind,
    PreferredAccessType,
};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing vdev statistics tracking...");
    
    // Create a simple database configuration with memory storage
    let config = DatabaseConfiguration {
        storage: StoragePoolConfiguration {
            tiers: vec![TierConfiguration {
                top_level_vdevs: vec![LeafVdev {
                    path: "/tmp/test_haura_vdev_stats".to_string(),
                    direct: false,
                }],
                preferred_access_type: PreferredAccessType::Unknown,
                storage_kind: StorageKind::Memory,
            }],
            queue_depth_factor: 1,
            thread_pool_size: None,
            thread_pool_pinned: false,
        },
        alloc_strategy: [[Some(0); 4]; 4],
        default_storage_class: 0,
        compression: betree_storage_stack::compression::CompressionConfiguration::None,
        cache_size: 1024 * 1024, // 1MB
        access_mode: betree_storage_stack::database::AccessMode::AlwaysCreateNew,
        sync_interval_ms: None,
        migration_policy: None,
        metrics: None,
    };
    
    // Create database
    let mut db = Database::build(config)?;
    
    // Insert some data to trigger memory access
    let dataset = db.open_or_create_dataset(b"test")?;
    for i in 0..10 {
        let key = format!("key_{}", i);
        let value = format!("value_{}", i);
        dataset.insert(key.as_bytes(), value.as_bytes())?;
    }
    
    // Flush to ensure data is written
    db.flush()?;
    
    // Read some data to trigger memory access
    for i in 0..10 {
        let key = format!("key_{}", i);
        if let Some(value) = dataset.get(key.as_bytes())? {
            println!("Read key_{}: {}", i, String::from_utf8_lossy(&value));
        }
    }
    
    println!("Test completed successfully!");
    println!("If memory metrics are working, you should see MEMORY_METRICS debug output above.");
    
    Ok(())
}