fn main() {
    println!("Testing memory metrics feature...");
    
    // Test if memory metrics is enabled at compile time
    #[cfg(feature = "memory_metrics")]
    {
        println!("SUCCESS: Memory metrics feature is ENABLED!");
        println!("This means the memory tracking code is compiled in.");
    }
    
    #[cfg(not(feature = "memory_metrics"))]
    {
        println!("ERROR: Memory metrics feature is DISABLED!");
        println!("The memory tracking code is not compiled in.");
    }
}