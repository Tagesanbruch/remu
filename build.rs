use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=.config");
    println!("cargo:rerun-if-changed=Kconfig");
    
    // Run config generator
    let status = Command::new("python3")
        .arg("scripts/gen_config.py")
        .status()
        .expect("Failed to run config generator");
    
    if !status.success() {
        panic!("Config generation failed");
    }
}
