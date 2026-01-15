#!/bin/bash
# Build and test script for RustNEMU

set -e

echo "=== RustNEMU Build Script ==="

# Check if Rust is installed
if ! command -v cargo &> /dev/null
then
    echo "Error: Rust is not installed."
    echo "Please install Rust by running:"
    echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    echo "Then restart your terminal and run this script again."
    exit 1
fi

echo "Rust version: $(rustc --version)"
echo "Cargo version: $(cargo --version)"

# Build in release mode
echo ""
echo "Building RustNEMU..."
cargo build --release

echo ""
echo "Build successful!"
echo "Binary location: ./target/release/rustnemu"

# Run built-in test
echo ""
echo "Running built-in test program..."
./target/release/rustnemu --batch

echo ""
echo "=== Build and Test Complete ==="
