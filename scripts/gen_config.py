#!/usr/bin/env python3
"""
Kconfig to Rust configuration generator
Reads .config and generates Rust code with constants
"""

import sys
import os

def parse_config(config_path):
    """Parse .config file and return key-value dict"""
    config = {}
    try:
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    config[key] = value
    except FileNotFoundError:
        print(f"Warning: {config_path} not found, using defaults", file=sys.stderr)
    return config

def value_to_rust(value):
    """Convert config value to Rust literal"""
    # Remove quotes
    value = value.strip('"')
    
    # Try to parse as number
    if value.startswith('0x'):
        return f"0x{value[2:]}"
    elif value.isdigit():
        return value
    elif value == 'y':
        return 'true'
    elif value == 'n':
        return 'false'
    else:
        return f'"{value}"'

def generate_rust_config(config):
    """Generate Rust configuration code"""
    output = []
    output.append("// Auto-generated configuration from Kconfig")
    output.append("// DO NOT EDIT MANUALLY")
    output.append("")
    
    # Generate constants
    for key, value in sorted(config.items()):
        if key.startswith('CONFIG_'):
            # Remove CONFIG_ prefix
            name = key[7:]
            rust_value = value_to_rust(value)
            
            # Skip empty strings
            if rust_value == '""':
                continue
            
            # Type inference
            if '"' in rust_value:
                output.append(f'pub const {name}: &str = {rust_value};')
            elif rust_value in ['true', 'false']:
                output.append(f'pub const {name}: bool = {rust_value};')
            elif rust_value.startswith('0x'):
                output.append(f'pub const {name}: u32 = {rust_value};')
            else:
                output.append(f'pub const {name}: u64 = {rust_value};')
    
    return '\n'.join(output)

def main():
    config_path = '.config'
    output_path = 'src/generated_config.rs'
    
    config = parse_config(config_path)
    rust_code = generate_rust_config(config)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(rust_code)
    
    print(f"Generated {output_path}")

if __name__ == '__main__':
    main()
