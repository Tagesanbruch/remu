
import sys

def modify_compiler():
    file_path = 'tvm-sw/compiler/tvm_compiler.py'
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Check if already modified
        if any("input_shape = tuple(map(int, sys.argv[4].split(',')))" in line for line in lines):
            print("File already modified.")
            return

        new_lines = []
        for line in lines:
            if 'model_name = sys.argv[3] if len(sys.argv) > 3 else "model"' in line:
                new_lines.append(line)
                new_lines.append('    input_shape = tuple(map(int, sys.argv[4].split(","))) if len(sys.argv) > 4 else (1, 3, 224, 224)\n')
            elif 'result = compile_model(model_path, output_dir, model_name)' in line:
                new_lines.append(line.replace('result = compile_model(model_path, output_dir, model_name)', 'result = compile_model(model_path, output_dir, model_name, input_shape)'))
            else:
                new_lines.append(line)
        
        with open(file_path, 'w') as f:
            f.writelines(new_lines)
            print(f"Successfully modified {file_path}")
            
    except Exception as e:
        print(f"Error modifying file: {e}")

if __name__ == "__main__":
    modify_compiler()
