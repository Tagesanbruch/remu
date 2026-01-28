#!/usr/bin/env python3
"""
Train LeNet-5 on MNIST using PyTorch, then export to ONNX.
Data will be downloaded to ./data directory.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

# Set data directory to current folder
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


class LeNet5(nn.Module):
    """Standard LeNet-5 architecture"""
    def __init__(self):
        super(LeNet5, self).__init__()
        # Conv layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)  # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)             # 14x14 -> 10x10
        
        # FC layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Conv1 -> ReLU -> Pool: 28x28 -> 14x14
        x = self.pool(self.relu(self.conv1(x)))
        # Conv2 -> ReLU -> Pool: 14x14 -> 5x5
        x = self.pool(self.relu(self.conv2(x)))
        # Flatten
        x = x.view(-1, 16 * 5 * 5)
        # FC layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST
    print(f"Downloading MNIST to {DATA_DIR}...")
    train_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=2)
    
    # Model
    model = LeNet5().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 100 == 99:
                print(f'Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {running_loss/100:.4f}')
                running_loss = 0.0
        
        # Test
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        print(f'Epoch {epoch+1} Accuracy: {100*correct/total:.2f}%')
    
    print(f"\nFinal accuracy: {100*correct/total:.2f}%")
    
    # Save PyTorch model
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'lenet5.pth'))
    
    # Export to ONNX
    model.eval()
    model.to('cpu')
    dummy_input = torch.randn(1, 1, 28, 28)
    onnx_path = os.path.join(OUTPUT_DIR, 'lenet5.onnx')
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Exported ONNX model to {onnx_path}")
    
    # Extract and save sample test images for REMU testing
    save_test_samples(test_dataset, model)
    
    return model


def save_test_samples(test_dataset, model):
    """Save some test samples as C header for REMU testing"""
    print("\nExtracting test samples...")
    
    model.eval()
    model.to('cpu')
    
    # Get one sample of each digit 0-9
    samples = {}
    for img, label in test_dataset:
        if label not in samples:
            samples[label] = img
        if len(samples) == 10:
            break
    
    # Generate C header with test images
    header_lines = [
        "// Auto-generated test images from MNIST",
        "// Each image is 28x28, normalized and quantized to int8",
        "#ifndef __TEST_IMAGES_H__",
        "#define __TEST_IMAGES_H__",
        "",
        "#include <stdint.h>",
        "",
        "#define IMG_H 28",
        "#define IMG_W 28",
        "#define IMG_SIZE (IMG_H * IMG_W)",
        "#define NUM_CLASSES 10",
        "",
    ]
    
    # Quantization parameters (for int8)
    # Input normalized: mean=0.1307, std=0.3081
    # We'll store raw uint8 pixels (0-255) and normalize in code
    
    for digit in range(10):
        img = samples[digit]  # Shape: (1, 28, 28)
        
        # Convert to uint8 (0-255)
        # Denormalize: x * std + mean, then scale to 0-255
        img_float = img.numpy().squeeze()  # (28, 28)
        img_denorm = img_float * 0.3081 + 0.1307
        img_uint8 = np.clip(img_denorm * 255, 0, 255).astype(np.uint8)
        
        # Also save quantized int8 version (pre-normalized)
        # For NPU: scale input to int8 range
        img_int8 = np.clip(img_float * 127, -128, 127).astype(np.int8)
        
        header_lines.append(f"// Digit {digit}")
        header_lines.append(f"static const int8_t test_img_{digit}[IMG_SIZE] = {{")
        
        for row in range(28):
            row_data = ", ".join(f"{img_int8[row, col]:4d}" for col in range(28))
            header_lines.append(f"    {row_data},")
        
        header_lines.append("};")
        header_lines.append("")
        
        # Run inference to get expected output
        with torch.no_grad():
            output = model(img.unsqueeze(0))
            _, predicted = torch.max(output, 1)
            print(f"  Digit {digit}: predicted={predicted.item()}, "
                  f"correct={predicted.item() == digit}")
    
    # Array of all test images
    header_lines.append("// Array of test image pointers")
    header_lines.append("static const int8_t* test_images[NUM_CLASSES] = {")
    for digit in range(10):
        header_lines.append(f"    test_img_{digit},")
    header_lines.append("};")
    header_lines.append("")
    
    # Expected labels
    header_lines.append("// Expected labels")
    header_lines.append("static const int expected_labels[NUM_CLASSES] = {")
    header_lines.append("    0, 1, 2, 3, 4, 5, 6, 7, 8, 9")
    header_lines.append("};")
    header_lines.append("")
    header_lines.append("#endif // __TEST_IMAGES_H__")
    
    # Save header
    header_path = os.path.join(OUTPUT_DIR, 'test_images.h')
    with open(header_path, 'w') as f:
        f.write('\n'.join(header_lines))
    print(f"Saved test images to {header_path}")


if __name__ == "__main__":
    train()
