import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class QuantizedWeight:
    """Quantized weight tensor with metadata."""
    name: str
    data: np.ndarray  # int8 quantized
    shape: Tuple[int, ...]
    scale: float
    zero_point: int
    original_dtype: str
    offset: int = 0
    
def quantize_symmetric(tensor: np.ndarray, bits: int = 8) -> Tuple[np.ndarray, float]:
    """
    Symmetric INT8 quantization.
    
    Args:
        tensor: Float tensor to quantize
        bits: Number of bits (default 8)
        
    Returns:
        (quantized_tensor, scale)
    """
    abs_max = max(abs(tensor.min()), abs(tensor.max()))
    if abs_max < 1e-10:
        return np.zeros_like(tensor, dtype=np.int8), 1.0
    
    qmax = (1 << (bits - 1)) - 1  # 127 for int8
    scale = abs_max / qmax
    quantized = np.clip(np.round(tensor / scale), -qmax - 1, qmax).astype(np.int8)
    return quantized, float(scale)


def quantize_asymmetric(tensor: np.ndarray, bits: int = 8) -> Tuple[np.ndarray, float, int]:
    """
    Asymmetric INT8 quantization.
    
    Args:
        tensor: Float tensor to quantize
        bits: Number of bits (default 8)
        
    Returns:
        (quantized_tensor, scale, zero_point)
    """
    qmin, qmax = 0, (1 << bits) - 1  # 0-255 for uint8
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    
    if tensor_max - tensor_min < 1e-10:
        return np.zeros_like(tensor, dtype=np.uint8), 1.0, 0
    
    scale = (tensor_max - tensor_min) / (qmax - qmin)
    zero_point = int(round(qmin - tensor_min / scale))
    zero_point = np.clip(zero_point, qmin, qmax)
    
    quantized = np.clip(np.round(tensor / scale) + zero_point, qmin, qmax).astype(np.uint8)
    return quantized, float(scale), int(zero_point)
