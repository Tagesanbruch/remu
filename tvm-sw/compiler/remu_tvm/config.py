from dataclasses import dataclass

@dataclass
class NPUConfig:
    """REMU NPU hardware configuration."""
    feature_sram_size: int = 16 * 1024  # 16KB
    weight_sram_size: int = 16 * 1024   # 16KB
    output_sram_size: int = 16 * 1024   # 16KB
    gemm_m_max: int = 256
    gemm_n_max: int = 256
    gemm_k_max: int = 256
    flash_base: int = 0x30000000
    mmio_base: int = 0x21000000
    sram_feature: int = 0x21001000
    sram_weight: int = 0x21005000
    sram_output: int = 0x21009000

NPU_CONFIG = NPUConfig()
