"""
Configuration for Diffusion Model Training
"""

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DiffusionConfig:
    """Configuration for floor plan diffusion model training"""
    
    # Model configuration
    base_model: str = "runwayml/stable-diffusion-v1-5"
    
    # Training configuration
    resolution: int = 512
    train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-5
    max_train_steps: int = 10000
    
    # LoRA configuration
    use_lora: bool = True
    lora_rank: int = 4
    lora_alpha: int = 4
    
    # Validation and checkpointing
    validation_steps: int = 500
    checkpointing_steps: int = 1000
    
    # Data configuration
    num_workers: int = 4
    
    # Logging
    logging_dir: str = "logs"
    
    def __post_init__(self):
        """Validate configuration"""
        assert self.resolution in [256, 512, 768, 1024], \
            f"Resolution must be one of [256, 512, 768, 1024], got {self.resolution}"
        assert self.train_batch_size > 0, "Batch size must be positive"
        assert self.learning_rate > 0, "Learning rate must be positive"