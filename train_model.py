"""
Training Script for Floor Plan Generation Model
Fine-tunes Stable Diffusion 1.5 with LoRA for CAD-style floor plan generation
Uses HuggingFace Diffusers library (completely free and open-source)
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Install required packages (run once):
# pip install diffusers transformers accelerate peft torchvision pillow

from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator


class FloorPlanDataset(Dataset):
    """Dataset loader for floor plan training data"""
    
    def __init__(self, data_dir: str, image_size: int = 512):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        
        # Load all samples
        self.samples = []
        images_dir = self.data_dir / 'images'
        prompts_dir = self.data_dir / 'prompts'
        
        for img_path in sorted(images_dir.glob('*.png')):
            sample_id = img_path.stem
            prompt_path = prompts_dir / f"{sample_id}.txt"
            
            if prompt_path.exists():
                with open(prompt_path, 'r') as f:
                    prompt = f.read().strip()
                
                self.samples.append({
                    'image_path': str(img_path),
                    'prompt': prompt,
                    'sample_id': sample_id
                })
        
        print(f"Loaded {len(self.samples)} training samples from {data_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        
        # Convert to tensor and normalize to [-1, 1]
        image = torch.from_numpy(np.array(image)).float() / 127.5 - 1.0
        image = image.permute(2, 0, 1)  # HWC -> CHW
        
        return {
            'pixel_values': image,
            'prompt': sample['prompt'],
            'sample_id': sample['sample_id']
        }


class FloorPlanTrainer:
    """Trainer for floor plan generation model"""
    
    def __init__(
        self,
        model_name: str = "runwayml/stable-diffusion-v1-5",
        data_dir: str = "./floor_plan_dataset",
        output_dir: str = "./trained_model",
        image_size: int = 512,
        batch_size: int = 4,
        num_epochs: int = 50,
        learning_rate: float = 1e-4,
        use_lora: bool = True,
        lora_rank: int = 4
    ):
        """
        Initialize trainer
        
        Args:
            model_name: Base Stable Diffusion model from HuggingFace
            data_dir: Directory containing training dataset
            output_dir: Directory to save trained model
            image_size: Image resolution (512 recommended for SD 1.5)
            batch_size: Training batch size (reduce if OOM)
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            use_lora: Use LoRA for efficient fine-tuning (recommended)
            lora_rank: LoRA rank (lower = fewer parameters, faster)
        """
        
        self.model_name = model_name
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize accelerator for distributed training / mixed precision
        self.accelerator = Accelerator(
            mixed_precision='fp16',  # Use fp16 for faster training
            gradient_accumulation_steps=1
        )
        
        print(f"Training on device: {self.accelerator.device}")
        
    def setup_model(self):
        """Load and setup Stable Diffusion model"""
        
        print("Loading Stable Diffusion model...")
        
        # Load tokenizer and text encoder
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.model_name, 
            subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.model_name, 
            subfolder="text_encoder"
        )
        
        # Load UNet
        self.unet = UNet2DConditionModel.from_pretrained(
            self.model_name, 
            subfolder="unet"
        )
        
        # Load noise scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.model_name, 
            subfolder="scheduler"
        )
        
        # Freeze text encoder (we only train UNet)
        self.text_encoder.requires_grad_(False)
        
        # Apply LoRA to UNet for efficient fine-tuning
        if self.use_lora:
            print(f"Applying LoRA with rank {self.lora_rank}...")
            lora_config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_rank,
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
                lora_dropout=0.1,
            )
            self.unet = get_peft_model(self.unet, lora_config)
            self.unet.print_trainable_parameters()
        
        # Move models to device
        self.text_encoder.to(self.accelerator.device)
        self.unet.to(self.accelerator.device)
        
        print("Model setup complete!")
    
    def setup_data(self):
        """Setup dataset and dataloader"""
        
        print("Setting up dataset...")
        
        self.dataset = FloorPlanDataset(
            data_dir=self.data_dir,
            image_size=self.image_size
        )
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        print(f"Dataset ready: {len(self.dataset)} samples, {len(self.dataloader)} batches")
    
    def train(self):
        """Main training loop"""
        
        # Setup model and data
        self.setup_model()
        self.setup_data()
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        # Prepare for distributed training
        self.unet, optimizer, self.dataloader = self.accelerator.prepare(
            self.unet, optimizer, self.dataloader
        )
        
        # Training loop
        global_step = 0
        
        print(f"\nStarting training for {self.num_epochs} epochs...")
        
        for epoch in range(self.num_epochs):
            self.unet.train()
            epoch_loss = 0.0
            
            progress_bar = tqdm(
                self.dataloader, 
                desc=f"Epoch {epoch+1}/{self.num_epochs}",
                disable=not self.accelerator.is_local_main_process
            )
            
            for batch_idx, batch in enumerate(progress_bar):
                with self.accelerator.accumulate(self.unet):
                    # Get inputs
                    pixel_values = batch['pixel_values']
                    prompts = batch['prompt']
                    
                    # Encode prompts to text embeddings
                    text_inputs = self.tokenizer(
                        prompts,
                        padding="max_length",
                        max_length=self.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt"
                    )
                    text_embeddings = self.text_encoder(
                        text_inputs.input_ids.to(self.accelerator.device)
                    )[0]
                    
                    # Sample noise
                    noise = torch.randn_like(pixel_values)
                    
                    # Sample random timesteps
                    timesteps = torch.randint(
                        0, 
                        self.noise_scheduler.config.num_train_timesteps,
                        (pixel_values.shape[0],),
                        device=pixel_values.device
                    ).long()
                    
                    # Add noise to images
                    noisy_images = self.noise_scheduler.add_noise(
                        pixel_values, noise, timesteps
                    )
                    
                    # Predict noise
                    noise_pred = self.unet(
                        noisy_images,
                        timesteps,
                        encoder_hidden_states=text_embeddings
                    ).sample
                    
                    # Calculate loss
                    loss = F.mse_loss(noise_pred, noise, reduction='mean')
                    
                    # Backprop
                    self.accelerator.backward(loss)
                    
                    # Clip gradients
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.unet.parameters(), 1.0)
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # Update progress
                    epoch_loss += loss.item()
                    global_step += 1
                    
                    progress_bar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'avg_loss': f"{epoch_loss / (batch_idx + 1):.4f}"
                    })
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0 or epoch == self.num_epochs - 1:
                self.save_checkpoint(epoch + 1)
        
        print("\nTraining complete!")
    
    def save_checkpoint(self, epoch: int):
        """Save model checkpoint"""
        
        checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{epoch}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save LoRA weights
        if self.use_lora:
            self.accelerator.unwrap_model(self.unet).save_pretrained(checkpoint_dir)
        else:
            # Save full UNet
            self.accelerator.unwrap_model(self.unet).save_pretrained(
                checkpoint_dir,
                safe_serialization=True
            )
        
        print(f"Checkpoint saved: {checkpoint_dir}")
    
    def save_final_model(self):
        """Save final trained model"""
        
        final_dir = os.path.join(self.output_dir, "final_model")
        os.makedirs(final_dir, exist_ok=True)
        
        # Save entire pipeline
        pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_name,
            unet=self.accelerator.unwrap_model(self.unet),
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            safety_checker=None  # Disable safety checker for floor plans
        )
        
        pipeline.save_pretrained(final_dir)
        
        print(f"Final model saved: {final_dir}")


def train_floor_plan_model(
    data_dir: str = "./floor_plan_dataset",
    output_dir: str = "./trained_model",
    epochs: int = 50,
    batch_size: int = 4,
    learning_rate: float = 1e-4
):
    """
    Main training function
    
    Usage:
        python train_model.py
    
    Or in Python:
        from train_model import train_floor_plan_model
        train_floor_plan_model(
            data_dir="./floor_plan_dataset",
            epochs=50
        )
    """
    
    trainer = FloorPlanTrainer(
        data_dir=data_dir,
        output_dir=output_dir,
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        use_lora=True,
        lora_rank=4
    )
    
    trainer.train()
    trainer.save_final_model()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train floor plan generation model")
    parser.add_argument("--data_dir", type=str, default="./floor_plan_dataset",
                       help="Directory containing training data")
    parser.add_argument("--output_dir", type=str, default="./trained_model",
                       help="Directory to save trained model")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    
    args = parser.parse_args()
    
    train_floor_plan_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
