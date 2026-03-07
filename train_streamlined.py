"""
Streamlined Floor Plan Generation Model Training Script
Optimized for Google Colab and local training
Supports multi-modal datasets (images, prompts, labels, controls)
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import argparse
import json
from dataclasses import dataclass, asdict

# Deep learning libraries
from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
import torchvision.transforms as transforms


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Model
    base_model: str = "runwayml/stable-diffusion-v1-5"
    resolution: int = 512
    
    # Training
    train_batch_size: int = 4
    num_epochs: int = 50
    learning_rate: float = 1e-4
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # LoRA
    use_lora: bool = True
    lora_rank: int = 4
    lora_alpha: int = 4
    
    # Checkpointing
    save_every_n_epochs: int = 10
    generate_samples_every_n_epochs: int = 10
    
    # Paths
    dataset_path: str = "./floor_plan_dataset"
    output_dir: str = "./trained_model"
    
    # Dataset options
    use_labels: bool = False
    use_controls: bool = False
    
    def save(self, path: str):
        """Save config to JSON"""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)


class FloorPlanDataset(Dataset):
    """
    Dataset loader for floor plan training
    Supports: images + prompts + optional (labels, controls)
    """
    
    def __init__(
        self,
        data_dir: str,
        resolution: int = 512,
        use_labels: bool = False,
        use_controls: bool = False,
        augment: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.use_labels = use_labels
        self.use_controls = use_controls
        
        # Directories
        self.images_dir = self.data_dir / 'images'
        self.prompts_dir = self.data_dir / 'prompts'
        self.labels_dir = self.data_dir / 'labels' if use_labels else None
        self.controls_dir = self.data_dir / 'controls' if use_controls else None
        
        # Validate directories exist
        if not self.images_dir.exists():
            raise ValueError(f"Images directory not found: {self.images_dir}")
        if not self.prompts_dir.exists():
            raise ValueError(f"Prompts directory not found: {self.prompts_dir}")
        
        # Load all samples
        self.samples = []
        
        for img_path in sorted(self.images_dir.glob('*.png')):
            sample_id = img_path.stem
            prompt_path = self.prompts_dir / f"{sample_id}.txt"
            
            if not prompt_path.exists():
                print(f"Warning: Skipping {sample_id} - no prompt file")
                continue
            
            sample = {
                'image_path': str(img_path),
                'prompt_path': str(prompt_path),
                'sample_id': sample_id
            }
            
            # Add optional paths
            if self.use_labels and self.labels_dir:
                label_path = self.labels_dir / f"{sample_id}.png"
                if label_path.exists():
                    sample['label_path'] = str(label_path)
            
            if self.use_controls and self.controls_dir:
                control_path = self.controls_dir / f"{sample_id}.png"
                if control_path.exists():
                    sample['control_path'] = str(control_path)
            
            self.samples.append(sample)
        
        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found in {data_dir}")
        
        print(f"✅ Loaded {len(self.samples)} samples")
        
        # Define transforms
        transform_list = [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
        ]
        
        if augment:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
            ])
        
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        self.transform = transforms.Compose(transform_list)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        image = self.transform(image)
        
        # Load prompt
        with open(sample['prompt_path'], 'r', encoding='utf-8') as f:
            prompt = f.read().strip()
        
        result = {
            'pixel_values': image,
            'caption': prompt,
            'sample_id': sample['sample_id']
        }
        
        # Load optional data
        if 'label_path' in sample:
            label = Image.open(sample['label_path']).convert('L')
            label = self.transform(label)
            result['label'] = label
        
        if 'control_path' in sample:
            control = Image.open(sample['control_path']).convert('RGB')
            control = self.transform(control)
            result['control'] = control
        
        return result


class FloorPlanTrainer:
    """Trainer for floor plan generation model"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Save config
        config.save(os.path.join(config.output_dir, 'training_config.json'))
        
        # Setup accelerator
        self.accelerator = Accelerator(
            mixed_precision="fp16",
            gradient_accumulation_steps=config.gradient_accumulation_steps
        )
        
        print(f"🚀 Device: {self.accelerator.device}")
        print(f"📊 Mixed precision: fp16")
    
    def load_models(self):
        """Load Stable Diffusion components"""
        print(f"\n📥 Loading {self.config.base_model}...")
        
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.config.base_model, subfolder="tokenizer"
        )
        
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.config.base_model, subfolder="text_encoder"
        )
        
        self.vae = AutoencoderKL.from_pretrained(
            self.config.base_model, subfolder="vae"
        )
        
        self.unet = UNet2DConditionModel.from_pretrained(
            self.config.base_model, subfolder="unet"
        )
        
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.config.base_model, subfolder="scheduler"
        )
        
        # Freeze VAE and text encoder
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        
        # Apply LoRA
        if self.config.use_lora:
            print("🔧 Applying LoRA...")
            lora_config = LoraConfig(
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
            self.unet = get_peft_model(self.unet, lora_config)
            self.unet.print_trainable_parameters()
        
        print("✅ Models loaded!")
    
    def prepare_dataset(self):
        """Prepare training dataset"""
        print(f"\n📂 Loading dataset from {self.config.dataset_path}...")
        
        self.train_dataset = FloorPlanDataset(
            data_dir=self.config.dataset_path,
            resolution=self.config.resolution,
            use_labels=self.config.use_labels,
            use_controls=self.config.use_controls,
            augment=True
        )
        
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
    
    def train(self):
        """Main training loop"""
        # Setup
        self.load_models()
        self.prepare_dataset()
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        # Prepare for distributed training
        self.unet, optimizer, self.train_dataloader = self.accelerator.prepare(
            self.unet, optimizer, self.train_dataloader
        )
        
        # Move to device
        self.vae.to(self.accelerator.device)
        self.text_encoder.to(self.accelerator.device)
        
        # Training loop
        print("\n" + "="*70)
        print("🏋️  STARTING TRAINING")
        print("="*70)
        print(f"Total epochs: {self.config.num_epochs}")
        print(f"Batch size: {self.config.train_batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Dataset size: {len(self.train_dataset)}")
        print("="*70 + "\n")
        
        global_step = 0
        
        for epoch in range(self.config.num_epochs):
            self.unet.train()
            epoch_loss = 0
            
            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch+1}/{self.config.num_epochs}",
                disable=not self.accelerator.is_local_main_process
            )
            
            for batch in progress_bar:
                with self.accelerator.accumulate(self.unet):
                    # Get batch
                    pixel_values = batch["pixel_values"].to(self.accelerator.device)
                    prompts = batch["caption"]
                    
                    # Encode prompts
                    text_inputs = self.tokenizer(
                        prompts,
                        padding="max_length",
                        max_length=self.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt"
                    )
                    
                    with torch.no_grad():
                        text_embeddings = self.text_encoder(
                            text_inputs.input_ids.to(self.accelerator.device)
                        )[0]
                    
                    # Encode to latent space
                    with torch.no_grad():
                        latents = self.vae.encode(pixel_values).latent_dist.sample()
                        latents = latents * self.vae.config.scaling_factor
                    
                    # Sample noise
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(
                        0,
                        self.noise_scheduler.config.num_train_timesteps,
                        (latents.shape[0],),
                        device=latents.device
                    ).long()
                    
                    # Add noise
                    noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
                    
                    # Predict noise
                    noise_pred = self.unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=text_embeddings
                    ).sample
                    
                    # Loss
                    loss = F.mse_loss(noise_pred, noise, reduction="mean")
                    
                    # Backprop
                    self.accelerator.backward(loss)
                    
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.unet.parameters(),
                            self.config.max_grad_norm
                        )
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # Update metrics
                    epoch_loss += loss.detach().item()
                    global_step += 1
                    
                    progress_bar.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "avg": f"{epoch_loss / (progress_bar.n + 1):.4f}"
                    })
            
            # Epoch summary
            avg_loss = epoch_loss / len(self.train_dataloader)
            print(f"📊 Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self.save_checkpoint(epoch + 1)
            
            # Generate samples
            if (epoch + 1) % self.config.generate_samples_every_n_epochs == 0:
                if self.accelerator.is_main_process:
                    self.generate_samples(epoch + 1)
        
        # Final save
        print("\n💾 Saving final model...")
        self.save_final_model()
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print("="*70)
        print(f"Model saved to: {self.config.output_dir}/final_model")
        print("="*70 + "\n")
    
    def save_checkpoint(self, epoch):
        """Save checkpoint"""
        save_dir = Path(self.config.output_dir) / f"checkpoint-epoch-{epoch}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        unet = self.accelerator.unwrap_model(self.unet)
        
        if self.config.use_lora:
            unet.save_pretrained(save_dir / "unet_lora")
        else:
            unet.save_pretrained(save_dir / "unet")
        
        print(f"💾 Checkpoint saved: {save_dir}")
    
    def save_final_model(self):
        """Save complete pipeline"""
        save_dir = Path(self.config.output_dir) / "final_model"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        pipeline = StableDiffusionPipeline(
            vae=self.accelerator.unwrap_model(self.vae),
            text_encoder=self.accelerator.unwrap_model(self.text_encoder),
            tokenizer=self.tokenizer,
            unet=self.accelerator.unwrap_model(self.unet),
            scheduler=self.noise_scheduler,
            safety_checker=None,
            feature_extractor=None
        )
        
        pipeline.save_pretrained(save_dir)
        print(f"✅ Final model: {save_dir}")
    
    @torch.no_grad()
    def generate_samples(self, epoch):
        """Generate validation samples"""
        print(f"\n🖼️  Generating samples (epoch {epoch})...")
        
        pipeline = StableDiffusionPipeline(
            vae=self.accelerator.unwrap_model(self.vae),
            text_encoder=self.accelerator.unwrap_model(self.text_encoder),
            tokenizer=self.tokenizer,
            unet=self.accelerator.unwrap_model(self.unet),
            scheduler=self.noise_scheduler,
            safety_checker=None,
            feature_extractor=None
        )
        pipeline = pipeline.to(self.accelerator.device)
        pipeline.set_progress_bar_config(disable=True)
        
        save_dir = Path(self.config.output_dir) / "samples" / f"epoch_{epoch}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        prompts = [
            "CAD architectural floor plan, 2BHK apartment, living room, kitchen, 2 bedrooms, bathroom, top view, black and white",
            "Professional blueprint floor plan, 3 bedroom house with open kitchen, modern layout, technical drawing",
            "Floor plan CAD drawing, 1BHK apartment, efficient layout, clean lines"
        ]
        
        for i, prompt in enumerate(prompts):
            image = pipeline(
                prompt,
                num_inference_steps=50,
                guidance_scale=7.5,
                height=self.config.resolution,
                width=self.config.resolution
            ).images[0]
            
            image.save(save_dir / f"sample_{i}.png")
        
        print(f"✅ Samples saved: {save_dir}")
        
        del pipeline
        torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description="Train floor plan generation model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="./trained_model", help="Output directory")
    parser.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--use_labels", action="store_true", help="Use label images")
    parser.add_argument("--use_controls", action="store_true", help="Use control images")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    config = TrainingConfig(
        base_model=args.base_model,
        resolution=args.resolution,
        train_batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        use_labels=args.use_labels,
        use_controls=args.use_controls
    )
    
    trainer = FloorPlanTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()