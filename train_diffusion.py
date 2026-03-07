"""
Diffusion Model Training Script
Fine-tune Stable Diffusion for architectural floor plan generation using LoRA
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm
from pathlib import Path
import argparse
import logging
import math
import sys

# Fix import paths - add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Try to import, with fallback if files don't exist
try:
    from dataset_generator import FloorPlanDataset
    from config import DiffusionConfig
except ImportError as e:
    print(f"Warning: Could not import from src/configs: {e}")
    print("Looking for modules in current directory...")
    
    # Try importing from current directory
    try:
        import dataset_generator
        import config as config_module
        FloorPlanDataset = dataset_generator.FloorPlanDataset
        DiffusionConfig = config_module.DiffusionConfig
    except ImportError:
        print("ERROR: Required modules not found!")
        print("Please ensure these files exist:")
        print("  - src/dataset_preparation.py (or dataset_preparation.py)")
        print("  - configs/config.py (or config.py)")
        sys.exit(1)

logger = get_logger(__name__)

class FloorPlanDiffusionTrainer:
    """
    Trainer for fine-tuning Stable Diffusion on floor plan datasets
    Uses LoRA for parameter-efficient fine-tuning
    """
    
    def __init__(
        self,
        config: DiffusionConfig, # type: ignore
        output_dir: str,
        dataset_path: str,
        resume_from_checkpoint: str = None
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_path = dataset_path
        self.resume_from_checkpoint = resume_from_checkpoint
        
        # Setup accelerator
        accelerator_project_config = ProjectConfiguration(
            project_dir=str(self.output_dir),
            logging_dir=str(self.output_dir / "logs")
        )
        
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            mixed_precision="fp16",
            log_with="tensorboard",
            project_config=accelerator_project_config,
        )
        
        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(self.accelerator.state, main_process_only=False)
        
    def load_models(self):
        """Load all required models"""
        logger.info(f"Loading models from {self.config.base_model}")
        
        # Load scheduler, tokenizer and models
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.config.base_model,
            subfolder="scheduler"
        )
        
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.config.base_model,
            subfolder="tokenizer"
        )
        
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.config.base_model,
            subfolder="text_encoder"
        )
        
        self.vae = AutoencoderKL.from_pretrained(
            self.config.base_model,
            subfolder="vae"
        )
        
        self.unet = UNet2DConditionModel.from_pretrained(
            self.config.base_model,
            subfolder="unet"
        )
        
        # Freeze vae and text_encoder
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        
        # Setup LoRA for UNet if enabled
        if self.config.use_lora:
            logger.info("Setting up LoRA for UNet")
            lora_config = LoraConfig(
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
            self.unet = get_peft_model(self.unet, lora_config)
            self.unet.print_trainable_parameters()
        else:
            # If not using LoRA, make UNet trainable
            self.unet.requires_grad_(True)
        
        logger.info("✓ Models loaded successfully")
    
    def prepare_datasets(self):
        """Prepare training and validation datasets"""
        logger.info(f"Loading datasets from {self.dataset_path}")
        
        self.train_dataset = FloorPlanDataset(
            data_dir=self.dataset_path,
            resolution=self.config.resolution,
            split='train',
            augment=True
        )
        
        self.val_dataset = FloorPlanDataset(
            data_dir=self.dataset_path,
            resolution=self.config.resolution,
            split='val',
            augment=False
        )
        
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        logger.info(f"✓ Loaded {len(self.train_dataset)} training samples")
        logger.info(f"✓ Loaded {len(self.val_dataset)} validation samples")
    
    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler"""
        # Only optimize UNet parameters
        params_to_optimize = self.unet.parameters()
        
        self.optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            eps=1e-08,
        )
        
        # Calculate total training steps
        num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.config.gradient_accumulation_steps
        )
        max_train_steps = self.config.max_train_steps
        
        self.lr_scheduler = get_scheduler(
            "cosine",
            optimizer=self.optimizer,
            num_warmup_steps=500,
            num_training_steps=max_train_steps,
        )
        
        logger.info(f"✓ Setup optimizer with LR: {self.config.learning_rate}")
    
    def encode_prompt(self, prompts):
        """Encode text prompts to embeddings"""
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        text_input_ids = text_inputs.input_ids
        
        with torch.no_grad():
            prompt_embeds = self.text_encoder(
                text_input_ids.to(self.accelerator.device),
            )[0]
        
        return prompt_embeds
    
    def train_step(self, batch):
        """Single training step"""
        pixel_values = batch["pixel_values"].to(self.accelerator.device)
        prompts = batch["caption"]
        
        # Convert images to latent space
        latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        
        # Sample noise
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        
        # Sample random timesteps
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=latents.device
        )
        timesteps = timesteps.long()
        
        # Add noise to latents
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Get text embeddings
        encoder_hidden_states = self.encode_prompt(prompts)
        
        # Predict the noise residual
        model_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states
        ).sample
        
        # Get target
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")
        
        # Calculate loss
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        
        return loss
    
    @torch.no_grad()
    def validation_step(self):
        """Run validation"""
        logger.info("Running validation...")
        
        self.unet.eval()
        val_loss = 0
        num_batches = 0
        
        for batch in self.val_dataloader:
            loss = self.train_step(batch)
            val_loss += loss.item()
            num_batches += 1
            
            if num_batches >= 10:  # Limit validation batches
                break
        
        avg_val_loss = val_loss / num_batches
        self.unet.train()
        
        return avg_val_loss
    
    @torch.no_grad()
    def generate_samples(self, prompts, save_dir):
        """Generate sample images during training"""
        logger.info("Generating sample images...")
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create pipeline
        pipeline = StableDiffusionPipeline(
            vae=self.accelerator.unwrap_model(self.vae),
            text_encoder=self.accelerator.unwrap_model(self.text_encoder),
            tokenizer=self.tokenizer,
            unet=self.accelerator.unwrap_model(self.unet),
            scheduler=self.noise_scheduler,
            safety_checker=None,
            feature_extractor=None,
        )
        pipeline = pipeline.to(self.accelerator.device)
        
        # Generate images
        for i, prompt in enumerate(prompts):
            image = pipeline(
                prompt,
                num_inference_steps=50,
                guidance_scale=7.5,
                height=self.config.resolution,
                width=self.config.resolution,
            ).images[0]
            
            image.save(save_dir / f"sample_{i}.png")
        
        logger.info(f"✓ Saved {len(prompts)} sample images to {save_dir}")
        
        del pipeline
        torch.cuda.empty_cache()
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        # Load everything
        self.load_models()
        self.prepare_datasets()
        self.setup_optimizer()
        
        # Prepare for distributed training
        self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler
        )
        
        # Move models to device
        self.vae.to(self.accelerator.device)
        self.text_encoder.to(self.accelerator.device)
        
        # Training loop
        global_step = 0
        progress_bar = tqdm(
            range(self.config.max_train_steps),
            disable=not self.accelerator.is_local_main_process,
        )
        progress_bar.set_description("Training")
        
        # Sample prompts for validation
        val_prompts = [
            "2BHK apartment floor plan, CAD style, clean lines, professional blueprint",
            "3 bedroom house layout, living room, kitchen, bathrooms, top view architectural drawing",
            "modern apartment floor plan with open kitchen, master bedroom, balcony, CAD drawing"
        ]
        
        for epoch in range(100):  # Large number, will stop at max_steps
            self.unet.train()
            
            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.unet):
                    loss = self.train_step(batch)
                    
                    # Backprop
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.unet.parameters(), 1.0)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                
                # Update progress bar
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    
                    logs = {
                        "loss": loss.detach().item(),
                        "lr": self.lr_scheduler.get_last_lr()[0]
                    }
                    progress_bar.set_postfix(**logs)
                    self.accelerator.log(logs, step=global_step)
                
                # Validation
                if global_step % self.config.validation_steps == 0:
                    val_loss = self.validation_step()
                    logger.info(f"Validation loss: {val_loss:.4f}")
                    self.accelerator.log({"val_loss": val_loss}, step=global_step)
                    
                    # Generate samples
                    if self.accelerator.is_main_process:
                        self.generate_samples(
                            val_prompts,
                            self.output_dir / f"samples/step_{global_step}"
                        )
                
                # Save checkpoint
                if global_step % self.config.checkpointing_steps == 0:
                    if self.accelerator.is_main_process:
                        self.save_checkpoint(global_step)
                
                # Check if we've reached max steps
                if global_step >= self.config.max_train_steps:
                    break
            
            if global_step >= self.config.max_train_steps:
                break
        
        # Final save
        if self.accelerator.is_main_process:
            self.save_checkpoint(global_step, final=True)
        
        self.accelerator.end_training()
        logger.info("✓ Training complete!")
    
    def save_checkpoint(self, step, final=False):
        """Save model checkpoint"""
        save_dir = self.output_dir / "checkpoints" / f"checkpoint-{step}"
        if final:
            save_dir = self.output_dir / "final_model"
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Unwrap model
        unet = self.accelerator.unwrap_model(self.unet)
        
        if self.config.use_lora:
            # Save LoRA weights
            unet.save_pretrained(save_dir / "unet_lora")
        else:
            # Save full UNet
            unet.save_pretrained(save_dir / "unet")
        
        # Save configuration
        import json
        config_dict = {k: v for k, v in self.config.__dict__.items() if not k.startswith('_')}
        config_dict['global_step'] = step
        with open(save_dir / "training_config.json", 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        logger.info(f"✓ Saved checkpoint to {save_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train floor plan diffusion model")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/floor_plan_diffusion",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Image resolution"
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=10000,
        help="Maximum number of training steps"
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Use LoRA for parameter-efficient fine-tuning"
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create config
    config = DiffusionConfig(
        resolution=args.resolution,
        train_batch_size=args.train_batch_size,
        learning_rate=args.learning_rate,
        max_train_steps=args.max_train_steps,
        use_lora=args.use_lora
    )
    
    # Create trainer
    trainer = FloorPlanDiffusionTrainer(
        config=config,
        output_dir=args.output_dir,
        dataset_path=args.dataset_path
    )
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()