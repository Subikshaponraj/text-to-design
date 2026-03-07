"""
Floor Plan Generation Inference Script
Generates CAD-style floor plans from text descriptions using trained model
"""

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import numpy as np
import cv2
from typing import Optional, List
import os


class FloorPlanGenerator:
    """
    Generate floor plans from text descriptions using fine-tuned Stable Diffusion
    """
    
    def __init__(
        self,
        model_path: str = "./trained_model/final_model",
        device: str = "auto"
    ):
        """
        Initialize generator
        
        Args:
            model_path: Path to trained model directory
            device: Device to run on ('cuda', 'cpu', or 'auto')
        """
        
        self.model_path = model_path
        
        # Auto-detect device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading model from {model_path}...")
        print(f"Using device: {self.device}")
        
        # Load pipeline
        """self.pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None
        )"""
        self.device = "cpu"

        # 1️⃣ Load base model
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float32
        )

        self.pipe.to(self.device)

        # 2️⃣ Load your LoRA weights
        self.pipe.load_lora_weights("./trained_model")
        
        # Use faster scheduler
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        # Move to device
        self.pipe = self.pipe.to(self.device)
        
        # Enable memory optimizations
        if self.device == "cuda":
            self.pipe.enable_attention_slicing()
            # Uncomment if you have limited VRAM:
            # self.pipe.enable_sequential_cpu_offload()
        
        print("Model loaded successfully!")
    
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        num_images: int = 1,
        seed: Optional[int] = None
    ) -> List[Image.Image]:
        """
        Generate floor plan images from text prompt
        
        Args:
            prompt: Text description of desired floor plan
            negative_prompt: Things to avoid in generation
            num_inference_steps: Number of denoising steps (higher = better quality, slower)
            guidance_scale: How closely to follow prompt (7-9 recommended)
            width, height: Output image dimensions
            num_images: Number of variations to generate
            seed: Random seed for reproducibility
        
        Returns:
            List of PIL Images
        """
        
        # Default negative prompt for CAD drawings
        if negative_prompt is None:
            negative_prompt = (
                "blurry, low quality, distorted, photograph, realistic, "
                "3d render, colored, painting, artistic, sketch, unclear lines"
            )
        
        # Set seed for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Generate images
        print(f"Generating {num_images} floor plan(s)...")
        print(f"Prompt: {prompt}")
        
        images = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            num_images_per_prompt=num_images,
            generator=generator
        ).images
        
        return images
    
    def post_process(self, image: Image.Image) -> Image.Image:
        """
        Post-process generated image for cleaner CAD appearance
        - Convert to grayscale
        - Enhance edges
        - Clean up lines
        """
        
        # Convert to numpy
        img = np.array(image)
        
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        
        # Apply adaptive thresholding for clean black/white
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(binary)
        
        # Sharpen edges
        kernel = np.array([[-1,-1,-1], 
                          [-1, 9,-1], 
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        # Convert back to PIL
        processed = Image.fromarray(sharpened)
        
        return processed
    
    def generate_with_variations(
        self,
        prompt: str,
        num_variations: int = 4,
        **kwargs
    ) -> List[Image.Image]:
        """
        Generate multiple variations with different seeds
        """
        
        images = []
        for i in range(num_variations):
            seed = kwargs.get('seed', 42) + i if 'seed' in kwargs else None
            img = self.generate(prompt, num_images=1, seed=seed, **kwargs)[0]
            images.append(img)
        
        return images
    
    def save_images(
        self,
        images: List[Image.Image],
        output_dir: str = "./outputs",
        prefix: str = "floor_plan"
    ):
        """Save generated images to directory"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        saved_paths = []
        for i, img in enumerate(images):
            filename = f"{prefix}_{i:03d}.png"
            filepath = os.path.join(output_dir, filename)
            img.save(filepath)
            saved_paths.append(filepath)
            print(f"Saved: {filepath}")
        
        return saved_paths


def generate_from_text(
    text_description: str,
    model_path: str = "./trained_model/final_model",
    output_dir: str = "./outputs",
    num_variations: int = 1,
    post_process: bool = True
):
    """
    High-level function to generate floor plans from text
    
    Usage:
        from inference import generate_from_text
        
        generate_from_text(
            "2BHK apartment with attached bathroom and open kitchen",
            num_variations=3
        )
    """
    
    # Initialize generator
    generator = FloorPlanGenerator(model_path=model_path)
    
    # Format prompt for CAD-style output
    formatted_prompt = (
        f"CAD architectural floor plan, {text_description}, "
        "top-down view, black and white line drawing, "
        "professional blueprint style, with room labels and dimensions, "
        "clean lines, technical drawing"
    )
    
    # Generate images
    images = generator.generate_with_variations(
        prompt=formatted_prompt,
        num_variations=num_variations,
        num_inference_steps=50,
        guidance_scale=7.5
    )
    
    # Post-process if requested
    if post_process:
        print("Post-processing images...")
        images = [generator.post_process(img) for img in images]
    
    # Save images
    saved_paths = generator.save_images(
        images,
        output_dir=output_dir,
        prefix=text_description.replace(' ', '_')[:30]
    )
    
    print(f"\nGeneration complete! {len(images)} images saved to {output_dir}")
    
    return images, saved_paths


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate floor plans from text")
    parser.add_argument("prompt", type=str, 
                       help="Text description of desired floor plan")
    parser.add_argument("--model_path", type=str, 
                       default="./trained_model/final_model",
                       help="Path to trained model")
    parser.add_argument("--output_dir", type=str, 
                       default="./outputs",
                       help="Output directory for generated images")
    parser.add_argument("--num_variations", type=int, 
                       default=3,
                       help="Number of variations to generate")
    parser.add_argument("--no_post_process", action="store_true",
                       help="Skip post-processing step")
    parser.add_argument("--steps", type=int, 
                       default=50,
                       help="Number of inference steps")
    parser.add_argument("--guidance", type=float, 
                       default=7.5,
                       help="Guidance scale")
    parser.add_argument("--seed", type=int, 
                       default=None,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Generate
    generator = FloorPlanGenerator(model_path=args.model_path)
    
    formatted_prompt = (
        f"CAD architectural floor plan, {args.prompt}, "
        "top-down view, black and white line drawing, "
        "professional blueprint style, with room labels and dimensions, "
        "clean lines, technical drawing"
    )
    
    images = generator.generate_with_variations(
        prompt=formatted_prompt,
        num_variations=args.num_variations,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed
    )
    
    if not args.no_post_process:
        images = [generator.post_process(img) for img in images]
    
    generator.save_images(images, output_dir=args.output_dir)
    
    print("\nDone!")
