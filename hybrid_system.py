"""
True Hybrid Floor Plan Generation System
Uses diffusion model to generate images, then extracts layout information

Pipeline:
Text → Diffusion Model → Image → Layout Extraction → Constraint Engine → Geometric Rendering

This is a REAL hybrid system where AI influences the final output!
"""

import torch
import numpy as np
from PIL import Image
import cv2
from typing import Dict, List, Tuple, Optional
import os
import json

try:
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    HAS_DIFFUSERS = True
except ImportError:
    print("Warning: diffusers not installed. Run: pip install diffusers")
    HAS_DIFFUSERS = False

try:
    from scipy import ndimage
    from scipy.spatial import distance
    from sklearn.cluster import DBSCAN
    HAS_SCIPY = True
except ImportError:
    print("Warning: scipy/sklearn not installed. Run: pip install scipy scikit-learn")
    HAS_SCIPY = False

from constraint_engine import ConstraintEngine
from geometric_layout_generator import GeometricFloorPlanGenerator
try:
    from enhanced_png_renderer import EnhancedGeometricFloorPlanGenerator
    HAS_PNG = True
except:
    HAS_PNG = False
    EnhancedGeometricFloorPlanGenerator = GeometricFloorPlanGenerator


class DiffusionLayoutGenerator:
    """
    Generates floor plan images using Stable Diffusion with LoRA
    """
    
    def __init__(
        self,
        model_path: str = "runwayml/stable-diffusion-v1-5",
        lora_path: Optional[str] = None,
        device: str = "auto"
    ):
        """
        Initialize diffusion model
        
        Args:
            model_path: Base Stable Diffusion model
            lora_path: Path to LoRA weights (optional)
            device: Device to use
        """
        
        if not HAS_DIFFUSERS:
            raise ImportError("diffusers not installed")
        
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu"
        print(f"Loading diffusion model on {self.device}...")
        
        # Load base model
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None
        )
        
        # Use faster scheduler
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        self.pipe = self.pipe.to(self.device)
        
        # Load LoRA weights if provided
        if lora_path and os.path.exists(lora_path):
            try:
                print(f"Loading LoRA weights from {lora_path}...")
                self.pipe.unet.load_attn_procs(lora_path)
                print("✓ LoRA weights loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load LoRA weights: {e}")
        
        # Enable memory optimizations
        if self.device == "cuda":
            self.pipe.enable_attention_slicing()
        
        print("✓ Diffusion model ready")
    
    def generate_floorplan_image(
        self,
        description: str,
        output_path: str = "ai_layout.png",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> Image.Image:
        """
        Generate floor plan image from text description
        
        Args:
            description: Text description (e.g., "2BHK apartment")
            output_path: Where to save generated image
            num_inference_steps: Number of diffusion steps
            guidance_scale: Guidance scale for generation
            seed: Random seed for reproducibility
            
        Returns:
            Generated PIL Image
        """
        
        # Create optimized prompt for floor plan generation
        prompt = self._create_floorplan_prompt(description)
        
        print(f"\nGenerating floor plan image with diffusion model...")
        print(f"Prompt: {prompt}")
        print(f"Steps: {num_inference_steps}, Guidance: {guidance_scale}")
        
        # Set random seed if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Generate image
        with torch.no_grad():
            result = self.pipe(
                prompt=prompt,
                negative_prompt="3D, perspective, blurry, photograph, realistic, colored, decorations",
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                height=512,
                width=512
            )
        
        image = result.images[0]
        
        # Save generated image
        image.save(output_path)
        print(f"✓ Generated image saved: {output_path}")
        
        return image
    
    def _create_floorplan_prompt(self, description: str) -> str:
        """
        Create optimized prompt for floor plan generation
        
        Args:
            description: User description
            
        Returns:
            Optimized prompt
        """
        
        # Base prompt for architectural floor plans
        base = "black and white architectural floor plan, top-down view, 2D layout, "
        
        # Add description
        base += description + ", "
        
        # Add style keywords
        base += "labeled rooms, clear walls, simple design, blueprint style, "
        base += "professional architectural drawing, CAD style, minimalist"
        
        return base


class ImageLayoutExtractor:
    """
    Extracts layout information from generated floor plan images
    """
    
    def __init__(self, grid_size: int = 32):
        """
        Initialize layout extractor
        
        Args:
            grid_size: Size of grid for analysis (32x32 recommended)
        """
        self.grid_size = grid_size
    
    def extract_layout_from_image(
        self,
        image: Image.Image,
        num_rooms: int,
        room_types: List[str]
    ) -> Dict:
        """
        Extract layout information from generated image
        
        Args:
            image: Generated floor plan image
            num_rooms: Expected number of rooms
            room_types: List of room types (e.g., ['bedroom', 'kitchen'])
            
        Returns:
            Layout specification dictionary
        """
        
        print(f"\nExtracting layout from AI-generated image...")
        print(f"Target: {num_rooms} rooms of types {room_types}")
        
        # Step 1: Preprocess image
        processed = self._preprocess_image(image)
        
        # Step 2: Detect room regions
        room_regions = self._detect_room_regions(processed, num_rooms)
        
        # Step 3: Convert regions to bounding boxes
        bounding_boxes = self._regions_to_bounding_boxes(room_regions)
        
        # Step 4: Create layout specification
        layout_spec = self._create_layout_spec(bounding_boxes, room_types)
        
        print(f"✓ Extracted {len(layout_spec['rooms'])} room regions from AI image")
        
        return layout_spec
    
    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess image for layout extraction
        
        Args:
            image: Input image
            
        Returns:
            Processed numpy array
        """
        
        # Convert to grayscale
        gray = image.convert('L')
        
        # Resize to grid size
        resized = gray.resize((self.grid_size, self.grid_size), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(resized)
        
        # Apply threshold to get binary image (dark = rooms/walls)
        # Invert so rooms are white, background is black
        threshold = np.mean(img_array)
        binary = (img_array < threshold).astype(np.uint8) * 255
        
        return binary
    
    def _detect_room_regions(self, binary_image: np.ndarray, target_rooms: int) -> List[np.ndarray]:
        """
        Detect room regions in binary image
        
        Args:
            binary_image: Binary image (rooms as white regions)
            target_rooms: Target number of rooms
            
        Returns:
            List of room region masks
        """
        
        if not HAS_SCIPY:
            # Fallback: Simple grid division
            return self._simple_grid_division(binary_image, target_rooms)
        
        # Label connected components
        labeled, num_features = ndimage.label(binary_image)
        
        # Get region properties
        regions = []
        for label_id in range(1, num_features + 1):
            region_mask = (labeled == label_id)
            region_size = np.sum(region_mask)
            
            # Filter out very small regions (noise)
            if region_size > (self.grid_size * self.grid_size) * 0.02:  # At least 2% of image
                regions.append(region_mask)
        
        # If we have too many regions, merge small ones
        if len(regions) > target_rooms * 1.5:
            regions = self._merge_small_regions(regions, target_rooms)
        
        # If we have too few regions, split large ones
        if len(regions) < target_rooms:
            regions = self._split_large_regions(regions, target_rooms, binary_image.shape)
        
        return regions[:target_rooms]  # Return target number of regions
    
    def _simple_grid_division(self, binary_image: np.ndarray, num_rooms: int) -> List[np.ndarray]:
        """
        Fallback: Simple grid division when scipy not available
        
        Args:
            binary_image: Binary image
            num_rooms: Number of rooms to create
            
        Returns:
            List of region masks
        """
        
        regions = []
        h, w = binary_image.shape
        
        # Determine grid layout
        if num_rooms <= 2:
            rows, cols = 1, num_rooms
        elif num_rooms <= 4:
            rows, cols = 2, 2
        elif num_rooms <= 6:
            rows, cols = 2, 3
        else:
            rows, cols = 3, 3
        
        # Create grid regions
        for i in range(min(rows, num_rooms)):
            for j in range(min(cols, num_rooms)):
                if len(regions) >= num_rooms:
                    break
                
                # Calculate region bounds
                y1 = int(i * h / rows)
                y2 = int((i + 1) * h / rows)
                x1 = int(j * w / cols)
                x2 = int((j + 1) * w / cols)
                
                # Create mask for this region
                mask = np.zeros_like(binary_image, dtype=bool)
                mask[y1:y2, x1:x2] = True
                
                regions.append(mask)
        
        return regions
    
    def _merge_small_regions(self, regions: List[np.ndarray], target: int) -> List[np.ndarray]:
        """Merge small regions to reach target count"""
        
        # Sort by size
        sorted_regions = sorted(regions, key=lambda r: np.sum(r), reverse=True)
        
        # Keep largest target regions
        return sorted_regions[:target]
    
    def _split_large_regions(
        self, 
        regions: List[np.ndarray], 
        target: int,
        image_shape: Tuple[int, int]
    ) -> List[np.ndarray]:
        """Split large regions to reach target count"""
        
        result = list(regions)
        
        while len(result) < target:
            # Find largest region
            largest_idx = max(range(len(result)), key=lambda i: np.sum(result[i]))
            largest = result[largest_idx]
            
            # Split it in half (vertically or horizontally based on aspect ratio)
            coords = np.argwhere(largest)
            
            if len(coords) == 0:
                break
            
            min_y, min_x = coords.min(axis=0)
            max_y, max_x = coords.max(axis=0)
            
            height = max_y - min_y
            width = max_x - min_x
            
            # Split along longer dimension
            mask1 = np.zeros_like(largest)
            mask2 = np.zeros_like(largest)
            
            if height > width:
                # Split horizontally
                mid_y = (min_y + max_y) // 2
                mask1[min_y:mid_y, :] = largest[min_y:mid_y, :]
                mask2[mid_y:max_y, :] = largest[mid_y:max_y, :]
            else:
                # Split vertically
                mid_x = (min_x + max_x) // 2
                mask1[:, min_x:mid_x] = largest[:, min_x:mid_x]
                mask2[:, mid_x:max_x] = largest[:, mid_x:max_x]
            
            # Replace largest with two split regions
            result[largest_idx] = mask1
            result.append(mask2)
        
        return result
    
    def _regions_to_bounding_boxes(self, regions: List[np.ndarray]) -> List[Dict]:
        """
        Convert region masks to bounding boxes
        
        Args:
            regions: List of binary masks
            
        Returns:
            List of bounding box dictionaries
        """
        
        bboxes = []
        
        for region in regions:
            coords = np.argwhere(region)
            
            if len(coords) == 0:
                continue
            
            # Get bounding box
            min_y, min_x = coords.min(axis=0)
            max_y, max_x = coords.max(axis=0)
            
            # Convert from grid coordinates to feet (scale factor)
            scale_factor = 40.0 / self.grid_size  # Assume 40 feet total width
            
            bbox = {
                'x': float(min_x * scale_factor),
                'y': float(min_y * scale_factor),
                'width': float((max_x - min_x) * scale_factor),
                'height': float((max_y - min_y) * scale_factor)
            }
            
            bboxes.append(bbox)
        
        return bboxes
    
    def _create_layout_spec(
        self,
        bounding_boxes: List[Dict],
        room_types: List[str]
    ) -> Dict:
        """
        Create layout specification from bounding boxes
        
        Args:
            bounding_boxes: List of bounding boxes
            room_types: List of room types to assign
            
        Returns:
            Layout specification dictionary
        """
        
        layout_spec = {
            'layout_type': 'ai_generated',
            'source': 'diffusion_model',
            'rooms': []
        }
        
        # Assign room types to bounding boxes
        for i, bbox in enumerate(bounding_boxes):
            # Get room type (cycle through if we have more boxes than types)
            room_type = room_types[i % len(room_types)] if room_types else 'room'
            
            # Create room label
            room_label = self._type_to_label(room_type, i)
            
            # Create room specification
            room_spec = {
                'type': room_type,
                'label': room_label,
                'width': max(bbox['width'], 6.0),  # Minimum 6 feet
                'height': max(bbox['height'], 6.0),  # Minimum 6 feet
                'position': {
                    'x': bbox['x'],
                    'y': bbox['y']
                },
                'doors': [
                    {
                        'x': bbox['x'],
                        'y': bbox['y'] + bbox['height'] / 2,
                        'width': 3.0
                    }
                ],
                'windows': [
                    {
                        'x': bbox['x'] + bbox['width'] / 2,
                        'y': bbox['y'] + bbox['height'],
                        'width': 4.0,
                        'height': 4.0
                    }
                ] if room_type not in ['bathroom', 'toilet'] else []
            }
            
            layout_spec['rooms'].append(room_spec)
        
        return layout_spec
    
    def _type_to_label(self, room_type: str, index: int) -> str:
        """Convert room type to display label"""
        
        labels = {
            'living_room': 'Living Room',
            'bedroom': f'Bedroom {index}' if index > 0 else 'Master Bedroom',
            'master_bedroom': 'Master Bedroom',
            'kitchen': 'Kitchen',
            'bathroom': f'Bathroom {index}' if index > 0 else 'Bathroom',
            'toilet': 'Toilet',
            'dining': 'Dining Room',
            'hall': 'Hall',
            'study': 'Study',
            'balcony': 'Balcony'
        }
        
        return labels.get(room_type, room_type.title())


class TrueHybridFloorPlanSystem:
    """
    True hybrid system that uses diffusion model for actual generation
    
    Pipeline:
    Text → Diffusion → Image → Extract Layout → Constraints → Geometry → Output
    """
    
    def __init__(
        self,
        model_path: str = "runwayml/stable-diffusion-v1-5",
        lora_path: Optional[str] = None,
        use_diffusion: bool = True
    ):
        """
        Initialize true hybrid system
        
        Args:
            model_path: Stable Diffusion model path
            lora_path: LoRA weights path (optional)
            use_diffusion: Whether to use diffusion model
        """
        
        self.use_diffusion = use_diffusion
        
        # Initialize diffusion generator if enabled
        self.diffusion_generator = None
        if use_diffusion and HAS_DIFFUSERS:
            try:
                self.diffusion_generator = DiffusionLayoutGenerator(
                    model_path=model_path,
                    lora_path=lora_path
                )
            except Exception as e:
                print(f"Warning: Could not initialize diffusion model: {e}")
                self.use_diffusion = False
        
        # Initialize layout extractor
        self.layout_extractor = ImageLayoutExtractor(grid_size=32)
        
        # Initialize constraint engine
        self.constraint_engine = ConstraintEngine(
            max_iterations=10,
            auto_correct=True,
            strict_mode=False
        )
        
        # Initialize geometric generator
        if HAS_PNG:
            self.geometric_generator = EnhancedGeometricFloorPlanGenerator()
        else:
            self.geometric_generator = GeometricFloorPlanGenerator()
        
        print(f"\n✓ True Hybrid System initialized")
        print(f"  Diffusion: {'Enabled' if self.use_diffusion else 'Disabled'}")
        print(f"  Constraint Engine: Enabled")
        print(f"  Geometric Rendering: Enabled")
    
    def generate(
        self,
        description: str,
        output_dir: str = "./hybrid_outputs",
        num_inference_steps: int = 30,
        save_intermediate: bool = True
    ) -> Dict:
        """
        Generate floor plan using true hybrid approach
        
        Args:
            description: Text description
            output_dir: Output directory
            num_inference_steps: Diffusion steps
            save_intermediate: Save intermediate outputs
            
        Returns:
            Generation results
        """
        
        print(f"\n{'='*80}")
        print("TRUE HYBRID FLOOR PLAN GENERATION")
        print(f"{'='*80}")
        print(f"Description: {description}")
        print(f"Using Diffusion: {self.use_diffusion}")
        print(f"{'='*80}\n")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Parse description to get room count and types
        room_info = self._parse_description(description)
        
        # STEP 1: Generate image with diffusion model
        print("STEP 1: Diffusion Model - Generating Floor Plan Image")
        print("-" * 40)
        
        if self.use_diffusion and self.diffusion_generator:
            ai_image_path = os.path.join(output_dir, "ai_layout.png")
            ai_image = self.diffusion_generator.generate_floorplan_image(
                description=description,
                output_path=ai_image_path,
                num_inference_steps=num_inference_steps
            )
            print(f"✓ AI-generated image: {ai_image_path}")
        else:
            print("⚠ Diffusion disabled, using rule-based layout")
            ai_image = None
        
        # STEP 2: Extract layout from AI image
        print("\nSTEP 2: Image Processing - Extracting Layout from AI Image")
        print("-" * 40)
        
        if ai_image:
            ai_layout_spec = self.layout_extractor.extract_layout_from_image(
                image=ai_image,
                num_rooms=room_info['num_rooms'],
                room_types=room_info['room_types']
            )
            
            if save_intermediate:
                spec_path = os.path.join(output_dir, "ai_extracted_layout.json")
                with open(spec_path, 'w') as f:
                    json.dump(ai_layout_spec, f, indent=2)
                print(f"✓ AI-extracted layout saved: {spec_path}")
        else:
            # Fallback to rule-based
            ai_layout_spec = self._create_fallback_layout(room_info)
            print("✓ Using fallback rule-based layout")
        
        # STEP 3: Constraint Engine - Validate and correct
        print("\nSTEP 3: Constraint Engine - Validating AI Layout")
        print("-" * 40)
        
        is_valid, validated_spec, violations = self.constraint_engine.validate_and_correct(
            ai_layout_spec
        )
        
        print(f"✓ AI layout validated and corrected")
        print(f"  Corrections applied: {len(self.constraint_engine.corrections_applied)}")
        
        # STEP 4: Geometric rendering
        print("\nSTEP 4: Geometric Rendering - Creating Final Floor Plan")
        print("-" * 40)
        
        self.geometric_generator.parse_layout_specification(validated_spec)
        print(f"✓ Geometric layout created with {len(self.geometric_generator.rooms)} rooms")
        
        # STEP 5: Export to formats
        print("\nSTEP 5: Multi-Format Export")
        print("-" * 40)
        
        output_files = {}
        base_name = description.replace(' ', '_')[:40]
        
        # PNG
        if hasattr(self.geometric_generator, 'export_to_png'):
            png_path = os.path.join(output_dir, f"{base_name}_final.png")
            self.geometric_generator.export_to_png(png_path)
            output_files['png'] = png_path
        
        # DXF
        dxf_path = os.path.join(output_dir, f"{base_name}_final.dxf")
        self.geometric_generator.export_to_dxf(dxf_path)
        output_files['dxf'] = dxf_path
        
        # SVG
        svg_path = os.path.join(output_dir, f"{base_name}_final.svg")
        self.geometric_generator.export_to_svg(svg_path)
        output_files['svg'] = svg_path
        
        # Save complete report
        report = {
            'description': description,
            'used_diffusion': self.use_diffusion,
            'ai_image': ai_image_path if ai_image else None,
            'ai_extracted_layout': ai_layout_spec,
            'constraint_validation': {
                'is_valid': is_valid,
                'corrections_applied': self.constraint_engine.corrections_applied,
                'violations': [
                    {
                        'type': v.type.value,
                        'severity': v.severity.value,
                        'message': v.message
                    }
                    for v in violations[:10]
                ]
            },
            'output_files': output_files
        }
        
        report_path = os.path.join(output_dir, f"{base_name}_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        output_files['report'] = report_path
        
        print(f"\n{'='*80}")
        print("GENERATION COMPLETE")
        print(f"{'='*80}")
        print(f"AI-Generated: {'Yes' if self.use_diffusion else 'No (fallback)'}")
        print(f"Constraints Applied: {len(self.constraint_engine.corrections_applied)}")
        print(f"\nOutput Files:")
        for fmt, path in output_files.items():
            print(f"  {fmt.upper()}: {path}")
        print(f"{'='*80}\n")
        
        return report
    
    def _parse_description(self, description: str) -> Dict:
        """Parse description to extract room information"""
        
        desc_lower = description.lower()
        
        # Extract bedroom count
        num_bedrooms = 1
        if '2bhk' in desc_lower or '2 bhk' in desc_lower:
            num_bedrooms = 2
        elif '3bhk' in desc_lower or '3 bhk' in desc_lower:
            num_bedrooms = 3
        elif '4bhk' in desc_lower or '4 bhk' in desc_lower:
            num_bedrooms = 4
        elif '1bhk' in desc_lower or '1 bhk' in desc_lower:
            num_bedrooms = 1
        
        # Build room types list
        room_types = []
        
        # Add living room or hall
        if 'hall' in desc_lower:
            room_types.append('hall')
        else:
            room_types.append('living_room')
        
        # Add bedrooms
        for i in range(num_bedrooms):
            if i == 0:
                room_types.append('master_bedroom')
            else:
                room_types.append('bedroom')
        
        # Add kitchen
        room_types.append('kitchen')
        
        # Add bathrooms
        num_bathrooms = max(1, num_bedrooms // 2)
        for i in range(num_bathrooms):
            room_types.append('bathroom')
        
        # Optional rooms
        if 'study' in desc_lower:
            room_types.append('study')
        if 'dining' in desc_lower:
            room_types.append('dining')
        if 'balcony' in desc_lower:
            room_types.append('balcony')
        
        return {
            'num_rooms': len(room_types),
            'room_types': room_types,
            'num_bedrooms': num_bedrooms
        }
    
    def _create_fallback_layout(self, room_info: Dict) -> Dict:
        """Create fallback rule-based layout if diffusion not available"""
        
        from architectural_layout_generator import ArchitecturalLayoutGenerator
        
        # Use architectural generator as fallback
        arch_gen = ArchitecturalLayoutGenerator()
        
        # Create simple description
        desc = f"{room_info['num_bedrooms']}BHK apartment"
        
        return arch_gen.generate_layout_from_text(desc)


# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="True Hybrid Floor Plan Generation - Uses Diffusion Model for Real!"
    )
    parser.add_argument('description', type=str, help='Floor plan description')
    parser.add_argument('--model_path', type=str, 
                       default="runwayml/stable-diffusion-v1-5",
                       help='Stable Diffusion model path')
    parser.add_argument('--lora_path', type=str, default=None,
                       help='Path to LoRA weights')
    parser.add_argument('--output_dir', type=str, default='./true_hybrid_outputs',
                       help='Output directory')
    parser.add_argument('--steps', type=int, default=30,
                       help='Number of diffusion steps')
    parser.add_argument('--no_diffusion', action='store_true',
                       help='Disable diffusion, use only rules')
    
    args = parser.parse_args()
    
    # Create system
    system = TrueHybridFloorPlanSystem(
        model_path=args.model_path,
        lora_path=args.lora_path,
        use_diffusion=not args.no_diffusion
    )
    
    # Generate
    result = system.generate(
        description=args.description,
        output_dir=args.output_dir,
        num_inference_steps=args.steps
    )