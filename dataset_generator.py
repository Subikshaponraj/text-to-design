"""
Dataset Preparation for Floor Plan Generation
Prepares training data for fine-tuning Stable Diffusion / ControlNet
"""

import os
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Tuple
import random


class FloorPlanDatasetGenerator:
    """
    Generates synthetic floor plan training data
    Can also process real floor plan images with annotations
    """
    
    def __init__(self, output_dir: str = './dataset'):
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, 'images')
        self.labels_dir = os.path.join(output_dir, 'labels')
        self.prompts_dir = os.path.join(output_dir, 'prompts')
        self.controls_dir = os.path.join(output_dir, 'controls')  # For ControlNet
        
        # Create directories
        for dir_path in [self.images_dir, self.labels_dir, self.prompts_dir, self.controls_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def generate_synthetic_floor_plan(self, 
                                     layout_spec: Dict, 
                                     image_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
        """
        Generate a CAD-style floor plan image from layout specification
        
        Args:
            layout_spec: Layout specification (from LLM parser or validation)
            image_size: Output image size
        
        Returns:
            numpy array of generated floor plan image
        """
        
        # Create blank white canvas
        img = Image.new('RGB', image_size, color='white')
        draw = ImageDraw.Draw(img)
        
        # Calculate scale factor (pixels per foot)
        total_area = layout_spec.get('total_area', 1000)
        assumed_width = total_area ** 0.5  # Assume roughly square layout
        scale = min(image_size) / (assumed_width * 1.2)  # 1.2 for padding
        
        wall_thickness = 3  # pixels
        door_size = int(3 * scale)  # 3 feet in pixels
        
        # Draw rooms
        rooms = layout_spec.get('rooms', [])
        for room in rooms:
            pos = room.get('position', {'x': 0, 'y': 0})
            dims = room.get('dimensions', {'width': 10, 'height': 10})
            
            # Convert to pixels
            x = int(pos['x'] * scale) + 50  # 50px padding
            y = int(pos['y'] * scale) + 50
            w = int(dims['width'] * scale)
            h = int(dims['height'] * scale)
            
            # Draw room outline (walls)
            draw.rectangle([x, y, x+w, y+h], outline='black', width=wall_thickness)
            
            # Add room label
            room_type = room.get('type', 'room')
            label = room_type.replace('_', ' ').title()
            
            # Calculate text position (center of room)
            text_bbox = draw.textbbox((0, 0), label)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_x = x + (w - text_width) // 2
            text_y = y + (h - text_height) // 2
            
            draw.text((text_x, text_y), label, fill='gray')
            
            # Draw doors
            doors = room.get('doors', [])
            for door in doors:
                door_pos = door.get('position', 'north')
                door_width = int(door.get('width', 2.5) * scale)
                
                if door_pos == 'north':
                    door_x = x + (w - door_width) // 2
                    door_y = y
                    # Draw door opening (gap in wall)
                    draw.rectangle([door_x, door_y-wall_thickness, 
                                   door_x+door_width, door_y+wall_thickness], 
                                  fill='white', outline='white')
                    # Draw door arc
                    draw.arc([door_x, door_y, door_x+door_width, door_y+door_width], 
                            start=0, end=90, fill='black', width=1)
                
                elif door_pos == 'south':
                    door_x = x + (w - door_width) // 2
                    door_y = y + h
                    draw.rectangle([door_x, door_y-wall_thickness, 
                                   door_x+door_width, door_y+wall_thickness], 
                                  fill='white', outline='white')
                    draw.arc([door_x, door_y-door_width, door_x+door_width, door_y], 
                            start=180, end=270, fill='black', width=1)
                
                elif door_pos == 'east':
                    door_x = x + w
                    door_y = y + (h - door_width) // 2
                    draw.rectangle([door_x-wall_thickness, door_y, 
                                   door_x+wall_thickness, door_y+door_width], 
                                  fill='white', outline='white')
                    draw.arc([door_x-door_width, door_y, door_x, door_y+door_width], 
                            start=90, end=180, fill='black', width=1)
                
                else:  # west
                    door_x = x
                    door_y = y + (h - door_width) // 2
                    draw.rectangle([door_x-wall_thickness, door_y, 
                                   door_x+wall_thickness, door_y+door_width], 
                                  fill='white', outline='white')
                    draw.arc([door_x, door_y, door_x+door_width, door_y+door_width], 
                            start=270, end=360, fill='black', width=1)
            
            # Draw windows
            windows = room.get('windows', [])
            window_thickness = 2
            for window in windows:
                win_pos = window.get('position', 'exterior')
                win_width = int(window.get('width', 4) * scale)
                
                # Simplified: draw windows on north wall
                win_x = x + (w - win_width) // 2
                win_y = y
                # Draw as thick line
                draw.line([win_x, win_y, win_x+win_width, win_y], 
                         fill='blue', width=window_thickness*2)
        
        # Add dimensions and annotations
        self._add_dimensions(draw, rooms, scale)
        
        return np.array(img)
    
    def _add_dimensions(self, draw, rooms, scale):
        """Add dimension lines to floor plan"""
        # Add dimension lines for main rooms
        # This is simplified - in production would add proper dimension chains
        for room in rooms[:1]:  # Just first room for example
            pos = room.get('position', {'x': 0, 'y': 0})
            dims = room.get('dimensions', {'width': 10, 'height': 10})
            
            x = int(pos['x'] * scale) + 50
            y = int(pos['y'] * scale) + 50
            w = int(dims['width'] * scale)
            
            # Draw dimension line
            dim_y = y - 20
            draw.line([x, dim_y, x+w, dim_y], fill='red', width=1)
            draw.line([x, dim_y-5, x, dim_y+5], fill='red', width=1)  # End cap
            draw.line([x+w, dim_y-5, x+w, dim_y+5], fill='red', width=1)  # End cap
            
            # Add dimension text
            dim_text = f"{dims['width']}'"
            draw.text((x + w//2 - 10, dim_y - 15), dim_text, fill='red')
    
    def create_control_image(self, layout_spec: Dict, image_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
        """
        Create control image for ControlNet (edge map / semantic map)
        This provides structural guidance to the diffusion model
        """
        
        # Create simplified edge map
        img = Image.new('L', image_size, color=255)  # Grayscale
        draw = ImageDraw.Draw(img)
        
        # Similar to floor plan but only edges, no text
        total_area = layout_spec.get('total_area', 1000)
        assumed_width = total_area ** 0.5
        scale = min(image_size) / (assumed_width * 1.2)
        
        rooms = layout_spec.get('rooms', [])
        for room in rooms:
            pos = room.get('position', {'x': 0, 'y': 0})
            dims = room.get('dimensions', {'width': 10, 'height': 10})
            
            x = int(pos['x'] * scale) + 50
            y = int(pos['y'] * scale) + 50
            w = int(dims['width'] * scale)
            h = int(dims['height'] * scale)
            
            # Draw room outline
            draw.rectangle([x, y, x+w, y+h], outline=0, width=2)
        
        # Apply Canny edge detection for cleaner edges
        img_np = np.array(img)
        edges = cv2.Canny(img_np, 100, 200)
        
        return edges
    
    def generate_text_prompt(self, layout_spec: Dict) -> str:
        """
        Generate natural language prompt for the layout
        This is what the diffusion model will be conditioned on
        """
        
        rooms = layout_spec.get('rooms', [])
        room_types = [r.get('type', '').replace('_', ' ') for r in rooms]
        
        # Count room types
        from collections import Counter
        room_counts = Counter(room_types)
        
        # Build description
        parts = []
        
        # Count bedrooms
        num_bedrooms = room_counts.get('bedroom', 0) + room_counts.get('master bedroom', 0)
        if num_bedrooms > 0:
            parts.append(f"{num_bedrooms}BHK")
        
        # Layout type
        layout_type = layout_spec.get('layout_type', 'apartment')
        parts.append(layout_type)
        
        # Mention key rooms
        if 'kitchen' in room_types:
            parts.append("with kitchen")
        if 'living room' in room_types:
            parts.append("and living room")
        if 'balcony' in room_types:
            parts.append("with balcony")
        
        # Check for special features
        special = layout_spec.get('metadata', {}).get('special_requirements', [])
        if 'modern' in special or layout_spec.get('design_style') == 'modern':
            parts.append("modern design")
        
        prompt = ' '.join(parts)
        
        # Add technical description
        full_prompt = f"CAD architectural floor plan, {prompt}, top-down view, "
        full_prompt += "black and white line drawing, professional blueprint style, "
        full_prompt += "with room labels and dimensions, clean lines, technical drawing"
        
        return full_prompt
    
    def save_training_sample(self, layout_spec: Dict, sample_id: str):
        """
        Save a complete training sample (image, control, label, prompt)
        """
        
        # Generate floor plan image
        floor_plan = self.generate_synthetic_floor_plan(layout_spec)
        cv2.imwrite(os.path.join(self.images_dir, f"{sample_id}.png"), 
                   cv2.cvtColor(floor_plan, cv2.COLOR_RGB2BGR))
        
        # Generate control image
        control = self.create_control_image(layout_spec)
        cv2.imwrite(os.path.join(self.controls_dir, f"{sample_id}.png"), control)
        
        # Save label (layout specification)
        with open(os.path.join(self.labels_dir, f"{sample_id}.json"), 'w') as f:
            json.dump(layout_spec, f, indent=2)
        
        # Save prompt
        prompt = self.generate_text_prompt(layout_spec)
        with open(os.path.join(self.prompts_dir, f"{sample_id}.txt"), 'w') as f:
            f.write(prompt)
        
        print(f"Saved training sample: {sample_id}")
        return floor_plan, control, prompt
    
    def process_real_floor_plan(self, image_path: str, annotation: Dict, sample_id: str):
        """
        Process a real floor plan image with manual annotations
        """
        
        # Load and preprocess image
        img = cv2.imread(image_path)
        img = cv2.resize(img, (512, 512))
        
        # Save processed image
        cv2.imwrite(os.path.join(self.images_dir, f"{sample_id}.png"), img)
        
        # Create control image from real image (edge detection)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        cv2.imwrite(os.path.join(self.controls_dir, f"{sample_id}.png"), edges)
        
        # Save annotation
        with open(os.path.join(self.labels_dir, f"{sample_id}.json"), 'w') as f:
            json.dump(annotation, f, indent=2)
        
        # Generate prompt from annotation
        prompt = self.generate_text_prompt(annotation)
        with open(os.path.join(self.prompts_dir, f"{sample_id}.txt"), 'w') as f:
            f.write(prompt)
        
        print(f"Processed real floor plan: {sample_id}")


def create_sample_dataset(num_samples: int = 100):
    """Create sample synthetic dataset for testing"""
    
    from llm_parser import LLMTextParser
    
    generator = FloorPlanDatasetGenerator(output_dir='./floor_plan_dataset')
    
    # Sample descriptions
    templates = [
        "2BHK apartment with attached bathroom and open kitchen",
        "3BHK house with separate dining and modular kitchen",
        "1BHK compact apartment with modern layout",
        "2BHK with balcony attached to living room and master bedroom",
        "3BHK luxury apartment with study room and servant quarter",
        "2BHK with efficient layout and good ventilation",
        "4BHK penthouse with multiple balconies",
        "2BHK with separate pooja room and utility area",
    ]
    
    # Create simple layouts without LLM for demo
    for i in range(num_samples):
        # Create simple layout spec
        layout_spec = {
            'layout_id': f'sample_{i:04d}',
            'total_area': random.randint(800, 1500),
            'layout_type': 'apartment',
            'rooms': [
                {
                    'type': 'living_room',
                    'area': random.randint(200, 300),
                    'dimensions': {'width': 12, 'height': 18},
                    'position': {'x': 0, 'y': 0},
                    'doors': [{'position': 'north', 'width': 3}],
                    'windows': [{'position': 'west', 'width': 4}]
                },
                {
                    'type': 'master_bedroom',
                    'area': random.randint(150, 200),
                    'dimensions': {'width': 12, 'height': 15},
                    'position': {'x': 12, 'y': 0},
                    'doors': [{'position': 'west', 'width': 2.5}],
                    'windows': [{'position': 'south', 'width': 4}]
                },
                {
                    'type': 'kitchen',
                    'area': random.randint(80, 120),
                    'dimensions': {'width': 8, 'height': 12},
                    'position': {'x': 0, 'y': 18},
                    'doors': [{'position': 'north', 'width': 2.5}],
                    'windows': [{'position': 'west', 'width': 3}],
                    'has_exhaust': True
                }
            ],
            'connections': [
                {'from': 'living_room', 'to': 'master_bedroom', 'type': 'door'},
                {'from': 'living_room', 'to': 'kitchen', 'type': 'door'}
            ],
            'metadata': {
                'design_style': random.choice(['modern', 'traditional', 'minimalist']),
                'special_requirements': []
            }
        }
        
        generator.save_training_sample(layout_spec, f"sample_{i:04d}")
    
    print(f"\nDataset created successfully!")
    print(f"Location: ./floor_plan_dataset/")
    print(f"Samples: {num_samples}")
    print(f"\nDirectory structure:")
    print(f"  - images/     : Floor plan images")
    print(f"  - controls/   : Control images for ControlNet")
    print(f"  - labels/     : Layout specifications (JSON)")
    print(f"  - prompts/    : Text prompts")


if __name__ == "__main__":
    # Create sample dataset
    create_sample_dataset(num_samples=50)
