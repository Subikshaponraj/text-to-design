"""
Main End-to-End Text-to-CAD Layout Generation System
Integrates: LLM Parser → Rule Engine → Image Generation → Validation
"""

import os
import json
from typing import Dict, List, Tuple, Optional
from PIL import Image
import argparse

# Import our modules
from llm_parser import LLMTextParser
from rule_engine import LayoutValidator, ValidationError
from inference import FloorPlanGenerator


class TextToCADSystem:
    """
    Complete end-to-end system for generating CAD layouts from text
    
    Pipeline:
    1. Parse natural language input with LLM
    2. Validate against engineering rules
    3. Generate floor plan image
    4. Validate output
    5. Return final layout
    """
    
    def __init__(
        self,
        model_path: str = "./trained_model/final_model",
        hf_token: Optional[str] = None,
        llm_model: str = 'qwen',
        max_iterations: int = 3,
        auto_fix: bool = True
    ):
        """
        Initialize the complete system
        
        Args:
            model_path: Path to trained floor plan generation model
            hf_token: HuggingFace API token (optional)
            llm_model: LLM to use for parsing ('qwen', 'mistral', 'phi')
            max_iterations: Maximum attempts to generate valid layout
            auto_fix: Automatically fix validation errors when possible
        """
        
        self.max_iterations = max_iterations
        self.auto_fix = auto_fix
        
        print("="*80)
        print("Initializing Text-to-CAD Layout Generation System")
        print("="*80)
        
        # Initialize components
        print("\n1. Loading LLM Text Parser...")
        self.parser = LLMTextParser(hf_token=hf_token, model=llm_model)
        
        print("\n2. Loading Engineering Rule Engine...")
        self.validator = LayoutValidator()
        
        print("\n3. Loading Floor Plan Generator...")
        self.generator = FloorPlanGenerator(model_path=model_path)
        
        print("\n✓ System ready!")
        print("="*80)
    
    def generate_layout(
        self,
        text_description: str,
        num_variations: int = 3,
        output_dir: str = "./outputs",
        save_intermediate: bool = True
    ) -> Dict:
        """
        Generate complete layout from text description
        
        Args:
            text_description: Natural language description
            num_variations: Number of image variations to generate
            output_dir: Directory to save outputs
            save_intermediate: Save intermediate JSON specifications
        
        Returns:
            Dictionary containing:
                - layout_spec: Final validated layout specification
                - images: List of generated PIL Images
                - validation_report: Validation results
                - metadata: Processing metadata
        """
        
        print(f"\n{'='*80}")
        print(f"Processing Request: '{text_description}'")
        print(f"{'='*80}\n")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Parse text input
        print("STEP 1: Parsing Natural Language Input")
        print("-" * 40)
        layout_spec = self.parser.parse_architectural_requirements(text_description)
        
        if save_intermediate:
            spec_path = os.path.join(output_dir, "parsed_specification.json")
            with open(spec_path, 'w') as f:
                json.dump(layout_spec, f, indent=2)
            print(f"✓ Parsed specification saved: {spec_path}")
        
        # Step 2: Convert to detailed specification
        print("\nSTEP 2: Converting to Detailed Specification")
        print("-" * 40)
        detailed_spec = self.parser.convert_to_detailed_specification(layout_spec)
        
        if save_intermediate:
            detailed_path = os.path.join(output_dir, "detailed_specification.json")
            with open(detailed_path, 'w') as f:
                json.dump(detailed_spec, f, indent=2)
            print(f"✓ Detailed specification saved: {detailed_path}")
        
        # Step 3: Validate layout against engineering rules
        print("\nSTEP 3: Validating Against Engineering Rules")
        print("-" * 40)
        
        is_valid, errors = self._validate_with_retries(detailed_spec)
        
        validation_report = {
            'is_valid': is_valid,
            'errors': [{'severity': e.severity, 'code': e.code, 'message': e.message} for e in errors],
            'num_errors': len([e for e in errors if e.severity == 'critical']),
            'num_warnings': len([e for e in errors if e.severity == 'warning'])
        }
        
        if save_intermediate:
            validation_path = os.path.join(output_dir, "validation_report.json")
            with open(validation_path, 'w') as f:
                json.dump(validation_report, f, indent=2)
            print(f"✓ Validation report saved: {validation_path}")
        
        # Step 4: Generate floor plan images
        print("\nSTEP 4: Generating Floor Plan Images")
        print("-" * 40)
        
        # Create optimized prompt
        prompt = self._create_generation_prompt(text_description, detailed_spec)
        print(f"Generation prompt: {prompt[:100]}...")
        
        images = self.generator.generate_with_variations(
            prompt=prompt,
            num_variations=num_variations,
            num_inference_steps=50,
            guidance_scale=7.5
        )
        
        print(f"✓ Generated {len(images)} floor plan variations")
        
        # Post-process images
        print("\nSTEP 5: Post-Processing Images")
        print("-" * 40)
        processed_images = [self.generator.post_process(img) for img in images]
        print(f"✓ Post-processed {len(processed_images)} images")
        
        # Save images
        saved_paths = self.generator.save_images(
            processed_images,
            output_dir=output_dir,
            prefix="final_layout"
        )
        
        # Step 6: Create summary report
        print("\nSTEP 6: Creating Summary Report")
        print("-" * 40)
        
        summary = self._create_summary_report(
            text_description,
            detailed_spec,
            validation_report,
            saved_paths
        )
        
        if save_intermediate:
            summary_path = os.path.join(output_dir, "generation_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"✓ Summary report saved: {summary_path}")
        
        # Print final summary
        print(f"\n{'='*80}")
        print("GENERATION COMPLETE")
        print(f"{'='*80}")
        print(f"Status: {'✓ VALID' if is_valid else '⚠ HAS WARNINGS'}")
        print(f"Images Generated: {len(processed_images)}")
        print(f"Output Directory: {output_dir}")
        
        if validation_report['num_errors'] > 0:
            print(f"\n⚠ Critical Errors: {validation_report['num_errors']}")
        if validation_report['num_warnings'] > 0:
            print(f"⚠ Warnings: {validation_report['num_warnings']}")
        
        print(f"{'='*80}\n")
        
        return {
            'layout_spec': detailed_spec,
            'images': processed_images,
            'validation_report': validation_report,
            'metadata': summary,
            'image_paths': saved_paths
        }
    
    def _validate_with_retries(self, layout_spec: Dict) -> Tuple[bool, List[ValidationError]]:
        """Validate layout with automatic fixes and retries"""
        
        for attempt in range(self.max_iterations):
            is_valid, errors = self.validator.validate_layout(layout_spec)
            
            print(f"Validation attempt {attempt + 1}/{self.max_iterations}")
            print(f"  Critical errors: {len([e for e in errors if e.severity == 'critical'])}")
            print(f"  Warnings: {len([e for e in errors if e.severity == 'warning'])}")
            
            if is_valid or not self.auto_fix:
                break
            
            # Try to auto-fix critical errors
            if attempt < self.max_iterations - 1:
                print("  Attempting auto-fix...")
                layout_spec = self._auto_fix_layout(layout_spec, errors)
        
        return is_valid, errors
    
    def _auto_fix_layout(self, layout_spec: Dict, errors: List[ValidationError]) -> Dict:
        """Automatically fix common validation errors"""
        
        rooms = layout_spec.get('rooms', [])
        
        for error in errors:
            if error.severity != 'critical':
                continue
            
            # Fix room size issues
            if error.code == 'ROOM_SIZE_TOO_SMALL':
                room_type = error.details.get('room')
                min_required = error.details.get('min_required')
                
                for room in rooms:
                    if room.get('type') == room_type:
                        # Increase area
                        room['area'] = min_required * 1.1
                        # Adjust dimensions proportionally
                        scale = (min_required * 1.1 / error.details.get('area')) ** 0.5
                        dims = room.get('dimensions', {})
                        room['dimensions'] = {
                            'width': dims.get('width', 10) * scale,
                            'height': dims.get('height', 10) * scale
                        }
                        print(f"    Fixed: Increased {room_type} area to {room['area']:.0f} sq ft")
            
            # Add missing windows
            elif error.code == 'MISSING_WINDOW':
                room_type = error.details.get('room')
                for room in rooms:
                    if room.get('type') == room_type:
                        if 'windows' not in room:
                            room['windows'] = []
                        room['windows'].append({
                            'position': 'exterior',
                            'width': 4,
                            'height': 5
                        })
                        print(f"    Fixed: Added window to {room_type}")
            
            # Add missing ventilation
            elif error.code == 'MISSING_VENTILATION':
                room_type = error.details.get('room')
                for room in rooms:
                    if room.get('type') == room_type:
                        room['has_exhaust'] = True
                        print(f"    Fixed: Added exhaust fan to {room_type}")
        
        return layout_spec
    
    def _create_generation_prompt(self, text_description: str, layout_spec: Dict) -> str:
        """Create optimized prompt for image generation"""
        
        # Extract key features
        rooms = layout_spec.get('rooms', [])
        room_types = ', '.join(sorted(set(r.get('type', '').replace('_', ' ') for r in rooms)))
        
        # Build comprehensive prompt
        prompt = (
            f"Professional CAD architectural floor plan, "
            f"{text_description}, "
            f"rooms: {room_types}, "
            f"top-down 2D view, black and white line drawing, "
            f"blueprint style, technical drawing with clean straight lines, "
            f"room labels, dimension annotations, door and window symbols, "
            f"wall thickness shown, engineering precision, "
            f"high quality architectural drawing"
        )
        
        return prompt
    
    def _create_summary_report(
        self,
        text_description: str,
        layout_spec: Dict,
        validation_report: Dict,
        image_paths: List[str]
    ) -> Dict:
        """Create comprehensive summary report"""
        
        rooms = layout_spec.get('rooms', [])
        
        return {
            'input': {
                'description': text_description,
                'timestamp': os.popen('date -u +"%Y-%m-%d %H:%M:%S UTC"').read().strip()
            },
            'layout': {
                'total_area': layout_spec.get('total_area'),
                'num_rooms': len(rooms),
                'room_types': [r.get('type') for r in rooms],
                'layout_type': layout_spec.get('layout_type')
            },
            'validation': validation_report,
            'outputs': {
                'num_images': len(image_paths),
                'image_paths': image_paths
            },
            'system': {
                'parser': 'LLM-based (HuggingFace)',
                'validator': 'Rule-based Engineering Validator',
                'generator': 'Fine-tuned Stable Diffusion 1.5',
                'version': '1.0'
            }
        }


def main():
    """Command-line interface for the system"""
    
    parser = argparse.ArgumentParser(
        description="Text-to-CAD Layout Generation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "2BHK apartment with attached bathroom"
  python main.py "3 bedroom house with open kitchen" --variations 5
  python main.py "Compact 1BHK with good ventilation" --model custom_model/
        """
    )
    
    parser.add_argument(
        'description',
        type=str,
        help='Natural language description of desired layout'
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        default='./trained_model/final_model',
        help='Path to trained model (default: ./trained_model/final_model)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./outputs',
        help='Output directory (default: ./outputs)'
    )
    
    parser.add_argument(
        '--variations',
        type=int,
        default=3,
        help='Number of layout variations to generate (default: 3)'
    )
    
    parser.add_argument(
        '--hf_token',
        type=str,
        default=None,
        help='HuggingFace API token (optional, for LLM access)'
    )
    
    parser.add_argument(
        '--llm_model',
        type=str,
        default='qwen',
        choices=['qwen', 'mistral', 'phi', 'llama'],
        help='LLM model to use for parsing (default: qwen)'
    )
    
    parser.add_argument(
        '--no_auto_fix',
        action='store_true',
        help='Disable automatic fixing of validation errors'
    )
    
    args = parser.parse_args()
    
    # Initialize system
    system = TextToCADSystem(
        model_path=args.model_path,
        hf_token=args.hf_token,
        llm_model=args.llm_model,
        auto_fix=not args.no_auto_fix
    )
    
    # Generate layout
    result = system.generate_layout(
        text_description=args.description,
        num_variations=args.variations,
        output_dir=args.output_dir
    )
    
    print("\n✓ Process complete!")
    print(f"Check {args.output_dir} for all outputs")


if __name__ == "__main__":
    main()
