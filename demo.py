"""
Quick Demo Script for Text-to-CAD System
Demonstrates the complete pipeline without requiring trained model
"""

import json
from llm_parser import LLMTextParser
from rule_engine import LayoutValidator
from dataset_generator import FloorPlanDatasetGenerator

def run_demo():
    """
    Run a quick demonstration of the system components
    This works without a trained model - shows parsing and validation
    """
    
    print("="*80)
    print("TEXT-TO-CAD SYSTEM DEMONSTRATION")
    print("="*80)
    
    # Sample text descriptions
    test_cases = [
        "2BHK apartment with attached bathroom in master bedroom and open kitchen",
        "3 bedroom house with separate dining area, modular kitchen, and balcony",
        "Compact 1BHK with modern layout and good ventilation",
    ]
    
    for i, description in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"TEST CASE {i}: {description}")
        print(f"{'='*80}\n")
        
        # Step 1: Parse with LLM
        print("STEP 1: LLM Text Parsing")
        print("-" * 40)
        
        # Note: This will use HuggingFace free API (rate-limited)
        # For better results, get a free token at https://huggingface.co/settings/tokens
        parser = LLMTextParser(hf_token=None, model='qwen')
        
        try:
            layout_spec = parser.parse_architectural_requirements(description)
            print("✓ Successfully parsed requirements")
            print(f"\nParsed Layout:")
            print(json.dumps(layout_spec, indent=2))
        except Exception as e:
            print(f"⚠ LLM parsing failed (likely rate limit): {e}")
            print("Using fallback structure...")
            layout_spec = parser._get_fallback_structure()
        
        # Step 2: Convert to detailed specification
        print("\n\nSTEP 2: Convert to Detailed Specification")
        print("-" * 40)
        detailed_spec = parser.convert_to_detailed_specification(layout_spec)
        print("✓ Created detailed specification")
        
        # Step 3: Validate
        print("\n\nSTEP 3: Engineering Validation")
        print("-" * 40)
        validator = LayoutValidator()
        is_valid, errors = validator.validate_layout(detailed_spec)
        
        print(f"Validation Status: {'✓ VALID' if is_valid else '⚠ HAS ISSUES'}")
        print(f"Critical Errors: {len([e for e in errors if e.severity == 'critical'])}")
        print(f"Warnings: {len([e for e in errors if e.severity == 'warning'])}")
        
        if errors:
            print("\nValidation Details:")
            for error in errors[:5]:  # Show first 5
                print(f"  [{error.severity.upper()}] {error.code}: {error.message}")
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more")
        
        # Step 4: Generate synthetic floor plan image
        print("\n\nSTEP 4: Generate Synthetic Floor Plan")
        print("-" * 40)
        generator = FloorPlanDatasetGenerator(output_dir=f'./demo_output/case_{i}')
        
        try:
            floor_plan, control, prompt = generator.save_training_sample(
                detailed_spec, 
                f"demo_{i}"
            )
            print(f"✓ Generated floor plan image")
            print(f"✓ Saved to: ./demo_output/case_{i}/")
            print(f"\nGenerated prompt: {prompt}")
        except Exception as e:
            print(f"⚠ Image generation failed: {e}")
        
        print(f"\n{'='*80}\n")
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    print("\nWhat was demonstrated:")
    print("1. ✓ LLM-based text parsing (HuggingFace free API)")
    print("2. ✓ Layout specification generation")
    print("3. ✓ Engineering rule validation")
    print("4. ✓ Synthetic floor plan image generation")
    print("\nWhat's NOT demonstrated (requires training):")
    print("  - Fine-tuned Stable Diffusion model")
    print("  - High-quality CAD-style image generation")
    print("  - End-to-end trained pipeline")
    print("\nTo use the full system:")
    print("  1. Prepare dataset: python dataset_generator.py")
    print("  2. Train model: python train_model.py --epochs 50")
    print("  3. Generate: python main.py 'your description'")
    print("\nDemo outputs saved in: ./demo_output/")
    print("="*80 + "\n")


def demo_rule_engine_only():
    """Demonstrate just the rule engine with predefined layouts"""
    
    print("\n" + "="*80)
    print("RULE ENGINE DEMONSTRATION")
    print("="*80 + "\n")
    
    # Sample layouts to validate
    layouts = {
        "Valid 2BHK": {
            "total_area": 1200,
            "rooms": [
                {
                    "type": "living_room",
                    "area": 250,
                    "dimensions": {"width": 12, "height": 20},
                    "windows": [{"width": 4, "height": 5}],
                    "doors": [{"position": "north", "width": 3}]
                },
                {
                    "type": "master_bedroom",
                    "area": 180,
                    "dimensions": {"width": 12, "height": 15},
                    "windows": [{"width": 4, "height": 5}],
                    "doors": [{"position": "west", "width": 2.5}]
                },
                {
                    "type": "bedroom",
                    "area": 150,
                    "dimensions": {"width": 10, "height": 15},
                    "windows": [{"width": 4, "height": 5}],
                    "doors": [{"position": "west", "width": 2.5}]
                },
                {
                    "type": "kitchen",
                    "area": 100,
                    "dimensions": {"width": 8, "height": 12},
                    "windows": [{"width": 3, "height": 4}],
                    "has_exhaust": True
                }
            ]
        },
        "Invalid - Room Too Small": {
            "total_area": 800,
            "rooms": [
                {
                    "type": "living_room",
                    "area": 100,  # Too small!
                    "dimensions": {"width": 8, "height": 12},
                    "windows": []  # Missing window!
                },
                {
                    "type": "bedroom",
                    "area": 80,  # Too small!
                    "dimensions": {"width": 8, "height": 10}
                }
            ]
        }
    }
    
    validator = LayoutValidator()
    
    for name, layout in layouts.items():
        print(f"\nValidating: {name}")
        print("-" * 60)
        
        is_valid, errors = validator.validate_layout(layout)
        
        print(f"Result: {'✓ VALID' if is_valid else '✗ INVALID'}")
        print(f"Errors: {len([e for e in errors if e.severity == 'critical'])}")
        print(f"Warnings: {len([e for e in errors if e.severity == 'warning'])}")
        
        if errors:
            for error in errors:
                print(f"  [{error.severity.upper()}] {error.message}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--rules-only':
        demo_rule_engine_only()
    else:
        print("\nThis demo shows the system components WITHOUT requiring a trained model.")
        print("For the full system, you need to train the Stable Diffusion model first.\n")
        
        try:
            run_demo()
        except KeyboardInterrupt:
            print("\n\nDemo interrupted by user")
        except Exception as e:
            print(f"\n\nDemo failed with error: {e}")
            print("This is normal if you don't have internet access for the LLM API")
            print("\nTrying rule engine demo instead...\n")
            demo_rule_engine_only()
