"""
Complete Geometric Floor Plan Generation System
Generates geometrically precise CAD layouts from text descriptions
Exports to DXF and SVG with accurate dimensions
"""

import os
import json
from typing import Dict, List, Optional
import argparse

from layout_spec_parser import LayoutParser, create_layout_from_description
from geometric_layout_generator import GeometricFloorPlanGenerator
from constraint_engine import ConstraintEngine, validate_layout
try:
    from enhanced_png_renderer import EnhancedGeometricFloorPlanGenerator
    HAS_PNG_SUPPORT = True
except ImportError:
    HAS_PNG_SUPPORT = False
    EnhancedGeometricFloorPlanGenerator = GeometricFloorPlanGenerator


class GeometricFloorPlanSystem:
    """
    Complete system for generating geometrically correct floor plans
    
    Pipeline:
    1. Parse text description → Layout specification
    2. Create geometric layout with precise dimensions
    3. Export to DXF (AutoCAD) and SVG
    """
    
    def __init__(self):
        self.parser = LayoutParser()
        if HAS_PNG_SUPPORT:
            self.generator = EnhancedGeometricFloorPlanGenerator()
        else:
            self.generator = GeometricFloorPlanGenerator()
            print("Warning: PNG support not available. Install Pillow: pip install Pillow")
    
    def generate_from_text(
        self,
        description: str,
        output_dir: str = "./geometric_outputs",
        output_formats: List[str] = ['dxf', 'svg'],
        save_spec: bool = True,
        custom_dimensions: Optional[Dict] = None
    ) -> Dict:
        """
        Generate geometric floor plan from text description
        
        Args:
            description: Natural language description (e.g., "2BHK apartment")
            output_dir: Output directory
            output_formats: List of formats ('dxf', 'svg', 'json')
            save_spec: Save intermediate specification JSON
            custom_dimensions: Optional dict to override room dimensions
            
        Returns:
            Dictionary with output paths and metadata
        """
        
        print(f"\n{'='*80}")
        print(f"Generating Geometric Floor Plan")
        print(f"{'='*80}")
        print(f"Description: {description}")
        print(f"{'='*80}\n")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Parse description to layout specification
        print("STEP 1: Parsing Description")
        print("-" * 40)
        
        layout_spec = self.parser.parse_text_description(description)
        
        # Apply custom dimensions if provided
        if custom_dimensions:
            self._apply_custom_dimensions(layout_spec, custom_dimensions)
        
        print(f"✓ Parsed layout:")
        print(f"  - Total rooms: {len(layout_spec['rooms'])}")
        for room in layout_spec['rooms']:
            print(f"    • {room['label']}: {room['width']}' x {room['height']}'")
        
        # Step 2: CONSTRAINT ENGINE - Validate and correct layout
        print("\nSTEP 2: Constraint Engine - Validating Layout")
        print("-" * 40)
        
        engine = ConstraintEngine(
            max_iterations=10,
            auto_correct=True,
            strict_mode=False
        )
        
        is_valid, corrected_spec, violations = engine.validate_and_correct(layout_spec)
        
        # Update specification with corrections
        layout_spec = corrected_spec
        
        # Create validation report
        validation_report = {
            'is_valid': is_valid,
            'total_violations': len(violations),
            'critical': sum(1 for v in violations if v.severity.value == 'critical'),
            'errors': sum(1 for v in violations if v.severity.value == 'error'),
            'warnings': sum(1 for v in violations if v.severity.value == 'warning'),
            'corrections_applied': engine.corrections_applied,
            'violations': [
                {
                    'type': v.type.value,
                    'severity': v.severity.value,
                    'message': v.message,
                    'affected_rooms': v.affected_rooms
                }
                for v in violations
            ]
        }
        
        # Save validation report
        if 'json' in output_formats:
            validation_path = os.path.join(output_dir, f"{description.replace(' ', '_')[:40]}_validation.json")
            with open(validation_path, 'w') as f:
                json.dump(validation_report, f, indent=2)
            print(f"✓ Validation report saved: {validation_path}")
        
        # Save corrected specification
        spec_path = None
        if save_spec or 'json' in output_formats:
            spec_path = os.path.join(output_dir, "layout_specification.json")
            with open(spec_path, 'w') as f:
                json.dump(layout_spec, f, indent=2)
            print(f"✓ Corrected specification saved: {spec_path}")
        
        # Step 3: Create geometric layout
        print("\nSTEP 3: Creating Geometric Layout")
        print("-" * 40)
        
        self.generator.parse_layout_specification(layout_spec)
        
        total_area = sum(room.area for room in self.generator.rooms)
        print(f"✓ Geometric layout created:")
        print(f"  - Total area: {total_area:.0f} sq ft")
        print(f"  - Total walls: {len(self.generator.walls)}")
        
        # Step 4: Export to requested formats
        print(f"\nSTEP 4: Exporting to Formats {output_formats}")
        print("-" * 40)
        
        output_paths = {}
        base_name = description.replace(' ', '_')[:40]
        
        if 'png' in output_formats:
            png_path = os.path.join(output_dir, f"{base_name}.png")
            if hasattr(self.generator, 'export_to_png'):
                self.generator.export_to_png(
                    png_path,
                    width=2400,
                    height=2400,
                    dpi=300,
                    show_dimensions=True,
                    show_grid=True,
                    show_room_areas=True
                )
                output_paths['png'] = png_path
            else:
                print("  ⚠ PNG export not available (Pillow not installed)")
        
        if 'dxf' in output_formats:
            dxf_path = os.path.join(output_dir, f"{base_name}.dxf")
            self.generator.export_to_dxf(dxf_path, scale=1.0)
            output_paths['dxf'] = dxf_path
        
        if 'svg' in output_formats:
            svg_path = os.path.join(output_dir, f"{base_name}.svg")
            self.generator.export_to_svg(svg_path, scale=20.0)
            output_paths['svg'] = svg_path
        
        if spec_path:
            output_paths['json'] = spec_path
        
        # Step 5: Create summary
        print("\nSTEP 5: Creating Summary")
        print("-" * 40)
        
        summary = {
            'description': description,
            'layout_spec': layout_spec,
            'validation': validation_report,
            'output_files': output_paths,
            'statistics': {
                'total_rooms': len(self.generator.rooms),
                'total_area_sqft': total_area,
                'total_walls': len(self.generator.walls),
                'room_breakdown': [
                    {
                        'label': room.label,
                        'type': room.type.value,
                        'width_ft': room.width,
                        'height_ft': room.height,
                        'area_sqft': room.area
                    }
                    for room in self.generator.rooms
                ]
            }
        }
        
        summary_path = os.path.join(output_dir, "generation_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Summary saved: {summary_path}")
        
        # Final summary
        print(f"\n{'='*80}")
        print("GENERATION COMPLETE")
        print(f"{'='*80}")
        print(f"Total Rooms: {len(self.generator.rooms)}")
        print(f"Total Area: {total_area:.0f} sq ft")
        print(f"Output Directory: {output_dir}")
        print(f"\nGenerated Files:")
        for format_name, file_path in output_paths.items():
            print(f"  - {format_name.upper()}: {file_path}")
        print(f"{'='*80}\n")
        
        return summary
    
    def _apply_custom_dimensions(self, layout_spec: Dict, custom_dims: Dict):
        """Apply custom room dimensions"""
        
        for room in layout_spec['rooms']:
            room_type = room['type']
            if room_type in custom_dims:
                dims = custom_dims[room_type]
                if 'width' in dims:
                    room['width'] = dims['width']
                if 'height' in dims:
                    room['height'] = dims['height']
    
    def generate_from_specification(
        self,
        spec_path: str,
        output_dir: str = "./geometric_outputs",
        output_formats: List[str] = ['dxf', 'svg']
    ) -> Dict:
        """
        Generate from existing layout specification JSON
        
        Args:
            spec_path: Path to layout specification JSON file
            output_dir: Output directory
            output_formats: List of formats to export
            
        Returns:
            Dictionary with output paths
        """
        
        # Load specification
        with open(spec_path, 'r') as f:
            layout_spec = json.load(f)
        
        print(f"Loaded specification from: {spec_path}")
        
        # Create geometric layout
        self.generator.parse_layout_specification(layout_spec)
        
        # Export
        os.makedirs(output_dir, exist_ok=True)
        output_paths = {}
        
        if 'dxf' in output_formats:
            dxf_path = os.path.join(output_dir, "floor_plan.dxf")
            self.generator.export_to_dxf(dxf_path)
            output_paths['dxf'] = dxf_path
        
        if 'svg' in output_formats:
            svg_path = os.path.join(output_dir, "floor_plan.svg")
            self.generator.export_to_svg(svg_path)
            output_paths['svg'] = svg_path
        
        return output_paths


def main():
    """Command-line interface"""
    
    parser = argparse.ArgumentParser(
        description="Geometric Floor Plan Generator - Create precise CAD layouts from text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate from text description
  python geometric_floor_plan_system.py "2BHK apartment with attached bathroom"
  
  # Generate with custom output directory
  python geometric_floor_plan_system.py "3 bedroom house" --output_dir ./my_plans
  
  # Generate only DXF
  python geometric_floor_plan_system.py "1BHK compact" --formats dxf
  
  # Generate from existing specification
  python geometric_floor_plan_system.py --spec layout_spec.json
        """
    )
    
    parser.add_argument(
        'description',
        type=str,
        nargs='?',
        help='Text description of floor plan (e.g., "2BHK apartment")'
    )
    
    parser.add_argument(
        '--spec',
        type=str,
        help='Path to existing layout specification JSON'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./geometric_outputs',
        help='Output directory (default: ./geometric_outputs)'
    )
    
    parser.add_argument(
        '--formats',
        nargs='+',
        default=['png', 'dxf', 'svg', 'json'],
        choices=['png', 'dxf', 'svg', 'json'],
        help='Output formats (default: png dxf svg json)'
    )
    
    parser.add_argument(
        '--no_spec',
        action='store_true',
        help='Do not save specification JSON'
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not args.description and not args.spec:
        parser.error("Either provide a description or --spec file")
    
    # Initialize system
    system = GeometricFloorPlanSystem()
    
    # Generate
    if args.spec:
        # Generate from specification
        system.generate_from_specification(
            spec_path=args.spec,
            output_dir=args.output_dir,
            output_formats=args.formats
        )
    else:
        # Generate from text
        system.generate_from_text(
            description=args.description,
            output_dir=args.output_dir,
            output_formats=args.formats,
            save_spec=not args.no_spec
        )
    
    print("\n✓ Generation complete!")
    print(f"Check {args.output_dir} for outputs")


if __name__ == "__main__":
    main()