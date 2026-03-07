# Geometric Floor Plan Generation System

## 🎯 Overview

This system generates **geometrically correct, precise CAD floor plans** with accurate dimensions, suitable for professional architectural use.

Unlike image-based generation (which creates artistic representations), this system produces **engineering-grade layouts** with:
- ✅ **Precise dimensions** (measured in feet and inches)
- ✅ **Accurate geometry** (perfect rectangles, aligned walls)
- ✅ **CAD-compatible output** (DXF for AutoCAD/Revit)
- ✅ **Scalable vectors** (SVG for web/graphics)
- ✅ **Dimension annotations** (all measurements labeled)

---

## 🏗️ Architecture

```
Text Description
      ↓
[Layout Parser] - Parse requirements
      ↓
Layout Specification (JSON)
      ↓
[Geometric Generator] - Create precise geometry
      ↓
CAD Output (DXF + SVG)
```

---

## 📦 Files

### Core System (3 files)
1. **`geometric_floor_plan_system.py`** - Main system (use this)
2. **`layout_spec_parser.py`** - Text → Specification parser
3. **`geometric_layout_generator.py`** - Geometry → DXF/SVG exporter

### Dependencies
- `ezdxf` - DXF file generation
- `svgwrite` - SVG file generation
- `numpy` - Mathematical operations

---

## 🚀 Quick Start

### Installation

```bash
pip install ezdxf svgwrite numpy
```

### Generate Your First Floor Plan

```bash
python geometric_floor_plan_system.py "2BHK apartment with attached bathroom"
```

This creates:
- `2BHK_apartment_with_attached_bathroom.dxf` - Open in AutoCAD
- `2BHK_apartment_with_attached_bathroom.svg` - Open in browser
- `layout_specification.json` - Geometric data
- `generation_summary.json` - Statistics

---

## 💡 Usage Examples

### Basic Generation

```bash
# Simple 2BHK
python geometric_floor_plan_system.py "2BHK apartment"

# With features
python geometric_floor_plan_system.py "2BHK apartment with attached bathroom and open kitchen"

# Larger layout
python geometric_floor_plan_system.py "3 bedroom house with separate dining area"

# Compact layout
python geometric_floor_plan_system.py "Compact 1BHK studio apartment"

# Luxury layout
python geometric_floor_plan_system.py "4BHK penthouse with study room and balcony"
```

### Advanced Options

```bash
# Custom output directory
python geometric_floor_plan_system.py "2BHK apartment" --output_dir ./my_plans

# Generate only DXF (no SVG)
python geometric_floor_plan_system.py "2BHK apartment" --formats dxf

# Generate only SVG (no DXF)
python geometric_floor_plan_system.py "2BHK apartment" --formats svg

# Generate from existing specification
python geometric_floor_plan_system.py --spec my_layout.json
```

---

## 🐍 Python API

### Simple Generation

```python
from geometric_floor_plan_system import GeometricFloorPlanSystem

# Create system
system = GeometricFloorPlanSystem()

# Generate floor plan
result = system.generate_from_text(
    description="2BHK apartment with attached bathroom",
    output_dir="./my_outputs",
    output_formats=['dxf', 'svg', 'json']
)

# Access outputs
print("DXF file:", result['output_files']['dxf'])
print("Total area:", result['statistics']['total_area_sqft'], "sq ft")
```

### Custom Dimensions

```python
# Override standard dimensions
custom_dims = {
    'living_room': {'width': 18, 'height': 14},
    'master_bedroom': {'width': 16, 'height': 13},
}

result = system.generate_from_text(
    description="2BHK apartment",
    custom_dimensions=custom_dims
)
```

### Generate from Specification

```python
# Create custom specification
spec = {
    "layout_type": "apartment",
    "rooms": [
        {
            "type": "living_room",
            "label": "Living Room",
            "width": 15,
            "height": 12,
            "position": {"x": 0, "y": 0},
            "doors": [{"x": 0, "y": 6}],
            "windows": [{"x": 7.5, "y": 12}]
        },
        {
            "type": "bedroom",
            "label": "Master Bedroom",
            "width": 14,
            "height": 12,
            "position": {"x": 15.5, "y": 0},
            "doors": [{"x": 15.5, "y": 6}],
            "windows": [{"x": 22.5, "y": 12}]
        }
    ]
}

# Generate from spec
from geometric_layout_generator import GeometricFloorPlanGenerator

generator = GeometricFloorPlanGenerator()
generator.parse_layout_specification(spec)
generator.export_to_dxf("my_custom_plan.dxf")
generator.export_to_svg("my_custom_plan.svg")
```

---

## 📐 Understanding the Output

### DXF File (AutoCAD Format)

The DXF file contains:

**Layers:**
- `WALLS` - Wall outlines with thickness (0.5 ft default)
- `DOORS` - Door symbols (arc + line)
- `WINDOWS` - Window symbols (rectangle with cross)
- `DIMENSIONS` - Dimension lines and measurements
- `ROOM_LABELS` - Room names and areas
- `ANNOTATIONS` - Notes and title block

**Measurements:**
- All dimensions in feet
- 1 DXF unit = 1 foot
- Wall thickness: 6 inches (0.5 ft)
- Door width: 3 feet (standard)
- Window width: 4 feet (standard)

**Opening in AutoCAD:**
1. Open DXF file
2. All layers visible by default
3. Use "LAYER" command to toggle layers
4. Scale is 1:1 (1 unit = 1 foot)

### SVG File (Web/Graphics Format)

The SVG file contains:
- Scalable vector graphics
- Same geometry as DXF
- Can be opened in:
  - Web browsers (Chrome, Firefox, Safari)
  - Adobe Illustrator
  - Inkscape
  - CorelDRAW

### JSON Specification

```json
{
  "layout_type": "apartment",
  "rooms": [
    {
      "type": "living_room",
      "label": "Living Room",
      "width": 15.0,
      "height": 12.0,
      "position": {"x": 0.0, "y": 0.0},
      "doors": [...],
      "windows": [...]
    }
  ]
}
```

---

## 📏 Standard Room Dimensions

| Room Type | Standard Width | Standard Height | Area (sq ft) |
|-----------|----------------|-----------------|--------------|
| Living Room | 15 ft | 12 ft | 180 |
| Master Bedroom | 14 ft | 12 ft | 168 |
| Bedroom | 12 ft | 10 ft | 120 |
| Kitchen | 10 ft | 8 ft | 80 |
| Dining | 12 ft | 10 ft | 120 |
| Bathroom | 7 ft | 5 ft | 35 |
| Toilet | 5 ft | 4 ft | 20 |
| Balcony | 8 ft | 4 ft | 32 |
| Study | 10 ft | 8 ft | 80 |
| Utility | 6 ft | 4 ft | 24 |

These are **default dimensions** and can be customized.

---

## 🔧 Customization

### Modify Standard Dimensions

Edit `layout_spec_parser.py`:

```python
ROOM_DIMENSIONS = {
    'living_room': (18, 14),  # Larger living room
    'master_bedroom': (16, 13),  # Bigger master bedroom
    # ... other rooms
}
```

### Change Wall Thickness

Edit `geometric_layout_generator.py`:

```python
WALL_THICKNESS = 0.67  # 8 inches instead of 6
```

### Adjust Door/Window Sizes

Edit `geometric_layout_generator.py`:

```python
DOOR_WIDTH = 3.5  # 3.5 feet instead of 3
WINDOW_WIDTH = 5.0  # 5 feet instead of 4
```

---

## 🎨 Layout Strategies

### Compact Layout
```bash
python geometric_floor_plan_system.py "Compact 1BHK apartment"
```
- Reduces room dimensions by 15%
- Maximum row width: 20 feet

### Standard Layout
```bash
python geometric_floor_plan_system.py "2BHK apartment"
```
- Standard dimensions
- Maximum row width: 30 feet

### Open Plan
```bash
python geometric_floor_plan_system.py "2BHK with open kitchen"
```
- Kitchen adjacent to living room
- No wall between kitchen and living area

---

## 📊 Output Examples

### Example 1: 2BHK Apartment

**Input:**
```bash
python geometric_floor_plan_system.py "2BHK apartment with attached bathroom"
```

**Output:**
- Living Room: 15' x 12' (180 sq ft)
- Master Bedroom: 14' x 12' (168 sq ft)
- Bedroom 2: 12' x 10' (120 sq ft)
- Kitchen: 10' x 8' (80 sq ft)
- Master Bathroom: 7' x 5' (35 sq ft)
- Bathroom 2: 7' x 5' (35 sq ft)
- **Total: ~618 sq ft**

### Example 2: 3BHK House

**Input:**
```bash
python geometric_floor_plan_system.py "3 bedroom house with dining area"
```

**Output:**
- Living Room: 15' x 12'
- Dining: 12' x 10'
- Master Bedroom: 14' x 12'
- Bedroom 2: 12' x 10'
- Bedroom 3: 12' x 10'
- Kitchen: 10' x 8'
- 2 Bathrooms
- **Total: ~800 sq ft**

---

## 🆚 Comparison: Geometric vs. Diffusion

| Feature | Geometric System | Diffusion Model |
|---------|------------------|-----------------|
| **Accuracy** | ✅ Perfect geometry | ⚠️ Approximate |
| **Dimensions** | ✅ Precise measurements | ❌ No measurements |
| **CAD Ready** | ✅ Import directly | ⚠️ Needs tracing |
| **Customization** | ✅ Full control | ⚠️ Limited |
| **Speed** | ✅ Instant (<1s) | ⚠️ Minutes |
| **Creativity** | ⚠️ Rule-based | ✅ Varied designs |
| **Use Case** | Engineering/Construction | Visualization/Concept |

**Recommendation:** Use **Geometric System** for:
- Construction documents
- Permit applications
- Engineering calculations
- Precise measurements

Use **Diffusion Model** for:
- Initial concepts
- Client presentations
- Design exploration
- Artistic visualization

---

## 🔄 Workflow Integration

### Workflow 1: Text → CAD

```bash
# Generate DXF from text
python geometric_floor_plan_system.py "2BHK apartment" --formats dxf

# Open in AutoCAD
# File → Open → Select .dxf file
# Edit as needed
```

### Workflow 2: Specification → Multiple Formats

```bash
# 1. Generate specification
python geometric_floor_plan_system.py "2BHK apartment" --formats json

# 2. Edit specification JSON (customize dimensions)
# Edit layout_specification.json

# 3. Generate DXF and SVG from edited spec
python geometric_floor_plan_system.py --spec layout_specification.json --formats dxf svg
```

### Workflow 3: Batch Generation

```python
descriptions = [
    "2BHK apartment",
    "3BHK house with study",
    "1BHK compact studio",
]

for desc in descriptions:
    system.generate_from_text(
        description=desc,
        output_dir=f"./plans/{desc.replace(' ', '_')}"
    )
```

---

## 💡 Tips & Best Practices

### 1. Writing Good Descriptions

**Good:**
- "2BHK apartment with attached bathroom and balcony"
- "3 bedroom house with separate dining area"
- "Compact 1BHK with open kitchen"

**Less Effective:**
- "Nice apartment" (too vague)
- "Huge mansion" (no specific requirements)

### 2. Verifying Output

**Check in AutoCAD:**
```
1. Open DXF file
2. Type "DIST" command
3. Click two points to verify dimensions
4. Check layer visibility (type "LAYER")
```

**Check in Browser (SVG):**
1. Open SVG in Chrome/Firefox
2. Right-click → Inspect
3. Zoom in to check details

### 3. Editing Generated Plans

**In AutoCAD:**
- Move rooms: Select walls → MOVE command
- Resize rooms: Select walls → STRETCH command
- Add elements: Draw → Line/Arc tools
- Change layers: Properties panel

---

## 🆘 Troubleshooting

### Issue: "ezdxf not installed"
**Solution:**
```bash
pip install ezdxf svgwrite numpy
```

### Issue: "Rooms overlap in generated plan"
**Solution:** Rooms shouldn't overlap with default settings. If custom positions used, ensure spacing:
```python
# Ensure at least 0.5 ft spacing between rooms
room1_position = {"x": 0, "y": 0}
room2_position = {"x": room1_width + 0.5, "y": 0}
```

### Issue: "DXF file opens incorrectly in AutoCAD"
**Solution:**
1. Check AutoCAD units: Type "UNITS" → Set to "Architectural"
2. Check scale: Type "ZOOM" → "E" (Extents)
3. Verify layers: Type "LAYER" → Ensure all layers are ON

### Issue: "Want to modify room dimensions"
**Solution:** Use custom dimensions:
```python
custom_dims = {
    'living_room': {'width': 20, 'height': 15}
}
system.generate_from_text(description, custom_dimensions=custom_dims)
```

---

## 📚 API Reference

### GeometricFloorPlanSystem

**Methods:**

```python
generate_from_text(
    description: str,
    output_dir: str = "./geometric_outputs",
    output_formats: List[str] = ['dxf', 'svg'],
    save_spec: bool = True,
    custom_dimensions: Optional[Dict] = None
) -> Dict
```

```python
generate_from_specification(
    spec_path: str,
    output_dir: str = "./geometric_outputs",
    output_formats: List[str] = ['dxf', 'svg']
) -> Dict
```

### GeometricFloorPlanGenerator

**Methods:**

```python
parse_layout_specification(spec: Dict) -> None
export_to_dxf(output_path: str, scale: float = 1.0) -> None
export_to_svg(output_path: str, scale: float = 20.0) -> None
```

### LayoutParser

**Methods:**

```python
parse_text_description(description: str) -> Dict
```

---

## 🎓 Next Steps

1. **Generate your first plan** - Try the examples above
2. **Open in AutoCAD** - Verify dimensions and geometry
3. **Customize** - Modify dimensions or create custom specs
4. **Integrate** - Add to your workflow
5. **Extend** - Build on top of the system

---

## 📄 License

This system uses:
- `ezdxf` (MIT License)
- `svgwrite` (MIT License)
- `numpy` (BSD License)

---

**Ready to generate precise floor plans? Start with:**
```bash
python geometric_floor_plan_system.py "2BHK apartment with attached bathroom"
```
