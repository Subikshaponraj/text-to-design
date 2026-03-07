# PNG Floor Plan Generation - Quick Start

## 🖼️ Generate Geometrically Correct Floor Plans as PNG Images

This system generates **high-quality PNG images** of floor plans with:
- ✅ **Precise measurements** labeled on the image
- ✅ **Perfect geometry** (exact rectangles, aligned walls)
- ✅ **Dimension annotations** (all room sizes shown)
- ✅ **Room labels and areas** displayed
- ✅ **Professional appearance** (300 DPI, print-ready)

---

## 🚀 Quick Start (2 Commands)

```bash
# 1. Install dependencies
pip install numpy ezdxf svgwrite Pillow

# 2. Generate PNG floor plan
python geometric_floor_plan_system.py "2BHK apartment with attached bathroom"
```

**Output:** Creates `2BHK_apartment_with_attached_bathroom.png` with:
- Floor plan layout with walls, doors, windows
- All measurements labeled (in feet)
- Room names and areas
- Grid background
- Title block with total area

---

## 📐 What You Get in the PNG

### Features Shown:
1. **Walls** - Black outlines with thickness
2. **Doors** - Brown arc symbols showing swing direction
3. **Windows** - Blue rectangles on exterior walls
4. **Room Labels** - Name of each room (center)
5. **Dimensions** - Width and height in feet (green lines)
6. **Room Areas** - Square footage for each room
7. **Grid** - Background grid (5-foot squares)
8. **Title Block** - Total area and room count

### Image Specifications:
- **Resolution**: 2400 x 2400 pixels (default)
- **DPI**: 300 (print quality)
- **Format**: PNG (lossless)
- **File Size**: ~500KB - 2MB
- **Scale**: ~40 pixels per foot (customizable)

---

## 💡 Usage Examples

### Basic PNG Generation

```bash
# Simple 2BHK with all features
python geometric_floor_plan_system.py "2BHK apartment"

# Output: 2BHK_apartment.png with measurements
```

### Generate Only PNG (No DXF/SVG)

```bash
python geometric_floor_plan_system.py "2BHK apartment" --formats png
```

### Generate All Formats (PNG + DXF + SVG)

```bash
python geometric_floor_plan_system.py "3 bedroom house" --formats png dxf svg
```

### Custom Output Directory

```bash
python geometric_floor_plan_system.py "2BHK apartment" \
  --output_dir ./my_floor_plans \
  --formats png
```

---

## 🐍 Python API

### Simple PNG Generation

```python
from geometric_floor_plan_system import GeometricFloorPlanSystem

# Create system
system = GeometricFloorPlanSystem()

# Generate PNG
result = system.generate_from_text(
    description="2BHK apartment with attached bathroom",
    output_dir="./outputs",
    output_formats=['png']  # PNG only
)

# Access output
print("PNG file:", result['output_files']['png'])
```

### Custom PNG Settings

```python
from enhanced_png_renderer import EnhancedGeometricFloorPlanGenerator

# Create generator
generator = EnhancedGeometricFloorPlanGenerator()

# Parse layout
layout_spec = {
    "layout_type": "apartment",
    "rooms": [
        {
            "type": "living_room",
            "width": 15,
            "height": 12,
            "position": {"x": 0, "y": 0},
            "label": "Living Room",
            "doors": [{"x": 0, "y": 6}],
            "windows": [{"x": 7.5, "y": 12}]
        }
    ]
}

generator.parse_layout_specification(layout_spec)

# Export PNG with custom settings
generator.export_to_png(
    "my_floor_plan.png",
    width=3000,              # Higher resolution
    height=3000,
    dpi=300,                 # Print quality
    show_dimensions=True,    # Show dimension lines
    show_grid=True,          # Show background grid
    show_room_areas=True     # Show room areas
)
```

---

## 🎨 Customization Options

### PNG Export Parameters

```python
generator.export_to_png(
    output_path="floor_plan.png",
    
    # Image size
    width=2400,              # Default: 2400 pixels
    height=2400,             # Default: 2400 pixels
    dpi=300,                 # Default: 300 (print quality)
    
    # Visual options
    show_dimensions=True,    # Show dimension lines (green)
    show_grid=True,          # Show background grid
    show_room_areas=True     # Show sq ft areas
)
```

### Resolution Options

| Purpose | Width/Height | DPI | File Size |
|---------|--------------|-----|-----------|
| Web Display | 1200 | 72 | ~200KB |
| Standard Print | 2400 | 300 | ~800KB |
| Large Print | 3600 | 300 | ~1.5MB |
| Professional | 4800 | 600 | ~3MB |

### Example: High-Resolution Print

```python
generator.export_to_png(
    "print_quality.png",
    width=4800,
    height=4800,
    dpi=600
)
```

---

## 📊 Sample Outputs

### Example 1: 2BHK Apartment

**Command:**
```bash
python geometric_floor_plan_system.py "2BHK apartment with attached bathroom"
```

**PNG Contains:**
- Living Room: 15' × 12' (180 sq ft)
- Master Bedroom: 14' × 12' (168 sq ft)
- Bedroom 2: 12' × 10' (120 sq ft)
- Kitchen: 10' × 8' (80 sq ft)
- Master Bathroom: 7' × 5' (35 sq ft)
- Bathroom 2: 7' × 5' (35 sq ft)

**Total: 618 sq ft** (shown in title)

Each room labeled with:
- Name
- Dimensions (width × height)
- Area in sq ft
- Green dimension lines showing measurements

---

## 🖨️ Using the PNG

### For Print
1. **Open** in image viewer
2. **Print** settings: 
   - 300 DPI = 8" × 8" at actual size
   - Scale as needed
3. **Quality**: Professional print ready

### For Presentations
1. **Insert** into PowerPoint/Google Slides
2. **Resize** as needed (vector-quality scaling)
3. **Share** via email or cloud storage

### For Web
1. **Optimize** if needed (reduce to 1200px for web)
2. **Embed** in website or documentation
3. **Share** online easily

### For Editing
1. **Open** in GIMP/Photoshop
2. **Add** annotations or colors
3. **Export** in needed format

---

## 🔧 Troubleshooting

### Issue: "Pillow not installed"
**Solution:**
```bash
pip install Pillow
```

### Issue: "Font not found" warning
**Solution:** System will use default font. To use custom fonts:
```python
# Edit enhanced_png_renderer.py, change font path:
font = ImageFont.truetype("/path/to/your/font.ttf", size)
```

### Issue: Image looks small
**Solution:** Increase resolution:
```bash
# Edit geometric_floor_plan_system.py, change:
self.generator.export_to_png(png_path, width=3600, height=3600)
```

### Issue: Want to hide grid
**Solution:** Use Python API:
```python
generator.export_to_png("plan.png", show_grid=False)
```

### Issue: Dimensions overlap
**Solution:** This happens with very small rooms. Increase layout size or hide dimensions:
```python
generator.export_to_png("plan.png", show_dimensions=False)
```

---

## 📐 Measurement Accuracy

### Precision
- All dimensions calculated from **exact geometry**
- Measurements displayed to **0.1 foot accuracy**
- Room areas calculated mathematically, not approximated
- Scale maintained throughout entire image

### Verification
To verify measurements in PNG:
1. Measure pixel distance between points
2. Divide by scale (default: 40 pixels per foot)
3. Compare to labeled dimension

Example:
- If dimension shows "15.0'"
- And pixel distance is ~600 pixels
- Then: 600 / 40 = 15 feet ✓

---

## 💡 Tips & Best Practices

### 1. Choose Right Resolution

**For screen display only:**
```python
generator.export_to_png("plan.png", width=1200, height=1200, dpi=72)
```

**For printing:**
```python
generator.export_to_png("plan.png", width=2400, height=2400, dpi=300)
```

### 2. Optimize File Size

**Smaller files (for email):**
- Use lower resolution (1200px)
- Use lower DPI (150)

**Print quality (for clients):**
- Use high resolution (2400px+)
- Use high DPI (300)

### 3. Multiple Variations

Generate the same layout in different formats:
```bash
# All at once
python geometric_floor_plan_system.py "2BHK apartment" \
  --formats png dxf svg

# Or separately
python geometric_floor_plan_system.py "2BHK apartment" --formats png
python geometric_floor_plan_system.py "2BHK apartment" --formats dxf
```

---

## 🆚 PNG vs DXF vs SVG

| Feature | PNG | DXF | SVG |
|---------|-----|-----|-----|
| **Use Case** | Presentations, Print | CAD Editing | Web, Scaling |
| **Editable** | No | Yes | Partial |
| **File Size** | Medium | Small | Small |
| **Quality** | Pixel-based | Vector | Vector |
| **Measurements** | ✅ Shown | ✅ Editable | ✅ Shown |
| **Best For** | Final output | Engineering | Graphics |

**Recommendation:** Generate all three formats for maximum flexibility!

---

## ✅ Checklist

Before sharing PNG floor plans:

- [ ] Check all room labels are visible
- [ ] Verify all dimensions are shown correctly
- [ ] Ensure total area is calculated properly
- [ ] Confirm image quality (zoom in to check clarity)
- [ ] Test print output if needed
- [ ] Verify file size is appropriate for use case

---

## 🎯 Next Steps

1. **Generate your first PNG:**
   ```bash
   python geometric_floor_plan_system.py "2BHK apartment" --formats png
   ```

2. **Open and verify** the generated PNG

3. **Customize** as needed using Python API

4. **Share or print** your floor plan

---

**Ready to create precise floor plan images?**

```bash
python geometric_floor_plan_system.py "2BHK apartment with attached bathroom"
```

Check `./geometric_outputs/` for your PNG file with measurements!
