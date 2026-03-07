# Cost-Free LLM-Driven Text-to-CAD Layout Generation System

## System Overview

This system generates CAD-style architectural floor plans from natural language descriptions using only open-source tools and free APIs.

## Architecture Components

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INPUT (Text)                         │
│  "2BHK apartment with attached bathrooms and open kitchen"   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              1. TEXT UNDERSTANDING MODULE                    │
│         (Open-Source LLM via HuggingFace API)               │
│  - Qwen2.5-7B-Instruct / Mistral-7B / Phi-3                 │
│  - Extracts: rooms, dimensions, constraints                  │
│  - Output: Structured JSON                                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│           2. ENGINEERING RULE ENGINE                         │
│  - Validates room sizes, adjacencies                         │
│  - Checks circulation, ventilation                           │
│  - Enforces building code constraints                        │
│  - Output: Validated layout specification                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│         3. LAYOUT GENERATION MODEL                           │
│  - ControlNet + Stable Diffusion (free)                      │
│  - Fine-tuned on CAD floor plan dataset                      │
│  - Generates CAD-style line drawings                         │
│  - Output: Floor plan image                                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│         4. POST-PROCESSING & VALIDATION                      │
│  - OCR/Vision model checks layout correctness                │
│  - Dimension verification                                    │
│  - Auto-correction if needed                                 │
│  - Output: Final CAD layout image                            │
└─────────────────────────────────────────────────────────────┘
```

## Technology Stack

### 1. Text Understanding
- **Primary LLM**: Qwen2.5-7B-Instruct (HuggingFace Inference API - FREE)
- **Backup**: Mistral-7B-Instruct, Phi-3-Mini
- **Framework**: transformers, langchain (optional)

### 2. Rule Engine
- **Language**: Python
- **Libraries**: NetworkX (graph-based constraints), Shapely (geometry validation)
- **Rules Database**: JSON/YAML configuration files

### 3. Image Generation
- **Model**: Stable Diffusion 1.5 + ControlNet
- **Fine-tuning**: LoRA adapters on floor plan dataset
- **Platform**: HuggingFace Diffusers (local) or Inference API
- **Alternative**: SDXL-Turbo for faster generation

### 4. Validation
- **Vision Model**: CLIP, LayoutLM, or fine-tuned CNN
- **Geometry Check**: OpenCV, PIL

## Data Requirements

### Training Dataset Structure
```
dataset/
├── images/              # Floor plan images (PNG/JPG)
│   ├── 001.png
│   ├── 002.png
│   └── ...
├── labels/              # Structured annotations (JSON)
│   ├── 001.json
│   ├── 002.json
│   └── ...
└── prompts/             # Text descriptions (TXT)
    ├── 001.txt
    ├── 002.txt
    └── ...
```

### Label Format (JSON)
```json
{
  "layout_id": "001",
  "total_area": 1200,
  "rooms": [
    {
      "type": "living_room",
      "area": 300,
      "dimensions": {"width": 15, "height": 20},
      "position": {"x": 0, "y": 0},
      "doors": [{"position": "north", "width": 3}],
      "windows": [{"position": "west", "width": 4}]
    },
    {
      "type": "bedroom",
      "area": 200,
      "dimensions": {"width": 10, "height": 20},
      "position": {"x": 15, "y": 0},
      "attached_bathroom": true
    }
  ],
  "connections": [
    {"from": "living_room", "to": "bedroom", "type": "door"},
    {"from": "bedroom", "to": "bathroom", "type": "door"}
  ],
  "prompt": "2BHK apartment with attached bathroom, open kitchen connected to living room"
}
```

## Engineering Rules

### Room Size Standards (in sq ft)
- Living Room: min 150, ideal 200-300
- Master Bedroom: min 120, ideal 150-200
- Bedroom: min 100, ideal 120-150
- Kitchen: min 60, ideal 80-120
- Bathroom: min 30, ideal 40-60
- Balcony: min 30, ideal 40-80

### Structural Rules
1. Wall thickness: 6-9 inches (standard), 12 inches (load-bearing)
2. Door width: 2.5-3 feet (standard), 4 feet (main entrance)
3. Window dimensions: min 3x4 feet
4. Corridor width: min 3 feet
5. Ventilation: Every room needs window/exhaust
6. Natural light: Living rooms and bedrooms need windows

### Adjacency Constraints
- Kitchen should not directly connect to bedrooms
- Bathrooms should be accessible from bedrooms/corridor
- Living room typically central/near entrance
- Balconies attached to living room or bedrooms

## Implementation Phases

### Phase 1: Data Preparation ✓
- Collect/annotate floor plan dataset
- Create structured labels
- Generate text descriptions

### Phase 2: Rule Engine Development ✓
- Implement validation rules
- Create constraint solver
- Build layout graph system

### Phase 3: LLM Integration
- Set up HuggingFace API
- Fine-tune prompt templates
- Parse text to structured format

### Phase 4: Image Generation
- Fine-tune Stable Diffusion on dataset
- Train ControlNet for layout control
- Optimize for CAD-style output

### Phase 5: Integration & Testing
- Connect all modules
- Build feedback loop
- Create evaluation metrics

## Expected Output Quality

✅ **Structural Correctness**: All rooms properly sized and positioned
✅ **Connectivity**: Valid door/corridor placement
✅ **Compliance**: Meets basic building standards
✅ **Clarity**: Clear CAD-style line drawings
✅ **Dimensions**: Accurate proportions and measurements

## Limitations

⚠️ Does not replace professional architectural review
⚠️ Generic rules (not jurisdiction-specific codes)
⚠️ Best for residential layouts (commercial may need adjustments)
⚠️ Requires quality training data for best results

## Next Steps

1. Set up development environment
2. Prepare training dataset
3. Implement and test each module
4. Train/fine-tune models
5. Integrate and deploy
