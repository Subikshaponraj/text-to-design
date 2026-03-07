# Text-to-CAD Layout Generation System 🏗️

A cost-free, open-source system that generates professional CAD-style architectural layouts from natural language descriptions.

## 🎯 Features

- **Natural Language Input**: Describe your layout in plain English
- **LLM-Powered Parsing**: Uses open-source LLMs (Qwen, Mistral, Phi) via HuggingFace
- **Engineering Validation**: Enforces building codes and architectural standards
- **CAD-Style Output**: Generates professional blueprint-style floor plans
- **100% Free**: No paid APIs required
- **Open Source**: Fully transparent and customizable

## 📋 Requirements

### System Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended, 8GB+ VRAM) or CPU
- 20GB disk space for models
- 16GB RAM minimum

### Python Dependencies
```bash
pip install torch torchvision
pip install diffusers transformers accelerate
pip install peft opencv-python pillow requests tqdm
```

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate training data
python dataset_generator.py

# 3. Train model
python train_model.py --epochs 50

# 4. Generate layouts
python main.py "2BHK apartment with attached bathroom"
```

## 📖 Usage Examples

```bash
# Simple
python main.py "2BHK apartment with attached bathroom"

# Detailed
python main.py "3 bedroom house with open kitchen, dining area, and balcony" --variations 5

# Custom model
python main.py "Modern 1BHK" --model_path ./my_model
```

## 📁 Key Files

- `main.py` - Main entry point
- `llm_parser.py` - Text parsing with LLM
- `rule_engine.py` - Engineering validation
- `train_model.py` - Model training
- `inference.py` - Image generation
- `dataset_generator.py` - Dataset creation

## 🎨 Output

Each generation produces:
- Multiple CAD-style floor plan images
- Parsed specification JSON
- Validation report
- Generation summary

## 🔧 Configuration

```bash
# Use HuggingFace token (optional, for better LLM access)
python main.py "..." --hf_token YOUR_TOKEN

# Choose LLM model
python main.py "..." --llm_model qwen  # or mistral, phi
```

## 📊 How It Works

```
Input Text → LLM Parser → Rule Engine → Image Generator → Output CAD Plans
```

1. **LLM Parser**: Extracts rooms, sizes, connections from text
2. **Rule Engine**: Validates against building standards
3. **Image Generator**: Creates CAD-style floor plans using fine-tuned Stable Diffusion
4. **Post-Processing**: Enhances clarity and format

## ⚠️ Limitations

- Educational/research tool, not for production use
- Generic building rules (not jurisdiction-specific)
- Best for residential layouts
- Requires professional review
- 2D plans only

## 📚 Documentation

See `SYSTEM_ARCHITECTURE.md` for complete technical details.

## 🤝 Contributing

Contributions welcome! Focus areas:
- Dataset quality
- Validation rules
- Performance optimization
- UI development
