# Complete Setup Guide

## Step-by-Step Installation and Usage

### Prerequisites

1. **System Requirements:**
   - Ubuntu 20.04+ / Windows 10+ / MacOS 11+
   - Python 3.8, 3.9, 3.10, or 3.11
   - 16GB RAM minimum (32GB recommended)
   - GPU with 8GB+ VRAM (for training)
   - 20GB free disk space

2. **Check Python Version:**
   ```bash
   python --version  # Should be 3.8+
   ```

### Installation Steps

#### 1. Set Up Python Environment (Recommended)

```bash
# Create virtual environment
python -m venv cad_env

# Activate it
# On Linux/Mac:
source cad_env/bin/activate
# On Windows:
cad_env\Scripts\activate
```

#### 2. Install PyTorch

Choose based on your system:

**With CUDA (NVIDIA GPU):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**CPU Only (slower training):**
```bash
pip install torch torchvision
```

**Verify Installation:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

#### 3. Install Other Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- diffusers (Stable Diffusion)
- transformers (LLMs)
- accelerate (training optimization)
- peft (LoRA fine-tuning)
- opencv-python (image processing)
- And more...

#### 4. Verify Installation

```bash
python -c "from diffusers import StableDiffusionPipeline; print('✓ All imports successful')"
```

### Quick Start Workflow

#### Method 1: Quick Demo (No Training Required)

```bash
# Run demo to see system components
python demo.py
```

This demonstrates:
- LLM text parsing
- Engineering validation
- Synthetic image generation

#### Method 2: Full System (With Training)

**Step 1: Generate Training Data**

```bash
# Generate 50 synthetic floor plans
python dataset_generator.py
```

Creates `./floor_plan_dataset/` with:
- images/ (floor plan PNGs)
- controls/ (edge maps)
- labels/ (JSON specifications)
- prompts/ (text descriptions)

**Step 2: Train the Model**

```bash
# Basic training (recommended for first try)
python train_model.py --epochs 50 --batch_size 4

# Advanced (if you have good GPU)
python train_model.py --epochs 100 --batch_size 8 --learning_rate 5e-5
```

Training time:
- RTX 4090: ~1-2 hours (50 epochs)
- RTX 3090: ~2-3 hours (50 epochs)
- RTX 3060: ~4-6 hours (50 epochs)
- CPU: Not recommended

Monitor training:
- Loss should decrease over time
- Model checkpoints saved every 10 epochs

**Step 3: Generate Layouts**

```bash
# Generate from text
python main.py "2BHK apartment with attached bathroom and open kitchen"

# With more variations
python main.py "3 bedroom house with balcony" --variations 5

# Custom output directory
python main.py "Modern 1BHK" --output_dir my_layouts/
```

### Advanced Configuration

#### Using HuggingFace API Token (Recommended)

1. Create free account: https://huggingface.co
2. Get token: https://huggingface.co/settings/tokens
3. Use token:
   ```bash
   python main.py "..." --hf_token YOUR_TOKEN_HERE
   ```

Benefits:
- Higher rate limits
- Better LLM access
- More consistent results

#### Choosing LLM Model

```bash
# Qwen (recommended - best structured output)
python main.py "..." --llm_model qwen

# Mistral (good balance)
python main.py "..." --llm_model mistral

# Phi (fastest, smaller)
python main.py "..." --llm_model phi
```

#### Custom Training Dataset

1. Organize your data:
   ```
   my_dataset/
   ├── images/          # Floor plan images (PNG, 512x512)
   ├── labels/          # JSON annotations
   └── prompts/         # TXT descriptions
   ```

2. Train on your data:
   ```bash
   python train_model.py --data_dir my_dataset/ --epochs 100
   ```

### Understanding Output Files

After running `python main.py "..."`, you get:

```
outputs/
├── final_layout_000.png          # Generated floor plan #1
├── final_layout_001.png          # Generated floor plan #2
├── final_layout_002.png          # Generated floor plan #3
├── parsed_specification.json     # LLM parsing output
├── detailed_specification.json   # Full layout spec
├── validation_report.json        # Engineering validation
└── generation_summary.json       # Complete metadata
```

**Key Files:**
- `final_layout_*.png` - Your CAD floor plans
- `validation_report.json` - Check for errors/warnings
- `parsed_specification.json` - See how LLM interpreted your text

### Troubleshooting

#### Problem: "CUDA out of memory"

**Solution 1:** Reduce batch size
```bash
python train_model.py --batch_size 2  # or even 1
```

**Solution 2:** Enable gradient checkpointing
In `train_model.py`, add:
```python
self.unet.enable_gradient_checkpointing()
```

**Solution 3:** Use CPU (slow but works)
```bash
export CUDA_VISIBLE_DEVICES=""
python train_model.py
```

#### Problem: "LLM API rate limit exceeded"

**Solution 1:** Get HuggingFace token
```bash
python main.py "..." --hf_token YOUR_TOKEN
```

**Solution 2:** Use lighter model
```bash
python main.py "..." --llm_model phi
```

**Solution 3:** Wait and retry
Free API has rate limits - wait 1 minute and retry.

#### Problem: Generated images are blurry/low quality

**Causes & Solutions:**

1. **Not enough training:**
   ```bash
   python train_model.py --epochs 100  # Train longer
   ```

2. **Dataset too small:**
   ```bash
   python dataset_generator.py  # Generate more data
   # Then retrain
   ```

3. **Wrong inference parameters:**
   Edit `inference.py`:
   ```python
   num_inference_steps=75,  # Increase from 50
   guidance_scale=8.5       # Increase from 7.5
   ```

#### Problem: "ModuleNotFoundError"

**Solution:** Install missing package
```bash
pip install <missing_package>
# Or reinstall all:
pip install -r requirements.txt --upgrade
```

#### Problem: Validation shows many errors

**Normal situation:** System auto-fixes most errors.

**Check:** Look at `validation_report.json`:
- **Critical errors**: Must be fixed
- **Warnings**: Suggestions, can ignore

**Manual fix:** Edit `detailed_specification.json` and regenerate.

### Best Practices

1. **Start Small:**
   - Use demo first
   - Generate small dataset (50 samples)
   - Train for 20 epochs to test
   - Then scale up

2. **Monitor Training:**
   - Watch loss values
   - Check generated samples periodically
   - Save checkpoints frequently

3. **Iterate on Prompts:**
   - Be specific in descriptions
   - Include key features
   - Mention room connections
   - Specify style preferences

4. **Validate Outputs:**
   - Always check validation_report.json
   - Review generated specifications
   - Verify room sizes and connections
   - Use professional review for real projects

### Performance Optimization

#### For Training:

1. **Use Mixed Precision:**
   Already enabled by default (FP16)

2. **Increase Batch Size:**
   ```bash
   python train_model.py --batch_size 8  # If VRAM allows
   ```

3. **Use Multiple GPUs:**
   ```bash
   accelerate config  # Set up distributed training
   accelerate launch train_model.py
   ```

#### For Inference:

1. **Batch Generation:**
   Modify `inference.py` to generate multiple images at once

2. **Lower Precision:**
   Already using FP16 on GPU

3. **Reduce Steps:**
   ```python
   num_inference_steps=30  # Faster but lower quality
   ```

### Common Use Cases

#### Case 1: Student Project
```bash
# Quick demo for presentation
python demo.py

# Or generate one layout
python dataset_generator.py  # Creates synthetic data
python train_model.py --epochs 20  # Quick training
python main.py "Your layout description"
```

#### Case 2: Research Project
```bash
# Full system with custom data
# 1. Prepare your dataset
# 2. Train extensively
python train_model.py --epochs 200 --batch_size 8
# 3. Generate and analyze
python main.py "..." --variations 10
```

#### Case 3: Architecture Course
```bash
# Generate various layouts for comparison
python main.py "Compact 1BHK efficient layout" --variations 5
python main.py "Spacious 3BHK luxury apartment" --variations 5
python main.py "Traditional 2BHK with vastu compliance" --variations 5
```

### Next Steps

1. **Improve Dataset:**
   - Add real floor plan images
   - Better annotations
   - More variety

2. **Fine-tune Further:**
   - Train on specific architectural styles
   - Add ControlNet for better control
   - Experiment with different base models

3. **Enhance Validation:**
   - Add jurisdiction-specific rules
   - Include more detailed checks
   - Implement auto-correction

4. **Build Interface:**
   - Create web UI
   - Add interactive editing
   - Enable real-time preview

### Resources

- **HuggingFace Models:** https://huggingface.co/models
- **Diffusers Docs:** https://huggingface.co/docs/diffusers
- **LoRA Training:** https://huggingface.co/docs/peft
- **Stable Diffusion:** https://stability.ai

### Support

If you encounter issues:

1. Check error messages carefully
2. Review this setup guide
3. Check validation_report.json
4. Verify all dependencies installed
5. Try demo.py first to isolate issues

### Safety Notes

⚠️ **Important Reminders:**

1. This is an educational/research tool
2. Not suitable for production use
3. Always verify with professionals
4. Building codes vary by location
5. Generated layouts may not be buildable as-is

---

**You're ready to go!** Start with `python demo.py` to test the system.
