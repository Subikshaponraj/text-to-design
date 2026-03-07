"""
Dataset Validation Script
Run this before training to verify your dataset is properly formatted
"""

import os
from pathlib import Path
from PIL import Image
import json


class DatasetValidator:
    """Validate floor plan dataset structure and content"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.errors = []
        self.warnings = []
        self.stats = {
            'total_samples': 0,
            'valid_samples': 0,
            'missing_prompts': 0,
            'missing_images': 0,
            'invalid_images': 0,
            'has_labels': False,
            'has_controls': False
        }
    
    def validate(self):
        """Run all validation checks"""
        print("🔍 Validating Dataset")
        print("=" * 70)
        print(f"Dataset path: {self.dataset_path}")
        print()
        
        # Check directory structure
        self._check_structure()
        
        # Check file matching
        self._check_files()
        
        # Check image properties
        self._check_images()
        
        # Check prompts
        self._check_prompts()
        
        # Print results
        self._print_results()
        
        return len(self.errors) == 0
    
    def _check_structure(self):
        """Verify directory structure"""
        print("📁 Checking directory structure...")
        
        if not self.dataset_path.exists():
            self.errors.append(f"Dataset path does not exist: {self.dataset_path}")
            return
        
        required_dirs = ['images', 'prompts']
        optional_dirs = ['labels', 'controls']
        
        for dir_name in required_dirs:
            dir_path = self.dataset_path / dir_name
            if not dir_path.exists():
                self.errors.append(f"Required directory missing: {dir_name}/")
            elif not dir_path.is_dir():
                self.errors.append(f"Not a directory: {dir_name}/")
        
        for dir_name in optional_dirs:
            dir_path = self.dataset_path / dir_name
            if dir_path.exists():
                if dir_name == 'labels':
                    self.stats['has_labels'] = True
                if dir_name == 'controls':
                    self.stats['has_controls'] = True
                print(f"  ✓ Found optional: {dir_name}/")
    
    def _check_files(self):
        """Check file matching between images and prompts"""
        print("\n📄 Checking file matching...")
        
        images_dir = self.dataset_path / 'images'
        prompts_dir = self.dataset_path / 'prompts'
        
        if not images_dir.exists() or not prompts_dir.exists():
            return
        
        # Get all file stems
        image_stems = {p.stem for p in images_dir.glob('*.png')}
        prompt_stems = {p.stem for p in prompts_dir.glob('*.txt')}
        
        # Check matching
        matched = image_stems & prompt_stems
        only_images = image_stems - prompt_stems
        only_prompts = prompt_stems - image_stems
        
        self.stats['total_samples'] = len(image_stems)
        self.stats['valid_samples'] = len(matched)
        self.stats['missing_prompts'] = len(only_images)
        self.stats['missing_images'] = len(only_prompts)
        
        print(f"  Total image files: {len(image_stems)}")
        print(f"  Total prompt files: {len(prompt_stems)}")
        print(f"  Matched pairs: {len(matched)}")
        
        if only_images:
            self.warnings.append(
                f"{len(only_images)} images have no matching prompt files"
            )
            if len(only_images) <= 5:
                print(f"  ⚠ Missing prompts for: {list(only_images)}")
        
        if only_prompts:
            self.warnings.append(
                f"{len(only_prompts)} prompts have no matching image files"
            )
            if len(only_prompts) <= 5:
                print(f"  ⚠ Missing images for: {list(only_prompts)}")
        
        if len(matched) == 0:
            self.errors.append("No matching image-prompt pairs found!")
        elif len(matched) < 100:
            self.warnings.append(
                f"Only {len(matched)} samples - recommend at least 100 for training"
            )
    
    def _check_images(self):
        """Validate image properties"""
        print("\n🖼️  Checking images...")
        
        images_dir = self.dataset_path / 'images'
        if not images_dir.exists():
            return
        
        image_files = list(images_dir.glob('*.png'))[:50]  # Check first 50
        
        sizes = []
        modes = []
        corrupted = []
        
        for img_path in image_files:
            try:
                with Image.open(img_path) as img:
                    sizes.append(img.size)
                    modes.append(img.mode)
            except Exception as e:
                corrupted.append(img_path.name)
                self.errors.append(f"Corrupted image: {img_path.name}")
        
        if sizes:
            unique_sizes = set(sizes)
            unique_modes = set(modes)
            
            print(f"  Checked {len(image_files)} sample images")
            print(f"  Image sizes: {unique_sizes}")
            print(f"  Image modes: {unique_modes}")
            
            # Check for consistency
            if len(unique_sizes) > 3:
                self.warnings.append(
                    f"Images have varying sizes: {unique_sizes}. "
                    "Training will resize all to target resolution."
                )
            
            # Check if images are too small
            min_size = min(min(s) for s in sizes)
            if min_size < 256:
                self.warnings.append(
                    f"Some images are very small (min: {min_size}px). "
                    "Consider using higher resolution images."
                )
            
            # Check mode
            if 'RGBA' in unique_modes:
                self.warnings.append(
                    "Some images have alpha channel (RGBA). "
                    "Will be converted to RGB during training."
                )
        
        if corrupted:
            print(f"  ❌ {len(corrupted)} corrupted images found")
            self.stats['invalid_images'] = len(corrupted)
    
    def _check_prompts(self):
        """Validate prompt content"""
        print("\n📝 Checking prompts...")
        
        prompts_dir = self.dataset_path / 'prompts'
        if not prompts_dir.exists():
            return
        
        prompt_files = list(prompts_dir.glob('*.txt'))[:50]  # Check first 50
        
        empty_prompts = []
        short_prompts = []
        long_prompts = []
        prompt_lengths = []
        
        for prompt_path in prompt_files:
            try:
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    
                    if not content:
                        empty_prompts.append(prompt_path.name)
                        continue
                    
                    length = len(content)
                    prompt_lengths.append(length)
                    
                    if length < 20:
                        short_prompts.append(prompt_path.name)
                    elif length > 500:
                        long_prompts.append(prompt_path.name)
                        
            except Exception as e:
                self.errors.append(f"Cannot read prompt: {prompt_path.name}")
        
        if prompt_lengths:
            avg_length = sum(prompt_lengths) / len(prompt_lengths)
            print(f"  Checked {len(prompt_files)} sample prompts")
            print(f"  Average length: {avg_length:.0f} characters")
            print(f"  Range: {min(prompt_lengths)} - {max(prompt_lengths)}")
        
        if empty_prompts:
            self.errors.append(f"{len(empty_prompts)} empty prompt files found")
        
        if short_prompts:
            self.warnings.append(
                f"{len(short_prompts)} prompts are very short (<20 chars). "
                "Consider adding more descriptive text."
            )
        
        if long_prompts:
            self.warnings.append(
                f"{len(long_prompts)} prompts are very long (>500 chars). "
                "They will be truncated during training."
            )
    
    def _print_results(self):
        """Print validation summary"""
        print("\n" + "=" * 70)
        print("📊 VALIDATION SUMMARY")
        print("=" * 70)
        
        # Statistics
        print("\nDataset Statistics:")
        print(f"  Total samples: {self.stats['total_samples']}")
        print(f"  Valid pairs: {self.stats['valid_samples']}")
        print(f"  Has labels: {'Yes' if self.stats['has_labels'] else 'No'}")
        print(f"  Has controls: {'Yes' if self.stats['has_controls'] else 'No'}")
        
        # Errors
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"  • {error}")
        
        # Warnings
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        # Final verdict
        print("\n" + "=" * 70)
        if not self.errors:
            if self.warnings:
                print("✅ Dataset is VALID (with warnings)")
                print("   You can proceed with training, but consider fixing warnings.")
            else:
                print("✅ Dataset is PERFECT!")
                print("   Ready for training!")
        else:
            print("❌ Dataset has ERRORS")
            print("   Please fix errors before training.")
        print("=" * 70)
        
        # Save report
        self._save_report()
    
    def _save_report(self):
        """Save validation report to JSON"""
        report = {
            'dataset_path': str(self.dataset_path),
            'stats': self.stats,
            'errors': self.errors,
            'warnings': self.warnings,
            'is_valid': len(self.errors) == 0
        }
        
        report_path = self.dataset_path / 'validation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Report saved: {report_path}")


def quick_check(dataset_path: str):
    """Quick dataset check - prints summary"""
    path = Path(dataset_path)
    
    if not path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return False
    
    images = list((path / 'images').glob('*.png')) if (path / 'images').exists() else []
    prompts = list((path / 'prompts').glob('*.txt')) if (path / 'prompts').exists() else []
    
    print("\n" + "=" * 50)
    print(f"Quick Check: {dataset_path}")
    print("=" * 50)
    print(f"Images:  {len(images)}")
    print(f"Prompts: {len(prompts)}")
    
    if len(images) > 0 and len(prompts) > 0:
        # Sample one
        sample = images[0]
        try:
            img = Image.open(sample)
            print(f"Sample:  {img.size} {img.mode}")
            
            prompt_file = path / 'prompts' / f"{sample.stem}.txt"
            if prompt_file.exists():
                with open(prompt_file) as f:
                    prompt = f.read().strip()
                print(f"Prompt:  {prompt[:80]}...")
        except Exception as e:
            print(f"Error: {e}")
    
    print("=" * 50 + "\n")
    
    return len(images) > 0 and len(prompts) > 0


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python validate_dataset.py <dataset_path>")
        print("\nExample:")
        print("  python validate_dataset.py ./floor_plan_dataset")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    
    # Run quick check first
    if not quick_check(dataset_path):
        print("❌ Basic check failed. Run full validation for details.")
        sys.exit(1)
    
    # Run full validation
    validator = DatasetValidator(dataset_path)
    is_valid = validator.validate()
    
    sys.exit(0 if is_valid else 1)