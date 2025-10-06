#!/usr/bin/env python3
"""
Show Available Experiment Options
================================

This script shows all available experiments, feature extractors, and models.
"""

import sys
import os
sys.path.append('src')

from pathlib import Path
import numpy as np

def get_available_experiments():
    """Get all available experiments."""
    features_path = Path('data/features')
    if not features_path.exists():
        return []
    
    experiments = []
    for item in features_path.iterdir():
        if item.is_dir() and item.name.startswith('experiment'):
            try:
                exp_num = int(item.name.replace('experiment', ''))
                experiments.append((exp_num, item.name))
            except ValueError:
                continue
    
    return sorted(experiments, key=lambda x: x[0])

def get_available_feature_extractors(experiment_id):
    """Get all available feature extractors for an experiment."""
    features_path = Path(f'data/features/{experiment_id}')
    if not features_path.exists():
        return []
    
    extractors = []
    for item in features_path.iterdir():
        if item.is_dir():
            extractors.append(item.name)
    
    return sorted(extractors)

def analyze_feature_extractor(experiment_id, extractor_name):
    """Analyze a specific feature extractor."""
    features_path = Path(f'data/features/{experiment_id}/{extractor_name}')
    if not features_path.exists():
        return None
    
    feature_files = list(features_path.glob('*.npy'))
    if not feature_files:
        return None
    
    # Load the first feature file to get shape
    feature_file = feature_files[0]
    features = np.load(feature_file)
    
    return {
        'name': extractor_name,
        'shape': features.shape,
        'files': [f.name for f in feature_files],
        'size_mb': feature_file.stat().st_size / (1024 * 1024)
    }

def main():
    """Main function."""
    print("🔍 ECoG Experiment Options")
    print("=" * 50)
    
    # Get available experiments
    experiments = get_available_experiments()
    if not experiments:
        print("❌ No experiments found!")
        return
    
    print(f"📂 Available Experiments: {len(experiments)}")
    for exp_num, exp_id in experiments:
        print(f"  {exp_num}. {exp_id}")
    
    print(f"\n📊 Detailed Analysis:")
    
    for exp_num, exp_id in experiments:
        print(f"\n🧪 {exp_id}")
        print("-" * 30)
        
        extractors = get_available_feature_extractors(exp_id)
        if not extractors:
            print("  ⚠️ No feature extractors found")
            continue
        
        for extractor_name in extractors:
            analysis = analyze_feature_extractor(exp_id, extractor_name)
            if analysis:
                print(f"  📈 {analysis['name']}")
                print(f"     Shape: {analysis['shape']}")
                print(f"     Files: {', '.join(analysis['files'])}")
                print(f"     Size: {analysis['size_mb']:.1f} MB")
            else:
                print(f"  ❌ {extractor_name}: No valid features")
    
    print(f"\n🎯 Available Models:")
    print("  1. Ensemble Model (MultiModalEnsemble)")
    print("  2. Temporal Attention Model (PyTorch)")
    print("  3. Progressive Learning Model (PyTorch)")
    print("  4. Simple ML Models (sklearn)")
    print("     - Random Forest")
    print("     - Logistic Regression")
    print("     - Support Vector Machine")
    
    print(f"\n🚀 How to Run Experiments:")
    print("  1. Quick test (ensemble + simple ML):")
    print("     python run_quick_experiments.py")
    print("")
    print("  2. All experiments (all models):")
    print("     python run_all_experiments.py --full")
    print("")
    print("  3. Specific experiments:")
    print("     python run_all_experiments.py --experiments experiment8 --models ensemble")
    print("")
    print("  4. Single model (ensemble only):")
    print("     python run_simple_ensemble.py")
    
    print(f"\n📋 Data Structure Summary:")
    print("  • Raw data: 4 stimulus codes (0, 1, 2, 3)")
    print("  • Task: 4-class multiclass classification")
    print("  • Goal: Predict stimulus type from brain activity")
    print("  • Video: 252-second continuous video")
    print("  • Categories: digit, kanji, face, body, object, hiragana, line")
    print("    (Note: Only 4 codes in actual data, 7 categories are theoretical)")

if __name__ == "__main__":
    main()
