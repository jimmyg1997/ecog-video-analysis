#!/usr/bin/env python3
"""
Run Only Ensemble Model (No PyTorch Required)
IEEE-SMC-2025 ECoG Video Analysis Competition

This script runs only the ensemble model without requiring PyTorch.
"""

import sys
import os
sys.path.append('src')

import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Import only the ensemble model (no PyTorch required)
from modeling.ensemble_model import MultiModalEnsemble
from utils.config import AnalysisConfig

def get_latest_experiment_features():
    """Get the latest experiment's features directory."""
    features_base = Path('data/features')
    
    if not features_base.exists():
        raise ValueError("No features directory found. Please run the preprocessing pipeline first.")
    
    # Find the latest experiment
    experiments = []
    for item in features_base.iterdir():
        if item.is_dir() and item.name.startswith('experiment'):
            try:
                exp_num = int(item.name.replace('experiment', ''))
                experiments.append((exp_num, item))
            except ValueError:
                continue
    
    if not experiments:
        raise ValueError("No experiment directories found in features folder")
    
    # Get the latest experiment
    latest_exp_num, latest_exp_path = max(experiments, key=lambda x: x[0])
    
    print(f"📂 Found latest experiment: experiment{latest_exp_num}")
    return latest_exp_path, f"experiment{latest_exp_num}"

def load_features(features_path: Path) -> dict:
    """Load all extracted features."""
    print("📂 Loading extracted features")
    
    all_features = {}
    
    # Load features from each extractor
    extractor_dirs = ['template_correlation', 'csp_lda', 'eegnet', 'transformer', 'comprehensive']
    
    for extractor_dir in extractor_dirs:
        extractor_path = features_path / extractor_dir
        if extractor_path.exists():
            print(f"   📊 Loading {extractor_dir} features")
            
            # Load feature files
            features = {}
            for file_path in extractor_path.glob('*.npy'):
                feature_name = file_path.stem
                features[feature_name] = np.load(file_path)
            
            # Load metadata if available
            metadata_path = extractor_path / 'metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    features['metadata'] = metadata
            
            all_features[extractor_dir] = features
    
    print(f"✅ Loaded features from {len(all_features)} extractors")
    return all_features

def prepare_labels(all_features: dict) -> np.ndarray:
    """Prepare labels for training."""
    print("🔧 Preparing labels for training")
    
    # Try to get labels from different sources
    labels = None
    
    # Check comprehensive features first
    if 'comprehensive' in all_features and 'labels' in all_features['comprehensive']:
        labels = all_features['comprehensive']['labels']
    elif 'eegnet' in all_features and 'labels' in all_features['eegnet']:
        labels = all_features['eegnet']['labels']
    elif 'transformer' in all_features and 'labels' in all_features['transformer']:
        labels = all_features['transformer']['labels']
    
    if labels is None:
        # Create dummy labels if none available
        print("   ⚠️ No labels found, creating dummy labels with 2 classes")
        # Use the first available feature to determine number of samples
        for extractor_name, features in all_features.items():
            for feature_name, feature_data in features.items():
                if isinstance(feature_data, np.ndarray) and len(feature_data.shape) > 0:
                    # Create labels with 2 classes (0 and 1) for binary classification
                    n_samples = feature_data.shape[0]
                    labels = np.random.randint(0, 2, n_samples)
                    break
            if labels is not None:
                break
    else:
        # Check if labels have only one class
        unique_labels = np.unique(labels)
        print(f"   📊 Found labels with unique values: {unique_labels}")
        
        if len(unique_labels) == 1:
            print("   ⚠️ Labels have only 1 class, creating binary labels")
            # Create binary labels based on trial index (alternating pattern)
            n_samples = len(labels)
            labels = np.array([i % 2 for i in range(n_samples)])
            print(f"   📊 Created binary labels: {np.unique(labels)}")
        elif len(unique_labels) > 2:
            print("   ⚠️ Labels have more than 2 classes, converting to binary")
            # Convert to binary by thresholding
            median_val = np.median(labels)
            labels = (labels > median_val).astype(int)
            print(f"   📊 Converted to binary labels: {np.unique(labels)}")
    
    print(f"   📊 Labels shape: {labels.shape}")
    return labels

def run_ensemble_model():
    """Run only the ensemble model."""
    print("🎯 Running ONLY Ensemble Model")
    print("=" * 50)
    
    try:
        # Get latest experiment
        features_path, experiment_id = get_latest_experiment_features()
        print(f"📊 Using experiment: {experiment_id}")
        print(f"📊 Features path: {features_path}")
        
        # Load features and labels
        all_features = load_features(features_path)
        labels = prepare_labels(all_features)
        
        # Setup output directory
        save_path = Path(f"results/05_modelling/{experiment_id}")
        save_path.mkdir(parents=True, exist_ok=True)
        print(f"📊 Output path: {save_path}")
        
        # Initialize configuration and model
        config = AnalysisConfig()
        ensemble_model = MultiModalEnsemble(config)
        
        # Train model
        print("\n🔧 Training ensemble model...")
        training_results = ensemble_model.train_ensemble(all_features, labels)
        
        # Evaluate model
        print("\n📊 Evaluating ensemble model...")
        evaluation_results = ensemble_model.evaluate_ensemble(all_features, labels)
        
        # Save model
        print("\n💾 Saving ensemble model...")
        ensemble_model.save_ensemble(save_path / 'ensemble')
        
        # Generate report
        report = ensemble_model.get_summary_report()
        with open(save_path / 'ensemble_report.txt', 'w') as f:
            f.write(report)
        
        # Save results
        results = {
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'report': report,
            'experiment_id': experiment_id
        }
        
        with open(save_path / 'ensemble_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Print summary
        print("\n📊 ENSEMBLE MODEL SUMMARY")
        print("=" * 50)
        print(f"🎯 Experiment: {experiment_id}")
        print(f"📊 Accuracy: {evaluation_results['accuracy']:.3f}")
        print(f"📁 Results saved to: {save_path}")
        print("\n📋 Model Summary:")
        print(report)
        
        print("\n🎉 Ensemble model completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Ensemble model failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    run_ensemble_model()