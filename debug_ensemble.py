#!/usr/bin/env python3
"""
Debug Ensemble Model - Check labels and features
"""

import sys
import os
sys.path.append('src')

import numpy as np
from pathlib import Path
import json

def debug_features_and_labels():
    """Debug the features and labels to understand the issue."""
    print("🔍 Debugging Features and Labels")
    print("=" * 50)
    
    # Get latest experiment
    features_path = Path('data/features')
    experiments = [d for d in features_path.iterdir() if d.is_dir() and d.name.startswith('experiment')]
    if not experiments:
        print("❌ No experiments found")
        return
    
    latest_exp = max(experiments, key=lambda x: int(x.name.replace('experiment', '')))
    print(f"📂 Using experiment: {latest_exp.name}")
    
    # Load features
    all_features = {}
    for extractor_dir in ['template_correlation', 'csp_lda', 'eegnet', 'transformer', 'comprehensive']:
        extractor_path = latest_exp / extractor_dir
        if extractor_path.exists():
            print(f"\n📊 {extractor_dir} features:")
            features = {}
            for file_path in extractor_path.glob('*.npy'):
                feature_name = file_path.stem
                feature_data = np.load(file_path)
                features[feature_name] = feature_data
                print(f"   • {feature_name}: {feature_data.shape}")
            all_features[extractor_dir] = features
    
    # Check for labels
    print(f"\n🔍 Looking for labels:")
    labels = None
    for extractor_name, features in all_features.items():
        if 'labels' in features:
            labels = features['labels']
            print(f"   • Found labels in {extractor_name}: {labels.shape}, unique values: {np.unique(labels)}")
            break
    
    if labels is None:
        print("   ⚠️ No labels found, creating dummy labels")
        # Create labels with 2 classes
        n_samples = 252  # Assuming 252 trials
        labels = np.random.randint(0, 2, n_samples)
        print(f"   • Created dummy labels: {labels.shape}, unique values: {np.unique(labels)}")
    
    # Test ensemble model preparation
    print(f"\n🔍 Testing ensemble model preparation:")
    from modeling.ensemble_model import MultiModalEnsemble
    from utils.config import AnalysisConfig
    
    config = AnalysisConfig()
    ensemble_model = MultiModalEnsemble(config)
    
    # Prepare features
    prepared_features = ensemble_model.prepare_features(all_features)
    print(f"   • Prepared features: {list(prepared_features.keys())}")
    
    for feature_type, feature_data in prepared_features.items():
        print(f"   • {feature_type}: {feature_data.shape}")
    
    # Check if labels match any feature
    print(f"\n🔍 Checking label-feature compatibility:")
    for feature_type, feature_data in prepared_features.items():
        if feature_data.shape[0] == labels.shape[0]:
            print(f"   ✅ {feature_type}: {feature_data.shape[0]} samples matches labels {labels.shape[0]}")
        else:
            print(f"   ❌ {feature_type}: {feature_data.shape[0]} samples != labels {labels.shape[0]}")
    
    # Test with just one feature type
    print(f"\n🔍 Testing with comprehensive features only:")
    if 'comprehensive' in prepared_features:
        X = prepared_features['comprehensive']
        y = labels
        
        print(f"   • X shape: {X.shape}")
        print(f"   • y shape: {y.shape}")
        print(f"   • y unique values: {np.unique(y)}")
        print(f"   • Number of classes: {len(np.unique(y))}")
        
        # Test simple model
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        try:
            scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
            print(f"   ✅ Cross-validation successful: {scores.mean():.3f} ± {scores.std():.3f}")
        except Exception as e:
            print(f"   ❌ Cross-validation failed: {str(e)}")

if __name__ == "__main__":
    debug_features_and_labels()
