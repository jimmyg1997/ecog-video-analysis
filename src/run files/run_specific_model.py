#!/usr/bin/env python3
"""
Run Specific Modeling Approach - Simplified Version
IEEE-SMC-2025 ECoG Video Analysis Competition

This script provides a simplified way to run specific modeling approaches.
"""

import sys
import os
sys.path.append('src')

import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Import our modeling modules
from modeling.ensemble_model import MultiModalEnsemble
from modeling.temporal_attention_model import TemporalAttentionModel
from modeling.progressive_learning_model import ProgressiveLearningModel
from utils.config import AnalysisConfig

def run_ensemble_only():
    """Run only the ensemble model."""
    print("🎯 Running ONLY Ensemble Model")
    print("=" * 50)
    
    # Get latest experiment
    features_path = Path('data/features')
    experiments = [d for d in features_path.iterdir() if d.is_dir() and d.name.startswith('experiment')]
    if not experiments:
        print("❌ No experiments found. Run preprocessing first.")
        return
    
    latest_exp = max(experiments, key=lambda x: int(x.name.replace('experiment', '')))
    print(f"📂 Using experiment: {latest_exp.name}")
    
    # Load features (simplified)
    all_features = {}
    for extractor_dir in ['template_correlation', 'csp_lda', 'eegnet', 'transformer', 'comprehensive']:
        extractor_path = latest_exp / extractor_dir
        if extractor_path.exists():
            features = {}
            for file_path in extractor_path.glob('*.npy'):
                features[file_path.stem] = np.load(file_path)
            all_features[extractor_dir] = features
    
    # Create dummy labels
    labels = np.zeros(252)  # Assuming 252 trials
    
    # Initialize and train ensemble model
    config = AnalysisConfig()
    ensemble_model = MultiModalEnsemble(config)
    
    print("🔧 Training ensemble model...")
    training_results = ensemble_model.train_ensemble(all_features, labels)
    
    print("📊 Evaluating ensemble model...")
    evaluation_results = ensemble_model.evaluate_ensemble(all_features, labels)
    
    print(f"✅ Ensemble model completed! Accuracy: {evaluation_results['accuracy']:.3f}")
    
    # Save results
    save_path = Path(f"results/05_modelling/{latest_exp.name}")
    save_path.mkdir(parents=True, exist_ok=True)
    ensemble_model.save_ensemble(save_path / 'ensemble')
    
    print(f"💾 Results saved to: {save_path}")

def run_temporal_attention_only():
    """Run only the temporal attention model."""
    print("🎯 Running ONLY Temporal Attention Model")
    print("=" * 50)
    
    # Get latest experiment
    features_path = Path('data/features')
    experiments = [d for d in features_path.iterdir() if d.is_dir() and d.name.startswith('experiment')]
    if not experiments:
        print("❌ No experiments found. Run preprocessing first.")
        return
    
    latest_exp = max(experiments, key=lambda x: int(x.name.replace('experiment', '')))
    print(f"📂 Using experiment: {latest_exp.name}")
    
    # Load transformer features
    transformer_path = latest_exp / 'transformer'
    if not transformer_path.exists():
        print("❌ Transformer features not found. Run feature extraction first.")
        return
    
    features = {}
    for file_path in transformer_path.glob('*.npy'):
        features[file_path.stem] = np.load(file_path)
    
    # Create dummy labels
    labels = np.zeros(252)  # Assuming 252 trials
    
    # Initialize and train temporal attention model
    config = AnalysisConfig()
    temporal_model = TemporalAttentionModel(config)
    
    # Dummy brain atlas
    class DummyBrainAtlas:
        def __init__(self):
            self.channel_to_region = {i: 'central' for i in range(160)}
    
    brain_atlas = DummyBrainAtlas()
    
    print("🔧 Training temporal attention model...")
    training_results = temporal_model.train(features, brain_atlas)
    
    print("📊 Evaluating temporal attention model...")
    predictions, probabilities = temporal_model.predict(features, brain_atlas)
    accuracy = np.mean(predictions == labels)
    
    print(f"✅ Temporal attention model completed! Accuracy: {accuracy:.3f}")
    
    # Save results
    save_path = Path(f"results/05_modelling/{latest_exp.name}")
    save_path.mkdir(parents=True, exist_ok=True)
    temporal_model.save_model(save_path / 'temporal_attention')
    
    print(f"💾 Results saved to: {save_path}")

def run_progressive_learning_only():
    """Run only the progressive learning model."""
    print("🎯 Running ONLY Progressive Learning Model")
    print("=" * 50)
    
    # Get latest experiment
    features_path = Path('data/features')
    experiments = [d for d in features_path.iterdir() if d.is_dir() and d.name.startswith('experiment')]
    if not experiments:
        print("❌ No experiments found. Run preprocessing first.")
        return
    
    latest_exp = max(experiments, key=lambda x: int(x.name.replace('experiment', '')))
    print(f"📂 Using experiment: {latest_exp.name}")
    
    # Load features (simplified)
    all_features = {}
    for extractor_dir in ['template_correlation', 'csp_lda', 'eegnet', 'transformer', 'comprehensive']:
        extractor_path = latest_exp / extractor_dir
        if extractor_path.exists():
            features = {}
            for file_path in extractor_path.glob('*.npy'):
                features[file_path.stem] = np.load(file_path)
            all_features[extractor_dir] = features
    
    # Create dummy labels
    labels = np.zeros(252)  # Assuming 252 trials
    
    # Initialize and train progressive learning model
    config = AnalysisConfig()
    progressive_model = ProgressiveLearningModel(config)
    
    print("🔧 Training progressive learning model...")
    training_results = progressive_model.train_progressive(all_features, labels)
    
    print("📊 Evaluating progressive learning model...")
    evaluation_results = progressive_model.evaluate_progressive(all_features, labels)
    
    print(f"✅ Progressive learning model completed! Accuracy: {evaluation_results['accuracy']:.3f}")
    
    # Save results
    save_path = Path(f"results/05_modelling/{latest_exp.name}")
    save_path.mkdir(parents=True, exist_ok=True)
    progressive_model.save_progressive_model(save_path / 'progressive_learning')
    
    print(f"💾 Results saved to: {save_path}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python run_specific_model.py ensemble")
        print("  python run_specific_model.py temporal_attention")
        print("  python run_specific_model.py progressive_learning")
        sys.exit(1)
    
    model_type = sys.argv[1].lower()
    
    if model_type == 'ensemble':
        run_ensemble_only()
    elif model_type == 'temporal_attention':
        run_temporal_attention_only()
    elif model_type == 'progressive_learning':
        run_progressive_learning_only()
    else:
        print(f"❌ Unknown model type: {model_type}")
        print("Available options: ensemble, temporal_attention, progressive_learning")
        sys.exit(1)
