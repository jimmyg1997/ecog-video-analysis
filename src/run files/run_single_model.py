#!/usr/bin/env python3
"""
Run Single Modeling Approach for ECoG Classification
IEEE-SMC-2025 ECoG Video Analysis Competition

This script allows you to run only one specific modeling approach with command-line arguments.

Usage:
    python run_single_model.py --model ensemble
    python run_single_model.py --model temporal_attention
    python run_single_model.py --model progressive_learning
    python run_single_model.py --model all
"""

import sys
import os
sys.path.append('src')

import numpy as np
import pandas as pd
from pathlib import Path
import json
import warnings
import argparse
from datetime import datetime
warnings.filterwarnings('ignore')

# Import our modeling modules
from modeling.ensemble_model import MultiModalEnsemble
from modeling.temporal_attention_model import TemporalAttentionModel
from modeling.progressive_learning_model import ProgressiveLearningModel
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
        print("   ⚠️ No labels found, creating dummy labels")
        # Use the first available feature to determine number of samples
        for extractor_name, features in all_features.items():
            for feature_name, feature_data in features.items():
                if isinstance(feature_data, np.ndarray) and len(feature_data.shape) > 0:
                    labels = np.zeros(feature_data.shape[0])
                    break
            if labels is not None:
                break
    
    print(f"   📊 Labels shape: {labels.shape}")
    return labels

def run_ensemble_model(all_features: dict, labels: np.ndarray, save_path: Path, config: dict):
    """Run only the ensemble model."""
    print("🎯 Running Multi-Modal Ensemble Model")
    print("=" * 50)
    
    # Initialize model
    ensemble_model = MultiModalEnsemble(config)
    
    # Train model
    print("🔧 Training ensemble model...")
    training_results = ensemble_model.train_ensemble(all_features, labels)
    
    # Evaluate model
    print("📊 Evaluating ensemble model...")
    evaluation_results = ensemble_model.evaluate_ensemble(all_features, labels)
    
    # Save model
    print("💾 Saving ensemble model...")
    ensemble_model.save_ensemble(save_path / 'ensemble')
    
    # Generate report
    report = ensemble_model.get_summary_report()
    with open(save_path / 'ensemble_report.txt', 'w') as f:
        f.write(report)
    
    print("✅ Ensemble model completed!")
    print(f"📊 Accuracy: {evaluation_results['accuracy']:.3f}")
    
    return {
        'training_results': training_results,
        'evaluation_results': evaluation_results,
        'report': report
    }

def run_temporal_attention_model(all_features: dict, labels: np.ndarray, save_path: Path, config: dict):
    """Run only the temporal attention model."""
    print("🎯 Running Temporal Attention Transformer")
    print("=" * 50)
    
    # Check if transformer features are available
    if 'transformer' not in all_features:
        print("❌ Transformer features not available. Please run the transformer feature extractor first.")
        return None
    
    # Initialize model
    temporal_attention_model = TemporalAttentionModel(config)
    
    # Create dummy brain atlas for now
    class DummyBrainAtlas:
        def __init__(self):
            self.channel_to_region = {i: 'central' for i in range(160)}
    
    brain_atlas = DummyBrainAtlas()
    
    # Train model
    print("🔧 Training temporal attention model...")
    training_results = temporal_attention_model.train(all_features['transformer'], brain_atlas)
    
    # Evaluate model
    print("📊 Evaluating temporal attention model...")
    predictions, probabilities = temporal_attention_model.predict(all_features['transformer'], brain_atlas)
    
    # Calculate accuracy
    accuracy = np.mean(predictions == labels)
    evaluation_results = {
        'accuracy': accuracy,
        'predictions': predictions,
        'probabilities': probabilities
    }
    
    # Save model
    print("💾 Saving temporal attention model...")
    temporal_attention_model.save_model(save_path / 'temporal_attention')
    
    # Generate report
    report = temporal_attention_model.get_summary_report()
    with open(save_path / 'temporal_attention_report.txt', 'w') as f:
        f.write(report)
    
    print("✅ Temporal attention model completed!")
    print(f"📊 Accuracy: {accuracy:.3f}")
    
    return {
        'training_results': training_results,
        'evaluation_results': evaluation_results,
        'report': report
    }

def run_progressive_learning_model(all_features: dict, labels: np.ndarray, save_path: Path, config: dict):
    """Run only the progressive learning model."""
    print("🎯 Running Progressive Learning Model")
    print("=" * 50)
    
    # Initialize model
    progressive_model = ProgressiveLearningModel(config)
    
    # Train model
    print("🔧 Training progressive learning model...")
    training_results = progressive_model.train_progressive(all_features, labels)
    
    # Evaluate model
    print("📊 Evaluating progressive learning model...")
    evaluation_results = progressive_model.evaluate_progressive(all_features, labels)
    
    # Save model
    print("💾 Saving progressive learning model...")
    progressive_model.save_progressive_model(save_path / 'progressive_learning')
    
    # Generate report
    report = progressive_model.get_summary_report()
    with open(save_path / 'progressive_learning_report.txt', 'w') as f:
        f.write(report)
    
    print("✅ Progressive learning model completed!")
    print(f"📊 Accuracy: {evaluation_results['accuracy']:.3f}")
    
    return {
        'training_results': training_results,
        'evaluation_results': evaluation_results,
        'report': report
    }

def main():
    """Main function to run a single modeling approach."""
    parser = argparse.ArgumentParser(description='Run a single ECoG modeling approach')
    parser.add_argument('--model', type=str, required=True,
                       choices=['ensemble', 'temporal_attention', 'progressive_learning', 'all'],
                       help='Which model to run: ensemble, temporal_attention, progressive_learning, or all')
    parser.add_argument('--experiment', type=str, default=None,
                       help='Specific experiment ID to use (default: latest)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (default: results/05_modelling/experiment_X)')
    
    args = parser.parse_args()
    
    print("🚀 Starting Single Model Execution")
    print("IEEE-SMC-2025 ECoG Video Analysis Competition")
    print("=" * 70)
    print(f"🎯 Model: {args.model}")
    print("=" * 70)
    
    try:
        # Get experiment data
        if args.experiment:
            features_path = Path(f'data/features/{args.experiment}')
            experiment_id = args.experiment
            if not features_path.exists():
                raise ValueError(f"Experiment {args.experiment} not found in features directory")
        else:
            features_path, experiment_id = get_latest_experiment_features()
        
        print(f"📊 Using experiment: {experiment_id}")
        print(f"📊 Features path: {features_path}")
        
        # Setup output directory
        if args.output:
            save_path = Path(args.output)
        else:
            save_path = Path(f"results/05_modelling/{experiment_id}")
        
        save_path.mkdir(parents=True, exist_ok=True)
        print(f"📊 Output path: {save_path}")
        
        # Load features and labels
        all_features = load_features(features_path)
        labels = prepare_labels(all_features)
        
        # Initialize configuration
        config = AnalysisConfig()
        
        # Run selected model(s)
        results = {}
        
        if args.model == 'ensemble' or args.model == 'all':
            print("\n" + "="*70)
            results['ensemble'] = run_ensemble_model(all_features, labels, save_path, config)
        
        if args.model == 'temporal_attention' or args.model == 'all':
            print("\n" + "="*70)
            results['temporal_attention'] = run_temporal_attention_model(all_features, labels, save_path, config)
        
        if args.model == 'progressive_learning' or args.model == 'all':
            print("\n" + "="*70)
            results['progressive_learning'] = run_progressive_learning_model(all_features, labels, save_path, config)
        
        # Save results
        with open(save_path / 'single_model_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Print summary
        print("\n📊 EXECUTION SUMMARY")
        print("=" * 70)
        print(f"🎯 Model(s) executed: {args.model}")
        print(f"📊 Experiment: {experiment_id}")
        print(f"📁 Results saved to: {save_path}")
        
        for model_name, model_results in results.items():
            if model_results and 'evaluation_results' in model_results:
                accuracy = model_results['evaluation_results']['accuracy']
                print(f"   • {model_name}: Accuracy = {accuracy:.3f}")
        
        print("\n🎉 Single model execution completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Model execution failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
