#!/usr/bin/env python3
"""
Quick ECoG Experiment Runner
===========================

This script runs quick experiments with all available feature extractors
and simple ML models for fast testing.

Usage:
    python run_quick_experiments.py
"""

import sys
import os
sys.path.append('src')

import time
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import ML models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

def get_latest_experiment():
    """Get the latest experiment ID."""
    features_path = Path('data/features')
    if not features_path.exists():
        raise ValueError("No features directory found")
    
    experiments = []
    for item in features_path.iterdir():
        if item.is_dir() and item.name.startswith('experiment'):
            try:
                exp_num = int(item.name.replace('experiment', ''))
                experiments.append((exp_num, item.name))
            except ValueError:
                continue
    
    if not experiments:
        raise ValueError("No experiment directories found")
    
    return max(experiments, key=lambda x: x[0])[1]

def load_all_features(experiment_id):
    """Load all available features for an experiment."""
    features_path = Path(f'data/features/{experiment_id}')
    if not features_path.exists():
        return {}
    
    all_features = {}
    
    for extractor_dir in features_path.iterdir():
        if extractor_dir.is_dir():
            extractor_name = extractor_dir.name
            
            # Find feature files
            feature_files = list(extractor_dir.glob('*.npy'))
            if feature_files:
                # Load the first available feature file
                feature_file = feature_files[0]
                features = np.load(feature_file)
                all_features[extractor_name] = features
                print(f"  ✅ Loaded {extractor_name}: {features.shape}")
    
    return all_features

def run_ml_experiment(features, labels, model_name, model):
    """Run a single ML experiment."""
    try:
        # Ensure features are 2D
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Cross-validation
        scores = cross_val_score(model, features_scaled, labels, cv=3, scoring='accuracy')
        
        return {
            'model': model_name,
            'accuracy': scores.mean(),
            'std': scores.std(),
            'status': 'success'
        }
        
    except Exception as e:
        return {
            'model': model_name,
            'accuracy': 0.0,
            'std': 0.0,
            'status': 'failed',
            'error': str(e)
        }

def main():
    """Main function."""
    print("🚀 Quick ECoG Experiment Runner")
    print("=" * 40)
    
    start_time = time.time()
    
    try:
        # Get latest experiment
        experiment_id = get_latest_experiment()
        print(f"📂 Using experiment: {experiment_id}")
        
        # Load all features
        print("📊 Loading all features...")
        all_features = load_all_features(experiment_id)
        
        if not all_features:
            print("❌ No features found!")
            return
        
        # Create labels (4-class multiclass)
        n_samples = min([features.shape[0] for features in all_features.values()])
        labels = np.random.randint(0, 4, n_samples)
        print(f"📊 Created labels: {labels.shape}, classes: {np.unique(labels)}")
        
        # Ensure all features have the same number of samples
        for name, features in all_features.items():
            if features.shape[0] > n_samples:
                all_features[name] = features[:n_samples, :]
        
        # Define models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        # Run experiments
        print("🎯 Running experiments...")
        all_results = []
        
        total_combinations = len(all_features) * len(models)
        
        with tqdm(total=total_combinations, desc="Running experiments") as pbar:
            for extractor_name, features in all_features.items():
                for model_name, model in models.items():
                    result = run_ml_experiment(features, labels, model_name, model)
                    result['extractor'] = extractor_name
                    all_results.append(result)
                    pbar.update(1)
        
        # Create results directory
        results_dir = Path(f"results/05_modelling/{experiment_id}")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create summary
        df = pd.DataFrame(all_results)
        successful_results = df[df['status'] == 'success']
        
        if len(successful_results) > 0:
            best_result = successful_results.loc[successful_results['accuracy'].idxmax()]
            print(f"\n🏆 Best Result:")
            print(f"  Extractor: {best_result['extractor']}")
            print(f"  Model: {best_result['model']}")
            print(f"  Accuracy: {best_result['accuracy']:.3f} ± {best_result['std']:.3f}")
        
        # Calculate summary statistics
        best_accuracy = successful_results['accuracy'].max() if len(successful_results) > 0 else 0.0
        avg_accuracy = successful_results['accuracy'].mean() if len(successful_results) > 0 else 0.0
        
        # Create detailed report
        report = f"""
# Quick ECoG Experiment Report
Experiment: {experiment_id}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- Total combinations: {len(all_results)}
- Successful: {len(successful_results)}
- Failed: {len(all_results) - len(successful_results)}
- Best accuracy: {best_accuracy:.3f}
- Average accuracy: {avg_accuracy:.3f}

## Results by Extractor
"""
        
        for extractor in all_features.keys():
            extractor_results = df[df['extractor'] == extractor]
            report += f"\n### {extractor}\n"
            for _, row in extractor_results.iterrows():
                status_icon = "✅" if row['status'] == 'success' else "❌"
                report += f"- {status_icon} {row['model']}: {row['accuracy']:.3f} ± {row['std']:.3f}\n"
        
        # Save report
        report_file = results_dir / f"quick_experiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Save results as JSON
        results_file = results_dir / f"quick_experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        elapsed_time = time.time() - start_time
        
        print(f"\n🎉 Quick Experiments Completed!")
        print("=" * 40)
        print(f"⏱️ Time: {elapsed_time:.1f} seconds")
        print(f"📊 Total combinations: {len(all_results)}")
        print(f"✅ Successful: {len(successful_results)}")
        print(f"❌ Failed: {len(all_results) - len(successful_results)}")
        print(f"📁 Report: {report_file}")
        print(f"📁 Results: {results_file}")
        
        # Show top 5 results
        if len(successful_results) > 0:
            print(f"\n🏆 Top 5 Results:")
            top_5 = successful_results.nlargest(5, 'accuracy')
            for i, (_, row) in enumerate(top_5.iterrows(), 1):
                print(f"  {i}. {row['extractor']} + {row['model']}: {row['accuracy']:.3f} ± {row['std']:.3f}")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
