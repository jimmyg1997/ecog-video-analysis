#!/usr/bin/env python3
"""
Test Meaningful Classification Tasks
===================================

This script tests different classification tasks using the meaningful labels
we created, instead of the basic 4-class approach.

Usage:
    python test_meaningful_classification.py
"""

import sys
import os
sys.path.append('src')

import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def load_features_and_labels(experiment_id):
    """Load features and meaningful labels."""
    print("📊 Loading features and labels...")
    
    # Load features
    features_path = Path(f'data/features/{experiment_id}')
    features = {}
    
    for extractor_dir in features_path.iterdir():
        if extractor_dir.is_dir():
            extractor_name = extractor_dir.name
            feature_files = list(extractor_dir.glob('*.npy'))
            if feature_files:
                feature_file = feature_files[0]
                features[extractor_name] = np.load(feature_file)
                print(f"  ✅ Loaded {extractor_name}: {features[extractor_name].shape}")
    
    # Load meaningful labels
    labels_path = Path(f'data/labels/{experiment_id}')
    if not labels_path.exists():
        print("❌ No meaningful labels found. Run create_meaningful_labels.py first.")
        return None, None
    
    labels = {}
    for label_file in labels_path.glob('*.npy'):
        label_name = label_file.stem
        labels[label_name] = np.load(label_file)
        print(f"  ✅ Loaded {label_name}: {labels[label_name].shape}")
    
    return features, labels

def test_classification_task(features, labels, task_name, feature_name, model_name, model):
    """Test a specific classification task."""
    try:
        # Ensure features are 2D
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        
        # Ensure same number of samples
        min_samples = min(features.shape[0], labels.shape[0])
        features = features[:min_samples, :]
        labels = labels[:min_samples]
        
        # Check if we have enough classes
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return {
                'task': task_name,
                'feature': feature_name,
                'model': model_name,
                'accuracy': 0.0,
                'status': 'failed',
                'error': 'Not enough classes'
            }
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Cross-validation
        scores = cross_val_score(model, features_scaled, labels, cv=3, scoring='accuracy')
        
        return {
            'task': task_name,
            'feature': feature_name,
            'model': model_name,
            'accuracy': scores.mean(),
            'std': scores.std(),
            'n_classes': len(unique_labels),
            'class_distribution': np.bincount(labels).tolist(),
            'status': 'success'
        }
        
    except Exception as e:
        return {
            'task': task_name,
            'feature': feature_name,
            'model': model_name,
            'accuracy': 0.0,
            'status': 'failed',
            'error': str(e)
        }

def run_comprehensive_testing(features, labels):
    """Run comprehensive testing across all combinations."""
    print("🎯 Running comprehensive testing...")
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42, probability=True)
    }
    
    all_results = []
    
    # Test each label type
    for label_name, label_data in labels.items():
        print(f"\n🔍 Testing {label_name}...")
        
        # Handle different label structures
        if isinstance(label_data, dict):
            # Dictionary of labels (e.g., binary_simple, binary_sophisticated)
            for sub_name, sub_labels in label_data.items():
                if isinstance(sub_labels, np.ndarray):
                    task_name = f"{label_name}_{sub_name}"
                    print(f"  📊 {task_name}: {sub_labels.shape}, classes: {np.unique(sub_labels)}")
                    
                    # Test with each feature extractor
                    for feature_name, feature_data in features.items():
                        for model_name, model in models.items():
                            result = test_classification_task(
                                feature_data, sub_labels, task_name, feature_name, model_name, model
                            )
                            all_results.append(result)
        else:
            # Single array of labels
            task_name = label_name
            print(f"  📊 {task_name}: {label_data.shape}, classes: {np.unique(label_data)}")
            
            # Test with each feature extractor
            for feature_name, feature_data in features.items():
                for model_name, model in models.items():
                    result = test_classification_task(
                        feature_data, label_data, task_name, feature_name, model_name, model
                    )
                    all_results.append(result)
    
    return all_results

def create_comprehensive_report(results, experiment_id):
    """Create a comprehensive report of results."""
    print("📋 Creating comprehensive report...")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Create results directory
    results_dir = Path(f"results/05_modelling/{experiment_id}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Summary statistics
    successful_results = df[df['status'] == 'success']
    
    if len(successful_results) == 0:
        print("❌ No successful results to report!")
        return
    
    # Best results by task
    best_by_task = successful_results.loc[successful_results.groupby('task')['accuracy'].idxmax()]
    
    # Best results overall
    best_overall = successful_results.loc[successful_results['accuracy'].idxmax()]
    
    # Create report
    report = f"""
# Meaningful Classification Results Report
Experiment: {experiment_id}
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- Total combinations tested: {len(results)}
- Successful: {len(successful_results)}
- Failed: {len(results) - len(successful_results)}
- Best overall accuracy: {best_overall['accuracy']:.3f} ± {best_overall['std']:.3f}
- Best task: {best_overall['task']}
- Best feature: {best_overall['feature']}
- Best model: {best_overall['model']}

## Best Results by Task
"""
    
    for _, row in best_by_task.iterrows():
        report += f"""
### {row['task']}
- **Best Accuracy**: {row['accuracy']:.3f} ± {row['std']:.3f}
- **Best Feature**: {row['feature']}
- **Best Model**: {row['model']}
- **Classes**: {row['n_classes']}
- **Distribution**: {row['class_distribution']}
"""
    
    # Top 10 results
    report += f"""
## Top 10 Results Overall
"""
    top_10 = successful_results.nlargest(10, 'accuracy')
    for i, (_, row) in enumerate(top_10.iterrows(), 1):
        report += f"{i}. **{row['task']}** + {row['feature']} + {row['model']}: {row['accuracy']:.3f} ± {row['std']:.3f}\n"
    
    # Save report
    report_file = results_dir / f"meaningful_classification_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_file, 'w') as f:
        f.write(report)
    
    # Save results as JSON
    results_file = results_dir / f"meaningful_classification_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"📁 Report saved: {report_file}")
    print(f"📁 Results saved: {results_file}")
    
    return best_overall, best_by_task

def create_visualizations(results, experiment_id):
    """Create visualizations of results."""
    print("🎨 Creating visualizations...")
    
    df = pd.DataFrame(results)
    successful_results = df[df['status'] == 'success']
    
    if len(successful_results) == 0:
        print("❌ No successful results to visualize!")
        return
    
    results_dir = Path(f"results/05_modelling/{experiment_id}")
    
    # 1. Accuracy by task
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Accuracy by task
    plt.subplot(2, 2, 1)
    task_accuracy = successful_results.groupby('task')['accuracy'].max().sort_values(ascending=True)
    task_accuracy.plot(kind='barh')
    plt.title('Best Accuracy by Task')
    plt.xlabel('Accuracy')
    plt.tight_layout()
    
    # Subplot 2: Accuracy by feature extractor
    plt.subplot(2, 2, 2)
    feature_accuracy = successful_results.groupby('feature')['accuracy'].max().sort_values(ascending=True)
    feature_accuracy.plot(kind='barh')
    plt.title('Best Accuracy by Feature Extractor')
    plt.xlabel('Accuracy')
    plt.tight_layout()
    
    # Subplot 3: Accuracy by model
    plt.subplot(2, 2, 3)
    model_accuracy = successful_results.groupby('model')['accuracy'].max().sort_values(ascending=True)
    model_accuracy.plot(kind='barh')
    plt.title('Best Accuracy by Model')
    plt.xlabel('Accuracy')
    plt.tight_layout()
    
    # Subplot 4: Task complexity (number of classes)
    plt.subplot(2, 2, 4)
    task_complexity = successful_results.groupby('task')['n_classes'].first().sort_values(ascending=True)
    task_complexity.plot(kind='barh')
    plt.title('Task Complexity (Number of Classes)')
    plt.xlabel('Number of Classes')
    plt.tight_layout()
    
    plt.suptitle('Meaningful Classification Results Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'meaningful_classification_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(results_dir / 'meaningful_classification_analysis.svg', bbox_inches='tight')
    plt.close()
    
    print("  📊 Analysis plot saved")

def main():
    """Main function."""
    print("🚀 Testing Meaningful Classification Tasks")
    print("=" * 50)
    
    experiment_id = 'experiment8'
    
    # Load features and labels
    features, labels = load_features_and_labels(experiment_id)
    
    if features is None or labels is None:
        print("❌ Could not load features or labels!")
        return
    
    # Run comprehensive testing
    results = run_comprehensive_testing(features, labels)
    
    # Create report
    best_overall, best_by_task = create_comprehensive_report(results, experiment_id)
    
    # Create visualizations
    create_visualizations(results, experiment_id)
    
    print("\n🎉 Meaningful Classification Testing Completed!")
    print("=" * 50)
    print(f"🏆 Best Overall Result:")
    print(f"  Task: {best_overall['task']}")
    print(f"  Feature: {best_overall['feature']}")
    print(f"  Model: {best_overall['model']}")
    print(f"  Accuracy: {best_overall['accuracy']:.3f} ± {best_overall['std']:.3f}")
    print(f"  Classes: {best_overall['n_classes']}")
    
    print(f"\n📊 Best Results by Task:")
    for _, row in best_by_task.iterrows():
        print(f"  {row['task']}: {row['accuracy']:.3f} ({row['feature']} + {row['model']})")

if __name__ == "__main__":
    main()
