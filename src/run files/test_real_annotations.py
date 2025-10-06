#!/usr/bin/env python3
"""
Test Real Annotation Labels with Existing Features
================================================

This script tests the real 7-class annotation labels with our existing features
to compare with the paper results (72.9% accuracy).

Usage:
    python test_real_annotations.py
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
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def load_features_and_real_labels(features_experiment_id, labels_experiment_id):
    """Load features and real annotation labels."""
    print("📊 Loading features and real annotation labels...")
    
    # Load features
    features_path = Path(f'data/features/{features_experiment_id}')
    features = {}
    
    for extractor_dir in features_path.iterdir():
        if extractor_dir.is_dir():
            extractor_name = extractor_dir.name
            feature_files = list(extractor_dir.glob('*.npy'))
            if feature_files:
                # Load the main feature file (usually the first one)
                feature_file = feature_files[0]
                features[extractor_name] = np.load(feature_file)
                print(f"  ✅ Loaded {extractor_name}: {features[extractor_name].shape}")
    
    # Load real annotation labels
    labels_path = Path(f'data/labels/{labels_experiment_id}')
    if not labels_path.exists():
        print("❌ No real annotation labels found. Run create_real_annotation_labels.py first.")
        return None, None
    
    labels = {}
    for label_file in labels_path.glob('*.npy'):
        label_name = label_file.stem
        labels[label_name] = np.load(label_file)
        print(f"  ✅ Loaded {label_name}: {labels[label_name].shape}")
    
    # Load metadata
    metadata_file = labels_path / 'real_annotation_metadata.json'
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        print(f"  📋 Categories: {metadata.get('categories', {})}")
    
    return features, labels

def test_classification_with_real_labels(features, labels, feature_name, label_name, model_name, model):
    """Test classification with real labels."""
    try:
        # Get feature and label data
        feature_data = features[feature_name]
        label_data = labels[label_name]
        
        # Ensure features are 2D
        if feature_data.ndim == 1:
            feature_data = feature_data.reshape(-1, 1)
        elif feature_data.ndim == 3:
            # For 3D features (like EEGNet), flatten
            feature_data = feature_data.reshape(feature_data.shape[0], -1)
        
        # Ensure same number of samples
        min_samples = min(feature_data.shape[0], label_data.shape[0])
        feature_data = feature_data[:min_samples, :]
        label_data = label_data[:min_samples]
        
        # Remove background samples (label = -1) for classification
        valid_mask = label_data >= 0
        if np.sum(valid_mask) == 0:
            return {
                'feature': feature_name,
                'label': label_name,
                'model': model_name,
                'accuracy': 0.0,
                'status': 'failed',
                'error': 'No valid samples (all background)'
            }
        
        feature_data = feature_data[valid_mask]
        label_data = label_data[valid_mask]
        
        # Check if we have enough classes
        unique_labels = np.unique(label_data)
        if len(unique_labels) < 2:
            return {
                'feature': feature_name,
                'label': label_name,
                'model': model_name,
                'accuracy': 0.0,
                'status': 'failed',
                'error': 'Not enough classes'
            }
        
        # Scale features
        scaler = StandardScaler()
        feature_data_scaled = scaler.fit_transform(feature_data)
        
        # Use stratified cross-validation
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(model, feature_data_scaled, label_data, cv=cv, scoring='accuracy')
        
        return {
            'feature': feature_name,
            'label': label_name,
            'model': model_name,
            'accuracy': scores.mean(),
            'std': scores.std(),
            'n_classes': len(unique_labels),
            'n_samples': len(label_data),
            'class_distribution': np.bincount(label_data).tolist(),
            'status': 'success'
        }
        
    except Exception as e:
        return {
            'feature': feature_name,
            'label': label_name,
            'model': model_name,
            'accuracy': 0.0,
            'status': 'failed',
            'error': str(e)
        }

def run_comprehensive_real_annotation_testing(features, labels):
    """Run comprehensive testing with real annotations."""
    print("🎯 Running comprehensive testing with real annotations...")
    
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
        print(f"  📊 Shape: {label_data.shape}, classes: {np.unique(label_data)}")
        
        # Test with each feature extractor
        for feature_name, feature_data in features.items():
            for model_name, model in models.items():
                result = test_classification_with_real_labels(
                    features, labels, feature_name, label_name, model_name, model
                )
                all_results.append(result)
    
    return all_results

def create_real_annotation_report(results, experiment_id):
    """Create a comprehensive report of real annotation results."""
    print("📋 Creating real annotation report...")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Create results directory
    results_dir = Path(f"results/05_modelling/{experiment_id}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Summary statistics
    successful_results = df[df['status'] == 'success']
    
    if len(successful_results) == 0:
        print("❌ No successful results to report!")
        return None, None
    
    # Best results by label type
    best_by_label = successful_results.loc[successful_results.groupby('label')['accuracy'].idxmax()]
    
    # Best results overall
    best_overall = successful_results.loc[successful_results['accuracy'].idxmax()]
    
    # Compare with paper results
    paper_results = {
        '7_class': 0.729,  # 72.9%
        'color_discrimination': 0.671,  # 67.1%
        '14_class': 0.521,  # 52.1%
        'real_time': 0.737  # 73.7%
    }
    
    # Create report
    report = f"""
# Real Annotation Classification Results Report
Experiment: {experiment_id}
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- Total combinations tested: {len(results)}
- Successful: {len(successful_results)}
- Failed: {len(results) - len(successful_results)}
- Best overall accuracy: {best_overall['accuracy']:.3f} ± {best_overall['std']:.3f}
- Best feature: {best_overall['feature']}
- Best model: {best_overall['model']}
- Best label type: {best_overall['label']}

## Comparison with Paper Results
- **Paper 7-class accuracy**: {paper_results['7_class']:.3f} (72.9%)
- **Our best 7-class accuracy**: {best_overall['accuracy']:.3f} ({best_overall['accuracy']*100:.1f}%)
- **Performance vs paper**: {((best_overall['accuracy'] / paper_results['7_class']) - 1) * 100:+.1f}%

## Best Results by Label Type
"""
    
    for _, row in best_by_label.iterrows():
        paper_target = paper_results.get('7_class', 0.729) if 'trial_based' in row['label'] else paper_results.get('14_class', 0.521)
        vs_paper = ((row['accuracy'] / paper_target) - 1) * 100
        
        report += f"""
### {row['label']}
- **Best Accuracy**: {row['accuracy']:.3f} ± {row['std']:.3f}
- **Best Feature**: {row['feature']}
- **Best Model**: {row['model']}
- **Classes**: {row['n_classes']}
- **Samples**: {row['n_samples']}
- **vs Paper**: {vs_paper:+.1f}%
"""
    
    # Top 10 results
    report += f"""
## Top 10 Results Overall
"""
    top_10 = successful_results.nlargest(10, 'accuracy')
    for i, (_, row) in enumerate(top_10.iterrows(), 1):
        report += f"{i}. **{row['label']}** + {row['feature']} + {row['model']}: {row['accuracy']:.3f} ± {row['std']:.3f}\n"
    
    # Save report
    report_file = results_dir / f"real_annotation_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_file, 'w') as f:
        f.write(report)
    
    # Save results as JSON
    results_file = results_dir / f"real_annotation_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"📁 Report saved: {report_file}")
    print(f"📁 Results saved: {results_file}")
    
    return best_overall, best_by_label

def create_real_annotation_visualizations(results, experiment_id):
    """Create visualizations of real annotation results."""
    print("🎨 Creating real annotation visualizations...")
    
    df = pd.DataFrame(results)
    successful_results = df[df['status'] == 'success']
    
    if len(successful_results) == 0:
        print("❌ No successful results to visualize!")
        return
    
    results_dir = Path(f"results/05_modelling/{experiment_id}")
    
    # Create comprehensive visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Accuracy by label type
    ax1_data = successful_results.groupby('label')['accuracy'].max().sort_values(ascending=True)
    ax1_data.plot(kind='barh', ax=ax1, color='skyblue')
    ax1.set_title('Best Accuracy by Label Type', fontweight='bold')
    ax1.set_xlabel('Accuracy')
    ax1.axvline(x=0.729, color='red', linestyle='--', label='Paper (72.9%)')
    ax1.legend()
    
    # 2. Accuracy by feature extractor
    ax2_data = successful_results.groupby('feature')['accuracy'].max().sort_values(ascending=True)
    ax2_data.plot(kind='barh', ax=ax2, color='lightgreen')
    ax2.set_title('Best Accuracy by Feature Extractor', fontweight='bold')
    ax2.set_xlabel('Accuracy')
    
    # 3. Accuracy by model
    ax3_data = successful_results.groupby('model')['accuracy'].max().sort_values(ascending=True)
    ax3_data.plot(kind='barh', ax=ax3, color='lightcoral')
    ax3.set_title('Best Accuracy by Model', fontweight='bold')
    ax3.set_xlabel('Accuracy')
    
    # 4. Class distribution
    trial_based_results = successful_results[successful_results['label'] == 'trial_based']
    if len(trial_based_results) > 0:
        best_trial_result = trial_based_results.loc[trial_based_results['accuracy'].idxmax()]
        class_dist = best_trial_result['class_distribution']
        ax4.bar(range(len(class_dist)), class_dist, color='gold', alpha=0.7)
        ax4.set_title('Class Distribution (Trial-based)', fontweight='bold')
        ax4.set_xlabel('Class ID')
        ax4.set_ylabel('Number of Samples')
        ax4.set_xticks(range(len(class_dist)))
        ax4.set_xticklabels([f'Class {i}' for i in range(len(class_dist))])
    
    plt.suptitle('Real Annotation Classification Results Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'real_annotation_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(results_dir / 'real_annotation_analysis.svg', bbox_inches='tight')
    plt.close()
    
    print("  📊 Analysis plot saved")

def main():
    """Main function."""
    print("🚀 Testing Real Annotation Labels with Existing Features")
    print("=" * 60)
    
    # Use the latest experiment with features
    experiment_id = 'experiment8'  # Has the most complete features
    labels_experiment_id = 'experiment9'  # Has the real annotation labels
    
    # Load features and real labels
    features, labels = load_features_and_real_labels(experiment_id, labels_experiment_id)
    
    if features is None or labels is None:
        print("❌ Could not load features or labels!")
        return
    
    # Run comprehensive testing
    results = run_comprehensive_real_annotation_testing(features, labels)
    
    # Create report
    best_overall, best_by_label = create_real_annotation_report(results, experiment_id)
    
    if best_overall is None:
        print("❌ No successful results to report!")
        return
    
    # Create visualizations
    create_real_annotation_visualizations(results, experiment_id)
    
    print("\n🎉 Real Annotation Testing Completed!")
    print("=" * 60)
    print(f"🏆 Best Overall Result:")
    print(f"  Label Type: {best_overall['label']}")
    print(f"  Feature: {best_overall['feature']}")
    print(f"  Model: {best_overall['model']}")
    print(f"  Accuracy: {best_overall['accuracy']:.3f} ± {best_overall['std']:.3f}")
    print(f"  Classes: {best_overall['n_classes']}")
    print(f"  Samples: {best_overall['n_samples']}")
    
    # Compare with paper
    paper_accuracy = 0.729  # 72.9%
    vs_paper = ((best_overall['accuracy'] / paper_accuracy) - 1) * 100
    print(f"  vs Paper (72.9%): {vs_paper:+.1f}%")
    
    print(f"\n📊 Best Results by Label Type:")
    for _, row in best_by_label.iterrows():
        print(f"  {row['label']}: {row['accuracy']:.3f} ({row['feature']} + {row['model']})")

if __name__ == "__main__":
    main()
