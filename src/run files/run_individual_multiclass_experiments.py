#!/usr/bin/env python3
"""
Individual Multiclass Experiment Runner
=====================================

This script allows running individual experiment combinations for detailed analysis.
Useful for testing specific feature extractor + model + multiclass problem combinations.

Usage:
    python run_individual_multiclass_experiments.py --extractor comprehensive --model "Random Forest" --problem 7class
    python run_individual_multiclass_experiments.py --extractor eegnet --model SVM --problem 14class
    python run_individual_multiclass_experiments.py --list-options
"""

import sys
import os
sys.path.append('src')

import argparse
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class IndividualMulticlassExperiment:
    """Individual multiclass experiment runner."""
    
    def __init__(self):
        self.experiment_id = self._get_next_experiment_id()
        self.results_dir = Path(f"results/05_modelling/{self.experiment_id}")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Define categories
        self.categories_7class = {
            'digit': 0, 'kanji': 1, 'face': 2, 'body': 3, 
            'object': 4, 'hiragana': 5, 'line': 6
        }
        
        self.categories_14class = {
            # Gray versions (0-6)
            'digit_gray': 0, 'kanji_gray': 1, 'face_gray': 2, 'body_gray': 3,
            'object_gray': 4, 'hiragana_gray': 5, 'line_gray': 6,
            # Color versions (7-13)
            'digit_color': 7, 'kanji_color': 8, 'face_color': 9, 'body_color': 10,
            'object_color': 11, 'hiragana_color': 12, 'line_color': 13
        }
        
        # Define models
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        # Available feature extractors
        self.available_extractors = ['comprehensive', 'template_correlation', 'eegnet', 'transformer']
        
        print(f"🚀 Individual Multiclass Experiment")
        print(f"📁 Experiment ID: {self.experiment_id}")
        print(f"📁 Results Directory: {self.results_dir}")
    
    def _get_next_experiment_id(self):
        """Get next experiment ID."""
        results_dir = Path("results/05_modelling")
        if not results_dir.exists():
            return "experiment1"
        
        existing_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('experiment')]
        if not existing_dirs:
            return "experiment1"
        
        numbers = []
        for d in existing_dirs:
            try:
                num = int(d.name.replace('experiment', ''))
                numbers.append(num)
            except ValueError:
                continue
        
        return f"experiment{max(numbers) + 1 if numbers else 1}"
    
    def list_available_options(self):
        """List all available options for experiments."""
        print("📋 Available Experiment Options")
        print("=" * 50)
        
        print("\n🔧 Feature Extractors:")
        for i, extractor in enumerate(self.available_extractors, 1):
            print(f"  {i}. {extractor}")
        
        print("\n🤖 Models:")
        for i, model in enumerate(self.models.keys(), 1):
            print(f"  {i}. {model}")
        
        print("\n🎯 Multiclass Problems:")
        print("  1. 7class - 7 visual categories (digit, kanji, face, body, object, hiragana, line)")
        print("  2. 14class - 14 categories (7 categories × 2 colors)")
        
        print("\n💡 Example Commands:")
        print("  python run_individual_multiclass_experiments.py --extractor comprehensive --model \"Random Forest\" --problem 7class")
        print("  python run_individual_multiclass_experiments.py --extractor eegnet --model SVM --problem 14class")
        print("  python run_individual_multiclass_experiments.py --extractor transformer --model \"Logistic Regression\" --problem 7class")
    
    def load_specific_features(self, extractor_name):
        """Load features for a specific extractor."""
        features_path = Path(f'data/features/experiment8/{extractor_name}')
        
        if not features_path.exists():
            raise FileNotFoundError(f"Features not found for extractor: {extractor_name}")
        
        feature_files = list(features_path.glob('*.npy'))
        if not feature_files:
            raise FileNotFoundError(f"No feature files found in: {features_path}")
        
        feature_file = feature_files[0]
        features = np.load(feature_file)
        
        print(f"  ✅ Loaded {extractor_name}: {features.shape}")
        return features
    
    def load_labels(self, problem_type):
        """Load labels for specific problem type."""
        labels_path = Path('data/labels/experiment9')
        
        if problem_type == '7class':
            labels = np.load(labels_path / 'trial_based.npy')
            categories = self.categories_7class
        elif problem_type == '14class':
            labels = np.load(labels_path / 'color_aware.npy')
            categories = self.categories_14class
        else:
            raise ValueError(f"Unknown problem type: {problem_type}")
        
        # Remove background trials
        valid_mask = labels >= 0
        labels = labels[valid_mask]
        
        print(f"  ✅ Loaded {problem_type} labels: {labels.shape}")
        print(f"  📊 Classes: {len(categories)}")
        print(f"  📊 Class distribution: {np.bincount(labels)}")
        
        return labels, categories
    
    def prepare_features(self, features, valid_mask):
        """Prepare features for classification."""
        # Ensure features are 2D
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        elif features.ndim == 3:
            # For 3D features (like EEGNet), flatten
            features = features.reshape(features.shape[0], -1)
        
        # Apply valid mask to remove background trials
        if features.shape[0] == len(valid_mask):
            features = features[valid_mask]
        
        return features
    
    def run_detailed_experiment(self, extractor_name, model_name, problem_type):
        """Run a detailed individual experiment."""
        print(f"\n🎯 Running Individual Experiment:")
        print(f"  Feature Extractor: {extractor_name}")
        print(f"  Model: {model_name}")
        print(f"  Problem Type: {problem_type}")
        print("=" * 50)
        
        # Load data
        features = self.load_specific_features(extractor_name)
        labels, categories = self.load_labels(problem_type)
        
        # Create valid mask (remove background)
        if problem_type == '7class':
            trial_labels = np.load(Path('data/labels/experiment9/trial_based.npy'))
        else:
            trial_labels = np.load(Path('data/labels/experiment9/color_aware.npy'))
        
        valid_mask = trial_labels >= 0
        
        # Prepare features
        features = self.prepare_features(features, valid_mask)
        
        print(f"  📊 Final data shape: {features.shape}")
        print(f"  📊 Labels shape: {labels.shape}")
        
        # Get model
        model = self.models[model_name]
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Run cross-validation
        print("  🔄 Running cross-validation...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, features_scaled, labels, cv=cv, scoring='accuracy')
        
        # Train final model for detailed analysis
        print("  🎓 Training final model...")
        model.fit(features_scaled, labels)
        predictions = model.predict(features_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        class_report = classification_report(labels, predictions, output_dict=True)
        conf_matrix = confusion_matrix(labels, predictions)
        
        # Create results
        results = {
            'experiment_id': self.experiment_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'extractor': extractor_name,
            'model': model_name,
            'problem_type': problem_type,
            'cv_accuracy_mean': scores.mean(),
            'cv_accuracy_std': scores.std(),
            'cv_scores': scores.tolist(),
            'train_accuracy': accuracy,
            'n_classes': len(categories),
            'n_samples': len(labels),
            'class_distribution': np.bincount(labels).tolist(),
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'feature_shape': features.shape,
            'categories': categories
        }
        
        # Print results
        print(f"\n📊 Results:")
        print(f"  CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
        print(f"  Train Accuracy: {accuracy:.3f}")
        print(f"  Classes: {len(categories)}")
        print(f"  Samples: {len(labels)}")
        
        # Save results
        self.save_detailed_results(results)
        
        # Create visualizations
        self.create_detailed_visualizations(results, features_scaled, labels, predictions, categories)
        
        return results
    
    def save_detailed_results(self, results):
        """Save detailed results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON results
        results_file = self.results_dir / f"individual_experiment_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save markdown report
        report_file = self.results_dir / f"individual_experiment_{timestamp}.md"
        with open(report_file, 'w') as f:
            f.write(self._create_markdown_report(results))
        
        print(f"  📁 Results saved: {results_file}")
        print(f"  📁 Report saved: {report_file}")
    
    def _create_markdown_report(self, results):
        """Create markdown report."""
        report = f"""# Individual Multiclass Experiment Report

**Experiment ID**: {results['experiment_id']}
**Timestamp**: {results['timestamp']}

## Configuration
- **Feature Extractor**: {results['extractor']}
- **Model**: {results['model']}
- **Problem Type**: {results['problem_type']}
- **Classes**: {results['n_classes']}
- **Samples**: {results['n_samples']}

## Results
- **CV Accuracy**: {results['cv_accuracy_mean']:.3f} ± {results['cv_accuracy_std']:.3f}
- **Train Accuracy**: {results['train_accuracy']:.3f}
- **CV Scores**: {[f'{s:.3f}' for s in results['cv_scores']]}

## Class Distribution
"""
        for class_name, class_id in results['categories'].items():
            count = results['class_distribution'][class_id] if class_id < len(results['class_distribution']) else 0
            report += f"- **{class_name}**: {count} samples\n"
        
        report += f"""
## Classification Report
```
{results['classification_report']}
```

## Confusion Matrix
```
{np.array(results['confusion_matrix'])}
```
"""
        return report
    
    def create_detailed_visualizations(self, results, features, labels, predictions, categories):
        """Create detailed visualizations for individual experiment."""
        print("  🎨 Creating detailed visualizations...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Confusion Matrix
        plt.subplot(2, 4, 1)
        conf_matrix = np.array(results['confusion_matrix'])
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=list(categories.keys()), 
                   yticklabels=list(categories.keys()))
        plt.title('Confusion Matrix', fontweight='bold')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        # 2. CV Scores
        plt.subplot(2, 4, 2)
        cv_scores = results['cv_scores']
        plt.bar(range(1, len(cv_scores) + 1), cv_scores, color='lightgreen', alpha=0.7)
        plt.axhline(np.mean(cv_scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(cv_scores):.3f}')
        plt.title('Cross-Validation Scores', fontweight='bold')
        plt.xlabel('Fold')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.ylim(0, 1)
        
        # 3. Class Distribution
        plt.subplot(2, 4, 3)
        class_counts = results['class_distribution']
        class_names = list(categories.keys())
        plt.bar(class_names, class_counts, color='skyblue', alpha=0.7)
        plt.title('Class Distribution', fontweight='bold')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # 4. Accuracy Summary
        plt.subplot(2, 4, 4)
        plt.axis('off')
        summary_text = f"""
🏆 EXPERIMENT SUMMARY

Feature Extractor: {results['extractor']}
Model: {results['model']}
Problem Type: {results['problem_type']}

📊 PERFORMANCE
CV Accuracy: {results['cv_accuracy_mean']:.3f} ± {results['cv_accuracy_std']:.3f}
Train Accuracy: {results['train_accuracy']:.3f}

📈 DATA INFO
Classes: {results['n_classes']}
Samples: {results['n_samples']}
Features: {results['feature_shape'][1]}

🎯 BEST FOLD: {max(results['cv_scores']):.3f}
        """
        plt.text(0.1, 0.5, summary_text, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
        
        # 5. Feature Importance (if available)
        plt.subplot(2, 4, 5)
        if hasattr(results.get('model'), 'feature_importances_'):
            # This would need the actual model object
            plt.text(0.5, 0.5, 'Feature Importance\n(Not available in this view)', 
                    ha='center', va='center', transform=plt.gca().transAxes)
        else:
            plt.text(0.5, 0.5, 'Feature Importance\n(Not available for this model)', 
                    ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Feature Importance', fontweight='bold')
        plt.axis('off')
        
        # 6. Prediction Accuracy by Class
        plt.subplot(2, 4, 6)
        class_accuracies = []
        for i in range(len(categories)):
            mask = labels == i
            if np.sum(mask) > 0:
                class_acc = np.mean(predictions[mask] == labels[mask])
                class_accuracies.append(class_acc)
            else:
                class_accuracies.append(0)
        
        plt.bar(range(len(class_accuracies)), class_accuracies, color='orange', alpha=0.7)
        plt.title('Accuracy by Class', fontweight='bold')
        plt.xlabel('Class')
        plt.ylabel('Accuracy')
        plt.xticks(range(len(class_names)), class_names, rotation=45)
        plt.ylim(0, 1)
        
        # 7. Error Analysis
        plt.subplot(2, 4, 7)
        errors = predictions != labels
        error_rate = np.mean(errors)
        correct_rate = 1 - error_rate
        
        plt.pie([correct_rate, error_rate], labels=['Correct', 'Error'], 
               colors=['lightgreen', 'lightcoral'], autopct='%1.1f%%')
        plt.title('Prediction Accuracy', fontweight='bold')
        
        # 8. Performance Comparison
        plt.subplot(2, 4, 8)
        metrics = ['CV Mean', 'CV Std', 'Train']
        values = [results['cv_accuracy_mean'], results['cv_accuracy_std'], results['train_accuracy']]
        colors = ['blue', 'red', 'green']
        
        bars = plt.bar(metrics, values, color=colors, alpha=0.7)
        plt.title('Performance Metrics', fontweight='bold')
        plt.ylabel('Value')
        plt.ylim(0, 1)
        
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', fontweight='bold')
        
        plt.suptitle(f'Individual Experiment: {results["extractor"]} + {results["model"]} ({results["problem_type"]})', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plots
        plt.savefig(self.results_dir / f'individual_experiment_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.results_dir / f'individual_experiment_{timestamp}.svg', bbox_inches='tight')
        plt.close()
        
        print(f"    📊 Detailed visualizations saved")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run individual multiclass ECoG experiments')
    parser.add_argument('--extractor', type=str, help='Feature extractor name')
    parser.add_argument('--model', type=str, help='Model name')
    parser.add_argument('--problem', type=str, choices=['7class', '14class'], help='Problem type')
    parser.add_argument('--list-options', action='store_true', help='List available options')
    
    args = parser.parse_args()
    
    experiment = IndividualMulticlassExperiment()
    
    if args.list_options:
        experiment.list_available_options()
        return
    
    if not all([args.extractor, args.model, args.problem]):
        print("❌ Error: Please provide --extractor, --model, and --problem")
        print("💡 Use --list-options to see available choices")
        return
    
    if args.extractor not in experiment.available_extractors:
        print(f"❌ Error: Unknown extractor '{args.extractor}'")
        print(f"💡 Available extractors: {experiment.available_extractors}")
        return
    
    if args.model not in experiment.models:
        print(f"❌ Error: Unknown model '{args.model}'")
        print(f"💡 Available models: {list(experiment.models.keys())}")
        return
    
    # Run the experiment
    start_time = time.time()
    results = experiment.run_detailed_experiment(args.extractor, args.model, args.problem)
    elapsed_time = time.time() - start_time
    
    print(f"\n🎉 Individual Experiment Completed!")
    print("=" * 50)
    print(f"⏱️ Time: {elapsed_time:.1f} seconds")
    print(f"📁 Results: {experiment.results_dir}")
    print(f"🏆 CV Accuracy: {results['cv_accuracy_mean']:.3f} ± {results['cv_accuracy_std']:.3f}")

if __name__ == "__main__":
    main()
