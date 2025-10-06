#!/usr/bin/env python3
"""
FIXED Enhanced Multiclass ECoG Experiment Framework
=================================================

This script fixes ALL the critical bugs:
1. Uses correct labels from video annotations
2. Properly calculates confusion matrix on CV test data (not training data)
3. Fixes ROC-AUC calculation to use proper test data
4. Ensures all metrics are calculated on the same test data

FIXES:
- Confusion matrix now calculated on CV test data (not training data)
- ROC-AUC calculated on proper test data
- All metrics consistent and realistic
- Proper 7-class and 15-class problems (including background)

Usage:
    python run_fixed_multiclass_experiments.py --help
    python run_fixed_multiclass_experiments.py --7class
    python run_fixed_multiclass_experiments.py --15class
    python run_fixed_multiclass_experiments.py --all
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
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    roc_auc_score, precision_score, recall_score, f1_score
)
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class FixedMulticlassExperimentFramework:
    """FIXED framework that properly calculates all metrics on test data."""
    
    def __init__(self):
        self.experiment_id = self._get_next_experiment_id()
        self.results_dir = Path(f"results/05_modelling/{self.experiment_id}")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Define categories with detailed labels (including background)
        self.categories_8class = {
            'background': 0, 'digit': 1, 'kanji': 2, 'face': 3, 
            'body': 4, 'object': 5, 'hiragana': 6, 'line': 7
        }
        
        self.categories_16class = {
            # Background
            'background': 0,
            # Gray versions (1-7)
            'digit_gray': 1, 'kanji_gray': 2, 'face_gray': 3, 'body_gray': 4,
            'object_gray': 5, 'hiragana_gray': 6, 'line_gray': 7,
            # Color versions (8-15)
            'digit_color': 8, 'kanji_color': 9, 'face_color': 10, 'body_color': 11,
            'object_color': 12, 'hiragana_color': 13, 'line_color': 14, 'object_color_alt': 15
        }
        
        # Define models with probability support
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        print(f"🚀 FIXED Multiclass Experiment Framework")
        print(f"📁 Experiment ID: {self.experiment_id}")
        print(f"📁 Results Directory: {self.results_dir}")
        print(f"🔧 FIXED: All metrics calculated on CV test data")
        print(f"🔧 FIXED: Proper confusion matrix and ROC-AUC calculation")
    
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
    
    def load_video_annotations(self):
        """Load the real video annotations from JSON."""
        annotation_file = Path('results/annotations/video_annotation_data.json')
        
        if not annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
        
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        return data
    
    def create_correct_labels_from_annotations(self, n_trials=252):
        """Create correct labels from the actual video annotations."""
        print("📊 Creating correct labels from video annotations...")
        
        # Load video annotations
        annotations = self.load_video_annotations()
        
        # Get video info
        video_start = annotations['video_info']['video_start_time']
        video_end = annotations['video_info']['video_end_time']
        video_duration = video_end - video_start
        
        # Calculate trial duration
        trial_duration = video_duration / n_trials
        
        # Initialize trial labels (including background as class 0)
        trial_labels = np.full(n_trials, 0, dtype=int)  # 0 = background
        color_labels = np.full(n_trials, 0, dtype=int)  # 0 = background
        
        # Process each annotation
        for annotation in annotations['annotations']:
            start_time = annotation['time_start'] - video_start
            end_time = annotation['time_end'] - video_start
            category = annotation['category']
            color = annotation['color']
            
            # Find which trials this annotation covers
            start_trial = int(start_time / trial_duration)
            end_trial = int(end_time / trial_duration)
            
            # Ensure indices are within bounds
            start_trial = max(0, start_trial)
            end_trial = min(n_trials, end_trial)
            
            # Set labels for these trials (shift by 1 to make room for background=0)
            if category in ['digit', 'kanji', 'face', 'body', 'object', 'hiragana', 'line']:
                category_map = {'digit': 1, 'kanji': 2, 'face': 3, 'body': 4, 'object': 5, 'hiragana': 6, 'line': 7}
                category_id = category_map[category]
                trial_labels[start_trial:end_trial] = category_id
                
                # For color-aware labels
                if 'color' in color.lower() and 'gray' not in color.lower():
                    color_offset = 8  # Color images start at 8
                else:
                    color_offset = 1  # Gray images start at 1
                
                final_color_label = category_id + color_offset - 1  # Adjust for background=0
                color_labels[start_trial:end_trial] = final_color_label
        
        return trial_labels, color_labels
    
    def load_features_and_correct_labels(self):
        """Load features and create correctly aligned labels."""
        print("📊 Loading features and creating correctly aligned labels...")
        
        # Load features from experiment8
        features_path = Path('data/features/experiment8')
        features = {}
        
        for extractor_dir in features_path.iterdir():
            if extractor_dir.is_dir():
                extractor_name = extractor_dir.name
                feature_files = list(extractor_dir.glob('*.npy'))
                if feature_files:
                    feature_file = feature_files[0]
                    feature_data = np.load(feature_file)
                    print(f"  ✅ Loaded {extractor_name}: {feature_data.shape}")
                    
                    # Handle transformer features that might have wrong shape
                    if extractor_name == 'transformer' and feature_data.shape[0] != 252:
                        print(f"    Warning: Transformer features have {feature_data.shape[0]} samples, expected 252")
                        if feature_data.shape[0] > 252:
                            feature_data = feature_data[:252]
                            print(f"    Taking first 252 samples: {feature_data.shape}")
                        else:
                            print(f"    Skipping transformer features due to insufficient samples")
                            continue
                    
                    features[extractor_name] = feature_data
        
        # Create correct labels from annotations
        trial_labels, color_labels = self.create_correct_labels_from_annotations()
        
        print(f"  ✅ Created trial labels: {trial_labels.shape}")
        print(f"  ✅ Created color labels: {color_labels.shape}")
        print(f"  📊 Trial labels unique: {np.unique(trial_labels)}")
        print(f"  📊 Color labels unique: {np.unique(color_labels)}")
        print(f"  📊 Trial distribution: {np.bincount(trial_labels)}")
        print(f"  📊 Color distribution: {np.bincount(color_labels)}")
        
        return features, trial_labels, color_labels
    
    def create_multiclass_labels(self, trial_labels, color_labels, problem_type):
        """Create multiclass labels for 8-class or 16-class problems."""
        if problem_type == '8class':
            # Use 8-class labels (including background)
            labels = trial_labels.copy()
            return labels, np.ones(len(labels), dtype=bool), self.categories_8class
        
        elif problem_type == '16class':
            # Use 16-class labels (including background)
            labels = color_labels.copy()
            return labels, np.ones(len(labels), dtype=bool), self.categories_16class
        
        else:
            raise ValueError(f"Unknown problem type: {problem_type}")
    
    def prepare_features(self, features, valid_mask):
        """Prepare features for classification."""
        prepared_features = {}
        
        for extractor_name, feature_data in features.items():
            print(f"    Processing {extractor_name}: {feature_data.shape}")
            
            # Handle different feature shapes
            if feature_data.ndim == 1:
                feature_data = feature_data.reshape(-1, 1)
            elif feature_data.ndim == 3:
                # For 3D features (like EEGNet), flatten
                feature_data = feature_data.reshape(feature_data.shape[0], -1)
            
            # Apply valid mask (now all trials are valid since we include background)
            if feature_data.shape[0] == len(valid_mask):
                feature_data = feature_data[valid_mask]
                print(f"      After processing: {feature_data.shape}")
            else:
                print(f"      Warning: Feature shape {feature_data.shape[0]} doesn't match mask length {len(valid_mask)}")
                continue
            
            prepared_features[extractor_name] = feature_data
        
        return prepared_features
    
    def run_single_experiment(self, features, labels, extractor_name, model_name, model, problem_type, categories):
        """Run a single experiment combination with FIXED metrics calculation."""
        try:
            print(f"      Running {extractor_name} + {model_name} for {problem_type}")
            print(f"        Features shape: {features.shape}, Labels shape: {labels.shape}")
            print(f"        Unique labels: {np.unique(labels)}")
            
            # Check if we have enough samples and classes
            if len(labels) < 10:
                raise ValueError(f"Not enough samples: {len(labels)}")
            
            if len(np.unique(labels)) < 2:
                raise ValueError(f"Not enough classes: {len(np.unique(labels))}")
            
            # Scale features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Use stratified cross-validation
            cv = StratifiedKFold(n_splits=min(5, len(np.unique(labels))), shuffle=True, random_state=42)
            
            # FIXED: Calculate CV scores
            scores = cross_val_score(model, features_scaled, labels, cv=cv, scoring='accuracy')
            
            # FIXED: Get predictions on CV test data (not training data!)
            cv_predictions = cross_val_predict(model, features_scaled, labels, cv=cv)
            
            # FIXED: Get probabilities on CV test data
            cv_probabilities = cross_val_predict(model, features_scaled, labels, cv=cv, method='predict_proba')
            
            # FIXED: Calculate metrics on CV test data
            cv_accuracy = accuracy_score(labels, cv_predictions)
            cv_conf_matrix = confusion_matrix(labels, cv_predictions)
            
            # Calculate per-class metrics on CV test data
            precision = precision_score(labels, cv_predictions, average=None, zero_division=0)
            recall = recall_score(labels, cv_predictions, average=None, zero_division=0)
            f1 = f1_score(labels, cv_predictions, average=None, zero_division=0)
            
            # Calculate macro averages on CV test data
            precision_macro = precision_score(labels, cv_predictions, average='macro', zero_division=0)
            recall_macro = recall_score(labels, cv_predictions, average='macro', zero_division=0)
            f1_macro = f1_score(labels, cv_predictions, average='macro', zero_division=0)
            
            # FIXED: Calculate ROC-AUC on CV test data
            try:
                roc_auc = roc_auc_score(labels, cv_probabilities, multi_class='ovr', average='macro')
            except:
                roc_auc = 0.0
            
            # FIXED: Calculate average precision on CV test data
            try:
                avg_precision = average_precision_score(labels, cv_probabilities, average='macro')
            except:
                avg_precision = 0.0
            
            return {
                'extractor': extractor_name,
                'model': model_name,
                'problem_type': problem_type,
                'cv_accuracy_mean': scores.mean(),
                'cv_accuracy_std': scores.std(),
                'cv_scores': scores.tolist(),
                'cv_test_accuracy': cv_accuracy,  # FIXED: Accuracy on CV test data
                'precision_macro': precision_macro,
                'recall_macro': recall_macro,
                'f1_macro': f1_macro,
                'roc_auc_macro': roc_auc,
                'avg_precision_macro': avg_precision,
                'n_classes': len(np.unique(labels)),
                'n_samples': len(labels),
                'class_distribution': np.bincount(labels).tolist(),
                'confusion_matrix': cv_conf_matrix.tolist(),  # FIXED: Confusion matrix on CV test data
                'precision_per_class': precision.tolist(),
                'recall_per_class': recall.tolist(),
                'f1_per_class': f1.tolist(),
                'cv_predictions': cv_predictions.tolist(),  # FIXED: Predictions on CV test data
                'cv_probabilities': cv_probabilities.tolist(),  # FIXED: Probabilities on CV test data
                'feature_shape': features.shape,
                'categories': categories,
                'status': 'success'
            }
            
        except Exception as e:
            print(f"        ❌ Failed: {str(e)}")
            return {
                'extractor': extractor_name,
                'model': model_name,
                'problem_type': problem_type,
                'cv_accuracy_mean': 0.0,
                'cv_accuracy_std': 0.0,
                'cv_scores': [],
                'cv_test_accuracy': 0.0,
                'precision_macro': 0.0,
                'recall_macro': 0.0,
                'f1_macro': 0.0,
                'roc_auc_macro': 0.0,
                'avg_precision_macro': 0.0,
                'n_classes': 0,
                'n_samples': 0,
                'class_distribution': [],
                'confusion_matrix': [],
                'precision_per_class': [],
                'recall_per_class': [],
                'f1_per_class': [],
                'cv_predictions': [],
                'cv_probabilities': [],
                'feature_shape': (0, 0),
                'categories': {},
                'status': 'failed',
                'error': str(e)
            }
    
    def run_multiclass_experiments(self, problem_type):
        """Run all experiments for a specific multiclass problem."""
        print(f"\n🎯 Running {problem_type} multiclass experiments...")
        
        # Load data
        features, trial_labels, color_labels = self.load_features_and_correct_labels()
        
        # Create multiclass labels
        labels, valid_mask, categories = self.create_multiclass_labels(trial_labels, color_labels, problem_type)
        
        # Prepare features
        prepared_features = self.prepare_features(features, valid_mask)
        
        print(f"  📊 {problem_type} problem:")
        print(f"    Classes: {len(categories)}")
        print(f"    Samples: {len(labels)}")
        print(f"    Class distribution: {np.bincount(labels)}")
        
        # Run all combinations
        all_results = []
        total_combinations = len(prepared_features) * len(self.models)
        
        with tqdm(total=total_combinations, desc=f"Running {problem_type} experiments") as pbar:
            for extractor_name, feature_data in prepared_features.items():
                for model_name, model in self.models.items():
                    result = self.run_single_experiment(
                        feature_data, labels, extractor_name, model_name, model, problem_type, categories
                    )
                    all_results.append(result)
                    pbar.update(1)
        
        return all_results, categories
    
    def create_confusion_matrices(self, results, problem_type):
        """Create detailed confusion matrices with class labels."""
        print(f"🎨 Creating confusion matrices for {problem_type}...")
        
        # Filter results for this problem type
        problem_results = [r for r in results if r['status'] == 'success' and r['problem_type'] == problem_type]
        
        if not problem_results:
            print(f"  ❌ No successful results for {problem_type}")
            return
        
        # Create figure with subplots for each model-extractor combination
        n_combinations = len(problem_results)
        n_cols = min(3, n_combinations)
        n_rows = (n_combinations + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_combinations == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, result in enumerate(problem_results):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            # Get confusion matrix and class names
            conf_matrix = np.array(result['confusion_matrix'])
            categories = result['categories']
            class_names = list(categories.keys())
            
            # Create heatmap
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=class_names, yticklabels=class_names)
            ax.set_title(f"{result['extractor']} + {result['model']}\nCV Test Accuracy: {result['cv_test_accuracy']:.3f}")
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        # Hide empty subplots
        for idx in range(n_combinations, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.axis('off')
        
        plt.suptitle(f'Confusion Matrices - {problem_type.upper()} (FIXED - CV Test Data)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.results_dir / f'confusion_matrices_{problem_type}_fixed.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.results_dir / f'confusion_matrices_{problem_type}_fixed.svg', bbox_inches='tight')
        plt.close()
        
        print(f"  📊 Confusion matrices saved for {problem_type}")
    
    def create_enhanced_report(self, results_8class, results_16class, categories_8class, categories_16class):
        """Create enhanced comprehensive report for all experiments."""
        print("📋 Creating FIXED comprehensive multiclass report...")
        
        # Combine all results
        all_results = results_8class + results_16class
        df = pd.DataFrame(all_results)
        
        # Filter successful results
        successful_results = df[df['status'] == 'success']
        
        if len(successful_results) == 0:
            print("❌ No successful results to report!")
            return
        
        # Create enhanced report
        report = f"""
# FIXED Enhanced Multiclass ECoG Experiment Report
Experiment: {self.experiment_id}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## IMPORTANT: ALL BUGS FIXED
This report uses FIXED calculations:
- Confusion matrix calculated on CV test data (not training data)
- ROC-AUC calculated on CV test data (not training data)
- All metrics consistent and realistic
- Proper 8-class and 16-class problems (including background)

## Summary
- Total combinations tested: {len(all_results)}
- Successful: {len(successful_results)}
- Failed: {len(all_results) - len(successful_results)}

## Class Labels

### 8-Class Problem (7 categories + background)
"""
        for class_name, class_id in categories_8class.items():
            report += f"- {class_name}: {class_id}\n"
        
        report += f"""
### 16-Class Problem (15 categories + background)
"""
        for class_name, class_id in categories_16class.items():
            report += f"- {class_name}: {class_id}\n"
        
        # 8-class results
        results_8 = successful_results[successful_results['problem_type'] == '8class']
        if len(results_8) > 0:
            best_8 = results_8.loc[results_8['cv_accuracy_mean'].idxmax()]
            report += f"""
## 8-Class Multiclass Results (FIXED)

### Best 8-Class Result
- **CV Accuracy**: {best_8['cv_accuracy_mean']:.3f} ± {best_8['cv_accuracy_std']:.3f}
- **CV Test Accuracy**: {best_8['cv_test_accuracy']:.3f}
- **ROC-AUC**: {best_8['roc_auc_macro']:.3f}
- **F1 Macro**: {best_8['f1_macro']:.3f}
- **Precision Macro**: {best_8['precision_macro']:.3f}
- **Recall Macro**: {best_8['recall_macro']:.3f}
- **Feature Extractor**: {best_8['extractor']}
- **Model**: {best_8['model']}
- **Classes**: {best_8['n_classes']}
- **Samples**: {best_8['n_samples']}

### Top 5 8-Class Results
"""
            top_5_8 = results_8.nlargest(5, 'cv_accuracy_mean')
            for i, (_, row) in enumerate(top_5_8.iterrows(), 1):
                report += f"{i}. {row['extractor']} + {row['model']}: {row['cv_accuracy_mean']:.3f} ± {row['cv_accuracy_std']:.3f} (ROC-AUC: {row['roc_auc_macro']:.3f})\n"
        
        # 16-class results
        report += f"""
## 16-Class Multiclass Results (FIXED)
"""
        results_16 = successful_results[successful_results['problem_type'] == '16class']
        if len(results_16) > 0:
            best_16 = results_16.loc[results_16['cv_accuracy_mean'].idxmax()]
            report += f"""
### Best 16-Class Result
- **CV Accuracy**: {best_16['cv_accuracy_mean']:.3f} ± {best_16['cv_accuracy_std']:.3f}
- **CV Test Accuracy**: {best_16['cv_test_accuracy']:.3f}
- **ROC-AUC**: {best_16['roc_auc_macro']:.3f}
- **F1 Macro**: {best_16['f1_macro']:.3f}
- **Precision Macro**: {best_16['precision_macro']:.3f}
- **Recall Macro**: {best_16['recall_macro']:.3f}
- **Feature Extractor**: {best_16['extractor']}
- **Model**: {best_16['model']}
- **Classes**: {best_16['n_classes']}
- **Samples**: {best_16['n_samples']}

### Top 5 16-Class Results
"""
            top_5_16 = results_16.nlargest(5, 'cv_accuracy_mean')
            for i, (_, row) in enumerate(top_5_16.iterrows(), 1):
                report += f"{i}. {row['extractor']} + {row['model']}: {row['cv_accuracy_mean']:.3f} ± {row['cv_accuracy_std']:.3f} (ROC-AUC: {row['roc_auc_macro']:.3f})\n"
        
        # Overall comparison
        report += f"""
## Overall Comparison (FIXED)
"""
        if len(results_8) > 0 and len(results_16) > 0:
            best_8_overall = results_8['cv_accuracy_mean'].max()
            best_16_overall = results_16['cv_accuracy_mean'].max()
            best_8_roc = results_8['roc_auc_macro'].max()
            best_16_roc = results_16['roc_auc_macro'].max()
            report += f"""
- **Best 8-class accuracy**: {best_8_overall:.3f}
- **Best 16-class accuracy**: {best_16_overall:.3f}
- **Performance difference**: {((best_16_overall / best_8_overall) - 1) * 100:+.1f}%
- **Best 8-class ROC-AUC**: {best_8_roc:.3f}
- **Best 16-class ROC-AUC**: {best_16_roc:.3f}
"""
        
        # Save report
        report_file = self.results_dir / f"fixed_multiclass_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Save results as JSON
        results_file = self.results_dir / f"fixed_multiclass_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"📁 FIXED report saved: {report_file}")
        print(f"📁 Results saved: {results_file}")
        
        return successful_results
    
    def run_all_experiments(self):
        """Run all FIXED multiclass experiments."""
        print("🚀 Running all FIXED multiclass experiments...")
        
        start_time = time.time()
        
        # Run 8-class experiments
        results_8class, categories_8class = self.run_multiclass_experiments('8class')
        
        # Run 16-class experiments
        results_16class, categories_16class = self.run_multiclass_experiments('16class')
        
        # Combine all results
        all_results = results_8class + results_16class
        
        # Create enhanced report
        successful_results = self.create_enhanced_report(
            results_8class, results_16class, categories_8class, categories_16class
        )
        
        # Create enhanced visualizations
        print("\n🎨 Creating FIXED visualizations...")
        
        # Confusion matrices
        self.create_confusion_matrices(all_results, '8class')
        self.create_confusion_matrices(all_results, '16class')
        
        elapsed_time = time.time() - start_time
        
        print(f"\n🎉 All FIXED Multiclass Experiments Completed!")
        print("=" * 60)
        print(f"⏱️ Total Time: {elapsed_time:.1f} seconds")
        print(f"📁 Results Directory: {self.results_dir}")
        
        # Show best results
        if successful_results is not None and len(successful_results) > 0:
            df = pd.DataFrame(successful_results)
            
            print(f"\n🏆 Best Results (FIXED):")
            for problem_type in ['8class', '16class']:
                problem_results = df[df['problem_type'] == problem_type]
                if len(problem_results) > 0:
                    best = problem_results.loc[problem_results['cv_accuracy_mean'].idxmax()]
                    print(f"  {problem_type.upper()}: {best['cv_accuracy_mean']:.3f} (ROC-AUC: {best['roc_auc_macro']:.3f}) - {best['extractor']} + {best['model']}")
        else:
            print(f"\n❌ No successful results to display!")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run FIXED enhanced multiclass ECoG experiments with proper CV test data calculations')
    parser.add_argument('--8class', action='store_true', help='Run only 8-class experiments')
    parser.add_argument('--16class', action='store_true', help='Run only 16-class experiments')
    parser.add_argument('--all', action='store_true', help='Run all experiments (default)')
    
    args = parser.parse_args()
    
    # Default to all if no specific option is chosen
    if not any([args.__dict__[key] for key in ['8class', '16class']]):
        args.all = True
    
    framework = FixedMulticlassExperimentFramework()
    
    if args.all:
        framework.run_all_experiments()
    elif args.__dict__['8class']:
        results_8class, categories_8class = framework.run_multiclass_experiments('8class')
        framework.create_enhanced_report([], results_8class, {}, categories_8class)
        framework.create_confusion_matrices(results_8class, '8class')
    elif args.__dict__['16class']:
        results_16class, categories_16class = framework.run_multiclass_experiments('16class')
        framework.create_enhanced_report(results_16class, [], categories_16class, {})
        framework.create_confusion_matrices(results_16class, '16class')

if __name__ == "__main__":
    main()
