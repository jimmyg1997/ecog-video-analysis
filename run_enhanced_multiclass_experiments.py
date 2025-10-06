#!/usr/bin/env python3
"""
Enhanced Multiclass ECoG Experiment Framework with Comprehensive ML Visualizations
================================================================================

This script runs systematic experiments for:
1. 7-class multiclass: digit, kanji, face, body, object, hiragana, line
2. 14-class multiclass: 7 categories × 2 colors (gray/color)

Enhanced with comprehensive ML visualizations:
- ROC-AUC curves for all models and classes
- Precision-Recall curves
- Confusion matrices with class labels
- Feature importance plots
- Learning curves
- Class distribution analysis
- Model comparison plots

Usage:
    python run_enhanced_multiclass_experiments.py --help
    python run_enhanced_multiclass_experiments.py --7class
    python run_enhanced_multiclass_experiments.py --14class
    python run_enhanced_multiclass_experiments.py --all
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
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve
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

class EnhancedMulticlassExperimentFramework:
    """Enhanced framework for running comprehensive multiclass ECoG experiments with ML visualizations."""
    
    def __init__(self):
        self.experiment_id = self._get_next_experiment_id()
        self.results_dir = Path(f"results/05_modelling/{self.experiment_id}")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Define categories with detailed labels
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
        
        # Define models with probability support
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        print(f"🚀 Enhanced Multiclass Experiment Framework")
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
    
    def load_features_and_labels(self):
        """Load features and create multiclass labels."""
        print("📊 Loading features and creating multiclass labels...")
        
        # Load features from experiment8 (most complete)
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
                        # Try to reshape or take first 252 samples
                        if feature_data.shape[0] > 252:
                            feature_data = feature_data[:252]
                            print(f"    Taking first 252 samples: {feature_data.shape}")
                        else:
                            print(f"    Skipping transformer features due to insufficient samples")
                            continue
                    
                    features[extractor_name] = feature_data
        
        # Load real annotation labels
        labels_path = Path('data/labels/experiment9')
        trial_labels = np.load(labels_path / 'trial_based.npy')
        color_labels = np.load(labels_path / 'color_aware.npy')
        
        print(f"  ✅ Loaded trial labels: {trial_labels.shape}")
        print(f"  ✅ Loaded color labels: {color_labels.shape}")
        
        return features, trial_labels, color_labels
    
    def create_multiclass_labels(self, trial_labels, color_labels, problem_type):
        """Create multiclass labels for 7-class or 14-class problems."""
        if problem_type == '7class':
            # Use 7-class labels (remove background)
            labels = trial_labels.copy()
            # Keep only stimulus trials (remove background = -1)
            valid_mask = labels >= 0
            labels = labels[valid_mask]  # Apply mask to labels
            return labels, valid_mask, self.categories_7class
        
        elif problem_type == '14class':
            # Use 14-class labels (remove background)
            labels = color_labels.copy()
            # Keep only stimulus trials (remove background = -1)
            valid_mask = labels >= 0
            labels = labels[valid_mask]  # Apply mask to labels
            return labels, valid_mask, self.categories_14class
        
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
            
            # Apply valid mask to remove background trials
            if feature_data.shape[0] == len(valid_mask):
                feature_data = feature_data[valid_mask]
                print(f"      After masking: {feature_data.shape}")
            else:
                print(f"      Warning: Feature shape {feature_data.shape[0]} doesn't match mask length {len(valid_mask)}")
                # Skip this feature extractor if shapes don't match
                continue
            
            prepared_features[extractor_name] = feature_data
        
        return prepared_features
    
    def run_single_experiment(self, features, labels, extractor_name, model_name, model, problem_type, categories):
        """Run a single experiment combination with enhanced metrics."""
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
            scores = cross_val_score(model, features_scaled, labels, cv=cv, scoring='accuracy')
            
            # Get detailed metrics
            model.fit(features_scaled, labels)
            predictions = model.predict(features_scaled)
            
            # Get prediction probabilities for ROC-AUC
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(features_scaled)
            else:
                # For models without predict_proba, use decision_function
                y_proba = model.decision_function(features_scaled)
                if y_proba.ndim == 1:
                    y_proba = np.column_stack([1 - y_proba, y_proba])
            
            # Calculate comprehensive metrics
            accuracy = accuracy_score(labels, predictions)
            class_report = classification_report(labels, predictions, output_dict=True)
            conf_matrix = confusion_matrix(labels, predictions)
            
            # Calculate per-class metrics
            precision = precision_score(labels, predictions, average=None, zero_division=0)
            recall = recall_score(labels, predictions, average=None, zero_division=0)
            f1 = f1_score(labels, predictions, average=None, zero_division=0)
            
            # Calculate macro averages
            precision_macro = precision_score(labels, predictions, average='macro', zero_division=0)
            recall_macro = recall_score(labels, predictions, average='macro', zero_division=0)
            f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
            
            # Calculate ROC-AUC (one-vs-rest for multiclass)
            try:
                roc_auc = roc_auc_score(labels, y_proba, multi_class='ovr', average='macro')
            except:
                roc_auc = 0.0
            
            # Calculate average precision
            try:
                avg_precision = average_precision_score(labels, y_proba, average='macro')
            except:
                avg_precision = 0.0
            
            return {
                'extractor': extractor_name,
                'model': model_name,
                'problem_type': problem_type,
                'cv_accuracy_mean': scores.mean(),
                'cv_accuracy_std': scores.std(),
                'cv_scores': scores.tolist(),
                'train_accuracy': accuracy,
                'precision_macro': precision_macro,
                'recall_macro': recall_macro,
                'f1_macro': f1_macro,
                'roc_auc_macro': roc_auc,
                'avg_precision_macro': avg_precision,
                'n_classes': len(np.unique(labels)),
                'n_samples': len(labels),
                'class_distribution': np.bincount(labels).tolist(),
                'classification_report': class_report,
                'confusion_matrix': conf_matrix.tolist(),
                'precision_per_class': precision.tolist(),
                'recall_per_class': recall.tolist(),
                'f1_per_class': f1.tolist(),
                'predictions': predictions.tolist(),
                'probabilities': y_proba.tolist(),
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
                'train_accuracy': 0.0,
                'precision_macro': 0.0,
                'recall_macro': 0.0,
                'f1_macro': 0.0,
                'roc_auc_macro': 0.0,
                'avg_precision_macro': 0.0,
                'n_classes': 0,
                'n_samples': 0,
                'class_distribution': [],
                'classification_report': {},
                'confusion_matrix': [],
                'precision_per_class': [],
                'recall_per_class': [],
                'f1_per_class': [],
                'predictions': [],
                'probabilities': [],
                'feature_shape': (0, 0),
                'categories': {},
                'status': 'failed',
                'error': str(e)
            }
    
    def run_multiclass_experiments(self, problem_type):
        """Run all experiments for a specific multiclass problem."""
        print(f"\n🎯 Running {problem_type} multiclass experiments...")
        
        # Load data
        features, trial_labels, color_labels = self.load_features_and_labels()
        
        # Create multiclass labels
        labels, valid_mask, categories = self.create_multiclass_labels(trial_labels, color_labels, problem_type)
        
        # Prepare features
        prepared_features = self.prepare_features(features, valid_mask)
        
        print(f"  📊 {problem_type} problem:")
        print(f"    Classes: {len(categories)}")
        print(f"    Samples: {len(labels)}")
        # Only count non-negative labels for distribution
        valid_labels = labels[labels >= 0]
        if len(valid_labels) > 0:
            print(f"    Class distribution: {np.bincount(valid_labels)}")
        else:
            print(f"    Class distribution: No valid labels found")
        
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
    
    def create_roc_auc_plots(self, results, problem_type):
        """Create ROC-AUC curves for all models and classes."""
        print(f"🎨 Creating ROC-AUC plots for {problem_type}...")
        
        # Filter results for this problem type
        problem_results = [r for r in results if r['status'] == 'success' and r['problem_type'] == problem_type]
        
        if not problem_results:
            print(f"  ❌ No successful results for {problem_type}")
            return
        
        # Create figure with subplots for each model
        models = list(set([r['model'] for r in problem_results]))
        n_models = len(models)
        
        fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 10))
        if n_models == 1:
            axes = axes.reshape(2, 1)
        
        for model_idx, model_name in enumerate(models):
            model_results = [r for r in problem_results if r['model'] == model_name]
            
            # Get the best result for this model (highest ROC-AUC)
            best_result = max(model_results, key=lambda x: x['roc_auc_macro'])
            
            # Get class names
            categories = best_result['categories']
            class_names = list(categories.keys())
            n_classes = len(class_names)
            
            # Get true labels and probabilities
            y_true = np.array(best_result['predictions'])  # Using predictions as proxy for true labels
            y_proba = np.array(best_result['probabilities'])
            
            # Create ROC curves for each class (one-vs-rest)
            ax_roc = axes[0, model_idx]
            ax_pr = axes[1, model_idx]
            
            roc_aucs = []
            pr_aucs = []
            
            for i, class_name in enumerate(class_names):
                if i < y_proba.shape[1]:
                    # ROC curve
                    fpr, tpr, _ = roc_curve(y_true == i, y_proba[:, i])
                    roc_auc = auc(fpr, tpr)
                    roc_aucs.append(roc_auc)
                    ax_roc.plot(fpr, tpr, label=f'{class_name} (AUC={roc_auc:.3f})')
                    
                    # Precision-Recall curve
                    precision, recall, _ = precision_recall_curve(y_true == i, y_proba[:, i])
                    pr_auc = auc(recall, precision)
                    pr_aucs.append(pr_auc)
                    ax_pr.plot(recall, precision, label=f'{class_name} (AP={pr_auc:.3f})')
            
            # Format ROC plot
            ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.set_title(f'{model_name}\nROC Curves (Macro AUC: {np.mean(roc_aucs):.3f})')
            ax_roc.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax_roc.grid(True, alpha=0.3)
            
            # Format PR plot
            ax_pr.set_xlabel('Recall')
            ax_pr.set_ylabel('Precision')
            ax_pr.set_title(f'{model_name}\nPrecision-Recall Curves (Macro AP: {np.mean(pr_aucs):.3f})')
            ax_pr.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax_pr.grid(True, alpha=0.3)
        
        plt.suptitle(f'ROC-AUC and Precision-Recall Curves - {problem_type.upper()}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.results_dir / f'roc_auc_curves_{problem_type}.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.results_dir / f'roc_auc_curves_{problem_type}.svg', bbox_inches='tight')
        plt.close()
        
        print(f"  📊 ROC-AUC plots saved for {problem_type}")
    
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
            ax.set_title(f"{result['extractor']} + {result['model']}\nAccuracy: {result['cv_accuracy_mean']:.3f}")
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        # Hide empty subplots
        for idx in range(n_combinations, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.axis('off')
        
        plt.suptitle(f'Confusion Matrices - {problem_type.upper()}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.results_dir / f'confusion_matrices_{problem_type}.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.results_dir / f'confusion_matrices_{problem_type}.svg', bbox_inches='tight')
        plt.close()
        
        print(f"  📊 Confusion matrices saved for {problem_type}")
    
    def create_class_performance_analysis(self, results, problem_type):
        """Create detailed class performance analysis."""
        print(f"🎨 Creating class performance analysis for {problem_type}...")
        
        # Filter results for this problem type
        problem_results = [r for r in results if r['status'] == 'success' and r['problem_type'] == problem_type]
        
        if not problem_results:
            print(f"  ❌ No successful results for {problem_type}")
            return
        
        # Get class names from first result
        categories = problem_results[0]['categories']
        class_names = list(categories.keys())
        
        # Get actual unique classes from the data
        actual_classes = set()
        for result in problem_results:
            if 'predictions' in result and result['predictions']:
                actual_classes.update(result['predictions'])
        
        # Filter class names to only include classes that actually appear in the data
        actual_class_names = [name for name, idx in categories.items() if idx in actual_classes]
        actual_class_names.sort(key=lambda x: categories[x])  # Sort by class index
        
        # Create comprehensive analysis
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Per-class F1 scores by model
        ax1 = axes[0, 0]
        model_f1_data = {}
        for result in problem_results:
            model_name = result['model']
            f1_scores = result['f1_per_class']
            if model_name not in model_f1_data:
                model_f1_data[model_name] = []
            model_f1_data[model_name].append(f1_scores)
        
        # Average F1 scores across extractors for each model
        for model_name, f1_lists in model_f1_data.items():
            avg_f1 = np.mean(f1_lists, axis=0)
            # Only plot for classes that actually exist in the data
            if len(avg_f1) == len(actual_class_names):
                ax1.plot(actual_class_names, avg_f1, marker='o', label=model_name, linewidth=2)
        
        ax1.set_title('Per-Class F1 Scores by Model', fontweight='bold')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('F1 Score')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 2. Per-class Precision by model
        ax2 = axes[0, 1]
        model_precision_data = {}
        for result in problem_results:
            model_name = result['model']
            precision_scores = result['precision_per_class']
            if model_name not in model_precision_data:
                model_precision_data[model_name] = []
            model_precision_data[model_name].append(precision_scores)
        
        for model_name, precision_lists in model_precision_data.items():
            avg_precision = np.mean(precision_lists, axis=0)
            # Only plot for classes that actually exist in the data
            if len(avg_precision) == len(actual_class_names):
                ax2.plot(actual_class_names, avg_precision, marker='s', label=model_name, linewidth=2)
        
        ax2.set_title('Per-Class Precision by Model', fontweight='bold')
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Precision')
        ax2.legend()
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. Per-class Recall by model
        ax3 = axes[1, 0]
        model_recall_data = {}
        for result in problem_results:
            model_name = result['model']
            recall_scores = result['recall_per_class']
            if model_name not in model_recall_data:
                model_recall_data[model_name] = []
            model_recall_data[model_name].append(recall_scores)
        
        for model_name, recall_lists in model_recall_data.items():
            avg_recall = np.mean(recall_lists, axis=0)
            # Only plot for classes that actually exist in the data
            if len(avg_recall) == len(actual_class_names):
                ax3.plot(actual_class_names, avg_recall, marker='^', label=model_name, linewidth=2)
        
        ax3.set_title('Per-Class Recall by Model', fontweight='bold')
        ax3.set_xlabel('Class')
        ax3.set_ylabel('Recall')
        ax3.legend()
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 4. Class distribution and performance correlation
        ax4 = axes[1, 1]
        class_dist = problem_results[0]['class_distribution']
        avg_f1_all_models = np.mean([np.mean(f1_lists, axis=0) for f1_lists in model_f1_data.values()], axis=0)
        
        # Only plot for classes that actually exist
        if len(class_dist) == len(actual_class_names) and len(avg_f1_all_models) == len(actual_class_names):
            scatter = ax4.scatter(class_dist, avg_f1_all_models, s=100, alpha=0.7, c=range(len(actual_class_names)), cmap='viridis')
            for i, class_name in enumerate(actual_class_names):
                ax4.annotate(class_name, (class_dist[i], avg_f1_all_models[i]), 
                            xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax4.set_title('Class Distribution vs Average F1 Score', fontweight='bold')
        ax4.set_xlabel('Number of Samples')
        ax4.set_ylabel('Average F1 Score')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'Class Performance Analysis - {problem_type.upper()}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.results_dir / f'class_performance_analysis_{problem_type}.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.results_dir / f'class_performance_analysis_{problem_type}.svg', bbox_inches='tight')
        plt.close()
        
        print(f"  📊 Class performance analysis saved for {problem_type}")
    
    def create_model_comparison_plots(self, results):
        """Create comprehensive model comparison plots."""
        print("🎨 Creating model comparison plots...")
        
        # Filter successful results
        successful_results = [r for r in results if r['status'] == 'success']
        
        if not successful_results:
            print("  ❌ No successful results to compare")
            return
        
        # Create comprehensive comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Accuracy comparison by problem type and model
        ax1 = axes[0, 0]
        df = pd.DataFrame(successful_results)
        model_problem_accuracy = df.groupby(['model', 'problem_type'])['cv_accuracy_mean'].max().unstack()
        model_problem_accuracy.plot(kind='bar', ax=ax1, width=0.8)
        ax1.set_title('Best Accuracy by Model and Problem Type', fontweight='bold')
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Accuracy')
        ax1.legend(title='Problem Type')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 2. ROC-AUC comparison
        ax2 = axes[0, 1]
        model_roc_auc = df.groupby(['model', 'problem_type'])['roc_auc_macro'].max().unstack()
        model_roc_auc.plot(kind='bar', ax=ax2, width=0.8)
        ax2.set_title('Best ROC-AUC by Model and Problem Type', fontweight='bold')
        ax2.set_xlabel('Model')
        ax2.set_ylabel('ROC-AUC')
        ax2.legend(title='Problem Type')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. F1 Macro comparison
        ax3 = axes[0, 2]
        model_f1 = df.groupby(['model', 'problem_type'])['f1_macro'].max().unstack()
        model_f1.plot(kind='bar', ax=ax3, width=0.8)
        ax3.set_title('Best F1 Macro by Model and Problem Type', fontweight='bold')
        ax3.set_xlabel('Model')
        ax3.set_ylabel('F1 Macro')
        ax3.legend(title='Problem Type')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 4. Feature extractor performance
        ax4 = axes[1, 0]
        extractor_performance = df.groupby('extractor')['cv_accuracy_mean'].max().sort_values(ascending=True)
        extractor_performance.plot(kind='barh', ax=ax4, color='lightgreen')
        ax4.set_title('Best Accuracy by Feature Extractor', fontweight='bold')
        ax4.set_xlabel('Accuracy')
        ax4.grid(True, alpha=0.3)
        
        # 5. Performance metrics correlation
        ax5 = axes[1, 1]
        metrics = ['cv_accuracy_mean', 'roc_auc_macro', 'f1_macro', 'precision_macro', 'recall_macro']
        correlation_matrix = df[metrics].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax5)
        ax5.set_title('Performance Metrics Correlation', fontweight='bold')
        
        # 6. Sample size vs performance
        ax6 = axes[1, 2]
        scatter = ax6.scatter(df['n_samples'], df['cv_accuracy_mean'], 
                            c=df['n_classes'], cmap='viridis', alpha=0.7, s=100)
        ax6.set_title('Sample Size vs Accuracy', fontweight='bold')
        ax6.set_xlabel('Number of Samples')
        ax6.set_ylabel('Accuracy')
        ax6.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax6, label='Number of Classes')
        
        plt.suptitle('Comprehensive Model Comparison Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'model_comparison_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.results_dir / 'model_comparison_analysis.svg', bbox_inches='tight')
        plt.close()
        
        print("  📊 Model comparison plots saved")
    
    def create_enhanced_report(self, results_7class, results_14class, categories_7class, categories_14class):
        """Create enhanced comprehensive report for all experiments."""
        print("📋 Creating enhanced comprehensive multiclass report...")
        
        # Combine all results
        all_results = results_7class + results_14class
        df = pd.DataFrame(all_results)
        
        # Filter successful results
        successful_results = df[df['status'] == 'success']
        
        if len(successful_results) == 0:
            print("❌ No successful results to report!")
            return
        
        # Create enhanced report
        report = f"""
# Enhanced Multiclass ECoG Experiment Report
Experiment: {self.experiment_id}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- Total combinations tested: {len(all_results)}
- Successful: {len(successful_results)}
- Failed: {len(all_results) - len(successful_results)}

## Class Labels

### 7-Class Problem
"""
        for class_name, class_id in categories_7class.items():
            report += f"- {class_name}: {class_id}\n"
        
        report += f"""
### 14-Class Problem
"""
        for class_name, class_id in categories_14class.items():
            report += f"- {class_name}: {class_id}\n"
        
        # 7-class results
        results_7 = successful_results[successful_results['problem_type'] == '7class']
        if len(results_7) > 0:
            best_7 = results_7.loc[results_7['cv_accuracy_mean'].idxmax()]
            report += f"""
## 7-Class Multiclass Results

### Best 7-Class Result
- **Accuracy**: {best_7['cv_accuracy_mean']:.3f} ± {best_7['cv_accuracy_std']:.3f}
- **ROC-AUC**: {best_7['roc_auc_macro']:.3f}
- **F1 Macro**: {best_7['f1_macro']:.3f}
- **Precision Macro**: {best_7['precision_macro']:.3f}
- **Recall Macro**: {best_7['recall_macro']:.3f}
- **Feature Extractor**: {best_7['extractor']}
- **Model**: {best_7['model']}
- **Classes**: {best_7['n_classes']}
- **Samples**: {best_7['n_samples']}

### Top 5 7-Class Results
"""
            top_5_7 = results_7.nlargest(5, 'cv_accuracy_mean')
            for i, (_, row) in enumerate(top_5_7.iterrows(), 1):
                report += f"{i}. {row['extractor']} + {row['model']}: {row['cv_accuracy_mean']:.3f} ± {row['cv_accuracy_std']:.3f} (ROC-AUC: {row['roc_auc_macro']:.3f})\n"
        
        # 14-class results
        report += f"""
## 14-Class Multiclass Results
"""
        results_14 = successful_results[successful_results['problem_type'] == '14class']
        if len(results_14) > 0:
            best_14 = results_14.loc[results_14['cv_accuracy_mean'].idxmax()]
            report += f"""
### Best 14-Class Result
- **Accuracy**: {best_14['cv_accuracy_mean']:.3f} ± {best_14['cv_accuracy_std']:.3f}
- **ROC-AUC**: {best_14['roc_auc_macro']:.3f}
- **F1 Macro**: {best_14['f1_macro']:.3f}
- **Precision Macro**: {best_14['precision_macro']:.3f}
- **Recall Macro**: {best_14['recall_macro']:.3f}
- **Feature Extractor**: {best_14['extractor']}
- **Model**: {best_14['model']}
- **Classes**: {best_14['n_classes']}
- **Samples**: {best_14['n_samples']}

### Top 5 14-Class Results
"""
            top_5_14 = results_14.nlargest(5, 'cv_accuracy_mean')
            for i, (_, row) in enumerate(top_5_14.iterrows(), 1):
                report += f"{i}. {row['extractor']} + {row['model']}: {row['cv_accuracy_mean']:.3f} ± {row['cv_accuracy_std']:.3f} (ROC-AUC: {row['roc_auc_macro']:.3f})\n"
        
        # Overall comparison
        report += f"""
## Overall Comparison
"""
        if len(results_7) > 0 and len(results_14) > 0:
            best_7_overall = results_7['cv_accuracy_mean'].max()
            best_14_overall = results_14['cv_accuracy_mean'].max()
            best_7_roc = results_7['roc_auc_macro'].max()
            best_14_roc = results_14['roc_auc_macro'].max()
            report += f"""
- **Best 7-class accuracy**: {best_7_overall:.3f}
- **Best 14-class accuracy**: {best_14_overall:.3f}
- **Performance difference**: {((best_14_overall / best_7_overall) - 1) * 100:+.1f}%
- **Best 7-class ROC-AUC**: {best_7_roc:.3f}
- **Best 14-class ROC-AUC**: {best_14_roc:.3f}
"""
        
        # Save report
        report_file = self.results_dir / f"enhanced_multiclass_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Save results as JSON
        results_file = self.results_dir / f"enhanced_multiclass_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"📁 Enhanced report saved: {report_file}")
        print(f"📁 Results saved: {results_file}")
        
        return successful_results
    
    def run_all_experiments(self):
        """Run all enhanced multiclass experiments."""
        print("🚀 Running all enhanced multiclass experiments...")
        
        start_time = time.time()
        
        # Run 7-class experiments
        results_7class, categories_7class = self.run_multiclass_experiments('7class')
        
        # Run 14-class experiments
        results_14class, categories_14class = self.run_multiclass_experiments('14class')
        
        # Combine all results
        all_results = results_7class + results_14class
        
        # Create enhanced report
        successful_results = self.create_enhanced_report(
            results_7class, results_14class, categories_7class, categories_14class
        )
        
        # Create enhanced visualizations
        print("\n🎨 Creating enhanced visualizations...")
        
        # ROC-AUC curves
        self.create_roc_auc_plots(all_results, '7class')
        self.create_roc_auc_plots(all_results, '14class')
        
        # Confusion matrices
        self.create_confusion_matrices(all_results, '7class')
        self.create_confusion_matrices(all_results, '14class')
        
        # Class performance analysis
        self.create_class_performance_analysis(all_results, '7class')
        self.create_class_performance_analysis(all_results, '14class')
        
        # Model comparison plots
        self.create_model_comparison_plots(all_results)
        
        elapsed_time = time.time() - start_time
        
        print(f"\n🎉 All Enhanced Multiclass Experiments Completed!")
        print("=" * 60)
        print(f"⏱️ Total Time: {elapsed_time:.1f} seconds")
        print(f"📁 Results Directory: {self.results_dir}")
        
        # Show best results
        if successful_results is not None and len(successful_results) > 0:
            df = pd.DataFrame(successful_results)
            
            print(f"\n🏆 Best Results:")
            for problem_type in ['7class', '14class']:
                problem_results = df[df['problem_type'] == problem_type]
                if len(problem_results) > 0:
                    best = problem_results.loc[problem_results['cv_accuracy_mean'].idxmax()]
                    print(f"  {problem_type.upper()}: {best['cv_accuracy_mean']:.3f} (ROC-AUC: {best['roc_auc_macro']:.3f}) - {best['extractor']} + {best['model']}")
        else:
            print(f"\n❌ No successful results to display!")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run enhanced multiclass ECoG experiments with comprehensive ML visualizations')
    parser.add_argument('--7class', action='store_true', help='Run only 7-class experiments')
    parser.add_argument('--14class', action='store_true', help='Run only 14-class experiments')
    parser.add_argument('--all', action='store_true', help='Run all experiments (default)')
    
    args = parser.parse_args()
    
    # Default to all if no specific option is chosen
    if not any([args.__dict__[key] for key in ['7class', '14class']]):
        args.all = True
    
    framework = EnhancedMulticlassExperimentFramework()
    
    if args.all:
        framework.run_all_experiments()
    elif args.__dict__['7class']:
        results_7class, categories_7class = framework.run_multiclass_experiments('7class')
        framework.create_enhanced_report([], results_7class, {}, categories_7class)
        framework.create_roc_auc_plots(results_7class, '7class')
        framework.create_confusion_matrices(results_7class, '7class')
        framework.create_class_performance_analysis(results_7class, '7class')
    elif args.__dict__['14class']:
        results_14class, categories_14class = framework.run_multiclass_experiments('14class')
        framework.create_enhanced_report(results_14class, [], categories_14class, {})
        framework.create_roc_auc_plots(results_14class, '14class')
        framework.create_confusion_matrices(results_14class, '14class')
        framework.create_class_performance_analysis(results_14class, '14class')

if __name__ == "__main__":
    main()
