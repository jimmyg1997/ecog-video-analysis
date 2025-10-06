#!/usr/bin/env python3
"""
Batch Multiclass Experiment Runner
================================

This script runs batches of experiments for systematic testing.
Allows running specific combinations of feature extractors, models, and problem types.

Usage:
    python run_batch_multiclass_experiments.py --extractors comprehensive,eegnet --models "Random Forest,SVM" --problems 7class,14class
    python run_batch_multiclass_experiments.py --all-extractors --all-models --problems 7class
    python run_batch_multiclass_experiments.py --quick-test
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

class BatchMulticlassExperiment:
    """Batch multiclass experiment runner."""
    
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
        
        print(f"🚀 Batch Multiclass Experiment")
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
    
    def load_features_and_labels(self, extractors, problem_types):
        """Load features and labels for specified extractors and problem types."""
        print("📊 Loading features and labels...")
        
        features = {}
        labels = {}
        
        # Load features
        features_path = Path('data/features/experiment8')
        for extractor in extractors:
            extractor_path = features_path / extractor
            if extractor_path.exists():
                feature_files = list(extractor_path.glob('*.npy'))
                if feature_files:
                    features[extractor] = np.load(feature_files[0])
                    print(f"  ✅ Loaded {extractor}: {features[extractor].shape}")
                else:
                    print(f"  ❌ No feature files found for {extractor}")
            else:
                print(f"  ❌ Feature directory not found: {extractor_path}")
        
        # Load labels
        labels_path = Path('data/labels/experiment9')
        for problem_type in problem_types:
            if problem_type == '7class':
                labels[problem_type] = np.load(labels_path / 'trial_based.npy')
            elif problem_type == '14class':
                labels[problem_type] = np.load(labels_path / 'color_aware.npy')
            print(f"  ✅ Loaded {problem_type} labels: {labels[problem_type].shape}")
        
        return features, labels
    
    def prepare_data(self, features, labels, problem_type):
        """Prepare data for a specific problem type."""
        # Get labels and categories
        if problem_type == '7class':
            problem_labels = labels.copy()
            categories = self.categories_7class
        else:
            problem_labels = labels.copy()
            categories = self.categories_14class
        
        # Remove background trials
        valid_mask = problem_labels >= 0
        problem_labels = problem_labels[valid_mask]
        
        # Prepare features
        prepared_features = {}
        for extractor_name, feature_data in features.items():
            # Ensure features are 2D
            if feature_data.ndim == 1:
                feature_data = feature_data.reshape(-1, 1)
            elif feature_data.ndim == 3:
                # For 3D features (like EEGNet), flatten
                feature_data = feature_data.reshape(feature_data.shape[0], -1)
            
            # Apply valid mask to remove background trials
            if feature_data.shape[0] == len(valid_mask):
                feature_data = feature_data[valid_mask]
            
            prepared_features[extractor_name] = feature_data
        
        return prepared_features, problem_labels, categories
    
    def run_single_experiment(self, features, labels, extractor_name, model_name, model, problem_type):
        """Run a single experiment combination."""
        try:
            # Scale features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Use stratified cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, features_scaled, labels, cv=cv, scoring='accuracy')
            
            # Get detailed metrics
            model.fit(features_scaled, labels)
            predictions = model.predict(features_scaled)
            
            # Calculate additional metrics
            accuracy = accuracy_score(labels, predictions)
            class_report = classification_report(labels, predictions, output_dict=True)
            conf_matrix = confusion_matrix(labels, predictions)
            
            return {
                'experiment_id': self.experiment_id,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'extractor': extractor_name,
                'model': model_name,
                'problem_type': problem_type,
                'cv_accuracy_mean': scores.mean(),
                'cv_accuracy_std': scores.std(),
                'cv_scores': scores.tolist(),
                'train_accuracy': accuracy,
                'n_classes': len(np.unique(labels)),
                'n_samples': len(labels),
                'class_distribution': np.bincount(labels).tolist(),
                'classification_report': class_report,
                'confusion_matrix': conf_matrix.tolist(),
                'feature_shape': features.shape,
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'experiment_id': self.experiment_id,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'extractor': extractor_name,
                'model': model_name,
                'problem_type': problem_type,
                'cv_accuracy_mean': 0.0,
                'cv_accuracy_std': 0.0,
                'cv_scores': [],
                'train_accuracy': 0.0,
                'n_classes': 0,
                'n_samples': 0,
                'class_distribution': [],
                'classification_report': {},
                'confusion_matrix': [],
                'feature_shape': (0, 0),
                'status': 'failed',
                'error': str(e)
            }
    
    def run_batch_experiments(self, extractors, models, problem_types):
        """Run batch experiments."""
        print(f"\n🎯 Running Batch Experiments:")
        print(f"  Extractors: {extractors}")
        print(f"  Models: {models}")
        print(f"  Problem Types: {problem_types}")
        print("=" * 50)
        
        # Load data
        features, labels = self.load_features_and_labels(extractors, problem_types)
        
        if not features:
            print("❌ No features loaded!")
            return []
        
        if not labels:
            print("❌ No labels loaded!")
            return []
        
        # Run experiments
        all_results = []
        total_combinations = len(extractors) * len(models) * len(problem_types)
        
        with tqdm(total=total_combinations, desc="Running batch experiments") as pbar:
            for problem_type in problem_types:
                if problem_type not in labels:
                    print(f"  ❌ Labels not found for {problem_type}")
                    continue
                
                # Prepare data for this problem type
                prepared_features, problem_labels, categories = self.prepare_data(
                    features, labels[problem_type], problem_type
                )
                
                print(f"  📊 {problem_type} problem:")
                print(f"    Classes: {len(categories)}")
                print(f"    Samples: {len(problem_labels)}")
                print(f"    Class distribution: {np.bincount(problem_labels)}")
                
                for extractor_name in extractors:
                    if extractor_name not in prepared_features:
                        print(f"    ❌ Features not found for {extractor_name}")
                        continue
                    
                    for model_name in models:
                        if model_name not in self.models:
                            print(f"    ❌ Model not found: {model_name}")
                            continue
                        
                        model = self.models[model_name]
                        result = self.run_single_experiment(
                            prepared_features[extractor_name], problem_labels, 
                            extractor_name, model_name, model, problem_type
                        )
                        all_results.append(result)
                        pbar.update(1)
        
        return all_results
    
    def create_batch_report(self, results):
        """Create comprehensive batch report."""
        print("📋 Creating batch experiment report...")
        
        if not results:
            print("❌ No results to report!")
            return
        
        df = pd.DataFrame(results)
        successful_results = df[df['status'] == 'success']
        
        if len(successful_results) == 0:
            print("❌ No successful results to report!")
            return
        
        # Create report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        report = f"""
# Batch Multiclass ECoG Experiment Report
Experiment: {self.experiment_id}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- Total combinations tested: {len(results)}
- Successful: {len(successful_results)}
- Failed: {len(results) - len(successful_results)}

## Results by Problem Type
"""
        
        # Results by problem type
        for problem_type in ['7class', '14class']:
            problem_results = successful_results[successful_results['problem_type'] == problem_type]
            if len(problem_results) > 0:
                best = problem_results.loc[problem_results['cv_accuracy_mean'].idxmax()]
                report += f"""
### {problem_type.upper()} Results
- **Best Accuracy**: {best['cv_accuracy_mean']:.3f} ± {best['cv_accuracy_std']:.3f}
- **Best Combination**: {best['extractor']} + {best['model']}
- **Classes**: {best['n_classes']}
- **Samples**: {best['n_samples']}

#### Top 5 Results:
"""
                top_5 = problem_results.nlargest(5, 'cv_accuracy_mean')
                for i, (_, row) in enumerate(top_5.iterrows(), 1):
                    report += f"{i}. {row['extractor']} + {row['model']}: {row['cv_accuracy_mean']:.3f} ± {row['cv_accuracy_std']:.3f}\n"
        
        # Overall best results
        report += f"""
## Overall Best Results
"""
        overall_best = successful_results.loc[successful_results['cv_accuracy_mean'].idxmax()]
        report += f"""
- **Best Overall**: {overall_best['cv_accuracy_mean']:.3f} ± {overall_best['cv_accuracy_std']:.3f}
- **Combination**: {overall_best['extractor']} + {overall_best['model']} ({overall_best['problem_type']})
- **Classes**: {overall_best['n_classes']}
- **Samples**: {overall_best['n_samples']}
"""
        
        # Save report
        report_file = self.results_dir / f"batch_experiment_report_{timestamp}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Save results as JSON
        results_file = self.results_dir / f"batch_experiment_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"📁 Report saved: {report_file}")
        print(f"📁 Results saved: {results_file}")
        
        return successful_results
    
    def create_batch_visualizations(self, successful_results):
        """Create batch experiment visualizations."""
        print("🎨 Creating batch experiment visualizations...")
        
        if successful_results is None or len(successful_results) == 0:
            print("❌ No successful results to visualize!")
            return
        
        df = pd.DataFrame(successful_results)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Accuracy by problem type
        plt.subplot(3, 3, 1)
        problem_accuracy = df.groupby('problem_type')['cv_accuracy_mean'].max()
        bars = plt.bar(problem_accuracy.index, problem_accuracy.values, color=['skyblue', 'lightcoral'])
        plt.title('Best Accuracy by Problem Type', fontweight='bold', fontsize=14)
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        for i, v in enumerate(problem_accuracy.values):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
        
        # 2. Accuracy by feature extractor
        plt.subplot(3, 3, 2)
        feature_accuracy = df.groupby('extractor')['cv_accuracy_mean'].max().sort_values(ascending=True)
        feature_accuracy.plot(kind='barh', color='lightgreen')
        plt.title('Best Accuracy by Feature Extractor', fontweight='bold', fontsize=14)
        plt.xlabel('Accuracy')
        
        # 3. Accuracy by model
        plt.subplot(3, 3, 3)
        model_accuracy = df.groupby('model')['cv_accuracy_mean'].max().sort_values(ascending=True)
        model_accuracy.plot(kind='barh', color='gold')
        plt.title('Best Accuracy by Model', fontweight='bold', fontsize=14)
        plt.xlabel('Accuracy')
        
        # 4. 7-class results heatmap
        plt.subplot(3, 3, 4)
        results_7 = df[df['problem_type'] == '7class']
        if len(results_7) > 0:
            pivot_7 = results_7.pivot_table(values='cv_accuracy_mean', index='extractor', columns='model', aggfunc='max')
            sns.heatmap(pivot_7, annot=True, fmt='.3f', cmap='Blues', ax=plt.gca())
            plt.title('7-Class Results Heatmap', fontweight='bold', fontsize=14)
        
        # 5. 14-class results heatmap
        plt.subplot(3, 3, 5)
        results_14 = df[df['problem_type'] == '14class']
        if len(results_14) > 0:
            pivot_14 = results_14.pivot_table(values='cv_accuracy_mean', index='extractor', columns='model', aggfunc='max')
            sns.heatmap(pivot_14, annot=True, fmt='.3f', cmap='Reds', ax=plt.gca())
            plt.title('14-Class Results Heatmap', fontweight='bold', fontsize=14)
        
        # 6. Accuracy distribution
        plt.subplot(3, 3, 6)
        plt.hist(df['cv_accuracy_mean'], bins=20, alpha=0.7, color='purple', edgecolor='black')
        plt.title('Accuracy Distribution', fontweight='bold', fontsize=14)
        plt.xlabel('Accuracy')
        plt.ylabel('Frequency')
        plt.axvline(df['cv_accuracy_mean'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df["cv_accuracy_mean"].mean():.3f}')
        plt.legend()
        
        # 7. Sample size vs accuracy
        plt.subplot(3, 3, 7)
        plt.scatter(df['n_samples'], df['cv_accuracy_mean'], alpha=0.7, c=df['n_classes'], cmap='viridis')
        plt.colorbar(label='Number of Classes')
        plt.title('Sample Size vs Accuracy', fontweight='bold', fontsize=14)
        plt.xlabel('Number of Samples')
        plt.ylabel('Accuracy')
        
        # 8. Feature dimension vs accuracy
        plt.subplot(3, 3, 8)
        feature_dims = [eval(str(shape))[1] if isinstance(shape, str) else shape[1] for shape in df['feature_shape']]
        plt.scatter(feature_dims, df['cv_accuracy_mean'], alpha=0.7, color='orange')
        plt.title('Feature Dimension vs Accuracy', fontweight='bold', fontsize=14)
        plt.xlabel('Feature Dimension')
        plt.ylabel('Accuracy')
        
        # 9. Best results summary
        plt.subplot(3, 3, 9)
        plt.axis('off')
        best_overall = df.loc[df['cv_accuracy_mean'].idxmax()]
        
        summary_text = f"""
🏆 BATCH EXPERIMENT SUMMARY

Total Experiments: {len(df)}
Successful: {len(df)}
Failed: 0

BEST RESULT:
{best_overall['extractor']} + {best_overall['model']}
({best_overall['problem_type']})
Accuracy: {best_overall['cv_accuracy_mean']:.3f} ± {best_overall['cv_accuracy_std']:.3f}

7-Class Best: {df[df['problem_type'] == '7class']['cv_accuracy_mean'].max():.3f}
14-Class Best: {df[df['problem_type'] == '14class']['cv_accuracy_mean'].max():.3f}

Experiment ID: {self.experiment_id}
        """
        plt.text(0.1, 0.5, summary_text, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
        
        plt.suptitle('Batch Multiclass ECoG Experiment Analysis', fontsize=18, fontweight='bold')
        plt.tight_layout()
        
        # Save plots
        plt.savefig(self.results_dir / f'batch_experiment_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.results_dir / f'batch_experiment_analysis_{timestamp}.svg', bbox_inches='tight')
        plt.close()
        
        print("  📊 Batch analysis plot saved")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run batch multiclass ECoG experiments')
    parser.add_argument('--extractors', type=str, help='Comma-separated list of extractors')
    parser.add_argument('--models', type=str, help='Comma-separated list of models')
    parser.add_argument('--problems', type=str, help='Comma-separated list of problem types')
    parser.add_argument('--all-extractors', action='store_true', help='Use all available extractors')
    parser.add_argument('--all-models', action='store_true', help='Use all available models')
    parser.add_argument('--quick-test', action='store_true', help='Quick test with 2 extractors, 2 models, 7class only')
    
    args = parser.parse_args()
    
    experiment = BatchMulticlassExperiment()
    
    # Determine extractors
    if args.all_extractors:
        extractors = experiment.available_extractors
    elif args.extractors:
        extractors = [e.strip() for e in args.extractors.split(',')]
    elif args.quick_test:
        extractors = ['comprehensive', 'eegnet']
    else:
        print("❌ Error: Please specify extractors or use --all-extractors")
        return
    
    # Determine models
    if args.all_models:
        models = list(experiment.models.keys())
    elif args.models:
        models = [m.strip() for m in args.models.split(',')]
    elif args.quick_test:
        models = ['Random Forest', 'SVM']
    else:
        print("❌ Error: Please specify models or use --all-models")
        return
    
    # Determine problem types
    if args.problems:
        problem_types = [p.strip() for p in args.problems.split(',')]
    elif args.quick_test:
        problem_types = ['7class']
    else:
        problem_types = ['7class', '14class']
    
    print(f"🎯 Batch Configuration:")
    print(f"  Extractors: {extractors}")
    print(f"  Models: {models}")
    print(f"  Problem Types: {problem_types}")
    print(f"  Total Combinations: {len(extractors) * len(models) * len(problem_types)}")
    
    # Run batch experiments
    start_time = time.time()
    results = experiment.run_batch_experiments(extractors, models, problem_types)
    elapsed_time = time.time() - start_time
    
    if results:
        # Create report and visualizations
        successful_results = experiment.create_batch_report(results)
        experiment.create_batch_visualizations(successful_results)
        
        print(f"\n🎉 Batch Experiments Completed!")
        print("=" * 50)
        print(f"⏱️ Total Time: {elapsed_time:.1f} seconds")
        print(f"📁 Results Directory: {experiment.results_dir}")
        
        # Show best results
        if successful_results is not None and len(successful_results) > 0:
            df = pd.DataFrame(successful_results)
            best = df.loc[df['cv_accuracy_mean'].idxmax()]
            print(f"🏆 Best Result: {best['cv_accuracy_mean']:.3f} ({best['extractor']} + {best['model']} + {best['problem_type']})")
    else:
        print("❌ No results generated!")

if __name__ == "__main__":
    main()
