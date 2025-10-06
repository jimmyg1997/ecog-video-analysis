#!/usr/bin/env python3
"""
Comprehensive Multiclass ECoG Experiment Framework
================================================

This script runs systematic experiments for:
1. 7-class multiclass: digit, kanji, face, body, object, hiragana, line
2. 14-class multiclass: 7 categories × 2 colors (gray/color)

Tests all combinations of:
- Feature Extractors: comprehensive, template_correlation, eegnet, transformer
- Models: Random Forest, Logistic Regression, SVM
- Multiclass Problems: 7-class, 14-class

Usage:
    python run_multiclass_experiments.py --help
    python run_multiclass_experiments.py --7class
    python run_multiclass_experiments.py --14class
    python run_multiclass_experiments.py --all
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

class MulticlassExperimentFramework:
    """Framework for running comprehensive multiclass ECoG experiments."""
    
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
        
        print(f"🚀 Multiclass Experiment Framework")
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
                    features[extractor_name] = np.load(feature_file)
                    print(f"  ✅ Loaded {extractor_name}: {features[extractor_name].shape}")
        
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
            return labels, valid_mask, self.categories_7class
        
        elif problem_type == '14class':
            # Use 14-class labels (remove background)
            labels = color_labels.copy()
            # Keep only stimulus trials (remove background = -1)
            valid_mask = labels >= 0
            return labels, valid_mask, self.categories_14class
        
        else:
            raise ValueError(f"Unknown problem type: {problem_type}")
    
    def prepare_features(self, features, valid_mask):
        """Prepare features for classification."""
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
        
        return prepared_features
    
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
                'extractor': extractor_name,
                'model': model_name,
                'problem_type': problem_type,
                'cv_accuracy_mean': scores.mean(),
                'cv_accuracy_std': scores.std(),
                'train_accuracy': accuracy,
                'n_classes': len(np.unique(labels)),
                'n_samples': len(labels),
                'class_distribution': np.bincount(labels).tolist(),
                'classification_report': class_report,
                'confusion_matrix': conf_matrix.tolist(),
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'extractor': extractor_name,
                'model': model_name,
                'problem_type': problem_type,
                'cv_accuracy_mean': 0.0,
                'cv_accuracy_std': 0.0,
                'train_accuracy': 0.0,
                'n_classes': 0,
                'n_samples': 0,
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
                        feature_data, labels, extractor_name, model_name, model, problem_type
                    )
                    all_results.append(result)
                    pbar.update(1)
        
        return all_results, categories
    
    def create_comprehensive_report(self, results_7class, results_14class, categories_7class, categories_14class):
        """Create comprehensive report for all experiments."""
        print("📋 Creating comprehensive multiclass report...")
        
        # Combine all results
        all_results = results_7class + results_14class
        df = pd.DataFrame(all_results)
        
        # Filter successful results
        successful_results = df[df['status'] == 'success']
        
        if len(successful_results) == 0:
            print("❌ No successful results to report!")
            return
        
        # Create report
        report = f"""
# Comprehensive Multiclass ECoG Experiment Report
Experiment: {self.experiment_id}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- Total combinations tested: {len(all_results)}
- Successful: {len(successful_results)}
- Failed: {len(all_results) - len(successful_results)}

## 7-Class Multiclass Results
"""
        
        # 7-class results
        results_7 = successful_results[successful_results['problem_type'] == '7class']
        if len(results_7) > 0:
            best_7 = results_7.loc[results_7['cv_accuracy_mean'].idxmax()]
            report += f"""
### Best 7-Class Result
- **Accuracy**: {best_7['cv_accuracy_mean']:.3f} ± {best_7['cv_accuracy_std']:.3f}
- **Feature Extractor**: {best_7['extractor']}
- **Model**: {best_7['model']}
- **Classes**: {best_7['n_classes']}
- **Samples**: {best_7['n_samples']}

### Top 5 7-Class Results
"""
            top_5_7 = results_7.nlargest(5, 'cv_accuracy_mean')
            for i, (_, row) in enumerate(top_5_7.iterrows(), 1):
                report += f"{i}. {row['extractor']} + {row['model']}: {row['cv_accuracy_mean']:.3f} ± {row['cv_accuracy_std']:.3f}\n"
        
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
- **Feature Extractor**: {best_14['extractor']}
- **Model**: {best_14['model']}
- **Classes**: {best_14['n_classes']}
- **Samples**: {best_14['n_samples']}

### Top 5 14-Class Results
"""
            top_5_14 = results_14.nlargest(5, 'cv_accuracy_mean')
            for i, (_, row) in enumerate(top_5_14.iterrows(), 1):
                report += f"{i}. {row['extractor']} + {row['model']}: {row['cv_accuracy_mean']:.3f} ± {row['cv_accuracy_std']:.3f}\n"
        
        # Overall comparison
        report += f"""
## Overall Comparison
"""
        if len(results_7) > 0 and len(results_14) > 0:
            best_7_overall = results_7['cv_accuracy_mean'].max()
            best_14_overall = results_14['cv_accuracy_mean'].max()
            report += f"""
- **Best 7-class accuracy**: {best_7_overall:.3f}
- **Best 14-class accuracy**: {best_14_overall:.3f}
- **Performance difference**: {((best_14_overall / best_7_overall) - 1) * 100:+.1f}%
"""
        
        # Save report
        report_file = self.results_dir / f"multiclass_comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Save results as JSON
        results_file = self.results_dir / f"multiclass_comprehensive_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"📁 Report saved: {report_file}")
        print(f"📁 Results saved: {results_file}")
        
        return successful_results
    
    def create_comprehensive_visualizations(self, successful_results):
        """Create comprehensive visualizations."""
        print("🎨 Creating comprehensive multiclass visualizations...")
        
        df = pd.DataFrame(successful_results)
        
        # Create large visualization
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Accuracy comparison by problem type
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
        
        # 6. Class distribution comparison
        plt.subplot(3, 3, 6)
        class_counts = df.groupby('problem_type')['n_classes'].first()
        plt.bar(class_counts.index, class_counts.values, color=['skyblue', 'lightcoral'])
        plt.title('Number of Classes by Problem Type', fontweight='bold', fontsize=14)
        plt.ylabel('Number of Classes')
        for i, v in enumerate(class_counts.values):
            plt.text(i, v + 0.1, f'{v}', ha='center', fontweight='bold')
        
        # 7. Sample size comparison
        plt.subplot(3, 3, 7)
        sample_counts = df.groupby('problem_type')['n_samples'].first()
        plt.bar(sample_counts.index, sample_counts.values, color=['lightblue', 'pink'])
        plt.title('Number of Samples by Problem Type', fontweight='bold', fontsize=14)
        plt.ylabel('Number of Samples')
        for i, v in enumerate(sample_counts.values):
            plt.text(i, v + 1, f'{v}', ha='center', fontweight='bold')
        
        # 8. Accuracy distribution
        plt.subplot(3, 3, 8)
        plt.hist(df['cv_accuracy_mean'], bins=20, alpha=0.7, color='purple', edgecolor='black')
        plt.title('Accuracy Distribution', fontweight='bold', fontsize=14)
        plt.xlabel('Accuracy')
        plt.ylabel('Frequency')
        plt.axvline(df['cv_accuracy_mean'].mean(), color='red', linestyle='--', label=f'Mean: {df["cv_accuracy_mean"].mean():.3f}')
        plt.legend()
        
        # 9. Best results summary
        plt.subplot(3, 3, 9)
        plt.axis('off')
        best_7 = df[df['problem_type'] == '7class']['cv_accuracy_mean'].max() if len(df[df['problem_type'] == '7class']) > 0 else 0
        best_14 = df[df['problem_type'] == '14class']['cv_accuracy_mean'].max() if len(df[df['problem_type'] == '14class']) > 0 else 0
        
        summary_text = f"""
🏆 BEST RESULTS SUMMARY

7-Class Multiclass:
  Best Accuracy: {best_7:.3f}
  Classes: 7 (digit, kanji, face, body, object, hiragana, line)

14-Class Multiclass:
  Best Accuracy: {best_14:.3f}
  Classes: 14 (7 categories × 2 colors)

Total Experiments: {len(df)}
Successful: {len(df)}
Failed: 0

Experiment ID: {self.experiment_id}
        """
        plt.text(0.1, 0.5, summary_text, transform=plt.gca().transAxes, fontsize=12,
                verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
        
        plt.suptitle('Comprehensive Multiclass ECoG Experiment Analysis', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'multiclass_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.results_dir / 'multiclass_comprehensive_analysis.svg', bbox_inches='tight')
        plt.close()
        
        print("  📊 Comprehensive analysis plot saved")
    
    def run_all_experiments(self):
        """Run all multiclass experiments."""
        print("🚀 Running all multiclass experiments...")
        
        start_time = time.time()
        
        # Run 7-class experiments
        results_7class, categories_7class = self.run_multiclass_experiments('7class')
        
        # Run 14-class experiments
        results_14class, categories_14class = self.run_multiclass_experiments('14class')
        
        # Create comprehensive report
        successful_results = self.create_comprehensive_report(
            results_7class, results_14class, categories_7class, categories_14class
        )
        
        # Create visualizations
        self.create_comprehensive_visualizations(successful_results)
        
        elapsed_time = time.time() - start_time
        
        print(f"\n🎉 All Multiclass Experiments Completed!")
        print("=" * 60)
        print(f"⏱️ Total Time: {elapsed_time:.1f} seconds")
        print(f"📁 Results Directory: {self.results_dir}")
        
        # Show best results
        if len(successful_results) > 0:
            df = pd.DataFrame(successful_results)
            
            print(f"\n🏆 Best Results:")
            for problem_type in ['7class', '14class']:
                problem_results = df[df['problem_type'] == problem_type]
                if len(problem_results) > 0:
                    best = problem_results.loc[problem_results['cv_accuracy_mean'].idxmax()]
                    print(f"  {problem_type.upper()}: {best['cv_accuracy_mean']:.3f} ({best['extractor']} + {best['model']})")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run comprehensive multiclass ECoG experiments')
    parser.add_argument('--7class', action='store_true', help='Run only 7-class experiments')
    parser.add_argument('--14class', action='store_true', help='Run only 14-class experiments')
    parser.add_argument('--all', action='store_true', help='Run all experiments (default)')
    
    args = parser.parse_args()
    
    # Default to all if no specific option is chosen
    if not any([args.__dict__[key] for key in ['7class', '14class']]):
        args.all = True
    
    framework = MulticlassExperimentFramework()
    
    if args.all:
        framework.run_all_experiments()
    elif args.__dict__['7class']:
        results_7class, categories_7class = framework.run_multiclass_experiments('7class')
        framework.create_comprehensive_report([], results_7class, {}, categories_7class)
    elif args.__dict__['14class']:
        results_14class, categories_14class = framework.run_multiclass_experiments('14class')
        framework.create_comprehensive_report(results_14class, [], categories_14class, {})

if __name__ == "__main__":
    main()
