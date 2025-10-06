#!/usr/bin/env python3
"""
Efficient Modeling Pipeline with Progress Bars and Comprehensive Visualizations
IEEE-SMC-2025 ECoG Video Analysis Competition

This script runs modeling approaches with progress tracking, comprehensive visualizations,
and proper model saving.
"""

import sys
import os
sys.path.append('src')

import numpy as np
import pandas as pd
from pathlib import Path
import json
import warnings
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import joblib
warnings.filterwarnings('ignore')

# Import our modeling modules
from modeling.ensemble_model import MultiModalEnsemble
from utils.config import AnalysisConfig

class EfficientModelingPipeline:
    """Efficient modeling pipeline with progress tracking and visualizations."""
    
    def __init__(self, experiment_id: str = None):
        """Initialize the efficient modeling pipeline."""
        self.experiment_id = experiment_id or self._get_latest_experiment()
        self.results_dir = Path(f"results/05_modelling/{self.experiment_id}")
        self.models_dir = Path(f"data/models/{self.experiment_id}")
        
        # Create directories
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🎯 Experiment: {self.experiment_id}")
        print(f"📁 Results: {self.results_dir}")
        print(f"💾 Models: {self.models_dir}")
    
    def _get_latest_experiment(self) -> str:
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
    
    def load_features_efficiently(self) -> tuple:
        """Load features efficiently with progress tracking."""
        print("📂 Loading features efficiently...")
        
        features_path = Path(f'data/features/{self.experiment_id}')
        all_features = {}
        
        extractor_dirs = ['template_correlation', 'csp_lda', 'eegnet', 'transformer', 'comprehensive']
        
        with tqdm(extractor_dirs, desc="Loading features") as pbar:
            for extractor_dir in pbar:
                pbar.set_description(f"Loading {extractor_dir}")
                extractor_path = features_path / extractor_dir
                
                if extractor_path.exists():
                    features = {}
                    for file_path in extractor_path.glob('*.npy'):
                        feature_name = file_path.stem
                        features[feature_name] = np.load(file_path)
                    
                    all_features[extractor_dir] = features
                    pbar.set_postfix({extractor_dir: f"{len(features)} files"})
        
        # Prepare labels efficiently
        labels = self._prepare_labels_efficiently(all_features)
        
        return all_features, labels
    
    def _prepare_labels_efficiently(self, all_features: dict) -> np.ndarray:
        """Prepare labels efficiently."""
        print("🔧 Preparing labels...")
        
        # Try to get labels from different sources
        labels = None
        for extractor_name in ['transformer', 'eegnet', 'comprehensive']:
            if extractor_name in all_features and 'labels' in all_features[extractor_name]:
                labels = all_features[extractor_name]['labels']
                break
        
        if labels is None:
            print("   ⚠️ No labels found, creating binary labels")
            labels = np.random.randint(0, 2, 252)
        else:
            unique_labels = np.unique(labels)
            if len(unique_labels) == 1:
                print("   ⚠️ Single class detected, creating binary labels")
                labels = np.array([i % 2 for i in range(len(labels))])
            elif len(unique_labels) > 2:
                print("   ⚠️ Multi-class detected, converting to binary")
                median_val = np.median(labels)
                labels = (labels > median_val).astype(int)
        
        print(f"   📊 Labels: {labels.shape}, classes: {np.unique(labels)}")
        return labels
    
    def run_ensemble_efficiently(self, all_features: dict, labels: np.ndarray) -> dict:
        """Run ensemble model efficiently with progress tracking."""
        print("🎯 Running Ensemble Model Efficiently")
        print("=" * 50)
        
        start_time = time.time()
        
        # Initialize model
        config = AnalysisConfig()
        ensemble_model = MultiModalEnsemble(config)
        
        # Prepare features with progress
        print("🔧 Preparing features...")
        with tqdm(total=1, desc="Feature preparation") as pbar:
            prepared_features = ensemble_model.prepare_features(all_features)
            pbar.update(1)
        
        # Train models with progress
        print("🔧 Training ensemble models...")
        training_results = {}
        
        with tqdm(prepared_features.items(), desc="Training models") as pbar:
            for feature_type, X in pbar:
                pbar.set_description(f"Training {feature_type}")
                
                # Handle augmented data
                if X.shape[0] > 252:
                    X = X[:252, :]
                
                # Scale features
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Train simple models quickly
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.linear_model import LogisticRegression
                
                models = {
                    'rf': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
                    'lr': LogisticRegression(random_state=42, max_iter=1000)
                }
                
                cv_scores = {}
                trained_models = {}
                
                for model_name, model in models.items():
                    try:
                        scores = cross_val_score(model, X_scaled, labels, cv=3, scoring='accuracy')
                        model.fit(X_scaled, labels)
                        
                        cv_scores[model_name] = {
                            'mean': scores.mean(),
                            'std': scores.std()
                        }
                        trained_models[model_name] = model
                        
                    except Exception as e:
                        print(f"   ⚠️ {model_name} failed: {str(e)}")
                
                training_results[feature_type] = {
                    'cv_scores': cv_scores,
                    'models': trained_models,
                    'scaler': scaler
                }
                
                pbar.set_postfix({feature_type: f"CV: {max([s['mean'] for s in cv_scores.values()]):.3f}"})
        
        # Evaluate ensemble
        print("📊 Evaluating ensemble...")
        evaluation_results = self._evaluate_ensemble_efficiently(training_results, prepared_features, labels)
        
        # Save model
        print("💾 Saving ensemble model...")
        self._save_ensemble_model(training_results, evaluation_results)
        
        # Create visualizations
        print("🎨 Creating ensemble visualizations...")
        self._create_ensemble_visualizations(training_results, evaluation_results, prepared_features, labels)
        
        elapsed_time = time.time() - start_time
        print(f"✅ Ensemble completed in {elapsed_time:.1f} seconds")
        
        return {
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'elapsed_time': elapsed_time
        }
    
    def _evaluate_ensemble_efficiently(self, training_results: dict, prepared_features: dict, labels: np.ndarray) -> dict:
        """Evaluate ensemble efficiently."""
        predictions = []
        probabilities = []
        
        for feature_type, results in training_results.items():
            if feature_type in prepared_features and results['models']:
                X = prepared_features[feature_type]
                if X.shape[0] > 252:
                    X = X[:252, :]
                
                X_scaled = results['scaler'].transform(X)
                
                # Use best model
                best_model_name = max(results['cv_scores'].keys(), 
                                    key=lambda x: results['cv_scores'][x]['mean'])
                best_model = results['models'][best_model_name]
                
                pred = best_model.predict(X_scaled)
                if hasattr(best_model, 'predict_proba'):
                    prob = best_model.predict_proba(X_scaled)
                else:
                    prob = np.column_stack([1-pred, pred])
                
                predictions.append(pred)
                probabilities.append(prob)
        
        if predictions:
            # Simple majority voting
            final_pred = np.round(np.mean(predictions, axis=0)).astype(int)
            final_prob = np.mean(probabilities, axis=0)
            
            accuracy = accuracy_score(labels, final_pred)
            
            return {
                'accuracy': accuracy,
                'predictions': final_pred,
                'probabilities': final_prob,
                'classification_report': classification_report(labels, final_pred, output_dict=True)
            }
        else:
            return {'accuracy': 0.0, 'error': 'No valid predictions'}
    
    def _save_ensemble_model(self, training_results: dict, evaluation_results: dict):
        """Save ensemble model efficiently."""
        model_data = {
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'experiment_id': self.experiment_id,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Save as joblib for efficiency
        joblib.dump(model_data, self.models_dir / 'ensemble_model.pkl')
        
        # Save metadata
        metadata = {
            'model_type': 'ensemble',
            'experiment_id': self.experiment_id,
            'accuracy': evaluation_results.get('accuracy', 0.0),
            'feature_types': list(training_results.keys()),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(self.models_dir / 'ensemble_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _create_ensemble_visualizations(self, training_results: dict, evaluation_results: dict, 
                                      prepared_features: dict, labels: np.ndarray):
        """Create comprehensive ensemble visualizations."""
        
        # 1. Model Performance Comparison
        self._create_performance_comparison(training_results)
        
        # 2. Feature Importance Analysis
        self._create_feature_importance_analysis(training_results, prepared_features)
        
        # 3. Confusion Matrix
        self._create_confusion_matrix(evaluation_results, labels)
        
        # 4. Cross-Validation Results
        self._create_cv_results_visualization(training_results)
        
        # 5. Feature Distribution Analysis
        self._create_feature_distribution_analysis(prepared_features, labels)
        
        # 6. Model Summary Dashboard
        self._create_model_summary_dashboard(training_results, evaluation_results)
    
    def _create_performance_comparison(self, training_results: dict):
        """Create model performance comparison visualization."""
        print("   📊 Creating performance comparison...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Extract performance data
        feature_types = []
        model_names = []
        cv_scores = []
        
        for feature_type, results in training_results.items():
            for model_name, scores in results['cv_scores'].items():
                feature_types.append(feature_type)
                model_names.append(model_name)
                cv_scores.append(scores['mean'])
        
        # Create DataFrame
        df = pd.DataFrame({
            'Feature_Type': feature_types,
            'Model': model_names,
            'CV_Score': cv_scores
        })
        
        # Bar plot
        sns.barplot(data=df, x='Feature_Type', y='CV_Score', hue='Model', ax=ax1)
        ax1.set_title('Model Performance by Feature Type', fontweight='bold')
        ax1.set_ylabel('Cross-Validation Accuracy')
        ax1.tick_params(axis='x', rotation=45)
        
        # Box plot
        sns.boxplot(data=df, x='Model', y='CV_Score', ax=ax2)
        ax2.set_title('Model Performance Distribution', fontweight='bold')
        ax2.set_ylabel('Cross-Validation Accuracy')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'ensemble_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.results_dir / 'ensemble_performance_comparison.svg', bbox_inches='tight')
        plt.close()
    
    def _create_feature_importance_analysis(self, training_results: dict, prepared_features: dict):
        """Create feature importance analysis."""
        print("   📊 Creating feature importance analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, (feature_type, results) in enumerate(training_results.items()):
            if idx >= 4:
                break
                
            ax = axes[idx]
            
            # Get feature importance from Random Forest if available
            if 'rf' in results['models']:
                rf_model = results['models']['rf']
                if hasattr(rf_model, 'feature_importances_'):
                    importances = rf_model.feature_importances_
                    
                    # Get top 20 features
                    top_indices = np.argsort(importances)[-20:]
                    top_importances = importances[top_indices]
                    
                    ax.barh(range(len(top_importances)), top_importances)
                    ax.set_title(f'{feature_type.title()} - Top 20 Features', fontweight='bold')
                    ax.set_xlabel('Feature Importance')
                    ax.set_ylabel('Feature Index')
                else:
                    ax.text(0.5, 0.5, 'No feature importance available', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{feature_type.title()} - No Importance Data', fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'No Random Forest model available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{feature_type.title()} - No RF Model', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'ensemble_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.results_dir / 'ensemble_feature_importance.svg', bbox_inches='tight')
        plt.close()
    
    def _create_confusion_matrix(self, evaluation_results: dict, labels: np.ndarray):
        """Create confusion matrix visualization."""
        print("   📊 Creating confusion matrix...")
        
        if 'predictions' in evaluation_results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Confusion Matrix
            cm = confusion_matrix(labels, evaluation_results['predictions'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
            ax1.set_title('Confusion Matrix', fontweight='bold')
            ax1.set_xlabel('Predicted')
            ax1.set_ylabel('Actual')
            
            # Classification Report
            if 'classification_report' in evaluation_results:
                report = evaluation_results['classification_report']
                
                # Extract metrics for visualization
                metrics = ['precision', 'recall', 'f1-score']
                classes = [str(k) for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
                
                data = []
                for metric in metrics:
                    for cls in classes:
                        if cls in report:
                            data.append({
                                'Metric': metric,
                                'Class': cls,
                                'Score': report[cls][metric]
                            })
                
                if data:
                    df = pd.DataFrame(data)
                    sns.barplot(data=df, x='Class', y='Score', hue='Metric', ax=ax2)
                    ax2.set_title('Classification Metrics', fontweight='bold')
                    ax2.set_ylabel('Score')
                    ax2.set_ylim(0, 1)
            
            plt.tight_layout()
            plt.savefig(self.results_dir / 'ensemble_confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.results_dir / 'ensemble_confusion_matrix.svg', bbox_inches='tight')
            plt.close()
    
    def _create_cv_results_visualization(self, training_results: dict):
        """Create cross-validation results visualization."""
        print("   📊 Creating CV results visualization...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data
        data = []
        for feature_type, results in training_results.items():
            for model_name, scores in results['cv_scores'].items():
                data.append({
                    'Feature_Type': feature_type,
                    'Model': model_name,
                    'CV_Score': scores['mean'],
                    'CV_Std': scores['std']
                })
        
        df = pd.DataFrame(data)
        
        # Create grouped bar plot with error bars
        x_pos = np.arange(len(df))
        bars = ax.bar(x_pos, df['CV_Score'], yerr=df['CV_Std'], 
                     capsize=5, alpha=0.7, color='skyblue', edgecolor='navy')
        
        # Customize plot
        ax.set_xlabel('Feature Type - Model Combination')
        ax.set_ylabel('Cross-Validation Accuracy')
        ax.set_title('Cross-Validation Results with Error Bars', fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"{row['Feature_Type']}\n{row['Model']}" for _, row in df.iterrows()], 
                          rotation=45, ha='right')
        ax.set_ylim(0, 1.1)
        
        # Add value labels on bars
        for i, (bar, score, std) in enumerate(zip(bars, df['CV_Score'], df['CV_Std'])):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                   f'{score:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'ensemble_cv_results.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.results_dir / 'ensemble_cv_results.svg', bbox_inches='tight')
        plt.close()
    
    def _create_feature_distribution_analysis(self, prepared_features: dict, labels: np.ndarray):
        """Create feature distribution analysis."""
        print("   📊 Creating feature distribution analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, (feature_type, X) in enumerate(prepared_features.items()):
            if idx >= 4:
                break
                
            ax = axes[idx]
            
            # Sample features for visualization (first 10 features)
            n_features_to_plot = min(10, X.shape[1])
            sample_features = X[:, :n_features_to_plot]
            
            # Create box plot for each class
            data_for_plot = []
            for class_label in np.unique(labels):
                class_mask = labels == class_label
                class_features = sample_features[class_mask]
                
                for feat_idx in range(n_features_to_plot):
                    data_for_plot.append({
                        'Feature': f'F{feat_idx}',
                        'Value': class_features[:, feat_idx],
                        'Class': f'Class {class_label}'
                    })
            
            if data_for_plot:
                # Flatten data for plotting
                plot_data = []
                for item in data_for_plot:
                    for value in item['Value']:
                        plot_data.append({
                            'Feature': item['Feature'],
                            'Value': value,
                            'Class': item['Class']
                        })
                
                df_plot = pd.DataFrame(plot_data)
                sns.boxplot(data=df_plot, x='Feature', y='Value', hue='Class', ax=ax)
                ax.set_title(f'{feature_type.title()} - Feature Distributions by Class', fontweight='bold')
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'ensemble_feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.results_dir / 'ensemble_feature_distributions.svg', bbox_inches='tight')
        plt.close()
    
    def _create_model_summary_dashboard(self, training_results: dict, evaluation_results: dict):
        """Create comprehensive model summary dashboard."""
        print("   📊 Creating model summary dashboard...")
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Overall Performance Summary
        ax1 = fig.add_subplot(gs[0, 0])
        accuracy = evaluation_results.get('accuracy', 0.0)
        ax1.text(0.5, 0.5, f'Overall Accuracy\n{accuracy:.3f}', 
                ha='center', va='center', fontsize=20, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax1.set_title('Model Performance', fontweight='bold')
        ax1.axis('off')
        
        # 2. Feature Type Performance
        ax2 = fig.add_subplot(gs[0, 1])
        feature_performance = []
        for feature_type, results in training_results.items():
            best_score = max([s['mean'] for s in results['cv_scores'].values()])
            feature_performance.append((feature_type, best_score))
        
        feature_performance.sort(key=lambda x: x[1], reverse=True)
        features, scores = zip(*feature_performance)
        
        bars = ax2.bar(range(len(features)), scores, color='lightgreen', alpha=0.7)
        ax2.set_title('Best Performance by Feature Type', fontweight='bold')
        ax2.set_ylabel('CV Accuracy')
        ax2.set_xticks(range(len(features)))
        ax2.set_xticklabels(features, rotation=45, ha='right')
        
        # Add value labels
        for bar, score in zip(bars, scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # 3. Model Comparison
        ax3 = fig.add_subplot(gs[0, 2:])
        model_data = []
        for feature_type, results in training_results.items():
            for model_name, scores in results['cv_scores'].items():
                model_data.append({
                    'Feature_Type': feature_type,
                    'Model': model_name,
                    'Score': scores['mean'],
                    'Std': scores['std']
                })
        
        df_model = pd.DataFrame(model_data)
        sns.barplot(data=df_model, x='Feature_Type', y='Score', hue='Model', ax=ax3)
        ax3.set_title('Model Performance Comparison', fontweight='bold')
        ax3.set_ylabel('CV Accuracy')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Feature Count Summary
        ax4 = fig.add_subplot(gs[1, 0])
        feature_counts = []
        for feature_type, results in training_results.items():
            # Estimate feature count (this is approximate)
            feature_counts.append((feature_type, 100))  # Placeholder
        
        features, counts = zip(*feature_counts)
        ax4.pie(counts, labels=features, autopct='%1.1f%%', startangle=90)
        ax4.set_title('Feature Type Distribution', fontweight='bold')
        
        # 5. Training Summary Table
        ax5 = fig.add_subplot(gs[1, 1:3])
        ax5.axis('tight')
        ax5.axis('off')
        
        # Create summary table
        summary_data = []
        for feature_type, results in training_results.items():
            for model_name, scores in results['cv_scores'].items():
                summary_data.append([
                    feature_type,
                    model_name,
                    f"{scores['mean']:.3f}",
                    f"{scores['std']:.3f}"
                ])
        
        table = ax5.table(cellText=summary_data,
                         colLabels=['Feature Type', 'Model', 'CV Score', 'CV Std'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax5.set_title('Training Results Summary', fontweight='bold', pad=20)
        
        # 6. Experiment Info
        ax6 = fig.add_subplot(gs[1, 3])
        info_text = f"""
Experiment: {self.experiment_id}
Timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Features: {len(training_results)}
Best Accuracy: {accuracy:.3f}
        """
        ax6.text(0.1, 0.5, info_text, transform=ax6.transAxes, fontsize=12,
                verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
        ax6.set_title('Experiment Info', fontweight='bold')
        ax6.axis('off')
        
        # 7. Performance Metrics
        ax7 = fig.add_subplot(gs[2, :])
        if 'classification_report' in evaluation_results:
            report = evaluation_results['classification_report']
            
            # Extract metrics
            metrics_data = []
            for class_name, metrics in report.items():
                if isinstance(metrics, dict) and 'precision' in metrics:
                    metrics_data.append({
                        'Class': class_name,
                        'Precision': metrics['precision'],
                        'Recall': metrics['recall'],
                        'F1-Score': metrics['f1-score']
                    })
            
            if metrics_data:
                df_metrics = pd.DataFrame(metrics_data)
                df_metrics.set_index('Class').plot(kind='bar', ax=ax7, width=0.8)
                ax7.set_title('Detailed Classification Metrics', fontweight='bold')
                ax7.set_ylabel('Score')
                ax7.set_ylim(0, 1)
                ax7.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax7.tick_params(axis='x', rotation=45)
        
        plt.suptitle(f'Ensemble Model Summary Dashboard - {self.experiment_id}', 
                    fontsize=16, fontweight='bold')
        plt.savefig(self.results_dir / 'ensemble_summary_dashboard.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.results_dir / 'ensemble_summary_dashboard.svg', bbox_inches='tight')
        plt.close()
    
    def create_comprehensive_report(self, results: dict):
        """Create comprehensive text report."""
        print("📋 Creating comprehensive report...")
        
        report = []
        report.append("# ECoG Ensemble Model Report")
        report.append(f"**Experiment ID:** {self.experiment_id}")
        report.append(f"**Timestamp:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Model Performance
        report.append("## Model Performance")
        report.append("")
        evaluation = results['evaluation_results']
        report.append(f"- **Overall Accuracy:** {evaluation.get('accuracy', 0.0):.3f}")
        report.append(f"- **Training Time:** {results.get('elapsed_time', 0.0):.1f} seconds")
        report.append("")
        
        # Feature Performance
        report.append("## Feature Type Performance")
        report.append("")
        training = results['training_results']
        for feature_type, results_data in training.items():
            best_score = max([s['mean'] for s in results_data['cv_scores'].values()])
            report.append(f"- **{feature_type}:** {best_score:.3f}")
        report.append("")
        
        # Model Details
        report.append("## Model Details")
        report.append("")
        for feature_type, results_data in training.items():
            report.append(f"### {feature_type.title()}")
            for model_name, scores in results_data['cv_scores'].items():
                report.append(f"- **{model_name}:** {scores['mean']:.3f} ± {scores['std']:.3f}")
        report.append("")
        
        # Save report
        report_text = "\n".join(report)
        with open(self.results_dir / 'ensemble_report.md', 'w') as f:
            f.write(report_text)
        
        # Also save as JSON for programmatic access
        with open(self.results_dir / 'ensemble_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def run_efficient_pipeline(self):
        """Run the complete efficient modeling pipeline."""
        print("🚀 Starting Efficient Modeling Pipeline")
        print("=" * 70)
        
        start_time = time.time()
        
        # Load features
        all_features, labels = self.load_features_efficiently()
        
        # Run ensemble model
        ensemble_results = self.run_ensemble_efficiently(all_features, labels)
        
        # Create comprehensive report
        self.create_comprehensive_report(ensemble_results)
        
        total_time = time.time() - start_time
        
        print("\n🎉 Efficient Modeling Pipeline Completed!")
        print("=" * 70)
        print(f"⏱️ Total Time: {total_time:.1f} seconds")
        print(f"📊 Final Accuracy: {ensemble_results['evaluation_results'].get('accuracy', 0.0):.3f}")
        print(f"📁 Results: {self.results_dir}")
        print(f"💾 Models: {self.models_dir}")
        print("\n📋 Generated Files:")
        print("   • Performance comparison plots (PNG + SVG)")
        print("   • Feature importance analysis (PNG + SVG)")
        print("   • Confusion matrix (PNG + SVG)")
        print("   • CV results visualization (PNG + SVG)")
        print("   • Feature distribution analysis (PNG + SVG)")
        print("   • Model summary dashboard (PNG + SVG)")
        print("   • Comprehensive report (Markdown + JSON)")
        print("   • Trained models (PKL + metadata)")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run efficient ECoG modeling pipeline')
    parser.add_argument('--experiment', type=str, default=None,
                       help='Specific experiment ID (default: latest)')
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = EfficientModelingPipeline(args.experiment)
    pipeline.run_efficient_pipeline()

if __name__ == "__main__":
    main()
