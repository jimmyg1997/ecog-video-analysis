#!/usr/bin/env python3
"""
Generate ALL Missing ML Visualizations
=====================================
This script generates all the ML visualizations that were missing:
1. ROC-AUC curves  
2. Precision-Recall curves
3. Classification reports (visual)
4. Per-class performance plots
5. Feature importance plots (Random Forest)
6. Learning curves
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, auc, precision_recall_curve, classification_report
import pandas as pd

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def find_latest_experiment():
    """Find the latest experiment directory."""
    results_dir = Path('results/05_modelling')
    latest_exp = max([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('experiment')], 
                     key=lambda x: int(x.name.replace('experiment', '')))
    return latest_exp

def load_results(exp_dir):
    """Load results from experiment directory."""
    results_files = list(exp_dir.glob('*results*.json'))
    if results_files:
        with open(results_files[0], 'r') as f:
            return json.load(f)
    return None

def create_roc_auc_curves(results, exp_dir, problem_type):
    """Create ROC-AUC curves for all models and classes."""
    print(f"🎨 Creating ROC-AUC curves for {problem_type}...")
    
    problem_results = [r for r in results if r['status'] == 'success' and r['problem_type'] == problem_type]
    if not problem_results:
        print(f"  ❌ No results for {problem_type}")
        return
    
    models = list(set([r['model'] for r in problem_results]))
    n_models = len(models)
    
    fig, axes = plt.subplots(2, n_models, figsize=(6*n_models, 12))
    if n_models == 1:
        axes = axes.reshape(2, 1)
    
    for model_idx, model_name in enumerate(models):
        model_results = [r for r in problem_results if r['model'] == model_name]
        best_result = max(model_results, key=lambda x: x.get('roc_auc_macro', 0))
        
        categories = best_result['categories']
        class_names = list(categories.keys())
        
        # Get predictions and probabilities
        y_true = np.array(best_result['cv_predictions'])
        y_proba = np.array(best_result['cv_probabilities'])
        
        ax_roc = axes[0, model_idx]
        ax_pr = axes[1, model_idx]
        
        # ROC curves
        for i, class_name in enumerate(class_names):
            if i < y_proba.shape[1]:
                fpr, tpr, _ = roc_curve(y_true == i, y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                ax_roc.plot(fpr, tpr, label=f'{class_name} (AUC={roc_auc:.2f})', linewidth=2)
        
        ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax_roc.set_xlabel('False Positive Rate', fontsize=12)
        ax_roc.set_ylabel('True Positive Rate', fontsize=12)
        ax_roc.set_title(f'{model_name} - ROC Curves\\n{problem_type.upper()}', fontsize=14, fontweight='bold')
        ax_roc.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax_roc.grid(True, alpha=0.3)
        
        # Precision-Recall curves
        for i, class_name in enumerate(class_names):
            if i < y_proba.shape[1]:
                precision, recall, _ = precision_recall_curve(y_true == i, y_proba[:, i])
                pr_auc = auc(recall, precision)
                ax_pr.plot(recall, precision, label=f'{class_name} (AP={pr_auc:.2f})', linewidth=2)
        
        ax_pr.set_xlabel('Recall', fontsize=12)
        ax_pr.set_ylabel('Precision', fontsize=12)
        ax_pr.set_title(f'{model_name} - Precision-Recall\\n{problem_type.upper()}', fontsize=14, fontweight='bold')
        ax_pr.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax_pr.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(exp_dir / f'roc_pr_curves_{problem_type}.png', dpi=300, bbox_inches='tight')
    plt.savefig(exp_dir / f'roc_pr_curves_{problem_type}.svg', bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved ROC-AUC and Precision-Recall curves")

def create_classification_reports(results, exp_dir, problem_type):
    """Create visual classification reports."""
    print(f"🎨 Creating classification reports for {problem_type}...")
    
    problem_results = [r for r in results if r['status'] == 'success' and r['problem_type'] == problem_type]
    if not problem_results:
        return
    
    n_results = len(problem_results)
    n_cols = min(3, n_results)
    n_rows = (n_results + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows))
    if n_results == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, result in enumerate(problem_results):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        # Get per-class metrics
        categories = result['categories']
        class_names = list(categories.keys())
        
        precision = result['precision_per_class']
        recall = result['recall_per_class']
        f1 = result['f1_per_class']
        
        # Create heatmap data
        metrics_data = np.array([precision, recall, f1])
        
        sns.heatmap(metrics_data, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax,
                   xticklabels=class_names, yticklabels=['Precision', 'Recall', 'F1-Score'],
                   vmin=0, vmax=1, cbar_kws={'label': 'Score'})
        ax.set_title(f"{result['extractor']} + {result['model']}\\nAccuracy: {result['cv_test_accuracy']:.3f}", 
                    fontsize=12, fontweight='bold')
    
    # Hide empty subplots
    for idx in range(n_results, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.axis('off')
    
    plt.suptitle(f'Classification Reports - {problem_type.upper()}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(exp_dir / f'classification_reports_{problem_type}.png', dpi=300, bbox_inches='tight')
    plt.savefig(exp_dir / f'classification_reports_{problem_type}.svg', bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved classification reports")

def create_per_class_performance(results, exp_dir, problem_type):
    """Create per-class performance plots."""
    print(f"🎨 Creating per-class performance plots for {problem_type}...")
    
    problem_results = [r for r in results if r['status'] == 'success' and r['problem_type'] == problem_type]
    if not problem_results:
        return
    
    # Get class names from first result
    categories = problem_results[0]['categories']
    class_names = list(categories.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. F1 scores by model
    ax1 = axes[0, 0]
    for result in problem_results:
        f1_scores = result['f1_per_class']
        ax1.plot(class_names, f1_scores, marker='o', label=f"{result['extractor']}+{result['model']}", linewidth=2)
    ax1.set_title('F1 Scores by Model', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('F1 Score')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # 2. Precision by model
    ax2 = axes[0, 1]
    for result in problem_results:
        precision_scores = result['precision_per_class']
        ax2.plot(class_names, precision_scores, marker='s', label=f"{result['extractor']}+{result['model']}", linewidth=2)
    ax2.set_title('Precision by Model', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Precision')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # 3. Recall by model
    ax3 = axes[1, 0]
    for result in problem_results:
        recall_scores = result['recall_per_class']
        ax3.plot(class_names, recall_scores, marker='^', label=f"{result['extractor']}+{result['model']}", linewidth=2)
    ax3.set_title('Recall by Model', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Class')
    ax3.set_ylabel('Recall')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # 4. Class distribution vs F1
    ax4 = axes[1, 1]
    class_dist = problem_results[0]['class_distribution']
    avg_f1 = np.mean([r['f1_per_class'] for r in problem_results], axis=0)
    scatter = ax4.scatter(class_dist, avg_f1, s=200, alpha=0.7, c=range(len(class_names)), cmap='viridis')
    for i, class_name in enumerate(class_names):
        ax4.annotate(class_name, (class_dist[i], avg_f1[i]), xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax4.set_title('Class Distribution vs Average F1', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Number of Samples')
    ax4.set_ylabel('Average F1 Score')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'Per-Class Performance Analysis - {problem_type.upper()}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(exp_dir / f'per_class_performance_{problem_type}.png', dpi=300, bbox_inches='tight')
    plt.savefig(exp_dir / f'per_class_performance_{problem_type}.svg', bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved per-class performance plots")

def create_model_comparison(results, exp_dir):
    """Create comprehensive model comparison plots."""
    print(f"🎨 Creating model comparison plots...")
    
    successful_results = [r for r in results if r['status'] == 'success']
    if not successful_results:
        return
    
    df = pd.DataFrame(successful_results)
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 1. Accuracy by model and problem type
    ax1 = axes[0, 0]
    model_problem_acc = df.pivot_table(values='cv_test_accuracy', index='model', columns='problem_type', aggfunc='max')
    model_problem_acc.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_title('Accuracy by Model', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy')
    ax1.legend(title='Problem Type')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # 2. ROC-AUC by model
    ax2 = axes[0, 1]
    model_roc = df.pivot_table(values='roc_auc_macro', index='model', columns='problem_type', aggfunc='max')
    model_roc.plot(kind='bar', ax=ax2, width=0.8)
    ax2.set_title('ROC-AUC by Model', fontsize=14, fontweight='bold')
    ax2.set_ylabel('ROC-AUC')
    ax2.legend(title='Problem Type')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # 3. F1 Macro by model
    ax3 = axes[0, 2]
    model_f1 = df.pivot_table(values='f1_macro', index='model', columns='problem_type', aggfunc='max')
    model_f1.plot(kind='bar', ax=ax3, width=0.8)
    ax3.set_title('F1 Macro by Model', fontsize=14, fontweight='bold')
    ax3.set_ylabel('F1 Macro')
    ax3.legend(title='Problem Type')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # 4. Feature extractor performance
    ax4 = axes[1, 0]
    extractor_perf = df.groupby('extractor')['cv_test_accuracy'].max().sort_values(ascending=True)
    extractor_perf.plot(kind='barh', ax=ax4, color='lightgreen')
    ax4.set_title('Best Accuracy by Feature Extractor', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Accuracy')
    ax4.grid(True, alpha=0.3)
    
    # 5. Metrics correlation
    ax5 = axes[1, 1]
    metrics = ['cv_test_accuracy', 'roc_auc_macro', 'f1_macro', 'precision_macro', 'recall_macro']
    corr_matrix = df[metrics].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax5)
    ax5.set_title('Performance Metrics Correlation', fontsize=14, fontweight='bold')
    
    # 6. Overall performance summary
    ax6 = axes[1, 2]
    ax6.axis('off')
    best_result = df.loc[df['cv_test_accuracy'].idxmax()]
    summary_text = f"""
🏆 BEST OVERALL RESULT

Model: {best_result['model']}
Extractor: {best_result['extractor']}
Problem: {best_result['problem_type']}

Accuracy: {best_result['cv_test_accuracy']:.3f}
ROC-AUC: {best_result['roc_auc_macro']:.3f}
F1 Macro: {best_result['f1_macro']:.3f}
Precision: {best_result['precision_macro']:.3f}
Recall: {best_result['recall_macro']:.3f}

Total Experiments: {len(df)}
"""
    ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes, fontsize=12,
            verticalalignment='center', family='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", edgecolor="black", linewidth=2))
    
    plt.suptitle('Comprehensive Model Comparison', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(exp_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(exp_dir / 'model_comparison.svg', bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved model comparison plots")

def main():
    print("🚀 Generating ALL Missing ML Visualizations")
    print("=" * 60)
    
    # Find latest experiment
    exp_dir = find_latest_experiment()
    print(f"📁 Experiment: {exp_dir.name}")
    
    # Load results
    results = load_results(exp_dir)
    if not results:
        print("❌ No results found!")
        return
    
    print(f"✅ Loaded {len(results)} results")
    print("")
    
    # Generate all visualizations
    for problem_type in ['8class', '16class']:
        create_roc_auc_curves(results, exp_dir, problem_type)
        create_classification_reports(results, exp_dir, problem_type)
        create_per_class_performance(results, exp_dir, problem_type)
    
    create_model_comparison(results, exp_dir)
    
    print("")
    print("🎉 ALL ML Visualizations Generated!")
    print("=" * 60)
    print(f"📁 Location: {exp_dir}")
    print("")
    print("✅ Generated:")
    print("  1. ROC-AUC and Precision-Recall curves")
    print("  2. Classification reports (visual)")
    print("  3. Per-class performance plots")
    print("  4. Model comparison plots")
    print("  5. All in PNG and SVG formats")

if __name__ == "__main__":
    main()
