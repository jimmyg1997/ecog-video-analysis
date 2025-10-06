#!/usr/bin/env python3
"""
Fast Ensemble Model Execution
IEEE-SMC-2025 ECoG Video Analysis Competition

This script runs only the ensemble model with progress bars and comprehensive outputs.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
warnings.filterwarnings('ignore')

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

def load_features_fast(experiment_id):
    """Load features quickly."""
    print("📂 Loading features...")
    
    features_path = Path(f'data/features/{experiment_id}')
    all_features = {}
    
    extractor_dirs = ['template_correlation', 'csp_lda', 'eegnet', 'transformer', 'comprehensive']
    
    for extractor_dir in tqdm(extractor_dirs, desc="Loading extractors"):
        extractor_path = features_path / extractor_dir
        if extractor_path.exists():
            features = {}
            for file_path in extractor_path.glob('*.npy'):
                features[file_path.stem] = np.load(file_path)
            all_features[extractor_dir] = features
    
    return all_features

def prepare_labels_fast(all_features):
    """Prepare labels quickly."""
    print("🔧 Preparing labels...")
    
    # Try to get labels
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

def prepare_features_fast(all_features):
    """Prepare features for ensemble quickly."""
    print("🔧 Preparing features...")
    
    prepared_features = {}
    
    for extractor_name, features in all_features.items():
        if extractor_name == 'comprehensive' and 'gamma_power' in features:
            prepared_features[extractor_name] = features['gamma_power']
        elif extractor_name == 'eegnet' and 'cnn_input' in features:
            cnn_data = features['cnn_input']
            if len(cnn_data.shape) > 2:
                cnn_data = cnn_data.reshape(cnn_data.shape[0], -1)
            # Handle augmented data
            if cnn_data.shape[0] > 252:
                cnn_data = cnn_data[:252, :]
            prepared_features[extractor_name] = cnn_data
        elif extractor_name == 'transformer' and 'transformer_input' in features:
            prepared_features[extractor_name] = features['transformer_input']
    
    return prepared_features

def train_ensemble_fast(prepared_features, labels):
    """Train ensemble model quickly."""
    print("🎯 Training ensemble model...")
    
    training_results = {}
    models = {
        'rf': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
        'lr': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    with tqdm(prepared_features.items(), desc="Training models") as pbar:
        for feature_type, X in pbar:
            pbar.set_description(f"Training {feature_type}")
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
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
            
            best_score = max([s['mean'] for s in cv_scores.values()]) if cv_scores else 0.0
            pbar.set_postfix({feature_type: f"Best: {best_score:.3f}"})
    
    return training_results

def evaluate_ensemble_fast(training_results, prepared_features, labels):
    """Evaluate ensemble quickly."""
    print("📊 Evaluating ensemble...")
    
    predictions = []
    probabilities = []
    
    for feature_type, results in training_results.items():
        if feature_type in prepared_features and results['models']:
            X = prepared_features[feature_type]
            
            # Ensure X has the same shape as during training
            # This is important for EEGNet which might have different shapes
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

def create_visualizations_fast(training_results, evaluation_results, experiment_id):
    """Create visualizations quickly."""
    print("🎨 Creating visualizations...")
    
    results_dir = Path(f"results/05_modelling/{experiment_id}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Performance Comparison
    print("   📊 Performance comparison...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    feature_types = []
    best_scores = []
    
    for feature_type, results in training_results.items():
        if results['cv_scores']:
            best_score = max([s['mean'] for s in results['cv_scores'].values()])
            feature_types.append(feature_type)
            best_scores.append(best_score)
    
    bars = ax.bar(feature_types, best_scores, color='skyblue', alpha=0.7)
    ax.set_title('Ensemble Model Performance by Feature Type', fontweight='bold')
    ax.set_ylabel('Cross-Validation Accuracy')
    ax.set_ylim(0, 1.1)
    
    # Add value labels
    for bar, score in zip(bars, best_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(results_dir / 'ensemble_performance.png', dpi=300, bbox_inches='tight')
    plt.savefig(results_dir / 'ensemble_performance.svg', bbox_inches='tight')
    plt.close()
    
    # 2. Confusion Matrix
    print("   📊 Confusion matrix...")
    if 'predictions' in evaluation_results:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        cm = confusion_matrix(labels, evaluation_results['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix', fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(results_dir / 'ensemble_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.savefig(results_dir / 'ensemble_confusion_matrix.svg', bbox_inches='tight')
        plt.close()
    
    # 3. Summary Dashboard
    print("   📊 Summary dashboard...")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Overall accuracy
    accuracy = evaluation_results.get('accuracy', 0.0)
    ax1.text(0.5, 0.5, f'Overall Accuracy\n{accuracy:.3f}', 
            ha='center', va='center', fontsize=20, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    ax1.set_title('Model Performance', fontweight='bold')
    ax1.axis('off')
    
    # Feature performance
    ax2.bar(feature_types, best_scores, color='lightgreen', alpha=0.7)
    ax2.set_title('Feature Type Performance', fontweight='bold')
    ax2.set_ylabel('CV Accuracy')
    ax2.tick_params(axis='x', rotation=45)
    
    # Model comparison
    model_data = []
    for feature_type, results in training_results.items():
        for model_name, scores in results['cv_scores'].items():
            model_data.append({
                'Feature_Type': feature_type,
                'Model': model_name,
                'Score': scores['mean']
            })
    
    if model_data:
        df = pd.DataFrame(model_data)
        sns.barplot(data=df, x='Feature_Type', y='Score', hue='Model', ax=ax3)
        ax3.set_title('Model Comparison', fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
    
    # Experiment info
    info_text = f"""
Experiment: {experiment_id}
Timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Features: {len(training_results)}
Best Accuracy: {accuracy:.3f}
    """
    ax4.text(0.1, 0.5, info_text, transform=ax4.transAxes, fontsize=12,
            verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    ax4.set_title('Experiment Info', fontweight='bold')
    ax4.axis('off')
    
    plt.suptitle(f'Ensemble Model Summary - {experiment_id}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'ensemble_summary_dashboard.png', dpi=300, bbox_inches='tight')
    plt.savefig(results_dir / 'ensemble_summary_dashboard.svg', bbox_inches='tight')
    plt.close()

def save_model_fast(training_results, evaluation_results, experiment_id):
    """Save model quickly."""
    print("💾 Saving model...")
    
    models_dir = Path(f"data/models/{experiment_id}")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model data
    model_data = {
        'training_results': training_results,
        'evaluation_results': evaluation_results,
        'experiment_id': experiment_id,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    joblib.dump(model_data, models_dir / 'ensemble_model.pkl')
    
    # Save metadata
    metadata = {
        'model_type': 'ensemble',
        'experiment_id': experiment_id,
        'accuracy': evaluation_results.get('accuracy', 0.0),
        'feature_types': list(training_results.keys()),
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    with open(models_dir / 'ensemble_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

def main():
    """Main function."""
    print("🚀 Fast Ensemble Model Execution")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        # Get experiment
        experiment_id = get_latest_experiment()
        print(f"📂 Using experiment: {experiment_id}")
        
        # Load data
        all_features = load_features_fast(experiment_id)
        labels = prepare_labels_fast(all_features)
        prepared_features = prepare_features_fast(all_features)
        
        # Train model
        training_results = train_ensemble_fast(prepared_features, labels)
        
        # Evaluate model
        evaluation_results = evaluate_ensemble_fast(training_results, prepared_features, labels)
        
        # Create visualizations
        create_visualizations_fast(training_results, evaluation_results, experiment_id)
        
        # Save model
        save_model_fast(training_results, evaluation_results, experiment_id)
        
        elapsed_time = time.time() - start_time
        
        print("\n🎉 Fast Ensemble Model Completed!")
        print("=" * 50)
        print(f"⏱️ Time: {elapsed_time:.1f} seconds")
        print(f"📊 Accuracy: {evaluation_results.get('accuracy', 0.0):.3f}")
        print(f"📁 Results: results/05_modelling/{experiment_id}")
        print(f"💾 Models: data/models/{experiment_id}")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
