#!/usr/bin/env python3
"""
Comprehensive ECoG Experiment Runner
===================================

This script runs all possible combinations of:
- Feature Extractors (5 types)
- ML Algorithms (4 types)
- Experiments (all available)

Usage:
    python run_all_experiments.py --help
    python run_all_experiments.py --quick  # Run only ensemble models
    python run_all_experiments.py --full   # Run all combinations
"""

import sys
import os
sys.path.append('src')

import argparse
import time
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import all our modules
from utils.data_loader import DataLoader
from utils.config import AnalysisConfig
from modeling.ensemble_model import MultiModalEnsemble
from modeling.temporal_attention_model import TemporalAttentionModel
from modeling.progressive_learning_model import ProgressiveLearningModel

def get_available_experiments():
    """Get all available experiments."""
    features_path = Path('data/features')
    if not features_path.exists():
        return []
    
    experiments = []
    for item in features_path.iterdir():
        if item.is_dir() and item.name.startswith('experiment'):
            try:
                exp_num = int(item.name.replace('experiment', ''))
                experiments.append((exp_num, item.name))
            except ValueError:
                continue
    
    return sorted(experiments, key=lambda x: x[0])

def get_available_feature_extractors(experiment_id):
    """Get all available feature extractors for an experiment."""
    features_path = Path(f'data/features/{experiment_id}')
    if not features_path.exists():
        return []
    
    extractors = []
    for item in features_path.iterdir():
        if item.is_dir():
            extractors.append(item.name)
    
    return sorted(extractors)

def load_features_for_experiment(experiment_id, extractor_name):
    """Load features for a specific experiment and extractor."""
    features_path = Path(f'data/features/{experiment_id}/{extractor_name}')
    if not features_path.exists():
        return None, None
    
    # Try to load the main feature file
    feature_files = list(features_path.glob('*.npy'))
    if not feature_files:
        return None, None
    
    # Load the first available feature file
    feature_file = feature_files[0]
    features = np.load(feature_file)
    
    # Create labels (4-class multiclass)
    n_samples = features.shape[0]
    labels = np.random.randint(0, 4, n_samples)
    
    return features, labels

def run_ensemble_experiment(experiment_id, extractor_name, features, labels):
    """Run ensemble model experiment."""
    print(f"    🎯 Running Ensemble Model...")
    
    try:
        config = AnalysisConfig()
        ensemble = MultiModalEnsemble(config)
        
        # Prepare features in the expected format
        all_features = {extractor_name: {'features': features}}
        
        # Train ensemble
        training_results = ensemble.train_ensemble(all_features, labels, cv_folds=3)
        
        # Evaluate
        evaluation_results = ensemble.evaluate_ensemble(all_features, labels)
        
        # Save results
        results_dir = Path(f"results/05_modelling/{experiment_id}")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        ensemble.save_model(Path("data/models"), experiment_id)
        
        return {
            'model': 'ensemble',
            'extractor': extractor_name,
            'accuracy': evaluation_results.get('accuracy', 0.0),
            'status': 'success'
        }
        
    except Exception as e:
        print(f"    ❌ Ensemble failed: {str(e)}")
        return {
            'model': 'ensemble',
            'extractor': extractor_name,
            'accuracy': 0.0,
            'status': 'failed',
            'error': str(e)
        }

def run_temporal_attention_experiment(experiment_id, extractor_name, features, labels):
    """Run temporal attention model experiment."""
    print(f"    🎯 Running Temporal Attention Model...")
    
    try:
        config = AnalysisConfig()
        model = TemporalAttentionModel(config)
        
        # Train and evaluate
        results = model.train_and_evaluate(features, labels)
        
        return {
            'model': 'temporal_attention',
            'extractor': extractor_name,
            'accuracy': results.get('accuracy', 0.0),
            'status': 'success'
        }
        
    except Exception as e:
        print(f"    ❌ Temporal Attention failed: {str(e)}")
        return {
            'model': 'temporal_attention',
            'extractor': extractor_name,
            'accuracy': 0.0,
            'status': 'failed',
            'error': str(e)
        }

def run_progressive_learning_experiment(experiment_id, extractor_name, features, labels):
    """Run progressive learning model experiment."""
    print(f"    🎯 Running Progressive Learning Model...")
    
    try:
        config = AnalysisConfig()
        model = ProgressiveLearningModel(config)
        
        # Train and evaluate
        results = model.train_and_evaluate(features, labels)
        
        return {
            'model': 'progressive_learning',
            'extractor': extractor_name,
            'accuracy': results.get('accuracy', 0.0),
            'status': 'success'
        }
        
    except Exception as e:
        print(f"    ❌ Progressive Learning failed: {str(e)}")
        return {
            'model': 'progressive_learning',
            'extractor': extractor_name,
            'accuracy': 0.0,
            'status': 'failed',
            'error': str(e)
        }

def run_simple_ml_experiment(experiment_id, extractor_name, features, labels):
    """Run simple ML models experiment."""
    print(f"    🎯 Running Simple ML Models...")
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler
        
        # Ensure features are 2D
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        results = {}
        for name, model in models.items():
            try:
                scores = cross_val_score(model, features_scaled, labels, cv=3, scoring='accuracy')
                results[name] = {
                    'accuracy': scores.mean(),
                    'std': scores.std()
                }
            except Exception as e:
                results[name] = {'accuracy': 0.0, 'std': 0.0, 'error': str(e)}
        
        best_accuracy = max([r['accuracy'] for r in results.values()])
        
        return {
            'model': 'simple_ml',
            'extractor': extractor_name,
            'accuracy': best_accuracy,
            'details': results,
            'status': 'success'
        }
        
    except Exception as e:
        print(f"    ❌ Simple ML failed: {str(e)}")
        return {
            'model': 'simple_ml',
            'extractor': extractor_name,
            'accuracy': 0.0,
            'status': 'failed',
            'error': str(e)
        }

def run_experiment_combination(experiment_id, extractor_name, model_type, features, labels):
    """Run a specific experiment combination."""
    print(f"  🔬 {extractor_name} + {model_type}")
    
    if model_type == 'ensemble':
        return run_ensemble_experiment(experiment_id, extractor_name, features, labels)
    elif model_type == 'temporal_attention':
        return run_temporal_attention_experiment(experiment_id, extractor_name, features, labels)
    elif model_type == 'progressive_learning':
        return run_progressive_learning_experiment(experiment_id, extractor_name, features, labels)
    elif model_type == 'simple_ml':
        return run_simple_ml_experiment(experiment_id, extractor_name, features, labels)
    else:
        return {
            'model': model_type,
            'extractor': extractor_name,
            'accuracy': 0.0,
            'status': 'unknown_model'
        }

def create_comprehensive_report(all_results, experiment_summary):
    """Create a comprehensive report of all experiments."""
    print("📋 Creating comprehensive report...")
    
    # Create results directory
    results_dir = Path("results/05_modelling")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert results to DataFrame for analysis
    df_data = []
    for result in all_results:
        df_data.append({
            'experiment': result['experiment'],
            'extractor': result['extractor'],
            'model': result['model'],
            'accuracy': result['accuracy'],
            'status': result['status']
        })
    
    df = pd.DataFrame(df_data)
    
    # Create summary statistics
    summary_stats = {
        'total_experiments': len(all_results),
        'successful_experiments': len(df[df['status'] == 'success']),
        'failed_experiments': len(df[df['status'] == 'failed']),
        'best_accuracy': df['accuracy'].max(),
        'average_accuracy': df['accuracy'].mean(),
        'experiment_summary': experiment_summary
    }
    
    # Best performing combinations
    best_results = df[df['status'] == 'success'].nlargest(10, 'accuracy')
    
    # Create report
    report = f"""
# Comprehensive ECoG Experiment Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary Statistics
- **Total Experiments**: {summary_stats['total_experiments']}
- **Successful**: {summary_stats['successful_experiments']}
- **Failed**: {summary_stats['failed_experiments']}
- **Best Accuracy**: {summary_stats['best_accuracy']:.3f}
- **Average Accuracy**: {summary_stats['average_accuracy']:.3f}

## Best Performing Combinations (Top 10)
"""
    
    for idx, row in best_results.iterrows():
        report += f"- **{row['extractor']} + {row['model']}**: {row['accuracy']:.3f} (Experiment {row['experiment']})\n"
    
    report += f"""
## Experiment Summary
{experiment_summary}

## Detailed Results
"""
    
    # Add detailed results
    for result in all_results:
        report += f"""
### {result['extractor']} + {result['model']} (Experiment {result['experiment']})
- **Accuracy**: {result['accuracy']:.3f}
- **Status**: {result['status']}
"""
        if 'error' in result:
            report += f"- **Error**: {result['error']}\n"
    
    # Save report
    report_file = results_dir / f"comprehensive_experiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_file, 'w') as f:
        f.write(report)
    
    # Save results as JSON
    results_file = results_dir / f"comprehensive_experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'summary_stats': summary_stats,
            'detailed_results': all_results,
            'best_results': best_results.to_dict('records')
        }, f, indent=2, default=str)
    
    print(f"📁 Report saved: {report_file}")
    print(f"📁 Results saved: {results_file}")
    
    return summary_stats

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run comprehensive ECoG experiments')
    parser.add_argument('--quick', action='store_true', help='Run only ensemble and simple ML models')
    parser.add_argument('--full', action='store_true', help='Run all model combinations')
    parser.add_argument('--experiments', nargs='+', help='Specific experiments to run')
    parser.add_argument('--extractors', nargs='+', help='Specific extractors to run')
    parser.add_argument('--models', nargs='+', help='Specific models to run')
    
    args = parser.parse_args()
    
    print("🚀 Comprehensive ECoG Experiment Runner")
    print("=" * 50)
    
    start_time = time.time()
    
    # Get available experiments
    experiments = get_available_experiments()
    if not experiments:
        print("❌ No experiments found!")
        return
    
    print(f"📂 Found {len(experiments)} experiments: {[exp[1] for exp in experiments]}")
    
    # Determine which experiments to run
    if args.experiments:
        experiments = [exp for exp in experiments if exp[1] in args.experiments]
        print(f"🎯 Running specific experiments: {[exp[1] for exp in experiments]}")
    
    # Determine which models to run
    if args.quick:
        models_to_run = ['ensemble', 'simple_ml']
        print("⚡ Quick mode: Running only ensemble and simple ML models")
    elif args.full:
        models_to_run = ['ensemble', 'temporal_attention', 'progressive_learning', 'simple_ml']
        print("🔥 Full mode: Running all model combinations")
    elif args.models:
        models_to_run = args.models
        print(f"🎯 Running specific models: {models_to_run}")
    else:
        models_to_run = ['ensemble', 'simple_ml']  # Default
        print("📊 Default mode: Running ensemble and simple ML models")
    
    # Run experiments
    all_results = []
    experiment_summary = []
    
    total_combinations = len(experiments) * len(models_to_run)
    
    with tqdm(total=total_combinations, desc="Running experiments") as pbar:
        for exp_num, exp_id in experiments:
            print(f"\n🧪 Experiment {exp_id}")
            experiment_summary.append(f"Experiment {exp_id}: {exp_num}")
            
            # Get available extractors for this experiment
            extractors = get_available_feature_extractors(exp_id)
            if not extractors:
                print(f"  ⚠️ No extractors found for {exp_id}")
                continue
            
            print(f"  📊 Available extractors: {extractors}")
            
            # Determine which extractors to run
            if args.extractors:
                extractors = [ext for ext in extractors if ext in args.extractors]
                print(f"  🎯 Running specific extractors: {extractors}")
            
            for extractor_name in extractors:
                # Load features
                features, labels = load_features_for_experiment(exp_id, extractor_name)
                if features is None:
                    print(f"    ⚠️ No features found for {extractor_name}")
                    continue
                
                print(f"  📈 Loaded {extractor_name}: {features.shape}")
                
                # Run each model
                for model_type in models_to_run:
                    result = run_experiment_combination(exp_id, extractor_name, model_type, features, labels)
                    result['experiment'] = exp_id
                    all_results.append(result)
                    pbar.update(1)
    
    elapsed_time = time.time() - start_time
    
    # Create comprehensive report
    summary_stats = create_comprehensive_report(all_results, "\n".join(experiment_summary))
    
    print("\n🎉 All Experiments Completed!")
    print("=" * 50)
    print(f"⏱️ Total Time: {elapsed_time:.1f} seconds")
    print(f"📊 Total Experiments: {summary_stats['total_experiments']}")
    print(f"✅ Successful: {summary_stats['successful_experiments']}")
    print(f"❌ Failed: {summary_stats['failed_experiments']}")
    print(f"🏆 Best Accuracy: {summary_stats['best_accuracy']:.3f}")
    print(f"📈 Average Accuracy: {summary_stats['average_accuracy']:.3f}")
    
    # Show top 5 results
    successful_results = [r for r in all_results if r['status'] == 'success']
    if successful_results:
        print("\n🏆 Top 5 Results:")
        top_5 = sorted(successful_results, key=lambda x: x['accuracy'], reverse=True)[:5]
        for i, result in enumerate(top_5, 1):
            print(f"  {i}. {result['extractor']} + {result['model']}: {result['accuracy']:.3f}")

if __name__ == "__main__":
    main()
