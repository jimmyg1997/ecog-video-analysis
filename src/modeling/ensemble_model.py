#!/usr/bin/env python3
"""
Multi-Modal Ensemble Model for ECoG Classification
IEEE-SMC-2025 ECoG Video Analysis Competition

This module implements a sophisticated ensemble approach that combines
all 4 feature extractors with cross-validation and weighted voting.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class MultiModalEnsemble:
    """Multi-modal ensemble classifier combining all feature types."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the ensemble model."""
        self.config = config or {}
        
        # Model configurations
        self.models = {
            'template_correlation': {
                'rf': RandomForestClassifier(n_estimators=100, random_state=42),
                'svm': SVC(kernel='rbf', probability=True, random_state=42),
                'gb': GradientBoostingClassifier(n_estimators=100, random_state=42)
            },
            'csp_lda': {
                'rf': RandomForestClassifier(n_estimators=100, random_state=42),
                'svm': SVC(kernel='linear', probability=True, random_state=42),
                'lr': LogisticRegression(random_state=42, max_iter=1000)
            },
            'eegnet': {
                'rf': RandomForestClassifier(n_estimators=100, random_state=42),
                'svm': SVC(kernel='rbf', probability=True, random_state=42),
                'gb': GradientBoostingClassifier(n_estimators=100, random_state=42)
            },
            'transformer': {
                'rf': RandomForestClassifier(n_estimators=100, random_state=42),
                'svm': SVC(kernel='rbf', probability=True, random_state=42),
                'lr': LogisticRegression(random_state=42, max_iter=1000)
            }
        }
        
        # Scalers for each feature type
        self.scalers = {}
        
        # Ensemble weights (will be learned)
        self.ensemble_weights = {}
        
        # Cross-validation results
        self.cv_results = {}
        
    def prepare_features(self, all_features: Dict[str, Dict]) -> Dict[str, np.ndarray]:
        """Prepare features from all extractors for training."""
        prepared_features = {}
        
        for extractor_name, features in all_features.items():
            if extractor_name == 'template_correlation':
                # Template correlation features
                if 'template_correlations' in features:
                    prepared_features[extractor_name] = features['template_correlations']
                elif 'loocv_features' in features:
                    prepared_features[extractor_name] = features['loocv_features']
                    
            elif extractor_name == 'csp_lda':
                # CSP+LDA features
                if 'csp_features' in features:
                    prepared_features[extractor_name] = features['csp_features']
                elif 'spatial_features' in features:
                    prepared_features[extractor_name] = features['spatial_features']
                    
            elif extractor_name == 'eegnet':
                # EEGNet features - handle augmented data
                if 'cnn_input' in features:
                    cnn_data = features['cnn_input']
                    if len(cnn_data.shape) > 2:
                        cnn_data = cnn_data.reshape(cnn_data.shape[0], -1)
                    
                    # If data is augmented (more samples than expected), take first half
                    if cnn_data.shape[0] > 252:  # Assuming 252 is the original number of trials
                        print(f"   ⚠️ EEGNet data is augmented ({cnn_data.shape[0]} samples), using first 252 samples")
                        cnn_data = cnn_data[:252, :]
                    
                    prepared_features[extractor_name] = cnn_data
                        
            elif extractor_name == 'transformer':
                # Transformer features
                if 'transformer_input' in features:
                    prepared_features[extractor_name] = features['transformer_input']
                elif 'attention_features' in features:
                    prepared_features[extractor_name] = features['attention_features']
                    
            elif extractor_name == 'comprehensive':
                # Original comprehensive features
                if 'gamma_power' in features:
                    prepared_features[extractor_name] = features['gamma_power']
        
        return prepared_features
    
    def train_individual_models(self, X: np.ndarray, y: np.ndarray, 
                              feature_type: str, cv_folds: int = 5) -> Dict[str, Any]:
        """Train individual models for a specific feature type."""
        print(f"   🔧 Training models for {feature_type}")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers[feature_type] = scaler
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = {}
        
        # Train each model type
        trained_models = {}
        for model_name, model in self.models[feature_type].items():
            # Cross-validation
            scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
            cv_scores[model_name] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores
            }
            
            # Train on full data
            model.fit(X_scaled, y)
            trained_models[model_name] = model
            
            print(f"     📊 {model_name}: CV Accuracy = {scores.mean():.3f} ± {scores.std():.3f}")
        
        return {
            'models': trained_models,
            'cv_scores': cv_scores,
            'scaler': scaler
        }
    
    def compute_ensemble_weights(self, cv_results: Dict[str, Dict]) -> Dict[str, float]:
        """Compute ensemble weights based on cross-validation performance."""
        print("   🔧 Computing ensemble weights")
        
        weights = {}
        total_performance = 0
        
        for feature_type, results in cv_results.items():
            # Use best model's performance for weighting
            best_score = 0
            for model_name, scores in results['cv_scores'].items():
                if scores['mean'] > best_score:
                    best_score = scores['mean']
            
            weights[feature_type] = best_score
            total_performance += best_score
        
        # Normalize weights
        if total_performance > 0:
            for feature_type in weights:
                weights[feature_type] /= total_performance
        
        print(f"   📊 Ensemble weights: {weights}")
        return weights
    
    def train_ensemble(self, all_features: Dict[str, Dict], 
                      labels: np.ndarray, cv_folds: int = 5) -> Dict[str, Any]:
        """Train the complete ensemble model."""
        print("🎯 Training Multi-Modal Ensemble Model")
        print("=" * 50)
        
        # Prepare features
        prepared_features = self.prepare_features(all_features)
        
        if not prepared_features:
            raise ValueError("No valid features found for training")
        
        # Train individual models
        self.cv_results = {}
        for feature_type, X in prepared_features.items():
            if X.size > 0:
                self.cv_results[feature_type] = self.train_individual_models(
                    X, labels, feature_type, cv_folds
                )
        
        # Compute ensemble weights
        self.ensemble_weights = self.compute_ensemble_weights(self.cv_results)
        
        # Store prepared features for prediction
        self.prepared_features = prepared_features
        
        print("✅ Ensemble training completed!")
        return {
            'cv_results': self.cv_results,
            'ensemble_weights': self.ensemble_weights,
            'feature_types': list(prepared_features.keys())
        }
    
    def predict_ensemble(self, all_features: Dict[str, Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Make ensemble predictions."""
        print("🔮 Making ensemble predictions")
        
        # Prepare features
        prepared_features = self.prepare_features(all_features)
        
        # Collect predictions from each feature type
        predictions = []
        probabilities = []
        
        for feature_type, X in prepared_features.items():
            if feature_type in self.cv_results and X.size > 0:
                # Scale features
                X_scaled = self.scalers[feature_type].transform(X)
                
                # Get predictions from best model
                best_model_name = max(
                    self.cv_results[feature_type]['cv_scores'].keys(),
                    key=lambda x: self.cv_results[feature_type]['cv_scores'][x]['mean']
                )
                best_model = self.cv_results[feature_type]['models'][best_model_name]
                
                # Predictions
                pred = best_model.predict(X_scaled)
                prob = best_model.predict_proba(X_scaled)
                
                predictions.append(pred)
                probabilities.append(prob)
        
        if not predictions:
            raise ValueError("No valid predictions generated")
        
        # Weighted ensemble prediction
        if len(predictions) == 1:
            return predictions[0], probabilities[0]
        
        # Weighted voting
        weighted_probs = np.zeros_like(probabilities[0])
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            feature_type = list(prepared_features.keys())[i]
            weight = self.ensemble_weights.get(feature_type, 1.0)
            weighted_probs += weight * prob
        
        # Final prediction
        final_pred = np.argmax(weighted_probs, axis=1)
        
        return final_pred, weighted_probs
    
    def evaluate_ensemble(self, all_features: Dict[str, Dict], 
                         labels: np.ndarray) -> Dict[str, Any]:
        """Evaluate ensemble performance."""
        print("📊 Evaluating ensemble performance")
        
        predictions, probabilities = self.predict_ensemble(all_features)
        
        # Metrics
        accuracy = accuracy_score(labels, predictions)
        report = classification_report(labels, predictions, output_dict=True)
        cm = confusion_matrix(labels, predictions)
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': predictions,
            'probabilities': probabilities
        }
        
        print(f"   📊 Ensemble Accuracy: {accuracy:.3f}")
        return results
    
    def save_ensemble(self, save_path: Path):
        """Save the trained ensemble model."""
        print(f"💾 Saving ensemble model to {save_path}")
        
        ensemble_data = {
            'models': self.cv_results,
            'weights': self.ensemble_weights,
            'scalers': self.scalers,
            'config': self.config
        }
        
        joblib.dump(ensemble_data, save_path / 'ensemble_model.pkl')
        print("✅ Ensemble model saved!")
    
    def load_ensemble(self, load_path: Path):
        """Load a trained ensemble model."""
        print(f"📂 Loading ensemble model from {load_path}")
        
        ensemble_data = joblib.load(load_path / 'ensemble_model.pkl')
        self.cv_results = ensemble_data['models']
        self.ensemble_weights = ensemble_data['weights']
        self.scalers = ensemble_data['scalers']
        self.config = ensemble_data['config']
        
        print("✅ Ensemble model loaded!")
    
    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Get feature importance from tree-based models."""
        importance_results = {}
        
        for feature_type, results in self.cv_results.items():
            importance_results[feature_type] = {}
            
            for model_name, model in results['models'].items():
                if hasattr(model, 'feature_importances_'):
                    importance_results[feature_type][model_name] = model.feature_importances_
        
        return importance_results
    
    def get_summary_report(self) -> str:
        """Generate a summary report of the ensemble model."""
        report = []
        report.append("🎯 Multi-Modal Ensemble Model Summary")
        report.append("=" * 50)
        
        # Feature types
        report.append(f"📊 Feature Types: {len(self.cv_results)}")
        for feature_type in self.cv_results.keys():
            report.append(f"   • {feature_type}")
        
        # Ensemble weights
        report.append(f"\n⚖️ Ensemble Weights:")
        for feature_type, weight in self.ensemble_weights.items():
            report.append(f"   • {feature_type}: {weight:.3f}")
        
        # Best models per feature type
        report.append(f"\n🏆 Best Models per Feature Type:")
        for feature_type, results in self.cv_results.items():
            best_model = max(
                results['cv_scores'].keys(),
                key=lambda x: results['cv_scores'][x]['mean']
            )
            best_score = results['cv_scores'][best_model]['mean']
            report.append(f"   • {feature_type}: {best_model} (CV: {best_score:.3f})")
        
        return "\n".join(report)
