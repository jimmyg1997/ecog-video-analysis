#!/usr/bin/env python3
"""
Progressive Learning Model with Data Augmentation
IEEE-SMC-2025 ECoG Video Analysis Competition

This module implements a progressive learning approach that starts with
simple models and gradually increases complexity using data augmentation.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class DataAugmentation:
    """Data augmentation techniques for ECoG data."""
    
    @staticmethod
    def add_noise(data: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
        """Add Gaussian noise to the data."""
        noise = np.random.normal(0, noise_level, data.shape)
        return data + noise
    
    @staticmethod
    def time_shift(data: np.ndarray, max_shift: int = 10) -> np.ndarray:
        """Apply random time shifts to the data."""
        shifted_data = np.zeros_like(data)
        
        for i in range(data.shape[0]):
            shift = np.random.randint(-max_shift, max_shift + 1)
            if shift > 0:
                shifted_data[i] = np.roll(data[i], shift, axis=-1)
            elif shift < 0:
                shifted_data[i] = np.roll(data[i], shift, axis=-1)
            else:
                shifted_data[i] = data[i]
        
        return shifted_data
    
    @staticmethod
    def amplitude_scaling(data: np.ndarray, scale_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """Apply random amplitude scaling to the data."""
        scaled_data = np.zeros_like(data)
        
        for i in range(data.shape[0]):
            scale_factor = np.random.uniform(scale_range[0], scale_range[1])
            scaled_data[i] = data[i] * scale_factor
        
        return scaled_data
    
    @staticmethod
    def channel_dropout(data: np.ndarray, dropout_rate: float = 0.1) -> np.ndarray:
        """Randomly set some channels to zero."""
        masked_data = data.copy()
        
        for i in range(data.shape[0]):
            num_channels = data.shape[1]
            num_drop = int(num_channels * dropout_rate)
            channels_to_drop = np.random.choice(num_channels, num_drop, replace=False)
            masked_data[i, channels_to_drop, :] = 0
        
        return masked_data
    
    @staticmethod
    def frequency_shift(data: np.ndarray, max_shift: float = 0.1) -> np.ndarray:
        """Apply frequency domain shifts (simplified)."""
        # This is a simplified version - in practice, you'd use FFT
        return data * (1 + np.random.uniform(-max_shift, max_shift))

class ProgressiveLearningModel:
    """Progressive learning model with increasing complexity."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the progressive learning model."""
        self.config = config or {}
        
        # Progressive stages
        self.stages = [
            {
                'name': 'Simple Linear',
                'models': {
                    'logistic': LogisticRegression(random_state=42, max_iter=1000),
                    'svm_linear': SVC(kernel='linear', probability=True, random_state=42)
                },
                'augmentation': None,
                'complexity': 1
            },
            {
                'name': 'Tree-based',
                'models': {
                    'random_forest': RandomForestClassifier(n_estimators=50, random_state=42),
                    'gradient_boosting': GradientBoostingClassifier(n_estimators=50, random_state=42)
                },
                'augmentation': ['add_noise'],
                'complexity': 2
            },
            {
                'name': 'Non-linear',
                'models': {
                    'svm_rbf': SVC(kernel='rbf', probability=True, random_state=42),
                    'mlp': MLPClassifier(hidden_layer_sizes=(100,), random_state=42, max_iter=1000)
                },
                'augmentation': ['add_noise', 'amplitude_scaling'],
                'complexity': 3
            },
            {
                'name': 'Advanced',
                'models': {
                    'random_forest_advanced': RandomForestClassifier(n_estimators=200, random_state=42),
                    'mlp_advanced': MLPClassifier(hidden_layer_sizes=(200, 100), random_state=42, max_iter=1000)
                },
                'augmentation': ['add_noise', 'amplitude_scaling', 'time_shift'],
                'complexity': 4
            },
            {
                'name': 'Complex',
                'models': {
                    'random_forest_complex': RandomForestClassifier(n_estimators=500, random_state=42),
                    'mlp_complex': MLPClassifier(hidden_layer_sizes=(500, 200, 100), random_state=42, max_iter=1000)
                },
                'augmentation': ['add_noise', 'amplitude_scaling', 'time_shift', 'channel_dropout'],
                'complexity': 5
            }
        ]
        
        # Training results
        self.stage_results = {}
        self.best_models = {}
        self.scalers = {}
        
        # Final ensemble
        self.final_ensemble = None
        self.ensemble_weights = {}
    
    def prepare_features(self, all_features: Dict[str, Dict]) -> Dict[str, np.ndarray]:
        """Prepare features from all extractors."""
        prepared_features = {}
        
        for extractor_name, features in all_features.items():
            if extractor_name == 'eegnet' and 'cnn_input' in features:
                # Use EEGNet features as primary (they're already augmented)
                cnn_data = features['cnn_input']
                if len(cnn_data.shape) > 2:
                    prepared_features[extractor_name] = cnn_data.reshape(cnn_data.shape[0], -1)
                else:
                    prepared_features[extractor_name] = cnn_data
                    
            elif extractor_name == 'comprehensive' and 'gamma_power' in features:
                # Use gamma power as baseline
                prepared_features[extractor_name] = features['gamma_power']
                
            elif extractor_name == 'transformer' and 'transformer_input' in features:
                # Use transformer features for complex stages
                prepared_features[extractor_name] = features['transformer_input']
        
        return prepared_features
    
    def apply_augmentation(self, data: np.ndarray, augmentation_types: List[str], 
                          augmentation_factor: int = 2) -> np.ndarray:
        """Apply data augmentation techniques."""
        if not augmentation_types:
            return data
        
        augmented_data = [data]  # Start with original data
        
        for _ in range(augmentation_factor):
            aug_data = data.copy()
            
            for aug_type in augmentation_types:
                if aug_type == 'add_noise':
                    aug_data = DataAugmentation.add_noise(aug_data, noise_level=0.05)
                elif aug_type == 'amplitude_scaling':
                    aug_data = DataAugmentation.amplitude_scaling(aug_data, scale_range=(0.9, 1.1))
                elif aug_type == 'time_shift':
                    aug_data = DataAugmentation.time_shift(aug_data, max_shift=5)
                elif aug_type == 'channel_dropout':
                    aug_data = DataAugmentation.channel_dropout(aug_data, dropout_rate=0.05)
                elif aug_type == 'frequency_shift':
                    aug_data = DataAugmentation.frequency_shift(aug_data, max_shift=0.05)
            
            augmented_data.append(aug_data)
        
        return np.vstack(augmented_data)
    
    def train_stage(self, stage: Dict[str, Any], X: np.ndarray, y: np.ndarray, 
                   cv_folds: int = 5) -> Dict[str, Any]:
        """Train models for a specific stage."""
        stage_name = stage['name']
        print(f"   🔧 Training stage: {stage_name}")
        
        # Apply augmentation if specified
        if stage['augmentation']:
            print(f"     📊 Applying augmentation: {stage['augmentation']}")
            X_aug = self.apply_augmentation(X, stage['augmentation'])
            y_aug = np.tile(y, len(X_aug) // len(y))  # Repeat labels for augmented data
        else:
            X_aug, y_aug = X, y
        
        print(f"     📊 Original data: {X.shape}, Augmented data: {X_aug.shape}")
        
        # Scale features
        scaler = RobustScaler()  # More robust to outliers
        X_scaled = scaler.fit_transform(X_aug)
        self.scalers[stage_name] = scaler
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = {}
        trained_models = {}
        
        # Train each model in the stage
        for model_name, model in stage['models'].items():
            print(f"     🔧 Training {model_name}")
            
            # Cross-validation
            scores = cross_val_score(model, X_scaled, y_aug, cv=cv, scoring='accuracy')
            cv_scores[model_name] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores
            }
            
            # Train on full data
            model.fit(X_scaled, y_aug)
            trained_models[model_name] = model
            
            print(f"       📊 CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
        
        # Find best model in this stage
        best_model_name = max(cv_scores.keys(), key=lambda x: cv_scores[x]['mean'])
        best_model = trained_models[best_model_name]
        best_score = cv_scores[best_model_name]['mean']
        
        print(f"     🏆 Best model: {best_model_name} (CV: {best_score:.3f})")
        
        return {
            'models': trained_models,
            'cv_scores': cv_scores,
            'best_model': best_model,
            'best_model_name': best_model_name,
            'best_score': best_score,
            'scaler': scaler,
            'augmentation': stage['augmentation'],
            'complexity': stage['complexity']
        }
    
    def train_progressive(self, all_features: Dict[str, Dict], 
                         labels: np.ndarray, cv_folds: int = 5) -> Dict[str, Any]:
        """Train the progressive learning model."""
        print("🎯 Training Progressive Learning Model")
        print("=" * 50)
        
        # Prepare features
        prepared_features = self.prepare_features(all_features)
        
        if not prepared_features:
            raise ValueError("No valid features found for training")
        
        # Start with simplest features and progress
        feature_progression = ['comprehensive', 'eegnet', 'transformer']
        
        for stage_idx, stage in enumerate(self.stages):
            print(f"\n📊 Stage {stage_idx + 1}: {stage['name']}")
            print("-" * 30)
            
            # Select appropriate features for this stage
            if stage['complexity'] <= 2:
                # Use simple features for early stages
                feature_type = 'comprehensive'
            elif stage['complexity'] <= 3:
                # Use EEGNet features for middle stages
                feature_type = 'eegnet'
            else:
                # Use transformer features for complex stages
                feature_type = 'transformer'
            
            if feature_type not in prepared_features:
                print(f"   ⚠️ Skipping stage - {feature_type} features not available")
                continue
            
            X = prepared_features[feature_type]
            
            # Train stage
            stage_result = self.train_stage(stage, X, labels, cv_folds)
            self.stage_results[stage['name']] = stage_result
            
            # Store best model
            self.best_models[stage['name']] = {
                'model': stage_result['best_model'],
                'model_name': stage_result['best_model_name'],
                'score': stage_result['best_score'],
                'feature_type': feature_type,
                'scaler': stage_result['scaler']
            }
        
        # Create final ensemble
        self.create_final_ensemble()
        
        print("\n✅ Progressive learning completed!")
        return self.stage_results
    
    def create_final_ensemble(self):
        """Create final ensemble from best models of each stage."""
        print("🔧 Creating final ensemble")
        
        if not self.best_models:
            print("   ⚠️ No models available for ensemble")
            return
        
        # Calculate ensemble weights based on performance
        total_score = sum(model_info['score'] for model_info in self.best_models.values())
        
        for stage_name, model_info in self.best_models.items():
            weight = model_info['score'] / total_score if total_score > 0 else 1.0
            self.ensemble_weights[stage_name] = weight
        
        print(f"   📊 Ensemble weights: {self.ensemble_weights}")
    
    def predict_progressive(self, all_features: Dict[str, Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Make progressive predictions."""
        print("🔮 Making progressive predictions")
        
        # Prepare features
        prepared_features = self.prepare_features(all_features)
        
        if not self.best_models:
            raise ValueError("No trained models available")
        
        # Collect predictions from each stage
        predictions = []
        probabilities = []
        
        for stage_name, model_info in self.best_models.items():
            feature_type = model_info['feature_type']
            
            if feature_type in prepared_features:
                X = prepared_features[feature_type]
                X_scaled = model_info['scaler'].transform(X)
                
                model = model_info['model']
                
                # Predictions
                pred = model.predict(X_scaled)
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(X_scaled)
                else:
                    # For models without predict_proba, create dummy probabilities
                    prob = np.zeros((len(pred), 2))
                    prob[np.arange(len(pred)), pred] = 1.0
                
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
            stage_name = list(self.best_models.keys())[i]
            weight = self.ensemble_weights.get(stage_name, 1.0)
            weighted_probs += weight * prob
        
        # Final prediction
        final_pred = np.argmax(weighted_probs, axis=1)
        
        return final_pred, weighted_probs
    
    def evaluate_progressive(self, all_features: Dict[str, Dict], 
                           labels: np.ndarray) -> Dict[str, Any]:
        """Evaluate progressive learning performance."""
        print("📊 Evaluating progressive learning performance")
        
        predictions, probabilities = self.predict_progressive(all_features)
        
        # Metrics
        accuracy = accuracy_score(labels, predictions)
        report = classification_report(labels, predictions, output_dict=True)
        cm = confusion_matrix(labels, predictions)
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': predictions,
            'probabilities': probabilities,
            'stage_results': self.stage_results,
            'ensemble_weights': self.ensemble_weights
        }
        
        print(f"   📊 Progressive Learning Accuracy: {accuracy:.3f}")
        return results
    
    def save_progressive_model(self, save_path: Path):
        """Save the progressive learning model."""
        print(f"💾 Saving progressive learning model to {save_path}")
        
        model_data = {
            'stage_results': self.stage_results,
            'best_models': self.best_models,
            'ensemble_weights': self.ensemble_weights,
            'scalers': self.scalers,
            'config': self.config
        }
        
        joblib.dump(model_data, save_path / 'progressive_learning_model.pkl')
        print("✅ Progressive learning model saved!")
    
    def load_progressive_model(self, load_path: Path):
        """Load a trained progressive learning model."""
        print(f"📂 Loading progressive learning model from {load_path}")
        
        model_data = joblib.load(load_path / 'progressive_learning_model.pkl')
        self.stage_results = model_data['stage_results']
        self.best_models = model_data['best_models']
        self.ensemble_weights = model_data['ensemble_weights']
        self.scalers = model_data['scalers']
        self.config = model_data['config']
        
        print("✅ Progressive learning model loaded!")
    
    def get_learning_curve(self) -> Dict[str, List[float]]:
        """Get learning curve showing performance progression."""
        learning_curve = {
            'stages': [],
            'scores': [],
            'complexities': []
        }
        
        for stage_name, result in self.stage_results.items():
            learning_curve['stages'].append(stage_name)
            learning_curve['scores'].append(result['best_score'])
            learning_curve['complexities'].append(result['complexity'])
        
        return learning_curve
    
    def get_summary_report(self) -> str:
        """Generate a summary report of the progressive learning model."""
        report = []
        report.append("🎯 Progressive Learning Model Summary")
        report.append("=" * 50)
        
        # Stages completed
        report.append(f"📊 Stages Completed: {len(self.stage_results)}")
        for stage_name, result in self.stage_results.items():
            report.append(f"   • {stage_name}: {result['best_model_name']} (CV: {result['best_score']:.3f})")
        
        # Ensemble weights
        report.append(f"\n⚖️ Ensemble Weights:")
        for stage_name, weight in self.ensemble_weights.items():
            report.append(f"   • {stage_name}: {weight:.3f}")
        
        # Learning progression
        learning_curve = self.get_learning_curve()
        if learning_curve['scores']:
            best_score = max(learning_curve['scores'])
            report.append(f"\n🏆 Best Stage Performance: {best_score:.3f}")
            
            # Show improvement
            if len(learning_curve['scores']) > 1:
                improvement = learning_curve['scores'][-1] - learning_curve['scores'][0]
                report.append(f"📈 Total Improvement: {improvement:.3f}")
        
        return "\n".join(report)
