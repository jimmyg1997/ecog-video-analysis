#!/usr/bin/env python3
"""
Comprehensive ECoG Preprocessing and Feature Extraction Pipeline
IEEE-SMC-2025 ECoG Video Analysis Competition

This pipeline implements the complete preprocessing and feature extraction workflow:
1. Preprocessing: Bandpass filter, notch filter, artifact rejection, stimulus alignment, 
   trial segmentation, and normalization
2. Feature Extraction: Broadband gamma power and canonical band powers
3. Data saving and analysis reporting
"""

import sys
import os
sys.path.append('src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from utils.data_loader import DataLoader
from utils.config import AnalysisConfig
from utils.progress_tracker import track_progress
from preprocessing.ecog_preprocessor import ECoGPreprocessor

class ComprehensivePipeline:
    """Comprehensive ECoG preprocessing and feature extraction pipeline."""
    
    def __init__(self, config: AnalysisConfig = None):
        """Initialize the pipeline with configuration."""
        self.config = config or AnalysisConfig()
        self.data_loader = DataLoader()
        self.preprocessor = ECoGPreprocessor(self.config)
        
        # Create output directories
        self.preprocessed_dir = Path('data/preprocessed')
        self.features_dir = Path('data/features')
        self.results_dir = Path('results/pipeline')
        
        for dir_path in [self.preprocessed_dir, self.features_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize data storage
        self.raw_data = None
        self.preprocessed_data = None
        self.features = {}
        self.analysis_results = {}
        
    def load_raw_data(self):
        """Load raw ECoG data."""
        print("🔄 Loading raw ECoG data...")
        self.raw_data = self.data_loader.load_raw_data()
        print(f"✅ Loaded raw data: {self.raw_data['ecog_data'].shape}")
        return self.raw_data
    
    def apply_preprocessing(self):
        """Apply complete preprocessing pipeline."""
        print("\n🔧 APPLYING PREPROCESSING PIPELINE")
        print("=" * 50)
        
        ecog_data = self.raw_data['ecog_data']
        photodiode = self.raw_data['photodiode']
        stimcode = self.raw_data['stimcode']
        sampling_rate = self.raw_data['sampling_rate']
        
        with track_progress("Preprocessing pipeline", 6) as pbar:
            # Step 1: Bandpass filter (0.5-150 Hz)
            pbar.update(1, "Bandpass filtering (0.5-150 Hz)")
            print("🔧 Step 1: Bandpass filtering (0.5-150 Hz)")
            filtered_data = self._apply_bandpass_filter(ecog_data, sampling_rate, 0.5, 150.0)
            
            # Step 2: Notch filter (50/60 Hz)
            pbar.update(1, "Notch filtering (50/60 Hz)")
            print("🔧 Step 2: Notch filtering (50/60 Hz)")
            filtered_data = self._apply_notch_filter(filtered_data, sampling_rate, [50, 60])
            
            # Step 3: Artifact rejection
            pbar.update(1, "Artifact rejection")
            print("🔧 Step 3: Artifact rejection")
            good_channels, bad_channels = self._detect_artifacts(filtered_data)
            print(f"   📊 Good channels: {len(good_channels)}/{ecog_data.shape[0]}")
            print(f"   🚫 Bad channels: {bad_channels}")
            
            # Step 4: Stimulus alignment
            pbar.update(1, "Stimulus alignment")
            print("🔧 Step 4: Stimulus alignment via photodiode")
            trial_onsets = self._detect_trial_onsets(photodiode, stimcode, sampling_rate)
            print(f"   📊 Detected {len(trial_onsets)} trial onsets")
            
            # Step 5: Trial segmentation
            pbar.update(1, "Trial segmentation")
            print("🔧 Step 5: Trial segmentation (100-400 ms post-onset)")
            epochs, time_vector = self._extract_trials(
                filtered_data[good_channels, :], trial_onsets, sampling_rate
            )
            print(f"   📊 Extracted {epochs.shape[0]} trials, {epochs.shape[1]} channels")
            
            # Step 6: Normalization (z-score per trial and channel)
            pbar.update(1, "Normalization")
            print("🔧 Step 6: Z-score normalization per trial and channel")
            normalized_epochs = self._normalize_epochs(epochs, time_vector, sampling_rate)
            print(f"   📊 Normalized epochs shape: {normalized_epochs.shape}")
        
        # Store preprocessed data
        self.preprocessed_data = {
            'epochs': normalized_epochs,
            'time_vector': time_vector,
            'trial_onsets': trial_onsets,
            'good_channels': good_channels,
            'bad_channels': bad_channels,
            'sampling_rate': sampling_rate,
            'stimcode': stimcode[trial_onsets] if len(trial_onsets) > 0 else np.array([]),
            'preprocessing_params': {
                'bandpass_low': 0.5,
                'bandpass_high': 150.0,
                'notch_freqs': [50, 60],
                'trial_window': (100, 400),  # ms
                'baseline_window': (-300, 0),  # ms
                'n_trials': len(trial_onsets),
                'n_channels': len(good_channels)
            }
        }
        
        print("✅ Preprocessing completed successfully!")
        return self.preprocessed_data
    
    def extract_features(self):
        """Extract features from preprocessed data."""
        print("\n🔧 EXTRACTING FEATURES")
        print("=" * 50)
        
        epochs = self.preprocessed_data['epochs']
        time_vector = self.preprocessed_data['time_vector']
        sampling_rate = self.preprocessed_data['sampling_rate']
        
        with track_progress("Feature extraction", 2) as pbar:
            # Feature A: Broadband Gamma Power (110-140 Hz)
            pbar.update(1, "Broadband gamma power (110-140 Hz)")
            print("🔧 Feature A: Broadband gamma power (110-140 Hz)")
            gamma_features = self._extract_gamma_power(epochs, time_vector, sampling_rate, 110, 140)
            self.features['gamma_power'] = gamma_features
            print(f"   📊 Gamma power features shape: {gamma_features.shape}")
            
            # Feature B: Canonical Band Powers
            pbar.update(1, "Canonical band powers")
            print("🔧 Feature B: Canonical band powers")
            band_features = self._extract_canonical_bands(epochs, time_vector, sampling_rate)
            self.features.update(band_features)
            print(f"   📊 Band power features: {list(band_features.keys())}")
        
        print("✅ Feature extraction completed successfully!")
        return self.features
    
    def save_data(self):
        """Save preprocessed data and features."""
        print("\n💾 SAVING DATA")
        print("=" * 50)
        
        # Save preprocessed data
        print("💾 Saving preprocessed data...")
        np.save(self.preprocessed_dir / 'epochs.npy', self.preprocessed_data['epochs'])
        np.save(self.preprocessed_dir / 'time_vector.npy', self.preprocessed_data['time_vector'])
        np.save(self.preprocessed_dir / 'trial_onsets.npy', self.preprocessed_data['trial_onsets'])
        np.save(self.preprocessed_dir / 'good_channels.npy', self.preprocessed_data['good_channels'])
        np.save(self.preprocessed_dir / 'bad_channels.npy', self.preprocessed_data['bad_channels'])
        np.save(self.preprocessed_dir / 'stimcode.npy', self.preprocessed_data['stimcode'])
        
        # Save preprocessing parameters
        with open(self.preprocessed_dir / 'preprocessing_params.json', 'w') as f:
            json.dump(self.preprocessed_data['preprocessing_params'], f, indent=2)
        
        # Save features
        print("💾 Saving extracted features...")
        for feature_name, feature_data in self.features.items():
            np.save(self.features_dir / f'{feature_name}.npy', feature_data)
        
        # Save feature metadata
        feature_metadata = {
            'gamma_power': {
                'description': 'Broadband gamma power (110-140 Hz)',
                'method': 'Bandpass filter + Hilbert transform + log-variance',
                'shape': list(self.features['gamma_power'].shape)
            },
            'theta_power': {
                'description': 'Theta band power (4-8 Hz)',
                'method': 'FFT power spectral density',
                'shape': list(self.features['theta_power'].shape)
            },
            'alpha_power': {
                'description': 'Alpha band power (8-13 Hz)',
                'method': 'FFT power spectral density',
                'shape': list(self.features['alpha_power'].shape)
            },
            'beta_power': {
                'description': 'Beta band power (13-30 Hz)',
                'method': 'FFT power spectral density',
                'shape': list(self.features['beta_power'].shape)
            },
            'gamma_power_30_80_power': {
                'description': 'Gamma band power (30-80 Hz)',
                'method': 'FFT power spectral density',
                'shape': list(self.features['gamma_power_30_80_power'].shape)
            }
        }
        
        with open(self.features_dir / 'feature_metadata.json', 'w') as f:
            json.dump(feature_metadata, f, indent=2)
        
        print("✅ Data saved successfully!")
    
    def generate_analysis_report(self):
        """Generate comprehensive analysis report."""
        print("\n📊 GENERATING ANALYSIS REPORT")
        print("=" * 50)
        
        # Create analysis results
        self.analysis_results = {
            'preprocessing_summary': self._analyze_preprocessing(),
            'feature_analysis': self._analyze_features(),
            'data_quality': self._assess_data_quality()
        }
        
        # Save analysis results
        with open(self.results_dir / 'analysis_results.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = self._convert_for_json(self.analysis_results)
            json.dump(json_results, f, indent=2)
        
        # Generate visualization report
        self._create_visualization_report()
        
        print("✅ Analysis report generated successfully!")
        return self.analysis_results
    
    def run_complete_pipeline(self):
        """Run the complete preprocessing and feature extraction pipeline."""
        print("🚀 STARTING COMPREHENSIVE ECoG PIPELINE")
        print("=" * 60)
        print("IEEE-SMC-2025 ECoG Video Analysis Competition")
        print("=" * 60)
        
        try:
            # Step 1: Load raw data
            self.load_raw_data()
            
            # Step 2: Apply preprocessing
            self.apply_preprocessing()
            
            # Step 3: Extract features
            self.extract_features()
            
            # Step 4: Save data
            self.save_data()
            
            # Step 5: Generate analysis report
            self.generate_analysis_report()
            
            print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print("📁 Preprocessed data saved to: data/preprocessed/")
            print("📁 Features saved to: data/features/")
            print("📁 Analysis results saved to: results/pipeline/")
            print("=" * 60)
            
        except Exception as e:
            print(f"\n❌ Pipeline failed with error: {str(e)}")
            raise
    
    # Helper methods for preprocessing
    def _apply_bandpass_filter(self, data, fs, low_freq, high_freq):
        """Apply bandpass filter."""
        nyquist = fs / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # Ensure frequencies are within valid range
        low = max(0.01, min(low, 0.99))
        high = max(low + 0.01, min(high, 0.99))
        
        b, a = signal.butter(4, [low, high], btype='band')
        return signal.filtfilt(b, a, data, axis=1)
    
    def _apply_notch_filter(self, data, fs, notch_freqs):
        """Apply notch filters."""
        filtered_data = data.copy()
        for freq in notch_freqs:
            if freq < fs / 2:  # Only apply if frequency is below Nyquist
                nyquist = fs / 2
                low = (freq - 1) / nyquist
                high = (freq + 1) / nyquist
                low = max(0.01, min(low, 0.99))
                high = max(low + 0.01, min(high, 0.99))
                
                b, a = signal.butter(4, [low, high], btype='bandstop')
                filtered_data = signal.filtfilt(b, a, filtered_data, axis=1)
        return filtered_data
    
    def _detect_artifacts(self, data):
        """Detect artifacts using variance threshold."""
        channel_vars = np.var(data, axis=1)
        var_threshold = np.mean(channel_vars) + 3 * np.std(channel_vars)
        bad_channels = np.where(channel_vars > var_threshold)[0]
        good_channels = np.where(channel_vars <= var_threshold)[0]
        return good_channels, bad_channels
    
    def _detect_trial_onsets(self, photodiode, stimcode, fs):
        """Detect trial onsets via photodiode signal."""
        # Find photodiode signal changes (stimulus onsets)
        photodiode_diff = np.diff(photodiode)
        threshold = np.std(photodiode_diff) * 2
        onset_indices = np.where(np.abs(photodiode_diff) > threshold)[0]
        
        # Filter out onsets that are too close together (minimum 500ms apart)
        min_interval = int(0.5 * fs)
        filtered_onsets = [onset_indices[0]] if len(onset_indices) > 0 else []
        
        for onset in onset_indices[1:]:
            if onset - filtered_onsets[-1] > min_interval:
                filtered_onsets.append(onset)
        
        return np.array(filtered_onsets)
    
    def _extract_trials(self, data, trial_onsets, fs):
        """Extract trial epochs (100-400 ms post-onset)."""
        if len(trial_onsets) == 0:
            return np.array([]), np.array([])
        
        # Convert time windows to samples
        pre_samples = int(0.3 * fs)  # 300ms before onset
        post_samples = int(0.4 * fs)  # 400ms after onset
        epoch_length = pre_samples + post_samples
        
        epochs = []
        valid_onsets = []
        
        for onset in trial_onsets:
            start_idx = onset - pre_samples
            end_idx = onset + post_samples
            
            # Check if epoch is within data bounds
            if start_idx >= 0 and end_idx < data.shape[1]:
                epoch = data[:, start_idx:end_idx]
                epochs.append(epoch)
                valid_onsets.append(onset)
        
        if len(epochs) == 0:
            return np.array([]), np.array([])
        
        epochs = np.array(epochs)
        time_vector = np.arange(-pre_samples, post_samples) / fs * 1000  # Convert to ms
        
        return epochs, time_vector
    
    def _normalize_epochs(self, epochs, time_vector, fs):
        """Z-score normalize epochs using baseline period (-300 to 0 ms)."""
        if epochs.size == 0:
            return epochs
        
        # Find baseline period indices
        baseline_mask = (time_vector >= -300) & (time_vector <= 0)
        baseline_data = epochs[:, :, baseline_mask]
        
        # Compute mean and std for each trial and channel
        baseline_mean = np.mean(baseline_data, axis=2, keepdims=True)
        baseline_std = np.std(baseline_data, axis=2, keepdims=True)
        
        # Avoid division by zero
        baseline_std = np.where(baseline_std == 0, 1, baseline_std)
        
        # Z-score normalize
        normalized_epochs = (epochs - baseline_mean) / baseline_std
        
        return normalized_epochs
    
    # Helper methods for feature extraction
    def _extract_gamma_power(self, epochs, time_vector, fs, low_freq, high_freq):
        """Extract broadband gamma power (110-140 Hz)."""
        if epochs.size == 0:
            return np.array([])
        
        # Find post-stimulus period (100-400 ms)
        post_stim_mask = (time_vector >= 100) & (time_vector <= 400)
        post_stim_data = epochs[:, :, post_stim_mask]
        
        # Apply bandpass filter
        filtered_data = self._apply_bandpass_filter(
            post_stim_data, fs, low_freq, high_freq
        )
        
        # Compute envelope using Hilbert transform
        envelope = np.abs(signal.hilbert(filtered_data, axis=2))
        
        # Compute log-variance
        log_variance = np.log(np.var(envelope, axis=2) + 1e-8)
        
        return log_variance
    
    def _extract_canonical_bands(self, epochs, time_vector, fs):
        """Extract canonical band powers."""
        if epochs.size == 0:
            return {}
        
        # Find post-stimulus period (100-400 ms)
        post_stim_mask = (time_vector >= 100) & (time_vector <= 400)
        post_stim_data = epochs[:, :, post_stim_mask]
        
        # Define frequency bands
        bands = {
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma_power_30_80': (30, 80)
        }
        
        band_features = {}
        
        for band_name, (low_freq, high_freq) in bands.items():
            # Apply bandpass filter
            filtered_data = self._apply_bandpass_filter(
                post_stim_data, fs, low_freq, high_freq
            )
            
            # Compute power using FFT
            freqs, psd = signal.welch(
                filtered_data, fs=fs, nperseg=min(256, filtered_data.shape[2]),
                axis=2
            )
            
            # Find frequency band indices
            band_mask = (freqs >= low_freq) & (freqs <= high_freq)
            
            # Compute mean power in band
            band_power = np.mean(psd[:, :, band_mask], axis=2)
            
            # Log transform
            log_power = np.log(band_power + 1e-8)
            
            band_features[f'{band_name}_power'] = log_power
        
        return band_features
    
    # Analysis methods
    def _analyze_preprocessing(self):
        """Analyze preprocessing results."""
        epochs = self.preprocessed_data['epochs']
        good_channels = self.preprocessed_data['good_channels']
        bad_channels = self.preprocessed_data['bad_channels']
        
        return {
            'n_trials': epochs.shape[0] if epochs.size > 0 else 0,
            'n_channels_total': len(good_channels) + len(bad_channels),
            'n_good_channels': len(good_channels),
            'n_bad_channels': len(bad_channels),
            'channel_quality_rate': len(good_channels) / (len(good_channels) + len(bad_channels)) * 100,
            'epoch_shape': list(epochs.shape) if epochs.size > 0 else [0, 0, 0]
        }
    
    def _analyze_features(self):
        """Analyze extracted features."""
        analysis = {}
        
        for feature_name, feature_data in self.features.items():
            if feature_data.size > 0:
                analysis[feature_name] = {
                    'shape': list(feature_data.shape),
                    'mean': float(np.mean(feature_data)),
                    'std': float(np.std(feature_data)),
                    'min': float(np.min(feature_data)),
                    'max': float(np.max(feature_data))
                }
        
        return analysis
    
    def _assess_data_quality(self):
        """Assess overall data quality."""
        epochs = self.preprocessed_data['epochs']
        
        if epochs.size == 0:
            return {'quality_score': 0, 'issues': ['No valid trials detected']}
        
        # Compute quality metrics
        channel_means = np.mean(epochs, axis=(0, 2))
        channel_stds = np.std(epochs, axis=(0, 2))
        
        # Check for flat channels (low variance)
        flat_channels = np.sum(channel_stds < 0.1)
        
        # Check for high-amplitude artifacts
        max_amplitudes = np.max(np.abs(epochs), axis=(0, 2))
        artifact_channels = np.sum(max_amplitudes > 10)  # Z-score > 10
        
        quality_score = 100 - (flat_channels + artifact_channels) * 5
        quality_score = max(0, quality_score)
        
        issues = []
        if flat_channels > 0:
            issues.append(f'{flat_channels} channels with low variance')
        if artifact_channels > 0:
            issues.append(f'{artifact_channels} channels with high-amplitude artifacts')
        
        return {
            'quality_score': quality_score,
            'flat_channels': int(flat_channels),
            'artifact_channels': int(artifact_channels),
            'issues': issues
        }
    
    def _convert_for_json(self, obj):
        """Convert numpy arrays to lists for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj
    
    def _create_visualization_report(self):
        """Create visualization report."""
        print("📊 Creating visualization report...")
        
        # This would create comprehensive visualizations
        # For now, we'll just create a summary
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('ECoG Pipeline Analysis Summary', fontsize=16, fontweight='bold')
        
        # Plot 1: Channel quality
        ax1 = axes[0, 0]
        quality_data = self.analysis_results['preprocessing_summary']
        labels = ['Good Channels', 'Bad Channels']
        values = [quality_data['n_good_channels'], quality_data['n_bad_channels']]
        colors = ['lightgreen', 'lightcoral']
        ax1.pie(values, labels=labels, colors=colors, autopct='%1.1f%%')
        ax1.set_title('Channel Quality')
        
        # Plot 2: Feature distributions
        ax2 = axes[0, 1]
        if 'gamma_power' in self.features and self.features['gamma_power'].size > 0:
            gamma_data = self.features['gamma_power'].flatten()
            ax2.hist(gamma_data, bins=30, alpha=0.7, color='blue')
            ax2.set_title('Gamma Power Distribution')
            ax2.set_xlabel('Log-Variance')
            ax2.set_ylabel('Count')
        
        # Plot 3: Data quality score
        ax3 = axes[1, 0]
        quality_score = self.analysis_results['data_quality']['quality_score']
        ax3.bar(['Data Quality'], [quality_score], color='lightblue')
        ax3.set_ylim(0, 100)
        ax3.set_ylabel('Quality Score (%)')
        ax3.set_title('Overall Data Quality')
        
        # Plot 4: Pipeline summary
        ax4 = axes[1, 1]
        ax4.text(0.1, 0.8, f"Trials: {quality_data['n_trials']}", transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.6, f"Channels: {quality_data['n_good_channels']}", transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.4, f"Quality: {quality_score:.1f}%", transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.2, f"Features: {len(self.features)}", transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Pipeline Summary')
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'pipeline_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Visualization report saved!")


def main():
    """Main function to run the comprehensive pipeline."""
    print("🚀 Starting Comprehensive ECoG Pipeline")
    print("IEEE-SMC-2025 ECoG Video Analysis Competition")
    print("=" * 60)
    
    # Initialize pipeline
    config = AnalysisConfig()
    pipeline = ComprehensivePipeline(config)
    
    # Run complete pipeline
    pipeline.run_complete_pipeline()
    
    print("\n🎉 Pipeline execution completed successfully!")
    print("📁 Check the following directories for results:")
    print("   • data/preprocessed/ - Preprocessed epochs and metadata")
    print("   • data/features/ - Extracted features")
    print("   • results/pipeline/ - Analysis results and visualizations")


if __name__ == "__main__":
    main()
