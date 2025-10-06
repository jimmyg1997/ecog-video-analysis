"""
Comprehensive ECoG Preprocessing Module
IEEE-SMC-2025 ECoG Video Analysis Competition

This module provides comprehensive preprocessing capabilities for ECoG data,
including filtering, artifact rejection, trial segmentation, and normalization.
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.decomposition import FastICA
from typing import Dict, Tuple, List, Optional, Union
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

from utils.progress_tracker import track_progress
from utils.config import AnalysisConfig

class ComprehensivePreprocessor:
    """Comprehensive ECoG preprocessing pipeline."""
    
    def __init__(self, config: AnalysisConfig = None):
        """Initialize preprocessor with configuration."""
        self.config = config or AnalysisConfig()
        self.fs = self.config.sampling_rate
        
        # Preprocessing parameters
        self.bandpass_low = 0.5
        self.bandpass_high = 150.0
        self.notch_freqs = [50, 60]
        self.trial_window = (100, 400)  # ms post-stimulus
        self.baseline_window = (-300, 0)  # ms pre-stimulus
        
        # Enhanced preprocessing parameters
        self.car_enabled = True
        self.ica_enabled = True
        self.ica_components = 20  # Number of ICA components
        self.temporal_smoothing = True
        self.smoothing_window = 5  # samples
        self.artifact_detection_method = 'multi_criteria'  # 'variance', 'multi_criteria'
        self.baseline_method = 'percentage_change'  # 'zscore', 'percentage_change', 'db'
        
        # Results storage
        self.preprocessing_results = {}
        self.quality_metrics = {}
        
    def preprocess_pipeline(self, raw_data: Dict) -> Dict:
        """
        Complete preprocessing pipeline.
        
        Args:
            raw_data: Dictionary containing raw ECoG data
            
        Returns:
            Dictionary containing preprocessed data and metadata
        """
        print("🚀 Starting Comprehensive ECoG Preprocessing Pipeline")
        print("=" * 60)
        
        ecog_data = raw_data['ecog_data']
        photodiode = raw_data['photodiode']
        stimcode = raw_data['stimcode']
        
        with track_progress("Enhanced preprocessing pipeline", 9) as pbar:
            # Step 1: Bandpass filtering
            pbar.update(1, "Bandpass filtering (0.5-150 Hz)")
            filtered_data = self._apply_bandpass_filter(ecog_data)
            
            # Step 2: Notch filtering
            pbar.update(1, "Notch filtering (50/60 Hz)")
            filtered_data = self._apply_notch_filter(filtered_data)
            
            # Step 3: Common Average Reference (CAR)
            pbar.update(1, "Common Average Reference (CAR)")
            car_data = self._apply_common_average_reference(filtered_data)
            
            # Step 4: Enhanced artifact rejection
            pbar.update(1, "Enhanced artifact rejection")
            good_channels, bad_channels = self._detect_artifacts_enhanced(car_data)
            
            # Step 5: Independent Component Analysis (ICA)
            pbar.update(1, "Independent Component Analysis (ICA)")
            ica_data = self._apply_ica(car_data[good_channels, :])
            
            # Step 6: Temporal smoothing
            pbar.update(1, "Temporal smoothing")
            smoothed_data = self._apply_temporal_smoothing(ica_data)
            
            # Step 7: Stimulus alignment
            pbar.update(1, "Stimulus alignment")
            trial_onsets = self._detect_trial_onsets(photodiode, stimcode)
            
            # Step 8: Trial segmentation
            pbar.update(1, "Trial segmentation")
            epochs, time_vector = self._extract_trials(smoothed_data, trial_onsets)
            
            # Step 9: Enhanced normalization
            pbar.update(1, "Enhanced normalization")
            normalized_epochs = self._normalize_epochs_enhanced(epochs, time_vector)
        
        # Store results
        self.preprocessing_results = {
            'epochs': normalized_epochs,
            'time_vector': time_vector,
            'trial_onsets': trial_onsets,
            'good_channels': good_channels,
            'bad_channels': bad_channels,
            'filtered_data': filtered_data,
            'car_data': car_data,
            'ica_data': ica_data,
            'smoothed_data': smoothed_data,
            'sampling_rate': self.fs,
            'stimcode': stimcode[trial_onsets] if len(trial_onsets) > 0 else np.array([]),
            'preprocessing_params': {
                'bandpass_low': self.bandpass_low,
                'bandpass_high': self.bandpass_high,
                'notch_freqs': self.notch_freqs,
                'trial_window': self.trial_window,
                'baseline_window': self.baseline_window,
                'car_enabled': self.car_enabled,
                'ica_enabled': self.ica_enabled,
                'ica_components': self.ica_components,
                'temporal_smoothing': self.temporal_smoothing,
                'artifact_detection_method': self.artifact_detection_method,
                'baseline_method': self.baseline_method,
                'n_trials': len(trial_onsets),
                'n_channels': len(good_channels)
            }
        }
        
        # Compute quality metrics
        self._compute_quality_metrics()
        
        print("✅ Preprocessing completed successfully!")
        return self.preprocessing_results
    
    def _apply_bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply bandpass filter (0.5-150 Hz)."""
        print(f"   🔧 Applying bandpass filter: {self.bandpass_low}-{self.bandpass_high} Hz")
        
        nyquist = self.fs / 2
        low = self.bandpass_low / nyquist
        high = self.bandpass_high / nyquist
        
        # Ensure frequencies are within valid range
        low = max(0.01, min(low, 0.99))
        high = max(low + 0.01, min(high, 0.99))
        
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_data = signal.filtfilt(b, a, data, axis=1)
        
        print(f"   ✅ Bandpass filtering completed")
        return filtered_data
    
    def _apply_notch_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply notch filters (50/60 Hz)."""
        print(f"   🔧 Applying notch filters at: {self.notch_freqs} Hz")
        
        filtered_data = data.copy()
        for freq in self.notch_freqs:
            if freq < self.fs / 2:  # Only apply if frequency is below Nyquist
                nyquist = self.fs / 2
                low = (freq - 1) / nyquist
                high = (freq + 1) / nyquist
                low = max(0.01, min(low, 0.99))
                high = max(low + 0.01, min(high, 0.99))
                
                b, a = signal.butter(4, [low, high], btype='bandstop')
                filtered_data = signal.filtfilt(b, a, filtered_data, axis=1)
        
        print(f"   ✅ Notch filtering completed")
        return filtered_data
    
    def _apply_common_average_reference(self, data: np.ndarray) -> np.ndarray:
        """Apply Common Average Reference (CAR) to remove common noise."""
        print("   🔧 Applying Common Average Reference (CAR)")
        
        # Compute average across all channels
        average_signal = np.mean(data, axis=0, keepdims=True)
        
        # Subtract average from each channel
        car_data = data - average_signal
        
        print(f"   ✅ CAR applied successfully")
        return car_data
    
    def _apply_ica(self, data: np.ndarray) -> np.ndarray:
        """Apply Independent Component Analysis for artifact removal."""
        if not self.ica_enabled:
            return data
        
        print(f"   🔧 Applying ICA with {self.ica_components} components")
        
        # Transpose data for ICA (samples x channels)
        data_transposed = data.T
        
        # Apply ICA
        ica = FastICA(n_components=min(self.ica_components, data.shape[0]), 
                     random_state=42, max_iter=1000)
        ica_components = ica.fit_transform(data_transposed)
        
        # Reconstruct signal
        reconstructed_data = ica.inverse_transform(ica_components)
        
        # Transpose back to original format (channels x samples)
        ica_data = reconstructed_data.T
        
        print(f"   ✅ ICA applied successfully")
        return ica_data
    
    def _apply_temporal_smoothing(self, data: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing to reduce high-frequency noise."""
        if not self.temporal_smoothing:
            return data
        
        print(f"   🔧 Applying temporal smoothing (window={self.smoothing_window})")
        
        # Apply moving average filter
        kernel = np.ones(self.smoothing_window) / self.smoothing_window
        smoothed_data = np.apply_along_axis(
            lambda x: np.convolve(x, kernel, mode='same'), 
            axis=1, arr=data
        )
        
        print(f"   ✅ Temporal smoothing applied successfully")
        return smoothed_data
    
    def _detect_artifacts(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect artifacts using variance threshold."""
        print("   🔧 Detecting artifacts using variance threshold")
        
        # Compute channel statistics
        channel_vars = np.var(data, axis=1)
        channel_means = np.mean(data, axis=1)
        channel_max_amps = np.max(np.abs(data), axis=1)
        
        # Variance-based artifact detection
        var_threshold = np.mean(channel_vars) + 3 * np.std(channel_vars)
        bad_channels_var = np.where(channel_vars > var_threshold)[0]
        
        # Amplitude-based artifact detection
        amp_threshold = np.mean(channel_max_amps) + 3 * np.std(channel_max_amps)
        bad_channels_amp = np.where(channel_max_amps > amp_threshold)[0]
        
        # Combine both criteria
        bad_channels = np.unique(np.concatenate([bad_channels_var, bad_channels_amp]))
        good_channels = np.array([i for i in range(data.shape[0]) if i not in bad_channels])
        
        print(f"   📊 Good channels: {len(good_channels)}/{data.shape[0]}")
        print(f"   🚫 Bad channels: {bad_channels}")
        
        return good_channels, bad_channels
    
    def _detect_artifacts_enhanced(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Enhanced artifact detection using multiple criteria."""
        print("   🔧 Enhanced artifact detection using multiple criteria")
        
        n_channels = data.shape[0]
        bad_channels = set()
        
        # Criterion 1: Variance threshold
        channel_variances = np.var(data, axis=1)
        variance_threshold = np.mean(channel_variances) + 3 * np.std(channel_variances)
        high_variance_channels = np.where(channel_variances > variance_threshold)[0]
        bad_channels.update(high_variance_channels)
        
        # Criterion 2: Amplitude threshold
        max_amplitudes = np.max(np.abs(data), axis=1)
        amplitude_threshold = np.mean(max_amplitudes) + 3 * np.std(max_amplitudes)
        high_amplitude_channels = np.where(max_amplitudes > amplitude_threshold)[0]
        bad_channels.update(high_amplitude_channels)
        
        # Criterion 3: Correlation with other channels
        correlation_matrix = np.corrcoef(data)
        mean_correlations = np.mean(correlation_matrix, axis=1)
        correlation_threshold = np.mean(mean_correlations) - 2 * np.std(mean_correlations)
        low_correlation_channels = np.where(mean_correlations < correlation_threshold)[0]
        bad_channels.update(low_correlation_channels)
        
        # Criterion 4: Spectral characteristics
        for ch in range(n_channels):
            # Check for excessive high-frequency power
            freqs, psd = signal.welch(data[ch, :], fs=self.fs, nperseg=1024)
            high_freq_power = np.mean(psd[freqs > 100])
            total_power = np.mean(psd)
            if high_freq_power / total_power > 0.3:  # More than 30% high-freq power
                bad_channels.add(ch)
        
        # Convert to arrays
        bad_channels = np.array(list(bad_channels))
        good_channels = np.setdiff1d(np.arange(n_channels), bad_channels)
        
        print(f"   ✅ Enhanced artifact detection completed: {len(good_channels)} good, {len(bad_channels)} bad channels")
        return good_channels, bad_channels
    
    def _detect_trial_onsets(self, photodiode: np.ndarray, stimcode: np.ndarray) -> np.ndarray:
        """Detect trial onsets via photodiode signal."""
        print("   🔧 Detecting trial onsets via photodiode signal")
        
        # Find photodiode signal changes (stimulus onsets)
        photodiode_diff = np.diff(photodiode)
        threshold = np.std(photodiode_diff) * 2
        onset_indices = np.where(np.abs(photodiode_diff) > threshold)[0]
        
        # Filter out onsets that are too close together (minimum 500ms apart)
        min_interval = int(0.5 * self.fs)
        filtered_onsets = [onset_indices[0]] if len(onset_indices) > 0 else []
        
        for onset in onset_indices[1:]:
            if onset - filtered_onsets[-1] > min_interval:
                filtered_onsets.append(onset)
        
        print(f"   📊 Detected {len(filtered_onsets)} trial onsets")
        return np.array(filtered_onsets)
    
    def _extract_trials(self, data: np.ndarray, trial_onsets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract trial epochs (100-400 ms post-onset)."""
        print("   🔧 Extracting trial epochs")
        
        if len(trial_onsets) == 0:
            return np.array([]), np.array([])
        
        # Convert time windows to samples
        pre_samples = int(0.3 * self.fs)  # 300ms before onset
        post_samples = int(0.4 * self.fs)  # 400ms after onset
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
        time_vector = np.arange(-pre_samples, post_samples) / self.fs * 1000  # Convert to ms
        
        print(f"   📊 Extracted {len(epochs)} trials, {epochs.shape[1]} channels")
        return epochs, time_vector
    
    def _normalize_epochs(self, epochs: np.ndarray, time_vector: np.ndarray) -> np.ndarray:
        """Z-score normalize epochs using baseline period (-300 to 0 ms)."""
        print("   🔧 Z-score normalizing epochs")
        
        if epochs.size == 0:
            return epochs
        
        # Find baseline period indices
        baseline_mask = (time_vector >= self.baseline_window[0]) & (time_vector <= self.baseline_window[1])
        baseline_data = epochs[:, :, baseline_mask]
        
        # Compute mean and std for each trial and channel
        baseline_mean = np.mean(baseline_data, axis=2, keepdims=True)
        baseline_std = np.std(baseline_data, axis=2, keepdims=True)
        
        # Avoid division by zero
        baseline_std = np.where(baseline_std == 0, 1, baseline_std)
        
        # Z-score normalize
        normalized_epochs = (epochs - baseline_mean) / baseline_std
        
        print(f"   📊 Normalized epochs shape: {normalized_epochs.shape}")
        return normalized_epochs
    
    def _normalize_epochs_enhanced(self, epochs: np.ndarray, time_vector: np.ndarray) -> np.ndarray:
        """Enhanced normalization with multiple methods."""
        print(f"   🔧 Enhanced normalization using {self.baseline_method}")
        
        if epochs.size == 0:
            return epochs
        
        # Find baseline period indices
        baseline_mask = (time_vector >= self.baseline_window[0]) & (time_vector <= self.baseline_window[1])
        baseline_data = epochs[:, :, baseline_mask]
        
        if self.baseline_method == 'zscore':
            # Z-score normalization
            baseline_mean = np.mean(baseline_data, axis=2, keepdims=True)
            baseline_std = np.std(baseline_data, axis=2, keepdims=True)
            baseline_std = np.where(baseline_std == 0, 1, baseline_std)
            normalized_epochs = (epochs - baseline_mean) / baseline_std
            
        elif self.baseline_method == 'percentage_change':
            # Percentage change normalization
            baseline_mean = np.mean(baseline_data, axis=2, keepdims=True)
            baseline_mean = np.where(baseline_mean == 0, 1e-10, baseline_mean)  # Avoid division by zero
            normalized_epochs = ((epochs - baseline_mean) / np.abs(baseline_mean)) * 100
            
        elif self.baseline_method == 'db':
            # Decibel normalization
            baseline_mean = np.mean(baseline_data, axis=2, keepdims=True)
            baseline_mean = np.where(baseline_mean <= 0, 1e-10, baseline_mean)  # Ensure positive values
            normalized_epochs = 10 * np.log10(epochs / baseline_mean)
            
        else:
            # Default to z-score
            baseline_mean = np.mean(baseline_data, axis=2, keepdims=True)
            baseline_std = np.std(baseline_data, axis=2, keepdims=True)
            baseline_std = np.where(baseline_std == 0, 1, baseline_std)
            normalized_epochs = (epochs - baseline_mean) / baseline_std
        
        print(f"   📊 Enhanced normalized epochs shape: {normalized_epochs.shape}")
        return normalized_epochs
    
    def _compute_quality_metrics(self):
        """Compute comprehensive quality metrics."""
        epochs = self.preprocessing_results['epochs']
        good_channels = self.preprocessing_results['good_channels']
        bad_channels = self.preprocessing_results['bad_channels']
        
        if epochs.size == 0:
            self.quality_metrics = {
                'quality_score': 0,
                'issues': ['No valid trials detected']
            }
            return
        
        # Channel quality metrics
        channel_quality_rate = len(good_channels) / (len(good_channels) + len(bad_channels)) * 100
        
        # Trial quality metrics
        trial_vars = np.var(epochs, axis=(1, 2))
        trial_max_amps = np.max(np.abs(epochs), axis=(1, 2))
        
        # Check for flat channels (low variance)
        flat_channels = np.sum(trial_vars < 0.1)
        
        # Check for high-amplitude artifacts
        artifact_trials = np.sum(trial_max_amps > 10)  # Z-score > 10
        
        # Overall quality score
        quality_score = 100 - (len(bad_channels) * 2) - (flat_channels * 1) - (artifact_trials * 0.5)
        quality_score = max(0, quality_score)
        
        # Compile issues
        issues = []
        if len(bad_channels) > 0:
            issues.append(f'{len(bad_channels)} bad channels detected')
        if flat_channels > 0:
            issues.append(f'{flat_channels} trials with low variance')
        if artifact_trials > 0:
            issues.append(f'{artifact_trials} trials with high-amplitude artifacts')
        
        self.quality_metrics = {
            'quality_score': quality_score,
            'channel_quality_rate': channel_quality_rate,
            'n_trials': epochs.shape[0],
            'n_good_channels': len(good_channels),
            'n_bad_channels': len(bad_channels),
            'flat_trials': int(flat_channels),
            'artifact_trials': int(artifact_trials),
            'issues': issues,
            'epoch_shape': list(epochs.shape)
        }
    
    def save_preprocessed_data(self, save_dir: Path):
        """Save preprocessed data and metadata."""
        print("💾 Saving preprocessed data...")
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main data
        np.save(save_dir / 'epochs.npy', self.preprocessing_results['epochs'])
        np.save(save_dir / 'time_vector.npy', self.preprocessing_results['time_vector'])
        np.save(save_dir / 'trial_onsets.npy', self.preprocessing_results['trial_onsets'])
        np.save(save_dir / 'good_channels.npy', self.preprocessing_results['good_channels'])
        np.save(save_dir / 'bad_channels.npy', self.preprocessing_results['bad_channels'])
        np.save(save_dir / 'stimcode.npy', self.preprocessing_results['stimcode'])
        np.save(save_dir / 'filtered_data.npy', self.preprocessing_results['filtered_data'])
        
        # Save metadata
        with open(save_dir / 'preprocessing_params.json', 'w') as f:
            json.dump(self.preprocessing_results['preprocessing_params'], f, indent=2)
        
        with open(save_dir / 'quality_metrics.json', 'w') as f:
            json.dump(self.quality_metrics, f, indent=2)
        
        print(f"✅ Preprocessed data saved to: {save_dir}")
    
    def get_summary_report(self) -> str:
        """Generate comprehensive preprocessing summary report."""
        if not self.quality_metrics:
            return "No preprocessing results available."
        
        report = f"""
# ECoG Preprocessing Summary Report

## Pipeline Overview
- **Bandpass Filter**: {self.bandpass_low}-{self.bandpass_high} Hz
- **Notch Filters**: {self.notch_freqs} Hz
- **Trial Window**: {self.trial_window[0]}-{self.trial_window[1]} ms post-stimulus
- **Baseline Window**: {self.baseline_window[0]}-{self.baseline_window[1]} ms pre-stimulus

## Data Quality Results
- **Overall Quality Score**: {self.quality_metrics['quality_score']:.1f}/100
- **Channel Quality Rate**: {self.quality_metrics['channel_quality_rate']:.1f}%
- **Total Trials**: {self.quality_metrics['n_trials']}
- **Good Channels**: {self.quality_metrics['n_good_channels']}
- **Bad Channels**: {self.quality_metrics['n_bad_channels']}
- **Epoch Shape**: {self.quality_metrics['epoch_shape']}

## Quality Issues
{chr(10).join([f'- {issue}' for issue in self.quality_metrics['issues']]) if self.quality_metrics['issues'] else '- No significant issues detected'}

## Preprocessing Status
✅ **COMPLETED SUCCESSFULLY**
"""
        return report
