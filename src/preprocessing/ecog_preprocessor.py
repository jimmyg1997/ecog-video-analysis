"""
ECoG preprocessing pipeline for visual stimulus analysis
"""

import numpy as np
from scipy import signal
from typing import Dict, Tuple, Optional
from utils.progress_tracker import track_progress
from utils.config import AnalysisConfig

class ECoGPreprocessor:
    """ECoG preprocessing pipeline optimized for visual stimulus analysis."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.fs = config.sampling_rate
        
    def apply_bandpass_filter(self, data: np.ndarray, 
                            low: float = None, high: float = None) -> np.ndarray:
        """
        Apply bandpass filter to ECoG data.
        
        Args:
            data: ECoG data (channels x samples)
            low: Low cutoff frequency
            high: High cutoff frequency
            
        Returns:
            Filtered data
        """
        if low is None:
            low = self.config.bandpass_low
        if high is None:
            high = self.config.bandpass_high
        
        print(f"🔧 Applying bandpass filter: {low}-{high} Hz")
        
        nyquist = self.fs / 2
        low_norm = low / nyquist
        high_norm = high / nyquist
        
        # Design Butterworth filter
        b, a = signal.butter(4, [low_norm, high_norm], btype='band')
        
        # Apply filter to each channel
        filtered_data = np.zeros_like(data)
        for ch in range(data.shape[0]):
            filtered_data[ch, :] = signal.filtfilt(b, a, data[ch, :])
        
        print(f"✅ Bandpass filtering completed")
        return filtered_data
    
    def apply_notch_filter(self, data: np.ndarray, 
                          notch_freqs: list = None) -> np.ndarray:
        """
        Apply notch filter to remove line noise.
        
        Args:
            data: ECoG data (channels x samples)
            notch_freqs: List of frequencies to notch out
            
        Returns:
            Filtered data
        """
        if notch_freqs is None:
            notch_freqs = self.config.notch_freqs
        
        print(f"🔧 Applying notch filter at: {notch_freqs} Hz")
        
        filtered_data = data.copy()
        
        for freq in notch_freqs:
            # Design notch filter
            b, a = signal.iirnotch(freq, Q=30, fs=self.fs)
            
            # Apply to each channel
            for ch in range(data.shape[0]):
                filtered_data[ch, :] = signal.filtfilt(b, a, filtered_data[ch, :])
        
        print(f"✅ Notch filtering completed")
        return filtered_data
    
    def apply_common_average_reference(self, data: np.ndarray, 
                                     bad_channels: list = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Common Average Reference (CAR).
        
        Args:
            data: ECoG data (channels x samples)
            bad_channels: List of bad channel indices to exclude
            
        Returns:
            CAR-referenced data and CAR signal
        """
        print("🔧 Applying Common Average Reference (CAR)")
        
        if bad_channels is None:
            bad_channels = []
        
        # Create good channels mask
        good_channels = [i for i in range(data.shape[0]) if i not in bad_channels]
        
        if len(good_channels) == 0:
            print("⚠️  No good channels for CAR, returning original data")
            return data, np.zeros(data.shape[1])
        
        # Compute CAR signal
        car_signal = np.mean(data[good_channels, :], axis=0)
        
        # Apply CAR to all channels
        car_data = data - car_signal[np.newaxis, :]
        
        print(f"✅ CAR applied using {len(good_channels)} good channels")
        return car_data, car_signal
    
    def detect_visual_events(self, photodiode: np.ndarray, 
                           stimcode: np.ndarray, threshold: float = 0.1) -> Dict:
        """
        Detect visual stimulus events from photodiode and stimcode.
        
        Args:
            photodiode: Photodiode signal
            stimcode: Stimulus code signal
            threshold: Threshold for event detection
            
        Returns:
            Dictionary with event information
        """
        print("🔍 Detecting visual stimulus events")
        
        # Find photodiode changes (visual transitions)
        photodiode_diff = np.abs(np.diff(photodiode))
        event_indices = np.where(photodiode_diff > threshold)[0]
        event_times = event_indices / self.fs
        
        # Find stimcode changes (stimulus type changes)
        stimcode_diff = np.abs(np.diff(stimcode))
        stim_indices = np.where(stimcode_diff > 0)[0]
        stim_times = stim_indices / self.fs
        
        events = {
            'event_times': event_times,
            'event_indices': event_indices,
            'stim_times': stim_times,
            'stim_indices': stim_indices,
            'n_events': len(event_times),
            'n_stim_changes': len(stim_times)
        }
        
        print(f"✅ Detected {len(event_times)} visual events and {len(stim_times)} stimulus changes")
        return events
    
    def extract_epochs(self, data: np.ndarray, events: Dict, 
                      pre: float = None, post: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract epochs around visual events.
        
        Args:
            data: ECoG data (channels x samples)
            events: Event information from detect_visual_events
            pre: Time before event (seconds)
            post: Time after event (seconds)
            
        Returns:
            Epochs (channels x epochs x samples) and time vector
        """
        if pre is None:
            pre = self.config.epoch_pre
        if post is None:
            post = self.config.epoch_post
        
        print(f"📊 Extracting epochs: {pre}s to {post}s around events")
        
        pre_samples = int(pre * self.fs)
        post_samples = int(post * self.fs)
        epoch_length = pre_samples + post_samples
        
        # Create time vector
        time_vector = np.arange(-pre, post, 1/self.fs)[:epoch_length]
        
        # Extract epochs
        epochs = []
        valid_events = []
        
        for i, event_idx in enumerate(events['event_indices']):
            start_idx = event_idx - pre_samples
            end_idx = event_idx + post_samples
            
            # Check bounds
            if start_idx >= 0 and end_idx < data.shape[1]:
                epoch = data[:, start_idx:end_idx]
                epochs.append(epoch)
                valid_events.append(i)
        
        if len(epochs) == 0:
            print("⚠️  No valid epochs extracted")
            return np.array([]), time_vector
        
        epochs_array = np.array(epochs).transpose(1, 0, 2)  # channels x epochs x samples
        
        print(f"✅ Extracted {len(epochs)} epochs of {epoch_length} samples each")
        return epochs_array, time_vector
    
    def baseline_correction(self, epochs: np.ndarray, time_vector: np.ndarray,
                          baseline_start: float = None, baseline_end: float = None) -> np.ndarray:
        """
        Apply baseline correction to epochs.
        
        Args:
            epochs: Epochs (channels x epochs x samples)
            time_vector: Time vector for epochs
            baseline_start: Start of baseline period (seconds)
            baseline_end: End of baseline period (seconds)
            
        Returns:
            Baseline-corrected epochs
        """
        if baseline_start is None:
            baseline_start = self.config.baseline_start
        if baseline_end is None:
            baseline_end = self.config.baseline_end
        
        print(f"🔧 Applying baseline correction: {baseline_start}s to {baseline_end}s")
        
        # Find baseline indices
        baseline_mask = (time_vector >= baseline_start) & (time_vector <= baseline_end)
        baseline_indices = np.where(baseline_mask)[0]
        
        if len(baseline_indices) == 0:
            print("⚠️  No baseline period found, returning original epochs")
            return epochs
        
        # Compute baseline mean for each channel and epoch
        baseline_corrected = epochs.copy()
        
        for ch in range(epochs.shape[0]):
            for epoch in range(epochs.shape[1]):
                baseline_mean = np.mean(epochs[ch, epoch, baseline_indices])
                baseline_corrected[ch, epoch, :] -= baseline_mean
        
        print(f"✅ Baseline correction completed")
        return baseline_corrected
    
    def preprocess_pipeline(self, data: Dict, bad_channels: list = None) -> Dict:
        """
        Complete preprocessing pipeline.
        
        Args:
            data: Raw data dictionary from DataLoader
            bad_channels: List of bad channel indices
            
        Returns:
            Preprocessed data dictionary
        """
        print("🚀 Starting ECoG preprocessing pipeline")
        
        ecog_data = data['ecog_data']
        
        with track_progress("Preprocessing pipeline", 6) as pbar:
            # 1. Bandpass filtering
            pbar.update(1, "Bandpass filtering")
            filtered_data = self.apply_bandpass_filter(ecog_data)
            
            # 2. Notch filtering
            pbar.update(1, "Notch filtering")
            filtered_data = self.apply_notch_filter(filtered_data)
            
            # 3. Common Average Reference
            pbar.update(1, "Common Average Reference")
            car_data, car_signal = self.apply_common_average_reference(filtered_data, bad_channels)
            
            # 4. Event detection
            pbar.update(1, "Event detection")
            events = self.detect_visual_events(data['photodiode'], data['stimcode'])
            
            # 5. Epoch extraction
            pbar.update(1, "Epoch extraction")
            epochs, time_vector = self.extract_epochs(car_data, events)
            
            # 6. Baseline correction
            pbar.update(1, "Baseline correction")
            if epochs.size > 0:
                epochs = self.baseline_correction(epochs, time_vector)
        
        # Prepare output
        preprocessed_data = {
            'ecog_data': car_data,
            'epochs': epochs,
            'time_vector': time_vector,
            'events': events,
            'car_signal': car_signal,
            'sampling_rate': self.fs,
            'n_channels': data['n_channels'],
            'n_epochs': epochs.shape[1] if epochs.size > 0 else 0
        }
        
        print("✅ Preprocessing pipeline completed")
        return preprocessed_data
