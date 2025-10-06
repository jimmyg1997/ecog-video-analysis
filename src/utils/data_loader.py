"""
Data loading utilities for ECoG video analysis
"""

import numpy as np
import pandas as pd
import scipy.io
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """Unified data loader for ECoG video analysis pipeline."""
    
    def __init__(self, data_dir: str = "data"):
        # Use absolute path to ensure it works from any directory
        if not Path(data_dir).is_absolute():
            # Find the project root (where this file is located)
            project_root = Path(__file__).parent.parent.parent
            self.data_dir = project_root / data_dir
        else:
            self.data_dir = Path(data_dir)
        
        self.raw_dir = self.data_dir / "raw"
        self.preprocessed_dir = self.data_dir / "preprocessed"
        self.features_dir = self.data_dir / "features"
        
    def load_raw_data(self, filename: str = "Walk.mat") -> Dict:
        """
        Load raw ECoG data from MATLAB file.
        
        Args:
            filename: Name of the MATLAB file
            
        Returns:
            Dictionary containing ECoG data and metadata
        """
        mat_path = self.raw_dir / filename
        
        if not mat_path.exists():
            raise FileNotFoundError(f"Raw data file not found: {mat_path}")
        
        print(f"📊 Loading raw data from: {mat_path}")
        mat_data = scipy.io.loadmat(str(mat_path))
        
        # Extract data based on documentation
        walk_data = mat_data['y']
        sampling_rate = mat_data['SR'][0, 0] if 'SR' in mat_data else 1200
        
        # Extract channels according to Walk_settings.txt
        data = {
            'ecog_data': walk_data[1:161, :],      # CH2-161: ECoG 1-160
            'photodiode': walk_data[161, :],       # CH162: DI (Photodiode Feedback)
            'stimcode': walk_data[162, :],         # CH163: StimCode
            'groupid': walk_data[163, :],          # CH164: GroupId
            'sampling_rate': sampling_rate,
            'n_channels': 160,
            'n_samples': walk_data.shape[1],
            'duration': walk_data.shape[1] / sampling_rate
        }
        
        print(f"✅ Loaded {data['n_channels']} channels, {data['n_samples']} samples")
        print(f"📈 Sampling rate: {data['sampling_rate']} Hz")
        print(f"⏱️  Duration: {data['duration']:.1f} seconds")
        
        return data
    
    def load_paradigm_info(self, filename: str = "Walk_paradigmInfo.mat") -> Dict:
        """Load paradigm information."""
        paradigm_path = self.raw_dir / filename
        
        if not paradigm_path.exists():
            print(f"⚠️  Paradigm info not found: {paradigm_path}")
            return {}
        
        print(f"📋 Loading paradigm info from: {paradigm_path}")
        paradigm_data = scipy.io.loadmat(str(paradigm_path))
        
        return paradigm_data
    
    def load_preprocessed_data(self, filename: str = "ecog_preprocessed.npy") -> np.ndarray:
        """Load preprocessed ECoG data."""
        preprocessed_path = self.preprocessed_dir / filename
        
        if not preprocessed_path.exists():
            raise FileNotFoundError(f"Preprocessed data not found: {preprocessed_path}")
        
        print(f"📊 Loading preprocessed data from: {preprocessed_path}")
        data = np.load(preprocessed_path)
        print(f"✅ Loaded preprocessed data shape: {data.shape}")
        
        return data
    
    def load_features(self, filename: str) -> np.ndarray:
        """Load extracted features."""
        features_path = self.features_dir / filename
        
        if not features_path.exists():
            raise FileNotFoundError(f"Features not found: {features_path}")
        
        print(f"📊 Loading features from: {features_path}")
        data = np.load(features_path)
        print(f"✅ Loaded features shape: {data.shape}")
        
        return data
    
    def save_preprocessed_data(self, data: np.ndarray, filename: str = "ecog_preprocessed.npy") -> None:
        """Save preprocessed data."""
        self.preprocessed_dir.mkdir(exist_ok=True)
        save_path = self.preprocessed_dir / filename
        
        print(f"💾 Saving preprocessed data to: {save_path}")
        np.save(save_path, data)
        print(f"✅ Saved preprocessed data shape: {data.shape}")
    
    def save_features(self, features: np.ndarray, filename: str) -> None:
        """Save extracted features."""
        self.features_dir.mkdir(exist_ok=True)
        save_path = self.features_dir / filename
        
        print(f"💾 Saving features to: {save_path}")
        np.save(save_path, features)
        print(f"✅ Saved features shape: {features.shape}")
    
    def get_brain_regions(self) -> Dict[str, list]:
        """Get brain region channel assignments."""
        return {
            'Occipital': list(range(1, 41)),      # Visual cortex
            'Temporal': list(range(41, 81)),      # Temporal lobe
            'Parietal': list(range(81, 121)),     # Parietal lobe
            'Central': list(range(121, 141)),     # Central region
            'Frontal': list(range(141, 161))      # Frontal lobe
        }
    
    def get_visual_categories(self) -> Dict[str, str]:
        """Get visual stimulus categories."""
        return {
            'digit': 'Numbers (0-9)',
            'kanji': 'Japanese Kanji characters',
            'face': 'Human faces',
            'body': 'Human bodies/figures',
            'object': 'Various objects',
            'hiragana': 'Japanese Hiragana characters',
            'line': 'Line patterns/shapes'
        }
