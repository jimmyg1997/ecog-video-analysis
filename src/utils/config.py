"""
Configuration management for ECoG video analysis pipeline
"""

import json
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class AnalysisConfig:
    """Configuration for ECoG analysis pipeline."""
    
    # Data parameters
    sampling_rate: int = 1200
    n_channels: int = 160
    
    # Preprocessing parameters
    bandpass_low: float = 0.5
    bandpass_high: float = 150.0
    notch_freqs: list = None
    car_enabled: bool = True
    
    # Feature extraction parameters
    gamma_low: float = 70.0
    gamma_high: float = 150.0
    high_gamma_low: float = 110.0
    high_gamma_high: float = 140.0
    
    # Epoch parameters
    epoch_pre: float = 0.3  # seconds before stimulus
    epoch_post: float = 0.4  # seconds after stimulus
    baseline_start: float = -0.3
    baseline_end: float = 0.0
    
    # Analysis parameters
    n_components_csp: int = 4
    cv_folds: int = 5
    random_state: int = 42
    
    def __post_init__(self):
        if self.notch_freqs is None:
            self.notch_freqs = [50, 60]
    
    @classmethod
    def from_file(cls, config_path: str) -> 'AnalysisConfig':
        """Load configuration from JSON file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            print(f"⚠️  Config file not found: {config_path}, using defaults")
            return cls()
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)
    
    def to_file(self, config_path: str) -> None:
        """Save configuration to JSON file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = {
            'sampling_rate': self.sampling_rate,
            'n_channels': self.n_channels,
            'bandpass_low': self.bandpass_low,
            'bandpass_high': self.bandpass_high,
            'notch_freqs': self.notch_freqs,
            'car_enabled': self.car_enabled,
            'gamma_low': self.gamma_low,
            'gamma_high': self.gamma_high,
            'high_gamma_low': self.high_gamma_low,
            'high_gamma_high': self.high_gamma_high,
            'epoch_pre': self.epoch_pre,
            'epoch_post': self.epoch_post,
            'baseline_start': self.baseline_start,
            'baseline_end': self.baseline_end,
            'n_components_csp': self.n_components_csp,
            'cv_folds': self.cv_folds,
            'random_state': self.random_state
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"💾 Configuration saved to: {config_path}")

# Default configuration
DEFAULT_CONFIG = AnalysisConfig()