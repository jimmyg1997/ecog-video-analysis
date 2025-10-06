#!/usr/bin/env python3
"""
Temporal-Spectral Analysis Dashboard
IEEE-SMC-2025 ECoG Video Analysis Competition

This module creates a comprehensive dashboard showing all aspects of the data
including raw signals, filtered data, features, and analysis results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import signal
from scipy.fft import fft, fftfreq
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class TemporalSpectralDashboard:
    """Comprehensive temporal-spectral analysis dashboard."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the temporal-spectral dashboard."""
        self.config = config or {}
        
        # Color schemes
        self.colors = {
            'raw': '#FF6B6B',
            'filtered': '#4ECDC4',
            'features': '#45B7D1',
            'gamma': '#96CEB4',
            'theta': '#FFEAA7',
            'alpha': '#DDA0DD',
            'beta': '#98FB98',
            'high_gamma': '#F0E68C'
        }
        
        # Frequency bands
        self.frequency_bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 80),
            'high_gamma': (80, 150)
        }
    
    def create_signal_overview(self, raw_data: Dict[str, Any],
                             preprocessed_data: Dict[str, Any],
                             save_path: Path = None) -> go.Figure:
        """Create overview of raw vs preprocessed signals."""
        print("🎨 Creating signal overview dashboard")
        
        # Extract data
        ecog_data = raw_data['ecog_data']
        sampling_rate = raw_data['sampling_rate']
        
        # Get preprocessed data
        if 'normalized_epochs' in preprocessed_data:
            processed_data = preprocessed_data['normalized_epochs']
        else:
            processed_data = ecog_data
        
        # Create time vectors
        time_raw = np.arange(ecog_data.shape[1]) / sampling_rate
        time_processed = np.arange(processed_data.shape[2]) / sampling_rate
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                "Raw ECoG Signal (Sample Channels)",
                "Preprocessed ECoG Signal (Sample Channels)",
                "Power Spectral Density - Raw",
                "Power Spectral Density - Processed",
                "Signal Statistics - Raw",
                "Signal Statistics - Processed"
            ],
            specs=[[{'secondary_y': False}, {'secondary_y': False}],
                   [{'secondary_y': False}, {'secondary_y': False}],
                   [{'secondary_y': False}, {'secondary_y': False}]]
        )
        
        # 1. Raw signal (sample channels)
        sample_channels = [0, 50, 100, 150]
        for i, ch in enumerate(sample_channels):
            if ch < ecog_data.shape[0]:
                fig.add_trace(
                    go.Scatter(
                        x=time_raw[:5000],  # First 5 seconds
                        y=ecog_data[ch, :5000],
                        mode='lines',
                        name=f'Ch{ch+1}',
                        line=dict(width=1),
                        opacity=0.7
                    ),
                    row=1, col=1
                )
        
        # 2. Processed signal (sample channels)
        for i, ch in enumerate(sample_channels):
            if ch < processed_data.shape[1]:
                # Average across trials
                avg_signal = np.mean(processed_data[:, ch, :], axis=0)
                fig.add_trace(
                    go.Scatter(
                        x=time_processed,
                        y=avg_signal,
                        mode='lines',
                        name=f'Ch{ch+1}',
                        line=dict(width=1),
                        opacity=0.7
                    ),
                    row=1, col=2
                )
        
        # 3. Power Spectral Density - Raw
        freqs, psd = signal.welch(ecog_data[0, :], fs=sampling_rate, nperseg=1024)
        fig.add_trace(
            go.Scatter(
                x=freqs,
                y=10 * np.log10(psd),
                mode='lines',
                name='Raw PSD',
                line=dict(color=self.colors['raw'])
            ),
            row=2, col=1
        )
        
        # 4. Power Spectral Density - Processed
        avg_processed = np.mean(processed_data[:, 0, :], axis=0)
        freqs_proc, psd_proc = signal.welch(avg_processed, fs=sampling_rate, nperseg=1024)
        fig.add_trace(
            go.Scatter(
                x=freqs_proc,
                y=10 * np.log10(psd_proc),
                mode='lines',
                name='Processed PSD',
                line=dict(color=self.colors['filtered'])
            ),
            row=2, col=2
        )
        
        # 5. Signal statistics - Raw
        raw_stats = {
            'Mean': np.mean(ecog_data, axis=1),
            'Std': np.std(ecog_data, axis=1),
            'Min': np.min(ecog_data, axis=1),
            'Max': np.max(ecog_data, axis=1)
        }
        
        for stat_name, stat_values in raw_stats.items():
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(stat_values))),
                    y=stat_values,
                    mode='lines',
                    name=f'Raw {stat_name}',
                    line=dict(width=2)
                ),
                row=3, col=1
            )
        
        # 6. Signal statistics - Processed
        processed_stats = {
            'Mean': np.mean(processed_data, axis=(0, 2)),
            'Std': np.std(processed_data, axis=(0, 2)),
            'Min': np.min(processed_data, axis=(0, 2)),
            'Max': np.max(processed_data, axis=(0, 2))
        }
        
        for stat_name, stat_values in processed_stats.items():
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(stat_values))),
                    y=stat_values,
                    mode='lines',
                    name=f'Processed {stat_name}',
                    line=dict(width=2)
                ),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            title="Signal Overview Dashboard",
            height=1200,
            width=1400,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Time (s)", row=1, col=1)
        fig.update_xaxes(title_text="Time (s)", row=1, col=2)
        fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
        fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=2)
        fig.update_xaxes(title_text="Channel", row=3, col=1)
        fig.update_xaxes(title_text="Channel", row=3, col=2)
        
        fig.update_yaxes(title_text="Amplitude (μV)", row=1, col=1)
        fig.update_yaxes(title_text="Amplitude (μV)", row=1, col=2)
        fig.update_yaxes(title_text="PSD (dB)", row=2, col=1)
        fig.update_yaxes(title_text="PSD (dB)", row=2, col=2)
        fig.update_yaxes(title_text="Value", row=3, col=1)
        fig.update_yaxes(title_text="Value", row=3, col=2)
        
        # Save if path provided
        if save_path:
            fig.write_html(save_path / "signal_overview_dashboard.html")
            fig.write_image(save_path / "signal_overview_dashboard.png")
        
        return fig
    
    def create_frequency_analysis(self, preprocessed_data: Dict[str, Any],
                                sampling_rate: int,
                                save_path: Path = None) -> go.Figure:
        """Create comprehensive frequency analysis dashboard."""
        print("🎨 Creating frequency analysis dashboard")
        
        # Extract data
        if 'normalized_epochs' in preprocessed_data:
            data = preprocessed_data['normalized_epochs']
        else:
            return None
        
        # Average across trials
        avg_data = np.mean(data, axis=0)  # (channels, time)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Power Spectral Density by Channel",
                "Frequency Band Power Distribution",
                "Time-Frequency Analysis",
                "Frequency Band Correlations"
            ],
            specs=[[{'type': 'heatmap'}, {'type': 'bar'}],
                   [{'type': 'heatmap'}, {'type': 'heatmap'}]]
        )
        
        # 1. Power Spectral Density by Channel
        psd_matrix = []
        freqs = None
        
        for ch in range(min(20, data.shape[1])):  # Sample channels
            freqs, psd = signal.welch(avg_data[ch, :], fs=sampling_rate, nperseg=1024)
            psd_matrix.append(10 * np.log10(psd))
        
        psd_matrix = np.array(psd_matrix)
        
        fig.add_trace(
            go.Heatmap(
                z=psd_matrix,
                x=freqs,
                y=[f'Ch{i+1}' for i in range(len(psd_matrix))],
                colorscale='Viridis',
                name='PSD by Channel'
            ),
            row=1, col=1
        )
        
        # 2. Frequency Band Power Distribution
        band_powers = {}
        for band_name, (low, high) in self.frequency_bands.items():
            # Find frequency indices
            band_mask = (freqs >= low) & (freqs <= high)
            band_power = np.mean(psd_matrix[:, band_mask], axis=1)
            band_powers[band_name] = band_power
        
        # Create bar plot
        band_names = list(band_powers.keys())
        band_values = [np.mean(band_powers[band]) for band in band_names]
        band_colors = [self.colors.get(band, '#CCCCCC') for band in band_names]
        
        fig.add_trace(
            go.Bar(
                x=band_names,
                y=band_values,
                marker_color=band_colors,
                name='Band Power'
            ),
            row=1, col=2
        )
        
        # 3. Time-Frequency Analysis (simplified)
        # Use spectrogram for one channel
        ch_idx = 0
        f, t, Sxx = signal.spectrogram(avg_data[ch_idx, :], fs=sampling_rate, nperseg=256)
        
        fig.add_trace(
            go.Heatmap(
                z=10 * np.log10(Sxx),
                x=t,
                y=f,
                colorscale='Viridis',
                name='Spectrogram'
            ),
            row=2, col=1
        )
        
        # 4. Frequency Band Correlations
        band_corr_matrix = np.corrcoef(list(band_powers.values()))
        
        fig.add_trace(
            go.Heatmap(
                z=band_corr_matrix,
                x=band_names,
                y=band_names,
                colorscale='RdBu',
                zmid=0,
                name='Band Correlations'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Frequency Analysis Dashboard",
            height=800,
            width=1200,
            showlegend=False
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=1)
        fig.update_xaxes(title_text="Frequency Band", row=1, col=2)
        fig.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig.update_xaxes(title_text="Frequency Band", row=2, col=2)
        
        fig.update_yaxes(title_text="Channel", row=1, col=1)
        fig.update_yaxes(title_text="Power (dB)", row=1, col=2)
        fig.update_yaxes(title_text="Frequency (Hz)", row=2, col=1)
        fig.update_yaxes(title_text="Frequency Band", row=2, col=2)
        
        # Save if path provided
        if save_path:
            fig.write_html(save_path / "frequency_analysis_dashboard.html")
            fig.write_image(save_path / "frequency_analysis_dashboard.png")
        
        return fig
    
    def create_feature_analysis(self, all_features: Dict[str, Dict],
                              save_path: Path = None) -> go.Figure:
        """Create feature analysis dashboard."""
        print("🎨 Creating feature analysis dashboard")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Feature Distributions",
                "Feature Correlations",
                "Feature Importance",
                "Feature Statistics"
            ],
            specs=[[{'type': 'box'}, {'type': 'heatmap'}],
                   [{'type': 'bar'}, {'type': 'table'}]]
        )
        
        # 1. Feature Distributions
        feature_data = []
        feature_names = []
        
        for extractor_name, features in all_features.items():
            if 'gamma_power' in features:
                feature_data.append(features['gamma_power'].flatten())
                feature_names.append(f'{extractor_name}_gamma')
            elif 'template_correlations' in features:
                feature_data.append(features['template_correlations'].flatten())
                feature_names.append(f'{extractor_name}_template')
            elif 'transformer_input' in features:
                feature_data.append(features['transformer_input'].flatten())
                feature_names.append(f'{extractor_name}_transformer')
        
        for i, (data, name) in enumerate(zip(feature_data, feature_names)):
            fig.add_trace(
                go.Box(
                    y=data,
                    name=name,
                    boxpoints='outliers'
                ),
                row=1, col=1
            )
        
        # 2. Feature Correlations
        if len(feature_data) > 1:
            # Align feature vectors
            min_length = min(len(data) for data in feature_data)
            aligned_data = [data[:min_length] for data in feature_data]
            
            corr_matrix = np.corrcoef(aligned_data)
            
            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix,
                    x=feature_names,
                    y=feature_names,
                    colorscale='RdBu',
                    zmid=0,
                    name='Feature Correlations'
                ),
                row=1, col=2
            )
        
        # 3. Feature Importance (simplified)
        importance_values = []
        for data in feature_data:
            importance_values.append(np.mean(np.abs(data)))
        
        fig.add_trace(
            go.Bar(
                x=feature_names,
                y=importance_values,
                name='Feature Importance'
            ),
            row=2, col=1
        )
        
        # 4. Feature Statistics Table
        stats_data = []
        for name, data in zip(feature_names, feature_data):
            stats_data.append([
                name,
                f"{len(data):,}",
                f"{np.mean(data):.3f}",
                f"{np.std(data):.3f}",
                f"{np.min(data):.3f}",
                f"{np.max(data):.3f}"
            ])
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['Feature', 'Count', 'Mean', 'Std', 'Min', 'Max'],
                    fill_color='lightblue',
                    align='center'
                ),
                cells=dict(
                    values=list(zip(*stats_data)),
                    fill_color='white',
                    align='center'
                )
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Feature Analysis Dashboard",
            height=800,
            width=1200,
            showlegend=False
        )
        
        # Save if path provided
        if save_path:
            fig.write_html(save_path / "feature_analysis_dashboard.html")
            fig.write_image(save_path / "feature_analysis_dashboard.png")
        
        return fig
    
    def create_comprehensive_dashboard(self, raw_data: Dict[str, Any],
                                     preprocessed_data: Dict[str, Any],
                                     all_features: Dict[str, Dict],
                                     save_path: Path = None) -> go.Figure:
        """Create comprehensive temporal-spectral dashboard."""
        print("🎨 Creating comprehensive temporal-spectral dashboard")
        
        # Create main dashboard with multiple sections
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                "Raw Signal (Sample)",
                "Preprocessed Signal (Sample)",
                "Power Spectral Density",
                "Frequency Band Analysis",
                "Feature Distributions",
                "Feature Correlations",
                "Time-Frequency Analysis",
                "Signal Statistics",
                "Feature Importance"
            ],
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'bar'}, {'type': 'box'}, {'type': 'heatmap'}],
                   [{'type': 'heatmap'}, {'type': 'bar'}, {'type': 'bar'}]]
        )
        
        # Extract data
        ecog_data = raw_data['ecog_data']
        sampling_rate = raw_data['sampling_rate']
        
        if 'normalized_epochs' in preprocessed_data:
            processed_data = preprocessed_data['normalized_epochs']
        else:
            processed_data = ecog_data
        
        # 1. Raw Signal (Sample)
        time_raw = np.arange(ecog_data.shape[1]) / sampling_rate
        fig.add_trace(
            go.Scatter(
                x=time_raw[:5000],
                y=ecog_data[0, :5000],
                mode='lines',
                name='Raw Signal',
                line=dict(color=self.colors['raw'])
            ),
            row=1, col=1
        )
        
        # 2. Preprocessed Signal (Sample)
        time_processed = np.arange(processed_data.shape[2]) / sampling_rate
        avg_processed = np.mean(processed_data[:, 0, :], axis=0)
        fig.add_trace(
            go.Scatter(
                x=time_processed,
                y=avg_processed,
                mode='lines',
                name='Processed Signal',
                line=dict(color=self.colors['filtered'])
            ),
            row=1, col=2
        )
        
        # 3. Power Spectral Density
        freqs, psd = signal.welch(ecog_data[0, :], fs=sampling_rate, nperseg=1024)
        fig.add_trace(
            go.Scatter(
                x=freqs,
                y=10 * np.log10(psd),
                mode='lines',
                name='PSD',
                line=dict(color=self.colors['features'])
            ),
            row=1, col=3
        )
        
        # 4. Frequency Band Analysis
        band_powers = []
        band_names = []
        band_colors = []
        
        for band_name, (low, high) in self.frequency_bands.items():
            band_mask = (freqs >= low) & (freqs <= high)
            band_power = np.mean(psd[band_mask])
            band_powers.append(band_power)
            band_names.append(band_name)
            band_colors.append(self.colors.get(band_name, '#CCCCCC'))
        
        fig.add_trace(
            go.Bar(
                x=band_names,
                y=band_powers,
                marker_color=band_colors,
                name='Band Power'
            ),
            row=2, col=1
        )
        
        # 5. Feature Distributions
        feature_data = []
        feature_names = []
        
        for extractor_name, features in all_features.items():
            if 'gamma_power' in features:
                feature_data.append(features['gamma_power'].flatten())
                feature_names.append(f'{extractor_name}_gamma')
        
        for i, (data, name) in enumerate(zip(feature_data, feature_names)):
            fig.add_trace(
                go.Box(
                    y=data,
                    name=name,
                    boxpoints='outliers'
                ),
                row=2, col=2
            )
        
        # 6. Feature Correlations
        if len(feature_data) > 1:
            min_length = min(len(data) for data in feature_data)
            aligned_data = [data[:min_length] for data in feature_data]
            corr_matrix = np.corrcoef(aligned_data)
            
            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix,
                    x=feature_names,
                    y=feature_names,
                    colorscale='RdBu',
                    zmid=0
                ),
                row=2, col=3
            )
        
        # 7. Time-Frequency Analysis
        f, t, Sxx = signal.spectrogram(ecog_data[0, :], fs=sampling_rate, nperseg=256)
        fig.add_trace(
            go.Heatmap(
                z=10 * np.log10(Sxx),
                x=t,
                y=f,
                colorscale='Viridis'
            ),
            row=3, col=1
        )
        
        # 8. Signal Statistics
        raw_stats = [np.mean(ecog_data, axis=1), np.std(ecog_data, axis=1)]
        stat_names = ['Mean', 'Std']
        
        for i, (stats, name) in enumerate(zip(raw_stats, stat_names)):
            fig.add_trace(
                go.Bar(
                    x=list(range(len(stats))),
                    y=stats,
                    name=name
                ),
                row=3, col=2
            )
        
        # 9. Feature Importance
        importance_values = [np.mean(np.abs(data)) for data in feature_data]
        fig.add_trace(
            go.Bar(
                x=feature_names,
                y=importance_values,
                name='Importance'
            ),
            row=3, col=3
        )
        
        # Update layout
        fig.update_layout(
            title="Comprehensive Temporal-Spectral Analysis Dashboard",
            height=1200,
            width=1400,
            showlegend=False
        )
        
        # Save if path provided
        if save_path:
            fig.write_html(save_path / "comprehensive_temporal_spectral_dashboard.html")
            fig.write_image(save_path / "comprehensive_temporal_spectral_dashboard.png")
        
        return fig
    
    def get_summary_report(self) -> str:
        """Generate a summary report of the temporal-spectral dashboard."""
        report = []
        report.append("🎨 Temporal-Spectral Dashboard Summary")
        report.append("=" * 50)
        
        report.append(f"📊 Frequency Bands: {len(self.frequency_bands)}")
        for band, (low, high) in self.frequency_bands.items():
            report.append(f"   • {band.title()}: {low}-{high} Hz")
        
        report.append(f"\n📊 Color Scheme:")
        for name, color in self.colors.items():
            report.append(f"   • {name.title()}: {color}")
        
        report.append(f"\n📊 Dashboard Components:")
        report.append("   • Signal Overview (Raw vs Processed)")
        report.append("   • Frequency Analysis (PSD, Bands, TF)")
        report.append("   • Feature Analysis (Distributions, Correlations)")
        report.append("   • Comprehensive Multi-Panel Dashboard")
        
        return "\n".join(report)
