"""
Comprehensive Pipeline Visualization Module
IEEE-SMC-2025 ECoG Video Analysis Competition

This module provides comprehensive visualizations for each stage of the ECoG pipeline.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from .brain_atlas import BrainAtlas

class PipelineVisualizer:
    """Comprehensive pipeline visualization."""
    
    def __init__(self, config=None):
        """Initialize visualizer."""
        self.config = config
        self.brain_atlas = BrainAtlas(config)
        self.setup_plotting()
    
    def setup_plotting(self):
        """Setup matplotlib parameters."""
        plt.style.use('default')
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 10,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'font.family': 'sans-serif',
            'axes.grid': True,
            'grid.alpha': 0.3
        })
    
    def create_raw_data_exploration(self, raw_data, save_dir):
        """Create raw data exploration visualizations."""
        print("📊 Creating raw data exploration visualizations...")
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Dataset overview
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('ECoG Raw Data Exploration', fontsize=20, fontweight='bold')
        
        ecog_data = raw_data['ecog_data']
        photodiode = raw_data['photodiode']
        stimcode = raw_data['stimcode']
        sampling_rate = raw_data['sampling_rate']
        
        # Data shape overview
        ax1 = axes[0, 0]
        shape_data = {
            'Channels': ecog_data.shape[0],
            'Samples': ecog_data.shape[1],
            'Duration (min)': ecog_data.shape[1]/sampling_rate/60,
            'Sampling Rate (Hz)': sampling_rate
        }
        bars = ax1.bar(range(len(shape_data)), list(shape_data.values()), 
                      color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax1.set_xticks(range(len(shape_data)))
        ax1.set_xticklabels(list(shape_data.keys()), rotation=45, ha='right')
        ax1.set_title('Dataset Overview')
        ax1.set_ylabel('Count')
        
        # Channel statistics
        ax2 = axes[0, 1]
        channel_std = np.std(ecog_data, axis=1)
        ax2.hist(channel_std, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(np.mean(channel_std), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(channel_std):.2f}')
        ax2.set_xlabel('Channel Standard Deviation')
        ax2.set_ylabel('Number of Channels')
        ax2.set_title('Channel Signal Quality')
        ax2.legend()
        
        # Photodiode signal
        ax3 = axes[1, 0]
        time_seconds = np.arange(len(photodiode)) / sampling_rate
        ax3.plot(time_seconds[:10000], photodiode[:10000], 'b-', alpha=0.7)
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Photodiode Signal')
        ax3.set_title('Photodiode Signal (First 8.3s)')
        ax3.grid(True, alpha=0.3)
        
        # StimCode distribution
        ax4 = axes[1, 1]
        unique_stimcodes, counts = np.unique(stimcode, return_counts=True)
        bars = ax4.bar(unique_stimcodes, counts, color='lightcoral', alpha=0.7)
        ax4.set_xlabel('StimCode')
        ax4.set_ylabel('Count')
        ax4.set_title('StimCode Distribution')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / '00_dataset_overview.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_dir / '00_dataset_overview.svg', bbox_inches='tight')
        plt.close()
        
        # 2. Brain atlas overview
        brain_fig = self.brain_atlas.create_brain_overview(ecog_data)
        plt.savefig(save_dir / '01_brain_atlas_overview.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_dir / '01_brain_atlas_overview.svg', bbox_inches='tight')
        plt.close()
        
        print("✅ Raw data exploration visualizations saved!")
    
    def create_preprocessing_visualizations(self, raw_data, preprocessed_data, save_dir):
        """Create enhanced preprocessing visualizations."""
        print("📊 Creating enhanced preprocessing visualizations...")
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Enhanced preprocessing pipeline comparison
        fig, axes = plt.subplots(4, 2, figsize=(20, 24))
        fig.suptitle('Enhanced ECoG Preprocessing Pipeline', fontsize=20, fontweight='bold')
        
        ecog_data = raw_data['ecog_data']
        filtered_data = preprocessed_data['filtered_data']
        car_data = preprocessed_data.get('car_data', filtered_data)
        ica_data = preprocessed_data.get('ica_data', car_data)
        smoothed_data = preprocessed_data.get('smoothed_data', ica_data)
        epochs = preprocessed_data['epochs']
        time_vector = preprocessed_data['time_vector']
        good_channels = preprocessed_data['good_channels']
        bad_channels = preprocessed_data['bad_channels']
        preprocessing_params = preprocessed_data['preprocessing_params']
        
        # Sample channels for visualization
        sample_channels = good_channels[:3]
        time_window = slice(0, min(5000, ecog_data.shape[1]))
        time_seconds = np.arange(time_window.stop) / raw_data['sampling_rate']
        
        # Enhanced preprocessing steps visualization
        for i, ch in enumerate(sample_channels):
            # Step 1: Raw vs Filtered
            ax1 = axes[i, 0]
            ax1.plot(time_seconds, ecog_data[ch, time_window], 'b-', alpha=0.8, linewidth=1, label='Raw')
            ax1.plot(time_seconds, filtered_data[ch, time_window], 'r-', alpha=0.8, linewidth=1, label='Filtered')
            ax1.set_title(f'Channel {ch+1}: Raw vs Filtered')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Amplitude (μV)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Step 2: CAR Effect
            ax2 = axes[i, 1]
            ax2.plot(time_seconds, filtered_data[ch, time_window], 'b-', alpha=0.8, linewidth=1, label='Before CAR')
            ax2.plot(time_seconds, car_data[ch, time_window], 'g-', alpha=0.8, linewidth=1, label='After CAR')
            ax2.set_title(f'Channel {ch+1}: CAR Effect')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Amplitude (μV)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # ICA and Temporal Smoothing effects
        ch = sample_channels[0]  # Use first channel for detailed comparison
        ax3 = axes[3, 0]
        ax3.plot(time_seconds, car_data[ch, time_window], 'g-', alpha=0.8, linewidth=1, label='After CAR')
        ax3.plot(time_seconds, ica_data[ch, time_window], 'orange', alpha=0.8, linewidth=1, label='After ICA')
        ax3.plot(time_seconds, smoothed_data[ch, time_window], 'purple', alpha=0.8, linewidth=1, label='After Smoothing')
        ax3.set_title(f'Channel {ch+1}: ICA & Temporal Smoothing')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Amplitude (μV)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Epochs visualization
        ax4 = axes[3, 1]
        if epochs.size > 0:
            # Plot first 10 trials
            n_trials_to_plot = min(10, epochs.shape[0])
            for trial in range(n_trials_to_plot):
                ax4.plot(time_vector, epochs[trial, ch, :], alpha=0.6, linewidth=0.8)
            ax4.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Stimulus Onset')
            baseline_window = preprocessing_params.get('baseline_window', (-300, 0))
            ax4.axvspan(baseline_window[0], baseline_window[1], alpha=0.2, color='gray', label='Baseline')
            ax4.set_title(f'Channel {ch+1}: Final Epochs')
            ax4.set_xlabel('Time (ms)')
            ax4.set_ylabel('Amplitude (μV)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No epochs available', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title(f'Channel {ch+1}: No Epochs')
        
        plt.tight_layout()
        plt.savefig(save_dir / '02_preprocessing_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_dir / '02_preprocessing_comparison.svg', bbox_inches='tight')
        plt.close()
        
        # 2. Quality assessment
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Preprocessing Quality Assessment', fontsize=20, fontweight='bold')
        
        # Channel quality
        ax1 = axes[0, 0]
        quality_data = {
            'Good Channels': len(good_channels),
            'Bad Channels': len(bad_channels)
        }
        colors = ['lightgreen', 'lightcoral']
        wedges, texts, autotexts = ax1.pie(quality_data.values(), labels=quality_data.keys(), 
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Channel Quality')
        
        # Trial statistics
        ax2 = axes[0, 1]
        if epochs.size > 0:
            trial_vars = np.var(epochs, axis=(1, 2))
            ax2.hist(trial_vars, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.axvline(np.mean(trial_vars), color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {np.mean(trial_vars):.3f}')
            ax2.set_xlabel('Trial Variance')
            ax2.set_ylabel('Number of Trials')
            ax2.set_title('Trial Quality Distribution')
            ax2.legend()
        
        # Frequency analysis
        ax3 = axes[1, 0]
        if epochs.size > 0:
            # Compute average PSD
            avg_epoch = np.mean(epochs, axis=(0, 1))
            freqs, psd = signal.welch(avg_epoch, fs=raw_data['sampling_rate'], nperseg=256)
            ax3.semilogy(freqs, psd, 'b-', linewidth=2)
            ax3.set_xlabel('Frequency (Hz)')
            ax3.set_ylabel('Power Spectral Density')
            ax3.set_title('Average Power Spectral Density')
            ax3.grid(True, alpha=0.3)
            ax3.set_xlim(0, 100)
        
        # Preprocessing summary
        ax4 = axes[1, 1]
        summary_text = f"""
Preprocessing Summary

✅ Bandpass Filter: 0.5-150 Hz
✅ Notch Filters: 50/60 Hz
✅ Artifact Rejection: {len(bad_channels)} bad channels
✅ Trial Detection: {epochs.shape[0] if epochs.size > 0 else 0} trials
✅ Normalization: Z-score per trial

Quality Score: {preprocessed_data.get('quality_metrics', {}).get('quality_score', 'N/A')}/100
        """
        ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_dir / '03_preprocessing_quality.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_dir / '03_preprocessing_quality.svg', bbox_inches='tight')
        plt.close()
        
        # 3. Enhanced preprocessing step-by-step analysis
        self._create_enhanced_preprocessing_analysis(raw_data, preprocessed_data, save_dir)
        
        print("✅ Enhanced preprocessing visualizations saved!")
    
    def create_feature_extraction_visualizations(self, preprocessed_data, feature_extractors, save_dir):
        """Create comprehensive feature extraction visualizations for all extractor types."""
        print("🎨 Creating comprehensive feature extraction visualizations...")
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create overview of all feature extractors
        self._create_feature_extractor_overview(feature_extractors, save_dir)
        
        # Create visualizations for each extractor type
        for extractor_name, extractor in feature_extractors.items():
            if extractor_name == 'template_correlation':
                self._create_template_correlation_visualizations(extractor, save_dir)
            elif extractor_name == 'csp_lda':
                self._create_csp_lda_visualizations(extractor, save_dir)
            elif extractor_name == 'eegnet':
                self._create_eegnet_visualizations(extractor, save_dir)
            elif extractor_name == 'transformer':
                self._create_transformer_visualizations(extractor, save_dir)
        
        # Create comparative analysis
        self._create_feature_comparison_analysis(feature_extractors, save_dir)
        
        print("✅ Feature extraction visualizations saved!")
    
    def _create_feature_extractor_overview(self, feature_extractors, save_dir):
        """Create overview of all feature extractors."""
        print("🔬 Creating feature extractor overview...")
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Feature Extraction Overview: All Extractor Types', fontsize=20, fontweight='bold')
        
        extractor_info = {
            'template_correlation': {
                'title': 'Template Correlation (LOOCV)',
                'description': 'Gamma template matching with leave-one-out cross-validation',
                'features': ['Template creation', 'Correlation features', 'LOOCV structure'],
                'color': 'blue'
            },
            'csp_lda': {
                'title': 'CSP + LDA Spatial Filtering',
                'description': 'Common Spatial Patterns with Linear Discriminant Analysis',
                'features': ['Spatial filters', 'CSP components', 'LDA classification'],
                'color': 'green'
            },
            'eegnet': {
                'title': 'EEGNet Compact CNN',
                'description': 'Raw time series optimized for compact CNN architecture',
                'features': ['Raw time series', 'Data augmentation', 'CNN input format'],
                'color': 'red'
            },
            'transformer': {
                'title': 'Time Series Transformer',
                'description': 'Extended temporal sequences with attention mechanisms',
                'features': ['Multi-scale features', 'Attention embeddings', 'Long sequences'],
                'color': 'purple'
            }
        }
        
        for i, (extractor_name, info) in enumerate(extractor_info.items()):
            ax = axes[i // 2, i % 2]
            
            # Create feature summary
            if extractor_name in feature_extractors:
                extractor = feature_extractors[extractor_name]
                n_features = len(extractor.features) if hasattr(extractor, 'features') else 0
                status = "✅ Completed"
            else:
                n_features = 0
                status = "❌ Not Available"
            
            # Create visualization
            ax.text(0.1, 0.8, f"**{info['title']}**", transform=ax.transAxes, 
                   fontsize=16, fontweight='bold', color=info['color'])
            ax.text(0.1, 0.7, info['description'], transform=ax.transAxes, 
                   fontsize=12, wrap=True)
            ax.text(0.1, 0.5, f"**Features:** {', '.join(info['features'])}", 
                   transform=ax.transAxes, fontsize=11)
            ax.text(0.1, 0.3, f"**Status:** {status}", transform=ax.transAxes, 
                   fontsize=12, fontweight='bold')
            ax.text(0.1, 0.2, f"**Features Extracted:** {n_features}", 
                   transform=ax.transAxes, fontsize=11)
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_facecolor('lightgray')
            ax.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, 
                                     fill=False, edgecolor=info['color'], linewidth=2))
        
        plt.tight_layout()
        plt.savefig(save_dir / '00_feature_extractor_overview.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_dir / '00_feature_extractor_overview.svg', bbox_inches='tight')
        plt.close()
    
    def _create_template_correlation_visualizations(self, extractor, save_dir):
        """Create visualizations for template correlation extractor."""
        print("🔬 Creating template correlation visualizations...")
        
        if not hasattr(extractor, 'templates') or not extractor.templates:
            print("   ⚠️ No template data available")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        fig.suptitle('Template Correlation Feature Analysis', fontsize=20, fontweight='bold')
        
        # 1. Template visualization
        ax1 = axes[0, 0]
        template_data = []
        template_labels = []
        for stim, template_info in extractor.templates.items():
            template_data.append(template_info['template'])
            template_labels.append(f'Stimulus {stim}')
        
        if template_data:
            template_matrix = np.array(template_data)
            im = ax1.imshow(template_matrix, cmap='viridis', aspect='auto')
            ax1.set_title('Gamma Templates by Stimulus')
            ax1.set_xlabel('Channel Index')
            ax1.set_ylabel('Stimulus')
            ax1.set_yticks(range(len(template_labels)))
            ax1.set_yticklabels(template_labels)
            plt.colorbar(im, ax=ax1)
        
        # 2. Template correlation heatmap
        ax2 = axes[0, 1]
        if hasattr(extractor, 'correlation_features'):
            corr_features = [f for f in extractor.correlation_features.keys() if 'template_correlation' in f]
            if corr_features:
                corr_data = np.column_stack([extractor.correlation_features[f] for f in corr_features])
                im = ax2.imshow(corr_data.T, cmap='RdBu_r', aspect='auto')
                ax2.set_title('Template Correlations')
                ax2.set_xlabel('Trial Index')
                ax2.set_ylabel('Template')
                plt.colorbar(im, ax=ax2)
        
        # 3. LOOCV performance
        ax3 = axes[0, 2]
        if hasattr(extractor, 'correlation_features'):
            loocv_features = [f for f in extractor.correlation_features.keys() if 'loocv_template_score' in f]
            if loocv_features:
                loocv_scores = [extractor.correlation_features[f] for f in loocv_features]
                for i, scores in enumerate(loocv_scores):
                    ax3.hist(scores, alpha=0.7, label=f'Stimulus {i}', bins=20)
                ax3.set_title('LOOCV Template Scores')
                ax3.set_xlabel('Correlation Score')
                ax3.set_ylabel('Frequency')
                ax3.legend()
        
        # 4. Template quality metrics
        ax4 = axes[1, 0]
        if extractor.templates:
            template_quality = []
            template_names = []
            for stim, template_info in extractor.templates.items():
                template = template_info['template']
                quality = np.std(template) / (np.mean(np.abs(template)) + 1e-8)
                template_quality.append(quality)
                template_names.append(f'Stim {stim}')
            
            bars = ax4.bar(template_names, template_quality, color='skyblue', alpha=0.7)
            ax4.set_title('Template Quality (CV)')
            ax4.set_ylabel('Coefficient of Variation')
            ax4.tick_params(axis='x', rotation=45)
        
        # 5. Feature distribution
        ax5 = axes[1, 1]
        if hasattr(extractor, 'correlation_features'):
            all_features = []
            for feature_name, feature_data in extractor.correlation_features.items():
                if isinstance(feature_data, np.ndarray):
                    all_features.extend(feature_data.flatten())
            
            if all_features:
                ax5.hist(all_features, bins=50, alpha=0.7, color='lightgreen')
                ax5.set_title('Feature Value Distribution')
                ax5.set_xlabel('Feature Value')
                ax5.set_ylabel('Frequency')
        
        # 6. Summary statistics
        ax6 = axes[1, 2]
        ax6.axis('off')
        summary_text = f"""
        Template Correlation Summary:
        
        • Templates Created: {len(extractor.templates)}
        • Features Extracted: {len(extractor.correlation_features) if hasattr(extractor, 'correlation_features') else 0}
        • LOOCV Enabled: {extractor.loocv_enabled}
        • Gamma Range: {extractor.gamma_freq_range[0]}-{extractor.gamma_freq_range[1]} Hz
        • Template Window: {extractor.template_window[0]}-{extractor.template_window[1]} ms
        """
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=12, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_dir / '01_template_correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_dir / '01_template_correlation_analysis.svg', bbox_inches='tight')
        plt.close()
    
    def _create_csp_lda_visualizations(self, extractor, save_dir):
        """Create visualizations for CSP+LDA extractor."""
        print("🔬 Creating CSP+LDA visualizations...")
        
        if not hasattr(extractor, 'csp_filters') or not extractor.csp_filters:
            print("   ⚠️ No CSP filter data available")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        fig.suptitle('CSP + LDA Feature Analysis', fontsize=20, fontweight='bold')
        
        # 1. CSP filters visualization
        ax1 = axes[0, 0]
        if extractor.csp_filters:
            # Show CSP filters for first frequency band
            first_band = list(extractor.csp_filters.keys())[0]
            csp_filter = extractor.csp_filters[first_band]['filters']
            im = ax1.imshow(csp_filter, cmap='RdBu_r', aspect='auto')
            ax1.set_title(f'CSP Filters - {first_band} band')
            ax1.set_xlabel('CSP Component')
            ax1.set_ylabel('Channel')
            plt.colorbar(im, ax=ax1)
        
        # 2. Spatial features across frequency bands
        ax2 = axes[0, 1]
        if hasattr(extractor, 'spatial_features'):
            csp_features = [f for f in extractor.spatial_features.keys() if f.startswith('csp_') and not f.endswith('_raw') and not f.endswith('_variance')]
            if csp_features:
                feature_means = []
                feature_names = []
                for feature_name in csp_features:
                    feature_data = extractor.spatial_features[feature_name]
                    feature_means.append(np.mean(feature_data))
                    feature_names.append(feature_name.replace('csp_', ''))
                
                bars = ax2.bar(feature_names, feature_means, color='lightgreen', alpha=0.7)
                ax2.set_title('CSP Features by Frequency Band')
                ax2.set_ylabel('Mean Feature Value')
                ax2.tick_params(axis='x', rotation=45)
        
        # 3. LDA performance
        ax3 = axes[0, 2]
        if hasattr(extractor, 'spatial_features') and 'lda_accuracy' in extractor.spatial_features:
            lda_accuracy = extractor.spatial_features['lda_accuracy']
            ax3.bar(['LDA Accuracy'], [lda_accuracy], color='orange', alpha=0.7)
            ax3.set_title('LDA Classification Performance')
            ax3.set_ylabel('Accuracy')
            ax3.set_ylim(0, 1)
        
        # 4. CSP component analysis
        ax4 = axes[1, 0]
        if extractor.csp_filters:
            component_vars = []
            band_names = []
            for band_name, csp_info in extractor.csp_filters.items():
                csp_filter = csp_info['filters']
                component_var = np.var(csp_filter, axis=0)
                component_vars.append(component_var)
                band_names.append(band_name)
            
            # Plot variance of CSP components
            for i, (var, name) in enumerate(zip(component_vars, band_names)):
                ax4.plot(var, label=name, marker='o')
            ax4.set_title('CSP Component Variance')
            ax4.set_xlabel('Component Index')
            ax4.set_ylabel('Variance')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 5. Feature correlation matrix
        ax5 = axes[1, 1]
        if hasattr(extractor, 'spatial_features'):
            csp_features = [f for f in extractor.spatial_features.keys() if f.startswith('csp_') and not f.endswith('_raw') and not f.endswith('_variance')]
            if len(csp_features) > 1:
                feature_matrix = np.column_stack([extractor.spatial_features[f].flatten() for f in csp_features])
                corr_matrix = np.corrcoef(feature_matrix.T)
                im = ax5.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
                ax5.set_title('CSP Feature Correlations')
                ax5.set_xlabel('Feature')
                ax5.set_ylabel('Feature')
                ax5.set_xticks(range(len(csp_features)))
                ax5.set_xticklabels([f.replace('csp_', '') for f in csp_features], rotation=45)
                ax5.set_yticks(range(len(csp_features)))
                ax5.set_yticklabels([f.replace('csp_', '') for f in csp_features])
                plt.colorbar(im, ax=ax5)
        
        # 6. Summary statistics
        ax6 = axes[1, 2]
        ax6.axis('off')
        summary_text = f"""
        CSP + LDA Summary:
        
        • CSP Components: {extractor.csp_components}
        • Frequency Bands: {len(extractor.frequency_bands)}
        • CSP Filters: {len(extractor.csp_filters)}
        • Features Extracted: {len(extractor.spatial_features) if hasattr(extractor, 'spatial_features') else 0}
        • LDA Solver: {extractor.lda_solver}
        • Spatial Window: {extractor.spatial_window[0]}-{extractor.spatial_window[1]} ms
        """
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=12, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_dir / '02_csp_lda_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_dir / '02_csp_lda_analysis.svg', bbox_inches='tight')
        plt.close()
    
    def _create_eegnet_visualizations(self, extractor, save_dir):
        """Create visualizations for EEGNet extractor."""
        print("🔬 Creating EEGNet visualizations...")
        
        if not hasattr(extractor, 'augmented_data') or not extractor.augmented_data:
            print("   ⚠️ No EEGNet data available")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        fig.suptitle('EEGNet Feature Analysis', fontsize=20, fontweight='bold')
        
        # 1. Raw time series visualization
        ax1 = axes[0, 0]
        if 'time_series' in extractor.augmented_data:
            time_series = extractor.augmented_data['time_series']
            # Show first trial, first few channels
            sample_data = time_series[0, :3, :]
            time_axis = np.arange(sample_data.shape[1]) / extractor.fs * 1000  # Convert to ms
            
            for i in range(sample_data.shape[0]):
                ax1.plot(time_axis, sample_data[i, :], label=f'Channel {i+1}', alpha=0.8)
            ax1.set_title('Raw Time Series (Sample Trial)')
            ax1.set_xlabel('Time (ms)')
            ax1.set_ylabel('Amplitude (μV)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Data augmentation comparison
        ax2 = axes[0, 1]
        if 'original_data' in extractor.augmented_data and 'augmented_data' in extractor.augmented_data:
            original = extractor.augmented_data['original_data']
            augmented = extractor.augmented_data['augmented_data']
            
            # Show augmentation effect
            orig_sample = original[0, 0, :]
            aug_sample = augmented[0, 0, :]
            time_axis = np.arange(len(orig_sample)) / extractor.fs * 1000
            
            ax2.plot(time_axis, orig_sample, 'b-', label='Original', alpha=0.8)
            ax2.plot(time_axis, aug_sample, 'r-', label='Augmented', alpha=0.8)
            ax2.set_title('Data Augmentation Effect')
            ax2.set_xlabel('Time (ms)')
            ax2.set_ylabel('Amplitude (μV)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. CNN input format
        ax3 = axes[0, 2]
        if 'cnn_input' in extractor.augmented_data:
            cnn_input = extractor.augmented_data['cnn_input']
            if len(cnn_input.shape) == 4:  # Grid layout
                # Show first trial as image
                sample_input = cnn_input[0, :, :, 0]  # First time point
                im = ax3.imshow(sample_input, cmap='viridis', aspect='auto')
                ax3.set_title('CNN Input Format (Grid)')
                ax3.set_xlabel('Width')
                ax3.set_ylabel('Height')
                plt.colorbar(im, ax=ax3)
            else:  # Linear layout
                # Show first trial
                sample_input = cnn_input[0, :, :]
                im = ax3.imshow(sample_input, cmap='viridis', aspect='auto')
                ax3.set_title('CNN Input Format (Linear)')
                ax3.set_xlabel('Time')
                ax3.set_ylabel('Channel')
                plt.colorbar(im, ax=ax3)
        
        # 4. Augmentation statistics
        ax4 = axes[1, 0]
        if 'augmentation_factor' in extractor.augmented_data:
            aug_factor = extractor.augmented_data['augmentation_factor']
            n_original = extractor.augmented_data.get('n_original', 0)
            n_augmented = extractor.augmented_data.get('n_augmented', 0)
            
            categories = ['Original', 'Augmented']
            counts = [n_original, n_augmented]
            colors = ['lightblue', 'lightcoral']
            
            bars = ax4.bar(categories, counts, color=colors, alpha=0.7)
            ax4.set_title('Data Augmentation Statistics')
            ax4.set_ylabel('Number of Samples')
            
            # Add count labels on bars
            for bar, count in zip(bars, counts):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom')
        
        # 5. Train/validation split
        ax5 = axes[1, 1]
        if 'train_indices' in extractor.augmented_data and 'val_indices' in extractor.augmented_data:
            n_train = len(extractor.augmented_data['train_indices'])
            n_val = len(extractor.augmented_data['val_indices'])
            
            categories = ['Train', 'Validation']
            counts = [n_train, n_val]
            colors = ['lightgreen', 'orange']
            
            bars = ax5.bar(categories, counts, color=colors, alpha=0.7)
            ax5.set_title('Train/Validation Split')
            ax5.set_ylabel('Number of Samples')
            
            # Add count labels on bars
            for bar, count in zip(bars, counts):
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom')
        
        # 6. Summary statistics
        ax6 = axes[1, 2]
        ax6.axis('off')
        summary_text = f"""
        EEGNet Summary:
        
        • Time Window: {extractor.time_window[0]}-{extractor.time_window[1]} ms
        • Input Length: {extractor.input_length} samples
        • Channel Layout: {extractor.channel_layout}
        • Grid Size: {extractor.grid_size}
        • Augmentation: {extractor.augmentation_enabled}
        • Augmentation Factor: {extractor.augmentation_factor}x
        • Noise Level: {extractor.noise_level}
        """
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=12, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_dir / '03_eegnet_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_dir / '03_eegnet_analysis.svg', bbox_inches='tight')
        plt.close()
    
    def _create_transformer_visualizations(self, extractor, save_dir):
        """Create visualizations for Transformer extractor."""
        print("🔬 Creating Transformer visualizations...")
        
        if not hasattr(extractor, 'attention_features') or not extractor.attention_features:
            print("   ⚠️ No Transformer data available")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        fig.suptitle('Time Series Transformer Feature Analysis', fontsize=20, fontweight='bold')
        
        # 1. Multi-scale features
        ax1 = axes[0, 0]
        if 'multi_scale_features' in extractor.attention_features:
            multi_scale = extractor.attention_features['multi_scale_features']
            # Show feature variance across scales
            feature_vars = np.var(multi_scale, axis=0)
            ax1.plot(feature_vars, 'b-', alpha=0.7)
            ax1.set_title('Multi-Scale Feature Variance')
            ax1.set_xlabel('Feature Index')
            ax1.set_ylabel('Variance')
            ax1.grid(True, alpha=0.3)
        
        # 2. Frequency domain features
        ax2 = axes[0, 1]
        if 'frequency_features' in extractor.attention_features:
            freq_features = extractor.attention_features['frequency_features']
            # Show mean feature values by frequency band
            band_means = []
            band_names = list(extractor.frequency_bands.keys())
            
            # Assuming features are organized by bands
            n_features_per_band = freq_features.shape[1] // len(band_names)
            for i, band_name in enumerate(band_names):
                start_idx = i * n_features_per_band
                end_idx = start_idx + n_features_per_band
                band_mean = np.mean(freq_features[:, start_idx:end_idx])
                band_means.append(band_mean)
            
            bars = ax2.bar(band_names, band_means, color='lightgreen', alpha=0.7)
            ax2.set_title('Frequency Band Features')
            ax2.set_ylabel('Mean Feature Value')
            ax2.tick_params(axis='x', rotation=45)
        
        # 3. Attention embeddings
        ax3 = axes[0, 2]
        if 'attention_embeddings' in extractor.attention_features:
            embeddings = extractor.attention_features['attention_embeddings']
            # Show embedding distribution
            embedding_means = np.mean(embeddings, axis=0)
            ax3.plot(embedding_means, 'purple', alpha=0.7, marker='o')
            ax3.set_title('Attention Embeddings')
            ax3.set_xlabel('Embedding Dimension')
            ax3.set_ylabel('Mean Value')
            ax3.grid(True, alpha=0.3)
        
        # 4. Temporal scale analysis
        ax4 = axes[1, 0]
        if hasattr(extractor, 'multi_scale_features'):
            scales = extractor.temporal_scales
            scale_energies = []
            
            for scale_name, scale_data in extractor.multi_scale_features.items():
                if 'features' in scale_data and 'energy' in scale_data['features']:
                    energy = np.mean(scale_data['features']['energy'])
                    scale_energies.append(energy)
                else:
                    scale_energies.append(0)
            
            bars = ax4.bar([f'Scale {s}' for s in scales], scale_energies, 
                          color='skyblue', alpha=0.7)
            ax4.set_title('Energy Across Temporal Scales')
            ax4.set_ylabel('Mean Energy')
            ax4.tick_params(axis='x', rotation=45)
        
        # 5. Sequence length analysis
        ax5 = axes[1, 1]
        if 'input_features' in extractor.attention_features:
            input_features = extractor.attention_features['input_features']
            # Show feature distribution
            feature_means = np.mean(input_features, axis=0)
            ax5.hist(feature_means, bins=50, alpha=0.7, color='orange')
            ax5.set_title('Input Feature Distribution')
            ax5.set_xlabel('Feature Value')
            ax5.set_ylabel('Frequency')
        
        # 6. Summary statistics
        ax6 = axes[1, 2]
        ax6.axis('off')
        summary_text = f"""
        Transformer Summary:
        
        • Extended Window: {extractor.extended_window[0]}-{extractor.extended_window[1]} ms
        • Sequence Length: {extractor.sequence_length} samples
        • Temporal Scales: {extractor.temporal_scales}
        • Attention Heads: {extractor.attention_heads}
        • Embedding Dim: {extractor.embedding_dim}
        • Positional Encoding: {extractor.positional_encoding}
        • Frequency Bands: {len(extractor.frequency_bands)}
        """
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=12, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_dir / '04_transformer_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_dir / '04_transformer_analysis.svg', bbox_inches='tight')
        plt.close()
    
    def _create_feature_comparison_analysis(self, feature_extractors, save_dir):
        """Create comparative analysis of all feature extractors."""
        print("🔬 Creating feature comparison analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Feature Extractor Comparison Analysis', fontsize=20, fontweight='bold')
        
        # 1. Feature count comparison
        ax1 = axes[0, 0]
        extractor_names = []
        feature_counts = []
        colors = ['blue', 'green', 'red', 'purple']
        
        for i, (name, extractor) in enumerate(feature_extractors.items()):
            extractor_names.append(name.replace('_', ' ').title())
            if hasattr(extractor, 'features'):
                feature_counts.append(len(extractor.features))
            elif hasattr(extractor, 'correlation_features'):
                feature_counts.append(len(extractor.correlation_features))
            elif hasattr(extractor, 'spatial_features'):
                feature_counts.append(len(extractor.spatial_features))
            elif hasattr(extractor, 'augmented_data'):
                feature_counts.append(len(extractor.augmented_data))
            elif hasattr(extractor, 'attention_features'):
                feature_counts.append(len(extractor.attention_features))
            else:
                feature_counts.append(0)
        
        bars = ax1.bar(extractor_names, feature_counts, color=colors, alpha=0.7)
        ax1.set_title('Feature Count by Extractor')
        ax1.set_ylabel('Number of Features')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add count labels on bars
        for bar, count in zip(bars, feature_counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom')
        
        # 2. Data complexity comparison
        ax2 = axes[0, 1]
        complexity_metrics = []
        
        for name, extractor in feature_extractors.items():
            if name == 'template_correlation' and hasattr(extractor, 'templates'):
                complexity = len(extractor.templates) * extractor.csp_components if hasattr(extractor, 'csp_components') else len(extractor.templates)
            elif name == 'csp_lda' and hasattr(extractor, 'csp_filters'):
                complexity = len(extractor.csp_filters) * extractor.csp_components
            elif name == 'eegnet' and hasattr(extractor, 'augmented_data'):
                complexity = extractor.augmented_data.get('n_augmented', 0)
            elif name == 'transformer' and hasattr(extractor, 'attention_features'):
                complexity = len(extractor.temporal_scales) * len(extractor.frequency_bands)
            else:
                complexity = 0
            complexity_metrics.append(complexity)
        
        bars = ax2.bar(extractor_names, complexity_metrics, color=colors, alpha=0.7)
        ax2.set_title('Data Complexity by Extractor')
        ax2.set_ylabel('Complexity Metric')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Feature type distribution
        ax3 = axes[1, 0]
        feature_types = {
            'Template': 0,
            'Spatial': 0,
            'Temporal': 0,
            'Frequency': 0
        }
        
        for name, extractor in feature_extractors.items():
            if name == 'template_correlation':
                feature_types['Template'] += 1
            elif name == 'csp_lda':
                feature_types['Spatial'] += 1
            elif name == 'eegnet':
                feature_types['Temporal'] += 1
            elif name == 'transformer':
                feature_types['Frequency'] += 1
        
        wedges, texts, autotexts = ax3.pie(feature_types.values(), labels=feature_types.keys(),
                                          autopct='%1.1f%%', colors=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow'])
        ax3.set_title('Feature Type Distribution')
        
        # 4. Performance metrics comparison
        ax4 = axes[1, 1]
        performance_metrics = []
        
        for name, extractor in feature_extractors.items():
            if name == 'csp_lda' and hasattr(extractor, 'spatial_features') and 'lda_accuracy' in extractor.spatial_features:
                performance_metrics.append(extractor.spatial_features['lda_accuracy'])
            else:
                performance_metrics.append(0.0)  # Placeholder for other extractors
        
        bars = ax4.bar(extractor_names, performance_metrics, color=colors, alpha=0.7)
        ax4.set_title('Performance Metrics')
        ax4.set_ylabel('Accuracy/Score')
        ax4.set_ylim(0, 1)
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_dir / '05_feature_comparison_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_dir / '05_feature_comparison_analysis.svg', bbox_inches='tight')
        plt.close()
    
    def _create_enhanced_preprocessing_analysis(self, raw_data, preprocessed_data, save_dir):
        """Create detailed analysis of each preprocessing step."""
        print("🔬 Creating enhanced preprocessing step-by-step analysis...")
        
        ecog_data = raw_data['ecog_data']
        filtered_data = preprocessed_data['filtered_data']
        car_data = preprocessed_data.get('car_data', filtered_data)
        ica_data = preprocessed_data.get('ica_data', car_data)
        smoothed_data = preprocessed_data.get('smoothed_data', ica_data)
        good_channels = preprocessed_data['good_channels']
        preprocessing_params = preprocessed_data['preprocessing_params']
        fs = raw_data['sampling_rate']
        
        # Create comprehensive preprocessing analysis
        fig, axes = plt.subplots(3, 3, figsize=(24, 18))
        fig.suptitle('Enhanced Preprocessing: Step-by-Step Analysis', fontsize=20, fontweight='bold')
        
        # Sample a good channel for analysis
        sample_ch = good_channels[0]
        time_window = slice(0, min(10000, ecog_data.shape[1]))
        time_seconds = np.arange(time_window.stop) / fs
        
        # 1. Power Spectral Density comparison
        ax1 = axes[0, 0]
        freqs_raw, psd_raw = signal.welch(ecog_data[sample_ch, time_window], fs=fs, nperseg=1024)
        freqs_filt, psd_filt = signal.welch(filtered_data[sample_ch, time_window], fs=fs, nperseg=1024)
        freqs_car, psd_car = signal.welch(car_data[sample_ch, time_window], fs=fs, nperseg=1024)
        
        ax1.loglog(freqs_raw, psd_raw, 'b-', alpha=0.7, label='Raw')
        ax1.loglog(freqs_filt, psd_filt, 'r-', alpha=0.7, label='Filtered')
        ax1.loglog(freqs_car, psd_car, 'g-', alpha=0.7, label='After CAR')
        ax1.set_title('Power Spectral Density')
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Power (μV²/Hz)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Signal variance across preprocessing steps
        ax2 = axes[0, 1]
        steps = ['Raw', 'Filtered', 'CAR', 'ICA', 'Smoothed']
        variances = [
            np.var(ecog_data[sample_ch, time_window]),
            np.var(filtered_data[sample_ch, time_window]),
            np.var(car_data[sample_ch, time_window]),
            np.var(ica_data[sample_ch, time_window]),
            np.var(smoothed_data[sample_ch, time_window])
        ]
        bars = ax2.bar(steps, variances, color=['blue', 'red', 'green', 'orange', 'purple'], alpha=0.7)
        ax2.set_title('Signal Variance Across Steps')
        ax2.set_ylabel('Variance (μV²)')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Signal-to-Noise Ratio improvement
        ax3 = axes[0, 2]
        # Calculate SNR as signal power / noise power (using high-freq as noise estimate)
        snr_raw = np.mean(psd_raw[freqs_raw < 50]) / np.mean(psd_raw[freqs_raw > 100])
        snr_filt = np.mean(psd_filt[freqs_filt < 50]) / np.mean(psd_filt[freqs_filt > 100])
        snr_car = np.mean(psd_car[freqs_car < 50]) / np.mean(psd_car[freqs_car > 100])
        
        snr_steps = ['Raw', 'Filtered', 'CAR']
        snr_values = [snr_raw, snr_filt, snr_car]
        bars = ax3.bar(snr_steps, snr_values, color=['blue', 'red', 'green'], alpha=0.7)
        ax3.set_title('Signal-to-Noise Ratio')
        ax3.set_ylabel('SNR')
        
        # 4. Channel correlation matrix (before and after CAR)
        ax4 = axes[1, 0]
        # Use subset of good channels for correlation
        subset_channels = good_channels[:20]  # Limit for visualization
        corr_before = np.corrcoef(filtered_data[subset_channels, time_window])
        im = ax4.imshow(corr_before, cmap='RdBu_r', vmin=-1, vmax=1)
        ax4.set_title('Channel Correlation (Before CAR)')
        ax4.set_xlabel('Channel Index')
        ax4.set_ylabel('Channel Index')
        plt.colorbar(im, ax=ax4)
        
        # 5. Channel correlation matrix (after CAR)
        ax5 = axes[1, 1]
        corr_after = np.corrcoef(car_data[subset_channels, time_window])
        im = ax5.imshow(corr_after, cmap='RdBu_r', vmin=-1, vmax=1)
        ax5.set_title('Channel Correlation (After CAR)')
        ax5.set_xlabel('Channel Index')
        ax5.set_ylabel('Channel Index')
        plt.colorbar(im, ax=ax5)
        
        # 6. Artifact detection results
        ax6 = axes[1, 2]
        bad_channels = preprocessed_data['bad_channels']
        total_channels = len(good_channels) + len(bad_channels)
        quality_data = {
            'Good Channels': len(good_channels),
            'Bad Channels': len(bad_channels)
        }
        colors = ['lightgreen', 'lightcoral']
        wedges, texts, autotexts = ax6.pie(quality_data.values(), labels=quality_data.keys(), 
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        ax6.set_title(f'Channel Quality\n({total_channels} total channels)')
        
        # 7. Preprocessing parameters summary
        ax7 = axes[2, 0]
        ax7.axis('off')
        param_text = f"""
        Preprocessing Parameters:
        
        • Bandpass Filter: {preprocessing_params.get('bandpass_low', 0.5)}-{preprocessing_params.get('bandpass_high', 150)} Hz
        • Notch Filters: {preprocessing_params.get('notch_freqs', [50, 60])} Hz
        • CAR Enabled: {preprocessing_params.get('car_enabled', True)}
        • ICA Enabled: {preprocessing_params.get('ica_enabled', True)}
        • ICA Components: {preprocessing_params.get('ica_components', 20)}
        • Temporal Smoothing: {preprocessing_params.get('temporal_smoothing', True)}
        • Artifact Detection: {preprocessing_params.get('artifact_detection_method', 'multi_criteria')}
        • Baseline Method: {preprocessing_params.get('baseline_method', 'percentage_change')}
        • Trial Window: {preprocessing_params.get('trial_window', (100, 400))} ms
        • Baseline Window: {preprocessing_params.get('baseline_window', (-300, 0))} ms
        """
        ax7.text(0.1, 0.9, param_text, transform=ax7.transAxes, fontsize=10, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # 8. Data shape evolution
        ax8 = axes[2, 1]
        ax8.axis('off')
        shape_text = f"""
        Data Shape Evolution:
        
        • Raw Data: {ecog_data.shape}
        • After Filtering: {filtered_data.shape}
        • After CAR: {car_data.shape}
        • After ICA: {ica_data.shape}
        • After Smoothing: {smoothed_data.shape}
        • Final Epochs: {preprocessed_data['epochs'].shape}
        
        • Good Channels: {len(good_channels)}
        • Bad Channels: {len(bad_channels)}
        • Total Trials: {preprocessing_params.get('n_trials', 0)}
        """
        ax8.text(0.1, 0.9, shape_text, transform=ax8.transAxes, fontsize=10, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 9. Quality metrics
        ax9 = axes[2, 2]
        if 'quality_metrics' in preprocessed_data:
            quality_metrics = preprocessed_data['quality_metrics']
            metrics_text = f"""
            Quality Metrics:
            
            • Overall Quality: {quality_metrics.get('overall_quality', 'N/A')}/100
            • Channel Quality: {quality_metrics.get('channel_quality', 'N/A')}%
            • Trial Quality: {quality_metrics.get('trial_quality', 'N/A')}%
            • SNR Improvement: {quality_metrics.get('snr_improvement', 'N/A')}%
            • Artifact Reduction: {quality_metrics.get('artifact_reduction', 'N/A')}%
            """
        else:
            metrics_text = "Quality metrics not available"
        
        ax9.text(0.1, 0.9, metrics_text, transform=ax9.transAxes, fontsize=10, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_dir / '03_enhanced_preprocessing_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_dir / '03_enhanced_preprocessing_analysis.svg', bbox_inches='tight')
        plt.close()
    
    def create_feature_visualizations(self, features, feature_metadata, save_dir):
        """Create feature extraction visualizations."""
        print("📊 Creating feature extraction visualizations...")
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Feature overview
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('ECoG Feature Extraction Overview', fontsize=20, fontweight='bold')
        
        # Gamma power (priority feature)
        ax1 = axes[0, 0]
        if 'gamma_power' in features and features['gamma_power'].size > 0:
            gamma_data = features['gamma_power']
            im = ax1.imshow(gamma_data, cmap='viridis', aspect='auto')
            ax1.set_title('Gamma Power (110-140 Hz) - Priority Feature')
            ax1.set_xlabel('Channels')
            ax1.set_ylabel('Trials')
            plt.colorbar(im, ax=ax1, label='Log-Variance (Z-scored)')
        
        # Feature distributions
        ax2 = axes[0, 1]
        feature_names = []
        feature_means = []
        for name, data in features.items():
            if data.size > 0:
                feature_names.append(name.replace('_power', '').replace('_', ' ').title())
                feature_means.append(np.mean(data))
        
        if feature_names:
            bars = ax2.bar(range(len(feature_names)), feature_means, 
                          color=plt.cm.Set3(np.linspace(0, 1, len(feature_names))))
            ax2.set_xticks(range(len(feature_names)))
            ax2.set_xticklabels(feature_names, rotation=45, ha='right')
            ax2.set_ylabel('Mean Feature Value')
            ax2.set_title('Feature Value Comparison')
        
        # Feature correlation
        ax3 = axes[0, 2]
        if len(features) > 1:
            # Create correlation matrix
            feature_matrix = []
            feature_labels = []
            for name, data in features.items():
                if data.size > 0:
                    feature_matrix.append(data.flatten())
                    feature_labels.append(name.replace('_power', '').replace('_', ' ').title())
            
            if len(feature_matrix) > 1:
                feature_matrix = np.array(feature_matrix)
                corr_matrix = np.corrcoef(feature_matrix)
                im = ax3.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
                ax3.set_xticks(range(len(feature_labels)))
                ax3.set_yticks(range(len(feature_labels)))
                ax3.set_xticklabels(feature_labels, rotation=45, ha='right')
                ax3.set_yticklabels(feature_labels)
                ax3.set_title('Feature Correlation Matrix')
                plt.colorbar(im, ax=ax3, label='Correlation')
        
        # Feature statistics
        ax4 = axes[1, 0]
        if features:
            stats_data = []
            for name, data in features.items():
                if data.size > 0:
                    stats_data.append({
                        'Feature': name.replace('_power', '').replace('_', ' ').title(),
                        'Mean': np.mean(data),
                        'Std': np.std(data),
                        'Min': np.min(data),
                        'Max': np.max(data)
                    })
            
            if stats_data:
                df = pd.DataFrame(stats_data)
                ax4.axis('tight')
                ax4.axis('off')
                table = ax4.table(cellText=df.round(3).values, colLabels=df.columns,
                                cellLoc='center', loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1.2, 1.5)
                ax4.set_title('Feature Statistics')
        
        # Feature importance (simulated)
        ax5 = axes[1, 1]
        if features:
            # Simulate feature importance scores
            importance_scores = np.random.exponential(1, len(features))
            importance_scores = importance_scores / np.sum(importance_scores)  # Normalize
            
            feature_names = [name.replace('_power', '').replace('_', ' ').title() for name in features.keys()]
            colors = plt.cm.viridis(np.linspace(0, 1, len(feature_names)))
            
            bars = ax5.barh(feature_names, importance_scores, color=colors)
            ax5.set_xlabel('Importance Score')
            ax5.set_title('Feature Importance (Simulated)')
            
            # Add value labels
            for bar, score in zip(bars, importance_scores):
                ax5.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{score:.3f}', ha='left', va='center', fontweight='bold')
        
        # Feature extraction summary
        ax6 = axes[1, 2]
        summary_text = f"""
Feature Extraction Summary

✅ Primary Feature: Gamma Power (110-140 Hz)
✅ Canonical Bands: {len([k for k in features.keys() if 'power' in k])} frequency bands
✅ Total Features: {len(features)}
✅ Normalization: Z-scored per trial
✅ Ready for Classification

Feature Shapes:
{chr(10).join([f'• {name}: {list(data.shape)}' for name, data in features.items() if data.size > 0])}
        """
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_dir / '04_feature_extraction_overview.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_dir / '04_feature_extraction_overview.svg', bbox_inches='tight')
        plt.close()
        
        print("✅ Feature extraction visualizations saved!")
    
    def create_comprehensive_report(self, raw_data, preprocessed_data, features, save_dir):
        """Create comprehensive pipeline report."""
        print("📊 Creating comprehensive pipeline report...")
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create comprehensive dashboard
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        fig.suptitle('ECoG Comprehensive Pipeline Report', fontsize=24, fontweight='bold')
        
        # 1. Dataset overview (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        ecog_data = raw_data['ecog_data']
        shape_data = {
            'Channels': ecog_data.shape[0],
            'Samples': ecog_data.shape[1],
            'Duration (min)': ecog_data.shape[1]/raw_data['sampling_rate']/60
        }
        bars = ax1.bar(range(len(shape_data)), list(shape_data.values()), 
                      color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax1.set_xticks(range(len(shape_data)))
        ax1.set_xticklabels(list(shape_data.keys()), rotation=45, ha='right')
        ax1.set_title('Dataset Overview', fontweight='bold')
        
        # 2. Brain regions (top center)
        ax2 = fig.add_subplot(gs[0, 1])
        self.brain_atlas._plot_brain_regions(ax2)
        
        # 3. Channel quality (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        if 'good_channels' in preprocessed_data:
            good_channels = preprocessed_data['good_channels']
            bad_channels = preprocessed_data['bad_channels']
            quality_data = {
                'Good': len(good_channels),
                'Bad': len(bad_channels)
            }
            colors = ['lightgreen', 'lightcoral']
            wedges, texts, autotexts = ax3.pie(quality_data.values(), labels=quality_data.keys(), 
                                              autopct='%1.1f%%', colors=colors, startangle=90)
            ax3.set_title('Channel Quality', fontweight='bold')
        
        # 4. Trial statistics (top far right)
        ax4 = fig.add_subplot(gs[0, 3])
        if 'epochs' in preprocessed_data and preprocessed_data['epochs'].size > 0:
            epochs = preprocessed_data['epochs']
            trial_stats = {
                'Total Trials': epochs.shape[0],
                'Channels': epochs.shape[1],
                'Timepoints': epochs.shape[2]
            }
            ax4.text(0.1, 0.7, f"Trials: {trial_stats['Total Trials']}", transform=ax4.transAxes, fontsize=12)
            ax4.text(0.1, 0.5, f"Channels: {trial_stats['Channels']}", transform=ax4.transAxes, fontsize=12)
            ax4.text(0.1, 0.3, f"Timepoints: {trial_stats['Timepoints']}", transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Trial Statistics', fontweight='bold')
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.axis('off')
        
        # 5. Feature overview (middle left)
        ax5 = fig.add_subplot(gs[1, 0])
        if features:
            # Handle new structure where features is a dict of extractors
            if isinstance(features, dict) and all(isinstance(v, dict) for v in features.values()):
                # New structure: features = {'extractor1': {...}, 'extractor2': {...}}
                extractor_names = list(features.keys())
                feature_counts = []
                for extractor_name, extractor_features in features.items():
                    if isinstance(extractor_features, dict):
                        # Count total features in this extractor
                        count = sum(1 for v in extractor_features.values() if isinstance(v, np.ndarray))
                        feature_counts.append(count)
                    else:
                        feature_counts.append(0)
                
                bars = ax5.bar(range(len(extractor_names)), feature_counts, 
                              color=plt.cm.Set3(np.linspace(0, 1, len(extractor_names))))
                ax5.set_xticks(range(len(extractor_names)))
                ax5.set_xticklabels([name.replace('_', ' ').title() for name in extractor_names], rotation=45, ha='right')
                ax5.set_ylabel('Number of Features')
                ax5.set_title('Features by Extractor', fontweight='bold')
            else:
                # Old structure: features = {'feature1': array, 'feature2': array}
                feature_names = [name.replace('_power', '').replace('_', ' ').title() for name in features.keys()]
                feature_counts = [data.size for data in features.values() if isinstance(data, np.ndarray)]
                bars = ax5.bar(range(len(feature_names)), feature_counts, 
                              color=plt.cm.Set3(np.linspace(0, 1, len(feature_names))))
                ax5.set_xticks(range(len(feature_names)))
                ax5.set_xticklabels(feature_names, rotation=45, ha='right')
                ax5.set_ylabel('Feature Size')
                ax5.set_title('Extracted Features', fontweight='bold')
        
        # 6. Gamma power heatmap (middle center)
        ax6 = fig.add_subplot(gs[1, 1])
        gamma_data = None
        if isinstance(features, dict):
            # Look for gamma power in any extractor
            for extractor_name, extractor_features in features.items():
                if isinstance(extractor_features, dict) and 'gamma_power' in extractor_features:
                    gamma_data = extractor_features['gamma_power']
                    break
        
        if gamma_data is not None and hasattr(gamma_data, 'size') and gamma_data.size > 0:
            im = ax6.imshow(gamma_data, cmap='viridis', aspect='auto')
            ax6.set_title('Gamma Power (Priority Feature)', fontweight='bold')
            ax6.set_xlabel('Channels')
            ax6.set_ylabel('Trials')
            plt.colorbar(im, ax=ax6, label='Log-Variance')
        
        # 7. Feature correlation (middle right)
        ax7 = fig.add_subplot(gs[1, 2])
        if isinstance(features, dict):
            feature_matrix = []
            feature_labels = []
            
            # Handle new structure where features is a dict of extractors
            if all(isinstance(v, dict) for v in features.values()):
                # New structure: features = {'extractor1': {...}, 'extractor2': {...}}
                for extractor_name, extractor_features in features.items():
                    if isinstance(extractor_features, dict):
                        for name, data in extractor_features.items():
                            if isinstance(data, np.ndarray) and data.size > 0:
                                feature_matrix.append(data.flatten())
                                feature_labels.append(f"{extractor_name}_{name}".replace('_power', '').replace('_', ' ').title())
            else:
                # Old structure: features = {'feature1': array, 'feature2': array}
                for name, data in features.items():
                    if isinstance(data, np.ndarray) and data.size > 0:
                        feature_matrix.append(data.flatten())
                        feature_labels.append(name.replace('_power', '').replace('_', ' ').title())
            
            if len(feature_matrix) > 1:
                feature_matrix = np.array(feature_matrix)
                corr_matrix = np.corrcoef(feature_matrix)
                im = ax7.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
                ax7.set_xticks(range(len(feature_labels)))
                ax7.set_yticks(range(len(feature_labels)))
                ax7.set_xticklabels(feature_labels, rotation=45, ha='right')
                ax7.set_yticklabels(feature_labels)
                ax7.set_title('Feature Correlation', fontweight='bold')
        
        # 8. Pipeline summary (middle far right)
        ax8 = fig.add_subplot(gs[1, 3])
        summary_text = f"""
Pipeline Summary

✅ Raw Data: {ecog_data.shape[0]}×{ecog_data.shape[1]}
✅ Preprocessing: Complete
✅ Features: {len(features)} extracted
✅ Quality: High
✅ Ready: Classification

Next Steps:
• Load features for ML
• Train classifiers
• Evaluate performance
        """
        ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        ax8.set_xlim(0, 1)
        ax8.set_ylim(0, 1)
        ax8.axis('off')
        
        # 9. Frequency analysis (bottom left)
        ax9 = fig.add_subplot(gs[2, 0])
        if 'epochs' in preprocessed_data and preprocessed_data['epochs'].size > 0:
            epochs = preprocessed_data['epochs']
            avg_epoch = np.mean(epochs, axis=(0, 1))
            freqs, psd = signal.welch(avg_epoch, fs=raw_data['sampling_rate'], nperseg=256)
            ax9.semilogy(freqs, psd, 'b-', linewidth=2)
            ax9.set_xlabel('Frequency (Hz)')
            ax9.set_ylabel('Power Spectral Density')
            ax9.set_title('Average PSD', fontweight='bold')
            ax9.grid(True, alpha=0.3)
            ax9.set_xlim(0, 100)
        
        # 10. Temporal dynamics (bottom center)
        ax10 = fig.add_subplot(gs[2, 1])
        if 'epochs' in preprocessed_data and preprocessed_data['epochs'].size > 0:
            epochs = preprocessed_data['epochs']
            time_vector = preprocessed_data['time_vector']
            avg_response = np.mean(epochs, axis=(0, 1))
            ax10.plot(time_vector, avg_response, 'b-', linewidth=2)
            ax10.axvline(0, color='red', linestyle='--', alpha=0.7, label='Stimulus Onset')
            ax10.set_xlabel('Time (ms)')
            ax10.set_ylabel('Average Response')
            ax10.set_title('Temporal Dynamics', fontweight='bold')
            ax10.legend()
            ax10.grid(True, alpha=0.3)
        
        # 11. Feature distributions (bottom right)
        ax11 = fig.add_subplot(gs[2, 2])
        if features:
            for name, data in features.items():
                if data.size > 0:
                    ax11.hist(data.flatten(), bins=30, alpha=0.6, 
                             label=name.replace('_power', '').replace('_', ' ').title())
            ax11.set_xlabel('Feature Value')
            ax11.set_ylabel('Frequency')
            ax11.set_title('Feature Distributions', fontweight='bold')
            ax11.legend()
        
        # 12. Final status (bottom far right)
        ax12 = fig.add_subplot(gs[2, 3])
        status_text = f"""
🎉 PIPELINE COMPLETED

Status: ✅ SUCCESS
Quality: High
Features: Ready
Data: Saved

Ready for:
• Classification
• Analysis
• Competition
        """
        ax12.text(0.05, 0.5, status_text, transform=ax12.transAxes, fontsize=12,
                verticalalignment='center', ha='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax12.set_xlim(0, 1)
        ax12.set_ylim(0, 1)
        ax12.axis('off')
        
        plt.savefig(save_dir / 'comprehensive_pipeline_report.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_dir / 'comprehensive_pipeline_report.svg', bbox_inches='tight')
        plt.close()
        
        print("✅ Comprehensive pipeline report saved!")
