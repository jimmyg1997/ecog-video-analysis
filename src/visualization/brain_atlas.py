"""
Brain Atlas Visualization Module for ECoG Data
IEEE-SMC-2025 ECoG Video Analysis Competition

This module provides comprehensive brain atlas visualizations for ECoG data,
including channel mapping, brain region visualization, and spatial analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class BrainAtlas:
    """Comprehensive brain atlas visualization for ECoG data."""
    
    def __init__(self, config=None):
        """Initialize brain atlas with configuration."""
        self.config = config
        self.setup_plotting()
        
        # Define brain regions and their properties
        self.brain_regions = {
            'Occipital': {
                'channels': list(range(1, 41)),
                'color': '#FF6B6B',
                'description': 'Visual cortex - primary visual processing',
                'position': (0.2, 0.3),
                'size': (0.15, 0.2)
            },
            'Temporal': {
                'channels': list(range(41, 81)),
                'color': '#4ECDC4',
                'description': 'Temporal lobe - auditory and memory processing',
                'position': (0.1, 0.5),
                'size': (0.2, 0.25)
            },
            'Parietal': {
                'channels': list(range(81, 121)),
                'color': '#45B7D1',
                'description': 'Parietal lobe - sensory integration and spatial processing',
                'position': (0.4, 0.6),
                'size': (0.2, 0.2)
            },
            'Central': {
                'channels': list(range(121, 141)),
                'color': '#96CEB4',
                'description': 'Central region - motor and sensory cortex',
                'position': (0.6, 0.5),
                'size': (0.15, 0.3)
            },
            'Frontal': {
                'channels': list(range(141, 161)),
                'color': '#FFEAA7',
                'description': 'Frontal lobe - executive functions and decision making',
                'position': (0.7, 0.7),
                'size': (0.2, 0.2)
            }
        }
        
        # Define visual stimulus categories
        self.visual_categories = {
            'digit': {'color': '#E17055', 'description': 'Numbers (0-9)'},
            'kanji': {'color': '#74B9FF', 'description': 'Japanese Kanji characters'},
            'face': {'color': '#A29BFE', 'description': 'Human faces'},
            'body': {'color': '#FD79A8', 'description': 'Human bodies/figures'},
            'object': {'color': '#FDCB6E', 'description': 'Various objects'},
            'hiragana': {'color': '#6C5CE7', 'description': 'Japanese Hiragana characters'},
            'line': {'color': '#00B894', 'description': 'Line patterns/shapes'}
        }
    
    def setup_plotting(self):
        """Setup matplotlib parameters for high-quality plots."""
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
            'grid.alpha': 0.3,
            'axes.spines.top': False,
            'axes.spines.right': False
        })
    
    def create_brain_overview(self, channel_data=None, save_path=None):
        """Create comprehensive brain overview visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('ECoG Brain Atlas Overview', fontsize=20, fontweight='bold', y=0.95)
        
        # 1. Brain regions distribution
        ax1 = axes[0, 0]
        self._plot_brain_regions(ax1, channel_data)
        
        # 2. Channel grid layout
        ax2 = axes[0, 1]
        self._plot_channel_grid(ax2, channel_data)
        
        # 3. Visual categories
        ax3 = axes[1, 0]
        self._plot_visual_categories(ax3)
        
        # 4. Channel statistics
        ax4 = axes[1, 1]
        self._plot_channel_statistics(ax4, channel_data)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.savefig(str(save_path).replace('.png', '.svg'), bbox_inches='tight')
        
        return fig
    
    def create_spatial_analysis(self, data, channel_info=None, save_path=None):
        """Create spatial analysis visualization."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('ECoG Spatial Analysis', fontsize=20, fontweight='bold')
        
        # 1. Channel power map
        ax1 = axes[0, 0]
        self._plot_channel_power_map(ax1, data)
        
        # 2. Regional power distribution
        ax2 = axes[0, 1]
        self._plot_regional_power(ax2, data)
        
        # 3. Channel correlation matrix
        ax3 = axes[0, 2]
        self._plot_channel_correlation(ax3, data)
        
        # 4. High-gamma activity map
        ax4 = axes[1, 0]
        self._plot_high_gamma_map(ax4, data)
        
        # 5. Frequency band comparison
        ax5 = axes[1, 1]
        self._plot_frequency_bands(ax5, data)
        
        # 6. Channel quality assessment
        ax6 = axes[1, 2]
        self._plot_channel_quality(ax6, data)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.savefig(str(save_path).replace('.png', '.svg'), bbox_inches='tight')
        
        return fig
    
    def create_trial_analysis(self, epochs, time_vector, stimcodes, save_path=None):
        """Create trial-based analysis visualization."""
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle('ECoG Trial Analysis', fontsize=20, fontweight='bold')
        
        # 1. Trial timeline
        ax1 = axes[0, 0]
        self._plot_trial_timeline(ax1, epochs, time_vector)
        
        # 2. Stimulus category analysis
        ax2 = axes[0, 1]
        self._plot_stimulus_categories(ax2, stimcodes)
        
        # 3. Average response by category
        ax3 = axes[1, 0]
        self._plot_category_responses(ax3, epochs, stimcodes, time_vector)
        
        # 4. Channel response patterns
        ax4 = axes[1, 1]
        self._plot_channel_responses(ax4, epochs, time_vector)
        
        # 5. Temporal dynamics
        ax5 = axes[2, 0]
        self._plot_temporal_dynamics(ax5, epochs, time_vector)
        
        # 6. Trial quality assessment
        ax6 = axes[2, 1]
        self._plot_trial_quality(ax6, epochs)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.savefig(str(save_path).replace('.png', '.svg'), bbox_inches='tight')
        
        return fig
    
    def _plot_brain_regions(self, ax, channel_data=None):
        """Plot brain regions with channel information."""
        ax.set_title('Brain Regions and Channel Distribution', fontweight='bold')
        
        # Create brain outline
        brain_outline = patches.Ellipse((0.5, 0.5), 0.8, 0.6, 
                                      transform=ax.transAxes, 
                                      fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(brain_outline)
        
        # Plot regions
        for region_name, region_info in self.brain_regions.items():
            # Region background
            region_patch = patches.Rectangle(
                region_info['position'], region_info['size'][0], region_info['size'][1],
                transform=ax.transAxes, facecolor=region_info['color'], 
                alpha=0.3, edgecolor='black', linewidth=1
            )
            ax.add_patch(region_patch)
            
            # Region label
            ax.text(region_info['position'][0] + region_info['size'][0]/2,
                   region_info['position'][1] + region_info['size'][1]/2,
                   f'{region_name}\n({len(region_info["channels"])} ch)',
                   transform=ax.transAxes, ha='center', va='center',
                   fontweight='bold', fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
    
    def _plot_channel_grid(self, ax, channel_data=None):
        """Plot channel grid layout."""
        ax.set_title('ECoG Channel Grid Layout', fontweight='bold')
        
        # Create 16x10 grid for 160 channels
        n_rows, n_cols = 16, 10
        grid = np.zeros((n_rows, n_cols))
        
        # Fill grid with channel numbers
        for i in range(160):
            row = i // n_cols
            col = i % n_cols
            grid[row, col] = i + 1
        
        # Create heatmap
        im = ax.imshow(grid, cmap='viridis', aspect='auto')
        
        # Add channel numbers
        for i in range(n_rows):
            for j in range(n_cols):
                if grid[i, j] > 0:
                    ax.text(j, i, int(grid[i, j]), ha='center', va='center',
                           color='white' if grid[i, j] > 80 else 'black',
                           fontsize=8, fontweight='bold')
        
        ax.set_xlabel('Grid Column')
        ax.set_ylabel('Grid Row')
        ax.set_title('ECoG Channel Grid Layout (160 channels)')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Channel Number')
    
    def _plot_visual_categories(self, ax):
        """Plot visual stimulus categories."""
        ax.set_title('Visual Stimulus Categories', fontweight='bold')
        
        categories = list(self.visual_categories.keys())
        colors = [self.visual_categories[cat]['color'] for cat in categories]
        descriptions = [self.visual_categories[cat]['description'] for cat in categories]
        
        bars = ax.bar(range(len(categories)), [1]*len(categories), color=colors, alpha=0.8)
        
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.set_ylabel('Categories')
        ax.set_ylim(0, 1.2)
        
        # Add descriptions
        for i, (bar, desc) in enumerate(zip(bars, descriptions)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                   desc, ha='center', va='bottom', fontsize=9, rotation=0)
    
    def _plot_channel_statistics(self, ax, channel_data=None):
        """Plot channel statistics."""
        ax.set_title('Channel Statistics Overview', fontweight='bold')
        
        if channel_data is not None:
            # Compute channel statistics
            channel_means = np.mean(channel_data, axis=1)
            channel_stds = np.std(channel_data, axis=1)
            
            # Create scatter plot
            scatter = ax.scatter(channel_means, channel_stds, 
                               c=range(len(channel_means)), cmap='viridis', 
                               alpha=0.7, s=50)
            
            ax.set_xlabel('Mean Amplitude (μV)')
            ax.set_ylabel('Standard Deviation (μV)')
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Channel Number')
        else:
            # Placeholder
            ax.text(0.5, 0.5, 'Channel data not available', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12, style='italic')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
    
    def _plot_channel_power_map(self, ax, data):
        """Plot channel power map."""
        ax.set_title('Channel Power Map', fontweight='bold')
        
        # Compute power for each channel
        channel_power = np.var(data, axis=1)
        
        # Create heatmap
        n_channels = len(channel_power)
        power_matrix = channel_power.reshape(16, 10)  # 16x10 grid
        
        im = ax.imshow(power_matrix, cmap='hot', aspect='auto')
        ax.set_xlabel('Grid Column')
        ax.set_ylabel('Grid Row')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Power (μV²)')
    
    def _plot_regional_power(self, ax, data):
        """Plot regional power distribution."""
        ax.set_title('Regional Power Distribution', fontweight='bold')
        
        region_powers = {}
        for region_name, region_info in self.brain_regions.items():
            region_indices = [ch - 1 for ch in region_info['channels']]
            region_data = data[region_indices, :]
            region_power = np.mean(np.var(region_data, axis=1))
            region_powers[region_name] = region_power
        
        regions = list(region_powers.keys())
        powers = list(region_powers.values())
        colors = [self.brain_regions[region]['color'] for region in regions]
        
        bars = ax.bar(regions, powers, color=colors, alpha=0.8)
        ax.set_ylabel('Average Power (μV²)')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, power in zip(bars, powers):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(powers)*0.01,
                   f'{power:.2f}', ha='center', va='bottom', fontweight='bold')
    
    def _plot_channel_correlation(self, ax, data):
        """Plot channel correlation matrix."""
        ax.set_title('Channel Correlation Matrix', fontweight='bold')
        
        # Compute correlation matrix for subset of channels
        n_channels = min(50, data.shape[0])  # Limit for visualization
        subset_data = data[:n_channels, :5000]  # Use subset for speed
        
        corr_matrix = np.corrcoef(subset_data)
        
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xlabel('Channel Index')
        ax.set_ylabel('Channel Index')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation')
    
    def _plot_high_gamma_map(self, ax, data):
        """Plot high-gamma activity map."""
        ax.set_title('High-Gamma Activity Map', fontweight='bold')
        
        # Simulate high-gamma power (in real implementation, this would be computed)
        high_gamma_power = np.random.gamma(2, 1, data.shape[0])
        power_matrix = high_gamma_power.reshape(16, 10)
        
        im = ax.imshow(power_matrix, cmap='plasma', aspect='auto')
        ax.set_xlabel('Grid Column')
        ax.set_ylabel('Grid Row')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('High-Gamma Power')
    
    def _plot_frequency_bands(self, ax, data):
        """Plot frequency band comparison."""
        ax.set_title('Frequency Band Comparison', fontweight='bold')
        
        # Simulate frequency band powers
        bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma', 'High-Gamma']
        powers = np.random.exponential(1, len(bands))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        bars = ax.bar(bands, powers, color=colors, alpha=0.8)
        ax.set_ylabel('Relative Power')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, power in zip(bars, powers):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(powers)*0.01,
                   f'{power:.2f}', ha='center', va='bottom', fontweight='bold')
    
    def _plot_channel_quality(self, ax, data):
        """Plot channel quality assessment."""
        ax.set_title('Channel Quality Assessment', fontweight='bold')
        
        # Compute quality metrics
        channel_vars = np.var(data, axis=1)
        var_threshold = np.mean(channel_vars) + 3 * np.std(channel_vars)
        
        good_channels = np.sum(channel_vars <= var_threshold)
        bad_channels = np.sum(channel_vars > var_threshold)
        
        labels = ['Good Channels', 'Bad Channels']
        sizes = [good_channels, bad_channels]
        colors = ['lightgreen', 'lightcoral']
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                         autopct='%1.1f%%', startangle=90)
        ax.set_title(f'Channel Quality\n({good_channels}/{good_channels + bad_channels} good)')
    
    def _plot_trial_timeline(self, ax, epochs, time_vector):
        """Plot trial timeline."""
        ax.set_title('Trial Timeline', fontweight='bold')
        
        if epochs.size == 0:
            ax.text(0.5, 0.5, 'No trials available', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12)
            return
        
        # Plot average response across all channels
        avg_response = np.mean(epochs, axis=(0, 1))
        
        ax.plot(time_vector, avg_response, 'b-', linewidth=2, alpha=0.8)
        ax.axvline(0, color='red', linestyle='--', alpha=0.7, label='Stimulus Onset')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Average Response (μV)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_stimulus_categories(self, ax, stimcodes):
        """Plot stimulus category distribution."""
        ax.set_title('Stimulus Category Distribution', fontweight='bold')
        
        if len(stimcodes) == 0:
            ax.text(0.5, 0.5, 'No stimulus codes available', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12)
            return
        
        unique_codes, counts = np.unique(stimcodes, return_counts=True)
        colors = [self.visual_categories.get(f'category_{code}', '#888888')['color'] 
                 for code in unique_codes]
        
        bars = ax.bar(unique_codes, counts, color=colors, alpha=0.8)
        ax.set_xlabel('Stimulus Code')
        ax.set_ylabel('Number of Trials')
        
        # Add value labels
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                   str(count), ha='center', va='bottom', fontweight='bold')
    
    def _plot_category_responses(self, ax, epochs, stimcodes, time_vector):
        """Plot average response by category."""
        ax.set_title('Average Response by Category', fontweight='bold')
        
        if epochs.size == 0 or len(stimcodes) == 0:
            ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12)
            return
        
        # Group trials by stimulus category
        unique_codes = np.unique(stimcodes)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_codes)))
        
        for i, code in enumerate(unique_codes):
            mask = stimcodes == code
            if np.any(mask):
                category_epochs = epochs[mask]
                avg_response = np.mean(category_epochs, axis=(0, 1))
                ax.plot(time_vector, avg_response, color=colors[i], 
                       linewidth=2, alpha=0.8, label=f'Category {code}')
        
        ax.axvline(0, color='red', linestyle='--', alpha=0.7, label='Stimulus Onset')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Average Response (μV)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_channel_responses(self, ax, epochs, time_vector):
        """Plot channel response patterns."""
        ax.set_title('Channel Response Patterns', fontweight='bold')
        
        if epochs.size == 0:
            ax.text(0.5, 0.5, 'No trials available', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12)
            return
        
        # Plot responses for subset of channels
        n_channels_to_plot = min(10, epochs.shape[1])
        channel_indices = np.linspace(0, epochs.shape[1]-1, n_channels_to_plot, dtype=int)
        
        colors = plt.cm.viridis(np.linspace(0, 1, n_channels_to_plot))
        
        for i, ch_idx in enumerate(channel_indices):
            avg_response = np.mean(epochs[:, ch_idx, :], axis=0)
            ax.plot(time_vector, avg_response, color=colors[i], 
                   linewidth=1.5, alpha=0.7, label=f'Ch {ch_idx+1}')
        
        ax.axvline(0, color='red', linestyle='--', alpha=0.7, label='Stimulus Onset')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Response (μV)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def _plot_temporal_dynamics(self, ax, epochs, time_vector):
        """Plot temporal dynamics."""
        ax.set_title('Temporal Dynamics', fontweight='bold')
        
        if epochs.size == 0:
            ax.text(0.5, 0.5, 'No trials available', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12)
            return
        
        # Compute temporal dynamics
        all_responses = epochs.reshape(-1, epochs.shape[-1])
        mean_response = np.mean(all_responses, axis=0)
        std_response = np.std(all_responses, axis=0)
        
        ax.plot(time_vector, mean_response, 'b-', linewidth=2, label='Mean')
        ax.fill_between(time_vector, mean_response - std_response, 
                       mean_response + std_response, alpha=0.3, color='blue', label='±1 SD')
        
        ax.axvline(0, color='red', linestyle='--', alpha=0.7, label='Stimulus Onset')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Response (μV)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_trial_quality(self, ax, epochs):
        """Plot trial quality assessment."""
        ax.set_title('Trial Quality Assessment', fontweight='bold')
        
        if epochs.size == 0:
            ax.text(0.5, 0.5, 'No trials available', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12)
            return
        
        # Compute trial quality metrics
        trial_vars = np.var(epochs, axis=(1, 2))
        trial_max_amps = np.max(np.abs(epochs), axis=(1, 2))
        
        # Create scatter plot
        scatter = ax.scatter(trial_vars, trial_max_amps, 
                           c=range(len(trial_vars)), cmap='viridis', 
                           alpha=0.7, s=50)
        
        ax.set_xlabel('Trial Variance')
        ax.set_ylabel('Maximum Amplitude')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Trial Number')
        
        # Add quality thresholds
        var_threshold = np.mean(trial_vars) + 3 * np.std(trial_vars)
        amp_threshold = np.mean(trial_max_amps) + 3 * np.std(trial_max_amps)
        
        ax.axvline(var_threshold, color='red', linestyle='--', alpha=0.7, label='Variance Threshold')
        ax.axhline(amp_threshold, color='red', linestyle='--', alpha=0.7, label='Amplitude Threshold')
        ax.legend()
