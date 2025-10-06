#!/usr/bin/env python3
"""
Visualization Utilities for ECoG Video Analysis
Provides utility functions for creating various visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class ECoGVisualizationUtils:
    """Utility class for ECoG visualizations."""
    
    def __init__(self):
        """Initialize visualization utilities."""
        # Set up plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.rcParams['figure.figsize'] = (15, 10)
        plt.rcParams['font.size'] = 12
        
        # Color schemes
        self.brain_region_colors = {
            'Occipital': '#FF6B6B',  # Red
            'Temporal': '#4ECDC4',   # Teal
            'Parietal': '#45B7D1',   # Blue
            'Central': '#96CEB4',    # Green
            'Frontal': '#FFEAA7'     # Yellow
        }
        
        self.category_colors = {
            'digit': '#2ECC71',      # Green
            'kanji': '#3498DB',      # Blue
            'face': '#E74C3C',       # Red
            'body': '#F39C12',       # Orange
            'object': '#9B59B6',     # Purple
            'hiragana': '#1ABC9C',   # Turquoise
            'line': '#34495E',       # Dark gray
            'baseline': '#95A5A6'    # Light gray
        }
    
    def create_brain_region_heatmap(self, activation_data: np.ndarray, 
                                  time_points: np.ndarray, 
                                  brain_regions: List[str],
                                  title: str = "Brain Region Activation Heatmap") -> plt.Figure:
        """Create brain region activation heatmap."""
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Create heatmap
        sns.heatmap(activation_data.T, 
                   xticklabels=[f"{t:.1f}s" for t in time_points[::5]],  # Show every 5th time point
                   yticklabels=brain_regions,
                   cmap='viridis',
                   cbar_kws={'label': 'Activation Level'},
                   ax=ax)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Brain Region')
        
        plt.tight_layout()
        return fig
    
    def create_category_response_plot(self, response_data: np.ndarray,
                                    time_points: np.ndarray,
                                    categories: List[str],
                                    title: str = "Category Response Over Time") -> plt.Figure:
        """Create category response plot."""
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plot each category
        for i, category in enumerate(categories):
            color = self.category_colors.get(category, '#95A5A6')
            ax.plot(time_points, response_data[:, i], 
                   label=category, color=color, linewidth=2)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Response Strength')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_high_gamma_timeline(self, envelope_data: np.ndarray,
                                 time_points: np.ndarray,
                                 channel_names: List[str],
                                 title: str = "High-Gamma Envelope Timeline") -> plt.Figure:
        """Create high-gamma envelope timeline."""
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot 1: Top 5 channels
        top_5_indices = min(5, len(channel_names))
        for i in range(top_5_indices):
            axes[0].plot(time_points, envelope_data[:, i], 
                        label=channel_names[i], linewidth=2)
        
        axes[0].set_title(f'{title} - Top {top_5_indices} Channels', fontsize=16, fontweight='bold')
        axes[0].set_xlabel('Time (seconds)')
        axes[0].set_ylabel('High-Gamma Envelope')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Heatmap of all channels
        sns.heatmap(envelope_data.T, 
                   xticklabels=[f"{t:.1f}s" for t in time_points[::10]],  # Show every 10th time point
                   yticklabels=channel_names,
                   cmap='inferno',
                   cbar_kws={'label': 'High-Gamma Envelope'},
                   ax=axes[1])
        
        axes[1].set_title('All Channels - High-Gamma Envelope Heatmap', fontsize=16, fontweight='bold')
        axes[1].set_xlabel('Time (seconds)')
        axes[1].set_ylabel('Channel')
        
        plt.tight_layout()
        return fig
    
    def create_ensemble_dashboard(self, prediction_data: Dict[str, np.ndarray],
                                time_points: np.ndarray,
                                title: str = "Ensemble Prediction Dashboard") -> plt.Figure:
        """Create ensemble prediction dashboard."""
        fig, axes = plt.subplots(4, 1, figsize=(15, 16))
        
        # Plot 1: ECoG Activity
        if 'ecog_activity' in prediction_data:
            axes[0].plot(time_points, prediction_data['ecog_activity'], 
                        color='blue', linewidth=2, label='ECoG Activity')
            axes[0].axhline(y=45.0, color='red', linestyle='--', alpha=0.7, label='Stimulus Threshold')
            axes[0].set_title('ECoG Activity Over Time', fontsize=16, fontweight='bold')
            axes[0].set_ylabel('Activity Level')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Predictions Timeline
        if 'ensemble_prediction' in prediction_data:
            prediction_colors = {
                'stimulus': 'green',
                'baseline': 'gray',
                'object': 'orange',
                'face': 'pink',
                'digit': 'blue'
            }
            
            for i, (t, pred) in enumerate(zip(time_points, prediction_data['ensemble_prediction'])):
                color = prediction_colors.get(pred, 'gray')
                axes[1].scatter(t, i % 3, c=color, s=50, alpha=0.7)
            
            axes[1].set_title('Ensemble Predictions Timeline', fontsize=16, fontweight='bold')
            axes[1].set_ylabel('Prediction Type')
            axes[1].set_ylim(-0.5, 2.5)
        
        # Plot 3: Confidence Levels
        if 'confidence' in prediction_data:
            axes[2].plot(time_points, prediction_data['confidence'], 
                        color='purple', linewidth=2, label='Confidence')
            axes[2].fill_between(time_points, prediction_data['confidence'], 
                               alpha=0.3, color='purple')
            axes[2].set_title('Prediction Confidence Over Time', fontsize=16, fontweight='bold')
            axes[2].set_ylabel('Confidence')
            axes[2].set_ylim(0, 1)
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Model Comparison
        if 'ecog_prediction' in prediction_data and 'ensemble_prediction' in prediction_data:
            ecog_counts = np.unique(prediction_data['ecog_prediction'], return_counts=True)
            ensemble_counts = np.unique(prediction_data['ensemble_prediction'], return_counts=True)
            
            x = np.arange(len(ecog_counts[0]))
            width = 0.35
            
            axes[3].bar(x - width/2, ecog_counts[1], width, label='ECoG Only', alpha=0.7)
            axes[3].bar(x + width/2, ensemble_counts[1], width, label='Ensemble', alpha=0.7)
            
            axes[3].set_title('Model Comparison - Prediction Counts', fontsize=16, fontweight='bold')
            axes[3].set_ylabel('Count')
            axes[3].set_xticks(x)
            axes[3].set_xticklabels(ecog_counts[0], rotation=45)
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_interactive_plotly_dashboard(self, data_dict: Dict[str, Any]) -> go.Figure:
        """Create interactive Plotly dashboard."""
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Brain Region Activation', 'Category Responses',
                          'High-Gamma Timeline', 'Confidence Levels',
                          'Model Comparison', 'Channel Activity'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Add traces based on available data
        if 'brain_regions' in data_dict:
            for region, values in data_dict['brain_regions'].items():
                fig.add_trace(
                    go.Scatter(x=data_dict['time_points'], y=values,
                             mode='lines', name=region,
                             line=dict(color=self.brain_region_colors.get(region, '#95A5A6'))),
                    row=1, col=1
                )
        
        if 'categories' in data_dict:
            for category, values in data_dict['categories'].items():
                fig.add_trace(
                    go.Scatter(x=data_dict['time_points'], y=values,
                             mode='lines', name=category,
                             line=dict(color=self.category_colors.get(category, '#95A5A6'))),
                    row=1, col=2
                )
        
        if 'high_gamma' in data_dict:
            for i, channel in enumerate(data_dict['high_gamma']['channels'][:5]):
                fig.add_trace(
                    go.Scatter(x=data_dict['time_points'], y=data_dict['high_gamma']['data'][:, i],
                             mode='lines', name=f'Ch{channel}'),
                    row=2, col=1
                )
        
        if 'confidence' in data_dict:
            fig.add_trace(
                go.Scatter(x=data_dict['time_points'], y=data_dict['confidence'],
                         mode='lines', name='Confidence',
                         line=dict(color='purple')),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Interactive ECoG Video Analysis Dashboard",
            showlegend=True,
            height=800
        )
        
        return fig
    
    def save_visualization(self, fig: plt.Figure, filename: str, 
                         output_dir: str = "../results/visualizations") -> bool:
        """Save visualization to file."""
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✅ Visualization saved: {filepath}")
            return True
        except Exception as e:
            print(f"❌ Error saving visualization: {str(e)}")
            return False
    
    def create_summary_statistics(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary statistics for the data."""
        stats = {}
        
        if 'brain_regions' in data_dict:
            stats['brain_regions'] = {}
            for region, values in data_dict['brain_regions'].items():
                stats['brain_regions'][region] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'max': np.max(values),
                    'min': np.min(values)
                }
        
        if 'categories' in data_dict:
            stats['categories'] = {}
            for category, values in data_dict['categories'].items():
                stats['categories'][category] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'max': np.max(values),
                    'min': np.min(values)
                }
        
        if 'confidence' in data_dict:
            stats['confidence'] = {
                'mean': np.mean(data_dict['confidence']),
                'std': np.std(data_dict['confidence']),
                'max': np.max(data_dict['confidence']),
                'min': np.min(data_dict['confidence'])
            }
        
        return stats

class RealTimeVisualization:
    """Real-time visualization utilities."""
    
    def __init__(self, update_interval: float = 0.1):
        """Initialize real-time visualization."""
        self.update_interval = update_interval
        self.viz_utils = ECoGVisualizationUtils()
    
    def create_live_dashboard(self, data_stream, max_points: int = 100):
        """Create live updating dashboard."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('ECoG Activity', 'Brain Regions', 'Categories', 'Confidence'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Initialize traces
        fig.add_trace(go.Scatter(x=[], y=[], name='ECoG Activity'), row=1, col=1)
        fig.add_trace(go.Scatter(x=[], y=[], name='Confidence'), row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title="Real-Time ECoG Video Analysis",
            showlegend=True,
            height=600
        )
        
        return fig
    
    def update_dashboard(self, fig, new_data: Dict[str, Any]):
        """Update dashboard with new data."""
        # This would be implemented for real-time updates
        # For now, it's a placeholder
        pass

def create_comprehensive_report(data_dict: Dict[str, Any], 
                              output_dir: str = "../results/reports") -> str:
    """Create comprehensive analysis report."""
    try:
        import os
        from datetime import datetime
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create report filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(output_dir, f"visualization_report_{timestamp}.md")
        
        # Create report content
        report_content = f"""# ECoG Video Analysis Visualization Report

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary

This report contains the results of the 4 smart annotation approaches for ECoG video analysis.

## Data Overview

"""
        
        # Add data statistics
        if 'time_points' in data_dict:
            report_content += f"- Analysis duration: {data_dict['time_points'][-1] - data_dict['time_points'][0]:.1f} seconds\n"
            report_content += f"- Number of time points: {len(data_dict['time_points'])}\n"
        
        if 'brain_regions' in data_dict:
            report_content += f"- Brain regions analyzed: {len(data_dict['brain_regions'])}\n"
        
        if 'categories' in data_dict:
            report_content += f"- Categories analyzed: {len(data_dict['categories'])}\n"
        
        report_content += """
## Approaches Implemented

1. **Real-Time Brain Region Activation Overlay**
   - Shows which brain regions are most active during each video frame
   - Uses top channels from analysis (Channel 131, 104-126, etc.)
   - Color-coded activation levels

2. **Category-Specific Channel Response Visualization**
   - Displays category-specific channel responses (digits, faces, objects, etc.)
   - Uses existing annotations and channel mappings
   - Shows confidence levels for each category

3. **High-Gamma Envelope Timeline with Video Sync**
   - Rolling high-gamma envelope visualization
   - Synchronized with video playback
   - Shows temporal dynamics of brain activity

4. **Multi-Model Ensemble Annotation Dashboard**
   - Combines ML object detection with ECoG responses
   - Shows predicted vs true labels
   - Interactive dashboard with multiple views

## Key Findings

"""
        
        # Add key findings based on data
        if 'brain_regions' in data_dict:
            report_content += "### Brain Region Analysis\n"
            for region, values in data_dict['brain_regions'].items():
                mean_activation = np.mean(values)
                report_content += f"- {region}: Mean activation = {mean_activation:.2f}\n"
        
        if 'categories' in data_dict:
            report_content += "\n### Category Analysis\n"
            for category, values in data_dict['categories'].items():
                mean_response = np.mean(values)
                report_content += f"- {category}: Mean response = {mean_response:.2f}\n"
        
        report_content += """
## Recommendations

1. Focus on the most responsive brain regions for classification
2. Use category-specific channel selections for better accuracy
3. Implement real-time visualization for live demonstrations
4. Combine multiple approaches for robust predictions

## Files Generated

- Brain region activation visualization
- Category response visualization  
- High-gamma timeline visualization
- Ensemble dashboard visualization
- This comprehensive report

---
*Generated by ECoG Video Analysis Pipeline*
"""
        
        # Write report
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        print(f"✅ Comprehensive report created: {report_file}")
        return report_file
        
    except Exception as e:
        print(f"❌ Error creating report: {str(e)}")
        return ""

if __name__ == "__main__":
    print("🎨 ECoG Visualization Utilities")
    print("This module provides utility functions for creating various visualizations.")
    print("Use the classes and functions in your notebook or other scripts.")
