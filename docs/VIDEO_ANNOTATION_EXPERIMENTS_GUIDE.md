# 🎬 Real-Time Video Annotation Experiments Guide

## 📋 Overview

This guide explains how to use the `run_video_annotation_experiments.py` script to create 6 different types of real-time video annotations with brain activity overlays.

## 🚀 Quick Start

### 1. List Available Approaches
```bash
python run_video_annotation_experiments.py --list-approaches
```

### 2. Run an Experiment
```bash
python run_video_annotation_experiments.py --approach 1 --duration 20
```

## 🎯 Available Approaches

### **Approach 1: Spatial Motor Cortex Activation Map**
- **What**: Shows a 2D electrode grid heatmap of motor cortex activation
- **Visual**: Split screen with video on left, electrode grid on right
- **Best for**: Analyzing motor cortex activity during movement
- **Command**: `--approach 1`

### **Approach 2: Gait-Phase Neural Signature Timeline**
- **What**: Detects gait phases and shows corresponding neural patterns
- **Visual**: Video with timeline at bottom showing gait phases and ECoG traces
- **Best for**: Analyzing walking patterns and neural signatures
- **Command**: `--approach 2`

### **Approach 3: Event-Related Spectral Perturbation (ERSP) Video Overlay**
- **What**: Shows real-time spectrograms for different electrode clusters
- **Visual**: Video with small spectrogram panels in corners
- **Best for**: Analyzing frequency-specific brain activity
- **Command**: `--approach 3`

### **Approach 4: Enhanced Brain Region Activation**
- **What**: Improved version of brain region and channel activation visualization
- **Visual**: Enhanced brain region bars with better formatting
- **Best for**: General brain activity analysis with improved visualization
- **Command**: `--approach 4`

### **Approach 5: Real-Time Object Detection Annotation (NEW)**
- **What**: Simulates real-time object detection with detailed metadata
- **Visual**: Yellow info box showing object type, properties, and brain activation
- **Best for**: Analyzing brain responses to different object types
- **Command**: `--approach 5`

### **Approach 6: Brain Atlas Activation Overlay (NEW)**
- **What**: Shows fusiform gyrus regions (Place, Face, Shape, Word Form) with real-time activation
- **Visual**: Brain atlas overlay with region-specific activation bars
- **Best for**: Analyzing specialized brain regions for visual processing
- **Command**: `--approach 6`

## 📝 Command Line Options

### Basic Usage
```bash
python run_video_annotation_experiments.py --approach <number> --duration <seconds>
```

### Parameters
- `--approach`: Approach number (1-6) - **Required**
- `--duration`: Duration in seconds (default: 20)
- `--list-approaches`: List all available approaches

### Examples

#### Run Approach 1 for 30 seconds
```bash
python run_video_annotation_experiments.py --approach 1 --duration 30
```

#### Run Approach 5 for 15 seconds
```bash
python run_video_annotation_experiments.py --approach 5 --duration 15
```

#### Run Approach 6 for 25 seconds
```bash
python run_video_annotation_experiments.py --approach 6 --duration 25
```

## 📁 Output Structure

Each experiment creates a new folder under `results/06_video_analysis/`:

```
results/06_video_analysis/
├── experiment1/
│   ├── walk_annotated_motor_cortex_exp1.mp4
│   └── experiment_info.json
├── experiment2/
│   ├── walk_annotated_gait_phase_exp2.mp4
│   └── experiment_info.json
└── experiment3/
    ├── walk_annotated_object_detection_exp3.mp4
    └── experiment_info.json
```

## 🔧 Technical Details

### Data Requirements
- **ECoG Data**: Automatically loads from latest experiment folder
- **Video**: `data/raw/walk.mp4`
- **Annotations**: `results/annotations/video_annotation_data.json`

### Brain Regions Mapped

#### Approach 1 (Motor Cortex)
- Primary Motor (M1): Channels 40-80
- Somatosensory (S1): Channels 80-120
- Premotor/SMA: Channels 20-40
- Leg Area: Channels 60-100
- Hand Area: Channels 100-140
- Face Area: Channels 140-160

#### Approach 5 (Object Detection)
- Numbers: Visual cortex (120-160)
- Faces: Fusiform face area (100-140)
- Objects: Object recognition (80-120)
- Text: Word form area (140-160)
- Body: Body representation (60-100)

#### Approach 6 (Brain Atlas)
- **Fusiform Gyrus Regions**:
  - Place: Channels 120-140 (Olive green)
  - Face: Channels 100-120 (Light blue)
  - Shape: Channels 80-100 (Dark blue)
  - Word Form: Channels 140-160 (Yellow)
- **Other Brain Regions**:
  - Motor Cortex: Channels 40-80
  - Visual Cortex: Channels 120-160
  - Temporal: Channels 80-120
  - Frontal: Channels 0-40

## 🎨 Visualization Features

### Color Coding
- **Red**: High activation
- **Green**: Medium activation
- **Blue**: Low activation
- **Yellow**: Special regions (Word Form, Frontal)
- **Cyan**: Face regions

### Real-Time Updates
- All approaches update in real-time as the video plays
- Brain activations are calculated using sliding windows
- Visualizations are synchronized with video timestamps

## 🐛 Troubleshooting

### Common Issues

#### 1. Video File Not Found
```
❌ Error: Video file not found: data/raw/walk.mp4
```
**Solution**: Ensure the video file exists in the correct location.

#### 2. ECoG Data Not Found
```
❌ Error: Could not load ECoG data
```
**Solution**: Run the preprocessing pipeline first to generate ECoG data.

#### 3. Annotation Data Not Found
```
❌ Error: Could not load annotation data
```
**Solution**: Ensure `results/annotations/video_annotation_data.json` exists.

### Performance Tips
- Use shorter durations (10-20 seconds) for testing
- Longer durations (30+ seconds) will take more time to process
- The script shows progress every 100 frames

## 🔬 Scientific Applications

### Approach 1: Motor Cortex Analysis
- Study motor cortex activation during movement
- Analyze electrode-specific responses
- Compare activation patterns across different movements

### Approach 2: Gait Analysis
- Study neural signatures of walking
- Analyze gait phase transitions
- Compare neural patterns across different walking speeds

### Approach 3: Spectral Analysis
- Study frequency-specific brain activity
- Analyze event-related spectral perturbations
- Compare power across different frequency bands

### Approach 4: General Brain Activity
- Study overall brain region activation
- Analyze channel-specific responses
- Compare activation patterns across different stimuli

### Approach 5: Object Recognition
- Study brain responses to different object types
- Analyze object detection properties
- Compare neural patterns for different visual stimuli

### Approach 6: Specialized Visual Processing
- Study fusiform gyrus activation
- Analyze specialized visual processing regions
- Compare activation patterns for different visual categories

## 📊 Expected Results

### Video Outputs
- **Format**: MP4 videos with annotations overlaid
- **Quality**: Same resolution as input video
- **Frame Rate**: 30 FPS
- **Duration**: As specified by `--duration` parameter

### Experiment Metadata
Each experiment saves metadata in `experiment_info.json`:
```json
{
  "experiment_number": 1,
  "timestamp": "2025-01-05T16:30:00",
  "approach": 1,
  "duration": 20,
  "data_source": "experiment2",
  "video_path": "data/raw/walk.mp4",
  "output_path": "results/06_video_analysis/experiment1/walk_annotated_motor_cortex_exp1.mp4"
}
```

## 🎉 Success!

After running an experiment, you'll see:
```
✅ Approach 1 completed: results/06_video_analysis/experiment1/walk_annotated_motor_cortex_exp1.mp4

🎉 Experiment 1 completed!
📁 Files saved to: results/06_video_analysis/experiment1
```

The annotated video will be ready for analysis and visualization! 🎬✨
