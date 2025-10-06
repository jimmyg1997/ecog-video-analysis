# IEEE-SMC-2025 ECoG Video Analysis Pipeline

## 🏆 Competition Overview

This repository contains a comprehensive pipeline for analyzing ECoG data from the IEEE-SMC-2025 ECoG Video Analysis Competition. The pipeline is designed to decode visual stimuli (faces, colors, shapes, etc.) from brain activity recorded while patients watch videos.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install numpy scipy matplotlib seaborn pandas scikit-learn mne tqdm jupyter
```

### 2. Run Complete Pipeline
```bash
python run_pipeline.py --task all
```

### 3. Run Individual Tasks
```bash
# Task 1: Exploratory Data Analysis
python run_pipeline.py --task 1

# Task 2: Preprocessing & Modeling
python run_pipeline.py --task 2

# Task 3: Visualization
python run_pipeline.py --task 3
```

## 📁 Project Structure

```
ecog-video-analysis/
├── data/
│   ├── raw/                    # Raw ECoG data (.mat files)
│   ├── preprocessed/           # Preprocessed data
│   ├── features/              # Extracted features
│   └── models/                # Trained models
├── notebooks/                 # Jupyter notebooks for analysis
│   ├── 1_EDA_and_DataUnderstanding.ipynb
│   ├── 2_Preprocessing_FeatureExtraction_Modeling.ipynb
│   └── 3_Live_Annotation_And_Visualization.ipynb
├── src/                       # Source code modules
│   ├── utils/                 # Utility functions
│   ├── preprocessing/         # Preprocessing modules
│   ├── features/             # Feature extraction
│   ├── models/               # Machine learning models
│   └── visualization/        # Visualization tools
├── results/                   # Analysis results
│   ├── visualizations/       # Generated plots
│   ├── analysis/             # Analysis data
│   └── reports/              # Summary reports
└── run_pipeline.py           # Main execution script
```

## 🔧 Pipeline Execution Flow

### Phase 1: Project Refactoring ✅
- [x] Restructured all files and folders
- [x] Created modular source code architecture
- [x] Implemented progress tracking
- [x] Organized output directories

### Task 1: Exploratory Data Analysis (EDA)
**Goal**: Build intuition about the dataset and identify brain regions, frequencies, and events.

**Steps**:
1. Load raw ECoG data from MATLAB files
2. Inspect metadata: channels, timestamps, stimulus markers
3. Analyze signal quality and detect bad channels
4. Visualize brain region activations
5. Analyze high-gamma power distribution (70-150 Hz)
6. Summarize visual categories and annotations

**Deliverables**:
- `notebooks/1_EDA_and_DataUnderstanding.ipynb`
- Channel statistics and bad channel detection
- PSD analysis and frequency band analysis
- Event detection and timeline analysis
- Summary report: `results/analysis/eda_summary.json`

### Task 2: Preprocessing, Feature Extraction & Algorithm Comparison
**Goal**: Preprocess data, extract features, and compare multiple algorithms.

**Steps**:
1. **Preprocessing**:
   - Bandpass filter: 0.5-150 Hz
   - Notch filter: 50/60 Hz
   - Artifact rejection and bad channel removal
   - Stimulus alignment via photodiode signal
   - Trial segmentation: 100-400 ms post-onset windows
   - Baseline normalization: -300 to 0 ms

2. **Feature Extraction**:
   - **A. Broadband Gamma Power** (Primary):
     - Filter: 110-140 Hz
     - Epoch: 100-400 ms post-stimulus
     - Feature: log-variance, z-scored per trial
   - **B. Canonical Band Powers** (Optional):
     - Theta (4-8 Hz), Alpha (8-13 Hz), Beta (13-30 Hz), Gamma (30-80 Hz)

3. **Modeling** (Compare A-D methods):
   - **A. Template correlation** (LOOCV) — gamma templates
   - **B. CSP + LDA** — spatial filtering + linear classification
   - **C. EEGNet** — compact CNN for neural decoding
   - **D. Time-Series Transformer** — long-range temporal attention

4. **Evaluation**:
   - Metrics: Accuracy, F1, AUC, Precision, Recall
   - Confusion matrix for error breakdown
   - Saliency/attention maps for interpretability

**Deliverables**:
- `notebooks/2_Preprocessing_FeatureExtraction_Modeling.ipynb`
- Preprocessed data in `data/preprocessed/`
- Extracted features in `data/features/`
- Trained models in `data/models/`
- Performance comparison reports

### Task 3: Live Video Annotation & Brain Activation Visualization
**Goal**: Create interactive visualization linking ECoG activity to visual stimuli.

**Steps**:
1. Synchronize ECoG trials with video frames/stimulus labels
2. Build visualizations:
   - Predicted label overlays (model output vs. true stimulus)
   - Activation overlays (most active electrodes per frame)
   - Time-based signal visualization (rolling high-gamma vs. video playback)
3. Develop dashboard integrating:
   - Real-time signal traces
   - Predicted vs. true category timeline
   - Channel activation visualization
4. Export demo video for presentations

**Deliverables**:
- `notebooks/3_Live_Annotation_And_Visualization.ipynb`
- Interactive visualization dashboard
- Demo video in `results/visualizations/`
- Video synchronization scripts

## 🧠 Key Technical Features

### High-Gamma Analysis (Gold Standard)
- **Frequency Range**: 70-150 Hz (primary), 110-140 Hz (broadband)
- **Rationale**: High-gamma activity is the gold standard for visual stimulus decoding
- **Implementation**: Bandpass filtering + Hilbert transform for envelope extraction

### Brain Region Specialization
- **Occipital**: Primary visual cortex (channels 1-40)
- **Temporal**: Object recognition (channels 41-80)
- **Parietal**: Spatial processing (channels 81-120)
- **Central**: Sensorimotor (channels 121-140)
- **Frontal**: Higher-order processing (channels 141-160)

### Visual Stimulus Categories
- **digit**: Numbers (0-9)
- **kanji**: Japanese Kanji characters
- **face**: Human faces
- **body**: Human bodies/figures
- **object**: Various objects
- **hiragana**: Japanese Hiragana characters
- **line**: Line patterns/shapes

## 📊 Progress Tracking

All long-running operations include progress indicators using `tqdm`:

```python
from src.utils.progress_tracker import track_progress

with track_progress("Processing data", total=1000) as pbar:
    for i in range(1000):
        # Process item
        pbar.update(1)
```

## 🔧 Configuration

The pipeline uses a centralized configuration system:

```python
from src.utils.config import AnalysisConfig

config = AnalysisConfig()
config.sampling_rate = 1200
config.gamma_low = 70.0
config.gamma_high = 150.0
```

## 📈 Expected Results

### Competition Advantages:
1. **High-Gamma Focus**: Uses gold standard frequency range for visual decoding
2. **Brain Region Specialization**: Optimized channel selection per region
3. **Advanced Preprocessing**: CAR, artifact rejection, baseline correction
4. **Multiple Algorithms**: Template correlation, CSP+LDA, EEGNet, Transformer
5. **Comprehensive Evaluation**: Multiple metrics and interpretability analysis

### Performance Targets:
- **Accuracy**: >80% for visual category classification
- **High-Gamma Channels**: Top 20 channels identified for optimal performance
- **Processing Speed**: Real-time capable preprocessing pipeline
- **Reproducibility**: Complete pipeline with saved intermediate results

## 🚀 Execution Examples

### Run Complete Pipeline:
```bash
python run_pipeline.py --task all --timeout 7200
```

### Run with Custom Configuration:
```bash
python run_pipeline.py --task 2 --skip-deps
```

### Monitor Progress:
The pipeline provides detailed progress tracking for all operations:
- Data loading and preprocessing
- Feature extraction
- Model training
- Visualization generation

## 📁 Output Files

After running the complete pipeline, you'll find:

- **Preprocessed Data**: `data/preprocessed/ecog_preprocessed.npy`
- **Features**: `data/features/high_gamma_features.npy`
- **Models**: `data/models/` (multiple model files)
- **Visualizations**: `results/visualizations/` (PNG/SVG files)
- **Analysis**: `results/analysis/` (CSV/JSON summaries)
- **Reports**: `results/reports/` (comprehensive analysis reports)

## 🏆 Competition Readiness

This pipeline is specifically designed for the IEEE-SMC-2025 ECoG Video Analysis Competition and includes:

- **Competition-specific optimizations** for visual stimulus decoding
- **Gold standard methods** (high-gamma analysis)
- **Multiple algorithm comparison** for robust performance
- **Comprehensive evaluation** with interpretability
- **Real-time visualization** capabilities
- **Complete reproducibility** with saved intermediate results

## 📞 Support

For questions or issues with the pipeline, please check:
1. The individual notebook documentation
2. The source code comments in `src/` modules
3. The analysis results in `results/analysis/`

---

**Ready for IEEE-SMC-2025 Competition! 🏆**