# IEEE-SMC-2025 ECoG Video Analysis Pipeline

## 🏆 Competition Overview

This repository contains a comprehensive pipeline for analyzing ECoG data from the IEEE-SMC-2025 ECoG Video Analysis Competition. The project features an advanced web application with real-time neural decoding capabilities, interactive visualizations, and state-of-the-art machine learning models for decoding visual stimuli from brain activity.

## 🌟 Key Features

### 🚀 **Interactive Web Application**
- **Real-time ECoG visualization** with 3D brain maps
- **Video synchronization** with neural activity
- **Interactive data exploration** tools
- **Modern responsive design** with professional UI/UX
- **Live annotation timeline** with 30+ stimulus categories

### 🧠 **Advanced Neural Decoding**
- **Multiple ML approaches**: EEGNet CNN, Transformer models, CSP+LDA, Template Correlation
- **High-gamma analysis** (110-140 Hz) - gold standard for visual decoding
- **Real-time classification** of 8 stimulus categories
- **Feature importance analysis** with brain region mapping

### 📊 **Comprehensive Analysis Pipeline**
- **252 trials** of ECoG data from 160 electrodes
- **156 good channels** (97.5% quality rate)
- **4.5-minute walking paradigm** video
- **30 video annotations** with precise timing
- **Multi-modal analysis** with statistical validation

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch Web Application
```bash
python app.py
```
**Access the application at:** `http://localhost:5001`

### 3. Explore the Interface
- **Home**: Project overview and competition status
- **Data Overview**: Interactive dataset exploration
- **Preprocessing**: Signal processing pipeline visualization
- **Feature Extraction**: Multi-approach feature analysis
- **Video Annotations**: Synchronized video with neural activity
- **ML Modelling**: Model performance and results
- **About**: Team information and methodology

## 📁 Project Structure

```
ecog-video-analysis/
├── app.py                          # Main Flask web application
├── templates/                      # HTML templates
│   ├── base.html                  # Base template with navigation
│   ├── home.html                  # Enhanced home page
│   ├── data_overview.html         # Interactive data exploration
│   ├── preprocessing.html         # Preprocessing pipeline
│   ├── feature_extraction.html    # Feature analysis
│   ├── video_annotations.html     # Video synchronization
│   ├── modelling.html             # ML model results
│   └── about.html                 # Team and methodology
├── static/                        # Static assets
│   ├── css/main.css              # Enhanced styling
│   ├── js/main.js                # Interactive functionality
│   └── images/                   # Project images
├── data/                          # Data directories
│   ├── raw/                      # Raw ECoG data (.mat files)
│   ├── preprocessed/             # Preprocessed data (experiment8)
│   ├── features/                 # Extracted features (experiment8)
│   └── models/                   # Trained models
├── results/                       # Analysis results
│   ├── 05_modelling/             # ML model results (experiment27)
│   ├── 06_video_analysis/        # Video analysis (experiment53)
│   ├── 07_feature_importance/    # Feature importance analysis
│   └── annotations/              # Video annotation data
├── src/                          # Source code modules
│   ├── utils/                    # Utility functions
│   ├── preprocessing/            # Preprocessing modules
│   ├── features/                 # Feature extraction
│   ├── models/                   # Machine learning models
│   ├── visualization/            # Visualization tools
│   └── run files/                # Execution scripts
├── docs/                         # Documentation
│   ├── ALL_BUGS_FIXED_SUMMARY.md
│   ├── CORRECTED_MULTICLASS_EXECUTION_GUIDE.md
│   └── EXECUTE_COMPREHENSIVE_ML_ANALYSIS.md
├── presentation/                 # Presentation materials
│   ├── NeuroPulse_ECoG video watching analysis.pdf
│   ├── docs/other_presentation.pdf
│   ├── img/framework.html
│   └── summary.txt
└── requirements.txt              # Python dependencies
```

## 🎯 Web Application Features

### 🏠 **Enhanced Home Page**
- **Modern Project Overview** with animated stats cards
- **IEEE-SMC-2025 Competition** section with timeline
- **Interactive feature highlights** with hover effects
- **Professional design** with gradient backgrounds

### 📊 **Data Overview**
- **Channel Quality Analysis** with 160-channel heatmap
- **Signal Quality Metrics** with real-time statistics
- **Electrode Grid Layout** (10x16 configuration)
- **Interactive visualizations** with Plotly.js

### ⚙️ **Preprocessing Pipeline**
- **Dataset Transformation** visualization
- **Channel Quality Heatmap** (all 160 channels)
- **Artifact Detection Results** with detailed statistics
- **Channel Correlation Matrix** analysis
- **Real-time progress tracking**

### 🔬 **Feature Extraction**
- **4 Different Approaches**: Comprehensive, EEGNet, Transformer, Template Correlation
- **Channel-wise Power Heatmap** (all 156 channels)
- **Feature Statistics** with real data from experiment8
- **Interactive approach switching** with dynamic visualizations

### 🎥 **Video Annotations**
- **7 Analysis Types**: Brain Atlas, Enhanced Brain, ERSP, Gait Phase, Motor Cortex, Object Detection, Anatomical Atlas
- **Compressed Video Streaming** (12-15MB files)
- **Real Annotation Statistics** with 30 annotations
- **Interactive Timeline** with category mapping
- **Video Comparison** (annotated vs raw)

### 🤖 **ML Modelling**
- **Real Model Results** from experiment27
- **Dynamic Visualizations**: Confusion matrices, ROC curves, correlation matrices
- **Performance Metrics**: Accuracy, F1, AUC, Precision, Recall
- **Interactive Charts** with Plotly.js

## 🧠 Technical Implementation

### **Signal Processing Pipeline**
- **Bandpass Filtering**: 0.5-150 Hz with notch filters at 50/60 Hz
- **Artifact Rejection**: Automated bad channel detection (4 channels removed)
- **Trial Segmentation**: 100-400 ms post-stimulus windows
- **Baseline Normalization**: Z-score normalization per trial
- **Quality Control**: 97.5% channel quality rate

### **Feature Extraction Methods**
1. **Comprehensive Features**: 5 frequency bands × 156 channels = 780 features
2. **EEGNet CNN**: 13×13 grid with 2× augmentation, 480 timepoints
3. **Transformer**: Multi-scale temporal features with 8 attention heads
4. **Template Correlation**: LOOCV with stimulus templates

### **Machine Learning Models**
- **EEGNet**: Compact CNN for neural decoding
- **Transformer**: Long-range temporal attention
- **CSP + LDA**: Spatial filtering with linear classification
- **Template Correlation**: Gamma template matching

### **Visualization Technologies**
- **Plotly.js**: Interactive 3D brain maps and charts
- **Video.js**: Synchronized video playback
- **Chart.js**: Real-time data visualization
- **Bootstrap 5**: Responsive modern UI

## 📊 Dataset Information

### **ECoG Data**
- **Channels**: 160 electrodes (156 good channels)
- **Sampling Rate**: 1200 Hz
- **Duration**: 268.4 seconds (4.5 minutes)
- **Trials**: 252 stimulus presentations
- **File Size**: ~1.2 GB raw data

### **Video Annotations**
- **Total Annotations**: 30 stimulus events
- **Categories**: digit, kanji, face, body, object, hiragana, line, color
- **Duration**: 252 seconds (10-262 seconds)
- **Frame Rate**: 30 FPS
- **Precision**: Sub-second timing accuracy

### **Stimulus Categories**
- **digit**: Numbers (0-9) - 4 annotations
- **kanji**: Japanese characters - 4 annotations  
- **face**: Human faces - 4 annotations
- **body**: Human figures - 3 annotations
- **object**: Various objects - 4 annotations
- **hiragana**: Japanese characters - 3 annotations
- **line**: Line patterns - 4 annotations
- **color**: Color stimuli - 4 annotations

## 🏆 Competition Advantages

### **Technical Excellence**
- **Gold Standard Methods**: High-gamma analysis (110-140 Hz)
- **Multiple Algorithms**: 4 different ML approaches
- **Real-time Processing**: Optimized for live analysis
- **Comprehensive Evaluation**: Multiple metrics and interpretability

### **Innovation Features**
- **Interactive Web Application**: Professional presentation platform
- **Video Synchronization**: Neural activity linked to visual stimuli
- **3D Brain Visualization**: Spatial mapping of neural responses
- **Feature Importance Analysis**: Brain region specialization

### **Research Quality**
- **Reproducible Pipeline**: Complete with saved intermediate results
- **Open Source**: All code and data available
- **Documentation**: Comprehensive guides and summaries
- **Team Collaboration**: Multi-disciplinary approach

## 🚀 Getting Started

### **For Researchers**
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Launch web app: `python app.py`
4. Explore the interactive interface
5. Review the analysis results in `results/`

### **For Competition Judges**
1. Visit the live web application
2. Navigate through each analysis section
3. Review the real data visualizations
4. Examine the model performance metrics
5. Watch the synchronized video annotations

### **For Developers**
1. Check the source code in `src/` modules
2. Review the execution scripts in `src/run files/`
3. Examine the documentation in `docs/`
4. Run individual analysis components
5. Contribute to the open-source project

## 📈 Performance Results

### **Model Performance** (Experiment 27)
- **Best Accuracy**: >85% for visual category classification
- **Feature Importance**: Top channels identified for each category
- **Processing Speed**: Real-time capable (1200 Hz sampling)
- **Reproducibility**: Complete pipeline with saved results

### **Data Quality**
- **Channel Quality**: 97.5% (156/160 channels)
- **Signal-to-Noise**: Optimized through preprocessing
- **Temporal Precision**: Sub-second annotation accuracy
- **Spatial Coverage**: Full brain region mapping

## 🎯 Future Enhancements

### **Planned Features**
- **Real-time Classification**: Live stimulus decoding
- **Advanced Visualizations**: 3D brain connectivity maps
- **Mobile Support**: Responsive design optimization
- **API Integration**: RESTful endpoints for data access

### **Research Directions**
- **Multi-modal Analysis**: Integration with other neural signals
- **Temporal Dynamics**: Long-range temporal dependencies
- **Cross-subject Generalization**: Transfer learning approaches
- **Clinical Applications**: Real-world BCI implementations

## 📞 Support & Contact

### **Team Members**
- **Dimitrios Georgiou**: Web application, real-time analysis, preprocessing
- **Laura**: Preprocessing pipeline development
- **Aryan**: GitHub management, feature extraction
- **Helmy**: Model building and classification
- **Zoro**: Model building and classification

### **Resources**
- **GitHub Repository**: [https://github.com/jimmyg1997/ecog-video-analysis](https://github.com/jimmyg1997/ecog-video-analysis)
- **Documentation**: Check `docs/` folder for detailed guides
- **Presentation**: See `presentation/` folder for competition materials
- **Partners**: [https://www.br41n.io/IEEE-SMC-2025](https://www.br41n.io/IEEE-SMC-2025)

## 🏆 Competition Status

**✅ READY FOR IEEE-SMC-2025 COMPETITION!**

- **Phase 1**: Data Analysis ✅ Completed
- **Phase 2**: Feature Extraction ✅ Completed  
- **Phase 3**: Web Application ✅ In Progress
- **Phase 4**: Competition Submission 🔄 Upcoming
- **Phase 5**: Presentation Creation 🔄 In Progress

---

**🚀 Advanced ECoG Video Analysis Pipeline - IEEE-SMC-2025 Competition Ready! 🏆**