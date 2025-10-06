# 🚀 Enhanced Multiclass ECoG Experiments - Execution Guide

## 📊 **NEW ENHANCED FEATURES**

The enhanced multiclass experiment framework now includes comprehensive ML-oriented visualizations:

### 🎯 **New Visualizations Added:**
1. **ROC-AUC Curves** - For all models and classes with macro-averaged metrics
2. **Precision-Recall Curves** - Detailed per-class performance analysis
3. **Enhanced Confusion Matrices** - With proper class labels and color coding
4. **Class Performance Analysis** - Per-class F1, Precision, Recall analysis
5. **Model Comparison Plots** - Comprehensive comparison across all metrics
6. **Feature Extractor Performance** - Detailed analysis of feature extraction methods
7. **Performance Metrics Correlation** - Correlation analysis between different metrics
8. **Class Distribution Analysis** - Sample size vs performance correlation

### 📋 **Class Labels Included:**
- **7-Class Problem**: digit, kanji, face, body, object, hiragana, line
- **14-Class Problem**: 7 categories × 2 colors (gray/color versions)

---

## 🚀 **HOW TO EXECUTE THE ENHANCED EXPERIMENTS**

### **Option 1: Run All Enhanced Experiments (Recommended)**
```bash
python run_enhanced_multiclass_experiments.py --all
```

### **Option 2: Run Only 7-Class Experiments**
```bash
python run_enhanced_multiclass_experiments.py --7class
```

### **Option 3: Run Only 14-Class Experiments**
```bash
python run_enhanced_multiclass_experiments.py --14class
```

---

## 📁 **OUTPUT STRUCTURE**

After running the experiments, you'll find the following in `results/05_modelling/{experiment_id}/`:

### **📊 Reports:**
- `enhanced_multiclass_report_YYYYMMDD_HHMMSS.md` - Comprehensive text report
- `enhanced_multiclass_results_YYYYMMDD_HHMMSS.json` - Detailed results in JSON format

### **🎨 Visualizations:**
- `roc_auc_curves_7class.png/svg` - ROC-AUC curves for 7-class problem
- `roc_auc_curves_14class.png/svg` - ROC-AUC curves for 14-class problem
- `confusion_matrices_7class.png/svg` - Confusion matrices for 7-class problem
- `confusion_matrices_14class.png/svg` - Confusion matrices for 14-class problem
- `class_performance_analysis_7class.png/svg` - Class performance analysis for 7-class
- `class_performance_analysis_14class.png/svg` - Class performance analysis for 14-class
- `model_comparison_analysis.png/svg` - Comprehensive model comparison

---

## 🔍 **WHAT EACH VISUALIZATION SHOWS**

### **1. ROC-AUC Curves**
- **Purpose**: Shows the trade-off between sensitivity and specificity for each class
- **What to look for**: Higher curves and larger AUC values indicate better performance
- **Includes**: Macro-averaged AUC across all classes

### **2. Precision-Recall Curves**
- **Purpose**: Shows the trade-off between precision and recall for each class
- **What to look for**: Higher curves and larger Average Precision (AP) values
- **Useful for**: Imbalanced datasets (which ECoG data often is)

### **3. Enhanced Confusion Matrices**
- **Purpose**: Shows exactly which classes are being confused with which
- **What to look for**: Darker diagonal elements indicate better classification
- **Includes**: Proper class labels and color coding

### **4. Class Performance Analysis**
- **Purpose**: Detailed per-class performance metrics
- **Shows**: F1, Precision, Recall for each class across all models
- **Includes**: Class distribution vs performance correlation

### **5. Model Comparison Plots**
- **Purpose**: Comprehensive comparison of all models and feature extractors
- **Shows**: Accuracy, ROC-AUC, F1 scores, feature extractor performance
- **Includes**: Performance metrics correlation matrix

---

## ⚡ **QUICK START COMMANDS**

### **🚀 Run Everything (Full Analysis)**
```bash
# This will run all experiments and generate all visualizations
python run_enhanced_multiclass_experiments.py --all
```

### **🎯 Quick Test (7-Class Only)**
```bash
# This will run only 7-class experiments (faster)
python run_enhanced_multiclass_experiments.py --7class
```

### **🔬 Detailed Analysis (14-Class Only)**
```bash
# This will run only 14-class experiments (more complex)
python run_enhanced_multiclass_experiments.py --14class
```

---

## 📈 **EXPECTED RESULTS**

### **Performance Metrics You'll See:**
- **Accuracy**: Overall classification accuracy
- **ROC-AUC**: Area under ROC curve (macro-averaged)
- **F1 Macro**: Macro-averaged F1 score
- **Precision Macro**: Macro-averaged precision
- **Recall Macro**: Macro-averaged recall
- **Average Precision**: Area under precision-recall curve

### **Class Labels You'll See:**

#### **7-Class Problem:**
- `digit` (0): Digit stimuli
- `kanji` (1): Kanji character stimuli  
- `face` (2): Face stimuli
- `body` (3): Body stimuli
- `object` (4): Object stimuli
- `hiragana` (5): Hiragana character stimuli
- `line` (6): Line stimuli

#### **14-Class Problem:**
- `digit_gray` (0), `digit_color` (7): Digit stimuli in gray/color
- `kanji_gray` (1), `kanji_color` (8): Kanji stimuli in gray/color
- `face_gray` (2), `face_color` (9): Face stimuli in gray/color
- `body_gray` (3), `body_color` (10): Body stimuli in gray/color
- `object_gray` (4), `object_color` (11): Object stimuli in gray/color
- `hiragana_gray` (5), `hiragana_color` (12): Hiragana stimuli in gray/color
- `line_gray` (6), `line_color` (13): Line stimuli in gray/color

---

## 🎯 **INTERPRETATION GUIDE**

### **Best Performance Indicators:**
1. **High Accuracy** (>0.5 for multiclass is good)
2. **High ROC-AUC** (>0.7 is good, >0.8 is excellent)
3. **Balanced F1 scores** across all classes
4. **Clear diagonal** in confusion matrices
5. **Consistent performance** across different feature extractors

### **What to Look For:**
- **Which feature extractor works best?** (EEGNet, comprehensive, etc.)
- **Which model performs best?** (Random Forest, SVM, Logistic Regression)
- **Which classes are hardest to classify?** (low F1 scores)
- **Are there systematic confusions?** (off-diagonal patterns in confusion matrix)

---

## 🔧 **TROUBLESHOOTING**

### **If you get import errors:**
```bash
pip install scikit-learn matplotlib seaborn numpy pandas tqdm
```

### **If you get memory errors:**
- The experiments are designed to be memory-efficient
- If issues persist, try running one problem type at a time

### **If visualizations don't appear:**
- Check that the results directory was created
- Ensure you have write permissions in the results folder
- Look for error messages in the console output

---

## 📊 **SAMPLE OUTPUT**

After running, you should see output like:
```
🚀 Enhanced Multiclass Experiment Framework
📁 Experiment ID: experiment21
📁 Results Directory: results/05_modelling/experiment21

📊 Loading features and creating multiclass labels...
  ✅ Loaded comprehensive: (252, 64)
  ✅ Loaded eegnet: (252, 1280)
  ✅ Loaded template_correlation: (252, 32)
  ✅ Loaded transformer: (252, 256)
  ✅ Loaded trial labels: (252,)
  ✅ Loaded color labels: (252,)

🎯 Running 7class multiclass experiments...
  📊 7class problem:
    Classes: 7
    Samples: 125
    Class distribution: [16 32 25 29 5 3 15]

🎨 Creating enhanced visualizations...
🎨 Creating ROC-AUC plots for 7class...
  📊 ROC-AUC plots saved for 7class
🎨 Creating confusion matrices for 7class...
  📊 Confusion matrices saved for 7class
🎨 Creating class performance analysis for 7class...
  📊 Class performance analysis saved for 7class

🎉 All Enhanced Multiclass Experiments Completed!
🏆 Best Results:
  7CLASS: 0.563 (ROC-AUC: 0.742) - eegnet + Random Forest
  14CLASS: 0.544 (ROC-AUC: 0.698) - eegnet + Random Forest
```

---

## 🎯 **NEXT STEPS**

1. **Run the experiments** using one of the commands above
2. **Examine the visualizations** in the results directory
3. **Read the comprehensive report** for detailed analysis
4. **Compare with previous results** to see improvements
5. **Use insights** to guide further experiments or model improvements

The enhanced framework provides a complete ML analysis pipeline with publication-ready visualizations and comprehensive performance metrics!
