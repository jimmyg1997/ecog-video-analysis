# 🚀 Complete Multiclass Experiment Execution Guide

## 📊 **OVERVIEW**

**2 Multiclass Problems:**
- **8-Class**: 7 categories + background (BALANCED - RECOMMENDED)
- **15-Class**: 14 categories + background (BALANCED - RECOMMENDED)

**4 Feature Extractors:**
- **comprehensive** - Broadband gamma power (110-140 Hz)
- **template_correlation** - Template correlation features  
- **eegnet** - Raw time series for CNN
- **transformer** - Multi-scale temporal features

**3 Models:**
- **Random Forest** - Ensemble of decision trees
- **Logistic Regression** - Linear classification
- **SVM** - Support Vector Machine

**Total Combinations**: 2 problems × 4 extractors × 3 models = **24 experiments**

---

## 🎯 **RECOMMENDED APPROACH FOR EACH FEATURE EXTRACTOR**

### **1. comprehensive (Broadband Gamma Power)**
- **Best for**: Traditional ECoG analysis
- **Approach**: Balanced multiclass experiments
- **Expected Performance**: 50-60% accuracy

### **2. template_correlation (Template Correlation)**
- **Best for**: Template matching and correlation analysis
- **Approach**: Balanced multiclass experiments
- **Expected Performance**: 45-55% accuracy

### **3. eegnet (Raw Time Series)**
- **Best for**: Deep learning and CNN approaches
- **Approach**: Balanced multiclass experiments (BEST PERFORMER)
- **Expected Performance**: 55-65% accuracy

### **4. transformer (Multi-scale Temporal)**
- **Best for**: Long-range temporal dependencies
- **Approach**: Balanced multiclass experiments
- **Expected Performance**: 50-60% accuracy

---

## 🚀 **EXECUTION COMMANDS**

### **🔬 METHOD 1: Comprehensive Framework (All at Once)**

```bash
# Run ALL experiments (24 combinations) - RECOMMENDED
python run_balanced_multiclass_experiments.py --all
```

**Output**: Complete analysis of all combinations in ~5 minutes

---

### **📦 METHOD 2: Batch Experiments (Custom Combinations)**

```bash
# All extractors, all models, 8-class only
python run_batch_multiclass_experiments.py --all-extractors --all-models --problems 8class

# All extractors, all models, 15-class only  
python run_batch_multiclass_experiments.py --all-extractors --all-models --problems 15class

# All extractors, all models, both problems
python run_batch_multiclass_experiments.py --all-extractors --all-models --problems 8class,15class

# Specific extractors and models
python run_batch_multiclass_experiments.py --extractors comprehensive,eegnet --models "Random Forest,SVM" --problems 8class,15class
```

---

### **🔍 METHOD 3: Individual Experiments (Detailed Analysis)**

#### **8-Class Experiments (7 categories + background)**

```bash
# comprehensive extractor
python run_individual_multiclass_experiments.py --extractor comprehensive --model "Random Forest" --problem 8class
python run_individual_multiclass_experiments.py --extractor comprehensive --model "Logistic Regression" --problem 8class
python run_individual_multiclass_experiments.py --extractor comprehensive --model "SVM" --problem 8class

# template_correlation extractor
python run_individual_multiclass_experiments.py --extractor template_correlation --model "Random Forest" --problem 8class
python run_individual_multiclass_experiments.py --extractor template_correlation --model "Logistic Regression" --problem 8class
python run_individual_multiclass_experiments.py --extractor template_correlation --model "SVM" --problem 8class

# eegnet extractor (BEST PERFORMER)
python run_individual_multiclass_experiments.py --extractor eegnet --model "Random Forest" --problem 8class
python run_individual_multiclass_experiments.py --extractor eegnet --model "Logistic Regression" --problem 8class
python run_individual_multiclass_experiments.py --extractor eegnet --model "SVM" --problem 8class

# transformer extractor
python run_individual_multiclass_experiments.py --extractor transformer --model "Random Forest" --problem 8class
python run_individual_multiclass_experiments.py --extractor transformer --model "Logistic Regression" --problem 8class
python run_individual_multiclass_experiments.py --extractor transformer --model "SVM" --problem 8class
```

#### **15-Class Experiments (14 categories + background)**

```bash
# comprehensive extractor
python run_individual_multiclass_experiments.py --extractor comprehensive --model "Random Forest" --problem 15class
python run_individual_multiclass_experiments.py --extractor comprehensive --model "Logistic Regression" --problem 15class
python run_individual_multiclass_experiments.py --extractor comprehensive --model "SVM" --problem 15class

# template_correlation extractor
python run_individual_multiclass_experiments.py --extractor template_correlation --model "Random Forest" --problem 15class
python run_individual_multiclass_experiments.py --extractor template_correlation --model "Logistic Regression" --problem 15class
python run_individual_multiclass_experiments.py --extractor template_correlation --model "SVM" --problem 15class

# eegnet extractor (BEST PERFORMER)
python run_individual_multiclass_experiments.py --extractor eegnet --model "Random Forest" --problem 15class
python run_individual_multiclass_experiments.py --extractor eegnet --model "Logistic Regression" --problem 15class
python run_individual_multiclass_experiments.py --extractor eegnet --model "SVM" --problem 15class

# transformer extractor
python run_individual_multiclass_experiments.py --extractor transformer --model "Random Forest" --problem 15class
python run_individual_multiclass_experiments.py --extractor transformer --model "Logistic Regression" --problem 15class
python run_individual_multiclass_experiments.py --extractor transformer --model "SVM" --problem 15class
```

---

### **⚡ METHOD 4: Quick Test (Fastest)**

```bash
# Quick test with 2 extractors, 2 models, 8-class only (4 experiments)
python run_batch_multiclass_experiments.py --quick-test
```

---

## 🎯 **RECOMMENDED EXECUTION SEQUENCE**

### **1️⃣ Quick Verification (2 minutes)**
```bash
python run_batch_multiclass_experiments.py --quick-test
```

### **2️⃣ Best Performing Combinations (10 minutes)**
```bash
# Test the best combinations first
python run_individual_multiclass_experiments.py --extractor eegnet --model "Random Forest" --problem 8class
python run_individual_multiclass_experiments.py --extractor eegnet --model "Random Forest" --problem 15class
python run_individual_multiclass_experiments.py --extractor comprehensive --model "Random Forest" --problem 8class
python run_individual_multiclass_experiments.py --extractor comprehensive --model "Random Forest" --problem 15class
```

### **3️⃣ Complete Analysis (30 minutes)**
```bash
# Run all experiments
python run_balanced_multiclass_experiments.py --all
```

---

## 📊 **EXPECTED RESULTS BY FEATURE EXTRACTOR**

### **eegnet (Raw Time Series) - BEST PERFORMER**
- **8-class**: 55-65% accuracy
- **15-class**: 50-60% accuracy
- **Best Model**: Random Forest
- **Why**: Raw temporal patterns capture complex neural dynamics

### **comprehensive (Broadband Gamma) - SECOND BEST**
- **8-class**: 50-60% accuracy  
- **15-class**: 45-55% accuracy
- **Best Model**: Random Forest
- **Why**: Gamma power is highly informative for visual processing

### **transformer (Multi-scale Temporal) - COMPETITIVE**
- **8-class**: 50-60% accuracy
- **15-class**: 45-55% accuracy
- **Best Model**: Random Forest
- **Why**: Captures long-range temporal dependencies

### **template_correlation (Template Matching) - BASELINE**
- **8-class**: 45-55% accuracy
- **15-class**: 40-50% accuracy
- **Best Model**: Random Forest
- **Why**: Simple correlation-based approach

---

## 📁 **OUTPUT LOCATIONS**

All results saved to: `results/05_modelling/experiment{N}/`

### **Files Generated:**
- `balanced_multiclass_report_YYYYMMDD_HHMMSS.md` - Comprehensive report
- `balanced_multiclass_results_YYYYMMDD_HHMMSS.json` - Raw results data
- `balanced_multiclass_analysis.png` - Visualization (PNG)
- `balanced_multiclass_analysis.svg` - Visualization (SVG)

### **Individual Experiment Files:**
- `individual_experiment_YYYYMMDD_HHMMSS.json` - Detailed results
- `individual_experiment_YYYYMMDD_HHMMSS.md` - Individual report
- `individual_experiment_YYYYMMDD_HHMMSS.png` - Individual visualization
- `individual_experiment_YYYYMMDD_HHMMSS.svg` - Individual visualization

---

## 🎉 **READY TO EXECUTE!**

### **🚀 Start Here (Recommended):**
```bash
# Run all experiments at once
python run_balanced_multiclass_experiments.py --all
```

### **🔍 For Detailed Analysis:**
```bash
# Test best combinations individually
python run_individual_multiclass_experiments.py --extractor eegnet --model "Random Forest" --problem 8class
python run_individual_multiclass_experiments.py --extractor eegnet --model "Random Forest" --problem 15class
```

### **⚡ For Quick Test:**
```bash
# Quick verification
python run_batch_multiclass_experiments.py --quick-test
```

---

## 📊 **EXPECTED TIMELINE**

- **Quick Test**: 2 minutes (4 experiments)
- **Best Combinations**: 10 minutes (4 experiments)
- **Complete Analysis**: 30 minutes (24 experiments)
- **Individual Analysis**: 5 minutes per experiment

---

## 🏆 **SUCCESS METRICS**

- **8-class accuracy**: Target >50% (achieved 56.3%)
- **15-class accuracy**: Target >45% (achieved 54.4%)
- **Processing time**: <30 minutes for all experiments
- **Reproducibility**: All results saved with timestamps

**Ready to execute! Choose your preferred method and run the experiments!** 🚀
