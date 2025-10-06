# 🎬 **RUN ALL APPROACHES TOGETHER - COMPLETE GUIDE**

## ✅ **YES! You can now run all approaches together under the same experiment!**

I've created **two ways** to run all approaches together:

---

## 🚀 **METHOD 1: Using the Original Script (Recommended)**

### **Run ALL Approaches (1-6):**
```bash
python run_video_annotation_experiments.py --approach 0 --duration 20
```

### **Run Specific Approaches:**
```bash
# Run approaches 1, 3, and 5 only
python run_video_annotation_experiments.py --approach 1 --duration 20
python run_video_annotation_experiments.py --approach 3 --duration 20
python run_video_annotation_experiments.py --approach 5 --duration 20
```

---

## 🚀 **METHOD 2: Using the New Dedicated Script**

### **Run ALL Approaches (1-6):**
```bash
python run_all_approaches.py --duration 20
```

### **Run Specific Approaches:**
```bash
# Run approaches 1, 3, and 5 only
python run_all_approaches.py --approaches "1,3,5" --duration 20

# Run approaches 2 and 4 only
python run_all_approaches.py --approaches "2,4" --duration 30
```

---

## 📁 **EXPERIMENT ORGANIZATION**

### **All videos are saved under the same experiment folder:**
```
results/06_video_analysis/experiment3/
├── walk_annotated_motor_cortex_exp3.mp4
├── walk_annotated_gait_phase_exp3.mp4
├── walk_annotated_ersp_exp3.mp4
├── walk_annotated_enhanced_brain_exp3.mp4
├── walk_annotated_object_detection_exp3.mp4
├── walk_annotated_brain_atlas_exp3.mp4
└── experiment_info.json
```

### **Each experiment gets a unique number:**
- **experiment1/**, **experiment2/**, **experiment3/**, etc.
- **Automatic numbering** - no manual folder creation needed
- **All approaches** in the same experiment folder
- **Metadata file** with experiment details

---

## 🎯 **RECOMMENDED USAGE**

### **For Testing All Approaches:**
```bash
# Quick test (20 seconds)
python run_video_annotation_experiments.py --approach 0 --duration 20

# Full analysis (60 seconds)
python run_video_annotation_experiments.py --approach 0 --duration 60
```

### **For Specific Approach Testing:**
```bash
# Test only the new approaches
python run_all_approaches.py --approaches "5,6" --duration 20

# Test only the original approaches
python run_all_approaches.py --approaches "1,2,3,4" --duration 20
```

---

## 📊 **WHAT YOU'LL GET**

### **6 Different Annotated Videos:**
1. **Motor Cortex Activation** - Brain region activation with motor/visual/cognitive systems
2. **Gait Phase Analysis** - Neural signatures during walking phases
3. **ERSP Overlay** - Time-frequency spectrograms for electrode clusters
4. **Enhanced Brain Regions** - Comprehensive brain activity visualization
5. **Object Detection** - Real-time object detection with brain activation
6. **Brain Atlas** - Fusiform gyrus activation with connectome visualization

### **All with Enhanced Features:**
- ✅ **Fixed Unicode/ASCII issues** (no more "????" symbols)
- ✅ **Enhanced scaling** (meaningful brain activation values)
- ✅ **Clear object impact** (visible differences when objects are present)
- ✅ **Professional quality** (clean, organized interfaces)
- ✅ **Interpretable results** (meaningful metrics and visualizations)

---

## 🎬 **EXAMPLE EXECUTION**

```bash
$ python run_video_annotation_experiments.py --approach 0 --duration 20

🎬 Real-Time Video Annotation Experiments
==================================================
🚀 Running ALL approaches (1-6) for 20 seconds
📁 Experiment folder: experiment3

📊 ECoG data shape: (158, 322050)
📊 Video start time: 0.0
📝 Annotations: 30 items

🎬 Running Approach 1...
✅ Approach 1 completed: results/06_video_analysis/experiment3/walk_annotated_motor_cortex_exp3.mp4

🎬 Running Approach 2...
✅ Approach 2 completed: results/06_video_analysis/experiment3/walk_annotated_gait_phase_exp3.mp4

🎬 Running Approach 3...
✅ Approach 3 completed: results/06_video_analysis/experiment3/walk_annotated_ersp_exp3.mp4

🎬 Running Approach 4...
✅ Approach 4 completed: results/06_video_analysis/experiment3/walk_annotated_enhanced_brain_exp3.mp4

🎬 Running Approach 5...
✅ Approach 5 completed: results/06_video_analysis/experiment3/walk_annotated_object_detection_exp3.mp4

🎬 Running Approach 6...
✅ Approach 6 completed: results/06_video_analysis/experiment3/walk_annotated_brain_atlas_exp3.mp4

🎉 Experiment 3 completed!
📁 Files saved to: results/06_video_analysis/experiment3/
📊 Successfully completed 6 approaches:
  ✅ Approach 1: Spatial Motor Cortex Activation Map
     📹 Video: walk_annotated_motor_cortex_exp3.mp4 (45.2 MB)
  ✅ Approach 2: Gait-Phase Neural Signature Timeline
     📹 Video: walk_annotated_gait_phase_exp3.mp4 (44.8 MB)
  ✅ Approach 3: Event-Related Spectral Perturbation (ERSP) Video Overlay
     📹 Video: walk_annotated_ersp_exp3.mp4 (45.1 MB)
  ✅ Approach 4: Enhanced Brain Region Activation
     📹 Video: walk_annotated_enhanced_brain_exp3.mp4 (44.9 MB)
  ✅ Approach 5: Real-Time Object Detection Annotation
     📹 Video: walk_annotated_object_detection_exp3.mp4 (45.3 MB)
  ✅ Approach 6: Brain Atlas Activation Overlay
     📹 Video: walk_annotated_brain_atlas_exp3.mp4 (45.0 MB)
```

---

## 🎯 **BENEFITS OF RUNNING ALL TOGETHER**

### **Efficiency:**
- **Single command** runs all approaches
- **Same experiment folder** for easy comparison
- **Consistent data** across all approaches
- **Automatic organization** with experiment numbering

### **Comparison:**
- **Side-by-side analysis** of all approaches
- **Same time periods** for fair comparison
- **Consistent annotation data** across all videos
- **Easy to identify** which approach works best for different scenarios

### **Organization:**
- **Clean experiment structure** with automatic numbering
- **All outputs in one place** for easy access
- **Metadata tracking** for experiment reproducibility
- **Professional workflow** for research and analysis

---

## 🚀 **READY TO RUN!**

**Choose your preferred method and run all approaches together:**

```bash
# Method 1 (Original script)
python run_video_annotation_experiments.py --approach 0 --duration 20

# Method 2 (Dedicated script)
python run_all_approaches.py --duration 20
```

**All approaches will be saved under the same experiment folder with enhanced, interpretable visualizations!** 🎬✨
