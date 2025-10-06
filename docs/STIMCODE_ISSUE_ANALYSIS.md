# Stimcode Labeling Issue Analysis

## 🚨 Problem Identified

Your colleagues are absolutely correct! The preprocessing pipeline has a **critical flaw** in how it assigns stimulus labels to trials. Here's what's happening:

### Current Issue
- **Raw data has 4 different stimcode values**: [0, 1, 2, 3] with proper distribution
- **Preprocessed data has only 1 stimcode value**: [2] for all 252 trials
- **Root cause**: Trial onset detection is based on photodiode changes, not stimcode changes

### Technical Details

#### Raw Data Analysis
```
Stimcode Analysis:
- Unique values: [0, 1, 2, 3]
- Value counts: [1772, 11886, 302562, 5829]
- Number of stimcode changes: 4
- Change times: [0.1s, 10.0s, 262.1s, 267.0s]
```

#### Photodiode Analysis  
```
Photodiode Analysis:
- Number of photodiode changes: 252
- Change times: [10.45s, 11.42s, 12.32s, 13.39s, ...]
```

#### The Problem
The preprocessing pipeline:
1. Detects 252 trial onsets from photodiode signal changes
2. Assigns stimcode values at those onset times
3. **All 252 onsets occur during the period when stimcode = 2**
4. Result: All trials get label "2"

## 🔍 Why This Happened

Looking at the code in `src/preprocessing/comprehensive_preprocessor.py`:

```python
def _detect_trial_onsets(self, photodiode: np.ndarray, stimcode: np.ndarray) -> np.ndarray:
    """Detect trial onsets via photodiode signal."""
    # Find photodiode signal changes (stimulus onsets)
    photodiode_diff = np.abs(np.diff(photodiode))
    threshold = np.std(photodiode_diff) * 2
    onset_indices = np.where(np.abs(photodiode_diff) > threshold)[0]
    # ... filtering logic ...
    return np.array(filtered_onsets)

# Later in the pipeline:
'stimcode': stimcode[trial_onsets] if len(trial_onsets) > 0 else np.array([]),
```

**The issue**: Trial onsets are detected from photodiode changes, but stimcode values are sampled at those onset times. Since the photodiode changes occur during a period when stimcode = 2, all trials get the same label.

## 📊 Data Timeline Analysis

Based on the diagnostic results:

1. **0.1s**: First stimcode change (0→1)
2. **10.0s**: Second stimcode change (1→2) 
3. **10.45s**: First photodiode change (trial onset detection starts)
4. **262.1s**: Third stimcode change (2→3)
5. **267.0s**: Fourth stimcode change (3→0)

**The 252 photodiode changes (trial onsets) all occur between 10.45s and ~268s, which is during the period when stimcode = 2.**

## 💡 Solutions

### Option 1: Fix Trial Onset Detection (Recommended)
Modify the preprocessing to detect trial onsets based on **stimcode changes** instead of just photodiode changes:

```python
def _detect_trial_onsets_fixed(self, photodiode: np.ndarray, stimcode: np.ndarray) -> np.ndarray:
    """Detect trial onsets based on stimcode changes."""
    # Find stimcode changes (actual stimulus changes)
    stimcode_diff = np.abs(np.diff(stimcode))
    stimcode_changes = np.where(stimcode_diff > 0)[0]
    
    # Use photodiode to refine timing within each stimulus period
    # ... refined logic ...
    
    return np.array(trial_onsets)
```

### Option 2: Hybrid Approach
Combine both photodiode and stimcode information:
1. Use stimcode changes to identify stimulus periods
2. Use photodiode changes to identify precise trial onsets within each period
3. Assign the correct stimcode value to each trial

### Option 3: Manual Trial Alignment
Create a mapping between photodiode-detected onsets and the correct stimcode periods.

## 🚀 Recommended Action

**You need to re-execute the preprocessing** with a corrected trial onset detection algorithm. The current preprocessing is fundamentally flawed and will not produce meaningful classification results.

### Immediate Steps:
1. **Fix the trial onset detection logic** in `comprehensive_preprocessor.py`
2. **Re-run preprocessing** for all experiments
3. **Verify** that the new stimcode.npy files contain multiple unique values
4. **Re-run feature extraction** and modeling with corrected labels

### Expected Results After Fix:
- Stimcode should have multiple unique values (0, 1, 2, 3)
- Proper distribution across trials
- Meaningful classification results

## 📈 Impact on Your Analysis

Your current analysis is **invalid** because:
- All trials have the same label (2)
- No meaningful classification can be performed
- Feature extraction is working on mislabeled data
- Model training is essentially learning to predict a constant value

**This explains why your colleagues are questioning the results - the preprocessing fundamentally broke the stimulus labeling.**

## 🔧 Next Steps

1. **Acknowledge the issue** - Your colleagues are correct
2. **Fix the preprocessing pipeline** 
3. **Re-run the entire pipeline** from preprocessing through modeling
4. **Validate results** with proper multi-class labels

The good news is that your feature extraction and modeling pipelines are likely working correctly - the issue is purely in the preprocessing stage where trial labels are assigned.
