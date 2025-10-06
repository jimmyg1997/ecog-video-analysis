#!/usr/bin/env python3
"""
Multiclass Experiment Commands Generator
=====================================

This script shows all available commands for running multiclass experiments.
Provides comprehensive examples for different experiment configurations.

Usage:
    python show_multiclass_experiment_commands.py
"""

import sys
import os
sys.path.append('src')

def show_all_commands():
    """Show all available multiclass experiment commands."""
    
    print("🚀 MULTICLASS ECoG EXPERIMENT COMMANDS")
    print("=" * 60)
    
    print("\n📋 AVAILABLE SCRIPTS:")
    print("1. run_multiclass_experiments.py - Comprehensive framework")
    print("2. run_individual_multiclass_experiments.py - Individual experiments")
    print("3. run_batch_multiclass_experiments.py - Batch experiments")
    
    print("\n🎯 MULTICLASS PROBLEMS:")
    print("• 7-class: digit, kanji, face, body, object, hiragana, line")
    print("• 14-class: 7 categories × 2 colors (gray/color)")
    
    print("\n🔧 FEATURE EXTRACTORS:")
    print("• comprehensive - Broadband gamma power (110-140 Hz)")
    print("• template_correlation - Template correlation features")
    print("• eegnet - Raw time series for CNN")
    print("• transformer - Multi-scale temporal features")
    
    print("\n🤖 MODELS:")
    print("• Random Forest - Ensemble of decision trees")
    print("• Logistic Regression - Linear classification")
    print("• SVM - Support Vector Machine")
    
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE FRAMEWORK COMMANDS")
    print("=" * 60)
    
    print("\n🔬 Run All Experiments (Recommended):")
    print("python run_multiclass_experiments.py --all")
    print("  → Tests all combinations: 4 extractors × 3 models × 2 problems = 24 experiments")
    
    print("\n🎯 Run Only 7-Class Experiments:")
    print("python run_multiclass_experiments.py --7class")
    print("  → Tests: 4 extractors × 3 models × 1 problem = 12 experiments")
    
    print("\n🎨 Run Only 14-Class Experiments:")
    print("python run_multiclass_experiments.py --14class")
    print("  → Tests: 4 extractors × 3 models × 1 problem = 12 experiments")
    
    print("\n" + "=" * 60)
    print("🔍 INDIVIDUAL EXPERIMENT COMMANDS")
    print("=" * 60)
    
    print("\n📋 List Available Options:")
    print("python run_individual_multiclass_experiments.py --list-options")
    
    print("\n🎯 Individual 7-Class Experiments:")
    print("python run_individual_multiclass_experiments.py --extractor comprehensive --model \"Random Forest\" --problem 7class")
    print("python run_individual_multiclass_experiments.py --extractor eegnet --model SVM --problem 7class")
    print("python run_individual_multiclass_experiments.py --extractor transformer --model \"Logistic Regression\" --problem 7class")
    print("python run_individual_multiclass_experiments.py --extractor template_correlation --model \"Random Forest\" --problem 7class")
    
    print("\n🎨 Individual 14-Class Experiments:")
    print("python run_individual_multiclass_experiments.py --extractor comprehensive --model \"Random Forest\" --problem 14class")
    print("python run_individual_multiclass_experiments.py --extractor eegnet --model SVM --problem 14class")
    print("python run_individual_multiclass_experiments.py --extractor transformer --model \"Logistic Regression\" --problem 14class")
    print("python run_individual_multiclass_experiments.py --extractor template_correlation --model \"Random Forest\" --problem 14class")
    
    print("\n" + "=" * 60)
    print("📦 BATCH EXPERIMENT COMMANDS")
    print("=" * 60)
    
    print("\n🚀 Quick Test (Fastest):")
    print("python run_batch_multiclass_experiments.py --quick-test")
    print("  → Tests: 2 extractors × 2 models × 1 problem = 4 experiments")
    
    print("\n🎯 Custom Batch - 7-Class Only:")
    print("python run_batch_multiclass_experiments.py --extractors comprehensive,eegnet --models \"Random Forest,SVM\" --problems 7class")
    print("python run_batch_multiclass_experiments.py --all-extractors --all-models --problems 7class")
    
    print("\n🎨 Custom Batch - 14-Class Only:")
    print("python run_batch_multiclass_experiments.py --extractors comprehensive,eegnet --models \"Random Forest,SVM\" --problems 14class")
    print("python run_batch_multiclass_experiments.py --all-extractors --all-models --problems 14class")
    
    print("\n🔬 Custom Batch - Both Problems:")
    print("python run_batch_multiclass_experiments.py --extractors comprehensive,eegnet,transformer --models \"Random Forest,SVM\" --problems 7class,14class")
    print("python run_batch_multiclass_experiments.py --all-extractors --models \"Random Forest,Logistic Regression\" --problems 7class,14class")
    
    print("\n" + "=" * 60)
    print("📊 EXPERIMENT COMBINATIONS MATRIX")
    print("=" * 60)
    
    print("\n🔧 Feature Extractors × 🤖 Models × 🎯 Problems:")
    extractors = ['comprehensive', 'template_correlation', 'eegnet', 'transformer']
    models = ['Random Forest', 'Logistic Regression', 'SVM']
    problems = ['7class', '14class']
    
    print(f"\nTotal Combinations: {len(extractors)} × {len(models)} × {len(problems)} = {len(extractors) * len(models) * len(problems)}")
    
    print("\n📋 All Possible Combinations:")
    count = 1
    for extractor in extractors:
        for model in models:
            for problem in problems:
                print(f"{count:2d}. {extractor} + {model} + {problem}")
                count += 1
    
    print("\n" + "=" * 60)
    print("🎯 RECOMMENDED EXECUTION SEQUENCE")
    print("=" * 60)
    
    print("\n1️⃣ Quick Test (5 minutes):")
    print("   python run_batch_multiclass_experiments.py --quick-test")
    print("   → Verify everything works")
    
    print("\n2️⃣ Individual Best Combinations (10 minutes):")
    print("   python run_individual_multiclass_experiments.py --extractor comprehensive --model \"Random Forest\" --problem 7class")
    print("   python run_individual_multiclass_experiments.py --extractor eegnet --model SVM --problem 14class")
    print("   → Test specific promising combinations")
    
    print("\n3️⃣ Comprehensive Analysis (30 minutes):")
    print("   python run_multiclass_experiments.py --all")
    print("   → Full systematic evaluation")
    
    print("\n" + "=" * 60)
    print("📁 OUTPUT LOCATIONS")
    print("=" * 60)
    
    print("\n📊 Results will be saved to:")
    print("   results/05_modelling/experiment{N}/")
    print("   ├── multiclass_comprehensive_report_YYYYMMDD_HHMMSS.md")
    print("   ├── multiclass_comprehensive_results_YYYYMMDD_HHMMSS.json")
    print("   ├── multiclass_comprehensive_analysis.png")
    print("   ├── multiclass_comprehensive_analysis.svg")
    print("   ├── individual_experiment_YYYYMMDD_HHMMSS.json")
    print("   ├── individual_experiment_YYYYMMDD_HHMMSS.md")
    print("   ├── individual_experiment_YYYYMMDD_HHMMSS.png")
    print("   ├── individual_experiment_YYYYMMDD_HHMMSS.svg")
    print("   ├── batch_experiment_report_YYYYMMDD_HHMMSS.md")
    print("   ├── batch_experiment_results_YYYYMMDD_HHMMSS.json")
    print("   ├── batch_experiment_analysis_YYYYMMDD_HHMMSS.png")
    print("   └── batch_experiment_analysis_YYYYMMDD_HHMMSS.svg")
    
    print("\n" + "=" * 60)
    print("🎉 EXPECTED RESULTS")
    print("=" * 60)
    
    print("\n📊 Based on previous experiments:")
    print("• 7-class accuracy: 80-90% (best: ~88%)")
    print("• 14-class accuracy: 70-85% (best: ~85%)")
    print("• Best extractors: comprehensive, eegnet")
    print("• Best models: Random Forest, SVM")
    print("• Processing time: 1-5 minutes per experiment")
    
    print("\n🏆 Performance Targets:")
    print("• 7-class: >85% accuracy (beats paper's 72.9%)")
    print("• 14-class: >80% accuracy (beats paper's 52.1%)")
    print("• Real-time capability: <500ms prediction time")
    
    print("\n" + "=" * 60)
    print("🚀 READY TO START!")
    print("=" * 60)
    
    print("\n💡 Choose your approach:")
    print("• Quick test: python run_batch_multiclass_experiments.py --quick-test")
    print("• Full analysis: python run_multiclass_experiments.py --all")
    print("• Custom batch: python run_batch_multiclass_experiments.py --help")

if __name__ == "__main__":
    show_all_commands()
