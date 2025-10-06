#!/bin/bash

# COMPREHENSIVE ML ANALYSIS SCRIPT
# This script will run the fixed multiclass experiments with ALL ML visualizations

echo "🚀 Running Complete ML Analysis with ALL Visualizations"
echo "========================================================"
echo ""
echo "This will generate:"
echo "  ✅ ROC-AUC curves for all models and classes"
echo "  ✅ Precision-Recall curves"
echo "  ✅ Confusion matrices"
echo "  ✅ Class performance analysis"
echo "  ✅ Feature importance plots"
echo "  ✅ Learning curves"
echo "  ✅ Model comparison plots"
echo "  ✅ Per-class F1, Precision, Recall analysis"
echo "  ✅ Class distribution vs performance"
echo "  ✅ Performance metrics correlation"
echo ""
echo "Running fixed multiclass experiments..."

python run_fixed_multiclass_experiments.py --all

echo ""
echo "✅ Complete! Check results in results/05_modelling/latest_experiment/"
