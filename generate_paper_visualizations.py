#!/usr/bin/env python3
"""
Generate visualizations for the research paper
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

def load_metrics():
    """Load baseline and optimized metrics"""
    baseline_path = Path("results/metrics/baseline/metrics.json")
    optimized_path = Path("results/metrics/optimized/metrics.json")
    
    with open(baseline_path) as f:
        baseline = json.load(f)
    
    with open(optimized_path) as f:
        optimized = json.load(f)
    
    return baseline, optimized

def create_performance_comparison_chart():
    """Create performance comparison chart"""
    baseline, optimized = load_metrics()
    
    metrics = ['accuracy', 'f1_weighted', 'f1_macro', 'precision_weighted', 'recall_weighted']
    metric_names = ['Accuracy', 'Weighted F1', 'Macro F1', 'Weighted Precision', 'Weighted Recall']
    
    baseline_values = [
        baseline['overall']['accuracy'],
        baseline['overall']['f1_weighted'],
        baseline['overall']['f1_macro'],
        baseline['overall']['precision_weighted'],
        baseline['overall']['recall_weighted']
    ]
    
    optimized_values = [
        optimized['overall']['accuracy'],
        optimized['overall']['f1_weighted'],
        optimized['overall']['f1_macro'],
        optimized['overall']['precision_weighted'],
        optimized['overall']['recall_weighted']
    ]
    
    x = np.arange(len(metric_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline', alpha=0.8, color='#2E86C1')
    bars2 = ax.bar(x + width/2, optimized_values, width, label='Optimized', alpha=0.8, color='#28B463')
    
    ax.set_xlabel('Performance Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Performance Comparison: Baseline vs Optimized LSTM', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    
    plt.tight_layout()
    plt.savefig('paper_figures/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('paper_figures/performance_comparison.pdf', bbox_inches='tight')
    plt.show()
    
    print("✓ Performance comparison chart saved")

def create_loso_comparison():
    """Create LOSO cross-validation comparison"""
    baseline, optimized = load_metrics()
    
    participants = [1, 2, 3, 4, 5]
    baseline_acc = [p['accuracy'] for p in baseline['per_participant']]
    baseline_f1 = [p['f1_weighted'] for p in baseline['per_participant']]
    optimized_acc = [p['accuracy'] for p in optimized['per_participant']]
    optimized_f1 = [p['f1_weighted'] for p in optimized['per_participant']]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy comparison
    x = np.arange(len(participants))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, baseline_acc, width, label='Baseline', alpha=0.8, color='#2E86C1')
    bars2 = ax1.bar(x + width/2, optimized_acc, width, label='Optimized', alpha=0.8, color='#28B463')
    
    ax1.set_xlabel('Participant', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('LOSO Cross-Validation: Accuracy', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'P{i}' for i in participants])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # F1-Score comparison
    bars3 = ax2.bar(x - width/2, baseline_f1, width, label='Baseline', alpha=0.8, color='#2E86C1')
    bars4 = ax2.bar(x + width/2, optimized_f1, width, label='Optimized', alpha=0.8, color='#28B463')
    
    ax2.set_xlabel('Participant', fontsize=12, fontweight='bold')
    ax2.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax2.set_title('LOSO Cross-Validation: F1-Score', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'P{i}' for i in participants])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    def add_value_labels(ax, bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    add_value_labels(ax1, bars1)
    add_value_labels(ax1, bars2)
    add_value_labels(ax2, bars3)
    add_value_labels(ax2, bars4)
    
    plt.tight_layout()
    plt.savefig('paper_figures/loso_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('paper_figures/loso_comparison.pdf', bbox_inches='tight')
    plt.show()
    
    print("✓ LOSO comparison chart saved")

def create_class_performance_comparison():
    """Create per-class performance comparison"""
    baseline, optimized = load_metrics()
    
    # Extract per-class F1 scores
    baseline_classes = baseline['overall']['per_class']
    optimized_classes = optimized['overall']['per_class']
    
    activities = ['Attacking', 'Biting', 'Eating snacks', 'Head banging', 
                  'Sitting quietly', 'Throwing things', 'Using phone', 'Walking']
    
    baseline_f1 = [baseline_classes[act]['f1-score'] for act in activities]
    optimized_f1 = [optimized_classes[act]['f1-score'] for act in activities]
    
    # Calculate improvements
    improvements = [(opt - base) for base, opt in zip(baseline_f1, optimized_f1)]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(activities))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, baseline_f1, width, label='Baseline', alpha=0.8, color='#2E86C1')
    bars2 = ax.bar(x + width/2, optimized_f1, width, label='Optimized', alpha=0.8, color='#28B463')
    
    ax.set_xlabel('Activity', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(activities, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add improvement annotations
    for i, (base, opt, imp) in enumerate(zip(baseline_f1, optimized_f1, improvements)):
        if imp > 0:
            ax.annotate(f'+{imp:.3f}', 
                       xy=(i, max(base, opt) + 0.02),
                       ha='center', va='bottom', 
                       fontsize=9, color='green', fontweight='bold')
        elif imp < 0:
            ax.annotate(f'{imp:.3f}', 
                       xy=(i, max(base, opt) + 0.02),
                       ha='center', va='bottom', 
                       fontsize=9, color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('paper_figures/class_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('paper_figures/class_performance_comparison.pdf', bbox_inches='tight')
    plt.show()
    
    print("✓ Class performance comparison chart saved")

def create_improvement_summary():
    """Create improvement summary visualization"""
    baseline, optimized = load_metrics()
    
    # Calculate overall improvements
    metrics = {
        'Accuracy': {
            'baseline': baseline['overall']['accuracy'],
            'optimized': optimized['overall']['accuracy']
        },
        'Weighted F1': {
            'baseline': baseline['overall']['f1_weighted'],
            'optimized': optimized['overall']['f1_weighted']
        },
        'Macro F1': {
            'baseline': baseline['overall']['f1_macro'],
            'optimized': optimized['overall']['f1_macro']
        }
    }
    
    improvements = {}
    for metric, values in metrics.items():
        improvement = values['optimized'] - values['baseline']
        improvement_pct = (improvement / values['baseline']) * 100
        improvements[metric] = {
            'absolute': improvement,
            'percentage': improvement_pct
        }
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Absolute improvements
    metric_names = list(improvements.keys())
    abs_improvements = [improvements[m]['absolute'] for m in metric_names]
    
    colors = ['#28B463' if x > 0 else '#E74C3C' for x in abs_improvements]
    bars1 = ax1.bar(metric_names, abs_improvements, color=colors, alpha=0.8)
    
    ax1.set_ylabel('Absolute Improvement', fontsize=12, fontweight='bold')
    ax1.set_title('Absolute Performance Improvements', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars1, abs_improvements):
        ax1.annotate(f'{val:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, val),
                    xytext=(0, 3 if val > 0 else -15),
                    textcoords="offset points",
                    ha='center', va='bottom' if val > 0 else 'top',
                    fontsize=10, fontweight='bold')
    
    # Percentage improvements
    pct_improvements = [improvements[m]['percentage'] for m in metric_names]
    colors2 = ['#28B463' if x > 0 else '#E74C3C' for x in pct_improvements]
    bars2 = ax2.bar(metric_names, pct_improvements, color=colors2, alpha=0.8)
    
    ax2.set_ylabel('Percentage Improvement (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Percentage Performance Improvements', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars2, pct_improvements):
        ax2.annotate(f'+{val:.2f}%' if val > 0 else f'{val:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, val),
                    xytext=(0, 3 if val > 0 else -15),
                    textcoords="offset points",
                    ha='center', va='bottom' if val > 0 else 'top',
                    fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('paper_figures/improvement_summary.png', dpi=300, bbox_inches='tight')
    plt.savefig('paper_figures/improvement_summary.pdf', bbox_inches='tight')
    plt.show()
    
    print("✓ Improvement summary chart saved")

def create_confusion_matrices():
    """Create confusion matrices for baseline and optimized models"""
    try:
        # Read confusion matrix data
        baseline_cm = pd.read_csv('results/metrics/baseline/confusion_matrix.csv', index_col=0)
        optimized_cm = pd.read_csv('results/metrics/optimized/confusion_matrix.csv', index_col=0)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Baseline confusion matrix
        sns.heatmap(baseline_cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=baseline_cm.columns, yticklabels=baseline_cm.index)
        ax1.set_title('Baseline LSTM Confusion Matrix', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Actual', fontsize=12, fontweight='bold')
        
        # Optimized confusion matrix  
        sns.heatmap(optimized_cm, annot=True, fmt='d', cmap='Greens', ax=ax2,
                   xticklabels=optimized_cm.columns, yticklabels=optimized_cm.index)
        ax2.set_title('Optimized LSTM Confusion Matrix', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Actual', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('paper_figures/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.savefig('paper_figures/confusion_matrices.pdf', bbox_inches='tight')
        plt.show()
        
        print("✓ Confusion matrices saved")
        
    except FileNotFoundError:
        print("⚠ Confusion matrix CSV files not found, skipping...")

def main():
    """Generate all visualizations for the paper"""
    print("="*60)
    print("GENERATING PAPER VISUALIZATIONS")
    print("="*60)
    
    # Create output directory
    Path("paper_figures").mkdir(exist_ok=True)
    
    # Generate all visualizations
    create_performance_comparison_chart()
    create_loso_comparison()
    create_class_performance_comparison()
    create_improvement_summary()
    create_confusion_matrices()
    
    print("\n" + "="*60)
    print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
    print("="*60)
    print("Files saved in 'paper_figures/' directory:")
    print("• performance_comparison.png/pdf")
    print("• loso_comparison.png/pdf")
    print("• class_performance_comparison.png/pdf")
    print("• improvement_summary.png/pdf")
    print("• confusion_matrices.png/pdf")

if __name__ == "__main__":
    main()