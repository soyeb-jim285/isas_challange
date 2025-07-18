#!/usr/bin/env python3
"""
Evaluation Metrics Module
Comprehensive evaluation tools for model assessment
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, 
    accuracy_score, precision_score, recall_score
)
from pathlib import Path

class ModelEvaluator:
    """
    Comprehensive model evaluation with metrics and visualizations
    """
    
    def __init__(self, class_names=None):
        self.class_names = class_names
        self.results = {}
    
    def evaluate_fold_results(self, fold_results, save_dir="results/metrics"):
        """Evaluate LOSO cross-validation results"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print("=== Comprehensive Model Evaluation ===")
        
        # Combine all predictions
        all_y_true = np.concatenate([fold['y_true'] for fold in fold_results])
        all_y_pred = np.concatenate([fold['y_pred'] for fold in fold_results])
        
        # Calculate overall metrics
        overall_metrics = self._calculate_comprehensive_metrics(all_y_true, all_y_pred)
        
        # Per-fold analysis
        fold_metrics = self._analyze_per_fold_performance(fold_results)
        
        # Generate visualizations
        self._create_confusion_matrix(all_y_true, all_y_pred, save_dir)
        self._create_performance_plots(fold_results, save_dir)
        self._create_class_performance_analysis(all_y_true, all_y_pred, save_dir)
        
        # Save detailed report
        self._save_evaluation_report(overall_metrics, fold_metrics, save_dir)
        
        return overall_metrics, fold_metrics
    
    def _calculate_comprehensive_metrics(self, y_true, y_pred):
        """Calculate comprehensive evaluation metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
            'f1_micro': f1_score(y_true, y_pred, average='micro'),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted')
        }
        
        # Per-class metrics
        if self.class_names:
            class_report = classification_report(
                y_true, y_pred, 
                target_names=self.class_names,
                output_dict=True
            )
            metrics['per_class'] = class_report
        
        return metrics
    
    def _analyze_per_fold_performance(self, fold_results):
        """Analyze performance across different folds (participants)"""
        fold_metrics = []
        
        for fold in fold_results:
            participant = fold['participant']
            y_true = fold['y_true']
            y_pred = fold['y_pred']
            
            fold_metric = {
                'participant': participant,
                'accuracy': accuracy_score(y_true, y_pred),
                'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
                'num_samples': len(y_true)
            }
            
            fold_metrics.append(fold_metric)
        
        return fold_metrics
    
    def _create_confusion_matrix(self, y_true, y_pred, save_dir):
        """Create and save confusion matrix visualization"""
        plt.figure(figsize=(10, 8))
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        if self.class_names:
            labels = self.class_names
        else:
            labels = [f'Class {i}' for i in range(len(cm))]
        
        sns.heatmap(
            cm_normalized, 
            annot=True, 
            fmt='.2f', 
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels
        )
        
        plt.title('Normalized Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(save_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save raw confusion matrix
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        cm_df.to_csv(save_dir / 'confusion_matrix.csv')
        
        print(f"Confusion matrix saved to {save_dir}")
    
    def _create_performance_plots(self, fold_results, save_dir):
        """Create performance visualization plots"""
        # Per-participant performance
        participants = [fold['participant'] for fold in fold_results]
        accuracies = [fold['accuracy'] for fold in fold_results]
        f1_scores = [fold['f1_score'] for fold in fold_results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy per participant
        ax1.bar(participants, accuracies, color='skyblue', alpha=0.7)
        ax1.set_title('Accuracy per Participant (LOSO CV)')
        ax1.set_xlabel('Participant')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        for i, v in enumerate(accuracies):
            ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # F1 Score per participant
        ax2.bar(participants, f1_scores, color='lightcoral', alpha=0.7)
        ax2.set_title('F1 Score per Participant (LOSO CV)')
        ax2.set_xlabel('Participant')
        ax2.set_ylabel('F1 Score')
        ax2.set_ylim(0, 1)
        for i, v in enumerate(f1_scores):
            ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'participant_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Training history plots
        self._plot_training_histories(fold_results, save_dir)
    
    def _plot_training_histories(self, fold_results, save_dir):
        """Plot training histories for all folds"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, fold in enumerate(fold_results):
            history = fold['history']
            participant = fold['participant']
            color = colors[i % len(colors)]
            
            # Training/validation loss
            axes[0, 0].plot(history['loss'], color=color, alpha=0.7, 
                           label=f'Participant {participant} (train)')
            axes[0, 0].plot(history['val_loss'], color=color, alpha=0.7, 
                           linestyle='--', label=f'Participant {participant} (val)')
            
            # Training/validation accuracy
            axes[0, 1].plot(history['accuracy'], color=color, alpha=0.7,
                           label=f'Participant {participant} (train)')
            axes[0, 1].plot(history['val_accuracy'], color=color, alpha=0.7,
                           linestyle='--', label=f'Participant {participant} (val)')
        
        axes[0, 0].set_title('Training/Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        axes[0, 1].set_title('Training/Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        
        # Summary statistics
        all_accuracies = [fold['accuracy'] for fold in fold_results]
        all_f1_scores = [fold['f1_score'] for fold in fold_results]
        
        axes[1, 0].boxplot([all_accuracies, all_f1_scores], labels=['Accuracy', 'F1 Score'])
        axes[1, 0].set_title('Performance Distribution')
        axes[1, 0].set_ylabel('Score')
        
        # Performance correlation
        axes[1, 1].scatter(all_accuracies, all_f1_scores, alpha=0.7)
        axes[1, 1].set_xlabel('Accuracy')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].set_title('Accuracy vs F1 Score')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'training_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_class_performance_analysis(self, y_true, y_pred, save_dir):
        """Analyze per-class performance"""
        if not self.class_names:
            return
        
        # Get classification report
        report = classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            output_dict=True
        )
        
        # Extract per-class metrics
        classes = self.class_names
        precision = [report[cls]['precision'] for cls in classes]
        recall = [report[cls]['recall'] for cls in classes]
        f1 = [report[cls]['f1-score'] for cls in classes]
        support = [report[cls]['support'] for cls in classes]
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Precision by class
        axes[0, 0].barh(classes, precision, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Precision by Class')
        axes[0, 0].set_xlabel('Precision')
        axes[0, 0].set_xlim(0, 1)
        
        # Recall by class
        axes[0, 1].barh(classes, recall, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('Recall by Class')
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_xlim(0, 1)
        
        # F1 score by class
        axes[1, 0].barh(classes, f1, color='lightgreen', alpha=0.7)
        axes[1, 0].set_title('F1 Score by Class')
        axes[1, 0].set_xlabel('F1 Score')
        axes[1, 0].set_xlim(0, 1)
        
        # Support (number of samples) by class
        axes[1, 1].barh(classes, support, color='gold', alpha=0.7)
        axes[1, 1].set_title('Support by Class')
        axes[1, 1].set_xlabel('Number of Samples')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'class_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed classification report
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(save_dir / 'classification_report.csv')
    
    def _save_evaluation_report(self, overall_metrics, fold_metrics, save_dir):
        """Save comprehensive evaluation report"""
        report_path = save_dir / 'evaluation_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("=== ISAS LSTM Model Evaluation Report ===\n\n")
            
            # Overall metrics
            f.write("OVERALL PERFORMANCE:\n")
            f.write(f"Accuracy: {overall_metrics['accuracy']:.4f}\n")
            f.write(f"F1 Score (Weighted): {overall_metrics['f1_weighted']:.4f}\n")
            f.write(f"F1 Score (Macro): {overall_metrics['f1_macro']:.4f}\n")
            f.write(f"F1 Score (Micro): {overall_metrics['f1_micro']:.4f}\n")
            f.write(f"Precision (Weighted): {overall_metrics['precision_weighted']:.4f}\n")
            f.write(f"Recall (Weighted): {overall_metrics['recall_weighted']:.4f}\n\n")
            
            # Per-fold performance
            f.write("PER-PARTICIPANT PERFORMANCE (LOSO CV):\n")
            for fold_metric in fold_metrics:
                f.write(f"Participant {fold_metric['participant']}: ")
                f.write(f"Accuracy={fold_metric['accuracy']:.4f}, ")
                f.write(f"F1={fold_metric['f1_weighted']:.4f}, ")
                f.write(f"Samples={fold_metric['num_samples']}\n")
            
            f.write(f"\nAverage across participants:\n")
            avg_accuracy = np.mean([f['accuracy'] for f in fold_metrics])
            avg_f1 = np.mean([f['f1_weighted'] for f in fold_metrics])
            f.write(f"Average Accuracy: {avg_accuracy:.4f}\n")
            f.write(f"Average F1 Score: {avg_f1:.4f}\n")
            
            # Performance statistics
            std_accuracy = np.std([f['accuracy'] for f in fold_metrics])
            std_f1 = np.std([f['f1_weighted'] for f in fold_metrics])
            f.write(f"Std Accuracy: {std_accuracy:.4f}\n")
            f.write(f"Std F1 Score: {std_f1:.4f}\n")
        
        print(f"Evaluation report saved to {report_path}")
        
        # Save metrics as JSON for programmatic access
        import json
        metrics_dict = {
            'overall': overall_metrics,
            'per_participant': fold_metrics
        }
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Apply conversion recursively
        def deep_convert(obj):
            if isinstance(obj, dict):
                return {k: deep_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [deep_convert(v) for v in obj]
            else:
                return convert_numpy(obj)
        
        metrics_dict = deep_convert(metrics_dict)
        
        with open(save_dir / 'metrics.json', 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        return overall_metrics, fold_metrics 