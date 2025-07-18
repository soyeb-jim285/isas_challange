#!/usr/bin/env python3
"""
Prediction Export Utility
Handles exporting predictions in various formats with comprehensive metadata
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

class PredictionExporter:
    """
    Utility for exporting model predictions with metadata
    """
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def export_test_predictions(self, predictions, frame_indices, test_data, 
                              model_name="LSTM", confidence_scores=None,
                              save_dir="results/predictions"):
        """Export test predictions with comprehensive metadata"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Exporting {len(predictions)} predictions...")
        
        # Create predictions DataFrame
        predictions_df = self._create_predictions_dataframe(
            predictions, frame_indices, test_data, confidence_scores
        )
        
        # Export to different formats
        exports = {}
        
        # CSV export
        csv_path = save_dir / f"{model_name}_predictions_{self.timestamp}.csv"
        predictions_df.to_csv(csv_path, index=False)
        exports['csv'] = csv_path
        
        # JSON export
        json_path = save_dir / f"{model_name}_predictions_{self.timestamp}.json"
        self._export_json(predictions_df, json_path, model_name)
        exports['json'] = json_path
        
        # Summary export
        summary_path = save_dir / f"{model_name}_summary_{self.timestamp}.txt"
        self._export_summary(predictions_df, summary_path, model_name)
        exports['summary'] = summary_path
        
        # Create frame-by-frame mapping
        frame_mapping_path = save_dir / f"{model_name}_frame_mapping_{self.timestamp}.csv"
        frame_mapping = self._create_frame_mapping(predictions, frame_indices, len(test_data))
        frame_mapping.to_csv(frame_mapping_path, index=False)
        exports['frame_mapping'] = frame_mapping_path
        
        print(f"Predictions exported to {save_dir}")
        return exports
    
    def _create_predictions_dataframe(self, predictions, frame_indices, test_data, confidence_scores):
        """Create comprehensive predictions DataFrame"""
        # Base prediction data
        pred_data = {
            'frame_index': frame_indices,
            'predicted_action': predictions,
            'sequence_center_frame': frame_indices  # Frame at center of prediction window
        }
        
        # Add confidence scores if available
        if confidence_scores is not None:
            # Get max confidence for each prediction
            max_confidences = np.max(confidence_scores, axis=1)
            pred_data['confidence'] = max_confidences
            
            # Add confidence for each class
            class_names = sorted(set(predictions))
            for i, class_name in enumerate(class_names):
                if i < confidence_scores.shape[1]:
                    pred_data[f'confidence_{class_name}'] = confidence_scores[:, i]
        
        # Add corresponding test data features
        if len(frame_indices) > 0:
            # Get keypoint data for predicted frames
            test_frame_data = []
            for frame_idx in frame_indices:
                if frame_idx < len(test_data):
                    frame_row = test_data.iloc[frame_idx].copy()
                    test_frame_data.append(frame_row)
                else:
                    # Create empty row if frame index is out of bounds
                    empty_row = pd.Series({col: np.nan for col in test_data.columns})
                    test_frame_data.append(empty_row)
            
            test_df = pd.DataFrame(test_frame_data)
            
            # Combine prediction data with test data
            predictions_df = pd.DataFrame(pred_data)
            
            # Add key keypoint coordinates for reference
            key_keypoints = ['nose_x', 'nose_y', 'left_wrist_x', 'left_wrist_y', 
                           'right_wrist_x', 'right_wrist_y', 'left_ankle_x', 'left_ankle_y',
                           'right_ankle_x', 'right_ankle_y']
            
            for kp in key_keypoints:
                if kp in test_df.columns:
                    predictions_df[kp] = test_df[kp].values
        else:
            predictions_df = pd.DataFrame(pred_data)
        
        return predictions_df
    
    def _create_frame_mapping(self, predictions, frame_indices, total_frames):
        """Create frame-by-frame prediction mapping"""
        # Initialize all frames as "Unknown"
        frame_predictions = ["Unknown"] * total_frames
        
        # Map predictions to frame indices
        for pred, frame_idx in zip(predictions, frame_indices):
            if 0 <= frame_idx < total_frames:
                frame_predictions[frame_idx] = pred
        
        # Create DataFrame
        frame_mapping = pd.DataFrame({
            'frame_id': range(total_frames),
            'predicted_action': frame_predictions
        })
        
        return frame_mapping
    
    def _export_json(self, predictions_df, json_path, model_name):
        """Export predictions as JSON with metadata"""
        # Convert DataFrame to records
        predictions_records = predictions_df.to_dict('records')
        
        # Add metadata
        export_data = {
            'metadata': {
                'model_name': model_name,
                'export_timestamp': self.timestamp,
                'total_predictions': len(predictions_records),
                'export_format': 'ISAS_Challenge_Predictions_v1.0'
            },
            'predictions': predictions_records,
            'summary': {
                'unique_actions': sorted(predictions_df['predicted_action'].unique().tolist()),
                'action_counts': predictions_df['predicted_action'].value_counts().to_dict()
            }
        }
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif pd.isna(obj):
                return None
            return obj
        
        def deep_convert(obj):
            if isinstance(obj, dict):
                return {k: deep_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [deep_convert(v) for v in obj]
            else:
                return convert_numpy(obj)
        
        export_data = deep_convert(export_data)
        
        with open(json_path, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def _export_summary(self, predictions_df, summary_path, model_name):
        """Export prediction summary"""
        with open(summary_path, 'w') as f:
            f.write(f"=== {model_name} Prediction Summary ===\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Basic statistics
            f.write(f"Total Predictions: {len(predictions_df)}\n")
            f.write(f"Unique Actions: {predictions_df['predicted_action'].nunique()}\n\n")
            
            # Action distribution
            f.write("Action Distribution:\n")
            action_counts = predictions_df['predicted_action'].value_counts()
            for action, count in action_counts.items():
                percentage = count / len(predictions_df) * 100
                f.write(f"  {action}: {count} ({percentage:.1f}%)\n")
            
            # Confidence statistics (if available)
            if 'confidence' in predictions_df.columns:
                f.write(f"\nConfidence Statistics:\n")
                f.write(f"  Mean Confidence: {predictions_df['confidence'].mean():.3f}\n")
                f.write(f"  Min Confidence: {predictions_df['confidence'].min():.3f}\n")
                f.write(f"  Max Confidence: {predictions_df['confidence'].max():.3f}\n")
                f.write(f"  Std Confidence: {predictions_df['confidence'].std():.3f}\n")
            
            # Frame coverage
            if 'frame_index' in predictions_df.columns:
                f.write(f"\nFrame Coverage:\n")
                f.write(f"  First Frame: {predictions_df['frame_index'].min()}\n")
                f.write(f"  Last Frame: {predictions_df['frame_index'].max()}\n")
                f.write(f"  Frame Range: {predictions_df['frame_index'].max() - predictions_df['frame_index'].min() + 1}\n")
    
    def export_model_comparison(self, baseline_predictions, optimized_predictions, 
                              frame_indices, test_data, save_dir="results/predictions"):
        """Export comparison between two models"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print("Exporting model comparison...")
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            'frame_index': frame_indices,
            'baseline_prediction': baseline_predictions,
            'optimized_prediction': optimized_predictions,
            'predictions_match': baseline_predictions == optimized_predictions
        })
        
        # Export comparison
        comparison_path = save_dir / f"model_comparison_{self.timestamp}.csv"
        comparison_df.to_csv(comparison_path, index=False)
        
        # Create comparison summary
        summary_path = save_dir / f"comparison_summary_{self.timestamp}.txt"
        with open(summary_path, 'w') as f:
            f.write("=== Model Comparison Summary ===\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            total_predictions = len(comparison_df)
            matching_predictions = comparison_df['predictions_match'].sum()
            agreement_rate = matching_predictions / total_predictions * 100
            
            f.write(f"Total Predictions: {total_predictions}\n")
            f.write(f"Matching Predictions: {matching_predictions}\n")
            f.write(f"Agreement Rate: {agreement_rate:.1f}%\n\n")
            
            f.write("Baseline Model Distribution:\n")
            baseline_counts = comparison_df['baseline_prediction'].value_counts()
            for action, count in baseline_counts.items():
                percentage = count / total_predictions * 100
                f.write(f"  {action}: {count} ({percentage:.1f}%)\n")
            
            f.write("\nOptimized Model Distribution:\n")
            optimized_counts = comparison_df['optimized_prediction'].value_counts()
            for action, count in optimized_counts.items():
                percentage = count / total_predictions * 100
                f.write(f"  {action}: {count} ({percentage:.1f}%)\n")
            
            # Disagreement analysis
            disagreements = comparison_df[~comparison_df['predictions_match']]
            if len(disagreements) > 0:
                f.write(f"\nDisagreement Analysis:\n")
                f.write(f"Total Disagreements: {len(disagreements)}\n")
                
                # Most common disagreements
                disagreement_pairs = list(zip(disagreements['baseline_prediction'], 
                                            disagreements['optimized_prediction']))
                from collections import Counter
                common_disagreements = Counter(disagreement_pairs).most_common(5)
                
                f.write("Most Common Disagreements:\n")
                for (baseline, optimized), count in common_disagreements:
                    f.write(f"  {baseline} -> {optimized}: {count} times\n")
        
        print(f"Model comparison exported to {save_dir}")
        return comparison_path, summary_path
    
    def export_performance_report(self, fold_results, overall_metrics, 
                                save_dir="results/predictions"):
        """Export comprehensive performance report"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create detailed performance DataFrame
        performance_data = []
        
        for fold in fold_results:
            fold_data = {
                'participant': fold['participant'],
                'accuracy': fold['accuracy'],
                'f1_score': fold['f1_score'],
                'num_samples': len(fold['y_true'])
            }
            performance_data.append(fold_data)
        
        performance_df = pd.DataFrame(performance_data)
        
        # Add overall metrics
        performance_df.loc['overall'] = {
            'participant': 'ALL',
            'accuracy': overall_metrics['accuracy'],
            'f1_score': overall_metrics['f1_weighted'],
            'num_samples': sum(performance_df['num_samples'])
        }
        
        # Export performance data
        performance_path = save_dir / f"performance_report_{self.timestamp}.csv"
        performance_df.to_csv(performance_path, index=False)
        
        print(f"Performance report exported to {performance_path}")
        return performance_path 