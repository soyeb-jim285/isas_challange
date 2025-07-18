#!/usr/bin/env python3
"""
Complete ISAS Challenge Pipeline
Comprehensive solution for abnormal activity recognition following ABC paper methodology

This script runs the complete pipeline including:
- Data loading and validation
- Feature extraction
- Model training with LOSO cross-validation
- Comprehensive evaluation with metrics and visualizations
- Test data prediction
- Video generation with skeleton animation
- Result export in multiple formats

Usage:
    python run_complete_pipeline.py [--baseline-only] [--optimized-only] [--no-video]
"""

import sys
import argparse
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# Add pipeline to Python path
sys.path.append(str(Path(__file__).parent / "pipeline"))

# Import pipeline components
from data.data_loader import KeypointDataLoader
from data.feature_extractor import ABCFeatureExtractor
from models.lstm_model import ISASLSTMModel
from evaluation.metrics import ModelEvaluator
from visualization.skeleton_animator import SkeletonAnimator
from utils.prediction_exporter import PredictionExporter

def run_complete_pipeline(run_baseline=True, run_optimized=True, generate_video=True):
    """Run the complete ISAS challenge pipeline"""
    
    print("="*70)
    print("ISAS CHALLENGE: COMPREHENSIVE ABNORMAL ACTIVITY RECOGNITION PIPELINE")
    print("Following ABC Paper Methodology with LSTM and LOSO Cross-Validation")
    print("="*70)
    
    start_time = time.time()
    
    # ===== STEP 1: DATA LOADING AND VALIDATION =====
    print("\n🔄 STEP 1: Loading and validating data...")
    
    data_loader = KeypointDataLoader()
    
    # Load training data
    training_data = data_loader.load_training_data()
    class_names = data_loader.get_class_names()
    
    # Load test data
    test_data = data_loader.load_test_data()
    
    print(f"✅ Training data: {len(training_data)} frames from {len(data_loader.participants)} participants")
    print(f"✅ Test data: {len(test_data)} frames")
    print(f"✅ Classes: {len(class_names)} activities")
    
    # ===== STEP 2: FEATURE EXTRACTION =====
    print("\n🔄 STEP 2: Extracting ABC paper features...")
    
    feature_extractor = ABCFeatureExtractor(smooth_features=True)
    
    # Extract features for all participants
    all_features = []
    all_labels = []
    all_participants = []
    
    for participant in data_loader.participants:
        participant_data = training_data[training_data['participant'] == participant]
        features, labels = feature_extractor.extract_features_for_participant(participant_data)
        
        all_features.extend(features)
        all_labels.extend(labels)
        all_participants.extend([participant] * len(features))
        
        print(f"✅ Participant {participant}: {len(features)} feature vectors extracted")
    
    # Convert to numpy arrays
    import numpy as np
    features = np.array(all_features)
    labels = np.array(all_labels)
    participant_ids = np.array(all_participants)
    
    print(f"✅ Total features shape: {features.shape}")
    
    # Extract test features
    test_keypoint_cols = [col for col in test_data.columns if col.endswith('_x') or col.endswith('_y')]
    test_features = feature_extractor._calculate_pose_features(test_data[test_keypoint_cols])
    
    print(f"✅ Test features shape: {test_features.shape}")
    
    # ===== STEP 3: MODEL TRAINING AND EVALUATION =====
    results = {}
    
    if run_baseline:
        print("\n🔄 STEP 3A: Training Baseline LSTM Model...")
        
        # Baseline model with ABC paper parameters
        baseline_model = ISASLSTMModel(
            window_size=30,
            overlap_rate=0.5,
            lstm_units=64,
            learning_rate=0.001,
            batch_size=32,
            epochs=20
        )
        
        # Create sequences
        sequences, seq_labels, seq_participants = baseline_model.create_sequences(
            features, labels, participant_ids
        )
        
        # LOSO Cross-validation
        baseline_fold_results, baseline_accuracy, baseline_f1 = baseline_model.train_with_loso_cv(
            sequences, seq_labels, seq_participants
        )
        
        print(f"✅ Baseline LOSO CV - Accuracy: {baseline_accuracy:.4f}, F1: {baseline_f1:.4f}")
        
        # Train final model on all data
        baseline_model.train_final_model(sequences, seq_labels, seq_participants)
        
        # Save baseline model
        baseline_paths = baseline_model.save_model("results/models/baseline_lstm")
        
        # Make test predictions
        baseline_predictions, baseline_frame_indices, baseline_confidence = baseline_model.predict_test_data(test_features)
        
        results['baseline'] = {
            'model': baseline_model,
            'fold_results': baseline_fold_results,
            'accuracy': baseline_accuracy,
            'f1_score': baseline_f1,
            'predictions': baseline_predictions,
            'frame_indices': baseline_frame_indices,
            'confidence': baseline_confidence,
            'model_paths': baseline_paths
        }
        
        print(f"✅ Baseline predictions: {len(baseline_predictions)} generated")
    
    if run_optimized:
        print("\n🔄 STEP 3B: Training Optimized LSTM Model...")
        
        # Optimized model with enhanced parameters
        optimized_model = ISASLSTMModel(
            window_size=90,  # Larger window for better temporal context
            overlap_rate=0.7,  # More overlap for better coverage
            lstm_units=128,  # More units for better representation
            learning_rate=0.0005,  # Lower learning rate for stability
            batch_size=16,  # Smaller batch for better gradients
            epochs=25  # More epochs for convergence
        )
        
        # Create sequences with optimized parameters
        opt_sequences, opt_seq_labels, opt_seq_participants = optimized_model.create_sequences(
            features, labels, participant_ids
        )
        
        # LOSO Cross-validation
        optimized_fold_results, optimized_accuracy, optimized_f1 = optimized_model.train_with_loso_cv(
            opt_sequences, opt_seq_labels, opt_seq_participants
        )
        
        print(f"✅ Optimized LOSO CV - Accuracy: {optimized_accuracy:.4f}, F1: {optimized_f1:.4f}")
        
        # Train final model on all data
        optimized_model.train_final_model(opt_sequences, opt_seq_labels, opt_seq_participants)
        
        # Save optimized model
        optimized_paths = optimized_model.save_model("results/models/optimized_lstm")
        
        # Make test predictions
        optimized_predictions, optimized_frame_indices, optimized_confidence = optimized_model.predict_test_data(test_features)
        
        results['optimized'] = {
            'model': optimized_model,
            'fold_results': optimized_fold_results,
            'accuracy': optimized_accuracy,
            'f1_score': optimized_f1,
            'predictions': optimized_predictions,
            'frame_indices': optimized_frame_indices,
            'confidence': optimized_confidence,
            'model_paths': optimized_paths
        }
        
        print(f"✅ Optimized predictions: {len(optimized_predictions)} generated")
    
    # ===== STEP 4: COMPREHENSIVE EVALUATION =====
    print("\n🔄 STEP 4: Comprehensive model evaluation...")
    
    evaluator = ModelEvaluator(class_names=class_names)
    
    # Evaluate each model
    evaluation_results = {}
    
    if 'baseline' in results:
        print("\n📊 Evaluating Baseline Model...")
        baseline_metrics, baseline_fold_metrics = evaluator.evaluate_fold_results(
            results['baseline']['fold_results'], 
            save_dir="results/metrics/baseline"
        )
        evaluation_results['baseline'] = {
            'overall_metrics': baseline_metrics,
            'fold_metrics': baseline_fold_metrics
        }
    
    if 'optimized' in results:
        print("\n📊 Evaluating Optimized Model...")
        optimized_metrics, optimized_fold_metrics = evaluator.evaluate_fold_results(
            results['optimized']['fold_results'], 
            save_dir="results/metrics/optimized"
        )
        evaluation_results['optimized'] = {
            'overall_metrics': optimized_metrics,
            'fold_metrics': optimized_fold_metrics
        }
    
    # ===== STEP 5: PREDICTION EXPORT =====
    print("\n🔄 STEP 5: Exporting predictions...")
    
    exporter = PredictionExporter()
    
    if 'baseline' in results:
        baseline_exports = exporter.export_test_predictions(
            results['baseline']['predictions'],
            results['baseline']['frame_indices'],
            test_data,
            model_name="baseline_lstm",
            confidence_scores=results['baseline']['confidence']
        )
        print(f"✅ Baseline predictions exported: {len(baseline_exports)} files")
    
    if 'optimized' in results:
        optimized_exports = exporter.export_test_predictions(
            results['optimized']['predictions'],
            results['optimized']['frame_indices'],
            test_data,
            model_name="optimized_lstm",
            confidence_scores=results['optimized']['confidence']
        )
        print(f"✅ Optimized predictions exported: {len(optimized_exports)} files")
    
    # Export model comparison if both models were trained
    if 'baseline' in results and 'optimized' in results:
        comparison_exports = exporter.export_model_comparison(
            results['baseline']['predictions'],
            results['optimized']['predictions'],
            results['baseline']['frame_indices'],  # Assuming same frame indices
            test_data
        )
        print(f"✅ Model comparison exported")
    
    # ===== STEP 6: VIDEO GENERATION =====
    if generate_video:
        print("\n🔄 STEP 6: Generating skeleton animation videos...")
        
        animator = SkeletonAnimator(video_width=1280, video_height=720, fps=30)
        
        # Sample subset of test data for video (first 1000 frames to keep video manageable)
        video_frames = min(1000, len(test_data))
        test_sample = test_data.head(video_frames)
        
        if 'baseline' in results:
            print("🎬 Creating baseline model video...")
            baseline_video = animator.create_skeleton_video(
                test_sample,
                results['baseline']['predictions'],
                results['baseline']['frame_indices'],
                output_path="results/videos/baseline_skeleton_animation.mp4"
            )
            print(f"✅ Baseline video saved: {baseline_video}")
        
        if 'optimized' in results:
            print("🎬 Creating optimized model video...")
            optimized_video = animator.create_skeleton_video(
                test_sample,
                results['optimized']['predictions'],
                results['optimized']['frame_indices'],
                output_path="results/videos/optimized_skeleton_animation.mp4"
            )
            print(f"✅ Optimized video saved: {optimized_video}")
        
        # Create comparison video if both models exist
        if 'baseline' in results and 'optimized' in results:
            print("🎬 Creating model comparison video...")
            comparison_video = animator.create_comparison_video(
                test_sample,
                results['baseline']['predictions'],
                results['optimized']['predictions'],
                results['baseline']['frame_indices'],
                output_path="results/videos/model_comparison.mp4"
            )
            print(f"✅ Comparison video saved: {comparison_video}")
    
    # ===== STEP 7: FINAL SUMMARY =====
    print("\n🔄 STEP 7: Generating final summary...")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Create comprehensive summary
    summary_path = Path("results/PIPELINE_SUMMARY.txt")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("ISAS CHALLENGE PIPELINE EXECUTION SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Execution Time: {total_time/60:.2f} minutes\n")
        f.write(f"Training Data: {len(training_data)} frames from {len(data_loader.participants)} participants\n")
        f.write(f"Test Data: {len(test_data)} frames\n")
        f.write(f"Classes: {len(class_names)} activities\n")
        f.write(f"Feature Dimensions: {features.shape[1]} ABC paper features\n\n")
        
        if 'baseline' in results:
            f.write("BASELINE LSTM MODEL RESULTS:\n")
            f.write(f"  LOSO CV Accuracy: {results['baseline']['accuracy']:.4f}\n")
            f.write(f"  LOSO CV F1 Score: {results['baseline']['f1_score']:.4f}\n")
            f.write(f"  Test Predictions: {len(results['baseline']['predictions'])}\n")
            f.write(f"  Model Architecture: window_size=30, lstm_units=64\n\n")
        
        if 'optimized' in results:
            f.write("OPTIMIZED LSTM MODEL RESULTS:\n")
            f.write(f"  LOSO CV Accuracy: {results['optimized']['accuracy']:.4f}\n")
            f.write(f"  LOSO CV F1 Score: {results['optimized']['f1_score']:.4f}\n")
            f.write(f"  Test Predictions: {len(results['optimized']['predictions'])}\n")
            f.write(f"  Model Architecture: window_size=90, lstm_units=128\n\n")
        
        if 'baseline' in results and 'optimized' in results:
            accuracy_improvement = results['optimized']['accuracy'] - results['baseline']['accuracy']
            f1_improvement = results['optimized']['f1_score'] - results['baseline']['f1_score']
            f.write("MODEL COMPARISON:\n")
            f.write(f"  Accuracy Improvement: {accuracy_improvement:+.4f}\n")
            f.write(f"  F1 Score Improvement: {f1_improvement:+.4f}\n\n")
        
        f.write("GENERATED FILES:\n")
        f.write("  📊 Metrics and Visualizations: results/metrics/\n")
        f.write("  💾 Saved Models: results/models/\n")
        f.write("  📄 Predictions: results/predictions/\n")
        if generate_video:
            f.write("  🎬 Videos: results/videos/\n")
        f.write("\n")
        
        f.write("ABC PAPER METHODOLOGY COMPLIANCE:\n")
        f.write("  ✅ 14 pose features extracted\n")
        f.write("  ✅ LSTM sequence modeling\n")
        f.write("  ✅ LOSO cross-validation\n")
        f.write("  ✅ Temporal smoothing applied\n")
        f.write("  ✅ Class imbalance handling\n")
        f.write("  ✅ Comprehensive evaluation\n")
    
    print(f"✅ Pipeline summary saved: {summary_path}")
    
    # Print final results
    print("\n" + "="*70)
    print("🎉 PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    if 'baseline' in results:
        print(f"📊 Baseline LSTM: Accuracy={results['baseline']['accuracy']:.4f}, F1={results['baseline']['f1_score']:.4f}")
    
    if 'optimized' in results:
        print(f"📊 Optimized LSTM: Accuracy={results['optimized']['accuracy']:.4f}, F1={results['optimized']['f1_score']:.4f}")
    
    print(f"⏱️  Total execution time: {total_time/60:.2f} minutes")
    print(f"📁 All results saved in: ./results/")
    print("="*70)
    
    return results, evaluation_results

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="ISAS Challenge Complete Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_complete_pipeline.py                    # Run complete pipeline
  python run_complete_pipeline.py --baseline-only    # Run only baseline model
  python run_complete_pipeline.py --optimized-only   # Run only optimized model
  python run_complete_pipeline.py --no-video         # Skip video generation
        """
    )
    
    parser.add_argument('--baseline-only', action='store_true',
                       help='Run only the baseline LSTM model')
    parser.add_argument('--optimized-only', action='store_true',
                       help='Run only the optimized LSTM model')
    parser.add_argument('--no-video', action='store_true',
                       help='Skip video generation (faster execution)')
    
    args = parser.parse_args()
    
    # Determine which models to run
    if args.baseline_only:
        run_baseline = True
        run_optimized = False
    elif args.optimized_only:
        run_baseline = False
        run_optimized = True
    else:
        run_baseline = True
        run_optimized = True
    
    generate_video = not args.no_video
    
    try:
        results, evaluation_results = run_complete_pipeline(
            run_baseline=run_baseline,
            run_optimized=run_optimized,
            generate_video=generate_video
        )
        
        return 0  # Success
        
    except Exception as e:
        print(f"\n❌ Pipeline execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1  # Error

if __name__ == "__main__":
    sys.exit(main()) 