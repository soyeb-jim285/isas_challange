#!/usr/bin/env python3
"""
Quick Pipeline Test
Tests basic functionality and imports before running the full pipeline
"""

import sys
from pathlib import Path

def test_imports():
    """Test all pipeline imports"""
    print("Testing pipeline imports...")
    
    try:
        # Add pipeline to path
        sys.path.append(str(Path(__file__).parent / "pipeline"))
        
        # Test data modules
        from data.data_loader import KeypointDataLoader
        from data.feature_extractor import ABCFeatureExtractor
        print("✅ Data modules imported successfully")
        
        # Test model modules
        from models.lstm_model import ISASLSTMModel
        print("✅ Model modules imported successfully")
        
        # Test evaluation modules
        from evaluation.metrics import ModelEvaluator
        print("✅ Evaluation modules imported successfully")
        
        # Test visualization modules
        from visualization.skeleton_animator import SkeletonAnimator
        print("✅ Visualization modules imported successfully")
        
        # Test utility modules
        from utils.prediction_exporter import PredictionExporter
        print("✅ Utility modules imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_loading():
    """Test basic data loading"""
    print("\nTesting data loading...")
    
    try:
        sys.path.append(str(Path(__file__).parent / "pipeline"))
        from data.data_loader import KeypointDataLoader
        
        # Create data loader
        loader = KeypointDataLoader()
        
        # Check if data files exist
        train_files_exist = all(
            (loader.data_dir / "keypointlabel" / f"keypoints_with_labels_{p}.csv").exists()
            for p in loader.participants
        )
        
        test_file_exists = Path("test data_keypoint.csv").exists()
        
        if train_files_exist:
            print("✅ Training data files found")
        else:
            print("⚠️  Some training data files missing")
        
        if test_file_exists:
            print("✅ Test data file found")
        else:
            print("⚠️  Test data file missing")
        
        return train_files_exist and test_file_exists
        
    except Exception as e:
        print(f"❌ Data loading test error: {e}")
        return False

def test_feature_extraction():
    """Test feature extraction"""
    print("\nTesting feature extraction...")
    
    try:
        import pandas as pd
        import numpy as np
        sys.path.append(str(Path(__file__).parent / "pipeline"))
        from data.feature_extractor import ABCFeatureExtractor
        
        # Create dummy data
        dummy_data = pd.DataFrame({
            'frame_id': range(10),
            'nose_x': np.random.rand(10) * 100,
            'nose_y': np.random.rand(10) * 100,
            'left_wrist_x': np.random.rand(10) * 100,
            'left_wrist_y': np.random.rand(10) * 100,
            'right_wrist_x': np.random.rand(10) * 100,
            'right_wrist_y': np.random.rand(10) * 100,
            'left_ankle_x': np.random.rand(10) * 100,
            'left_ankle_y': np.random.rand(10) * 100,
            'right_ankle_x': np.random.rand(10) * 100,
            'right_ankle_y': np.random.rand(10) * 100,
            'left_shoulder_x': np.random.rand(10) * 100,
            'left_shoulder_y': np.random.rand(10) * 100,
            'right_shoulder_x': np.random.rand(10) * 100,
            'right_shoulder_y': np.random.rand(10) * 100,
            'left_eye_x': np.random.rand(10) * 100,
            'left_eye_y': np.random.rand(10) * 100,
            'right_eye_x': np.random.rand(10) * 100,
            'right_eye_y': np.random.rand(10) * 100,
            'left_hip_x': np.random.rand(10) * 100,
            'left_hip_y': np.random.rand(10) * 100,
            'right_hip_x': np.random.rand(10) * 100,
            'right_hip_y': np.random.rand(10) * 100,
            'left_knee_x': np.random.rand(10) * 100,
            'left_knee_y': np.random.rand(10) * 100,
            'right_knee_x': np.random.rand(10) * 100,
            'right_knee_y': np.random.rand(10) * 100,
            'Action Label': ['Walking'] * 10
        })
        
        # Test feature extraction
        extractor = ABCFeatureExtractor()
        features, labels = extractor.extract_features_for_participant(dummy_data)
        
        print(f"✅ Feature extraction successful: {features.shape}")
        print(f"✅ Expected 14 features, got {features.shape[1]}")
        
        return features.shape[1] == 14
        
    except Exception as e:
        print(f"❌ Feature extraction test error: {e}")
        return False

def test_dependencies():
    """Test required dependencies"""
    print("\nTesting dependencies...")
    
    required_packages = [
        'numpy', 'pandas', 'sklearn', 'tensorflow', 
        'matplotlib', 'seaborn', 'cv2', 'scipy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("✅ All dependencies available")
        return True

def main():
    """Run all tests"""
    print("="*60)
    print("ISAS CHALLENGE PIPELINE VALIDATION")
    print("="*60)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Imports", test_imports),
        ("Data Loading", test_data_loading),
        ("Feature Extraction", test_feature_extraction),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if not result:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("🎉 All tests passed! Pipeline is ready to run.")
        print("Execute: python run_complete_pipeline.py")
    else:
        print("⚠️  Some tests failed. Please fix issues before running pipeline.")
    
    print("="*60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main()) 