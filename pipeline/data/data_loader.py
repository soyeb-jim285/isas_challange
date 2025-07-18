#!/usr/bin/env python3
"""
Data Loading and Preprocessing Module
Handles loading keypoint data and labels for ISAS Challenge
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class KeypointDataLoader:
    """
    Data loader for keypoint data with labels
    """
    
    def __init__(self, data_dir="Train Data"):
        self.data_dir = Path(data_dir)
        self.participants = [1, 2, 3, 5]
        self.combined_data = None
        
    def load_training_data(self):
        """Load and combine all training data"""
        print("Loading training data...")
        
        all_data = []
        
        for participant in self.participants:
            file_path = self.data_dir / "keypointlabel" / f"keypoints_with_labels_{participant}.csv"
            if file_path.exists():
                data = pd.read_csv(file_path)
                data['participant'] = participant
                all_data.append(data)
                print(f"Participant {participant}: {len(data)} frames")
            else:
                print(f"Warning: File not found for participant {participant}")
        
        if not all_data:
            raise FileNotFoundError("No training data files found")
        
        # Combine all data
        self.combined_data = pd.concat(all_data, ignore_index=True)
        
        # Clean data
        self._clean_data()
        
        return self.combined_data
    
    def load_test_data(self, test_file="test data_keypoint.csv"):
        """Load test data"""
        print(f"Loading test data from {test_file}...")
        
        test_data = pd.read_csv(test_file)
        print(f"Test data: {len(test_data)} frames")
        
        return test_data
    
    def _clean_data(self):
        """Clean and preprocess the data"""
        print("Cleaning data...")
        
        # Remove rows with missing labels
        original_length = len(self.combined_data)
        self.combined_data = self.combined_data.dropna(subset=['Action Label'])
        
        print(f"Total frames: {original_length}")
        print(f"After removing missing labels: {len(self.combined_data)}")
        
        # Standardize labels
        self.combined_data['Action Label'] = self.combined_data['Action Label'].replace('Throwing', 'Throwing things')
        
        # Sort by participant and frame_id for proper temporal order
        self.combined_data = self.combined_data.sort_values(['participant', 'frame_id']).reset_index(drop=True)
        
        # Show class distribution
        self._show_class_distribution()
    
    def _show_class_distribution(self):
        """Display class distribution"""
        print("\nClass distribution:")
        class_counts = self.combined_data['Action Label'].value_counts()
        for activity, count in class_counts.items():
            percentage = count / len(self.combined_data) * 100
            print(f"  {activity}: {count} frames ({percentage:.1f}%)")
    
    def get_keypoint_columns(self):
        """Get list of keypoint coordinate columns"""
        if self.combined_data is None:
            raise ValueError("Data not loaded. Call load_training_data() first.")
        
        keypoint_cols = [col for col in self.combined_data.columns 
                        if col.endswith('_x') or col.endswith('_y')]
        return keypoint_cols
    
    def get_class_names(self):
        """Get unique class names"""
        if self.combined_data is None:
            raise ValueError("Data not loaded. Call load_training_data() first.")
        
        return sorted(self.combined_data['Action Label'].unique()) 