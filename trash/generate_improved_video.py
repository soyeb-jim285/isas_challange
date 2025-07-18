#!/usr/bin/env python3
"""
Generate improved skeleton video with better centering and visual enhancements.
"""

import cv2
import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple

class ImprovedSkeletonVideoGenerator:
    def __init__(self, keypoint_file: str, frame_mapping_file: str, output_path: str = "results/videos/"):
        """
        Initialize the improved skeleton video generator.
        
        Args:
            keypoint_file: Path to the keypoint CSV file
            frame_mapping_file: Path to the frame mapping CSV file with predictions for every frame
            output_path: Directory to save the output video
        """
        self.keypoint_file = keypoint_file
        self.frame_mapping_file = frame_mapping_file
        self.output_path = output_path
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Define skeleton connections (COCO format)
        self.skeleton_connections = [
            (0, 1),   # nose to left_eye
            (0, 2),   # nose to right_eye
            (1, 3),   # left_eye to left_ear
            (2, 4),   # right_eye to right_ear
            (5, 6),   # left_shoulder to right_shoulder
            (5, 7),   # left_shoulder to left_elbow
            (7, 9),   # left_elbow to left_wrist
            (6, 8),   # right_shoulder to right_elbow
            (8, 10),  # right_elbow to right_wrist
            (5, 11),  # left_shoulder to left_hip
            (6, 12),  # right_shoulder to right_hip
            (11, 12), # left_hip to right_hip
            (11, 13), # left_hip to left_knee
            (13, 15), # left_knee to left_ankle
            (12, 14), # right_hip to right_knee
            (14, 16), # right_knee to right_ankle
        ]
        
        # Define keypoint names in order
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Color scheme for different actions
        self.action_colors = {
            'Sitting quietly': (0, 255, 0),      # Green
            'Using phone': (255, 165, 0),        # Orange
            'Biting': (255, 0, 0),              # Red
            'Attacking': (128, 0, 128),         # Purple
            'Throwing things': (255, 255, 0),    # Yellow
            'Head banging': (0, 255, 255),      # Cyan
            'Eating snacks': (255, 192, 203),   # Pink
            'Walking': (0, 0, 255)              # Blue
        }
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load keypoint and frame mapping data."""
        print("Loading keypoint data...")
        keypoints_df = pd.read_csv(self.keypoint_file)
        
        print("Loading frame mapping data...")
        frame_mapping_df = pd.read_csv(self.frame_mapping_file)
        
        return keypoints_df, frame_mapping_df
    
    def fix_unknown_actions(self, frame_mapping_df: pd.DataFrame) -> pd.DataFrame:
        """Replace Unknown actions with the nearest known action."""
        print("Fixing Unknown actions...")
        
        # Find indices of known actions
        known_indices = frame_mapping_df[frame_mapping_df['predicted_action'] != 'Unknown'].index.tolist()
        
        if not known_indices:
            print("Warning: No known actions found!")
            return frame_mapping_df
        
        # Create a copy to modify
        fixed_df = frame_mapping_df.copy()
        
        # For each Unknown action, find the nearest known action
        for idx in frame_mapping_df[frame_mapping_df['predicted_action'] == 'Unknown'].index:
            # Find the nearest known action index
            distances = [abs(idx - known_idx) for known_idx in known_indices]
            nearest_idx = known_indices[distances.index(min(distances))]
            nearest_action = frame_mapping_df.loc[nearest_idx, 'predicted_action']
            
            # Replace Unknown with nearest known action
            fixed_df.loc[idx, 'predicted_action'] = nearest_action
        
        print(f"Fixed {len(frame_mapping_df[frame_mapping_df['predicted_action'] == 'Unknown'])} Unknown actions")
        return fixed_df
    
    def get_keypoint_coordinates(self, row: pd.Series) -> List[Tuple[int, int]]:
        """Extract keypoint coordinates from a row."""
        keypoints = []
        for i, name in enumerate(self.keypoint_names):
            x_col = f"{name}_x"
            y_col = f"{name}_y"
            
            if x_col in row and y_col in row:
                x = int(row[x_col])
                y = int(row[y_col])
                keypoints.append((x, y))
            else:
                keypoints.append((0, 0))  # Default if not found
                
        return keypoints
    
    def scale_keypoints_to_frame(self, keypoints: List[Tuple[int, int]], 
                                frame_width: int, frame_height: int) -> List[Tuple[int, int]]:
        """Scale keypoints to fit within the frame bounds and center them properly."""
        if not keypoints:
            return keypoints
        
        # Find current bounds
        x_coords = [x for x, y in keypoints if x > 0]
        y_coords = [y for x, y in keypoints if y > 0]
        
        if not x_coords or not y_coords:
            return keypoints
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # Calculate current range
        current_width = max_x - min_x
        current_height = max_y - min_y
        
        if current_width == 0 or current_height == 0:
            return keypoints
        
        # Calculate scale factors to fit in frame with padding
        padding = 100  # Increased padding for better centering
        scale_x = (frame_width - 2 * padding) / current_width
        scale_y = (frame_height - 2 * padding) / current_height
        scale = min(scale_x, scale_y, 1.0)  # Don't scale up, only down
        
        # Calculate center of the frame
        frame_center_x = frame_width // 2
        frame_center_y = frame_height // 2
        
        # Calculate center of the keypoints
        keypoint_center_x = (min_x + max_x) // 2
        keypoint_center_y = (min_y + max_y) // 2
        
        # Scale and center keypoints
        scaled_keypoints = []
        for x, y in keypoints:
            if x > 0 and y > 0:
                # Scale relative to keypoint center
                scaled_x = int((x - keypoint_center_x) * scale + frame_center_x)
                scaled_y = int((y - keypoint_center_y) * scale + frame_center_y)
                scaled_keypoints.append((scaled_x, scaled_y))
            else:
                scaled_keypoints.append((0, 0))
        
        return scaled_keypoints
    
    def draw_skeleton(self, frame: np.ndarray, keypoints: List[Tuple[int, int]], 
                     action: str, frame_num: int) -> np.ndarray:
        """Draw skeleton on the frame with improved visualization."""
        # Create a copy of the frame
        frame_with_skeleton = frame.copy()
        
        # Scale keypoints to fit in frame
        scaled_keypoints = self.scale_keypoints_to_frame(keypoints, frame.shape[1], frame.shape[0])
        
        # Draw skeleton connections first (behind keypoints)
        for connection in self.skeleton_connections:
            start_idx, end_idx = connection
            if (start_idx < len(scaled_keypoints) and end_idx < len(scaled_keypoints) and
                scaled_keypoints[start_idx][0] > 0 and scaled_keypoints[start_idx][1] > 0 and
                scaled_keypoints[end_idx][0] > 0 and scaled_keypoints[end_idx][1] > 0):
                
                start_point = scaled_keypoints[start_idx]
                end_point = scaled_keypoints[end_idx]
                # Draw thicker lines for better visibility
                cv2.line(frame_with_skeleton, start_point, end_point, (255, 255, 255), 4)
        
        # Draw keypoints on top
        for i, (x, y) in enumerate(scaled_keypoints):
            if x > 0 and y > 0:  # Only draw if coordinates are valid
                # Draw larger, more visible keypoints
                cv2.circle(frame_with_skeleton, (x, y), 6, (255, 255, 255), -1)
                cv2.circle(frame_with_skeleton, (x, y), 8, (0, 0, 0), 2)
        
        # Draw action label with improved styling
        color = self.action_colors.get(action, (255, 255, 255))
        label = f"{action}"
        
        # Get text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 3
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Draw background rectangle with rounded corners effect
        cv2.rectangle(frame_with_skeleton, 
                     (15, 15), 
                     (15 + text_width + 30, 15 + text_height + 30), 
                     (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(frame_with_skeleton, label, (30, 30 + text_height), 
                   font, font_scale, color, thickness)
        
        # Draw frame number with better positioning
        frame_label = f"Frame: {frame_num:,}"
        cv2.putText(frame_with_skeleton, frame_label, (20, frame_with_skeleton.shape[0] - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame_with_skeleton
    
    def generate_video(self, fps: int = 30, resolution: Tuple[int, int] = (854, 480)):
        """Generate the improved skeleton video with action labels."""
        print("Loading data...")
        keypoints_df, frame_mapping_df = self.load_data()
        
        # Fix Unknown actions
        frame_mapping_df = self.fix_unknown_actions(frame_mapping_df)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_filename = os.path.join(self.output_path, "improved_skeleton_action_video.mp4")
        out = cv2.VideoWriter(output_filename, fourcc, fps, resolution)
        
        print(f"Generating improved video with {len(keypoints_df)} frames...")
        print("This may take several minutes...")
        
        # Process each frame
        for frame_idx in range(len(keypoints_df)):
            if frame_idx % 5000 == 0:
                print(f"Processing frame {frame_idx}/{len(keypoints_df)} ({frame_idx/len(keypoints_df)*100:.1f}%)")
            
            # Get keypoint data for this frame
            keypoint_row = keypoints_df.iloc[frame_idx]
            keypoints = self.get_keypoint_coordinates(keypoint_row)
            
            # Get action for this frame
            action = frame_mapping_df.iloc[frame_idx]['predicted_action']
            
            # Create frame
            frame = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
            
            # Draw skeleton
            frame = self.draw_skeleton(frame, keypoints, action, frame_idx)
            
            # Write frame
            out.write(frame)
        
        # Release video writer
        out.release()
        print(f"Improved video saved to: {output_filename}")
        
        return output_filename

def main():
    """Main function to generate the improved skeleton video."""
    # File paths
    keypoint_file = "test data_keypoint.csv"
    frame_mapping_file = "results/predictions/baseline_lstm_frame_mapping_20250710_032920.csv"
    
    # Create generator
    generator = ImprovedSkeletonVideoGenerator(keypoint_file, frame_mapping_file)
    
    # Generate improved video
    output_file = generator.generate_video(fps=30, resolution=(854, 480))
    print(f"Improved video generation complete: {output_file}")

if __name__ == "__main__":
    main() 