#!/usr/bin/env python3
"""
Skeleton Animation Module
Creates high-quality skeleton animation videos with action predictions
"""

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from pathlib import Path

class SkeletonAnimator:
    """
    Creates skeleton animation videos with real-time action predictions
    """
    
    def __init__(self, video_width=1280, video_height=720, fps=30):
        self.video_width = video_width
        self.video_height = video_height
        self.fps = fps
        
        # COCO-17 keypoint connections for human skeleton
        self.skeleton_connections = [
            # Head
            (0, 1), (0, 2), (1, 3), (2, 4),  # nose-eyes, eyes-ears
            
            # Torso
            (5, 6),   # left_shoulder - right_shoulder
            (5, 11),  # left_shoulder - left_hip
            (6, 12),  # right_shoulder - right_hip
            (11, 12), # left_hip - right_hip
            
            # Arms
            (5, 7), (7, 9),   # left arm: shoulder-elbow-wrist
            (6, 8), (8, 10),  # right arm: shoulder-elbow-wrist
            
            # Legs
            (11, 13), (13, 15),  # left leg: hip-knee-ankle
            (12, 14), (14, 16),  # right leg: hip-knee-ankle
        ]
        
        # Keypoint names in COCO-17 format
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Color scheme
        self.colors = {
            'skeleton': 'cyan',
            'joints': 'red',
            'text_bg': 'black',
            'text': 'white'
        }
    
    def create_skeleton_video(self, keypoint_data, predictions=None, frame_indices=None, 
                            output_path="results/videos/skeleton_animation.mp4"):
        """Create skeleton animation video with predictions"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Creating skeleton animation video: {output_path}")
        print(f"Total frames: {len(keypoint_data)}")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str(output_path), fourcc, self.fps, 
            (self.video_width, self.video_height)
        )
        
        # Normalize keypoint coordinates to video dimensions
        normalized_keypoints = self._normalize_keypoints(keypoint_data)
        
        # Create prediction mapping
        prediction_map = self._create_prediction_map(predictions, frame_indices, len(keypoint_data))
        
        # Generate frames
        for frame_idx in range(len(normalized_keypoints)):
            if frame_idx % 100 == 0:
                print(f"Processing frame {frame_idx}/{len(normalized_keypoints)}")
            
            # Create frame
            frame = self._create_frame(
                normalized_keypoints[frame_idx], 
                prediction_map.get(frame_idx, "Unknown"),
                frame_idx
            )
            
            # Write frame
            video_writer.write(frame)
        
        # Release video writer
        video_writer.release()
        print(f"Video saved: {output_path}")
        
        return output_path
    
    def _normalize_keypoints(self, keypoint_data):
        """Normalize keypoint coordinates to video dimensions"""
        normalized_data = []
        
        for frame_idx in range(len(keypoint_data)):
            frame_keypoints = []
            
            # Extract x, y coordinates for each keypoint
            for kp_name in self.keypoint_names:
                x_col = f"{kp_name}_x"
                y_col = f"{kp_name}_y"
                
                if x_col in keypoint_data.columns and y_col in keypoint_data.columns:
                    x = keypoint_data.iloc[frame_idx][x_col]
                    y = keypoint_data.iloc[frame_idx][y_col]
                    
                    # Normalize to video dimensions (assuming input is in pixel coordinates)
                    # Add some padding and center the skeleton
                    x_norm = int(x * 0.8 + self.video_width * 0.1)
                    y_norm = int(y * 0.8 + self.video_height * 0.1)
                    
                    frame_keypoints.append((x_norm, y_norm))
                else:
                    frame_keypoints.append((0, 0))  # Missing keypoint
            
            normalized_data.append(frame_keypoints)
        
        return normalized_data
    
    def _create_prediction_map(self, predictions, frame_indices, total_frames):
        """Create mapping from frame index to prediction"""
        prediction_map = {}
        
        if predictions is not None and frame_indices is not None:
            for pred, frame_idx in zip(predictions, frame_indices):
                if frame_idx < total_frames:
                    prediction_map[frame_idx] = pred
        
        return prediction_map
    
    def _create_frame(self, keypoints, prediction, frame_number):
        """Create a single video frame with skeleton and prediction"""
        # Create blank frame
        frame = np.zeros((self.video_height, self.video_width, 3), dtype=np.uint8)
        
        # Draw skeleton connections
        for connection in self.skeleton_connections:
            start_idx, end_idx = connection
            
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                start_point = keypoints[start_idx]
                end_point = keypoints[end_idx]
                
                # Only draw if both points are valid (not (0,0))
                if start_point != (0, 0) and end_point != (0, 0):
                    cv2.line(frame, start_point, end_point, (0, 255, 255), 3)  # Cyan line
        
        # Draw keypoints
        for i, (x, y) in enumerate(keypoints):
            if (x, y) != (0, 0):  # Only draw valid keypoints
                cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)  # Red circle
                
                # Label keypoints (optional, for debugging)
                if i in [0, 5, 6, 9, 10, 11, 12, 15, 16]:  # Key points only
                    cv2.putText(frame, str(i), (x-5, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Add prediction text
        self._add_prediction_text(frame, prediction, frame_number)
        
        # Add frame info
        self._add_frame_info(frame, frame_number)
        
        return frame
    
    def _add_prediction_text(self, frame, prediction, frame_number):
        """Add prediction text to frame"""
        # Background rectangle for text
        text_bg_height = 80
        cv2.rectangle(frame, (0, 0), (self.video_width, text_bg_height), 
                     (0, 0, 0), -1)
        
        # Prediction text
        text = f"Action: {prediction}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        color = (255, 255, 255)  # White
        thickness = 2
        
        # Get text size for centering
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (self.video_width - text_size[0]) // 2
        text_y = 40
        
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)
        
        # Add confidence bar or additional info if needed
        confidence_text = "Real-time Activity Recognition"
        conf_size = cv2.getTextSize(confidence_text, font, 0.6, 1)[0]
        conf_x = (self.video_width - conf_size[0]) // 2
        cv2.putText(frame, confidence_text, (conf_x, 65), font, 0.6, (200, 200, 200), 1)
    
    def _add_frame_info(self, frame, frame_number):
        """Add frame information"""
        # Frame counter
        frame_text = f"Frame: {frame_number:06d}"
        cv2.putText(frame, frame_text, (10, self.video_height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Timestamp (assuming 30 fps)
        seconds = frame_number / self.fps
        minutes = int(seconds // 60)
        seconds = seconds % 60
        time_text = f"Time: {minutes:02d}:{seconds:05.2f}"
        cv2.putText(frame, time_text, (self.video_width - 200, self.video_height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    def create_comparison_video(self, keypoint_data, baseline_predictions, optimized_predictions,
                              frame_indices, output_path="results/videos/model_comparison.mp4"):
        """Create side-by-side comparison video of two models"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Creating model comparison video: {output_path}")
        
        # Double width for side-by-side comparison
        comparison_width = self.video_width * 2
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str(output_path), fourcc, self.fps, 
            (comparison_width, self.video_height)
        )
        
        # Normalize keypoints
        normalized_keypoints = self._normalize_keypoints(keypoint_data)
        
        # Create prediction mappings
        baseline_map = self._create_prediction_map(baseline_predictions, frame_indices, len(keypoint_data))
        optimized_map = self._create_prediction_map(optimized_predictions, frame_indices, len(keypoint_data))
        
        # Generate comparison frames
        for frame_idx in range(len(normalized_keypoints)):
            if frame_idx % 100 == 0:
                print(f"Processing frame {frame_idx}/{len(normalized_keypoints)}")
            
            # Create baseline frame
            baseline_frame = self._create_frame(
                normalized_keypoints[frame_idx],
                f"Baseline: {baseline_map.get(frame_idx, 'Unknown')}",
                frame_idx
            )
            
            # Create optimized frame  
            optimized_frame = self._create_frame(
                normalized_keypoints[frame_idx],
                f"Optimized: {optimized_map.get(frame_idx, 'Unknown')}",
                frame_idx
            )
            
            # Combine frames side by side
            combined_frame = np.hstack([baseline_frame, optimized_frame])
            
            # Add center divider
            cv2.line(combined_frame, (self.video_width, 0), 
                    (self.video_width, self.video_height), (255, 255, 255), 2)
            
            video_writer.write(combined_frame)
        
        video_writer.release()
        print(f"Comparison video saved: {output_path}")
        
        return output_path
    
    def create_prediction_summary_video(self, predictions, frame_indices, class_names,
                                      output_path="results/videos/prediction_summary.mp4"):
        """Create a summary video showing prediction statistics"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Creating prediction summary video: {output_path}")
        
        # Calculate prediction statistics
        unique_predictions, counts = np.unique(predictions, return_counts=True)
        prediction_stats = dict(zip(unique_predictions, counts))
        
        # Create static summary frame
        summary_frame = self._create_summary_frame(prediction_stats, class_names)
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str(output_path), fourcc, self.fps, 
            (self.video_width, self.video_height)
        )
        
        # Write 5 seconds of the same frame
        for _ in range(self.fps * 5):
            video_writer.write(summary_frame)
        
        video_writer.release()
        print(f"Summary video saved: {output_path}")
        
        return output_path
    
    def _create_summary_frame(self, prediction_stats, class_names):
        """Create summary statistics frame"""
        frame = np.zeros((self.video_height, self.video_width, 3), dtype=np.uint8)
        
        # Title
        title = "Activity Recognition Summary"
        cv2.putText(frame, title, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        
        # Statistics
        y_offset = 120
        for activity, count in sorted(prediction_stats.items(), key=lambda x: x[1], reverse=True):
            percentage = count / sum(prediction_stats.values()) * 100
            text = f"{activity}: {count} frames ({percentage:.1f}%)"
            cv2.putText(frame, text, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
            y_offset += 40
        
        return frame 