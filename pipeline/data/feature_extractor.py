#!/usr/bin/env python3
"""
Feature Extraction Module
Implements ABC paper methodology for pose feature extraction
"""

import numpy as np
import pandas as pd
from scipy import signal

class ABCFeatureExtractor:
    """
    Feature extractor following ABC paper methodology
    Extracts 14 key features from pose keypoints
    """
    
    def __init__(self, smooth_features=True):
        self.smooth_features = smooth_features
        self.body_parts = [
            'right_wrist', 'left_wrist', 'right_ankle', 'left_ankle', 
            'right_shoulder', 'left_shoulder', 'right_eye', 'left_eye',
            'nose', 'right_hip', 'left_hip', 'right_knee', 'left_knee'
        ]
    
    def extract_features_for_participant(self, participant_data):
        """Extract features for a single participant"""
        keypoint_cols = [col for col in participant_data.columns 
                        if col.endswith('_x') or col.endswith('_y')]
        
        # Sort by frame_id to maintain temporal order
        participant_data = participant_data.sort_values('frame_id').reset_index(drop=True)
        
        # Extract features
        features = self._calculate_pose_features(participant_data[keypoint_cols])
        labels = participant_data['Action Label'].values
        
        return features, labels
    
    def _calculate_pose_features(self, keypoint_data):
        """Calculate 14 features from pose keypoints following ABC paper"""
        features = []
        
        # Get coordinates for key body parts
        coords = {}
        for part in self.body_parts:
            if f'{part}_x' in keypoint_data.columns and f'{part}_y' in keypoint_data.columns:
                coords[part] = {
                    'x': keypoint_data[f'{part}_x'].values,
                    'y': keypoint_data[f'{part}_y'].values
                }
        
        n_frames = len(keypoint_data)
        
        # Apply temporal smoothing if enabled
        if self.smooth_features:
            for part in coords:
                coords[part]['x'] = self._smooth_signal(coords[part]['x'])
                coords[part]['y'] = self._smooth_signal(coords[part]['y'])
        
        for i in range(n_frames):
            frame_features = self._extract_frame_features(coords, i)
            features.append(frame_features)
        
        return np.array(features)
    
    def _extract_frame_features(self, coords, frame_idx):
        """Extract 14 features for a single frame"""
        features = []
        
        # Features 1-4: Hand and foot speeds
        if frame_idx > 0:
            # Right hand speed
            right_hand_speed = self._calculate_speed(
                coords, 'right_wrist', frame_idx
            )
            
            # Left hand speed
            left_hand_speed = self._calculate_speed(
                coords, 'left_wrist', frame_idx
            )
            
            # Right foot speed
            right_foot_speed = self._calculate_speed(
                coords, 'right_ankle', frame_idx
            )
            
            # Left foot speed
            left_foot_speed = self._calculate_speed(
                coords, 'left_ankle', frame_idx
            )
        else:
            right_hand_speed = left_hand_speed = 0
            right_foot_speed = left_foot_speed = 0
        
        # Features 5-8: Hand and foot accelerations
        if frame_idx > 1:
            # Previous speeds
            prev_right_hand_speed = self._calculate_speed(
                coords, 'right_wrist', frame_idx - 1
            )
            prev_left_hand_speed = self._calculate_speed(
                coords, 'left_wrist', frame_idx - 1
            )
            prev_right_foot_speed = self._calculate_speed(
                coords, 'right_ankle', frame_idx - 1
            )
            prev_left_foot_speed = self._calculate_speed(
                coords, 'left_ankle', frame_idx - 1
            )
            
            right_hand_acceleration = right_hand_speed - prev_right_hand_speed
            left_hand_acceleration = left_hand_speed - prev_left_hand_speed
            right_foot_acceleration = right_foot_speed - prev_right_foot_speed
            left_foot_acceleration = left_foot_speed - prev_left_foot_speed
        else:
            right_hand_acceleration = left_hand_acceleration = 0
            right_foot_acceleration = left_foot_acceleration = 0
        
        # Features 9-10: Shoulder-wrist angles
        right_shoulder_wrist_angle = self._calculate_angle(
            coords, 'right_shoulder', 'right_wrist', frame_idx
        )
        left_shoulder_wrist_angle = self._calculate_angle(
            coords, 'left_shoulder', 'left_wrist', frame_idx
        )
        
        # Features 11-12: Hip-ankle angles
        right_hip_ankle_angle = self._calculate_angle(
            coords, 'right_hip', 'right_ankle', frame_idx
        )
        left_hip_ankle_angle = self._calculate_angle(
            coords, 'left_hip', 'left_ankle', frame_idx
        )
        
        # Features 13-14: Head orientation and body center
        head_orientation = self._calculate_head_orientation(coords, frame_idx)
        body_center_movement = self._calculate_body_center_movement(coords, frame_idx)
        
        # Combine all features
        features = [
            right_hand_speed, left_hand_speed,
            right_foot_speed, left_foot_speed,
            right_hand_acceleration, left_hand_acceleration,
            right_foot_acceleration, left_foot_acceleration,
            right_shoulder_wrist_angle, left_shoulder_wrist_angle,
            right_hip_ankle_angle, left_hip_ankle_angle,
            head_orientation, body_center_movement
        ]
        
        return features
    
    def _calculate_speed(self, coords, body_part, frame_idx):
        """Calculate speed for a body part"""
        if frame_idx == 0 or body_part not in coords:
            return 0
        
        dx = coords[body_part]['x'][frame_idx] - coords[body_part]['x'][frame_idx - 1]
        dy = coords[body_part]['y'][frame_idx] - coords[body_part]['y'][frame_idx - 1]
        
        return np.sqrt(dx**2 + dy**2)
    
    def _calculate_angle(self, coords, part1, part2, frame_idx):
        """Calculate angle between two body parts"""
        if part1 not in coords or part2 not in coords:
            return 0
        
        dx = coords[part2]['x'][frame_idx] - coords[part1]['x'][frame_idx]
        dy = coords[part2]['y'][frame_idx] - coords[part1]['y'][frame_idx]
        
        return np.arctan2(dy, dx)
    
    def _calculate_head_orientation(self, coords, frame_idx):
        """Calculate head orientation using eye positions"""
        if 'right_eye' not in coords or 'left_eye' not in coords:
            return 0
        
        dx = coords['right_eye']['x'][frame_idx] - coords['left_eye']['x'][frame_idx]
        dy = coords['right_eye']['y'][frame_idx] - coords['left_eye']['y'][frame_idx]
        
        return np.arctan2(dy, dx)
    
    def _calculate_body_center_movement(self, coords, frame_idx):
        """Calculate body center movement using shoulder midpoint"""
        if frame_idx == 0 or 'right_shoulder' not in coords or 'left_shoulder' not in coords:
            return 0
        
        # Current center
        curr_center_x = (coords['right_shoulder']['x'][frame_idx] + 
                        coords['left_shoulder']['x'][frame_idx]) / 2
        curr_center_y = (coords['right_shoulder']['y'][frame_idx] + 
                        coords['left_shoulder']['y'][frame_idx]) / 2
        
        # Previous center
        prev_center_x = (coords['right_shoulder']['x'][frame_idx - 1] + 
                        coords['left_shoulder']['x'][frame_idx - 1]) / 2
        prev_center_y = (coords['right_shoulder']['y'][frame_idx - 1] + 
                        coords['left_shoulder']['y'][frame_idx - 1]) / 2
        
        # Movement distance
        dx = curr_center_x - prev_center_x
        dy = curr_center_y - prev_center_y
        
        return np.sqrt(dx**2 + dy**2)
    
    def _smooth_signal(self, signal_data, window_size=5):
        """Apply temporal smoothing to signal"""
        if len(signal_data) < window_size:
            return signal_data
        
        # Apply moving average filter
        smoothed = signal.savgol_filter(signal_data, window_size, 2)
        return smoothed 