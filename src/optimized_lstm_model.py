#!/usr/bin/env python3
"""
ISAS Challenge: Optimized LSTM with Temporal Parameter Optimization
Following ABC Paper Methodology with LLM-guided Parameter Search
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import itertools
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

class OptimizedISASLSTM:
    """
    Optimized LSTM with temporal parameter optimization following ABC paper
    """
    
    def __init__(self):
        self.data_dir = Path("Train Data")
        self.participants = [1, 2, 3, 4, 5]
        
        # Parameter ranges for optimization (inspired by ABC paper)
        self.param_ranges = {
            'window_size': [60, 90, 120],  # ABC paper suggests 90 is optimal
            'overlap_rate': [0.3, 0.5, 0.7],
            'lstm_units': [64, 128],
            'learning_rate': [0.001, 0.0005],
            'batch_size': [16, 32]
        }
        
        # Current best parameters
        self.best_params = {
            'window_size': 90,
            'overlap_rate': 0.5,
            'lstm_units': 64,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 25
        }
        
        # Data containers
        self.combined_data = None
        self.features = None
        self.labels = None
        self.participant_ids = None
        
        print("Optimized ISAS LSTM Model initialized")
    
    def load_and_preprocess_data(self):
        """Load and preprocess the dataset with improved handling"""
        print("Loading and preprocessing data...")
        
        all_data = []
        
        for participant in self.participants:
            file_path = self.data_dir / "keypointlabel" / f"keypoints_with_labels_{participant}.csv"
            data = pd.read_csv(file_path)
            data['participant'] = participant
            all_data.append(data)
        
        # Combine all data
        self.combined_data = pd.concat(all_data, ignore_index=True)
        
        # Handle missing labels
        self.combined_data = self.combined_data.dropna(subset=['Action Label'])
        
        # Normalize labels
        self.combined_data['Action Label'] = self.combined_data['Action Label'].replace('Throwing', 'Throwing things')
        
        # Sort by participant and frame_id for proper temporal order
        self.combined_data = self.combined_data.sort_values(['participant', 'frame_id']).reset_index(drop=True)
        
        print(f"Total frames after preprocessing: {len(self.combined_data)}")
        
        # Analyze class distribution
        class_counts = self.combined_data['Action Label'].value_counts()
        print("\nClass distribution:")
        for activity, count in class_counts.items():
            print(f"  {activity}: {count} frames ({count/len(self.combined_data)*100:.1f}%)")
        
        return self.combined_data
    
    def extract_enhanced_features(self):
        """Extract enhanced features with better temporal modeling"""
        print("Extracting enhanced features...")
        
        all_features = []
        all_labels = []
        all_participants = []
        
        # Process each participant separately
        for participant in self.participants:
            participant_data = self.combined_data[self.combined_data['participant'] == participant].copy()
            participant_data = participant_data.sort_values('frame_id').reset_index(drop=True)
            
            # Extract keypoint coordinates
            keypoint_cols = [col for col in participant_data.columns if col.endswith('_x') or col.endswith('_y')]
            
            # Calculate enhanced features
            features = self._calculate_enhanced_pose_features(participant_data[keypoint_cols])
            labels = participant_data['Action Label'].values
            
            all_features.extend(features)
            all_labels.extend(labels)
            all_participants.extend([participant] * len(features))
        
        self.features = np.array(all_features)
        self.labels = np.array(all_labels)
        self.participant_ids = np.array(all_participants)
        
        print(f"Extracted enhanced features shape: {self.features.shape}")
        
        return self.features, self.labels
    
    def _calculate_enhanced_pose_features(self, keypoint_data):
        """Calculate enhanced 14+ features from pose keypoints"""
        features = []
        
        # Get coordinates
        coords = {}
        body_parts = ['right_wrist', 'left_wrist', 'right_ankle', 'left_ankle', 
                     'right_shoulder', 'left_shoulder', 'right_eye', 'left_eye',
                     'nose', 'right_hip', 'left_hip', 'right_knee', 'left_knee']
        
        for part in body_parts:
            coords[part] = {
                'x': keypoint_data[f'{part}_x'].values,
                'y': keypoint_data[f'{part}_y'].values
            }
        
        n_frames = len(keypoint_data)
        
        # Apply temporal smoothing
        for part in body_parts:
            coords[part]['x'] = self._smooth_signal(coords[part]['x'])
            coords[part]['y'] = self._smooth_signal(coords[part]['y'])
        
        for i in range(n_frames):
            frame_features = []
            
            # Original 14 features
            if i > 0:
                # Hand speeds
                right_hand_speed = np.sqrt(
                    (coords['right_wrist']['x'][i] - coords['right_wrist']['x'][i-1])**2 +
                    (coords['right_wrist']['y'][i] - coords['right_wrist']['y'][i-1])**2
                )
                left_hand_speed = np.sqrt(
                    (coords['left_wrist']['x'][i] - coords['left_wrist']['x'][i-1])**2 +
                    (coords['left_wrist']['y'][i] - coords['left_wrist']['y'][i-1])**2
                )
                
                # Foot speeds
                right_foot_speed = np.sqrt(
                    (coords['right_ankle']['x'][i] - coords['right_ankle']['x'][i-1])**2 +
                    (coords['right_ankle']['y'][i] - coords['right_ankle']['y'][i-1])**2
                )
                left_foot_speed = np.sqrt(
                    (coords['left_ankle']['x'][i] - coords['left_ankle']['x'][i-1])**2 +
                    (coords['left_ankle']['y'][i] - coords['left_ankle']['y'][i-1])**2
                )
            else:
                right_hand_speed = left_hand_speed = right_foot_speed = left_foot_speed = 0
            
            # Accelerations
            if i > 1:
                prev_right_hand_speed = np.sqrt(
                    (coords['right_wrist']['x'][i-1] - coords['right_wrist']['x'][i-2])**2 +
                    (coords['right_wrist']['y'][i-1] - coords['right_wrist']['y'][i-2])**2
                )
                prev_left_hand_speed = np.sqrt(
                    (coords['left_wrist']['x'][i-1] - coords['left_wrist']['x'][i-2])**2 +
                    (coords['left_wrist']['y'][i-1] - coords['left_wrist']['y'][i-2])**2
                )
                prev_right_foot_speed = np.sqrt(
                    (coords['right_ankle']['x'][i-1] - coords['right_ankle']['x'][i-2])**2 +
                    (coords['right_ankle']['y'][i-1] - coords['right_ankle']['y'][i-2])**2
                )
                prev_left_foot_speed = np.sqrt(
                    (coords['left_ankle']['x'][i-1] - coords['left_ankle']['x'][i-2])**2 +
                    (coords['left_ankle']['y'][i-1] - coords['left_ankle']['y'][i-2])**2
                )
                
                right_hand_acceleration = right_hand_speed - prev_right_hand_speed
                left_hand_acceleration = left_hand_speed - prev_left_hand_speed
                right_foot_acceleration = right_foot_speed - prev_right_foot_speed
                left_foot_acceleration = left_foot_speed - prev_left_foot_speed
            else:
                right_hand_acceleration = left_hand_acceleration = 0
                right_foot_acceleration = left_foot_acceleration = 0
            
            # Shoulder-wrist angles
            right_shoulder_wrist_angle = np.arctan2(
                coords['right_wrist']['y'][i] - coords['right_shoulder']['y'][i],
                coords['right_wrist']['x'][i] - coords['right_shoulder']['x'][i]
            ) * 180 / np.pi
            
            left_shoulder_wrist_angle = np.arctan2(
                coords['left_wrist']['y'][i] - coords['left_shoulder']['y'][i],
                coords['left_wrist']['x'][i] - coords['left_shoulder']['x'][i]
            ) * 180 / np.pi
            
            # Eye displacements
            if i > 0:
                right_eye_vertical_displacement = coords['right_eye']['y'][i] - coords['right_eye']['y'][i-1]
                left_eye_vertical_displacement = coords['left_eye']['y'][i] - coords['left_eye']['y'][i-1]
                right_eye_horizontal_displacement = coords['right_eye']['x'][i] - coords['right_eye']['x'][i-1]
                left_eye_horizontal_displacement = coords['left_eye']['x'][i] - coords['left_eye']['x'][i-1]
            else:
                right_eye_vertical_displacement = left_eye_vertical_displacement = 0
                right_eye_horizontal_displacement = left_eye_horizontal_displacement = 0
            
            # Enhanced features
            # 15. Head movement (nose displacement)
            if i > 0:
                head_movement = np.sqrt(
                    (coords['nose']['x'][i] - coords['nose']['x'][i-1])**2 +
                    (coords['nose']['y'][i] - coords['nose']['y'][i-1])**2
                )
            else:
                head_movement = 0
            
            # 16. Body center displacement
            if i > 0:
                body_center_x = (coords['right_hip']['x'][i] + coords['left_hip']['x'][i]) / 2
                body_center_y = (coords['right_hip']['y'][i] + coords['left_hip']['y'][i]) / 2
                prev_body_center_x = (coords['right_hip']['x'][i-1] + coords['left_hip']['x'][i-1]) / 2
                prev_body_center_y = (coords['right_hip']['y'][i-1] + coords['left_hip']['y'][i-1]) / 2
                
                body_center_displacement = np.sqrt(
                    (body_center_x - prev_body_center_x)**2 +
                    (body_center_y - prev_body_center_y)**2
                )
            else:
                body_center_displacement = 0
            
            # 17. Arm span
            arm_span = np.sqrt(
                (coords['right_wrist']['x'][i] - coords['left_wrist']['x'][i])**2 +
                (coords['right_wrist']['y'][i] - coords['left_wrist']['y'][i])**2
            )
            
            # 18. Leg span
            leg_span = np.sqrt(
                (coords['right_ankle']['x'][i] - coords['left_ankle']['x'][i])**2 +
                (coords['right_ankle']['y'][i] - coords['left_ankle']['y'][i])**2
            )
            
            # Combine all features (18 total)
            frame_features = [
                right_hand_speed, left_hand_speed, right_foot_speed, left_foot_speed,
                right_hand_acceleration, left_hand_acceleration, right_foot_acceleration, left_foot_acceleration,
                right_shoulder_wrist_angle, left_shoulder_wrist_angle,
                right_eye_vertical_displacement, left_eye_vertical_displacement,
                right_eye_horizontal_displacement, left_eye_horizontal_displacement,
                head_movement, body_center_displacement, arm_span, leg_span
            ]
            
            features.append(frame_features)
        
        return features
    
    def _smooth_signal(self, signal, window_size=5):
        """Apply temporal smoothing to reduce noise"""
        if len(signal) < window_size:
            return signal
        
        # Simple moving average
        smoothed = np.convolve(signal, np.ones(window_size)/window_size, mode='same')
        return smoothed
    
    def create_sequences(self, features, labels, participant_ids, window_size, overlap_rate):
        """Create sequences for LSTM training"""
        sequences = []
        sequence_labels = []
        sequence_participants = []
        
        step_size = max(1, int(window_size * (1 - overlap_rate)))
        
        for participant in np.unique(participant_ids):
            participant_mask = participant_ids == participant
            participant_features = features[participant_mask]
            participant_labels = labels[participant_mask]
            
            for i in range(0, len(participant_features) - window_size + 1, step_size):
                sequence = participant_features[i:i + window_size]
                # Use majority label in the sequence
                sequence_labels_slice = participant_labels[i:i + window_size]
                label = self._get_majority_label(sequence_labels_slice)
                
                sequences.append(sequence)
                sequence_labels.append(label)
                sequence_participants.append(participant)
        
        return np.array(sequences), np.array(sequence_labels), np.array(sequence_participants)
    
    def _get_majority_label(self, labels):
        """Get the majority label in a sequence"""
        unique_labels, counts = np.unique(labels, return_counts=True)
        return unique_labels[np.argmax(counts)]
    
    def optimize_temporal_parameters(self, max_trials=8):
        """Optimize temporal parameters using grid search"""
        print("Optimizing temporal parameters...")
        
        # Extract features once
        features, labels = self.extract_enhanced_features()
        
        # Define parameter combinations to try
        param_combinations = [
            {'window_size': 60, 'overlap_rate': 0.5, 'lstm_units': 64, 'learning_rate': 0.001, 'batch_size': 32},
            {'window_size': 90, 'overlap_rate': 0.5, 'lstm_units': 64, 'learning_rate': 0.001, 'batch_size': 32},
            {'window_size': 120, 'overlap_rate': 0.5, 'lstm_units': 64, 'learning_rate': 0.001, 'batch_size': 32},
            {'window_size': 90, 'overlap_rate': 0.3, 'lstm_units': 64, 'learning_rate': 0.001, 'batch_size': 32},
            {'window_size': 90, 'overlap_rate': 0.7, 'lstm_units': 64, 'learning_rate': 0.001, 'batch_size': 32},
            {'window_size': 90, 'overlap_rate': 0.5, 'lstm_units': 128, 'learning_rate': 0.001, 'batch_size': 32},
            {'window_size': 90, 'overlap_rate': 0.5, 'lstm_units': 64, 'learning_rate': 0.0005, 'batch_size': 32},
            {'window_size': 90, 'overlap_rate': 0.5, 'lstm_units': 64, 'learning_rate': 0.001, 'batch_size': 16},
        ]
        
        best_f1 = 0
        best_params = None
        optimization_results = []
        
        for i, params in enumerate(param_combinations[:max_trials]):
            print(f"\nTrial {i+1}/{max_trials}: {params}")
            
            # Create sequences with current parameters
            sequences, seq_labels, seq_participants = self.create_sequences(
                features, labels, self.participant_ids, 
                params['window_size'], params['overlap_rate']
            )
            
            if len(sequences) < 100:  # Skip if too few sequences
                print(f"Skipping - too few sequences: {len(sequences)}")
                continue
            
            # Quick evaluation with one fold
            f1_score = self._quick_evaluate(sequences, seq_labels, seq_participants, params)
            
            optimization_results.append({
                'params': params,
                'f1_score': f1_score
            })
            
            if f1_score > best_f1:
                best_f1 = f1_score
                best_params = params
                
            print(f"F1 Score: {f1_score:.4f}")
        
        # Save optimization results
        with open('temporal_optimization_results.json', 'w') as f:
            json.dump(optimization_results, f, indent=2)
        
        print(f"\nBest parameters found: {best_params}")
        print(f"Best F1 Score: {best_f1:.4f}")
        
        self.best_params.update(best_params)
        return best_params
    
    def _quick_evaluate(self, sequences, labels, participants, params):
        """Quick evaluation using one fold"""
        # Use participant 1 as test set
        test_participant = 1
        train_mask = participants != test_participant
        test_mask = participants == test_participant
        
        if np.sum(test_mask) < 10:  # Skip if too few test samples
            return 0.0
        
        X_train = sequences[train_mask]
        y_train = labels[train_mask]
        X_test = sequences[test_mask]
        y_test = labels[test_mask]
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)
        
        # Normalize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1]))
        X_train_scaled = X_train_scaled.reshape(X_train.shape)
        
        X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1]))
        X_test_scaled = X_test_scaled.reshape(X_test.shape)
        
        # Build model
        model = Sequential([
            LSTM(params['lstm_units'], return_sequences=False, 
                 input_shape=(params['window_size'], 18)),
            Dropout(0.3),
            Dense(len(label_encoder.classes_), activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=params['learning_rate']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train with reduced epochs for quick evaluation
        model.fit(
            X_train_scaled, y_train_encoded,
            validation_split=0.2,
            epochs=10,
            batch_size=params['batch_size'],
            verbose=0
        )
        
        # Evaluate
        predictions = model.predict(X_test_scaled, verbose=0)
        predicted_labels = np.argmax(predictions, axis=1)
        
        f1 = f1_score(y_test_encoded, predicted_labels, average='weighted')
        return f1
    
    def train_optimized_model(self):
        """Train the optimized LSTM model"""
        print("\nTraining optimized LSTM model...")
        
        # Optimize parameters first
        best_params = self.optimize_temporal_parameters()
        
        # Extract features
        features, labels = self.extract_enhanced_features()
        
        # Create sequences with optimized parameters
        sequences, seq_labels, seq_participants = self.create_sequences(
            features, labels, self.participant_ids,
            best_params['window_size'], best_params['overlap_rate']
        )
        
        print(f"Created {len(sequences)} sequences with optimized parameters")
        
        # Full LOSO CV with optimized parameters
        return self._full_loso_evaluation(sequences, seq_labels, seq_participants, best_params)
    
    def _full_loso_evaluation(self, sequences, labels, participants, params):
        """Full Leave-One-Subject-Out evaluation"""
        print("Running full LOSO cross-validation...")
        
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        
        fold_results = []
        all_predictions = []
        all_true_labels = []
        
        for test_participant in np.unique(participants):
            print(f"\nFold: Testing on participant {test_participant}")
            
            train_mask = participants != test_participant
            test_mask = participants == test_participant
            
            X_train = sequences[train_mask]
            y_train = encoded_labels[train_mask]
            X_test = sequences[test_mask]
            y_test = encoded_labels[test_mask]
            
            # Normalize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1]))
            X_train_scaled = X_train_scaled.reshape(X_train.shape)
            
            X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1]))
            X_test_scaled = X_test_scaled.reshape(X_test.shape)
            
            # Calculate class weights
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(y_train),
                y=y_train
            )
            class_weight_dict = dict(enumerate(class_weights))
            
            # Build enhanced model
            model = Sequential([
                LSTM(params['lstm_units'], return_sequences=True, 
                     input_shape=(params['window_size'], 18)),
                Dropout(0.3),
                LSTM(params['lstm_units']//2, return_sequences=False),
                Dropout(0.3),
                Dense(64, activation='relu'),
                BatchNormalization(),
                Dropout(0.3),
                Dense(len(label_encoder.classes_), activation='softmax')
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=params['learning_rate']),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Train with enhanced callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)
            ]
            
            model.fit(
                X_train_scaled, y_train,
                validation_split=0.2,
                epochs=30,
                batch_size=params['batch_size'],
                class_weight=class_weight_dict,
                callbacks=callbacks,
                verbose=0
            )
            
            # Evaluate
            predictions = model.predict(X_test_scaled, verbose=0)
            predicted_labels = np.argmax(predictions, axis=1)
            
            accuracy = np.mean(predicted_labels == y_test)
            f1 = f1_score(y_test, predicted_labels, average='weighted')
            
            fold_results.append({
                'participant': test_participant,
                'accuracy': accuracy,
                'f1_score': f1,
                'test_samples': len(X_test)
            })
            
            all_predictions.extend(predicted_labels)
            all_true_labels.extend(y_test)
            
            print(f"Participant {test_participant}: Accuracy = {accuracy:.4f}, F1 = {f1:.4f}")
        
        # Calculate final results
        overall_accuracy = np.mean([r['accuracy'] for r in fold_results])
        overall_f1 = f1_score(all_true_labels, all_predictions, average='weighted')
        
        print(f"\n{'='*70}")
        print(f"OPTIMIZED LSTM RESULTS")
        print(f"{'='*70}")
        print(f"Overall Accuracy: {overall_accuracy:.4f}")
        print(f"Overall F1 Score: {overall_f1:.4f}")
        
        print(f"\nOptimized Parameters Used:")
        for param, value in params.items():
            print(f"  {param}: {value}")
        
        print(f"\nDetailed Classification Report:")
        print(classification_report(all_true_labels, all_predictions, 
                                  target_names=label_encoder.classes_))
        
        # Create results directory if it doesn't exist
        import os
        os.makedirs('results/metrics', exist_ok=True)
        
        # Save results
        results = {
            'overall_accuracy': overall_accuracy,
            'overall_f1_score': overall_f1,
            'fold_results': fold_results,
            'optimized_parameters': params
        }
        
        with open('results/metrics/optimized_lstm_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: results/metrics/optimized_lstm_results.json")
        
        return fold_results, overall_accuracy, overall_f1, label_encoder, scaler

    def save_best_model(self, label_encoder, scaler, fold_results):
        """Save the best performing model"""
        # Find the best fold based on F1 score
        best_fold = max(fold_results, key=lambda x: x['f1_score'])
        print(f"\nBest performing fold: Participant {best_fold['participant']} (F1: {best_fold['f1_score']:.4f})")
        
        # Create models directory
        import os
        os.makedirs('models', exist_ok=True)
        
        # Save preprocessing objects
        import joblib
        joblib.dump(label_encoder, 'models/optimized_lstm_label_encoder.pkl')
        joblib.dump(scaler, 'models/optimized_lstm_scaler.pkl')
        
        print(f"✓ Label encoder saved to: models/optimized_lstm_label_encoder.pkl")
        print(f"✓ Scaler saved to: models/optimized_lstm_scaler.pkl")

    def train_final_optimized_model_and_predict(self, label_encoder, scaler, best_params):
        """Train a final optimized model on all data and generate test predictions"""
        print("\n=== TRAINING FINAL OPTIMIZED MODEL AND GENERATING TEST PREDICTIONS ===")
        
        # Extract enhanced features from all training data
        features, labels = self.extract_enhanced_features()
        
        # Create sequences from all data with optimized parameters
        sequences, seq_labels, seq_participants = self.create_sequences(
            features, labels, self.participant_ids,
            best_params['window_size'], best_params['overlap_rate']
        )
        
        # Encode labels
        encoded_labels = label_encoder.transform(seq_labels)
        
        # Scale features
        X_scaled = scaler.transform(sequences.reshape(-1, sequences.shape[-1])).reshape(sequences.shape)
        
        # Build and train final optimized model on all data
        final_model = Sequential([
            LSTM(best_params['lstm_units'], return_sequences=True, 
                 input_shape=(best_params['window_size'], 18)),
            Dropout(0.3),
            LSTM(best_params['lstm_units']//2, return_sequences=False),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(len(label_encoder.classes_), activation='softmax')
        ])
        
        final_model.compile(
            optimizer=Adam(learning_rate=best_params['learning_rate']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train final model
        print("Training final optimized model on all data...")
        final_model.fit(
            X_scaled, encoded_labels,
            epochs=30,
            batch_size=best_params['batch_size'],
            verbose=0
        )
        
        # Save final model
        import os
        os.makedirs('models', exist_ok=True)
        final_model.save('models/final_optimized_lstm_model.h5')
        print(f"✓ Final optimized model saved to: models/final_optimized_lstm_model.h5")
        
        # Load test data
        test_file = 'test data_keypoint.csv'
        if not Path(test_file).exists():
            print(f"❌ Test data file '{test_file}' not found!")
            return None
        
        test_df = pd.read_csv(test_file)
        print(f"✓ Loaded test data: {len(test_df)} frames")
        
        # Prepare test features
        keypoint_cols = [col for col in test_df.columns if col.endswith('_x') or col.endswith('_y')]
        test_features = self._calculate_enhanced_pose_features(test_df[keypoint_cols])
        
        # Create test sequences
        test_sequences = []
        for i in range(0, len(test_features) - best_params['window_size'] + 1, best_params['window_size']):
            sequence = test_features[i:i + best_params['window_size']]
            test_sequences.append(sequence)
        
        if len(test_sequences) == 0:
            print("❌ No valid sequences created from test data")
            return None
        
        X_test = np.array(test_sequences)
        
        # Scale test features
        X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        
        # Make predictions
        predictions = final_model.predict(X_test_scaled, verbose=0)
        predicted_labels = np.argmax(predictions, axis=1)
        predicted_classes = label_encoder.inverse_transform(predicted_labels)
        confidence_scores = np.max(predictions, axis=1)
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'frame_id': range(len(predicted_classes)),
            'predicted_action': predicted_classes,
            'confidence': confidence_scores
        })
        
        # Save predictions
        os.makedirs('results/predictions', exist_ok=True)
        results_df.to_csv('results/predictions/test_predictions_optimized_lstm.csv', index=False)
        
        print(f"✓ Test predictions saved to: results/predictions/test_predictions_optimized_lstm.csv")
        print(f"✓ Predicted {len(predicted_classes)} frames")
        
        # Show prediction distribution
        pred_counts = pd.Series(predicted_classes).value_counts()
        print(f"\nPrediction Distribution:")
        for action, count in pred_counts.items():
            print(f"  {action}: {count} frames ({count/len(predicted_classes)*100:.1f}%)")
        
        return results_df

def main():
    """Main execution"""
    print("="*80)
    print("ISAS CHALLENGE: OPTIMIZED LSTM WITH TEMPORAL PARAMETER OPTIMIZATION")
    print("Following ABC Paper Methodology")
    print("="*80)
    
    # Initialize optimized model
    model = OptimizedISASLSTM()
    
    # Load data
    model.load_and_preprocess_data()
    
    # Train optimized model
    fold_results, accuracy, f1, label_encoder, scaler = model.train_optimized_model()
    
    print(f"\nFinal Optimized Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Save best model
    if fold_results:
        best_fold = max(fold_results, key=lambda x: x['f1_score'])
        print(f"\nSaving best optimized model from participant {best_fold['participant']}...")
        model.save_best_model(label_encoder, scaler, fold_results)
    
    # Generate test predictions with optimized model
    print(f"\nGenerating predictions for test data with optimized model...")
    best_params = model.best_params
    test_predictions = model.train_final_optimized_model_and_predict(label_encoder, scaler, best_params)
    
    # Compare with ABC paper and previous results
    print(f"\nComparison:")
    print(f"ABC Paper F1: 96-100%")
    print(f"Previous LSTM F1: 46.6%")
    print(f"Optimized LSTM F1: {f1*100:.1f}%")
    
    if f1 >= 0.80:
        print("✓ Significant improvement achieved!")
    elif f1 >= 0.70:
        print("✓ Good improvement, approaching ABC paper results")
    else:
        print("⚠ Further optimization needed")

if __name__ == "__main__":
    main()
