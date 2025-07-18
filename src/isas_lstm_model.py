#!/usr/bin/env python3
"""
ISAS Challenge: LSTM-based Abnormal Activity Recognition
Following ABC Paper Methodology with Feature Engineering and LOSO Cross-Validation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML and Deep Learning
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

class ISASLSTMModel:
    """
    LSTM model for abnormal activity recognition following ABC paper methodology
    """
    
    def __init__(self):
        self.data_dir = Path("Train Data")
        self.participants = [1, 2, 3, 5]
        
        # ABC paper hyperparameters
        self.lstm_units = 64
        self.epochs = 20
        self.batch_size = 32
        self.window_size = 30
        self.overlap_rate = 0.5
        
        # Data containers
        self.combined_data = None
        self.features = None
        self.labels = None
        self.participant_ids = None
        
        print("ISAS LSTM Model initialized with ABC paper configuration")
    
    def load_and_preprocess_data(self):
        """Load and preprocess the dataset"""
        print("Loading and preprocessing data...")
        
        all_data = []
        
        for participant in self.participants:
            # Load labeled data
            file_path = self.data_dir / "keypointlabel" / f"keypoints_with_labels_{participant}.csv"
            data = pd.read_csv(file_path)
            data['participant'] = participant
            all_data.append(data)
            
            print(f"Participant {participant}: {len(data)} frames")
        
        # Combine all data
        self.combined_data = pd.concat(all_data, ignore_index=True)
        
        # Remove rows with missing labels
        original_length = len(self.combined_data)
        self.combined_data = self.combined_data.dropna(subset=['Action Label'])
        
        print(f"Total frames: {original_length}")
        print(f"After removing missing labels: {len(self.combined_data)}")
        
        # Handle the 'Throwing' vs 'Throwing things' issue
        self.combined_data['Action Label'] = self.combined_data['Action Label'].replace('Throwing', 'Throwing things')
        
        # Class distribution
        print("\nClass distribution:")
        class_counts = self.combined_data['Action Label'].value_counts()
        for activity, count in class_counts.items():
            print(f"  {activity}: {count} frames ({count/len(self.combined_data)*100:.1f}%)")
        
        return self.combined_data
    
    def extract_features(self):
        """Extract 14 features from keypoint data as per ABC paper"""
        print("\nExtracting features...")
        
        # Get keypoint columns
        keypoint_cols = [col for col in self.combined_data.columns if col.endswith('_x') or col.endswith('_y')]
        
        # Feature lists
        all_features = []
        all_labels = []
        all_participants = []
        
        # Process each participant separately to maintain temporal order
        for participant in self.participants:
            participant_data = self.combined_data[self.combined_data['participant'] == participant].copy()
            participant_data = participant_data.sort_values('frame_id').reset_index(drop=True)
            
            # Extract keypoint coordinates
            features = self._calculate_pose_features(participant_data[keypoint_cols])
            labels = participant_data['Action Label'].values
            
            all_features.extend(features)
            all_labels.extend(labels)
            all_participants.extend([participant] * len(features))
        
        self.features = np.array(all_features)
        self.labels = np.array(all_labels)
        self.participant_ids = np.array(all_participants)
        
        print(f"Extracted features shape: {self.features.shape}")
        print(f"Number of samples: {len(self.labels)}")
        
        return self.features, self.labels
    
    def _calculate_pose_features(self, keypoint_data):
        """Calculate 14 features from pose keypoints"""
        features = []
        
        # Get coordinates for key body parts
        coords = {}
        body_parts = ['right_wrist', 'left_wrist', 'right_ankle', 'left_ankle', 
                     'right_shoulder', 'left_shoulder', 'right_eye', 'left_eye']
        
        for part in body_parts:
            coords[part] = {
                'x': keypoint_data[f'{part}_x'].values,
                'y': keypoint_data[f'{part}_y'].values
            }
        
        n_frames = len(keypoint_data)
        
        for i in range(n_frames):
            frame_features = []
            
            # 1-4: Hand and foot speeds
            if i > 0:
                # Right hand speed
                right_hand_speed = np.sqrt(
                    (coords['right_wrist']['x'][i] - coords['right_wrist']['x'][i-1])**2 +
                    (coords['right_wrist']['y'][i] - coords['right_wrist']['y'][i-1])**2
                )
                
                # Left hand speed
                left_hand_speed = np.sqrt(
                    (coords['left_wrist']['x'][i] - coords['left_wrist']['x'][i-1])**2 +
                    (coords['left_wrist']['y'][i] - coords['left_wrist']['y'][i-1])**2
                )
                
                # Right foot speed
                right_foot_speed = np.sqrt(
                    (coords['right_ankle']['x'][i] - coords['right_ankle']['x'][i-1])**2 +
                    (coords['right_ankle']['y'][i] - coords['right_ankle']['y'][i-1])**2
                )
                
                # Left foot speed
                left_foot_speed = np.sqrt(
                    (coords['left_ankle']['x'][i] - coords['left_ankle']['x'][i-1])**2 +
                    (coords['left_ankle']['y'][i] - coords['left_ankle']['y'][i-1])**2
                )
            else:
                right_hand_speed = left_hand_speed = right_foot_speed = left_foot_speed = 0
            
            # 5-8: Hand and foot accelerations
            if i > 1:
                # Previous speeds
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
            
            # 9-10: Shoulder-wrist angles
            right_shoulder_wrist_angle = np.arctan2(
                coords['right_wrist']['y'][i] - coords['right_shoulder']['y'][i],
                coords['right_wrist']['x'][i] - coords['right_shoulder']['x'][i]
            ) * 180 / np.pi
            
            left_shoulder_wrist_angle = np.arctan2(
                coords['left_wrist']['y'][i] - coords['left_shoulder']['y'][i],
                coords['left_wrist']['x'][i] - coords['left_shoulder']['x'][i]
            ) * 180 / np.pi
            
            # 11-14: Eye displacements
            if i > 0:
                right_eye_vertical_displacement = coords['right_eye']['y'][i] - coords['right_eye']['y'][i-1]
                left_eye_vertical_displacement = coords['left_eye']['y'][i] - coords['left_eye']['y'][i-1]
                right_eye_horizontal_displacement = coords['right_eye']['x'][i] - coords['right_eye']['x'][i-1]
                left_eye_horizontal_displacement = coords['left_eye']['x'][i] - coords['left_eye']['x'][i-1]
            else:
                right_eye_vertical_displacement = left_eye_vertical_displacement = 0
                right_eye_horizontal_displacement = left_eye_horizontal_displacement = 0
            
            # Combine all 14 features
            frame_features = [
                right_hand_speed, left_hand_speed, right_foot_speed, left_foot_speed,
                right_hand_acceleration, left_hand_acceleration, right_foot_acceleration, left_foot_acceleration,
                right_shoulder_wrist_angle, left_shoulder_wrist_angle,
                right_eye_vertical_displacement, left_eye_vertical_displacement,
                right_eye_horizontal_displacement, left_eye_horizontal_displacement
            ]
            
            features.append(frame_features)
        
        return features
    
    def create_sequences(self, features, labels, participant_ids):
        """Create sequences for LSTM training"""
        print(f"Creating sequences with window size: {self.window_size}")
        
        sequences = []
        sequence_labels = []
        sequence_participants = []
        
        step_size = max(1, int(self.window_size * (1 - self.overlap_rate)))
        
        # Process each participant separately
        for participant in self.participants:
            participant_mask = participant_ids == participant
            participant_features = features[participant_mask]
            participant_labels = labels[participant_mask]
            
            # Create sequences for this participant
            for i in range(0, len(participant_features) - self.window_size + 1, step_size):
                sequence = participant_features[i:i + self.window_size]
                # Use the label at the end of the sequence
                label = participant_labels[i + self.window_size - 1]
                
                sequences.append(sequence)
                sequence_labels.append(label)
                sequence_participants.append(participant)
        
        return np.array(sequences), np.array(sequence_labels), np.array(sequence_participants)
    
    def train_with_loso_cv(self):
        """Train LSTM with Leave-One-Subject-Out cross-validation"""
        print("\nTraining LSTM with Leave-One-Subject-Out cross-validation...")
        
        # Extract features
        features, labels = self.extract_features()
        
        # Create sequences
        sequences, seq_labels, seq_participants = self.create_sequences(features, labels, self.participant_ids)
        
        print(f"Created {len(sequences)} sequences")
        print(f"Sequence shape: {sequences.shape}")
        
        # Label encoding
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(seq_labels)
        
        # Store results
        fold_results = []
        all_predictions = []
        all_true_labels = []
        
        # LOSO CV
        for test_participant in self.participants:
            print(f"\nFold: Testing on participant {test_participant}")
            
            # Split data
            train_mask = seq_participants != test_participant
            test_mask = seq_participants == test_participant
            
            X_train = sequences[train_mask]
            y_train = encoded_labels[train_mask]
            X_test = sequences[test_mask]
            y_test = encoded_labels[test_mask]
            
            if len(X_train) == 0 or len(X_test) == 0:
                print(f"Skipping participant {test_participant} - insufficient data")
                continue
            
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
            
            # Build LSTM model
            model = Sequential([
                LSTM(self.lstm_units, return_sequences=False, 
                     input_shape=(self.window_size, 14)),
                Dropout(0.2),
                Dense(len(label_encoder.classes_), activation='softmax')
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Train
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            
            history = model.fit(
                X_train_scaled, y_train,
                validation_split=0.2,
                epochs=self.epochs,
                batch_size=self.batch_size,
                class_weight=class_weight_dict,
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Evaluate
            predictions = model.predict(X_test_scaled, verbose=0)
            predicted_labels = np.argmax(predictions, axis=1)
            
            # Calculate metrics
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
        
        # Overall results
        overall_accuracy = np.mean([r['accuracy'] for r in fold_results])
        overall_f1 = f1_score(all_true_labels, all_predictions, average='weighted')
        
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Overall Accuracy: {overall_accuracy:.4f}")
        print(f"Overall F1 Score: {overall_f1:.4f}")
        
        # Detailed results
        print(f"\nDetailed Classification Report:")
        print(classification_report(all_true_labels, all_predictions, 
                                  target_names=label_encoder.classes_))
        
        # Confusion matrix
        cm = confusion_matrix(all_true_labels, all_predictions)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=label_encoder.classes_,
                   yticklabels=label_encoder.classes_)
        plt.title('Confusion Matrix - LSTM Model (LOSO CV)')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('lstm_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save results
        results_summary = {
            'overall_accuracy': overall_accuracy,
            'overall_f1_score': overall_f1,
            'fold_results': fold_results,
            'hyperparameters': {
                'lstm_units': self.lstm_units,
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'window_size': self.window_size,
                'overlap_rate': self.overlap_rate
            }
        }
        
        # Create results directory if it doesn't exist
        import os
        os.makedirs('results/metrics', exist_ok=True)
        os.makedirs('results/visualizations', exist_ok=True)
        
        with open('results/metrics/lstm_results.json', 'w') as f:
            import json
            json.dump(results_summary, f, indent=2)
        
        plt.savefig('results/visualizations/lstm_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nResults saved to: results/metrics/lstm_results.json")
        print(f"Confusion matrix saved to: results/visualizations/lstm_confusion_matrix.png")
        
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
        joblib.dump(label_encoder, 'models/lstm_label_encoder.pkl')
        joblib.dump(scaler, 'models/lstm_scaler.pkl')
        
        print(f"✓ Label encoder saved to: models/lstm_label_encoder.pkl")
        print(f"✓ Scaler saved to: models/lstm_scaler.pkl")

    def train_final_model_and_predict(self, label_encoder, scaler):
        """Train a final model on all data and generate test predictions"""
        print("\n=== TRAINING FINAL MODEL AND GENERATING TEST PREDICTIONS ===")
        
        # Extract features from all training data
        features, labels = self.extract_features()
        
        # Create sequences from all data
        sequences, seq_labels, seq_participants = self.create_sequences(features, labels, self.participant_ids)
        
        # Encode labels
        encoded_labels = label_encoder.transform(seq_labels)
        
        # Scale features
        X_scaled = scaler.transform(sequences.reshape(-1, sequences.shape[-1])).reshape(sequences.shape)
        
        # Build and train final model on all data
        final_model = Sequential([
            LSTM(self.lstm_units, return_sequences=False, 
                 input_shape=(self.window_size, 14)),
            Dropout(0.2),
            Dense(len(label_encoder.classes_), activation='softmax')
        ])
        
        final_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train final model
        print("Training final model on all data...")
        final_model.fit(
            X_scaled, encoded_labels,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0
        )
        
        # Save final model
        import os
        os.makedirs('models', exist_ok=True)
        final_model.save('models/final_lstm_model.h5')
        print(f"✓ Final model saved to: models/final_lstm_model.h5")
        
        # Load test data
        test_file = 'test data_keypoint.csv'
        if not Path(test_file).exists():
            print(f"❌ Test data file '{test_file}' not found!")
            return None
        
        test_df = pd.read_csv(test_file)
        print(f"✓ Loaded test data: {len(test_df)} frames")
        
        # Prepare test features
        keypoint_cols = [col for col in test_df.columns if col.endswith('_x') or col.endswith('_y')]
        test_features = self._calculate_pose_features(test_df[keypoint_cols])
        
        # Create test sequences
        test_sequences = []
        for i in range(0, len(test_features) - self.window_size + 1, self.window_size):
            sequence = test_features[i:i + self.window_size]
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
        results_df.to_csv('results/predictions/test_predictions_lstm.csv', index=False)
        
        print(f"✓ Test predictions saved to: results/predictions/test_predictions_lstm.csv")
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
    print("ISAS CHALLENGE: LSTM-BASED ABNORMAL ACTIVITY RECOGNITION")
    print("Following ABC Paper Methodology")
    print("="*80)
    
    # Initialize model
    model = ISASLSTMModel()
    
    # Load data
    model.load_and_preprocess_data()
    
    # Train with LOSO CV
    fold_results, accuracy, f1, label_encoder, scaler = model.train_with_loso_cv()
    
    print(f"\nFinal Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Save best model
    if fold_results:
        # Get the best model from the best performing fold
        best_fold = max(fold_results, key=lambda x: x['f1_score'])
        print(f"\nSaving best model from participant {best_fold['participant']}...")
        model.save_best_model(label_encoder, scaler, fold_results)
    
    # Generate test predictions
    print(f"\nGenerating predictions for test data...")
    test_predictions = model.train_final_model_and_predict(label_encoder, scaler)
    
    # Compare with ABC paper results
    print(f"\nComparison with ABC Paper:")
    print(f"ABC Paper F1: 96-100%")
    print(f"Our F1: {f1*100:.1f}%")
    
    if f1 >= 0.96:
        print("✓ Performance matches ABC paper expectations!")
    else:
        print("⚠ Performance below ABC paper - consider temporal parameter optimization")

if __name__ == "__main__":
    main()
