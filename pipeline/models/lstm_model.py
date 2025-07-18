#!/usr/bin/env python3
"""
LSTM Model Implementation
Handles sequence creation and LSTM training for activity recognition
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib

class ISASLSTMModel:
    """
    LSTM model for abnormal activity recognition
    """
    
    def __init__(self, window_size=30, overlap_rate=0.5, lstm_units=64, 
                 learning_rate=0.001, batch_size=32, epochs=20):
        self.window_size = window_size
        self.overlap_rate = overlap_rate
        self.lstm_units = lstm_units
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        print(f"LSTM Model initialized with window_size={window_size}, units={lstm_units}")
    
    def create_sequences(self, features, labels, participant_ids):
        """Create sequences for LSTM training"""
        print(f"Creating sequences with window_size={self.window_size}, overlap_rate={self.overlap_rate}")
        
        sequences = []
        sequence_labels = []
        sequence_participants = []
        
        step_size = max(1, int(self.window_size * (1 - self.overlap_rate)))
        
        # Process each participant separately to maintain temporal order
        for participant in np.unique(participant_ids):
            participant_mask = participant_ids == participant
            participant_features = features[participant_mask]
            participant_labels = labels[participant_mask]
            
            # Create sequences for this participant
            for i in range(0, len(participant_features) - self.window_size + 1, step_size):
                sequence = participant_features[i:i + self.window_size]
                sequence_label = self._get_majority_label(participant_labels[i:i + self.window_size])
                
                sequences.append(sequence)
                sequence_labels.append(sequence_label)
                sequence_participants.append(participant)
        
        sequences = np.array(sequences)
        sequence_labels = np.array(sequence_labels)
        sequence_participants = np.array(sequence_participants)
        
        print(f"Created {len(sequences)} sequences")
        print(f"Sequence shape: {sequences.shape}")
        
        return sequences, sequence_labels, sequence_participants
    
    def _get_majority_label(self, labels):
        """Get majority label in a sequence"""
        unique_labels, counts = np.unique(labels, return_counts=True)
        return unique_labels[np.argmax(counts)]
    
    def build_model(self, input_shape, num_classes):
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(self.lstm_units, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            LSTM(self.lstm_units // 2, return_sequences=False),
            Dropout(0.3),
            BatchNormalization(),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_with_loso_cv(self, sequences, labels, participants):
        """Train model using Leave-One-Subject-Out cross-validation"""
        print("\nStarting LOSO Cross-Validation...")
        
        unique_participants = np.unique(participants)
        fold_results = []
        
        # Encode labels
        all_labels_encoded = self.label_encoder.fit_transform(labels)
        
        for test_participant in unique_participants:
            print(f"\n--- Fold: Test Participant {test_participant} ---")
            
            # Split data
            train_mask = participants != test_participant
            test_mask = participants == test_participant
            
            X_train, X_test = sequences[train_mask], sequences[test_mask]
            y_train, y_test = all_labels_encoded[train_mask], all_labels_encoded[test_mask]
            
            print(f"Train sequences: {len(X_train)}, Test sequences: {len(X_test)}")
            
            # Scale features
            X_train_scaled = self._scale_sequences(X_train, fit=True)
            X_test_scaled = self._scale_sequences(X_test, fit=False)
            
            # Build and train model
            model = self.build_model(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                num_classes=len(np.unique(all_labels_encoded))
            )
            
            # Calculate class weights
            class_weights = self._calculate_class_weights(y_train)
            
            # Callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
            ]
            
            # Train model
            history = model.fit(
                X_train_scaled, y_train,
                validation_data=(X_test_scaled, y_test),
                epochs=self.epochs,
                batch_size=self.batch_size,
                class_weight=class_weights,
                callbacks=callbacks,
                verbose=0
            )
            
            # Evaluate
            y_pred = model.predict(X_test_scaled, verbose=0)
            y_pred_classes = np.argmax(y_pred, axis=1)
            
            # Calculate metrics
            accuracy = np.mean(y_pred_classes == y_test)
            f1 = f1_score(y_test, y_pred_classes, average='weighted')
            
            fold_result = {
                'participant': test_participant,
                'accuracy': accuracy,
                'f1_score': f1,
                'y_true': y_test,
                'y_pred': y_pred_classes,
                'model': model,
                'history': history.history
            }
            
            fold_results.append(fold_result)
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1 Score: {f1:.4f}")
        
        # Calculate overall metrics
        all_y_true = np.concatenate([fold['y_true'] for fold in fold_results])
        all_y_pred = np.concatenate([fold['y_pred'] for fold in fold_results])
        
        overall_accuracy = np.mean(all_y_true == all_y_pred)
        overall_f1 = f1_score(all_y_true, all_y_pred, average='weighted')
        
        print(f"\n=== LOSO Cross-Validation Results ===")
        print(f"Overall Accuracy: {overall_accuracy:.4f}")
        print(f"Overall F1 Score: {overall_f1:.4f}")
        
        return fold_results, overall_accuracy, overall_f1
    
    def _scale_sequences(self, sequences, fit=False):
        """Scale sequence features"""
        original_shape = sequences.shape
        sequences_reshaped = sequences.reshape(-1, sequences.shape[-1])
        
        if fit:
            sequences_scaled = self.scaler.fit_transform(sequences_reshaped)
        else:
            sequences_scaled = self.scaler.transform(sequences_reshaped)
        
        return sequences_scaled.reshape(original_shape)
    
    def _calculate_class_weights(self, y):
        """Calculate class weights for imbalanced data"""
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        return dict(zip(classes, weights))
    
    def train_final_model(self, sequences, labels, participants):
        """Train final model on all data"""
        print("\nTraining final model on all data...")
        
        # Encode labels
        labels_encoded = self.label_encoder.fit_transform(labels)
        
        # Scale features
        X_scaled = self._scale_sequences(sequences, fit=True)
        
        # Build model
        self.model = self.build_model(
            input_shape=(sequences.shape[1], sequences.shape[2]),
            num_classes=len(np.unique(labels_encoded))
        )
        
        # Calculate class weights
        class_weights = self._calculate_class_weights(labels_encoded)
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-6)
        ]
        
        # Train
        history = self.model.fit(
            X_scaled, labels_encoded,
            epochs=self.epochs,
            batch_size=self.batch_size,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        print("Final model training completed!")
        return history
    
    def predict_test_data(self, test_features):
        """Make predictions on test data"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_final_model() first.")
        
        print(f"\nPredicting on test data with {len(test_features)} frames...")
        
        # Create sequences from test data
        test_sequences = []
        frame_indices = []
        
        step_size = max(1, int(self.window_size * (1 - self.overlap_rate)))
        
        for i in range(0, len(test_features) - self.window_size + 1, step_size):
            sequence = test_features[i:i + self.window_size]
            test_sequences.append(sequence)
            frame_indices.append(i + self.window_size // 2)  # Middle frame of sequence
        
        test_sequences = np.array(test_sequences)
        
        # Scale sequences
        test_sequences_scaled = self._scale_sequences(test_sequences, fit=False)
        
        # Predict
        predictions = self.model.predict(test_sequences_scaled, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        predicted_labels = self.label_encoder.inverse_transform(predicted_classes)
        
        print(f"Generated {len(predicted_labels)} predictions")
        
        return predicted_labels, frame_indices, predictions
    
    def save_model(self, filepath_prefix):
        """Save model and preprocessing objects"""
        if self.model is None:
            raise ValueError("Model not trained.")
        
        model_path = f"{filepath_prefix}_model.h5"
        scaler_path = f"{filepath_prefix}_scaler.joblib"
        encoder_path = f"{filepath_prefix}_label_encoder.joblib"
        
        # Save model
        self.model.save(model_path)
        
        # Save preprocessing objects
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.label_encoder, encoder_path)
        
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
        print(f"Label encoder saved to {encoder_path}")
        
        return model_path, scaler_path, encoder_path 