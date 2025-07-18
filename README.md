# ISAS Challenge: Comprehensive Abnormal Activity Recognition Pipeline

A complete machine learning pipeline for abnormal activity recognition using skeleton keypoints, following the ABC paper methodology with LSTM models and Leave-One-Subject-Out (LOSO) cross-validation.

## 🎯 Project Overview

This pipeline implements a comprehensive solution for the ISAS Challenge, featuring:

- **ABC Paper Methodology**: 14 pose features extracted from skeleton keypoints
- **LSTM Models**: Both baseline and optimized architectures
- **LOSO Cross-Validation**: Rigorous evaluation with participant-independent testing
- **Comprehensive Evaluation**: Detailed metrics, confusion matrices, and visualizations
- **Video Generation**: HD skeleton animation with real-time predictions
- **Multiple Export Formats**: CSV, JSON, and summary reports

## 📁 Project Structure

```
new approach 1/
├── pipeline/                     # Modular pipeline components
│   ├── data/                     # Data loading and processing
│   │   ├── data_loader.py        # Training/test data loader
│   │   └── feature_extractor.py  # ABC paper feature extraction
│   ├── models/                   # Model implementations
│   │   └── lstm_model.py         # LSTM with LOSO cross-validation
│   ├── evaluation/               # Evaluation and metrics
│   │   └── metrics.py            # Comprehensive evaluation suite
│   ├── visualization/            # Video and plot generation
│   │   └── skeleton_animator.py  # HD skeleton animation
│   └── utils/                    # Utilities
│       └── prediction_exporter.py # Multi-format export
├── src/                          # Legacy individual scripts
├── Train Data/                   # Training dataset
│   ├── keypointlabel/           # Labeled keypoint data
│   └── keypoint/                # Raw keypoint data
├── test data_keypoint.csv       # Test dataset (117,921 frames)
├── results/                     # Generated outputs
│   ├── models/                  # Saved models and preprocessing
│   ├── metrics/                 # Evaluation results and plots
│   ├── predictions/             # Prediction files
│   └── videos/                  # Generated animation videos
├── run_complete_pipeline.py     # Main pipeline orchestrator
├── test_pipeline.py             # Pipeline validation script
└── requirements.txt             # Dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Validate Setup

```bash
python test_pipeline.py
```

### 3. Run Complete Pipeline

```bash
# Run full pipeline (baseline + optimized models + videos)
python run_complete_pipeline.py

# Run only baseline model
python run_complete_pipeline.py --baseline-only

# Run only optimized model  
python run_complete_pipeline.py --optimized-only

# Skip video generation (faster)
python run_complete_pipeline.py --no-video
```

## 📊 Pipeline Components

### Data Processing
- **KeypointDataLoader**: Loads and validates training/test data
- **ABCFeatureExtractor**: Extracts 14 ABC paper features with temporal smoothing

### Models
- **Baseline LSTM**: window_size=30, lstm_units=64, following ABC paper specs
- **Optimized LSTM**: window_size=90, lstm_units=128, enhanced parameters

### Evaluation
- **ModelEvaluator**: LOSO cross-validation with comprehensive metrics
- **Metrics**: Accuracy, F1 (macro/weighted/micro), precision, recall
- **Visualizations**: Confusion matrices, training curves, per-class analysis

### Outputs
- **Model Files**: Trained models with preprocessing objects
- **Predictions**: CSV/JSON exports with confidence scores
- **Videos**: HD skeleton animation with real-time predictions
- **Reports**: Detailed evaluation and comparison summaries

## 🎬 Generated Videos

The pipeline creates high-quality skeleton animation videos:

- **Individual Model Videos**: Show predictions in real-time (1280x720, 30fps)
- **Comparison Videos**: Side-by-side model comparison
- **Summary Videos**: Statistical overview of predictions

## 📈 Expected Results

Based on the ABC paper methodology and LOSO cross-validation:

- **Training Data**: ~304K frames from 4 participants
- **Test Data**: 117,921 frames
- **Classes**: 8 activities (Walking, Sitting quietly, Eating snacks, etc.)
- **Features**: 14 pose-based features per frame
- **Sequences**: Variable count based on window size and overlap
- **Coverage**: Full test dataset prediction with frame-by-frame mapping

## 🔧 Customization

### Model Parameters
Edit `run_complete_pipeline.py` to modify:
- Window size and overlap rate
- LSTM architecture (units, layers)
- Training parameters (epochs, batch size, learning rate)

### Feature Engineering
Modify `pipeline/data/feature_extractor.py` to:
- Add new pose features
- Adjust temporal smoothing
- Change body part selections

### Evaluation Metrics
Extend `pipeline/evaluation/metrics.py` for:
- Custom metrics
- Additional visualizations
- Different evaluation schemes

## 📝 Output Files

### Models
- `results/models/baseline_lstm_model.h5` - Trained baseline model
- `results/models/optimized_lstm_model.h5` - Trained optimized model
- `results/models/*_scaler.joblib` - Feature scalers
- `results/models/*_label_encoder.joblib` - Label encoders

### Predictions
- `results/predictions/*_predictions_*.csv` - Detailed predictions with metadata
- `results/predictions/*_frame_mapping_*.csv` - Frame-by-frame prediction mapping
- `results/predictions/*_summary_*.txt` - Prediction statistics

### Evaluation
- `results/metrics/*/confusion_matrix.png` - Confusion matrix heatmaps
- `results/metrics/*/classification_report.csv` - Per-class metrics
- `results/metrics/*/participant_performance.png` - LOSO CV results
- `results/metrics/*/evaluation_report.txt` - Comprehensive evaluation

### Videos
- `results/videos/baseline_skeleton_animation.mp4` - Baseline model video
- `results/videos/optimized_skeleton_animation.mp4` - Optimized model video
- `results/videos/model_comparison.mp4` - Side-by-side comparison

## 🏆 Key Features

✅ **ABC Paper Compliant**: Full implementation of methodology  
✅ **LOSO Cross-Validation**: Rigorous participant-independent evaluation  
✅ **Comprehensive Metrics**: Multi-level performance assessment  
✅ **Production Ready**: Complete pipeline with error handling  
✅ **Reproducible**: Deterministic results with saved models  
✅ **Extensible**: Modular design for easy customization  
✅ **Visual Analytics**: Rich visualizations and animations  
✅ **Multiple Formats**: CSV, JSON, video, and report outputs  

## 🐛 Troubleshooting

### Common Issues

1. **ImportError**: Run `python test_pipeline.py` to check dependencies
2. **Memory Issues**: Reduce batch size or window size in model parameters
3. **GPU Issues**: Pipeline automatically falls back to CPU if GPU unavailable
4. **File Not Found**: Ensure training data is in `Train Data/` directory
5. **Video Generation**: Install `opencv-python` if video creation fails

### Performance Optimization

- Use `--no-video` flag for faster execution
- Run single model with `--baseline-only` or `--optimized-only`
- Reduce sequence window size for lower memory usage
- Use GPU acceleration for faster training

## 📚 References

- ABC Paper: [Original methodology paper]
- ISAS Challenge: [Challenge specification]
- COCO Keypoints: [17-point skeleton format]
- LOSO Cross-Validation: [Leave-One-Subject-Out methodology]

## 📧 Support

For issues or questions:
1. Check `test_pipeline.py` output for setup validation
2. Review `results/PIPELINE_SUMMARY.txt` for execution details
3. Examine error logs for specific debugging information

---

**Total Pipeline Execution Time**: ~15-30 minutes (depending on hardware)  
**Generated Files**: ~50+ comprehensive outputs  
**Video Quality**: HD 1280x720 at 30fps  
**Prediction Coverage**: 100% test dataset with confidence scores
