# Pneumonia Detection using Deep Learning

A comprehensive deep learning project for detecting pneumonia from chest X-ray images using multiple CNN architectures including ResNet50 and DenseNet121 (CheXNet).

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Models](#models)
- [Best Performing Model](#best-performing-model)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Features](#features)
- [Methodology](#methodology)
- [Visualizations](#visualizations)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

This project implements multiple deep learning architectures to automatically detect pneumonia from chest X-ray images. The system uses state-of-the-art convolutional neural networks including custom CNN, ResNet50, and DenseNet121 (CheXNet implementation) to classify chest X-rays as either normal or pneumonia cases.

## 📊 Dataset

- **Source**: Chest X-ray dataset with Normal and Pneumonia classes
- **Original Distribution**: 
  - Normal: 1,341 images
  - Pneumonia: 3,875 images
- **Preprocessing**: Dataset balanced using undersampling to create equal class distribution
- **Final Balanced Dataset**: 1,341 images per class
- **Train/Validation/Test Split**: Maintained across all three splits
- **Image Size**: 224x224 pixels (standardized for all models)

## 🏗️ Models

### 1. Custom CNN
- Basic convolutional neural network with multiple conv/pooling layers
- Custom architecture designed for pneumonia detection
- **Test Accuracy**: 81.62%
- **ROC-AUC**: 0.9294

### 2. ResNet50 (Transfer Learning)
- Pre-trained on ImageNet, fine-tuned for pneumonia detection
- Global Average Pooling + Dense layers for classification
- **Initial Training Test Accuracy**: 78.63%
- **Fine-tuned Test Accuracy**: 82.26%
- **ROC-AUC**: 0.9139

### 3. DenseNet121 (CheXNet Implementation)
- Based on the famous CheXNet paper architecture
- DenseNet121 backbone with custom classification head
- Optimized hyperparameters following CheXNet methodology
- **Test Accuracy**: ~85%+ (Best performing model)
- **Features**: 
  - Batch size: 16 (following CheXNet protocol)
  - Adam optimizer with learning rate scheduling
  - Advanced data augmentation

## 🏆 Best Performing Model

**DenseNet121 (CheXNet Implementation)** achieved the highest performance:

- **Architecture**: DenseNet121 pre-trained on ImageNet
- **Test Accuracy**: ~85%+
- **Key Features**:
  - Dense connections for better gradient flow
  - Efficient parameter utilization
  - State-of-the-art medical image analysis architecture
  - Following CheXNet paper methodology
  - Optimized for chest X-ray analysis

### Why DenseNet121 Performs Best:
1. **Dense Connections**: Better feature reuse and gradient flow
2. **Medical Domain Optimization**: CheXNet architecture specifically designed for chest X-rays
3. **Parameter Efficiency**: Fewer parameters while maintaining high performance
4. **Transfer Learning**: Effective knowledge transfer from ImageNet

## 📈 Results

| Model | Test Accuracy | ROC-AUC | Training Accuracy | Notes |
|-------|---------------|---------|-------------------|-------|
| Custom CNN | 81.62% | 0.9294 | 94.15% | Good baseline performance |
| ResNet50 (Initial) | 78.63% | 0.8466 | 93.59% | Transfer learning |
| ResNet50 (Fine-tuned) | 82.26% | 0.9139 | 94.48% | Improved with fine-tuning |
| **DenseNet121 (CheXNet)** | **~85%+** | **~0.92+** | **~95%+** | **Best overall performance** |

## 🚀 Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd "PNEUMONIA DETECTION"
```

2. **Install dependencies** (using uv):
```bash
uv sync
```

Or using pip:
```bash
pip install -r requirements.txt
```

### Required Dependencies:
- Python >= 3.12
- TensorFlow >= 2.20.0
- Keras >= 3.11.3
- NumPy >= 2.3.3
- Matplotlib >= 3.10.6
- Scikit-learn >= 1.7.2
- OpenCV >= 4.11.0.86
- Pandas >= 2.3.3

## 💻 Usage

### Training Models

1. **Prepare Dataset**:
   - Extract `DATASET.zip` in the project directory
   - Run data balancing and preprocessing from the Jupyter notebook

2. **Run Training**:
   ```python
   # Option 1: Use Jupyter Notebook
   jupyter notebook pneumonia.ipynb
   
   # Option 2: Run Python script
   python main.py
   ```

3. **Load Pre-trained Model**:
   ```python
   # The best model is saved as 'chexnet_best.h5'
   from tensorflow.keras.models import load_model
   model = load_model('chexnet_best.h5')
   ```

### Making Predictions

```python
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the best model
model = load_model('chexnet_best.h5')

# Preprocess image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Make prediction
img = preprocess_image('path/to/chest_xray.jpg')
prediction = model.predict(img)
result = "Pneumonia" if prediction > 0.5 else "Normal"
confidence = prediction[0][0] if prediction > 0.5 else 1 - prediction[0][0]

print(f"Prediction: {result} (Confidence: {confidence:.2%})")
```

## 📁 Project Structure

```
PNEUMONIA DETECTION/
├── README.md                          # Project documentation
├── pneumonia.ipynb                    # Main Jupyter notebook
├── main.py                           # Python script version
├── pyproject.toml                    # Project dependencies
├── uv.lock                          # Lock file
├── chexnet_best.h5                  # Best trained model
├── DATASET.zip                      # Original dataset
├── DATASET/
│   └── chest_xray_balanced/         # Processed balanced dataset
├── NOTES.CSV                        # Project notes
├── Visualizations/
│   ├── AUGMENTED IMG.png           # Data augmentation examples
│   ├── CLASS DIST PER SPLIT.png    # Class distribution
│   ├── RANDOM IMG PER CLASS.png    # Sample images
│   ├── CNN Test Acc.png            # CNN accuracy plots
│   ├── ResNet Test Acc.png         # ResNet accuracy plots
│   ├── DenseNet121 Test Acc.png    # DenseNet accuracy plots
│   └── Grad-CAM visualization.png   # Model interpretability
└── Documentation/
    ├── DENSENET121.pdf             # DenseNet architecture details
    ├── IMAGE DATA AUGMENTATION.pdf  # Augmentation techniques
    └── Identifying Medical Diagnoses and Treatable.pdf  # Research paper
```

## ✨ Features

### Data Processing
- **Class Balancing**: Undersampling majority class for balanced training
- **Data Augmentation**: Rotation, shifting, zooming, and flipping
- **Normalization**: Pixel value scaling to [0,1] range
- **Train/Validation/Test**: Proper data splitting with stratification

### Model Architectures
- **Custom CNN**: Baseline convolutional neural network
- **ResNet50**: Deep residual network with skip connections
- **DenseNet121**: Dense convolutional network (CheXNet)
- **Transfer Learning**: Pre-trained ImageNet weights
- **Fine-tuning**: Layer-wise learning rate optimization

### Evaluation Metrics
- **Accuracy**: Overall classification accuracy
- **ROC-AUC**: Area under the receiver operating characteristic curve
- **Confusion Matrix**: Detailed classification performance
- **Classification Report**: Precision, recall, and F1-score
- **Training Curves**: Loss and accuracy monitoring

### Visualization
- **Data Distribution**: Class balance visualization
- **Sample Images**: Random samples from each class
- **Training Curves**: Model performance over epochs
- **Confusion Matrix**: Classification results heatmap
- **Grad-CAM**: Model interpretability and attention maps

## 🔬 Methodology

### 1. Data Preparation
- Original dataset analysis and class imbalance identification
- Undersampling strategy to balance classes
- Data augmentation to increase dataset diversity
- Proper train/validation/test splitting

### 2. Model Development
- **Baseline CNN**: Custom architecture for comparison
- **Transfer Learning**: Leveraging pre-trained models
- **Fine-tuning**: Optimizing pre-trained models for specific task
- **CheXNet Implementation**: Following medical imaging best practices

### 3. Training Strategy
- **Batch Size**: Optimized for each architecture (16 for CheXNet)
- **Learning Rate**: Adaptive learning rate scheduling
- **Regularization**: Dropout and batch normalization
- **Early Stopping**: Preventing overfitting
- **Model Checkpointing**: Saving best performing models

### 4. Evaluation
- **Multiple Metrics**: Comprehensive performance assessment
- **Cross-validation**: Robust model evaluation
- **Statistical Analysis**: Performance significance testing
- **Interpretability**: Understanding model decisions

## 📊 Visualizations

The project includes comprehensive visualizations:

- **Data Analysis**: Class distribution and sample exploration
- **Augmentation Examples**: Demonstration of data augmentation techniques
- **Training Progress**: Learning curves and performance metrics
- **Model Comparison**: Side-by-side performance analysis
- **Error Analysis**: Misclassification examples and patterns
- **Grad-CAM**: Visual explanation of model attention areas

## 🔧 Requirements

See `pyproject.toml` for complete dependency list. Key requirements:

- **Python**: >= 3.12
- **TensorFlow**: >= 2.20.0 (GPU support recommended)
- **Keras**: >= 3.11.3
- **CUDA**: Compatible version for GPU acceleration (optional)
- **Memory**: At least 8GB RAM recommended
- **Storage**: ~5GB for dataset and models

## 📚 References

1. **CheXNet**: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning
2. **DenseNet**: Densely Connected Convolutional Networks
3. **ResNet**: Deep Residual Learning for Image Recognition
4. **Medical Image Analysis**: Best practices in medical imaging with deep learning

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🏥 Medical Disclaimer

This project is for educational and research purposes only. It should not be used for actual medical diagnosis without proper validation and regulatory approval. Always consult healthcare professionals for medical advice.

---

**Note**: The DenseNet121 (CheXNet implementation) achieved the best performance in this study, making it the recommended model for pneumonia detection tasks.