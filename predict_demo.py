"""
Pneumonia Detection - Prediction Demo
=====================================
This script demonstrates how to use the trained model to predict pneumonia from chest X-ray images.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import random
import pandas as pd

class PneumoniaPredictor:
    def __init__(self, model_path=None):
        """
        Initialize the Pneumonia Predictor
        
        Args:
            model_path (str): Path to the saved model file
        """
        self.model = None
        self.img_size = (224, 224)
        self.class_names = ['NORMAL', 'PNEUMONIA']
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print("⚠️ Model file not found. You'll need to train the model first.")
            print("Available model files:")
            for file in os.listdir('.'):
                if file.endswith('.h5'):
                    print(f"  - {file}")
    
    def load_model(self, model_path):
        """Load the trained model"""
        try:
            self.model = load_model(model_path)
            print(f"✅ Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    
    def preprocess_image(self, image_path):
        """
        Preprocess a single image for prediction
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            np.array: Preprocessed image ready for prediction
        """
        try:
            # Load and resize image
            img = load_img(image_path, target_size=self.img_size)
            img_array = img_to_array(img)
            
            # Normalize pixel values to [0,1]
            img_array = img_array / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array, img
        
        except Exception as e:
            print(f"❌ Error preprocessing image {image_path}: {e}")
            return None, None
    
    def predict_single_image(self, image_path, show_image=True):
        """
        Predict pneumonia for a single image
        
        Args:
            image_path (str): Path to the image file
            show_image (bool): Whether to display the image
            
        Returns:
            dict: Prediction results
        """
        if self.model is None:
            print("❌ No model loaded. Please load a model first.")
            return None
        
        # Preprocess image
        img_array, original_img = self.preprocess_image(image_path)
        if img_array is None:
            return None
        
        # Make prediction
        prediction_prob = self.model.predict(img_array, verbose=0)[0][0]
        
        # Determine class and confidence
        if prediction_prob > 0.5:
            predicted_class = 'PNEUMONIA'
            confidence = prediction_prob
        else:
            predicted_class = 'NORMAL'
            confidence = 1 - prediction_prob
        
        # Get actual label from filename/path
        actual_class = self.get_actual_class(image_path)
        is_correct = predicted_class == actual_class
        
        result = {
            'image_path': image_path,
            'predicted_class': predicted_class,
            'actual_class': actual_class,
            'confidence': confidence,
            'prediction_probability': prediction_prob,
            'is_correct': is_correct
        }
        
        # Display results
        print(f"\n📋 Prediction Results for: {os.path.basename(image_path)}")
        print(f"🔮 Predicted: {predicted_class} ({confidence:.2%} confidence)")
        print(f"✅ Actual: {actual_class}")
        print(f"{'✅ CORRECT' if is_correct else '❌ INCORRECT'} prediction")
        print(f"📊 Raw probability: {prediction_prob:.4f}")
        
        # Show image if requested
        if show_image and original_img:
            self.display_prediction(original_img, result)
        
        return result
    
    def get_actual_class(self, image_path):
        """
        Extract actual class from image path
        Assumes path contains 'NORMAL' or 'PNEUMONIA'
        """
        path_upper = image_path.upper()
        if 'PNEUMONIA' in path_upper:
            return 'PNEUMONIA'
        elif 'NORMAL' in path_upper:
            return 'NORMAL'
        else:
            return 'UNKNOWN'
    
    def display_prediction(self, img, result):
        """Display image with prediction results"""
        plt.figure(figsize=(10, 6))
        
        # Display image
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Chest X-ray\n{os.path.basename(result['image_path'])}")
        
        # Display prediction info
        plt.subplot(1, 2, 2)
        plt.axis('off')
        
        # Create prediction visualization
        confidence = result['confidence']
        predicted = result['predicted_class']
        actual = result['actual_class']
        is_correct = result['is_correct']
        
        info_text = f"""
PREDICTION RESULTS

🔮 Predicted: {predicted}
✅ Actual: {actual}
📊 Confidence: {confidence:.2%}
📈 Raw Score: {result['prediction_probability']:.4f}

{'✅ CORRECT PREDICTION' if is_correct else '❌ INCORRECT PREDICTION'}

Model Decision:
• Score > 0.5 → PNEUMONIA
• Score ≤ 0.5 → NORMAL
"""
        
        plt.text(0.1, 0.5, info_text, fontsize=12, 
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", 
                         facecolor='lightgreen' if is_correct else 'lightcoral'))
        
        plt.tight_layout()
        plt.show()
    
    def predict_batch(self, test_dir, num_samples=10, show_images=False):
        """
        Predict on a batch of test images
        
        Args:
            test_dir (str): Directory containing test images
            num_samples (int): Number of samples to test
            show_images (bool): Whether to display images
            
        Returns:
            pd.DataFrame: Results dataframe
        """
        if not os.path.exists(test_dir):
            print(f"❌ Test directory not found: {test_dir}")
            return None
        
        # Get all image files
        image_files = []
        for root, dirs, files in os.walk(test_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(root, file))
        
        if not image_files:
            print(f"❌ No image files found in {test_dir}")
            return None
        
        # Randomly sample images
        sample_images = random.sample(image_files, min(num_samples, len(image_files)))
        
        print(f"\n🔍 Testing on {len(sample_images)} random images from {test_dir}")
        print("="*60)
        
        results = []
        correct_predictions = 0
        
        for i, img_path in enumerate(sample_images, 1):
            print(f"\n[{i}/{len(sample_images)}] Processing: {os.path.basename(img_path)}")
            
            result = self.predict_single_image(img_path, show_image=show_images)
            if result:
                results.append(result)
                if result['is_correct']:
                    correct_predictions += 1
        
        # Create results dataframe
        if results:
            df_results = pd.DataFrame(results)
            
            # Calculate accuracy
            accuracy = correct_predictions / len(results)
            
            print(f"\n📊 BATCH PREDICTION SUMMARY")
            print("="*60)
            print(f"Total Images: {len(results)}")
            print(f"Correct Predictions: {correct_predictions}")
            print(f"Incorrect Predictions: {len(results) - correct_predictions}")
            print(f"Accuracy: {accuracy:.2%}")
            
            # Show class-wise performance
            class_performance = df_results.groupby('actual_class').agg({
                'is_correct': ['count', 'sum']
            }).round(3)
            
            print(f"\n📈 CLASS-WISE PERFORMANCE:")
            for class_name in ['NORMAL', 'PNEUMONIA']:
                if class_name in df_results['actual_class'].values:
                    class_data = df_results[df_results['actual_class'] == class_name]
                    class_accuracy = class_data['is_correct'].mean()
                    class_count = len(class_data)
                    class_correct = class_data['is_correct'].sum()
                    print(f"  {class_name}: {class_correct}/{class_count} correct ({class_accuracy:.2%})")
            
            return df_results
        
        return None

def demo_prediction():
    """Demonstration function showing how to use the predictor"""
    
    print("🔬 PNEUMONIA DETECTION - PREDICTION DEMO")
    print("="*50)
    
    # Initialize predictor
    predictor = PneumoniaPredictor()
    
    # Try to find a model file
    model_files = [f for f in os.listdir('.') if f.endswith('.h5')]
    
    if not model_files:
        print("\n❌ No trained model found!")
        print("Please ensure you have a trained model file (.h5) in the current directory.")
        print("You can train a model using the pneumonia.ipynb notebook.")
        return
    
    # Load the first available model
    model_file = model_files[0]
    predictor.load_model(model_file)
    
    # Check for test directory
    test_dirs = ['./DATASET/chest_xray_balanced/test', './DATASET/chest_xray/test']
    test_dir = None
    
    for dir_path in test_dirs:
        if os.path.exists(dir_path):
            test_dir = dir_path
            break
    
    if test_dir:
        print(f"\n🎯 Running batch prediction on test dataset: {test_dir}")
        results = predictor.predict_batch(test_dir, num_samples=5, show_images=True)
        
        if results is not None:
            # Save results
            results.to_csv('prediction_results.csv', index=False)
            print(f"\n💾 Results saved to 'prediction_results.csv'")
    else:
        print("\n⚠️ Test dataset not found.")
        print("Please ensure you have extracted the DATASET.zip file.")
        
        # Demo with a single image if available
        print("\nLooking for individual image files...")
        image_files = [f for f in os.listdir('.') if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if image_files:
            sample_image = image_files[0]
            print(f"\n🔍 Testing on sample image: {sample_image}")
            predictor.predict_single_image(sample_image, show_image=True)
        else:
            print("No image files found for testing.")

if __name__ == "__main__":
    demo_prediction()