"""
Simple Pneumonia Detection Prediction Example
============================================
This is a simplified example showing how to use your trained model to predict pneumonia.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

def simple_prediction_demo():
    """
    Simple demonstration of how predictions work
    """
    print("🔬 PNEUMONIA DETECTION - SIMPLE PREDICTION EXAMPLE")
    print("=" * 55)
    
    # Check if we have the required libraries
    try:
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing.image import load_img, img_to_array
        print("✅ TensorFlow and Keras imported successfully!")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install required packages: pip install tensorflow keras matplotlib")
        return
    
    # Look for trained model files
    print("\n🔍 Looking for trained model files...")
    model_files = [f for f in os.listdir('.') if f.endswith('.h5')]
    
    if not model_files:
        print("❌ No trained model (.h5) files found!")
        print("Please run the training notebook (pneumonia.ipynb) first to create a model.")
        return
    
    print(f"✅ Found model files: {model_files}")
    
    # Try to load the first model
    model_path = model_files[0]
    print(f"\n🔄 Loading model: {model_path}")
    
    try:
        model = load_model(model_path)
        print(f"✅ Model loaded successfully!")
        print(f"📊 Model input shape: {model.input_shape}")
        print(f"📊 Model output shape: {model.output_shape}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Look for test dataset
    print("\n🔍 Looking for test dataset...")
    test_dirs = ['./DATASET/chest_xray_balanced/test', './DATASET/chest_xray/test']
    test_dir = None
    
    for dir_path in test_dirs:
        if os.path.exists(dir_path):
            test_dir = dir_path
            print(f"✅ Found test directory: {test_dir}")
            break
    
    if not test_dir:
        print("⚠️ Test dataset not found!")
        print("Please extract DATASET.zip to use the test images.")
        
        # Look for any image files in current directory
        image_files = [f for f in os.listdir('.') if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if image_files:
            print(f"\nFound {len(image_files)} image files in current directory:")
            for img in image_files[:5]:  # Show first 5
                print(f"  - {img}")
            
            # Test on first available image
            test_image = image_files[0]
            print(f"\n🔍 Testing on: {test_image}")
            result = predict_image(model, test_image)
            display_result(result)
        return
    
    # Get test images
    test_images = []
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                test_images.append(os.path.join(root, file))
    
    if not test_images:
        print("❌ No images found in test directory!")
        return
    
    print(f"✅ Found {len(test_images)} test images")
    
    # Test on a few random images
    import random
    sample_images = random.sample(test_images, min(3, len(test_images)))
    
    print(f"\n🔍 Testing on {len(sample_images)} random images...")
    print("=" * 55)
    
    results = []
    for i, img_path in enumerate(sample_images, 1):
        print(f"\n[{i}/{len(sample_images)}] Testing: {os.path.basename(img_path)}")
        result = predict_image(model, img_path)
        results.append(result)
        display_result(result)
    
    # Summary
    correct_count = sum(1 for r in results if r['is_correct'])
    accuracy = correct_count / len(results) if results else 0
    
    print(f"\n📊 SUMMARY:")
    print(f"Images tested: {len(results)}")
    print(f"Correct predictions: {correct_count}")
    print(f"Sample accuracy: {accuracy:.1%}")

def predict_image(model, image_path):
    """
    Make a prediction on a single image
    
    Args:
        model: Trained Keras model
        image_path: Path to the image file
    
    Returns:
        dict: Prediction results
    """
    try:
        from tensorflow.keras.preprocessing.image import load_img, img_to_array
        
        # Load and preprocess the image
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = img_array / 255.0  # Normalize to [0,1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Make prediction
        prediction = model.predict(img_array, verbose=0)
        prediction_prob = prediction[0][0]
        
        # Determine predicted class
        if prediction_prob > 0.5:
            predicted_class = 'PNEUMONIA'
            confidence = prediction_prob
        else:
            predicted_class = 'NORMAL'
            confidence = 1 - prediction_prob
        
        # Get actual class from filename/path
        actual_class = get_actual_class(image_path)
        is_correct = predicted_class == actual_class
        
        return {
            'image_path': image_path,
            'predicted_class': predicted_class,
            'actual_class': actual_class,
            'confidence': confidence,
            'raw_probability': prediction_prob,
            'is_correct': is_correct
        }
        
    except Exception as e:
        print(f"❌ Error predicting image {image_path}: {e}")
        return None

def get_actual_class(image_path):
    """Get the actual class from the image path"""
    path_upper = image_path.upper()
    if 'PNEUMONIA' in path_upper:
        return 'PNEUMONIA'
    elif 'NORMAL' in path_upper:
        return 'NORMAL'
    else:
        return 'UNKNOWN'

def display_result(result):
    """Display prediction result"""
    if result is None:
        return
    
    filename = os.path.basename(result['image_path'])
    predicted = result['predicted_class']
    actual = result['actual_class']
    confidence = result['confidence']
    raw_prob = result['raw_probability']
    is_correct = result['is_correct']
    
    print(f"📋 File: {filename}")
    print(f"🔮 Predicted: {predicted} ({confidence:.1%} confidence)")
    print(f"✅ Actual: {actual}")
    print(f"📊 Raw probability: {raw_prob:.4f}")
    print(f"{'✅ CORRECT' if is_correct else '❌ INCORRECT'} prediction")
    
    # Show decision logic
    print("💡 Model Logic:")
    print("   • Score > 0.5 → PNEUMONIA")
    print("   • Score ≤ 0.5 → NORMAL")

if __name__ == "__main__":
    simple_prediction_demo()