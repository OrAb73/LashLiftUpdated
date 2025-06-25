from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import base64
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for Flutter app communication

# Global variables for model
model = None
device = None
transform = None
class_names = ['lift_rod_eyes', 'round_rod_eyes']

def initialize_model():
    """Initialize the PyTorch model - same in training script"""
    model = models.resnet50(weights='IMAGENET1K_V1')
    
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze specific layers (same as your training script)
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    # Modify final layer for binary classification
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 2)  # 2 classes: lift rod and round rod
    )
    
    return model

def load_model(model_path):
    """Load the trained model"""
    global model, device, transform
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize model
    model = initialize_model()
    
    # Load trained weights
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"Model loaded from {model_path}")
    else:
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = model.to(device)
    model.eval()
    
    # Define the same transform used during training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def predict_image(image):
    """Make prediction on a single image"""
    try:
        # Convert image to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        image_tensor = image_tensor.to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            _, preds = torch.max(outputs, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
        predicted_class = class_names[preds[0].item()]
        confidence = probs[0][preds[0]].item()
        
        return predicted_class, confidence
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise e

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device) if device else 'not set'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get image from request
        if 'image' not in request.json:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode base64 image
        image_data = request.json['image']
        
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Make prediction
        predicted_class, confidence = predict_image(image)
        
        # Prepare response
        response = {
            'prediction': predicted_class,
            'confidence': float(confidence),
            'success': True
        }
        
        logger.info(f"Prediction: {predicted_class} with confidence: {confidence:.4f}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in /predict endpoint: {str(e)}")
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/classes', methods=['GET'])
def get_classes():
    """Get available classes"""
    return jsonify({
        'classes': class_names,
        'num_classes': len(class_names)
    })

if __name__ == '__main__':
    MODEL_PATH = r'C:\Users\Or\lash_rod_classification\data\lash_rod_classifier.pth'
    
    
    try:
        # Load the model
        load_model(MODEL_PATH)
        logger.info("Model loaded successfully!")
        
        # Start the Flask server
        app.run(host='0.0.0.0', port=5000, debug=True)
        
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        print(f"Error: {str(e)}")
        print("Make sure your model file exists and the path is correct!")