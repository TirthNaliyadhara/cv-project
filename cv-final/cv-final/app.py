# app.py
from flask import Flask, request, jsonify, render_template, send_from_directory
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import base64
import io
import os
from PIL import Image
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')

# Load model once at startup
MODEL_PATH = 'dependencies/candlestick_classifier.keras'
model = None
class_labels = ['buyinput', 'sellinput', 'sidewaysinput']  

def load_model():
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found at path: {MODEL_PATH}")
            return False
            
        model = keras.models.load_model(MODEL_PATH)
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def crop_and_detect_edges(image_data):
    """Process the image for candlestick pattern recognition with improved handling for various chart types."""
    try:
        # Ensure we're working with just the base64 data
        if ',' in image_data:
            image_data = image_data.split(',', 1)[1]
        
        # Convert base64 to numpy array
        img_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(img_bytes))
        img = np.array(image)
        
        # Convert to BGR format if needed (for OpenCV compatibility)
        if len(img.shape) == 3:
            if img.shape[2] == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif img.shape[2] == 3:  # RGB
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif len(img.shape) == 2:  # Already grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Save original image for reference
        img_original = img.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Determine if background is dark or light
        avg_intensity = np.mean(gray)
        is_dark_background = avg_intensity < 127
        logger.info(f"Average intensity: {avg_intensity}, Dark background: {is_dark_background}")
        
        # Create a copy for chart area detection
        detection_img = gray.copy()
        
        # For TradingView-like images (white background with UI elements)
        if not is_dark_background:
            # Apply adaptive thresholding for better feature detection on light backgrounds
            binary = cv2.adaptiveThreshold(detection_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 11, 2)
            
            # Remove small noise
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # For volume bars detection - find horizontal regions with vertical bars
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
            
            # Find contours for potential chart areas
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            # For dark background charts (traditional trading platform)
            _, binary = cv2.threshold(detection_img, 40, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Initialize variables for chart area
        chart_x, chart_y, chart_w, chart_h = 0, 0, img.shape[1], img.shape[0]
        chart_area_found = False
        
        # Try to locate the chart area by finding significant contours
        if contours:
            # Filter contours by size to avoid UI elements
            significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > img.shape[0] * img.shape[1] * 0.05]
            
            if significant_contours:
                # Find the largest contour that's likely to be the chart area
                largest_contour = max(significant_contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Only use if the contour is substantial
                if w > img.shape[1] * 0.3 and h > img.shape[0] * 0.3:
                    chart_x, chart_y, chart_w, chart_h = x, y, w, h
                    chart_area_found = True
                    logger.info(f"Chart area detected: x={x}, y={y}, w={w}, h={h}")
        
        # If no good contour was found, try an alternative approach based on color detection
        if not chart_area_found:
            logger.info("Using color-based chart area detection")
            
            # For light backgrounds (TradingView-like), look for the grid lines
            if not is_dark_background:
                # Convert to HSV for better color detection
                hsv = cv2.cvtColor(img_original, cv2.COLOR_BGR2HSV)
                
                # Detect grid lines (typically light gray in TradingView)
                lower_gray = np.array([0, 0, 200])
                upper_gray = np.array([180, 30, 255])
                grid_mask = cv2.inRange(hsv, lower_gray, upper_gray)
                
                # Find horizontal and vertical lines to identify the grid area
                horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
                vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
                
                horizontal_lines = cv2.morphologyEx(grid_mask, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
                vertical_lines = cv2.morphologyEx(grid_mask, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
                
                # Combine to get grid area
                grid_area = cv2.bitwise_or(horizontal_lines, vertical_lines)
                
                # Find contours of the grid area
                grid_contours, _ = cv2.findContours(grid_area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if grid_contours:
                    # Find the largest grid contour
                    largest_grid = max(grid_contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_grid)
                    
                    # Use if substantial
                    if w > img.shape[1] * 0.3 and h > img.shape[0] * 0.3:
                        chart_x, chart_y, chart_w, chart_h = x, y, w, h
                        chart_area_found = True
                        logger.info(f"Grid-based chart area: x={x}, y={y}, w={w}, h={h}")
        
        # If still no chart area found, use a reasonable default (middle portion of image)
        if not chart_area_found:
            logger.info("Using default chart area (middle portion)")
            # Cut off potential UI elements by using the middle portion
            margin_x = int(img.shape[1] * 0.1)
            margin_y = int(img.shape[0] * 0.1)
            chart_x = margin_x
            chart_y = margin_y
            chart_w = img.shape[1] - (2 * margin_x)
            chart_h = img.shape[0] - (2 * margin_y)
        
        # Crop the chart region with a small margin
        cropped_chart = img_original[chart_y:chart_y+chart_h, chart_x:chart_x+chart_w]
        
        # Resize cropped image to match model input size
        cropped_chart_resized = cv2.resize(cropped_chart, (512, 512), interpolation=cv2.INTER_AREA)
        
        # Convert to grayscale for edge detection
        cropped_gray = cv2.cvtColor(cropped_chart_resized, cv2.COLOR_BGR2GRAY)
        
        # If light background, invert the image for consistent processing
        if not is_dark_background:
            cropped_gray = cv2.bitwise_not(cropped_gray)
        
        # Apply Gaussian Blur to reduce noise
        blurred_gray = cv2.GaussianBlur(cropped_gray, (5, 5), 0)
        
        # Apply Canny Edge Detection with adaptive parameters
        if is_dark_background:
            edges = cv2.Canny(blurred_gray, threshold1=20, threshold2=100)
        else:
            # For inverted light backgrounds, adjust Canny parameters
            edges = cv2.Canny(blurred_gray, threshold1=30, threshold2=150)
        
        # Convert edges to 3-channel image (if model expects RGB)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        # Convert processed image back to base64 for display
        _, encoded_img = cv2.imencode('.png', edges)
        processed_img_base64 = base64.b64encode(encoded_img).decode('utf-8')
        
        # Return the processed image and the tensor for prediction
        img_tensor = tf.convert_to_tensor(edges_rgb, dtype=tf.float32)
        img_tensor = img_tensor / 255.0  # Normalize
        img_tensor = tf.expand_dims(img_tensor, 0)  # Add batch dimension
        
        return processed_img_base64, img_tensor

    except Exception as e:
        logger.error(f"Error in image processing: {e}")
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/api/predict', methods=['POST'])
def predict():
    if not model:
        logger.error("Model not loaded")
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.json
        
        if not data or 'image' not in data:
            logger.warning("No image data provided")
            return jsonify({'error': 'No image data provided'}), 400
        
        # Get image data
        image_data = data['image']
        
        # Process image
        processed_img_base64, img_tensor = crop_and_detect_edges(image_data)
        
        # Make prediction
        predictions = model.predict(img_tensor)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = class_labels[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx])
        
        logger.info(f"Prediction: {predicted_class} with confidence {confidence:.4f}")
        
        # Map class labels to user-friendly names
        class_display_names = {
            'buyinput': 'BUY',
            'sellinput': 'SELL',
            'sidewaysinput': 'WAIT'
        }
        
        display_class = class_display_names.get(predicted_class, predicted_class)
        
        # Return results
        return jsonify({
            'class': predicted_class,
            'displayClass': display_class,
            'confidence': confidence,
            'processedImage': f'data:image/png;base64,{processed_img_base64}'
        })
    
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/model_status')
def model_status():
    status = model is not None
    logger.info(f"Model status requested. Status: {'Loaded' if status else 'Not loaded'}")
    return jsonify({'loaded': status})

if __name__ == '__main__':
    # Load the model before starting the server
    if load_model():
        app.run(debug=False, host='0.0.0.0', port=5000)
    else:
        logger.critical("Failed to load model. Exiting.")