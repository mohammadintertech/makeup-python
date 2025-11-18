#!/usr/bin/env python3
"""
Flask Makeup API Server - Fixed Version
Install dependencies: pip install -r requirements.txt
"""

import sys
import os

# Set UTF-8 encoding for stdout/stderr to handle unicode characters
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# Import required modules
try:
    import numpy as np
    import cv2
    import io
    import traceback
    import logging
    import gc
    import time
    from flask import Flask, request, send_file, jsonify
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install requirements: pip install -r requirements.txt")
    sys.exit(1)

# Initialize Flask app
app = Flask(__name__)

# Configure logging with ASCII-safe format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration constants
MAX_IMAGE_SIZE = 2048
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

# Import makeup processors with error handling
PROCESSORS_LOADED = False
try:
    from lipstick_processor import apply_lipstick
    from eyeliner_processor import apply_eyeliner
    from blusher_processor import apply_blusher
    from foundation_processor import apply_foundation
    from eyecolor_processor import apply_eyecolor
    from eye_shadow_processor import apply_eyeshadow
    PROCESSORS_LOADED = True
    logger.info("Makeup processors loaded successfully")
except ImportError as e:
    logger.warning(f"Failed to import makeup processors: {e}")
    # Create dummy functions
    def apply_lipstick(image, color, intensity, edge): return image
    def apply_eyeliner(image, color): return image
    def apply_blusher(image, color, intensity): return image
    def apply_foundation(image, color, intensity): return image
    def apply_eyecolor(image, r, g, b): return image
    def apply_eyeshadow(image, color, intensity): return image

# ==================== TEST ENDPOINTS ====================

@app.route("/")
def home():
    """Root endpoint with server info"""
    return jsonify({
        "message": "Makeup API Server is Running",
        "status": "active",
        "endpoints": {
            "test": "/test - Basic functionality test",
            "health": "/health - Detailed health check",
            "ping": "/ping - Quick status check",
            "apply": "/apply - Apply makeup (POST)"
        },
        "processors_loaded": PROCESSORS_LOADED,
        "timestamp": time.time()
    })

@app.route("/test")
def test():
    """Comprehensive test endpoint"""
    test_results = {
        "server_status": "Running",
        "processors_loaded": PROCESSORS_LOADED,
        "flask_import": "Success",
        "numpy_import": "Success",
        "opencv_import": "Success",
        "timestamp": time.time(),
        "python_version": sys.version,
        "test_image": "Try uploading an image to /apply endpoint"
    }
    
    # Test numpy
    try:
        test_array = np.array([1, 2, 3])
        test_results["numpy_test"] = f"Working (sum: {test_array.sum()})"
    except Exception as e:
        test_results["numpy_test"] = f"Failed: {e}"
    
    # Test OpenCV
    try:
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        success, encoded = cv2.imencode(".png", test_image)
        test_results["opencv_test"] = f"Working (encode: {success})"
    except Exception as e:
        test_results["opencv_test"] = f"Failed: {e}"
    
    return jsonify(test_results)

@app.route("/ping")
def ping():
    return jsonify({
        "status": "pong",
        "processors_loaded": PROCESSORS_LOADED,
        "timestamp": time.time()
    })

@app.route("/health")
def health():
    """Detailed health check endpoint"""
    return jsonify({
        "status": "healthy",
        "processors_loaded": PROCESSORS_LOADED,
        "max_image_size": MAX_IMAGE_SIZE,
        "supported_formats": list(SUPPORTED_FORMATS),
        "memory_usage_mb": get_memory_usage()
    })

def get_memory_usage():
    """Get current memory usage in MB"""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except:
        return "N/A"

# ==================== MAIN APPLICATION ENDPOINTS ====================

def validate_image_file(file):
    """Validate uploaded image file"""
    if not file or not file.filename:
        return False, "No file provided"
    
    ext = os.path.splitext(file.filename.lower())[1]
    if ext not in SUPPORTED_FORMATS:
        return False, f"Unsupported format. Supported: {', '.join(SUPPORTED_FORMATS)}"
    
    return True, "Valid"

def safe_process_image(image, processor_func, *args, **kwargs):
    """Safely apply makeup processor with error handling"""
    try:
        return processor_func(image, *args, **kwargs)
    except Exception as e:
        logger.error(f"Processor error: {e}")
        return image

@app.route("/apply", methods=['POST'])
def apply_makeup():
    """Main makeup application endpoint"""
    try:
        logger.info("Received makeup application request")
        
        if not PROCESSORS_LOADED:
            return jsonify({"error": "Makeup processors not loaded properly"}), 500
        
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        is_valid, message = validate_image_file(file)
        if not is_valid:
            return jsonify({"error": message}), 400
        
        # Process image
        try:
            contents = file.read()
            if len(contents) == 0:
                return jsonify({"error": "Empty file"}), 400
            
            np_arr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if image is None:
                return jsonify({"error": "Invalid or corrupted image"}), 400
                
        except Exception as e:
            return jsonify({"error": f"Image processing failed: {str(e)}"}), 400

        # Apply makeup processing
        processed = image.copy()
        form = request.form
        
        # Apply makeup based on form parameters
        try:
            # Lipstick
            if form.get('lipstick_color'):
                color = form.get('lipstick_color', '#FF0000')
                intensity = float(form.get('lipstick_intensity', 0.5))
                edge = int(form.get('lipstick_edge', 1))
                processed = safe_process_image(processed, apply_lipstick, color, intensity, edge)
            
            # Eyeliner
            if form.get('eyeliner_color'):
                color = form.get('eyeliner_color', '#000000')
                processed = safe_process_image(processed, apply_eyeliner, color)
            
            # Blusher
            if form.get('blusher_color'):
                color = form.get('blusher_color', '#FF6B9D')
                intensity = float(form.get('blusher_intensity', 0.3))
                processed = safe_process_image(processed, apply_blusher, color, intensity)
            
            # Foundation
            if form.get('foundation_color'):
                color = form.get('foundation_color', '#F5D6C6')
                intensity = float(form.get('foundation_intensity', 0.2))
                processed = safe_process_image(processed, apply_foundation, color, intensity)
            
            # Eye color
            if form.get('eyecolor_r'):
                r = int(form.get('eyecolor_r', 0))
                g = int(form.get('eyecolor_g', 0))
                b = int(form.get('eyecolor_b', 255))
                processed = safe_process_image(processed, apply_eyecolor, r, g, b)
            
            # Eye shadow
            if form.get('eyeshadow_color'):
                color = form.get('eyeshadow_color', '#8B4513')
                intensity = float(form.get('eyeshadow_intensity', 0.4))
                processed = safe_process_image(processed, apply_eyeshadow, color, intensity)
                
        except Exception as e:
            logger.error(f"Makeup application error: {e}")
            return jsonify({"error": f"Makeup application failed: {str(e)}"}), 400
        
        # Encode result
        encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 6]
        success, img_encoded = cv2.imencode(".png", processed, encode_param)
        
        if not success:
            return jsonify({"error": "Failed to encode result image"}), 500
        
        return send_file(
            io.BytesIO(img_encoded.tobytes()),
            mimetype="image/png",
            as_attachment=False,
            download_name="makeup_result.png"
        )

    except Exception as e:
        logger.error(f"Server error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": "Internal server error"}), 500
    finally:
        # Clean up memory
        gc.collect()

# ==================== WSGI APPLICATION ====================

# WSGI application entry point for hosting platforms
application = app

# For local development
if __name__ == '__main__':
    print("\n" + "="*50)
    print("Starting Makeup API Server...")
    print("="*50)
    print(f"Local URL: http://127.0.0.1:5000")
    print(f"Test URL: http://127.0.0.1:5000/test")
    print(f"Health Check: http://127.0.0.1:5000/health")
    print(f"Processors loaded: {'Yes' if PROCESSORS_LOADED else 'No'}")
    print("="*50 + "\n")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        print(f"Failed to start server: {e}")
        sys.exit(1)