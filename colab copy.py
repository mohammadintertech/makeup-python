# First run this installation cell
import os

# After runtime restarts, run this cell with your complete application:
from flask import Flask, request, send_file, jsonify
import numpy as np
import cv2
import io
import traceback
import logging
import os
import gc
from pyngrok import ngrok
import threading
import time
import requests
import sys
sys.path.append("/content/sample_data")

# Import makeup processors with error handling
try:
    from lipstick_processor import apply_lipstick
    from eyeliner_processor import apply_eyeliner
    from blusher_processor import apply_blusher
    from foundation_processor import apply_foundation
    from eyecolor_processor import apply_eyecolor
    from eye_shadow_processor import apply_eyeshadow
    PROCESSORS_LOADED = True
except ImportError as e:
    logging.error(f"Failed to import makeup processors: {e}")
    PROCESSORS_LOADED = False

# Initialize Flask app
app = Flask(__name__)

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration constants
MAX_IMAGE_SIZE = 2048
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

# Set your ngrok authtoken
ngrok.set_auth_token("30XtsXpxg1sYXAd6QnkfkRUKTi6_3ATFdXVW7zNc88ZK3t8Sh")

@app.route("/test")
def test():
    status = "✅ Running" if PROCESSORS_LOADED else "⚠️ Running (processors not loaded)"
    return f"Makeup API is {status} properly in Colab!"

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

def validate_image_file(file):
    """Validate uploaded image file"""
    if not file or not file.filename:
        return False, "No file provided"
    
    # Check file extension
    ext = os.path.splitext(file.filename.lower())[1]
    if ext not in SUPPORTED_FORMATS:
        return False, f"Unsupported format. Supported: {', '.join(SUPPORTED_FORMATS)}"
    
    return True, "Valid"

def safe_process_image(image, processor_func, *args, **kwargs):
    """Safely apply makeup processor with error handling"""
    try:
        return processor_func(image, *args, **kwargs)
    except ValueError as e:
        logger.warning(f"Processor failed with ValueError: {e}")
        return image  # Return original image
    except RuntimeError as e:
        logger.warning(f"Processor failed with RuntimeError: {e}")
        return image  # Return original image
    except Exception as e:
        logger.error(f"Unexpected processor error: {e}")
        return image  # Return original image

@app.route("/apply", methods=['POST'])
def apply_makeup():
    """Main makeup application endpoint with comprehensive error handling"""
    try:
        logger.info("Received makeup application request")
        
        # Check if processors are loaded
        if not PROCESSORS_LOADED:
            return jsonify({"error": "Makeup processors not loaded properly"}), 500
        
        # Validate file upload
        if 'file' not in request.files:
            logger.error("No file uploaded")
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        
        # Validate file
        is_valid, message = validate_image_file(file)
        if not is_valid:
            logger.error(f"Invalid file: {message}")
            return jsonify({"error": message}), 400
        
        logger.info(f"Processing file: {file.filename}")
        
        # Process image with enhanced error handling
        try:
            contents = file.read()
            if len(contents) == 0:
                return jsonify({"error": "Empty file"}), 400
            
            np_arr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if image is None:
                logger.error("Failed to decode image - possibly corrupted")
                return jsonify({"error": "Invalid or corrupted image"}), 400
            
            # Check image dimensions
            if len(image.shape) != 3 or image.shape[2] != 3:
                logger.error(f"Invalid image format: shape={image.shape}")
                return jsonify({"error": "Image must be a color image (3 channels)"}), 400
                
            # Resize if too large
            original_size = max(image.shape)
            if original_size > MAX_IMAGE_SIZE:
                logger.info(f"Resizing large image from {image.shape} (size: {original_size})")
                scale = MAX_IMAGE_SIZE / original_size
                new_width = int(image.shape[1] * scale)
                new_height = int(image.shape[0] * scale)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                logger.info(f"Resized to: {image.shape}")
                
        except Exception as e:
            logger.error(f"Image processing error: {str(e)}")
            return jsonify({"error": f"Image processing failed: {str(e)}"}), 400

        # Initialize processing
        processed = image.copy()
        form = request.form
        processing_steps = []
        failed_steps = []
        
        try:
            # Foundation processing
            if form.get('enable_foundation') == 'true':
                logger.info("Applying foundation")
                try:
                    r, g, b = form.get('foundation_r'), form.get('foundation_g'), form.get('foundation_b')
                    intensity = form.get('foundation_intensity')
                    if all([r, g, b, intensity]):
                        processed = safe_process_image(processed, apply_foundation, 
                                                     [int(r), int(g), int(b)], float(intensity))
                        processing_steps.append("foundation")
                        gc.collect()
                    else:
                        failed_steps.append("foundation (missing parameters)")
                except Exception as e:
                    logger.error(f"Foundation processing failed: {e}")
                    failed_steps.append(f"foundation ({str(e)})")
            
            # Blusher processing with robust error handling
            if form.get('enable_blusher') == 'true':
                logger.info("Applying blusher")
                try:
                    r, g, b = form.get('blusher_r'), form.get('blusher_g'), form.get('blusher_b')
                    intensity = form.get('blusher_intensity')
                    if all([r, g, b, intensity]):
                        processed = safe_process_image(processed, apply_blusher, 
                                                     [int(r), int(g), int(b)], float(intensity))
                        processing_steps.append("blusher")
                        gc.collect()
                    else:
                        failed_steps.append("blusher (missing parameters)")
                except Exception as e:
                    logger.error(f"Blusher processing failed: {e}")
                    failed_steps.append(f"blusher ({str(e)})")
            
            # Lipstick processing
            if form.get('enable_lipstick') == 'true':
                logger.info("Applying lipstick")
                try:
                    r, g, b = form.get('lipstick_r'), form.get('lipstick_g'), form.get('lipstick_b')
                    intensity = form.get('lipstick_intensity')
                    edge = form.get('lipstick_edge')
                    if all([r, g, b, intensity, edge]):
                        processed = safe_process_image(processed, apply_lipstick, 
                                                     [int(r), int(g), int(b)], float(intensity), int(edge))
                        processing_steps.append("lipstick")
                        gc.collect()
                    else:
                        failed_steps.append("lipstick (missing parameters)")
                except Exception as e:
                    logger.error(f"Lipstick processing failed: {e}")
                    failed_steps.append(f"lipstick ({str(e)})")
            
            # Eyeliner processing
            if form.get('enable_eyeliner') == 'true':
                logger.info("Applying eyeliner")
                try:
                    r, g, b = form.get('eyeliner_r'), form.get('eyeliner_g'), form.get('eyeliner_b')
                    if all([r, g, b]):
                        processed = safe_process_image(processed, apply_eyeliner, [int(r), int(g), int(b)])
                        processing_steps.append("eyeliner")
                        gc.collect()
                    else:
                        failed_steps.append("eyeliner (missing parameters)")
                except Exception as e:
                    logger.error(f"Eyeliner processing failed: {e}")
                    failed_steps.append(f"eyeliner ({str(e)})")
            
            # Eye color processing
            if form.get('enable_eyecolor') == 'true':
                logger.info("Applying eye color")
                try:
                    r, g, b = form.get('eyecolor_r'), form.get('eyecolor_g'), form.get('eyecolor_b')
                    if all([r, g, b]):
                        processed = safe_process_image(processed, apply_eyecolor, int(r), int(g), int(b))
                        processing_steps.append("eyecolor")
                        gc.collect()
                    else:
                        failed_steps.append("eyecolor (missing parameters)")
                except Exception as e:
                    logger.error(f"Eye color processing failed: {e}")
                    failed_steps.append(f"eyecolor ({str(e)})")
            
            # Eyeshadow processing
            if form.get('enable_eyeshadow') == 'true':
                logger.info("Applying eyeshadow")
                try:
                    r, g, b = form.get('eyeshadow_r'), form.get('eyeshadow_g'), form.get('eyeshadow_b')
                    intensity = form.get('eyeshadow_intensity')
                    if all([r, g, b, intensity]):
                        processed = safe_process_image(processed, apply_eyeshadow, 
                                                     [int(r), int(g), int(b)], float(intensity))
                        processing_steps.append("eyeshadow")
                        gc.collect()
                    else:
                        failed_steps.append("eyeshadow (missing parameters)")
                except Exception as e:
                    logger.error(f"Eyeshadow processing failed: {e}")
                    failed_steps.append(f"eyeshadow ({str(e)})")
            
            # Log results
            if processing_steps:
                logger.info(f"Successfully completed: {', '.join(processing_steps)}")
            if failed_steps:
                logger.warning(f"Failed steps: {', '.join(failed_steps)}")
            
            # Encode result
            try:
                encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 6]  # Medium compression
                success, img_encoded = cv2.imencode(".png", processed, encode_param)
                
                if not success:
                    logger.error("Failed to encode processed image")
                    return jsonify({"error": "Failed to encode result image"}), 500
                
                # Create response
                response = send_file(
                    io.BytesIO(img_encoded.tobytes()),
                    mimetype="image/png",
                    as_attachment=False,
                    download_name="makeup_result.png"
                )
                
                # Add custom headers with processing info
                response.headers['X-Processing-Steps'] = ','.join(processing_steps) if processing_steps else 'none'
                response.headers['X-Failed-Steps'] = ','.join(failed_steps) if failed_steps else 'none'
                
                # Cleanup memory
                del image, processed, img_encoded, contents, np_arr
                gc.collect()
                
                return response
                
            except Exception as e:
                logger.error(f"Image encoding error: {str(e)}")
                return jsonify({"error": f"Failed to encode result: {str(e)}"}), 500
            
        except Exception as e:
            logger.error(f"Processing pipeline error: {str(e)}\n{traceback.format_exc()}")
            return jsonify({"error": f"Processing failed: {str(e)}"}), 500

    except Exception as e:
        logger.error(f"Unexpected server error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large"}), 413

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500

def run_flask():
    """Run Flask app with error handling"""
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except Exception as e:
        logger.error(f"Flask server error: {e}")

# Start Flask in thread
flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()

# Enhanced keep-alive function
def keep_alive():
    """Enhanced keep-alive with health monitoring"""
    consecutive_failures = 0
    while True:
        try:
            response = requests.get("http://127.0.0.1:5000/ping", timeout=5)
            if response.status_code == 200:
                consecutive_failures = 0
                logger.debug("Keep-alive ping successful")
            else:
                consecutive_failures += 1
                logger.warning(f"Keep-alive ping failed with status {response.status_code}")
        except Exception as e:
            consecutive_failures += 1
            logger.warning(f"Keep-alive ping failed: {e}")
        
        if consecutive_failures > 5:
            logger.error("Multiple consecutive keep-alive failures - server may be unresponsive")
        
        time.sleep(30)

# Start enhanced keep-alive
keep_alive_thread = threading.Thread(target=keep_alive, daemon=True)
keep_alive_thread.start()

# Setup ngrok with error handling
try:
    public_url = ngrok.connect(5000).public_url
    logger.info(f"Ngrok tunnel established: {public_url}")
    print(f"\n🎨 Makeup API is running!")
    print(f"📱 Public URL: {public_url}")
    print(f"🔍 Test URL: {public_url}/test")
    print(f"❤️ Health Check: {public_url}/health")
    print("⚡ Keep this cell running!\n")
except Exception as e:
    logger.error(f"Failed to setup ngrok tunnel: {e}")
    print("❌ Failed to setup public URL. Check ngrok configuration.")

# Keep the cell alive with better monitoring
startup_time = time.time()
try:
    while True:
        # Simple heartbeat every 60 seconds
        if int(time.time() - startup_time) % 60 == 0:
            uptime_hours = (time.time() - startup_time) / 3600
            logger.info(f"Server running - Uptime: {uptime_hours:.1f} hours")
        time.sleep(1)
except KeyboardInterrupt:
    logger.info("Server sshutdown requested")
    print("🛑 Server shutting down...")
except Exception as e:
    logger.error(f"Server loop error: {e}")
    print("❌ Server encountered an error")