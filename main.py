from flask import Flask, request, send_file, jsonify
import numpy as np
import cv2
import io
import traceback
import logging
from lipstick_processor import apply_lipstick
from eyeliner_processor import apply_eyeliner
from blusher_processor import apply_blusher
from foundation_processor import apply_foundation
from eyecolor_processor import apply_eyecolor
from eye_shadow_processor import apply_eyeshadow
import os
import gc

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set maximum image size to prevent memory issues
MAX_IMAGE_SIZE = 2048  # pixels

@app.route("/test")
def test():
    return "Server is running!"

@app.route("/apply", methods=['POST'])
def apply_makeup():
    try:
        logger.info("Received makeup application request")
        
        # File handling
        if 'file' not in request.files:
            logger.error("No file uploaded")
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        logger.info(f"Processing file: {file.filename}")
        
        try:
            # Read and decode image with size check
            contents = file.read()
            np_arr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if image is None:
                logger.error("Failed to decode image")
                return jsonify({"error": "Invalid image"}), 400
                
            # Check image dimensions
            if max(image.shape) > MAX_IMAGE_SIZE:
                logger.warning(f"Image too large, resizing. Original size: {image.shape}")
                scale = MAX_IMAGE_SIZE / max(image.shape)
                image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
                
        except Exception as e:
            logger.error(f"Image processing error: {str(e)}")
            return jsonify({"error": f"Image processing failed: {str(e)}"}), 400

        # Process image
        processed = image.copy()
        form = request.form
        
        # Track processing steps
        processing_steps = []
        
        try:
            # Foundation
            if form.get('enable_foundation') == 'true':
                logger.info("Applying foundation")
                r, g, b = form.get('foundation_r'), form.get('foundation_g'), form.get('foundation_b')
                intensity = form.get('foundation_intensity')
                if all([r, g, b, intensity]):
                    processed = apply_foundation(processed, [int(r), int(g), int(b)], float(intensity))
                    processing_steps.append("foundation")
                    gc.collect()  # Force garbage collection
            
            # Blusher
            if form.get('enable_blusher') == 'true':
                logger.info("Applying blusher")
                r, g, b = form.get('blusher_r'), form.get('blusher_g'), form.get('blusher_b')
                intensity = form.get('blusher_intensity')
                if all([r, g, b, intensity]):
                    processed = apply_blusher(processed, [int(r), int(g), int(b)], float(intensity))
                    processing_steps.append("blusher")
                    gc.collect()
            
            # Lipstick
            if form.get('enable_lipstick') == 'true':
                logger.info("Applying lipstick")
                r, g, b = form.get('lipstick_r'), form.get('lipstick_g'), form.get('lipstick_b')
                intensity = form.get('lipstick_intensity')
                edge = form.get('lipstick_edge')
                
                if all([r, g, b, intensity, edge]):
                    try:
                        processed = apply_lipstick(processed, [int(r), int(g), int(b)], float(intensity), int(edge))
                        processing_steps.append("lipstick")
                        gc.collect()
                    except Exception as e:
                        logger.error(f"Lipstick application failed: {str(e)}")
                        # Continue with other processing steps even if lipstick fails
            
            # Eyeliner
            if form.get('enable_eyeliner') == 'true':
                logger.info("Applying eyeliner")
                r, g, b = form.get('eyeliner_r'), form.get('eyeliner_g'), form.get('eyeliner_b')
                if all([r, g, b]):
                    processed = apply_eyeliner(processed, [int(r), int(g), int(b)])
                    processing_steps.append("eyeliner")
                    gc.collect()
            
            # Eye color
            if form.get('enable_eyecolor') == 'true':
                logger.info("Applying eye color")
                r, g, b = form.get('eyecolor_r'), form.get('eyecolor_g'), form.get('eyecolor_b')
                if all([r, g, b]):
                    processed = apply_eyecolor(processed, int(r), int(g), int(b))
                    processing_steps.append("eyecolor")
                    gc.collect()
            
            # Eyeshadow
            if form.get('enable_eyeshadow') == 'true':
                logger.info("Applying eyeshadow")
                r, g, b = form.get('eyeshadow_r'), form.get('eyeshadow_g'), form.get('eyeshadow_b')
                intensity = form.get('eyeshadow_intensity')
                if all([r, g, b, intensity]):
                    processed = apply_eyeshadow(processed, [int(r), int(g), int(b)], float(intensity))
                    processing_steps.append("eyeshadow")
                    gc.collect()
            
            logger.info(f"Completed processing steps: {', '.join(processing_steps)}")
            
            # Encode the final image
            try:
                _, img_encoded = cv2.imencode(".png", processed)
                logger.info("Image successfully processed and encoded")
                
                # Create response
                response = send_file(
                    io.BytesIO(img_encoded.tobytes()),
                    mimetype="image/png",
                    as_attachment=False,
                    download_name="result.png"
                )
                
                # Clear memory
                del image, processed, img_encoded
                gc.collect()
                
                return response
                
            except Exception as e:
                logger.error(f"Image encoding error: {str(e)}")
                return jsonify({"error": f"Failed to encode final image: {str(e)}"}), 500
                
        except Exception as e:
            logger.error(f"Processing error at steps {processing_steps}: {str(e)}\n{traceback.format_exc()}")
            return jsonify({
                "error": f"Makeup application failed at step: {processing_steps[-1] if processing_steps else 'unknown'}",
                "details": str(e)
            }), 500

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            "error": "Unexpected server error",
            "details": str(e)
        }), 500

@app.route('/')
def home():
    return "Hello from Flask on Render!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
