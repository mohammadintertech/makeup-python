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

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            contents = file.read()
            np_arr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if image is None:
                logger.error("Failed to decode image")
                return jsonify({"error": "Invalid image"}), 400
        except Exception as e:
            logger.error(f"Image processing error: {str(e)}")
            return jsonify({"error": f"Image processing failed: {str(e)}"}), 400

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
            
            # Blusher
            if form.get('enable_blusher') == 'true':
                logger.info("Applying blusher")
                r, g, b = form.get('blusher_r'), form.get('blusher_g'), form.get('blusher_b')
                intensity = form.get('blusher_intensity')
                if all([r, g, b, intensity]):
                    processed = apply_blusher(processed, [int(r), int(g), int(b)], float(intensity))
                    processing_steps.append("blusher")
            
            # Lipstick
            if form.get('enable_lipstick') == 'true':
                logger.info("Applying lipstick")
                r, g, b = form.get('lipstick_r'), form.get('lipstick_g'), form.get('lipstick_b')
                intensity = form.get('lipstick_intensity')
                edge = form.get('lipstick_edge')
                if all([r, g, b, intensity, edge]):
                    processed = apply_lipstick(processed, [int(r), int(g), int(b)], float(intensity), int(edge))
                    processing_steps.append("lipstick")
            
            # Eyeliner
            if form.get('enable_eyeliner') == 'true':
                logger.info("Applying eyeliner")
                r, g, b = form.get('eyeliner_r'), form.get('eyeliner_g'), form.get('eyeliner_b')
                if all([r, g, b]):
                    processed = apply_eyeliner(processed, [int(r), int(g), int(b)])
                    processing_steps.append("eyeliner")
            
            # Eye color
            if form.get('enable_eyecolor') == 'true':
                logger.info("Applying eye color")
                r, g, b = form.get('eyecolor_r'), form.get('eyecolor_g'), form.get('eyecolor_b')
                if all([r, g, b]):
                    processed = apply_eyecolor(processed, int(r), int(g), int(b))
                    processing_steps.append("eyecolor")
            
            # Eyeshadow
            if form.get('enable_eyeshadow') == 'true':
                logger.info("Applying eyeshadow")
                r, g, b = form.get('eyeshadow_r'), form.get('eyeshadow_g'), form.get('eyeshadow_b')
                intensity = form.get('eyeshadow_intensity')
                if all([r, g, b, intensity]):
                    processed = apply_eyeshadow(processed, [int(r), int(g), int(b)], float(intensity))
                    processing_steps.append("eyeshadow")
            
            logger.info(f"Completed processing steps: {', '.join(processing_steps)}")
            
            # Encode the final image
            try:
                _, img_encoded = cv2.imencode(".png", processed)
                logger.info("Image successfully processed and encoded")
                
                return send_file(
                    io.BytesIO(img_encoded.tobytes()),
                    mimetype="image/png",
                    as_attachment=False,
                    download_name="result.png"
                )
            except Exception as e:
                logger.error(f"Image encoding error: {str(e)}")
                return jsonify({"error": f"Failed to encode final image: {str(e)}"}), 500
                
        except Exception as e:
            logger.error(f"Processing error at steps {processing_steps}: {str(e)}\n{traceback.format_exc()}")
            return jsonify({
                "error": f"Makeup application failed at step: {processing_steps[-1] if processing_steps else 'unknown'}",
                "details": str(e),
                "traceback": traceback.format_exc()
            }), 500

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            "error": "Unexpected server error",
            "details": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/')
def home():
    return "Hello from Flask on Render!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
