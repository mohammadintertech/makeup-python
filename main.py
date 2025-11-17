#!/usr/bin/env python3
"""
Complete Makeup API Server - All Processors in One File
Install: pip install flask flask-cors numpy opencv-python mediapipe
"""

import sys
import os
import cv2
import numpy as np
import io
import traceback
import logging
import gc
import time
import mediapipe as mp
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS, cross_origin

# Set UTF-8 encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# Initialize Flask app with CORS
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
MAX_IMAGE_SIZE = 2048
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh_lock = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    refine_landmarks=True,
    max_num_faces=1,
    min_detection_confidence=0.5
)

# ==================== MAKEUP PROCESSORS ====================

def hex_to_rgb(hex_color):
    """Convert hex color to RGB list"""
    hex_color = hex_color.lstrip('#')
    return [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]

def apply_lipstick(image: np.ndarray, lip_color: str, intensity_factor: float = 0.5, edge_width: int = 1) -> np.ndarray:
    """Apply lipstick to image"""
    try:
        rgb_color = hex_to_rgb(lip_color)
        image_float = image.astype(np.float32) / 255.0
        output = image_float.copy()
        h, w = image.shape[:2]
        
        results = face_mesh_lock.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return image
            
        face_landmarks = results.multi_face_landmarks[0]
        LIPS_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
        
        lip_points = []
        for i in LIPS_OUTER:
            landmark = face_landmarks.landmark[i]
            x, y = int(landmark.x * w), int(landmark.y * h)
            lip_points.append([x, y])
        
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(lip_points, dtype=np.int32)], 255)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_width*2+1, edge_width*2+1))
        mask = cv2.dilate(mask, kernel)
        mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), edge_width/2) / 255.0
        
        target_color = np.array(rgb_color[::-1], dtype=np.float32) / 255.0
        colored_lips = np.zeros_like(output)
        colored_lips[:,:,:] = target_color
        
        alpha = mask[..., np.newaxis] * intensity_factor
        output = output * (1 - alpha) + colored_lips * alpha
        
        return (np.clip(output, 0, 1) * 255).astype(np.uint8)
    except Exception as e:
        logger.error(f"Lipstick error: {e}")
        return image

def apply_eyeliner(image: np.ndarray, eyeliner_color: str) -> np.ndarray:
    """Apply eyeliner to image"""
    try:
        rgb_color = hex_to_rgb(eyeliner_color)
        eyeliner_color_bgr = rgb_color[::-1]
        
        results = face_mesh_lock.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return image
            
        h, w = image.shape[:2]
        overlay = image.copy()
        output = image.copy()
        
        for face_landmarks in results.multi_face_landmarks:
            LEFT_EYE_OUTER = 130
            RIGHT_EYE_OUTER = 359

            left_corner = (int(face_landmarks.landmark[LEFT_EYE_OUTER].x * w), 
                          int(face_landmarks.landmark[LEFT_EYE_OUTER].y * h))
            right_corner = (int(face_landmarks.landmark[RIGHT_EYE_OUTER].x * w), 
                           int(face_landmarks.landmark[RIGHT_EYE_OUTER].y * h))

            max_thickness, min_thickness = 8, 2
            eyeliner_length = int(w * 0.04)

            def draw_smooth_eyeliner(start_point, direction='right'):
                points = []
                for i in range(eyeliner_length):
                    x = start_point[0] + (i if direction == 'right' else -i)
                    y = start_point[1] - (i * 0.2)
                    points.append((int(x), int(y)))
                points = np.array(points, np.int32).reshape((-1, 1, 2))
                
                for i in range(1, len(points)):
                    thickness = int(max_thickness * (1 - i/len(points)) + min_thickness * (i/len(points)))
                    cv2.line(overlay, tuple(points[i-1][0]), tuple(points[i][0]), 
                            eyeliner_color_bgr, thickness, lineType=cv2.LINE_AA)

            draw_smooth_eyeliner(left_corner, 'left')
            draw_smooth_eyeliner(right_corner, 'right')

        cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)
        return output
    except Exception as e:
        logger.error(f"Eyeliner error: {e}")
        return image

def apply_blusher(image: np.ndarray, blusher_color: str, intensity_factor: float = 0.5) -> np.ndarray:
    """Apply blusher to image"""
    try:
        rgb_color = hex_to_rgb(blusher_color)
        blusher_color_bgr = np.array(rgb_color[::-1], dtype=np.float32) / 255.0
        
        results = face_mesh_lock.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return image
            
        h, w = image.shape[:2]
        image_float = image.astype(np.float32) / 255.0
        output = image_float.copy()

        for face_landmarks in results.multi_face_landmarks:
            def get_landmark_point(index):
                return int(face_landmarks.landmark[index].x * w), int(face_landmarks.landmark[index].y * h)
            
            left_cheek_ref = get_landmark_point(123)
            right_cheek_ref = get_landmark_point(352)
            
            circle_radius = int(w * 0.15)
            left_center = (left_cheek_ref[0] + int(w * 0.06), left_cheek_ref[1] - int(h * 0.01))
            right_center = (right_cheek_ref[0] - int(w * 0.06), right_cheek_ref[1] - int(h * 0.01))

            def create_circular_mask(center, radius):
                mask = np.zeros((h, w), dtype=np.float32)
                cv2.circle(mask, center, radius, 1, -1)
                dist = cv2.distanceTransform((mask * 255).astype(np.uint8), cv2.DIST_L2, 5)
                dist = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
                mask = np.power(dist, 1.5)
                mask = cv2.GaussianBlur(mask, (101, 101), 0)
                return mask

            left_mask = create_circular_mask(left_center, circle_radius)
            right_mask = create_circular_mask(right_center, circle_radius)
            combined_mask = np.maximum(left_mask, right_mask)

            ycrcb_image = cv2.cvtColor(output, cv2.COLOR_BGR2YCrCb)
            ycrcb_blusher = cv2.cvtColor(blusher_color_bgr.reshape(1,1,3), cv2.COLOR_BGR2YCrCb)
            
            blended_ycrcb = ycrcb_image.copy()
            blended_ycrcb[..., 1] = ycrcb_image[..., 1] * (1 - intensity_factor * combined_mask*2) + \
                                   ycrcb_blusher[0,0,1] * intensity_factor * combined_mask*2
            blended_ycrcb[..., 2] = ycrcb_image[..., 2] * (1 - intensity_factor * combined_mask*2) + \
                                   ycrcb_blusher[0,0,2] * intensity_factor * combined_mask*2
            
            blended_bgr = cv2.cvtColor(blended_ycrcb, cv2.COLOR_YCrCb2BGR)
            output = output * (1 - combined_mask[..., np.newaxis]) + blended_bgr * combined_mask[..., np.newaxis]

            highlight_mask = cv2.GaussianBlur(combined_mask, (151, 151), 0) * 0.15
            output = np.minimum(output + highlight_mask[..., np.newaxis] * np.array([0.1, 0.1, 0.1]), 1.0)

        return (output * 255).astype(np.uint8)
    except Exception as e:
        logger.error(f"Blusher error: {e}")
        return image

def apply_foundation(image: np.ndarray, foundation_color: str, intensity: float) -> np.ndarray:
    """Apply foundation to image"""
    try:
        rgb_color = hex_to_rgb(foundation_color)
        foundation_bgr = np.array(rgb_color[::-1], dtype=np.float32) / 255.0
        
        # Simple face detection using skin tone areas
        h, w = image.shape[:2]
        image_float = image.astype(np.float32) / 255.0
        output = image_float.copy()
        
        # Create elliptical face mask (simplified)
        mask = np.zeros((h, w), dtype=np.float32)
        center = (w//2, h//3)
        axes = (int(w * 0.4), int(h * 0.3))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)
        
        # Smooth mask
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
        
        # Apply foundation color
        foundation_layer = np.zeros_like(output)
        foundation_layer[:, :] = foundation_bgr
        
        # Blend with original
        mask_3d = np.stack([mask] * 3, axis=2)
        result = output * (1 - mask_3d * intensity) + foundation_layer * mask_3d * intensity
        
        return (np.clip(result, 0, 1) * 255).astype(np.uint8)
    except Exception as e:
        logger.error(f"Foundation error: {e}")
        return image

def apply_eyecolor(image: np.ndarray, r: int, g: int, b: int) -> np.ndarray:
    """Apply eye color to image"""
    try:
        results = face_mesh_lock.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return image
            
        h, w = image.shape[:2]
        output = image.copy()
        base_color = (min(b * 0.7, 255), min(g * 0.7, 255), min(r * 0.7, 255))
        
        for face_landmarks in results.multi_face_landmarks:
            overlay = np.zeros_like(image)
            iris_ranges = [range(468, 473), range(473, 478)]
            
            for iris_range in iris_ranges:
                iris_points = np.array([(int(face_landmarks.landmark[i].x * w), 
                                       int(face_landmarks.landmark[i].y * h)) for i in iris_range])
                
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillConvexPoly(mask, cv2.convexHull(iris_points), 255)
                mask = cv2.GaussianBlur(mask, (27, 27), 0).astype(float)/255
                
                for c in range(3):
                    overlay[:,:,c] = np.where(mask > 0,
                                            (base_color[c] * 0.7 + image[:,:,c] * 0.3) * mask,
                                            overlay[:,:,c]).astype(np.uint8)
            
            output = cv2.addWeighted(overlay, 0.6, output, 0.9, 0)
            output = cv2.addWeighted(output, 1.1, np.zeros_like(output), 0, 5)

        return output
    except Exception as e:
        logger.error(f"Eye color error: {e}")
        return image

def apply_eyeshadow(image: np.ndarray, eyeshadow_color: str, intensity: float = 0.7) -> np.ndarray:
    """Apply eyeshadow to image"""
    try:
        rgb_color = hex_to_rgb(eyeshadow_color)
        shadow_color = rgb_color[::-1]
        
        results = face_mesh_lock.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return image
            
        h, w = image.shape[:2]
        output = image.copy()

        for face_landmarks in results.multi_face_landmarks:
            LEFT_EYE = [33, 246, 161, 160, 159, 158, 157, 173, 133]
            LEFT_BROW = [70, 63, 105, 66, 107, 55, 65, 52]
            RIGHT_EYE = [362, 398, 384, 385, 386, 387, 388, 466, 263]
            RIGHT_BROW = [336, 296, 334, 293, 300, 276, 283, 282]

            def get_landmark_points(indices):
                return np.array([(int(face_landmarks.landmark[i].x * w), 
                                int(face_landmarks.landmark[i].y * h)) for i in indices], dtype=np.int32)

            left_eye = get_landmark_points(LEFT_EYE)
            left_brow = get_landmark_points(LEFT_BROW)
            right_eye = get_landmark_points(RIGHT_EYE)
            right_brow = get_landmark_points(RIGHT_BROW)

            def create_eyeshadow_area(eye_points, brow_points):
                lower_brow_y = min(brow_points, key=lambda x: x[1])[1] + 5
                adjusted_brow_points = [point for point in brow_points if point[1] > lower_brow_y]
                all_points = np.vstack((eye_points, adjusted_brow_points))
                return cv2.convexHull(all_points)

            left_area = create_eyeshadow_area(left_eye, left_brow)
            right_area = create_eyeshadow_area(right_eye, right_brow)

            shadow_overlay = np.zeros_like(image)
            for area in [left_area, right_area]:
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillConvexPoly(mask, area, 255)
                mask_blur = cv2.GaussianBlur(mask, (51, 51), 0) / 255.0

                for c in range(3):
                    shadow_overlay[:, :, c] = np.clip(
                        shadow_overlay[:, :, c] + mask_blur * shadow_color[c] * intensity,
                        0, 255
                    ).astype(np.uint8)

            output = cv2.addWeighted(output, 1.0, shadow_overlay, 0.7, 0)

        return output
    except Exception as e:
        logger.error(f"Eyeshadow error: {e}")
        return image

# ==================== FLASK ENDPOINTS ====================

@app.route("/")
def home():
    return jsonify({
        "message": "Makeup API Server is Running",
        "status": "active",
        "endpoints": {
            "test": "/test - Basic functionality test",
            "health": "/health - Detailed health check",
            "ping": "/ping - Quick status check",
            "apply": "/apply - Apply makeup (POST)"
        },
        "timestamp": time.time()
    })

@app.route("/test")
def test():
    test_results = {
        "server_status": "Running",
        "flask_import": "Success",
        "numpy_import": "Success",
        "opencv_import": "Success",
        "mediapipe_import": "Success",
        "timestamp": time.time(),
        "python_version": sys.version
    }
    
    try:
        test_array = np.array([1, 2, 3])
        test_results["numpy_test"] = f"Working (sum: {test_array.sum()})"
    except Exception as e:
        test_results["numpy_test"] = f"Failed: {e}"
    
    try:
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        success, encoded = cv2.imencode(".png", test_image)
        test_results["opencv_test"] = f"Working (encode: {success})"
    except Exception as e:
        test_results["opencv_test"] = f"Failed: {e}"
    
    return jsonify(test_results)

@app.route("/ping")
def ping():
    return jsonify({"status": "pong", "timestamp": time.time()})

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "max_image_size": MAX_IMAGE_SIZE,
        "supported_formats": list(SUPPORTED_FORMATS),
        "timestamp": time.time()
    })

def validate_image_file(file):
    if not file or not file.filename:
        return False, "No file provided"
    ext = os.path.splitext(file.filename.lower())[1]
    if ext not in SUPPORTED_FORMATS:
        return False, f"Unsupported format. Supported: {', '.join(SUPPORTED_FORMATS)}"
    return True, "Valid"

@app.route("/apply", methods=['POST', 'OPTIONS'])
@cross_origin()
def apply_makeup():
    if request.method == 'OPTIONS':
        return jsonify({"status": "preflight"}), 200
    
    try:
        logger.info("Received makeup application request")
        
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        is_valid, message = validate_image_file(file)
        if not is_valid:
            return jsonify({"error": message}), 400
        
        contents = file.read()
        if len(contents) == 0:
            return jsonify({"error": "Empty file"}), 400
        
        np_arr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"error": "Invalid or corrupted image"}), 400

        processed = image.copy()
        form = request.form
        
        try:
            if form.get('lipstick_color'):
                color = form.get('lipstick_color', '#FF0000')
                intensity = float(form.get('lipstick_intensity', 0.5))
                edge = int(form.get('lipstick_edge', 1))
                processed = apply_lipstick(processed, color, intensity, edge)
            
            if form.get('eyeliner_color'):
                color = form.get('eyeliner_color', '#000000')
                processed = apply_eyeliner(processed, color)
            
            if form.get('blusher_color'):
                color = form.get('blusher_color', '#FF6B9D')
                intensity = float(form.get('blusher_intensity', 0.3))
                processed = apply_blusher(processed, color, intensity)
            
            if form.get('foundation_color'):
                color = form.get('foundation_color', '#F5D6C6')
                intensity = float(form.get('foundation_intensity', 0.2))
                processed = apply_foundation(processed, color, intensity)
            
            if form.get('eyecolor_r'):
                r = int(form.get('eyecolor_r', 0))
                g = int(form.get('eyecolor_g', 0))
                b = int(form.get('eyecolor_b', 255))
                processed = apply_eyecolor(processed, r, g, b)
            
            if form.get('eyeshadow_color'):
                color = form.get('eyeshadow_color', '#8B4513')
                intensity = float(form.get('eyeshadow_intensity', 0.4))
                processed = apply_eyeshadow(processed, color, intensity)
                
        except Exception as e:
            logger.error(f"Makeup application error: {e}")
            return jsonify({"error": f"Makeup application failed: {str(e)}"}), 400
        
        success, img_encoded = cv2.imencode(".png", processed, [int(cv2.IMWRITE_PNG_COMPRESSION), 6])
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
        gc.collect()

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed"}), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

# WSGI entry point
application = app

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Starting Complete Makeup API Server...")
    print("="*50)
    print(f"Local URL: http://127.0.0.1:5000")
    print(f"Test URL: http://127.0.0.1:5000/test")
    print("="*50 + "\n")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        print(f"Failed to start server: {e}")
        sys.exit(1)
