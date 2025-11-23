# app.py - COMBINED PROXY & MAKEUP PROCESSING
from flask import Flask, request, jsonify, Response, send_file
from flask_cors import CORS
import requests
import io
import logging
import os
import cv2
import numpy as np
import traceback
import gc
import threading
import time
import base64
import mediapipe as mp

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# API URLs (only Oasis remains since we handle makeup locally now)
OASIS_API_URL = "https://www.oasisonline.ps/WS"

# Track if we've already pinged
_ping_executed = False

# ============================================================================
# OPTIMIZE OPENCV FOR RENDER ENVIRONMENT
# ============================================================================

# Limit OpenCV threads to prevent thread creation errors
cv2.setNumThreads(1)
os.environ['OPENCV_OPENCL_RUNTIME'] = '0'
os.environ['OPENCV_OPENCL_DEVICE'] = ':0'

# ============================================================================
# MEDIAPIPE INITIALIZATION & MAKEUP PROCESSING FUNCTIONS
# ============================================================================

# Initialize MediaPipe Face Mesh with optimized settings for Render
mp_face_mesh = mp.solutions.face_mesh
face_mesh_lock = threading.Lock()

# Configuration constants
MAX_IMAGE_SIZE = 800  # Further reduced for better performance
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
PROCESSORS_LOADED = True

# Global face mesh instance to avoid repeated initialization
face_mesh = None

def get_face_mesh():
    """Get or create face mesh instance with optimized settings"""
    global face_mesh
    if face_mesh is None:
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            refine_landmarks=False,  # Disable refine for better performance
            max_num_faces=1,
            min_detection_confidence=0.5
        )
    return face_mesh

# ============================================================================
# OPTIMIZED MAKEUP PROCESSOR FUNCTIONS FOR RENDER
# ============================================================================

def apply_blusher(image: np.ndarray, blusher_color_rgb: list, intensity_factor: float = 0.5) -> np.ndarray:
    """Apply blusher to an image with given RGB color"""
    try:
        blusher_color = np.array(blusher_color_rgb[::-1], dtype=np.float32) / 255.0
        h, w = image.shape[:2]
        
        # Use simpler approach without complex processing
        with face_mesh_lock:
            face_mesh_instance = get_face_mesh()
            results = face_mesh_instance.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                
            if results.multi_face_landmarks:
                output = image.copy().astype(np.float32) / 255.0
                for face_landmarks in results.multi_face_landmarks:
                    def get_landmark_point(index):
                        return int(face_landmarks.landmark[index].x * w), int(face_landmarks.landmark[index].y * h)
                    
                    left_cheek_ref = get_landmark_point(123)
                    right_cheek_ref = get_landmark_point(352)
                    circle_radius = int(w * 0.08)  # Further reduced radius
                    
                    # Simple circular masks
                    left_center = (left_cheek_ref[0] + int(w * 0.03), left_cheek_ref[1])
                    right_center = (right_cheek_ref[0] - int(w * 0.03), right_cheek_ref[1])

                    def create_simple_mask(center, radius):
                        mask = np.zeros((h, w), dtype=np.float32)
                        cv2.circle(mask, center, radius, 1, -1)
                        mask = cv2.GaussianBlur(mask, (25, 25), 0)  # Minimal blur
                        return mask

                    left_mask = create_simple_mask(left_center, circle_radius)
                    right_mask = create_simple_mask(right_center, circle_radius)
                    combined_mask = np.maximum(left_mask, right_mask)

                    # Simple color blending
                    for c in range(3):
                        output[:, :, c] = output[:, :, c] * (1 - combined_mask * intensity_factor) + \
                                         blusher_color[c] * combined_mask * intensity_factor

                    return (np.clip(output, 0, 1) * 255).astype(np.uint8)
            else:
                return image
    except Exception as e:
        logger.error(f"Blusher error: {e}")
        return image

def apply_eyeshadow(image: np.ndarray, eyeshadow_color_rgb: list, transparency_factor: float = 0.9) -> np.ndarray:
    """Apply eyeshadow to an image with given RGB color"""
    try:
        shadow_color = np.array(eyeshadow_color_rgb[::-1], dtype=np.float32) / 255.0
        h, w = image.shape[:2]
        
        with face_mesh_lock:
            face_mesh_instance = get_face_mesh()
            results = face_mesh_instance.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                
            if results.multi_face_landmarks:
                output = image.copy().astype(np.float32) / 255.0
                for face_landmarks in results.multi_face_landmarks:
                    # Simplified eye landmarks
                    LEFT_EYE = [33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161]
                    RIGHT_EYE = [362, 263, 373, 374, 380, 381, 382, 385, 386, 387, 388, 390]

                    def get_landmark_points(indices):
                        return np.array([(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in indices], dtype=np.int32)

                    left_eye = get_landmark_points(LEFT_EYE)
                    right_eye = get_landmark_points(RIGHT_EYE)

                    def create_simple_eye_mask(eye_points):
                        hull = cv2.convexHull(eye_points)
                        mask = np.zeros((h, w), dtype=np.float32)
                        cv2.fillConvexPoly(mask, hull, 1.0)
                        mask = cv2.GaussianBlur(mask, (15, 15), 0)  # Minimal blur
                        return mask

                    left_mask = create_simple_eye_mask(left_eye)
                    right_mask = create_simple_eye_mask(right_eye)
                    combined_mask = np.maximum(left_mask, right_mask)

                    # Simple color application
                    for c in range(3):
                        output[:, :, c] = output[:, :, c] * (1 - combined_mask * transparency_factor) + \
                                         shadow_color[c] * combined_mask * transparency_factor

                return (np.clip(output, 0, 1) * 255).astype(np.uint8)
            else:
                return image
    except Exception as e:
        logger.error(f"Eyeshadow error: {e}")
        return image

def apply_eyecolor(image: np.ndarray, r: int, g: int, b: int) -> np.ndarray:
    """Apply eye color change to irises"""
    try:
        base_color = (min(b * 0.7, 255), min(g * 0.7, 255), min(r * 0.7, 255))
        h, w = image.shape[:2]
        
        with face_mesh_lock:
            face_mesh_instance = get_face_mesh()
            results = face_mesh_instance.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                
            if results.multi_face_landmarks:
                output = image.copy()
                for face_landmarks in results.multi_face_landmarks:
                    # Simple iris detection
                    LEFT_IRIS = [468, 469, 470, 471, 472]
                    RIGHT_IRIS = [473, 474, 475, 476, 477]

                    def get_iris_mask(iris_indices):
                        points = []
                        for idx in iris_indices:
                            landmark = face_landmarks.landmark[idx]
                            px = int(landmark.x * w)
                            py = int(landmark.y * h)
                            points.append([px, py])
                        
                        if len(points) < 3:
                            return None
                            
                        hull = cv2.convexHull(np.array(points, dtype=np.int32))
                        mask = np.zeros((h, w), dtype=np.uint8)
                        cv2.fillConvexPoly(mask, hull, 255)
                        mask = cv2.GaussianBlur(mask, (9, 9), 0)  # Minimal blur
                        return mask.astype(float) / 255.0

                    left_mask = get_iris_mask(LEFT_IRIS)
                    right_mask = get_iris_mask(RIGHT_IRIS)

                    if left_mask is not None:
                        for c in range(3):
                            output[:, :, c] = output[:, :, c] * (1 - left_mask) + base_color[c] * left_mask * 0.7
                    
                    if right_mask is not None:
                        for c in range(3):
                            output[:, :, c] = output[:, :, c] * (1 - right_mask) + base_color[c] * right_mask * 0.7

                return output
            else:
                return image
    except Exception as e:
        logger.error(f"Eyecolor error: {e}")
        return image

def apply_eyeliner(image: np.ndarray, eyeliner_color_rgb: list) -> np.ndarray:
    """Apply eyeliner to an image with given RGB color"""
    try:
        eyeliner_color = eyeliner_color_rgb[::-1]
        
        with face_mesh_lock:
            face_mesh_instance = get_face_mesh()
            results = face_mesh_instance.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                
            if results.multi_face_landmarks:
                output = image.copy()
                for face_landmarks in results.multi_face_landmarks:
                    # Simple eyeliner - just draw lines at eye corners
                    LEFT_EYE_CORNER = 33
                    RIGHT_EYE_CORNER = 263

                    left_corner = (int(face_landmarks.landmark[LEFT_EYE_CORNER].x * w), 
                                 int(face_landmarks.landmark[LEFT_EYE_CORNER].y * h))
                    right_corner = (int(face_landmarks.landmark[RIGHT_EYE_CORNER].x * w), 
                                  int(face_landmarks.landmark[RIGHT_EYE_CORNER].y * h))

                    # Simple short lines
                    line_length = int(w * 0.02)
                    thickness = 2

                    # Left eye line
                    left_end = (left_corner[0] - line_length, left_corner[1] - line_length // 2)
                    cv2.line(output, left_corner, left_end, eyeliner_color, thickness, cv2.LINE_AA)

                    # Right eye line
                    right_end = (right_corner[0] + line_length, right_corner[1] - line_length // 2)
                    cv2.line(output, right_corner, right_end, eyeliner_color, thickness, cv2.LINE_AA)

                return output
            else:
                return image
    except Exception as e:
        logger.error(f"Eyeliner error: {e}")
        return image

# ============================================================================
# FOUNDATION SECTION
# ============================================================================

class FoundationApplier:
    def __init__(self):
        self.initialized = True
        logger.info("✅ Foundation processor initialized successfully")

    def apply_foundation(self, image: np.ndarray, foundation_rgb: list, intensity: float) -> np.ndarray:
        if not self.initialized:
            return image
            
        try:
            h, w = image.shape[:2]
            foundation_bgr = np.array(foundation_rgb[::-1], dtype=np.uint8)
            
            with face_mesh_lock:
                face_mesh_instance = get_face_mesh()
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh_instance.process(rgb_image)

                if not results.multi_face_landmarks:
                    logger.info("No face detected for foundation")
                    return image

                logger.info("Face detected for foundation application")
                
                for face_landmarks in results.multi_face_landmarks:
                    # Simple face oval
                    FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
                    
                    face_points = []
                    for landmark_id in FACE_OVAL:
                        try:
                            landmark = face_landmarks.landmark[landmark_id]
                            px = int(landmark.x * w)
                            py = int(landmark.y * h)
                            face_points.append([px, py])
                        except:
                            continue
                    
                    if len(face_points) < 3:
                        continue
                        
                    mask = np.zeros((h, w), dtype=np.float32)
                    face_contour = np.array(face_points, dtype=np.int32)
                    cv2.fillPoly(mask, [face_contour], 1.0)
                    
                    # Simple blur
                    mask = cv2.GaussianBlur(mask, (21, 21), 0)
                    
                    # Apply foundation with simple blending
                    foundation_layer = np.full_like(image, foundation_bgr, dtype=np.uint8)
                    output = cv2.addWeighted(image, 1 - intensity * 0.3, foundation_layer, intensity * 0.3, 0)
                    
                    # Blend with original based on mask
                    result = image.copy()
                    mask_3d = np.stack([mask] * 3, axis=2)
                    result = result * (1 - mask_3d) + output * mask_3d
                    
                    return result.astype(np.uint8)
            
            return image
            
        except Exception as e:
            logger.error(f"Foundation application error: {e}")
            return image

# Initialize the foundation applier
try:
    foundation_applier = FoundationApplier()
    FOUNDATION_WORKING = True
    logger.info("✅ Foundation processor initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize foundation processor: {e}")
    foundation_applier = None
    FOUNDATION_WORKING = False

def apply_foundation(image, foundation_rgb, intensity):
    """Public interface for foundation application"""
    if not FOUNDATION_WORKING or foundation_applier is None:
        logger.warning("Foundation processor not available - returning original image")
        return image
    
    try:
        logger.info(f"Applying foundation with color {foundation_rgb} and intensity {intensity}")
        result = foundation_applier.apply_foundation(image, foundation_rgb, float(intensity))
        return result if result is not None else image
    except Exception as e:
        logger.error(f"Error applying foundation: {e}")
        return image

# ============================================================================
# LIPSTICK FUNCTION
# ============================================================================

def apply_lipstick(image: np.ndarray, lip_color: list, intensity_factor: float, edge_width: int) -> np.ndarray:
    """Apply lipstick to an image with improved handling"""
    try:
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            return image
        if len(lip_color) != 3 or not all(0 <= c <= 255 for c in lip_color):
            return image
        
        h, w = image.shape[:2]
        
        with face_mesh_lock:
            face_mesh_instance = get_face_mesh()
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh_instance.process(rgb_image)

            if not results.multi_face_landmarks:
                return image

            face_landmarks = results.multi_face_landmarks[0]
            
            # Simplified lip points
            LIPS_OUTER = [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
            
            lip_points = []
            for i in LIPS_OUTER:
                try:
                    landmark = face_landmarks.landmark[i]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    lip_points.append([x, y])
                except:
                    continue
            
            if len(lip_points) < 3:
                return image
            
            lip_points = np.array(lip_points, dtype=np.int32)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [lip_points], 255)

            # Simple dilation and blur
            kernel = np.ones((edge_width, edge_width), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            mask = cv2.GaussianBlur(mask.astype(np.float32), (9, 9), 0)
            mask = mask / 255.0

            # Simple color application
            target_color = np.array(lip_color[::-1], dtype=np.uint8)
            colored_lips = np.full_like(image, target_color, dtype=np.uint8)
            
            output = cv2.addWeighted(image, 1 - intensity_factor, colored_lips, intensity_factor, 0)
            
            # Blend with mask
            mask_3d = np.stack([mask] * 3, axis=2)
            result = image * (1 - mask_3d) + output * mask_3d
            
            return result.astype(np.uint8)
    except Exception as e:
        logger.error(f"Lipstick error: {e}")
        return image

# ============================================================================
# IMAGE PROCESSING UTILITIES
# ============================================================================

def validate_image_file(file):
    if not file or not file.filename:
        return False, "No file provided"
    ext = os.path.splitext(file.filename.lower())[1]
    if ext not in SUPPORTED_FORMATS:
        return False, f"Unsupported format. Supported: {', '.join(SUPPORTED_FORMATS)}"
    return True, "Valid"

def safe_process_image(image, processor_func, *args, **kwargs):
    try:
        return processor_func(image, *args, **kwargs)
    except Exception as e:
        logger.error(f"Processor error: {e}")
        return image

# ============================================================================
# LOCAL MAKEUP ENDPOINTS
# ============================================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "processors_loaded": PROCESSORS_LOADED,
        "max_image_size": MAX_IMAGE_SIZE,
        "supported_formats": list(SUPPORTED_FORMATS),
        "message": "Local Makeup API is running"
    })

@app.route('/apply', methods=['POST'])
def apply_makeup():
    """
    Local makeup application endpoint - processes images directly using MediaPipe
    """
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
        
        try:
            # Read file content
            file_content = file.read()
            if len(file_content) == 0:
                return jsonify({"error": "Empty file"}), 400
            
            np_arr = np.frombuffer(file_content, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if image is None:
                return jsonify({"error": "Invalid or corrupted image"}), 400
            
            if len(image.shape) != 3 or image.shape[2] != 3:
                return jsonify({"error": "Image must be a color image (3 channels)"}), 400
                
            original_size = max(image.shape)
            if original_size > MAX_IMAGE_SIZE:
                scale = MAX_IMAGE_SIZE / original_size
                new_width = int(image.shape[1] * scale)
                new_height = int(image.shape[0] * scale)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                
        except Exception as e:
            return jsonify({"error": f"Image processing failed: {str(e)}"}), 400

        processed = image.copy()
        form = request.form
        processing_steps = []
        
        try:
            # Process makeup steps with proper parameter handling
            if form.get('enable_foundation') == 'true':
                r, g, b = form.get('foundation_r'), form.get('foundation_g'), form.get('foundation_b')
                intensity = form.get('foundation_intensity')
                if all([r, g, b, intensity]):
                    processed = safe_process_image(processed, apply_foundation, 
                                                 [int(r), int(g), int(b)], float(intensity))
                    processing_steps.append("foundation")
                    gc.collect()

            if form.get('enable_blusher') == 'true':
                r, g, b = form.get('blusher_r'), form.get('blusher_g'), form.get('blusher_b')
                intensity = form.get('blusher_intensity')
                if all([r, g, b, intensity]):
                    processed = safe_process_image(processed, apply_blusher, 
                                                 [int(r), int(g), int(b)], float(intensity))
                    processing_steps.append("blusher")
                    gc.collect()

            if form.get('enable_lipstick') == 'true':
                r, g, b = form.get('lipstick_r'), form.get('lipstick_g'), form.get('lipstick_b')
                intensity = form.get('lipstick_intensity')
                edge = form.get('lipstick_edge')
                if all([r, g, b, intensity, edge]):
                    processed = safe_process_image(processed, apply_lipstick, 
                                                 [int(r), int(g), int(b)], float(intensity), int(edge))
                    processing_steps.append("lipstick")
                    gc.collect()

            if form.get('enable_eyeliner') == 'true':
                r, g, b = form.get('eyeliner_r'), form.get('eyeliner_g'), form.get('eyeliner_b')
                if all([r, g, b]):
                    processed = safe_process_image(processed, apply_eyeliner, 
                                                 [int(r), int(g), int(b)])
                    processing_steps.append("eyeliner")
                    gc.collect()

            if form.get('enable_eyecolor') == 'true':
                r, g, b = form.get('eyecolor_r'), form.get('eyecolor_g'), form.get('eyecolor_b')
                if all([r, g, b]):
                    processed = safe_process_image(processed, apply_eyecolor, 
                                                 int(r), int(g), int(b))
                    processing_steps.append("eyecolor")
                    gc.collect()

            if form.get('enable_eyeshadow') == 'true':
                r, g, b = form.get('eyeshadow_r'), form.get('eyeshadow_g'), form.get('eyeshadow_b')
                intensity = form.get('eyeshadow_intensity')
                if all([r, g, b, intensity]):
                    processed = safe_process_image(processed, apply_eyeshadow, 
                                                 [int(r), int(g), int(b)], float(intensity))
                    processing_steps.append("eyeshadow")
                    gc.collect()
            
            try:
                # Encode result
                success, img_encoded = cv2.imencode(".png", processed)
                
                if not success:
                    return jsonify({"error": "Failed to encode result image"}), 500
                
                # FIXED: Return image data directly as Response to avoid fileno issues
                response = Response(
                    img_encoded.tobytes(),
                    mimetype='image/png',
                    headers={
                        'Content-Type': 'image/png',
                        'Content-Disposition': 'inline; filename="makeup_result.png"',
                        'X-Processing-Steps': ','.join(processing_steps) if processing_steps else 'none'
                    }
                )
                
                # Clean up
                del image, processed, img_encoded, file_content, np_arr
                gc.collect()
                
                return response
                
            except Exception as e:
                return jsonify({"error": f"Failed to encode result: {str(e)}"}), 500
            
        except Exception as e:
            return jsonify({"error": f"Processing failed: {str(e)}"}), 500

    except Exception as e:
        logger.error(f"Unexpected server error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/apply_base64', methods=['POST'])
def apply_makeup_base64():
    """
    Alternative endpoint that returns base64 encoded image + JSON
    """
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Use the same processing logic as /apply
        response = apply_makeup()
        
        # If response is a file response, convert to base64
        if hasattr(response, 'data'):
            # This means we got an image response
            image_data = response.data
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            return jsonify({
                "success": True,
                "message": "Makeup applied successfully",
                "image_data": image_base64,
                "format": "base64",
                "size": len(image_data)
            })
        else:
            # This means we got an error response
            return response
        
    except Exception as e:
        logger.error(f"Base64 proxy error: {str(e)}")
        return jsonify({"error": f"Proxy error: {str(e)}"}), 500

# ============================================================================
# OASIS ONLINE API PROXY (UNCHANGED - keep all the same endpoints)
# ============================================================================

@app.route('/oasis/<path:endpoint>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def oasis_proxy(endpoint):
    """Generic proxy for Oasis Online API"""
    try:
        target_url = f"{OASIS_API_URL}/{endpoint}"
        headers = {'User-Agent': 'MakeupApp/1.0', 'Accept': 'application/json'}
        
        if request.method == 'GET':
            response = requests.get(target_url, params=request.args, headers=headers, timeout=30)
        elif request.method == 'POST':
            response = requests.post(target_url, json=request.get_json(), data=request.form, headers=headers, timeout=30)
        elif request.method == 'PUT':
            response = requests.put(target_url, json=request.get_json(), headers=headers, timeout=30)
        elif request.method == 'DELETE':
            response = requests.delete(target_url, headers=headers, timeout=30)
        else:
            return jsonify({"error": "Method not allowed"}), 405
        
        try:
            return jsonify(response.json()), response.status_code
        except:
            return Response(response.text, status=response.status_code, mimetype='text/plain')
            
    except requests.exceptions.Timeout:
        return jsonify({"error": "Oasis API timeout"}), 504
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Cannot connect to Oasis API"}), 503
    except Exception as e:
        logger.error(f"Oasis proxy error: {str(e)}")
        return jsonify({"error": f"Oasis proxy error: {str(e)}"}), 500

@app.route('/getProductsAPP', methods=['GET'])
def get_products_app():
    """Proxy for makeup products API"""
    try:
        target_url = f"{OASIS_API_URL}/getProductsAPP"
        params = {
            'limit': request.args.get('limit', '100'),
            'page': request.args.get('page', '1'),
            'c_id': request.args.get('c_id', '136,72,78,84,87,137,139,138'),
            'makeup': request.args.get('makeup', '1')
        }
        
        for key in request.args:
            if key not in ['limit', 'page', 'c_id', 'makeup']:
                params[key] = request.args.get(key)
        
        response = requests.get(target_url, params=params, headers={'User-Agent': 'MakeupApp/1.0'}, timeout=30)
        
        if response.status_code == 200:
            try:
                data = response.json()
                return jsonify(data)
            except ValueError:
                return jsonify({"error": "Invalid JSON response from API"}), 500
        else:
            return Response(response.text, status=response.status_code, mimetype='text/plain')
            
    except requests.exceptions.Timeout:
        return jsonify({"error": "Oasis API timeout"}), 504
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Cannot connect to Oasis API"}), 503
    except Exception as e:
        logger.error(f"Makeup products proxy error: {str(e)}")
        return jsonify({"error": f"Makeup products proxy error: {str(e)}"}), 500

@app.route('/getMakeupProducts', methods=['GET'])
def get_makeup_products():
    """Convenience endpoint for makeup products"""
    try:
        params = {
            'limit': request.args.get('limit', '50'),
            'page': request.args.get('page', '1'),
            'c_id': '136,72,78,84,87,137,139,138',
            'makeup': '1'
        }
        
        request.args = type('Args', (), {**params, 'get': lambda self, key, default=None: params.get(key, default)})()
        return get_products_app()
        
    except Exception as e:
        logger.error(f"Makeup products convenience endpoint error: {str(e)}")
        return jsonify({"error": f"Error fetching makeup products: {str(e)}"}), 500

@app.route('/getProducts', methods=['GET'])
def get_products():
    return oasis_proxy('getProducts')

@app.route('/getCategories', methods=['GET'])
def get_categories():
    return oasis_proxy('getCategories')

@app.route('/getProductDetails', methods=['GET'])
def get_product_details():
    return oasis_proxy('getProductDetails')

@app.route('/test', methods=['GET'])
def test():
    return jsonify({
        "message": "Combined Makeup API Server is running", 
        "status": "healthy",
        "makeup_processing": "local",
        "oasis_proxy": "active"
    })

@app.route('/')
def index():
    return jsonify({
        "message": "Combined Makeup API Server",
        "status": "running",
        "endpoints": {
            "makeup": "/apply, /apply_base64",
            "products": "/getProductsAPP, /getMakeupProducts",
            "oasis": "/oasis/*",
            "health": "/health, /test"
        }
    })

@app.route('/ping')
def ping():
    return jsonify({"status": "pong", "processors_loaded": PROCESSORS_LOADED, "timestamp": time.time()})

@app.before_request
def before_first_request():
    global _ping_executed
    if not _ping_executed:
        logger.info("Starting up Combined Makeup API Server - First request")
        _ping_executed = True

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large"}), 413

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)

application = app