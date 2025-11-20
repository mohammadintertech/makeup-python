import subprocess
import sys



import io  
import os
import cv2
import numpy as np
import logging
import traceback
import gc
import threading
import time
import requests
from flask import Flask, request, send_file, jsonify
import mediapipe as mp

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# OPTIMIZED FACE MESH MANAGEMENT (BIGGEST PERFORMANCE GAIN)
# ============================================================================

mp_face_mesh = mp.solutions.face_mesh

class FaceMeshManager:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.3,  # Reduced for speed
            min_tracking_confidence=0.3
        )
        self.lock = threading.Lock()
        logger.info("✅ FaceMeshManager initialized with optimized settings")
    
    def process(self, image):
        """Fast face landmark detection using shared instance"""
        with self.lock:
            try:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                rgb_image.flags.writeable = False
                return self.face_mesh.process(rgb_image)
            except Exception as e:
                logger.error(f"Face detection error: {e}")
                return None

# Initialize shared face mesh instance
face_mesh_manager = FaceMeshManager()

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

MAX_IMAGE_SIZE = 1024  # Reduced from 2048 for faster processing
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
OPTIMIZED_SIZE = 768   # Target size for processing

# ============================================================================
# OPTIMIZED IMAGE PROCESSING UTILITIES
# ============================================================================

def optimize_image_size(image, target_size=OPTIMIZED_SIZE):
    """Downsample large images for faster processing"""
    h, w = image.shape[:2]
    current_max = max(h, w)
    
    if current_max <= target_size:
        return image
    
    scale = target_size / current_max
    new_width = int(w * scale)
    new_height = int(h * scale)
    
    optimized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    logger.info(f"📐 Image optimized: {w}x{h} -> {new_width}x{new_height}")
    return optimized

def timed_processor(func):
    """Decorator to log processing time"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"⏱️ {func.__name__} took {end_time - start_time:.3f}s")
        return result
    return wrapper

# ============================================================================
# OPTIMIZED MAKEUP PROCESSOR FUNCTIONS
# ============================================================================

@timed_processor
def apply_blusher_fast(image: np.ndarray, face_landmarks, blusher_color_rgb: list, intensity_factor: float = 0.5) -> np.ndarray:
    """Optimized blusher application using pre-detected landmarks"""
    try:
        blusher_color = np.array(blusher_color_rgb[::-1], dtype=np.float32) / 255.0
        h, w = image.shape[:2]
        image_float = image.astype(np.float32) / 255.0
        output = image_float.copy()
        
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
            mask = cv2.GaussianBlur(mask, (51, 51), 0)  # Reduced from 101
            return mask

        left_mask = create_circular_mask(left_center, circle_radius)
        right_mask = create_circular_mask(right_center, circle_radius)
        combined_mask = np.maximum(left_mask, right_mask)

        ycrcb_image = cv2.cvtColor(output, cv2.COLOR_BGR2YCrCb)
        ycrcb_blusher = cv2.cvtColor(blusher_color.reshape(1,1,3), cv2.COLOR_BGR2YCrCb)
        
        blended_ycrcb = ycrcb_image.copy()
        blended_ycrcb[..., 1] = ycrcb_image[..., 1] * (1 - intensity_factor * combined_mask*2) + \
                               ycrcb_blusher[0,0,1] * intensity_factor * combined_mask*2
        blended_ycrcb[..., 2] = ycrcb_image[..., 2] * (1 - intensity_factor * combined_mask*2) + \
                               ycrcb_blusher[0,0,2] * intensity_factor * combined_mask*2
        
        blended_bgr = cv2.cvtColor(blended_ycrcb, cv2.COLOR_YCrCb2BGR)
        output = output * (1 - combined_mask[..., np.newaxis]) + blended_bgr * combined_mask[..., np.newaxis]

        highlight_mask = cv2.GaussianBlur(combined_mask, (75, 75), 0) * 0.15  # Reduced from 151
        output = np.minimum(output + highlight_mask[..., np.newaxis] * np.array([0.1, 0.1, 0.1]), 1.0)
        return (output * 255).astype(np.uint8)
        
    except Exception as e:
        logger.error(f"Blusher error: {e}")
        return image

@timed_processor
def apply_eyeshadow_fast(image: np.ndarray, face_landmarks, eyeshadow_color_rgb: list, transparency_factor: float = 0.9) -> np.ndarray:
    """Optimized eyeshadow application using pre-detected landmarks"""
    try:
        shadow_color = eyeshadow_color_rgb[::-1]
        h, w = image.shape[:2]
        output = image.copy()

        # More focused eye landmarks for closer application
        LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Use fewer brow points to keep it closer to eyes
        LEFT_BROW_LOWER = [70, 63, 105]
        RIGHT_BROW_LOWER = [336, 296, 334]

        def get_landmark_points(indices):
            return np.array([(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in indices], dtype=np.int32)

        left_eye = get_landmark_points(LEFT_EYE)
        left_brow_lower = get_landmark_points(LEFT_BROW_LOWER)
        right_eye = get_landmark_points(RIGHT_EYE)
        right_brow_lower = get_landmark_points(RIGHT_BROW_LOWER)

        def create_eyeshadow_area_close_to_eye(eye_points, brow_lower_points):
            eye_upper = [point for point in eye_points if point[1] < np.mean(eye_points[:, 1])]
            
            adjusted_brow_points = []
            for point in brow_lower_points:
                new_y = point[1] + (np.mean(eye_points[:, 1]) - point[1]) * 0.6
                adjusted_brow_points.append([point[0], int(new_y)])
            
            all_points = np.vstack((eye_upper, adjusted_brow_points))
            return cv2.convexHull(all_points.astype(np.int32))

        left_area = create_eyeshadow_area_close_to_eye(left_eye, left_brow_lower)
        right_area = create_eyeshadow_area_close_to_eye(right_eye, right_brow_lower)
        shadow_overlay = np.zeros_like(image)

        for area in [left_area, right_area]:
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, area, 255)
            
            mask_blur = cv2.GaussianBlur(mask, (21, 21), 0)  # Reduced from 31
            mask_normalized = mask_blur / 255.0

            for c in range(3):
                shadow_overlay[:, :, c] = np.clip(
                    shadow_overlay[:, :, c] + mask_normalized * shadow_color[c] * transparency_factor * 1.3, 
                    0, 255
                ).astype(np.uint8)

        output = cv2.addWeighted(output, 1.0, shadow_overlay, 0.9, 0)
        return output
        
    except Exception as e:
        logger.error(f"Eyeshadow error: {e}")
        return image

@timed_processor
def apply_eyecolor_fast(image: np.ndarray, face_landmarks, r: int, g: int, b: int) -> np.ndarray:
    """Optimized eye color change using pre-detected landmarks"""
    try:
        h, w = image.shape[:2]
        base_color = (min(b * 0.7, 255), min(g * 0.7, 255), min(r * 0.7, 255))
        output = image.copy()
        overlay = np.zeros_like(image)
        iris_ranges = [range(468, 473), range(473, 478)]
        
        for iris_range in iris_ranges:
            iris_points = np.array([(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in iris_range])
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, cv2.convexHull(iris_points), 255)
            mask = cv2.GaussianBlur(mask, (15, 15), 0)  # Reduced from 27
            mask = mask.astype(float)/255
            
            for c in range(3):
                overlay[:,:,c] = np.where(mask > 0, (base_color[c] * 0.7 + image[:,:,c] * 0.3) * mask, overlay[:,:,c]).astype(np.uint8)
        
        output = cv2.addWeighted(overlay, 0.6, output, 0.9, 0)
        output = cv2.addWeighted(output, 1.1, np.zeros_like(output), 0, 5)
        return output
        
    except Exception as e:
        logger.error(f"Eyecolor error: {e}")
        return image

@timed_processor
def apply_eyeliner_fast(image: np.ndarray, face_landmarks, eyeliner_color_rgb: list, alpha: float = 0.7) -> np.ndarray:
    """Optimized eyeliner application using pre-detected landmarks"""
    try:
        eyeliner_color = eyeliner_color_rgb[::-1]
        h, w = image.shape[:2]
        overlay = image.copy()
        output = image.copy()

        LEFT_EYE_OUTER = 130
        RIGHT_EYE_OUTER = 359

        left_corner = (int(face_landmarks.landmark[LEFT_EYE_OUTER].x * w), int(face_landmarks.landmark[LEFT_EYE_OUTER].y * h))
        right_corner = (int(face_landmarks.landmark[RIGHT_EYE_OUTER].x * w), int(face_landmarks.landmark[RIGHT_EYE_OUTER].y * h))

        max_thickness = 6  # Reduced from 8
        min_thickness = 2
        eyeliner_length = int(w * 0.03)  # Reduced from 0.04

        def draw_smooth_eyeliner(start_point, direction='right', tilt_up=True):
            points = []
            for i in range(eyeliner_length):
                alpha = i / eyeliner_length
                x = start_point[0] + (i if direction == 'right' else -i)
                y = start_point[1] - (i * 0.2 if tilt_up else 0)
                points.append((int(x), int(y)))

            points = np.array(points, np.int32)
            points = points.reshape((-1, 1, 2))
            
            for i in range(1, len(points)):
                thickness = int(max_thickness * (1 - i / len(points)) + min_thickness * (i / len(points)))
                cv2.line(overlay, tuple(points[i-1][0]), tuple(points[i][0]), eyeliner_color, thickness, lineType=cv2.LINE_AA)

        draw_smooth_eyeliner(left_corner, direction='left', tilt_up=True)
        draw_smooth_eyeliner(right_corner, direction='right', tilt_up=True)

        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
        return output
        
    except Exception as e:
        logger.error(f"Eyeliner error: {e}")
        return image

# ============================================================================
# OPTIMIZED FOUNDATION SECTION
# ============================================================================

class FoundationApplier:
    def __init__(self):
        self.initialized = True
        logger.info("✅ Foundation processor initialized successfully")

    @timed_processor
    def apply_foundation_fast(self, image: np.ndarray, face_landmarks, foundation_rgb: list, intensity: float) -> np.ndarray:
        if not self.initialized:
            return image
            
        try:
            h, w = image.shape[:2]
            result = image.copy().astype(np.float32) / 255.0
            foundation_bgr = np.array(foundation_rgb[::-1], dtype=np.float32) / 255.0
            
            FACE_OVAL = [
                10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 
                397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 
                172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
            ]
            
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
                return image
                
            mask = np.zeros((h, w), dtype=np.float32)
            face_contour = np.array(face_points, dtype=np.int32)
            cv2.fillPoly(mask, [face_contour], 1.0)
            
            LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
            RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
            LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
            
            for eye_landmarks in [LEFT_EYE, RIGHT_EYE]:
                eye_points = []
                for landmark_id in eye_landmarks:
                    try:
                        landmark = face_landmarks.landmark[landmark_id]
                        px = int(landmark.x * w)
                        py = int(landmark.y * h)
                        eye_points.append([px, py])
                    except:
                        continue
                
                if len(eye_points) > 2:
                    eye_contour = np.array(eye_points, dtype=np.int32)
                    cv2.fillPoly(mask, [eye_contour], 0.0)
            
            lip_points = []
            for landmark_id in LIPS:
                try:
                    landmark = face_landmarks.landmark[landmark_id]
                    px = int(landmark.x * w)
                    py = int(landmark.y * h)
                    lip_points.append([px, py])
                except:
                    continue
            
            if len(lip_points) > 2:
                lip_contour = np.array(lip_points, dtype=np.int32)
                cv2.fillPoly(mask, [lip_contour], 0.0)
            
            mask = cv2.GaussianBlur(mask, (31, 31), 0)  # Reduced from 51
            
            mask_3d = np.stack([mask] * 3, axis=2)
            
            foundation_layer = np.ones_like(result)
            foundation_layer[:, :] = foundation_bgr
            
            adjusted_intensity = intensity * 0.4
            result = result * (1 - mask_3d * adjusted_intensity) + foundation_layer * mask_3d * adjusted_intensity
    
            return (np.clip(result, 0, 1) * 255).astype(np.uint8)
            
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

def apply_foundation_fast(image, face_landmarks, foundation_rgb, intensity):
    """Public interface for foundation application"""
    if not FOUNDATION_WORKING or foundation_applier is None:
        logger.warning("Foundation processor not available - returning original image")
        return image
    
    try:
        result = foundation_applier.apply_foundation_fast(image, face_landmarks, foundation_rgb, float(intensity))
        return result if result is not None else image
    except Exception as e:
        logger.error(f"Error applying foundation: {e}")
        return image

# ============================================================================
# OPTIMIZED LIPSTICK FUNCTION
# ============================================================================

@timed_processor
def apply_lipstick_fast(image: np.ndarray, face_landmarks, lip_color: list, intensity_factor: float, edge_width: int) -> np.ndarray:
    """Optimized lipstick application using pre-detected landmarks"""
    try:
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            return image
        if len(lip_color) != 3 or not all(0 <= c <= 255 for c in lip_color):
            return image
        
        image_float = image.astype(np.float32) / 255.0
        output = image_float.copy()
        h, w = image.shape[:2]
        
        LIPS_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 76, 62, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87]
        
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

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_width*2+1, edge_width*2+1))
        mask = cv2.dilate(mask, kernel)
        mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), edge_width/2)
        mask = mask / 255.0

        target_color = np.array(lip_color[::-1], dtype=np.float32) / 255.0
        colored_lips = np.zeros_like(output)
        colored_lips[:,:,:] = target_color

        alpha = mask * intensity_factor
        alpha = alpha[..., np.newaxis]
        output = output * (1 - alpha) + colored_lips * alpha

        return (np.clip(output, 0, 1) * 255).astype(np.uint8)
    except Exception as e:
        logger.error(f"Lipstick error: {e}")
        return image

# ============================================================================
# BATCH MAKEUP PROCESSOR (MAIN OPTIMIZATION)
# ============================================================================

def batch_apply_makeup(image, makeup_config):
    """
    Apply all makeup effects with single face detection
    This is the main performance optimization
    """
    start_time = time.time()
    
    try:
        # Single face detection for all effects
        results = face_mesh_manager.process(image)
        if not results or not results.multi_face_landmarks:
            logger.warning("No face detected in batch processing")
            return image
        
        face_landmarks = results.multi_face_landmarks[0]
        processed = image.copy()
        
        # Apply effects in sequence using same landmarks
        if makeup_config.get('foundation'):
            config = makeup_config['foundation']
            processed = apply_foundation_fast(processed, face_landmarks, config['color'], config['intensity'])
        
        if makeup_config.get('blusher'):
            config = makeup_config['blusher']
            processed = apply_blusher_fast(processed, face_landmarks, config['color'], config['intensity'])
        
        if makeup_config.get('lipstick'):
            config = makeup_config['lipstick']
            processed = apply_lipstick_fast(processed, face_landmarks, config['color'], config['intensity'], config['edge'])
        
        if makeup_config.get('eyeliner'):
            config = makeup_config['eyeliner']
            processed = apply_eyeliner_fast(processed, face_landmarks, config['color'])
            
        if makeup_config.get('eyecolor'):
            config = makeup_config['eyecolor']
            processed = apply_eyecolor_fast(processed, face_landmarks, config['r'], config['g'], config['b'])
            
        if makeup_config.get('eyeshadow'):
            config = makeup_config['eyeshadow']
            processed = apply_eyeshadow_fast(processed, face_landmarks, config['color'], config['intensity'])
        
        total_time = time.time() - start_time
        logger.info(f"🎯 Batch processing completed in {total_time:.3f}s - Effects: {list(makeup_config.keys())}")
        
        return processed
        
    except Exception as e:
        logger.error(f"Batch makeup error: {e}")
        return image

# ============================================================================
# FLASK APPLICATION (MAINTAINING EXACT API INTERFACE)
# ============================================================================

app = Flask(__name__)
PROCESSORS_LOADED = True

@app.route("/test")
def test():
    status = "✅ Running" if PROCESSORS_LOADED else "⚠️ Running (processors not loaded)"
    return f"Makeup API is {status} properly!"

@app.route("/ping")
def ping():
    return jsonify({"status": "pong", "processors_loaded": PROCESSORS_LOADED, "timestamp": time.time()})

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "processors_loaded": PROCESSORS_LOADED,
        "max_image_size": MAX_IMAGE_SIZE,
        "supported_formats": list(SUPPORTED_FORMATS),
        "optimized_size": OPTIMIZED_SIZE
    })

def validate_image_file(file):
    if not file or not file.filename:
        return False, "No file provided"
    ext = os.path.splitext(file.filename.lower())[1]
    if ext not in SUPPORTED_FORMATS:
        return False, f"Unsupported format. Supported: {', '.join(SUPPORTED_FORMATS)}"
    return True, "Valid"

@app.route("/apply", methods=['POST'])
def apply_makeup():
    try:
        logger.info("🚀 Received makeup application request")
        
        if not PROCESSORS_LOADED:
            return jsonify({"error": "Makeup processors not loaded properly"}), 500
        
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        is_valid, message = validate_image_file(file)
        if not is_valid:
            return jsonify({"error": message}), 400
        
        try:
            contents = file.read()
            if len(contents) == 0:
                return jsonify({"error": "Empty file"}), 400
            
            np_arr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if image is None:
                return jsonify({"error": "Invalid or corrupted image"}), 400
            
            if len(image.shape) != 3 or image.shape[2] != 3:
                return jsonify({"error": "Image must be a color image (3 channels)"}), 400
                
            # OPTIMIZATION: Downsample large images
            original_size = max(image.shape)
            if original_size > MAX_IMAGE_SIZE:
                image = optimize_image_size(image, MAX_IMAGE_SIZE)
            else:
                # Still optimize to target size for faster processing
                image = optimize_image_size(image, OPTIMIZED_SIZE)
                
        except Exception as e:
            return jsonify({"error": f"Image processing failed: {str(e)}"}), 400

        # Collect all makeup configurations (maintaining your exact form field names)
        form = request.form
        makeup_config = {}
        
        # Foundation
        if form.get('enable_foundation') == 'true':
            r, g, b = form.get('foundation_r'), form.get('foundation_g'), form.get('foundation_b')
            intensity = form.get('foundation_intensity')
            if all([r, g, b, intensity]):
                makeup_config['foundation'] = {
                    'color': [int(r), int(g), int(b)],
                    'intensity': float(intensity)
                }

        # Blusher
        if form.get('enable_blusher') == 'true':
            r, g, b = form.get('blusher_r'), form.get('blusher_g'), form.get('blusher_b')
            intensity = form.get('blusher_intensity')
            if all([r, g, b, intensity]):
                makeup_config['blusher'] = {
                    'color': [int(r), int(g), int(b)],
                    'intensity': float(intensity)
                }

        # Lipstick
        if form.get('enable_lipstick') == 'true':
            r, g, b = form.get('lipstick_r'), form.get('lipstick_g'), form.get('lipstick_b')
            intensity = form.get('lipstick_intensity')
            edge = form.get('lipstick_edge')
            if all([r, g, b, intensity, edge]):
                makeup_config['lipstick'] = {
                    'color': [int(r), int(g), int(b)],
                    'intensity': float(intensity),
                    'edge': int(edge)
                }

        # Eyeliner
        if form.get('enable_eyeliner') == 'true':
            r, g, b = form.get('eyeliner_r'), form.get('eyeliner_g'), form.get('eyeliner_b')
            if all([r, g, b]):
                makeup_config['eyeliner'] = {
                    'color': [int(r), int(g), int(b)]
                }

        # Eye color
        if form.get('enable_eyecolor') == 'true':
            r, g, b = form.get('eyecolor_r'), form.get('eyecolor_g'), form.get('eyecolor_b')
            if all([r, g, b]):
                makeup_config['eyecolor'] = {
                    'r': int(r), 'g': int(g), 'b': int(b)
                }

        # Eyeshadow
        if form.get('enable_eyeshadow') == 'true':
            r, g, b = form.get('eyeshadow_r'), form.get('eyeshadow_g'), form.get('eyeshadow_b')
            intensity = form.get('eyeshadow_intensity')
            if all([r, g, b, intensity]):
                makeup_config['eyeshadow'] = {
                    'color': [int(r), int(g), int(b)],
                    'intensity': float(intensity)
                }
        
        # OPTIMIZATION: Use batch processing with single face detection
        logger.info(f"🎨 Applying {len(makeup_config)} effects: {list(makeup_config.keys())}")
        processed = batch_apply_makeup(image, makeup_config)
        
        try:
            encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 8]  # Slightly higher compression for speed
            success, img_encoded = cv2.imencode(".png", processed, encode_param)
            
            if not success:
                return jsonify({"error": "Failed to encode result image"}), 500
            
            response = send_file(
                io.BytesIO(img_encoded.tobytes()),
                mimetype="image/png",
                as_attachment=False,
                download_name="makeup_result.png"
            )
            
            response.headers['X-Processing-Steps'] = ','.join(makeup_config.keys()) if makeup_config else 'none'
            
            # Force cleanup
            del image, processed, img_encoded, contents, np_arr
            gc.collect()
            
            logger.info("✅ Request completed successfully")
            return response
            
        except Exception as e:
            return jsonify({"error": f"Failed to encode result: {str(e)}"}), 500
            
    except Exception as e:
        logger.error(f"❌ Unexpected server error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large"}), 413

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

# ============================================================================
# AUTO REQUEST FUNCTIONALITY (OPTIMIZED)
# ============================================================================

def auto_ping_server():
    """Automatically ping the server every 5 minutes to keep it awake"""
    def ping_loop():
        target_url = "https://makeup-python-5-6oru.onrender.com/ping"
        while True:
            try:
                response = requests.get(target_url, timeout=10)
                logger.info(f"🔄 Auto-ping sent to {target_url} - Status: {response.status_code}")
            except requests.exceptions.RequestException as e:
                logger.warning(f"⚠️ Auto-ping failed: {e}")
            except Exception as e:
                logger.error(f"❌ Unexpected error in auto-ping: {e}")
            
            # Wait 300 seconds (5 minutes) before next ping
            time.sleep(300)
    
    # Start the auto-ping thread
    ping_thread = threading.Thread(target=ping_loop, daemon=True)
    ping_thread.start()
    logger.info("✅ Auto-ping service started (requests every 300 seconds)")

# Start the auto-ping service when the application starts
auto_ping_server()

# ============================================================================
# START APPLICATION FOR RENDER.COM
# ============================================================================

if __name__ == "__main__":
    logger.info("🚀 Starting OPTIMIZED Makeup API on Render.com")
    print(f"\n🎨 OPTIMIZED Makeup API is running!")
    print(f"⚡ Performance optimizations:")
    print(f"   • Single face detection per request")
    print(f"   • Shared MediaPipe instance")
    print(f"   • Optimized image size: {OPTIMIZED_SIZE}px")
    print(f"   • Reduced blur and processing parameters")
    print(f"🔍 Test URL: http://0.0.0.0:{os.environ.get('PORT', 5000)}/test")
    print(f"❤️ Health Check: http://0.0.0.0:{os.environ.get('PORT', 5000)}/health")
    print("⚡ Ready to process FAST makeup requests!\n")
    
    # Run Flask directly (Render will handle the port)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)