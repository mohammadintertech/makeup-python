import cv2
import mediapipe as mp
import numpy as np
import requests
from google.colab.patches import cv2_imshow

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True,
                                  refine_landmarks=True,
                                  max_num_faces=1,
                                  min_detection_confidence=0.5)

# Image URL
image_url = 'https://i.pinimg.com/474x/ea/b3/a3/eab3a381fab1bd6b4a4e61ebabc73b63.jpg'

# Shadow parameters
SHADOW_COLOR = (40, 40, 80)  # BGR format (dark blue-gray)
SHADOW_WIDTH = 15            # Width of shadow effect
SHADOW_OPACITY = 0.7         # Intensity of shadow

def create_shadow_effect(image, line_points, shadow_width, shadow_color, opacity):
    """Apply shadow effect along a line"""
    shadow_layer = np.zeros_like(image, dtype=np.float32)
    
    # Create polygon around the line
    line_vec = line_points[1] - line_points[0]
    perpendicular = np.array([-line_vec[1], line_vec[0]])
    perpendicular = perpendicular / np.linalg.norm(perpendicular) * shadow_width
    
    shadow_poly = np.array([
        line_points[0] + perpendicular,
        line_points[0] - perpendicular,
        line_points[1] - perpendicular,
        line_points[1] + perpendicular
    ], dtype=np.int32)
    
    # Draw and blur the shadow
    cv2.fillPoly(shadow_layer, [shadow_poly], shadow_color)
    shadow_layer = cv2.GaussianBlur(shadow_layer, (51, 51), 0)
    
    # Apply to original image
    return cv2.addWeighted(image, 1.0, shadow_layer.astype(np.uint8), opacity, 0)

def draw_nose_triangle(landmarks, image):
    h, w = image.shape[:2]
    
    # Landmark indices
    LEFT_NOSTRIL = 49
    RIGHT_NOSTRIL = 279
    LEFT_EYE_INNER = 133
    RIGHT_EYE_INNER = 362
    
    # Get coordinates
    left_nostril = np.array([int(landmarks.landmark[LEFT_NOSTRIL].x * w), 
                           int(landmarks.landmark[LEFT_NOSTRIL].y * h)])
    right_nostril = np.array([int(landmarks.landmark[RIGHT_NOSTRIL].x * w), 
                            int(landmarks.landmark[RIGHT_NOSTRIL].y * h)])
    
    # Calculate center point between eyes
    left_eye = np.array([int(landmarks.landmark[LEFT_EYE_INNER].x * w),
                       int(landmarks.landmark[LEFT_EYE_INNER].y * h)])
    right_eye = np.array([int(landmarks.landmark[RIGHT_EYE_INNER].x * w),
                        int(landmarks.landmark[RIGHT_EYE_INNER].y * h)])
    
    center_eyes = (left_eye + right_eye) // 2
    
    # Create base image with triangle
    output = image.copy()
    triangle_points = np.array([left_nostril, right_nostril, center_eyes], np.int32)
    cv2.polylines(output, [triangle_points], isClosed=True, 
                 color=(0, 255, 0), thickness=2)
    
    # Apply shadow to left line (center_eyes to left_nostril)
    output = create_shadow_effect(output, 
                                np.array([center_eyes, left_nostril]),
                                SHADOW_WIDTH, 
                                SHADOW_COLOR, 
                                SHADOW_OPACITY)
    
    # Apply shadow to right line (center_eyes to right_nostril)
    output = create_shadow_effect(output, 
                                np.array([center_eyes, right_nostril]),
                                SHADOW_WIDTH, 
                                SHADOW_COLOR, 
                                SHADOW_OPACITY)
    
    # Draw small circles at each vertex point
    for point in [left_nostril, right_nostril, center_eyes]:
        cv2.circle(output, tuple(point), 3, (0, 0, 255), -1)
    
    return output

try:
    # Download and process image
    response = requests.get(image_url)
    response.raise_for_status()
    img_array = np.array(bytearray(response.content), dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    # Process face mesh
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if results.multi_face_landmarks:
        output = draw_nose_triangle(results.multi_face_landmarks[0], image.copy())
        
        print("Nose Triangle with Shadow Effects:")
        cv2_imshow(output)
    else:
        print("No face detected")

except Exception as e:
    print(f"Error: {e}")

finally:
    face_mesh.close()