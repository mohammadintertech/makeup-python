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

# Shadow color palette (BGR format)
SHADOW_COLORS = {
    '1': {'name': 'black', 'value': (40, 40, 40)},
    '2': {'name': 'brown', 'value': (50, 80, 120)},
    '3': {'name': 'taupe', 'value': (80, 90, 110)},
    '4': {'name': 'cool gray', 'value': (100, 100, 120)},
    '5': {'name': 'warm gray', 'value': (80, 100, 110)},
    '6': {'name': 'contour brown', 'value': (60, 90, 130)}
}

# Shadow parameters
SHADOW_WIDTH = 20            # Width of shadow effect
SHADOW_OPACITY = 0.6         # Intensity of shadow
BLUR_INTENSITY = (51, 51)    # Gaussian blur kernel size

def choose_shadow_color():
    """Display color options and return selected color"""
    print("\nAvailable Shadow Colors:")
    for key, color in SHADOW_COLORS.items():
        print(f"{key}. {color['name'].title()}")
    
    while True:
        choice = input("\nEnter the number of your preferred shadow color (1-6): ")
        if choice in SHADOW_COLORS:
            return SHADOW_COLORS[choice]['value'], SHADOW_COLORS[choice]['name']
        print("Invalid choice. Please enter a number between 1 and 6.")

def create_shadow_effect(image, line_points, shadow_width, shadow_color, opacity, blur_intensity):
    """Apply professional shadow effect along a line"""
    shadow_layer = np.zeros_like(image, dtype=np.float32)
    line_points = np.array(line_points)
    
    # Create polygon around the line with tapered ends
    line_vec = line_points[1] - line_points[0]
    perpendicular = np.array([-line_vec[1], line_vec[0]])
    perpendicular = perpendicular / np.linalg.norm(perpendicular)
    
    # Tapered width (narrower at center, wider at ends)
    width_start = shadow_width * 0.6
    width_end = shadow_width
    
    shadow_poly = np.array([
        line_points[0] + perpendicular * width_start,
        line_points[0] - perpendicular * width_start,
        line_points[1] - perpendicular * width_end,
        line_points[1] + perpendicular * width_end
    ], dtype=np.int32)
    
    # Draw and blur the shadow
    cv2.fillPoly(shadow_layer, [shadow_poly], shadow_color)
    shadow_layer = cv2.GaussianBlur(shadow_layer, blur_intensity, 0)
    
    # Apply with opacity
    return cv2.addWeighted(image, 1.0, shadow_layer.astype(np.uint8), opacity, 0)

def draw_nose_contour(landmarks, image, shadow_color_value, shadow_color_name='taupe'):
    h, w = image.shape[:2]
    
    # Landmark indices
    LEFT_NOSTRIL = 49
    RIGHT_NOSTRIL = 279
    LEFT_EYE_INNER = 133
    RIGHT_EYE_INNER = 362
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_OUTER = 263
    NOSE_TIP = 4
    
    # Get coordinates
    left_nostril = np.array([int(landmarks.landmark[LEFT_NOSTRIL].x * w), 
                           int(landmarks.landmark[LEFT_NOSTRIL].y * h)])
    right_nostril = np.array([int(landmarks.landmark[RIGHT_NOSTRIL].x * w), 
                            int(landmarks.landmark[RIGHT_NOSTRIL].y * h)])
    nose_tip = np.array([int(landmarks.landmark[NOSE_TIP].x * w),
                        int(landmarks.landmark[NOSE_TIP].y * h)])
    
    # Calculate center point between eyes
    left_eye_inner = np.array([int(landmarks.landmark[LEFT_EYE_INNER].x * w),
                             int(landmarks.landmark[LEFT_EYE_INNER].y * h)])
    right_eye_inner = np.array([int(landmarks.landmark[RIGHT_EYE_INNER].x * w),
                              int(landmarks.landmark[RIGHT_EYE_INNER].y * h)])
    
    center_eyes = (left_eye_inner + right_eye_inner) // 2
    
    # Get outer eye points for top line (now much closer to center)
    left_eye_outer = np.array([int(landmarks.landmark[LEFT_EYE_OUTER].x * w),
                             int(landmarks.landmark[LEFT_EYE_OUTER].y * h)])
    right_eye_outer = np.array([int(landmarks.landmark[RIGHT_EYE_OUTER].x * w),
                              int(landmarks.landmark[RIGHT_EYE_OUTER].y * h)])
    
    # Create points for the top line (now only 10% of the distance from center to outer eye)
    top_left = center_eyes + (left_eye_outer - center_eyes) * 0.1
    top_right = center_eyes + (right_eye_outer - center_eyes) * 0.1
    
    # Adjust the nostril points to be closer to the nose tip (bring them inward)
    nostril_adjustment = 0.3  # 30% closer to nose tip
    adjusted_left_nostril = left_nostril + (nose_tip - left_nostril) * nostril_adjustment
    adjusted_right_nostril = right_nostril + (nose_tip - right_nostril) * nostril_adjustment
    
    # Create base image
    output = image.copy()
    
    # Apply shadows only to left and right sides
    # Left side (from top left to adjusted left nostril)
    output = create_shadow_effect(output, 
                                [top_left, adjusted_left_nostril],
                                SHADOW_WIDTH, 
                                shadow_color_value, 
                                SHADOW_OPACITY,
                                BLUR_INTENSITY)
    
    # Right side (from top right to adjusted right nostril)
    output = create_shadow_effect(output, 
                                [top_right, adjusted_right_nostril],
                                SHADOW_WIDTH, 
                                shadow_color_value, 
                                SHADOW_OPACITY,
                                BLUR_INTENSITY)
    
    # Draw subtle quadrilateral outline
    quad_points = np.array([adjusted_left_nostril, adjusted_right_nostril, top_right, top_left], np.int32)
    cv2.polylines(output, [quad_points], isClosed=True, 
                 color=(100, 100, 100), thickness=1, lineType=cv2.LINE_AA)
    
    # Mark key points for visualization
    for point, name in [(adjusted_left_nostril, "L Nostril"), 
                       (adjusted_right_nostril, "R Nostril"),
                       (top_left, "L Top"),
                       (top_right, "R Top")]:
        cv2.circle(output, tuple(point.astype(int)), 3, (0, 0, 255), -1)
        cv2.putText(output, name, tuple(point.astype(int) + np.array([5,5])), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
    
    # Display selected color
    cv2.putText(output, f"Color: {shadow_color_name}", (20, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return output

try:
    # Download and process image
    response = requests.get(image_url)
    response.raise_for_status()
    img_array = np.array(bytearray(response.content), dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    # Let user choose shadow color
    selected_color, color_name = choose_shadow_color()
    
    # Process face mesh
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if results.multi_face_landmarks:
        # Create contour with selected color
        final_output = draw_nose_contour(results.multi_face_landmarks[0], image, selected_color, color_name)
        
        # Display final result
        print(f"\nEnhanced Nose Contouring with {color_name} color:")
        cv2_imshow(final_output)
    else:
        print("No face detected")

except Exception as e:
    print(f"Error: {e}")

finally:
    face_mesh.close()