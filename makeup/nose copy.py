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

def draw_nose_triangle(landmarks, image):
    h, w = image.shape[:2]
    
    # Landmark indices
    LEFT_NOSTRIL = 49
    RIGHT_NOSTRIL = 279
    LEFT_EYE_INNER = 133  # Inner corner of left eye
    RIGHT_EYE_INNER = 362 # Inner corner of right eye
    
    # Get coordinates
    left_nostril = (int(landmarks.landmark[LEFT_NOSTRIL].x * w), 
                    int(landmarks.landmark[LEFT_NOSTRIL].y * h))
    right_nostril = (int(landmarks.landmark[RIGHT_NOSTRIL].x * w), 
                     int(landmarks.landmark[RIGHT_NOSTRIL].y * h))
    
    # Calculate center point between eyes
    left_eye = (int(landmarks.landmark[LEFT_EYE_INNER].x * w),
                int(landmarks.landmark[LEFT_EYE_INNER].y * h))
    right_eye = (int(landmarks.landmark[RIGHT_EYE_INNER].x * w),
                 int(landmarks.landmark[RIGHT_EYE_INNER].y * h))
    
    center_eyes = ((left_eye[0] + right_eye[0]) // 2,
                   (left_eye[1] + right_eye[1]) // 2)
    
    # Draw the triangle
    triangle_points = np.array([left_nostril, right_nostril, center_eyes], np.int32)
    cv2.polylines(image, [triangle_points], isClosed=True, 
                 color=(0, 255, 0), thickness=2)
    
    # Draw small circles at each vertex point
    for point in [left_nostril, right_nostril, center_eyes]:
        cv2.circle(image, point, 3, (0, 0, 255), -1)
    
    # Label the points
    cv2.putText(image, "Left Nostril", (left_nostril[0]-50, left_nostril[1]), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
    cv2.putText(image, "Right Nostril", (right_nostril[0], right_nostril[1]), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
    cv2.putText(image, "Center Eyes", (center_eyes[0]-40, center_eyes[1]-10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
    
    return image

try:
    # Download and process image
    response = requests.get(image_url)
    response.raise_for_status()
    img_array = np.array(bytearray(response.content), dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    # Process face mesh
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if results.multi_face_landmarks:
        output = image.copy()
        output = draw_nose_triangle(results.multi_face_landmarks[0], output)
        
        print("Nose Triangle with Center Eyes Point:")
        cv2_imshow(output)
    else:
        print("No face detected")

except Exception as e:
    print(f"Error: {e}")

finally:
    face_mesh.close()