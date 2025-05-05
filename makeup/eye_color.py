import cv2
import mediapipe as mp
import numpy as np
import requests
from google.colab.patches import cv2_imshow

# Pupil color selection
print("Choose pupil color:")
print("1. Deep Blue")
print("2. Forest Green")
print("3. Amber Brown")
print("4. Custom Color")
choice = input("Enter choice (1-4): ").strip()

# Natural-looking color presets (BGR format)
if choice == '1':
    base_color = (130, 90, 20)   # Deep Blue
elif choice == '2':
    base_color = (40, 90, 50)    # Forest Green
elif choice == '3':
    base_color = (15, 70, 120)   # Amber Brown
elif choice == '4':
    r = min(int(input("Red (0-255): ")) * 0.7, 255)
    g = min(int(input("Green (0-255): ")) * 0.7, 255)
    b = min(int(input("Blue (0-255): ")) * 0.7, 255)
    base_color = (int(b), int(g), int(r))
else:
    print("Defaulting to Deep Blue")
    base_color = (130, 90, 20)

# Load image
image_url = 'https://i.pinimg.com/474x/ea/b3/a3/eab3a381fab1bd6b4a4e61ebabc73b63.jpg'
response = requests.get(image_url)
img_array = np.array(bytearray(response.content), dtype=np.uint8)
image = cv2.imdecode(img_array, -1)
h, w = image.shape[:2]

# Initialize Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, 
                                 refine_landmarks=True,
                                 max_num_faces=1)

# Process image
results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

if results.multi_face_landmarks:
    output = image.copy()
    
    for face_landmarks in results.multi_face_landmarks:
        # Create empty overlay
        overlay = np.zeros_like(image)
        
        # Process BOTH irises (left and right)
        iris_ranges = [range(468, 473), range(473, 478)]  # Left then right iris
        
        for iris_range in iris_ranges:
            iris_points = np.array([(int(lm.x * w), int(lm.y * h)) 
                                  for lm in [face_landmarks.landmark[i] for i in iris_range]])
            
            # Create smooth mask with two layers
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, cv2.convexHull(iris_points), 255)
            
            # Apply Gaussian blur for soft edges
            mask = cv2.GaussianBlur(mask, (27, 27), 0)
            mask = mask.astype(float)/255  # Normalize to 0-1
            
            # Apply color while preserving some original eye texture
            for c in range(3):
                overlay[:,:,c] = np.where(mask > 0,
                                        (base_color[c] * 0.7 + image[:,:,c] * 0.3) * mask,
                                        overlay[:,:,c]).astype(np.uint8)
        
        # Natural blending with original image
        output = cv2.addWeighted(overlay, 0.6, output, 0.9, 0)
        
        # Slightly enhance contrast
        output = cv2.addWeighted(output, 1.1, np.zeros_like(output), 0, 5)

    cv2_imshow(output)
else:
    print("No face detected")