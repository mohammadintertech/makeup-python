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

# ======= COLOR SELECTION =======
print("=== Eyeshadow Color Choices ===")
print("1. Smoky Gray")
print("2. Rose Gold")
print("3. Earthy Brown")
print("4. Custom Color")

shadow_choice = input("Enter choice (1-4): ").strip()

# Eye shadow color presets (BGR format)
if shadow_choice == '1':
    shadow_color = (100, 100, 100)  # Smoky Gray
elif shadow_choice == '2':
    shadow_color = (80, 150, 200)   # Rose Gold
elif shadow_choice == '3':
    shadow_color = (40, 70, 140)    # Earthy Brown
elif shadow_choice == '4':
    r = min(int(input("Red (0-255): ")), 255)
    g = min(int(input("Green (0-255): ")), 255)
    b = min(int(input("Blue (0-255): ")), 255)
    shadow_color = (b, g, r)
else:
    print("Invalid choice. Defaulting to Smoky Gray.")
    shadow_color = (100, 100, 100)

# ======= INPUT TRANSPARENCY =======
transparency_factor = float(input("Enter the transparency factor (0.1 - 1.0): ").strip())
# Ensure the transparency factor is within valid range
transparency_factor = max(0.1, min(transparency_factor, 1.0))

# ======= LOAD IMAGE =======
image_url = 'https://i.pinimg.com/474x/ea/b3/a3/eab3a381fab1bd6b4a4e61ebabc73b63.jpg'
try:
    response = requests.get(image_url)
    response.raise_for_status()
    img_array = np.array(bytearray(response.content), dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image")
    h, w = image.shape[:2]
except Exception as e:
    print(f"Error loading image: {e}")
    exit()

# ======= PROCESS IMAGE =======
results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

if results.multi_face_landmarks:
    output = image.copy()

    for face_landmarks in results.multi_face_landmarks:
        # Define landmark groups for eyes and brows
        LEFT_EYE = [33, 246, 161, 160, 159, 158, 157, 173, 133]
        LEFT_BROW = [70, 63, 105, 66, 107, 55, 65, 52]
        
        RIGHT_EYE = [362, 398, 384, 385, 386, 387, 388, 466, 263]
        RIGHT_BROW = [336, 296, 334, 293, 300, 276, 283, 282]

        # Function to get points
        def get_landmark_points(indices):
            return np.array([
                (int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h))
                for i in indices
            ], dtype=np.int32)

        left_eye = get_landmark_points(LEFT_EYE)
        left_brow = get_landmark_points(LEFT_BROW)
        right_eye = get_landmark_points(RIGHT_EYE)
        right_brow = get_landmark_points(RIGHT_BROW)

        # Ensure the shadow only applies between the eye and brow, avoiding the brow itself
        # Define the space between the eye and brow
        def create_eyeshadow_area(eye_points, brow_points):
            # Only select the area between the eye and brow by selecting only the middle points
            # and excluding the brow region above the eye
            lower_brow_y = min(brow_points, key=lambda x: x[1])[1] + 5  # Adjust slightly below the brow

            # Masking points: this will create a small gap between the shadow and the brow area
            adjusted_brow_points = [point for point in brow_points if point[1] > lower_brow_y]

            all_points = np.vstack((eye_points, adjusted_brow_points))
            hull = cv2.convexHull(all_points)
            return hull

        left_area = create_eyeshadow_area(left_eye, left_brow)
        right_area = create_eyeshadow_area(right_eye, right_brow)

        # ======= DRAW EYESHADOW BETWEEN EDGES =======
        shadow_overlay = np.zeros_like(image)

        for area in [left_area, right_area]:
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, area, 255)

            # Blur for soft edges
            mask_blur = cv2.GaussianBlur(mask, (51, 51), 0)
            mask_normalized = mask_blur / 255.0

            # Apply a more transparent shadow based on user input
            for c in range(3):
                shadow_overlay[:, :, c] = np.clip(
                    shadow_overlay[:, :, c] + mask_normalized * shadow_color[c] * transparency_factor,
                    0, 255
                ).astype(np.uint8)

        # Combine
        output = cv2.addWeighted(output, 1.0, shadow_overlay, 0.7, 0)

    # ======= SHOW FINAL RESULT =======
    print(f"\nUpper eyelid (eye to brow) eyeshadow result with transparency factor {transparency_factor} and color {shadow_color}:")
    cv2_imshow(output)
    print("Eyeshadow applied successfully!")
else:
    print("No face detected.")

# Release resources
face_mesh.close()
