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

# ======= ASK FOR EYELINER COLOR =======
print("=== Eyeliner Color Options ===")
print("1. Classic Black")
print("2. Dark Brown")
print("3. Navy Blue")
print("4. Emerald Green")
print("5. Deep Purple")
print("6. Custom Color")

eyeliner_choice = input("Enter eyeliner color choice (1-6): ").strip()

eyeliner_colors = {
    '1': (0, 0, 0),        # Black
    '2': (30, 60, 120),    # Dark Brown
    '3': (120, 0, 0),      # Navy Blue
    '4': (0, 100, 0),      # Emerald Green
    '5': (130, 0, 130),    # Deep Purple,
}

if eyeliner_choice in eyeliner_colors:
    eyeliner_color = eyeliner_colors[eyeliner_choice]
elif eyeliner_choice == '6':
    r = min(int(input("Red (0-255): ")), 255)
    g = min(int(input("Green (0-255): ")), 255)
    b = min(int(input("Blue (0-255): ")), 255)
    eyeliner_color = (b, g, r)
else:
    print("Invalid choice. Defaulting to Black.")
    eyeliner_color = (0, 0, 0)

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
    # Create a transparent overlay for the eyeliner
    overlay = image.copy()
    output = image.copy()
    alpha = 0.7  # Transparency factor (0 = fully transparent, 1 = fully opaque)

    for face_landmarks in results.multi_face_landmarks:
        # Use exact outer corner points
        LEFT_EYE_OUTER = 130
        RIGHT_EYE_OUTER = 359

        left_corner = (
            int(face_landmarks.landmark[LEFT_EYE_OUTER].x * w),
            int(face_landmarks.landmark[LEFT_EYE_OUTER].y * h)
        )
        right_corner = (
            int(face_landmarks.landmark[RIGHT_EYE_OUTER].x * w),
            int(face_landmarks.landmark[RIGHT_EYE_OUTER].y * h)
        )

        # Eyeliner settings
        max_thickness = 8
        min_thickness = 2
        eyeliner_length = int(w * 0.04)  # Slightly smaller wing

        def draw_smooth_eyeliner(start_point, direction='right', tilt_up=True):
            points = []
            for i in range(eyeliner_length):
                alpha = i / eyeliner_length
                x = start_point[0] + (i if direction == 'right' else -i)
                y = start_point[1] - (i * 0.2 if tilt_up else 0)
                points.append((int(x), int(y)))

            points = np.array(points, np.int32)
            points = points.reshape((-1, 1, 2))
            
            # Smooth, anti-aliased polyline
            for i in range(1, len(points)):
                thickness = int(max_thickness * (1 - i / len(points)) + min_thickness * (i / len(points)))
                cv2.line(overlay, tuple(points[i-1][0]), tuple(points[i][0]), eyeliner_color, thickness, lineType=cv2.LINE_AA)

        # Draw left eye wing (outward)
        draw_smooth_eyeliner(left_corner, direction='left', tilt_up=True)

        # Draw right eye wing (outward)
        draw_smooth_eyeliner(right_corner, direction='right', tilt_up=True)

    # Blend the overlay with the original image
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    # ======= SHOW RESULT =======
    print("\nBeautiful high-quality eyeliner applied! 😍")
    cv2_imshow(output)
else:
    print("No face detected.")

# Release resources
face_mesh.close()