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

# ======= MOST POPULAR BLUSHER COLORS =======
print("=== Top 5 Most Popular Blusher Colors ===")
print("1. Nude Peach (Universal everyday shade)")
print("2. Rosy Pink (Classic healthy flush)")
print("3. Warm Apricot (Perfect for warm undertones)")
print("4. Berry Stain (Deeper winter color)")
print("5. Soft Mauve (Cool-toned elegant look)")

color_choice = input("Enter choice (1-5): ").strip()

# Professional makeup artist-approved colors (BGR format)
if color_choice == '1':
    blusher_color = np.array([155, 185, 240], dtype=np.float32)  # Nude Peach
elif color_choice == '2':
    blusher_color = np.array([180, 120, 220], dtype=np.float32)  # Rosy Pink
elif color_choice == '3':
    blusher_color = np.array([120, 160, 250], dtype=np.float32)  # Warm Apricot
elif color_choice == '4':
    blusher_color = np.array([80, 60, 150], dtype=np.float32)    # Berry Stain
elif color_choice == '5':
    blusher_color = np.array([130, 110, 180], dtype=np.float32)  # Soft Mauve
else:
    print("Invalid choice. Defaulting to Nude Peach.")
    blusher_color = np.array([155, 185, 240], dtype=np.float32)

# ======= INPUT INTENSITY =======
intensity_factor = float(input("Enter the intensity factor (0.1 - 1.0): ").strip())
intensity_factor = max(0.1, min(intensity_factor, 1.0))

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
    image_float = image.astype(np.float32) / 255.0
except Exception as e:
    print(f"Error loading image: {e}")
    exit()

# ======= PROCESS IMAGE =======
results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

if results.multi_face_landmarks:
    output = image_float.copy()

    for face_landmarks in results.multi_face_landmarks:
        def get_landmark_point(index):
            return int(face_landmarks.landmark[index].x * w), int(face_landmarks.landmark[index].y * h)
        
        # Key landmarks for positioning
        left_cheek_ref = get_landmark_point(123)
        right_cheek_ref = get_landmark_point(352)
        
        # Calculate circle parameters
        circle_radius = int(w * 0.15)
        
        # Position circles slightly higher for natural look
        left_center = (left_cheek_ref[0] + int(w * 0.06), left_cheek_ref[1] - int(h * 0.01))
        right_center = (right_cheek_ref[0] - int(w * 0.06), right_cheek_ref[1] - int(h * 0.01))

        # Create professional-grade mask with tapered edges
        def create_circular_mask(center, radius):
            mask = np.zeros((h, w), dtype=np.float32)
            cv2.circle(mask, center, radius, 1, -1)
            
            # Create tapered edges (like real makeup brushes)
            dist = cv2.distanceTransform((mask * 255).astype(np.uint8), cv2.DIST_L2, 5)
            dist = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
            mask = np.power(dist, 1.5)  # More natural falloff
            
            # Directional blur mimics real brush strokes
            mask = cv2.GaussianBlur(mask, (101, 101), 0)
            return mask

        left_mask = create_circular_mask(left_center, circle_radius)
        right_mask = create_circular_mask(right_center, circle_radius)
        combined_mask = np.maximum(left_mask, right_mask)

        # PROFESSIONAL COLOR BLENDING TECHNIQUE
        blusher_color_float = blusher_color / 255.0
        
        # Convert to YCrCb to preserve skin luminance
        ycrcb_image = cv2.cvtColor(output, cv2.COLOR_BGR2YCrCb)
        ycrcb_blusher = cv2.cvtColor(blusher_color_float.reshape(1,1,3), cv2.COLOR_BGR2YCrCb)
        
        # Blend chroma channels only (keeps natural skin texture)
        blended_ycrcb = ycrcb_image.copy()
        blended_ycrcb[..., 1] = ycrcb_image[..., 1] * (1 - intensity_factor * combined_mask*2) + \
                               ycrcb_blusher[0,0,1] * intensity_factor * combined_mask*2
        blended_ycrcb[..., 2] = ycrcb_image[..., 2] * (1 - intensity_factor * combined_mask*2) + \
                               ycrcb_blusher[0,0,2] * intensity_factor * combined_mask*2
        
        # Convert back to BGR
        blended_bgr = cv2.cvtColor(blended_ycrcb, cv2.COLOR_YCrCb2BGR)
        
        # Final blending with luminosity preservation
        output = output * (1 - combined_mask[..., np.newaxis]) + \
                blended_bgr * combined_mask[..., np.newaxis]

    # Add subtle highlight effect
    highlight_mask = cv2.GaussianBlur(combined_mask, (151, 151), 0) * 0.15
    output = np.minimum(output + highlight_mask[..., np.newaxis] * np.array([0.1, 0.1, 0.1]), 1.0)

    # Convert back to 8-bit
    output = (output * 255).astype(np.uint8)

    # ======= SHOW FINAL RESULT =======
    print("\nProfessional blusher application:")
    cv2_imshow(output)
    print(f"Applied {['Nude Peach','Rosy Pink','Warm Apricot','Berry Stain','Soft Mauve'][int(color_choice)-1]} successfully!")
else:
    print("No face detected.")

# Release resources
face_mesh.close()