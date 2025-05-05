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

# ======= ENHANCED COLOR SELECTION =======
print("=== Professional Lipstick Palette ===")
print("1. Classic Red (B:0, G:0, R:255)")
print("2. Ruby Woo (B:0, G:35, R:215)")       # MAC's iconic red
print("3. Pink Nouveau (B:130, G:90, R:220)") # Bright cool pink
print("4. Velvet Teddy (B:45, G:110, R:180)") # MAC's famous nude
print("5. Whirl (B:50, G:85, R:140)")        # 90's brown nude
print("6. Diva (B:0, G:35, R:125)")          # Deep matte burgundy
print("7. Russian Red (B:0, G:45, R:175)")   # Blue-toned red
print("8. Mehr (B:60, G:95, R:160)")         # Dusty rose
print("9. Faux (B:70, G:120, R:190)")        # Muted mauve
print("10. Brave (B:80, G:130, R:210)")      # Pinky nude")
print("11. Twig (B:50, G:80, R:130)")        # Soft muted brown")
print("12. Rebel (B:80, G:20, R:120)")       # Berry sangria")
print("13. Dark Side (B:10, G:10, R:50)")    # Deep vampy")
print("14. Flat Out Fabulous (B:120, G:40, R:180)") # Retro matte")
print("15. Candy Yum-Yum (B:0, G:130, R:255)") # Neon pink")
print("16. Taupe (B:40, G:75, R:120)")       # 90's brown")
print("17. Persistence (B:50, G:100, R:160)") # Greige nude")
print("18. Sin (B:0, G:40, R:110)")          # Dark berry")
print("19. Lady Danger (B:0, G:65, R:240)")  # Orange-red")
print("20. Peach Blossom (B:0, G:150, R:220)") # Coral nude")
print("21. Custom Color")

color_choice = input("Enter choice (1-21): ").strip()

# Enhanced lip color presets (BGR format)
color_presets = {
    '1': [0, 0, 255],    # Classic Red
    '2': [0, 35, 215],   # Ruby Woo
    '3': [130, 90, 220], # Pink Nouveau
    '4': [45, 110, 180], # Velvet Teddy
    '5': [50, 85, 140],  # Whirl
    '6': [0, 35, 125],   # Diva
    '7': [0, 45, 175],   # Russian Red
    '8': [60, 95, 160],  # Mehr
    '9': [70, 120, 190], # Faux
    '10': [80, 130, 210], # Brave
    '11': [50, 80, 130], # Twig
    '12': [80, 20, 120], # Rebel
    '13': [10, 10, 50],  # Dark Side
    '14': [120, 40, 180], # Flat Out Fabulous
    '15': [0, 130, 255], # Candy Yum-Yum
    '16': [40, 75, 120], # Taupe
    '17': [50, 100, 160], # Persistence
    '18': [0, 40, 110],  # Sin
    '19': [0, 65, 240],  # Lady Danger
    '20': [0, 150, 220]  # Peach Blossom
}

if color_choice in color_presets:
    lip_color = np.array(color_presets[color_choice], dtype=np.float32)
elif color_choice == '21':
    r = min(int(input("Red (0-255): ")), 255)
    g = min(int(input("Green (0-255): ")), 255)
    b = min(int(input("Blue (0-255): ")), 255)
    lip_color = np.array([b, g, r], dtype=np.float32)
else:
    print("Invalid choice. Defaulting to Classic Red.")
    lip_color = np.array([0, 0, 255], dtype=np.float32)

# ======= INTENSITY CONTROL =======
intensity_factor = float(input("Enter color intensity (0.1 - 1.0): ").strip())
intensity_factor = max(0.1, min(intensity_factor, 1.0))

# ======= EDGE WIDTH CONTROL =======
edge_width = int(input("Enter edge transparent width (1-20 pixels): ").strip())
edge_width = max(1, min(edge_width, 20))

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
        # Complete lip landmarks
        LIPS_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]

        def get_landmark_points(indices):
            return np.array([
                (int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h))
                for i in indices
            ], dtype=np.int32)

        outer_lips = get_landmark_points(LIPS_OUTER)

        # Create lip mask
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [outer_lips], 255)
        
        # Edge transparency with user-controlled width
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_width*2+1, edge_width*2+1))
        inner_mask = cv2.erode(mask, kernel)
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        edge_gradient = np.where(mask > 0, np.clip(distance / edge_width, 0, 1), 0)
        final_mask = np.maximum(inner_mask/255.0, edge_gradient)
        final_mask = cv2.GaussianBlur(final_mask, (0,0), sigmaX=edge_width/2)

        # Texture-preserving color application
        lips_region = output.copy()
        target_color_bgr = lip_color / 255.0
        
        # Convert to HSV for better blending
        target_color_hsv = cv2.cvtColor(np.array([[target_color_bgr]], dtype=np.float32), cv2.COLOR_BGR2HSV)[0,0]
        lips_hsv = cv2.cvtColor(lips_region, cv2.COLOR_BGR2HSV)
        
        # Blend hue and saturation while preserving value (brightness)
        lips_hsv[..., 0] = lips_hsv[..., 0] * (1 - intensity_factor*final_mask) + target_color_hsv[0] * intensity_factor * final_mask
        lips_hsv[..., 1] = lips_hsv[..., 1] * (1 - intensity_factor*final_mask) + target_color_hsv[1] * intensity_factor * final_mask
        
        # Convert back to BGR
        colored_lips = cv2.cvtColor(lips_hsv, cv2.COLOR_HSV2BGR)
        
        # Final blend
        output = output * (1 - final_mask[..., np.newaxis]) + colored_lips * final_mask[..., np.newaxis]

    # Convert back to 8-bit
    output = (output * 255).astype(np.uint8)

    # ======= SHOW FINAL RESULT =======
    print("\nLip color applied successfully!")
    cv2_imshow(output)
else:
    print("No face detected.")

# Release resources
face_mesh.close()