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

def load_image(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        img_array = np.array(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image")
        return image.astype(np.float32) / 255.0
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def create_face_mask(image, face_landmarks):
    h, w = image.shape[:2]
    landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]
    
    # Create convex hull mask with chin extension
    hull = cv2.convexHull(np.array(landmarks))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)
    
    # Extend chin area downward
    chin_points = [152, 148, 176, 149, 150, 136, 172, 58, 132]
    chin_bottom = max([landmarks[i][1] for i in chin_points])
    extension = int(h * 0.08)  # Extend mask downward
    for x in range(w):
        if mask[chin_bottom, x] == 1:
            for y in range(chin_bottom, min(chin_bottom + extension, h)):
                mask[y, x] = 1
    
    # Distance transform for smooth edges
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    mask = (dist / dist.max()) ** 0.7
    
    # Create exclusion zones
    def create_exclusion(indices):
        points = np.array([landmarks[i] for i in indices])
        x,y,w_,h_ = cv2.boundingRect(points)
        center = (x + w_//2, y + h_//2)
        axes = (int(w_*0.7), int(h_*0.7))
        exclusion = np.zeros((h,w), dtype=np.float32)
        cv2.ellipse(exclusion, center, axes, 0, 0, 360, 1, -1)
        return cv2.GaussianBlur(exclusion, (71,71), 0)
    
    eyes_left = [33,133,144,145,153,154,155,157,158,159,160,161,163]
    eyes_right = [362,263,373,374,380,381,382,384,385,386,387,388,390]
    mouth = [61,185,40,39,37,267,269,270,409,291,375,321,405,314,17,84,181,91,146]
    
    exclusion = np.maximum.reduce([
        create_exclusion(eyes_left),
        create_exclusion(eyes_right),
        create_exclusion(mouth)
    ])
    
    return np.clip(mask - exclusion, 0, 1)

def adjust_foundation_color(target_bgr, skin_bgr):
    """Adjust foundation color to match skin tone in LAB space for better perceptual results"""
    # Convert to LAB color space
    target_lab = cv2.cvtColor(np.array([[target_bgr]], dtype=np.float32), cv2.COLOR_BGR2LAB)[0,0]
    skin_lab = cv2.cvtColor(np.array([[skin_bgr]], dtype=np.float32), cv2.COLOR_BGR2LAB)[0,0]
    
    # Blend colors while preserving foundation characteristics
    adjusted_lab = np.array([
        skin_lab[0] * 0.2 + target_lab[0] * 0.8,  # Mostly keep foundation lightness
        target_lab[1] * 0.7 + skin_lab[1] * 0.3,  # Blend color channels
        target_lab[2] * 0.7 + skin_lab[2] * 0.3
    ])
    
    # Convert back to BGR
    adjusted_bgr = cv2.cvtColor(np.array([[[adjusted_lab[0], adjusted_lab[1], adjusted_lab[2]]]], dtype=np.float32), 
                              cv2.COLOR_LAB2BGR)[0,0]
    return np.clip(adjusted_bgr, 0, 1)

def apply_foundation(image, face_landmarks, foundation_bgr, coverage):
    h, w = image.shape[:2]
    mask = create_face_mask(image, face_landmarks)
    
    # Sample skin tone from multiple areas
    sample_points = [123, 50, 351, 346, 129, 358]  # Cheeks and forehead
    skin_samples = [image[int(face_landmarks.landmark[i].y * h),
                         int(face_landmarks.landmark[i].x * w)] for i in sample_points]
    skin_bgr = np.mean(skin_samples, axis=0)
    
    # Adjust foundation color
    adjusted_color = adjust_foundation_color(foundation_bgr, skin_bgr)
    
    # Create foundation layer with texture preservation
    foundation_layer = np.zeros_like(image)
    foundation_layer[:,:,:] = adjusted_color
    
    # Preserve skin texture
    texture = image - cv2.GaussianBlur(image, (0,0), 3)
    texture_strength = 0.5 * (1 - coverage)
    
    # Multi-level blending for natural look
    soft_mask = cv2.GaussianBlur(mask, (15,15), 0)
    result = image * (1 - soft_mask[...,None] * coverage) + foundation_layer * soft_mask[...,None] * coverage
    result = np.clip(result + texture * texture_strength, 0, 1)
    
    return result

def get_custom_color():
    """Get custom foundation color from user input"""
    print("\nEnter custom foundation color (RGB values 0-255):")
    while True:
        try:
            r = int(input("Red (0-255): "))
            g = int(input("Green (0-255): "))
            b = int(input("Blue (0-255): "))
            if 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255:
                return (r, g, b)
            print("Values must be between 0-255. Try again.")
        except ValueError:
            print("Please enter valid integers.")

# Main execution
if __name__ == "__main__":
    image_url = 'https://i.pinimg.com/474x/ea/b3/a3/eab3a381fab1bd6b4a4e61ebabc73b63.jpg'
    image = load_image(image_url)
    if image is None:
        exit()

    # Foundation options in RGB format (user-friendly)
    foundations = {
        '1': (220, 210, 200),  # Ivory
        '2': (210, 190, 180),  # Porcelain
        '3': (190, 170, 160),  # Beige
        '4': (170, 150, 140),  # Sand
        '5': (150, 130, 120),  # Honey
        '6': (130, 110, 100),  # Caramel
        '7': (110, 90, 80),    # Cocoa
        '8': (90, 70, 60),     # Espresso
        '9': "Custom"          # Custom color option
    }

    print("=== Advanced Foundation Simulator ===")
    print("1. Ivory\n2. Porcelain\n3. Beige\n4. Sand\n5. Honey\n6. Caramel\n7. Cocoa\n8. Espresso\n9. Custom Color")
    choice = input("Select foundation option (1-9): ").strip()
    
    if choice == '9':
        foundation_rgb = get_custom_color()
    else:
        foundation_rgb = foundations.get(choice, foundations['3'])
    
    coverage = float(input("Coverage level (0.1-1.0): "))
    coverage = np.clip(coverage, 0.1, 1.0)

    # Convert RGB to BGR for processing
    foundation_bgr = np.array(foundation_rgb[::-1], dtype=np.float32) / 255.0

    results = face_mesh.process(cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
    
    if results.multi_face_landmarks:
        output = image.copy()
        for face_landmarks in results.multi_face_landmarks:
            output = apply_foundation(output, face_landmarks, foundation_bgr, coverage)
        
        output = (output * 255).astype(np.uint8)
        print("\n=== Before & After ===")
        cv2_imshow(np.hstack([(image * 255).astype(np.uint8), output]))
        
        # Show selected foundation color
        color_swatch = np.zeros((100, 100, 3), dtype=np.uint8)
        color_swatch[:,:,:] = foundation_rgb[::-1]  # Convert back to BGR for display
        print("\nSelected Foundation Color:")
        cv2_imshow(color_swatch)
    else:
        print("No face detected")

    face_mesh.close()