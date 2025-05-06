import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

def apply_lipstick(image: np.ndarray, lip_color: list, intensity_factor: float, edge_width: int) -> np.ndarray:
    """
    Apply lipstick to an image with improved handling of dark colors
    
    Args:
        image: Input image in BGR format
        lip_color: Lipstick color as [R, G, B] values (0-255)
        intensity_factor: Strength of the effect (0.0-1.0)
        edge_width: Width of the feathered edge (pixels)
        
    Returns:
        Image with applied lipstick in BGR format
    """
    h, w = image.shape[:2]
    image_float = image.astype(np.float32) / 255.0

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks:
            output = image_float.copy()
            face_landmarks = results.multi_face_landmarks[0]

            # Lip landmarks (outer lips)
            LIPS_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]

            outer_lips = np.array([
                (int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h))
                for i in LIPS_OUTER
            ], dtype=np.int32)

            # Create lip mask
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [outer_lips], 255)

            # Create feathered edge
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_width*2+1, edge_width*2+1))
            inner_mask = cv2.erode(mask, kernel)
            distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            edge_gradient = np.where(mask > 0, np.clip(distance / edge_width, 0, 1), 0)
            final_mask = np.maximum(inner_mask/255.0, edge_gradient)
            final_mask = cv2.GaussianBlur(final_mask, (0,0), sigmaX=edge_width/2)

            # Convert target color to HSV and adjust value for dark colors
            target_color_bgr = np.array(lip_color[::-1], dtype=np.float32) / 255.0  # Convert RGB to BGR
            target_color_hsv = cv2.cvtColor(np.array([[target_color_bgr]], dtype=np.float32), cv2.COLOR_BGR2HSV)[0,0]
            
            # Boost value (brightness) for dark colors while preserving hue
            if target_color_hsv[2] < 0.3:  # If color is dark
                target_color_hsv[2] = min(target_color_hsv[2] * 1.5, 0.8)  # Boost brightness but keep it natural

            # Convert image to HSV for color manipulation
            lips_hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)
            
            # Blend hue and saturation
            lips_hsv[..., 0] = lips_hsv[..., 0] * (1 - intensity_factor*final_mask) + target_color_hsv[0] * intensity_factor * final_mask
            lips_hsv[..., 1] = lips_hsv[..., 1] * (1 - intensity_factor*final_mask) + target_color_hsv[1] * intensity_factor * final_mask
            
            # Special handling for value channel - preserve natural lip highlights
            original_value = lips_hsv[..., 2].copy()
            lips_hsv[..., 2] = lips_hsv[..., 2] * (1 - intensity_factor*final_mask*0.7) + target_color_hsv[2] * intensity_factor * final_mask * 0.7
            
            # Restore highlights from original image
            highlight_mask = np.where(original_value > 0.7, original_value - 0.7, 0)
            lips_hsv[..., 2] = np.minimum(lips_hsv[..., 2] + highlight_mask * 0.5, 1.0)

            # Convert back to BGR
            colored_lips = cv2.cvtColor(lips_hsv, cv2.COLOR_HSV2BGR)

            # Blend with original image
            output = output * (1 - final_mask[..., np.newaxis]) + colored_lips * final_mask[..., np.newaxis]
            return (output * 255).astype(np.uint8)

        else:
            raise ValueError("No face detected")

# Example usage:
# image = cv2.imread("face.jpg")
# result = apply_lipstick(image, [120, 30, 60], intensity_factor=0.8, edge_width=5)  # Dark red color
# cv2.imwrite("result.jpg", result)