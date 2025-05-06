import cv2
import mediapipe as mp
import numpy as np

def apply_eyecolor(image: np.ndarray, r: int, g: int, b: int) -> np.ndarray:
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
        
        # Base color for the iris in BGR format
        base_color = (min(b * 0.7, 255), min(g * 0.7, 255), min(r * 0.7, 255))
        
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

        return output
    else:
        return None
