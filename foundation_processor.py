import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, Optional

class FoundationApplier:
    def __init__(self):
        """Initialize face mesh model"""
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )

    def apply_foundation(self, image: np.ndarray, foundation_rgb: list, intensity: float) -> Optional[np.ndarray]:
        """
        Apply foundation to face in image
        
        Args:
            image: Input BGR image (numpy array)
            foundation_rgb: Foundation color as [R, G, B] (0-255)
            intensity: Strength of application (0.0-1.0)
            
        Returns:
            Result image with foundation applied or None if no face detected
        """
        # Validate inputs
        if not isinstance(image, np.ndarray) or image.ndim != 3:
            raise ValueError("Invalid input image")
        if len(foundation_rgb) != 3 or not all(0 <= c <= 255 for c in foundation_rgb):
            raise ValueError("Foundation color must be [R,G,B] with values 0-255")
        intensity = np.clip(intensity, 0.0, 1.0)
        
        # Convert image to float and foundation to BGR
        image_float = image.astype(np.float32) / 255.0
        foundation_bgr = np.array(foundation_rgb[::-1], dtype=np.float32) / 255.0
        
        # Process face landmarks
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return None
        
        # Apply foundation to each detected face
        output = image_float.copy()
        for face_landmarks in results.multi_face_landmarks:
            output = self._apply_foundation_to_face(output, face_landmarks, foundation_bgr, intensity)
        
        # Convert back to 8-bit format
        return (output * 255).astype(np.uint8)

    def _apply_foundation_to_face(self, image: np.ndarray, face_landmarks, foundation_bgr: np.ndarray, coverage: float) -> np.ndarray:
        """Apply foundation to a single face"""
        h, w = image.shape[:2]
        mask = self._create_face_mask(image, face_landmarks)
        
        # Sample skin tone from multiple areas
        sample_points = [123, 50, 351, 346, 129, 358]  # Cheeks and forehead
        skin_samples = [image[int(face_landmarks.landmark[i].y * h),
                        int(face_landmarks.landmark[i].x * w)] for i in sample_points]
        skin_bgr = np.mean(skin_samples, axis=0)
        
        # Adjust foundation color to match skin tone
        adjusted_color = self._adjust_foundation_color(foundation_bgr, skin_bgr)
        
        # Create foundation layer
        foundation_layer = np.zeros_like(image)
        foundation_layer[:,:,:] = adjusted_color
        
        # Preserve skin texture
        texture = image - cv2.GaussianBlur(image, (0,0), 3)
        texture_strength = 0.5 * (1 - coverage)
        
        # Multi-level blending for natural look
        soft_mask = cv2.GaussianBlur(mask, (15,15), 0)
        result = image * (1 - soft_mask[...,None] * coverage) + foundation_layer * soft_mask[...,None] * coverage
        return np.clip(result + texture * texture_strength, 0, 1)

    def _create_face_mask(self, image: np.ndarray, face_landmarks) -> np.ndarray:
        """Create facial mask with exclusions for eyes and mouth"""
        h, w = image.shape[:2]
        landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]
        
        # Convex hull mask with chin extension
        hull = cv2.convexHull(np.array(landmarks))
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 1)
        
        # Extend chin area
        chin_points = [152, 148, 176, 149, 150, 136, 172, 58, 132]
        chin_bottom = max([landmarks[i][1] for i in chin_points])
        extension = int(h * 0.08)
        for x in range(w):
            if mask[chin_bottom, x] == 1:
                mask[chin_bottom:min(chin_bottom + extension, h), x] = 1
        
        # Smooth edges
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        mask = (dist / dist.max()) ** 0.7
        
        # Create exclusion zones for eyes and mouth
        def create_exclusion(indices):
            points = np.array([landmarks[i] for i in indices])
            x, y, w_, h_ = cv2.boundingRect(points)
            exclusion = np.zeros((h, w), dtype=np.float32)
            cv2.ellipse(exclusion, 
                        (x + w_//2, y + h_//2),
                        (int(w_*0.7), int(h_*0.7)),
                        0, 0, 360, 1, -1)
            return cv2.GaussianBlur(exclusion, (71, 71), 0)
        
        exclusion = np.maximum.reduce([
            create_exclusion([33,133,144,145,153,154,155,157,158,159,160,161,163]),  # Left eye
            create_exclusion([362,263,373,374,380,381,382,384,385,386,387,388,390]),  # Right eye
            create_exclusion([61,185,40,39,37,267,269,270,409,291,375,321,405,314,17,84,181,91,146])  # Mouth
        ])
        
        return np.clip(mask - exclusion, 0, 1)

    def _adjust_foundation_color(self, target_bgr: np.ndarray, skin_bgr: np.ndarray) -> np.ndarray:
        """Adjust foundation color to better match skin tone in LAB space"""
        target_lab = cv2.cvtColor(np.array([[target_bgr]], dtype=np.float32), cv2.COLOR_BGR2LAB)[0,0]
        skin_lab = cv2.cvtColor(np.array([[skin_bgr]], dtype=np.float32), cv2.COLOR_BGR2LAB)[0,0]
        
        adjusted_lab = np.array([
            skin_lab[0] * 0.2 + target_lab[0] * 0.8,  # Preserve foundation lightness
            target_lab[1] * 0.7 + skin_lab[1] * 0.3,  # Blend color channels
            target_lab[2] * 0.7 + skin_lab[2] * 0.3
        ])
        
        adjusted_bgr = cv2.cvtColor(np.array([[[adjusted_lab[0], adjusted_lab[1], adjusted_lab[2]]]], dtype=np.float32), 
                                  cv2.COLOR_LAB2BGR)[0,0]
        return np.clip(adjusted_bgr, 0, 1)


# Initialize the applier instance (should be done once at startup)
foundation_applier = FoundationApplier()

def apply_foundation(image: np.ndarray, foundation_rgb: list, intensity: float) -> Optional[np.ndarray]:
    """
    Public interface function that matches your API requirements
    """
    return foundation_applier.apply_foundation(image, foundation_rgb, intensity)