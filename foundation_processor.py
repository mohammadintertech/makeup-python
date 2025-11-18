import cv2
import numpy as np
from typing import Optional

class FoundationApplier:
    def __init__(self):
        """Initialize OpenCV face detector instead of MediaPipe"""
        try:
            # Use OpenCV's built-in face detector
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.initialized = True
            print("OpenCV face detector initialized successfully")
        except Exception as e:
            print(f"Failed to initialize OpenCV face detector: {e}")
            self.initialized = False

    def apply_foundation(self, image: np.ndarray, foundation_rgb: list, intensity: float) -> Optional[np.ndarray]:
        """
        Apply foundation to face in image using OpenCV face detection
        
        Args:
            image: Input BGR image (numpy array)
            foundation_rgb: Foundation color as [R, G, B] (0-255)
            intensity: Strength of application (0.0-1.0)
            
        Returns:
            Result image with foundation applied
        """
        if not self.initialized:
            return image
            
        # Validate inputs
        if not isinstance(image, np.ndarray) or image.ndim != 3:
            return image
        if len(foundation_rgb) != 3 or not all(0 <= c <= 255 for c in foundation_rgb):
            return image
        intensity = np.clip(intensity, 0.0, 1.0)
        
        # Convert image to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            print("No face detected for foundation")
            return image
        
        # Apply foundation to each detected face
        result = image.copy()
        foundation_bgr = np.array(foundation_rgb[::-1], dtype=np.float32) / 255.0
        
        for (x, y, w, h) in faces:
            result = self._apply_foundation_to_face_region(result, x, y, w, h, foundation_bgr, intensity)
        
        return result

    def _apply_foundation_to_face_region(self, image: np.ndarray, x: int, y: int, w: int, h: int, 
                                       foundation_bgr: np.ndarray, intensity: float) -> np.ndarray:
        """Apply foundation to a detected face region"""
        # Create an elliptical mask for the face
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        
        # Create ellipse covering the face area
        center = (x + w//2, y + h//2)
        axes = (int(w * 0.5), int(h * 0.6))  # Make it slightly taller for better coverage
        
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)
        
        # Create exclusions for eyes and mouth (approximate positions)
        eye_y = y + h//3
        eye_radius = w//12
        mouth_y = y + int(h * 0.75)
        mouth_radius = w//8
        
        # Left eye exclusion
        left_eye_x = x + w//3
        cv2.circle(mask, (left_eye_x, eye_y), eye_radius, 0, -1)
        
        # Right eye exclusion
        right_eye_x = x + 2*w//3
        cv2.circle(mask, (right_eye_x, eye_y), eye_radius, 0, -1)
        
        # Mouth exclusion
        mouth_x = x + w//2
        cv2.ellipse(mask, (mouth_x, mouth_y), (mouth_radius, mouth_radius//2), 0, 0, 360, 0, -1)
        
        # Smooth the mask
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        
        # Sample skin tone from the face region
        face_region = image[y:y+h, x:x+w]
        if face_region.size > 0:
            # Sample from cheek areas (avoid center where nose might be)
            cheek_samples = []
            cheek_y = y + h//2
            cheek_left_x = x + w//4
            cheek_right_x = x + 3*w//4
            
            # Sample multiple points around cheeks
            for dy in range(-h//8, h//8, 5):
                for dx in range(-w//12, w//12, 5):
                    # Left cheek
                    py, px = cheek_y + dy, cheek_left_x + dx
                    if 0 <= py < image.shape[0] and 0 <= px < image.shape[1]:
                        cheek_samples.append(image[py, px].astype(np.float32) / 255.0)
                    
                    # Right cheek  
                    py, px = cheek_y + dy, cheek_right_x + dx
                    if 0 <= py < image.shape[0] and 0 <= px < image.shape[1]:
                        cheek_samples.append(image[py, px].astype(np.float32) / 255.0)
            
            if cheek_samples:
                skin_bgr = np.mean(cheek_samples, axis=0)
            else:
                skin_bgr = np.array([0.7, 0.6, 0.5])  # Default skin tone
        else:
            skin_bgr = np.array([0.7, 0.6, 0.5])  # Default skin tone
        
        # Adjust foundation color to match skin tone
        adjusted_foundation = self._adjust_foundation_color(foundation_bgr, skin_bgr)
        
        # Convert image to float for processing
        image_float = image.astype(np.float32) / 255.0
        
        # Create foundation layer
        foundation_layer = np.zeros_like(image_float)
        foundation_layer[:, :] = adjusted_foundation
        
        # Preserve skin texture by adding high-frequency details
        blurred = cv2.GaussianBlur(image_float, (0, 0), 2.0)
        texture = image_float - blurred
        texture_strength = 0.4 * (1 - intensity)
        
        # Apply foundation with blending
        mask_3d = np.stack([mask] * 3, axis=2)
        result = image_float * (1 - mask_3d * intensity) + foundation_layer * mask_3d * intensity
        result = np.clip(result + texture * texture_strength, 0, 1)
        
        # Convert back to uint8
        return (result * 255).astype(np.uint8)

    def _adjust_foundation_color(self, target_bgr: np.ndarray, skin_bgr: np.ndarray) -> np.ndarray:
        """Adjust foundation color to better match skin tone"""
        # Simple RGB blending to adapt foundation to skin tone
        # Preserve foundation's lightness but adapt color channels
        adapted = target_bgr * 0.7 + skin_bgr * 0.3
        return np.clip(adapted, 0, 1)


# Initialize the applier instance
try:
    foundation_applier = FoundationApplier()
    FOUNDATION_WORKING = True
except Exception as e:
    print(f"Failed to initialize foundation processor: {e}")
    foundation_applier = None
    FOUNDATION_WORKING = False

def apply_foundation(image, foundation_rgb, intensity):
    """
    Public interface function that matches your API requirements
    """
    if not FOUNDATION_WORKING or foundation_applier is None:
        print("Foundation processor not available - returning original image")
        return image
    
    try:
        result = foundation_applier.apply_foundation(image, foundation_rgb, intensity)
        return result if result is not None else image
    except Exception as e:
        print(f"Error applying foundation: {e}")
        return image