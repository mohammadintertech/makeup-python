import cv2
import mediapipe as mp
import numpy as np
import logging
from threading import Lock

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mp_face_mesh = mp.solutions.face_mesh

# Create a lock for thread-safe MediaPipe operations
face_mesh_lock = Lock()

def apply_lipstick(image: np.ndarray, lip_color: list, intensity_factor: float, edge_width: int) -> np.ndarray:
    """
    Apply lipstick to an image with improved handling of dark colors and memory safety
    
    Args:
        image: Input image in BGR format
        lip_color: Lipstick color as [R, G, B] values (0-255)
        intensity_factor: Strength of the effect (0.0-1.0)
        edge_width: Width of the feathered edge (pixels)
        
    Returns:
        Image with applied lipstick in BGR format
        
    Raises:
        ValueError: If no face is detected or input parameters are invalid
        Exception: For other processing errors
    """
    try:
        logger.info("Applying lipstick")
        
        # Validate input parameters
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a 3-channel BGR numpy array")
        if len(lip_color) != 3 or not all(0 <= c <= 255 for c in lip_color):
            raise ValueError("Lip color must be a list of 3 values between 0-255")
        if not 0 <= intensity_factor <= 1:
            raise ValueError("Intensity factor must be between 0.0 and 1.0")
        if edge_width < 0:
            raise ValueError("Edge width must be a positive integer")

        # Convert image to float and make a copy
        image_float = image.astype(np.float32) / 255.0
        output = image_float.copy()
        h, w = image.shape[:2]
        
        try:
            # Use lock to prevent concurrent MediaPipe operations
            with face_mesh_lock:
                with mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    refine_landmarks=True,
                    max_num_faces=1,
                    min_detection_confidence=0.5
                ) as face_mesh:
                    # Convert to RGB and process
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    rgb_image.flags.writeable = False  # Improve performance
                    results = face_mesh.process(rgb_image)

                    if not results.multi_face_landmarks:
                        raise ValueError("No face detected")

                    face_landmarks = results.multi_face_landmarks[0]

                    # Updated lip landmarks (more comprehensive)
                    LIPS_OUTER = [
                        61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 
                        291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
                        61, 76, 62, 78, 191, 80, 81, 82, 13, 312,
                        311, 310, 415, 308, 324, 318, 402, 317, 14, 87
                    ]
                    
                    # Get lip coordinates
                    lip_points = []
                    for i in LIPS_OUTER:
                        try:
                            landmark = face_landmarks.landmark[i]
                            x = int(landmark.x * w)
                            y = int(landmark.y * h)
                            lip_points.append([x, y])
                        except Exception as e:
                            logger.warning(f"Error processing landmark {i}: {str(e)}")
                            continue
                    
                    if len(lip_points) < 3:
                        raise ValueError("Not enough valid lip points detected")
                    
                    lip_points = np.array(lip_points, dtype=np.int32)

                    # Create mask
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(mask, [lip_points], 255)

                    # Create smooth edge
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_width*2+1, edge_width*2+1))
                    mask = cv2.dilate(mask, kernel)
                    mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), edge_width/2)
                    mask = mask / 255.0  # Normalize to [0,1]

                    # Convert lip color to BGR and normalize
                    target_color = np.array(lip_color[::-1], dtype=np.float32) / 255.0  # RGB to BGR and normalize

                    # Create colored lips
                    colored_lips = np.zeros_like(output)
                    colored_lips[:,:,:] = target_color

                    # Blend with original image
                    alpha = mask * intensity_factor
                    alpha = alpha[..., np.newaxis]  # Add channel dimension for broadcasting
                    
                    # Apply the effect
                    output = output * (1 - alpha) + colored_lips * alpha

                    # Convert back to 8-bit
                    return (np.clip(output, 0, 1) * 255).astype(np.uint8)
                    
        except Exception as e:
            logger.error(f"Error during face processing: {str(e)}")
            raise

    except Exception as e:
        logger.error(f"Error in apply_lipstick: {str(e)}")
        raise
