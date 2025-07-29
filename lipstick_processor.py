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
    logger.info("Applying lipstick2")

    # Convert image to float and make a copy
    image_float = image.astype(np.float32) / 255.0
    output = image_float.copy()
    h, w = image.shape[:2]
    logger.info("Applying lipstick6")
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

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
            landmark = face_landmarks.landmark[i]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            lip_points.append([x, y])
        
        lip_points = np.array(lip_points, dtype=np.int32)

        # Create mask
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [lip_points], 255)

        # Create smooth edge
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_width*2+1, edge_width*2+1))
        mask = cv2.dilate(mask, kernel)
        mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), edge_width/2)
        mask = mask / 255.0  # Normalize to [0,1]
        logger.info("Applying lipstick9")
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
        logger.info("Applying lipstick88")
        # Convert back to 8-bit
        return (np.clip(output, 0, 1) * 255).astype(np.uint8)
