import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

def apply_eyeshadow(image: np.ndarray, eyeshadow_color_rgb: list, transparency_factor: float = 0.7) -> np.ndarray:
    """
    Apply eyeshadow to an image with given RGB color
    
    Args:
        image: Input image in BGR format (OpenCV default)
        eyeshadow_color_rgb: Eyeshadow color as [R, G, B] values (0-255)
        transparency_factor: Strength of the eyeshadow effect (0.1-1.0)
        
    Returns:
        Image with applied eyeshadow in BGR format
        
    Raises:
        ValueError: If no face is detected in the image
    """
    # Convert RGB color to BGR for OpenCV
    shadow_color = eyeshadow_color_rgb[::-1]
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5
    ) as face_mesh:
        h, w = image.shape[:2]
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if results.multi_face_landmarks:
            output = image.copy()

            for face_landmarks in results.multi_face_landmarks:
                # Define landmark groups for eyes and brows
                LEFT_EYE = [33, 246, 161, 160, 159, 158, 157, 173, 133]
                LEFT_BROW = [70, 63, 105, 66, 107, 55, 65, 52]
                RIGHT_EYE = [362, 398, 384, 385, 386, 387, 388, 466, 263]
                RIGHT_BROW = [336, 296, 334, 293, 300, 276, 283, 282]

                # Function to get points
                def get_landmark_points(indices):
                    return np.array([
                        (int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h))
                        for i in indices
                    ], dtype=np.int32)

                left_eye = get_landmark_points(LEFT_EYE)
                left_brow = get_landmark_points(LEFT_BROW)
                right_eye = get_landmark_points(RIGHT_EYE)
                right_brow = get_landmark_points(RIGHT_BROW)

                # Create eyeshadow area between eye and brow
                def create_eyeshadow_area(eye_points, brow_points):
                    lower_brow_y = min(brow_points, key=lambda x: x[1])[1] + 5
                    adjusted_brow_points = [point for point in brow_points if point[1] > lower_brow_y]
                    all_points = np.vstack((eye_points, adjusted_brow_points))
                    return cv2.convexHull(all_points)

                left_area = create_eyeshadow_area(left_eye, left_brow)
                right_area = create_eyeshadow_area(right_eye, right_brow)

                # Create eyeshadow overlay
                shadow_overlay = np.zeros_like(image)

                for area in [left_area, right_area]:
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillConvexPoly(mask, area, 255)
                    mask_blur = cv2.GaussianBlur(mask, (51, 51), 0)
                    mask_normalized = mask_blur / 255.0

                    # Apply color with transparency
                    for c in range(3):
                        shadow_overlay[:, :, c] = np.clip(
                            shadow_overlay[:, :, c] + mask_normalized * shadow_color[c] * transparency_factor,
                            0, 255
                        ).astype(np.uint8)

                # Blend with original image
                output = cv2.addWeighted(output, 1.0, shadow_overlay, 0.7, 0)

            return output
            
        else:
            raise ValueError("No face detected in the image")

# Example usage:
# image = cv2.imread("face.jpg")
# result = apply_eyeshadow(image, [200, 150, 80], transparency_factor=0.8)  # Rose Gold in RGB
# cv2.imwrite("result.jpg", result)