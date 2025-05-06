import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

def apply_eyeliner(image: np.ndarray, eyeliner_color_rgb: list, alpha: float = 0.7) -> np.ndarray:
    """
    Apply eyeliner to an image with given RGB color
    
    Args:
        image: Input image in BGR format (OpenCV default)
        eyeliner_color_rgb: Eyeliner color as [R, G, B] values (0-255)
        alpha: Transparency factor (0-1) where 1 is fully opaque
        
    Returns:
        Image with applied eyeliner in BGR format
        
    Raises:
        ValueError: If no face is detected in the image
    """
    # Convert RGB color to BGR for OpenCV
    eyeliner_color = eyeliner_color_rgb[::-1]
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if results.multi_face_landmarks:
            h, w = image.shape[:2]
            overlay = image.copy()
            output = image.copy()
            
            for face_landmarks in results.multi_face_landmarks:
                # Use exact outer corner points
                LEFT_EYE_OUTER = 130
                RIGHT_EYE_OUTER = 359

                left_corner = (
                    int(face_landmarks.landmark[LEFT_EYE_OUTER].x * w),
                    int(face_landmarks.landmark[LEFT_EYE_OUTER].y * h)
                )
                right_corner = (
                    int(face_landmarks.landmark[RIGHT_EYE_OUTER].x * w),
                    int(face_landmarks.landmark[RIGHT_EYE_OUTER].y * h)
                )

                # Eyeliner settings
                max_thickness = 8
                min_thickness = 2
                eyeliner_length = int(w * 0.04)  # Slightly smaller wing

                def draw_smooth_eyeliner(start_point, direction='right', tilt_up=True):
                    points = []
                    for i in range(eyeliner_length):
                        alpha = i / eyeliner_length
                        x = start_point[0] + (i if direction == 'right' else -i)
                        y = start_point[1] - (i * 0.2 if tilt_up else 0)
                        points.append((int(x), int(y)))

                    points = np.array(points, np.int32)
                    points = points.reshape((-1, 1, 2))
                    
                    # Smooth, anti-aliased polyline
                    for i in range(1, len(points)):
                        thickness = int(max_thickness * (1 - i / len(points)) + min_thickness * (i / len(points)))
                        cv2.line(overlay, tuple(points[i-1][0]), tuple(points[i][0]), 
                                eyeliner_color, thickness, lineType=cv2.LINE_AA)

                # Draw left eye wing (outward)
                draw_smooth_eyeliner(left_corner, direction='left', tilt_up=True)
                # Draw right eye wing (outward)
                draw_smooth_eyeliner(right_corner, direction='right', tilt_up=True)

            # Blend the overlay with the original image
            cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
            return output
            
        else:
            raise ValueError("No face detected in the image")

# Example usage:
# image = cv2.imread("face.jpg")
# result = apply_eyeliner(image, [0, 0, 0])  # Black eyeliner in RGB
# cv2.imwrite("result.jpg", result)