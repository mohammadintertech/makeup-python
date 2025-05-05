import cv2
import mediapipe as mp
import numpy as np
import requests

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True,
                                  refine_landmarks=True,
                                  max_num_faces=1,
                                  min_detection_confidence=0.5)

def apply_eyeshadow(file: bytes, r: int, g: int, b: int, transparency_factor: float) -> np.ndarray:
    # Load image from bytes
    img_array = np.asarray(bytearray(file), dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    h, w = image.shape[:2]

    # Process image with MediaPipe
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        output = image.copy()

        for face_landmarks in results.multi_face_landmarks:
            # Define landmark groups for eyes and brows
            LEFT_EYE = [33, 246, 161, 160, 159, 158, 157, 173, 133]
            LEFT_BROW = [70, 63, 105, 66, 107, 55, 65, 52]
            RIGHT_EYE = [362, 398, 384, 385, 386, 387, 388, 466, 263]
            RIGHT_BROW = [336, 296, 334, 293, 300, 276, 283, 282]

            def get_landmark_points(indices):
                return np.array([
                    (int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h))
                    for i in indices
                ], dtype=np.int32)

            left_eye = get_landmark_points(LEFT_EYE)
            left_brow = get_landmark_points(LEFT_BROW)
            right_eye = get_landmark_points(RIGHT_EYE)
            right_brow = get_landmark_points(RIGHT_BROW)

            def create_eyeshadow_area(eye_points, brow_points):
                lower_brow_y = min(brow_points, key=lambda x: x[1])[1] + 5  # Adjust slightly below the brow
                adjusted_brow_points = [point for point in brow_points if point[1] > lower_brow_y]
                all_points = np.vstack((eye_points, adjusted_brow_points))
                return cv2.convexHull(all_points)

            left_area = create_eyeshadow_area(left_eye, left_brow)
            right_area = create_eyeshadow_area(right_eye, right_brow)

            shadow_overlay = np.zeros_like(image)

            for area in [left_area, right_area]:
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillConvexPoly(mask, area, 255)
                mask_blur = cv2.GaussianBlur(mask, (51, 51), 0)
                mask_normalized = mask_blur / 255.0

                for c in range(3):
                    shadow_overlay[:, :, c] = np.clip(
                        shadow_overlay[:, :, c] + mask_normalized * (b * transparency_factor, g * transparency_factor, r * transparency_factor)[c],
                        0, 255
                    ).astype(np.uint8)

            output = cv2.addWeighted(output, 1.0, shadow_overlay, 0.7, 0)

        return output
    else:
        raise HTTPException(status_code=400, detail="No face detected in the image")

