import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

def apply_lipstick(image: np.ndarray, lip_color: list, intensity_factor: float, edge_width: int) -> np.ndarray:
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

            LIPS_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]

            outer_lips = np.array([
                (int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h))
                for i in LIPS_OUTER
            ], dtype=np.int32)

            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [outer_lips], 255)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_width*2+1, edge_width*2+1))
            inner_mask = cv2.erode(mask, kernel)
            distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            edge_gradient = np.where(mask > 0, np.clip(distance / edge_width, 0, 1), 0)
            final_mask = np.maximum(inner_mask/255.0, edge_gradient)
            final_mask = cv2.GaussianBlur(final_mask, (0,0), sigmaX=edge_width/2)

            target_color_bgr = np.array(lip_color, dtype=np.float32) / 255.0
            target_color_hsv = cv2.cvtColor(np.array([[target_color_bgr]], dtype=np.float32), cv2.COLOR_BGR2HSV)[0,0]

            lips_hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)
            lips_hsv[..., 0] = lips_hsv[..., 0] * (1 - intensity_factor*final_mask) + target_color_hsv[0] * intensity_factor * final_mask
            lips_hsv[..., 1] = lips_hsv[..., 1] * (1 - intensity_factor*final_mask) + target_color_hsv[1] * intensity_factor * final_mask
            colored_lips = cv2.cvtColor(lips_hsv, cv2.COLOR_HSV2BGR)

            output = output * (1 - final_mask[..., np.newaxis]) + colored_lips * final_mask[..., np.newaxis]
            return (output * 255).astype(np.uint8)

        else:
            raise ValueError("No face detected")
