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

# Function to apply blusher
def apply_blusher(file: bytes, r: int, g: int, b: int, intensity_factor: float):
    # Set the blusher color directly based on the RGB input
    blusher_color = np.array([b, g, r], dtype=np.float32)  # BGR format for OpenCV

    # Decode the image from the byte content (instead of requesting from URL)
    image_array = np.asarray(bytearray(file), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    h, w = image.shape[:2]
    image_float = image.astype(np.float32) / 255.0

    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        output = image_float.copy()

        for face_landmarks in results.multi_face_landmarks:
            def get_landmark_point(index):
                return int(face_landmarks.landmark[index].x * w), int(face_landmarks.landmark[index].y * h)

            # Key landmarks for positioning
            left_cheek_ref = get_landmark_point(123)
            right_cheek_ref = get_landmark_point(352)

            # Calculate circle parameters
            circle_radius = int(w * 0.15)

            # Position circles slightly higher for natural look
            left_center = (left_cheek_ref[0] + int(w * 0.06), left_cheek_ref[1] - int(h * 0.01))
            right_center = (right_cheek_ref[0] - int(w * 0.06), right_cheek_ref[1] - int(h * 0.01))

            # Create professional-grade mask with tapered edges
            def create_circular_mask(center, radius):
                mask = np.zeros((h, w), dtype=np.float32)
                cv2.circle(mask, center, radius, 1, -1)

                # Create tapered edges (like real makeup brushes)
                dist = cv2.distanceTransform((mask * 255).astype(np.uint8), cv2.DIST_L2, 5)
                dist = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
                mask = np.power(dist, 1.5)  # More natural falloff

                # Directional blur mimics real brush strokes
                mask = cv2.GaussianBlur(mask, (101, 101), 0)
                return mask

            left_mask = create_circular_mask(left_center, circle_radius)
            right_mask = create_circular_mask(right_center, circle_radius)
            combined_mask = np.maximum(left_mask, right_mask)

            # PROFESSIONAL COLOR BLENDING TECHNIQUE
            blusher_color_float = blusher_color / 255.0

            # Convert to YCrCb to preserve skin luminance
            ycrcb_image = cv2.cvtColor(output, cv2.COLOR_BGR2YCrCb)
            ycrcb_blusher = cv2.cvtColor(blusher_color_float.reshape(1, 1, 3), cv2.COLOR_BGR2YCrCb)

            # Blend chroma channels only (keeps natural skin texture)
            blended_ycrcb = ycrcb_image.copy()
            blended_ycrcb[..., 1] = ycrcb_image[..., 1] * (1 - intensity_factor * combined_mask * 2) + \
                                   ycrcb_blusher[0, 0, 1] * intensity_factor * combined_mask * 2
            blended_ycrcb[..., 2] = ycrcb_image[..., 2] * (1 - intensity_factor * combined_mask * 2) + \
                                   ycrcb_blusher[0, 0, 2] * intensity_factor * combined_mask * 2

            # Convert back to BGR
            blended_bgr = cv2.cvtColor(blended_ycrcb, cv2.COLOR_YCrCb2BGR)

            # Final blending with luminosity preservation
            output = output * (1 - combined_mask[..., np.newaxis]) + \
                    blended_bgr * combined_mask[..., np.newaxis]

        # Add subtle highlight effect
        highlight_mask = cv2.GaussianBlur(combined_mask, (151, 151), 0) * 0.15
        output = np.minimum(output + highlight_mask[..., np.newaxis] * np.array([0.1, 0.1, 0.1]), 1.0)

        # Convert back to 8-bit
        output = (output * 255).astype(np.uint8)

        return output
    else:
        return None

