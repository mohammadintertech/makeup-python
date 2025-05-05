import cv2
import numpy as np
import mediapipe as mp
# Function to apply eyeliner
def apply_eyeliner(file: bytes, r: int, g: int, b: int):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True,
                                      max_num_faces=1, min_detection_confidence=0.5)

    # Set the eyeliner color directly based on the RGB input
    eyeliner_color = (b, g, r)  # BGR format for OpenCV

    # Decode the image from the byte content (instead of requesting from URL)
    image_array = np.asarray(bytearray(file), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    h, w = image.shape[:2]

    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None

    overlay = image.copy()
    output = image.copy()
    alpha = 0.7

    for face_landmarks in results.multi_face_landmarks:
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

        max_thickness = 8
        min_thickness = 2
        eyeliner_length = int(w * 0.04)

        def draw_smooth_eyeliner(start_point, direction='right', tilt_up=True):
            points = []
            for i in range(eyeliner_length):
                x = start_point[0] + (i if direction == 'right' else -i)
                y = start_point[1] - (i * 0.2 if tilt_up else 0)
                points.append((int(x), int(y)))

            points = np.array(points, np.int32).reshape((-1, 1, 2))
            for i in range(1, len(points)):
                thickness = int(max_thickness * (1 - i / len(points)) + min_thickness * (i / len(points)))
                cv2.line(overlay, tuple(points[i - 1][0]), tuple(points[i][0]), eyeliner_color, thickness, lineType=cv2.LINE_AA)

        draw_smooth_eyeliner(left_corner, direction='left', tilt_up=True)
        draw_smooth_eyeliner(right_corner, direction='right', tilt_up=True)

    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    face_mesh.close()
    _, buffer = cv2.imencode('.jpg', output)
    return buffer.tobytes()

