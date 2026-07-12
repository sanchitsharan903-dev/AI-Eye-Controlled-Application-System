import cv2
import numpy as np


class HeadPoseEstimator:

    def __init__(self):
        pass

    def estimate(self, frame, face_landmarks):

        h, w, _ = frame.shape

        # -----------------------------
        # MediaPipe Landmark IDs
        # -----------------------------
        # Nose tip      -> 1
        # Chin          -> 199
        # Left eye      -> 33
        # Right eye     -> 263
        # Left mouth    -> 61
        # Right mouth   -> 291

        image_points = np.array([
            (face_landmarks.landmark[1].x * w, face_landmarks.landmark[1].y * h),
            (face_landmarks.landmark[199].x * w, face_landmarks.landmark[199].y * h),
            (face_landmarks.landmark[33].x * w, face_landmarks.landmark[33].y * h),
            (face_landmarks.landmark[263].x * w, face_landmarks.landmark[263].y * h),
            (face_landmarks.landmark[61].x * w, face_landmarks.landmark[61].y * h),
            (face_landmarks.landmark[291].x * w, face_landmarks.landmark[291].y * h),
        ], dtype="double")

        # Generic 3D face model
        model_points = np.array([
            (0.0, 0.0, 0.0),          # Nose
            (0.0, -63.6, -12.5),      # Chin
            (-43.3, 32.7, -26.0),     # Left eye
            (43.3, 32.7, -26.0),      # Right eye
            (-28.9, -28.9, -24.1),    # Left mouth
            (28.9, -28.9, -24.1)      # Right mouth
        ])

        focal_length = w
        center = (w / 2, h / 2)

        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return frame, {
                "pitch": 0,
                "yaw": 0,
                "roll": 0
            }

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        proj_matrix = np.hstack((rotation_matrix, translation_vector))

        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(
            proj_matrix
        )

        pitch = float(euler_angles[0][0])
        yaw = float(euler_angles[1][0])
        roll = float(euler_angles[2][0])

        # -----------------------------
        # Display values
        # -----------------------------

        cv2.putText(
            frame,
            f"Pitch : {pitch:.1f}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Yaw   : {yaw:.1f}",
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Roll  : {roll:.1f}",
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        head_pose = {
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll
        }

        return frame, head_pose