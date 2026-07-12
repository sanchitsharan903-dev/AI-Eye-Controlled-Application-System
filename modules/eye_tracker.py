import cv2
import math

# ---------------- IRIS ---------------- #

LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

# ---------------- Eye Corners ---------------- #

LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133

RIGHT_EYE_LEFT = 362
RIGHT_EYE_RIGHT = 263

# ---------------- Eyelids ---------------- #

LEFT_TOP = 159
LEFT_BOTTOM = 145

RIGHT_TOP = 386
RIGHT_BOTTOM = 374


class EyeTracker:

    def __init__(self):
        pass

    def distance(self, p1, p2):
        return math.sqrt(
            (p1[0] - p2[0]) ** 2 +
            (p1[1] - p2[1]) ** 2
        )

    def draw_iris(self, frame, face_landmarks):

        h, w, _ = frame.shape

        # ================= LEFT IRIS ================= #

        left_points = []

        for idx in LEFT_IRIS:

            landmark = face_landmarks.landmark[idx]

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            left_points.append((x, y))

            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # ================= RIGHT IRIS ================= #

        right_points = []

        for idx in RIGHT_IRIS:

            landmark = face_landmarks.landmark[idx]

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            right_points.append((x, y))

            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # ================= IRIS CENTER ================= #

        lx = sum(p[0] for p in left_points) // len(left_points)
        ly = sum(p[1] for p in left_points) // len(left_points)

        rx = sum(p[0] for p in right_points) // len(right_points)
        ry = sum(p[1] for p in right_points) // len(right_points)

        left_iris = (lx, ly)
        right_iris = (rx, ry)

        cv2.circle(frame, left_iris, 5, (255, 0, 0), -1)
        cv2.circle(frame, right_iris, 5, (255, 0, 0), -1)

        # ================= LEFT EYE ================= #

        l_left = face_landmarks.landmark[LEFT_EYE_LEFT]
        l_right = face_landmarks.landmark[LEFT_EYE_RIGHT]
        l_top = face_landmarks.landmark[LEFT_TOP]
        l_bottom = face_landmarks.landmark[LEFT_BOTTOM]

        left_eye_left = (
            int(l_left.x * w),
            int(l_left.y * h)
        )

        left_eye_right = (
            int(l_right.x * w),
            int(l_right.y * h)
        )

        left_top = (
            int(l_top.x * w),
            int(l_top.y * h)
        )

        left_bottom = (
            int(l_bottom.x * w),
            int(l_bottom.y * h)
        )

        # ================= RIGHT EYE ================= #

        r_left = face_landmarks.landmark[RIGHT_EYE_LEFT]
        r_right = face_landmarks.landmark[RIGHT_EYE_RIGHT]
        r_top = face_landmarks.landmark[RIGHT_TOP]
        r_bottom = face_landmarks.landmark[RIGHT_BOTTOM]

        right_eye_left = (
            int(r_left.x * w),
            int(r_left.y * h)
        )

        right_eye_right = (
            int(r_right.x * w),
            int(r_right.y * h)
        )

        right_top = (
            int(r_top.x * w),
            int(r_top.y * h)
        )

        right_bottom = (
            int(r_bottom.x * w),
            int(r_bottom.y * h)
        )

        # ================= DRAW POINTS ================= #

        debug_points = [
            left_eye_left,
            left_eye_right,
            right_eye_left,
            right_eye_right,
            left_top,
            left_bottom,
            right_top,
            right_bottom,
        ]

        for point in debug_points:
            cv2.circle(frame, point, 3, (0, 0, 255), -1)

        # ================= MEASUREMENTS ================= #

        left_width = self.distance(
            left_eye_left,
            left_eye_right
        )

        right_width = self.distance(
            right_eye_left,
            right_eye_right
        )

        left_height = self.distance(
            left_top,
            left_bottom
        )

        right_height = self.distance(
            right_top,
            right_bottom
        )

        # ================= RETURN DATA ================= #

        eye_data = {

            "left_iris": left_iris,
            "right_iris": right_iris,

            "left_eye_left": left_eye_left,
            "left_eye_right": left_eye_right,

            "right_eye_left": right_eye_left,
            "right_eye_right": right_eye_right,

            "left_top": left_top,
            "left_bottom": left_bottom,

            "right_top": right_top,
            "right_bottom": right_bottom,

            "left_width": left_width,
            "right_width": right_width,

            "left_height": left_height,
            "right_height": right_height
        }

        return frame, eye_data