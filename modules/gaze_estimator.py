import cv2


class GazeEstimator:

    def __init__(self):

        # Smoothed gaze values
        self.horizontal = 0.5
        self.vertical = 0.5

        # Smoothing factor (0-1)
        self.alpha = 0.20

    def smooth(self, old, new):
        return old * (1 - self.alpha) + new * self.alpha

    def estimate(self, eye_data):

        left_iris = eye_data["left_iris"]
        right_iris = eye_data["right_iris"]

        left_eye_left = eye_data["left_eye_left"]
        left_eye_right = eye_data["left_eye_right"]

        right_eye_left = eye_data["right_eye_left"]
        right_eye_right = eye_data["right_eye_right"]

        left_top = eye_data["left_top"]
        left_bottom = eye_data["left_bottom"]

        right_top = eye_data["right_top"]
        right_bottom = eye_data["right_bottom"]

        # -----------------------------
        # Horizontal Ratio
        # -----------------------------

        left_width = max(1, left_eye_right[0] - left_eye_left[0])
        right_width = max(1, right_eye_right[0] - right_eye_left[0])

        left_ratio = (
            left_iris[0] - left_eye_left[0]
        ) / left_width

        right_ratio = (
            right_iris[0] - right_eye_left[0]
        ) / right_width

        horizontal = (left_ratio + right_ratio) / 2

        # -----------------------------
        # Vertical Ratio
        # -----------------------------

        left_height = max(1, left_bottom[1] - left_top[1])
        right_height = max(1, right_bottom[1] - right_top[1])

        left_v = (
            left_iris[1] - left_top[1]
        ) / left_height

        right_v = (
            right_iris[1] - right_top[1]
        ) / right_height

        vertical = (left_v + right_v) / 2

        # Clamp values

        horizontal = max(0.0, min(horizontal, 1.0))
        vertical = max(0.0, min(vertical, 1.0))

        # Smooth values

        self.horizontal = self.smooth(
            self.horizontal,
            horizontal
        )

        self.vertical = self.smooth(
            self.vertical,
            vertical
        )

        return self.horizontal, self.vertical

    def draw_debug(self, frame, horizontal, vertical):

        cv2.putText(
            frame,
            f"H : {horizontal:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"V : {vertical:.2f}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        return frame