import cv2

LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]


class EyeTracker:

    def draw_iris(self, frame, face_landmarks):

        h, w, _ = frame.shape

        left_points = []
        right_points = []

        # LEFT IRIS
        for idx in LEFT_IRIS:

            landmark = face_landmarks.landmark[idx]

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            left_points.append((x, y))

            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # RIGHT IRIS
        for idx in RIGHT_IRIS:

            landmark = face_landmarks.landmark[idx]

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            right_points.append((x, y))

            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # LEFT CENTER
        lx = sum(p[0] for p in left_points) // len(left_points)
        ly = sum(p[1] for p in left_points) // len(left_points)

        # RIGHT CENTER
        rx = sum(p[0] for p in right_points) // len(right_points)
        ry = sum(p[1] for p in right_points) // len(right_points)

        # Draw center
        cv2.circle(frame, (lx, ly), 5, (255, 0, 0), -1)
        cv2.circle(frame, (rx, ry), 5, (255, 0, 0), -1)

        return frame