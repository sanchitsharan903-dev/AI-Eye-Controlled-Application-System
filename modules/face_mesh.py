import cv2
import mediapipe as mp

from modules.eye_tracker import EyeTracker
from modules.gaze_estimator import GazeEstimator
from modules.head_pose import HeadPoseEstimator


class FaceMeshDetector:

    def __init__(self):

        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.drawer = mp.solutions.drawing_utils

        self.eye_tracker = EyeTracker()
        self.gaze = GazeEstimator()
        self.head_pose = HeadPoseEstimator()

    def process(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb)

        if results.multi_face_landmarks:

            for face in results.multi_face_landmarks:

                # Draw Face Mesh
                self.drawer.draw_landmarks(
                    frame,
                    face,
                    self.mp_face_mesh.FACEMESH_TESSELATION
                )

                # Eye Tracking
                frame, eye_data = self.eye_tracker.draw_iris(
                    frame,
                    face
                )

                # Gaze Estimation
                horizontal, vertical = self.gaze.estimate(eye_data)

                frame = self.gaze.draw_debug(
                    frame,
                    horizontal,
                    vertical
                )

                # Head Pose Estimation
                frame, head_pose = self.head_pose.estimate(
                    frame,
                    face
                )

        return frame