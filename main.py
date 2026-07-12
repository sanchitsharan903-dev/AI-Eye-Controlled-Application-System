import cv2

from modules.camera import Camera
from modules.face_mesh import FaceMeshDetector


camera = Camera()

detector = FaceMeshDetector()


while True:

    frame = camera.get_frame()

    if frame is None:
        break

    frame = cv2.flip(frame,1)
    
    frame = detector.process(frame)

    cv2.imshow("AI Eye Controlled Application System", frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break


camera.release()

cv2.destroyAllWindows()