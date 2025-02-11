import cv2


def initialize_camera():
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 300)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)
    cap.set(cv2.CAP_PROP_FPS, 120)
    return cap
