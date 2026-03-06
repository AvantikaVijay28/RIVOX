import cv2
import numpy as np

prev_gray = None
motion_history = []

def stage1_is_fight(frame, threshold=2.0):
    """
    Returns True if aggressive motion detected
    """

    global prev_gray, motion_history

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    if prev_gray is None:
        prev_gray = gray
        return False

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray,
        None, 0.5, 3, 15, 3, 5, 1.2, 0
    )

    mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
    motion_score = np.mean(mag)

    motion_history.append(motion_score)
    if len(motion_history) > 10:
        motion_history.pop(0)

    prev_gray = gray

    avg_motion = np.mean(motion_history)

    return avg_motion > threshold
