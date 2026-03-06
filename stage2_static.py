import cv2
import numpy as np
from tensorflow.keras.models import load_model

SEQ_LEN = 16
FRAME_SIZE = (64, 64)

model = load_model("fight_model.h5")


def analyze_clip(video_path, threshold):

    cap = cv2.VideoCapture(video_path)
    frames = []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total < SEQ_LEN:
        cap.release()
        return False, 0.0

    frame_ids = np.linspace(0, total-1, SEQ_LEN, dtype=int)

    for fid in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()

        if not ret:
            frame = np.zeros((64,64,3), dtype=np.uint8)

        frame = cv2.resize(frame, FRAME_SIZE)
        frame = frame.astype("float32") / 255.0
        frames.append(frame)

    cap.release()

    X = np.expand_dims(np.array(frames), axis=0)

    prob = model.predict(X)[0][0]
    is_fight = prob >= threshold

    return is_fight, float(prob)
