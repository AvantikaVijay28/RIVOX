
from ultralytics import YOLO
import cv2

# ---------------- Paths ----------------
WEAPON_MODEL_PATH = "weapon_weight.pt"   # handgun + knife
FIRE_MODEL_PATH   = "fire_model.pt"      # fire-only (class 'fire')

# ---------------- Performance knobs ----------------
IMG_SIZE = 640
CONF_SHOW = 0.20           # show boxes above this
EVERY_N_WEAPON = 3         # run weapon YOLO every N frames
EVERY_N_FIRE = 3           # run fire YOLO every N frames

# ---------------- Weapon thresholds ----------------
WEAPON_ALERT_THR = {
    "Handgun": 0.80,
    "Knife": 0.55,
}
WEAPON_DEFAULT_THR = 0.80  # fallback

# confirmation window for each weapon class
M_WEAPON = 10
NEED_WEAPON = 3

# ---------------- Fire threshold ----------------
FIRE_ALERT_THR = 0.65

# confirmation window for fire
M_FIRE = 10
NEED_FIRE = 8


def norm(s: str) -> str:
    return (s or "").strip().lower()


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def main():
    # ---- Load models (CUDA) ----
    weapon_model = YOLO(WEAPON_MODEL_PATH)
    weapon_model.to("cuda")
    try:
        weapon_model.fuse()
    except Exception:
        pass

    fire_model = YOLO(FIRE_MODEL_PATH)
    fire_model.to("cuda")
    try:
        fire_model.fuse()
    except Exception:
        pass

    print("Weapon model classes:", weapon_model.names)
    print("Fire model classes:", fire_model.names)

    # ---- Camera ----
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Webcam not found")

    # ---- Warmup (removes first-lag spike) ----
    ok, frame = cap.read()
    if ok and frame is not None:
        weapon_model.predict(frame, imgsz=IMG_SIZE, conf=CONF_SHOW, device=0, half=True, verbose=False)
        fire_model.predict(frame, imgsz=IMG_SIZE, conf=CONF_SHOW, device=0, half=True, verbose=False)

    # ---- Confirmation histories ----
    # Weapon per class
    weapon_hist = {name: [] for name in WEAPON_ALERT_THR.keys()}
    # Fire
    fire_hist = []

    # ---- Last results for drawing ----
    last_weapon_result = None
    last_fire_result = None

    frame_i = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        frame_i += 1

        # ---------------- WEAPON INFERENCE ----------------
        if frame_i % EVERY_N_WEAPON == 0:
            preds = weapon_model.predict(
                frame,
                imgsz=IMG_SIZE,
                conf=CONF_SHOW,
                device=0,
                half=True,
                verbose=False
            )
            last_weapon_result = preds[0]

            hits_this = {k: 0 for k in weapon_hist.keys()}

            if last_weapon_result.boxes is not None and len(last_weapon_result.boxes) > 0:
                for b in last_weapon_result.boxes:
                    c = float(b.conf[0])
                    cls = int(b.cls[0])
                    name = weapon_model.names.get(cls, str(cls))

                    thr = WEAPON_ALERT_THR.get(name, WEAPON_DEFAULT_THR)
                    if name in hits_this and c >= thr:
                        hits_this[name] = 1

            for k in weapon_hist.keys():
                weapon_hist[k].append(hits_this.get(k, 0))
                weapon_hist[k] = weapon_hist[k][-M_WEAPON:]

        # ---------------- FIRE INFERENCE ----------------
        if frame_i % EVERY_N_FIRE == 0:
            preds = fire_model.predict(
                frame,
                imgsz=IMG_SIZE,
                conf=CONF_SHOW,
                device=0,
                half=True,
                verbose=False
            )
            last_fire_result = preds[0]

            fire_hit = False
            if last_fire_result.boxes is not None and len(last_fire_result.boxes) > 0:
                for b in last_fire_result.boxes:
                    c = float(b.conf[0])
                    cls = int(b.cls[0])
                    name = norm(fire_model.names.get(cls, ""))

                    # fire-only model: confirm only 'fire'
                    if name == "fire" and c >= FIRE_ALERT_THR:
                        fire_hit = True
                        break

            fire_hist.append(1 if fire_hit else 0)
            fire_hist = fire_hist[-M_FIRE:]

        # ---------------- CONFIRMATION STATUS ----------------
        confirmed_weapons = [k for k, w in weapon_hist.items() if sum(w) >= NEED_WEAPON]
        confirmed_fire = sum(fire_hist) >= NEED_FIRE

        # ---------------- DRAWING ----------------
        view = frame.copy()

        # draw weapon boxes (red-ish)
        if last_weapon_result is not None:
            view = last_weapon_result.plot(img=view)

        # draw fire boxes (will overlay; YOLO plot draws its own)
        if last_fire_result is not None:
            view = last_fire_result.plot(img=view)

        # status text
        y = 40
        if confirmed_fire:
            cv2.putText(
                view, "CONFIRMED FIRE", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2
            )
            y += 40

        if confirmed_weapons:
            label = " + ".join(confirmed_weapons)
            cv2.putText(
                view, f"CONFIRMED WEAPON - {label}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2
            )
            y += 40

        cv2.imshow("Fire + Weapons (CUDA)", view)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
