import argparse
import cv2
import time
import os
import threading
from datetime import datetime
from collections import defaultdict, deque

from flask import Flask, jsonify, request, Response, send_from_directory

from stage1_fast import stage1_is_fight
from stage2_static import analyze_clip

# Optional: YOLOv8 for fire/weapons (runs only if weights are provided)
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


# -------------------- STORAGE --------------------

SAVE_DIR = "captured_clips"
os.makedirs(SAVE_DIR, exist_ok=True)

latest_frame = None
frame_lock = threading.Lock()

# -------------------- GLOBAL LOGS --------------------
# Keep these simple so Flutter can consume easily.

FIGHT_LOG = []   # [{filename, timestamp, probability}]
EVENT_LOG = []   # [{type, filename, timestamp, confidence, label}]

# -------------------- TUNABLE PARAMS (editable from Flutter via /config) --------------------

# Fight
STAGE1_ELASTICITY_SECONDS = 5.0     # stage1 suspicious time before recording
STAGE2_THRESHOLD = 0.30            # stage2 classifier threshold

# Fire/Weapon
FIRE_ALERT_THR = 0.50

YOLO_EVERY_N_WEAPON = 5
YOLO_EVERY_N_FIRE = 5

WEAPON_DEFAULT_THR = 0.80
WEAPON_ALERT_THR = {
    "Handgun": 0.80,
    "Knife": 0.75,
}

NEED_WEAPON = 3
M_WEAPON = 10

NEED_FIRE = 5
M_FIRE = 10


# -------------------- FLASK SERVER --------------------

app = Flask(__name__)

@app.route('/fight_detected', methods=['GET'])
def fight_detected():
    return jsonify(FIGHT_LOG)

@app.route('/events', methods=['GET'])
def events():
    # fire/weapon (and potentially others)
    return jsonify(EVENT_LOG)

@app.route('/videos/<path:filename>')
def get_video(filename):
    safe_name = os.path.basename(filename)
    return send_from_directory(SAVE_DIR, safe_name)

@app.route("/thumb/<path:filename>")
def thumb(filename):
    safe_name = os.path.basename(filename)
    path = os.path.join(SAVE_DIR, safe_name)
    if not os.path.exists(path):
        return jsonify({"error": "not found"}), 404

    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        return jsonify({"error": "cannot read frame"}), 500

    frame = cv2.resize(frame, (320, 180))  # CCTV-ish size
    ok, jpg = cv2.imencode(".jpg", frame)
    if not ok:
        return jsonify({"error": "encode failed"}), 500

    return Response(jpg.tobytes(), mimetype="image/jpeg")

@app.route("/history", methods=["GET"])
def history():
    # Return newest first; include best-effort metadata from logs.
    files = []
    if os.path.exists(SAVE_DIR):
        for f in os.listdir(SAVE_DIR):
            if f.lower().endswith((".avi", ".mp4", ".mov")):
                files.append(f)

    files.sort(reverse=True)

    # Build quick lookup from logs
    fight_by_file = {e.get("filename"): e for e in FIGHT_LOG}
    event_by_file = {e.get("filename"): e for e in EVENT_LOG}

    out = []
    for f in files:
        if f in fight_by_file:
            e = fight_by_file[f]
            out.append({
                "filename": f,
                "timestamp": e.get("timestamp", f.split(".")[0]),
                "probability": e.get("probability", 0.0),
                "type": "fight",
                "label": "fight",
                "confidence": e.get("probability", 0.0),
            })
        elif f in event_by_file:
            e = event_by_file[f]
            out.append({
                "filename": f,
                "timestamp": e.get("timestamp", f.split(".")[0]),
                "probability": e.get("confidence", 0.0),
                "type": e.get("type", "event"),
                "label": e.get("label", ""),
                "confidence": e.get("confidence", 0.0),
            })
        else:
            out.append({
                "filename": f,
                "timestamp": f.split(".")[0],
                "probability": 0.0,
                "type": "unknown",
                "label": "",
                "confidence": 0.0,
            })

    return jsonify(out)

@app.route("/delete/<path:filename>", methods=["DELETE"])
def delete_video(filename):
    global FIGHT_LOG, EVENT_LOG

    safe_name = os.path.basename(filename)
    video_path = os.path.join(SAVE_DIR, safe_name)

    if not os.path.exists(video_path):
        return jsonify({"error": "not found"}), 404

    try:
        os.remove(video_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # remove from logs so analytics updates
    FIGHT_LOG = [e for e in FIGHT_LOG if e.get("filename") != safe_name]
    EVENT_LOG = [e for e in EVENT_LOG if e.get("filename") != safe_name]

    return jsonify({"deleted": safe_name})


@app.route("/config", methods=["GET"])
def get_config():
    return jsonify({
        "stage1_elasticity_seconds": STAGE1_ELASTICITY_SECONDS,
        "stage2_threshold": STAGE2_THRESHOLD,

        "fire_alert_thr": FIRE_ALERT_THR,
        "weapon_default_thr": WEAPON_DEFAULT_THR,
        "weapon_class_thr": WEAPON_ALERT_THR,

        "yolo_every_n_weapon": YOLO_EVERY_N_WEAPON,
        "yolo_every_n_fire": YOLO_EVERY_N_FIRE,

        "need_weapon": NEED_WEAPON,
        "m_weapon": M_WEAPON,
        "need_fire": NEED_FIRE,
        "m_fire": M_FIRE,
    })

@app.route("/config", methods=["POST"])
def set_config():
    global STAGE1_ELASTICITY_SECONDS, STAGE2_THRESHOLD
    global FIRE_ALERT_THR
    global WEAPON_DEFAULT_THR, WEAPON_ALERT_THR
    global YOLO_EVERY_N_WEAPON, YOLO_EVERY_N_FIRE
    global NEED_WEAPON, M_WEAPON, NEED_FIRE, M_FIRE

    data = request.get_json(silent=True) or {}

    if "stage1_elasticity_seconds" in data:
        STAGE1_ELASTICITY_SECONDS = float(data["stage1_elasticity_seconds"])

    if "stage2_threshold" in data:
        STAGE2_THRESHOLD = float(data["stage2_threshold"])

    if "fire_alert_thr" in data:
        FIRE_ALERT_THR = float(data["fire_alert_thr"])

    if "weapon_default_thr" in data:
        WEAPON_DEFAULT_THR = float(data["weapon_default_thr"])

    if "weapon_class_thr" in data and isinstance(data["weapon_class_thr"], dict):
        for k, v in data["weapon_class_thr"].items():
            WEAPON_ALERT_THR[str(k)] = float(v)

    if "yolo_every_n_weapon" in data:
        YOLO_EVERY_N_WEAPON = int(data["yolo_every_n_weapon"])

    if "yolo_every_n_fire" in data:
        YOLO_EVERY_N_FIRE = int(data["yolo_every_n_fire"])

    if "need_weapon" in data:
        NEED_WEAPON = int(data["need_weapon"])
    if "m_weapon" in data:
        M_WEAPON = int(data["m_weapon"])

    if "need_fire" in data:
        NEED_FIRE = int(data["need_fire"])
    if "m_fire" in data:
        M_FIRE = int(data["m_fire"])

    return get_config()

@app.route("/live_feed")
def live_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route("/analytics", methods=["GET"])
def analytics():
    per_day = defaultdict(int)
    for event in FIGHT_LOG:
        ts = event.get("timestamp")
        if not ts:
            continue
        day = ts.split("T")[0]
        per_day[day] += 1
    return jsonify({"per_day": per_day})


def start_server():
    # threaded=True lets Flask handle multiple requests while video loop runs
    app.run(host="0.0.0.0", port=5000, threaded=True)


# -------------------- HELPERS --------------------

def record_clip(cap, seconds=5):
    """Record a short clip from an already-open cv2.VideoCapture."""
    name = datetime.now().strftime("%Y%m%d_%H%M%S") + ".avi"
    path = os.path.join(SAVE_DIR, name)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(path, fourcc, 20, (640, 480))

    start = time.time()
    while time.time() - start < seconds:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))
        out.write(frame)

    out.release()
    return name  # filename only


def stage2_worker(video_filename):
    """Fight verification on a saved clip."""
    video_path = os.path.join(SAVE_DIR, video_filename)
    is_fight, prob = analyze_clip(video_path, threshold=STAGE2_THRESHOLD)

    if is_fight:
        print(f"🚨 FIGHT CONFIRMED ({prob:.2f})")
        FIGHT_LOG.append({
            "filename": os.path.basename(video_path),
            "timestamp": datetime.now().isoformat(),
            "probability": round(float(prob), 2)
        })
    else:
        print(f"❌ FALSE ALARM ({prob:.2f})")


def generate_frames():
    """MJPEG stream generator from latest_frame."""
    global latest_frame
    while True:
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()

        if frame is None:
            time.sleep(0.05)
            continue

        frame = cv2.resize(frame, (640, 480))
        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            time.sleep(0.02)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )


def _yolo_detect_any(model, frame_bgr):
    """
    Returns list of detections as tuples: (label, conf, (x1,y1,x2,y2))
    """
    dets = []
    if model is None:
        return dets

    # ultralytics expects BGR fine; we'll keep as is.
    results = model(frame_bgr, verbose=False)
    if not results:
        return dets

    r = results[0]
    names = getattr(r, "names", None) or getattr(model, "names", None) or {}
    boxes = getattr(r, "boxes", None)
    if boxes is None:
        return dets

    for b in boxes:
        try:
            conf = float(b.conf.item())
            cls = int(b.cls.item())
            xyxy = b.xyxy[0].tolist()
        except Exception:
            continue

        label = str(names.get(cls, cls))
        x1, y1, x2, y2 = map(int, xyxy)
        dets.append((label, conf, (x1, y1, x2, y2)))
    return dets


def live_detection(fire_weights=None, weapon_weights=None, show_window=True):
    global latest_frame

    cap = cv2.VideoCapture(0)

    # Fight stage1 timer state
    fight_start = None
    recording_fight = False

    # YOLO models (optional)
    fire_model = None
    weapon_model = None
    if YOLO is None and (fire_weights or weapon_weights):
        print("[WARN] ultralytics not installed. Fire/weapon detection disabled.")
    else:
        if fire_weights:
            fire_model = YOLO(fire_weights)
            print(f"[INFO] Loaded fire model: {fire_weights}")
        if weapon_weights:
            weapon_model = YOLO(weapon_weights)
            print(f"[INFO] Loaded weapon model: {weapon_weights}")

    # Fire/weapon debouncing windows (sliding)
    fire_window = deque(maxlen=M_FIRE)
    weapon_window = deque(maxlen=M_WEAPON)

    frame_i = 0
    last_fire_alert = 0.0
    last_weapon_alert = 0.0
    ALERT_COOLDOWN_SEC = 3.0  # prevent spamming recordings

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_i += 1

        # Keep latest frame for MJPEG
        with frame_lock:
            latest_frame = frame.copy()

        # -------------------- FIGHT DETECTION --------------------
        suspicious = stage1_is_fight(frame)
        elapsed = 0.0

        if suspicious:
            if fight_start is None:
                fight_start = time.time()
            elapsed = time.time() - fight_start

            if elapsed >= STAGE1_ELASTICITY_SECONDS and not recording_fight:
                recording_fight = True
                clip_name = record_clip(cap)

                threading.Thread(
                    target=stage2_worker,
                    args=(clip_name,),
                    daemon=True
                ).start()

                fight_start = None
                recording_fight = False
        else:
            fight_start = None

        # -------------------- FIRE DETECTION (YOLO) --------------------
        fire_active = False
        fire_best = ("", 0.0)
        fire_boxes = []

        if fire_model is not None and (frame_i % max(1, YOLO_EVERY_N_FIRE) == 0):
            dets = _yolo_detect_any(fire_model, frame)
            # treat any label as fire candidate (your fire model should only output fire/smoke anyway)
            best = None
            for label, conf, box in dets:
                if conf >= FIRE_ALERT_THR:
                    if best is None or conf > best[1]:
                        best = (label, conf)
                    fire_boxes.append((label, conf, box))
            fire_window.append(1 if best is not None else 0)
            fire_active = (sum(fire_window) >= NEED_FIRE)
            if best is not None:
                fire_best = best
        else:
            # keep window stability
            if fire_model is not None:
                fire_window.append(0)
                fire_active = (sum(fire_window) >= NEED_FIRE)

        # -------------------- WEAPON DETECTION (YOLO) --------------------
        weapon_active = False
        weapon_best = ("", 0.0)
        weapon_boxes = []

        if weapon_model is not None and (frame_i % max(1, YOLO_EVERY_N_WEAPON) == 0):
            dets = _yolo_detect_any(weapon_model, frame)
            best = None
            for label, conf, box in dets:
                thr = float(WEAPON_ALERT_THR.get(label, WEAPON_DEFAULT_THR))
                if conf >= thr:
                    if best is None or conf > best[1]:
                        best = (label, conf)
                    weapon_boxes.append((label, conf, box))

            weapon_window.append(1 if best is not None else 0)
            weapon_active = (sum(weapon_window) >= NEED_WEAPON)
            if best is not None:
                weapon_best = best
        else:
            if weapon_model is not None:
                weapon_window.append(0)
                weapon_active = (sum(weapon_window) >= NEED_WEAPON)

        # -------------------- ALERT ACTIONS (optional recording) --------------------
        now = time.time()

        if fire_active and (now - last_fire_alert) > ALERT_COOLDOWN_SEC:
            last_fire_alert = now
            clip_name = record_clip(cap)
            EVENT_LOG.append({
                "type": "fire",
                "filename": clip_name,
                "timestamp": datetime.now().isoformat(),
                "confidence": round(float(fire_best[1]), 2),
                "label": str(fire_best[0]),
            })
            print(f"🔥 FIRE ALERT ({fire_best[0]} {fire_best[1]:.2f}) saved {clip_name}")

        if weapon_active and (now - last_weapon_alert) > ALERT_COOLDOWN_SEC:
            last_weapon_alert = now
            clip_name = record_clip(cap)
            EVENT_LOG.append({
                "type": "weapon",
                "filename": clip_name,
                "timestamp": datetime.now().isoformat(),
                "confidence": round(float(weapon_best[1]), 2),
                "label": str(weapon_best[0]),
            })
            print(f"🔫 WEAPON ALERT ({weapon_best[0]} {weapon_best[1]:.2f}) saved {clip_name}")

        # -------------------- DRAW OVERLAY --------------------
        status_lines = []
        status_lines.append(("MONITORING", (0, 255, 0)))

        if suspicious:
            status_lines.append((f"FIGHT: suspicious {elapsed:.1f}s", (0, 255, 255)))

        if fire_model is not None:
            if fire_active:
                status_lines.append((f"FIRE DETECTED ({fire_best[0]} {fire_best[1]:.2f})", (0, 0, 255)))
            else:
                status_lines.append((f"Fire thr={FIRE_ALERT_THR:.2f}", (200, 200, 200)))

        if weapon_model is not None:
            if weapon_active:
                status_lines.append((f"WEAPON DETECTED ({weapon_best[0]} {weapon_best[1]:.2f})", (0, 0, 255)))
            else:
                status_lines.append((f"Weapon def thr={WEAPON_DEFAULT_THR:.2f}", (200, 200, 200)))

        y = 30
        for line, color in status_lines:
            cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y += 26

        # draw boxes (only when we ran YOLO)
        for label, conf, (x1, y1, x2, y2) in fire_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        for label, conf, (x1, y1, x2, y2) in weapon_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if show_window:
            cv2.imshow("Hybrid Detection (Fight + Fire + Weapons)", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                print("[INFO] Closed")
                break

            if cv2.getWindowProperty("Hybrid Detection (Fight + Fire + Weapons)", cv2.WND_PROP_VISIBLE) < 1:
                print("[INFO] Window closed using X button")
                break

    cap.release()
    if show_window:
        cv2.destroyAllWindows()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fire-weights", default=None, help="Path to YOLO fire model weights (.pt)")
    ap.add_argument("--weapon-weights", default=None, help="Path to YOLO weapon model weights (.pt)")
    ap.add_argument("--no-window", action="store_true", help="Run headless (no OpenCV imshow window)")
    ap.add_argument("--port", type=int, default=5000)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Update server port if needed
    # (Simple approach: overwrite start_server closure by capturing args.port)
    def start_server_with_port():
        app.run(host="0.0.0.0", port=args.port, threaded=True)

    threading.Thread(target=start_server_with_port, daemon=True).start()

    # Start camera + AI in main thread
    live_detection(
        fire_weights=args.fire_weights,
        weapon_weights=args.weapon_weights,
        show_window=(not args.no_window),
    )
