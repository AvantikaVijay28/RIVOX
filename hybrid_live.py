import cv2
import time
import os
import threading
from datetime import datetime
from flask import Response, request
from collections import defaultdict

from flask import Flask, jsonify, send_from_directory

from stage1_fast import stage1_is_fight
from stage2_static import analyze_clip

# ✅ YOLO (Weapons + Fire)
from ultralytics import YOLO

import json
import requests
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

import bcrypt
from flask_jwt_extended import create_access_token

import re

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

def is_valid_email(email: str) -> bool:
    return bool(EMAIL_RE.match(email or ""))

def send_push_notification(title, body):
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE,
        scopes=["https://www.googleapis.com/auth/firebase.messaging"],
    )

    credentials.refresh(Request())
    access_token = credentials.token

    url = f"https://fcm.googleapis.com/v1/projects/{PROJECT_ID}/messages:send"

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    payload = {
        "message": {
            "token": DEVICE_FCM_TOKEN,
            "notification": {
                "title": title,
                "body": body,
            },
        }
    }

    response = requests.post(url, headers=headers, json=payload)

    print("Push response:", response.status_code, response.text)

# ==================== STORAGE ====================

SAVE_DIR = "captured_clips"
os.makedirs(SAVE_DIR, exist_ok=True)

latest_frame = None
frame_lock = threading.Lock()

SERVICE_ACCOUNT_FILE = "firebase_service_account.json"
PROJECT_ID = "fight-detection1"
DEVICE_FCM_TOKEN = "f6UInnDaT1ubxcwZ45JJxf:APA91bG9oX4lCE68VddVWRDZEFH5cgEV1_BGuQ3meAci1zv3Q6RSdiSU5FZ3SU9VkbLkwRoGDttZPpsiAHk3Bgrvy_nQ8CH7eieZbRXXz5g64tgcRDJEFv0"
# ==================== FLASK SERVER ====================

app = Flask(__name__)

#------------------------------DB-------------------------#
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager

import os
basedir = os.path.abspath(os.path.dirname(__file__))

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(basedir, "app.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["JWT_SECRET_KEY"] = "super-secret-change-this"
jwt = JWTManager(app)
db = SQLAlchemy(app)


from datetime import datetime

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    google_sub = db.Column(db.String(255), unique=True, nullable=True)

    name = db.Column(db.String(255))
    email = db.Column(db.String(255), unique=True, nullable=False)

    password_hash = db.Column(db.String(255), nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)



class Fight(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    clip_path = db.Column(db.String(255))
    fight_score = db.Column(db.Float)

    fire_detected = db.Column(db.Boolean, default=False)
    weapon_detected = db.Column(db.Boolean, default=False)

FIGHT_LOG = []

# ---- Tunable params (editable from Flutter) ----
STAGE1_ELASTICITY_SECONDS = 5.0     # was hardcoded "elapsed >= 2"
STAGE2_THRESHOLD = 0.50            # passed into analyze_clip()

# ---- Detection toggles ----
ENABLE_FIRE_DETECTION = False
ENABLE_WEAPON_DETECTION = False


@app.route('/fight_detected', methods=['GET'])
def fight_detected():
    return jsonify(FIGHT_LOG)

@app.route('/videos/<filename>')
def get_video(filename):
    return send_from_directory(SAVE_DIR, filename)

@app.route("/thumb/<filename>")
def thumb(filename):
    path = os.path.join(SAVE_DIR, filename)
    if not os.path.exists(path):
        return jsonify({"error": "not found"}), 404

    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        return jsonify({"error": "cannot read frame"}), 500

    frame = cv2.resize(frame, (320, 180))
    ok, jpg = cv2.imencode(".jpg", frame)
    if not ok:
        return jsonify({"error": "encode failed"}), 500

    return Response(jpg.tobytes(), mimetype="image/jpeg")

@app.route("/history", methods=["GET"])
def history():
    # 1) list all video files present
    files = []
    if os.path.exists(SAVE_DIR):
        for f in os.listdir(SAVE_DIR):
            if f.lower().endswith((".avi", ".mp4", ".mov")):
                files.append(f)
    files.sort(reverse=True)

    # 2) build a lookup from logs (so we can attach type/probability)
    log_map = {}
    for item in FIGHT_LOG:
        # item["filename"] is expected to be just the filename (not full path)
        log_map[item.get("filename")] = item

    # 3) return a unified list: always returns all files, but adds type when known
    out = []
    for f in files:
        ts = f.split(".")[0]
        if f in log_map:
            item = log_map[f]
            out.append({
                "filename": f,
                "timestamp": item.get("timestamp", ts),
                "probability": item.get("probability", 0.99),
                "type": item.get("type", "unknown"),
            })
        else:
            # file exists but not in log yet
            out.append({
                "filename": f,
                "timestamp": ts,
                "probability": 0.99,
                "type": "unknown",
            })

    return jsonify(out)


@app.route("/delete/<path:filename>", methods=["DELETE"])
def delete_video(filename):
    global FIGHT_LOG
    safe_name = os.path.basename(filename)
    video_path = os.path.join(SAVE_DIR, safe_name)

    if not os.path.exists(video_path):
        return jsonify({"error": "not found"}), 404

    try:
        os.remove(video_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    FIGHT_LOG = [e for e in FIGHT_LOG if e.get("filename") != safe_name]
    return jsonify({"deleted": safe_name})

@app.route("/config", methods=["GET"])
def get_config():
    return jsonify({
        "stage1_elasticity_seconds": STAGE1_ELASTICITY_SECONDS,
        "stage2_threshold": STAGE2_THRESHOLD,
        "fire_alert_thr": FIRE_ALERT_THR,
        "weapon_default_thr": WEAPON_DEFAULT_THR,
        "weapon_class_thr": WEAPON_ALERT_THR,
        "enable_fire_detection": ENABLE_FIRE_DETECTION,
        "enable_weapon_detection": ENABLE_WEAPON_DETECTION
    })

@app.route("/config", methods=["POST"])
def set_config():
    global STAGE1_ELASTICITY_SECONDS, STAGE2_THRESHOLD
    global FIRE_ALERT_THR, WEAPON_DEFAULT_THR, WEAPON_ALERT_THR
    global ENABLE_FIRE_DETECTION, ENABLE_WEAPON_DETECTION

    data = request.get_json(silent=True) or {}
    if "stage1_elasticity_seconds" in data:
        STAGE1_ELASTICITY_SECONDS = float(data["stage1_elasticity_seconds"])

    if "stage2_threshold" in data:
        STAGE2_THRESHOLD = float(data["stage2_threshold"])

    if "fire_alert_thr" in data:
        FIRE_ALERT_THR = float(data["fire_alert_thr"])

    if "weapon_default_thr" in data:
        WEAPON_DEFAULT_THR = float(data["weapon_default_thr"])

    if "weapon_class_thr" in data:
        WEAPON_ALERT_THR.update(data["weapon_class_thr"])

    if "enable_fire_detection" in data:
        ENABLE_FIRE_DETECTION = bool(data["enable_fire_detection"])

    if "enable_weapon_detection" in data:
        ENABLE_WEAPON_DETECTION = bool(data["enable_weapon_detection"])

    return jsonify({
        "ok": True,
        "stage1_elasticity_seconds": STAGE1_ELASTICITY_SECONDS,
        "stage2_threshold": STAGE2_THRESHOLD,
        "fire_alert_thr": FIRE_ALERT_THR,
        "weapon_default_thr": WEAPON_DEFAULT_THR,
        "weapon_class_thr": WEAPON_ALERT_THR,
        "enable_fire_detection": ENABLE_FIRE_DETECTION,
        "enable_weapon_detection": ENABLE_WEAPON_DETECTION
    })

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
    app.run(host="0.0.0.0", port=5000)

@app.route("/latest_event", methods=["GET"])
def latest_event():
    if not FIGHT_LOG:
        return jsonify({"event": None})

    latest = FIGHT_LOG[-1]
    return jsonify({"event": latest})

@app.route("/auth/signup", methods=["POST"])
def signup():
    data = request.get_json(silent=True) or {}

    name = data.get("name")
    email = data.get("email")
    password = data.get("password")

    if not name or not email or not password:
        return jsonify({"error": "Missing required fields"}), 400

    if not is_valid_email(email):
        return jsonify({"error": "Invalid email format"}), 400

    # Check if user already exists
    if User.query.filter_by(email=email).first():
        return jsonify({"error": "Email already registered"}), 400

    # Hash password
    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

    new_user = User(
        name=name,
        email=email,
        password_hash=hashed.decode("utf-8")
    )

    db.session.add(new_user)
    db.session.commit()

    access_token = create_access_token(identity=str(new_user.id))

    return jsonify({
        "access_token": access_token,
        "user": {
            "id": new_user.id,
            "name": new_user.name,
            "email": new_user.email
        }
    })

from flask_jwt_extended import jwt_required, get_jwt_identity

@app.route("/profile", methods=["GET"])
@jwt_required()
def get_profile():
    user_id = get_jwt_identity()

    user = User.query.get(int(user_id))

    if not user:
        return jsonify({"error": "User not found"}), 404

    return jsonify({
        "id": user.id,
        "email": user.email,
        "name": user.name,
    })

@app.route("/auth/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}

    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"error": "Missing email or password"}), 400

    user = User.query.filter_by(email=email).first()

    if not user:
        return jsonify({"error": "User not found"}), 401

    # Check hashed password using bcrypt
    if not bcrypt.checkpw(
        password.encode("utf-8"),
        user.password_hash.encode("utf-8")
    ):
        return jsonify({"error": "Invalid credentials"}), 401

    access_token = create_access_token(identity=str(user.id))

    return jsonify({
        "access_token": access_token,
        "user": {
            "id": user.id,
            "name": user.name,
            "email": user.email
        }
    }), 200


# ==================== FIGHT LOGIC ====================

def record_clip(cap, seconds=5):
    name = datetime.now().strftime("%Y%m%d_%H%M%S") + ".avi"
    path = os.path.join(SAVE_DIR, name)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(path, fourcc, 20, (640, 480))

    start = time.time()
    while time.time() - start < seconds:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    out.release()
    return name  # return filename only

def stage2_worker(video_filename):
    video_path = os.path.join(SAVE_DIR, video_filename)
    is_fight, prob = analyze_clip(video_path, threshold=STAGE2_THRESHOLD)

    if is_fight:
        print(f" FIGHT CONFIRMED ({prob:.2f})")
        FIGHT_LOG.append({
            "filename": os.path.basename(video_path),
            "timestamp": datetime.now().isoformat(),
            "probability": round(prob, 2),
            "type": "fight"
        })

        with app.app_context():
            new_fight = Fight(
                clip_path=os.path.basename(video_path),
                fight_score=round(prob, 2),
                fire_detected=False,
                weapon_detected=False
            )
            db.session.add(new_fight)
            db.session.commit()
            print("✅ Fight saved to SQL")

        send_push_notification(" Fight Detected", "A fight has been confirmed.")

    else:
        print(f"❌ FALSE ALARM ({prob:.2f})")

def generate_frames():
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
            time.sleep(0.05)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )

# ==================== YOLO WEAPON + FIRE ====================

WEAPON_MODEL_PATH = "weapon_weight.pt"
FIRE_MODEL_PATH = "fire_model.pt"

YOLO_IMG_SIZE = 640
YOLO_CONF_SHOW = 0.25

# run YOLO less frequently than fight stage
YOLO_EVERY_N_WEAPON = 5
YOLO_EVERY_N_FIRE = 5

WEAPON_ALERT_THR = {
    "Handgun": 0.80,
    "Knife": 0.75,
}
WEAPON_DEFAULT_THR = 0.80
M_WEAPON = 10
NEED_WEAPON = 3

FIRE_ALERT_THR = 0.50
M_FIRE = 10
NEED_FIRE = 5

def _norm(s: str) -> str:
    return (s or "").strip().lower()

def load_yolo_models():
    """
    Loads both YOLO models once (CUDA + fp16).
    If CUDA isn't available, it will still run (slower) on CPU.
    """
    weapon_model = YOLO(WEAPON_MODEL_PATH)
    fire_model = YOLO(FIRE_MODEL_PATH)

    # Prefer CUDA if available
    try:
        weapon_model.to("cuda")
        fire_model.to("cuda")
        print("cuda")

    except Exception:
        print("[WARN] Could not move YOLO models to CUDA. Running on CPU.")

    try:
        weapon_model.fuse()
    except Exception:
        pass
    try:
        fire_model.fuse()
    except Exception:
        pass

    print("Weapon model classes:", weapon_model.names)
    print("Fire model classes:", fire_model.names)

    return weapon_model, fire_model

# ==================== MAIN LIVE LOOP ====================

def live_detection():
    global latest_frame

    # Load YOLO once
    weapon_model, fire_model = load_yolo_models()

    # confirmation histories
    weapon_hist = {name: [] for name in WEAPON_ALERT_THR.keys()}
    fire_hist = []

    last_weapon_result = None
    last_fire_result = None

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Webcam not found")

    # warmup
    ok, frame = cap.read()
    if ok and frame is not None:
        try:
            weapon_model.predict(frame, imgsz=YOLO_IMG_SIZE, conf=YOLO_CONF_SHOW, device=0, half=True, verbose=False)
            fire_model.predict(frame, imgsz=YOLO_IMG_SIZE, conf=YOLO_CONF_SHOW, device=0, half=True, verbose=False)
        except Exception:
            # fallback if device/half not supported
            weapon_model.predict(frame, imgsz=YOLO_IMG_SIZE, conf=YOLO_CONF_SHOW, verbose=False)
            fire_model.predict(frame, imgsz=YOLO_IMG_SIZE, conf=YOLO_CONF_SHOW, verbose=False)

    fight_start = None
    recording = False
    frame_i = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_i += 1

        # ---------------- FIGHT STAGE 1 (every frame) ----------------
        # ---------------- FIGHT STAGE 1 ----------------
        fight_enabled = not (ENABLE_FIRE_DETECTION or ENABLE_WEAPON_DETECTION)

        if fight_enabled:
            suspicious = stage1_is_fight(frame)
        else:
            suspicious = False
            fight_start = None

        if suspicious:
            if fight_start is None:
                fight_start = time.time()
            elapsed = time.time() - fight_start

            if elapsed >= STAGE1_ELASTICITY_SECONDS and not recording:
                recording = True
                clip_name = record_clip(cap)

                threading.Thread(
                    target=stage2_worker,
                    args=(clip_name,),
                    daemon=True
                ).start()

                fight_start = None
                recording = False
        else:
            fight_start = None
            elapsed = 0.0

        # ---------------- YOLO: WEAPON (every N frames) ----------------
        if ENABLE_WEAPON_DETECTION and frame_i % YOLO_EVERY_N_WEAPON == 0:

            if not ENABLE_FIRE_DETECTION:
                fire_hist.clear()
                confirmed_fire = False

            try:
                preds = weapon_model.predict(
                    frame,
                    imgsz=YOLO_IMG_SIZE,
                    conf=YOLO_CONF_SHOW,
                    device=0,
                    half=True,
                    verbose=False
                )
            except Exception:
                preds = weapon_model.predict(
                    frame,
                    imgsz=YOLO_IMG_SIZE,
                    conf=YOLO_CONF_SHOW,
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

        # ---------------- YOLO: FIRE (every N frames) ----------------
        if ENABLE_FIRE_DETECTION and frame_i % YOLO_EVERY_N_FIRE == 0:

            if not ENABLE_WEAPON_DETECTION:
                for k in weapon_hist.keys():
                    weapon_hist[k].clear()
                confirmed_weapons = []

            try:
                preds = fire_model.predict(
                    frame,
                    imgsz=YOLO_IMG_SIZE,
                    conf=YOLO_CONF_SHOW,
                    device=0,
                    half=True,
                    verbose=False
                )
            except Exception:
                preds = fire_model.predict(
                    frame,
                    imgsz=YOLO_IMG_SIZE,
                    conf=YOLO_CONF_SHOW,
                    verbose=False
                )

            last_fire_result = preds[0]

            fire_hit = False
            if last_fire_result.boxes is not None and len(last_fire_result.boxes) > 0:
                for b in last_fire_result.boxes:
                    c = float(b.conf[0])
                    cls = int(b.cls[0])
                    name = _norm(fire_model.names.get(cls, ""))
                    if name == "fire" and c >= FIRE_ALERT_THR:
                        fire_hit = True
                        break

            fire_hist.append(1 if fire_hit else 0)
            fire_hist = fire_hist[-M_FIRE:]

        # ---------------- CONFIRMATION STATUS ----------------
        confirmed_weapons = [k for k, w in weapon_hist.items() if sum(w) >= NEED_WEAPON]
        confirmed_fire = sum(fire_hist) >= NEED_FIRE

        # ---------------- OVERLAY + STREAM FRAME ----------------
        view = frame.copy()

        # Draw YOLO boxes (overlay on display + live_feed)
        if last_weapon_result is not None:
            try:
                view = last_weapon_result.plot(img=view)
            except Exception:
                view = last_weapon_result.plot()
        if last_fire_result is not None:
            try:
                view = last_fire_result.plot(img=view)
            except Exception:
                view = last_fire_result.plot()

        # Fight overlay
        status = "MONITORING"
        color = (0, 255, 0)

        if suspicious:
            status = f"SUSPICIOUS {elapsed:.1f}s"
            color = (0, 255, 255)

        if recording:
            status = "RECORDING CLIP..."
            color = (0, 0, 255)
        if fight_enabled:
            cv2.putText(view, status, (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, color, 2)

        # Fire + Weapon overlay
        y = 70
        if ENABLE_FIRE_DETECTION and confirmed_fire:

                cv2.putText(view, "CONFIRMED FIRE", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                y += 40

                if not recording:
                    recording = True
                    clip_name = record_clip(cap)

                    FIGHT_LOG.append({
                            "filename": clip_name,
                            "timestamp": datetime.now().isoformat(),
                            "probability": 1.0,
                            "type": "fire"
                    })
                send_push_notification("Fire Detected", "Fire has been detected.")

        recording = False


        if ENABLE_WEAPON_DETECTION and confirmed_weapons:

                label = " + ".join(confirmed_weapons)
                cv2.putText(view, "CONFIRMED WEAPON", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                y += 40

                if not recording:
                    recording = True
                    clip_name = record_clip(cap)

                    FIGHT_LOG.append({
                            "filename": clip_name,
                            "timestamp": datetime.now().isoformat(),
                            "probability": 1.0,
                            "type": "weapon"
                    })
                send_push_notification("Weapon Detected", "A weapon has been detected.")

        recording = False


        # Update latest_frame for Flask stream (with overlays!)
        with frame_lock:
            latest_frame = view.copy()

        cv2.imshow("Hybrid Fight + Weapon + Fire", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("[INFO] Closed")
            break
        if cv2.getWindowProperty("Hybrid Fight + Weapon + Fire", cv2.WND_PROP_VISIBLE) < 1:
            print("[INFO] Window closed using X button")
            break

    cap.release()
    cv2.destroyAllWindows()

with app.app_context():
    db.create_all()

if __name__ == "__main__":
    # Start Flask server in background
    threading.Thread(target=start_server, daemon=True).start()

    # Start camera + AI in main thread
    live_detection()

