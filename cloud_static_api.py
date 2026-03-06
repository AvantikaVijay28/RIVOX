import os
import uuid
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from stage2_static import analyze_clip  # re-use your existing model pipeline

app = Flask(__name__)

# -------------------- CONFIG --------------------
UPLOAD_DIR = "cloud_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXT = {".mp4", ".avi", ".mov", ".mkv"}
STAGE2_THRESHOLD = 0.50

# In-memory store for demo (use DB later if you want persistence)
UPLOAD_LOG = []  # [{id, filename, saved_as, timestamp, fight, prob}]


def _allowed_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXT


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "service": "cloud_static_api"})


@app.route("/upload", methods=["POST"])
def upload_video():
    """
    Upload a video -> run stage2_static fight model -> return JSON result
    Multipart form-data: file=<video>
    """
    if "file" not in request.files:
        return jsonify({"error": "Missing 'file' field"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    original = secure_filename(f.filename)
    if not _allowed_file(original):
        return jsonify({"error": f"Unsupported format. Allowed: {sorted(ALLOWED_EXT)}"}), 400

    # Save with unique name to avoid collisions
    vid_id = str(uuid.uuid4())[:8]
    ext = os.path.splitext(original)[1].lower()
    saved_as = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{vid_id}{ext}"
    save_path = os.path.join(UPLOAD_DIR, saved_as)
    f.save(save_path)

    # Run your existing Stage-2 model on the uploaded video
    is_fight, prob = analyze_clip(save_path, threshold=STAGE2_THRESHOLD)

    entry = {
        "id": vid_id,
        "original_filename": original,
        "saved_as": saved_as,
        "timestamp": datetime.now().isoformat(),
        "fight": bool(is_fight),
        "probability": round(float(prob), 4),
    }
    UPLOAD_LOG.append(entry)

    return jsonify(entry)


@app.route("/uploads", methods=["GET"])
def list_uploads():
    """
    Returns list of processed uploads (latest first)
    """
    return jsonify(list(reversed(UPLOAD_LOG)))


@app.route("/videos/<path:filename>", methods=["GET"])
def get_video(filename):
    """
    Download/stream the stored uploaded video
    """
    safe = os.path.basename(filename)
    return send_from_directory(UPLOAD_DIR, safe, as_attachment=False)


@app.route("/config", methods=["GET", "POST"])
def config():
    """
    Allow threshold tuning remotely (optional)
    """
    global STAGE2_THRESHOLD
    if request.method == "GET":
        return jsonify({"stage2_threshold": STAGE2_THRESHOLD})

    data = request.get_json(silent=True) or {}
    if "stage2_threshold" in data:
        STAGE2_THRESHOLD = float(data["stage2_threshold"])
    return jsonify({"ok": True, "stage2_threshold": STAGE2_THRESHOLD})


if __name__ == "__main__":
    # Cloud server will bind 0.0.0.0
    app.run(host="0.0.0.0", port=5001, debug=False)