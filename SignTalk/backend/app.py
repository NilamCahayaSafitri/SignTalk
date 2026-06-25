from flask import Flask, render_template, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import joblib
import base64
import torch
import torch.nn as nn
from collections import deque, Counter
import warnings
import os

warnings.filterwarnings("ignore")

app = Flask(
    __name__,
    template_folder="../frontend/templates",
    static_folder="../frontend/static"
)

BASE_DIR = os.path.dirname(__file__)
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================
# ARSITEKTUR BI-GRU (Wajib sama dengan saat training)
# ==============================================================
class IsyaratBiGRU(nn.Module):
    def __init__(self, input_size=104, hidden_size=128, num_layers=2, num_classes=16):
        super(IsyaratBiGRU, self).__init__()
        self.bigru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        out, _ = self.bigru(x)
        out = self.fc(out[:, -1, :])
        return out


# ==============================================================
# LOAD MODEL
# ==============================================================
try:
    model_huruf = joblib.load(os.path.join(BASE_DIR, "model_huruf_indo.pkl"))
    print("[OK] Model huruf berhasil dimuat.")
except FileNotFoundError as e:
    print(f"[WARN] {e}")
    model_huruf = None

try:
    model_angka = joblib.load(os.path.join(BASE_DIR, "model_angka.pkl"))
    print("[OK] Model angka berhasil dimuat.")
except FileNotFoundError as e:
    print(f"[WARN] {e}")
    model_angka = None

try:
    model_kata = IsyaratBiGRU(input_size=104, hidden_size=128, num_layers=2, num_classes=16)
    model_kata.load_state_dict(
        torch.load(os.path.join(BASE_DIR, "model_gru_kata.pth"), map_location=device)
    )
    model_kata.to(device)
    model_kata.eval()
    print("[OK] Model kata (BiGRU PyTorch) berhasil dimuat.")
except FileNotFoundError as e:
    print(f"[WARN] {e}")
    model_kata = None

LABEL_KATA = {
    0: "berasal", 1: "berpikir", 2: "maaf", 3: "makan",
    4: "mandi", 5: "nama", 6: "perkenalkan", 7: "saya",
    8: "tidur", 9: "tolong", 10: "tunggu", 11: "umur",
    12: "berteman", 13: "sementara", 14: "santai aja", 15: "bareng",
}

# ==============================================================
# MEDIAPIPE
# ==============================================================
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6
)

FACE_POINTS = [1, 33, 61, 199, 263]
POSE_POINTS = [11, 12, 13, 14, 15]

# ==============================================================
# STATE (server-side, single user/session)
# ==============================================================
pred_buffer  = deque(maxlen=8)
frame_buffer = deque(maxlen=30)   
history      = {"huruf": [], "angka": [], "kata": []}
last_saved   = {"huruf": "", "angka": "", "kata": ""}


# ==============================================================
# HELPERS
# ==============================================================
def get_confidence(model, data):
    if model is None: return None
    try:
        if hasattr(model, "predict_proba"):
            return float(np.max(model.predict_proba(data)[0]))
    except Exception: pass
    return None

def get_stable_prediction(pred):
    pred_buffer.append(pred)
    if len(pred_buffer) < 6: return None
    kandidat, jumlah = Counter(pred_buffer).most_common(1)[0]
    return kandidat if jumlah >= 5 else None

def simpan_history(mode, pred):
    if pred is None or pred == "-": return
    if pred != last_saved[mode]:
        history[mode].append(pred)
        last_saved[mode] = pred

def decode_frame(b64_data: str):
    if "," in b64_data:
        b64_data = b64_data.split(",")[1]
    img_bytes = base64.b64decode(b64_data)
    arr   = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame

def extract_one_hand_live(hand_landmarks):
    lm  = hand_landmarks.landmark
    pts = np.array([[p.x, p.y] for p in lm])
    base = pts[0]
    pts  = pts - base
    max_val = np.max(np.abs(pts))
    if max_val != 0: pts = pts / max_val
    data = pts.flatten().tolist()

    def dist_live(a, b):
        return ((lm[a].x - lm[b].x) ** 2 + (lm[a].y - lm[b].y) ** 2) ** 0.5
    data.extend([dist_live(4, 8), dist_live(8, 12), dist_live(12, 16), dist_live(16, 20)])
    return data

def extract_angka(hand_lm):
    lm_list = hand_lm.landmark
    bx, by  = lm_list[0].x, lm_list[0].y
    data    = []
    for lm in lm_list:
        data.extend([lm.x - bx, lm.y - by])
    data = np.array(data)
    data = data - np.min(data)
    if np.max(data) != 0: data = data / np.max(data)
    data = data.tolist()

    def d(a, b): return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5
    data.extend([d(lm_list[4], lm_list[8]), d(lm_list[8], lm_list[12]), d(lm_list[12], lm_list[16]), d(lm_list[16], lm_list[20])])
    return np.array(data).reshape(1, -1) if len(data) == 46 else None

def extract_kata_frame(result):
    data = []
    if result.left_hand_landmarks:
        for lm in result.left_hand_landmarks.landmark: data += [lm.x, lm.y]
    else: data += [0.0] * 42

    if result.right_hand_landmarks:
        for lm in result.right_hand_landmarks.landmark: data += [lm.x, lm.y]
    else: data += [0.0] * 42

    if result.face_landmarks:
        for i in FACE_POINTS:
            box_lm = result.face_landmarks.landmark[i]
            data += [box_lm.x, box_lm.y]
    else: data += [0.0] * 10

    if result.pose_landmarks:
        for i in POSE_POINTS:
            box_lm = result.pose_landmarks.landmark[i]
            data += [box_lm.x, box_lm.y]
    else: data += [0.0] * 10

    return data if len(data) == 104 else None


# ==============================================================
# ROUTES
# ==============================================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(force=True)
    mode = body.get("mode", "huruf")
    b64  = body.get("image", "")

    if not b64: return jsonify({"error": "No image"}), 400
    frame = decode_frame(b64)
    frame = cv2.flip(frame, 1)
    if frame is None: return jsonify({"error": "Decode failed"}), 400

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pred   = "-"
    conf   = None
    stable = None

    # ======================================================
    # MODE HURUF
    # ======================================================
    if mode == "huruf":
        result_hands = hands_detector.process(rgb)
        if result_hands.multi_hand_landmarks and model_huruf:
            empty_hand = [0.0] * 46
            left_data  = empty_hand.copy()
            right_data = empty_hand.copy()

            for hand_landmarks, handedness in zip(result_hands.multi_hand_landmarks, result_hands.multi_handedness):
                hand_label = handedness.classification[0].label
                if hand_label == "Left":  left_data = extract_one_hand_live(hand_landmarks)
                elif hand_label == "Right": right_data = extract_one_hand_live(hand_landmarks)

            data_huruf = left_data + right_data
            if len(data_huruf) == 92:
                data_huruf = np.array(data_huruf).reshape(1, -1)
                if isinstance(model_huruf, dict) and "model" in model_huruf:
                    pred_encoded = model_huruf["model"].predict(data_huruf)[0]
                    pred = str(model_huruf["label_encoder"].inverse_transform([pred_encoded])[0])
                    conf = get_confidence(model_huruf["model"], data_huruf)
                else:
                    pred = str(model_huruf.predict(data_huruf)[0])
                    conf = get_confidence(model_huruf, data_huruf)

    # ======================================================
    # MODE ANGKA
    # ======================================================
    elif mode == "angka":
        result = holistic.process(rgb)
        hand = result.right_hand_landmarks or result.left_hand_landmarks
        if hand and model_angka:
            feat = extract_angka(hand)
            if feat is not None:
                pred = str(model_angka.predict(feat)[0])
                conf = get_confidence(model_angka, feat)

    # ======================================================
    # MODE KATA (Sesuai Logika Rem Tangan Program Testing Baru)
    # ======================================================
    elif mode == "kata":
        result = holistic.process(rgb)
        
        # 1. CEK KEBERADAAN TANGAN (Kiri ATAU Kanan)
        tangan_terdeteksi = result.left_hand_landmarks is not None or result.right_hand_landmarks is not None

        if model_kata:
            feat_frame = extract_kata_frame(result)
            if feat_frame is not None:
                frame_buffer.append(feat_frame)

            # 2. LOGIKA PREDIKSI DENGAN REM TANGAN
            if len(frame_buffer) == 30 and tangan_terdeteksi:
                tensor_input = torch.tensor([list(frame_buffer)], dtype=torch.float32).to(device)
                
                with torch.no_grad():
                    outputs = model_kata(tensor_input)
                    probabilities = torch.softmax(outputs, dim=1)
                    conf_val, pred_idx_tensor = torch.max(probabilities, 1)
                    
                    pred_index = int(pred_idx_tensor.item())
                    conf       = float(conf_val.item())
                    pred       = LABEL_KATA.get(pred_index, f"?{pred_index}")
            else:
                # JIKA BUFFER BELUM 30 ATAU TANGAN TIDAK ADA: Langsung set ke "-"
                pred = "-"

    if pred != "-":
        stable = get_stable_prediction(pred)
        simpan_history(mode, stable)

    return jsonify({
        "raw":        pred,
        "stable":     stable or "-",
        "confidence": round(conf * 100, 1) if conf is not None else None,
        "mode":       mode,
        "history":    history[mode][-10:],
        "buffer_progress": len(frame_buffer) if mode == "kata" else None
    })


@app.route("/api/status", methods=["GET"])
def api_status():
    return jsonify({
        "server":      "online",
        "model_huruf": model_huruf is not None,
        "model_angka": model_angka is not None,
        "model_kata":  model_kata  is not None,
    })


@app.route("/api/history", methods=["GET"])
def api_history(): return jsonify(history)


@app.route("/api/history/clear", methods=["POST"])
def api_clear():
    for k in history:
        history[k].clear()
        last_saved[k] = ""
    pred_buffer.clear()
    frame_buffer.clear()
    return jsonify({"status": "ok"})


@app.route("/api/change_mode", methods=["POST"])
def api_change_mode():
    pred_buffer.clear()
    frame_buffer.clear()   
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)