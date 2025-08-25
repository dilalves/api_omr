import os, base64, cv2, numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/omr": {"origins": "*"}, r"/health": {"origins": "*"}})

def decode_base64_to_gray(b64: str) -> np.ndarray:
    blob = base64.b64decode(b64)
    arr = np.frombuffer(blob, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Imagem inválida.")
    return img

def binarize(gray: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(gray, (5,5), 0)
    thr = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 41, 10)
    return thr

@app.get("/health")
def health():
    return jsonify({"ok": True})

@app.post("/omr")
def omr():
    data = request.get_json(force=True) or {}
    img_b64 = data.get("image_base64")
    if not img_b64:
        return jsonify({"error": "image_base64 ausente"}), 400

    gray = decode_base64_to_gray(img_b64)
    bin_img = binarize(gray)

    # TODO: Implementar ROIs para detectar inscrição
    inscricao = ""
    confidence = 0.0

    return jsonify({
        "inscricao": inscricao,
        "confidence": confidence,
        "w": int(bin_img.shape[1]),
        "h": int(bin_img.shape[0])
    })
