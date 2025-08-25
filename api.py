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

# 🚨 Aqui você mantém apenas essa versão, não a antiga
@app.post("/omr")
def omr():
    data = request.get_json(force=True) or {}
    img_b64 = data.get("image_base64")
    if not img_b64:
        return jsonify({"error": "image_base64 ausente"}), 400

    # 1) decodifica e binariza
    gray = decode_base64_to_gray(img_b64)
    H, W = gray.shape[:2]
    bin_img = binarize(gray)

    # 2) Configuração do grid (ajuste com sua folha)
    CFG = {
        "Y0": 0.78, "ALT": 0.18,
        "X0": 0.10, "LARG": 0.80,
        "ROWS": 7, "COLS": 10,
        "BUBBLE_H": 0.70, "BUBBLE_W": 0.70,
        "MARGIN_DELTA": 0.05
    }

    # 3) Dimensões do grid
    grid_y = int(CFG["Y0"] * H)
    grid_h = int(CFG["ALT"] * H)
    grid_x = int(CFG["X0"] * W)
    grid_w = int(CFG["LARG"] * W)
    cell_h = grid_h / CFG["ROWS"]
    cell_w = grid_w / CFG["COLS"]

    def bubble_roi(r, c):
        cy0 = int(grid_y + r * cell_h)
        cx0 = int(grid_x + c * cell_w)
        bh = int(cell_h * CFG["BUBBLE_H"])
        bw = int(cell_w * CFG["BUBBLE_W"])
        y = int(cy0 + (cell_h - bh) / 2)
        x = int(cx0 + (cell_w - bw) / 2)
        return y, x, bh, bw

    digits, confs = [], []
    debug = gray.copy()

    for r in range(CFG["ROWS"]):
        scores = []
        for c in range(CFG["COLS"]):
            y, x, h, w = bubble_roi(r, c)
            roi = bin_img[y:y+h, x:x+w]
            if roi.size == 0:
                scores.append(0.0)
                continue
            fill = float((roi > 0).mean())
            scores.append(fill)
            cv2.rectangle(debug, (x,y), (x+w,y+h), (0,0,0), 1)

        idx = int(np.argmax(scores))
        top1, top2 = float(scores[idx]), float(sorted(scores, reverse=True)[1])
        line_conf = max(0.0, top1 - top2)
        confs.append(line_conf)
        digits.append(str(idx))
        y,x,h,w = bubble_roi(r, idx)
        cv2.rectangle(debug, (x,y), (x+w,y+h), (0,0,0), 2)

    confidence = float(np.mean(confs)) if confs else 0.0
    inscricao = "".join(digits)

    _, dbg_png = cv2.imencode(".png", debug)
    debug_b64 = base64.b64encode(dbg_png).decode("ascii")
    debug_url = f"data:image/png;base64,{debug_b64}"

    return jsonify({
        "inscricao": inscricao,
        "confidence": round(confidence, 3),
        "debug_url": debug_url
    })
