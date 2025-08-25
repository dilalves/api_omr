import os, base64
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/omr": {"origins": "*"}, r"/health": {"origins": "*"}})

# ---------- utils básicos ----------
def decode_base64_to_gray(b64: str) -> np.ndarray:
    blob = base64.b64decode(b64)
    arr  = np.frombuffer(blob, dtype=np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Imagem inválida.")
    return img

def binarize(gray: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    thr = cv2.adaptiveThreshold(
        g, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
        41, 10
    )
    # fecha pequenos buracos em marcações fracas
    kernel = np.ones((3, 3), np.uint8)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)
    return thr

# ---------- detecção dos 4 marcadores + warp ----------
def find_corners(gray):
    thr = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
        41, 10
    )
    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = gray.shape[:2]

    boxes = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < (H * W) * 0.00015 or area > (H * W) * 0.02:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) != 4:
            continue
        x, y, w, h = cv2.boundingRect(approx)
        ar = w / float(h)
        if 0.75 <= ar <= 1.25:  # quase quadrado
            cx, cy = x + w / 2, y + h / 2
            boxes.append(((cx, cy), approx.reshape(-1, 2).astype(np.float32)))

    if len(boxes) < 4:
        return None

    # pega um candidato para cada canto (mais próximo de TL, TR, BR, BL)
    corners_img = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=np.float32)
    chosen, used = [], set()
    for corner in corners_img:
        best_i, best_d = -1, 1e18
        for i, (cent, pts) in enumerate(boxes):
            if i in used:
                continue
            d = (cent[0] - corner[0]) ** 2 + (cent[1] - corner[1]) ** 2
            if d < best_d:
                best_d, best_i = d, i
        if best_i >= 0:
            used.add(best_i)
            chosen.append(boxes[best_i][1])

    centers = np.array([pts.mean(axis=0) for pts in chosen], dtype=np.float32)
    s = centers.sum(axis=1)              # TL menor soma, BR maior
    d = np.diff(centers, axis=1).ravel() # TR menor (x-y), BL maior
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = centers[np.argmin(s)]   # TL
    ordered[2] = centers[np.argmax(s)]   # BR
    ordered[1] = centers[np.argmin(d)]   # TR
    ordered[3] = centers[np.argmax(d)]   # BL
    return ordered

def warp_to_template(gray, target_size=(2480, 3508)):  # (W,H) A4 ~300dpi
    corners = find_corners(gray)
    if corners is None:
        return gray, False
    Wt, Ht = target_size
    dst = np.array([[0, 0], [Wt - 1, 0], [Wt - 1, Ht - 1], [0, Ht - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners, dst)
    aligned = cv2.warpPerspective(gray, M, (Wt, Ht))
    return aligned, True

# ---------- endpoints ----------
@app.get("/health")
def health():
    return jsonify({"ok": True})

@app.post("/omr")
def omr():
    data = request.get_json(force=True) or {}
    img_b64 = data.get("image_base64")
    if not img_b64:
        return jsonify({"error": "image_base64 ausente"}), 400

    # 1) decodifica -> alinha (warp) -> binariza
    gray = decode_base64_to_gray(img_b64)
    gray, warped = warp_to_template(gray)      # se não achar marcadores, segue sem warp
    H, W = gray.shape[:2]
    bin_img = binarize(gray)

    # 2) configuração do grid (7 linhas x 10 colunas)
    CFG = {
      "Y0": 0.8347,
      "ALT": 0.1043,
      "X0": 0.6989,
      "LARG": 0.2068,
      "ROWS": 7, "COLS": 10,
      "BUBBLE_H": 0.75,
      "BUBBLE_W": 0.82,
      "MARGIN_DELTA": 0.06
    }

    # 3) dimensões reais do grid
    grid_y = int(CFG["Y0"] * H)
    grid_h = int(CFG["ALT"] * H)
    grid_x = int(CFG["X0"] * W)
    grid_w = int(CFG["LARG"] * W)
    cell_h = grid_h / CFG["ROWS"]
    cell_w = grid_w / CFG["COLS"]

    def bubble_roi(r, c):
        cy0 = int(grid_y + r * cell_h)
        cx0 = int(grid_x + c * cell_w)
        bh  = int(cell_h * CFG["BUBBLE_H"])
        bw  = int(cell_w * CFG["BUBBLE_W"])
        y   = int(cy0 + (cell_h - bh) / 2)
        x   = int(cx0 + (cell_w - bw) / 2)
        return y, x, bh, bw

    # 4) varre as 7 linhas; cada coluna 0..9 vira score (proporção de preto)
    digits, confs = [], []
    debug = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # para desenhar em preto

    for r in range(CFG["ROWS"]):
        scores = []
        for c in range(CFG["COLS"]):
            y, x, h, w = bubble_roi(r, c)
            roi = bin_img[y:y + h, x:x + w]
            if roi.size == 0:
                scores.append(0.0)
                continue
            fill = float((roi > 0).mean())
            scores.append(fill)
            cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 0, 0), 1)

        idx  = int(np.argmax(scores))  # coluna mais “preta”
        top1 = float(scores[idx])
        top2 = float(sorted(scores, reverse=True)[1]) if len(scores) > 1 else 0.0
        line_conf = max(0.0, top1 - top2)
        confs.append(line_conf)
        digits.append(str(idx))

        y, x, h, w = bubble_roi(r, idx)
        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 0, 0), 2)

    confidence = float(np.mean(confs)) if confs else 0.0
    inscricao  = "".join(digits)

    # 5) imagem de debug (útil para calibrar CFG)
    _, dbg_png = cv2.imencode(".png", debug)
    debug_b64  = base64.b64encode(dbg_png).decode("ascii")
    debug_url  = f"data:image/png;base64,{debug_b64}"

    return jsonify({
        "inscricao": inscricao,
        "confidence": round(confidence, 3),
        "warped": bool(warped),
        "debug_url": debug_url
    })

