import os, base64
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={
    r"/omr": {"origins": "*"},
    r"/health": {"origins": "*"},
    r"/warp_image": {"origins": "*"},   # <- adicione isto
})

# ---------- utils básicos ----------
def decode_base64_to_gray(b64: str) -> np.ndarray:
    blob = base64.b64decode(b64)
    arr  = np.frombuffer(blob, dtype=np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Imagem inválida.")
    return img

# 🚀 Normaliza a largura da imagem para padronizar
def normalize(img, width=1000):
    H, W = img.shape[:2]
    scale = width / W
    new_h = int(H * scale)
    return cv2.resize(img, (width, new_h))

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
            d = (cent[0] - corner[0])**2 + (cent[1] - corner[1])**2
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
    try:
        data = request.get_json(force=True) or {}
        img_b64 = data.get("image_base64")
        if not img_b64:
            return jsonify({"error": "image_base64 ausente"}), 400

        # opcional: forçar sem warp (igual calibrador)
        no_warp = bool(data.get("no_warp", False))

        # 1) decodifica -> decide base (warp ou normalize) -> binariza
        gray = decode_base64_to_gray(img_b64)

        if no_warp:
            warped = False
            gray = normalize(gray, width=1000)
        else:
            gray, warped = warp_to_template(gray)   # tenta warp pelos 4 quadrados
            if not warped:
                gray = normalize(gray, width=1000)  # fallback

        H, W = gray.shape[:2]
        bin_img = binarize(gray)

        # 2) configuração do grid (padrão + override recebido)
        DEFAULT_CFG = {
            "Y0": 0.8744,
            "ALT": 0.1173,
            "X0": 0.7359,
            "LARG": 0.2305,
            "ROWS": 7, "COLS": 10,
            "BUBBLE_H": 0.75,
            "BUBBLE_W": 0.80,
            "MARGIN_DELTA": 0.06,  # diferença mínima entre top1 e top2
            "MIN_FILL": 0.12       # preenchimento mínimo pra considerar "marcado"
        }
        user_cfg = data.get("cfg") or {}
        CFG = {**DEFAULT_CFG, **user_cfg}

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

        # 4) varre linhas/colunas e escolhe a mais “preta”
        digits, confs = [], []
        debug = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

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
                cv2.rectangle(debug, (x, y), (x+w, y+h), (0, 0, 0), 1)
            
            idx  = int(np.argmax(scores))
            top1 = float(scores[idx])
            top2 = float(sorted(scores, reverse=True)[1]) if len(scores) > 1 else 0.0
            
            if top1 < CFG["MIN_FILL"]:
                digit = "#"   # nenhuma bolha preenchida
            elif (top1 - top2) < CFG["MARGIN_DELTA"]:
                digit = "?"   # mais de uma bolha (ou ambíguo)
            else:
                digit = str(idx)  # escolha normal
            
            digits.append(digit)
            confs.append(max(0.0, top1 - top2))

            y, x, h, w = bubble_roi(r, idx)
            cv2.rectangle(debug, (x, y), (x+w, y+h), (0, 0, 0), 2)

        confidence = float(np.mean(confs)) if confs else 0.0
        inscricao  = "".join(digits)

        # 5) imagem de debug
        ok, dbg_png = cv2.imencode(".png", debug)
        debug_b64  = base64.b64encode(dbg_png).decode("ascii") if ok else None
        debug_url  = f"data:image/png;base64,{debug_b64}" if debug_b64 else None

        return jsonify({
            "inscricao": inscricao,
            "confidence": round(confidence, 3),
            "warped": bool(warped),
            "H": H, "W": W,
            "cfg_used": CFG,
            "debug_url": debug_url
        })

    except Exception as e:
        # log simples no stderr (aparece nos logs do Render)
        import traceback, sys
        traceback.print_exc(file=sys.stderr)
        return jsonify({"error": str(e)}), 500


@app.post("/warp_image")
def warp_image():
    """Recebe image_base64 e (opcional) no_warp.
       Retorna a imagem usada para OMR: warpeada (quando possível) ou normalizada."""
    import base64, numpy as np, cv2  # já estão importados no topo; repita se quiser

    data = request.get_json(force=True) or {}
    img_b64 = data.get("image_base64")
    no_warp = bool(data.get("no_warp", False))
    if not img_b64:
        return jsonify({"error": "image_base64 ausente"}), 400

    # 1) decodifica
    gray = decode_base64_to_gray(img_b64)

    # 2) decide base
    if no_warp:
        warped = False
        base = normalize(gray, width=1000)
    else:
        base, warped = warp_to_template(gray)
        if not warped:
            base = normalize(gray, width=1000)

    H, W = base.shape[:2]

    # 3) retorna JPEG base64 sem desenhos
    ok, buf = cv2.imencode(".jpg", base)
    b64 = base64.b64encode(buf).decode("ascii") if ok else None

    return jsonify({
        "warped": bool(warped),
        "H": H, "W": W,
        "image_base64": f"data:image/jpeg;base64,{b64}" if b64 else None
    })




