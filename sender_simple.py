import cv2
import numpy as np
import time
import zstandard as zstd
import socket
import struct
try:
    from picamera2 import Picamera2
except Exception:
    Picamera2 = None

VIDEO_MODE = False
VIDEO_PATH = r"C:\Users\Matan\Documents\Matan\LoRa_video\videos\DJI_0008.MOV"
CAMERA_BACKEND = "OPENCV"
PICAM2_SIZE = (640, 480)
PICAM2_FORMAT = "RGB888"
USE_ZDICT = True
ZSTD_DICT_PATH = "contours_full_128k.zdict"

if USE_ZDICT:
    with open(ZSTD_DICT_PATH, "rb") as f:
        _DICT_BYTES = f.read()
    _ZDICT = zstd.ZstdCompressionDict(_DICT_BYTES)

def simplify_boundary(boundary, epsilon=3.0):
    boundary = boundary.astype(np.float32)
    simp = cv2.approxPolyDP(boundary, epsilon=epsilon, closed=True)
    return np.ascontiguousarray(simp.squeeze())

def auto_canny(img_u8: np.ndarray, sigma: float = 0.33,
               aperture_size: int = 3, l2: bool = True) -> np.ndarray:
    v = np.median(img_u8)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(img_u8, lower, upper, apertureSize=aperture_size, L2gradient=l2)

def encode_frame(frame, percentile_pin=50, scharr_percentile=92):
    times = {}
    start = time.time()
    frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_AREA)
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    times['preprocess'] = time.time() - start

    start = time.time()
    img_median = cv2.medianBlur(img_gray, 3)
    times['median_filter'] = time.time() - start

    start = time.time()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_median)
    times['clahe'] = time.time() - start

    start = time.time()
    lap = cv2.Laplacian(img_clahe, cv2.CV_64F)
    lap_abs = np.abs(lap)
    thresh = np.percentile(lap_abs, 95)
    _, lap_binary = cv2.threshold(lap_abs.astype(np.uint8), int(thresh), 255, cv2.THRESH_BINARY)
    edges_canny = auto_canny(cv2.GaussianBlur(img_clahe, (3, 3), 0))
    gx = cv2.Scharr(img_clahe, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(img_clahe, cv2.CV_32F, 0, 1)
    scharr_mag = cv2.magnitude(gx, gy)
    scharr_t = np.percentile(scharr_mag, float(scharr_percentile))
    scharr_bin = (scharr_mag >= scharr_t).astype(np.uint8) * 255
    img_binary = cv2.bitwise_or(cv2.bitwise_or(lap_binary, edges_canny), scharr_bin)
    times['edge_detection'] = time.time() - start

    start = time.time()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    times['morph_close'] = time.time() - start

    start = time.time()
    contours, _ = cv2.findContours(img_binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c.squeeze().astype(np.float32) for c in contours if len(c) > 2]
    times['find_contours'] = time.time() - start

    start = time.time()
    simplified_all = [simplify_boundary(b, epsilon=3.0) for b in valid_contours]
    simplified_all = [s for s in simplified_all if len(s.shape) == 2 and s.shape[0] >= 3]
    if not simplified_all:
        simplified = []
    else:
        def _geom_len(poly: np.ndarray) -> float:
            d = np.diff(poly.astype(np.float32), axis=0)
            g = float(np.sum(np.sqrt((d**2).sum(axis=1))))
            if poly.shape[0] >= 3:
                g += float(np.linalg.norm(poly[-1] - poly[0]))
            return g
        lengths = np.array([_geom_len(p) for p in simplified_all], dtype=np.float32)
        points  = np.array([max(1, p.shape[0]) for p in simplified_all], dtype=np.int32)
        effvals = lengths / points
        q20, q40, q60, q80 = np.quantile(effvals, [0.2, 0.4, 0.6, 0.8])
        band_idx = np.empty(len(effvals), dtype=np.int32)
        band_idx[effvals >= q80] = 0
        band_idx[(effvals < q80) & (effvals >= q60)] = 1
        band_idx[(effvals < q60) & (effvals >= q40)] = 2
        band_idx[(effvals < q40) & (effvals >= q20)] = 3
        band_idx[effvals < q20] = 4
        total_N  = len(simplified_all)
        target_N = max(1, int(round(total_N * (percentile_pin / 100.0))))
        base_q   = np.array([0.20, 0.20, 0.06, 0.03, 0.01], dtype=np.float32)
        quotas   = np.floor(base_q * total_N + 1e-6).astype(int)
        delta = target_N - int(quotas.sum())
        if delta != 0:
            order = [0,1,2,3,4] if delta > 0 else [4,3,2,1,0]
            i = 0
            while delta != 0:
                quotas[order[i]] += 1 if delta > 0 else -1
                delta += -1 if delta > 0 else 1
                i = (i + 1) % 5
        selected_idx = []
        for b in range(5):
            idxs = np.where(band_idx == b)[0]
            if idxs.size == 0: continue
            areas_s = np.array([cv2.contourArea(simplified_all[i]) for i in idxs], dtype=np.float32)
            order = idxs[np.argsort(-areas_s)]
            k = max(0, min(quotas[b], order.size))
            selected_idx.extend(order[:k].tolist())
        if len(selected_idx) < target_N:
            need = target_N - len(selected_idx)
            taken = set(selected_idx)
            for b in [0,1,2,3,4]:
                idxs = [i for i in np.where(band_idx == b)[0] if i not in taken]
                if not idxs: continue
                areas_s = np.array([cv2.contourArea(simplified_all[i]) for i in idxs], dtype=np.float32)
                order = np.array(idxs)[np.argsort(-areas_s)]
                grab = order[:need].tolist()
                selected_idx.extend(grab)
                need -= len(grab)
                if need <= 0: break
        simplified = [simplified_all[i] for i in selected_idx]
    times['simplification'] = time.time() - start

    total_t = sum(times.values())
    ordered = [(k, times[k]) for k in sorted(times.keys())]
    breakdown = " ".join([f"{k}={v*1000:.1f}ms" for k, v in ordered])
    print(f"[ENCODE breakdown] total={total_t*1000:.1f}ms {breakdown}")
    return simplified

def encode_contour_relative(pts_i16: np.ndarray):
    assert pts_i16.ndim == 2 and pts_i16.shape[1] == 2 and pts_i16.dtype == np.int16
    L = pts_i16.shape[0]
    if L < 1:
        return b"A", b""
    head = pts_i16[0:1].astype(np.int16)
    if L == 1:
        return b"A", head.tobytes()
    prev = pts_i16[:-1].astype(np.int32)
    curr = pts_i16[1:].astype(np.int32)
    deltas = (curr - prev)
    max_abs = int(np.max(np.abs(deltas))) if deltas.size else 0
    if max_abs <= 127:
        return b"8", head.tobytes() + deltas.astype(np.int8).tobytes()
    elif max_abs <= 32767:
        return b"6", head.tobytes() + deltas.astype(np.int16).tobytes()
    else:
        return b"A", pts_i16.tobytes()

def pack_full_relative(contours):
    buf = bytearray()
    cnt_A = cnt_8 = cnt_6 = 0
    for pts in contours:
        pts = pts.astype(np.int16)
        L = pts.shape[0]
        buf.extend(struct.pack(">H", L))
        mode, encoded = encode_contour_relative(pts)
        buf.extend(mode)
        buf.extend(encoded)
        if mode == b"A": cnt_A += 1
        elif mode == b"8": cnt_8 += 1
        elif mode == b"6": cnt_6 += 1
    if USE_ZDICT:
        comp = zstd.ZstdCompressor(level=22, dict_data=_ZDICT).compress(bytes(buf))
    else:
        comp = zstd.ZstdCompressor(level=22).compress(bytes(buf))
    return comp, (cnt_A, cnt_8, cnt_6)

if __name__ == "__main__":
    if VIDEO_MODE:
        cap = cv2.VideoCapture(VIDEO_PATH)
        input_desc = f"VIDEO file ({VIDEO_PATH})"
    else:
        if CAMERA_BACKEND.upper() == "PICAM2":
            if Picamera2 is None:
                raise RuntimeError("Picamera2 not available.")
            picam = Picamera2()
            picam.configure(picam.create_preview_configuration(main={"size": PICAM2_SIZE, "format": PICAM2_FORMAT}))
            picam.start()
            cap = None
            input_desc = f"PICAM2 {PICAM2_SIZE} {PICAM2_FORMAT}"
        else:
            cap = cv2.VideoCapture(0)
            input_desc = "LIVE camera (OpenCV)"

    HOST = "127.0.0.1"
    PORT = 5001
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"[Sender-UDP] Target {HOST}:{PORT}")
    print(f"[Sender] Connected to receiver on {HOST}:{PORT}")
    print(f"[Sender] Input source: {input_desc}")

    if (VIDEO_MODE and not cap.isOpened()) or (not VIDEO_MODE and CAMERA_BACKEND.upper() != "PICAM2" and not cap.isOpened()):
        raise RuntimeError("Failed to open input source")

    H, W = 512, 512
    RAW_BASELINE_BGR = H * W * 3
    RAW_BASELINE_KB = RAW_BASELINE_BGR / 1024.0

    frame_id = 0
    while True:
        t_cycle = time.time()
        t0 = time.time()
        if VIDEO_MODE or CAMERA_BACKEND.upper() != "PICAM2":
            ret, frame = cap.read()
            t_capture = time.time() - t0
            if not ret:
                print("[Sender] End of stream; stopping.")
                break
        else:
            frame = picam.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            t_capture = time.time() - t0

        t1 = time.time()
        contours = encode_frame(frame, percentile_pin=50, scharr_percentile=92)
        t_encode = time.time() - t1

        t2 = time.time()
        full_rel_comp, (full_A, full_8, full_6) = pack_full_relative(contours)
        full_bytes = len(full_rel_comp)
        t_pack = time.time() - t2

        t3 = time.time()
        udp_header = struct.pack(">II", frame_id, full_bytes)  # frame_id, length
        packet = udp_header + full_rel_comp
        sock.sendto(packet, (HOST, PORT))
        t_send = time.time() - t3

        num_contours = len(contours)
        print(f"[FULL] {full_bytes/1024.0:.2f} KB ({(full_bytes/RAW_BASELINE_BGR)*100.0:.2f}% of raw) "
              f"contours={num_contours} raw={RAW_BASELINE_KB:.2f} KB")
        print(f"[FULL modes] A={full_A} 8={full_8} 6={full_6}")

        t_total = time.time() - t_cycle
        t_other = max(0.0, t_total - (t_capture + t_encode + t_pack + t_send))
        print(
            "[SENDER timing] "
            f"frame={frame_id} "
            f"capture={t_capture*1000:.1f}ms "
            f"encode={t_encode*1000:.1f}ms "
            f"pack={t_pack*1000:.1f}ms "
            f"send={t_send*1000:.1f}ms "
            f"other={t_other*1000:.1f}ms "
            f"total={t_total*1000:.1f}ms"
        )
        frame_id += 1

    sock.close()
    if VIDEO_MODE or CAMERA_BACKEND.upper() != "PICAM2":
        cap.release()
    else:
        try:
            picam.stop()
        except Exception:
            pass
    cv2.destroyAllWindows()