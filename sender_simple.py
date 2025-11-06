import cv2
import numpy as np
import time
import socket
import struct
from bitstring import BitStream
try:
    from picamera2 import Picamera2
except Exception:
    Picamera2 = None

VIDEO_MODE = True
VIDEO_PATH = r"C:\Users\Matan\Documents\Matan\LoRa_video\videos\DJI_0008.MOV"
CAMERA_BACKEND = "OPENCV"
PICAM2_SIZE = (640, 480)
PICAM2_FORMAT = "RGB888"

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
    simplified_initial = [simplify_boundary(b, epsilon=3.0) for b in valid_contours]

    simplified_all = []
    valid_original_indices = []

    for idx, s in enumerate(simplified_initial):
        if len(s.shape) == 2 and s.shape[0] >= 3:
            simplified_all.append(s)
            valid_original_indices.append(idx)

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

        TARGET_POINTS = 20 # Max points per contour
        MAX_EPSILON = 6  # Safety break to prevent infinite loops
        STEP_EPSILON = 0.5   # How much to increase epsilon each time

        simplified = []
        for i in selected_idx:
            # Get the contour that passed the filter (simplified with epsilon=3.0)
            pts = simplified_all[i]
            
            # If it's already "cheap" (<= 20 points), just use it.
            if pts.shape[0] <= TARGET_POINTS:
                simplified.append(pts)
            else:
                correct_original_index = valid_original_indices[i]
                original_contour = valid_contours[correct_original_index]
                current_epsilon = 3.0 + STEP_EPSILON

                last_valid_pts = pts 

                while current_epsilon < MAX_EPSILON:
                    new_pts = simplify_boundary(original_contour, epsilon=current_epsilon)
                    
                    if new_pts.shape[0] < 3:
                        pts = last_valid_pts 
                        break

                    last_valid_pts = new_pts 
                    pts = new_pts            

                    if new_pts.shape[0] <= TARGET_POINTS:
                        break

                    current_epsilon += STEP_EPSILON

                if pts.shape[0] >= 3:
                    simplified.append(pts)

    times['simplification'] = time.time() - start

    total_t = sum(times.values())
    ordered = [(k, times[k]) for k in sorted(times.keys())]
    breakdown = " ".join([f"{k}={v*1000:.1f}ms" for k, v in ordered])
    print(f"[ENCODE breakdown] total={total_t*1000:.1f}ms {breakdown}")
    return simplified

def encode_contour_relative(pts_i16: np.ndarray):
    # This function returns the mode (int 6, 8, 10, or 16) and the raw deltas array
    assert pts_i16.ndim == 2 and pts_i16.shape[1] == 2 and pts_i16.dtype == np.int16
    L = pts_i16.shape[0]
    if L < 2:
        return None, None, None # Mode, Head, Deltas
    head = pts_i16[0:1].astype(np.int16)
    prev = pts_i16[:-1].astype(np.int32)
    curr = pts_i16[1:].astype(np.int32)
    deltas = (curr - prev) # Keep deltas as int32 numpy array
    max_abs = int(np.max(np.abs(deltas))) if deltas.size else 0

    if max_abs <= 31:      # Range for sint:6 is [-32, 31]
        mode = 6
    elif max_abs <= 127:   # Range for sint:8 is [-128, 127]
        mode = 8
    elif max_abs <= 511:   # Range for sint:10 is [-512, 511]
        mode = 10
    elif max_abs <= 32767: # Range for sint:16 is [-32768, 32767]
        mode = 16
    else:
        # Deltas too large, cannot encode
        return None, None, None
        
    return mode, head, deltas


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
        contours_final = encode_frame(frame, percentile_pin=50, scharr_percentile=92)
        t_encode = time.time() - t1

        full_A = full_8 = full_6 = 0
        total_bytes_sent = 0
        start_pack_and_send_all = time.time()
        
        # Use 6 bits for frame_id, wrapping around at 64
        frame_id_6bit = frame_id % 64

        # Define mode mapping to 2-bit flags
        # 00: 6-bit, 01: 8-bit, 10: 10-bit, 11: 16-bit
        mode_map = {6: 0, 8: 1, 10: 2, 16: 3}
        mode_counts = {6:0, 8:0, 10:0, 16:0} # For stats
        point_counts = {6:0, 8:0, 10:0, 16:0}

        for pts in contours_final:
            pts = pts.astype(np.int16)
            mode, head, deltas = encode_contour_relative(pts)
            
            # Skip contours that couldn't be encoded
            if mode is None:
                full_A += 1 # Count as skipped
                continue

            mode_counts[mode] += 1
            point_counts[mode] += pts.shape[0]
            mode_bits = mode_map[mode]
            
            # Pack 6-bit ID and 2-bit mode into a single byte
            header_byte = (frame_id_6bit << 2) | mode_bits
            
            # Create a BitStream for packing
            s = BitStream()
            # Add the 1-byte header
            s.append(struct.pack(">B", header_byte))
            # Add the head point (absolute, 16-bit signed x and y)
            s.append(f'int:16={head[0, 0]}, int:16={head[0, 1]}')
            # Add all deltas with the determined bit length (signed)
            format_string = f'int:{mode}'
            for dx, dy in deltas:
                s.append(f'{format_string}={dx}, {format_string}={dy}')

            # Get the packed bytes
            packet_payload = s.tobytes()
            total_bytes_sent += len(packet_payload)
            sock.sendto(packet_payload, (HOST, PORT))
        end_pack_and_send_all = time.time() - start_pack_and_send_all

        num_contours = len(contours_final)
        sent_count = sum(mode_counts.values())
        total_points_overall = sum(point_counts.values())
        print(
            f"[SENDER-STATS] "
            f"frame={frame_id} (ID_6bit={frame_id_6bit}) "
            f"contours={num_contours} (sent={sent_count}, skipped={full_A}) "
            f"modes (6/8/10/16)={mode_counts[6]}/{mode_counts[8]}/{mode_counts[10]}/{mode_counts[16]} "
            f"total_points={total_points_overall} "
            f"points (6/8/10/16)={point_counts[6]}/{point_counts[8]}/{point_counts[10]}/{point_counts[16]} "
            f"total_KB={(total_bytes_sent / 1024.0):.1f}"
            )

        t_total = time.time() - t_cycle
        t_known = t_capture + t_encode + end_pack_and_send_all
        t_other = max(0.0, t_total - t_known)
        print(
            "[SENDER timing] "
            f"frame={frame_id} "
            f"capture={t_capture*1000:.1f}ms "
            f"encode={t_encode*1000:.1f}ms "
            f"pack_and_send_all={end_pack_and_send_all*1000:.1f}ms "
            f"other={t_other*1000:.1f}ms "
            f"total={t_total*1000:.1f}ms"
        )
        # Calculate and print Sender FPS
        if t_total > 0:
            print(f"[SENDER-FPS] {1.0 / t_total:.1f} FPS")

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