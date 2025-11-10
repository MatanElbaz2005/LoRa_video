import cv2
import numpy as np
import time
import struct
from bitstring import BitStream
try:
    import cv2.ximgproc
except ImportError:
    print("Warning: cv2.ximgproc not found. Thinning will be skipped.")
    print("Please run: pip install opencv-contrib-python")
try:
    from picamera2 import Picamera2
except Exception:
    Picamera2 = None

VIDEO_MODE = True
VIDEO_PATH = r"C:\Users\Matan\Documents\Matan\LoRa_video\videos\DJI_0008.MOV"
CAMERA_BACKEND = "OPENCV"
PICAM2_SIZE = (640, 480)
PICAM2_FORMAT = "RGB888"

USE_CANNY = True
USE_LAP = True
USE_SCHARR = True

# --- Helper Functions (Copied from sender_simple.py) ---

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

def encode_contour_relative(pts_i16: np.ndarray):
    assert pts_i16.ndim == 2 and pts_i16.shape[1] == 2 and pts_i16.dtype == np.int16
    L = pts_i16.shape[0]
    if L < 2:
        return None, None, None
    head = pts_i16[0:1].astype(np.int16)
    prev = pts_i16[:-1].astype(np.int32)
    curr = pts_i16[1:].astype(np.int32)
    deltas = (curr - prev)
    max_abs = int(np.max(np.abs(deltas))) if deltas.size else 0

    if max_abs <= 7:
        mode = 4
    elif max_abs <= 31:
        mode = 6
    elif max_abs <= 127:
        mode = 8
    elif max_abs <= 511:
        mode = 10
    else:
        return None, None, None
        
    return mode, head, deltas

def _geom_len(poly: np.ndarray) -> float:
    d = np.diff(poly.astype(np.float32), axis=0)
    g = float(np.sum(np.sqrt((d**2).sum(axis=1))))
    if poly.shape[0] >= 3:
        g += float(np.linalg.norm(poly[-1] - poly[0]))
    return g

def calculate_final_stats(contours_final):
    total_bytes_sent = 0
    total_points_overall = 0
    mode_map = {4: 0, 6: 1, 8: 2, 10: 3}

    for pts in contours_final:
        pts = pts.astype(np.int16)
        mode, head, deltas = encode_contour_relative(pts)
        if mode is None: continue
        
        total_points_overall += pts.shape[0]

        s = BitStream()
        s.append(struct.pack(">B", 0))
        s.append(f'int:16={head[0, 0]}, int:16={head[0, 1]}')
        format_string = f'int:{mode}'
        for dx, dy in deltas:
            s.append(f'{format_string}={dx}, {format_string}={dy}')
        
        packet_payload = s.tobytes()
        total_bytes_sent += len(packet_payload)
        
    return total_points_overall, (total_bytes_sent / 1024.0)

def add_stats_to_canvas(canvas, stats):
    before, after, pixels, kb, converted = stats
    
    cv2.putText(canvas, f"Contours (Pre-Echo): {before}", (10, 512 - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(canvas, f"Contours (Final): {after}", (10, 512 - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(canvas, f"Converted to Lines: {converted}", (10, 512 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
    cv2.putText(canvas, f"Total Points: {pixels}", (10, 512 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(canvas, f"Total KB: {kb:.2f}", (10, 512 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return canvas

# --- Pipeline Functions ---

def preprocess_frame(frame, scharr_percentile=92):
    frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_AREA)
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    img_median = cv2.medianBlur(img_gray, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_median)

    is_default_case = not (USE_CANNY or USE_LAP or USE_SCHARR)
    active_edges = []

    if USE_LAP and not is_default_case:
        lap = cv2.Laplacian(img_clahe, cv2.CV_64F)
        lap_abs = np.abs(lap)
        thresh = np.percentile(lap_abs, 95)
        _, lap_binary = cv2.threshold(lap_abs.astype(np.uint8), int(thresh), 255, cv2.THRESH_BINARY)
        active_edges.append(lap_binary)

    if USE_CANNY or is_default_case:
        edges_canny = auto_canny(cv2.GaussianBlur(img_clahe, (3, 3), 0))
        active_edges.append(edges_canny)

    if USE_SCHARR and not is_default_case:
        gx = cv2.Scharr(img_clahe, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(img_clahe, cv2.CV_32F, 0, 1)
        scharr_mag = cv2.magnitude(gx, gy)
        scharr_t = np.percentile(scharr_mag, float(scharr_percentile))
        scharr_bin = (scharr_mag >= scharr_t).astype(np.uint8) * 255
        active_edges.append(scharr_bin)

    if not active_edges:
        img_binary = np.zeros_like(img_clahe, dtype=np.uint8)
    else:
        img_binary = active_edges[0]
        for i in range(1, len(active_edges)):
            img_binary = cv2.bitwise_or(img_binary, active_edges[i])
            
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img_binary_closed = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    try:
        img_binary_thinned = cv2.ximgproc.thinning(img_binary_closed)
    except Exception:
        img_binary_thinned = img_binary_closed

    return img_binary_closed, img_binary_thinned

def process_contours_pipeline(binary_input, percentile_pin=50, use_smart_cut=False):
    
    debug_canvas = np.zeros((512, 512, 3), dtype=np.uint8)
    
    contours, hierarchy = cv2.findContours(binary_input, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    dropped_contours_count = 0
    
    if contours is None or hierarchy is None:
        stats = (0, 0, 0, 0.0, 0)
        return debug_canvas, stats

    hierarchy = hierarchy[0] 
    valid_contours = []
    original_indices_map_vc = []
    
    for i, contour in enumerate(contours):
        if len(contour) > 2:
            valid_contours.append(contour.squeeze().astype(np.float32))
            original_indices_map_vc.append(i)

    simplified_initial = [simplify_boundary(b, epsilon=3.0) for b in valid_contours]

    simplified_all = []
    valid_original_indices = []
    original_indices_map_sa = []

    for idx, s in enumerate(simplified_initial):
        if len(s.shape) == 2 and s.shape[0] >= 3:
            simplified_all.append(s)
            valid_original_indices.append(idx)
            original_indices_map_sa.append(original_indices_map_vc[idx])

    if not simplified_all:
        stats = (0, 0, 0, 0.0, 0)
        return debug_canvas, stats
    
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
    
    stats_before_drop = len(selected_idx)
    
    final_selected_indices_in_sa = []
    dropped_contours_simplified = []
    DISTANCE_THRESHOLD = 10.0
    set_of_selected_original_indices = {original_indices_map_sa[i] for i in selected_idx}

    if hierarchy is not None:
        for sa_index in selected_idx:
            original_idx = original_indices_map_sa[sa_index]
            parent_index = hierarchy[original_idx][3]
            is_inner_hole = (parent_index != -1)

            if not is_inner_hole:
                final_selected_indices_in_sa.append(sa_index)
                continue

            if parent_index not in set_of_selected_original_indices:
                final_selected_indices_in_sa.append(sa_index)
                continue
            
            outer_contour = contours[parent_index]
            inner_contour = contours[original_idx]
            is_redundant = True
            
            inner_points = inner_contour.squeeze()
            if inner_points.ndim == 1:
                inner_points = np.array([inner_points])
            
            if inner_points.size == 0:
                continue
            
            for pt in inner_points:
                dist = cv2.pointPolygonTest(outer_contour, tuple(pt.astype(float)), measureDist=True)
                if abs(dist) > DISTANCE_THRESHOLD:
                    is_redundant = False
                    break
            
            if is_redundant:
                dropped_contours_count += 1
                dropped_contours_simplified.append(simplified_all[sa_index])
            else:
                final_selected_indices_in_sa.append(sa_index)
    else:
        final_selected_indices_in_sa = selected_idx

    stats_after_drop = len(final_selected_indices_in_sa)

    TARGET_POINTS = 20
    MAX_EPSILON = 10
    STEP_EPSILON = 0.5

    simplified = []
    for i in final_selected_indices_in_sa:
        pts = simplified_all[i]
        
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

    final_contours_for_stats = []
    final_contours_for_drawing = []
    converted_to_lines = 0

    if use_smart_cut:
        THINNESS_THRESHOLD = 0.5
        for pts in simplified:
            area = cv2.contourArea(pts)
            perimeter = cv2.arcLength(pts, True)
            ratio = area / (perimeter + 1e-6)

            if ratio < THINNESS_THRESHOLD and len(pts) > 3:
                # ... (כל הלוגיקה של החיתוך החכם נשארת כאן) ...
                rect = cv2.minAreaRect(pts.astype(np.float32))
                box = cv2.boxPoints(rect)
                dist1 = np.linalg.norm(box[0] - box[1])
                dist2 = np.linalg.norm(box[1] - box[2])
                if dist1 > dist2:
                    end1 = (box[0] + box[3]) / 2.0
                    end2 = (box[1] + box[2]) / 2.0
                else:
                    end1 = (box[0] + box[1]) / 2.0
                    end2 = (box[2] + box[3]) / 2.0
                start_idx = np.argmin(np.sum((pts - end1)**2, axis=1))
                end_idx = np.argmin(np.sum((pts - end2)**2, axis=1))
                i1 = min(start_idx, end_idx)
                i2 = max(start_idx, end_idx)
                path1_len = i2 - i1
                path2_len = (len(pts) - i2) + i1
                if path1_len > path2_len:
                    pts_open = pts[i1 : i2 + 1]
                else:
                    pts_open = np.concatenate((pts[i2:], pts[:i1 + 1]))

                if len(pts_open) < 2:
                    final_contours_for_stats.append(pts)
                    final_contours_for_drawing.append((pts, True))
                else:
                    final_contours_for_stats.append(pts_open)
                    final_contours_for_drawing.append((pts_open, False))
                    converted_to_lines += 1
            else:
                final_contours_for_stats.append(pts)
                final_contours_for_drawing.append((pts, True))
    else:
        # This is the "Original" pipeline logic (no cutting)
        final_contours_for_stats = simplified
        for pts in simplified:
            final_contours_for_drawing.append((pts, True))

    stats_total_pixels, stats_total_kb = calculate_final_stats(final_contours_for_stats)
    stats = (stats_before_drop, stats_after_drop, stats_total_pixels, stats_total_kb, converted_to_lines)

    try:
        for pts, is_closed in final_contours_for_drawing:
            cv2.polylines(debug_canvas, [pts.astype(np.int32)], isClosed=is_closed, color=(255, 255, 255), thickness=1)
            
        if dropped_contours_simplified:
            cv2.polylines(debug_canvas, [pts.astype(np.int32) for pts in dropped_contours_simplified], isClosed=True, color=(0, 0, 255), thickness=1)
    except Exception as e:
        pass 

    return debug_canvas, stats

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

    print(f"[Debug-Pipeline-Compare] Input source: {input_desc}")
    print("[Debug-Pipeline-Compare] Press 'q' or ESC to quit.")
    print("[Debug-Pipeline-Compare] Press any other key to advance to the next frame.")

    if (VIDEO_MODE and not cap.isOpened()) or (not VIDEO_MODE and CAMERA_BACKEND.upper() != "PICAM2" and not cap.isOpened()):
        raise RuntimeError("Failed to open input source")

    frame_id = 0
    while True:
        t_cycle = time.time()
        t0 = time.time()
        if VIDEO_MODE or CAMERA_BACKEND.upper() != "PICAM2":
            ret, frame = cap.read()
            t_capture = time.time() - t0
            if not ret:
                print("[Debug-Pipeline-Compare] End of stream; stopping.")
                break
        else:
            frame = picam.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            t_capture = time.time() - t0

        t1 = time.time()
        
        # --- 1. Preprocessing (Done once) ---
        img_binary_closed, img_binary_thinned = preprocess_frame(frame)
        
        # --- 2. Run Original Pipeline (Thick, No Cut) ---
        canvas_orig, stats_orig = process_contours_pipeline(img_binary_closed, use_smart_cut=False)
        canvas_orig = add_stats_to_canvas(canvas_orig, stats_orig)

        # --- 3. Run SmartCut Pipeline (Thick, With Cut) ---
        canvas_smart_thick, stats_smart_thick = process_contours_pipeline(img_binary_closed, use_smart_cut=True)
        canvas_smart_thick = add_stats_to_canvas(canvas_smart_thick, stats_smart_thick)

        # --- 4. Run SmartCut Pipeline (Thinned, With Cut) ---
        canvas_smart_thinned, stats_smart_thinned = process_contours_pipeline(img_binary_thinned, use_smart_cut=True)
        canvas_smart_thinned = add_stats_to_canvas(canvas_smart_thinned, stats_smart_thinned)
        
        t_encode = time.time() - t1
        
        print(f"[Debug-Pipeline-Compare] Frame: {frame_id}, Capture: {t_capture*1000:.1f}ms, Process (All 3): {t_encode*1000:.1f}ms")
        
        # --- 5. Display ---
        cv2.imshow("1 - Binary (Thick)", img_binary_closed)
        cv2.imshow("2 - Binary (Thinned)", img_binary_thinned)
        cv2.imshow("3 - Final (Original Pipeline)", canvas_orig)
        cv2.imshow("4 - Final (SmartCut on Thick)", canvas_smart_thick)
        cv2.imshow("5 - Final (SmartCut on Thinned)", canvas_smart_thinned)
        
        key = cv2.waitKey(0)
        
        if key == 27 or key == ord('q'):
            print("[Debug-Pipeline-Compare] Quitting.")
            break
        else:
            frame_id += 1
            continue

    if VIDEO_MODE or CAMERA_BACKEND.upper() != "PICAM2":
        cap.release()
    else:
        try:
            picam.stop()
        except Exception:
            pass
    cv2.destroyAllWindows()