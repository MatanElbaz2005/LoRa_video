import cv2
import numpy as np
import time
import zstandard as zstd
import socket
import struct
from decode_frame import decode_frame, decode_frame_delta

# Mode switch
FULL_MODE = True
FULL_BATCH_ENABLE = False
FULL_BATCH_COUNT = 3  # number of full frames to group per message (>=1)
COLORED_CONTOURS = True  # visualize contour efficiency

def simplify_boundary(boundary, epsilon=2.0):
    """
    Simplify a boundary using OpenCV's implementation of Ramer-Douglas-Peucker (RDP) algorithm.

    Args:
        boundary (np.ndarray): Array of (x,y) points representing a contour.
        epsilon (float): RDP simplification parameter (higher = more aggressive).

    Returns:
        np.ndarray: Simplified boundary points.
    """
    boundary = boundary.astype(np.float32)
    simp = cv2.approxPolyDP(boundary, epsilon=epsilon, closed=True)
    return simp.squeeze()

def auto_canny(img_u8: np.ndarray, sigma: float = 0.33,
               aperture_size: int = 3, l2: bool = True) -> np.ndarray:
    v = np.median(img_u8)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(img_u8, lower, upper, apertureSize=aperture_size, L2gradient=l2)

def encode_frame(frame, percentile_pin=50, scharr_percentile=92):
    """
    Process a frame to extract, simplify, and compress connected component boundaries.

    The pipeline includes:
    1. Resize to 512x512, convert to grayscale.
    2. Apply median filter for noise reduction.
    3. Apply CLAHE for contrast enhancement.
    4. Detect edges using Laplacian (percentile), Canny (auto), and Scharr (percentile), then OR-combine.
    5. Extract and sort connected components (CCs) by area using findContours.
    6. Simplify boundaries of top percentile_pin% CCs.
    7. Compress simplified points with Zstandard, including boundary lengths.

    Args:
        frame (np.ndarray): Input frame from camera.
        percentile_pin (float): Percentage of top CCs to process (e.g., 50 for top 50%).

    Returns:
        bytes: Compressed boundary data.
        np.ndarray: Binary image.
        list: Simplified polygons for reconstruction.
    """
    times = {}  # Store timing for each step

    # Resize and convert to grayscale
    start = time.time()
    frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_AREA)
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    times['preprocess'] = time.time() - start

    # Median filter to reduce noise
    start = time.time()
    img_median = cv2.medianBlur(img_gray, 3)
    times['median_filter'] = time.time() - start

    # Apply real CLAHE for adaptive contrast enhancement
    start = time.time()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_median)
    times['clahe'] = time.time() - start

    # Edge detection with Laplacian and adaptive threshold (top 5%)
    start = time.time()
    lap = cv2.Laplacian(img_clahe, cv2.CV_64F)
    lap_abs = np.abs(lap)
    thresh = np.percentile(lap_abs, 95)
    _, lap_binary = cv2.threshold(lap_abs.astype(np.uint8), int(thresh), 255, cv2.THRESH_BINARY)

    # Canny edge detection
    edges_canny = auto_canny(cv2.GaussianBlur(img_clahe, (3, 3), 0))

    # NEW: Scharr magnitude + percentile threshold (sensitive to fine internal details)
    gx = cv2.Scharr(img_clahe, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(img_clahe, cv2.CV_32F, 0, 1)
    scharr_mag = cv2.magnitude(gx, gy)
    scharr_t = np.percentile(scharr_mag, float(scharr_percentile))
    scharr_bin = (scharr_mag >= scharr_t).astype(np.uint8) * 255

    # Combine Laplacian, Canny, and Scharr with OR
    img_binary = cv2.bitwise_or(cv2.bitwise_or(lap_binary, edges_canny), scharr_bin)
    times['edge_detection'] = time.time() - start

    # Morphology close
    start = time.time()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    times['morph_close'] = time.time() - start
    # Trace boundaries with findContours
    start = time.time()
    contours, _ = cv2.findContours(img_binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c.squeeze().astype(np.float32) for c in contours if len(c) > 2]
    areas = [cv2.contourArea(c) for c in contours if len(c) > 2]
    num_ccs = len(valid_contours)

    # Sort by area (descending)
    sorted_idx = np.argsort(areas)[::-1]
    sorted_boundaries = [valid_contours[i] for i in sorted_idx]
    times['tracing_sorting'] = time.time() - start

    # Pin to top percentile (e.g., 50%)
    start = time.time()
    max_trace = max(1, int(num_ccs * (percentile_pin / 100)))
    boundaries = sorted_boundaries[:max_trace]

    # Simplify boundaries
    simplified = [simplify_boundary(b, epsilon=2.0) for b in boundaries]
    # Filter out invalid simplified contours (e.g., < 3 points)
    simplified = [s for s in simplified if len(s.shape) == 2 and s.shape[0] >= 3]
    times['simplification'] = time.time() - start

    # Prepare data for compression: store number of boundaries and lengths
    start = time.time()
    if simplified:
        num_boundaries = len(simplified)
        boundary_lengths = [len(b) for b in simplified]
        all_points = np.vstack(simplified).astype(np.int16)
        # Create header: num_boundaries (int32) + lengths (int32 array)
        header = np.array([num_boundaries] + boundary_lengths, dtype=np.int32)
        data_to_compress = header.tobytes() + all_points.tobytes()
    else:
        data_to_compress = np.array([], dtype=np.int16).tobytes()
    compressed = zstd.ZstdCompressor(level=22).compress(data_to_compress)
    times['compression'] = time.time() - start

    # Print timing breakdown
    # print("Timing breakdown (seconds):")
    for step, t in times.items():
        # print(f"  {step}: {t:.4f}")
        pass
    # print(f"Total time: {sum(times.values()):.4f}")
    return compressed, img_binary, simplified, (512, 512)


# Example usage for live camera
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  # 0 for default webcam
    # --- Network setup ---
    HOST = "127.0.0.1"
    PORT = 5001
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT))
    print(f"[Sender] Connected to receiver on {HOST}:{PORT}")
    if not cap.isOpened():
        raise RuntimeError("Failed to open webcam")

    # Consistent baseline (512x512x3) and helpers
    H, W = 512, 512
    RAW_BASELINE_BGR = H * W * 3
    RAW_BASELINE_KB = RAW_BASELINE_BGR / 1024.0

    # Object-Delta state & matching params
    NEXT_ID = 1                      # counter for new contour IDs
    free_ids = set()                 # recycled IDs released from deleted contours
    prev_contours_by_id = {}         # id -> np.ndarray of points
    MATCH_MAX_DIST = 12.0            # max centroid distance (pixels)
    MATCH_MIN_IOU  = 0.15            # min IoU (tiny raster) to accept a match
    REFRESH_MIN_IOU = 0.60           # threshold below which we resend full contour (N)
    MAX_POINT_DELTA = 20             # if max(|Δx|,|Δy|) > this -> fallback to N

    # --- helpers for matching ---
    def get_new_id():
        global NEXT_ID
        if free_ids:
            return free_ids.pop()

        start = NEXT_ID
        while True:
            cid = NEXT_ID & 0xFFFF
            NEXT_ID = (NEXT_ID + 1) & 0xFFFF
            if cid not in prev_contours_by_id:
                return cid
            if NEXT_ID == start:
                raise RuntimeError("ID space exhausted: no free 16-bit IDs available")

    def release_id(cid: int):
        """Release an ID back into the free pool when contour disappears."""
        cid &= 0xFFFF
        if cid in prev_contours_by_id:
            return
        free_ids.add(cid)


    def _centroid(poly: np.ndarray) -> np.ndarray:
        return np.mean(poly, axis=0) if poly.size else np.array([0.0, 0.0], dtype=np.float32)

    def _small_mask(poly: np.ndarray, size=64) -> np.ndarray:
        m = np.zeros((size, size), dtype=np.uint8)
        if poly.ndim == 2 and poly.shape[0] >= 3:
            pts = poly.copy()
            pts[:,0] = np.clip(pts[:,0] * (size-1) / (W-1), 0, size-1)
            pts[:,1] = np.clip(pts[:,1] * (size-1) / (H-1), 0, size-1)
            cv2.fillPoly(m, [pts.astype(np.int32)], 255)
        return m

    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        inter = np.count_nonzero(cv2.bitwise_and(a, b))
        union = np.count_nonzero(cv2.bitwise_or(a, b))
        return (inter/union) if union else 0.0


    def _fmt_kb(n_bytes: int) -> str:
        return f"{n_bytes / 1024.0:.2f} KB"

    def _pct(n_bytes: int, denom_bytes: int) -> str:
        return f"{(n_bytes / denom_bytes) * 100.0:.2f}%"
    
    def contour_metric(pts: np.ndarray, mode: bytes) -> tuple[float, int, float]:
        """
        Compute geometry-only efficiency metric and carry size stats.

        Returns:
            len_per_point (float): geometric length / number of points  [px/pt] (higher = more efficient)
            weight_bytes (int)   : encoded bytes for this contour (header+payload) for summary panes
            geom_len_px (float)  : geometric length (perimeter-like) in pixels
        """
        if pts.shape[0] < 2:
            return 0.0, 0, 0.0

        L = pts.shape[0]
        # payload size by mode (for summary only)
        if mode == b"8":
            payload_bytes = 4 + (L-1)*2
        elif mode == b"6":
            payload_bytes = 4 + (L-1)*4
        else:
            payload_bytes = L * 4
        weight_bytes = 3 + payload_bytes

        # geometric length (close polyline)
        diff = np.diff(pts.astype(np.float32), axis=0)
        geom_len = float(np.sum(np.sqrt((diff**2).sum(axis=1))))
        if L >= 3:
            geom_len += float(np.linalg.norm(pts[-1] - pts[0]))

        L = max(L, 1)  # guard
        len_per_point = geom_len / float(L)  # px per vertex (higher == better)
        return len_per_point, int(weight_bytes), geom_len

    
    def pack_full_relative_raw(contours):
        """
        Pack a list of contours into bytes (no compression).
        Layout per contour: [u16 L][1B mode][payload]
        """
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
        return bytes(buf), (cnt_A, cnt_8, cnt_6)
    
    def pack_full_relative(contours):
        """
        Pack a list of contours using the same per-contour mode as 'N':
        for each contour: [u16 L][1B mode][payload].
        Return compressed bytes and per-mode counts for debug.
        """
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

        comp = zstd.ZstdCompressor(level=22).compress(bytes(buf))
        return comp, (cnt_A, cnt_8, cnt_6)

        
    def encode_contour_relative(pts_i16: np.ndarray):
        """
        Decide the best encoding for a 'NEW' contour:
        - '8': first point int16 (x0,y0), then (dx,dy) as int8 for the rest
        - '6': first point int16 (x0,y0), then (dx,dy) as int16 for the rest
        - 'A': absolute int16 pairs (legacy)
        Returns: (mode_byte: bytes, payload: bytes)
        """
        assert pts_i16.ndim == 2 and pts_i16.shape[1] == 2 and pts_i16.dtype == np.int16
        L = pts_i16.shape[0]
        if L < 1:
            return b"A", b""  # shouldn't happen with valid contours

        # First point absolute (int16)
        head = pts_i16[0:1].astype(np.int16)

        if L == 1:
            return b"A", head.tobytes()

        # Compute deltas vs previous point
        prev = pts_i16[:-1].astype(np.int32)
        curr = pts_i16[1:].astype(np.int32)
        deltas = (curr - prev)  # int32 range

        max_abs = np.max(np.abs(deltas))
        if max_abs <= 127:
            # '8' mode: int8 deltas
            return b"8", head.tobytes() + deltas.astype(np.int8).tobytes()
        elif max_abs <= 32767:
            # '6' mode: int16 deltas
            return b"6", head.tobytes() + deltas.astype(np.int16).tobytes()
        else:
            # Fallback: absolute
            return b"A", pts_i16.tobytes()
    
    frame_id = 0
    # batch buffer for FULL mode
    _full_batch_raw = []
    prev_edges = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        start_cycle = time.time()

        # FULL FRAME: encode + decode
        compressed_full, img_binary, simplified, _ = encode_frame(
            frame, percentile_pin=50, scharr_percentile=92
        )
        curr_contours = simplified

        # FULL (hypothetical) payload size vs fixed baseline
        full_rel_comp, (full_A, full_8, full_6) = pack_full_relative(curr_contours)
        full_bytes = len(full_rel_comp)  # size if we send full frame
        num_full_contours = len(curr_contours)

        if FULL_MODE:
            if FULL_BATCH_ENABLE:
                # BATCH ON: accumulate RAW frames, compress together with Zstd
                full_raw, (full_A, full_8, full_6) = pack_full_relative_raw(curr_contours)
                _full_batch_raw.append(full_raw)

                comp_single_for_stats = zstd.ZstdCompressor(level=22).compress(full_raw)
                print(
                    f"[FULL] {_fmt_kb(len(comp_single_for_stats))} "
                    f"({_pct(len(comp_single_for_stats), RAW_BASELINE_BGR)} of raw) "
                    f"contours={num_full_contours} "
                    f"raw={RAW_BASELINE_KB:.2f} KB "
                    f"batch={len(_full_batch_raw)}/{FULL_BATCH_COUNT}"
                )
                print(f"[FULL modes] A={full_A} 8={full_8} 6={full_6}")

                if len(_full_batch_raw) >= max(1, int(FULL_BATCH_COUNT)):
                    agg = bytearray()
                    N = len(_full_batch_raw)
                    agg.extend(struct.pack(">H", N))
                    for fr in _full_batch_raw:
                        agg.extend(struct.pack(">I", len(fr)))
                        agg.extend(fr)

                    comp = zstd.ZstdCompressor(level=22).compress(bytes(agg))
                    sock.sendall(struct.pack(">I", len(comp)) + comp)

                    print(
                        f"[SEND][FULL][BATCH] total={len(comp)} B ({_fmt_kb(len(comp))}) "
                        f"(raw={_fmt_kb(len(agg))}) "
                        f"N={N} frames"
                    )
                    _full_batch_raw.clear()

                frame_id += 1
                continue

            else:
                # BATCH OFF: classic single-frame FULL
                full_rel_comp, (full_A, full_8, full_6) = pack_full_relative(curr_contours)
                full_bytes = len(full_rel_comp)
                sock.sendall(struct.pack(">I", full_bytes) + full_rel_comp)

                print(
                    f"[FULL] {_fmt_kb(full_bytes)} ({_pct(full_bytes, RAW_BASELINE_BGR)} of raw) "
                    f"contours={num_full_contours} "
                    f"raw={RAW_BASELINE_KB:.2f} KB"
                )
                print(f"[FULL modes] A={full_A} 8={full_8} 6={full_6}")

                # visualize contour efficiency if enabled
                if COLORED_CONTOURS and FULL_MODE and not FULL_BATCH_ENABLE and frame_id % 10 == 0:
                    eff_canvas = np.zeros((H, W, 3), dtype=np.uint8)

                    # palette (BGR): white, blue, yellow, orange, red
                    PALETTE = [
                        (255,255,255),  # Top 20% (most efficient)
                        (255,  0,  0),  # 20–40%  (blue)
                        (  0,255,255),  # 40–60%  (yellow)
                        (  0,165,255),  # 60–80%  (orange)
                        (  0,  0,255),  # Bottom 20% (least efficient)
                    ]
                    BAND_LABELS = ["Top 20%", "20-40%", "40-60%", "60-80%", "Bottom 20%"]

                    # first pass: compute metric per contour
                    metrics = []  # list of (len_per_point, weight_bytes, geom_len, poly)
                    for poly in curr_contours:
                        mode, _ = encode_contour_relative(poly.astype(np.int16))
                        m, w_bytes, g_len = contour_metric(poly, mode)
                        metrics.append((m, w_bytes, g_len, poly))

                    if metrics:
                        vals = np.array([m for (m, _, _, _) in metrics], dtype=np.float32)
                        q = np.quantile(vals, [0.2, 0.4, 0.6, 0.8])  # relative thresholds

                    # stats per relative band (indices 0..4)
                    band_stats = {i: {"weight":0, "contours":0, "pixels":0.0} for i in range(5)}

                    # second pass: assign band per contour, draw, and accumulate stats
                    for (m, w_bytes, g_len, poly) in metrics:
                        # band index by quantiles (higher m == better)
                        if   m >= q[3]: band_idx = 0        # top 20% (white)
                        elif m >= q[2]: band_idx = 1        # 20–40% (blue)
                        elif m >= q[1]: band_idx = 2        # 40–60% (yellow)
                        elif m >= q[0]: band_idx = 3        # 60–80% (orange)
                        else:           band_idx = 4        # bottom 20% (red)

                        color = PALETTE[band_idx]
                        cv2.polylines(eff_canvas, [poly.astype(np.int32).reshape(-1,1,2)], True, color, 1)

                        band_stats[band_idx]["weight"]   += w_bytes
                        band_stats[band_idx]["contours"] += 1
                        band_stats[band_idx]["pixels"]   += g_len


                    # draw color scale legend on the right
                    legend_w = 140
                    legend = np.zeros((H, legend_w, 3), dtype=np.uint8)
                    cv2.putText(legend, "Relative bands", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1, cv2.LINE_AA)

                    y0 = 50
                    for i, label in enumerate(BAND_LABELS):
                        y = y0 + i*40
                        cv2.rectangle(legend, (10,y-10), (30,y+10), PALETTE[i], -1)
                        cv2.putText(legend, label, (40,y+6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

                    combined = np.hstack((eff_canvas, legend))

                    # mean of the new metric: px per point
                    if metrics:
                        mean_len_per_point = float(np.mean([m for (m,_,_,_) in metrics]))
                    else:
                        mean_len_per_point = 0.0

                    cv2.putText(combined, f"Mean length/point: {mean_len_per_point:.2f} px/pt", (10, H-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                    cv2.imshow("Contour efficiency (sender)", combined)
                    # Summary table (per frame)
                    # totals (indices 0..4)
                    total_weight   = sum(band_stats[i]["weight"]   for i in range(5)) or 1
                    total_contours = sum(band_stats[i]["contours"] for i in range(5)) or 1
                    total_pixels   = sum(band_stats[i]["pixels"]   for i in range(5)) or 1.0

                    summary_w, summary_h = 760, 260
                    summary_img = np.zeros((summary_h, summary_w, 3), dtype=np.uint8)

                    cv2.putText(summary_img, "Contour Efficiency - Relative Bands Summary", (18,30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

                    x0, y0 = 18, 60
                    col_x = [18, 170, 310, 450, 590, 690]  # Band, Weight (KB), #Contours, Pixels, %Weight, %Contours
                    headers = ["Band (relative)", "Weight (KB)", "#Contours", "Pixels", "%Weight", "%Contours"]
                    for i, h in enumerate(headers):
                        cv2.putText(summary_img, h, (col_x[i], y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1, cv2.LINE_AA)

                    row_h = 30
                    for idx in range(5):
                        clr   = PALETTE[idx]
                        label = BAND_LABELS[idx]
                        y = y0 + 10 + (idx+1)*row_h

                        w_bytes = band_stats[idx]["weight"]
                        c_count = band_stats[idx]["contours"]
                        p_pixels = band_stats[idx]["pixels"]

                        pct_w = 100.0 * (w_bytes / total_weight)
                        pct_c = 100.0 * (c_count  / total_contours)

                        # swatch + label
                        cv2.rectangle(summary_img, (col_x[0], y-12), (col_x[0]+24, y+8), clr, -1)
                        cv2.putText(summary_img, label, (col_x[0]+32, y+6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

                        # numbers
                        kb = w_bytes/1024.0
                        cv2.putText(summary_img, f"{kb:6.2f}",               (col_x[1], y+6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
                        cv2.putText(summary_img, f"{c_count:6d}",            (col_x[2], y+6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
                        cv2.putText(summary_img, f"{int(round(p_pixels)):6d}",(col_x[3], y+6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
                        cv2.putText(summary_img, f"{pct_w:5.1f}%",           (col_x[4], y+6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
                        cv2.putText(summary_img, f"{pct_c:5.1f}%",           (col_x[5], y+6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

                    cv2.imshow("Contour efficiency summary", summary_img)
                    cv2.waitKey(1)

                frame_id += 1
                continue


        ops = []
        used_curr = set()
        n_mode_A = 0
        n_mode_8 = 0
        n_mode_6 = 0
        count_V = 0
        count_N_delta = 0
        count_N_iou = 0
        
        if not prev_contours_by_id:
            # I-frame: send all as NEW
            for poly in curr_contours:
                cid = get_new_id()
                prev_contours_by_id[cid] = poly.copy()
                ops.append(("N", cid, poly))
            frame_type = b"I"
        else:
            # P-frame: match previous contours to current ones
            frame_type = b"P"
            curr_centroids = [_centroid(p) for p in curr_contours]
            curr_masks64  = [_small_mask(p, 64) for p in curr_contours]

            for cid, prev_poly in list(prev_contours_by_id.items()):
                pc = _centroid(prev_poly)
                pm = _small_mask(prev_poly, 64)

                # nearest by centroid
                best_j, best_d = None, 1e9
                for j, cc in enumerate(curr_contours):
                    if j in used_curr: continue
                    d = np.linalg.norm(curr_centroids[j] - pc)
                    if d < best_d:
                        best_d, best_j = d, j

                ok = False
                if best_j is not None and best_d <= MATCH_MAX_DIST:
                    iou = _iou(pm, curr_masks64[best_j])
                    ok = (iou >= MATCH_MIN_IOU)

                if ok:
                    shift = _centroid(curr_contours[best_j]) - pc
                    dx, dy = int(round(shift[0])), int(round(shift[1]))

                    tx_prev = prev_poly.copy()
                    if dx != 0 or dy != 0:
                        tx_prev[:, 0] += dx
                        tx_prev[:, 1] += dy

                    iou2 = _iou(_small_mask(tx_prev, 64), curr_masks64[best_j])

                    if iou2 >= REFRESH_MIN_IOU:
                        prev_pts = tx_prev.astype(np.int16)
                        curr_pts = curr_contours[best_j].astype(np.int16)

                        if prev_pts.shape[0] != curr_pts.shape[0]:
                            L = min(prev_pts.shape[0], curr_pts.shape[0])
                            prev_pts = prev_pts[:L]
                            curr_pts = curr_pts[:L]

                        deltas = curr_pts - prev_pts
                        max_delta = np.max(np.abs(deltas))

                        if max_delta <= MAX_POINT_DELTA:
                            ops.append(("D", cid, ("V", deltas.astype(np.int8))))
                            prev_contours_by_id[cid] = curr_pts.copy()
                            count_V += 1
                        else:
                            ops.append(("N", cid, curr_contours[best_j]))
                            prev_contours_by_id[cid] = curr_contours[best_j].copy()
                            count_N_delta += 1
                    else:
                        ops.append(("N", cid, curr_contours[best_j]))
                        prev_contours_by_id[cid] = curr_contours[best_j].copy()
                        count_N_iou += 1

                    used_curr.add(best_j)
                else:
                    ops.append(("X", cid, None))
                    prev_contours_by_id.pop(cid, None)
                    release_id(cid)


            # any unmatched current contours are NEW
            for j, poly in enumerate(curr_contours):
                if j in used_curr: 
                    continue
                cid = get_new_id()
                prev_contours_by_id[cid] = poly.copy()
                ops.append(("N", cid, poly))

        # --- Pack ops and compress (Zstd) ---
        buf = bytearray()
        buf.extend(frame_type)                      # 1B
        buf.extend(struct.pack(">H", len(ops)))     # u16 count

        for tag, cid, payload in ops:
            buf.extend(tag.encode("ascii"))         # 'N'/'D'/'X'
            buf.extend(struct.pack(">H", cid))      # u16 id
            if tag == "N":
                pts = payload.astype(np.int16)
                L = pts.shape[0]
                buf.extend(struct.pack(">H", L))    # u16 length

                # NEW: pick mode and count it
                mode, encoded = encode_contour_relative(pts)
                if mode == b"A":
                    n_mode_A += 1
                elif mode == b"8":
                    n_mode_8 += 1
                elif mode == b"6":
                    n_mode_6 += 1

                buf.extend(mode)                    # 1B: b'8'/b'6'/b'A'
                buf.extend(encoded)                 # payload per mode

            elif tag == "D":
                model, pdata = payload
                if model == "T":
                    buf.extend(b"T")
                    dx, dy = pdata
                    buf.extend(struct.pack(">hh", dx, dy))
                elif model == "V":
                    buf.extend(b"V")
                    L = pdata.shape[0]
                    buf.extend(struct.pack(">H", L))  # length
                    buf.extend(pdata.tobytes())       # Δx,Δy as int8
            elif tag == "X":
                pass

        compressed_delta = zstd.ZstdCompressor(level=22).compress(bytes(buf))


        # On-wire size includes 4B length header
        payload_len = len(compressed_delta)
        wire_len = 4 + payload_len
        sock.sendall(struct.pack(">I", payload_len) + compressed_delta)

        # breakdown counts for delta ops
        nN = sum(1 for t, _, _ in ops if t == "N")
        nD = sum(1 for t, _, _ in ops if t == "D")
        nX = sum(1 for t, _, _ in ops if t == "X")

        # single-line summary (FULL + DELTA)
        print(
            f"[FULL] {_fmt_kb(full_bytes)} ({_pct(full_bytes, RAW_BASELINE_BGR)} of raw) "
            f"contours={num_full_contours} "
            f"[DELTA] {_fmt_kb(payload_len)} ({_pct(payload_len, RAW_BASELINE_BGR)} of raw) "
            f"raw={RAW_BASELINE_KB:.2f} KB"
        )

        # second line: delta breakdown
        print(f"[DELTA breakdown] N={nN} D={nD} X={nX} ops={len(ops)}")
        print(f"[DELTA breakdown] N(Δ>20)={count_N_delta} N(IoU)={count_N_iou} D(V)={count_V} total_ops={len(ops)}")
        print(f"[N modes] A={n_mode_A} 8={n_mode_8} 6={n_mode_6}")
        print(f"[FULL modes] A={full_A} 8={full_8} 6={full_6}")


        # Build a local preview (what receiver will see) by drawing current ID map
        preview = np.zeros((H, W), dtype=np.uint8)
        for poly in prev_contours_by_id.values():
            cv2.polylines(preview, [poly.astype(np.int32).reshape(-1,1,2)], True, 255, 1)
        prev_edges = preview  # keep for your own on-screen continuity

        # Draw full reconstruction (no deltas) for comparison
        full_preview = np.zeros((H, W), dtype=np.uint8)
        for poly in curr_contours:
            cv2.polylines(full_preview, [poly.astype(np.int32).reshape(-1,1,2)], True, 255, 1)

        cv2.imshow("Object-Delta preview (sender)", preview)
        cv2.imshow("Full contours (no delta)", full_preview)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # flush remaining partial batch (only when batching is enabled)
            if FULL_MODE and FULL_BATCH_ENABLE and _full_batch_raw:
                agg = bytearray()
                N = len(_full_batch_raw)
                agg.extend(struct.pack(">H", N))
                for fr in _full_batch_raw:
                    agg.extend(struct.pack(">I", len(fr)))
                    agg.extend(fr)
                comp = zstd.ZstdCompressor(level=22).compress(bytes(agg))
                sock.sendall(struct.pack(">I", len(comp)) + comp)
                print(f"[FULL-batch-FLUSH] N={N} agg_raw={len(agg)}B sent_zstd={len(comp)}B")
                _full_batch_raw.clear()
            break
        
        frame_id += 1
    
    sock.close()
    cap.release()
    cv2.destroyAllWindows()
