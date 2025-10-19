import cv2
import numpy as np
import time
import zstandard as zstd
import socket
import struct
from decode_frame import decode_frame, decode_frame_delta

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
    compressed = zstd.ZstdCompressor(level=3).compress(data_to_compress)
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
    
    frame_id = 0
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
        full_bytes = len(compressed_full)
        num_full_contours = len(simplified)
        # Build object-delta ops (N/D/X)
        ops = []          # (tag, id, payload)
        used_curr = set() # indices of curr_contours that were matched

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
                buf.extend(struct.pack(">H", pts.shape[0]))  # u16 length
                buf.extend(pts.tobytes())                    # int16 x,y pairs
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

        compressed_delta = zstd.ZstdCompressor(level=3).compress(bytes(buf))


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
            break
        
        frame_id += 1
    
    sock.close()
    cap.release()
    cv2.destroyAllWindows()
