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
    RAW_BASELINE_BGR = H * W * 3  # bytes for 512x512x3
    RAW_BASELINE_KB = RAW_BASELINE_BGR / 1024.0

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

        # FULL (hypothetical) payload size vs fixed baseline
        full_bytes = len(compressed_full)
        print(
            f"[FULL] payload={full_bytes} B ({_fmt_kb(full_bytes)}) "
            f"vs raw_512x512x3={RAW_BASELINE_KB:.2f} KB "
            f"→ { _pct(full_bytes, RAW_BASELINE_BGR) } of raw"
        )
        # DELTA (Contour XOR)
        reconstructed_full = decode_frame(compressed_full, image_shape=(512, 512))
        if prev_edges is None:
            delta_mask = reconstructed_full.copy()
        else:
            delta_mask = cv2.bitwise_xor(reconstructed_full, prev_edges)

        contours_d, _ = cv2.findContours(delta_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours_d = [c.squeeze().astype(np.float32) for c in contours_d if len(c) > 2]
        areas_d = [cv2.contourArea(c) for c in contours_d if len(c) > 2]
        idx_d = np.argsort(areas_d)[::-1]
        sorted_boundaries_d = [valid_contours_d[i] for i in idx_d]
        boundaries_d = sorted_boundaries_d
        simplified_d = [b.astype(np.float32).squeeze() for b in boundaries_d]
        simplified_d = [s for s in simplified_d if isinstance(s, np.ndarray) and len(s.shape)==2 and s.shape[0] >= 3]

        if simplified_d:
            num_boundaries_d = len(simplified_d)
            boundary_lengths_d = [len(b) for b in simplified_d]
            all_points_d = np.vstack(simplified_d).astype(np.int16)
            header_d = np.array([num_boundaries_d] + boundary_lengths_d, dtype=np.int32)
            data_to_compress_d = header_d.tobytes() + all_points_d.tobytes()
        else:
            data_to_compress_d = np.array([], dtype=np.int16).tobytes()
        compressed_delta = zstd.ZstdCompressor(level=3).compress(data_to_compress_d)

        # On-wire size includes 4B length header
        payload_len = len(compressed_delta)
        wire_len = 4 + payload_len
        sock.sendall(struct.pack(">I", payload_len) + compressed_delta)

        # Print DELTA consistently vs the same baseline (512x512x3)
        print(
            f"[DELTA] wire={wire_len} B ({_fmt_kb(wire_len)}) "
            f"| payload={payload_len} B ({_fmt_kb(payload_len)}) "
            f"vs raw_512x512x3={RAW_BASELINE_KB:.2f} KB "
            f"→ { _pct(wire_len, RAW_BASELINE_BGR) } of raw"
        )

        # Also show how much we saved vs sending FULL for this frame
        if full_bytes > 0:
            savings_bytes = full_bytes - payload_len
            savings_pct = 100.0 * (1.0 - (payload_len / full_bytes))
            print(
                f"[DELTA vs FULL] full_payload={_fmt_kb(full_bytes)} "
                f"→ delta_payload={_fmt_kb(payload_len)} "
                f"(saves {savings_bytes/1024.0:.2f} KB, {savings_pct:.1f}%)"
            )


        # DECODE DELTA
        reconstructed_delta, delta_canvas = decode_frame_delta(
            prev_edges, compressed_delta, image_shape=(512, 512)
        )

        prev_edges = reconstructed_delta.copy()

        cycle_time = time.time() - start_cycle
        # print(f"Dual encode-decode (full + delta): {cycle_time:.4f} seconds")

        # GUI
        cv2.imshow("Original Frame", frame)
        cv2.imshow("Delta mask (XOR to send)", delta_mask)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_id += 1
    
    sock.close()
    cap.release()
    cv2.destroyAllWindows()
