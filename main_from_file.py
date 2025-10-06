import cv2
import numpy as np
import time
import zstandard as zstd
from decode_frame import decode_frame

# =========[ Configure your input video here ]=========
VIDEO_PATH = r"C:\\Users\\Matan\\Documents\\Matan\\3_eye_awareness\\videos\\19700101_002302.mp4"
# =====================================================

def simplify_boundary(boundary, epsilon=2.0):
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

    1) Resize->Gray, 2) Median, 3) CLAHE, 4) Laplacian + Canny + Scharr (OR),
    5) findContours, 6) RDP simplify, 7) Zstd compress.
    """
    times = {}

    # Resize & gray
    start = time.time()
    frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_AREA)
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    times['preprocess'] = time.time() - start

    # Median
    start = time.time()
    img_median = cv2.medianBlur(img_gray, 3)
    times['median_filter'] = time.time() - start

    # CLAHE
    start = time.time()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_median)
    times['clahe'] = time.time() - start

    # Edges: Laplacian + Canny + Scharr
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

    # Morph close
    start = time.time()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    times['morph_close'] = time.time() - start

    # Contours
    start = time.time()
    contours, _ = cv2.findContours(img_binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c.squeeze().astype(np.float32) for c in contours if len(c) > 2]
    areas = [cv2.contourArea(c) for c in contours if len(c) > 2]
    num_ccs = len(valid_contours)
    sorted_idx = np.argsort(areas)[::-1]
    sorted_boundaries = [valid_contours[i] for i in sorted_idx]
    times['tracing_sorting'] = time.time() - start

    # Top percentile
    start = time.time()
    max_trace = max(1, int(num_ccs * (percentile_pin / 100)))
    boundaries = sorted_boundaries[:max_trace]

    # Simplify
    simplified = [simplify_boundary(b, epsilon=2.0) for b in boundaries]
    simplified = [s for s in simplified if len(s.shape) == 2 and s.shape[0] >= 3]
    times['simplification'] = time.time() - start

    # Compress
    start = time.time()
    if simplified:
        num_boundaries = len(simplified)
        boundary_lengths = [len(b) for b in simplified]
        all_points = np.vstack(simplified).astype(np.int16)
        header = np.array([num_boundaries] + boundary_lengths, dtype=np.int32)
        data_to_compress = header.tobytes() + all_points.tobytes()
    else:
        data_to_compress = np.array([], dtype=np.int16).tobytes()
    compressed = zstd.ZstdCompressor(level=3).compress(data_to_compress)
    times['compression'] = time.time() - start

    # print("Timing breakdown (seconds):")
    # for step, t in times.items():
    #     print(f"  {step}: {t:.4f}")
    # print(f"Total time: {sum(times.values()):.4f}")
    # print(f"Compressed size: {len(compressed)} bytes")

    return compressed, img_binary, simplified, (512, 512)

if __name__ == "__main__":
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {VIDEO_PATH}")

    delay_ms = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            # cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
            break

        start_cycle = time.time()
        compressed, img_binary, simplified, _ = encode_frame(frame, percentile_pin=50, scharr_percentile=92)
        reconstructed = decode_frame(compressed, image_shape=(512, 512))

        # Compute compressed size and compare to raw frame
        compressed_kb = len(compressed) / 1024
        raw_kb = frame.nbytes / 1024
        ratio = (compressed_kb / raw_kb) * 100 if raw_kb > 0 else 0
        print(f"Compressed frame: {compressed_kb:.2f} KB (vs raw {raw_kb:.2f} KB, {ratio:.2f}% of original)")
        # print(f"Full encode-decode cycle: {time.time() - start_cycle:.4f}s")

        # Windows
        cv2.imshow("Original Video", frame)
        cv2.imshow("Binary Video (OR: Laplacian + Canny + Scharr)", img_binary)
        cv2.imshow("Reconstructed Video", reconstructed)

        key = cv2.waitKey(delay_ms) & 0xFF
        if key == ord('q'):
            break
        # if key == ord('r'):
        #     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # restart

    cap.release()
    cv2.destroyAllWindows()
