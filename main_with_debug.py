import cv2
import numpy as np
import time
import zstandard as zstd
from decode_frame import decode_frame
from gui_helpers import build_dashboard

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
    # Compute robust thresholds from image median
    v = np.median(img_u8)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(img_u8, lower, upper, apertureSize=aperture_size, L2gradient=l2)

def encode_frame(frame, percentile_pin=50):
    """
    Process a frame to extract, simplify, and compress connected component boundaries.

    The pipeline includes:
    1. Resize to 512x512, convert to grayscale.
    2. Apply median filter for noise reduction.
    3. Apply CLAHE for contrast enhancement.
    4. Detect edges using Laplacian with adaptive threshold (top 5%).
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
    # normalize for visualization (0..255)
    lap_norm = cv2.normalize(lap_abs, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    lap_vis = lap_norm.astype(np.uint8)
    thresh = np.percentile(lap_abs, 95)
    _, lap_binary = cv2.threshold(lap_abs.astype(np.uint8), int(thresh), 255, cv2.THRESH_BINARY)

    # Canny edge detection
    edges_canny = auto_canny(cv2.GaussianBlur(img_clahe, (3, 3), 0), 0.01)

    # Combine Laplacian and Canny with OR (any edge from either)
    img_binary = cv2.bitwise_or(lap_binary, edges_canny)
    times['edge_detection'] = time.time() - start

    # Morphology close
    img_binary_raw = img_binary.copy()
    start = time.time()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    times['morph_close'] = time.time() - start
    img_binary_closed = img_binary  # alias for clarity in GUI

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
    print("Timing breakdown (seconds):")
    for step, t in times.items():
        print(f"  {step}: {t:.4f}")
    print(f"Total time: {sum(times.values()):.4f}")

    # Print file sizes (since no file, use len(compressed))
    compressed_size = len(compressed)
    print(f"Compressed size: {compressed_size} bytes")

    # Collect intermediates for GUI
    intermediates = {
        'img_median': img_median,
        'img_clahe': img_clahe,
        'laplacian': lap_vis,             # 0..255 uint8 visualization
        'lap_binary': lap_binary,         # Laplacian-only edges (after threshold)
        'edges_canny': edges_canny,       # Canny-only edges
        'binary_or_raw': img_binary_raw,  # OR (Laplacian ∪ Canny) before closing
        'binary_closed': img_binary_closed
    }
    return compressed, img_binary, simplified, (512, 512), intermediates


# Example usage for live camera
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  # 0 for default webcam
    if not cap.isOpened():
        raise RuntimeError("Failed to open webcam")
    
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        start_cycle = time.time()
        compressed, img_binary, simplified, out_shape, im = encode_frame(frame, percentile_pin=50)
        # Decode compressed data
        reconstructed = decode_frame(compressed, image_shape=out_shape)

        # Build a single dashboard image (all in one window)
        tiles = [
            im['img_median'],
            im['img_clahe'],
            im['laplacian'],       # visualization of |Laplacian|
            im['lap_binary'],      # Laplacian-only (binary)
            im['edges_canny'],     # Canny-only (binary)
            im['binary_or_raw'],   # OR (before closing)
            im['binary_closed'],   # OR (after closing)
            reconstructed
        ]
        titles = [
            "Median (gray)",
            "CLAHE (gray)",
            "Laplacian |abs| (0..255)",
            "Laplacian (binary)",
            "Canny (binary)",
            "OR Laplacian Canny (raw)",
            "OR Laplacian Canny (closed)",
            "Reconstructed"
        ]
        dashboard = build_dashboard(tiles, titles, cols=3, tile_size=(256, 256), pad=10)

        cv2.imshow("Pipeline Dashboard", dashboard)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_id += 1
    
    cap.release()
    cv2.destroyAllWindows()