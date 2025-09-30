import cv2
import numpy as np
import time
import zstandard as zstd
from decode_frame import decode_frame
from gui_helpers import build_dashboard, ControlState, render_controls, handle_controls_click, controls_to_params

def scharr_edges_u8(
    img_u8: np.ndarray,
    scharr_percentile: float = 90.0,
    scharr_preblur_sigma: float = 0.0,
    scharr_unsharp_amount: float = 1.0,
    scharr_magnitude: str = "l2",
    scharr_abs_threshold: float | None = None,
    scharr_scale: float = 1.0
) -> np.ndarray:
    # Input: single-channel uint8 (0..255); Output: uint8 mask (0/255)
    g = img_u8
    if scharr_preblur_sigma and scharr_preblur_sigma > 0:
        g = cv2.GaussianBlur(g, (0, 0), scharr_preblur_sigma)
    if scharr_unsharp_amount and scharr_unsharp_amount > 0:
        sigma_boost = max(0.6, scharr_preblur_sigma if scharr_preblur_sigma > 0 else 0.6)
        blur_boost = cv2.GaussianBlur(g, (0, 0), sigma_boost)
        g = cv2.addWeighted(g, 1.0 + scharr_unsharp_amount, blur_boost, -scharr_unsharp_amount, 0)
        g = np.clip(g, 0, 255).astype(np.uint8)

    gx = cv2.Scharr(g, cv2.CV_32F, 1, 0, scale=scharr_scale)
    gy = cv2.Scharr(g, cv2.CV_32F, 0, 1, scale=scharr_scale)

    if str(scharr_magnitude).lower() == "l1":
        mag = np.abs(gx) + np.abs(gy)
    else:
        mag = cv2.magnitude(gx, gy)

    if scharr_abs_threshold is not None:
        t = float(scharr_abs_threshold)
    else:
        t = np.percentile(mag, float(scharr_percentile))

    return (mag >= t).astype(np.uint8) * 255


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

def encode_frame(frame, percentile_pin=50,
                 canny_sigma=0.33, canny_aperture=3, canny_l2=True,
                 lap_percentile=95, median_ksize=3,
                 scharr_percentile=90, scharr_preblur_sigma=0.3,
                 scharr_unsharp_amount=1.0, scharr_magnitude="l2",
                 contour_mode=cv2.RETR_CCOMP, contour_method=cv2.CHAIN_APPROX_SIMPLE):

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
    orig_bgr = frame.copy()
    
    # Resize and convert to grayscale
    start = time.time()
    frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_AREA)
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    times['preprocess'] = time.time() - start

    # Median filter to reduce noise
    start = time.time()
    img_median = cv2.medianBlur(img_gray, int(median_ksize))
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
    thresh = np.percentile(lap_abs, float(lap_percentile))
    _, lap_binary = cv2.threshold(lap_abs.astype(np.uint8), int(thresh), 255, cv2.THRESH_BINARY)

    # Canny edge detection
    edges_canny = auto_canny(
        cv2.GaussianBlur(img_clahe, (3, 3), 0),
        sigma=float(canny_sigma),
        aperture_size=int(canny_aperture),
        l2=bool(canny_l2)
    )

    # OR of Laplacian + Canny ONLY (for separate reconstruction)
    img_binary_lc = cv2.bitwise_or(lap_binary, edges_canny)

    # Scharr magnitude + percentile threshold (sensitive to fine facial details)
    scharr_bin = scharr_edges_u8(
        img_clahe,
        scharr_percentile=float(scharr_percentile),
        scharr_preblur_sigma=float(scharr_preblur_sigma),
        scharr_unsharp_amount=float(scharr_unsharp_amount),
        scharr_magnitude=str(scharr_magnitude)
    )

    # Combine Laplacian, Canny, and Scharr with OR
    img_binary = cv2.bitwise_or(img_binary_lc, scharr_bin)
    times['edge_detection'] = time.time() - start

    # Morphology close (ALL = Lap+Canny+Scharr)
    img_binary_raw = img_binary.copy()
    start = time.time()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    times['morph_close'] = time.time() - start
    img_binary_closed = img_binary  # alias for clarity in GUI

    # Trace & compress for ALL (Lap + Canny + Scharr)
    start = time.time()
    contours_all, _ = cv2.findContours(img_binary_closed, int(contour_mode), int(contour_method))
    valid_contours_all = [c.squeeze().astype(np.float32) for c in contours_all if len(c) > 2]
    areas_all = [cv2.contourArea(c) for c in contours_all if len(c) > 2]
    num_ccs_all = len(valid_contours_all)

    sorted_idx_all = np.argsort(areas_all)[::-1]
    sorted_boundaries_all = [valid_contours_all[i] for i in sorted_idx_all]
    max_trace_all = max(1, int(num_ccs_all * (percentile_pin / 100)))
    boundaries_all = sorted_boundaries_all[:max_trace_all]

    simplified = [simplify_boundary(b, epsilon=2.0) for b in boundaries_all]
    simplified = [s for s in simplified if len(s.shape) == 2 and s.shape[0] >= 3]

    start_comp_all = time.time()
    if simplified:
        num_boundaries = len(simplified)
        boundary_lengths = [len(b) for b in simplified]
        all_points = np.vstack(simplified).astype(np.int16)
        header = np.array([num_boundaries] + boundary_lengths, dtype=np.int32)
        data_to_compress = header.tobytes() + all_points.tobytes()
    else:
        data_to_compress = np.array([], dtype=np.int16).tobytes()
    compressed = zstd.ZstdCompressor(level=3).compress(data_to_compress)
    times['compression_all'] = time.time() - start_comp_all

    # Morphology close for Lap+Canny ONLY (for separate reconstruction)
    img_binary_lc_raw = img_binary_lc.copy()
    img_binary_lc_closed = cv2.morphologyEx(img_binary_lc, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Trace & compress for Lap+Canny ONLY
    start = time.time()
    contours_lc, _ = cv2.findContours(img_binary_lc_closed, int(contour_mode), int(contour_method))
    valid_contours_lc = [c.squeeze().astype(np.float32) for c in contours_lc if len(c) > 2]
    areas_lc = [cv2.contourArea(c) for c in contours_lc if len(c) > 2]
    num_ccs_lc = len(valid_contours_lc)
    sorted_idx_lc = np.argsort(areas_lc)[::-1]
    sorted_boundaries_lc = [valid_contours_lc[i] for i in sorted_idx_lc]
    max_trace_lc = max(1, int(num_ccs_lc * (percentile_pin / 100)))
    boundaries_lc = sorted_boundaries_lc[:max_trace_lc]
    simplified_lc = [simplify_boundary(b, epsilon=2.0) for b in boundaries_lc]
    simplified_lc = [s for s in simplified_lc if len(s.shape) == 2 and s.shape[0] >= 3]

    if simplified_lc:
        num_boundaries_lc = len(simplified_lc)
        boundary_lengths_lc = [len(b) for b in simplified_lc]
        all_points_lc = np.vstack(simplified_lc).astype(np.int16)
        header_lc = np.array([num_boundaries_lc] + boundary_lengths_lc, dtype=np.int32)
        data_to_compress_lc = header_lc.tobytes() + all_points_lc.tobytes()
    else:
        data_to_compress_lc = np.array([], dtype=np.int16).tobytes()
    compressed_lc = zstd.ZstdCompressor(level=3).compress(data_to_compress_lc)
    times['compression_lc'] = time.time() - start

    # Print timing breakdown
    print("Timing breakdown (seconds):")
    for step, t in times.items():
        print(f"  {step}: {t:.4f}")
    print(f"Total time: {sum(times.values()):.4f}")

    # Print file sizes (since no file, use len(compressed))
    print(f"Compressed size (ALL): {len(compressed)} bytes")
    print(f"Compressed size (Lap+Canny): {len(compressed_lc)} bytes")

    # Collect intermediates for GUI
    intermediates = {
        'orig_bgr': orig_bgr,
        'lap_binary': lap_binary,
        'edges_canny': edges_canny,
        'edges_scharr': scharr_bin,
        'binary_or_raw': img_binary_raw,              # ALL (before close)
        'binary_closed': img_binary_closed,           # ALL (after close)
        'binary_lc_raw': img_binary_lc_raw,           # NEW: Lap+Canny only (before close)
        'binary_lc_closed': img_binary_lc_closed      # NEW: Lap+Canny only (after close)
    }
    return compressed, img_binary, simplified, (512, 512), intermediates, compressed_lc



# Example usage for live camera
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  # 0 for default webcam
    if not cap.isOpened():
        raise RuntimeError("Failed to open webcam")
    # Custom controls window (pretty panel)
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    state = ControlState()
    cv2.setMouseCallback("Controls", handle_controls_click, state)
    cv2.resizeWindow("Controls", 640, 540)
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        start_cycle = time.time()
        # read params from the pretty control panel state
        params = controls_to_params(state)
        sigma          = params["canny_sigma"]
        aperture       = params["canny_aperture"]
        l2_val         = params["canny_l2"]
        lap_pct        = params["lap_percentile"]
        median_ksize   = params["median_ksize"]
        contour_mode   = params["contour_mode"]
        contour_method = params["contour_method"]

        # --- NEW: Scharr params from GUI ---
        sch_pct        = params["scharr_percentile"]
        sch_pre        = params["scharr_preblur_sigma"]
        sch_unsharp    = params["scharr_unsharp_amount"]
        sch_mag        = params["scharr_magnitude"]

        compressed, img_binary, simplified, out_shape, im, compressed_lc = encode_frame(
            frame, percentile_pin=50,
            canny_sigma=sigma,
            canny_aperture=aperture,
            canny_l2=bool(l2_val),
            lap_percentile=int(lap_pct),
            median_ksize=int(median_ksize),
            scharr_percentile=int(sch_pct),
            scharr_preblur_sigma=float(sch_pre),
            scharr_unsharp_amount=float(sch_unsharp),
            scharr_magnitude=str(sch_mag),
            contour_mode=contour_mode,
            contour_method=contour_method
        )

        # Decode compressed data
        reconstructed = decode_frame(compressed, image_shape=out_shape)
        reconstructed_lc = decode_frame(compressed_lc, image_shape=out_shape)

        # Build a single dashboard image (all in one window)
        tiles = [
            im['orig_bgr'],
            im['lap_binary'],
            im['edges_canny'],
            im['edges_scharr'],
            im['binary_or_raw'],
            im['binary_lc_closed'],
            reconstructed_lc,
            reconstructed
        ]
        titles = [
            "Original (BGR)",
            "Laplacian (binary)",
            "Canny (binary)",
            "Scharr (binary)",
            "OR Lap+Canny+Scharr (raw)",
            "OR Lap+Canny (closed)",
            "Reconstructed (Lap+Canny)",
            "Reconstructed"
        ]
        dashboard = build_dashboard(
            tiles, titles,
            cols=3,
            tile_size=(300, 220),
            pad=16,
            bg_color=(24, 24, 28),
            title_bar_h=28,
            title_fg=(235, 235, 240),
            title_bg=(40, 40, 48),
            card_radius=10,
            shadow=3,
        )
        controls_img = render_controls(state)
        cv2.imshow("Controls", controls_img)
        cv2.namedWindow("Pipeline Dashboard", cv2.WINDOW_NORMAL)
        cv2.imshow("Pipeline Dashboard", dashboard)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_id += 1
    
    cap.release()
    cv2.destroyAllWindows()