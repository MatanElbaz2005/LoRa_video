import cv2
import numpy as np
import time
import os
import zlib
import zstandard as zstd

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

def process_and_compress_image(input_path, output_path, percentile_pin=50):
    """
    Process an image to extract, simplify, and compress connected component boundaries.

    The pipeline includes:
    1. Load image, resize to 512x512, and convert to grayscale.
    2. Apply median filter for noise reduction.
    3. Apply CLAHE for contrast enhancement.
    4. Detect edges using Laplacian with adaptive threshold (top 5%).
    5. Extract and sort connected components (CCs) by area using findContours.
    6. Simplify boundaries of top percentile_pin% CCs.
    7. Compress simplified points with zlib, including boundary lengths, and save to output_path.
    8. Print timing and file size metrics.

    Args:
        input_path (str): Path to input image (e.g., JPG, PNG).
        output_path (str): Path to save compressed binary file.
        percentile_pin (float): Percentage of top CCs to process (e.g., 50 for top 50%).

    Returns:
        None (prints timings, file sizes, and saves output file).
    """
    times = {}  # Store timing for each step

    # Load image
    start = time.time()
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Cannot read image: {input_path}")
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)  # Resize for consistency
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    times['load_image'] = time.time() - start

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
    _, img_binary = cv2.threshold(lap_abs.astype(np.uint8), int(thresh), 255, cv2.THRESH_BINARY)
    input_to_binary = "/home/matan/Documents/matan/LoRa/binary_img2.png"
    cv2.imwrite(input_to_binary, img_binary)
    times['edge_detection'] = time.time() - start

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

    # Save compressed data to output file
    with open(output_path, 'wb') as f:
        f.write(compressed)

    # Print timing breakdown
    print("Timing breakdown (seconds):")
    for step, t in times.items():
        print(f"  {step}: {t:.4f}")
    print(f"Total time: {sum(times.values()):.4f}")

    # Print file sizes
    original_size = os.path.getsize(input_path)
    compressed_size = os.path.getsize(output_path)
    print(f"Original file size: {original_size} bytes")
    print(f"Compressed file size: {compressed_size} bytes")
    print(f"Compression ratio: {original_size / compressed_size:.2f}x" if compressed_size > 0 else "N/A")

# Example usage
if __name__ == "__main__":
    input_path = "/home/matan/Documents/matan/LoRa/flat,750x,075,f-pad,750x1000,f8f8f8.u1.webp"
    output_path = "output_compressed.bin.zst"
    process_and_compress_image(input_path, output_path)