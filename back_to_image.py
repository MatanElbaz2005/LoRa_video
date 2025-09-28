import cv2
import numpy as np
import time
import os
import zlib
import zstandard as zstd

def decompress_and_reconstruct_image(input_compressed_path, output_image_path, image_shape=(512, 512)):
    """
    Decompress a zlib-compressed binary file of boundary points and reconstruct an image.

    The pipeline:
    1. Read and decompress the input file to retrieve boundary points (int16 x,y coordinates).
    2. Parse the header (number of boundaries and their lengths).
    3. Split points into individual boundaries.
    4. Create a blank canvas and draw boundaries as white lines (int32 points).
    5. Save the reconstructed image as a PNG.
    6. Print timing breakdown and file sizes.

    Args:
        input_compressed_path (str): Path to compressed binary file (e.g., output_compressed.bin.gz).
        output_image_path (str): Path to save reconstructed PNG image.
        image_shape (tuple): Output image dimensions (height, width), default (512, 512).

    Returns:
        None (prints timings, file sizes, and saves output image).
    """
    times = {}  # Store timing for each step

    # Read and decompress input file
    start = time.time()
    if not os.path.exists(input_compressed_path):
        raise ValueError(f"Cannot find compressed file: {input_compressed_path}")
    with open(input_compressed_path, 'rb') as f:
        compressed_data = f.read()
    decompressed_data = zstd.ZstdDecompressor().decompress(compressed_data)
    times['decompression'] = time.time() - start

    # Parse header and boundary points
    start = time.time()
    if len(decompressed_data) == 0:
        points = np.array([], dtype=np.int16).reshape(0, 2)
        boundary_lengths = []
    else:
        # First 4 bytes: num_boundaries (int32)
        num_boundaries = np.frombuffer(decompressed_data[:4], dtype=np.int32)[0]
        # Next 4 * num_boundaries bytes: boundary lengths (int32 array)
        boundary_lengths = np.frombuffer(decompressed_data[4:4 + 4 * num_boundaries], dtype=np.int32)
        # Remaining bytes: points (int16 x,y pairs)
        points = np.frombuffer(decompressed_data[4 + 4 * num_boundaries:], dtype=np.int16)
        if len(points) % 2 != 0:
            raise ValueError("Invalid decompressed data: must have even number of values for (x,y) pairs")
        points = points.reshape(-1, 2)  # Reshape to (N, 2)
    times['parse_points'] = time.time() - start

    # Split points into individual boundaries
    start = time.time()
    boundaries = []
    start_idx = 0
    for length in boundary_lengths:
        if start_idx + length <= len(points):
            boundary = points[start_idx:start_idx + length]
            if len(boundary) >= 3:  # Only include valid contours (3+ points)
                boundaries.append(boundary.astype(np.int32))  # Convert to int32 for cv2.polylines
            start_idx += length
    times['split_boundaries'] = time.time() - start

    # Create blank canvas (black background)
    start = time.time()
    canvas = np.zeros(image_shape, dtype=np.uint8)  # Binary image: 0 (black), 255 (white)

    # Draw boundaries
    if boundaries:
        for boundary in boundaries:
            contour = boundary.reshape(-1, 1, 2)  # Format for cv2.polylines: (N, 1, 2)
            cv2.polylines(canvas, [contour], isClosed=True, color=255, thickness=1)
    times['draw_boundaries'] = time.time() - start

    # Save reconstructed image as PNG
    start = time.time()
    cv2.imwrite(output_image_path, canvas)
    times['save_image'] = time.time() - start

    # Print timing breakdown
    print("Timing breakdown (seconds):")
    for step, t in times.items():
        print(f"  {step}: {t:.4f}")
    print(f"Total time: {sum(times.values()):.4f}")

    # Print file sizes
    compressed_size = os.path.getsize(input_compressed_path)
    output_image_size = os.path.getsize(output_image_path)
    print(f"Compressed input file size: {compressed_size} bytes")
    print(f"Reconstructed image file size: {output_image_size} bytes")
    print(f"Size ratio (compressed / image): {compressed_size / output_image_size:.2f}x" if output_image_size > 0 else "N/A")

# Example usage
if __name__ == "__main__":
    input_compressed_path = "C:\\Users\\Matan\\Documents\\Matan\\LoRa_video\\output_compressed.bin.zst"
    output_image_path = "reconstructed_image.png"
    decompress_and_reconstruct_image(input_compressed_path, output_image_path)