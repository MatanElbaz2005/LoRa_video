import cv2
import numpy as np
import time
import os
import zlib
import zstandard as zstd
import socket
import struct

def recv_exact(sock, n):
    data = b""
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def decode_frame(compressed_data, image_shape=(512, 512)):
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

    # Print timing breakdown
    # print("Timing breakdown (seconds):")
    for step, t in times.items():
        # print(f"  {step}: {t:.4f}")
        pass
    # print(f"Total time: {sum(times.values()):.4f}")

    return canvas

def decode_frame_delta(prev_canvas: np.ndarray, compressed_delta: bytes, image_shape=(512, 512)):
    """
    Decode a delta (contour XOR) and apply it over prev_canvas.
    Returns:
        updated_canvas: prev_canvas XOR delta_canvas
        delta_canvas:   the drawn delta-only canvas (for debugging/vis)
    """
    # 1) Decode delta to a binary canvas
    delta_canvas = decode_frame(compressed_delta, image_shape=image_shape)

    # 2) Apply XOR with the previous reconstructed canvas
    if prev_canvas is None:
        # first frame delta == full frame
        updated = delta_canvas.copy()
    else:
        updated = cv2.bitwise_xor(prev_canvas, delta_canvas)

    return updated, delta_canvas

# --- Receiver main loop ---
if __name__ == "__main__":
    HOST = "127.0.0.1"
    PORT = 5001

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(1)
    print(f"[Receiver] Listening on {HOST}:{PORT} ...")
    conn, addr = server.accept()
    print(f"[Receiver] Connected by {addr}")

    prev_canvas = None
    while True:
        # --- Read 4-byte length header ---
        length_data = recv_exact(conn, 4)
        if not length_data: break
        msg_len = struct.unpack(">I", length_data)[0]

        payload = recv_exact(conn, msg_len)
        if not payload: break

        # decompress the object-delta payload
        decompressed = zstd.ZstdDecompressor().decompress(payload)

        # parse: [1B FrameType][u16 Count][ops...]
        p = 0
        frame_type = decompressed[p:p+1]; p += 1
        (count,) = struct.unpack_from(">H", decompressed, p); p += 2

        # persistent map of id -> contour
        if 'contours_by_id' not in globals():
            contours_by_id = {}

        for _ in range(count):
            tag = decompressed[p:p+1]; p += 1
            (cid,) = struct.unpack_from(">H", decompressed, p); p += 2

            if tag == b"N":
                (L,) = struct.unpack_from(">H", decompressed, p); p += 2
                pts = np.frombuffer(decompressed, dtype=np.int16, count=2*L, offset=p)
                p += 2*L*2
                pts = pts.reshape(-1,2).astype(np.int32)
                contours_by_id[cid] = pts

            elif tag == b"D":
                model = decompressed[p:p+1]; p += 1
                if model == b"T":
                    dx, dy = struct.unpack_from(">hh", decompressed, p); p += 4
                    if cid in contours_by_id:
                        c = contours_by_id[cid].copy()
                        c[:,0] += dx; c[:,1] += dy
                        contours_by_id[cid] = c
                else:
                    # reserved for future affine model
                    pass

            elif tag == b"X":
                if cid in contours_by_id:
                    contours_by_id.pop(cid)

        # draw current frame from the map
        canvas = np.zeros((512, 512), dtype=np.uint8)
        for pts in contours_by_id.values():
            cv2.polylines(canvas, [pts.reshape(-1,1,2)], True, 255, 1)

        prev_canvas = canvas.copy()
        cv2.imshow("Reconstructed (Object-Delta)", canvas)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

    conn.close()
    server.close()
    cv2.destroyAllWindows()