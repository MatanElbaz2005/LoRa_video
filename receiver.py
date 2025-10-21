import cv2
import numpy as np
import time
import os
import zlib
import zstandard as zstd
import socket
import struct

# Mode switch
FULL_MODE = True
FULL_BATCH_ENABLE = False
FULL_BATCH_COUNT = 3

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

    total_t = sum(times.values())
    ordered = [(k, times[k]) for k in sorted(times.keys())]
    breakdown = " ".join([f"{k}={v*1000:.1f}ms" for k, v in ordered])
    print(f"[DECODE breakdown] total={total_t*1000:.1f}ms {breakdown}")
    return canvas

def decode_frame_delta(prev_canvas: np.ndarray, compressed_delta: bytes, image_shape=(512, 512)):
    """
    Decode a delta (contour XOR) and apply it over prev_canvas.
    Returns:
        updated_canvas: prev_canvas XOR delta_canvas
        delta_canvas:   the drawn delta-only canvas (for debugging/vis)
    """
    # Timings
    times = {}

    # 1) Decode delta to a binary canvas
    start = time.time()
    delta_canvas = decode_frame(compressed_delta, image_shape=image_shape)
    times['decode_frame'] = time.time() - start

    # 2) Apply XOR with the previous reconstructed canvas
    start = time.time()
    if prev_canvas is None:
        # first frame delta == full frame
        updated = delta_canvas.copy()
    else:
        updated = cv2.bitwise_xor(prev_canvas, delta_canvas)
    times['xor_apply'] = time.time() - start

    # Report
    total_t = sum(times.values())
    ordered = [(k, times[k]) for k in sorted(times.keys())]
    breakdown = " ".join([f"{k}={v*1000:.1f}ms" for k, v in ordered])
    print(f"[DECODE-DELTA breakdown] total={total_t*1000:.1f}ms {breakdown}")

    return updated, delta_canvas

def decode_full_relative_payload(decompressed: bytes) -> list[np.ndarray]:
    """
    Decode a 'FULL' payload that contains only contours packed as:
      repeat:
        [u16 L][1B mode ('A'/'8'/'6')][payload per mode]
    Returns: list of int16 (L,2) arrays
    """
    p = 0
    contours = []
    n = len(decompressed)
    while p < n:
        if p + 2 > n: break
        (L,) = struct.unpack_from(">H", decompressed, p); p += 2
        if L == 0:
            contours.append(np.empty((0,2), dtype=np.int16))
            continue
        if p + 1 > n: break
        mode = decompressed[p:p+1]; p += 1  # b'A' / b'8' / b'6'

        if mode == b"A":
            need = 2*L*2
            if p + need > n: break
            pts = np.frombuffer(decompressed, dtype=np.int16, count=2*L, offset=p).reshape(L,2)
            p += need

        elif mode == b"8":
            # head (int16*2)
            if p + 4 > n: break
            head = np.frombuffer(decompressed, dtype=np.int16, count=2, offset=p).reshape(1,2)
            p += 4
            if L == 1:
                pts = head.astype(np.int16)
            else:
                need = (L-1)*2  # int8 pairs → 2 bytes per point
                if p + need > n: break
                deltas = np.frombuffer(decompressed, dtype=np.int8, count=2*(L-1), offset=p).reshape(L-1,2).astype(np.int16)
                p += need
                pts = np.empty((L,2), dtype=np.int16)
                pts[0] = head[0]
                pts[1:] = (head[0].astype(np.int32) + np.cumsum(deltas.astype(np.int32), axis=0)).astype(np.int16)

        elif mode == b"6":
            # head (int16*2)
            if p + 4 > n: break
            head = np.frombuffer(decompressed, dtype=np.int16, count=2, offset=p).reshape(1,2)
            p += 4
            if L == 1:
                pts = head.astype(np.int16)
            else:
                need = (L-1)*2*2  # int16 pairs → 4 bytes per point
                if p + need > n: break
                deltas = np.frombuffer(decompressed, dtype=np.int16, count=2*(L-1), offset=p).reshape(L-1,2)
                p += need
                pts = np.empty((L,2), dtype=np.int16)
                pts[0] = head[0]
                pts[1:] = (head[0].astype(np.int32) + np.cumsum(deltas.astype(np.int32), axis=0)).astype(np.int16)

        else:
            # fallback: treat as absolute
            need = 2*L*2
            if p + need > n: break
            pts = np.frombuffer(decompressed, dtype=np.int16, count=2*L, offset=p).reshape(L,2)
            p += need

        pts = np.clip(pts, [0,0], [511,511]).astype(np.int16)
        contours.append(np.ascontiguousarray(pts))

    return contours

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
        t_cycle = time.time()
        t0 = time.time()
        length_data = recv_exact(conn, 4)
        if not length_data: break
        msg_len = struct.unpack(">I", length_data)[0]
        t_header = time.time() - t0

        t1 = time.time()
        payload = recv_exact(conn, msg_len)
        t_payload = time.time() - t1
        if not payload: break

        # decompress payload
        decompressed = zstd.ZstdDecompressor().decompress(payload)

        if FULL_MODE:
            if FULL_BATCH_ENABLE:
                # BATCH ON: payload is an aggregate of N frames
                p = 0
                if p + 2 > len(decompressed):
                    continue
                (N,) = struct.unpack_from(">H", decompressed, p); p += 2

                last_canvas = None
                for _ in range(N):
                    if p + 4 > len(decompressed):
                        break
                    (L,) = struct.unpack_from(">I", decompressed, p); p += 4
                    if p + L > len(decompressed):
                        break
                    frame_raw = decompressed[p:p+L]; p += L

                    contours = decode_full_relative_payload(frame_raw)

                    canvas = np.zeros((512, 512), dtype=np.uint8)
                    for pts in contours:
                        if isinstance(pts, np.ndarray) and pts.ndim == 2 and pts.shape[0] >= 3 and pts.shape[1] == 2:
                            try:
                                cv2.polylines(canvas, [np.ascontiguousarray(pts.astype(np.int32)).reshape(-1,1,2)], True, 255, 1)
                            except Exception as e:
                                print(f"[DRAW FAIL] full pts shape={pts.shape} err={e}")
                    last_canvas = canvas

                if last_canvas is not None:
                    cv2.imshow("Reconstructed (FULL)", last_canvas)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            else:
                # BATCH OFF: single-frame FULL
                t2 = time.time()
                contours = decode_full_relative_payload(decompressed)
                canvas = np.zeros((512, 512), dtype=np.uint8)
                t_decode = time.time() - t2
                for pts in contours:
                    if isinstance(pts, np.ndarray) and pts.ndim == 2 and pts.shape[0] >= 3 and pts.shape[1] == 2:
                        try:
                            cv2.polylines(canvas, [np.ascontiguousarray(pts.astype(np.int32)).reshape(-1,1,2)], True, 255, 1)
                        except Exception as e:
                            print(f"[DRAW FAIL] full pts shape={pts.shape} err={e}")
                t3 = time.time()
                cv2.imshow("Reconstructed (FULL)", canvas)
                t_display = time.time() - t3
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                t_total = time.time() - t_cycle
                print(
                    "[RECEIVER timing FULL] "
                    f"header={t_header*1000:.1f}ms "
                    f"payload={t_payload*1000:.1f}ms "
                    f"decode+render={t_decode*1000:.1f}ms "
                    f"display={t_display*1000:.1f}ms "
                    f"total={t_total*1000:.1f}ms"
                )
                continue

        p = 0
        frame_type = decompressed[p:p+1]; p += 1
        (count,) = struct.unpack_from(">H", decompressed, p); p += 2

        if 'contours_by_id' not in locals():
            contours_by_id = {}

        for _ in range(count):
            tag = decompressed[p:p+1]; p += 1
            (cid,) = struct.unpack_from(">H", decompressed, p); p += 2

            if tag == b"N":
                (L,) = struct.unpack_from(">H", decompressed, p); p += 2
                mode = decompressed[p:p+1]; p += 1  # b'8' / b'6' / b'A'

                if L == 0:
                    contours_by_id[cid] = np.empty((0,2), dtype=np.int16)
                elif mode == b"A":
                    # Absolute int16 pairs (legacy)
                    pts = np.frombuffer(decompressed, dtype=np.int16, count=2*L, offset=p)
                    p += 2*L*2
                    pts = pts.reshape(-1,2).astype(np.int16)

                elif mode == b"8":
                    # First point absolute int16
                    head = np.frombuffer(decompressed, dtype=np.int16, count=2, offset=p).reshape(1,2)
                    p += 4
                    if L == 1:
                        pts = head.astype(np.int16)
                    else:
                        # (L-1) deltas as int8
                        deltas = np.frombuffer(decompressed, dtype=np.int8, count=2*(L-1), offset=p).reshape(L-1, 2).astype(np.int16)
                        p += (L-1)*2
                        pts = np.empty((L,2), dtype=np.int16)
                        pts[0] = head[0]
                        # cumulative sum of deltas
                        pts[1:] = (head[0].astype(np.int32) + np.cumsum(deltas.astype(np.int32), axis=0)).astype(np.int16)

                elif mode == b"6":
                    # First point absolute int16
                    head = np.frombuffer(decompressed, dtype=np.int16, count=2, offset=p).reshape(1,2)
                    p += 4
                    if L == 1:
                        pts = head.astype(np.int16)
                    else:
                        # (L-1) deltas as int16
                        deltas = np.frombuffer(decompressed, dtype=np.int16, count=2*(L-1), offset=p).reshape(L-1, 2)
                        p += (L-1)*2*2
                        pts = np.empty((L,2), dtype=np.int16)
                        pts[0] = head[0]
                        pts[1:] = (head[0].astype(np.int32) + np.cumsum(deltas.astype(np.int32), axis=0)).astype(np.int16)

                else:
                    # Unknown mode: skip safely as absolute fallback
                    pts = np.frombuffer(decompressed, dtype=np.int16, count=2*L, offset=p)
                    p += 2*L*2
                    pts = pts.reshape(-1,2).astype(np.int16)

                # clamp & store
                pts = np.clip(pts, [0,0], [511,511]).astype(np.int16)
                contours_by_id[cid] = np.ascontiguousarray(pts)


            elif tag == b"D":
                model = decompressed[p:p+1]; p += 1
                if model == b"T":
                    dx, dy = struct.unpack_from(">hh", decompressed, p); p += 4
                    if cid in contours_by_id:
                        poly = contours_by_id[cid].astype(np.int32)
                        poly[:, 0] += dx; poly[:, 1] += dy
                        poly = np.clip(poly, [0,0], [511,511]).astype(np.int16)
                        contours_by_id[cid] = np.ascontiguousarray(poly)

                elif model == b"V":
                    (L,) = struct.unpack_from(">H", decompressed, p); p += 2
                    deltas = np.frombuffer(decompressed, dtype=np.int8, count=L*2, offset=p).reshape(L, 2)
                    p += L * 2
                    if cid in contours_by_id:
                        poly = contours_by_id[cid]
                        if poly.shape[0] != L:
                            poly = poly[:L]
                        poly = poly.astype(np.int16) + deltas.astype(np.int16)
                        poly = np.clip(poly, [0,0], [511,511]).astype(np.int16)
                        contours_by_id[cid] = np.ascontiguousarray(poly)
                else:
                    # reserved for future affine model
                    pass

            elif tag == b"X":
                if cid in contours_by_id:
                    contours_by_id.pop(cid)

            if p > len(decompressed):
                print(f"[WARN] Packet overrun at cid={cid}, skipping remaining ops.")
                break

        print(f"[Frame] count={count} total contours={len(contours_by_id)}")

        # draw current frame from the map
        canvas = np.zeros((512, 512), dtype=np.uint8)
        for cid, pts in contours_by_id.items():
            if not isinstance(pts, np.ndarray):
                print(f"[WARN] cid={cid} not ndarray -> {type(pts)}")
                continue
            if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 3:
                print(f"[WARN] cid={cid} invalid shape={pts.shape}")
                continue
            if pts.size % 2 != 0:
                print(f"[WARN] cid={cid} odd number of coordinates ({pts.size})")
                continue
            try:
                pts32 = np.ascontiguousarray(pts.astype(np.int32))
                cv2.polylines(canvas, [pts32.reshape(-1,1,2)], True, 255, 1)
            except Exception as e:
                print(f"[DRAW FAIL] cid={cid} dtype={pts.dtype} shape={pts.shape} err={e}")

        prev_canvas = canvas.copy()
        cv2.imshow("Reconstructed (Object-Delta)", canvas)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

    conn.close()
    server.close()
    cv2.destroyAllWindows()