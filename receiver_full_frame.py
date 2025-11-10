import cv2
import numpy as np
import time
import socket
import struct
import zstandard as zstd
from bitstring import BitStream, ReadError

def recv_exact(sock, n):
    data = b""
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

if __name__ == "__main__":
    HOST = "127.0.0.1"
    PORT = 5001

    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind((HOST, PORT))
    print(f"[Receiver-UDP] Listening on {HOST}:{PORT} ...")

    try:
        ZSTD_DICT_PATH = r"C:\Users\Matan\Documents\Matan\LoRa_video\contours_full_128k.zdict"
        with open(ZSTD_DICT_PATH, 'rb') as f:
            zstd_dict_data = f.read()
        zstd_dict = zstd.ZstdCompressionDict(zstd_dict_data)
        zstd_decompressor = zstd.ZstdDecompressor(dict_data=zstd_dict)
        print(f"[Receiver] Zstd dictionary loaded from {ZSTD_DICT_PATH}")
    except Exception as e:
        print(f"[Receiver] Warning: Failed to load Zstd dictionary. Using default decompression. Error: {e}")
        zstd_decompressor = zstd.ZstdDecompressor()

    current_frame_id = -1
    canvas = np.zeros((512, 512), dtype=np.uint8)
    mode_len_map = {0: 4, 1: 6, 2: 8, 3: 10}
    t_last_frame_start = time.time()

    while True:
        try:
            data, addr = server.recvfrom(65535)

            if len(data) < 3:
                print("[Receiver-UDP] Short packet, skipping.")
                continue

            expected_len = struct.unpack(">H", data[:2])[0]
            compressed_data = data[2:]

            if len(compressed_data) != expected_len:
                print(f"[Receiver-UDP] Mismatched packet length. Got {len(compressed_data)}, expected {expected_len}. Skipping.")
                continue
            
            uncompressed_data = zstd_decompressor.decompress(compressed_data)
            s = BitStream(bytes=uncompressed_data)

            frame_id = s.read('uint:8')
            
            if current_frame_id == -1:
                current_frame_id = frame_id
            
            diff = (frame_id - current_frame_id + 256) % 256
            
            if diff == 0 or diff > 128:
                print(f"[Receiver-UDP] Stale or duplicate frame {frame_id}. Current is {current_frame_id}. Skipping.")
                continue

            # This is a new, valid frame. Update ID.
            current_frame_id = frame_id
            
            # --- Reset canvas and stats for the new frame ---
            canvas = np.zeros((512, 512), dtype=np.uint8)
            contours_in_frame = 0
            t_delta = time.time() - t_last_frame_start
            t_last_frame_start = time.time()
            
            if t_delta > 0:
                print(f"[RECV-FPS] {1.0 / t_delta:.1f} FPS")

            num_contours = s.read('uint:8')

            # --- 1. Read ALL mini-headers first ---
            contour_headers = []
            try:
                for _ in range(num_contours):
                    is_closed_bit = s.read('uint:1')
                    mode_bits = s.read('uint:2')
                    payload_len_bits = s.read('uint:13')
                    contour_headers.append((is_closed_bit, mode_bits, payload_len_bits))
            except ReadError as e:
                print(f"[Receiver-UDP] Error reading headers: {e}. Skipping frame.")
                continue

            # --- 2. Now 's' is at the start of payloads. Read them. ---
            for is_closed_bit, mode_bits, payload_len_bits in contour_headers:
                try:
                    is_closed = (is_closed_bit == 1)
                    mode_len = mode_len_map[mode_bits]
                    delta_format = f'int:{mode_len}'

                    head_x = s.read('int:16')
                    head_y = s.read('int:16')
                    head = np.array([[head_x, head_y]], dtype=np.int32)
                    bits_read = 32
                    
                    deltas_list = []
                    while bits_read < payload_len_bits:
                        dx = s.read(delta_format)
                        dy = s.read(delta_format)
                        deltas_list.append([dx, dy])
                        bits_read += (mode_len * 2)
                    
                    if not deltas_list:
                        pts = head.astype(np.int16)
                    else:
                        deltas = np.array(deltas_list, dtype=np.int32)
                        pts = np.empty((1 + len(deltas), 2), dtype=np.int32)
                        pts[0] = head[0]
                        pts[1:] = head[0] + np.cumsum(deltas, axis=0)
                        pts = pts.astype(np.int16)

                    pts = np.clip(pts, [0,0], [511,511]).astype(np.int32)
                    cv2.polylines(canvas, [pts.reshape(-1,1,2)], is_closed, 255, 1)
                    contours_in_frame += 1
                
                except ReadError as e:
                    print(f"[Receiver-UDP] Error parsing payload: {e}. Skipping contour.")
                    # Try to recover by skipping the rest of this payload if possible
                    remaining_bits = payload_len_bits - bits_read
                    if remaining_bits > 0:
                        try:
                            s.read(f'bits:{remaining_bits}')
                        except ReadError:
                            print("[Receiver-UDP] Failed to skip bad contour data. Frame likely corrupt.")
                            break # Break from inner contour loop
                    continue # Continue to next contour
            print(f"[RECV] Displaying frame {current_frame_id} with {contours_in_frame} contours.")
            cv2.imshow("Reconstructed (FULL/UDP)", canvas)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        except Exception as e:
            print(f"[Receiver-UDP] decode error: {e}")

    server.close()
    cv2.destroyAllWindows()