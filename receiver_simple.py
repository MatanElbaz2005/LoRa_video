import cv2
import numpy as np
import time
import socket
import struct
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

    current_frame_id = -1
    canvas = np.zeros((512, 512), dtype=np.uint8)
    contours_in_frame = 0
    
    t_last_frame_start = time.time()
    last_update_time = time.time()

    while True:
        t_cycle = time.time()
        t0 = time.time()
        data, addr = server.recvfrom(65535)

        # header: [frame_id:7bit][mode:1bit]
        if len(data) < 1:
            print("[Receiver-UDP] short datagram, skip")
            continue
        
        # Unpack the 1-byte header
        header_byte = data[0]
        frame_id = header_byte >> 2  # Get first 6 bits
        mode_bits = header_byte & 0x03 # Get last 2 bits
        
        # Map mode_bits to bit length (inverse of sender map)
        mode_len_map = {0: 4, 1: 6, 2: 8, 3: 10}
        if mode_bits not in mode_len_map:
             print(f"[Receiver-UDP] Received packet with invalid mode_bits={mode_bits}. Discarding.")
             continue
        
        mode_len = mode_len_map[mode_bits]
        payload = data[1:] # Payload starts after the 1-byte header

        # Handle first-ever packet
        if current_frame_id == -1:
            current_frame_id = frame_id

        # 6-bit sequence arithmetic logic with a 32-frame window
        diff = (frame_id - current_frame_id + 64) % 64

        if diff == 0:
            pass
        elif diff < 32:
            # Display the *previous* frame, which is now complete
            if current_frame_id >= 0:
                # Calculate time since last display
                t_now = time.time()
                t_delta = t_now - t_last_frame_start
                
                cv2.imshow("Reconstructed (FULL/UDP)", canvas)
                print(f"[RECV] Displaying frame {current_frame_id} with {contours_in_frame} contours.")
                
                # Print Receiver FPS
                if t_delta > 0:
                    print(f"[RECV-FPS] {1.0 / t_delta:.1f} FPS")
                
                if cv2.waitKey(1) & 0xFF == 27:
                    break # Exit the main loop

            # Reset for the new frame
            current_frame_id = frame_id
            canvas = np.zeros((512, 512), dtype=np.uint8)
            contours_in_frame = 0
            t_last_frame_start = time.time()
        
        else: # diff >= 32
            print(f"[Receiver-UDP] Stale/OOR packet frame={frame_id}, current={current_frame_id}. Discarding.")
            continue

        try:
            # Create BitStream from the received payload bytes
            s = BitStream(bytes=payload)
            
            # Read the head point (absolute, 16-bit signed x and y)
            try:
                head_x = s.read('int:16')
                head_y = s.read('int:16')
            except ReadError:
                raise ValueError("Payload too short for head point")

            head = np.array([[head_x, head_y]], dtype=np.int32)

            # Read all delta pairs using the determined bit length
            delta_format = f'int:{mode_len}'
            deltas_list = []
            while s.pos < s.len: # While there are still bits left to read
                try:
                    dx = s.read(delta_format)
                    dy = s.read(delta_format)
                    deltas_list.append([dx, dy])
                except ReadError:
                    break 
            
            # Reconstruct points
            if not deltas_list:
                # Only head point was sent/read
                pts = head.astype(np.int16)
            else:
                deltas = np.array(deltas_list, dtype=np.int32)
                pts = np.empty((1 + len(deltas), 2), dtype=np.int32)
                pts[0] = head[0]
                pts[1:] = head[0] + np.cumsum(deltas, axis=0)
                pts = pts.astype(np.int16)

            # Clipping remains the same
            pts = np.clip(pts, [0,0], [511,511]).astype(np.int32)
            cv2.polylines(canvas, [pts.reshape(-1,1,2)], True, 255, 1)

            contours_in_frame += 1

        except Exception as e:
            print(f"[Receiver-UDP] decode error: {e}")

    server.close()
    cv2.destroyAllWindows()