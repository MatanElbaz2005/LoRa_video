import cv2
import numpy as np
import time
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
        
        # Map mode_bits back to mode_u8 for existing logic
        if mode_bits == 0:
             mode_u8 = ord('8')
        elif mode_bits == 1:
             mode_u8 = ord('6')
        else:
            print(f"[Receiver-UDP] Received packet with unused mode_bits={mode_bits}. Discarding.")
            continue 

        payload = data[1:]

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
            if mode_u8 == ord('8'):
                if len(payload) < 4 or (len(payload)-4) % 2 != 0:
                    raise ValueError("8-mode payload bad size")
                head = np.frombuffer(payload, dtype=np.int16, count=2, offset=0).reshape(1,2).astype(np.int32)
                k = (len(payload)-4)//2
                deltas = np.frombuffer(payload, dtype=np.int8,  count=2*k, offset=4).reshape(k,2).astype(np.int32)
                pts = np.empty((1+k, 2), dtype=np.int32)
                pts[0] = head[0]
                if k:
                    pts[1:] = head[0] + np.cumsum(deltas, axis=0)
                pts = pts.astype(np.int16)

            elif mode_u8 == ord('6'):
                if len(payload) < 4 or (len(payload)-4) % 4 != 0:
                    raise ValueError("6-mode payload bad size")
                head = np.frombuffer(payload, dtype=np.int16, count=2, offset=0).reshape(1,2).astype(np.int32)
                k = (len(payload)-4)//4
                deltas = np.frombuffer(payload, dtype=np.int16, count=2*k, offset=4).reshape(k,2).astype(np.int32)
                pts = np.empty((1+k, 2), dtype=np.int32)
                pts[0] = head[0]
                if k:
                    pts[1:] = head[0] + np.cumsum(deltas, axis=0)
                pts = pts.astype(np.int16)

            else:
                raise ValueError(f"unknown mode {mode_u8!r}")

            pts = np.clip(pts, [0,0], [511,511]).astype(np.int32)
            cv2.polylines(canvas, [pts.reshape(-1,1,2)], True, 255, 1)

            contours_in_frame += 1
            cv2.putText(canvas, f"F {current_frame_id}", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"Contours {contours_in_frame}", (8, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1, cv2.LINE_AA)

        except Exception as e:
            print(f"[Receiver-UDP] decode error: {e}")

    server.close()
    cv2.destroyAllWindows()