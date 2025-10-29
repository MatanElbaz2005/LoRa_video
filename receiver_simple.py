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

        # header: [frame_id:U32][mode:U8]
        if len(data) < 5:
            print("[Receiver-UDP] short datagram, skip")
            continue
        frame_id = struct.unpack_from(">I", data, 0)[0]
        mode_u8  = data[4]
        payload  = data[5:]

        if frame_id > current_frame_id:
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
            t_last_frame_start = time.time() # Reset FPS timer

        elif frame_id < current_frame_id:
            print(f"[Receiver-UDP] Stale packet from frame {frame_id}, current is {current_frame_id}. Discarding.")
            continue

        try:
            if mode_u8 == ord('A'):
                if len(payload) % 4 != 0:
                    raise ValueError("A-mode payload not multiple of 4")
                L = len(payload) // 4
                pts = np.frombuffer(payload, dtype=np.int16, count=2*L).reshape(L, 2)

            elif mode_u8 == ord('8'):
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