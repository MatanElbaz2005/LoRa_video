import cv2
import numpy as np
import time
import zstandard as zstd
import socket
import struct

ZSTD_DICT_PATH = "contours_full_128k.zdict"
USE_ZDICT = True

if USE_ZDICT:
    with open(ZSTD_DICT_PATH, "rb") as f:
        _DICT_BYTES = f.read()
    _ZDICT = zstd.ZstdCompressionDict(_DICT_BYTES)
    ZD = zstd.ZstdDecompressor(dict_data=_ZDICT)
else:
    ZD = zstd.ZstdDecompressor()

def recv_exact(sock, n):
    data = b""
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def decode_full_relative_payload(decompressed: bytes) -> list[np.ndarray]:
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
        mode = decompressed[p:p+1]; p += 1
        if mode == b"A":
            need = 2*L*2
            if p + need > n: break
            pts = np.frombuffer(decompressed, dtype=np.int16, count=2*L, offset=p).reshape(L,2)
            p += need
        elif mode == b"8":
            if p + 4 > n: break
            head = np.frombuffer(decompressed, dtype=np.int16, count=2, offset=p).reshape(1,2)
            p += 4
            if L == 1:
                pts = head.astype(np.int16)
            else:
                need = (L-1)*2
                if p + need > n: break
                deltas = np.frombuffer(decompressed, dtype=np.int8, count=2*(L-1), offset=p).reshape(L-1,2).astype(np.int16)
                p += need
                pts = np.empty((L,2), dtype=np.int16)
                pts[0] = head[0]
                pts[1:] = (head[0].astype(np.int32) + np.cumsum(deltas.astype(np.int32), axis=0)).astype(np.int16)
        elif mode == b"6":
            if p + 4 > n: break
            head = np.frombuffer(decompressed, dtype=np.int16, count=2, offset=p).reshape(1,2)
            p += 4
            if L == 1:
                pts = head.astype(np.int16)
            else:
                need = (L-1)*2*2
                if p + need > n: break
                deltas = np.frombuffer(decompressed, dtype=np.int16, count=2*(L-1), offset=p).reshape(L-1,2)
                p += need
                pts = np.empty((L,2), dtype=np.int16)
                pts[0] = head[0]
                pts[1:] = (head[0].astype(np.int32) + np.cumsum(deltas.astype(np.int32), axis=0)).astype(np.int16)
        else:
            need = 2*L*2
            if p + need > n: break
            pts = np.frombuffer(decompressed, dtype=np.int16, count=2*L, offset=p).reshape(L,2)
            p += need
        pts = np.clip(pts, [0,0], [511,511]).astype(np.int16)
        contours.append(np.ascontiguousarray(pts))
    return contours

if __name__ == "__main__":
    HOST = "127.0.0.1"
    PORT = 5001

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(1)
    print(f"[Receiver] Listening on {HOST}:{PORT} ...")
    conn, addr = server.accept()
    print(f"[Receiver] Connected by {addr}")

    while True:
        t_cycle = time.time()
        t0 = time.time()
        length_data = recv_exact(conn, 4)
        if not length_data: break
        msg_len = struct.unpack(">I", length_data)[0]
        t_header = time.time() - t0

        t1 = time.time()
        payload = recv_exact(conn, msg_len)
        if not payload: break
        t_payload = time.time() - t1

        t2 = time.time()
        decompressed = ZD.decompress(payload)
        contours = decode_full_relative_payload(decompressed)
        canvas = np.zeros((512, 512), dtype=np.uint8)
        for pts in contours:
            if isinstance(pts, np.ndarray) and pts.ndim == 2 and pts.shape[0] >= 3 and pts.shape[1] == 2:
                try:
                    cv2.polylines(canvas, [np.ascontiguousarray(pts.astype(np.int32)).reshape(-1,1,2)], True, 255, 1)
                except Exception as e:
                    print(f"[DRAW FAIL] full pts shape={pts.shape} err={e}")
        t_decode = time.time() - t2

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

    conn.close()
    server.close()
    cv2.destroyAllWindows()