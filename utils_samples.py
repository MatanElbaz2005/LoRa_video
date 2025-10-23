import os, time, hashlib

def save_sample(kind: str, data: bytes, root: str = "zstd_samples"):
    os.makedirs(os.path.join(root, kind), exist_ok=True)
    h = hashlib.blake2b(data, digest_size=16).hexdigest()
    path = os.path.join(root, kind, f"{int(time.time()*1000)}_{h}.bin")
    with open(path, "wb") as f:
        f.write(data)