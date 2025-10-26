import os, random
import zstandard as zstd

SAMPLES_DIR = os.path.join("zstd_samples", "full")
DICT_SIZE = 131072  # 128KB. Option: 65536 for 64KB

def load_samples(d):
    out = []
    if not os.path.isdir(d):
        print(f"Samples dir not found: {d}")
        return out
    for name in os.listdir(d):
        p = os.path.join(d, name)
        try:
            with open(p, "rb") as f:
                data = f.read()
            if 20 <= len(data) <= 256_000:
                out.append(data)
        except Exception:
            pass
    return out

if __name__ == "__main__":
    samples = load_samples(SAMPLES_DIR)
    random.shuffle(samples)
    if len(samples) < 100:
        print(f"Warning: only {len(samples)} samples. Collect more for better results.")
    if not samples:
        raise SystemExit("No samples found. Run sender to generate FULL samples first.")
    cdict = zstd.train_dictionary(DICT_SIZE, samples)
    out_path = f"contours_full_{DICT_SIZE//1024}k.zdict"
    with open(out_path, "wb") as f:
        f.write(cdict.as_bytes())
    print(f"Dictionary saved to {out_path}")
    print(f"Trained on {len(samples)} frames")