import cv2
import numpy as np

def _ensure_uint8(img):
    if img.dtype == np.float32 or img.dtype == np.float64:
        # normalize float to 0..255 for display
        imin, imax = float(np.min(img)), float(np.max(img))
        if imax > imin:
            img = (img - imin) / (imax - imin) * 255.0
        else:
            img = np.zeros_like(img, dtype=np.float32)
        img = np.clip(img, 0, 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def _to_bgr(img):
    img = _ensure_uint8(img)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def _fit_to_tile(img, tile_size):
    h, w = tile_size
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

def _draw_title(tile_bgr, title):
    # put a semi-transparent bar and text at the top
    overlay = tile_bgr.copy()
    bar_h = max(18, tile_bgr.shape[0] // 14)
    cv2.rectangle(overlay, (0, 0), (tile_bgr.shape[1], bar_h), (0, 0, 0), -1)
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, tile_bgr, 1 - alpha, 0, tile_bgr)
    cv2.putText(tile_bgr, title, (8, bar_h - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return tile_bgr

def build_dashboard(images, titles, cols=3, tile_size=(256, 256), pad=8, bg_color=(24, 24, 24)):
    """
    images: list of np.ndarray (each gray or BGR)
    titles: list of str with same length
    returns: single BGR image composing all tiles in a grid
    """
    assert len(images) == len(titles), "images and titles length mismatch"
    tiles = []
    for img, title in zip(images, titles):
        tile = _to_bgr(img)
        tile = _fit_to_tile(tile, tile_size)
        tile = _draw_title(tile, title)
        tiles.append(tile)

    rows = (len(tiles) + cols - 1) // cols
    th, tw = tile_size
    H = rows * th + (rows + 1) * pad
    W = cols * tw + (cols + 1) * pad
    canvas = np.full((H, W, 3), bg_color, dtype=np.uint8)

    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= len(tiles):
                break
            y0 = pad + r * (th + pad)
            x0 = pad + c * (tw + pad)
            canvas[y0:y0 + th, x0:x0 + tw] = tiles[idx]
            idx += 1
    return canvas
