import cv2
import numpy as np
from dataclasses import dataclass

def _to_bgr(tile: np.ndarray) -> np.ndarray:
    if tile is None:
        return np.zeros((1,1,3), dtype=np.uint8)
    if tile.ndim == 2:  # gray -> BGR
        return cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
    if tile.ndim == 3 and tile.shape[2] == 3:
        return tile
    if tile.ndim == 3 and tile.shape[2] == 4:
        return cv2.cvtColor(tile, cv2.COLOR_BGRA2BGR)
    return np.dstack([tile]*3).astype(np.uint8)

def _resize_keep_aspect(img: np.ndarray, target_wh: tuple[int,int]) -> np.ndarray:
    tw, th = target_wh
    h, w = img.shape[:2]
    scale = min(tw / max(w,1), th / max(h,1))
    nw, nh = max(1, int(round(w*scale))), max(1, int(round(h*scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
    # pad to exact target
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    y0 = (th - nh) // 2
    x0 = (tw - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas

def _rounded_rect_mask(size: tuple[int,int], radius: int) -> np.ndarray:
    h, w = size
    if radius <= 0:
        return np.ones((h, w, 1), dtype=np.uint8)*255
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (radius, 0), (w - radius - 1, h - 1), 255, -1)
    cv2.rectangle(mask, (0, radius), (w - 1, h - radius - 1), 255, -1)
    for (cx, cy) in [(radius, radius), (w - radius - 1, radius), (radius, h - radius - 1), (w - radius - 1, h - radius - 1)]:
        cv2.ellipse(mask, (cx, cy), (radius, radius), 0, 0, 360, 255, -1)
    return mask[..., None]

def _alpha_blend(dst: np.ndarray, src: np.ndarray, alpha: float, y: int, x: int):
    h, w = src.shape[:2]
    roi = dst[y:y+h, x:x+w].astype(np.float32)
    dst[y:y+h, x:x+w] = (alpha*src.astype(np.float32) + (1.0-alpha)*roi).clip(0,255).astype(np.uint8)

def build_dashboard(
    tiles: list[np.ndarray],
    titles: list[str],
    cols: int = 3,
    tile_size: tuple[int,int] = (300, 220),  # (w, h)
    pad: int = 16,
    bg_color: tuple[int,int,int] = (24, 24, 28),
    title_bar_h: int = 28,
    title_fg: tuple[int,int,int] = (235, 235, 240),
    title_bg: tuple[int,int,int] = (40, 40, 48),
    card_radius: int = 10,
    shadow: int = 3
) -> np.ndarray:
    assert len(tiles) == len(titles), "tiles and titles must have same length"
    n = len(tiles)
    cols = max(1, int(cols))
    rows = (n + cols - 1) // cols

    tw, th = tile_size
    card_w, card_h = tw, th + title_bar_h
    grid_w = cols * card_w + (cols + 1) * pad
    grid_h = rows * card_h + (rows + 1) * pad

    dashboard = np.full((grid_h, grid_w, 3), bg_color, dtype=np.uint8)

    # pre-make card mask (rounded corners)
    card_mask = _rounded_rect_mask((card_h, card_w), card_radius)

    for idx, (tile, title) in enumerate(zip(tiles, titles)):
        r = idx // cols
        c = idx % cols
        y = pad + r * (card_h + pad)
        x = pad + c * (card_w + pad)

        # shadow
        if shadow > 0:
            sh = np.zeros((card_h, card_w, 3), dtype=np.uint8)
            sh[:] = (0, 0, 0)
            y_s, x_s = y + shadow, x + shadow
            _alpha_blend(dashboard, sh, 0.20, y_s, x_s)

        # card base
        card = np.full((card_h, card_w, 3), (32, 32, 38), dtype=np.uint8)

        # title bar (semi-transparent blend over base)
        bar = np.full((title_bar_h, card_w, 3), title_bg, dtype=np.uint8)
        card[:title_bar_h] = cv2.addWeighted(bar, 0.85, card[:title_bar_h], 0.15, 0)

        # draw title centered-left with padding
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(title, font, scale, thickness)
        tx = 10
        ty = (title_bar_h + text_h) // 2
        cv2.putText(card, title, (tx, ty), font, scale, title_fg, thickness, cv2.LINE_AA)

        # tile image normalized to BGR & resized with aspect
        tile_bgr = _to_bgr(tile)
        tile_resized = _resize_keep_aspect(tile_bgr, (tw, th))
        card[title_bar_h:title_bar_h+th, 0:tw] = tile_resized

        # apply rounded corners via mask onto dashboard
        roi = dashboard[y:y+card_h, x:x+card_w]
        np.copyto(roi, card, where=(card_mask==255))

    return dashboard

@dataclass
class ControlState:
    canny_l2: bool = True
    canny_ap_idx: int = 0
    median_idx: int = 0
    contour_mode_idx: int = 1
    # 0:SIMPLE,1:NONE,2:TC89_L1,3:TC89_KCOS
    contour_method_idx: int = 0

    # sliders (1..100 -> 0.01..1.00), (50..99)
    canny_sigma_x100: int = 33
    lap_percentile: int = 95

    # --- NEW: Scharr controls ---
    scharr_percentile: int = 90          # 70..99 recommended
    scharr_preblur_x10: int = 3          # maps to 0.0..1.5 (0..15)
    scharr_unsharp_x10: int = 10         # maps to 0.0..1.5 (0..15), default 1.0
    scharr_mag_l2: bool = True           # True="l2", False="l1"

    # dragging state
    dragging: bool = False
    drag_target: str | None = None

# geometry of control panel
P_W, P_H = 640, 540
ROW_H = 52
PAD = 16
FONT = cv2.FONT_HERSHEY_SIMPLEX
FG = (235,235,240)
BG = (28,28,32)
LINE = (70,70,80)
ACCENT = (120,170,255)
BTN_BG = (50,50,58)
BTN_ON = (70,150,70)
BTN_OFF = (120,120,120)

# clickable regions registry
_HITBOXES = []  # list of tuples: (name, (x0,y0,x1,y1))

def _label(img, text, p, scale=0.5, color=FG, thickness=1):
    cv2.putText(img, text, p, FONT, scale, color, thickness, cv2.LINE_AA)

def _button(img, x, y, w, h, text, active=False, name=None):
    color = BTN_ON if active else BTN_BG
    cv2.rectangle(img, (x,y), (x+w,y+h), color, -1, lineType=cv2.LINE_AA)
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,0), 1, lineType=cv2.LINE_AA)
    (tw,th), _ = cv2.getTextSize(text, FONT, 0.6, 1)
    cv2.putText(img, text, (x + (w-tw)//2, y + (h+th)//2 - 2), FONT, 0.6, FG, 1, cv2.LINE_AA)
    if name:
        _HITBOXES.append((name, (x,y,x+w,y+h)))

def _stepper(img, x, y, w, h, label, val_str, name_minus, name_plus):
    _label(img, label, (x, y-8), scale=0.6)
    # minus (ASCII '-')
    _button(img, x, y, 36, h, "-", name=name_minus)
    # value area (wider)
    cv2.rectangle(img, (x+40, y), (x+w-40, y+h), BTN_BG, -1, lineType=cv2.LINE_AA)
    (tw,th), _ = cv2.getTextSize(val_str, FONT, 0.7, 1)
    cv2.putText(img, val_str, (x + (w-tw)//2, y + (h+th)//2 - 2), FONT, 0.7, FG, 1, cv2.LINE_AA)
    # plus
    _button(img, x+w-36, y, 36, h, "+", name=name_plus)

def render_controls(state: ControlState) -> np.ndarray:
    """Draw pretty control panel and register click hitboxes."""
    global _HITBOXES
    _HITBOXES = []
    panel = np.full((P_H, P_W, 3), BG, dtype=np.uint8)

    # section title
    _label(panel, "Controls", (PAD, PAD+10), scale=0.9, color=ACCENT, thickness=2)
    cv2.line(panel, (PAD, PAD+18), (P_W-PAD, PAD+18), LINE, 1, cv2.LINE_AA)

    y = PAD + 64

    _label(panel, "Canny:", (PAD, y))
    # L2 toggle bigger
    _button(panel, PAD+90, y-22, 86, 28, "L2 ON" if state.canny_l2 else "L2 OFF",
            active=state.canny_l2, name="canny_l2_toggle")
    # Aperture 3/5/7 with more spacing
    ap_vals = [3,5,7]
    x0 = PAD+190
    for i,av in enumerate(ap_vals):
        _button(panel, x0 + i*58, y-22, 52, 28, str(av),
                active=(state.canny_ap_idx==i), name=f"ap_{i}")

    y += ROW_H
    # Sigma slider (0.01..1.00 mapped from 1..100)
    _sigma_val = state.canny_sigma_x100 / 100.0
    _slider(panel, "sigma", PAD+90, y-22, 320, 28, "Sigma",
            _sigma_val, 0.01, 1.00, "{:.2f}")

    y += ROW_H
    # Laplacian Percentile slider (50..99)
    _slider(panel, "lap", PAD+90, y-22, 320, 28, "Laplacian Percentile",
            float(state.lap_percentile), 50.0, 99.0, "{:.0f}")

    y += ROW_H
    _label(panel, "Median ksize:", (PAD, y))
    for i,kv in enumerate([3,5,7]):
        _button(panel, PAD+160 + i*58, y-22, 52, 28, str(kv),
                active=(state.median_idx==i), name=f"med_{i}")

    y += ROW_H
    _label(panel, "Contours mode:", (PAD, y))
    modes = [("EXT",0), ("CCOMP",1), ("TREE",2), ("LIST",3)]
    for i,(label,idx) in enumerate(modes):
        _button(panel, PAD+160 + i*66, y-22, 62, 28, label,
                active=(state.contour_mode_idx==idx), name=f"mode_{idx}")

    y += ROW_H
    _label(panel, "Contours method:", (PAD, y))
    methods = [("SIMPLE",0), ("NONE",1)]
    for i,(label,idx) in enumerate(methods):
        _button(panel, PAD+160 + i*92, y-22, 90, 28, label,
                active=(state.contour_method_idx==idx), name=f"meth_{idx}")

    y += ROW_H
    _label(panel, "Scharr:", (PAD, y))
    # Magnitude toggle (L2/L1)
    _button(panel, PAD+90, y-22, 86, 28,
            "Mag L2" if state.scharr_mag_l2 else "Mag L1",
            active=state.scharr_mag_l2, name="scharr_mag_toggle")

    y += ROW_H
    # Percentile (70..99)
    _slider(panel, "scharr_pct", PAD+90, y-22, 320, 28, "Percentile",
            float(state.scharr_percentile), 70.0, 99.0, "{:.0f}")

    y += ROW_H
    # Pre-blur sigma (0.0..1.5) mapped from 0..15
    _slider(panel, "scharr_pre", PAD+90, y-22, 320, 28, "Preblur",
            state.scharr_preblur_x10/10.0, 0.0, 1.5, "{:.1f}")

    y += ROW_H
    # Unsharp amount (0.0..1.5) mapped from 0..15
    _slider(panel, "scharr_unsharp", PAD+90, y-22, 320, 28, "Unsharp amt",
            state.scharr_unsharp_x10/10.0, 0.0, 1.5, "{:.1f}")

    return panel

def handle_controls_click(event, x, y, flags, state: ControlState):
    def _hit(name):
        for n,(x0,y0,x1,y1) in _HITBOXES:
            if n == name and x0 <= x <= x1 and y0 <= y <= y1:
                return True
        return False

    if event == cv2.EVENT_LBUTTONDOWN:
        # sliders
        for n,(x0,y0,x1,y1) in _HITBOXES:
            if n in ("sigma_track","lap_track","scharr_pct_track","scharr_pre_track","scharr_unsharp_track") \
               and x0 <= x <= x1 and y0 <= y <= y1:
                state.dragging = True
                state.drag_target = n
                break
        # buttons
        if not state.dragging:
            for name,(x0,y0,x1,y1) in _HITBOXES:
                if x0 <= x <= x1 and y0 <= y <= y1:
                    if name == "canny_l2_toggle":
                        state.canny_l2 = not state.canny_l2
                    elif name.startswith("ap_"):
                        state.canny_ap_idx = int(name.split("_")[1])
                    elif name.startswith("med_"):
                        state.median_idx = int(name.split("_")[1])
                    elif name.startswith("mode_"):
                        state.contour_mode_idx = int(name.split("_")[1])
                    elif name.startswith("meth_"):
                        state.contour_method_idx = int(name.split("_")[1])
                    elif name == "scharr_mag_toggle":                      # NEW
                        state.scharr_mag_l2 = not state.scharr_mag_l2
                    break

    elif event == cv2.EVENT_MOUSEMOVE and state.dragging and state.drag_target:
        # find track rect
        for n,(x0,y0,x1,y1) in _HITBOXES:
            if n == state.drag_target:
                t = np.clip((x - x0) / max(1, (x1 - x0)), 0.0, 1.0)
                if n == "sigma_track":
                    state.canny_sigma_x100 = max(1, min(100, int(round(1 + t * 99))))
                elif n == "lap_track":
                    state.lap_percentile = max(50, min(99, int(round(50 + t * (99-50)))))
                elif n == "scharr_pct_track":                               # NEW (70..99)
                    state.scharr_percentile = max(70, min(99, int(round(70 + t * (99-70)))))
                elif n == "scharr_pre_track":                               # NEW (0..15 -> 0.0..1.5)
                    state.scharr_preblur_x10 = max(0, min(15, int(round(0 + t * 15))))
                elif n == "scharr_unsharp_track":                           # NEW (0..15 -> 0.0..1.5)
                    state.scharr_unsharp_x10 = max(0, min(15, int(round(0 + t * 15))))
                break

    # end drag
    elif event == cv2.EVENT_LBUTTONUP:
        state.dragging = False
        state.drag_target = None


def controls_to_params(state: ControlState) -> dict:
    ap_map = {0:3, 1:5, 2:7}
    median_map = {0:3, 1:5, 2:7}
    mode_map = {0: cv2.RETR_EXTERNAL, 1: cv2.RETR_CCOMP, 2: cv2.RETR_TREE, 3: cv2.RETR_LIST}
    method_map = {
        0: cv2.CHAIN_APPROX_SIMPLE,
        1: cv2.CHAIN_APPROX_NONE
    }
    return {
        "canny_sigma": max(0.01, min(1.0, state.canny_sigma_x100 / 100.0)),
        "canny_aperture": {0:3,1:5,2:7}.get(state.canny_ap_idx, 3),
        "canny_l2": bool(state.canny_l2),
        "lap_percentile": int(state.lap_percentile),
        "median_ksize": int(median_map.get(state.median_idx, 3)),
        "contour_mode": mode_map.get(state.contour_mode_idx, cv2.RETR_CCOMP),
        "contour_method": method_map.get(state.contour_method_idx, cv2.CHAIN_APPROX_SIMPLE),

        "scharr_percentile": int(state.scharr_percentile),
        "scharr_preblur_sigma": state.scharr_preblur_x10 / 10.0,
        "scharr_unsharp_amount": state.scharr_unsharp_x10 / 10.0,
        "scharr_magnitude": "l2" if state.scharr_mag_l2 else "l1",
    }



def _slider(img, name, x, y, w, h, label, val, vmin, vmax, fmt):
    """Draw a horizontal slider and register its hitbox."""
    _label(img, label, (x, y-10), scale=0.6)
    # track
    track_y = y + h//2 - 2
    cv2.rectangle(img, (x, track_y-3), (x+w, track_y+3), (70,70,80), -1, cv2.LINE_AA)
    # value -> knob x
    t = (val - vmin) / max(1e-6, (vmax - vmin))
    knob_x = int(round(x + t * w))
    # knob
    cv2.circle(img, (knob_x, track_y), 9, (120,170,255), -1, cv2.LINE_AA)
    cv2.circle(img, (knob_x, track_y), 9, (0,0,0), 1, cv2.LINE_AA)
    # value box
    vs = fmt.format(val)
    (tw,th), _ = cv2.getTextSize(vs, FONT, 0.7, 1)
    box_w = max(64, tw + 14)
    bx = x + w + 10
    by = y
    cv2.rectangle(img, (bx, by), (bx+box_w, by+h), BTN_BG, -1, cv2.LINE_AA)
    cv2.rectangle(img, (bx, by), (bx+box_w, by+h), (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(img, vs, (bx + (box_w-tw)//2, by + (h+th)//2 - 2), FONT, 0.7, FG, 1, cv2.LINE_AA)

    # register hitbox for dragging
    _HITBOXES.append((f"{name}_track", (x, y, x+w, y+h)))
