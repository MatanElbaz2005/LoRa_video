## Frame Processing Pipeline

### 1. **Preprocess**

```python
frame = cv2.resize(frame, (512, 512))
img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
```

* Resize the incoming frame to a uniform size (512x512).
* Convert it to grayscale to simplify processing (drop RGB channels).

---

### 2. **Noise Reduction**

```python
img_median = cv2.medianBlur(img_gray, 3)
```

* Apply a **median filter** to remove small isolated noise points.
* Keeps edges sharp while smoothing random noise.

---

### 3. **Contrast Enhancement**

```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img_clahe = clahe.apply(img_median)
```

* Use **CLAHE (Contrast Limited Adaptive Histogram Equalization)** to locally enhance contrast.
* Improves visibility of edges in dark or low-contrast regions.

---

### 4. **Edge Detection**

Three edge detection methods are used in parallel:

#### 🔹 Laplacian

```python
lap = cv2.Laplacian(img_clahe, cv2.CV_64F)
lap_abs = np.abs(lap)
_, lap_binary = cv2.threshold(lap_abs.astype(np.uint8), int(thresh), 255, cv2.THRESH_BINARY)
```

* Detects sharp changes in intensity (second derivative).
* Produces strong outlines of major structures.

#### 🔹 Canny

```python
edges_canny = auto_canny(cv2.GaussianBlur(img_clahe, (3, 3), 0))
```

* Classic double-threshold edge detector.
* Captures clean, well-defined contours of objects.

#### 🔹 Scharr

```python
gx = cv2.Scharr(img_clahe, cv2.CV_32F, 1, 0)
gy = cv2.Scharr(img_clahe, cv2.CV_32F, 0, 1)
scharr_mag = cv2.magnitude(gx, gy)
scharr_bin = (scharr_mag >= scharr_t).astype(np.uint8) * 255
```

* A refined Sobel operator for enhanced precision.
* Very sensitive to fine details such as textures, wrinkles, or facial features.

#### 🔹 Combine All

```python
img_binary = cv2.bitwise_or(cv2.bitwise_or(lap_binary, edges_canny), scharr_bin)
```

* Merge all edge maps using logical OR.
* Ensures that any edge detected by one method is preserved.

---

### 5. **Morphological Closing**

```python
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel, iterations=1)
```

* Fills small gaps and connects broken edges.
* Produces cleaner, closed shapes for contour detection.

---

### 6. **Contour Extraction**

```python
contours, _ = cv2.findContours(img_binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
```

* Extracts all connected edge components as vectorized contours.
* Each contour is a sequence of (x, y) coordinates representing a boundary.

---

### 7. **Simplification**

```python
simplified = [simplify_boundary(b, epsilon=2.0) for b in boundaries]
```

* Simplifies each contour using the **Ramer–Douglas–Peucker (RDP)** algorithm.
* Reduces the number of points while maintaining overall shape.
* Significantly decreases storage and processing cost.

---

### 8. **Compression**

```python
header = np.array([num_boundaries] + boundary_lengths, dtype=np.int32)
data_to_compress = header.tobytes() + all_points.tobytes()
compressed = zstd.ZstdCompressor(level=3).compress(data_to_compress)
```

* Concatenates all simplified contours into a single byte stream.
* Compresses it with **Zstandard (zstd)** for fast and efficient size reduction.
* Typically results in a file size only 2–5% of the original frame.
