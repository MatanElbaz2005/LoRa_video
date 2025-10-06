# Build OpenCV with GStreamer on Linux

This guide shows how to compile OpenCV with **GStreamer** support (and opencv_contrib) on Debian/Ubuntu/Raspberry Pi OS–based systems. It follows a clean, repeatable flow and includes verification and example pipelines for Python.

---

## 1) Install system dependencies

```bash
sudo apt update
sudo apt install -y build-essential cmake git pkg-config libgtk-3-dev \
libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
gfortran openexr libatlas-base-dev python3-dev python3-numpy \
libtbb-dev libdc1394-dev libopenexr-dev \
libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev \
gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-x \
gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 \
gstreamer1.0-qt5 gstreamer1.0-pulseaudio
```

> Tip: If you plan to use a Python virtual environment, create/activate it **before** building (so `cv2` uses the right interpreter later).

---

## 2) Fetch OpenCV sources (core + contrib)

```bash
cd ~
git clone --depth=1 -b 4.10.0 https://github.com/opencv/opencv.git
git clone --depth=1 -b 4.10.0 https://github.com/opencv/opencv_contrib.git
mkdir -p ~/opencv/build && cd ~/opencv/build
```

---

## 3) Configure CMake (enable GStreamer)

```bash
cmake -D CMAKE_BUILD_TYPE=RELEASE \
 -D CMAKE_INSTALL_PREFIX=/usr/local \
 -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
 -D ENABLE_NEON=ON \
 -D WITH_OPENMP=ON \
 -D WITH_OPENCL=OFF \
 -D BUILD_TIFF=ON \
 -D WITH_FFMPEG=ON \
 -D WITH_TBB=ON \
 -D BUILD_TBB=ON \
 -D WITH_GSTREAMER=ON \
 -D BUILD_TESTS=OFF \
 -D WITH_EIGEN=OFF \
 -D WITH_V4L=ON \
 -D WITH_LIBV4L=ON \
 -D WITH_VTK=OFF \
 -D WITH_QT=OFF \
 -D WITH_PROTOBUF=ON \
 -D OPENCV_ENABLE_NONFREE=ON \
 -D INSTALL_C_EXAMPLES=OFF \
 -D INSTALL_PYTHON_EXAMPLES=ON \
 -D PYTHON3_PACKAGES_PATH=/usr/lib/python3/dist-packages \
 -D OPENCV_GENERATE_PKGCONFIG=ON \
 -D BUILD_EXAMPLES=ON ..
```

**Notes on key flags**

* `WITH_GSTREAMER=ON` — builds OpenCV’s Video I/O against GStreamer.
* `OPENCV_EXTRA_MODULES_PATH` — adds contrib modules (e.g., `xfeatures2d`).
* `PYTHON3_PACKAGES_PATH` — where the built Python `cv2` will be installed. Adjust if using a venv.
* `OPENCV_GENERATE_PKGCONFIG=ON` — installs `opencv4.pc` for `pkg-config`.

---

## 4) Build

```bash
make -j4
```

Change `-j4` according to your CPU cores (e.g., `-j$(nproc)`).

---

## 5) Install

```bash
sudo make install
sudo ldconfig
```

This places libraries in `/usr/local` and refreshes the loader cache.

---

## 6) Verify the build (Python)

```python
import cv2
print(cv2.__version__)
print(cv2.getBuildInformation())
```

Check that the build info contains lines like:

* `GStreamer:                      YES`
* `Video I/O: ... GStreamer ...`

If it says `NO`, verify you installed `libgstreamer1.0-dev` and `libgstreamer-plugins-base1.0-dev` **before** running CMake, then re-run CMake from a clean `build/` directory.

---

## 7) Python/Numpy compatibility note

Some platform builds of OpenCV 4.10.0 are incompatible with NumPy ≥ 2.0. If you hit import errors or ABI mismatches, pin NumPy below 2:

```bash
pip uninstall -y numpy
pip install "numpy<2.0"
```

Prefer doing this inside a virtual environment.

---

## Quick GStreamer sanity checks (outside OpenCV)

Test your camera/decoder stack directly with GStreamer first:

```bash
# List GStreamer version and plugins
gst-inspect-1.0 --version

# Simple v4l2 webcam preview (X11/Wayland display)
gst-launch-1.0 v4l2src device=/dev/video0 ! videoconvert ! autovideosink

# H.264 file playback (if gstreamer1.0-libav installed)
gst-launch-1.0 filesrc location=sample.mp4 ! qtdemux ! h264parse ! avdec_h264 ! videoconvert ! autovideosink
```

If these work, OpenCV’s GStreamer backend is far more likely to behave.


## What you should see when things work

* `cv2.getBuildInformation()` lists `GStreamer: YES`.
* `gst-launch-1.0` test pipelines run without errors.
* Your Python examples with `CAP_GSTREAMER` open and read frames successfully.

Happy streaming! 🎥
