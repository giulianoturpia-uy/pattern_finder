# pattern_finder

This repository offers a basic pattern detection in **.tif** images, where the pattern is formated in **.bmp**. The functions are meant to be run inside a constrained device with very few hardware resources (for instance Cortex ARMv7 devices).

It includes a basic API for uploa is coded using python, mainly targeting Linux-based environments.
The target devices are resource-constrained(4GB RAM at most) and are relatively old. For this reason, the Python version we target is relatively "outdated" given the versions available at the time of writing this.
If the device supports newer versions of Python, please use a virtual environment targeting your desired version (for instance, Ubuntu 22.04.5 LTS utilizes Python 3.10.12).

We assume the images are mostly mono-colored, as well as the pattern to be found in the image.

## 1. Installation

### 1.0. Prerequisites (system packages)

On Debian/Ubuntu, `venv` and `pip` ship as separate apt packages and may be
absent on a minimal device image. Install them before anything else:

```bash
sudo apt install python3-venv python3-pip
```

If you target a specific interpreter, install its matching `-venv` package
(e.g. `python3.8-venv` for Python 3.8, `python3.10-venv` for Python 3.10).

### 1.1. Create and activate a virtual environment

Go to the base directory of this repository (where `src` is located), and run the following:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 1.2. Install the package

```bash
pip install -e .
```

With dev dependencies (pytest, coverage):

```bash
pip install -e ".[dev]"
```

### 1.3. A note on OpenCV

OpenCV is published on PyPI only as monolithic wheels — every wheel contains
all modules; pip cannot install individual ones (e.g. just `imgproc`). Of the
available wheels, `opencv-python-headless` is the lightest: it strips the GUI
(`highgui`) layer while still providing the only functions this library uses —
`Canny`, `findContours` and `approxPolyDP`, all in the `imgproc` module. That
is why it is the dependency pinned in `setup.py`.

## 2. Code Flow

The public API is the `PatternFinder` class (the *context* of a Strategy
pattern). It receives the two files, loads them, runs the chosen detection
strategy, and returns a `DetectionResult`.

```python
from pattern_finder import PatternFinder

finder = PatternFinder("opencv")                 # pick a strategy by name
result = finder.find("patron.bmp", "imagen.tif") # pass the files as-is
if result.found:
    print(result.vertices)   # ((apex_x, apex_y), (start_x, start_y), (end_x, end_y))
```

`DetectionResult` carries:

- `found` — the success code (`True`/`False`),
- `vertices` — the 3 extremes (apex, arc_start, arc_end) when found,
- `corner_count` — `len(vertices)`.

After each call, reports the time spent decoding the files versus running detection.
`main.py` is a thin command-line demo over this API. For instance, if we use
the folder labeled `images` as a source of our files to detect, we would check the behaviour in the following way:

```bash
python main.py PATTERN IMAGE [STRATEGY]      # STRATEGY: opencv | geometric
python main.py images/patron.bmp images/imagen.tif opencv
```

It prints the status, the 3 coordinates, and the load/compute timing.

## 3. Design

The target is a **quarter circle** (a 90° circular sector): two straight radii
meeting at an **apex**, joined by a 90° **arc**. Its 3 extremes are the apex
plus the two **arc endpoints**. The target may appear at any rotation on a
mostly-black background. Two interchangeable strategies are provided.

### 3.1. OpenCV strategy (`opencv`)

Uses OpenCV's edge-detection and contour tools. The search image can be huge while the target is small, so a single full-resolution pass would exceed a tight budget. Detection therefore runs in **two stages**:

1. **Locate.** Downscale the image so its longest side is ~`work_dim` px
   (default 1000). Run Canny edge detection, dilate to close 1-px gaps, then
   `findContours`; the largest contour is the target. Take its bounding box.
2. **Refine.** Map that box back to full resolution, add a margin, and crop.
   Re-run the same edge/contour detection on the small crop. Because this stage
   works at full resolution on a tiny image, the reported coordinates are
   pixel-accurate **and** fast.

**Extracting the 3 extremes** (rotation-invariant by construction):

- Approximate the contour with `approxPolyDP`.
- The **two longest edges** of that polygon are the straight radii of the
  sector.
- Their **shared vertex is the apex**; the other two ends are the
  **arc endpoints**.
- The arc endpoints are ordered deterministically by their angle about the apex.

Detecting only on the downscaled image and scaling the coordinates back up *works*, but the precision is changed: shrinking blurs the corners, so the apex drifts a few pixels into the background, and the error grows as the downscale gets more aggressive.

Rather than template/shape matching, the detected triangle is validated geometrically: the two radii must be roughly equal in length and the apex angle must be near 90°.

### 3.2. How the OpenCV parameters were chosen

The defaults (in `config.json`) were derived empirically against the sample
`imagen.tif`, not guessed:

- **`work_dim` (1000).** A full-resolution pass measured ~113 ms — over the
  100 ms budget — which is what motivated downscaling. Testing `work_dim` of
  700 / 1000 / 1500 traded locate cost against accuracy; 1000 gives a fast,
  reliable locate, and the refine stage restores full precision regardless (see
  the apex-error image above).
- **`approx_eps_frac` (0.02).** Sweeping `approxPolyDP`'s epsilon showed the
  vertex count collapsing as it grew (9 → 6 → 4 vertices at 0.005 / 0.02 /
  0.03). At 0.02 the two straight radii become single long edges while the arc
  stays a line. Larger values fold the arc into one chord and break that assumption.
- **`canny_low`/`canny_high` (50/150).** The conventional Canny defaults. The
  target is bright-on-black (very high contrast), so the contour is found
  robustly and the exact thresholds are not critical here.
- **`crop_margin` (40).** Chosen so the full shape plus Canny's edge halo sits
  comfortably inside the refine crop, with room to spare around the target.
- **`radii_ratio_min` (0.75) and `apex_angle_range` (80–100°).** Derived from
  the geometry of a 90° sector — equal radii and a 90° apex. Because the apex
  angle is intrinsic to the shape (rotation does not change it), the window is
  tight at 90° ± 10°: enough to absorb pixel discretisation, tight enough to
  reject non-sector blobs (e.g. the 60°/60°/60° triangle a full circle yields).
- **`min_area` (1000).** Expressed in **full-resolution** px². The downscaled
  locate-stage area is normalised back to full resolution before the check, so
  the threshold is a real-image size and does not need re-tuning when
  `work_dim` changes. The sample target is ~52 000 px², so 1000 comfortably
  rejects outliers while accepting the target.

## 4. Testing

```bash
pip install -e ".[dev]"
pytest
```

Two suites:

- **Visual tests** (`tests/test_visual.py`) — run both strategies on synthetic
  quarter circles (several rotations), on a full circle (must be rejected), and
  on the sample images. Each test makes a real assertion *and* saves an
  annotated PNG to `tests/output/` so the detection can be manually checked. The
  output directory is git-ignored.
- **Resource tests** (`tests/test_resources.py`) — see the next section.

## 5. Resource Budget & Deployment

The target is a low-end device (≤4 GB RAM, ARMv7 Cortex, 100 ms per search).
We can run some of the constraints here, without a proper device emulator (QEMU, for instance). Given this, we can do the following:

- **Memory — tested.** `tests/test_resources.py` runs a detection in a fresh
  subprocess and reads its peak RSS (actual physical memory) via `getrusage`,
  asserting it stays under 1 GB.

  A hard `RLIMIT_AS` cap is deliberately **not** used: it limits *virtual*
  address space, and numpy's multithreaded BLAS reserves large virtual regions
  unrelated to real RAM use, so it fails for the wrong reason. Peak RSS is the
  honest metric.

- **Compute time — regression guard only.** The same test asserts compute stays
  under 100 ms, but this is a *guard*, not a guarantee.

- **Thread pinning (deployment tip).** On a few-core CPU, numpy/OpenCV spawning
  one BLAS thread per core can oversubscribe and *slow things down*. Pin the
  math libraries to a single thread on the device:

  ```bash
  export OMP_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  ```

  The resource tests set these for the same reason.
