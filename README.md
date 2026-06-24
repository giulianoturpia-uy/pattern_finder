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

### 1.4. A note on OpenCV

OpenCV is published on PyPI only as monolithic wheels — every wheel contains
all modules; pip cannot install individual ones (e.g. just `imgproc`). Of the
available wheels, `opencv-python-headless` is the lightest: it strips the GUI
(`highgui`) layer while still providing the only functions this library uses —
`Canny`, `findContours` and `approxPolyDP`, all in the `imgproc` module. That
is why it is the dependency pinned in `setup.py`.

## 2. Code Flow

## 3. Design

### 3.1. Edge Detection, Algorithm