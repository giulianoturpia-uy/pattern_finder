from setuptools import setup, find_packages

setup(
    name="pattern-finder",
    version="0.1.0",
    description=(
        "Pattern detection in TIF images using a strategy pattern, "
        "targeting constrained devices."
    ),
    python_requires=">=3.8",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "opencv-python-headless>=4.5,<5",
        "numpy>=1.21,<3",
        "Pillow>=9.0,<12",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
        ],
    },
)
