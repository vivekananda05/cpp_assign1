from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
import subprocess
import sys

class CMakeBuild(build_ext):
    def run(self):
        # Build C++ extension with CMake
        build_dir = "build"
        
        cmake_command = [
            "cmake",
            "-S", ".",
            "-B", build_dir,
            "-DCMAKE_BUILD_TYPE=Release"
        ]
        
        if sys.platform == "win32":
            cmake_command.extend(["-G", "Visual Studio 17 2022"])
        else:
            cmake_command.extend(["-G", "Unix Makefiles"])
        
        # Run CMake
        subprocess.run(cmake_command, check=True)
        
        # Build
        build_command = ["cmake", "--build", build_dir, "--config", "Release"]
        subprocess.run(build_command, check=True)

setup(
    name="custom_dl_framework",
    version="1.0.0",
    author="Your Name",
    description="Custom Deep Learning Framework with C++ Backend",
    packages=find_packages(where="src/python"),
    package_dir={"": "src/python"},
    install_requires=[
        "pybind11>=2.10.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.65.0",
    ],
    python_requires=">=3.8",
    cmdclass={"build_ext": CMakeBuild},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
    ],
)