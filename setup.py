from setuptools import setup, find_packages
from pathlib import Path

root = Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (root / "README.rst").read_text(encoding="utf-8")

setup(
    name="pyqn",
    version="1.4.1",
    description="A package for managing physical units and quantities",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/xnx/pyqn",
    author="Christian Hill",
    author_email="xn.hill@gmail.com",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: OS Independent",
    ],
    keywords="chemistry, units, physical quantities, unit conversion",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=[
        "pyparsing>=2.3",
        'importlib-resources>=1.0; python_version < "3.7.0"',
    ],
    extras_require={"dev": ["black", "pytest-cov", "tox", "ipython"]},
    # package_data will include all the resolved globs into both the wheel and sdist
    # package_data={},
    # no need for MANIFEST.in, which should be reserved only for build-time files
    project_urls={
        "Bug Reports": "https://github.com/xnx/pyqn/issues",
    },
)
