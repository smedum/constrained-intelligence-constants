
"""
Setup configuration for Constrained Intelligence Constants package.
"""

from setuptools import setup, find_packages
import os

# Read long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Read dev requirements
with open("requirements-dev.txt", "r", encoding="utf-8") as fh:
    dev_requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="constrained-intelligence-constants",
    version="1.0.0",
    author="Constrained Intelligence Research Team",
    author_email="constrained-intelligence@example.com",
    description="A foundational framework for discovering and applying mathematical constants in bounded intelligent systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/constrained-intelligence-constants",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/constrained-intelligence-constants/issues",
        "Documentation": "https://github.com/yourusername/constrained-intelligence-constants#readme",
        "Source Code": "https://github.com/yourusername/constrained-intelligence-constants",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "validation"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "test": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "machine-learning",
        "optimization",
        "mathematical-constants",
        "golden-ratio",
        "bounded-systems",
        "resource-allocation",
        "convergence-analysis",
        "artificial-intelligence",
    ],
    entry_points={
        "console_scripts": [
            "ci-constants-validate=validation.experimental_validation:main",
        ],
    },
)
