from setuptools import setup, find_packages

# Read requirements
with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Read long description
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="graphfusionai",
    version="0.1.0",
    description="A powerful framework for building graph-based AI agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="contact@graphfusionai.dev",
    url="https://github.com/yourusername/GraphFusionAI",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    include_package_data=True,
    zip_safe=False,
)