import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyread7k",
    version="1.0.0",
    author="Teledyne RESON",
    description="Reading 7k files",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Teledyne-Marine/pyread7k",
    packages=["pyread7k"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "psutil==5.8.0",
        "numpy==1.20.1",
        "geopy==2.1.0",
    ],
    python_requires=">=3.6",
)
