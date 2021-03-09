import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyread7k",
    version="0.0.2",
    author="Rasmus Klett Mortensen",
    author_email="rasmus.mortensen@teledyne.com",
    description="Reading 7k files",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://rot-bitbucket/projects/CB7123/repos/pyread7k",
    packages=["pyread7k"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "psutil==5.8.0",
        "xarray==0.16.2",
        "numpy==1.20.1",
        "scipy==1.6.0",
        "geopy==2.1.0",
        "scikit-image==0.18.1",
        "numba==0.52.0",
    ],
    python_requires=">=3.6",
)
