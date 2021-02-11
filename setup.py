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
        "pytest",
        "numpy",
        "scipy",
        "geopy",
        "matplotlib",
        "python-dotenv",
        "psutil"
    ],
    python_requires='>=3.6',
)
