import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyread7k",
    version="0.0.1",
    author="Rasmus Klett Mortensen",
    author_email="rasmus.mortensen@teledyne.com",
    description="Reading 7k files",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://rot-bitbucket/projects/CB7123/repos/pyread7k",
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "pytest",
        "numpy",
        "scipy",
        "matplotlib",
        "python-dotenv",
        "psutil",
        "xarray"
    ],
    python_requires='>=3.6',
)
