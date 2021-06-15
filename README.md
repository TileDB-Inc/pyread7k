[![Teledyne Logo](images/TeledyneLogo.png)](teledynemarine.com)


# Pyread7k
Pyread7k is a library for reading 7k files. It provides a high-level interface to the data in a file, with an API that is a compromise between being ergonomic, while still being easy to correllate with the Data Format Definition.

Using the PingDataset class, file data can be read on an as-needed basis with low overhead. Each Ping in the dataset contains the full set of records which are related to that specifc ping. This means you can read the amplitudes, settings, motion, etc. for any single ping without worrying about file offsets or any low-level details at all.

Low-level direct control is available with the _utils.py functions, which are reexported at top level. These allow you to read all records of a specific type manually

The 7k protocol is described in the [Data Format Definition](documents/DATA%20FORMAT%20DEFINITION%20-%20%207k%20Data%20Format.pdf) document.

# Installation
The library can be installed directly from the Teledyne github repository using pip as seen below. All dependencies should be automatically installed.
```bash
pip install git+https://github.com/Teledyne-Marine/pyread7k.git
```


# Getting started
Working with the pyread7k library is quite intuitive, and given that you have a s7k file, you can load a dataset using the PingDataset class:
```python
import pyread7k
dataset = pyread7k.PingDataset("path/to/file.7k")
```
This gives you access to the pings, which consist of IQ or beamformed records, along with related data. All data is loaded on-demand when accessed:
```python
import numpy as np
for ping in dataset:
    if ping.has_beamformed:
        # Print mean amplitude for each ping with 7018 data
        print("Mean:", np.mean(ping.beamformed.amplitudes)) 
    # Print selected gain level for each ping
    print("Gain:", ping.sonar_settings.gain_selection)
```


# Dependencies

* `Python` 3.6 or later
* `psutil` 5.8.0
* `numpy` 1.20.1
* `scipy` 1.6.0
* `geopy` 2.1.0
* `scikit-image` 1.18.0
* `numba` 0.53.0


# Developing
It is easy to add new functionality, such as supporting more record types, to pyread7k. Get up and running:
- Install the [Poetry](https://python-poetry.org/docs/) dependency/package manager
- Clone the repositorty by executing `git clone git@github.com:Teledyne-Marine/pyread7k.git`
- Create a development environment by navigating to the repo folder and running `poetry install`

You should now have a functional environment! You can run the test suite by executing `poetry run pytest`. The first time you do so, you must run it with the `-s` argument: `poetry run pytest -s`. This allows the test suite to ask you to input paths to 7k files to use for the testing.
Issues and pull requests are always welcome!