# Pyread7k

Library for reading 7k files. Low-level direct control is available with the _utils.py functions, which are reexported at top level. A higher-level abstraction is provided by Ping and PingDataset in _ping.py which allows you to work with pings, and automatically get relevant data, instead of reading and managing records individual records manually.

The library also contains some additional functionality for processing, motion correcting, translating, and plotting ping data.

# Requirements

The library doesn't have any other dependencies than the ones defined in the setup.py, so it should manage all python based dependencies.
However you will need to aquire some s7k files on your own.

The library can be installed directly from the Teledyne github repository using pip as seen below:
```bash
pip install git+https://github.com/Teledyne-Marine/pyread7k.git
```

# Getting started
Working with the pyread7k library is quite intuitive, and given that you have a s7k file, you can load a dataset using the PingDataset class.
This gives you access to the pings, which consist of IQ or beamformed data. Much of the functionality currently implemented is geared towards
beamformed data, but Beamforming is in the backlog, so that s7k files with only IQ data can be processed as well.

The library also contains code to process pings in a variety of ways. These processing functions are grouped into an object called Beamformed, which when instantiated will allow you to process beamformed pings like so:

```python
# Excluding ranges or beams
ping.exclude_ranges(min_range_meter=30, max_range_meter=100)
ping.exclude_bearings(min_beam_index=10, max_beam_index=70)
```

It also contains functions for decimating, resampling, and normalizing the data in a variety of ways.

Last but not least, the library contains code for motion correction and translation of the pings, i.e. the process of transforming the ping from a rolled/pitched/yawed ping into a ping taken by a stationary vessel. The translation functionality can be used to get the ping data as if the vessel had been in a different location, and is mainly used for ping stacking.


# Notes

## Speeding up interpolation

Scipy's implementation of griddata is quite slow when using the linear method, which is necessary to take into account that translations and rotations would result in areas in the wedge of a stationary boat are not necessarily visible. Currently this has been solved by defining a convex hull over the points of the rotated and the translated wedge and then filtering the stationary points using the points-in-polygon method. This has some negative effects because the convex hull doesn't actually create the minimum shape that encapsulate the points, but the minimum number of points that encapsulate the points. Alternative approaches have been to define the points by scanning angles and storing the smallest and largest ranges, however, the point-in-polygon algorithm is highly dependent on the order of points. There is definitely a solution to this, which is basically to sort the points by eithe angle or distance, however, currently none of the approaches have resulted in better polygons. Because this might take a day or two more, I've decided to accept the small disturbances in the short ranges and handle them through a simple min range filter.

The results of the modification - going from linear to nearest is for the translation a ten fold speedup going from 4.4 seconds to 0.42 seconds and for the rotation it's a twenty five fold speedup going from 22 to 0.86 seconds on my machine on a dataset of size 1000x256.

We can still add some speedup by using cuda, but currently the most used cuda based python library Rapids.ai, isn't available to install through pip. It can be installed using conda or through the source code, but I have not alloted time for this right now, so it has to go into the backlog

# Backlog
* Implement rapids
* Add opening angle to seabed intersection function
* Create a function to transform roll, pitch and heave, so that they apply to a rotated sonar
* Add sonar displacement from rotation axis to motion correction code
* Add beamforming