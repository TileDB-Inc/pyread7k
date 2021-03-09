# Pyread7k

Library for reading 7k files. Low-level direct control is available with the _utils.py functions, which are reexported at top level. A higher-level abstraction is provided by Ping and PingDataset in _ping.py which allows you to work with pings, and automatically get relevant data, instead of reading and managing records individual records manually.


# Work

## Speeding up interpolation

Scipy's implementation of griddata is quite slow when using the linear method, which is necessary to take into account that translations and rotations would result in areas in the wedge of a stationary boat are not necessarily visible. Currently this has been solved by defining a convex hull over the points of the rotated and the translated wedge and then filtering the stationary points using the points-in-polygon method. This has some negative effects because the convex hull doesn't actually create the minimum shape that encapsulate the points, but the minimum number of points that encapsulate the points. Alternative approaches have been to define the points by scanning angles and storing the smallest and largest ranges, however, the point-in-polygon algorithm is highly dependent on the order of points. There is definitely a solution to this, which is basically to sort the points by eithe angle or distance, however, currently none of the approaches have resulted in better polygons. Because this might take a day or two more, I've decided to accept the small disturbances in the short ranges and handle them through a simple min range filter.

The results of the modification - going from linear to nearest is for the translation a ten fold speedup going from 4.4 seconds to 0.42 seconds and for the rotation it's a twenty five fold speedup going from 22 to 0.86 seconds on my machine on a dataset of size 1000x256.

We can still add some speedup by using cuda, but currently the most used cuda based python library Rapids.ai, isn't available to install through pip. It can be installed using conda or through the source code, but I have not alloted time for this right now, so it has to go into the backlog

# Backlog
* Implement rapids
* Add opening angle to seabed intersection function
* Create a function to transform roll, pitch and heave, so that they apply to a rotated sonar
* Add sonar displacement from rotation axis to motion correction code