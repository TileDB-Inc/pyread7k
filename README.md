# Pyread7k

Library for reading 7k files. Low-level direct control is available with the _utils.py functions, which are reexported at top level. A higher-level abstraction is provided by Ping and PingDataset in _ping.py which allows you to work with pings, and automatically get relevant data, instead of reading and managing records individual records manually.