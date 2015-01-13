---------
d4dttimes
---------

This repository contains the python code for extracting travel times between different locations as provided by Orange for the Data for Development Challenge Senegal (D4D 2014).
(See http://www.d4d.orange.com/en/home/ )

The format of the input data is defined in http://arxiv.org/abs/1407.4885

This package provides only the 'minimal set' of tools for extracting travel times from CDR data.
For more sophisticated pipelining of the data, you may contact the author of this repository.
(This code is a stripped down version of the actual research code.)

Author:
-------
Rainer Kujala, Rainer.Kujala [at sign) aalto.fi

Contents:
---------
aux.py

	A couple of helper functions.

cell_tower_groups_from_coords.py

	Function how to group cell towers given a location and a radius.

comp_tower_tower_travel_times.py

	Functions for computing all travel times between two locations.

compute_on_road_distances.py

	Functions for computing the on-road distances between locations.

dataio.py

	Data input helpers.

ttime_estimator.py

	Estimates travel times given the travel times obtained using `comp_tower_tower_travel_times.py`

example_pipeline.py

	Simple script to show how these should be used.


Testing:
--------

Test modules:

	test_aux.py
	test_comps.py

Dependencies:
-------------

This package depends on the following packages.

- The SciPy stack: (Numpy, Scipy, Matplotlib)
- geopy (used for computing distances between coordinates)
