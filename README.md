---------
d4dttimes
---------

This repository contains the python code for extracting travel times between different locations based on anonymized call detail record data.

The format of the input data is explained in http://arxiv.org/abs/1407.4885 (tower level mobility data).
The original research data was provided by Orange and Sonatel in conjunction with the Data for Development Challenge Senegal (D4D 2014, http://www.d4d.orange.com/en/home/).

This repository contains the 'minimal set' of tools for extracting travel times from CDR data provided in the above format.
(This code is a cleaned, stripped down version of the actual research code.)
For more advanced pipelining of data, you may contact the author of this repository.


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

	Helper functions for loading data.

ttime_estimator.py

	Estimates travel times given the travel times obtained using `comp_tower_tower_travel_times.py`
  A function for computing bootstrap estimates.

example_pipeline.py

	Simple script to show how these should be used in combination.


Testing
-------

Test modules:

	test_aux.py
	test_comps.py

Dependencies
------------

This package depends on the following packages.

- The SciPy stack: (Numpy, Scipy, Matplotlib)
- geopy (used for computing distances between coordinates)

City coordinate data (cities_coords_senegal.csv) has been obtained from http://www.tageo.com/index-e-sg-cities-SN.htm
