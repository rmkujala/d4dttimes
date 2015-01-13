import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import cell_tower_groups_from_coords as ctgfc
import comp_tower_tower_travel_times as ctimes
import ttime_estimator as tte
import aux

"""
The data analysis pipeline is as follows:

1. Given a set of coordinates, and all cell tower locations, compute cell tower
    groups corresponding to each coordinate point.
2. Compute all travel times between the cell tower groups.
3. Extract typical travel times from sets of travel times.

Note that the end results of this pipeline are not meaningful in any sense as
the input data is just 'dummy data'.
"""

# 1. extracting cell tower groups
data = pd.read_csv('cities_coords.csv', sep="\t", )
lats = data['lat'].values
lons = data['lon'].values

coords = zip(lats, lons)
radius = 10  # in kilometers
# site data fname
groups = ctgfc.get_cell_tower_groups(coords, radius, 'SITE_ARR_LONLAT.CSV')

# 2. compute all travel times between the cell tower groups.

ctimes.comp_filtered_travel_times_between_cell_tower_groups(
    groups,
    ['sample_cdr_data.csv'],
    1,  # ncpus
    "sample_travel_times",  # fname prefix
    verbose=True,
    filter_loops_and_stops=False
)

data = np.load("sample_travel_times.npy")

i = 0
j = 1
gi = groups[i]
gj = groups[j]


# 3. then extract the travel times

fig, ax = plt.subplots()

tte.estimate_travel_time(
    data[i, j],
    speed_limit=120,
    bandwidth=30,
    xspacing=5,
    peak_size_fraction=0.5,
    show_results=True,
    ax=ax,
    xscale='lin',
    distance=aux.great_circle_distance((lats[i], lons[i]), (lats[j], lons[j])),
    labelrot=0
)

ax.set_xlim([0, 500])
ax.set_xlabel('Travel time $t$ (min)')
ax.set_ylabel('Prob. density $P(t)$')
plt.show()
