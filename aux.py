from geopy.distance import great_circle
import numpy as np

"""
Contains helper functions for computing various stuff.
"""


def great_circle_distance(coords_1, coords_2):
    """
    Return the great circle distance in kilometers.

    Parameters
    ----------

    coords_1 : tuple / list-like
        (lat, lon) in decimal units
    coords_2 : tuple / list-like
        (lat, lon) in decimal units


    Returns
    -------
    dist : float
        distance between coords_1, and coords_2 in kilometers
    """
    return great_circle(coords_1, coords_2).kilometers


def counter_to_array(counter):
    """
    Transform a collections.Counter object to a numpy array
    """
    keys = counter.keys()
    vals = counter.values()
    tot = np.sum(vals)
    arr = np.zeros(tot, dtype=int)
    i = 0
    for key, val in zip(keys, vals):
        arr[i:i + val] = key
        i += val
    return arr
