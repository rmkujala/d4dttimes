import pandas as pd

# from this project
import aux


def get_cell_tower_groups(group_coords, tr_dist, site_data_fname):
    """
    Given a list of coordinate pairs (lat, lon), find the cell
    towers which are closest to them.

    Parameters
    ----------
    group_coords : list-like
        list of coordinate pairs (lat, lon)
    tr_dist : float
        Threshold distance in kilometers.
        If the distance from some cell tower to the closest
        element in `group_coords` is larger than `tr_dist`, then
        that cell tower is not assigned to any group.
    site_data_fname : str
        path to the site data

    Returns
    -------
    site_groups : list of lists
        list (in the same order as group_coords)
        where each element is a list of cell tower ids
        ('site_id' in the SITE_ARR_LON_LAT file)
    """
    site_data = pd.read_csv(site_data_fname)
    site_groups = [[] for coord in group_coords]

    # loop over cell towers
    for index, row in site_data.iterrows():
        coords_ct = (row['lat'], row['lon'])

        min_i = None
        min_dist = float('inf')
        for i, other_coords in enumerate(group_coords):
            dist = aux.great_circle_distance(coords_ct, other_coords)
            if dist < min_dist:
                min_i = i
                min_dist = dist
        if min_dist < tr_dist:
            site_groups[min_i].append(row['site_id'])
    return site_groups

    # assign each cell tower to a group
    # corresponding to a coordinate

if __name__ == '__main__':
    data = pd.read_csv('cities_coords.csv', sep="\t")
    print data
    lats = data['lat'].values
    lons = data['lon'].values
    coords = zip(lats, lons)

    radius = 10  # in kilometers
    groups = get_cell_tower_groups(coords, radius)
