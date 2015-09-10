import networkx as nx

from collections import Counter

import dataio

import numpy as np
import cPickle as pkl
import pandas as pd

import parallel_run_helper as prh
import sys


def comp_travel_times_between_cell_tower_groups(
        cell_tower_id_groups,
        csv_fnames=None,
        n_cpus=1,
        verbose=False,
        fname_prefix=None):

        # compute travel times in parallel:
    if verbose:
        print "extracting travel times:"
    n_csvs = len(csv_fnames)
    args = zip(csv_fnames, [cell_tower_id_groups for i in range(n_csvs)])
    results = prh.run_in_parallel(
        _get_travel_times_between_cell_tower_groups_worker,
        args,
        n_cpus,
        chunksize=1
    )

    if verbose:
        print "extracted travel times"

    result_mat = results[0]
    for mat in results[1:]:
        for i in range(len(result_mat)):
            for j in range(len(result_mat[i])):
                # result_mat[i,j] should be a counter object!
                result_mat[i, j].update(mat[i, j])

    del results

    all_cell_tower_ids = set([])
    for i in range(len(cell_tower_id_groups)):
        group_i = set(cell_tower_id_groups[i])
        all_cell_tower_ids.update(group_i)

    if fname_prefix is None:
        fname_prefix = "tmp"

    result_fname = fname_prefix + ".npy"
    np.save(result_fname, result_mat)
    cell_tower_id_groups_fname = fname_prefix + "_info.pkl"
    with open(cell_tower_id_groups_fname, "w") as f:
        pkl.dump(cell_tower_id_groups, f)


def comp_filtered_travel_times_between_cell_tower_groups(
        cell_tower_id_groups,
        csv_fnames,
        n_cpus,
        fname_prefix,
        verbose=False,
        filter_loops_and_stops=False,
        valid_start_interval=None,
        filter_transitions=False,
        travel_time_as_steps=False,
        only_transtitions=False,
        n_rows=None):

            # compute travel times in parallel:
    if verbose:
        print "extracting travel times:"

    n_csvs = len(csv_fnames)
    args = zip(csv_fnames,
               [cell_tower_id_groups for i in range(n_csvs)],
               [verbose for i in range(n_csvs)],
               [n_rows for i in range(n_csvs)],
               [filter_loops_and_stops for i in range(n_csvs)],
               [valid_start_interval for i in range(n_csvs)],
               [filter_transitions for i in range(n_csvs)],
               [travel_time_as_steps for i in range(n_csvs)],
               [only_transtitions for i in range(n_csvs)]
               )

    results = prh.run_in_parallel(
        _get_filtered_travel_times_between_ctgroups_worker,
        args,
        n_cpus,
        chunksize=1
    )

    if verbose:
        print "extracted travel times"

    result_mat = results[0]
    for mat in results[1:]:
        for i in range(len(result_mat)):
            for j in range(len(result_mat[i])):
                # result_mat[i,j] should be a counter object!
                result_mat[i, j].update(mat[i, j])

    del results

    all_cell_tower_ids = set([])
    for i in range(len(cell_tower_id_groups)):
        group_i = set(cell_tower_id_groups[i])
        all_cell_tower_ids.update(group_i)

    result_fname = fname_prefix + ".npy"
    np.save(result_fname, result_mat)
    info = {'cell_tower_id_groups': cell_tower_id_groups,
            'csv_fnames': csv_fnames,
            'n_cpus': n_cpus,
            'verbose': verbose,
            'fname_prefix': fname_prefix,
            'filter_loops_and_stops': filter_loops_and_stops,
            'valid_start_interval': valid_start_interval,
            'filter_transitions': filter_transitions,
            'travel_time_as_steps': travel_time_as_steps,
            'n_rows': n_rows,
            }
    cell_tower_id_groups_fname = fname_prefix + "_info.pkl"
    with open(cell_tower_id_groups_fname, "w") as f:
        pkl.dump(info, f)


def _get_travel_times_between_cell_tower_groups_worker(args):
    return _get_travel_times_between_cell_tower_groups(*args)


def _get_filtered_travel_times_between_ctgroups_worker(args):
    print args[0]
    sys.stdout.flush()
    return _get_filtered_travel_times_between_ctgroups(*args)


def _get_all_cell_tower_ids_and_assert_no_overlap(cell_tower_id_groups):
    """
    Get cell tower id groups, and assert that they do not share any common
    ids. Returns also a set of the all cell_tower_ids.
    Main use as a helper function for the computations funcs.
    """
    all_cell_tower_ids = set([])
    for i in range(len(cell_tower_id_groups)):
        group_i = set(cell_tower_id_groups[i])
        all_cell_tower_ids.update(group_i)
        for j in range(i + 1, len(cell_tower_id_groups)):
            group_j = set(cell_tower_id_groups[j])
            assert len(group_i.intersection(group_j)) == 0
    return all_cell_tower_ids


def _transform_time_stamps_to_minutes(data, ts_column_label="timestamp"):
    """
    Transform the pandas (?) timestamps to minutes.

    Parameters
    ----------
    data : pd.DataFrame object
        the data whose data corresponding to param ts_column_label should
        be transformed
    ts_column_label : str, optional
        defaults to 'timestamp', which is used when reading the data with
        dataio.read_mobility_csv.

    Returns
    -------
    data : pd.DataFrame
        the transfromed object
    """
    old_chained_assignment_option = pd.options.mode.chained_assignment
    # to supress warnings
    pd.options.mode.chained_assignment = None
    microseconds_to_mins_divisor = 10 ** 9 * 60
    data[ts_column_label] = pd.to_datetime(data[ts_column_label])
    data[ts_column_label] = data[ts_column_label].values.astype(
        np.int64) / microseconds_to_mins_divisor
    # put warnings back on if they were before:
    pd.options.mode.chained_assignment = old_chained_assignment_option
    return data


def _get_filtered_travel_times_between_ctgroups(
        fname,
        cell_tower_id_groups,
        verbose=True,
        nrows=None,
        filter_loops_and_stops=False,
        valid_start_interval=None,
        filter_transitions=False,
        travel_time_as_steps=False,
        only_transtitions=False):
    """
    Obtain filtered travel times between cell tower groups.

    Parameters
    ----------
    fname : str
        Path to the csv filename
    cell_tower_id_groups : list-like
        List of cell tower ids (indexed from 1->) corresponding to the
        first location
    verbose : bool
        Whether to print out a lot of stuff
    nrows : int, optional
        How many rows to consider from the csv file.
        (Mainly useful for debugging and estimating the computation times)
        By default the whole csv file is considered
    filter_loops_and_stops : bool, optional
        In a travel of the following form: i-j-l-l-j-j-l-k,
        minimizes the travel time, by first creating a directed graph
        with links (i->j, j->l, l->j, l->k), and then finding the shortest path
        in this directed graph from i to k.
    valid_start_interval : tuple of ints or str, optional
        "morning": consider only travels starting between 04-16
        "evening": consider only travels starting between 16-04
        if tuple of floats, consider events starting between
            valid_start_interval[0] and valid_start_interval[1]
            where these values are represented as hours:
            examples: (end and start points are included!)
                [6.5, 8]
                [16, 4] # from 16 to 04
                [22, 21] # all but one hour
    travel_time_as_steps :  bool, optional
        return the number of steps required for traveling instead of any
        time related quantity
    store_travel_details : bool, optional

    Returns
    -------
    arr : 2d numpy array
        If store_travel_details == False:
            arr consists from collections.Counter objects
            (storing the counts of each travel time)
        If store_travel_details == True:
            arr consists of
            pandas.DataFrame with fields
            departure_time, travel_time

    See also
    --------
    :py:func:`_get_travel_times_between_cell_tower_groups`
    (does the same thing in a more difficult way)
    """

    n_groups = len(cell_tower_id_groups)
    all_cell_tower_ids = \
        _get_all_cell_tower_ids_and_assert_no_overlap(cell_tower_id_groups)
    data = dataio.read_mobility_csv(fname, nrows=nrows)

    data.sort(["user_id", "timestamp"], inplace=True)

    data = _transform_time_stamps_to_minutes(data)

    # initialize the container for travel times
    arr = np.zeros((n_groups, n_groups), dtype=object)
    for i in range(n_groups):
        for j in range(n_groups):
            if store_travel_details:
                arr[i, j] = []
            else:
                arr[i, j] = Counter()

    interesting_indices_bool = data['site_id'].isin(all_cell_tower_ids)
    idx_to_group = np.zeros(len(interesting_indices_bool), dtype=int)
    for i, group in enumerate(cell_tower_id_groups):
        idx_to_group[data['site_id'].isin(group).values] = i
    interesting_indices = np.nonzero(interesting_indices_bool)[0]

    prev_visit_indices = [None] * n_groups
    last_user = None

    groups = set(range(n_groups))
    for cur_index in interesting_indices:
        row = data.iloc[cur_index]
        cur_user = row['user_id']
        cur_group = idx_to_group[cur_index]

        # if cur_user == 1 and cur_group == 1:
        #     from nose.tools import set_trace
        #     set_trace()

        if last_user == cur_user:
            g_last_idx = prev_visit_indices[cur_group]
            # ignore the same group! -> loop oever "groups - cur_group"
            for other_group in groups.difference([cur_group]):
                og_last_idx = prev_visit_indices[other_group]
                if og_last_idx is not None and og_last_idx > g_last_idx:
                    path = data.iloc[og_last_idx:cur_index + 1]
                    t = extract_travel_time(
                        path,
                        filter_loops_and_stops=filter_loops_and_stops,
                        valid_start_interval=valid_start_interval,
                        filter_transitions=filter_transitions,
                        travel_time_as_steps=travel_time_as_steps,
                        only_transitions=only_transtitions
                    )
                    if t is not None:
                        if not store_travel_details:
                            # add one instance of t
                            # to the counter "other to cur"
                            arr[other_group, cur_group][t] += 1
                        else:
                            # store the pair of (travel_time, duration)
                            timestamp = path['timestamp'].values[0]
                            departure_id = path.iloc[0]['site_id']
                            arrival_id = path.iloc[-1]['site_id']
                            trip_detail_tuple = (timestamp, t, departure_id, arrival_id)
                            arr[other_group, cur_group].append(trip_detail_tuple)
        else:
            prev_visit_indices = [None] * n_groups

        # after visiting:
        last_user = cur_user
        prev_visit_indices[cur_group] = cur_index

    if store_travel_details:
        # Make lists pandas DataFrames, if departure times and travel durations
        # are to be stored
        columns = ("departure_time", "travel_duration", "from_id", "to_id")
        for i in range(n_groups):
            for j in range(n_groups):
                if len(arr[i,j]) > 0:
                    arr[i, j] = pd.DataFrame(arr[i,j], columns=columns)
                else:
                    arr[i, j] = pd.DataFrame(None, columns=columns)
    return arr


def extract_travel_time(df,
                        filter_loops_and_stops=False,
                        valid_start_interval=None,
                        filter_transitions=False,
                        travel_time_as_steps=False,
                        only_transitions=False):
    """
    Extract travel time from a pandas time-frame using some advanced filters.

    Parameters
    ----------
    df : pandas DataFrame
        should have at least fields 'user_id', 'timestamp' and 'site_id'
    filter_loops_and_stops : bool, optional
        whether to filter loops and stops away
    valid_start_interval : tuple of ints
    filter_transitions: bool
        whether to filter transitions
    travel_time_as_steps: bool
        measure travel time as number of intermediate steps taken

    Returns
    -------
    ttime : int
        the computed travel time
    """
    timestamps = df['timestamp'].values
    # whether to completely ignore
    # based on travel start time
    if valid_start_interval is not None:
        timestamp_in_mins = df.iloc[0]['timestamp']
        if not _timestamp_within_interval(timestamp_in_mins,
                                          valid_start_interval):
            return None
    # ignore if transition?
    if filter_transitions:
        if len(timestamps) == 2:
            return None
    if only_transitions:
        if len(timestamps) != 2:
            return None
    # measure travel time as steps?
    if travel_time_as_steps:
        return len(timestamps) - 1
    # filter loops and stops?
    if filter_loops_and_stops:
        digraph = nx.MultiDiGraph()  # multi for choosing the minimum
        start_node = df.iloc[0]['site_id']
        end_node = df.iloc[-1]['site_id']
        for i in range(1, len(df)):  # this loop could be omitted?
            prev_site = df.iloc[i - 1]['site_id']
            prev_time = df.iloc[i - 1]['timestamp']
            cur_site = df.iloc[i]['site_id']
            cur_time = df.iloc[i]['timestamp']
            weight = cur_time - prev_time
            assert weight >= 0
            digraph.add_edge(
                prev_site, cur_site, weight=cur_time - prev_time)
        ttime = nx.shortest_path_length(
            digraph, start_node, end_node, weight="weight")
        return ttime
    # or just go with the basic setup?
    return timestamps[-1] - timestamps[0]


def _timestamp_within_interval(timestamp_in_mins, valid_interval):
    """
    Whether the time of the day falls within ghe range of the given valid start
    interval

    Parameters
    ----------
    timestamp_in_mins : int, or array like
        minutes after epoch,
    valid_interval:
        see :py:func:`_get_filtered_travel_times_between_ctgroups`
        valid_start_interval

    Returns
    -------
    within: bool
        whether the time of the day corresponding to the timestamp falls within
        the range given by valid_interval
    """
    hours = (timestamp_in_mins % (24 * 60)) / 60.0
    if valid_interval[0] <= valid_interval[1]:
        to_return = (
            (hours >= valid_interval[0]) * (hours <= valid_interval[1]))
    else:
        to_return = (
            (hours >= valid_interval[0]) + (hours <= valid_interval[1]))
    if isinstance(to_return, int):
        to_return = bool(to_return)
    return to_return


def _get_travel_times_between_cell_tower_groups(
        fname,
        cell_tower_id_groups,
        verbose=True,
        nrows=None):
    """
    OLD IMPLEMENTATION!

    Compute travel times between cell tower_id_groups.
    Here "travel time" between i and j is defined as the time taken
    for moving from i to j independent of where a user is seen in
    between.

    Parameters
    ----------
    fname : str
        path to a mobility csv file
    cell_tower_id_groups : list of collections
        list of cell_tower_id_groups, e.g. [[1], [3], [4,5]]

    Returns
    -------
    arr : a 2D numpy array of lists
        arr[i, j] corresponds to the list of travel times
        from cell tower_id_groups[i] to cell_tower_id_groups[j]

    See also
    --------
    :py:func:`_get_filtered_travel_times_between_ctgroups` does the same
    and even more than this function. There the looping structure over
    the data is also a lot simpler than here.
    """

    # check the validity of the sets (should be non-overlapping)
    # and get a list of all of the cell tower ids

    all_cell_tower_ids = \
        _get_all_cell_tower_ids_and_assert_no_overlap(cell_tower_id_groups)

    pd.options.mode.chained_assignment = None
    data = dataio.read_mobility_csv(fname, nrows=nrows)
    f_data = data[data['site_id'].isin(all_cell_tower_ids)]
    f_data.sort(["user_id", "timestamp"], inplace=True)

    if verbose:
        print "sorted"

    f_data = _transform_time_stamps_to_minutes(f_data)

    # consider only data fields with a site id corresponding to the cell_tower
    # ids

    # transform site_ids to groupwise indices (0, 1, .. )
    groupwise_events = []
    for group in cell_tower_id_groups:
        indices = f_data['site_id'].isin(group)
        groupwise_events.append(indices)

    # to suppress warnings:
    for i, events in enumerate(groupwise_events):
        f_data.loc[events, 'site_id'] = i
    # test that the warning does not hold here:

    # assert f_data['site_id'].isin(range(len(groupwise_events))).all()
    pd.options.mode.chained_assignment = "warn"

    # from this point on, cell_tower, or site corresponds to one element of
    # cell_tower_id_groups!!!!
    cell_tower_ids = np.arange(len(cell_tower_id_groups))

    n_cell_ids = len(cell_tower_id_groups)

    arr = np.zeros((n_cell_ids, n_cell_ids), dtype=object)
    for row in arr:
        for i in range(len(row)):
            row[i] = Counter()

    prev_user = None
    prev_site = None

    # last visit times of sites
    last_visit_times = dict(zip(cell_tower_ids, [None] * n_cell_ids))

    # for site-sequence such as [1,2,2]
    # only one travel time from [1 -> 2] should be counted
    # the matrix below is used for storing the information needed during
    # computation
    # the name of the variable should be self-explaining
    visited_i_after_being_in_j = np.zeros((n_cell_ids, n_cell_ids), dtype=bool)

    if verbose:
        print "staring to loop over data"
        print "number of rows = ", len(f_data)

    # need to loop through each row separately, as ther are
    # pretty complicated conditions for computing events:
    for _, row in f_data.iterrows():
        user, time, site = row
        if user == prev_user:  # count times only if still the same user
            # update the last visit time of the cur. site
            last_visit_times[site] = time
            if site != prev_site:  # neglect if same site
                # get the site index
                cur_s_index = np.nonzero(cell_tower_ids == site)[0][0]
                # loop over all the other sites, to get all possible travel
                # times:
                for some_prev_s_index, other_site in enumerate(cell_tower_ids):
                    if other_site != site:  # neglect if other is current site
                        # the current site should not have been visited after visiting
                        # other_site, to count a travel time
                        if not visited_i_after_being_in_j[cur_s_index, some_prev_s_index]:
                            # travel time can only be counted if last_visit
                            # time exists:
                            if last_visit_times[other_site] is not None:
                                travel_time_i_to_j = (
                                    time - last_visit_times[other_site])
                                arr[some_prev_s_index, cur_s_index][
                                    travel_time_i_to_j] += 1
                visited_i_after_being_in_j[cur_s_index, :] = True
                visited_i_after_being_in_j[:, cur_s_index] = False
        else:
            # start dealing with a new user:
            # initialize all data structure used in computation:
            last_visit_times = \
                dict(zip(cell_tower_ids, [None] * n_cell_ids))
            visited_i_after_being_in_j = np.zeros((n_cell_ids, n_cell_ids),
                                                  dtype=bool)
            prev_user = user

        prev_site = site
        last_visit_times[site] = time

    return arr
