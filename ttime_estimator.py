import aux

from scipy.stats import gaussian_kde
import numpy as np
from matplotlib import pyplot as plt

from scipy.signal import argrelextrema


def estimate_travel_time(counter,
                         distance=None,
                         coords1=None,
                         coords2=None,
                         speed_limit=None,
                         bandwidth=30,
                         xspacing=5,
                         peak_size_fraction=0.5,
                         show_results=False,
                         ax=None,
                         title="",
                         xscale="",
                         labelrot=30,
                         dashed_line=False):
    """
    Estimate travel time based on data using kernel density estimation.
    Picks the first peak in the prob. density. function.

    Parameters
    ----------
    counter : collections.Counter
        contains the observed raw travel times
    distance : float
        distance between locations in km (used with speed_limit)
    coords1 : (lat, lon)
        coordinates of the from location
    coords2 : (lat, lon)
        coordinates of the to location
    speed_limit : float, optional
        Travel time can not be smaller than this; all peaks
        with travel times smaller than this in the pdf are omitted.
    bandwidth : float, optional
        The bandwidth (stdev) of the used (gaussian) kernel to be used.
        Defaulting to 30.
    xspacing :  float, optional
        The resolution of the kernel density estimate.
        Defaults to 30.
    peak_size_fraction : float, optional
        how large the peak should be
    show_results : bool
        Whether to show the distribution with its results.
        Defaults to false
    ax : matplotlib axes object
        plot the data, kernel density estimate, and peak estimate there
    labelrot : float
        label rotation in degrees
    dashed_line: bool
        dashed line for the original signal?

    Returns
    -------
    peak_time : float
        the estimated travel time
    """
    data_arr = aux.counter_to_array(counter)
    # no other way to go with kernel density estimations currently...

    x_grid = np.arange(0, 3 * 60 * 24, 5)
    kde = gaussian_kde(data_arr, bw_method=bandwidth / data_arr.std(ddof=1))
    density = kde.evaluate(x_grid)

    peak_indices = argrelextrema(density, np.greater)[0]
    if len(peak_indices) < 1:
        return float('nan')

    peak_indices = np.sort(peak_indices)  # should be now sorted
    peak_times = x_grid[peak_indices]
    if distance is None:
        # distance in km
        distance = aux.great_circle_distance(coords1, coords2)

    if speed_limit is not None:
        valid_peak_indices = peak_indices[
            (distance / (peak_times / 60.) < speed_limit)]
    else:
        valid_peak_indices = peak_indices

    peak_times = x_grid[valid_peak_indices]
    peak_vals = density[valid_peak_indices]

    max_val = np.max(peak_vals)

    # actual peak detection:
    peak_time = None
    for time, peak in zip(peak_times, peak_vals):
        if peak < max_val * peak_size_fraction:
            print time, peak, max_val
            continue
        peak_time = time
        break

    if show_results:
        if ax is None:
            _, ax = plt.subplots()
        fig = ax.get_figure()

        # plot (normalized) raw data:
        vals = np.array(counter.keys())
        counts = np.array(counter.values())
        count_sum = np.sum(counts)
        norm_factor = 10.0 * count_sum  # 10 as 10 min spacing
        arg_ord = np.argsort(vals)
        ax.plot(
            vals[arg_ord], np.array(counts[arg_ord]) / norm_factor, '.', c="k", ms=3)
        if dashed_line:
            ax.plot(vals[arg_ord], np.array(
                counts[arg_ord]) / norm_factor, '--', c="k")

        ax.plot(x_grid, density,
                lw=2.5, label='density estimate',
                color="r", alpha=0.6)

        if peak_time is not None:
            ax.text(peak_time + 100, max_val, r"\textbf{" + str(peak_time) + "}", ha='left',
                    va='bottom', color="b", fontsize=16)
            # transform=ax.transAxes, color="b")
            ax.axvline(x=peak_time, linewidth=3.0, color='b')

        if title != 'ttime_as_steps':
            # plt.axvline(
            #     x=60 * 24, linewidth=1.5, color='g', alpha=0.5)
            # plt.axvline(x=60, linewidth=1.5, color='g', alpha=0.5)
            if xscale == 'lin':
                ax.set_xlim([0, 2 * 60 * 24])
            if xscale == 'log':
                ax.set_xlim(10, 10 ** 5)
                ax.set_xscale('log')

        # set titles
        ttitle = title  # + " #" + str(count_sum)
        ax.set_title(ttitle, fontsize=14)

        labels = ax.get_xticklabels()
        for label in labels:
            label.set_rotation(labelrot)
        ax.grid()

        ax.yaxis.get_major_formatter().set_powerlimits((-1, 1))
        return fig, ax, peak_time
    else:
        return peak_time
