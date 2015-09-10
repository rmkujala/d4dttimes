import numpy as np
from scipy.spatial.distance import cdist
from collections import Counter


def gaussian_1d_kde(data, bandwidth, eval_points, vec_weights=None):
    """
    Simple gaussian kernel density estimator in one dimension.
    Inspiration taken from scipy.stats.gaussian_kde
    and
    http://stackoverflow.com/questions/27623919/weighted-gaussian-kernel-density-estimation-in-python
    -> http://nbviewer.ipython.org/gist/tillahoffmann/f844bce2ec264c1c8cb5


    Parameters
    ----------
    data: 1d numpy array or a collections.Counter
        the observed values
    bandwidth: float
        width of the gaussian to use for filtering
    eval_points: np.array
        1d vector of floats where one should estimate the error
    vec_weights: np.array, optional
        if data is a 1d numpy array, weights for each sample can be provided
    """
    if isinstance(data, Counter):
        datavec = data.keys()
        weights = data.values()

    else:
        datavec = data
        if vec_weights is None:
            weights = np.ones(data.shape)
        else:
            weights = vec_weights

    datavec = np.atleast_2d(datavec)
    eval_points = np.atleast_2d(eval_points)
    weights = np.atleast_2d(weights)
    weights = weights/float(np.sum(weights))


    # compute the normalised residuals
    chi2 = cdist(eval_points.T, datavec.T, 'euclidean')**2
    # compute the pdf
    # print chi2
    s = float(bandwidth)
    norm_factor = (s*np.sqrt(2*np.pi))

    pdf = np.sum(np.exp((-.5 * chi2)/s**2) * weights, axis=1) / norm_factor
    return pdf



if __name__ == "__main__":
    "Testing the implementation:"

    mult = 10000
    data = np.zeros(100*mult)
    weights = np.ones(100*mult)
    data[75*mult:] = 1
    weights[75*mult:] = 1
    eval_points = np.linspace(-3, 3, 200)

    c = Counter(iterable=data)

    import time
    print time.time()
    pdf = gaussian_1d_kde(data, 0.1, eval_points, vec_weights=weights)
    print time.time()
    pdf2 = gaussian_1d_kde(c, 0.1, eval_points)
    print time.time()

    import pylab
    pylab.plot(eval_points, pdf)
    pylab.plot(eval_points, pdf2, "o")
    pylab.show()

