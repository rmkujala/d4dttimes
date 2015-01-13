import multiprocessing


def run_in_parallel(work_func, arg_list, n_cpus, chunksize=1):
    """
    Run ``work_func(args)`` with n_cpus number of processors in parallel

    Parameters
    ----------
    work_func : callable object (function)

    arg_list : list-like
        list of lists containing the input arguments

    n_cpus : int
        number of cpus

    Returns
    -------
    result_list : list
        [work_func(args) for args in arg_list]
    """
    # mainly for debugging purposes and generality
    if n_cpus == 1:
        result_list = []
        for args in arg_list:
            result_list.append(work_func(args))
    else:
        pool = multiprocessing.Pool(processes=n_cpus)
        result_list = \
            pool.map_async(work_func, arg_list, chunksize=1).get(31536000)
    return result_list
