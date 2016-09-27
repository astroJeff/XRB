import sys      # for sys.path
import time     # for time()
import math     # for sqrt()


# DEFAULT VALUES

_DEF_TIME_IT_THRESHOLD = 0.01
_DEF_TIME_IT_RUNS = 30
_DEF_TIME_RESOLUTION_REPS = 1000


# IMPORT ALL USEFUL MODULES

sys.path = ['../constants', \
            '../SF_history', \
            '../binary', \
            '../notebooks', \
            '../pop_synth', \
            '..'
           ] + sys.path

import stats
import constants
import sf_history
import pop_synth
import density_contour
import binary_evolve
import load_sse


# TIMING FUNCTIONS

_core_timeref = time.time() # set with current time during module initialization

def tic():
    """ Sets the time reference to the current time """
    global _core_timeref
    _core_timeref = time.time()


def toc():
    """ Returns the time elapsed since
        (i)  the last call of toc(), or
        (ii) the initialization of the module
    """
    global _core_timeref
    return time.time() - _core_timeref


def time_it(expr, threshold = _DEF_TIME_IT_THRESHOLD, runs = _DEF_TIME_IT_RUNS):
    """ Returns the execution time of an expression. If the result is smaller
        than a threshold, it repeats until the total duration is larger and
        returns the average. The total process is repeated 'runs' times so that
        the average and standard error of the execution to be returned.

        Parameters
        ----------
        expr:       string
                    python expression to be timed, e.g. a function call

        threshold:  float >= 0
                    threshold for total duration of each run

        runs:       integer > 0
                    number of runs

        Returns
        -------
        mean:       float
                    average execution time of 'expr'

        std:        float
                    standard error of execution time
    """

    assert threshold >= 0
    assert runs > 0

    results = []                            # stores the outcome of each run
    for run in range(runs):
        calls = 0                           # count how many timings took place
        duration = 0                        # total duration
        tic()                               # start clock
        while duration < threshold:
            exec(expr)                      # execute expression
            calls += 1
            duration = toc()
        results.append(duration / calls)    # return the average

    mean = sum(results) / runs
    std = math.sqrt(sum([(x - mean) ** 2 for x in results]) / (runs - 1.0))
    return mean, std


def time_resolution(reps = _DEF_TIME_RESOLUTION_REPS, report = False):
    """ Discovers the timing resolution of tic() and toc() through experiments

    Parameters
    ----------
    reps:   integer
            Number of resolution experiments which should be larger than 20 so
            that confidence intervals do not return minimum and maximum value.
            If < 20, assertion error occurs.

    report: boolean
            Whether or not the function will print report

    Returns
    -------
    median: float
            Median of the results from resolution experiments

    mean:   float
            The sample mean of the results

    std:    float
            The sample standard deviation of the results

    ci:     list of two floats
            The confidence interval at 0.95 significance level

    """

    assert reps >= 20

    # measure minimum duration measured by successive tic() - toc() calls
    results = []
    for i in range(reps):
        tic()
        while True:
            duration = toc()
            if duration > 0: break
        results.append(duration)

    # sort results so that median and confidence intervals can be computed
    results.sort()

    # find median
    median = results[reps / 2]
    if reps % 2 == 0:
        median = (median + results[reps / 2 - 1]) / 2

    # find two-sided 95% confidence interval
    index = reps / 20
    ci = [results[index - 1], results[reps - index]]
    mean = sum(results) / reps
    std = math.sqrt(sum([(x-mean)**2 for x in results]) / (reps * (reps - 1.0)))

    if report:
        print "median   =", median
        print "mean     =", mean
        print "std      =", std
        print "CI (95%) =", ci

    return median, mean, std, ci


# BENCHMARK CODE (execute module)

if __name__ == "__main__":
    """ Report values computed by module and benchmark results """

    print "CORE.PY BENCHMARK"

    # report paths
    print "\n[PATHS]"
    print "\n".join(sys.path)

    # discover tic() - toc() resolution
    print "\n[TIME RESOLUTION]"
    median = time_resolution(report = True)

    # perform a complexity experiment as an example
    print "\n[CREATION OF LIST SCALES LIKE THAT...]"
    print "{0:<13}{1:^13}{2:>13}".format('N', 'average', 'std.err. (%)')
    for n in [10, 100, 1000, 10000, 100000]:
        m, s = time_it("[x for x in range(" + str(n) + ")]", threshold = 0.01, runs = 30)
        print "{0:<13d}{1:^13.2e}{2:>13.1f}".format(n, m, s / m * 100)
