import sys      # for sys.path
import time     # for time()
import math     # for sqrt()


# default values for function parameters

_DEFAULT_TIMEIT_THRESHOLD = 0.1
_DEFAULT_TIME_RESOLUTION_REPS = 1000


# inclusion of the parent directories of the modules to the system path

sys.path = ['../constants', \
            '../SF_history', \
            '../binary', \
            '../stats', \
            '../notebooks', \
            '../pop_synth' \
           ] + sys.path


# import some common modules

import constants


# timing functions

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


def time_it(expr, threshold = _DEFAULT_TIMEIT_THRESHOLD):
    """ Returns the execution time of an expression. If the result is smaller
        than a threshold, it repeats until the total duration is larger and
        returns the average.

        Parameters
        ----------
        expr:       string
                    python expression to be timed, e.g. a function call

        threshold:  float
                    threshold for total duration

        Returns
        -------
        duration:   float
                    the execution time (or the average) of 'expr'
    """
    calls = 0                   # count how many timings were used
    duration = 0                # total duration
    tic()                       # start clock
    while duration < threshold:
        exec(expr)              # execute expression
        calls += 1
        duration = toc()
    return duration / calls     # return the average


def time_resolution(reps = _DEFAULT_TIME_RESOLUTION_REPS, report = False):
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


# benchmark code

if __name__ == "__main__":
    """ Report values computed by module and benchmark results """

    print "CORE.PY BENCHMARK"

    print "\n[PATHS]"
    print "\n".join(sys.path)

    print "\n[TIME RESOLUTION]"
    median = time_resolution(report = True)

    print "\n[CREATION OF LIST SCALES LIKE THAT...]"
    for n in [1000, 10000, 100000, 1000000]:
        print "For N = %10d, %1.4e" % (n, time_it("[x for x in range(" + str(n) + ")]"))
