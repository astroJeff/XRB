from xrb.src.core import *
import matplotlib.pyplot as plt
import scipy.optimize as so

def find_confidence_interval(x, pdf, confidence_level):
    return pdf[pdf > x].sum() - confidence_level

def density_contour(xdata, ydata, nbins_x, nbins_y, ax=None, sigma=False, **contour_kwargs):
    """ Create a density contour plot.
    Parameters
    ----------
    xdata : numpy.ndarray
    ydata : numpy.ndarray
    nbins_x : int
        Number of bins along x dimension
    nbins_y : int
        Number of bins along y dimension
    ax : matplotlib.Axes (optional)
        If supplied, plot the contour to this axis. Otherwise, open a new figure
    sigma : bool
	Use 1, 2, and 3-sigma confidence levels as contours, quantiles otherwise
    contour_kwargs : dict
        kwargs to be passed to pyplot.contour()
    """

    H, xedges, yedges = np.histogram2d(xdata, ydata, bins=(nbins_x,nbins_y), normed=True)
    x_bin_sizes = (xedges[1:] - xedges[:-1]).reshape((1,nbins_x))
    y_bin_sizes = (yedges[1:] - yedges[:-1]).reshape((nbins_y,1))

    pdf = (H*(x_bin_sizes*y_bin_sizes))

    if sigma:
       one_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.68))
       two_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.95))
       three_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.99))
       levels = [one_sigma, two_sigma, three_sigma]
    else:
       one_quarter = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.25))
       two_quarter = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.50))
       three_quarter = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.75))
       levels = [one_quarter, two_quarter, three_quarter]


    X, Y = 0.5*(xedges[1:]+xedges[:-1]), 0.5*(yedges[1:]+yedges[:-1])
    Z = pdf.T

    if ax == None:
        contour = plt.contour(X, Y, Z, levels=levels[::-1], origin="lower", **contour_kwargs)
    else:
        contour = ax.contour(X, Y, Z, levels=levels[::-1], origin="lower", **contour_kwargs)

    return contour

def test_density_contour():
    norm = np.random.normal(10., 15., size=(12540035, 2))
    density_contour(norm[:,0], norm[:,1], 100, 100)
    plt.show()
