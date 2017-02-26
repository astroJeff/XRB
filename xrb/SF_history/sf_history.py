from xrb.src.core import *
from xrb.SF_history import *

import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import units as u
from scipy.interpolate import interp1d
import scipy.optimize as so




ra_max = None
ra_min = None
dec_max = None
dec_min = None

sf_sfh = None
sf_coor = None
sf_dist = None


def get_theta_proj_degree(ra, dec, ra_b, dec_b):
    """ Return angular distance between two points

    Parameters
    ----------
    ra : float64
        Right ascension of first coordinate (degrees)
    dec : float64
        Declination of first coordinate (degrees)
    ra_b : float64
        Right ascension of second coordinate (degrees)
    dec_b : float64
        Declination of second coordinate (degrees)

    Returns
    -------
    theta : float64
        Angular distance (radians)
    """

    ra1 = c.deg_to_rad * ra
    dec1 = c.deg_to_rad * dec
    ra2 = c.deg_to_rad * ra_b
    dec2 = c.deg_to_rad * dec_b

    return np.sqrt((ra1-ra2)**2 * np.cos(dec1)*np.cos(dec2) + (dec1-dec2)**2)

def get_dist_closest(ra, dec, coor):
    """ Returns the distance to the closest star formation history region
    Parameters
    ----------
    ra : float64 or ndarray
        (Individual or ndarray of) right ascensions (degrees)
    dec : float64 or ndarray
        (Individual or ndarray of) declinations (degrees)
    coor : ndarray
        Array of already loaded LMC or SMC region coordinates

    Returns
    -------
    dist : float
        Distance to closest star formation history region (degrees)
    """

    ra1 = c.deg_to_rad * ra
    dec1 = c.deg_to_rad * dec
    ra2 = c.deg_to_rad * coor["ra"]
    dec2 = c.deg_to_rad * coor["dec"]

    dist = np.sqrt((ra1-ra2)**2*np.cos(dec1)*np.cos(dec2) + (dec1-dec2)**2)
    index = np.argmin(dist)

    return c.rad_to_deg * dist[index]


def reset_sf_history():
    """ Clear the SF_history module variables """

    global sf_coor
    global sf_sfh
    global sf_dist

    global ra_min
    global ra_max
    global dec_min
    global dec_max

    ra_max = None
    ra_min = None
    dec_max = None
    dec_min = None

    sf_sfh = None
    sf_coor = None
    sf_dist = None


def load_sf_history(z=0.008):
    """ Load star formation history data for both SMC and LMC

    Parameters
    ----------
    z : float
        Metallicity of star formation history
        Default = 0.008
    """


    global sf_coor
    global sf_sfh
    global sf_dist

    global ra_min
    global ra_max
    global dec_min
    global dec_max

    if c.sf_scheme is None:
        print "You must provide a scheme for the star formation history"
        exit(-1)

    if (sf_coor is None) or (sf_sfh is None) or (sf_dist is None) or (ra_min is None):
        if c.sf_scheme is "SMC":
            sf_coor = load_smc_data.load_smc_coor()
            sf_sfh = load_smc_data.load_smc_sfh(z)
            sf_dist = c.dist_SMC
            pad = 0.2

        if c.sf_scheme is "LMC":
            sf_coor = load_lmc_data.load_lmc_coor()
            sf_sfh = load_lmc_data.load_lmc_sfh(z)
            sf_dist = c.dist_LMC
            pad = 0.2

        if c.sf_scheme is "NGC4244":
            sf_coor, sf_sfh = load_NGC4244_data.load_NGC4244_sfh()
            sf_dist = c.dist_NGC4244
            pad = 0.005

    # if ra_min is None: ra_min = min(sf_coor['ra'])-0.2
    # if ra_max is None: ra_max = max(sf_coor['ra'])+0.2
    # if dec_min is None: dec_min = min(sf_coor['dec'])-0.2
    # if dec_max is None: dec_max = max(sf_coor['dec'])+0.2

        ra_min = min(sf_coor['ra'])-pad
        ra_max = max(sf_coor['ra'])+pad
        dec_min = min(sf_coor['dec'])-pad
        dec_max = max(sf_coor['dec'])+pad



def get_SFH(ra, dec, t_b, coor, sfh):
    """ Returns the star formation rate in Msun/Myr for a sky position and age

    Parameters
    ----------
    ra : float64 or ndarray
        (Individual or ndarray of) right ascensions (degrees)
    dec : float64 or ndarray
        (Individual or ndarray of) declinations (degrees)
    t_b : float64 or ndarray
        (Individual or ndarray of) times (Myr)
    coor : ndarray
        Array of already loaded LMC or SMC region coordinates
    sfh : ndarray
        Array of star formation histories (1D interpolations) for each region
        in the LMC or SMC

    Returns
    -------
    SFH : float64 or ndarray
        Star formation history (Msun/Myr)
    """


    if (coor is None) or (sfh is None): load_sf_history()

    if isinstance(ra, np.ndarray):

        ra1, ra2 = np.meshgrid(c.deg_to_rad * ra, c.deg_to_rad * coor["ra"])
        dec1, dec2 = np.meshgrid(c.deg_to_rad * dec, c.deg_to_rad * coor["dec"])

        dist = np.sqrt((ra1-ra2)**2*np.cos(dec1)*np.cos(dec2) + (dec1-dec2)**2)
        indices = dist.argmin(axis=0)

        SFR = np.zeros(len(ra))

        for i in np.arange(len(indices)):

            if ra[i]>ra_min and ra[i]<ra_max and dec[i]>dec_min and dec[i]<dec_max:
                SFR[i] = sfh[indices[i]](np.log10(t_b[i]*1.0e6))
            # # If outside the SMC, set to zero
            # if ra[i]<ra_min or ra[i]>ra_max or dec[i]<dec_min or dec[i]>dec_max:
            #     SFR[i] = 0
            # else:
            #     SFR[i] = sfh[indices[i]](np.log10(t_b[i]*1.0e6))


        #     SFR[i] = sfh[indices[i]](np.log10(t_b[i]*1.0e6))
        #
        # SFR[ra<ra_min] = 0.0
        # SFR[ra>ra_max] = 0.0
        # SFR[dec<dec_min] = 0.0
        # SFR[dec>dec_max] = 0.0


        return SFR

    else:
        ra1 = c.deg_to_rad * ra
        dec1 = c.deg_to_rad * dec
        ra2 = c.deg_to_rad * coor["ra"]
        dec2 = c.deg_to_rad * coor["dec"]

        dist = np.sqrt((ra1-ra2)**2*np.cos(dec1)*np.cos(dec2) + (dec1-dec2)**2)

        # If outside the SMC, set to zero
        if ra<ra_min or ra>ra_max or dec<dec_min or dec>dec_max:
            return 0.0
        else:
            index = np.argmin(dist)
            return sfh[index](np.log10(t_b*1.0e6))








def test_LMC_SFH_plots():
    """ Crete an array of 12 plots that show the LMC's star formation history
    at 12 different times.
    """

    plt.figure(figsize=(12,15))


    global sf_coor
    global sf_sfh
    global sf_dist


    if (sf_coor is None) or (sf_sfh is None):
        c.sf_scheme = "LMC"
        load_sf_history(z=0.008)



    def get_LMC_plot(age):
        sfr = np.array([])
        for i in np.arange(len(sf_coor)):
            sfr = np.append(sfr, get_SFH(sf_coor["ra"][i], \
                            sf_coor["dec"][i], age, sf_coor, sf_sfh))

        plt.tricontourf(sf_coor["ra"], sf_coor["dec"], sfr)
        plt.title(str(int(age)) + ' Myr')
        plt.ylim(-73, -64)

        return plt


    plt.subplot(4,3,1)
    get_LMC_plot(7.0)

    plt.subplot(4,3,2)
    get_LMC_plot(10.0)

    plt.subplot(4,3,3)
    get_LMC_plot(15.0)

    plt.subplot(4,3,4)
    get_LMC_plot(20.0)

    plt.subplot(4,3,5)
    get_LMC_plot(25.0)

    plt.subplot(4,3,6)
    get_LMC_plot(30.0)

    plt.subplot(4,3,7)
    get_LMC_plot(40.0)

    plt.subplot(4,3,8)
    get_LMC_plot(60.0)

    plt.subplot(4,3,9)
    get_LMC_plot(80.0)

    plt.subplot(4,3,10)
    get_LMC_plot(100.0)

    plt.subplot(4,3,11)
    get_LMC_plot(150.0)

    plt.subplot(4,3,12)
    get_LMC_plot(200.0)

    plt.show()

def get_LMC_plot(age, ax=None):
    """ return a plot of the star formation history of the LMC at a particular age

    Parameters
    ----------
    age : float
        Star formation history age to calculate (Myr)
    ax : matplotlib.Axes (optional)
        If supplied, plot the contour to this axis. Otherwise, open a new figure


    Returns
    -------
    plt : matplotlib.pyplot plot
        Contour plot of the star formation history
    """


    global sf_coor
    global sf_sfh
    global sf_dist


    if (sf_coor is None) or (sf_sfh is None):
        c.sf_scheme = "LMC"
        load_sf_history()


    sfr = np.array([])
    for i in np.arange(len(sf_coor)):
        sfr = np.append(sfr, get_SFH(sf_coor["ra"][i], \
                        sf_coor["dec"][i], age, sf_coor, sf_sfh))

    if ax:
        lmc_plot = ax.tricontourf(sf_coor["ra"], sf_coor["dec"], sfr)
        lmc_plot = ax.set_title(str(int(age)) + ' Myr')
    else:
        lmc_plot = plt.tricontourf(sf_coor["ra"], sf_coor["dec"], sfr)
        lmc_plot = plt.title(str(int(age)) + ' Myr')
        lmc_plot = plt.gca().invert_xaxis()

    lmc_plot = plt.xlabel("Right Ascension (degrees)")
    lmc_plot = plt.ylabel("Declination (degrees)")

    return lmc_plot






def test_SMC_SFH_plots():
    """ Crete an array of 12 plots that show the LMC's star formation history
    at 12 different times.
    """

    global sf_coor
    global sf_sfh
    global sf_dist

    if (sf_coor is None) or (sf_sfh is None):
        c.sf_scheme = "SMC"
        load_sf_history(z=0.008)

    plt.figure(figsize=(12,15))


    def get_SMC_plot(age):
        sfr = np.array([])
        for i in np.arange(len(sf_coor)):
            sfr = np.append(sfr, get_SFH(sf_coor["ra"][i], \
                            sf_coor["dec"][i], age, sf_coor, sf_sfh))

        plt.tricontourf(sf_coor["ra"], sf_coor["dec"], sfr)
        plt.title(str(int(age)) + ' Myr')

        return plt


    plt.subplot(4,3,1)
    get_SMC_plot(20.0)

    plt.subplot(4,3,2)
    get_SMC_plot(40.0)

    plt.subplot(4,3,3)
    get_SMC_plot(60.0)

    plt.subplot(4,3,4)
    get_SMC_plot(100.0)

    plt.subplot(4,3,5)
    get_SMC_plot(160.0)

    plt.subplot(4,3,6)
    get_SMC_plot(250.0)

    plt.subplot(4,3,7)
    get_SMC_plot(400.0)

    plt.subplot(4,3,8)
    get_SMC_plot(600.0)

    plt.subplot(4,3,9)
    get_SMC_plot(1000.0)

    plt.subplot(4,3,10)
    get_SMC_plot(2500.0)

    plt.subplot(4,3,11)
    get_SMC_plot(4000.0)

    plt.subplot(4,3,12)
    get_SMC_plot(7000.0)

    plt.show()


def get_SMC_plot(age, ax=None):
    """ return a plot of the star formation history of the SMC at a particular age

    Parameters
    ----------
    age : float
        Star formation history age to calculate (Myr)
    ax : matplotlib.Axes (optional)
        If supplied, plot the contour to this axis. Otherwise, open a new figure

    Returns
    -------
    plt : matplotlib.pyplot plot
        Contour plot of the star formation history
    """

    global sf_coor
    global sf_sfh
    global sf_dist

    if (sf_coor is None) or (sf_sfh is None):
        c.sf_scheme = "SMC"
        load_sf_history(z=0.008)


    sfr = np.array([])
    for i in np.arange(len(sf_coor)):
        sfr = np.append(sfr, get_SFH(sf_coor["ra"][i], \
                        sf_coor["dec"][i], age, sf_coor, sf_sfh))

    if ax:
        smc_plot = ax.tricontourf(sf_coor["ra"], sf_coor["dec"], sfr)
        smc_plot = ax.set_title(str(int(age)) + ' Myr')
    else:
        smc_plot = plt.tricontourf(sf_coor["ra"], sf_coor["dec"], sfr)
        smc_plot = plt.title(str(int(age)) + ' Myr')
        smc_plot = plt.gca().invert_xaxis()

    smc_plot = plt.xlabel("Right Ascension (degrees)")
    smc_plot = plt.ylabel("Declination (degrees)")

    return smc_plot


def get_SMC_plot_polar(age, fig_in=None, ax=None, gs=None, ra_dist=None, dec_dist=None,
        dist_bins=25, ra=None, dec=None, xcenter=0.0, ycenter=17.3, xwidth=1.5, ywidth=1.5,
        xlabel="Right Ascension", ylabel="Declination", xgrid_density=8, ygrid_density=5,
        color_map='Blues'):
    """ return a plot of the star formation history of the SMC at a particular age.
    In this case, the plot should be curvelinear, instead of flattened.

    Parameters
    ----------
    age : float
        Star formation history age to calculate (Myr)
    fig : matplotlib.figure (optional)
        If supplied, plot the contour to this axis. Otherwise, open a new figure
    rect : int
        Subplot number
    gs : gridspec object (optional)
        If supplied, plot goes inside gridspec object provided
    ra_dist, dec_dist : array (optional)
        If supplied, plots contours around the distribution of these inputs
    dist_bins : int (optional)
        Number of bins for ra_dist-dec_dist contours
    ra, dec : float (optional)
        If supplied, plot a red star at these coordinates (degrees)
    xcenter, ycenter : float (optional)
        If supplied, center the x,y-axis on these coordinates
    xwidth, ywidth : float (optional)
        If supplied, determines the scale of the plot
    xlabel, ylabel : string (optional)
        X-axis, y-axis label
    xgrid_density, ygrid_density : int (optional)
        Density of RA, Dec grid axes
    color_map : string (optional)
        One of the color map options from plt.cmap

    Returns
    -------
    plt : matplotlib.pyplot plot
        Contour plot of the star formation history
    """

    import mpl_toolkits.axisartist.angle_helper as angle_helper
    from matplotlib.projections import PolarAxes
    from matplotlib.transforms import Affine2D
    from mpl_toolkits.axisartist import SubplotHost
    from mpl_toolkits.axisartist import GridHelperCurveLinear
    import matplotlib.gridspec as gridspec

    global sf_coor
    global sf_sfh
    global sf_dist

    if (sf_coor is None) or (sf_sfh is None):
        c.sf_scheme = "SMC"
        load_sf_history(z=0.008)


    def curvelinear_test2(fig, gs=None, xcenter=0.0, ycenter=17.3, xwidth=1.5, ywidth=1.5,
            xlabel=xlabel, ylabel=ylabel, xgrid_density=8, ygrid_density=5):
        """
        polar projection, but in a rectangular box.
        """

        tr = Affine2D().translate(0,90)
        tr += Affine2D().scale(np.pi/180., 1.)
        tr += PolarAxes.PolarTransform()
        tr += Affine2D().rotate(1.34)  # This rotates the grid

        extreme_finder = angle_helper.ExtremeFinderCycle(10, 60,
                                                        lon_cycle = 360,
                                                        lat_cycle = None,
                                                        lon_minmax = None,
                                                        lat_minmax = (-90, np.inf),
                                                        )

        grid_locator1 = angle_helper.LocatorHMS(xgrid_density) #changes theta gridline count
        tick_formatter1 = angle_helper.FormatterHMS()
        grid_locator2 = angle_helper.LocatorDMS(ygrid_density) #changes theta gridline count
        tick_formatter2 = angle_helper.FormatterDMS()


        grid_helper = GridHelperCurveLinear(tr,
                                            extreme_finder=extreme_finder,
                                            grid_locator1=grid_locator1,
                                            grid_locator2=grid_locator2,
                                            tick_formatter1=tick_formatter1,
                                            tick_formatter2=tick_formatter2
                                            )

        # ax1 = SubplotHost(fig, rect, grid_helper=grid_helper)
        if gs is None:
            ax1 = SubplotHost(fig, 111, grid_helper=grid_helper)
        else:
            ax1 = SubplotHost(fig, gs, grid_helper=grid_helper)



        # make ticklabels of right and top axis visible.
        ax1.axis["right"].major_ticklabels.set_visible(False)
        ax1.axis["top"].major_ticklabels.set_visible(False)
        ax1.axis["bottom"].major_ticklabels.set_visible(True) #Turn off?

        # let right and bottom axis show ticklabels for 1st coordinate (angle)
        ax1.axis["right"].get_helper().nth_coord_ticks=0
        ax1.axis["bottom"].get_helper().nth_coord_ticks=0


        fig.add_subplot(ax1)

        grid_helper = ax1.get_grid_helper()

        # These move the grid
        ax1.set_xlim(xcenter-xwidth, xcenter+xwidth) # moves the origin left-right in ax1
        ax1.set_ylim(ycenter-ywidth, ycenter+ywidth) # moves the origin up-down
        # ax1.set_xlim(-1.5, 1.4)
        # ax1.set_ylim(15.8, 18.8)

        if xlabel is not None:
            ax1.set_xlabel(xlabel)
        if ylabel is not None:
            ax1.set_ylabel(ylabel)
        # ax1.set_ylabel('Declination')
        # ax1.set_xlabel('Right Ascension')
        ax1.grid(True, linestyle='-')
        #ax1.grid(linestyle='--', which='x') # either keyword applies to both
        #ax1.grid(linestyle=':', which='y')  # sets of gridlines


        return ax1,tr


    # User supplied input
    if fig_in is None:
        fig = plt.figure(1, figsize=(8, 6))
        fig.clf()
    else:
        fig = fig_in

    # tr.transform_point((x, 0)) is always (0,0)
            # => (theta, r) in but (r, theta) out...
    ax1, tr = curvelinear_test2(fig, gs, xcenter=xcenter, ycenter=ycenter,
                    xwidth=xwidth, ywidth=ywidth, xlabel=xlabel, ylabel=ylabel,
                    xgrid_density=xgrid_density, ygrid_density=ygrid_density)


    sfr = np.array([])


    # CREATING OUR OWN, LARGER GRID FOR STAR FORMATION CONTOURS
    x_tmp = np.linspace(min(sf_coor['ra'])-1.0, max(sf_coor['ra'])+1.0, 30)
    y_tmp = np.linspace(min(sf_coor['dec'])-1.0, max(sf_coor['dec'])+1.0, 30)

    XX, YY = np.meshgrid(x_tmp, y_tmp)

    for i in np.arange(len(XX.flatten())):
        sfr = np.append(sfr, get_SFH(XX.flatten()[i], \
                        YY.flatten()[i], age, sf_coor, sf_sfh))
    out_test = tr.transform(zip(XX.flatten(), YY.flatten()))

    # USING smc_coor AS THE POINTS FOR STAR FORMATION CONTOURS
    # for i in np.arange(len(smc_coor)):
    #     sfr = np.append(sfr, get_SFH(smc_coor["ra"][i], \
    #                     smc_coor["dec"][i], age, smc_coor, smc_sfh))

    # Apply transformation to smc_coor ra and dec
    # out_test = tr.transform(zip(smc_coor["ra"], smc_coor["dec"]))





    # Plot star formation histories on adjusted coordinates
    # Plot color contours with linear spacing
    #levels = np.arange(1.0e8, 1.0e9, 1.0e8)
    levels = np.linspace(1.0e7, 1.0e9, 10)
    smc_plot = plt.tricontourf(out_test[:,0], out_test[:,1], sfr, cmap=color_map, levels=levels, extend='max')
    # Plot color contours with logarithmic spacing
    # levels = np.linspace(7.0, 10.0, 10)
    # smc_plot = plt.tricontourf(out_test[:,0], out_test[:,1], np.log10(sfr), cmap=color_map, levels=levels, extend='max')
    smc_plot = plt.title(str(int(age)) + ' Myr')
    # smc_plot = plt.colorbar()


    # Plot the contours defining the distributions of ra_dist and dec_dist
    if ra_dist is not None and dec_dist is not None:

        # Need this function
        def find_confidence_interval(x, pdf, confidence_level):
            return pdf[pdf > x].sum() - confidence_level

        # Transform distribution
        coor_dist_polar = tr.transform(zip(ra_dist, dec_dist))

        # Create 2D histogram
        nbins_x = dist_bins
        nbins_y = dist_bins
        H, xedges, yedges = np.histogram2d(coor_dist_polar[:,0], coor_dist_polar[:,1], bins=(nbins_x,nbins_y), normed=True)
        x_bin_sizes = (xedges[1:] - xedges[:-1]).reshape((1,nbins_x))
        y_bin_sizes = (yedges[1:] - yedges[:-1]).reshape((nbins_y,1))
        pdf = (H*(x_bin_sizes*y_bin_sizes))

        # Find intervals
        one_quad = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.25))
        two_quad = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.50))
        three_quad = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.75))
        levels = [one_quad, two_quad, three_quad]
        X, Y = 0.5*(xedges[1:]+xedges[:-1]), 0.5*(yedges[1:]+yedges[:-1])
        Z = pdf.T

        # Plot contours
        contour = plt.contour(X, Y, Z, levels=levels[::-1], origin="lower", colors=['k'])
        #contour = plt.contour(X, Y, Z, levels=levels[::-1], origin="lower", colors=['r','g','b'])

        # To change linewidths
        zc = contour.collections
        plt.setp(zc, linewidth=1.5)

    # Plot a star at the coordinate position, if supplied
    if ra is not None and dec is not None:
        coor_pol1, coor_pol2 = tr.transform(zip(np.array([ra, ra]), np.array([dec, dec])))
        smc_plot = plt.scatter(coor_pol1[0], coor_pol1[1], color='r', s=75, marker="*", zorder=10)



    return smc_plot, ax1



def get_plot_polar(age, fig_in=None, ax=None, gs=None, ra_dist=None, dec_dist=None,
        dist_bins=25, ra=None, dec=None, xcenter=None, ycenter=None, xwidth=None, ywidth=None,
        xlabel="Right Ascension", ylabel="Declination", xgrid_density=8, ygrid_density=5,
        color_map='Blues', title=None):
    """ return a plot of the star formation history of the SMC at a particular age.
    In this case, the plot should be curvelinear, instead of flattened.

    Parameters
    ----------
    age : float
        Star formation history age to calculate (Myr)
    fig : matplotlib.figure (optional)
        If supplied, plot the contour to this axis. Otherwise, open a new figure
    rect : int
        Subplot number
    gs : gridspec object (optional)
        If supplied, plot goes inside gridspec object provided
    ra_dist, dec_dist : array (optional)
        If supplied, plots contours around the distribution of these inputs
    dist_bins : int (optional)
        Number of bins for ra_dist-dec_dist contours
    ra, dec : float (optional)
        If supplied, plot a red star at these coordinates (degrees)
    xcenter, ycenter : float (optional)
        If supplied, center the x,y-axis on these coordinates
    xwidth, ywidth : float (optional)
        If supplied, determines the scale of the plot
    xlabel, ylabel : string (optional)
        X-axis, y-axis label
    xgrid_density, ygrid_density : int (optional)
        Density of RA, Dec grid axes
    color_map : string (optional)
        One of the color map options from plt.cmap
    title : string
        Add a title to the plot. Default is the age.

    Returns
    -------
    plt : matplotlib.pyplot plot
        Contour plot of the star formation history
    """

    import mpl_toolkits.axisartist.angle_helper as angle_helper
    from matplotlib.projections import PolarAxes
    from matplotlib.transforms import Affine2D
    from mpl_toolkits.axisartist import SubplotHost
    from mpl_toolkits.axisartist import GridHelperCurveLinear
    import matplotlib.gridspec as gridspec

    global sf_coor
    global sf_sfh
    global sf_dist

    global ra_min
    global ra_max
    global dec_min
    global dec_max

    if c.sf_scheme is None:
        c.sf_scheme = "SMC"

    if (sf_coor is None) or (sf_sfh is None):
        load_sf_history(z=0.008)

    if c.sf_scheme == "SMC":
        if xcenter is None: xcenter=0.0
        if ycenter is None: ycenter=17.3
        if xwidth is None: xwidth=1.5
        if ywidth is None: ywidth=1.5

    if c.sf_scheme == "LMC":
        if xcenter is None: xcenter=0.0
        if ycenter is None: ycenter=21.0
        if xwidth is None: xwidth=5.0
        if ywidth is None: ywidth=5.0

    if c.sf_scheme == 'NGC4244':
        if xcenter is None: xcenter = 0.0
        if ycenter is None: ycenter = 127.8
        if xwidth is None: xwidth = 0.3
        if ywidth is None: ywidth = 0.1


    def curvelinear_test2(fig, gs=None, xcenter=0.0, ycenter=17.3, xwidth=1.5, ywidth=1.5,
            xlabel=xlabel, ylabel=ylabel, xgrid_density=8, ygrid_density=5):
        """
        polar projection, but in a rectangular box.
        """

        tr = Affine2D().translate(0,90)
        tr += Affine2D().scale(np.pi/180., 1.)
        tr += PolarAxes.PolarTransform()
        if c.sf_scheme == "SMC":
            rot_angle = 1.34
        if c.sf_scheme == "LMC":
            rot_angle = 0.2
        if c.sf_scheme == "NGC4244":
            rot_angle = 4.636

        tr += Affine2D().rotate(rot_angle)  # This rotates the grid

        extreme_finder = angle_helper.ExtremeFinderCycle(10, 60,
                                                        lon_cycle = 360,
                                                        lat_cycle = None,
                                                        lon_minmax = None,
                                                        lat_minmax = (-90, np.inf),
                                                        )

        grid_locator1 = angle_helper.LocatorHMS(xgrid_density) #changes theta gridline count
        tick_formatter1 = angle_helper.FormatterHMS()
        grid_locator2 = angle_helper.LocatorDMS(ygrid_density) #changes theta gridline count
        tick_formatter2 = angle_helper.FormatterDMS()


        grid_helper = GridHelperCurveLinear(tr,
                                            extreme_finder=extreme_finder,
                                            grid_locator1=grid_locator1,
                                            grid_locator2=grid_locator2,
                                            tick_formatter1=tick_formatter1,
                                            tick_formatter2=tick_formatter2
                                            )

        # ax1 = SubplotHost(fig, rect, grid_helper=grid_helper)
        if gs is None:
            ax1 = SubplotHost(fig, 111, grid_helper=grid_helper)
        else:
            ax1 = SubplotHost(fig, gs, grid_helper=grid_helper)



        # make ticklabels of right and top axis visible.
        ax1.axis["right"].major_ticklabels.set_visible(False)
        ax1.axis["top"].major_ticklabels.set_visible(False)
        ax1.axis["bottom"].major_ticklabels.set_visible(True) #Turn off?

        # let right and bottom axis show ticklabels for 1st coordinate (angle)
        ax1.axis["right"].get_helper().nth_coord_ticks=0
        ax1.axis["bottom"].get_helper().nth_coord_ticks=0


        fig.add_subplot(ax1)

        grid_helper = ax1.get_grid_helper()

        # These move the grid
        ax1.set_xlim(xcenter-xwidth, xcenter+xwidth) # moves the origin left-right in ax1
        ax1.set_ylim(ycenter-ywidth, ycenter+ywidth) # moves the origin up-down
        # ax1.set_xlim(-1.5, 1.4)
        # ax1.set_ylim(15.8, 18.8)

        if xlabel is not None:
            ax1.set_xlabel(xlabel)
        if ylabel is not None:
            ax1.set_ylabel(ylabel)
        # ax1.set_ylabel('Declination')
        # ax1.set_xlabel('Right Ascension')
        ax1.grid(True, linestyle='-')
        #ax1.grid(linestyle='--', which='x') # either keyword applies to both
        #ax1.grid(linestyle=':', which='y')  # sets of gridlines


        return ax1,tr


    # User supplied input
    if fig_in is None:
        fig = plt.figure(1, figsize=(8, 6))
        fig.clf()
    else:
        fig = fig_in

    # tr.transform_point((x, 0)) is always (0,0)
            # => (theta, r) in but (r, theta) out...
    ax1, tr = curvelinear_test2(fig, gs, xcenter=xcenter, ycenter=ycenter,
                    xwidth=xwidth, ywidth=ywidth, xlabel=xlabel, ylabel=ylabel,
                    xgrid_density=xgrid_density, ygrid_density=ygrid_density)


    sfr = np.array([])


    if c.sf_scheme == "SMC":
        levels = np.linspace(1.0e7, 1.0e9, 10)
        bins = 30
    if c.sf_scheme == "LMC":
        levels = np.linspace(1.0e7, 2.0e8, 10)
        bins = 30
    if c.sf_scheme == "NGC4244":
        levels = np.linspace(0.1, 25, 25)
        bins = 80


    # CREATING OUR OWN, LARGER GRID FOR STAR FORMATION CONTOURS
    # x_tmp = np.linspace(min(sf_coor['ra'])-1.0, max(sf_coor['ra'])+1.0, 30)
    # y_tmp = np.linspace(min(sf_coor['dec'])-1.0, max(sf_coor['dec'])+1.0, 30)
    x_tmp = np.linspace(ra_min, ra_max, bins)
    y_tmp = np.linspace(dec_min, dec_max, bins)


    XX, YY = np.meshgrid(x_tmp, y_tmp)

    for i in np.arange(len(XX.flatten())):
        sfr = np.append(sfr, get_SFH(XX.flatten()[i], \
                        YY.flatten()[i], age, sf_coor, sf_sfh))
    out_test = tr.transform(zip(XX.flatten(), YY.flatten()))


    # USING smc_coor AS THE POINTS FOR STAR FORMATION CONTOURS
    # for i in np.arange(len(smc_coor)):
    #     sfr = np.append(sfr, get_SFH(smc_coor["ra"][i], \
    #                     smc_coor["dec"][i], age, smc_coor, smc_sfh))

    # Apply transformation to smc_coor ra and dec
    # out_test = tr.transform(zip(smc_coor["ra"], smc_coor["dec"]))





    # Plot star formation histories on adjusted coordinates
    # Plot color contours with linear spacing
    #levels = np.arange(1.0e8, 1.0e9, 1.0e8)

    sf_plot = plt.tricontourf(out_test[:,0], out_test[:,1], sfr, cmap=color_map, levels=levels, extend='max')
    # sf_plot = plt.tricontourf(out_test[:,0], out_test[:,1], sfr, cmap=color_map, extend='max')
    # sf_plot = plt.colorbar()
    # Plot color contours with logarithmic spacing
    # levels = np.linspace(7.0, 10.0, 10)
    # smc_plot = plt.tricontourf(out_test[:,0], out_test[:,1], np.log10(sfr), cmap=color_map, levels=levels, extend='max')
    if title is None:
        sf_plot = plt.title(str(int(age)) + ' Myr')
    else:
        sf_plot = plt.title(title)
    # smc_plot = plt.colorbar()


    # Plot the contours defining the distributions of ra_dist and dec_dist
    if ra_dist is not None and dec_dist is not None:

        # Need this function
        def find_confidence_interval(x, pdf, confidence_level):
            return pdf[pdf > x].sum() - confidence_level

        # Transform distribution
        coor_dist_polar = tr.transform(zip(ra_dist, dec_dist))

        # Create 2D histogram
        nbins_x = dist_bins
        nbins_y = dist_bins
        H, xedges, yedges = np.histogram2d(coor_dist_polar[:,0], coor_dist_polar[:,1], bins=(nbins_x,nbins_y), normed=True)
        x_bin_sizes = (xedges[1:] - xedges[:-1]).reshape((1,nbins_x))
        y_bin_sizes = (yedges[1:] - yedges[:-1]).reshape((nbins_y,1))
        pdf = (H*(x_bin_sizes*y_bin_sizes))

        # Find intervals
        one_quad = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.25))
        two_quad = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.50))
        three_quad = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.75))
        levels = [one_quad, two_quad, three_quad]
        X, Y = 0.5*(xedges[1:]+xedges[:-1]), 0.5*(yedges[1:]+yedges[:-1])
        Z = pdf.T

        # Plot contours
        contour = plt.contour(X, Y, Z, levels=levels[::-1], origin="lower", colors=['k'])
        #contour = plt.contour(X, Y, Z, levels=levels[::-1], origin="lower", colors=['r','g','b'])

        # To change linewidths
        zc = contour.collections
        plt.setp(zc, linewidth=1.5)

    # Plot a star at the coordinate position, if supplied
    if ra is not None and dec is not None:
        coor_pol1, coor_pol2 = tr.transform(zip(np.array([ra, ra]), np.array([dec, dec])))
        sf_plot = plt.scatter(coor_pol1[0], coor_pol1[1], color='r', s=75, marker="*", zorder=10)



    return sf_plot, ax1
