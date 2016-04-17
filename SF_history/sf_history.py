import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import units as u
from scipy.interpolate import interp1d

lmc_sfh = None
lmc_coor = None
smc_sfh = None
smc_coor = None



def deg_to_rad(theta):
    """ Convert from degrees to radians """
    return np.pi * theta / 180.0

def rad_to_deg(theta):
    """ Convert from radians to degrees """
    return 180.0 * theta / np.pi

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

    ra1 = deg_to_rad(ra)
    dec1 = deg_to_rad(dec)
    ra2 = deg_to_rad(ra_b)
    dec2 = deg_to_rad(dec_b)

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

    ra1 = deg_to_rad(ra)
    dec1 = deg_to_rad(dec)
    ra2 = deg_to_rad(coor["ra"])
    dec2 = deg_to_rad(coor["dec"])

    dist = np.sqrt((ra1-ra2)**2*np.cos(dec1)*np.cos(dec2) + (dec1-dec2)**2)
    index = np.argmin(dist)

    return rad_to_deg(dist[index])


def load_sf_history(z=0.008):
    """ Load star formation history data for both SMC and LMC

    Parameters
    ----------
    z : float
        Metallicity of star formation history
        Default = 0.008
    """

    if lmc_coor is None: load_lmc_coor()
    if lmc_sfh is None: load_lmc_sfh(z)
    if smc_coor is None: load_smc_coor()
    if smc_sfh is None: load_smc_sfh(z)


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

    if isinstance(ra, np.ndarray):

        ra1, ra2 = np.meshgrid(deg_to_rad(ra), deg_to_rad(coor["ra"]))
        dec1, dec2 = np.meshgrid(deg_to_rad(dec), deg_to_rad(coor["dec"]))

        dist = np.sqrt((ra1-ra2)**2*np.cos(dec1)*np.cos(dec2) + (dec1-dec2)**2)
        indices = dist.argmin(axis=0)

        SFR = np.zeros(len(ra))
        for i in np.arange(len(indices)):
            SFR[i] = sfh[indices[i]](np.log10(t_b[i]*1.0e6))

        return SFR

    else:
        ra1 = deg_to_rad(ra)
        dec1 = deg_to_rad(dec)
        ra2 = deg_to_rad(coor["ra"])
        dec2 = deg_to_rad(coor["dec"])

        dist = np.sqrt((ra1-ra2)**2*np.cos(dec1)*np.cos(dec2) + (dec1-dec2)**2)
        index = np.argmin(dist)

        return sfh[index](np.log10(t_b*1.0e6))



def load_lmc_data():
    """ Return LMC star formation history per unit steradian

    Returns
    -------
    lmc_sfh : np structured array
        LMC star formation history
        dtype: [('region','<S10'),
                ('log_age','<f8'),
                ('sfh_z008','<f8'),
                ('sfh_z004','<f8'),
                ('sfh_z0025','<f8'),
                ('sfh_z001','<f8')]
    """

    # Create an empty array to start with
    dtypes = [('region','<S10'), \
            ('log_age','<f8'), \
            ('sfh_z008','<f8'), \
            ('sfh_z004','<f8'), \
            ('sfh_z0025','<f8'), \
            ('sfh_z001','<f8')]
    lmc_data = np.recarray(0, dtype=dtypes)
    out_line = np.recarray(1, dtype=dtypes)

    # Test to load data
    this_dir, this_filename = os.path.split(__file__)
    file_path = os.path.join(this_dir, "lmc_sfh_reduced.dat")

    with open(file_path) as f:
#    with open("./lmc_sfh_reduced.dat") as f:
        line_num = 0

        for line in f:
            line_num += 1

            if line_num < 17: continue
            if "Region" in line:
                region = np.array(line.split()[2]).astype(np.str)
            elif "(" in line:
                1 == 1
            else:
                line_data = line.split()
                line_data = np.array(line_data).astype(np.float64)

                if "_" in str(region):
                    area = 1.218e-5
                else:
                    area = 4.874e-5

                out_line[0][0] = region
                out_line[0][1] = line_data[0]
                out_line[0][2] = line_data[1] / area
                out_line[0][3] = line_data[4] / area
                out_line[0][4] = line_data[7] / area
                out_line[0][5] = line_data[10] / area

                lmc_data = np.append(lmc_data, out_line[0])

    return lmc_data


def load_lmc_coor():
    """ Load coordinates to LMC regions

    Returns
    -------
    lmc_coor: np structured array
        Coordinates of LMC regions in degrees
        dtype: [('region','<S10'),
                ('ra','float64'),
                ('dec','float64')]
    """

    global lmc_coor

    # If already loaded, no need to reload
    if lmc_coor is not None: return lmc_coor

    # Load data
    this_dir, this_filename = os.path.split(__file__)
    data_file = os.path.join(this_dir, "lmc_coordinates.dat")

    lmc_coor_2 = np.genfromtxt(data_file, dtype="S10,S2,S2,S3,S2")

    lmc_coor = np.recarray(0, dtype=[('region','<S10'),('ra','float64'),('dec','float64')])
    tmp = np.recarray(1, dtype=[('region','<S10'),('ra','float64'),('dec','float64')])


    for coor in lmc_coor_2:
        ra = str(coor[1])+"h"+str(coor[2])+"m"
        dec = str(coor[3])+"d"+str(coor[4])+"m"

        region = coor[0]

        coor = SkyCoord(ra, dec)

        tmp["region"] = region
        tmp["ra"] = coor.ra.degree
        tmp["dec"] = coor.dec.degree

        lmc_coor = np.append(lmc_coor, tmp)

    return lmc_coor




def load_lmc_sfh(z=0.008):
    """ Create array of 1D interpolations in time of the
    star formation histories for each region in the LMC.

    Parameters
    ----------
    z : float (0.001, 0.0025, 0.004, 0.008)
        Metallicity for which to return star formation history
        Default = 0.008

    Returns
    -------
    SF_history : ndarray
        Array of star formation histories for each region
    """

    global lmc_sfh

    # If already loaded, no need to reload
    if lmc_sfh is not None: return lmc_sfh

    # Load the LMC coordinates and SFH data
    lmc_data = load_lmc_data()

    regions = np.unique(lmc_data["region"])

    lmc_sfh = np.array([])
    age = np.array([])
    sfr = np.array([])
    for r in regions:

        age = lmc_data["log_age"][np.where(lmc_data["region"] == r)]

        if z == 0.008:
            sfr = lmc_data["sfh_z008"][np.where(lmc_data["region"] == r)]
        elif z == 0.004:
            sfr = lmc_data["sfh_z004"][np.where(lmc_data["region"] == r)]
        elif z == 0.0025:
            sfr = lmc_data["sfh_z0025"][np.where(lmc_data["region"] == r)]
        elif z == 0.001:
            sfr = lmc_data["sfh_z001"][np.where(lmc_data["region"] == r)]
        else:
            print "ERROR: You must choose an appropriate metallicity input"
            print "Possible options are 0.001, 0.0025, 0.004, 0.008"
            return -1

        lmc_sfh = np.append(lmc_sfh, interp1d(age[::-1], sfr[::-1], bounds_error=False, fill_value=0.0))

    return lmc_sfh



def test_LMC_SFH_plots():
    """ Crete an array of 12 plots that show the LMC's star formation history
    at 12 different times.
    """

    plt.figure(figsize=(12,15))

    # Load LMC data
    lmc_coor = load_lmc_coor()
    lmc_sfh = load_lmc_sfh()


    def get_LMC_plot(age):
        sfr = np.array([])
        for i in np.arange(len(lmc_coor)):
            sfr = np.append(sfr, get_SFH(lmc_coor["ra"][i], \
                            lmc_coor["dec"][i], age, lmc_coor, lmc_sfh))

        plt.tricontourf(lmc_coor["ra"], lmc_coor["dec"], sfr)
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

def get_LMC_plot(age):
    """ return a plot of the star formation history of the LMC at a particular age

    Parameters
    ----------
    age : float
        Star formation history age to calculate (Myr)

    Returns
    -------
    plt : matplotlib.pyplot plot
        Contour plot of the star formation history
    """

    if lmc_coor is None: load_lmc_coor()
    if lmc_sfh is None: load_lmc_sfh()

    sfr = np.array([])
    for i in np.arange(len(lmc_coor)):
        sfr = np.append(sfr, get_SFH(lmc_coor["ra"][i], \
                        lmc_coor["dec"][i], age, lmc_coor, lmc_sfh))

    plt.tricontourf(lmc_coor["ra"], lmc_coor["dec"], sfr)
    plt.title(str(int(age)) + ' Myr')

    return plt




def load_smc_coor():
    """ Load coordinates to SMC regions

    Returns
    -------
    smc_coor: np structured array
        Coordinates of SMC regions in degrees
        dtype: [('region','<S10'),
                ('ra','float64'),
                ('dec','float64')]
    """

    global smc_coor

    # If already loaded, no need to reload
    if smc_coor is not None: return smc_coor

    # Load data
    this_dir, this_filename = os.path.split(__file__)
    data_file = os.path.join(this_dir, "smc_coordinates.dat")

    smc_coor_2 = np.genfromtxt(data_file, dtype="S10,S2,S2,S3,S2")

    smc_coor = np.recarray(0, dtype=[('region','<S10'),('ra','float64'),('dec','float64')])
    tmp = np.recarray(1, dtype=[('region','<S10'),('ra','float64'),('dec','float64')])


    for coor in smc_coor_2:
        ra = str(coor[1])+"h"+str(coor[2])+"m"
        dec = str(coor[3])+"d"+str(coor[4])+"m"

        region = coor[0]

        coor = SkyCoord(ra, dec)

        tmp["region"] = region
        tmp["ra"] = coor.ra.degree
        tmp["dec"] = coor.dec.degree

        smc_coor = np.append(smc_coor, tmp)

    return smc_coor


def load_smc_data():
    """ Return SMC star formation history per unit steradian

    Returns
    -------
    smc_sfh : np structured array
        SMC star formation history
        dtype: [('region','<S10'),
                ('log_age','<f8'),
                ('sfh_z008','<f8'),
                ('sfh_z004','<f8'),
                ('sfh_z001','<f8')]
    """

    # Create an empty array to start with
    dtypes = [('region','<S10'), \
            ('log_age','<f8'), \
            ('sfh_z008','<f8'), \
            ('sfh_z004','<f8'), \
            ('sfh_z001','<f8')]

    smc_data = np.recarray(0, dtype=dtypes)
    out_data = np.recarray(1, dtype=dtypes)

    smc_coor = load_smc_coor()

    # Each region has an area of 12' x 12', or 1.218e-5 steradians
    area = 1.218e-5

    # Star formation history file
    # Test to load data
    this_dir, this_filename = os.path.split(__file__)
    file_path = os.path.join(this_dir, "smc_sfh.dat")

    with open(file_path) as f:
        line_num = 0

        for line in f:

            line_num += 1

            if line_num < 27: continue

            line_data = np.array(line.split()).astype(str)

            out_data[0][0] = line_data[0]
            out_data[0][1] = (line_data[5].astype(np.float64)+line_data[6].astype(np.float64))/2.0
            out_data[0][2] = line_data[7].astype(np.float64) / area
            out_data[0][3] = line_data[10].astype(np.float64) / area

            if len(line_data) < 15:
                out_data[0][4] = 0.0
            else:
                out_data[0][4] = line_data[13].astype(np.float64) / area

            smc_data = np.append(smc_data, out_data[0])

    return smc_data


def load_smc_sfh(z=0.008):
    """ Create array of 1D interpolations in time of the
    star formation histories for each region in the SMC.

    Parameters
    ----------
    z : float (0.001, 0.004, 0.008)
        Metallicity for which to return star formation history
        Default = 0.008

    Returns
    -------
    SF_history : ndarray
        Array of star formation histories for each region
    """

    global smc_sfh

    # If already loaded, no need to reload
    if smc_sfh is not None: return smc_sfh

    # Load the LMC coordinates and SFH data
    smc_data = load_smc_data()

    smc_sfh = np.array([])
    age = np.array([])
    sfr = np.array([])

    _, idx = np.unique(smc_data["region"], return_index=True)
    regions = smc_data["region"][np.sort(idx)]

    for i in np.arange(len(regions)):
#    for r in regions:
        r = regions[i]

        age = smc_data["log_age"][np.where(smc_data["region"] == r)]
        if z == 0.008:
            sfr = smc_data["sfh_z008"][np.where(smc_data["region"] == r)]
        elif z == 0.004:
            sfr = smc_data["sfh_z004"][np.where(smc_data["region"] == r)]
        elif z == 0.001:
            sfr = smc_data["sfh_z001"][np.where(smc_data["region"] == r)]
        else:
            print "ERROR: You must choose an appropriate metallicity input"
            print "Possible options are 0.001, 0.004, 0.008"
            return -1

        smc_sfh = np.append(smc_sfh, interp1d(age[::-1], sfr[::-1], bounds_error=False, fill_value=0.0))

    return smc_sfh


def test_SMC_SFH_plots():
    """ Crete an array of 12 plots that show the LMC's star formation history
    at 12 different times.
    """

    plt.figure(figsize=(12,15))

    # Load SMC data
    smc_coor = load_smc_coor()
    smc_sfh = load_smc_sfh(z=0.008)

    def get_SMC_plot(age):
        sfr = np.array([])
        for i in np.arange(len(smc_coor)):
            sfr = np.append(sfr, get_SFH(smc_coor["ra"][i], \
                            smc_coor["dec"][i], age, smc_coor, smc_sfh))

        plt.tricontourf(smc_coor["ra"], smc_coor["dec"], sfr)
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


def get_SMC_plot(age):
    """ return a plot of the star formation history of the SMC at a particular age

    Parameters
    ----------
    age : float
        Star formation history age to calculate (Myr)

    Returns
    -------
    plt : matplotlib.pyplot plot
        Contour plot of the star formation history
    """

    if smc_coor is None: load_smc_coor()
    if smc_sfh is None: load_smc_sfh()

    sfr = np.array([])
    for i in np.arange(len(smc_coor)):
        sfr = np.append(sfr, get_SFH(smc_coor["ra"][i], \
                        smc_coor["dec"][i], age, smc_coor, smc_sfh))

    plt.tricontourf(smc_coor["ra"], smc_coor["dec"], sfr)
    plt.title(str(int(age)) + ' Myr')

    return plt
