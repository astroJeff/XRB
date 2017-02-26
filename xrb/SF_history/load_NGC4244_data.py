import os
import numpy as np
from astropy.coordinates import SkyCoord
from scipy.interpolate import interp1d


def load_NGC4244_data():
    """ Return NGC4244 star formation rate per unit steradian

    Returns
    -------
    NGC4244_coor : ndarray
        NGC4244 region coordinates
        dtype: [('region','<i8'), ('ra','<f8'), ('dec','<f8')]

    NGC4244_sfh : ndarray
        NGC4244 star formation history
        dtype: [('region','<i8'), ('sfh','<f8')]
    """

    # Each region has an area of 12' x 12', or 1.218e-5 steradians
    area = 1.218e-5

    # Star formation history file
    # Test to load data
    this_dir, this_filename = os.path.split(__file__)
    file_path = os.path.join(this_dir, "NGC4244_sfr.dat")

    sfr_data = np.genfromtxt(file_path, names=True)


    # Load coordinates into coor array
    NGC4244_coor = np.zeros(len(sfr_data), dtype=[('region','<i8'),('ra','f8'),('dec','f8')])
    NGC4244_coor['region'] = sfr_data['region']
    NGC4244_coor['ra'] = sfr_data['ra']
    NGC4244_coor['dec'] = sfr_data['dec']


    # load star formation rates into array
    NGC4244_data = np.zeros(len(sfr_data), dtype=[('region','<i8'), ('sfh','<f8')])
    NGC4244_data['region'] = sfr_data['region']
    NGC4244_data['sfh'] = sfr_data['sfr']


    return NGC4244_coor, NGC4244_data





def load_NGC4244_sfh():
    """ Create array of 1D interpolations in time of the
    star formation histories for each region in the NGC4244.

    Returns
    -------
    NGC4244_coor : ndarray
        NGC4244 region coordinates
        dtype: [('region','<i8'), ('ra','<f8'), ('dec','<f8')]

    SF_history : ndarray
        Array of lambda functions providing the star formation
        rate in each region
    """


    # Load the LMC coordinates and SFH data
    NGC4244_coor, NGC4244_data = load_NGC4244_data()

    NGC4244_sfh = np.array([])

    # Create an array of callable constants corresponding to sfr's
    for sfr in NGC4244_data["sfh"]:
        NGC4244_sfh = np.append(NGC4244_sfh, lambda x, z=sfr: z)

    return NGC4244_coor, NGC4244_sfh
