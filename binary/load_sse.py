import os
import glob
import numpy as np
from scipy.interpolate import interp1d


# Module-wide variables
func_sse_mdot = None
func_sse_mass = None
func_sse_radius = None
func_sse_tmax = None
func_sse_min_mass = None
func_sse_r_ZAMS = None
func_sse_rmax = None
func_sse_r_MS_max = None
func_sse_he_mass = None
func_sse_ms_time = None
func_sse_k_type = None


def load_sse():
    """ Load all sse interpolations """
    read_sse_data()
    read_he_star_data()


def read_sse_data():
    """ Reads data from sse files.

    Interpolations with global access are created
    ---------------------------------------------
    func_sse_mass : interp1d ndarray
        mass of a star as a function of time
    func_sse_radius : interp1d ndarray
        radius of a star as a function of time
    func_sse_mdot : interp1d ndarray
        mass loss rate of a star as a function of time
    func_sse_tmax : interp1d
        lifetime of a star of input mass
    func_sse_min_mass : interp1d
        mass of a star of input lifetime
    func_sse_r_ZAMS : interp1d
        zero-age main sequence radius of a star of input mass
    func_sse_rmax : interp1d
        maximum radius of a star of input mass
    func_sse_r_MS_max : interp1d
        maximum radius of a star on the main sequence of input mass
    """
    names = ["time","type","mass","mdot","radius"]

    f_list = glob.glob("../data/sse_data/mdot_*.dat")

    f_list = np.sort(f_list)

    # Create empty data storage structure
    sse_data = np.recarray(0, names=names, formats=['f8,float64,float64'])

    global func_sse_mdot
    global func_sse_mass
    global func_sse_radius
    global func_sse_tmax
    global func_sse_min_mass
    global func_sse_r_ZAMS
    global func_sse_rmax
    global func_sse_r_MS_max
    global func_sse_k_type

    func_sse_mass = np.array([])
    func_sse_mdot = np.array([])
    func_sse_radius = np.array([])
    func_sse_k_type = np.array([])
    sse_tmp_mass = np.array([])
    sse_tmp_time = np.array([])
    sse_tmp_radius = np.array([])
    sse_tmp_MS_radius = np.array([])
    sse_tmp_ZAMS_radius = np.array([])

    for f in f_list:
        datafile = os.path.abspath(f)
#       sse_tmp_data = np.genfromtxt(datafile, usecols=(0,2,3,4), dtype="f8,float64,float64,float64", skip_header=1, names=names)
        sse_tmp_data = np.genfromtxt(datafile, dtype="f8,int,float64,float64,float64", skip_header=1, names=names)

        func_sse_mass = np.append(func_sse_mass, interp1d(sse_tmp_data["time"], sse_tmp_data["mass"], bounds_error=False, fill_value=sse_tmp_data["mass"][-1]))
        func_sse_mdot = np.append(func_sse_mdot, interp1d(sse_tmp_data["time"], sse_tmp_data["mdot"], bounds_error=False, fill_value=0.0))
        func_sse_radius = np.append(func_sse_radius, interp1d(sse_tmp_data["time"], sse_tmp_data["radius"], bounds_error=False, fill_value=0.0))
        func_sse_k_type = np.append(func_sse_k_type, func_sse_calc_k(sse_tmp_data["time"],sse_tmp_data["type"]))

        sse_tmp_time = np.append(sse_tmp_time, max(sse_tmp_data["time"])-1.0)
        sse_tmp_mass = np.append(sse_tmp_mass, sse_tmp_data["mass"][0])
        sse_tmp_ZAMS_radius = np.append(sse_tmp_ZAMS_radius, sse_tmp_data["radius"][0])
        sse_tmp_radius = np.append(sse_tmp_radius, max(sse_tmp_data["radius"]))
        sse_tmp_MS_radius = np.append(sse_tmp_MS_radius, max(sse_tmp_data["radius"][np.where(sse_tmp_data["type"]==1)]))

    # Lifetime function
    func_sse_tmax = interp1d(sse_tmp_mass, sse_tmp_time, bounds_error=False, fill_value=0.0)
    func_sse_min_mass = interp1d(sse_tmp_time, sse_tmp_mass, bounds_error=False, fill_value=1.0)
    func_sse_r_ZAMS = interp1d(sse_tmp_mass, sse_tmp_ZAMS_radius, bounds_error=False, fill_value=0.0)
    func_sse_rmax = interp1d(sse_tmp_mass, sse_tmp_radius, bounds_error=False, fill_value=0.0)
    func_sse_r_MS_max = interp1d(sse_tmp_mass, sse_tmp_MS_radius, bounds_error=False, fill_value=0.0)


def func_sse_calc_k(time, k_type):
    """ Creates a numpy piecewise function to determine
    the k-type of a star at every age for every stellar mass

    Parameters
    ----------
    time : ndarray float
        array of times
    k_type : ndarray int
        array of stellar k-types

    Returns
    -------
    func_k_type : lambda function
        lambda function containing a numpy piecewise function
    """

    # Get unique k-types, but unsorted
    _, idx = np.unique(k_type, return_index=True)
    values = k_type[np.sort(idx)]
    conditions = np.zeros(len(values))
    times = np.zeros(len(values)-1)

    idx = 0
    for i in np.arange(len(k_type)):
        k = k_type[i]

        if k != values[idx]:
            times[idx] = time[i]
            idx = idx+1

    if len(values) == 2:
        return lambda x: np.piecewise(x, [x<=times[0], times[0]<x], values)
    elif len(values) == 3:
        return lambda x: np.piecewise(x, [x<=times[0], (times[0]<x) & (x<=times[1]), (times[1]<x)], values)
    elif len(values) == 4:
        return lambda x: np.piecewise(x, [x<=times[0], (times[0]<x) & (x<=times[1]), (times[1]<x) & (x<=times[2]), \
                                        times[2]<x], values)
    elif len(values) == 5:
        return lambda x: np.piecewise(x, [x<=times[0], (times[0]<x) & (x<=times[1]), (times[1]<x) & (x<=times[2]), \
                        (times[2]<x) & (x<=times[3]), times[3]<x], values)
    elif len(values) == 6:
        return lambda x: np.piecewise(x, [x<=times[0], (times[0]<x) & (x<=times[1]), (times[1]<x) & (x<=times[2]), \
                        (times[2]<x) & (x<=times[3]), (times[3]<x) & (x<=times[4]), times[4]<x], values)
    elif len(values) == 7:
        return lambda x: np.piecewise(x, [x<=times[0], (times[0]<x) & (x<=times[1]), (times[1]<x) & (x<=times[2]), \
                        (times[2]<x) & (x<=times[3]), (times[3]<x) & (x<=times[4]), (times[4]<x) & (x<=times[5]), \
                        times[5]<x], values)
    elif len(values) == 8:
        return lambda x: np.piecewise(x, [x<=times[0], (times[0]<x) & (x<=times[1]), (times[1]<x) & (x<=times[2]), \
                        (times[2]<x) & (x<=times[3]), (times[3]<x) & (x<=times[4]), (times[4]<x) & (x<=times[5]), \
                        (times[5]<x) & (x<=times[6]), times[6]<x], values)
    else:
        print "ERROR: k_type conditionals"
        return None





def read_he_star_data():
    """ Reads data from the sse He star file, He_star.dat

    Interpolations with global access are created
    ---------------------------------------------
    func_sse_he_mass : interp1d
        He star mass of a star of input mass
    func_sse_ms_time : interp1d
        Main sequence lifetime of a star of input mass
    """

    global func_sse_he_mass
    global func_sse_ms_time

    names = ["mass","he_mass","t_ms"]
    f = "../data/sse_data/He_star.dat"

    sse_he_star = np.genfromtxt(os.path.abspath(f), usecols=(0,1,2), names=names)

    func_sse_he_mass = interp1d(sse_he_star["mass"], sse_he_star["he_mass"], bounds_error=False, fill_value=0.001)
    func_sse_ms_time = interp1d(sse_he_star["mass"], sse_he_star["t_ms"], bounds_error=False, fill_value=-1.0e10)



def func_get_sse_star(mass, time):
    """ Use the interpolation ndarrays defined in read_sse_data()
    to return the mass, mass loss rate, and radius of a star

    Parameters
    ----------
    mass : float
        Initial mass of the star
    time : float
        Time of interest

    Returns
    -------
    mass_out : float64
        mass of the star at the input time
    mdot_out : float64
        mass loss rate of the star at the input time
    radius_out : float64
        radius of the star at the input time
    k_out : float64
        the k-type of the star at the input time
    """

    if func_sse_mass is None or func_sse_mdot is None or func_sse_radius is None or func_sse_k_type is None:
        read_sse_data()

    mass_out = np.array([])
    mdot_out = np.array([])
    radius_out = np.array([])
    k_out = np.array([])

    if isinstance(mass, np.ndarray):

        if len(mass) == 1:
            mass_out = np.append(mass_out, func_sse_mass[int(mass*100.0)-100](time))
            mdot_out = np.append(mdot_out, func_sse_mdot[int(mass*100.0)-100](time))
            radius_out = np.append(radius_out, func_sse_radius[int(mass*100.0)-100](time))
            k_out = np.append(k_out, func_sse_k_type[int(mass*100.0)-100](time))
        else:
            for i in np.arange(len(mass)):
                if (int(mass[i]*100.0)-100<0 or int(mass[i]*100.0)-100>len(func_sse_mass)): continue
                mass_out = np.append(mass_out, func_sse_mass[int(mass[i]*100.0)-100](time[i]))
                mdot_out = np.append(mdot_out, func_sse_mdot[int(mass[i]*100.0)-100](time[i]))
                radius_out = np.append(radius_out, func_sse_radius[int(mass[i]*100.0)-100](time[i]))
                k_out = np.append(k_out, func_sse_k_type[int(mass[i]*100.0)-100](time[i]))
    else:
        mass_out = np.append(mass_out, func_sse_mass[int(mass*100.0)-100](time))
        mdot_out = np.append(mdot_out, func_sse_mdot[int(mass*100.0)-100](time))
        radius_out = np.append(radius_out, func_sse_radius[int(mass*100.0)-100](time))
        k_out = np.append(k_out, func_sse_k_type[int(mass*100.0)-100](time))

    return mass_out, mdot_out, radius_out, k_out
