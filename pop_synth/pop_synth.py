import sys
import numpy as np
from scipy.stats import maxwell, norm, uniform, powerlaw, truncnorm
sys.path.append('../constants')
import constants as c
sys.path.append('../SF_history')
import sf_history
from sf_history import deg_to_rad, rad_to_deg
sys.path.append('../binary')
import load_sse
import binary_evolve
from binary_evolve import A_to_P, P_to_A


# Define random deviate functions
def get_v_k(sigma, N):
    """ Generate random kick velocities from a Maxwellian distribution

    Parameters
    ----------
    sigma : float
        Maxwellian dispersion velocity (km/s)
    N : int
        Number of random samples to generate

    Returns
    -------
    v_k : ndarray
        Array of random kick velocities
    """
    return maxwell.rvs(scale = sigma, size = N)

def get_theta(N):
    """ Generate N random polar angles """
    return np.arccos(1.0-2.0*uniform.rvs(size = N))

def get_phi(N):
    """ Generate N random azimuthal angles """
    return 2.0*np.pi*uniform.rvs(size = N)

def get_M1(x1, x2, alpha, N):
    """ Generate random primary masses

    Parameters
    ----------
    x1 : float
        Lower limit for primary mass
    x2 : float
        Upper limit for primary mass
    alpha : float
        IMF exponent
    N : int
        Number of random samples to generate

    Returns
    -------
    M1 : ndarray
        Array of random primary masses
    """

    A = (alpha+1.0) / (np.power(x2, alpha+1.0) - np.power(x1, alpha+1.0))
    x = uniform.rvs(size = N)

    return np.power(x*(alpha+1.0)/A + np.power(x1, alpha+1.0), 1.0/(alpha+1.0))

# Mass ratio - uniform [0.3,1.0]
def get_q(N):
    """ Generate N random mass ratios """
    return 0.7 * uniform.rvs(size = N) + 0.3

def get_A(a1, a2, N):
    """ Generate random orbital separations

    Parameters
    ----------
    a1 : float
        Lower limit for orbital separation (Rsun)
    a2 : float
        Upper limit for orbital separation (Rsun)
    N : int
        Number of random samples to generate

    Returns
    -------
    A : ndarray
        Array of random orbital separations (Rsun)
    """

    x1 = np.log10(a1)
    x2 = np.log10(a2)

    return np.power(10.0, (x2-x1)*uniform.rvs(size=N) + x1)

def get_ecc(N):
    """ Generate N random eccentricities """
    return np.sqrt(uniform.rvs(size=N))


def generate_population(N):
    """ Generate initial conditions for a set of binaries

    Parameters
    ----------
    N : int
        Number of binary initial conditions to generate

    Returns
    -------
    M_1_a : ndarray
        Array of primary masses (Msun)
    M_2_a : ndarray
        Array of secondary masses (Msun)
    A_a : ndarray
        Array of orbital separations (Rsun)
    ecc_a : ndarray
        Array of orbital eccentricities (unitless)
    v_k : ndarray
        Array of SN kick velocities (km/s)
    theta : ndarray
        Array of SN kick polar angles (radians)
    phi : ndarray
        Array of SN kick azimuthal angles (radians)
    """

    theta = get_theta(N)
    phi = get_phi(N)
    M_1_a = get_M1(c.min_mass, c.max_mass, c.alpha, N)
    M_2_a = get_q(N) * M_1_a

    # Kick velocities depend on the core mass
    sigma = map(lambda m: c.v_k_sigma_ECS if m<c.ECS_Fe_mass else c.v_k_sigma, M_1_a)
    v_k = get_v_k(sigma, N)

    # To get Orbital Separation limits, need to take into account star radii
#     r_1_MS_max = func_sse_r_MS_max(M_1_a)
#     r_1_ZAMS = func_sse_r_ZAMS(M_1_a)
#     r_1_roche = func_Roche_radius(M_1_a, M_2_a, 1.0)
#     r_2_MS_max = func_sse_r_MS_max(M_2_a)
#     r_2_ZAMS = func_sse_r_ZAMS(M_2_a)
#     r_2_roche = func_Roche_radius(M_2_a, M_1_a, 1.0)
#     # Neither star can fill its Roche lobe at ZAMS
#     A_min = np.zeros(N)
#     for i in np.arange(N):
#         A_min[i] = max(r_1_ZAMS[i]/r_1_roche[i], r_2_ZAMS[i]/r_2_roche[i])
#     # Now, adjust A for eccentricity
#     A_min = A_min / (1.0 - ecc_a)
#     r_1_max = func_sse_rmax(M_1_a)
#     # But the primary must fill its Roche lobe at some point
#     A_max = r_1_max/r_1_roche
#     A_a = np.zeros(N)
#     for i in np.arange(N): A_a[i] = get_A(A_min[i], A_max[i], 1)

    # Orbital parameters
    A_a = np.zeros(N)
    ecc_a = np.zeros(N)
    for i in np.arange(N):
        ecc_a[i] = get_ecc(1)
        A_a[i] = get_A(c.min_A, c.max_A, 1)
        while (A_a[i]*(1.0-ecc_a[i]) < c.min_A or A_a[i]*(1.0+ecc_a[i]) > c.max_A):
            ecc_a[i] = get_ecc(1)
            A_a[i] = get_A(c.min_A, c.max_A, 1)


    return M_1_a, M_2_a, A_a, ecc_a, v_k, theta, phi



def get_random_positions(N, t_b, ra_in=-1.0, dec_in=-1.0):
    """ Use the star formation history to generate a population of new binaries

    Parameters
    ----------
    N : integer
        Number of positions to calculate
    t_b : float
        Birth time to calculate star formation history (Myr)
    ra_in : float
        RA of system (optional)
    dec_in : float
        Dec of system (optional)

    Returns
    -------
    ra_out : ndarray
        Array of output RA's (degrees)
    dec_out : ndarray
        Array of output Dec's (degrees)
    N_stars : int
        Normalization constant calculated from number of stars formed at time t_b
    """

    N_regions = len(sf_history.smc_sfh)

    # If provided with an ra and dec, only generate stars within 3 degrees of input position
    SF_regions = np.zeros((2,N_regions))
    for i in np.arange(N_regions):
        SF_regions[0,i] = i

        if ra_in == -1:
            SF_regions[1,i] = sf_history.smc_sfh[i](np.log10(t_b*1.0e6))
        elif sf_history.get_theta_proj_degree(sf_history.smc_coor["ra"][i], sf_history.smc_coor["dec"][i], ra_in, dec_in) < deg_to_rad(3.0):
            SF_regions[1,i] = sf_history.smc_sfh[i](np.log10(t_b*1.0e6))
        else:
            SF_regions[1,i] = 0.0

    N_stars = np.sum(SF_regions, axis=1)[1]

    # Normalize
    SF_regions[1] = SF_regions[1] / N_stars

    # Sort
    SF_sort = SF_regions[:,SF_regions[1].argsort()]

    # Move from normed PDF to CDF
    SF_sort[1] = np.cumsum(SF_sort[1])

    # TEST #
#    ra_out = lmc_coor["ra"][SF_sort[0][-100:].astype(int)]
#    dec_out = lmc_coor["dec"][SF_sort[0][-100:].astype(int)]
#    return ra_out, dec_out
    # TEST #

    # Random numbers
    y = uniform.rvs(size=N)

    # Create a 2D grid of CDFs, and random numbers
    SF_out, y_out = np.meshgrid(SF_sort[1], y)

    # Get index of closest region in sorted array
    indices = np.argmin((SF_out - y_out)**2,axis=1)

    # Move to indices of stored LMC SFH data array
    indices = SF_sort[0][indices].astype(int)

    # Get random ra's and dec's of each region
    ra_out = sf_history.smc_coor["ra"][indices]
    dec_out = sf_history.smc_coor["dec"][indices]

    # Width is 12 arcmin or 12/60 degrees for outermost regions
    # Width is 6 arcmin or 6/60 degrees for inner regions
#    width = 12.0 / 60.0 * np.ones(len(indices))
    width = 6.0 / 60.0 * np.ones(len(indices))
#    for i in np.arange(len(indices)):
#        if str(smc_coor["region"][indices[i]]).find("_") != -1:
#            width[i] = 6.0 / 60.0

    tmp_delta_ra = width * (2.0 * uniform.rvs(size=len(indices)) - 1.0) / np.cos(deg_to_rad(dec_out)) * 2.0
    tmp_delta_dec = width * (2.0 * uniform.rvs(size=len(indices)) - 1.0)

    ra_out = ra_out + tmp_delta_ra
    dec_out = dec_out + tmp_delta_dec

    return ra_out, dec_out, N_stars

def get_new_ra_dec(ra, dec, theta_proj, pos_ang):
    """ Find the new ra, dec from an initial ra, dec and how it moved
    in angular distance and position angle

    Parameters
    ----------
    ra : float
        RA birth place (degrees)
    dec : float
        Dec birth place (degrees)
    theta_proj : float
        Projected distance traveled (radians)
    pos_angle : float
        Position angle moved (radians)

    Returns
    -------
    ra_out : float
        New RA
    dec_out : float
        New Dec
    """

    delta_dec = theta_proj * np.cos(pos_ang)
    delta_ra = theta_proj * np.sin(pos_ang) / np.cos(deg_to_rad(dec))

    ra_out = ra + rad_to_deg(delta_ra)
    dec_out = dec + rad_to_deg(delta_dec)

    return ra_out, dec_out


def create_HMXBs(t_b, N_sys=1000, ra_in=-1, dec_in=-1):
    """ Randomly generate a sample of binaries and evolve them forward

    Parameters
    ----------
    t_b : float
        Birth time (Myr)
    N_sys : int
        Number of systems to generate (default=1000)
    ra_in : float
        RA of observed system
    dec_in : float
        Declination of observed system

    Returns
    -------
    HMXB : numpy recarray
        Array of HMXBs similar to the observed system
        names = ["ra", "dec", "ra_b", "dec_b", "P_orb", "ecc", "M_2_d", "theta_proj", "age", "norm"]
    init_params : numpy recarray
        Initial conditions of those systems similar to the observed
        names = ["M1","M2","A","ecc","v_k","theta","phi","ra_b","dec_b","t_b"]
    """

    names = ["M1","M2","A","ecc","v_k","theta","phi","ra_b","dec_b","t_b"]
    init_params = np.recarray(N_sys, names=names, formats=['float64,float64,float64,float64,float64,float64,float64,float64,float64,float64'])

    # Initial population
    M1_i, M2_i, A_i, ecc_i, v_k, theta, phi = generate_population(N_sys)
    ra_birth, dec_birth, N_stars_time = get_random_positions(N_sys, t_b, ra_in, dec_in)
    init_params["M1"] = M1_i
    init_params["M2"] = M2_i
    init_params["A"] = A_i
    init_params["ecc"] = ecc_i
    init_params["v_k"] = v_k
    init_params["theta"] = theta
    init_params["phi"] = phi
    init_params["ra_b"] = ra_birth
    init_params["dec_b"] = dec_birth
    init_params["t_b"] = t_b

    # Evolve population
    M_NS_out, M_2_out, L_x_out, v_sys_out, M2_dot_out, A_out, ecc_out, theta_out =  \
            full_forward(M1_i, M2_i, A_i, ecc_i, v_k, theta, phi, t_b)


    # Restrict for only HMXBs
    idx = np.intersect1d(np.where(L_x_out>1.0e25), np.where(ecc_out>0.0))
    idx = np.intersect1d(idx, np.where(ecc_out<1.0))
    idx = np.intersect1d(idx, np.where(A_out>1.0))

    N_survive = len(A_out[idx])




    pos_ang_expand = 2.0*np.pi * uniform.rvs(size = N_survive)
    ecc_expand = np.zeros(N_survive)
    P_orb_expand = np.zeros(N_survive)
    theta_proj_expand = np.zeros(N_survive)
    M_2_d_expand = np.zeros(N_survive)

    ecc_expand = ecc_out[idx]
    P_orb_expand = A_to_P(M_NS_out[idx], M_2_out[idx], A_out[idx])
    theta_proj_expand = theta_out[idx]
    M_2_d_expand = M_2_out[idx]


    ra_new, dec_new = get_new_ra_dec(ra_birth[idx], dec_birth[idx], theta_proj_expand, pos_ang_expand)

    names = ["ra", "dec", "ra_b", "dec_b", "P_orb", "ecc", "M_2_d", "theta_proj", "age", "norm"]
    HMXB = np.recarray(N_survive, names=names, formats=['float64,float64,float64,float64,float64,float64,float64,float64,float64,float64'])
    HMXB["ra"] = ra_new
    HMXB["dec"] = dec_new
    HMXB["ra_b"] = ra_birth[idx]
    HMXB["dec_b"] = dec_birth[idx]
    HMXB["P_orb"] = P_orb_expand
    HMXB["ecc"] = ecc_expand
    HMXB["M_2_d"] = M_2_d_expand
    HMXB["theta_proj"] = theta_proj_expand
    HMXB["age"] = t_b
    HMXB["norm"] = N_stars_time

    return HMXB, init_params[idx]



def full_forward(M1, M2, A, ecc, v_k, theta, phi, t_obs):
    """ Evolve a binary forward from its initial conditions

    Parameters
    ----------
    M1 : float
        Initial primary mass (Msun)
    M2 : float
        Initial secondary mass (Msun)
    A : float
        Initial orbital separation (Rsun)
    ecc : float
        Initial orbital eccentricity (unitless)
    v_k : float
        SN kick velocity
    theta : float
        SN kick polar angle
    phi : float
        SN kick azimuthal angle
    t_obs : float
        observation time

    Returns
    -------
    M_NS : float or ndarray
        Array of final primary masses (Currently set to the NS mass, c.M_NS)
    M_2 : float or ndarray
        Array of final secondary masses (Msun)
    L_x : float or ndarray
        X-ray luminosity (erg/s)
    v_sys : float or ndarray
        Systemic velocity (km/s)
    M2_dot : float or ndarray
        Mass accretion rate (Msun/yr)
    A : float or ndarray
        Orbital separation (Rsun)
    ecc : float or ndarray
        Orbital eccentricity (unitless)
    theta : float or ndarray
        Projected angular distance traveled from birth location (radians)
    """

    if load_sse.func_sse_mass is None:
        load_sse.load_sse()


    if isinstance(M1, np.ndarray):
        dtypes = [('M_NS','<f8'), \
                ('M_2','<f8'), \
                ('L_x','<f8'), \
                ('v_sys','<f8'), \
                ('M2_dot','<f8'), \
                ('A','<f8'), \
                ('ecc','<f8'), \
                ('theta','<f8')]

        HMXB = np.recarray(len(M1), dtype=dtypes)

        for i in np.arange(len(M1)):

            if isinstance(t_obs, np.ndarray):
                if t_obs[i] < load_sse.func_sse_ms_time(M1[i]):
                    HMXB["M_NS"][i] = M1[i]
                    HMXB["M_2"][i] = M2[i]
                    HMXB["A"][i] = A[i]
                    continue
            else:
                if t_obs < load_sse.func_sse_ms_time(M1[i]):
                    HMXB["M_NS"][i] = M1[i]
                    HMXB["M_2"][i] = M2[i]
                    HMXB["A"][i] = A[i]
                    continue


            # First MT phase
            M_1_b, M_2_b, A_b = binary_evolve.func_MT_forward(M1[i], M2[i], A[i], ecc[i])

            if isinstance(t_obs, np.ndarray):
                if t_obs[i] < load_sse.func_sse_tmax(M1[i]):
                    HMXB["M_NS"][i] = M_1_b
                    HMXB["M_2"][i] = M_2_b
                    HMXB["A"][i] = A_b
                    continue
            else:
                if t_obs < load_sse.func_sse_tmax(M1[i]):
                    HMXB["M_NS"][i] = M_1_b
                    HMXB["M_2"][i] = M_2_b
                    HMXB["A"][i] = A_b
                    continue


            # SN
            A_tmp, v_sys_tmp, e_tmp = binary_evolve.func_SN_forward(M_1_b, M_2_b, A_b, v_k[i], theta[i], phi[i])

            # XRB
            if isinstance(t_obs, np.ndarray):
                M_2_tmp, L_x_tmp, M2_dot_out, A_out = binary_evolve.func_Lx_forward(M1[i], M2[i], M_2_b, A_tmp, e_tmp, t_obs[i])
                theta_out = (t_obs[i] - load_sse.func_sse_tmax(M1[i])) * v_sys_tmp / c.dist_SMC * c.yr_to_sec * 1.0e6 * np.sin(get_theta(1))
            else:
                M_2_tmp, L_x_tmp, M2_dot_out, A_out = binary_evolve.func_Lx_forward(M1[i], M2[i], M_2_b, A_tmp, e_tmp, t_obs)
                theta_out = (t_obs - load_sse.func_sse_tmax(M1[i])) * v_sys_tmp / c.dist_SMC * c.yr_to_sec * 1.0e6 * np.sin(get_theta(1))


            HMXB["M_NS"][i] = c.M_NS
            HMXB["M_2"][i] = M_2_tmp
            HMXB["L_x"][i] = L_x_tmp
            HMXB["v_sys"][i] = v_sys_tmp
            HMXB["M2_dot"][i] = M2_dot_out
            HMXB["A"][i] = A_out
            HMXB["ecc"][i] = e_tmp
            HMXB["theta"][i] = theta_out


        return HMXB["M_NS"], HMXB["M_2"], HMXB["L_x"], HMXB["v_sys"], HMXB["M2_dot"], HMXB["A"], HMXB["ecc"], HMXB["theta"]

    else:

        # Star does not make it to MT phase
        if t_obs < load_sse.func_sse_ms_time(M1): return M1, M2, 0.0, 0.0, 0.0, A, ecc, 0.0

        # MT phase
        M_1_b, M_2_b, A_b = binary_evolve.func_MT_forward(M1, M2, A, ecc)

        # Star does not make it to SN
        if t_obs < load_sse.func_sse_tmax(M1): return M_1_b, M_2_b, 0.0, 0.0, 0.0, A_b, ecc, 0.0

        # SN
        A_tmp, v_sys_tmp, e_tmp = binary_evolve.func_SN_forward(M_1_b, M_2_b, A_b, v_k, theta, phi)

        # XRB
        M_2_tmp, L_x_tmp, M2_dot_out, A_out = binary_evolve.func_Lx_forward(M1, M2, M_2_b, A_tmp, e_tmp, t_obs)

        theta_out = (t_obs - load_sse.func_sse_tmax(M1)) * v_sys_tmp / c.dist_SMC * c.yr_to_sec * 1.0e6 * np.sin(get_theta(1))

        return c.M_NS, M_2_tmp, L_x_tmp, v_sys_tmp, M2_dot_out, A_out, e_tmp, theta_out


def run_pop_synth(input_sys, N_sys=10000, t_low=15.0, t_high=60.0, delta_t=1):
    """ Run a forward population synthesis

    Parameters
    ----------
    input_sys : ra_sys, dec_sys, P_orb_sys, ecc_sys, M2_d_sys
        Observed values of individual system
    N_sys : int
        Number of systems per age calculated (10000)
    t_low : float
        Lower limit to age range tested (15 Myr)
    t_high : float
        Upper limit to age range tested (60 Myr)
    delta_t : float
        Age resolution (1 Myr)

    Returns
    -------
    HMXB_sys : numpy recarray
        Systems that evolve into the observed system
        names = ["ra", "dec", "ra_b", "dec_b", "P_orb", "ecc", "M_2_d", "theta_proj", "age", "norm"]
    init_params_sys : numpy recarray
        Initial conditions for the observed Systems
        names = ["M1","M2","A","ecc","v_k","theta","phi","ra_b","dec_b","t_b"]
    """


    ra_sys, dec_sys, P_orb_sys, ecc_sys, M2_d_sys = input_sys

    # First thing is to load the sse data and SF_history data
    load_sse.load_sse()
    sf_history.load_sf_history()

    names = ["ra", "dec", "ra_b", "dec_b", "P_orb", "ecc", "M_2_d", "theta_proj", "age", "norm"]

    HMXB = np.recarray(0, names=names, formats=['float64,float64,float64,float64,float64,float64,float64,float64,float64,float64'])
    HMXB_sys = np.recarray(0, names=names, formats=['float64,float64,float64,float64,float64,float64,float64,float64,float64,float64'])

    names = ["M1","M2","A","ecc","v_k","theta","phi","ra_b","dec_b","t_b"]
    init_params_sys = np.recarray(0, names=names, formats=['float64,float64,float64,float64,float64,float64,float64,float64,float64,float64'])


    theta_sep = np.array([])

    batch_size = 1000
    n_batch = int(np.ceil(float(N_sys)/float(batch_size)))

    for t_b in np.linspace(14.0, 56.0, 43):

        for batch in np.arange(n_batch):
            n_run = min(batch_size, N_sys - (batch)*batch_size)
            HMXB_t, init_params_t = create_HMXBs(t_b, N_sys=n_run, ra_in=ra_sys, dec_in=dec_sys)
            HMXB = np.concatenate((HMXB, HMXB_t))

            for i in np.arange(len(HMXB_t)):

                h = HMXB_t[i]
                p = init_params_t[i]

                angle = rad_to_deg(sf_history.get_theta_proj_degree(ra_sys, dec_sys, h["ra"], h["dec"]))
                theta_sep = np.append(theta_sep, angle)

                if angle < 0.2 \
                    and np.abs(h["P_orb"] - P_orb_sys) < 5.0 \
                    and np.abs(h["ecc"] - ecc_sys) < 0.1 \
                    and np.abs(h["M_2_d"] - M2_d_sys) < 1.0:

                    HMXB_sys = np.append(HMXB_sys, h)
                    init_params_sys = np.append(init_params_sys, p)


    return HMXB_sys, init_params_sys
