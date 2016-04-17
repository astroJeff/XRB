import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import maxwell, norm, uniform, powerlaw, truncnorm
import emcee
sys.path.append('../binary')
import load_sse
import binary_evolve
from binary_evolve import A_to_P, P_to_A
sys.path.append('../constants')
import constants as c
sys.path.append('../sf_history')
import sf_history
from sf_history import deg_to_rad, rad_to_deg


nwalkers = 32


def get_stars_formed(ra, dec, t_min, t_max, v_sys, dist, N_size=512):
    """ Get the normalization constant for stars formed at ra and dec

    Parameters
    ----------
    ra : float
        right ascension input (decimals)
    dec : float
        declination input (decimals)
    t_min : float
        minimum time for a star to have been formed (Myr)
    t_max : float
        maximum time for a star to have been formed (Myr)
    v_sys : float
        Systemic velocity of system (km/s)
    dist : float
        Distance to the star forming region (km)

    Returns
    -------
    SFR : float
        Star formation normalization constant
    """

    ran_phi = 2.0*np.pi*uniform.rvs(size = N_size)

    c_1 = 3.0 / np.pi / (t_max - t_min)**3 * (dist/v_sys)**2
    ran_x = uniform.rvs(size = N_size)
    ran_t_b = (3.0 * ran_x / (c_1 * np.pi * (v_sys/dist)**2))**(1.0/3.0) + t_min

#    c_2 = dist_SMC / (np.pi * v_sys * (ran_t_b - t_min))
    theta_c = v_sys / dist * (ran_t_b - t_min)
    c_2 = 1.0 / (np.pi * theta_c**2)
    ran_y = uniform.rvs(size = N_size)
    ran_theta = np.sqrt(ran_y / (c_2 * np.pi))

    ran_ra = rad_to_deg(ran_theta) * np.cos(ran_phi) / np.cos(deg_to_rad(dec)) + ra
    ran_dec = rad_to_deg(ran_theta) * np.sin(ran_phi) + dec

    # Specific star formation rate (Msun/Myr/steradian)
    SFR = sf_history.get_SFH(ran_ra, ran_dec, ran_t_b/(c.yr_to_sec*1.0e6), sf_history.smc_coor, sf_history.smc_sfh)

    return np.mean(SFR)



# Priors
def ln_priors(y):
    """ Priors on the model parameters

    Parameters
    ----------
    y : ra, dec, M1, M2, A, ecc, v_k, theta, phi, ra_b, dec_b, t_b
        Current HMXB location (ra, dec) and 10 model parameters

    Returns
    -------
    lp : float
        Natural log of the prior
    """

#    M1, M2, A, v_k, theta, phi, ra_b, dec_b, t_b = y
    ra, dec, M1, M2, A, ecc, v_k, theta, phi, ra_b, dec_b, t_b = y

    lp = 0.0

    # P(M1)
    if M1 < c.min_mass or M1 > c.max_mass: return -np.inf
    norm_const = (c.alpha+1.0) / (np.power(c.max_mass, c.alpha+1.0) - np.power(c.min_mass, c.alpha+1.0))
    lp += np.log( norm_const * np.power(M1, c.alpha) )

    # M1 must be massive enough to evolve off the MS by t_obs
    if load_sse.func_sse_tmax(M1) > t_b: return -np.inf

    # P(M2)
    q = M2 / M1
    if q < 0.3 or q > 1.0: return -np.inf
    lp += np.log( (1.0 / 0.7) * (1.0 / M1 ) )

    # P(A)
    if A*(1.0-ecc) < c.min_A or A*(1.0+ecc) > c.max_A: return -np.inf
    norm_const = np.log(c.max_A) - np.log(c.min_A)
    lp += np.log( norm_const / A )

    # P(ecc)
    if ecc < 0.0 or ecc > 1.0: return -np.inf
    lp += np.log(2.0 * ecc)

    # P(v_k)
    if v_k < 0.0: return -np.inf
    lp += np.log( maxwell.pdf(v_k, scale=c.v_k_sigma) )

    # P(theta)
    if theta <= 0.0 or theta >= np.pi: return -np.inf
    lp += np.log(np.sin(theta) / 2.0)

    # P(phi)
    if phi < 0.0 or phi > 2.0*np.pi: return -np.inf
    lp += -np.log( 2.0*np.pi )

    # Get star formation history
    sfh = sf_history.get_SFH(ra_b, dec_b, t_b, sf_history.smc_coor, sf_history.smc_sfh)
    if sfh == 0.0: return -np.inf

    # P(alpha, delta)
    # Closest point must be within survey. We estimate using the
    # field of view of the CCD in the survey: 24' x 24' which is
    # 0.283 degrees from center to corner. We round up to 0.3
    # Area probability depends only on declination
    dist_closest = sf_history.get_dist_closest(ra_b, dec_b, sf_history.smc_coor)
    if dist_closest > 0.3: return -np.inf
    lp += np.log(np.cos(deg_to_rad(dec_b)) / 2.0)

    ##################################################################
    # We add an additional prior that scales the RA and Dec by the
    # area available to it, i.e. pi theta^2, where theta is the angle
    # of the maximum projected separation over the distance.
    #
    # Still under construction
    ##################################################################
    M1_b, M2_b, A_b = binary_evolve.func_MT_forward(M1, M2, A, ecc)
    A_c, v_sys, ecc = binary_evolve.func_SN_forward(M1_b, M2_b, A_b, v_k, theta, phi)
    if ecc < 0.0 or ecc > 1.0 or np.isnan(ecc): return -np.inf

#    t_sn = (t_b - func_sse_tmax(M1)) * 1.0e6 * yr_to_sec  # The time since the primary's core collapse
#    theta_max = (v_sys * t_sn) / dist_LMC  # Unitless
#    area = np.pi * rad_to_dec(theta_max)**2
#    lp += np.log(1.0 / area)
    ##################################################################
    # Instead, let's estimate the number of stars formed within a cone
    # around the observed position, over solid angle and time.
    # Prior is in Msun/Myr/steradian
    ##################################################################
    t_min = load_sse.func_sse_tmax(M1) * 1.0e6 * c.yr_to_sec
    t_max = (load_sse.func_sse_tmax(M2_b) - binary_evolve.func_get_time(M1, M2, 0.0)) * 1.0e6 * c.yr_to_sec
    if t_max-t_min < 0.0: return -np.inf
    theta_C = (v_sys * (t_max - t_min)) / c.dist_SMC
    stars_formed = get_stars_formed(ra, dec, t_min, t_max, v_sys, c.dist_SMC)
    if stars_formed == 0.0: return -np.inf
    volume_cone = (np.pi/3.0 * theta_C**2 * (t_max - t_min) / c.yr_to_sec / 1.0e6)
    lp += np.log(sfh / stars_formed / volume_cone)
    ##################################################################

#    # P(t_b | alpha, delta)
#    sfh_normalization = 1.0e-6
#    lp += np.log(sfh_normalization * sfh)

    # Add a prior so that the post-MT secondary is within the correct bounds
    M2_c = M1 + M2 - load_sse.func_sse_he_mass(M1)
    if M2_c > c.max_mass or M2_c < c.min_mass: return -np.inf

    # Add a prior so the effective time remains bounded
    t_eff_obs = binary_evolve.func_get_time(M1, M2, t_b)
    if t_eff_obs < 0.0: return -np.inf

    return lp


def get_theta_proj(ra, dec, ra_b, dec_b):
    """ Get the angular distance between two coordinates

    Parameters
    ----------
    ra : float
        RA of coordinate 1 (radians)
    dec : float
        Dec of coordinate 1 (radians)
    ra_b : float
        RA of coordinate 2 (radians)
    dec_b : float
        Dec of coordinate 2 (radians)

    Returns
    -------
    theta : float
        Angular separation of the two coordinates (radians)
    """

    return np.sqrt((ra-ra_b)**2 * np.cos(dec)*np.cos(dec_b) + (dec-dec_b)**2)


# Functions for coordinate jacobian transformation
def get_dtheta_dalpha(alpha, delta, alpha_b, delta_b):
    """ Calculate the coordinate transformation derivative dtheta/dalpha """

    theta_proj = get_theta_proj(alpha, delta, alpha_b, delta_b)
    return (alpha-alpha_b) * np.cos(delta) * np.cos(delta_b) / theta_proj

def get_dtheta_ddelta(alpha, delta, alpha_b, delta_b):
    """ Calculate the coordinate transformation derivative dtheta/ddelta """

    theta_proj = get_theta_proj(alpha, delta, alpha_b, delta_b)
    return - 1.0/(2.0*theta_proj) * (np.cos(delta_b)*np.sin(delta)*(alpha_b-alpha)**2 + 2.0*(delta_b-delta))

def get_domega_dalpha(alpha, delta, alpha_b, delta_b):
    """ Calculate the coordinate transformation derivative domega/dalpha """

    z = (delta_b-delta) / ((alpha_b-alpha) * np.cos(delta_b))
    return 1.0 / (1.0 + z*z) * z / (alpha_b - alpha)

def get_domega_ddelta(alpha, delta, alpha_b, delta_b):
    """ Calculate the coordinate transformation derivative domega/ddelta """

    z = (delta_b-delta) / ((alpha_b-alpha) * np.cos(delta_b))
    return - 1.0 / (1.0 + z*z) / ((alpha_b-alpha) * np.cos(delta_b))

def get_J_coor(alpha, delta, alpha_b, delta_b):
    """ Calculate the Jacobian (determinant of the jacobian matrix) of
    the coordinate transformation
    """

    dt_da = get_dtheta_dalpha(alpha, delta, alpha_b, delta_b)
    dt_dd = get_dtheta_ddelta(alpha, delta, alpha_b, delta_b)
    do_da = get_domega_dalpha(alpha, delta, alpha_b, delta_b)
    do_dd = get_domega_ddelta(alpha, delta, alpha_b, delta_b)

    return dt_da*do_dd - dt_dd*do_da



def ln_posterior(x, args):
    """ Calculate the natural log of the posterior probability

    Parameters
    ----------
    x : M1, M2, A, ecc, v_k, theta, phi, ra_b, dec_b, t_b
        10 model parameters
    args : M2_d, P_orb_obs, ecc_obs, ra, dec
        System observations

    Returns
    -------
    lp : float
        Natural log of the posterior probability
    """

    M1, M2, A, ecc, v_k, theta, phi, ra_b, dec_b, t_b = x
    M2_d, P_orb_obs, ecc_obs, ra, dec = args
    y = ra, dec, M1, M2, A, ecc, v_k, theta, phi, ra_b, dec_b, t_b


    # Call priors
    lp = ln_priors(y)
    if np.isinf(lp): return -np.inf

    ll = 0

    M1_b, M2_b, A_b = binary_evolve.func_MT_forward(M1, M2, A, ecc)
    A_c, v_sys, ecc_out = binary_evolve.func_SN_forward(M1_b, M2_b, A_b, v_k, theta, phi)
    M2_d_out, L_x_out, M_dot_out, A_d = binary_evolve.func_Lx_forward(M1, M2, M2_b, A_c, ecc_out, t_b)
    P_orb_d = A_to_P(c.M_NS, M2_d_out, A_d)

    # If system disrupted or no X-ray luminosity, return -infty
    if ecc_out < 0.0 or ecc_out > 1.0 or np.isnan(ecc) or L_x_out==0.0: return -np.inf

    # Observed secondary mass
    delta_M_err = 0.2
    coeff_M = -0.5 * np.log( 2. * np.pi * delta_M_err**2 )
    argument_M = -( M2_d - M2_d_out ) * ( M2_d - M2_d_out ) / ( 2. * delta_M_err**2 )
    ll += coeff_M + argument_M

    # Observed X-ray luminosity
#    delta_ln_L_x_err = 0.2
#    coeff_ln_L_x = -0.5 * np.log( 2. * np.pi * delta_ln_L_x_err**2 )
#    argument_ln_L_x = -( np.log(L_x) - np.log(L_x_out) ) * ( np.log(L_x) - np.log(L_x_out) ) / ( 2. * delta_ln_L_x_err**2 )
#    ll += coeff_ln_L_x + argument_ln_L_x

    # Observed orbital period
    delta_P_orb_err = 1.0 # uncertainty: 1 day
    coeff_P_orb = -0.5 * np.log( 2. * np.pi * delta_P_orb_err**2)
    argument_P_orb = -( P_orb_obs - P_orb_d )**2 / ( 2. * delta_P_orb_err**2 )
    ll += coeff_P_orb + argument_P_orb

    # Observed eccentricity
    delta_ecc_err = 0.05 # uncertainty: 0.05
    coeff_ecc = -0.5 * np.log( 2. * np.pi * delta_ecc_err**2)
    argument_ecc = -( ecc_obs - ecc_out )**2 / ( 2. * delta_ecc_err**2 )
    ll += coeff_ecc + argument_ecc

    ######## Under Construction #######
    theta_proj = get_theta_proj(deg_to_rad(ra), deg_to_rad(dec), deg_to_rad(ra_b), deg_to_rad(dec_b))  # Projected travel distance
    t_sn = (t_b - load_sse.func_sse_tmax(M1)) * 1.0e6 * c.yr_to_sec  # The time since the primary's core collapse
    tmp = (v_sys * t_sn) / c.dist_SMC  # Unitless
    conds = [theta_proj>tmp, theta_proj<=tmp]  # Define conditional
    funcs = [lambda theta_proj: -np.inf, lambda theta_proj: np.log(np.tan(np.arcsin(theta_proj/tmp))/tmp)]
    J_coor = np.abs(get_J_coor(deg_to_rad(ra), deg_to_rad(dec), deg_to_rad(ra_b), deg_to_rad(dec_b))) # Jacobian for coordinate change
    P_omega = 1.0 / (2.0 * np.pi)
    ll += np.piecewise(theta_proj, conds, funcs) + np.log(P_omega) + np.log(1.0 / J_coor)
#    print np.piecewise(theta_proj, conds, funcs), np.log(J_coor), np.log(P_omega), rad_to_dec(np.arcsin(theta_proj/tmp))


#    print rad_to_dec(theta_proj)*3600.0, tmp, t_sn, v_sys, v_sys*t_sn, \
#        np.arcsin(theta_proj/tmp), np.tan(np.arcsin(theta_proj/tmp)), np.piecewise(theta_proj, conds, funcs), \
#        np.log(J_coor * P_omega)

    # Observed distance from the birth cluster
#    t_travel = (t_b - func_sse_tmax(M1)) * 1.0e6 * yr_to_sec
#    sin_theta = theta_proj * dist_LMC / (v_sys * t_travel)
#    if sin_theta < 0.0 or sin_theta > 1.0: return -np.inf  # sine must be bounded

#    cos_theta = np.sqrt(1.0 - sin_theta*sin_theta)
#    prob = sin_theta / cos_theta * v_sys * t_travel / dist_LMC
#    ll += np.log(prob)

    if np.isnan(ll): return -np.inf

    return ll + lp


# This function runs emcee
def run_emcee(M2_d, P_orb_obs, ecc_obs, ra, dec, nburn=1000, nsteps=1000):
    """ Run the emcee function

    Parameters
    ----------
    M2_d : float
        Observed secondary mass
    P_orb_obs : float
        Observed orbital period
    ecc_obs : float
        Observed orbital eccentricity
    ra : float
        Observed right ascension
    dec : float
        Observed declination

    Returns
    -------
    sampler : emcee object
    """

    # First thing is to load the sse data and SF_history data
    load_sse.load_sse()
    sf_history.load_sf_history()

    # Get initial values
    initial_vals = get_initial_values(M2_d)

    # Define sampler
    args = [[M2_d, P_orb_obs, ecc_obs, ra, dec]]
    sampler = emcee.EnsembleSampler(nwalkers=nwalkers, dim=10, lnpostfn=ln_posterior, args=args)


    # Assign initial values
    p0 = np.zeros((nwalkers,10))
    p0 = set_walkers(initial_vals, args[0])


    # Burn-in
    pos,prob,state = sampler.run_mcmc(p0, N=nburn)


    # Full run
    sampler.reset()
    pos,prob,state = sampler.run_mcmc(pos, N=nsteps)


    return sampler

def set_walkers(initial_masses, args, nwalkers=32):
    """ Get the initial positions for the walkers

    Parameters
    ----------
    initial_masses : ndarray
        array of initial masses of length nwalkers
    args : M2_d, P_orb_obs, ecc_obs, ra, dec
        observed system parameters
    """

    M2_d, P_orb_obs, ecc_obs, ra, dec = args

    p0 = np.zeros((nwalkers,10))
    p0[:,0] = initial_masses.T[0] # M1
    p0[:,1] = initial_masses.T[1] # M2

    p0[:,2] = np.power(10.0, np.random.uniform(2.0, 3.0, size=nwalkers)) # A
    p0[:,3] = np.random.uniform(0.0, 0.99, size=nwalkers) # ecc
    p0[:,4] = np.random.normal(50.0, 10.0, size=nwalkers) # v_k
    p0[:,5] = np.random.normal(np.pi, 0.2, size=nwalkers) # theta
    p0[:,6] = np.random.normal(1.0, 0.2, size=nwalkers) # phi
    p0[:,7] = np.random.normal(ra, 0.01, size=nwalkers) # ra
    p0[:,8] = np.random.normal(dec, 0.01, size=nwalkers) # dec
    p0[:,9] = initial_masses.T[2] # t_b

    for i in np.arange(nwalkers):
        counter = 0

        prob = ln_posterior(p0[i], args)
        while(np.isinf(prob)):
            p0[i,2] = np.power(10.0, np.random.uniform(2.0, 3.0)) # A
            p0[i,3] = np.random.uniform(0.0, 0.99) # ecc
            p0[i,4] = np.random.normal(50.0, 10.0) # v_k
            p0[i,5] = np.random.normal(np.pi, 0.2) # theta
            p0[i,6] = np.random.normal(1.0, 0.2) # phi
            p0[i,7] = np.random.normal(ra, 0.01) # ra
            p0[i,8] = np.random.normal(dec, 0.01) # dec
#            p0[:,8] = np.random.normal(1.2 * func_sse_tmax(initial_masses.T[0]), 1.0, size=nwalkers) # t_b

            prob = ln_posterior(p0[i], args)

            counter += 1

            if counter > 1000: break



    # Check if there are still bad walkers
    bad_walkers = False
    for i in np.arange(nwalkers):
        if np.isinf(ln_posterior(p0[i], args)):
            bad_walkers = True

    # If there still are any bad walkers, we move walker to a value close to a good walker
    # Get the index of a good walker
    good_index = -1
    if bad_walkers == True:
        for i in np.arange(nwalkers):
            if not np.isinf(ln_posterior(p0[i], args)):
                good_index = i
                break

        # If there are no good walkers, we're screwed
        if good_index == -1:
            print "NO VALID WALKERS"
            sys.exit(0)

        # Now we move any bad walkers near (within ~1%) the good_index walker
        for i in np.arange(nwalkers):
            if np.isinf(ln_posterior(p0[i], args)):
                for j in np.arange(7):
                    p0[i][j] = np.random.normal(p0[good_index][j], 0.01*p0[good_index][j])
                p0[i][7] = np.random.normal(p0[good_index][7], 0.0001*p0[good_index][j])
                p0[i][8] = np.random.normal(p0[good_index][8], 0.0001*p0[good_index][j])
                p0[i][9] = p0[good_index][9]

    return p0

def get_initial_values(M2_d):
    """ Calculate an array of initial masses and birth times

    Parameters
    ----------
    M2_d : float
        Observed secondary mass

    Returns
    -------
    pos : ndarray, shape=(nwalkers,3)
        Array of (M1, M2, t_b)
    """


    # Start by using MCMC on just the masses to get a distribution of M1 and M2

    args = [[M2_d]]
    sampler = emcee.EnsembleSampler(nwalkers=nwalkers, dim=3, lnpostfn=ln_posterior_initial, args=args)

    # Picking the initial masses and birth time will need to be optimized
    t_b = 1000.0
    M1_tmp = max(0.6*M2_d, c.min_mass)
    M2_tmp = 1.1*M2_d - M1_tmp
    p_i = [M1_tmp, M2_tmp,t_b]
    tmp = binary_evolve.func_get_time(*p_i) - 1000.0
    t_b = 0.9 * (load_sse.func_sse_tmax(p_i[0] + p_i[1] - load_sse.func_sse_he_mass(p_i[0])) - tmp)
    p_i[2] = t_b

    t_eff_obs = binary_evolve.func_get_time(*p_i)
    M_b_prime = p_i[0] + p_i[1] - load_sse.func_sse_he_mass(p_i[0])
    M_tmp, Mdot_tmp, R_tmp = load_sse.func_get_sse_star(M_b_prime, t_eff_obs)

    min_M = load_sse.func_sse_min_mass(t_b)

    n_tries = 0
    while t_eff_obs < 0.0 or Mdot_tmp == 0.0:

        p_i[0] = (c.max_mass - min_M) * np.random.uniform() + min_M
        p_i[1] = (0.7 * np.random.uniform() + 0.3) * p_i[0]
        p_i[2] = (np.random.uniform(5.0) + 1.2) * load_sse.func_sse_tmax(M2_d*0.6)

        t_eff_obs = binary_evolve.func_get_time(*p_i)
        if t_eff_obs < 0.0: continue

        M_b_prime = p_i[0] + p_i[1] - load_sse.func_sse_he_mass(p_i[0])
        if M_b_prime > c.max_mass: continue

        M_tmp, Mdot_tmp, R_tmp = load_sse.func_get_sse_star(M_b_prime, t_eff_obs)

        # Exit condition
        n_tries += 1
        if n_tries > 100: break


    # initial positions for walkers
    p0 = np.zeros((nwalkers,3))
    a, b = (min_M - p_i[0]) / 0.5, (c.max_mass - p_i[0]) / 0.5
    p0[:,0] = truncnorm.rvs(a, b, loc=p_i[0], scale=1.0, size=nwalkers) # M1
    p0[:,1] = np.random.normal(p_i[1], 0.5, size=nwalkers) # M2
    p0[:,2] = np.random.normal(p_i[2], 0.2, size=nwalkers) # t_b

    # burn-in
    pos,prob,state = sampler.run_mcmc(p0, N=100)

    return pos


# The posterior function for the initial parameters
def ln_posterior_initial(x, args):
    """ Calculate the posterior probability for the initial mass
    and birth time calculations.

    Parameters
    ----------
    x : M1, M2, t_obs
        model parameters
    args : M2_d
        observed companion mass

    Returns
    -------
    lp : float
        posterior probability
    """

    M1, M2, t_obs = x
    M2_d = args

    y = M1, M2, M2_d, t_obs
    lp = ln_priors_initial(y)
    if np.isinf(lp): return -np.inf

    # Get observed mass, mdot
    t_eff_obs = binary_evolve.func_get_time(M1, M2, t_obs)
    M2_c = M1 + M2 - load_sse.func_sse_he_mass(M1)
    M2_tmp, M2_dot, R_tmp = load_sse.func_get_sse_star(M2_c, t_eff_obs)

    # Somewhat arbitrary definition of mass error
    delta_M_err = 1.0
    coeff = -0.5 * np.log( 2. * np.pi * delta_M_err*delta_M_err )
    argument = -( M2_d - M2_tmp ) * ( M2_d - M2_tmp ) / ( 2. * delta_M_err*delta_M_err )

    return coeff + argument + lp

# Prior function for the initial parameters
def ln_priors_initial(x):
    """ Calculate the prior probability for the initial mass
    and birth time calculations.

    Parameters
    ----------
    x : M1, M2, M2_d, t_obs
        Model parameters plus the observed companion mass

    Returns
    -------
    ll : float
        Prior probability of the model parameters
    """

    M1, M2, M2_d, t_obs = x

    # M1
    if M1 < c.min_mass or M1 > c.max_mass: return -np.inf

    # M2
    if M2 < 0.3*M1 or M2 > M1: return -np.inf

    # Add a prior so that the post-MT secondary is within the correct bounds
    M2_c = M1 + M2 - load_sse.func_sse_he_mass(M1)
    if M2_c > c.max_mass or M2_c < c.min_mass: return -np.inf

    # Add a prior so the primary can go through a SN by t_obs
    if load_sse.func_sse_tmax(M1) > t_obs: return -np.inf

    # Add a prior so the effective time remains bounded
    t_eff_obs = binary_evolve.func_get_time(M1, M2, t_obs)
    if t_eff_obs < 0.0: return -np.inf

    # Add a prior so that only those masses with a non-zero Mdot are allowed
    M2_tmp, M2_dot, R_tmp = load_sse.func_get_sse_star(M2_c, t_eff_obs)
    if M2_dot == 0.0: return -np.inf

    return 0.0
