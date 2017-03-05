from xrb.src.core import *
import copy
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import maxwell, norm, uniform, powerlaw, truncnorm
from scipy.integrate import quad
import emcee
from emcee.utils import MPIPool

import binary_c

from xrb.binary import load_sse, binary_evolve
from xrb.binary.binary_evolve import A_to_P, P_to_A


nwalkers = 80


# Priors
def ln_priors_population(y):
    """ Priors on the model parameters

    Parameters
    ----------
    y : M1, M2, A, ecc, v_k_1, theta_1, phi_1, v_k_2, theta_2, phi_2, metallicity
        11 model parameters

    Returns
    -------
    lp : float
        Natural log of the prior
    """

    M1, M2, A, ecc, v_k_1, theta_1, phi_1, v_k_2, theta_2, phi_2, metallicity = y

    lp = 0.0

    # P(M1)
    if M1 < 0.1 or M1 > 200.0: return -np.inf
    norm_const = (c.alpha+1.0) / (np.power(c.max_mass, c.alpha+1.0) - np.power(c.min_mass, c.alpha+1.0))
    lp += np.log( norm_const * np.power(M1, c.alpha) )

    # P(M2)
    if M2 < 0.1 or M2 > M1: return -np.inf
    # Normalization is over full q in (0,1.0)
    lp += np.log( (1.0 / M1 ) )

    # P(ecc)
    if ecc < 0.0 or ecc > 1.0: return -np.inf
    lp += np.log(2.0 * ecc)

    # P(A)
    if A*(1.0-ecc) < c.min_A or A*(1.0+ecc) > c.max_A: return -np.inf
    norm_const = np.log(c.max_A) - np.log(c.min_A)
    lp += np.log( norm_const / A )

    # P(v_k_1)
    if v_k_1 < 0.0: return -np.inf
    lp += np.log( maxwell.pdf(v_k_1, scale=c.v_k_sigma) )

    # P(theta_1)
    if theta_1 <= 0.0 or theta_1 >= np.pi: return -np.inf
    lp += np.log( np.sin(theta_1) / 2.0 )

    # P(phi_1)
    if phi_1 < 0.0 or phi_1 > np.pi: return -np.inf
    lp += -np.log( np.pi )

    # P(v_k_2)
    if v_k_2 < 0.0: return -np.inf
    lp += np.log( maxwell.pdf(v_k_2, scale=c.v_k_sigma) )

    # P(theta_2)
    if theta_2 <= 0.0 or theta_2 >= np.pi: return -np.inf
    lp += np.log( np.sin(theta_2) / 2.0 )

    # P(phi_2)
    if phi_2 < 0.0 or phi_2 > np.pi: return -np.inf
    lp += -np.log( np.pi )

    # P(metallicity)
    if metallicity < c.min_z or metallicity > c.max_z: return -np.inf
    a, b = (c.min_z - 0.02) / 0.005, (c.max_z - 0.02) / 0.005
    lp += np.log(truncnorm.pdf(metallicity, a, b, loc=0.02, scale=0.005))


    return lp



def ln_posterior_population(x):
    """ Calculate the natural log of the posterior probability

    Parameters
    ----------
    x : M1, M2, A, ecc, v_k_1, theta_1, phi_1, v_k_2, theta_2, phi_2, metallicity
        11 model parameters

    Returns
    -------
    lp : float
        Natural log of the posterior probability
    """

    M1, M2, A, ecc, v_k_1, theta_1, phi_1, v_k_2, theta_2, phi_2, metallicity = x
    time_max = 100.0 # Max time to evolve is 100 Myr

    # Empty array
    empty_arry = np.zeros(11) 

    # Call priors
    lp = ln_priors_population(x)
    if np.isinf(lp): return -np.inf, empty_arr 


    # Run binary_c evolution
    orbital_period = A_to_P(M1, M2, A)
    output = binary_c.run_binary(M1, M2, orbital_period, ecc, metallicity, time_max, v_k_1, theta_1, phi_1, v_k_2, theta_2, phi_2, 0, 1)
    m1_out, m2_out, A_out, ecc_out, v_sys, L_x, t_SN1, t_SN2, t_cur, k1, k2, comenv_count, evol_hist = output


    # Check to see if we've formed an HMXB
    if check_output(output, binary_type='BHBH'):

        # Calculate the merger time
        t_merge = merger_time(m1_out, m2_out, A_out, ecc_out, formula="peters")

        # Data saved to blobs
        binary_evolved = [m1_out, m2_out, A_out, ecc_out, v_sys, t_SN1, t_SN2, t_cur, t_merge, k1, k2]

        return lp, np.array(binary_evolved) 

    else:

        return -np.inf, empty_arr


def run_emcee_population(nburn=10000, nsteps=100000, nwalkers=80, threads=1, mpi=False):
    """ Run emcee on the entire X-ray binary population

    Parameters
    ----------
    nburn : int (optional)
        number of steps for the Burn-in (default=10000)
    nsteps : int (optional)
        number of steps for the simulation (default=100000)
    nwalkers : int (optional)
        number of walkers for the sampler (default=80)
    threads : int
        Number of threads to use for parallelization
    mpi : bool
        If true, use MPIPool for parallelization

    Returns
    -------
    sampler : emcee object
    """


    # Define posterior function
    posterior_function = ln_posterior_population

    # Set walkers
    print "Setting walkers..."
    p0 = np.zeros((nwalkers,11))
    p0 = set_walkers(nwalkers=nwalkers)
    print "...walkers are set"

    # Define sampler
    if mpi == True:
        pool = MPIPool()
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        sampler = emcee.EnsembleSampler(nwalkers=nwalkers, dim=11, lnpostfn=posterior_function, pool=pool)

    elif threads != 1:
        sampler = emcee.EnsembleSampler(nwalkers=nwalkers, dim=11, lnpostfn=posterior_function, threads=threads)
    else:
        sampler = emcee.EnsembleSampler(nwalkers=nwalkers, dim=11, lnpostfn=posterior_function)

    # Burn-in
    print "Starting burn-in..."
    pos,prob,state,binary_data = sampler.run_mcmc(p0, N=nburn)
    print "...finished running burn-in"

    # Full run
    print "Starting full run..."
    sampler.reset()
    pos,prob,state,binary_data = sampler.run_mcmc(pos, N=nsteps)
    print "...full run finished"

    return sampler



def set_walkers(nwalkers=80):
    """ Set the positions for the walkers

    Parameters
    ----------
    nwalkers : int (optional)
        Number of walkers for the sampler (default=80)

    Returns
    -------
    p0 : ndarray
        The initial walker positions

    """


    # Initial values
    m1 = 50.0
    m2 = 45.0
    eccentricity = 0.41
    metallicity = 0.002
    orbital_period = 500.0
    time = 100.0

    # SN kicks
    v_k_1 = 0.0
    theta_1 = 0.0
    phi_1 = 0.0
    v_k_2 = 0.0
    theta_2 = 0.0
    phi_2 = 0.0

    evol_flag = 0
    dco_flag = 1


    for i in np.arange(1000000):
        # Possible solution set: 243.57951517 2.42092161 181.48894684 1.75826025
        # v_k_1 = 82.67
        # theta_1 = 2.11
        # v_k_2 = 26.82
        # theta_2 = 2.27

        m1 = 50.0 * np.random.uniform(size=1) + 20.0
        m2 = m1 * (0.2 * np.random.uniform(size=1) + 0.8)

        v_k_1 = 300.0 * np.random.uniform(size=1) + 20.0
        theta_1 = np.pi * np.random.uniform(size=1)
        v_k_2 = 300.0 * np.random.uniform(size=1) + 20.0
        theta_2 = np.pi * np.random.uniform(size=1)

        phi_1 = np.pi * np.random.uniform(size=1)
        phi_2 = np.pi * np.random.uniform(size=1)

        orbital_period = 2000.0 * np.random.uniform(size=1) + 100.0

        output = binary_c.run_binary(m1, m2, orbital_period, eccentricity, metallicity, time,
                                     v_k_1, theta_1, phi_1,v_k_2, theta_2, phi_2, evol_flag, dco_flag)

        if check_output(output, "BHBH"):
       

            m1_out, m2_out, orbital_separation_out, eccentricity_out, system_velocity, L_x, \
                    time_SN_1, time_SN_2, time_current, ktype_1, ktype_2, comenv_count, evol_hist = output

            A = P_to_A(m1, m2, orbital_period)
            x = m1, m2, A, eccentricity, v_k_1, theta_1, phi_1, v_k_2, theta_2, phi_2, metallicity

            # ...and merges within a Hubble time...
            if not np.isinf(ln_posterior_population(x)[0]):

                # ...then use it as our starting system
                break

    if i==99999:
        print "Walkers could not be set"
        sys.exit(-1)


    output = binary_c.run_binary(m1, m2, orbital_period, eccentricity, metallicity, time,
                                 v_k_1, theta_1, phi_1,v_k_2, theta_2, phi_2, evol_flag, dco_flag)

    m1_out, m2_out, orbital_separation_out, eccentricity_out, system_velocity, L_x, \
            time_SN_1, time_SN_2, time_current, ktype_1, ktype_2, comenv_count, evol_hist = output

    A = P_to_A(m1, m2, orbital_period)

    x = m1, m2, A, eccentricity, v_k_1, theta_1, phi_1, v_k_2, theta_2, phi_2, metallicity


    ln_posterior = ln_posterior_population(x)[0]  
    if not np.isinf(ln_posterior): print "Found Initial Walker Solution: ", v_k_1, theta_1, v_k_2, theta_2, ln_posterior_population(x)[0] 



    # Now to generate a ball around these parameters

    # binary parameters
    m1_set = m1 + np.random.normal(0.0, 0.1, nwalkers)
    m2_set = m2 + np.random.normal(0.0, 0.1, nwalkers)
    e_set = eccentricity + np.random.normal(0.0, 0.01, nwalkers)
    metallicity_set = metallicity + np.random.normal(0.0, 0.0002, nwalkers)
    P_orb_set = orbital_period + np.random.normal(0.0, 2.0, nwalkers)
    a_set = binary_evolve.P_to_A(m1_set, m2_set, P_orb_set)

    # SN kick perameters
    v_k_1_set = v_k_1 + np.random.normal(0.0, 1.0, nwalkers)
    theta_1_set = theta_1 + np.random.normal(0.0, 0.01, nwalkers)
    phi_1_set = 1.5 + np.random.normal(0.0, 0.01, nwalkers)
    v_k_2_set = v_k_2 + np.random.normal(0.0, 1.0, nwalkers)
    theta_2_set = theta_2 + np.random.normal(0.0, 0.01, nwalkers)
    phi_2_set = 1.5 + np.random.normal(0.0, 0.01, nwalkers)


    # Check if any of these have posteriors with -infinity
    for i in np.arange(nwalkers):


        p = m1_set[i], m2_set[i], a_set[i], e_set[i], v_k_1_set[i], theta_1_set[i], \
                phi_1_set[i], v_k_2_set[i], theta_2_set[i], phi_2_set[i], metallicity_set[i]
        ln_posterior = ln_posterior_population(p)[0] 


        while ln_posterior < -10000.0:

            # binary parameters
            m1_set[i] = m1 + np.random.normal(0.0, 0.1, 1)
            m2_set[i] = m2 + np.random.normal(0.0, 0.1, 1)
            e_set[i] = eccentricity + np.random.normal(0.0, 0.01, 1)
            metallicity_set[i] = metallicity + np.random.normal(0.0, 0.0002, 1)
            P_orb_set[i] = orbital_period + np.random.normal(0.0, 2.0, 1)
            a_set[i] = binary_evolve.P_to_A(m1_set[i], m2_set[i], P_orb_set[i])

            # SN kick perameters
            v_k_1_set[i] = v_k_1 + np.random.normal(0.0, 1.0, 1)
            theta_1_set[i] = theta_1 + np.random.normal(0.0, 0.01, 1)
            phi_1_set[i] = phi_1 + np.random.normal(0.0, 0.01, 1)
            v_k_2_set[i] = v_k_2 + np.random.normal(0.0, 1.0, 1)
            theta_2_set[i] = theta_2 + np.random.normal(0.0, 0.01, 1)
            phi_2_set[i] = phi_2 + np.random.normal(0.0, 0.01, 1)

            p = m1_set[i], m2_set[i], a_set[i], e_set[i], v_k_1_set[i], theta_1_set[i], \
                    phi_1_set[i], v_k_2_set[i], theta_2_set[i], phi_2_set[i], metallicity_set[i]
            ln_posterior = ln_posterior_population(p)[0] 



    # Save and return the walker positions
    p0 = np.zeros((nwalkers,11))

    p0[:,0] = m1_set
    p0[:,1] = m2_set
    p0[:,2] = a_set
    p0[:,3] = e_set
    p0[:,4] = v_k_1_set
    p0[:,5] = theta_1_set
    p0[:,6] = phi_1_set
    p0[:,7] = v_k_2_set
    p0[:,8] = theta_2_set
    p0[:,9] = phi_2_set
    p0[:,10] = metallicity_set

    return p0




def merger_time(M1, M2, A, ecc, formula="peters"):
    """ Calculate the merger time of a compact binary

    Parameters
    ----------
    M1, M2 : float
        Masses of the two compact objects (Msun)
    A, ecc : float
        Orbital separation (Rsun) and eccentricity of the orbit
    formula : str (optional)
        Formula used to calculate the merger time
        Options: 'peters'

    Returns
    -------
    t_merge : float
        Time before the two stars to merge (Myr)
    """

    if formula is "peters":
        # Eqn 5.14 from Peters (1964), PhRv, 136, 1224

        def peters_integrand(e):
            return np.power(e, 29./19.) * np.power(1. + 121./304. * e*e, 1181./2299.) / np.power(1.-e*e, 1.5)

        # beta has units cm^4/s
        beta = 64./5. * (c.G**3) * M1 * M2 * (M1 + M2) / (c.c_light**5) * (c.Msun_to_g**3)

        # c0 has units cm
        c0 = A * (1.-ecc*ecc) * np.power(ecc, -12./19.) * np.power(1.0 + 121./304.*ecc*ecc, -870./2299.) * c.Rsun_to_cm

        # Calculate the merger time, convert from s to Myr
        t_merge = 12.0/19.0 * np.power(c0, 4.) / beta * quad(peters_integrand, 0.0, ecc)[0] / c.yr_to_sec / 1.0e6

        return t_merge


    else:
        print "You must provide an appropriate formula"
        print "Available options: 'peters'"
        sys.exit(-1)



def check_output(output, binary_type='BHBH'):
    """ Determine if the resulting binary from binary_c is of the type desired

    Parameters
    ----------
    M1_out, M2_out : float
        Masses of each object returned by binary_c
    A_out, ecc_out : float
        Orbital separation and eccentricity
    v_sys : float
        Systemic velocity of the system
    L_x : float
        X-ray luminosity of the system
    t_SN1, t_SN2 : float
        Time of the first and second SN, respectively
    t_cur : float
        Time at the end of the simulation
    k1, k2 : int
        K-types for each star
    comenv_count : int
        Number of common envelopes the star went through
    evol_hist : string
        String corresponding to the evolutionary history of the binary

    Returns
    -------
    binary_type : bool
        Is the binary of the requested type?

    """

    m1_out, m2_out, A_out, ecc_out, v_sys, L_x, t_SN1, t_SN2, t_cur, k1, k2, comenv_count, evol_hist = output


    type_options = ["HMXB", "BHBH", "NSNS", "BHNS"]

    if not np.any(binary_type == type_options):   
        print "The code is not set up to detect the type of binary you are interested in"
        sys.exit(-1)

    # Check if object is an HMXB
    if binary_type == "HMXB":
        if k1 < 13 or k1 > 14: return False
        if k2 > 9: return False
        if A_out <= 0.0: return False
        if ecc_out < 0.0 or ecc_out > 1.0: return False
        if L_x <= 0.0: return False
        if m2_out < 4.0: return False
        return True

    elif binary_type == "BHBH":
        if k1 != 14 or k2 != 14: return False
        if A_out <= 0.0: return False
        if ecc_out < 0.0 or ecc_out > 1.0: return False
        return True 
 
    elif binary_type == "NSNS":
        if k1 != 13 or k2 != 13: return False
        if A_out <= 0.0: return False
        if ecc_out < 0.0 or ecc_out > 1.0: return False
        return True

    elif binary_type == "BHNS":
        if (k1 != 14 or k2 != 13) and (k1 != 13 or k2 != 14): return False
        if A_out <= 0.0: return False
        if ecc_out < 0.0 or ecc_out > 1.0: return False
        return True


    return True
