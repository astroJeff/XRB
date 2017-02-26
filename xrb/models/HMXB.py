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
    y : M1, M2, A, ecc, v_k_1, theta_1, phi_1, t_b
        8 model parameters

    Returns
    -------
    lp : float
        Natural log of the prior
    """

    M1, M2, A, ecc, v_k_1, theta_1, phi_1, t_b = y


    lp = 0.0

    # P(M1)
    if M1 < c.min_mass: return -np.inf
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
    norm_const = 1.0 / (np.log(c.max_A) - np.log(c.min_A))
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

    # P(t_b) - formally, this is an improper prior
    if t_b < 0.0: return -np.inf
    if t_b > 1000.0: return -np.inf  # This may need to change in the future

    return lp



def ln_posterior_population(x):
    """ Calculate the natural log of the posterior probability

    Parameters
    ----------
    x : M1, M2, A, ecc, v_k_1, theta_1, phi_1, t_b
        8 model parameters

    Returns
    -------
    lp : float
        Natural log of the posterior probability
    """

    M1, M2, A, ecc, v_k_1, theta_1, phi_1, t_b = x


    metallicity = 0.008

    # Call priors
    lp = ln_priors_population(x)
    if np.isinf(lp): return -np.inf


    # Run binary_c evolution
    orbital_period = A_to_P(M1, M2, A)
    output = binary_c.run_binary(M1, M2, orbital_period, ecc, metallicity, t_b, v_k_1, theta_1, phi_1, v_k_1, theta_1, phi_1, 0, 0)
    # m1_out, m2_out, A_out, ecc_out, v_sys, L_x, t_SN1, t_SN2, t_cur, k1, k2, comenv_count, evol_hist = output


    # Check to see if we've formed an HMXB
    if check_output(output, binary_type='HMXB'):
        return lp

    return -np.inf



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
    p0 = np.zeros((nwalkers,8))
    p0 = set_walkers(nwalkers=nwalkers)
    print "...walkers are set"

    # Define sampler
    if mpi == True:
        pool = MPIPool()
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        sampler = emcee.EnsembleSampler(nwalkers=nwalkers, dim=8, lnpostfn=posterior_function, pool=pool)

    elif threads != 1:
        sampler = emcee.EnsembleSampler(nwalkers=nwalkers, dim=8, lnpostfn=posterior_function, threads=threads)
    else:
        sampler = emcee.EnsembleSampler(nwalkers=nwalkers, dim=8, lnpostfn=posterior_function)

    # Burn-in
    print "Starting burn-in..."
    pos,prob,state = sampler.run_mcmc(p0, N=nburn)
    print "...finished running burn-in"

    # Full run
    print "Starting full run..."
    sampler.reset()
    pos,prob,state = sampler.run_mcmc(pos, N=nsteps)
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
    m1 = 12.0
    m2 = 10.0
    eccentricity = 0.41
    metallicity = 0.008
    orbital_period = 500.0
    time = 30.0

    # SN kicks
    v_k_1 = 0.0
    theta_1 = 0.0
    phi_1 = 0.0

    evol_flag = 0  # Have evolutionary history as an output
    dco_flag = 0  # Stop evolution when two compact objects are formed


    # Iterate randomly through initial conditions until a viable parameter set is found
    for i in np.arange(1000000):

        m1 = 5.0 * np.random.uniform(size=1) + 8.0
        m2 = m1 * (0.2 * np.random.uniform(size=1) + 0.8)
        A = 300.0 * np.random.uniform(size=1) + 20.0
        eccentricity = np.random.uniform(size=1)

        v_k_1 = 300.0 * np.random.uniform(size=1) + 20.0
        theta_1 = np.pi * np.random.uniform(size=1)
        phi_1 = np.pi * np.random.uniform(size=1)
        time = 40.0 * np.random.uniform(size=1) + 10.0

        x = m1, m2, A, eccentricity, v_k_1, theta_1, phi_1, time

        # print "trial:", x, ln_posterior_population(x)

        # If the system has a viable posterior probability
        if ln_posterior_population(x) > -10000.0:

            # ...then use it as our starting system
            break


    if i==99999:
        print "Walkers could not be set"
        sys.exit(-1)



    # Now to generate a ball around these parameters

    # Binary parameters
    m1_set = m1 + np.random.normal(0.0, 0.1, nwalkers)
    m2_set = m2 + np.random.normal(0.0, 0.1, nwalkers)
    e_set = eccentricity + np.random.normal(0.0, 0.01, nwalkers)
    A_set = A + np.random.normal(0.0, 2.0, nwalkers)

    # SN kick perameters
    v_k_1_set = v_k_1 + np.random.normal(0.0, 1.0, nwalkers)
    theta_1_set = theta_1 + np.random.normal(0.0, 0.01, nwalkers)
    phi_1_set = phi_1 + np.random.normal(0.0, 0.01, nwalkers)

    # Birth time
    time_set = time + np.random.normal(0.0, 0.2, nwalkers)


    # Check if any of these have posteriors with -infinity
    for i in np.arange(nwalkers):


        p = m1_set[i], m2_set[i], A_set[i], e_set[i], \
                v_k_1_set[i], theta_1_set[i], phi_1_set[i], time_set[i]
        ln_posterior = ln_posterior_population(p)


        while ln_posterior < -10000.0:


            # Binary parameters
            m1_set[i] = m1 + np.random.normal(0.0, 0.1, 1)
            m2_set[i] = m2 + np.random.normal(0.0, 0.1, 1)
            e_set[i] = eccentricity + np.random.normal(0.0, 0.01, 1)
            A_set[i] = A + np.random.normal(0.0, 2.0, 1)

            # SN kick perameters
            v_k_1_set[i] = v_k_1 + np.random.normal(0.0, 1.0, 1)
            theta_1_set[i] = theta_1 + np.random.normal(0.0, 0.01, 1)
            phi_1_set[i] = phi_1 + np.random.normal(0.0, 0.01, 1)

            # Birth time
            time_set[i] = time + np.random.normal(0.0, 0.2, 1)


            p = m1_set[i], m2_set[i], A_set[i], e_set[i], \
                    v_k_1_set[i], theta_1_set[i], phi_1_set[i], time_set[i]

            ln_posterior = ln_posterior_population(p)



    # Save and return the walker positions
    p0 = np.zeros((nwalkers,8))

    p0[:,0] = m1_set
    p0[:,1] = m2_set
    p0[:,2] = A_set
    p0[:,3] = e_set
    p0[:,4] = v_k_1_set
    p0[:,5] = theta_1_set
    p0[:,6] = phi_1_set
    p0[:,7] = time_set

    return p0



def check_output(output, binary_type='HMXB'):
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


    if binary_type != "HMXB":
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
