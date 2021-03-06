from xrb.src.core import *

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from xrb.binary import load_sse




# To Do: Check for thermal timescale MT criterion
# To Do: What happens when companion's lifetime falls between primary's MS lifetime and stellar lifetime?
# To Do: Remove RLOF systems



def func_MT_forward(M_1_in, M_2_in, A_in, ecc_in, beta=1.0):
    """ Evolve a binary through thermal timescale mass transfer.
    It is assumed that the binary (instantaneously) circularizes
    at the pericenter distance.

    Parameters
    ----------
    M_1_in : float
        Primary mass input (Msun)
    M_2_in : float
        Secondary mass input (Msun)
    A_in : float
        Orbital separation (any unit)
    ecc_in : float
        Eccentricity (unitless)
    beta : float
        Mass transfer efficiency

    Returns
    -------
    M_1_out : float
        Primary mass output (Msun)
    M_2_out : float
        Secondary mass output (Msun)
    A_out : float
        Orbital separation output (any unit)
    """

    M_1_out = load_sse.func_sse_he_mass(M_1_in)
    M_2_out = M_2_in + beta * (M_1_in - M_1_out)


    # Mass is lost with specific angular momentum of donor (M1)
    alpha = (M_2_out / (M_1_out + M_2_out))**2


    # Old formula for conservative MT: A_out = A_in * (1.0-ecc_in) * (M_1_in*M_2_in/M_1_out/M_2_out)**2
    # Allowing for non-conservative MT (also works for conservative MT)
    C_1 = 2.0 * alpha * (1.0-beta) - 2.0
    C_2 = -2.0 * alpha / beta * (1.0 - beta) - 2.0
    A_out = A_in * (1.0-ecc_in) * (M_1_out+M_2_out)/(M_1_in+M_2_in) * (M_1_out/M_1_in)**C_1 * (M_2_out/M_2_in)**C_2

    ##################################
    # NOTE: Technically, the above equation only works for a fixed alpha. If
    # Alpha is indeed varying as mass is lost, then the above equation needs
    # to be adjusted. This is probably solved elsewhere in the literature.
    ##################################

    # Make sure systems don't overfill their Roche lobes
#    r_1_max = load_sse.func_sse_r_MS_max(M_1_out)
#    r_1_roche = func_Roche_radius(M_1_in, M_2_in, A_in*(1.0-ecc_in))
#    r_2_max = load_sse.func_sse_r_MS_max(M_2_out)
#    r_2_roche = func_Roche_radius(M_2_in, M_1_in, A_in)


    ##### TESTING #####
    # # Get the k-type when the primary overfills its Roche lobe
    # if isinstance(M_1_in, np.ndarray):
    #     k_RLOF = load_sse.func_sse_get_k_from_r(M_1_in, r_1_roche)
    # else:
    #     k_RLOF = load_sse.func_sse_get_k_from_r(np.asarray([M_1_in]), np.asarray([r_1_roche]))
    #
    #
    # # Only systems with k_RLOF = 2, 4 survive
    # if isinstance(A_out, np.ndarray):
    #     idx = np.intersect1d(np.where(k_RLOF!=2), np.where(k_RLOF!=4))
    #     A_out[idx] = -1.0
    # else:
    #     if (k_RLOF != 2) & (k_RLOF != 4): A_out = -1.0
    ##### TESTING #####

    # Post-MT, Pre-SN evolution
    M2_He_in = M_2_out
    M2_He_out = M2_He_in
    M1_He_in = M_1_out
    M1_He_out = load_sse.func_sse_he_star_final(M_1_out)
    A_He_in = A_out
    A_He_out = A_He_in * (M1_He_in + M2_He_in) / (M1_He_out + M2_He_out)


#    return M_1_out, M_2_out, A_out
    return M1_He_out, M2_He_out, A_He_out


def func_common_envelope(M1_in, M2_in, A_in, ecc_in, alpha_CE=1.0, lambda_struc=1.0):
    """ Calculate orbital evolution due to a common envelope. We use the envelope
    binding energy approximation from De Marco et al. (2011). This will hopefully
    be updated soon to account for actual stellar structure models, which will
    include internal energy as well. Obviously reality is much more complex.



    """

    # M_1 is determined as the core of the primary
    M_1_out = load_sse.func_sse_he_mass(M_1_in)
    # M_2 accretes nothing
    M_2_out = M_2_in

    # r_1_roche is at its roche radius when the primary overfills its Roche lobe, entering instability
    r_1_roche = func_Roche_radius(M_1_in, M_2_in, A_in*(1.0-ecc_in))

    # Envelope mass
    M_env = M_1_in - M_1_out

    # alpha-lambda prescription
    E_binding = -c.G * M_env * (M_env/2.0 + M_1_out) / (lambda_struc * r_1_roche)
    E_orb_in = -c.G * M_1_in * M_2_in / (2.0 * A_in)
    E_orb_out = E_orb_in + (1.0/alpha_CE) * E_binding

    A_out = -c.G * M_1_out * M_2_out / (2.0 * E_orb_out)

    return M_1_out, M_2_out, A_out



def func_Roche_radius(M1, M2, A):
    """ Get Roche lobe radius (Eggleton 1983)

    Parameters
    ----------
    M1 : float
        Primary mass (Msun)
    M2 : float
        Secondary mass (Msun)
    A : float
        Orbital separation (any unit)

    Returns
    -------
    Roche radius : float
        in units of input, A
    """
    q = M1 / M2
    return A * 0.49*q**(2.0/3.0) / (0.6*q**(2.0/3.0) + np.log(1.0 + q**(1.0/3.0)))


def P_to_A(M1, M2, P):
    """ Orbital period (days) to separation (Rsun) """
    mu = c.G * (M1 + M2) * c.Msun_to_g
    n = 2.0*np.pi / P / c.day_to_sec
    A = np.power(mu/(n*n), 1.0/3.0) / c.Rsun_to_cm
    return A

def A_to_P(M1, M2, A):
    """ Orbital separation (Rsun) to period (days) """
    mu = c.G * (M1 + M2) * c.Msun_to_g
    n = np.sqrt(mu/(A**3 * c.Rsun_to_cm**3))
    P = 2.0*np.pi / n
    return P / c.day_to_sec


def func_SN_forward(M_1_in, M_2, A_in, v_k, theta, phi):
    """ Evolve a binary through a supernova

    Parameters
    ----------
    M_1_in : float
        Primary mass input (Msun)
    M_2 : float
        Secondary mass (Msun)
    A_in : float
        Orbital separation input (Rsun)
    v_k : float
        Supernova kick velocity (km/s)
    theta : float
        Supernova kick direction polar angle (radians)
    phi : float
        Supernova kick direction azimuthal angle (radians)

    Returns
    -------
    A_out : float
        Orbital separation output (Rsun)
    v_sys : float
        Systemic velocity (km/s)
    ecc : float
        Eccentricity output (unitless)
    """

    if isinstance(A_in, np.ndarray):
        A_in[np.where(A_in<=0.0)] = 1.0e-50
    else:
        if A_in<=0.0: A_in = 1.0e-50

    v_r = np.sqrt(c.GGG*(M_1_in + M_2)/A_in)
    v_1 = np.sqrt(2.0*v_k*v_r*np.cos(theta) + v_k*v_k + v_r*v_r)

    A_out = 1.0 / (2.0/A_in - v_1*v_1/(c.GGG*(c.M_NS+M_2)))
#    v_sys = (M_NS / (M_NS + M_2)) * v_1

    # Systemic velocity
    alpha = (M_1_in / (M_1_in + M_2))
    beta = (c.M_NS / (c.M_NS + M_2))

    v_sys = beta*beta*v_k*v_k
    v_sys = v_sys + v_r*v_r*(beta-alpha)*(beta-alpha)
    v_sys = v_sys + 2.0*beta*v_k*v_r*np.cos(theta)*(beta-alpha)
    v_sys = np.sqrt(v_sys)

    # Eccentricity
    e_tmp = v_k*v_k*np.cos(theta)*np.cos(theta)
    e_tmp = e_tmp + v_k*v_k*np.sin(theta)*np.sin(theta)*np.sin(phi)*np.sin(phi)
    e_tmp = e_tmp + 2.0*v_k*v_r*np.cos(theta)
    e_tmp = e_tmp + v_r*v_r
    e_tmp = 1.0 - (A_in*A_in)/(A_out*c.GGG*(c.M_NS+M_2)) * e_tmp

    if isinstance(e_tmp, np.ndarray):

        ecc = np.sqrt(e_tmp)
        ecc[np.where(e_tmp < 0.0)] = -1.0
        ecc[np.where(e_tmp > 1.0)] = -1.0
        ecc[np.where(M_2 < c.min_mass)] = -1.0
        ecc[np.where(A_in < 1.0e-10)] = -1.0
#        ecc = np.array([np.sqrt(x) if x > 0.0 or M_2 > min_mass or A_in>1.0e-10 else -1.0 for x in e_tmp])
    else:
        if e_tmp < 0.0 or M_2 < c.min_mass or A_in < 1.0e-10: return A_out, v_sys, -1.0
        ecc = np.sqrt(e_tmp)

    return A_out, v_sys, ecc




def func_get_time(M1, M2, t_obs):
    """ Get the adjusted time for a secondary that accreted
    the primary's envelope in thermal timescale MT

    parameters
    ----------
    M1 : float
        Primary mass before mass transfer (Msun)
    M2 : float
        Secondary mass before mass transfer (Msun)
    t_obs : float
        Observation time (Myr)

    Returns
    -------
    Effective observed time: float
        Time to be fed into load_sse.py function func_get_sse_star() (Myr)
    """

    t_lifetime_1 = load_sse.func_sse_ms_time(M1)
    he_mass_1 = load_sse.func_sse_he_mass(M1)

    t_lifetime_2 = load_sse.func_sse_ms_time(M2)
    he_mass_2 = load_sse.func_sse_he_mass(M2)

    # Relative lifetime through star 2 at mass gain
    he_mass = t_lifetime_1/t_lifetime_2 * he_mass_2

    # Get new secondary parameters
    mass_new = M2 + M1 - he_mass_1
    t_lifetime_new = load_sse.func_sse_ms_time(mass_new)
    he_mass_new = load_sse.func_sse_he_mass(mass_new)

    # New, effective lifetime
    t_eff = he_mass / he_mass_new * t_lifetime_new

    # Now, we obtain the "effective observed time"
    return t_eff + t_obs - t_lifetime_1



def get_v_wind(mass, radius):
    """ Stellar wind velocity at infinity

    Parameters
    ----------
    mass : float
        Current stellar mass (Msun)
    radius : float
        Current stellar radius (Rsun)

    Returns
    -------
    v_wind : float
        Wind velocity at infinity (km/s)

    """

    slope = (7.0 - 0.5) / (120.0 - 1.4)
    intercept = 7.0 - 120.0 * slope

    beta = slope * mass + intercept


    # For systems with radius = 0.0, set wind arbitrarily high
    if isinstance(radius, np.ndarray):
        one_over_radius = 1.0e50 * np.ones(len(radius))
        one_over_radius[np.where(radius>0.0)] = 1.0 / radius[np.where(radius>0.0)]
    else:
        one_over_radius = 1.0e50
        if radius>0.0: one_over_radius = 1.0/radius

    return np.sqrt(2.0 * beta * c.GGG * mass * one_over_radius)



def func_Lx_forward(M_1_a, M_2_a, M_2_in, A_in, ecc_in, t_obs):
    """ Calculate the X-ray luminosity from accretion for a binary

    Parameters
    ----------
    M_1_a : float
        ZAMS mass of the primary, now a NS (Msun)
    M_2_a : float
        ZAMS mass of the secondary (Msun)
    M_2_in : float
        Post-mass transfer mass of the secondary (Msun)
    A_in : float
        Post-SN orbital separation (Rsun)
    ecc_in : float
        Post-SN eccentricity (unitless)
    t_obs : float
        Time which the binary is being observed

    Returns
    -------
    M_2_out : float
        Current mass of the secondary
    L_x : float
        X_ray luminosity
    M_dot_out : float
        Mass accretion rate
    A_out : float
        Current orbital separation
    """

    t_eff_obs = func_get_time(M_1_a, M_2_a, t_obs)

    if isinstance(t_eff_obs, np.ndarray):
        M_2_out = np.array([])
        M_dot_wind = np.array([])
        R_out = np.array([])
        k_out = np.array([])
        for i in np.arange(len(t_eff_obs)):
            if (t_eff_obs[i] < 0.0 or ecc_in[i] < 0.0 or ecc_in[i] >= 1.0):
                ecc_in[i] = 0.0
                if isinstance(M_2_in, np.ndarray):
                    M_2_out = np.append(M_2_out, M_2_in[i])
                else:
                    M_2_out = np.append(M_2_out, M_2_in)
                M_dot_wind = np.append(M_dot_wind, 0.0)
                R_out = np.append(R_out, 0.0)
            else:
                if isinstance(M_2_in, np.ndarray):
                    if M_2_in[i] > c.max_mass:
                        aa, bb, cc = 0.0, 0.0, 0.0
                    else:
                        aa, bb, cc, dd = load_sse.func_get_sse_star(M_2_in[i], t_eff_obs[i])
                else:
                    if M_2_in > c.max_mass:
                        aa, bb, cc = 0.0, 0.0, 0.0
                    else:
                        aa, bb, cc, dd = load_sse.func_get_sse_star(M_2_in, t_eff_obs[i])

                M_2_out = np.append(M_2_out, aa)
                M_dot_wind = np.append(M_dot_wind, bb)
                R_out = np.append(R_out, cc)
                k_out = np.append(k_out, dd)
    else:
        if (t_eff_obs < 0.0 or M_2_in > c.max_mass or ecc_in < 0.0 or ecc_in > 1.0 or t_eff_obs > load_sse.func_sse_tmax(M_2_in)):
            M_2_out = M_2_in
            M_dot_wind = 0.0
            R_out = 0.0
            ecc_in = 0.0
        else:
            M_2_out, M_dot_wind, R_out, k_out = load_sse.func_get_sse_star(M_2_in, t_eff_obs)

    # Get wind velocity
    v_wind = get_v_wind(M_2_out, R_out)
    if isinstance(v_wind, np.ndarray):
        v_wind[np.where(v_wind <= 0.0)] = 1.0e50 # To eliminate "bad" winds
    else:
        if v_wind <= 0.0: v_wind = 1.0e50

    # Get final orbital separation
    if isinstance(A_in, np.ndarray):
        A_in[np.where(A_in <= 0.0)] = 1.0e50 # To eliminate "bad" separations
    else:
        if A_in <= 0.0: A_in = 1.0e50
    A_out = (c.M_NS + M_2_in) / (c.M_NS + M_2_out) * A_in

    # Capture fraction takes into account eccentricity
    f_capture = (c.GGG*c.M_NS / (v_wind*v_wind*A_out))**2 / np.sqrt(1.0 - ecc_in**2)
    M_dot_out = f_capture * M_dot_wind

    L_bol = c.GGG * c.M_NS * M_dot_out / c.R_NS * c.km_to_cm * c.Msun_to_g * c.Rsun_to_cm / c.yr_to_sec
    L_x = c.eta_bol * L_bol

    return M_2_out, L_x, M_dot_out, A_out
