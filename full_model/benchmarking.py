import numpy as np

import xrb
from xrb.src.core import *
set_data_path("../data")
from xrb.binary import load_sse, binary_evolve
import xrb.src.constants as c
from xrb.pop_synth import pop_synth
from xrb.SF_history import sf_history

from astropy.coordinates import SkyCoord

load_sse.load_sse()
sf_history.load_sf_history()





# Set J0045 parameters
coor_J0045 = SkyCoord('00h45m35.26s', '-73d19m03.32s')

ra_J0045 = coor_J0045.ra.degree
dec_J0045 = coor_J0045.dec.degree
M2_d_J0045 = 8.8  # M2 in Msun
M2_d_J0045_err = 1.8
P_orb_J0045 = 51.17  # P_orb in days
P_orb_J0045_err = 1.0
ecc_J0045 = 0.808  # eccentricity
ecc_J0045_err = 0.05

t_min, t_max = 5.0, 60.0
N_times = (t_max-t_min) * 10
N = 10

N_good = 0
N_good_J0045 = 0
for t_b in np.arange(N_times)/10.0+t_min:

    HMXB, init_params = pop_synth.create_HMXBs(t_b, N_sys=N)
    M1 = init_params['M1']
    ecc = HMXB['ecc']


    theta_from_J0045 = sf_history.get_theta_proj_degree(ra_J0045, dec_J0045, init_params['ra_b'], init_params['dec_b'])

    fwd_M_NS, fwd_M2, fwd_L_x, fwd_v_sys, fwd_M2_dot, fwd_A, fwd_ecc, fwd_theta, fwd_k = \
        pop_synth.full_forward(init_params['M1'], init_params['M2'], init_params['A'], \
                 init_params['ecc'], init_params['v_k'], init_params['theta'], \
                 init_params['phi'], t_b)

    v_sys = fwd_v_sys

    P_orb = binary_evolve.A_to_P(fwd_M_NS, fwd_M2, fwd_A)


    theta_max = c.rad_to_deg * (t_b - load_sse.func_sse_tmax(M1)) * v_sys / c.dist_SMC * c.yr_to_sec * 1.0e6


    # Now, select the systems similar to J0045
    for i in np.arange(len(HMXB)):
        if theta_from_J0045[i] < theta_max[i] and \
           np.abs(M2_d_J0045 - fwd_M2[i]) < 3.0*M2_d_J0045_err and \
           np.abs(P_orb_J0045 - P_orb[i]) < 3.0*P_orb_J0045_err and \
           np.abs(ecc_J0045 - ecc[i]) < 3.0*ecc_J0045_err:

            N_good_J0045 = N_good_J0045 + 1

    N_good = N_good + len(HMXB)

print "Found", N_good, "HMXBs out of", float(N) * float(N_times), "binaries"
print "Found", N_good_J0045, "HMXBs similar to J0045"
