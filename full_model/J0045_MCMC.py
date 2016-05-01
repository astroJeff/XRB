# Run J0045-7319

import sys
from astropy.coordinates import SkyCoord
from astropy import units as u
import pickle
import time

sys.path.append('../stats')
import stats



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

# Record time
start_time = time.time()

sampler = stats.run_emcee(M2_d_J0045, P_orb_J0045, ecc_J0045, ra_J0045, dec_J0045, \
    M2_d_err=M2_d_J0045_err, P_orb_obs_err=P_orb_J0045_err, ecc_obs_err=ecc_J0045_err, \
    nburn=10000, nsteps=50000)

print "Simulation took", time.time()-start_time, "seconds"

pickle.dump( sampler, open( "../data/J0045_MCMC_sampler.obj", "wb" ) )
