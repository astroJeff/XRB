# Run J0045-7319

import sys
from astropy.coordinates import SkyCoord
from astropy import units as u
import pickle

sys.path.append('../stats')
import stats




coor_J0045 = SkyCoord('00h45m35.26s', '-73d19m03.32s')

ra_J0045 = coor_J0045.ra.degree
dec_J0045 = coor_J0045.dec.degree
M2_d_J0045 = 11.0  # M2 in Msun
P_orb_J0045 = 51.17  # P_orb in days
ecc_J0045 = 0.808  # eccentricity

sampler = stats.run_emcee(M2_d_J0045, P_orb_J0045, ecc_J0045, ra_J0045, dec_J0045, nburn=10000, nsteps=500000)

pickle.dump( sampler, open( "../data/J0045_MCMC_sampler.obj", "wb" ) )
