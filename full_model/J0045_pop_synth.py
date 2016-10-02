# Run J0045-7319
from src.core import *
set_data_path("../data")

from astropy.coordinates import SkyCoord
from astropy import units as u
import pickle

from xrb.pop_synth import pop_synth




coor_J0045 = SkyCoord('00h45m35.26s', '-73d19m03.32s')

ra_J0045 = coor_J0045.ra.degree
dec_J0045 = coor_J0045.dec.degree
M2_d_J0045 = 11.0  # M2 in Msun
P_orb_J0045 = 51.17  # P_orb in days
ecc_J0045 = 0.808  # eccentricity
J0045 = ra_J0045, dec_J0045, P_orb_J0045, ecc_J0045, M2_d_J0045

start_time = time.time()

HMXB_J0045, init_params_J0045 = pop_synth.run_pop_synth(J0045, N_sys=1000000)

print "Population Synthesis ran 2000000 binaries in", time.time()-start_time, "seconds"

pickle.dump( init_params_J0045, open( INDATA("J0045_pop_synth_init_conds.obj"), "wb" ) )
pickle.dump( HMXB_J0045, open( INDATA("J0045_pop_synth_HMXB.obj"), "wb" ) )
