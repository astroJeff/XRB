# Run J0045-7319
from src.core import *

from astropy.coordinates import SkyCoord
from astropy import units as u
import pickle

from xrb.pop_synth import pop_synth
from xrb.binary import binary_evolve



# Start with initial binary conditions for our test system
M1 = 13.0
M2 = 10.0
A = 100.0
ecc = 0.3
v_k = 250.0
theta = 2.9
phi = 0.9
t_b = 22.0

# Now, run full evolution
M1_obs, M2_obs, L_x_obs, v_sys_obs, M2_dot_obs, A_obs, ecc_obs, theta_obs, k_type \
    = pop_synth.full_forward(M1,M2,A,ecc,v_k,theta,phi,t_b)


# Now, define system observations
ra_obs = 13.5
dec_obs = -72.63
P_obs = binary_evolve.A_to_P(M1_obs, M2_obs, A_obs)

sys1 = ra_obs, dec_obs, P_obs, ecc_obs, M2_obs


start_time = time.time()

HMXB, init_params = pop_synth.run_pop_synth(sys1, N_sys=1000000)

print "Population Synthesis ran 1000000 binaries in", time.time()-start_time, "seconds"

pickle.dump( init_params, open( "../data/sys1_pop_synth_init_conds.obj", "wb" ) )
pickle.dump( HMXB, open( "../data/sys1_pop_synth_HMXB.obj", "wb" ) )
