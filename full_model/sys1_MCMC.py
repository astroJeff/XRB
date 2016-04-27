# Run J0045-7319

import sys
from astropy.coordinates import SkyCoord
from astropy import units as u
import pickle
import time

sys.path.append('../stats')
import stats
sys.path.append('../pop_synth')
import pop_synth
sys.path.append('../binary')
import binary_evolve



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
M1_obs, M2_obs, L_x_obs, v_sys_obs, M2_dot_obs, A_obs, ecc_obs, theta_obs \
    = pop_synth.full_forward(M1,M2,A,ecc,v_k,theta,phi,t_b)


# Now, define system observations
ra_obs = 13.5
dec_obs = -72.6
P_obs = binary_evolve.A_to_P(M1_obs, M2_obs, A_obs)

# Record time
start_time = time.time()

# Run sampler
sampler = stats.run_emcee(M2_obs, P_obs, ecc_obs, ra_obs, dec_obs, nburn=10000, nsteps=250000)

print "Simulation took", time.time()-start_time, "seconds"

# Save samples
pickle.dump( sampler, open( "../data/sys1_test_MCMC_sampler.obj", "wb" ) )
