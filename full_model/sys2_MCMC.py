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
M1 = 9.0
M2 = 4.0
A = 500.0
ecc = 0.5
v_k = 100.0
theta = 2.7
phi = 1.2
t_b = 50.0

# Now, run full evolution
M1_obs, M2_obs, L_x_obs, v_sys_obs, M2_dot_obs, A_obs, ecc_obs, theta_obs \
    = pop_synth.full_forward(M1,M2,A,ecc,v_k,theta,phi,t_b)

# Let's put some uncertainties on those observations
M2_d_err = 1.0
P_orb_obs_err = 1.0
ecc_obs_err = 0.05

# Now, define system observations
ra_obs = 15.9
dec_obs = -72.25
P_obs = binary_evolve.A_to_P(M1_obs, M2_obs, A_obs)

# Record time
start_time = time.time()

# Run sampler
sampler = stats.run_emcee(M2_obs, P_obs, ecc_obs, ra_obs, dec_obs, \
    M2_d_err=M2_d_err, P_orb_obs_err=P_orb_obs_err, ecc_obs_err=ecc_obs_err, \
    nburn=10000, nsteps=50000)

print "Simulation took", time.time()-start_time, "seconds"

# Save samples
pickle.dump( sampler, open( "../data/sys2_test_MCMC_sampler.obj", "wb" ) )
