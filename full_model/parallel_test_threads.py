# Run test system 1
from src.core import *

from astropy.coordinates import SkyCoord
from astropy import units as u
import pickle
import matplotlib.pyplot as plt
import emcee
import corner

import stats
import pop_synth
import binary_evolve



# Start with initial binary conditions for our test system
M1 = 13.0
M2 = 10.0
A = 150.0
ecc = 0.7
v_k = 250.0
theta = 2.9
phi = 0.9
t_b = 22.0


# Now, run full evolution
M1_obs, M2_obs, L_x_obs, v_sys_obs, M2_dot_obs, A_obs, ecc_obs, theta_obs \
    = pop_synth.full_forward(M1,M2,A,ecc,v_k,theta,phi,t_b)

# Let's put some uncertainties on those observations
M2_d_err = 1.0
P_orb_obs_err = 1.0
ecc_obs_err = 0.05

# Now, define system observations
ra_obs = 13.5
dec_obs = -72.63
P_obs = binary_evolve.A_to_P(M1_obs, M2_obs, A_obs)

# Birth position
ra_b = 13.51
dec_b = -72.7

# Truths
truths=[M1, M2, A, ecc, v_k, theta, phi, ra_b, dec_b, t_b]


############## Print prior, posterior probabilities ##############
x = M1, M2, A, ecc, v_k, theta, phi, ra_b, dec_b, t_b
y = ra_obs, dec_obs, M1, M2, A, ecc, v_k, theta, phi, ra_b, dec_b, t_b
args = M2_obs, M2_d_err, P_obs, P_orb_obs_err, ecc_obs, ecc_obs_err, ra_obs, dec_obs

prior_truths = stats.ln_priors(y)
posterior_truths = stats.ln_posterior(x, args)
print "Prior:", prior_truths
print "Posterior:", posterior_truths



############## Run sampler ###################
start_time = time.time()

sampler1, sampler2, sampler3, sampler4, sampler = stats.run_emcee_2(M2_obs, P_obs, ecc_obs, ra_obs, dec_obs, \
    M2_d_err=M2_d_err, P_orb_obs_err=P_orb_obs_err, ecc_obs_err=ecc_obs_err, \
    nburn=5, nsteps=10, threads=1)

print "Simulation took", time.time()-start_time, "seconds"
