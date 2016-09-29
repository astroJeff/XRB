# Run test system 1
from src.core import *

from astropy.coordinates import SkyCoord
from astropy import units as u
import pickle
import matplotlib.pyplot as plt
import emcee
from emcee.utils import MPIPool
import corner
import copy

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






mpi = True
nwalkers = 80
nburn = 5
nsteps = 10


# First thing is to load the sse data and SF_history data
load_sse.load_sse()
sf_history.load_sf_history()


# Define sampler
args = [[M2_obs, M2_d_err, P_obs, P_orb_obs_err, ecc_obs, ecc_obs_err, ra_obs, dec_obs]]

if mpi == True:
    pool = MPIPool()
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

    # Get initial values
    initial_vals = stats.get_initial_values(M2_obs, nwalkers=nwalkers)
    sampler = emcee.EnsembleSampler(nwalkers=nwalkers, dim=10, lnpostfn=stats.ln_posterior, args=args, pool=pool)

elif threads != 1:
    # Get initial values
    initial_vals = stats.get_initial_values(M2_obs, nwalkers=nwalkers)
    sampler = emcee.EnsembleSampler(nwalkers=nwalkers, dim=10, lnpostfn=stats.ln_posterior, args=args, threads=threads)
else:
    # Get initial values
    initial_vals = stats.get_initial_values(M2_obs, nwalkers=nwalkers)
    sampler = emcee.EnsembleSampler(nwalkers=nwalkers, dim=10, lnpostfn=stats.ln_posterior, args=args)

# Assign initial values
p0 = np.zeros((nwalkers,10))
p0 = stats.set_walkers(initial_vals, args[0], nwalkers=nwalkers)

# Burn-in 1
pos,prob,state = sampler.run_mcmc(p0, N=nburn)
sampler1 = copy.copy(sampler)

# TESTING BEGIN - Get limiting ln_prob for worst 8 chains
prob_lim = (np.sort(prob)[7] + np.sort(prob)[8])/2.0
index_best = np.argmax(prob)
for i in np.arange(len(prob)):
    if prob[i] < prob_lim:  pos[i] = np.copy(pos[index_best]) + np.random.normal(0.0, 0.005, size=10)
# TESTING END

print "Burn-in 1 finished."
print "Starting burn-in 2..."

# Burn-in 2
sampler.reset()
pos,prob,state = sampler.run_mcmc(pos, N=nburn)
sampler2 = copy.copy(sampler)

# TESTING BEGIN - Get limiting ln_prob for worst 8 chains
prob_lim = (np.sort(prob)[7] + np.sort(prob)[8])/2.0
index_best = np.argmax(prob)
for i in np.arange(len(prob)):
    if prob[i] < prob_lim:  pos[i] = np.copy(pos[index_best]) + np.random.normal(0.0, 0.005, size=10)
# TESTING END

print "Burn-in 2 finished."
print "Starting burn-in 3..."

# Burn-in 3
sampler.reset()
pos,prob,state = sampler.run_mcmc(pos, N=nburn)
sampler3 = copy.copy(sampler)

# TESTING BEGIN - Get limiting ln_prob for worst 8 chains
prob_lim = (np.sort(prob)[7] + np.sort(prob)[8])/2.0
index_best = np.argmax(prob)
for i in np.arange(len(prob)):
    if prob[i] < prob_lim:  pos[i] = np.copy(pos[index_best]) + np.random.normal(0.0, 0.005, size=10)
# TESTING END

print "Burn-in 3 finished."
print "Starting burn-in 4..."

# Burn-in 4
sampler.reset()
pos,prob,state = sampler.run_mcmc(pos, N=nburn)
sampler4 = copy.copy(sampler)

print "Burn-in 4 finished."
print "Starting production run..."

# Full run
sampler.reset()
pos,prob,state = sampler.run_mcmc(pos, N=nsteps)

print "Finished production run"

if mpi is True: pool.close()












#
# sampler1, sampler2, sampler3, sampler4, sampler = stats.run_emcee_2(M2_obs, P_obs, ecc_obs, ra_obs, dec_obs, \
#     M2_d_err=M2_d_err, P_orb_obs_err=P_orb_obs_err, ecc_obs_err=ecc_obs_err, \
#     nburn=5, nsteps=10, mpi=True)


print "Simulation took", time.time()-start_time, "seconds"
