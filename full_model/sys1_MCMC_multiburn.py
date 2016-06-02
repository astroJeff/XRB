# Run test system 1

import sys
from astropy.coordinates import SkyCoord
from astropy import units as u
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner

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

sampler1, sampler2, sampler3 = stats.run_emcee_2(M2_obs, P_obs, ecc_obs, ra_obs, dec_obs, \
    M2_d_err=M2_d_err, P_orb_obs_err=P_orb_obs_err, ecc_obs_err=ecc_obs_err, \
    nburn=5000, nsteps=10000)

print "Simulation took", time.time()-start_time, "seconds"



############## Corner pyplot ###################
labels = [r"$M_1$", r"$M_2$", r"$A$", r"$e$", r"$v_k$", r"$\theta$", r"$\phi$", r"$\alpha_{\rm b}$", r"$\delta_{\rm b}$", r"$t_{\rm b}$"]
fig = corner.corner(sampler3.flatchain, labels=labels, truths=truths)
plt.rc('font', size=18)
plt.savefig('../figures/sys1_corner_multiburn.pdf')
plt.rc('font', size=10)




################# Posterior plots ##################
plt.rc('font', size=8)
fig, ax = plt.subplots(2,5, figsize=(14,6))

for i in np.arange(10):
    a = np.int(i/5)
    b = i%5

    xmin = np.min(sampler3.chain[:,:,i])
    xmax = np.max(sampler3.chain[:,:,i])
    corner.hist2d(sampler3.chain[:,:,i], sampler3.lnprobability, ax=ax[a,b], bins=30, range=((xmin,xmax),(-50,0)))
    ax[a,b].set_xlabel(labels[i])

    ax[a,b].set_ylim(-30,0)
    ax[a,b].axhline(posterior_truths)
    ax[a,b].axvline(truths[i], color='r')

plt.tight_layout()
plt.savefig('../figures/sys1_posterior_params_multiburn.pdf')


################# Chains plot #####################
fig, ax = plt.subplots(sampler1.dim, 3, sharex=False, figsize=(16.0,20.0))
for i in range(sampler1.dim):
    for j in np.arange(len(sampler1.chain[...])):

        chain1 = sampler1.chain[...,i][j]
        ax[i,0].plot(chain1, alpha=0.25, color='k', drawstyle='steps')

        chain2 = sampler2.chain[...,i][j]
        ax[i,1].plot(chain2, alpha=0.25, color='k', drawstyle='steps')

        chain3 = sampler3.chain[...,i][j]
        ax[i,2].plot(chain3, alpha=0.25, color='k', drawstyle='steps')

    # Remove tick labels from y-axis
    ax[i,1].set_yticklabels([])
    ax[i,2].set_yticklabels([])

# Make all plots have the same y-range - plots must already be created
for i in range(sampler1.dim):
    ymin = min(ax[i,0].get_ylim()[0],ax[i,1].get_ylim()[0],ax[i,2].get_ylim()[0])
    ymax = max(ax[i,0].get_ylim()[1],ax[i,1].get_ylim()[1],ax[i,2].get_ylim()[1])

    ax[i,0].set_ylim(ymin, ymax)
    ax[i,1].set_ylim(ymin, ymax)
    ax[i,2].set_ylim(ymin, ymax)

fig.subplots_adjust(hspace=0, wspace=0)
plt.yticks(fontsize = 8)
plt.savefig('../figures/sys1_chain_multiburn.pdf')



# Print autocorrelation length
print "Autocorrelation lengths:"
print "Burn-in 1:", sampler1.acor
print "Burn-in 2:", sampler2.acor
print "Production run 1:", sampler3.acor

# Save samples
pickle.dump( sampler, open( "../data/sys1_MCMC_multiburn_sampler.obj", "wb" ) )
