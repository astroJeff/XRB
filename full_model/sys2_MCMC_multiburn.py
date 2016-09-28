# Run test system 1
import sys
sys.path.append("../")
from src.core import *

from astropy.coordinates import SkyCoord
from astropy import units as u
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner

from src import stats
from pop_synth import pop_synth
from binary import binary_evolve



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

# Birth position
ra_b = 15.8
dec_b = -72.1

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
    nwalkers=640, nburn=100, nsteps=500)

print "Simulation took", time.time()-start_time, "seconds"



# Print autocorrelation length
print "Burn-in 1:"
print "Autocorrelation lengths", sampler1.acor
print "Acceptance fraction", sampler1.acceptance_fraction

print "Burn-in 2:"
print "Autocorrelation lengths", sampler2.acor
print "Acceptance fraction", sampler2.acceptance_fraction

print "Burn-in 3:"
print "Autocorrelation lengths", sampler3.acor
print "Acceptance fraction", sampler3.acceptance_fraction

print "Burn-in 4:"
print "Autocorrelation lengths", sampler4.acor
print "Acceptance fraction", sampler4.acceptance_fraction

print "Production run:"
print "Autocorrelation lengths", sampler.acor
print "Acceptance fraction", sampler.acceptance_fraction


# Save samples
pickle.dump( sampler1, open( "../data/sys2_MCMC_multiburn_burn1.obj", "wb" ) )
pickle.dump( sampler2, open( "../data/sys2_MCMC_multiburn_burn2.obj", "wb" ) )
pickle.dump( sampler3, open( "../data/sys2_MCMC_multiburn_burn3.obj", "wb" ) )
pickle.dump( sampler4, open( "../data/sys2_MCMC_multiburn_burn4.obj", "wb" ) )
pickle.dump( sampler, open( "../data/sys2_MCMC_multiburn_sampler.obj", "wb" ) )




############## Corner pyplot ###################
labels = [r"$M_1$", r"$M_2$", r"$A$", r"$e$", r"$v_k$", r"$\theta$", r"$\phi$", r"$\alpha_{\rm b}$", r"$\delta_{\rm b}$", r"$t_{\rm b}$"]
hist2d_kwargs = {"plot_datapoints" : False}
fig = corner.corner(sampler.flatchain, labels=labels, truths=truths, **hist2d_kwargs)
plt.rc('font', size=18)
plt.savefig('../figures/sys2_corner_multiburn.pdf')
plt.rc('font', size=10)




################# Posterior plots ##################
plt.rc('font', size=8)
fig, ax = plt.subplots(2,5, figsize=(14,6))

for i in np.arange(10):
    a = np.int(i/5)
    b = i%5

    xmin = np.min(sampler.chain[:,:,i])
    xmax = np.max(sampler.chain[:,:,i])
    corner.hist2d(sampler.chain[:,:,i], sampler.lnprobability, ax=ax[a,b], bins=30, range=((xmin,xmax),(-50,0)), plot_datapoints=False)
    ax[a,b].set_xlabel(labels[i])

    ax[a,b].set_ylim(-30,0)
    ax[a,b].axhline(posterior_truths)
    ax[a,b].axvline(truths[i], color='r')

plt.tight_layout()
plt.savefig('../figures/sys2_posterior_params_multiburn.pdf')


################# Chains plot #####################
fig, ax = plt.subplots(sampler1.dim, 5, sharex=False, figsize=(18.0,20.0))
for i in range(sampler1.dim):
    for j in np.arange(len(sampler1.chain[...])):

        chain1 = sampler1.chain[...,i][j]
        ax[i,0].plot(chain1, alpha=0.25, color='k', drawstyle='steps')

        chain2 = sampler2.chain[...,i][j]
        ax[i,1].plot(chain2, alpha=0.25, color='k', drawstyle='steps')

        chain3 = sampler3.chain[...,i][j]
        ax[i,2].plot(chain3, alpha=0.25, color='k', drawstyle='steps')

        chain4 = sampler4.chain[...,i][j]
        ax[i,3].plot(chain4, alpha=0.25, color='k', drawstyle='steps')

        chain5 = sampler.chain[...,i][j]
        ax[i,4].plot(chain5, alpha=0.25, color='k', drawstyle='steps')

    # Remove tick labels from y-axis
    ax[i,1].set_yticklabels([])
    ax[i,2].set_yticklabels([])
    ax[i,3].set_yticklabels([])
    ax[i,4].yaxis.tick_right()

    # Remove all but bottom x-axis ticks
    if i != sampler1.dim-1:
        ax[i,0].set_xticklabels([])
        ax[i,1].set_xticklabels([])
        ax[i,2].set_xticklabels([])
        ax[i,3].set_xticklabels([])
        ax[i,4].set_xticklabels([])

    # Add truths as a red lines across entire plot
    ax[i,0].axhline(truths[i], color='r')
    ax[i,1].axhline(truths[i], color='r')
    ax[i,2].axhline(truths[i], color='r')
    ax[i,3].axhline(truths[i], color='r')
    ax[i,4].axhline(truths[i], color='r')

# Make all plots have the same y-range - plots must already be created
for i in range(sampler1.dim):
    ymin = min(ax[i,0].get_ylim()[0],ax[i,1].get_ylim()[0],ax[i,2].get_ylim()[0])
    ymax = max(ax[i,0].get_ylim()[1],ax[i,1].get_ylim()[1],ax[i,2].get_ylim()[1])

    ax[i,0].set_ylim(ymin, ymax)
    ax[i,1].set_ylim(ymin, ymax)
    ax[i,2].set_ylim(ymin, ymax)
    ax[i,3].set_ylim(ymin, ymax)
    ax[i,4].set_ylim(ymin, ymax)

fig.subplots_adjust(hspace=0, wspace=0)
plt.yticks(fontsize = 8)
plt.savefig('../figures/sys2_chain_multiburn.pdf')



