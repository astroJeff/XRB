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



############## Run sampler ###################
start_time = time.time()


sampler1, sampler2, sampler3, sampler4, sampler = stats.run_emcee_2(M2_d_J0045, P_orb_J0045, ecc_J0045, ra_J0045, dec_J0045, \
    M2_d_err=M2_d_J0045_err, P_orb_obs_err=P_orb_J0045_err, ecc_obs_err=ecc_J0045_err, \
    nburn=5000, nsteps=10000)

print "Simulation took", time.time()-start_time, "seconds"



############## Corner pyplot ###################
labels = [r"$M_1$", r"$M_2$", r"$A$", r"$e$", r"$v_k$", r"$\theta$", r"$\phi$", r"$\alpha_{\rm b}$", r"$\delta_{\rm b}$", r"$t_{\rm b}$"]
hist2d_kwargs = {"plot_datapoints" : False}
fig = corner.corner(sampler.flatchain, labels=labels, **hist2d_kwargs)
plt.rc('font', size=18)
plt.savefig('../figures/J0045_corner_multiburn.pdf')
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

plt.tight_layout()
plt.savefig('../figures/J0045_posterior_params_multiburn.pdf')


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
plt.savefig('../figures/J0045_chain_multiburn.pdf')



# Print autocorrelation length
print "Autocorrelation lengths:"
print "Burn-in 1:", sampler1.acor
print "Burn-in 2:", sampler2.acor
print "Burn-in 3:", sampler3.acor
print "Burn-in 4:", sampler4.acor
print "Production run:", sampler.acor

# Save samples
pickle.dump( sampler, open( "../data/J0045_MCMC_multiburn_sampler.obj", "wb" ) )
