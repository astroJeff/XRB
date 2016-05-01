# Create plots from saved pickles of SMC HMXB simulations

import sys
import numpy as np
import matplotlib.pyplot as plt
import corner
import pickle
from astropy.coordinates import SkyCoord
from astropy import units as u

sys.path.append('../SF_history')
import sf_history
sys.path.append('../stats')
import stats
sys.path.append('../notebooks')
import density_contour


# Load sampler using pickle
sampler = pickle.load( open( "../data/smc_MCMC_sampler.obj", "rb" ) )


# Chains plot
fig, ax = plt.subplots(sampler.dim, 1, sharex=True, figsize=(7.0,20.0))
for i in range(sampler.dim):
    for chain in sampler.chain[...,i]:
        ax[i].plot(chain, alpha=0.25, color='k', drawstyle='steps')
        yloc = plt.MaxNLocator(3)
        ax[i].yaxis.set_major_locator(yloc)
#        ax[i].set_yticks(fontsize=8)
fig.subplots_adjust(hspace=0)
#plt.yticks(fontsize = 8)
plt.savefig('../figures/smc_population_chains.pdf')


# Corner plot
labels = [r"$M_1$", r"$M_2$", r"$A$", r"$e$", r"$v_k$", r"$\theta$", r"$\phi$", r"$\alpha_{\rm b}$", r"$\delta_{\rm b}$", r"$t_{\rm b}$"]
truths = [M1_true, M2_true, A_true, ecc_true, v_k_true, theta_true, phi_true, ra_true, dec_true, t_b_true]
fig = corner.corner(sampler.flatchain, labels=labels, truths=truths)
plt.rc('font', size=18)
plt.savefig('../figures/smc_population_corner.pdf')
plt.rc('font', size=10)



# M1 vs. M2
plt.subplot(4,1,1)
corner.hist2d(sampler.flatchain.T[0], sampler.flatchain.T[1])
plt.scatter(init_params["M1"], init_params["M2"], color='r')
plt.xlabel(r"$M_1$", size=16)
plt.ylabel(r"$M_2$", size=16)
#plt.xlim(8.5, 12.0)
#plt.ylim(3.0, 4.5)
plt.savefig('../figures/smc_population_M1_M2.pdf')

# Orbital separation vs. eccentricity
plt.subplot(4,1,2)
corner.hist2d(sampler.flatchain.T[2], sampler.flatchain.T[3])
plt.scatter(init_params["A"], init_params["ecc"], color='r')
plt.xlabel(r"$a$", size=16)
plt.ylabel(r"$e$", size=16)
#plt.xlim(10.0, 1500.0)
#plt.ylim(0.0, 1.0)
plt.savefig('../figures/smc_population_A_ecc.pdf')
