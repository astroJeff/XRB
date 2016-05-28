# Create plots from saved pickles of J0045-7319 simulations

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


# System 1 test parameters
M1_true = 9.0
M2_true = 4.0
A_true = 500.0
ecc_true = 0.5
v_k_true = 100.0
theta_true = 2.7
phi_true = 1.2
ra_true = 15.8
dec_true = -72.1
t_b_true = 50.0



# Load pickled data
sampler = pickle.load( open( "../data/sys2_MCMC_sampler.obj", "rb" ) )
#init_params = pickle.load( open( "../data/sys2_pop_synth_init_conds.obj", "rb" ) )
#HMXB = pickle.load( open( "../data/sys2_pop_synth_HMXB.obj", "rb" ) )




# Chains plot
#fig, ax = plt.subplots(sampler.dim, 1, sharex=True, figsize=(7.0,20.0))
#for i in range(sampler.dim):
#    for chain in sampler.chain[...,i]:
#        ax[i].plot(chain, alpha=0.25, color='k', drawstyle='steps')
#        yloc = plt.MaxNLocator(3)
#        ax[i].yaxis.set_major_locator(yloc)
#        ax[i].set_yticks(fontsize=8)
#fig.subplots_adjust(hspace=0)
#plt.yticks(fontsize = 8)
#plt.savefig('../figures/sys2_chains.pdf')


# Corner plot
labels = [r"$M_1$", r"$M_2$", r"$A$", r"$e$", r"$v_k$", r"$\theta$", r"$\phi$", r"$\alpha_{\rm b}$", r"$\delta_{\rm b}$", r"$t_{\rm b}$"]
truths = [M1_true, M2_true, A_true, ecc_true, v_k_true, theta_true, phi_true, ra_true, dec_true, t_b_true]
fig = corner.corner(sampler.flatchain, labels=labels, truths=truths)
plt.rc('font', size=18)
plt.savefig('../figures/sys2_corner.pdf')
plt.rc('font', size=10)





# Birth position plot
#plt.figure(figsize=(10.0, 6.0))
ra_out = sampler.flatchain.T[7]
dec_out = sampler.flatchain.T[8]
#plt.subplot(1,2,2)
#sf_history.get_SMC_plot(42.0)
#plt.scatter(ra_true, dec_true, marker="*", s=20, color='r')
#plt_kwargs = {'colors':'k'}
#density_contour.density_contour(ra_out, dec_out, nbins_x=25, nbins_y=25, **plt_kwargs)
#plt.xlim(10.0, 13.0)
#plt.ylim(-73.7, -72.5)

#plt.subplot(1,2,1)
#sf_history.get_SMC_plot(42.0)
#plt.scatter(ra_true, dec_true, marker="*", s=20, color='r')
#plt_kwargs = {'colors':'k'}
#density_contour.density_contour(ra_out, dec_out, nbins_x=25, nbins_y=25, **plt_kwargs)
#plt.xlim(9.0, 18.0)
#plt.ylim(-74.0, -71.5)
#plt.tight_layout()
#plt.savefig('../figures/sys2_dist_birth_location.pdf')



# Better birth distribution plot
plt.figure(figsize=(8,8))
ra_obs = 13.5
dec_obs = -72.63
sf_history.get_SMC_plot_polar(24, ra_dist=ra_out, dec_dist=dec_out, ra=ra_obs, dec=dec_obs)
plt.savefig('../figures/sys2_dist_birth_location.pdf')



# MCMC vs. population synthesis plot
plt.figure(figsize=(6,15))

# Orbital period
plt.subplot(4,1,1)
corner.hist2d(sampler.flatchain.T[0], sampler.flatchain.T[1])
plt.scatter(init_params["M1"], init_params["M2"], color='r')
plt.xlabel(r"$M_1$", size=16)
plt.ylabel(r"$M_2$", size=16)
plt.xlim(8.5, 12.0)
plt.ylim(3.0, 4.5)

# Orbital eccentricity
plt.subplot(4,1,2)
corner.hist2d(sampler.flatchain.T[2], sampler.flatchain.T[3])
plt.scatter(init_params["A"], init_params["ecc"], color='r')
plt.xlabel(r"$a$", size=16)
plt.ylabel(r"$e$", size=16)
plt.xlim(10.0, 1500.0)
plt.ylim(0.0, 1.0)

# Companion mass
plt.subplot(4,1,3)
corner.hist2d(sampler.flatchain.T[4], sampler.flatchain.T[5])
plt.scatter(init_params["v_k"], init_params["theta"], color='r')
plt.xlabel(r"$v_k$", size=16)
plt.ylabel(r"$\theta$", size=16)
plt.xlim(0.0, 250.0)
plt.ylim(1.9, np.pi)

# Birth position
plt.subplot(4,1,4)
plt.hist(sampler.flatchain.T[9], histtype='step', color='k', bins=30)
for i in np.arange(len(init_params)):
    plt.axvline(init_params["t_b"][i])
plt.xlabel(r"$t_b$", size=16)
plt.tight_layout()
plt.savefig('../figures/sys2_MCMC_pop_synth_compare.pdf')
