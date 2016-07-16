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
sampler = pickle.load( open( "../data/sys2_MCMC_multiburn_sampler.obj", "rb" ) )
#init_params = pickle.load( open( "../data/sys2_pop_synth_init_conds.obj", "rb" ) )
#HMXB = pickle.load( open( "../data/sys2_pop_synth_HMXB.obj", "rb" ) )




# Chains plot
# fig, ax = plt.subplots(sampler.dim, 1, sharex=True, figsize=(7.0,20.0))
# for i in range(sampler.dim):
#     for chain in sampler.chain[...,i]:
#         ax[i].plot(chain, alpha=0.25, color='k', drawstyle='steps')
#         yloc = plt.MaxNLocator(3)
#         ax[i].yaxis.set_major_locator(yloc)
#         # ax[i].set_yticks(fontsize=8)
# fig.subplots_adjust(hspace=0)
# plt.yticks(fontsize = 8)
# plt.savefig('../figures/sys2_chains.pdf', rasterized=True)



# Likelihood as a function of each parametersplt.rc('font', size=8)
# fig, ax = plt.subplots(2,5, figsize=(14,6))
# labels = [r"$M_1$", r"$M_2$", r"$A$", r"$e$", r"$v_k$", r"$\theta$", r"$\phi$", r"$\alpha_{\rm b}$", r"$\delta_{\rm b}$", r"$t_{\rm b}$"]
# for i in np.arange(10):
#     a = np.int(i/5)
#     b = i%5
#     corner.hist2d(sampler.chain[:,:,i], sampler.lnprobability, ax=ax[a,b], bins=30)
#     ax[a,b].set_xlabel(labels[i])
# plt.tight_layout()
# plt.savefig('../figures/sys2_likelihoods.pdf')



# Corner plot
plt.rc('font', size=18)

labels = [r"$M_1$", r"$M_2$", r"$A$", r"$e$", r"$v_k$", r"$\theta$", r"$\phi$", r"$\alpha_{\rm b}$", r"$\delta_{\rm b}$", r"$t_{\rm b}$"]
truths = [M1_true, M2_true, A_true, ecc_true, v_k_true, theta_true, phi_true, ra_true, dec_true, t_b_true]
hist2d_kwargs = {"plot_datapoints" : False}
fig = corner.corner(sampler.flatchain, labels=labels, truths=truths, **hist2d_kwargs)

#ax2 = plt.subplot2grid((5,5), (0,3), colspan=2, rowspan=2)
ra_out = sampler.flatchain.T[7]
dec_out = sampler.flatchain.T[8]
ra_obs = 15.9
dec_obs = -72.25
sf_history.get_SMC_plot_polar(50, fig_in=fig, rect=333, ra_dist=ra_out, dec_dist=dec_out, ra=ra_obs, dec=dec_obs, xwidth=0.5, ywidth=0.5, xgrid_density=6)

plt.tight_layout()
plt.savefig('../figures/sys2_corner_multiburn.pdf')


# Birth location distribution
# plt.figure(figsize=(8,8))
# ra_out = sampler.flatchain.T[7]
# dec_out = sampler.flatchain.T[8]
# sf_history.get_SMC_plot_polar(t_b_true, ra_dist=ra_out, dec_dist=dec_out, ra=ra_true, dec=dec_true)
#
# plt.savefig('../figures/sys2_dist_birth_location.pdf')
