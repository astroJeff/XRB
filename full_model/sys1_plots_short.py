# Create plots from saved pickles of J0045-7319 simulations
from src.core import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib.gridspec as gridspec
import corner
import pickle
from astropy.coordinates import SkyCoord
from astropy import units as u

from xrb.SF_history import sf_history
from xrb.src import stats


# System 1 test parameters
M1_true = 13.0
M2_true = 10.0
A_true = 150.0
ecc_true = 0.7
v_k_true = 250.0
theta_true = 2.9
phi_true = 0.9
ra_true = 13.51
dec_true = -72.7
t_b_true = 22.0



# Load pickled data
sampler = pickle.load( open( "../data/sys1_MCMC_multiburn_sampler.obj", "rb" ) )
# init_params = pickle.load( open( "../data/sys1_pop_synth_init_conds.obj", "rb" ) )
# HMXB = pickle.load( open( "../data/sys1_pop_synth_HMXB.obj", "rb" ) )




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
# plt.savefig('../figures/sys1_chains.pdf', rasterized=True)



# Likelihood as a function of each parametersplt.rc('font', size=8)
# fig, ax = plt.subplots(2,5, figsize=(14,6))
# labels = [r"$M_1$", r"$M_2$", r"$A$", r"$e$", r"$v_k$", r"$\theta$", r"$\phi$", r"$\alpha_{\rm b}$", r"$\delta_{\rm b}$", r"$t_{\rm b}$"]
# for i in np.arange(10):
#     a = np.int(i/5)
#     b = i%5
#     corner.hist2d(sampler.chain[:,:,i], sampler.lnprobability, ax=ax[a,b], bins=30)
#     ax[a,b].set_xlabel(labels[i])
# plt.tight_layout()
# plt.savefig('../figures/sys1_likelihoods.pdf')



############## Corner pyplot ###################
fontProperties = {'family':'serif', 'serif':['Times New Roman'], 'weight':'normal', 'size':12}
ticks_font = font_manager.FontProperties(family='Times New Roman', style='normal', \
                                         weight='normal', stretch='normal', size=12)
plt.rc('font', **fontProperties)

fig, ax = plt.subplots(10,10, figsize=(10,10))


labels = [r"$M_{\rm 1, i}\ (M_{\odot})$", r"$M_{\rm 2, i}\ (M_{\odot})$", r"$a_{\rm i}\ (R_{\odot})$", \
          r"$e_{\rm i}$", r"$v_{\rm k, i}\ ({\rm km}\ {\rm s}^{-1})$", r"$\theta_{\rm k}\ ({\rm rad.})$", \
          r"$\phi_{\rm k}\ ({\rm rad.})$", r"$\alpha_{\rm i}\ ({\rm deg.})$", \
          r"$\delta_{\rm i}\ ({\rm deg.}) $", r"$t_{\rm i}\ ({\rm Myr})$"]
truths = [M1_true, M2_true, A_true, ecc_true, v_k_true, theta_true, phi_true, ra_true, dec_true, t_b_true]
hist2d_kwargs = {"plot_datapoints" : False}
fig = corner.corner(sampler.flatchain, fig=fig, labels=labels, truths=truths, max_n_ticks=4, **hist2d_kwargs)



#ax2 = plt.subplot2grid((5,5), (0,3), colspan=2, rowspan=2)
ra_out = sampler.flatchain.T[7]
dec_out = sampler.flatchain.T[8]
ra_obs = 13.5
dec_obs = -72.63
gs = gridspec.GridSpec(2, 2,
                       width_ratios=[3,2],
                       height_ratios=[2,3]
                       )
sf_history.get_SMC_plot_polar(22, fig_in=fig, gs=gs[1], ra_dist=ra_out, dec_dist=dec_out, ra=ra_obs, dec=dec_obs, xwidth=0.5, ywidth=0.5, xgrid_density=6)



# Shift axis labels
for i in np.arange(10):
    ax[i,0].yaxis.set_label_coords(-0.5, 0.5)
    ax[9,i].xaxis.set_label_coords(0.5, -0.5)

# Set declination ticks
ax[9,8].set_xticks([-72.9, -72.7, -72.5])
ax[9,8].set_xticklabels(["-72.9", "-72.7", "-72.5"])
for i in np.arange(8):
    ax[8,i].set_yticks([-72.9, -72.7, -72.5])
ax[8,0].set_yticklabels(["-72.9", "-72.7", "-72.5"])

# Set theta ticks
for i in np.arange(5)+5:
    ax[i,5].set_xticks([np.pi/2., 3.*np.pi/4., np.pi])
ax[9,5].set_xticklabels([r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$'])
for i in np.arange(4):
    ax[5,i].set_yticks([np.pi/2., 3.*np.pi/4., np.pi])
ax[5,0].set_yticklabels([r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$'])


# Set phi ticks
for i in np.arange(4)+6:
    ax[i,6].set_xticks([np.pi/2., np.pi])
ax[9,6].set_xticklabels([r'$\frac{\pi}{2}$', r'$\pi$'])
for i in np.arange(6):
    ax[6,i].set_yticks([np.pi/2., np.pi])
ax[6,0].set_yticklabels([r'$\frac{\pi}{2}$', r'$\pi$'])




plt.subplots_adjust(bottom=0.07, left=0.07, top=0.97)

#plt.tight_layout()

plt.savefig('../figures/sys1_corner_multiburn.pdf')
plt.rc('font', size=10)


# Corner plot
# plt.rc('font', size=18)
# labels = [r"$M_1$", r"$M_2$", r"$A$", r"$e$", r"$v_k$", r"$\theta$", r"$\phi$", r"$\alpha_{\rm b}$", r"$\delta_{\rm b}$", r"$t_{\rm b}$"]
# truths = [M1_true, M2_true, A_true, ecc_true, v_k_true, theta_true, phi_true, ra_true, dec_true, t_b_true]
# fig = corner.corner(sampler.flatchain, labels=labels, truths=truths, bins=30)
#
# ax2 = plt.subplot2grid((5,5), (0,3), colspan=2, rowspan=2)
# ra_out = sampler.flatchain.T[7]
# dec_out = sampler.flatchain.T[8]
# ra_obs = 13.5
# dec_obs = -72.63
# sf_history.get_SMC_plot_polar(24, ra_dist=ra_out, dec_dist=dec_out, ra=ra_obs, dec=dec_obs)
# plt.savefig('../figures/sys1_corner.pdf')
# plt.rc('font', size=10)


# Birth location distribution
# plt.figure(figsize=(8,8))
# ra_out = sampler.flatchain.T[7]
# dec_out = sampler.flatchain.T[8]
# sf_history.get_SMC_plot_polar(t_b_true, ra_dist=ra_out, dec_dist=dec_out, ra=ra_true, dec=dec_true)
#
# plt.savefig('../figures/sys1_dist_birth_location.pdf')




#
# # Orbital period
# plt.subplot(4,1,1)
# corner.hist2d(sampler.flatchain.T[0], sampler.flatchain.T[1])
# plt.scatter(init_params["M1"], init_params["M2"], color='r')
# plt.xlabel(r"$M_1$", size=16)
# plt.ylabel(r"$M_2$", size=16)
# plt.xlim(8.5, 12.0)
# plt.ylim(3.0, 4.5)
#
# # Orbital eccentricity
# plt.subplot(4,1,2)
# corner.hist2d(sampler.flatchain.T[2], sampler.flatchain.T[3])
# plt.scatter(init_params["A"], init_params["ecc"], color='r')
# plt.xlabel(r"$a$", size=16)
# plt.ylabel(r"$e$", size=16)
# plt.xlim(10.0, 1500.0)
# plt.ylim(0.0, 1.0)
#
# # Companion mass
# plt.subplot(4,1,3)
# corner.hist2d(sampler.flatchain.T[4], sampler.flatchain.T[5])
# plt.scatter(init_params["v_k"], init_params["theta"], color='r')
# plt.xlabel(r"$v_k$", size=16)
# plt.ylabel(r"$\theta$", size=16)
# plt.xlim(0.0, 250.0)
# plt.ylim(1.9, np.pi)
#
# # Birth position
# plt.subplot(4,1,4)
# plt.hist(sampler.flatchain.T[9], histtype='step', color='k', bins=30)
# for i in np.arange(len(init_params)):
#     plt.axvline(init_params["t_b"][i])
# plt.xlabel(r"$t_b$", size=16)
# plt.tight_layout()
# plt.savefig('../figures/sys1_MCMC_pop_synth_compare.pdf')
