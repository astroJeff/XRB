# Create plots from saved pickles of J0045-7319 simulations
from src.core import *
set_data_path("../data")

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
from xrb.source import stats


# J0045-7319 coordinates
coor_J0045 = SkyCoord('00h45m35.26s', '-73d19m03.32s')

ra_J0045 = coor_J0045.ra.degree
dec_J0045 = coor_J0045.dec.degree


# Load pickled data
sampler = pickle.load( open( INDATA("J0045_MCMC_multiburn_sampler.obj"), "rb" ) )
#init_params_J0045 = pickle.load( open( INDATA("J0045_pop_synth_init_conds.obj"), "rb" ) )
#HMXB_J0045 = pickle.load( open( INDATA("J0045_pop_synth_HMXB.obj"), "rb" ) )


# Specific distribution plots
# plt.rc('font', size=18)
# # M1 vs M2
# M1 = sampler.flatchain.T[0]
# M2 = sampler.flatchain.T[1]
# corner.hist2d(M1, M2)
# plt.xlabel(r'$M_1\ ({\rm M_{\odot}})$', fontsize=22)
# plt.ylabel(r'$M_2\ ({\rm M_{\odot}})$', fontsize=22)
# plt.savefig('../figures/J0045_M1_M2.pdf')
# #plt.show()
# # V_k vs theta
# v_k = sampler.flatchain.T[4]
# theta = sampler.flatchain.T[5]
# corner.hist2d(v_k, theta)
# plt.xlabel(r'$v_k\ ({\rm km/s})$', fontsize=22)
# plt.ylabel(r'$\theta$', fontsize=22)
# plt.savefig('../figures/J0045_vk_theta.pdf')
# #plt.show()
# # t_b histogram
# t_b = sampler.flatchain.T[9]
# plt.hist(t_b, histtype='step', color='k', bins=50)
# plt.xlabel(r'$t_b\ ({\rm Myr})$')
# plt.savefig('../figures/J0045_tb.pdf')
#plt.show()


# Corner plot
fontProperties = {'family':'serif', 'serif':['Times New Roman'], 'weight':'normal', 'size':12}
ticks_font = font_manager.FontProperties(family='Times New Roman', style='normal', \
                                         weight='normal', stretch='normal', size=12)
plt.rc('font', **fontProperties)

fig, ax = plt.subplots(10,10, figsize=(10,10))


labels = [r"$M_{\rm 1, i}\ (M_{\odot})$", r"$M_{\rm 2, i}\ (M_{\odot})$", r"$a_{\rm i}\ (R_{\odot})$", \
          r"$e_{\rm i}$", r"$v_{\rm k, i}\ ({\rm km}\ {\rm s}^{-1})$", r"$\theta_{\rm k}\ ({\rm rad.})$", \
          r"$\phi_{\rm k}\ ({\rm rad.})$", r"$\alpha_{\rm i}\ ({\rm deg.})$", \
          r"$\delta_{\rm i}\ ({\rm deg.}) $", r"$t_{\rm i}\ ({\rm Myr})$"]
hist2d_kwargs = {"plot_datapoints" : False}
fig = corner.corner(sampler.flatchain, fig=fig, labels=labels, max_n_ticks=4, **hist2d_kwargs)

ra_out = sampler.flatchain.T[7]
dec_out = sampler.flatchain.T[8]
gs = gridspec.GridSpec(2, 2,
                       width_ratios=[3,2],
                       height_ratios=[2,3]
                       )
smc_plot, ax1 = sf_history.get_SMC_plot_polar(50, fig_in=fig, gs=gs[1], ra_dist=ra_out, dec_dist=dec_out, ra=ra_J0045, dec=dec_J0045, xgrid_density=6)


ax1.set_position([0.55, 0.55, 0.3, 0.3])




# gs = gridspec.GridSpec(2, 2,
#                        width_ratios=[3,2],
#                        height_ratios=[2,3]
#                        )
#
# ax1 = plt.subplot(gs[1])
# ax1.scatter(np.random.randint(5, size=20), np.random.randint(2, size=20))




# Shift axis labels
for i in np.arange(10):
    ax[i,0].yaxis.set_label_coords(-0.5, 0.5)
    ax[9,i].xaxis.set_label_coords(0.5, -0.5)

# Set declination ticks
ax[9,8].set_xticks([-73.5, -73.0])
ax[9,8].set_xticklabels(["-73.5", "-73.0"])
for i in np.arange(8):
    ax[8,i].set_yticks([-73.5, -73.0])
ax[8,0].set_yticklabels(["-73.5", "-73.0"])

# Set theta ticks
for i in np.arange(5)+5:
    ax[i,5].set_xticks([np.pi/2., 3.*np.pi/4., np.pi])
ax[9,5].set_xticklabels([r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$'])
for i in np.arange(4):
    ax[5,i].set_yticks([np.pi/2., 3.*np.pi/4., np.pi])
ax[5,0].set_yticklabels([r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$'])


# Set phi ticks
for i in np.arange(4)+6:
    ax[i,6].set_xticks([np.pi/2., np.pi, 3.*np.pi/2.])
ax[9,6].set_xticklabels([r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$'])
for i in np.arange(6):
    ax[6,i].set_yticks([np.pi/2., np.pi, 3.*np.pi/2.])
ax[6,0].set_yticklabels([r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$'])




plt.subplots_adjust(bottom=0.07, left=0.07, top=0.97)

#plt.tight_layout()

plt.savefig('../figures/J0045_corner_multiburn.pdf')
plt.rc('font', size=10)


# # Chains plot
# for i in range(sampler.dim):
#     plt.figure()
#     for chain in sampler.chain[...,i]:
#         plt.plot(chain, alpha=0.25, color='k', drawstyle='steps')
# plt.savefig('../figures/J0045_chains.pdf')



# Birth position plot
#plt.figure(figsize=(10.0, 6.0))
#plt.subplot(1,2,2)
#sf_history.get_SMC_plot(42.0)
#plt.scatter(ra_J0045, dec_J0045, marker="*", s=20, color='r')
#plt_kwargs = {'colors':'k'}
#density_contour.density_contour(ra_out, dec_out, nbins_x=25, nbins_y=25, **plt_kwargs)
#plt.xlim(13.0, 10.0)
#plt.ylim(-73.7, -72.5)

#plt.subplot(1,2,1)
#sf_history.get_SMC_plot(42.0)
#plt.scatter(ra_J0045, dec_J0045, marker="*", s=20, color='r')
#plt_kwargs = {'colors':'k'}
#density_contour.density_contour(ra_out, dec_out, nbins_x=25, nbins_y=25, **plt_kwargs)
#plt.xlim(18.0, 9.0)
#plt.ylim(-74.0, -71.5)
#plt.tight_layout()
#plt.savefig('../figures/J0045_dist_birth_location.pdf')

# Better birth distribution plot
# plt.figure(figsize=(8,8))
# ra_out = sampler.flatchain.T[7]
# dec_out = sampler.flatchain.T[8]
# sf_history.get_SMC_plot_polar(50, ra_dist=ra_out, dec_dist=dec_out, ra=ra_J0045, dec=dec_J0045)
# plt.savefig('../figures/J0045_dist_birth_location.pdf')



# # MCMC vs. population synthesis plot
# plt.figure(figsize=(6,15))
#
# # Orbital period
# plt.subplot(4,1,1)
# corner.hist2d(sampler.flatchain.T[0], sampler.flatchain.T[1])
# plt.scatter(init_params_J0045["M1"], init_params_J0045["M2"], color='r')
# plt.xlabel(r"$M_1$", size=16)
# plt.ylabel(r"$M_2$", size=16)
# plt.xlim(8.5, 12.0)
# plt.ylim(3.0, 4.5)
#
# # Orbital eccentricity
# plt.subplot(4,1,2)
# corner.hist2d(sampler.flatchain.T[2], sampler.flatchain.T[3])
# plt.scatter(init_params_J0045["A"], init_params_J0045["ecc"], color='r')
# plt.xlabel(r"$a$", size=16)
# plt.ylabel(r"$e$", size=16)
# plt.xlim(10.0, 1500.0)
# plt.ylim(0.0, 1.0)
#
# # Companion mass
# plt.subplot(4,1,3)
# corner.hist2d(sampler.flatchain.T[4], sampler.flatchain.T[5])
# plt.scatter(init_params_J0045["v_k"], init_params_J0045["theta"], color='r')
# plt.xlabel(r"$v_k$", size=16)
# plt.ylabel(r"$\theta$", size=16)
# plt.xlim(0.0, 250.0)
# plt.ylim(1.9, np.pi)
#
# # Birth position
# plt.subplot(4,1,4)
# plt.hist(sampler.flatchain.T[9], histtype='step', color='k', bins=30)
# for i in np.arange(len(init_params_J0045)):
#     plt.axvline(init_params_J0045["t_b"][i])
# plt.xlabel(r"$t_b$", size=16)
# plt.tight_layout()
# plt.savefig('../figures/J0045_MCMC_pop_synth_compare.pdf')
