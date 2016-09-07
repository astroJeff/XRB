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
sys.path.append('../pop_synth')
import pop_synth
sys.path.append('../binary')
import binary_evolve
import load_sse
sys.path.append('../constants')
import constants as c


print "Loading data"

# Load sampler using pickle
sampler = pickle.load( open( "../data/SMC_MCMC_sampler.obj", "rb" ) )
HMXB = pickle.load( open("../data/SMC_MCMC_HMXB.obj", "rb" ) )


print "Finished loading data"

# # Chains plot
# fig, ax = plt.subplots(sampler.dim, 1, sharex=True, figsize=(7.0,20.0))
# for i in range(sampler.dim):
#     for chain in sampler.chain[...,i]:
#         ax[i].plot(chain, alpha=0.25, color='k', drawstyle='steps', rasterized=True)
#         yloc = plt.MaxNLocator(3)
#         ax[i].yaxis.set_major_locator(yloc)
# fig.subplots_adjust(hspace=0)
# plt.yticks(fontsize = 8)
#plt.savefig('../figures/smc_population_chains.pdf', rasterized=True)


# Corner plot
# labels = [r"$M_1\ (M_{\odot})$", r"$M_2\ (M_{\odot})$", r"$a\ (R_{\odot})$", r"$e$", r"$v_k\ ({\rm km}\ {\rm s}^{-1})$", r"$\theta\ ({\rm rad})$", r"$\phi\ ({\rm rad})$", r"$\alpha_{\rm b}\ ({\rm deg})$", r"$\delta_{\rm b}\ ({\rm deg}) $", r"$t_{\rm b}\ ({\rm Myr})$"]
# plt_range = ([7,24], [2.5,15], [0,1500], [0,1], [0,450], [0,np.pi], [0,2.0*np.pi], [6,21], [-76,-70], [0,70])
# hist2d_kwargs = {"plot_datapoints" : False}
# fig = corner.corner(sampler.flatchain, labels=labels, range=plt_range, bins=40, max_n_ticks=4, **hist2d_kwargs)
# plt.rc('font', size=18)
# plt.savefig('../figures/smc_population_corner.pdf')
# plt.rc('font', size=10)




# Now, we want to run all the sampler positions forward to
# get the distribution today of HMXBs
# l = len(sampler.flatchain.T[0])
# 
# print "length:", l
# 
# HMXB_ra = np.zeros(l)
# HMXB_dec = np.zeros(l)
# HMXB_Porb = np.zeros(l)
# HMXB_ecc = np.zeros(l)
# HMXB_M2 = np.zeros(l)
# HMXB_vsys = np.zeros(l)
# HMXB_Lx = np.zeros(l)
# HMXB_theta = np.zeros(l)
# 
# for i in np.arange(l):
# 
#     s = sampler.flatchain[i]
#
#     # Run forward model
#     data_out = pop_synth.full_forward(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[9])
#
#     # Get a random phi for position angle
#     ran_phi = pop_synth.get_phi(1)
#
#     # Get the new ra and dec
#    ra_out, dec_out = pop_synth.get_new_ra_dec(s[7], s[8], data_out[7], ran_phi)
# 
#     # Get the output orbital period
#     Porb = binary_evolve.A_to_P(data_out[0], data_out[1], data_out[5])
# 
#     # Save outputs
#     HMXB_ra[i] = ra_out
#     HMXB_dec[i] = dec_out
#     HMXB_Porb[i] = Porb
#     HMXB_ecc[i] = data_out[6]
#     HMXB_M2[i] = data_out[1]
#     HMXB_vsys[i] = data_out[3]
#     HMXB_Lx[i] = data_out[2]
#     HMXB_theta[i] = data_out[7] * 180.0*60.0/np.pi
# 
# # Save current HMXB parameters as an object
# HMXB = np.array([HMXB_ra, HMXB_dec, HMXB_Porb, HMXB_ecc, HMXB_M2, HMXB_vsys, HMXB_Lx, HMXB_theta])
# pickle.dump( HMXB, open( "../data/SMC_MCMC_HMXB.obj", "wb" ) )







# # HMXB Orbital period vs. eccentricity
# fig, ax = plt.subplots(3, 1, figsize=(6,8))
# plt.rc('font', size=14)
# plt_range = ([0, 4], [0,1])
# corner.hist2d(np.log10(HMXB[2]), HMXB[3], ax=ax[0], bins=40, range=plt_range, plot_datapoints=False)
# ax[0].set_xlabel(r"${\rm log}\ P_{\rm orb}\ {\rm (days)}$", size=16)
# ax[0].set_ylabel(r"$e$", size=16)
# ax[0].set_xticks([0,1,2,3,4])
# ax[0].set_yticks([0,0.25,0.5,0.75,1.0])
# 
# plt_range = ([8, 24], [0,80])
# corner.hist2d(HMXB[4], HMXB[5], ax=ax[1], bins=40, range=plt_range, plot_datapoints=False)
# ax[1].set_xlabel(r"$M_2\ ({\rm M}_{\odot})$", size=16)
# ax[1].set_ylabel(r"$v_{\rm sys}\ {\rm (km\ s}^{-1})$", size=16)
# ax[1].set_xticks([8,12,16,20,24])
# ax[1].set_yticks([0,20, 40, 60, 80])
# 
# 
# # Get flight time from birth time and M1 lifetime
# load_sse.load_sse()
# t_flight = sampler.flatchain.T[9] - load_sse.func_sse_tmax(sampler.flatchain.T[0])
# plt_range = ([0,55], [0,25])
# contour_kwargs = {'colors':'r', 'linestyles':'dashed'}
# corner.hist2d(t_flight, HMXB[7], ax=ax[2], bins=40, range=plt_range, plot_density=False, 
# 		plot_datapoints=False, contour_kwargs=contour_kwargs)
# corner.hist2d(sampler.flatchain.T[9], HMXB[7], ax=ax[2], bins=40, range=plt_range, 
# 		plot_density=False, plot_datapoints=False)
# ax[2].set_xlabel(r"$t\ {\rm (Myr)}$", size=16)
# ax[2].set_ylabel(r"$\theta\ {\rm (amin)}$", size=16)
# 
# plt.tight_layout()
# 
# plt.savefig('../figures/smc_population_HMXB.pdf')



plt.rc('font', size=10)

# Birth location
# fig, host = plt.subplots(figsize=(5,5))
# sf_history.get_SMC_plot(30.0)
# plt_kwargs = {'colors':'k'}
# density_contour.density_contour(sampler.flatchain.T[7], sampler.flatchain.T[8], nbins_x=40, nbins_y=40, **plt_kwargs)
# plt.tight_layout()
plt.figure(figsize=(4,4))
sf_history.get_SMC_plot_polar(40.0, ra_dist=sampler.flatchain.T[7], dec_dist=sampler.flatchain.T[8], xwidth=3.0, ywidth=3.0)
plt.tight_layout()
plt.savefig('../figures/smc_population_ra_dec.pdf')




# Current location
# fig, host = plt.subplots(figsize=(5,5))
# sf_history.get_SMC_plot(30.0)
# plt_kwargs = {'colors':'k'}
# density_contour.density_contour(HMXB[0], HMXB[1], nbins_x=40, nbins_y=40, **plt_kwargs)
# plt.tight_layout()
fig, ax = plt.subplots(2, 1, figsize=(4.5,7))
#plt.figure(figsize=(5,5))
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[1].set_xticks([])
ax[1].set_yticks([])
sf_history.get_SMC_plot_polar(12.0, ra_dist=HMXB[0], dec_dist=HMXB[1], fig_in=fig, rect=211, xwidth=3.0, ywidth=3.0)
sf_history.get_SMC_plot_polar(40.0, ra_dist=HMXB[0], dec_dist=HMXB[1], fig_in=fig, rect=212, xwidth=3.0, ywidth=3.0)
plt.tight_layout()
plt.savefig('../figures/smc_population_HMXB_ra_dec.pdf')


