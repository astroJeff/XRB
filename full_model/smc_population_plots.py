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
sys.path.append('../constants')
import constants as c


# Load sampler using pickle
sampler = pickle.load( open( "../data/SMC_MCMC_sampler.obj", "rb" ) )


# Chains plot
fig, ax = plt.subplots(sampler.dim, 1, sharex=True, figsize=(7.0,20.0))
for i in range(sampler.dim):
    for chain in sampler.chain[...,i]:
        ax[i].plot(chain, alpha=0.25, color='k', drawstyle='steps', rasterized=True)
        yloc = plt.MaxNLocator(3)
        ax[i].yaxis.set_major_locator(yloc)
#        ax[i].set_yticks(fontsize=8)
fig.subplots_adjust(hspace=0)
plt.yticks(fontsize = 8)
plt.savefig('../figures/smc_population_chains.pdf', rasterized=True)


# Corner plot
labels = [r"$M_1$", r"$M_2$", r"$a$", r"$e$", r"$v_k$", r"$\theta$", r"$\phi$", r"$\alpha_{\rm b}$", r"$\delta_{\rm b}$", r"$t_{\rm b}$"]
plt_range = ([7,24], [2.5,15], [0,1500], [0,1], [0,450], [0,np.pi], [0,2.0*np.pi], [6,21], [-76,-70], [0,70])
hist2d_kwargs = {"plot_datapoints" : False}
fig = corner.corner(sampler.flatchain, labels=labels, range=plt_range, bins=40, max_n_ticks=4, **hist2d_kwargs)
plt.rc('font', size=18)
plt.savefig('../figures/smc_population_corner.pdf')
plt.rc('font', size=10)


# M1 vs. M2
fig, host = plt.subplots(figsize=(5,5))
plt_range = ([7,25], [2.5,15])
corner.hist2d(sampler.flatchain.T[0], sampler.flatchain.T[1], bins=40, range=plt_range, plot_datapoints=False)
plt.xlabel(r"$M_1$", size=16)
plt.ylabel(r"$M_2$", size=16)
plt.savefig('../figures/smc_population_M1_M2.pdf')

# Orbital separation vs. eccentricity
fig, host = plt.subplots(figsize=(5,5))
plt_range = ([0,2000], [0,1])
corner.hist2d(sampler.flatchain.T[2], sampler.flatchain.T[3], bins=40, range=plt_range, plot_datapoints=False)
plt.xlabel(r"$a$", size=16)
plt.ylabel(r"$e$", size=16)
plt.savefig('../figures/smc_population_A_ecc.pdf')



# Now, we want to run all the sampler positions forward to
# get the distribution today of HMXBs
l = len(sampler.flatchain)

HMXB_ra = np.zeros(l)
HMXB_dec = np.zeros(l)
HMXB_Porb = np.zeros(l)
HMXB_ecc = np.zeros(l)
HMXB_M2 = np.zeros(l)
HMXB_vsys = np.zeros(l)
HMXB_Lx = np.zeros(l)

for i in np.arange(l):

    s = sampler.flatchain[i]

    # Run forward model
    data_out = pop_synth.full_forward(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[9])

    # Get a random phi for position angle
    ran_phi = pop_synth.get_phi(1)

    # Get the new ra and dec
    ra_out, dec_out = pop_synth.get_new_ra_dec(s[7], s[8], data_out[7], ran_phi)

    # Get the output orbital period
    Porb = binary_evolve.A_to_P(data_out[0], data_out[1], data_out[5])

    # Save outputs
    HMXB_ra[i] = ra_out
    HMXB_dec[i] = dec_out
    HMXB_Porb[i] = Porb
    HMXB_ecc[i] = data_out[6]
    HMXB_M2[i] = data_out[1]
    HMXB_vsys[i] = data_out[3]
    HMXB_Lx[i] = data_out[2]


# HMXB Orbital period vs. eccentricity
fig, ax = plt.subplots(3, 1, figsize=(5,5))
plt_range = ([0, 1.0e7], [0,1])
corner.hist2d(HMXB_Porb, HMXB_ecc, ax=ax[0], bins=40, range=plt_range, plot_datapoints=False)
plt.xlabel(r"$P_{\rm orb}$", size=16)
plt.ylabel(r"$e$", size=16)

plt_range = ([5, 20], [0,200])
corner.hist2d(HMXB_M2, HMXB_vsys, ax=ax[1], bins=40, range=plt_range, plot_datapoints=False)
plt.xlabel(r"$M_2$", size=16)
plt.ylabel(r"$v_{\rm sys}$", size=16)

plt_range = ([0, 70], [0,200])
corner.hist2d(sampler.flatchain.T[9], data_out[7]*180.0/np.pi*3600.0, ax=ax[2], bins=40, range=plt_range, plot_datapoints=False)
plt.xlabel(r"$t_i$", size=16)
plt.ylabel(r"$\theta$", size=16)

plt.savefig('../figures/smc_population_HMXB.pdf')





# Birth location
fig, host = plt.subplots(figsize=(5,5))
sf_history.get_SMC_plot(30.0)
plt_kwargs = {'colors':'k'}
density_contour.density_contour(HMXB_ra, HMXB_dec, nbins_x=40, nbins_y=40, **plt_kwargs)
plt.savefig('../figures/smc_population_HMXB_ra_dec.pdf')
