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
        ax[i].plot(chain, alpha=0.25, color='k', drawstyle='steps')
        yloc = plt.MaxNLocator(3)
        ax[i].yaxis.set_major_locator(yloc)
#        ax[i].set_yticks(fontsize=8)
fig.subplots_adjust(hspace=0)
#plt.yticks(fontsize = 8)
plt.savefig('../figures/smc_population_chains.pdf')


# Corner plot
labels = [r"$M_1$", r"$M_2$", r"$A$", r"$e$", r"$v_k$", r"$\theta$", r"$\phi$", r"$\alpha_{\rm b}$", r"$\delta_{\rm b}$", r"$t_{\rm b}$"]
fig = corner.corner(sampler.flatchain, labels=labels)
plt.rc('font', size=18)
plt.savefig('../figures/smc_population_corner.pdf')
plt.rc('font', size=10)


# M1 vs. M2
#plt.subplot(4,1,1)
corner.hist2d(sampler.flatchain.T[0], sampler.flatchain.T[1])
#plt.scatter(init_params["M1"], init_params["M2"], color='r')
plt.xlabel(r"$M_1$", size=16)
plt.ylabel(r"$M_2$", size=16)
#plt.xlim(8.5, 12.0)
#plt.ylim(3.0, 4.5)
plt.savefig('../figures/smc_population_M1_M2.pdf')

# Orbital separation vs. eccentricity
#plt.subplot(4,1,2)
corner.hist2d(sampler.flatchain.T[2], sampler.flatchain.T[3])
#plt.scatter(init_params["A"], init_params["ecc"], color='r')
plt.xlabel(r"$a$", size=16)
plt.ylabel(r"$e$", size=16)
#plt.xlim(10.0, 1500.0)
#plt.ylim(0.0, 1.0)
plt.savefig('../figures/smc_population_A_ecc.pdf')

# v_kick vs. theta
corner.hist2d(sampler.flatchain.T[4], sampler.flatchain.T[5])
plt.xlabel(r"$v_k$")
plt.ylabel(r"$\theta$")
plt.savefig("../figures/smc_population_vk_theta.pdf")



# Now, we want to run all the sampler positions forward to
# get the distribution today of HMXBs
HMXB_ra = np.array([])
HMXB_dec = np.array([])
HMXB_Porb = np.array([])
HMXB_ecc = np.array([])
HMXB_M2 = np.array([])
HMXB_vsys = np.array([])
HMXB_Lx = np.array([])

for s in sampler.flatchain():

    # Run forward model
    data_out = pop_synth.full_forward(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[9])

    # Get a random phi for position angle
    ran_phi = pop_synth.get_phi(1)

    # Get the new ra and dec
    ra_out, dec_out = pop_synth.get_new_ra_dec(s[7], s[8], data_out[7], ran_phi)

    # Get the output orbital period
    Porb = binary_evolve.A_to_P(data_out[0], data_out[1], data_out[5])

    # Save outputs
    HMXB_ra = np.append(HMXB_ra, ra_out)
    HMXB_dec = np.append(HMXB_dec, dec_out)
    HMXB_Porb = np.append(HMXB_Porb, Porb)
    HMXB_ecc = np.append(HMXB_ecc, data_out[6])
    HMXB_M2 = np.append(HMXB_M2, data_out[1])
    HMXB_vsys = np.append(HMXB_vsys, data_out[3])
    HMXB_Lx = np.append(HMXB_Lx, data_out[2])


# HMXB Orbital period vs. eccentricity
corner.hist2d(HMXB_Porb, HMXB_ecc)
plt.xlabel(r"$P_{\rm orb}$", size=16)
plt.ylabel(r"$e$", size=16)
plt.savefig('../figures/smc_population_HMXB_P_ecc.pdf')

# X-ray luminosity
plt.hist(HMXB_Lx, color='k', histtype='step', bins=50)
plt.xlabel(r"$L_x$", size=16)
plt.savefig('../figures/smc_population_HMXB_Lx.pdf')

# Birth location
sf_history.get_SMC_plot(30.0)
plt_kwargs = {'colors':'k'}
density_contour.density_contour(HMXB_ra, HMXB_dec, nbins_x=40, nbins_y=40, **plt_kwargs)
plt.savefig('../figures/smc_population_HMXB_ra_dec.pdf')
