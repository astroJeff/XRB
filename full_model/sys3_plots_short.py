# Create plots from saved pickles of J0045-7319 simulations
from src.core import *
set_data_path("../data")

import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib.gridspec as gridspec
import corner
import pickle
from astropy.coordinates import SkyCoord
from astropy import units as u

from xrb.SF_history import sf_history
from xrb.src import stats


# System 3 test parameters
M1_true = 9.0
M2_true = 4.0
A_true = 300.0
ecc_true = 0.5
v_k_true = 100.0
theta_true = 2.4
phi_true = 1.2
ra_true = 15.8
dec_true = -72.1
t_b_true = 40.0



# Load pickled data
sampler = pickle.load( open( INDATA("sys3_MCMC_multiburn_sampler.obj"), "rb" ) )





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
truths = [M1_true, M2_true, A_true, ecc_true, v_k_true, theta_true, phi_true, ra_true, dec_true, t_b_true]
hist2d_kwargs = {"plot_datapoints" : False}
fig = corner.corner(sampler.flatchain, fig=fig, labels=labels, truths=truths, max_n_ticks=4, **hist2d_kwargs)

#ax2 = plt.subplot2grid((5,5), (0,3), colspan=2, rowspan=2)
ra_out = sampler.flatchain.T[7]
dec_out = sampler.flatchain.T[8]
ra_obs = 15.9
dec_obs = -72.05
gs = gridspec.GridSpec(2, 2,
                       width_ratios=[3,2],
                       height_ratios=[2,3]
                       )
sf_history.get_SMC_plot_polar(50, fig_in=fig, gs=gs[1], ra_dist=ra_out, dec_dist=dec_out, ra=ra_obs, dec=dec_obs, xcenter=-0.7, ycenter=17.8, xwidth=0.5, ywidth=0.5, xgrid_density=6)


# Shift axis labels
for i in np.arange(10):
    ax[i,0].yaxis.set_label_coords(-0.5, 0.5)
    ax[9,i].xaxis.set_label_coords(0.5, -0.5)

# Set declination ticks
ax[9,8].set_xticks([-72.1, -71.9])
ax[9,8].set_xticklabels(["-72.1", "-71.9"])
for i in np.arange(8):
    ax[8,i].set_yticks([-72.1, -71.9])
ax[8,0].set_yticklabels(["-72.1", "-71.9"])

# Set theta ticks
for i in np.arange(5)+5:
    ax[i,5].set_xticks([np.pi/2., 3.*np.pi/4., np.pi])
    ax[i,5].set_xlim(1.4, np.pi)
ax[9,5].set_xticklabels([r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$'])
for i in np.arange(4):
    ax[5,i].set_yticks([np.pi/2., 3.*np.pi/4., np.pi])
    ax[5,i].set_ylim(1.4, np.pi)
ax[5,0].set_yticklabels([r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$'])


# Set phi ticks
for i in np.arange(4)+6:
    ax[i,6].set_xticks([np.pi/4., np.pi/2., 3.*np.pi/4.])
    ax[i,6].set_xlim(0.0, np.pi)
ax[9,6].set_xticklabels([r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$'])
for i in np.arange(6):
    ax[6,i].set_yticks([np.pi/4., np.pi/2., 3.*np.pi/4.])
    ax[6,i].set_ylim(0.0, np.pi)
ax[6,0].set_yticklabels([r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$'])




plt.subplots_adjust(bottom=0.07, left=0.07, top=0.97)

plt.savefig('../figures/sys3_corner_multiburn.pdf')


