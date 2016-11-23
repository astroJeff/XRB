from src.core import *
set_data_path("../data")

import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import emcee
import matplotlib.gridspec as gridspec

from xrb.src import stats


print "Loading data..."
sampler1 = pickle.load( open( INDATA("sys1_MCMC_multiburn_burn1.obj"), "rb" ) )
sampler2 = pickle.load( open( INDATA("sys1_MCMC_multiburn_burn2.obj"), "rb" ) )
sampler3 = pickle.load( open( INDATA("sys1_MCMC_multiburn_burn3.obj"), "rb" ) )
sampler4 = pickle.load( open( INDATA("sys1_MCMC_multiburn_burn4.obj"), "rb" ) )
sampler = pickle.load( open( INDATA("sys1_MCMC_multiburn_sampler.obj"), "rb" ) )
print "Finished loading data."




M1 = 13.0
M2 = 10.0
A = 150.0
ecc = 0.7
v_k = 250.0
theta = 2.9
phi = 0.9
t_b = 22.0
ra_b = 13.51
dec_b = -72.7


# Truths
truths=[M1, M2, A, ecc, v_k, theta, phi, ra_b, dec_b, t_b]


# Plot properties
fontProperties = {'family':'serif', 'serif':['Times New Roman'], 'weight':'normal', 'size':10}
ticks_font = font_manager.FontProperties(family='Times New Roman', style='normal', \
                                         weight='normal', stretch='normal', size=10)
plt.rc('font', **fontProperties)



# gridspec
gs = gridspec.GridSpec(10, 5,
                       width_ratios=[1,1,1,1,5],
                       height_ratios=[1,1,1,1,1,1,1,1,1,1]
                       )

################# Chains plot #####################
fig, ax = plt.subplots(10, 5, figsize=(8.0,9.0))


chains = np.array([sampler1, sampler2, sampler3, sampler4, sampler])


ymin = [9.5, 4.0, 0.0, 0.0, 0.0, 0.9, 0.0, 13.0, -73.0, 0]
ymax = [21.0, 17.0, 800.0, 1.0, 600.0, 3.2, 2.0*np.pi, 14.5, -72.2, 40.0]

labels = [r"$M_{\rm 1, i}\ (M_{\odot})$", r"$M_{\rm 2, i}\ (M_{\odot})$", r"$a_{\rm i}\ (R_{\odot})$", \
          r"$e_{\rm i}$", r"$v_{\rm k, i}\ ({\rm km}\ {\rm s}^{-1})$", r"$\theta_{\rm k}\ ({\rm rad.})$", \
          r"$\phi_{\rm k}\ ({\rm rad.})$", r"$\alpha_{\rm i}\ ({\rm deg.})$", \
          r"$\delta_{\rm i}\ ({\rm deg.}) $", r"$t_{\rm i}\ ({\rm Myr})$"]

for i in np.arange(10):
    for j in np.arange(5):
        idx = 5*i + j

        ax[i,j] = plt.subplot(gs[idx])

        for k in np.arange(len(sampler1.chain[...])):

            chain = chains[j].chain[...,i][k]
            ax[i,j].plot(chain, alpha=0.25, color='k', drawstyle='steps', rasterized=True)


        if j != 0: ax[i,j].set_yticklabels([])
        if i != 9: ax[i,j].set_xticklabels([])


        if j != 4:
            ax[i,j].set_xticks([0])

            if i == 9:
                if j == 0: ax[i,j].set_xticklabels([0])
                if j == 1: ax[i,j].set_xticklabels([10000])
                if j == 2: ax[i,j].set_xticklabels([20000])
                if j == 3: ax[i,j].set_xticklabels([30000])


        if j == 4:
            ax[i,j].set_xticks([0, 10000, 20000, 30000, 40000, 50000])
            if i == 9: ax[i,j].set_xticklabels([40000, 50000, 60000, 70000, 80000, 90000])

        if i == 0: ax[i,j].set_yticks([10,14,18])
        if i == 1: ax[i,j].set_yticks([5, 10, 15])
        if i == 2: ax[i,j].set_yticks([200, 400, 600])
        if i == 3: ax[i,j].set_yticks([0.25, 0.5, 0.75])
        if i == 4: ax[i,j].set_yticks([100, 200, 300, 400, 500])
        if i == 5: ax[i,j].set_yticks([np.pi/2., 3.0/4.0*np.pi, np.pi])
        if i == 6: ax[i,j].set_yticks([np.pi/2., np.pi, 3./2.*np.pi])
        if i == 7: ax[i,j].set_yticks([13.5, 14.0])
        if i == 8: ax[i,j].set_yticks([-72.3, -72.5, -72.7, -72.9])
        if i == 9: ax[i,j].set_yticks([10, 20, 30])

        # Add truths as a red lines across entire plot
        ax[i,j].axhline(truths[i], color='r')
        ax[i,j].set_ylim(ymin[i], ymax[i])

    ax[i,0].set_ylabel(labels[i])



# Set y-axis tick labels
ax[0,0].set_yticklabels([10,14,18])
ax[1,0].set_yticklabels([5, 10, 15])
ax[2,0].set_yticklabels([200, 400, 600])
ax[3,0].set_yticklabels([0.25, 0.5, 0.75])
ax[4,0].set_yticklabels([100, 200, 300, 400, 500])
ax[5,0].set_yticklabels([r'$\pi/2$', r'$3 \pi/4$', r'$\pi$'])
ax[6,0].set_yticklabels([r'$\pi/2$', r'$\pi$', r'$3\pi/2$'])
ax[7,0].set_yticklabels([13.5, 14.])
ax[8,0].set_yticklabels([-72.3, -72.5, -72.7, -72.9])
ax[9,0].set_yticklabels([10, 20, 30])


fig.text(0.5, 0.02, 'Model Number')

fig.subplots_adjust(hspace=0, wspace=0)
plt.subplots_adjust(bottom=0.06, left=0.08, top=0.98, right=0.96)


plt.savefig('../figures/sys1_chain_multiburn.pdf', rasterized=True)



#
# #fig, ax = plt.subplots(sampler1.dim, 5, sharex=False, figsize=(18.0,20.0))
# for i in range(sampler1.dim):
#     for j in np.arange(len(sampler1.chain[...])):
#
#         chain1 = sampler1.chain[...,i][j]
#         ax[i,0].plot(chain1, alpha=0.25, color='k', drawstyle='steps')
#
#         chain2 = sampler2.chain[...,i][j]
#         ax[i,1].plot(chain2, alpha=0.25, color='k', drawstyle='steps')
#
#         chain3 = sampler3.chain[...,i][j]
#         ax[i,2].plot(chain3, alpha=0.25, color='k', drawstyle='steps')
#
#         chain4 = sampler4.chain[...,i][j]
#         ax[i,3].plot(chain4, alpha=0.25, color='k', drawstyle='steps')
#
#         chain5 = sampler.chain[...,i][j]
#         ax[i,4].plot(chain5, alpha=0.25, color='k', drawstyle='steps')
#
#     # Remove tick labels from y-axis
#     ax[i,1].set_yticklabels([])
#     ax[i,2].set_yticklabels([])
#     ax[i,3].set_yticklabels([])
#     ax[i,4].set_yticklabels([])
#
#     # Remove all but bottom x-axis ticks
#     if i != sampler1.dim-1:
#         ax[i,0].set_xticklabels([])
#         ax[i,1].set_xticklabels([])
#         ax[i,2].set_xticklabels([])
#         ax[i,3].set_xticklabels([])
#         ax[i,4].set_xticklabels([])
#
#     # Add truths as a red lines across entire plot
#     ax[i,0].axhline(truths[i], color='r')
#     ax[i,1].axhline(truths[i], color='r')
#     ax[i,2].axhline(truths[i], color='r')
#     ax[i,3].axhline(truths[i], color='r')
#     ax[i,4].axhline(truths[i], color='r')
#
# # Make all plots have the same y-range - plots must already be created
# for i in range(sampler1.dim):
#     ymin = min(ax[i,0].get_ylim()[0],ax[i,1].get_ylim()[0],ax[i,2].get_ylim()[0])
#     ymax = max(ax[i,0].get_ylim()[1],ax[i,1].get_ylim()[1],ax[i,2].get_ylim()[1])
#
#     ax[i,0].set_ylim(ymin, ymax)
#     ax[i,1].set_ylim(ymin, ymax)
#     ax[i,2].set_ylim(ymin, ymax)
#     ax[i,3].set_ylim(ymin, ymax)
#     ax[i,4].set_ylim(ymin, ymax)
#
# fig.subplots_adjust(hspace=0, wspace=0)
# plt.yticks(fontsize = 8)
# plt.savefig('../figures/sys2_chain_multiburn.pdf')
