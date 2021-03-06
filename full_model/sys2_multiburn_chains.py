import matplotlib
matplotlib.use('Agg')

import pickle
import matplotlib.pyplot as plt
from matplotlib import font_manager
import emcee
import matplotlib.gridspec as gridspec

from xrb.src import stats
from xrb import src
import numpy as np

print "Loading data..."
sampler1 = pickle.load( open( "../data/sys2_MCMC_multiburn_burn1.obj", "rb" ) )
sampler2 = pickle.load( open( "../data/sys2_MCMC_multiburn_burn2.obj", "rb" ) )
sampler3 = pickle.load( open( "../data/sys2_MCMC_multiburn_burn3.obj", "rb" ) )
sampler4 = pickle.load( open( "../data/sys2_MCMC_multiburn_burn4.obj", "rb" ) )
sampler = pickle.load( open( "../data/sys2_MCMC_multiburn_sampler.obj", "rb" ) )
print "Finished loading data."



M1 = 9.0
M2 = 4.0
A = 500.0
ecc = 0.5
v_k = 100.0
theta = 2.7
phi = 1.2
t_b = 50.0
ra_b = 15.8
dec_b = -72.1

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


ymin = [7.5, 2.0, 50.0, 0.0, 0.0, 1.5, 0.0, 14.8, -72.6, 20]
ymax = [13.0, 8.0, 700.0, 1.0, 400.0, 3.2, np.pi, 16.6, -71.8, 75.0]

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

#            if i == 9:
#                if j == 0: ax[i,j].set_xticklabels([0])
#                if j == 1: ax[i,j].set_xticklabels([10000])
#                if j == 2: ax[i,j].set_xticklabels([20000])
#                if j == 3: ax[i,j].set_xticklabels([30000])


#        if j == 4:
#            ax[i,j].set_xticks([0, 10000, 20000, 30000, 40000, 50000])
#            if i == 9: ax[i,j].set_xticklabels([40000, 50000, 60000, 70000, 80000, 90000])

        if i == 0: ax[i,j].set_yticks([8,10,12])
        if i == 1: ax[i,j].set_yticks([4, 6])
        if i == 2: ax[i,j].set_yticks([200, 400, 600])
        if i == 3: ax[i,j].set_yticks([0.25, 0.5, 0.75])
        if i == 4: ax[i,j].set_yticks([100, 200, 300])
        if i == 5: ax[i,j].set_yticks([np.pi/2., 3.0/4.0*np.pi, np.pi])
        if i == 6: ax[i,j].set_yticks([1.0/4.0*np.pi, np.pi/2., 3.0/4.0*np.pi])
        if i == 7: ax[i,j].set_yticks([15., 16.])
        if i == 8: ax[i,j].set_yticks([-72, -72.2, -72.4])
        if i == 9: ax[i,j].set_yticks([20, 40, 60])

        # Add truths as a red lines across entire plot
        ax[i,j].axhline(truths[i], color='r')
        ax[i,j].set_ylim(ymin[i], ymax[i])

    ax[i,0].set_ylabel(labels[i])



# Set y-axis tick labels
ax[0,0].set_yticklabels([8,10,12])
ax[1,0].set_yticklabels([4, 6])
ax[2,0].set_yticklabels([200, 400, 600])
ax[3,0].set_yticklabels([0.25, 0.5, 0.75])
ax[4,0].set_yticklabels([100, 200, 300])
ax[5,0].set_yticklabels([r'$\pi/2$', r'$3 \pi/4$', r'$\pi$'])
ax[6,0].set_yticklabels([r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$'])
ax[7,0].set_yticklabels([15., 16.])
ax[8,0].set_yticklabels([-72.0, -72.2, -72.4])
ax[9,0].set_yticklabels([20, 40, 60])


fig.text(0.5, 0.02, 'Model Number')

fig.subplots_adjust(hspace=0, wspace=0)
plt.subplots_adjust(bottom=0.06, left=0.08, top=0.98, right=0.96)


plt.savefig('../figures/sys2_chain_multiburn.pdf', rasterized=True)



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
