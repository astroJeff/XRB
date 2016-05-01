# SMC Population

import sys
from astropy.coordinates import SkyCoord
from astropy import units as u
import pickle
import time

sys.path.append('../stats')
import stats

# Record time
start_time = time.time()

sampler = stats.run_emcee(nburn=10000, nsteps=50000)

print "Simulation took", time.time()-start_time, "seconds"

pickle.dump( sampler, open( "../data/SMC_MCMC_sampler.obj", "wb" ) )
