# SMC Population

import sys
from astropy.coordinates import SkyCoord
from astropy import units as u
import pickle
import time

sys.path.append('../src')
import stats

# Record time
start_time = time.time()

sampler = stats.run_emcee_population(nburn=5, nsteps=10)

print "Simulation took", time.time()-start_time, "seconds"


print "Production run:"
print "Autocorrelation lengths", sampler.acor
print "Acceptance fraction", sampler.acceptance_fraction


#pickle.dump( sampler, open( "../data/SMC_MCMC_sampler.obj", "wb" ) )
