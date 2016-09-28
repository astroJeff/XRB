import sys
sys.path.append("../")
from src.core import *

# SMC Population
from astropy.coordinates import SkyCoord
from astropy import units as u
import pickle
import time
from src import stats
from binary import load_sse

# Record time
start_time = time.time()

sampler = stats.run_emcee_population(nburn=5, nsteps=10)

print "Simulation took", time.time()-start_time, "seconds"


print "Production run:"
print "Autocorrelation lengths", sampler.acor
print "Acceptance fraction", sampler.acceptance_fraction


#pickle.dump( sampler, open( "../data/SMC_MCMC_sampler.obj", "wb" ) )
