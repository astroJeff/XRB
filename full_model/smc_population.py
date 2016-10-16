# SMC Population
from xrb.src.core import *
set_data_path("../data")

from astropy.coordinates import SkyCoord
from astropy import units as u
import pickle

from xrb.src import stats

# Record time
start_time = time.time()

sampler = stats.run_emcee_population(nburn=10000, nsteps=50000, nwalkers=320)


print "Simulation took", time.time()-start_time, "seconds"


print "Production run:"
print "Autocorrelation lengths", sampler.acor
print "Acceptance fraction", sampler.acceptance_fraction


pickle.dump( sampler, open( INDATA("SMC_MCMC_sampler.obj"), "wb" ) )
