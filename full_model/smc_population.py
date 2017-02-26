# SMC Population
from xrb.src.core import *
set_data_path("../data")

from astropy.coordinates import SkyCoord
from astropy import units as u
import pickle

from xrb.src import stats

c.sf_scheme = "SMC" 

# Record time
start_time = time.time()

sampler = stats.run_emcee_population(nburn=10000, nsteps=500000, nwalkers=40)


print "Simulation took", time.time()-start_time, "seconds"


pickle.dump( sampler, open( INDATA("SMC_MCMC_sampler_test.obj"), "wb" ) )


print "Production run:"
print "Acceptance fraction", sampler.acceptance_fraction
print "Autocorrelation lengths", sampler.acor

