from xrb.src.core import *
set_data_path("../data")
import emcee
from xrb.src import stats

# Record time
start_time = time.time()
sampler = stats.run_emcee_population(nburn=5, nsteps=10)

print "Simulation took", time.time()-start_time, "seconds"

print "Acceptance fraction", sampler.acceptance_fraction
