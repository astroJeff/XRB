from src.core import *
import emcee
import stats

# Record time
start_time = time.time()
sampler = stats.run_emcee_population(nburn=5, nsteps=10)

print "Simulation took", time.time()-start_time, "seconds"

print "Acceptance fraction", sampler.acceptance_fraction
