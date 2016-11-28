import numpy as np

import xrb
from xrb.src.core import *
set_data_path("../data")
from xrb.binary import load_sse
import xrb.src.constants as c
from xrb.pop_synth import pop_synth
from xrb.SF_history import sf_history

load_sse.load_sse()
sf_history.load_sf_history()


t_min, t_max = 5.0, 60.0
N_times = (t_max-t_min) * 10
N = 10000

N_good = 0
for t_b in np.arange(N_times)/10.0+t_min:

    HMXB, init_params = pop_synth.create_HMXBs(t_b, N_sys=N)

    N_good = N_good + len(HMXB)

print "Found", N_good, "HMXBs out of", float(N) * float(N_times), "binaries"
