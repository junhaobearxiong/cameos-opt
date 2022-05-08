# sample h, J, run gibbs sampler for different lambda for the following setting
# 	- start_pos is fixed to be optimal, wt_nt_seq is fixed to be optimal
#	- start_pos is sampled uniformly random, wt_nt_seq is fixed to be optimal
#	- start_pos is sampled uniformly random, wt_nt_seq is fixed to be a random non-optimal sequence 
# save results of gibbs sampler

import numpy as np
from sampling import *

np.random.seed(1)