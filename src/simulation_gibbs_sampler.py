# sample h, J, run gibbs sampler for different lambda
# save results of gibbs sampler

import numpy as np
import pickle
from sampling import *
import time

np.random.seed(1)
Lx = 4
Ly = 2
q = 21
num_chains = 100
num_mcmc_steps = 5000
lamb_list = [0, 1, 1.5, 2, 3]

h_mtx_x = sample_h(Lx, q)
J_mtx_x = sample_J(Lx, q)
h_mtx_y = sample_h(Ly, q)
J_mtx_y = sample_J(Ly, q)
# wt is fixed to be a random sequence
wt_nt_seq = sample_nt_seq(Lx * 3)

start_time = time.time()
# gibbs sampling
gibbs_results = run_gibbs_for_lamb(lamb_list, h_mtx_x, J_mtx_x, h_mtx_y, J_mtx_y, wt_nt_seq, start_pos_prior=None, num_chains=num_chains, num_mcmc_steps=num_mcmc_steps)
end_time = time.time()
print('gibbs sampling took {:.2}s'.format(end_time - start_time))

# get map
map_result = brute_force_map_double_encoding(h_mtx_x, J_mtx_x, h_mtx_y, J_mtx_y, start_pos=None, wt_nt_seq=wt_nt_seq)
gibbs_results['map'] = map_result

with open('results/gibbs_Lx{}_Ly{}_nchains{}_nmcsteps{}.pkl'.format(Lx, Ly, num_chains, num_mcmc_steps), 'wb') as f:
	pickle.dump(gibbs_results, f)

