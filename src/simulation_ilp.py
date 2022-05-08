# randomly sample h, J, starting position and wt nt seq for many iterations
# save brute force vs. ilp solutions (sequence and energy)


import numpy as np
from utils import *
from optimization import *
from tqdm import tqdm
import pickle


num_iters = 10
Lx = 4
Ly = 2
q = 21
start_pos_samples = sample_start_pos(Lx, Ly, num_iters)
all_results = []

for i in tqdm(range(num_iters)):
    # randomly sample h, J, start_pos, wt_nt_seq
    h_mtx_x = sample_h(Lx, q)
    J_mtx_x = sample_J(Lx, q)
    h_mtx_y = sample_h(Ly, q)
    J_mtx_y = sample_J(Ly, q)
    h_mtx_x_codon = convert_h_to_codon_space(h_mtx_x)
    J_mtx_x_codon = convert_J_to_codon_space(J_mtx_x)
    h_mtx_y_codon = convert_h_to_codon_space(h_mtx_y)
    J_mtx_y_codon = convert_J_to_codon_space(J_mtx_y)
    start_pos = start_pos_samples[i]
    wt_nt_seq = sample_nt_seq(Lx * 3)

    # get brute force solution
    map_result = brute_force_map_double_encoding(h_mtx_x, J_mtx_x, h_mtx_y, J_mtx_y, start_pos=start_pos, wt_nt_seq=wt_nt_seq)

    # get ilp solution
    result = double_encode_lp_map(h_mtx_x_codon, J_mtx_x_codon, h_mtx_y_codon, J_mtx_y_codon, start_pos=start_pos, wt_nt_seq=wt_nt_seq)
    ilp_result = process_ilp_result(result, h_mtx_x, J_mtx_x, h_mtx_y, J_mtx_y)

    all_results.append((map_result, ilp_result))

with open('results/ilp_Lx{}_Ly{}_niters{}.pkl'.format(Lx, Ly, num_iters), 'wb') as f:
    pickle.dump(all_results, f)