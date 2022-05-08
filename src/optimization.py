import numpy as np
import cvxpy as cp
import itertools
from utils import *



''' LP utils functions '''
# mu is L x q
def convert_mu_to_seq(mu):
    # TODO: not sure when fractional solutions will appear 
    # and if it is valid to max over fractional solutions
    assignments = np.argmax(mu, axis=1) 
    return ''.join([idx_to_aa[a] for a in assignments])


# project h, J to codon space, i.e. q = 64
# return: h_mtx_codon is L by q
def convert_h_to_codon_space(h_mtx):
    L = int(h_mtx.size / 21)
    h_mtx_codon = np.zeros((L, 64))
    for i in range(L):
        for j in range(21):
            for codon in aa_to_codons[idx_to_aa[j]]:
                h_mtx_codon[i, codon_to_idx[codon]] = h_mtx[i * 21 + j]
    return h_mtx_codon


# return: J_mtx_codon is L by L by q by q
def convert_J_to_codon_space(J_mtx):
    L = int(J_mtx.shape[0] / 21)
    J_mtx_codon = np.zeros((L, L, 64, 64))
    for i in range(L):
        for j in range(L):
            for k in range(21):
                for l in range(21):
                    for c1 in aa_to_codons[idx_to_aa[k]]:
                        for c2 in aa_to_codons[idx_to_aa[l]]:
                            J_mtx_codon[i, j, codon_to_idx[c1], codon_to_idx[c2]] = J_mtx[i * 21 + k, j * 21 + l]
    return J_mtx_codon


def convert_mu_to_codon(mu):
    assignments = np.argmax(mu, axis=1)
    return [idx_to_codon[i] for i in assignments]


def convert_mu_codon_to_seq(mu):
    codon_list = convert_mu_to_codon(mu)
    return ''.join([codon_to_aa[c] for c in codon_list])



''' optimization functions'''
# ILP MAP calculation for one potts
# boolean=True: ILP
def lp_map(h_mtx, J_mtx, boolean=True):
    q = 21
    L = int(h_mtx.size / q)
    mu_ind = cp.Variable((L, q), boolean=boolean) # variable for each individual position
    mu_pairs = [cp.Variable((q, q), boolean=boolean) for _ in range(int(L*(L-1)/2))]
    mu_pair_dict = dict(zip(itertools.combinations(np.arange(L), 2), mu_pairs)) # variables for each pair of positions

    constraints_ind = [
        cp.sum(mu_ind, axis=1) == 1, # marginal constraints
        mu_ind >= 0, # nonnegative constraints
    ]

    constraints_pair = [
        cp.sum(mu) == 1 for idx, mu in mu_pair_dict.items() # marginal constraints
    ]
    constraints_pair += [
        mu >= 0 for idx, mu in mu_pair_dict.items() # nonnegative constraints
    ]

    constraints_pair += [
        cp.sum(mu, axis=0) == mu_ind[idx[1]] for idx, mu in mu_pair_dict.items() # consistency constraints for 2nd position
    ]
    constraints_pair += [
        cp.sum(mu, axis=1) == mu_ind[idx[0]] for idx, mu in mu_pair_dict.items() # consistency constraints for 1st position
    ]

    constraints = constraints_ind + constraints_pair
    
    obj_ind = cp.sum(cp.multiply(h_mtx, mu_ind))
    obj_pair = cp.sum([cp.sum(cp.multiply(J_mtx[idx[0], idx[1]], mu)) for idx, mu in mu_pair_dict.items()])
    obj = cp.Maximize(obj_ind + obj_pair)
    prob = cp.Problem(obj, constraints)
    prob.solve()
    
    return mu_ind, prob


# double encoding MAP using ILP / LP
def double_encode_lp_map(h_mtx_x, J_mtx_x, h_mtx_y, J_mtx_y, start_pos, wt_nt_seq, boolean=True):
    # for double encoding, we need to work in the space of codon
    q = 64
    Lx = h_mtx_x.shape[0]
    Ly = h_mtx_y.shape[0]
    check_start_pos(Lx, Ly, start_pos)

    mu_ind_x = cp.Variable((Lx, q), boolean=boolean) # variable for each individual position
    mu_pairs_x = [cp.Variable((q, q), boolean=boolean) for _ in range(int(Lx * (Lx - 1) / 2))] # variables for each pair of positions
    mu_pair_dict_x = dict(zip(itertools.combinations(np.arange(Lx), 2), mu_pairs_x))
    mu_ind_y = cp.Variable((Ly, q), boolean=boolean)
    mu_pairs_y = [cp.Variable((q, q), boolean=boolean) for _ in range(int(Ly * (Ly - 1) / 2))]
    mu_pair_dict_y = dict(zip(itertools.combinations(np.arange(Ly), 2), mu_pairs_y)) 

    constraints_ind = [
        cp.sum(mu_ind_x, axis=1) == 1, # marginal constraints
        mu_ind_x >= 0, # nonnegative constraints
        cp.sum(mu_ind_y, axis=1) == 1, # marginal constraints
        mu_ind_y >= 0, # nonnegative constraints
    ]

    # pair constraints for x
    constraints_pair = [
        cp.sum(mu) == 1 for idx, mu in mu_pair_dict_x.items() # marginal constraints
    ]
    constraints_pair += [
        mu >= 0 for idx, mu in mu_pair_dict_x.items() # nonnegative constraints
    ]
    
    constraints_pair += [
        cp.sum(mu, axis=0) == mu_ind_x[idx[1]] for idx, mu in mu_pair_dict_x.items() # consistency constraints for 2nd position
    ]
    constraints_pair += [
        cp.sum(mu, axis=1) == mu_ind_x[idx[0]] for idx, mu in mu_pair_dict_x.items() # consistency constraints for 1st position
    ]
    
    # pair constraints for y
    constraints_pair += [
        cp.sum(mu) == 1 for idx, mu in mu_pair_dict_y.items() # marginal constraints
    ]
    constraints_pair += [
        mu >= 0 for idx, mu in mu_pair_dict_y.items() # nonnegative constraints
    ]

    constraints_pair += [
        cp.sum(mu, axis=0) == mu_ind_y[idx[1]] for idx, mu in mu_pair_dict_y.items() # consistency constraints for 2nd position
    ]
    constraints_pair += [
        cp.sum(mu, axis=1) == mu_ind_y[idx[0]] for idx, mu in mu_pair_dict_y.items() # consistency constraints for 1st position
    ]
    
    
    start_pos_x = start_pos // 3 # the corresponding index in x of the starting position
    
    # constrain the assignments of x before the double-encoding region
    constraints_wt_out = []
    if start_pos_x > 0:
        constraints_wt_out += [
            mu_ind_x[i, codon_to_idx[wt_nt_seq[i * 3 : (i + 1) * 3]]] == 1 for i in range(start_pos_x)
        ]
        constraints_wt_out += [
            mu_ind_x[i, :codon_to_idx[wt_nt_seq[i * 3 : (i + 1) * 3]]] == 0 for i in range(start_pos_x)
        ]
        constraints_wt_out += [
            mu_ind_x[i, codon_to_idx[wt_nt_seq[i * 3 : (i + 1) * 3]] + 1:] == 0 for i in range(start_pos_x)
            if codon_to_idx[wt_nt_seq[i * 3 : (i + 1) * 3]] < q - 1
        ]
        
    # constrain the assignments of x after the double-encoding region
    if start_pos_x + Ly < Lx - 1:
        constraints_wt_out += [
            mu_ind_x[i, codon_to_idx[wt_nt_seq[i * 3 : (i + 1) * 3]]] == 1 for i in range(start_pos_x + Ly + 1, Lx)
        ]
        constraints_wt_out += [
            mu_ind_x[i, :codon_to_idx[wt_nt_seq[i * 3 : (i + 1) * 3]]] == 0 for i in range(start_pos_x + Ly + 1, Lx)
        ]
        constraints_wt_out += [
            mu_ind_x[i, codon_to_idx[wt_nt_seq[i * 3 : (i + 1) * 3]] + 1:] == 0 for i in range(start_pos_x + Ly + 1, Lx)
            if codon_to_idx[wt_nt_seq[i * 3 : (i + 1) * 3]] < q - 1
        ]
    
    # constrain the first and last codon of x in the double encoding region 
    constraints_wt_in = []
    first_overlap_codon = wt_nt_seq[start_pos_x * 3 : (start_pos_x + 1) * 3]
    last_overlap_codon = wt_nt_seq[(start_pos_x + Ly) * 3 : (start_pos_x + Ly + 1) * 3]
    if start_pos % 3 == 1:
        constraints_wt_in += [
            mu_ind_x[start_pos_x, i] == 0 for i in range(64)
            if idx_to_codon[i][0] != first_overlap_codon[0]
        ]
        constraints_wt_in += [
            mu_ind_x[start_pos_x + Ly, i] == 0 for i in range(64)
            if idx_to_codon[i][1:] != last_overlap_codon[1:]
        ]
    elif start_pos % 3 == 2:
        constraints_wt_in += [
            mu_ind_x[start_pos_x, i] == 0 for i in range(64)
            if idx_to_codon[i][:2] != first_overlap_codon[:2]
        ]
        constraints_wt_in += [
            mu_ind_x[start_pos_x + Ly, i] == 0 for i in range(64)
            if idx_to_codon[i][-1] != last_overlap_codon[-1]
        ]
    
    
    # constrain the compatibility of overlapping codons, depending on how the reading frames overlap
    constraints_cp = []
    if start_pos % 3 == 1:
        # coupling constraints
        constraints_cp += [
            mu_ind_x[start_pos_x : start_pos_x + Ly, idx[0]] + mu_ind_y[:, idx[1]] <= 1
            for idx, compat in codon_compat_2.items()
            if not compat
        ]
        constraints_cp += [
            mu_ind_y[:, idx[0]] + mu_ind_x[start_pos_x + 1 : start_pos_x + Ly + 1, idx[1]] <= 1
            for idx, compat in codon_compat_1.items()
            if not compat
        ]
    elif start_pos % 3 == 2:
        constraints_cp += [
            mu_ind_x[start_pos_x : start_pos_x + Ly, idx[0]] + mu_ind_y[:, idx[1]] <= 1
            for idx, compat in codon_compat_1.items()
            if not compat
        ]
        constraints_cp += [
            mu_ind_y[:, idx[0]] + mu_ind_x[start_pos_x + 1 : start_pos_x + Ly + 1, idx[1]] <= 1
            for idx, compat in codon_compat_2.items()
            if not compat
        ]
    
    constraints = constraints_ind + constraints_pair + constraints_cp + constraints_wt_in + constraints_wt_out
    
    obj_ind_x = cp.sum(cp.multiply(h_mtx_x, mu_ind_x))
    # times 2 to match cameos implementation of energy calculation
    obj_pair_x = 2 * cp.sum([cp.sum(cp.multiply(J_mtx_x[idx[0], idx[1]], mu)) for idx, mu in mu_pair_dict_x.items()])
    obj_ind_y = cp.sum(cp.multiply(h_mtx_y, mu_ind_y))
    obj_pair_y = 2 * cp.sum([cp.sum(cp.multiply(J_mtx_y[idx[0], idx[1]], mu)) for idx, mu in mu_pair_dict_y.items()])

    obj = cp.Maximize(obj_ind_x + obj_pair_x + obj_ind_y + obj_pair_y)
    prob = cp.Problem(obj, constraints)
    prob.solve()

    results = {
        'mu_ind_x': mu_ind_x.value,
        'mu_ind_y': mu_ind_y.value,
        'start_pos': start_pos,
        'wt_nt_seq': wt_nt_seq
    }
    return results


# the h, J matrices here are in amino acid space
def process_ilp_result(results, h_mtx_x, J_mtx_x, h_mtx_y, J_mtx_y): 
    codon_x = convert_mu_to_codon(results['mu_ind_x'])
    seq_x = convert_mu_codon_to_seq(results['mu_ind_x'])
    energy_x = compute_energy(seq_x, h_mtx_x, J_mtx_x)

    codon_y = convert_mu_to_codon(results['mu_ind_y'])
    seq_y = convert_mu_codon_to_seq(results['mu_ind_y'])
    energy_y = compute_energy(seq_y, h_mtx_y, J_mtx_y)
    
    processed_results = {
        'energy_x': energy_x,
        'energy_y': energy_y,
        'x': seq_x,
        'y': seq_y,
        'codon_x': codon_x,
        'codon_y': codon_y,
        'start_pos': results['start_pos'],
        'wt_nt_seq': results['wt_nt_seq']
    }
    return processed_results