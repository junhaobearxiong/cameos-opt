import numpy as np
import itertools
import cvxpy as cp
from tqdm import tqdm


''' global variables and dictionaries'''

# global variables
bases = [l.upper() for l in 'agct']
bases_pairs = list(itertools.product(bases, bases))
codons = [a+b+c for a in bases for b in bases for c in bases]
amino_acids = 'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG'

# global dictionary
# 'idx' here are used for indexing everywhere, in particular for indexing h, J

# nucleotides and indices
nt_to_idx = dict(zip(bases, np.arange(4)))
idx_to_nt = dict(zip(np.arange(4), bases))


# codons and indices
codon_to_idx = dict(zip(codons, np.arange(64)))
idx_to_codon = dict(zip(np.arange(64), codons))


# amino acids and indices
unique_amino_acids = np.unique([a for a in amino_acids])
# the map between idx in h, J and aa's might depend on the specific ways we get h and J
# so don't define as globals
aa_to_idx = dict(zip(unique_amino_acids, np.arange(21))) 
idx_to_aa = dict(zip(np.arange(21), unique_amino_acids))


# codons and amino acids
codon_to_aa = dict(zip(codons, amino_acids))
aa_to_codons = {}
for i, aa in enumerate(amino_acids):
    if aa in aa_to_codons:
        aa_to_codons[aa] += [codons[i]]
    else:
        aa_to_codons[aa] = [codons[i]]


# calculate compatibility on codon level: 4^3 x 4^3 pairs
# order matters! the first codon is assumed to start first
def get_codon_idx_compat(overlap, codon_to_idx):
    assert overlap == 1 or overlap == 2
    compat_dict = {}
    for c1 in codon_to_idx.keys():
        for c2 in codon_to_idx.keys():
            if overlap == 2:
                compat = c1[1:] == c2[:2]
            else:
                compat = c1[2] == c2[0]
            compat_dict[(codon_to_idx[c1], codon_to_idx[c2])] = compat
    return compat_dict

# codon compatibility dictionary
codon_compat_2 = get_codon_idx_compat(2, codon_to_idx) # keys are indices of codon, values are if the 2 codons are compat
codon_compat_1 = get_codon_idx_compat(1, codon_to_idx)




''' general utils functions '''
# convert a string of bases into a list of codons
def translate_nt_to_codon(nt_seq):
    if isinstance(nt_seq, list):
        nt_seq = ''.join([t.upper() for t in nt_seq])
    elif isinstance(nt_seq, str):
        nt_seq = nt_seq.upper()
    else:
        raise ValueError('`nt_seq` needs to be a list or a str')
    return filter(lambda s: len(s) == 3, [nt_seq[x:x+3] for x in range(0, len(nt_seq), 3)])


# convert a list of codons into a string of amino acids 
def translate_codon_to_aa(codon_list):
    # Separate sequence into codons of length 3 (above), return translations of these codons (below).
    return ''.join([codon_to_aa[codon] for codon in codon_list])


# convert a string of bases into a string of amino acids
def translate_nt_to_aa(nt_seq):
    return translate_codon_to_aa(translate_nt_to_codon(nt_seq))


# implement calculation of energy / pseudolikelihood
# TODO: this is naive, can probably optimize by reformating h and J and do matrix multiplication
# be careful: i need to know which rows of h and J correspond to which amino acids
# compute energy given an amino acid sequence, represented as a string
def compute_energy(seq, h_mtx, J_mtx):
    # check that h, J are defined for the length of this sequence
    assert len(seq) == h_mtx.shape[0]
    assert len(seq) == J_mtx.shape[0]
    assert len(seq) == J_mtx.shape[1]
    
    # the index of the amino acid type for each position
    h_idx = np.array([aa_to_idx[a] for a in seq])
    h_values = h_mtx[np.arange(len(seq)), h_idx]
    
    J_idx = list(zip(*itertools.combinations(h_idx, 2)))
    pairs_as_list = list(zip(*itertools.combinations(np.arange(len(seq)), 2)))
    J_values = J_mtx[pairs_as_list[0], pairs_as_list[1], J_idx[0], J_idx[1]]
    
    return np.sum(h_values) + np.sum(J_values)




''' LP utils functions '''
# mu is L x q
def convert_mu_to_seq(mu):
    # TODO: not sure when fractional solutions will appear 
    # and if it is valid to max over fractional solutions
    assignments = np.argmax(mu, axis=1) 
    return ''.join([idx_to_aa[a] for a in assignments])


# project h, J to codon space, i.e. q = 64
def convert_h_to_codon_space(h_mtx, idx_to_aa):
    h_mtx_codon = np.zeros((h_mtx.shape[0], 64))
    for i in range(h_mtx.shape[0]):
        for j in range(h_mtx.shape[1]):
            for codon in aa_to_codons[idx_to_aa[j]]:
                h_mtx_codon[i, codon_to_idx[codon]] = h_mtx[i, j]
    return h_mtx_codon


def convert_J_to_codon_space(J_mtx, idx_to_aa):
    L = J_mtx.shape[0]
    q = J_mtx.shape[2]
    J_mtx_codon = np.zeros((L, L, 64, 64))
    for i in range(L):
        for j in range(L):
            for k in range(q):
                for l in range(q):
                    for c1 in aa_to_codons[idx_to_aa[k]]:
                        for c2 in aa_to_codons[idx_to_aa[l]]:
                            J_mtx_codon[i, j, codon_to_idx[c1], codon_to_idx[c2]] = J_mtx[i, j, k, l]
    return J_mtx_codon


def convert_mu_to_codon(mu):
    assignments = np.argmax(mu, axis=1)
    return [idx_to_codon[i] for i in assignments]


def convert_mu_codon_to_seq(mu):
    codon_list = convert_mu_to_codon(mu)
    return ''.join([codon_to_aa[c] for c in codon_list])




''' simulation utils functions '''
# implement sampling of h, J
def sample_h(L, q):
    h_mtx = np.random.normal(0, 1, (L, q))
    return h_mtx


def sample_J(L, q):
    J_mtx = np.random.normal(0, 1, (L, L, q, q))
    # enforces each Jij is symmetric
    for i in range(L):
        for j in range(L):
            J_mtx[i, j] = 0.5 * (J_mtx[i, j] + J_mtx[i, j].T)
    # enforces Jij = Jji.T
    J_mtx_sym = np.zeros((L, L, q, q))
    for i in range(L):
        for j in range(L):
            J_mtx_sym[i, j] = 0.5 * (J_mtx[i, j] + J_mtx[j, i].T)
    return J_mtx_sym


def sample_nt_seq(L):
    return ''.join(np.random.choice(bases, L))


def sample_codon_arr(L):
    return np.random.choice(codons, L)


def sample_aa_seq(L):
    return translate_nt_to_aa(sample_nt_seq(3 * L)) 


# brute force find the MAP, for checking correctness
def brute_force_map(h_mtx, J_mtx):
    L, q = h_mtx.shape
    map_seq_energy = 0 
    map_seq = None
    for idx in itertools.product(np.arange(q), repeat=L):
        seq = ''.join([idx_to_aa[idx[i]] for i in range(L)])
        energy = compute_energy(seq, h_mtx, J_mtx)
        if energy > map_seq_energy:
            map_seq = seq
            map_seq_energy = energy
    return map_seq, map_seq_energy


# brute force find the MAP of double encoding soln
# this is very slow, only for the purpose for checking correctness
def brute_force_map_double_encoding(h_mtx_x, J_mtx_x, h_mtx_y, J_mtx_y, start_pos):
    Lx, q = h_mtx_x.shape
    Ly = h_mtx_y.shape[0]
    check_start_pos(Lx, Ly, start_pos)
    
    map_result = {
        'energy': 0,
        'x': None,
        'y': None,
        'nt_seq': None
    }

    for nts in itertools.product(bases, repeat=Lx * 3):
        nt_seq = ''.join(nts)
        aa_seq_x = translate_nt_to_aa(nt_seq)
        aa_seq_y = translate_nt_to_aa(nt_seq[start_pos : start_pos + 3 * Ly])
        energy_x = compute_energy(aa_seq_x, h_mtx_x, J_mtx_x)
        energy_y = compute_energy(aa_seq_y, h_mtx_y, J_mtx_y)
        energy = energy_x + energy_y
        if energy > map_result['energy']:
            map_result['x'] = aa_seq_x
            map_result['y'] = aa_seq_y
            map_result['energy'] = energy
            map_result['nt_seq'] = nt_seq
    return map_result





''' sampling functions (for double encoding) '''
# change the subsequence of `wt_seq` starting at `start_pos` (including) of length `len(subseq)` to be `subseq`
def replace_wt_subseq(wt_seq, subseq, start_pos):
    assert len(wt_seq) >= len(subseq)
    assert start_pos + len(subseq) <= len(wt_seq)
    return wt_seq[:start_pos] + subseq + wt_seq[start_pos + len(subseq):]


def check_start_pos(Lx, Ly, start_pos):
    # x is assumed to be strictly longer than y
    assert Lx > Ly, 'x needs to be longer than y'
    # assume nt seq of y can be fully contained in nt seq of x
    assert start_pos > 0 and start_pos < 3 * (Lx - Ly), 'start_pos needs to be > 0 and < {}'.format(3 * (Lx - Ly))
    # if the reading frames for x and y overlap exactly, then the amino acids they encode will be exactly the same
    assert start_pos % 3 != 0, 'reading frames should not overlap exactly'


# this computes p(Xi = (a, b, c) | X-i) for all Xi
# optionally can specify the codons to sum in `codons_to_sum`
# for each codon k, the numerator of the conditional is the joint probability p(Xi=k, X-i=x-i)
# the denominator is \sum_{k'} p(Xi=k', X-i=x-i)
# `seq`: a string of amino acids
# `index`: the index of `seq` to compute conditionals for, conditioning on the rest of the sequence
# `lamb` is the inverse temperature parameter: X ~ p(exp(E(x) * lamb))
def compute_conditional_for_position(seq, index, h_mtx, J_mtx, lamb=1, codons_to_sum=None):
    if codons_to_sum is None:
        codons_to_sum = codons
    energy_arr = np.zeros(len(codons_to_sum))
    seq_tmp = list(seq)
    
    for i, codon in enumerate(codons_to_sum):
        seq_tmp[index] = codon_to_aa[codon]
        energy_arr[i] = compute_energy(''.join(seq_tmp), h_mtx, J_mtx)
    
    energy_arr *= lamb
    return np.exp(energy_arr) / np.sum(np.exp(energy_arr))


# conditional sampler for 1 iteration of gibbs sampling for double encoding
# samples every nt of a sequence that's within the double encoding region
# `nt_seq`: a string of nucleotides 
def conditional_sampler(nt_seq, h_mtx_x, J_mtx_x, h_mtx_y, J_mtx_y, start_pos, lamb=1):
    Lx = h_mtx_x.shape[0]
    Ly = h_mtx_y.shape[0]
    nt_list_tmp = list(nt_seq)
    # seq_x, seq_y are amino acid sequences of x, y
    seq_x = translate_nt_to_aa(nt_list_tmp)
    seq_y = translate_nt_to_aa(nt_list_tmp[start_pos : start_pos + 3 * Ly])
    
    # i iterates over amino acidx of y
    for i in range(Ly):
        # j iterates over amino acids of x
        # so 3 * j + k - 1 index the kth nt of the jth codon of x (k = 1, 2, 3)
        j = start_pos // 3 + i

        # when reading frames overlap with 2 nts
        if start_pos % 3 == 1:
            # conditionals of xj, yi
            codons23_x = [''.join([nt_list_tmp[3 * j]] + [bp[0], bp[1]]) for bp in bases_pairs]
            codons23_y = [''.join([bp[0], bp[1]] + [nt_list_tmp[3 * (j + 1)]]) for bp in bases_pairs]
            prob23_x = compute_conditional_for_position(seq_x, j, h_mtx_x, J_mtx_x, lamb=lamb, codons_to_sum=codons23_x)
            prob23_y = compute_conditional_for_position(seq_y, i, h_mtx_y, J_mtx_y, lamb=lamb, codons_to_sum=codons23_y)
        
            # sample sj_2, sj_3 (the 1st, 2nd positions of the yi tri-nt)
            prob23 = np.multiply(prob23_x, prob23_y)
            prob23 = prob23 / np.sum(prob23)
            nt23 = bases_pairs[np.random.choice(np.arange(16), p=prob23)]
            nt_list_tmp[3 * j + 1] = nt23[0]
            nt_list_tmp[3 * j + 2] = nt23[1]
        
            # conditionals of xj+1, yi
            codons1_x = [''.join([b] + nt_list_tmp[3 * (j + 1) + 1 : 3 * (j + 2)]) for b in bases]
            codons1_y = [''.join(nt_list_tmp[3 * j + 1 : 3 * (j + 1)] + [b]) for b in bases]
            prob1_x = compute_conditional_for_position(seq_x, j + 1, h_mtx_x, J_mtx_x, lamb=lamb, codons_to_sum=codons1_x)
            prob1_y = compute_conditional_for_position(seq_y, i, h_mtx_y, J_mtx_y, lamb=lamb, codons_to_sum=codons1_y)
            
            # sample sj+1_1 (the 3rd position of the yi tri-nt)
            prob1 = np.multiply(prob1_x, prob1_y)
            prob1 = prob1 / np.sum(prob1)
            nt1 = np.random.choice(bases, p=prob1)
            nt_list_tmp[3 * (j + 1)] = nt1
            
        # when reading frames overlap with 1 nt
        elif start_pos % 3 == 2:
            # conditionals of xj, yi
            codons3_x = [''.join(nt_list_tmp[3 * j : 3 * j + 2] + [b]) for b in bases]
            codons3_y = [''.join([b] + nt_list_tmp[3 * (j + 1) : 3 * (j + 1) + 2]) for b in bases]
            prob3_x = compute_conditional_for_position(seq_x, j, h_mtx_x, J_mtx_x, lamb=lamb, codons_to_sum=codons3_x)
            prob3_y = compute_conditional_for_position(seq_y, i, h_mtx_y, J_mtx_y, lamb=lamb, codons_to_sum=codons3_y)

            # sample sj_3 (the 1st position of the yi tri-nt)
            prob3 = np.multiply(prob3_x, prob3_y)
            prob3 = prob3 / np.sum(prob3)
            nt3 = np.random.choice(bases, p=prob3)
            nt_list_tmp[3 * j + 2] = nt3
            
            # conditionals of xj+1, yi
            codons12_x = [''.join([bp[0], bp[1]] + [nt_list_tmp[3 * (j + 1) + 2]]) for bp in bases_pairs]
            codons12_y = [''.join([nt_list_tmp[3 * j + 2]] + [bp[0], bp[1]]) for bp in bases_pairs]
            prob12_x = compute_conditional_for_position(seq_x, j + 1, h_mtx_x, J_mtx_x, lamb=lamb, codons_to_sum=codons12_x)
            prob12_y = compute_conditional_for_position(seq_y, i, h_mtx_y, J_mtx_y, lamb=lamb, codons_to_sum=codons12_y)
        
            # sample sj+1_1, sj+1_2 (the 2nd, 3rd position of the yi tri_nt)
            prob12 = np.multiply(prob12_x, prob12_y)
            prob12 = prob12 / np.sum(prob12)
            nt12 = bases_pairs[np.random.choice(np.arange(16), p=prob12)]
            nt_list_tmp[3 * (j + 1)] = nt12[0]
            nt_list_tmp[3 * (j + 1) + 1] = nt12[1]
    
    return ''.join(nt_list_tmp)


# sample nt seq s of length 3|x|, where x is obtained by translating the entire sequence s
# y is obtained by translating the length 3|y| subsequence of s that starts at `start_pos`
# `wt_nt_seq` is the WT *nt* sequence of x
# `start_pos` is the starting position of double encoding in nucleotide index
# note: we sample assuming the reading frames overlap with either 1 or 2 nts
def gibbs_sampler(h_mtx_x, J_mtx_x, h_mtx_y, J_mtx_y, wt_nt_seq, start_pos, lamb=1, num_iter=100000, nt_seq_init=None):
    # note: Lx, Ly are lengths of *amino acid sequences*, to get nucleotide sequence length, multiply by 3
    Lx = h_mtx_x.shape[0]
    Ly = h_mtx_y.shape[0]
    check_start_pos(Lx, Ly, start_pos)
    assert Lx * 3 == len(wt_nt_seq), 'length of wt_nt_seq needs to be {}'.format(Lx * 3)

    if nt_seq_init is None:
        nt_seq = sample_nt_seq(3 * Ly)
        nt_seq = replace_wt_subseq(wt_nt_seq, nt_seq, start_pos)
    else:
        assert len(nt_seq_init) == len(wt_nt_seq)
        nt_seq = nt_seq_init

    nt_seq_samples = [None for i in range(num_iter+1)]
    energy_x_samples = np.zeros(num_iter+1)
    energy_y_samples = np.zeros(num_iter+1)
    
    for i in tqdm(range(num_iter+1)):
        if i > 0:
            nt_seq = conditional_sampler(nt_seq, h_mtx_x, J_mtx_x, h_mtx_y, J_mtx_y, start_pos, lamb=lamb)
        # make sure the non-double-encoding region stays the same as the wt
        assert nt_seq[:start_pos] + nt_seq[start_pos + 3 * Ly:] == wt_nt_seq[:start_pos] + wt_nt_seq[start_pos + 3 * Ly:]

        nt_seq_samples[i] = nt_seq
        seq_x = translate_nt_to_aa(nt_seq)
        seq_y = translate_nt_to_aa(nt_seq[start_pos : start_pos + 3 * Ly])
        energy_x = compute_energy(seq_x, h_mtx_x, J_mtx_x)
        energy_y = compute_energy(seq_y, h_mtx_y, J_mtx_y)
        energy_x_samples[i] = energy_x
        energy_y_samples[i] = energy_y
    
    return nt_seq_samples, energy_x_samples, energy_y_samples





''' optimization functions'''
# ILP MAP calculation for one potts
# boolean=True: ILP
def lp_map(h_mtx, J_mtx, boolean=True):
    L = h_mtx.shape[0]
    q = h_mtx.shape[1]
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
# TODO: this assumes the ORF has overlap 1, for ORF with overlap 2, needs to change how the compat constraints are defined 
def double_encode_lp_map(h_mtx_x, J_mtx_x, h_mtx_y, J_mtx_y, boolean=True):
    assert h_mtx_x.shape == h_mtx_y.shape
    assert J_mtx_x.shape == J_mtx_y.shape
    assert h_mtx_x.shape[1] == 64 # for double encoding, we need to work in the space of codon
    L = h_mtx_x.shape[0]
    q = h_mtx_x.shape[1] 

    mu_ind_x = cp.Variable((L, q), boolean=boolean) # variable for each individual position
    mu_pairs_x = [cp.Variable((q, q), boolean=boolean) for _ in range(int(L*(L-1)/2))] # variables for each pair of positions
    mu_pair_dict_x = dict(zip(itertools.combinations(np.arange(L), 2), mu_pairs_x))
    mu_ind_y = cp.Variable((L, q), boolean=boolean)
    mu_pairs_y = [cp.Variable((q, q), boolean=boolean) for _ in range(int(L*(L-1)/2))]
    mu_pair_dict_y = dict(zip(itertools.combinations(np.arange(L), 2), mu_pairs_y)) 

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
    
    # coupling constraints
    constraints_cp = [
        mu_ind_x[:, idx[0]] + mu_ind_y[:, idx[1]] <= 1
        for idx, compat in codon_compat_2.items()
        if not compat
    ]
    constraints_cp += [
        mu_ind_x[1:, idx[1]] + mu_ind_y[:-1, idx[0]] <= 1
        for idx, compat in codon_compat_1.items()
        if not compat
    ]
    
    constraints = constraints_ind + constraints_pair + constraints_cp
    
    obj_ind_x = cp.sum(cp.multiply(h_mtx_x, mu_ind_x))
    obj_pair_x = cp.sum([cp.sum(cp.multiply(J_mtx_x[idx[0], idx[1]], mu)) for idx, mu in mu_pair_dict_x.items()])
    obj_ind_y = cp.sum(cp.multiply(h_mtx_y, mu_ind_y))
    obj_pair_y = cp.sum([cp.sum(cp.multiply(J_mtx_y[idx[0], idx[1]], mu)) for idx, mu in mu_pair_dict_y.items()])

    obj = cp.Maximize(obj_ind_x + obj_pair_x + obj_ind_y + obj_pair_y)
    prob = cp.Problem(obj, constraints)
    prob.solve()
    
    return mu_ind_x, mu_ind_y, prob




''' unit tests '''
def test_compute_energy():
    seq = 'ACBA'
    L = len(seq)
    q = np.unique([a for a in seq]).size
    aa_dict = {'A': 0, 'B': 1, 'C': 2}
    h_mtx = np.full((L, q), -1)
    J_mtx = np.full((L, L, q, q), -1)
    for i in range(L):
        h_mtx[i, aa_dict[seq[i]]] = 1
    for i in range(L):
        for j in range(i+1, L):
            J_mtx[i, j, aa_dict[seq[i]], aa_dict[seq[j]]] = 1
    energy = compute_energy(seq, h_mtx, J_mtx, aa_dict)
    assert energy == L + L*(L-1)/2


def test_sample_J():
    L = 10
    q = 21
    J_mtx = sample_J(L, q)
    for i in range(L):
        for j in range(L):
            assert np.all(J_mtx[i, j] == J_mtx[i, j].T)
            assert np.all(J_mtx[i, j] == J_mtx[j, i].T)



# Non-stop codon codons beginning with defined first letter.
first_letter_codons = {'T': ['TTT', 'TTC', 'TTA', 'TTG', 'TCT', 'TCC', 
                             'TCA', 'TCG', 'TAT', 'TAC', 'TGT', 'TGC', 'TGG'],
                       'G': ['GTT', 'GTC', 'GTA', 'GTG', 'GCT', 'GCC', 'GCA', 'GCG', 
                             'GAT', 'GAC', 'GAA', 'GAG', 'GGT', 'GGC', 'GGA', 'GGG'],
                       'A': ['ATT', 'ATC', 'ATA', 'ATG', 'ACT', 'ACC', 'ACA', 'ACG', 
                             'AAT', 'AAC', 'AAA', 'AAG', 'AGT', 'AGC', 'AGA', 'AGG'],
                       'C': ['CTT', 'CTC', 'CTA', 'CTG', 'CCT', 'CCC', 'CCA', 'CCG', 
                             'CAT', 'CAC', 'CAA', 'CAG', 'CGT', 'CGC', 'CGA', 'CGG']}

last_letter_codons = {'C': ['TTC', 'TCC', 'TAC', 'TGC', 'CTC', 'CCC', 'CAC', 'CGC', 
                            'ATC', 'ACC', 'AAC', 'AGC', 'GTC', 'GCC', 'GAC', 'GGC'],
                      'T': ['TTT', 'TCT', 'TAT', 'TGT', 'CTT', 'CCT', 'CAT', 'CGT', 
                            'ATT', 'ACT', 'AAT', 'AGT', 'GTT', 'GCT', 'GAT', 'GGT'],
                      'G': ['TTG', 'TCG', 'TGG', 'CTG', 'CCG', 'CAG', 'CGG', 'ATG', 
                            'ACG', 'AAG', 'AGG', 'GTG', 'GCG', 'GAG', 'GGG'],
                      'A': ['TTA', 'TCA', 'CTA', 'CCA', 'CAA', 'CGA', 'ATA', 'ACA', 
                            'AAA', 'AGA', 'GTA', 'GCA', 'GAA', 'GGA']}
