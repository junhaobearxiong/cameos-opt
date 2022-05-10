import numpy as np
from tqdm import tqdm
from scipy.special import logsumexp
from utils import *



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
    return np.exp(energy_arr - logsumexp(energy_arr))


# conditional sampler for 1 iteration of gibbs sampling for double encoding
# samples every nt of a sequence that's within the double encoding region
# `nt_seq`: a string of nucleotides 
def conditional_sampler(nt_seq, h_mtx_x, J_mtx_x, h_mtx_y, J_mtx_y, start_pos, lamb=1):
    Lx = int(h_mtx_x.size / 21)
    Ly = int(h_mtx_y.size / 21)
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


# one mcmc chain for a given starting position, wt sequence, and fixed temperature
# sample nt seq s of length 3|x|, where x is obtained by translating the entire sequence s
# y is obtained by translating the length 3|y| subsequence of s that starts at `start_pos`
# `wt_nt_seq` is the WT *nt* sequence of x
# note: we sample assuming the reading frames overlap with either 1 or 2 nts
def gibbs_sampler(h_mtx_x, J_mtx_x, h_mtx_y, J_mtx_y, wt_nt_seq, start_pos, lamb=1, num_iter=100000, nt_seq_init=None):
    # note: Lx, Ly are lengths of *amino acid sequences*, to get nucleotide sequence length, multiply by 3
    Lx = int(h_mtx_x.size / 21)
    Ly = int(h_mtx_y.size / 21)
    assert Lx * 3 == len(wt_nt_seq), 'length of wt_nt_seq needs to be {}'.format(Lx * 3)

    # initialize nt sequence
    if nt_seq_init is None:
        nt_seq = sample_nt_seq(3 * Ly)
        nt_seq = replace_wt_subseq(wt_nt_seq, nt_seq, start_pos)
    else:
        assert len(nt_seq_init) == len(wt_nt_seq)
        nt_seq = nt_seq_init

    nt_seq_samples = [None for i in range(num_iter + 1)]
    energy_x_samples = np.zeros(num_iter + 1)
    energy_y_samples = np.zeros(num_iter + 1)
    
    for i in tqdm(range(num_iter + 1)):
        if i > 0:
            nt_seq = conditional_sampler(nt_seq, h_mtx_x, J_mtx_x, h_mtx_y, J_mtx_y, start_pos, lamb=lamb)
        # assert nt_seq[:start_pos] + nt_seq[start_pos + 3 * Ly:] == wt_nt_seq[:start_pos] + wt_nt_seq[start_pos + 3 * Ly:]

        nt_seq_samples[i] = nt_seq
        seq_x = translate_nt_to_aa(nt_seq)
        seq_y = translate_nt_to_aa(nt_seq[start_pos : start_pos + 3 * Ly])
        energy_x = compute_energy(seq_x, h_mtx_x, J_mtx_x)
        energy_y = compute_energy(seq_y, h_mtx_y, J_mtx_y)
        energy_x_samples[i] = energy_x
        energy_y_samples[i] = energy_y

    results = {
        'nt_seq': nt_seq_samples,
        'energy_x': energy_x_samples,
        'energy_y': energy_y_samples,
        'start_pos': start_pos,
        'wt_nt_seq': wt_nt_seq
    }    
    return results


# `start_pos_prior` is a distribution over the starting position of double encoding in nucleotide index
#   - `None`: sample starting positions uniformly
#   - a number: fix the starting position
#   - an array: sample starting positions with prob = `start_pos_prior` 
# note: if start_pos is sampled, it would still be fixed for all the gibbs samples
# for each lambda, we run `num_chains` mcmc chains, each for `num_mcmc_steps`
# for each chain, we sample a start_pos based on `start_pos_prior` 
def run_gibbs_for_lamb(lamb_list, h_mtx_x, J_mtx_x, h_mtx_y, J_mtx_y, wt_nt_seq, start_pos_prior, num_chains, num_mcmc_steps):
    # uniformly sample from sequence space, conditioning on wt seq and start pos
    def get_unif_samples(Lx, Ly, wt_nt_seq, start_pos):
        samples = [replace_wt_subseq(wt_nt_seq, sample_nt_seq(3 * Ly), start_pos) for _ in range(num_mcmc_steps)]
        energies_x = np.array([compute_energy(translate_nt_to_aa(s), h_mtx_x, J_mtx_x) for s in samples])
        energies_y = np.array([compute_energy(translate_nt_to_aa(s[start_pos : start_pos + 3 * Ly]), h_mtx_y, J_mtx_y) for s in samples])
        results = {
            'nt_seq': samples,
            'energy_x': energies_x,
            'energy_y': energies_y,
            'start_pos': start_pos,
            'wt_nt_seq': wt_nt_seq
        }
        return results
    
    Lx = int(h_mtx_x.size / 21)
    Ly = int(h_mtx_y.size / 21)
    # sample starting positions
    if start_pos_prior is None:
        # sample uniformly
        start_pos_samples = sample_start_pos(Lx, Ly, num_chains)
    else:
        try:
            # fix starting position
            start_pos = int(start_pos_prior)
            check_start_pos(Lx, Ly, start_pos)
            start_pos_samples = np.full(num_chains, start_pos)
        except:
            # sample based on distribution given by `start_pos_prior`
            start_pos_samples = sample_start_pos(Lx, Ly, num_chains, start_pos_prior)

    all_results = {}
    Lx = int(h_mtx_x.size / 21)
    Ly = int(h_mtx_y.size / 21)
    for lamb in lamb_list:
        gibbs_results = [None] * num_chains
        for i in range(num_chains):
            print('lamb: {}, chain: {}'.format(lamb, i))
            start_pos = start_pos_samples[i]
            if lamb == 0:
                results = get_unif_samples(Lx, Ly, wt_nt_seq, start_pos)
            else:
                results = gibbs_sampler(h_mtx_x, J_mtx_x, h_mtx_y, J_mtx_y,
                                        wt_nt_seq=wt_nt_seq, start_pos=start_pos, lamb=lamb, num_iter=num_mcmc_steps)
            gibbs_results[i] = results
        all_results[lamb] = gibbs_results
    return all_results
