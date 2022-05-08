import numpy as np
import itertools
from tqdm import tqdm
from scipy.special import logsumexp



''' global variables and dictionaries'''
# global variables
bases = [l.upper() for l in 'agct']
bases_pairs = list(itertools.product(bases, bases))
codons = [a+b+c for a in bases for b in bases for c in bases]
# the order of amino acids here is to match `codons`
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
# the order of amino acids and indices here is used to match `convert_protein` in bio_seq.jl of cameos
unique_amino_acids = list('ARNDCQEGHILKMFPSTWYV*')
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
def get_codon_idx_compat(overlap):
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
codon_compat_2 = get_codon_idx_compat(2) # keys are indices of codon, values are if the 2 codons are compat
codon_compat_1 = get_codon_idx_compat(1)




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


# for length L amino acid seq, get a 21 * L one-hot encoding of seq
# where the amino acids are indexed in the same order as `aa_to_idx`
def convert_seq_one_hot(seq):
    one_hot_seq = np.zeros(21 * len(seq))
    for i, aa in enumerate(seq):
        one_hot_seq[i * 21 + aa_to_idx[aa]] = 1
    return one_hot_seq


# compute energy given an amino acid sequence, represented as a string
# note: to match cameos implementation (mrf.jl: basic_energy_calc), a pair of position is counted twice in the energy calculation 
def compute_energy(seq, h_mtx, J_mtx):
    assert h_mtx.size == len(seq) * 21
    assert J_mtx.shape[0] == len(seq) * 21
    assert J_mtx.shape[1] == len(seq) * 21
    assert len(J_mtx.shape) == 2

    seq_one_hot = convert_seq_one_hot(seq)
    return seq_one_hot @ h_mtx + seq_one_hot @ J_mtx @ seq_one_hot

    '''
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
    '''



''' simulation utils functions '''
# implement sampling of h, J
# h is a L * q vector
def sample_h(L, q):
    h_mtx = np.random.normal(0, 1, (L, q))
    h_mtx[:, -1] = 0
    return h_mtx.reshape(L * q)


# J is L*q by L*q matrix
def sample_J(L, q):
    J_mtx = np.random.normal(0, 1, (L * q, L * q))
    # enforces Jij = Jji.T
    J_mtx_sym = np.zeros((L * q, L * q))
    for i in range(L):
        for j in range(L):
            if i != j:
                Jij = J_mtx[i * q : (i + 1) * q, j * q : (j + 1) * q] # 21-by-21
                Jji = J_mtx[j * q : (j + 1) * q, i * q : (i + 1) * q] # 21-by-21
                J_mtx_sym[i * q : (i + 1) * q, j * q : (j + 1) * q] = 0.5 * (Jij + Jji)
                J_mtx_sym[j * q : (j + 1) * q, i * q : (i + 1) * q] = 0.5 * (Jij + Jji).T
    return J_mtx_sym


def sample_nt_seq(L):
    return ''.join(np.random.choice(bases, L))


def sample_codon_arr(L):
    return np.random.choice(codons, L)


def sample_aa_seq(L):
    return translate_nt_to_aa(sample_nt_seq(3 * L)) 


# sample over the possible double encoding starting positions given Lx, Ly
# `prob`: probability distribution over the *allowed* starting positions, sample uniformly if None 
def sample_start_pos(Lx, Ly, num_samples, prob=None):
    all_start_pos = [i for i in range(3 * (Lx - Ly)) if i % 3 != 0]
    if prob is not None:
        prob = np.array(prob)
        assert prob.size == len(all_start_pos), 'distribution of start_pos needs to be over the {} possible positions (see `check_start_pos`)'.format(2 * (Lx - Ly))
        assert np.sum(prob) == 1
    return np.random.choice(all_start_pos, size=num_samples, p=prob)


# brute force find the MAP, for checking correctness
def brute_force_map(h_mtx, J_mtx):
    q = 21
    L = int(h_mtx.size / q)
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
# if `wt_nt_seq` is given, the map is conditioning on the wt sequence outside the double encoding region
def brute_force_map_double_encoding(h_mtx_x, J_mtx_x, h_mtx_y, J_mtx_y, start_pos=None, wt_nt_seq=None):
    # print('brute force map given start_pos={}, wt_nt_seq={}'.format(start_pos, wt_nt_seq))

    Lx = int(h_mtx_x.size / 21)
    Ly = int(h_mtx_y.size / 21)

    if start_pos is None:
        all_start_pos = [i for i in range(3 * (Lx - Ly)) if i % 3 != 0]
    else:
        check_start_pos(Lx, Ly, start_pos)
        all_start_pos = [start_pos]

    map_result = {
        'energy_x': 0,
        'energy_y': 0,
        'x': None,
        'y': None,
        'nt_seq': None,
        'start_pos': None
    }

    if wt_nt_seq is None:
        all_nts = list(itertools.product(bases, repeat=Lx * 3))
    else:
        assert Lx * 3 == len(wt_nt_seq), 'length of wt_nt_seq needs to be {}'.format(Lx * 3)
        all_nts = list(itertools.product(bases, repeat=Ly * 3))

    for start_pos in all_start_pos:
        for nts in all_nts:
            if wt_nt_seq is None:
                nt_seq = ''.join(nts)
            else:
                nt_seq = ''.join(wt_nt_seq[:start_pos] + ''.join(nts) + wt_nt_seq[start_pos + 3 * Ly:])

            aa_seq_x = translate_nt_to_aa(nt_seq)
            aa_seq_y = translate_nt_to_aa(nt_seq[start_pos : start_pos + 3 * Ly])
            energy_x = compute_energy(aa_seq_x, h_mtx_x, J_mtx_x)
            energy_y = compute_energy(aa_seq_y, h_mtx_y, J_mtx_y)
            energy = energy_x + energy_y
            if energy > map_result['energy_x'] + map_result['energy_y']:
                map_result['x'] = aa_seq_x
                map_result['y'] = aa_seq_y
                map_result['energy_x'] = energy_x
                map_result['energy_y'] = energy_y
                map_result['nt_seq'] = nt_seq
                map_result['start_pos'] = start_pos 

    return map_result


# change the subsequence of `wt_seq` starting at `start_pos` (including) of length `len(subseq)` to be `subseq`
def replace_wt_subseq(wt_seq, subseq, start_pos):
    assert len(wt_seq) >= len(subseq)
    assert start_pos + len(subseq) <= len(wt_seq)
    return wt_seq[:start_pos] + subseq + wt_seq[start_pos + len(subseq):]


# check if the starting position is valid
def check_start_pos(Lx, Ly, start_pos):
    # x is assumed to be strictly longer than y
    assert Lx > Ly, 'x needs to be longer than y'
    # assume nt seq of y can be fully contained in nt seq of x
    assert start_pos > 0 and start_pos < 3 * (Lx - Ly), 'start_pos needs to be > 0 and < {}'.format(3 * (Lx - Ly))
    # if the reading frames for x and y overlap exactly, then the amino acids they encode will be exactly the same
    assert start_pos % 3 != 0, 'reading frames should not overlap exactly'
