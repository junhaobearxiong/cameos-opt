import numpy as np
import matplotlib.pyplot as plt


def plot_trace_plot(axs, gibbs_results):
    for lamb, results in gibbs_results.items():
        if lamb != 0:
            axs.plot(np.arange(results['energy_x'].size), results['energy_x'] + results['energy_y'], label=lamb, alpha=0.5)
    axs.legend()
    return axs


def plot_energy_hist(axs, gibbs_results, map_energy, start_pos_prior, wt_nt_seq):
    for lamb, results in gibbs_results.items():
        _, idx_unique = np.unique(results['nt_seq'], return_index=True)
        axs.hist(results['energy_x'][idx_unique] + results['energy_y'][idx_unique], 
                 bins=20, histtype='step', density=True,
                 label='lamb={}, N unique={}'.format(lamb, idx_unique.size))

    axs.axvline(x=map_energy, color='m', label='map')
    axs.legend()
    axs.set_xlabel('Energy')
    axs.set_ylabel('Empirical frequency')
    axs.set_title('Distribution of Energy of Gibbs Samples (Starting Position: {}, WT: {})'.format(start_pos_prior, wt_nt_seq))
    return axs