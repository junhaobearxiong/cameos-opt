import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# load results
Lx = 4
Ly = 2
lamb_list = [0, 1, 1.5, 2, 3]
num_chains = 100
num_mcmc_steps = 5000

with open('results/gibbs_Lx{}_Ly{}_nchains{}_nmcsteps{}.pkl'.format(Lx, Ly, num_chains, num_mcmc_steps), 'rb') as f:
    all_results = pickle.load(f)
map_results = all_results.pop('map')

# process results into dataframes
df_list = []
for lamb in lamb_list:
    for i in range(num_chains):
        df = pd.DataFrame.from_dict(all_results[lamb][i])
        df['lamb'] = lamb
        df['chain_idx'] = i
        df['energy'] = df['energy_x'] + df['energy_y']
        df_list.append(df)

all_df_orig = pd.concat(df_list).reset_index()
all_df_orig.rename(columns={'index': 'tsp'}, inplace=True)

# take the last `samples_pct` samples, set to 1 if want all the samples 
samples_pct = 0.95
all_df = []
for df in df_list:
    all_df.append(df.iloc[-int(samples_pct * num_mcmc_steps):])
all_df = pd.concat(all_df).reset_index(drop=True)

# count the number of unique samples for each chain
unique_samples_df = pd.DataFrame({'num_unique_samples': all_df.groupby(['lamb', 'chain_idx']).apply(lambda x: len(x['nt_seq'].unique()))}).reset_index()
unique_samples_df['log_num_unique_samples'] = np.log10(unique_samples_df['num_unique_samples'].values)


# boxplot
sns.set_theme(style='whitegrid', font_scale=2)
fig, axs = plt.subplots(1, 2, figsize=(24, 8))

sns.boxplot(x='lamb', y='energy', data=all_df, ax=axs[0])
axs[0].axhline(y=map_results['energy_x'] + map_results['energy_y'], color='r', label='MAP solution')
axs[0].set_xlabel('Inverse temperature')
axs[0].set_ylabel('Score')
axs[0].legend()

sns.boxplot(x='lamb', y='log_num_unique_samples', data=unique_samples_df, ax=axs[1])
axs[1].set_xlabel('Inverse temperature')
axs[1].set_ylabel('Number of unique samples (log10)')

plt.savefig('figures/simulation_gibbs_boxplot_Lx{}_Ly{}_nchains{}_nmcsteps{}.png'.format(Lx, Ly, num_chains, num_mcmc_steps))


# traceplot
df_agg_chains = all_df_orig.groupby(['lamb', 'start_pos', 'tsp']).energy.agg(['mean', 'std']).reset_index()

fig, axs = plt.subplots(5, 2, figsize=(20, 20), sharex=True, sharey='col')

for i, lamb in enumerate(lamb_list):
    df = df_agg_chains.query('lamb == {}'.format(lamb))
    if i == 0:
        legend = 'full'
    else:
        legend = False
    sns.lineplot(ax=axs[i, 0], data=df, x='tsp', y='mean', hue='start_pos', legend=legend, palette='Set2')
    axs[i, 0].set_ylabel('Score (mean)')
    axs[i, 0].set_xlabel('MCMC step')
    axs[i, 0].set_title('Inverse temperature = {}'.format(lamb))
    axs[i, 0].axhline(y=map_results['energy_x'] + map_results['energy_y'], color='y', label='MAP solution')

    sns.lineplot(ax=axs[i, 1], data=df, x='tsp', y='std', hue='start_pos', legend=False, palette='Set2')
    axs[i, 1].set_ylabel('Score (1 std)')
    axs[i, 1].set_xlabel('MCMC step')
    axs[i, 1].set_title('Inverse temperature = {}'.format(lamb))
    axs[i, 1].axhline(y=map_results['energy_x'] + map_results['energy_y'], color='y', label='MAP solution')
    
axs[0, 0].legend(title='Starting position')
plt.tight_layout()

plt.savefig('figures/simulation_gibbs_traceplot_Lx{}_Ly{}_nchains{}_nmcsteps{}.png'.format(Lx, Ly, num_chains, num_mcmc_steps))