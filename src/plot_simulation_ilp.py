import numpy as np
import pickle
import matplotlib.pyplot as plt


Lx = 4
Ly = 2
num_iters = 10
with open('results/ilp_Lx{}_Ly{}_niters{}.pkl'.format(Lx, Ly, num_iters), 'rb') as f:
    all_results = pickle.load(f)


fig, axs = plt.subplots(1, 2, figsize=(12, 4))

energy_x_map = np.zeros(len(all_results))
energy_x_ilp = np.zeros(len(all_results))
energy_y_map = np.zeros(len(all_results))
energy_y_ilp = np.zeros(len(all_results))

for i, result in enumerate(all_results):
    energy_x_map[i] = result[0]['energy_x']
    energy_x_ilp[i] = result[1]['energy_x']
    energy_y_map[i] = result[0]['energy_y']
    energy_y_ilp[i] = result[1]['energy_y']

line = np.linspace(energy_x_map.min(), energy_x_map.max(), 1000)
axs[0].scatter(energy_x_map, energy_x_ilp)
axs[0].plot(line, line, color='r', label='y=x')
axs[0].set_xlabel('Brute force solution')
axs[0].set_ylabel('ILP solution')
axs[0].set_title('Energy of sequence x')
axs[0].legend()

line = np.linspace(energy_y_map.min(), energy_y_map.max(), 1000)
axs[1].scatter(energy_y_map, energy_y_ilp)
axs[1].plot(line, line, color='r', label='y=x')
axs[1].set_xlabel('Brute force solution')
axs[1].set_ylabel('ILP solution')
axs[1].set_title('Energy of sequence y')

plt.savefig('figures/simulation_ilp_Lx{}_Ly{}_niters{}.png'.format(Lx, Ly, num_iters))