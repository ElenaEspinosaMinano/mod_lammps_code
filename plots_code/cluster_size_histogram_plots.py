### Using trimmed cluster_size outfiles to calculate histogram plots

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter

# specify paths to read data from + save plots to
local_or_cplab = input("Local or cplab: ")

if local_or_cplab == "local":
    path_to_trimmed_outfiles = '/home/elenaespinosa/OneDrive/Uni/Summer_courses/Summer_project/mod_lammps_code/outfiles/trimmed_outfiles/'
    save_plots_to = '/home/elenaespinosa/OneDrive/Uni/Summer_courses/Summer_project/mod_lammps_code/plots/'
else:
    path_to_trimmed_outfiles = '/home/s2205640/Documents/summer_project/mod_lammps_code/outfiles/trimmed_outfiles/'
    save_plots_to = '/home/s2205640/Documents/summer_project/mod_lammps_code/plots/'


def calc_stats(data_frame):

    mean = data_frame.mean()
    std = data_frame.std() # sample standard deviation, normalized by 1 / 1(N-1)
    sem = std / np.sqrt(data_frame.count()) # data_frame.count() = N = 201

    return mean, std, sem


###
#   First investigation - Model 1, 2 + 3 plots
###

trimmed_outfiles_list_123 = [f'trimmed_outfile_{i}_run_1.dat' for i in range(1, 4)]

# dictionary to store data frames
data_frames_123_trimmed = {}

# parse data
for i in range(1, 4):
    data_frames_123_trimmed[i] = pd.read_csv(path_to_trimmed_outfiles + trimmed_outfiles_list_123[i-1], sep=' ', comment='#', header=None)
    data_frames_123_trimmed[i].columns = ['Timesteps', 'No_of_clusters', 'Mean_size_of_clusters', 
                                        'Size_of_largest_cluster', 'No_of_clusters_of_size_1', 'No_proteins_bound_to_poly']



# get stats of models 123 for a particular dataframe and column + append it to a list
def get_stats_123(data_frames, column):
    mean_list = []
    std_list = []
    sem_list = []
    for i in range(1, 4):
        mean, std, sem = calc_stats(data_frames[i][column])
        mean_list.append(mean)
        std_list.append(std)
        sem_list.append(sem)
    return mean_list, std_list, sem_list


prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']



# plot 1 - Number of clusters vs Timesteps - Mean ± SEM and STD
mean_123_1, std_123_1, sem_123_1 = get_stats_123(data_frames_123_trimmed, 'No_of_clusters')

models = ('Model 1', 'Model 2', 'Model 3')
x_pos = np.arange(len(models))

bar_labels_mean = [f'{model}: {mean_123_1[i]:.2f} ± {sem_123_1[i]:.2f} (1 SEM)' for i, model in enumerate(models)]
bar_labels_std = [f'{model}: {std_123_1[i]:.2f} ± ---' for i, model in enumerate(models)]

fig, axs = plt.subplots(1, 2, sharey=True, figsize=(16, 10), tight_layout=True)

left_bar_1 = axs[0].bar(models, mean_123_1, yerr=sem_123_1, label=bar_labels_mean, color=colors[:4])

axs[0].set_xticks(x_pos)
axs[0].tick_params(axis='x', labelsize=14)
axs[0].tick_params(axis='y', labelsize=14)
#axs[0].set_xlabel('Models', fontsize ='16')
axs[0].set_ylabel('Number of clusters', fontsize ='16')
axs[0].set_title('Number of clusters for different models - Mean ± 1 SEM', fontsize ='16')
axs[0].legend(fontsize=14)


right_bar_1 = axs[1].bar(models, std_123_1, label=bar_labels_std, color=colors[:4])

axs[1].set_xticks(x_pos)
axs[1].tick_params(axis='x', labelsize=14)
axs[1].tick_params(axis='y', labelsize=14)
#axs[1].set_xlabel('Models', fontsize ='16')
axs[1].set_ylabel('Number of clusters', fontsize ='16')
axs[1].set_title('Number of clusters for different models - STD', fontsize ='16')
axs[1].legend(fontsize=14)

plt.savefig(save_plots_to + "plot_1_model_123_SS_run_1.png", dpi='figure')
plt.show()

