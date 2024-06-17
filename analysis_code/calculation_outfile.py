### Using trimmed outfiles to calculate the mean, standard error on the mean and standard deviation of quantities

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# specify paths to read data from + save plots to
local_or_cplab = input("Local or cplab: ")

if local_or_cplab == "local":
    path_to_trimmed_outfiles = '/home/elenaespinosa/OneDrive/Uni/Summer_courses/Summer_project/mod_lammps_code/outfiles/trimmed_outfiles/'
    save_plots_to = '/home/elenaespinosa/OneDrive/Uni/Summer_courses/Summer_project/mod_lammps_code/plots/'
else:
    path_to_trimmed_outfiles = '/home/s2205640/Documents/summer_project/mod_lammps_code/outfiles/trimmed_outfiles/'
    save_plots_to = '/home/s2205640/Documents/summer_project/mod_lammps_code/plots/'



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


def calc_stats(data_frame):

    mean = data_frame.mean()
    std = data_frame.std()
    sem = std / np.sqrt(data_frame.count())

    return mean, std, sem

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']



# plot 1 - Number of clusters vs Timesteps
plt.figure(figsize=(16, 10))

for i in range(1, 4):
    mean, std, sem = calc_stats(data_frames_123_trimmed[i]['No_of_clusters'])
    print(f"Mean: {mean}, STD: {std} and SEM: {sem}")

    plt.errorbar(data_frames_123_trimmed[i]['Timesteps'], data_frames_123_trimmed[i]['No_of_clusters'], yerr=sem, fmt='', ecolor=colors[i-1],
                 marker='.', alpha=0.7, color=colors[i-1], label=f'Model {i}, Mean: {mean:.2f} ± {sem:.2f}') # should error bars be std or sem??? :/

    plt.axhline(y=mean, linestyle='--', alpha=0.5, color=colors[i-1])

plt.xlabel('Timesteps', fontsize ='16')
plt.xticks(fontsize='14')

plt.ylabel('Number of clusters', fontsize ='16')
plt.yticks(fontsize='14')

plt.title('Number of clusters vs Timesteps', fontsize ='16')
plt.ticklabel_format(style='plain')

plt.legend(fontsize="14", loc ="center right")
plt.grid(True)

#plt.savefig(save_plots_to + "plot_1_model_4_run_1_control.png")
plt.show()


# plot 2 - Mean size of clusters vs Timesteps
plt.figure(figsize=(16, 10))

for i in range(1, 4):
    mean, std, sem = calc_stats(data_frames_123_trimmed[i]['Mean_size_of_clusters'])
    print(f"Mean: {mean}, STD: {std} and SEM: {sem}")

    plt.errorbar(data_frames_123_trimmed[i]['Timesteps'], data_frames_123_trimmed[i]['Mean_size_of_clusters'], yerr=sem, fmt='', ecolor=colors[i-1],
                 marker='.', alpha=0.7, color=colors[i-1], label=f'Model {i}, Mean: {mean:.2f} ± {sem:.2f}') # should error bars be std or sem??? :/

    plt.axhline(y=mean, linestyle='--', alpha=0.5, color=colors[i-1])

plt.xlabel('Timesteps', fontsize ='16')
plt.xticks(fontsize='14')

plt.ylabel('Mean size of clusters', fontsize ='16')
plt.yticks(fontsize='14')

plt.title('Mean size of clusters vs Timesteps', fontsize ='16')
plt.ticklabel_format(style='plain')

plt.legend(fontsize="14", loc ="upper left")
plt.grid(True)

#plt.savefig(save_plots_to + "plot_5_model_4_run_1_control.png")
plt.show()


# plot 3 - Size of largest cluster vs Timesteps
plt.figure(figsize=(16, 10))

for i in range(1, 4):
    mean, std, sem = calc_stats(data_frames_123_trimmed[i]['Size_of_largest_cluster'])
    print(f"Mean: {mean}, STD: {std} and SEM: {sem}")

    plt.errorbar(data_frames_123_trimmed[i]['Timesteps'], data_frames_123_trimmed[i]['Size_of_largest_cluster'], yerr=sem, fmt='', ecolor=colors[i-1],
                 marker='.', alpha=0.7, color=colors[i-1], label=f'Model {i}, Mean: {mean:.2f} ± {sem:.2f}') # should error bars be std or sem??? :/

    plt.axhline(y=mean, linestyle='--', alpha=0.5, color=colors[i-1])

plt.xlabel('Timesteps', fontsize ='16')
plt.xticks(fontsize='14')

plt.ylabel('Size of largest cluster', fontsize ='16')
plt.yticks(fontsize='14')

plt.title('Size of largest cluster vs Timesteps', fontsize ='16')
plt.ticklabel_format(style='plain')

plt.legend(fontsize="14", loc ="center left")
plt.grid(True)

#plt.savefig(save_plots_to + "plot_5_model_4_run_1_control.png")
plt.show()


# plot 4 - Number of clusters of size 1 vs Timesteps
plt.figure(figsize=(16, 10))

for i in range(1, 4):
    mean, std, sem = calc_stats(data_frames_123_trimmed[i]['No_of_clusters_of_size_1'])
    print(f"Mean: {mean}, STD: {std} and SEM: {sem}")

    plt.errorbar(data_frames_123_trimmed[i]['Timesteps'], data_frames_123_trimmed[i]['No_of_clusters_of_size_1'], yerr=sem, fmt='', ecolor=colors[i-1],
                 marker='.', alpha=0.7, color=colors[i-1], label=f'Model {i}, Mean: {mean:.2f} ± {sem:.2f}') # should error bars be std or sem??? :/

    plt.axhline(y=mean, linestyle='--', alpha=0.5, color=colors[i-1])

plt.xlabel('Timesteps', fontsize ='16')
plt.xticks(fontsize='14')

plt.ylabel('Number of clusters of size 1', fontsize ='16')
plt.yticks(fontsize='14')

plt.title('Number of clusters of size 1 vs Timesteps', fontsize ='16')
plt.ticklabel_format(style='plain')

plt.legend(fontsize="14", loc ="center right")
plt.grid(True)

#plt.savefig(save_plots_to + "plot_5_model_4_run_1_control.png")
plt.show()


# plot 5 - Number of proteins bound to polymer vs Timesteps
plt.figure(figsize=(16, 10))

for i in range(1, 4):
    mean, std, sem = calc_stats(data_frames_123_trimmed[i]['No_proteins_bound_to_poly'])
    print(f"Mean: {mean}, STD: {std} and SEM: {sem}")

    plt.errorbar(data_frames_123_trimmed[i]['Timesteps'], data_frames_123_trimmed[i]['No_proteins_bound_to_poly'], yerr=sem, fmt='', ecolor=colors[i-1],
                 marker='.', alpha=0.7, color=colors[i-1], label=f'Model {i}, Mean: {mean:.2f} ± {sem:.2f}') # should error bars be std or sem??? :/

    plt.axhline(y=mean, linestyle='--', alpha=0.5, color=colors[i-1])

plt.xlabel('Timesteps', fontsize ='16')
plt.xticks(fontsize='14')

plt.ylabel('Number of proteins bound to polymer', fontsize ='16')
plt.yticks(fontsize='14')

plt.title('Number of proteins bound to polymer vs Timesteps', fontsize ='16')
plt.ticklabel_format(style='plain')

plt.legend(fontsize="14", loc ="center right")
plt.grid(True)

#plt.savefig(save_plots_to + "plot_5_model_4_run_1_control.png")
plt.show()
