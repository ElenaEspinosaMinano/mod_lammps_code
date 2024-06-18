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
    std = data_frame.std() # sample standard deviation, normalized by 1 / 1(N-1)
    sem = std / np.sqrt(data_frame.count()) # data_frame.count() = N = 201

    return mean, std, sem

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

"""

# plot 1 - Number of clusters vs Timesteps
plt.figure(figsize=(16, 10))

for i in range(1, 4):
    mean, std, sem = calc_stats(data_frames_123_trimmed[i]['No_of_clusters'])
    print(f"Model {i} - Mean: {mean}, STD: {std} and SEM: {sem}")

    plt.errorbar(data_frames_123_trimmed[i]['Timesteps'], data_frames_123_trimmed[i]['No_of_clusters'], yerr=2*sem, fmt='', ecolor=colors[i-1],
                 marker='.', alpha=0.7, color=colors[i-1], label=f'Model {i} - Mean: {mean:.2f} ± {2*sem:.2f} (2 SEM)')

    plt.axhline(y=mean, linestyle='--', alpha=0.5, color=colors[i-1])

plt.xlabel('Timesteps', fontsize ='16')
plt.xticks(fontsize='14')

plt.ylabel('Number of clusters', fontsize ='16')
plt.yticks(fontsize='14')

plt.title('Number of clusters vs Timesteps - steady state', fontsize ='16')
plt.ticklabel_format(style='plain')

plt.legend(fontsize="14", loc ="center right")
plt.grid(True)

plt.savefig(save_plots_to + "plot_1_model_123_SS_run_1.png")
plt.show()


# plot 2 - Mean size of clusters vs Timesteps
plt.figure(figsize=(16, 10))

for i in range(1, 4):
    mean, std, sem = calc_stats(data_frames_123_trimmed[i]['Mean_size_of_clusters'])
    print(f"Model {i} - Mean: {mean}, STD: {std} and SEM: {sem}")

    plt.errorbar(data_frames_123_trimmed[i]['Timesteps'], data_frames_123_trimmed[i]['Mean_size_of_clusters'], yerr=2*sem, fmt='', ecolor=colors[i-1],
                 marker='.', alpha=0.7, color=colors[i-1], label=f'Model {i} - Mean: {mean:.2f} ± {2*sem:.2f} (2 SEM)')

    plt.axhline(y=mean, linestyle='--', alpha=0.5, color=colors[i-1])

plt.xlabel('Timesteps', fontsize ='16')
plt.xticks(fontsize='14')

plt.ylabel('Mean size of clusters', fontsize ='16')
plt.yticks(fontsize='14')

plt.title('Mean size of clusters vs Timesteps - steady state', fontsize ='16')
plt.ticklabel_format(style='plain')

plt.legend(fontsize="14", loc ="upper left")
plt.grid(True)

plt.savefig(save_plots_to + "plot_2_model_123_SS_run_1.png")
plt.show()


# plot 3 - Size of largest cluster vs Timesteps
plt.figure(figsize=(16, 10))

for i in range(1, 4):
    mean, std, sem = calc_stats(data_frames_123_trimmed[i]['Size_of_largest_cluster'])
    print(f"Model {i} - Mean: {mean}, STD: {std} and SEM: {sem}")

    plt.errorbar(data_frames_123_trimmed[i]['Timesteps'], data_frames_123_trimmed[i]['Size_of_largest_cluster'], yerr=2*sem, fmt='', ecolor=colors[i-1],
                 marker='.', alpha=0.7, color=colors[i-1], label=f'Model {i} - Mean: {mean:.2f} ± {2*sem:.2f} (2 SEM)')

    plt.axhline(y=mean, linestyle='--', alpha=0.5, color=colors[i-1])

plt.xlabel('Timesteps', fontsize ='16')
plt.xticks(fontsize='14')

plt.ylabel('Size of largest cluster', fontsize ='16')
plt.yticks(fontsize='14')

plt.title('Size of largest cluster vs Timesteps - steady state', fontsize ='16')
plt.ticklabel_format(style='plain')

plt.legend(fontsize="14", loc ="center left")
plt.grid(True)

plt.savefig(save_plots_to + "plot_3_model_123_SS_run_1.png")
plt.show()


# plot 4 - Number of clusters of size 1 vs Timesteps
plt.figure(figsize=(16, 10))

for i in range(1, 4):
    mean, std, sem = calc_stats(data_frames_123_trimmed[i]['No_of_clusters_of_size_1'])
    print(f"Model {i} - Mean: {mean}, STD: {std} and SEM: {sem}")

    plt.errorbar(data_frames_123_trimmed[i]['Timesteps'], data_frames_123_trimmed[i]['No_of_clusters_of_size_1'], yerr=2*sem, fmt='', ecolor=colors[i-1],
                 marker='.', alpha=0.7, color=colors[i-1], label=f'Model {i} - Mean: {mean:.2f} ± {2*sem:.2f} (2 SEM)')

    plt.axhline(y=mean, linestyle='--', alpha=0.5, color=colors[i-1])

plt.xlabel('Timesteps', fontsize ='16')
plt.xticks(fontsize='14')

plt.ylabel('Number of clusters of size 1', fontsize ='16')
plt.yticks(fontsize='14')

plt.title('Number of clusters of size 1 vs Timesteps - steady state', fontsize ='16')
plt.ticklabel_format(style='plain')

plt.legend(fontsize="14", loc ="center right")
plt.grid(True)

plt.savefig(save_plots_to + "plot_4_model_123_SS_run_1.png")
plt.show()


# plot 5 - Number of proteins bound to polymer vs Timesteps
plt.figure(figsize=(16, 10))

for i in range(1, 4):
    mean, std, sem = calc_stats(data_frames_123_trimmed[i]['No_proteins_bound_to_poly'])
    print(f"Model {i} - Mean: {mean}, STD: {std} and SEM: {sem}")

    plt.errorbar(data_frames_123_trimmed[i]['Timesteps'], data_frames_123_trimmed[i]['No_proteins_bound_to_poly'], yerr=2*sem, fmt='', ecolor=colors[i-1],
                 marker='.', alpha=0.7, color=colors[i-1], label=f'Model {i} - Mean: {mean:.2f} ± {2*sem:.2f} (2 SEM)')

    plt.axhline(y=mean, linestyle='--', alpha=0.5, color=colors[i-1])

plt.xlabel('Timesteps', fontsize ='16')
plt.xticks(fontsize='14')

plt.ylabel('Number of proteins bound to polymer', fontsize ='16')
plt.yticks(fontsize='14')

plt.title('Number of proteins bound to polymer vs Timesteps - steady state', fontsize ='16')
plt.ticklabel_format(style='plain')

plt.legend(fontsize="14", loc ="center right")
plt.grid(True)

plt.savefig(save_plots_to + "plot_5_model_123_SS_run_1.png")
plt.show()

"""

###
#   Second investigation - Model 4 plots (normal + control)
###

trimmed_outfiles_list_4 = [f'trimmed_outfile_4_var_{i}_run_1.dat' for i in range(1, 9)]
trimmed_outfiles_list_4_control = [f'trimmed_outfile_4_var_{i}_run_1_control.dat' for i in range(1, 9)]

# dictionary to store data frames
data_frames_4_trimmed = {}
data_frames_4_trimmed_control = {}

# parse data
for i in range(1, 9):
    data_frames_4_trimmed[i] = pd.read_csv(path_to_trimmed_outfiles + trimmed_outfiles_list_4[i-1], sep=' ', comment='#', header=None)
    data_frames_4_trimmed[i].columns = ['Timesteps', 'No_of_clusters', 'Mean_size_of_clusters', 
                                        'Size_of_largest_cluster', 'No_of_clusters_of_size_1', 'No_proteins_bound_to_poly']
    data_frames_4_trimmed_control[i] = pd.read_csv(path_to_trimmed_outfiles + trimmed_outfiles_list_4_control[i-1], sep=' ', comment='#', header=None)
    data_frames_4_trimmed_control[i].columns = ['Timesteps', 'No_of_clusters', 'Mean_size_of_clusters', 
                                        'Size_of_largest_cluster', 'No_of_clusters_of_size_1', 'No_proteins_bound_to_poly']


# plot 1 - 
mean_1 = []
std_1 = []
sem_1 = []

mean_1_control = []
std_1_control = []
sem_1_control = []

for i in range(1, 9):
    mean, std, sem = calc_stats(data_frames_4_trimmed[i]['No_of_clusters'])
    mean_1.append(mean) ; std_1.append(std) ; sem_1.append(2*sem)

    mean2, std2, sem2 = calc_stats(data_frames_4_trimmed_control[i]['No_of_clusters'])
    mean_1_control.append(mean2) ; std_1_control.append(std2) ; sem_1_control.append(2*sem2)


pp_attraction = ('1', '2', '3', '4', '5', '6', '7', '8')
x_pos = np.arange(len(pp_attraction))

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

left_bar = axs[0].bar(pp_attraction, mean_1, yerr=sem_1, color=colors[:9])
axs[0].set_xticks(x_pos)
axs[0].bar_label(left_bar, labels=[f'{m:.2f}±{std_1[i]:.2f}' for i, m in enumerate(mean_1)], color=colors[:9], fontsize=14)

axs[0].set_xlabel('Protein-protein attraction strength (kBT)')
axs[0].set_ylabel('Number of clusters')
axs[0].set_title('Number of clusters vs. protein-protein attraction strength')


axs[1].bar(pp_attraction, mean_1_control, yerr=sem_1_control, color=colors[:9])
axs[1].set_xticks(x_pos)

axs[1].set_xlabel('Protein-protein attraction strength (kBT)')
axs[1].set_ylabel('Number of clusters')
axs[1].set_title('Number of clusters vs. protein-protein attraction strength - Control')

plt.show()