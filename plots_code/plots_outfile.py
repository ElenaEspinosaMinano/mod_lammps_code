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


# get states for a particular dataframe and column
def get_stats(data_frames, column):
    mean_list = []
    std_list = []
    sem_list = []
    for i in range(1, 9):
        mean, std, sem = calc_stats(data_frames[i][column])
        mean_list.append(mean)
        std_list.append(std)
        sem_list.append(2 * sem)
    return mean_list, std_list, sem_list



# plot 1 - Number of clusters vs. protein-protein attraction strength
mean_1, std_1, sem_1 = get_stats(data_frames_4_trimmed, 'No_of_clusters')
mean_1_control, std_1_control, sem_1_control = get_stats(data_frames_4_trimmed_control, 'No_of_clusters')

pp_attraction = ('1', '2', '3', '4', '5', '6', '7', '8')
x_pos = np.arange(len(pp_attraction))

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

left_bar_1 = axs[0].bar(pp_attraction, mean_1, yerr=sem_1, color=colors[:9])
axs[0].bar_label(left_bar_1, labels=[f'{m:.2f}±{std_1[i]:.2f}' for i, m in enumerate(mean_1)], fontsize=10)

axs[0].set_xticks(x_pos)
axs[0].tick_params(axis='x', labelsize=14)
axs[0].tick_params(axis='y', labelsize=14)

axs[0].set_xlabel('Protein-protein attraction strength (kBT)', fontsize ='16')
axs[0].set_ylabel('Number of clusters', fontsize ='16')
axs[0].set_title('Number of clusters vs. protein-protein attraction strength', fontsize ='16')


right_bar_1 = axs[1].bar(pp_attraction, mean_1_control, yerr=sem_1_control, color=colors[:9])
axs[1].bar_label(right_bar_1, labels=[f'{m:.2f}±{std_1_control[i]:.2f}' for i, m in enumerate(mean_1_control)], fontsize=10)

axs[1].set_xticks(x_pos)
axs[1].tick_params(axis='x', labelsize=14)
axs[1].tick_params(axis='y', labelsize=14)

axs[1].set_xlabel('Protein-protein attraction strength (kBT)', fontsize ='16')
axs[1].set_ylabel('Number of clusters', fontsize ='16')
axs[1].set_title('Number of clusters vs. protein-protein attraction strength - Control', fontsize ='16')

plt.show()


# plot 2 - Mean size of clusters vs. protein-protein attraction strength
mean_2, std_2, sem_2 = get_stats(data_frames_4_trimmed, 'Mean_size_of_clusters')
mean_2_control, std_2_control, sem_2_control = get_stats(data_frames_4_trimmed_control, 'Mean_size_of_clusters')

pp_attraction = ('1', '2', '3', '4', '5', '6', '7', '8')
x_pos = np.arange(len(pp_attraction))

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

left_bar_2 = axs[0].bar(pp_attraction, mean_2, yerr=sem_2, color=colors[:9])
axs[0].bar_label(left_bar_2, labels=[f'{m:.2f}±{std_2[i]:.2f}' for i, m in enumerate(mean_2)], fontsize=10)

axs[0].set_xticks(x_pos)
axs[0].tick_params(axis='x', labelsize=14)
axs[0].tick_params(axis='y', labelsize=14)

axs[0].set_xlabel('Protein-protein attraction strength (kBT)', fontsize ='16')
axs[0].set_ylabel('Mean size of clusters', fontsize ='16')
axs[0].set_title('Mean size of clusters vs. protein-protein attraction strength', fontsize ='16')


right_bar_2 = axs[1].bar(pp_attraction, mean_2_control, yerr=sem_2_control, color=colors[:9])
axs[1].bar_label(right_bar_2, labels=[f'{m:.2f}±{std_2_control[i]:.2f}' for i, m in enumerate(mean_2_control)], fontsize=10)

axs[1].set_xticks(x_pos)
axs[1].tick_params(axis='x', labelsize=14)
axs[1].tick_params(axis='y', labelsize=14)

axs[1].set_xlabel('Protein-protein attraction strength (kBT)', fontsize ='16')
axs[1].set_ylabel('Mean size of clusters', fontsize ='16')
axs[1].set_title('Mean size of clusters vs. protein-protein attraction strength - Control', fontsize ='16')

plt.show()


# plot 3 - Size of largest cluster vs. protein-protein attraction strength
mean_3, std_3, sem_3 = get_stats(data_frames_4_trimmed, 'Size_of_largest_cluster')
mean_3_control, std_3_control, sem_3_control = get_stats(data_frames_4_trimmed_control, 'Size_of_largest_cluster')

pp_attraction = ('1', '2', '3', '4', '5', '6', '7', '8')
x_pos = np.arange(len(pp_attraction))

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

left_bar_3 = axs[0].bar(pp_attraction, mean_3, yerr=sem_3, color=colors[:9])
axs[0].bar_label(left_bar_3, labels=[f'{m:.2f}±{std_3[i]:.2f}' for i, m in enumerate(mean_3)], fontsize=10)

axs[0].set_xticks(x_pos)
axs[0].tick_params(axis='x', labelsize=14)
axs[0].tick_params(axis='y', labelsize=14)

axs[0].set_xlabel('Protein-protein attraction strength (kBT)', fontsize ='16')
axs[0].set_ylabel('Size of largest cluster', fontsize ='16')
axs[0].set_title('Size of largest cluster vs. protein-protein attraction strength', fontsize ='16')


right_bar_3 = axs[1].bar(pp_attraction, mean_3_control, yerr=sem_3_control, color=colors[:9])
axs[1].bar_label(right_bar_3, labels=[f'{m:.2f}±{std_3_control[i]:.2f}' for i, m in enumerate(mean_3_control)], fontsize=10)

axs[1].set_xticks(x_pos)
axs[1].tick_params(axis='x', labelsize=14)
axs[1].tick_params(axis='y', labelsize=14)

axs[1].set_xlabel('Protein-protein attraction strength (kBT)', fontsize ='16')
axs[1].set_ylabel('Size of largest cluster', fontsize ='16')
axs[1].set_title('Size of largest cluster vs. protein-protein attraction strength - Control', fontsize ='16')

plt.show()


# plot 4 - Number of clusters of size 1 vs. protein-protein attraction strength
mean_4, std_4, sem_4 = get_stats(data_frames_4_trimmed, 'No_of_clusters_of_size_1')
mean_4_control, std_4_control, sem_4_control = get_stats(data_frames_4_trimmed_control, 'No_of_clusters_of_size_1')

pp_attraction = ('1', '2', '3', '4', '5', '6', '7', '8')
x_pos = np.arange(len(pp_attraction))

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

left_bar_4 = axs[0].bar(pp_attraction, mean_4, yerr=sem_4, color=colors[:9])
axs[0].bar_label(left_bar_4, labels=[f'{m:.2f}±{std_4[i]:.2f}' for i, m in enumerate(mean_4)], fontsize=10)

axs[0].set_xticks(x_pos)
axs[0].tick_params(axis='x', labelsize=14)
axs[0].tick_params(axis='y', labelsize=14)

axs[0].set_xlabel('Protein-protein attraction strength (kBT)', fontsize ='16')
axs[0].set_ylabel('Number of clusters of size 1', fontsize ='16')
axs[0].set_title('Number of clusters of size 1 vs. protein-protein attraction strength', fontsize ='16')


right_bar_4 = axs[1].bar(pp_attraction, mean_4_control, yerr=sem_4_control, color=colors[:9])
axs[1].bar_label(right_bar_4, labels=[f'{m:.2f}±{std_4_control[i]:.2f}' for i, m in enumerate(mean_4_control)], fontsize=10)

axs[1].set_xticks(x_pos)
axs[1].tick_params(axis='x', labelsize=14)
axs[1].tick_params(axis='y', labelsize=14)

axs[1].set_xlabel('Protein-protein attraction strength (kBT)', fontsize ='16')
axs[1].set_ylabel('Number of clusters of size 1', fontsize ='16')
axs[1].set_title('Number of clusters of size 1 vs. protein-protein attraction strength - Control', fontsize ='16')

plt.show()


# plot 5 - Number of proteins bound to polymer vs. protein-protein attraction strength
mean_5, std_5, sem_5 = get_stats(data_frames_4_trimmed, 'No_proteins_bound_to_poly')
mean_5_control, std_5_control, sem_5_control = get_stats(data_frames_4_trimmed_control, 'No_proteins_bound_to_poly')

pp_attraction = ('1', '2', '3', '4', '5', '6', '7', '8')
x_pos = np.arange(len(pp_attraction))

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

left_bar_5 = axs[0].bar(pp_attraction, mean_5, yerr=sem_5, color=colors[:9])
axs[0].bar_label(left_bar_5, labels=[f'{m:.2f}±{std_5[i]:.2f}' for i, m in enumerate(mean_5)], fontsize=10)

axs[0].set_xticks(x_pos)
axs[0].tick_params(axis='x', labelsize=14)
axs[0].tick_params(axis='y', labelsize=14)

axs[0].set_xlabel('Protein-protein attraction strength (kBT)', fontsize ='16')
axs[0].set_ylabel('Number of proteins bound to polymer', fontsize ='16')
axs[0].set_title('Number of proteins bound to polymer vs. protein-protein attraction strength', fontsize ='16')


right_bar_5 = axs[1].bar(pp_attraction, mean_5_control, yerr=sem_5_control, color=colors[:9])
axs[1].bar_label(right_bar_5, labels=[f'{m:.2f}±{std_5_control[i]:.2f}' for i, m in enumerate(mean_5_control)], fontsize=10)

axs[1].set_xticks(x_pos)
axs[1].tick_params(axis='x', labelsize=14)
axs[1].tick_params(axis='y', labelsize=14)

axs[1].set_xlabel('Protein-protein attraction strength (kBT)', fontsize ='16')
axs[1].set_ylabel('Number of proteins bound to polymer', fontsize ='16')
axs[1].set_title('Number of proteins bound to polymer vs. protein-protein attraction strength - Control', fontsize ='16')

plt.show()