### Using trimmed outfiles to calculate the mean, standard error on the mean and standard deviation of quantities - fixed :)

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


def calc_stats(data_frame):

    mean = data_frame.mean()
    std = data_frame.std() # sample standard deviation, normalized by 1 / sqrt(N-1)
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
                                        'Size_of_largest_cluster', 'No_of_clusters_of_size_1', 'No_proteins_bound_to_poly', 
                                        'Fraction_clusters_bound_to_poly', 'No_type_2_poly_bound_to_prot']


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


models = ('Model 1', 'Model 2', 'Model 3')
x_pos = np.arange(len(models))


# plot 1 - Number of clusters vs Timesteps - Mean ± SEM and STD
mean_123_1, std_123_1, sem_123_1 = get_stats_123(data_frames_123_trimmed, 'No_of_clusters')

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
axs[0].grid(True, alpha=0.5)


right_bar_1 = axs[1].bar(models, std_123_1, label=bar_labels_std, color=colors[:4])

axs[1].set_xticks(x_pos)
axs[1].tick_params(axis='x', labelsize=14)
axs[1].tick_params(axis='y', labelsize=14)
#axs[1].set_xlabel('Models', fontsize ='16')
axs[1].set_ylabel('Number of clusters', fontsize ='16')
axs[1].set_title('Number of clusters for different models - STD', fontsize ='16')
axs[1].legend(fontsize=14)
axs[1].grid(True, alpha=0.5)

plt.savefig(save_plots_to + "plot_1_model_123_SS_run_1.png", dpi='figure')
plt.show()


# plot 2 - Mean size of clusters vs Timesteps - Mean ± SEM and STD
mean_123_2, std_123_2, sem_123_2 = get_stats_123(data_frames_123_trimmed, 'Mean_size_of_clusters')

bar_labels_mean = [f'{model}: {mean_123_2[i]:.2f} ± {sem_123_2[i]:.2f} (1 SEM)' for i, model in enumerate(models)]
bar_labels_std = [f'{model}: {std_123_2[i]:.2f} ± ---' for i, model in enumerate(models)]

fig, axs = plt.subplots(1, 2, sharey=True, figsize=(16, 10), tight_layout=True)

left_bar_2 = axs[0].bar(models, mean_123_2, yerr=sem_123_2, label=bar_labels_mean, color=colors[:4])

axs[0].set_xticks(x_pos)
axs[0].tick_params(axis='x', labelsize=14)
axs[0].tick_params(axis='y', labelsize=14)
#axs[0].set_xlabel('Models', fontsize ='16')
axs[0].set_ylabel('Mean size of clusters', fontsize ='16')
axs[0].set_title('Mean size of clusters for different models - Mean ± 1 SEM', fontsize ='16')
axs[0].legend(fontsize=14)
axs[0].grid(True, alpha=0.5)


right_bar_2 = axs[1].bar(models, std_123_2, label=bar_labels_std, color=colors[:4])

axs[1].set_xticks(x_pos)
axs[1].tick_params(axis='x', labelsize=14)
axs[1].tick_params(axis='y', labelsize=14)
#axs[1].set_xlabel('Models', fontsize ='16')
axs[1].set_ylabel('Mean size of clusters', fontsize ='16')
axs[1].set_title('Mean size of clusters for different models - STD', fontsize ='16')
axs[1].legend(fontsize=14)
axs[1].grid(True, alpha=0.5)

plt.savefig(save_plots_to + "plot_2_model_123_SS_run_1.png", dpi='figure')
plt.show()


# plot 3 - Size of largest cluster vs Timesteps - Mean ± SEM and STD
mean_123_3, std_123_3, sem_123_3 = get_stats_123(data_frames_123_trimmed, 'Size_of_largest_cluster')

bar_labels_mean = [f'{model}: {mean_123_3[i]:.2f} ± {sem_123_3[i]:.2f} (1 SEM)' for i, model in enumerate(models)]
bar_labels_std = [f'{model}: {std_123_3[i]:.2f} ± ---' for i, model in enumerate(models)]

fig, axs = plt.subplots(1, 2, sharey=True, figsize=(16, 10), tight_layout=True)

left_bar_3 = axs[0].bar(models, mean_123_3, yerr=sem_123_3, label=bar_labels_mean, color=colors[:4])

axs[0].set_xticks(x_pos)
axs[0].tick_params(axis='x', labelsize=14)
axs[0].tick_params(axis='y', labelsize=14)
#axs[0].set_xlabel('Models', fontsize ='16')
axs[0].set_ylabel('Size of largest cluster', fontsize ='16')
axs[0].set_title('Size of largest cluster for different models - Mean ± 1 SEM', fontsize ='16')
axs[0].legend(fontsize=14)
axs[0].grid(True, alpha=0.5)


right_bar_3 = axs[1].bar(models, std_123_3, label=bar_labels_std, color=colors[:4])

axs[1].set_xticks(x_pos)
axs[1].tick_params(axis='x', labelsize=14)
axs[1].tick_params(axis='y', labelsize=14)
#axs[1].set_xlabel('Models', fontsize ='16')
axs[1].set_ylabel('Size of largest cluster', fontsize ='16')
axs[1].set_title('Size of largest cluster for different models - STD', fontsize ='16')
axs[1].legend(fontsize=14)
axs[1].grid(True, alpha=0.5)

plt.savefig(save_plots_to + "plot_3_model_123_SS_run_1.png", dpi='figure')
plt.show()


# plot 4 - Number of clusters of size 1 vs Timesteps - Mean ± SEM and STD
mean_123_4, std_123_4, sem_123_4 = get_stats_123(data_frames_123_trimmed, 'No_of_clusters_of_size_1')

bar_labels_mean = [f'{model}: {mean_123_4[i]:.2f} ± {sem_123_4[i]:.2f} (1 SEM)' for i, model in enumerate(models)]
bar_labels_std = [f'{model}: {std_123_4[i]:.2f} ± ---' for i, model in enumerate(models)]

fig, axs = plt.subplots(1, 2, sharey=True, figsize=(16, 10), tight_layout=True)

left_bar_4 = axs[0].bar(models, mean_123_4, yerr=sem_123_4, label=bar_labels_mean, color=colors[:4])

axs[0].set_xticks(x_pos)
axs[0].tick_params(axis='x', labelsize=14)
axs[0].tick_params(axis='y', labelsize=14)
#axs[0].set_xlabel('Models', fontsize ='16')
axs[0].set_ylabel('Number of clusters of size 1', fontsize ='16')
axs[0].set_title('Number of clusters of size 1 for different models - Mean ± 1 SEM', fontsize ='16')
axs[0].legend(fontsize=14)
axs[0].grid(True, alpha=0.5)


right_bar_4 = axs[1].bar(models, std_123_4, label=bar_labels_std, color=colors[:4])

axs[1].set_xticks(x_pos)
axs[1].tick_params(axis='x', labelsize=14)
axs[1].tick_params(axis='y', labelsize=14)
#axs[1].set_xlabel('Models', fontsize ='16')
axs[1].set_ylabel('Number of clusters of size 1', fontsize ='16')
axs[1].set_title('Number of clusters of size 1 for different models - STD', fontsize ='16')
axs[1].legend(fontsize=14)
axs[1].grid(True, alpha=0.5)

plt.savefig(save_plots_to + "plot_4_model_123_SS_run_1.png", dpi='figure')
plt.show()


# plot 5 - Number of proteins bound to polymer vs Timesteps - Mean ± 2 SEM and STD
mean_123_5, std_123_5, sem_123_5 = get_stats_123(data_frames_123_trimmed, 'No_proteins_bound_to_poly')

bar_labels_mean = [f'{model}: {mean_123_5[i]:.2f} ± {sem_123_5[i]:.2f} (1 SEM)' for i, model in enumerate(models)]
bar_labels_std = [f'{model}: {std_123_5[i]:.2f} ± ---' for i, model in enumerate(models)]

fig, axs = plt.subplots(1, 2, sharey=True, figsize=(16, 10), tight_layout=True)

left_bar_5 = axs[0].bar(models, mean_123_5, yerr=sem_123_5, label=bar_labels_mean, color=colors[:4])

axs[0].set_xticks(x_pos)
axs[0].tick_params(axis='x', labelsize=14)
axs[0].tick_params(axis='y', labelsize=14)
#axs[0].set_xlabel('Models', fontsize ='16')
axs[0].set_ylabel('Number of proteins bound to polymer', fontsize ='16')
axs[0].set_title('Number of proteins bound to polymer for different models - Mean ± 1 SEM', fontsize ='16')
axs[0].legend(fontsize=14)
axs[0].grid(True, alpha=0.5)


right_bar_5 = axs[1].bar(models, std_123_5, label=bar_labels_std, color=colors[:4])

axs[1].set_xticks(x_pos)
axs[1].tick_params(axis='x', labelsize=14)
axs[1].tick_params(axis='y', labelsize=14)
#axs[1].set_xlabel('Models', fontsize ='16')
axs[1].set_ylabel('Number of proteins bound to polymer', fontsize ='16')
axs[1].set_title('Number of proteins bound to polymer for different models - STD', fontsize ='16')
axs[1].legend(fontsize=14)
axs[1].grid(True, alpha=0.5)

plt.savefig(save_plots_to + "plot_5_model_123_SS_run_1.png", dpi='figure')
plt.show()


# plot 6 - Fraction of clusters bound to polymer vs Timesteps - Mean ± 1 SEM and STD
mean_123_6, std_123_6, sem_123_6 = get_stats_123(data_frames_123_trimmed, 'Fraction_clusters_bound_to_poly')

bar_labels_mean = [f'{model}: {mean_123_6[i]:.2f} ± {sem_123_6[i]:.2f} (1 SEM)' for i, model in enumerate(models)]
bar_labels_std = [f'{model}: {std_123_6[i]:.2f} ± ---' for i, model in enumerate(models)]

fig, axs = plt.subplots(1, 2, sharey=True, figsize=(16, 10), tight_layout=True)

left_bar_6 = axs[0].bar(models, mean_123_6, yerr=sem_123_6, label=bar_labels_mean, color=colors[:4])

axs[0].set_xticks(x_pos)
axs[0].tick_params(axis='x', labelsize=14)
axs[0].tick_params(axis='y', labelsize=14)
#axs[0].set_xlabel('Models', fontsize ='16')
axs[0].set_ylabel('Fraction of clusters bound to polymer', fontsize ='16')
axs[0].set_title('Fraction of clusters bound to polymer for different models - Mean ± 1 SEM', fontsize ='16')
axs[0].legend(fontsize=14)
axs[0].grid(True, alpha=0.5)


right_bar_6 = axs[1].bar(models, std_123_6, label=bar_labels_std, color=colors[:4])

axs[1].set_xticks(x_pos)
axs[1].tick_params(axis='x', labelsize=14)
axs[1].tick_params(axis='y', labelsize=14)
#axs[1].set_xlabel('Models', fontsize ='16')
axs[1].set_ylabel('Fraction of clusters bound to polymer', fontsize ='16')
axs[1].set_title('Fraction of clusters bound to polymer for different models - STD', fontsize ='16')
axs[1].legend(fontsize=14)
axs[1].grid(True, alpha=0.5)

plt.savefig(save_plots_to + "plot_6_model_123_SS_run_1.png", dpi='figure')
plt.show()


# plot 7 - Number of type 2 polymers bound to proteins vs Timesteps - Mean ± 1 SEM and STD
mean_123_7, std_123_7, sem_123_7 = get_stats_123(data_frames_123_trimmed, 'No_type_2_poly_bound_to_prot')

bar_labels_mean = [f'{model}: {mean_123_7[i]:.2f} ± {sem_123_7[i]:.2f} (1 SEM)' for i, model in enumerate(models)]
bar_labels_std = [f'{model}: {std_123_7[i]:.2f} ± ---' for i, model in enumerate(models)]

fig, axs = plt.subplots(1, 2, sharey=True, figsize=(16, 10), tight_layout=True)

left_bar_7 = axs[0].bar(models, mean_123_7, yerr=sem_123_7, label=bar_labels_mean, color=colors[:4])

axs[0].set_xticks(x_pos)
axs[0].tick_params(axis='x', labelsize=14)
axs[0].tick_params(axis='y', labelsize=14)
#axs[0].set_xlabel('Models', fontsize ='16')
axs[0].set_ylabel('Number of type 2 polymers bound to proteins', fontsize ='16')
axs[0].set_title('Number of type 2 polymers bound to proteins for different models - Mean ± 1 SEM', fontsize ='16')
axs[0].legend(fontsize=14)
axs[0].grid(True, alpha=0.5)


right_bar_7 = axs[1].bar(models, std_123_7, label=bar_labels_std, color=colors[:4])

axs[1].set_xticks(x_pos)
axs[1].tick_params(axis='x', labelsize=14)
axs[1].tick_params(axis='y', labelsize=14)
#axs[1].set_xlabel('Models', fontsize ='16')
axs[1].set_ylabel('Number of type 2 polymers bound to proteins', fontsize ='16')
axs[1].set_title('Number of type 2 polymers bound to proteins for different models - STD', fontsize ='16')
axs[1].legend(fontsize=14)
axs[1].grid(True, alpha=0.5)

plt.savefig(save_plots_to + "plot_7_model_123_SS_run_1.png", dpi='figure')
plt.show()



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
                                        'Size_of_largest_cluster', 'No_of_clusters_of_size_1', 'No_proteins_bound_to_poly', 
                                        'Fraction_clusters_bound_to_poly', 'No_type_2_poly_bound_to_prot']
    data_frames_4_trimmed_control[i] = pd.read_csv(path_to_trimmed_outfiles + trimmed_outfiles_list_4_control[i-1], sep=' ', comment='#', header=None)
    data_frames_4_trimmed_control[i].columns = ['Timesteps', 'No_of_clusters', 'Mean_size_of_clusters', 
                                        'Size_of_largest_cluster', 'No_of_clusters_of_size_1', 'No_proteins_bound_to_poly', 
                                        'Fraction_clusters_bound_to_poly', 'No_type_2_poly_bound_to_prot']


# get stats of model 4 for a particular dataframe and column + append it to a list
def get_stats_4(data_frames, column):
    mean_list = []
    std_list = []
    sem_list = []
    for i in range(1, 9):
        mean, std, sem = calc_stats(data_frames[i][column])
        mean_list.append(mean)
        std_list.append(std)
        sem_list.append(sem)
    return mean_list, std_list, sem_list


pp_attraction = [1, 2, 3, 4, 5, 6, 7, 8]


# plot 1 - Number of clusters vs. protein-protein attraction strength - Mean ± SEM with control
mean_4_1, std_4_1, sem_4_1 = get_stats_4(data_frames_4_trimmed, 'No_of_clusters')
mean_4_1_control, std_4_1_control, sem_4_1_control = get_stats_4(data_frames_4_trimmed_control, 'No_of_clusters')

fig, axs = plt.subplots(1, 2, sharey=True, figsize=(16, 10), tight_layout=True)

left_bar_1 = axs[0].errorbar(pp_attraction, mean_4_1, yerr=sem_4_1, fmt='or', alpha=0.7, ecolor='black')

axs[0].set_xticks(pp_attraction)
axs[0].tick_params(axis='x', labelsize=14)
axs[0].tick_params(axis='y', labelsize=14)
axs[0].set_xlabel('Protein-protein attraction strength (kBT)', fontsize ='16')
axs[0].set_ylabel('Number of clusters', fontsize ='16')
axs[0].set_title('Number of clusters vs. protein-protein attraction strength', fontsize ='16')
axs[0].grid(True, alpha=0.5)


right_bar_1 = axs[1].errorbar(pp_attraction, mean_4_1_control, yerr=sem_4_1_control, fmt='or', alpha=0.7, ecolor='black')

axs[1].set_xticks(pp_attraction)
axs[1].tick_params(axis='x', labelsize=14)
axs[1].tick_params(axis='y', labelsize=14)
axs[1].set_xlabel('Protein-protein attraction strength (kBT)', fontsize ='16')
axs[1].set_ylabel('Number of clusters', fontsize ='16')
axs[1].set_title('Number of clusters vs. protein-protein attraction strength - Control', fontsize ='16')
axs[1].grid(True, alpha=0.5)

plt.savefig(save_plots_to + "plot_1_model_4_SS_run_1.png", dpi='figure')
plt.show()


# plot 2 - Mean size of clusters vs. protein-protein attraction strength
mean_5_1, std_5_1, sem_5_1 = get_stats_4(data_frames_4_trimmed, 'Mean_size_of_clusters')
mean_5_1_control, std_5_1_control, sem_5_1_control = get_stats_4(data_frames_4_trimmed_control, 'Mean_size_of_clusters')

fig, axs = plt.subplots(1, 2, sharey=True, figsize=(16, 10), tight_layout=True)

left_bar_2 = axs[0].errorbar(pp_attraction, mean_5_1, yerr=sem_5_1, fmt='or', alpha=0.7, ecolor='black')

axs[0].set_xticks(pp_attraction)
axs[0].tick_params(axis='x', labelsize=14)
axs[0].tick_params(axis='y', labelsize=14)
axs[0].set_xlabel('Protein-protein attraction strength (kBT)', fontsize ='16')
axs[0].set_ylabel('Mean size of clusters', fontsize ='16')
axs[0].set_title('Mean size of clusters vs. protein-protein attraction strength', fontsize ='16')
axs[0].grid(True, alpha=0.5)


right_bar_2 = axs[1].errorbar(pp_attraction, mean_5_1_control, yerr=sem_5_1_control, fmt='or', alpha=0.7, ecolor='black')

axs[1].set_xticks(pp_attraction)
axs[1].tick_params(axis='x', labelsize=14)
axs[1].tick_params(axis='y', labelsize=14)
axs[1].set_xlabel('Protein-protein attraction strength (kBT)', fontsize ='16')
axs[1].set_ylabel('Mean size of clusters', fontsize ='16')
axs[1].set_title('Mean size of clusters vs. protein-protein attraction strength - Control', fontsize ='16')
axs[1].grid(True, alpha=0.5)

plt.savefig(save_plots_to + "plot_2_model_4_SS_run_1.png", dpi='figure')
plt.show()


# plot 3 - Size of largest cluster vs. protein-protein attraction strength
mean_4_3, std_4_3, sem_4_3 = get_stats_4(data_frames_4_trimmed, 'Size_of_largest_cluster')
mean_4_3_control, std_4_3_control, sem_4_3_control = get_stats_4(data_frames_4_trimmed_control, 'Size_of_largest_cluster')

fig, axs = plt.subplots(1, 2, sharey=True, figsize=(16, 10), tight_layout=True)

left_bar_3 = axs[0].errorbar(pp_attraction, mean_4_3, yerr=sem_4_3, fmt='or', alpha=0.7, ecolor='black')

axs[0].set_xticks(pp_attraction)
axs[0].tick_params(axis='x', labelsize=14)
axs[0].tick_params(axis='y', labelsize=14)
axs[0].set_xlabel('Protein-protein attraction strength (kBT)', fontsize ='16')
axs[0].set_ylabel('Size of largest cluster', fontsize ='16')
axs[0].set_title('Size of largest cluster vs. protein-protein attraction strength', fontsize ='16')
axs[0].grid(True, alpha=0.5)


right_bar_3 = axs[1].errorbar(pp_attraction, mean_4_3_control, yerr=sem_4_3_control, fmt='or', alpha=0.7, ecolor='black')

axs[1].set_xticks(pp_attraction)
axs[1].tick_params(axis='x', labelsize=14)
axs[1].tick_params(axis='y', labelsize=14)
axs[1].set_xlabel('Protein-protein attraction strength (kBT)', fontsize ='16')
axs[1].set_ylabel('Size of largest cluster', fontsize ='16')
axs[1].set_title('Size of largest cluster vs. protein-protein attraction strength - Control', fontsize ='16')
axs[1].grid(True, alpha=0.5)

plt.savefig(save_plots_to + "plot_3_model_4_SS_run_1.png", dpi='figure')
plt.show()


# plot 4 - Number of clusters of size 1 vs. protein-protein attraction strength
mean_4_4, std_4_4, sem_4_4 = get_stats_4(data_frames_4_trimmed, 'No_of_clusters_of_size_1')
mean_4_4_control, std_4_4_control, sem_4_4_control = get_stats_4(data_frames_4_trimmed_control, 'No_of_clusters_of_size_1')

fig, axs = plt.subplots(1, 2, sharey=True, figsize=(16, 10), tight_layout=True)

left_bar_4 = axs[0].errorbar(pp_attraction, mean_4_4, yerr=sem_4_4, fmt='or', alpha=0.7, ecolor='black')

axs[0].set_xticks(pp_attraction)
axs[0].tick_params(axis='x', labelsize=14)
axs[0].tick_params(axis='y', labelsize=14)
axs[0].set_xlabel('Protein-protein attraction strength (kBT)', fontsize ='16')
axs[0].set_ylabel('Number of clusters of size 1', fontsize ='16')
axs[0].set_title('Number of clusters of size 1 vs. protein-protein attraction strength', fontsize ='16')
axs[0].grid(True, alpha=0.5)


right_bar_4 = axs[1].errorbar(pp_attraction, mean_4_4_control, yerr=sem_4_4_control, fmt='or', alpha=0.7, ecolor='black')

axs[1].set_xticks(pp_attraction)
axs[1].tick_params(axis='x', labelsize=14)
axs[1].tick_params(axis='y', labelsize=14)
axs[1].set_xlabel('Protein-protein attraction strength (kBT)', fontsize ='16')
axs[1].set_ylabel('Number of clusters of size 1', fontsize ='16')
axs[1].set_title('Number of clusters of size 1 vs. protein-protein attraction strength - Control', fontsize ='16')
axs[1].grid(True, alpha=0.5)

plt.savefig(save_plots_to + "plot_4_model_4_SS_run_1.png", dpi='figure')
plt.show()


# plot 5 - Number of proteins bound to polymer vs. protein-protein attraction strength
mean_4_5, std_4_5, sem_4_5 = get_stats_4(data_frames_4_trimmed, 'No_proteins_bound_to_poly')
mean_4_5_control, std_4_5_control, sem_4_5_control = get_stats_4(data_frames_4_trimmed_control, 'No_proteins_bound_to_poly')

fig, axs = plt.subplots(1, 2, sharey=True, figsize=(16, 10), tight_layout=True)

left_bar_5 = axs[0].errorbar(pp_attraction, mean_4_5, yerr=sem_4_5, fmt='or', alpha=0.7, ecolor='black')

axs[0].set_xticks(pp_attraction)
axs[0].tick_params(axis='x', labelsize=14)
axs[0].tick_params(axis='y', labelsize=14)
axs[0].set_xlabel('Protein-protein attraction strength (kBT)', fontsize ='16')
axs[0].set_ylabel('Number of proteins bound to polymer', fontsize ='16')
axs[0].set_title('Number of proteins bound to polymer vs. protein-protein attraction strength', fontsize ='16')
axs[0].grid(True, alpha=0.5)


right_bar_5 = axs[1].errorbar(pp_attraction, mean_4_5_control, yerr=sem_4_5_control, fmt='or', alpha=0.7, ecolor='black')

axs[1].set_xticks(pp_attraction)
axs[1].tick_params(axis='x', labelsize=14)
axs[1].tick_params(axis='y', labelsize=14)
axs[1].set_xlabel('Protein-protein attraction strength (kBT)', fontsize ='16')
axs[1].set_ylabel('Number of proteins bound to polymer', fontsize ='16')
axs[1].set_title('Number of proteins bound to polymer vs. protein-protein attraction strength - Control', fontsize ='16')
axs[1].grid(True, alpha=0.5)

plt.savefig(save_plots_to + "plot_5_model_4_SS_run_1.png", dpi='figure')
plt.show()


# plot 6 - Fraction of clusters bound to polymer vs. protein-protein attraction strength
mean_4_6, std_4_6, sem_4_6 = get_stats_4(data_frames_4_trimmed, 'Fraction_clusters_bound_to_poly')
mean_4_6_control, std_4_6_control, sem_4_6_control = get_stats_4(data_frames_4_trimmed_control, 'Fraction_clusters_bound_to_poly')

fig, axs = plt.subplots(1, 2, sharey=True, figsize=(16, 10), tight_layout=True)

left_bar_6 = axs[0].errorbar(pp_attraction, mean_4_6, yerr=sem_4_6, fmt='or', alpha=0.7, ecolor='black')

axs[0].set_xticks(pp_attraction)
axs[0].tick_params(axis='x', labelsize=14)
axs[0].tick_params(axis='y', labelsize=14)
axs[0].set_xlabel('Protein-protein attraction strength (kBT)', fontsize ='16')
axs[0].set_ylabel('Fraction of clusters bound to polymer', fontsize ='16')
axs[0].set_title('Fraction of clusters bound to polymer vs. protein-protein attraction strength', fontsize ='16')
axs[0].grid(True, alpha=0.5)


right_bar_6 = axs[1].errorbar(pp_attraction, mean_4_6_control, yerr=sem_4_6_control, fmt='or', alpha=0.7, ecolor='black')

axs[1].set_xticks(pp_attraction)
axs[1].tick_params(axis='x', labelsize=14)
axs[1].tick_params(axis='y', labelsize=14)
axs[1].set_xlabel('Protein-protein attraction strength (kBT)', fontsize ='16')
axs[1].set_ylabel('Fraction of clusters bound to polymer', fontsize ='16')
axs[1].set_title('Fraction of clusters bound to polymer vs. protein-protein attraction strength - Control', fontsize ='16')
axs[1].grid(True, alpha=0.5)

plt.savefig(save_plots_to + "plot_6_model_4_SS_run_1.png", dpi='figure')
plt.show()


# plot 7 - Number of type 2 polymers bound to proteins vs. protein-protein attraction strength
mean_4_7, std_4_7, sem_4_7 = get_stats_4(data_frames_4_trimmed, 'No_type_2_poly_bound_to_prot')
mean_4_7_control, std_4_7_control, sem_4_7_control = get_stats_4(data_frames_4_trimmed_control, 'No_type_2_poly_bound_to_prot')

fig, axs = plt.subplots(1, 2, sharey=True, figsize=(16, 10), tight_layout=True)

left_bar_7 = axs[0].errorbar(pp_attraction, mean_4_7, yerr=sem_4_7, fmt='or', alpha=0.7, ecolor='black')

axs[0].set_xticks(pp_attraction)
axs[0].tick_params(axis='x', labelsize=14)
axs[0].tick_params(axis='y', labelsize=14)
axs[0].set_xlabel('Protein-protein attraction strength (kBT)', fontsize ='16')
axs[0].set_ylabel('Number of type 2 polymers bound to proteins', fontsize ='16')
axs[0].set_title('Number of type 2 polymers bound to proteins vs. protein-protein attraction strength', fontsize ='16')
axs[0].grid(True, alpha=0.5)


right_bar_7 = axs[1].errorbar(pp_attraction, mean_4_7_control, yerr=sem_4_7_control, fmt='or', alpha=0.7, ecolor='black')

axs[1].set_xticks(pp_attraction)
axs[1].tick_params(axis='x', labelsize=14)
axs[1].tick_params(axis='y', labelsize=14)
axs[1].set_xlabel('Protein-protein attraction strength (kBT)', fontsize ='16')
axs[1].set_ylabel('Number of type 2 polymers bound to proteins', fontsize ='16')
axs[1].set_title('Number of type 2 polymers bound to proteins vs. protein-protein attraction strength - Control', fontsize ='16')
axs[1].grid(True, alpha=0.5)

plt.savefig(save_plots_to + "plot_7_model_4_SS_run_1.png", dpi='figure')
plt.show()