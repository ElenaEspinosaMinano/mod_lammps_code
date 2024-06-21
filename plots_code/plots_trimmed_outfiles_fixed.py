### Using trimmed outfiles to calculate the mean, standard error on the mean and standard deviation of quantities - fixed :)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# specify paths to read data from + save plots to
local_or_cplab = input("Local or cplab: ")

if local_or_cplab == "local":
    path_to_trimmed_outfiles = '/home/elenaespinosa/OneDrive/Uni/Summer_courses/Summer_project/mod_lammps_code/outfiles/trimmed_outfiles/'
    save_plots_to = '/home/elenaespinosa/OneDrive/Uni/Summer_courses/Summer_project/mod_lammps_code/plots/steady_state/'
else:
    path_to_trimmed_outfiles = '/home/s2205640/Documents/summer_project/mod_lammps_code/outfiles/trimmed_outfiles/'
    save_plots_to = '/home/s2205640/Documents/summer_project/mod_lammps_code/plots/steady_state/'


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


def plots_models_123(mean_123, std_123, sem_123, column_name, column_no):
    bar_labels_mean = [f'{model}: {mean_123[i]:.2f} ± {sem_123[i]:.2f} (1 SEM)' for i, model in enumerate(models)]
    bar_labels_std = [f'{model}: {std_123[i]:.2f} ± ---' for i, model in enumerate(models)]

    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(16, 10), tight_layout=True)

    fig.supxlabel('Models', fontsize ='16')
    fig.supylabel(column_name, fontsize ='16')
    fig.suptitle(f'{column_name} for different models - Mean ± 1 SEM (left) and STD (right)', fontsize='16')

    left_bar = axs[0].bar(models, mean_123, yerr=sem_123, capsize=2, label=bar_labels_mean, color=colors[:4])

    axs[0].set_xticks(x_pos)
    axs[0].tick_params(labelsize=14)
    axs[0].legend(fontsize=14)
    axs[0].grid(True, alpha=0.5)


    right_bar = axs[1].bar(models, std_123, label=bar_labels_std, color=colors[:4])

    axs[1].set_xticks(x_pos)
    axs[1].tick_params(labelsize=14)
    axs[1].legend(fontsize=14)
    axs[1].grid(True, alpha=0.5)

    plt.savefig(save_plots_to + f"plot_{column_no}_model_123_SS_run_1.png", dpi='figure')
    plt.show()


# plots 1 to 7 for models 1, 2 and 3 - Mean ± 1 SEM (left) and STD (right)


mean_123_1, std_123_1, sem_123_1 = get_stats_123(data_frames_123_trimmed, 'No_of_clusters')
plots_models_123(mean_123_1, std_123_1, sem_123_1, 'Number of clusters', 1)

mean_123_2, std_123_2, sem_123_2 = get_stats_123(data_frames_123_trimmed, 'Mean_size_of_clusters')
plots_models_123(mean_123_2, std_123_2, sem_123_2, 'Mean size of clusters', 2)

mean_123_3, std_123_3, sem_123_3 = get_stats_123(data_frames_123_trimmed, 'Size_of_largest_cluster')
plots_models_123(mean_123_3, std_123_3, sem_123_3, 'Size of largest cluster', 3)

mean_123_4, std_123_4, sem_123_4 = get_stats_123(data_frames_123_trimmed, 'No_of_clusters_of_size_1')
plots_models_123(mean_123_4, std_123_4, sem_123_4, 'Number of clusters of size 1', 4)

mean_123_5, std_123_5, sem_123_5 = get_stats_123(data_frames_123_trimmed, 'No_proteins_bound_to_poly')
plots_models_123(mean_123_5, std_123_5, sem_123_5, 'Number of proteins bound to polymer', 5)

mean_123_6, std_123_6, sem_123_6 = get_stats_123(data_frames_123_trimmed, 'Fraction_clusters_bound_to_poly')
plots_models_123(mean_123_6, std_123_6, sem_123_6, 'Fraction of clusters bound to polymer', 6)

mean_123_7, std_123_7, sem_123_7 = get_stats_123(data_frames_123_trimmed, 'No_type_2_poly_bound_to_prot')
plots_models_123(mean_123_7, std_123_7, sem_123_7, 'Number of type 2 polymers bound to proteins', 7)



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


def plots_model_4(mean_4, std_4, sem_4, mean_4_control, std_4_control, sem_4_control, column_name, column_no):
    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(16, 10), tight_layout=True)

    fig.supxlabel('Protein-protein attraction strength (kBT)', fontsize ='16')
    fig.supylabel(f'{column_name}', fontsize ='16')
    fig.suptitle(f'{column_name} vs. protein-protein attraction strength - Mean ± 1 SEM (left) and Control (right)', fontsize='16')

    left_bar_1 = axs[0].errorbar(pp_attraction, mean_4, yerr=sem_4, capsize=2, fmt='.r', alpha=0.7, ecolor='black')

    axs[0].set_xticks(pp_attraction)
    axs[0].tick_params(labelsize=14)
    axs[0].grid(True, alpha=0.5)


    right_bar_1 = axs[1].errorbar(pp_attraction, mean_4_control, yerr=sem_4_control, capsize=2, fmt='.r', alpha=0.7, ecolor='black')

    axs[1].set_xticks(pp_attraction)
    axs[1].tick_params(labelsize=14)
    axs[1].grid(True, alpha=0.5)

    plt.savefig(save_plots_to + f"plot_{column_no}_model_4_SS_run_1.png", dpi='figure')
    plt.show()


# plots 1 to 7 for model 4 - Mean ± 1 SEM (left) and Control (right)


mean_4_1, std_4_1, sem_4_1 = get_stats_4(data_frames_4_trimmed, 'No_of_clusters')
mean_4_1_control, std_4_1_control, sem_4_1_control = get_stats_4(data_frames_4_trimmed_control, 'No_of_clusters')

plots_model_4(mean_4_1, std_4_1, sem_4_1, mean_4_1_control, std_4_1_control, sem_4_1_control, 'Number of clusters', 1)

mean_4_2, std_4_2, sem_4_2 = get_stats_4(data_frames_4_trimmed, 'Mean_size_of_clusters')
mean_4_2_control, std_4_2_control, sem_4_2_control = get_stats_4(data_frames_4_trimmed_control, 'Mean_size_of_clusters')

plots_model_4(mean_4_2, std_4_2, sem_4_2, mean_4_2_control, std_4_2_control, sem_4_2_control, 'Mean size of clusters', 2)

mean_4_3, std_4_3, sem_4_3 = get_stats_4(data_frames_4_trimmed, 'Size_of_largest_cluster')
mean_4_3_control, std_4_3_control, sem_4_3_control = get_stats_4(data_frames_4_trimmed_control, 'Size_of_largest_cluster')

plots_model_4(mean_4_3, std_4_3, sem_4_3, mean_4_3_control, std_4_3_control, sem_4_3_control, 'Size of largest cluster', 3)

mean_4_4, std_4_4, sem_4_4 = get_stats_4(data_frames_4_trimmed, 'No_of_clusters_of_size_1')
mean_4_4_control, std_4_4_control, sem_4_4_control = get_stats_4(data_frames_4_trimmed_control, 'No_of_clusters_of_size_1')

plots_model_4(mean_4_4, std_4_4, sem_4_4, mean_4_4_control, std_4_4_control, sem_4_4_control, 'Number of clusters of size 1', 4)

mean_4_5, std_4_5, sem_4_5 = get_stats_4(data_frames_4_trimmed, 'No_proteins_bound_to_poly')
mean_4_5_control, std_4_5_control, sem_4_5_control = get_stats_4(data_frames_4_trimmed_control, 'No_proteins_bound_to_poly')

plots_model_4(mean_4_5, std_4_5, sem_4_5, mean_4_5_control, std_4_5_control, sem_4_5_control, 'Number of proteins bound to polymer', 5)

mean_4_6, std_4_6, sem_4_6 = get_stats_4(data_frames_4_trimmed, 'Fraction_clusters_bound_to_poly')
mean_4_6_control, std_4_6_control, sem_4_6_control = get_stats_4(data_frames_4_trimmed_control, 'Fraction_clusters_bound_to_poly')

plots_model_4(mean_4_6, std_4_6, sem_4_6, mean_4_6_control, std_4_6_control, sem_4_6_control, 'Fraction of clusters bound to polymer', 6)

mean_4_7, std_4_7, sem_4_7 = get_stats_4(data_frames_4_trimmed, 'No_type_2_poly_bound_to_prot')
mean_4_7_control, std_4_7_control, sem_4_7_control = get_stats_4(data_frames_4_trimmed_control, 'No_type_2_poly_bound_to_prot')

plots_model_4(mean_4_7, std_4_7, sem_4_7, mean_4_7_control, std_4_7_control, sem_4_7_control, 'Number of type 2 polymers bound to proteins', 7)