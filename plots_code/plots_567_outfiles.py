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
    sem = std / np.sqrt(data_frame.count()) # data_frame.count() = N = 1401

    return mean, std, sem


###
#   Third investigation - Model 5, 6 + 7 plots
###

trimmed_outfiles_list_567 = [f'trimmed_outfile_{i}_run_1.dat' for i in range(5, 8)]

# dictionary to store data frames
data_frames_567_trimmed = {}

# parse data
for i in range(1, 4):
    data_frames_567_trimmed[i] = pd.read_csv(path_to_trimmed_outfiles + trimmed_outfiles_list_567[i-1], sep=' ', comment='#', header=None)
    data_frames_567_trimmed[i].columns = ['Timesteps', 'No_of_clusters', 'Mean_size_of_clusters', 
                                        'Size_of_largest_cluster', 'No_of_clusters_of_size_1', 'No_proteins_bound_to_poly', 
                                        'Fraction_clusters_bound_to_poly', 'No_type_2_poly_bound_to_prot', 'Mean_no_type_2_in_cluster']


# get stats of models 567 for a particular dataframe and column + append it to a list
def get_stats_567(data_frames, column):
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


models = ('Model 5', 'Model 6', 'Model 7')
column_name = ['Number of clusters', 'Mean size of clusters', 'Size of largest cluster', 'Number of clusters of size 1', 'Number of proteins bound to polymer', 
                'Fraction of clusters bound to polymer', 'Number of type 2 polymers bound to proteins', 'Mean number of type 2 polymers per protein cluster']
data_frame_name = ['No_of_clusters', 'Mean_size_of_clusters', 
                   'Size_of_largest_cluster', 'No_of_clusters_of_size_1', 'No_proteins_bound_to_poly', 
                   'Fraction_clusters_bound_to_poly', 'No_type_2_poly_bound_to_prot', 'Mean_no_type_2_in_cluster']

x_pos = np.arange(len(models))


def SS_plots_models_567(mean_567, std_567, sem_567, column_name, column_no):
    bar_labels_mean = [f'{model}: {mean_567[i]:.2f} ± {sem_567[i]:.2f} (1 SEM)' for i, model in enumerate(models)]
    bar_labels_std = [f'{model}: {std_567[i]:.2f} ± ---' for i, model in enumerate(models)]

    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(16, 10), tight_layout=True)

    fig.supxlabel('Models', fontsize ='16')
    fig.supylabel(column_name, fontsize ='16')
    fig.suptitle(f'{column_name} for different models - Mean ± 1 SEM (left) and STD (right) - fixed', fontsize='16')

    left_bar = axs[0].bar(models, mean_567, yerr=sem_567, capsize=2, label=bar_labels_mean, color=colors[:4])

    axs[0].set_xticks(x_pos)
    axs[0].tick_params(labelsize=14)
    axs[0].legend(fontsize=14)
    axs[0].grid(True, alpha=0.5)


    right_bar = axs[1].bar(models, std_567, label=bar_labels_std, color=colors[:4])

    axs[1].set_xticks(x_pos)
    axs[1].tick_params(labelsize=14)
    axs[1].legend(fontsize=14)
    axs[1].grid(True, alpha=0.5)

    plt.savefig(save_plots_to + f"plot_{column_no}_model_567_SS_run_1_5.png", dpi='figure') # change this to _5 for last calc.
    plt.show()


# steady state plots 1 to 8 for models 5, 6 and 7 - Mean ± 1 SEM (left) and STD (right) - proteins being type 4 only
"""
for i, column in enumerate(column_name):
    mean_567, std_567, sem_567 = get_stats_567(data_frames_567_trimmed, data_frame_name[i])
    SS_plots_models_567(mean_567, std_567, sem_567, column_name[i], (i+1))
"""



trimmed_outfiles_list_567_5 = [f'trimmed_outfile_{i}_run_1_5.dat' for i in range(5, 8)]

# dictionary to store data frames
data_frames_567_trimmed_5 = {}

# parse data
for i in range(1, 4):
    data_frames_567_trimmed_5[i] = pd.read_csv(path_to_trimmed_outfiles + trimmed_outfiles_list_567_5[i-1], sep=' ', comment='#', header=None)
    data_frames_567_trimmed_5[i].columns = ['Timesteps', 'No_of_clusters', 'Mean_size_of_clusters', 
                                        'Size_of_largest_cluster', 'No_of_clusters_of_size_1', 'No_proteins_bound_to_poly', 
                                        'Fraction_clusters_bound_to_poly', 'No_type_2_poly_bound_to_prot', 'Mean_no_type_2_in_cluster']

# proteins being type 4 + 5

for i, column in enumerate(column_name):
    mean_567, std_567, sem_567 = get_stats_567(data_frames_567_trimmed_5, data_frame_name[i])
    SS_plots_models_567(mean_567, std_567, sem_567, column_name[i], (i+1))





def full_plots(data_frame_name, column_name, column_no):
    plt.figure(figsize=(16, 10))

    plt.plot(data_frames_567_trimmed_5[1]['Timesteps'], data_frames_567_trimmed_5[1][data_frame_name], marker='.', alpha=0.7, label='Model 5')
    plt.plot(data_frames_567_trimmed_5[2]['Timesteps'], data_frames_567_trimmed_5[2][data_frame_name], marker='.', alpha=0.7, label='Model 6')
    plt.plot(data_frames_567_trimmed_5[3]['Timesteps'], data_frames_567_trimmed_5[3][data_frame_name], marker='.', alpha=0.7, label='Model 7')

    plt.xlabel('Timesteps', fontsize ='16')
    plt.xticks(fontsize='14')

    plt.ylabel(f'{column_name}', fontsize ='16')
    plt.yticks(fontsize='14')

    plt.title(f'{column_name} vs Timesteps - fixed', fontsize ='16')
    plt.ticklabel_format(style='plain')

    plt.legend(fontsize="14")
    plt.grid(True)

    plt.savefig(save_plots_to + f"plot_{column_no}_model_5_run_1_5.png", dpi='figure')
    plt.show()


for i, column in enumerate(column_name):
    full_plots(data_frame_name[i], column_name[i], (i+1))






"""
# file path
path_test_file = '/home/s2205640/Documents/summer_project/mod_lammps_code/outfiles/outfile_5_run_1.dat'

# data test!
data_test = pd.read_csv(path_test_file, sep=' ', comment='#', header=None)
data_test.columns = ['Timesteps', 'No_of_clusters', 'Mean_size_of_clusters', 
                    'Size_of_largest_cluster', 'No_of_clusters_of_size_1', 'No_proteins_bound_to_poly', 
                    'Fraction_clusters_bound_to_poly', 'No_type_2_poly_bound_to_prot', 'Mean_no_type_2_in_cluster']


# plot test 1
plt.figure(figsize=(16, 10))


plt.xlabel('Timesteps', fontsize ='16')
plt.xticks(fontsize='14')

plt.ylabel('Number of clusters', fontsize ='16')
plt.yticks(fontsize='14')

plt.title('Number of clusters vs Timesteps', fontsize ='16')
plt.ticklabel_format(style='plain')

plt.legend(fontsize="14", loc ="upper left")
plt.grid(True)

plt.savefig(save_plots_to + f"plot_1_model_5_run_1.png", dpi='figure')
#plt.show()



# plot test 2
plt.figure(figsize=(16, 10))

plt.plot(data_test['Timesteps'], data_test['Mean_size_of_clusters'], marker='.', alpha=0.7, label='Model 5')

plt.xlabel('Timesteps', fontsize ='16')
plt.xticks(fontsize='14')

plt.ylabel('Mean size of clusters', fontsize ='16')
plt.yticks(fontsize='14')

plt.title('Mean size of clusters vs Timesteps', fontsize ='16')
plt.ticklabel_format(style='plain')

plt.legend(fontsize="14", loc ="upper left")
plt.grid(True)

plt.savefig(save_plots_to + f"plot_2_model_5_run_1.png", dpi='figure')
#plt.show()



# plot test 3
plt.figure(figsize=(16, 10))

plt.plot(data_test['Timesteps'], data_test['Size_of_largest_cluster'], marker='.', alpha=0.7, label='Model 5')

plt.xlabel('Timesteps', fontsize ='16')
plt.xticks(fontsize='14')

plt.ylabel('Size of largest cluster', fontsize ='16')
plt.yticks(fontsize='14')

plt.title('Size of largest cluster vs Timesteps', fontsize ='16')
plt.ticklabel_format(style='plain')

plt.legend(fontsize="14", loc ="upper left")
plt.grid(True)

plt.savefig(save_plots_to + f"plot_3_model_5_run_1.png", dpi='figure')
#plt.show()



# plot test 4
plt.figure(figsize=(16, 10))

plt.plot(data_test['Timesteps'], data_test['No_of_clusters_of_size_1'], marker='.', alpha=0.7, label='Model 5')

plt.xlabel('Timesteps', fontsize ='16')
plt.xticks(fontsize='14')

plt.ylabel('Number of clusters of size 1', fontsize ='16')
plt.yticks(fontsize='14')

plt.title('Number of clusters of size 1 vs Timesteps', fontsize ='16')
plt.ticklabel_format(style='plain')

plt.legend(fontsize="14", loc ="upper left")
plt.grid(True)

plt.savefig(save_plots_to + f"plot_4_model_5_run_1.png", dpi='figure')
#plt.show()



# plot test 5
plt.figure(figsize=(16, 10))

plt.plot(data_test['Timesteps'], data_test['No_proteins_bound_to_poly'], marker='.', alpha=0.7, label='Model 5')

plt.xlabel('Timesteps', fontsize ='16')
plt.xticks(fontsize='14')

plt.ylabel('Number of proteins bound to polymer', fontsize ='16')
plt.yticks(fontsize='14')

plt.title('Number of proteins bound to polymer vs Timesteps', fontsize ='16')
plt.ticklabel_format(style='plain')

plt.legend(fontsize="14", loc ="upper left")
plt.grid(True)

plt.savefig(save_plots_to + f"plot_5_model_5_run_1.png", dpi='figure')
#plt.show()



# plot test 6
plt.figure(figsize=(16, 10))

plt.plot(data_test['Timesteps'], data_test['Fraction_clusters_bound_to_poly'], marker='.', alpha=0.7, label='Model 5')

plt.xlabel('Timesteps', fontsize ='16')
plt.xticks(fontsize='14')

plt.ylabel('Fraction of clusters bound to polymer', fontsize ='16')
plt.yticks(fontsize='14')

plt.title('Fraction of clusters bound to polymer vs Timesteps', fontsize ='16')
plt.ticklabel_format(style='plain')

plt.legend(fontsize="14", loc ="upper left")
plt.grid(True)

plt.savefig(save_plots_to + f"plot_6_model_5_run_1.png", dpi='figure')
#plt.show()



# plot test 7
plt.figure(figsize=(16, 10))

plt.plot(data_test['Timesteps'], data_test['No_type_2_poly_bound_to_prot'], marker='.', alpha=0.7, label='Model 5')

plt.xlabel('Timesteps', fontsize ='16')
plt.xticks(fontsize='14')

plt.ylabel('Number of type 2 polymers bound to proteins', fontsize ='16')
plt.yticks(fontsize='14')

plt.title('Number of type 2 polymers bound to proteins vs Timesteps', fontsize ='16')
plt.ticklabel_format(style='plain')

plt.legend(fontsize="14", loc ="upper left")
plt.grid(True)

plt.savefig(save_plots_to + f"plot_7_model_5_run_1.png", dpi='figure')
#plt.show()



# plot test 8
plt.figure(figsize=(16, 10))

plt.plot(data_test['Timesteps'], data_test['Mean_no_type_2_in_cluster'], marker='.', alpha=0.7, label='Model 5')

plt.xlabel('Timesteps', fontsize ='16')
plt.xticks(fontsize='14')

plt.ylabel('Mean number of type 2 polymers per protein cluster', fontsize ='16')
plt.yticks(fontsize='14')

plt.title('Mean number of type 2 polymers per protein cluster vs Timesteps', fontsize ='16')
plt.ticklabel_format(style='plain')

plt.legend(fontsize="14", loc ="upper left")
plt.grid(True)

plt.savefig(save_plots_to + f"plot_8_model_5_run_1.png", dpi='figure')
#plt.show()


"""