### Using trimmed outfiles to calculate the mean, standard error on the mean and standard deviation of quantities - fixed :)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter

# specify paths to read data from + save plots to
local_or_cplab = input("Local or cplab: ")

if local_or_cplab == "local":
    path_to_outfiles = '/home/elenaespinosa/OneDrive/Uni/Summer_courses/Summer_project/mod_lammps_code/outfiles/'
    path_to_trimmed_outfiles = '/home/elenaespinosa/OneDrive/Uni/Summer_courses/Summer_project/mod_lammps_code/outfiles/trimmed_outfiles/'
    save_plots_to = '/home/elenaespinosa/OneDrive/Uni/Summer_courses/Summer_project/mod_lammps_code/plots/'
    save_plots_to_SS = '/home/elenaespinosa/OneDrive/Uni/Summer_courses/Summer_project/mod_lammps_code/plots/steady_state/'
    save_plots_to_hists = '/home/elenaespinosa/OneDrive/Uni/Summer_courses/Summer_project/mod_lammps_code/plots/hists/'

else:
    path_to_outfiles = '/home/s2205640/Documents/summer_project/mod_lammps_code/outfiles/'
    path_to_trimmed_outfiles = '/home/s2205640/Documents/summer_project/mod_lammps_code/outfiles/trimmed_outfiles/'
    save_plots_to = '/home/s2205640/Documents/summer_project/mod_lammps_code/plots/'
    save_plots_to_SS = '/home/s2205640/Documents/summer_project/mod_lammps_code/plots/steady_state/'
    save_plots_to_hists = '/home/s2205640/Documents/summer_project/mod_lammps_code/plots/hists/'


def calc_stats(data_frame):

    mean = data_frame.mean()
    std = data_frame.std() # sample standard deviation, normalized by 1 / sqrt(N-1)
    sem = std / np.sqrt(data_frame.count()) # data_frame.count() = N = 1401

    return mean, std, sem


# get stats of models 5678 for a particular dataframe and column + append it to a list
def get_stats_5678(data_frames, column):
    mean_list = []
    std_list = []
    sem_list = []
    for i in range(1, 5):
        mean, std, sem = calc_stats(data_frames[i][column])
        mean_list.append(mean)
        std_list.append(std)
        sem_list.append(sem)
    return mean_list, std_list, sem_list


def get_stats_05678(data_frames, column):
    mean_list = []
    std_list = []
    sem_list = []
    for i in range(1, 6):
        mean, std, sem = calc_stats(data_frames[i][column])
        mean_list.append(mean)
        std_list.append(std)
        sem_list.append(sem)
    return mean_list, std_list, sem_list



###
#   Third investigation - Model 5, 6, 7 + 8 plots - proteins being type 4+5 and clusters defined as > 1 protein
###



prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

models = ('Model 5', 'Model 6', 'Model 7', 'Model 8')
column_name = ['Number of clusters', 'Mean size of clusters', 'Size of largest cluster', 'Number of clusters of size 1', 'Number of proteins bound to polymer', 
                'Fraction of clusters bound to polymer', 'Number of type 2 polymers bound to proteins', 'Mean number of type 2 polymers per protein cluster']
data_frame_name = ['No_of_clusters', 'Mean_size_of_clusters', 
                   'Size_of_largest_cluster', 'No_of_clusters_of_size_1', 'No_proteins_bound_to_poly', 
                   'Fraction_clusters_bound_to_poly', 'No_type_2_poly_bound_to_prot', 'Mean_no_type_2_in_cluster']

x_pos = np.arange(len(models))


models_05678 = (0, 5, 6, 7, 8)
x_pos_05678 = np.arange(len(models_05678))



# full outfile plots for models 5678

outfiles_list_567_v2 = [f'outfile_{i}_run_3_v2.dat' for i in range(5, 9)]

# dictionary to store data frames
data_frames_567_v2 = {}

# parse data
for i in range(1, 5):
    data_frames_567_v2[i] = pd.read_csv(path_to_outfiles + outfiles_list_567_v2[i-1], sep=' ', comment='#', header=None)
    data_frames_567_v2[i].columns = ['Timesteps', 'No_of_clusters', 'Mean_size_of_clusters', 
                                        'Size_of_largest_cluster', 'No_of_clusters_of_size_1', 'No_proteins_bound_to_poly', 
                                        'Fraction_clusters_bound_to_poly', 'No_type_2_poly_bound_to_prot', 'Mean_no_type_2_in_cluster']


def full_plots_5678(data_frame_name, column_name, column_no):
    plt.figure(figsize=(16, 10))

    plt.plot(data_frames_567_v2[1]['Timesteps'], data_frames_567_v2[1][data_frame_name], marker='.', alpha=0.7, label='Model 5')
    plt.plot(data_frames_567_v2[2]['Timesteps'], data_frames_567_v2[2][data_frame_name], marker='.', alpha=0.7, label='Model 6')
    plt.plot(data_frames_567_v2[3]['Timesteps'], data_frames_567_v2[3][data_frame_name], marker='.', alpha=0.7, label='Model 7')
    plt.plot(data_frames_567_v2[4]['Timesteps'], data_frames_567_v2[4][data_frame_name], marker='.', alpha=0.7, label='Model 8')

    plt.xlabel('Timesteps', fontsize ='16')
    plt.xticks(fontsize='14')

    plt.ylabel(f'{column_name}', fontsize ='16')
    plt.yticks(fontsize='14')

    plt.title(f'{column_name} vs Timesteps - v2', fontsize ='16')
    plt.ticklabel_format(style='plain')

    plt.legend(fontsize="14")
    plt.grid(True)

    plt.savefig(save_plots_to + f"plot_{column_no}_model_5678_run_3_v2.png", dpi='figure')
    plt.show()

"""
for i, column in enumerate(column_name):
    full_plots_5678(data_frame_name[i], column_name[i], (i+1))
"""



# trimmed outfile SS plots

trimmed_outfiles_list_567_v2 = [f'trimmed_outfile_{i}_run_3_v2.dat' for i in range(5, 9)]

# dictionary to store data frames
data_frames_567_trimmed_v2 = {}

# parse data
for i in range(1, 5):
    data_frames_567_trimmed_v2[i] = pd.read_csv(path_to_trimmed_outfiles + trimmed_outfiles_list_567_v2[i-1], sep=' ', comment='#', header=None)
    data_frames_567_trimmed_v2[i].columns = ['Timesteps', 'No_of_clusters', 'Mean_size_of_clusters', 
                                        'Size_of_largest_cluster', 'No_of_clusters_of_size_1', 'No_proteins_bound_to_poly', 
                                        'Fraction_clusters_bound_to_poly', 'No_type_2_poly_bound_to_prot', 'Mean_no_type_2_in_cluster']


def SS_plots_models_5678(mean_567, std_567, sem_567, column_name, column_no):
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

    plt.savefig(save_plots_to_SS + f"plot_{column_no}_model_5678_SS_run_3_v2.png", dpi='figure')
    plt.show()


# steady state plots 1 to 8 for models 5, 6, 7 + 8 - Mean ± 1 SEM (left) and STD (right) - proteins being type 4 + 5
"""
for i, column in enumerate(column_name):
    mean_567, std_567, sem_567 = get_stats_5678(data_frames_567_trimmed_v2, data_frame_name[i])
    SS_plots_models_5678(mean_567, std_567, sem_567, column_name[i], (i+1))
"""



# trimmed outfiles hist plots

def lines_in_file(filename):
    """ Get the number of frames from lines in the file """

    with open(path_to_trimmed_outfiles+filename) as f:
        for i, l in enumerate(f):
            pass

    return i # don't add 1 as first line is the header 

"""
def count_cluster_sizes(cs_outfile_name):

    # initialise Counter object to keep track of cluster sizes in cs_outfile
    cluster_size_counter = Counter()

    no_of_frames = lines_in_file(cs_outfile_name)

    with open(f"{path_to_trimmed_outfiles}{cs_outfile_name}", "r") as file:
        for line in file:

            if line.strip() == "" or line.startswith("#"):  # skip empty lines and comments
                continue

            _, sizes_str = line.split(":") # extract cluster sizes part after the colon 
            sizes_str = sizes_str.strip()  # strip leading/trailing whitespace

            sizes_list = eval(sizes_str) # evaluate str representation of list into actual list obeject

            cluster_size_counter.update(sizes_list) # update counter with sizes list

    return cluster_size_counter, no_of_frames
"""

def count_cluster_sizes(cs_outfile_name, line_no=None):
    
    # initialise Counter object to keep track of cluster sizes in cs_outfile
    cluster_size_counter = Counter()

    with open(f"{path_to_trimmed_outfiles}{cs_outfile_name}", "r") as file:
        for i, line in enumerate(file):

            if line.strip() == "" or line.startswith("#"): # skip empty lines and comments
                continue

            if line_no is not None and i != line_no: # skip lines in file which are not line_no = i
                continue

            _, sizes_str = line.split(":") # extract cluster sizes part after the colon 
            sizes_str = sizes_str.strip() # strip leading/trailing whitespace

            sizes_list = eval(sizes_str) # evaluate str representation of list into actual list object

            cluster_size_counter.update(sizes_list)  # update counter with sizes list

            if line_no is not None and i == line_no:
                break

    no_of_frames = 1 if line_no is not None else lines_in_file(cs_outfile_name)

    return cluster_size_counter, no_of_frames


def get_counts_and_sizes(cluster_size_counter, no_of_frames):
    
    # extract cluster sizes and number of counts from cluster_size_counter
    cluster_sizes = list(cluster_size_counter.keys())
    counts = list(cluster_size_counter.values())

    sizes_list_step_1 = (np.arange(0, max(cluster_sizes)+1, 1)).tolist() # +1 as np.arange doesn't include endpoint

    # gets value for key if in cluster_size_counter (ie. gets counts), if not it defaults that count to 0
    counts_list = [cluster_size_counter.get(size, 0) for size in sizes_list_step_1]

    return sizes_list_step_1, (np.array(counts_list)/no_of_frames).tolist()


def plot_histogram_models_05678(cs_list_step_1, counts_list, model, color):

    plt.figure(figsize=(16, 10))
    
    plt.bar(cs_list_step_1, counts_list, color=color)

    plt.xlabel('Cluster sizes', fontsize ='16')
    plt.ylabel('Counts', fontsize ='16')
    plt.title(f'Distribution of cluster sizes for Model {model}', fontsize ='16')

    plt.xticks((np.arange(0, max(cs_list_step_1)+1, 5)).tolist())  # ensure each cluster size is a tick on the x-axis
    plt.grid(True, alpha=0.5)

    plt.tick_params('both', labelsize=14)
    plt.savefig(save_plots_to_hists + f"cs_hist_plot_model_{model}_SS_run_3_v2.png", dpi='figure')
    plt.show()


def plot_histogram_model_7(cs_list_step_1, counts_list, color):

    plt.figure(figsize=(16, 10))
    
    plt.bar(cs_list_step_1, counts_list, color=color)

    plt.xlabel('Cluster sizes', fontsize ='16')
    plt.ylabel('Counts', fontsize ='16')
    plt.title(f'Distribution of cluster sizes for Model 7 at a single timestep', fontsize ='16')
    
    plt.xticks((np.arange(0, max(cs_list_step_1)+1, 5)).tolist())  # ensure each cluster size is a tick on the x-axis
    plt.grid(True, alpha=0.5)
    
    plt.tick_params('both', labelsize=14)
    plt.savefig(save_plots_to_hists + f"cs_hist_plot_model_7_SS_timestep_run_3_v2.png", dpi='figure')
    plt.show()


trimmed_outfiles_cs_list_05678_v2 = ['trimmed_outfile_cs_0_run_1_v2.dat'] + [f'trimmed_outfile_cs_{i}_run_3_v2.dat' for i in range(5, 9)]
"""
# parse data
for i in range(1, 6):
    model_i_cs_counter, no_of_frames_05678_v2 = count_cluster_sizes(trimmed_outfiles_cs_list_05678_v2[i-1])
    model_i_cs_list_step_1, model_i_counts_list = get_counts_and_sizes(model_i_cs_counter, no_of_frames_05678_v2)
    
    plot_histogram_models_05678(model_i_cs_list_step_1, model_i_counts_list, models_05678[i-1], colors[i-1])
"""


# trimmed outfile hist plot model 7 single timestep
"""
line_no = 1401  # specify the line no in cs outfile you want to plot - no 1401 is the last time step can use this to check results in vmd

model_7_cs_counter, no_of_frames_7_v2 = count_cluster_sizes(trimmed_outfiles_cs_list_5678_v2[2], line_no)
model_7_cs_list_step_1, model_7_counts_list = get_counts_and_sizes(model_7_cs_counter, no_of_frames_7_v2)

plot_histogram_model_7(model_7_cs_list_step_1, model_7_counts_list, colors[2])
"""



# trimmed outfile plots for Model 7 with varying switching rate

trimmed_outfiles_list_7_v2 = [f'trimmed_outfile_7_sw_{i}00_run_3_v2.dat' for i in range(3, 8)]
trimmed_outfiles_list_cs_7_v2 = [f'trimmed_outfile_cs_7_sw_{i}00_run_3_v2.dat' for i in range(3, 8)]

# dictionary to store data frames
data_frames_7_trimmed = {}

# parse data
for i in range(1, 6):
    data_frames_7_trimmed[i] = pd.read_csv(path_to_trimmed_outfiles + trimmed_outfiles_list_7_v2[i-1], sep=' ', comment='#', header=None)
    data_frames_7_trimmed[i].columns = ['Timesteps', 'No_of_clusters', 'Mean_size_of_clusters', 
                                        'Size_of_largest_cluster', 'No_of_clusters_of_size_1', 'No_proteins_bound_to_poly', 
                                        'Fraction_clusters_bound_to_poly', 'No_type_2_poly_bound_to_prot', 'Mean_no_type_2_in_cluster']


def get_stats_7(data_frames, column):
    mean_list = []
    std_list = []
    sem_list = []
    for i in range(1, 6):
        mean, std, sem = calc_stats(data_frames[i][column])
        mean_list.append(mean)
        std_list.append(std)
        sem_list.append(sem)
    return mean_list, std_list, sem_list


tau_sw = [300, 400, 500, 600, 700]
k_sw = (1/(np.arange(300, 800, 100))).tolist()


def plots_model_7_sw(mean_7_sw, std_7_sw, sem_7_sw, column_name, column_no):
    """fig, axs = plt.subplots(1, 2, sharey=True, figsize=(16, 10), tight_layout=True)

    fig.supylabel(f'{column_name}', fontsize ='16')
    fig.suptitle(f'{column_name} vs. switching interval (time units) (left) and switch rate (1/time units) (right)', fontsize='16')

    left_bar_1 = axs[0].errorbar(tau_sw, mean_7_sw, yerr=sem_7_sw, capsize=2, fmt='.r', alpha=0.7, ecolor='black')

    axs[0].set_xlabel('Switching interval (time units)', fontsize ='16')
    axs[0].set_xticks(tau_sw)
    axs[0].tick_params(labelsize=14)
    axs[0].grid(True, alpha=0.5)


    right_bar_1 = axs[1].errorbar(k_sw, mean_7_sw, yerr=sem_7_sw, capsize=2, fmt='.b', alpha=0.7, ecolor='black')
    
    axs[1].set_xlabel('Switching rate (1/time units)', fontsize ='16')
    axs[1].set_xticks(k_sw)
    axs[1].tick_params(labelsize=14)
    axs[1].grid(True, alpha=0.5)

    plt.savefig(save_plots_to_SS + f"plot_{column_no}_model_sw_7_SS_run_3_v2.png", dpi='figure')
    plt.show()"""


    plt.figure(figsize=(16, 10))
    
    plt.errorbar(tau_sw, mean_7_sw, yerr=sem_7_sw, capsize=2, fmt='.r', alpha=0.7, ecolor='black')

    plt.xlabel('Switching interval (time units)', fontsize ='16')
    plt.ylabel(f'{column_name}', fontsize ='16')
    plt.title(f'{column_name} vs. switching interval (time units)', fontsize ='16')
    
    plt.xticks(tau_sw)  # ensure each cluster size is a tick on the x-axis
    plt.grid(True, alpha=0.5)

    plt.tick_params('both', labelsize=14)
    plt.savefig(save_plots_to_SS + f"plot_{column_no}_model_sw_7_SS_run_3_v2.png", dpi='figure')
    plt.show()


# plots 1 to 8 for Model 7 with varying switching rate - Tau_sw
"""
for i, column in enumerate(column_name):
    mean_7_sw, std_7_sw, sem_7_sw = get_stats_7(data_frames_7_trimmed, data_frame_name[i])
    plots_model_7_sw(mean_7_sw, std_7_sw, sem_7_sw, column_name[i], (i+1))
"""



# trimmed outfile SS plot model 0

trimmed_outfiles_list_05678_v2 = ['trimmed_outfile_0_run_1_v2.dat'] + [f'trimmed_outfile_{i}_run_3_v2.dat' for i in range(5, 9)]

# dictionary to store data frames
data_frames_05678_trimmed_v2 = {}

# parse data
for i in range(1, 6):
    data_frames_05678_trimmed_v2[i] = pd.read_csv(path_to_trimmed_outfiles + trimmed_outfiles_list_05678_v2[i-1], sep=' ', comment='#', header=None)
    data_frames_05678_trimmed_v2[i].columns = ['Timesteps', 'No_of_clusters', 'Mean_size_of_clusters', 
                                        'Size_of_largest_cluster', 'No_of_clusters_of_size_1', 'No_proteins_bound_to_poly', 
                                        'Fraction_clusters_bound_to_poly', 'No_type_2_poly_bound_to_prot', 'Mean_no_type_2_in_cluster']


def SS_plots_models_05678(mean_05678, std_05678, sem_05678, column_name, column_no):
    bar_labels_mean = [f'Model {model}: {mean_05678[i]:.2f} ± {sem_05678[i]:.2f} (1 SEM)' for i, model in enumerate(models_05678)]
    bar_labels_std = [f'Model {model}: {std_05678[i]:.2f} ± ---' for i, model in enumerate(models_05678)]
    model_labels = [f'Model {model}' for model in (models_05678)]

    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(16, 10), tight_layout=True)

    fig.supxlabel('Models', fontsize ='16')
    fig.supylabel(column_name, fontsize ='16')
    fig.suptitle(f'{column_name} for different models - Mean ± 1 SEM (left) and STD (right) - fixed', fontsize='16')

    left_bar = axs[0].bar(model_labels, mean_05678, yerr=sem_05678, capsize=2, label=bar_labels_mean, color=['k']+colors[:4])

    axs[0].set_xticks(x_pos_05678)
    axs[0].tick_params(labelsize=14)
    axs[0].legend(fontsize=14)
    axs[0].grid(True, alpha=0.5)


    right_bar = axs[1].bar(model_labels, std_05678, label=bar_labels_std, color=['k']+colors[:4])

    axs[1].set_xticks(x_pos_05678)
    axs[1].tick_params(labelsize=14)
    axs[1].legend(fontsize=14)
    axs[1].grid(True, alpha=0.5)

    plt.savefig(save_plots_to_SS + f"plot_{column_no}_model_05678_SS_run_3_v2.png", dpi='figure')
    plt.show()


# steady state plots 1 to 8 for models 0, 5, 6, 7 + 8 - Mean ± 1 SEM (left) and STD (right) - proteins being type 4 + 5
"""
for i, column in enumerate(column_name):
    mean_05678, std_05678, sem_05678 = get_stats_05678(data_frames_05678_trimmed_v2, data_frame_name[i])
    SS_plots_models_05678(mean_05678, std_05678, sem_05678, column_name[i], (i+1))
"""


# full outfile plots for models 5678 with varying no of proteins

outfiles_list_5678_var_nop_v2 = [f'outfile_{i}_var_{j}00_run_3_v2.dat' for i in range(5, 9) for j in range(3, 7)]

# dictionary to store data frames
data_frames_5678_var_nop_v2 = {}

# parse data
for i in range(1, 17):
    data_frames_5678_var_nop_v2[i] = pd.read_csv(path_to_outfiles + outfiles_list_5678_var_nop_v2[i-1], sep=' ', comment='#', header=None)
    data_frames_5678_var_nop_v2[i].columns = ['Timesteps', 'No_of_clusters', 'Mean_size_of_clusters', 
                                                'Size_of_largest_cluster', 'No_of_clusters_of_size_1', 'No_proteins_bound_to_poly', 
                                                'Fraction_clusters_bound_to_poly', 'No_type_2_poly_bound_to_prot', 'Mean_no_type_2_in_cluster']


def full_plots_5678_var_nop(data_frame_name, column_name, column_no):
    plt.figure(figsize=(16, 10))

    for i in range(1, 5):

        plt.plot(data_frames_5678_var_nop_v2[i*1]['Timesteps'], data_frames_5678_var_nop_v2[i*1][data_frame_name], marker='.', alpha=0.7, label=f'Model {i+4} - 300 proteins')
        plt.plot(data_frames_5678_var_nop_v2[i*2]['Timesteps'], data_frames_5678_var_nop_v2[i*2][data_frame_name], marker='.', alpha=0.7, label=f'Model {i+4} - 400 proteins')
        plt.plot(data_frames_5678_var_nop_v2[i*3]['Timesteps'], data_frames_5678_var_nop_v2[i*3][data_frame_name], marker='.', alpha=0.7, label=f'Model {i+4} - 500 proteins')
        plt.plot(data_frames_5678_var_nop_v2[i*4]['Timesteps'], data_frames_5678_var_nop_v2[i*4][data_frame_name], marker='.', alpha=0.7, label=f'Model {i+4} - 600 proteins')

    plt.xlabel('Timesteps', fontsize ='16')
    plt.xticks(fontsize='14')

    plt.ylabel(f'{column_name}', fontsize ='16')
    plt.yticks(fontsize='14')

    plt.title(f'{column_name} vs Timesteps - v2', fontsize ='16')
    plt.ticklabel_format(style='plain')

    plt.legend(fontsize="14")
    plt.grid(True)

    #plt.savefig(save_plots_to + f"plot_{column_no}_model_5678_run_3_v2.png", dpi='figure')
    plt.show()

"""
for i, column in enumerate(column_name):
    full_plots_5678_var_nop(data_frame_name[i], column_name[i], (i+1))
"""



# trimmed outfile plots for models 5678 with varying no of proteins

def get_stats_5678_nop(data_frames, column):
    mean_list = []
    std_list = []
    sem_list = []
    for i in range(1, 17):
        mean, std, sem = calc_stats(data_frames[i][column])
        mean_list.append(mean)
        std_list.append(std)
        sem_list.append(sem)
    return mean_list, std_list, sem_list


nop = [300, 400, 500, 600]


def plots_models_5678_var_nop(means, stds, sems, column_name, column_no):

    plt.figure(figsize=(16, 10))
    
    for i in range(0, 4):

        plt.errorbar(nop, means[i*4:(i+1)*4], yerr=sems[i*4:(i+1)*4], capsize=2, fmt='.-', alpha=0.7, ecolor='black', label=f'Model {i+5}', color=colors[i])
    
    plt.xlabel('No of proteins', fontsize ='16')
    plt.ylabel(f'{column_name}', fontsize ='16')
    plt.title(f'{column_name} vs. no of proteins', fontsize ='16')
    
    plt.xticks(nop)  # ensure each cluster size is a tick on the x-axis
    plt.grid(True, alpha=0.5)

    plt.tick_params('both', labelsize=14)
    plt.legend(fontsize="14")
    plt.savefig(save_plots_to_SS + f"plot_{column_no}_models_5678_SS_var_nop_run_3_v2.png", dpi='figure')
    plt.show()


# plots 1 to 8 for models 5678 with varying no of proteins
"""
for i, column in enumerate(column_name):
    means, stds, sems = get_stats_5678_nop(data_frames_5678_var_nop_v2, data_frame_name[i])
    plots_models_5678_var_nop(means, stds, sems, column_name[i], (i+1))
"""



# trimmed outfile SS plots for models 123 version 2

trimmed_outfiles_list_123 = [f'trimmed_outfile_{i}_run_1_v2.dat' for i in range(1, 4)]

# dictionary to store data frames
data_frames_123_trimmed = {}

# parse data
for i in range(1, 4):
    data_frames_123_trimmed[i] = pd.read_csv(path_to_trimmed_outfiles + trimmed_outfiles_list_123[i-1], sep=' ', comment='#', header=None)
    data_frames_123_trimmed[i].columns = ['Timesteps', 'No_of_clusters', 'Mean_size_of_clusters', 
                                        'Size_of_largest_cluster', 'No_of_clusters_of_size_1', 'No_proteins_bound_to_poly', 
                                        'Fraction_clusters_bound_to_poly', 'No_type_2_poly_bound_to_prot', 'Mean_no_type_2_in_cluster']

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


models_123 = ('Model 1', 'Model 2', 'Model 3')
x_pos_123 = np.arange(len(models_123))


def plots_models_123(mean_123, std_123, sem_123, column_name, column_no):
    bar_labels_mean = [f'{model}: {mean_123[i]:.2f} ± {sem_123[i]:.2f} (1 SEM)' for i, model in enumerate(models_123)]
    bar_labels_std = [f'{model}: {std_123[i]:.2f} ± ---' for i, model in enumerate(models_123)]

    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(16, 10), tight_layout=True)

    fig.supxlabel('Models', fontsize ='16')
    fig.supylabel(column_name, fontsize ='16')
    fig.suptitle(f'{column_name} for different models - Mean ± 1 SEM (left) and STD (right)', fontsize='16')

    left_bar = axs[0].bar(models_123, mean_123, yerr=sem_123, capsize=2, label=bar_labels_mean, color=colors[:4])

    axs[0].set_xticks(x_pos_123)
    axs[0].tick_params(labelsize=14)
    axs[0].legend(fontsize=14)
    axs[0].grid(True, alpha=0.5)


    right_bar = axs[1].bar(models_123, std_123, label=bar_labels_std, color=colors[:4])

    axs[1].set_xticks(x_pos_123)
    axs[1].tick_params(labelsize=14)
    axs[1].legend(fontsize=14)
    axs[1].grid(True, alpha=0.5)

    plt.savefig(save_plots_to_SS + f"plot_{column_no}_model_123_SS_run_1_v2.png", dpi='figure')
    plt.show()


# plots 1 to 8 for models 1, 2 and 3 - Mean ± 1 SEM (left) and STD (right)

for i, column in enumerate(column_name):
    mean_123_1, std_123_1, sem_123_1 = get_stats_123(data_frames_123_trimmed, data_frame_name[i])
    plots_models_123(mean_123_1, std_123_1, sem_123_1, column_name[i], (i+1))

