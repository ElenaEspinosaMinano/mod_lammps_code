### condensed steady state plots of trimmed outfiles for summer project report - v3 cluster definition (cluster >= 2 type 4 within threshold 2.3)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# specify paths to read data from + save plots to
local_or_cplab = input("Local or cplab: ")

if local_or_cplab == "local":
    path_to_outfiles = '/home/elenaespinosa/OneDrive/Uni/Summer_courses/Summer_project/mod_lammps_code/outfiles_v3/'
    path_to_trimmed_outfiles = '/home/elenaespinosa/OneDrive/Uni/Summer_courses/Summer_project/mod_lammps_code/outfiles_v3/trimmed_outfiles/'
    save_plots_to_report = '/home/elenaespinosa/OneDrive/Uni/Summer_courses/Summer_project/mod_lammps_code/plots_v3/report/'
    save_plots_to_SS = '/home/elenaespinosa/OneDrive/Uni/Summer_courses/Summer_project/mod_lammps_code/plots_v3/steady_state/'


else:
    path_to_outfiles = '/home/s2205640/Documents/summer_project/mod_lammps_code/outfiles_v3/'
    path_to_trimmed_outfiles = '/home/s2205640/Documents/summer_project/mod_lammps_code/outfiles_v3/trimmed_outfiles/'
    save_plots_to_report = '/home/s2205640/Documents/summer_project/mod_lammps_code/plots_v3/report/'
    save_plots_to_SS = '/home/s2205640/Documents/summer_project/mod_lammps_code/plots_v3/steady_state/'


### -----------------------------------------------------------------------------------------------------------------------------

###
#   Functions - calc_stats, get_stats_12_05678, lines_in_file, count_cluster_sizes, get_counts_and_sizes
###

### -----------------------------------------------------------------------------------------------------------------------------

def calc_stats(data_frame):

    mean = data_frame.mean()
    std = data_frame.std() # sample standard deviation, normalized by 1 / sqrt(N-1)
    sem = std / np.sqrt(data_frame.count()) # data_frame.count() = N = 1401

    return mean, std, sem


def get_stats_12_05678(data_frames, column):
    mean_list = []
    std_list = []
    sem_list = []
    for i in range(1, 8):
        mean, std, sem = calc_stats(data_frames[i][column])
        mean_list.append(mean)
        std_list.append(std)
        sem_list.append(sem)
    return mean_list, std_list, sem_list



def lines_in_file(filename):
    """ Get the number of frames from lines in the file """

    with open(path_to_trimmed_outfiles+filename) as f:
        for i, l in enumerate(f):
            pass

    return i # don't add 1 as first line is the header 


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


def plot_histogram_models(cs_list_step_1, counts_list, model, color):

    fig, ax = plt.subplots(figsize=(16, 10))
    
    ax.bar(cs_list_step_1, counts_list, color=color)

    ax.set_xlabel('Cluster sizes', fontsize ='16')
    ax.set_ylabel('Counts', fontsize ='16')
    ax.set_title(f'Distribution of cluster sizes for Model {model} - v3', fontsize ='16')

    ax.set_xticks((np.arange(0, max(cs_list_step_1)+1, 5)).tolist())  # ensure each cluster size is a tick on the x-axis
    ax.grid(True, alpha=0.5)
    ax.tick_params('both', labelsize=14)
    
    # creates inset axes, specifies location and distance from borders
    ax_inset = inset_axes(plt.gca(), width="40%", height="40%", loc='upper right', borderpad=2)
    
    # gets data for first 15 cluster sizes and counts - has to be the same length
    inset_cs_list_step_1 = cs_list_step_1[:15]
    inset_counts_list = counts_list[:15]
    
    ax_inset.bar(inset_cs_list_step_1, inset_counts_list, color=color)
    
    ax_inset.set_yscale('log') # sets y-axis scale to log

    # set the xy limits
    ax_inset.set_xlim(1, 15)
    ax_inset.set_ylim(0.2, 3)

    # sets the xy ticks
    ax_inset.set_xticks(np.arange(1, 16, 1))
    ax_inset.set_yticks(np.arange(1, 4, 1))
    
    ax_inset.get_yaxis().set_major_formatter(plt.ScalarFormatter()) # formats the y-axis scale
    
    ax_inset.set_xlabel('Cluster sizes')
    ax_inset.set_ylabel('Counts (Log)')
    ax_inset.grid(True, alpha=0.5)
    
    fig.canvas.draw() # draws everything before inset
    ax.indicate_inset_zoom(ax_inset, edgecolor="red") # indicates the inset zoom
    
    plt.savefig(save_plots_to_hists + f"cs_hist_plot_model_{model}_SS_run_1_v3.png", dpi='figure')
    plt.show()


def plot_histogram_models_0123(cs_list_step_1, counts_list, model, color):

    plt.figure(figsize=(16, 10))
    
    plt.bar(cs_list_step_1, counts_list, color=color)

    plt.xlabel('Cluster sizes', fontsize ='16')
    plt.ylabel('Counts', fontsize ='16')
    plt.title(f'Distribution of cluster sizes for Model {model} - v3', fontsize ='16')

    plt.xticks((np.arange(0, max(cs_list_step_1)+1, 5)).tolist())  # ensure each cluster size is a tick on the x-axis
    plt.grid(True, alpha=0.5)

    plt.tick_params('both', labelsize=14)
    plt.savefig(save_plots_to_hists + f"cs_hist_plot_model_{model}_SS_run_1_v3.png", dpi='figure')
    plt.show()


def hist_plot_model_i_single_timestep(cs_list_step_1, counts_list, model, color):

    plt.figure(figsize=(16, 10))
    
    plt.bar(cs_list_step_1, counts_list, color=color)

    plt.xlabel('Cluster sizes', fontsize ='16')
    plt.ylabel('Counts', fontsize ='16')
    plt.title(f'Distribution of cluster sizes for Model {model} at a single timestep - v3', fontsize ='16')
    
    plt.xticks((np.arange(0, max(cs_list_step_1)+1, 5)).tolist())  # ensure each cluster size is a tick on the x-axis
    plt.grid(True, alpha=0.5)
    
    plt.tick_params('both', labelsize=14)
    plt.savefig(save_plots_to_hists + f"cs_hist_plot_model_{model}_SS_timestep_run_1_v3.png", dpi='figure')
    plt.show()


prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
colors_list = colors[:2]+['black']+colors[:4]

column_name = ['Number of clusters', 'Mean size of clusters (in proteins)', 'Size of largest cluster (in proteins)', 'Number of clusters of size 1', 'Number of proteins bound to polymer', 
                'Fraction of clusters bound to polymer', 'Number of type 2 polymers bound to proteins', 'Mean number of type 2 polymers per protein cluster']
data_frame_name = ['No_of_clusters', 'Mean_size_of_clusters', 
                   'Size_of_largest_cluster', 'No_of_clusters_of_size_1', 'No_proteins_bound_to_poly', 
                   'Fraction_clusters_bound_to_poly', 'No_type_2_poly_bound_to_prot', 'Mean_no_type_2_in_cluster']

models_12_05678 = (1, 2, 0, 5, 6, "7A", "7B")
x_pos_12_05678 = np.arange(len(models_12_05678))


### -----------------------------------------------------------------------------------------------------------------------------

###
#   First investigation - Model 1, 2 + 0, 5, 6, 7, 8 plots - proteins being type 4, clusters defined as >= 2 protein + ct 2.3
###

### -----------------------------------------------------------------------------------------------------------------------------

models_1234_05678 = (1, 2, 3, 4, 0, 5, 6, "7A", "7B")
x_pos_1234_05678 = np.arange(len(models_1234_05678))
colors_list_2 = colors[:4]+['black']+colors[:4]

outfiles_list_1234_05678 = [f'outfile_{i}_run_1_v3.dat' for i in range(1, 4)] + ['outfile_4_var_2_run_1_v3.dat','outfile_0_run_1_v3.dat'] + [f'outfile_{i}_run_1_v3.dat' for i in range(5, 9)]

# dictionary to store data frames
data_frames_1234_05678 = {}

# parse data
for i in range(1, 10):
    data_frames_1234_05678[i] = pd.read_csv(path_to_outfiles + outfiles_list_1234_05678[i-1], sep=' ', comment='#', header=None)
    data_frames_1234_05678[i].columns = ['Timesteps', 'No_of_clusters', 'Mean_size_of_clusters', 
                                        'Size_of_largest_cluster', 'No_of_clusters_of_size_1', 'No_proteins_bound_to_poly', 
                                        'Fraction_clusters_bound_to_poly', 'No_type_2_poly_bound_to_prot', 'Mean_no_type_2_in_cluster']


def full_plots_1234_05678(column_name, column_no):

    fig, axs = plt.subplots(2, figsize=(16, 10), tight_layout=True)

    fig.supxlabel('Timesteps', fontsize ='16')
    fig.supylabel(column_name, fontsize ='16')
    #fig.suptitle('Number of clusters vs Timesteps for Models 1-4 (top) and Models 0, 5-7 (bottom)', fontsize='16')

    for i in range(1, 5):
        axs[0].plot(data_frames_1234_05678[i]['Timesteps'], data_frames_1234_05678[i]['No_of_clusters'], marker='.', alpha=0.7, label=f'Model {models_1234_05678[i-1]}')

    axs[0].axvline(1000000, linestyle='--', alpha=0.7, label='Steady state timestep', color='black')
    axs[0].tick_params(labelsize=14)
    axs[0].ticklabel_format(style='plain')
    axs[0].legend(fontsize=14)
    axs[0].grid(True, alpha=0.5)


    for j in range(5, 10):
        axs[1].plot(data_frames_1234_05678[j]['Timesteps'], data_frames_1234_05678[j]['No_of_clusters'], marker='.', alpha=0.7, label=f'Model {models_1234_05678[j-1]}', color=colors_list_2[j-1])

    axs[1].axvline(3000000, linestyle='--', alpha=0.7, label='Steady state timestep', color='black')
    axs[1].tick_params(labelsize=14)
    axs[1].ticklabel_format(style='plain')
    axs[1].legend(fontsize=14)
    axs[1].grid(True, alpha=0.5)

    plt.savefig(save_plots_to_report + f"full_outfile_NoC_plot_models_1234_05678_v3.png", dpi='figure')

    plt.show()


# full outfile plots for models 1234 and 05678
#full_plots_1234_05678(column_name[0], 1)

### -----------------------------------------------------------------------------------------------------------------------------


trimmed_outfiles_list_12_05678 = [f'trimmed_outfile_{i}_run_1_v3.dat' for i in range(1, 3)] + ['trimmed_outfile_0_run_1_v3.dat'] + [f'trimmed_outfile_{i}_run_1_v3.dat' for i in range(5, 9)]

# dictionary to store data frames
data_frames_12_05678_trimmed = {}

# parse data
for i in range(1, 8):
    data_frames_12_05678_trimmed[i] = pd.read_csv(path_to_trimmed_outfiles + trimmed_outfiles_list_12_05678[i-1], sep=' ', comment='#', header=None)
    data_frames_12_05678_trimmed[i].columns = ['Timesteps', 'No_of_clusters', 'Mean_size_of_clusters', 
                                        'Size_of_largest_cluster', 'No_of_clusters_of_size_1', 'No_proteins_bound_to_poly', 
                                        'Fraction_clusters_bound_to_poly', 'No_type_2_poly_bound_to_prot', 'Mean_no_type_2_in_cluster']


def plots_models_12_05678(mean_123, std_123, sem_123, column_name, column_no):
    bar_labels_mean = [f'Model {model}: {mean_123[i]:.2f} ± {sem_123[i]:.2f} (1 SEM)' for i, model in enumerate(models_12_05678)]
    bar_labels_std = [f'Model {model}: {std_123[i]:.2f} ± ---' for i, model in enumerate(models_12_05678)]
    model_labels = [f'Model {model}' for model in (models_12_05678)]

    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(16, 10), tight_layout=True)

    fig.supxlabel('Models', fontsize ='16')
    fig.supylabel(column_name, fontsize ='16')
    fig.suptitle(f'{column_name} for different models - Mean ± 1 SEM (left) and STD (right) - v3', fontsize='16')

    left_bar = axs[0].bar(model_labels, mean_123, yerr=sem_123, capsize=2, label=bar_labels_mean, color=colors[:4])

    axs[0].set_xticks(x_pos_12_05678)
    axs[0].tick_params(labelsize=14)
    axs[0].legend(fontsize=14)
    axs[0].grid(True, alpha=0.5)


    right_bar = axs[1].bar(model_labels, std_123, label=bar_labels_std, color=colors[:4])

    axs[1].set_xticks(x_pos_12_05678)
    axs[1].tick_params(labelsize=14)
    axs[1].legend(fontsize=14)
    axs[1].grid(True, alpha=0.5)

    #plt.savefig(save_plots_to_report + f"plot_{column_no}_model_1234_SS_run_1_v3.png", dpi='figure')
    plt.show()


# Function to create a single subplot
def plot_mean_with_sem(ax, means, sems, column_name):
    model_labels = [f'Model {model}' for model in models_12_05678]
    
    bars = ax.bar(model_labels, means, yerr=sems, capsize=2, color=colors_list)
    ax.set_xticks(x_pos_12_05678)
    ax.tick_params(labelsize=14)
    ax.grid(True, alpha=0.5)
    ax.set_ylabel(column_name, fontsize=16)

    # Adding legend entries for each bar
    for bar, label, mean, sem in zip(bars, model_labels, means, sems):
        bar.set_label(f'{label}: {mean:.2f} ± {sem:.2f}')
    
    ax.legend(fontsize=10)
"""
# Create a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(16, 12), tight_layout=True)

# Flatten the axs array for easy iteration
axs = axs.flatten()

# Specify the columns to plot
columns_to_plot = data_frame_name[:4]

# Iterate over the specified columns and plot
for i, column in enumerate(columns_to_plot):
    means, stds, sems = get_stats_12_05678(data_frames_12_05678_trimmed, column)
    plot_mean_with_sem(axs[i], means, sems, column_name[i])

# Set common labels
#fig.supxlabel('Models', fontsize=16)
#fig.supylabel('Values', fontsize=16)
#fig.suptitle('Mean ± 1 SEM for Different Models (Selected Columns)', fontsize=20)


# Show the plot
plt.savefig(save_plots_to_report + f"SS_plots_1_to_4_models_12_05678_v3.png", dpi='figure')
plt.show()



# Create a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(16, 12), tight_layout=True)

# Flatten the axs array for easy iteration
axs = axs.flatten()

# Specify the columns to plot
columns_to_plot = data_frame_name[4:8]

# Iterate over the specified columns and plot
for i, column in enumerate(columns_to_plot):
    means, stds, sems = get_stats_12_05678(data_frames_12_05678_trimmed, column)
    plot_mean_with_sem(axs[i], means, sems, column_name[i+4])


# Show the plot
plt.savefig(save_plots_to_report + f"SS_plots_5_to_8_models_12_05678_v3.png", dpi='figure')
plt.show()
"""

### -----------------------------------------------------------------------------------------------------------------------------

models_678 = (1, 2, 3)
x_pos_678 = np.arange(len(models_678))
colors_list_678 = colors[1:4] # this will show colors orange, green and red

trimmed_outfiles_list_678 = ['trimmed_outfile_6_run_1_v3.dat'] + ['trimmed_outfile_8_run_1_v3.dat'] +['trimmed_outfile_7_run_1_v3.dat'] #[f'trimmed_outfile_{i}_run_1_v3.dat' for i in range(6, 9)]

def get_stats_678(data_frames, column):
    mean_list = []
    std_list = []
    sem_list = []
    for i in range(1, 4):
        mean, std, sem = calc_stats(data_frames[i][column])
        mean_list.append(mean)
        std_list.append(std)
        sem_list.append(sem)
    return mean_list, std_list, sem_list

# dictionary to store data frames
data_frames_678_trimmed = {}

# parse data
for i in range(1, 4):
    data_frames_678_trimmed[i] = pd.read_csv(path_to_trimmed_outfiles + trimmed_outfiles_list_678[i-1], sep=' ', comment='#', header=None)
    data_frames_678_trimmed[i].columns = ['Timesteps', 'No_of_clusters', 'Mean_size_of_clusters', 
                                        'Size_of_largest_cluster', 'No_of_clusters_of_size_1', 'No_proteins_bound_to_poly', 
                                        'Fraction_clusters_bound_to_poly', 'No_type_2_poly_bound_to_prot', 'Mean_no_type_2_in_cluster']

# Function to create a single subplot
def plot_mean_with_sem(ax, means, sems, column_name):
    model_labels = [f'Model {model}' for model in models_678]
    
    bars = ax.bar(model_labels, means, yerr=sems, capsize=2, color=colors_list_678)
    ax.set_xticks(x_pos_678)
    ax.tick_params(labelsize=14)
    ax.grid(True, alpha=0.5)
    ax.set_ylabel(column_name, fontsize=16)

    # Adding legend entries for each bar
    for bar, label, mean, sem in zip(bars, model_labels, means, sems):
        bar.set_label(f'{label}: {mean:.2f} ± {sem:.2f}')
    
    ax.legend(fontsize=10)

# Create a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(16, 12), tight_layout=True)

# Flatten the axs array for easy iteration
axs = axs.flatten()

# Specify the columns to plot
columns_to_plot = data_frame_name[:4]

# Iterate over the specified columns and plot
for i, column in enumerate(columns_to_plot):
    means, stds, sems = get_stats_678(data_frames_678_trimmed, column)
    plot_mean_with_sem(axs[i], means, sems, column_name[i])

# Set common labels
#fig.supxlabel('Models', fontsize=16)
#fig.supylabel('Values', fontsize=16)
#fig.suptitle('Mean ± 1 SEM for Different Models (Selected Columns)', fontsize=20)


# Show the plot
plt.savefig(save_plots_to_report + f"SS_plots_1_to_4_models_678_v3_mod.png", dpi='figure')
plt.show()



# Create a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(16, 12), tight_layout=True)

# Flatten the axs array for easy iteration
axs = axs.flatten()

# Specify the columns to plot
columns_to_plot = data_frame_name[4:8]

# Iterate over the specified columns and plot
for i, column in enumerate(columns_to_plot):
    means, stds, sems = get_stats_678(data_frames_678_trimmed, column)
    plot_mean_with_sem(axs[i], means, sems, column_name[i+4])


# Show the plot
plt.savefig(save_plots_to_report + f"SS_plots_5_to_8_models_678_v3_mod.png", dpi='figure')
plt.show()

### -----------------------------------------------------------------------------------------------------------------------------

colors_list_3 = colors[:2]+colors[:4]
models_12_5678 = (1, 2, 5, 6, "7A", "7B")

def plot_histogram_models_12_5678(ax, cs_list_step_1, counts_list, model, color):
    
    hist = ax.bar(cs_list_step_1, counts_list, color=color, label=f'Model {model}')

    ax.set_xticks((np.arange(0, max(cs_list_step_1)+1, 5)).tolist())  # ensure each cluster size is a tick on the x-axis
    ax.grid(True, alpha=0.5)

    ax.tick_params('both', labelsize=12)
    ax.legend(fontsize=12)


trimmed_outfiles_cs_list_12_5678 = [f'trimmed_outfile_cs_{i}_run_1_v3.dat' for i in range(1, 3)] + [f'trimmed_outfile_cs_{i}_run_1_v3.dat' for i in range(5, 9)]

# cluster sizes histogram plots for models 1, 2, 5, 6, 7 + 8
"""
fig, axs = plt.subplots(3, 2, figsize=(16, 12), tight_layout=True)

axs = axs.flatten()

for j, trimmed_outfile_cs in enumerate(trimmed_outfiles_cs_list_12_5678):

    model_i_cs_counter, no_of_frames = count_cluster_sizes(trimmed_outfile_cs)
    model_i_cs_list_step_1, model_i_counts_list = get_counts_and_sizes(model_i_cs_counter, no_of_frames)

    plot_histogram_models_12_5678(axs[j], model_i_cs_list_step_1, model_i_counts_list, models_12_5678[j], colors_list_3[j])

fig.supxlabel('Cluster size distribution', fontsize=16)
fig.supylabel('Counts', fontsize=16)

plt.savefig(save_plots_to_report + f"hists.png", dpi='figure')
plt.show()
"""


### -----------------------------------------------------------------------------------------------------------------------------

###
#   Second investigation - Model 0, 5, 6, 7 + 8 plots - proteins being type 4, clusters defined as > 1 protein + ct 2.3
###

### -----------------------------------------------------------------------------------------------------------------------------

outfiles_list_05678_v3 = ['outfile_0_run_1_v3.dat'] + [f'outfile_{i}_run_1_v3.dat' for i in range(5, 9)]

# dictionary to store data frames
data_frames_05678_v3 = {}

# parse data
for i in range(1, 6):
    data_frames_05678_v3[i] = pd.read_csv(path_to_outfiles + outfiles_list_05678_v3[i-1], sep=' ', comment='#', header=None)
    data_frames_05678_v3[i].columns = ['Timesteps', 'No_of_clusters', 'Mean_size_of_clusters', 
                                        'Size_of_largest_cluster', 'No_of_clusters_of_size_1', 'No_proteins_bound_to_poly', 
                                        'Fraction_clusters_bound_to_poly', 'No_type_2_poly_bound_to_prot', 'Mean_no_type_2_in_cluster']


def full_plots_05678(data_frame_name, column_name, column_no):
    plt.figure(figsize=(16, 10))

    plt.plot(data_frames_05678_v3[1]['Timesteps'], data_frames_05678_v3[1][data_frame_name], marker='.', alpha=0.7, color='black', label=f'Model {models_05678[0]}')

    for i in range(2, 6):
        plt.plot(data_frames_05678_v3[i]['Timesteps'], data_frames_05678_v3[i][data_frame_name], marker='.', alpha=0.7, label=f'Model {models_05678[i-1]}')

    plt.xlabel('Timesteps', fontsize ='16')
    plt.xticks(fontsize='14')

    plt.ylabel(f'{column_name}', fontsize ='16')
    plt.yticks(fontsize='14')

    plt.title(f'{column_name} vs Timesteps - v3', fontsize ='16')
    plt.ticklabel_format(style='plain')

    plt.legend(fontsize="14")
    plt.grid(True)

    plt.savefig(save_plots_to + f"plot_{column_no}_model_05678_run_1_v3.png", dpi='figure')
    plt.show()

# full outfile plots for models 05678
"""
for i, column in enumerate(column_name):
    full_plots_05678(data_frame_name[i], column_name[i], (i+1))
"""


### -----------------------------------------------------------------------------------------------------------------------------

###
#   Third investigation - Model 7 with varying switching rate - proteins being type 4, clusters defined as > 1 protein + ct 2.3
###

### -----------------------------------------------------------------------------------------------------------------------------

trimmed_outfiles_list_7_v3 = [f'trimmed_outfile_7_sw_{i}00_run_1_v3.dat' for i in range(3, 8)]

# dictionary to store data frames
data_frames_7_trimmed = {}

# parse data
for i in range(1, 6):
    data_frames_7_trimmed[i] = pd.read_csv(path_to_trimmed_outfiles + trimmed_outfiles_list_7_v3[i-1], sep=' ', comment='#', header=None)
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


def plots_model_7_sw(mean_7_sw, std_7_sw, sem_7_sw, column_name, column_no):

    plt.figure(figsize=(16, 10))
    
    plt.errorbar(tau_sw, mean_7_sw, yerr=sem_7_sw, capsize=2, fmt='.-g', alpha=0.7, ecolor='black')

    plt.xlabel('Switching interval (time units)', fontsize ='16')
    plt.ylabel(f'{column_name}', fontsize ='16')
    #plt.title(f'{column_name} vs. switching interval (time units) - v3', fontsize ='16')
    
    plt.xticks(tau_sw)  # ensure each cluster size is a tick on the x-axis
    plt.grid(True, alpha=0.5)

    plt.tick_params('both', labelsize=14)
    plt.savefig(save_plots_to_SS + f"plot_{column_no}_model_sw_7_SS_run_1_v3.png", dpi='figure')
    plt.show()


# plots 1 to 8 for Model 7 with varying switching rate - Tau_sw
"""
for i, column in enumerate(column_name):
    mean_7_sw, std_7_sw, sem_7_sw = get_stats_7(data_frames_7_trimmed, data_frame_name[i])
    plots_model_7_sw(mean_7_sw, std_7_sw, sem_7_sw, column_name[i], (i+1))
"""


### -----------------------------------------------------------------------------------------------------------------------------

###
#   Fourth investigation - Models 5678 with varying no of proteins - proteins being type 4, clusters defined as > 1 protein + ct 2.3
###

### -----------------------------------------------------------------------------------------------------------------------------

trimmed_outfiles_list_5678_var_nop_v3 = [f'trimmed_outfile_{i}_var_{j}00_run_1_v3.dat' for i in range(5, 9) for j in range(3, 7)]

# dictionary to store data frames
data_frames_5678_var_nop_v3 = {}

# parse data
for i in range(1, 17):
    data_frames_5678_var_nop_v3[i] = pd.read_csv(path_to_trimmed_outfiles + trimmed_outfiles_list_5678_var_nop_v3[i-1], sep=' ', comment='#', header=None)
    data_frames_5678_var_nop_v3[i].columns = ['Timesteps', 'No_of_clusters', 'Mean_size_of_clusters', 
                                                'Size_of_largest_cluster', 'No_of_clusters_of_size_1', 'No_proteins_bound_to_poly', 
                                                'Fraction_clusters_bound_to_poly', 'No_type_2_poly_bound_to_prot', 'Mean_no_type_2_in_cluster']


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

    label_names = (5, 6, "7A", "7B")

    plt.figure(figsize=(16, 10))
    
    for i in range(0, 4):

        plt.errorbar(nop, means[i*4:(i+1)*4], yerr=sems[i*4:(i+1)*4], capsize=2, fmt='.-', alpha=0.7, ecolor='black', label=f'Model {label_names[i]}', color=colors[i])
    
    plt.xlabel('No of proteins', fontsize ='16')
    plt.ylabel(f'{column_name}', fontsize ='16')
    #plt.title(f'{column_name} vs. no of proteins - v3', fontsize ='16')
    
    plt.xticks(nop)  # ensure each cluster size is a tick on the x-axis
    plt.grid(True, alpha=0.5)

    plt.tick_params('both', labelsize=14)
    plt.legend(fontsize="14")
    plt.savefig(save_plots_to_SS + f"plot_{column_no}_models_5678_SS_var_nop_run_1_v3.png", dpi='figure')
    plt.show()


# plots 1 to 8 for models 5678 with varying no of proteins
"""
for i, column in enumerate(column_name):
    means, stds, sems = get_stats_5678_nop(data_frames_5678_var_nop_v3, data_frame_name[i])
    plots_models_5678_var_nop(means, stds, sems, column_name[i], (i+1))
"""

# ------------------------------------------------------------------------------------------------------------------------------------------


# Function to create a single subplot for increasing no of proteins
def subplot_models_5678_var_nop(ax, means, sems, column_name):

    label_names = (5, 6, "7A", "7B")

    ax.set_xticks(nop)
    ax.tick_params(labelsize=14)
    ax.grid(True, alpha=0.5)
    ax.set_ylabel(column_name, fontsize=16)

    for i in range(0, 4):

        ax.errorbar(nop, means[i*4:(i+1)*4], yerr=sems[i*4:(i+1)*4], capsize=2, fmt='.-', alpha=0.7, ecolor='black', label=f'Model {label_names[i]}', color=colors[i])
    
    ax.legend(fontsize=10)

"""
# Create a 2x2 grid of subplots
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(16, 12), tight_layout=True)

fig.supxlabel('Number of proteins', fontsize ='16')

# Flatten the axs array for easy iteration
axs = axs.flatten()

# Specify the columns to plot
columns_to_plot = data_frame_name[:4] + data_frame_name[6:8]
column_name_to_plot = column_name[:4] + column_name[6:8]

# Iterate over the specified columns and plot
for i, column in enumerate(columns_to_plot):
    means, stds, sems = get_stats_5678_nop(data_frames_5678_var_nop_v3, column)
    subplot_models_5678_var_nop(axs[i], means, sems, column_name_to_plot[i])

# Show the plot
plt.savefig(save_plots_to_report + f"condensed_plot_models_5678_SS_var_nop.png", dpi='figure')
plt.show()
"""

# ------------------------------------------------------------------------------------------------------------------------------------------


# Function to create a single subplot
def subplot_model_7_sw(ax, means, sems, column_name):

    ax.set_xticks(tau_sw)
    ax.tick_params(labelsize=14)
    ax.grid(True, alpha=0.5)
    ax.set_ylabel(column_name, fontsize=16)

    ax.errorbar(tau_sw, means, yerr=sems, capsize=2, fmt='.-g', alpha=0.7, ecolor='black')

"""
# Create a 2x2 grid of subplots
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(16, 12), tight_layout=True)

fig.supxlabel('Switching interval (time units)', fontsize ='16')

# Flatten the axs array for easy iteration
axs = axs.flatten()

# Specify the columns to plot
columns_to_plot = data_frame_name[:5] + data_frame_name[7:8]
column_name_to_plot = column_name[:5] + column_name[7:8]

# Iterate over the specified columns and plot
for i, column in enumerate(columns_to_plot):
    means, stds, sems = get_stats_7(data_frames_7_trimmed, column)
    subplot_model_7_sw(axs[i], means, sems, column_name_to_plot[i])

# Show the plot
plt.savefig(save_plots_to_report + f"condensed_plot_model_7_sw.png", dpi='figure')
plt.show()
"""
