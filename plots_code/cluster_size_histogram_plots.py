### Using trimmed cluster_size outfiles to calculate histogram plots

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter

# specify paths to read data from + save plots to
local_or_cplab = input("Local or cplab: ")

if local_or_cplab == "local":
    path_to_trimmed_outfiles = '/home/elenaespinosa/OneDrive/Uni/Summer_courses/Summer_project/mod_lammps_code/outfiles/trimmed_outfiles/'
    save_plots_to = '/home/elenaespinosa/OneDrive/Uni/Summer_courses/Summer_project/mod_lammps_code/plots/hists/'
else:
    path_to_trimmed_outfiles = '/home/s2205640/Documents/summer_project/mod_lammps_code/outfiles/trimmed_outfiles/'
    save_plots_to = '/home/s2205640/Documents/summer_project/mod_lammps_code/plots/hists/'


def lines_in_file(filename):
    """ Get the number of frames from lines in the file """

    with open(path_to_trimmed_outfiles+filename) as f:
        for i, l in enumerate(f):
            pass

    return i # don't add 1 as first line is the header 


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


def get_counts_and_sizes(cluster_size_counter, no_of_frames):
    
    # extract cluster sizes and number of counts from cluster_size_counter
    cluster_sizes = list(cluster_size_counter.keys())
    counts = list(cluster_size_counter.values())

    sizes_list_step_1 = (np.arange(0, max(cluster_sizes)+1, 1)).tolist() # +1 as np.arange doesn't include endpoint

    # gets value for key if in cluster_size_counter (ie. gets counts), if not it defaults that count to 0
    counts_list = [cluster_size_counter.get(size, 0) for size in sizes_list_step_1]

    return sizes_list_step_1, (np.array(counts_list)/no_of_frames).tolist() # (np.array(counts_list)/no_of_frames).tolist()


def plot_histogram_models_123(cs_list_step_1, counts_list, model, color):

    plt.figure(figsize=(16, 10))
    
    plt.bar(cs_list_step_1, counts_list, color=color)

    plt.xlabel('Cluster sizes')
    plt.ylabel('Counts')
    plt.title(f'Distribution of cluster sizes for Model {model} - fixed')
    plt.xticks((np.arange(0, max(cs_list_step_1)+1, 5)).tolist())  # ensure each cluster size is a tick on the x-axis
    plt.grid(True, alpha=0.5)

    plt.savefig(save_plots_to + f"cs_hist_plot_model_{model}_SS_run_1_v2.png", dpi='figure')
    plt.show()

"""
def plot_histogram_model_4(cs_list_step_1, counts_list, model, color):

    plt.figure(figsize=(16, 10))
    
    plt.bar(cs_list_step_1, counts_list, color=color)

    plt.xlabel('Cluster sizes')
    plt.ylabel('Counts')
    plt.title(f'Distribution of cluster sizes for Model 4 - protein attraction strength: {model} kBT')
    plt.xticks((np.arange(0, max(cs_list_step_1)+1, 5)).tolist())  # Ensure each cluster size is a tick on the x-axis
    plt.grid(True, alpha=0.5)

    #plt.savefig(save_plots_to + f"cs_hist_plot_model_4_{model}_SS_run_1.png", dpi='figure')
    plt.show()


def plot_histogram_model_4_control(cs_list_step_1, counts_list, model, color):

    plt.figure(figsize=(16, 10))
    
    plt.bar(cs_list_step_1, counts_list, color=color)

    plt.xlabel('Cluster sizes')
    plt.ylabel('Counts')
    plt.title(f'Distribution of cluster sizes for Model 4 (control) - protein attraction strength: {model} kBT')
    plt.xticks((np.arange(0, max(cs_list_step_1)+1, 5)).tolist())  # Ensure each cluster size is a tick on the x-axis
    plt.grid(True, alpha=0.5)

    #plt.savefig(save_plots_to + f"cs_hist_plot_model_4_{model}_SS_run_1_control.png", dpi='figure')
    plt.show()
"""

def subplots_hist_model_4(cs_list_step_1, counts_list, cs_list_step_1_control, counts_list_control, model, color):

    fig, axs = plt.subplots(1, 2, figsize=(16, 10))

    fig.supxlabel('Cluster sizes', fontsize ='16')
    fig.supylabel('Counts', fontsize ='16')
    fig.suptitle(f'Distribution of cluster sizes for Model 4 (left) and control (right) - protein attraction strength: {model} kBT', fontsize='16')
    
    left_bar_1 = axs[0].bar(cs_list_step_1, counts_list, color=color)

    axs[0].set_xticks((np.arange(0, max(cs_list_step_1)+1, 5)).tolist())  # Ensure each cluster size is a tick on the x-axis

    axs[0].tick_params(labelsize=14)
    axs[0].grid(True, alpha=0.5)

    right_bar_1 = axs[1].bar(cs_list_step_1_control, counts_list_control, color=color)

    axs[1].set_xticks((np.arange(0, max(cs_list_step_1_control)+1, 10)).tolist())  # Ensure each cluster size is a tick on the x-axis

    axs[1].tick_params(labelsize=14)
    axs[1].grid(True, alpha=0.5)
    
    fig.tight_layout(pad=1.08)

    plt.savefig(save_plots_to + f"cs_hist_subplot_model_4_{model}_SS_run_1.png", dpi='figure')
    plt.show()


prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

"""
###
#   First investigation - Models 1, 2 + 3 histrogram plots of steady state cluster sizes
###

trimmed_outfiles_cs_list_123 = [f'trimmed_outfile_cs_{i}_run_1.dat' for i in range(1, 4)]

for i in range(1, 4):
    model_i_cs_counter, no_of_frames_123 = count_cluster_sizes(trimmed_outfiles_cs_list_123[i-1])
    model_i_cs_list_step_1, model_i_counts_list = get_counts_and_sizes(model_i_cs_counter, no_of_frames_123)
    
    plot_histogram_models_123(model_i_cs_list_step_1, model_i_counts_list, i, colors[i-1])



###
#   Second investigation - Model 4 histrogram plots of steady state cluster sizes + control
###

trimmed_outfiles_cs_list_4 = [f'trimmed_outfile_cs_4_var_{i}_run_1.dat' for i in range(1, 9)]
trimmed_outfiles_cs_list_4_control = [f'trimmed_outfile_cs_4_var_{i}_run_1_control.dat' for i in range(1, 9)]

for i in range(1, 9):
    model_i_cs_counter, no_of_frames_4 = count_cluster_sizes(trimmed_outfiles_cs_list_4[i-1])
    model_i_cs_counter_control, no_of_frames_4_control = count_cluster_sizes(trimmed_outfiles_cs_list_4_control[i-1])

    model_i_cs_list_step_1, model_i_counts_list = get_counts_and_sizes(model_i_cs_counter, no_of_frames_4)
    model_i_cs_list_step_1_control, model_i_counts_list_control = get_counts_and_sizes(model_i_cs_counter_control, no_of_frames_4_control)

    subplots_hist_model_4(model_i_cs_list_step_1, model_i_counts_list, model_i_cs_list_step_1_control, model_i_counts_list_control, i, colors[i-1])



###
#   Third investigation - Models 5, 6 + 7 histrogram plots of steady state cluster sizes
###

trimmed_outfiles_cs_list_567 = [f'trimmed_outfile_cs_{i}_run_1.dat' for i in range(5, 8)]
print(trimmed_outfiles_cs_list_567)

for i in range(1, 4):
    model_i_cs_counter, no_of_frames_567 = count_cluster_sizes(trimmed_outfiles_cs_list_567[i-1])
    model_i_cs_list_step_1, model_i_counts_list = get_counts_and_sizes(model_i_cs_counter, no_of_frames_567)
    
    plot_histogram_models_123(model_i_cs_list_step_1, model_i_counts_list, (i+4), colors[i-1])

### for proteins being type 4 + 5

trimmed_outfiles_cs_list_567_5 = [f'trimmed_outfile_cs_{i}_run_1_5.dat' for i in range(5, 8)]

# parse data
for i in range(1, 4):
    model_i_cs_counter, no_of_frames_567_5 = count_cluster_sizes(trimmed_outfiles_cs_list_567_5[i-1])
    model_i_cs_list_step_1, model_i_counts_list = get_counts_and_sizes(model_i_cs_counter, no_of_frames_567_5)
    
    plot_histogram_models_123(model_i_cs_list_step_1, model_i_counts_list, (i+4), colors[i-1])
"""

###
#   Third investigation - Models 5, 6, 7 + 8 histrogram plots of steady state cluster sizes
###

trimmed_outfiles_cs_list_5678_v2 = [f'trimmed_outfile_cs_{i}_run_1_v2.dat' for i in range(5, 9)]

# parse data
for i in range(1, 5):
    model_i_cs_counter, no_of_frames_5678_v2 = count_cluster_sizes(trimmed_outfiles_cs_list_5678_v2[i-1])
    model_i_cs_list_step_1, model_i_counts_list = get_counts_and_sizes(model_i_cs_counter, no_of_frames_5678_v2)
    
    plot_histogram_models_123(model_i_cs_list_step_1, model_i_counts_list, (i+4), colors[i-1])