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


def count_cluster_sizes(cs_outfile_name):

    # initialise Counter object to keep track of cluster sizes in cs_outfile
    cluster_size_counter = Counter()

    with open(f"{path_to_trimmed_outfiles}{cs_outfile_name}", "r") as file:
        for line in file:

            if line.strip() == "" or line.startswith("#"):  # skip empty lines and comments
                continue

            _, sizes_str = line.split(":") # extract cluster sizes part after the colon 
            sizes_str = sizes_str.strip()  # strip leading/trailing whitespace

            sizes_list = eval(sizes_str) # evaluate str representation of list into actual list obeject

            cluster_size_counter.update(sizes_list) # update counter with sizes list
    
    return cluster_size_counter


def get_counts_and_sizes(cluster_size_counter):
    
    # extract cluster sizes and number of counts from cluster_size_counter
    cluster_sizes = list(cluster_size_counter.keys())
    counts = list(cluster_size_counter.values())

    sizes_list_step_1 = (np.arange(0, max(cluster_sizes)+1, 1)).tolist() # +1 as np.arange doesn't include endpoint

    # gets value for key if in cluster_size_counter (ie. gets counts), if not it defaults that count to 0
    counts_list = [cluster_size_counter.get(size, 0) for size in sizes_list_step_1]

    return sizes_list_step_1, counts_list


def plot_histogram_models_123(cs_list_step_1, counts_list, model, color):

    plt.figure(figsize=(16, 10))
    
    plt.bar(cs_list_step_1, counts_list, color=color)

    plt.xlabel('Cluster sizes')
    plt.ylabel('Counts')
    plt.title(f'Distribution of cluster sizes for Model {model}')
    plt.xticks((np.arange(0, max(cs_list_step_1)+1, 5)).tolist())  # Ensure each cluster size is a tick on the x-axis
    plt.grid(True, alpha=0.5)

    plt.savefig(save_plots_to + f"cs_hist_plot_model_{model}_SS_run_1.png", dpi='figure')
    plt.show()


def plot_histogram_model_4(cs_list_step_1, counts_list, model, color):

    plt.figure(figsize=(16, 10))
    
    plt.bar(cs_list_step_1, counts_list, color=color)

    plt.xlabel('Cluster sizes')
    plt.ylabel('Counts')
    plt.title(f'Distribution of cluster sizes for Model 4 - protein attraction strength: {model} kBT')
    plt.xticks((np.arange(0, max(cs_list_step_1)+1, 5)).tolist())  # Ensure each cluster size is a tick on the x-axis
    plt.grid(True, alpha=0.5)

    plt.savefig(save_plots_to + f"cs_hist_plot_model_4_{model}_SS_run_1.png", dpi='figure')
    plt.show()


def plot_histogram_model_4_control(cs_list_step_1, counts_list, model, color):

    plt.figure(figsize=(16, 10))
    
    plt.bar(cs_list_step_1, counts_list, color=color)

    plt.xlabel('Cluster sizes')
    plt.ylabel('Counts')
    plt.title(f'Distribution of cluster sizes for Model 4 (control) - protein attraction strength: {model} kBT')
    plt.xticks((np.arange(0, max(cs_list_step_1)+1, 5)).tolist())  # Ensure each cluster size is a tick on the x-axis
    plt.grid(True, alpha=0.5)

    plt.savefig(save_plots_to + f"cs_hist_plot_model_4_{model}_SS_run_1_control.png", dpi='figure')
    plt.show()


prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']


###
#   First investigation - Models 1, 2 + 3 histrogram plots of steady state cluster sizes
###

trimmed_outfiles_cs_list_123 = [f'trimmed_outfile_cs_{i}_run_1.dat' for i in range(1, 4)]

for i in range(1, 4):
    model_i_cs_counter = count_cluster_sizes(trimmed_outfiles_cs_list_123[i-1])
    model_i_cs_list_step_1, model_i_counts_list = get_counts_and_sizes(model_i_cs_counter)
    
    plot_histogram_models_123(model_i_cs_list_step_1, model_i_counts_list, i, colors[i-1])



###
#   Second investigation - Model 4 histrogram plots of steady state cluster sizes + control
###

trimmed_outfiles_cs_list_4 = [f'trimmed_outfile_cs_4_var_{i}_run_1.dat' for i in range(1, 9)]

for i in range(1, 9):
    model_i_cs_counter = count_cluster_sizes(trimmed_outfiles_cs_list_4[i-1])
    model_i_cs_list_step_1, model_i_counts_list = get_counts_and_sizes(model_i_cs_counter)
    
    plot_histogram_model_4(model_i_cs_list_step_1, model_i_counts_list, i, colors[i-1])

trimmed_outfiles_cs_list_4_control = [f'trimmed_outfile_cs_4_var_{i}_run_1_control.dat' for i in range(1, 9)]

for i in range(1, 9):
    model_i_cs_counter = count_cluster_sizes(trimmed_outfiles_cs_list_4_control[i-1])
    model_i_cs_list_step_1, model_i_counts_list = get_counts_and_sizes(model_i_cs_counter)
    
    plot_histogram_model_4_control(model_i_cs_list_step_1, model_i_counts_list, i, colors[i-1])
