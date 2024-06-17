########### PLOTS :)
import matplotlib.pyplot as plt
import pandas as pd

# specify paths to read data from + save plots to
local_or_cplab = input("Local or cplab: ")

if local_or_cplab == "local":
    path_to_outfiles = '/home/elenaespinosa/OneDrive/Uni/Summer_courses/Summer_project/mod_lammps_code/outfiles/'
    save_plots_to = '/home/elenaespinosa/OneDrive/Uni/Summer_courses/Summer_project/mod_lammps_code/plots/'
else:
    path_to_outfiles = '/home/s2205640/Documents/summer_project/mod_lammps_code/outfiles/'
    save_plots_to = '/home/s2205640/Documents/summer_project/mod_lammps_code/plots/'




###
#   First investigation - Model 1, 2 + 3 plots
###

outfiles_list_123 = [f'outfile_{i}_run_1.dat' for i in range(1, 4)]

# dictionary to store data frames
data_frames_123 = {}

# parse data
for i in range(1, 4):
    data_frames_123[i] = pd.read_csv(path_to_outfiles + outfiles_list_123[i-1], sep=' ', comment='#', header=None)
    data_frames_123[i].columns = ['Timesteps', 'No_of_clusters', 'Mean_size_of_clusters', 
                              'Size_of_largest_cluster', 'No_of_clusters_of_size_1', 'No_proteins_bound_to_poly']


# plot 1 - Number of clusters vs Timesteps
plt.figure(figsize=(16, 10))

plt.plot(data_frames_123[1]['Timesteps'], data_frames_123[1]['No_of_clusters'], marker='.', alpha=0.7, label='Model 1')
plt.plot(data_frames_123[2]['Timesteps'], data_frames_123[2]['No_of_clusters'], marker='.', alpha=0.7, label='Model 2')
plt.plot(data_frames_123[3]['Timesteps'], data_frames_123[3]['No_of_clusters'], marker='.', alpha=0.7, label='Model 3')

plt.xlabel('Timesteps', fontsize ='16')
plt.xticks(fontsize='14')

plt.ylabel('Number of clusters', fontsize ='16')
plt.yticks(fontsize='14')

plt.title('Number of clusters vs Timesteps', fontsize ='16')
plt.ticklabel_format(style='plain')

plt.legend(fontsize="14", loc ="upper right")
plt.grid(True)

plt.savefig(save_plots_to + "plot_1_run_1.png")
plt.show()


# plot 2 - Mean size of clusters vs Timesteps
plt.figure(figsize=(16, 10))

plt.plot(data_frames_123[1]['Timesteps'], data_frames_123[1]['Mean_size_of_clusters'], marker='.', alpha=0.7, label='Model 1')
plt.plot(data_frames_123[2]['Timesteps'], data_frames_123[2]['Mean_size_of_clusters'], marker='.', alpha=0.7, label='Model 2')
plt.plot(data_frames_123[3]['Timesteps'], data_frames_123[3]['Mean_size_of_clusters'], marker='.', alpha=0.7, label='Model 3')

plt.xlabel('Timesteps', fontsize ='16')
plt.xticks(fontsize='14')

plt.ylabel('Mean size of clusters', fontsize ='16')
plt.yticks(fontsize='14')

plt.title('Mean size of clusters vs Timesteps', fontsize ='16')
plt.ticklabel_format(style='plain')

plt.legend(fontsize="14", loc ="upper left")
plt.grid(True)

plt.savefig(save_plots_to + "plot_2_run_1.png")
plt.show()


# plot 3 - Size of largest cluster vs Timesteps
plt.figure(figsize=(16, 10))

plt.plot(data_frames_123[1]['Timesteps'], data_frames_123[1]['Size_of_largest_cluster'], marker='.', alpha=0.7, label='Model 1')
plt.plot(data_frames_123[2]['Timesteps'], data_frames_123[2]['Size_of_largest_cluster'], marker='.', alpha=0.7, label='Model 2')
plt.plot(data_frames_123[3]['Timesteps'], data_frames_123[3]['Size_of_largest_cluster'], marker='.', alpha=0.7, label='Model 3')

plt.xlabel('Timesteps', fontsize ='16')
plt.xticks(fontsize='14')

plt.ylabel('Size of largest cluster', fontsize ='16')
plt.yticks(fontsize='14')

plt.title('Size of largest cluster vs Timesteps', fontsize ='16')
plt.ticklabel_format(style='plain')

plt.legend(fontsize="14", loc ="upper left")
plt.grid(True)

plt.savefig(save_plots_to + "plot_3_run_1.png")
plt.show()


# plot 4 - Number of clusters of size 1 vs Timesteps
plt.figure(figsize=(16, 10))

plt.plot(data_frames_123[1]['Timesteps'], data_frames_123[1]['No_of_clusters_of_size_1'], marker='.', alpha=0.7, label='Model 1')
plt.plot(data_frames_123[2]['Timesteps'], data_frames_123[2]['No_of_clusters_of_size_1'], marker='.', alpha=0.7, label='Model 2')
plt.plot(data_frames_123[3]['Timesteps'], data_frames_123[3]['No_of_clusters_of_size_1'], marker='.', alpha=0.7, label='Model 3')

plt.xlabel('Timesteps', fontsize ='16')
plt.xticks(fontsize='14')

plt.ylabel('Number of clusters of size 1', fontsize ='16')
plt.yticks(fontsize='14')

plt.title('Number of clusters of size 1 vs Timesteps', fontsize ='16')
plt.ticklabel_format(style='plain')

plt.legend(fontsize="14", loc ="upper right")
plt.grid(True)

plt.savefig(save_plots_to + "plot_4_run_1.png")
plt.show()


# plot 5 - Number of proteins bound to type 2 polymer vs Timesteps
plt.figure(figsize=(16, 10))

plt.plot(data_frames_123[1]['Timesteps'], data_frames_123[1]['No_proteins_bound_to_poly'], marker='.', alpha=0.7, label='Model 1')
plt.plot(data_frames_123[2]['Timesteps'], data_frames_123[2]['No_proteins_bound_to_poly'], marker='.', alpha=0.7, label='Model 2')
plt.plot(data_frames_123[3]['Timesteps'], data_frames_123[3]['No_proteins_bound_to_poly'], marker='.', alpha=0.7, label='Model 3')

plt.xlabel('Timesteps', fontsize ='16')
plt.xticks(fontsize='14')

plt.ylabel('Number of proteins bound to type 2 polymer', fontsize ='16')
plt.yticks(fontsize='14')

plt.title('Number of proteins bound to type 2 polymer vs Timesteps', fontsize ='16')
plt.ticklabel_format(style='plain')

plt.legend(fontsize="14", loc ="upper left")
plt.grid(True)

plt.savefig(save_plots_to + "plot_5_run_1.png")
plt.show()




###
#   Second investigation - Model 4 plots
###

outfiles_list_4 = [f'outfile_4_var_{i}_run_1.dat' for i in range(1, 9)]

# dictionary to store data frames
data_frames_4 = {}

# parse data
for i in range(1, 9):
    data_frames_4[i] = pd.read_csv(path_to_outfiles + outfiles_list_4[i-1], sep=' ', comment='#', header=None)
    data_frames_4[i].columns = ['Timesteps', 'No_of_clusters', 'Mean_size_of_clusters', 
                              'Size_of_largest_cluster', 'No_of_clusters_of_size_1', 'No_proteins_bound_to_poly']


# plot 1 - Number of clusters vs Timesteps
plt.figure(figsize=(16, 10))

plt.plot(data_frames_4[1]['Timesteps'], data_frames_4[1]['No_of_clusters'], marker='.', alpha=0.7, label='1 kBT')
plt.plot(data_frames_4[2]['Timesteps'], data_frames_4[2]['No_of_clusters'], marker='.', alpha=0.7, label='2 kBT')
plt.plot(data_frames_4[3]['Timesteps'], data_frames_4[3]['No_of_clusters'], marker='.', alpha=0.7, label='3 kBT')
plt.plot(data_frames_4[4]['Timesteps'], data_frames_4[4]['No_of_clusters'], marker='.', alpha=0.7, label='4 kBT')
plt.plot(data_frames_4[5]['Timesteps'], data_frames_4[5]['No_of_clusters'], marker='.', alpha=0.7, label='5 kBT')
plt.plot(data_frames_4[6]['Timesteps'], data_frames_4[6]['No_of_clusters'], marker='.', alpha=0.7, label='6 kBT')
plt.plot(data_frames_4[7]['Timesteps'], data_frames_4[7]['No_of_clusters'], marker='.', alpha=0.7, label='7 kBT')
plt.plot(data_frames_4[8]['Timesteps'], data_frames_4[8]['No_of_clusters'], marker='.', alpha=0.7, label='8 kBT')

plt.xlabel('Timesteps', fontsize ='16')
plt.xticks(fontsize='14')

plt.ylabel('Number of clusters', fontsize ='16')
plt.yticks(fontsize='14')

plt.title('Number of clusters vs Timesteps', fontsize ='16')
plt.ticklabel_format(style='plain')

plt.legend(fontsize="14", loc ="upper right")
plt.grid(True)

plt.savefig(save_plots_to + "plot_1_model_4_run_1.png")
plt.show()


# plot 2 - Mean size of clusters vs Timesteps
plt.figure(figsize=(16, 10))

plt.plot(data_frames_4[1]['Timesteps'], data_frames_4[1]['Mean_size_of_clusters'], marker='.', alpha=0.7, label='1 kBT')
plt.plot(data_frames_4[2]['Timesteps'], data_frames_4[2]['Mean_size_of_clusters'], marker='.', alpha=0.7, label='2 kBT')
plt.plot(data_frames_4[3]['Timesteps'], data_frames_4[3]['Mean_size_of_clusters'], marker='.', alpha=0.7, label='3 kBT')
plt.plot(data_frames_4[4]['Timesteps'], data_frames_4[4]['Mean_size_of_clusters'], marker='.', alpha=0.7, label='4 kBT')
plt.plot(data_frames_4[5]['Timesteps'], data_frames_4[5]['Mean_size_of_clusters'], marker='.', alpha=0.7, label='5 kBT')
plt.plot(data_frames_4[6]['Timesteps'], data_frames_4[6]['Mean_size_of_clusters'], marker='.', alpha=0.7, label='6 kBT')
plt.plot(data_frames_4[7]['Timesteps'], data_frames_4[7]['Mean_size_of_clusters'], marker='.', alpha=0.7, label='7 kBT')
plt.plot(data_frames_4[8]['Timesteps'], data_frames_4[8]['Mean_size_of_clusters'], marker='.', alpha=0.7, label='8 kBT')

plt.xlabel('Timesteps', fontsize ='16')
plt.xticks(fontsize='14')

plt.ylabel('Mean size of clusters', fontsize ='16')
plt.yticks(fontsize='14')

plt.title('Mean size of clusters vs Timesteps', fontsize ='16')
plt.ticklabel_format(style='plain')

plt.legend(fontsize="14", loc ="upper left")
plt.grid(True)

plt.savefig(save_plots_to + "plot_2_model_4_run_1.png")
plt.show()


# plot 3 - Size of largest cluster vs Timesteps
plt.figure(figsize=(16, 10))

plt.plot(data_frames_4[1]['Timesteps'], data_frames_4[1]['Size_of_largest_cluster'], marker='.', alpha=0.7, label='1 kBT')
plt.plot(data_frames_4[2]['Timesteps'], data_frames_4[2]['Size_of_largest_cluster'], marker='.', alpha=0.7, label='2 kBT')
plt.plot(data_frames_4[3]['Timesteps'], data_frames_4[3]['Size_of_largest_cluster'], marker='.', alpha=0.7, label='3 kBT')
plt.plot(data_frames_4[4]['Timesteps'], data_frames_4[4]['Size_of_largest_cluster'], marker='.', alpha=0.7, label='4 kBT')
plt.plot(data_frames_4[5]['Timesteps'], data_frames_4[5]['Size_of_largest_cluster'], marker='.', alpha=0.7, label='5 kBT')
plt.plot(data_frames_4[6]['Timesteps'], data_frames_4[6]['Size_of_largest_cluster'], marker='.', alpha=0.7, label='6 kBT')
plt.plot(data_frames_4[7]['Timesteps'], data_frames_4[7]['Size_of_largest_cluster'], marker='.', alpha=0.7, label='7 kBT')
plt.plot(data_frames_4[8]['Timesteps'], data_frames_4[8]['Size_of_largest_cluster'], marker='.', alpha=0.7, label='8 kBT')

plt.xlabel('Timesteps', fontsize ='16')
plt.xticks(fontsize='14')

plt.ylabel('Size of largest cluster', fontsize ='16')
plt.yticks(fontsize='14')

plt.title('Size of largest cluster vs Timesteps', fontsize ='16')
plt.ticklabel_format(style='plain')

plt.legend(fontsize="14", loc ="upper left")
plt.grid(True)

plt.savefig(save_plots_to + "plot_3_model_4_run_1.png")
plt.show()


# plot 4 - Number of clusters of size 1 vs Timesteps
plt.figure(figsize=(16, 10))

plt.plot(data_frames_4[1]['Timesteps'], data_frames_4[1]['No_of_clusters_of_size_1'], marker='.', alpha=0.7, label='1 kBT')
plt.plot(data_frames_4[2]['Timesteps'], data_frames_4[2]['No_of_clusters_of_size_1'], marker='.', alpha=0.7, label='2 kBT')
plt.plot(data_frames_4[3]['Timesteps'], data_frames_4[3]['No_of_clusters_of_size_1'], marker='.', alpha=0.7, label='3 kBT')
plt.plot(data_frames_4[4]['Timesteps'], data_frames_4[4]['No_of_clusters_of_size_1'], marker='.', alpha=0.7, label='4 kBT')
plt.plot(data_frames_4[5]['Timesteps'], data_frames_4[5]['No_of_clusters_of_size_1'], marker='.', alpha=0.7, label='5 kBT')
plt.plot(data_frames_4[6]['Timesteps'], data_frames_4[6]['No_of_clusters_of_size_1'], marker='.', alpha=0.7, label='6 kBT')
plt.plot(data_frames_4[7]['Timesteps'], data_frames_4[7]['No_of_clusters_of_size_1'], marker='.', alpha=0.7, label='7 kBT')
plt.plot(data_frames_4[8]['Timesteps'], data_frames_4[8]['No_of_clusters_of_size_1'], marker='.', alpha=0.7, label='8 kBT')

plt.xlabel('Timesteps', fontsize ='16')
plt.xticks(fontsize='14')

plt.ylabel('Number of clusters of size 1', fontsize ='16')
plt.yticks(fontsize='14')

plt.title('Number of clusters of size 1 vs Timesteps', fontsize ='16')
plt.ticklabel_format(style='plain')

plt.legend(fontsize="14", loc ="upper right")
plt.grid(True)

plt.savefig(save_plots_to + "plot_4_model_4_run_1.png")
plt.show()


# plot 5 - Number of proteins bound to polymer vs Timesteps
plt.figure(figsize=(16, 10))

plt.plot(data_frames_4[1]['Timesteps'], data_frames_4[1]['No_proteins_bound_to_poly'], marker='.', alpha=0.7, label='1 kBT')
plt.plot(data_frames_4[2]['Timesteps'], data_frames_4[2]['No_proteins_bound_to_poly'], marker='.', alpha=0.7, label='2 kBT')
plt.plot(data_frames_4[3]['Timesteps'], data_frames_4[3]['No_proteins_bound_to_poly'], marker='.', alpha=0.7, label='3 kBT')
plt.plot(data_frames_4[4]['Timesteps'], data_frames_4[4]['No_proteins_bound_to_poly'], marker='.', alpha=0.7, label='4 kBT')
plt.plot(data_frames_4[5]['Timesteps'], data_frames_4[5]['No_proteins_bound_to_poly'], marker='.', alpha=0.7, label='5 kBT')
plt.plot(data_frames_4[6]['Timesteps'], data_frames_4[6]['No_proteins_bound_to_poly'], marker='.', alpha=0.7, label='6 kBT')
plt.plot(data_frames_4[7]['Timesteps'], data_frames_4[7]['No_proteins_bound_to_poly'], marker='.', alpha=0.7, label='7 kBT')
plt.plot(data_frames_4[8]['Timesteps'], data_frames_4[8]['No_proteins_bound_to_poly'], marker='.', alpha=0.7, label='8 kBT')

plt.xlabel('Timesteps', fontsize ='16')
plt.xticks(fontsize='14')

plt.ylabel('Number of proteins bound to polymer', fontsize ='16')
plt.yticks(fontsize='14')

plt.title('Number of proteins bound to polymer vs Timesteps', fontsize ='16')
plt.ticklabel_format(style='plain')

plt.legend(fontsize="14", loc ="lower right")
plt.grid(True)

plt.savefig(save_plots_to + "plot_5_model_4_run_1.png")
plt.show()



###
#   Second investigation - Model 4 plots - control
###

outfiles_list_4_control = [f'outfile_4_var_{i}_run_1_control.dat' for i in range(1, 9)]

# dictionary to store data frames
data_frames_4_control = {}

# parse data
for i in range(1, 9):
    data_frames_4_control[i] = pd.read_csv(path_to_outfiles + outfiles_list_4_control[i-1], sep=' ', comment='#', header=None)
    data_frames_4_control[i].columns = ['Timesteps', 'No_of_clusters', 'Mean_size_of_clusters', 
                              'Size_of_largest_cluster', 'No_of_clusters_of_size_1', 'No_proteins_bound_to_poly']

# plot graphs

# plot 1 - Number of clusters vs Timesteps
plt.figure(figsize=(16, 10))

for i in range(1,9):
    plt.plot(data_frames_4_control[i]['Timesteps'], data_frames_4_control[i]['No_of_clusters'], marker='.', alpha=0.7, label=f'{i} kBT')

plt.xlabel('Timesteps', fontsize ='16')
plt.xticks(fontsize='14')

plt.ylabel('Number of clusters', fontsize ='16')
plt.yticks(fontsize='14')

plt.title('Number of clusters vs Timesteps', fontsize ='16')
plt.ticklabel_format(style='plain')

plt.legend(fontsize="14", loc ="center right")
plt.grid(True)

plt.savefig(save_plots_to + "plot_1_model_4_run_1_control.png")
plt.show()


# plot 2 - Mean size of clusters vs Timesteps
plt.figure(figsize=(16, 10))

for i in range(1,9):
    plt.plot(data_frames_4_control[i]['Timesteps'], data_frames_4_control[i]['Mean_size_of_clusters'], marker='.', alpha=0.7, label=f'{i} kBT')

plt.xlabel('Timesteps', fontsize ='16')
plt.xticks(fontsize='14')

plt.ylabel('Mean size of clusters', fontsize ='16')
plt.yticks(fontsize='14')

plt.title('Mean size of clusters vs Timesteps', fontsize ='16')
plt.ticklabel_format(style='plain')

plt.legend(fontsize="14", loc ="upper left")
plt.grid(True)

plt.savefig(save_plots_to + "plot_2_model_4_run_1_control.png")
plt.show()


# plot 3 - Size of largest cluster vs Timesteps
plt.figure(figsize=(16, 10))

for i in range(1,9):
    plt.plot(data_frames_4_control[i]['Timesteps'], data_frames_4_control[i]['Size_of_largest_cluster'], marker='.', alpha=0.7, label=f'{i} kBT')

plt.xlabel('Timesteps', fontsize ='16')
plt.xticks(fontsize='14')

plt.ylabel('Size of largest cluster', fontsize ='16')
plt.yticks(fontsize='14')

plt.title('Size of largest cluster vs Timesteps', fontsize ='16')
plt.ticklabel_format(style='plain')

plt.legend(fontsize="14", loc ="upper left")
plt.grid(True)

plt.savefig(save_plots_to + "plot_3_model_4_run_1_control.png")
plt.show()


# plot 4 - Number of clusters of size 1 vs Timesteps
plt.figure(figsize=(16, 10))

for i in range(1,9):
    plt.plot(data_frames_4_control[i]['Timesteps'], data_frames_4_control[i]['No_of_clusters_of_size_1'], marker='.', alpha=0.7, label=f'{i} kBT')

plt.xlabel('Timesteps', fontsize ='16')
plt.xticks(fontsize='14')

plt.ylabel('Number of clusters of size 1', fontsize ='16')
plt.yticks(fontsize='14')

plt.title('Number of clusters of size 1 vs Timesteps', fontsize ='16')
plt.ticklabel_format(style='plain')

plt.legend(fontsize="14", loc ="center right")
plt.grid(True)

plt.savefig(save_plots_to + "plot_4_model_4_run_1_control.png")
plt.show()


# plot 5 - Number of proteins bound to polymer vs Timesteps
plt.figure(figsize=(16, 10))

for i in range(1,9):
    plt.plot(data_frames_4_control[i]['Timesteps'], data_frames_4_control[i]['No_proteins_bound_to_poly'], marker='.', alpha=0.7, label=f'{i} kBT')

plt.xlabel('Timesteps', fontsize ='16')
plt.xticks(fontsize='14')

plt.ylabel('Number of proteins bound to polymer', fontsize ='16')
plt.yticks(fontsize='14')

plt.title('Number of proteins bound to polymer vs Timesteps', fontsize ='16')
plt.ticklabel_format(style='plain')

plt.legend(fontsize="14", loc ="center right")
plt.grid(True)

plt.savefig(save_plots_to + "plot_5_model_4_run_1_control.png")
plt.show()



###
#   Testing plot for new function calculating number of proteins bound to polymer type 2 + mean number of polymer beads proteins are bound
###


# file path
path_test_file = '/home/s2205640/Documents/summer_project/mod_lammps_code/outfiles/outfile_test.run'

# data test!
data_test = pd.read_csv(path_test_file, sep=' ', comment='#', header=None)
data_test.columns = ['Timesteps', 'No_proteins_bound', 'Mean_no_polymers_bound_to']


# plot test 1
plt.figure(figsize=(16, 10))

plt.plot(data_test['Timesteps'], data_test['No_proteins_bound'], marker='.', alpha=0.7, label='Model 2')

plt.xlabel('Timesteps', fontsize ='16')
plt.xticks(fontsize='14')

plt.ylabel('Number of proteins bound to polymer type 2', fontsize ='16')
plt.yticks(fontsize='14')

plt.title('Number of proteins bound to polymer type 2 vs Timesteps', fontsize ='16')
plt.ticklabel_format(style='plain')

plt.legend(fontsize="14", loc ="upper left")
plt.grid(True)

#plt.savefig(save_plots_to + "plot_1_run_1.png")
plt.show()


# plot test 2
plt.figure(figsize=(16, 10))

plt.plot(data_test['Timesteps'], data_test['Mean_no_polymers_bound_to'], marker='.', alpha=0.7, label='Model 2')

plt.xlabel('Timesteps', fontsize ='16')
plt.xticks(fontsize='14')

plt.ylabel('Mean number of polymer beads proteins are bound to', fontsize ='16')
plt.yticks(fontsize='14')

plt.title('Mean number of polymer beads proteins are bound to vs Timesteps', fontsize ='16')
plt.ticklabel_format(style='plain')

plt.legend(fontsize="14", loc ="upper left")
plt.grid(True)

#plt.savefig(save_plots_to + "plot_1_run_1.png")
plt.show()

