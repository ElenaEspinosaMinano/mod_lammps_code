import numpy as np
import statistics as s
import sys
from tqdm import tqdm

from calculation_functions import (dbscan, size_of_clusters, clusters_greater_than_1, no_proteins_bound_to_poly, 
                                    fraction_clusters_bound_to_poly, no_type_2_poly_bound_to_prot, mean_no_type_2_poly_in_cluster)
from process_dump_file import Atom, readframe, lines_in_file

### main programs --> need to change stuff cause very sucky!

######################## - User inputs (sys argv command line) but worse cause need to remember the order of inputs :/

name_dumpfile = sys.argv[1] # name of dump file to read - dump.sticky_DNA+proteins
"""
n_atoms = int(sys.argv[2]) # no of atoms
n_poly_atoms = int(sys.argv[3]) # no of polymer atoms
"""
name_outfile = sys.argv[2] # name of output file
name_cluster_size_outfile = sys.argv[3]  # name of output file for cluster sizes

######################## - User inputs (command line) - best way imo
"""
name_dumpfile = input("Name of dumpfile: ")

n_atoms = 5300 #int(input("Integer no of atoms (in df): "))
n_poly_atoms = 5000 #int(input("Integer no of polymer atoms (in df): "))

name_outfile = input("Name of output file: ")
"""
########################

# TO DO: could make this more automatic by giving a default of everything... user can say y/n to default and change if req

########################

threshold = 2.3 # cluster threshold - *** 2.3 *** - reduced in version 3
cluster_types = {4}  # cluster target atom type - *** 4 *** - only type 4 can form clusters in version 3

n_atoms = int(sys.argv[4]) # no of atoms - for models 05678 it should be 5600 unless varying no of prot (recall for models 1234 it is 5300)
n_poly_atoms = 5000 # no of polymer atoms

path_to_dumpfiles = '/storage/cmstore01/groups/brackley/s2205640/dumpfiles/' # need to change this if running on local computer
path_to_outfiles_v3 = '../outfiles_v3/'

n_lines = lines_in_file(path_to_dumpfiles + name_dumpfile) # no of lines in file
n_frames = int(n_lines / (n_atoms + 9)) # +9 as 9 header lines in each frame

# open the input file
file_in = open(path_to_dumpfiles + name_dumpfile, 'r')

# open the output files and print a header
file_out = open(path_to_outfiles_v3 + name_outfile, 'w')  
file_out.write("# V2 - Timesteps, No of clusters, Mean cluster size, Size largest cluster, No clusters size 1, No prot bound to poly, Fraction clusters bound to poly, No type 2 poly bound to prot, Mean no type 2 in prot cluster\n")

file_cluster_sizes_out = open(path_to_outfiles_v3 + name_cluster_size_outfile, 'w')
file_cluster_sizes_out.write("# Timesteps, List of cluster sizes\n")

# go through the file frame by frame - tqdm is a progress bar ;)
for frame in tqdm(range(n_frames)):
    # read the frame, unwrapping periodic coordinates
    atoms, timesteps = readframe(file_in, n_atoms)
    
    # unwarp period boundary coordinates -- is it needed for clusters? - think so as a cluster can form between boundaries
    for i in range(len(atoms)):
        atoms[i].unwrap()

    # perform calculations + measurements on clusters
    no_of_clusters, cluster_ids = dbscan(atoms, threshold, cluster_types)
    cluster_sizes = size_of_clusters(cluster_ids)

    no_size_1_clusters, no_of_clusters_v2, cluster_sizes_v2, largest_cluster_size, cluster_ids_v2, mean_cluster_size_v2 = clusters_greater_than_1(cluster_ids, cluster_sizes, no_of_clusters)

    no_proteins_bound, no_polymers_bound_to = no_proteins_bound_to_poly(atoms)
    frac_clusters_bound = fraction_clusters_bound_to_poly(atoms, cluster_ids_v2, no_of_clusters_v2)
    
    no_type_2_poly_bound, no_proteins_bound_to = no_type_2_poly_bound_to_prot(atoms)

    mean_no_type_2_in_cluster = mean_no_type_2_poly_in_cluster(atoms, cluster_ids_v2)

    # output results to files
    file_out.write(f"{timesteps} {no_of_clusters_v2} {mean_cluster_size_v2:.5f} {largest_cluster_size} {no_size_1_clusters} {no_proteins_bound} {frac_clusters_bound:.5f} {no_type_2_poly_bound} {mean_no_type_2_in_cluster:.5f}\n")
    file_cluster_sizes_out.write(f"{timesteps}: {cluster_sizes_v2}\n")

# close the files
file_in.close()
file_out.close()
file_cluster_sizes_out.close()







### make it a bit nicer + reuseable!
"""
def main():
    pass
        
if __name__ == "__main__":
    main()
"""