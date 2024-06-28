import numpy as np
import statistics as s
import sys
from tqdm import tqdm

from calculation_functions import (dbscan, size_of_clusters, mean_size_of_clusters, size_of_largest_cluster, no_of_clusters_size_1,
                                    no_proteins_bound_to_poly, fraction_clusters_bound_to_poly, no_type_2_poly_bound_to_prot, mean_no_type_2_poly_in_cluster)
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

threshold = 2.4 # cluster threshold - 2.4
target_types = {4, 5}  # cluster target atom type - 4 and 5 are proteins (on / off)

n_atoms = 5600 # no of atoms - had to change this for models 567
n_poly_atoms = 5000 # no of polymer atoms

path_to_dumpfiles = '/storage/cmstore01/groups/brackley/s2205640/dumpfiles/' # need to change this when running on local computer
path_to_outfiles = '../outfiles/'

n_lines = lines_in_file(path_to_dumpfiles + name_dumpfile) # no of lines in file
n_frames = int(n_lines / (n_atoms + 9)) # +9 as 9 header lines in each frame

# open the input file
file_in = open(path_to_dumpfiles + name_dumpfile, 'r')

# open the output files and print a header
file_out = open(path_to_outfiles + name_outfile, 'w')  
file_out.write("# Timesteps, No of clusters, Mean cluster size, Size largest cluster, No clusters size 1, No prot bound to poly, Fraction clusters bound to poly, No type 2 poly bound to prot, Mean no type 2 in prot cluster\n")

file_cluster_sizes_out = open(path_to_outfiles + name_cluster_size_outfile, 'w')
file_cluster_sizes_out.write("# Timesteps, List of cluster sizes\n")

# go through the file frame by frame - tqdm is a progress bar
for frame in tqdm(range(n_frames)):
    # read the frame, unwrapping periodic coordinates
    atoms, timesteps = readframe(file_in, n_atoms)
    
    # unwarp period boundary coordinates -- is it needed for clusters? - think so as a cluster can form between boundaries
    for i in range(len(atoms)):
        atoms[i].unwrap()

    # perform calculations + measurements on clusters
    no_of_clusters, cluster_ids = dbscan(atoms, threshold, target_types)
    cluster_size = size_of_clusters(cluster_ids)
    mean_cluster_size = mean_size_of_clusters(cluster_size)
    largest_cluster_size = size_of_largest_cluster(cluster_size)
    size_1_clusters = no_of_clusters_size_1(cluster_size)

    no_proteins_bound, no_polymers_bound_to = no_proteins_bound_to_poly(atoms)
    frac_clusters_bound = fraction_clusters_bound_to_poly(atoms, cluster_ids, no_of_clusters)
    
    no_type_2_poly_bound, no_proteins_bound_to = no_type_2_poly_bound_to_prot(atoms)

    mean_no_type_2_in_cluster = mean_no_type_2_poly_in_cluster(atoms, cluster_ids)

    # output results to files
    file_out.write(f"{timesteps} {no_of_clusters} {mean_cluster_size:.5f} {largest_cluster_size} {size_1_clusters} {no_proteins_bound} {frac_clusters_bound:.5f} {no_type_2_poly_bound} {mean_no_type_2_in_cluster:.5f}\n")
    file_cluster_sizes_out.write(f"{timesteps}: {cluster_size}\n")

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