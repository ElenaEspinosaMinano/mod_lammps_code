import numpy as np
import operator # do we need this?
import statistics as s
import sys
from tqdm import tqdm

from calculation_functions import dbscan, size_of_clusters, mean_size_of_clusters, size_of_largest_cluster, no_of_clusters_size_1, no_proteins_bound_to_poly
from process_dump_file import Atom, readframe, lines_in_file

### main programs --> need to change stuff cause very sucky!

######################## - User inputs (sys argv command line) but worse cause need to remember the order of inputs :/


name_dumpfile = sys.argv[1] # name of dump file to read - dump.sticky_DNA+proteins

n_atoms = int(sys.argv[2]) # no of atoms - 220
n_poly_atoms = int(sys.argv[3]) # no of polymer atoms - 200

name_outfile = sys.argv[4] # name of output file - r_g_sticky_DNA


######################## - Hardcoded (use for testing)

#name_dumpfile = 'dump.sticky_DNA+proteins' # name of dump file to read - dump.sticky_DNA+proteins

#n_atoms = 220 # no of atoms - 220
#n_poly_atoms = 200 # no of polymer atoms - 200

#name_outfile = 'r_g_sticky_DNA' # name of output file - r_g_sticky_DNA

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
target_type = 4  # target atom type - should change this to type 4 for model simulations... / could add as user input

path_to_dumpfiles = '../../lammps_sims/dumpfiles/' # need to change this when running on local computer
path_to_outfiles = '../outfiles/'

n_lines = lines_in_file(path_to_dumpfiles + name_dumpfile) # no of lines in file
n_frames = int(n_lines / (n_atoms + 9)) # +9 as 9 header lines in each frame

# open the input file
file_in = open(path_to_dumpfiles + name_dumpfile, 'r')

# open the output file and print a header
file_out = open(path_to_outfiles + name_outfile, 'w')  
file_out.write("# Timesteps, No of clusters, Mean cluster size, Size largest cluster, No clusters size 1, No proteins bound to poly\n")

# go through the file frame by frame - tqdm is a progress bar
for frame in tqdm(range(n_frames)):
    # read the frame, unwrapping periodic coordinates
    atoms, timesteps = readframe(file_in, n_atoms)
    
    # unwarp period boundary coordinates -- is it needed for clusters? - think so as a cluster can form between boundaries
    for i in range(len(atoms)):
        atoms[i].unwrap()

    # perform calculations on clusters
    no_of_clusters, cluster_ids = dbscan(atoms, threshold, target_type)
    cluster_size = size_of_clusters(cluster_ids)
    mean_cluster_size = mean_size_of_clusters(cluster_size)
    largest_cluster_size = size_of_largest_cluster(cluster_size)
    size_1_count = no_of_clusters_size_1(cluster_size)

    no_proteins_bound, no_polymers_bound_to = no_proteins_bound_to_poly(atoms)
    
    # output some results to file
    file_out.write("%i %i %.5f %i %i %i\n"%(timesteps, no_of_clusters, mean_cluster_size, largest_cluster_size, size_1_count, no_proteins_bound))


# close the files
file_in.close()
file_out.close()


### make it a bit nicer + reuseable!
"""
def main():
    pass
        
if __name__ == "__main__":
    main()
"""