### calculation functions

import statistics as s

def dbscan(atoms, threshold, target_type):
    if not atoms:
        return []

    cluster_id = 0
    cluster_ids = [-1] * len(atoms)  # Initialize cluster IDs for each atom; -1 means unclassified

    def find_neighbors(atom_index):
        return [i for i, other_atom in enumerate(atoms)
                if i != atom_index and atoms[atom_index].type == target_type and other_atom.type == target_type and 
                atoms[atom_index].sep(other_atom) < threshold]
    
    
    for i in range(len(atoms)):
        if cluster_ids[i] != -1 or atoms[i].type != target_type:
            if atoms[i].type != target_type:
                cluster_ids[i] = -2  # Mark atoms of non-interest with -2 or another special ID
            continue  # Skip if atom is already processed or not of target type

        neighbors = find_neighbors(i)

        cluster_id += 1
        cluster_ids[i] = cluster_id

        k = 0
        while k < len(neighbors):
            neighbor_idx = neighbors[k]
            if cluster_ids[neighbor_idx] == -1:
                cluster_ids[neighbor_idx] = cluster_id

                new_neighbors = find_neighbors(neighbor_idx)
                for new_neighbor in new_neighbors:
                    if cluster_ids[new_neighbor] == -1:
                        neighbors.append(new_neighbor)
            k += 1
            
    no_of_clusters = cluster_id # cluster_id acts like a 'cluster count'

    return no_of_clusters, cluster_ids



def size_of_clusters(cluster_ids):
    """ Returns a list of the number of proteins in a cluster """
    
    size_of_clusters = [0] * max(cluster_ids) # 0 means 0 atoms in cluster i
    
    # loop through list of cluster ids
    for i in cluster_ids:
        
        # if the cluster id is not -2 (-2 are atoms of non-interest)
        if i != -2:
            # increase value of i-1 by 1
            size_of_clusters[i-1] += 1
                    
    return size_of_clusters



def mean_size_of_clusters(size_of_clusters):
    """ Takes in list of size of clusters (in number of proteins) and returns the mean of proteins in a cluster """
    
    return s.fmean(size_of_clusters)



def size_of_largest_cluster(size_of_clusters):
    """ Takes in list of size of clusters and returns the no of proteins in largest cluster """
    
    return max(size_of_clusters)
