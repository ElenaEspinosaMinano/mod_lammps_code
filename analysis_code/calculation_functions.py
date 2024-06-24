### calculation functions

import statistics as s

def dbscan(atoms, threshold, target_type):
    """ Takes in a list of Atom objects, distance threshold + target type. 
        Sets the cluster id for atoms we are not interested in to -2.
        Returns a list of cluster ids where the ith element is the cluster id for ith atom in input list """

    if not atoms: # edge case of empty atom list - return an empty cluster id list
        return []

    cluster_id = 0
    cluster_ids = [-1] * len(atoms)  # Initialize cluster IDs for each atom; -1 means unclassified

    threshold_2 = threshold**2

    def find_neighbors(atom_index):
        """ Takes in index of an atom and find the neighbours of that atom. 
            Atoms are neighbours if within threshold distance of 2.4.
            Returns a list of neighbours of type=4 for atom index inputted """

        return [i for i, other_atom in enumerate(atoms)
                if i != atom_index and atoms[atom_index].type == target_type and other_atom.type == target_type and 
                atoms[atom_index].sep_2(other_atom) < threshold_2]
    
    # loops through atoms list
    for i in range(len(atoms)):

        # checks to see if cluster id of ith atom is -1 - if not atom already processed
        if cluster_ids[i] != -1 or atoms[i].type != target_type:

            # checks to see if atom type is target type - if not sets cluster id of that atom to -2 (not of interest)
            if atoms[i].type != target_type:
                cluster_ids[i] = -2

            continue

        neighbors = find_neighbors(i) # finds neighbours of atom i

        cluster_id += 1
        cluster_ids[i] = cluster_id

        k = 0

        # for every neighbour of atom i, finds the neighbours of that neighbour and adds it to original neighbour list
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
    """ Takes in a list of cluster ids. Returns a list of the number of proteins in a cluster, ignoring cluster ids of -2.
        Eg: [-2, -2, ..., 1, 4, 2, 4, 1, 3, 4, ...] --> 2 in cluster 1, 1 in cluster 2, 1 in cluster 3, 3 in cluster 4 ... """
    
    size_of_clusters = [0] * max(cluster_ids) # 0 means 0 atoms in cluster i
    
    # loop through list of cluster ids
    for i in cluster_ids:
        
        # if the cluster id is not -2 (-2 are atoms of non-interest)
        if i != -2:
            # increase value of i-1 by 1
            size_of_clusters[i-1] += 1
                    
    return size_of_clusters



def mean_size_of_clusters(size_of_clusters):
    """ Takes in list of cluster sizes (in number of proteins) and returns the mean number of proteins in a cluster """
    
    return s.fmean(size_of_clusters) # fmean runs faster than mean apparently



def size_of_largest_cluster(size_of_clusters):
    """ Takes in list of cluster sizes and returns the number of proteins in largest cluster """
    
    return max(size_of_clusters)


def no_of_clusters_size_1(size_of_clusters):
    """ Takes in a list of cluster sizes and returns the number of clusters with only 1 protein - i.e. size 1 """

    return size_of_clusters.count(1)


def no_proteins_bound_to_poly(atoms, target_types={1, 2, 3}, threshold_2=3.24):
    """ Takes in a list of Atom objects. 
        Returns the no of proteins bound to the polymer of type=1, 2 or 3 + list of no of polymer beads an atom is bound to. """

    def find_polymers_bound_to(j):
        """ Takes in index of a protein (type 4) and finds if it is bound to polymer (type=1, 2 or 3). 
            Bound to polymer if within threshold distance of 1.8. Or if sep_2 < threshold_2 (3.24).
            Returns a list of indices of polymer beads of type=1, 2 or 3 for atom index inputted. 
            Length of list is no of polymer beads that protein is bound to. """

        return [i for i, other_atom in enumerate(atoms)
                #if i != j and other_atom.type in {2} and atoms[j].sep2(other_atom) < threshold_2]
                if i != j and other_atom.type in target_types and atoms[j].sep_2(other_atom) < threshold_2]


    no_proteins_bound = 0 # intialises counter of number of proteins bound to a polymer bead to 0
    no_polymers_bound_to = [0] * len(atoms) # initialises list of number of polymer beads that each protein is bound to, to 0

    # loops through all atoms - could make it loop through just the last 300 as those are the proteins!
    for j, atom in enumerate(atoms):

        # if atom is a protein (type 4)
        if atom.type == 4:

            # find number of polymer beads (of any type - 1, 2 or 3) that protein is bound to
            poly_beads_list = find_polymers_bound_to(j)
            no_polymers_bound_to[j] = len(poly_beads_list) # length of list is the no of polymer beads that protein is bound to

            # if list is not empty
            if poly_beads_list:
                no_proteins_bound += 1

    return no_proteins_bound, no_polymers_bound_to


def fraction_clusters_bound_to_poly(atoms, cluster_ids, no_of_clusters, target_types={1, 2, 3}, threshold_2=3.24):
    """ Takes in a list of Atom objects, their cluster ids and no_of_clusters (at each frame). 
        Returns the fraction of clusters bound to the polymer of type=1, 2 or 3 """

    def protein_bound_to_poly(j):
        """ Takes in index of a protein (type 4) and finds if it is bound to polymer (type=1, 2 or 3). 
            Bound to polymer if within threshold distance of 1.8. Or if sep_2 < threshold_2 (3.24). 
            Stops and returns True if it is bound. False if not bound. """
        for i, other_atom in enumerate(atoms):
            if i != j and other_atom.type in target_types and atoms[j].sep_2(other_atom) < threshold_2:
                return True # stop when you find a protein in cluster that is within threshold distance of any polymer bead
        return False

    bound_clusters = []  # list of cluster ids of bound clusters
    
    # loop through all cluster ids
    for j, cluster_id in enumerate(cluster_ids):

        # if cluster_id is not -2 (ie: it is the cluster id of a protein) and it is not already in bound_clusters
        if cluster_id != -2 and cluster_id not in bound_clusters:

            # check if the protein belonging to that cluster id is bound to a polymer bead. If yes, add to bound_clusters list.
            if protein_bound_to_poly(j):
                bound_clusters.append(cluster_id)

    fraction_bound = len(bound_clusters) / no_of_clusters # no of bound clusters / total no of clusters

    return fraction_bound


def no_type_2_poly_bound_to_prot(atoms, target_types={4}, threshold_2=3.24):
    """ Takes in a list of Atom objects. 
        Returns the no of type 2 polymer bound to proteins + list of no of proteins that atom is bound to. """

    def find_proteins_bound_to(j):
        """ Takes in index of a sticky polymer (type=2) and finds if it is bound to protein (type=4). 
            Bound to protein if within threshold distance of 1.8. Or if sep_2 < threshold_2 (3.24).
            Returns a list of indices of proteins that polymer index inputted is bound to. 
            Length of list is no of proteins that polymer is bound to. """

        return [i for i, other_atom in enumerate(atoms)
                if i != j and other_atom.type in target_types and atoms[j].sep_2(other_atom) < threshold_2]


    no_type_2_poly_bound = 0 # intialises counter of number of proteins bound to a polymer bead to 0
    no_proteins_bound_to = [0] * len(atoms) # initialises list of number of proteins that each polymer type 2 is bound to, to 0

    # loops through all atoms
    for j, atom in enumerate(atoms):

        # if atom is a sticky polymer (type 2)
        if atom.type == 2:

            # find number of proteins (type 4) that sticky polymer is bound to
            proteins_list = find_proteins_bound_to(j)
            no_proteins_bound_to[j] = len(proteins_list) # length of list is the no of proteins that sticky polymer is bound to

            # if list is not empty
            if proteins_list:
                no_type_2_poly_bound += 1

    return no_type_2_poly_bound, no_proteins_bound_to


def mean_no_type_2_poly_in_cluster(atoms, cluster_ids, target_poly_types={2}, threshold_2=3.24):
    """ Takes in a list of Atom objects and cluster_ids.
        Returns the mean number of type 2 polymers bound to proteins in a cluster """

    def find_type_2_polymers_bound_to(j):
        """ Takes in index of a protein (type 4) and finds if it is bound to polymer (type=2). 
            Bound to polymer if within threshold distance of 1.8. Or if sep_2 < threshold_2 (3.24).
            Returns a list of indices of polymer beads of type=2 for protein index inputted. 
            Length of list is no of polymer beads that protein is bound to. """

        return [i for i, other_atom in enumerate(atoms)
                if i != j and other_atom.type in target_poly_types and atoms[j].sep_2(other_atom) < threshold_2]

    no_type_2_poly_in = {} # dictionary of cluster ids to no of type 2 polymers in that cluster id

    # loop through all cluster ids
    for j, cluster_id in enumerate(cluster_ids):
        
        # if cluster_id is not -2 (ie: it is the cluster id of a protein)
        if cluster_id != -2:

            # find the atom indices of type 2 polymer beads that protein is bound to
            poly_beads_list = find_type_2_polymers_bound_to(j)

            # find the rest of the proteins in the cluster that the protein is in
            for i, cluster_id_i in enumerate(cluster_ids):

                if cluster_id_i == cluster_id and i != j:
                    
                    # add to list of indices of type 2 polymers bound to proteins in cluster_id
                    poly_beads_list += find_type_2_polymers_bound_to(i) # note: there will be repeated indices in the list
        
        # remove duplicates from list + add to dictionary
        no_type_2_poly_in[cluster_id] = len(set(poly_beads_list))

    return s.fmean(no_type_2_poly_in.values())



"""
    # loops through all atoms
    for j, atom in enumerate(atoms):

        # if atom is a protein (type 4)
        if atom.type == 4:

            # find the atom indices of type 2 polymer beads that protein is bound to
            poly_beads_list = find_type_2_polymers_bound_to(j)

            # find the rest of the proteins in the cluster that the protein is in
            cluster_id = cluster_ids[j]
            for i, cluster_id_i in enumerate(cluster_ids):
                if cluster_id_i == cluster_id and i != j:
                    poly_beads_list += find_type_2_polymers_bound_to(i)
"""