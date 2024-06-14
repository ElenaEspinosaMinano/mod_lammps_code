### Atom class + readframe + lines_in_file functions - used to process dump files

import numpy as np
import operator # do we need this?

class Atom:
    """ A Class for storing atom information """

    def __init__(self):
        """ Initialise the class """
        self.id = 0                                              # id of the atom
        self.type = 0                                            # type of the atom
        self.L = np.array([0.0,0.0,0.0],dtype=np.float64)        # size of simulation box
        self.half_L = self.L / 2                                 # half the size of simulation box
        self.x = np.array([0.0,0.0,0.0],dtype=np.float64)        # position of the atom
        self.image = np.array([0,0,0],dtype=np.int32)            # image flags for atoms
        self.x_unwrap = np.array([0.0,0.0,0.0],dtype=np.float64) # position of the atom - unwrapped coords
        self.unwrap_flag = False

    
    def sep(self, atom2):
        """ Takes in self and atom2 and finds separation between them taking into account periodic BCs """

        dx = self.x[0] - atom2.x[0]
        dy = self.x[1] - atom2.x[1]
        dz = self.x[2] - atom2.x[2]

        if dx > self.half_L[0]:
            dx = self.L[0] - dx
        if dy > self.half_L[1]:
            dy = self.L[1] - dy
        if dz > self.half_L[2]:
            dz = self.L[2] - dz

        return np.sqrt(dx**2 + dy**2 + dz**2)
    
    def sep_2(self, atom2):
        """ Takes in self and atom2 and finds their separation SQUARED WITHOUT taking into account periodic BCs """

        dx = self.x[0] - atom2.x[0]
        dy = self.x[1] - atom2.x[1]
        dz = self.x[2] - atom2.x[2]

        """
        if dx > self.half_L[0]:
            dx = self.L[0] - dx
        if dy > self.half_L[1]:
            dy = self.L[1] - dy
        if dz > self.half_L[2]:
            dz = self.L[2] - dz
        """

        return dx**2 + dy**2 + dz**2

    def minus(self,B):
        """ Subtract B.x vector from self.x vector """
        
        AminusB = np.array([0.0,0.0,0.0],dtype=np.float64)
        for i in range(3):
            AminusB[i] = self.x[i] - B.x[i]
            
        return AminusB

    def xdot(self,B):
        """ Find dot product of position x of this Atom and Atom B """
        
        AdotB = np.array([0.0,0.0,0.0],dtype=np.float64) # TO DO : find AdotB ??
        
        return AdotB

    # confused as to what this function does :/ - isn't unwrap_flag always False?
    def unwrap(self):
        """ Unwraps the coordinates for periodic box to generate x_unwrap array """
        
        if not self.unwrap_flag:   # first check it has not already been done
            for j in range(3):
                self.x_unwrap[j] = self.x[j] + self.image[j]*self.L[j] # unwrap
            unwrap_flag = True
            
            
### functions

def readframe(infile, N):
    """ Read a single frame of N atoms from a dump file 
        Expects coordinates to be in range -L/2 to L/2
        DOES NOT Unwrap corrdinates for periodic box """

    atoms = [Atom() for i in range(N)]
    L = np.array([0.0,0.0,0.0],dtype=np.float64)

    # read in the 9 header lines of the dump file
    # get box size
    for i in range(9):
        line = infile.readline()
        if i==1:  ## second line of frame is timestep
            timestep = np.int32(line)
        if i==5 or i==6 or i==7:   # 6th-8th lines are box size in x,y,z dimensions
            # get the box size
            line = line.split()
            L[i-5]=np.float64(line[1]) - np.float64(line[0]);

    # now read the atoms, putting them at the correct index (index=id-1)
    for i in range(N):
        line = infile.readline()
        line = line.split()
        index = int(line[0])-1  # LAMMPS atom ids start from 1, python index starts from 0
        atoms[index].id = int(line[0])
        atoms[index].type = int(line[1])
        atoms[index].L = L
        for j in range(3):
            atoms[index].x[j] = np.float64(line[j+2])
        for j in range(3):
            atoms[index].image[j] = np.int32(line[j+5])

    return atoms, timestep


def lines_in_file(filename):
    """ Get the number of lines in the file """

    with open(filename) as f:
        for i, l in enumerate(f):
            pass

    return i + 1
