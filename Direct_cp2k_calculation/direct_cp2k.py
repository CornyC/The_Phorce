import copy
import os
import fnmatch
from pathlib import Path
import numpy as np
import re
from System.paths import *

#### Calculate Mulliken, Hirshfeld, RESP charges, and BSSE using cp2k ####

class Direct_Calculator:
    """
    Get atomic charges from DFT and calculate the basis set superposition error directly from cp2k
    bc ASE does not have this implemented

    Parameters
    ----------
    project_path : str
        path to directory where the cp2k files will be read/written
    project_name : str
        name of the cp2k project, also used to name files
    run_type : str
        type of cp2k calculation: Either charges or BSSE
    """
    def __init__(self, paths, run_type=None):
    
        assert run_type in [systype for systype in ['single_point', 'optmization']], "Run type {} is" \
                                                                                " not implemented.".format(run_type)

        self.project_path = paths.traj_file_path
        self.project_name = paths.project_name
        self.run_type = run_type

        if self.project_path == None:
            self.project_path = os.getcwd() + '/'
        else:
            assert Path(self.project_path).is_dir() is True, 'directory {} does not exist'.format(self.project_path)
            if self.project_path[-1] != '/':
                self.project_path += '/'

        if self.project_name == None:
            self.project_name = 'cp2k_direct'
            
        os.chdir(self.project_path)
        
        if Path(self.project_path + self.project_name).is_dir() is True:
            os.chdir(self.project_name)
        else:
            os.system('mkdir '+self.project_name)
            os.chdir(self.project_name)

        # other params
        self.energy = None
        self.forces = None
        self.charges = None
        self.bsse_total = None
        self.bsse_free = None

    def generate_cp2k_input_file(self, cp2k_input: str):
        """
        writes the .inp control file for cp2k

        Parameters
        ----------
        cp2k_input : str
            all the commands and strings that go into a cp2k .inp file (see cp2k doc)
        """

        cp2k_input_file = open(self.project_path+self.project_name+self.run_type+'.inp', 'w')
        cp2k_input_file.write(cp2k_input)
        cp2k_input_file.close()

    def run_cp2k(self, omp_threads: int, cp2k_binary_name: str, path: str):
        """
        starts the cp2k process and feeds the input file

        Parameters
        ----------
        omp_threads : int
            set the number of OpenMP threads to use
        cp2k_binary_name : str
            name by which the cp2k binary is called on your machine, e.g. cp2k.ssmp or cp2k.psmp
        path : str
            directory in which the calculation is run
        """

        assert type(omp_threads) == int, 'invalid number of OMP threads'
        assert os.system('which '+str(cp2k_binary_name)) == 0, 'cp2k binary with name {} not found'\
                                                                .format(cp2k_binary_name)

        os.chdir(path)
        os.system('export OMP_NUM_THREADS='+str(omp_threads))
        os.system(str(cp2k_binary_name)+' -o '+str(self.project_name)+'_'+str(self.run_type)+'.out '
                  +str(self.project_name)+'_'+str(self.run_type)+'.inp')


    def extract_bsse(self, read_lines):
        """
        reads the basis set superposition error from the .out file
        
        Parameters
        ----------
        read_lines : 
        	read-in lines from .out file in str format
        """

        bsse_line = [index for index, string in enumerate(read_lines) if 'BSSE RESULTS' in string]
        self.bsse_total = float(re.findall(r"[-+]?(?:\d*\.*\d+)", read_lines[bsse_line+2])[0])

        bsse_free_line = [index for index, string in enumerate(read_lines) if 'BSSE-free interaction energy:' in string]
        self.bsse_free = float(re.findall(r"[-+]?(?:\d*\.*\d+)", read_lines[bsse_free_line])[0])

    def extract_energy(self, read_lines):
        """
        reads the energy from the .out file
        
        Parameters
        ----------
        read_lines : 
        	read-in lines from .out file in str format
        """

        energy_line = [index for index, string in enumerate(read_lines) if 'ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]:'
                       in string]

        self.energy = float(re.findall(r"[-+]?(?:\d*\.*\d+)", read_lines[energy_line[-1]])[0]) # Hartree 

    def extract_forces(self, n_atoms, read_lines):

        """
        reads the relevant forces from the .out file

        Parameters
        ----------
        n_atoms : int
            number of all atoms
        read_lines : 
            read-in lines from .out file in str format
        """

        forces_start = [index for index, string in enumerate(read_lines) if 'ATOMIC FORCES in [a.u.]' in string]
        self.forces = np.loadtxt(self.project_path+self.project_name+self.run_type+'.out', skiprows=forces_start[-1]+3,
                             max_rows=n_atoms, usecols=(3,4,5), dtype=float) # Hartree/Bohr
