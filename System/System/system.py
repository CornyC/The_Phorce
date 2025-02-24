#### package imports ####
import copy
import numpy as np
import Coord_Toolz.mdanalysis as ct
from .paths import *
import MDAnalysis as mda
from MDAnalysis.topology.PSFParser import PSFParser
import os, re
from ASE_interface.ase_calculation import *
import os
from itertools import combinations, permutations, product


class Molecular_system:
    os.environ['OMP_NUM_THREADS'] = '6'
    """
    Contains properties of molecular system that is to be parametrized.

    Parameters
    ----------
    parametrization_type : str
        'total_properties' or 'net_properties'
    parametrization_method : str
        'energy', 'forces', or 'energy&forces'

    other parameters:
    openmm_systems : OMM_interface.openmm object
        OpenMM system instance
    ini_coords: numpy array of shape (n_conformations, n_atoms, 3)
        coordinates of the sampled structures in Angström
    n_atoms : int
        number of atoms
    n_conformations : int
        number of conformations aka sampled structures
    ase_sys : ASE_interface.ase_calculation object
        ASE calculation instance
    opt_coords: numpy array of shape (n_conformations, n_atoms, 3)
        coordinates of the sampled structures in Angström after QM geometry optimization
    qm_charges : numpy array
        charges of each atom calculated by QM method
    mm_charges : numpy array
        charges from the classical force field
    qm_forces: numpy array
        forces on each atom of the ini_coords evaluated by the quantum chemical method for each sampled structure
    qm_energies: numpy array
        potential energies of each sampled structure evaluated by the quantum chemical method
    mm_forces: numpy array
        forces on each atom of the ini_coords evaluated by the classical MD method for each sampled structure
    mm_energies: numpy array
        potential energies of each sampled structure evaluated by the classical MD method
    eqm_bsse: numpy array
        Basis set superposition error
    ...net... : numpy array
        subtracted net property (e.g. between 2 molecules or molecule and solvent)
    weights : float
        weights to weigh property
    dupes : dict of list of ints
        e.g. {'all': {'OG311': [3, 4, 14, 15], 'OG2P1': [5], 'OG303': [1, 12]},
              'mol2': {'OG311': [3, 4], 'OG2P1': [5], 'OG303': [1]},
              'mol1': {'OG311': [3, 4], 'OG2P1': [5], 'OG303': [1]}} 
        containing duplicate atom types and their atom indices
    interaction_dupes : nested dict
        contains atom index combinations consisting of duplicate atom types only and their positions (indices) in ff_optimizable[sys_type][force_group]
    slice_list : dict
        defines atoms of interest based on psf topology. Applied in reduce_ff_optimizable. 
    reduced_indexed_ff_optimizable : dict of np.arrays
        parameters from ff_optimizable and their corresponding system atom indices (column 0) and atom types (column 1)
    reduced_ff_otimizable_values : dict of np.arrays
        parameters from ff_optimizable. Arrays have the same length as the ones in reduced_indexed_ff_optimizable
    vectorized_reduced_ff_optimizable_values : dict of 1d np.arrays
        dictionary of flattened parameters
    vectorized_parameters : np.array
        all selected parameters flattened into one long vector
    scaling_factors : np.array
        scale the parameters to roughly the same magnitude, array has same length as vectorized_parameters
    scaled_parameters : np.array
        vectorized parameters extracted from ff_optimizable scaled to the same magnitude by scaling_factors, same shape as vectorized_parameters
    lower_constraints : dict of np.arrays
        holds lower limits of parameters for scaling & constraining :)
    upper_constraints : dict of np.arrays
        holds upper limits of parameters for scaling & constraining
    """

    def __init__(self, parametrization_type: str, parametrization_method: str):
        """
        Initialize object with desired settings.
        """
        self.paths = None

        self.parametrization_types = ('total_properties', 'net_properties')
        self.parametrization_methods = ['energy', 'forces', 'energy&forces']

        self.openmm_systems = {'all': None,
                              'nosol': None,
                              'mol1': None,
                              'mol2': None}

        # read-in params
        self.parametrization_type = parametrization_type
        self.parametrization_method = parametrization_method

        assert self.parametrization_type in [ptype for ptype in self.parametrization_types], "Parametrization" \
                                                        " of type {} is not implemented.".format(parametrization_type)
        assert self.parametrization_method in [pmeth for pmeth in self.parametrization_methods], "Parametrization" \
                                                        " method {} is not implemented.".format(parametrization_method)
        
        # init data storage

        self.ini_coords = copy.deepcopy(self.openmm_systems)
        self.n_atoms = copy.deepcopy(self.openmm_systems)
        self.n_conformations = None
        self.ase_sys = copy.deepcopy(self.openmm_systems)
        self.opt_coords = copy.deepcopy(self.openmm_systems)

        self.mm_charges = copy.deepcopy(self.openmm_systems)
        self.qm_charges = copy.deepcopy(self.openmm_systems)

        self.mm_energies = copy.deepcopy(self.openmm_systems)
        self.qm_energies = copy.deepcopy(self.openmm_systems)
        self.eqm_bsse = copy.deepcopy(self.openmm_systems)

        self.mm_forces = copy.deepcopy(self.openmm_systems)
        self.mm_net_forces = None
        self.qm_forces = copy.deepcopy(self.openmm_systems)
        self.qm_net_forces = None

        self.weights = None

        self.slice_list = None
        self.dupes = None
        self.interaction_dupes = None
        self.reduced_indexed_ff_optimizable = None
        self.reduced_ff_optimizable_values = None
        self.vectorized_reduced_ff_optimizable_values = None
        self.vectorized_parameters = None
        self.scaling_factors = None
        self.scaled_parameters = None

        self.constraints = None

    def set_ini_coords(self, MDA_reader_object):
        """
        reads in the initial atoms & coordinates from file(s) using the MDA_reader, stores them in Molecular_system,
        and acquires properties derived from them based on the settings in Molecular_system.  

        Parameters
        ----------
        MDA_reader_object: Coord_Toolz.mdanalysis.MDA_reader object
            Contains MDA Universe and atomgroups
        """

        coords = ct.get_coords(MDA_reader_object.universes['all'].atoms)
        self.n_conformations = len(coords) 
        self.n_atoms['all'] = len(MDA_reader_object.universes['all'].atoms)

        self.ini_coords['all'] = coords

        assert self.paths is not None, 'Molecular_system.paths not set'

        assert self.paths.mm_traj is not None, 'Cnformations trajectory not found.'

        assert self.paths.mm_crd is not None, 'Coordinate file not found.'

        assert self.paths.mm_top is not None, 'Topology fiel not found.'

        if MDA_reader_object.universes['nosol'] is not None:

            nosol_coords = ct.get_coords(MDA_reader_object.universes['nosol'])
            self.ini_coords['nosol'] = nosol_coords
            self.n_atoms['nosol'] = len(MDA_reader_object.universes['nosol'])

            assert self.paths.mm_nosol_crd is not None, 'Coordinate file for molecule '\
                                                        'w/o solvent not found.'
            assert self.paths.mm_nosol_top is not None, 'Topology file for molecule '\
                                                        'w/o solvent not found.'

        elif MDA_reader_object.universes['mol1'] is not None:

            mol1_coords = ct.get_coords(MDA_reader_object.universes['mol1'])
            self.ini_coords['mol1'] = mol1_coords
            self.n_atoms['mol1'] = len(MDA_reader_object.universes['mol1'])
            mol2_coords = ct.get_coords(MDA_reader_object.universes['mol2'])          
            self.ini_coords['mol2'] = mol2_coords
            self.n_atoms['mol2'] = len(MDA_reader_object.universes['mol2'])

            assert self.paths.mm_mol1_crd is not None, 'Coordinate file for molecule 1 '\
                                                        'w/o molecule 2 not found.'
            assert self.paths.mm_mol1_top is not None, 'Topology file for molecule '\
                                                        'w/o molecule 2 not found.'
            assert self.paths.mm_mol2_crd is not None, 'Cordinate file for molecule 2 '\
                                                        'w/o molecule 1 not found'
            assert self.paths.mm_mol2_top is not None, 'Topology file for molecule 2 '\
                                                        'w/o molecule 1 not found.'


    def read_external_file(path: str, filename: str):

        """
        standard line-by-line reader for human readable files

        Parameters
        ----------
        path : str
            path to the dirctory where the file can be found
        filename : str
            name of the file including extension

        returns:
            the lines of the file in str format
        """

        if path[-1] != '/':
            path = path[:-1]

        f = open(path + filename, 'r')
        read = []
        for index, line in enumerate(f.readlines()):
            line = line.strip()
            read.append(line)
        f.close()

        return read

    def read_qm_charges(self, read_lines, charge_type, path, outfilename, sys_type, n):

        """
        reads cp2k .out files and collects the calculated atom charges based on the type


        Parameters
        -----------
        read_lines : 
            read-in file content as str
        charge_type : str
            one of the following: 'Mulliken', 'Hirshfeld', 'RESP'     
        path : str
            path to externally cp2k optimized structures
        outfilename : str
            equal to <cp2k>.out
        sys_type : str
            'all' or 'nosol' or 'mol1' or 'mol2'
        n : int
            frame number or conformation number    

        returns: 
            charges (qm charges) as numpy arrary   
        """

        read = read_lines

        assert charge_type in ['Mulliken', 'Hirshfeld', 'RESP'], 'invalid charge_type {}'.format(charge_type)

        if charge_type == 'Mulliken':
            mullken_start = [index for index, string in enumerate(read) if 'Mulliken Population Analysis' in string]
            charges = np.loadtxt(path + '/frame' + str(n) + '/' + outfilename + '.out',
                                skiprows = mullken_start[0] + 3,
                                max_rows = self.ini_coords[sys_type].shape[1], usecols=4, dtype=float)

        elif charge_type == 'Hirshfeld':
            hirshfeld_start = [index for index, string in enumerate(read) if 'Hirshfeld Charges' in string]
            charges = np.loadtxt(path + '/frame' + str(n) + '/' + outfilename + '.out',
                                skiprows = hirshfeld_start[0] + 3, max_rows = self.ini_coords[sys_type].shape[1],
                                usecols=5, dtype=float)

        elif charge_type == 'RESP':
            resp_start = [index for index, string in enumerate(read) if 'RESP charges:' in string]
            charges = np.loadtxt(path + '/frame' + str(n) + '/' + outfilename + '.out',
                                skiprows = resp_start[0] + 3,
                                max_rows = self.ini_coords[sys_type].shape[1], usecols=3, dtype=float)
                
        return charges
    
    def read_qm_energies_forces(self, sys_type:str, path: str, filename: str, outfilename: str, engine_type: str):

        """
        Useful if cp2k calculations have been run externally (e.g. on a cluster).
        reads cp2k .out files and extracts energies and forces from it. Can also import 
        optimized coordinates for sys_type = 'all'.

        Parameters
        ----------
        sys_type : str
            'all' or 'nosol' or 'mol1' or 'mol2' 
        path : str
            path to externally cp2k optimized structures
        filename : str
            equal to &GLOBAL>PROJECT str in cp2k.inp file
        outfilename : str
            equal to <cp2k>.out
        engine_type : str
            'ase' or 'cp2k_direct'

        self.ini_coords : unoptimized coordinates of conformations
        self.n_conformations : number of conformations
        
        sets:
            self.qm_forces : quantum forces
            self.qm_energies : quantum energies
            (self.opt_coords : geometry-optimized coordinates of conformations)
        """
        assert engine_type in [etype for etype in ['ase', 'cp2k_direct']], "engine_type {} is not supported.".format(engine_type)

        if path[-1] != '/':
            path += '/'

        #### init arrays ####
        self.qm_forces[sys_type] = np.zeros((len(self.ini_coords[sys_type]), self.ini_coords[sys_type].shape[1], 3))
        self.qm_energies[sys_type] = np.zeros((len(self.ini_coords[sys_type]), 1))
        if sys_type == 'all':
            self.opt_coords[sys_type] = np.zeros((len(self.ini_coords[sys_type]), self.ini_coords[sys_type].shape[1], 3))

        #### fill the arrays ####

        for n in range(self.n_conformations): 

            if engine_type is 'cp2k_direct':

                read = read_external_file(path+'frame'+str(n)+'/', outfilename+'.out')

                energy_line = [index for index, string in enumerate(read) if
                            'ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]:'
                            in string]
                
                if len(energy_line) == 0:
                    raise ValueError('ENERGY not found in frame'+str(n)+'/'+outfilename+'.out')

                energy = float(re.findall(r"[-+]?(?:\d*\.*\d+)", read[energy_line[-1]])[0])

                forces_start = [index for index, string in enumerate(read) if 'ATOMIC FORCES in [a.u.]' in string]
                forces = np.loadtxt(path + 'frame' + str(n) + '/' + outfilename + '.out',
                                        skiprows = forces_start[0] + 3,
                                        max_rows = self.ini_coords[sys_type].shape[1], usecols=(3, 4, 5), dtype=float)
                
            elif engine_type is 'ase':

                forces = np.genfromtxt(path + 'frame' + str(n) + '/forces_energy_'+outfilename+'_frame'+str(n)+'.txt', skip_header=1)
               
                f = open(path + 'frame' + str(n) + '/forces_energy_'+outfilename+'_frame'+str(n)+'.txt', 'r')
                read = []
                for i, line in enumerate(f.readlines()):
                    line = line.strip('# E:\n')
                    if i == 0:
                        read.append(line)
                f.close()
                energy = float(read[0])

            self.qm_forces[sys_type][n, :, :] = forces
            self.qm_energies[sys_type][n] = energy

            if sys_type == 'all':

                u = mda.Universe(path + 'frame' + str(n) + '/' + filename + '-pos-1.xyz')
                coords = ct.get_coords(u.atoms)[-1]
                self.opt_coords[sys_type][n, :, :] = coords


    def read_qm_charges_energies_forces_optcoords(self, sys_type:str, path: str, filename: str, outfilename: str, charge_type: str):

        """
        Useful if cp2k calculations have been run externally (e.g. on a cluster).
        reads cp2k .out files and extracts charges, energies, and forces from it

        Parameters
        ----------
        sys_type : str
            'all' or 'nosol' or 'mol1' or 'mol2' 
        path : str
            path to externally cp2k optimized structures
        filename : str
            equal to &GLOBAL>PROJECT str in cp2k.inp file
        outfilename : str
            equal to <cp2k>.out
        charge_type : str
            one of the following: 'Mulliken', 'Hirshfeld', 'RESP'

        self.ini_coords : unoptimized coordinates of conformations
        self.n_conformations : number of conformations
        
        sets:
            self.opt_coords : geometry-optimized coordinates of conformations
            self.qm_forces : quantum forces
            self.qm_energies : quantum energies
            self.qm_charges : quantum charges
        """
        if path[-1] != '/':
            path += '/'

        #### init arrays ####
        self.opt_coords[sys_type] = np.zeros(self.ini_coords[sys_type].shape)
        self.qm_forces[sys_type] = np.zeros((len(self.ini_coords[sys_type]), self.ini_coords[sys_type].shape[1], 3))
        self.qm_energies[sys_type] = np.zeros((len(self.ini_coords[sys_type]), 1))
        self.qm_charges[sys_type] = np.zeros((len(self.ini_coords[sys_type]), self.ini_coords[sys_type].shape[1]))

        #### fill the arrays ####

        for n in range(self.n_conformations): 

            read = read_external_file(path+'frame'+str(n)+'/', outfilename+'.out')

            energy_line = [index for index, string in enumerate(read) if
                           'ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]:'
                           in string]
            
            if len(energy_line) == 0:
                raise ValueError('ENERGY not found in frame'+str(n)+'/'+outfilename+'.out')

            energy = float(re.findall(r"[-+]?(?:\d*\.*\d+)", read[energy_line[-1]])[0])

            forces_start = [index for index, string in enumerate(read) if 'ATOMIC FORCES in [a.u.]' in string]
            forces = np.loadtxt(path + 'frame' + str(n) + '/' + outfilename + '.out',
                                     skiprows = forces_start[0] + 3,
                                     max_rows = self.ini_coords[sys_type].shape[1], usecols=(3, 4, 5), dtype=float)
            
            charges = self.read_qm_charges(read, charge_type, path, outfilename, sys_type, n)

            self.qm_forces[sys_type][n, :, :] = forces
            self.qm_energies[sys_type][n] = energy
            self.qm_charges[sys_type][n, :] = charges

            if sys_type == 'all':

                u = mda.Universe(path + 'frame' + str(n) + '/' + filename + '-pos-1.xyz')
                coords = ct.get_coords(u.atoms)[-1]
                self.opt_coords[sys_type][n, :, :] = coords



    def generate_qm_energies_forces(self, atomgroup, atomgroup_name, paths, cp2k_inp, sys_type):

        """
        calculates QM energies and forces w/o geometry optimization. Single-point only.
        Needed for net forces (raw sys forces - mol1 with water forces - mol2 with water forces = net forces)

        Parameters
        ----------
        atomgroup : MDAnalysis AtomGroup object
            e.g. molecule1
        atomgroup_name : str
            e.g. 'molecule1', needed for output files
        paths : paths.Paths object
            contains working dir n stuff
        cp2k_inp : str
            input for the inp parameter of the ASE cp2k calculator w/ all the necessary cp2k control parameters
        sys_type : str
            'all' or 'nosol' or 'mol1' or 'mol2' 

        sets:
            self.qm_forces[sys_type] : quantum forces of atomgroup
            self.qm_energies[sys_type] : quantum energies of atomgroup
        """
        
        #### init arrays ####
        self.qm_forces[sys_type] = np.zeros((len(self.ini_coords[sys_type]), self.ini_coords[sys_type].shape[1], 3))
        self.qm_energies[sys_type] = np.zeros((len(self.ini_coords[sys_type]), 1))

        if Path(paths.working_dir + paths.project_name).is_dir() is True:

            os.chdir(paths.working_dir + paths.project_name)
        
        else:

            os.chdir(paths.working_dir)
            os.system('mkdir '+paths.project_name)
            os.chdir(paths.project_name)

        coords = ct.get_coords(atomgroup)

        #### run the ase calc ####
        for frame_nr, frame in enumerate(coords):

            os.system('mkdir frame'+str(frame_nr))
            os.chdir('frame'+str(frame_nr))

            ase_sys = ASE_system(atomgroup.elements, frame)
            ase_sys.cell = ([16.0, 16.0, 16.0])
            ase_sys.pbc = ([True, True, True])
            ase_sys.construct_atoms_object()
            calc = CP2K(debug=True, basis_set=None,
                        basis_set_file=None,
                        max_scf=None,
                        charge=None,
                        cutoff=None,
                        force_eval_method=None,
                        potential_file=None,
                        poisson_solver=None,
                        pseudo_potential=None,
                        stress_tensor=False,
                        uks=False,
                        xc=None,
                        inp=cp2k_inp,
                        print_level='MEDIUM',
                        command = 'env OMP_NUM_THREADS=6 cp2k_shell.ssmp')
            # TODO: set OMP_NUM_THREADS somewhere else? (can just call os.environ where it's needed -> currently initialized below the main class)
            #optional method to set OMP_NUM_THREADS
            #def set_omp_num_threads(num_threads):
             #   original_value = os.environ.get('OMP_NUM_THREADS', None)
             #    os.environ['OMP_NUM_THREADS'] = str(num_threads)
             #    yield
             # if original_value is not None:
             #   os.environ['OMP_NUM_THREADS'] = original_value
             # else:
             #     del os.environ['OMP_NUM_THREADS']
            ase_sys.atoms.calc = calc
            ase_sys.run_calculation(run_type='single_point')
            np.savetxt('forces_energy_' + atomgroup_name + '_frame' + str(frame_nr) + '.txt', ase_sys.forces,
                       header='E: ' + str(ase_sys.energy))
            outstr = 'cp cp2k.out ' + atomgroup_name + '_frame' + str(frame_nr) + '.out'
            os.system(outstr)

            # grab forces
            self.qm_forces[sys_type][frame_nr, :, :] = ase_sys.forces
            
            # grab energies
            self.qm_energies[sys_type][frame_nr] = ase_sys.energy

            os.system('rm cp2k.out')
            os.system('pkill cp2k_shell.ssmp')
            os.system('touch cp2k.out')
            os.chdir('..')


    def generate_qm_charges_energies_forces(self, atomgroup, atomgroup_name, paths, cp2k_inp, charge_type, sys_type):

        """
        calculates QM charges, energies, and forces w/o geometry optimization. Single-point only.
        Needed for net forces (raw sys forces - mol1 with water forces - mol2 with water forces = net forces)

        Parameters
        ----------
        atomgroup : MDAnalysis AtomGroup object
            e.g. molecule1
        atomgroup_name : str
            e.g. 'molecule1', needed for output files
        paths : paths.Paths object
            contains working dir n stuff
        cp2k_inp : str
            input for the inp parameter of the ASE cp2k calculator w/ all the necessary cp2k control parameters. 
            Don't forget to set up the charge calculation in there!
        charge_type : str
            one of the following: 'Mulliken', 'Hirshfeld', 'RESP'
        sys_type : str
            'all' or 'nosol' or 'mol1' or 'mol2' 

        sets:
            self.qm_charges[sys_type] : quantum charges of atomgroup
            self.qm_forces[sys_type] : quantum forces of atomgroup
            self.qm_energies[sys_type] : quantum energies of atomgroup
        """

        #### init arrays ####
        self.qm_forces[sys_type] = np.zeros((len(self.ini_coords[sys_type]), self.ini_coords[sys_type].shape[1], 3))
        self.qm_energies[sys_type] = np.zeros((len(self.ini_coords[sys_type]), 1))
        self.qm_charges[sys_type] = np.zeros((len(self.ini_coords[sys_type]), self.ini_coords[sys_type].shape[1]))


        if Path(paths.working_dir + paths.project_name).is_dir() is True:

            os.chdir(paths.working_dir + paths.project_name)
        
        else:

            os.chdir(paths.working_dir)
            os.system('mkdir '+paths.project_name)
            os.chdir(paths.project_name)

        coords = ct.get_coords(atomgroup)

        #### run the ase calc ####
        for frame_nr, frame in enumerate(coords):

            os.system('mkdir frame'+str(frame_nr))
            os.chdir('frame'+str(frame_nr))

            ase_sys = ASE_system(atomgroup.elements, frame)
            ase_sys.cell = ([16.0, 16.0, 16.0])
            ase_sys.pbc = ([True, True, True])
            ase_sys.construct_atoms_object()
            calc = CP2K(basis_set=None,
                        basis_set_file=None,
                        max_scf=None,
                        charge=None,
                        cutoff=None,
                        force_eval_method=None,
                        potential_file=None,
                        poisson_solver=None,
                        pseudo_potential=None,
                        stress_tensor=False,
                        uks=False,
                        xc=None,
                        inp=cp2k_inp,
                        print_level='MEDIUM',
                        command = 'env OMP_NUM_THREADS=6 cp2k_shell.ssmp')
            # TODO: set OMP_NUM_THREADS somewhere else? (can just call os.environ where it's needed -> currently initialized below the main class)
            #optional method to set OMP_NUM_THREADS
            #def set_omp_num_threads(num_threads):
             #   original_value = os.environ.get('OMP_NUM_THREADS', None)
             #    os.environ['OMP_NUM_THREADS'] = str(num_threads)
             #    yield
             # if original_value is not None:
             #   os.environ['OMP_NUM_THREADS'] = original_value
             # else:
             #     del os.environ['OMP_NUM_THREADS']
            ase_sys.atoms.calc = calc
            ase_sys.run_calculation(run_type='single_point')
            np.savetxt('forces_energy_' + atomgroup_name + '_frame' + str(frame_nr) + '.txt', ase_sys.forces,
                       header='E: ' + str(ase_sys.energy))
            
            # grab charges
            path = os.getcwd()
            read = read_external_file(path, 'cp2k.out')
            charges = self.read_qm_charges(read, charge_type, path, 'cp2k.out', sys_type, frame_nr)
            self.qm_charges[sys_type][frame_nr, :] = charges

            outstr = 'cp cp2k.out ' + atomgroup_name + '_frame' + str(frame_nr) + '.out'
            os.system(outstr)

            # grab forces
            self.qm_forces[sys_type][frame_nr, :, :] = ase_sys.forces
            
            # grab energies
            self.qm_energies[sys_type][frame_nr] = ase_sys.energy

            os.system('rm cp2k.out')
            os.system('pkill cp2k_shell.ssmp')
            os.system('touch cp2k.out')
            os.chdir('..')

    def generate_qm_energies_forces_optcoords(self, atomgroup, atomgroup_name, paths, cp2k_inp, sys_type): 
        """
        calculates QM forces and energies of the optimized conformations using ASE. Runs a geometry optimization.

        Parameters
        ----------
        atomgroup : MDAnalysis AtomGroup object
            e.g. molecule1
        atomgroup_name : str
            e.g. 'molecule1', needed for output files
        paths : paths.Paths object
            contains working dir n stuff
        cp2k_inp : str
            input for the inp parameter of the ASE cp2k calculator w/ all the necessary cp2k control parameters. 
            Don't forget to set up the charge calculation in there!
        sys_type : str
            'all' or 'nosol' or 'mol1' or 'mol2' 

        sets:
            self.qm_forces[sys_type] : quantum forces of atomgroup after geometry opt
            self.qm_energies[sys_type] : quantum energies of atomgroup after geom opt
            self.opt_coords[sys_type] : coordinates after DFT geometry optimization
        """

        #### init arrays ####
        self.qm_forces[sys_type] = np.zeros((len(self.ini_coords[sys_type]), self.ini_coords[sys_type].shape[1], 3))
        self.qm_energies[sys_type] = np.zeros((len(self.ini_coords[sys_type]), 1))        
        self.opt_coords[sys_type] = np.zeros((len(self.ini_coords[sys_type]), self.ini_coords[sys_type].shape[1], 3))
        self.qm_charges[sys_type] = np.zeros((len(self.ini_coords[sys_type]), self.ini_coords[sys_type].shape[1]))

        if Path(paths.working_dir + paths.project_name).is_dir() is True:

            os.chdir(paths.working_dir + paths.project_name)
        
        else:

            os.chdir(paths.working_dir)
            os.system('mkdir '+paths.project_name)
            os.chdir(paths.project_name)

        coords = ct.get_coords(atomgroup)

        #### run the ase calc ####
        for frame_nr, frame in enumerate(coords):

            os.system('mkdir frame'+str(frame_nr))
            os.chdir('frame'+str(frame_nr))

            ase_sys = ASE_system(atomgroup.elements, frame)
            ase_sys.cell = ([16.0, 16.0, 16.0])
            ase_sys.pbc = ([True, True, True])
            ase_sys.construct_atoms_object()
            calc = CP2K(basis_set=None,
                        basis_set_file=None,
                        max_scf=None,
                        charge=None,
                        cutoff=None,
                        force_eval_method=None,
                        potential_file=None,
                        poisson_solver=None,
                        pseudo_potential=None,
                        stress_tensor=False,
                        uks=False,
                        xc=None,
                        inp=cp2k_inp,
                        print_level='MEDIUM',
                        command = 'env OMP_NUM_THREADS=6 cp2k_shell.ssmp')
             # TODO: set OMP_NUM_THREADS somewhere else? (can just call os.environ where it's needed -> currently initialized below the main class)
            #optional method to set OMP_NUM_THREADS
            #def set_omp_num_threads(num_threads):
             #   original_value = os.environ.get('OMP_NUM_THREADS', None)
             #    os.environ['OMP_NUM_THREADS'] = str(num_threads)
             #    yield
             # if original_value is not None:
             #   os.environ['OMP_NUM_THREADS'] = original_value
             # else:
             #     del os.environ['OMP_NUM_THREADS']
            ase_sys.atoms.calc = calc
            ase_sys.run_calculation(run_type='optimization')
            np.savetxt('forces_energy_' + atomgroup_name + '_frame' + str(frame_nr) + '.txt', ase_sys.forces,
                       header='E: ' + str(ase_sys.energy))
            outstr = 'cp cp2k.out ' + atomgroup_name + '_frame' + str(frame_nr) + '.out'
            os.system(outstr)

            # grab forces
            self.qm_forces[sys_type][frame_nr, :, :] = ase_sys.forces
            
            # grab energies
            self.qm_energies[sys_type][frame_nr] = ase_sys.energy

            # grab optcoords
            self.opt_coords[sys_type][frame_nr, :, :] = ase_sys.opt_coords

            os.system('rm cp2k.out')
            os.system('pkill cp2k_shell.ssmp')
            os.system('touch cp2k.out')
            os.chdir('..') 

    def generate_qm_charges_energies_forces_optcoords(self, paths, MDA_reader, cp2k_input: str, omp_threads: str, cp2k_binary_name: str, charge_type: str, 
                                                      sys_type: str): 

        """
        Calculates QM energies, forces, and charges using native cp2k. 
        Requires topology file and starting coords as well as a valid cp2k input file (please check cp2k doc).
        Only recommended for sys_type = 'all' in order to represent valid biochemical conditions.

        Parameters
        ----------
        paths : paths.Paths object
            contains paths and filenames etc
        MDA_reader : Coord_Toolz.mdanalysis.MDA_reader object
            contains MDA universes w/ atom info, coords, ...
        cp2k_input : str
            all the commands and strings that go into a cp2k .inp file (see cp2k doc). It is recommended to load the
            input into a separate variable beforehand
        omp_threads : int
            number of openMP threads to use for the parallel cp2k calculation
        cp2k_binary_name : str
            name by which the cp2k binary is called on your machine
        charge_type : str
            Options are 'Mulliken', 'Hirshfeld', 'RESP'
        sys_type : str
            'all' or 'nosol' or 'mol1' or 'mol2' 

        self.n_conformations : number of conformations

        sets:
            self.qm_charges[sys_type] : quantum charges of atomgroup after geom opt
            self.qm_forces[sys_type] : quantum forces of atomgroup after geometry opt
            self.qm_energies[sys_type] : quantum energies of atomgroup after geom opt
            self.opt_coords[sys_type] : coordinates after DFT geometry optimization
        """

        from ..Direct_cp2k_calculation.direct_cp2k import Direct_Calculator as cc
    
        #### init arrays ####
        self.qm_forces[sys_type] = np.zeros((len(self.ini_coords[sys_type]), self.ini_coords[sys_type].shape[1], 3))
        self.qm_energies[sys_type] = np.zeros((len(self.ini_coords[sys_type]), 1))        
        self.opt_coords[sys_type] = np.zeros((len(self.ini_coords[sys_type]), self.ini_coords[sys_type].shape[1], 3))



        #### fill the arrays #### 
        for n in range(self.n_conformations):

            #### construct charge calculator object ####
            run_type = 'charges'
            charge_calc = cc(paths, run_type)

            charge_calc.generate_cp2k_input_file(cp2k_input)

            os.system('mkdir frame'+str(n))
            os.chdir('frame'+str(n))
            os.system('cp '+paths.mm_top+' .') # copy the topol file
            os.system('cp '+charge_calc.project_path+charge_calc.project_name+charge_calc.run_type+'.inp .')
            MDA_reader.universes[sys_type].atoms.write('frame'+str(n)+'.pdb', frames=MDA_reader.universes[sys_type].trajectory[n:n+1])

            path = os.getcwd()

            # run the calculation
            charge_calc.run_cp2k(omp_threads, cp2k_binary_name, path)

            # grab results
            read = read_external_file(charge_calc.project_path+'frame'+str(n)+'/', charge_calc.project_name+run_type+'.out')

            charge_calc.charges = self.read_qm_charges(read, charge_type, charge_calc.project_path, charge_calc.project_name+run_type+'.out',
                                                        sys_type, n)

            self.qm_charges[sys_type][n, :] = charge_calc.charges

            charge_calc.extract_energy(read)
            self.qm_energies[sys_type][n] = charge_calc.energy * 2.62549961709828E+03 #Hartree to kJ/mol

            charge_calc.extract_forces(self.ini_coords[sys_type].shape[1], read)
            self.qm_forces[sys_type][n, :, :] = charge_calc.forces * 2.62549961709828E+03 * 10.0 #a.u. to kJ/mol/nm

            os.chdir('..') 
    

    def calculate_qm_net_forces(self): 

        """
        Computes the net forces between 2 molecules by subtracting the forces caused by interaction w/ surrounding
        water. (raw sys forces - mol1 with water forces - mol2 with water forces = net forces)
   
        """

        # determine which data is present and decide on net force type

        empty_sys_types = [sys_type for sys_type, value in self.qm_forces.items() if value is None]     

        if 'nosol' in empty_sys_types:

            if 'all' in empty_sys_types:

                if 'mol1' in empty_sys_types and 'mol2' in empty_sys_types:
                    raise ValueError('No forces present. Run some QM scheme first.')

                elif 'mol1' in empty_sys_types or 'mol2' in empty_sys_types:
                    raise ValueError('Missing data. Cannot subtract empty force.')

            else:
                if 'mol1' in empty_sys_types and 'mol2' in empty_sys_types:
                    print("Attention: qm_net_forces are being set as qm_forces['all']")
                    self.qm_net_forces = self.qm_forces['all']

                elif 'mol1' in empty_sys_types or 'mol2' in empty_sys_types:
                    raise ValueError("Subtraction not possible. Register molecule as 'nosol' instead.")
                
                else:
                    mol1_slice = self.ini_coords['all'].shape[1] - self.ini_coords['mol2'].shape[1]
                    mol2_slice = self.ini_coords['all'].shape[1] - self.ini_coords['mol1'].shape[1]

                    self.qm_net_forces = self.qm_forces['all'][:,:(mol1_slice + mol2_slice),:] - np.concatenate((self.qm_forces['mol1'][:,:mol1_slice,:], self.qm_forces['mol2'][:,:mol2_slice,:]), axis = 1)
                    
        else:
            mol_slice = self.ini_coords['nosol'].shape[1]

            self.qm_net_forces = self.qm_forces['all'][:,:mol_slice,:] - self.qm_forces['nosol']


    def generate_mm_energies_forces(self, sys_type): 
        """
        collects the OpenMM system's classical properties (charges,energies&forces)

        Parameters
        ----------
        sys_type : str
            'all' or 'nosol' or 'mol1' or 'mol2' 

        self.openmm_systems : OpenMM system object

        sets:
            self.mm_energies : classical energies
            self.mm_forces : classical forces
        """

        assert self.openmm_systems[sys_type] is not None, 'Please set up an OpenMM system first and register it in openmm_systems.'

        #### init arrays ####
        self.mm_forces[sys_type] = np.zeros((len(self.ini_coords[sys_type]), self.ini_coords[sys_type].shape[1], 3))
        self.mm_energies[sys_type] = np.zeros((len(self.ini_coords[sys_type]), 1))  

        for frame_nr, frame in enumerate(self.ini_coords[sys_type]):

            epot, forces = self.openmm_systems[sys_type].run_calculation(frame*0.1) # positions are converted from Angström to nm

            # grab forces
            self.mm_forces[sys_type][frame_nr, :, :] = forces
        
            # grab energies
            self.mm_energies[sys_type][frame_nr] = epot

        print('########################################')
        print('# calculated MM F&E of '+str(frame_nr+1)+' frames')
        print('########################################')


    def calculate_mm_net_forces(self):
        
        """
        Computes the net forces between 2 molecules by subtracting the forces caused by interaction w/ surrounding
        water. (raw sys forces - mol1 with water forces - mol2 with water forces = net forces)
   
        """

        # determine which data is present and decide on net force type

        empty_sys_types = [sys_type for sys_type, value in self.mm_forces.items() if value is None]        

        if 'nosol' in empty_sys_types:

            if 'all' in empty_sys_types:

                if 'mol1' in empty_sys_types and 'mol2' in empty_sys_types:
                    raise ValueError('No forces present. Run some MM scheme first.')

                elif 'mol1' in empty_sys_types or 'mol2' in empty_sys_types:
                    raise ValueError('Missing data. Cannot subtract empty force.')

            else:
                if 'mol1' in empty_sys_types and 'mol2' in empty_sys_types:
                    print("Attention: mm_net_forces are being set as mm_forces['all']")
                    self.mm_net_forces = self.mm_forces['all']

                elif 'mol1' in empty_sys_types or 'mol2' in empty_sys_types:
                    raise ValueError("Subtraction not possible. Register molecule as 'nosol' instead.")
                
                else:
                    mol1_slice = self.ini_coords['all'].shape[1] - self.ini_coords['mol2'].shape[1]
                    mol2_slice = self.ini_coords['all'].shape[1] - self.ini_coords['mol1'].shape[1]

                    self.mm_net_forces = self.mm_forces['all'][:,:(mol1_slice + mol2_slice),:] - np.concatenate((self.mm_forces['mol1'][:,:mol1_slice,:], self.mm_forces['mol2'][:,:mol2_slice,:]), axis = 1)
                    
        else:
            mol_slice = self.ini_coords['nosol'].shape[1]

            self.mm_net_forces = self.mm_forces['all'][:,:mol_slice,:] - self.mm_forces['nosol']



    def get_bsse(self, paths, MDA_reader, cp2k_input: str, omp_threads: int, cp2k_binary_name: str, sys_type:str): #TODO: fix this

        """
        calculate the basis set superposition error (BSSE) using native cp2k

        Parameters
        ----------
        paths : paths.Paths object
            contains paths and filenames etc
        MDA_reader : Coord_Toolz.mdanalysis.MDA_reader object
            contains MDA universes w/ atom info, coords, ...
        cp2k_input : str
            all the commands and strings that go into a cp2k .inp file (see cp2k doc). It is recommended to load the
            input into a separate variable beforehand
        omp_threads : int
            number of openMP threads to use for the parallel cp2k calculation
        cp2k_binary_name : str
            name by which the cp2k binary is called on your machine
        sys_type : str
            'all' or 'nosol' or 'mol1' or 'mol2' 

        self.n_conformations : number of conformations

        sets:
            self.eqm_bsse : float, basis set superposition error
        """

        from ..Direct_cp2k_calculation.direct_cp2k import Direct_Calculator as cc

        #### init array ####
        self.eqm_bsse[sys_type] = np.zeros((len(self.ini_coords[sys_type]), 1))        

        #### fill the arrays ####
        for n in range(self.n_conformations):

            #### construct charge calculator object ####
            run_type = 'BSSE'
            bsse_calc = cc(paths, run_type)

            bsse_calc.generate_cp2k_input_file(cp2k_input)

            os.system('mkdir frame'+str(n))
            os.chdir('frame'+str(n))
            os.system('cp '+paths.mm_top+' .') # copy the topol file
            os.system('cp '+bsse_calc.project_path+bsse_calc.project_name+bsse_calc.run_type+'.inp .')
            MDA_reader.universes[sys_type].atoms.write('frame'+str(n)+'.pdb', frames=MDA_reader.universes[sys_type].trajectory[n:n+1])           

            path = os.getcwd()

            #### run calc ####
            bsse_calc.run_cp2k(omp_threads, cp2k_binary_name, path)

            # grab results
            read = read_external_file(bsse_calc.project_path+'frame'+str(n)+'/', bsse_calc.project_name+run_type+'.out')

            bsse_calc.extract_bsse(read)

            self.eqm_bsse[sys_type][n] = bsse_calc.bsse_total * 2.62549961709828E+03  # Hartree to kJ/mol
            os.chdir('..') 

    # def get_dihedrals(self):
            
    def get_mm_charges(self, sys_type: str):
        """
        Copies the charges from the current force field to Molecular_system

        Parameters
        ----------
        sys_type : str
            'all' or 'nosol' or 'mol1' or 'mol2' 
        """
        self.openmm_systems[sys_type].get_charges()
        self.mm_charges[sys_type] = self.openmm_systems[sys_type].charges

    def correct_charges(self, sys_type: str):
        """
        Use if the numerical total charge does not equal the real total charge (due to numerical errors)

        Parameters
        ----------
        sys_type : str
            'all' or 'nosol' or 'mol1' or 'mol2' 

        self.mm_charges : dict containing charges from current force field

        """
        real_charge = sum(self.openmm_systems[sys_type].extracted_ff['NonbondedForce'][0].charge)

        total_charge = sum(self.mm_charges[sys_type])
        print('current total charge = ' + str(total_charge))

        if total_charge != real_charge:

            charge_correction = np.abs(total_charge - real_charge) / len(self.mm_charges[sys_type])
            print('charge correction per atom = ' + str(charge_correction))

            for atom in range(len(self.mm_charges[sys_type])):
                self.mm_charges[sys_type] -= charge_correction

                corrected_total_charge = sum(self.mm_charges[sys_type])

                print('corrected total charge = ' + str(corrected_total_charge))


    def generate_weights(self):
        """
        For now weighs all conformations equally. Can be modified tho.

        sets:
            self.weights
        """

        self.weights = np.ones((self.n_conformations))
        self.weights = self.weights / np.sum(self.weights)

    def get_types_from_psf_topology_file(self):
        """
        extracts force field atom types from psf topology file

        returns array of [int,str] with ff_atom_types (column1) and their atom indices (column0)
        """
        all_atom_types_indices = {k: [] for k in set(self.openmm_systems.keys())}

        for sys_type in self.openmm_systems.keys():

            if (sys_type == 'all' and self.openmm_systems[sys_type] is not None) == True:

                assert self.paths.mm_top is not None, 'Path to system topology file is not configured.'
                assert str(self.paths.mm_top[-3:]) == 'psf', 'Only psf format is supported for system topology.'

                psfparser = PSFParser(self.paths.mm_top)
                topol = psfparser.parse()
                _all_ff_atom_types = topol.types.values
                _all_atom_types_indices = np.vstack((np.linspace(0,len(_all_ff_atom_types)-1,len(_all_ff_atom_types),dtype=int),_all_ff_atom_types)).T
                all_atom_types_indices['all'] = _all_atom_types_indices

            elif (sys_type == 'nosol' and self.openmm_systems[sys_type] is not None) == True:

                assert self.paths.mm_top is not None, 'Path to nosol topology file is not configured.'
                assert str(self.paths.mm_top[-3:]) == 'psf', 'Only psf format is supported for nosol topology.'

                psfparser = PSFParser(self.paths.mm_nosol_top)
                topol = psfparser.parse()
                _all_ff_atom_types = topol.types.values
                _all_atom_types_indices = np.vstack((np.linspace(0,len(_all_ff_atom_types)-1,len(_all_ff_atom_types),dtype=int),_all_ff_atom_types)).T
                all_atom_types_indices['nosol'] = _all_atom_types_indices

            elif (sys_type == 'mol1' and self.openmm_systems[sys_type] is not None) == True:

                assert self.paths.mm_top is not None, 'Path to mol1 topology file is not configured.'
                assert str(self.paths.mm_top[-3:]) == 'psf', 'Only psf format is supported mol1 topology.'

                psfparser = PSFParser(self.paths.mm_mol1_top)
                topol = psfparser.parse()
                _all_ff_atom_types = topol.types.values
                _all_atom_types_indices = np.vstack((np.linspace(0,len(_all_ff_atom_types)-1,len(_all_ff_atom_types),dtype=int),_all_ff_atom_types)).T
                all_atom_types_indices['mol1'] = _all_atom_types_indices

            elif (sys_type == 'mol2' and self.openmm_systems[sys_type] is not None) == True:

                assert self.paths.mm_top is not None, 'Path to mol2 topology file is not configured.'
                assert str(self.paths.mm_top[-3:]) == 'psf', 'Only psf format is supported for mol2 topology.'

                psfparser = PSFParser(self.paths.mm_mol2_top)
                topol = psfparser.parse()
                _all_ff_atom_types = topol.types.values
                _all_atom_types_indices = np.vstack((np.linspace(0,len(_all_ff_atom_types)-1,len(_all_ff_atom_types),dtype=int),_all_ff_atom_types)).T
                all_atom_types_indices['mol2'] = _all_atom_types_indices

        return all_atom_types_indices
    

    def _eliminate_duplicate_atomtypes(self, sliceable_ff_optimizable, sys_type, force_group, ff_atom_types_indices):

        """
        helper function to remove duplicate atom types and their parameter values from selected_ff_optimizable, is called in reduce_ff_optimizable. Only applicable to nonbonded parameters.

        Parameters
        ----------
        sliceable_ff_optimizable : nd np.array
            Contains parameters
        sys_type : str
            one of ['all', 'nosol', 'mol1', 'mol2']
        force_group : str
            Force group name, e.g. HarmonicBondForce
        ff_atom_types_indices : np.array of [int,str]
            ff_atom_types (column1) and their atom indices (column0)


        other (internal) parameters : 
        self.to_be_removed : dict of lists of ints
            Contains indices of atoms within the user-based selection that are duplicates regardimg their atom type
        self.dupes : dict of lists of ints
            Contains atom types and indices of duplicate atom types
        self.interaction_dupes : nested dict
            Contains atom type combinations and their indices (where to find them in ff_optimizable)

        returns : reduced_indexed_ff_opt (np.array, contains atom indices, atom types, and parameters ('NonbondedForce') or an atom types tuple and the parameters (all other force groups))
          and reduced_ff_opt_values (np.array, contains parameters only) 
        """

        if force_group == 'NonbondedForce':

            indexed_sel_ff_opt = np.concatenate((ff_atom_types_indices, sliceable_ff_optimizable), axis=1) # homogeneous np.array

            delete_list = []
            for index, entry in enumerate(indexed_sel_ff_opt):

                if entry[0] in self.to_be_removed[sys_type]:
                    delete_list.append(index)

            reduced_indexed_ff_opt = np.delete(indexed_sel_ff_opt, delete_list, 0)
            reduced_ff_opt_values = reduced_indexed_ff_opt[:,2:]

        elif force_group in ['NBException', 'HarmonicBondForce', 'HarmonicAngleForce', 'PeriodicTorsionForce']:

            number_of_atoms_involved = {'NBException': 2,
                                        'HarmonicBondForce': 2,
                                        'HarmonicAngleForce': 3,
                                        'PeriodicTorsionForce': 4,
                                        }

            atom_types = list(self.dupes[sys_type].keys())
            atom_types.sort()
            atom_combinations = list(combinations(atom_types, r = number_of_atoms_involved[force_group]))
            involved_atoms = {comb: [] for comb in atom_combinations} # looks like {('OG2P1','OG303'): [], ...} for HarmonicBondForce
            dupe_indices = copy.deepcopy(involved_atoms) # will contain location indices

            atom_indices = []
            for atom_type in atom_types:

                atom_indices.append(self.dupes[sys_type][atom_type])

            atom_index_combos = list(combinations(atom_indices, r = number_of_atoms_involved[force_group]))

            for atype_combo_no, atype_combo in enumerate(involved_atoms.keys()):

                possible_combinations = list(product(*atom_index_combos[atype_combo_no]))
                all_permutations = set()
                
                for possible_combo in possible_combinations:

                    all_permutations.update(permutations(possible_combo))

                involved_atoms[atype_combo] = sorted(all_permutations) # ends up looking like {('OG2P1','OG303'): [(5,1),(5,12),(1,5),...], ...} for HarmonicBondForce

            for atype_combo in dupe_indices:

                atom_combos_in_sliceable_ff_opt = sliceable_ff_optimizable[:,:number_of_atoms_involved[force_group]].view('<i8') # sliceable array of atom index tuples

                for atom_tuple in involved_atoms[atype_combo]:

                    for param_no, params in enumerate(atom_combos_in_sliceable_ff_opt):

                        if atom_tuple == tuple(params):

                            dupe_indices[atype_combo].append(param_no) # ends up looking like {('OG2P1','OG303'): [3, 121,...], ...} for HarmonicBondForce

            self.interaction_dupes[sys_type][force_group] = dupe_indices

            reduced_ff_opt_values = []
            reduced_indexed_ff_opt = []

            for atype_combo in self.interaction_dupes[sys_type][force_group].keys():

                reduced_ff_opt_values.append(sliceable_ff_optimizable[:,number_of_atoms_involved[force_group]:][self.interaction_dupes[sys_type][force_group][atype_combo][0]])
                reduced_indexed_ff_opt.append(list(reduced_ff_opt_values[-1]))
                reduced_indexed_ff_opt[-1].insert(0,atype_combo) 

        return reduced_indexed_ff_opt, reduced_ff_opt_values
    

    
    
    def reduce_ff_optimizable(self, slice_list):
        """
        extracts force field parameters from ff_optimizable based on an atom selection broadcasted through slice_list. 
        Also eliminates 'duplicates' of the same atom type.

        Parameters
        ----------
        slice_list : dict of lists of ints
            atom slices for the desired atoms of all systems. Can make use of numpy.r_, e.g. np.r_[1,3:6,12,14:16] for multiple, 
            not connected slices at once. E.g slice_list = {'all': [np.r_[1,3:6,12,14:16]],
                                                            'mol1': [np.r_[1,3:6]],
                                                            'mol2': [np.r_[1,3:6]],
                                                            }

        other (internal) parameters:
            self.openmm_systems[sys_type].ff_optimizable
        
        sets:
            self.reduced_indexed_ff_optimizable[sys_type][force_group] : dict of dict of np.arrays containing atom indices, atom types, and ff term values (nonbonded)
                or line indices, atom indices, and ff term values (bonded); not mutable
            self.reduced_ff_optimizable_values[sys_type][force_group] : dict of dict of np.arrays containing only the ff term values (nonbonded or bonded); mutable
        """

        reshape_column_indices = {'HarmonicBondForce': 4,
                    'HarmonicAngleForce': 5,
                    'PeriodicTorsionForce': 7,
                    'NonbondedForce': 3,
                    'NBException': 5}

        self.slice_list = slice_list
        
        # define and init the dictionaries by sys_type
        self.reduced_indexed_ff_optimizable = {k: [] for k in sorted(set(self.openmm_systems.keys())) if self.openmm_systems[k] != None} 
        self.reduced_ff_optimizable_values = {k: [] for k in sorted(set(self.openmm_systems.keys())) if self.openmm_systems[k] != None}
        self.dupes = {k: [] for k in sorted(set(self.openmm_systems.keys())) if self.openmm_systems[k] != None}
        self.interaction_dupes = {k: {} for k in sorted(set(self.openmm_systems.keys())) if self.openmm_systems[k] != None}
        self.to_be_removed = {k: [] for k in sorted(set(self.openmm_systems.keys())) if self.openmm_systems[k] != None}

        for sys_type in self.reduced_indexed_ff_optimizable.keys():

            # define and init the nested dictionaries by force_key
            self.reduced_indexed_ff_optimizable[sys_type] = {k: [] for k in sorted(set(self.openmm_systems[sys_type].ff_optimizable.keys()))} # not mutable
            self.reduced_ff_optimizable_values[sys_type] = {k: [] for k in sorted(set(self.openmm_systems[sys_type].ff_optimizable.keys()))} # mutable

            # find duplicate atom types
            ff_atom_types_indices = self.get_types_from_psf_topology_file()[sys_type][slice_list[sys_type]][0]
            self.dupes[sys_type] = find_same_type(ff_atom_types_indices)

            self.to_be_removed[sys_type] = []
            for atom_type in self.dupes[sys_type].keys():

                for index in self.dupes[sys_type][atom_type][1:]:
                    self.to_be_removed[sys_type].append(index)

            self.to_be_removed[sys_type].sort()

            # filter out duplicate atom types and reduce parameter set 
            for force_group in self.openmm_systems[sys_type].ff_optimizable.keys():

                for _array in self.openmm_systems[sys_type].ff_optimizable[force_group]:

                    if force_group == 'NonbondedForce':

                        # atom selection-based slicing
                        selected_ff_optimizable = _array[slice_list[sys_type]][0]
                        sliceable_ff_optimizable = selected_ff_optimizable.view('<f8').reshape(len(selected_ff_optimizable), reshape_column_indices[force_group]) # turns 1d recarray into sliceable nd array

                        reduced_indexed_ff_opt, reduced_ff_opt_values = self._eliminate_duplicate_atomtypes(sliceable_ff_optimizable, sys_type, force_group, ff_atom_types_indices)

                    elif force_group in ['NBException', 'HarmonicBondForce', 'HarmonicAngleForce', 'PeriodicTorsionForce']: 

                        sliceable_ff_optimizable = _array.view('<f8').reshape(len(_array), reshape_column_indices[force_group]) # turns 1d recarray into sliceable nd array
                        
                        reduced_indexed_ff_opt, reduced_ff_opt_values = self._eliminate_duplicate_atomtypes(sliceable_ff_optimizable, sys_type, force_group, ff_atom_types_indices)
                   
                    self.reduced_indexed_ff_optimizable[sys_type][force_group].append(reduced_indexed_ff_opt)
                    self.reduced_ff_optimizable_values[sys_type][force_group].append(reduced_ff_opt_values)   
                   
        if list(sorted(self.reduced_indexed_ff_optimizable.keys())) == ['all', 'nosol']:

            for fg_nr, force_group in enumerate(self.reduced_indexed_ff_optimizable['all'].keys()):

                if force_group == 'NonbondedForce':

                    if not sorted(set(self.reduced_indexed_ff_optimizable['all'][force_group][0][:,1])) == sorted(set(self.reduced_indexed_ff_optimizable['nosol'][force_group][0][:,1])):

                        raise ValueError('Sliced atoms do not match. Check slice_list.')
                    
                elif force_group in ['NBException', 'HarmonicBondForce', 'HarmonicAngleForce', 'PeriodicTorsionForce']: 

                    if not sorted(set(np.array(self.reduced_indexed_ff_optimizable['all'][force_group][0], dtype = object)[:,0])) == \
                            sorted(set(np.array(self.reduced_indexed_ff_optimizable['nosol'][force_group][0], dtype = object)[:,0])):

                        raise ValueError('Sliced atoms do not match. Check slice_list.')

        elif list(sorted(self.reduced_indexed_ff_optimizable.keys())) == ['all', 'mol1', 'mol2']:

            for fg_nr, force_group in enumerate(self.reduced_indexed_ff_optimizable['all'].keys()): 

                if force_group in 'NonbondedForce':

                    if not sorted(set(self.reduced_indexed_ff_optimizable['all'][force_group][0][:,1])) == sorted(set(np.concatenate((self.reduced_indexed_ff_optimizable['mol1'][force_group][0][:,1] \
                                                                                                  , self.reduced_indexed_ff_optimizable['mol2'][force_group][0][:,1])))):

                        raise ValueError('Sliced atoms do not match. Check slice_list.')
                    
                elif force_group in ['NBException', 'HarmonicBondForce', 'HarmonicAngleForce', 'PeriodicTorsionForce']: 

                    if not sorted(set(np.array(self.reduced_indexed_ff_optimizable['all'][force_group][0], dtype = object)[:,0])) == \
                        sorted(set(np.concatenate((np.array(self.reduced_indexed_ff_optimizable['mol1'][force_group][0], dtype = object)[:,0] \
                                                    , np.array(self.reduced_indexed_ff_optimizable['mol2'][force_group][0], dtype = object)[:,0])))):

                        raise ValueError('Sliced atoms do not match. Check slice_list.')
                        
                    
    def expand_reduced_parameters(self): 
        """
        puts parameter values from reduced_ff_optimizable_values back into ff_optimizable

        internal parameters:
            self.reduced_ff_optimizable_values
            self.reduced_indexed_ff_optimizable
            self.dupes
            self.interaction_dupes
            self.openmm_systems[sys_type].ff_optimizable

        sets:
            self.openmm_systems[sys_type].ff_optimizable
        """

        for force_group in self.reduced_ff_optimizable_values['all'].keys():

            if force_group == 'NonbondedForce':
                #print(force_group)

                for array_no, _array in enumerate(self.reduced_indexed_ff_optimizable['all'][force_group]):

                    for line_no, parameter_line in enumerate(_array):

                        for atom_type in self.dupes['all'].keys():

                            if atom_type == parameter_line[1]:

                                for atom_index in self.dupes['all'][atom_type]:
                                    """
                                    print(atom_type)
                                    print(atom_index)
                                    print(self.reduced_ff_optimizable_values['all'][force_group][array_no][line_no])
                                    """

                                    self.openmm_systems['all'].ff_optimizable[force_group][array_no][atom_index] = \
                                        tuple(self.reduced_ff_optimizable_values['all'][force_group][array_no][line_no])

                                for sys_type in list(self.reduced_ff_optimizable_values.keys())[1:]: # sth wrong here TODO fix

                                    if atom_type in list(self.dupes[sys_type].keys()):

                                        for atom_idx in self.dupes[sys_type][atom_type]:
                                            """
                                            print(atom_type)
                                            print(sys_type)
                                            print(atom_idx)
                                            print(self.reduced_ff_optimizable_values['all'][force_group][array_no][line_no])
                                            """

                                            self.openmm_systems[sys_type].ff_optimizable[force_group][array_no][atom_idx] = \
                                                tuple(self.reduced_ff_optimizable_values['all'][force_group][array_no][line_no])

            elif force_group in ['HarmonicBondForce', 'HarmonicAngleForce', 'PeriodicTorsionForce', 'NBException']: 
                #print(force_group)

                field_names = {'NBException': ['chargeProd', 'sigma', 'epsilon'],
                            'HarmonicBondForce': ['bond_length', 'force_constant'],
                            'HarmonicAngleForce': ['angle', 'force_constant'],
                            'PeriodicTorsionForce': ['periodicity', 'phase', 'force_constant'],
                            }

                for array_no, _array in enumerate(self.reduced_indexed_ff_optimizable['all'][force_group]):

                    for line_no, parameter_line in enumerate(_array): #

                        for atype_tuple in self.interaction_dupes['all'][force_group].keys():

                            if atype_tuple == parameter_line[0]:

                                for parameter_index in self.interaction_dupes['all'][force_group][atype_tuple]:
                                    """
                                    print(atype_tuple)
                                    print(force_group)
                                    print(array_no)
                                    print(field_names[force_group])
                                    print(parameter_index)
                                    print(self.openmm_systems['all'].ff_optimizable[force_group][array_no][field_names[force_group]][parameter_index])
                                    """
                                                                                                                    
                                    self.openmm_systems['all'].ff_optimizable[force_group][array_no][field_names[force_group]][parameter_index] = \
                                        tuple(self.reduced_ff_optimizable_values['all'][force_group][array_no][line_no])

                                for sys_type in list(self.reduced_ff_optimizable_values.keys())[1:]:

                                    if atype_tuple in list(self.interaction_dupes[sys_type][force_group].keys()):

                                        for parameter_idx in self.interaction_dupes[sys_type][force_group][atype_tuple]:
                                            """
                                            print(atype_tuple)
                                            print(sys_type)
                                            print(force_group)
                                            print(array_no)
                                            print(field_names[force_group])
                                            print(parameter_idx)
                                            print(self.reduced_ff_optimizable_values['all'][force_group][array_no][line_no])
                                            """

                                            self.openmm_systems[sys_type].ff_optimizable[force_group][array_no][field_names[force_group]][parameter_idx] = \
                                                tuple(self.reduced_ff_optimizable_values['all'][force_group][array_no][line_no]) 


    def vectorize_reduced_parameters(self): 
        """
        flattens nd np.arrays of parameters to 1d np.arrays of parameters

        internal parameters:
            self.reduced_ff_optimizable_values

        sets:
            self.vectorized_reduced_ff_optimizable_values
        """
        # define and init dictionaries
        self.vectorized_reduced_ff_optimizable_values = {k: [] for k in sorted(set(self.openmm_systems['all'].ff_optimizable.keys()))}

        for force_group in self.reduced_ff_optimizable_values['all'].keys():

            for array_no, _array in enumerate(self.reduced_ff_optimizable_values['all'][force_group]):

                self.vectorized_reduced_ff_optimizable_values[force_group].append([])
                self.vectorized_reduced_ff_optimizable_values[force_group][array_no] = np.array(_array).flatten()

    
    def reshape_vectorized_parameters(self): # watch out, only accesses self.reduced_ff_optimizable_values['all']
        """
        reshapes flattened 1d np.arrays to the original nd np.array dimensions.

        internal parameters:
            self.vectorized_reduced_ff_optimizable_values
            self.reduced_ff_optimizable_values

        sets:
            self.reduced_ff_optimizable_values
        """

        for force_group in self.vectorized_reduced_ff_optimizable_values.keys():

            for array_no, _array in enumerate(self.vectorized_reduced_ff_optimizable_values[force_group]):

                self.reduced_ff_optimizable_values['all'][force_group][array_no] = np.array(_array).reshape(np.array(self.reduced_ff_optimizable_values['all'][force_group][0]).shape)


    def merge_vectorized_parameters(self): 
        """
        'flattens' the dictionary of vectorized parameters into one long vector

        internal parameters:
            self.vectorized_reduced_ff_optimizable_values (dict of arrays)

        sets:
            self.vectorized_parameters
        """
        vectorized_parameters = []

        for force_group in self.vectorized_reduced_ff_optimizable_values.keys():

            for parameter_array in self.vectorized_reduced_ff_optimizable_values[force_group]:

                vectorized_parameters.append(parameter_array)

        self.vectorized_parameters = np.concatenate(vectorized_parameters)
        self.vectorized_parameters = self.vectorized_parameters.astype('float')


    def redistribute_vectorized_parameters(self):
        """
        puts the optimized parameters back into a dictionary of individual vectors

        internal parameters:
            self.vectorized_parameters (array)
            self.vectorized_reduced_ff_optimizable_values (dict of arrays)

        sets: 
            self.vectorized_reduced_ff_optimizable_values (dict of arrays)
        """

        slice_start = 0
        slice_end = 0

        for force_group in self.vectorized_reduced_ff_optimizable_values.keys(): 

            for _array_no, _array in enumerate(self.vectorized_reduced_ff_optimizable_values[force_group]):

                slice_end += len(_array)
                self.vectorized_reduced_ff_optimizable_values[force_group][_array_no] = self.vectorized_parameters[slice_start:slice_end]
                slice_start += slice_end


    def scale_parameters(self): 
        """
        Scales the magnitude of the parameters extracted from ff_optimizable using the z-score. 

        internal parameters:
            self.vectorized_parameters : np.array

        sets:
            self.scaling_factors, not mutable
            self.scaled_parameters, mutable
        """

        individual_mean = np.mean(self.vectorized_parameters)
        individual_sdev = np.std(self.vectorized_parameters)
        scaling_factors = (self.vectorized_parameters - individual_mean) / individual_sdev
        self.scaling_factors = scaling_factors.astype('float')
        scaled_parameters = self.vectorized_parameters * self.scaling_factors
        self.scaled_parameters = scaled_parameters.astype('float')


    def unscale_parameters(self):  
        """
        Transforms optimized parameters (scaled) back to their original magnitude

        internal parameters:
            self.scaled_parameters
            self.scaling_factors
        
        sets: 
            self.vectorized_parameters
        """

        self.vectorized_parameters = self.scaled_parameters / self.scaling_factors


def read_external_file(path: str, filename: str):

    """
    standard line-by-line reader for human readable files

    Parameters
    ----------
    path : str
        path to the dirctory where the file can be found
    filename : str
        name of the file including extension

    returns:
        the lines of the file in str format
    """

    if path[-1] != '/':
        path = path[:-1]

    f = open(path + filename, 'r')
    read = []
    for index, line in enumerate(f.readlines()):
        line = line.strip()
        read.append(line)
    f.close()

    return read

def find_same_type(ff_atom_types_indices):
    """
    id atoms of the same ff atom type and return their indices and types

    Parameters
    ----------
    ff_atom_types_indices : array of [int,str]
        ff atom types (column1) extracted from topology file with their respective atom indices (column0)

    returns:
        dictionary with non-unique atom types and their indices
        dict{'atom_type1':[1,12],
             'atom_type3':[3,4,14,15],
             'atom_type6':[5,16],
             ...}

    """
    dupes = {a:[] for a in sorted(set(ff_atom_types_indices[:,1]))}

    for atom_type in ff_atom_types_indices:
        for dupe in dupes.keys():
            if atom_type[1] == dupe:
                dupes[dupe].append(atom_type[0])

    return dupes